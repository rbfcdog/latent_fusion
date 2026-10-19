"""Frozen MiniLM event tokens and a price-only model with a bounded news residual.

Article text never enters the price model; all event lookup is restricted to decisions
on or after the article's lagged availability date.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.fusion.models import NumericEncoder

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
FEED_END = np.datetime64("2023-11-30", "D")
EVENT_DAYS = 5
EVENT_COUNT = 8


def load_event_panel(path):
    with np.load(path, allow_pickle=False) as z:
        if int(z["version"]) != 1 or int(z["horizon"]) != 5 or int(z["seq_len"]) != 10:
            raise ValueError("Painel incompatível: esperado version=1, horizon=5, seq_len=10")
        tickers = z["tickers"].tolist()
        keys = ("dates", "target_end", "mom", "past_vol", "known_ret", "target",
                "fwd_return", "market_fwd_return")
        arrays = [{key: z[f"arr_{i}_{key}"] for key in keys} for i in range(len(tickers))]
        splits = {s: z[f"samples_{s}"].astype(np.int32) for s in ("train", "val", "test")}
        bounds = z["bounds"].astype("datetime64[D]")
        note = str(z["source_note"]) if "source_note" in z else "Feed indisponível após 2023-11-30."
    if not tickers or any(not len(v) or v.shape[1:] != (2,) for v in splits.values()):
        raise ValueError("Painel sem amostras em algum período")
    for a in arrays:
        dates = a["dates"].astype("datetime64[D]")
        if np.any(dates > FEED_END) or np.any(dates[1:] <= dates[:-1]):
            raise ValueError("Datas desordenadas ou posteriores ao período auditado do feed")
        if any(len(a[k]) != len(dates) for k in keys):
            raise ValueError("Comprimentos incoerentes no painel")
        a["dates"] = dates
        a["target_end"] = a["target_end"].astype("datetime64[D]")
    if any(np.any(s[:, 0] < 0) or np.any(s[:, 0] >= len(tickers)) or
           any(np.any(s[s[:, 0] == i, 1] < 9) or
               np.any(s[s[:, 0] == i, 1] >= len(a["dates"])) for i, a in enumerate(arrays))
           for s in splits.values()):
        raise ValueError("Índices de amostra inválidos")
    return {"tickers": tickers, "arrays": arrays, "splits": splits, "bounds": bounds,
            "seq_len": 10, "horizon": 5, "source_note": note}


def load_article_embeddings(news_path, cache_dir, batch_size=32):
    """Hash raw manifest bytes and encoder identity; write only embeddings/metadata to Drive."""
    path, cache_dir = Path(news_path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest_hash = digest.hexdigest()
    cache_key = hashlib.sha256(f"{manifest_hash}|{ENCODER}|v1|normalized".encode()).hexdigest()
    cache_path = cache_dir / f"articles_{cache_key[:20]}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as z:
            if str(z["cache_key"]) != cache_key:
                raise ValueError("Cache de artigos incompatível")
            return {k: z[k] for k in ("ticker", "decision_date", "published_at", "embedding")}, cache_path

    records = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if not all(isinstance(row.get(k), str) for k in
                       ("ticker", "decision_date", "published_at", "url", "source", "text")):
                raise ValueError("Notícia fora do esquema do manifesto")
            if np.datetime64(row["decision_date"], "D") <= FEED_END:
                records.append(row)
    if not records:
        raise ValueError("Nenhum evento no período auditado")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER, device="cuda")
    model.eval()
    embeddings = np.empty((len(records), 384), dtype=np.float32)
    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        embeddings[start:end] = model.encode(
            [r["text"] for r in records[start:end]], batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )
    result = {"ticker": np.asarray([r["ticker"] for r in records]),
              "decision_date": np.asarray([r["decision_date"] for r in records], dtype="datetime64[D]"),
              "published_at": np.asarray([r["published_at"] for r in records]),
              "embedding": embeddings}
    np.savez_compressed(cache_path, cache_key=np.asarray(cache_key), **result)
    return result, cache_path


class EventDataset(Dataset):
    def __init__(self, panel, events, split, event_count=EVENT_COUNT, event_days=EVENT_DAYS,
                 shuffled=False, seed=42):
        self.panel, self.samples = panel, panel["splits"][split]
        self.event_count, self.event_days = event_count, event_days
        self.shuffled = shuffled
        self.date_indices = []
        self.tokens = []
        self.available = []
        for ticker, a in zip(panel["tickers"], panel["arrays"]):
            idx = np.flatnonzero(events["ticker"] == ticker)
            # Within ticker sort by availability first, publication second. No future
            # article can contribute to a decision, including in the ablation.
            idx = idx[np.lexsort((events["published_at"][idx], events["decision_date"][idx]))]
            dates = events["decision_date"][idx].astype("datetime64[D]")
            if np.any(dates > FEED_END):
                raise ValueError("Notícia fora da cobertura do feed")
            emb = events["embedding"][idx].astype(np.float32, copy=False)
            previous = np.vstack((np.zeros((1, emb.shape[1]), dtype=np.float32), emb[:-1]))
            delta = emb - previous
            similarity = (emb * previous).sum(axis=1)
            novelty = np.clip(1 - similarity, 0, 2)
            if len(novelty):
                novelty[0] = 0
                delta[0] = 0
            tokens = np.concatenate((emb, delta, novelty[:, None]), axis=1)
            self.date_indices.append(np.searchsorted(a["dates"], dates, side="left"))
            self.available.append(dates)
            self.tokens.append(tokens)
        self.dim = events["embedding"].shape[1] * 2 + 2
        if shuffled:
            # Ablation: rotate source tickers on each decision date and lag them five
            # sessions. This breaks ticker/time association without injecting future news.
            rng = np.random.default_rng(seed)
            self.donors = rng.permutation(len(panel["tickers"]))
            if len(self.donors) > 1 and np.any(self.donors == np.arange(len(self.donors))):
                self.donors = np.roll(np.arange(len(self.donors)), 1)

    def _news(self, ti, j):
        a = self.panel["arrays"][ti]
        if self.shuffled:
            ti = int(self.donors[ti])
            a_donor = self.panel["arrays"][ti]
            # Donor cutoff is an actual prior trading date, not a source future date.
            prior = max(0, j - self.event_days)
            cutoff = a["dates"][prior]
            donor_end = int(np.searchsorted(a_donor["dates"], cutoff, side="right")) - 1
            if donor_end < 0:
                return np.zeros((self.event_count, self.dim), np.float32), np.zeros(self.event_count, bool)
            decision = a_donor["dates"][donor_end]
            start_day = max(0, donor_end - self.event_days + 1)
            earliest = a_donor["dates"][start_day]
        else:
            decision = a["dates"][j]
            start_day = max(0, j - self.event_days + 1)
            earliest = a["dates"][start_day]
        dates = self.available[ti]
        end = np.searchsorted(dates, decision, side="right")
        start = np.searchsorted(dates, earliest, side="left")
        indices = np.arange(max(start, end - self.event_count), end)
        out = np.zeros((self.event_count, self.dim), dtype=np.float32)
        mask = np.zeros(self.event_count, dtype=bool)
        if len(indices):
            n = len(indices)
            out[-n:, :-1] = self.tokens[ti][indices]
            # Trade-session age; zero for articles available today.
            available_idx = self.date_indices[ti][indices]
            age = (donor_end if self.shuffled else j) - available_idx
            out[-n:, -1] = np.clip(age / self.event_days, 0, 1)
            mask[-n:] = True
        return out, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ti, j = map(int, self.samples[index])
        a = self.panel["arrays"][ti]
        sl = slice(j - self.panel["seq_len"] + 1, j + 1)
        text, mask = self._news(ti, j)
        return (torch.from_numpy(a["mom"][sl]), torch.tensor(a["past_vol"][j], dtype=torch.float32),
                torch.tensor(a["known_ret"][j], dtype=torch.float32), torch.from_numpy(text),
                torch.from_numpy(mask), torch.tensor(a["target"][j], dtype=torch.float32))


class PriceOnly(nn.Module):
    def __init__(self, moment_dim, hidden=64):
        super().__init__()
        self.numeric = NumericEncoder(moment_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, mom, past_vol, known_ret):
        numeric = self.numeric(mom)
        score = self.head(torch.cat((numeric, torch.log(past_vol.clamp_min(1e-6))[:, None],
                                     known_ret[:, None]), dim=-1))
        return score, numeric


class EventResidual(nn.Module):
    def __init__(self, baseline, event_dim=770, hidden=64, max_residual=0.05):
        super().__init__()
        self.baseline = baseline.eval()
        for param in self.baseline.parameters():
            param.requires_grad_(False)
        self.text_proj = nn.Sequential(nn.Linear(event_dim, hidden), nn.LayerNorm(hidden))
        self.cross_attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.gate = nn.Linear(hidden * 2 + 1, 1)
        self.residual = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.max_residual = max_residual

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, mom, past_vol, known_ret, events, mask):
        with torch.no_grad():
            base, query = self.baseline(mom, past_vol, known_ret)
        tokens = self.text_proj(events)
        # Explicit valid null key avoids NaN attention for entirely empty feeds.
        null = torch.zeros_like(tokens[:, :1])
        keys = torch.cat((tokens, null), dim=1)
        padding = torch.cat((~mask, torch.zeros_like(mask[:, :1])), dim=1)
        context, _ = self.cross_attn(query[:, None], keys, keys,
                                     key_padding_mask=padding, need_weights=False)
        context = context[:, 0]
        joint = torch.cat((query, context), dim=1)
        gate = torch.sigmoid(self.gate(torch.cat((joint, mask.float().mean(1, keepdim=True)), dim=1)))
        correction = self.max_residual * gate * torch.tanh(self.residual(joint))
        return torch.where(mask.any(dim=1, keepdim=True), base + correction, base)


def _check_event_index():
    """Pequena checagem de causalidade sem carregar modelos nem fazer forward."""
    dates = np.arange(np.datetime64("2020-01-01", "D"), np.datetime64("2020-01-20", "D"))
    panel = {"tickers": ["AAA"], "arrays": [{"dates": dates}], "splits": {
        "test": np.asarray([[0, 9], [0, 10], [0, 15]], dtype=np.int32)}, "seq_len": 10}
    events = {"ticker": np.asarray(["AAA"]), "decision_date": dates[[10]],
              "published_at": np.asarray(["2020-01-10"]), "embedding": np.ones((1, 384), np.float32)}
    dataset = EventDataset(panel, events, "test")
    assert not dataset._news(0, 9)[1].any()
    assert dataset._news(0, 10)[1].sum() == 1
    assert not dataset._news(0, 15)[1].any()


if __name__ == "__main__":
    _check_event_index()
