"""Headline-only event tokens and numeric-query attention; no price-only output path."""

from __future__ import annotations

from collections import deque
import gzip
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.fusion.event_model import EVENT_COUNT, EVENT_DAYS, FEED_END, EventDataset
from src.fusion.headline_features import FEATURE_NAMES, extract_headline_features
from src.fusion.models import NumericEncoder


def load_headline_events(news_path, cache):
    """Stream the audited manifest in the exact order of its cached MiniLM vectors."""
    embeddings = cache["embedding"]
    if (embeddings.ndim != 2 or not all(len(cache[field]) == len(embeddings)
                                         for field in ("ticker", "decision_date", "published_at")) or
            not np.isfinite(embeddings).all()):
        raise ValueError("Cache de embeddings inválido")
    features = np.empty((len(embeddings), len(FEATURE_NAMES)), dtype=np.float32)
    novelty = np.empty(len(embeddings), dtype=np.float32)
    history = {}
    pending = {}
    current_date = {}
    count = 0
    with gzip.open(news_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in
                                                     ("ticker", "decision_date", "published_at", "url", "source", "text")):
                raise ValueError("Notícia fora do esquema do manifesto")
            date = np.datetime64(row["decision_date"], "D")
            if date > FEED_END:
                continue  # Same audited cutoff applied by load_article_embeddings.
            if count >= len(embeddings) or (str(cache["ticker"][count]) != row["ticker"] or
                                           np.datetime64(cache["decision_date"][count], "D") != date or
                                           str(cache["published_at"][count]) != row["published_at"]):
                raise ValueError(f"Cache de artigos desalinhado com o manifesto na linha {count + 1}")
            ticker = row["ticker"]
            previous_date = current_date.get(ticker)
            if previous_date is not None and date < previous_date:
                raise ValueError("Manchetes fora da ordem de disponibilidade por ticker")
            if previous_date is None or date > previous_date:
                earlier = history.setdefault(ticker, deque(maxlen=10))
                earlier.extend(pending.pop(ticker, ()))
                current_date[ticker] = date
            else:
                earlier = history[ticker]
            embedding = embeddings[count]
            # Same-day headlines have ambiguous ordering: none may seed another's novelty.
            novelty[count] = (np.clip(1 - max(float(embedding @ prior) for prior in earlier), 0, 2)
                              if earlier else 0)
            pending.setdefault(ticker, []).append(embedding)
            extracted = np.asarray(extract_headline_features(row["text"]), dtype=np.float32)
            if extracted.shape != (len(FEATURE_NAMES),) or not np.isfinite(extracted).all():
                raise ValueError("Características de manchete inválidas")
            features[count] = extracted
            count += 1
    if count != len(embeddings):
        raise ValueError(f"Cache de artigos desalinhado: {count} manchetes vs {len(embeddings)} vetores")
    return {**cache, "features": features, "novelty": novelty}


class TextEventDataset(EventDataset):
    """Eight most recent titles in five actual ticker trading sessions, including today."""

    def __init__(self, panel, events, split, event_count=EVENT_COUNT, event_days=EVENT_DAYS):
        self.panel, self.samples = panel, panel["splits"][split]
        self.event_count, self.event_days = event_count, event_days
        self.shuffled = False
        self.available, self.date_indices, self.tokens = [], [], []
        self.dim = events["embedding"].shape[1] + len(FEATURE_NAMES) + 2
        for ticker, a in zip(panel["tickers"], panel["arrays"]):
            idx = np.flatnonzero(events["ticker"] == ticker)
            idx = idx[np.lexsort((events["published_at"][idx], events["decision_date"][idx]))]
            dates = events["decision_date"][idx].astype("datetime64[D]")
            if np.any(dates > FEED_END):
                raise ValueError("Notícia fora da cobertura do feed")
            available_idx = np.searchsorted(a["dates"].astype("datetime64[D]"), dates)
            if np.any(available_idx >= len(a["dates"])):
                raise ValueError("Disponibilidade ausente do calendário de pregões do ticker")
            if np.any(a["dates"][available_idx].astype("datetime64[D]") != dates):
                raise ValueError("Disponibilidade ausente do calendário de pregões do ticker")
            self.date_indices.append(available_idx)
            self.available.append(dates)
            self.tokens.append(np.concatenate((events["embedding"][idx].astype(np.float32, copy=False),
                                               events["features"][idx], events["novelty"][idx, None]), axis=1))
        if sum(len(days) for days in self.available) != len(events["ticker"]):
            raise ValueError("Ticker de notícia ausente do painel")


class MetadataDataset(Dataset):
    """Age and count/mask only; same samples and article selection as text dataset."""

    def __init__(self, text_dataset):
        self.text_dataset = text_dataset
        self.samples = text_dataset.samples

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, index):
        mom, vol, ret, text, mask, target = self.text_dataset[index]
        return mom, vol, ret, text[:, -1:], mask, target


class TextAttention(nn.Module):
    """Numeric history selects news; only attended news reaches the prediction head."""

    def __init__(self, moment_dim, event_dim, hidden=64):
        super().__init__()
        self.numeric = NumericEncoder(moment_dim, hidden)
        self.query = nn.Linear(hidden + 2, hidden)
        self.text_proj = nn.Sequential(nn.Linear(event_dim, hidden), nn.LayerNorm(hidden))
        self.event_attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, mom, past_vol, known_ret, events, mask):
        text = self.text_proj(events)
        null = torch.zeros_like(text[:, :1])
        padding = torch.cat((~mask, torch.zeros_like(mask[:, :1])), dim=1)
        context, _ = self.event_attn(text, torch.cat((text, null), dim=1),
                                     torch.cat((text, null), dim=1),
                                     key_padding_mask=padding, need_weights=False)
        news = torch.cat((context, null), dim=1)
        numeric = self.numeric(mom)
        query = self.query(torch.cat((numeric, torch.log(past_vol.clamp_min(1e-6))[:, None],
                                      known_ret[:, None]), dim=1))
        attended, _ = self.cross_attn(query[:, None], news, news,
                                      key_padding_mask=padding, need_weights=False)
        score = self.head(attended[:, 0])
        return torch.where(mask.any(dim=1, keepdim=True), score, torch.zeros_like(score))
