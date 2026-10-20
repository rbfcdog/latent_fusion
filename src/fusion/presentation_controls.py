"""Colab-only controls on the already inspected 2022–23 fast-fusion test period.

The headline falsification replaces content, not article availability or coverage.
Neither this shuffle nor its comparison constitutes a fresh holdout.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.fusion.event_experiment import _prediction_frame, _scores, _train
from src.fusion.event_model import EventResidual
from src.fusion.fast_demo_experiment import _predict, _summarize
from src.fusion.fast_residual_experiment import _SavedNumericBaseline, load_saved_demo
from src.fusion.models import TSOnlyPredictor
from src.fusion.text_attention import MetadataDataset, TextEventDataset


CONTENT_KEYS = ("embedding", "features", "novelty")
SEED = 42


def shuffle_headline_content(events, *, seed=SEED):
    """Shuffle only content from headlines available no later than each destination.

    A same-day group of at least two is rotated by one or more positions;
    singleton dates draw from strictly earlier dates. The first single article
    cannot be replaced causally and stays unchanged (reported in the audit).
    """
    dates = np.asarray(events["decision_date"]).astype("datetime64[D]")
    if np.isnat(dates).any() or any(len(events[key]) != len(dates) for key in
                                    ("ticker", "published_at", *CONTENT_KEYS)):
        raise ValueError("Manchetes com disponibilidade ou comprimentos inválidos")
    order = np.argsort(dates, kind="stable")
    donors = np.arange(len(dates))
    rng = np.random.default_rng(seed)
    _, starts, counts = np.unique(dates[order], return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        current = order[start:start + count]
        if count > 1:
            # A nonzero cyclic shift changes every article and stays on the same day.
            donors[current] = np.roll(current, int(rng.integers(1, count)))
        elif start:
            donors[current[0]] = rng.choice(order[:start])
    if np.any(dates[donors] > dates):
        raise ValueError("Manchete doadora ainda indisponível na data de destino")
    changed = donors != np.arange(len(dates))
    if len(dates) > 1 and not changed.any():
        raise ValueError("Controle sem manchetes substituíveis")
    shuffled = {**events, **{key: events[key][donors] for key in CONTENT_KEYS}}
    mapping = pd.DataFrame({
        "destination_index": np.arange(len(dates)), "donor_index": donors,
        "destination_ticker": events["ticker"], "destination_decision_date": dates,
        "destination_published_at": events["published_at"],
        "donor_ticker": events["ticker"][donors],
        "donor_decision_date": dates[donors],
        "donor_published_at": events["published_at"][donors], "changed": changed,
    })
    return shuffled, mapping


class _ContentOnlyDataset(Dataset):
    """Keep only headline embeddings, lexical/sentiment features and novelty."""

    def __init__(self, text_dataset):
        self.text_dataset = text_dataset
        self.samples = text_dataset.samples
        self.dim = text_dataset.dim - 1  # Excludes the article-age metadata channel.

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, index):
        ticker_index, session = map(int, self.samples[index])
        text, mask = self.text_dataset._news(ticker_index, session)
        target = self.text_dataset.panel["arrays"][ticker_index]["target"][session]
        return (torch.from_numpy(text[:, :-1]), torch.from_numpy(mask),
                torch.tensor(target, dtype=torch.float32))


class _ContentOnlyPredictor(nn.Module):
    """No numeric encoder, prices, returns, article ages, or ticker identifiers."""

    def __init__(self, event_dim, hidden=64):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(event_dim, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, text, mask):
        weights = mask.unsqueeze(-1)
        pooled = (self.token(text) * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return torch.where(mask.any(dim=1, keepdim=True), self.head(pooled),
                           torch.zeros((len(text), 1), device=text.device, dtype=text.dtype))


def _fit_content(model, train, val, *, epochs=8, patience=2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=128, shuffle=True, pin_memory=True,
                        generator=torch.Generator().manual_seed(SEED))
    validation = DataLoader(val, batch_size=256, pin_memory=True)
    best, weights, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        total = count = 0
        for text, mask, target in loader:
            text, mask, target = (item.to("cuda", non_blocking=True) for item in (text, mask, target))
            active = mask.any(dim=1)
            if not active.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(text, mask)[active, 0], target[active])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * int(active.sum())
            count += int(active.sum())
        model.eval()
        vtotal = vcount = 0
        with torch.inference_mode():
            for text, mask, target in validation:
                text, mask, target = (item.to("cuda", non_blocking=True) for item in (text, mask, target))
                active = mask.any(dim=1)
                if active.any():
                    vtotal += nn.functional.mse_loss(model(text, mask)[active, 0],
                                                     target[active], reduction="sum").item()
                    vcount += int(active.sum())
        if not count or not vcount:
            raise ValueError("Treino/validação sem manchetes disponíveis")
        value = vtotal / vcount
        history.append({"epoch": epoch + 1, "train_mse": total / count, "val_mse": value})
        if value < best - 1e-8:
            best, weights, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(weights)
    return pd.DataFrame(history)


def _score_content(model, dataset):
    model.eval()
    scores, masks = [], []
    with torch.inference_mode():
        for text, mask, _ in DataLoader(dataset, batch_size=256, pin_memory=True):
            text, mask = (item.to("cuda", non_blocking=True) for item in (text, mask))
            scores.append(model(text, mask)[:, 0].cpu().numpy())
            masks.append(mask.any(dim=1).cpu().numpy())
    return np.concatenate(scores), np.concatenate(masks)


def _reuse_previous(previous_results, panel, originals, checkpoint, provenance, baseline, base_scores, mask):
    """Verify old score files against saved weights and exact current test rows."""
    previous = Path(previous_results)
    config = json.loads((previous / "run_config.json").read_text(encoding="utf-8"))
    if (config.get("embedding_provenance") != provenance or
            config.get("baseline_seed") != checkpoint["seed"] or
            config.get("residual_seed") != SEED or config.get("max_residual") != 0.02 or
            config.get("max_epochs") != 8 or config.get("patience") != 2):
        raise ValueError("Proveniência ou configuração anterior incompatível")
    expected = _prediction_frame(panel, originals["test"].samples, base_scores, mask)

    def checked_scores(name, scores):
        old = pd.read_csv(previous / f"predictions_{name}.csv", parse_dates=["date"])
        if (len(old) != len(expected) or not old[["ticker", "date", "has_event"]].equals(
                expected[["ticker", "date", "has_event"]]) or
                any(not np.allclose(old[field], expected[field], rtol=0, atol=1e-7)
                    for field in ("excess_return_5d", "asset_return_5d", "market_return_5d")) or
                not np.isfinite(old["score"].to_numpy()).all() or
                not np.allclose(old["score"], scores, rtol=1e-5, atol=1e-6)):
            raise ValueError(f"Previsões anteriores incompatíveis: {name}")
        return old["score"].to_numpy()

    reused = {"ts_only": checked_scores("ts_only", base_scores)}
    histories = {}
    for name, dataset, dimension in (
        ("metadata", MetadataDataset(originals["test"]), 1),
        ("text_residual", originals["test"], originals["test"].dim),
    ):
        saved = torch.load(previous / f"{name}.pt", map_location="cpu", weights_only=True)
        if (saved.get("model") != name or saved.get("source_provenance") != provenance or
                saved.get("baseline_seed") != checkpoint["seed"]):
            raise ValueError(f"Checkpoint anterior incompatível: {name}")
        if any((f"baseline.predictor.{key}" not in saved["state_dict"] or
                not torch.equal(saved["state_dict"][f"baseline.predictor.{key}"], value.cpu()))
               for key, value in baseline.state_dict().items()):
            raise ValueError(f"Checkpoint anterior altera o baseline numérico: {name}")
        model = EventResidual(_SavedNumericBaseline(baseline), event_dim=dimension,
                              max_residual=0.02).to("cuda")
        model.load_state_dict(saved["state_dict"], strict=True)
        inferred, inferred_mask = _scores(model, dataset, torch.device("cuda"), residual=True)
        if not np.array_equal(inferred_mask, mask):
            raise ValueError(f"Cobertura anterior incompatível: {name}")
        reused[name] = checked_scores(name, inferred)
        histories[name] = pd.read_csv(previous / f"{name}_history.csv")
    return reused, histories


def run_controls(saved_dir, out_dir, cost_bps=10, *, previous_results=None):
    """Train/evaluate controls on Colab CUDA; optionally reuse verified old fits.

    Returns prediction frames, signal/portfolio tables, equity, histories and
    provenance; writes each result to isolated CSV/checkpoint files in out_dir.
    """
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Execute exclusivamente no Google Colab com GPU") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Ative uma GPU no Colab; nenhum forward local é permitido")
    if not 0 <= cost_bps < 10_000:
        raise ValueError("Custo de transação em bps inválido")

    panel, events, checkpoint, provenance = load_saved_demo(saved_dir)
    shuffled_events, mapping = shuffle_headline_content(events)
    originals = {split: TextEventDataset(panel, events, split) for split in ("train", "val", "test")}
    shuffled = {split: TextEventDataset(panel, shuffled_events, split)
                for split in ("train", "val", "test")}
    # Check every token slot: same article selection, age, mask, and coverage.
    for split in originals:
        left, right = originals[split], shuffled[split]
        if not np.array_equal(left.samples, right.samples):
            raise ValueError("Amostras do controle desalinhadas")
        for original, control in zip(left.tokens, right.tokens):
            if len(original) != len(control):
                raise ValueError("Cobertura de manchetes alterada")
        for original, control in zip(left.available, right.available):
            if not np.array_equal(original, control):
                raise ValueError("Disponibilidade de manchetes alterada")
        for original, control in zip(left.date_indices, right.date_indices):
            if not np.array_equal(original, control):
                raise ValueError("Idade das manchetes alterada")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    baseline = TSOnlyPredictor(d_ts=checkpoint["d_ts"]).to("cuda")
    baseline.load_state_dict(checkpoint["state_dict"], strict=True)
    baseline.eval()
    base_scores, base_mask = _predict(baseline, originals["test"])
    if previous_results is None:
        scores, histories = {"ts_only": base_scores}, {}
        comparisons = (
            ("metadata", {split: MetadataDataset(ds) for split, ds in originals.items()}, 1),
            ("text_residual", originals, originals["train"].dim),
        )
    else:
        scores, histories = _reuse_previous(previous_results, panel, originals, checkpoint,
                                            provenance, baseline, base_scores, base_mask)
        comparisons = ()
    for name, datasets, dimension in (*comparisons,
                                      ("shuffled_content", shuffled, shuffled["train"].dim)):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = EventResidual(_SavedNumericBaseline(baseline), event_dim=dimension,
                              max_residual=0.02).to("cuda")
        nn.init.zeros_(model.residual[-1].weight)
        nn.init.zeros_(model.residual[-1].bias)
        histories[name] = _train(model, datasets["train"], datasets["val"],
                                 torch.device("cuda"), residual=True, epochs=8, patience=2)
        scores[name], mask = _scores(model, datasets["test"], torch.device("cuda"), residual=True)
        if not np.array_equal(mask, base_mask):
            raise ValueError("Máscaras do controle desalinhadas")
        torch.save({"model": name, "state_dict": model.state_dict(),
                    "source_provenance": provenance, "seed": SEED}, out / f"{name}.pt")
    content = {split: _ContentOnlyDataset(ds) for split, ds in originals.items()}
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    text_model = _ContentOnlyPredictor(content["train"].dim).to("cuda")
    histories["text_only"] = _fit_content(text_model, content["train"], content["val"])
    scores["text_only"], mask = _score_content(text_model, content["test"])
    if not np.array_equal(mask, base_mask):
        raise ValueError("Máscaras do controle textual desalinhadas")
    torch.save({"model": "text_only", "state_dict": text_model.state_dict(),
                "source_provenance": provenance, "seed": SEED}, out / "text_only.pt")

    scores["zero"] = np.zeros_like(base_scores)
    predictions, signal, portfolio, equity = _summarize(
        panel, originals["test"].samples, base_mask, scores, cost_bps)
    score_deltas = predictions["text_residual"][["ticker", "date", "has_event"]].copy()
    score_deltas["text_minus_shuffled"] = (
        predictions["text_residual"]["score"] - predictions["shuffled_content"]["score"])
    score_deltas["text_minus_metadata"] = (
        predictions["text_residual"]["score"] - predictions["metadata"]["score"])
    score_deltas["text_only_minus_zero"] = predictions["text_only"]["score"]
    score_deltas.to_csv(out / "paired_score_deltas.csv", index=False)
    for name, frame in predictions.items():
        frame.to_csv(out / f"predictions_{name}.csv", index=False)
    for name, history in histories.items():
        history.to_csv(out / f"{name}_history.csv", index=False)
    mapping.to_csv(out / "shuffled_content_mapping.csv", index=False)
    signal.to_csv(out / "signal_metrics.csv")
    portfolio.to_csv(out / "portfolio_metrics.csv")
    equity.to_csv(out / "equity_curves.csv")
    run_info = {"embedding_provenance": provenance, "baseline_seed": checkpoint["seed"],
                "control_seed": SEED, "max_epochs": 8, "patience": 2, "cost_bps": cost_bps,
                "previous_results": str(previous_results) if previous_results is not None else None,
                "models": [*scores, "buy_hold"], "shuffled_articles": int(mapping.changed.sum()),
                "unavoidable_unchanged_articles": int((~mapping.changed).sum()),
                "shuffle_note": "Only content is replaced; same-day derangement or earlier available donor; "
                                "singleton on earliest date remains unchanged.",
                "text_only_features": "MiniLM headline vector, lexical/sentiment features, novelty; no price or age",
                "period_note": "2022–23 was previously inspected; controls are not a fresh holdout.",
                "source_note": panel["source_note"]}
    (out / "run_config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return {"predictions": predictions, "signal": signal, "portfolio": portfolio,
            "equity": equity, "histories": histories, "mapping": mapping,
            "output": out, "provenance": run_info}
