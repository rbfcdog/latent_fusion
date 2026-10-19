"""Fresh, GPU-only embeddings for the audited eight-asset headline demo.

``prepare_demo_structure`` is CPU-only: its ``mom`` arrays have zero columns until
``prepare_demo_inputs`` replaces them with newly inferred MOMENT vectors. Nothing
from the source panel's cached MOMENT or article vectors is loaded.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.event_model import FEED_END
from src.fusion.text_attention import load_headline_events

DEMO_TICKERS = ("ADBE", "BABA", "COST", "CVX", "GILD", "NKE", "PYPL", "QCOM")
MOMENT_ID = "AutonLab/MOMENT-1-small"
MOMENT_REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"
TEXT_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEQ_LEN = 10


def prepare_demo_structure(panel_path, news_path, window=64):
    """Select the audited calendar and labels without loading any pretrained vectors.

    Rows in the returned list are manifest-order title records, not article bodies.
    The empty float32 ``mom`` matrices are placeholders, NEVER model features.
    """
    if not 1 <= window <= 512:
        raise ValueError("MOMENT window must be between 1 and 512 sessions")
    with np.load(panel_path, allow_pickle=False) as source:
        if int(source["version"]) != 1 or int(source["seq_len"]) != SEQ_LEN or int(source["horizon"]) != 5:
            raise ValueError("Expected audited 10-session, 5-session-forward panel version 1")
        names = source["tickers"].tolist()
        if any(names.count(ticker) != 1 for ticker in DEMO_TICKERS):
            raise ValueError("Missing or duplicate demo asset in source panel")
        bounds = source["bounds"].astype("datetime64[ns]")
        if bounds.shape != (2,) or not bounds[0] < bounds[1] < FEED_END:
            raise ValueError("Invalid chronological source split bounds")
        arrays = []
        original_indices = []
        for ticker in DEMO_TICKERS:
            i = names.index(ticker)
            original_indices.append(i)
            # Deliberately never access source[f"arr_{i}_mom"] or old target/market.
            fields = ("dates", "target_end", "past_vol", "known_ret", "fwd_return")
            a = {key: source[f"arr_{i}_{key}"] for key in fields}
            a["dates"] = a["dates"].astype("datetime64[ns]")
            n = int(np.searchsorted(a["dates"], FEED_END.astype("datetime64[ns]"), side="right"))
            a = {key: value[:n] for key, value in a.items()}
            if not n or np.any(a["dates"][1:] <= a["dates"][:-1]) or any(len(v) != n for v in a.values()):
                raise ValueError(f"Invalid price calendar for {ticker}")
            a["target_end"] = a["target_end"].astype("datetime64[ns]")
            for key in ("past_vol", "known_ret", "fwd_return"):
                a[key] = a[key].astype(np.float32)
            a["mom"] = np.empty((n, 0), dtype=np.float32)
            arrays.append(a)
        raw_splits = {split: source[f"samples_{split}"].astype(np.int32)
                      for split in ("train", "val", "test")}
        note = str(source["source_note"]) if "source_note" in source else "Audited titles end 2023-11-30."

    # Align the SAME eight price series for both labels and buy-and-hold; a
    # missing date is never silently filled with a zero return.
    aligned = pd.concat([pd.Series(a["fwd_return"], index=pd.DatetimeIndex(a["dates"]), name=ticker)
                         for ticker, a in zip(DEMO_TICKERS, arrays)], axis=1)
    market = aligned.mean(axis=1, skipna=False)
    for a in arrays:
        a["market_fwd_return"] = market.reindex(a["dates"]).to_numpy(dtype=np.float32)
        a["target"] = a["fwd_return"] - a["market_fwd_return"]

    splits = {}
    minimum = window + SEQ_LEN - 1
    for split, rows in raw_splits.items():
        if rows.ndim != 2 or rows.shape[1] != 2:
            raise ValueError(f"Invalid {split} source split")
        selected = []
        for new_i, old_i in enumerate(original_indices):
            a = arrays[new_i]
            for old_ti, j in rows:
                if old_ti != old_i or j < minimum or j >= len(a["dates"]):
                    continue
                date, end = a["dates"][j], a["target_end"][j]
                if np.isnat(end) or not np.isfinite(a["target"][j]):
                    continue
                if ((split == "train" and date <= bounds[0] and end <= bounds[0]) or
                    (split == "val" and bounds[0] < date <= bounds[1] and end <= bounds[1]) or
                    (split == "test" and bounds[1] < date and end <= FEED_END)):
                    selected.append((new_i, int(j)))
        if not selected:
            raise ValueError(f"No usable {split} samples after window and label purge")
        splits[split] = np.asarray(selected, dtype=np.int32)

    first_test = min(arrays[ti]["dates"][j] for ti, j in splits["test"])
    calendar = pd.concat([pd.Series(a["known_ret"], index=pd.DatetimeIndex(a["dates"]), name=ticker)
                          for ticker, a in zip(DEMO_TICKERS, arrays)], axis=1).loc[pd.Timestamp(first_test):]
    if calendar.empty or calendar.isna().any().any() or not np.isfinite(calendar.to_numpy()).all():
        raise ValueError("Incomplete eight-asset daily price calendar after first test date")

    # Verify every row that inference will need (including the prior 9 sequence
    # positions), not merely the return observed on a decision date.
    for ti, a in enumerate(arrays):
        required = _required_rows(splits, ti, len(a["dates"]))
        for j in np.flatnonzero(required):
            if j < window - 1 or not np.isfinite(a["known_ret"][j - window + 1:j + 1]).all():
                raise ValueError(f"Incomplete trailing returns for {DEMO_TICKERS[ti]} on {a['dates'][j]}")
        for split, samples in splits.items():
            at = samples[samples[:, 0] == ti, 1]
            if len(at) and not np.isfinite(a["past_vol"][at]).all():
                raise ValueError(f"Invalid past volatility in {split} for {DEMO_TICKERS[ti]}")

    rows = []
    with gzip.open(news_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if not isinstance(record, dict) or not all(isinstance(record.get(key), str) for key in
                                                       ("ticker", "decision_date", "published_at", "url", "source", "text")):
                raise ValueError("Invalid audited headline manifest record")
            if record["ticker"] not in DEMO_TICKERS or np.datetime64(record["decision_date"], "D") > FEED_END:
                continue
            ticker = DEMO_TICKERS.index(record["ticker"])
            date = np.datetime64(record["decision_date"], "ns")
            if date not in arrays[ticker]["dates"]:
                raise ValueError(f"Headline decision date absent from {record['ticker']} price calendar: {date}")
            rows.append({key: record[key] for key in
                         ("ticker", "decision_date", "published_at", "url", "source", "text")})
    if not rows:
        raise ValueError("No audited headlines for selected assets")
    return {"tickers": list(DEMO_TICKERS), "arrays": arrays, "splits": splits,
            "bounds": bounds, "seq_len": SEQ_LEN, "horizon": 5, "source_note": note}, rows


def _required_rows(splits, ti, n):
    required = np.zeros(n, dtype=bool)
    for samples in splits.values():
        for _, j in samples[samples[:, 0] == ti]:
            required[j - SEQ_LEN + 1:j + 1] = True
    return required


def _moment_batch(returns, indices, window):
    """Return masked 512-step sequences using exclusively returns at or before each index."""
    x = np.zeros((len(indices), 1, 512), dtype=np.float32)
    for k, j in enumerate(indices):
        x[k, 0, :window] = returns[j - window + 1:j + 1]
    mask = np.zeros((len(indices), 512), dtype=bool)
    mask[:, :window] = True
    return x, mask


def prepare_demo_inputs(panel_path, news_path, out_dir, *, window=64, batch_size=32):
    """Re-encode every used time-series row and selected verified title on CUDA, every run."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh MOMENT and MiniLM inference requires a Colab CUDA GPU")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    panel, rows = prepare_demo_structure(panel_path, news_path, window=window)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(MOMENT_ID, model_kwargs={"task_name": "embedding"}, revision=MOMENT_REVISION)
    model.init()
    model.to("cuda").eval()
    moment_count = 0
    with torch.inference_mode():
        for ti, a in enumerate(panel["arrays"]):
            indices = np.flatnonzero(_required_rows(panel["splits"], ti, len(a["dates"])))
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                x, mask = _moment_batch(a["known_ret"], batch, window)
                output = model(x_enc=torch.from_numpy(x).to("cuda"),
                               input_mask=torch.from_numpy(mask).to("cuda")).embeddings
                encoded = output.detach().cpu().numpy().astype(np.float32, copy=False)
                del output
                if encoded.ndim != 2 or len(encoded) != len(batch) or not np.isfinite(encoded).all():
                    raise ValueError("Invalid freshly generated MOMENT embeddings")
                if a["mom"].shape[1] == 0:
                    a["mom"] = np.full((len(a["dates"]), encoded.shape[1]), np.nan, dtype=np.float32)
                if encoded.shape[1] != a["mom"].shape[1]:
                    raise ValueError("MOMENT embedding dimension changed across batches")
                a["mom"][batch] = encoded
                moment_count += len(batch)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for ti, a in enumerate(panel["arrays"]):
        if a["mom"].shape[1] == 0:
            raise ValueError(f"No new MOMENT embeddings for {DEMO_TICKERS[ti]}")
        for samples in panel["splits"].values():
            for _, j in samples[samples[:, 0] == ti]:
                if not np.isfinite(a["mom"][j - SEQ_LEN + 1:j + 1]).all():
                    raise ValueError(f"Missing newly generated MOMENT window: {DEMO_TICKERS[ti]} {j}")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(TEXT_ID, device="cuda")
    encoder.eval()
    embeddings = np.empty((len(rows), 384), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        embeddings[start:start + len(subset)] = encoder.encode(
            [row["text"] for row in subset], batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid freshly generated headline embeddings")
    cache = {"ticker": np.asarray([r["ticker"] for r in rows]),
             "decision_date": np.asarray([r["decision_date"] for r in rows], dtype="datetime64[D]"),
             "published_at": np.asarray([r["published_at"] for r in rows]),
             "embedding": embeddings}
    selected_manifest = out / "demo_titles.jsonl.gz"
    with gzip.open(selected_manifest, "wt", encoding="utf-8") as stream:
        for record in rows:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    events = load_headline_events(selected_manifest, cache)

    digest = hashlib.sha256()
    with Path(news_path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": digest.hexdigest(), "moment_model_id": MOMENT_ID,
                  "moment_revision": MOMENT_REVISION, "text_model_id": TEXT_ID,
                  "moment_embeddings": moment_count, "article_embeddings": len(rows),
                  "window": window, "assets": list(DEMO_TICKERS),
                  "source_panel": str(panel_path), "selected_manifest": str(selected_manifest),
                  "fresh_inference": True, "old_embedding_cache_used": False}
    saved = {"version": np.array(1), "horizon": np.array(5), "seq_len": np.array(SEQ_LEN),
             "tickers": np.asarray(DEMO_TICKERS), "bounds": panel["bounds"],
             "source_note": np.asarray(panel["source_note"]),
             "created_at_utc": np.asarray(provenance["created_at_utc"])}
    for i, a in enumerate(panel["arrays"]):
        saved.update({f"arr_{i}_{key}": value for key, value in a.items()})
    for split, samples in panel["splits"].items():
        saved[f"samples_{split}"] = samples
    np.savez_compressed(out / "fast_demo_panel.npz", **saved)
    np.savez_compressed(out / "fast_demo_events.npz", **events)
    (out / "fast_demo_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return panel, events, provenance
