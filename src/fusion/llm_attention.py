"""Align title-only LLM facts with cached MiniLM tokens and the price panel."""

from __future__ import annotations

from collections import deque
import gzip
import json

import numpy as np

from src.fusion.event_model import FEED_END
from src.fusion.llm_headlines import LLM_FEATURE_NAMES
from src.fusion.text_attention import TextEventDataset


def align_llm_events(news_path, embeddings, facts):
    """Require one-to-one manifest provenance before joining independent caches."""
    fields = ("ticker", "decision_date", "published_at")
    vectors = embeddings["embedding"]
    features = facts["features"]
    n = len(vectors)
    if (vectors.ndim != 2 or features.shape != (n, len(LLM_FEATURE_NAMES)) or
            any(len(embeddings[field]) != n or len(facts[field]) != n for field in fields) or
            len(facts["status"]) != n or
            not np.isfinite(vectors).all() or not np.isfinite(features).all() or
            not np.isin(facts["status"], ("ok", "invalid")).all() or
            np.any(features[facts["status"] == "invalid"] != 0)):
        raise ValueError("Caches de manchetes inválidos")
    novelty = np.empty(n, dtype=np.float32)
    history, pending, current_date = {}, {}, {}
    count = 0
    with gzip.open(news_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in
                                                    (*fields, "url", "source", "text")):
                raise ValueError("Notícia fora do esquema do manifesto")
            date = np.datetime64(row["decision_date"], "D")
            if date > FEED_END:
                continue
            if count >= n or any(
                str(data["ticker"][count]) != row["ticker"] or
                np.datetime64(data["decision_date"][count], "D") != date or
                str(data["published_at"][count]) != row["published_at"]
                for data in (embeddings, facts)
            ):
                raise ValueError(f"Caches desalinhados com manifesto na linha {count + 1}")
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
            vector = vectors[count]
            novelty[count] = (np.clip(1 - max(float(vector @ prior) for prior in earlier), 0, 2)
                              if earlier else 0)
            pending.setdefault(ticker, []).append(vector)
            count += 1
    if count != n:
        raise ValueError(f"Caches desalinhados com manifesto: {count} vs {n}")
    return {**embeddings, "features": features, "novelty": novelty, "status": facts["status"]}


class LLMEventDataset(TextEventDataset):
    """Same eight-title/five-session selection for full and embeddings-only tokens."""

    def __init__(self, panel, events, split, embeddings_only=False):
        tokens = {**events, "features": events["features"][:, :0] if embeddings_only else events["features"]}
        super().__init__(panel, tokens, split)
        self.dim = events["embedding"].shape[1] + tokens["features"].shape[1] + 2
