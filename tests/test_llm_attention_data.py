"""Data-only checks; never instantiate or forward a neural model."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.fusion.llm_attention import LLMEventDataset, align_llm_events
from src.fusion.llm_attention_experiment import _audit
from src.fusion.llm_headlines import LLM_FEATURE_NAMES


class LLMAttentionDataTest(unittest.TestCase):
    def setUp(self):
        self.days = np.busday_offset(np.datetime64("2020-01-02", "D"), np.arange(18)).astype("datetime64[ns]")
        self.panel = {"tickers": ["AAA"], "arrays": [{"dates": self.days,
                      "mom": np.zeros((18, 2), np.float32), "known_ret": np.zeros(18, np.float32),
                      "past_vol": np.ones(18, np.float32), "target": np.zeros(18, np.float32)}],
                      "splits": {"test": np.array([[0, 5], [0, 6], [0, 7], [0, 9], [0, 10], [0, 15]],
                                                  np.int32)}, "seq_len": 3}
        self.rows = [{"ticker": "AAA", "decision_date": str(self.days[j])[:10],
                      "published_at": f"2020-01-02T{9 + i:02d}:00:00Z",
                      "url": f"https://example.com/{i}", "source": "example.com",
                      "text": f"AAA headline {i}"}
                     for i, j in enumerate((6, 6, 7, 10))]
        keys = {"ticker": np.array([row["ticker"] for row in self.rows]),
                "decision_date": np.array([row["decision_date"] for row in self.rows], dtype="datetime64[D]"),
                "published_at": np.array([row["published_at"] for row in self.rows])}
        self.embeddings = {**keys, "embedding": np.array([[1, 0], [0, 1],
                          [2**-0.5, 2**-0.5], [2**-0.5, 2**-0.5]], np.float32)}
        features = np.zeros((4, len(LLM_FEATURE_NAMES)), np.float32)
        features[0, 0] = 1
        features[0, 2] = 1
        features[0, 9] = 1
        self.facts = {**keys, "features": features, "status": np.array(["ok", "ok", "invalid", "ok"])}

    def _align(self, path):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + "\n")
        return align_llm_events(path, self.embeddings, self.facts)

    def test_provenance_every_row_and_cache_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "titles.jsonl.gz"
            events = self._align(path)
            self.assertEqual(len(events["features"]), len(self.rows))
            for source in (self.embeddings, self.facts):
                for field in ("ticker", "decision_date", "published_at"):
                    original = source[field].copy()
                    altered = original.copy()
                    altered[0] = ("BBB" if field == "ticker" else np.datetime64("2020-02-01")
                                  if field == "decision_date" else "2020-02-01T00:00:00Z")
                    source[field] = altered
                    with self.subTest(source="embedding" if source is self.embeddings else "LLM", field=field):
                        with self.assertRaisesRegex(ValueError, "desalinhados"):
                            self._align(path)
                    source[field] = original
            self.rows[0]["published_at"] = "2020-02-01T00:00:00Z"
            with self.assertRaisesRegex(ValueError, "desalinhados"):
                self._align(path)

    def test_same_day_novelty_and_causal_masks_identical_in_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = self._align(Path(tmp) / "titles.jsonl.gz")
        np.testing.assert_allclose(events["novelty"][:2], [0, 0])
        self.assertAlmostEqual(float(events["novelty"][2]), 1 - 2**-0.5, places=6)
        self.assertAlmostEqual(float(events["novelty"][3]), 0.0, places=6)
        full = LLMEventDataset(self.panel, events, "test")
        embedding = LLMEventDataset(self.panel, events, "test", embeddings_only=True)
        self.assertEqual(full.dim - embedding.dim, len(LLM_FEATURE_NAMES))
        for j in range(18):
            tokens, mask = full._news(0, j)
            ablated, ablation_mask = embedding._news(0, j)
            np.testing.assert_array_equal(mask, ablation_mask)
            np.testing.assert_array_equal(tokens[:, :2], ablated[:, :2])
            np.testing.assert_array_equal(tokens[:, -2:], ablated[:, -2:])
            if j < 6 or j >= 15:
                self.assertFalse(mask.any())
        tokens, mask = full._news(0, 9)
        self.assertEqual(int(mask.sum()), 3)
        np.testing.assert_allclose(tokens[mask, -1], [3 / 5, 3 / 5, 2 / 5])
        self.assertEqual(int(full._news(0, 10)[1].sum()), 4)

    def test_invalid_and_valid_unknown_have_separate_denominators(self):
        audit = _audit(self.facts["features"], self.facts["status"])
        self.assertEqual(audit["invalid"], 1)
        self.assertEqual(audit["invalid_pct"], 25)
        self.assertEqual(audit["valid"], 3)
        self.assertAlmostEqual(audit["papel_desconhecido_pct_entre_validos"], 200 / 3)
        self.assertAlmostEqual(audit["sem_fato_estruturado_pct_entre_validos"], 200 / 3)


if __name__ == "__main__":
    unittest.main()
