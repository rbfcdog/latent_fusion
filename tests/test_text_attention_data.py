"""Data-only regression checks; never instantiate or run a neural network."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.fusion.headline_features import FEATURE_NAMES
from src.fusion.text_attention import TextEventDataset, load_headline_events


class TextAttentionDataTest(unittest.TestCase):
    def setUp(self):
        self.days = np.busday_offset(np.datetime64("2020-01-02", "D"), np.arange(18)).astype("datetime64[ns]")
        self.panel = {"tickers": ["AAA"], "arrays": [{"dates": self.days, "mom": np.zeros((18, 2), np.float32),
                       "known_ret": np.zeros(18, np.float32), "past_vol": np.ones(18, np.float32),
                       "target": np.zeros(18, np.float32)}],
                      "splits": {"test": np.asarray([[0, 5], [0, 6], [0, 7], [0, 9], [0, 10], [0, 15]],
                                                 dtype=np.int32)}, "seq_len": 3}
        self.rows = [
            {"ticker": "AAA", "decision_date": str(self.days[j])[:10],
             "published_at": f"2020-01-{j + 2:02d}T{8 + i:02d}:00:00Z", "url": f"https://example.com/{i}",
             "source": "example.com", "text": headline}
            for i, (j, headline) in enumerate(((6, "Company raises earnings guidance"),
                                                 (6, "Company announces merger"),
                                                 (7, "Company announces update"),
                                                 (10, "Company raises earnings guidance")))
        ]
        self.cache = {"ticker": np.array(["AAA"] * 4),
                      "decision_date": np.array([r["decision_date"] for r in self.rows], dtype="datetime64[D]"),
                      "published_at": np.array([r["published_at"] for r in self.rows]),
                      "embedding": np.array([[1, 0], [0, 1], [2**-0.5, 2**-0.5],
                                             [2**-0.5, 2**-0.5]], dtype=np.float32)}

    def _load(self, path):
        with gzip.open(path, "wt", encoding="utf-8") as file:
            for row in self.rows:
                file.write(json.dumps(row) + "\n")
        return load_headline_events(path, self.cache)

    def test_same_day_and_future_never_enter_novelty_or_news_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = self._load(Path(tmp) / "titles.jsonl.gz")
        self.assertEqual(events["features"].shape, (4, len(FEATURE_NAMES)))
        # Same-day siblings cannot seed one another, nor can tomorrow's matching vector.
        np.testing.assert_array_equal(events["novelty"][:2], [0, 0])
        self.assertAlmostEqual(float(events["novelty"][2]), 1 - 2**-0.5, places=6)
        self.assertAlmostEqual(float(events["novelty"][3]), 0.0, places=6)
        ds = TextEventDataset(self.panel, events, "test")
        self.assertEqual(ds.dim, 2 + len(FEATURE_NAMES) + 2)
        self.assertFalse(ds._news(0, 0)[1].any())
        self.assertFalse(ds._news(0, 5)[1].any())  # First event is available on day 6.
        self.assertEqual(int(ds._news(0, 6)[1].sum()), 2)
        text, mask = ds._news(0, 9)
        self.assertEqual(int(mask.sum()), 3)  # Future day-10 article excluded on day 9.
        np.testing.assert_allclose(text[mask, -1], [3 / 5, 3 / 5, 2 / 5], rtol=1e-6)
        self.assertEqual(int(ds._news(0, 10)[1].sum()), 4)
        self.assertFalse(ds._news(0, 15)[1].any())  # All four titles older than five sessions.
        self.assertEqual(ds[3][4].sum().item(), 3)

    def test_cache_metadata_order_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "titles.jsonl.gz"
            for field in ("ticker", "decision_date", "published_at"):
                original = self.cache[field].copy()
                self.cache[field] = original[[1, 0, 2, 3]] if field == "published_at" else original.copy()
                if field != "published_at":
                    self.cache[field][0] = "BBB" if field == "ticker" else np.datetime64("2020-02-01")
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "desalinhado"):
                    self._load(path)
                self.cache[field] = original
            self.cache["embedding"] = self.cache["embedding"][:-1]
            with self.assertRaisesRegex(ValueError, "inválido"):
                self._load(path)

    def test_ten_prior_events_not_same_day_or_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates = self.days[:14]
            self.rows = [{"ticker": "AAA", "decision_date": str(day)[:10],
                          "published_at": f"2020-01-01T{i:02d}:00:00Z", "url": f"https://example.com/{i}",
                          "source": "example.com", "text": "Company earnings update"}
                         for i, day in enumerate(dates)]
            # First vector is identical to last; ten preceding events for index 11
            # exclude index 0, leaving ten orthogonal vectors.
            self.cache = {"ticker": np.array(["AAA"] * 14),
                          "decision_date": dates.astype("datetime64[D]"),
                          "published_at": np.array([r["published_at"] for r in self.rows]),
                          "embedding": np.array([[1, 0]] + [[0, 1]] * 10 + [[1, 0]] * 3, np.float32)}
            events = self._load(Path(tmp) / "titles.jsonl.gz")
            self.assertEqual(float(events["novelty"][11]), 1.0)
            self.assertEqual(float(events["novelty"][12]), 0.0)


if __name__ == "__main__":
    unittest.main()
