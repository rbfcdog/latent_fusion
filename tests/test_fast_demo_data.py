"""CPU-only causality and eight-asset structure checks; no encoder is imported/run."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.fast_demo_data import DEMO_TICKERS, _moment_batch, prepare_demo_structure


class FastDemoDataTest(unittest.TestCase):
    def _fixture(self, root, missing_day=False, future_shift=0):
        dates = pd.bdate_range("2023-01-02", periods=120).to_numpy()
        bounds = dates[[50, 80]]
        out = {"version": 1, "horizon": 5, "seq_len": 10,
               "tickers": np.asarray(DEMO_TICKERS), "bounds": bounds,
               "samples_train": np.asarray([(i, j) for i in range(8) for j in (15, 45, 49)], np.int32),
               "samples_val": np.asarray([(i, j) for i in range(8) for j in (70, 79)], np.int32),
               "samples_test": np.asarray([(i, j) for i in range(8) for j in (90, 100)], np.int32)}
        for i in range(8):
            kept = np.delete(np.arange(len(dates)), 91) if missing_day and i == 7 else np.arange(len(dates))
            ret = (np.arange(len(dates), dtype=np.float32) + 1) * (i + 1) * 1e-4
            forward = np.full(len(dates), (i + 1) * .01 + (future_shift if i == 0 else 0), np.float32)
            end = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
            end[:-5] = dates[5:]
            out.update({f"arr_{i}_dates": dates[kept], f"arr_{i}_known_ret": ret[kept],
                        f"arr_{i}_past_vol": np.full(len(kept), .2, np.float32),
                        f"arr_{i}_fwd_return": forward[kept], f"arr_{i}_target_end": end[kept],
                        f"arr_{i}_mom": np.full((len(kept), 2), 999., np.float32),
                        f"arr_{i}_target": np.full(len(kept), -999., np.float32)})
        panel_path = root / "panel.npz"
        np.savez(panel_path, **out)
        news_path = root / "titles.jsonl.gz"
        with gzip.open(news_path, "wt", encoding="utf-8") as file:
            for ticker in DEMO_TICKERS:
                for j in (90, 100):
                    file.write(json.dumps({"ticker": ticker, "decision_date": str(dates[j])[:10],
                                           "published_at": str(dates[j - 1])[:10] + "T10:00:00Z",
                                           "url": "https://example.com/title", "source": "example.com",
                                           "text": f"Verified title {ticker}"}) + "\n")
            file.write(json.dumps({"ticker": DEMO_TICKERS[0], "decision_date": "2023-12-01",
                                   "published_at": "2023-11-30T10:00:00Z", "url": "https://example.com/late",
                                   "source": "example.com", "text": "After audit"}) + "\n")
        return panel_path, news_path

    def test_causal_windows_split_purge_basket_and_no_old_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, news = self._fixture(root)
            panel, rows = prepare_demo_structure(source, news, window=8)
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(r["decision_date"] <= "2023-11-30" for r in rows))
            self.assertTrue(all(r["published_at"][:10] < r["decision_date"] for r in rows))
            self.assertEqual(set(panel["tickers"]), set(DEMO_TICKERS))
            for a in panel["arrays"]:
                self.assertEqual(a["mom"].shape, (120, 0))
                self.assertEqual(a["mom"].dtype, np.float32)
                self.assertAlmostEqual(float(a["market_fwd_return"][90]), .045, places=6)
                self.assertNotEqual(float(a["target"][90]), -999.)
            for split, bound in (("train", panel["bounds"][0]), ("val", panel["bounds"][1])):
                for ti, j in panel["splits"][split]:
                    self.assertGreaterEqual(j, 17)
                    self.assertLessEqual(panel["arrays"][ti]["target_end"][j], bound)
            self.assertEqual(set(panel["splits"]["train"][:, 1]), {45})
            self.assertEqual(set(panel["splits"]["val"][:, 1]), {70})
            a = panel["arrays"][0]
            x, mask = _moment_batch(a["known_ret"], np.array([45, 70]), 8)
            np.testing.assert_array_equal(x[0, 0, :8], a["known_ret"][38:46])
            self.assertTrue(mask[:, :8].all())
            self.assertFalse(mask[:, 8:].any())
            self.assertTrue((x[:, :, 8:] == 0).all())
            np.testing.assert_array_equal(x[1, 0, :8], a["known_ret"][63:71])
            changed, _ = self._fixture(root, future_shift=.5)
            changed_panel, _ = prepare_demo_structure(changed, news, window=8)
            x_changed, _ = _moment_batch(changed_panel["arrays"][0]["known_ret"], np.array([45, 70]), 8)
            np.testing.assert_array_equal(x, x_changed)
            self.assertAlmostEqual(float(changed_panel["arrays"][0]["target"][90] - a["target"][90]),
                                   .5 * 7 / 8, places=6)

    def test_reject_missing_bh_calendar_after_first_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, news = self._fixture(Path(tmp), missing_day=True)
            with self.assertRaisesRegex(ValueError, "Incomplete eight-asset daily price calendar"):
                prepare_demo_structure(source, news, window=8)


if __name__ == "__main__":
    unittest.main()
