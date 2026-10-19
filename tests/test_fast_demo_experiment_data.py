"""Data-only fast-demo checks: no embedding model or predictor forward."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.fusion.fast_demo_experiment import _summarize, run_experiment

from src.fusion.fast_demo_data import DEMO_TICKERS, prepare_demo_structure

class FastDemoExperimentDataTest(unittest.TestCase):
    def test_new_universe_labels_and_window_without_old_vectors(self):
        dates = np.busday_offset(np.datetime64("2020-01-02", "D"),
                                  np.arange(125)).astype("datetime64[ns]")
        old = {"version": 1, "seq_len": 10, "horizon": 5, "tickers": DEMO_TICKERS,
               "bounds": dates[[88, 102]], "source_note": "verified-title feed"}
        for i in range(len(DEMO_TICKERS)):
            old.update({f"arr_{i}_dates": dates,
                        f"arr_{i}_target_end": np.concatenate((dates[5:],
                                                                 np.full(5, np.datetime64("NaT", "ns")))),
                        f"arr_{i}_past_vol": np.ones(len(dates), np.float32),
                        f"arr_{i}_known_ret": np.full(len(dates), 0.001 * (i + 1), np.float32),
                        f"arr_{i}_fwd_return": np.full(len(dates), 0.01 * (i + 1), np.float32),
                        f"arr_{i}_mom": np.full((len(dates), 7), -999, np.float32),
                        f"arr_{i}_target": np.full(len(dates), 42, np.float32),
                        f"arr_{i}_market_fwd_return": np.full(len(dates), -88, np.float32)})
        for split, indices in (("train", (72, 75, 85)), ("val", (94, 100)),
                               ("test", (106, 110, 115))):
            old[f"samples_{split}"] = np.array([(i, j) for i in range(8)
                                                 for j in indices], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            panel_path, news_path = Path(tmp) / "panel.npz", Path(tmp) / "news.jsonl.gz"
            np.savez_compressed(panel_path, **old)
            with gzip.open(news_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps({"ticker": "ADBE", "decision_date": str(dates[110])[:10],
                                         "published_at": "2020-01-01T12:00:00Z",
                                         "url": "https://example.org/title", "source": "example.org",
                                         "text": "Adobe announces earnings"}) + "\n")
            panel, rows = prepare_demo_structure(panel_path, news_path, window=64)
        self.assertEqual(panel["tickers"], list(DEMO_TICKERS))
        self.assertEqual(len(rows), 1)
        self.assertEqual([set(samples[:, 1]) for samples in panel["splits"].values()],
                         [{75}, {94}, {106, 110, 115}])
        for i, array in enumerate(panel["arrays"]):
            self.assertEqual(array["mom"].shape, (125, 0))
            self.assertAlmostEqual(float(array["market_fwd_return"][110]), 0.045, places=6)
            self.assertAlmostEqual(float(array["target"][110]), 0.01 * (i + 1) - 0.045,
                                   places=6)

    def test_full_calendar_target_and_portfolio_alignment(self):
        tickers = ("ADBE", "BABA", "COST", "CVX", "GILD", "NKE", "PYPL", "QCOM")
        dates = np.busday_offset(np.datetime64("2022-09-01", "D"), np.arange(26)).astype("datetime64[ns]")
        returns = np.array([[0.0003 * (i + 1) + 0.00005 * ((j + i) % 7)
                             for j in range(len(dates))] for i in range(len(tickers))])
        future = np.array([[np.prod(1 + row[j + 1:j + 6]) - 1
                            for j in range(19)] for row in returns])
        basket = future.mean(axis=0)
        panel = {"tickers": tickers, "arrays": [
            {"dates": dates, "known_ret": returns[i], "fwd_return": future[i],
             "market_fwd_return": basket, "target": future[i] - basket}
            for i in range(len(tickers))]}
        samples = np.array([(i, j) for j in range(10, 19) for i in range(len(tickers))], dtype=np.int32)
        target = np.array([panel["arrays"][i]["target"][j] for i, j in samples])
        mask = np.array([j % 2 == 0 for _, j in samples])
        scores = {"ts_only": target.copy(), "fusion": target + 0.001, "zero": np.zeros_like(target)}
        frames, signal, portfolio, equity = _summarize(panel, samples, mask, scores, 10)
        self.assertEqual(len(frames["fusion"]), 9 * 8)
        self.assertEqual(int(frames["fusion"]["has_event"].sum()), 5 * 8)
        self.assertEqual(set(frames["ts_only"]["date"]), set(pd.DatetimeIndex(dates[10:19])))
        np.testing.assert_allclose(frames["ts_only"]["excess_return_5d"], target)
        np.testing.assert_allclose(frames["ts_only"]["asset_return_5d"],
                                   [future[i, j] for i, j in samples])
        self.assertAlmostEqual(float(signal.loc["ts_only", "mse_5d"]), 0)
        self.assertAlmostEqual(float(signal.loc["fusion", "mse_com_evento"]), 1e-6)
        self.assertAlmostEqual(float(signal.loc["fusion", "mae_5d"]), 0.001)
        self.assertEqual(int(signal.loc["zero", "n_amostras"]), len(samples))
        self.assertEqual(set(portfolio.index), {"ts_only", "fusion", "buy_hold"})
        self.assertFalse(equity.isna().any().any())
        self.assertEqual(list(equity.columns), ["ts_only", "fusion", "buy_hold"])
        self.assertEqual(equity.index[0], pd.Timestamp(dates[10]))
        self.assertEqual(equity.index[-1], pd.Timestamp(dates[20]))
        self.assertAlmostEqual(float(equity["buy_hold"].iloc[0]), 0.999)
        self.assertAlmostEqual(float(portfolio.loc["buy_hold", "bh_return_pct"]),
                               float(portfolio.loc["buy_hold", "return_pct"]))
        self.assertTrue(np.allclose([sum(a["target"][j] for a in panel["arrays"])
                                     for j in range(10, 19)], 0, atol=1e-12))

    def test_training_refuses_local_runtime_before_embeddings(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "src.fusion.fast_demo_experiment.prepare_demo_inputs",
            side_effect=AssertionError("must not generate embeddings locally"),
        ) as prepare:
            with self.assertRaisesRegex(RuntimeError, "Colab|GPU"):
                run_experiment("unread_panel.npz", "unread_events.gz", "unwritten_output")
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
