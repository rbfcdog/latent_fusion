import unittest

import numpy as np
import pandas as pd

from src.fusion.presentation_regimes import analyze_regimes


def sample_panel():
    dates = np.busday_offset(np.datetime64("2022-01-03"), np.arange(110)).astype("datetime64[ns]")
    tickers = ["A", "B", "C", "D"]
    arrays = []
    for i, ticker in enumerate(tickers):
        returns = 0.003 * np.sin(np.arange(len(dates)) * 0.7 + i) + 0.002 * np.cos(np.arange(len(dates)) / 8)
        arrays.append({"dates": dates, "known_ret": returns,
                       "target": np.arange(len(dates), dtype=float) + i,
                       "fwd_return": np.arange(len(dates), dtype=float) * 2,
                       "market_fwd_return": np.arange(len(dates), dtype=float) * 3})
    test = np.array([(i, j) for j in range(78, 99) for i in range(4)])
    panel = {"tickers": tickers, "arrays": arrays,
             "bounds": dates[[60, 73]],
             "splits": {"train": np.array([(i, j) for j in range(10, 55) for i in range(4)]),
                        "val": np.array([(i, j) for j in range(64, 69) for i in range(4)]),
                        "test": test}}
    records = [{"ticker": tickers[i], "date": dates[j],
                "score": float((i - 1.5) * 0.002 + 0.0001 * j),
                "excess_return_5d": float((i - 1.5) * 0.003 + 0.001 * np.cos(j)),
                "has_event": i % 2 == 0} for i, j in test]
    predictions = {name: pd.DataFrame(records).copy() for name in
                   ("ts_only", "metadata", "text_residual", "shuffled", "text_only", "sentiment_only")}
    for name, delta in (("metadata", 0.0002), ("text_residual", 0.0004),
                        ("shuffled", -0.0002), ("text_only", 0.0001), ("sentiment_only", 0.0003)):
        predictions[name]["score"] += delta
    return panel, predictions, dates


class RegimeDiagnosticTest(unittest.TestCase):
    def test_future_closes_and_forward_targets_cannot_change_past_states(self):
        panel, predictions, dates = sample_panel()
        summary, states = analyze_regimes(panel, predictions)
        changed, same_predictions, _ = sample_panel()
        for asset in changed["arrays"]:
            asset["known_ret"][88:] += 0.20
            asset["target"][:] = -987654
            asset["fwd_return"][:] = -123456
            asset["market_fwd_return"][:] = -789012
        _, changed_states = analyze_regimes(changed, same_predictions)
        past = states["date"] < pd.Timestamp(dates[88])
        pd.testing.assert_frame_equal(states.loc[past].reset_index(drop=True),
                                      changed_states.loc[past].reset_index(drop=True),
                                      check_exact=True)
        self.assertEqual(states.attrs["train_variance_by_regime"],
                         changed_states.attrs["train_variance_by_regime"])
        self.assertEqual(states.attrs["fit_days"], 61)
        self.assertEqual(int(summary["n"].sum()), 84)
        self.assertEqual(int(summary["n_with_news"].sum()), 42)
        self.assertIn("mse_gain_vs_numeric_shuffled", summary)
        self.assertIn("ic_change_vs_numeric_text_only", summary)
        self.assertTrue((states.loc[states["date"] > pd.Timestamp(dates[60]), "is_fit_date"] == False).all())

    def test_missing_close_and_misaligned_controls_are_rejected(self):
        panel, predictions, _ = sample_panel()
        predictions["shuffled"] = predictions["shuffled"].iloc[:-1]
        with self.assertRaisesRegex(ValueError, "calendar"):
            analyze_regimes(panel, predictions)
        panel, predictions, _ = sample_panel()
        panel["arrays"][1]["known_ret"][80] = np.nan
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            analyze_regimes(panel, predictions)


if __name__ == "__main__":
    unittest.main()
