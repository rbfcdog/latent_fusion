import unittest

import numpy as np
import pandas as pd

from src.fusion.event_experiment import _portfolio
from src.fusion.presentation_profiles import PROFILES, compare_profiles, profile_weights


class PresentationProfilesTest(unittest.TestCase):
    def test_fully_invested_caps_hold_for_extreme_and_flat_scores(self):
        for policy in PROFILES.values():
            for scores in (np.array([100.0, -100.0, 0, 0, 0, 0, 0, 0]), np.ones(8)):
                weights = profile_weights(scores, policy)
                self.assertAlmostEqual(weights.sum(), 1)
                self.assertTrue(np.all(weights >= 0))
                self.assertLessEqual(weights.max(), policy["max_position"] + 1e-12)
        with self.assertRaisesRegex(ValueError, "inviável"):
            profile_weights(np.ones(8), {"strength": 1.0, "max_position": 0.1})

    def test_profile_changes_only_past_scores_and_keeps_benchmark_calendar(self):
        tickers = [f"T{i}" for i in range(8)]
        dates = np.busday_offset(np.datetime64("2023-01-02"), np.arange(145)).astype("datetime64[ns]")
        returns = np.array([0.0005 * np.sin(np.arange(145) / 13 + i) for i in range(8)])
        panel = {"tickers": tickers, "arrays": [{"dates": dates, "known_ret": row} for row in returns]}
        records = [{"date": pd.Timestamp(dates[j]), "ticker": ticker,
                    "score": (i - 3.5) * (1 + 0.01 * j)}
                   for j in range(20, 140) for i, ticker in enumerate(tickers)]
        frames = {name: pd.DataFrame(records).copy() for name in ("ts_only", "metadata", "text_residual")}
        end = pd.Timestamp(dates[144])
        summary, curves, allocations, benchmark = compare_profiles(panel, frames, end, cost_bps=10)
        self.assertEqual(len(summary), 9)
        self.assertTrue(curves.index.equals(benchmark.index))
        self.assertEqual(benchmark.index[-1], end)
        self.assertTrue(np.allclose(allocations.groupby(level=[0, 1]).peso_pct.sum(), 100))
        self.assertTrue(np.allclose(curves[("arrojado", "ts_only")],
                                    curves[("arrojado", "metadata")]))
        self.assertLess(summary.loc[("conservador", "ts_only"), "rebalanceamentos"],
                        summary.loc[("arrojado", "ts_only"), "rebalanceamentos"])
        one = _portfolio(panel, frames["ts_only"], 10, end_date=end)
        np.testing.assert_allclose(one["bh"], benchmark)
        cost_free = _portfolio(panel, frames["ts_only"], 0, end_date=end)
        self.assertGreater(cost_free["strategy"].iloc[-1], one["strategy"].iloc[-1])
        frame_later_changed = frames["ts_only"].copy()
        frame_later_changed.loc[frame_later_changed["date"] > pd.Timestamp(dates[20]), "score"] *= -1
        alternate = _portfolio(panel, frame_later_changed, 10, end_date=end)
        self.assertAlmostEqual(float(one["strategy"].iloc[0]), float(alternate["strategy"].iloc[0]))
        self.assertAlmostEqual(float(one["strategy"].iloc[1]), float(alternate["strategy"].iloc[1]))
        with self.assertRaisesRegex(ValueError, "Data final"):
            _portfolio(panel, frames["ts_only"], 10, end_date=dates[21])


if __name__ == "__main__":
    unittest.main()
