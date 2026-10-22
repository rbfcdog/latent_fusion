import unittest
import numpy as np
import pandas as pd

from src.fusion.llama_advanced_agent import (
    apply_hysteresis_guard, compute_risk_parity_weights, detect_market_regime,
    simulate_weekly, weekly_decision_dates
)


def _mock_prices(n=45):
    sessions = pd.bdate_range("2024-08-01", periods=n)
    data = {
        "AAA": [100 + i * 1.5 for i in range(n)],  # Strong up, low vol
        "BBB": [100 + i * 0.2 + (5 if i % 2 else -5) for i in range(n)],  # Flat, high vol
        "CCC": [100 - i * 1.0 for i in range(n)],  # Steady down
        "DDD": [100 for _ in range(n)],
    }
    return pd.DataFrame(data, index=sessions)


class LlamaAdvancedAgentTest(unittest.TestCase):
    def test_weekly_decision_dates(self):
        prices = _mock_prices(40)
        weekly = weekly_decision_dates(prices, "2024-08-01", str(prices.index[-1].date()))
        self.assertGreater(len(weekly), 5)
        # Every decision must be followed by another session in prices
        for d in weekly:
            loc = prices.index.get_loc(d)
            self.assertLess(loc + 1, len(prices.index))
            # Most should be Fridays (day 4 in 0-indexed Monday=0)
            self.assertIn(d.dayofweek, [3, 4])

    def test_detect_market_regime(self):
        prices = _mock_prices(35)
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        reg = detect_market_regime(prices, 30, tickers)
        self.assertIn("regime", reg)
        self.assertIn("target_exposure", reg)
        self.assertIn(reg["regime"], ["BULL_TREND", "NEUTRAL", "HIGH_VOL_CORRECTION"])
        self.assertGreaterEqual(reg["target_exposure"], 0.40)
        self.assertLessEqual(reg["target_exposure"], 1.00)

    def test_compute_risk_parity_weights(self):
        prices = _mock_prices(35)
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        held = {"AAA", "BBB"}  # AAA has lower vol than BBB
        weights = compute_risk_parity_weights(held, prices, 30, tickers, target_exposure=1.0)
        idx_aaa = tickers.index("AAA")
        idx_bbb = tickers.index("BBB")
        self.assertGreater(weights[idx_aaa], weights[idx_bbb])  # Lower vol gets higher weight
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_apply_hysteresis_guard(self):
        prices = _mock_prices(35)
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        # AAA is rank 1, CCC is rank 4 (lowest in 4-asset basket)
        raw_actions = {"AAA": "BUY", "BBB": "BUY", "CCC": "BUY", "DDD": "SELL"}
        guarded, ranks = apply_hysteresis_guard(raw_actions, held={"DDD"}, prices=prices, location=30, tickers=tickers)
        # AAA is top momentum -> BUY allowed
        self.assertEqual(guarded["AAA"], "BUY")
        # DDD is held and low rank -> SELL allowed
        self.assertEqual(guarded["DDD"], "SELL")

    def test_simulate_weekly(self):
        prices = _mock_prices(25)
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        weekly = weekly_decision_dates(prices, "2024-08-01", str(prices.index[-1].date()))
        # BBB is rank 1 at index 1 in _mock_prices
        actions = {weekly[0]: {"BBB": "BUY", "AAA": "HOLD", "CCC": "HOLD", "DDD": "HOLD"}}
        sim = simulate_weekly(actions, weekly[:1], prices, tickers, cost_bps=10)
        self.assertIn("turnover_total", sim)
        self.assertGreater(sim["turnover_total"], 0)
        # Final net equity should be positive
        self.assertGreater(sim["series"]["net"].iloc[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
