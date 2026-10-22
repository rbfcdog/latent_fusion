import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.fusion.llama_tool_agent import (
    exec_analyze_numeric, exec_check_correlation_risk, exec_estimate_trade_cost,
    parse_agent_actions, run_single_decision_agent
)


def _mock_prices():
    sessions = pd.bdate_range("2024-08-01", periods=30)
    data = {
        "AAA": [100 + i for i in range(30)],      # Steady up
        "BBB": [100 - i * 0.5 for i in range(30)],# Steady down
        "CCC": [100 + i * 2 for i in range(30)],  # Strong up
        "DDD": [100 for _ in range(30)],          # Flat
    }
    return pd.DataFrame(data, index=sessions)


class LlamaToolAgentTest(unittest.TestCase):
    def test_exec_analyze_numeric(self):
        prices = _mock_prices()
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        res = exec_analyze_numeric(prices, 25, ["AAA", "CCC"], tickers)
        self.assertIn("AAA", res)
        self.assertIn("CCC", res)
        self.assertEqual(res["CCC"]["rank_of_16"], 1)  # CCC has highest return
        self.assertGreater(res["CCC"]["20d_momentum_pct"], res["AAA"]["20d_momentum_pct"])
        self.assertIn("z_score_20d_ema", res["AAA"])

    def test_exec_check_correlation_risk(self):
        prices = _mock_prices()
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        # AAA and CCC are perfectly positively correlated linear trends
        res = exec_check_correlation_risk(prices, 25, ["AAA", "CCC"], tickers)
        self.assertIn("mean_correlation", res)
        self.assertGreater(res["mean_correlation"], 0.9)
        self.assertTrue(len(res["high_correlation_pairs"]) > 0)

    def test_exec_estimate_trade_cost(self):
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        held = {"AAA"}
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        # Candidate action: BUY BBB, HOLD AAA
        res = exec_estimate_trade_cost({"AAA": "HOLD", "BBB": "BUY"}, held, tickers, cost_bps=10, current_weights=weights)
        self.assertEqual(res["current_held"], ["AAA"])
        self.assertEqual(res["target_held"], ["AAA", "BBB"])
        # Turnover from [1, 0, 0, 0] to [0.5, 0.5, 0, 0] is |0.5-1| + |0.5-0| = 1.0
        self.assertAlmostEqual(res["turnover_weight"], 1.0)
        self.assertAlmostEqual(res["estimated_fee_bps"], 10.0)

    def test_parse_agent_actions(self):
        tickers = ["AAA", "BBB"]
        raw = '{"actions": {"AAA": "BUY", "BBB": "SELL"}, "reason": "momentum"}'
        actions, reason, ok = parse_agent_actions(raw, tickers, held={"BBB"})
        self.assertTrue(ok)
        self.assertEqual(actions, {"AAA": "BUY", "BBB": "SELL"})
        self.assertEqual(reason, "momentum")


if __name__ == "__main__":
    unittest.main()
