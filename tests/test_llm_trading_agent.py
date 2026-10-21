import json
import unittest

import numpy as np
import pandas as pd

from src.fusion.llm_trading_agent import _backtest, parse_actions


class LLMTradingAgentTest(unittest.TestCase):
    def test_agent_trades_after_decision_close_and_sells_to_cash(self):
        dates = np.busday_offset(np.datetime64("2023-01-02"), np.arange(12)).astype("datetime64[ns]")
        first = np.zeros(12)
        first[0] = 1.0
        first[1] = 0.10
        first[6] = 0.10
        panel = {"tickers": ["AAA", "BBB"], "arrays": [
            {"dates": dates, "known_ret": first},
            {"dates": dates, "known_ret": np.zeros(12)}]}
        first_day, sell_day = pd.Timestamp(dates[0]), pd.Timestamp(dates[5])
        actions = {first_day: {"AAA": "BUY", "BBB": "HOLD"},
                   sell_day: {"AAA": "SELL", "BBB": "HOLD"}}
        curve = _backtest(panel, [first_day, sell_day], actions, cost_bps=100)
        self.assertAlmostEqual(curve.loc[first_day, "strategy"], 0.99)
        self.assertAlmostEqual(curve.iloc[1]["strategy"], 0.99 * 1.10)
        self.assertAlmostEqual(curve.loc[sell_day, "strategy"], 0.99 * 1.10 * 0.99)
        self.assertAlmostEqual(curve.iloc[-1]["strategy"], curve.loc[sell_day, "strategy"])
        self.assertAlmostEqual(curve["turnover"].sum(), 2.0)
        self.assertAlmostEqual(curve.loc[sell_day, "cash_weight"], 1.0)
        self.assertGreater(curve.iloc[-1]["bh"], curve.loc[sell_day, "bh"])
        with self.assertRaisesRegex(ValueError, "Venda sem posição"):
            _backtest(panel, [first_day], {first_day: {"AAA": "SELL", "BBB": "HOLD"}}, cost_bps=10)

    def test_rejects_unheld_sales_and_invented_evidence(self):
        raw = json.dumps({"actions": {"AAA": "BUY", "BBB": "HOLD"},
                          "evidence": {"AAA": ["Audited headline"], "BBB": []},
                          "reason": "Possible event; uncertainty remains."})
        self.assertEqual(parse_actions(raw, ["AAA", "BBB"], set(),
                                       {"AAA": ["Audited headline"], "BBB": []})["actions"]["AAA"], "BUY")
        with self.assertRaisesRegex(ValueError, "Citação"):
            parse_actions(raw, ["AAA", "BBB"], set(), {"AAA": [], "BBB": []})
        with self.assertRaisesRegex(ValueError, "Ação inválida"):
            parse_actions(raw, ["AAA", "BBB"], {"AAA"}, {"AAA": ["Audited headline"], "BBB": []})


if __name__ == "__main__":
    unittest.main()
