import unittest

import pandas as pd

from src.fusion.llm_numeric_hybrid import consensus_actions


class HybridConsensusTest(unittest.TestCase):
    def test_buys_only_after_both_hold_and_exits_when_either_exits(self):
        dates = pd.bdate_range("2023-01-02", periods=4)
        tickers = ["AAA", "BBB"]
        luna = {
            dates[0]: {"AAA": "BUY", "BBB": "BUY"},
            dates[1]: {"AAA": "HOLD", "BBB": "HOLD"},
            dates[2]: {"AAA": "SELL", "BBB": "HOLD"},
            dates[3]: {"AAA": "BUY", "BBB": "HOLD"},
        }
        numeric = {
            dates[0]: {"AAA": "HOLD", "BBB": "BUY"},
            dates[1]: {"AAA": "BUY", "BBB": "HOLD"},
            dates[2]: {"AAA": "HOLD", "BBB": "SELL"},
            dates[3]: {"AAA": "HOLD", "BBB": "HOLD"},
        }
        actions, audit = consensus_actions(tickers, dates, luna, numeric)
        self.assertEqual([actions[day]["AAA"] for day in dates], ["HOLD", "BUY", "SELL", "BUY"])
        self.assertEqual([actions[day]["BBB"] for day in dates], ["BUY", "HOLD", "SELL", "HOLD"])
        self.assertFalse(bool(audit.loc[(audit.date == dates[0]) & (audit.ticker == "AAA"), "hybrid_held"].iloc[0]))
        self.assertTrue(bool(audit.loc[(audit.date == dates[1]) & (audit.ticker == "AAA"), "hybrid_held"].iloc[0]))
        with self.assertRaisesRegex(ValueError, "Calendários"):
            consensus_actions(tickers, dates, luna, {dates[0]: numeric[dates[0]]})


if __name__ == "__main__":
    unittest.main()
