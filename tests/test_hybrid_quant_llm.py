import unittest
import numpy as np
import pandas as pd

from src.fusion.hybrid_quant_llm import (
    classify_headline_signal, compute_hybrid_actions,
    extract_llm_event_signals, simulate_hybrid
)
from src.fusion.llama_advanced_agent import weekly_decision_dates


def _mock_data():
    sessions = pd.bdate_range("2024-08-01", periods=30)
    data = {
        "AAA": [100 + i * 2.0 for i in range(30)],  # Rank 1 (Strong up)
        "BBB": [100 + i * 1.5 for i in range(30)],  # Rank 2
        "CCC": [100 + i * 0.8 for i in range(30)],  # Rank 3
        "DDD": [100 - i * 1.0 for i in range(30)],  # Rank 4 (Down)
    }
    prices = pd.DataFrame(data, index=sessions)
    news = [
        {"ticker": "AAA", "title": "SEC opens probe into accounting fraud at AAA",
         "published_ts": pd.Timestamp("2024-08-05T12:00:00Z")},
        {"ticker": "CCC", "title": "FDA grants breakthrough drug approval for CCC therapy",
         "published_ts": pd.Timestamp("2024-08-05T12:00:00Z")},
    ]
    return prices, news


class HybridQuantLLMTest(unittest.TestCase):
    def test_classify_headline_signal(self):
        self.assertEqual(classify_headline_signal("SEC opens antitrust probe into tech giant"), -1)
        self.assertEqual(classify_headline_signal("FDA approval granted for cancer treatment"), +1)
        self.assertEqual(classify_headline_signal("Alibaba unveils new AI chip"), +1)
        self.assertEqual(classify_headline_signal("Company holds annual general meeting in Chicago"), 0)

    def test_veto_blocks_top_momentum_buy(self):
        prices, _ = _mock_data()
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        # AAA is Rank 1, but has a negative probe event
        headlines = {
            "AAA": [{"title": "SEC opens probe into accounting fraud at AAA"}],
            "BBB": [], "CCC": [], "DDD": []
        }
        actions, ranks, sigs, notes = compute_hybrid_actions(
            prices, 20, tickers, held=set(), headlines_by_ticker=headlines,
            buy_rank_cutoff=2, sell_rank_cutoff=3, boost_max_rank=3
        )
        self.assertEqual(ranks["AAA"], 1)
        self.assertEqual(sigs["AAA"], -1)
        # Even though AAA is rank 1, LLM Veto blocks the purchase!
        self.assertEqual(actions["AAA"], "HOLD")
        self.assertIn("LLM BLOCK", notes["AAA"])
        # BBB is rank 2 with no news -> regular momentum BUY allowed!
        self.assertEqual(actions["BBB"], "BUY")

    def test_catalyst_boost_promotes_mid_rank(self):
        prices, _ = _mock_data()
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        # CCC is rank 3 (outside buy_rank_cutoff=2), but has a positive FDA approval event
        headlines = {
            "AAA": [], "BBB": [],
            "CCC": [{"title": "FDA approval granted for cancer treatment at CCC"}],
            "DDD": []
        }
        actions, ranks, sigs, notes = compute_hybrid_actions(
            prices, 20, tickers, held=set(), headlines_by_ticker=headlines,
            buy_rank_cutoff=1, sell_rank_cutoff=3, boost_max_rank=3
        )
        self.assertEqual(ranks["CCC"], 3)
        self.assertEqual(sigs["CCC"], +1)
        # CCC was promoted into BUY by catalyst boost!
        self.assertEqual(actions["CCC"], "BUY")
        self.assertIn("BOOST", notes["CCC"])

    def test_veto_triggers_immediate_sell_of_held_stock(self):
        prices, _ = _mock_data()
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        # AAA is held and Rank 1, but has an active probe
        headlines = {
            "AAA": [{"title": "DOJ probe launched into AAA executive"}],
            "BBB": [], "CCC": [], "DDD": []
        }
        actions, ranks, sigs, notes = compute_hybrid_actions(
            prices, 20, tickers, held={"AAA", "BBB"}, headlines_by_ticker=headlines,
            buy_rank_cutoff=2, sell_rank_cutoff=3
        )
        # AAA is rank 1 but forced to SELL due to LLM Veto!
        self.assertEqual(actions["AAA"], "SELL")
        self.assertIn("LLM VETO", notes["AAA"])
        # BBB is rank 2, held -> keeps HOLD
        self.assertEqual(actions["BBB"], "HOLD")

    def test_simulate_hybrid(self):
        prices, news = _mock_data()
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        weekly = weekly_decision_dates(prices, "2024-08-01", str(prices.index[-1].date()))
        sim = simulate_hybrid(weekly[:3], prices, tickers, news, cost_bps=10)
        self.assertIn("metrics", sim)
        self.assertIn("series", sim)
        self.assertGreater(len(sim["audit"]), 0)


if __name__ == "__main__":
    unittest.main()
