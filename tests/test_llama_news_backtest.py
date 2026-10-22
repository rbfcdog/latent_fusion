import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.llama_news_backtest import (
    build_context, decision_dates, load_news, messages_for, numeric_actions,
    parse_actions, run_llama_backtest, simulate, _apply)


def _prices():
    sessions = pd.bdate_range("2024-08-01", periods=40)
    data = {}
    for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        rng = np.random.default_rng(i)
        data[ticker] = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(sessions)))
    return pd.DataFrame(data, index=sessions)


def _news_file(root):
    rows = [
        {"ticker": "AAA", "title": "AAA beats earnings", "source": "Reuters",
         "url": "https://a/1", "published_at": "2024-08-05T14:00:00+00:00",
         "first_seen_at": "2026-10-22T02:00:00Z", "capture_run": "t"},
        {"ticker": "AAA", "title": "AAA surges on guidance", "source": "CNBC",
         "url": "https://a/2", "published_at": "2024-08-12T14:00:00+00:00",
         "first_seen_at": "2026-10-22T02:00:00Z", "capture_run": "t"},
        {"ticker": "BBB", "title": "BBB probe reported", "source": "WSJ",
         "url": "https://b/1", "published_at": "2024-08-20T20:00:00+00:00",
         "first_seen_at": "2026-10-22T02:00:00Z", "capture_run": "t"},
    ]
    path = root / "news.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path

class LlamaNewsBacktestTest(unittest.TestCase):
    def test_headline_lag_and_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            news = load_news(_news_file(root))
            prices = _prices()
            date = prices.index[10]
            context = build_context(prices, news, date, ["AAA", "BBB"], "news", max_headlines=8)
            titles = [h["title"] for h in context["assets"]["AAA"]["available_headlines"]]
            self.assertEqual(titles, ["AAA surges on guidance", "AAA beats earnings"])
            self.assertEqual([h["title"] for h in context["assets"]["BBB"]["available_headlines"]],
                             [])
            empty = build_context(prices, news, date, ["AAA"], "no_news")
            self.assertEqual(empty["assets"]["AAA"]["available_headlines"], [])
            self.assertEqual(context["execution_date"], prices.index[11].date().isoformat())

    def test_parse_actions_clamps_and_fallback(self):
        tickers = ["AAA", "BBB"]
        titles = {"AAA": ["AAA beats earnings"], "BBB": []}
        raw = json.dumps({"actions": {"AAA": "BUY", "BBB": "SELL"},
                          "evidence": {"AAA": "AAA beats earnings", "BBB": "fabricada"}})
        actions, evidence, ok = parse_actions(raw, tickers, set(), titles)
        self.assertTrue(ok)
        self.assertEqual(actions, {"AAA": "BUY", "BBB": "HOLD"})
        self.assertEqual(evidence, {"AAA": "AAA beats earnings"})
        actions, _, ok = parse_actions("no json here", tickers, {"AAA"}, titles)
        self.assertFalse(ok)
        self.assertEqual(actions, {"AAA": "HOLD", "BBB": "HOLD"})

    def test_simulate_next_close_fill_and_cost(self):
        prices = _prices()
        dates = [prices.index[0]]
        equity = simulate({dates[0]: {"AAA": "BUY", "BBB": "HOLD", "CCC": "HOLD", "DDD": "HOLD"}},
                          dates, prices, ["AAA", "BBB", "CCC", "DDD"], cost_bps=10)
        self.assertEqual(equity.index[0], prices.index[1])
        self.assertAlmostEqual(equity["gross"].iloc[0], 1.0)
        self.assertAlmostEqual(equity["net"].iloc[0], 1.0 - 10 / 10_000)

    def test_cache_resume_skips_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            news_path = _news_file(root)
            prices = _prices()
            calls = []

            def generate(messages):
                calls.append(messages)
                return json.dumps({"actions": {t: "HOLD" for t in ["AAA", "BBB", "CCC", "DDD"]},
                                   "evidence": {}})

            first = run_llama_backtest(prices, news_path, root / "out",
                                       tickers=["AAA", "BBB", "CCC", "DDD"], model_tag="stub",
                                       generate_fn=generate, start="2024-08-01",
                                       end=str(prices.index[15].date()), every=5)
            n_calls = len(calls)
            self.assertGreater(n_calls, 0)
            second = run_llama_backtest(prices, news_path, root / "out",
                                        tickers=["AAA", "BBB", "CCC", "DDD"], model_tag="stub",
                                        generate_fn=generate, start="2024-08-01",
                                        end=str(prices.index[15].date()), every=5)
            self.assertEqual(len(calls), n_calls)
            self.assertAlmostEqual(first["summary"]["news"]["total_return"],
                                   second["summary"]["news"]["total_return"])
            self.assertIn("paired_diff", second["summary"])

    def test_numeric_rule_no_lookahead(self):
        prices = _prices()
        date = prices.index[10]
        actions = numeric_actions(prices, date, ["AAA", "BBB", "CCC", "DDD"], set(), top_k=2)
        self.assertEqual(sum(1 for v in actions.values() if v == "BUY"), 2)
        self.assertEqual(_apply({"AAA"}, {"AAA": "HOLD", "BBB": "BUY"}), {"AAA", "BBB"})


if __name__ == "__main__":
    unittest.main()
