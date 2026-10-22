import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.llama_news_backtest import (
    build_context, buy_and_hold, decision_dates, load_news, messages_for,
    numeric_actions, parse_actions, run_llama_backtest, simulate, _apply)


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
    def test_headline_lag_cap_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            news = load_news(_news_file(root))
            prices = _prices()
            date = prices.index[10]
            context = build_context(prices, news, date, ["AAA", "BBB"], "news",
                                    max_headlines=8, held={"BBB"}, cost_bps=10)
            titles = [h["title"] for h in context["assets"]["AAA"]["available_headlines"]]
            self.assertEqual(titles, ["AAA surges on guidance", "AAA beats earnings"])
            self.assertEqual([h["title"] for h in context["assets"]["BBB"]["available_headlines"]],
                             [])
            self.assertEqual(context["held_before"], ["BBB"])
            self.assertFalse(context["cash_before"])
            self.assertTrue(context["assets"]["BBB"]["held"])
            empty = build_context(prices, news, date, ["AAA"], "no_news")
            self.assertEqual(empty["assets"]["AAA"]["available_headlines"], [])
            self.assertTrue(empty["cash_before"])
            self.assertEqual(context["execution_date"], prices.index[11].date().isoformat())
            messages = messages_for(context)
            self.assertIn("Currently holding: BBB", messages[1]["content"])
            self.assertIn("[HELD]", messages[1]["content"])

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

    def test_simulate_fill_cost_drift_and_hold_freeze(self):
        prices = _prices()
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        first = prices.index[0]
        equity = simulate({first: {"AAA": "BUY", "BBB": "HOLD", "CCC": "HOLD", "DDD": "HOLD"}},
                          [first], prices, tickers, cost_bps=10)
        self.assertAlmostEqual(equity["net"].iloc[0], 1.0 - 10 / 10_000)
        second = prices.index[5]
        both = simulate({first: {"AAA": "BUY", "BBB": "HOLD", "CCC": "HOLD", "DDD": "HOLD"},
                         second: {"AAA": "HOLD", "BBB": "BUY", "CCC": "HOLD", "DDD": "HOLD"}},
                        [first, second], prices, tickers, cost_bps=10)
        self.assertEqual(len(both), len(prices) - 1)
        frozen = simulate({first: {"AAA": "BUY", "BBB": "HOLD", "CCC": "HOLD", "DDD": "HOLD"},
                           second: {"AAA": "HOLD", "BBB": "HOLD", "CCC": "HOLD", "DDD": "HOLD"}},
                          [first, second], prices, tickers, cost_bps=10)
        self.assertAlmostEqual((frozen["net"] / frozen["gross"]).iloc[-1], 1.0 - 10 / 10_000)

    def test_drift_matches_buy_and_hold_without_extra_orders(self):
        base = pd.date_range("2024-08-01", periods=8, freq="D")
        sessions = pd.DatetimeIndex([base[0], base[2], base[4], base[6]])
        prices = pd.DataFrame({"AAA": [100.0, 100.0, 200.0, 200.0],
                               "BBB": [100.0, 100.0, 100.0, 200.0]}, index=sessions)
        decision = prices.index[0]
        equity = simulate({decision: {"AAA": "BUY", "BBB": "BUY"}}, [decision],
                          prices, ["AAA", "BBB"], cost_bps=10)
        self.assertAlmostEqual(float(equity["net"].iloc[-1]), 2.0 * (1 - 10 / 10_000))

    def test_buy_and_hold_benchmark(self):
        prices = _prices()
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        bench = buy_and_hold(prices, tickers, prices.index[1], cost_bps=10)
        expected = (1 - 10 / 10_000) * float(
            ((prices.iloc[-1] / prices.loc[prices.index[1]])).mean())
        self.assertAlmostEqual(float(bench.iloc[-1]), expected)

    def test_cache_resume_skips_generate_and_records_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            news_path = _news_file(root)
            prices = _prices()
            calls = []

            def generate(messages):
                calls.append(messages)
                return json.dumps({"actions": {t: "HOLD" for t in ["AAA", "BBB", "CCC", "DDD"]},
                                   "evidence": {}}), {"prompt_tokens": 10, "completion_tokens": 5}

            kwargs = dict(prices=prices, news_path=news_path, tickers=["AAA", "BBB", "CCC", "DDD"],
                          model_tag="stub", start="2024-08-01",
                          end=str(prices.index[15].date()), every=5)
            first = run_llama_backtest(out_dir=root / "out", generate_fn=generate, **kwargs)
            n_calls = len(calls)
            self.assertGreater(n_calls, 0)
            self.assertEqual(first["usage_total"]["prompt_tokens"], 10 * n_calls)
            second = run_llama_backtest(out_dir=root / "out", generate_fn=generate, **kwargs)
            self.assertEqual(len(calls), n_calls)
            self.assertAlmostEqual(first["summary"]["news"]["total_return"],
                                   second["summary"]["news"]["total_return"])
            self.assertEqual(second["usage_total"]["calls"], len(calls))

    def test_numeric_rule_no_lookahead(self):
        prices = _prices()
        date = prices.index[10]
        actions = numeric_actions(prices, date, ["AAA", "BBB", "CCC", "DDD"], set(), top_k=2)
        self.assertEqual(sum(1 for v in actions.values() if v == "BUY"), 2)
        self.assertEqual(_apply({"AAA"}, {"AAA": "HOLD", "BBB": "BUY"}), {"AAA", "BBB"})


if __name__ == "__main__":
    unittest.main()
