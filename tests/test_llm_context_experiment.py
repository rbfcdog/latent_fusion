import json
import unittest

import numpy as np

from src.fusion.llm_context_experiment import _make_requests, parse_response


class LLMContextTest(unittest.TestCase):
    def test_available_context_excludes_future_news_returns_and_targets(self):
        dates = np.busday_offset(np.datetime64("2023-01-02"), np.arange(80)).astype("datetime64[ns]")
        arrays = [{"dates": dates, "known_ret": np.arange(80, dtype=float) / 1000,
                   "past_vol": np.full(80, 0.2), "target": np.full(80, 888.0),
                   "fwd_return": np.full(80, 999.0), "mom": np.full((80, 1), 777.0)}
                  for _ in range(8)]
        for array in arrays:
            array["known_ret"][74] = 999.0
        panel = {"tickers": [f"EQ{i}" for i in range(8)], "arrays": arrays,
                 "splits": {"test": np.asarray([[0, 73], [0, 74]])}}
        news = [{"ticker": "EQ0", "decision_date": str(dates[index])[:10],
                 "published_at": str(dates[index - 1])[:10] + "T12:00:00Z", "source": "verified",
                 "text": title}
                for index, title in ((68, "Expired title"), (69, "Available title"),
                                     (73, "Today's title"), (75, "Future title"))]
        with self.assertRaisesRegex(ValueError, "atraso"):
            _make_requests(panel, [{**news[1], "published_at": news[1]["decision_date"] + "T12:00:00Z"}], "test")
        requests = _make_requests(panel, news, "test")
        context = requests[0]["context"]
        self.assertEqual([row["title"] for row in context["available_headlines"]],
                         ["Available title", "Today's title"])
        self.assertEqual(context["asset_past_20_daily_returns"],
                         [round(index / 1000, 6) for index in range(54, 74)])
        self.assertEqual(context["basket_latest_daily_returns"]["EQ0"], 0.073)
        self.assertNotIn("Future title", requests[0]["messages"][1]["content"])
        self.assertNotIn("target", requests[0]["messages"][1]["content"])
        self.assertNotIn("888", requests[0]["messages"][1]["content"])
        valid = json.dumps({"excess_return_5d": 0.01, "evidence": ["Today's title"],
                            "reason": "Tentative assessment."})
        self.assertEqual(parse_response(valid, ["Today's title"])["excess_return_5d"], 0.01)
        with self.assertRaisesRegex(ValueError, "Evidência"):
            parse_response(valid, ["Other title"])
        with self.assertRaisesRegex(ValueError, "Previsão"):
            parse_response('{"excess_return_5d": true, "evidence": [], "reason": "unknown"}', [])
        self.assertEqual(requests[1]["context"]["asset_past_20_daily_returns"][-1], 999.0)


if __name__ == "__main__":
    unittest.main()
