import json
import unittest
from types import SimpleNamespace

import pandas as pd

from src.fusion.llm_tool_agent import _model_decision, _numeric_result


class NumericToolLoopTest(unittest.TestCase):
    def test_model_reasons_across_two_calls_without_future_forecasts(self):
        tickers = list("ABCDEFGH")
        date = pd.Timestamp("2023-02-01")
        tomorrow = pd.Timestamp("2023-02-02")
        numeric = pd.DataFrame([[.01] * 8, [.99] * 8], index=[date, tomorrow], columns=tickers)
        group = {ticker: {"asset_past_20_daily_returns": [.001] * 20,
                          "asset_trailing_annualized_volatility": .2} for ticker in tickers}
        context = {"basket": tickers, "decision_date": date.date().isoformat(), "held_before": [],
                   "assets": {ticker: {"available_headlines": []} for ticker in tickers}}
        context["assets"]["A"]["available_headlines"] = [{"title": 'Growth stock says "Yes"'}]
        context["assets"]["B"]["available_headlines"] = [{"title": "Audited title"}]
        tool = {"name": "analyze_numeric"}
        calls = [SimpleNamespace(type="function_call", name="analyze_numeric", call_id="one",
                                 arguments='{"tickers":["A"]}'),
                 SimpleNamespace(type="function_call", name="analyze_numeric", call_id="two",
                                 arguments='{"tickers":["B"]}')]
        raw = json.dumps({"actions": {ticker: "HOLD" for ticker in tickers},
                          "evidence": {ticker: [] for ticker in tickers}, "reason": "No trade."})
        replies = [SimpleNamespace(id=f"r{i}", model="gpt-6-luna", service_tier="default",
                                   usage=None, status="completed", output=[call], output_text="")
                   for i, call in enumerate(calls)]
        replies.append(SimpleNamespace(id="r2", model="gpt-6-luna", service_tier="default",
                                       usage=None, status="completed", output=[], output_text=raw))
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return replies[len(requests) - 1]

        model = SimpleNamespace(responses=SimpleNamespace(create=create))
        result, audit, response_ids = _model_decision(model, "gpt-6-luna", context, tool,
                                                       date, numeric, group, lambda *args: None)
        self.assertEqual(result, raw)
        self.assertEqual(response_ids, ["r0", "r1", "r2"])
        self.assertEqual([request["tool_choice"] for request in requests], ["required", "auto", "auto"])
        self.assertNotIn("previous_response_id", requests[0])
        evidence = requests[0]["text"]["format"]["schema"]["properties"]["evidence"]["properties"]
        self.assertEqual(evidence["A"], {"type": "array", "items": {"type": "string"}, "maxItems": 1})
        self.assertEqual(evidence["B"], {"type": "array",
                                         "items": {"type": "string", "enum": ["Audited title"]}, "maxItems": 1})
        self.assertEqual(evidence["C"]["maxItems"], 0)
        self.assertEqual([request["previous_response_id"] for request in requests[1:]], ["r0", "r1"])
        self.assertEqual([request["input"][0]["call_id"] for request in requests[1:]], ["one", "two"])
        self.assertEqual([call["result"]["assets"][ticker]["forecast_excess_5d_pct"]
                          for call, ticker in zip(audit, "AB")], [1.0, 1.0])
        self.assertNotIn("99.0", json.dumps(requests))
        with self.assertRaisesRegex(ValueError, "únicos"):
            _numeric_result('{"tickers":["A","A"]}', tickers, date, numeric, group)


if __name__ == "__main__":
    unittest.main()
