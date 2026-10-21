import unittest

from src.fusion.llm_costs import _price_usage


class LLMCostTest(unittest.TestCase):
    def test_standard_luna_price_counts_cached_input_and_reasoning_output(self):
        response = {"response_model": "gpt-6-luna", "service_tier": "default",
                    "usage": {"prompt_tokens": 2000, "completion_tokens": 300,
                              "prompt_tokens_details": {"cached_tokens": 500},
                              "completion_tokens_details": {"reasoning_tokens": 80}}}
        row = _price_usage(response)
        self.assertEqual(row["status"], "standard_list_estimate")
        self.assertEqual(row["reasoning_tokens"], 80)
        self.assertAlmostEqual(row["list_usd"], (1500 * .10 + 500 * .01 + 300 * .50) / 1_000_000)
        response["usage"]["prompt_tokens_details"] = {}
        self.assertEqual(_price_usage(response)["status"], "cached_token_count_unavailable")
        self.assertIsNone(_price_usage(response)["list_usd"])
        response["usage"]["prompt_tokens_details"] = {"cached_tokens": 0}
        response["service_tier"] = "priority"
        self.assertEqual(_price_usage(response)["status"], "price_unavailable_for_model_or_tier")
        response["service_tier"] = "default"
        response["usage"]["prompt_tokens"] = 272001
        self.assertAlmostEqual(_price_usage(response)["list_usd"], (272001 * .20 + 300 * .75) / 1_000_000)
        self.assertIsNone(_price_usage({"response_model": "gpt-6-luna"})["list_usd"])


if __name__ == "__main__":
    unittest.main()
