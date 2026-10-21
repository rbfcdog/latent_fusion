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

    def test_responses_usage_includes_reasoning_and_cached_tokens(self):
        response = {"response_model": "gpt-6-luna", "service_tier": "default",
                    "usage": {"input_tokens": 1000, "output_tokens": 400,
                              "input_tokens_details": {"cached_tokens": 250},
                              "output_tokens_details": {"reasoning_tokens": 300}}}
        priced = _price_usage(response)
        self.assertEqual((priced["input_tokens"], priced["output_tokens"], priced["reasoning_tokens"]),
                         (1000, 400, 300))
        self.assertAlmostEqual(priced["list_usd"], (750 * .10 + 250 * .01 + 400 * .50) / 1_000_000)

    def test_responses_default_30m_cache_writes_are_not_double_billed(self):
        response = {"response_model": "gpt-6-luna", "service_tier": "default",
                    "prompt_cache_ttl": "30m",
                    "usage": {"input_tokens": 1000, "output_tokens": 100,
                              "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 600}}}
        priced = _price_usage(response)
        self.assertEqual(priced["status"], "standard_list_estimate")
        self.assertAlmostEqual(priced["list_usd"],
                               (200 * .10 + 200 * .01 + 600 * .125 + 100 * .50) / 1_000_000)
        response.pop("prompt_cache_ttl")
        self.assertEqual(_price_usage(response)["status"], "cache_write_duration_unavailable")


if __name__ == "__main__":
    unittest.main()
