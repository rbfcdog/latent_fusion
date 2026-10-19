"""CPU-only checks for title provenance and interrupted Colab extraction."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.fusion.llm_headlines import LLM_FEATURE_NAMES, _parse_response, extract_llm_features


class LLMHeadlineFactsTest(unittest.TestCase):
    @staticmethod
    def manifest(path, titles):
        with gzip.open(path, "wt", encoding="utf-8") as source:
            for title in titles:
                source.write(json.dumps({"ticker": "AAL", "published_at": "2020-01-10T00:00:00Z",
                                         "decision_date": "2020-01-13", "url": "https://example.com/x",
                                         "source": "example.com", "text": title}) + "\n")

    @staticmethod
    def answer(title):
        return json.dumps({"role": "subject", "class": "earnings", "direction": "positive",
                           "status": "confirmed", "magnitude_pct": 5,
                           "evidence": title})

    def test_strict_title_evidence_and_uncertain_issuer(self):
        title = "American Airlines earnings rose 5%"
        features, parsed, error = _parse_response(self.answer(title), title, "AAL")
        self.assertIsNone(error)
        self.assertEqual(parsed["evidence"], title)
        self.assertEqual(features[LLM_FEATURE_NAMES.index("magnitude_pct")], 0.05)
        capped = json.loads(self.answer("American Airlines earnings rose 200%"))
        capped["magnitude_pct"] = 200
        bounded, _, error = _parse_response(json.dumps(capped), capped["evidence"], "AAL")
        self.assertIsNone(error)
        self.assertEqual(bounded[-1], 1.0)
        fabricated = json.loads(self.answer(title))
        fabricated["evidence"] = "American Airlines earnings rose 9%"
        vector, _, error = _parse_response(json.dumps(fabricated), title, "AAL")
        self.assertTrue(error)
        self.assertEqual(vector, [0] * len(LLM_FEATURE_NAMES))
        fabricated["evidence"] = title
        fabricated["magnitude_pct"] = 9
        self.assertIn("percentage", _parse_response(json.dumps(fabricated), title, "AAL")[2])
        uncertain = {"role": "unknown", "class": "unknown", "direction": "unknown",
                     "status": "unknown", "magnitude_pct": None, "evidence": ""}
        self.assertIsNone(_parse_response(json.dumps(uncertain), "American Airlines and Visa merger", "AAL")[2])
        self.assertEqual(_parse_response(json.dumps(uncertain), title, "AAL")[0], vector)
        ambiguous = {"role": "subject", "class": "merger", "direction": "unknown",
                     "status": "confirmed", "magnitude_pct": None,
                     "evidence": "American Airlines and Visa merger"}
        self.assertIn("multiple named", _parse_response(json.dumps(ambiguous),
                      ambiguous["evidence"], "AAL")[2])
        self.assertTrue(_parse_response("```json\n" + self.answer(title) + "\n```", title, "AAL")[2])
        self.assertTrue(_parse_response(self.answer(title) + " extra", title, "AAL")[2])
        duplicate = self.answer(title).replace('"role": "subject",', '"role": "subject", "role": "object",')
        self.assertTrue(_parse_response(duplicate, title, "AAL")[2])
        stock = "American Airlines stock price rose 5%"
        stock_claim = json.loads(self.answer(stock))
        stock_claim["class"] = "unknown"
        self.assertIn("stock price", _parse_response(json.dumps(stock_claim), stock, "AAL")[2])

    def test_resume_partial_line_stale_manifest_and_audited_invalid(self):
        first, second = "American Airlines earnings rose 5%", "American Airlines earnings fell 6%"
        with tempfile.TemporaryDirectory() as tmp:
            manifest, cache_dir = Path(tmp) / "events.jsonl.gz", Path(tmp) / "drive"
            self.manifest(manifest, [first, second])
            calls = []

            def interrupted(rows, batch_size):
                calls.append([row["text"] for row in rows])
                yield [self.answer(first)]
                raise RuntimeError("interrupted Colab session")

            with patch("src.fusion.llm_headlines._require_colab_cuda"), patch(
                    "src.fusion.llm_headlines._generate_responses", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    extract_llm_features(manifest, cache_dir, batch_size=1)
            (cache_path,) = cache_dir.glob("*.jsonl")
            with cache_path.open("ab") as cache:
                cache.write(b'{"partial":')

            def resumed(rows, batch_size):
                calls.append([row["text"] for row in rows])
                yield ["NOT JSON"]

            with patch("src.fusion.llm_headlines._require_colab_cuda"), patch(
                    "src.fusion.llm_headlines._generate_responses", side_effect=resumed):
                result, resumed_path = extract_llm_features(manifest, cache_dir, batch_size=1)
            self.assertEqual(cache_path, resumed_path)
            self.assertEqual(calls, [[first, second], [second]])
            self.assertEqual(result["status"].tolist(), ["ok", "invalid"])
            self.assertEqual(result["features"].dtype, np.float32)
            self.assertEqual(result["features"][1].tolist(), [0] * len(LLM_FEATURE_NAMES))
            self.assertEqual(result["decision_date"].dtype, np.dtype("datetime64[D]"))
            records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["index"] for row in records], [0, 1])
            self.assertEqual(records[1]["raw"], "NOT JSON")
            self.assertTrue(records[1]["error"])
            with patch("src.fusion.llm_headlines._require_colab_cuda", side_effect=AssertionError("GPU accessed")):
                restored, _ = extract_llm_features(manifest, cache_dir)
            np.testing.assert_array_equal(restored["features"], result["features"])
            records[0]["model_revision"] = "stale"
            cache_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incompatible headline cache"):
                extract_llm_features(manifest, cache_dir)
            self.manifest(manifest, [first, "American Airlines earnings rose 7%"])
            with patch("src.fusion.llm_headlines._require_colab_cuda", side_effect=RuntimeError("GPU required")):
                with self.assertRaisesRegex(RuntimeError, "GPU required"):
                    extract_llm_features(manifest, cache_dir)
            self.assertEqual(len(list(cache_dir.glob("*.jsonl"))), 2)


if __name__ == "__main__":
    unittest.main()
