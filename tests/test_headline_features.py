"""CPU-only checks for title-derived, conservative news-token facts."""

import unittest

import numpy as np

from src.fusion.headline_features import FEATURE_NAMES, extract_headline_features


class HeadlineFeaturesTest(unittest.TestCase):
    def facts(self, title):
        values = extract_headline_features(title)
        self.assertEqual(values.shape, (len(FEATURE_NAMES),))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(np.abs(values) <= 1))
        self.assertEqual(values[:6].sum(), 1)
        self.assertTrue(np.array_equal(values, extract_headline_features(title)))
        return dict(zip(FEATURE_NAMES, values))

    def test_announcement_categories_and_actual_vs_preview(self):
        cases = {
            "American Airlines Group Inc Q4 adjusted earnings of $1.15 per share": "event_earnings_guidance",
            "Issues FY20 Earnings Guidance - Quick Facts": "event_earnings_guidance",
            "B Co. bought by A Corp. for $20 billion": "event_ma",
            "FTC opens antitrust investigation of A": "event_legal_regulatory",
            "Analyst downgrades A to sell": "event_rating",
            "A unveils new product": "event_product",
            "3 Big Stock Charts for Friday: Square, American Airlines, and Kinder Morgan": "event_other",
            "Q4 19 Earnings Conference Call At 8:30 AM ET": "event_other",
            "Q4 2019 Earnings Call Transcript": "event_other",
            "Ex-Div Reminder": "event_other",
            "American Airlines to report earnings next week": "event_other",
        }
        for title, event in cases.items():
            with self.subTest(title=title):
                facts = self.facts(title)
                self.assertEqual(facts[event], 1)
                if "call" in title.lower() or "transcript" in title.lower():
                    self.assertFalse(any(facts[k] for k in FEATURE_NAMES[6:14]))
        actual_eps = self.facts("American Airlines Q4 adjusted earnings of $1.15 per share")
        self.assertEqual(actual_eps["earnings_beat"], 0)  # EPS != consensus surprise.

    def test_explicit_cues_case_punctuation_and_negation(self):
        self.assertEqual(self.facts("ACME: EPS BEATS analyst ESTIMATES!")["earnings_beat"], 1)
        self.assertEqual(self.facts("Acme EPS misses consensus estimates.")["earnings_miss"], 1)
        self.assertEqual(self.facts("Acme raises its FY20 earnings guidance")["guidance_raise"], 1)
        self.assertEqual(self.facts("Acme CUTS revenue outlook")["guidance_cut"], 1)
        self.assertEqual(self.facts("Analyst upgrades Acme stock")["rating_upgrade"], 1)
        self.assertEqual(self.facts("Acme rating downgraded to sell")["rating_downgrade"], 1)
        for title, cue in (("Acme did not beat earnings estimates", "earnings_beat"),
                           ("Analyst does not upgrade Acme", "rating_upgrade"),
                           ("Acme does not cut its guidance", "guidance_cut")):
            with self.subTest(title=title):
                self.assertEqual(self.facts(title)[cue], 0)

    def test_mixed_roles_and_speculative_titles_have_no_direction(self):
        for title in (
            "Acme acquires Beta at a 30% premium",
            "Acme beats Beta in device sales",
            "Acme upgrades its software",
            "Acme EPS beats estimates, Beta cuts guidance",
            "Will American Airlines Continue to Struggle Over the Next 5 Years?",
            "Rumor: Acme may beat EPS estimates",
            "Acme revenue rose 12% but sales fell 8%",
        ):
            with self.subTest(title=title):
                facts = self.facts(title)
                self.assertFalse(any(facts[k] for k in FEATURE_NAMES[6:12]))
                if "revenue rose" in title:
                    self.assertEqual(facts["reported_pct_signed"], 0)
        speculative = self.facts("Update: Acme reportedly in talks to acquire Beta?")
        self.assertEqual((speculative["update"], speculative["rumor"], speculative["uncertainty"]),
                         (1, 1, 1))

    def test_percentage_is_observed_and_only_signed_with_clear_metric_change(self):
        for title, magnitude, signed in (
            ("Acme revenue rose 12.5%", .125, .125),
            ("Acme May revenue rose 6%", .06, .06),
            ("Acme profits FELL by 8 percent", .08, -.08),
            ("Acme EPS up .5%", .005, .005),
            ("Acme sales increased 120%", 1, 1),
            ("Acme stock rises 12% after earnings", .12, 0),
            ("Acme acquired 20% stake in Beta", .2, 0),
            ("Acme reported EPS of $1.15 per share", 0, 0),
        ):
            with self.subTest(title=title):
                facts = self.facts(title)
                self.assertAlmostEqual(float(facts["reported_pct_magnitude"]), magnitude, places=6)
                self.assertAlmostEqual(float(facts["reported_pct_signed"]), signed, places=6)

    def test_unknown_input_is_title_only_and_needs_no_history(self):
        self.assertEqual(self.facts("Unrelated meeting in Paris")["event_other"], 1)
        self.assertEqual(self.facts("")["event_other"], 1)
        # A manifest's mismatched Article body cannot be passed as a headline record.
        with self.assertRaises(TypeError):
            extract_headline_features({"text": "Acme reports EPS", "Article": "Beta beats estimates"})


if __name__ == "__main__":
    unittest.main()
