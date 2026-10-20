import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from src.fusion.event_experiment import _prediction_frame
from src.fusion.headline_features import FEATURE_NAMES
from src.fusion.presentation_controls import (
    _ContentOnlyDataset, _reuse_previous, run_controls, shuffle_headline_content,
)
from src.fusion.text_attention import TextEventDataset


class PresentationControlsTest(unittest.TestCase):
    def setUp(self):
        dates = np.busday_offset(np.datetime64("2023-01-02"), np.arange(16)).astype("datetime64[ns]")
        samples = np.array([[ti, j] for ti in range(2) for j in range(9, 16)], dtype=np.int32)
        self.panel = {
            "tickers": ["AAA", "BBB"], "seq_len": 10,
            "arrays": [{"dates": dates, "mom": np.zeros((16, 10, 3), np.float32),
                        "past_vol": np.ones(16, np.float32),
                        "known_ret": np.zeros(16, np.float32),
                        "target": np.zeros(16, np.float32),
                        "fwd_return": np.zeros(16, np.float32),
                        "market_fwd_return": np.zeros(16, np.float32)} for _ in range(2)],
            "splits": {split: samples for split in ("train", "val", "test")},
        }
        # Earliest date has one causally irreplaceable article; other dates
        # exercise same-date derangement and singletons with older donors.
        ids = np.array([0, 1, 2, 3, 4, 5])
        article_days = dates[[0, 2, 2, 4, 4, 8]].astype("datetime64[D]")
        self.events = {
            "ticker": np.array(["AAA", "AAA", "BBB", "AAA", "BBB", "AAA"]),
            "decision_date": article_days,
            "published_at": np.array([f"{day}T10:00:00Z" for day in article_days]),
            "embedding": np.stack((ids, ids + 10), axis=1).astype(np.float32),
            "features": np.stack([ids + n for n in range(len(FEATURE_NAMES))], axis=1).astype(np.float32),
            "novelty": ids.astype(np.float32),
        }

    def test_shuffling_preserves_availability_masks_ages_and_content_alignment(self):
        shuffled, mapping = shuffle_headline_content(self.events, seed=42)
        again, same_mapping = shuffle_headline_content(self.events, seed=42)
        self.assertTrue(mapping.equals(same_mapping))
        np.testing.assert_array_equal(shuffled["embedding"], again["embedding"])
        self.assertEqual(int(mapping.changed.sum()), len(mapping) - 1)
        self.assertFalse(bool(mapping.changed.iloc[0]))
        for row in mapping.itertuples():
            self.assertLessEqual(row.donor_decision_date, row.destination_decision_date)
            self.assertTrue(np.array_equal(shuffled["embedding"][row.destination_index],
                                           self.events["embedding"][row.donor_index]))
            self.assertTrue(np.array_equal(shuffled["features"][row.destination_index],
                                           self.events["features"][row.donor_index]))
            self.assertEqual(shuffled["novelty"][row.destination_index],
                             self.events["novelty"][row.donor_index])
        for field in ("ticker", "decision_date", "published_at"):
            np.testing.assert_array_equal(shuffled[field], self.events[field])
        original = TextEventDataset(self.panel, self.events, "test")
        control = TextEventDataset(self.panel, shuffled, "test")
        for i in range(len(original)):
            left, right = original[i], control[i]
            np.testing.assert_array_equal(left[4], right[4])  # masked event slots
            np.testing.assert_array_equal(left[3][:, -1], right[3][:, -1])  # trade-session age
            self.assertEqual(bool(left[4].any()), bool(right[4].any()))
            self.assertTrue(torch.equal(left[5], right[5]))
            np.testing.assert_array_equal(left[0], right[0])
        text, mask, target = _ContentOnlyDataset(original)[0]
        self.assertEqual(text.shape[-1], original.dim - 1)
        self.assertTrue(torch.equal(mask, original[0][4]))
        self.assertTrue(torch.equal(target, original[0][5]))
        self.assertEqual(len(_ContentOnlyDataset(original)[0]), 3)  # No price tensors.

    def test_cached_predictions_require_matching_provenance_scores_and_test_rows(self):
        original = TextEventDataset(self.panel, self.events, "test")
        mask = np.array([original[i][4].any() for i in range(len(original))], dtype=bool)
        base = np.full(len(original), 0.01, np.float32)
        provenance = {"audit": "example"}
        checkpoint = {"seed": 123}
        with tempfile.TemporaryDirectory() as dirname:
            folder = Path(dirname)
            (folder / "run_config.json").write_text(json.dumps({
                "embedding_provenance": provenance, "baseline_seed": 123,
                "residual_seed": 42, "max_residual": 0.02, "max_epochs": 8, "patience": 2,
            }), encoding="utf-8")
            for name, score in (("ts_only", base), ("metadata", base + .01),
                                ("text_residual", base + .02)):
                _prediction_frame(self.panel, original.samples, score, mask).to_csv(
                    folder / f"predictions_{name}.csv", index=False)
                if name != "ts_only":
                    torch.save({"model": name, "source_provenance": provenance,
                                "baseline_seed": 123, "state_dict": {}}, folder / f"{name}.pt")
                    pd.DataFrame({"epoch": [1], "val_mse": [.01]}).to_csv(
                        folder / f"{name}_history.csv", index=False)
            with patch("src.fusion.presentation_controls.EventResidual", return_value=MagicMock()), \
                    patch("src.fusion.presentation_controls._scores",
                          side_effect=[(base + .01, mask), (base + .02, mask)]):
                scores, history = _reuse_previous(folder, self.panel, {"test": original}, checkpoint,
                                                  provenance, MagicMock(), base, mask)
            self.assertEqual(set(scores), {"ts_only", "metadata", "text_residual"})
            self.assertEqual(set(history), {"metadata", "text_residual"})
            frame = pd.read_csv(folder / "predictions_metadata.csv")
            frame.loc[0, "ticker"] = "OTHER"
            frame.to_csv(folder / "predictions_metadata.csv", index=False)
            with patch("src.fusion.presentation_controls.EventResidual", return_value=MagicMock()), \
                    patch("src.fusion.presentation_controls._scores",
                          return_value=(base + .01, mask)):
                with self.assertRaisesRegex(ValueError, "Previsões anteriores incompatíveis"):
                    _reuse_previous(folder, self.panel, {"test": original}, checkpoint,
                                    provenance, MagicMock(), base, mask)

    def test_colab_cuda_guard_precedes_loading_or_forward(self):
        google = types.ModuleType("google")
        google.__path__ = []
        colab = types.ModuleType("google.colab")
        with patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                patch("src.fusion.presentation_controls.torch.cuda.is_available", return_value=False), \
                patch("src.fusion.presentation_controls.load_saved_demo") as load:
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                run_controls("unavailable", "unavailable")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
