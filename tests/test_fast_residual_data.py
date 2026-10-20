import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.fusion.fast_residual_experiment import load_saved_demo
from src.fusion.text_attention import MetadataDataset, TextEventDataset


class FastResidualDataTest(unittest.TestCase):
    def test_saved_checkpoint_and_news_must_share_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = np.busday_offset(np.datetime64("2022-01-03", "D"),
                                      np.arange(24)).astype("datetime64[ns]")
            metadata = {"version": 1, "horizon": 5, "seq_len": 10,
                        "tickers": np.array(["AAA"]), "bounds": dates[[12, 15]],
                        "source_note": "audited", "created_at_utc": "2026-10-20T00:00:00+00:00",
                        "samples_train": np.array([[0, 10]], np.int32),
                        "samples_val": np.array([[0, 13]], np.int32),
                        "samples_test": np.array([[0, 16]], np.int32)}
            for key, value in {"dates": dates, "target_end": dates,
                               "mom": np.ones((24, 4), np.float32),
                               "past_vol": np.ones(24, np.float32),
                               "known_ret": np.zeros(24, np.float32),
                               "target": np.zeros(24, np.float32),
                               "fwd_return": np.zeros(24, np.float32),
                               "market_fwd_return": np.zeros(24, np.float32)}.items():
                metadata[f"arr_0_{key}"] = value
            np.savez(root / "fast_demo_panel.npz", **metadata)
            np.savez(root / "fast_demo_events.npz", ticker=["AAA"],
                     decision_date=dates[[10]].astype("datetime64[D]"),
                     published_at=["2022-01-03T00:00:00Z"],
                     embedding=np.ones((1, 384), np.float32),
                     features=np.ones((1, 17), np.float32),
                     novelty=np.zeros(1, np.float32))
            provenance = {"created_at_utc": str(metadata["created_at_utc"]),
                          "assets": ["AAA"], "article_embeddings": 1}
            (root / "fast_demo_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            checkpoint = {"model": "ts_only", "provenance": provenance,
                          "d_ts": 4, "d_text": 403, "seed": 42, "state_dict": {}}
            torch.save(checkpoint, root / "ts_only.pt")
            panel, events, _, _ = load_saved_demo(root)
            news = TextEventDataset(panel, events, "val")
            only_metadata = MetadataDataset(news)
            self.assertEqual(only_metadata[0][3].shape, (8, 1))
            self.assertTrue(only_metadata[0][4].any())
            self.assertFalse(MetadataDataset(TextEventDataset(panel, events, "test"))[0][4].any())
            checkpoint["provenance"] = {**provenance, "created_at_utc": "other run"}
            torch.save(checkpoint, root / "ts_only.pt")
            with self.assertRaisesRegex(ValueError, "mesma execução"):
                load_saved_demo(root)


if __name__ == "__main__":
    unittest.main()
