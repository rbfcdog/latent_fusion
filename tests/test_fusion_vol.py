import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.fusion.dataset import FusionDataset, load_panel, load_panel_file, save_panel
from src.fusion.models import build_model


class FusionVolatilityTest(unittest.TestCase):
    def test_forward_target_purged_splits_and_no_news(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for part in ("cache/text", "cache/moment", "prices"):
                (root / part).mkdir(parents=True)
            days = pd.bdate_range("2020-01-01", periods=320)
            text_rows, mom_rows, text_vectors, mom_vectors = [], [], [], []
            for ticker in ("AAA", "BBB"):
                step = 3 if ticker == "AAA" else 25
                log_returns = 0.001 + 0.01 * np.sin(np.arange(len(days)) / (9 if ticker == "AAA" else 13))
                close = 100 * np.exp(np.cumsum(log_returns))
                pd.DataFrame({"Date": days, "Close": close}).to_csv(root / "prices" / f"{ticker.lower()}.csv", index=False)
                for j, day in enumerate(days):
                    mom_rows.append((ticker, day.date().isoformat()))
                    mom_vectors.append([float(j % 17), float(j % 23), float(j % 7), float(j % 11)])
                    if j % step == 0 and not (ticker == "AAA" and j == 42):
                        text_rows.append((ticker, day.date().isoformat()))
                        text_vectors.append([float(j), float(j % 9), float(j % 5)])
                if ticker == "AAA":
                    text_rows.append((ticker, "2020-02-29"))
                    text_vectors.append([999.0, 5.0, 5.0])
            pd.DataFrame(text_rows, columns=["ticker", "date"]).to_csv(root / "cache/text/top50_daily_metadata.csv", index=False)
            np.save(root / "cache/text/top50_daily_embeddings.npy", np.asarray(text_vectors, dtype=np.float32))
            pd.DataFrame(mom_rows, columns=["ticker", "date"]).to_csv(root / "cache/moment/top50_moment_metadata.csv", index=False)
            np.save(root / "cache/moment/top50_moment_embeddings.npy", np.asarray(mom_vectors, dtype=np.float32))

            panel = load_panel(root / "cache", root / "prices")
            a = panel["arrays"][0]
            j = 40
            date = pd.Timestamp(a["dates"][j])
            day_index = days.get_loc(date)
            price = pd.read_csv(root / "prices/aaa.csv")["Close"].to_numpy()
            returns = np.diff(np.log(price))
            expected = np.sqrt(252 * np.mean(returns[day_index:day_index + 20] ** 2))
            self.assertAlmostEqual(float(a["target_vol"][j]), expected, places=6)
            self.assertEqual(a["target_end"][j], days[day_index + 20].to_datetime64())
            self.assertEqual(bool(a["text_mask"][j]), (day_index - 1) % 3 == 0)
            monday = np.flatnonzero(a["dates"] == days[43].to_datetime64())[0]
            tuesday = np.flatnonzero(a["dates"] == days[44].to_datetime64())[0]
            self.assertFalse(bool(a["text_mask"][monday]))
            self.assertTrue(bool(a["text_mask"][tuesday]))
            self.assertEqual(a["text"].shape[1], 7)
            self.assertTrue(np.isfinite(a["text"][a["text_mask"]]).all())
            for split in ("train", "val"):
                bound = panel["val_start"] if split == "train" else panel["train_end"]
                self.assertTrue(all(panel["arrays"][ti]["target_end"][row] <= bound
                                    for ti, row in panel["splits"][split]))
            path = root / "panel.npz"
            save_panel(panel, path)
            restored = load_panel_file(path)
            self.assertEqual(len(FusionDataset(restored, "test")), len(FusionDataset(panel, "test")))
            self.assertEqual(FusionDataset(restored, "test")[0][0].shape, (10, 4))

    def test_model_mask_and_persistence_fallback(self):
        model = build_model("fusion", d_ts=4, d_text=3).eval()
        z_ts = torch.randn(2, 10, 4)
        z_text = torch.randn(2, 10, 3)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[1, [1, 4]] = True
        past = torch.full((2,), 0.2)
        with torch.no_grad():
            pred, attn = model(z_ts, z_text, mask, past, return_attn=True)
            changed, _ = model(z_ts, z_text + 100, mask, past)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(attn["attn_ts_to_news"].shape, (2, 1, 11))
        self.assertTrue(torch.allclose(pred[0], changed[0]))
        self.assertAlmostEqual(float(attn["attn_ts_to_news"][0, 0, -1]), 1.0)
        text_only = build_model("text_only", d_text=3).eval()
        with torch.no_grad():
            text_pred, _ = text_only(z_ts, z_text, mask, past)
        self.assertTrue(torch.allclose(text_pred[0], past[0].log().reshape(1)))


if __name__ == "__main__":
    unittest.main()
