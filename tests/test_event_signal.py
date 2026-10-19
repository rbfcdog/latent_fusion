"""Verificação de CPU: publicação, excesso, purga e versão do artefato."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.event_data import export_events, load_event_panel, load_event_panel_file, save_event_panel


class EventSignalTest(unittest.TestCase):
    def test_weekend_lag_excess_dedup_purge_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("cache/moment", "prices", "text"):
                (root / directory).mkdir(parents=True)
            days = pd.bdate_range("2020-01-01", periods=160)
            metadata, embeddings = [], []
            for ticker in ("AAL", "BABA"):
                close = np.full(len(days), 200.0) if ticker == "BABA" else 100 + np.arange(len(days), dtype=float)
                pd.DataFrame({"Date": days, "Close": close}).to_csv(root / "prices" / f"{ticker.lower()}.csv", index=False)
                for j, day in enumerate(days):
                    metadata.append((ticker, day.strftime("%Y-%m-%d")))
                    embeddings.append([float(j), float(j % 7)])
            pd.DataFrame(metadata, columns=["ticker", "date"]).to_csv(
                root / "cache/moment/top50_moment_metadata.csv", index=False)
            np.save(root / "cache/moment/top50_moment_embeddings.npy", np.asarray(embeddings, dtype=np.float32))

            friday = "2020-01-10"
            sunday = "2020-01-12"
            articles = [
                {"Date": friday, "Stock_symbol": "aal", "Url": "https://example.com/american-airlines-plans-growth",
                 "Article_title": "American Airlines plans growth", "Article": "CORPO ERRADO"},
                {"Date": friday, "Stock_symbol": "aal", "Url": "https://example.com/american-airlines-plans-growth",
                 "Article_title": "American Airlines plans growth", "Article": "Outro corpo"},
                {"Date": "2020-01-10T23:30:00-05:00", "Stock_symbol": "aal",
                 "Url": "https://example.com/american-airlines-adds-routes",
                 "Article_title": "American Airlines adds routes", "Article": "TEXTO INCORRETO"},
                {"Date": sunday, "Stock_symbol": "aal", "Url": "https://example.com/american-airlines-posts-update",
                 "Article_title": "American Airlines posts update", "Article": "Corpo não pareado"},
                {"Date": sunday, "Stock_symbol": "aal", "Url": "https://example.com/other-topic",
                 "Article_title": "American Airlines posts earnings", "Article": "Corpo falso"},
                {"Date": "2024-01-03", "Stock_symbol": "aal", "Url": "https://example.com/american-airlines-outlook",
                 "Article_title": "American Airlines outlook", "Article": "sem backfill"},
            ]
            (root / "text/AAL.jsonl").write_text("\n".join(json.dumps(row) for row in articles) + "\n", encoding="utf-8")
            (root / "text/BABA.jsonl").write_text("", encoding="utf-8")
            panel = load_event_panel(root / "cache", root / "prices")
            news_path = root / "events.jsonl.gz"
            available = {ticker: array["dates"] for ticker, array in zip(panel["tickers"], panel["arrays"])}
            coverage = export_events(root / "text", panel["trading_dates"], news_path, available)
            with gzip.open(news_path, "rt", encoding="utf-8") as file:
                events = [json.loads(line) for line in file]
            self.assertEqual(len(events), 3)
            self.assertEqual([row["decision_date"] for row in events],
                             ["2020-01-13", "2020-01-14", "2020-01-14"])
            self.assertEqual([row["source"] for row in events], ["example.com"] * 3)
            self.assertEqual(events[0]["published_at"], "2020-01-10T00:00:00.000000Z")
            self.assertEqual(events[1]["published_at"], "2020-01-11T04:30:00.000000Z")
            self.assertTrue(all("CORPO" not in row["text"] and "corpo" not in row["text"] for row in events))
            self.assertEqual(int(coverage["counts"].sum()), 3)

            aaa = panel["arrays"][0]
            j = int(np.flatnonzero(aaa["dates"] == np.datetime64("2020-02-03"))[0])
            index = int(np.flatnonzero(days == pd.Timestamp("2020-02-03"))[0])
            expected = (100 + index + 5) / (100 + index) - 1
            self.assertAlmostEqual(float(aaa["fwd_return"][j]), expected, places=6)
            self.assertAlmostEqual(float(aaa["market_fwd_return"][j]), expected / 2, places=6)
            self.assertAlmostEqual(float(aaa["target"][j]), expected / 2, places=6)
            self.assertEqual(aaa["target_end"][j], days[index + 5].to_datetime64())
            self.assertTrue(np.isnan(aaa["target"][-1]))
            self.assertTrue(np.isnat(aaa["target_end"][-1]))
            self.assertTrue(np.isfinite(aaa["known_ret"][-1]))
            for split, bound in (("train", panel["val_end"]), ("val", panel["train_end"])):
                self.assertTrue(all(panel["arrays"][ti]["target_end"][row] <= bound
                                    for ti, row in panel["splits"][split]))
                crossing = np.flatnonzero((aaa["dates"] <= bound) & (aaa["target_end"] > bound))
                self.assertGreater(len(crossing), 0)
                self.assertTrue(all((0, int(row)) not in panel["splits"][split] for row in crossing))
            self.assertTrue(all(panel["arrays"][ti]["target_end"][row] <= np.datetime64("2023-11-30")
                                for ti, row in panel["splits"]["test"]))
            save_event_panel(panel, root / "panel.npz", coverage)
            restored = load_event_panel_file(root / "panel.npz")
            self.assertEqual(len(restored["splits"]["test"]), len(panel["splits"]["test"]))
            self.assertEqual(restored["arrays"][0]["mom"].shape[1], 2)
            np.savez(root / "obsolete.npz", version=np.array(999))
            with self.assertRaisesRegex(ValueError, "obsoleto"):
                load_event_panel_file(root / "obsolete.npz")


    def test_rebalance_uses_future_returns_and_buy_hold_same_dates(self):
        from src.fusion.event_experiment import _portfolio, _portfolio_metrics

        dates = pd.bdate_range("2020-03-02", periods=20)
        daily = np.full(len(dates), 0.01, dtype=np.float32)
        daily[0] = 0
        panel = {"tickers": ["AAL", "BABA"], "arrays": [
            {"dates": dates.to_numpy(), "known_ret": daily},
            {"dates": dates.to_numpy(), "known_ret": np.zeros_like(daily)},
        ]}
        signals = pd.DataFrame([
            {"date": day, "ticker": ticker, "score": value}
            for day in dates[:12] for ticker, value in (("AAL", 1.0), ("BABA", 0.0))
        ])
        curve = _portfolio(panel, signals, cost_bps=0)
        self.assertEqual(curve.index[0], dates[0])
        self.assertAlmostEqual(float(curve["strategy"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(curve["bh"].iloc[0]), 1.0)
        self.assertGreater(float(curve["strategy"].iloc[5]), float(curve["bh"].iloc[5]))
        self.assertAlmostEqual(float(curve["bh"].iloc[5]), (1.01 ** 5 + 1) / 2, places=6)
        self.assertGreater(float(curve["strategy"].iloc[-1]), float(curve["bh"].iloc[-1]))
        paid = _portfolio(panel, signals, cost_bps=10)
        self.assertAlmostEqual(float(paid["strategy"].iloc[0]), 0.999)
        self.assertAlmostEqual(float(paid["bh"].iloc[0]), 0.999)
        metrics = _portfolio_metrics(paid, 10)
        self.assertAlmostEqual(metrics["excess_pct"],
                               (float(paid["strategy"].iloc[-1]) - float(paid["bh"].iloc[-1])) * 100)

if __name__ == "__main__":
    unittest.main()
