"""CPU-only data-contract checks; no training, embeddings, or network access."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from src.fusion.new_period_news import TICKERS, build_new_period_package


class NewPeriodNewsTest(unittest.TestCase):
    @staticmethod
    def prices(root):
        dates = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                 "2024-01-08", "2024-01-09", "2024-01-10")
        folder = root / "prices"
        folder.mkdir()
        for ticker in TICKERS:
            with (folder / f"{ticker.lower()}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Date", "Close", "Dividends", "Stock Splits"))
                writer.writerows((day + " 00:00:00-05:00", 100 + ix, 0, 0)
                                 for ix, day in enumerate(dates))
        return folder

    def test_missing_feed_retains_full_price_calendar_but_never_claims_quiet_news(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prices = self.prices(root)
            out = root / "package"
            result = build_new_period_package(out, price_dir=prices,
                                              start="2024-01-02", end="2024-01-10")
            self.assertFalse(result["ready"])
            self.assertEqual(result["price_sessions"], 7)
            self.assertEqual(result["tickers"]["ADBE"]["feed_status"], "missing_or_unverified")
            self.assertIsNone(result["tickers"]["ADBE"]["no_news_days"])
            self.assertFalse((out / "articles.jsonl").exists())
            with (out / "prices.csv").open(newline="") as stream:
                daily = list(csv.DictReader(stream))
            self.assertEqual(len(daily), 7 * 8)
            self.assertEqual(daily[0]["known_return"], "")
            self.assertAlmostEqual(float(daily[1]["known_return"]), .01)
            self.assertNotIn("target", daily[0])
            with np.load(out / "new_period_prices.npz", allow_pickle=False) as panel:
                self.assertEqual(panel["tickers"].tolist(), list(TICKERS))
                self.assertEqual(panel["dates"].shape, (7,))
                self.assertEqual(panel["close"].shape, (8, 7))
                self.assertTrue(np.isnan(panel["known_return"][0, 0]))
                self.assertAlmostEqual(float(panel["known_return"][0, 1]), .01)
                self.assertNotIn("fwd_return", panel.files)

    def test_attested_feed_dedup_relevance_lag_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prices = self.prices(root)
            news = root / "news"
            news.mkdir()
            metadata = {"provider": "Licensed Example", "tickers": {}}
            titles = {"ADBE": "Adobe releases update", "BABA": "Alibaba releases update",
                      "COST": "Costco releases update", "CVX": "Chevron releases update",
                      "GILD": "Gilead releases update", "NKE": "Nike releases update",
                      "PYPL": "PayPal releases update", "QCOM": "Qualcomm releases update"}
            for ticker in TICKERS:
                metadata["tickers"][ticker] = {
                    "status": "complete", "start": "2024-01-02", "end": "2024-01-10",
                    "retrieved_at": "2024-01-11T00:00:00Z",
                    "request_url": f"https://provider.example/news/{ticker}"}
                article = {"ticker": ticker, "published_at": "2024-01-05T23:30:00Z",
                           "title": titles[ticker], "source": "Example Publisher",
                           "url": f"https://publisher.example/{ticker}"}
                extra = [{**article, "title": "Completely unrelated headline",
                          "url": "https://publisher.example/other"}] if ticker == "ADBE" else []
                (news / f"{ticker}.jsonl").write_text("\n".join(json.dumps(row) for row in
                    [article, article, *extra]) + "\n", encoding="utf-8")
            (news / "coverage.json").write_text(json.dumps(metadata), encoding="utf-8")
            out = root / "package"
            result = build_new_period_package(out, news, price_dir=prices,
                                              start="2024-01-02", end="2024-01-10")
            self.assertTrue(result["ready"])
            self.assertEqual(result["tickers"]["ADBE"]["filtered_irrelevant"], 1)
            self.assertEqual(result["tickers"]["ADBE"]["news_days"], 1)
            self.assertEqual(result["tickers"]["ADBE"]["no_news_days"], 6)
            articles = [json.loads(row) for row in (out / "articles.jsonl").read_text().splitlines()]
            self.assertEqual(len(articles), 8)
            self.assertEqual({row["decision_date"] for row in articles}, {"2024-01-09"})
            self.assertEqual(set(articles[0]),
                             {"ticker", "published_at", "source", "url", "title", "decision_date"})

    def test_unattested_or_malformed_news_is_never_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prices = self.prices(root)
            news = root / "news"
            news.mkdir()
            for ticker in TICKERS:
                (news / f"{ticker}.jsonl").write_text(json.dumps({
                    "ticker": ticker, "published_at": "2024-01-05", "title": "Adobe headline",
                    "source": "Publisher", "url": "https://publisher.example/item"}) + "\n")
            out = root / "package"
            result = build_new_period_package(out, news, price_dir=prices,
                                              start="2024-01-02", end="2024-01-10")
            self.assertFalse(result["ready"])
            self.assertFalse((out / "articles.jsonl").exists())
            metadata = {"provider": "Licensed Example", "tickers": {
                ticker: {"status": "complete", "start": "2024-01-02", "end": "2024-01-10",
                         "retrieved_at": "2024-01-11T00:00:00Z",
                         "request_url": f"https://provider.example/{ticker}"} for ticker in TICKERS}}
            (news / "coverage.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "explicit UTC offset"):
                build_new_period_package(out, news, price_dir=prices,
                                         start="2024-01-02", end="2024-01-10")
            self.assertFalse((out / "articles.jsonl").exists())
            for ticker in TICKERS:
                (news / f"{ticker}.jsonl").write_text(json.dumps({
                    "ticker": ticker, "published_at": "2024-01-05T10:00:00Z",
                    "title": "Adobe update" if ticker == "ADBE" else ticker + " update",
                    "source": "Publisher", "url": f"https://publisher.example/{ticker}"}) + "\n")
            (news / "BABA.jsonl").write_text("")
            result = build_new_period_package(out, news, price_dir=prices,
                                              start="2024-01-02", end="2024-01-10")
            self.assertFalse(result["ready"])
            self.assertEqual(result["tickers"]["BABA"]["feed_status"], "missing_or_unverified")
            self.assertIsNone(result["tickers"]["BABA"]["no_news_days"])
            self.assertFalse((out / "articles.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
