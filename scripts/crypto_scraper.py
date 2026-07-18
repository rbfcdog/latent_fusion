#!/usr/bin/env python3
import sys
import time
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests
import pandas as pd

warnings.filterwarnings("ignore")

GNEWS_BASE = "https://news.google.com/rss/search"
GNEWS_DELAY = 2.5

CRYPTO_ASSETS = [
    ("BTC-USD", "Bitcoin BTC crypto"),
    ("ETH-USD", "Ethereum ETH crypto"),
    ("SOL-USD", "Solana SOL crypto"),
    ("XRP-USD", "Ripple XRP crypto"),
    ("DOGE-USD", "Dogecoin DOGE crypto"),
    ("BNB-USD", "BNB Binance coin crypto"),
    ("ADA-USD", "Cardano ADA crypto"),
    ("AVAX-USD", "Avalanche AVAX crypto"),
    ("LINK-USD", "Chainlink LINK crypto"),
    ("DOT-USD", "Polkadot DOT crypto"),
]

OUTPUT_DIR = Path("data/news_crypto")


def _gnews_rss_url(query: str) -> str:
    return f"{GNEWS_BASE}?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en"


def _parse_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return date_str[:10] if len(date_str) >= 10 else date_str


def fetch_crypto_news(ticker: str, query: str, months: int = 12) -> list[dict]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    date_part = f"after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
    full_query = f"{query} {date_part}"
    url = _gnews_rss_url(full_query)

    time.sleep(GNEWS_DELAY)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        print(f"  [!] {ticker}: {e}")
        return []

    root = ET.fromstring(resp.content)
    records = []
    for item in root.findall(".//item"):
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        pub_date = item.findtext("pubDate", default="")
        source_elem = item.find("source")
        source = source_elem.text if source_elem is not None else "google_news"
        dt_str = _parse_date(pub_date)
        records.append({
            "ticker": ticker,
            "date": dt_str,
            "title": title,
            "source": source,
            "url": link,
        })

    return records


def run(months: int = 12, tickers: list[str] | None = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = CRYPTO_ASSETS
    if tickers:
        assets = [(t, q) for t, q in CRYPTO_ASSETS if t in tickers]

    total_records = 0
    for ticker, query in assets:
        print(f"Fetching {ticker}...")
        records = fetch_crypto_news(ticker, query, months)
        if records:
            df = pd.DataFrame(records)
            df = df.drop_duplicates(subset=["title", "date"])
            df = df.sort_values("date").reset_index(drop=True)
            out_path = OUTPUT_DIR / f"{ticker}.csv"
            df.to_csv(out_path, index=False)
            print(f"  {len(df)} articles -> {out_path}")
            total_records += len(df)
        else:
            print(f"  0 articles for {ticker}")

    print(f"\nTotal: {total_records} articles across {len(assets)} crypto assets")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Scrape crypto news from Google News RSS")
    p.add_argument("--months", type=int, default=12, help="Months of news to fetch")
    p.add_argument("--tickers", nargs="*", help="Specific tickers (default: all 10)")
    args = p.parse_args()
    run(months=args.months, tickers=args.tickers)
