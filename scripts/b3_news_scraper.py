import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests

GNEWS_BASE = "https://news.google.com/rss/search"
GNEWS_DELAY = 2.5
OUT_DIR = Path("data/news_b3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gnews_rss_url(query):
    return f"{GNEWS_BASE}?q={quote_plus(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"


def parse_date(entry):
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        dt = datetime(*entry.published_parsed[:6])
        return dt.strftime("%Y-%m-%d")
    if hasattr(entry, "published"):
        for fmt in ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ"]:
            try:
                return datetime.strptime(entry.published, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return ""


def extract_title_source(raw_title):
    parts = raw_title.rsplit(" - ", 1)
    title = parts[0].strip()
    source = parts[1].strip() if len(parts) > 1 else ""
    return title, source


def fetch_window(ticker, name, after, before):
    query = f"{name} OR {ticker} after:{after} before:{before}"
    url = gnews_rss_url(query)
    records = []
    try:
        time.sleep(GNEWS_DELAY)
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            dt = parse_date(entry)
            if not dt:
                continue
            title, source = extract_title_source(entry.get("title", ""))
            records.append(
                {
                    "ticker": ticker,
                    "date": dt,
                    "title": title,
                    "source": source,
                    "url": entry.get("link", ""),
                }
            )
    except Exception as e:
        print(f"  [!] {ticker} {after}->{before}: {e}")
    return records


def month_windows(start, end):
    windows = []
    current = end
    while current > start:
        w_end = current
        w_start = max(start, current - timedelta(days=30))
        windows.append((w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        current = w_start
    return windows


def scrape_ticker(ticker, name, start, end):
    print(f"\n[{ticker}] {name} | {start} -> {end}")
    windows = month_windows(
        datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
    )
    all_records = []
    for w_start, w_end in windows:
        records = fetch_window(ticker, name, w_start, w_end)
        all_records.extend(records)
        if records:
            print(f"  {w_start} -> {w_end}: {len(records)} artigos")
    if all_records:
        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=["date", "title"])
        df = df.sort_values("date")
        out_path = OUT_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f"  [OK] {len(df)} artigos salvos -> {out_path}")
    else:
        print(f"  [--] Nenhum artigo encontrado")


def main():
    parser = argparse.ArgumentParser(description="B3 Google News RSS Scraper")
    parser.add_argument(
        "--tickers", nargs="+", default=["PETR4", "VALE3", "ITUB4"]
    )
    parser.add_argument("--names", nargs="+", default=["Petrobras", "Vale", "Itau"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2020-02-01")
    args = parser.parse_args()

    for ticker, name in zip(args.tickers, args.names):
        scrape_ticker(ticker, name, args.start, args.end)

    print(f"\nDone. Files in {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
