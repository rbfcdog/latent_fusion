#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import requests

TICKERS = {
    "ADBE": "Adobe",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "BABA": "Alibaba",
    "COST": "Costco",
    "CVX": "Chevron",
    "GILD": "Gilead Sciences",
    "GOOGL": "Alphabet Google",
    "JPM": "JPMorgan",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "NKE": "Nike",
    "NVDA": "Nvidia",
    "PYPL": "PayPal",
    "QCOM": "Qualcomm",
    "TSLA": "Tesla",
}
GNEWS_BASE = "https://news.google.com/rss/search"
GNEWS_DELAY = 2.5
OUT_DIR = Path("data/live_news")
RAW_DIR = OUT_DIR / "raw"
CAPTURE_PATH = OUT_DIR / "capture.jsonl"


def _clean_title(raw_title):
    parts = raw_title.rsplit(" - ", 1)
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "google_news")


def _published_iso(entry):
    raw = entry.get("published", "")
    try:
        dt = parsedate_to_datetime(raw)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def fetch_ticker(ticker, name, days, run_id):
    after = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f"{name} OR {ticker} after:{after}"
    url = f"{GNEWS_BASE}?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en"
    time.sleep(GNEWS_DELAY)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    raw_path = RAW_DIR / f"{ticker}_{run_id}.xml"
    raw_path.write_bytes(resp.content)
    parsed = feedparser.parse(resp.content)
    records = []
    for entry in parsed.entries:
        title, source = _clean_title(entry.get("title", ""))
        if not title or not entry.get("link"):
            continue
        records.append({
            "ticker": ticker,
            "title": title,
            "source": source,
            "url": entry.get("link", ""),
            "published_at": _published_iso(entry),
            "first_seen_at": f"{run_id}Z",
            "capture_run": run_id,
        })
    return records


def run(days):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    known = set()
    if CAPTURE_PATH.is_file():
        with CAPTURE_PATH.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    known.add((row.get("ticker"), row.get("url")))
                except json.JSONDecodeError:
                    continue
    summary = {}
    with CAPTURE_PATH.open("a", encoding="utf-8") as out:
        for ticker, name in TICKERS.items():
            records = fetch_ticker(ticker, name, days, run_id)
            fresh, seen = 0, set()
            for record in records:
                key = (record["ticker"], record["url"])
                if key in known or key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                fresh += 1
            known |= seen
            summary[ticker] = {"captured": len(records), "new_appended": fresh}
    print(json.dumps({"run": run_id, "days_window": days,
                      "file": str(CAPTURE_PATH), "by_ticker": summary},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append-only live news capture with first_seen_at")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    run(args.days)
