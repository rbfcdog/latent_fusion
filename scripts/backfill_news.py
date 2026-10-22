#!/usr/bin/env python3
import argparse
import csv
import json
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import requests

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fetch_live_news import TICKERS

GNEWS_BASE = "https://news.google.com/rss/search"
GNEWS_DELAY = 2.0
OUT_DIR = Path("data/live_news")
RAW_DIR = OUT_DIR / "raw" / "backfill"
CAPTURE_PATH = OUT_DIR / "capture.jsonl"
COVERAGE_PATH = OUT_DIR / "backfill_coverage.csv"


def _clean_title(raw_title):
    parts = raw_title.rsplit(" - ", 1)
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "google_news")


def _published_iso(entry):
    try:
        return parsedate_to_datetime(entry.get("published", "")).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _windows(start, end):
    current = start
    while current < end:
        w_end = min(current + timedelta(days=30), end)
        yield current.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")
        current = w_end


def fetch_window(ticker, name, after, before, run_id):
    query = f"{name} OR {ticker} after:{after} before:{before}"
    url = f"{GNEWS_BASE}?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en"
    time.sleep(GNEWS_DELAY)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:
        print(f"[!] {ticker} {after}->{before}: {exc}")
        return None, 1
    (RAW_DIR / f"{ticker}_{after}_{before}.xml").write_bytes(resp.content)
    records = []
    for entry in feedparser.parse(resp.content).entries:
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
    return records, 0


def run(start, end):
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
    coverage_rows = []
    with CAPTURE_PATH.open("a", encoding="utf-8") as out:
        for ticker, name in TICKERS.items():
            appended, failures, no_pub = 0, 0, 0
            seen = set()
            for after, before in _windows(start, end):
                records, failed = fetch_window(ticker, name, after, before, run_id)
                if records is None:
                    failures += 1
                    coverage_rows.append([ticker, after, -1])
                    continue
                coverage_rows.append([ticker, after, len(records)])
                for record in records:
                    key = (record["ticker"], record["url"])
                    if record["published_at"] is None:
                        no_pub += 1
                    if key in known or key in seen:
                        continue
                    seen.add(key)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    appended += 1
            known |= seen
            summary[ticker] = {"appended": appended, "window_failures": failures,
                               "no_published_at": no_pub}
            print(f"[{ticker}] +{appended} novas | {failures} falhas de janela | {no_pub} sem data")
    with COVERAGE_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ticker", "window_start", "records_returned"])
        writer.writerows(coverage_rows)
    print(json.dumps({"run": run_id, "start": start.strftime("%Y-%m-%d"),
                      "end": end.strftime("%Y-%m-%d"), "by_ticker": summary,
                      "file": str(CAPTURE_PATH), "coverage": str(COVERAGE_PATH)},
                     ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Google News RSS per monthly window")
    parser.add_argument("--start", default="2023-12-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc))
    run(start, end)
