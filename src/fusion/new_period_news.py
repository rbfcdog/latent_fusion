"""Data-only new-period handoff; no model embeddings, predictions, or return labels.

build_new_period_package(out_dir, news_dir=None, price_dir="data/time_series",
                         start="2024-01-02", end="2025-03-28") writes:
  prices.csv       every session for all eight equities, not only news days;
  new_period_prices.npz  the same 8 x session closes, dates and trailing returns;
  coverage.json    price provenance, per-ticker news status and missing-feed gate;
  articles.jsonl   sourced titles with a one-additional-session decision lag,
                   ONLY when all eight raw feeds attest complete retrieval.

A provider import is a directory of ADBE.jsonl ... QCOM.jsonl and coverage.json:
{"provider":"Licensed Provider", "tickers":{
 "ADBE":{"status":"complete", "start":"2024-01-02", "end":"2025-03-28",
         "retrieved_at":"2025-04-01T00:00:00Z",
         "request_url":"https://provider.example/history?..."}, ...}}
Each JSONL row has ticker, published_at (timezone-aware ISO timestamp), source,
url (direct publisher/article URL), title. Input files and complete request-window
attestations MUST originate from a provider with rights to use the titles; the
function cannot independently certify licensing, publication time, or completeness.
news_dir is an operator attestation, NOT provider verification.
Missing/partial feeds yield a price-only package and ready=False; do not interpret
its empty article count as quiet news. Raw response errors must not be relabeled
"complete". This is not compatible with old panel embeddings/checkpoints.
"""

from bisect import bisect_left
import csv
from datetime import date, datetime, time, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import numpy as np

TICKERS = ("ADBE", "BABA", "COST", "CVX", "GILD", "NKE", "PYPL", "QCOM")
# Fixed issuer aliases from event_data; avoid importing its model-facing torch dependency.
ALIASES = {"ADBE": "adobe", "BABA": "alibaba", "COST": "costco",
           "CVX": "chevron", "GILD": "gilead", "NKE": "nike",
           "PYPL": "paypal", "QCOM": "qualcomm"}


def _digest(path):
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def _utc(value):
    if not isinstance(value, str) or not re.search(r"(?:Z|[+-]\d\d:\d\d)$", value):
        raise ValueError("published_at/retrieved_at must have an explicit UTC offset")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc


def _url(value):
    if not isinstance(value, str):
        raise ValueError("Article/request URL missing")
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError(f"Expected HTTPS URL with a host: {value!r}")
    return value


def _relevant(title, ticker):
    text = title.lower()
    alias = ALIASES[ticker]
    if re.search(r"(?<![a-z0-9])" + alias + r"(?![a-z0-9])", text):
        return True
    return bool(re.search(r"(?:\$|(?:NYSE|NASDAQ|AMEX)\s*:\s*)" + ticker + r"\b",
                          title, re.IGNORECASE))


def _prices(price_dir, start, end):
    """Read existing Yahoo-format Close; fail on absent sessions."""
    dates = None
    rows = []
    sources = {}
    for ticker in TICKERS:
        path = price_dir / f"{ticker.lower()}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"Date", "Close", "Dividends", "Stock Splits"} <= set(reader.fieldnames or ()):
                raise ValueError(f"Missing adjusted-close/source columns: {path}")
            selected = {}
            for row in reader:
                day = date.fromisoformat(row["Date"][:10])
                if start <= day <= end:
                    close = float(row["Close"])
                    if close <= 0 or not (close < float("inf")) or day in selected:
                        raise ValueError(f"Invalid or duplicate price for {ticker} on {day}")
                    selected[day] = close
        days = sorted(selected)
        if dates is None:
            dates = days
            if len(dates) < 2 or dates[0] != start or dates[-1] != end:
                raise ValueError("Price window must start and end on common trading sessions")
        elif days != dates:
            raise ValueError(f"Price sessions incomplete/misaligned for {ticker}")
        prev = None
        for day in days:
            close = selected[day]
            rows.append((day.isoformat(), ticker, close, "" if prev is None else close / prev - 1))
            prev = close
        sources[ticker] = {"path": str(path), "sha256": _digest(path), "sessions": len(days)}
    return dates, rows, sources


def build_new_period_package(out_dir, news_dir=None, *, price_dir="data/time_series",
                             start="2024-01-02", end="2025-03-28"):
    """Return coverage dict; ready=False means prices only, with missing feed explicit.

    No model work runs here. Reject malformed raw records rather than presenting
    corrupt metadata as genuine news. Complete feed attestations remain operator
    statements, not independently verified truth. Price returns are trailing only.
    """
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    if start < date(2023, 12, 1) or end <= start:
        raise ValueError("Need a fresh contiguous period strictly after 2023-11-30")
    out_dir, price_dir = Path(out_dir), Path(price_dir)
    days, price_rows, sources = _prices(price_dir, start, end)
    coverage = {"start": start.isoformat(), "end": end.isoformat(),
                "price_source": "existing Yahoo-format CSV Close (adjustment not independently certified)",
                "price_files": sources, "price_sessions": len(days), "tickers": {},
                "news_provider": None, "ready": False,
                "note": "Missing/partial feed is UNKNOWN, not a no-news day. No model embeddings or evaluation."}
    raw = {}
    if news_dir is not None:
        news_dir = Path(news_dir)
        if news_dir.resolve() == out_dir.resolve():
            raise ValueError("news_dir and out_dir must differ to preserve raw provider evidence")
        metadata_path = news_dir / "coverage.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("News coverage must be a JSON object")
            provider = metadata.get("provider")
            if not isinstance(provider, str) or not provider.strip():
                raise ValueError("News coverage needs a named provider")
            coverage["news_provider"] = provider.strip()
            coverage["news_coverage_sha256"] = _digest(metadata_path)
            raw = metadata.get("tickers", {})
            if not isinstance(raw, dict):
                raise ValueError("News coverage tickers must be an object")
    for ticker in TICKERS:
        path = None if news_dir is None else news_dir / f"{ticker}.jsonl"
        entry = raw.get(ticker, {})
        if not isinstance(entry, dict):
            raise ValueError(f"Bad feed coverage entry for {ticker}")
        complete = (path is not None and path.is_file() and entry.get("status") == "complete"
                    and entry.get("start") == start.isoformat() and entry.get("end") == end.isoformat()
                    and isinstance(entry.get("request_url"), str)
                    and isinstance(entry.get("retrieved_at"), str))
        coverage["tickers"][ticker] = {"feed_status": "complete" if complete else "missing_or_unverified",
                                        "article_count": None, "news_days": None,
                                        "no_news_days": None}
    if all(entry["feed_status"] == "complete" for entry in coverage["tickers"].values()):
        candidates = []
        for ticker in TICKERS:
            spec = raw[ticker]
            _url(spec["request_url"])
            retrieved_at = _utc(spec["retrieved_at"])
            source = news_dir / f"{ticker}.jsonl"
            coverage["tickers"][ticker]["raw_sha256"] = _digest(source)
            seen = set()
            filtered = 0
            late = 0
            with source.open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    try:
                        item = json.loads(line)
                        if not isinstance(item, dict) or item.get("ticker") != ticker:
                            raise ValueError("Wrong or absent ticker")
                        published = _utc(item.get("published_at"))
                        if published > retrieved_at:
                            raise ValueError("Article published after provider retrieval")
                        title = item.get("title")
                        publisher = item.get("source")
                        if not isinstance(title, str) or not title.strip() or not isinstance(publisher, str) or not publisher.strip():
                            raise ValueError("Article title/source missing")
                        url = _url(item.get("url"))
                        if not start <= published.date() <= end:
                            raise ValueError("Article outside attested request window")
                        if not _relevant(title, ticker):
                            filtered += 1
                            continue
                        if url in seen:
                            continue
                        seen.add(url)
                        # A headline after the US close first becomes tradable next session;
                        # then delay one additional full session regardless of publication time.
                        local = published.astimezone(ZoneInfo("America/New_York"))
                        first = bisect_left(days, local.date())
                        if first < len(days) and days[first] == local.date() and local.time() >= time(16):
                            first += 1
                        ix = first + 1
                        if ix >= len(days):
                            late += 1
                            continue
                        candidates.append({"ticker": ticker, "published_at": published.isoformat().replace("+00:00", "Z"),
                                           "source": publisher.strip(), "url": url,
                                           "title": " ".join(title.split()), "decision_date": days[ix].isoformat()})
                    except (ValueError, TypeError, KeyError) as exc:
                        raise ValueError(f"Invalid news row {source}:{line_no}: {exc}") from exc
            record = coverage["tickers"][ticker]
            record["filtered_irrelevant"] = filtered
            record["after_last_decision_session"] = late
        candidates.sort(key=lambda row: (row["decision_date"], row["ticker"], row["published_at"], row["url"]))
        for ticker in TICKERS:
            record = coverage["tickers"][ticker]
            subset = [row for row in candidates if row["ticker"] == ticker]
            if not subset:
                # An empty/unrelated raw file is not evidence that this issuer had no news.
                record["feed_status"] = "missing_or_unverified"
                record["reason"] = "no relevant articles in attested window"
                continue
            news_days = {row["decision_date"] for row in subset}
            record["article_count"] = len(subset)
            record["news_days"] = len(news_days)
            record["no_news_days"] = len(days) - len(news_days)
        coverage["ready"] = all(record["feed_status"] == "complete"
                                for record in coverage["tickers"].values())
        if coverage["ready"]:
            coverage["note"] = ("Complete window as ATTESTED by provider import, not independently verified. "
                                "No-news days depend on feed completeness; no model embeddings/evaluation.")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "prices.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("date", "ticker", "close", "known_return"))
        writer.writerows(price_rows)
    close = np.asarray([row[2] for row in price_rows], dtype=np.float64).reshape(len(TICKERS), len(days))
    known = np.full_like(close, np.nan)
    known[:, 1:] = close[:, 1:] / close[:, :-1] - 1
    np.savez_compressed(out_dir / "new_period_prices.npz",
                        tickers=np.asarray(TICKERS), dates=np.asarray(days, dtype="datetime64[ns]"),
                        close=close, known_return=known,
                        provenance=np.asarray(json.dumps({"start": start.isoformat(), "end": end.isoformat(),
                                                          "price_source": coverage["price_source"],
                                                          "price_files": sources}, sort_keys=True)))
    article_path = out_dir / "articles.jsonl"
    if coverage["ready"]:
        with article_path.open("w", encoding="utf-8") as stream:
            for row in candidates:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif article_path.exists():
        article_path.unlink()  # Never leave an older supposedly ready feed beside a missing-feed report.
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return coverage
