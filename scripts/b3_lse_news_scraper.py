import time
import argparse
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd

GNEWS_BASE = "https://news.google.com/rss/search"
GNEWS_DELAY = 2.5
MAX_WORKERS = 4

DATA_DIR = Path("data/lse_market_data/b3")
OUT_DIR = Path("data/news_b3_lse")
OUT_DIR.mkdir(parents=True, exist_ok=True)

B3_NAMES = {
    "ABEV": "Ambev",
    "AFYA": "Afya",
    "AXIA": "AXIA Energia",
    "BAK": "Braskem",
    "BBD": "Bradesco",
    "BSBR": "Santander Brasil",
    "CIG": "CEMIG",
    "CINT": "CI&T",
    "CSAN": "Cosan",
    "ELPC": "Copel",
    "EMBJ": "Embraer",
    "GGB": "Gerdau",
    "ITUB": "Itaú Unibanco",
    "NU": "Nubank",
    "PAGS": "PagSeguro",
    "PBR": "Petrobras",
    "PBR.A": "Petrobras",
    "SBS": "Sabesp",
    "SGML": "Sigma Lithium",
    "SID": "CSN Siderúrgica",
    "SUZ": "Suzano",
    "TIMB": "TIM",
    "UGP": "Ultrapar",
    "VALE": "Vale",
    "VIV": "Vivo Telefônica",
    "XP": "XP Investimentos",
}


def gnews_rss_url(query):
    return f"{GNEWS_BASE}?q={quote_plus(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"


def parse_date(entry):
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
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


def scrape_ticker(ticker):
    name = B3_NAMES.get(ticker, ticker)
    price_path = DATA_DIR / f"{ticker}_1d.csv"
    if not price_path.exists():
        print(f"[{ticker}] SKIP — no price data")
        return None

    try:
        prices = pd.read_csv(price_path)
        date_col = "timestamp" if "timestamp" in prices.columns else "Date"
        dates = pd.to_datetime(prices[date_col]).dropna()
        start = dates.min().strftime("%Y-%m-%d")
        end = dates.max().strftime("%Y-%m-%d")
    except Exception:
        print(f"[{ticker}] SKIP — failed to read price data")
        return None

    out_path = OUT_DIR / f"{ticker}.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if not existing.empty:
            ex_start = existing["date"].min()
            ex_end = existing["date"].max()
            if ex_start <= start and ex_end >= end:
                print(f"[{ticker}] SKIP — already complete ({len(existing)} artigos)")
                return len(existing)

    print(f"[{ticker}] {name} | {start} -> {end}")
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
        df.to_csv(out_path, index=False)
        print(f"  [OK] {len(df)} artigos -> {out_path}")
        return len(df)
    else:
        print(f"  [--] Nenhum artigo")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--tickers", nargs="*")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else sorted(
        p.stem.replace("_1d", "") for p in DATA_DIR.glob("*_1d.csv")
    )
    print(f"B3 tickers with price data: {len(tickers)}")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scrape_ticker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                print(f"[{ticker}] FAILED: {e}")

    total = sum(v for v in results.values() if v)
    print(f"\nDone. {total} total artigos em {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
