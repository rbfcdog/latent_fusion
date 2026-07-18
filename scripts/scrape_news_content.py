import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import html2text
import pandas as pd

GNEWS_BASE = "https://news.google.com/rss/search"
OUT_DIR = Path("data/news_b3_content")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DELAY = 2.0

H = html2text.HTML2Text()
H.ignore_images = True
H.body_width = 0
H.skip_internal_links = True

B3_NAMES = {
    "ABEV": "Ambev", "AFYA": "Afya", "AXIA": "AXIA", "BAK": "Braskem",
    "BBD": "Bradesco", "BSBR": "Santander Brasil", "CIG": "CEMIG", "CINT": "CI&T",
    "CSAN": "Cosan", "ELPC": "Copel", "EMBJ": "Embraer", "GGB": "Gerdau",
    "ITUB": "Itau", "NU": "Nubank", "PAGS": "PagSeguro", "PBR": "Petrobras",
    "PBR.A": "Petrobras", "SBS": "Sabesp", "SGML": "Sigma Lithium", "SID": "CSN",
    "SUZ": "Suzano", "TIMB": "TIM", "UGP": "Ultrapar", "VALE": "Vale",
    "VIV": "Vivo", "XP": "XP",
}


def parse_date(entry):
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
    return ""


def extract_summary_md(entry):
    summary = entry.get("summary", "")
    if not summary:
        return None
    text = H.handle(summary)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    body = "\n".join(lines)
    return body if len(body) > 60 else None


def extract_source(entry):
    title = entry.get("title", "")
    parts = title.rsplit(" - ", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def month_windows(start, end):
    windows = []
    current = end
    while current > start:
        w_end = current
        w_start = max(start, current - timedelta(days=30))
        windows.append((w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        current = w_start
    return windows


def scrape_ticker(ticker, name):
    out_path = OUT_DIR / f"{ticker}.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if "content" in existing.columns and existing["has_content"].sum() > 0:
            n = existing["has_content"].sum()
            print(f"[{ticker}] SKIP — {n}/{len(existing)} have content")
            return

    end = datetime.now()
    start = end - timedelta(days=3 * 365)
    windows = month_windows(start, end)

    print(f"[{ticker}] {name} | {len(windows)} windows")
    records = []

    for w_start, w_end in windows:
        query = f"{name} OR {ticker} after:{w_start} before:{w_end}"
        url = f"{GNEWS_BASE}?q={quote_plus(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        time.sleep(DELAY)

        try:
            parsed = feedparser.parse(url)
        except Exception:
            continue

        for entry in parsed.entries:
            dt = parse_date(entry)
            title = entry.get("title", "")
            content = extract_summary_md(entry)
            records.append({
                "ticker": ticker,
                "date": dt,
                "title": title,
                "source": extract_source(entry),
                "url": entry.get("link", ""),
                "content": content,
            })

    if not records:
        print(f"  [--] No articles found")
        return

    df = pd.DataFrame(records)
    df["has_content"] = df["content"].notna()
    df = df.drop_duplicates(subset=["date", "title"])
    df = df.sort_values("date")
    df.to_csv(out_path, index=False)

    n_content = df["has_content"].sum()
    print(f"  [OK] {len(df)} articles ({n_content} with content) -> {out_path}")


def main():
    tickers = sorted(B3_NAMES.keys())
    print(f"Processing {len(tickers)} tickers...")
    for ticker in tickers:
        try:
            scrape_ticker(ticker, B3_NAMES[ticker])
        except Exception as e:
            print(f"[{ticker}] FAILED: {e}")

    total = 0
    total_all = 0
    for f in OUT_DIR.glob("*.csv"):
        d = pd.read_csv(f)
        total += d.get("has_content", pd.Series(0)).sum()
        total_all += len(d)
    print(f"\nDone. {total}/{total_all} articles with content in {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
