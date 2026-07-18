import os
import time
from pathlib import Path

import pandas as pd
from lse import LSE

API_KEY = os.environ.get("LSE_API_KEY", "lse_live_e4f4842a647dd80491a0b65a25ece225")
OUT_DIR = Path("data/lse_market_data/b3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIMEFRAME = "1d"
SLEEP = 0.25


def fetch_symbols():
    client = LSE(api_key=API_KEY)
    catalog = client.catalog()
    return sorted(
        [
            item
            for item in catalog
            if item.get("dataset") == "stocks" and item.get("country") == "Brazil"
        ],
        key=lambda x: x["symbol"],
    )


def fetch_and_save(symbol, start="2020-01-01"):
    client = LSE(api_key=API_KEY)
    csv_path = OUT_DIR / f"{symbol}_1d.csv"
    pq_path = OUT_DIR / f"{symbol}_1d.parquet"

    try:
        candles = client.candles(
            symbol, TIMEFRAME, start=start, order="asc"
        )
        if not candles:
            return 0

        df = pd.DataFrame(candles)
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        if "time" in df.columns:
            df["Date"] = pd.to_datetime(df["time"])
            df = df.set_index("Date")
        df = df.sort_index()

        df.to_csv(csv_path)
        df.to_parquet(pq_path)
        return len(df)
    except Exception as e:
        print(f"  [!] {symbol}: {e}")
        return -1


def main():
    symbols = fetch_symbols()
    print(f"B3 stocks in LSE catalog: {len(symbols)}")
    existing = {p.stem.replace("_1d", "") for p in OUT_DIR.glob("*.csv")}
    missing = [(s["symbol"], s["name"]) for s in symbols if s["symbol"] not in existing]
    print(f"Already downloaded: {len(existing)}, Missing: {len(missing)}")

    if not missing:
        print("All B3 stocks already downloaded.")
        return

    for symbol, name in missing:
        print(f"\n[{symbol}] {name}")
        time.sleep(SLEEP)
        n = fetch_and_save(symbol)
        if n > 0:
            print(f"  [OK] {n} rows -> {OUT_DIR}/{symbol}_1d.csv")
        elif n == 0:
            print(f"  [--] No data returned")
        else:
            print(f"  [!!] Error")

    total = len(list(OUT_DIR.glob("*.csv")))
    print(f"\nDone. {total} B3 stocks in {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
