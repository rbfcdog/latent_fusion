from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"


def _safe_get(url: str, params: dict | None = None, retries: int = 3, timeout: int = 20) -> Any:
    last_exc = None
    for _ in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(0.5)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("HTTP failed")


@dataclass
class CryptoDataConfig:
    symbol: str = "BTCUSDT"
    ccxt_symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    start_iso: str = "2025-01-01T00:00:00Z"
    oi_period: str = "5m"


def fetch_funding_rate(symbol: str = "BTCUSDT", start_ms: int | None = None) -> pd.DataFrame:
    url = f"{BINANCE_FUTURES}/fapi/v1/fundingRate"
    rows: list[dict] = []
    cursor = start_ms or 0

    while True:
        params = {"symbol": symbol, "startTime": cursor, "limit": 1000}
        data = _safe_get(url, params=params)
        if not data:
            break
        rows.extend(data)
        cursor = int(data[-1]["fundingTime"]) + 1
        if len(data) < 1000:
            break
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["fundingTime", "fundingRate"]].dropna().sort_values("fundingTime")


def fetch_open_interest_hist(symbol: str = "BTCUSDT", period: str = "5m", limit: int = 500) -> pd.DataFrame:
    url = f"{BINANCE_FUTURES}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    data = _safe_get(url, params=params)
    if not data:
        return pd.DataFrame(columns=["oi_time", "open_interest"])

    df = pd.DataFrame(data)
    df["oi_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    return df[["oi_time", "open_interest"]].dropna().sort_values("oi_time")


def fetch_recent_trades(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    url = f"{BINANCE_SPOT}/api/v3/aggTrades"
    params = {"symbol": symbol, "limit": limit}
    data = _safe_get(url, params=params)
    if not data:
        return pd.DataFrame(columns=["trade_time", "price", "quantity", "is_buyer_maker", "quote_qty"])

    df = pd.DataFrame(data)
    df["trade_time"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["q"], errors="coerce")
    df["is_buyer_maker"] = df["m"].astype(bool)
    df["quote_qty"] = df["price"] * df["quantity"]
    return df[["trade_time", "price", "quantity", "is_buyer_maker", "quote_qty"]].dropna()


def build_trade_features(trades: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "trade_count", "buy_qty", "sell_qty", "trade_delta"])

    trades = trades.copy()
    trades["timestamp"] = trades["trade_time"].dt.floor(timeframe)

    grouped = trades.groupby("timestamp", as_index=False).agg(
        trade_count=("quantity", "count"),
        total_qty=("quantity", "sum"),
    )

    buy = trades.loc[~trades["is_buyer_maker"]].groupby("timestamp")["quantity"].sum().reset_index(name="buy_qty")
    sell = trades.loc[trades["is_buyer_maker"]].groupby("timestamp")["quantity"].sum().reset_index(name="sell_qty")

    out = grouped.merge(buy, on="timestamp", how="left").merge(sell, on="timestamp", how="left")
    out[["buy_qty", "sell_qty"]] = out[["buy_qty", "sell_qty"]].fillna(0.0)
    out["trade_delta"] = out["buy_qty"] - out["sell_qty"]
    return out.sort_values("timestamp")


def build_volume_profile(trades: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["price_bin", "volume"])
    t = trades.copy()
    t["price_bin"] = pd.cut(t["price"], bins=bins)
    vp = t.groupby("price_bin", observed=False)["quantity"].sum().reset_index()
    vp = vp.rename(columns={"quantity": "volume"})
    return vp


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fundingRate" in out.columns:
        fr_mean = out["fundingRate"].rolling(50).mean()
        fr_std = out["fundingRate"].rolling(50).std()
        out["funding_z"] = (out["fundingRate"] - fr_mean) / (fr_std + 1e-8)

    if "open_interest" in out.columns:
        out["oi_change"] = out["open_interest"].pct_change().fillna(0.0)

    if "buy_qty" in out.columns and "sell_qty" in out.columns:
        total = out["buy_qty"] + out["sell_qty"] + 1e-8
        out["trade_imbalance"] = out["trade_delta"] / total

    if "close" in out.columns:
        out["returns"] = out["close"].pct_change()
        out["volatility"] = out["returns"].rolling(20).std()
        out["momentum"] = out["close"] / (out["close"].rolling(20).mean() + 1e-8)

    return out


def merge_crypto_datasets(
    price_df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
    oi_df: pd.DataFrame | None = None,
    trade_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = price_df.sort_values("timestamp") if "timestamp" in price_df.columns else price_df.sort_index()

    if funding_df is not None and not funding_df.empty:
        df = pd.merge_asof(
            df, funding_df.sort_values("fundingTime"),
            left_on="timestamp", right_on="fundingTime", direction="backward",
        )

    if oi_df is not None and not oi_df.empty:
        df = pd.merge_asof(
            df, oi_df.sort_values("oi_time"),
            left_on="timestamp", right_on="oi_time", direction="backward",
        )

    if trade_features_df is not None and not trade_features_df.empty:
        df = df.merge(trade_features_df, on="timestamp", how="left")

    ffill_cols = ["fundingRate", "open_interest", "buy_qty", "sell_qty", "trade_delta", "trade_count", "total_qty"]
    for col in ffill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    return add_microstructure_features(df)
