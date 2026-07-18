from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def hma(series: pd.Series, period: int) -> pd.Series:
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    wma_half = 2 * series.rolling(half, min_periods=half).mean()
    wma_full = series.rolling(period, min_periods=period).mean()
    diff = wma_half - wma_full
    return diff.rolling(sqrt_p, min_periods=sqrt_p).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def atr(high, low, close, period=14):
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high, low, close, period=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = atr(high, low, close, period)
    atr_val = tr.where(tr > 0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_val, plus_di, minus_di


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series - series.shift(period)


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


def bollinger_bands(series: pd.Series, period=20, num_std=2):
    middle = sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle.replace(0, np.nan) * 100
    return upper, middle, lower, bandwidth


def rolling_volatility(series: pd.Series, period=20, ann_factor=252):
    ret = series.pct_change()
    return ret.rolling(period, min_periods=period).std() * np.sqrt(ann_factor)


def parkinson_volatility(high, low, period=20, ann_factor=252):
    log_hl = np.log(high / low)
    k = 1 / (4 * np.log(2))
    return np.sqrt(k * (log_hl ** 2).rolling(period, min_periods=period).mean()) * np.sqrt(ann_factor)


def garman_klass_volatility(open_, high, low, close, period=20, ann_factor=252):
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    return np.sqrt(
        (0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
        .rolling(period, min_periods=period)
        .mean()
    ) * np.sqrt(ann_factor)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["close"]
    high = result.get("high", close)
    low = result.get("low", close)
    open_ = result.get("open", close)

    result["sma_20"] = sma(close, 20)
    result["sma_50"] = sma(close, 50)
    result["sma_200"] = sma(close, 200)
    result["ema_20"] = ema(close, 20)
    result["rsi_14"] = rsi(close, 14)
    result["atr_14"] = atr(high, low, close, 14)
    result["adx_14"], result["plus_di"], result["minus_di"] = adx(high, low, close, 14)
    result["macd"], result["macd_signal"], result["macd_hist"] = macd(close)
    result["mom_10"] = momentum(close, 10)
    result["roc_10"] = roc(close, 10)
    _, _, _, result["bb_bandwidth"] = bollinger_bands(close, 20)
    result["vol_20"] = rolling_volatility(close, 20)
    result["parkinson_vol"] = parkinson_volatility(high, low, 20)
    result["gk_vol"] = garman_klass_volatility(open_, high, low, close, 20)

    return result
