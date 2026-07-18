from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator
    return pd.Series(np.log(ratio.to_numpy(dtype=float)), index=ratio.index)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True,
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    if period < 2:
        return series.copy()
    half = max(period // 2, 1)
    sqrt_n = max(int(np.sqrt(period)), 1)
    raw_hma = 2 * wma(series, half) - wma(series, period)
    return wma(raw_hma, sqrt_n)


def vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    pv = close * volume
    return pv.rolling(period, min_periods=period).sum() / volume.rolling(
        period,
        min_periods=period,
    ).sum()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    k_smooth: int = 3,
    d_period: int = 3,
) -> pd.DataFrame:
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return tr_components.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def natr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return 100 * atr(high, low, close, period) / close.replace(0, np.nan)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = true_range(high, low, close)
    atr_smoothed = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smoothed.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smoothed.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line})


def awesome_oscillator(
    high: pd.Series,
    low: pd.Series,
    fast_period: int = 5,
    slow_period: int = 34,
) -> pd.Series:
    median_price = (high + low) / 2.0
    return sma(median_price, fast_period) - sma(median_price, slow_period)


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return close - close.shift(period)


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return 100 * (close / close.shift(period) - 1)


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    macd_line = ema(close, fast_period) - ema(close, slow_period)
    signal = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    hist = macd_line - signal
    return pd.DataFrame({"macd": macd_line, "signal": signal, "histogram": hist})


def stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> pd.DataFrame:
    rsi_series = rsi(close, rsi_period)
    low_rsi = rsi_series.rolling(stoch_period, min_periods=stoch_period).min()
    high_rsi = rsi_series.rolling(stoch_period, min_periods=stoch_period).max()
    stoch_rsi_raw = 100 * (rsi_series - low_rsi) / (high_rsi - low_rsi).replace(0, np.nan)
    fast_k = stoch_rsi_raw.rolling(k_period, min_periods=k_period).mean()
    fast_d = fast_k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_rsi_k": fast_k, "stoch_rsi_d": fast_d})


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def bull_bear_power(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 13,
) -> pd.DataFrame:
    baseline = ema(close, ema_period)
    bull = high - baseline
    bear = low - baseline
    return pd.DataFrame({"bull_power": bull, "bear_power": bear, "bull_bear_power": bull + bear})


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short: int = 7,
    medium: int = 14,
    long: int = 28,
) -> pd.Series:
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)

    avg_short = bp.rolling(short, min_periods=short).sum() / tr.rolling(short, min_periods=short).sum()
    avg_medium = bp.rolling(medium, min_periods=medium).sum() / tr.rolling(medium, min_periods=medium).sum()
    avg_long = bp.rolling(long, min_periods=long).sum() / tr.rolling(long, min_periods=long).sum()
    return 100 * (4 * avg_short + 2 * avg_medium + avg_long) / 7


def ichimoku_base_line(high: pd.Series, low: pd.Series, period: int = 26) -> pd.Series:
    return (high.rolling(period, min_periods=period).max() + low.rolling(period, min_periods=period).min()) / 2


def rolling_volatility(
    close: pd.Series,
    period: int = 20,
    annualization: int = 252,
    log_returns: bool = True,
) -> pd.Series:
    if log_returns:
        returns = _log_ratio(close, close.shift(1))
    else:
        returns = close.pct_change()
    return returns.rolling(period, min_periods=period).std() * np.sqrt(annualization)


def historical_volatility(
    close: pd.Series,
    annualization: int = 252,
    log_returns: bool = True,
) -> float:
    if log_returns:
        returns = _log_ratio(close, close.shift(1)).dropna()
    else:
        returns = close.pct_change().dropna()
    if len(returns) == 0:
        return float("nan")
    return float(returns.std() * np.sqrt(annualization))


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    mid = sma(close, period)
    stdev = close.rolling(period, min_periods=period).std()
    upper = mid + std_dev * stdev
    lower = mid - std_dev * stdev
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent_b": (close - lower) / (upper - lower).replace(0, np.nan),
        }
    )


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0,
) -> pd.DataFrame:
    center = ema(close, ema_period)
    band_atr = atr(high, low, close, atr_period)
    upper = center + atr_multiplier * band_atr
    lower = center - atr_multiplier * band_atr
    width = (upper - lower) / center.replace(0, np.nan)
    return pd.DataFrame(
        {
            "kc_center": center,
            "kc_upper": upper,
            "kc_lower": lower,
            "kc_width": width,
        }
    )


def donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    mid = (upper + lower) / 2
    width = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame(
        {
            "donchian_upper": upper,
            "donchian_lower": lower,
            "donchian_mid": mid,
            "donchian_width": width,
        }
    )


def chaikin_volatility(
    high: pd.Series,
    low: pd.Series,
    ema_period: int = 10,
    roc_period: int = 10,
) -> pd.Series:
    ema_range = ema(high - low, ema_period)
    return 100 * (ema_range / ema_range.shift(roc_period) - 1)


def ulcer_index(close: pd.Series, period: int = 14) -> pd.Series:
    rolling_max = close.rolling(period, min_periods=period).max()
    drawdown_pct = 100 * (close - rolling_max) / rolling_max.replace(0, np.nan)
    squared = drawdown_pct.pow(2)
    return squared.rolling(period, min_periods=period).mean().pow(0.5)


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
    annualization: int = 252,
) -> pd.Series:
    log_hl = _log_ratio(high, low)
    factor = 1 / (4 * np.log(2))
    variance = factor * (log_hl**2).rolling(period, min_periods=period).mean()
    return np.sqrt(variance * annualization)


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    annualization: int = 252,
) -> pd.Series:
    log_hl = _log_ratio(high, low)
    log_co = _log_ratio(close, open_)
    variance = (0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)).rolling(
        period,
        min_periods=period,
    ).mean()
    return np.sqrt(variance * annualization)


def rogers_satchell_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    annualization: int = 252,
) -> pd.Series:
    log_ho = _log_ratio(high, open_)
    log_lo = _log_ratio(low, open_)
    log_co = _log_ratio(close, open_)
    rs = (log_ho * (log_ho - log_co)) + (log_lo * (log_lo - log_co))
    variance = rs.rolling(period, min_periods=period).mean()
    return (variance * annualization).pow(0.5)


def yang_zhang_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    annualization: int = 252,
) -> pd.Series:
    prev_close = close.shift(1)

    log_oc = _log_ratio(open_, prev_close)
    log_co = _log_ratio(close, open_)
    log_ho = _log_ratio(high, open_)
    log_lo = _log_ratio(low, open_)

    rs = (log_ho * (log_ho - log_co)) + (log_lo * (log_lo - log_co))

    sigma_o2 = log_oc.rolling(period, min_periods=period).var()
    sigma_c2 = log_co.rolling(period, min_periods=period).var()
    sigma_rs = rs.rolling(period, min_periods=period).mean()

    k = 0.34 / (1.34 + (period + 1) / (period - 1))
    variance = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs
    return (variance * annualization).pow(0.5)


@dataclass(frozen=True)
class FibonacciLevels:
    swing_high: float
    swing_low: float
    retracement: dict[str, float]
    extension: dict[str, float]


def fibonacci_levels(swing_high: float, swing_low: float) -> FibonacciLevels:
    diff = swing_high - swing_low
    retracement = {
        "0.0": swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1.0": swing_low,
    }
    extension = {
        "1.272": swing_high + 0.272 * diff,
        "1.414": swing_high + 0.414 * diff,
        "1.618": swing_high + 0.618 * diff,
        "2.0": swing_high + 1.0 * diff,
        "2.618": swing_high + 1.618 * diff,
        "3.618": swing_high + 2.618 * diff,
    }
    return FibonacciLevels(
        swing_high=swing_high,
        swing_low=swing_low,
        retracement=retracement,
        extension=extension,
    )


def fibonacci_from_window(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 100,
) -> FibonacciLevels:
    window_high = float(high.tail(lookback).max())
    window_low = float(low.tail(lookback).min())
    return fibonacci_levels(window_high, window_low)


def indicator_action(
    value: float,
    buy_threshold: float,
    sell_threshold: float,
    reverse: bool = False,
) -> str:
    if np.isnan(value):
        return "Neutral"
    if reverse:
        if value <= buy_threshold:
            return "Buy"
        if value >= sell_threshold:
            return "Sell"
        return "Neutral"
    if value >= buy_threshold:
        return "Buy"
    if value <= sell_threshold:
        return "Sell"
    return "Neutral"


def summarize_core_indicators(
    df: pd.DataFrame,
    close_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    open_col: str = "Open",
    volume_col: str = "Volume",
) -> pd.DataFrame:
    close = df[close_col]
    high = df[high_col]
    low = df[low_col]
    open_ = df[open_col] if open_col in df.columns else close
    volume = df[volume_col] if volume_col in df.columns else pd.Series(1.0, index=df.index)

    stoch_df = stochastic(high, low, close, 14, 3, 3)
    adx_df = adx(high, low, close, 14)
    macd_df = macd(close, 12, 26, 9)
    stoch_rsi_df = stochastic_rsi(close, 14, 14, 3, 3)
    bbp_df = bull_bear_power(high, low, close, 13)

    indicators = {
        "rsi_14": rsi(close, 14),
        "stoch_k_14_3_3": stoch_df["stoch_k"],
        "cci_20": cci(high, low, close, 20),
        "adx_14": adx_df["adx"],
        "awesome_oscillator": awesome_oscillator(high, low),
        "momentum_10": momentum(close, 10),
        "macd_12_26": macd_df["macd"],
        "stoch_rsi_fast_k": stoch_rsi_df["stoch_rsi_k"],
        "williams_r_14": williams_r(high, low, close, 14),
        "bull_bear_power": bbp_df["bull_bear_power"],
        "ultimate_osc_7_14_28": ultimate_oscillator(high, low, close, 7, 14, 28),
        "ema_10": ema(close, 10),
        "sma_10": sma(close, 10),
        "ema_20": ema(close, 20),
        "sma_20": sma(close, 20),
        "ema_30": ema(close, 30),
        "sma_30": sma(close, 30),
        "ema_50": ema(close, 50),
        "sma_50": sma(close, 50),
        "ema_100": ema(close, 100),
        "sma_100": sma(close, 100),
        "ema_200": ema(close, 200),
        "sma_200": sma(close, 200),
        "ichimoku_base_26": ichimoku_base_line(high, low, 26),
        "vwma_20": vwma(close, volume, 20),
        "hma_9": hma(close, 9),
        "atr_14": atr(high, low, close, 14),
        "natr_14": natr(high, low, close, 14),
        "rolling_vol_20": rolling_volatility(close, 20, 252, True),
        "parkinson_vol_20": parkinson_volatility(high, low, 20, 252),
        "garman_klass_vol_20": garman_klass_volatility(open_, high, low, close, 20, 252),
        "rogers_satchell_vol_20": rogers_satchell_volatility(open_, high, low, close, 20, 252),
        "yang_zhang_vol_20": yang_zhang_volatility(open_, high, low, close, 20, 252),
        "chaikin_vol_10_10": chaikin_volatility(high, low, 10, 10),
        "ulcer_index_14": ulcer_index(close, 14),
    }

    out = pd.DataFrame(indicators)
    out["action_rsi_14"] = out["rsi_14"].apply(lambda v: indicator_action(v, 70, 30, reverse=True))
    out["action_stoch_k"] = out["stoch_k_14_3_3"].apply(lambda v: indicator_action(v, 80, 20, reverse=True))
    out["action_cci_20"] = out["cci_20"].apply(lambda v: indicator_action(v, 100, -100, reverse=True))
    out["action_adx_14"] = out["adx_14"].apply(lambda v: indicator_action(v, 25, 20, reverse=False))
    out["action_macd"] = out["macd_12_26"].apply(lambda v: indicator_action(v, 0, 0))
    out["action_momentum_10"] = out["momentum_10"].apply(lambda v: indicator_action(v, 0, 0))
    out["action_stoch_rsi_k"] = out["stoch_rsi_fast_k"].apply(
        lambda v: indicator_action(v, 80, 20, reverse=True)
    )
    out["action_williams_r_14"] = out["williams_r_14"].apply(
        lambda v: indicator_action(v, -20, -80, reverse=True)
    )
    out["action_bull_bear_power"] = out["bull_bear_power"].apply(lambda v: indicator_action(v, 0, 0))
    out["action_ultimate_osc"] = out["ultimate_osc_7_14_28"].apply(
        lambda v: indicator_action(v, 70, 30, reverse=True)
    )

    for n in [10, 20, 30, 50, 100, 200]:
        out[f"action_ema_{n}"] = np.where(close > out[f"ema_{n}"], "Buy", "Sell")
        out[f"action_sma_{n}"] = np.where(close > out[f"sma_{n}"], "Buy", "Sell")

    out["action_ichimoku_base_26"] = np.where(close > out["ichimoku_base_26"], "Buy", "Sell")
    out["action_vwma_20"] = np.where(close > out["vwma_20"], "Buy", "Sell")
    out["action_hma_9"] = np.where(close > out["hma_9"], "Buy", "Sell")
    return out