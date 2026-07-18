from __future__ import annotations

import numpy as np
import pandas as pd


def _log_windows(min_window: int, max_window: int, n_windows: int) -> np.ndarray:
    windows = np.logspace(np.log10(min_window), np.log10(max_window), num=n_windows).astype(int)
    windows = np.unique(windows)
    windows = windows[windows >= min_window]
    return windows


def hurst_rs(
    series: pd.Series,
    min_window: int = 4,
    max_window: int | None = None,
    n_windows: int = 20,
) -> float:
    values = np.asarray(series.dropna(), dtype=float)
    n = len(values)
    if n < min_window:
        return float("nan")
    if max_window is None:
        max_window = n // 2
    if max_window < min_window:
        return float("nan")
    windows = _log_windows(min_window, max_window, n_windows)
    rs_vals = []
    for w in windows:
        n_seg = n // w
        if n_seg < 1:
            continue
        usable = n_seg * w
        segs = values[:usable].reshape(n_seg, w)
        rs_seg = []
        for seg in segs:
            mean = np.mean(seg)
            cumdev = np.cumsum(seg - mean)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(seg, ddof=1)
            if s > 0:
                rs_seg.append(r / s)
        if rs_seg:
            rs_vals.append(np.mean(rs_seg))
        else:
            rs_vals.append(np.nan)
    rs_vals = np.asarray(rs_vals)
    valid = ~np.isnan(rs_vals) & (rs_vals > 0)
    if valid.sum() < 2:
        return float("nan")
    log_w = np.log(windows[: len(rs_vals)][valid])
    log_rs = np.log(rs_vals[valid])
    return float(np.polyfit(log_w, log_rs, 1)[0])


def hurst_dfa(
    series: pd.Series,
    min_window: int = 4,
    max_window: int | None = None,
    n_windows: int = 20,
    order: int = 1,
) -> float:
    values = np.asarray(series.dropna(), dtype=float)
    n = len(values)
    if n < min_window:
        return float("nan")
    if max_window is None:
        max_window = n // 2
    if max_window < min_window:
        return float("nan")
    windows = _log_windows(min_window, max_window, n_windows)
    f_vals = []
    for w in windows:
        n_seg = n // w
        if n_seg < 1:
            continue
        usable = n_seg * w
        segs = values[:usable].reshape(n_seg, w)
        f_seg = []
        for seg in segs:
            t = np.arange(len(seg))
            coeffs = np.polyfit(t, seg, order)
            trend = np.polyval(coeffs, t)
            resid = seg - trend
            f_seg.append(np.sqrt(np.mean(resid ** 2)))
        if f_seg:
            f_vals.append(np.mean(f_seg))
        else:
            f_vals.append(np.nan)
    f_vals = np.asarray(f_vals)
    valid = ~np.isnan(f_vals) & (f_vals > 0)
    if valid.sum() < 2:
        return float("nan")
    log_w = np.log(windows[: len(f_vals)][valid])
    log_f = np.log(f_vals[valid])
    return float(np.polyfit(log_w, log_f, 1)[0])


def fractal_dimension(series: pd.Series, method: str = "rs") -> float:
    if method == "rs":
        h = hurst_rs(series)
    elif method == "dfa":
        h = hurst_dfa(series)
    else:
        raise ValueError(f"unknown method: {method}")
    if np.isnan(h):
        return float("nan")
    return float(2.0 - h)


def compute_fractal_features(
    df: pd.DataFrame,
    col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [50, 100, 200]
    series = df[col]
    result = pd.DataFrame(index=df.index)
    for w in windows:
        hurst_rs_col = f"hurst_rs_{w}"
        hurst_dfa_col = f"hurst_dfa_{w}"
        fd_rs_col = f"fd_rs_{w}"
        fd_dfa_col = f"fd_dfa_{w}"
        h_rs = series.rolling(w, min_periods=w).apply(
            lambda s: hurst_rs(s, min_window=4, max_window=len(s) // 2, n_windows=20),
            raw=False,
        )
        h_dfa = series.rolling(w, min_periods=w).apply(
            lambda s: hurst_dfa(s, min_window=4, max_window=len(s) // 2, n_windows=20),
            raw=False,
        )
        result[hurst_rs_col] = h_rs
        result[hurst_dfa_col] = h_dfa
        result[fd_rs_col] = 2.0 - h_rs
        result[fd_dfa_col] = 2.0 - h_dfa
    return result
