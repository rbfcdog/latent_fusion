from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def realized_volatility(close: pd.Series, window: int = 20, ann_factor: int = 252) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=window).std() * np.sqrt(ann_factor)


def hawkes_intensity(
    events: pd.Series | np.ndarray,
    alpha: float = 0.8,
    beta: float = 0.2,
    initial: float = 0.0,
) -> pd.Series:
    if isinstance(events, np.ndarray):
        events = pd.Series(events)
    intensity = pd.Series(0.0, index=events.index)
    lam = initial
    for i in range(len(events)):
        lam = np.exp(-beta) * lam + alpha * float(events.iloc[i])
        intensity.iloc[i] = lam
    return intensity


def hawkes_log_likelihood(
    events: np.ndarray,
    alpha: float,
    beta: float,
    mu: float = 0.01,
) -> float:
    n = len(events)
    if n < 2:
        return -1e9

    intensity = mu
    log_lik = 0.0
    for t in range(1, n):
        intensity = mu + np.exp(-beta) * (intensity - mu) + alpha * events[t - 1]
        if intensity <= 0:
            intensity = 1e-10
        if events[t] > 0:
            log_lik += np.log(intensity)
        log_lik -= intensity

    return float(log_lik)


def estimate_hawkes_params(events: np.ndarray) -> tuple[float, float, float]:
    def neg_ll(params):
        alpha, beta = params
        return -hawkes_log_likelihood(events, max(0.01, min(0.99, alpha)), max(0.01, min(0.99, beta)))

    result = minimize(neg_ll, [0.5, 0.3], bounds=[(0.01, 0.99), (0.01, 0.99)], method="L-BFGS-B")
    if result.success:
        return float(result.x[0]), float(result.x[1]), 0.01
    return 0.5, 0.3, 0.01


def volatility_regime_labels(
    close: pd.Series,
    vol_window: int = 20,
    n_regimes: int = 3,
) -> pd.Series:
    rv = realized_volatility(close, vol_window)
    rv_clean = rv.dropna()
    if len(rv_clean) < n_regimes * 3:
        return pd.Series(0, index=close.index)

    quants = np.percentile(rv_clean, np.linspace(0, 100, n_regimes + 1))
    labels = np.digitize(rv.values, quants[1:-1])
    return pd.Series(labels, index=rv.index).fillna(0).astype(int)


@dataclass
class VolatilityRegime:
    label: int
    name: str
    mean_vol: float
    mean_return: float
    frequency: float


def analyze_volatility_regimes(
    close: pd.Series,
    vol_window: int = 20,
    n_regimes: int = 3,
) -> list[VolatilityRegime]:
    labels = volatility_regime_labels(close, vol_window, n_regimes)
    rv = realized_volatility(close, vol_window)
    ret = close.pct_change()

    regime_names = {0: "low_vol", 1: "medium_vol", 2: "high_vol"}

    results = []
    for label_val in range(n_regimes):
        mask = labels == label_val
        freq = mask.mean()
        mean_vol = float(rv[mask].mean()) if mask.any() else 0.0
        mean_ret = float(ret[mask].mean()) if mask.any() else 0.0
        results.append(VolatilityRegime(
            label=label_val,
            name=regime_names.get(label_val, f"regime_{label_val}"),
            mean_vol=mean_vol,
            mean_return=mean_ret,
            frequency=freq,
        ))

    return results


def compute_embedding_drift(
    embeddings: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    n, d = embeddings.shape
    drift = np.zeros(n)
    prev = embeddings[0].copy()
    for i in range(1, n):
        current = embeddings[i]
        if np.all(current == fill_value) or np.all(prev == fill_value):
            drift[i] = 0.0
        else:
            drift[i] = float(np.linalg.norm(current - prev))
        prev = current
    return drift
