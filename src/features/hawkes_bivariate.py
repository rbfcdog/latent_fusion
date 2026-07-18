from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BivarHawkesFit:
    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    branching_ratio: np.ndarray
    log_likelihood: float


def _to_numeric_times(df: pd.DataFrame, timestamp_col: str) -> tuple[pd.DataFrame, float]:
    ts = pd.to_datetime(df[timestamp_col], errors="coerce", format="mixed")
    if pd.api.types.is_timedelta64_dtype(ts):
        ts = ts.dt.total_seconds()
    elif pd.api.types.is_datetime64_any_dtype(ts):
        t0 = ts.min()
        ts = (ts - t0).dt.total_seconds()
    else:
        ts = pd.to_numeric(ts, errors="coerce")
    out = df.copy()
    out["_t"] = ts.astype(float)
    out = out.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)
    out["_t"] = out["_t"] - out["_t"].iloc[0] if len(out) else out["_t"]
    t_end = float(out["_t"].iloc[-1]) if len(out) else 0.0
    return out, t_end


def extract_events(
    df: pd.DataFrame,
    price_col: str = "close",
    timestamp_col: str = "timestamp",
    jump_sigma: float = 3.0,
    liquidation_col: str | None = None,
) -> tuple[list[np.ndarray], float]:
    work, t_end = _to_numeric_times(df, timestamp_col)
    prices = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[price_col]).reset_index(drop=True)
    times = work["_t"].to_numpy(dtype=float)

    if liquidation_col is not None and liquidation_col in work.columns:
        liq_vals = pd.to_numeric(work[liquidation_col], errors="coerce").fillna(0.0)
        liq_mask = liq_vals.to_numpy() > 0.0
    else:
        log_price = np.log(prices.to_numpy(dtype=float))
        returns = np.diff(log_price, prepend=np.nan)
        ret_std = float(np.nanstd(returns))
        if ret_std <= 0.0 or np.isnan(ret_std):
            liq_mask = np.zeros(len(work), dtype=bool)
        else:
            threshold = jump_sigma * ret_std
            liq_mask = np.where(np.isnan(returns), False, returns < -threshold)

    liq_events = np.unique(times[liq_mask])

    log_price = np.log(prices.to_numpy(dtype=float))
    returns = np.diff(log_price, prepend=np.nan)
    ret_std = float(np.nanstd(returns))
    if ret_std <= 0.0 or np.isnan(ret_std):
        jump_mask = np.zeros(len(work), dtype=bool)
    else:
        threshold = jump_sigma * ret_std
        jump_mask = np.where(np.isnan(returns), False, returns > threshold)

    jump_events = np.unique(times[jump_mask])

    return [liq_events, jump_events], t_end


def _intensity_at(
    fit_mu: np.ndarray,
    fit_alpha: np.ndarray,
    fit_beta: np.ndarray,
    events: list[np.ndarray],
    t: float,
) -> np.ndarray:
    n = len(events)
    lam = np.array(fit_mu, dtype=float)
    for i in range(n):
        for j in range(n):
            tj = events[j]
            if len(tj) == 0:
                continue
            past = tj[tj < t]
            if past.size == 0:
                continue
            lam[i] += fit_alpha[i, j] * np.sum(np.exp(-np.clip(fit_beta[i, j] * (t - past), -700, 700)))
    return lam


def _log_likelihood(
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    events: list[np.ndarray],
    t_end: float,
) -> float:
    n = len(events)
    total = 0.0
    for i in range(n):
        ti = events[i]
        if len(ti) == 0:
            continue
        for t in ti:
            lam = _intensity_at(mu, alpha, beta, events, t)
            if lam[i] <= 0.0:
                return -np.inf
            total += np.log(lam[i])

    for i in range(n):
        total -= mu[i] * t_end
        for j in range(n):
            tj = events[j]
            if len(tj) == 0:
                continue
            integral = np.sum(1.0 - np.exp(-np.clip(beta[i, j] * (t_end - tj), -700, 700)))
            total -= (alpha[i, j] / beta[i, j]) * integral
    return float(total)


def fit_bivariate_hawkes(
    events: list[np.ndarray],
    t_end: float,
    decay: float | tuple[float, float] | np.ndarray = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> BivarHawkesFit:
    n = len(events)
    events = [np.unique(np.asarray(e, dtype=float)) for e in events]

    if t_end <= 0.0:
        raise ValueError("t_end must be positive")

    if isinstance(decay, (int, float)):
        beta = np.full((n, n), float(decay))
    else:
        decay_arr = np.asarray(decay, dtype=float).reshape(-1)
        beta = np.tile(decay_arr[:, None], (1, n))
        beta = np.maximum(beta, 1e-6)

    mu = np.array([max(len(e) / t_end, 1e-6) for e in events], dtype=float)
    alpha = np.full((n, n), 0.1, dtype=float)
    for i in range(n):
        alpha[i, i] = 0.0 if len(events[i]) == 0 else 0.1

    prev_ll = -np.inf
    for _ in range(max_iter):
        trig_counts = np.zeros((n, n), dtype=float)
        imm_counts = np.zeros(n, dtype=float)

        for i in range(n):
            ti = events[i]
            if len(ti) == 0:
                continue
            for t in ti:
                lam = _intensity_at(mu, alpha, beta, events, t)
                lam_i = max(lam[i], 1e-12)
                imm_counts[i] += mu[i] / lam_i
                for j in range(n):
                    tj = events[j]
                    if len(tj) == 0:
                        continue
                    past = tj[tj < t]
                    if past.size == 0:
                        continue
                    contrib = alpha[i, j] * np.sum(np.exp(-np.clip(beta[i, j] * (t - past), -700, 700)))
                    trig_counts[i, j] += contrib / lam_i

        mu = imm_counts / t_end
        mu = np.maximum(mu, 1e-8)
        for i in range(n):
            for j in range(n):
                tj = events[j]
                if len(tj) == 0:
                    alpha[i, j] = 0.0
                    continue
                denom = np.sum(1.0 - np.exp(-np.clip(beta[i, j] * (t_end - tj), -700, 700)))
                if denom <= 0.0:
                    alpha[i, j] = 0.0
                else:
                    alpha[i, j] = beta[i, j] * trig_counts[i, j] / denom
                    alpha[i, j] = min(alpha[i, j], 0.95 * beta[i, j])

        ll = _log_likelihood(mu, alpha, beta, events, t_end)
        if np.abs(ll - prev_ll) < tol:
            prev_ll = ll
            break
        prev_ll = ll

    with np.errstate(divide="ignore", invalid="ignore"):
        per_node = np.zeros(n, dtype=float)
        for i in range(n):
            row = alpha[i] / beta[i]
            per_node[i] = float(np.sum(row))

    return BivarHawkesFit(
        mu=mu,
        alpha=alpha,
        beta=beta,
        branching_ratio=per_node,
        log_likelihood=prev_ll,
    )


def hawkes_intensity(
    fit: BivarHawkesFit,
    events: list[np.ndarray],
    t_grid: np.ndarray,
) -> np.ndarray:
    t_grid = np.asarray(t_grid, dtype=float)
    n = len(events)
    out = np.zeros((n, t_grid.size), dtype=float)
    for i in range(n):
        out[i, :] = fit.mu[i]
        for j in range(n):
            tj = events[j]
            if len(tj) == 0:
                continue
            a_ij = fit.alpha[i, j]
            b_ij = fit.beta[i, j]
            for evt in tj:
                mask = t_grid >= evt
                if np.any(mask):
                    out[i, mask] += a_ij * np.exp(-np.clip(b_ij * (t_grid[mask] - evt), -700, 700))
    return out
