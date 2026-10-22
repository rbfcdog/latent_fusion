"""Weekly Quantitative Trading Engine for LLM Agents.

Strictly weekly schedule:
- Decision: Last trading session of each calendar week (Friday close, or Thursday if Friday is a holiday).
- Execution: First trading session of the following calendar week (Monday close, or Tuesday if Monday is a holiday).
- Holding: Positions held throughout the full week; no daily stops or mid-week turnover.

Integrated Quantitative Improvements:
1. Weekly Momentum Hysteresis Guard (Banding: BUY only rank<=4, SELL only rank>=8)
2. Weekly Market Regime Detector & Exposure Gate (BULL vs HIGH_VOL_CORRECTION)
3. Weekly Risk Parity / Inverse Volatility Weighting (w_i proportional to 1/vol_i)
4. Weekly Material Corporate Events (filtered hard events from the week)
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd

from src.fusion.llama_news_backtest import buy_and_hold, load_news, portfolio_metrics


def weekly_decision_dates(prices, start, end):
    """Return the last trading session of each calendar week within [start, end]."""
    sessions = prices.index
    window = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    if len(window) < 2:
        raise ValueError("período de decisões vazio")
    df_weeks = pd.DataFrame({"session": window, "week": window.to_period("W")})
    weekly = df_weeks.groupby("week")["session"].last().tolist()
    return [d for d in weekly if sessions.get_loc(d) + 1 < len(sessions)]


def detect_market_regime(prices, location, tickers, vol_threshold=0.26, ema_breadth_threshold=0.35):
    closes = prices[tickers].iloc[max(0, location - 20):location + 1]
    returns = closes.pct_change().dropna()
    if len(returns) > 1:
        mkt_returns = returns.mean(axis=1)
        mkt_vol = float(mkt_returns.std() * math.sqrt(252))
        if math.isnan(mkt_vol):
            mkt_vol = 0.16
    else:
        mkt_vol = 0.16

    above_ema = 0
    for t in tickers:
        t_closes = closes[t]
        span_n = min(20, max(2, len(t_closes)))
        ema20 = float(t_closes.ewm(span=span_n, adjust=False).mean().iloc[-1])
        if float(t_closes.iloc[-1]) > ema20:
            above_ema += 1
    breadth = above_ema / len(tickers)

    if mkt_vol > vol_threshold or breadth < ema_breadth_threshold:
        regime = "HIGH_VOL_CORRECTION"
        target_exposure = 0.40  # 60% cash defensive allocation
    elif breadth > 0.65 and mkt_vol < 0.20:
        regime = "BULL_TREND"
        target_exposure = 1.00  # 100% invested
    else:
        regime = "NEUTRAL"
        target_exposure = 0.80  # 20% cash buffer

    return {
        "regime": regime,
        "market_vol_annualized": round(mkt_vol, 4),
        "breadth_above_ema": round(breadth, 2),
        "target_exposure": target_exposure
    }


def compute_risk_parity_weights(held, prices, location, tickers, target_exposure=1.0):
    weights = np.zeros(len(tickers))
    if not held:
        return weights

    closes = prices[tickers].iloc[max(0, location - 20):location + 1]
    returns = closes.pct_change().dropna()

    inv_vols = []
    indices = []
    for t in held:
        idx = tickers.index(t)
        vol = float(returns[t].std() * math.sqrt(252)) if len(returns) > 1 else 0.20
        if math.isnan(vol) or vol <= 0.01:
            vol = 0.20
        inv_vols.append(1.0 / max(vol, 0.05))
        indices.append(idx)

    total_inv = sum(inv_vols)
    if total_inv > 0:
        for idx, inv in zip(indices, inv_vols):
            weights[idx] = target_exposure * (inv / total_inv)
    return weights


def apply_hysteresis_guard(candidate_actions, held, prices, location, tickers, has_pos_events=None, has_neg_events=None):
    has_pos = has_pos_events or {}
    has_neg = has_neg_events or {}

    buy_cutoff = max(1, len(tickers) // 4)
    sell_cutoff = max(2, len(tickers) // 2)

    m20 = {t: float(prices[t].iloc[location] / prices[t].iloc[max(0, location - 20)] - 1)
           for t in tickers}
    ranks = {t: r + 1 for r, t in enumerate(sorted(tickers, key=m20.get, reverse=True))}

    guarded = {}
    for t in tickers:
        act = candidate_actions.get(t, "HOLD")
        r = ranks[t]
        if act == "BUY":
            if t in held:
                guarded[t] = "HOLD"
            elif r <= buy_cutoff or has_pos.get(t, False):
                guarded[t] = "BUY"
            else:
                guarded[t] = "HOLD"
        elif act == "SELL":
            if t not in held:
                guarded[t] = "HOLD"
            elif r >= sell_cutoff or has_neg.get(t, False):
                guarded[t] = "SELL"
            else:
                guarded[t] = "HOLD"
        else:
            guarded[t] = "HOLD"
    return guarded, ranks


def simulate_weekly(actions_by_date, dates, prices, tickers, cost_bps=10,
                    use_risk_parity=True, use_regime_gate=True, use_hysteresis=True):
    """Strictly weekly simulation:
    - Decisions on Friday close (last day of week).
    - Fills on Monday close (first day of following week).
    - Positions held throughout the full week with market drift.
    - No daily stops or mid-week turnover.
    """
    sessions = prices.index
    daily = prices.pct_change().fillna(0.0)
    held = set()
    weights = np.zeros(len(tickers))
    gross = net = 1.0
    rows = []
    turnover_total = 0.0

    fill_map = {sessions[sessions.get_loc(d) + 1]: d for d in dates}
    start = min(fill_map)

    for session in sessions[sessions.get_loc(start):]:
        returns = daily.loc[session].to_numpy(dtype=float)
        growth = 1.0 + float(weights @ returns)
        gross *= growth
        net *= growth
        weights = weights * (1 + returns) / growth if growth > 0 else weights

        # Weekly fill execution on Monday close
        if session in fill_map:
            d_date = fill_map[session]
            d_loc = sessions.get_loc(d_date)
            raw_actions = actions_by_date.get(d_date, {})

            if use_hysteresis:
                guarded_actions, _ = apply_hysteresis_guard(raw_actions, held, prices, d_loc, tickers)
            else:
                guarded_actions = raw_actions

            target_held = {t for t in held if guarded_actions.get(t) != "SELL"} | \
                          {t for t in guarded_actions if guarded_actions.get(t) == "BUY"}

            if use_regime_gate:
                reg = detect_market_regime(prices, d_loc, tickers)
                exposure = reg["target_exposure"]
            else:
                exposure = 1.00

            if use_risk_parity:
                target_weights = compute_risk_parity_weights(target_held, prices, d_loc, tickers, exposure)
            else:
                target_weights = np.zeros(len(tickers))
                if target_held:
                    target_weights[[tickers.index(t) for t in target_held]] = exposure / len(target_held)

            turnover = float(np.abs(target_weights - weights).sum())
            turnover_total += turnover
            net *= (1.0 - (cost_bps / 10_000) * turnover)

            weights = target_weights
            held = target_held

        rows.append({
            "session": session, "gross": gross, "net": net, "held": len(held),
            "invested_weight": round(float(weights.sum()), 4)
        })

    df = pd.DataFrame(rows).set_index("session")
    return {
        "series": df,
        "turnover_total": turnover_total
    }
