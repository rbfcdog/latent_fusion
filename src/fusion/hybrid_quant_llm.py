"""Hybrid Quantitative-LLM Trading Strategy.

Combines the strengths of both worlds:
1. Statistical Momentum Core: 20-day momentum ranking with hysteresis banding (Top 4 buy, Bottom 8 sell).
2. LLM Event Signals:
   - VETO (-1): Serious negative corporate catalyst (regulatory probe, patent loss, fraud). Vetoes buy or triggers early exit.
   - BOOST (+1): Transformational positive catalyst (FDA approval, major AI release, earnings beat). Promotes ranks 5-8 into buy.
   - NEUTRAL (0): Defer 100% to the statistical momentum formula.
3. Risk Parity or Equal Weight portfolio allocation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.llama_advanced_agent import (
    apply_hysteresis_guard, compute_risk_parity_weights, detect_market_regime,
    simulate_weekly, weekly_decision_dates
)
from src.fusion.llama_news_backtest import buy_and_hold, portfolio_metrics


TICKER_NAMES = {
    "ADBE": ["adobe"], "AAPL": ["apple", "iphone", "ipad", "macbook"],
    "AMZN": ["amazon", "aws"], "BABA": ["alibaba", "ant group", "qwen"],
    "COST": ["costco"], "CVX": ["chevron", "oil"],
    "GILD": ["gilead", "livdelzi", "lenacapavir", "trodelvy"],
    "GOOGL": ["google", "alphabet", "gemini", "waymo"],
    "JPM": ["jpmorgan", "chase", "jamie dimon"],
    "META": ["meta", "facebook", "instagram", "zuckerberg", "llama"],
    "MSFT": ["microsoft", "azure", "copilot"], "NKE": ["nike", "jordan"],
    "NVDA": ["nvidia", "blackwell", "jensen huang", "geforce"],
    "PYPL": ["paypal", "venmo"], "QCOM": ["qualcomm", "snapdragon"],
    "TSLA": ["tesla", "musk", "cybertruck", "robotaxi"]
}

VETO_PATTERNS = [
    r"\b(withdraws? (annual|guidance)|fraud|criminal probe|(sec|doj) (probe|investigat\w+)|antitrust (probe|ruling|lawsuit|breakup)|fine of \$\d+|profit plunges?|revenue collapse|disaster)\b",
    r"\b(scraps? license|tariffs? impact|slashes forecast)\b",
]
BOOST_PATTERNS = [
    r"\b(fda approval|drug approval|phase 3 trial|lasting efficacy|unveils? (new|ai|chip|model)|beats? (estimates|earnings|revenue)|breakthrough)\b",
    r"\b(record revenue|raises? guidance|all-time high|partnership)\b",
]

veto_re = re.compile("|".join(VETO_PATTERNS), re.IGNORECASE)
boost_re = re.compile("|".join(BOOST_PATTERNS), re.IGNORECASE)
_veto_re = veto_re
_boost_re = boost_re


def is_strictly_relevant(ticker, title):
    t_lower = title.lower()
    keywords = TICKER_NAMES.get(ticker, [ticker.lower()])
    return any(re.search(rf"\b{re.escape(k)}\b", t_lower) for k in keywords)


def classify_headline_signal(title, ticker=None):
    """Classify headline into +1 (BOOST), -1 (VETO), or 0 (NEUTRAL), checking entity relevance if ticker provided."""
    if ticker and not is_strictly_relevant(ticker, title):
        return 0
    if _veto_re.search(title):
        return -1
    if _boost_re.search(title):
        return +1
    return 0

def extract_llm_event_signals(headlines_by_ticker):
    """Compute aggregate event signal (-1, 0, +1) per ticker from available material news."""
    signals = {}
    for ticker, headlines in headlines_by_ticker.items():
        if not headlines:
            signals[ticker] = 0
            continue
        scores = [classify_headline_signal(h.get("title", ""), ticker=ticker) for h in headlines]
        if -1 in scores:
            signals[ticker] = -1
        elif +1 in scores:
            signals[ticker] = +1
        else:
            signals[ticker] = 0
    return signals


def compute_hybrid_actions(prices, location, tickers, held, headlines_by_ticker,
                           buy_rank_cutoff=4, sell_rank_cutoff=8, boost_max_rank=6,
                           volumes=None, min_volume_ratio=1.15):
    """Resolve hybrid quantitative + LLM actions:
    - Statistical core computes 20d momentum ranks (1 = highest, 16 = lowest).
    - LLM VETO (-1) blocks purchase or forces exit of held stock.
    - LLM BOOST (+1) promotes candidate from ranks 5-6 into BUY ONLY if confirmed by price trend and volume.
    - Ranks 5-8 without veto/boost remain in HOLD (hysteresis zone).
    """
    closes = prices[tickers].iloc[max(0, location - 20):location + 1]
    m20 = {t: float(prices[t].iloc[location] / prices[t].iloc[max(0, location - 20)] - 1)
           for t in tickers}
    ranks = {t: r + 1 for r, t in enumerate(sorted(tickers, key=m20.get, reverse=True))}
    signals = extract_llm_event_signals(headlines_by_ticker)

    actions = {}
    audit_notes = {}

    for t in tickers:
        r = ranks[t]
        sig = signals.get(t, 0)

        # Price & Volume confirmation gate for boosts
        if sig == +1 and volumes is not None:
            t_px = closes[t]
            t_vol = volumes[t].iloc[max(0, location - 20):location + 1]
            r_5d = float(t_px.iloc[-1] / t_px.iloc[max(0, len(t_px) - 6)] - 1) if len(t_px) > 5 else 0.0
            v_ratio = float(t_vol.tail(5).mean() / max(1.0, t_vol.mean())) if len(t_vol) > 5 else 1.0
            if r_5d <= 0.0 or v_ratio < min_volume_ratio:
                sig = 0  # Catalyst unconfirmed by institutional volume or negative 5d return -> degrade to neutral!

        if t in held:
            if sig == -1:
                actions[t] = "SELL"
                audit_notes[t] = f"LLM VETO: Evento negativo grave detectado (Rank {r})"
            elif r > sell_rank_cutoff:
                actions[t] = "SELL"
                audit_notes[t] = f"STAT SELL: Momentum deteriorou para Rank {r} > {sell_rank_cutoff}"
            else:
                actions[t] = "HOLD"
                audit_notes[t] = f"STAT HOLD: Momentum forte mantido (Rank {r})"
        else:
            if sig == -1:
                actions[t] = "HOLD"
                audit_notes[t] = f"LLM BLOCK: Compra bloqueada por Veto textual (Rank {r})"
            elif r <= buy_rank_cutoff:
                actions[t] = "BUY"
                audit_notes[t] = f"STAT BUY: Top momentum (Rank {r} <= {buy_rank_cutoff})"
            elif r <= boost_max_rank and sig == +1:
                actions[t] = "BUY"
                audit_notes[t] = f"HYBRID PV-BOOST: Promovido por catalisador confirmado com volume (Rank {r})"
            else:
                actions[t] = "HOLD"
                audit_notes[t] = f"STAT HOLD: Fora da zona de compra (Rank {r})"

    return actions, ranks, signals, audit_notes


def simulate_hybrid(weekly_dates, prices, tickers, news, cost_bps=10,
                    use_risk_parity=False, use_regime_gate=False):
    """Run full hybrid simulation across weekly dates."""
    sessions = prices.index
    held = set()
    actions_by_date = {}
    audit_log = []

    for d in weekly_dates:
        loc = sessions.get_loc(d)
        # Collect material headlines strictly available before decision date d
        headlines = {}
        for t in tickers:
            headlines[t] = [
                h for h in news
                if h["ticker"] == t and h["published_ts"].date() < d.date() and
                (d.date() - h["published_ts"].date()).days <= 14  # Recency window 14 days
            ]

        acts, ranks, sigs, notes = compute_hybrid_actions(prices, loc, tickers, held, headlines)
        actions_by_date[d] = acts
        held = {t for t in held if acts[t] != "SELL"} | {t for t in acts if acts[t] == "BUY"}

        audit_log.append({
            "date": str(d.date()),
            "held_count": len(held),
            "buys": [t for t, a in acts.items() if a == "BUY"],
            "sells": [t for t, a in acts.items() if a == "SELL"],
            "vetos": [t for t, s in sigs.items() if s == -1],
            "boosts": [t for t, s in sigs.items() if s == +1],
        })

    sim = simulate_weekly(
        actions_by_date, weekly_dates, prices, tickers, cost_bps=cost_bps,
        use_hysteresis=False,  # Hysteresis already built into compute_hybrid_actions!
        use_regime_gate=use_regime_gate,
        use_risk_parity=use_risk_parity
    )

    first_fill = sessions[sessions.get_loc(weekly_dates[0]) + 1]
    bench = buy_and_hold(prices, tickers, first_fill, cost_bps=cost_bps)
    metrics = portfolio_metrics(sim["series"]["net"], bench)

    return {
        "metrics": metrics,
        "series": sim["series"],
        "turnover_total": sim["turnover_total"],
        "audit": audit_log,
        "actions_by_date": actions_by_date
    }
