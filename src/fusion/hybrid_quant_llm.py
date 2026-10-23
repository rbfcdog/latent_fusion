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


# Regex event classifier for material corporate catalysts
VETO_PATTERNS = [
    r"\b(sec probe|doj probe|antitrust|lawsuit|investigat\w+|scraps? license|sanctions?|sales decline|weak holiday season|disaster|fine|fraud)\b",
    r"\b(withdraws? (annual|guidance)|delays?|cuts? price target|slashes|plunges?)\b",
]
BOOST_PATTERNS = [
    r"\b(fda approval|drug approval|phase [23] (success|trial)|lasting efficacy|unveils? (new|ai|chip|model)|beats? (estimates|earnings|revenue)|breakthrough)\b",
    r"\b(growth|record revenue|raises? guidance|all-time high|expansion|partnership)\b",
]

_veto_re = re.compile("|".join(VETO_PATTERNS), re.IGNORECASE)
_boost_re = re.compile("|".join(BOOST_PATTERNS), re.IGNORECASE)


def classify_headline_signal(title):
    """Classify headline into +1 (BOOST), -1 (VETO), or 0 (NEUTRAL)."""
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
        scores = [classify_headline_signal(h.get("title", "")) for h in headlines]
        # If any strong veto, signal is -1
        if -1 in scores:
            signals[ticker] = -1
        elif +1 in scores:
            signals[ticker] = +1
        else:
            signals[ticker] = 0
    return signals


def compute_hybrid_actions(prices, location, tickers, held, headlines_by_ticker,
                           buy_rank_cutoff=4, sell_rank_cutoff=8, boost_max_rank=8):
    """Resolve hybrid quantitative + LLM actions:
    - Statistical core computes 20d momentum ranks (1 = highest, 16 = lowest).
    - LLM VETO (-1) blocks purchase or forces exit of held stock.
    - LLM BOOST (+1) promotes candidate from ranks 5-8 into BUY.
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

        if t in held:
            if sig == -1:
                actions[t] = "SELL"
                audit_notes[t] = f"LLM VETO: Evento negativo detectado (Rank {r})"
            elif r > sell_rank_cutoff:
                actions[t] = "SELL"
                audit_notes[t] = f"STAT SELL: Momentum deteriorou para Rank {r} > {sell_rank_cutoff}"
            else:
                actions[t] = "HOLD"
                audit_notes[t] = f"STAT HOLD: Momentum forte mantido (Rank {r})"
        else:
            # Not held
            if sig == -1:
                actions[t] = "HOLD"
                audit_notes[t] = f"LLM BLOCK: Compra bloqueada por Veto textual (Rank {r})"
            elif r <= buy_rank_cutoff:
                actions[t] = "BUY"
                audit_notes[t] = f"STAT BUY: Top momentum (Rank {r} <= {buy_rank_cutoff})"
            elif r <= boost_max_rank and sig == +1:
                actions[t] = "BUY"
                audit_notes[t] = f"HYBRID BOOST: Promovido para Compra por catalisador positivo (Rank {r})"
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
