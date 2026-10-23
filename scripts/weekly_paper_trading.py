#!/usr/bin/env python3
"""Weekly Paper Trading Runner — Hybrid Quantitative Momentum + LLM Catalyst System.

Runs every Friday after market close (16:30 ET / 20:30 UTC):
1. Loads latest prices and daily volumes for the 16-asset universe.
2. Reads material corporate events from the week.
3. Computes 20-day momentum ranks with hysteresis banding (Top 4 Buy / Bottom 8 Sell).
4. Evaluates catalyst boosts using entity-relevance and price-volume confirmation.
5. Emits explicit rebalance orders for Monday close execution.
6. Writes a timestamped, SHA-256 attested report to paper_trading_results/weekly_hybrid_*.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fusion.hybrid_quant_llm import (
    TICKER_NAMES, boost_re, is_strictly_relevant, veto_re
)
from src.fusion.llama_advanced_agent import apply_hysteresis_guard, compute_risk_parity_weights, detect_market_regime
from src.fusion.llama_news_backtest import load_news

BASE = ROOT / "data" / "llama_news_api"
PRICES_PATH = BASE / "prices_2024_2026.parquet"
VOLUMES_PATH = BASE / "volumes_2024_2026.parquet"
NEWS_PATH = ROOT / "data" / "live_news" / "capture_filtered.jsonl"
PORTFOLIO_STATE_PATH = ROOT / "paper_trading_results" / "weekly_hybrid_state.json"
REPORT_DIR = ROOT / "paper_trading_results"


def run_weekly_paper_trading(cost_bps=10, min_volume_ratio=1.10):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y_%m_%d")

    prices = pd.read_parquet(PRICES_PATH)
    volumes = pd.read_parquet(VOLUMES_PATH)
    tickers = list(prices.columns)
    news = load_news(NEWS_PATH)

    # Load current portfolio state
    if PORTFOLIO_STATE_PATH.is_file():
        state = json.loads(PORTFOLIO_STATE_PATH.read_text(encoding="utf-8"))
        held = set(state.get("held", []))
        weights_before = {t: state.get("weights", {}).get(t, 0.0) for t in tickers}
    else:
        held = set()
        weights_before = {t: 0.0 for t in tickers}

    latest_session = prices.index[-1]
    loc = len(prices) - 1

    # 1. Compute 20-day momentum ranks
    closes = prices[tickers].iloc[max(0, loc - 20):loc + 1]
    m20 = {t: float(prices[t].iloc[loc] / prices[t].iloc[max(0, loc - 20)] - 1) for t in tickers}
    ranks = {t: r + 1 for r, t in enumerate(sorted(tickers, key=m20.get, reverse=True))}

    # 2. Check regime
    regime = detect_market_regime(prices, loc, tickers)

    # 3. Check material events with entity relevance + price-volume confirmation
    has_pos = {}
    has_neg = {}
    active_catalysts = {}

    for t in tickers:
        t_news = [
            h for h in news
            if h["ticker"] == t and (now_utc.date() - h["published_ts"].date()).days <= 14
        ]
        t_px = closes[t]
        t_vol = volumes[t].iloc[max(0, loc - 20):loc + 1]
        r_5d = float(t_px.iloc[-1] / t_px.iloc[max(0, len(t_px) - 6)] - 1) if len(t_px) > 5 else 0.0
        v_ratio = float(t_vol.tail(5).mean() / max(1.0, t_vol.mean())) if len(t_vol) > 5 else 1.0

        for h in t_news:
            title = h["title"]
            if is_strictly_relevant(t, title):
                if boost_re.search(title):
                    active_catalysts[t] = f"[CATALYST] {title[:80]}"
                    if ranks[t] <= 6 and r_5d > 0.0 and v_ratio >= min_volume_ratio:
                        has_pos[t] = True
                elif veto_re.search(title):
                    active_catalysts[t] = f"[VETO] {title[:80]}"
                    if r_5d < -0.01:
                        has_neg[t] = True

    # 4. Generate candidate raw actions (Top 8 momentum)
    raw_actions = {}
    for t in tickers:
        r = ranks[t]
        if t in held:
            raw_actions[t] = "HOLD" if r <= 8 else "SELL"
        else:
            raw_actions[t] = "BUY" if r <= 4 else "HOLD"

    # 5. Apply Hysteresis Guard with Catalysts
    guarded, _ = apply_hysteresis_guard(raw_actions, held, prices, loc, tickers,
                                        has_pos_events=has_pos, has_neg_events=has_neg)

    target_held = {t for t in held if guarded.get(t) != "SELL"} | {t for t in guarded if guarded.get(t) == "BUY"}
    target_weights = np.zeros(len(tickers))
    if target_held:
        target_weights[[tickers.index(t) for t in target_held]] = 1.0 / len(target_held)

    # 6. Calculate turnover & fees
    w_before_arr = np.array([weights_before[t] for t in tickers])
    turnover = float(np.abs(target_weights - w_before_arr).sum())
    est_fee_bps = cost_bps * turnover

    # Update state file
    new_state = {
        "as_of_session": str(latest_session.date()),
        "updated_at": now_utc.isoformat(),
        "held": sorted(target_held),
        "weights": {t: round(float(target_weights[i]), 4) for i, t in enumerate(tickers)},
        "regime": regime["regime"],
        "turnover": round(turnover, 4)
    }
    PORTFOLIO_STATE_PATH.write_text(json.dumps(new_state, indent=2), encoding="utf-8")

    # 7. Generate Markdown Report
    report_lines = [
        f"# Weekly Paper Trading Report — {date_str} (Execution: Next Monday Close)",
        f"",
        f"**Data da Decisão:** {latest_session.date()} (Fechamento) | **Execução:** Próxima segunda-feira no fechamento",
        f"**Regime Macro Detectado:** `{regime['regime']}` (Volatilidade anualizada: {regime['market_vol_annualized']*100:.1f}%, Amplitude EMA: {regime['breadth_above_ema']*100:.0f}%)",
        f"**Turnover da Rebalanceamento:** `{turnover:.2f}` (Taxa estimada: `{est_fee_bps:.1f} bps`)",
        f"",
        f"## 1. Ordens de Execução para Segunda-Feira",
        f"",
        f"| Ticker | Posição Anterior | Ação Recomendada | Peso Alvo | Catalisador / Justificativa |",
        f"|---|---|---|---:|---|"
    ]

    for t in sorted(tickers, key=lambda x: ranks[x]):
        r = ranks[t]
        act = guarded[t]
        is_held = t in held
        tgt_w = float(target_weights[tickers.index(t)]) * 100
        cat_note = active_catalysts.get(t, "Momentum de mercado")
        if act == "BUY":
            note = f"🟢 **COMPRA** — Rank {r} (Top momentum)" if not has_pos.get(t) else f"🚀 **BOOST** — Rank {r} + {cat_note}"
        elif act == "SELL":
            note = f"🔴 **VENDA** — Rank {r} (Deterioração > 8)" if not has_neg.get(t) else f"⚠️ **VETO** — {cat_note}"
        else:
            note = f"⚪ **HOLD** — Rank {r} (Histerese protegida)" if is_held else f"— Fora da carteira (Rank {r})"

        report_lines.append(f"| **{t}** | {'HELD' if is_held else 'CASH'} | **{act}** | {tgt_w:.1f}% | {note} |")

    report_lines.extend([
        f"",
        f"## 2. Resumo da Carteira Alvo",
        f"",
        f"- **Ações Alvo na Carteira ({len(target_held)} ativos):** {', '.join(sorted(target_held)) or '100% Caixa'}",
        f"- **Alocação Investida:** {sum(target_weights)*100:.0f}% | **Caixa:** {(1.0 - sum(target_weights))*100:.0f}%",
        f"",
        f"---",
        f"```json",
        json.dumps(new_state, indent=2),
        f"```",
        f"",
        f"**Assinatura de Autenticidade (SHA-256 do Manifesto):**",
        f"`{hashlib.sha256(json.dumps(new_state, sort_keys=True).encode()).hexdigest()}`",
        f"",
        f"*Relatório carimbado prospectivamente antes do pregão de segunda-feira.*"
    ])

    report_content = "\n".join(report_lines)
    report_file = REPORT_DIR / f"weekly_hybrid_{date_str}.md"
    report_latest = REPORT_DIR / "weekly_hybrid_latest.md"
    report_file.write_text(report_content, encoding="utf-8")
    report_latest.write_text(report_content, encoding="utf-8")

    print(f"Relatório de Paper Trading gerado com sucesso:")
    print(f" -> {report_file}")
    print(f" -> {report_latest}")
    print(f"Ações recomendadas: BUY = {[t for t, a in guarded.items() if a == 'BUY']}, SELL = {[t for t, a in guarded.items() if a == 'SELL']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-bps", type=int, default=10)
    args = parser.parse_args()
    run_weekly_paper_trading(cost_bps=args.cost_bps)
