#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.visualization import BG, CYAN, GOLD, GREEN, ORANGE, PANEL, PURPLE, RED, WHITE
from src.fusion.llama_advanced_agent import (
    apply_hysteresis_guard, compute_risk_parity_weights, detect_market_regime,
    simulate_weekly, weekly_decision_dates
)
from src.fusion.llama_news_backtest import buy_and_hold, numeric_actions, portfolio_metrics

BASE = ROOT / "data" / "llama_news_api"
PRICES = pd.read_parquet(BASE / "prices_2024_2026.parquet")
TICKERS = list(PRICES.columns)
OUT_DIR = BASE / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_weekly_ablation():
    weekly_dates = weekly_decision_dates(PRICES, "2024-08-01", str(PRICES.index[-1].date()))
    sessions = PRICES.index
    first_fill = sessions[sessions.get_loc(weekly_dates[0]) + 1]

    # Benchmark: Buy and hold
    bench = buy_and_hold(PRICES, TICKERS, first_fill, cost_bps=10)

    # 1. Generate Raw Numeric Momentum Actions (Top 8) on each weekly Friday
    raw_numeric_actions = {}
    held_simple = set()
    for d in weekly_dates:
        acts = numeric_actions(PRICES, d, TICKERS, held_simple, top_k=8)
        raw_numeric_actions[d] = acts
        held_simple = {t for t in held_simple if acts[t] != "SELL"} | {t for t in acts if acts[t] == "BUY"}

    # 2. Ablation configurations
    configs = {
        "Comprar e Manter (16 ativos)": {
            "type": "bench"
        },
        "Numérico Semanal (Sem Histerese, Pesos Iguais)": {
            "actions": raw_numeric_actions,
            "use_hysteresis": False, "use_regime_gate": False, "use_risk_parity": False
        },
        "+ Histerese de Momentum (Banding Top 4 / Bottom 8)": {
            "actions": raw_numeric_actions,
            "use_hysteresis": True, "use_regime_gate": False, "use_risk_parity": False
        },
        "+ Histerese + Regime Gate (Proteção Macro de Alta Vol)": {
            "actions": raw_numeric_actions,
            "use_hysteresis": True, "use_regime_gate": True, "use_risk_parity": False
        },
        "+ Histerese + Regime Gate + Risk Parity (Todas Melhorias)": {
            "actions": raw_numeric_actions,
            "use_hysteresis": True, "use_regime_gate": True, "use_risk_parity": True
        },
    }

    # 3. Add Llama 8B and 70B mapped to weekly dates with improvements
    # Check if models have actions available
    for model_tag, label in [
        ("openrouter-deepinfra-llama-3.1-8b_filtered", "Llama 8B Semanal (Todas Melhorias)"),
        ("openrouter-deepinfra-llama-3.1-70b_filtered", "Llama 70B Semanal (Todas Melhorias)")
    ]:
        p = BASE / model_tag / "decisions.jsonl"
        if p.is_file():
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            news_rows = {pd.Timestamp(r["decision_date"]): r["actions"] for r in rows if r["arm"] == "news"}
            # Map nearest decision to each weekly date
            mapped_acts = {}
            for d in weekly_dates:
                # Find closest decision date <= d
                avail = [dt for dt in news_rows if dt <= d]
                if avail:
                    mapped_acts[d] = news_rows[max(avail)]
                else:
                    mapped_acts[d] = {t: "HOLD" for t in TICKERS}
            configs[label] = {
                "actions": mapped_acts,
                "use_hysteresis": True, "use_regime_gate": True, "use_risk_parity": True
            }

    results = {}
    curves = {}

    for name, cfg in configs.items():
        if cfg.get("type") == "bench":
            metrics = portfolio_metrics(bench, bench)
            curves[name] = bench
            results[name] = {
                "net_return": metrics["total_return"],
                "gross_return": metrics["total_return"] + 0.001,
                "fee_drag_pp": 0.10,
                "turnover": 1.0,
                "alpha_annual": 0.0,
                "beta": 1.0,
                "max_drawdown": metrics["max_drawdown"]
            }
        else:
            sim_net = simulate_weekly(
                cfg["actions"], weekly_dates, PRICES, TICKERS, cost_bps=10,
                use_hysteresis=cfg["use_hysteresis"],
                use_regime_gate=cfg["use_regime_gate"],
                use_risk_parity=cfg["use_risk_parity"]
            )
            sim_gross = simulate_weekly(
                cfg["actions"], weekly_dates, PRICES, TICKERS, cost_bps=0,
                use_hysteresis=cfg["use_hysteresis"],
                use_regime_gate=cfg["use_regime_gate"],
                use_risk_parity=cfg["use_risk_parity"]
            )
            net_eq = sim_net["series"]["net"]
            gross_eq = sim_gross["series"]["gross"]
            metrics = portfolio_metrics(net_eq, bench)
            fee_drag = (gross_eq.iloc[-1] - net_eq.iloc[-1]) * 100
            results[name] = {
                "net_return": metrics["total_return"],
                "gross_return": float(gross_eq.iloc[-1] - 1),
                "fee_drag_pp": round(fee_drag, 2),
                "turnover": round(sim_net["turnover_total"], 2),
                "alpha_annual": metrics["alpha_annual"],
                "beta": metrics["beta"],
                "max_drawdown": metrics["max_drawdown"]
            }
            curves[name] = net_eq

    # Save JSON summary
    (OUT_DIR / "weekly_ablation.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False, default=float), encoding="utf-8"
    )

    # Plot dark-mode comparison chart
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_title("Estratégia Semanal (Sexta-Feira Fechamento $\\rightarrow$ Segunda Execução) com Melhorias Quantitativas", fontsize=15, color=WHITE)
    ax.set_ylabel("Patrimônio Líquido (base 1, com 10 bps de taxa)", fontsize=13, color=WHITE)
    ax.tick_params(colors=WHITE, labelsize=11)
    ax.grid(alpha=0.12)

    palette = [GOLD, RED, PURPLE, CYAN, GREEN, ORANGE, WHITE]
    for idx, (label, curve) in enumerate(curves.items()):
        color = palette[idx % len(palette)]
        lw = 2.4 if "Todas Melhorias" in label or "Comprar e Manter" in label else 1.5
        ls = "--" if "Sem Histerese" in label else "-"
        ax.plot(curve, color=color, label=f"{label} [{results[label]['net_return']*100:+.1f}%]", linewidth=lw, linestyle=ls)

    ax.axhline(1.0, color=WHITE, alpha=0.2, linestyle=":")
    ax.legend(fontsize=10, loc="upper left")

    out_img = OUT_DIR / "equity_weekly_improvements.png"
    fig.savefig(out_img, dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Salvo gráfico: {out_img}")
    print(json.dumps(results, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    run_weekly_ablation()
