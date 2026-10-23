#!/usr/bin/env python3
import json
import math
import re
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
from src.fusion.llama_advanced_agent import apply_hysteresis_guard, simulate_weekly, weekly_decision_dates
from src.fusion.llama_news_backtest import buy_and_hold, load_news, numeric_actions, portfolio_metrics

BASE = ROOT / "data" / "llama_news_api"
PRICES = pd.read_parquet(BASE / "prices_2024_2026.parquet")
NEWS = load_news(ROOT / "data" / "live_news" / "capture_filtered.jsonl")
TICKERS = list(PRICES.columns)
OUT_DIR = BASE / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

BOOST_PATTERNS = [
    r"\b(fda approval|drug approval|phase 3 trial|lasting efficacy|unveils? (new|ai|chip|model)|beats? (estimates|earnings|revenue)|breakthrough)\b",
    r"\b(record revenue|raises? guidance|all-time high|partnership)\b",
]
boost_re = re.compile("|".join(BOOST_PATTERNS), re.IGNORECASE)

VETO_PATTERNS = [
    r"\b(withdraws? (annual|guidance)|fraud|criminal probe|antitrust (ruling|breakup)|fine of \$\d+|profit plunges?|revenue collapse|disaster)\b",
    r"\b(scraps? license|tariffs? impact|slashes forecast)\b",
]
veto_re = re.compile("|".join(VETO_PATTERNS), re.IGNORECASE)


def is_strictly_relevant(ticker, title):
    t_lower = title.lower()
    keywords = TICKER_NAMES.get(ticker, [ticker.lower()])
    return any(re.search(rf"\b{re.escape(k)}\b", t_lower) for k in keywords)


def run_hybrid_experiment(cost_bps=10):
    weekly_dates = weekly_decision_dates(PRICES, "2024-08-01", str(PRICES.index[-1].date()))
    sessions = PRICES.index
    first_fill = sessions[sessions.get_loc(weekly_dates[0]) + 1]

    # Benchmark: Buy and hold
    bench = buy_and_hold(PRICES, TICKERS, first_fill, cost_bps=cost_bps)

    # Pre-compute raw numeric actions (Top 8 momentum)
    raw_numeric_actions = {}
    held_simple = set()
    for d in weekly_dates:
        acts = numeric_actions(PRICES, d, TICKERS, held_simple, top_k=8)
        raw_numeric_actions[d] = acts
        held_simple = {t for t in held_simple if acts[t] != "SELL"} | {t for t in acts if acts[t] == "BUY"}

    # Simulation engine for hybrid variations
    def sim_engine(use_hysteresis=True, use_events=True, use_regime=False, use_rp=False):
        fill_map = {sessions[sessions.get_loc(d) + 1]: d for d in weekly_dates}
        start = min(fill_map)
        held = set()
        weights = np.zeros(len(TICKERS))
        daily = PRICES.pct_change().fillna(0.0)
        gross = net = 1.0
        rows = []
        turnover_total = 0.0

        for session in sessions[sessions.get_loc(start):]:
            returns = daily.loc[session].to_numpy(dtype=float)
            growth = 1.0 + float(weights @ returns)
            gross *= growth
            net *= growth
            weights = weights * (1 + returns) / growth if growth > 0 else weights

            if session in fill_map:
                d = fill_map[session]
                loc = sessions.get_loc(d)
                raw = raw_numeric_actions[d]
                m20 = {t: float(PRICES[t].iloc[loc] / PRICES[t].iloc[max(0, loc - 20)] - 1) for t in TICKERS}
                ranks = {t: r + 1 for r, t in enumerate(sorted(TICKERS, key=m20.get, reverse=True))}

                has_pos = {}
                has_neg = {}
                if use_events:
                    for t in TICKERS:
                        t_news = [h for h in NEWS if h["ticker"] == t and h["published_ts"].date() < d.date() and (d.date() - h["published_ts"].date()).days <= 14]
                        if any(boost_re.search(h["title"]) and is_strictly_relevant(t, h["title"]) for h in t_news):
                            if ranks[t] <= 6:
                                has_pos[t] = True
                        if any(veto_re.search(h["title"]) and is_strictly_relevant(t, h["title"]) for h in t_news):
                            has_neg[t] = True

                if use_hysteresis:
                    guarded, _ = apply_hysteresis_guard(raw, held, PRICES, loc, TICKERS, has_pos_events=has_pos, has_neg_events=has_neg)
                else:
                    guarded = raw

                target_held = {t for t in held if guarded.get(t) != "SELL"} | {t for t in guarded if guarded.get(t) == "BUY"}

                target_weights = np.zeros(len(TICKERS))
                exposure = 0.40 if (use_regime and (returns.mean() < 0 or returns.std() > 0.018)) else 1.00
                if target_held:
                    target_weights[[TICKERS.index(t) for t in target_held]] = exposure / len(target_held)

                turnover = float(np.abs(target_weights - weights).sum())
                turnover_total += turnover
                net *= (1.0 - (cost_bps / 10_000) * turnover)
                weights = target_weights
                held = target_held

            rows.append({"session": session, "net": net, "gross": gross, "held": len(held)})

        df = pd.DataFrame(rows).set_index("session")
        return df, turnover_total

    strategies = {
        "Comprar e Manter (16 ativos)": {"type": "bench"},
        "Numérico Simples (Sem Histerese, Pesos Iguais)": {"hyst": False, "events": False, "regime": False},
        "Estatístico Puro (Histerese Top 4 / Bottom 8)": {"hyst": True, "events": False, "regime": False},
        "Sistema Híbrido: Histerese + Catalisadores Auditados": {"hyst": True, "events": True, "regime": False},
        "Sistema Híbrido + Regime Gate Defensivo": {"hyst": True, "events": True, "regime": True},
    }

    # Add Tool Agent result from summary.json if available
    tool_agent_summary = BASE / "tool_agent_70b" / "summary.json"
    tool_agent_decisions = BASE / "tool_agent_70b" / "decisions.jsonl"
    if tool_agent_decisions.is_file():
        dec_tool = [json.loads(l) for l in tool_agent_decisions.read_text(encoding="utf-8").splitlines()]
        dates_tool = sorted([pd.Timestamp(d["decision_date"]) for d in dec_tool])
        acts_tool = {pd.Timestamp(d["decision_date"]): d["actions"] for d in dec_tool}
        sim_tool = simulate_weekly(acts_tool, dates_tool, PRICES, TICKERS, cost_bps=cost_bps,
                                   use_hysteresis=False, use_regime_gate=False, use_risk_parity=False)
        strategies["Llama 70B Tool Agent (Function Calling Interativo)"] = {"type": "custom", "series": sim_tool["series"], "turnover": sim_tool["turnover_total"]}

    results = {}
    curves = {}

    for name, cfg in strategies.items():
        if cfg.get("type") == "bench":
            metrics = portfolio_metrics(bench, bench)
            results[name] = {
                "net_return": metrics["total_return"],
                "gross_return": metrics["total_return"] + 0.001,
                "fee_drag_pp": 0.10,
                "turnover": 1.0,
                "alpha_annual": 0.0,
                "beta": 1.0,
                "max_drawdown": metrics["max_drawdown"]
            }
            curves[name] = bench
        elif cfg.get("type") == "custom":
            df = cfg["series"].reindex(bench.index).ffill()
            df["net"] = df["net"] / df["net"].iloc[0]
            df["gross"] = df["gross"] / df["gross"].iloc[0]
            metrics = portfolio_metrics(df["net"], bench)
            results[name] = {
                "net_return": metrics["total_return"],
                "gross_return": float(df["gross"].iloc[-1] - 1),
                "fee_drag_pp": round((df["gross"].iloc[-1] - df["net"].iloc[-1]) * 100, 2),
                "turnover": round(cfg["turnover"], 2),
                "alpha_annual": metrics["alpha_annual"],
                "beta": metrics["beta"],
                "max_drawdown": metrics["max_drawdown"]
            }
            curves[name] = df["net"]
        else:
            df, turn = sim_engine(use_hysteresis=cfg["hyst"], use_events=cfg["events"], use_regime=cfg["regime"])
            metrics = portfolio_metrics(df["net"], bench)
            fee_drag = (df["gross"].iloc[-1] - df["net"].iloc[-1]) * 100
            results[name] = {
                "net_return": metrics["total_return"],
                "gross_return": float(df["gross"].iloc[-1] - 1),
                "fee_drag_pp": round(fee_drag, 2),
                "turnover": round(turn, 2),
                "alpha_annual": metrics["alpha_annual"],
                "beta": metrics["beta"],
                "max_drawdown": metrics["max_drawdown"]
            }
            curves[name] = df["net"]

    # Save JSON summary
    out_json = OUT_DIR / "hybrid_experiment.json"
    out_json.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=float), encoding="utf-8")

    # Plot dark-mode comparison
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_title("Experimento Híbrido: Combinação Ótima Quantitativa-Estatística + LLM", fontsize=15, color=WHITE)
    ax.set_ylabel("Patrimônio Líquido (base 1, com 10 bps de taxa)", fontsize=13, color=WHITE)
    ax.tick_params(colors=WHITE, labelsize=11)
    ax.grid(alpha=0.12)

    palette = [GOLD, RED, PURPLE, GREEN, CYAN, ORANGE]
    for idx, (label, curve) in enumerate(curves.items()):
        color = palette[idx % len(palette)]
        lw = 2.4 if "Híbrido" in label or "Histerese" in label or "Comprar e Manter" in label else 1.5
        ls = "--" if "Regime" in label else "-"
        ax.plot(curve, color=color, label=f"{label} [{results[label]['net_return']*100:+.1f}%]", linewidth=lw, linestyle=ls)

    ax.axhline(1.0, color=WHITE, alpha=0.2, linestyle=":")
    ax.legend(fontsize=10, loc="upper left")

    out_img = OUT_DIR / "equity_hybrid_experiment.png"
    fig.savefig(out_img, dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Salvo gráfico: {out_img}")
    print(json.dumps(results, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    run_hybrid_experiment()
