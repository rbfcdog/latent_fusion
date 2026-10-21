"""No-training intersection of saved Luna and numeric held positions at each decision close."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import pandas as pd

from src.backtest.visualization import BG, PANEL, CYAN, GOLD, GREEN, PURPLE, RED, WHITE
from src.fusion.event_experiment import _portfolio_metrics
from src.fusion.llm_costs import load_saved_signals
from src.fusion.llm_trading_agent import _apply, _backtest


def consensus_actions(tickers, schedule, luna_actions, numeric_actions):
    if len(schedule) == 0 or set(schedule) != set(luna_actions) or set(schedule) != set(numeric_actions):
        raise ValueError("Calendários de decisão não coincidem")
    luna_held, numeric_held, hybrid_held = set(), set(), set()
    combined, audit = {}, []
    for date in schedule:
        if set(luna_actions[date]) != set(tickers) or set(numeric_actions[date]) != set(tickers):
            raise ValueError("Emissor ausente na decisão")
        luna_held = _apply(luna_held, luna_actions[date])
        numeric_held = _apply(numeric_held, numeric_actions[date])
        target = luna_held & numeric_held
        today = {ticker: ("BUY" if ticker in target else "SELL")
                 if (ticker in target) != (ticker in hybrid_held) else "HOLD" for ticker in tickers}
        hybrid_held = _apply(hybrid_held, today)
        for ticker in tickers:
            audit.append({"date": date, "ticker": ticker, "luna_action": luna_actions[date][ticker],
                          "numeric_action": numeric_actions[date][ticker], "hybrid_action": today[ticker],
                          "luna_held": ticker in luna_held, "numeric_held": ticker in numeric_held,
                          "hybrid_held": ticker in hybrid_held})
        combined[date] = today
    return combined, pd.DataFrame(audit)


def run_hybrid(panel_path, news_path, baseline_csv, luna_dir, out_dir, *, cost_bps=10):
    panel, schedule, luna, numeric, _, source = load_saved_signals(
        panel_path, news_path, baseline_csv, luna_dir, cost_bps=cost_bps)
    if source["model_id"] != "gpt-6-luna":
        raise ValueError("Esperadas decisões auditadas do GPT-6 Luna")
    hybrid, decisions = consensus_actions(panel["tickers"], schedule, luna, numeric)
    actions = {"hybrid": hybrid, "luna": luna, "numeric": numeric}
    curves = {name: _backtest(panel, schedule, policy, cost_bps) for name, policy in actions.items()}
    gross_curves = {name: _backtest(panel, schedule, policy, 0) for name, policy in actions.items()}
    if any(not curve.index.equals(curves["hybrid"].index) for curve in curves.values()):
        raise ValueError("Calendários de avaliação diferentes")
    bh = curves["hybrid"].copy()
    bh["strategy"] = bh["bh"]
    bh["turnover"] = bh["bh_turnover"]
    curves["buy_hold"] = bh
    summary = {}
    bh_final = (bh.strategy.iloc[-1] - 1) * 100
    for name, curve in curves.items():
        gross = gross_curves[name].strategy if name in actions else gross_curves["hybrid"].bh
        stats = _portfolio_metrics(curve, cost_bps)
        summary[name] = {"gross_return_pct": (gross.iloc[-1] - 1) * 100,
                         "net_return_pct": stats["return_pct"],
                         "excess_vs_bh_pp": stats["return_pct"] - bh_final,
                         "trading_cost_drag_pp": (gross.iloc[-1] - curve.strategy.iloc[-1]) * 100,
                         "turnover_total": stats["turnover_total"],
                         "max_drawdown_pct": (curve.strategy / curve.strategy.cummax() - 1).min() * 100,
                         "alpha_annual_pct": stats["alpha_annual_pct"], "beta": stats["beta"]}
    comparison = pd.DataFrame.from_dict(summary, orient="index")
    equity = pd.DataFrame({name: curve.strategy for name, curve in curves.items()})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    forecast_copy = out / "numeric_predictions.csv"
    if Path(baseline_csv).resolve() != forecast_copy.resolve():
        copyfile(baseline_csv, forecast_copy)
    decisions.to_csv(out / "hybrid_decisions.csv", index=False)
    comparison.to_csv(out / "hybrid_comparison.csv", index_label="strategy")
    equity.to_csv(out / "equity_curves.csv")
    (out / "run_config.json").write_text(json.dumps({
        "policy": "Held only when both independently saved Luna and numeric policies hold the ticker; equal weight among intersection; otherwise cash",
        "luna_model": source["model_id"], "luna_cache": source["cache"],
        "panel_sha256": source["panel_sha256"], "news_sha256": source["news_sha256"],
        "baseline_sha256": source["baseline_sha256"], "baseline_file": forecast_copy.name,
        "rebalance_sessions": source["rebalance_sessions"],
        "cost_bps": cost_bps, "decision_dates": len(schedule),
        "note": "No new LLM calls or model inference; the examined 2022–23 test is exploratory."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for name, label, color in (("hybrid", "Luna ∩ numérico", CYAN), ("luna", "Luna", PURPLE),
                               ("numeric", "Numérico", GOLD), ("buy_hold", "Comprar e manter", GREEN)):
        ax.plot(equity.index, equity[name], label=label, color=color, linewidth=2.2)
    ax.fill_between(equity.index, equity.hybrid, equity.buy_hold,
                    where=equity.hybrid >= equity.buy_hold, color=GREEN, alpha=0.12)
    ax.fill_between(equity.index, equity.hybrid, equity.buy_hold,
                    where=equity.hybrid < equity.buy_hold, color=RED, alpha=0.12)
    ax.set_title("Interseção de posições: Luna e previsão numérica", fontsize=16, color="white")
    ax.set_ylabel("Capital líquido (base 1)", fontsize=14, color="white")
    ax.tick_params(colors="white", labelsize=12)
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_color(WHITE)
    ax.legend(fontsize=14, facecolor=PANEL, labelcolor="white", edgecolor=WHITE)
    fig.tight_layout()
    fig.savefig(out / "hybrid_vs_benchmarks.png", dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)
    return {"comparison": comparison, "equity": equity, "decisions": decisions, "output": out}
