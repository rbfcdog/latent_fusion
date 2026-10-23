import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.visualization import BG, CYAN, GOLD, GREEN, ORANGE, PANEL, PURPLE, RED, WHITE
from src.fusion.llama_advanced_agent import apply_hysteresis_guard, weekly_decision_dates
from src.fusion.llama_news_backtest import buy_and_hold, load_news, numeric_actions

BASE = ROOT / "data" / "llama_news_api"
PRICES = pd.read_parquet(BASE / "prices_2024_2026.parquet")
VOLUMES = pd.read_parquet(BASE / "volumes_2024_2026.parquet")
NEWS = load_news(ROOT / "data" / "live_news" / "capture_filtered.jsonl")
TICKERS = list(PRICES.columns)
OUT_DIR = ROOT / "outputs" / "advisor_presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_academic_metrics(net_series, bench_series, turnover, fee_drag_pp):
    n = len(net_series)
    years = n / 252.0
    r_daily = net_series.pct_change().fillna(0.0).to_numpy()
    b_daily = bench_series.pct_change().fillna(0.0).to_numpy()

    total_return = float(net_series.iloc[-1] - 1.0)
    cagr = float(net_series.iloc[-1] ** (1.0 / years) - 1.0) if net_series.iloc[-1] > 0 else -1.0
    ann_vol = float(np.std(r_daily, ddof=1) * math.sqrt(252))

    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    neg_r = r_daily[r_daily < 0]
    downside_vol = float(np.sqrt(np.mean(neg_r ** 2)) * math.sqrt(252)) if len(neg_r) > 0 else 1e-6
    sortino = cagr / downside_vol if downside_vol > 0 else 0.0

    peaks = np.maximum.accumulate(net_series.to_numpy())
    dd = (net_series.to_numpy() - peaks) / peaks
    max_dd = float(np.min(dd))
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0

    var_b = float(np.var(b_daily, ddof=1))
    beta = float(np.cov(r_daily, b_daily)[0, 1] / var_b) if var_b > 0 else 1.0
    alpha_ann = float((np.mean(r_daily) - beta * np.mean(b_daily)) * 252)

    diff = r_daily - b_daily
    std_diff = float(np.std(diff, ddof=1))
    t_stat = float(np.mean(diff) / (std_diff / math.sqrt(n))) if std_diff > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))

    return {
        "Total Net": total_return,
        "CAGR": cagr,
        "Vol Anual": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": max_dd,
        "Alpha Anual": alpha_ann,
        "Beta": beta,
        "Turnover": turnover,
        "Fee Drag": fee_drag_pp,
        "t-stat": t_stat,
        "p-valor": p_val
    }


def run_unified_simulation(actions_map, weekly_dates, cost_bps=10):
    sessions = PRICES.index
    fill_map = {sessions[sessions.get_loc(d) + 1]: d for d in weekly_dates}
    start = min(fill_map)
    held = set()
    weights = np.zeros(len(TICKERS))
    daily = PRICES.pct_change().fillna(0.0)
    net = gross = 1.0
    rows = []
    turnover_total = 0.0

    for session in sessions[sessions.get_loc(start):]:
        r_vec = daily.loc[session].to_numpy(dtype=float)
        g = 1.0 + float(weights @ r_vec)
        gross *= g
        net *= g
        weights = weights * (1 + r_vec) / g if g > 0 else weights

        if session in fill_map:
            d = fill_map[session]
            acts = actions_map.get(d, {t: "HOLD" for t in TICKERS})
            target_held = {t for t in held if acts.get(t) != "SELL"} | {t for t in acts if acts.get(t) == "BUY"}
            target_weights = np.zeros(len(TICKERS))
            if target_held:
                target_weights[[TICKERS.index(t) for t in target_held]] = 1.0 / len(target_held)

            turn = float(np.abs(target_weights - weights).sum())
            turnover_total += turn
            net *= (1.0 - (cost_bps / 10_000) * turn)
            weights = target_weights
            held = target_held

        rows.append({"session": session, "net": net, "gross": gross, "held": len(held)})

    df = pd.DataFrame(rows).set_index("session")
    fee_drag = (df["gross"].iloc[-1] - df["net"].iloc[-1]) * 100
    return df, turnover_total, fee_drag


def main():
    weekly_dates = weekly_decision_dates(PRICES, "2024-08-01", str(PRICES.index[-1].date()))
    sessions = PRICES.index
    first_fill = sessions[sessions.get_loc(weekly_dates[0]) + 1]
    cost_bps = 10

    # True Buy and Hold Benchmark
    bench = buy_and_hold(PRICES, TICKERS, first_fill, cost_bps=cost_bps)

    # 1. Momentum Naive (Top 8 semanal)
    held_s = set()
    acts_naive = {}
    for d in weekly_dates:
        acts = numeric_actions(PRICES, d, TICKERS, held_s, top_k=8)
        acts_naive[d] = acts
        held_s = {t for t in held_s if acts[t] != "SELL"} | {t for t in acts if acts[t] == "BUY"}

    # 2. Histerese Estatística Pura (Top 4 Compra / Bottom 8 Venda)
    held_h = set()
    acts_hyst = {}
    for d in weekly_dates:
        loc = PRICES.index.get_loc(d)
        raw = acts_naive[d]
        guarded, _ = apply_hysteresis_guard(raw, held_h, PRICES, loc, TICKERS)
        acts_hyst[d] = guarded
        held_h = {t for t in held_h if guarded.get(t) != "SELL"} | {t for t in guarded if guarded.get(t) == "BUY"}

    # 3. Llama 3.1 8B (Notícias Filtradas)
    p_8b = BASE / "openrouter-deepinfra-llama-3.1-8b_filtered" / "decisions.jsonl"
    rows_8b = [json.loads(l) for l in p_8b.read_text(encoding="utf-8").splitlines() if l.strip()]
    acts_8b = {pd.Timestamp(r["decision_date"]): r["actions"] for r in rows_8b if r["arm"] == "news"}

    # 4. Llama 3.1 70B (Notícias Filtradas)
    p_70b = BASE / "openrouter-deepinfra-llama-3.1-70b_filtered" / "decisions.jsonl"
    rows_70b = [json.loads(l) for l in p_70b.read_text(encoding="utf-8").splitlines() if l.strip()]
    acts_70b = {pd.Timestamp(r["decision_date"]): r["actions"] for r in rows_70b if r["arm"] == "news"}

    # 5. Llama 3.1 70B Tool Agent (215 chamadas interativas)
    p_tool = BASE / "tool_agent_70b" / "decisions.jsonl"
    rows_tool = [json.loads(l) for l in p_tool.read_text(encoding="utf-8").splitlines() if l.strip()]
    acts_tool = {pd.Timestamp(r["decision_date"]): r["actions"] for r in rows_tool}

    strategies = {
        "Comprar e Manter (Benchmark)": None,
        "Momentum Naive (Sem Histerese)": acts_naive,
        "Histerese Estatística (Top 4 / Bottom 8)": acts_hyst,
        "Llama 3.1 8B (Notícias Filtradas)": acts_8b,
        "Llama 3.1 70B (Notícias Filtradas)": acts_70b,
        "Llama 3.1 70B Tool Agent (Com Ferramentas)": acts_tool,
    }

    metrics_table = {}
    curves = {}

    for name, acts in strategies.items():
        if acts is None:
            metrics = compute_academic_metrics(bench, bench, turnover=1.0, fee_drag_pp=0.10)
            metrics_table[name] = metrics
            curves[name] = bench
        else:
            df, turn, drag = run_unified_simulation(acts, weekly_dates, cost_bps=cost_bps)
            metrics = compute_academic_metrics(df["net"], bench, turnover=turn, fee_drag_pp=drag)
            metrics_table[name] = metrics
            curves[name] = df["net"]

    df_metrics = pd.DataFrame(metrics_table).T

    print("\n" + "=" * 105)
    print("BENCHMARK ACADÊMICO DE PERFORMANCE 100% HOMOGÊNEO (UNICAMP IC / AUDITORIA FORENSE)")
    print("=" * 105)
    formatters = {
        "Total Net": "{:+6.2%}".format,
        "CAGR": "{:+6.2%}".format,
        "Vol Anual": "{:6.2%}".format,
        "Sharpe": "{:6.2f}".format,
        "Sortino": "{:6.2f}".format,
        "Max DD": "{:6.2%}".format,
        "Alpha Anual": "{:+6.2%}".format,
        "Beta": "{:5.2f}".format,
        "Turnover": "{:6.1f}".format,
        "Fee Drag": "{:5.2f}pp".format,
        "t-stat": "{:+5.2f}".format,
        "p-valor": "{:6.3f}".format,
    }
    print(df_metrics.to_string(formatters=formatters))
    print("=" * 105)

    # Export LaTeX Table
    latex_file = OUT_DIR / "tabela_ic_orientador.tex"
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write("% Tabela gerada automaticamente para o relatorio de IC\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparacao de Desempenho Homogenea: Estatistico vs LLM (2024--2026)}\n")
        f.write("\\label{tab:benchmark_ic}\n")
        f.write("\\small\n")
        f.write(df_metrics.to_latex(formatters=formatters, escape=False))
        f.write("\\end{table}\n")
    print(f"\nTabela LaTeX exportada para: {latex_file}")

    # Plot Publication Chart
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_title("Desempenho Comparativo 100% Homogêneo: Estatístico vs LLM vs Benchmark", fontsize=15, color=WHITE)
    ax.set_ylabel("Patrimônio Líquido Acumulado (Base 1.0, com custos de 10 bps)", fontsize=13, color=WHITE)
    ax.tick_params(colors=WHITE, labelsize=11)
    ax.grid(alpha=0.12)

    palette = [GOLD, RED, PURPLE, GREEN, CYAN, ORANGE]
    for idx, (name, curve) in enumerate(curves.items()):
        color = palette[idx % len(palette)]
        lw = 2.6 if "Histerese" in name or "Benchmark" in name else 1.8
        ls = "--" if "Naive" in name else "-"
        ret_val = metrics_table[name]["Total Net"]
        sharpe_val = metrics_table[name]["Sharpe"]
        ax.plot(curve, color=color, label=f"{name} [{ret_val:+5.1%} | Sharpe {sharpe_val:.2f}]", linewidth=lw, linestyle=ls)

    ax.axhline(1.0, color=WHITE, alpha=0.2, linestyle=":")
    ax.legend(fontsize=10, loc="upper left")

    out_png = OUT_DIR / "benchmark_academico_ic.png"
    fig.savefig(out_png, dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Gráfico de publicação salvo: {out_png}\n")


if __name__ == "__main__":
    main()
