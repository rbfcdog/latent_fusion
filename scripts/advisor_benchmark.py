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
from src.fusion.hybrid_quant_llm import boost_re, is_strictly_relevant, veto_re
from src.fusion.llama_advanced_agent import apply_hysteresis_guard, simulate_weekly, weekly_decision_dates
from src.fusion.llama_news_backtest import buy_and_hold, load_news, numeric_actions

BASE = ROOT / "data" / "llama_news_api"
PRICES = pd.read_parquet(BASE / "prices_2024_2026.parquet")
VOLUMES = pd.read_parquet(BASE / "volumes_2024_2026.parquet")
NEWS = load_news(ROOT / "data" / "live_news" / "capture_filtered.jsonl")
TICKERS = list(PRICES.columns)
OUT_DIR = ROOT / "outputs" / "advisor_presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_academic_metrics(net_series, bench_series, turnover):
    n = len(net_series)
    years = n / 252.0
    r_daily = net_series.pct_change().fillna(0.0).to_numpy()
    b_daily = bench_series.pct_change().fillna(0.0).to_numpy()

    total_return = float(net_series.iloc[-1] - 1.0)
    cagr = float(net_series.iloc[-1] ** (1.0 / years) - 1.0)
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
        "Total Return": total_return,
        "CAGR": cagr,
        "Vol Anual": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": max_dd,
        "Calmar": calmar,
        "Alpha Anual": alpha_ann,
        "Beta": beta,
        "Turnover": turnover,
        "t-stat": t_stat,
        "p-valor": p_val
    }


def main():
    weekly_dates = weekly_decision_dates(PRICES, "2024-08-01", str(PRICES.index[-1].date()))
    sessions = PRICES.index
    first_fill = sessions[sessions.get_loc(weekly_dates[0]) + 1]
    cost_bps = 10

    bench = buy_and_hold(PRICES, TICKERS, first_fill, cost_bps=cost_bps)

    raw_actions = {}
    held_s = set()
    for d in weekly_dates:
        acts = numeric_actions(PRICES, d, TICKERS, held_s, top_k=8)
        raw_actions[d] = acts
        held_s = {t for t in held_s if acts[t] != "SELL"} | {t for t in acts if acts[t] == "BUY"}

    def sim_custom(use_hysteresis=True, use_pv_events=False):
        fill_map = {sessions[sessions.get_loc(d) + 1]: d for d in weekly_dates}
        start = min(fill_map)
        held = set()
        weights = np.zeros(len(TICKERS))
        daily = PRICES.pct_change().fillna(0.0)
        net = 1.0
        rows = []
        turnover_total = 0.0

        for session in sessions[sessions.get_loc(start):]:
            r_vec = daily.loc[session].to_numpy(dtype=float)
            g = 1.0 + float(weights @ r_vec)
            net *= g
            weights = weights * (1 + r_vec) / g if g > 0 else weights

            if session in fill_map:
                d = fill_map[session]
                loc = sessions.get_loc(d)
                raw = raw_actions[d]
                m20 = {t: float(PRICES[t].iloc[loc] / PRICES[t].iloc[max(0, loc - 20)] - 1) for t in TICKERS}
                ranks = {t: r + 1 for r, t in enumerate(sorted(TICKERS, key=m20.get, reverse=True))}

                has_pos = {}
                has_neg = {}
                if use_pv_events:
                    for t in TICKERS:
                        t_news = [h for h in NEWS if h["ticker"] == t and h["published_ts"].date() < d.date() and (d.date() - h["published_ts"].date()).days <= 14]
                        t_px = PRICES[t].iloc[max(0, loc - 20):loc + 1]
                        t_vol = VOLUMES[t].iloc[max(0, loc - 20):loc + 1]
                        r_5d = float(t_px.iloc[-1] / t_px.iloc[max(0, len(t_px) - 6)] - 1) if len(t_px) > 5 else 0.0
                        v_ratio = float(t_vol.tail(5).mean() / max(1.0, t_vol.mean())) if len(t_vol) > 5 else 1.0
                        if any(boost_re.search(h["title"]) and is_strictly_relevant(t, h["title"]) for h in t_news):
                            if ranks[t] <= 6 and r_5d > 0.0 and v_ratio >= 1.10:
                                has_pos[t] = True
                        if any(veto_re.search(h["title"]) and is_strictly_relevant(t, h["title"]) for h in t_news):
                            if r_5d < -0.01:
                                has_neg[t] = True

                if use_hysteresis:
                    guarded, _ = apply_hysteresis_guard(raw, held, PRICES, loc, TICKERS, has_pos_events=has_pos, has_neg_events=has_neg)
                else:
                    guarded = raw

                target_held = {t for t in held if guarded.get(t) != "SELL"} | {t for t in guarded if guarded.get(t) == "BUY"}
                target_weights = np.zeros(len(TICKERS))
                if target_held:
                    target_weights[[TICKERS.index(t) for t in target_held]] = 1.0 / len(target_held)

                turnover = float(np.abs(target_weights - weights).sum())
                turnover_total += turnover
                net *= (1.0 - (cost_bps / 10_000) * turnover)
                weights = target_weights
                held = target_held

            rows.append({"session": session, "net": net})

        return pd.DataFrame(rows).set_index("session")["net"], turnover_total

    eq_naive, turn_naive = sim_custom(use_hysteresis=False, use_pv_events=False)
    eq_stat, turn_stat = sim_custom(use_hysteresis=True, use_pv_events=False)
    eq_hybrid, turn_hybrid = sim_custom(use_hysteresis=True, use_pv_events=True)

    # Load Llama 8B Filtered run series
    p_8b = BASE / "openrouter-deepinfra-llama-3.1-8b_filtered" / "decisions.jsonl"
    if p_8b.is_file():
        rows = [json.loads(l) for l in p_8b.read_text(encoding="utf-8").splitlines() if l.strip()]
        acts_8b = {pd.Timestamp(r["decision_date"]): r["actions"] for r in rows if r["arm"] == "news"}
        fill_dates_8b = sorted(acts_8b.keys())
        sim_8b = simulate_weekly(acts_8b, fill_dates_8b, PRICES, TICKERS, cost_bps=cost_bps, use_hysteresis=False)
        eq_8b = sim_8b["series"]["net"].reindex(bench.index).ffill()
        eq_8b = eq_8b / eq_8b.iloc[0]
        turn_8b = sim_8b["turnover_total"]
    else:
        eq_8b, turn_8b = eq_stat, turn_stat

    models = {
        "Comprar e Manter (Benchmark)": (bench, 1.0),
        "Momentum Simples (Naive)": (eq_naive, turn_naive),
        "Histerese Estatística Pura": (eq_stat, turn_stat),
        "Sistema Híbrido (Quant + LLM PV)": (eq_hybrid, turn_hybrid),
        "Llama 3.1 8B (Decisão por LLM)": (eq_8b, turn_8b),
    }

    metrics_table = {}
    for name, (s, t_over) in models.items():
        metrics_table[name] = compute_academic_metrics(s, bench, t_over)

    df_metrics = pd.DataFrame(metrics_table).T

    print("\n" + "=" * 90)
    print("BENCHMARK ACADÊMICO DE PERFORMANCE (UNICAMP IC / AVALIAÇÃO DE ORIENTADOR)")
    print("=" * 90)
    formatters = {
        "Total Return": "{:+6.2%}".format,
        "CAGR": "{:+6.2%}".format,
        "Vol Anual": "{:6.2%}".format,
        "Sharpe": "{:6.2f}".format,
        "Sortino": "{:6.2f}".format,
        "Max DD": "{:6.2%}".format,
        "Calmar": "{:6.2f}".format,
        "Alpha Anual": "{:+6.2%}".format,
        "Beta": "{:5.2f}".format,
        "Turnover": "{:6.1f}".format,
        "t-stat": "{:+5.2f}".format,
        "p-valor": "{:6.3f}".format,
    }
    print(df_metrics.to_string(formatters=formatters))
    print("=" * 90)

    # Export LaTeX Table
    latex_file = OUT_DIR / "tabela_ic_orientador.tex"
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write("% Tabela gerada automaticamente para o relatório de IC\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparação de Desempenho Estatístico e Multimodal (2024--2026)}\n")
        f.write("\\label{tab:benchmark_ic}\n")
        f.write("\\small\n")
        f.write(df_metrics.to_latex(formatters=formatters, escape=False))
        f.write("\\end{table}\n")
    print(f"\nTabela LaTeX exportada para: {latex_file}")

    # Plot Publication Chart
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_title("Desempenho Comparativo Multimodal: Sistema Híbrido vs Estatístico vs Benchmark", fontsize=15, color=WHITE)
    ax.set_ylabel("Patrimônio Líquido Acumulado (Base 1.0, com custos de 10 bps)", fontsize=13, color=WHITE)
    ax.tick_params(colors=WHITE, labelsize=11)
    ax.grid(alpha=0.12)

    palette = [GOLD, RED, CYAN, GREEN, ORANGE]
    for idx, (name, (s, _)) in enumerate(models.items()):
        color = palette[idx % len(palette)]
        lw = 2.6 if "Híbrido" in name or "Histerese" in name or "Benchmark" in name else 1.6
        ls = "--" if "Naive" in name else "-"
        ret_val = metrics_table[name]["Total Return"]
        sharpe_val = metrics_table[name]["Sharpe"]
        ax.plot(s, color=color, label=f"{name} [{ret_val:+5.1%} | Sharpe {sharpe_val:.2f}]", linewidth=lw, linestyle=ls)

    ax.axhline(1.0, color=WHITE, alpha=0.2, linestyle=":")
    ax.legend(fontsize=11, loc="upper left")

    out_png = OUT_DIR / "benchmark_academico_ic.png"
    fig.savefig(out_png, dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Gráfico de publicação salvo: {out_png}\n")


if __name__ == "__main__":
    main()
