from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

GOLD = "#D4A843"
GREEN = "#00E676"
RED = "#FF1744"
CYAN = "#18FFFF"
WHITE = "#AAAAAA"
PURPLE = "#CE93D8"
ORANGE = "#FF9800"
BLUE = "#42A5F5"

PALETTE = [GOLD, CYAN, PURPLE, ORANGE, BLUE, GREEN, RED, WHITE]

BG = "#0d0d1a"
PANEL = "#1a1a2e"
GRID_ALPHA = 0.12


def _setup_dark(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.spines["bottom"].set_color(WHITE)
    ax.spines["left"].set_color(WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color="white", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color="white", fontsize=8)


def _save(fig, filepath, out_dir="images"):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def plot_equity_curve(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    title: str = "Equity Curve",
    out_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(BG)

    eq = equity_df.set_index("timestamp")["equity"] if "timestamp" in equity_df.columns else equity_df["equity"]
    ax.plot(eq.index, eq.values, color=GOLD, lw=2, label="Strategy")
    ax.fill_between(eq.index, eq.values, eq.iloc[0], where=eq.values >= eq.iloc[0], color=GREEN, alpha=0.08)
    ax.fill_between(eq.index, eq.values, eq.iloc[0], where=eq.values < eq.iloc[0], color=RED, alpha=0.08)

    if benchmark_df is not None:
        bench = benchmark_df.set_index("timestamp")["equity"] if "timestamp" in benchmark_df.columns else benchmark_df["equity"]
        ax.plot(bench.index, bench.values, color=WHITE, lw=1, alpha=0.5, ls="--", label="Benchmark")

    ax.axhline(equity_df["equity"].iloc[0] if "equity" in equity_df.columns else eq.iloc[0], color="white", ls="--", alpha=0.2)
    _setup_dark(ax, title)
    ax.legend(loc="upper left", fontsize=9, facecolor=PANEL, edgecolor="white", labelcolor="white")

    if out_path:
        _save(fig, out_path)
    return fig


def plot_monte_carlo_fan(
    mc_result,
    title: str = "Monte Carlo Simulation",
    out_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(BG)

    base = mc_result.base_equity.set_index("timestamp")["equity"]
    ax.plot(base.index, base.values, color=GOLD, lw=2.5, label="Base")

    pcts = mc_result.percentile_curves
    ax.fill_between(base.index, pcts["p5"], pcts["p95"], color=GOLD, alpha=0.08)
    ax.fill_between(base.index, pcts["p25"], pcts["p75"], color=GOLD, alpha=0.12)
    ax.plot(base.index, pcts["p50"], color=GOLD, lw=1, alpha=0.5, ls="--", label="Median")

    ax.axhline(base.iloc[0], color="white", ls="--", alpha=0.2)
    _setup_dark(ax, title)
    ax.legend(loc="upper left", fontsize=9, facecolor=PANEL, edgecolor="white", labelcolor="white")

    if out_path:
        _save(fig, out_path)
    return fig


def plot_stress_test(
    stress_result,
    title: str = "Stress Test Performance",
    out_path: str | None = None,
):
    df = stress_result.summary
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG)

    scenarios = df["scenario"].tolist()
    y = np.arange(len(scenarios))
    strategy_rets = df["return_pct"].values
    bench_rets = df.get("benchmark_return_pct", pd.Series([None] * len(scenarios))).values

    ax.barh(y - 0.15, strategy_rets, 0.3, color=GOLD, alpha=0.8, label="Strategy")
    has_bench = any(b is not None and not np.isnan(b) for b in bench_rets)
    if has_bench:
        ax.barh(y + 0.15, bench_rets, 0.3, color=WHITE, alpha=0.4, label="Benchmark")

    ax.set_yticks(y)
    ax.set_yticklabels(scenarios, color="white", fontsize=8)
    ax.axvline(0, color="white", lw=0.5)
    _setup_dark(ax, title, "Return %")
    ax.legend(loc="lower right", fontsize=8, facecolor=PANEL, edgecolor="white", labelcolor="white")

    for i, (v_s, v_b) in enumerate(zip(strategy_rets, bench_rets)):
        ax.text(v_s + (0.5 if v_s >= 0 else -3), i - 0.15, f"{v_s:.1f}%", va="center", color="white", fontsize=7)
        if has_bench and v_b is not None and not np.isnan(v_b):
            ax.text(v_b + (0.5 if v_b >= 0 else -3), i + 0.15, f"{v_b:.1f}%", va="center", color=WHITE, fontsize=7)

    if out_path:
        _save(fig, out_path)
    return fig


def plot_param_heatmap(
    gs_result,
    title: str = "Parameter Heatmap",
    out_path: str | None = None,
):
    heatmaps = gs_result.param_heatmap
    if not heatmaps:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)

    hm = heatmaps.get("score", list(heatmaps.values())[0])
    im = ax.imshow(hm.values, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(hm.columns)))
    ax.set_xticklabels(hm.columns, color="white", fontsize=9)
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(hm.index, color="white", fontsize=9)

    for i in range(len(hm.index)):
        for j in range(len(hm.columns)):
            ax.text(j, i, f"{hm.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    _setup_dark(ax, title)
    if out_path:
        _save(fig, out_path)
    return fig


def plot_walk_forward(
    wf_result,
    title: str = "Walk-Forward Performance",
    out_path: str | None = None,
):
    folds = wf_result.fold_results
    if not folds:
        return None

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG)

    fold_nums = [f["fold"] for f in folds]
    sharpes = [f.get("test_sharpe", 0) for f in folds]
    returns = [f.get("test_total_return_pct", 0) for f in folds]
    drawdowns = [abs(f.get("test_max_drawdown_pct", 0)) for f in folds]

    x = np.arange(len(folds))
    w = 0.25
    ax.bar(x - w, sharpes, w, color=GOLD, alpha=0.8, label="Sharpe")
    ax.bar(x, returns, w, color=CYAN, alpha=0.8, label="Return %")
    ax.bar(x + w, drawdowns, w, color=RED, alpha=0.6, label="Max DD %")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {n}" for n in fold_nums], color="white", fontsize=8)
    ax.axhline(0, color="white", lw=0.5)
    _setup_dark(ax, title)
    ax.legend(loc="upper right", fontsize=8, facecolor=PANEL, edgecolor="white", labelcolor="white")

    if out_path:
        _save(fig, out_path)
    return fig
