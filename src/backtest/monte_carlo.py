from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestEngine, BacktestResult, Strategy


@dataclass
class MonteCarloConfig:
    n_simulations: int = 200
    noise_bps: float = 5.0
    delay_bars: tuple[int, int] = (0, 3)
    seed: int = 42


@dataclass
class MonteCarloResult:
    base_equity: pd.DataFrame
    simulations: list[pd.Series]
    percentile_curves: dict[str, pd.Series]
    summary_metrics: pd.DataFrame


def _permute_signals(
    signals: pd.Series,
    noise_bps: float,
    delay_bars: tuple[int, int],
    rng: np.random.Generator,
) -> pd.Series:
    perturbed = signals.copy().astype(float)
    if noise_bps > 0:
        noise = rng.normal(0, noise_bps / 10_000, len(perturbed))
        perturbed = (perturbed + noise).clip(-1, 1)
    delay = rng.integers(delay_bars[0], delay_bars[1] + 1)
    if delay > 0:
        perturbed = perturbed.shift(delay).fillna(0)
    return perturbed


def monte_carlo_simulate(
    df: pd.DataFrame,
    strategy: Strategy,
    config: MonteCarloConfig | None = None,
    engine_config: BacktestConfig | None = None,
    verbose: bool = True,
) -> MonteCarloResult:
    cfg = config or MonteCarloConfig()
    eng_cfg = engine_config or BacktestConfig()
    engine = BacktestEngine(eng_cfg)
    rng = np.random.default_rng(cfg.seed)

    base_result = engine.run(df, strategy)
    base_equity = base_result.equity_curve.set_index("timestamp")["equity"]
    base_equity.name = "base"

    if verbose:
        print(f"Monte Carlo: {cfg.n_simulations} simulations | noise={cfg.noise_bps}bps | delay={cfg.delay_bars}")

    original_signals = strategy.generate_signals(df)

    class _PerturbedStrategy:
        def __init__(self, perturbed):
            self.values = perturbed.values.astype(float)
        def generate_signals(self, df):
            n_out = len(df)
            vals = np.zeros(n_out, dtype=float)
            n_copy = min(len(self.values), n_out)
            vals[:n_copy] = self.values[:n_copy]
            return pd.Series(vals, index=df.index)

    sim_equities: list[pd.Series] = []
    for i in range(cfg.n_simulations):
        perturbed = _permute_signals(original_signals, cfg.noise_bps, cfg.delay_bars, rng)
        pert_strat = _PerturbedStrategy(perturbed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engine.run(df, pert_strat)
        eq = result.equity_curve.set_index("timestamp")["equity"]
        eq.name = f"sim_{i}"
        sim_equities.append(eq)

    if verbose:
        print(f"  Done. {len(sim_equities)} simulation paths generated.")

    all_eq = pd.concat([base_equity] + sim_equities, axis=1)
    percentiles = {}
    for pct in [5, 25, 50, 75, 95]:
        percentiles[f"p{pct}"] = all_eq.iloc[:, 1:].apply(
            lambda row: np.percentile(row.dropna(), pct) if row.dropna().size > 0 else np.nan, axis=1
        )

    summary_rows = []
    for col in all_eq.columns:
        curve = all_eq[col].dropna()
        if len(curve) < 2:
            continue
        rets = curve.pct_change().dropna()
        summary_rows.append({
            "name": col,
            "final_equity": float(curve.iloc[-1]),
            "total_return_pct": (float(curve.iloc[-1]) / float(curve.iloc[0]) - 1) * 100,
            "max_drawdown_pct": float((curve / curve.cummax() - 1).min() * 100),
            "sharpe": float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0,
            "volatility_pct": float(rets.std() * (252 ** 0.5) * 100),
        })

    return MonteCarloResult(
        base_equity=base_result.equity_curve,
        simulations=sim_equities,
        percentile_curves={k: v for k, v in percentiles.items()},
        summary_metrics=pd.DataFrame(summary_rows),
    )
