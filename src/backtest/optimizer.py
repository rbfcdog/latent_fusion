from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestEngine, BacktestResult, Strategy


@dataclass
class ParamGrid:
    values: list[Any]
    name: str = ""


@dataclass
class WalkForwardConfig:
    train_window: int = 500
    test_window: int = 100
    min_train: int = 200
    purge_bars: int = 10
    embargo_bars: int = 0
    window_mode: str = "rolling"
    hybrid_overlap: float = 0.5


@dataclass
class GridSearchResult:
    best_params: dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    param_heatmap: dict[str, pd.DataFrame]


@dataclass
class WalkForwardResult:
    fold_results: list[dict[str, Any]]
    aggregate_metrics: dict[str, float]
    equity_curve_combined: pd.DataFrame


def _build_strategy_class(base_class: type, param_dict: dict[str, Any]) -> Strategy:
    class _ParamStrategy(base_class):
        def __init__(self, **kwargs):
            merged = {**param_dict, **kwargs}
            super().__init__(**merged)
    return _ParamStrategy(**param_dict)


def grid_search(
    df: pd.DataFrame,
    strategy_class: type,
    param_grid: dict[str, list[Any]],
    metric: str = "sharpe",
    config: BacktestConfig | None = None,
    verbose: bool = True,
) -> GridSearchResult:
    engine = BacktestEngine(config)
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    results: list[dict] = []

    if verbose:
        print(f"Grid search: {len(combinations)} combinations | metric={metric}")

    for combo in combinations:
        params = dict(zip(keys, combo))
        try:
            strategy = strategy_class(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = engine.run(df, strategy)
            results.append({
                **params,
                **{f"metric_{k}": v for k, v in result.metrics.items()},
                "_score": result.metrics.get(metric, 0),
            })
        except Exception as e:
            if verbose:
                print(f"  [!] {params}: {e}")
            continue

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No successful grid search results")

    best_idx = results_df["_score"].idxmax()
    best_params = {k: results_df.loc[best_idx, k] for k in keys}
    best_score = float(results_df.loc[best_idx, "_score"])

    if verbose:
        print(f"  Best: {best_params} | {metric}={best_score:.4f}")

    heatmaps = {}
    if len(keys) == 2:
        pivot = results_df.pivot_table(
            values="_score", index=keys[0], columns=keys[1], aggfunc="mean"
        )
        heatmaps["score"] = pivot
    elif len(keys) == 1:
        heatmaps["score"] = results_df.set_index(keys[0])[["_score"]]

    return GridSearchResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results_df,
        param_heatmap=heatmaps,
    )


def walk_forward_optimize(
    df: pd.DataFrame,
    strategy_class: type,
    param_grid: dict[str, list[Any]],
    wf_config: WalkForwardConfig | None = None,
    metric: str = "sharpe",
    engine_config: BacktestConfig | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    cfg = wf_config or WalkForwardConfig()
    n = len(df)
    fold_results: list[dict] = []
    test_equities: list[pd.Series] = []
    all_signals: list[pd.Series] = []

    i = cfg.min_train
    fold_num = 0
    while i + cfg.test_window <= n:
        train_end = i
        test_start = i + cfg.purge_bars
        test_end = min(test_start + cfg.test_window, n)

        if cfg.window_mode == "expanding":
            train_start_idx = 0
        elif cfg.window_mode == "hybrid":
            overlap = min(cfg.hybrid_overlap, 1.0)
            rolling_start = max(0, train_end - cfg.train_window)
            train_start_idx = int(rolling_start * (1.0 - overlap))
        else:
            train_start_idx = max(0, train_end - cfg.train_window)

        train_df = df.iloc[train_start_idx:train_end]
        test_df = df.iloc[test_start:test_end]

        if len(train_df) < cfg.min_train:
            i = test_end
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs = grid_search(train_df, strategy_class, param_grid, metric, engine_config, verbose=False)

        strategy = strategy_class(**gs.best_params)
        engine = BacktestEngine(engine_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engine.run(test_df, strategy)

        fold_num += 1
        fold_results.append({
            "fold": fold_num,
            "train_start": train_df.iloc[0]["timestamp"] if "timestamp" in train_df.columns else train_df.index[0],
            "train_end": train_df.iloc[-1]["timestamp"] if "timestamp" in train_df.columns else train_df.index[-1],
            "test_start": test_df.iloc[0]["timestamp"] if "timestamp" in test_df.columns else test_df.index[0],
            "test_end": test_df.iloc[-1]["timestamp"] if "timestamp" in test_df.columns else test_df.index[-1],
            "best_params": gs.best_params,
            "train_score": gs.best_score,
            **{f"test_{k}": v for k, v in result.metrics.items()},
        })

        eq = result.equity_curve.set_index("timestamp")["equity"]
        eq_normalized = eq / eq.iloc[0] * engine_config.initial_cash if engine_config else eq / eq.iloc[0] * 10000
        test_equities.append(eq_normalized)

        if verbose:
            print(
                f"  Fold {fold_num}: train={train_df.index[0]}->{train_df.index[-1]} "
                f"test={test_df.index[0]}->{test_df.index[-1]} "
                f"params={gs.best_params} "
                f"test_sharpe={result.metrics.get('sharpe', 0):.3f}"
            )

        i = test_end + cfg.embargo_bars

    combined = pd.concat(test_equities) if test_equities else pd.Series(dtype=float)
    aggregate = {}
    if len(fold_results) > 0:
        for key in fold_results[0]:
            if key.startswith("test_") and isinstance(fold_results[0][key], (int, float, np.floating, np.integer)):
                vals = [f[key] for f in fold_results if key in f and isinstance(f[key], (int, float, np.floating, np.integer))]
                if vals:
                    aggregate[key.replace("test_", "avg_")] = float(np.mean(vals))
                    aggregate[key.replace("test_", "std_")] = float(np.std(vals))

    if verbose and fold_results:
        print(f"\n  Walk-forward: {len(fold_results)} folds | avg_sharpe={aggregate.get('avg_sharpe', 0):.3f}")

    return WalkForwardResult(
        fold_results=fold_results,
        aggregate_metrics=aggregate,
        equity_curve_combined=pd.DataFrame({"equity": combined}) if not combined.empty else pd.DataFrame(),
    )
