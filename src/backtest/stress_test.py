from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestEngine, BacktestResult, Strategy


@dataclass
class StressScenario:
    name: str
    start: str
    end: str
    description: str = ""


STRESS_SCENARIOS = [
    StressScenario("covid_crash", "2020-02-19", "2020-03-23", "COVID-19 crash"),
    StressScenario("covid_full", "2020-01-01", "2020-06-30", "COVID-19 full period"),
    StressScenario("financial_crisis", "2008-09-01", "2009-03-09", "2008 Financial Crisis"),
    StressScenario("bear_market_2022", "2022-01-03", "2022-10-12", "2022 bear market"),
    StressScenario("flash_crash_2010", "2010-05-06", "2010-05-07", "Flash Crash 2010"),
    StressScenario("volmageddon_2018", "2018-01-26", "2018-02-09", "Volmageddon 2018"),
    StressScenario("taper_tantrum", "2013-05-22", "2013-06-25", "Taper Tantrum 2013"),
    StressScenario("dotcom_burst", "2000-03-10", "2002-10-09", "Dot-com bust"),
]


@dataclass
class StressTestResult:
    scenario_results: dict[str, dict[str, Any]]
    summary: pd.DataFrame
    benchmark_summary: pd.DataFrame


def _slice_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    ts_col = "timestamp" if "timestamp" in df.columns else "Date"
    if ts_col not in df.columns:
        ts_col = df.index.name if df.index.name else "timestamp"
        df = df.reset_index()

    if ts_col not in df.columns:
        raise ValueError("Cannot find timestamp column")

    mask = (pd.to_datetime(df[ts_col]) >= start) & (pd.to_datetime(df[ts_col]) <= end)
    sliced = df.loc[mask].copy()
    return sliced


def run_stress_tests(
    df: pd.DataFrame,
    strategy: Strategy,
    scenarios: list[StressScenario] | None = None,
    benchmark_df: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    verbose: bool = True,
) -> StressTestResult:
    scenarios = scenarios or STRESS_SCENARIOS
    engine = BacktestEngine(config)

    scenario_results: dict[str, dict] = {}
    for scenario in scenarios:
        try:
            test_df = _slice_df(df, scenario.start, scenario.end)
            if len(test_df) < 5:
                scenario_results[scenario.name] = {"error": "Insufficient data", "scenario": scenario}
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = engine.run(test_df, strategy)

            bench_metrics = {}
            if benchmark_df is not None:
                bench_test = _slice_df(benchmark_df, scenario.start, scenario.end)
                if len(bench_test) >= 5:
                    bench_result = _benchmark_metrics(bench_test, config)
                    bench_metrics = bench_result

            scenario_results[scenario.name] = {
                "scenario": scenario,
                "return_pct": result.metrics.get("total_return_pct", 0.0),
                "max_drawdown_pct": result.metrics.get("max_drawdown_pct", 0.0),
                "sharpe": result.metrics.get("sharpe", 0.0),
                "trades": result.metrics.get("trade_count", 0.0),
                "benchmark_return_pct": bench_metrics.get("total_return_pct", None),
                "benchmark_drawdown_pct": bench_metrics.get("max_drawdown_pct", None),
            }

            if verbose:
                print(
                    f"  {scenario.name:25s} | ret={result.metrics.get('total_return_pct', 0):+7.2f}% "
                    f"dd={result.metrics.get('max_drawdown_pct', 0):+6.2f}% "
                    f"sharpe={result.metrics.get('sharpe', 0):+5.2f}"
                )

        except Exception as e:
            scenario_results[scenario.name] = {"error": str(e), "scenario": scenario}
            if verbose:
                print(f"  {scenario.name:25s} | [!] {e}")

    summary = pd.DataFrame([
        {
            "scenario": k,
            "description": v["scenario"].description,
            "period": f"{v['scenario'].start} -> {v['scenario'].end}",
            **{key: val for key, val in v.items() if key not in ("scenario", "error")},
        }
        for k, v in scenario_results.items() if "error" not in v
    ])

    bench_summary = pd.DataFrame()
    if benchmark_df is not None:
        bench_rows = []
        for scenario in scenarios:
            bench_test = _slice_df(benchmark_df, scenario.start, scenario.end)
            if len(bench_test) >= 5:
                m = _benchmark_metrics(bench_test, config)
                bench_rows.append({
                    "scenario": scenario.name,
                    "description": scenario.description,
                    "return_pct": m.get("total_return_pct", 0),
                    "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                })
        bench_summary = pd.DataFrame(bench_rows)

    return StressTestResult(
        scenario_results=scenario_results,
        summary=summary,
        benchmark_summary=bench_summary,
    )


def _benchmark_metrics(df: pd.DataFrame, config: BacktestConfig | None = None) -> dict[str, float]:
    cfg = config or BacktestConfig()
    close = "close" if "close" in df.columns else "Close"
    if close not in df.columns:
        return {}
    prices = pd.to_numeric(df[close], errors="coerce").dropna()
    if len(prices) < 2:
        return {}
    curve = prices / prices.iloc[0] * cfg.initial_cash
    returns = curve.pct_change().dropna()
    dd = (curve / curve.cummax() - 1).min() * 100
    sharpe = float(returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
    return {
        "total_return_pct": (float(curve.iloc[-1]) / float(curve.iloc[0]) - 1) * 100,
        "max_drawdown_pct": float(dd),
        "sharpe": sharpe,
    }
