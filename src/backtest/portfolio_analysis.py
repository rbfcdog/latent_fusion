from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestEngine, Strategy
from .ticker_analysis import compute_ticker_stats


@dataclass
class WorstCaseReport:
    worst_return_ticker: str
    worst_return_pct: float
    worst_drawdown_ticker: str
    worst_drawdown_pct: float
    worst_sharpe_ticker: str
    worst_sharpe: float
    degrading_tickers: list[str]
    portfolio_impact_pct: float
    per_ticker_metrics: pd.DataFrame


def backtest_per_ticker(
    data: dict[str, pd.DataFrame] | pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    date_col: str = "timestamp",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    engine = BacktestEngine(config)
    rows = []

    if isinstance(data, dict):
        items = data.items()
    else:
        df = data.copy()
        if ticker_col not in df.columns and "symbol" in df.columns:
            df = df.rename(columns={"symbol": ticker_col})
        items = [(t, g) for t, g in df.groupby(ticker_col)]

    for ticker, df_t in items:
        try:
            result = engine.run(df_t, strategy)
            rows.append({
                "ticker": ticker,
                "return_pct": result.metrics["total_return_pct"],
                "sharpe": result.metrics["sharpe"],
                "max_drawdown_pct": result.metrics["max_drawdown_pct"],
                "bh_return_pct": result.metrics["bh_return_pct"],
                "excess_return_pct": result.metrics["excess_return_pct"],
                "trade_count": result.metrics["trade_count"],
                "volatility_pct": result.metrics.get("sortino", 0.0),
                "final_equity": result.metrics["final_equity"],
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def worst_case_analysis(
    per_ticker: pd.DataFrame,
    degradation_threshold: float = -5.0,
    max_dd_threshold: float = -40.0,
) -> WorstCaseReport:
    if per_ticker.empty:
        return WorstCaseReport(
            worst_return_ticker="",
            worst_return_pct=0.0,
            worst_drawdown_ticker="",
            worst_drawdown_pct=0.0,
            worst_sharpe_ticker="",
            worst_sharpe=0.0,
            degrading_tickers=[],
            portfolio_impact_pct=0.0,
            per_ticker_metrics=per_ticker,
        )

    worst_ret_idx = per_ticker["return_pct"].idxmin()
    worst_dd_idx = per_ticker["max_drawdown_pct"].idxmin()
    worst_sharpe_idx = per_ticker["sharpe"].idxmin()

    degrading = per_ticker[
        (per_ticker["excess_return_pct"] < degradation_threshold)
        | (per_ticker["max_drawdown_pct"] < max_dd_threshold)
    ]["ticker"].tolist()

    portfolio_impact = float(
        per_ticker[per_ticker["ticker"].isin(degrading)]["return_pct"].mean()
        if degrading
        else 0.0
    )

    return WorstCaseReport(
        worst_return_ticker=per_ticker.loc[worst_ret_idx, "ticker"],
        worst_return_pct=float(per_ticker.loc[worst_ret_idx, "return_pct"]),
        worst_drawdown_ticker=per_ticker.loc[worst_dd_idx, "ticker"],
        worst_drawdown_pct=float(per_ticker.loc[worst_dd_idx, "max_drawdown_pct"]),
        worst_sharpe_ticker=per_ticker.loc[worst_sharpe_idx, "ticker"],
        worst_sharpe=float(per_ticker.loc[worst_sharpe_idx, "sharpe"]),
        degrading_tickers=degrading,
        portfolio_impact_pct=portfolio_impact,
        per_ticker_metrics=per_ticker,
    )


def auto_exclude_tickers(
    per_ticker: pd.DataFrame,
    min_excess_return: float = -5.0,
    min_sharpe: float = -0.5,
    max_drawdown: float = -50.0,
) -> tuple[list[str], list[str]]:
    keep_mask = (
        (per_ticker["excess_return_pct"] >= min_excess_return)
        & (per_ticker["sharpe"] >= min_sharpe)
        & (per_ticker["max_drawdown_pct"] >= max_drawdown)
    )
    kept = per_ticker[keep_mask]["ticker"].tolist()
    excluded = per_ticker[~keep_mask]["ticker"].tolist()
    return kept, excluded
