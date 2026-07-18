from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.backtest.investor_profile import InvestorProfile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_bps: float = 4.0
    slippage_bps: float = 1.0
    spread_bps: float = 0.0
    tax_rate: float = 0.0
    periods_per_year: int = 365 * 24 * 12
    rebalance_freq: str = "daily"
    rebalance_day: int | None = None
    execution_delay_bars: int = 0
    min_volume: float = 0.0
    max_order_pct_volume: float = 0.0


@dataclass
class Trade:
    timestamp: pd.Timestamp
    side: str
    units: float
    price: float
    fee: float
    slippage: float
    cash_after: float
    position_after: float
    notional: float
    signal: float


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


class Strategy(Protocol):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series: ...


def _prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError("Expected column 'timestamp' or 'Date' in input data")

    required = {"open", "high", "low", "close"}
    rename_map = {}
    for col in required:
        if col not in df.columns:
            cap = col.capitalize()
            if cap in df.columns:
                rename_map[cap] = col
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = required - set(df.columns)
    if "close" in missing:
        raise ValueError(f"Missing required columns: {missing}")

    bars = df.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    for col in required:
        if col in bars.columns:
            bars[col] = pd.to_numeric(bars[col], errors="coerce")

    bars = bars.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    return bars


def _compute_metrics(
    equity_curve: pd.Series,
    trade_count: int,
    periods_per_year: int,
    benchmark_curve: pd.Series | None = None,
) -> dict[str, float]:
    if len(equity_curve) < 2:
        return {
            "final_equity": equity_curve.iloc[0] if len(equity_curve) else 0,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "trade_count": float(trade_count),
            "win_rate_pct": 0.0,
            "avg_trade_pct": 0.0,
            "profit_factor": 0.0,
            "capm_alpha": 0.0,
            "capm_beta": 0.0,
            "info_ratio": 0.0,
        }

    curve = equity_curve.astype(float)
    returns = curve.pct_change().fillna(0.0)
    n = len(curve)

    rolling_peak = curve.cummax()
    drawdown = (curve / rolling_peak) - 1.0
    max_dd = float(drawdown.min() * 100)

    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    ann_factor = periods_per_year ** 0.5

    sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * ann_factor if downside_std > 0 else 0.0

    total_return_pct = (float(curve.iloc[-1]) / float(curve.iloc[0]) - 1.0) * 100
    years = n / periods_per_year
    cagr = ((float(curve.iloc[-1]) / float(curve.iloc[0])) ** (1 / max(years, 0.01)) - 1) * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    capm_alpha = 0.0
    capm_beta = 0.0
    info_ratio = 0.0
    if benchmark_curve is not None and len(benchmark_curve) == len(curve):
        bench_returns = benchmark_curve.pct_change().fillna(0.0).astype(float)
        aligned = pd.DataFrame({"strategy": returns.values, "benchmark": bench_returns.values}).dropna()
        if len(aligned) > 10:
            cov = np.cov(aligned["strategy"], aligned["benchmark"])
            bench_var = float(np.var(aligned["benchmark"]))
            if bench_var > 1e-12:
                capm_beta = float(cov[0, 1] / bench_var)
            capm_alpha = float(aligned["strategy"].mean() - capm_beta * aligned["benchmark"].mean()) * periods_per_year * 100
            tracking_error = float((aligned["strategy"] - aligned["benchmark"]).std())
            info_ratio = float((aligned["strategy"].mean() - aligned["benchmark"].mean()) / tracking_error) * ann_factor if tracking_error > 0 else 0.0

    return {
        "final_equity": float(curve.iloc[-1]),
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "trade_count": float(trade_count),
        "win_rate_pct": 0.0,
        "avg_trade_pct": 0.0,
        "profit_factor": 0.0,
        "capm_alpha": capm_alpha,
        "capm_beta": capm_beta,
        "info_ratio": info_ratio,
    }


def _buy_and_hold_equity(bars: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    close = bars.set_index("timestamp")["close"]
    return close / close.iloc[0] * config.initial_cash


def decompose_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    aligned = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
    if len(aligned) < 10:
        return {"alpha_pct": 0.0, "beta": 0.0, "beta_return_pct": 0.0, "alpha_return_pct": 0.0, "r_squared": 0.0}

    cov = np.cov(aligned["strategy"], aligned["benchmark"])
    bench_var = float(np.var(aligned["benchmark"]))
    beta = float(cov[0, 1] / bench_var) if bench_var > 1e-12 else 0.0

    bench_annual = float(aligned["benchmark"].mean()) * periods_per_year
    strategy_annual = float(aligned["strategy"].mean()) * periods_per_year
    beta_return = beta * bench_annual * 100
    alpha = (strategy_annual - beta * bench_annual) * 100

    residuals = aligned["strategy"] - beta * aligned["benchmark"]
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((aligned["strategy"] - aligned["strategy"].mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "alpha_pct": alpha,
        "beta": beta,
        "beta_return_pct": beta_return,
        "alpha_return_pct": strategy_annual * 100 - beta_return,
        "r_squared": r_squared,
    }



def _rebalance_mask(timestamps: pd.Series, freq: str, rebalance_day: int | None) -> np.ndarray:
    n = len(timestamps)
    if freq == "daily":
        return np.ones(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    ts = pd.to_datetime(timestamps)
    if freq == "weekly":
        if rebalance_day is not None:
            mask |= (ts.dt.dayofweek == rebalance_day).values
        else:
            weeks = ts.dt.isocalendar().week.values.astype(int)
            years = ts.dt.year.values
            prev_week = -1
            prev_year = -1
            for i in range(n):
                w = weeks[i]; y = years[i]
                if w != prev_week or y != prev_year:
                    mask[i] = True
                    prev_week = w; prev_year = y
    elif freq == "monthly":
        if rebalance_day is not None:
            mask |= (ts.dt.day == rebalance_day).values
        else:
            months = ts.dt.month.values
            years = ts.dt.year.values
            prev_month = -1
            prev_year = -1
            for i in range(n):
                m = months[i]; y = years[i]
                if m != prev_month or y != prev_year:
                    mask[i] = True
                    prev_month = m; prev_year = y
    return mask

class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        benchmark_df: pd.DataFrame | None = None,
        profile: InvestorProfile | None = None,
    ) -> BacktestResult:
        bars = _prepare_bars(df)
        targets = strategy.generate_signals(bars).reindex(bars.index).fillna(0.0).clip(-1.0, 1.0)

        if self.config.execution_delay_bars > 0:
            targets = targets.shift(self.config.execution_delay_bars).fillna(0.0)

        rebalance_mask = _rebalance_mask(
            bars["timestamp"], self.config.rebalance_freq, self.config.rebalance_day
        )

        if profile is not None:
            targets = targets.apply(profile.apply_signal)

        cash = float(self.config.initial_cash)
        position_units = 0.0
        equity_peak = float(self.config.initial_cash)
        equity_rows: list[dict] = []
        trades: list[Trade] = []
        fee_rate = self.config.fee_bps / 10_000.0
        slippage_rate = self.config.slippage_bps / 10_000.0
        spread_rate = self.config.spread_bps / 10_000.0
        total_fees = 0.0
        total_slippage = 0.0
        total_spread = 0.0
        returns = bars.set_index("timestamp")["close"].pct_change().fillna(0.0)
        for pos, (_, row) in enumerate(bars.iterrows()):
            price = float(row["close"])
            if price <= 0:
                continue

            raw_target = float(targets.iloc[pos])
            target = raw_target
            equity_before = cash + position_units * price

            if profile is not None:
                vs = profile.vol_scale(returns, pos)
                target = raw_target * vs
                dd_guard = profile.drawdown_guard(equity_before, equity_peak)
                target = target * dd_guard

            if rebalance_mask[pos]:
                target_units = (equity_before * target) / price
            else:
                target_units = position_units
            delta_units = target_units - position_units

            if self.config.min_volume > 0 and "volume" in row.index:
                vol = float(row.get("volume", 0))
                if vol < self.config.min_volume:
                    delta_units = 0.0

            if self.config.max_order_pct_volume > 0 and "volume" in row.index:
                vol = float(row.get("volume", 0))
                max_units = vol * self.config.max_order_pct_volume / price
                if abs(delta_units) > max_units:
                    delta_units = np.sign(delta_units) * max_units

            if abs(delta_units) > 1e-12:
                side = "buy" if delta_units > 0 else "sell"
                slippage = slippage_rate if delta_units > 0 else -slippage_rate
                spread_cost = abs(delta_units) * price * spread_rate
                exec_price = price * (1 + slippage)
                notional = abs(delta_units) * exec_price
                fee = notional * fee_rate

                cash -= delta_units * exec_price
                cash -= fee
                cash -= spread_cost
                position_units += delta_units

                total_fees += fee
                total_slippage += abs(slippage * price * abs(delta_units))
                total_spread += spread_cost

                trades.append(Trade(
                    timestamp=row["timestamp"],
                    side=side,
                    units=abs(delta_units),
                    price=exec_price,
                    fee=fee,
                    slippage=abs(slippage * price),
                    cash_after=cash,
                    position_after=position_units,
                    notional=notional,
                    signal=target,
                ))

            equity = cash + position_units * price
            if equity > equity_peak:
                equity_peak = equity
            equity_rows.append({
                "timestamp": row["timestamp"],
                "close": price,
                "target": target,
                "position_units": position_units,
                "cash": cash,
                "equity": equity,
            })

        equity_df = pd.DataFrame(equity_rows)
        trades_df = pd.DataFrame([t.__dict__ for t in trades])

        bh_curve = _buy_and_hold_equity(bars, self.config)
        if benchmark_df is not None:
            bench = _prepare_bars(benchmark_df)
            bench_aligned = bench.set_index("timestamp")["close"].reindex(
                equity_df["timestamp"], method="ffill"
            )
            if not bench_aligned.isna().all():
                bh_curve = bench_aligned / bench_aligned.iloc[0] * self.config.initial_cash

        equity_series = equity_df.set_index("timestamp")["equity"]
        metrics = _compute_metrics(equity_series, len(trades_df), self.config.periods_per_year, bh_curve)

        strategy_rets = equity_series.pct_change().fillna(0.0)
        bench_rets = bh_curve.pct_change().fillna(0.0)
        ab = decompose_alpha_beta(strategy_rets, bench_rets, self.config.periods_per_year)
        metrics.update(ab)

        metrics["bh_return_pct"] = (float(bh_curve.iloc[-1]) / float(bh_curve.iloc[0]) - 1) * 100
        metrics["excess_return_pct"] = metrics["total_return_pct"] - metrics["bh_return_pct"]
        total_cost = total_fees + total_slippage + total_spread
        gross_profit = max(equity_series.iloc[-1] - self.config.initial_cash, 0.0)
        tax_amount = gross_profit * self.config.tax_rate
        net_equity = equity_series.iloc[-1] - tax_amount
        net_return_pct = (net_equity / self.config.initial_cash - 1) * 100

        metrics["total_fees"] = float(total_fees)
        metrics["total_slippage_cost"] = float(total_slippage)
        metrics["total_spread_cost"] = float(total_spread)
        metrics["total_cost"] = float(total_cost)
        metrics["total_cost_pct"] = float(total_cost / self.config.initial_cash * 100)
        metrics["tax_amount"] = float(tax_amount)
        metrics["net_return_pct"] = float(net_return_pct)
        metrics["net_excess_return_pct"] = float(net_return_pct - metrics["bh_return_pct"])

        if len(trades_df) > 0:
            trade_pnls = []
            for _, t in trades_df.iterrows():
                if t["side"] == "sell":
                    trade_pnls.append(-t["notional"] * (t["price"] / equity_df.loc[equity_df["timestamp"] == t["timestamp"], "close"].values[0] if not equity_df.empty else 0))
            if trade_pnls:
                metrics["win_rate_pct"] = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) * 100
                metrics["avg_trade_pct"] = np.mean(trade_pnls)
                gains = sum(p for p in trade_pnls if p > 0)
                losses = abs(sum(p for p in trade_pnls if p < 0))
                metrics["profit_factor"] = gains / losses if losses > 0 else float("inf")

        return BacktestResult(equity_curve=equity_df, trades=trades_df, metrics=metrics)

    def k_fold(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        k: int = 5,
        purge_bars: int = 5,
        metric_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        bars = _prepare_bars(df)
        n = len(bars)
        fold_size = n // k
        metric_keys = metric_keys or ["total_return_pct", "sharpe", "max_drawdown_pct", "excess_return_pct"]

        fold_results: list[dict] = []
        all_metrics: list[dict] = []

        for i in range(k):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            train_end = max(0, test_start - purge_bars)
            train_start = 0

            if train_end - train_start < 50:
                continue

            train_df = bars.iloc[train_start:train_end]
            test_df = bars.iloc[test_start:test_end]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.run(test_df, strategy)

            fold_info = {
                "fold": i + 1,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "test_start": str(test_df.iloc[0]["timestamp"]),
                "test_end": str(test_df.iloc[-1]["timestamp"]),
            }
            for mk in metric_keys:
                fold_info[mk] = result.metrics.get(mk, 0.0)
            fold_results.append(fold_info)
            all_metrics.append({mk: result.metrics.get(mk, 0.0) for mk in metric_keys})

        aggregate: dict[str, Any] = {}
        for mk in metric_keys:
            vals = [m[mk] for m in all_metrics if mk in m]
            if vals:
                aggregate[f"{mk}_mean"] = float(np.mean(vals))
                aggregate[f"{mk}_std"] = float(np.std(vals))
                aggregate[f"{mk}_min"] = float(np.min(vals))
                aggregate[f"{mk}_max"] = float(np.max(vals))

        return {
            "folds": fold_results,
            "aggregate": aggregate,
            "fold_dataframe": pd.DataFrame(fold_results),
        }

    def monte_carlo(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        n_simulations: int = 200,
        noise_bps: float = 5.0,
        delay_bars: tuple[int, int] = (0, 3),
        seed: int = 42,
    ) -> dict[str, Any]:
        from .monte_carlo import MonteCarloConfig, monte_carlo_simulate

        mc_cfg = MonteCarloConfig(
            n_simulations=n_simulations,
            noise_bps=noise_bps,
            delay_bars=delay_bars,
            seed=seed,
        )
        mc_result = monte_carlo_simulate(df, strategy, mc_cfg, self.config, verbose=False)

        sim_metrics = mc_result.summary_metrics
        base_row = sim_metrics[sim_metrics["name"] == "base"]
        sim_rows = sim_metrics[sim_metrics["name"] != "base"]

        aggregate: dict[str, Any] = {}
        for col in ["total_return_pct", "max_drawdown_pct", "sharpe", "volatility_pct"]:
            if col in sim_rows.columns:
                vals = sim_rows[col].dropna()
                if len(vals) > 0:
                    aggregate[f"mc_{col}_mean"] = float(vals.mean())
                    aggregate[f"mc_{col}_std"] = float(vals.std())
                    aggregate[f"mc_{col}_p5"] = float(vals.quantile(0.05))
                    aggregate[f"mc_{col}_p95"] = float(vals.quantile(0.95))

        if not base_row.empty:
            aggregate["base_return_pct"] = float(base_row["total_return_pct"].iloc[0])
            aggregate["base_sharpe"] = float(base_row["sharpe"].iloc[0])
            aggregate["base_max_dd"] = float(base_row["max_drawdown_pct"].iloc[0])

        return {
            "base_equity": mc_result.base_equity,
            "simulations": mc_result.simulations,
            "percentile_curves": mc_result.percentile_curves,
            "summary": sim_metrics,
            "aggregate": aggregate,
        }
