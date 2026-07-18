from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    market_data_path: Path = Path("data/lse_market_data/combined_1d.parquet")
    indicators_path: Path = Path("data/volatility_regime_analysis/market_indicators.parquet")
    regimes_path: Path = Path("data/volatility_regime_analysis/hmm_regimes.parquet")
    option_iv_path: Path = Path("data/volatility_regime_analysis/option_iv_indicators.csv")
    output_dir: Path = Path("data/regime_backtest_comparison")
    trading_cost_bps: float = 5.0
    target_vol: float = 0.18
    max_leverage: float = 1.0
    min_weight: float = 0.0
    rebalance_frequency: str = "D"
    stress_weight: float = 0.0
    neutral_weight: float = 0.55
    calm_weight: float = 1.0
    momentum_window: int = 20
    iv_rich_threshold: float = 0.03
    iv_cheap_threshold: float = -0.05


def load_inputs(cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = pd.read_parquet(cfg.market_data_path)
    indicators = pd.read_parquet(cfg.indicators_path)
    regimes = pd.read_parquet(cfg.regimes_path)
    option_iv = pd.read_csv(cfg.option_iv_path) if cfg.option_iv_path.exists() else pd.DataFrame()
    for df in [market, indicators, regimes]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return market, indicators, regimes, option_iv


def prepare_backtest_frame(
    market: pd.DataFrame,
    indicators: pd.DataFrame,
    regimes: pd.DataFrame,
    option_iv: pd.DataFrame,
) -> pd.DataFrame:
    base = market[["asset_group", "symbol", "timestamp", "close"]].copy()
    base = base.sort_values(["symbol", "timestamp"])
    base["return"] = base.groupby("symbol")["close"].pct_change()

    feature_cols = [
        "symbol",
        "timestamp",
        "rv_close_20",
        "rv_yang_zhang_20",
        "ewma_vol_20",
        "atr_pct_14",
        "rsi_14",
        "drawdown_60",
        "event_flag",
    ]
    available_feature_cols = [c for c in feature_cols if c in indicators.columns]
    merged = base.merge(indicators[available_feature_cols], on=["symbol", "timestamp"], how="left")
    if not regimes.empty:
        merged = merged.merge(
            regimes[["symbol", "timestamp", "hmm_regime", "hmm_state_rank"]],
            on=["symbol", "timestamp"],
            how="left",
        )
    else:
        merged["hmm_regime"] = np.nan
        merged["hmm_state_rank"] = np.nan

    merged["momentum_20"] = merged.groupby("symbol")["close"].pct_change(20)
    if not option_iv.empty and "underlying" in option_iv.columns:
        iv_cols = ["underlying", "atm_iv", "iv_minus_rv20", "put_downside_minus_call_upside_skew", "term_slope_far_minus_near"]
        iv_cols = [c for c in iv_cols if c in option_iv.columns]
        merged = merged.merge(
            option_iv[iv_cols].rename(columns={"underlying": "symbol"}),
            on="symbol",
            how="left",
        )
    else:
        merged["iv_minus_rv20"] = np.nan
    return merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _pivot(frame: pd.DataFrame, value: str, symbols: list[str] | None = None) -> pd.DataFrame:
    p = frame.pivot(index="timestamp", columns="symbol", values=value).sort_index()
    if symbols is not None:
        p = p.reindex(columns=symbols)
    return p


def equal_weight_weights(mask: pd.DataFrame) -> pd.DataFrame:
    weights = mask.astype(float)
    row_sum = weights.sum(axis=1).replace(0, np.nan)
    weights = weights.div(row_sum, axis=0).fillna(0.0)
    return weights


def inverse_vol_weights(vol: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / vol.replace(0, np.nan)
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0) * eligible.astype(float)
    row_sum = inv.sum(axis=1).replace(0, np.nan)
    return inv.div(row_sum, axis=0).fillna(0.0)


def apply_vol_target(weights: pd.DataFrame, vol: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    weighted_vol = (weights.abs() * vol.fillna(vol.median())).sum(axis=1)
    scale = (cfg.target_vol / weighted_vol.replace(0, np.nan)).clip(upper=cfg.max_leverage).fillna(0.0)
    return weights.mul(scale, axis=0)


def _turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.fillna(0.0).diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))


def run_weight_backtest(
    returns: pd.DataFrame,
    weights_signal: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    returns = returns.reindex_like(weights_signal).fillna(0.0)
    weights = weights_signal.shift(1).fillna(0.0)
    turnover = _turnover(weights)
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    gross_return = (weights * returns).sum(axis=1)
    cost = turnover * cfg.trading_cost_bps * 1e-4
    net_return = gross_return - cost
    equity = (1.0 + net_return).cumprod()
    out = pd.DataFrame(
        {
            "return": net_return,
            "gross_return": gross_return,
            "cost": cost,
            "turnover": turnover,
            "gross": gross,
            "net": net,
            "equity": equity,
        }
    )
    return out


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def performance_metrics(perf: pd.DataFrame, annualization: int = 252) -> dict[str, float]:
    r = perf["return"].dropna()
    if len(r) == 0:
        return {}
    equity = perf["equity"].dropna()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(len(r) / annualization, 1 / annualization)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0)
    vol = float(r.std(ddof=0) * math.sqrt(annualization))
    sharpe = float(r.mean() / r.std(ddof=0) * math.sqrt(annualization)) if r.std(ddof=0) > 0 else np.nan
    downside = r[r < 0].std(ddof=0) * math.sqrt(annualization)
    sortino = float(r.mean() / r[r < 0].std(ddof=0) * math.sqrt(annualization)) if downside and downside > 0 else np.nan
    hit_rate = float((r > 0).mean())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown(equity),
        "calmar": cagr / abs(max_drawdown(equity)) if max_drawdown(equity) < 0 else np.nan,
        "hit_rate": hit_rate,
        "avg_turnover": float(perf["turnover"].mean()),
        "avg_gross": float(perf["gross"].mean()),
        "total_cost": float(perf["cost"].sum()),
        "days": int(len(r)),
    }


def build_strategy_weights(frame: pd.DataFrame, symbols: list[str], cfg: BacktestConfig) -> dict[str, pd.DataFrame]:
    returns = _pivot(frame, "return", symbols)
    close = _pivot(frame, "close", symbols)
    vol = _pivot(frame, "rv_yang_zhang_20", symbols).ffill()
    regime = _pivot(frame, "hmm_regime", symbols).ffill()
    momentum = _pivot(frame, "momentum_20", symbols).ffill()
    iv_spread = _pivot(frame, "iv_minus_rv20", symbols).ffill()

    available = close.notna()
    buy_hold = equal_weight_weights(available)

    risk_on = regime.isin(["calm", "neutral"]) & available
    regime_flat = equal_weight_weights(risk_on)

    regime_scale = regime.replace(
        {
            "calm": cfg.calm_weight,
            "neutral": cfg.neutral_weight,
            "stress": cfg.stress_weight,
        }
    ).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    regime_scaled = equal_weight_weights(available).mul(regime_scale, axis=0)
    # Preserve cross-sectional equal weights but allow cash in stress.
    regime_scaled = regime_scaled.div(regime_scaled.abs().sum(axis=1).clip(lower=1.0), axis=0).fillna(0.0)

    inv_vol = inverse_vol_weights(vol, risk_on)
    inv_vol_targeted = apply_vol_target(inv_vol, vol, cfg)

    momentum_eligible = risk_on & (momentum > 0)
    momentum_regime = equal_weight_weights(momentum_eligible)

    iv_overlay = risk_on.copy()
    if iv_spread.notna().any().any():
        # For symbols with option data: avoid names with rich vol and prefer cheap-vol names.
        rich = iv_spread > cfg.iv_rich_threshold
        cheap = iv_spread < cfg.iv_cheap_threshold
        iv_multiplier = pd.DataFrame(1.0, index=iv_spread.index, columns=iv_spread.columns)
        iv_multiplier = iv_multiplier.mask(rich, 0.5).mask(cheap, 1.25)
        iv_overlay_w = equal_weight_weights(risk_on).mul(iv_multiplier, axis=0)
        iv_overlay_w = iv_overlay_w.div(iv_overlay_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        iv_overlay_w = equal_weight_weights(risk_on)

    return {
        "buy_hold_equal_weight": buy_hold,
        "hmm_regime_long_flat": regime_flat,
        "hmm_regime_scaled_cash": regime_scaled,
        "hmm_inverse_vol_target": inv_vol_targeted,
        "hmm_momentum_filter": momentum_regime,
        "hmm_iv_overlay": iv_overlay_w,
    }


def run_comparison_for_universe(
    frame: pd.DataFrame,
    cfg: BacktestConfig,
    universe_name: str,
    symbols: Iterable[str],
    annualization: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    symbols = list(dict.fromkeys(symbols))
    local = frame[frame["symbol"].isin(symbols)].copy()
    weights = build_strategy_weights(local, symbols, cfg)
    returns = _pivot(local, "return", symbols).fillna(0.0)
    perf_frames: dict[str, pd.DataFrame] = {}
    metrics = []
    for strategy, w in weights.items():
        perf = run_weight_backtest(returns, w, cfg)
        perf["strategy"] = strategy
        perf["universe"] = universe_name
        perf_frames[strategy] = perf
        metric = performance_metrics(perf, annualization=annualization)
        metric.update({"universe": universe_name, "strategy": strategy, "symbols": len(symbols)})
        metrics.append(metric)
    equity = pd.concat(perf_frames.values()).reset_index(names="timestamp")
    metrics_df = pd.DataFrame(metrics)
    return metrics_df, equity, weights


def run_all_comparisons(frame: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_specs: dict[str, tuple[list[str], int]] = {}
    for group, group_df in frame.groupby("asset_group"):
        annualization = 365 if group == "crypto" else 252
        universe_specs[group] = (sorted(group_df["symbol"].unique()), annualization)
    universe_specs["all_assets"] = (sorted(frame["symbol"].unique()), 252)

    metric_frames = []
    equity_frames = []
    for universe, (symbols, annualization) in universe_specs.items():
        metrics, equity, _ = run_comparison_for_universe(frame, cfg, universe, symbols, annualization)
        metric_frames.append(metrics)
        equity_frames.append(equity)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(equity_frames, ignore_index=True)


def save_backtest_outputs(cfg: BacktestConfig, metrics: pd.DataFrame, equity: pd.DataFrame) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.output_dir / "strategy_vs_buyhold_metrics.csv"
    equity_path = cfg.output_dir / "strategy_vs_buyhold_equity.parquet"
    metrics.to_csv(metrics_path, index=False)
    equity.to_parquet(equity_path, index=False)
    return {"metrics": metrics_path, "equity": equity_path}

