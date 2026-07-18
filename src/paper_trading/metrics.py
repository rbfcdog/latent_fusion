from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int,
    bh_equity: pd.Series | None = None,
) -> dict[str, float]:
    if len(equity) < 2:
        return {"total_return_pct": 0.0}
    returns = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) * 100
    ann = periods_per_year
    ann_ret = float((equity.iloc[-1] / equity.iloc[0]) ** (ann / max(len(equity) - 1, 1)) - 1) * 100

    std = float(returns.std()) if len(returns) > 1 else 0.0
    down = returns.clip(upper=0.0)
    down_std = float(down[down < 0].std()) if (down < 0).any() else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(ann)) if std > 0 else 0.0
    sortino = float(returns.mean() / down_std * np.sqrt(ann)) if down_std > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = float(drawdown.min()) * 100
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

    n_trades = int(len(trades))
    win_rate = 0.0
    profit_factor = 0.0
    avg_trade = 0.0
    if n_trades > 0 and "pnl" in trades.columns:
        pnls = trades["pnl"].fillna(0.0)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = float(len(wins) / n_trades * 100) if n_trades else 0.0
        avg_trade = float(pnls.mean()) if n_trades else 0.0
        g = float(wins.sum())
        l = abs(float(losses.sum()))
        profit_factor = g / l if l > 0 else (float("inf") if g > 0 else 0.0)

    metrics: dict[str, float] = {
        "total_return_pct": total_return,
        "annualized_return_pct": ann_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "calmar": calmar,
        "volatility_pct": std * np.sqrt(ann) * 100,
        "n_trades": float(n_trades),
        "win_rate_pct": win_rate,
        "avg_trade": avg_trade,
        "profit_factor": profit_factor,
    }

    if bh_equity is not None and len(bh_equity) >= 2:
        bh_aligned = bh_equity.reindex(equity.index, method="ffill").ffill()
        if bh_aligned.empty or bh_aligned.iloc[0] == 0:
            return metrics
        bh_ret = float(bh_equity.iloc[-1] / bh_equity.iloc[0] - 1) * 100
        metrics["bh_return_pct"] = bh_ret
        metrics["excess_return_pct"] = total_return - bh_ret
        bh_rets = bh_aligned.pct_change().fillna(0.0)
        cov = float(np.cov(returns, bh_rets)[0, 1]) if len(returns) > 1 else 0.0
        var = float(bh_rets.var()) if len(bh_rets) > 1 else 0.0
        beta = cov / var if var > 0 else 0.0
        alpha = float(returns.mean() - beta * bh_rets.mean()) * ann * 100
        metrics["beta"] = beta
        metrics["alpha_annualized_pct"] = alpha

    return metrics


def bh_curve_from_prices(close: pd.Series, initial_cash: float) -> pd.Series:
    if close.empty:
        return pd.Series(dtype=float)
    base = float(close.iloc[0])
    if base <= 0:
        return pd.Series(initial_cash, index=close.index)
    return close / base * initial_cash
