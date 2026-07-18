from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TickerStats:
    ticker: str
    mean_return_pct: float
    volatility_pct: float
    sharpe: float
    max_drawdown_pct: float
    avg_volume: float
    n_bars: int
    performance_rank: int = 0
    vol_rank: int = 0
    liquidity_rank: int = 0


@dataclass
class TickerSplitResult:
    stats: pd.DataFrame
    train_tickers: list[str]
    test_tickers: list[str]
    split_report: dict


def compute_ticker_stats(
    data: dict[str, pd.DataFrame] | pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    date_col: str = "timestamp",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    if isinstance(data, dict):
        rows = []
        for ticker, df in data.items():
            ts = pd.to_datetime(df[date_col]) if date_col in df.columns else df.index
            prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
            if len(prices) < 10:
                continue
            rets = prices.pct_change().dropna()
            vol = float(rets.std() * np.sqrt(periods_per_year) * 100)
            mean_ret = float(rets.mean() * periods_per_year * 100)
            sharpe = mean_ret / vol if vol > 0 else 0.0
            cum = (1 + rets).cumprod()
            dd = (cum / cum.cummax() - 1).min() * 100
            avg_vol = float(df[volume_col].mean()) if volume_col in df.columns else 0.0
            rows.append({
                "ticker": ticker,
                "mean_return_pct": mean_ret,
                "volatility_pct": vol,
                "sharpe": sharpe,
                "max_drawdown_pct": float(dd),
                "avg_volume": avg_vol,
                "n_bars": len(prices),
            })
        return pd.DataFrame(rows)

    df = data.copy()
    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})

    rows = []
    for ticker, group in df.groupby("ticker"):
        prices = pd.to_numeric(group[price_col], errors="coerce").dropna()
        if len(prices) < 10:
            continue
        rets = prices.pct_change().dropna()
        vol = float(rets.std() * np.sqrt(periods_per_year) * 100)
        mean_ret = float(rets.mean() * periods_per_year * 100)
        sharpe = mean_ret / vol if vol > 0 else 0.0
        cum = (1 + rets).cumprod()
        dd = (cum / cum.cummax() - 1).min() * 100
        avg_vol = float(group[volume_col].mean()) if volume_col in group.columns else 0.0
        rows.append({
            "ticker": ticker,
            "mean_return_pct": mean_ret,
            "volatility_pct": vol,
            "sharpe": sharpe,
            "max_drawdown_pct": float(dd),
            "avg_volume": avg_vol,
            "n_bars": len(prices),
        })
    return pd.DataFrame(rows)


def stratified_ticker_split(
    stats: pd.DataFrame,
    test_ratio: float = 0.3,
    n_strata: int = 3,
    random_state: int = 42,
) -> TickerSplitResult:
    rng = np.random.default_rng(random_state)
    stats = stats.copy()
    stats["performance_rank"] = stats["mean_return_pct"].rank(method="dense").astype(int)
    stats["vol_rank"] = stats["volatility_pct"].rank(method="dense").astype(int)
    stats["liquidity_rank"] = stats["avg_volume"].rank(method="dense").astype(int)

    stats["stratum"] = pd.qcut(
        stats["mean_return_pct"], n_strata, labels=False, duplicates="drop"
    ).astype(int)

    train_tickers: list[str] = []
    test_tickers: list[str] = []

    for stratum, group in stats.groupby("stratum"):
        tickers = group["ticker"].tolist()
        rng.shuffle(tickers)
        n_test = max(1, int(len(tickers) * test_ratio))
        test_tickers.extend(tickers[:n_test])
        train_tickers.extend(tickers[n_test:])

    train_stats = stats[stats["ticker"].isin(train_tickers)]
    test_stats = stats[stats["ticker"].isin(test_tickers)]

    split_report = {
        "n_train": len(train_tickers),
        "n_test": len(test_tickers),
        "train_mean_return_pct": float(train_stats["mean_return_pct"].mean()),
        "test_mean_return_pct": float(test_stats["mean_return_pct"].mean()),
        "train_mean_volatility_pct": float(train_stats["volatility_pct"].mean()),
        "test_mean_volatility_pct": float(test_stats["volatility_pct"].mean()),
        "train_mean_sharpe": float(train_stats["sharpe"].mean()),
        "test_mean_sharpe": float(test_stats["sharpe"].mean()),
        "train_mean_max_dd": float(train_stats["max_drawdown_pct"].mean()),
        "test_mean_max_dd": float(test_stats["max_drawdown_pct"].mean()),
        "return_balance_diff": abs(
            float(train_stats["mean_return_pct"].mean())
            - float(test_stats["mean_return_pct"].mean())
        ),
    }

    return TickerSplitResult(
        stats=stats,
        train_tickers=train_tickers,
        test_tickers=test_tickers,
        split_report=split_report,
    )


def correlation_matrix(
    data: dict[str, pd.DataFrame] | pd.DataFrame,
    price_col: str = "close",
    date_col: str = "timestamp",
) -> pd.DataFrame:
    if isinstance(data, dict):
        price_dict = {}
        for ticker, df in data.items():
            ts = pd.to_datetime(df[date_col]) if date_col in df.columns else df.index
            prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
            price_dict[ticker] = pd.Series(prices.values, index=ts if date_col in df.columns else df.index[:len(prices)])
        all_prices = pd.DataFrame(price_dict)
    else:
        df = data.copy()
        if "ticker" not in df.columns and "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})
        pivot = df.pivot_table(index=date_col, columns="ticker", values=price_col)
        all_prices = pivot

    returns = all_prices.pct_change().dropna()
    return returns.corr()
