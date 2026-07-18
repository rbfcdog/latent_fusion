import sys
from pathlib import Path

# Allow running as: python backtest/hmm_technical_backtest.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features.technical import (
    sma,
    ema,
    hma,
    rsi,
    stochastic,
    atr,
    adx,
    macd,
    momentum,
    roc,
    bollinger_bands,
    rolling_volatility,
    parkinson_volatility,
    garman_klass_volatility,
)
from models.hmm_regimes import train_hmm_with_embeddings
from backtest.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)


def _load_data(data_dir, tickers=None, limit=None):
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(path.glob("*.csv"))
    if tickers:
        tickers = {t.upper() for t in tickers}
    data = {}
    for fp in files:
        symbol = fp.stem.upper()
        if tickers and symbol not in tickers:
            continue
        if limit is not None and len(data) >= limit:
            break
        df = pd.read_csv(fp)
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        data[symbol] = df
    if not data:
        raise ValueError("No valid CSV files found")
    logger.info("Loaded %d ticker(s) from %s", len(data), data_dir)
    return data


def _compute_indicators(df):
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        logger.warning("Skipping ticker with missing OHLCV columns")
        return None
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    volume = df["Volume"].astype(float)

    df["returns"] = close.pct_change()
    df["vol_20d"] = df["returns"].rolling(20).std() * np.sqrt(252)

    df["sma_20"] = sma(close, 20)
    df["sma_50"] = sma(close, 50)
    df["ema_12"] = ema(close, 12)
    df["ema_26"] = ema(close, 26)
    df["hma_9"] = hma(close, 9)

    df["rsi_14"] = rsi(close, 14)

    macd_df = macd(close)
    df["macd"] = macd_df["macd"]
    df["macd_signal"] = macd_df["signal"]
    df["macd_hist"] = macd_df["histogram"]

    stoch_df = stochastic(high, low, close)
    df["stoch_k"] = stoch_df["stoch_k"]
    df["stoch_d"] = stoch_df["stoch_d"]

    df["atr_14"] = atr(high, low, close, 14)

    adx_df = adx(high, low, close, 14)
    df["plus_di"] = adx_df["plus_di"]
    df["minus_di"] = adx_df["minus_di"]
    df["adx"] = adx_df["adx"]

    df["momentum_10"] = momentum(close, 10)
    df["roc_10"] = roc(close, 10)

    bb_df = bollinger_bands(close, 20, 2.0)
    df["bb_upper"] = bb_df["bb_upper"]
    df["bb_mid"] = bb_df["bb_mid"]
    df["bb_lower"] = bb_df["bb_lower"]
    df["bb_bandwidth"] = bb_df["bb_bandwidth"]
    df["bb_percent_b"] = bb_df["bb_percent_b"]

    df["vol_rolling_20"] = rolling_volatility(close, 20)
    df["vol_parkinson"] = parkinson_volatility(high, low, 20)
    df["vol_garman_klass"] = garman_klass_volatility(open_, high, low, close, 20)

    return df


def _build_features(df):
    feature_cols = [
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        "hma_9",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "stoch_k",
        "stoch_d",
        "atr_14",
        "plus_di",
        "minus_di",
        "adx",
        "momentum_10",
        "roc_10",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "bb_bandwidth",
        "bb_percent_b",
        "vol_rolling_20",
        "vol_parkinson",
        "vol_garman_klass",
    ]
    cols = [c for c in feature_cols if c in df.columns]
    data = df[["Date", "returns", "vol_20d"] + cols].dropna()
    if len(data) < 60:
        logger.warning("Not enough samples after indicator prep: %d", len(data))
        return None
    features = data[cols].values
    returns = data["returns"].values
    vol = data["vol_20d"].values
    dates = data["Date"].values
    return features, returns, vol, dates


def _build_signal(regime_labels, dates):
    labels = np.array(regime_labels, dtype=int)
    unique = np.unique(labels)
    if len(unique) == 1:
        mapping = {int(unique[0]): 0.0}
    elif len(unique) == 3:
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
    else:
        scale = np.linspace(-1.0, 1.0, len(unique))
        mapping = {int(state): float(scale[i]) for i, state in enumerate(sorted(unique))}
    signal = np.array([mapping.get(int(r), 0.0) for r in labels], dtype=float)
    return pd.Series(signal, index=pd.to_datetime(dates))


def _state_order_from_returns(states, returns, n_states):
    means = []
    for i in range(n_states):
        mask = states == i
        if mask.any():
            means.append(float(np.nanmean(returns[mask])))
        else:
            means.append(float("-inf"))
    return list(np.argsort(means))


def _performance_stats(perf):
    daily = perf["return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ann_ret = (1.0 + daily.mean()) ** 252 - 1.0
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)
    total = perf["value"].iloc[-1] / perf["value"].iloc[0] - 1.0
    return {
        "total_return": total,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
    }


def _buy_and_hold_performance(price_df, initial_cash):
    returns = price_df.ffill().pct_change(fill_method=None).fillna(0.0)
    bh_ret = returns.mean(axis=1)
    value = initial_cash * (1.0 + bh_ret).cumprod()
    return pd.DataFrame({"value": value, "return": bh_ret}, index=price_df.index)


def _plot_results(perf, weights, buy_hold_perf, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = weights[~weights.index.duplicated(keep="last")].sort_index()
    equity = perf["value"]
    bh_equity = buy_hold_perf["value"].reindex(equity.index).ffill()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    bh_drawdown = bh_equity / bh_equity.cummax() - 1.0
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    w_plot = weights.reindex(equity.index).ffill().fillna(0.0)
    strategy_ret = perf["return"].reindex(equity.index).fillna(0.0)
    market_ret = buy_hold_perf["return"].reindex(equity.index).fillna(0.0)
    window = 63
    beta = strategy_ret.rolling(window).cov(market_ret) / market_ret.rolling(window).var()
    alpha = strategy_ret.rolling(window).mean() - beta * market_ret.rolling(window).mean()
    alpha_ann = alpha * 252

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    axes[0].plot(equity.index, equity.values, color="#4C72B0", linewidth=1.8, label="Strategy")
    axes[0].plot(bh_equity.index, bh_equity.values, color="#55A868", linewidth=1.6, label="Buy and Hold")
    axes[0].set_title("Equity Curve: Strategy vs Buy and Hold")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="#C44E52", alpha=0.35, label="Strategy")
    axes[1].plot(bh_drawdown.index, bh_drawdown.values, color="#8172B2", linewidth=1.4, label="Buy and Hold")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].stackplot(w_plot.index, w_plot.values.T, alpha=0.8)
    axes[2].set_title("Ticker Allocation Over Time")
    axes[2].set_ylabel("Weight")
    axes[2].grid(True, alpha=0.3)

    ax_beta = axes[3]
    ax_beta.plot(beta.index, beta.values, color="#4C72B0", linewidth=1.6, label="Beta")
    ax_beta.set_title("Rolling Alpha and Beta")
    ax_beta.set_ylabel("Beta")
    ax_beta.grid(True, alpha=0.3)
    ax_alpha = ax_beta.twinx()
    ax_alpha.plot(alpha_ann.index, alpha_ann.values, color="#C44E52", linewidth=1.4, label="Alpha (ann)")
    ax_alpha.set_ylabel("Alpha (ann)")
    lines, labels = ax_beta.get_legend_handles_labels()
    lines2, labels2 = ax_alpha.get_legend_handles_labels()
    ax_beta.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    out_path = output_dir / "hmm_technical_backtest.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot: %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/time_series")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--trading-cost-bps", type=float, default=0.0)
    parser.add_argument("--max-gross", type=float, default=1.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--embedding-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", default="backtest_images")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--ticker-limit", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    tickers = None
    if args.tickers:
        tickers = []
        for item in args.tickers:
            tickers.extend([t.strip() for t in item.split(",") if t.strip()])

    data = _load_data(args.data_dir, tickers=tickers, limit=args.ticker_limit)

    signals = {}
    filtered = {}
    feature_cache = {}
    for symbol, df in data.items():
        logger.info("Processing %s", symbol)
        df_ind = _compute_indicators(df)
        if df_ind is None:
            logger.warning("No indicators for %s", symbol)
            continue
        features_pack = _build_features(df_ind)
        if features_pack is None:
            logger.warning("No features for %s", symbol)
            continue
        features, returns, vol, dates = features_pack
        feature_cache[symbol] = (features, returns, vol, dates)
        filtered[symbol] = df

    if not feature_cache:
        raise ValueError("No signals produced. Check data and indicator windows.")

    all_features = np.vstack([pack[0] for pack in feature_cache.values()])
    all_returns = np.concatenate([pack[1] for pack in feature_cache.values()])
    all_vol = np.concatenate([pack[2] for pack in feature_cache.values()])

    logger.info(
        "Fitting global HMM | tickers=%d samples=%d features=%d",
        len(feature_cache),
        len(all_returns),
        all_features.shape[1],
    )
    t0 = time.perf_counter()
    result = train_hmm_with_embeddings(
        all_features,
        all_returns,
        volatility=all_vol,
        n_states=args.n_states,
        random_state=42,
        embedding_weight=args.embedding_weight,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Global HMM logprob: %.4f | fit_time=%.2fs", result.logprob, elapsed)

    state_order = _state_order_from_returns(result.states, all_returns, args.n_states)
    state_to_label = {state: idx for idx, state in enumerate(state_order)}
    logger.info("State order by mean return (low->high): %s", state_order)

    for symbol, pack in feature_cache.items():
        features, returns, vol, dates = pack
        logger.info("Predicting regimes for %s | samples=%d", symbol, len(returns))
        X_combined = np.hstack([features, returns.reshape(-1, 1), vol.reshape(-1, 1)])
        X_scaled = result.scaler.transform(X_combined)
        states = result.model.predict(X_scaled)
        regime_labels = np.array([state_to_label.get(int(s), 0) for s in states], dtype=int)
        counts = np.bincount(regime_labels, minlength=args.n_states)
        logger.info("Regime counts for %s: %s", symbol, counts.tolist())
        signal_series = _build_signal(regime_labels, dates)
        signals[symbol] = signal_series

    engine = BacktestEngine(
        filtered,
        price_col="Close",
        date_col="Date",
        start=args.start,
        end=args.end,
        trading_cost_bps=args.trading_cost_bps,
        max_gross=args.max_gross,
        allow_short=args.allow_short,
    )

    signal_df = pd.DataFrame(index=engine.price_df.index)
    for symbol in engine.symbols:
        signal_series = signals[symbol]
        signal_df[symbol] = signal_series.reindex(engine.price_df.index).ffill().fillna(0.0)

    class RegimeModel:
        def __init__(self, signals_df):
            self.signals_df = signals_df

        def allocate(self, state):
            date = state["date"]
            if date not in self.signals_df.index:
                return np.zeros(len(state["symbols"]), dtype=float)
            w = self.signals_df.loc[date].values.astype(float)
            gross = np.sum(np.abs(w))
            if gross > 0:
                w = w / gross
            return w

    results = engine.run(RegimeModel(signal_df))
    perf = results["performance"]
    weights = results["weights"]
    buy_hold_perf = _buy_and_hold_performance(engine.price_df, engine.initial_cash)

    stats = _performance_stats(perf)
    stats_df = pd.DataFrame([stats])
    metrics_path = Path(args.output_dir) / "hmm_technical_backtest_metrics.csv"
    stats_df.to_csv(metrics_path, index=False)
    logger.info("Saved metrics: %s", metrics_path)

    _plot_results(perf, weights, buy_hold_perf, args.output_dir)


if __name__ == "__main__":
    main()
