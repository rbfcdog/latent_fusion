from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RegimeAnalysisConfig:
    input_path: Path = Path("data/lse_market_data/combined_1d.parquet")
    output_dir: Path = Path("data/volatility_regime_analysis")
    annualization_crypto: int = 365
    annualization_trading: int = 252
    windows: tuple[int, ...] = (10, 20, 60)
    hmm_components: int = 3
    min_hmm_rows: int = 120
    event_window: int = 60
    event_z_threshold: float = 2.25
    event_abs_return_quantile: float = 0.90
    hawkes_beta_default: float = 1 / 20
    max_hawkes_symbols: int = 44
    selected_symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "AAPL", "NVDA", "PBR", "VALE")
    option_underlyings: tuple[str, ...] = ("AAPL", "NVDA", "TSLA", "MSFT", "PBR", "VALE")
    option_min_dte: int = 7
    option_max_dte: int = 180
    option_limit: int = 5000


def load_market_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input extension: {path.suffix}")
    required = {"asset_group", "symbol", "open", "high", "low", "close", "volume", "timestamp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required market columns: {sorted(missing)}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "close"]).sort_values(["symbol", "timestamp"])
    return df.reset_index(drop=True)


def _annualization_for_group(asset_group: str, cfg: RegimeAnalysisConfig) -> int:
    return cfg.annualization_crypto if str(asset_group).lower() == "crypto" else cfg.annualization_trading


def _safe_log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    valid = (num > 0) & (den > 0)
    out = pd.Series(np.nan, index=num.index, dtype=float)
    out.loc[valid] = np.log(num.loc[valid] / den.loc[valid])
    return out.replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _max_drawdown(close: pd.Series, window: int = 60) -> pd.Series:
    peak = close.rolling(window, min_periods=2).max()
    return close / peak - 1


def compute_symbol_indicators(group: pd.DataFrame, cfg: RegimeAnalysisConfig) -> pd.DataFrame:
    g = group.sort_values("timestamp").copy()
    annualization = _annualization_for_group(str(g["asset_group"].iloc[0]), cfg)

    g["log_close"] = np.log(g["close"].where(g["close"] > 0))
    g["log_return"] = g["log_close"].diff()
    g["simple_return"] = g["close"].pct_change()
    g["abs_return"] = g["log_return"].abs()
    g["squared_return"] = g["log_return"] ** 2
    g["overnight_return"] = _safe_log_ratio(g["open"], g["close"].shift(1))
    g["intraday_return"] = _safe_log_ratio(g["close"], g["open"])
    g["hl_log_range"] = _safe_log_ratio(g["high"], g["low"])
    g["oc_log_range"] = _safe_log_ratio(g["close"], g["open"])
    g["ho_log_range"] = _safe_log_ratio(g["high"], g["open"])
    g["lo_log_range"] = _safe_log_ratio(g["low"], g["open"])

    true_ranges = pd.concat(
        [
            (g["high"] - g["low"]).abs(),
            (g["high"] - g["close"].shift(1)).abs(),
            (g["low"] - g["close"].shift(1)).abs(),
        ],
        axis=1,
    )
    g["true_range"] = true_ranges.max(axis=1)
    g["atr_14"] = g["true_range"].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    g["atr_pct_14"] = g["atr_14"] / g["close"]

    g["ewma_vol_20"] = g["log_return"].ewm(span=20, adjust=False, min_periods=10).std() * math.sqrt(annualization)
    g["rsi_14"] = _rsi(g["close"], 14)
    g["drawdown_60"] = _max_drawdown(g["close"], 60)
    g["volume_z_20"] = (g["volume"] - g["volume"].rolling(20, min_periods=10).mean()) / g["volume"].rolling(20, min_periods=10).std()

    for window in cfg.windows:
        g[f"rv_close_{window}"] = g["log_return"].rolling(window, min_periods=max(5, window // 2)).std() * math.sqrt(annualization)
        g[f"rv_parkinson_{window}"] = np.sqrt(
            annualization
            * (g["hl_log_range"] ** 2).rolling(window, min_periods=max(5, window // 2)).mean()
            / (4 * math.log(2))
        )
        g[f"rv_garman_klass_{window}"] = np.sqrt(
            annualization
            * (
                0.5 * g["hl_log_range"] ** 2
                - (2 * math.log(2) - 1) * g["oc_log_range"] ** 2
            )
            .clip(lower=0)
            .rolling(window, min_periods=max(5, window // 2))
            .mean()
        )
        rs_var = (
            g["ho_log_range"] * _safe_log_ratio(g["high"], g["close"])
            + g["lo_log_range"] * _safe_log_ratio(g["low"], g["close"])
        )
        g[f"rv_rogers_satchell_{window}"] = np.sqrt(
            annualization
            * rs_var.clip(lower=0).rolling(window, min_periods=max(5, window // 2)).mean()
        )

        open_var = g["overnight_return"].rolling(window, min_periods=max(5, window // 2)).var()
        close_var = g["intraday_return"].rolling(window, min_periods=max(5, window // 2)).var()
        rs_mean = rs_var.clip(lower=0).rolling(window, min_periods=max(5, window // 2)).mean()
        k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
        yz_var = open_var + k * close_var + (1 - k) * rs_mean
        g[f"rv_yang_zhang_{window}"] = np.sqrt(annualization * yz_var.clip(lower=0))

    z_mean = g["log_return"].rolling(cfg.event_window, min_periods=max(20, cfg.event_window // 2)).mean()
    z_std = g["log_return"].rolling(cfg.event_window, min_periods=max(20, cfg.event_window // 2)).std()
    g["return_z_60"] = (g["log_return"] - z_mean) / z_std
    g["abs_return_q90_60"] = g["abs_return"].rolling(cfg.event_window, min_periods=max(20, cfg.event_window // 2)).quantile(cfg.event_abs_return_quantile)
    g["jump_event"] = (g["return_z_60"].abs() >= cfg.event_z_threshold).fillna(False)
    g["high_vol_event"] = (g["abs_return"] >= g["abs_return_q90_60"]).fillna(False)
    g["event_flag"] = (g["jump_event"] | g["high_vol_event"]).astype(int)
    return g


def compute_market_indicators(df: pd.DataFrame, cfg: RegimeAnalysisConfig) -> pd.DataFrame:
    pieces = [compute_symbol_indicators(g, cfg) for _, g in df.groupby("symbol", sort=False)]
    return pd.concat(pieces, ignore_index=True).sort_values(["symbol", "timestamp"])


def fit_hmm_regimes(indicators: pd.DataFrame, cfg: RegimeAnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    from hmmlearn.hmm import GaussianHMM

    feature_cols = ["log_return", "rv_close_20", "rv_parkinson_20", "atr_pct_14", "volume_z_20", "drawdown_60"]
    outputs: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for symbol, group in indicators.groupby("symbol", sort=False):
        g = group.sort_values("timestamp").copy()
        features = g[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(features) < cfg.min_hmm_rows:
            continue

        scaler = StandardScaler()
        x = scaler.fit_transform(features)
        model = GaussianHMM(
            n_components=cfg.hmm_components,
            covariance_type="diag",
            n_iter=500,
            random_state=42,
            min_covar=1e-4,
        )
        model.fit(x)
        states = model.predict(x)
        scored = g.loc[features.index, ["asset_group", "symbol", "timestamp", "close"] + feature_cols].copy()
        scored["hmm_state_raw"] = states
        state_stats = scored.groupby("hmm_state_raw").agg(
            mean_return=("log_return", "mean"),
            mean_vol=("rv_close_20", "mean"),
            mean_drawdown=("drawdown_60", "mean"),
            rows=("hmm_state_raw", "size"),
        )
        ordered = state_stats.sort_values(["mean_vol", "mean_drawdown"], ascending=[True, False]).index.tolist()
        labels = ["calm", "neutral", "stress", "extreme"]
        state_map = {state: labels[min(i, len(labels) - 1)] for i, state in enumerate(ordered)}
        scored["hmm_regime"] = scored["hmm_state_raw"].map(state_map)
        scored["hmm_state_rank"] = scored["hmm_state_raw"].map({state: i for i, state in enumerate(ordered)})
        outputs.append(scored)

        for state, row in state_stats.iterrows():
            summaries.append(
                {
                    "asset_group": scored["asset_group"].iloc[0],
                    "symbol": symbol,
                    "hmm_state_raw": int(state),
                    "hmm_regime": state_map[state],
                    "mean_return": row["mean_return"],
                    "mean_vol": row["mean_vol"],
                    "mean_drawdown": row["mean_drawdown"],
                    "rows": int(row["rows"]),
                    "converged": bool(model.monitor_.converged),
                    "log_likelihood": float(model.monitor_.history[-1]),
                }
            )

    regimes = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    return regimes, summary


def _event_times(group: pd.DataFrame) -> tuple[np.ndarray, float]:
    g = group.sort_values("timestamp").copy()
    g = g[g["event_flag"].fillna(0).astype(int) == 1]
    if g.empty:
        return np.array([], dtype=float), 0.0
    all_dates = group["timestamp"].sort_values()
    start = all_dates.iloc[0]
    end = all_dates.iloc[-1]
    times = (g["timestamp"] - start).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    horizon = max((end - start).total_seconds() / 86400.0, 1.0)
    return np.sort(times), horizon


def hawkes_negative_log_likelihood(params: np.ndarray, times: np.ndarray, horizon: float) -> float:
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or alpha >= 1 or beta <= 0 or horizon <= 0:
        return 1e12
    if len(times) == 0:
        return mu * horizon

    excitation = 0.0
    loglik = 0.0
    last_t = 0.0
    for t in times:
        excitation *= math.exp(-beta * max(t - last_t, 0.0))
        intensity = mu + alpha * beta * excitation
        if intensity <= 0 or not np.isfinite(intensity):
            return 1e12
        loglik += math.log(intensity)
        excitation += 1.0
        last_t = t
    compensator = mu * horizon + alpha * np.sum(1 - np.exp(-beta * (horizon - times)))
    return float(-(loglik - compensator))


def fit_hawkes_exponential(times: np.ndarray, horizon: float, beta_default: float) -> dict[str, float]:
    if len(times) < 5 or horizon <= 0:
        return {
            "mu": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "branching_ratio": np.nan,
            "event_count": int(len(times)),
            "horizon_days": float(horizon),
            "success": False,
            "negative_log_likelihood": np.nan,
        }
    baseline = max(len(times) / horizon, 1e-6)
    x0 = np.array([baseline * 0.7, 0.25, beta_default], dtype=float)
    result = minimize(
        hawkes_negative_log_likelihood,
        x0=x0,
        args=(times, horizon),
        method="L-BFGS-B",
        bounds=[(1e-8, 5.0), (1e-8, 0.98), (1e-4, 10.0)],
        options={"maxiter": 2000},
    )
    mu, alpha, beta = result.x
    return {
        "mu": float(mu),
        "alpha": float(alpha),
        "beta": float(beta),
        "branching_ratio": float(alpha),
        "event_count": int(len(times)),
        "horizon_days": float(horizon),
        "success": bool(result.success),
        "negative_log_likelihood": float(result.fun),
    }


def hawkes_intensity_on_grid(group: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    g = group.sort_values("timestamp").copy()
    mu, alpha, beta = params.get("mu", np.nan), params.get("alpha", np.nan), params.get("beta", np.nan)
    out = g[["asset_group", "symbol", "timestamp", "event_flag", "jump_event", "high_vol_event"]].copy()
    if not np.isfinite([mu, alpha, beta]).all():
        out["hawkes_intensity"] = np.nan
        out["poisson_intensity"] = np.nan
        out["hawkes_cluster_ratio"] = np.nan
        return out
    start = g["timestamp"].iloc[0]
    event_times, horizon = _event_times(g)
    poisson = len(event_times) / max(horizon, 1.0)
    intensities = []
    for ts in g["timestamp"]:
        t = (ts - start).total_seconds() / 86400.0
        prior = event_times[event_times < t]
        if len(prior):
            excitation = np.exp(-beta * (t - prior)).sum()
        else:
            excitation = 0.0
        intensities.append(mu + alpha * beta * excitation)
    out["hawkes_intensity"] = intensities
    out["poisson_intensity"] = poisson
    out["hawkes_cluster_ratio"] = out["hawkes_intensity"] / poisson if poisson > 0 else np.nan
    return out


def fit_hawkes_for_symbols(indicators: pd.DataFrame, cfg: RegimeAnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    intensity_frames: list[pd.DataFrame] = []
    symbols = indicators["symbol"].drop_duplicates().tolist()[: cfg.max_hawkes_symbols]

    for symbol in symbols:
        group = indicators[indicators["symbol"] == symbol].copy()
        times, horizon = _event_times(group)
        params = fit_hawkes_exponential(times, horizon, cfg.hawkes_beta_default)
        params.update({"asset_group": group["asset_group"].iloc[0], "symbol": symbol})
        summaries.append(params)
        intensity_frames.append(hawkes_intensity_on_grid(group, params))

    summary = pd.DataFrame(summaries).sort_values(["success", "branching_ratio"], ascending=[False, False])
    intensity = pd.concat(intensity_frames, ignore_index=True) if intensity_frames else pd.DataFrame()
    return summary, intensity


def load_env_key(name: str = "LSE_API_KEY", env_path: str | Path = ".env") -> str | None:
    if os.getenv(name):
        return os.getenv(name)
    path = Path(env_path)
    if not path.exists():
        return None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return os.getenv(name)


def fetch_lse_options(underlyings: Iterable[str], cfg: RegimeAnalysisConfig) -> pd.DataFrame:
    key = load_env_key("LSE_API_KEY")
    if not key:
        return pd.DataFrame()
    try:
        from lse import LSE
    except ModuleNotFoundError:
        return pd.DataFrame()

    client = LSE(api_key=key)
    frames: list[pd.DataFrame] = []
    for underlying in underlyings:
        try:
            rows = client.options(
                underlying,
                min_dte=cfg.option_min_dte,
                max_dte=cfg.option_max_dte,
                limit=cfg.option_limit,
            )
        except Exception:
            continue
        if rows:
            frames.append(pd.DataFrame(rows).assign(requested_underlying=underlying))
    if not frames:
        return pd.DataFrame()
    options = pd.concat(frames, ignore_index=True)
    for col in ["strike", "last_price", "volume_today", "premium_today", "underlying_price", "dte", "iv", "delta", "gamma", "theta", "vega", "rho"]:
        if col in options.columns:
            options[col] = pd.to_numeric(options[col], errors="coerce")
    for col in ["expiry", "last_trade_at", "updated_at"]:
        if col in options.columns:
            options[col] = pd.to_datetime(options[col], errors="coerce", utc=True)
    if {"strike", "underlying_price"}.issubset(options.columns):
        options["moneyness"] = options["strike"] / options["underlying_price"]
    return options


def summarize_option_indicators(options: pd.DataFrame, realized_indicators: pd.DataFrame) -> pd.DataFrame:
    if options.empty:
        return pd.DataFrame()
    opts = options.copy()
    opts = opts[np.isfinite(opts.get("iv", np.nan)) & (opts["iv"] > 0)]
    if opts.empty:
        return pd.DataFrame()

    latest_rv = (
        realized_indicators.sort_values("timestamp")
        .groupby("symbol")
        .tail(1)[["symbol", "rv_close_20", "rv_yang_zhang_20", "timestamp"]]
        .rename(columns={"symbol": "underlying", "timestamp": "rv_timestamp"})
    )

    records: list[dict[str, Any]] = []
    for underlying, group in opts.groupby("underlying"):
        g = group.dropna(subset=["iv", "dte", "moneyness"]).copy()
        if g.empty:
            continue
        atm = g.loc[(g["moneyness"] - 1.0).abs().idxmin()]
        near_expiry = g.groupby("expiry")["dte"].median().idxmin() if "expiry" in g else None
        far_expiry = g.groupby("expiry")["dte"].median().idxmax() if "expiry" in g else None
        near = g[g["expiry"] == near_expiry] if near_expiry is not None else g.nsmallest(100, "dte")
        far = g[g["expiry"] == far_expiry] if far_expiry is not None else g.nlargest(100, "dte")
        near_atm = near.loc[(near["moneyness"] - 1.0).abs().idxmin(), "iv"] if len(near) else np.nan
        far_atm = far.loc[(far["moneyness"] - 1.0).abs().idxmin(), "iv"] if len(far) else np.nan
        downside_puts = g[(g["contract_type"].astype(str).str.lower() == "put") & (g["moneyness"].between(0.80, 0.95))]
        upside_calls = g[(g["contract_type"].astype(str).str.lower() == "call") & (g["moneyness"].between(1.05, 1.20))]
        call_gex = g.loc[g["contract_type"].astype(str).str.lower() == "call", "gamma"].fillna(0) * g.loc[g["contract_type"].astype(str).str.lower() == "call", "volume_today"].fillna(0)
        put_gex = g.loc[g["contract_type"].astype(str).str.lower() == "put", "gamma"].fillna(0) * g.loc[g["contract_type"].astype(str).str.lower() == "put", "volume_today"].fillna(0)

        records.append(
            {
                "underlying": underlying,
                "contracts": int(len(g)),
                "expiries": int(g["expiry"].nunique()) if "expiry" in g else np.nan,
                "underlying_price": float(g["underlying_price"].median()),
                "atm_iv": float(atm["iv"]),
                "mean_iv": float(g["iv"].mean()),
                "median_iv": float(g["iv"].median()),
                "iv_dispersion": float(g["iv"].std()),
                "term_slope_far_minus_near": float(far_atm - near_atm) if np.isfinite([far_atm, near_atm]).all() else np.nan,
                "put_downside_minus_call_upside_skew": float(downside_puts["iv"].mean() - upside_calls["iv"].mean()) if len(downside_puts) and len(upside_calls) else np.nan,
                "net_gamma_volume_proxy": float(call_gex.sum() - put_gex.sum()),
                "vega_volume_proxy": float((g["vega"].fillna(0) * g["volume_today"].fillna(0)).sum()) if "vega" in g and "volume_today" in g else np.nan,
                "min_dte": float(g["dte"].min()),
                "max_dte": float(g["dte"].max()),
                "updated_at": g["updated_at"].max() if "updated_at" in g else pd.NaT,
            }
        )
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary
    summary = summary.merge(latest_rv, on="underlying", how="left")
    summary["iv_minus_rv20"] = summary["atm_iv"] - summary["rv_close_20"]
    summary["iv_minus_yz20"] = summary["atm_iv"] - summary["rv_yang_zhang_20"]
    return summary.sort_values("underlying")


def save_outputs(
    cfg: RegimeAnalysisConfig,
    indicators: pd.DataFrame,
    hmm_regimes: pd.DataFrame,
    hmm_summary: pd.DataFrame,
    hawkes_summary: pd.DataFrame,
    hawkes_intensity: pd.DataFrame,
    option_indicators: pd.DataFrame,
    options_raw: pd.DataFrame,
) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "indicators": cfg.output_dir / "market_indicators.parquet",
        "hmm_regimes": cfg.output_dir / "hmm_regimes.parquet",
        "hmm_summary": cfg.output_dir / "hmm_summary.csv",
        "hawkes_summary": cfg.output_dir / "hawkes_summary.csv",
        "hawkes_intensity": cfg.output_dir / "hawkes_intensity.parquet",
        "option_indicators": cfg.output_dir / "option_iv_indicators.csv",
        "options_raw": cfg.output_dir / "lse_options_raw.parquet",
    }
    indicators.to_parquet(paths["indicators"], index=False)
    hmm_regimes.to_parquet(paths["hmm_regimes"], index=False)
    hmm_summary.to_csv(paths["hmm_summary"], index=False)
    hawkes_summary.to_csv(paths["hawkes_summary"], index=False)
    hawkes_intensity.to_parquet(paths["hawkes_intensity"], index=False)
    option_indicators.to_csv(paths["option_indicators"], index=False)
    if not options_raw.empty:
        options_raw.to_parquet(paths["options_raw"], index=False)
    return paths
