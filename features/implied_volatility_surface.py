from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OptionType = Literal["call", "put", "both"]


@dataclass(frozen=True)
class SurfaceConfig:
    ticker: str = "AAPL"
    option_type: OptionType = "call"
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 540
    min_moneyness: float = 0.70
    max_moneyness: float = 1.30
    min_iv: float = 0.01
    max_iv: float = 3.00
    max_spread_pct: float = 0.35
    min_open_interest: int = 0
    grid_strikes: int = 80
    grid_maturities: int = 50
    interpolation_method: Literal["linear", "nearest", "cubic"] = "linear"
    output_dir: Path = Path("images/implied_vol_surface")


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    option_type: str,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    option_type = normalize_option_type(option_type)
    if not np.isfinite([spot, strike, tau, volatility]).all():
        return np.nan
    if spot <= 0 or strike <= 0 or tau <= 0 or volatility <= 0:
        return np.nan

    sqrt_tau = math.sqrt(tau)
    discount = math.exp(-rate * tau)
    dividend_discount = math.exp(-dividend_yield * tau)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * tau
    ) / (volatility * sqrt_tau)
    d2 = d1 - volatility * sqrt_tau

    if option_type == "call":
        return spot * dividend_discount * normal_cdf(d1) - strike * discount * normal_cdf(d2)
    return strike * discount * normal_cdf(-d2) - spot * dividend_discount * normal_cdf(-d1)


def implied_volatility_bisection(
    price: float,
    option_type: str,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    low: float = 1e-4,
    high: float = 5.0,
    tolerance: float = 1e-6,
    max_iter: int = 100,
) -> float:
    if not np.isfinite([price, spot, strike, tau, rate, dividend_yield]).all():
        return np.nan
    if price <= 0 or spot <= 0 or strike <= 0 or tau <= 0:
        return np.nan

    option_type = normalize_option_type(option_type)
    discount = math.exp(-rate * tau)
    dividend_discount = math.exp(-dividend_yield * tau)
    if option_type == "call":
        intrinsic = max(spot * dividend_discount - strike * discount, 0.0)
    else:
        intrinsic = max(strike * discount - spot * dividend_discount, 0.0)
    if price + tolerance < intrinsic:
        return np.nan

    low_price = black_scholes_price(
        option_type, spot, strike, tau, rate, dividend_yield, low
    )
    high_price = black_scholes_price(
        option_type, spot, strike, tau, rate, dividend_yield, high
    )
    if not np.isfinite([low_price, high_price]).all():
        return np.nan
    if price < low_price - tolerance or price > high_price + tolerance:
        return np.nan

    lo, hi = low, high
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_price = black_scholes_price(
            option_type, spot, strike, tau, rate, dividend_yield, mid
        )
        if not np.isfinite(mid_price):
            return np.nan
        error = mid_price - price
        if abs(error) < tolerance:
            return mid
        if error > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def normalize_option_type(value: object) -> Literal["call", "put"]:
    text = str(value).strip().lower()
    if text in {"c", "call", "calls"}:
        return "call"
    if text in {"p", "put", "puts"}:
        return "put"
    raise ValueError(f"Unsupported option_type: {value!r}")


def _canonical_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_COLUMN_ALIASES = {
    "quotedate": "quote_date",
    "date": "quote_date",
    "asof": "quote_date",
    "snapshotdate": "quote_date",
    "expiration": "expiration",
    "expiry": "expiration",
    "expirationdate": "expiration",
    "maturity": "expiration",
    "strike": "strike",
    "k": "strike",
    "optiontype": "option_type",
    "type": "option_type",
    "right": "option_type",
    "putcall": "option_type",
    "underlyingprice": "underlying_price",
    "spot": "underlying_price",
    "spotprice": "underlying_price",
    "underlying": "underlying_price",
    "bid": "bid",
    "ask": "ask",
    "mid": "mid_price",
    "midprice": "mid_price",
    "mark": "mid_price",
    "last": "last_price",
    "lastprice": "last_price",
    "impliedvolatility": "implied_volatility",
    "iv": "implied_volatility",
    "volume": "volume",
    "openinterest": "open_interest",
    "contractsymbol": "contract_symbol",
    "lasttradedate": "last_trade_date",
}


def _rename_to_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    used: set[str] = set()
    for column in df.columns:
        canonical = _COLUMN_ALIASES.get(_canonical_column(str(column)))
        if canonical and canonical not in used:
            rename[column] = canonical
            used.add(canonical)
    return df.rename(columns=rename)


def _to_utc_naive_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None)


def load_option_quotes(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported option quote file extension: {path.suffix}")


def fetch_yahoo_option_snapshot(
    ticker: str,
    config: SurfaceConfig,
    max_expirations: int | None = None,
) -> pd.DataFrame:
    import yfinance as yf

    yahoo_ticker = yf.Ticker(ticker)
    history = yahoo_ticker.history(period="5d", auto_adjust=False)
    if history.empty or "Close" not in history:
        raise RuntimeError(f"Could not fetch a recent underlying price for {ticker}")
    spot = float(history["Close"].dropna().iloc[-1])

    expirations = list(yahoo_ticker.options or [])
    if max_expirations is not None:
        expirations = expirations[:max_expirations]
    if not expirations:
        raise RuntimeError(f"No option expirations returned by yfinance for {ticker}")

    frames: list[pd.DataFrame] = []
    quote_date = pd.Timestamp.utcnow().tz_convert(None)
    include_calls = config.option_type in {"call", "both"}
    include_puts = config.option_type in {"put", "both"}

    for expiration in expirations:
        try:
            chain = yahoo_ticker.option_chain(expiration)
        except Exception:
            continue
        if include_calls and not chain.calls.empty:
            frames.append(
                chain.calls.assign(
                    option_type="call",
                    expiration=expiration,
                    quote_date=quote_date,
                    underlying_price=spot,
                )
            )
        if include_puts and not chain.puts.empty:
            frames.append(
                chain.puts.assign(
                    option_type="put",
                    expiration=expiration,
                    quote_date=quote_date,
                    underlying_price=spot,
                )
            )

    if not frames:
        raise RuntimeError(f"No option chains could be fetched for {ticker}")
    return pd.concat(frames, ignore_index=True)


def build_synthetic_option_timeseries(
    config: SurfaceConfig,
    quote_dates: int = 12,
    spot: float = 100.0,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=quote_dates)
    maturity_days = np.array([14, 30, 60, 90, 180, 365])
    strike_multipliers = np.linspace(0.72, 1.28, 35)
    rows: list[dict[str, object]] = []

    for i, quote_date in enumerate(dates):
        spot_i = spot * (1.0 + 0.015 * math.sin(i / 2.0) + rng.normal(0.0, 0.004))
        regime_shift = 0.015 * math.sin(i / 3.0)
        for days in maturity_days:
            tau = days / 365.25
            expiration = quote_date + pd.Timedelta(days=int(days))
            for mult in strike_multipliers:
                strike = round(spot_i * mult, 2)
                log_m = math.log(strike / spot_i)
                base_iv = (
                    0.21
                    + regime_shift
                    + 0.035 * math.sqrt(tau)
                    + 0.17 * log_m * log_m
                    - 0.055 * log_m
                )
                iv = float(np.clip(base_iv + rng.normal(0.0, 0.004), 0.08, 0.75))
                for option_type in ("call", "put") if config.option_type == "both" else (config.option_type,):
                    mid = black_scholes_price(
                        option_type,
                        spot_i,
                        strike,
                        tau,
                        config.risk_free_rate,
                        config.dividend_yield,
                        iv,
                    )
                    spread = max(0.03, 0.035 * mid)
                    rows.append(
                        {
                            "quote_date": quote_date,
                            "expiration": expiration,
                            "strike": strike,
                            "option_type": option_type,
                            "underlying_price": spot_i,
                            "bid": max(mid - spread / 2.0, 0.01),
                            "ask": mid + spread / 2.0,
                            "last_price": mid,
                            "implied_volatility": iv,
                            "volume": int(rng.integers(5, 500)),
                            "open_interest": int(rng.integers(25, 5000)),
                            "source": "synthetic_demo",
                        }
                    )
    return pd.DataFrame(rows)


def standardize_option_quotes(raw_quotes: pd.DataFrame, config: SurfaceConfig) -> pd.DataFrame:
    if raw_quotes.empty:
        raise ValueError("Option quote data is empty")

    df = _rename_to_canonical_columns(raw_quotes.copy())
    required = {
        "quote_date",
        "expiration",
        "strike",
        "option_type",
        "underlying_price",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required option quote columns: {missing}")

    df["quote_date"] = _to_utc_naive_series(df["quote_date"])
    df["expiration"] = _to_utc_naive_series(df["expiration"])
    for column in [
        "strike",
        "underlying_price",
        "bid",
        "ask",
        "mid_price",
        "last_price",
        "implied_volatility",
        "volume",
        "open_interest",
    ]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["option_type"] = df["option_type"].map(normalize_option_type)
    df["dte"] = (
        df["expiration"].dt.normalize() - df["quote_date"].dt.normalize()
    ).dt.days.astype("float")
    df["tau"] = df["dte"] / 365.25
    df["moneyness"] = df["strike"] / df["underlying_price"]

    if "mid_price" not in df:
        df["mid_price"] = np.nan
    has_bid_ask = {"bid", "ask"}.issubset(df.columns)
    if has_bid_ask:
        quoted_mid = np.where(
            (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"]),
            0.5 * (df["bid"] + df["ask"]),
            np.nan,
        )
        df["mid_price"] = df["mid_price"].fillna(pd.Series(quoted_mid, index=df.index))
        df["spread_pct"] = np.where(
            (df["mid_price"] > 0) & (df["ask"] >= df["bid"]),
            (df["ask"] - df["bid"]) / df["mid_price"],
            np.nan,
        )
    else:
        df["spread_pct"] = np.nan
    if "last_price" in df:
        df["mid_price"] = df["mid_price"].fillna(df["last_price"])

    if "implied_volatility" not in df:
        df["implied_volatility"] = np.nan
    df["iv_source"] = np.where(df["implied_volatility"].notna(), "provided", "missing")
    return df


def fill_missing_implied_volatility(
    quotes: pd.DataFrame,
    config: SurfaceConfig,
) -> pd.DataFrame:
    df = quotes.copy()
    invalid_iv = (
        ~np.isfinite(df["implied_volatility"])
        | (df["implied_volatility"] < config.min_iv)
        | (df["implied_volatility"] > config.max_iv)
    )
    rows_to_solve = df.index[invalid_iv & (df["mid_price"] > 0)]

    solved: dict[int, float] = {}
    for idx in rows_to_solve:
        row = df.loc[idx]
        solved[idx] = implied_volatility_bisection(
            price=float(row["mid_price"]),
            option_type=str(row["option_type"]),
            spot=float(row["underlying_price"]),
            strike=float(row["strike"]),
            tau=float(row["tau"]),
            rate=config.risk_free_rate,
            dividend_yield=config.dividend_yield,
        )

    if solved:
        solved_series = pd.Series(solved)
        valid_solved = solved_series.dropna()
        df.loc[valid_solved.index, "implied_volatility"] = valid_solved
        df.loc[valid_solved.index, "iv_source"] = "model_inverted"
    return df


def clean_option_quotes(raw_quotes: pd.DataFrame, config: SurfaceConfig) -> pd.DataFrame:
    df = standardize_option_quotes(raw_quotes, config)
    df = fill_missing_implied_volatility(df, config)

    mask = (
        df["quote_date"].notna()
        & df["expiration"].notna()
        & np.isfinite(df["strike"])
        & np.isfinite(df["underlying_price"])
        & np.isfinite(df["implied_volatility"])
        & (df["dte"] >= config.min_days_to_expiry)
        & (df["dte"] <= config.max_days_to_expiry)
        & (df["moneyness"] >= config.min_moneyness)
        & (df["moneyness"] <= config.max_moneyness)
        & (df["implied_volatility"] >= config.min_iv)
        & (df["implied_volatility"] <= config.max_iv)
    )
    if config.max_spread_pct is not None:
        mask &= df["spread_pct"].isna() | (df["spread_pct"] <= config.max_spread_pct)
    if config.min_open_interest and "open_interest" in df:
        mask &= df["open_interest"].fillna(0) >= config.min_open_interest

    keep_columns = [
        column
        for column in [
            "quote_date",
            "expiration",
            "dte",
            "tau",
            "strike",
            "underlying_price",
            "moneyness",
            "option_type",
            "bid",
            "ask",
            "mid_price",
            "last_price",
            "spread_pct",
            "implied_volatility",
            "iv_source",
            "volume",
            "open_interest",
            "contract_symbol",
            "source",
        ]
        if column in df.columns
    ]
    cleaned = df.loc[mask, keep_columns].copy()
    cleaned["quote_date"] = cleaned["quote_date"].dt.normalize()
    cleaned = cleaned.drop_duplicates(
        subset=["quote_date", "expiration", "strike", "option_type"],
        keep="last",
    )
    return cleaned.sort_values(["quote_date", "expiration", "strike", "option_type"])


def snapshot_dates(clean_quotes: pd.DataFrame) -> list[pd.Timestamp]:
    dates = clean_quotes["quote_date"].dropna().drop_duplicates().sort_values()
    return [pd.Timestamp(date) for date in dates]


def select_surface_snapshot(
    clean_quotes: pd.DataFrame,
    quote_date: str | pd.Timestamp | None = None,
    option_type: str | None = None,
) -> pd.DataFrame:
    if clean_quotes.empty:
        raise ValueError("No cleaned option quotes available")

    dates = snapshot_dates(clean_quotes)
    if not dates:
        raise ValueError("No quote dates available")
    if quote_date is None:
        selected_date = dates[-1]
    else:
        target = pd.to_datetime(quote_date).normalize()
        candidates = [date for date in dates if date <= target]
        selected_date = candidates[-1] if candidates else dates[0]

    snapshot = clean_quotes[clean_quotes["quote_date"] == selected_date].copy()
    if option_type and option_type != "both":
        option_type = normalize_option_type(option_type)
        snapshot = snapshot[snapshot["option_type"] == option_type].copy()
    if snapshot.empty:
        raise ValueError(f"No quotes for {selected_date.date()} and option_type={option_type}")
    return snapshot


def _axis_values(snapshot: pd.DataFrame, x_axis: Literal["strike", "moneyness"]) -> np.ndarray:
    if x_axis == "strike":
        return snapshot["strike"].to_numpy(dtype=float)
    if x_axis == "moneyness":
        return snapshot["moneyness"].to_numpy(dtype=float)
    raise ValueError("x_axis must be 'strike' or 'moneyness'")


def interpolate_surface_grid(
    snapshot: pd.DataFrame,
    config: SurfaceConfig,
    x_axis: Literal["strike", "moneyness"] = "strike",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(snapshot) < 4:
        raise ValueError("At least four option quotes are required for surface interpolation")

    x = _axis_values(snapshot, x_axis)
    y = snapshot["dte"].to_numpy(dtype=float)
    tau = snapshot["tau"].to_numpy(dtype=float)
    total_variance = np.square(snapshot["implied_volatility"].to_numpy(dtype=float)) * tau

    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), config.grid_strikes)
    y_grid = np.linspace(np.nanmin(y), np.nanmax(y), config.grid_maturities)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)

    try:
        from scipy.interpolate import griddata

        grid_total_variance = griddata(
            np.column_stack([x, y]),
            total_variance,
            (grid_x, grid_y),
            method=config.interpolation_method,
        )
        nearest = griddata(
            np.column_stack([x, y]),
            total_variance,
            (grid_x, grid_y),
            method="nearest",
        )
        grid_total_variance = np.where(np.isnan(grid_total_variance), nearest, grid_total_variance)
    except Exception:
        import matplotlib.tri as tri

        triangulation = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triangulation, total_variance)
        grid_total_variance = interpolator(grid_x, grid_y).filled(np.nan)

    tau_grid = np.maximum(grid_y / 365.25, 1e-8)
    grid_iv = np.sqrt(np.maximum(grid_total_variance / tau_grid, 0.0))
    observed_iv = snapshot["implied_volatility"].to_numpy(dtype=float)
    grid_iv = np.clip(grid_iv, np.nanmin(observed_iv), np.nanmax(observed_iv))
    return grid_x, grid_y, grid_iv


def animate_surface_timeseries(
    clean_quotes: pd.DataFrame,
    config: SurfaceConfig,
    title: str,
    option_type: str | None = None,
    x_axis: Literal["strike", "moneyness"] = "strike",
    max_frames: int = 48,
    fps: int = 4,
    save_path: str | Path | None = None,
) -> Path:
    from matplotlib import cm, colors
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    if clean_quotes.empty:
        raise ValueError("No cleaned option quotes available for animation")

    animation_quotes = clean_quotes.copy()
    if option_type and option_type != "both":
        option_type = normalize_option_type(option_type)
        animation_quotes = animation_quotes[animation_quotes["option_type"] == option_type].copy()
    if animation_quotes.empty:
        raise ValueError(f"No quotes available for animation option_type={option_type}")

    dates = snapshot_dates(animation_quotes)
    if len(dates) < 2:
        raise ValueError("At least two quote dates are required to animate a surface over time")
    if max_frames and len(dates) > max_frames:
        frame_idx = np.linspace(0, len(dates) - 1, max_frames).round().astype(int)
        dates = [dates[i] for i in sorted(set(frame_idx))]

    frames: list[dict[str, object]] = []
    for date in dates:
        snapshot = select_surface_snapshot(animation_quotes, quote_date=date, option_type=option_type)
        if len(snapshot) < 4:
            continue
        grid_x, grid_y, grid_iv = interpolate_surface_grid(snapshot, config, x_axis=x_axis)
        frames.append(
            {
                "date": date,
                "snapshot": snapshot,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_iv_pct": grid_iv * 100.0,
                "x_points": _axis_values(snapshot, x_axis),
                "y_points": snapshot["dte"].to_numpy(dtype=float),
                "z_points": snapshot["implied_volatility"].to_numpy(dtype=float) * 100.0,
            }
        )

    if len(frames) < 2:
        raise ValueError("Fewer than two usable animation frames after cleaning")

    all_x = np.concatenate([frame["x_points"] for frame in frames])
    all_y = np.concatenate([frame["y_points"] for frame in frames])
    all_z = np.concatenate([frame["z_points"] for frame in frames])
    x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    z_min, z_max = float(np.nanmin(all_z)), float(np.nanmax(all_z))
    z_pad = max((z_max - z_min) * 0.08, 1.0)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    norm = colors.Normalize(vmin=z_min, vmax=z_max)
    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.64, pad=0.10, label="Implied volatility (%)")

    x_label = "Strike price" if x_axis == "strike" else "Moneyness K/S"

    def update(frame_number: int):
        frame = frames[frame_number]
        ax.clear()
        ax.plot_surface(
            frame["grid_x"],
            frame["grid_y"],
            frame["grid_iv_pct"],
            cmap="viridis",
            norm=norm,
            linewidth=0,
            antialiased=True,
            alpha=0.92,
        )
        ax.scatter(
            frame["x_points"],
            frame["y_points"],
            frame["z_points"],
            color="black",
            s=9,
            alpha=0.32,
            label="Listed quotes",
        )
        date = pd.Timestamp(frame["date"]).date()
        ax.set_title(f"{title}\n{date} ({frame_number + 1}/{len(frames)})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Maturity, calendar days")
        ax.set_zlabel("Implied volatility (%)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(max(0.0, z_min - z_pad), z_max + z_pad)
        ax.view_init(elev=27, azim=-132 + frame_number * 1.5)
        ax.legend(loc="upper left")
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=max(1, int(1000 / max(fps, 1))),
        repeat=True,
        blit=False,
    )

    if save_path is None:
        suffix = ".mp4" if FFMpegWriter.isAvailable() else ".gif"
        save_path = config.output_dir / f"{config.ticker}_implied_vol_surface_timeseries{suffix}"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.suffix.lower() == ".mp4" and FFMpegWriter.isAvailable():
        animation.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=140)
    elif save_path.suffix.lower() == ".gif" or not FFMpegWriter.isAvailable():
        if save_path.suffix.lower() != ".gif":
            save_path = save_path.with_suffix(".gif")
        animation.save(save_path, writer=PillowWriter(fps=fps), dpi=120)
    else:
        raise ValueError(f"Unsupported animation output suffix: {save_path.suffix}")

    plt.close(fig)
    return save_path


def plot_surface_3d(
    snapshot: pd.DataFrame,
    config: SurfaceConfig,
    title: str,
    x_axis: Literal["strike", "moneyness"] = "strike",
    save_path: str | Path | None = None,
) -> plt.Figure:
    x = _axis_values(snapshot, x_axis)
    y = snapshot["dte"].to_numpy(dtype=float)
    z = snapshot["implied_volatility"].to_numpy(dtype=float) * 100.0

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    try:
        grid_x, grid_y, grid_iv = interpolate_surface_grid(snapshot, config, x_axis=x_axis)
        surface = ax.plot_surface(
            grid_x,
            grid_y,
            grid_iv * 100.0,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            alpha=0.90,
        )
        fig.colorbar(surface, ax=ax, shrink=0.64, pad=0.10, label="Implied volatility (%)")
    except Exception:
        surface = ax.plot_trisurf(x, y, z, cmap="viridis", linewidth=0.2, alpha=0.90)
        fig.colorbar(surface, ax=ax, shrink=0.64, pad=0.10, label="Implied volatility (%)")

    ax.scatter(x, y, z, color="black", s=10, alpha=0.35, label="Listed quotes")
    ax.set_xlabel("Strike price" if x_axis == "strike" else "Moneyness K/S")
    ax.set_ylabel("Maturity, calendar days")
    ax.set_zlabel("Implied volatility (%)")
    ax.set_title(title)
    ax.view_init(elev=27, azim=-132)
    ax.legend(loc="upper left")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_surface_heatmap(
    snapshot: pd.DataFrame,
    config: SurfaceConfig,
    title: str,
    x_axis: Literal["strike", "moneyness"] = "strike",
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 8))
    try:
        grid_x, grid_y, grid_iv = interpolate_surface_grid(snapshot, config, x_axis=x_axis)
        contour = ax.contourf(grid_x, grid_y, grid_iv * 100.0, levels=28, cmap="viridis")
    except Exception:
        x = _axis_values(snapshot, x_axis)
        contour = ax.tricontourf(
            x,
            snapshot["dte"].to_numpy(dtype=float),
            snapshot["implied_volatility"].to_numpy(dtype=float) * 100.0,
            levels=28,
            cmap="viridis",
        )
    ax.scatter(
        _axis_values(snapshot, x_axis),
        snapshot["dte"],
        s=8,
        color="white",
        alpha=0.45,
        linewidths=0,
    )
    ax.set_xlabel("Strike price" if x_axis == "strike" else "Moneyness K/S")
    ax.set_ylabel("Maturity, calendar days")
    ax.set_title(title)
    fig.colorbar(contour, ax=ax, label="Implied volatility (%)")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_smile_by_maturity(
    snapshot: pd.DataFrame,
    title: str,
    target_maturities: Iterable[int] = (30, 60, 90, 180, 365),
    x_axis: Literal["strike", "moneyness"] = "strike",
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 7))
    used_expirations: set[pd.Timestamp] = set()
    for target in target_maturities:
        nearest_idx = (snapshot["dte"] - target).abs().idxmin()
        expiration = pd.Timestamp(snapshot.loc[nearest_idx, "expiration"])
        if expiration in used_expirations:
            continue
        used_expirations.add(expiration)
        curve = snapshot[snapshot["expiration"] == expiration].sort_values("strike")
        x = _axis_values(curve, x_axis)
        label = f"{int(curve['dte'].median())}d exp {expiration.date()}"
        ax.plot(
            x,
            curve["implied_volatility"] * 100.0,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=label,
        )
    ax.set_xlabel("Strike price" if x_axis == "strike" else "Moneyness K/S")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def summarize_surface_features(clean_quotes: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_cols = ["quote_date", "option_type"]
    for (quote_date, option_type), group in clean_quotes.groupby(group_cols):
        group = group.copy()
        atm_idx = (group["moneyness"] - 1.0).abs().idxmin()
        near_exp = group.groupby("expiration")["dte"].median().idxmin()
        far_exp = group.groupby("expiration")["dte"].median().idxmax()
        near = group[group["expiration"] == near_exp]
        far = group[group["expiration"] == far_exp]
        near_atm = near.loc[(near["moneyness"] - 1.0).abs().idxmin(), "implied_volatility"]
        far_atm = far.loc[(far["moneyness"] - 1.0).abs().idxmin(), "implied_volatility"]
        downside = group[(group["moneyness"] >= 0.85) & (group["moneyness"] <= 0.95)]
        upside = group[(group["moneyness"] >= 1.05) & (group["moneyness"] <= 1.15)]
        records.append(
            {
                "quote_date": quote_date,
                "option_type": option_type,
                "quote_count": int(len(group)),
                "expiration_count": int(group["expiration"].nunique()),
                "spot": float(group["underlying_price"].median()),
                "atm_iv": float(group.loc[atm_idx, "implied_volatility"]),
                "mean_iv": float(group["implied_volatility"].mean()),
                "median_iv": float(group["implied_volatility"].median()),
                "term_slope_far_minus_near": float(far_atm - near_atm),
                "downside_minus_upside_skew": float(downside["implied_volatility"].mean() - upside["implied_volatility"].mean()),
                "min_dte": float(group["dte"].min()),
                "max_dte": float(group["dte"].max()),
                "min_strike": float(group["strike"].min()),
                "max_strike": float(group["strike"].max()),
            }
        )
    return pd.DataFrame(records).sort_values(["option_type", "quote_date"])


def plot_feature_timeseries(
    features: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    metrics = [
        ("atm_iv", "ATM IV"),
        ("term_slope_far_minus_near", "Term slope"),
        ("downside_minus_upside_skew", "Downside-upside skew"),
    ]
    for ax, (column, label) in zip(axes, metrics):
        for option_type, group in features.groupby("option_type"):
            ax.plot(group["quote_date"], group[column] * 100.0, marker="o", label=option_type)
        ax.set_ylabel(f"{label} (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Quote date")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig
