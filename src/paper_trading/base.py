from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import pandas as pd


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    DAY = "day"


@dataclass
class Bar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    vwap: float | None = None


@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    fee: float = 0.0
    created_at: pd.Timestamp | None = None
    client_order_id: str | None = None


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float
    timestamp: pd.Timestamp


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    current_price: float = 0.0


@dataclass
class Account:
    cash: float
    equity: float
    buying_power: float = 0.0
    currency: str = "USD"
    initial_cash: float = 0.0


@runtime_checkable
class BrokerClient(Protocol):
    name: str
    is_live: bool

    def get_account(self) -> Account: ...

    def get_position(self, symbol: str) -> Position | None: ...

    def get_latest_bar(self, symbol: str) -> Bar: ...

    def get_bars(self, symbol: str, limit: int, timeframe: str = "1m") -> pd.DataFrame: ...

    def get_market_status(self) -> dict: ...

    def submit_market_order(self, symbol: str, side: OrderSide, qty: float) -> Order: ...

    def submit_limit_order(self, symbol: str, side: OrderSide, qty: float, limit_price: float) -> Order: ...

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool: ...

    def get_order(self, order_id: str, symbol: str | None = None) -> Order: ...

    def close_all(self, symbol: str | None = None) -> list[Order]: ...


@dataclass
class EngineConfig:
    symbol: str
    timeframe: str = "1m"
    bar_window: int = 200
    min_notional: float = 1.0
    min_qty: float = 0.0
    long_only: bool = True
    max_weight: float = 1.0
    poll_seconds: float = 60.0
    initial_cash: float = 10_000.0
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    periods_per_year: int = 365 * 24 * 60
    state_path: str = "paper_trading/state/paper_trading.sqlite"
    max_steps: int | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


@dataclass
class EngineSnapshot:
    timestamp: pd.Timestamp
    close: float
    signal: float
    target_weight: float
    position: float
    cash: float
    equity: float
    bh_equity: float
    order_id: str | None = None


@dataclass
class EngineResult:
    snapshots: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    metrics: dict[str, float]
    final_equity: float
    bh_final_equity: float

    def summary(self) -> dict[str, float]:
        out = dict(self.metrics)
        out["final_equity"] = self.final_equity
        out["bh_final_equity"] = self.bh_final_equity
        out["excess_return_pct"] = (self.final_equity / self.bh_final_equity - 1) * 100 if self.bh_final_equity else 0.0
        return out


_BARS_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    rows = []
    for b in bars:
        rows.append({
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "vwap": b.vwap if b.vwap is not None else (b.high + b.low + b.close) / 3.0,
        })
    df = pd.DataFrame(rows, columns=_BARS_COLUMNS)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want in _BARS_COLUMNS:
        if want not in df.columns and want in cols:
            rename[cols[want]] = want
    if rename:
        df = df.rename(columns=rename)
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
