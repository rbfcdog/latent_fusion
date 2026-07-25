from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any

import pandas as pd

from .base import (
    Account,
    Bar,
    BrokerClient,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
    normalize_bars,
)

logger = logging.getLogger(__name__)

_PAPER_URL = "https://paper-api.alpaca.markets"
_LIVE_URL = "https://api.alpaca.markets"

_TF_MAP = {
    "1m": ("1Min", "1Min"), "5m": ("5Min", "5Min"), "15m": ("15Min", "15Min"),
    "1h": ("1Hour", "1Hour"), "1d": ("1Day", "1Day"),
}


def _require_alpaca() -> Any:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import (
            MarketOrderRequest, LimitOrderRequest, GetOrdersRequest,
        )
        from alpaca.trading.enums import OrderSide as ASide, TimeInForce as ATIF, QueryOrderStatus
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as e:
        raise ImportError(
            "alpaca-py is required for AlpacaBroker. Install with: uv add alpaca-py"
        ) from e
    return (TradingClient, MarketOrderRequest, LimitOrderRequest, GetOrdersRequest,
            ASide, ATIF, QueryOrderStatus, StockHistoricalDataClient, StockBarsRequest, TimeFrame)


def _ts(v: Any) -> pd.Timestamp:
    if v is None:
        return pd.Timestamp.now(tz="UTC")
    if isinstance(v, pd.Timestamp):
        return v if v.tzinfo else v.tz_localize("UTC")
    s = str(v)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return pd.Timestamp(s)
    except Exception:
        return pd.Timestamp.now(tz="UTC")


class AlpacaBroker:
    name = "alpaca"
    is_live: bool

    def __init__(self, api_key: str | None = None, secret_key: str | None = None, paper: bool = True):
        (TradingClient, MarketOrderRequest, LimitOrderRequest, GetOrdersRequest,
         ASide, ATIF, QueryOrderStatus, StockHistoricalDataClient,
         StockBarsRequest, TimeFrame) = _require_alpaca()

        self._TradingClient = TradingClient
        self._MarketOrderRequest = MarketOrderRequest
        self._LimitOrderRequest = LimitOrderRequest
        self._GetOrdersRequest = GetOrdersRequest
        self._ASide = ASide
        self._ATIF = ATIF
        self._QueryOrderStatus = QueryOrderStatus
        self._StockHistoricalDataClient = StockHistoricalDataClient
        self._StockBarsRequest = StockBarsRequest
        self._TimeFrame = TimeFrame

        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise ValueError("Alpaca API credentials required: set ALPACA_API_KEY / ALPACA_API_SECRET")
        self.is_live = not paper
        url = _LIVE_URL if self.is_live else _PAPER_URL
        self._client = TradingClient(api_key=key, secret_key=secret, paper=paper)
        self._data = StockHistoricalDataClient(api_key=key, secret_key=secret)
        logger.info("AlpacaBroker ready (paper=%s url=%s)", paper, url)

    def _map_tif(self, tif: TimeInForce) -> Any:
        return self._ATIF(tif.value.upper())

    def _map_side(self, side: OrderSide) -> Any:
        return self._ASide.BUY if side == OrderSide.BUY else self._ASide.SELL

    def _map_tf(self, timeframe: str) -> Any:
        tf = _TF_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"unsupported timeframe {timeframe}; choose from {list(_TF_MAP)}")
        attr = {"1m": "Minute", "5m": "Minute", "15m": "Minute", "1h": "Hour", "1d": "Day"}[timeframe]
        n = int(timeframe[:-1]) if timeframe != "1d" else 1
        tf_obj = getattr(self._TimeFrame, attr, self._TimeFrame.Minute)
        return tf_obj if n == 1 else tf_obj * n

    def get_account(self) -> Account:
        a = self._client.get_account()
        return Account(
            cash=float(getattr(a, "cash", 0) or 0),
            equity=float(getattr(a, "equity", 0) or 0),
            buying_power=float(getattr(a, "buying_power", 0) or 0),
            currency=getattr(a, "currency", "USD") or "USD",
            initial_cash=float(getattr(a, "last_equity", getattr(a, "equity", 0)) or 0),
        )

    def get_position(self, symbol: str) -> Position | None:
        try:
            p = self._client.get_open_position(symbol)
        except Exception:
            return None
        qty = float(getattr(p, "qty", 0) or 0)
        if abs(qty) < 1e-12:
            return None
        return Position(
            symbol=symbol, qty=qty,
            avg_entry_price=float(getattr(p, "avg_entry_price", 0) or 0),
            market_value=float(getattr(p, "market_value", 0) or 0),
            current_price=float(getattr(p, "current_price", 0) or 0),
            unrealized_pl=float(getattr(p, "unrealized_pl", 0) or 0),
        )

    def _bar_from_obj(self, b: Any) -> Bar:
        return Bar(
            timestamp=_ts(getattr(b, "timestamp", None)),
            open=float(getattr(b, "open", 0) or 0),
            high=float(getattr(b, "high", 0) or 0),
            low=float(getattr(b, "low", 0) or 0),
            close=float(getattr(b, "close", 0) or 0),
            volume=float(getattr(b, "volume", 0) or 0),
            vwap=float(getattr(b, "vwap", 0) or 0) or None,
        )

    def get_latest_bar(self, symbol: str) -> Bar:
        req = self._StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=self._map_tf("1m"), limit=1,
        )
        bars = self._data.get_stock_bars(req)
        try:
            blk = bars[symbol]
        except Exception:
            blk = bars
        lst = list(blk)
        if not lst:
            raise RuntimeError(f"no bars returned for {symbol}")
        return self._bar_from_obj(lst[-1])

    def get_bars(self, symbol: str, limit: int, timeframe: str = "1m") -> pd.DataFrame:
        req = self._StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=self._map_tf(timeframe), limit=limit,
        )
        bars = self._data.get_stock_bars(req)
        try:
            blk = bars[symbol]
        except Exception:
            blk = bars
        rows = []
        for b in blk:
            rows.append({
                "timestamp": _ts(getattr(b, "timestamp", None)),
                "open": float(getattr(b, "open", 0) or 0),
                "high": float(getattr(b, "high", 0) or 0),
                "low": float(getattr(b, "low", 0) or 0),
                "close": float(getattr(b, "close", 0) or 0),
                "volume": float(getattr(b, "volume", 0) or 0),
                "vwap": float(getattr(b, "vwap", 0) or 0) or None,
            })
        return normalize_bars(pd.DataFrame(rows))

    def get_market_status(self) -> dict:
        try:
            clock = self._client.get_clock()
            return {"open": bool(getattr(clock, "is_open", False)),
                    "next_open": str(getattr(clock, "next_open", "")),
                    "next_close": str(getattr(clock, "next_close", ""))}
        except Exception as e:
            logger.warning("get_clock failed: %s", e)
            return {"open": False, "error": str(e)}

    def _order_from_obj(self, o: Any) -> Order:
        status_map = {
            "new": OrderStatus.OPEN, "accepted": OrderStatus.OPEN, "pending_new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIAL, "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED, "cancelled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED, "expired": OrderStatus.REJECTED,
        }
        raw_status = str(getattr(o, "status", "pending")).lower()
        return Order(
            id=str(getattr(o, "id", uuid.uuid4().hex)),
            symbol=str(getattr(o, "symbol", "")),
            side=OrderSide.BUY if str(getattr(o, "side", "buy")).lower() == "buy" else OrderSide.SELL,
            qty=float(getattr(o, "qty", 0) or 0),
            order_type=OrderType.MARKET if "market" in str(getattr(o, "order_type", "market")).lower() else OrderType.LIMIT,
            limit_price=float(getattr(o, "limit_price", 0) or 0) or None,
            time_in_force=TimeInForce.GTC,
            status=status_map.get(raw_status, OrderStatus.PENDING),
            filled_qty=float(getattr(o, "filled_qty", 0) or 0),
            filled_avg_price=float(getattr(o, "filled_avg_price", 0) or 0) or None,
            fee=abs(float(getattr(o, "fees", 0) or 0)),
            created_at=_ts(getattr(o, "created_at", None)),
            client_order_id=getattr(o, "client_order_id", None),
        )

    def submit_market_order(self, symbol: str, side: OrderSide, qty: float) -> Order:
        req = self._MarketOrderRequest(
            symbol=symbol, qty=qty, side=self._map_side(side),
            time_in_force=self._ATIF.DAY,
        )
        o = self._client.submit_order(req)
        return self._order_from_obj(o)

    def submit_limit_order(self, symbol: str, side: OrderSide, qty: float, limit_price: float) -> Order:
        req = self._LimitOrderRequest(
            symbol=symbol, qty=qty, side=self._map_side(side),
            time_in_force=self._ATIF.DAY, limit_price=limit_price,
        )
        o = self._client.submit_order(req)
        return self._order_from_obj(o)

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.warning("cancel_order failed: %s", e)
            return False

    def get_order(self, order_id: str, symbol: str | None = None) -> Order:
        o = self._client.get_order_by_id(order_id=order_id)
        return self._order_from_obj(o)

    def close_all(self, symbol: str | None = None) -> list[Order]:
        orders = []
        if symbol:
            try:
                o = self._client.close_position(symbol)
                orders.append(self._order_from_obj(o))
            except Exception as e:
                logger.warning("close_position %s failed: %s", symbol, e)
            return orders
        try:
            resp = self._client.close_all_positions(cancel_orders=True)
            for o in resp:
                orders.append(self._order_from_obj(o))
        except Exception as e:
            logger.warning("close_all failed: %s", e)
        return orders
