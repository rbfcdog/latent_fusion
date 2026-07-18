from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import uuid
from typing import Any
from urllib.parse import urlencode


import pandas as pd
import requests

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

_PAPER_URL = "https://testnet.binance.vision"
_LIVE_URL = "https://api.binance.com"
_DATA_URL = "https://data-api.binance.vision"

_TF_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1d", "1w": "1w",
}


class BinanceBroker:
    name = "binance"
    is_live: bool

    def __init__(self, api_key: str | None = None, api_secret: str | None = None, paper: bool = True, initial_cash: float = 10_000.0):
        key = api_key or os.environ.get("BINANCE_API_KEY")
        secret = api_secret or os.environ.get("BINANCE_API_SECRET")
        self.is_live = not paper
        if paper and not key:
            logger.warning("Binance testnet market-data works without keys; signed endpoints need keys")
        self._key = key
        self._secret = secret or ""
        self._base = _PAPER_URL if paper else _LIVE_URL
        self._session = requests.Session()
        if self._key:
            self._session.headers.update({"X-MBX-APIKEY": self._key})
        self._recv_window = 5000
        self._initial_cash = float(initial_cash)
        logger.info("BinanceBroker ready (paper=%s base=%s)", paper, self._base)

    def _sign(self, params: dict) -> dict:
        if not self._secret:
            return params
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self._recv_window
        query = urlencode(params)
        sig = hmac.new(self._secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _get(self, path: str, params: dict | None = None, signed: bool = False, base: str | None = None) -> Any:
        params = dict(params or {})
        if signed:
            params = self._sign(params)
        url = (base or self._base) + path
        r = self._session.get(url, params=params, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"binance GET {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def _post(self, path: str, params: dict | None = None, signed: bool = True) -> Any:
        params = self._sign(params or {})
        url = self._base + path
        r = self._session.post(url, params=params, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"binance POST {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def _delete(self, path: str, params: dict | None = None, signed: bool = True) -> Any:
        params = self._sign(params or {})
        url = self._base + path
        r = self._session.delete(url, params=params, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"binance DELETE {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def get_account(self) -> Account:
        if not self._key:
            return Account(cash=self._initial_cash, equity=self._initial_cash,
                           buying_power=self._initial_cash, currency="USDT",
                           initial_cash=self._initial_cash)
        data = self._get("/api/v3/account", signed=True)
        balances = {b["asset"]: float(b["free"]) + float(b["locked"]) for b in data.get("balances", [])}
        usdt = balances.get("USDT", 0.0)
        return Account(
            cash=usdt, equity=usdt, buying_power=usdt,
            currency="USDT", initial_cash=usdt,
        )
    def get_position(self, symbol: str) -> Position | None:
        if not self._key:
            return None
        base = symbol.replace("USDT", "").replace("BTC", "BTC").upper()
        data = self._get("/api/v3/account", signed=True)
        for b in data.get("balances", []):
            if b["asset"] == base and (float(b["free"]) + float(b["locked"])) > 1e-12:
                qty = float(b["free"]) + float(b["locked"])
                price = self._last_price(symbol)
                return Position(
                    symbol=symbol, qty=qty, avg_entry_price=price,
                    market_value=qty * price, current_price=price, unrealized_pl=0.0,
                )
        return None

    def _last_price(self, symbol: str) -> float:
        d = self._get("/api/v3/ticker/price", {"symbol": symbol}, base=_DATA_URL)
        return float(d["price"])

    def _klines_to_bars(self, rows: list) -> pd.DataFrame:
        out = []
        for k in rows:
            out.append({
                "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5]),
                "vwap": float(k[7]) / float(k[5]) if float(k[5]) > 0 else None,
            })
        return normalize_bars(pd.DataFrame(out))

    def get_latest_bar(self, symbol: str) -> Bar:
        df = self.get_bars(symbol, limit=1)
        if df.empty:
            raise RuntimeError(f"no klines for {symbol}")
        r = df.iloc[-1]
        return Bar(
            timestamp=r["timestamp"], open=float(r["open"]), high=float(r["high"]),
            low=float(r["low"]), close=float(r["close"]),
            volume=float(r["volume"]),
            vwap=float(r["vwap"]) if pd.notna(r["vwap"]) else None,
        )

    def get_bars(self, symbol: str, limit: int, timeframe: str = "1m") -> pd.DataFrame:
        if timeframe not in _TF_MAP:
            raise ValueError(f"unsupported timeframe {timeframe}; choose from {list(_TF_MAP)}")
        rows = self._get("/api/v3/klines", {"symbol": symbol, "interval": _TF_MAP[timeframe], "limit": limit}, base=_DATA_URL)
        return self._klines_to_bars(rows)

    def get_market_status(self) -> dict:
        try:
            d = self._get("/api/v3/ping", base=_DATA_URL)
            return {"open": True, "status": d}
        except Exception as e:
            return {"open": False, "error": str(e)}

    def _order_from_obj(self, o: dict) -> Order:
        status_map = {
            "NEW": OrderStatus.OPEN, "ACCEPTED": OrderStatus.OPEN, "PENDING_NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIAL, "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED, "CANCELLED": OrderStatus.CANCELED,
            "REJECTED": OrderStatus.REJECTED, "EXPIRED": OrderStatus.REJECTED,
        }
        return Order(
            id=str(o.get("orderId", uuid.uuid4().hex)),
            symbol=str(o.get("symbol", "")),
            side=OrderSide.BUY if str(o.get("side", "BUY")).upper() == "BUY" else OrderSide.SELL,
            qty=float(o.get("origQty", 0) or 0),
            order_type=OrderType.MARKET if str(o.get("type", "MARKET")).upper() == "MARKET" else OrderType.LIMIT,
            limit_price=float(o.get("price", 0) or 0) or None,
            time_in_force=TimeInForce.GTC,
            status=status_map.get(str(o.get("status", "")).upper(), OrderStatus.PENDING),
            filled_qty=float(o.get("executedQty", 0) or 0),
            filled_avg_price=float(o.get("cummulativeQuoteQty", 0) or 0) / float(o.get("executedQty", 1) or 1)
            if float(o.get("executedQty", 0) or 0) > 0 else None,
            fee=float(o.get("commission", 0) or 0),
            created_at=pd.Timestamp(o.get("transactTime", time.time() * 1000), unit="ms", tz="UTC")
            if o.get("transactTime") else pd.Timestamp.now(tz="UTC"),
            client_order_id=o.get("clientOrderId"),
        )

    def submit_market_order(self, symbol: str, side: OrderSide, qty: float) -> Order:
        if not self._key:
            raise RuntimeError("Binance signed endpoints require API key/secret; set BINANCE_API_KEY/SECRET")
        o = self._post("/api/v3/order", {
            "symbol": symbol, "side": "BUY" if side == OrderSide.BUY else "SELL",
            "type": "MARKET", "quantity": self._fmt_qty(qty),
        })
        return self._order_from_obj(o)

    def submit_limit_order(self, symbol: str, side: OrderSide, qty: float, limit_price: float) -> Order:
        if not self._key:
            raise RuntimeError("Binance signed endpoints require API key/secret")
        o = self._post("/api/v3/order", {
            "symbol": symbol, "side": "BUY" if side == OrderSide.BUY else "SELL",
            "type": "LIMIT", "timeInForce": "GTC",
            "quantity": self._fmt_qty(qty), "price": self._fmt_price(limit_price),
        })
        return self._order_from_obj(o)

    def _fmt_qty(self, q: float) -> str:
        return f"{q:.6f}".rstrip("0").rstrip(".")

    def _fmt_price(self, p: float) -> str:
        return f"{p:.8f}".rstrip("0").rstrip(".")

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        if not symbol:
            raise ValueError("Binance cancel needs symbol")
        try:
            self._delete("/api/v3/order", {"symbol": symbol, "orderId": order_id})
            return True
        except Exception as e:
            logger.warning("cancel_order failed: %s", e)
            return False

    def get_order(self, order_id: str, symbol: str | None = None) -> Order:
        if not symbol:
            raise ValueError("Binance get_order needs symbol")
        o = self._get("/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)
        return self._order_from_obj(o)

    def close_all(self, symbol: str | None = None) -> list[Order]:
        if not symbol:
            raise ValueError("Binance close_all needs symbol")
        pos = self.get_position(symbol)
        if not pos or abs(pos.qty) < 1e-12:
            return []
        side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
        return [self.submit_market_order(symbol, side, abs(pos.qty))]
