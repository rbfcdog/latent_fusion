from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

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


class SimulatedBroker:
    name = "simulated"
    is_live = False

    def __init__(
        self,
        symbol: str,
        bars: pd.DataFrame,
        initial_cash: float = 10_000.0,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ):
        self.symbol = symbol
        self.bars = normalize_bars(bars).reset_index(drop=True)
        self.initial_cash = initial_cash
        self.fee_rate = fee_bps / 10_000.0
        self.slip_rate = slippage_bps / 10_000.0
        self._cursor = 0
        self._cash = float(initial_cash)
        self._position_qty = 0.0
        self._avg_entry = 0.0
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._current_bar: Bar | None = None

    def _bar_at(self, i: int) -> Bar | None:
        if i < 0 or i >= len(self.bars):
            return None
        r = self.bars.iloc[i]
        return Bar(
            timestamp=r["timestamp"],
            open=float(r["open"]), high=float(r["high"]),
            low=float(r["low"]), close=float(r["close"]),
            volume=float(r.get("volume", 0.0)),
            vwap=float(r["vwap"]) if pd.notna(r.get("vwap")) else None,
        )

    def advance(self) -> Bar | None:
        if self._cursor >= len(self.bars):
            return None
        bar = self._bar_at(self._cursor)
        self._current_bar = bar
        self._cursor += 1
        return bar

    def reset(self) -> None:
        self._cursor = 0
        self._cash = float(self.initial_cash)
        self._position_qty = 0.0
        self._avg_entry = 0.0
        self._orders.clear()
        self._fills.clear()
        self._current_bar = None

    def _equity(self, price: float) -> float:
        return self._cash + self._position_qty * price

    def get_account(self) -> Account:
        price = self._current_bar.close if self._current_bar else 0.0
        eq = self._equity(price)
        return Account(
            cash=self._cash, equity=eq, buying_power=self._cash,
            currency="USD", initial_cash=self.initial_cash,
        )

    def get_position(self, symbol: str) -> Position | None:
        if symbol != self.symbol or self._position_qty == 0.0:
            return None
        price = self._current_bar.close if self._current_bar else self._avg_entry
        mv = self._position_qty * price
        return Position(
            symbol=self.symbol, qty=self._position_qty,
            avg_entry_price=self._avg_entry, market_value=mv,
            current_price=price, unrealized_pl=mv - self._position_qty * self._avg_entry,
        )

    def get_latest_bar(self, symbol: str) -> Bar:
        if symbol != self.symbol:
            raise ValueError(f"unknown symbol {symbol}")
        bar = self.advance()
        if bar is None:
            raise RuntimeError("no more bars to replay")
        return bar

    def get_bars(self, symbol: str, limit: int, timeframe: str = "1m") -> pd.DataFrame:
        if symbol != self.symbol:
            raise ValueError(f"unknown symbol {symbol}")
        if self._cursor == 0:
            self._cursor = min(limit, len(self.bars))
            self._current_bar = self._bar_at(self._cursor - 1)
        end = self._cursor
        start = max(0, end - limit)
        return self.bars.iloc[start:end].copy()

    def get_market_status(self) -> dict:
        return {"open": self._cursor < len(self.bars), "symbol": self.symbol}

    def _fill_market(self, order: Order) -> Order:
        price = self._current_bar.close if self._current_bar else 0.0
        if price <= 0:
            order.status = OrderStatus.REJECTED
            return order
        slip = self.slip_rate if order.side == OrderSide.BUY else -self.slip_rate
        exec_price = price * (1 + slip)
        signed = order.qty if order.side == OrderSide.BUY else -order.qty
        notional = order.qty * exec_price
        fee = notional * self.fee_rate
        self._cash -= signed * exec_price + fee
        new_qty = self._position_qty + signed
        if abs(new_qty) < 1e-12:
            new_qty = 0.0
        if (self._position_qty > 0) == (new_qty > 0) and self._position_qty != 0.0:
            self._avg_entry = (self._avg_entry * self._position_qty + signed * exec_price) / new_qty if new_qty != 0 else 0.0
        else:
            self._avg_entry = exec_price if new_qty != 0 else 0.0
        self._position_qty = new_qty
        order.filled_qty = order.qty
        order.filled_avg_price = exec_price
        order.fee = fee
        order.status = OrderStatus.FILLED
        self._fills.append(Fill(
            order_id=order.id, symbol=order.symbol, side=order.side,
            qty=order.qty, price=exec_price, fee=fee,
            timestamp=self._current_bar.timestamp,
        ))
        return order

    def submit_market_order(self, symbol: str, side: OrderSide, qty: float) -> Order:
        if symbol != self.symbol:
            raise ValueError(f"unknown symbol {symbol}")
        oid = uuid.uuid4().hex[:16]
        order = Order(
            id=oid, symbol=symbol, side=side, qty=qty,
            order_type=OrderType.MARKET, time_in_force=TimeInForce.GTC,
            status=OrderStatus.PENDING,
            created_at=self._current_bar.timestamp if self._current_bar else pd.Timestamp.now(tz="UTC"),
        )
        order = self._fill_market(order)
        self._orders[oid] = order
        return order

    def submit_limit_order(self, symbol: str, side: OrderSide, qty: float, limit_price: float) -> Order:
        if symbol != self.symbol:
            raise ValueError(f"unknown symbol {symbol}")
        oid = uuid.uuid4().hex[:16]
        order = Order(
            id=oid, symbol=symbol, side=side, qty=qty,
            order_type=OrderType.LIMIT, limit_price=limit_price,
            time_in_force=TimeInForce.GTC, status=OrderStatus.OPEN,
            created_at=self._current_bar.timestamp if self._current_bar else pd.Timestamp.now(tz="UTC"),
        )
        price = self._current_bar.close if self._current_bar else 0.0
        if price > 0 and (
            (side == OrderSide.BUY and price <= limit_price)
            or (side == OrderSide.SELL and price >= limit_price)
        ):
            order = self._fill_market(order)
        self._orders[oid] = order
        return order

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        o = self._orders.get(order_id)
        if o and o.status in (OrderStatus.PENDING, OrderStatus.OPEN):
            o.status = OrderStatus.CANCELED
            return True
        return False

    def get_order(self, order_id: str, symbol: str | None = None) -> Order:
        return self._orders[order_id]

    def close_all(self, symbol: str | None = None) -> list[Order]:
        if abs(self._position_qty) < 1e-12:
            return []
        side = OrderSide.SELL if self._position_qty > 0 else OrderSide.BUY
        o = self.submit_market_order(self.symbol, side, abs(self._position_qty))
        return [o]

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)
