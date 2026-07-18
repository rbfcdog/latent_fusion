from __future__ import annotations

import logging
import uuid

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
)

logger = logging.getLogger(__name__)


class LivePaperBroker:
    """True live paper trading: real-time market data from a broker sandbox,
    simulated execution at live prices. No order API keys required.

    Wraps a market-data broker (BinanceBroker / AlpacaBroker) for get_bars /
    get_latest_bar / get_market_status, and maintains internal cash, position,
    orders and fills — filling market orders at the current live bar's close
    plus configurable slippage and fees.
    """

    name = "livepaper"
    is_live = False

    def __init__(
        self,
        market_data_broker: BrokerClient,
        symbol: str,
        initial_cash: float = 10_000.0,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        currency: str | None = None,
    ):
        self.md = market_data_broker
        self.symbol = symbol
        self.initial_cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000.0
        self.slip_rate = slippage_bps / 10_000.0
        self._currency = currency or ("USDT" if getattr(market_data_broker, "name", "") == "binance" else "USD")
        self._cash = float(initial_cash)
        self._position_qty = 0.0
        self._avg_entry = 0.0
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._current_bar: Bar | None = None
        self._warmup: list[Bar] = []

    def get_bars(self, symbol: str, limit: int, timeframe: str = "1m"):
        return self.md.get_bars(symbol, limit, timeframe)

    def get_latest_bar(self, symbol: str) -> Bar:
        bar = self.md.get_latest_bar(symbol)
        self._current_bar = bar
        self._warmup.append(bar)
        if len(self._warmup) > 5000:
            self._warmup = self._warmup[-5000:]
        return bar

    def get_market_status(self) -> dict:
        return self.md.get_market_status()

    def _price(self) -> float:
        return float(self._current_bar.close) if self._current_bar else 0.0

    def get_account(self) -> Account:
        price = self._price()
        eq = self._cash + self._position_qty * price
        return Account(
            cash=self._cash, equity=eq, buying_power=max(self._cash, 0.0),
            currency=self._currency, initial_cash=self.initial_cash,
        )

    def get_position(self, symbol: str) -> Position | None:
        if abs(self._position_qty) < 1e-12:
            return None
        price = self._price() or self._avg_entry
        mv = self._position_qty * price
        return Position(
            symbol=self.symbol, qty=self._position_qty,
            avg_entry_price=self._avg_entry, market_value=mv,
            current_price=price, unrealized_pl=mv - self._position_qty * self._avg_entry,
        )

    def _apply_fill(self, side: OrderSide, qty: float, exec_price: float, fee: float, ts) -> Order:
        signed = qty if side == OrderSide.BUY else -qty
        self._cash -= signed * exec_price + fee
        new_qty = self._position_qty + signed
        if abs(new_qty) < 1e-12:
            new_qty = 0.0
        if self._position_qty != 0.0 and ((self._position_qty > 0) == (new_qty > 0)):
            self._avg_entry = (self._avg_entry * self._position_qty + signed * exec_price) / new_qty if new_qty != 0 else 0.0
        else:
            self._avg_entry = exec_price if new_qty != 0 else 0.0
        self._position_qty = new_qty
        oid = uuid.uuid4().hex[:16]
        order = Order(
            id=oid, symbol=self.symbol, side=side, qty=qty,
            order_type=OrderType.MARKET, time_in_force=TimeInForce.GTC,
            status=OrderStatus.FILLED, filled_qty=qty,
            filled_avg_price=exec_price, fee=fee, created_at=ts,
        )
        self._orders[oid] = order
        self._fills.append(Fill(
            order_id=oid, symbol=self.symbol, side=side, qty=qty,
            price=exec_price, fee=fee, timestamp=ts,
        ))
        return order

    def submit_market_order(self, symbol: str, side: OrderSide, qty: float) -> Order:
        price = self._price()
        if price <= 0:
            raise RuntimeError("no live price available; call get_latest_bar first")
        slip = self.slip_rate if side == OrderSide.BUY else -self.slip_rate
        exec_price = price * (1 + slip)
        notional = qty * exec_price
        fee = notional * self.fee_rate
        ts = self._current_bar.timestamp if self._current_bar else None
        return self._apply_fill(side, qty, exec_price, fee, ts)

    def submit_limit_order(self, symbol: str, side: OrderSide, qty: float, limit_price: float) -> Order:
        price = self._price()
        if price <= 0:
            raise RuntimeError("no live price available")
        hit = (side == OrderSide.BUY and price <= limit_price) or (side == OrderSide.SELL and price >= limit_price)
        ts = self._current_bar.timestamp if self._current_bar else None
        if hit:
            notional = qty * limit_price
            fee = notional * self.fee_rate
            return self._apply_fill(side, qty, limit_price, fee, ts)
        oid = uuid.uuid4().hex[:16]
        order = Order(
            id=oid, symbol=self.symbol, side=side, qty=qty,
            order_type=OrderType.LIMIT, limit_price=limit_price,
            time_in_force=TimeInForce.GTC, status=OrderStatus.OPEN, created_at=ts,
        )
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
        return [self.submit_market_order(self.symbol, side, abs(self._position_qty))]

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def orders(self) -> list[Order]:
        return list(self._orders.values())
