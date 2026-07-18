from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..backtest.engine import Strategy
from .base import (
    Account,
    Bar,
    BrokerClient,
    EngineConfig,
    EngineResult,
    EngineSnapshot,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    bars_to_frame,
    normalize_bars,
)
from .metrics import bh_curve_from_prices, compute_metrics
from .state import StateStore

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    def __init__(
        self,
        broker: BrokerClient,
        strategy: Strategy,
        config: EngineConfig | None = None,
        state: StateStore | None = None,
    ):
        self.broker = broker
        self.strategy = strategy
        self.config = config or EngineConfig()
        self.state = state
        self.run_id = uuid.uuid4().hex[:12]
        self._warmup: list[Bar] = []
        self._bh_base: float | None = None
        self._step = 0

    def _now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def _warmup_bars(self) -> pd.DataFrame:
        try:
            df = self.broker.get_bars(self.config.symbol, self.config.bar_window, self.config.timeframe)
        except Exception as e:
            logger.warning("warmup get_bars failed: %s", e)
            df = pd.DataFrame()
        df = normalize_bars(df)
        if df.empty:
            return df
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df.sort_values("timestamp").reset_index(drop=True)

    def _account_equity(self, account: Account, price: float) -> float:
        return float(account.equity)

    def _current_position_qty(self) -> float:
        try:
            pos = self.broker.get_position(self.config.symbol)
        except Exception:
            pos = None
        return float(pos.qty) if pos else 0.0

    def _target_weight_to_qty(self, target_weight: float, equity: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return (equity * target_weight) / price

    def _signal_to_target_weight(self, signal: float) -> float:
        w = float(np.clip(signal, -1.0, 1.0))
        if self.config.long_only:
            w = max(0.0, w)
        if abs(w) > self.config.max_weight:
            w = np.sign(w) * self.config.max_weight
        return float(w)

    def _rebalance(
        self,
        target_weight: float,
        account: Account,
        price: float,
    ) -> Order | None:
        current_qty = self._current_position_qty()
        equity = self._account_equity(account, price)
        target_qty = self._target_weight_to_qty(target_weight, equity, price)
        delta_qty = target_qty - current_qty

        if abs(delta_qty) < self.config.min_qty:
            return None
        notional = abs(delta_qty) * price
        if notional < self.config.min_notional:
            return None

        side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
        try:
            order = self.broker.submit_market_order(self.config.symbol, side, abs(delta_qty))
        except Exception as e:
            logger.error("submit_market_order failed: %s", e)
            return None

        if self.state is not None:
            self.state.save_order(order)
            if order.status in (OrderStatus.FILLED, OrderStatus.PARTIAL) and order.filled_qty > 0:
                fill = Fill(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.filled_qty,
                    price=order.filled_avg_price or price,
                    fee=order.fee,
                    timestamp=order.created_at or self._now(),
                )
                self.state.save_fill(fill)
        return order

    def _sl_tp_exit(self, account: Account, price: float) -> Order | None:
        if self.config.stop_loss_pct is None and self.config.take_profit_pct is None:
            return None
        try:
            pos = self.broker.get_position(self.config.symbol)
        except Exception:
            pos = None
        if not pos or pos.qty <= 0:
            return None
        entry = pos.avg_entry_price
        if entry <= 0:
            return None
        sl_hit = self.config.stop_loss_pct is not None and price <= entry * (1 - self.config.stop_loss_pct)
        tp_hit = self.config.take_profit_pct is not None and price >= entry * (1 + self.config.take_profit_pct)
        if sl_hit or tp_hit:
            logger.info("SL/TP exit triggered (sl=%s tp=%s price=%.4f entry=%.4f)",
                        sl_hit, tp_hit, price, entry)
            try:
                orders = self.broker.close_all(self.config.symbol)
            except Exception as e:
                logger.error("close_all failed: %s", e)
                return None
            for o in orders:
                if self.state is not None:
                    self.state.save_order(o)
                    if o.filled_qty > 0:
                        self.state.save_fill(Fill(
                            order_id=o.id, symbol=o.symbol, side=o.side, qty=o.filled_qty,
                            price=o.filled_avg_price or price, fee=o.fee,
                            timestamp=o.created_at or self._now(),
                        ))
            return orders[0] if orders else None
        return None

    def _step_once(self) -> EngineSnapshot | None:
        try:
            bar = self.broker.get_latest_bar(self.config.symbol)
        except Exception as e:
            logger.warning("get_latest_bar failed: %s", e)
            return None
        if bar is None or bar.close <= 0:
            return None

        self._warmup.append(bar)
        if len(self._warmup) > self.config.bar_window:
            self._warmup = self._warmup[-self.config.bar_window:]

        df = bars_to_frame(list(self._warmup))
        if df.empty or len(df) < 2:
            return None

        try:
            account = self.broker.get_account()
        except Exception as e:
            logger.warning("get_account failed: %s", e)
            return None

        sl_tp_order = self._sl_tp_exit(account, bar.close)
        if sl_tp_order is not None:
            try:
                account = self.broker.get_account()
            except Exception:
                pass

        try:
            signal = float(self.strategy.generate_signals(df).iloc[-1])
        except Exception as e:
            logger.error("strategy.generate_signals failed: %s", e)
            signal = 0.0
        if not np.isfinite(signal):
            signal = 0.0

        target_weight = self._signal_to_target_weight(signal)
        order = self._rebalance(target_weight, account, bar.close)

        try:
            account = self.broker.get_account()
            pos = self.broker.get_position(self.config.symbol)
        except Exception:
            pos = None
        position = float(pos.qty) if pos else 0.0
        cash = float(account.cash)
        equity = float(account.equity)

        if self._bh_base is None:
            self._bh_base = bar.close
        bh_equity = (bar.close / self._bh_base) * self.config.initial_cash if self._bh_base else equity

        snap = EngineSnapshot(
            timestamp=bar.timestamp,
            close=bar.close,
            signal=signal,
            target_weight=target_weight,
            position=position,
            cash=cash,
            equity=equity,
            bh_equity=bh_equity,
            order_id=order.id if order else None,
        )
        if self.state is not None:
            self.state.save_snapshot(self.run_id, snap)
        self._step += 1
        return snap

    def step_once(self) -> EngineSnapshot | None:
        return self._step_once()

    def run(self, on_step: "callable | None" = None) -> EngineResult:
        warm = self._warmup_bars()
        if not warm.empty:
            self._warmup = [
                Bar(
                    timestamp=r["timestamp"],
                    open=float(r["open"]), high=float(r["high"]),
                    low=float(r["low"]), close=float(r["close"]),
                    volume=float(r.get("volume", 0.0)),
                    vwap=float(r["vwap"]) if "vwap" in r and pd.notna(r["vwap"]) else None,
                )
                for _, r in warm.tail(self.config.bar_window).iterrows()
            ]

        snaps: list[EngineSnapshot] = []
        max_steps = self.config.max_steps
        while True:
            if max_steps is not None and self._step >= max_steps:
                break
            try:
                snap = self._step_once()
            except KeyboardInterrupt:
                logger.info("interrupted by user")
                break
            if snap is None:
                time.sleep(max(self.config.poll_seconds, 0.1))
                continue
            snaps.append(snap)
            if on_step is not None:
                try:
                    on_step(snap)
                except Exception as e:
                    logger.warning("on_step callback failed: %s", e)
            if self.config.poll_seconds > 0:
                time.sleep(self.config.poll_seconds)

        return self._finalize(snaps)

    def _finalize(self, snaps: list[EngineSnapshot]) -> EngineResult:
        if snaps:
            snap_df = pd.DataFrame([asdict(s) for s in snaps])
            snap_df["timestamp"] = pd.to_datetime(snap_df["timestamp"])
            equity = snap_df.set_index("timestamp")["equity"]
            bh = snap_df.set_index("timestamp")["bh_equity"]
        else:
            snap_df = pd.DataFrame()
            equity = pd.Series(dtype=float)
            bh = pd.Series(dtype=float)

        if self.state is not None:
            orders_df = self.state.load_orders()
            fills_df = self.state.load_fills()
        else:
            orders_df = pd.DataFrame()
            fills_df = pd.DataFrame()

        trades = self._build_trades(fills_df, snap_df)
        metrics = compute_metrics(equity, trades, self.config.periods_per_year, bh)
        final_eq = float(equity.iloc[-1]) if not equity.empty else self.config.initial_cash
        bh_final = float(bh.iloc[-1]) if not bh.empty else self.config.initial_cash
        return EngineResult(
            snapshots=snap_df,
            trades=trades,
            orders=orders_df,
            metrics=metrics,
            final_equity=final_eq,
            bh_final_equity=bh_final,
        )

    def _build_trades(self, fills_df: pd.DataFrame, snap_df: pd.DataFrame) -> pd.DataFrame:
        if fills_df.empty:
            return pd.DataFrame()
        fills = fills_df.copy()
        for c in ("qty", "price", "fee"):
            if c in fills.columns:
                fills[c] = pd.to_numeric(fills[c], errors="coerce")
        if "timestamp" in fills.columns:
            fills["timestamp"] = pd.to_datetime(fills["timestamp"], errors="coerce")
        fills["notional"] = fills["qty"] * fills["price"]
        fills["signed_qty"] = np.where(fills["side"] == "buy", fills["qty"], -fills["qty"])
        return fills.reset_index(drop=True)


def run_engine(
    broker: BrokerClient,
    strategy: Strategy,
    config: EngineConfig | None = None,
    on_step: "callable | None" = None,
) -> EngineResult:
    state = StateStore(config.state_path) if config else None
    try:
        engine = PaperTradingEngine(broker, strategy, config, state)
        return engine.run(on_step=on_step)
    finally:
        if state is not None:
            state.close()
