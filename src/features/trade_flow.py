from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeFlowAccumulator:
    symbol: str = "BTCUSDT"
    ws_base: str = "wss://stream.binance.com:9443/ws"
    buffer_minutes: int = 5
    _buy_vol: float = field(default=0.0, init=False)
    _sell_vol: float = field(default=0.0, init=False)
    _current_minute: int | None = field(default=None, init=False)
    _buffer: dict[int, dict] = field(default_factory=dict, init=False)
    _last_flush: float = field(default=0.0, init=False)

    def process_message(self, msg: dict) -> dict | None:
        ts = msg.get("T", 0) // 60000
        qty = float(msg.get("q", 0))
        is_sell = msg.get("m", False)

        if self._current_minute is None:
            self._current_minute = ts

        if ts != self._current_minute:
            ts_ms = self._current_minute * 60000
            self._buffer[ts_ms] = {
                "buy": self._buy_vol,
                "sell": self._sell_vol,
                "delta": self._buy_vol - self._sell_vol,
            }
            self._buy_vol = 0.0
            self._sell_vol = 0.0
            self._current_minute = ts
            return self._buffer[ts_ms]

        if is_sell:
            self._sell_vol += qty
        else:
            self._buy_vol += qty
        return None

    def flush_buffer(self) -> pd.DataFrame:
        if not self._buffer:
            return pd.DataFrame(columns=["timestamp", "buy_volume", "sell_volume", "delta"])
        df = pd.DataFrame([
            {"timestamp": ts, "buy_volume": v["buy"], "sell_volume": v["sell"], "delta": v["delta"]}
            for ts, v in sorted(self._buffer.items())
        ])
        self._buffer.clear()
        return df

    async def stream(self, on_flush: callable | None = None, duration: int | None = None) -> None:
        import websockets

        url = f"{self.ws_base}/{self.symbol.lower()}@aggTrade"
        reconnect_delay = 5.0
        max_reconnect = 300.0
        loop = asyncio.get_event_loop()
        self._last_flush = loop.time()
        deadline = loop.time() + duration if duration else None

        while True:
            if deadline and loop.time() >= deadline:
                break
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10, close_timeout=10) as ws:
                    logger.info(f"Trade flow connected: {self.symbol}")
                    reconnect_delay = 5.0
                    while True:
                        if deadline and loop.time() >= deadline:
                            break
                        msg = json.loads(await ws.recv())
                        self.process_message(msg)
                        now = loop.time()
                        if now - self._last_flush >= self.buffer_minutes * 60:
                            df = self.flush_buffer()
                            if on_flush is not None and not df.empty:
                                on_flush(df)
                            self._last_flush = now
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning(f"Trade flow error: {exc}, reconnect in {reconnect_delay:.0f}s")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect)


def compute_order_flow_imbalance(buy_qty: float, sell_qty: float) -> float:
    total = buy_qty + sell_qty
    if total <= 0:
        return 0.0
    return (buy_qty - sell_qty) / total


def vpin(trades: pd.DataFrame, bucket_size: float = 1.0) -> np.ndarray:
    if trades.empty:
        return np.array([])

    t = trades.sort_values("trade_time").reset_index(drop=True)
    quantities = t["quantity"].to_numpy(dtype=float)
    is_sell = t["is_buyer_maker"].to_numpy(dtype=bool)
    cumulative = np.cumsum(quantities)
    n_buckets = int(cumulative[-1] / bucket_size) if cumulative[-1] > 0 else 1
    bucket_boundaries = np.linspace(0, cumulative[-1], n_buckets + 1)

    vpin_values = []
    for i in range(n_buckets):
        start = bucket_boundaries[i]
        end = bucket_boundaries[i + 1]
        mask = (cumulative > start) & (cumulative <= end)
        if mask.sum() == 0:
            vpin_values.append(0.0)
            continue
        bucket_mask = mask & ~is_sell
        buy = quantities[bucket_mask].sum()
        sell_mask = mask & is_sell
        sell = quantities[sell_mask].sum()
        total = buy + sell
        vpin_values.append(abs(buy - sell) / total if total > 0 else 0.0)

    return np.array(vpin_values)
