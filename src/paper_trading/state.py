from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .base import Fill, Order, OrderSide, OrderStatus, OrderType, TimeInForce


_SCHEMA = """
CREATE TABLE IF NOT EXISTS engine_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    order_type TEXT NOT NULL,
    limit_price REAL,
    time_in_force TEXT NOT NULL,
    status TEXT NOT NULL,
    filled_qty REAL NOT NULL,
    filled_avg_price REAL,
    fee REAL NOT NULL,
    created_at TEXT,
    client_order_id TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    fee REAL NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    close REAL,
    signal REAL,
    target_weight REAL,
    position REAL,
    cash REAL,
    equity REAL,
    bh_equity REAL,
    order_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_snapshots_run ON snapshots(run_id);
"""


def _connect(path: str) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _load(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw


class StateStore:
    def __init__(self, path: str):
        self.path = path
        self._conn = _connect(path)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM engine_state WHERE key=?", (key,)).fetchone()
        return _load(row["value"]) if row else default

    def set_state(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO engine_state(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, _dump(value)),
        )

    def save_order(self, order: Order) -> None:
        self._conn.execute(
            "INSERT INTO orders(id, symbol, side, qty, order_type, limit_price, time_in_force, "
            "status, filled_qty, filled_avg_price, fee, created_at, client_order_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, filled_qty=excluded.filled_qty, "
            "filled_avg_price=excluded.filled_avg_price, fee=excluded.fee",
            (
                order.id, order.symbol, order.side.value, float(order.qty),
                order.order_type.value, order.limit_price, order.time_in_force.value,
                order.status.value, float(order.filled_qty), order.filled_avg_price,
                float(order.fee),
                order.created_at.isoformat() if order.created_at is not None else None,
                order.client_order_id,
            ),
        )

    def save_fill(self, fill: Fill) -> None:
        self._conn.execute(
            "INSERT INTO fills(order_id, symbol, side, qty, price, fee, timestamp) VALUES (?,?,?,?,?,?,?)",
            (
                fill.order_id, fill.symbol, fill.side.value, float(fill.qty),
                float(fill.price), float(fill.fee),
                fill.timestamp.isoformat(),
            ),
        )

    def save_snapshot(self, run_id: str, snap) -> None:
        self._conn.execute(
            "INSERT INTO snapshots(run_id, timestamp, close, signal, target_weight, position, "
            "cash, equity, bh_equity, order_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                snap.timestamp.isoformat(),
                float(snap.close) if pd.notna(snap.close) else None,
                float(snap.signal) if pd.notna(snap.signal) else None,
                float(snap.target_weight) if pd.notna(snap.target_weight) else None,
                float(snap.position) if pd.notna(snap.position) else None,
                float(snap.cash) if pd.notna(snap.cash) else None,
                float(snap.equity) if pd.notna(snap.equity) else None,
                float(snap.bh_equity) if pd.notna(snap.bh_equity) else None,
                snap.order_id,
            ),
        )

    def load_orders(self) -> pd.DataFrame:
        rows = [dict(r) for r in self._conn.execute("SELECT * FROM orders").fetchall()]
        return pd.DataFrame(rows)

    def load_fills(self) -> pd.DataFrame:
        rows = [dict(r) for r in self._conn.execute("SELECT * FROM fills").fetchall()]
        return pd.DataFrame(rows)

    def load_snapshots(self, run_id: str | None = None) -> pd.DataFrame:
        if run_id:
            rows = [dict(r) for r in self._conn.execute(
                "SELECT * FROM snapshots WHERE run_id=? ORDER BY rowid", (run_id,)).fetchall()]
        else:
            rows = [dict(r) for r in self._conn.execute(
                "SELECT * FROM snapshots ORDER BY rowid").fetchall()]
        return pd.DataFrame(rows)

    def clear_run(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM snapshots WHERE run_id=?", (run_id,))

    def clear_all(self) -> None:
        for t in ("engine_state", "orders", "fills", "snapshots"):
            self._conn.execute(f"DELETE FROM {t}")
