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
    OrderType,
    Position,
    TimeInForce,
    bars_to_frame,
    normalize_bars,
)
from .engine import PaperTradingEngine, run_engine
from .metrics import bh_curve_from_prices, compute_metrics
from .state import StateStore
from .simulated import SimulatedBroker
from .live_paper import LivePaperBroker

__all__ = [
    "Account", "Bar", "BrokerClient", "EngineConfig", "EngineResult", "EngineSnapshot",
    "Fill", "Order", "OrderSide", "OrderStatus", "OrderType", "Position", "TimeInForce",
    "bars_to_frame", "normalize_bars",
    "PaperTradingEngine", "run_engine",
    "bh_curve_from_prices", "compute_metrics",
    "StateStore",
    "SimulatedBroker", "LivePaperBroker",
    "AlpacaBroker", "BinanceBroker",
]


def __getattr__(name: str):
    if name == "AlpacaBroker":
        from .alpaca import AlpacaBroker
        return AlpacaBroker
    if name == "BinanceBroker":
        from .binance import BinanceBroker
        return BinanceBroker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
