from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields

from .base import EngineConfig
from .engine import PaperTradingEngine
from .state import StateStore


def _strategies():
    from src.strategy import (
        HMMRegimeStrategy,
        InstitutionalV3Strategy,
        IntensityGatedStrategy,
        MeanReversionStrategy,
        RegimeRouterStrategy,
        S1Hard70Strategy,
        SmaCrossStrategy,
        VwapReversionStrategy,
    )
    return {
        "sma": SmaCrossStrategy,
        "meanrev": MeanReversionStrategy,
        "hmm": HMMRegimeStrategy,
        "vwap": VwapReversionStrategy,
        "instv3": InstitutionalV3Strategy,
        "router": RegimeRouterStrategy,
        "intensity": IntensityGatedStrategy,
        "s1hard70": S1Hard70Strategy,
    }


def _brokers():
    return {"alpaca", "binance", "simulated", "livepaper"}


def _build_broker(name: str, args, initial_cash: float):
    if name == "simulated":
        import pandas as pd
        from .simulated import SimulatedBroker
        from .binance import BinanceBroker
        bb = BinanceBroker(paper=True, initial_cash=initial_cash)
        bars = bb.get_bars(args.symbol, limit=max(args.bar_window, 300), timeframe=args.timeframe)
        return SimulatedBroker(args.symbol, bars, initial_cash=initial_cash,
                               fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    if name == "alpaca":
        from .alpaca import AlpacaBroker
        return AlpacaBroker(paper=not args.live)
    if name == "binance":
        from .binance import BinanceBroker
        return BinanceBroker(paper=not args.live, initial_cash=initial_cash)
    if name == "livepaper":
        from .live_paper import LivePaperBroker
        if args.data_source == "binance":
            from .binance import BinanceBroker
            md = BinanceBroker(paper=True)
        elif args.data_source == "alpaca":
            from .alpaca import AlpacaBroker
            md = AlpacaBroker(paper=True)
        else:
            raise ValueError(f"unknown data-source {args.data_source}")
        return LivePaperBroker(md, args.symbol, initial_cash=initial_cash,
                               fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    raise ValueError(f"unknown broker {name}")


def _build_strategy(name: str, args):
    registry = _strategies()
    if name not in registry:
        raise ValueError(f"unknown strategy {name}; choose from {list(registry)}")
    cls = registry[name]
    kwargs = {}
    if name == "sma":
        kwargs = {"fast_window": args.sma_fast, "slow_window": args.sma_slow}
    elif name == "meanrev":
        kwargs = {"lookback": args.meanrev_lookback}
    elif name == "hmm":
        kwargs = {"n_states": args.hmm_states}
    return cls(**kwargs)


def _on_step(snap, verbose: bool, log):
    if verbose:
        log.info("step %s close=%.4f signal=%+.3f w=%+.3f pos=%.6f cash=%.2f eq=%.2f bh=%.2f order=%s",
                 snap.timestamp, snap.close, snap.signal, snap.target_weight,
                 snap.position, snap.cash, snap.equity, snap.bh_equity,
                 snap.order_id or "-")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.paper_trading",
        description="Paper-trading engine: run a src.strategy against a live broker sandbox.",
    )
    p.add_argument("--broker", choices=_brokers(), default="simulated")
    p.add_argument("--data-source", choices=["binance", "alpaca"], default="binance",
                   help="market-data feed for --broker livepaper")
    p.add_argument("--strategy", choices=list(_strategies().keys()), default="sma")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--bar-window", type=int, default=200)
    p.add_argument("--initial-cash", type=float, default=10_000.0)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--fee-bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--long-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-weight", type=float, default=1.0)
    p.add_argument("--min-notional", type=float, default=1.0)
    p.add_argument("--min-qty", type=float, default=0.0)
    p.add_argument("--stop-loss-pct", type=float, default=None)
    p.add_argument("--take-profit-pct", type=float, default=None)
    p.add_argument("--state-path", default="paper_trading/state/paper_trading.sqlite")
    p.add_argument("--live", action="store_true", help="use LIVE broker URL (DANGER: real money)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--log-level", default="INFO")
    g = p.add_argument_group("strategy params")
    g.add_argument("--sma-fast", type=int, default=20)
    g.add_argument("--sma-slow", type=int, default=50)
    g.add_argument("--meanrev-lookback", type=int, default=50)
    g.add_argument("--hmm-states", type=int, default=3)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("paper_trading")
    config = EngineConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bar_window=args.bar_window,
        initial_cash=args.initial_cash,
        poll_seconds=args.poll_seconds,
        max_steps=args.max_steps,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        long_only=args.long_only,
        max_weight=args.max_weight,
        min_notional=args.min_notional,
        min_qty=args.min_qty,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        state_path=args.state_path,
    )
    broker = _build_broker(args.broker, args, config.initial_cash)
    strategy = _build_strategy(args.strategy, args)
    state = StateStore(config.state_path)
    try:
        engine = PaperTradingEngine(broker, strategy, config, state)
        result = engine.run(on_step=lambda s: _on_step(s, args.verbose, log))
    finally:
        state.close()

    print("\n=== Paper Trading Summary ===", flush=True)
    for k, v in result.summary().items():
        if isinstance(v, float):
            print(f"  {k:28s}: {v:.4f}", flush=True)
        else:
            print(f"  {k:28s}: {v}", flush=True)
    print(f"\n  snapshots: {len(result.snapshots)}  trades: {len(result.trades)}  orders: {len(result.orders)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
