#!/usr/bin/env python3
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(os.getcwd())
while not ((_root / "src").exists() and (_root / "pyproject.toml").exists()) and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))


try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")
except ImportError:
    pass

from src.paper_trading.binance import BinanceBroker
from src.paper_trading.simulated import SimulatedBroker
from src.paper_trading.engine import PaperTradingEngine, EngineConfig
from src.paper_trading.state import StateStore
from src.strategy.strategies import (
    SmaCrossStrategy,
    MeanReversionStrategy,
    RegimeRouterStrategy,
    S1Hard70Strategy,
    IntensityGatedStrategy,
    HMMRegimeStrategy,
)
from src.paper_trading.alpaca_broker import AlpacaBroker
logger = logging.getLogger("paper_daemon")

SYMBOLS_CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
SYMBOLS_ALPACA_STOCKS = ["AAPL", "MSFT", "NVDA", "COST", "ADBE", "GOOGL"]

_has_alpaca = bool(os.environ.get("ALPACA_API_KEY"))

STRATEGIES = {
    "sma": lambda: SmaCrossStrategy(),
    "meanrev": lambda: MeanReversionStrategy(),
    "router": lambda: RegimeRouterStrategy(),
    "s1hard70": lambda: S1Hard70Strategy(),
    "intensity": lambda: IntensityGatedStrategy(),
}

PORTFOLIOS = [
    {"symbols": SYMBOLS_CRYPTO, "strategy": "sma", "broker": "binance"},
    {"symbols": SYMBOLS_CRYPTO, "strategy": "router", "broker": "binance"},
    {"symbols": SYMBOLS_CRYPTO, "strategy": "s1hard70", "broker": "binance"},
    {"symbols": SYMBOLS_CRYPTO, "strategy": "meanrev", "broker": "binance"},
    {"symbols": SYMBOLS_CRYPTO, "strategy": "intensity", "broker": "binance"},
]

if _has_alpaca:
    for strat in ["sma", "router", "s1hard70"]:
        PORTFOLIOS.append({"symbols": SYMBOLS_ALPACA_STOCKS, "strategy": strat, "broker": "alpaca"})
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
STATS_PATH = _root / "outputs/logs/paper_trading_stats.jsonl"

_stop_event = threading.Event()
_stats_lock = threading.Lock()

def run_single_cycle(symbol, strategy_name, broker_name, initial_cash=10_000.0):
    strategy = STRATEGIES[strategy_name]()
    t0 = time.time()

    if broker_name == "binance":
        b = BinanceBroker(paper=True)
        try:
            df = b.get_bars(symbol, limit=500, timeframe="1m")
        except Exception as e:
            logger.error("erro Binance %s: %s", symbol, e)
            return None
        if df is None or len(df) < 30:
            logger.warning("poucas barras para %s: %s", symbol, len(df) if df is not None else 0)
            return None
        broker = SimulatedBroker(symbol, df, initial_cash=initial_cash, fee_bps=4.0, slippage_bps=1.0)

        warmup_bars = df.iloc[:200] if len(df) >= 200 else df
        max_steps = min(len(df) - 1, 500)

        def _next_bar():
            return broker.advance()
    elif broker_name == "alpaca":
        broker = AlpacaBroker(paper=True)
        try:
            warmup_bars = broker.get_bars(symbol, limit=200, timeframe="1m")
        except Exception as e:
            logger.error("erro Alpaca get_bars %s: %s", symbol, e)
            return None
        status = broker.get_market_status()
        logger.info("Alpaca %s market=%s next_close=%s", symbol, status.get("open"), status.get("next_close", "?"))
        max_steps = 100

        def _next_bar():
            try:
                return broker.get_latest_bar(symbol)
            except Exception:
                return None
    else:
        raise ValueError(f"broker desconhecido: {broker_name}")


    warmup = list(warmup_bars.itertuples(index=False)) if len(warmup_bars) > 0 else []
    bh_base = float(warmup_bars.iloc[-1]["close"]) if len(warmup_bars) > 0 else None
    cash = initial_cash
    position = 0.0
    fee_rate = 4.0 / 10_000.0
    slippage_rate = 1.0 / 10_000.0
    fills_list = []
    all_snaps = []

    for step in range(max_steps):
        if _stop_event.is_set():
            break

        bar = _next_bar()
        if bar is None:
            if broker_name == "alpaca":
                time.sleep(1.0)
            continue

        warmup.append(bar)
        if len(warmup) > 200:
            warmup = warmup[-200:]

        wdf = pd.DataFrame([{
            "timestamp": b.timestamp, "open": b.open, "high": b.high,
            "low": b.low, "close": b.close, "volume": b.volume,
            "vwap": b.vwap if b.vwap is not None else float("nan"),
        } for b in warmup[-200:]])

        if len(wdf) < 2:
            continue

        try:
            signal_val = float(strategy.generate_signals(wdf).iloc[-1])
        except Exception as e:
            logger.error("erro signal %s/%s: %s", symbol, strategy_name, e)
            signal_val = 0.0

        if not np.isfinite(signal_val):
            signal_val = 0.0

        w_val = float(np.clip(signal_val, -1.0, 1.0))
        if w_val < 0:
            w_val = 0.0

        price = float(bar.close)
        equity = cash + position * price
        target_units = (equity * w_val) / price if price > 0 else 0.0
        delta = target_units - position

        fee = 0.0
        slippage = 0.0
        if abs(delta) > 1e-12:
            notional = delta * price
            fee = abs(notional) * fee_rate
            cost_price = price * (1 + np.sign(delta) * slippage_rate)
            slippage = abs(notional) * slippage_rate
            cash -= fee + slippage
            cash -= delta * cost_price
            position += delta

            fills_list.append({
                "timestamp": bar.timestamp, "side": "buy" if delta > 0 else "sell",
                "qty": abs(delta), "price": cost_price, "fee": fee, "slippage": slippage,
                "notional": abs(notional), "cash_after": cash, "position_after": position,
            })

        equity_after = cash + position * price
        if bh_base is None:
            bh_base = price
        bh_equity = (price / bh_base) * initial_cash if bh_base else equity_after

        all_snaps.append({
            "timestamp": bar.timestamp, "close": price, "signal": signal_val,
            "target_weight": w_val, "position": position, "cash": cash,
            "equity": equity_after, "bh_equity": bh_equity,
        })

    if len(all_snaps) < 5:
        return None

    snap_df = pd.DataFrame(all_snaps)
    bh_curve = snap_df["bh_equity"].values
    eq_curve = snap_df["equity"].values

    if len(eq_curve) < 2:
        return None

    rets = np.diff(eq_curve) / eq_curve[:-1]
    bh_rets = np.diff(bh_curve) / bh_curve[:-1]

    total_ret = ((eq_curve[-1] / eq_curve[0]) - 1) * 100
    bh_ret = ((bh_curve[-1] / bh_curve[0]) - 1) * 100
    excess = total_ret - bh_ret

    avg_ret = np.mean(rets)
    vol = np.std(rets) + 1e-12
    sharpe = avg_ret / vol * np.sqrt(365 * 24 * 60)

    peak = np.maximum.accumulate(eq_curve)
    dd = ((eq_curve / peak) - 1).min() * 100

    n_trades = len(fills_list)

    elapsed = time.time() - t0

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "symbol": symbol,
        "strategy": strategy_name,
        "broker": broker_name,
        "total_return_pct": round(float(total_ret), 4),
        "bh_return_pct": round(float(bh_ret), 4),
        "excess_return_pct": round(float(excess), 4),
        "sharpe": round(float(sharpe), 4),
        "max_dd_pct": round(float(dd), 4),
        "steps": len(all_snaps),
        "n_trades": n_trades,
        "elapsed_s": round(elapsed, 1),
    }

    with _stats_lock:
        with open(STATS_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        "✅ %s/%-10s ret=%+7.2f%% bh=%+7.2f%% exc=%+7.2f%% sharpe=%+.3f dd=%.1f%% trades=%d (%d steps, %.1fs)",
        symbol, strategy_name, total_ret, bh_ret, excess, sharpe, dd, n_trades, len(all_snaps), elapsed,
    )
    return entry

def worker(symbol, strategy_name, broker_name):
    msg = f"{symbol}/{strategy_name}"
    logger.info("▶  %s", msg)
    cycle = 0

    while not _stop_event.is_set():
        cycle += 1
        result = run_single_cycle(symbol, strategy_name, broker_name)

        if result is None:
            logger.warning("❌ %s [c%d]: sem resultado", msg, cycle)

        for _ in range(60):
            if _stop_event.is_set():
                break
            time.sleep(1)

    logger.info("◼  %s (encerrado)", msg)

def print_status():
    if STATS_PATH.exists():
        try:
            df = pd.read_json(STATS_PATH, lines=True)
            if len(df) > 0:
                recent = df[df["run_id"] == RUN_ID]
                print(f"\n═══ RESUMO — {RUN_ID} — {datetime.now().strftime('%H:%M:%S')} ═══")
                if len(recent) > 0:
                    summary = recent.groupby(["symbol", "strategy"]).agg(
                        ciclos=("excess_return_pct", "count"),
                        exc_medio=("excess_return_pct", "mean"),
                        ret_medio=("total_return_pct", "mean"),
                        sharpe_medio=("sharpe", "mean"),
                        trades_total=("n_trades", "sum"),
                    ).round(2)
                    print(summary.to_string())
                print(f"Total de ciclos: {len(recent)}")
                print("═" * 50)
        except Exception:
            pass

def signal_handler(sig, frame):
    logger.info("sinal %s — encerrando workers...", sig)
    _stop_event.set()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    threads = []
    stagger = 0.0
    for pf in PORTFOLIOS:
        for sym in pf["symbols"]:
            t = threading.Thread(
                target=worker,
                args=(sym, pf["strategy"], pf["broker"]),
                daemon=True,
                name=f"{sym}/{pf['strategy']}",
            )
            t.start()
            threads.append(t)
            stagger += 0.3
            time.sleep(0.3)

    n = len(threads)
    logger.info("🚀 PAPER TRADING DAEMON INICIADO")
    logger.info("   Workers ativos: %d", n)
    logger.info("   Símbolos: %s", ", ".join(SYMBOLS_CRYPTO))
    logger.info("   Estratégias: %s", ", ".join(sorted(set(pf["strategy"] for pf in PORTFOLIOS))))
    logger.info("   Run ID: %s", RUN_ID)
    logger.info("   Log: outputs/logs/paper_trading_daemon.log")
    logger.info("   Stats: outputs/logs/paper_trading_stats.jsonl")
    logger.info("   Para encerrar: tmux kill-session -t paper_trading")
    logger.info("   Para ver status: cat outputs/logs/paper_trading_stats.jsonl | tail -5")
    logger.info("")

    status_interval = 90
    last_status = time.time()

    while not _stop_event.is_set():
        now = time.time()
        if now - last_status > status_interval:
            print_status()
            last_status = now
        time.sleep(5)

    logger.info("aguardando workers (max 15s)...")
    for t in threads:
        t.join(timeout=15)
    print_status()
    logger.info("🏁 DAEMON FINALIZADO — %s"),

if __name__ == "__main__":
    main()
