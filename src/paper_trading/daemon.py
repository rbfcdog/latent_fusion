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

REBALANCE_THRESHOLD = 0.05


def _new_state(initial_cash):
    return {
        "strategy": None,
        "broker": None,
        "cash": initial_cash,
        "position": 0.0,
        "bh_base": None,
        "bars_hist": [],
        "snaps": [],
        "n_fills": 0,
        "last_ts": None,
        "last_fill_ts": None,
    }


def run_single_cycle(state, symbol, strategy_name, broker_name, initial_cash=10_000.0):
    if state["strategy"] is None:
        state["strategy"] = STRATEGIES[strategy_name]()
    strategy = state["strategy"]
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
        fetched = list(df.itertuples(index=False))
    elif broker_name == "alpaca":
        if state["broker"] is None:
            state["broker"] = AlpacaBroker(paper=True)
        broker = state["broker"]
        try:
            if not state["bars_hist"]:
                fetched = list(broker.get_bars(symbol, limit=200, timeframe="1m").itertuples(index=False))
            else:
                bar = broker.get_latest_bar(symbol)
                fetched = [bar] if bar is not None else []
        except Exception as e:
            logger.error("erro Alpaca %s: %s", symbol, e)
            return None
    else:
        raise ValueError(f"broker desconhecido: {broker_name}")

    if not fetched:
        return None

    if state["last_ts"] is None:
        state["bars_hist"] = list(fetched[-200:])
        step_bars = list(fetched[-60:])
    else:
        step_bars = [b for b in fetched if b.timestamp > state["last_ts"]]
    if not step_bars:
        return None

    cash = state["cash"]
    position = state["position"]
    fee_rate = 4.0 / 10_000.0
    slippage_rate = 1.0 / 10_000.0
    for bar in step_bars:
        if _stop_event.is_set():
            break
        state["last_ts"] = bar.timestamp
        state["bars_hist"].append(bar)
        if len(state["bars_hist"]) > 200:
            state["bars_hist"] = state["bars_hist"][-200:]

        wdf = pd.DataFrame([{
            "timestamp": b.timestamp, "open": b.open, "high": b.high,
            "low": b.low, "close": b.close, "volume": b.volume,
            "vwap": b.vwap if getattr(b, "vwap", None) is not None else float("nan"),
        } for b in state["bars_hist"]])

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
        cur_weight = (position * price) / equity if equity > 0 else 0.0
        if abs(w_val - cur_weight) < REBALANCE_THRESHOLD:
            target_units = position
        else:
            target_units = (equity * w_val) / price if price > 0 else 0.0
        delta = target_units - position

        ts = pd.Timestamp(bar.timestamp)
        cooldown_ok = (
            state["last_fill_ts"] is None
            or (ts - state["last_fill_ts"]).total_seconds() >= 900
        )
        if abs(delta) > 1e-12 and cooldown_ok:
            notional = delta * price
            fee = abs(notional) * fee_rate
            cost_price = price * (1 + np.sign(delta) * slippage_rate)
            slippage = abs(notional) * slippage_rate
            cash -= fee + slippage
            cash -= delta * cost_price
            position += delta
            state["n_fills"] += 1
            state["last_fill_ts"] = ts

        equity_after = cash + position * price
        if state["bh_base"] is None:
            state["bh_base"] = price
        bh_equity = (price / state["bh_base"]) * initial_cash

        state["snaps"].append({
            "timestamp": bar.timestamp, "close": price, "signal": signal_val,
            "target_weight": w_val, "position": position, "cash": cash,
            "equity": equity_after, "bh_equity": bh_equity,
        })

    state["cash"] = cash
    state["position"] = position

    if len(state["snaps"]) < 5:
        return None

    snap_df = pd.DataFrame(state["snaps"])
    bh_curve = snap_df["bh_equity"].values
    eq_curve = snap_df["equity"].values

    rets = np.diff(eq_curve) / eq_curve[:-1]

    total_ret = ((eq_curve[-1] / eq_curve[0]) - 1) * 100
    bh_ret = ((bh_curve[-1] / bh_curve[0]) - 1) * 100
    excess = total_ret - bh_ret

    sharpe = None
    if len(rets) >= 240:
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(365 * 24 * 60))

    peak = np.maximum.accumulate(eq_curve)
    dd = ((eq_curve / peak) - 1).min() * 100

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
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "max_dd_pct": round(float(dd), 4),
        "steps": len(snap_df),
        "new_bars": len(step_bars),
        "n_trades": state["n_fills"],
        "elapsed_s": round(elapsed, 1),
    }

    with _stats_lock:
        with open(STATS_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        "✅ %s/%-10s ret=%+7.2f%% bh=%+7.2f%% exc=%+7.2f%% sharpe=%s dd=%.1f%% trades=%d (+%d barras, %d total, %.1fs)",
        symbol, strategy_name, total_ret, bh_ret, excess,
        f"{sharpe:+.3f}" if sharpe is not None else "n/a",
        dd, state["n_fills"], len(step_bars), len(snap_df), elapsed,
    )
    return entry

def worker(symbol, strategy_name, broker_name):
    msg = f"{symbol}/{strategy_name}"
    logger.info("▶  %s", msg)
    state = _new_state(10_000.0)

    while not _stop_event.is_set():
        result = run_single_cycle(state, symbol, strategy_name, broker_name)

        if result is None and state["last_ts"] is None:
            logger.warning("❌ %s: sem barras", msg)

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(_root / "outputs/logs/paper_trading_daemon.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
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
    logger.info("🏁 DAEMON FINALIZADO — %s", RUN_ID)

if __name__ == "__main__":
    main()
