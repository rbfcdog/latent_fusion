#!/usr/bin/env python3
import json
import logging
import os
import signal as _signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(os.getcwd())
while not ((_root / "src").exists() and (_root / "pyproject.toml").exists()) and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

os.environ.setdefault("ALPACA_API_KEY", "PKWW4JEN4DZCOEOUADEJ5B42EP")
os.environ.setdefault("ALPACA_API_SECRET", "Cg1L3LYkWaS3ayVzkymihwcFeJZ2tB8uTSb26TEEpZgW")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

LOG_DIR = _root / "outputs/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run_service(strategy_name, StrategyClass, symbols, max_weight, order_pct):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    perf_path = LOG_DIR / f"perf_{strategy_name}.jsonl"
    trade_path = LOG_DIR / f"trades_{strategy_name}.jsonl"
    log_path = LOG_DIR / f"{strategy_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(str(log_path)), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(strategy_name)

    _stop = False
    def handle(sig, frame):
        nonlocal _stop
        logger.info("sinal %s - encerrando", sig)
        _stop = True
    _signal.signal(_signal.SIGINT, handle)
    _signal.signal(_signal.SIGTERM, handle)

    key = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_API_SECRET"]
    trading = TradingClient(api_key=key, secret_key=secret, paper=True)
    data_client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    acct = trading.get_account()
    initial_equity = float(acct.equity)
    logger.info("SERVICE STARTED run_id=%s strategy=%s symbols=%s equity=$%.0f",
                run_id, strategy_name, ",".join(symbols), initial_equity)

    strategies = {sym: StrategyClass() for sym in symbols}
    last_trade = {sym: 0 for sym in symbols}
    positions = {sym: 0.0 for sym in symbols}
    cooldown = 5
    step = 0
    last_summary = time.time()
    SUMMARY_INTERVAL = 300

    while not _stop:
        step += 1
        try:
            clock = trading.get_clock()
            market_open = clock.is_open
        except Exception:
            market_open = False

        if not market_open:
            if step == 1 or step % 300 == 0:
                logger.info("mercado fechado (step=%d)", step)
            time.sleep(1)
            continue

        for symbol in symbols:
            try:
                req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=200)
                bars_resp = data_client.get_stock_bars(req)
                bar_list = list(bars_resp.data.get(symbol, []))
                if not bar_list:
                    continue
                rows = [{"timestamp": b.timestamp, "open": b.open, "high": b.high,
                         "low": b.low, "close": b.close, "volume": b.volume,
                         "vwap": getattr(b, "vwap", b.close)} for b in bar_list]
                df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
                if len(df) < 60:
                    continue
                sig_series = strategies[symbol].generate_signals(df)
                sig_val = float(sig_series.iloc[-1])
            except Exception:
                continue

            if abs(sig_val) < 0.5 or step - last_trade[symbol] < cooldown:
                continue

            try:
                pos = trading.get_open_position(symbol)
                positions[symbol] = float(pos.qty)
            except Exception:
                positions[symbol] = 0.0

            try:
                acct = trading.get_account()
                equity = float(acct.equity)
            except Exception:
                continue

            price = float(df["close"].iloc[-1])
            if price <= 0:
                continue

            current_qty = positions[symbol]
            current_w = (current_qty * price) / equity if equity > 0 else 0.0

            trade_entry = None

            if sig_val > 0 and current_w < max_weight:
                order_qty = max(1, int((equity * order_pct) / price))
                try:
                    req = MarketOrderRequest(symbol=symbol, qty=order_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    o = trading.submit_order(req)
                    trade_entry = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                                   "side": "BUY", "symbol": symbol, "qty": order_qty, "price": price,
                                   "signal": round(sig_val, 4), "equity": round(equity, 2),
                                   "order_id": str(o.id)[:12]}
                    logger.info("BUY  %-6s qty=%-4d @ $%.2f sig=%+.3f eq=$%.0f",
                                symbol, order_qty, price, sig_val, equity)
                    last_trade[symbol] = step
                except Exception as e:
                    logger.warning("BUY %s failed: %s", symbol, e)

            elif sig_val < 0 and current_qty > 0:
                try:
                    o = trading.close_position(symbol)
                    trade_entry = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                                   "side": "SELL", "symbol": symbol, "qty": int(current_qty), "price": price,
                                   "signal": round(sig_val, 4), "equity": round(equity, 2),
                                   "order_id": str(o.id)[:12]}
                    logger.info("SELL %-6s qty=%-4d @ $%.2f sig=%+.3f eq=$%.0f",
                                symbol, int(current_qty), price, sig_val, equity)
                    last_trade[symbol] = step
                except Exception as e:
                    logger.warning("SELL %s failed: %s", symbol, e)

            if trade_entry:
                with open(trade_path, "a") as f:
                    f.write(json.dumps(trade_entry) + "\n")

        now = time.time()
        if now - last_summary > SUMMARY_INTERVAL:
            try:
                acct = trading.get_account()
                equity = float(acct.equity)
                ret_pct = ((equity - initial_equity) / initial_equity) * 100
                perf = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                        "equity": round(equity, 2), "return_pct": round(ret_pct, 4),
                        "steps": step, "strategy": strategy_name}
                with open(perf_path, "a") as f:
                    f.write(json.dumps(perf) + "\n")
                logger.info("PERF step=%d equity=$%.0f ret=%+.4f%%", step, equity, ret_pct)
                last_summary = now
            except Exception:
                pass

        time.sleep(1.0)

    logger.info("SERVICE STOPPED step=%d", step)

if __name__ == "__main__":
    strat = os.environ.get("ML_STRATEGY", "hft")
    if strat == "hft":
        from src.strategy.ml_hft import MLHFTStrategy
        run_service("ml_hft", MLHFTStrategy,
                    ["AAPL", "MSFT", "NVDA", "COST", "GOOGL", "ADBE"],
                    max_weight=0.20, order_pct=0.05)
    elif strat == "ensemble":
        from src.strategy.ml_ensemble import MLEnsembleStrategy
        run_service("ml_ensemble", MLEnsembleStrategy,
                    ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
                    max_weight=0.25, order_pct=0.06)
    else:
        print(f"unknown ML_STRATEGY={strat}", file=sys.stderr)
        sys.exit(1)
