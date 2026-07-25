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

try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")
except ImportError:
    pass

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.strategy.ml_hft import MLHFTStrategy

os.makedirs("outputs/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("outputs/logs/ml_hft.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ml_hft")

SYMBOLS = ["AAPL", "MSFT", "NVDA", "COST", "GOOGL", "ADBE"]
MAX_POSITION_PCT = 0.20
ORDER_SIZE_PCT = 0.05

_stop = False

def handle_signal(sig, frame):
    global _stop
    logger.info("sinal %s - encerrando", sig)
    _stop = True

def main():
    global _stop
    _signal.signal(_signal.SIGINT, handle_signal)
    _signal.signal(_signal.SIGTERM, handle_signal)

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        logger.error("ALPACA_API_KEY e ALPACA_API_SECRET necessarios")
        return 1

    trading = TradingClient(api_key=key, secret_key=secret, paper=True)
    data_client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    acct = trading.get_account()
    logger.info("ML HFT ALGORITHM INICIADO")
    logger.info("  Conta: equity=$%s cash=$%s", acct.equity, acct.cash)
    logger.info("  Simbolos: %s", ", ".join(SYMBOLS))
    logger.info("  Modelo: SGDRegressor online (Huber loss)")
    logger.info("  Timeframe: 1m | Ordem: PAPER (sem dinheiro real)")
    logger.info("")

    strategies = {sym: MLHFTStrategy(lookback=60, retrain_every=10) for sym in SYMBOLS}
    last_trade = {sym: 0 for sym in SYMBOLS}
    cooldown_bars = 5
    step = 0

    while not _stop:
        step += 1

        try:
            clock = trading.get_clock()
            market_open = clock.is_open
        except Exception:
            market_open = False

        if not market_open:
            now = datetime.now().strftime("%H:%M:%S")
            if step == 1 or step % 60 == 0:
                logger.info("[%s] Mercado fechado. Aguardando... (step=%d)", now, step)
            time.sleep(1)
            continue

        t0 = time.time()

        for symbol in SYMBOLS:
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    limit=200,
                )
                bars_resp = data_client.get_stock_bars(req)
                bar_list = list(bars_resp.data.get(symbol, []))
                if not bar_list:
                    continue

                rows = [{
                    "timestamp": b.timestamp, "open": b.open, "high": b.high,
                    "low": b.low, "close": b.close, "volume": b.volume,
                    "vwap": getattr(b, "vwap", b.close),
                } for b in bar_list]
                df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
                if len(df) < 60:
                    continue

                sig_series = strategies[symbol].generate_signals(df)
                sig_val = float(sig_series.iloc[-1])

                if abs(sig_val) < 0.5 or step - last_trade[symbol] < cooldown_bars:
                    continue

                pos = trading.get_open_position(symbol)
                current_qty = float(pos.qty)
            except Exception:
                continue

            try:
                acct = trading.get_account()
                equity = float(acct.equity)
            except Exception:
                continue

            price = float(df["close"].iloc[-1])
            if price <= 0:
                continue

            current_weight = (current_qty * price) / equity if equity > 0 else 0.0

            if sig_val > 0 and current_weight < MAX_POSITION_PCT:
                order_qty = max(1, int((equity * ORDER_SIZE_PCT) / price))
                try:
                    req = MarketOrderRequest(
                        symbol=symbol, qty=order_qty,
                        side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                    )
                    o = trading.submit_order(req)
                    logger.info("BUY  %-6s qty=%-4d price=$%.2f id=%s", symbol, order_qty, price, str(o.id)[:8])
                    last_trade[symbol] = step
                except Exception as e:
                    logger.warning("BUY %s failed: %s", symbol, e)

            elif sig_val < 0 and current_qty > 0:
                try:
                    o = trading.close_position(symbol)
                    logger.info("SELL %-6s qty=%-4d price=$%.2f id=%s", symbol, int(current_qty), price, str(o.id)[:8])
                    last_trade[symbol] = step
                except Exception as e:
                    logger.warning("SELL %s failed: %s", symbol, e)

        elapsed = time.time() - t0
        if elapsed < 5.0:
            time.sleep(5.0 - elapsed)

    logger.info("ML HFT FINALIZADO apos %d steps", step)
    try:
        acct = trading.get_account()
        logger.info("Equity final: $%s", acct.equity)
    except Exception:
        pass

if __name__ == "__main__":
    sys.exit(main() or 0)
