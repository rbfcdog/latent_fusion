# Usage

## Quick start — simulated replay (no broker keys)

Replays real Binance testnet klines through the engine with the SMA-cross strategy.
No credentials required.

```bash
uv run python -m src.paper_trading \
  --broker simulated \
  --strategy sma \
  --symbol BTCUSDT \
  --timeframe 1m \
  --bar-window 200 \
  --max-steps 80 \
  --poll-seconds 0 \
  --fee-bps 4 --slippage-bps 1 \
  --verbose
```

Output (verified smoke test):

```
[2026-07-17 02:34:00+00:00] close=63494.61 signal=+1.000 w=+1.000 pos=0.157494 cash=9965.51 eq=10000.00 bh=10000.00 order=a1b2...
...
=== Paper Trading Summary ===
  total_return_pct             : -0.2452
  bh_return_pct                :  0.0998
  excess_return_pct            : -0.3945
  n_trades                     :  4.0000
  final_equity                 : 9970.4947
  bh_final_equity              : 10009.9835
```

## Real paper trading — Alpaca

```bash
export ALPACA_API_KEY=PK...
export ALPACA_API_SECRET=...
uv run python -m src.paper_trading \
  --broker alpaca \
  --strategy sma \
  --symbol AAPL \
  --timeframe 1m \
  --poll-seconds 60 \
  --max-steps 100
```

See `alpaca_setup.md` for credentials and symbol notes.

## Real paper trading — Binance Spot Testnet

```bash
export BINANCE_API_KEY=...   # from testnet.binance.vision (GitHub login)
export BINANCE_API_SECRET=...
uv run python -m src.paper_trading \
  --broker binance \
  --strategy router \
  --symbol BTCUSDT \
  --timeframe 1m \
  --poll-seconds 60
```

Market-data (klines, ping) works **without keys**; only order submission needs them.
See `binance_setup.md`.

## Programmatic API

```python
from src.paper_trading import (
    AlpacaBroker, BinanceBroker, SimulatedBroker,
    PaperTradingEngine, EngineConfig, StateStore,
)
from src.paper_trading.binance import BinanceBroker
from src.strategy import RegimeRouterStrategy

# Offline replay of real testnet bars
broker = BinanceBroker(paper=True, initial_cash=10_000.0)
bars = broker.get_bars("BTCUSDT", limit=300, timeframe="1m")
sim = SimulatedBroker("BTCUSDT", bars, initial_cash=10_000.0,
                      fee_bps=4.0, slippage_bps=1.0)

cfg = EngineConfig(symbol="BTCUSDT", bar_window=200,
                   poll_seconds=0.0, max_steps=80,
                   fee_bps=4.0, slippage_bps=1.0,
                   state_path="/tmp/paper.sqlite")
state = StateStore(cfg.state_path)
engine = PaperTradingEngine(sim, RegimeRouterStrategy(), cfg, state)
result = engine.run()
state.close()

print(result.summary())          # dict of metrics + final/BH equity
print(result.snapshots.tail())   # per-step DataFrame
print(result.trades)             # fills DataFrame
```

## Strategies available (`--strategy`)

From `src.strategy` — same classes used in backtests:

| CLI key | Class | Notes |
|---|---|---|
| `sma` | `SmaCrossStrategy` | `--sma-fast 20 --sma-slow 50` |
| `meanrev` | `MeanReversionStrategy` | `--meanrev-lookback 50` |
| `hmm` | `HMMRegimeStrategy` | `--hmm-states 3` |
| `vwap` | `VwapReversionStrategy` | needs `vwap` column |
| `instv3` | `InstitutionalV3Strategy` | |
| `router` | `RegimeRouterStrategy` | vol-regime routed |
| `intensity` | `IntensityGatedStrategy` | |
| `s1hard70` | `S1Hard70Strategy` | |

Any strategy implementing `generate_signals(df) -> pd.Series` (the `Strategy` protocol
shared with `src.backtest.engine`) works.

## CLI flags

```
--broker {alpaca,binance,simulated}   default: simulated
--strategy KEY                        default: sma
--symbol STR                          default: BTCUSDT
--timeframe STR                       default: 1m  (1/5/15m,1h,1d)
--bar-window N                        default: 200
--initial-cash FLOAT                  default: 10000
--poll-seconds FLOAT                  default: 60 (0 = back-to-back replay)
--max-steps N                         default: unlimited
--fee-bps / --slippage-bps FLOAT      default: 0 (SimulatedBroker)
--long-only / --no-long-only          default: long-only (AGENTS.md rule 3)
--max-weight FLOAT                    default: 1.0
--min-notional FLOAT                  default: 1.0
--min-qty FLOAT                       default: 0
--stop-loss-pct / --take-profit-pct   optional auto-exit
--state-path PATH                     SQLite state file
--live                                use LIVE broker URL (DANGER: real money)
--verbose / -v                        print each step
```

## Recovering a run

```python
from src.paper_trading import StateStore
state = StateStore("paper_trading/state/paper_trading.sqlite")
print(state.load_snapshots("a1b2c3d4e5f6").tail())  # by run_id
print(state.load_orders())
print(state.load_fills())
```
