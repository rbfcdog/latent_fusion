# Architecture

`src/paper_trading/` runs a `src.strategy.Strategy` against a live broker sandbox in
real time, mirroring `src.backtest.BacktestEngine` but with real order submission,
real market data, and SQLite-persisted state.

## Pipeline

```
broker.get_bars()  ──►  warmup window (bar_window bars)
        │
        ▼  each poll interval
broker.get_latest_bar()  ──►  append to warmup df
        │
        ▼
strategy.generate_signals(df)  ──►  signal ∈ [-1, 1]  (last value)
        │
        ▼
clip to long-only / max_weight  ──►  target_weight
        │
        ▼
rebalance: target_qty = equity * target_weight / price
        │   delta = target_qty - current_position
        │   skip if |delta| < min_qty or notional < min_notional
        ▼
broker.submit_market_order(symbol, side, |delta|)  ──►  Fill
        │
        ▼
snapshot {close, signal, weight, position, cash, equity, bh_equity}  ──►  SQLite
        │
        ▼  (optional) stop-loss / take-profit check on position
        ▼
loop until max_steps or interrupted
        │
        ▼
compute_metrics(equity, trades, bh_equity)  ──►  EngineResult
```

## Components

```
src/paper_trading/
├── base.py          # Protocol + dataclasses: BrokerClient, Order, Fill, Position,
│                    #   Account, Bar, EngineConfig, EngineResult, EngineSnapshot
├── engine.py        # PaperTradingEngine: warmup → step loop → rebalance → metrics
├── state.py         # StateStore: SQLite persistence of orders/fills/snapshots/kv
├── metrics.py       # compute_metrics: Sharpe, Sortino, max DD, Calmar, α/β vs BH
├── simulated.py     # SimulatedBroker: in-process bar replay (offline/CI)
├── alpaca.py        # AlpacaBroker: real Alpaca paper API (alpaca-py)
├── binance.py       # BinanceBroker: real Binance Spot Testnet (requests + HMAC)
├── __main__.py      # CLI: --broker --strategy --symbol --timeframe ...
└── docs/            # this directory
```

## Key contracts

### `BrokerClient` Protocol (`base.py`)

Every adapter implements:
- `get_account() -> Account`
- `get_position(symbol) -> Position | None`
- `get_latest_bar(symbol) -> Bar`
- `get_bars(symbol, limit, timeframe) -> pd.DataFrame`
- `get_market_status() -> dict`
- `submit_market_order(symbol, side, qty) -> Order`
- `submit_limit_order(symbol, side, qty, limit_price) -> Order`
- `cancel_order(order_id, symbol) -> bool`
- `get_order(order_id, symbol) -> Order`
- `close_all(symbol) -> list[Order]`

`name: str` and `is_live: bool` identify the adapter and whether it points at a live
URL. The engine never inspects these — they are for logging/CLI safety.

### `Strategy` Protocol (reused from `src.backtest.engine`)

```python
class Strategy(Protocol):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series: ...
```

The engine calls `generate_signals(warmup_df)` each step and takes `.iloc[-1]` as the
target weight ∈ [-1, 1]. **Same contract as the backtest engine** — any strategy that
backtests can paper-trade unchanged.

### `EngineConfig` knobs

| Field | Default | Purpose |
|---|---|---|
| `symbol` | — | trade symbol (e.g. `BTCUSDT`, `AAPL`) |
| `timeframe` | `1m` | bar interval (broker-mapped) |
| `bar_window` | 200 | warmup + rolling window length |
| `long_only` | True | clip negative weights to 0 (AGENTS.md rule 3) |
| `max_weight` | 1.0 | cap |target_weight| |
| `min_notional` | 1.0 | skip orders below this notional (broker min) |
| `poll_seconds` | 60.0 | sleep between steps (0 = back-to-back, for replay) |
| `max_steps` | None | stop after N steps (None = until interrupted) |
| `fee_bps` / `slippage_bps` | 0/0 | applied by SimulatedBroker |
| `stop_loss_pct` / `take_profit_pct` | None | optional auto-exit |
| `state_path` | `paper_trading/state/paper_trading.sqlite` | SQLite path |

## No-lookahead compliance (AGENTS.md rule 1)

The engine never uses forward returns. At time `t` it:
1. Fetches bars strictly up to `t` (`get_bars`, `get_latest_bar`).
2. Computes the signal from that window only.
3. Places a market order at the current price.

`ret_df` (forward returns) is never imported or referenced. The signal at `t` trades
the bar at `t`'s close — the same convention as `BacktestEngine`, which uses
`.shift(1)` inside strategies that need it (e.g. `HMMRegimeStrategy`).

## Buy-and-hold benchmark (AGENTS.md rule 5)

Every run records `bh_equity = close / close[0] * initial_cash` per snapshot, and
`compute_metrics` reports `bh_return_pct`, `excess_return_pct`, `alpha`, `beta` vs
that benchmark. The CLI summary always prints both strategy and BH final equity.

## State persistence

SQLite (`state.py`) stores:
- `engine_state` (KV): run metadata.
- `orders`: every submitted order, upserted on status change.
- `fills`: every execution fill.
- `snapshots`: per-step equity/position/signal, keyed by `run_id`.

`StateStore.load_orders() / load_fills() / load_snapshots(run_id)` recover a run for
post-hoc analysis. `clear_all()` resets for a clean run.

## Safety

- `--live` flag on the CLI is required to point adapters at live URLs. Default is paper.
- `BinanceBroker` and `AlpacaBroker` log the base URL on init.
- `SimulatedBroker` never touches the network (except to fetch seed bars via the
  public market-data endpoint, or a passed-in DataFrame).
