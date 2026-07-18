# B3 / Brazil Setup (MetaTrader 5) — future work

The project's TODO targets B3 (Brazilian exchange) and NuBank at 1m timeframe. The
cleanest paper path for B3 is a **MetaTrader 5 demo account** from a Brazilian broker.
No `MT5Broker` adapter is implemented yet — this documents the path.

## Why MT5 for B3

- B3 brokers (XP, Clear, Rico, BTG Pactual, Nelogica) offer MT5 with demo accounts.
  An open-source trading-robot framework survey (arXiv:2101.08169) explicitly lists XP
  and Clear as providing demo accounts for B3.
- OnTick (ontick.com.br) automates XP/Rico/Clear accounts — confirms retail automation
  on B3 is broker-mediated (no public exchange FIX for individuals).
- There is no B3 public paper REST API for individuals; the broker terminal is the gate.

## Setup steps (when implementing)

1. **Open a demo account** with XP / Clear / Rico / BTG / Nelogica. You receive:
   - login (integer), password, server name (e.g. `XPMT5-DEMO`).
2. **Install MT5 terminal** — Windows native, or via Wine on Linux. The terminal must
   be running and logged in for the Python API to connect.
3. **Install the Python package:**
   ```bash
   uv add MetaTrader5
   ```
4. **Implement `MT5Broker`** against the `BrokerClient` protocol:

| Protocol method | MT5 call |
|---|---|
| `get_account` | `mt5.account_info()` |
| `get_position` | `mt5.positions_get(symbol=symbol)` |
| `get_latest_bar` / `get_bars` | `mt5.copy_rates_from_pos(symbol, timeframe, start, count)` |
| `get_market_status` | `mt5.symbol_info_tick(symbol)` + session logic |
| `submit_market_order` | `mt5.order_send(request)` with `ORDER_TYPE_BUY/SELL` |
| `submit_limit_order` | `mt5.order_send` with `ORDER_TYPE_BUY/SELL_LIMIT` |
| `cancel_order` | `mt5.order_send` with `ORDER_TYPE_REMOVE` |
| `get_order` | `mt5.history_orders_get(ticket=...)` |
| `close_all` | `mt5.positions_get` → opposite `order_send` per position |

5. **Symbol format:** B3 suffixes — e.g. `PETR4`, `VALE3`, `WDOF25` (mini dollar
   future), `WINV25` (mini index). Forex uses `USDJPY` etc. Use the broker's symbol.
6. **Timeframes:** MT5 enum `mt5.TIMEFRAME_M1`, `_M5`, `_H1`, `_D1`.

## Constraints

- **Not headless:** the MT5 terminal GUI must run. On Linux, Wine + a virtual display
  (Xvfb) is the usual workaround. Heavier than Alpaca/Binance.
- **Demo = paper:** the demo account holds virtual BRL. No real-money risk.
- **Market hours:** B3 equities 10:00–17:30 BRT; futures 9:00–18:00 + overnight session.
  Respect the broker's session calendar.

## Alternative B3 paths (not recommended)

- **B3 direct / UMDF / PUMA:** institutional, licensed members only. Not for this
  research project.
- **Crawler on cotahist (`data/cotahist/`):** end-of-day only; not real-time paper.

## Sources

- https://arxiv.org/pdf/2101.08169 (B3/XP/Clear demo accounts)
- https://www.ontick.com.br/ (XP/Rico/Clear retail automation)
- https://www.mql5.com/en/docs/integration/python_metatrader5 (MT5 Python API)
- https://www.btgpactual.com / https://www.nelogica.com.br/ (broker terminals)
