# Broker Options for Paper Trading

Comparison of broker APIs evaluated for the `src.paper_trading` module. Goal: a real
paper-trading sandbox that executes live orders against a broker's test environment,
bridging `src.strategy` signals → broker orders → equity curve, with buy-and-hold
benchmark (per `AGENTS.md` rule 5).

---

## Summary table

| Broker | Paper env | Markets | Python SDK | Runs on Linux headless | Auth | Cost | Adapter |
|---|---|---|---|---|---|---|---|
| **Alpaca** | `paper-api.alpaca.markets` | US stocks, crypto, options | `alpaca-py` (official) | ✅ | API key/secret (REST) | Free paper, no min | `AlpacaBroker` ✅ |
| **Binance Spot Testnet** | `testnet.binance.vision` | Crypto spot | raw `requests` + HMAC-SHA256 | ✅ | API key/secret (HMAC) | Free | `BinanceBroker` ✅ |
| **Interactive Brokers** | IBKR paper account | Global stocks/options/futures/cfd | `ib_async` | ⚠️ needs TWS or IB Gateway running | TWS gateway session | Free paper w/ account | documented (not impl) |
| **MetaTrader 5** | MT5 demo account | B3 (XP/Clear/Rico/BTG/Nelogica), forex, CFD | `MetaTrader5` | ⚠️ MT5 terminal (Win/Wine) | login/pwd/server | Free demo | documented (not impl) |
| **Tradier** | Tradier sandbox | US stocks/options | REST | ✅ | API token | Free sandbox | not impl |
| **QuantConnect LEAN** | LEAN paper brokerage | Multi-asset | `lean-cli` | ✅ | CLI + broker creds | Free local | not impl |
| **SimulatedBroker** | in-process replay | any | none | ✅ | none | free | `SimulatedBroker` ✅ |

---

## 1. Alpaca (primary — implemented)

- **Paper URL:** `https://paper-api.alpaca.markets`
- **Live URL:** `https://api.alpaca.markets`
- **SDK:** `alpaca-py` (added to `pyproject.toml`)
- **Why primary:** Free unlimited paper trading, commission-free, developer-first REST +
  streaming, first-class Python SDK, runs headless on Linux. Repeatedly cited as the
  best free-to-start API for Python algo traders (TradeAlgo 2026; Alpaca docs 2025–2026;
  mindstudio.ai 2026 ADK trading-agent guide).
- **Covers:** US equities + crypto. Good for the S&P/FinMultiTime thesis side and the
  crypto TODO track.
- **Setup:** see `alpaca_setup.md`.

Sources:
- https://alpaca.markets/
- https://www.tradealgo.com/trading-guides/tools/best-broker-apis-for-algorithmic-trading-in-2026
- https://wdenniss.com/ap/1-agent/1.1-create-an-adk-agent/

## 2. Binance Spot Testnet (implemented — crypto track)

- **Testnet base:** `https://testnet.binance.vision` (signed trading endpoints)
- **Market-data base:** `https://data-api.binance.vision` (public, no key needed)
- **Auth:** HMAC-SHA256 signature over query string; header `X-MBX-APIKEY`.
- **Why:** Free, instant testnet keys via GitHub login, real matching engine, no SDK
  dependency (raw `requests`). Matches the TODO "crypto — DRW dataset, 15m–1h" track.
- **Verified live:** `GET /api/v3/klines` returns real BTCUSDT 1m candles from the
  testnet without credentials (see smoke test in `docs/usage.md`).
- **Setup:** see `binance_setup.md`.

Sources:
- https://developers.binance.com/en/docs/products/spot/testnet/rest-api
- https://developers.binance.com/en/docs/products/spot/testnet/general-info
- https://github.com/binance/binance-signature-examples/blob/master/python/spot/spot.py
- https://www.binance.com/en/support/faq/detail/9be58f73e5e14338809e3b705b9687dd

## 3. Interactive Brokers (documented — professional, not implemented)

- **Env:** IBKR paper account (separate account number flagged "P"); TWS or IB Gateway
  must be running and API socket enabled.
- **SDK:** `ib_async` (modern successor to `ib_insync`); sync/async wrapper over TWS API.
- **Why not primary:** Requires a desktop client (TWS/IB Gateway) running; heavier
  operational footprint. Best when you need global multi-asset + options market data.
- **Path to implement:** add `IBKRBroker` implementing `BrokerClient`; map
  `reqMktData`/`placeOrder` to the protocol. Paper account isolates risk.

Sources:
- https://github.com/ib-api-reloaded/ib_async
- https://www.interactivebrokers.com/campus/ibkr-quant-news/the-new-synchronous-wrapper-for-tws-api/
- https://www.interactivebrokers.com/en/trading/ib-api.php

## 4. MetaTrader 5 (documented — B3 / Brazil track)

- **Env:** MT5 demo account from a B3 broker (XP, Clear, Rico, BTG Pactual, Nelogica).
- **SDK:** `MetaTrader5` Python package.
- **Why:** Direct path to the project's B3 / NuBank goal (TODO: "Build model for B3
  (1m timeframe) for NuBank"). B3 brokers provide demo accounts (arxiv 2101.08169 cites
  XP/Clear as providing demo accounts).
- **Constraint:** MT5 terminal must run (Windows native, or Wine on Linux). Not headless.
- **Path to implement:** `MT5Broker` wrapping `mt5.symbol_info_tick`,
  `mt5.copy_rates_from_pos`, `mt5.order_send`. Demo account = paper.

Sources:
- https://arxiv.org/pdf/2101.08169 (B3/XP/Clear demo accounts)
- https://www.ontick.com.br/ (XP/Rico/Clear automation)
- https://www.btgpactual.com / https://www.nelogica.com.br/

## 5. Tradier & QuantConnect LEAN (not implemented)

- **Tradier:** REST sandbox, token auth, US stocks/options. Easy to add as
  `TradierBroker` if needed.
- **QuantConnect LEAN:** `lean-cli --brokerage "Paper Trading"` runs the LEAN engine
  locally with multiple brokerages. Heavier; a full alternative runtime, not just a
  broker. Useful for cross-validation of backtest vs paper.

Sources:
- https://github.com/QuantConnect/lean-cli

---

## Decision

- **Default real adapter: `AlpacaBroker`** — free, headless, official SDK, stocks+crypto.
- **Crypto track: `BinanceBroker`** — testnet, no SDK, verified working.
- **Offline / CI / replay: `SimulatedBroker`** — replays historical bars through the
  same engine loop, no network.
- **B3/NuBank future work: `MT5Broker`** — documented path, blocked on MT5 terminal.
