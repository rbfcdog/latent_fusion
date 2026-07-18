# Alpaca Setup

Alpaca is the default real broker adapter (`AlpacaBroker`). Free paper trading, US
stocks + crypto, official `alpaca-py` SDK.

## 1. Create a paper account

1. Sign up at https://alpaca.markets/ (free).
2. The paper account is created automatically — separate from any live account.
3. Paper base URL: `https://paper-api.alpaca.markets`.

## 2. Get API keys

Dashboard → "Your API Keys" → generate a key/secret pair. Paper keys start with `PK...`.

Store them in `.env` (gitignored):

```env
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXX
ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The adapter reads `ALPACA_API_KEY` / `ALPACA_API_SECRET` from the environment.

## 3. Install

Already in `pyproject.toml`:

```bash
uv add alpaca-py      # already added: alpaca-py==0.43.5
```

## 4. Run

```bash
uv run python -m src.paper_trading \
  --broker alpaca --strategy sma \
  --symbol AAPL --timeframe 1m \
  --poll-seconds 60 --max-steps 100 --verbose
```

## Notes

- **Symbol format:** Alpaca uses tickers (`AAPL`, `MSFT`, `BTCUSD` for crypto). Not
  `BTCUSDT`.
- **Market hours:** US equity market is session-bound. The engine will still poll and
  fetch the latest bar outside hours, but order fills respect market status. Use crypto
  (`BTCUSD`) for 24/7.
- **Fractional shares:** Alpaca supports fractional qty by default on paper.
- **Rate limits:** 200 req/min for trading API; data API has separate limits. The
  default 60s poll keeps you well under.
- **`--live` flag:** omits it for paper. `AlpacaBroker(paper=True)` (default) points at
  the paper URL. Passing `--live` sets `paper=False` → live URL + real money. Never use
  `--live` unless you intend to trade real capital.

## Adapter mapping (`alpaca.py`)

| Protocol method | Alpaca SDK call |
|---|---|
| `get_account` | `TradingClient.get_account()` |
| `get_position` | `TradingClient.get_open_position(symbol)` |
| `get_latest_bar` / `get_bars` | `StockHistoricalDataClient.get_stock_bars(StockBarsRequest)` |
| `get_market_status` | `TradingClient.get_clock()` |
| `submit_market_order` | `TradingClient.submit_order(MarketOrderRequest)` |
| `submit_limit_order` | `TradingClient.submit_order(LimitOrderRequest)` |
| `cancel_order` | `TradingClient.cancel_order_by_id(id)` |
| `close_all` | `TradingClient.close_all_positions(cancel_orders=True)` |

Timeframes are mapped (`1m`→`TimeFrame.Minute`, `1h`→`TimeFrame.Hour`, `1d`→`TimeFrame.Day`).
