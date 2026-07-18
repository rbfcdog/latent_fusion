# Binance Spot Testnet Setup

`BinanceBroker` talks to the Binance **Spot Testnet** — a real matching engine with
virtual funds. No SDK dependency (raw `requests` + HMAC-SHA256).

## 1. Get testnet API keys

1. Go to https://testnet.binance.vision/
2. Log in with **GitHub** (Binance authorizes the `binance-exchange` OAuth app).
3. Click "Generate HMAC_SHA256 Key" → receive `API Key` + `Secret Key`.

These keys are **testnet-only**. They do not work on the real Binance API.

Store in `.env`:

```env
BINANCE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
BINANCE_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 2. Market data works without keys

Public endpoints (`/api/v3/klines`, `/api/v3/ping`, `/api/v3/ticker/price`) are served
from `https://data-api.binance.vision` and need **no auth**. Verified live:

```bash
curl -sS "https://testnet.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=2"
# [[1784261160000,"63511.99","63512.00","63489.31","63497.62",...], ...]
```

So `SimulatedBroker` can replay testnet bars with zero credentials — useful for CI.

## 3. Run

```bash
uv run python -m src.paper_trading \
  --broker binance --strategy router \
  --symbol BTCUSDT --timeframe 1m \
  --poll-seconds 60 --verbose
```

Without keys, the engine will fetch bars and compute signals but raise on
`submit_market_order`. With keys, it places real testnet orders.

## Endpoints (`binance.py`)

| Protocol method | Binance endpoint |
|---|---|
| `get_account` | `GET /api/v3/account` (signed) |
| `get_position` | `GET /api/v3/account` → filter balances (signed) |
| `get_latest_bar` / `get_bars` | `GET /api/v3/klines` (public, `data-api.binance.vision`) |
| `get_market_status` | `GET /api/v3/ping` |
| `submit_market_order` | `POST /api/v3/order` type=MARKET (signed) |
| `submit_limit_order` | `POST /api/v3/order` type=LIMIT GTC (signed) |
| `cancel_order` | `DELETE /api/v3/order` (signed) |
| `get_order` | `GET /api/v3/order` (signed) |
| `close_all` | reads position, `POST /api/v3/order` MARKET opposite (signed) |

## Auth

HMAC-SHA256 over the URL-encoded query string (params + `timestamp` + `recvWindow`),
sent as the `signature` query param. API key in the `X-MBX-APIKEY` header. Implemented
in `BinanceBroker._sign`.

## Notes

- **Symbol format:** `BTCUSDT`, `ETHUSDT`, etc. (Binance quote convention).
- **Quantity precision:** the adapter formats qty to 6 decimals, price to 8 decimals.
  Binance enforces per-symbol `LOT_SIZE` / `PRICE_FILTER` step/precision; if a real
  order is rejected for precision, query `GET /api/v3/exchangeInfo` and round to the
  symbol's `stepSize`.
- **Testnet parity:** testnet mirrors live endpoints but market data on testnet is
  synthetic-ish (real-ish price feed, virtual balances). Quote IDs and order behavior
  are representative, not identical to live.
- **`--live` flag:** switches base to `https://api.binance.com` — real money. Default
  is paper (`testnet.binance.vision`).

## Sources

- https://developers.binance.com/en/docs/products/spot/testnet/rest-api
- https://developers.binance.com/en/docs/products/spot/testnet/general-info
- https://github.com/binance/binance-signature-examples/blob/master/python/spot/spot.py
- https://www.binance.com/en/support/faq/detail/9be58f73e5e14338809e3b705b9687dd
