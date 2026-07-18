# Cloud Deployment — Best Ways to Keep Paper Trading Running

How to run the `src.paper_trading` daemon continuously in the cloud. Goal: a
long-running process that polls live market data, executes the strategy, persists
state to SQLite, and survives crashes, reboots, and deploys — with monitoring.

> "Live" = **live paper trading** (real-time data + virtual money). The `--live`
> real-money flag is a separate, dangerous switch — do not use it for unattended runs.

---

## TL;DR — recommended setup

| Need | Recommendation |
|---|---|
| Cheapest reliable single host | **Hetzner CX22 (€3.79/mo)** + `systemd` unit |
| Portable / multi-host | **Docker Compose** on any VPS |
| Zero-ops PaaS | **Fly.io** with a persistent volume |
| Avoid | Serverless (Lambda/Cloud Run) — stateful long-poll + SQLite don't fit |

All options run the same command:
```
uv run python -m src.paper_trading --broker livepaper --data-source binance \
  --strategy router --symbol BTCUSDT --timeframe 1m --poll-seconds 60 \
  --state-path paper_trading/state/live.sqlite --verbose
```

---

## 1. VPS + systemd (best price/reliability)

Provision a small Linux VM, clone the repo, install `uv`, then install the unit in
`deploy/README.md`. Why systemd is the right tool:

- `Restart=always` + `RestartSec=10` → auto-recovers from crashes.
- `WantedBy=multi-user.target` → starts on boot.
- `EnvironmentFile=.env` → secrets out of the unit file.
- Built-in hardening (`NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp`).
- Journald + file logging; rotate with `logrotate`.

### Host comparison

| Provider | Smallest plan | Price | Notes |
|---|---|---|---|
| **Hetzner Cloud** | CX22 (2 vCPU, 4GB) | €3.79/mo | Best value, EU/US, generous bandwidth |
| **DigitalOcean** | Basic droplet (1 vCPU, 512MB) | $4/mo | Simple UI, NYC/ams/sgp |
| **Linode/Akamai** | Nanode 1GB | $5/mo | Good network, flat pricing |
| **OVHcloud** | VPS Starter | ~€3.50/mo | Cheapest, EU-focused |
| **AWS EC2** | t3.micro (free tier 12mo, then ~$8/mo) | $0–8/mo | Overkill; egress fees |
| **GCP Compute** | e2-micro (always-free tier, limited) | $0–7/mo | Free tier US-region only |
| **Azure** | B1ls | ~$4/mo | No advantage here |

**Pick Hetzner or DigitalOcean.** 512MB RAM is enough — the engine is a poll loop with
a 120-bar pandas window. Use 1GB+ if you later load `sentence-transformers` / MOMENT
embeddings on the same host.

---

## 2. Docker Compose (portable)

`deploy/docker-compose.yml` is ready:
```bash
cd src/paper_trading/deploy
cp ../../.env .env       # optional for livepaper
docker compose up -d --build
```
- `restart: unless-stopped` → survives host reboot (if Docker daemon enabled on boot).
- `paper-state` volume → SQLite survives `docker compose down` / image rebuilds.
- `healthcheck` queries the SQLite snapshot count; unhealthy containers restart.
- Log rotation via `json-file` `max-size: 10m, max-file: 5`.

Run on any VPS with Docker, or on Fly.io/Railway (they consume Compose-ish configs).

---

## 3. PaaS (zero-ops)

### Fly.io
```bash
fly launch --no-deploy          # generates fly.toml
fly volumes create paper_state --size 1
fly secrets set BINANCE_API_KEY=... BINANCE_API_SECRET=...   # if needed
fly deploy
```
`fly.toml` needs a persistent volume mount at `/app/paper_trading/state` and
`auto_stop_machines = false` (a trading bot must run 24/7). ~$2–5/mo for a 256MB
shared-cpu VM + 1GB volume.

### Railway / Render
Both support a "background worker" / persistent process with a volume. Set the start
command to the `uv run python -m src.paper_trading ...` line. ~$5/mo. Simpler than
Fly but pricier at scale.

### ❌ Serverless (Lambda, Cloud Run, Functions)
Not suitable:
- Max execution duration (Cloud Run 60min, Lambda 15min) < a 24/7 bot.
- No persistent disk → SQLite state lost each invocation.
- Cold starts break the poll cadence.
- You'd need external state (Redis/DynamoDB) — defeats the simple SQLite design.

---

## 4. Reliability playbook

| Concern | Fix |
|---|---|
| **Crash recovery** | systemd `Restart=always` / Docker `restart: unless-stopped` / Fly `auto_stop_machines=false` |
| **State durability** | SQLite on a persistent volume; **back up** `live.sqlite` daily (cron + S3/r2/Backblaze) |
| **Log growth** | `logrotate` (systemd), `json-file max-size/max-file` (Docker), or ship to Papertrail/Logtail |
| **Health monitoring** | Docker healthcheck (SQLite row count); systemd `OnFailure=` unit; external uptime probe on a tiny HTTP endpoint |
| **Alerts** | Add an error webhook (Telegram/ntfy/Slack) in the engine's `on_step`/exception path; or alert on log line `ERROR` |
| **Clock drift** | Ensure NTP (`systemd-timesyncd`) — poll cadence depends on accurate sleep |
| **Broker API downtime** | Engine already wraps each broker call in try/except and skips the step; it self-heals next poll |
| **Secret leakage** | `.env` gitignored (it is); use provider secret managers (Fly secrets, Docker `env_file`, systemd `EnvironmentFile`); never bake keys into the image |
| **Resource limits** | Docker `cpus: 0.5, memory: 512M`; systemd `MemoryMax=512M` |

### Backing up state
```bash
# cron, daily
cp /opt/latent_fusion/paper_trading/state/live.sqlite \
   /backup/live-$(date +%F).sqlite
# or offsite: rclone copy ... r2:paper-trading/
```

### Adding an alert webhook (optional, not yet implemented)
The engine's `run(on_step=...)` accepts a callback. A thin wrapper script can POST to
a Telegram bot on `logger.error`. Left as an extension point — do not add unless you
want alerts; the engine already logs all errors.

---

## 5. Recommended architecture

```
┌─────────────────────────────────────────────┐
│  VPS (Hetzner CX22 / DigitalOcean droplet)  │
│  ┌───────────────────────────────────────┐  │
│  │ systemd: paper-trading.service        │  │
│  │   uv run python -m src.paper_trading  │  │
│  │        --broker livepaper ...         │  │
│  │   Restart=always  RestartSec=10       │  │
│  └──────────────┬────────────────────────┘  │
│                 │ polls every 60s            │
│         ┌───────▼────────┐  ┌──────────────┐ │
│         │ SQLite state   │  │ logrotate    │ │
│         │ live.sqlite    │  │ out.log      │ │
│         └───────┬────────┘  └──────────────┘ │
│                 │ nightly cron backup         │
│         ┌───────▼────────┐                    │
│         │ /backup/*.sqlite│                   │
│         └────────────────┘                    │
└───────────────────────────────────────────────┘
            │ HTTPS
            ▼
   testnet.binance.vision  (market data)
   paper-api.alpaca.markets (if --broker alpaca)
```

External (optional): UptimeRobot probes a status endpoint; Logtail ingests logs;
rclone ships nightly SQLite backups to R2.

---

## 6. Already deployed in this environment

The daemon is running now as a detached, persisted process:
```
name: paper-trading
broker: livepaper (Binance testnet data + simulated fills)
strategy: RegimeRouterStrategy
symbol: BTCUSDT, timeframe 1m, poll 60s
state: paper_trading/state/live.sqlite
restart: on-failure
```
Inspect with: `hub logs paper-trading` / `hub ps`. State with `StateStore`.

For a real cloud move, the same command goes into `paper-trading.service` or the
Dockerfile `CMD` — no code changes.

---

## 7. Real-time inference patterns

Three patterns for running the strategy with live data, ordered by latency:

### 7.1 Polling (simplest, recommended for daily/weekly strategies)

```
┌────────────┐     poll every 60s      ┌──────────────┐
│  VPS cron  │ ──────────────────────► │  Exchange API │
│  + systemd │ ◄────────────────────── │  (Binance/    │
│            │     OHLCV + price       │   Alpaca)     │
│  Strategy  │                         └──────────────┘
│  Engine    │ ──────signal──────────► ┌──────────────┐
│            │                         │  SQLite state │
└────────────┘                         └──────────────┘
```

The `paper_trading` module already implements this. Key parameters:

| Parameter | Default | Effect |
|---|---|---|
| `--poll-seconds` | 60 | How often to check for new data |
| `--timeframe` | 1m | Candle size (1m, 5m, 15m, 1h, 1d) |
| `--strategy` | router | Strategy name (sma, meanrev, hmm, router) |

For daily/weekly strategies (R1 rebalance), set `--timeframe 1d --poll-seconds 3600`
and use `rebalance_freq="weekly"` in `BacktestConfig`.

### 7.2 WebSocket streaming (low-latency, for intraday)

```python
from src.paper_trading.binance import BinanceFeed

feed = BinanceFeed(symbol="BTCUSDT", timeframe="1m")
async for bar in feed.stream():
    signal = strategy.generate_signals(bar)
    if rebalance_day(bar.timestamp):
        engine.execute(signal)
```

Binance and Alpaca both offer WebSocket feeds. The engine processes each bar
as it arrives — no polling delay. Best for 1m–15m timeframes where 60s latency
matters.

### 7.3 Webhook (event-driven, for news-triggered strategies)

```
┌──────────┐   news event    ┌───────────────┐   signal   ┌──────────┐
│ News RSS │ ──────────────► │  VPS webhook  │ ─────────► │  Broker  │
│ /GDELT   │                 │  (Flask/Fast  │            │  API     │
└──────────┘                 │   API on :8080)│            └──────────┘
                              └───────────────┘
```

Set up a lightweight HTTP endpoint that receives news payloads, runs the
embedding model, and triggers a rebalance if the signal changes significantly.
Use with `rebalance_freq="daily"` and the text-embedding strategy.

```python
# webhook_server.py (run alongside paper_trading)
from fastapi import FastAPI
app = FastAPI()

@app.post("/news")
async def on_news(payload: dict):
    embedding = model.encode(payload["title"])
    signal = trained_strategy.predict(embedding)
    if abs(signal) > 0.3:
        state_store.queue_rebalance(payload["ticker"], signal)
    return {"status": "queued"}
```

### Choosing a pattern

| Strategy type | Pattern | Latency | Complexity |
|---|---|---|---|
| Daily/weekly rebalance | Polling (1h) | ~1h | Low |
| Intraday (15m–1h) | Polling (60s) | ~60s | Low |
| HFT (1m–5m) | WebSocket | <1s | Medium |
| News-triggered | Webhook | <10s | Medium |

---

## 8. Cost estimation

### 8.1 Infrastructure cost by scale

| Scale | Assets | VPS plan | Monthly cost | Components |
|---|---|---|---|---|
| **Hobby** | 1–5 | Hetzner CX22 (2vCPU, 4GB) | €3.79 (~$4) | Engine + SQLite |
| **Small** | 5–20 | Hetzner CX32 (4vCPU, 8GB) | €6.49 (~$7) | Engine + embeddings cache |
| **Medium** | 20–50 | Hetzner CX42 (8vCPU, 16GB) | €12.49 (~$13) | Engine + MOMENT + embeddings |
| **Large** | 50+ | Hetzner CPX31 (8vCPU, 16GB) | €15.49 (~$16) | + Redis state + webhook server |

### 8.2 Cost per asset monitored

| Component | Cost/asset/month | Notes |
|---|---|---|
| VPS (amortized) | $0.20–$0.80 | Depends on plan tier |
| Exchange API | $0 | Binance/Alpaca free tier |
| News (Google RSS) | $0 | Free |
| News (NewsAPI) | $0–$449 | Free=30d, paid=full archive |
| Embedding inference | $0 | Local `sentence-transformers` |
| MOMENT inference | $0 | Local PyTorch |
| SQLite storage | $0 | <1MB per asset/year |
| Backup (R2/S3) | $0.01 | ~$0.015/GB/month |

**Total per asset: ~$0.20–$0.80/month** (VPS-dominant).

### 8.3 Cost breakdown for 10-asset crypto portfolio

| Item | Monthly cost |
|---|---|
| Hetzner CX22 VPS | $4.00 |
| Domain (optional) | $1.00 |
| R2 backup (1GB) | $0.02 |
| UptimeRobot (free) | $0.00 |
| **Total** | **~$5.00/month** |

### 8.4 Adding embedding models

If running `sentence-transformers` (all-MiniLM-L6-v2) on the same VPS:
- Model size: ~90MB
- RAM: +500MB during inference
- CPU: ~50ms per headline on 2vCPU
- **No extra cost** — runs on the same CX22

If running MOMENT-1-large for time-series embeddings:
- Model size: ~300MB
- RAM: +2GB during inference
- **Upgrade to CX32 (8GB)** — adds ~$3/month

### 8.5 Scaling beyond 50 assets

| Bottleneck | Solution | Cost impact |
|---|---|---|
| CPU (embedding inference) | Batch inference every 5 min | $0 (same VPS) |
| RAM (MOMENT + 50 tickers) | Upgrade to CX42 | +$6/month |
| SQLite write contention | Migrate to PostgreSQL | +$5/month (managed) |
| API rate limits | Add 2nd VPS + load balancer | +$4/month |

**Rule of thumb**: $5/month covers 10 assets. $15/month covers 50 assets.
$30/month covers 100+ assets with embeddings.

---

## 9. Quick-start: deploy in 5 minutes

```bash
# 1. SSH into your VPS
ssh root@your-server

# 2. Clone and install
git clone https://github.com/youruser/latent_fusion.git
cd latent_fusion
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. Configure
cp .env.example .env
# Edit .env with your API keys (optional for paper trading)

# 4. Start the daemon
cd src/paper_trading/deploy
cp paper-trading.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable paper-trading
systemctl start paper-trading

# 5. Verify
systemctl status paper-trading
journalctl -u paper-trading -f
```

For Docker:
```bash
cd src/paper_trading/deploy
docker compose up -d --build
docker logs -f paper-trading
```
