# Paper Trading — Deployment

Three ways to run the live paper-trading daemon persistently. See
`cloud_deployment.md` for the full cloud-host comparison and reliability playbook.

> "Live" here means **live paper trading**: real-time market data + virtual
> execution. It is NOT the `--live` real-money flag. Never pass `--live` unless you
> intend to trade real capital.

## 1. systemd (recommended for a VPS)

```bash
sudo useradd -r -s /usr/sbin/nologin paperbot
sudo mkdir -p /opt/latent_fusion /var/log/paper-trading
sudo chown paperbot:paperbot /var/log/paper-trading
# clone/copy the repo to /opt/latent_fusion, then:
sudo cp src/paper_trading/deploy/paper-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now paper-trading
sudo journalctl -u paper-trading -f     # or tail /var/log/paper-trading/out.log
```

`Restart=always` brings it back after crashes or reboots.

## 2. Docker Compose (recommended for portability)

```bash
cd src/paper_trading/deploy
cp ../../.env .env   # broker keys (optional for livepaper; required for alpaca/binance)
docker compose up -d --build
docker compose logs -f
docker compose down      # stop
```

`restart: unless-stopped` + a SQLite-based healthcheck. State persists in the
`paper-state` volume across rebuilds.

## 3. Quick local (tmux/screen — dev only, not robust)

```bash
tmux new -s paper
uv run python -m src.paper_trading --broker livepaper --strategy router \
  --symbol BTCUSDT --poll-seconds 60 --state-path paper_trading/state/live.sqlite --verbose
# Ctrl-b d to detach; tmux attach -t paper to reattach
```

## Run state

- **SQLite:** `paper_trading/state/live.sqlite` — orders, fills, snapshots (keyed by
  `run_id`). Back up this file; it is the trade blotter.
- **Logs:** stdout/stderr (systemd → `/var/log/paper-trading/`, docker → `docker logs`).
- **Inspect a run:**
  ```python
  from src.paper_trading import StateStore
  s = StateStore("paper_trading/state/live.sqlite")
  print(s.load_snapshots().tail())   # live equity/position history
  print(s.load_fills())              # every executed fill
  ```

## Switching brokers

Edit `ExecStart` / `command` / `CMD`:
- `--broker livepaper` (default, no keys, Binance testnet data + sim fills)
- `--broker binance` (real Binance testnet orders — needs `BINANCE_API_KEY/SECRET`)
- `--broker alpaca` (real Alpaca paper orders — needs `ALPACA_API_KEY/SECRET`)

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Container image (uv sync + source) |
| `docker-compose.yml` | Service with restart, volumes, healthcheck |
| `paper-trading.service` | systemd unit with hardening + restart |
| `cloud_deployment.md` | Cloud host comparison + reliability playbook |
