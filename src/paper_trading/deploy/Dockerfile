# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY backtest ./backtest
COPY models ./models
COPY features ./features
COPY README.md ./

RUN mkdir -p paper_trading/state outputs/logs

ENV PYTHONPATH=/app

CMD ["uv","run","python","-m","src.paper_trading","--broker","livepaper","--data-source","binance","--strategy","router","--symbol","BTCUSDT","--timeframe","1m","--bar-window","120","--poll-seconds","60","--initial-cash","10000","--fee-bps","4","--slippage-bps","1","--long-only","--state-path","paper_trading/state/live.sqlite","--log-level","INFO","--verbose"]
