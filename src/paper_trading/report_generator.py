#!/usr/bin/env python3
import json
import os
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

LOG_DIR = _root / "outputs/logs"
RESULTS_DIR = _root / "paper-trading-results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = {
    "ml_hft": "ML HFT (SGDRegressor Online)",
    "ml_ensemble": "ML Ensemble (SGDClassifier LogLoss)",
}

def load_perf(strategy):
    path = LOG_DIR / f"perf_{strategy}.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return None
    return pd.DataFrame(rows)

def load_trades(strategy):
    path = LOG_DIR / f"trades_{strategy}.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return None
    return pd.DataFrame(rows)

def compute_stats(strategy, label, perf_df, trades_df):
    stats = {"strategy": strategy, "label": label}

    if perf_df is None or len(perf_df) == 0:
        stats["status"] = "sem dados"
        return stats

    stats["status"] = "ok"
    stats["snapshots"] = len(perf_df)
    stats["first_ts"] = perf_df["ts"].iloc[0]
    stats["last_ts"] = perf_df["ts"].iloc[-1]

    equity = perf_df["equity"].values
    stats["equity_start"] = round(float(equity[0]), 2)
    stats["equity_current"] = round(float(equity[-1]), 2)
    stats["equity_min"] = round(float(np.min(equity)), 2)
    stats["equity_max"] = round(float(np.max(equity)), 2)
    stats["return_pct"] = round(float(((equity[-1] - equity[0]) / equity[0]) * 100), 4)

    steps = perf_df["steps"].values
    stats["total_steps"] = int(steps[-1])

    if trades_df is not None and len(trades_df) > 0:
        buys = trades_df[trades_df["side"] == "BUY"]
        sells = trades_df[trades_df["side"] == "SELL"]
        stats["n_buys"] = len(buys)
        stats["n_sells"] = len(sells)
        stats["n_trades"] = len(trades_df)
        stats["symbols_traded"] = sorted(trades_df["symbol"].unique().tolist())
    else:
        stats["n_buys"] = 0
        stats["n_sells"] = 0
        stats["n_trades"] = 0
        stats["symbols_traded"] = []

    if len(equity) >= 3:
        daily_rets = np.diff(equity) / equity[:-1]
        stats["volatility_pct"] = round(float(np.std(daily_rets) * 100), 4)
        sharpe = np.mean(daily_rets) / (np.std(daily_rets) + 1e-12)
        stats["sharpe"] = round(float(sharpe * np.sqrt(252 * 390)), 4)
    else:
        stats["volatility_pct"] = 0
        stats["sharpe"] = 0

    return stats

def generate_report():
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d_%H-%M")
    lines = []
    lines.append(f"# Paper Trading Report — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Estratégias Ativas")
    lines.append("")
    lines.append("| Estratégia | Modelo | Símbolos |")
    lines.append("|---|---|---|")
    lines.append("| ML HFT | SGDRegressor (Huber, online) | AAPL, MSFT, NVDA, COST, GOOGL, ADBE |")
    lines.append("| ML Ensemble | SGDClassifier (LogLoss, online) | AAPL, MSFT, NVDA, GOOGL, AMZN, META |")
    lines.append("")
    lines.append("## Performance")
    lines.append("")

    has_data = False
    all_stats = []

    for strat, label in STRATEGIES.items():
        perf_df = load_perf(strat)
        trades_df = load_trades(strat)
        stats = compute_stats(strat, label, perf_df, trades_df)
        all_stats.append(stats)

        lines.append(f"### {label}")
        lines.append("")

        if stats["status"] != "ok":
            lines.append(f"> ⏳ **Sem dados ainda.** Estratégia rodando, aguardando mercado abrir (segunda 9:30 ET).")
            lines.append("")
            continue

        has_data = True
        r = stats["return_pct"]
        emoji = "🟢" if r > 0 else "🔴" if r < 0 else "⚪"

        lines.append(f"| Métrica | Valor |")
        lines.append(f"|---|---|")
        lines.append(f"| Snapshots | {stats['snapshots']} |")
        lines.append(f"| Período | {stats['first_ts'][:19]} → {stats['last_ts'][:19]} |")
        lines.append(f"| Equity inicial | ${stats['equity_start']:,.2f} |")
        lines.append(f"| Equity atual | ${stats['equity_current']:,.2f} |")
        lines.append(f"| Equity máx | ${stats['equity_max']:,.2f} |")
        lines.append(f"| Equity mín | ${stats['equity_min']:,.2f} |")
        lines.append(f"| Retorno | {emoji} **{r:+.4f}%** |")
        lines.append(f"| Sharpe (anualizado) | {stats['sharpe']:+.4f} |")
        lines.append(f"| Volatilidade (snapshot) | {stats['volatility_pct']:.4f}% |")
        lines.append(f"| Total steps | {stats['total_steps']:,} |")
        lines.append(f"| Trades (buy/sell) | {stats['n_buys']} buys / {stats['n_sells']} sells |")
        lines.append(f"| Símbolos tradados | {', '.join(stats['symbols_traded']) if stats['symbols_traded'] else 'nenhum'} |")
        lines.append("")

    if has_data:
        lines.append("## AI Analysis")
        lines.append("")
        analysis = generate_ai_analysis(all_stats)
        lines.append(analysis)
    else:
        lines.append("## Notas")
        lines.append("")
        lines.append("Serviços `ml-hft` e `ml-ensemble` estão rodando via systemd com `Restart=always`.")
        lines.append("Ambos aguardam abertura do mercado (segunda-feira 9:30 AM ET) para começar a tradar.")
        lines.append("")
        lines.append("```bash")
        lines.append("# Verificar status dos serviços")
        lines.append("systemctl --user status ml-hft ml-ensemble")
        lines.append("")
        lines.append("# Ver logs em tempo real")
        lines.append("tail -f outputs/logs/ml_hft.log")
        lines.append("tail -f outputs/logs/ml_ensemble.log")
        lines.append("```")

    lines.append("")
    lines.append("---")
    lines.append(f"*Gerado automaticamente em {now.strftime('%Y-%m-%d %H:%M:%S UTC')}*")

    report_path = RESULTS_DIR / f"{ts}.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    latest_path = RESULTS_DIR / "latest.md"
    with open(latest_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return report_path

def generate_ai_analysis(all_stats):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("[openai"):
        return "_AI analysis skipped — OpenAI API key not configured._"

    stats_text = ""
    for s in all_stats:
        if s["status"] != "ok":
            continue
        stats_text += f"""
### {s['label']}
- Return: {s['return_pct']:+.4f}%
- Sharpe: {s['sharpe']:+.4f}
- Volatility: {s['volatility_pct']:.4f}%
- Trades: {s['n_trades']} ({s['n_buys']} buys, {s['n_sells']} sells)
- Symbols traded: {', '.join(s['symbols_traded']) if s['symbols_traded'] else 'none'}
- Snapshots: {s['snapshots']} over {s['total_steps']} steps
"""

    prompt = f"""You are a quantitative trading analyst reviewing two ML paper-trading strategies running on Alpaca markets.

{stats_text}

Write a concise markdown analysis (200-400 words) covering:
1. Overall assessment — are the strategies performing well?
2. Risk assessment — volatility, drawdown signals, concentration risk
3. Which symbols are driving performance (or dragging it down)
4. One actionable recommendation for next steps
5. Rate each strategy: ⭐ (poor) to ⭐⭐⭐⭐⭐ (excellent)

Be honest. If returns are negative, say so. Use Portuguese (Brazilian). Keep it technical and direct."""

    try:
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 800,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"_AI analysis failed: HTTP {resp.status_code}_"
    except Exception as e:
        return f"_AI analysis failed: {e}_"

if __name__ == "__main__":
    report_path = generate_report()
    print(f"Report saved: {report_path}")
