#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
from openai import OpenAI
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from src.fusion.llama_tool_agent import run_single_decision_agent
from src.fusion.llama_news_backtest import (
    build_context, buy_and_hold, decision_dates, load_news, portfolio_metrics, simulate, _apply, _target_weights
)

BASE = ROOT / "data" / "llama_news_api"
PRICES_PATH = BASE / "prices_2024_2026.parquet"
NEWS_PATH = ROOT / "data" / "live_news" / "capture_filtered.jsonl"
OUT_DIR = BASE / "tool_agent_70b"

SPEC_70B = {
    "model": "meta-llama/llama-3.1-70b-instruct",
    "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
    "temperature": 0,
    "tag": "openrouter-deepinfra-llama-3.1-70b-tool-agent"
}


def run_tool_backtest(news_path=NEWS_PATH, out_dir=OUT_DIR, cost_bps=10):
    prices = pd.read_parquet(PRICES_PATH)
    tickers = list(prices.columns)
    news = load_news(news_path)
    dates = decision_dates(prices, "2024-08-01", str(prices.index[-1].date()), every=5)
    sessions = prices.index

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "decisions.jsonl"

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_INFERENCE_KEY") or os.environ["OPENROUTER_API_KEY"],
        timeout=180,
        max_retries=0
    )

    cached = {}
    if cache_path.is_file():
        with cache_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    cached[row["decision_date"]] = row
                except Exception:
                    continue

    held = set()
    current_weights = np.zeros(len(tickers))
    actions_by_date = {}
    records = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_tools_called = 0

    print(f"Iniciando Tool Agent com 70B em {len(dates)} datas de decisão...")
    with cache_path.open("a", encoding="utf-8") as cache:
        for idx, date in enumerate(dates):
            loc = sessions.get_loc(date)
            # Drift weights from last fill if any
            if idx > 0:
                prev_fill_loc = sessions.get_loc(dates[idx - 1]) + 1
                curr_fill_loc = loc + 1
                daily_slice = prices.pct_change().fillna(0.0).iloc[prev_fill_loc:curr_fill_loc]
                for s in range(len(daily_slice)):
                    r = daily_slice.iloc[s].to_numpy(dtype=float)
                    g = 1.0 + float(current_weights @ r)
                    current_weights = current_weights * (1 + r) / g if g > 0 else current_weights

            context = build_context(prices, news, date, tickers, "news", max_headlines=5,
                                    held=frozenset(held), cost_bps=cost_bps)
            prompt_sha = hashlib.sha256(json.dumps({
                "date": context["decision_date"], "held": sorted(held), "tag": SPEC_70B["tag"]
            }, sort_keys=True).encode()).hexdigest()

            row = cached.get(context["decision_date"])
            if row is None or row.get("prompt_sha256") != prompt_sha:
                res = run_single_decision_agent(
                    client, SPEC_70B, context, prices, loc, held, tickers, cost_bps, current_weights
                )
                row = {
                    "decision_date": context["decision_date"],
                    "execution_date": context["execution_date"],
                    "prompt_sha256": prompt_sha,
                    "actions": res["actions"],
                    "reason": res["reason"],
                    "parsed": res["parsed"],
                    "tool_calls": res["tool_calls"],
                    "usage": res["usage"]
                }
                cache.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache.flush()

            actions = row["actions"]
            actions_by_date[date] = actions
            t_calls = len(row.get("tool_calls", []))
            total_tools_called += t_calls
            usage = row.get("usage", {})
            total_tokens_in += usage.get("prompt_tokens", 0)
            total_tokens_out += usage.get("completion_tokens", 0)

            # Update target weights at fill
            held = _apply(held, actions)
            current_weights = _target_weights(held, tickers)
            records.append({
                "date": context["decision_date"],
                "parsed": row["parsed"],
                "tools_called": t_calls,
                "buys": sum(1 for v in actions.values() if v == "BUY"),
                "sells": sum(1 for v in actions.values() if v == "SELL"),
                "held_after": len(held)
            })

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"Decisão {idx+1}/{len(dates)} ({context['decision_date']}) | Tools: {t_calls} | Held: {len(held)} ações", flush=True)

    first_fill = sessions[sessions.get_loc(dates[0]) + 1]
    bench = buy_and_hold(prices, tickers, first_fill, cost_bps=cost_bps)
    equity = simulate(actions_by_date, dates, prices, tickers, cost_bps=cost_bps)
    metrics = portfolio_metrics(equity["net"], bench)

    summary = {
        "model": SPEC_70B["model"],
        "tag": SPEC_70B["tag"],
        "metrics": metrics,
        "total_tools_called": total_tools_called,
        "tokens": {"prompt_tokens": total_tokens_in, "completion_tokens": total_tokens_out},
        "records_count": len(records),
        "audit": pd.DataFrame(records).to_dict("records")
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"tag": SPEC_70B["tag"], "net_return": metrics["total_return"],
                      "alpha_annual": metrics["alpha_annual"], "max_drawdown": metrics["max_drawdown"],
                      "total_tools": total_tools_called}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-bps", type=int, default=10)
    args = parser.parse_args()
    run_tool_backtest(cost_bps=args.cost_bps)
