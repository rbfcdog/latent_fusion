"""Llama Tool Agent with interactive quantitative tools for portfolio management.

Tools provided:
1. analyze_numeric(tickers): query 20d momentum, rank, vol, and distance from 20d EMA.
2. estimate_trade_cost(actions): estimate turnover and fee impact in bps before committing.
3. check_correlation_risk(tickers): evaluate 20d trailing pairwise correlation matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.llama_news_backtest import (
    ACTION_VALUES, _apply, _clamp_actions, _target_weights,
    buy_and_hold, decision_dates, load_news, portfolio_metrics, simulate
)

SYSTEM_PROMPT = (
    "You are an elite quantitative portfolio manager managing a 16-asset portfolio after the decision-date close. "
    "Orders execute at the NEXT session close with transaction fees. "
    "You are given current holdings, cash status, and recent material corporate events. "
    "Before deciding, you have access to quantitative tools:\n"
    "- 'analyze_numeric': inspect trailing momentum rank, volatility, and EMA distance for candidate tickers.\n"
    "- 'check_correlation_risk': check pairwise correlation to avoid overconcentration in similar stocks.\n"
    "- 'estimate_trade_cost': preview the turnover and fee in basis points for your proposed trades.\n\n"
    "Use tools to verify numerical momentum and costs before trading. Avoid excessive turnover on weak evidence. "
    "When done investigating, output your final decision ONLY as JSON: "
    "{\"actions\": {\"TICKER\": \"BUY\"|\"SELL\"|\"HOLD\", ...}, \"reason\": \"summary of rationale\"}."
)

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_numeric",
            "description": "Query trailing 20-day momentum rank of 16, annualized volatility, distance from 20d EMA (z-score), and recent returns for candidate tickers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tickers to analyze from the basket."
                    }
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_correlation_risk",
            "description": "Calculate 20-day pairwise correlation matrix among candidate tickers to detect concentration risk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of candidate tickers to evaluate for correlation."
                    }
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_trade_cost",
            "description": "Preview turnover, transaction cost in basis points and percentage of equity for proposed actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "object",
                        "description": "Candidate actions mapping ticker to BUY, SELL, or HOLD.",
                        "additionalProperties": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]}
                    }
                },
                "required": ["actions"]
            }
        }
    }
]


def exec_analyze_numeric(prices, location, tickers, all_tickers):
    valid = [t for t in tickers if t in all_tickers]
    if not valid:
        return {"error": "Nenhum ticker válido fornecido"}
    # Compute 20d return across all tickers to establish global rank
    closes = prices.iloc[max(0, location - 20):location + 1]
    returns = closes.pct_change().dropna()
    m20 = {t: float(prices[t].iloc[location] / prices[t].iloc[max(0, location - 20)] - 1)
           for t in all_tickers}
    ranks = {t: r + 1 for r, t in enumerate(sorted(all_tickers, key=m20.get, reverse=True))}
    out = {}
    for t in valid:
        t_closes = closes[t]
        t_returns = returns[t]
        ema20 = float(t_closes.ewm(span=20, adjust=False).mean().iloc[-1])
        std20 = float(t_closes.std())
        z = round((float(t_closes.iloc[-1]) - ema20) / std20, 2) if std20 > 0 else 0.0
        out[t] = {
            "20d_momentum_pct": round(m20[t] * 100, 2),
            "rank_of_16": ranks[t],
            "vol_20d_annualized_pct": round(float(t_returns.std() * math.sqrt(252)) * 100, 2),
            "z_score_20d_ema": z,
            "last_5d_returns_pct": [round(float(v) * 100, 2) for v in t_returns.tail(5)]
        }
    return out


def exec_check_correlation_risk(prices, location, tickers, all_tickers):
    valid = [t for t in tickers if t in all_tickers]
    if len(valid) < 2:
        return {"error": "Necessário ao menos 2 tickers válidos para correlação"}
    window = prices[valid].iloc[max(0, location - 20):location + 1].pct_change().dropna()
    corr = window.corr()
    high_pairs = []
    pairs_corr = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            val = float(corr.iloc[i, j])
            if not math.isnan(val):
                pairs_corr.append(val)
                if val > 0.70:
                    high_pairs.append({"pair": f"{valid[i]}-{valid[j]}", "correlation": round(val, 2)})
    return {
        "mean_correlation": round(float(np.mean(pairs_corr)), 2) if pairs_corr else 0.0,
        "high_correlation_pairs": high_pairs,
        "warning": "Alta concentração detectada (>0.70)" if high_pairs else "Diversificação adequada"
    }


def exec_estimate_trade_cost(actions, held, all_tickers, cost_bps, current_weights):
    target_held = _apply(held, actions)
    target = _target_weights(target_held, all_tickers)
    turnover = float(np.abs(target - current_weights).sum())
    fee_bps = cost_bps * turnover
    return {
        "current_held": sorted(held),
        "target_held": sorted(target_held),
        "turnover_weight": round(turnover, 4),
        "estimated_fee_bps": round(fee_bps, 2),
        "estimated_fee_pct_of_equity": round(fee_bps / 100, 4),
        "recommendation": "Alerta: giro alto (>1.0)" if turnover > 1.0 else "Giro aceitável"
    }


def parse_agent_actions(text, tickers, held):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {t: "HOLD" for t in tickers}, "", False
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {t: "HOLD" for t in tickers}, "", False
    if not isinstance(obj, dict) or not isinstance(obj.get("actions"), dict):
        return {t: "HOLD" for t in tickers}, "", False
    reason = obj.get("reason", "")
    return _clamp_actions(obj["actions"], tickers, held), str(reason), True


def run_single_decision_agent(client, spec, context, prices, location, held, all_tickers,
                              cost_bps, current_weights, max_turns=4):
    lines = [
        f"Decision date: {context['decision_date']} (after close). Execution at next close: {context['execution_date']}.",
        f"Transaction fee: {cost_bps} bps of traded weight.",
        f"Currently holding: {', '.join(sorted(held)) or 'ALL CASH (nothing)'}.",
        "Available material corporate events since last decision:"
    ]
    for t in all_tickers:
        headlines = context["assets"][t]["available_headlines"]
        if headlines:
            lines.append(f"\n[{t}] {'(HELD)' if t in held else ''}")
            for h in headlines:
                lines.append(f"  * [{h['published_at'][:10]} {h['source']}] {h['title']}")
        else:
            lines.append(f"[{t}] (No material news)")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)}
    ]
    tool_log = []
    total_p_tok, total_c_tok, total_ms = 0, 0, 0
    final_text = ""

    for turn in range(max_turns):
        kwargs = {
            "model": spec["model"], "messages": messages, "tools": TOOLS_SCHEMA,
            "tool_choice": "auto", "max_tokens": 1024
        }
        if spec.get("temperature") is not None:
            kwargs["temperature"] = spec["temperature"]
        if spec.get("provider") is not None:
            kwargs["extra_body"] = {"provider": spec["provider"]}

        started = time.time()
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                time.sleep(min(60, 4 * (2 ** attempt)))
        else:
            raise RuntimeError("API falhou após 5 tentativas no turno da ferramenta")

        ms = int((time.time() - started) * 1000)
        total_ms += ms
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        if usage:
            total_p_tok += getattr(usage, "prompt_tokens", 0) or 0
            total_c_tok += getattr(usage, "completion_tokens", 0) or 0

        msg = choice.message
        # Check if model made tool calls
        if msg.tool_calls:
            # Build assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    } for tc in msg.tool_calls
                ]
            })
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    fn_args = {}
                if fn_name == "analyze_numeric":
                    res = exec_analyze_numeric(prices, location, fn_args.get("tickers", []), all_tickers)
                elif fn_name == "check_correlation_risk":
                    res = exec_check_correlation_risk(prices, location, fn_args.get("tickers", []), all_tickers)
                elif fn_name == "estimate_trade_cost":
                    res = exec_estimate_trade_cost(fn_args.get("actions", {}), held, all_tickers, cost_bps, current_weights)
                else:
                    res = {"error": f"Ferramenta desconhecida: {fn_name}"}

                tool_log.append({"tool": fn_name, "args": fn_args, "res": res})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(res, ensure_ascii=False)
                })
        else:
            candidate_text = msg.content or ""
            _, _, ok = parse_agent_actions(candidate_text, all_tickers, held)
            if ok:
                final_text = candidate_text
                break
            # Model produced intermediate text without tools, prompt for final JSON
            messages.append({"role": "assistant", "content": candidate_text})
            messages.append({
                "role": "user",
                "content": "Now conclude your analysis and output your final portfolio decision JSON containing exactly 'actions' (BUY/SELL/HOLD for all 16 tickers) and 'reason'."
            })
    if not final_text:
        # One forced final call without tools
        kwargs_final = {
            "model": spec["model"], "messages": messages, "max_tokens": 1024
        }
        if spec.get("temperature") is not None:
            kwargs_final["temperature"] = spec["temperature"]
        if spec.get("provider") is not None:
            kwargs_final["extra_body"] = {"provider": spec["provider"]}
        started = time.time()
        resp_final = client.chat.completions.create(**kwargs_final)
        total_ms += int((time.time() - started) * 1000)
        final_text = resp_final.choices[0].message.content or ""
        usage_final = getattr(resp_final, "usage", None)
        if usage_final:
            total_p_tok += getattr(usage_final, "prompt_tokens", 0) or 0
            total_c_tok += getattr(usage_final, "completion_tokens", 0) or 0
    actions, reason, ok = parse_agent_actions(final_text, all_tickers, held)
    return {
        "actions": actions,
        "reason": reason,
        "parsed": ok,
        "raw": final_text,
        "tool_calls": tool_log,
        "usage": {
            "prompt_tokens": total_p_tok,
            "completion_tokens": total_c_tok,
            "elapsed_ms": total_ms,
            "turns": turn + 1
        }
    }
