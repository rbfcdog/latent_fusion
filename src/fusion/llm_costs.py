"""Offline agent cost audit; never re-generates an LLM decision."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.fusion.llm_context_experiment import _sha256

PRICE_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICE_AS_OF = "2026-10-21"


def _price_usage(api):
    usage = api.get("usage") if isinstance(api, dict) else None
    if not isinstance(usage, dict):
        return {"status": "usage_unavailable", "input_tokens": None, "output_tokens": None,
                "cached_tokens": None, "reasoning_tokens": None, "list_usd": None}
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details") or {}
    output_details = usage.get("completion_tokens_details") or {}
    cached = details.get("cached_tokens")
    reasoning = output_details.get("reasoning_tokens")
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (prompt, completion))
            or (cached is not None and (isinstance(cached, bool) or not isinstance(cached, int)
                                        or not 0 <= cached <= prompt))):
        raise ValueError("Contagem de tokens inválida na resposta da API")
    row = {"input_tokens": prompt, "output_tokens": completion, "cached_tokens": cached,
           "reasoning_tokens": reasoning, "list_usd": None}
    if api.get("response_model") != "gpt-6-luna" or api.get("service_tier") not in (None, "default"):
        row["status"] = "price_unavailable_for_model_or_tier"
    elif details.get("cache_write_tokens"):
        row["status"] = "cache_write_duration_unavailable"
    elif cached is None:
        row["status"] = "cached_token_count_unavailable"
    else:
        input_price, output_price, cached_price = ((.10, .50, .01) if prompt <= 272_000
                                                   else (.20, .75, .02))
        row["list_usd"] = ((prompt - cached) * input_price + cached * cached_price
                           + completion * output_price) / 1_000_000
        row["status"] = "standard_list_estimate"
    return row


def write_cost_reports(panel_path, news_path, baseline_csv, out_dir, *, cost_bps=10):
    from src.fusion.llm_trading_agent import _apply, _backtest, _numeric_actions, _prepare, parse_actions

    out = Path(out_dir)
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    if (config.get("panel_sha256") != _sha256(panel_path) or
            config.get("news_sha256") != _sha256(news_path) or
            config.get("baseline_sha256") != _sha256(baseline_csv) or
            config.get("cost_bps") != cost_bps or
            not isinstance(config.get("cache"), str) or
            Path(config["cache"]).name != config["cache"]):
        raise ValueError("Configuração de custos incompatível com as fontes")
    panel, grouped, schedule, numeric = _prepare(panel_path, news_path, baseline_csv)
    cache_path = out / config["cache"]
    records = [json.loads(line) for line in cache_path.open(encoding="utf-8")]
    if len(records) != len(schedule) or config.get("decisions") != len(schedule):
        raise ValueError("Cache não cobre todas as decisões")
    held, actions = set(), {}
    for index, (date, saved) in enumerate(zip(schedule, records)):
        group = grouped[date]
        titles = {ticker: [article["title"] for article in group[ticker]["available_headlines"]]
                  for ticker in panel["tickers"]}
        if (saved.get("index") != index or not isinstance(saved.get("raw"), str) or
                saved.get("metadata", {}).get("model_id") != config.get("model_id") or
                saved.get("metadata", {}).get("panel_sha256") != config["panel_sha256"]):
            raise ValueError("Resposta salva incompatível com o modelo/painel")
        parsed = parse_actions(saved["raw"], panel["tickers"], held, titles)
        if parsed != saved.get("parsed"):
            raise ValueError("Decisão salva alterada")
        actions[date] = parsed["actions"]
        held = _apply(held, parsed["actions"])
    numeric_actions = _numeric_actions(panel, schedule, numeric)
    rows = {}
    for name, decisions in (("llm_agent", actions), ("numeric_agent", numeric_actions)):
        gross = _backtest(panel, schedule, decisions, 0)
        net = _backtest(panel, schedule, decisions, cost_bps)
        rows[name] = {"gross_return_pct": (gross.strategy.iloc[-1] - 1) * 100,
                      "net_return_pct": (net.strategy.iloc[-1] - 1) * 100,
                      "trading_cost_drag_pp": (gross.strategy.iloc[-1] - net.strategy.iloc[-1]) * 100,
                      "turnover_total": float(net.turnover.sum()),
                      "buys": sum(action == "BUY" for day in decisions.values() for action in day.values()),
                      "sells": sum(action == "SELL" for day in decisions.values() for action in day.values())}
        if name == "llm_agent":
            bh_gross, bh_net = gross.bh.iloc[-1], net.bh.iloc[-1]
            rows["buy_hold"] = {"gross_return_pct": (bh_gross - 1) * 100,
                                "net_return_pct": (bh_net - 1) * 100,
                                "trading_cost_drag_pp": (bh_gross - bh_net) * 100,
                                "turnover_total": float(net.bh_turnover.sum()),
                                "buys": len(panel["tickers"]), "sells": 0}
    trade = pd.DataFrame.from_dict(rows, orient="index")
    trade.to_csv(out / "trading_cost_comparison.csv", index_label="strategy")

    attempts_path = out / config["cache"].replace("actions_", "api_attempts_", 1)
    attempts = [json.loads(line) for line in attempts_path.open(encoding="utf-8")] if attempts_path.is_file() else []
    usage_rows = []
    attempt_ids = set()
    for attempt in attempts:
        index = attempt.get("index")
        if (not isinstance(index, int) or not 0 <= index < len(schedule) or
                attempt.get("date") != schedule[index].date().isoformat() or
                attempt.get("prompt_sha256") != records[index]["prompt_sha256"]):
            raise ValueError("Tentativa de API desalinhada com as decisões")
        api = attempt.get("api")
        response_id = api.get("response_id") if isinstance(api, dict) else None
        if response_id is not None and (index, response_id) in attempt_ids:
            raise ValueError("Resposta de API duplicada no histórico de custos")
        attempt_ids.add((index, response_id))
        usage_rows.append({"date": attempt["date"], "decision_index": index,
                           "response_id": response_id,
                           "response_model": api.get("response_model") if isinstance(api, dict) else None,
                           **_price_usage(api)})
    for index, saved in enumerate(records):
        api = saved.get("api")
        response_id = api.get("response_id") if isinstance(api, dict) else None
        if (index, response_id) not in attempt_ids:
            usage_rows.append({"date": schedule[index].date().isoformat(), "decision_index": index,
                               "response_id": response_id,
                               "response_model": api.get("response_model") if isinstance(api, dict) else None,
                               **_price_usage(api)})
    ledger = pd.DataFrame(usage_rows)
    ledger.to_csv(out / "api_usage_ledger.csv", index=False)
    known = ledger.loc[ledger.status.eq("standard_list_estimate"), "list_usd"].sum()
    unknown = int((~ledger.status.eq("standard_list_estimate")).sum())
    summary = {"model_id": config["model_id"], "decisions": len(schedule),
               "api_responses_or_attempts": len(ledger), "priced_responses": len(ledger) - unknown,
               "unpriced_responses": unknown,
               "input_tokens_observed": int(ledger.input_tokens.dropna().sum()),
               "output_tokens_observed": int(ledger.output_tokens.dropna().sum()),
               "known_standard_list_usd": float(known),
               "total_standard_list_usd": float(known) if not unknown else None,
               "pricing_source": PRICE_SOURCE, "pricing_as_of": PRICE_AS_OF,
               "note": "Historical requests without API usage cannot be priced exactly. List prices are not the invoice. Fees in USD are separate from portfolio percent returns; no starting capital was supplied."}
    (out / "api_cost_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"trading": trade, "api_ledger": ledger, "api_summary": summary}
