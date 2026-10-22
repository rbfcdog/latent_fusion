#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion.llama_news_backtest import run_llama_backtest

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "data" / "llama_news_api"
PRICES_PATH = OUT_BASE / "prices_2024_2026.parquet"
NEWS_PATH = ROOT / "data" / "live_news" / "capture.jsonl"

MODELS = {
    "70b": {
        "backend": "openrouter", "model": "meta-llama/llama-3.1-70b-instruct",
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
        "tag": "openrouter-deepinfra-llama-3.1-70b", "temperature": 0,
    },
    "8b": {
        "backend": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct",
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
        "tag": "openrouter-deepinfra-llama-3.1-8b", "temperature": 0,
    },
    "luna": {
        "backend": "openai", "model": "gpt-6-luna", "provider": None,
        "tag": "api-gpt-6-luna", "temperature": None,
    },
}


def _client(spec):
    import os
    from openai import OpenAI

    if spec["backend"] == "openrouter":
        key = os.environ.get("OPENROUTER_INFERENCE_KEY") or os.environ["OPENROUTER_API_KEY"]
        return OpenAI(base_url="https://openrouter.ai/api/v1",
                      api_key=key, timeout=180, max_retries=0)
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180, max_retries=0)


def _generate(spec):
    client = _client(spec)

    def generate(messages):
        started = time.time()
        kwargs = {"model": spec["model"], "messages": messages, "max_tokens": 1024}
        if spec["temperature"] is not None:
            kwargs["temperature"] = spec["temperature"]
        if spec["provider"] is not None:
            kwargs["extra_body"] = {"provider": spec["provider"]}
        last_error = None
        for attempt in range(5):
            try:
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                return text, {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "served_model": getattr(response, "model", None) or spec["model"],
                    "elapsed_ms": int((time.time() - started) * 1000),
                }
            except Exception as exc:
                last_error = exc
                time.sleep(min(60, 5 * (2 ** attempt)))
        raise RuntimeError(f"chamada falhou após retries: {last_error}")

    return generate


def _pricing_snapshot(spec):
    if spec["backend"] != "openrouter":
        return {"model": spec["model"], "endpoints": "preço não consultável"}
    try:
        data = requests.get(
            f"https://openrouter.ai/api/v1/models/{spec['model']}/endpoints", timeout=30).json()
        wanted = [e for e in data["data"]["endpoints"]
                  if e.get("provider_name", "").lower() == "deepinfra"]
        return {"model": spec["model"],
                "endpoints": [{k: e.get(k) for k in ("provider_name", "tag", "quantization",
                                                     "status", "pricing")} for e in wanted]}
    except Exception as exc:
        return {"model": spec["model"], "error": str(exc)}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke(spec):
    generate = _generate(spec)
    text, usage = generate([{"role": "user",
                             "content": 'Reply with exactly: {"actions":{},"evidence":{}}'}])
    print(json.dumps({"tag": spec["tag"], "served": usage["served_model"],
                      "elapsed_ms": usage["elapsed_ms"], "tokens": usage["prompt_tokens"],
                      "preview": text[:120]}, ensure_ascii=False))


def run(spec, out_dir):
    prices = pd.read_parquet(PRICES_PATH)
    tickers = list(prices.columns)
    out = Path(out_dir)
    config_path = out / "run_config.json"
    config = {"model_tag": spec["tag"], "model": spec["model"], "backend": spec["backend"],
              "provider": spec["provider"], "temperature": spec["temperature"],
              "prices_sha256": _sha256(PRICES_PATH), "news_sha256": _sha256(NEWS_PATH),
              "start": "2024-08-01", "end": str(prices.index[-1].date()), "every": 5,
              "cost_bps": 10, "max_headlines": 8,
              "pricing": _pricing_snapshot(spec)}
    if config_path.is_file():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in saved.items() if k != "pricing"}
        if comparable != {k: v for k, v in config.items() if k != "pricing"}:
            raise ValueError(f"config incompatível em {out}; use diretório novo")
    else:
        out.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    result = run_llama_backtest(prices, NEWS_PATH, out, tickers=tickers,
                                model_tag=spec["tag"], generate_fn=_generate(spec),
                                start=config["start"], end=config["end"], every=5,
                                cost_bps=10, max_headlines=8)
    records = result["records"]
    audit = records.groupby("arm").agg(
        decisões=("decision_date", "count"), json_ok=("parsed", "mean"),
        buys=("buy", "sum"), sells=("sell", "sum"))
    summary = {"summary": result["summary"], "usage_total": result["usage_total"],
               "audit": audit.to_dict(), "pricing": config["pricing"]}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"tag": spec["tag"], "net_news": result["summary"]["news"]["total_return"],
                      "net_no_news": result["summary"]["no_news"]["total_return"],
                      "usage": result["usage_total"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Run news backtest via hosted APIs")
    parser.add_argument("--models", nargs="+", default=["70b"], choices=list(MODELS))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    if not PRICES_PATH.is_file():
        source = Path("/tmp/llama_colab_review/prices.parquet")
        if not source.is_file():
            raise SystemExit(f"preços ausentes: {source}")
        PRICES_PATH.write_bytes(source.read_bytes())
    for name in args.models:
        spec = MODELS[name]
        smoke(spec)
        if not args.smoke_only:
            run(spec, OUT_BASE / spec["tag"])


if __name__ == "__main__":
    main()
