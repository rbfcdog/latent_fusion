"""Colab-only, title-grounded Qwen facts with a restartable audited JSONL cache."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from src.fusion.event_data import ALIASES

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PROMPT_VERSION = "issuer-title-json-v1"
LLM_FEATURE_NAMES: tuple[str, ...] = (
    "subject", "object", "earnings", "guidance", "merger", "regulatory",
    "rating", "product", "macro", "positive", "negative", "confirmed",
    "expected", "rumor", "magnitude_pct",
)
_CLASSES = frozenset(LLM_FEATURE_NAMES[2:9]) | {"unknown"}
_ROLES = {"subject", "object", "unknown"}
_DIRECTIONS = {"positive", "negative", "unknown"}
_STATUSES = {"confirmed", "expected", "rumor", "unknown"}
_KEYS = {"role", "class", "direction", "status", "magnitude_pct", "evidence"}
_PERCENT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(?:%|percent\b)", re.IGNORECASE)
_EXPECTED = re.compile(r"\b(?:expects?|expected|anticipates?|forecast|forecasts|plans?|planned|to\s+(?:buy|acquire|launch)|may|might|could|would|possible|potential)\b", re.I)
_RUMOR = re.compile(r"\b(?:rumou?rs?|rumou?red|reportedly|unconfirmed|sources\s+say|in\s+talks)\b", re.I)
_POSITIVE = re.compile(r"\b(?:beats?|beat|tops?|topped|exceeds?|exceeded|rises?|rose|increases?|increased|grew|grows?|raises?|raised|upgrades?|upgraded|approved|gains?|gained|record\s+(?:profit|revenue|sales))\b", re.I)
_NEGATIVE = re.compile(r"\b(?:miss(?:es|ed)?|falls?|fell|declines?|declined|drops?|dropped|decreases?|decreased|cuts?|cut|lowers?|lowered|downgrades?|downgraded|fined|rejected|lawsuit|losses|lost)\b", re.I)
_STOCK_PRICE = re.compile(r"\b(?:stock|share|shares)\s+(?:price|prices|rises?|rose|falls?|fell|gains?|gained|drops?|dropped)\b|\b(?:stock|share|shares)\b.{0,20}\b(?:rises?|rose|falls?|fell|gains?|gained|drops?|dropped)\b", re.I)
_CLASS_CUES = {
    "earnings": re.compile(r"\b(?:earnings|eps|revenue|sales|profits?|income|financial\s+results)\b", re.I),
    "guidance": re.compile(r"\b(?:guidance|outlook|forecast)\b", re.I),
    "merger": re.compile(r"\b(?:acquir(?:es?|ed|ing)|acquisition|merger|takeover|buyout|buys|bought)\b", re.I),
    "regulatory": re.compile(r"\b(?:regulat(?:or|ory|ion)|antitrust|lawsuit|litigation|settlement|fda|sec|ftc|investigation|approved|fined)\b", re.I),
    "rating": re.compile(r"\b(?:analyst|rating|upgrade|upgraded|downgrade|downgraded)\b", re.I),
    "product": re.compile(r"\b(?:products?|launch(?:es|ed|ing)?|unveil(?:s|ed|ing)?|releas(?:es?|ed|ing)|recall(?:s|ed|ing)?)\b", re.I),
    "macro": re.compile(r"\b(?:inflation|interest\s+rates?|fed|economy|economic|recession|oil\s+prices?)\b", re.I),
}
_SYSTEM = (
    "Extract only facts literally stated in the JSON-escaped headline title. The tagged issuer "
    "and aliases are identity hints, NOT proof that the issuer is subject or object. Never use an "
    "article body, outside knowledge, later performance, predicted returns, stock price advice or "
    "buy/sell recommendations. For uncertain issuer attribution or ambiguous multi-entity events, "
    "abstain: role/class/direction/status=unknown, magnitude_pct=null, evidence=\"\". "
    "Return ONLY one JSON object with exactly these keys: role (subject/object/unknown), "
    "class (earnings/guidance/merger/regulatory/rating/product/macro/unknown), direction "
    "(positive/negative/unknown), status (confirmed/expected/rumor/unknown), "
    "magnitude_pct (nonnegative literal numeric percent in the evidence or null), evidence "
    "(an exact verbatim substring of the title mentioning the tagged issuer, or empty for abstention). "
    "Direction describes the reported issuer event, NOT an investment recommendation; do not "
    "infer direction from a bare acquisition, forecast, stock move, or percentage. "
    "Confirmed means a reported completed fact, expected means explicitly forecast/planned, "
    "rumor means explicitly unconfirmed. If multiple event interpretations conflict, abstain."
)


def _issuer_mentions(title: str, ticker: str) -> bool:
    aliases = ALIASES.get(ticker, ())
    if any(re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", title, re.I)
           for alias in aliases):
        return True
    qualified = r"(?:\$|(?:NYSE|NASDAQ|AMEX|NYSEARCA|OTC)\s*:\s*)" + re.escape(ticker) + r"\b"
    return bool(re.search(qualified, title, re.I) or
                (len(ticker) >= 4 and re.search(r"(?<![A-Z0-9])" + re.escape(ticker) + r"(?![A-Z0-9])", title)))


def _unique_object(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate JSON key")
    return obj


def _parse_response(raw: str, title: str, ticker: str) -> tuple[list[float], dict | None, str | None]:
    """Reject malformed or unsupported assertions, preserving raw text for audit."""
    neutral = [0.0] * len(LLM_FEATURE_NAMES)
    try:
        obj = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(obj, dict) or set(obj) != _KEYS:
            raise ValueError("JSON object must have exactly the six schema keys")
        role, kind, direction, status = (obj[k] for k in ("role", "class", "direction", "status"))
        if role not in _ROLES or kind not in _CLASSES or direction not in _DIRECTIONS or status not in _STATUSES:
            raise ValueError("unknown category value")
        evidence, magnitude = obj["evidence"], obj["magnitude_pct"]
        if not isinstance(evidence, str) or (evidence and evidence not in title):
            raise ValueError("evidence must be a verbatim title substring")
        if magnitude is not None:
            if isinstance(magnitude, bool) or not isinstance(magnitude, (float, int)) or not np.isfinite(magnitude) or magnitude < 0:
                raise ValueError("invalid percentage")
            if not evidence or not any(float(m.group(1)) == magnitude for m in _PERCENT.finditer(evidence)):
                raise ValueError("percentage absent from title evidence")
        if role == "unknown":
            if (kind, direction, status, magnitude, evidence) != ("unknown", "unknown", "unknown", None, ""):
                raise ValueError("uncertain issuer role must abstain completely")
            return neutral, obj, None
        if not evidence or not _issuer_mentions(evidence, ticker):
            raise ValueError("issuer not named in evidence")
        if any(other != ticker and _issuer_mentions(title, other) for other in ALIASES):
            raise ValueError("multiple named issuers require abstention")
        if kind != "unknown" and not _CLASS_CUES[kind].search(evidence):
            raise ValueError("event class absent from evidence")
        if direction != "unknown" and not (_POSITIVE if direction == "positive" else _NEGATIVE).search(evidence):
            raise ValueError("direction absent from evidence")
        if direction != "unknown" and _STOCK_PRICE.search(evidence):
            raise ValueError("stock price move is not an issuer event direction")
        if role == "object" and direction != "unknown":
            raise ValueError("object direction is ambiguous")
        if status == "expected" and not _EXPECTED.search(evidence):
            raise ValueError("expectation absent from evidence")
        if status == "rumor" and not _RUMOR.search(evidence):
            raise ValueError("rumor absent from evidence")
        if status == "confirmed" and (_RUMOR.search(evidence) or _EXPECTED.search(evidence)):
            raise ValueError("uncertain report cannot be confirmed")
        values = {role, kind, direction, status} - {"unknown"}
        features = [float(name in values) for name in LLM_FEATURE_NAMES[:-1]]
        features.append(min(float(magnitude) / 100.0, 1.0) if magnitude is not None else 0.0)
        return features, obj, None
    except (ValueError, TypeError, OverflowError, json.JSONDecodeError) as exc:
        return neutral, None, str(exc)


def _require_colab_cuda() -> None:
    # Cached CPU-only reads and parsing remain available outside Colab.
    if "google.colab" not in sys.modules:
        raise RuntimeError("Uncached headline extraction requires Google Colab with CUDA")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Uncached headline extraction requires Google Colab with CUDA")


def _generate_responses(rows: list[dict], batch_size: int):
    """Load pinned 4-bit weights once and emit only newly generated tokens."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_compute_dtype=torch.float16),
    )
    model.eval()
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = [tokenizer.apply_chat_template([
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps({"ticker": row["ticker"],
                    "aliases": ALIASES[row["ticker"]], "title": row["text"]}, ensure_ascii=False)},
            ], tokenize=False, add_generation_prompt=True) for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
            prompt_width = inputs["input_ids"].shape[1]
            yield tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            del generated, inputs
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()


def _manifest_rows(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            if (not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in
                    ("ticker", "decision_date", "published_at", "url", "source", "text")) or
                    row["ticker"] not in ALIASES or not row["text"] or
                    not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["decision_date"]) or
                    row["decision_date"] > "2023-11-30"):
                raise ValueError(f"Invalid verified headline manifest row {line_no}")
            np.datetime64(row["decision_date"], "D")
            rows.append(row)
    if not rows:
        raise ValueError("Empty verified headline manifest")
    return rows


def extract_llm_features(news_path, cache_dir, batch_size=4) -> tuple[dict, Path]:
    """Return ordered facts; append and fsync each finished batch on Drive."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1 or batch_size > 4:
        raise ValueError("batch_size must be between 1 and 4 for 4-bit Colab extraction")
    path, cache_dir = Path(news_path), Path(cache_dir)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest_hash = digest.hexdigest()
    key = hashlib.sha256(f"{manifest_hash}|{MODEL_ID}|{MODEL_REVISION}|{PROMPT_VERSION}|{','.join(LLM_FEATURE_NAMES)}".encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"llm_headlines_{key}.jsonl"
    rows = _manifest_rows(path)
    metadata = {"cache_key": key, "manifest_sha256": manifest_hash, "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION, "prompt_version": PROMPT_VERSION,
                "schema": list(LLM_FEATURE_NAMES)}
    features = np.zeros((len(rows), len(LLM_FEATURE_NAMES)), dtype=np.float32)
    statuses = []
    # r+b creates no holes: a trailing interrupted JSON line is truncated only after
    # every completed row has been validated against the current manifest.
    if not cache_path.exists():
        cache_path.touch()
    with cache_path.open("r+b") as cache:
        while True:
            position = cache.tell()
            line = cache.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                cache.truncate(position)
                cache.seek(position)
                break
            try:
                saved = json.loads(line)
                index = len(statuses)
                if not isinstance(saved, dict) or index >= len(rows):
                    raise ValueError("extra or malformed cached row")
                row = rows[index]
                if (any(saved.get(k) != v for k, v in metadata.items()) or
                    any(saved.get(k) != row[k] for k in ("ticker", "decision_date", "published_at", "url", "source", "text")) or
                    saved.get("index") != index or not isinstance(saved.get("raw"), str)):
                    raise ValueError("cache metadata or headline order mismatch")
                parsed_features, parsed, error = _parse_response(saved["raw"], row["text"], row["ticker"])
                status = "invalid" if error else "ok"
                stored = np.asarray(saved.get("features"), dtype=np.float32)
                if (saved.get("status") != status or saved.get("parsed") != parsed or
                    saved.get("error") != error or stored.shape != features.shape[1:] or
                    not np.isfinite(stored).all() or not np.array_equal(stored, np.asarray(parsed_features, dtype=np.float32))):
                    raise ValueError("cached facts or audit status mismatch")
                features[index] = stored
                statuses.append(status)
            except (ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError) as exc:
                raise ValueError(f"Incompatible headline cache row {len(statuses) + 1}: {cache_path}") from exc
        if len(statuses) < len(rows):
            print(f"LLM headline extraction: {len(statuses)}/{len(rows)} cached; generating remaining titles", flush=True)
            _require_colab_cuda()
            pending = rows[len(statuses):]
            generator = _generate_responses(pending, batch_size)
            try:
                for batch in generator:
                    size = min(batch_size, len(rows) - len(statuses))
                    if not isinstance(batch, list) or len(batch) != size or not all(isinstance(raw, str) for raw in batch):
                        raise ValueError("LLM returned an incomplete batch; no placeholder rows written")
                    lines = []
                    for raw in batch:
                        index = len(statuses)
                        row = rows[index]
                        vector, parsed, error = _parse_response(raw, row["text"], row["ticker"])
                        status = "invalid" if error else "ok"
                        record = {**metadata, "index": index, "ticker": row["ticker"],
                                  "decision_date": row["decision_date"], "published_at": row["published_at"],
                                  "url": row["url"], "source": row["source"], "text": row["text"],
                                  "raw": raw, "parsed": parsed, "error": error, "status": status,
                                  "features": vector}
                        lines.append(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        features[index] = vector
                        statuses.append(status)
                    cache.write("".join(lines).encode("utf-8"))
                    cache.flush()
                    os.fsync(cache.fileno())
                    if len(statuses) % 400 < size or len(statuses) == len(rows):
                        print(f"LLM headline extraction: {len(statuses)}/{len(rows)} titles persisted", flush=True)
            finally:
                generator.close()
            if len(statuses) != len(rows):
                raise ValueError("LLM did not complete the manifest; downstream training is forbidden")
    result = {"ticker": np.asarray([r["ticker"] for r in rows]),
              "decision_date": np.asarray([r["decision_date"] for r in rows], dtype="datetime64[D]"),
              "published_at": np.asarray([r["published_at"] for r in rows]),
              "features": features, "status": np.asarray(statuses)}
    return result, cache_path
