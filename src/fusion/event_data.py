"""Painel de retornos e manchetes com datas de decisão conservadoras (sem inferência)."""

from collections import Counter
import gzip
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd

from src.fusion.dataset import _load_prices

VERSION = 1
START = pd.Timestamp("2020-01-01")
CUTOFF = pd.Timestamp("2023-11-30")
HORIZON = 5
SEQ_LEN = 10
MAX_MOMENT_AGE = 5
MAX_EVENTS_PER_DAY = 5
SOURCE_NOTE = (
    "Fonte bruta: títulos/URL e corpos de artigos às vezes não correspondem; "
    "somente manchetes coerentes com URL e com nome/símbolo do ativo foram exportadas; "
    "corpos de artigo nunca são utilizados. Datas de publicação sem horário são "
    "registradas como meia-noite UTC apenas para ordenação; a negociação aguarda "
    "mais um pregão inteiro após o primeiro pregão possível. "
    "Datas sem notícia não comprovam ausência de notícias. Seleção retrospectiva do "
    "universo de 49 ativos pode gerar viés de sobrevivência/seleção. "
    "Há falha conhecida no feed a partir de 2024 (zero notícias em janeiro/2024); "
    "GDELT histórico respondeu HTTP 429, sem acesso a API histórica paga. "
    "Experimento limitado a 2020-01-01..2023-11-30; 2024+ não pode ser avaliado."
)

# Correspondência explícita da manchete, nunca inferida apenas da etiqueta do arquivo.
ALIASES = {
    "AAL": ("american airlines",), "ABBV": ("abbvie",), "ABT": ("abbott",),
    "ADBE": ("adobe",), "AMGN": ("amgen",), "AMT": ("american tower",),
    "BABA": ("alibaba",), "BBY": ("best buy",), "BHP": ("bhp",),
    "BIIB": ("biogen",), "BX": ("blackstone",), "CAT": ("caterpillar",),
    "CMCSA": ("comcast",), "CMG": ("chipotle",), "COP": ("conocophillips",),
    "COST": ("costco",), "CRM": ("salesforce",), "CVX": ("chevron",),
    "EBAY": ("ebay",), "EPD": ("enterprise products partners",),
    "GE": ("general electric",), "GILD": ("gilead",),
    "GLD": ("spdr gold", "gold etf"), "GS": ("goldman sachs",),
    "GSK": ("glaxosmithkline",), "HPQ": ("hewlett packard", "hp inc"),
    "KO": ("coca cola", "coca-cola",), "MRK": ("merck",),
    "MS": ("morgan stanley",), "MU": ("micron",),
    "NEE": ("nextera",), "NKE": ("nike",), "NVS": ("novartis",),
    "ORCL": ("oracle",), "PEP": ("pepsico", "pepsi"),
    "PYPL": ("paypal",), "QCOM": ("qualcomm",),
    "QQQ": ("invesco qqq", "nasdaq 100 etf"),
    "T": ("at&t", "at & t"), "TM": ("toyota",),
    "TMUS": ("t-mobile", "t mobile"),
    "TSM": ("taiwan semiconductor", "tsmc",),
    "UNG": ("united states natural gas fund", "natural gas etf"),
    "UPS": ("united parcel service",),
    "USO": ("united states oil fund", "oil etf"),
    "V": ("visa",), "WFC": ("wells fargo",),
    "XLE": ("energy select sector",), "XLF": ("financial select sector",),
}
STOP_WORDS = {"the", "a", "an", "of", "and", "or", "to", "for", "on", "in", "with", "at", "is", "as", "stock", "stocks", "shares", "news"}
TOKEN = re.compile(r"[a-z0-9]+")


def _published(value):
    """Retorna a publicação UTC; data sem horário preserva sua incerteza."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        timestamp = pd.Timestamp(value.strip())
        if pd.isna(timestamp):
            return None
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z"), timestamp.tz_localize(None).normalize()
    except (ValueError, TypeError, OverflowError):
        return None


def _headline(record, ticker):
    url = record.get("Url")
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return None
    except ValueError:
        return None
    slug = unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
    slug_text = re.sub(r"[-_]+", " ", slug).strip()
    slug_tokens = set(TOKEN.findall(slug_text.lower())) - STOP_WORDS
    title = record.get("Article_title")
    title = " ".join(title.split()) if isinstance(title, str) else ""
    if not title or not slug_tokens:
        return None
    title_tokens = set(TOKEN.findall(title.lower())) - STOP_WORDS
    # Um título desalinhado com a URL não comprova a associação do artigo.
    if len(slug_tokens & title_tokens) < min(2, len(slug_tokens), len(title_tokens)):
        return None
    text = title[:400]
    normalized = " ".join(TOKEN.findall(text.lower()))
    aliases = ALIASES.get(ticker, ())
    relevant = any(re.search(r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])", text.lower()) for alias in aliases)
    # Símbolos curtos são ambíguos; requerem cashtag ou bolsa explícita.
    qualified = r"(?:\$|(?:NYSE|NASDAQ|AMEX|NYSEARCA|OTC)\s*:\s*)" + re.escape(ticker) + r"\b"
    relevant |= bool(re.search(qualified, text, flags=re.IGNORECASE))
    relevant |= len(ticker) >= 4 and bool(re.search(r"(?<![A-Z0-9])" + re.escape(ticker) + r"(?![A-Z0-9])", text))
    if not relevant or not normalized:
        return None
    return url, parsed.hostname.lower(), text, normalized


def export_events(text_dir, trading_dates, path, decision_dates=None):
    """Lê JSONL em fluxo; ordena/deduplica manchetes antes do limite diário.

    `trading_dates` contém sessões de preços; `decision_dates` contém sessões
    utilizáveis pelo painel. Exige uma sessão extra após a primeira possível.
    """
    text_dir, path = Path(text_dir), Path(path)
    months = pd.period_range("2020-01", "2023-11", freq="M").astype(str).tolist()
    month_index = {month: i for i, month in enumerate(months)}
    coverage = np.zeros((len(trading_dates), len(months)), dtype=np.int32)
    monthly = Counter()
    raw_monthly = Counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as out:
        for ti, ticker in enumerate(trading_dates):
            source = text_dir / f"{ticker}.jsonl"
            if not source.is_file():
                raise FileNotFoundError(f"Arquivo de notícias ausente: {source}")
            dates = np.asarray(trading_dates[ticker], dtype="datetime64[ns]")
            usable = None if decision_dates is None else set(np.asarray(decision_dates[ticker], dtype="datetime64[ns]"))
            candidates = []
            with source.open(encoding="utf-8") as rows:
                for line_no, line in enumerate(rows, 1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSONL inválido: {source}:{line_no}") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"Registro não é um objeto JSON: {source}:{line_no}")
                    tag = record.get("Stock_symbol")
                    if not isinstance(tag, str) or tag.upper() != ticker:
                        continue
                    published = _published(record.get("Date"))
                    if published is None or not START <= published[1] <= CUTOFF:
                        continue
                    raw_monthly[published[1].strftime("%Y-%m")] += 1
                    headline = _headline(record, ticker)
                    if headline is None:
                        continue
                    pub, day = published
                    ix = np.searchsorted(dates, np.datetime64(day)) + 1
                    if ix >= len(dates) or dates[ix] > np.datetime64(CUTOFF):
                        continue
                    if usable is not None and dates[ix] not in usable:
                        continue
                    url, domain, text, normalized = headline
                    candidates.append((pub, url, domain, text, normalized, str(dates[ix])[:10]))
            candidates.sort(key=lambda item: (item[0], item[1]))
            seen_urls, seen_text, day_count = set(), set(), Counter()
            for pub, url, domain, text, normalized, decision in candidates:
                canonical_url = urlsplit(url)._replace(fragment="").geturl().lower()
                if canonical_url in seen_urls or normalized in seen_text:
                    continue
                seen_urls.add(canonical_url)
                seen_text.add(normalized)
                if day_count[decision] >= MAX_EVENTS_PER_DAY:
                    continue
                day_count[decision] += 1
                month = decision[:7]
                monthly[month] += 1
                coverage[ti, month_index[month]] += 1
                out.write(json.dumps({"ticker": ticker, "published_at": pub,
                                      "decision_date": decision, "url": url,
                                      "source": domain, "text": text}, ensure_ascii=False) + "\n")
    return {"months": months, "counts": np.array([monthly[m] for m in months], dtype=np.int32),
            "raw_counts": np.array([raw_monthly[m] for m in months], dtype=np.int32),
            "ticker_month_counts": coverage}


def load_event_panel(cache_dir="cache", ts_dir="data/time_series", seq_len=SEQ_LEN,
                     horizon=HORIZON):
    """Constrói painel diário 2020..2023 com rótulos de retorno relativo de 5 sessões."""
    if seq_len < 1 or horizon < 1:
        raise ValueError("Janela e horizonte devem ser positivos")
    meta = pd.read_csv(Path(cache_dir) / "moment/top50_moment_metadata.csv", usecols=["ticker", "date"])
    embeddings = np.load(Path(cache_dir) / "moment/top50_moment_embeddings.npy", mmap_mode="r")
    if len(meta) != len(embeddings):
        raise ValueError("Metadados e embeddings MOMENT desalinhados")
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    meta["mom_idx"] = np.arange(len(meta))
    tickers = sorted(meta["ticker"].dropna().unique())
    arrays, prices, market_sources = [], {}, []
    for ticker in tickers:
        px = _load_prices(ticker, ts_dir)
        px = px.loc[px["Date"].between(START, CUTOFF)].copy()
        if len(px) < seq_len + horizon + 30:
            raise ValueError(f"Histórico insuficiente: {ticker}")
        prices[ticker] = px["Date"].to_numpy(dtype="datetime64[ns]")
        px["known_ret"] = px["Close"].pct_change()
        px["past_vol"] = np.sqrt(252 * np.log(px["Close"]).diff().pow(2).rolling(horizon, min_periods=horizon).mean())
        px["fwd_return"] = px["Close"].shift(-horizon) / px["Close"] - 1
        px["target_end"] = px["Date"].shift(-horizon)
        market_sources.append(pd.Series(px["fwd_return"].to_numpy(), index=pd.DatetimeIndex(px["Date"]), name=ticker))
        m = meta.loc[meta["ticker"] == ticker, ["date", "mom_idx"]]
        px = px.merge(m, left_on="Date", right_on="date", how="left").drop(columns="date")
        px["mom_idx"] = px["mom_idx"].ffill(limit=MAX_MOMENT_AGE)
        px = px.dropna(subset=["mom_idx", "known_ret", "past_vol"]).reset_index(drop=True)
        if len(px) < seq_len + horizon + 30:
            raise ValueError(f"Sem sequência MOMENT utilizável: {ticker}")
        arrays.append({"dates": px["Date"].to_numpy(dtype="datetime64[ns]"),
                       "target_end": px["target_end"].to_numpy(dtype="datetime64[ns]"),
                       "mom": np.asarray(embeddings[px["mom_idx"].to_numpy(dtype=int)], dtype=np.float32),
                       "past_vol": px["past_vol"].to_numpy(dtype=np.float32),
                       "known_ret": px["known_ret"].to_numpy(dtype=np.float32),
                       "fwd_return": px["fwd_return"].to_numpy(dtype=np.float32)})
    # Média do buy-and-hold por 5 sessões de TODO o universo de preços, inclusive
    # ativos sem embedding nesta sessão. Jamais calcula a partir dos rótulos filtrados.
    market = pd.concat(market_sources, axis=1).mean(axis=1)
    for a in arrays:
        a["market_fwd_return"] = market.reindex(a["dates"]).to_numpy(dtype=np.float32)
        a["target"] = a["fwd_return"] - a["market_fwd_return"]
    all_dates = np.sort(np.unique(np.concatenate([a["dates"] for a in arrays])))
    train_end = all_dates[int(len(all_dates) * 0.70) - 1]
    val_end = all_dates[int(len(all_dates) * 0.70 * 0.85) - 1]
    splits = {s: [] for s in ("train", "val", "test")}
    for ti, a in enumerate(arrays):
        for row in range(seq_len - 1, len(a["dates"])):
            d, end = a["dates"][row], a["target_end"][row]
            if np.isnat(end) or not np.isfinite(a["target"][row]):
                continue
            if d <= val_end:
                if end <= val_end:
                    splits["train"].append((ti, row))
            elif d <= train_end:
                if end <= train_end:
                    splits["val"].append((ti, row))
            elif end <= np.datetime64(CUTOFF):
                splits["test"].append((ti, row))
    if any(not rows for rows in splits.values()):
        raise ValueError("Split temporal vazio após purga do horizonte")
    tr_mom = np.concatenate([a["mom"][np.asarray([r for ti, r in splits["train"] if ti == i], dtype=int)]
                             for i, a in enumerate(arrays)])
    mu, sd = tr_mom.mean(axis=0), tr_mom.std(axis=0) + 1e-6
    for a in arrays:
        a["mom"] = (a["mom"] - mu) / sd
    return {"tickers": tickers, "arrays": arrays, "splits": splits,
            "val_end": val_end, "train_end": train_end, "seq_len": seq_len,
            "horizon": horizon, "trading_dates": prices}


def save_event_panel(panel, path, coverage=None):
    out = {"version": np.array(VERSION), "tickers": np.asarray(panel["tickers"]),
           "horizon": np.array(panel["horizon"]), "seq_len": np.array(panel["seq_len"]),
           "bounds": np.asarray([panel["val_end"], panel["train_end"]], dtype="datetime64[ns]"),
           "source_note": np.array(SOURCE_NOTE), "source_start": np.array(str(START.date())),
           "source_cutoff": np.array(str(CUTOFF.date()))}
    for i, a in enumerate(panel["arrays"]):
        out.update({f"arr_{i}_{k}": v for k, v in a.items()})
    for name, samples in panel["splits"].items():
        out[f"samples_{name}"] = np.asarray(samples, dtype=np.int32).reshape(-1, 2)
    if coverage is not None:
        out.update(news_months=np.asarray(coverage["months"]), news_counts=coverage["counts"],
                   news_raw_counts=coverage["raw_counts"], news_ticker_month_counts=coverage["ticker_month_counts"])
    np.savez_compressed(path, **out)


def load_event_panel_file(path):
    with np.load(path, allow_pickle=False) as data:
        if "version" not in data or int(data["version"]) != VERSION:
            raise ValueError("Painel de eventos obsoleto; execute export_event_signal.py novamente")
        tickers = data["tickers"].tolist()
        keys = ("dates", "target_end", "mom", "past_vol", "known_ret", "target", "fwd_return", "market_fwd_return")
        arrays = [{k: data[f"arr_{i}_{k}"] for k in keys} for i in range(len(tickers))]
        splits = {s: data[f"samples_{s}"].astype(np.int32) for s in ("train", "val", "test")}
        result = {"tickers": tickers, "arrays": arrays, "splits": splits,
                  "val_end": data["bounds"][0], "train_end": data["bounds"][1],
                  "seq_len": int(data["seq_len"]), "horizon": int(data["horizon"]),
                  "source_note": str(data["source_note"])}
        for key in ("news_months", "news_counts", "news_raw_counts", "news_ticker_month_counts"):
            if key in data:
                result[key] = data[key]
        return result
