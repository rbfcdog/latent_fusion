#!/usr/bin/env python3
"""
B3 Brazilian Stock Exchange — Historical Data & News Scraper v4
===============================================================
Fetches OHLCV price data + news since 2022 for 110+ B3 tickers.

NEWS SOURCES (guaranteed coverage, layered by reliability):
  Tier 1 — Structural Guarantees
    1. CVM Fatos Relevantes   — official B3 regulatory filings, 100% of listed cos.
    2. CVM ITR/DFP Filings    — quarterly/annual reports metadata
  Tier 2 — Broad Coverage
    3. Google News RSS        — widest Brazilian financial press coverage
    4. Valor Econômico RSS    — Brazil's top financial newspaper
    5. InfoMoney RSS          — leading retail investor site (RSS, not scraped HTML)
    6. Exame RSS              — major business press
  Tier 3 — Supplemental
    7. GDELT DOC 2.0          — global news index, PT-BR, since 2017
    8. yfinance headlines     — English-language recent (~90 days)

PRICE SOURCES:
    1. yfinance (Yahoo Finance) — adjusted OHLCV, full history
    2. B3 COTAHIST              — official raw files, 1995-present (--cotahist flag)

OUTPUT LAYOUT:
    data/prices/<TICKER>.csv          Daily OHLCV
    data/news/<TICKER>_news.csv       Headlines: date, title, url, source, tone, language
    data/fundamentals/<TICKER>.json   Key financial metrics
    data/cvm/<TICKER>_cvm.csv         Regulatory filings from CVM
    data/summary.csv                  Per-ticker run statistics
    b3_scraper.log                    Full execution log

QUICK START:
    pip install yfinance pandas requests feedparser python-dateutil lxml
    python b3_scraper_v4.py                          # all tickers, 2022→today
    python b3_scraper_v4.py --limit 5               # test with 5 tickers
    python b3_scraper_v4.py --tickers PETR4 VALE3   # specific tickers
    python b3_scraper_v4.py --no-gdelt              # skip GDELT (faster)
    python b3_scraper_v4.py --start 2020-01-01      # custom start date
    python b3_scraper_v4.py --workers 4             # parallel news fetching
    python b3_scraper_v4.py --cotahist              # add official B3 price files
"""

import io
import json
import time
import zipfile
import logging
import argparse
import warnings
import re
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import yfinance as yf
import feedparser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("b3_scraper.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS & DEFAULTS
# ════════════════════════════════════════════════════════════════════════════

START_DATE   = "2022-01-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")
GDELT_START  = "2022-01-01"

OUTPUT_DIR   = Path("data")
PRICES_DIR   = OUTPUT_DIR / "prices"
NEWS_DIR     = OUTPUT_DIR / "news"
FUND_DIR     = OUTPUT_DIR / "fundamentals"
CVM_DIR      = OUTPUT_DIR / "cvm"
COTAHIST_DIR = OUTPUT_DIR / "cotahist"

NEWS_COLUMNS = [
    "ticker", "source", "date", "title", "url",
    "publisher", "summary", "language",
    "gdelt_tone", "gdelt_themes", "gdelt_countries",
]


# ════════════════════════════════════════════════════════════════════════════
# TICKER LIST — 110+ liquid B3 stocks
# ════════════════════════════════════════════════════════════════════════════

B3_TICKERS = [
    # ── Mega caps / Ibovespa core ──────────────────────────────────────────
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "RENT3",
    "SUZB3", "JBSS3", "LREN3", "MGLU3", "GGBR4", "USIM5", "CSNA3", "CSAN3",
    "RADL3", "HAPV3", "RDOR3", "RAIL3", "SBSP3",
    # ── Energy / Utilities ────────────────────────────────────────────────
    "ELET3", "ELET6", "CMIG4", "CPFE3", "ENGI11", "TAEE11", "TRPL4", "EQTL3",
    "ENBR3", "NEOE3", "CPLE6", "LIGT3", "AURE3", "EGIE3",
    # ── Banks & Financials ────────────────────────────────────────────────
    "BBAS3", "SANB11", "ITSA4", "BPAC11", "IRBR3", "BBSE3", "SULA11",
    "PSSA3", "BRSR6", "PINE4", "BMGB4", "BPAN4",
    # ── Consumer & Retail ─────────────────────────────────────────────────
    "VIIA3", "AMER3", "PETZ3", "NTCO3", "SOMA3", "CEAB3", "GRND3",
    "PCAR3", "ASAI3", "LWSA3",
    # ── Healthcare ────────────────────────────────────────────────────────
    "FLRY3", "DASA3", "PARD3", "QUAL3", "GNDI3",
    # ── Agribusiness & Food ───────────────────────────────────────────────
    "SLCE3", "AGRO3", "SMTO3", "MRFG3", "BEEF3", "BRFS3",
    # ── Real Estate ───────────────────────────────────────────────────────
    "MULT3", "IGTI11", "EVEN3", "JHSF3", "CYRE3", "MRVE3",
    "EZTC3", "DIRR3", "TEND3",
    # ── Telecoms & Tech ───────────────────────────────────────────────────
    "VIVT3", "TIMS3", "TOTVS3", "CASH3", "BIDI11", "POSI3",
    # ── Mining, Steel & Materials ─────────────────────────────────────────
    "CMIN3", "FESA4", "ROMI3", "UNIP6",
    # ── Oil & Gas ─────────────────────────────────────────────────────────
    "PRIO3", "RECV3", "RRRP3", "UGPA3", "VBBR3",
    # ── Transport & Logistics ─────────────────────────────────────────────
    "GOLL4", "AZUL4", "CCRO3", "ECOR3",
    # ── Education ─────────────────────────────────────────────────────────
    "COGN3", "YDUQ3", "ANIM3", "SEER3",
    # ── Pulp & Paper ──────────────────────────────────────────────────────
    "KLBN11", "DXCO3",
    # ── Sanitation ────────────────────────────────────────────────────────
    "SAPR11", "CSMG3",
    # ── Other liquid tickers ──────────────────────────────────────────────
    "VVAR3", "OIBR3", "GMAT3", "INTB3", "CXSE3", "WIZS3",
]
B3_TICKERS = list(dict.fromkeys(B3_TICKERS))  # deduplicate, preserve order

# Canonical company names used in search queries
TICKER_NAMES: dict[str, str] = {
    "PETR4": "Petrobras", "VALE3": "Vale", "ITUB4": "Itaú Unibanco",
    "BBDC4": "Bradesco",  "ABEV3": "Ambev", "B3SA3": "B3",
    "WEGE3": "WEG", "RENT3": "Localiza", "SUZB3": "Suzano",
    "JBSS3": "JBS", "LREN3": "Lojas Renner", "MGLU3": "Magazine Luiza",
    "GGBR4": "Gerdau", "USIM5": "Usiminas", "CSNA3": "CSN",
    "CSAN3": "Cosan", "RADL3": "Raia Drogasil", "HAPV3": "Hapvida",
    "RDOR3": "Rede D'Or", "RAIL3": "Rumo", "SBSP3": "Sabesp",
    "ELET3": "Eletrobras", "ELET6": "Eletrobras", "CMIG4": "Cemig",
    "CPFE3": "CPFL Energia", "ENGI11": "Energisa", "TAEE11": "Taesa",
    "EQTL3": "Equatorial Energia", "ENBR3": "EDP Brasil",
    "CPLE6": "Copel", "LIGT3": "Light", "EGIE3": "Engie Brasil",
    "BBAS3": "Banco do Brasil", "SANB11": "Santander Brasil",
    "ITSA4": "Itaúsa", "BPAC11": "BTG Pactual", "IRBR3": "IRB Brasil Re",
    "BBSE3": "BB Seguridade", "SULA11": "SulAmérica",
    "PSSA3": "Porto Seguro", "BRSR6": "Banrisul",
    "VIIA3": "Via Varejo", "AMER3": "Americanas", "PETZ3": "Petz",
    "NTCO3": "Grupo Boticário", "SOMA3": "Grupo Soma",
    "PCAR3": "GPA Pão de Açúcar", "ASAI3": "Assaí Atacadista",
    "FLRY3": "Fleury", "DASA3": "Diagnósticos da América",
    "QUAL3": "Qualicorp", "GNDI3": "Grupo Notre Dame Intermédica",
    "SLCE3": "SLC Agrícola", "AGRO3": "BrasilAgro",
    "MRFG3": "Marfrig", "BEEF3": "Minerva Foods", "BRFS3": "BRF",
    "MULT3": "Multiplan", "CYRE3": "Cyrela", "MRVE3": "MRV Engenharia",
    "VIVT3": "Telefônica Brasil", "TIMS3": "TIM Brasil",
    "TOTVS3": "TOTVS", "PRIO3": "PRIO", "RECV3": "PetroRecôncavo",
    "UGPA3": "Ultrapar", "VBBR3": "Vibra Energia",
    "GOLL4": "Gol Linhas Aéreas", "AZUL4": "Azul Linhas Aéreas",
    "CCRO3": "CCR", "ECOR3": "EcoRodovias",
    "COGN3": "Cogna Educação", "YDUQ3": "Yduqs",
    "KLBN11": "Klabin", "DXCO3": "Dexco",
    "SAPR11": "Sanepar", "CSMG3": "Copasa",
}

# CVM CNPJ / company code map for fatos relevantes lookup
# CVM code (cod_cvm) for top tickers — extend as needed
CVM_CODES: dict[str, str] = {
    "PETR4": "9512",  "VALE3": "4170",  "ITUB4": "19348", "BBDC4": "906",
    "ABEV3": "16462", "WEGE3": "5410",  "RENT3": "20737", "SUZB3": "19615",
    "JBSS3": "20305", "MGLU3": "18732", "GGBR4": "3505",  "CSNA3": "11259",
    "RADL3": "4243",  "HAPV3": "20036", "RDOR3": "21067", "RAIL3": "20257",
    "SBSP3": "15164", "ELET3": "2437",  "CMIG4": "1587",  "CPFE3": "18295",
    "ENGI11":"21695", "TAEE11":"22411", "EQTL3": "22470", "CPLE6": "1023",
    "BBAS3": "906",   "SANB11":"90400027", "ITSA4": "19348","BPAC11":"21033",
    "IRBR3": "14427", "BBSE3": "21580", "PSSA3": "9849",  "BRSR6": "17671",
    "PETZ3": "21961", "FLRY3": "7530",  "DASA3": "15611", "QUAL3": "16071",
    "SLCE3": "20613", "AGRO3": "20087", "MRFG3": "19458", "BRFS3": "16248",
    "MULT3": "20575", "CYRE3": "19437", "MRVE3": "18678", "VIVT3": "21701",
    "TIMS3": "18112", "TOTVS3":"20808", "PRIO3": "9329",  "UGPA3": "6351",
    "GOLL4": "20435", "AZUL4": "21156", "CCRO3": "17387", "COGN3": "20680",
    "YDUQ3": "20540", "KLBN11":"31170", "SAPR11":"15539", "CSMG3": "4391",
    "ELET6": "2437",  "TRPL4": "17450", "ENBR3": "14453", "LIGT3": "1406",
    "EGIE3": "21032", "NEOE3": "21490", "AURE3": "22268",
}


# ════════════════════════════════════════════════════════════════════════════
# HTTP SESSION — shared across all modules
# ════════════════════════════════════════════════════════════════════════════

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    })
    return s

SESSION = make_session()


def setup_dirs():
    for d in [PRICES_DIR, NEWS_DIR, FUND_DIR, CVM_DIR, COTAHIST_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _company_name(ticker: str) -> str:
    return TICKER_NAMES.get(ticker, ticker.rstrip("0123456789"))


def _make_record(ticker: str, source: str, date_str: str, title: str,
                 url: str, publisher: str = "", summary: str = "",
                 language: str = "pt", gdelt_tone: float | None = None,
                 gdelt_themes: str = "", gdelt_countries: str = "") -> dict:
    """Canonical news record constructor — ensures consistent schema."""
    return {
        "ticker":          ticker,
        "source":          source,
        "date":            date_str,
        "title":           title.strip(),
        "url":             url.strip(),
        "publisher":       publisher,
        "summary":         summary,
        "language":        language,
        "gdelt_tone":      gdelt_tone,
        "gdelt_themes":    gdelt_themes,
        "gdelt_countries": gdelt_countries,
    }


def _dedup_url(url: str) -> str:
    """Normalise URL for dedup: strip tracking params, lowercase scheme/host."""
    url = re.sub(r'[?&](utm_[^&]+|ref=[^&]+|source=[^&]+)', '', url)
    url = re.sub(r'[?&]$', '', url)
    return url


# ════════════════════════════════════════════════════════════════════════════
# TIER 1: CVM — OFFICIAL REGULATORY FILINGS (100% GUARANTEED)
# ════════════════════════════════════════════════════════════════════════════
#
# CVM (Brazil's SEC) publishes every regulatory disclosure at dados.cvm.gov.br.
# Two endpoints are used:
#
#   A) Fatos Relevantes (Material Facts) — ITR, DFP, press releases, M&A, etc.
#      https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRENOT/DADOS/
#      Annual CSV files, one row per filing.
#
#   B) CVM FCA (Formulário Cadastral) metadata — company info
#      https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/
#
# Files are plain CSV (pipe-delimited, ISO-8859-1) updated daily.
# No auth required. Covers 100% of B3-listed companies.
#
# Columns of interest in frenot files:
#   DT_REFER  — reference date (YYYY-MM-DD)
#   DENOM_CIA — company name
#   CD_CVM    — CVM company code
#   CATEG_DOC — document category (e.g. "Fato Relevante", "ITR", "DFP")
#   DT_RECEB  — date received by CVM
#   LINK_DOC  — URL to the actual document PDF

CVM_BASE   = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC"
CVM_CAD    = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"

# CVM publishes one file per year for each document type
CVM_DOC_TYPES = {
    "frenot": "FRENOT",   # Fatos Relevantes / Comunicados
    "itr":    "ITR",      # Quarterly reports
    "dfp":    "DFP",      # Annual reports
}

_cvm_cadaster_cache: dict[str, str] | None = None  # cod_cvm → denom_cia


def _load_cvm_cadaster() -> dict[str, str]:
    """Load CVM company register: returns {denom_cia_upper: cod_cvm}."""
    global _cvm_cadaster_cache
    if _cvm_cadaster_cache is not None:
        return _cvm_cadaster_cache

    try:
        log.info("[CVM] Loading company cadaster...")
        resp = SESSION.get(CVM_CAD, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(
            io.BytesIO(resp.content),
            sep=";",
            encoding="iso-8859-1",
            dtype=str,
            on_bad_lines="skip",
        )
        df.columns = [c.strip().upper() for c in df.columns]
        # Build lookup: normalized company name → cod_cvm
        lookup = {}
        for _, row in df.iterrows():
            cod = str(row.get("CD_CVM", "")).strip()
            nome = str(row.get("DENOM_CIA", "")).strip().upper()
            if cod and nome:
                lookup[nome] = cod
        _cvm_cadaster_cache = lookup
        log.info(f"[CVM] Cadaster loaded: {len(lookup):,} companies")
        return lookup
    except Exception as e:
        log.warning(f"[CVM] Cadaster load failed: {e}")
        _cvm_cadaster_cache = {}
        return {}


def _cvm_cod_for_ticker(ticker: str) -> str | None:
    """Resolve CVM cod_cvm for a B3 ticker, using hardcoded map first."""
    if ticker in CVM_CODES:
        return CVM_CODES[ticker]
    # Fallback: fuzzy match against cadaster by company name
    cad = _load_cvm_cadaster()
    name = _company_name(ticker).upper()
    for cad_name, cod in cad.items():
        if name in cad_name or cad_name in name:
            return cod
    return None


def _cvm_annual_file_url(doc_type: str, year: int) -> str:
    """Build URL for a CVM annual data file."""
    folder = CVM_DOC_TYPES[doc_type]
    return f"{CVM_BASE}/{folder}/DADOS/{doc_type.lower()}_{year}.csv"


def fetch_cvm_filings(ticker: str,
                      start: str = START_DATE,
                      end: str   = END_DATE,
                      doc_types: list[str] | None = None) -> list[dict]:
    """
    Fetch CVM regulatory filings for a ticker from start to end.

    Downloads annual CSV files, filters by CVM code, and returns records
    formatted as news items so they merge cleanly with other sources.

    Returns [] on any failure — never raises.
    """
    if doc_types is None:
        doc_types = list(CVM_DOC_TYPES.keys())

    cod = _cvm_cod_for_ticker(ticker)
    if not cod:
        log.warning(f"[{ticker}] CVM: no company code found, skipping CVM")
        return []

    start_year = int(start[:4])
    end_year   = int(end[:4])
    records    = []
    seen_urls: set[str] = set()

    for doc_type in doc_types:
        for year in range(start_year, end_year + 1):
            url = _cvm_annual_file_url(doc_type, year)
            try:
                resp = SESSION.get(url, timeout=60)
                if resp.status_code == 404:
                    continue  # year not yet published
                resp.raise_for_status()

                df = pd.read_csv(
                    io.BytesIO(resp.content),
                    sep=";",
                    encoding="iso-8859-1",
                    dtype=str,
                    on_bad_lines="skip",
                )
                df.columns = [c.strip().upper() for c in df.columns]

                # Filter to this company
                if "CD_CVM" not in df.columns:
                    continue
                df = df[df["CD_CVM"].astype(str).str.strip() == str(cod)]
                if df.empty:
                    continue

                # Normalise date column — could be DT_REFER or DT_RECEB
                date_col = next(
                    (c for c in ["DT_RECEB", "DT_REFER", "DT_INI_EXERC"]
                     if c in df.columns),
                    None,
                )
                if date_col is None:
                    continue

                df["_date_parsed"] = pd.to_datetime(
                    df[date_col], errors="coerce", dayfirst=False
                )
                # Apply date range filter
                mask = (
                    (df["_date_parsed"] >= pd.Timestamp(start)) &
                    (df["_date_parsed"] <= pd.Timestamp(end))
                )
                df = df[mask]

                for _, row in df.iterrows():
                    # Build title from category + description if available
                    categ  = str(row.get("CATEG_DOC", row.get("CATEG", ""))).strip()
                    descr  = str(row.get("DESCR_DOC", row.get("DESCR", ""))).strip()
                    denom  = str(row.get("DENOM_CIA", ticker)).strip()
                    title  = f"[{categ}] {descr}" if descr and descr != "nan" \
                             else f"[{categ}] {denom}"
                    title  = title.replace("nan", "").strip(" []")

                    link_col = next(
                        (c for c in ["LINK_DOC", "URL_DOC"] if c in df.columns), None
                    )
                    doc_url = str(row.get(link_col, "")).strip() if link_col else ""
                    if not doc_url or doc_url == "nan":
                        # Construct a deterministic synthetic URL for dedup
                        _hash = hashlib.md5(
                            f"{ticker}{row['_date_parsed']}{title}".encode()
                        ).hexdigest()[:12]
                        doc_url = f"https://dados.cvm.gov.br/#{ticker}_{doc_type}_{_hash}"

                    if _dedup_url(doc_url) in seen_urls:
                        continue
                    seen_urls.add(_dedup_url(doc_url))

                    dt_str = row["_date_parsed"].strftime("%Y-%m-%d %H:%M:%S")
                    records.append(_make_record(
                        ticker    = ticker,
                        source    = f"cvm_{doc_type}",
                        date_str  = dt_str,
                        title     = title or f"CVM {doc_type.upper()} filing",
                        url       = doc_url,
                        publisher = "CVM",
                        summary   = descr if descr != "nan" else "",
                        language  = "pt",
                    ))

            except Exception as e:
                log.debug(f"[{ticker}] CVM {doc_type} {year}: {e}")

    log.info(f"[{ticker}] CVM: {len(records):,} filings "
             f"({', '.join(doc_types)})")
    return records


# ════════════════════════════════════════════════════════════════════════════
# TIER 2A: GOOGLE NEWS RSS — WIDEST COVERAGE
# ════════════════════════════════════════════════════════════════════════════
#
# Google News RSS is completely free, requires no API key, and indexes
# virtually every Brazilian financial publication. The RSS feed returns
# the ~100 most recent articles matching the query; for older dates the
# 'when:Nd' modifier restricts to the last N days, letting us page backward.
#
# Strategy: use overlapping backward windows of 30 days from end → start.
# Each window uses `when:30d after:YYYY-MM-DD before:YYYY-MM-DD` operators.
# Google quietly rate-limits aggressive scrapers; we wait 2 s between calls.
#
# RSS feed structure (feedparser):
#   entry.title    — headline
#   entry.link     — redirect URL (Google wraps it)
#   entry.published— date string
#   entry.summary  — short snippet
#   entry.source.title — publisher name

GNEWS_BASE    = "https://news.google.com/rss/search"
GNEWS_DELAY   = 2.5  # seconds between RSS calls to avoid soft rate-limits


def _gnews_rss_url(query: str) -> str:
    return f"{GNEWS_BASE}?q={quote_plus(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"


def _parse_gnews_date(entry) -> str:
    """Parse feedparser entry date to ISO string."""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(entry, "published"):
            return str(entry.published)[:19]
    except Exception:
        pass
    return ""


def _gnews_window_query(name: str, ticker: str,
                         after: str, before: str,
                         language: str = "pt") -> list[dict]:
    """Query Google News RSS for one time window."""
    # Google News date operators: after:YYYY-MM-DD before:YYYY-MM-DD
    base_terms = f'"{name}" OR "{ticker}" bolsa B3'
    date_part  = f"after:{after} before:{before}"
    query      = f"{base_terms} {date_part}"
    url        = _gnews_rss_url(query)

    try:
        time.sleep(GNEWS_DELAY)
        parsed = feedparser.parse(url)
        records = []
        for entry in parsed.entries:
            dt_str = _parse_gnews_date(entry)
            link   = getattr(entry, "link", "")
            title  = getattr(entry, "title", "")
            pub    = getattr(getattr(entry, "source", None), "title", "")
            summ   = getattr(entry, "summary", "")
            if not title or not link:
                continue
            records.append({
                "link": link, "title": title, "date": dt_str,
                "publisher": pub, "summary": summ,
            })
        return records
    except Exception as e:
        log.debug(f"[Google News] {name} {after}→{before}: {e}")
        return []


def fetch_google_news(ticker: str,
                      start: str = START_DATE,
                      end: str   = END_DATE) -> list[dict]:
    """
    Fetch Google News RSS in monthly backward windows from end → start.
    Returns news records.
    """
    name = _company_name(ticker)
    all_records: dict[str, dict] = {}

    cur_end  = datetime.strptime(end, "%Y-%m-%d").date()
    cur_start_limit = datetime.strptime(start, "%Y-%m-%d").date()

    windows = []
    cur = cur_end
    while cur >= cur_start_limit:
        w_start = max(cur - timedelta(days=29), cur_start_limit)
        windows.append((w_start.strftime("%Y-%m-%d"), cur.strftime("%Y-%m-%d")))
        cur = w_start - timedelta(days=1)

    log.info(f"[{ticker}] Google News: querying {len(windows)} windows...")

    for after, before in windows:
        recs = _gnews_window_query(name, ticker, after, before)
        for r in recs:
            key = _dedup_url(r["link"])
            if key not in all_records:
                all_records[key] = _make_record(
                    ticker    = ticker,
                    source    = "google_news",
                    date_str  = r["date"],
                    title     = r["title"],
                    url       = r["link"],
                    publisher = r["publisher"],
                    summary   = r["summary"],
                    language  = "pt",
                )

    records = list(all_records.values())
    log.info(f"[{ticker}] Google News: {len(records):,} articles")
    return records


# ════════════════════════════════════════════════════════════════════════════
# TIER 2B: VALOR ECONÔMICO RSS — TOP FINANCIAL NEWSPAPER
# ════════════════════════════════════════════════════════════════════════════
#
# Valor Econômico is Brazil's equivalent of the Financial Times.
# It provides section-level RSS feeds (bolsas, empresas, finanças, etc.)
# and a search RSS: https://valor.globo.com/rss/busca/QUERY/
# Returns ~20 items per feed, refreshed throughout the day.

VALOR_SEARCH_RSS = "https://valor.globo.com/rss/busca/{query}/"
VALOR_SECTION_FEEDS = [
    "https://valor.globo.com/rss/empresas/",
    "https://valor.globo.com/rss/financas/",
    "https://valor.globo.com/rss/mercados/",
    "https://valor.globo.com/rss/brasil/",
]
VALOR_DELAY = 1.5


def fetch_valor_news(ticker: str) -> list[dict]:
    """Fetch Valor Econômico via search RSS + section feeds."""
    name = _company_name(ticker)
    all_records: dict[str, dict] = {}

    # Search RSS
    for query_term in [ticker, name]:
        url = VALOR_SEARCH_RSS.format(query=quote_plus(query_term))
        try:
            time.sleep(VALOR_DELAY)
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                dt_str = _parse_gnews_date(entry)
                link   = getattr(entry, "link", "")
                title  = getattr(entry, "title", "")
                summ   = getattr(entry, "summary", "")
                if not title or not link:
                    continue
                key = _dedup_url(link)
                if key not in all_records:
                    all_records[key] = _make_record(
                        ticker    = ticker,
                        source    = "valor",
                        date_str  = dt_str,
                        title     = title,
                        url       = link,
                        publisher = "Valor Econômico",
                        summary   = summ,
                        language  = "pt",
                    )
        except Exception as e:
            log.debug(f"[{ticker}] Valor RSS {query_term}: {e}")

    log.info(f"[{ticker}] Valor: {len(all_records):,} articles")
    return list(all_records.values())


# ════════════════════════════════════════════════════════════════════════════
# TIER 2C: INFOMONEY RSS — RETAIL INVESTOR SITE
# ════════════════════════════════════════════════════════════════════════════
#
# InfoMoney provides stable per-ticker RSS feeds at:
#   https://www.infomoney.com.br/feeds/company-news/{TICKER}/
# And a general search RSS:
#   https://www.infomoney.com.br/feeds/search/?q={query}
# These are more reliable than scraping HTML pages.

INFOMONEY_TICKER_RSS  = "https://www.infomoney.com.br/feeds/company-news/{ticker}/"
INFOMONEY_SEARCH_RSS  = "https://www.infomoney.com.br/feeds/search/?q={query}"
INFOMONEY_DELAY       = 1.5


def fetch_infomoney_news(ticker: str) -> list[dict]:
    """Fetch InfoMoney via per-ticker RSS feed and search RSS."""
    all_records: dict[str, dict] = {}

    feeds = [
        INFOMONEY_TICKER_RSS.format(ticker=ticker.lower()),
        INFOMONEY_SEARCH_RSS.format(query=quote_plus(_company_name(ticker))),
    ]

    for feed_url in feeds:
        try:
            time.sleep(INFOMONEY_DELAY)
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries:
                dt_str = _parse_gnews_date(entry)
                link   = getattr(entry, "link", "")
                title  = getattr(entry, "title", "")
                summ   = re.sub(r"<[^>]+>", "", getattr(entry, "summary", ""))
                if not title or not link:
                    continue
                key = _dedup_url(link)
                if key not in all_records:
                    all_records[key] = _make_record(
                        ticker    = ticker,
                        source    = "infomoney",
                        date_str  = dt_str,
                        title     = title,
                        url       = link,
                        publisher = "InfoMoney",
                        summary   = summ[:300],
                        language  = "pt",
                    )
        except Exception as e:
            log.debug(f"[{ticker}] InfoMoney RSS: {e}")

    log.info(f"[{ticker}] InfoMoney: {len(all_records):,} articles")
    return list(all_records.values())


# ════════════════════════════════════════════════════════════════════════════
# TIER 2D: EXAME RSS — MAJOR BUSINESS PRESS
# ════════════════════════════════════════════════════════════════════════════
#
# Exame has a search RSS at:
#   https://exame.com/feed/?s={query}
# Returns ~20 most recent results. Good for top-tier tickers.

EXAME_RSS_URL = "https://exame.com/feed/?s={query}"
EXAME_DELAY   = 1.5


def fetch_exame_news(ticker: str) -> list[dict]:
    """Fetch Exame via search RSS feed."""
    all_records: dict[str, dict] = {}
    name = _company_name(ticker)

    for query_term in [ticker, name]:
        url = EXAME_RSS_URL.format(query=quote_plus(query_term))
        try:
            time.sleep(EXAME_DELAY)
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                dt_str = _parse_gnews_date(entry)
                link   = getattr(entry, "link", "")
                title  = getattr(entry, "title", "")
                summ   = re.sub(r"<[^>]+>", "", getattr(entry, "summary", ""))
                if not title or not link:
                    continue
                key = _dedup_url(link)
                if key not in all_records:
                    all_records[key] = _make_record(
                        ticker    = ticker,
                        source    = "exame",
                        date_str  = dt_str,
                        title     = title,
                        url       = link,
                        publisher = "Exame",
                        summary   = summ[:300],
                        language  = "pt",
                    )
        except Exception as e:
            log.debug(f"[{ticker}] Exame RSS {query_term}: {e}")

    log.info(f"[{ticker}] Exame: {len(all_records):,} articles")
    return list(all_records.values())


# ════════════════════════════════════════════════════════════════════════════
# TIER 3A: GDELT DOC 2.0 — GLOBAL NEWS INDEX
# ════════════════════════════════════════════════════════════════════════════
#
# GDELT DOC API provides full-text search across global media since 2017.
# Strong for PT-BR coverage of mid/large caps, weak for small caps.
# Rate limit: ~1 request per 5 seconds (enforced with sleep).
# We slide 3-month windows across the requested date range.

GDELT_DOC_URL              = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_WINDOW_MONTHS        = 3
GDELT_MIN_REQUEST_INTERVAL = 5.2
_GDELT_LAST_REQUEST_AT     = 0.0


def _gdelt_date_windows(start: str, end: str, months: int = GDELT_WINDOW_MONTHS):
    cur = datetime.strptime(start, "%Y-%m-%d").date()
    fin = datetime.strptime(end,   "%Y-%m-%d").date()
    while cur <= fin:
        win_end = min(cur + relativedelta(months=months) - timedelta(days=1), fin)
        yield cur.strftime("%Y%m%d%H%M%S"), win_end.strftime("%Y%m%d%H%M%S")
        cur = win_end + timedelta(days=1)


def _gdelt_wait():
    global _GDELT_LAST_REQUEST_AT
    elapsed = time.monotonic() - _GDELT_LAST_REQUEST_AT
    if elapsed < GDELT_MIN_REQUEST_INTERVAL:
        time.sleep(GDELT_MIN_REQUEST_INTERVAL - elapsed)
    _GDELT_LAST_REQUEST_AT = time.monotonic()


def _gdelt_query_window(query: str, ts_start: str, ts_end: str,
                         maxrecords: int = 250) -> list[dict]:
    _gdelt_wait()
    params = {
        "query":         query,
        "mode":          "ArtList",
        "maxrecords":    maxrecords,
        "startdatetime": ts_start,
        "enddatetime":   ts_end,
        "sort":          "DateDesc",
        "format":        "json",
    }
    try:
        resp = SESSION.get(GDELT_DOC_URL, params=params, timeout=60)
        if resp.status_code != 200:
            log.debug(f"GDELT HTTP {resp.status_code}: {resp.text[:120]}")
            return []
        data = resp.json()
        return data.get("articles", [])
    except Exception as e:
        log.debug(f"GDELT query error: {e}")
        return []


def fetch_gdelt_news(ticker: str,
                     start: str = GDELT_START,
                     end: str   = END_DATE) -> list[dict]:
    """Fetch GDELT news across quarterly windows for a ticker."""
    name  = _company_name(ticker)
    q_pt  = f'("{ticker}" OR "{name}") (Bovespa OR Ibovespa OR bolsa OR ações) sourcelang:Portuguese'
    q_en  = f'("{ticker}" OR "{name}") (Brazil OR Bovespa OR Ibovespa) sourcelang:English'

    all_articles: dict[str, dict] = {}
    windows = list(_gdelt_date_windows(start, end))
    log.info(f"[{ticker}] GDELT: {len(windows)} windows ({start}→{end})...")

    for i, (ts_start, ts_end) in enumerate(windows):
        for query in [q_pt, q_en]:
            for art in _gdelt_query_window(query, ts_start, ts_end):
                url = art.get("url", "")
                if not url or _dedup_url(url) in all_articles:
                    continue
                seendate = art.get("seendate", "")
                try:
                    dt = (datetime.strptime(seendate[:14], "%Y%m%dT%H%M%S")
                          if "T" in seendate
                          else datetime.strptime(seendate[:8], "%Y%m%d"))
                    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt_str = seendate

                lang = "en" if "English" in query else "pt"
                all_articles[_dedup_url(url)] = _make_record(
                    ticker    = ticker,
                    source    = "gdelt",
                    date_str  = dt_str,
                    title     = art.get("title", ""),
                    url       = url,
                    publisher = art.get("domain", ""),
                    language  = lang,
                    gdelt_countries = art.get("sourcecountry", ""),
                )

        if (i + 1) % 5 == 0:
            log.info(f"[{ticker}] GDELT: {i+1}/{len(windows)} windows, "
                     f"{len(all_articles):,} articles so far")

    records = list(all_articles.values())
    log.info(f"[{ticker}] GDELT: {len(records):,} total articles")
    return records


# ════════════════════════════════════════════════════════════════════════════
# TIER 3B: YFINANCE HEADLINES
# ════════════════════════════════════════════════════════════════════════════

def fetch_yfinance_news(ticker: str) -> list[dict]:
    """Pull recent English-language headlines via yfinance (~90 days)."""
    yf_sym = f"{ticker}.SA" if not ticker.endswith(".SA") else ticker
    try:
        news = yf.Ticker(yf_sym).news or []
        records = []
        for item in news:
            pub_ts = item.get("providerPublishTime", 0)
            dt_str = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M:%S")
            records.append(_make_record(
                ticker    = ticker,
                source    = "yfinance",
                date_str  = dt_str,
                title     = item.get("title", ""),
                url       = item.get("link", ""),
                publisher = item.get("publisher", ""),
                summary   = item.get("summary", ""),
                language  = "en",
            ))
        return records
    except Exception as e:
        log.debug(f"[{ticker}] yfinance news: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# OPTIONAL: NEWSAPI
# ════════════════════════════════════════════════════════════════════════════

def fetch_newsapi_news(ticker: str, api_key: str) -> list[dict]:
    """NewsAPI — free tier: last 30 days. Paid: full archive."""
    if not api_key:
        return []
    name = _company_name(ticker)
    params = {
        "q":        f'"{ticker}" OR "{name}" bolsa B3',
        "language": "pt",
        "sortBy":   "publishedAt",
        "pageSize": 100,
        "apiKey":   api_key,
    }
    try:
        resp = SESSION.get("https://newsapi.org/v2/everything",
                           params=params, timeout=20)
        resp.raise_for_status()
        return [
            _make_record(
                ticker    = ticker,
                source    = "newsapi",
                date_str  = art.get("publishedAt", "")[:19],
                title     = art.get("title", ""),
                url       = art.get("url", ""),
                publisher = art.get("source", {}).get("name", ""),
                summary   = art.get("description", ""),
                language  = "pt",
            )
            for art in resp.json().get("articles", [])
        ]
    except Exception as e:
        log.debug(f"[{ticker}] NewsAPI: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# NEWS AGGREGATOR — combines all sources, deduplicates, date-filters
# ════════════════════════════════════════════════════════════════════════════

def fetch_all_news(ticker: str,
                   start: str        = START_DATE,
                   end: str          = END_DATE,
                   newsapi_key: str  = "",
                   use_gdelt: bool   = True,
                   workers: int      = 3) -> pd.DataFrame:
    """
    Aggregate news from all sources for a single ticker.

    Sources run in parallel (RSS/CVM) while GDELT runs sequentially
    due to its strict rate limit.

    Returns a DataFrame with NEWS_COLUMNS, date-filtered to [start, end],
    sorted newest-first, deduplicated by URL.
    """
    all_records: list[dict] = []

    # ── Parallel fetch: fast sources (RSS + CVM + yfinance) ──────────────
    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.warning(f"Source fetch error ({fn.__name__}): {e}")
            return []

    fast_tasks = {
        "cvm":        (fetch_cvm_filings,    ticker, start, end),
        "google":     (fetch_google_news,    ticker, start, end),
        "valor":      (fetch_valor_news,     ticker),
        "infomoney":  (fetch_infomoney_news, ticker),
        "exame":      (fetch_exame_news,     ticker),
        "yfinance":   (fetch_yfinance_news,  ticker),
    }

    with ThreadPoolExecutor(max_workers=min(workers, len(fast_tasks))) as pool:
        futures = {
            pool.submit(_safe, fn, *args): name
            for name, (fn, *args) in fast_tasks.items()
        }
        for future in as_completed(futures):
            src_name = futures[future]
            try:
                all_records.extend(future.result())
            except Exception as e:
                log.warning(f"[{ticker}] {src_name} future error: {e}")

    # Optional NewsAPI
    if newsapi_key:
        all_records.extend(_safe(fetch_newsapi_news, ticker, newsapi_key))

    # ── Sequential: GDELT (rate-limited) ─────────────────────────────────
    if use_gdelt:
        all_records.extend(_safe(fetch_gdelt_news, ticker, start, end))

    if not all_records:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    df = pd.DataFrame(all_records)
    for col in NEWS_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Normalise dates and filter to requested range
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    mask = (
        (df["date"] >= pd.Timestamp(start)) &
        (df["date"] <= pd.Timestamp(end))
    )
    df = df[mask | df["date"].isna()]   # keep undated CVM records

    df = (df[NEWS_COLUMNS]
          .drop_duplicates(subset=["url"])
          .sort_values("date", ascending=False, na_position="last")
          .reset_index(drop=True))
    return df


# ════════════════════════════════════════════════════════════════════════════
# PRICE DATA — yfinance
# ════════════════════════════════════════════════════════════════════════════

def fetch_price_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    yf_sym = f"{ticker}.SA" if not ticker.endswith(".SA") else ticker
    try:
        df = yf.download(yf_sym, start=start, end=end,
                         auto_adjust=True, progress=False, timeout=30)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.rename(columns={"index": "Date"}, inplace=True)
        df["Date"]   = pd.to_datetime(df["Date"]).dt.date
        df["Ticker"] = ticker
        df["Source"] = "yfinance"
        log.info(f"[{ticker}] yfinance: {len(df):,} rows "
                 f"({df['Date'].min()} → {df['Date'].max()})")
        return df
    except Exception as e:
        log.warning(f"[{ticker}] yfinance price: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# PRICE DATA — B3 COTAHIST (official annual files, fallback)
# ════════════════════════════════════════════════════════════════════════════

COTAHIST_COLUMNS = [
    ("tipo_registro",        2), ("data_pregao",          8),
    ("cod_bdi",              2), ("cod_negociacao",       12),
    ("tipo_mercado",         3), ("nome_resumido",        12),
    ("especificacao_papel", 10), ("prazo_dias_mto",        3),
    ("moeda_referencia",     4), ("preco_abertura",       13),
    ("preco_maximo",        13), ("preco_minimo",         13),
    ("preco_medio",         13), ("preco_ultimo",         13),
    ("preco_oferta_compra", 13), ("preco_oferta_venda",   13),
    ("numero_negocios",      5), ("quantidade_papeis",    18),
    ("volume_total",        18), ("preco_exercicio",      13),
    ("indicador_correcao",   1), ("data_vencimento",       8),
    ("fator_cotacao",        7), ("preco_exercicio_pts",  13),
    ("codigo_isin",         12), ("num_distribuicao",      3),
]
COTAHIST_URL = (
    "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
)


def download_cotahist_year(year: int) -> pd.DataFrame | None:
    url = COTAHIST_URL.format(year=year)
    try:
        log.info(f"[COTAHIST] Downloading {year}...")
        resp = SESSION.get(url, timeout=180, stream=True)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            fname = [n for n in z.namelist() if n.endswith(".TXT")][0]
            raw   = z.read(fname).decode("latin-1")
        rows = []
        for line in raw.splitlines():
            if not line.startswith("01"):
                continue
            rec, pos = {}, 0
            for name, width in COTAHIST_COLUMNS:
                rec[name] = line[pos:pos+width].strip()
                pos += width
            rows.append(rec)
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df = df[df["cod_bdi"].isin(["02", "12"])]
        for col in ["preco_abertura", "preco_maximo", "preco_minimo",
                    "preco_ultimo", "preco_medio"]:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100
        df["volume_total"]      = pd.to_numeric(df["volume_total"],      errors="coerce") / 100
        df["numero_negocios"]   = pd.to_numeric(df["numero_negocios"],   errors="coerce")
        df["quantidade_papeis"] = pd.to_numeric(df["quantidade_papeis"], errors="coerce")
        df["data_pregao"]       = pd.to_datetime(df["data_pregao"], format="%Y%m%d", errors="coerce")
        df["cod_negociacao"]    = df["cod_negociacao"].str.strip()
        df.rename(columns={
            "data_pregao":    "Date",
            "cod_negociacao": "Ticker",
            "preco_abertura": "Open",
            "preco_maximo":   "High",
            "preco_minimo":   "Low",
            "preco_ultimo":   "Close",
            "volume_total":   "Volume",
        }, inplace=True)
        df["Source"] = "cotahist"
        log.info(f"[COTAHIST] {year}: {len(df):,} rows")
        return df
    except Exception as e:
        log.error(f"[COTAHIST] {year}: {e}")
        return None


def fetch_cotahist_range(start_year: int, end_year: int,
                          tickers: list[str]) -> dict[str, pd.DataFrame]:
    results    = {t: [] for t in tickers}
    ticker_set = set(tickers)
    for year in range(start_year, end_year + 1):
        cache_path = COTAHIST_DIR / f"cotahist_{year}.parquet"
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
            except Exception:
                df = download_cotahist_year(year)
        else:
            df = download_cotahist_year(year)
        if df is None:
            continue
        if not cache_path.exists():
            try:
                df.to_parquet(cache_path, index=False)
            except Exception:
                pass
        for ticker, grp in df[df["Ticker"].isin(ticker_set)].groupby("Ticker"):
            results[ticker].append(grp)
    final = {}
    for ticker, frames in results.items():
        if frames:
            combined = pd.concat(frames, ignore_index=True).sort_values("Date")
            combined["Date"] = combined["Date"].dt.date
            final[ticker] = combined
    return final


# ════════════════════════════════════════════════════════════════════════════
# FUNDAMENTALS
# ════════════════════════════════════════════════════════════════════════════

FUNDAMENTAL_KEYS = [
    "shortName", "longName", "sector", "industry", "country",
    "marketCap", "enterpriseValue", "trailingPE", "forwardPE",
    "priceToBook", "priceToSalesTrailing12Months", "dividendYield",
    "trailingEps", "revenuePerShare", "totalRevenue", "grossProfits",
    "ebitda", "totalDebt", "totalCash", "debtToEquity",
    "returnOnAssets", "returnOnEquity", "operatingMargins",
    "profitMargins", "currentRatio", "quickRatio",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "beta",
    "sharesOutstanding", "floatShares", "heldPercentInsiders",
    "fullTimeEmployees", "website", "currency",
]


def fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(f"{ticker}.SA").info or {}
        return {k: info.get(k) for k in FUNDAMENTAL_KEYS}
    except Exception as e:
        log.warning(f"[{ticker}] Fundamentals: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════════════
# TICKER ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def scrape_ticker(ticker: str,
                  start: str          = START_DATE,
                  end: str            = END_DATE,
                  newsapi_key: str    = "",
                  skip_existing: bool = True,
                  use_gdelt: bool     = True,
                  workers: int        = 3,
                  cotahist_data: dict | None = None) -> dict:

    result = {
        "ticker": ticker,
        "price_rows": 0, "news_rows": 0,
        "cvm_rows": 0,   "gdelt_rows": 0,
        "price_source": "", "status": "ok",
    }

    # ── Prices ────────────────────────────────────────────────────────────
    price_path = PRICES_DIR / f"{ticker}.csv"
    if skip_existing and price_path.exists():
        n = len(pd.read_csv(price_path))
        log.info(f"[{ticker}] Price cache hit ({n:,} rows)")
        result.update(price_rows=n, price_source="cache")
    else:
        df_price = fetch_price_yfinance(ticker, start, end)
        if (df_price is None or df_price.empty) and cotahist_data:
            df_price = cotahist_data.get(ticker)
            if df_price is not None and not df_price.empty:
                result["price_source"] = "cotahist"
        if df_price is not None and not df_price.empty:
            df_price.to_csv(price_path, index=False)
            result.update(
                price_rows   = len(df_price),
                price_source = result.get("price_source") or "yfinance",
            )
        else:
            result["status"] = "no_price_data"

    # ── News (all sources) ─────────────────────────────────────────────────
    news_path = NEWS_DIR / f"{ticker}_news.csv"
    if skip_existing and news_path.exists():
        existing = pd.read_csv(news_path)
        n = len(existing)
        cvm_n   = int((existing["source"].str.startswith("cvm")).sum()) \
                  if "source" in existing.columns else 0
        gdelt_n = int((existing["source"] == "gdelt").sum()) \
                  if "source" in existing.columns else 0
        log.info(f"[{ticker}] News cache hit "
                 f"({n:,} rows, {cvm_n:,} CVM, {gdelt_n:,} GDELT)")
        result.update(news_rows=n, cvm_rows=cvm_n, gdelt_rows=gdelt_n)
    else:
        df_news = fetch_all_news(
            ticker,
            start       = start,
            end         = end,
            newsapi_key = newsapi_key,
            use_gdelt   = use_gdelt,
            workers     = workers,
        )
        if not df_news.empty:
            df_news.to_csv(news_path, index=False)
            cvm_n   = int(df_news["source"].str.startswith("cvm").sum())
            gdelt_n = int((df_news["source"] == "gdelt").sum())
            result.update(
                news_rows  = len(df_news),
                cvm_rows   = cvm_n,
                gdelt_rows = gdelt_n,
            )
            log.info(
                f"[{ticker}] Saved {len(df_news):,} articles "
                f"({cvm_n:,} CVM, {gdelt_n:,} GDELT) → {news_path}"
            )
        else:
            log.warning(f"[{ticker}] No news found from any source")

    # ── Fundamentals ──────────────────────────────────────────────────────
    fund_path = FUND_DIR / f"{ticker}.json"
    if not (skip_existing and fund_path.exists()):
        info = fetch_fundamentals(ticker)
        if info:
            fund_path.write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    return result


# ════════════════════════════════════════════════════════════════════════════
# MAIN RUN
# ════════════════════════════════════════════════════════════════════════════

def run(tickers: list[str],
        start: str          = START_DATE,
        end: str            = END_DATE,
        newsapi_key: str    = "",
        delay: float        = 1.0,
        skip_existing: bool = True,
        use_gdelt: bool     = True,
        use_cotahist: bool  = False,
        workers: int        = 3) -> pd.DataFrame:

    setup_dirs()
    log.info("=" * 70)
    log.info(f"B3 Scraper v4  |  {len(tickers)} tickers  |  {start} → {end}")
    log.info(f"GDELT:    {'ON' if use_gdelt else 'OFF'}")
    log.info(f"COTAHIST: {'ON' if use_cotahist else 'OFF'}")
    log.info(f"Workers:  {workers} (parallel RSS/CVM fetching)")
    log.info("=" * 70)

    cotahist_data = None
    if use_cotahist:
        sy, ey = int(start[:4]), int(end[:4])
        log.info(f"Downloading COTAHIST {sy}–{ey}...")
        cotahist_data = fetch_cotahist_range(sy, ey, tickers)

    summary, failed = [], []

    for i, ticker in enumerate(tickers, 1):
        log.info(f"── [{i:>3}/{len(tickers)}] {ticker} ──────────────────────────")
        try:
            row = scrape_ticker(
                ticker,
                start         = start,
                end           = end,
                newsapi_key   = newsapi_key,
                skip_existing = skip_existing,
                use_gdelt     = use_gdelt,
                workers       = workers,
                cotahist_data = cotahist_data,
            )
            summary.append(row)
        except Exception as e:
            log.error(f"[{ticker}] Unhandled exception: {e}", exc_info=True)
            summary.append({
                "ticker": ticker, "price_rows": 0, "news_rows": 0,
                "cvm_rows": 0, "gdelt_rows": 0,
                "status": f"error: {e}",
            })
            failed.append(ticker)

        time.sleep(delay)

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    ok = len(tickers) - len(failed)
    log.info("=" * 70)
    log.info(f"Done  |  {ok}/{len(tickers)} OK  |  {len(failed)} failed")
    if failed:
        log.warning(f"Failed tickers: {failed}")
    log.info(f"Outputs → {OUTPUT_DIR}/")
    log.info("=" * 70)
    return df_summary


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="B3 Scraper v4 — prices + layered news (CVM + RSS + GDELT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tickers",     nargs="+", default=None,
                   help="Specific tickers (default: all 110)")
    p.add_argument("--limit",       type=int,  default=None,
                   help="Only first N tickers (for testing)")
    p.add_argument("--start",       default=START_DATE,
                   help=f"Start date YYYY-MM-DD (default {START_DATE})")
    p.add_argument("--end",         default=END_DATE,
                   help=f"End date YYYY-MM-DD (default today)")
    p.add_argument("--newsapi-key", default="", metavar="KEY",
                   help="Optional NewsAPI.org key for extra coverage")
    p.add_argument("--delay",       type=float, default=1.0,
                   help="Seconds between tickers (default 1.0)")
    p.add_argument("--workers",     type=int,   default=3,
                   help="Parallel workers for RSS/CVM sources (default 3)")
    p.add_argument("--no-skip",     action="store_true",
                   help="Re-download even if output files already exist")
    p.add_argument("--no-gdelt",    action="store_true",
                   help="Skip GDELT (faster but fewer deep-archive articles)")
    p.add_argument("--cotahist",    action="store_true",
                   help="Download official B3 COTAHIST files as price fallback")
    args = p.parse_args()

    tickers = args.tickers or B3_TICKERS
    if args.limit:
        tickers = tickers[:args.limit]

    df = run(
        tickers       = tickers,
        start         = args.start,
        end           = args.end,
        newsapi_key   = args.newsapi_key,
        delay         = args.delay,
        skip_existing = not args.no_skip,
        use_gdelt     = not args.no_gdelt,
        use_cotahist  = args.cotahist,
        workers       = args.workers,
    )

    print("\n📊 Run Summary:")
    print(df.to_string(index=False))