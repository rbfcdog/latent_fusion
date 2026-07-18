#!/usr/bin/env python3
"""
B3 Brazilian Stock Exchange — Historical Data & News Scraper
============================================================
Fetches up to 20 years of OHLCV price data + news for 100+ B3 tickers.

News sources (all free, no key required unless noted):
  1. GDELT Project  — global news index since 2017 via DOC 2.0, PT-BR coverage
  2. yfinance       — recent headlines (~90 days)
  3. Infomoney      — Brazilian financial news, per-ticker pages
  4. NewsAPI        — optional free key (newsapi.org), deeper archive

Price sources:
  1. yfinance (Yahoo Finance) — adjusted OHLCV, ~20 years
  2. B3 COTAHIST              — official raw files, 1995-present (--cotahist flag)

Output layout:
  data/prices/<TICKER>.csv          Daily OHLCV
  data/news/<TICKER>_news.csv       Headlines: date, title, url, source, tone, language
  data/fundamentals/<TICKER>.json   Key financial metrics
  data/summary.csv                  Per-ticker run statistics
  b3_scraper.log                    Full execution log

Quick start:
  pip install yfinance pandas requests pyarrow
  python b3_scraper.py                         # all 110 tickers
  python b3_scraper.py --limit 5              # test with 5 tickers
  python b3_scraper.py --tickers PETR4 VALE3  # specific tickers
  python b3_scraper.py --no-gdelt             # skip GDELT (faster)
  python b3_scraper.py --cotahist             # add official B3 price files
  python b3_scraper.py --newsapi-key KEY      # even more news via NewsAPI
"""

import io
import json
import time
import zipfile
import logging
import argparse
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import quote_plus

import requests
import pandas as pd
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
# TICKER LIST — 110 liquid B3 stocks
# ════════════════════════════════════════════════════════════════════════════
B3_TICKERS = [
    # ── Mega caps / Ibovespa core ─────────────────────────────────────────
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
    # ── Education ────────────────────────────────────────────────────────
    "COGN3", "YDUQ3", "ANIM3", "SEER3",
    # ── Pulp & Paper ──────────────────────────────────────────────────────
    "KLBN11", "DXCO3",
    # ── Sanitation ────────────────────────────────────────────────────────
    "SAPR11", "CSMG3",
    # ── Other liquid tickers ──────────────────────────────────────────────
    "VVAR3", "OIBR3", "GMAT3", "INTB3", "CXSE3", "WIZS3",
]
B3_TICKERS = list(dict.fromkeys(B3_TICKERS))

# Map tickers → company name fragments used in news queries
# If a ticker isn't here, the script falls back to stripping digits.
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

START_DATE   = "2004-01-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")
GDELT_START  = "2017-01-01"   # DOC 2.0 historical search starts here

OUTPUT_DIR   = Path("data")
PRICES_DIR   = OUTPUT_DIR / "prices"
NEWS_DIR     = OUTPUT_DIR / "news"
FUND_DIR     = OUTPUT_DIR / "fundamentals"
COTAHIST_DIR = OUTPUT_DIR / "cotahist"


# ════════════════════════════════════════════════════════════════════════════
# HTTP SESSION
# ════════════════════════════════════════════════════════════════════════════

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; B3DataScraper/3.0)"})
    return s

SESSION = make_session()


def setup_dirs():
    for d in [PRICES_DIR, NEWS_DIR, FUND_DIR, COTAHIST_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _company_name(ticker: str) -> str:
    """Return a human-readable company name for news queries."""
    return TICKER_NAMES.get(ticker, ticker.rstrip("0123456789"))


# ════════════════════════════════════════════════════════════════════════════
# GDELT NEWS MODULE
# ════════════════════════════════════════════════════════════════════════════
#
# GDELT usage here:
#
#   1. DOC 2.0 API  — full-text article search, returns up to 250 articles
#      per query (timespan max 3 months). We loop over quarterly windows
#      to cover the full date range → potentially thousands of articles.
#      Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
#
# The old GKG GeoJSON API and bulk GKG files are separate products; there is
# no working /api/v2/gkg/gkg ArtList endpoint for per-article enrichment here.
# Rate limit: GDELT asks for roughly one request every 5 seconds.
#
# Columns added by GDELT that are not in other sources:
#   gdelt_tone       float  overall sentiment (positive = +, negative = -)
#   gdelt_themes     str    pipe-separated GDELT themes (e.g. ECON_INFLATION)
#   gdelt_countries  str    pipe-separated ISO2 country codes mentioned
#   seendate         str    GDELT's own timestamp (YYYYMMDDTHHMMSSZ)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_GKG_URL = ""

# How many months per GDELT query window (max ~3 months for DOC API)
GDELT_WINDOW_MONTHS = 3
GDELT_MIN_REQUEST_INTERVAL = 5.1
_GDELT_LAST_REQUEST_AT = 0.0

# GDELT DOC API field map (ArtList format)
GDELT_DOC_FIELDS = {
    "url":        "url",
    "title":      "title",
    "seendate":   "seendate",
    "socialimage":"socialimage",
    "domain":     "domain",
    "language":   "language",
    "sourcecountry": "sourcecountry",
}


def _gdelt_date_windows(start: str, end: str, months: int = GDELT_WINDOW_MONTHS):
    """Generate (window_start, window_end) tuples covering [start, end]."""
    from dateutil.relativedelta import relativedelta
    cur = datetime.strptime(start, "%Y-%m-%d").date()
    fin = datetime.strptime(end,   "%Y-%m-%d").date()
    while cur <= fin:
        win_end = min(cur + relativedelta(months=months) - timedelta(days=1), fin)
        yield cur.strftime("%Y%m%d%H%M%S"), win_end.strftime("%Y%m%d%H%M%S")
        cur = win_end + timedelta(days=1)


def _wait_for_gdelt_rate_limit() -> None:
    """GDELT returns 429s if requests are much faster than ~1 per 5 seconds."""
    global _GDELT_LAST_REQUEST_AT
    elapsed = time.monotonic() - _GDELT_LAST_REQUEST_AT
    if elapsed < GDELT_MIN_REQUEST_INTERVAL:
        time.sleep(GDELT_MIN_REQUEST_INTERVAL - elapsed)
    _GDELT_LAST_REQUEST_AT = time.monotonic()


def _gdelt_json_response(url: str, params: dict, endpoint: str) -> dict | None:
    _wait_for_gdelt_rate_limit()
    try:
        resp = SESSION.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            log.warning(f"GDELT {endpoint}: HTTP {resp.status_code}: {resp.text[:160]}")
            return None
        try:
            return resp.json()
        except ValueError:
            log.warning(f"GDELT {endpoint}: non-JSON response: {resp.text[:160]}")
            return None
    except Exception as e:
        log.warning(f"GDELT {endpoint} query error: {e}")
        return None


def _gdelt_doc_query(query: str, start_ts: str, end_ts: str,
                     mode: str = "ArtList", maxrecords: int = 250) -> list[dict]:
    """
    Call GDELT DOC 2.0 API for one time window.
    Returns a list of article dicts.
    """
    params = {
        "query":      query,
        "mode":       mode,
        "maxrecords": maxrecords,
        "startdatetime": start_ts,
        "enddatetime":   end_ts,
        "sort":       "DateDesc",
        "format":     "json",
    }
    data = _gdelt_json_response(GDELT_DOC_URL, params, "DOC")
    return data.get("articles", []) if data else []


def _gdelt_gkg_query(query: str, start_ts: str, end_ts: str,
                     maxrecords: int = 250) -> list[dict]:
    """
    Placeholder for optional GKG enrichment.

    GDELT does not expose a compatible /api/v2/gkg/gkg ArtList endpoint, so
    this returns no records unless a working enrichment source is added later.
    """
    if not GDELT_GKG_URL:
        return []
    params = {
        "query":      query,
        "mode":       "artlist",
        "maxrecords": maxrecords,
        "startdatetime": start_ts,
        "enddatetime":   end_ts,
        "sort":       "DateDesc",
        "format":     "json",
    }
    data = _gdelt_json_response(GDELT_GKG_URL, params, "GKG")
    return data.get("gkgrecords", []) if data else []


def _build_gdelt_query(ticker: str, lang_filter: bool = True) -> str:
    """
    Build a GDELT search query for a B3 ticker.
    Combines ticker symbol + company name + Brazilian context.
    Optionally restrict to Portuguese/English sources.
    """
    name = _company_name(ticker)
    parts = [f'"{ticker}"', f'"{name}"']
    # Keep terms unquoted here. GDELT rejects quoted short phrases like "B3"
    # with a text response, which otherwise looks like an empty result set.
    context = '(Bovespa OR Ibovespa OR bolsa OR ações OR Brasil)'
    query = f"({' OR '.join(parts)}) {context}"
    if lang_filter:
        query += ' sourcelang:Portuguese'
    return query


def fetch_gdelt_news(ticker: str,
                     start: str = GDELT_START,
                     end: str   = END_DATE,
                     use_gkg: bool = False) -> list[dict]:
    """
    Fetch all GDELT news for a ticker across its full history.

    Strategy:
      - Slide a 3-month window from `start` to `end`
      - Query DOC API articles
      - De-duplicate by URL
      - De-duplicate by URL

    Returns a list of record dicts ready for DataFrame conversion.
    """
    query      = _build_gdelt_query(ticker, lang_filter=True)
    query_en   = _build_gdelt_query(ticker, lang_filter=False)  # English sources too

    all_articles: dict[str, dict] = {}  # url → record

    windows = list(_gdelt_date_windows(start, end))
    log.info(f"[{ticker}] GDELT: querying {len(windows)} windows "
             f"({start} → {end}) ...")

    for i, (ts_start, ts_end) in enumerate(windows):
        # ── DOC API: Portuguese sources ──────────────────────────────────
        arts = _gdelt_doc_query(query, ts_start, ts_end)
        # ── DOC API: English sources (less noise, still useful) ──────────
        arts_en = _gdelt_doc_query(query_en + " sourcelang:English",
                                   ts_start, ts_end, maxrecords=50)
        arts += arts_en

        new_doc_urls = []
        for art in arts:
            url = art.get("url", "")
            if not url or url in all_articles:
                continue
            new_doc_urls.append(url)
            seendate = art.get("seendate", "")
            try:
                dt = datetime.strptime(seendate[:14], "%Y%m%dT%H%M%S") \
                     if "T" in seendate else \
                     datetime.strptime(seendate[:8], "%Y%m%d")
                date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                date_str = seendate

            all_articles[url] = {
                "ticker":          ticker,
                "source":          "gdelt",
                "date":            date_str,
                "title":           art.get("title", ""),
                "url":             url,
                "publisher":       art.get("domain", ""),
                "summary":         "",
                "language":        art.get("language", ""),
                "gdelt_tone":      None,
                "gdelt_themes":    "",
                "gdelt_countries": art.get("sourcecountry", ""),
            }

        # ── GKG API: enrich with tone + themes ───────────────────────────
        if use_gkg and new_doc_urls:
            gkg_records = _gdelt_gkg_query(query, ts_start, ts_end)
            for rec in gkg_records:
                url = rec.get("DocumentIdentifier", "")
                if url in all_articles:
                    # Tone: GKG tone field is "tone,pos,neg,polarity,arv,self"
                    tone_str = rec.get("Tone", "")
                    try:
                        tone_val = float(tone_str.split(",")[0])
                    except Exception:
                        tone_val = None
                    all_articles[url]["gdelt_tone"]    = tone_val
                    all_articles[url]["gdelt_themes"]  = rec.get("Themes", "")
                    all_articles[url]["gdelt_countries"] = rec.get("Locations", "")

        if (i + 1) % 10 == 0:
            log.info(f"[{ticker}] GDELT: {i+1}/{len(windows)} windows done, "
                     f"{len(all_articles):,} articles so far")

    records = list(all_articles.values())
    log.info(f"[{ticker}] GDELT: total {len(records):,} unique articles")
    return records


# ════════════════════════════════════════════════════════════════════════════
# OTHER NEWS SOURCES
# ════════════════════════════════════════════════════════════════════════════

def fetch_yfinance_news(ticker: str) -> list[dict]:
    """Pull recent headlines from yfinance (~last 90 days)."""
    yf_sym = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
    try:
        news = yf.Ticker(yf_sym).news or []
        records = []
        for item in news:
            records.append({
                "ticker":          ticker,
                "source":          "yfinance",
                "date":            datetime.fromtimestamp(
                                       item.get("providerPublishTime", 0)
                                   ).strftime("%Y-%m-%d %H:%M:%S"),
                "title":           item.get("title", ""),
                "url":             item.get("link", ""),
                "publisher":       item.get("publisher", ""),
                "summary":         item.get("summary", ""),
                "language":        "en",
                "gdelt_tone":      None,
                "gdelt_themes":    "",
                "gdelt_countries": "",
            })
        return records
    except Exception as e:
        log.warning(f"[{ticker}] yfinance news: {e}")
        return []


class _InfomoneyParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._in_article = False
        self._current: dict = {}
        self._capture = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self._in_article = True
            self._current = {}
        if self._in_article:
            if tag == "a" and "href" in attrs and "url" not in self._current:
                self._current["url"] = attrs["href"]
                self._capture = True
            if tag == "time" and "datetime" in attrs:
                self._current["date"] = attrs["datetime"][:19]

    def handle_endtag(self, tag):
        if tag == "a":
            self._capture = False
        if tag == "article":
            self._in_article = False
            if self._current.get("url") and self._current.get("title"):
                self.results.append(self._current.copy())
            self._current = {}

    def handle_data(self, data):
        if self._capture and data.strip() and "title" not in self._current:
            self._current["title"] = data.strip()


def fetch_infomoney_news(ticker: str) -> list[dict]:
    url = f"https://www.infomoney.com.br/mercados/acoes/{ticker.lower()}/"
    records = []
    try:
        resp = SESSION.get(url, timeout=20)
        if resp.status_code != 200:
            return []
        parser = _InfomoneyParser()
        parser.feed(resp.text)
        for item in parser.results[:100]:
            records.append({
                "ticker":          ticker,
                "source":          "infomoney",
                "date":            item.get("date", ""),
                "title":           item.get("title", ""),
                "url":             item.get("url", ""),
                "publisher":       "Infomoney",
                "summary":         "",
                "language":        "pt",
                "gdelt_tone":      None,
                "gdelt_themes":    "",
                "gdelt_countries": "BR",
            })
    except Exception as e:
        log.warning(f"[{ticker}] Infomoney: {e}")
    return records


def fetch_newsapi_news(ticker: str, api_key: str) -> list[dict]:
    """NewsAPI — free key = last 30 days; paid = full archive."""
    if not api_key:
        return []
    name  = _company_name(ticker)
    query = f'"{ticker}" OR "{name}" bolsa B3 ações'
    params = {
        "q":        query,
        "language": "pt",
        "sortBy":   "publishedAt",
        "pageSize": 100,
        "apiKey":   api_key,
    }
    records = []
    try:
        resp = SESSION.get("https://newsapi.org/v2/everything",
                           params=params, timeout=20)
        resp.raise_for_status()
        for art in resp.json().get("articles", []):
            records.append({
                "ticker":          ticker,
                "source":          "newsapi",
                "date":            art.get("publishedAt", "")[:19],
                "title":           art.get("title", ""),
                "url":             art.get("url", ""),
                "publisher":       art.get("source", {}).get("name", ""),
                "summary":         art.get("description", ""),
                "language":        "pt",
                "gdelt_tone":      None,
                "gdelt_themes":    "",
                "gdelt_countries": "BR",
            })
    except Exception as e:
        log.warning(f"[{ticker}] NewsAPI: {e}")
    return records


# ════════════════════════════════════════════════════════════════════════════
# NEWS AGGREGATOR
# ════════════════════════════════════════════════════════════════════════════

NEWS_COLUMNS = [
    "ticker", "source", "date", "title", "url",
    "publisher", "summary", "language",
    "gdelt_tone", "gdelt_themes", "gdelt_countries",
]


def fetch_all_news(ticker: str,
                   newsapi_key: str = "",
                   use_gdelt: bool = True,
                   gdelt_start: str = GDELT_START,
                   gdelt_end: str   = END_DATE) -> pd.DataFrame:
    records = []

    # GDELT first (most records)
    if use_gdelt:
        records += fetch_gdelt_news(ticker, start=gdelt_start, end=gdelt_end)

    # Supplement with other sources
    records += fetch_yfinance_news(ticker)
    records += fetch_infomoney_news(ticker)

    if newsapi_key:
        records += fetch_newsapi_news(ticker, newsapi_key)

    if not records:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    df = pd.DataFrame(records)
    # Ensure all expected columns exist
    for col in NEWS_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = (df[NEWS_COLUMNS]
            .drop_duplicates(subset=["url"])
            .sort_values("date", ascending=False)
            .reset_index(drop=True))
    return df


# ════════════════════════════════════════════════════════════════════════════
# PRICE DATA — yfinance
# ════════════════════════════════════════════════════════════════════════════

def fetch_price_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    yf_sym = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
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
# PRICE DATA — B3 COTAHIST (official annual files)
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
COTAHIST_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"


def download_cotahist_year(year: int) -> pd.DataFrame | None:
    url = COTAHIST_URL.format(year=year)
    try:
        log.info(f"[COTAHIST] Downloading {year}...")
        resp = SESSION.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            fname = [n for n in z.namelist() if n.endswith(".TXT")][0]
            raw = z.read(fname).decode("latin-1")

        rows, colnames = [], [n for n, _ in COTAHIST_COLUMNS]
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

        for col in ["preco_abertura","preco_maximo","preco_minimo","preco_ultimo","preco_medio"]:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100
        df["volume_total"]      = pd.to_numeric(df["volume_total"],      errors="coerce") / 100
        df["numero_negocios"]   = pd.to_numeric(df["numero_negocios"],   errors="coerce")
        df["quantidade_papeis"] = pd.to_numeric(df["quantidade_papeis"], errors="coerce")
        df["data_pregao"]       = pd.to_datetime(df["data_pregao"], format="%Y%m%d", errors="coerce")
        df["cod_negociacao"]    = df["cod_negociacao"].str.strip()

        df.rename(columns={"data_pregao":"Date","cod_negociacao":"Ticker",
                            "preco_abertura":"Open","preco_maximo":"High",
                            "preco_minimo":"Low","preco_ultimo":"Close",
                            "volume_total":"Volume"}, inplace=True)
        df["Source"] = "cotahist"
        log.info(f"[COTAHIST] {year}: {len(df):,} rows")
        return df
    except Exception as e:
        log.error(f"[COTAHIST] {year}: {e}")
        return None


def fetch_cotahist_range(start_year: int, end_year: int,
                         tickers: list[str]) -> dict[str, pd.DataFrame]:
    results = {t: [] for t in tickers}
    ticker_set = set(tickers)
    for year in range(start_year, end_year + 1):
        cache = COTAHIST_DIR / f"cotahist_{year}.parquet"
        df = pd.read_parquet(cache) if cache.exists() else download_cotahist_year(year)
        if df is None:
            continue
        if not cache.exists():
            df.to_parquet(cache, index=False)
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
    "shortName","longName","sector","industry","country",
    "marketCap","enterpriseValue","trailingPE","forwardPE",
    "priceToBook","priceToSalesTrailing12Months","dividendYield",
    "trailingEps","revenuePerShare","totalRevenue","grossProfits",
    "ebitda","totalDebt","totalCash","debtToEquity",
    "returnOnAssets","returnOnEquity","operatingMargins",
    "profitMargins","currentRatio","quickRatio",
    "fiftyTwoWeekHigh","fiftyTwoWeekLow","beta",
    "sharesOutstanding","floatShares","heldPercentInsiders",
    "fullTimeEmployees","website","currency",
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
                  newsapi_key: str  = "",
                  skip_existing: bool = True,
                  use_gdelt: bool   = True,
                  gdelt_start: str  = GDELT_START,
                  start: str        = START_DATE,
                  end: str          = END_DATE,
                  cotahist_data: dict | None = None) -> dict:

    result = {"ticker": ticker, "price_rows": 0, "news_rows": 0,
              "gdelt_rows": 0, "price_source": "", "status": "ok"}

    # ── Prices ───────────────────────────────────────────────────────────
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
            result.update(price_rows=len(df_price),
                          price_source=result.get("price_source") or "yfinance")
        else:
            result["status"] = "no_price_data"

    # ── News (GDELT + others) ─────────────────────────────────────────────
    news_path = NEWS_DIR / f"{ticker}_news.csv"
    if skip_existing and news_path.exists():
        existing = pd.read_csv(news_path)
        n = len(existing)
        gdelt_n = len(existing[existing["source"] == "gdelt"]) if "source" in existing.columns else 0
        log.info(f"[{ticker}] News cache hit ({n:,} rows, {gdelt_n:,} from GDELT)")
        result.update(news_rows=n, gdelt_rows=gdelt_n)
    else:
        df_news = fetch_all_news(
            ticker,
            newsapi_key=newsapi_key,
            use_gdelt=use_gdelt,
            gdelt_start=gdelt_start,
            gdelt_end=end,
        )
        if not df_news.empty:
            df_news.to_csv(news_path, index=False)
            gdelt_n = int((df_news["source"] == "gdelt").sum())
            result.update(news_rows=len(df_news), gdelt_rows=gdelt_n)
            log.info(f"[{ticker}] Saved {len(df_news):,} articles "
                     f"({gdelt_n:,} from GDELT) → {news_path}")
        else:
            log.warning(f"[{ticker}] No news found")

    # ── Fundamentals ──────────────────────────────────────────────────────
    fund_path = FUND_DIR / f"{ticker}.json"
    if not (skip_existing and fund_path.exists()):
        info = fetch_fundamentals(ticker)
        if info:
            fund_path.write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return result


# ════════════════════════════════════════════════════════════════════════════
# MAIN RUN
# ════════════════════════════════════════════════════════════════════════════

def run(tickers: list[str],
        newsapi_key:  str  = "",
        delay:        float = 1.2,
        skip_existing:bool  = True,
        use_gdelt:    bool  = True,
        use_cotahist: bool  = False,
        gdelt_start:  str   = GDELT_START,
        start:        str   = START_DATE,
        end:          str   = END_DATE) -> pd.DataFrame:

    setup_dirs()
    log.info("=" * 65)
    log.info(f"B3 Scraper v3  |  {len(tickers)} tickers  |  {start} → {end}")
    log.info(f"GDELT: {'ON ('+gdelt_start+' → '+end+')' if use_gdelt else 'OFF'}")
    log.info("=" * 65)

    cotahist_data = None
    if use_cotahist:
        sy, ey = int(start[:4]), int(end[:4])
        log.info(f"Downloading COTAHIST {sy}–{ey}...")
        cotahist_data = fetch_cotahist_range(sy, ey, tickers)

    summary, failed = [], []

    for i, ticker in enumerate(tickers, 1):
        log.info(f"── [{i:>3}/{len(tickers)}] {ticker} ──")
        try:
            row = scrape_ticker(
                ticker,
                newsapi_key=newsapi_key,
                skip_existing=skip_existing,
                use_gdelt=use_gdelt,
                gdelt_start=gdelt_start,
                start=start,
                end=end,
                cotahist_data=cotahist_data,
            )
            summary.append(row)
        except Exception as e:
            log.error(f"[{ticker}] Unhandled: {e}")
            summary.append({"ticker": ticker, "price_rows": 0, "news_rows": 0,
                            "gdelt_rows": 0, "status": f"error: {e}"})
            failed.append(ticker)

        time.sleep(delay)

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    ok = len(tickers) - len(failed)
    log.info("=" * 65)
    log.info(f"Done  |  {ok}/{len(tickers)} OK  |  {len(failed)} failed")
    if failed:
        log.warning(f"Failed: {failed}")
    log.info(f"Outputs → {OUTPUT_DIR}/")
    log.info("=" * 65)
    return df_summary


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="B3 Scraper v3 — 20yr prices + GDELT news for 110 tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tickers",      nargs="+", default=None)
    p.add_argument("--limit",        type=int,  default=None,
                   help="Only first N tickers (for testing)")
    p.add_argument("--newsapi-key",  default="", metavar="KEY")
    p.add_argument("--delay",        type=float, default=1.2)
    p.add_argument("--no-skip",      action="store_true",
                   help="Re-download even if files exist")
    p.add_argument("--no-gdelt",     action="store_true",
                   help="Disable GDELT (faster, far fewer news articles)")
    p.add_argument("--gdelt-start",  default=GDELT_START,
                   help=f"GDELT start date (default {GDELT_START})")
    p.add_argument("--cotahist",     action="store_true",
                   help="Download official B3 COTAHIST price files as fallback")
    p.add_argument("--start",        default=START_DATE)
    p.add_argument("--end",          default=END_DATE)
    args = p.parse_args()

    # dateutil needed for GDELT windows
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "python-dateutil", "-q"])
        from dateutil.relativedelta import relativedelta

    tickers = args.tickers or B3_TICKERS
    if args.limit:
        tickers = tickers[:args.limit]

    df = run(
        tickers=tickers,
        newsapi_key=args.newsapi_key,
        delay=args.delay,
        skip_existing=not args.no_skip,
        use_gdelt=not args.no_gdelt,
        use_cotahist=args.cotahist,
        gdelt_start=args.gdelt_start,
        start=args.start,
        end=args.end,
    )

    print("\n📊 Run Summary:")
    print(df.to_string(index=False))
