from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

_FINVIZ_URL = "https://finviz.com/quote.ashx?t={ticker}&p=d"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finviz.com/",
}


def scrape_finviz_news(ticker: str, max_items: int = 100) -> pd.DataFrame:
    url = _FINVIZ_URL.format(ticker=ticker.upper())
    response = requests.get(url, headers=_HEADERS, timeout=15)
    response.raise_for_status()
    html = BeautifulSoup(response.text, "html.parser")

    rows = html.find_all("tr", class_="news_table-row")

    records: list[dict[str, Any]] = []
    for row in rows:
        if len(records) >= max_items:
            break
        title: str | None = None
        link_url: str | None = None
        title_cell = row.find("td", class_="news_link-cell")
        if title_cell:
            link = title_cell.find("a", class_="nn-tab-link")
            if link:
                title = link.get_text(strip=True)
                link_url = link.get("href")

        date_str: str | None = None
        date_cell = row.find("td", class_="news_date-cell")
        if date_cell is None:
            date_cell = row.find("td", class_="text-right")
        if date_cell is not None:
            date_str = date_cell.get_text(strip=True)

        if title is None:
            continue

        records.append(
            {
                "ticker": ticker.upper(),
                "date": date_str,
                "title": title,
                "url": link_url,
            }
        )

    if not records:
        return pd.DataFrame(columns=["ticker", "date", "title", "url"])

    df = pd.DataFrame(records)
    parsed = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["date"] = parsed
    return df


def extract_entities(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        from spacy.cli import download as _download

        _download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def finbert_sentiment(texts: list[str], batch_size: int = 16) -> list[dict]:
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
        batch_size=batch_size,
    )

    results: list[dict] = []
    for raw in classifier(texts, batch_size=batch_size):
        scores = {entry["label"].lower(): float(entry["score"]) for entry in raw}
        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        neutral = scores.get("neutral", 0.0)
        label = max(
            ("positive", positive),
            ("negative", negative),
            ("neutral", neutral),
            key=lambda kv: kv[1],
        )[0]
        results.append(
            {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "label": label,
            }
        )
    return results


def compute_sentiment_features(news_df: pd.DataFrame) -> pd.DataFrame:
    df = news_df.copy()
    df = df.dropna(subset=["title"]).reset_index(drop=True)

    sentiments = finbert_sentiment(df["title"].tolist())
    sent_df = pd.DataFrame(sentiments)
    df = pd.concat([df, sent_df], axis=1)

    df["entities"] = df["title"].apply(extract_entities)

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["date"]).copy()
    df["date_only"] = df["date"].dt.date

    agg = (
        df.groupby(["ticker", "date_only"])
        .agg(
            positive=("positive", "mean"),
            negative=("negative", "mean"),
            neutral=("neutral", "mean"),
            n_headlines=("title", "size"),
        )
        .reset_index()
        .rename(columns={"date_only": "date"})
    )

    agg["sentiment_score"] = agg["positive"] - agg["negative"]
    return agg
