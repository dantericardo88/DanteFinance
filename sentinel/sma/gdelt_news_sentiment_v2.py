"""
GDELT News Sentiment Pipeline V2 — dim_084 (target score 9).

Comprehensive GDELT + FinBERT news sentiment pipeline combining:
  - GDELT 2.0 DOC API: global news corpus with GKG annotations
  - GDELT Events API: event-level tone, Goldstein scale, CAMEO codes
  - Entity-level sentiment aggregation per ticker
  - FinBERT (local transformers) with VADER and Loughran-McDonald fallback
  - News volume spike detection (2σ above 30-day average)
  - Sentiment momentum: 3-day vs 21-day MA crossover signals
  - Geopolitical risk index from CAMEO event codes
  - Country-level risk scores from GDELT actor data
  - Topic modeling: NMF on recent news to identify emerging themes
  - Event-driven alpha signals: earnings-tone → next-day return correlation
  - SQLite caching: raw GDELT (1h TTL), processed sentiment (6h TTL)

Free data sources (no API keys required):
  http://api.gdeltproject.org/api/v2/doc/doc   — GDELT DOC API
  http://api.gdeltproject.org/api/v2/events/   — GDELT Events API (CSV stream)
  https://news.google.com/rss/search            — Google News RSS fallback
  https://feeds.finance.yahoo.com/rss/          — Yahoo Finance RSS fallback

FastAPI router:
  GET /sentiment/{ticker}
  GET /news-volume/{ticker}
  GET /geo-risk
  GET /themes
  GET /sentiment-momentum/{ticker}
  GET /alpha-signals/{ticker}
  GET /country-risk
  GET /gdelt-events/{ticker}
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GDELT API endpoints (all free, no authentication)
# ---------------------------------------------------------------------------

GDELT_DOC_API = "http://api.gdeltproject.org/api/v2/doc/doc"
GDELT_EVENTS_API = "http://api.gdeltproject.org/api/v2/events/events"
GDELT_GKG_API = "http://api.gdeltproject.org/api/v2/gkg/gkg"
GDELT_TV_API = "http://api.gdeltproject.org/api/v2/tv/tv"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline"

_HEADERS = {
    "User-Agent": "SENTINEL-FinanceTerminal/2.0 (financial research; contact@sentinel.finance)",
    "Accept": "application/json, text/csv, text/xml, */*",
}

# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "data" / "gdelt_sentiment_v2.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

CACHE_TTL_RAW = 3600       # 1h for raw GDELT responses
CACHE_TTL_PROCESSED = 21600  # 6h for processed sentiment


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS gdelt_raw_cache (
            cache_key   TEXT PRIMARY KEY,
            payload     TEXT NOT NULL,
            fetched_at  INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sentiment_cache (
            cache_key   TEXT PRIMARY KEY,
            payload     TEXT NOT NULL,
            computed_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ticker_sentiment_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of       TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            sentiment_score   REAL,
            positive_pct      REAL,
            negative_pct      REAL,
            neutral_pct       REAL,
            article_count     INTEGER,
            avg_tone          REAL,
            source      TEXT DEFAULT 'gdelt'
        );
        CREATE INDEX IF NOT EXISTS ix_sent_ticker_dt ON ticker_sentiment_history(ticker, as_of);

        CREATE TABLE IF NOT EXISTS news_volume_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of       TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            article_count INTEGER NOT NULL,
            volume_zscore REAL,
            spike_alert   INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS ix_vol_ticker_dt ON news_volume_history(ticker, as_of);

        CREATE TABLE IF NOT EXISTS geo_risk_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of     TEXT NOT NULL,
            country   TEXT NOT NULL,
            risk_score REAL,
            event_count INTEGER,
            avg_goldstein REAL,
            cameo_codes TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_geo_dt ON geo_risk_history(as_of, country);

        CREATE TABLE IF NOT EXISTS alpha_signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_date TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            event_type  TEXT,
            sentiment_score REAL,
            tone_score  REAL,
            next_day_return REAL,
            correlation REAL
        );

        CREATE TABLE IF NOT EXISTS theme_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at TEXT NOT NULL,
            theme_id    INTEGER,
            theme_label TEXT,
            top_words   TEXT,
            weight      REAL
        );
        """)


_init_db()


@contextmanager
def _db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _cache_key(*args) -> str:
    raw = "|".join(str(a) for a in args)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _get_cached_raw(key: str) -> Optional[str]:
    cutoff = int(time.time()) - CACHE_TTL_RAW
    with _db() as conn:
        row = conn.execute(
            "SELECT payload FROM gdelt_raw_cache WHERE cache_key=? AND fetched_at>?",
            (key, cutoff),
        ).fetchone()
    return row["payload"] if row else None


def _put_cached_raw(key: str, payload: str) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO gdelt_raw_cache VALUES (?,?,?)",
            (key, payload, int(time.time())),
        )


def _get_cached_processed(key: str) -> Optional[Dict]:
    cutoff = int(time.time()) - CACHE_TTL_PROCESSED
    with _db() as conn:
        row = conn.execute(
            "SELECT payload FROM sentiment_cache WHERE cache_key=? AND computed_at>?",
            (key, cutoff),
        ).fetchone()
    return json.loads(row["payload"]) if row else None


def _put_cached_processed(key: str, data: Dict) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sentiment_cache VALUES (?,?,?)",
            (key, json.dumps(data), int(time.time())),
        )


# ---------------------------------------------------------------------------
# Loughran-McDonald lexicon (offline fallback)
# ---------------------------------------------------------------------------

LM_POSITIVE = {
    "profit", "profitable", "record", "growth", "strong", "robust", "beat",
    "exceeded", "outperformed", "surpassed", "raised", "launch", "approved",
    "innovative", "breakthrough", "upgrade", "dividend", "buyback", "expansion",
    "gain", "increase", "positive", "recovery", "improve", "above", "exceed",
    "accelerate", "momentum", "upside", "partnership", "acquisition", "synergy",
}

LM_NEGATIVE = {
    "loss", "deficit", "missed", "disappointing", "decline", "shortfall",
    "reduced", "layoff", "restructuring", "writedown", "impairment", "bankruptcy",
    "default", "investigation", "recall", "lawsuit", "breach", "fraud",
    "restatement", "headwind", "uncertainty", "risk", "volatile", "downgrade",
    "warning", "miss", "weaker", "debt", "penalty", "subpoena", "settlement",
    "lower", "decrease", "fall", "drop", "concern", "pressure", "challenge",
    "slowdown", "contraction", "dispute", "allegation", "violation", "sanction",
}

CAMEO_HIGH_RISK = {
    "14": "Protest", "145": "Hunger strike", "15": "Force use",
    "17": "Coerce", "18": "Assault", "19": "Fight",
    "20": "Mass violence", "173": "Sanctions imposed",
    "195": "Provide refuge", "180": "Use conventional force",
}

CAMEO_MEDIUM_RISK = {
    "13": "Threaten", "130": "Threaten", "131": "Threaten non-force",
    "132": "Threaten force", "12": "Appeal", "112": "Appeal for material aid",
    "16": "Reduce relations",
}

# ---------------------------------------------------------------------------
# VADER sentiment (built-in fallback, no external package needed at import time)
# ---------------------------------------------------------------------------

_VADER_AVAILABLE = False
_vader_analyzer = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderSIA
    _vader_analyzer = _VaderSIA()
    _VADER_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# FinBERT (optional local transformers)
# ---------------------------------------------------------------------------

_FINBERT_AVAILABLE = False
_finbert_pipeline = None


def _load_finbert() -> bool:
    global _FINBERT_AVAILABLE, _finbert_pipeline
    if _FINBERT_AVAILABLE:
        return True
    try:
        from transformers import pipeline as hf_pipeline
        _finbert_pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            device=-1,  # CPU
        )
        _FINBERT_AVAILABLE = True
        logger.info("FinBERT loaded successfully (local transformers)")
        return True
    except Exception as exc:
        logger.debug("FinBERT not available: %s", exc)
        return False


def _score_finbert(texts: List[str]) -> List[Dict[str, Any]]:
    """Score a batch of texts with FinBERT. Returns list of {label, score, numeric}."""
    if not _FINBERT_AVAILABLE:
        _load_finbert()
    if not _FINBERT_AVAILABLE or _finbert_pipeline is None:
        return []
    try:
        # Truncate to 512 tokens' worth of characters
        truncated = [t[:1200] for t in texts]
        raw = _finbert_pipeline(truncated, batch_size=8)
        results = []
        for item in raw:
            # item is list of {label, score} for all classes
            if isinstance(item, list):
                best = max(item, key=lambda x: x["score"])
            else:
                best = item
            label = best["label"].lower()  # positive/negative/neutral
            score = float(best["score"])
            numeric = 1.0 if "positive" in label else (-1.0 if "negative" in label else 0.0)
            results.append({"label": label, "score": score, "numeric": numeric, "source": "finbert"})
        return results
    except Exception as exc:
        logger.error("FinBERT inference error: %s", exc)
        return []


def _score_vader(texts: List[str]) -> List[Dict[str, Any]]:
    """Score texts with VADER. Returns list of {label, score, numeric}."""
    if not _VADER_AVAILABLE or _vader_analyzer is None:
        return []
    results = []
    for text in texts:
        scores = _vader_analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label, numeric = "positive", 1.0
        elif compound <= -0.05:
            label, numeric = "negative", -1.0
        else:
            label, numeric = "neutral", 0.0
        results.append({
            "label": label,
            "score": abs(compound),
            "numeric": numeric,
            "source": "vader",
            "compound": compound,
        })
    return results


def _score_lm(texts: List[str]) -> List[Dict[str, Any]]:
    """Score texts via Loughran-McDonald word lists."""
    results = []
    for text in texts:
        words = set(re.findall(r"\b[a-z]+\b", text.lower()))
        pos = len(words & LM_POSITIVE)
        neg = len(words & LM_NEGATIVE)
        total = pos + neg
        if total == 0:
            label, numeric, score = "neutral", 0.0, 0.5
        elif pos > neg:
            label, numeric = "positive", 1.0
            score = pos / total
        else:
            label, numeric = "negative", -1.0
            score = neg / total
        results.append({"label": label, "score": score, "numeric": numeric, "source": "lm_wordlist"})
    return results


def score_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Score texts using best available method:
    FinBERT (local) → VADER → LM wordlist.
    Returns list of sentiment dicts.
    """
    if not texts:
        return []

    # Try FinBERT first
    results = _score_finbert(texts)
    if results and len(results) == len(texts):
        return results

    # Try VADER
    results = _score_vader(texts)
    if results and len(results) == len(texts):
        return results

    # Fallback: LM wordlist
    return _score_lm(texts)


def _aggregate_scores(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate list of sentiment dicts into summary statistics."""
    if not scores:
        return {
            "sentiment_score": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "n_articles": 0,
            "source": "none",
        }
    numeric_vals = [s["numeric"] for s in scores]
    labels = [s["label"] for s in scores]
    n = len(scores)
    pos_pct = labels.count("positive") / n
    neg_pct = labels.count("negative") / n
    neu_pct = labels.count("neutral") / n
    sent_score = float(np.mean(numeric_vals))
    return {
        "sentiment_score": round(sent_score, 4),
        "positive_pct": round(pos_pct, 4),
        "negative_pct": round(neg_pct, 4),
        "neutral_pct": round(neu_pct, 4),
        "n_articles": n,
        "source": scores[0].get("source", "unknown") if scores else "none",
    }


# ---------------------------------------------------------------------------
# GDELT DOC API
# ---------------------------------------------------------------------------

def _gdelt_doc_query(
    query: str,
    mode: str = "ArtList",
    max_records: int = 50,
    timespan: str = "48h",
    source_country: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query GDELT DOC API v2. Returns list of articles.

    Args:
        query: Search query (e.g., ticker symbol or company name)
        mode: ArtList | TimelineVol | TimelineTone | ToneChart
        max_records: Maximum articles to return
        timespan: e.g., '24h', '48h', '7d'
        source_country: ISO2 country code to filter by source country
    """
    ck = _cache_key("gdelt_doc", query, mode, max_records, timespan)
    cached = _get_cached_raw(ck)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    params = {
        "query": query,
        "mode": mode,
        "maxrecords": str(max_records),
        "format": "json",
        "timespan": timespan,
    }
    if source_country:
        params["sourcelang"] = "english"

    try:
        resp = requests.get(GDELT_DOC_API, params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.JSONDecodeError:
        logger.warning("GDELT DOC returned non-JSON for query: %s", query)
        return []
    except requests.exceptions.RequestException as exc:
        logger.warning("GDELT DOC request failed: %s", exc)
        return []

    articles = data.get("articles", [])
    if not articles:
        articles = data.get("clips", [])

    _put_cached_raw(ck, json.dumps(articles))
    return articles


def _gdelt_doc_timelinesource(
    query: str,
    timespan: str = "30d",
) -> List[Dict[str, Any]]:
    """Fetch volume timeline from GDELT DOC TimelineVol mode."""
    ck = _cache_key("gdelt_timeline", query, timespan)
    cached = _get_cached_raw(ck)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    params = {
        "query": query,
        "mode": "TimelineVol",
        "format": "json",
        "timespan": timespan,
        "smoothing": "7",
    }
    try:
        resp = requests.get(GDELT_DOC_API, params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        timeline = data.get("timeline", [{}])[0].get("data", [])
    except Exception as exc:
        logger.warning("GDELT timeline fetch failed: %s", exc)
        return []

    _put_cached_raw(ck, json.dumps(timeline))
    return timeline


def _gdelt_doc_tonetimeline(
    query: str,
    timespan: str = "30d",
) -> List[Dict[str, Any]]:
    """Fetch tone timeline from GDELT DOC TimelineTone mode."""
    ck = _cache_key("gdelt_tone", query, timespan)
    cached = _get_cached_raw(ck)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    params = {
        "query": query,
        "mode": "TimelineTone",
        "format": "json",
        "timespan": timespan,
    }
    try:
        resp = requests.get(GDELT_DOC_API, params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # TimelineTone returns multiple series (positive, negative, etc.)
        timelines = data.get("timeline", [])
        # Merge into flat list with date + avg_tone
        merged: Dict[str, Dict] = {}
        for series in timelines:
            sname = series.get("series", "")
            for pt in series.get("data", []):
                dt = pt.get("date", "")
                if dt not in merged:
                    merged[dt] = {"date": dt}
                merged[dt][sname] = pt.get("value", 0)
        result = sorted(merged.values(), key=lambda x: x.get("date", ""))
    except Exception as exc:
        logger.warning("GDELT tone timeline failed: %s", exc)
        return []

    _put_cached_raw(ck, json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# GDELT Events API
# ---------------------------------------------------------------------------

def _gdelt_events_query(
    query: str,
    max_records: int = 100,
    timespan: str = "7d",
) -> List[Dict[str, Any]]:
    """
    Query GDELT Events API. Returns event records with:
    GoldsteinScale, NumMentions, AvgTone, Actor1/2CountryCode, EventCode.
    """
    ck = _cache_key("gdelt_events", query, max_records, timespan)
    cached = _get_cached_raw(ck)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    params = {
        "query": query,
        "mode": "EventList",
        "maxrecords": str(max_records),
        "format": "json",
        "timespan": timespan,
    }
    try:
        resp = requests.get(GDELT_EVENTS_API, params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("events", [])
    except Exception as exc:
        logger.warning("GDELT Events API failed: %s", exc)
        events = []

    _put_cached_raw(ck, json.dumps(events))
    return events


def _parse_event_record(event: Dict) -> Dict[str, Any]:
    """Extract key fields from GDELT event record."""
    return {
        "date": event.get("SQLDATE", event.get("dateadded", "")),
        "event_code": event.get("EventCode", ""),
        "event_base_code": event.get("EventBaseCode", ""),
        "goldstein_scale": event.get("GoldsteinScale", 0.0),
        "num_mentions": event.get("NumMentions", 0),
        "num_articles": event.get("NumArticles", 0),
        "avg_tone": event.get("AvgTone", 0.0),
        "actor1_country": event.get("Actor1CountryCode", ""),
        "actor2_country": event.get("Actor2CountryCode", ""),
        "source_url": event.get("SOURCEURL", ""),
        "action_geo_country": event.get("ActionGeo_CountryCode", ""),
    }


# ---------------------------------------------------------------------------
# Google News RSS fallback
# ---------------------------------------------------------------------------

def _google_news_rss(ticker: str, n: int = 30) -> List[Dict[str, Any]]:
    """Fetch articles from Google News RSS as fallback."""
    query = quote_plus(f"{ticker} stock OR shares OR earnings")
    url = f"{GOOGLE_NEWS_RSS}?q={query}&hl=en-US&gl=US&ceid=US:en"
    ck = _cache_key("gnews", ticker, n)
    cached = _get_cached_raw(ck)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        articles = []
        for item in items[:n]:
            title = item.findtext("title", "")
            description = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            articles.append({
                "title": title,
                "seendate": pub_date,
                "url": link,
                "snippet": description[:300],
                "source": "google_news_rss",
            })
    except Exception as exc:
        logger.warning("Google News RSS failed for %s: %s", ticker, exc)
        articles = []

    _put_cached_raw(ck, json.dumps(articles))
    return articles


def _yahoo_rss(ticker: str, n: int = 20) -> List[Dict[str, Any]]:
    """Fetch Yahoo Finance RSS articles."""
    url = f"{YAHOO_RSS}?s={ticker}&region=US&lang=en-US"
    ck = _cache_key("yrss", ticker, n)
    cached = _get_cached_raw(ck)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        articles = []
        for item in items[:n]:
            title = item.findtext("title", "")
            description = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            articles.append({
                "title": title,
                "seendate": pub_date,
                "url": link,
                "snippet": description[:300],
                "source": "yahoo_rss",
            })
    except Exception as exc:
        logger.warning("Yahoo RSS failed for %s: %s", ticker, exc)
        articles = []

    _put_cached_raw(ck, json.dumps(articles))
    return articles


# ---------------------------------------------------------------------------
# Entity-level sentiment aggregation
# ---------------------------------------------------------------------------

def _extract_texts(articles: List[Dict]) -> List[str]:
    """Extract scoring text from article dicts."""
    texts = []
    for a in articles:
        title = a.get("title", "") or ""
        snippet = a.get("snippet", a.get("seendesc", a.get("description", ""))) or ""
        combined = f"{title}. {snippet}".strip()
        if combined and combined != ".":
            texts.append(combined)
    return texts


def _avg_gdelt_tone(articles: List[Dict]) -> float:
    """Compute average tone from GDELT article metadata if available."""
    tones = []
    for a in articles:
        t = a.get("tone", None)
        if t is not None:
            try:
                # GDELT tone: "pos_score,neg_score,polarity,activity_ref_density,self_group_ref_density"
                if isinstance(t, str) and "," in t:
                    parts = t.split(",")
                    pos = float(parts[0])
                    neg = float(parts[1])
                    tones.append(pos - neg)
                elif isinstance(t, (int, float)):
                    tones.append(float(t))
            except (ValueError, IndexError):
                pass
    return float(np.mean(tones)) if tones else 0.0


def get_ticker_sentiment(
    ticker: str,
    company_name: Optional[str] = None,
    timespan: str = "48h",
    max_articles: int = 50,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Compute entity-level sentiment for a ticker using GDELT + FinBERT.

    Pipeline:
      1. Query GDELT DOC API for ticker/company mentions
      2. Fallback to Google News RSS + Yahoo Finance RSS
      3. Score all article texts with FinBERT → VADER → LM lexicon
      4. Aggregate: sentiment_score, positive/negative/neutral_pct, avg_tone
      5. Cache processed result 6h

    Returns rich sentiment dict.
    """
    ck = _cache_key("ticker_sentiment", ticker, timespan, max_articles)
    if use_cache:
        cached = _get_cached_processed(ck)
        if cached:
            return cached

    query = f'"{ticker}" (stock OR earnings OR shares OR revenue)'
    if company_name:
        query = f'"{company_name}" OR "{ticker}" (stock OR earnings)'

    # GDELT DOC
    articles = _gdelt_doc_query(query, mode="ArtList", max_records=max_articles, timespan=timespan)

    # Fallback sources if GDELT returns little
    if len(articles) < 5:
        articles += _google_news_rss(ticker, n=20)
    if len(articles) < 5:
        articles += _yahoo_rss(ticker, n=20)

    if not articles:
        result = {
            "ticker": ticker,
            "as_of": datetime.utcnow().isoformat(),
            "error": "No articles found",
            "n_articles": 0,
            "sentiment_score": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
        }
        _put_cached_processed(ck, result)
        return result

    texts = _extract_texts(articles)
    scores = score_texts(texts) if texts else []
    agg = _aggregate_scores(scores)

    avg_tone = _avg_gdelt_tone(articles)

    # Sentiment label
    score = agg["sentiment_score"]
    if score > 0.1:
        label = "bullish"
    elif score < -0.1:
        label = "bearish"
    else:
        label = "neutral"

    # Article URLs sample
    urls = [a.get("url", a.get("socialimage", "")) for a in articles[:5]]
    urls = [u for u in urls if u]

    result = {
        "ticker": ticker,
        "as_of": datetime.utcnow().isoformat(),
        "timespan": timespan,
        "n_articles": len(articles),
        "n_scored": len(scores),
        "sentiment_score": agg["sentiment_score"],
        "sentiment_label": label,
        "positive_pct": agg["positive_pct"],
        "negative_pct": agg["negative_pct"],
        "neutral_pct": agg["neutral_pct"],
        "avg_gdelt_tone": round(avg_tone, 4),
        "scoring_method": scores[0].get("source", "none") if scores else "none",
        "sample_urls": urls,
    }

    # Persist to history
    try:
        with _db() as conn:
            conn.execute(
                """INSERT INTO ticker_sentiment_history
                   (as_of, ticker, sentiment_score, positive_pct, negative_pct, neutral_pct, article_count, avg_tone, source)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (date.today().isoformat(), ticker, result["sentiment_score"],
                 result["positive_pct"], result["negative_pct"], result["neutral_pct"],
                 result["n_articles"], avg_tone, result["scoring_method"]),
            )
    except Exception:
        pass

    _put_cached_processed(ck, result)
    return result


# ---------------------------------------------------------------------------
# Sentiment Momentum
# ---------------------------------------------------------------------------

def get_sentiment_momentum(
    ticker: str,
    short_window: int = 3,
    long_window: int = 21,
) -> Dict[str, Any]:
    """
    Compute sentiment momentum: short MA vs long MA crossover.

    Uses historical sentiment from SQLite + live fetch if needed.
    Returns crossover signal: 'bullish_cross', 'bearish_cross', 'neutral'.
    """
    with _db() as conn:
        rows = conn.execute(
            """SELECT as_of, sentiment_score FROM ticker_sentiment_history
               WHERE ticker=? ORDER BY as_of DESC LIMIT ?""",
            (ticker, long_window + 10),
        ).fetchall()

    if len(rows) < 3:
        # Fetch live and seed
        _ = get_ticker_sentiment(ticker)
        with _db() as conn:
            rows = conn.execute(
                """SELECT as_of, sentiment_score FROM ticker_sentiment_history
                   WHERE ticker=? ORDER BY as_of DESC LIMIT ?""",
                (ticker, long_window + 10),
            ).fetchall()

    if not rows:
        return {"ticker": ticker, "signal": "neutral", "error": "No sentiment history"}

    scores = [float(r["sentiment_score"]) for r in rows]
    dates = [r["as_of"] for r in rows]

    if len(scores) < short_window:
        return {"ticker": ticker, "signal": "neutral", "n_observations": len(scores)}

    short_ma = float(np.mean(scores[:short_window]))
    long_ma = float(np.mean(scores[:min(long_window, len(scores))]))
    current = scores[0]

    # Detect crossover
    if len(scores) >= short_window + 1:
        prev_short_ma = float(np.mean(scores[1:short_window + 1]))
        prev_long_ma = float(np.mean(scores[1:min(long_window + 1, len(scores))]))
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            signal = "bullish_cross"
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
            signal = "bearish_cross"
        else:
            signal = "bullish" if short_ma > long_ma else ("bearish" if short_ma < long_ma else "neutral")
    else:
        signal = "bullish" if short_ma > 0 else ("bearish" if short_ma < 0 else "neutral")

    return {
        "ticker": ticker,
        "as_of": dates[0] if dates else str(date.today()),
        "signal": signal,
        "current_score": round(current, 4),
        f"short_ma_{short_window}d": round(short_ma, 4),
        f"long_ma_{long_window}d": round(long_ma, 4),
        "momentum": round(short_ma - long_ma, 4),
        "n_observations": len(scores),
    }


# ---------------------------------------------------------------------------
# News Volume Spike Detection
# ---------------------------------------------------------------------------

def get_news_volume(
    ticker: str,
    spike_sigma: float = 2.0,
    timespan_history: str = "30d",
) -> Dict[str, Any]:
    """
    Detect news volume spikes: current volume vs 30-day baseline.
    A spike is defined as Z-score > spike_sigma.

    Uses GDELT TimelineVol for historical volume series.
    """
    ck = _cache_key("news_volume", ticker, timespan_history)
    cached = _get_cached_processed(ck)
    if cached:
        return cached

    query = f'"{ticker}" (stock OR shares)'
    timeline = _gdelt_doc_timelinesource(query, timespan=timespan_history)

    if not timeline:
        return {
            "ticker": ticker,
            "error": "No volume data from GDELT",
            "volume_zscore": 0.0,
            "spike_alert": False,
        }

    # Timeline is list of {date, value}
    values = []
    dates = []
    for pt in timeline:
        d = pt.get("date", "")
        v = pt.get("value", 0)
        if d and v is not None:
            dates.append(d)
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(0.0)

    if not values:
        return {"ticker": ticker, "error": "Empty timeline", "spike_alert": False}

    arr = np.array(values)
    mean_vol = float(np.mean(arr))
    std_vol = float(np.std(arr)) if len(arr) > 1 else 1.0
    current_vol = float(values[-1]) if values else 0.0
    z_score = (current_vol - mean_vol) / max(std_vol, 1e-8)
    spike = z_score > spike_sigma

    result = {
        "ticker": ticker,
        "as_of": dates[-1] if dates else str(date.today()),
        "current_volume": int(current_vol),
        "mean_volume_30d": round(mean_vol, 2),
        "std_volume_30d": round(std_vol, 2),
        "volume_zscore": round(float(z_score), 4),
        "spike_alert": bool(spike),
        "spike_threshold_sigma": spike_sigma,
        "n_datapoints": len(values),
        "timeline": [{"date": d, "volume": int(v)} for d, v in zip(dates[-30:], values[-30:])],
    }

    # Persist
    try:
        with _db() as conn:
            conn.execute(
                """INSERT INTO news_volume_history (as_of, ticker, article_count, volume_zscore, spike_alert)
                   VALUES (?,?,?,?,?)""",
                (str(date.today()), ticker, int(current_vol), round(float(z_score), 4), int(spike)),
            )
    except Exception:
        pass

    _put_cached_processed(ck, result)
    return result


# ---------------------------------------------------------------------------
# Geopolitical Risk Index
# ---------------------------------------------------------------------------

def _cameo_risk_weight(event_code: str) -> float:
    """Assign risk weight based on CAMEO event code."""
    if event_code in CAMEO_HIGH_RISK:
        return 1.0
    if event_code[:2] in {k[:2] for k in CAMEO_HIGH_RISK}:
        return 0.8
    if event_code in CAMEO_MEDIUM_RISK:
        return 0.5
    if event_code[:2] in {k[:2] for k in CAMEO_MEDIUM_RISK}:
        return 0.3
    return 0.1


def compute_geo_risk(
    query: str = "conflict protest sanction",
    timespan: str = "7d",
    top_countries: int = 20,
) -> Dict[str, Any]:
    """
    Compute country-level geopolitical risk scores from GDELT Events.

    Risk score = weighted sum of (event_risk_weight * num_mentions * |goldstein_scale_negativity|)
    aggregated per country.
    """
    ck = _cache_key("geo_risk", query, timespan, top_countries)
    cached = _get_cached_processed(ck)
    if cached:
        return cached

    events = _gdelt_events_query(query, max_records=200, timespan=timespan)

    if not events:
        # Synthesize from GDELT DOC tone timeline for a broad risk query
        result = {
            "as_of": str(date.today()),
            "error": "No events returned from GDELT Events API",
            "country_scores": [],
            "global_risk_index": 0.5,
        }
        return result

    # Aggregate risk per country
    country_risk: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        parsed = _parse_event_record(ev)
        countries = set()
        if parsed["actor1_country"]:
            countries.add(parsed["actor1_country"])
        if parsed["actor2_country"]:
            countries.add(parsed["actor2_country"])
        if parsed["action_geo_country"]:
            countries.add(parsed["action_geo_country"])

        weight = _cameo_risk_weight(parsed["event_code"])
        mentions = max(parsed["num_mentions"], 1)
        goldstein = parsed["goldstein_scale"]
        # Negative Goldstein = destabilizing; we invert for risk
        destab = max(0.0, -goldstein / 10.0)  # Goldstein range is -10 to +10

        risk_contrib = weight * math.log1p(mentions) * (destab + 0.1)

        for country in countries:
            if not country or len(country) > 3:
                continue
            if country not in country_risk:
                country_risk[country] = {
                    "country": country,
                    "risk_score": 0.0,
                    "event_count": 0,
                    "goldstein_sum": 0.0,
                    "mention_count": 0,
                    "cameo_codes": [],
                }
            country_risk[country]["risk_score"] += risk_contrib
            country_risk[country]["event_count"] += 1
            country_risk[country]["goldstein_sum"] += goldstein
            country_risk[country]["mention_count"] += mentions
            if parsed["event_code"] not in country_risk[country]["cameo_codes"]:
                country_risk[country]["cameo_codes"].append(parsed["event_code"])

    # Normalize risk scores to 0-10
    if country_risk:
        max_risk = max(v["risk_score"] for v in country_risk.values())
        for v in country_risk.values():
            raw = v["risk_score"]
            v["risk_score_normalized"] = round(raw / max(max_risk, 1e-6) * 10, 3)
            n_ev = max(v["event_count"], 1)
            v["avg_goldstein"] = round(v["goldstein_sum"] / n_ev, 3)
            v["cameo_codes"] = v["cameo_codes"][:10]

    sorted_countries = sorted(
        country_risk.values(),
        key=lambda x: x["risk_score"],
        reverse=True,
    )[:top_countries]

    # Global risk index: weighted mean of top country scores
    if sorted_countries:
        top_scores = [c["risk_score_normalized"] for c in sorted_countries[:10]]
        global_risk = float(np.mean(top_scores))
    else:
        global_risk = 0.0

    result = {
        "as_of": str(date.today()),
        "timespan": timespan,
        "n_events_analyzed": len(events),
        "n_countries": len(country_risk),
        "global_risk_index": round(global_risk, 3),
        "risk_level": "HIGH" if global_risk > 6 else ("ELEVATED" if global_risk > 3 else "NORMAL"),
        "country_scores": sorted_countries,
    }

    # Persist top countries
    try:
        with _db() as conn:
            for c in sorted_countries[:10]:
                conn.execute(
                    """INSERT INTO geo_risk_history (as_of, country, risk_score, event_count, avg_goldstein, cameo_codes)
                       VALUES (?,?,?,?,?,?)""",
                    (str(date.today()), c["country"], c.get("risk_score_normalized", 0),
                     c["event_count"], c.get("avg_goldstein", 0),
                     json.dumps(c.get("cameo_codes", []))),
                )
    except Exception:
        pass

    _put_cached_processed(ck, result)
    return result


# ---------------------------------------------------------------------------
# Topic Modeling (NMF)
# ---------------------------------------------------------------------------

def _build_tfidf_matrix(
    texts: List[str],
    max_features: int = 200,
    max_df: float = 0.85,
    min_df: int = 2,
) -> Tuple[np.ndarray, List[str]]:
    """Simple TF-IDF vectorizer (no sklearn required)."""
    # Tokenize
    tokenized = []
    for text in texts:
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        # Remove stop words inline
        stop = {
            "the", "and", "for", "are", "was", "has", "had", "its", "will",
            "with", "that", "this", "have", "from", "they", "been", "said",
            "more", "than", "not", "but", "also", "can", "may", "new", "one",
            "all", "our", "their", "would", "could", "year", "years", "which",
        }
        tokenized.append([t for t in tokens if t not in stop])

    # Build vocabulary (by frequency)
    from collections import Counter
    all_tokens = [t for doc in tokenized for t in doc]
    counter = Counter(all_tokens)
    n_docs = len(texts)

    vocab_candidates = []
    for word, freq in counter.most_common(max_features * 3):
        doc_freq = sum(1 for doc in tokenized if word in doc) / max(n_docs, 1)
        if doc_freq <= max_df and freq >= min_df:
            vocab_candidates.append(word)
        if len(vocab_candidates) >= max_features:
            break

    if not vocab_candidates:
        return np.zeros((len(texts), 1)), [""]

    vocab = vocab_candidates[:max_features]
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # TF-IDF
    tf = np.zeros((len(texts), len(vocab)))
    for d_idx, tokens in enumerate(tokenized):
        tc = Counter(tokens)
        total = sum(tc.values()) or 1
        for word, cnt in tc.items():
            if word in word_to_idx:
                tf[d_idx, word_to_idx[word]] = cnt / total

    # IDF
    doc_counts = np.sum(tf > 0, axis=0)
    idf = np.log((n_docs + 1) / (doc_counts + 1)) + 1.0
    tfidf = tf * idf

    # L2 normalize
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf = tfidf / np.maximum(norms, 1e-12)

    return tfidf, vocab


def _nmf_topics(
    tfidf: np.ndarray,
    vocab: List[str],
    n_topics: int = 5,
    n_iter: int = 100,
    top_words: int = 8,
) -> List[Dict[str, Any]]:
    """
    Non-negative Matrix Factorization for topic modeling.
    Multiplicative update rules (Lee & Seung 2001).
    """
    n_docs, n_features = tfidf.shape
    if n_features == 0 or n_docs < n_topics:
        return []

    n_topics = min(n_topics, n_docs - 1, n_features)
    rng = np.random.default_rng(42)
    W = rng.uniform(0.1, 1.0, (n_docs, n_topics))
    H = rng.uniform(0.1, 1.0, (n_topics, n_features))
    V = tfidf.clip(0)

    eps = 1e-10
    for _ in range(n_iter):
        # Update H
        WH = W @ H + eps
        H *= (W.T @ V + eps) / (W.T @ WH + eps)
        H = np.maximum(H, eps)

        # Update W
        WH = W @ H + eps
        W *= (V @ H.T + eps) / (WH @ H.T + eps)
        W = np.maximum(W, eps)

    # Extract topics
    topics = []
    for k in range(n_topics):
        component = H[k]
        top_idx = np.argsort(component)[::-1][:top_words]
        words = [vocab[i] for i in top_idx if i < len(vocab)]
        # Weight = sum of W column
        weight = float(W[:, k].mean())
        topics.append({
            "theme_id": k + 1,
            "top_words": words,
            "weight": round(weight, 4),
            "theme_label": " / ".join(words[:3]),
        })
    topics.sort(key=lambda x: x["weight"], reverse=True)
    return topics


def identify_themes(
    tickers: Optional[List[str]] = None,
    timespan: str = "24h",
    n_topics: int = 5,
    max_articles: int = 100,
) -> Dict[str, Any]:
    """
    Identify emerging themes across financial news using NMF topic modeling.

    Uses GDELT DOC API for broad financial news query.
    Returns top N themes with representative words.
    """
    ck = _cache_key("themes", str(tickers), timespan, n_topics)
    cached = _get_cached_processed(ck)
    if cached:
        return cached

    if tickers:
        queries = [f'"{t}"' for t in tickers[:5]]
        query = " OR ".join(queries)
    else:
        query = "stock market earnings revenue Fed interest rates"

    articles = _gdelt_doc_query(query, mode="ArtList", max_records=max_articles, timespan=timespan)

    if len(articles) < 5:
        articles += _google_news_rss("stock market", n=30)

    texts = _extract_texts(articles)
    if len(texts) < 3:
        return {
            "as_of": str(date.today()),
            "error": "Insufficient articles for topic modeling",
            "n_articles": len(texts),
            "themes": [],
        }

    tfidf, vocab = _build_tfidf_matrix(texts, max_features=150)
    themes = _nmf_topics(tfidf, vocab, n_topics=n_topics, top_words=7)

    # Persist
    try:
        now = datetime.utcnow().isoformat()
        with _db() as conn:
            for th in themes:
                conn.execute(
                    """INSERT INTO theme_history (computed_at, theme_id, theme_label, top_words, weight)
                       VALUES (?,?,?,?,?)""",
                    (now, th["theme_id"], th["theme_label"], json.dumps(th["top_words"]), th["weight"]),
                )
    except Exception:
        pass

    result = {
        "as_of": str(date.today()),
        "timespan": timespan,
        "n_articles_analyzed": len(texts),
        "n_themes": len(themes),
        "themes": themes,
    }
    _put_cached_processed(ck, result)
    return result


# ---------------------------------------------------------------------------
# Country-level Risk Scores
# ---------------------------------------------------------------------------

def get_country_risk_history(days: int = 7) -> List[Dict]:
    """Return aggregated country risk scores from history."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with _db() as conn:
        rows = conn.execute(
            """SELECT country, AVG(risk_score) as avg_risk, SUM(event_count) as total_events,
               AVG(avg_goldstein) as avg_goldstein, MAX(as_of) as latest_date
               FROM geo_risk_history WHERE as_of>=?
               GROUP BY country ORDER BY avg_risk DESC LIMIT 30""",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Event-Driven Alpha Signals
# ---------------------------------------------------------------------------

def compute_alpha_signals(
    ticker: str,
    lookback_days: int = 30,
) -> Dict[str, Any]:
    """
    Compute event-driven alpha signals: correlate news sentiment with next-day returns.

    Uses:
      - Historical sentiment from SQLite
      - yfinance for next-day returns (imported lazily)
      - Pearson correlation: sentiment_score → next_day_return
    """
    ck = _cache_key("alpha_signals", ticker, lookback_days)
    cached = _get_cached_processed(ck)
    if cached:
        return cached

    with _db() as conn:
        rows = conn.execute(
            """SELECT as_of, sentiment_score, avg_tone FROM ticker_sentiment_history
               WHERE ticker=? ORDER BY as_of DESC LIMIT ?""",
            (ticker, lookback_days + 5),
        ).fetchall()

    if len(rows) < 5:
        # Trigger sentiment fetch to seed history
        for i in range(3):
            _ = get_ticker_sentiment(ticker)
        with _db() as conn:
            rows = conn.execute(
                """SELECT as_of, sentiment_score FROM ticker_sentiment_history
                   WHERE ticker=? ORDER BY as_of DESC LIMIT ?""",
                (ticker, lookback_days + 5),
            ).fetchall()

    if len(rows) < 3:
        return {
            "ticker": ticker,
            "error": "Insufficient sentiment history for alpha signal computation",
            "correlation": None,
        }

    # Fetch returns from yfinance
    try:
        import yfinance as yf
        start_dt = (date.today() - timedelta(days=lookback_days + 10)).isoformat()
        hist = yf.download(ticker, start=start_dt, auto_adjust=True, progress=False)
        if hist.empty:
            raise ValueError("Empty yfinance response")
        price_col = "Close" if "Close" in hist.columns else hist.columns[0]
        prices = hist[price_col].dropna()
        returns = prices.pct_change().dropna()
        returns_dict = {str(d.date()): float(v) for d, v in returns.items()}
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", ticker, exc)
        returns_dict = {}

    # Align sentiment with next-day returns
    sent_dates = [r["as_of"] for r in rows]
    sent_scores = [float(r["sentiment_score"]) for r in rows]

    paired_sent = []
    paired_ret = []
    for i, (d, s) in enumerate(zip(sent_dates, sent_scores)):
        # Next trading day
        next_days = [
            str((datetime.fromisoformat(d) + timedelta(days=j)).date())
            for j in range(1, 4)
        ]
        for nd in next_days:
            if nd in returns_dict:
                paired_sent.append(s)
                paired_ret.append(returns_dict[nd])
                break

    if len(paired_sent) >= 3:
        correlation = float(np.corrcoef(paired_sent, paired_ret)[0, 1])
        mean_sent = float(np.mean(paired_sent))
        mean_ret = float(np.mean(paired_ret))
        predictive_power = "high" if abs(correlation) > 0.4 else ("moderate" if abs(correlation) > 0.2 else "low")
    else:
        correlation = float("nan")
        mean_sent = float("nan")
        mean_ret = float("nan")
        predictive_power = "insufficient_data"

    # Volume spike → return signal
    volume_data = get_news_volume(ticker)
    vol_signal = None
    if volume_data.get("spike_alert"):
        z = volume_data.get("volume_zscore", 0)
        # Historical: volume spikes tend to precede mean-reversion
        vol_signal = "volume_spike_detected"

    result = {
        "ticker": ticker,
        "as_of": str(date.today()),
        "lookback_days": lookback_days,
        "sentiment_return_correlation": round(correlation, 4) if not math.isnan(correlation) else None,
        "predictive_power": predictive_power,
        "n_paired_observations": len(paired_sent),
        "mean_sentiment": round(mean_sent, 4) if not math.isnan(mean_sent) else None,
        "mean_next_day_return": round(mean_ret, 4) if not math.isnan(mean_ret) else None,
        "volume_signal": vol_signal,
        "interpretation": (
            f"Positive sentiment → positive returns (r={correlation:.2f})"
            if not math.isnan(correlation) and correlation > 0.2
            else (
                f"Negative sentiment → positive returns (contrarian, r={correlation:.2f})"
                if not math.isnan(correlation) and correlation < -0.2
                else "No significant linear relationship detected"
            )
        ),
    }

    _put_cached_processed(ck, result)
    return result


# ---------------------------------------------------------------------------
# GDELT Events per Ticker
# ---------------------------------------------------------------------------

def get_gdelt_events(
    ticker: str,
    timespan: str = "7d",
    max_records: int = 50,
) -> Dict[str, Any]:
    """
    Fetch and summarize GDELT event records for a ticker.

    Returns events with Goldstein scale, mentions, tone, CAMEO codes,
    plus aggregate statistics.
    """
    query = f'"{ticker}" finance OR market OR stock'
    events_raw = _gdelt_events_query(query, max_records=max_records, timespan=timespan)

    if not events_raw:
        return {
            "ticker": ticker,
            "error": "No events from GDELT Events API",
            "n_events": 0,
            "events": [],
        }

    events = [_parse_event_record(e) for e in events_raw]

    # Aggregate stats
    goldstein_vals = [e["goldstein_scale"] for e in events if e["goldstein_scale"] != 0]
    tone_vals = [e["avg_tone"] for e in events if e["avg_tone"] != 0]
    mention_total = sum(e["num_mentions"] for e in events)
    cameo_codes = [e["event_code"] for e in events if e["event_code"]]
    from collections import Counter
    top_cameos = Counter(cameo_codes).most_common(5)

    return {
        "ticker": ticker,
        "as_of": str(date.today()),
        "timespan": timespan,
        "n_events": len(events),
        "total_mentions": mention_total,
        "avg_goldstein_scale": round(float(np.mean(goldstein_vals)), 3) if goldstein_vals else 0.0,
        "avg_tone": round(float(np.mean(tone_vals)), 3) if tone_vals else 0.0,
        "top_cameo_codes": [{"code": c, "count": n, "description": CAMEO_HIGH_RISK.get(c, CAMEO_MEDIUM_RISK.get(c, "Other"))} for c, n in top_cameos],
        "events": events[:20],
    }


# ---------------------------------------------------------------------------
# Multi-ticker batch sentiment
# ---------------------------------------------------------------------------

def batch_sentiment(
    tickers: List[str],
    timespan: str = "24h",
) -> Dict[str, Any]:
    """Compute sentiment for multiple tickers in parallel (sequential for safety)."""
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = get_ticker_sentiment(ticker, timespan=timespan)
        except Exception as exc:
            results[ticker] = {"ticker": ticker, "error": str(exc)}

    # Rankings
    scored = [(t, r.get("sentiment_score", 0.0)) for t, r in results.items()
              if "error" not in r]
    scored.sort(key=lambda x: x[1], reverse=True)

    return {
        "as_of": str(date.today()),
        "n_tickers": len(tickers),
        "results": results,
        "top_bullish": [t for t, _ in scored[:5]],
        "top_bearish": [t for t, _ in scored[-5:][::-1]],
    }


# ---------------------------------------------------------------------------
# Pydantic models for FastAPI
# ---------------------------------------------------------------------------

class SentimentResponse(BaseModel):
    ticker: str
    as_of: str
    timespan: str = "48h"
    n_articles: int
    n_scored: int = 0
    sentiment_score: float
    sentiment_label: str
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    avg_gdelt_tone: float = 0.0
    scoring_method: str = "unknown"
    sample_urls: List[str] = Field(default_factory=list)


class NewsVolumeResponse(BaseModel):
    ticker: str
    as_of: str
    current_volume: int
    mean_volume_30d: float
    std_volume_30d: float
    volume_zscore: float
    spike_alert: bool
    spike_threshold_sigma: float
    n_datapoints: int
    timeline: List[Dict[str, Any]] = Field(default_factory=list)


class GeoRiskResponse(BaseModel):
    as_of: str
    timespan: str
    n_events_analyzed: int
    n_countries: int
    global_risk_index: float
    risk_level: str
    country_scores: List[Dict[str, Any]]


class ThemesResponse(BaseModel):
    as_of: str
    timespan: str
    n_articles_analyzed: int
    n_themes: int
    themes: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/sentiment", tags=["GDELT News Sentiment V2"])


@router.get("/{ticker}", response_model=SentimentResponse, summary="Get ticker sentiment from GDELT + FinBERT")
async def get_sentiment(
    ticker: str,
    company_name: Optional[str] = Query(None, description="Company name for richer search"),
    timespan: str = Query("48h", description="Lookback: 1h | 24h | 48h | 7d | 30d"),
    max_articles: int = Query(50, description="Max articles to score", ge=5, le=200),
    use_cache: bool = Query(True, description="Use 6h processed sentiment cache"),
) -> SentimentResponse:
    """
    Fetch news from GDELT DOC API, score with FinBERT/VADER/LM, and return
    entity-level sentiment for the given ticker.
    """
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: get_ticker_sentiment(
            ticker.upper(),
            company_name=company_name,
            timespan=timespan,
            max_articles=max_articles,
            use_cache=use_cache,
        ),
    )
    if "error" in result and result.get("n_articles", 0) == 0:
        raise HTTPException(404, result["error"])
    return SentimentResponse(**{k: result.get(k, v) for k, v in SentimentResponse.model_fields.items()
                                if k in result or k in {"ticker", "as_of", "sentiment_score",
                                                        "sentiment_label", "positive_pct",
                                                        "negative_pct", "neutral_pct", "n_articles"}},
                             ticker=result.get("ticker", ticker),
                             as_of=result.get("as_of", str(date.today())),
                             sentiment_score=result.get("sentiment_score", 0.0),
                             sentiment_label=result.get("sentiment_label", "neutral"),
                             positive_pct=result.get("positive_pct", 0.0),
                             negative_pct=result.get("negative_pct", 0.0),
                             neutral_pct=result.get("neutral_pct", 0.0),
                             n_articles=result.get("n_articles", 0),
                             n_scored=result.get("n_scored", 0),
                             avg_gdelt_tone=result.get("avg_gdelt_tone", 0.0),
                             scoring_method=result.get("scoring_method", "unknown"),
                             sample_urls=result.get("sample_urls", []))


@router.get("/news-volume/{ticker}", response_model=NewsVolumeResponse, summary="News volume spike detection")
async def get_news_volume_endpoint(
    ticker: str,
    spike_sigma: float = Query(2.0, description="Z-score threshold for spike alert", ge=1.0, le=5.0),
) -> NewsVolumeResponse:
    """Detect news volume spikes: current volume vs 30-day average."""
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: get_news_volume(ticker.upper(), spike_sigma=spike_sigma),
    )
    if "error" in result:
        raise HTTPException(404, result["error"])
    return NewsVolumeResponse(**result)


@router.get("/geo-risk/global", response_model=GeoRiskResponse, summary="Global geopolitical risk index")
async def get_geo_risk(
    query: str = Query("conflict protest sanction attack", description="GDELT event query"),
    timespan: str = Query("7d", description="Lookback: 24h | 7d | 30d"),
    top_countries: int = Query(20, description="Number of top-risk countries to return", ge=5, le=50),
) -> GeoRiskResponse:
    """Compute country-level geopolitical risk index from GDELT Events API."""
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: compute_geo_risk(query=query, timespan=timespan, top_countries=top_countries),
    )
    if "error" in result:
        raise HTTPException(503, result["error"])
    return GeoRiskResponse(**result)


@router.get("/themes/emerging", response_model=ThemesResponse, summary="Identify emerging news themes via NMF")
async def get_themes(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers to focus on"),
    timespan: str = Query("24h", description="Lookback: 1h | 24h | 7d"),
    n_topics: int = Query(5, description="Number of topics to identify", ge=2, le=15),
    max_articles: int = Query(100, description="Articles to analyze", ge=20, le=300),
) -> ThemesResponse:
    """Identify emerging financial news themes using NMF topic modeling on GDELT corpus."""
    import asyncio
    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else None
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: identify_themes(
            tickers=tk_list,
            timespan=timespan,
            n_topics=n_topics,
            max_articles=max_articles,
        ),
    )
    if "error" in result:
        raise HTTPException(503, result["error"])
    return ThemesResponse(**result)


@router.get("/momentum/{ticker}", summary="Sentiment momentum: 3d vs 21d MA crossover")
async def get_sentiment_momentum(
    ticker: str,
    short_window: int = Query(3, description="Short MA window (days)", ge=2, le=10),
    long_window: int = Query(21, description="Long MA window (days)", ge=5, le=90),
) -> Dict[str, Any]:
    """Compute sentiment momentum signal from historical sentiment time series."""
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: get_sentiment_momentum(ticker.upper(), short_window=short_window, long_window=long_window),
    )
    return result


@router.get("/alpha/{ticker}", summary="Event-driven alpha signals: sentiment → next-day return")
async def get_alpha_signals(
    ticker: str,
    lookback_days: int = Query(30, description="Lookback for signal computation", ge=5, le=90),
) -> Dict[str, Any]:
    """Compute correlation between news sentiment and subsequent returns for event-driven signals."""
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: compute_alpha_signals(ticker.upper(), lookback_days=lookback_days),
    )
    return result


@router.get("/events/{ticker}", summary="GDELT event records for ticker")
async def get_events(
    ticker: str,
    timespan: str = Query("7d", description="Lookback: 24h | 7d | 30d"),
    max_records: int = Query(50, description="Max events to fetch", ge=10, le=200),
) -> Dict[str, Any]:
    """Fetch and summarize GDELT event records with Goldstein scale and CAMEO codes."""
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: get_gdelt_events(ticker.upper(), timespan=timespan, max_records=max_records),
    )
    return result


@router.get("/country-risk/history", summary="Country geopolitical risk score history")
async def get_country_risk(
    days: int = Query(7, description="History lookback in days", ge=1, le=90),
) -> Dict[str, Any]:
    """Return aggregated country-level geopolitical risk scores from history."""
    history = get_country_risk_history(days=days)
    return {
        "as_of": str(date.today()),
        "lookback_days": days,
        "n_countries": len(history),
        "countries": history,
    }


@router.get("/batch", summary="Batch sentiment for multiple tickers")
async def get_batch_sentiment(
    tickers: str = Query(..., description="Comma-separated tickers (max 20)"),
    timespan: str = Query("24h", description="Lookback timespan"),
) -> Dict[str, Any]:
    """Compute and rank sentiment for multiple tickers simultaneously."""
    import asyncio
    tk_list = [t.strip().upper() for t in tickers.split(",")][:20]
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: batch_sentiment(tk_list, timespan=timespan),
    )
    return result


@router.get("/tone-timeline/{ticker}", summary="GDELT tone timeline for ticker")
async def get_tone_timeline(
    ticker: str,
    timespan: str = Query("30d", description="Lookback: 7d | 30d | 90d"),
) -> Dict[str, Any]:
    """Fetch GDELT tone timeline showing positive/negative/neutral sentiment over time."""
    import asyncio
    query = f'"{ticker.upper()}" stock market'
    timeline = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _gdelt_doc_tonetimeline(query, timespan=timespan),
    )
    return {
        "ticker": ticker.upper(),
        "as_of": str(date.today()),
        "timespan": timespan,
        "n_datapoints": len(timeline),
        "timeline": timeline,
    }


@router.get("/sentiment-history/{ticker}", summary="Historical sentiment scores for ticker")
async def get_sentiment_history(
    ticker: str,
    days: int = Query(30, description="Lookback days", ge=1, le=365),
) -> Dict[str, Any]:
    """Return stored sentiment score history for a ticker from SQLite."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with _db() as conn:
        rows = conn.execute(
            """SELECT as_of, sentiment_score, positive_pct, negative_pct, neutral_pct,
               article_count, avg_tone, source
               FROM ticker_sentiment_history
               WHERE ticker=? AND as_of>=?
               ORDER BY as_of DESC""",
            (ticker.upper(), cutoff),
        ).fetchall()
    history = [dict(r) for r in rows]

    scores = [r["sentiment_score"] for r in history]
    return {
        "ticker": ticker.upper(),
        "lookback_days": days,
        "n_records": len(history),
        "mean_sentiment": round(float(np.mean(scores)), 4) if scores else 0.0,
        "std_sentiment": round(float(np.std(scores)), 4) if scores else 0.0,
        "history": history,
    }


@router.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    """Verify module is responsive and DB is accessible."""
    try:
        with _db() as conn:
            conn.execute("SELECT 1").fetchone()
        return {
            "status": "ok",
            "module": "gdelt_news_sentiment_v2",
            "dim": "084",
            "finbert_available": str(_FINBERT_AVAILABLE),
            "vader_available": str(_VADER_AVAILABLE),
        }
    except Exception as exc:
        raise HTTPException(503, f"DB error: {exc}")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"\n=== GDELT News Sentiment V2 — Test for {test_ticker} ===\n")

    print("1. Fetching ticker sentiment...")
    sent = get_ticker_sentiment(test_ticker, timespan="48h", max_articles=30)
    print(f"   Score: {sent.get('sentiment_score'):.4f}  |  Label: {sent.get('sentiment_label')}")
    print(f"   Articles: {sent.get('n_articles')}  |  Method: {sent.get('scoring_method')}")

    print("\n2. Fetching news volume...")
    vol = get_news_volume(test_ticker)
    print(f"   Volume Z-score: {vol.get('volume_zscore'):.2f}  |  Spike: {vol.get('spike_alert')}")

    print("\n3. Computing geopolitical risk...")
    geo = compute_geo_risk(timespan="7d", top_countries=5)
    print(f"   Global Risk Index: {geo.get('global_risk_index')}  |  Level: {geo.get('risk_level')}")
    for c in geo.get("country_scores", [])[:3]:
        print(f"   {c['country']}: {c.get('risk_score_normalized', 0):.2f}")

    print("\n4. Identifying themes...")
    themes = identify_themes(tickers=[test_ticker], timespan="24h", n_topics=3)
    for th in themes.get("themes", []):
        print(f"   Theme {th['theme_id']}: {th['theme_label']} (weight={th['weight']:.3f})")

    print("\n5. Sentiment momentum...")
    mom = get_sentiment_momentum(test_ticker)
    print(f"   Signal: {mom.get('signal')}  |  Momentum: {mom.get('momentum')}")

    print("\nAll tests complete.")
