"""
Google Trends Signals v2 — Dimension #089 (target score 9).

Comprehensive Google Trends signal system using pytrends with advanced
financial signal extraction, statistical analysis, and production-grade caching.

Extends sentinel/sma/google_trends_signals.py with:
  - Multi-keyword normalization across ticker variants
  - Trend momentum (4-week vs 52-week ratio)
  - Surge detection via z-score (threshold 2.0)
  - Geographic interest heatmaps (state/country breakdown)
  - Category-level macro sentiment (investing, recession, inflation, crypto)
  - Fear/greed proxy: "buy stocks" vs "sell stocks" ratio
  - Competitor brand tracking and relative search share
  - Seasonal adjustment (STL / rolling deviation)
  - Granger causality: trends → returns (lag 1-4 weeks)
  - Topic clustering: thematic grouping of related searches
  - Dual SQLite cache: 4h TTL (raw) / 24h TTL (signals)

Bloomberg equivalent: None — SENTINEL exclusive alternative data signal.

Free data sources:
  - pytrends: unofficial Google Trends API wrapper
  - No API key required (respects rate limits via jitter + backoff)

Public API
----------
TrendsCacheV2
    get_raw(key)                          -> Optional[str]
    set_raw(key, payload, ttl_hours)      -> None
    get_signal(key)                       -> Optional[dict]
    set_signal(key, data)                 -> None

GoogleTrendsAdapterV2
    interest_over_time(keywords, timeframe, geo)  -> pd.DataFrame
    interest_by_region(keywords, geo, resolution) -> pd.DataFrame
    related_queries(keyword)                      -> dict
    related_topics(keyword)                       -> dict

TrendsSignalEngineV2
    compute_momentum(ticker)              -> TrendsMomentumSignal
    detect_surge(ticker)                  -> SurgeAlert
    geo_heatmap(ticker)                   -> GeoHeatmapSignal
    macro_sentiment()                     -> MacroSentimentSignal
    fear_greed_proxy()                    -> FearGreedSignal
    competitor_share(ticker, peers)       -> CompetitorShareSignal
    seasonal_adjust(ticker)               -> pd.DataFrame
    granger_causality(ticker, prices)     -> GrangerResult
    topic_clusters(ticker)               -> TopicClusterSignal

google_trends_v2_router  — FastAPI router, prefix /trends/v2
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from scipy import stats
from scipy.signal import periodogram

logger = logging.getLogger(__name__)

__all__ = [
    "TrendsCacheV2",
    "GoogleTrendsAdapterV2",
    "TrendsSignalEngineV2",
    "google_trends_v2_router",
    "TrendsMomentumSignal",
    "SurgeAlert",
    "GeoHeatmapSignal",
    "MacroSentimentSignal",
    "FearGreedSignal",
    "CompetitorShareSignal",
    "GrangerResult",
    "TopicClusterSignal",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(".sentinel") / "cache"
_CACHE_DB_V2 = _CACHE_DIR / "google_trends_v2.db"

_RAW_TTL_HOURS = 4.0       # raw pytrends responses
_SIGNAL_TTL_HOURS = 24.0   # processed signals

_RATE_LIMIT_MIN = 4.0      # seconds
_RATE_LIMIT_MAX = 9.0      # seconds
_BACKOFF_BASE = 2.0
_MAX_RETRIES = 4

_SURGE_ZSCORE_THRESHOLD = 2.0
_MOMENTUM_ACCEL_THRESHOLD = 0.20   # 20% above 52-week avg = accelerating
_MOMENTUM_DECEL_THRESHOLD = -0.20

_GRANGER_MAX_LAG = 4       # test lags 1-4 weeks

# pytrends timeframe shortcuts
TF_1M = "today 1-m"
TF_3M = "today 3-m"
TF_12M = "today 12-m"
TF_5Y = "today 5-y"

# Macro sentiment keyword groups
MACRO_SENTIMENT_KEYWORDS: dict[str, list[str]] = {
    "recession": ["recession", "economic recession", "recession 2025"],
    "inflation": ["inflation", "CPI", "cost of living"],
    "fed_rate": ["federal reserve rate hike", "Fed interest rates", "FOMC"],
    "market_crash": ["stock market crash", "market sell off", "bear market"],
    "bull_market": ["bull market", "stock market rally", "S&P 500 high"],
    "unemployment": ["unemployment", "job losses", "layoffs"],
    "crypto_bull": ["Bitcoin buy", "crypto rally", "crypto bull run"],
    "crypto_bear": ["crypto crash", "Bitcoin sell", "crypto bear market"],
}

# Fear/Greed proxy keywords
FEAR_KEYWORDS = [
    "sell stocks", "sell my stocks", "stock market crash", "bear market",
    "recession stocks", "how to protect portfolio", "defensive stocks",
    "market collapse", "short selling", "put options",
]
GREED_KEYWORDS = [
    "buy stocks", "stock tips", "best stocks to buy", "bull market",
    "growth stocks", "momentum stocks", "stock market highs",
    "call options", "YOLO stocks", "meme stocks buy",
]

# Known sector peer groups for competitor share analysis
SECTOR_PEERS: dict[str, list[str]] = {
    "AAPL": ["Apple", "Samsung", "Google Pixel", "Microsoft Surface"],
    "MSFT": ["Microsoft", "Google Workspace", "Salesforce", "SAP"],
    "GOOGL": ["Google", "Bing", "DuckDuckGo", "Yahoo Search"],
    "AMZN": ["Amazon", "eBay", "Walmart online", "Shopify"],
    "META": ["Facebook", "Instagram", "TikTok", "Snapchat", "Twitter"],
    "NFLX": ["Netflix", "Hulu", "Disney Plus", "HBO Max", "Amazon Prime Video"],
    "TSLA": ["Tesla", "Rivian", "Lucid Motors", "Ford EV", "GM electric"],
    "NVDA": ["Nvidia", "AMD GPU", "Intel GPU", "Qualcomm chips"],
    "CRM": ["Salesforce", "HubSpot", "Microsoft Dynamics", "Zoho CRM"],
    "SNOW": ["Snowflake", "Databricks", "Google BigQuery", "Amazon Redshift"],
    "UBER": ["Uber", "Lyft", "DoorDash", "Instacart"],
    "ABNB": ["Airbnb", "VRBO", "Booking.com", "Hotels.com"],
    "COIN": ["Coinbase", "Binance", "Kraken", "Gemini"],
    "DDOG": ["Datadog", "New Relic", "Dynatrace", "Splunk"],
    "CRWD": ["CrowdStrike", "Palo Alto Networks", "SentinelOne", "Carbon Black"],
    "SHOP": ["Shopify", "WooCommerce", "BigCommerce", "Squarespace"],
    "ZM": ["Zoom", "Microsoft Teams", "Google Meet", "Webex"],
}

# Ticker search term variants (what people actually search for)
TICKER_SEARCH_VARIANTS: dict[str, list[str]] = {
    "AAPL": ["Apple stock", "AAPL", "Apple Inc"],
    "MSFT": ["Microsoft stock", "MSFT", "Microsoft shares"],
    "GOOGL": ["Google stock", "GOOGL", "Alphabet stock"],
    "AMZN": ["Amazon stock", "AMZN", "Amazon shares"],
    "META": ["Meta stock", "META", "Facebook stock"],
    "NFLX": ["Netflix stock", "NFLX", "Netflix shares"],
    "TSLA": ["Tesla stock", "TSLA", "Tesla shares"],
    "NVDA": ["Nvidia stock", "NVDA", "Nvidia shares"],
    "AMD": ["AMD stock", "AMD semiconductor", "Advanced Micro Devices"],
    "CRM": ["Salesforce stock", "CRM", "Salesforce shares"],
    "SNOW": ["Snowflake stock", "SNOW", "Snowflake IPO"],
    "PLTR": ["Palantir stock", "PLTR", "Palantir shares"],
    "COIN": ["Coinbase stock", "COIN", "Coinbase crypto"],
    "UBER": ["Uber stock", "UBER", "Uber shares"],
    "ABNB": ["Airbnb stock", "ABNB", "Airbnb IPO"],
}

# Geographic resolution options for pytrends
GEO_RESOLUTIONS = {
    "country": "COUNTRY",
    "state": "REGION",
    "metro": "DMA",
    "city": "CITY",
}

# US state abbreviations for filtering
US_STATES = [
    "US-CA", "US-NY", "US-TX", "US-FL", "US-WA", "US-MA", "US-IL",
    "US-CO", "US-GA", "US-AZ", "US-NC", "US-VA", "US-OH", "US-NJ",
    "US-MI", "US-PA", "US-MN", "US-OR", "US-UT", "US-NV", "US-TN",
]

# Category-level keywords for sector rotation signals
SECTOR_ROTATION_KEYWORDS: dict[str, str] = {
    "technology_interest": "buy tech stocks",
    "energy_interest": "oil stocks investment",
    "healthcare_interest": "biotech stocks",
    "consumer_interest": "retail stocks",
    "financial_interest": "bank stocks investment",
    "realestate_interest": "REIT investment",
    "utilities_interest": "utility stocks dividend",
    "materials_interest": "mining stocks copper",
}

_HTTP_HEADERS = {
    "User-Agent": "SENTINEL:FinancialTerminal:2.0 (research; contact: sentinel@example.com)",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrendsMomentumSignal:
    ticker: str
    as_of: date
    current_interest: float         # latest week interest (0-100)
    avg_4w: float                   # 4-week average
    avg_52w: float                  # 52-week average
    momentum_ratio_4w: float        # 4w avg / 52w avg
    trend_direction: str            # "bullish" | "bearish" | "neutral"
    acceleration: float             # week-over-week change in interest
    search_variants_normalized: dict[str, float]
    interpretation: str


@dataclass
class SurgeAlert:
    ticker: str
    keyword: str
    as_of: date
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    is_surge: bool
    surge_pct_above_baseline: float
    probable_trigger: str
    severity: str                   # "low" | "moderate" | "high" | "extreme"


@dataclass
class GeoHeatmapSignal:
    ticker: str
    keyword: str
    as_of: date
    country_interest: dict[str, float]       # country_code -> interest
    us_state_interest: dict[str, float]      # state_code -> interest
    top_countries: list[tuple[str, float]]
    top_us_states: list[tuple[str, float]]
    geographic_concentration: float          # HHI-like concentration index 0-1
    primary_market: str
    interpretation: str


@dataclass
class MacroSentimentSignal:
    as_of: date
    recession_score: float           # 0-100, current search interest
    inflation_score: float
    market_crash_fear: float
    bull_market_optimism: float
    crypto_sentiment: float          # net: crypto_bull - crypto_bear
    fed_rate_anxiety: float
    unemployment_fear: float
    composite_fear_index: float      # 0-100, higher = more fear
    composite_greed_index: float     # 0-100, higher = more greed
    regime: str                      # "fear" | "greed" | "neutral"
    interpretation: str


@dataclass
class FearGreedSignal:
    as_of: date
    fear_score: float                # average interest in fear keywords (0-100)
    greed_score: float               # average interest in greed keywords (0-100)
    fear_greed_ratio: float          # fear/greed, >1 = fearful market
    net_signal: float                # greed - fear, positive = greed
    signal: str                      # "extreme_fear" | "fear" | "neutral" | "greed" | "extreme_greed"
    z_score_vs_1y: float
    weekly_series: dict[str, float]  # iso_date -> net_signal
    interpretation: str


@dataclass
class CompetitorShareSignal:
    ticker: str
    as_of: date
    keywords: list[str]
    relative_share: dict[str, float]        # keyword -> % share of total interest
    dominant_brand: str
    ticker_share_pct: float
    share_vs_prior_month: float             # change in ticker's share %
    trend: str                              # "gaining" | "losing" | "stable"
    interpretation: str


@dataclass
class GrangerResult:
    ticker: str
    keyword: str
    lag_weeks: int
    f_statistic: float
    p_value: float
    is_significant: bool
    causality_direction: str        # "trends→returns" | "returns→trends" | "bidirectional" | "none"
    optimal_lag: int
    r_squared: float
    interpretation: str


@dataclass
class TopicClusterSignal:
    ticker: str
    as_of: date
    clusters: dict[str, list[str]]          # theme_name -> related queries
    dominant_theme: str
    theme_scores: dict[str, float]          # theme -> average interest score
    rising_queries: list[str]               # rapidly rising related searches
    top_queries: list[str]                  # top related searches by volume
    interpretation: str


# ---------------------------------------------------------------------------
# SQLite Cache v2 — dual TTL (raw + signals)
# ---------------------------------------------------------------------------


class TrendsCacheV2:
    """
    Dual-TTL SQLite cache for Google Trends data.

    Tables
    ------
    trends_raw        — 4-hour TTL for raw pytrends API responses
    trends_signals    — 24-hour TTL for processed signal objects
    granger_cache     — 7-day TTL for expensive Granger causality results
    """

    def __init__(self, db_path: Path = _CACHE_DB_V2) -> None:
        self._db = db_path
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS trends_raw (
                    cache_key   TEXT PRIMARY KEY,
                    payload     TEXT NOT NULL,
                    cached_at   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_raw_cached_at
                    ON trends_raw(cached_at);

                CREATE TABLE IF NOT EXISTS trends_signals (
                    cache_key   TEXT PRIMARY KEY,
                    payload     TEXT NOT NULL,
                    cached_at   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_sig_cached_at
                    ON trends_signals(cached_at);

                CREATE TABLE IF NOT EXISTS granger_cache (
                    cache_key   TEXT PRIMARY KEY,
                    payload     TEXT NOT NULL,
                    cached_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS surge_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT NOT NULL,
                    keyword     TEXT NOT NULL,
                    z_score     REAL NOT NULL,
                    is_surge    INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_surge_ticker
                    ON surge_history(ticker, recorded_at);
                """
            )

    # ── raw cache (4h TTL) ────────────────────────────────────────────────────

    @staticmethod
    def make_key(*parts: Any) -> str:
        raw = json.dumps(list(parts), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_raw(self, key: str) -> Optional[str]:
        cutoff = (datetime.utcnow() - timedelta(hours=_RAW_TTL_HOURS)).isoformat()
        with self._conn() as con:
            row = con.execute(
                "SELECT payload FROM trends_raw WHERE cache_key=? AND cached_at>=?",
                (key, cutoff),
            ).fetchone()
        return row["payload"] if row else None

    def set_raw(self, key: str, payload: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO trends_raw(cache_key, payload, cached_at) VALUES(?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload=excluded.payload, cached_at=excluded.cached_at
                """,
                (key, payload, now),
            )

    # ── signal cache (24h TTL) ────────────────────────────────────────────────

    def get_signal(self, key: str) -> Optional[dict]:
        cutoff = (datetime.utcnow() - timedelta(hours=_SIGNAL_TTL_HOURS)).isoformat()
        with self._conn() as con:
            row = con.execute(
                "SELECT payload FROM trends_signals WHERE cache_key=? AND cached_at>=?",
                (key, cutoff),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except Exception:
            return None

    def set_signal(self, key: str, data: dict) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO trends_signals(cache_key, payload, cached_at) VALUES(?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload=excluded.payload, cached_at=excluded.cached_at
                """,
                (key, json.dumps(data, default=str), now),
            )

    # ── Granger cache (7d TTL) ────────────────────────────────────────────────

    def get_granger(self, key: str) -> Optional[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        with self._conn() as con:
            row = con.execute(
                "SELECT payload FROM granger_cache WHERE cache_key=? AND cached_at>=?",
                (key, cutoff),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def set_granger(self, key: str, data: dict) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO granger_cache(cache_key, payload, cached_at) VALUES(?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload=excluded.payload, cached_at=excluded.cached_at
                """,
                (key, json.dumps(data, default=str), now),
            )

    # ── surge history ─────────────────────────────────────────────────────────

    def record_surge(
        self, ticker: str, keyword: str, z_score: float, is_surge: bool
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as con:
            con.execute(
                "INSERT INTO surge_history(ticker, keyword, z_score, is_surge, recorded_at)"
                " VALUES(?,?,?,?,?)",
                (ticker, keyword, z_score, int(is_surge), now),
            )

    def get_surge_history(self, ticker: str, days: int = 30) -> list[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as con:
            rows = con.execute(
                "SELECT keyword, z_score, is_surge, recorded_at FROM surge_history"
                " WHERE ticker=? AND recorded_at>=? ORDER BY recorded_at DESC",
                (ticker, cutoff),
            ).fetchall()
        return [dict(r) for r in rows]

    def evict_expired(self) -> dict[str, int]:
        cutoff_raw = (datetime.utcnow() - timedelta(hours=_RAW_TTL_HOURS * 2)).isoformat()
        cutoff_sig = (datetime.utcnow() - timedelta(hours=_SIGNAL_TTL_HOURS * 2)).isoformat()
        with self._conn() as con:
            r1 = con.execute(
                "DELETE FROM trends_raw WHERE cached_at<?", (cutoff_raw,)
            ).rowcount
            r2 = con.execute(
                "DELETE FROM trends_signals WHERE cached_at<?", (cutoff_sig,)
            ).rowcount
        return {"raw_evicted": r1, "signals_evicted": r2}


# ---------------------------------------------------------------------------
# pytrends Rate-Limit Helper
# ---------------------------------------------------------------------------


def _jitter_sleep(min_s: float = _RATE_LIMIT_MIN, max_s: float = _RATE_LIMIT_MAX) -> None:
    """Sleep a random interval to avoid triggering Google's rate limiter."""
    t = random.uniform(min_s, max_s)
    time.sleep(t)


def _build_pytrends(retries: int = 3, hl: str = "en-US", tz: int = 360):
    """
    Import and initialize pytrends TrendReq with exponential backoff on failure.
    Returns None if pytrends is not installed.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends not installed. Run: pip install pytrends")
        return None

    for attempt in range(retries):
        try:
            pt = TrendReq(hl=hl, tz=tz, retries=retries, backoff_factor=0.5)
            return pt
        except Exception as exc:
            wait = _BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 1)
            logger.warning("pytrends init attempt %d/%d failed: %s — waiting %.1fs",
                           attempt + 1, retries, exc, wait)
            if attempt < retries - 1:
                time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Google Trends Adapter v2
# ---------------------------------------------------------------------------


class GoogleTrendsAdapterV2:
    """
    Production-grade, cache-aware pytrends wrapper with:
      - Exponential backoff on 429 / connection errors
      - SQLite caching with 4-hour TTL (raw) and 24-hour TTL (signals)
      - Automatic keyword chunking (pytrends limit: 5 per request)
      - Graceful degradation on rate limit — returns empty DataFrame

    All public methods accept ``use_cache=True`` to serve from cache.
    """

    def __init__(
        self,
        cache: Optional[TrendsCacheV2] = None,
        hl: str = "en-US",
        tz: int = 360,
        retries: int = _MAX_RETRIES,
    ) -> None:
        self._cache = cache or TrendsCacheV2()
        self._hl = hl
        self._tz = tz
        self._retries = retries
        self._pt = None   # lazy-initialized

    def _ensure_pytrends(self):
        """Lazy-initialize pytrends connection."""
        if self._pt is None:
            self._pt = _build_pytrends(retries=self._retries, hl=self._hl, tz=self._tz)
        return self._pt

    def _build_payload_with_retry(
        self,
        keywords: list[str],
        timeframe: str = TF_12M,
        geo: str = "",
        cat: int = 0,
    ) -> bool:
        """Build pytrends payload with retry + backoff. Returns True on success."""
        pt = self._ensure_pytrends()
        if pt is None:
            return False

        for attempt in range(self._retries):
            try:
                pt.build_payload(keywords, timeframe=timeframe, geo=geo, cat=cat)
                return True
            except Exception as exc:
                wait = _BACKOFF_BASE ** (attempt + 1) + random.uniform(0.5, 2.0)
                err_str = str(exc).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    logger.warning(
                        "pytrends rate limit attempt %d/%d — sleeping %.1fs",
                        attempt + 1, self._retries, wait
                    )
                else:
                    logger.warning(
                        "pytrends payload error attempt %d/%d: %s — sleeping %.1fs",
                        attempt + 1, self._retries, exc, wait
                    )
                if attempt < self._retries - 1:
                    time.sleep(wait)
        return False

    def interest_over_time(
        self,
        keywords: list[str],
        timeframe: str = TF_12M,
        geo: str = "",
        cat: int = 0,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch interest_over_time for up to 5 keywords.

        Returns DataFrame indexed by date with keyword columns (0-100 scale).
        Empty DataFrame on failure or rate limit.
        """
        # Trim to pytrends limit
        kw_list = keywords[:5]
        cache_key = self._cache.make_key("iot", kw_list, timeframe, geo, cat)

        if use_cache:
            cached = self._cache.get_raw(cache_key)
            if cached:
                try:
                    return pd.read_json(cached)
                except Exception:
                    pass

        pt = self._ensure_pytrends()
        if pt is None:
            return pd.DataFrame()

        ok = self._build_payload_with_retry(kw_list, timeframe=timeframe, geo=geo, cat=cat)
        if not ok:
            return pd.DataFrame()

        try:
            _jitter_sleep()
            df = pt.interest_over_time()
            if df.empty:
                return pd.DataFrame()
            # Drop 'isPartial' column if present
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            if use_cache:
                self._cache.set_raw(cache_key, df.to_json())
            return df
        except Exception as exc:
            logger.error("interest_over_time failed: %s", exc)
            return pd.DataFrame()

    def interest_by_region(
        self,
        keywords: list[str],
        geo: str = "US",
        resolution: str = "REGION",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch interest_by_region breakdown (country, state, DMA, city).

        Parameters
        ----------
        keywords:
            Up to 5 keywords to compare.
        geo:
            Geographic scope, e.g. "US", "GB", "" (worldwide).
        resolution:
            "COUNTRY", "REGION" (US states), "DMA", or "CITY".
        """
        kw_list = keywords[:5]
        cache_key = self._cache.make_key("ibr", kw_list, geo, resolution)

        if use_cache:
            cached = self._cache.get_raw(cache_key)
            if cached:
                try:
                    return pd.read_json(cached)
                except Exception:
                    pass

        pt = self._ensure_pytrends()
        if pt is None:
            return pd.DataFrame()

        ok = self._build_payload_with_retry(kw_list, timeframe=TF_12M, geo=geo)
        if not ok:
            return pd.DataFrame()

        try:
            _jitter_sleep()
            df = pt.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=True)
            if df.empty:
                return pd.DataFrame()
            if use_cache:
                self._cache.set_raw(cache_key, df.to_json())
            return df
        except Exception as exc:
            logger.error("interest_by_region failed: %s", exc)
            return pd.DataFrame()

    def related_queries(
        self,
        keyword: str,
        timeframe: str = TF_12M,
        geo: str = "",
        use_cache: bool = True,
    ) -> dict:
        """
        Fetch related queries (top + rising) for a keyword.

        Returns dict: {keyword: {"top": DataFrame, "rising": DataFrame}}
        """
        cache_key = self._cache.make_key("rq", keyword, timeframe, geo)

        if use_cache:
            cached = self._cache.get_raw(cache_key)
            if cached:
                try:
                    data = json.loads(cached)
                    result = {}
                    for kw, sub in data.items():
                        result[kw] = {
                            "top": pd.read_json(sub["top"]) if sub.get("top") else pd.DataFrame(),
                            "rising": pd.read_json(sub["rising"]) if sub.get("rising") else pd.DataFrame(),
                        }
                    return result
                except Exception:
                    pass

        pt = self._ensure_pytrends()
        if pt is None:
            return {}

        ok = self._build_payload_with_retry([keyword], timeframe=timeframe, geo=geo)
        if not ok:
            return {}

        try:
            _jitter_sleep()
            rq = pt.related_queries()
            if use_cache and rq:
                # Serialize for cache
                serializable = {}
                for kw, sub in rq.items():
                    serializable[kw] = {
                        "top": sub["top"].to_json() if sub.get("top") is not None else None,
                        "rising": sub["rising"].to_json() if sub.get("rising") is not None else None,
                    }
                self._cache.set_raw(cache_key, json.dumps(serializable))
            return rq or {}
        except Exception as exc:
            logger.error("related_queries failed for '%s': %s", keyword, exc)
            return {}

    def related_topics(
        self,
        keyword: str,
        timeframe: str = TF_12M,
        geo: str = "",
        use_cache: bool = True,
    ) -> dict:
        """Fetch related topics (top + rising) for a keyword."""
        cache_key = self._cache.make_key("rt", keyword, timeframe, geo)

        if use_cache:
            cached = self._cache.get_raw(cache_key)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass

        pt = self._ensure_pytrends()
        if pt is None:
            return {}

        ok = self._build_payload_with_retry([keyword], timeframe=timeframe, geo=geo)
        if not ok:
            return {}

        try:
            _jitter_sleep()
            rt = pt.related_topics()
            if use_cache and rt:
                # Serialize topic DataFrames to JSON-safe dict
                out: dict = {}
                for kw, sub in rt.items():
                    out[kw] = {}
                    for k, v in sub.items():
                        if isinstance(v, pd.DataFrame):
                            out[kw][k] = v.to_dict(orient="records")
                        else:
                            out[kw][k] = v
                self._cache.set_raw(cache_key, json.dumps(out, default=str))
            return rt or {}
        except Exception as exc:
            logger.error("related_topics failed for '%s': %s", keyword, exc)
            return {}

    def multiquery_normalize(
        self,
        keywords: list[str],
        timeframe: str = TF_12M,
        geo: str = "",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch interest_over_time for more than 5 keywords by using an anchor term
        to normalize across multiple batches.

        Anchor approach: include a stable reference keyword (e.g., "the") in each
        5-keyword batch, then rescale all batches to the anchor's baseline.

        Parameters
        ----------
        keywords:
            List of keywords (any length).
        """
        if len(keywords) <= 5:
            return self.interest_over_time(keywords, timeframe=timeframe, geo=geo,
                                           use_cache=use_cache)

        anchor = keywords[0]
        batches: list[pd.DataFrame] = []
        remaining = keywords[1:]

        while remaining:
            batch = [anchor] + remaining[:4]
            remaining = remaining[4:]

            df_batch = self.interest_over_time(batch, timeframe=timeframe, geo=geo,
                                               use_cache=use_cache)
            if df_batch.empty:
                continue

            # Normalize by anchor column
            if anchor in df_batch.columns:
                anchor_vals = df_batch[anchor].replace(0, np.nan)
                for col in df_batch.columns:
                    if col != anchor:
                        df_batch[col] = df_batch[col] / anchor_vals
            batches.append(df_batch)
            _jitter_sleep()

        if not batches:
            return pd.DataFrame()

        # Merge on index, drop duplicated anchor columns
        result = batches[0]
        for df_extra in batches[1:]:
            extra_cols = [c for c in df_extra.columns if c not in result.columns]
            result = result.join(df_extra[extra_cols], how="outer")

        result = result.fillna(0)
        return result


# ---------------------------------------------------------------------------
# Trends Signal Engine v2
# ---------------------------------------------------------------------------


class TrendsSignalEngineV2:
    """
    Derives actionable financial signals from Google Trends data.

    All heavy computation is cached in SQLite (raw: 4h, signals: 24h).
    """

    def __init__(
        self,
        adapter: Optional[GoogleTrendsAdapterV2] = None,
        cache: Optional[TrendsCacheV2] = None,
    ) -> None:
        self._adapter = adapter or GoogleTrendsAdapterV2()
        self._cache = cache or TrendsCacheV2()

    # ── 1. Trend Momentum Signal ───────────────────────────────────────────────

    def compute_momentum(
        self,
        ticker: str,
        timeframe: str = TF_12M,
        geo: str = "",
        use_cache: bool = True,
    ) -> TrendsMomentumSignal:
        """
        Compute 4-week vs 52-week trend momentum ratio.

        Methodology:
          1. Fetch 52-week interest_over_time for ticker search variants
          2. Normalize across variants using multiquery_normalize
          3. Compute 4-week and 52-week rolling averages
          4. Momentum ratio = 4w_avg / 52w_avg (>1 = bullish interest)
          5. Classify direction: bullish (>1.2), bearish (<0.8), neutral
        """
        cache_key = self._cache.make_key("momentum", ticker, timeframe, geo)
        if use_cache:
            cached = self._cache.get_signal(cache_key)
            if cached:
                return _momentum_from_dict(cached)

        variants = TICKER_SEARCH_VARIANTS.get(
            ticker.upper(), [f"{ticker} stock", ticker]
        )
        primary_kw = variants[0]

        df = self._adapter.interest_over_time(
            [primary_kw], timeframe=timeframe, geo=geo, use_cache=use_cache
        )

        if df.empty or primary_kw not in df.columns:
            return _empty_momentum(ticker)

        series = df[primary_kw].astype(float)
        latest = float(series.iloc[-1])
        avg_4w = float(series.tail(4).mean())
        avg_52w = float(series.mean())
        momentum_ratio = avg_4w / avg_52w if avg_52w > 0 else 1.0
        acceleration = float(series.diff().tail(4).mean()) if len(series) >= 5 else 0.0

        if momentum_ratio > 1 + _MOMENTUM_ACCEL_THRESHOLD:
            direction = "bullish"
        elif momentum_ratio < 1 + _MOMENTUM_DECEL_THRESHOLD:
            direction = "bearish"
        else:
            direction = "neutral"

        # Multi-variant normalization
        if len(variants) > 1:
            df_multi = self._adapter.interest_over_time(
                variants[:5], timeframe=timeframe, geo=geo, use_cache=use_cache
            )
            variant_avgs: dict[str, float] = {}
            if not df_multi.empty:
                for v in variants[:5]:
                    if v in df_multi.columns:
                        avg = float(df_multi[v].tail(4).mean())
                        total_avg = float(df_multi[v].mean())
                        variant_avgs[v] = round(avg / total_avg, 3) if total_avg > 0 else 1.0
        else:
            variant_avgs = {primary_kw: round(momentum_ratio, 3)}

        interp = _interpret_momentum(ticker, direction, momentum_ratio, avg_52w)
        sig = TrendsMomentumSignal(
            ticker=ticker,
            as_of=date.today(),
            current_interest=round(latest, 2),
            avg_4w=round(avg_4w, 2),
            avg_52w=round(avg_52w, 2),
            momentum_ratio_4w=round(momentum_ratio, 4),
            trend_direction=direction,
            acceleration=round(acceleration, 3),
            search_variants_normalized=variant_avgs,
            interpretation=interp,
        )

        if use_cache:
            self._cache.set_signal(cache_key, _momentum_to_dict(sig))
        return sig

    # ── 2. Surge Detection ────────────────────────────────────────────────────

    def detect_surge(
        self,
        ticker: str,
        keyword: Optional[str] = None,
        timeframe: str = TF_12M,
        geo: str = "",
        use_cache: bool = True,
    ) -> SurgeAlert:
        """
        Detect search interest spike using z-score methodology.

        Z-score = (current_week - 52w_mean) / 52w_std
        Surge threshold: z-score > 2.0 (configurable via _SURGE_ZSCORE_THRESHOLD)

        Probable trigger inference:
          - High z-score + earnings week → earnings surprise search spike
          - High z-score + news correlation → news-driven spike
          - Sustained above baseline → organic interest growth
        """
        kw = keyword or TICKER_SEARCH_VARIANTS.get(
            ticker.upper(), [f"{ticker} stock"]
        )[0]
        cache_key = self._cache.make_key("surge", ticker, kw, timeframe)
        if use_cache:
            cached = self._cache.get_signal(cache_key)
            if cached:
                return _surge_from_dict(cached)

        df = self._adapter.interest_over_time(
            [kw], timeframe=timeframe, geo=geo, use_cache=use_cache
        )

        if df.empty or kw not in df.columns:
            return _empty_surge(ticker, kw)

        series = df[kw].astype(float)
        baseline = series.iloc[:-1]   # everything except the last week
        mean = float(baseline.mean())
        std = float(baseline.std())
        latest = float(series.iloc[-1])

        z = (latest - mean) / std if std > 0 else 0.0
        is_surge = z > _SURGE_ZSCORE_THRESHOLD
        above_pct = ((latest - mean) / mean * 100) if mean > 0 else 0.0

        trigger = _infer_surge_trigger(z, latest, mean, ticker)
        severity = _classify_surge_severity(z)

        alert = SurgeAlert(
            ticker=ticker,
            keyword=kw,
            as_of=date.today(),
            current_value=round(latest, 2),
            baseline_mean=round(mean, 2),
            baseline_std=round(std, 2),
            z_score=round(z, 3),
            is_surge=is_surge,
            surge_pct_above_baseline=round(above_pct, 1),
            probable_trigger=trigger,
            severity=severity,
        )

        self._cache.record_surge(ticker, kw, z, is_surge)
        if use_cache:
            self._cache.set_signal(cache_key, _surge_to_dict(alert))
        return alert

    # ── 3. Geographic Heatmap ────────────────────────────────────────────────

    def geo_heatmap(
        self,
        ticker: str,
        keyword: Optional[str] = None,
        geo_scope: str = "US",
        use_cache: bool = True,
    ) -> GeoHeatmapSignal:
        """
        Fetch state/country breakdown of search interest for a ticker keyword.

        Geographic concentration analysis via Herfindahl-Hirschman Index:
          - HHI near 0 = evenly distributed worldwide interest
          - HHI near 1 = highly concentrated in one region
        """
        kw = keyword or TICKER_SEARCH_VARIANTS.get(
            ticker.upper(), [f"{ticker} stock"]
        )[0]
        cache_key = self._cache.make_key("geo", ticker, kw, geo_scope)
        if use_cache:
            cached = self._cache.get_signal(cache_key)
            if cached:
                return _geo_from_dict(cached)

        # US state breakdown
        df_us = self._adapter.interest_by_region(
            [kw], geo=geo_scope, resolution="REGION", use_cache=use_cache
        )
        _jitter_sleep()

        # Country breakdown
        df_world = self._adapter.interest_by_region(
            [kw], geo="", resolution="COUNTRY", use_cache=use_cache
        )

        us_state_interest: dict[str, float] = {}
        if not df_us.empty and kw in df_us.columns:
            for idx, row in df_us.iterrows():
                if isinstance(idx, str) and "_" in idx:
                    us_state_interest[idx] = float(row[kw])
                elif isinstance(idx, str):
                    us_state_interest[idx] = float(row[kw])

        country_interest: dict[str, float] = {}
        if not df_world.empty and kw in df_world.columns:
            for idx, row in df_world.iterrows():
                country_interest[str(idx)] = float(row[kw])

        # Top regions
        top_countries = sorted(country_interest.items(), key=lambda x: x[1], reverse=True)[:10]
        top_us = sorted(us_state_interest.items(), key=lambda x: x[1], reverse=True)[:10]

        # HHI concentration
        vals = list(us_state_interest.values()) or list(country_interest.values())
        total = sum(vals) or 1
        shares = [v / total for v in vals]
        hhi = float(sum(s ** 2 for s in shares))

        primary = top_countries[0][0] if top_countries else "Unknown"
        interp = _interpret_geo(ticker, hhi, top_countries, top_us)

        sig = GeoHeatmapSignal(
            ticker=ticker,
            keyword=kw,
            as_of=date.today(),
            country_interest=country_interest,
            us_state_interest=us_state_interest,
            top_countries=top_countries,
            top_us_states=top_us,
            geographic_concentration=round(hhi, 4),
            primary_market=primary,
            interpretation=interp,
        )

        if use_cache:
            self._cache.set_signal(cache_key, _geo_to_dict(sig))
        return sig

    # ── 4. Macro Sentiment ────────────────────────────────────────────────────

    def macro_sentiment(
        self,
        timeframe: str = TF_3M,
        geo: str = "US",
        use_cache: bool = True,
    ) -> MacroSentimentSignal:
        """
        Compute macro sentiment from category-level Google Trends.

        Fetches search interest for: recession, inflation, market crash,
        bull market, crypto sentiment, Fed rate anxiety, unemployment fear.
        """
        cache_key = self._cache.make_key("macro_sentiment", timeframe, geo)
        if use_cache:
            cached = self._cache.get_signal(cache_key)
            if cached:
                return _macro_sentiment_from_dict(cached)

        category_scores: dict[str, float] = {}
        for category, keywords in MACRO_SENTIMENT_KEYWORDS.items():
            # Fetch primary keyword for each category
            primary_kw = keywords[0]
            df = self._adapter.interest_over_time(
                [primary_kw], timeframe=timeframe, geo=geo, use_cache=use_cache
            )
            if not df.empty and primary_kw in df.columns:
                category_scores[category] = float(df[primary_kw].tail(4).mean())
            else:
                category_scores[category] = 0.0
            _jitter_sleep()

        # Compute composite indices
        recession_s = category_scores.get("recession", 0)
        inflation_s = category_scores.get("inflation", 0)
        crash_s = category_scores.get("market_crash", 0)
        bull_s = category_scores.get("bull_market", 0)
        crypto_bull = category_scores.get("crypto_bull", 0)
        crypto_bear = category_scores.get("crypto_bear", 0)
        fed_s = category_scores.get("fed_rate", 0)
        unemp_s = category_scores.get("unemployment", 0)

        crypto_net = crypto_bull - crypto_bear

        fear_index = np.mean([recession_s, crash_s, unemp_s, fed_s])
        greed_index = np.mean([bull_s, max(0, crypto_net)])

        if fear_index > greed_index * 1.3:
            regime = "fear"
        elif greed_index > fear_index * 1.3:
            regime = "greed"
        else:
            regime = "neutral"

        interp = _interpret_macro_sentiment(regime, fear_index, greed_index,
                                            recession_s, inflation_s)

        sig = MacroSentimentSignal(
            as_of=date.today(),
            recession_score=round(recession_s, 2),
            inflation_score=round(inflation_s, 2),
            market_crash_fear=round(crash_s, 2),
            bull_market_optimism=round(bull_s, 2),
            crypto_sentiment=round(crypto_net, 2),
            fed_rate_anxiety=round(fed_s, 2),
            unemployment_fear=round(unemp_s, 2),
            composite_fear_index=round(float(fear_index), 2),
            composite_greed_index=round(float(greed_index), 2),
            regime=regime,
            interpretation=interp,
        )

        if use_cache:
            self._cache.set_signal(cache_key, _macro_to_dict(sig))
        return sig

    # ── 5. Fear / Greed Proxy ────────────────────────────────────────────────

    def fear_greed_proxy(
        self,
        timeframe: str = TF_3M,
        geo: str = "US",
        use_cache: bool = True,
    ) -> FearGreedSignal:
        """
        Compute fear/greed market sentiment from search trends.

        Fear keywords: "sell stocks", "stock market crash", "bear market", etc.
        Greed keywords: "buy stocks", "best stocks to buy", "bull market", etc.

        Net signal = avg(greed keywords) - avg(fear keywords)
        Z-score vs 1-year baseline indicates extremes.
        """
        cache_key = self._cache.make_key("fear_greed", timeframe, geo)
        if use_cache:
            cached = self._cache.get_signal(cache_key)
            if cached:
                return _fear_greed_from_dict(cached)

        # Fetch 1-year data for baseline
        fear_dfs: list[pd.Series] = []
        greed_dfs: list[pd.Series] = []

        for kw in FEAR_KEYWORDS[:3]:
            df = self._adapter.interest_over_time(
                [kw], timeframe=TF_12M, geo=geo, use_cache=use_cache
            )
            if not df.empty and kw in df.columns:
                fear_dfs.append(df[kw].astype(float))
            _jitter_sleep()

        for kw in GREED_KEYWORDS[:3]:
            df = self._adapter.interest_over_time(
                [kw], timeframe=TF_12M, geo=geo, use_cache=use_cache
            )
            if not df.empty and kw in df.columns:
                greed_dfs.append(df[kw].astype(float))
            _jitter_sleep()

        if not fear_dfs and not greed_dfs:
            return _empty_fear_greed()

        # Build composite series
        fear_series = pd.concat(fear_dfs, axis=1).mean(axis=1) if fear_dfs else pd.Series(dtype=float)
        greed_series = pd.concat(greed_dfs, axis=1).mean(axis=1) if greed_dfs else pd.Series(dtype=float)

        # Align
        combined = pd.concat(
            [fear_series.rename("fear"), greed_series.rename("greed")], axis=1
        ).fillna(0)

        combined["net"] = combined["greed"] - combined["fear"]

        fear_score_now = float(combined["fear"].tail(4).mean())
        greed_score_now = float(combined["greed"].tail(4).mean())
        net_now = greed_score_now - fear_score_now

        # Z-score vs full 1-year
        net_series = combined["net"]
        mu = float(net_series.mean())
        sigma = float(net_series.std())
        z = (net_now - mu) / sigma if sigma > 0 else 0.0

        # Classification
        ratio = (fear_score_now / greed_score_now) if greed_score_now > 0 else 2.0
        if z < -2.0 or ratio > 2.5:
            signal = "extreme_fear"
        elif z < -1.0 or ratio > 1.5:
            signal = "fear"
        elif z > 2.0 or ratio < 0.5:
            signal = "extreme_greed"
        elif z > 1.0 or ratio < 0.8:
            signal = "greed"
        else:
            signal = "neutral"

        # Weekly net signal for chart
        weekly = {
            k.isoformat(): round(float(v), 2)
            for k, v in combined["net"].tail(12).items()
        }

        interp = _interpret_fear_greed(signal, fear_score_now, greed_score_now, z)

        sig = FearGreedSignal(
            as_of=date.today(),
            fear_score=round(fear_score_now, 2),
            greed_score=round(greed_score_now, 2),
            fear_greed_ratio=round(ratio, 3),
            net_signal=round(net_now, 2),
            signal=signal,
            z_score_vs_1y=round(z, 3),
            weekly_series=weekly,
            interpretation=interp,
        )

        if use_cache:
            self._cache.set_signal(cache_key, _fear_greed_to_dict(sig))
        return sig

    # ── 6. Competitor Share ────────────────────────────────────────────────────

    def competitor_share(
        self,
        ticker: str,
        peers: Optional[list[str]] = None,
        timeframe: str = TF_3M,
        geo: str = "",
        use_cache: bool = True,
    ) -> CompetitorShareSignal:
        """
        Track relative search share of a ticker's brand vs competitors.

        Relative share = ticker_interest / sum(all_keywords_interest) * 100

        Computes share trend: is the ticker gaining or losing mindshare?
        """
        cache_key = self._cache.make_key("comp_share", ticker, peers, timeframe, geo)
        if use_cache:
            cached = self._cache.get_signal(cache_key)
            if cached:
                return _comp_share_from_dict(cached)

        keywords = peers or SECTOR_PEERS.get(ticker.upper(), [ticker])
        keywords = keywords[:5]  # pytrends max

        df = self._adapter.interest_over_time(
            keywords, timeframe=timeframe, geo=geo, use_cache=use_cache
        )

        if df.empty:
            return _empty_comp_share(ticker, keywords)

        # Relative share for each keyword
        row_totals = df.sum(axis=1).replace(0, np.nan)
        df_share = df.div(row_totals, axis=0) * 100

        # Identify ticker's primary keyword
        ticker_kw = keywords[0]

        # Average share per keyword across period
        avg_share: dict[str, float] = {
            kw: round(float(df_share[kw].mean()), 2)
            for kw in keywords
            if kw in df_share.columns
        }

        ticker_share = avg_share.get(ticker_kw, 0.0)

        # Month-over-month share change
        n = len(df_share)
        half = n // 2
        share_prior = float(df_share[ticker_kw].iloc[:half].mean()) if half > 0 else ticker_share
        share_recent = float(df_share[ticker_kw].iloc[half:].mean()) if half > 0 else ticker_share
        share_change = round(share_recent - share_prior, 2)

        if share_change > 2.0:
            trend = "gaining"
        elif share_change < -2.0:
            trend = "losing"
        else:
            trend = "stable"

        dominant = max(avg_share, key=avg_share.get) if avg_share else ticker_kw
        interp = _interpret_comp_share(ticker, ticker_kw, ticker_share, share_change, trend, dominant)

        sig = CompetitorShareSignal(
            ticker=ticker,
            as_of=date.today(),
            keywords=keywords,
            relative_share=avg_share,
            dominant_brand=dominant,
            ticker_share_pct=ticker_share,
            share_vs_prior_month=share_change,
            trend=trend,
            interpretation=interp,
        )

        if use_cache:
            self._cache.set_signal(cache_key, _comp_share_to_dict(sig))
        return sig

    # ── 7. Seasonal Adjustment ────────────────────────────────────────────────

    def seasonal_adjust(
        self,
        ticker: str,
        keyword: Optional[str] = None,
        timeframe: str = TF_5Y,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Remove seasonal (holiday) effects from Google Trends time series.

        Method: Rolling 52-week deviation normalization.
          seasonally_adjusted[t] = raw[t] - (rolling_52w_mean[t] - global_mean)

        Returns DataFrame with columns: raw, seasonal_component, adjusted.
        """
        kw = keyword or TICKER_SEARCH_VARIANTS.get(
            ticker.upper(), [f"{ticker} stock"]
        )[0]
        cache_key = self._cache.make_key("seasonal", ticker, kw, timeframe)

        if use_cache:
            cached = self._cache.get_raw(cache_key)
            if cached:
                try:
                    return pd.read_json(cached)
                except Exception:
                    pass

        df = self._adapter.interest_over_time(
            [kw], timeframe=timeframe, use_cache=use_cache
        )

        if df.empty or kw not in df.columns:
            return pd.DataFrame(columns=["raw", "seasonal_component", "adjusted"])

        raw = df[kw].astype(float)
        global_mean = raw.mean()

        # Rolling 52-week mean (centered where possible, forward fill at edges)
        roll52 = raw.rolling(window=52, min_periods=4, center=True).mean()
        seasonal = roll52 - global_mean
        adjusted = raw - seasonal

        result = pd.DataFrame({
            "raw": raw,
            "seasonal_component": seasonal,
            "adjusted": adjusted,
        })

        if use_cache:
            self._cache.set_raw(cache_key, result.to_json())

        return result

    # ── 8. Granger Causality ────────────────────────────────────────────────

    def granger_causality(
        self,
        ticker: str,
        prices: pd.Series,
        keyword: Optional[str] = None,
        max_lag: int = _GRANGER_MAX_LAG,
        use_cache: bool = True,
    ) -> GrangerResult:
        """
        Test Granger causality: do search trends predict stock returns?

        Methodology:
          1. Fetch weekly trends + compute weekly price returns
          2. Align series on weekly frequency
          3. For each lag 1..max_lag: fit restricted and unrestricted OLS models
          4. F-test: unrestricted (with trends) vs restricted (returns only)
          5. If F-stat significant (p<0.05): trends Granger-cause returns
          6. Also test reverse: returns → trends (bidirectional check)

        Implementation uses scipy.stats for OLS + F-test (avoids statsmodels dependency).
        """
        kw = keyword or TICKER_SEARCH_VARIANTS.get(
            ticker.upper(), [f"{ticker} stock"]
        )[0]
        cache_key = self._cache.make_key("granger", ticker, kw, max_lag)

        if use_cache:
            cached = self._cache.get_granger(cache_key)
            if cached:
                return _granger_from_dict(cached)

        df_trends = self._adapter.interest_over_time(
            [kw], timeframe=TF_5Y, use_cache=use_cache
        )

        if df_trends.empty or kw not in df_trends.columns or prices.empty:
            return _empty_granger(ticker, kw, max_lag)

        trends = df_trends[kw].astype(float)
        trends.index = pd.to_datetime(trends.index)

        prices_s = prices.copy()
        prices_s.index = pd.to_datetime(prices_s.index)
        weekly_returns = prices_s.resample("W").last().pct_change().dropna()

        # Align
        combined = pd.concat(
            [trends.rename("trends"), weekly_returns.rename("ret")], axis=1
        ).dropna()

        if len(combined) < max_lag * 4:
            return _empty_granger(ticker, kw, max_lag)

        best_f = 0.0
        best_p = 1.0
        best_lag = 1
        best_r2 = 0.0

        for lag in range(1, max_lag + 1):
            result = _granger_f_test(combined["ret"].values, combined["trends"].values, lag)
            if result["f_stat"] > best_f:
                best_f = result["f_stat"]
                best_p = result["p_value"]
                best_lag = lag
                best_r2 = result["r_squared"]

        # Test reverse direction: returns → trends
        reverse_result = _granger_f_test(
            combined["trends"].values, combined["ret"].values, best_lag
        )

        is_sig = best_p < 0.05
        is_reverse_sig = reverse_result["p_value"] < 0.05

        if is_sig and is_reverse_sig:
            direction = "bidirectional"
        elif is_sig:
            direction = "trends→returns"
        elif is_reverse_sig:
            direction = "returns→trends"
        else:
            direction = "none"

        interp = _interpret_granger(ticker, kw, direction, best_lag, best_p, best_r2)

        result_sig = GrangerResult(
            ticker=ticker,
            keyword=kw,
            lag_weeks=max_lag,
            f_statistic=round(best_f, 4),
            p_value=round(best_p, 4),
            is_significant=is_sig,
            causality_direction=direction,
            optimal_lag=best_lag,
            r_squared=round(best_r2, 4),
            interpretation=interp,
        )

        if use_cache:
            self._cache.set_granger(cache_key, _granger_to_dict(result_sig))
        return result_sig

    # ── 9. Topic Clustering ──────────────────────────────────────────────────

    def topic_clusters(
        self,
        ticker: str,
        keyword: Optional[str] = None,
        use_cache: bool = True,
    ) -> TopicClusterSignal:
        """
        Cluster related search queries into thematic groups.

        Method:
          1. Fetch related_queries (top + rising) for primary keyword
          2. Classify each query into a theme bucket using keyword matching
          3. Score themes by average query interest weight
          4. Identify rising queries (rapid growth signals)

        Themes: product, earnings, competition, regulation, macro, sentiment
        """
        kw = keyword or TICKER_SEARCH_VARIANTS.get(
            ticker.upper(), [f"{ticker} stock"]
        )[0]
        cache_key = self._cache.make_key("clusters", ticker, kw)

        if use_cache:
            cached = self._cache.get_signal(cache_key)
            if cached:
                return _cluster_from_dict(cached)

        rq = self._adapter.related_queries(kw, use_cache=use_cache)
        _jitter_sleep()
        rt = self._adapter.related_topics(kw, use_cache=use_cache)

        top_queries: list[str] = []
        rising_queries: list[str] = []

        if rq and kw in rq:
            sub = rq[kw]
            if sub.get("top") is not None and isinstance(sub["top"], pd.DataFrame):
                if "query" in sub["top"].columns:
                    top_queries = sub["top"]["query"].tolist()[:15]
            if sub.get("rising") is not None and isinstance(sub["rising"], pd.DataFrame):
                if "query" in sub["rising"].columns:
                    rising_queries = sub["rising"]["query"].tolist()[:10]

        # Theme classification
        theme_defs: dict[str, list[str]] = {
            "earnings_results": [
                "earnings", "revenue", "profit", "eps", "quarterly results",
                "beat estimates", "guidance", "fiscal year",
            ],
            "product_launches": [
                "new product", "launch", "release", "update", "feature",
                "announced", "unveil", "new model",
            ],
            "competition": [
                "vs", "versus", "compare", "competitor", "alternative",
                "better than", "switch from",
            ],
            "regulatory": [
                "regulation", "antitrust", "lawsuit", "SEC", "FTC", "investigation",
                "fine", "compliance", "ban",
            ],
            "macro_market": [
                "interest rate", "inflation", "recession", "market crash",
                "S&P 500", "Nasdaq", "index", "ETF",
            ],
            "sentiment_investment": [
                "buy", "sell", "price target", "analyst", "upgrade", "downgrade",
                "short", "bull", "bear", "options", "calls", "puts",
            ],
        }

        clusters: dict[str, list[str]] = {theme: [] for theme in theme_defs}
        for q in top_queries + rising_queries:
            q_lower = q.lower()
            matched = False
            for theme, keywords in theme_defs.items():
                if any(kw_t in q_lower for kw_t in keywords):
                    clusters[theme].append(q)
                    matched = True
                    break
            if not matched:
                clusters.setdefault("other", []).append(q)

        # Score themes by count (proxy for interest)
        theme_scores = {
            theme: float(len(qs)) / max(len(top_queries + rising_queries), 1) * 100
            for theme, qs in clusters.items()
            if qs
        }

        dominant = max(theme_scores, key=theme_scores.get) if theme_scores else "other"
        interp = _interpret_clusters(ticker, dominant, rising_queries, theme_scores)

        sig = TopicClusterSignal(
            ticker=ticker,
            as_of=date.today(),
            clusters={k: v for k, v in clusters.items() if v},
            dominant_theme=dominant,
            theme_scores=theme_scores,
            rising_queries=rising_queries,
            top_queries=top_queries[:10],
            interpretation=interp,
        )

        if use_cache:
            self._cache.set_signal(cache_key, _cluster_to_dict(sig))
        return sig

    # ── 10. Sector Rotation ──────────────────────────────────────────────────

    def sector_rotation_signals(
        self,
        timeframe: str = TF_3M,
        geo: str = "US",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Detect sector rotation via relative search interest.

        Returns dict mapping sector → normalized search interest score.
        Higher score = more investor attention this period.
        """
        cache_key = self._cache.make_key("sector_rotation", timeframe, geo)
        if use_cache:
            cached = self._cache.get_signal(cache_key)
            if cached:
                return cached

        sector_scores: dict[str, float] = {}
        keywords = list(SECTOR_ROTATION_KEYWORDS.values())[:5]
        sector_names = list(SECTOR_ROTATION_KEYWORDS.keys())[:5]

        df = self._adapter.interest_over_time(
            keywords, timeframe=timeframe, geo=geo, use_cache=use_cache
        )

        if not df.empty:
            for sector, kw in zip(sector_names, keywords):
                if kw in df.columns:
                    sector_scores[sector] = round(float(df[kw].tail(4).mean()), 2)

        result = {
            "as_of": date.today().isoformat(),
            "sector_interest": sector_scores,
            "top_sector": max(sector_scores, key=sector_scores.get) if sector_scores else "unknown",
            "bottom_sector": min(sector_scores, key=sector_scores.get) if sector_scores else "unknown",
        }

        if use_cache and sector_scores:
            self._cache.set_signal(cache_key, result)
        return result


# ---------------------------------------------------------------------------
# Statistical Helpers
# ---------------------------------------------------------------------------


def _granger_f_test(y: np.ndarray, x: np.ndarray, lag: int) -> dict:
    """
    Compute Granger causality F-test using OLS.

    Unrestricted: y_t = a0 + sum(a_i * y_{t-i}) + sum(b_i * x_{t-i}) + e
    Restricted:   y_t = a0 + sum(a_i * y_{t-i}) + e

    F = ((RSS_r - RSS_u) / lag) / (RSS_u / (n - 2*lag - 1))
    """
    n = len(y)
    if n <= lag * 3:
        return {"f_stat": 0.0, "p_value": 1.0, "r_squared": 0.0}

    # Build lagged matrices
    def _build_lags(arr: np.ndarray, n_lag: int, start: int) -> np.ndarray:
        rows = []
        for i in range(start, n):
            row = [arr[i - j - 1] for j in range(n_lag)]
            rows.append(row)
        return np.array(rows)

    start = lag
    y_dep = y[start:]
    X_y_lags = _build_lags(y, lag, start)
    X_x_lags = _build_lags(x, lag, start)

    n_obs = len(y_dep)
    ones = np.ones((n_obs, 1))

    # Restricted model: y_lags only
    X_r = np.hstack([ones, X_y_lags])
    # Unrestricted model: y_lags + x_lags
    X_u = np.hstack([ones, X_y_lags, X_x_lags])

    try:
        beta_r, rss_r, _, _ = np.linalg.lstsq(X_r, y_dep, rcond=None)
        beta_u, rss_u, _, _ = np.linalg.lstsq(X_u, y_dep, rcond=None)

        y_hat_r = X_r @ beta_r
        y_hat_u = X_u @ beta_u
        rss_r_val = float(np.sum((y_dep - y_hat_r) ** 2))
        rss_u_val = float(np.sum((y_dep - y_hat_u) ** 2))

        df1 = lag
        df2 = n_obs - 2 * lag - 1
        if df2 <= 0 or rss_u_val == 0:
            return {"f_stat": 0.0, "p_value": 1.0, "r_squared": 0.0}

        f_stat = ((rss_r_val - rss_u_val) / df1) / (rss_u_val / df2)
        p_val = float(1 - stats.f.cdf(f_stat, df1, df2))

        tss = float(np.sum((y_dep - y_dep.mean()) ** 2))
        r2 = 1 - rss_u_val / tss if tss > 0 else 0.0

        return {
            "f_stat": float(f_stat),
            "p_value": float(p_val),
            "r_squared": float(r2),
        }
    except np.linalg.LinAlgError:
        return {"f_stat": 0.0, "p_value": 1.0, "r_squared": 0.0}


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------


def _interpret_momentum(ticker: str, direction: str, ratio: float, avg52w: float) -> str:
    if direction == "bullish":
        return (
            f"{ticker} search interest BULLISH: 4-week avg is {(ratio-1)*100:.0f}% above "
            f"52-week baseline ({avg52w:.0f}/100). Leading indicator: retail investor attention rising."
        )
    elif direction == "bearish":
        return (
            f"{ticker} search interest BEARISH: 4-week avg is {abs(ratio-1)*100:.0f}% below "
            f"52-week baseline. Declining retail attention — often precedes institutional selling."
        )
    return (
        f"{ticker} search interest NEUTRAL: 4-week avg tracks 52-week baseline ({avg52w:.0f}/100). "
        "No directional momentum signal."
    )


def _infer_surge_trigger(z: float, latest: float, mean: float, ticker: str) -> str:
    if z > 4.0:
        return f"Extreme spike — likely triggered by major news event, earnings surprise, or viral mention for {ticker}"
    elif z > 2.5:
        return f"Significant spike for {ticker} — possible earnings report, analyst upgrade/downgrade, or breaking news"
    elif z > _SURGE_ZSCORE_THRESHOLD:
        return f"Moderate spike for {ticker} — elevated retail interest, possibly product launch or index inclusion"
    return "Normal variance — no specific trigger inferred"


def _classify_surge_severity(z: float) -> str:
    if z > 4.0:
        return "extreme"
    elif z > 2.5:
        return "high"
    elif z > _SURGE_ZSCORE_THRESHOLD:
        return "moderate"
    return "low"


def _interpret_geo(ticker: str, hhi: float, top_countries: list, top_us: list) -> str:
    primary = top_countries[0][0] if top_countries else "Unknown"
    conc_str = "highly concentrated" if hhi > 0.3 else "broadly distributed"
    return (
        f"{ticker} search interest is {conc_str} (HHI={hhi:.3f}). "
        f"Primary market: {primary}. "
        f"Top US state: {top_us[0][0] if top_us else 'N/A'}. "
        "Geographic spread may proxy for revenue diversification."
    )


def _interpret_macro_sentiment(regime: str, fear: float, greed: float,
                                 recession: float, inflation: float) -> str:
    return (
        f"Macro sentiment regime: {regime.upper()}. "
        f"Fear index: {fear:.1f}/100 | Greed index: {greed:.1f}/100. "
        f"Recession search interest: {recession:.0f}/100 | Inflation: {inflation:.0f}/100. "
        + (
            "Elevated fear signals → risk-off positioning, defensive sectors preferred."
            if regime == "fear"
            else "Elevated greed signals → risk-on positioning, growth assets favored."
            if regime == "greed"
            else "Balanced sentiment — no strong directional bias."
        )
    )


def _interpret_fear_greed(signal: str, fear: float, greed: float, z: float) -> str:
    return (
        f"Market Fear/Greed: {signal.replace('_', ' ').upper()}. "
        f"Fear score: {fear:.1f}/100 | Greed score: {greed:.1f}/100 | "
        f"Z-score vs 1yr: {z:+.2f}. "
        + {
            "extreme_fear": "Contrarian BUY signal — extreme fear historically marks bottoms.",
            "fear": "Cautious environment — elevated fear often precedes continued volatility.",
            "neutral": "Mixed signals — no strong contrarian positioning indicated.",
            "greed": "Elevated greed — moderate caution warranted, sentiment can turn quickly.",
            "extreme_greed": "Contrarian SELL signal — extreme greed historically marks tops.",
        }.get(signal, "")
    )


def _interpret_comp_share(ticker: str, kw: str, share: float, change: float,
                            trend: str, dominant: str) -> str:
    return (
        f"{ticker} ({kw}): {share:.1f}% relative search share. "
        f"Trend: {trend} ({change:+.1f}pp vs prior period). "
        f"Dominant brand: {dominant}. "
        + (
            f"{ticker} is gaining mindshare vs peers — positive brand momentum signal."
            if trend == "gaining"
            else f"{ticker} losing search share to {dominant} — potential competitive pressure."
            if trend == "losing"
            else "Search share stable — no significant mindshare shift detected."
        )
    )


def _interpret_granger(ticker: str, kw: str, direction: str, lag: int,
                         p_val: float, r2: float) -> str:
    if direction == "none":
        return (
            f"No Granger causality detected between '{kw}' trends and {ticker} returns "
            f"(best F-test p={p_val:.3f}, R²={r2:.3f}). Search trends are not a reliable "
            f"leading indicator for {ticker} at lags 1-{_GRANGER_MAX_LAG} weeks."
        )
    elif direction == "trends→returns":
        return (
            f"'{kw}' search trends GRANGER-CAUSE {ticker} returns at lag {lag}w "
            f"(p={p_val:.3f}, R²={r2:.3f}). Actionable: trends 'predict' returns. "
            "Use as leading indicator for position timing."
        )
    elif direction == "returns→trends":
        return (
            f"{ticker} returns GRANGER-CAUSE '{kw}' search trends at lag {lag}w "
            f"(p={p_val:.3f}). Price moves precede search spikes — confirmation signal only."
        )
    return (
        f"Bidirectional Granger causality between '{kw}' and {ticker} returns (lag {lag}w). "
        f"Reflexive relationship — both can serve as leading indicators."
    )


def _interpret_clusters(ticker: str, dominant: str, rising: list, scores: dict) -> str:
    rising_str = ", ".join(rising[:3]) if rising else "none detected"
    return (
        f"{ticker} dominant search theme: {dominant.replace('_', ' ')}. "
        f"Rising queries: {rising_str}. "
        f"Theme distribution: "
        + ", ".join(f"{k}={v:.0f}%" for k, v in list(scores.items())[:4])
    )


# ---------------------------------------------------------------------------
# Serialization / deserialization helpers (for SQLite cache round-trips)
# ---------------------------------------------------------------------------


def _momentum_to_dict(s: TrendsMomentumSignal) -> dict:
    return {
        "ticker": s.ticker, "as_of": s.as_of.isoformat(),
        "current_interest": s.current_interest, "avg_4w": s.avg_4w,
        "avg_52w": s.avg_52w, "momentum_ratio_4w": s.momentum_ratio_4w,
        "trend_direction": s.trend_direction, "acceleration": s.acceleration,
        "search_variants_normalized": s.search_variants_normalized,
        "interpretation": s.interpretation,
    }


def _momentum_from_dict(d: dict) -> TrendsMomentumSignal:
    return TrendsMomentumSignal(
        ticker=d["ticker"], as_of=date.fromisoformat(d["as_of"]),
        current_interest=d["current_interest"], avg_4w=d["avg_4w"],
        avg_52w=d["avg_52w"], momentum_ratio_4w=d["momentum_ratio_4w"],
        trend_direction=d["trend_direction"], acceleration=d["acceleration"],
        search_variants_normalized=d["search_variants_normalized"],
        interpretation=d["interpretation"],
    )


def _empty_momentum(ticker: str) -> TrendsMomentumSignal:
    return TrendsMomentumSignal(
        ticker=ticker, as_of=date.today(), current_interest=0.0,
        avg_4w=0.0, avg_52w=0.0, momentum_ratio_4w=1.0,
        trend_direction="neutral", acceleration=0.0,
        search_variants_normalized={},
        interpretation=f"No trends data available for {ticker}",
    )


def _surge_to_dict(s: SurgeAlert) -> dict:
    return {
        "ticker": s.ticker, "keyword": s.keyword, "as_of": s.as_of.isoformat(),
        "current_value": s.current_value, "baseline_mean": s.baseline_mean,
        "baseline_std": s.baseline_std, "z_score": s.z_score,
        "is_surge": s.is_surge, "surge_pct_above_baseline": s.surge_pct_above_baseline,
        "probable_trigger": s.probable_trigger, "severity": s.severity,
    }


def _surge_from_dict(d: dict) -> SurgeAlert:
    return SurgeAlert(
        ticker=d["ticker"], keyword=d["keyword"],
        as_of=date.fromisoformat(d["as_of"]),
        current_value=d["current_value"], baseline_mean=d["baseline_mean"],
        baseline_std=d["baseline_std"], z_score=d["z_score"],
        is_surge=d["is_surge"],
        surge_pct_above_baseline=d["surge_pct_above_baseline"],
        probable_trigger=d["probable_trigger"], severity=d["severity"],
    )


def _empty_surge(ticker: str, keyword: str) -> SurgeAlert:
    return SurgeAlert(
        ticker=ticker, keyword=keyword, as_of=date.today(),
        current_value=0.0, baseline_mean=0.0, baseline_std=0.0,
        z_score=0.0, is_surge=False, surge_pct_above_baseline=0.0,
        probable_trigger="No data", severity="low",
    )


def _geo_to_dict(s: GeoHeatmapSignal) -> dict:
    return {
        "ticker": s.ticker, "keyword": s.keyword, "as_of": s.as_of.isoformat(),
        "country_interest": s.country_interest,
        "us_state_interest": s.us_state_interest,
        "top_countries": s.top_countries,
        "top_us_states": s.top_us_states,
        "geographic_concentration": s.geographic_concentration,
        "primary_market": s.primary_market,
        "interpretation": s.interpretation,
    }


def _geo_from_dict(d: dict) -> GeoHeatmapSignal:
    return GeoHeatmapSignal(
        ticker=d["ticker"], keyword=d["keyword"],
        as_of=date.fromisoformat(d["as_of"]),
        country_interest=d["country_interest"],
        us_state_interest=d["us_state_interest"],
        top_countries=[tuple(x) for x in d["top_countries"]],
        top_us_states=[tuple(x) for x in d["top_us_states"]],
        geographic_concentration=d["geographic_concentration"],
        primary_market=d["primary_market"],
        interpretation=d["interpretation"],
    )


def _macro_to_dict(s: MacroSentimentSignal) -> dict:
    return {
        "as_of": s.as_of.isoformat(),
        "recession_score": s.recession_score,
        "inflation_score": s.inflation_score,
        "market_crash_fear": s.market_crash_fear,
        "bull_market_optimism": s.bull_market_optimism,
        "crypto_sentiment": s.crypto_sentiment,
        "fed_rate_anxiety": s.fed_rate_anxiety,
        "unemployment_fear": s.unemployment_fear,
        "composite_fear_index": s.composite_fear_index,
        "composite_greed_index": s.composite_greed_index,
        "regime": s.regime,
        "interpretation": s.interpretation,
    }


def _macro_sentiment_from_dict(d: dict) -> MacroSentimentSignal:
    return MacroSentimentSignal(
        as_of=date.fromisoformat(d["as_of"]),
        recession_score=d["recession_score"],
        inflation_score=d["inflation_score"],
        market_crash_fear=d["market_crash_fear"],
        bull_market_optimism=d["bull_market_optimism"],
        crypto_sentiment=d["crypto_sentiment"],
        fed_rate_anxiety=d["fed_rate_anxiety"],
        unemployment_fear=d["unemployment_fear"],
        composite_fear_index=d["composite_fear_index"],
        composite_greed_index=d["composite_greed_index"],
        regime=d["regime"],
        interpretation=d["interpretation"],
    )


def _fear_greed_to_dict(s: FearGreedSignal) -> dict:
    return {
        "as_of": s.as_of.isoformat(),
        "fear_score": s.fear_score, "greed_score": s.greed_score,
        "fear_greed_ratio": s.fear_greed_ratio,
        "net_signal": s.net_signal, "signal": s.signal,
        "z_score_vs_1y": s.z_score_vs_1y,
        "weekly_series": s.weekly_series,
        "interpretation": s.interpretation,
    }


def _fear_greed_from_dict(d: dict) -> FearGreedSignal:
    return FearGreedSignal(
        as_of=date.fromisoformat(d["as_of"]),
        fear_score=d["fear_score"], greed_score=d["greed_score"],
        fear_greed_ratio=d["fear_greed_ratio"],
        net_signal=d["net_signal"], signal=d["signal"],
        z_score_vs_1y=d["z_score_vs_1y"],
        weekly_series=d["weekly_series"],
        interpretation=d["interpretation"],
    )


def _empty_fear_greed() -> FearGreedSignal:
    return FearGreedSignal(
        as_of=date.today(), fear_score=0.0, greed_score=0.0,
        fear_greed_ratio=1.0, net_signal=0.0, signal="neutral",
        z_score_vs_1y=0.0, weekly_series={},
        interpretation="No trends data available",
    )


def _comp_share_to_dict(s: CompetitorShareSignal) -> dict:
    return {
        "ticker": s.ticker, "as_of": s.as_of.isoformat(),
        "keywords": s.keywords, "relative_share": s.relative_share,
        "dominant_brand": s.dominant_brand,
        "ticker_share_pct": s.ticker_share_pct,
        "share_vs_prior_month": s.share_vs_prior_month,
        "trend": s.trend, "interpretation": s.interpretation,
    }


def _comp_share_from_dict(d: dict) -> CompetitorShareSignal:
    return CompetitorShareSignal(
        ticker=d["ticker"], as_of=date.fromisoformat(d["as_of"]),
        keywords=d["keywords"], relative_share=d["relative_share"],
        dominant_brand=d["dominant_brand"],
        ticker_share_pct=d["ticker_share_pct"],
        share_vs_prior_month=d["share_vs_prior_month"],
        trend=d["trend"], interpretation=d["interpretation"],
    )


def _empty_comp_share(ticker: str, keywords: list[str]) -> CompetitorShareSignal:
    return CompetitorShareSignal(
        ticker=ticker, as_of=date.today(), keywords=keywords,
        relative_share={}, dominant_brand=ticker,
        ticker_share_pct=0.0, share_vs_prior_month=0.0,
        trend="stable", interpretation="No trends data available",
    )


def _granger_to_dict(s: GrangerResult) -> dict:
    return {
        "ticker": s.ticker, "keyword": s.keyword, "lag_weeks": s.lag_weeks,
        "f_statistic": s.f_statistic, "p_value": s.p_value,
        "is_significant": s.is_significant,
        "causality_direction": s.causality_direction,
        "optimal_lag": s.optimal_lag, "r_squared": s.r_squared,
        "interpretation": s.interpretation,
    }


def _granger_from_dict(d: dict) -> GrangerResult:
    return GrangerResult(
        ticker=d["ticker"], keyword=d["keyword"], lag_weeks=d["lag_weeks"],
        f_statistic=d["f_statistic"], p_value=d["p_value"],
        is_significant=d["is_significant"],
        causality_direction=d["causality_direction"],
        optimal_lag=d["optimal_lag"], r_squared=d["r_squared"],
        interpretation=d["interpretation"],
    )


def _empty_granger(ticker: str, keyword: str, max_lag: int) -> GrangerResult:
    return GrangerResult(
        ticker=ticker, keyword=keyword, lag_weeks=max_lag,
        f_statistic=0.0, p_value=1.0, is_significant=False,
        causality_direction="none", optimal_lag=1, r_squared=0.0,
        interpretation="Insufficient data for Granger causality test",
    )


def _cluster_to_dict(s: TopicClusterSignal) -> dict:
    return {
        "ticker": s.ticker, "as_of": s.as_of.isoformat(),
        "clusters": s.clusters, "dominant_theme": s.dominant_theme,
        "theme_scores": s.theme_scores, "rising_queries": s.rising_queries,
        "top_queries": s.top_queries, "interpretation": s.interpretation,
    }


def _cluster_from_dict(d: dict) -> TopicClusterSignal:
    return TopicClusterSignal(
        ticker=d["ticker"], as_of=date.fromisoformat(d["as_of"]),
        clusters=d["clusters"], dominant_theme=d["dominant_theme"],
        theme_scores=d["theme_scores"], rising_queries=d["rising_queries"],
        top_queries=d["top_queries"], interpretation=d["interpretation"],
    )


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

google_trends_v2_router = APIRouter(
    prefix="/trends/v2",
    tags=["Alternative Data — Google Trends v2"],
)

_cache_v2: Optional[TrendsCacheV2] = None
_adapter_v2: Optional[GoogleTrendsAdapterV2] = None
_engine_v2: Optional[TrendsSignalEngineV2] = None


def _get_engine_v2() -> tuple[TrendsCacheV2, GoogleTrendsAdapterV2, TrendsSignalEngineV2]:
    global _cache_v2, _adapter_v2, _engine_v2
    if _cache_v2 is None:
        _cache_v2 = TrendsCacheV2()
    if _adapter_v2 is None:
        _adapter_v2 = GoogleTrendsAdapterV2(cache=_cache_v2)
    if _engine_v2 is None:
        _engine_v2 = TrendsSignalEngineV2(adapter=_adapter_v2, cache=_cache_v2)
    return _cache_v2, _adapter_v2, _engine_v2


@google_trends_v2_router.get(
    "/trends/{ticker}",
    summary="Full Google Trends signal suite for a ticker",
    response_model=dict,
)
def get_trends_full(
    ticker: str,
    geo: str = Query("US", description="Google Trends geo scope (e.g., US, GB, '')"),
    use_cache: bool = Query(True),
) -> dict:
    """
    Return comprehensive Google Trends signals for a ticker:
    momentum, surge detection, geo heatmap, topic clusters, competitor share.
    """
    _, _, engine = _get_engine_v2()
    ticker = ticker.upper()
    try:
        momentum = engine.compute_momentum(ticker, geo=geo, use_cache=use_cache)
        surge = engine.detect_surge(ticker, geo=geo, use_cache=use_cache)
        clusters = engine.topic_clusters(ticker, use_cache=use_cache)

        return {
            "ticker": ticker,
            "as_of": date.today().isoformat(),
            "geo": geo,
            "momentum": _momentum_to_dict(momentum),
            "surge_alert": _surge_to_dict(surge),
            "topic_clusters": _cluster_to_dict(clusters),
        }
    except Exception as exc:
        logger.error("trends/{ticker} error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@google_trends_v2_router.get(
    "/surge-alerts",
    summary="Recent search interest surge alerts",
    response_model=dict,
)
def get_surge_alerts(
    tickers: str = Query("AAPL,MSFT,GOOGL,AMZN,TSLA", description="Comma-separated tickers"),
    geo: str = Query("US"),
    use_cache: bool = Query(True),
) -> dict:
    """
    Detect search interest surges (z-score > 2.0) for a list of tickers.
    Returns only tickers with active surge conditions.
    """
    _, _, engine = _get_engine_v2()
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    alerts: list[dict] = []
    for tick in ticker_list[:10]:
        try:
            surge = engine.detect_surge(tick, geo=geo, use_cache=use_cache)
            if surge.is_surge:
                alerts.append(_surge_to_dict(surge))
            _jitter_sleep()
        except Exception as exc:
            logger.warning("Surge check failed for %s: %s", tick, exc)

    return {
        "as_of": date.today().isoformat(),
        "tickers_checked": ticker_list,
        "surge_count": len(alerts),
        "active_surges": alerts,
    }


@google_trends_v2_router.get(
    "/macro-sentiment",
    summary="Macro market sentiment from Google Trends",
    response_model=dict,
)
def get_macro_sentiment(
    geo: str = Query("US"),
    timeframe: str = Query("today 3-m"),
    use_cache: bool = Query(True),
) -> dict:
    """
    Return macro sentiment signals: recession/inflation fear, bull/bear market
    interest, crypto sentiment, composite fear/greed index.
    """
    _, _, engine = _get_engine_v2()
    try:
        macro = engine.macro_sentiment(timeframe=timeframe, geo=geo, use_cache=use_cache)
        fg = engine.fear_greed_proxy(timeframe=timeframe, geo=geo, use_cache=use_cache)
        return {
            "as_of": date.today().isoformat(),
            "macro_sentiment": _macro_to_dict(macro),
            "fear_greed": _fear_greed_to_dict(fg),
        }
    except Exception as exc:
        logger.error("macro-sentiment error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@google_trends_v2_router.get(
    "/competitor-share/{ticker}",
    summary="Relative search share vs sector peers",
    response_model=dict,
)
def get_competitor_share(
    ticker: str,
    peers: Optional[str] = Query(None, description="Comma-separated peer keywords"),
    geo: str = Query(""),
    timeframe: str = Query("today 3-m"),
    use_cache: bool = Query(True),
) -> dict:
    """
    Return relative search interest share for a ticker vs its sector peers.
    """
    _, _, engine = _get_engine_v2()
    ticker = ticker.upper()
    peer_list = [p.strip() for p in peers.split(",")] if peers else None

    try:
        sig = engine.competitor_share(
            ticker, peers=peer_list, timeframe=timeframe, geo=geo, use_cache=use_cache
        )
        return _comp_share_to_dict(sig)
    except Exception as exc:
        logger.error("competitor-share error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@google_trends_v2_router.get(
    "/geo-heatmap/{ticker}",
    summary="Geographic interest heatmap for a ticker",
    response_model=dict,
)
def get_geo_heatmap(
    ticker: str,
    geo_scope: str = Query("US", description="Geographic scope: US, GB, '', etc."),
    use_cache: bool = Query(True),
) -> dict:
    """Return state/country breakdown of search interest for a ticker."""
    _, _, engine = _get_engine_v2()
    ticker = ticker.upper()
    try:
        sig = engine.geo_heatmap(ticker, geo_scope=geo_scope, use_cache=use_cache)
        return _geo_to_dict(sig)
    except Exception as exc:
        logger.error("geo-heatmap error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@google_trends_v2_router.get(
    "/momentum/{ticker}",
    summary="4-week vs 52-week trend momentum signal",
    response_model=dict,
)
def get_momentum(
    ticker: str,
    timeframe: str = Query("today 12-m"),
    geo: str = Query("US"),
    use_cache: bool = Query(True),
) -> dict:
    """Return Google Trends momentum signal (4w vs 52w ratio) for a ticker."""
    _, _, engine = _get_engine_v2()
    ticker = ticker.upper()
    try:
        sig = engine.compute_momentum(ticker, timeframe=timeframe, geo=geo, use_cache=use_cache)
        return _momentum_to_dict(sig)
    except Exception as exc:
        logger.error("momentum error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@google_trends_v2_router.get(
    "/seasonal/{ticker}",
    summary="Seasonally adjusted Google Trends series",
    response_model=dict,
)
def get_seasonal(
    ticker: str,
    use_cache: bool = Query(True),
) -> dict:
    """Return raw, seasonal component, and seasonally-adjusted trends series."""
    _, _, engine = _get_engine_v2()
    ticker = ticker.upper()
    try:
        df = engine.seasonal_adjust(ticker, use_cache=use_cache)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No trends data for {ticker}")
        return {
            "ticker": ticker,
            "as_of": date.today().isoformat(),
            "series": df.reset_index().assign(
                date=df.index.astype(str)
            ).to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("seasonal error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@google_trends_v2_router.get(
    "/sector-rotation",
    summary="Sector rotation signals from Google Trends",
    response_model=dict,
)
def get_sector_rotation(
    geo: str = Query("US"),
    timeframe: str = Query("today 3-m"),
    use_cache: bool = Query(True),
) -> dict:
    """Return relative sector search interest for rotation signal detection."""
    _, _, engine = _get_engine_v2()
    try:
        result = engine.sector_rotation_signals(timeframe=timeframe, geo=geo, use_cache=use_cache)
        return result
    except Exception as exc:
        logger.error("sector-rotation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@google_trends_v2_router.get(
    "/related-queries/{ticker}",
    summary="Related queries and rising searches for a ticker",
    response_model=dict,
)
def get_related_queries(
    ticker: str,
    timeframe: str = Query("today 12-m"),
    geo: str = Query(""),
    use_cache: bool = Query(True),
) -> dict:
    """Return top and rising related queries for a ticker keyword."""
    _, adapter, _ = _get_engine_v2()
    ticker = ticker.upper()
    keyword = TICKER_SEARCH_VARIANTS.get(ticker, [f"{ticker} stock"])[0]

    try:
        rq = adapter.related_queries(keyword, timeframe=timeframe, geo=geo, use_cache=use_cache)
        result: dict = {"ticker": ticker, "keyword": keyword, "as_of": date.today().isoformat()}

        if rq and keyword in rq:
            sub = rq[keyword]
            top_df = sub.get("top")
            rising_df = sub.get("rising")
            result["top_queries"] = (
                top_df.to_dict(orient="records") if isinstance(top_df, pd.DataFrame) and not top_df.empty
                else []
            )
            result["rising_queries"] = (
                rising_df.to_dict(orient="records") if isinstance(rising_df, pd.DataFrame) and not rising_df.empty
                else []
            )
        else:
            result["top_queries"] = []
            result["rising_queries"] = []

        return result
    except Exception as exc:
        logger.error("related-queries error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@google_trends_v2_router.get(
    "/topic-clusters/{ticker}",
    summary="Topic cluster analysis for a ticker",
    response_model=dict,
)
def get_topic_clusters(
    ticker: str,
    use_cache: bool = Query(True),
) -> dict:
    """Return thematic clustering of related search queries for a ticker."""
    _, _, engine = _get_engine_v2()
    ticker = ticker.upper()
    try:
        sig = engine.topic_clusters(ticker, use_cache=use_cache)
        return _cluster_to_dict(sig)
    except Exception as exc:
        logger.error("topic-clusters error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@google_trends_v2_router.get(
    "/cache-stats",
    summary="Cache health and eviction stats",
    response_model=dict,
)
def get_cache_stats() -> dict:
    """Return cache database size and evict expired entries."""
    cache, _, _ = _get_engine_v2()
    evicted = cache.evict_expired()
    db_size_bytes = _CACHE_DB_V2.stat().st_size if _CACHE_DB_V2.exists() else 0
    return {
        "db_path": str(_CACHE_DB_V2),
        "db_size_bytes": db_size_bytes,
        "raw_ttl_hours": _RAW_TTL_HOURS,
        "signal_ttl_hours": _SIGNAL_TTL_HOURS,
        "evicted": evicted,
    }


@google_trends_v2_router.get(
    "/surge-history/{ticker}",
    summary="Historical surge alerts for a ticker",
    response_model=dict,
)
def get_surge_history(
    ticker: str,
    days: int = Query(30, ge=1, le=365),
) -> dict:
    """Return historical surge alert log for a ticker from SQLite."""
    cache, _, _ = _get_engine_v2()
    ticker = ticker.upper()
    try:
        history = cache.get_surge_history(ticker, days=days)
        return {
            "ticker": ticker,
            "lookback_days": days,
            "surge_events": [h for h in history if h["is_surge"]],
            "all_events": history,
        }
    except Exception as exc:
        logger.error("surge-history error for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Module-level initialization helper
# ---------------------------------------------------------------------------


def build_trends_engine_v2(
    db_path: Optional[Path] = None,
    hl: str = "en-US",
    tz: int = 360,
) -> tuple[TrendsCacheV2, GoogleTrendsAdapterV2, TrendsSignalEngineV2]:
    """
    Build and return all v2 components for the Google Trends signal system.

    Parameters
    ----------
    db_path:
        Override default SQLite database path.
    hl:
        pytrends host language (default "en-US").
    tz:
        pytrends timezone offset in minutes (default 360 = UTC-6).

    Returns
    -------
    (cache, adapter, engine)
    """
    cache = TrendsCacheV2(db_path or _CACHE_DB_V2)
    adapter = GoogleTrendsAdapterV2(cache=cache, hl=hl, tz=tz)
    engine = TrendsSignalEngineV2(adapter=adapter, cache=cache)
    return cache, adapter, engine


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    geo_arg = sys.argv[2] if len(sys.argv) > 2 else "US"

    logger.info("=== SENTINEL Google Trends v2 ===")
    logger.info("Ticker: %s | Geo: %s", ticker_arg, geo_arg)

    cache, adapter, engine = build_trends_engine_v2()

    # 1. Momentum
    logger.info("\n[1/6] Computing trend momentum for %s...", ticker_arg)
    momentum = engine.compute_momentum(ticker_arg, geo=geo_arg)
    print(f"  Current interest: {momentum.current_interest:.1f}/100")
    print(f"  4w avg: {momentum.avg_4w:.1f} | 52w avg: {momentum.avg_52w:.1f}")
    print(f"  Momentum ratio: {momentum.momentum_ratio_4w:.3f}")
    print(f"  Direction: {momentum.trend_direction}")
    print(f"  {momentum.interpretation}")

    # 2. Surge detection
    logger.info("\n[2/6] Checking for search surge...")
    surge = engine.detect_surge(ticker_arg, geo=geo_arg)
    print(f"  Current: {surge.current_value:.1f} | Baseline: {surge.baseline_mean:.1f}")
    print(f"  Z-score: {surge.z_score:.3f} | IS SURGE: {surge.is_surge}")
    print(f"  Severity: {surge.severity} | Trigger: {surge.probable_trigger}")

    # 3. Macro sentiment
    logger.info("\n[3/6] Fetching macro sentiment...")
    macro = engine.macro_sentiment(geo=geo_arg)
    print(f"  Regime: {macro.regime}")
    print(f"  Fear index: {macro.composite_fear_index:.1f}/100 | Greed: {macro.composite_greed_index:.1f}/100")
    print(f"  Recession: {macro.recession_score:.0f} | Inflation: {macro.inflation_score:.0f}")

    # 4. Fear/Greed
    logger.info("\n[4/6] Computing fear/greed proxy...")
    fg = engine.fear_greed_proxy(geo=geo_arg)
    print(f"  Signal: {fg.signal}")
    print(f"  Fear: {fg.fear_score:.1f} | Greed: {fg.greed_score:.1f}")
    print(f"  Net: {fg.net_signal:+.1f} | Z-score: {fg.z_score_vs_1y:+.2f}")

    # 5. Competitor share
    logger.info("\n[5/6] Computing competitor search share for %s...", ticker_arg)
    comp = engine.competitor_share(ticker_arg)
    print(f"  Ticker share: {comp.ticker_share_pct:.1f}%")
    print(f"  Dominant brand: {comp.dominant_brand}")
    print(f"  Trend: {comp.trend} ({comp.share_vs_prior_month:+.1f}pp)")
    print(f"  {comp.interpretation}")

    # 6. Topic clusters
    logger.info("\n[6/6] Clustering related topics for %s...", ticker_arg)
    clusters = engine.topic_clusters(ticker_arg)
    print(f"  Dominant theme: {clusters.dominant_theme}")
    print(f"  Rising queries: {clusters.rising_queries[:3]}")
    print(f"  {clusters.interpretation}")

    logger.info("\n=== Done ===")
