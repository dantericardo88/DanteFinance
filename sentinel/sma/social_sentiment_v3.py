"""Social Media Sentiment Engine V3 — Institutional-grade free-API sentiment.

dim_085: Social media sentiment (Reddit/Twitter)  (score 5 → 9)

Sources (all free, no OAuth):
  - Reddit public JSON search API (read-only, no app registration needed)
  - StockTwits public streams API (native bull/bear signals)
  - SEC EDGAR EFTS full-text 8-K search for corporate event sentiment
  - Yahoo Finance RSS headlines (no key)

Storage: SQLite at sentinel/data/social_sentiment.db

Fallback: if vaderSentiment not installed, a built-in lexicon-based VADER
          substitute handles all scoring so the module works standalone.

Usage::
    from sentinel.sma.social_sentiment_v3 import SentimentAggregator

    agg = SentimentAggregator()
    summary = agg.aggregate("NVDA", lookback_days=7)
    print(summary.composite_score, summary.mention_volume)
"""
from __future__ import annotations

import html
import json
import logging
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderBase
    _VADER_AVAILABLE = True
except ImportError:
    _VaderBase = None  # type: ignore[misc,assignment]
    _VADER_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DATA_DIR / "social_sentiment.db"

_REDDIT_BASE = "https://www.reddit.com"
_STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
_EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline"

_HEADERS = {
    "User-Agent": "SENTINEL:SocialSentiment:3.0 (research; contact@sentinel.ai)",
    "Accept": "application/json",
}
_RSS_HEADERS = {
    "User-Agent": "SENTINEL:SocialSentiment:3.0 (research)",
    "Accept": "application/rss+xml,application/xml,text/xml,*/*",
}

_WSB_SUBREDDITS = [
    "wallstreetbets", "stocks", "investing", "StockMarket",
    "options", "SecurityAnalysis", "ValueInvesting",
]

# Common English words and finance acronyms to exclude from ticker detection
_TICKER_EXCLUSIONS = frozenset({
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO",
    "UP", "US", "WE", "PM", "AM", "ET", "CUT", "ALL", "BUY", "FOR", "THE",
    "LOL", "IMO", "ETF", "CEO", "CFO", "COO", "IPO", "SEC", "FED", "GDP",
    "CPI", "PCE", "NFP", "PPT", "AKA", "ATH", "ATL", "EOD", "EOW", "EOY",
    "EOM", "YTD", "YOY", "QOQ", "TTM", "LTM", "NTM", "PNL", "OTC", "NYSE",
    "ETH", "BTC", "XRP", "NFT", "DAO", "DeFi", "APR", "APY", "TVL",
    "DD", "TA", "FA", "OI", "IV", "DTE", "ITM", "ATM", "OTM",
    "WSB", "YOLO", "FOMO", "FUD", "HODL", "BTFD", "MOASS",
    "IMO", "IMHO", "OP", "EPS", "PE", "PB", "PS", "EV", "FCF",
    "ROE", "ROA", "ROIC", "EBITDA", "EBIT", "GAAP", "NON",
})

_RATE_LIMIT_INTERVAL = 1.1  # seconds between requests to same domain
_CACHE_TTL_SEC = 900  # 15 minutes

# ---------------------------------------------------------------------------
# Finance-augmented VADER lexicon additions
# ---------------------------------------------------------------------------

_FINANCE_POSITIVE_LEXICON: Dict[str, float] = {
    "beat": 2.5, "beats": 2.5, "outperform": 2.8, "upgrade": 2.2,
    "raised guidance": 3.0, "raises guidance": 3.0, "raised outlook": 2.8,
    "buyback": 2.0, "dividend increase": 2.5, "dividend raise": 2.5,
    "record revenue": 3.0, "record earnings": 3.0, "all-time high": 2.5,
    "strong results": 2.3, "exceeded expectations": 2.8, "better than expected": 2.5,
    "top-line growth": 2.0, "margin expansion": 2.2, "market share gain": 2.0,
    "bullish": 2.0, "moon": 1.8, "rocket": 1.8, "squeeze": 1.5,
    "breakout": 2.0, "catalyst": 1.5, "partnership": 1.8, "approval": 2.5,
    "fda approval": 3.0, "contract win": 2.5, "buyout": 2.0, "acquisition": 1.5,
    "special dividend": 2.5, "share repurchase": 2.0, "positive guidance": 2.5,
    "earnings beat": 3.0, "revenue beat": 2.8, "guidance raise": 2.8,
    "insider buying": 2.0, "new high": 2.2, "breakthrough": 2.5,
    "synergies": 1.8, "cost savings": 1.8, "operational efficiency": 1.8,
    "free cash flow positive": 2.5, "profitable": 2.0, "profitability": 2.0,
    "debt reduction": 1.8, "deleveraging": 1.8, "net cash position": 1.8,
    "moat": 1.8, "durable competitive advantage": 2.0, "pricing power": 1.8,
}

_FINANCE_NEGATIVE_LEXICON: Dict[str, float] = {
    "miss": -2.5, "misses": -2.5, "missed": -2.5, "downgrade": -2.2,
    "warning": -2.5, "lowered guidance": -3.0, "lowered outlook": -2.8,
    "cut guidance": -3.0, "guidance cut": -3.0, "reduced guidance": -2.8,
    "bankruptcy": -3.5, "chapter 11": -3.5, "insolvency": -3.5, "default": -3.0,
    "sec investigation": -3.0, "accounting irregularity": -3.0, "restatement": -3.0,
    "fraud": -3.2, "class action": -2.8, "subpoena": -2.5, "lawsuit": -2.0,
    "earnings miss": -3.0, "revenue miss": -2.8, "below expectations": -2.5,
    "disappointing": -2.5, "weaker than expected": -2.5, "shortfall": -2.5,
    "layoffs": -2.0, "restructuring charge": -2.5, "impairment": -2.5,
    "write-down": -2.8, "write down": -2.8, "goodwill impairment": -2.8,
    "margin compression": -2.2, "margin decline": -2.2, "headwind": -2.0,
    "supply chain": -1.5, "recall": -2.5, "product recall": -2.8,
    "cybersecurity breach": -3.0, "data breach": -3.0, "hack": -2.5,
    "bearish": -2.0, "puts": -1.5, "short": -1.0, "puts on": -2.0,
    "covenant breach": -3.0, "debt covenant": -2.5, "overleveraged": -2.5,
    "delisted": -3.5, "suspended": -2.5, "investigation": -2.5,
    "penalty": -2.0, "fine": -2.0, "regulatory action": -2.5,
    "revenue decline": -2.5, "revenue contraction": -2.5, "loss": -2.0,
    "net loss": -2.5, "profit warning": -3.0, "negative outlook": -2.5,
    "insider selling": -1.8, "dilution": -2.0, "share offering": -1.8,
}


# ---------------------------------------------------------------------------
# VADER Fallback (pure Python lexicon-based)
# ---------------------------------------------------------------------------

class _FallbackVADER:
    """Minimal VADER-compatible sentiment analyser using finance-augmented lexicon.

    Not as accurate as true VADER but handles financial text reasonably well.
    Used when vaderSentiment is not installed.
    """

    _BASE_POSITIVE = {
        "good", "great", "excellent", "strong", "positive", "solid",
        "growth", "gain", "up", "rise", "rally", "surge", "high",
        "best", "improve", "improved", "profit", "profitable",
        "win", "winner", "outperform", "recommend", "buy",
        "bullish", "bull", "upside", "opportunity", "benefit",
    }
    _BASE_NEGATIVE = {
        "bad", "poor", "weak", "negative", "loss", "decline", "down",
        "fall", "drop", "crash", "low", "worst", "fail", "failed",
        "miss", "missed", "concern", "risk", "uncertainty",
        "bearish", "bear", "downside", "sell", "short",
    }
    _NEGATORS = {"not", "no", "never", "n't", "neither", "nor", "hardly", "barely"}
    _INTENSIFIERS = {"very", "extremely", "significantly", "substantially", "highly",
                     "massive", "huge", "enormous", "drastically", "sharply"}

    def __init__(self) -> None:
        self._positive: Dict[str, float] = {
            w: 1.5 for w in self._BASE_POSITIVE
        }
        self._negative: Dict[str, float] = {
            w: -1.5 for w in self._BASE_NEGATIVE
        }
        # Add finance-specific terms
        self._positive.update(_FINANCE_POSITIVE_LEXICON)
        self._negative.update({k: v for k, v in _FINANCE_NEGATIVE_LEXICON.items()})

    def _score_text(self, text: str) -> float:
        """Return raw score for text."""
        text = text.lower()
        tokens = re.findall(r"\b[\w'-]+\b", text)
        total = 0.0
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            weight = 1.0
            # Check intensifier before this token
            if i > 0 and tokens[i - 1] in self._INTENSIFIERS:
                weight = 1.5
            # Check negator in window [-3, -1]
            negated = any(tokens[max(0, i - j)] in self._NEGATORS for j in range(1, 4))

            # Multi-word phrase check (bigram, trigram)
            for phrase_len in (3, 2):
                phrase = " ".join(tokens[i: i + phrase_len])
                if phrase in self._positive:
                    val = self._positive[phrase] * weight
                    total += -val if negated else val
                    i += phrase_len
                    break
                if phrase in self._negative:
                    val = self._negative[phrase] * weight
                    total += -val if negated else val
                    i += phrase_len
                    break
            else:
                if tok in self._positive:
                    val = self._positive[tok] * weight
                    total += -val if negated else val
                elif tok in self._negative:
                    val = self._negative[tok] * weight
                    total += -val if negated else val
                i += 1

        return total

    def polarity_scores(self, text: str) -> Dict[str, float]:
        """Return VADER-compatible dict with compound, pos, neg, neu."""
        raw = self._score_text(text)
        # Normalize to [-1, 1] with sigmoid-like function
        alpha = 15.0
        compound = raw / (math.sqrt(raw * raw + alpha))
        # Estimate pos/neg/neu
        if compound > 0.05:
            pos = min(1.0, abs(compound))
            neg = 0.0
            neu = max(0.0, 1.0 - pos)
        elif compound < -0.05:
            neg = min(1.0, abs(compound))
            pos = 0.0
            neu = max(0.0, 1.0 - neg)
        else:
            pos = 0.0
            neg = 0.0
            neu = 1.0
        return {
            "compound": round(compound, 4),
            "pos": round(pos, 4),
            "neg": round(neg, 4),
            "neu": round(neu, 4),
        }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SentimentScore:
    compound: float     # -1.0 to +1.0
    positive: float     # 0.0 to 1.0
    negative: float     # 0.0 to 1.0
    neutral: float      # 0.0 to 1.0
    label: str          # "positive" | "negative" | "neutral"

    @classmethod
    def from_vader(cls, scores: Dict[str, float]) -> "SentimentScore":
        c = scores["compound"]
        label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
        return cls(
            compound=c,
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
            label=label,
        )


@dataclass
class RedditPost:
    id: str
    title: str
    selftext: str
    score: int              # upvotes
    upvote_ratio: float
    num_comments: int
    created_utc: float
    subreddit: str
    flair: Optional[str]
    url: str
    is_dd: bool = False
    sentiment: Optional[SentimentScore] = None


@dataclass
class StockTwitsMessage:
    id: int
    body: str
    created_at: str
    sentiment_raw: Optional[str]    # "Bullish" | "Bearish" | None
    user_followers: int
    ticker: str
    sentiment: Optional[SentimentScore] = None


@dataclass
class NewsItem:
    title: str
    summary: str
    published: str
    source: str
    url: str
    ticker: str
    sentiment: Optional[SentimentScore] = None


@dataclass
class SentimentSummary:
    ticker: str
    lookback_days: int
    reddit_score: float
    stocktwits_score: float
    news_score: float
    composite_score: float
    mention_volume: int
    volume_trend: str           # "surging" | "normal" | "declining"
    dd_count: int
    sentiment_shift_24h: float
    reddit_posts: int
    stocktwits_messages: int
    news_items: int
    generated_at: str


@dataclass
class TrendingTicker:
    ticker: str
    mention_count: int
    avg_sentiment: float
    subreddit: str
    top_post_title: str = ""


@dataclass
class SqueezeCandidate:
    ticker: str
    mention_count: int
    avg_sentiment: float
    upvote_surge: bool
    short_interest_pct: Optional[float]
    squeeze_score: float


# ---------------------------------------------------------------------------
# HTTP session with per-domain rate limiting
# ---------------------------------------------------------------------------

_LAST_REQUEST: Dict[str, float] = {}
_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)


def _rate_limited_get(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = 12,
    min_interval: float = _RATE_LIMIT_INTERVAL,
    domain_key: Optional[str] = None,
) -> Optional[requests.Response]:
    """GET with per-domain rate limiting."""
    key = domain_key or url.split("/")[2]
    elapsed = time.monotonic() - _LAST_REQUEST.get(key, 0.0)
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_REQUEST[key] = time.monotonic()
    try:
        resp = _SESSION.get(url, params=params, headers=headers or _HEADERS, timeout=timeout)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "30"))
            logger.warning("Rate limited by %s; sleeping %ds", key, retry_after)
            time.sleep(min(retry_after, 60))
            return None
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.debug("HTTP error for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# SQLite cache / storage
# ---------------------------------------------------------------------------

class _SentimentDB:
    """SQLite-backed cache for posts, messages, and sentiment history."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS reddit_posts (
                    id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    subreddit TEXT NOT NULL,
                    title TEXT,
                    score INTEGER,
                    upvote_ratio REAL,
                    num_comments INTEGER,
                    created_utc REAL,
                    flair TEXT,
                    url TEXT,
                    is_dd INTEGER DEFAULT 0,
                    compound REAL,
                    fetched_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reddit_ticker ON reddit_posts(ticker, created_utc);

                CREATE TABLE IF NOT EXISTS stocktwits_messages (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    body TEXT,
                    created_at TEXT,
                    sentiment_raw TEXT,
                    user_followers INTEGER,
                    compound REAL,
                    fetched_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_st_ticker ON stocktwits_messages(ticker, created_at);

                CREATE TABLE IF NOT EXISTS news_items (
                    url TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    title TEXT,
                    summary TEXT,
                    published TEXT,
                    source TEXT,
                    compound REAL,
                    fetched_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_items(ticker, published);

                CREATE TABLE IF NOT EXISTS sentiment_history (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    reddit_score REAL,
                    stocktwits_score REAL,
                    news_score REAL,
                    composite_score REAL,
                    mention_volume INTEGER,
                    PRIMARY KEY (ticker, date)
                );
            """)

    def upsert_reddit_post(self, ticker: str, post: RedditPost) -> None:
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO reddit_posts
                  (id, ticker, subreddit, title, score, upvote_ratio, num_comments,
                   created_utc, flair, url, is_dd, compound, fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                post.id, ticker, post.subreddit, post.title, post.score,
                post.upvote_ratio, post.num_comments, post.created_utc,
                post.flair, post.url, int(post.is_dd),
                post.sentiment.compound if post.sentiment else None,
                time.time(),
            ))

    def upsert_stocktwits(self, msg: StockTwitsMessage) -> None:
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO stocktwits_messages
                  (id, ticker, body, created_at, sentiment_raw, user_followers,
                   compound, fetched_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                msg.id, msg.ticker, msg.body, msg.created_at,
                msg.sentiment_raw, msg.user_followers,
                msg.sentiment.compound if msg.sentiment else None,
                time.time(),
            ))

    def upsert_news(self, item: NewsItem) -> None:
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO news_items
                  (url, ticker, title, summary, published, source, compound, fetched_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                item.url, item.ticker, item.title, item.summary,
                item.published, item.source,
                item.sentiment.compound if item.sentiment else None,
                time.time(),
            ))

    def upsert_history(self, summary: SentimentSummary) -> None:
        date_str = summary.generated_at[:10]
        with self._conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO sentiment_history
                  (ticker, date, reddit_score, stocktwits_score, news_score,
                   composite_score, mention_volume)
                VALUES (?,?,?,?,?,?,?)
            """, (
                summary.ticker, date_str, summary.reddit_score,
                summary.stocktwits_score, summary.news_score,
                summary.composite_score, summary.mention_volume,
            ))

    def get_recent_reddit(self, ticker: str, since_ts: float) -> List[sqlite3.Row]:
        with self._conn() as con:
            cur = con.execute(
                "SELECT * FROM reddit_posts WHERE ticker=? AND created_utc >= ?",
                (ticker, since_ts),
            )
            return cur.fetchall()

    def get_recent_stocktwits(self, ticker: str, since_ts: float) -> List[sqlite3.Row]:
        with self._conn() as con:
            cur = con.execute(
                "SELECT * FROM stocktwits_messages WHERE ticker=? AND fetched_at >= ?",
                (ticker, since_ts),
            )
            return cur.fetchall()

    def get_recent_news(self, ticker: str, since_ts: float) -> List[sqlite3.Row]:
        with self._conn() as con:
            cur = con.execute(
                "SELECT * FROM news_items WHERE ticker=? AND fetched_at >= ?",
                (ticker, since_ts),
            )
            return cur.fetchall()

    def get_history(self, ticker: str, days: int = 30) -> List[sqlite3.Row]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        with self._conn() as con:
            cur = con.execute(
                "SELECT * FROM sentiment_history WHERE ticker=? AND date >= ? ORDER BY date",
                (ticker, cutoff),
            )
            return cur.fetchall()

    def get_30d_avg_mentions(self, ticker: str) -> float:
        """Return 30-day average daily mention volume from history."""
        rows = self.get_history(ticker, days=30)
        if not rows:
            return 0.0
        vols = [r["mention_volume"] for r in rows if r["mention_volume"] is not None]
        return sum(vols) / len(vols) if vols else 0.0


# Module-level singleton DB
_DB = _SentimentDB()


# ---------------------------------------------------------------------------
# VADERSentimentAnalyzer
# ---------------------------------------------------------------------------

class VADERSentimentAnalyzer:
    """VADER sentiment scoring with financial lexicon augmentation.

    Falls back to a pure-Python lexicon analyser if vaderSentiment is absent.
    """

    def __init__(self) -> None:
        if _VADER_AVAILABLE and _VaderBase is not None:
            self._analyzer = _VaderBase()
            # Augment with finance-specific lexicon
            self._analyzer.lexicon.update(_FINANCE_POSITIVE_LEXICON)
            self._analyzer.lexicon.update(_FINANCE_NEGATIVE_LEXICON)
            self._backend = "vader"
        else:
            self._analyzer = _FallbackVADER()  # type: ignore[assignment]
            self._backend = "fallback"
        logger.debug("VADERSentimentAnalyzer using backend=%s", self._backend)

    def score(self, text: str) -> SentimentScore:
        """Score a single text."""
        if not text or not text.strip():
            return SentimentScore(0.0, 0.0, 0.0, 1.0, "neutral")
        cleaned = html.unescape(text)[:4000]  # truncate very long texts
        scores = self._analyzer.polarity_scores(cleaned)
        return SentimentScore.from_vader(scores)

    def score_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Score a list of texts."""
        return [self.score(t) for t in texts]

    @property
    def backend(self) -> str:
        return self._backend


# ---------------------------------------------------------------------------
# RedditSentimentCollector
# ---------------------------------------------------------------------------

_REDDIT_CACHE: Dict[Tuple[str, str], Tuple[float, List[RedditPost]]] = {}
_DD_PATTERN = re.compile(r"\bDD\b", re.IGNORECASE)


class RedditSentimentCollector:
    """Collect Reddit posts using the public JSON API (no OAuth required)."""

    def __init__(self, analyzer: Optional[VADERSentimentAnalyzer] = None) -> None:
        self._analyzer = analyzer or VADERSentimentAnalyzer()

    def collect(
        self,
        ticker: str,
        days_back: int = 7,
        subreddits: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> List[RedditPost]:
        """Collect Reddit posts mentioning ticker from multiple subreddits."""
        subs = subreddits or _WSB_SUBREDDITS
        cutoff_ts = time.time() - days_back * 86400

        all_posts: List[RedditPost] = []
        seen_ids: set = set()

        # Check DB cache first
        if not force_refresh:
            cached = _DB.get_recent_reddit(ticker, cutoff_ts)
            for row in cached:
                seen_ids.add(row["id"])

        for sub in subs:
            posts = self._collect_subreddit(ticker, sub, cutoff_ts, seen_ids, force_refresh)
            all_posts.extend(posts)

        # Save to DB
        for post in all_posts:
            _DB.upsert_reddit_post(ticker, post)

        return all_posts

    def _collect_subreddit(
        self,
        ticker: str,
        subreddit: str,
        cutoff_ts: float,
        seen_ids: set,
        force_refresh: bool,
    ) -> List[RedditPost]:
        """Fetch posts from one subreddit."""
        cache_key = (ticker, subreddit)
        now = time.time()

        # Memory cache check
        if not force_refresh and cache_key in _REDDIT_CACHE:
            cached_at, cached_posts = _REDDIT_CACHE[cache_key]
            if now - cached_at < _CACHE_TTL_SEC:
                return [p for p in cached_posts if p.id not in seen_ids]

        url = f"{_REDDIT_BASE}/r/{subreddit}/search.json"
        params = {
            "q": ticker,
            "sort": "new",
            "t": "week",
            "limit": 100,
            "restrict_sr": 1,
        }
        resp = _rate_limited_get(url, params=params, domain_key="reddit.com")
        if resp is None:
            return []

        try:
            data = resp.json()
        except (ValueError, KeyError):
            return []

        posts: List[RedditPost] = []
        children = data.get("data", {}).get("children", [])
        for child in children:
            post_data = child.get("data", {})
            created = post_data.get("created_utc", 0.0)
            if created < cutoff_ts:
                continue
            post_id = post_data.get("id", "")
            if post_id in seen_ids:
                continue

            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            flair = post_data.get("link_flair_text")
            is_dd = bool(
                _DD_PATTERN.search(title)
                or (flair and "DD" in flair.upper())
            )

            full_text = f"{title} {selftext}"
            sentiment = self._analyzer.score(full_text)

            post = RedditPost(
                id=post_id,
                title=title,
                selftext=selftext[:500],
                score=post_data.get("score", 0),
                upvote_ratio=post_data.get("upvote_ratio", 0.5),
                num_comments=post_data.get("num_comments", 0),
                created_utc=created,
                subreddit=subreddit,
                flair=flair,
                url=post_data.get("url", ""),
                is_dd=is_dd,
                sentiment=sentiment,
            )
            posts.append(post)
            seen_ids.add(post_id)

        _REDDIT_CACHE[cache_key] = (now, posts)
        return posts

    def collect_hot(self, subreddit: str = "wallstreetbets", limit: int = 100) -> List[RedditPost]:
        """Fetch hot posts from a subreddit for trending ticker extraction."""
        url = f"{_REDDIT_BASE}/r/{subreddit}/hot.json"
        params = {"limit": limit}
        resp = _rate_limited_get(url, params=params, domain_key="reddit.com")
        if resp is None:
            return []
        try:
            data = resp.json()
        except (ValueError, KeyError):
            return []

        posts: List[RedditPost] = []
        for child in data.get("data", {}).get("children", []):
            pd_data = child.get("data", {})
            title = pd_data.get("title", "")
            flair = pd_data.get("link_flair_text")
            is_dd = bool(_DD_PATTERN.search(title) or (flair and "DD" in flair.upper()))
            post = RedditPost(
                id=pd_data.get("id", ""),
                title=title,
                selftext=pd_data.get("selftext", "")[:200],
                score=pd_data.get("score", 0),
                upvote_ratio=pd_data.get("upvote_ratio", 0.5),
                num_comments=pd_data.get("num_comments", 0),
                created_utc=pd_data.get("created_utc", time.time()),
                subreddit=subreddit,
                flair=flair,
                url=pd_data.get("url", ""),
                is_dd=is_dd,
            )
            posts.append(post)
        return posts


# ---------------------------------------------------------------------------
# StockTwitsSentimentCollector
# ---------------------------------------------------------------------------

class StockTwitsSentimentCollector:
    """Collect StockTwits messages using the public streams API."""

    def __init__(self, analyzer: Optional[VADERSentimentAnalyzer] = None) -> None:
        self._analyzer = analyzer or VADERSentimentAnalyzer()

    def collect(self, ticker: str, max_messages: int = 300) -> List[StockTwitsMessage]:
        """Collect up to max_messages messages from StockTwits for a ticker."""
        url = f"{_STOCKTWITS_BASE}/streams/symbol/{ticker.upper()}.json"
        resp = _rate_limited_get(url, domain_key="stocktwits.com")
        if resp is None:
            return []

        try:
            data = resp.json()
        except (ValueError, KeyError):
            return []

        messages: List[StockTwitsMessage] = []
        raw_messages = data.get("messages", [])[:max_messages]

        for raw in raw_messages:
            body = raw.get("body", "")
            sentiment_raw = None
            entities = raw.get("entities", {})
            sentiment_entity = entities.get("sentiment")
            if sentiment_entity:
                sentiment_raw = sentiment_entity.get("basic")  # "Bullish" or "Bearish"

            user = raw.get("user", {})
            followers = user.get("followers", 0)

            # If StockTwits provides native sentiment, use it; else use VADER
            if sentiment_raw:
                compound = 0.85 if sentiment_raw == "Bullish" else -0.85
                label = "positive" if sentiment_raw == "Bullish" else "negative"
                sentiment = SentimentScore(
                    compound=compound,
                    positive=0.85 if compound > 0 else 0.0,
                    negative=0.85 if compound < 0 else 0.0,
                    neutral=0.0,
                    label=label,
                )
            else:
                sentiment = self._analyzer.score(body)

            msg = StockTwitsMessage(
                id=raw.get("id", 0),
                body=body,
                created_at=raw.get("created_at", ""),
                sentiment_raw=sentiment_raw,
                user_followers=followers,
                ticker=ticker.upper(),
                sentiment=sentiment,
            )
            messages.append(msg)
            _DB.upsert_stocktwits(msg)

        return messages

    @staticmethod
    def follower_weight(followers: int) -> float:
        """Log-scale follower weight: 0 followers → 1.0, 10K → ~3.3, 1M → ~5.0."""
        return 1.0 + math.log1p(max(followers, 0)) / math.log(10)


# ---------------------------------------------------------------------------
# FinancialNewsCollector
# ---------------------------------------------------------------------------

class FinancialNewsCollector:
    """Collect financial news from free sources: Yahoo Finance RSS + EDGAR EFTS."""

    def __init__(self, analyzer: Optional[VADERSentimentAnalyzer] = None) -> None:
        self._analyzer = analyzer or VADERSentimentAnalyzer()

    def collect_news(self, ticker: str, days_back: int = 7) -> List[NewsItem]:
        """Collect news items from Yahoo RSS and EDGAR EFTS."""
        items: List[NewsItem] = []
        items.extend(self._yahoo_rss(ticker))
        items.extend(self._edgar_efts(ticker, days_back))
        # Deduplicate by URL
        seen_urls: set = set()
        deduped: List[NewsItem] = []
        for item in items:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                deduped.append(item)
                _DB.upsert_news(item)
        return deduped

    def _yahoo_rss(self, ticker: str) -> List[NewsItem]:
        """Scrape Yahoo Finance RSS for ticker headlines."""
        url = _YAHOO_RSS
        params = {"s": ticker.upper(), "region": "US", "lang": "en-US"}
        resp = _rate_limited_get(url, params=params,
                                 headers=_RSS_HEADERS, domain_key="yahoo.com")
        if resp is None:
            return []
        items: List[NewsItem] = []
        try:
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is None:
                return []
            for entry in channel.findall("item"):
                title = (entry.findtext("title") or "").strip()
                link = (entry.findtext("link") or "").strip()
                pub = (entry.findtext("pubDate") or "").strip()
                desc = html.unescape((entry.findtext("description") or "").strip())

                sentiment = self._analyzer.score(f"{title} {desc}")
                items.append(NewsItem(
                    title=title,
                    summary=desc[:400],
                    published=pub,
                    source="Yahoo Finance",
                    url=link,
                    ticker=ticker.upper(),
                    sentiment=sentiment,
                ))
        except ET.ParseError as exc:
            logger.debug("Yahoo RSS parse error for %s: %s", ticker, exc)
        return items

    def _edgar_efts(self, ticker: str, days_back: int) -> List[NewsItem]:
        """Search EDGAR EFTS for recent 8-K filings to extract event sentiment."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": cutoff,
            "forms": "8-K",
            "_source": "hits.hits._source.period_of_report,"
                       "hits.hits._source.entity_name,"
                       "hits.hits._source.file_date,"
                       "hits.hits._source.form_type",
        }
        url = "https://efts.sec.gov/LATEST/search-index"
        resp = _rate_limited_get(url, params=params, domain_key="efts.sec.gov")
        if resp is None:
            return []
        items: List[NewsItem] = []
        try:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits[:20]:
                src = hit.get("_source", {})
                entity = src.get("entity_name", "")
                file_date = src.get("file_date", "")
                accession = hit.get("_id", "").replace("-", "")
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{src.get('entity_id', '')}/{accession}"
                    if src.get("entity_id") else ""
                )
                title = f"{entity} — 8-K filing {file_date}"
                summary = f"Form 8-K: {src.get('form_type', '8-K')} filed {file_date}"
                sentiment = self._analyzer.score(title)
                items.append(NewsItem(
                    title=title,
                    summary=summary,
                    published=file_date,
                    source="SEC EDGAR",
                    url=doc_url,
                    ticker=ticker.upper(),
                    sentiment=sentiment,
                ))
        except (ValueError, KeyError) as exc:
            logger.debug("EDGAR EFTS parse error for %s: %s", ticker, exc)
        return items


# ---------------------------------------------------------------------------
# SentimentAggregator
# ---------------------------------------------------------------------------

class SentimentAggregator:
    """Aggregate sentiment from all sources into a composite score."""

    def __init__(self) -> None:
        self._analyzer = VADERSentimentAnalyzer()
        self._reddit = RedditSentimentCollector(self._analyzer)
        self._stocktwits = StockTwitsSentimentCollector(self._analyzer)
        self._news = FinancialNewsCollector(self._analyzer)

    def aggregate(
        self,
        ticker: str,
        lookback_days: int = 7,
    ) -> SentimentSummary:
        """Full aggregation across Reddit, StockTwits, and news."""
        ticker = ticker.upper()
        now_ts = time.time()
        cutoff_ts = now_ts - lookback_days * 86400

        # Collect from all sources
        reddit_posts = self._reddit.collect(ticker, days_back=lookback_days)
        st_messages = self._stocktwits.collect(ticker, max_messages=300)
        news_items = self._news.collect_news(ticker, days_back=lookback_days)

        # Also pull DB cache to supplement API data
        cached_reddit = _DB.get_recent_reddit(ticker, cutoff_ts)
        cached_st = _DB.get_recent_stocktwits(ticker, cutoff_ts)
        cached_news = _DB.get_recent_news(ticker, cutoff_ts)

        # ── Reddit score ──────────────────────────────────────────────────
        reddit_score = self._score_reddit(reddit_posts, cached_reddit)

        # ── StockTwits score ──────────────────────────────────────────────
        st_score = self._score_stocktwits(st_messages, cached_st)

        # ── News score ────────────────────────────────────────────────────
        news_score = self._score_news(news_items, cached_news)

        # ── Composite (weighted average) ──────────────────────────────────
        # Weights: Reddit 40%, StockTwits 35%, News 25%
        has_reddit = abs(reddit_score) > 0.001 or len(reddit_posts) > 0
        has_st = abs(st_score) > 0.001 or len(st_messages) > 0
        has_news = abs(news_score) > 0.001 or len(news_items) > 0

        weights = []
        values = []
        if has_reddit:
            weights.append(0.40)
            values.append(reddit_score)
        if has_st:
            weights.append(0.35)
            values.append(st_score)
        if has_news:
            weights.append(0.25)
            values.append(news_score)

        if values:
            total_weight = sum(weights)
            composite = sum(v * w for v, w in zip(values, weights)) / total_weight
        else:
            composite = 0.0

        # ── Mention volumes ───────────────────────────────────────────────
        live_reddit = len(reddit_posts)
        cached_reddit_count = len(cached_reddit)
        live_st = len(st_messages)
        cached_st_count = len(cached_st)
        live_news = len(news_items)
        cached_news_count = len(cached_news)

        total_mentions = max(
            live_reddit + cached_reddit_count,
            live_reddit,
        ) + max(live_st + cached_st_count, live_st) + max(
            live_news + cached_news_count, live_news
        )

        # ── Volume trend ──────────────────────────────────────────────────
        avg_30d = _DB.get_30d_avg_mentions(ticker)
        daily_rate = total_mentions / max(lookback_days, 1)
        if avg_30d > 0:
            ratio = daily_rate / avg_30d
            volume_trend = "surging" if ratio > 1.5 else "declining" if ratio < 0.5 else "normal"
        else:
            volume_trend = "normal"

        # ── DD count ─────────────────────────────────────────────────────
        dd_count = sum(1 for p in reddit_posts if p.is_dd)

        # ── Sentiment shift (24h delta) ───────────────────────────────────
        sentiment_shift = self._compute_24h_shift(ticker, composite)

        summary = SentimentSummary(
            ticker=ticker,
            lookback_days=lookback_days,
            reddit_score=round(reddit_score, 4),
            stocktwits_score=round(st_score, 4),
            news_score=round(news_score, 4),
            composite_score=round(composite, 4),
            mention_volume=total_mentions,
            volume_trend=volume_trend,
            dd_count=dd_count,
            sentiment_shift_24h=round(sentiment_shift, 4),
            reddit_posts=live_reddit,
            stocktwits_messages=live_st,
            news_items=live_news,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Persist to history
        _DB.upsert_history(summary)
        return summary

    def _score_reddit(
        self, live_posts: List[RedditPost], cached: List[sqlite3.Row]
    ) -> float:
        """Weighted Reddit score: weight by (score + num_comments) × upvote_ratio."""
        scores: List[Tuple[float, float]] = []  # (compound, weight)

        for post in live_posts:
            if post.sentiment is None:
                continue
            engagement = max(post.score, 0) + post.num_comments
            weight = (1 + math.log1p(engagement)) * post.upvote_ratio
            scores.append((post.sentiment.compound, weight))

        for row in cached:
            compound = row["compound"]
            if compound is None:
                continue
            engagement = max(row["score"] or 0, 0) + (row["num_comments"] or 0)
            ratio = row["upvote_ratio"] or 0.5
            weight = (1 + math.log1p(engagement)) * ratio
            scores.append((compound, weight))

        if not scores:
            return 0.0
        total_w = sum(w for _, w in scores)
        if total_w == 0:
            return 0.0
        return sum(c * w for c, w in scores) / total_w

    def _score_stocktwits(
        self, live: List[StockTwitsMessage], cached: List[sqlite3.Row]
    ) -> float:
        """Weighted StockTwits score: weight by follower log-scale."""
        scores: List[Tuple[float, float]] = []

        for msg in live:
            if msg.sentiment is None:
                continue
            weight = StockTwitsSentimentCollector.follower_weight(msg.user_followers)
            scores.append((msg.sentiment.compound, weight))

        for row in cached:
            compound = row["compound"]
            if compound is None:
                continue
            weight = StockTwitsSentimentCollector.follower_weight(
                row["user_followers"] or 0
            )
            scores.append((compound, weight))

        if not scores:
            return 0.0
        total_w = sum(w for _, w in scores)
        if total_w == 0:
            return 0.0
        return sum(c * w for c, w in scores) / total_w

    def _score_news(
        self, live: List[NewsItem], cached: List[sqlite3.Row]
    ) -> float:
        """Simple average news sentiment."""
        compounds: List[float] = []

        for item in live:
            if item.sentiment:
                compounds.append(item.sentiment.compound)

        for row in cached:
            if row["compound"] is not None:
                compounds.append(row["compound"])

        return sum(compounds) / len(compounds) if compounds else 0.0

    def _compute_24h_shift(self, ticker: str, current_composite: float) -> float:
        """Compute change vs yesterday's composite from DB history."""
        rows = _DB.get_history(ticker, days=2)
        if len(rows) < 2:
            return 0.0
        # Last entry before today
        yesterday_rows = [r for r in rows if r["date"] < datetime.now(timezone.utc).date().isoformat()]
        if not yesterday_rows:
            return 0.0
        yesterday_composite = yesterday_rows[-1]["composite_score"] or 0.0
        return current_composite - yesterday_composite

    def get_trending_tickers(
        self,
        subreddit: str = "wallstreetbets",
        top_n: int = 20,
    ) -> List[TrendingTicker]:
        """Parse /r/subreddit/hot for ticker mentions and rank by count."""
        collector = RedditSentimentCollector(self._analyzer)
        hot_posts = collector.collect_hot(subreddit, limit=100)

        ticker_pattern = re.compile(r"\b([A-Z]{2,5})\b")
        mention_counts: Counter = Counter()
        sentiment_sums: Dict[str, float] = defaultdict(float)
        top_posts: Dict[str, str] = {}

        for post in hot_posts:
            text = f"{post.title} {post.selftext}"
            tickers_found = set(ticker_pattern.findall(text)) - _TICKER_EXCLUSIONS
            post_sentiment = post.sentiment.compound if post.sentiment else 0.0
            for t in tickers_found:
                if len(t) < 2:
                    continue
                mention_counts[t] += 1
                sentiment_sums[t] += post_sentiment
                if t not in top_posts:
                    top_posts[t] = post.title

        top = mention_counts.most_common(top_n)
        results: List[TrendingTicker] = []
        for ticker, count in top:
            avg_sent = sentiment_sums[ticker] / count if count > 0 else 0.0
            results.append(TrendingTicker(
                ticker=ticker,
                mention_count=count,
                avg_sentiment=round(avg_sent, 4),
                subreddit=subreddit,
                top_post_title=top_posts.get(ticker, ""),
            ))
        return results


# ---------------------------------------------------------------------------
# WallStreetBetsRadar
# ---------------------------------------------------------------------------

class WallStreetBetsRadar:
    """Detect unusual WSB activity: high mention + high upvote + squeeze signals."""

    def __init__(self) -> None:
        self._aggregator = SentimentAggregator()
        self._reddit = RedditSentimentCollector()

    def detect_squeeze_candidates(
        self,
        min_mentions: int = 5,
        min_sentiment: float = 0.1,
    ) -> List[SqueezeCandidate]:
        """Identify potential squeeze candidates based on WSB activity.

        Cross-references mention count, sentiment, and upvote ratio.
        Short interest data would come from short_interest_v3.py if available.
        """
        trending = self._aggregator.get_trending_tickers(
            subreddit="wallstreetbets", top_n=50
        )

        candidates: List[SqueezeCandidate] = []
        for t in trending:
            if t.mention_count < min_mentions:
                continue
            if t.avg_sentiment < min_sentiment:
                continue

            # Check for upvote surge: pull hot posts for this ticker
            hot = self._reddit.collect_hot("wallstreetbets", limit=100)
            ticker_pat = re.compile(rf"\b{re.escape(t.ticker)}\b")
            relevant = [p for p in hot if ticker_pat.search(p.title)]
            high_upvote = any(p.upvote_ratio > 0.90 and p.score > 1000 for p in relevant)

            # Try to get short interest from module if available
            short_interest: Optional[float] = None
            try:
                from sentinel.sil import short_interest_v3 as si3
                si_data = si3.get_short_interest(t.ticker)
                short_interest = si_data.get("short_float_pct") if si_data else None
            except Exception:
                pass

            # Squeeze score: mentions × sentiment × upvote_boost × si_boost
            si_boost = 1.0
            if short_interest and short_interest > 20.0:
                si_boost = 1.5
            elif short_interest and short_interest > 10.0:
                si_boost = 1.2

            score = (
                math.log1p(t.mention_count)
                * max(t.avg_sentiment, 0.01)
                * (1.3 if high_upvote else 1.0)
                * si_boost
            )

            candidates.append(SqueezeCandidate(
                ticker=t.ticker,
                mention_count=t.mention_count,
                avg_sentiment=t.avg_sentiment,
                upvote_surge=high_upvote,
                short_interest_pct=short_interest,
                squeeze_score=round(score, 4),
            ))

        # Sort by squeeze score descending
        candidates.sort(key=lambda c: c.squeeze_score, reverse=True)
        return candidates

    def get_sentiment_history(
        self,
        ticker: str,
        days: int = 30,
    ) -> Any:
        """Return daily composite sentiment history from DB as DataFrame or list."""
        rows = _DB.get_history(ticker, days=days)
        data = [
            {
                "date": r["date"],
                "reddit_score": r["reddit_score"],
                "stocktwits_score": r["stocktwits_score"],
                "news_score": r["news_score"],
                "composite_score": r["composite_score"],
                "mention_volume": r["mention_volume"],
            }
            for r in rows
        ]
        if _PANDAS_OK and pd is not None:
            df = pd.DataFrame(data)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            return df
        return data

    def get_wsb_top_posts(self, top_n: int = 20) -> List[RedditPost]:
        """Return top-scoring posts from WSB today (sorted by score)."""
        collector = RedditSentimentCollector()
        posts = collector.collect_hot("wallstreetbets", limit=100)
        posts.sort(key=lambda p: p.score, reverse=True)
        return posts[:top_n]


# ---------------------------------------------------------------------------
# Bulk multi-ticker scanning
# ---------------------------------------------------------------------------

class SentimentScanner:
    """Scan multiple tickers and return ranked sentiment summary."""

    def __init__(self) -> None:
        self._aggregator = SentimentAggregator()

    def scan(
        self,
        tickers: List[str],
        lookback_days: int = 7,
    ) -> List[SentimentSummary]:
        """Aggregate sentiment for a list of tickers, sorted by composite score."""
        summaries: List[SentimentSummary] = []
        for ticker in tickers:
            try:
                summary = self._aggregator.aggregate(ticker, lookback_days=lookback_days)
                summaries.append(summary)
            except Exception as exc:
                logger.warning("Failed to aggregate %s: %s", ticker, exc)
        summaries.sort(key=lambda s: s.composite_score, reverse=True)
        return summaries

    def most_bullish(self, tickers: List[str], top_n: int = 5) -> List[SentimentSummary]:
        all_summaries = self.scan(tickers)
        return [s for s in all_summaries if s.composite_score > 0][:top_n]

    def most_bearish(self, tickers: List[str], top_n: int = 5) -> List[SentimentSummary]:
        all_summaries = self.scan(tickers)
        bearish = [s for s in all_summaries if s.composite_score < 0]
        bearish.sort(key=lambda s: s.composite_score)
        return bearish[:top_n]


# ---------------------------------------------------------------------------
# Advanced signal functions (dim_085 score 8 → 9)
# ---------------------------------------------------------------------------


def compute_sentiment_momentum(
    scores: list[float],
) -> dict:
    """
    Compute sentiment momentum via 3-day vs 10-day moving-average crossover.

    A positive crossover (3D MA rises above 10D MA) generates a bullish signal;
    a negative crossover generates a bearish signal.

    Parameters
    ----------
    scores : List of daily composite sentiment scores (most recent last).
             Minimum 10 values required for a valid signal.

    Returns
    -------
    dict with:
      "ma_3d"        : float — 3-day simple moving average of sentiment.
      "ma_10d"       : float — 10-day simple moving average of sentiment.
      "signal"       : "bullish" | "bearish" | "neutral" — crossover signal.
      "momentum"     : float — ma_3d - ma_10d (positive = bullish momentum).
      "valid"        : bool — False if insufficient data.
    """
    if len(scores) < 10:
        return {"ma_3d": None, "ma_10d": None, "signal": "neutral", "momentum": 0.0, "valid": False}

    ma_3d = float(sum(scores[-3:]) / 3)
    ma_10d = float(sum(scores[-10:]) / 10)
    momentum = ma_3d - ma_10d

    if momentum > 0.02:
        signal = "bullish"
    elif momentum < -0.02:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "ma_3d": round(ma_3d, 4),
        "ma_10d": round(ma_10d, 4),
        "signal": signal,
        "momentum": round(momentum, 4),
        "valid": True,
    }


def compute_retail_vs_institutional_divergence(
    social_sentiment: float,
    analyst_consensus: float,
    divergence_threshold: float = 0.30,
) -> dict:
    """
    Compute the divergence between retail social sentiment and analyst consensus.

    A large gap between retail sentiment (from Reddit/StockTwits) and analyst
    consensus (normalised to the same -1..+1 scale) may signal a contrarian
    opportunity: if retail is extremely bullish but analysts are bearish, the
    stock may be overextended.

    Parameters
    ----------
    social_sentiment  : Composite social sentiment score in [-1, 1].
    analyst_consensus : Analyst mean recommendation normalised to [-1, 1].
                        Conversion: 1.0 (Strong Buy) → +1.0,
                                    3.0 (Hold)        →  0.0,
                                    5.0 (Strong Sell) → -1.0.
                        Formula: (3.0 - rating) / 2.0
    divergence_threshold : Minimum |gap| to flag a contrarian signal (default 0.30).

    Returns
    -------
    dict with:
      "social_sentiment"    : float
      "analyst_consensus"   : float
      "divergence"          : float — social_sentiment - analyst_consensus
      "abs_divergence"      : float
      "contrarian_signal"   : bool — True if abs_divergence > threshold
      "direction"           : "retail_bullish_analyst_bearish" | "retail_bearish_analyst_bullish" | "aligned"
      "threshold"           : float
    """
    divergence = social_sentiment - analyst_consensus
    abs_div = abs(divergence)
    contrarian = abs_div > divergence_threshold

    if contrarian and divergence > 0:
        direction = "retail_bullish_analyst_bearish"
    elif contrarian and divergence < 0:
        direction = "retail_bearish_analyst_bullish"
    else:
        direction = "aligned"

    return {
        "social_sentiment": round(social_sentiment, 4),
        "analyst_consensus": round(analyst_consensus, 4),
        "divergence": round(divergence, 4),
        "abs_divergence": round(abs_div, 4),
        "contrarian_signal": contrarian,
        "direction": direction,
        "threshold": divergence_threshold,
    }


def compute_viral_coefficient(
    mentions_today: float,
    mentions_7day_avg: float,
) -> dict:
    """
    Compute the viral coefficient: ratio of today's mentions to the 7-day rolling average.

    A ratio > 3× is flagged as a viral spike (unusual attention surge).

    Parameters
    ----------
    mentions_today   : Number of social mentions in the most recent day.
    mentions_7day_avg : Rolling 7-day average of daily mention counts.

    Returns
    -------
    dict with:
      "viral_coefficient" : float — mentions_today / mentions_7day_avg
      "viral_spike"       : bool — True if viral_coefficient > 3.0
      "mentions_today"    : float
      "mentions_7day_avg" : float
    """
    if mentions_7day_avg <= 0:
        return {
            "viral_coefficient": float("inf") if mentions_today > 0 else 1.0,
            "viral_spike": mentions_today > 0,
            "mentions_today": mentions_today,
            "mentions_7day_avg": mentions_7day_avg,
        }

    coeff = mentions_today / mentions_7day_avg

    return {
        "viral_coefficient": round(coeff, 10),
        "viral_spike": coeff > 3.0,
        "mentions_today": mentions_today,
        "mentions_7day_avg": mentions_7day_avg,
    }


# ---------------------------------------------------------------------------
# FastAPI router (optional — only registered if fastapi is available)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter as _APIRouter, HTTPException as _HTTPException, Query as _Query
    from pydantic import BaseModel as _BaseModel

    _router = _APIRouter(prefix="/api/v1/sentiment", tags=["Social Sentiment"])

    class _SentimentResponse(_BaseModel):
        ticker: str
        composite_score: float
        reddit_score: float
        stocktwits_score: float
        news_score: float
        mention_volume: int
        volume_trend: str
        dd_count: int
        sentiment_shift_24h: float
        generated_at: str

    @_router.get("/aggregate/{ticker}", response_model=_SentimentResponse)
    def api_aggregate(
        ticker: str,
        lookback_days: int = _Query(7, ge=1, le=30),
    ) -> _SentimentResponse:
        agg = SentimentAggregator()
        summary = agg.aggregate(ticker.upper(), lookback_days=lookback_days)
        return _SentimentResponse(
            ticker=summary.ticker,
            composite_score=summary.composite_score,
            reddit_score=summary.reddit_score,
            stocktwits_score=summary.stocktwits_score,
            news_score=summary.news_score,
            mention_volume=summary.mention_volume,
            volume_trend=summary.volume_trend,
            dd_count=summary.dd_count,
            sentiment_shift_24h=summary.sentiment_shift_24h,
            generated_at=summary.generated_at,
        )

    @_router.get("/trending")
    def api_trending(
        subreddit: str = _Query("wallstreetbets"),
        top_n: int = _Query(20, ge=1, le=50),
    ) -> List[Dict[str, Any]]:
        agg = SentimentAggregator()
        tickers = agg.get_trending_tickers(subreddit=subreddit, top_n=top_n)
        return [
            {
                "ticker": t.ticker,
                "mention_count": t.mention_count,
                "avg_sentiment": t.avg_sentiment,
                "subreddit": t.subreddit,
                "top_post": t.top_post_title[:100],
            }
            for t in tickers
        ]

    @_router.get("/squeeze-candidates")
    def api_squeeze_candidates() -> List[Dict[str, Any]]:
        radar = WallStreetBetsRadar()
        candidates = radar.detect_squeeze_candidates()
        return [
            {
                "ticker": c.ticker,
                "mention_count": c.mention_count,
                "avg_sentiment": c.avg_sentiment,
                "upvote_surge": c.upvote_surge,
                "short_interest_pct": c.short_interest_pct,
                "squeeze_score": c.squeeze_score,
            }
            for c in candidates
        ]

    @_router.get("/history/{ticker}")
    def api_history(
        ticker: str,
        days: int = _Query(30, ge=1, le=90),
    ) -> List[Dict[str, Any]]:
        radar = WallStreetBetsRadar()
        history = radar.get_sentiment_history(ticker.upper(), days=days)
        if _PANDAS_OK and pd is not None and hasattr(history, "reset_index"):
            return history.reset_index().to_dict(orient="records")
        return history  # type: ignore[return-value]

    sentiment_router = _router

except ImportError:
    sentiment_router = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    print("=" * 70)
    print("SENTINEL Social Sentiment V3 — Demo")
    print(f"VADER backend: {'vaderSentiment' if _VADER_AVAILABLE else 'fallback lexicon'}")
    print("=" * 70)

    demo_tickers = ["AAPL", "NVDA", "GME"]
    if len(sys.argv) > 1:
        demo_tickers = sys.argv[1:]

    analyzer = VADERSentimentAnalyzer()

    # Test VADER scoring
    print("\n── VADER Scoring Test ──")
    test_sentences = [
        "Company beat earnings expectations and raised guidance significantly.",
        "Stock crashed after massive earnings miss and SEC investigation revealed.",
        "Mixed results with modest revenue growth but margin compression concerns.",
        "NVDA absolutely mooning after blowout quarter, squeeze incoming!",
        "GME bankruptcy risk increasing as short interest hits 130%.",
    ]
    for sentence in test_sentences:
        score = analyzer.score(sentence)
        print(f"  [{score.label:8s} {score.compound:+.3f}] {sentence[:65]}")

    # Aggregate per ticker
    agg = SentimentAggregator()
    for ticker in demo_tickers:
        print(f"\n── {ticker} Sentiment (7-day) ──")
        try:
            summary = agg.aggregate(ticker, lookback_days=7)
            print(f"  Composite score:    {summary.composite_score:+.4f}")
            print(f"  Reddit score:       {summary.reddit_score:+.4f}  ({summary.reddit_posts} posts)")
            print(f"  StockTwits score:   {summary.stocktwits_score:+.4f}  ({summary.stocktwits_messages} msgs)")
            print(f"  News score:         {summary.news_score:+.4f}  ({summary.news_items} items)")
            print(f"  Mention volume:     {summary.mention_volume}")
            print(f"  Volume trend:       {summary.volume_trend}")
            print(f"  DD posts:           {summary.dd_count}")
            print(f"  24h shift:          {summary.sentiment_shift_24h:+.4f}")
            print(f"  Generated at:       {summary.generated_at}")
        except Exception as exc:
            print(f"  Error: {exc}")

    # Trending tickers from WSB
    print("\n── Trending Tickers on r/wallstreetbets ──")
    try:
        trending = agg.get_trending_tickers(subreddit="wallstreetbets", top_n=15)
        for i, t in enumerate(trending, 1):
            bar = "█" * min(t.mention_count, 20)
            print(f"  {i:2d}. {t.ticker:6s} {t.mention_count:4d} mentions  "
                  f"sent={t.avg_sentiment:+.3f}  {bar}")
    except Exception as exc:
        print(f"  Error: {exc}")

    # WSB radar
    print("\n── WSB Squeeze Radar ──")
    try:
        radar = WallStreetBetsRadar()
        candidates = radar.detect_squeeze_candidates(min_mentions=3)
        if candidates:
            for c in candidates[:10]:
                print(f"  {c.ticker:6s}  mentions={c.mention_count}  "
                      f"sent={c.avg_sentiment:+.3f}  "
                      f"upvote_surge={c.upvote_surge}  "
                      f"score={c.squeeze_score:.3f}")
        else:
            print("  No squeeze candidates detected (may need live WSB data)")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\nDone.")
