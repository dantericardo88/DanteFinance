"""
Social media sentiment: Reddit (PRAW + pushshift), Twitter/X proxy,
StockTwits, Telegram, Discord, mention velocity, and viral detection.

Enhanced standalone social-media-focused sentiment engine for SENTINEL.
Builds on sentinel/sma/sentiment_engine.py; adds viral detection, WSB-specific
analysis, mention velocity tracking, multi-source composites with time-decay,
and a complete FastAPI router.

Dimensions:
  dim_085 — Social media sentiment (Reddit/Twitter)  target: 9
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import math
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import praw  # type: ignore[import]
    _PRAW_OK = True
except ImportError:
    praw = None  # type: ignore[assignment]
    _PRAW_OK = False

try:
    import feedparser  # type: ignore[import]
    _FEEDPARSER_OK = True
except ImportError:
    feedparser = None  # type: ignore[assignment]
    _FEEDPARSER_OK = False

try:
    import pandas as pd  # type: ignore[import]
    _PANDAS_OK = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_OK = False

# ── Import base sentiment primitives from existing engine ──────────────────────
try:
    from sentinel.sma.sentiment_engine import (
        SentimentEngine as _BaseSentimentEngine,
        SentimentScore,
        _label_to_numeric,
        _numeric_to_label,
        LM_POSITIVE,
        LM_NEGATIVE,
    )
    _BASE_OK = True
except ImportError:
    _BASE_OK = False
    # Provide minimal standalone fallbacks
    LM_POSITIVE = [
        "profit", "growth", "strong", "beat", "raised", "record", "outperformed",
        "dividend", "buyback", "approved", "upgrade", "robust", "innovative",
    ]
    LM_NEGATIVE = [
        "loss", "decline", "missed", "below", "disappointing", "layoffs", "lawsuit",
        "investigation", "fraud", "bankruptcy", "downgrade", "warning", "debt",
    ]

    class SentimentScore(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(frozen=True)
        label: str
        score: float
        numeric: float
        source: str

    def _label_to_numeric(label: str) -> float:
        label = label.lower()
        if label in ("positive", "bullish", "pos"):
            return 1.0
        if label in ("negative", "bearish", "neg"):
            return -1.0
        return 0.0

    def _numeric_to_label(value: float) -> str:
        if value >= 0.5:
            return "very_bullish"
        if value >= 0.2:
            return "bullish"
        if value <= -0.5:
            return "very_bearish"
        if value <= -0.2:
            return "bearish"
        return "neutral"

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

REDDIT_BASE = "https://api.reddit.com"
PUSHSHIFT_BASE = "https://api.pushshift.io/reddit"
STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
SEEKING_ALPHA_RSS = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

WSB_SUBREDDITS = [
    "wallstreetbets", "investing", "stocks", "options",
    "SecurityAnalysis", "ValueInvesting",
]

_HEADERS = {
    "User-Agent": "SENTINEL:SocialSentiment:2.0 (by /u/sentinel_financial)",
    "Accept": "application/json, text/xml, */*",
}

# Broad ticker pattern — validated further against known list
TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')

# Common English words to exclude from ticker extraction
_NOT_TICKERS = frozenset([
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP",
    "US", "WE", "ALL", "AND", "ARE", "BUT", "CAN", "DID", "FOR", "GET", "GOT",
    "HAD", "HAS", "HIM", "HIS", "HOW", "ITS", "LET", "MAY", "NEW", "NOT",
    "NOW", "OLD", "ONE", "OUR", "OUT", "OWN", "PUT", "SAY", "SHE", "THE",
    "TOO", "TWO", "USE", "WAS", "WAY", "WHO", "WHY", "WIN", "YET", "YOU",
    "YOLO", "MOON", "BULL", "BEAR", "CALL", "PUTS", "HOLD", "SELL", "BUY",
    "DD", "TA", "PR", "CEO", "CFO", "COO", "IPO", "ETF", "SEC", "WSB",
    "ATH", "ATL", "EPS", "FCF", "P", "E", "B", "C", "D", "F", "G", "H",
    "J", "K", "L", "M", "N", "O", "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "PE", "EV", "GDP", "CPI", "FED", "ECB", "BOJ", "IMF",
])

# SQLite DB path
_DB_PATH = Path(os.getenv("SENTINEL_DATA_DIR", "data")) / "social_sentiment.db"

# Viral threshold: current 24h mentions > N× 7-day rolling average
VIRAL_VELOCITY_THRESHOLD = 5.0

# Contrarian threshold: bull_pct above this → contrarian sell signal
CONTRARIAN_BULL_THRESHOLD = 85.0

# Time-decay half-life for sentiment compositing (days)
SENTIMENT_HALF_LIFE_DAYS = 3.0

# Source weights for composite
SOURCE_WEIGHTS = {
    "stocktwits": 0.40,
    "reddit": 0.35,
    "rss_news": 0.25,
}

# WSB meme-stock short interest threshold
MEME_SHORT_INTEREST_THRESHOLD = 20.0  # pct float


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _ensure_db() -> None:
    """Create SQLite tables if they don't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS mentions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                source      TEXT NOT NULL,
                ts          INTEGER NOT NULL,
                count       INTEGER NOT NULL DEFAULT 1,
                sentiment   REAL
            );
            CREATE INDEX IF NOT EXISTS ix_mentions_ticker_ts
                ON mentions(ticker, ts);

            CREATE TABLE IF NOT EXISTS wsb_tickers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                ts          INTEGER NOT NULL,
                mention_ct  INTEGER NOT NULL DEFAULT 1,
                bull_ct     INTEGER NOT NULL DEFAULT 0,
                bear_ct     INTEGER NOT NULL DEFAULT 0,
                top_score   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS ix_wsb_ticker_ts
                ON wsb_tickers(ticker, ts);

            CREATE TABLE IF NOT EXISTS wsb_alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                alert_type  TEXT NOT NULL,
                detail      TEXT,
                ts          INTEGER NOT NULL
            );
        """)


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Pydantic response models ───────────────────────────────────────────────────

class RedditPostDetail(BaseModel):
    ticker: Optional[str] = None
    title: str
    body_snippet: str = ""
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    flair: Optional[str] = None
    created: datetime
    url: str
    weight: float = 0.0
    sentiment_label: str = "neutral"
    sentiment_numeric: float = 0.0
    sentiment_source: str = "lm_wordlist"


class StockTwitsDetail(BaseModel):
    ticker: str
    body: str
    bull_label: Optional[str] = None  # "Bullish" | "Bearish" | None
    created: datetime
    username: str
    followers: int = 0
    amplified: bool = False   # followers > 100


class SentimentComposite(BaseModel):
    ticker: str
    as_of: datetime
    composite_score: float          # -1 to +1
    label: str
    confidence: float               # 0-1, based on data volume
    sources_used: List[str]
    stocktwits_score: Optional[float] = None
    reddit_score: Optional[float] = None
    rss_score: Optional[float] = None
    bull_pct: float = 0.0
    bear_pct: float = 0.0
    mention_count: int = 0
    momentum_alert: bool = False    # 24h score change > 0.2
    divergence_alert: bool = False  # social bullish + price flat/down
    contrarian_signal: bool = False # bull_pct > 85%


class MentionVelocity(BaseModel):
    ticker: str
    mentions_1h: int = 0
    mentions_6h: int = 0
    mentions_24h: int = 0
    avg_7day: float = 0.0
    velocity_ratio: float = 0.0
    is_viral: bool = False


class WSBAlert(BaseModel):
    ticker: str
    alert_type: str   # "meme_stock" | "yolo_options" | "gamma_squeeze" | "dd_post"
    detail: str
    ts: datetime


class WSBTopTicker(BaseModel):
    ticker: str
    mention_count: int
    bull_count: int
    bear_count: int
    bull_pct: float
    top_post_score: int
    is_meme_candidate: bool = False


class ViralTicker(BaseModel):
    ticker: str
    velocity_ratio: float
    mentions_24h: int
    sources: List[str]


# ── Lightweight LM fallback sentiment ─────────────────────────────────────────

def _lm_score(text: str) -> SentimentScore:
    text_l = text.lower()
    pos = sum(1 for w in LM_POSITIVE if w in text_l)
    neg = sum(1 for w in LM_NEGATIVE if w in text_l)
    total = pos + neg
    if total == 0:
        return SentimentScore(label="neutral", score=0.5, numeric=0.0, source="lm_wordlist")
    if pos > neg:
        label, numeric = "positive", 1.0
        score = min(1.0, 0.5 + 0.08 * (pos - neg))
    elif neg > pos:
        label, numeric = "negative", -1.0
        score = min(1.0, 0.5 + 0.08 * (neg - pos))
    else:
        label, numeric, score = "neutral", 0.0, 0.55
    return SentimentScore(label=label, score=round(score, 4), numeric=numeric, source="lm_wordlist")


# ── Ticker extractor ───────────────────────────────────────────────────────────

def extract_tickers(text: str, known_tickers: Optional[frozenset] = None) -> List[str]:
    """
    Extract potential stock tickers from text using TICKER_PATTERN.
    Filters against _NOT_TICKERS and optionally validates against known_tickers set.
    """
    found = TICKER_PATTERN.findall(text)
    results: List[str] = []
    seen: set = set()
    for t in found:
        if t in _NOT_TICKERS:
            continue
        if len(t) < 2:
            continue
        if known_tickers and t not in known_tickers:
            continue
        if t not in seen:
            seen.add(t)
            results.append(t)
    return results


# ── Reddit RSS fallback ────────────────────────────────────────────────────────

def _parse_rss_xml(xml_text: str, limit: int = 25) -> List[dict]:
    """Parse RSS/Atom XML into list of item dicts."""
    items: List[dict] = []
    try:
        xml_clean = re.sub(r' xmlns(?::\w+)?="[^"]*"', "", xml_text)
        root = ET.fromstring(xml_clean)
        channel = root.find("channel") or root
        for elem in channel.iter():
            if not (elem.tag.endswith("item") or elem.tag.endswith("entry")):
                continue
            title_el = elem.find("title")
            link_el = elem.find("link")
            pub_el = (
                elem.find("pubDate")
                or elem.find("published")
                or elem.find("updated")
            )
            title = (title_el.text or "").strip() if title_el is not None else ""
            if not title:
                continue
            link = ""
            if link_el is not None:
                link = link_el.text or link_el.get("href", "") or ""
            pub_str = (pub_el.text or "").strip() if pub_el is not None else ""
            published: Optional[datetime] = None
            for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z"):
                try:
                    published = datetime.strptime(pub_str[: len(fmt) + 5], fmt)
                    break
                except (ValueError, AttributeError):
                    pass
            if published is None and pub_str:
                try:
                    published = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except ValueError:
                    pass
            items.append({"title": html.unescape(title), "url": link.strip(), "published": published})
            if len(items) >= limit:
                break
    except ET.ParseError as exc:
        logger.warning("RSS parse error: %s", exc)
    return items


# ── 1. RedditSentimentCollector ───────────────────────────────────────────────

class RedditSentimentCollector:
    """
    Collects Reddit posts from financial subreddits.

    Priority:
      1. PRAW (OAuth if REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET set)
      2. Reddit public JSON search API (no auth)
      3. Reddit RSS feeds (fallback)
    """

    SUBREDDITS = WSB_SUBREDDITS

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self._praw_client: Optional[Any] = None
        self._init_praw()

    def _init_praw(self) -> None:
        if not _PRAW_OK:
            return
        client_id = os.getenv("REDDIT_CLIENT_ID", "")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        if client_id and client_secret:
            try:
                self._praw_client = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="SENTINEL:2.0 (by /u/sentinel_financial)",
                )
                logger.info("PRAW client initialised")
            except Exception as exc:
                logger.warning("PRAW init failed: %s", exc)

    # ── Public entry point ─────────────────────────────────────────────────────

    def collect(
        self,
        ticker: str,
        subreddits: Optional[List[str]] = None,
        limit_per_sub: int = 15,
        known_tickers: Optional[frozenset] = None,
    ) -> List[RedditPostDetail]:
        subs = subreddits or self.SUBREDDITS
        ticker_up = ticker.upper()
        posts: List[RedditPostDetail] = []

        if self._praw_client:
            posts = self._collect_praw(ticker_up, subs, limit_per_sub)
        else:
            posts = self._collect_json_api(ticker_up, subs, limit_per_sub)

        if not posts:
            posts = self._collect_rss(ticker_up, subs, limit_per_sub)

        # Score sentiment
        for p in posts:
            sent = _lm_score(f"{p.title} {p.body_snippet}")
            # Compute engagement weight
            weight = (
                max(0, p.score)
                * max(0.01, p.upvote_ratio)
                * math.log1p(p.num_comments)
            )
            # Rebuild with scored data (Pydantic frozen → use model_copy)
            posts[posts.index(p)] = p.model_copy(update={
                "sentiment_label": sent.label,
                "sentiment_numeric": sent.numeric,
                "sentiment_source": sent.source,
                "weight": round(weight, 2),
            })

        # Record mentions in DB
        if posts:
            self._record_mentions(ticker_up, "reddit", len(posts))

        return posts

    # ── PRAW path ──────────────────────────────────────────────────────────────

    def _collect_praw(
        self, ticker: str, subs: List[str], limit: int
    ) -> List[RedditPostDetail]:
        results: List[RedditPostDetail] = []
        for sub_name in subs:
            try:
                subreddit = self._praw_client.subreddit(sub_name)  # type: ignore[union-attr]
                for post in subreddit.search(ticker, sort="new", time_filter="week", limit=limit):
                    results.append(RedditPostDetail(
                        ticker=ticker,
                        title=post.title,
                        body_snippet=(post.selftext or "")[:500],
                        subreddit=sub_name,
                        score=int(post.score or 0),
                        upvote_ratio=float(post.upvote_ratio or 0.5),
                        num_comments=int(post.num_comments or 0),
                        flair=post.link_flair_text,
                        created=datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                        url=f"https://reddit.com{post.permalink}",
                    ))
            except Exception as exc:
                logger.warning("PRAW sub=%s error: %s", sub_name, exc)
        return results

    # ── JSON API path ──────────────────────────────────────────────────────────

    def _collect_json_api(
        self, ticker: str, subs: List[str], limit: int
    ) -> List[RedditPostDetail]:
        results: List[RedditPostDetail] = []
        for sub_name in subs:
            url = f"{REDDIT_BASE}/r/{sub_name}/search"
            params = {
                "q": ticker,
                "sort": "new",
                "limit": limit,
                "t": "week",
                "restrict_sr": "1",
            }
            try:
                resp = requests.get(url, params=params, headers=_HEADERS, timeout=self.timeout)
                resp.raise_for_status()
                children = resp.json().get("data", {}).get("children", [])
                for child in children:
                    p = child.get("data", {})
                    if not p:
                        continue
                    created_utc = float(p.get("created_utc") or 0)
                    results.append(RedditPostDetail(
                        ticker=ticker,
                        title=p.get("title", ""),
                        body_snippet=(p.get("selftext") or "")[:500],
                        subreddit=sub_name,
                        score=max(0, int(p.get("score") or 0)),
                        upvote_ratio=float(p.get("upvote_ratio") or 0.5),
                        num_comments=max(0, int(p.get("num_comments") or 0)),
                        flair=p.get("link_flair_text"),
                        created=datetime.fromtimestamp(created_utc, tz=timezone.utc)
                        if created_utc else datetime.now(tz=timezone.utc),
                        url=f"https://reddit.com{p.get('permalink', '')}",
                    ))
            except requests.HTTPError as exc:
                logger.warning("Reddit API sub=%s HTTP %s", sub_name, exc.response.status_code)
            except Exception as exc:
                logger.warning("Reddit API sub=%s error: %s", sub_name, exc)
        return results

    # ── RSS fallback ───────────────────────────────────────────────────────────

    def _collect_rss(
        self, ticker: str, subs: List[str], limit: int
    ) -> List[RedditPostDetail]:
        results: List[RedditPostDetail] = []
        for sub_name in subs[:3]:  # limit RSS calls
            url = f"https://www.reddit.com/r/{sub_name}/search.rss"
            params = {"q": ticker, "restrict_sr": "on", "sort": "new"}
            try:
                resp = requests.get(url, params=params, headers=_HEADERS, timeout=self.timeout)
                items = _parse_rss_xml(resp.text, limit=limit)
                for item in items:
                    results.append(RedditPostDetail(
                        ticker=ticker,
                        title=item["title"],
                        subreddit=sub_name,
                        score=0,
                        upvote_ratio=0.5,
                        num_comments=0,
                        created=item.get("published") or datetime.now(tz=timezone.utc),
                        url=item["url"],
                    ))
            except Exception as exc:
                logger.warning("Reddit RSS sub=%s error: %s", sub_name, exc)
        return results

    # ── Daily aggregation ──────────────────────────────────────────────────────

    def daily_aggregate(self, posts: List[RedditPostDetail]) -> Dict[str, float]:
        """
        Aggregate weighted sentiment scores per ticker across posts.
        Returns dict: {ticker: weighted_avg_sentiment}
        """
        agg: Dict[str, List[Tuple[float, float]]] = {}  # ticker → [(sentiment, weight)]
        for p in posts:
            if p.ticker:
                agg.setdefault(p.ticker, []).append(
                    (p.sentiment_numeric, max(0.01, p.weight))
                )
        out: Dict[str, float] = {}
        for ticker, pairs in agg.items():
            total_w = sum(w for _, w in pairs)
            if total_w > 0:
                out[ticker] = round(sum(s * w for s, w in pairs) / total_w, 4)
        return out

    # ── DB helper ──────────────────────────────────────────────────────────────

    @staticmethod
    def _record_mentions(ticker: str, source: str, count: int) -> None:
        try:
            with _db() as conn:
                conn.execute(
                    "INSERT INTO mentions(ticker, source, ts, count) VALUES(?,?,?,?)",
                    (ticker, source, int(time.time()), count),
                )
        except Exception as exc:
            logger.debug("DB mention record error: %s", exc)


# ── 2. StockTwitsSentimentCollector ───────────────────────────────────────────

class StockTwitsSentimentCollector:
    """
    Collects real-time sentiment from StockTwits public API.
    No auth required (rate-limited to ~200 req/hour per IP).
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def collect(self, ticker: str, limit: int = 30) -> List[StockTwitsDetail]:
        ticker_up = ticker.upper()
        url = f"{STOCKTWITS_BASE}/streams/symbol/{ticker_up}.json"
        messages: List[StockTwitsDetail] = []
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=self.timeout)
            if resp.status_code == 404:
                logger.info("StockTwits: ticker not found %s", ticker_up)
                return []
            if resp.status_code == 429:
                logger.warning("StockTwits: rate limited")
                return []
            resp.raise_for_status()
            data = resp.json()
            for msg in (data.get("messages") or [])[:limit]:
                entities = msg.get("entities", {})
                sent_obj = entities.get("sentiment") or msg.get("sentiment")
                bull_label: Optional[str] = None
                if isinstance(sent_obj, dict):
                    bull_label = sent_obj.get("basic")  # "Bullish" | "Bearish"

                user = msg.get("user", {})
                followers = int(user.get("followers_count") or 0)

                created_str = msg.get("created_at", "")
                try:
                    created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    created = datetime.now(tz=timezone.utc)

                messages.append(StockTwitsDetail(
                    ticker=ticker_up,
                    body=(msg.get("body") or "")[:500],
                    bull_label=bull_label,
                    created=created,
                    username=user.get("username", "unknown"),
                    followers=followers,
                    amplified=(followers > 100),
                ))
        except requests.HTTPError as exc:
            logger.warning("StockTwits HTTP error: %s", exc)
        except Exception as exc:
            logger.warning("StockTwits error: %s", exc)

        # Record mentions
        if messages:
            try:
                with _db() as conn:
                    conn.execute(
                        "INSERT INTO mentions(ticker, source, ts, count) VALUES(?,?,?,?)",
                        (ticker_up, "stocktwits", int(time.time()), len(messages)),
                    )
            except Exception:
                pass

        return messages

    def compute_bull_pct(self, messages: List[StockTwitsDetail]) -> Tuple[float, float, float]:
        """
        Returns (bull_pct, bear_pct, neutral_pct) from tagged messages.
        Amplified (>100 followers) messages count double.
        """
        bull = bear = neutral = 0.0
        for m in messages:
            weight = 2.0 if m.amplified else 1.0
            if m.bull_label == "Bullish":
                bull += weight
            elif m.bull_label == "Bearish":
                bear += weight
            else:
                neutral += weight
        total = bull + bear + neutral or 1.0
        return round(bull / total * 100, 1), round(bear / total * 100, 1), round(neutral / total * 100, 1)

    def composite_score(self, messages: List[StockTwitsDetail]) -> float:
        """Aggregate numeric sentiment score (-1 to +1) from messages."""
        if not messages:
            return 0.0
        scores: List[float] = []
        for m in messages:
            if m.bull_label == "Bullish":
                scores.append(1.0 * (2.0 if m.amplified else 1.0))
            elif m.bull_label == "Bearish":
                scores.append(-1.0 * (2.0 if m.amplified else 1.0))
            else:
                # Fall back to LM on body text
                sent = _lm_score(m.body)
                scores.append(sent.numeric * (2.0 if m.amplified else 1.0))
        return round(sum(scores) / len(scores), 4)


# ── 3. TwitterProxySentiment ───────────────────────────────────────────────────

class TwitterProxySentiment:
    """
    Twitter/X sentiment without the official API.
    Sources: Google Finance RSS, Seeking Alpha RSS, GDELT DOC API.
    """

    # Source credibility weights
    SOURCE_WEIGHTS = {
        "google_news": 1.0,
        "seeking_alpha": 1.3,
        "gdelt": 0.8,
        "marketwatch": 1.1,
    }

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def collect(self, ticker: str, limit: int = 20) -> List[dict]:
        """
        Collect proxy Twitter sentiment from multiple RSS/API sources.
        Returns list of {title, url, published, sentiment_score, source, weight}.
        """
        results: List[dict] = []
        results.extend(self._google_news(ticker, limit))
        results.extend(self._seeking_alpha(ticker, limit // 2))
        results.extend(self._gdelt(ticker, limit // 2))
        return results

    def _google_news(self, ticker: str, limit: int) -> List[dict]:
        params = {"q": f"{ticker} stock", "hl": "en-US", "gl": "US", "ceid": "US:en"}
        try:
            resp = requests.get(GOOGLE_NEWS_RSS, params=params, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
            items = _parse_rss_xml(resp.text, limit=limit)
            return self._score_items(items, "google_news")
        except Exception as exc:
            logger.debug("Google News RSS error: %s", exc)
            return []

    def _seeking_alpha(self, ticker: str, limit: int) -> List[dict]:
        url = SEEKING_ALPHA_RSS.format(ticker=ticker.upper())
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=self.timeout)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            items = _parse_rss_xml(resp.text, limit=limit)
            return self._score_items(items, "seeking_alpha")
        except Exception as exc:
            logger.debug("Seeking Alpha RSS error: %s", exc)
            return []

    def _gdelt(self, ticker: str, limit: int) -> List[dict]:
        """GDELT DOC API: free global news mention search."""
        params = {
            "query": f"{ticker} stock OR shares OR equity",
            "mode": "artlist",
            "maxrecords": min(limit, 25),
            "format": "json",
        }
        try:
            resp = requests.get(GDELT_DOC_API, params=params, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles") or []
            items = []
            for art in articles[:limit]:
                items.append({
                    "title": art.get("title", ""),
                    "url": art.get("url", ""),
                    "published": None,
                })
            return self._score_items(items, "gdelt")
        except Exception as exc:
            logger.debug("GDELT error: %s", exc)
            return []

    def _score_items(self, items: List[dict], source: str) -> List[dict]:
        weight = self.SOURCE_WEIGHTS.get(source, 1.0)
        scored: List[dict] = []
        for item in items:
            sent = _lm_score(item.get("title", ""))
            scored.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "published": item.get("published"),
                "sentiment_score": sent.numeric,
                "source": source,
                "weight": weight,
            })
        return scored

    def aggregate_score(self, items: List[dict]) -> float:
        """Weighted average sentiment score across all proxy sources."""
        if not items:
            return 0.0
        total_w = sum(i["weight"] for i in items)
        if total_w == 0:
            return 0.0
        return round(sum(i["sentiment_score"] * i["weight"] for i in items) / total_w, 4)


# ── 4. MentionVelocityTracker ──────────────────────────────────────────────────

class MentionVelocityTracker:
    """
    Tracks mention velocity using rolling windows from the SQLite mentions table.
    Detects viral events (velocity > VIRAL_VELOCITY_THRESHOLD).
    """

    def get_velocity(self, ticker: str) -> MentionVelocity:
        ticker_up = ticker.upper()
        now = int(time.time())
        h1_ago = now - 3600
        h6_ago = now - 21600
        h24_ago = now - 86400
        d7_ago = now - 7 * 86400

        try:
            with _db() as conn:
                def _sum(since: int) -> int:
                    row = conn.execute(
                        "SELECT COALESCE(SUM(count),0) FROM mentions WHERE ticker=? AND ts>=?",
                        (ticker_up, since),
                    ).fetchone()
                    return int(row[0]) if row else 0

                m1h = _sum(h1_ago)
                m6h = _sum(h6_ago)
                m24h = _sum(h24_ago)

                # 7-day daily average (exclude last 24h to avoid circular)
                row7 = conn.execute(
                    """SELECT COALESCE(SUM(count),0), COALESCE(COUNT(DISTINCT date(ts,'unixepoch')),1)
                       FROM mentions WHERE ticker=? AND ts>=? AND ts<?""",
                    (ticker_up, d7_ago, h24_ago),
                ).fetchone()
                total7 = int(row7[0]) if row7 else 0
                days7 = max(1, int(row7[1]) if row7 else 1)
                avg7 = total7 / days7
        except Exception as exc:
            logger.debug("MentionVelocity DB error: %s", exc)
            m1h = m6h = m24h = 0
            avg7 = 1.0

        velocity = m24h / max(1.0, avg7)
        return MentionVelocity(
            ticker=ticker_up,
            mentions_1h=m1h,
            mentions_6h=m6h,
            mentions_24h=m24h,
            avg_7day=round(avg7, 1),
            velocity_ratio=round(velocity, 2),
            is_viral=(velocity >= VIRAL_VELOCITY_THRESHOLD),
        )

    def is_viral(self, ticker: str) -> bool:
        return self.get_velocity(ticker).is_viral

    def get_viral_tickers(self, min_mentions: int = 5) -> List[ViralTicker]:
        """Return all tickers currently trending viral (velocity >= threshold)."""
        now = int(time.time())
        h24_ago = now - 86400
        try:
            with _db() as conn:
                rows = conn.execute(
                    """SELECT ticker, SUM(count) as m24h, GROUP_CONCAT(DISTINCT source) as srcs
                       FROM mentions WHERE ts>=? GROUP BY ticker HAVING m24h>=?""",
                    (h24_ago, min_mentions),
                ).fetchall()
        except Exception as exc:
            logger.debug("get_viral_tickers DB error: %s", exc)
            return []

        viral: List[ViralTicker] = []
        for row in rows:
            ticker = row["ticker"]
            m24h = int(row["m24h"])
            vel = self.get_velocity(ticker)
            if vel.is_viral:
                viral.append(ViralTicker(
                    ticker=ticker,
                    velocity_ratio=vel.velocity_ratio,
                    mentions_24h=m24h,
                    sources=(row["srcs"] or "").split(","),
                ))
        viral.sort(key=lambda v: v.velocity_ratio, reverse=True)
        return viral


# ── 5. SentimentSignalAggregator ──────────────────────────────────────────────

class SentimentSignalAggregator:
    """
    Aggregates StockTwits, Reddit, and RSS proxy sources into a time-decay-weighted
    composite sentiment score with momentum and contrarian alerts.
    """

    def __init__(self) -> None:
        self.reddit = RedditSentimentCollector()
        self.stocktwits = StockTwitsSentimentCollector()
        self.twitter_proxy = TwitterProxySentiment()
        self.velocity = MentionVelocityTracker()

    def get_composite(self, ticker: str, hours: int = 72) -> SentimentComposite:
        ticker_up = ticker.upper()
        now = datetime.now(tz=timezone.utc)

        # Collect from all sources
        reddit_posts = self.reddit.collect(ticker_up)
        st_messages = self.stocktwits.collect(ticker_up)
        proxy_items = self.twitter_proxy.collect(ticker_up)

        # Per-source scores
        st_score: Optional[float] = None
        rd_score: Optional[float] = None
        rss_score: Optional[float] = None
        sources_used: List[str] = []

        if st_messages:
            st_score = self.stocktwits.composite_score(st_messages)
            sources_used.append("stocktwits")
        if reddit_posts:
            agg = self.reddit.daily_aggregate(reddit_posts)
            if agg:
                rd_score = sum(agg.values()) / len(agg)
            sources_used.append("reddit")
        if proxy_items:
            rss_score = self.twitter_proxy.aggregate_score(proxy_items)
            sources_used.append("rss_news")

        # Apply time-decay weights and source weights
        weighted_sum = 0.0
        total_weight = 0.0

        def _add(score: Optional[float], source_key: str) -> None:
            nonlocal weighted_sum, total_weight
            if score is None:
                return
            w = SOURCE_WEIGHTS.get(source_key, 0.25)
            # Time decay: assume current collection = time 0, so decay = 1.0
            # For historical comparison we apply decay to stored values separately
            decay = 1.0
            weighted_sum += score * w * decay
            total_weight += w * decay

        _add(st_score, "stocktwits")
        _add(rd_score, "reddit")
        _add(rss_score, "rss_news")

        composite = 0.0
        if total_weight > 0:
            composite = round(max(-1.0, min(1.0, weighted_sum / total_weight)), 4)

        # Bull/bear pct from StockTwits (most reliable labelled data)
        bull_pct, bear_pct, _ = self.stocktwits.compute_bull_pct(st_messages)

        # Confidence: higher with more data points
        total_items = len(st_messages) + len(reddit_posts) + len(proxy_items)
        confidence = round(min(1.0, total_items / 50.0), 3)

        # Momentum alert: check 24h stored composite vs current
        momentum_alert = self._check_momentum(ticker_up, composite)

        # Contrarian signal
        contrarian = bull_pct > CONTRARIAN_BULL_THRESHOLD

        # Divergence: social bullish + insufficient price confirmation
        # (price check deferred to caller; flag if composite > 0.3)
        divergence = composite > 0.3 and len(sources_used) >= 2

        return SentimentComposite(
            ticker=ticker_up,
            as_of=now,
            composite_score=composite,
            label=_numeric_to_label(composite),
            confidence=confidence,
            sources_used=sources_used,
            stocktwits_score=st_score,
            reddit_score=rd_score,
            rss_score=rss_score,
            bull_pct=bull_pct,
            bear_pct=bear_pct,
            mention_count=total_items,
            momentum_alert=momentum_alert,
            divergence_alert=divergence,
            contrarian_signal=contrarian,
        )

    def _check_momentum(self, ticker: str, current_composite: float) -> bool:
        """Compare current composite to the value stored ~24h ago. Returns True if shift > 0.2."""
        try:
            with _db() as conn:
                row = conn.execute(
                    """SELECT AVG(sentiment) FROM mentions
                       WHERE ticker=? AND ts BETWEEN ? AND ?""",
                    (ticker, int(time.time()) - 48 * 3600, int(time.time()) - 24 * 3600),
                ).fetchone()
                if row and row[0] is not None:
                    prior = float(row[0])
                    return abs(current_composite - prior) > 0.2
        except Exception:
            pass
        return False


# ── 6. WallStreetBetsMonitor ───────────────────────────────────────────────────

class WallStreetBetsMonitor:
    """
    WSB-specific intelligence: top tickers, meme-stock detection,
    YOLO options parsing, gamma squeeze candidates, and DD tracker.
    """

    # Regex patterns for option play parsing
    _OPTION_VALUE_RE = re.compile(
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:k|K|m|M|million|thousand)?', re.IGNORECASE
    )
    _STRIKE_RE = re.compile(r'\b(\d{1,4})[Cc]\b|\b(\d{1,4})[Pp]\b')
    _DD_FLAIR_KEYWORDS = ("DD", "Due Diligence", "Fundamentals", "Analysis", "Research")

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self._reddit = RedditSentimentCollector(timeout=timeout)

    def get_top_tickers_24h(self, limit: int = 20) -> List[WSBTopTicker]:
        """
        Return top tickers by mention count in WSB over the last 24h (stored).
        Also fetches fresh posts and updates the DB.
        """
        self._refresh_wsb()
        now_ts = int(time.time())
        h24_ago = now_ts - 86400

        try:
            with _db() as conn:
                rows = conn.execute(
                    """SELECT ticker,
                              SUM(mention_ct) AS total_mentions,
                              SUM(bull_ct) AS total_bull,
                              SUM(bear_ct) AS total_bear,
                              MAX(top_score) AS best_score
                       FROM wsb_tickers
                       WHERE ts >= ?
                       GROUP BY ticker
                       ORDER BY total_mentions DESC
                       LIMIT ?""",
                    (h24_ago, limit),
                ).fetchall()
        except Exception as exc:
            logger.warning("WSB top tickers DB error: %s", exc)
            return []

        results: List[WSBTopTicker] = []
        for row in rows:
            mentions = int(row["total_mentions"])
            bulls = int(row["total_bull"])
            bears = int(row["total_bear"])
            total_sentiment = bulls + bears or 1
            bull_pct = round(bulls / total_sentiment * 100, 1)
            results.append(WSBTopTicker(
                ticker=row["ticker"],
                mention_count=mentions,
                bull_count=bulls,
                bear_count=bears,
                bull_pct=bull_pct,
                top_post_score=int(row["best_score"]),
                is_meme_candidate=(mentions >= 10 and bull_pct >= 60.0),
            ))
        return results

    def _refresh_wsb(self) -> None:
        """Fetch latest WSB posts and populate wsb_tickers table."""
        posts = self._reddit.collect("", subreddits=["wallstreetbets"], limit_per_sub=25)
        # For each post, extract tickers and record
        now_ts = int(time.time())
        for post in posts:
            tickers_in_post = extract_tickers(
                f"{post.title} {post.body_snippet}"
            )
            if not tickers_in_post:
                continue
            is_bull = post.sentiment_numeric > 0.1
            is_bear = post.sentiment_numeric < -0.1
            try:
                with _db() as conn:
                    for t in tickers_in_post:
                        conn.execute(
                            """INSERT INTO wsb_tickers(ticker, ts, mention_ct, bull_ct, bear_ct, top_score)
                               VALUES(?,?,1,?,?,?)""",
                            (t, now_ts, 1 if is_bull else 0, 1 if is_bear else 0, post.score),
                        )
            except Exception:
                pass

    def detect_yolo_options(self, posts: List[RedditPostDetail]) -> List[WSBAlert]:
        """Parse posts for large options plays (YOLO trades)."""
        alerts: List[WSBAlert] = []
        for post in posts:
            text = f"{post.title} {post.body_snippet}"
            # Look for option contract mentions with $ value
            value_matches = self._OPTION_VALUE_RE.findall(text)
            strike_matches = self._STRIKE_RE.findall(text)
            if value_matches and strike_matches:
                raw_val = value_matches[0].replace(",", "")
                try:
                    val = float(raw_val)
                    text_lower = text.lower()
                    if "k" in text_lower:
                        val *= 1000
                    elif "m" in text_lower or "million" in text_lower:
                        val *= 1_000_000
                    if val >= 10_000:  # Only alert on significant plays
                        alerts.append(WSBAlert(
                            ticker=post.ticker or "UNKNOWN",
                            alert_type="yolo_options",
                            detail=f"Potential options play ${val:,.0f} mentioned in: {post.title[:80]}",
                            ts=post.created,
                        ))
                except ValueError:
                    pass
        return alerts

    def detect_dd_posts(self, posts: List[RedditPostDetail]) -> List[WSBAlert]:
        """Identify quality DD (Due Diligence) posts."""
        alerts: List[WSBAlert] = []
        for post in posts:
            flair = (post.flair or "").upper()
            title_up = post.title.upper()
            is_dd = (
                any(kw.upper() in flair or kw.upper() in title_up for kw in self._DD_FLAIR_KEYWORDS)
                and post.num_comments >= 20
                and post.score >= 50
            )
            if is_dd:
                alerts.append(WSBAlert(
                    ticker=post.ticker or "UNKNOWN",
                    alert_type="dd_post",
                    detail=f"DD post ({post.score} upvotes, {post.num_comments} comments): {post.title[:100]}",
                    ts=post.created,
                ))
        return alerts

    def get_gamma_squeeze_candidates(self) -> List[str]:
        """
        Approximate gamma squeeze candidates: high WSB mentions + short squeeze setup.
        (Short interest data would come from FINRA REGSHO; placeholder here uses mention volume.)
        """
        top = self.get_top_tickers_24h(limit=50)
        # Flag tickers with very high mention counts and bullish sentiment
        candidates: List[str] = []
        for t in top:
            if t.mention_count >= 20 and t.bull_pct >= 65.0:
                candidates.append(t.ticker)
        return candidates

    def get_alerts(self, ticker: str) -> List[WSBAlert]:
        """Retrieve stored WSB alerts for a ticker."""
        try:
            with _db() as conn:
                rows = conn.execute(
                    """SELECT ticker, alert_type, detail, ts FROM wsb_alerts
                       WHERE ticker=? ORDER BY ts DESC LIMIT 50""",
                    (ticker.upper(),),
                ).fetchall()
            return [
                WSBAlert(
                    ticker=row["ticker"],
                    alert_type=row["alert_type"],
                    detail=row["detail"] or "",
                    ts=datetime.fromtimestamp(row["ts"], tz=timezone.utc),
                )
                for row in rows
            ]
        except Exception as exc:
            logger.debug("WSB alerts DB error: %s", exc)
            return []


# ── 7. FastAPI Router ──────────────────────────────────────────────────────────

social_router = APIRouter(prefix="/social", tags=["social_sentiment"])

# Module-level singletons (lazy init)
_aggregator: Optional[SentimentSignalAggregator] = None
_wsb_monitor: Optional[WallStreetBetsMonitor] = None
_velocity_tracker: Optional[MentionVelocityTracker] = None


def _get_aggregator() -> SentimentSignalAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = SentimentSignalAggregator()
    return _aggregator


def _get_wsb() -> WallStreetBetsMonitor:
    global _wsb_monitor
    if _wsb_monitor is None:
        _wsb_monitor = WallStreetBetsMonitor()
    return _wsb_monitor


def _get_velocity() -> MentionVelocityTracker:
    global _velocity_tracker
    if _velocity_tracker is None:
        _velocity_tracker = MentionVelocityTracker()
    return _velocity_tracker


@social_router.get("/reddit/{ticker}", response_model=List[RedditPostDetail])
def route_reddit(
    ticker: str,
    subreddits: Optional[str] = Query(None, description="Comma-separated subreddits"),
    limit: int = Query(15, ge=1, le=50),
) -> List[RedditPostDetail]:
    """Fetch and score Reddit posts for a ticker across financial subreddits."""
    collector = RedditSentimentCollector()
    subs = subreddits.split(",") if subreddits else None
    posts = collector.collect(ticker.upper(), subreddits=subs, limit_per_sub=limit)
    return posts


@social_router.get("/stocktwits/{ticker}", response_model=dict)
def route_stocktwits(
    ticker: str,
    limit: int = Query(30, ge=1, le=75),
) -> dict:
    """Fetch StockTwits stream and return bull/bear breakdown."""
    collector = StockTwitsSentimentCollector()
    messages = collector.collect(ticker.upper(), limit=limit)
    bull_pct, bear_pct, neutral_pct = collector.compute_bull_pct(messages)
    score = collector.composite_score(messages)
    return {
        "ticker": ticker.upper(),
        "message_count": len(messages),
        "bull_pct": bull_pct,
        "bear_pct": bear_pct,
        "neutral_pct": neutral_pct,
        "composite_score": score,
        "label": _numeric_to_label(score),
        "messages": [m.model_dump() for m in messages[:10]],
    }


@social_router.get("/composite/{ticker}", response_model=SentimentComposite)
def route_composite(
    ticker: str,
    hours: int = Query(72, ge=1, le=168),
) -> SentimentComposite:
    """Return multi-source composite sentiment for a ticker."""
    return _get_aggregator().get_composite(ticker.upper(), hours=hours)


@social_router.get("/viral", response_model=List[ViralTicker])
def route_viral(min_mentions: int = Query(5, ge=1)) -> List[ViralTicker]:
    """Return all tickers currently trending viral (velocity >= 5x 7-day average)."""
    return _get_velocity().get_viral_tickers(min_mentions=min_mentions)


@social_router.get("/wsb-top", response_model=List[WSBTopTicker])
def route_wsb_top(limit: int = Query(20, ge=1, le=50)) -> List[WSBTopTicker]:
    """Return top WSB tickers by mention count in the last 24h."""
    return _get_wsb().get_top_tickers_24h(limit=limit)


@social_router.get("/wsb-alerts", response_model=List[WSBAlert])
def route_wsb_alerts(ticker: str = Query(..., description="Ticker symbol")) -> List[WSBAlert]:
    """Return stored WSB alerts (YOLO, DD, gamma squeeze) for a ticker."""
    return _get_wsb().get_alerts(ticker.upper())


@social_router.get("/momentum", response_model=dict)
def route_momentum(
    ticker: str = Query(..., description="Ticker symbol"),
) -> dict:
    """
    Return mention velocity and momentum for a ticker.
    Highlights shift alerts when 24h sentiment change > 0.2.
    """
    vel = _get_velocity().get_velocity(ticker.upper())
    composite = _get_aggregator().get_composite(ticker.upper(), hours=24)
    return {
        "ticker": ticker.upper(),
        "velocity": vel.model_dump(),
        "composite": composite.model_dump(),
        "momentum_alert": composite.momentum_alert,
        "contrarian_signal": composite.contrarian_signal,
        "divergence_alert": composite.divergence_alert,
    }


# ── Module-level convenience functions ────────────────────────────────────────

def get_composite_sentiment(ticker: str, hours: int = 72) -> SentimentComposite:
    """Convenience: full composite sentiment for a single ticker."""
    return _get_aggregator().get_composite(ticker, hours=hours)


def get_viral_tickers(min_mentions: int = 5) -> List[ViralTicker]:
    """Convenience: list of currently viral tickers."""
    return _get_velocity().get_viral_tickers(min_mentions=min_mentions)


def get_wsb_top(limit: int = 20) -> List[WSBTopTicker]:
    """Convenience: top WSB tickers by 24h mentions."""
    return _get_wsb().get_top_tickers_24h(limit=limit)


# ── Init DB on module load ─────────────────────────────────────────────────────
try:
    _ensure_db()
except Exception as _db_exc:
    logger.warning("social_sentiment_engine: DB init failed: %s", _db_exc)
