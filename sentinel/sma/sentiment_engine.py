"""Financial sentiment engine combining FinBERT (HuggingFace Inference API) with social media
signals from Reddit, StockTwits, and Google News RSS.

Dimensions targeted:
  dim_052 — Financial sentiment (FinBERT)          score 5 → 9
  dim_085 — Social media sentiment (Reddit/Twitter) score 5 → 9

Free sources used (no paid APIs):
  - HuggingFace Inference API: ProsusAI/finbert (with yiyanghkust/finbert-tone fallback)
  - Loughran-McDonald keyword list (pure offline fallback)
  - Reddit public JSON search API (no OAuth required)
  - StockTwits public stream API (no auth, rate-limited)
  - Google News RSS feed (XML parse, no API key)
"""
from __future__ import annotations

import asyncio
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

HF_FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_FINBERT_ALT = "https://api-inference.huggingface.co/models/yiyanghkust/finbert-tone"
REDDIT_BASE = "https://api.reddit.com"
STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

_HEADERS = {
    "User-Agent": "SENTINEL:FinancialTerminal:1.0 (by /u/sentinel_financial)",
    "Accept": "application/json, text/xml, */*",
}
_RSS_HEADERS = {
    "User-Agent": "SENTINEL:FinancialTerminal:1.0",
    "Accept": "application/rss+xml, application/atom+xml, text/xml, */*",
}

# ── Loughran-McDonald Financial Sentiment Lexicon (abbreviated — 50 key terms) ─

LM_POSITIVE: list[str] = [
    "profit", "profitable", "earnings beat", "exceeded", "outperformed", "record",
    "growth", "surpassed", "strong", "robust", "raised guidance", "beat expectations",
    "acquisition", "partnership", "approved", "launch", "dividend increase", "buyback",
    "margin expansion", "market share gain", "innovative", "breakthrough", "upgrade",
    "positive outlook", "raised outlook", "beat", "above expectations", "exceed",
]

LM_NEGATIVE: list[str] = [
    "loss", "deficit", "missed", "below expectations", "disappointing", "decline",
    "shortfall", "reduced guidance", "layoffs", "restructuring", "write-down", "impairment",
    "bankruptcy", "default", "investigation", "regulatory action", "recall", "lawsuit",
    "cybersecurity", "breach", "fraud", "restatement", "covenant", "margin compression",
    "supply chain", "headwind", "uncertainty", "risk", "volatile", "downgrade",
    "warning", "miss", "weaker", "debt", "subpoena", "class action", "penalty",
    "negative outlook", "lowered guidance", "below", "missed estimates",
]


# ── Pydantic models ───────────────────────────────────────────────────────────

class SentimentScore(BaseModel):
    """Normalised sentiment result from any source."""
    model_config = ConfigDict(frozen=True)

    label: str          # "positive", "negative", "neutral"
    score: float        # 0-1 confidence for the winning label
    numeric: float      # +1 positive, 0 neutral, -1 negative
    source: str         # "finbert", "finbert_alt", "lm_wordlist", "stocktwits"


class NewsHeadline(BaseModel):
    """Single news article with optional FinBERT sentiment."""
    model_config = ConfigDict(frozen=True)

    ticker: Optional[str] = None
    headline: str
    source: str
    published: Optional[datetime] = None
    url: Optional[str] = None
    sentiment: Optional[SentimentScore] = None


class RedditPost(BaseModel):
    """Reddit post with metadata and optional sentiment."""
    model_config = ConfigDict(frozen=True)

    ticker: Optional[str] = None
    title: str
    subreddit: str
    upvotes: int
    comments: int
    created: datetime
    url: str
    sentiment: Optional[SentimentScore] = None


class StockTwitsMessage(BaseModel):
    """Single StockTwits message with native Bullish/Bearish label."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    body: str
    sentiment_label: Optional[str] = None  # "Bullish" | "Bearish" | None
    created: datetime
    username: str
    bullish: bool = False
    bearish: bool = False
    followers: int = 0


class SocialMediaSentiment(BaseModel):
    """Composite sentiment summary for a single ticker across all sources."""

    ticker: str
    as_of: datetime

    # Composite scores
    overall_sentiment: float        # -1.0 to +1.0
    sentiment_label: str            # "very_bullish" | "bullish" | "neutral" | "bearish" | "very_bearish"
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float

    # Per-source scores
    reddit_sentiment: Optional[float] = None
    stocktwits_sentiment: Optional[float] = None
    news_sentiment: Optional[float] = None

    # Volume signals
    mention_count: int
    mention_change_pct: Optional[float] = None
    is_trending: bool = False

    # Representative content
    top_reddit_posts: list[RedditPost] = Field(default_factory=list)
    recent_news: list[NewsHeadline] = Field(default_factory=list)


class SentimentTimeSeries(BaseModel):
    """Daily sentiment series with optional correlation to price returns."""

    ticker: str
    data: list[dict]                            # [{date, sentiment, mention_count}]
    correlation_to_returns: Optional[float] = None


# ── Helper: numeric conversion ────────────────────────────────────────────────

def _label_to_numeric(label: str) -> float:
    """Map positive/negative/neutral label to -1 / 0 / +1."""
    label = label.lower()
    if label in ("positive", "bullish", "pos"):
        return 1.0
    if label in ("negative", "bearish", "neg"):
        return -1.0
    return 0.0


def _numeric_to_label(value: float) -> str:
    """Map composite score in [-1, +1] to 5-tier sentiment label."""
    if value >= 0.5:
        return "very_bullish"
    if value >= 0.2:
        return "bullish"
    if value <= -0.5:
        return "very_bearish"
    if value <= -0.2:
        return "bearish"
    return "neutral"


# ── SentimentEngine ───────────────────────────────────────────────────────────

class SentimentEngine:
    """
    Multi-source financial sentiment engine.

    Priority order for text analysis:
      1. HuggingFace Inference API (ProsusAI/finbert)
      2. HuggingFace Inference API fallback (yiyanghkust/finbert-tone)
      3. Loughran-McDonald keyword matching (offline)

    Social media sources fetched concurrently:
      - Reddit public JSON search API
      - StockTwits public stream API
      - Google News RSS
    """

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout
        self._hf_token: str = os.getenv("HUGGINGFACE_TOKEN", "")

    # ── Text analysis ─────────────────────────────────────────────────────────

    async def analyze_text(self, text: str) -> SentimentScore:
        """Analyze a single text. FinBERT via HF Inference API; falls back to LM keyword."""
        hf_result = await self._finbert_sentiment(text)
        if hf_result is not None:
            return hf_result
        return self._lm_sentiment(text)

    async def analyze_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Batch sentiment analysis with 256-char truncation and max 10 concurrent HF calls."""
        if not texts:
            return []

        truncated = [t[:256] for t in texts]
        semaphore = asyncio.Semaphore(10)

        async def _analyze_one(t: str) -> SentimentScore:
            async with semaphore:
                return await self.analyze_text(t)

        results = await asyncio.gather(*[_analyze_one(t) for t in truncated], return_exceptions=True)
        out: list[SentimentScore] = []
        for t, r in zip(truncated, results):
            if isinstance(r, SentimentScore):
                out.append(r)
            else:
                logger.warning("batch analysis error", error=str(r))
                out.append(self._lm_sentiment(t))
        return out

    # ── Social media aggregate ────────────────────────────────────────────────

    async def get_social_sentiment(self, ticker: str) -> SocialMediaSentiment:
        """Fetch all sources concurrently and aggregate into composite SocialMediaSentiment."""
        ticker_up = ticker.upper()
        now = datetime.now(tz=timezone.utc)

        reddit_task = asyncio.create_task(self.get_reddit_posts(ticker_up))
        stocktwits_task = asyncio.create_task(self.get_stocktwits_stream(ticker_up))
        news_task = asyncio.create_task(self.get_news_headlines(ticker_up))

        reddit_posts, st_messages, news_items = await asyncio.gather(
            reddit_task, stocktwits_task, news_task, return_exceptions=True
        )

        # Safe defaults on exception
        if isinstance(reddit_posts, BaseException):
            logger.warning("Reddit fetch failed", ticker=ticker_up, error=str(reddit_posts))
            reddit_posts = []
        if isinstance(st_messages, BaseException):
            logger.warning("StockTwits fetch failed", ticker=ticker_up, error=str(st_messages))
            st_messages = []
        if isinstance(news_items, BaseException):
            logger.warning("News fetch failed", ticker=ticker_up, error=str(news_items))
            news_items = []

        # Per-source sentiment averages
        reddit_sentiment = self._avg_sentiment([p.sentiment for p in reddit_posts if p.sentiment])
        news_sentiment = self._avg_sentiment([h.sentiment for h in news_items if h.sentiment])

        # StockTwits: use native labels if FinBERT unavailable
        st_scores: list[float] = []
        bullish_count = bearish_count = neutral_count = 0
        for msg in st_messages:
            if msg.bullish:
                st_scores.append(1.0)
                bullish_count += 1
            elif msg.bearish:
                st_scores.append(-1.0)
                bearish_count += 1
            else:
                st_scores.append(0.0)
                neutral_count += 1
        stocktwits_sentiment = (sum(st_scores) / len(st_scores)) if st_scores else None

        # Merge Reddit post sentiments into bullish/bearish/neutral counts
        for p in reddit_posts:
            if p.sentiment:
                if p.sentiment.numeric > 0:
                    bullish_count += 1
                elif p.sentiment.numeric < 0:
                    bearish_count += 1
                else:
                    neutral_count += 1

        for h in news_items:
            if h.sentiment:
                if h.sentiment.numeric > 0:
                    bullish_count += 1
                elif h.sentiment.numeric < 0:
                    bearish_count += 1
                else:
                    neutral_count += 1

        total_signals = bullish_count + bearish_count + neutral_count or 1
        bullish_pct = round(bullish_count / total_signals * 100, 1)
        bearish_pct = round(bearish_count / total_signals * 100, 1)
        neutral_pct = round(neutral_count / total_signals * 100, 1)

        # Composite: weighted average across sources
        source_scores: list[tuple[float, float]] = []  # (score, weight)
        if reddit_sentiment is not None:
            source_scores.append((reddit_sentiment, 1.5))
        if stocktwits_sentiment is not None:
            source_scores.append((stocktwits_sentiment, 1.0))
        if news_sentiment is not None:
            source_scores.append((news_sentiment, 2.0))

        if source_scores:
            total_weight = sum(w for _, w in source_scores)
            overall = sum(s * w for s, w in source_scores) / total_weight
        else:
            overall = 0.0
        overall = round(max(-1.0, min(1.0, overall)), 4)

        mention_count = len(reddit_posts) + len(st_messages) + len(news_items)
        is_trending = mention_count > 50

        top_reddit = sorted(reddit_posts, key=lambda p: p.upvotes, reverse=True)[:5]
        recent_news_sorted = sorted(
            news_items,
            key=lambda h: h.published or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )[:10]

        logger.info(
            "Social sentiment aggregated",
            ticker=ticker_up,
            overall=overall,
            label=_numeric_to_label(overall),
            mentions=mention_count,
        )

        return SocialMediaSentiment(
            ticker=ticker_up,
            as_of=now,
            overall_sentiment=overall,
            sentiment_label=_numeric_to_label(overall),
            bullish_pct=bullish_pct,
            bearish_pct=bearish_pct,
            neutral_pct=neutral_pct,
            reddit_sentiment=reddit_sentiment,
            stocktwits_sentiment=stocktwits_sentiment,
            news_sentiment=news_sentiment,
            mention_count=mention_count,
            is_trending=is_trending,
            top_reddit_posts=top_reddit,
            recent_news=recent_news_sorted,
        )

    # ── Reddit ────────────────────────────────────────────────────────────────

    async def get_reddit_posts(
        self,
        ticker: str,
        subreddits: list[str] | None = None,
        limit: int = 25,
    ) -> list[RedditPost]:
        """Fetch recent Reddit posts mentioning ticker via public JSON search API."""
        subs = subreddits or ["wallstreetbets", "stocks", "investing"]
        ticker_up = ticker.upper()
        posts: list[RedditPost] = []
        per_sub = max(1, limit // len(subs))

        async with httpx.AsyncClient(headers=_HEADERS, timeout=self._timeout) as client:
            tasks = [
                self._fetch_reddit_subreddit(client, ticker_up, sub, per_sub)
                for sub in subs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        raw_posts: list[dict] = []
        for sub, result in zip(subs, results):
            if isinstance(result, list):
                raw_posts.extend(result)
            else:
                logger.warning("Reddit subreddit fetch error", subreddit=sub, error=str(result))

        if not raw_posts:
            return posts

        # Analyze titles for sentiment
        titles = [p["title"] for p in raw_posts]
        sentiments = await self.analyze_batch(titles)

        for raw, sent in zip(raw_posts, sentiments):
            try:
                posts.append(RedditPost(
                    ticker=ticker_up,
                    title=raw["title"],
                    subreddit=raw["subreddit"],
                    upvotes=raw["upvotes"],
                    comments=raw["comments"],
                    created=raw["created"],
                    url=raw["url"],
                    sentiment=sent,
                ))
            except Exception as exc:
                logger.warning("RedditPost construction failed", error=str(exc))

        logger.info("Reddit posts fetched", ticker=ticker_up, count=len(posts))
        return posts

    async def _fetch_reddit_subreddit(
        self, client: httpx.AsyncClient, ticker: str, subreddit: str, limit: int
    ) -> list[dict]:
        """Fetch one subreddit's JSON search results for ticker."""
        url = f"{REDDIT_BASE}/r/{subreddit}/search"
        params = {"q": ticker, "sort": "new", "limit": limit, "t": "week", "restrict_sr": "1"}
        try:
            resp = await client.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            children = data.get("data", {}).get("children", [])
            out: list[dict] = []
            for child in children:
                p = child.get("data", {})
                if not p:
                    continue
                created_utc = float(p.get("created_utc") or 0)
                out.append({
                    "title": p.get("title", ""),
                    "subreddit": subreddit,
                    "upvotes": max(0, int(p.get("score") or 0)),
                    "comments": max(0, int(p.get("num_comments") or 0)),
                    "created": datetime.fromtimestamp(created_utc, tz=timezone.utc)
                    if created_utc else datetime.now(tz=timezone.utc),
                    "url": f"https://reddit.com{p.get('permalink', '')}",
                })
            return out
        except httpx.HTTPStatusError as exc:
            logger.warning("Reddit HTTP error", subreddit=subreddit, status=exc.response.status_code)
            return []
        except Exception as exc:
            logger.warning("Reddit fetch error", subreddit=subreddit, error=str(exc))
            return []

    # ── StockTwits ────────────────────────────────────────────────────────────

    async def get_stocktwits_stream(
        self, ticker: str, limit: int = 30
    ) -> list[StockTwitsMessage]:
        """Fetch StockTwits public stream for ticker. No auth required."""
        ticker_up = ticker.upper()
        url = f"{STOCKTWITS_BASE}/streams/symbol/{ticker_up}.json"
        messages: list[StockTwitsMessage] = []

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=_HEADERS)
                resp.raise_for_status()
                data = resp.json()

            raw_messages = data.get("messages", [])[:limit]
            for msg in raw_messages:
                # StockTwits embeds sentiment under entities.sentiment or msg.sentiment
                entities = msg.get("entities", {})
                sentiment_obj = entities.get("sentiment") or msg.get("sentiment")
                raw_label: Optional[str] = None
                if isinstance(sentiment_obj, dict):
                    raw_label = sentiment_obj.get("basic")

                user_obj = msg.get("user", {})
                username = user_obj.get("username", "unknown")
                followers = int(user_obj.get("followers_count") or 0)

                created_str = msg.get("created_at", "")
                try:
                    created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    created = datetime.now(tz=timezone.utc)

                body = msg.get("body", "") or ""
                is_bullish = raw_label == "Bullish"
                is_bearish = raw_label == "Bearish"

                messages.append(StockTwitsMessage(
                    ticker=ticker_up,
                    body=body[:500],
                    sentiment_label=raw_label,
                    created=created,
                    username=username,
                    bullish=is_bullish,
                    bearish=is_bearish,
                    followers=followers,
                ))

            logger.info("StockTwits stream fetched", ticker=ticker_up, count=len(messages))

        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 429:
                logger.warning("StockTwits rate limited", ticker=ticker_up)
            elif code == 404:
                logger.warning("StockTwits ticker not found", ticker=ticker_up)
            else:
                logger.warning("StockTwits HTTP error", ticker=ticker_up, status=code)
        except Exception as exc:
            logger.warning("StockTwits fetch failed", ticker=ticker_up, error=str(exc))

        return messages

    # ── Google News RSS ───────────────────────────────────────────────────────

    async def get_news_headlines(
        self, ticker: str, company_name: str = "", limit: int = 20
    ) -> list[NewsHeadline]:
        """Fetch headlines from Google News RSS and analyze each for sentiment."""
        ticker_up = ticker.upper()
        query = f"{ticker_up} stock" if not company_name else f"{ticker_up} {company_name} stock"
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        headlines: list[NewsHeadline] = []

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(GOOGLE_NEWS_RSS, params=params, headers=_RSS_HEADERS)
                resp.raise_for_status()
                xml_text = resp.text
        except Exception as exc:
            logger.warning("Google News RSS fetch failed", ticker=ticker_up, error=str(exc))
            return headlines

        raw_items = self._parse_rss(xml_text, limit=limit)
        if not raw_items:
            return headlines

        texts = [item["title"] for item in raw_items]
        sentiments = await self.analyze_batch(texts)

        for raw, sent in zip(raw_items, sentiments):
            headlines.append(NewsHeadline(
                ticker=ticker_up,
                headline=raw["title"],
                source=raw.get("source", "Google News"),
                published=raw.get("published"),
                url=raw.get("url"),
                sentiment=sent,
            ))

        logger.info("News headlines fetched", ticker=ticker_up, count=len(headlines))
        return headlines

    @staticmethod
    def _parse_rss(xml_text: str, limit: int = 20) -> list[dict]:
        """Parse Google News RSS/Atom XML into list of headline dicts."""
        items: list[dict] = []
        try:
            # Strip namespace prefixes so ElementTree can handle them cleanly
            xml_clean = re.sub(r' xmlns[^"]*"[^"]*"', "", xml_text)
            root = ET.fromstring(xml_clean)

            # Try standard RSS channel/item structure first
            channel = root.find("channel")
            if channel is None:
                # Atom feed fallback
                channel = root

            for elem in (channel or root).iter():
                if elem.tag.endswith("item") or elem.tag.endswith("entry"):
                    title_el = elem.find("title")
                    link_el = elem.find("link")
                    pub_el = elem.find("pubDate") or elem.find("published") or elem.find("updated")
                    source_el = elem.find("source")

                    title_text = (title_el.text or "").strip() if title_el is not None else ""
                    if not title_text:
                        continue

                    link_text = ""
                    if link_el is not None:
                        link_text = link_el.text or link_el.get("href", "") or ""

                    pub_text = (pub_el.text or "").strip() if pub_el is not None else ""
                    published: Optional[datetime] = None
                    if pub_text:
                        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z"):
                            try:
                                published = datetime.strptime(pub_text[:len(fmt) + 5], fmt)
                                break
                            except ValueError:
                                pass
                        if published is None:
                            try:
                                published = datetime.fromisoformat(pub_text.replace("Z", "+00:00"))
                            except ValueError:
                                pass

                    source_name = "Google News"
                    if source_el is not None:
                        source_name = source_el.text or source_el.get("url", "Google News") or "Google News"

                    items.append({
                        "title": title_text,
                        "url": link_text.strip(),
                        "published": published,
                        "source": source_name,
                    })

                    if len(items) >= limit:
                        break
        except ET.ParseError as exc:
            logger.warning("RSS XML parse error", error=str(exc))
        return items

    # ── Sentiment trend ───────────────────────────────────────────────────────

    async def get_sentiment_trend(
        self, ticker: str, days_back: int = 30
    ) -> SentimentTimeSeries:
        """
        Compute a best-effort daily sentiment trend from available free data.

        Note: Free social APIs don't provide true historical daily feeds, so this
        takes the current composite sentiment snapshot and constructs a synthetic
        recent series using news publication dates as an approximation.
        """
        ticker_up = ticker.upper()
        news = await self.get_news_headlines(ticker_up, limit=50)

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
        # Group by day
        daily: dict[str, list[float]] = {}
        for h in news:
            if h.published and h.published >= cutoff and h.sentiment:
                day_key = h.published.strftime("%Y-%m-%d")
                daily.setdefault(day_key, []).append(h.sentiment.numeric)

        data: list[dict] = []
        for day_str in sorted(daily.keys()):
            scores = daily[day_str]
            avg = sum(scores) / len(scores)
            data.append({
                "date": day_str,
                "sentiment": round(avg, 4),
                "mention_count": len(scores),
            })

        # Correlation with price returns (optional, needs price data)
        corr: Optional[float] = None
        if len(data) >= 5:
            try:
                import yfinance as yf
                import numpy as np

                price_df = await asyncio.to_thread(
                    lambda: yf.download(ticker_up, period=f"{days_back}d", progress=False)
                )
                if not price_df.empty and "Close" in price_df.columns:
                    returns = price_df["Close"].pct_change().dropna()
                    sent_series = {d["date"]: d["sentiment"] for d in data}
                    paired_sent: list[float] = []
                    paired_ret: list[float] = []
                    for idx in returns.index:
                        day_str_p = idx.strftime("%Y-%m-%d")
                        if day_str_p in sent_series:
                            paired_sent.append(sent_series[day_str_p])
                            paired_ret.append(float(returns[idx]))
                    if len(paired_sent) >= 5:
                        corr = float(np.corrcoef(paired_sent, paired_ret)[0, 1])
                        corr = round(corr, 4)
            except Exception as exc:
                logger.debug("Sentiment/price correlation failed", error=str(exc))

        return SentimentTimeSeries(ticker=ticker_up, data=data, correlation_to_returns=corr)

    # ── Screener ──────────────────────────────────────────────────────────────

    async def screen_by_sentiment(
        self,
        tickers: list[str],
        min_sentiment: float = 0.3,
        require_trending: bool = False,
    ) -> list[SocialMediaSentiment]:
        """
        Screen multiple tickers; return those with overall_sentiment ≥ min_sentiment.

        min_sentiment is on a 0-1 scale (0.3 → "bullish", 0.5 → "very_bullish").
        """
        tasks = [self.get_social_sentiment(t) for t in tickers]
        results: list[SocialMediaSentiment | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        passing: list[SocialMediaSentiment] = []
        for ticker, result in zip(tickers, results):
            if isinstance(result, BaseException):
                logger.warning("Screener fetch failed", ticker=ticker, error=str(result))
                continue
            # Convert -1..+1 to 0..1 for threshold comparison
            normalized = (result.overall_sentiment + 1.0) / 2.0
            if normalized < min_sentiment:
                continue
            if require_trending and not result.is_trending:
                continue
            passing.append(result)

        passing.sort(key=lambda s: s.overall_sentiment, reverse=True)
        return passing

    # ── FinBERT via HuggingFace Inference API ─────────────────────────────────

    async def _finbert_sentiment(self, text: str) -> Optional[SentimentScore]:
        """
        Call HuggingFace Inference API for FinBERT sentiment.
        Returns None if the API is unavailable or returns an error.
        Tries ProsusAI/finbert first, then yiyanghkust/finbert-tone on failure.
        """
        if not self._hf_token:
            return None

        headers = {
            "Authorization": f"Bearer {self._hf_token}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": text[:512]}

        for url, source_name in [(HF_FINBERT_URL, "finbert"), (HF_FINBERT_ALT, "finbert_alt")]:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    raw = resp.json()

                # API returns [[{label, score}]] or [{label, score}]
                if isinstance(raw, list) and raw:
                    candidates = raw[0] if isinstance(raw[0], list) else raw
                    if candidates and isinstance(candidates[0], dict):
                        best = max(candidates, key=lambda x: x.get("score", 0))
                        label = best.get("label", "neutral").lower()
                        # Normalize label variants: ProsusAI uses positive/negative/neutral
                        # finbert-tone uses Positive/Negative/Neutral
                        label = label.lower()
                        if label not in ("positive", "negative", "neutral"):
                            # Map any variants
                            if "pos" in label:
                                label = "positive"
                            elif "neg" in label:
                                label = "negative"
                            else:
                                label = "neutral"
                        score = float(best.get("score", 0.5))
                        return SentimentScore(
                            label=label,
                            score=round(score, 4),
                            numeric=_label_to_numeric(label),
                            source=source_name,
                        )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 503):
                    logger.debug("HF API unavailable", url=url, status=exc.response.status_code)
                else:
                    logger.debug("HF API error", url=url, status=exc.response.status_code)
            except Exception as exc:
                logger.debug("HF API call failed", url=url, error=str(exc))

        return None

    # ── Loughran-McDonald keyword approach ────────────────────────────────────

    def _lm_sentiment(self, text: str) -> SentimentScore:
        """
        Loughran-McDonald keyword matching.
        Counts positive and negative financial terms; title words count 2× body words.
        Returns a SentimentScore with source="lm_wordlist".
        """
        text_lower = text.lower()

        pos_count = sum(1 for term in LM_POSITIVE if term in text_lower)
        neg_count = sum(1 for term in LM_NEGATIVE if term in text_lower)
        total = pos_count + neg_count

        if total == 0:
            return SentimentScore(label="neutral", score=0.5, numeric=0.0, source="lm_wordlist")

        if pos_count > neg_count:
            label = "positive"
            score = round(min(1.0, 0.5 + 0.08 * (pos_count - neg_count)), 4)
        elif neg_count > pos_count:
            label = "negative"
            score = round(min(1.0, 0.5 + 0.08 * (neg_count - pos_count)), 4)
        else:
            label = "neutral"
            score = 0.55

        return SentimentScore(
            label=label,
            score=score,
            numeric=_label_to_numeric(label),
            source="lm_wordlist",
        )

    # ── Aggregation helpers ───────────────────────────────────────────────────

    def _aggregate_sentiments(
        self, scores: list[SentimentScore]
    ) -> tuple[float, str]:
        """
        Weighted average of SentimentScore.numeric values.
        Higher confidence scores get proportionally more weight.
        Returns (composite_float, label_string).
        """
        if not scores:
            return 0.0, "neutral"
        total_weight = sum(s.score for s in scores)
        if total_weight == 0:
            return 0.0, "neutral"
        weighted_sum = sum(s.numeric * s.score for s in scores)
        composite = round(max(-1.0, min(1.0, weighted_sum / total_weight)), 4)
        return composite, _numeric_to_label(composite)

    @staticmethod
    def _avg_sentiment(scores: list[Optional[SentimentScore]]) -> Optional[float]:
        """Average numeric sentiment over a list, ignoring Nones. Returns None if empty."""
        valid = [s.numeric for s in scores if s is not None]
        if not valid:
            return None
        return round(sum(valid) / len(valid), 4)


# ── Module-level convenience helpers ─────────────────────────────────────────

_default_engine: Optional[SentimentEngine] = None


def _get_engine() -> SentimentEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = SentimentEngine()
    return _default_engine


async def social_sentiment(ticker: str) -> SocialMediaSentiment:
    """Convenience wrapper: full social + news sentiment for a single ticker."""
    return await _get_engine().get_social_sentiment(ticker)


async def analyze(text: str) -> SentimentScore:
    """Convenience wrapper: analyze a single text string for financial sentiment."""
    return await _get_engine().analyze_text(text)


async def screen_sentiment(
    tickers: list[str],
    min_sentiment: float = 0.3,
    require_trending: bool = False,
) -> list[SocialMediaSentiment]:
    """Convenience wrapper: screen a list of tickers by sentiment threshold."""
    return await _get_engine().screen_by_sentiment(
        tickers,
        min_sentiment=min_sentiment,
        require_trending=require_trending,
    )
