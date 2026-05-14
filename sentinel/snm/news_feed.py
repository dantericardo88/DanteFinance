"""News aggregation — multi-source RSS + Finnhub + GDELT with entity tagging and deduplication."""
from __future__ import annotations
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import feedparser
import httpx
from sentinel.core.types import NewsArticle, SentimentResult
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# Free financial RSS sources
RSS_FEEDS = {
    "reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    "bloomberg_economics": "https://feeds.bloomberg.com/economics/news.rss",
    "ft_markets": "https://www.ft.com/rss/home/uk",
    "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "seeking_alpha": "https://seekingalpha.com/feed.xml",
    "investopedia": "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline",
    "cnbc_investing": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
    "yahoo_finance": "https://finance.yahoo.com/rss/",
    "motley_fool": "https://www.fool.com/feeds/index.aspx",
    "zerohedge": "https://feeds.feedburner.com/zerohedge/feed",
    "econbrowser": "https://econbrowser.com/feed",
    "calculated_risk": "https://www.calculatedriskblog.com/feeds/posts/default",
}

# GDELT GKG for macro/geopolitical context
GDELT_TOP_THEMES = "https://api.gdeltproject.org/api/v2/doc/doc?query=economy+finance&mode=artlist&maxrecords=25&format=json"


class NewsFeedAggregator:
    """Fetches, deduplicates, and scores news from multiple sources."""

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()
        self._cache: list[NewsArticle] = []

    async def fetch_rss(
        self,
        sources: Optional[list[str]] = None,
        max_age_hours: int = 24,
    ) -> list[NewsArticle]:
        """Fetch articles from all (or specified) RSS feeds concurrently."""
        feed_names = sources or list(RSS_FEEDS.keys())
        tasks = [
            asyncio.create_task(self._fetch_single_rss(name, RSS_FEEDS[name]))
            for name in feed_names
            if name in RSS_FEEDS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        articles: list[NewsArticle] = []
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        for result in results:
            if isinstance(result, list):
                for a in result:
                    if a.published_at >= cutoff:
                        h = _hash_article(a)
                        if h not in self._seen_hashes:
                            self._seen_hashes.add(h)
                            articles.append(a)
        articles.sort(key=lambda a: a.published_at, reverse=True)
        logger.info("RSS articles fetched", count=len(articles))
        return articles

    async def _fetch_single_rss(self, source: str, url: str) -> list[NewsArticle]:
        """Fetch and parse a single RSS feed."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers={"User-Agent": "SENTINEL/1.0"})
                content = resp.text
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, lambda: feedparser.parse(content))
            articles = []
            for entry in feed.entries[:20]:
                pub_time = _parse_feed_time(entry)
                articles.append(NewsArticle(
                    headline=entry.get("title", ""),
                    summary=entry.get("summary", "")[:500],
                    source=source,
                    url=entry.get("link", ""),
                    published_at=pub_time,
                    tickers=[],
                    sentiment=None,
                ))
            return articles
        except Exception as exc:
            logger.warning("RSS fetch failed", source=source, error=str(exc))
            return []

    async def fetch_gdelt(self, query: str = "economy finance", max_articles: int = 25) -> list[NewsArticle]:
        """Fetch geopolitical and macro news from GDELT."""
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=artlist&maxrecords={max_articles}&format=json"
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            articles = []
            for item in data.get("articles", []):
                articles.append(NewsArticle(
                    headline=item.get("title", ""),
                    summary="",
                    source="gdelt",
                    url=item.get("url", ""),
                    published_at=_parse_gdelt_time(item.get("seendate", "")),
                    tickers=[],
                    sentiment=None,
                ))
            return articles
        except Exception as exc:
            logger.warning("GDELT fetch failed", error=str(exc))
            return []

    async def tag_tickers(
        self, articles: list[NewsArticle], known_tickers: set[str]
    ) -> list[NewsArticle]:
        """Tag articles with mentioned tickers using simple string matching."""
        import re
        # Build regex from known tickers (sorted by length desc to avoid partial matches)
        sorted_tickers = sorted(known_tickers, key=len, reverse=True)
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in sorted_tickers) + r")\b"
        )
        for article in articles:
            text = f"{article.headline} {article.summary}"
            found = list(set(pattern.findall(text)))
            article = article.model_copy(update={"tickers": found})
        return articles

    async def score_sentiments(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Add FinBERT sentiment to all articles."""
        from sentinel.sil.sentiment import score_sentiment_batch
        texts = [a.headline for a in articles]
        sentiments = await score_sentiment_batch(texts)
        result = []
        for article, sent in zip(articles, sentiments):
            result.append(article.model_copy(update={"sentiment": sent}))
        return result

    def get_articles_for_ticker(self, ticker: str) -> list[NewsArticle]:
        return [a for a in self._cache if ticker in a.tickers]


def _hash_article(a: NewsArticle) -> str:
    return hashlib.md5(f"{a.headline}{a.url}".encode()).hexdigest()


def _parse_feed_time(entry: dict) -> datetime:
    """Parse feedparser time tuple to datetime."""
    import time
    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if published:
        try:
            return datetime.utcfromtimestamp(time.mktime(published))
        except Exception:
            pass
    return datetime.utcnow()


def _parse_gdelt_time(s: str) -> datetime:
    """Parse GDELT date format YYYYMMDDTHHMMSSZ."""
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ")
    except Exception:
        return datetime.utcnow()
