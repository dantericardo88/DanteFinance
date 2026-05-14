"""
Reddit WSB / r/investing and StockTwits social sentiment pipeline.

Sources (all free):
  - Reddit: PRAW (pip install praw) if credentials provided, else public JSON API (no auth).
  - StockTwits: https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json — no auth.

FinBERT scoring via sentinel.sil.sentiment.score_sentiment_batch.
pyproject.toml: praw = ">=7.7"  # optional
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SUBREDDITS = ["wallstreetbets", "investing", "stocks", "SecurityAnalysis"]
STOCKTWITS_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
REDDIT_SEARCH_URL = "https://www.reddit.com/r/{subreddit}/search.json"
REDDIT_BASE_HEADERS = {
    "User-Agent": "SENTINEL/1.0 (financial terminal; contact: sentinel@example.com)"
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class SocialPost(BaseModel):
    model_config = ConfigDict(frozen=True)

    platform: str                       # "reddit" | "stocktwits"
    ticker: str
    post_id: str
    created_at: datetime
    title: str
    body: Optional[str]
    score: int                          # upvotes (reddit) or likes (stocktwits)
    author_sentiment: Optional[str]     # stocktwits: "Bullish" | "Bearish" | None
    finbert_sentiment: Optional[str]    # "positive" | "negative" | "neutral"
    finbert_confidence: Optional[float]


class SocialSentimentSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: datetime
    total_posts: int
    reddit_posts: int
    stocktwits_posts: int
    bullish_pct: float          # % posts with positive/bullish signal
    bearish_pct: float
    neutral_pct: float
    avg_finbert_score: float    # -1 to +1, weighted by post score
    sentiment_signal: str       # "strong_bullish" | "bullish" | "neutral" | "bearish" | "strong_bearish"
    top_posts: list[SocialPost] # top 5 by score


# ── HTTP retry helper ─────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
async def _get_json(client: httpx.AsyncClient, url: str, **kwargs) -> dict:
    resp = await client.get(url, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ── Reddit fetching ───────────────────────────────────────────────────────────

async def _fetch_reddit_via_praw(
    ticker: str,
    subreddits: list[str],
    limit: int,
    client_id: str,
    client_secret: str,
) -> list[SocialPost]:
    """Use PRAW read-only mode to search subreddits for ticker mentions."""
    loop = asyncio.get_event_loop()

    def _praw_fetch() -> list[SocialPost]:
        import praw  # type: ignore

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="SENTINEL/1.0 financial terminal (read-only)",
        )
        posts: list[SocialPost] = []
        per_sub = max(1, limit // len(subreddits))

        for sub_name in subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                for submission in subreddit.search(
                    ticker,
                    sort="new",
                    time_filter="week",
                    limit=per_sub,
                ):
                    posts.append(
                        SocialPost(
                            platform="reddit",
                            ticker=ticker.upper(),
                            post_id=submission.id,
                            created_at=datetime.fromtimestamp(
                                submission.created_utc, tz=timezone.utc
                            ),
                            title=submission.title or "",
                            body=submission.selftext[:1000] if submission.selftext else None,
                            score=max(0, submission.score),
                            author_sentiment=None,
                            finbert_sentiment=None,
                            finbert_confidence=None,
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "PRAW subreddit search failed",
                    subreddit=sub_name,
                    ticker=ticker,
                    error=str(exc),
                )
        return posts

    try:
        return await loop.run_in_executor(None, _praw_fetch)
    except Exception as exc:
        logger.error("PRAW fetch failed", ticker=ticker, error=str(exc))
        return []


async def _fetch_reddit_via_json_api(
    ticker: str,
    subreddits: list[str],
    limit: int,
) -> list[SocialPost]:
    """
    Fallback: use Reddit's public JSON API — no auth required.
    Endpoint: https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&sort=new&t=week
    """
    posts: list[SocialPost] = []
    per_sub = max(1, limit // len(subreddits))

    async with httpx.AsyncClient(headers=REDDIT_BASE_HEADERS, timeout=15) as client:
        for sub_name in subreddits:
            url = REDDIT_SEARCH_URL.format(subreddit=sub_name)
            params = {
                "q": ticker,
                "sort": "new",
                "t": "week",
                "limit": per_sub,
                "restrict_sr": "1",
                "type": "link",
            }
            try:
                data = await _get_json(client, url, params=params)
                children = data.get("data", {}).get("children", [])
                for child in children:
                    p = child.get("data", {})
                    if not p:
                        continue
                    created_utc = p.get("created_utc", 0)
                    posts.append(
                        SocialPost(
                            platform="reddit",
                            ticker=ticker.upper(),
                            post_id=str(p.get("id", "")),
                            created_at=datetime.fromtimestamp(
                                float(created_utc), tz=timezone.utc
                            ) if created_utc else datetime.now(tz=timezone.utc),
                            title=p.get("title", ""),
                            body=p.get("selftext", "")[:1000] or None,
                            score=max(0, int(p.get("score", 0))),
                            author_sentiment=None,
                            finbert_sentiment=None,
                            finbert_confidence=None,
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Reddit JSON API failed",
                    subreddit=sub_name,
                    ticker=ticker,
                    error=str(exc),
                )

    logger.info(
        "Reddit JSON API posts fetched",
        ticker=ticker,
        count=len(posts),
        subreddits=subreddits,
    )
    return posts


async def fetch_reddit_posts(
    ticker: str,
    subreddits: Optional[list[str]] = None,
    limit: int = 50,
    reddit_client_id: Optional[str] = None,
    reddit_client_secret: Optional[str] = None,
) -> list[SocialPost]:
    """Fetch Reddit posts mentioning ticker. PRAW if credentials given, else public JSON API."""
    subs = subreddits or DEFAULT_SUBREDDITS

    if reddit_client_id and reddit_client_secret:
        try:
            posts = await _fetch_reddit_via_praw(
                ticker, subs, limit, reddit_client_id, reddit_client_secret
            )
            if posts:
                return posts
            logger.warning("PRAW returned 0 posts, falling back to JSON API", ticker=ticker)
        except Exception as exc:
            logger.warning("PRAW unavailable, using JSON API fallback", error=str(exc))

    return await _fetch_reddit_via_json_api(ticker, subs, limit)


# ── StockTwits fetching ───────────────────────────────────────────────────────

async def fetch_stocktwits(ticker: str, limit: int = 30) -> list[SocialPost]:
    """Fetch StockTwits stream for ticker. Free, no auth. Maps Bullish/Bearish labels."""
    url = STOCKTWITS_STREAM_URL.format(ticker=ticker.upper())
    posts: list[SocialPost] = []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            data = await _get_json(client, url)

        messages = data.get("messages", [])[:limit]
        for msg in messages:
            # StockTwits sentiment is nested under entities.sentiment
            entities = msg.get("entities", {})
            sentiment_obj = entities.get("sentiment", None)
            if sentiment_obj is None:
                # Older API format: sentiment directly on message
                sentiment_obj = msg.get("sentiment", None)

            author_sentiment: Optional[str] = None
            if isinstance(sentiment_obj, dict):
                raw_label = sentiment_obj.get("basic", "")
                if raw_label in ("Bullish", "Bearish"):
                    author_sentiment = raw_label

            created_str = msg.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except Exception:
                created_at = datetime.now(tz=timezone.utc)

            body_text = msg.get("body", "") or ""
            posts.append(
                SocialPost(
                    platform="stocktwits",
                    ticker=ticker.upper(),
                    post_id=str(msg.get("id", "")),
                    created_at=created_at,
                    title=body_text[:280],  # Use body as title (no separate title on ST)
                    body=body_text if len(body_text) > 280 else None,
                    score=int(msg.get("likes", {}).get("total", 0) if isinstance(msg.get("likes"), dict) else 0),
                    author_sentiment=author_sentiment,
                    finbert_sentiment=None,
                    finbert_confidence=None,
                )
            )

        logger.info("StockTwits posts fetched", ticker=ticker, count=len(posts))

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            logger.warning("StockTwits rate limited", ticker=ticker)
        elif exc.response.status_code == 404:
            logger.warning("StockTwits ticker not found", ticker=ticker)
        else:
            logger.error("StockTwits HTTP error", ticker=ticker, status=exc.response.status_code)
    except Exception as exc:
        logger.error("StockTwits fetch failed", ticker=ticker, error=str(exc))

    return posts


# ── FinBERT scoring ───────────────────────────────────────────────────────────

async def score_posts_with_finbert(posts: list[SocialPost]) -> list[SocialPost]:
    """Run FinBERT on post titles via sentinel.sil.sentiment.score_sentiment_batch."""
    if not posts:
        return posts

    from sentinel.sil.sentiment import score_sentiment_batch

    titles = [p.title for p in posts]
    try:
        results = await score_sentiment_batch(titles)
    except Exception as exc:
        logger.error("FinBERT batch scoring failed", error=str(exc))
        return posts

    scored: list[SocialPost] = []
    for post, sentiment_result in zip(posts, results):
        scored.append(
            post.model_copy(
                update={
                    "finbert_sentiment": sentiment_result.label,
                    "finbert_confidence": round(sentiment_result.score, 4),
                }
            )
        )
    return scored


# ── Signal computation ────────────────────────────────────────────────────────

def _compute_signal(bullish_pct: float, avg_score: float) -> str:
    """
    Map bullish percentage and average FinBERT score to a 5-tier signal.

    Tiers:
      strong_bullish: bullish_pct >= 65 AND avg_score >= 0.3
      bullish:        bullish_pct >= 50 OR avg_score >= 0.15
      bearish:        bullish_pct <= 35 OR avg_score <= -0.15
      strong_bearish: bullish_pct <= 25 AND avg_score <= -0.3
      neutral:        everything else
    """
    if bullish_pct >= 65 and avg_score >= 0.3:
        return "strong_bullish"
    if bullish_pct <= 25 and avg_score <= -0.3:
        return "strong_bearish"
    if bullish_pct >= 50 or avg_score >= 0.15:
        return "bullish"
    if bullish_pct <= 35 or avg_score <= -0.15:
        return "bearish"
    return "neutral"


def _aggregate_posts(ticker: str, posts: list[SocialPost]) -> SocialSentimentSummary:
    """Aggregate scored posts into SocialSentimentSummary. FinBERT avg weighted by post score."""
    total = len(posts)
    if total == 0:
        return SocialSentimentSummary(
            ticker=ticker.upper(),
            as_of=datetime.now(tz=timezone.utc),
            total_posts=0,
            reddit_posts=0,
            stocktwits_posts=0,
            bullish_pct=0.0,
            bearish_pct=0.0,
            neutral_pct=0.0,
            avg_finbert_score=0.0,
            sentiment_signal="neutral",
            top_posts=[],
        )

    reddit_posts = sum(1 for p in posts if p.platform == "reddit")
    stocktwits_posts = sum(1 for p in posts if p.platform == "stocktwits")

    # Determine bullish/bearish/neutral per post:
    # A post is bullish if: finbert_sentiment == "positive" OR author_sentiment == "Bullish"
    # A post is bearish if: finbert_sentiment == "negative" OR author_sentiment == "Bearish"
    # Prefer finbert when available; fall back to author_sentiment.
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    weighted_score_sum = 0.0
    weight_sum = 0.0

    for p in posts:
        weight = max(1, p.score)

        # Determine sentiment signal for this post
        if p.finbert_sentiment == "positive" or (
            p.finbert_sentiment is None and p.author_sentiment == "Bullish"
        ):
            bullish_count += 1
        elif p.finbert_sentiment == "negative" or (
            p.finbert_sentiment is None and p.author_sentiment == "Bearish"
        ):
            bearish_count += 1
        else:
            neutral_count += 1

        # FinBERT score contribution: positive → +confidence, negative → -confidence
        if p.finbert_sentiment == "positive" and p.finbert_confidence is not None:
            weighted_score_sum += p.finbert_confidence * weight
            weight_sum += weight
        elif p.finbert_sentiment == "negative" and p.finbert_confidence is not None:
            weighted_score_sum += -p.finbert_confidence * weight
            weight_sum += weight
        elif p.author_sentiment == "Bullish":
            # No FinBERT — use a weak fixed signal from StockTwits label
            weighted_score_sum += 0.3 * weight
            weight_sum += weight
        elif p.author_sentiment == "Bearish":
            weighted_score_sum += -0.3 * weight
            weight_sum += weight

    bullish_pct = round(bullish_count / total * 100, 1)
    bearish_pct = round(bearish_count / total * 100, 1)
    neutral_pct = round(neutral_count / total * 100, 1)
    avg_finbert_score = round(weighted_score_sum / weight_sum, 4) if weight_sum > 0 else 0.0

    signal = _compute_signal(bullish_pct, avg_finbert_score)

    top_posts = sorted(posts, key=lambda p: p.score, reverse=True)[:5]

    return SocialSentimentSummary(
        ticker=ticker.upper(),
        as_of=datetime.now(tz=timezone.utc),
        total_posts=total,
        reddit_posts=reddit_posts,
        stocktwits_posts=stocktwits_posts,
        bullish_pct=bullish_pct,
        bearish_pct=bearish_pct,
        neutral_pct=neutral_pct,
        avg_finbert_score=avg_finbert_score,
        sentiment_signal=signal,
        top_posts=top_posts,
    )


# ── Full pipeline ─────────────────────────────────────────────────────────────

async def get_social_sentiment(
    ticker: str,
    include_reddit: bool = True,
    include_stocktwits: bool = True,
    reddit_client_id: Optional[str] = None,
    reddit_client_secret: Optional[str] = None,
) -> SocialSentimentSummary:
    """Full pipeline: fetch Reddit + StockTwits → FinBERT score → aggregate → 5-tier signal."""
    if not include_reddit and not include_stocktwits:
        raise ValueError("At least one of include_reddit or include_stocktwits must be True.")

    # Fetch sources in parallel
    fetch_tasks = []
    if include_reddit:
        fetch_tasks.append(
            asyncio.create_task(
                fetch_reddit_posts(
                    ticker,
                    reddit_client_id=reddit_client_id,
                    reddit_client_secret=reddit_client_secret,
                )
            )
        )
    if include_stocktwits:
        fetch_tasks.append(asyncio.create_task(fetch_stocktwits(ticker)))

    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    all_posts: list[SocialPost] = []
    for result in results:
        if isinstance(result, list):
            all_posts.extend(result)
        elif isinstance(result, Exception):
            logger.warning("Social fetch task failed", error=str(result))

    if not all_posts:
        logger.warning("No social posts retrieved", ticker=ticker)
        return _aggregate_posts(ticker, [])

    # FinBERT scoring
    scored_posts = await score_posts_with_finbert(all_posts)

    summary = _aggregate_posts(ticker, scored_posts)
    logger.info(
        "Social sentiment computed",
        ticker=ticker,
        total=summary.total_posts,
        reddit=summary.reddit_posts,
        stocktwits=summary.stocktwits_posts,
        signal=summary.sentiment_signal,
        bullish_pct=summary.bullish_pct,
        avg_finbert=summary.avg_finbert_score,
    )
    return summary


# ── Convenience helpers ───────────────────────────────────────────────────────

def format_summary_display(summary: SocialSentimentSummary) -> str:
    """Compact terminal string for a SocialSentimentSummary."""
    icons = {"strong_bullish": "^^", "bullish": "^", "neutral": "->", "bearish": "v", "strong_bearish": "vv"}
    icon = icons.get(summary.sentiment_signal, "?")
    return (
        f"[{summary.ticker}] {icon} {summary.sentiment_signal.upper()} | "
        f"Posts:{summary.total_posts} R:{summary.reddit_posts} ST:{summary.stocktwits_posts} | "
        f"Bull:{summary.bullish_pct:.1f}% Bear:{summary.bearish_pct:.1f}% | "
        f"FinBERT:{summary.avg_finbert_score:+.3f} | {summary.as_of.strftime('%Y-%m-%d %H:%M UTC')}"
    )
