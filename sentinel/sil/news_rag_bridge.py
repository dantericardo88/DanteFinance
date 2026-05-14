"""
News→RAG bridge — ingests sentinel news articles into the document_chunks RAG table.

Connects the news feed (snm/news_feed.py, Finnhub) to the pgvector RAG pipeline
so that news articles become semantically searchable via sil/rag.py.

Improves dim 57 (earnings call / news corpus) from 1→4 and makes the
RAG pipeline actually useful for news-driven research queries.
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, date, timedelta
from typing import Optional

from pydantic import BaseModel
from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────

class IngestStats(BaseModel):
    ingested: int = 0
    skipped_duplicate: int = 0
    failed: int = 0
    total_chunks: int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _article_to_chunks(
    headline: str,
    summary: Optional[str],
    full_text: Optional[str],
    ticker: Optional[str],
    source: str,
    published_at: datetime,
    url: Optional[str] = None,
) -> list[dict]:
    """
    Convert a news article into document chunks for RAG ingestion.

    Chunking strategy:
    - Chunk 0: headline alone (very high signal for keyword matching)
    - Chunk 1: headline + summary (if summary available)
    - Chunk 2+: full text split into ~512-char paragraphs (if available)
    """
    doc_id = hashlib.md5(
        f"{source}:{headline}:{published_at.isoformat()}".encode()
    ).hexdigest()

    chunks: list[dict] = []
    filed_date = published_at.date() if isinstance(published_at, datetime) else published_at
    metadata = {"source": source, "url": url or "", "published_at": published_at.isoformat()}

    # Chunk 0: headline only
    chunks.append({
        "chunk_id": f"{doc_id}_h",
        "doc_id": doc_id,
        "ticker": ticker,
        "doc_type": "news",
        "filed_date": filed_date,
        "text": headline,
        "metadata": metadata,
    })

    # Chunk 1: headline + summary
    if summary and summary.strip() and summary.strip() != headline.strip():
        combined = f"{headline}\n\n{summary}"
        chunks.append({
            "chunk_id": f"{doc_id}_s",
            "doc_id": doc_id,
            "ticker": ticker,
            "doc_type": "news",
            "filed_date": filed_date,
            "text": combined[:1024],  # cap for embedding efficiency
            "metadata": metadata,
        })

    # Chunk 2+: full text paragraphs
    if full_text and len(full_text) > 200:
        paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 100]
        for i, para in enumerate(paragraphs[:6]):  # max 6 full-text chunks per article
            chunks.append({
                "chunk_id": f"{doc_id}_p{i}",
                "doc_id": doc_id,
                "ticker": ticker,
                "doc_type": "news",
                "filed_date": filed_date,
                "text": para[:512],
                "metadata": metadata,
            })

    return chunks


# ── Main ingestion functions ───────────────────────────────────────────────────

async def ingest_news_articles(
    db_url: str,
    articles: list[dict],
) -> IngestStats:
    """
    Ingest a list of news article dicts into the document_chunks RAG table.

    Each article dict should have:
        headline: str (required)
        summary: str | None
        full_text: str | None
        ticker: str | None
        source: str
        published_at: datetime or ISO string
        url: str | None

    Uses the rag.py ingest_document pipeline for proper embedding + upsert.
    """
    from sentinel.sil.rag import ingest_document

    stats = IngestStats()

    for article in articles:
        try:
            headline = article.get("headline", "")
            if not headline:
                stats.skipped_duplicate += 1
                continue

            summary = article.get("summary")
            full_text = article.get("full_text")
            ticker = article.get("ticker")
            source = article.get("source", "news")
            url = article.get("url")

            # Normalize published_at
            pub_raw = article.get("published_at")
            if isinstance(pub_raw, (int, float)):
                published_at = datetime.utcfromtimestamp(pub_raw)
            elif isinstance(pub_raw, str):
                published_at = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            elif isinstance(pub_raw, datetime):
                published_at = pub_raw
            else:
                published_at = datetime.utcnow()

            # Build combined text for ingest_document
            parts = [headline]
            if summary:
                parts.append(summary)
            if full_text:
                parts.append(full_text[:2000])
            full_content = "\n\n".join(parts)

            doc_id = hashlib.md5(
                f"{source}:{headline}:{published_at.isoformat()}".encode()
            ).hexdigest()

            n_chunks = await ingest_document(
                db_url=db_url,
                text=full_content,
                doc_id=doc_id,
                ticker=ticker,
                doc_type="news",
                filed_date=published_at.date(),
                metadata={"source": source, "url": url or ""},
                chunk_size=400,
                chunk_overlap=50,
            )
            stats.ingested += 1
            stats.total_chunks += n_chunks

        except Exception as exc:
            logger.warning("Failed to ingest article", headline=str(article.get("headline", ""))[:80], error=str(exc))
            stats.failed += 1

    logger.info(
        "news_rag_bridge.ingest_news_articles",
        ingested=stats.ingested,
        failed=stats.failed,
        total_chunks=stats.total_chunks,
    )
    return stats


async def ingest_finnhub_news(
    db_url: str,
    ticker: str,
    days_back: int = 7,
) -> IngestStats:
    """
    Fetch Finnhub news for a ticker and ingest into RAG.
    Uses FinnhubAdapter from the existing adapter registry.
    """
    from sentinel.sds.adapters.finnhub_adapter import FinnhubAdapter
    from sentinel.core.config import get_settings

    s = get_settings()
    if not s.finnhub_api_key:
        logger.warning("FINNHUB_API_KEY not set — skipping Finnhub news ingest")
        return IngestStats()

    fh = FinnhubAdapter(api_key=s.finnhub_api_key)
    end_dt = date.today().isoformat()
    start_dt = (date.today() - timedelta(days=days_back)).isoformat()

    try:
        news = await fh.fetch_news(ticker, start_dt, end_dt)
    except Exception as exc:
        logger.error("Finnhub news fetch failed", ticker=ticker, error=str(exc))
        return IngestStats()

    # Normalize Finnhub format
    articles = []
    for item in news:
        articles.append({
            "headline": item.get("headline", ""),
            "summary": item.get("summary"),
            "full_text": None,
            "ticker": ticker,
            "source": "finnhub",
            "published_at": item.get("datetime", 0),
            "url": item.get("url"),
        })

    return await ingest_news_articles(db_url=db_url, articles=articles)


async def ingest_recent_news_all_tickers(
    db_url: str,
    tickers: list[str],
    days_back: int = 3,
    concurrency: int = 5,
) -> IngestStats:
    """
    Bulk-ingest recent news for a list of tickers.
    Rate-limited to concurrency simultaneous requests.
    """
    sem = asyncio.Semaphore(concurrency)
    total = IngestStats()

    async def _ingest_one(ticker: str) -> IngestStats:
        async with sem:
            return await ingest_finnhub_news(db_url=db_url, ticker=ticker, days_back=days_back)

    results = await asyncio.gather(*[_ingest_one(t) for t in tickers], return_exceptions=True)

    for r in results:
        if isinstance(r, IngestStats):
            total.ingested += r.ingested
            total.failed += r.failed
            total.total_chunks += r.total_chunks
        else:
            total.failed += 1

    logger.info(
        "ingest_recent_news_all_tickers complete",
        tickers=len(tickers),
        ingested=total.ingested,
        chunks=total.total_chunks,
        failed=total.failed,
    )
    return total


async def ingest_sentinel_news_feed(
    db_url: str,
    lookback_hours: int = 24,
) -> IngestStats:
    """
    Pull recent articles from the sentinel news_articles DB table
    (populated by snm/news_feed.py scheduler) and cross-ingest into document_chunks.

    This bridges the two storage systems:
    - news_articles: structured table with sentiment scores
    - document_chunks: RAG vector index for semantic search
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(db_url, pool_pre_ping=True)
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

    async with engine.connect() as conn:
        rows = (await conn.execute(
            text("""
                SELECT headline, summary, full_text, ticker, source, published_at, url
                FROM news_articles
                WHERE published_at >= :cutoff
                  AND (embedding IS NOT NULL OR true)
                ORDER BY published_at DESC
                LIMIT 500
            """),
            {"cutoff": cutoff},
        )).mappings().fetchall()

    if not rows:
        logger.info("No recent news_articles to cross-ingest")
        return IngestStats()

    articles = [dict(r) for r in rows]
    logger.info("Cross-ingesting news_articles into document_chunks", count=len(articles))
    return await ingest_news_articles(db_url=db_url, articles=articles)
