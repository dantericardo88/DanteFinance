"""News article vector embeddings — pgvector semantic search pipeline.

Generates 384-dimensional sentence embeddings for news headlines using
all-MiniLM-L6-v2 (sentence-transformers). Embeddings are stored in the
news_articles.embedding column (vector(384)) for similarity search.

Usage:
    from sentinel.sil.news_embeddings import embed_pending_articles, semantic_search

    # Populate embeddings for all unembed articles
    n = await embed_pending_articles(session, batch_size=128)

    # Semantic similarity search
    results = await semantic_search("Fed rate hike inflation concerns", session, limit=10)

The model is lazy-loaded on first use (~200MB download on cold start).
After the first load it's cached in memory for the process lifetime.
"""
from __future__ import annotations
import asyncio
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_encoder = None  # lazy-loaded sentence transformer


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers all-MiniLM-L6-v2...")
            _encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence encoder loaded")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — "
                "run: pip install sentence-transformers"
            )
            raise
    return _encoder


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode a list of strings to 384-dim float vectors.

    Runs in a thread pool to avoid blocking the event loop.
    Returns an empty list if sentence-transformers is not installed.
    """
    if not texts:
        return []
    loop = asyncio.get_event_loop()
    try:
        encoder = _get_encoder()
        embeddings = await loop.run_in_executor(
            None, lambda: encoder.encode(texts, batch_size=64, show_progress_bar=False)
        )
        return [e.tolist() for e in embeddings]
    except Exception as exc:
        logger.error("Embedding error", error=str(exc))
        return []


async def embed_pending_articles(
    session: AsyncSession,
    batch_size: int = 128,
) -> int:
    """Populate embeddings for all news_articles rows where embedding IS NULL.

    Processes in batches to avoid OOM on large backlogs.
    Returns the total number of rows updated.
    """
    total_updated = 0

    while True:
        # Fetch a batch of articles without embeddings
        result = await session.execute(
            text("""
                SELECT id, headline, summary
                FROM news_articles
                WHERE embedding IS NULL
                ORDER BY published_at DESC
                LIMIT :batch_size
            """),
            {"batch_size": batch_size},
        )
        rows = result.mappings().fetchall()
        if not rows:
            break

        ids = [r["id"] for r in rows]
        # Concatenate headline + summary for richer embedding
        texts = [
            (r["headline"] or "") + " " + (r["summary"] or "")
            for r in rows
        ]

        embeddings = await embed_texts(texts)
        if not embeddings:
            logger.warning("Embedding generation failed — aborting batch")
            break

        if len(embeddings) != len(ids):
            logger.warning("Embedding count mismatch", expected=len(ids), got=len(embeddings))
            break

        # Write embeddings back using pgvector array syntax
        for article_id, emb in zip(ids, embeddings):
            emb_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
            await session.execute(
                text("UPDATE news_articles SET embedding = :emb::vector WHERE id = :id"),
                {"emb": emb_str, "id": article_id},
            )

        await session.commit()
        total_updated += len(ids)
        logger.info("Embeddings written", batch=len(ids), total=total_updated)

    return total_updated


async def semantic_search(
    query: str,
    session: AsyncSession,
    ticker: Optional[str] = None,
    limit: int = 10,
    similarity_threshold: float = 0.3,
) -> list[dict]:
    """Find news articles semantically similar to a query string.

    Uses cosine similarity (<=> operator in pgvector). Returns articles
    ordered by similarity descending. Requires embeddings to be populated first.

    Args:
        query: Natural language search query
        ticker: Optionally filter to articles mentioning this ticker
        limit: Max results
        similarity_threshold: Min cosine similarity (0-1); 0.3 is a loose match
    """
    query_embeddings = await embed_texts([query])
    if not query_embeddings:
        return []

    emb_str = "[" + ",".join(f"{v:.6f}" for v in query_embeddings[0]) + "]"

    ticker_filter = ""
    params: dict = {
        "emb": emb_str,
        "limit": limit,
        "threshold": 1 - similarity_threshold,  # pgvector distance = 1 - cosine_sim
    }
    if ticker:
        ticker_filter = "AND :ticker = ANY(tickers)"
        params["ticker"] = ticker.upper()

    result = await session.execute(
        text(f"""
            SELECT
                id, headline, summary, source, url, published_at,
                tickers, sentiment_label, sentiment_score,
                1 - (embedding <=> :emb::vector) AS similarity
            FROM news_articles
            WHERE embedding IS NOT NULL
            {ticker_filter}
            AND (embedding <=> :emb::vector) < :threshold
            ORDER BY embedding <=> :emb::vector
            LIMIT :limit
        """),
        params,
    )
    rows = result.mappings().fetchall()

    return [
        {
            "id": r["id"],
            "headline": r["headline"],
            "summary": r["summary"],
            "source": r["source"],
            "url": r["url"],
            "published": r["published_at"].isoformat() if hasattr(r.get("published_at"), "isoformat") else str(r.get("published_at")),
            "tickers": r["tickers"],
            "sentiment": r.get("sentiment_label"),
            "score": float(r["sentiment_score"]) if r.get("sentiment_score") is not None else None,
            "similarity": round(float(r["similarity"]), 4),
        }
        for r in rows
    ]


async def build_vector_index(session: AsyncSession) -> None:
    """Create the IVFFlat cosine index for fast approximate nearest-neighbour search.

    Only call this after embedding population is complete (needs > 100 rows).
    Re-running is safe — CREATE INDEX IF NOT EXISTS is idempotent.
    The index speeds up semantic_search by ~100x on large tables.
    """
    result = await session.execute(
        text("SELECT COUNT(*) FROM news_articles WHERE embedding IS NOT NULL")
    )
    count = result.scalar() or 0
    if count < 100:
        logger.warning(
            "Too few embeddings to build IVFFlat index",
            count=count, needed=100,
        )
        return

    lists = min(int(count / 39), 1000)  # pgvector recommendation: sqrt(rows)
    lists = max(lists, 10)

    await session.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS ix_news_embedding_ivfflat
            ON news_articles USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists})
        """)
    )
    await session.commit()
    logger.info("IVFFlat vector index built", lists=lists, rows=count)
