"""RAG pipeline — dense (pgvector) + sparse (BM25/tsvector) + RRF over financial documents."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import date
from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from sentinel.core.logging import get_logger
from sentinel.core.config import get_settings

logger = get_logger(__name__)

# ── Lazy embedder ─────────────────────────────────────────────────────────────

_embedder = None


def _get_embedder():
    """Lazy-load all-MiniLM-L6-v2 (384-dim)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading all-MiniLM-L6-v2 embedder...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder loaded")
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts. Returns list of 384-dim float vectors."""
    return [v.tolist() for v in _get_embedder().encode(texts, show_progress_bar=False, batch_size=32)]


# ── Models ────────────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    ticker: str | None
    doc_type: str        # "10-K" | "10-Q" | "earnings_call" | "news" | "8-K"
    filed_date: date | None
    text: str
    embedding: list[float] | None = None
    metadata: dict = {}

class RAGResult(BaseModel):
    chunk_id: str
    ticker: str | None
    doc_type: str
    filed_date: date | None
    text: str
    score: float           # RRF combined score
    dense_rank: int | None
    sparse_rank: int | None
    source: str            # "dense" | "sparse" | "fusion"

class RAGResponse(BaseModel):
    query: str
    results: list[RAGResult]
    answer: str | None     # Claude-generated answer if synthesize=True
    tokens_used: int = 0

# ── Engine cache ──────────────────────────────────────────────────────────────

_engines: dict[str, AsyncEngine] = {}

def _get_engine(db_url: str) -> AsyncEngine:
    if db_url not in _engines:
        _engines[db_url] = create_async_engine(db_url, pool_size=5, max_overflow=10, pool_pre_ping=True)
    return _engines[db_url]


# ── DDL — ensure document_chunks table exists ─────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    ticker      VARCHAR(20),
    doc_type    VARCHAR(30) NOT NULL,
    filed_date  DATE,
    chunk_text  TEXT NOT NULL,
    embedding   vector(384),
    metadata    JSONB NOT NULL DEFAULT '{}',
    ts_vec      TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_dc_ticker    ON document_chunks (ticker) WHERE ticker IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_dc_doc_type  ON document_chunks (doc_type);
CREATE INDEX IF NOT EXISTS ix_dc_ts_vec    ON document_chunks USING GIN (ts_vec);
"""


_schema_ready: set[str] = set()

async def _maybe_init_schema(db_url: str) -> None:
    if db_url in _schema_ready:
        return
    async with _get_engine(db_url).begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                await conn.execute(text(stmt))
    _schema_ready.add(db_url)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _sentence_split(text: str) -> list[str]:
    """Split on sentence boundaries using punctuation heuristic."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    text: str,
    doc_id: str,
    ticker: str | None,
    doc_type: str,
    filed_date: date | None,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[DocumentChunk]:
    """Split into overlapping word-count chunks respecting sentence boundaries.
    chunk_size/overlap in whitespace-token words (≈ BPE tokens); no tiktoken needed."""
    words = [w for sent in _sentence_split(text) for w in sent.split()]
    if not words:
        return []
    chunks: list[DocumentChunk] = []
    start, total = 0, len(words)
    while start < total:
        end = min(start + chunk_size, total)
        chunk_id = hashlib.sha256(f"{doc_id}:{len(chunks)}".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            chunk_id=chunk_id, doc_id=doc_id, ticker=ticker, doc_type=doc_type,
            filed_date=filed_date, text=" ".join(words[start:end]),
        ))
        if end == total:
            break
        start = end - overlap
    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

async def ingest_document(
    db_url: str,
    text: str,
    doc_id: str,
    ticker: str | None,
    doc_type: str,
    filed_date: date | None,
    metadata: dict = {},
) -> int:
    """Chunk → embed → upsert into document_chunks table. Returns chunks ingested."""
    await _maybe_init_schema(db_url)

    chunks = chunk_document(text, doc_id, ticker, doc_type, filed_date)
    if not chunks:
        logger.warning("ingest_document: empty text", doc_id=doc_id)
        return 0

    # Embed all chunks (CPU-bound — offload to executor)
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None, embed_texts, [c.text for c in chunks]
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    engine = _get_engine(db_url)
    import json

    upsert_sql = text("""
        INSERT INTO document_chunks
            (chunk_id, doc_id, ticker, doc_type, filed_date, chunk_text, embedding, metadata)
        VALUES
            (:chunk_id, :doc_id, :ticker, :doc_type, :filed_date, :chunk_text, :embedding, :metadata)
        ON CONFLICT (chunk_id) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding  = EXCLUDED.embedding,
            metadata   = EXCLUDED.metadata
    """)

    async with engine.begin() as conn:
        for chunk in chunks:
            vec_str = "[" + ",".join(f"{x:.6f}" for x in chunk.embedding) + "]"
            await conn.execute(upsert_sql, {
                "chunk_id":  chunk.chunk_id,
                "doc_id":    chunk.doc_id,
                "ticker":    chunk.ticker,
                "doc_type":  chunk.doc_type,
                "filed_date": chunk.filed_date,
                "chunk_text": chunk.text,
                "embedding": vec_str,
                "metadata":  json.dumps(metadata),
            })

    logger.info("ingest_document complete", doc_id=doc_id, chunks=len(chunks))
    return len(chunks)


async def ingest_batch(db_url: str, documents: list[dict]) -> int:
    """Ingest multiple documents concurrently. Each dict: text, doc_id, ticker, doc_type, filed_date."""
    tasks = [
        ingest_document(
            db_url=db_url,
            text=d["text"],
            doc_id=d["doc_id"],
            ticker=d.get("ticker"),
            doc_type=d["doc_type"],
            filed_date=d.get("filed_date"),
            metadata=d.get("metadata", {}),
        )
        for d in documents
    ]
    results = await asyncio.gather(*tasks)
    total = sum(results)
    logger.info("ingest_batch complete", docs=len(documents), total_chunks=total)
    return total


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _build_filters(ticker: str | None, doc_type: str | None) -> tuple[str, dict]:
    """Return a WHERE clause fragment and bind params for optional filters."""
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if ticker:
        clauses.append("ticker = :ticker")
        params["ticker"] = ticker
    if doc_type:
        clauses.append("doc_type = :doc_type")
        params["doc_type"] = doc_type
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# ── Dense retrieval ───────────────────────────────────────────────────────────

async def dense_retrieve(
    db_url: str,
    query: str,
    top_k: int = 20,
    ticker: str | None = None,
    doc_type: str | None = None,
) -> list[tuple[RAGResult, int]]:
    """pgvector cosine similarity search. Returns (result, rank) tuples."""
    await _maybe_init_schema(db_url)

    loop = asyncio.get_event_loop()
    [q_vec] = await loop.run_in_executor(None, embed_texts, [query])
    vec_str = "[" + ",".join(f"{x:.6f}" for x in q_vec) + "]"

    where, params = _build_filters(ticker, doc_type)
    params.update({"vec": vec_str, "top_k": top_k})

    sql = text(f"""
        SELECT chunk_id, ticker, doc_type, filed_date, chunk_text,
               1 - (embedding <=> :vec::vector) AS similarity
        FROM document_chunks
        {where}
        ORDER BY embedding <=> :vec::vector
        LIMIT :top_k
    """)

    engine = _get_engine(db_url)
    async with engine.connect() as conn:
        rows = (await conn.execute(sql, params)).fetchall()

    results: list[tuple[RAGResult, int]] = []
    for rank, row in enumerate(rows, start=1):
        results.append((
            RAGResult(
                chunk_id=row.chunk_id,
                ticker=row.ticker,
                doc_type=row.doc_type,
                filed_date=row.filed_date,
                text=row.chunk_text,
                score=float(row.similarity),
                dense_rank=rank,
                sparse_rank=None,
                source="dense",
            ),
            rank,
        ))

    logger.debug("dense_retrieve", query=query[:60], hits=len(results))
    return results


# ── Sparse retrieval (BM25 via tsvector) ──────────────────────────────────────

async def sparse_retrieve(
    db_url: str,
    query: str,
    top_k: int = 20,
    ticker: str | None = None,
    doc_type: str | None = None,
) -> list[tuple[RAGResult, int]]:
    """BM25 via PostgreSQL full-text search. Returns (result, rank) tuples."""
    await _maybe_init_schema(db_url)

    where, params = _build_filters(ticker, doc_type)
    # Prepend ts_rank condition
    ts_clause = "ts_vec @@ plainto_tsquery('english', :query)"
    if where:
        where = where + " AND " + ts_clause
    else:
        where = "WHERE " + ts_clause
    params["query"] = query
    params["top_k"] = top_k

    sql = text(f"""
        SELECT chunk_id, ticker, doc_type, filed_date, chunk_text,
               ts_rank_cd(ts_vec, plainto_tsquery('english', :query)) AS bm25_score
        FROM document_chunks
        {where}
        ORDER BY bm25_score DESC
        LIMIT :top_k
    """)

    engine = _get_engine(db_url)
    async with engine.connect() as conn:
        rows = (await conn.execute(sql, params)).fetchall()

    results: list[tuple[RAGResult, int]] = []
    for rank, row in enumerate(rows, start=1):
        results.append((
            RAGResult(
                chunk_id=row.chunk_id,
                ticker=row.ticker,
                doc_type=row.doc_type,
                filed_date=row.filed_date,
                text=row.chunk_text,
                score=float(row.bm25_score),
                dense_rank=None,
                sparse_rank=rank,
                source="sparse",
            ),
            rank,
        ))

    logger.debug("sparse_retrieve", query=query[:60], hits=len(results))
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[tuple[RAGResult, int]],
    sparse_results: list[tuple[RAGResult, int]],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[RAGResult]:
    """RRF formula: score = dense_w/(k+dense_rank) + sparse_w/(k+sparse_rank).
    Merge by chunk_id. Sets source='fusion'. Returns sorted by combined score desc."""
    scores: dict[str, float] = {}
    dense_map: dict[str, tuple[RAGResult, int]] = {}
    sparse_map: dict[str, tuple[RAGResult, int]] = {}

    for result, rank in dense_results:
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank)
        dense_map[cid] = (result, rank)

    for result, rank in sparse_results:
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0.0) + sparse_weight / (k + rank)
        sparse_map[cid] = (result, rank)

    merged: list[RAGResult] = []
    for cid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        # Prefer dense result as the base record; fall back to sparse
        base_result = dense_map[cid][0] if cid in dense_map else sparse_map[cid][0]
        d_rank = dense_map[cid][1] if cid in dense_map else None
        s_rank = sparse_map[cid][1] if cid in sparse_map else None
        source = "fusion" if (cid in dense_map and cid in sparse_map) else base_result.source

        merged.append(RAGResult(
            chunk_id=cid,
            ticker=base_result.ticker,
            doc_type=base_result.doc_type,
            filed_date=base_result.filed_date,
            text=base_result.text,
            score=round(rrf_score, 6),
            dense_rank=d_rank,
            sparse_rank=s_rank,
            source=source,
        ))

    return merged


# ── Claude synthesis ──────────────────────────────────────────────────────────

async def _synthesize(
    query: str,
    results: list[RAGResult],
    api_key: str,
) -> tuple[str, int]:
    """Call Claude Haiku with retrieved context and return (answer, tokens_used)."""
    context_parts = []
    for i, r in enumerate(results[:8], start=1):
        date_str = r.filed_date.isoformat() if r.filed_date else "unknown date"
        ticker_str = r.ticker or "N/A"
        context_parts.append(
            f"[{i}] ({r.doc_type} | {ticker_str} | {date_str})\n{r.text}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        f"You are a financial analyst assistant. Using ONLY the excerpts below, "
        f"answer the user's question concisely and cite excerpt numbers where relevant.\n\n"
        f"EXCERPTS:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )

    import anthropic

    loop = asyncio.get_event_loop()

    def _call() -> tuple[str, int]:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = msg.content[0].text if msg.content else ""
        tokens = msg.usage.input_tokens + msg.usage.output_tokens
        return answer, tokens

    return await loop.run_in_executor(None, _call)


# ── Main interface ─────────────────────────────────────────────────────────────

async def query(
    db_url: str,
    query_text: str,
    top_k: int = 10,
    ticker: str | None = None,
    doc_type: str | None = None,
    synthesize: bool = False,
    anthropic_api_key: str | None = None,
    expand_query: bool = True,
    use_hyde: bool = False,
) -> RAGResponse:
    """Full RAG pipeline: query expansion → dense + sparse → RRF → optional Claude synthesis."""
    retrieve_k = max(top_k * 2, 20)  # over-fetch for fusion quality

    # Query expansion (dim 56 — smart synonym / query expansion)
    effective_query = query_text
    if expand_query:
        try:
            from sentinel.sil.query_expander import expand_for_rag
            effective_query = await expand_for_rag(
                query=query_text,
                use_hyde=use_hyde,
            )
        except Exception as exc:
            logger.debug("Query expansion skipped", error=str(exc))

    dense_task = dense_retrieve(db_url, effective_query, retrieve_k, ticker, doc_type)
    sparse_task = sparse_retrieve(db_url, effective_query, retrieve_k, ticker, doc_type)

    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    top_results = fused[:top_k]

    logger.info(
        "rag.query",
        query=query_text[:80],
        dense_hits=len(dense_results),
        sparse_hits=len(sparse_results),
        fused=len(top_results),
        synthesize=synthesize,
    )

    answer: str | None = None
    tokens_used = 0

    if synthesize and top_results:
        key = anthropic_api_key or get_settings().anthropic_api_key
        if not key:
            logger.warning("synthesize=True but no Anthropic API key — skipping synthesis")
        else:
            try:
                answer, tokens_used = await _synthesize(query_text, top_results, key)
            except Exception as exc:
                logger.error("Claude synthesis failed", error=str(exc))

    return RAGResponse(
        query=query_text,
        results=top_results,
        answer=answer,
        tokens_used=tokens_used,
    )
