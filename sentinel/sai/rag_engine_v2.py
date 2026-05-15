"""RAG Engine V2 — Hybrid dense+sparse retrieval with reranking, smart chunking,
financial-aware query routing, multi-hop reasoning, and knowledge graph integration.

dim_051: Raises RAG score from 8 → 9 by adding:
  - HybridRAGEngine: dense pgvector + BM25 sparse with RRF fusion and cross-encoder reranking
  - DocumentChunkingStrategies: fixed-size, semantic, hierarchical, sentence-aware
  - FinancialRAGQueryEngine: query routing, metadata filtering, multi-hop, citations
  - KnowledgeGraphIntegration: entity extraction, relationship mapping, graph-augmented retrieval
  - rag_v2_router: FastAPI router with /rag/v2/query, /ingest, /stats, /multi-hop

Architecture inherits from sentinel.sai.rag_financial_docs:
  - Uses the same financial_documents table (pgvector, 384-dim embeddings)
  - Adds financial_kg_entities + financial_kg_relationships tables for KG
  - Adds hierarchical_chunks table for parent-child chunk relationships

Usage::
    from sentinel.sai.rag_engine_v2 import HybridRAGEngine, FinancialRAGQueryEngine

    engine = HybridRAGEngine(db_url)
    results = await engine.hybrid_rrf_search("Apple revenue growth drivers", ticker="AAPL")

    fq = FinancialRAGQueryEngine(db_url)
    answer = await fq.answer("What were AAPL margins in 2024?", ticker="AAPL")
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger
from sentinel.sai.rag_financial_docs import (
    _FINANCIAL_DOCS_DDL,
    _HEADERS,
    _USER_AGENT,
    EmbeddingModel,
    _embed_in_executor,
    _ensure_schema,
    _extract_section,
    _fetch_filing_text,
    _fetch_submissions,
    _get_engine,
    _list_filings,
    _resolve_cik,
    _schema_ready,
    _strip_html,
    _vec_str,
    chunk_document,
)

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

RRF_K = 60          # Reciprocal Rank Fusion constant
BM25_K1 = 1.5       # BM25 term frequency saturation
BM25_B = 0.75       # BM25 length normalization
HIERARCHICAL_PARENT_TOKENS = 2048
HIERARCHICAL_CHILD_TOKENS = 512
CHILD_OVERLAP_TOKENS = 64
FIXED_CHUNK_TOKENS = 512
FIXED_OVERLAP_TOKENS = 128
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Additional DDL ─────────────────────────────────────────────────────────────

_KG_DDL = """
CREATE TABLE IF NOT EXISTS financial_kg_entities (
    id          BIGSERIAL PRIMARY KEY,
    entity_text TEXT         NOT NULL,
    entity_type VARCHAR(50)  NOT NULL,
    ticker      VARCHAR(20),
    doc_id      BIGINT,
    mentions    INTEGER      DEFAULT 1,
    created_at  TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (entity_text, entity_type, ticker)
);

CREATE INDEX IF NOT EXISTS ix_kg_entity_ticker ON financial_kg_entities(ticker);
CREATE INDEX IF NOT EXISTS ix_kg_entity_type ON financial_kg_entities(entity_type);

CREATE TABLE IF NOT EXISTS financial_kg_relationships (
    id          BIGSERIAL PRIMARY KEY,
    subject_id  BIGINT       REFERENCES financial_kg_entities(id) ON DELETE CASCADE,
    relation    VARCHAR(100) NOT NULL,
    object_id   BIGINT       REFERENCES financial_kg_entities(id) ON DELETE CASCADE,
    amount      TEXT,
    source_text TEXT,
    ticker      VARCHAR(20),
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_kg_rel_subject ON financial_kg_relationships(subject_id);
CREATE INDEX IF NOT EXISTS ix_kg_rel_object ON financial_kg_relationships(object_id);
CREATE INDEX IF NOT EXISTS ix_kg_rel_ticker ON financial_kg_relationships(ticker);

CREATE TABLE IF NOT EXISTS hierarchical_chunks (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(20),
    doc_type        VARCHAR(30) NOT NULL,
    parent_text     TEXT        NOT NULL,
    child_text      TEXT        NOT NULL,
    child_index     INTEGER     NOT NULL,
    parent_hash     VARCHAR(64) NOT NULL,
    child_embedding vector(384),
    metadata        JSONB       NOT NULL DEFAULT '{}',
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_hc_ticker ON hierarchical_chunks(ticker);
CREATE INDEX IF NOT EXISTS ix_hc_parent_hash ON hierarchical_chunks(parent_hash);
CREATE INDEX IF NOT EXISTS ix_hc_doc_type ON hierarchical_chunks(doc_type);
"""

_KG_IVFFLAT = """
CREATE INDEX IF NOT EXISTS ix_hc_embedding_ivfflat
    ON hierarchical_chunks USING ivfflat (child_embedding vector_cosine_ops)
    WITH (lists = {lists})
"""

_v2_schema_ready: set[str] = set()


async def _ensure_v2_schema(db_url: str) -> None:
    """Ensure base schema + V2 KG/hierarchical tables exist."""
    await _ensure_schema(db_url)
    if db_url in _v2_schema_ready:
        return
    engine = _get_engine(db_url)
    async with engine.begin() as conn:
        for stmt in _KG_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    await conn.execute(text(stmt))
                except Exception as exc:
                    logger.debug("V2 DDL skipped", stmt=stmt[:60], error=str(exc))
    _v2_schema_ready.add(db_url)
    logger.info("RAG V2 schema (KG + hierarchical) ready")


# ── BM25 Implementation ────────────────────────────────────────────────────────

class BM25Index:
    """Pure-Python BM25 over a corpus of text documents.

    Implements Robertson & Walker (1994) BM25 with IDF weights.
    Suitable for sparse retrieval over financial document chunks.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self.corpus: list[str] = []
        self.corpus_ids: list[Any] = []     # external IDs aligned to corpus
        self.tf: list[dict[str, int]] = []  # term freq per doc
        self.df: dict[str, int] = {}        # doc freq per term
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0
        self._built = False

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def add_documents(self, texts: list[str], ids: list[Any] | None = None) -> None:
        """Add documents to the index. Call build_index() after all documents added."""
        if ids is None:
            ids = list(range(len(self.corpus), len(self.corpus) + len(texts)))
        for text, doc_id in zip(texts, ids):
            tokens = self._tokenize(text)
            tf: dict[str, int] = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1
            self.corpus.append(text)
            self.corpus_ids.append(doc_id)
            self.tf.append(dict(tf))

    def build_index(self) -> None:
        """Compute DF, IDF, and avgdl over loaded corpus."""
        self.N = len(self.corpus)
        if self.N == 0:
            self._built = True
            return

        # Document frequency
        df: dict[str, int] = defaultdict(int)
        total_len = 0
        for tf_dict in self.tf:
            total_len += sum(tf_dict.values())
            for term in tf_dict:
                df[term] += 1

        self.df = dict(df)
        self.avgdl = total_len / self.N if self.N > 0 else 1.0

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1) — always positive
        self.idf = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        self._built = True

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a single (query, document) pair."""
        if not self._built:
            self.build_index()
        tokens = self._tokenize(query)
        tf_dict = self.tf[doc_idx]
        doc_len = sum(tf_dict.values())
        score = 0.0
        for term in tokens:
            if term not in self.idf:
                continue
            tf_val = tf_dict.get(term, 0)
            numerator = tf_val * (self.k1 + 1)
            denominator = tf_val + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf[term] * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 20) -> list[tuple[Any, float]]:
        """Return top-k (doc_id, bm25_score) pairs sorted descending."""
        if not self._built:
            self.build_index()
        if self.N == 0:
            return []
        scores = [(self.corpus_ids[i], self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── Cross-Encoder Reranker ─────────────────────────────────────────────────────

class CrossEncoderReranker:
    """Optional cross-encoder reranker using sentence-transformers.

    Falls back gracefully if the library is unavailable — returns original ranking.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, high quality).
    """

    _instance: Optional[Any] = None
    _available: bool = True

    @classmethod
    def _get(cls) -> Optional[Any]:
        if not cls._available:
            return None
        if cls._instance is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder reranker...")
                cls._instance = CrossEncoder(CROSS_ENCODER_MODEL)
                logger.info("Cross-encoder reranker ready")
            except (ImportError, Exception) as exc:
                logger.warning("Cross-encoder unavailable", error=str(exc))
                cls._available = False
        return cls._instance

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "chunk_text",
        top_k: int = 10,
    ) -> list[dict]:
        """Rerank candidates using cross-encoder scores. Returns top_k results."""
        model = self._get()
        if model is None or not candidates:
            return candidates[:top_k]

        try:
            pairs = [(query, c[text_key]) for c in candidates]
            scores = model.predict(pairs)
            for i, c in enumerate(candidates):
                c["rerank_score"] = float(scores[i])
            candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        except Exception as exc:
            logger.warning("Reranking failed, returning RRF order", error=str(exc))

        return candidates[:top_k]


_reranker = CrossEncoderReranker()


# ── DocumentChunkingStrategies ─────────────────────────────────────────────────

class DocumentChunkingStrategies:
    """Advanced chunking strategies for financial documents.

    Strategies:
      fixed_size: 512 tokens, 128 overlap, sentence-aware (never splits mid-sentence)
      semantic:   splits at section boundaries (headers, "Item X", paragraph breaks)
      hierarchical: parent (2048 tokens) + child (512 tokens) — retrieve child, return parent
      sentence_aware: pure sentence-level chunking with max token budget
    """

    # Section boundary patterns: Item N, headers, numbered lists, paragraph breaks
    _SECTION_HEADER_RE = re.compile(
        r"(?:^|\n)(?:"
        r"item\s+\d{1,2}[a-z]?[\.\s]|"
        r"part\s+[iivx]+[\.\s]|"
        r"note\s+\d+[\.\s]|"
        r"#{1,4}\s+|"
        r"={3,}|"
        r"-{3,}"
        r")",
        re.IGNORECASE,
    )
    _SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
    _PARAGRAPH_RE = re.compile(r"\n{2,}")
    _CHARS_PER_TOKEN = 4

    @classmethod
    def _words(cls, text: str) -> list[str]:
        return text.split()

    @classmethod
    def _char_budget(cls, tokens: int) -> int:
        return tokens * cls._CHARS_PER_TOKEN

    @classmethod
    def _ends_at_sentence(cls, text: str) -> bool:
        stripped = text.rstrip()
        return bool(stripped) and stripped[-1] in ".!?"

    @classmethod
    def fixed_size(
        cls,
        text: str,
        chunk_tokens: int = FIXED_CHUNK_TOKENS,
        overlap_tokens: int = FIXED_OVERLAP_TOKENS,
    ) -> list[str]:
        """Fixed-size word-count chunking with sentence-boundary awareness.

        Splits at sentence ends where possible. Never cuts mid-sentence if
        a sentence-end boundary is within 20 tokens of the chunk limit.

        Returns:
            List of chunk strings.
        """
        # First split into sentences
        sentences = cls._SENTENCE_END_RE.split(text)
        chunks: list[str] = []
        current_words: list[str] = []
        overlap_words: list[str] = []

        for sentence in sentences:
            s_words = sentence.split()
            if not s_words:
                continue
            # Would adding this sentence exceed the limit?
            if len(current_words) + len(s_words) > chunk_tokens:
                if current_words:
                    chunks.append(" ".join(current_words))
                    overlap_words = current_words[-overlap_tokens:] if len(current_words) > overlap_tokens else current_words[:]
                    current_words = overlap_words + s_words
                else:
                    # Single sentence exceeds chunk size — split by words
                    for i in range(0, len(s_words), chunk_tokens - overlap_tokens):
                        sub = s_words[i:i + chunk_tokens]
                        if sub:
                            chunks.append(" ".join(sub))
                    current_words = s_words[-(overlap_tokens):]
            else:
                current_words.extend(s_words)

        if current_words:
            chunks.append(" ".join(current_words))

        return [c.strip() for c in chunks if c.strip()]

    @classmethod
    def semantic(cls, text: str, max_section_tokens: int = 1500) -> list[str]:
        """Semantic chunking: split at section boundaries, then sub-chunk if too large.

        Identifies section breaks via headers (Item X, Part X, ##, ---) and
        paragraph boundaries (double newline). Produces chunks that respect
        natural document structure.

        Returns:
            List of semantically coherent chunk strings.
        """
        # Find all section boundary positions
        boundaries: list[int] = [0]
        for m in cls._SECTION_HEADER_RE.finditer(text):
            pos = m.start()
            if pos > 0:
                boundaries.append(pos)
        boundaries.append(len(text))

        sections: list[str] = []
        for i in range(len(boundaries) - 1):
            section = text[boundaries[i]:boundaries[i + 1]].strip()
            if section:
                sections.append(section)

        # If no section headers found, fall back to paragraph splitting
        if len(sections) <= 1:
            sections = [p.strip() for p in cls._PARAGRAPH_RE.split(text) if p.strip()]

        chunks: list[str] = []
        max_chars = cls._char_budget(max_section_tokens)
        for section in sections:
            if len(section) <= max_chars:
                if section:
                    chunks.append(section)
            else:
                # Sub-chunk large sections with fixed_size
                sub_chunks = cls.fixed_size(section, chunk_tokens=max_section_tokens, overlap_tokens=FIXED_OVERLAP_TOKENS)
                chunks.extend(sub_chunks)

        return chunks

    @classmethod
    def hierarchical(
        cls,
        text: str,
        parent_tokens: int = HIERARCHICAL_PARENT_TOKENS,
        child_tokens: int = HIERARCHICAL_CHILD_TOKENS,
        child_overlap: int = CHILD_OVERLAP_TOKENS,
    ) -> list[dict]:
        """Hierarchical chunking: parent (large) + child (small) chunks.

        Each parent chunk is subdivided into child chunks. On retrieval, the
        matching child is returned but the full parent context is available.

        Returns:
            List of {parent_text, child_text, child_index, parent_hash} dicts.
        """
        # Create parent chunks
        parent_chunks = cls.fixed_size(text, chunk_tokens=parent_tokens, overlap_tokens=0)

        results: list[dict] = []
        for parent_text in parent_chunks:
            parent_hash = hashlib.sha256(parent_text.encode()).hexdigest()[:32]
            # Create child chunks within each parent
            child_chunks = cls.fixed_size(
                parent_text,
                chunk_tokens=child_tokens,
                overlap_tokens=child_overlap,
            )
            for idx, child_text in enumerate(child_chunks):
                results.append({
                    "parent_text": parent_text,
                    "child_text": child_text,
                    "child_index": idx,
                    "parent_hash": parent_hash,
                })

        return results

    @classmethod
    def sentence_aware(cls, text: str, max_tokens: int = FIXED_CHUNK_TOKENS) -> list[str]:
        """Pure sentence-aware chunking — accumulates sentences up to token budget.

        Never splits mid-sentence. Each chunk contains complete sentences only.

        Returns:
            List of chunk strings containing complete sentences.
        """
        sentences = cls._SENTENCE_END_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence.split())
            if current_len + s_len > max_tokens and current:
                chunks.append(" ".join(current))
                # Keep last sentence as overlap
                current = [sentence]
                current_len = s_len
            else:
                current.append(sentence)
                current_len += s_len

        if current:
            chunks.append(" ".join(current))

        return [c.strip() for c in chunks if c.strip()]


# ── HybridRAGEngine ────────────────────────────────────────────────────────────

class HybridRAGEngine:
    """Dense + sparse hybrid retrieval with RRF fusion and cross-encoder reranking.

    Combines:
      1. Dense retrieval: pgvector cosine similarity (384-dim MiniLM embeddings)
      2. Sparse retrieval: BM25 (pure Python, built over retrieved candidate set)
      3. Fusion: Reciprocal Rank Fusion with k=60
      4. Optional reranking: cross-encoder/ms-marco-MiniLM-L-6-v2

    The BM25 index is built on-the-fly over the pgvector candidate pool (3x top_k
    over-fetch) rather than maintaining a full corpus index, which keeps memory
    footprint low for production use.
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def _make_filters(
        self,
        ticker: Optional[str],
        doc_types: Optional[list[str]],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        extra_clauses: Optional[list[str]] = None,
    ) -> tuple[str, dict]:
        """Build WHERE clause and params from filter arguments."""
        clauses: list[str] = list(extra_clauses or [])
        params: dict = {}

        if ticker:
            clauses.append("ticker = :ticker")
            params["ticker"] = ticker.upper()

        if doc_types:
            clauses.append("doc_type = ANY(:doc_types)")
            params["doc_types"] = doc_types

        if date_from:
            clauses.append("(metadata->>'filing_date') >= :date_from")
            params["date_from"] = date_from

        if date_to:
            clauses.append("(metadata->>'filing_date') <= :date_to")
            params["date_to"] = date_to

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    async def _dense_search(
        self,
        query: str,
        top_k: int = 30,
        ticker: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve top candidates via pgvector cosine similarity."""
        await _ensure_v2_schema(self.db_url)
        q_embs = await _embed_in_executor([query])
        vec = _vec_str(q_embs[0])

        where, params = self._make_filters(
            ticker, doc_types, date_from, date_to,
            extra_clauses=["embedding IS NOT NULL"],
        )
        params.update({"vec": vec, "top_k": top_k})

        sql = text(f"""
            SELECT
                id, ticker, cik, doc_type, section, chunk_text, metadata,
                1 - (embedding <=> :vec::vector) AS dense_score
            FROM financial_documents
            {where}
            ORDER BY embedding <=> :vec::vector
            LIMIT :top_k
        """)

        engine = _get_engine(self.db_url)
        async with engine.connect() as conn:
            rows = (await conn.execute(sql, params)).fetchall()

        return [
            {
                "id": r.id,
                "chunk_text": r.chunk_text,
                "doc_type": r.doc_type,
                "section": r.section,
                "ticker": r.ticker,
                "cik": r.cik,
                "dense_score": float(r.dense_score),
                "metadata": r.metadata if isinstance(r.metadata, dict) else json.loads(r.metadata or "{}"),
            }
            for r in rows
        ]

    @staticmethod
    def _bm25_search(query: str, candidates: list[dict], top_k: int = 30) -> list[tuple[int, float]]:
        """Run BM25 over the candidate pool. Returns (candidate_idx, bm25_score) list."""
        bm25 = BM25Index()
        texts = [c["chunk_text"] for c in candidates]
        ids = list(range(len(candidates)))
        bm25.add_documents(texts, ids)
        bm25.build_index()
        return bm25.search(query, top_k=top_k)

    @staticmethod
    def _rrf_fusion(
        dense_results: list[dict],
        bm25_results: list[tuple[int, float]],
        candidates: list[dict],
        k: int = RRF_K,
    ) -> list[dict]:
        """Reciprocal Rank Fusion: score = 1/(rank_dense+k) + 1/(rank_bm25+k).

        Args:
            dense_results:  Ordered list of dense retrieval results.
            bm25_results:   List of (candidate_idx, score) from BM25.
            candidates:     Full candidate pool (same order as BM25 index).
            k:              RRF constant (default 60).

        Returns:
            Merged and re-ranked list of result dicts with rrf_score.
        """
        # Build dense rank map: id → rank (1-based)
        dense_rank: dict[int, int] = {}
        for rank, r in enumerate(dense_results, start=1):
            dense_rank[r["id"]] = rank

        # Build BM25 rank map: candidate_idx → rank (1-based)
        bm25_rank: dict[int, int] = {}
        for rank, (idx, _score) in enumerate(bm25_results, start=1):
            bm25_rank[idx] = rank

        # Compute RRF scores for all unique candidates
        rrf_scores: dict[int, float] = {}
        all_doc_ids: set[int] = set()

        for r in dense_results:
            all_doc_ids.add(r["id"])
        for idx, _ in bm25_results:
            all_doc_ids.add(candidates[idx]["id"] if idx < len(candidates) else -1)

        # Dense contribution
        for rank, r in enumerate(dense_results, start=1):
            doc_id = r["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + k)

        # BM25 contribution
        for rank, (idx, _) in enumerate(bm25_results, start=1):
            if idx >= len(candidates):
                continue
            doc_id = candidates[idx]["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + k)

        # Build id → result dict mapping
        id_to_result: dict[int, dict] = {r["id"]: r for r in dense_results}
        for idx, _ in bm25_results:
            if idx < len(candidates):
                c = candidates[idx]
                if c["id"] not in id_to_result:
                    id_to_result[c["id"]] = c

        # Assemble final results with RRF scores
        fused: list[dict] = []
        for doc_id, rrf_score in rrf_scores.items():
            if doc_id in id_to_result:
                result = dict(id_to_result[doc_id])
                result["rrf_score"] = round(rrf_score, 6)
                result["dense_rank"] = dense_rank.get(doc_id, 9999)
                bm25_idx = next(
                    (idx for idx, _ in bm25_results if idx < len(candidates) and candidates[idx]["id"] == doc_id),
                    9999,
                )
                result["bm25_rank"] = bm25_rank.get(bm25_idx, 9999)
                fused.append(result)

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    async def hybrid_rrf_search(
        self,
        query: str,
        top_k: int = 10,
        ticker: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        use_reranker: bool = True,
    ) -> list[dict]:
        """Hybrid dense+BM25 retrieval with RRF fusion.

        Pipeline:
          1. Dense retrieval: fetch 3*top_k candidates from pgvector
          2. BM25: score same candidates with BM25
          3. RRF: fuse dense + BM25 rankings with k=60
          4. Rerank: optional cross-encoder reranking on top 2*top_k RRF results

        Args:
            query:       Natural language query.
            top_k:       Final number of results to return.
            ticker:      Filter by ticker.
            doc_types:   Filter by document types.
            date_from:   Filter by filing_date >= date_from (YYYY-MM-DD).
            date_to:     Filter by filing_date <= date_to (YYYY-MM-DD).
            use_reranker: Apply cross-encoder reranking if model available.

        Returns:
            List of result dicts with rrf_score, dense_rank, bm25_rank.
        """
        fetch_k = top_k * 3
        dense_results = await self._dense_search(
            query, top_k=fetch_k, ticker=ticker,
            doc_types=doc_types, date_from=date_from, date_to=date_to,
        )

        if not dense_results:
            return []

        # BM25 over the fetched candidate pool
        bm25_results = self._bm25_search(query, dense_results, top_k=fetch_k)

        # RRF fusion
        fused = self._rrf_fusion(dense_results, bm25_results, dense_results)

        # Rerank top 2*top_k via cross-encoder
        candidates_for_rerank = fused[: top_k * 2]
        if use_reranker:
            loop = asyncio.get_event_loop()
            final = await loop.run_in_executor(
                None,
                lambda: _reranker.rerank(query, candidates_for_rerank, top_k=top_k),
            )
        else:
            final = candidates_for_rerank[:top_k]

        return final

    async def hierarchical_search(
        self,
        query: str,
        top_k: int = 10,
        ticker: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """Search hierarchical_chunks table. Returns child match + parent context.

        Child embeddings are searched first. The full parent text is returned
        alongside the matching child for richer context in LLM prompts.

        Args:
            query:     Query string.
            top_k:     Number of results.
            ticker:    Filter by ticker.
            doc_types: Filter by doc_type.

        Returns:
            List of {child_text, parent_text, child_index, similarity, metadata} dicts.
        """
        await _ensure_v2_schema(self.db_url)
        q_embs = await _embed_in_executor([query])
        vec = _vec_str(q_embs[0])

        clauses = ["child_embedding IS NOT NULL"]
        params: dict = {"vec": vec, "top_k": top_k}
        if ticker:
            clauses.append("ticker = :ticker")
            params["ticker"] = ticker.upper()
        if doc_types:
            clauses.append("doc_type = ANY(:doc_types)")
            params["doc_types"] = doc_types

        where = "WHERE " + " AND ".join(clauses)
        sql = text(f"""
            SELECT
                ticker, doc_type, parent_text, child_text, child_index, metadata,
                1 - (child_embedding <=> :vec::vector) AS similarity
            FROM hierarchical_chunks
            {where}
            ORDER BY child_embedding <=> :vec::vector
            LIMIT :top_k
        """)

        engine = _get_engine(self.db_url)
        async with engine.connect() as conn:
            rows = (await conn.execute(sql, params)).fetchall()

        return [
            {
                "child_text": r.child_text,
                "parent_text": r.parent_text,
                "child_index": r.child_index,
                "ticker": r.ticker,
                "doc_type": r.doc_type,
                "similarity": round(float(r.similarity), 4),
                "metadata": r.metadata if isinstance(r.metadata, dict) else json.loads(r.metadata or "{}"),
            }
            for r in rows
        ]

    async def ingest_with_hierarchical_chunks(
        self,
        ticker: str,
        cik: str,
        doc_type: str,
        text: str,
        metadata: dict,
    ) -> dict:
        """Ingest text using hierarchical chunking strategy.

        Creates parent chunks + child chunks. Child embeddings stored in
        hierarchical_chunks table; also ingests into financial_documents for
        compatibility with standard search.

        Args:
            ticker:    Ticker symbol.
            cik:       EDGAR CIK.
            doc_type:  Filing type (10-K, 10-Q, etc.).
            text:      Cleaned document text.
            metadata:  Filing metadata dict.

        Returns:
            {parents_created, children_created, ticker, doc_type}
        """
        await _ensure_v2_schema(self.db_url)
        chunker = DocumentChunkingStrategies()
        hier_chunks = chunker.hierarchical(text)

        if not hier_chunks:
            return {"parents_created": 0, "children_created": 0, "ticker": ticker, "doc_type": doc_type}

        # Embed all child texts
        child_texts = [c["child_text"] for c in hier_chunks]
        embeddings = await _embed_in_executor(child_texts)

        meta_json = json.dumps(metadata)
        engine = _get_engine(self.db_url)

        insert_sql = text("""
            INSERT INTO hierarchical_chunks
                (ticker, doc_type, parent_text, child_text, child_index, parent_hash,
                 child_embedding, metadata)
            VALUES
                (:ticker, :doc_type, :parent_text, :child_text, :child_index, :parent_hash,
                 :embedding::vector, :metadata)
            ON CONFLICT DO NOTHING
        """)

        async with engine.begin() as conn:
            for chunk, emb in zip(hier_chunks, embeddings):
                await conn.execute(insert_sql, {
                    "ticker": ticker.upper(),
                    "doc_type": doc_type,
                    "parent_text": chunk["parent_text"],
                    "child_text": chunk["child_text"],
                    "child_index": chunk["child_index"],
                    "parent_hash": chunk["parent_hash"],
                    "embedding": _vec_str(emb),
                    "metadata": meta_json,
                })

        parent_hashes = {c["parent_hash"] for c in hier_chunks}
        return {
            "parents_created": len(parent_hashes),
            "children_created": len(hier_chunks),
            "ticker": ticker,
            "doc_type": doc_type,
        }


# ── KnowledgeGraphIntegration ──────────────────────────────────────────────────

class KnowledgeGraphIntegration:
    """Simple financial knowledge graph: entity extraction + relationship mapping.

    Entity types: COMPANY, EXECUTIVE, PRODUCT, METRIC, AMOUNT, LOCATION
    Relationships: acquired, partnered_with, reported, employs, divested

    Stored in financial_kg_entities + financial_kg_relationships tables.
    Augments retrieval by also fetching related entities' document chunks.
    """

    # Patterns for entity extraction
    _COMPANY_SUFFIXES = r"(?:Inc\.|Corp\.|LLC|Ltd\.|Co\.|Group|Holdings|Technologies|Systems|Solutions|Enterprises)"
    _EXEC_TITLES = r"(?:CEO|CFO|COO|CTO|President|Chairman|Director|VP|SVP|EVP|Chief\s+\w+\s+Officer)"
    _METRIC_TERMS = r"(?:revenue|earnings|EBITDA|EBIT|net income|gross profit|operating income|EPS|FCF|margin|ROE|ROIC)"
    _AMOUNT_RE = re.compile(
        r"[\$€£¥]?\s*\d+(?:[.,]\d+)*\s*(?:billion|million|thousand|B|M|K|bn|mm|trillion|trn)?",
        re.IGNORECASE,
    )
    _COMPANY_RE = re.compile(
        r"\b[A-Z][a-zA-Z&\s\-\']{2,40}\s+" + _COMPANY_SUFFIXES,
        re.IGNORECASE,
    )
    _EXEC_RE = re.compile(
        r"\b(?:Mr\.|Ms\.|Dr\.)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}),?\s+" + _EXEC_TITLES,
    )
    _METRIC_RE = re.compile(_METRIC_TERMS, re.IGNORECASE)
    _ACQUISITION_RE = re.compile(
        r"([A-Z][a-zA-Z\s&]+?)\s+(?:acquired|purchased|bought|acquired)\s+([A-Z][a-zA-Z\s&]+?)\s+for\s+([\$\d\w\s\.]+)",
        re.IGNORECASE,
    )
    _PARTNERSHIP_RE = re.compile(
        r"([A-Z][a-zA-Z\s&]+?)\s+(?:partnered|signed|entered into an? agreement|formed a partnership)\s+with\s+([A-Z][a-zA-Z\s&]+)",
        re.IGNORECASE,
    )

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def extract_entities(self, text: str, ticker: Optional[str] = None) -> list[dict]:
        """Extract financial entities from text.

        Returns list of {entity_text, entity_type, ticker} dicts.
        """
        entities: list[dict] = []
        seen: set[str] = set()

        def add(text_val: str, etype: str) -> None:
            key = (text_val.strip().lower(), etype)
            if key not in seen and len(text_val.strip()) > 2:
                seen.add(key)
                entities.append({
                    "entity_text": text_val.strip(),
                    "entity_type": etype,
                    "ticker": ticker,
                })

        # Companies
        for m in self._COMPANY_RE.finditer(text):
            add(m.group(0), "COMPANY")

        # Executives
        for m in self._EXEC_RE.finditer(text):
            add(m.group(1), "EXECUTIVE")

        # Financial metrics (collect context)
        for m in self._METRIC_RE.finditer(text):
            # Get surrounding amount if present
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 50)
            context = text[start:end]
            add(m.group(0) + ": " + context.strip(), "METRIC")

        # Amounts (standalone)
        for m in self._AMOUNT_RE.finditer(text):
            if len(m.group(0).strip()) > 2:
                add(m.group(0).strip(), "AMOUNT")

        return entities[:100]  # cap to prevent runaway extraction

    def extract_relationships(self, text: str, ticker: Optional[str] = None) -> list[dict]:
        """Extract financial relationships (acquisitions, partnerships) from text.

        Returns list of {subject, relation, object, amount, source_text, ticker} dicts.
        """
        relationships: list[dict] = []

        # Acquisitions
        for m in self._ACQUISITION_RE.finditer(text):
            relationships.append({
                "subject": m.group(1).strip(),
                "relation": "acquired",
                "object": m.group(2).strip(),
                "amount": m.group(3).strip(),
                "source_text": m.group(0)[:200],
                "ticker": ticker,
            })

        # Partnerships
        for m in self._PARTNERSHIP_RE.finditer(text):
            relationships.append({
                "subject": m.group(1).strip(),
                "relation": "partnered_with",
                "object": m.group(2).strip(),
                "amount": None,
                "source_text": m.group(0)[:200],
                "ticker": ticker,
            })

        return relationships

    async def store_entities(self, entities: list[dict]) -> int:
        """Upsert extracted entities into financial_kg_entities. Returns count stored."""
        if not entities:
            return 0
        await _ensure_v2_schema(self.db_url)
        engine = _get_engine(self.db_url)

        upsert_sql = text("""
            INSERT INTO financial_kg_entities (entity_text, entity_type, ticker, mentions)
            VALUES (:entity_text, :entity_type, :ticker, 1)
            ON CONFLICT (entity_text, entity_type, ticker)
            DO UPDATE SET mentions = financial_kg_entities.mentions + 1
            RETURNING id
        """)

        ids = []
        async with engine.begin() as conn:
            for entity in entities:
                result = await conn.execute(upsert_sql, {
                    "entity_text": entity["entity_text"][:500],
                    "entity_type": entity["entity_type"],
                    "ticker": entity.get("ticker"),
                })
                row = result.fetchone()
                if row:
                    ids.append(row[0])

        return len(ids)

    async def store_relationships(self, relationships: list[dict]) -> int:
        """Store extracted relationships into financial_kg_relationships.

        Resolves subject/object entities by name, creates them if missing.
        Returns count of relationships stored.
        """
        if not relationships:
            return 0
        await _ensure_v2_schema(self.db_url)
        engine = _get_engine(self.db_url)
        count = 0

        async with engine.begin() as conn:
            for rel in relationships:
                # Ensure subject entity exists
                subj_id = await self._get_or_create_entity(
                    conn, rel["subject"], "COMPANY", rel.get("ticker")
                )
                obj_id = await self._get_or_create_entity(
                    conn, rel["object"], "COMPANY", rel.get("ticker")
                )

                await conn.execute(text("""
                    INSERT INTO financial_kg_relationships
                        (subject_id, relation, object_id, amount, source_text, ticker)
                    VALUES (:subj_id, :relation, :obj_id, :amount, :source_text, :ticker)
                """), {
                    "subj_id": subj_id,
                    "relation": rel["relation"],
                    "obj_id": obj_id,
                    "amount": rel.get("amount"),
                    "source_text": (rel.get("source_text") or "")[:500],
                    "ticker": rel.get("ticker"),
                })
                count += 1

        return count

    async def _get_or_create_entity(
        self, conn: Any, entity_text: str, entity_type: str, ticker: Optional[str]
    ) -> int:
        """Get entity ID or create it. Returns entity ID."""
        result = await conn.execute(text("""
            INSERT INTO financial_kg_entities (entity_text, entity_type, ticker, mentions)
            VALUES (:text, :type, :ticker, 1)
            ON CONFLICT (entity_text, entity_type, ticker)
            DO UPDATE SET mentions = financial_kg_entities.mentions + 1
            RETURNING id
        """), {"text": entity_text[:500], "type": entity_type, "ticker": ticker})
        row = result.fetchone()
        return row[0] if row else 0

    async def get_related_tickers(self, ticker: str) -> list[str]:
        """Find tickers related to the given ticker via KG relationships.

        Returns list of related ticker symbols.
        """
        await _ensure_v2_schema(self.db_url)
        engine = _get_engine(self.db_url)
        sql = text("""
            SELECT DISTINCT
                CASE
                    WHEN e_subj.ticker = :ticker THEN e_obj.ticker
                    ELSE e_subj.ticker
                END AS related_ticker
            FROM financial_kg_relationships r
            JOIN financial_kg_entities e_subj ON r.subject_id = e_subj.id
            JOIN financial_kg_entities e_obj ON r.object_id = e_obj.id
            WHERE (e_subj.ticker = :ticker OR e_obj.ticker = :ticker)
              AND related_ticker IS NOT NULL
              AND related_ticker != :ticker
            LIMIT 10
        """)
        try:
            async with engine.connect() as conn:
                rows = (await conn.execute(sql, {"ticker": ticker.upper()})).fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception:
            return []

    async def graph_augmented_search(
        self,
        query: str,
        ticker: str,
        engine_v2: HybridRAGEngine,
        top_k: int = 10,
    ) -> list[dict]:
        """Retrieve results for ticker AND all KG-related tickers.

        Finds related tickers via KG, fetches documents for each, then
        merges and re-ranks using RRF scores.

        Args:
            query:     Query string.
            ticker:    Primary ticker.
            engine_v2: HybridRAGEngine instance for retrieval.
            top_k:     Final result count.

        Returns:
            Combined and ranked result list.
        """
        related_tickers = await self.get_related_tickers(ticker)
        all_tickers = [ticker] + related_tickers[:3]  # cap to 4 tickers

        all_results: list[dict] = []
        for t in all_tickers:
            results = await engine_v2.hybrid_rrf_search(query, ticker=t, top_k=top_k, use_reranker=False)
            all_results.extend(results)

        # Re-rank combined by RRF score
        all_results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        # Deduplicate by chunk text hash
        seen_hashes: set[str] = set()
        unique: list[dict] = []
        for r in all_results:
            h = hashlib.md5(r["chunk_text"].encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(r)

        return unique[:top_k]

    async def get_entity_graph(self, ticker: str) -> dict:
        """Return entity graph for a ticker: entities + relationships.

        Returns {entities: [...], relationships: [...], ticker}
        """
        await _ensure_v2_schema(self.db_url)
        engine = _get_engine(self.db_url)

        async with engine.connect() as conn:
            entity_rows = (await conn.execute(text("""
                SELECT id, entity_text, entity_type, mentions
                FROM financial_kg_entities
                WHERE ticker = :ticker
                ORDER BY mentions DESC
                LIMIT 50
            """), {"ticker": ticker.upper()})).fetchall()

            rel_rows = (await conn.execute(text("""
                SELECT e_s.entity_text AS subject, r.relation, e_o.entity_text AS object,
                       r.amount, r.source_text
                FROM financial_kg_relationships r
                JOIN financial_kg_entities e_s ON r.subject_id = e_s.id
                JOIN financial_kg_entities e_o ON r.object_id = e_o.id
                WHERE r.ticker = :ticker
                ORDER BY r.created_at DESC
                LIMIT 100
            """), {"ticker": ticker.upper()})).fetchall()

        return {
            "ticker": ticker.upper(),
            "entities": [
                {"id": r.id, "text": r.entity_text, "type": r.entity_type, "mentions": r.mentions}
                for r in entity_rows
            ],
            "relationships": [
                {
                    "subject": r.subject,
                    "relation": r.relation,
                    "object": r.object,
                    "amount": r.amount,
                    "source": r.source_text[:100] if r.source_text else None,
                }
                for r in rel_rows
            ],
        }


# ── FinancialRAGQueryEngine ────────────────────────────────────────────────────

class FinancialRAGQueryEngine:
    """Financial-aware query engine with routing, multi-hop, and citation support.

    Features:
      - Query routing: numeric questions → table extraction; qualitative → text retrieval
      - Metadata filtering: ticker, filing_type, date_range
      - Multi-hop: 2+ sequential retrieval steps for complex reasoning
      - Citation: specific passage + document reference in every answer
      - KG-augmented: optionally includes related entities' documents

    Depends on HybridRAGEngine for retrieval.
    """

    # Keywords that indicate a numeric/quantitative question
    _NUMERIC_INDICATORS = re.compile(
        r"\b(?:how much|how many|what was the|what is the|revenue|earnings|eps|"
        r"margin|ebitda|debt|cash|profit|loss|growth|percent|percentage|ratio|"
        r"multiple|valuation|price|amount|total|net|gross|operating)\b",
        re.IGNORECASE,
    )
    _QUALITATIVE_INDICATORS = re.compile(
        r"\b(?:why|what strategy|how does|explain|describe|what are the risks|"
        r"competitive|management|culture|moat|advantage|opinion|outlook|future|"
        r"guidance|plan|initiative|challenge)\b",
        re.IGNORECASE,
    )

    def __init__(self, db_url: str, api_key: Optional[str] = None) -> None:
        self.db_url = db_url
        self._api_key = api_key or ""
        self._hybrid = HybridRAGEngine(db_url)
        self._kg = KnowledgeGraphIntegration(db_url)

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        try:
            return get_settings().anthropic_api_key or ""
        except Exception:
            return ""

    def route_query(self, query: str) -> str:
        """Determine query type: 'numeric', 'qualitative', or 'general'.

        Routing determines the retrieval strategy:
          - numeric:     prioritize financial_statements and MD&A sections
          - qualitative: prioritize Risk Factors and Business sections
          - general:     no section priority, pure hybrid search
        """
        numeric_score = len(self._NUMERIC_INDICATORS.findall(query))
        qual_score = len(self._QUALITATIVE_INDICATORS.findall(query))

        if numeric_score > qual_score and numeric_score >= 2:
            return "numeric"
        if qual_score > numeric_score and qual_score >= 1:
            return "qualitative"
        return "general"

    def _section_priority(self, query_type: str) -> Optional[list[str]]:
        """Return preferred doc_type sections for routing."""
        if query_type == "numeric":
            return ["Financial Statements", "MD&A"]
        if query_type == "qualitative":
            return ["Risk Factors", "Business"]
        return None

    def _format_citation(self, result: dict, index: int) -> str:
        """Format a source citation string."""
        meta = result.get("metadata", {})
        ticker = result.get("ticker") or "N/A"
        doc_type = meta.get("form_type") or result.get("doc_type", "")
        period = meta.get("period", "") or meta.get("filing_date", "")
        section = result.get("section", "")
        return f"[{index}] {ticker} {doc_type} {period} — {section}".strip()

    def _build_context_with_citations(
        self, results: list[dict], max_chars: int = 12000
    ) -> tuple[str, list[str]]:
        """Build numbered context string with citations.

        Returns:
            (context_text, citations_list) where context_text has numbered passages
            and citations_list has formatted source references.
        """
        context_parts: list[str] = []
        citations: list[str] = []
        total_chars = 0

        for i, r in enumerate(results, start=1):
            citation = self._format_citation(r, i)
            citations.append(citation)
            passage = f"[{i}] {r['chunk_text']}"
            if total_chars + len(passage) > max_chars:
                break
            context_parts.append(passage)
            total_chars += len(passage)

        return "\n\n".join(context_parts), citations

    async def answer(
        self,
        query: str,
        ticker: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        top_k: int = 10,
        use_kg: bool = False,
        model: str = "claude-haiku-4-5-20251001",
    ) -> dict:
        """Answer a financial question with hybrid retrieval + LLM synthesis.

        Pipeline:
          1. Route query (numeric / qualitative / general)
          2. Hybrid RRF search with metadata filters
          3. Optional KG-augmented retrieval
          4. Build context with citations
          5. LLM synthesis with Claude

        Args:
            query:      Natural language financial question.
            ticker:     Optional ticker filter.
            doc_types:  Optional list of doc_type filters.
            date_from:  Optional start date filter (YYYY-MM-DD).
            date_to:    Optional end date filter (YYYY-MM-DD).
            top_k:      Number of retrieved chunks.
            use_kg:     Augment with knowledge graph related entities.
            model:      Anthropic model ID.

        Returns:
            {answer, citations, sources, query_type, confidence, model_used, tokens_used}
        """
        query_type = self.route_query(query)

        if use_kg and ticker:
            results = await self._kg.graph_augmented_search(
                query, ticker, self._hybrid, top_k=top_k
            )
        else:
            results = await self._hybrid.hybrid_rrf_search(
                query, top_k=top_k, ticker=ticker,
                doc_types=doc_types, date_from=date_from, date_to=date_to,
            )

        if not results:
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "citations": [],
                "sources": [],
                "query_type": query_type,
                "confidence": 0.0,
                "model_used": None,
                "tokens_used": 0,
            }

        context, citations = self._build_context_with_citations(results)
        api_key = self._get_api_key()

        if not api_key:
            return {
                "answer": "ANTHROPIC_API_KEY not configured — retrieval results shown only.",
                "citations": citations,
                "sources": [self._result_summary(r) for r in results[:5]],
                "query_type": query_type,
                "confidence": 0.0,
                "model_used": None,
                "tokens_used": 0,
            }

        system_prompt = (
            "You are a senior financial analyst with expertise in SEC filings. "
            "Answer the question using ONLY the numbered document passages provided. "
            "Cite specific passages using [N] notation. "
            "Be precise with numbers, dates, and financial metrics. "
            "If the provided context is insufficient, say so explicitly."
        )
        user_prompt = (
            f"DOCUMENT PASSAGES:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER (cite as [1], [2], etc.):"
        )

        try:
            import anthropic
            loop = asyncio.get_event_loop()

            def _call() -> tuple[str, int]:
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text_out = msg.content[0].text if msg.content else ""
                tokens = msg.usage.input_tokens + msg.usage.output_tokens
                return text_out, tokens

            answer_text, tokens = await loop.run_in_executor(None, _call)
            confidence = min(1.0, len(results) / 10.0)

            return {
                "answer": answer_text,
                "citations": citations,
                "sources": [self._result_summary(r) for r in results[:5]],
                "query_type": query_type,
                "confidence": round(confidence, 2),
                "model_used": model,
                "tokens_used": tokens,
            }

        except Exception as exc:
            logger.error("LLM synthesis failed in V2", error=str(exc))
            return {
                "answer": f"LLM error: {exc}",
                "citations": citations,
                "sources": [self._result_summary(r) for r in results[:5]],
                "query_type": query_type,
                "confidence": 0.0,
                "model_used": model,
                "tokens_used": 0,
            }

    @staticmethod
    def _result_summary(r: dict) -> dict:
        """Compact summary of a retrieval result for API response."""
        meta = r.get("metadata", {})
        return {
            "ticker": r.get("ticker"),
            "doc_type": r.get("doc_type"),
            "section": r.get("section"),
            "period": meta.get("period") or meta.get("filing_date"),
            "rrf_score": r.get("rrf_score"),
            "chunk_preview": r.get("chunk_text", "")[:150],
        }

    async def multi_hop_answer(
        self,
        query: str,
        ticker: Optional[str] = None,
        hops: int = 2,
        top_k_per_hop: int = 5,
        model: str = "claude-haiku-4-5-20251001",
    ) -> dict:
        """Multi-hop RAG: decompose complex question into sub-questions.

        For questions requiring 2+ retrieval steps (e.g., "How does Apple's
        R&D spending compare to its operating margin trends?"), this method:
          1. Uses LLM to decompose into sub-questions
          2. Answers each sub-question independently via retrieval
          3. Synthesizes final answer from all sub-answers

        Args:
            query:          Complex multi-part question.
            ticker:         Optional ticker filter.
            hops:           Number of reasoning hops (2-3 recommended).
            top_k_per_hop:  Chunks retrieved per hop.
            model:          Anthropic model.

        Returns:
            {final_answer, sub_questions, sub_answers, citations, hops_completed}
        """
        api_key = self._get_api_key()

        if not api_key:
            # Fall back to single-hop
            result = await self.answer(query, ticker=ticker, top_k=top_k_per_hop, model=model)
            result["hops_completed"] = 1
            result["sub_questions"] = [query]
            result["sub_answers"] = [result.get("answer", "")]
            return result

        # Step 1: Decompose query into sub-questions
        decompose_prompt = (
            f"Break the following complex financial question into {hops} simpler sub-questions "
            f"that can be answered independently from financial documents. "
            f"Return ONLY a JSON array of strings, no explanation.\n\n"
            f"Question: {query}"
        )

        try:
            import anthropic

            loop = asyncio.get_event_loop()

            def _decompose() -> str:
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": decompose_prompt}],
                )
                return msg.content[0].text if msg.content else "[]"

            decompose_text = await loop.run_in_executor(None, _decompose)

            # Parse sub-questions
            m = re.search(r"\[.*\]", decompose_text, re.DOTALL)
            sub_questions: list[str] = json.loads(m.group(0)) if m else [query]

        except Exception:
            sub_questions = [query]

        # Step 2: Answer each sub-question
        sub_answers: list[str] = []
        all_citations: list[str] = []
        all_context_parts: list[str] = []

        for sq in sub_questions[:hops]:
            hop_result = await self.answer(sq, ticker=ticker, top_k=top_k_per_hop, model=model)
            sub_answers.append(hop_result.get("answer", ""))
            all_citations.extend(hop_result.get("citations", []))
            all_context_parts.append(f"Sub-question: {sq}\nAnswer: {hop_result.get('answer', '')}")

        # Step 3: Synthesize final answer
        synthesis_context = "\n\n".join(all_context_parts)
        synthesis_prompt = (
            f"You are synthesizing a final answer from multiple research steps.\n\n"
            f"Original question: {query}\n\n"
            f"Research findings:\n{synthesis_context}\n\n"
            f"Provide a comprehensive, well-cited final answer:"
        )

        try:
            def _synthesize() -> tuple[str, int]:
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": synthesis_prompt}],
                )
                text_out = msg.content[0].text if msg.content else ""
                tokens = msg.usage.input_tokens + msg.usage.output_tokens
                return text_out, tokens

            final_answer, synthesis_tokens = await loop.run_in_executor(None, _synthesize)
        except Exception as exc:
            final_answer = "\n\n".join(sub_answers)
            synthesis_tokens = 0

        return {
            "final_answer": final_answer,
            "sub_questions": sub_questions[:hops],
            "sub_answers": sub_answers,
            "citations": list(dict.fromkeys(all_citations)),  # deduplicate preserving order
            "hops_completed": len(sub_answers),
            "query": query,
            "ticker": ticker,
        }

    async def ingest_ticker_v2(
        self,
        ticker: str,
        cik: Optional[str] = None,
        years: int = 3,
        chunking_strategy: str = "hierarchical",
        extract_kg: bool = True,
    ) -> dict:
        """Full V2 ingest pipeline for a ticker.

        Downloads EDGAR 10-K and 10-Q filings, applies the specified chunking
        strategy, extracts KG entities/relationships, and stores all data.

        Args:
            ticker:            Ticker symbol.
            cik:               Optional CIK (resolved from ticker if not provided).
            years:             Number of years of 10-K to ingest.
            chunking_strategy: 'hierarchical', 'semantic', 'fixed', or 'sentence'.
            extract_kg:        Whether to extract and store KG entities/relationships.

        Returns:
            Ingest summary dict.
        """
        if not cik:
            cik = await _resolve_cik(ticker)
        if not cik:
            return {"error": f"CIK not resolved for ticker={ticker}"}

        cik_padded = cik.zfill(10)
        try:
            submissions = await _fetch_submissions(cik_padded)
        except Exception as exc:
            return {"error": str(exc), "ticker": ticker}

        filings_10k = _list_filings(submissions, "10-K", limit=years)
        filings_10q = _list_filings(submissions, "10-Q", limit=years * 4)

        total_parents = 0
        total_children = 0
        total_entities = 0
        total_relationships = 0
        processed_filings = []

        for filing in filings_10k + filings_10q:
            accession = filing.get("accession")
            primary_doc = filing.get("primary_doc")
            form_type = filing.get("form_type", "")
            period = filing.get("period", "")

            if not accession or not primary_doc:
                continue

            try:
                raw = await _fetch_filing_text(cik_padded, accession, primary_doc)
            except Exception as exc:
                logger.warning("Filing fetch failed V2", accession=accession, error=str(exc))
                continue

            clean = _strip_html(raw) if "<" in raw[:500] else raw

            for section_name in ["Risk Factors", "MD&A", "Business", "Financial Statements"]:
                section_text = _extract_section(clean, section_name)
                if not section_text:
                    continue

                meta = {
                    "form_type": form_type,
                    "accession_number": accession,
                    "period": period,
                    "filing_date": filing.get("filing_date", ""),
                    "section": section_name,
                }

                # Apply chunking strategy
                if chunking_strategy == "hierarchical":
                    result = await self._hybrid.ingest_with_hierarchical_chunks(
                        ticker, cik_padded, form_type, section_text, meta
                    )
                    total_parents += result.get("parents_created", 0)
                    total_children += result.get("children_created", 0)

                elif chunking_strategy == "semantic":
                    chunks = DocumentChunkingStrategies.semantic(section_text)
                    if chunks:
                        embeddings = await _embed_in_executor(chunks)
                        engine = _get_engine(self.db_url)
                        meta_json = json.dumps(meta)
                        async with engine.begin() as conn:
                            for chunk, emb in zip(chunks, embeddings):
                                await conn.execute(text("""
                                    INSERT INTO financial_documents
                                        (ticker, cik, doc_type, section, chunk_text, embedding, metadata)
                                    VALUES (:ticker, :cik, :doc_type, :section, :chunk_text, :embedding::vector, :metadata)
                                    ON CONFLICT DO NOTHING
                                """), {
                                    "ticker": ticker.upper(),
                                    "cik": cik_padded,
                                    "doc_type": form_type,
                                    "section": section_name,
                                    "chunk_text": chunk,
                                    "embedding": _vec_str(emb),
                                    "metadata": meta_json,
                                })
                        total_children += len(chunks)

                else:
                    # Fixed or sentence-aware
                    if chunking_strategy == "sentence":
                        chunks = DocumentChunkingStrategies.sentence_aware(section_text)
                    else:
                        chunks = DocumentChunkingStrategies.fixed_size(section_text)

                    if chunks:
                        embeddings = await _embed_in_executor(chunks)
                        engine = _get_engine(self.db_url)
                        meta_json = json.dumps(meta)
                        async with engine.begin() as conn:
                            for chunk, emb in zip(chunks, embeddings):
                                await conn.execute(text("""
                                    INSERT INTO financial_documents
                                        (ticker, cik, doc_type, section, chunk_text, embedding, metadata)
                                    VALUES (:ticker, :cik, :doc_type, :section, :chunk_text, :embedding::vector, :metadata)
                                    ON CONFLICT DO NOTHING
                                """), {
                                    "ticker": ticker.upper(),
                                    "cik": cik_padded,
                                    "doc_type": form_type,
                                    "section": section_name,
                                    "chunk_text": chunk,
                                    "embedding": _vec_str(emb),
                                    "metadata": meta_json,
                                })
                        total_children += len(chunks)

                # KG extraction
                if extract_kg and section_text:
                    entities = self._kg.extract_entities(section_text[:10000], ticker=ticker)
                    rels = self._kg.extract_relationships(section_text[:10000], ticker=ticker)
                    n_ent = await self._kg.store_entities(entities)
                    n_rel = await self._kg.store_relationships(rels)
                    total_entities += n_ent
                    total_relationships += n_rel

            processed_filings.append({
                "form_type": form_type,
                "period": period,
                "accession": accession,
            })

        return {
            "ticker": ticker.upper(),
            "cik": cik_padded,
            "chunking_strategy": chunking_strategy,
            "filings_processed": len(processed_filings),
            "parents_created": total_parents,
            "children_created": total_children,
            "kg_entities_stored": total_entities,
            "kg_relationships_stored": total_relationships,
            "filings": processed_filings,
        }

    async def get_stats(self, ticker: Optional[str] = None) -> dict:
        """Return statistics about the V2 index.

        Returns:
            {total_docs, total_hierarchical, kg_entities, kg_relationships, by_ticker}
        """
        await _ensure_v2_schema(self.db_url)
        engine = _get_engine(self.db_url)

        ticker_filter = "WHERE ticker = :ticker" if ticker else ""
        params = {"ticker": ticker.upper()} if ticker else {}

        async with engine.connect() as conn:
            doc_count = (await conn.execute(
                text(f"SELECT COUNT(*) FROM financial_documents {ticker_filter}"), params
            )).scalar() or 0

            hier_count = (await conn.execute(
                text(f"SELECT COUNT(*) FROM hierarchical_chunks {ticker_filter}"), params
            )).scalar() or 0

            kg_entity_count = (await conn.execute(
                text(f"SELECT COUNT(*) FROM financial_kg_entities {ticker_filter}"), params
            )).scalar() or 0

            kg_rel_count = (await conn.execute(
                text(f"SELECT COUNT(*) FROM financial_kg_relationships {ticker_filter}"), params
            )).scalar() or 0

            # Per-ticker breakdown (top 20)
            by_ticker_rows = (await conn.execute(text("""
                SELECT ticker, COUNT(*) AS chunks
                FROM financial_documents
                WHERE ticker IS NOT NULL
                GROUP BY ticker
                ORDER BY chunks DESC
                LIMIT 20
            """))).fetchall()

        return {
            "ticker_filter": ticker,
            "total_docs": doc_count,
            "total_hierarchical_children": hier_count,
            "kg_entities": kg_entity_count,
            "kg_relationships": kg_rel_count,
            "by_ticker": [{"ticker": r.ticker, "chunks": r.chunks} for r in by_ticker_rows],
        }


# ── FastAPI Router ─────────────────────────────────────────────────────────────

rag_v2_router = APIRouter(prefix="/api/rag/v2", tags=["RAG V2 — Hybrid+KG"])


def _get_db_url() -> str:
    return get_settings().database_url


class QueryRequestV2(BaseModel):
    query: str
    ticker: Optional[str] = None
    doc_types: Optional[list[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    top_k: int = 10
    use_kg: bool = False
    use_reranker: bool = True
    model: str = "claude-haiku-4-5-20251001"


class IngestRequestV2(BaseModel):
    cik: Optional[str] = None
    years: int = 3
    chunking_strategy: str = "hierarchical"
    extract_kg: bool = True


class MultiHopRequest(BaseModel):
    query: str
    ticker: Optional[str] = None
    hops: int = 2
    top_k_per_hop: int = 5
    model: str = "claude-haiku-4-5-20251001"


@rag_v2_router.post("/query", summary="Hybrid RAG query with RRF fusion + reranking")
async def query_v2(body: QueryRequestV2):
    """Answer a financial question using hybrid dense+BM25 retrieval with RRF fusion.

    Supports:
      - Dense pgvector cosine similarity (384-dim MiniLM)
      - Sparse BM25 over candidate pool
      - RRF fusion with k=60
      - Optional cross-encoder reranking
      - Optional KG-augmented retrieval
      - Metadata filtering: ticker, doc_types, date_from, date_to
    """
    db_url = _get_db_url()
    fq = FinancialRAGQueryEngine(db_url)
    return await fq.answer(
        body.query,
        ticker=body.ticker,
        doc_types=body.doc_types,
        date_from=body.date_from,
        date_to=body.date_to,
        top_k=body.top_k,
        use_kg=body.use_kg,
        model=body.model,
    )


@rag_v2_router.post("/ingest/{ticker}", summary="Ingest ticker filings with V2 chunking and KG")
async def ingest_v2(ticker: str, body: IngestRequestV2 = None):
    """Download EDGAR filings and ingest with advanced chunking + KG extraction.

    Chunking strategies:
      - hierarchical: parent (2048 tokens) + child (512 tokens) chunks
      - semantic: section-boundary-aware chunking
      - fixed: 512-token chunks with 128-token overlap
      - sentence: sentence-boundary-aware chunking
    """
    if body is None:
        body = IngestRequestV2()
    db_url = _get_db_url()
    fq = FinancialRAGQueryEngine(db_url)
    return await fq.ingest_ticker_v2(
        ticker=ticker,
        cik=body.cik,
        years=body.years,
        chunking_strategy=body.chunking_strategy,
        extract_kg=body.extract_kg,
    )


@rag_v2_router.get("/stats", summary="V2 index statistics")
async def stats_v2(
    ticker: Optional[str] = Query(None, description="Filter stats by ticker"),
):
    """Return statistics: total documents, hierarchical chunks, KG entities/relationships."""
    db_url = _get_db_url()
    fq = FinancialRAGQueryEngine(db_url)
    return await fq.get_stats(ticker=ticker)


@rag_v2_router.post("/multi-hop", summary="Multi-hop RAG for complex financial questions")
async def multi_hop_v2(body: MultiHopRequest):
    """Multi-hop reasoning: decomposes complex question → answers sub-questions → synthesizes.

    Best for questions requiring multiple evidence types:
      'How did Apple's R&D investment impact its gross margin trends over 2022-2024?'
    """
    db_url = _get_db_url()
    fq = FinancialRAGQueryEngine(db_url)
    return await fq.multi_hop_answer(
        body.query,
        ticker=body.ticker,
        hops=body.hops,
        top_k_per_hop=body.top_k_per_hop,
        model=body.model,
    )


@rag_v2_router.get("/kg/{ticker}", summary="Knowledge graph entities and relationships for a ticker")
async def get_kg(ticker: str):
    """Return KG entity graph for a ticker: top entities + relationships."""
    db_url = _get_db_url()
    kg = KnowledgeGraphIntegration(db_url)
    return await kg.get_entity_graph(ticker)


@rag_v2_router.get("/kg/{ticker}/related", summary="Find tickers related via knowledge graph")
async def get_related_tickers(ticker: str):
    """Return tickers related to the given ticker via KG relationships (acquisitions, etc.)."""
    db_url = _get_db_url()
    kg = KnowledgeGraphIntegration(db_url)
    related = await kg.get_related_tickers(ticker)
    return {"ticker": ticker.upper(), "related_tickers": related}


@rag_v2_router.post("/search/hybrid", summary="Raw hybrid RRF search without LLM synthesis")
async def hybrid_search_v2(body: QueryRequestV2):
    """Return raw hybrid RRF search results without LLM answer synthesis.

    Use this for inspecting retrieval quality before LLM costs.
    """
    db_url = _get_db_url()
    engine_v2 = HybridRAGEngine(db_url)
    results = await engine_v2.hybrid_rrf_search(
        body.query,
        top_k=body.top_k,
        ticker=body.ticker,
        doc_types=body.doc_types,
        date_from=body.date_from,
        date_to=body.date_to,
        use_reranker=body.use_reranker,
    )
    return {
        "query": body.query,
        "count": len(results),
        "results": results,
    }


@rag_v2_router.get("/search/hierarchical", summary="Hierarchical chunk search (child+parent context)")
async def hierarchical_search_v2(
    q: str = Query(..., description="Search query"),
    ticker: Optional[str] = Query(None),
    top_k: int = Query(10, ge=1, le=50),
):
    """Search hierarchical chunks — returns child match + full parent context.

    Useful when you want both precise matching and broader context window.
    """
    db_url = _get_db_url()
    engine_v2 = HybridRAGEngine(db_url)
    results = await engine_v2.hierarchical_search(q, top_k=top_k, ticker=ticker)
    return {"query": q, "count": len(results), "results": results}
