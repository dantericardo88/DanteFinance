"""
Financial RAG V3 — Persistent Vector Store + Citation Tracking + Portfolio Q&A.

dim_051: Upgrades RAG score 7 → 9 over earnings_rag_v3.py by adding:
  - PersistentVectorStore: ChromaDB → sqlite-vec → numpy TF-IDF fallback
  - EmbeddingEngine: sentence-transformers → TF-IDF random projection fallback
  - FinancialDocumentChunker: SEC Item-aware, table-aware, numeric-aware splitting
  - FinancialEntityExtractor: companies, tickers, dollar amounts, numeric facts
  - CitationTracker: page/section references with validation
  - FinancialRAGPipeline: multi-doc synthesis, company comparison, portfolio Q&A
  - LlamaIndexAdapter: optional PDF ingestion
  - FinancialRAGEngine: orchestrator with bulk indexing and export

FastAPI router: financial_rag_v3_router
  POST /frag/v3/index
  POST /frag/v3/query
  POST /frag/v3/compare
  POST /frag/v3/portfolio
  GET  /frag/v3/stats

Usage::

    from sentinel.sai.financial_rag_v3 import FinancialRAGEngine
    engine = FinancialRAGEngine()
    engine.index_universe(["AAPL", "MSFT"], years=2)
    ans = engine.ask("What were the main growth drivers?")
    print(ans.answer, ans.citations)
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
import re
import sqlite3
import struct
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote_plus, urljoin

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHROMA_PATH = Path("sentinel/data/chroma_db")
_SQLITE_VEC_PATH = Path("sentinel/data/vec_store.db")
_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_HEADERS = {
    "User-Agent": "SENTINEL/3.0 research@sentinel.ai",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_EMBEDDING_DIM = 384
_BATCH_SIZE = 64
_CHUNK_SIZE_TOKENS = 500
_CHUNK_OVERLAP_SENTENCES = 1

# SEC 10-K Item boundaries
_10K_ITEMS = [
    "Item 1.", "Item 1A.", "Item 1B.", "Item 2.", "Item 3.", "Item 4.",
    "Item 5.", "Item 6.", "Item 7.", "Item 7A.", "Item 8.", "Item 9.",
    "Item 9A.", "Item 9B.", "Item 10.", "Item 11.", "Item 12.", "Item 13.",
    "Item 14.", "Item 15.",
]

_ITEM_PATTERN = re.compile(
    r"(?:^|\n)\s*(Item\s+\d+[A-C]?\.?\s+[A-Z][^\n]{3,80})",
    re.IGNORECASE,
)

# Numeric extraction patterns
_DOLLAR_PATTERN = re.compile(
    r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?",
    re.IGNORECASE,
)
_PCT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DATE_PATTERN = re.compile(
    r"(?:Q[1-4]\s*20\d{2}|FY\s*20\d{2}|fiscal\s+(?:year\s+)?20\d{2}|\b20\d{2}\b)"
)
_TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")
_METRIC_KEYWORDS = {
    "revenue", "sales", "ebitda", "ebit", "eps", "earnings per share",
    "gross margin", "operating margin", "net income", "net loss", "cash flow",
    "free cash flow", "capex", "capital expenditure", "guidance", "backlog",
    "bookings", "arr", "mrr", "churn", "nrr", "ltv", "cac", "roic", "roa",
    "roe", "debt", "leverage", "interest coverage", "dividend", "buyback",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    content: str
    ticker: str
    form_type: str           # "10-K", "10-Q", "8-K", "earnings_call", "news"
    year: Optional[int] = None
    quarter: Optional[str] = None
    section: Optional[str] = None
    page_num: Optional[int] = None
    source_url: Optional[str] = None
    file_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    doc_id: str
    content: str
    score: float
    ticker: str
    form_type: str
    year: Optional[int] = None
    quarter: Optional[str] = None
    section: Optional[str] = None
    source_url: Optional[str] = None
    file_date: Optional[str] = None
    rank: int = 0


@dataclass
class FinancialEntities:
    companies: List[str] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    dollar_amounts: List[Tuple[float, str]] = field(default_factory=list)   # (value, unit)
    percentages: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)


@dataclass
class NumericFact:
    metric: str
    value: float
    unit: str
    context: str
    confidence: float
    ticker: Optional[str] = None
    period: Optional[str] = None


@dataclass
class CitedAnswer:
    question: str
    answer: str
    citations: List[str] = field(default_factory=list)
    source_chunks: List[RetrievedChunk] = field(default_factory=list)
    numeric_facts: List[NumericFact] = field(default_factory=list)
    confidence: float = 0.0
    model_used: str = "extractive"


@dataclass
class CompanyComparison:
    ticker1: str
    ticker2: str
    aspect: str
    company1_facts: List[NumericFact] = field(default_factory=list)
    company2_facts: List[NumericFact] = field(default_factory=list)
    narrative: str = ""
    citations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Compute text embeddings with backend priority:
      1. sentence-transformers (all-MiniLM-L6-v2, 384-dim)
      2. TF-IDF sparse → random projection to 384-dim (numpy fallback)
    """

    def __init__(self, dim: int = _EMBEDDING_DIM):
        self.dim = dim
        self._backend = "tfidf"
        self._model = None
        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._projection: Optional[np.ndarray] = None
        self._corpus_tokens: List[List[str]] = []

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._backend = "sentence_transformers"
            logger.info("EmbeddingEngine: using sentence-transformers")
        except ImportError:
            logger.info("EmbeddingEngine: sentence-transformers not available, using TF-IDF fallback")

    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (n, dim) float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._backend == "sentence_transformers":
            results = []
            for i in range(0, len(texts), _BATCH_SIZE):
                batch = texts[i : i + _BATCH_SIZE]
                vecs = self._model.encode(
                    batch, normalize_embeddings=True, show_progress_bar=False
                )
                results.append(vecs.astype(np.float32))
            return np.vstack(results)

        return self._tfidf_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Return (1, dim) float32 array."""
        return self.embed([query])

    # ------------------------------------------------------------------
    # TF-IDF fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens

    def _build_vocab_and_idf(self, texts: List[str]) -> None:
        """Build vocabulary and IDF weights from a corpus."""
        tokenized = [self._tokenize(t) for t in texts]
        self._corpus_tokens = tokenized

        # Vocabulary
        all_tokens: Counter = Counter()
        for tokens in tokenized:
            all_tokens.update(set(tokens))  # df count
        self._vocab = {tok: idx for idx, (tok, _) in enumerate(all_tokens.most_common(8192))}

        # IDF
        n = len(texts)
        idf = np.zeros(len(self._vocab), dtype=np.float32)
        for tok, idx in self._vocab.items():
            df = all_tokens[tok]
            idf[idx] = math.log((n + 1) / (df + 1)) + 1.0
        self._idf = idf

        # Stable random projection matrix (seed from dim)
        rng = np.random.RandomState(seed=self.dim * 31337)
        self._projection = rng.randn(len(self._vocab), self.dim).astype(np.float32)
        # Normalize columns
        norms = np.linalg.norm(self._projection, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._projection /= norms

    def _texts_to_sparse(self, texts: List[str]) -> np.ndarray:
        """TF-IDF sparse matrix (n, vocab_size)."""
        n = len(texts)
        v = len(self._vocab)
        mat = np.zeros((n, v), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf: Counter = Counter(tokens)
            total = max(len(tokens), 1)
            for tok, cnt in tf.items():
                idx = self._vocab.get(tok)
                if idx is not None:
                    mat[i, idx] = (cnt / total) * self._idf[idx]
        return mat

    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        """Compute TF-IDF embeddings projected to self.dim."""
        if not self._vocab:
            self._build_vocab_and_idf(texts)
        sparse = self._texts_to_sparse(texts)
        dense = sparse @ self._projection  # (n, dim)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        dense /= norms
        return dense.astype(np.float32)

    def update_corpus(self, texts: List[str]) -> None:
        """Rebuild TF-IDF vocab/IDF from updated corpus (no-op for ST)."""
        if self._backend == "tfidf":
            self._build_vocab_and_idf(texts)


# ---------------------------------------------------------------------------
# PersistentVectorStore
# ---------------------------------------------------------------------------

class PersistentVectorStore:
    """
    Persistent vector storage with backend priority:
      1. ChromaDB (persistent client)
      2. sqlite-vec extension
      3. SQLite + numpy cosine (pure stdlib fallback)
    """

    def __init__(
        self,
        collection_name: str = "financial_docs",
        embedding_dim: int = _EMBEDDING_DIM,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.engine = embedding_engine or EmbeddingEngine(embedding_dim)
        self._backend = "sqlite_numpy"
        self._chroma_collection = None
        self._conn: Optional[sqlite3.Connection] = None

        # Try ChromaDB first
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(_CHROMA_PATH),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._chroma_collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._backend = "chromadb"
            logger.info("PersistentVectorStore: using ChromaDB at %s", _CHROMA_PATH)
        except ImportError:
            logger.info("PersistentVectorStore: ChromaDB not available")
        except Exception as exc:
            logger.warning("PersistentVectorStore: ChromaDB init failed: %s", exc)

        # Try sqlite-vec if chromadb failed
        if self._backend == "sqlite_numpy":
            try:
                import sqlite_vec  # noqa: F401

                self._init_sqlite_vec()
                self._backend = "sqlite_vec"
                logger.info("PersistentVectorStore: using sqlite-vec")
            except ImportError:
                pass

        # Pure numpy fallback
        if self._backend == "sqlite_numpy":
            self._init_sqlite_numpy()
            logger.info("PersistentVectorStore: using SQLite + numpy dot product")

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            _SQLITE_VEC_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(_SQLITE_VEC_PATH), check_same_thread=False)
        return self._conn

    def _init_sqlite_numpy(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_docs (
                doc_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                ticker TEXT,
                form_type TEXT,
                year INTEGER,
                quarter TEXT,
                section TEXT,
                source_url TEXT,
                file_date TEXT,
                embedding BLOB NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vd_ticker ON vec_docs(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vd_form ON vec_docs(form_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vd_year ON vec_docs(year)")
        # FTS5 virtual table for hybrid keyword search
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs_fts
                USING fts5(doc_id UNINDEXED, content, ticker, section,
                           content='vec_docs', content_rowid='rowid')
            """)
            # Trigger to keep FTS in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS vec_docs_fts_ai
                AFTER INSERT ON vec_docs BEGIN
                    INSERT INTO vec_docs_fts(rowid, doc_id, content, ticker, section)
                    VALUES (new.rowid, new.doc_id, new.content, new.ticker, new.section);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS vec_docs_fts_ad
                AFTER DELETE ON vec_docs BEGIN
                    INSERT INTO vec_docs_fts(vec_docs_fts, rowid, doc_id, content, ticker, section)
                    VALUES ('delete', old.rowid, old.doc_id, old.content, old.ticker, old.section);
                END
            """)
        except Exception as fts_exc:
            logger.debug("FTS5 setup note: %s", fts_exc)
        conn.commit()

    def _init_sqlite_vec(self) -> None:
        import sqlite_vec

        conn = self._get_conn()
        sqlite_vec.load(conn)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(
                doc_id TEXT PRIMARY KEY,
                embedding float[{self.embedding_dim}]
            )
        """)
        self._init_sqlite_numpy()  # also create metadata table
        conn.commit()

    # ------------------------------------------------------------------
    # Add documents
    # ------------------------------------------------------------------

    def add_documents(
        self,
        docs: List[Document],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        if not docs:
            return

        if embeddings is None:
            texts = [d.content for d in docs]
            embeddings = self.engine.embed(texts)

        if self._backend == "chromadb":
            self._chroma_add(docs, embeddings)
        elif self._backend == "sqlite_vec":
            self._sqlite_vec_add(docs, embeddings)
        else:
            self._sqlite_numpy_add(docs, embeddings)

    def _chroma_add(self, docs: List[Document], embeddings: np.ndarray) -> None:
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            batch_emb = embeddings[i : i + batch_size]
            ids = [d.doc_id for d in batch_docs]
            metas = [
                {
                    "ticker": d.ticker or "",
                    "form_type": d.form_type or "",
                    "year": d.year or 0,
                    "quarter": d.quarter or "",
                    "section": d.section or "",
                    "source_url": d.source_url or "",
                    "file_date": d.file_date or "",
                }
                for d in batch_docs
            ]
            self._chroma_collection.upsert(
                ids=ids,
                embeddings=batch_emb.tolist(),
                documents=[d.content for d in batch_docs],
                metadatas=metas,
            )

    def _sqlite_numpy_add(self, docs: List[Document], embeddings: np.ndarray) -> None:
        conn = self._get_conn()
        rows = []
        for doc, emb in zip(docs, embeddings):
            blob = struct.pack(f"{len(emb)}f", *emb.tolist())
            rows.append((
                doc.doc_id,
                doc.content,
                doc.ticker,
                doc.form_type,
                doc.year,
                doc.quarter,
                doc.section,
                doc.source_url,
                doc.file_date,
                blob,
                json.dumps(doc.metadata),
            ))
        conn.executemany("""
            INSERT OR REPLACE INTO vec_docs
            (doc_id, content, ticker, form_type, year, quarter, section,
             source_url, file_date, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()

    def _sqlite_vec_add(self, docs: List[Document], embeddings: np.ndarray) -> None:
        self._sqlite_numpy_add(docs, embeddings)
        conn = self._get_conn()
        for doc, emb in zip(docs, embeddings):
            emb_list = emb.tolist()
            conn.execute(
                "INSERT OR REPLACE INTO vec_index(doc_id, embedding) VALUES (?, ?)",
                (doc.doc_id, str(emb_list)),
            )
        conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        query_vec = self.engine.embed_query(query)

        if self._backend == "chromadb":
            return self._chroma_search(query_vec, k, filter)
        return self._numpy_search(query_vec, k, filter)

    def _build_where_clause(
        self, filter: Optional[Dict[str, Any]]
    ) -> Tuple[str, list]:
        if not filter:
            return "", []
        clauses = []
        params = []
        for key, val in filter.items():
            if val is not None:
                clauses.append(f"{key} = ?")
                params.append(val)
        if not clauses:
            return "", []
        return "WHERE " + " AND ".join(clauses), params

    def _numpy_search(
        self,
        query_vec: np.ndarray,
        k: int,
        filter: Optional[Dict[str, Any]],
    ) -> List[RetrievedChunk]:
        conn = self._get_conn()
        where, params = self._build_where_clause(filter)
        rows = conn.execute(
            f"SELECT doc_id, content, ticker, form_type, year, quarter, "
            f"section, source_url, file_date, embedding FROM vec_docs {where}",
            params,
        ).fetchall()

        if not rows:
            return []

        q = query_vec[0]  # (dim,)
        scores = []
        for row in rows:
            blob = row[9]
            n = len(blob) // 4
            emb = np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)
            score = float(np.dot(q, emb))
            scores.append((score, row))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, row) in enumerate(scores[:k]):
            results.append(
                RetrievedChunk(
                    doc_id=row[0],
                    content=row[1],
                    score=score,
                    ticker=row[2] or "",
                    form_type=row[3] or "",
                    year=row[4],
                    quarter=row[5],
                    section=row[6],
                    source_url=row[7],
                    file_date=row[8],
                    rank=rank,
                )
            )
        return results

    def _chroma_search(
        self,
        query_vec: np.ndarray,
        k: int,
        filter: Optional[Dict[str, Any]],
    ) -> List[RetrievedChunk]:
        where_doc = None
        if filter:
            chroma_filter: Dict[str, Any] = {}
            for key, val in filter.items():
                if val is not None:
                    chroma_filter[key] = {"$eq": val}
            if chroma_filter:
                where_doc = chroma_filter

        kwargs: Dict[str, Any] = {
            "query_embeddings": query_vec.tolist(),
            "n_results": min(k, max(1, self._chroma_collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_doc:
            kwargs["where"] = where_doc

        try:
            res = self._chroma_collection.query(**kwargs)
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return []

        chunks = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for rank, (doc_id, doc_text, meta, dist) in enumerate(
            zip(ids, docs, metas, dists)
        ):
            score = 1.0 - dist  # cosine distance → similarity
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    content=doc_text,
                    score=score,
                    ticker=meta.get("ticker", ""),
                    form_type=meta.get("form_type", ""),
                    year=meta.get("year") or None,
                    quarter=meta.get("quarter") or None,
                    section=meta.get("section") or None,
                    source_url=meta.get("source_url") or None,
                    file_date=meta.get("file_date") or None,
                    rank=rank,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def delete_collection(self) -> None:
        if self._backend == "chromadb":
            try:
                self._chroma_collection.delete(
                    where={"ticker": {"$ne": "__sentinel_reserved__"}}
                )
            except Exception:
                pass
        else:
            conn = self._get_conn()
            conn.execute("DELETE FROM vec_docs")
            conn.commit()

    def get_collection_stats(self) -> Dict[str, Any]:
        if self._backend == "chromadb":
            count = self._chroma_collection.count()
            return {"backend": "chromadb", "total_chunks": count}

        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM vec_docs").fetchone()[0]
        tickers = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM vec_docs"
        ).fetchone()[0]
        forms = conn.execute(
            "SELECT form_type, COUNT(*) FROM vec_docs GROUP BY form_type"
        ).fetchall()
        return {
            "backend": self._backend,
            "total_chunks": total,
            "unique_tickers": tickers,
            "by_form_type": {r[0]: r[1] for r in forms},
        }

    def update_document(self, doc_id: str, new_content: str) -> None:
        if self._backend == "chromadb":
            new_emb = self.engine.embed([new_content])
            self._chroma_collection.update(
                ids=[doc_id],
                embeddings=new_emb.tolist(),
                documents=[new_content],
            )
        else:
            new_emb = self.engine.embed([new_content])[0]
            blob = struct.pack(f"{len(new_emb)}f", *new_emb.tolist())
            conn = self._get_conn()
            conn.execute(
                "UPDATE vec_docs SET content=?, embedding=? WHERE doc_id=?",
                (new_content, blob, doc_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Hybrid search (vector cosine + FTS5 BM25 keyword)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[RetrievedChunk]:
        """
        Reciprocal-rank fusion of vector similarity + FTS5 BM25 keyword search.
        Falls back to pure vector search on non-SQLite backends.
        """
        if self._backend == "chromadb":
            return self.search(query, k, filter)

        # 1. Vector search (top 2k candidates)
        vector_results = self.search(query, k=k * 2, filter=filter)
        vector_rank: Dict[str, int] = {r.doc_id: i for i, r in enumerate(vector_results)}

        # 2. FTS5 keyword search
        fts_rank: Dict[str, int] = {}
        try:
            conn = self._get_conn()
            # Build FTS query: escape special chars, require all words
            fts_query = " ".join(
                w for w in re.findall(r"[a-zA-Z0-9]+", query) if len(w) > 2
            )
            if fts_query:
                where_clause = ""
                params: list = [fts_query]
                if filter:
                    extra = []
                    for col, val in filter.items():
                        if val is not None and col in ("ticker", "section"):
                            extra.append(f"AND vd.{col} = ?")
                            params.append(val)
                    where_clause = " ".join(extra)
                rows = conn.execute(
                    f"""SELECT vd.doc_id FROM vec_docs_fts fts
                        JOIN vec_docs vd ON vd.rowid = fts.rowid
                        WHERE vec_docs_fts MATCH ?
                        {where_clause}
                        ORDER BY rank
                        LIMIT ?""",
                    params + [k * 2],
                ).fetchall()
                fts_rank = {row[0]: i for i, row in enumerate(rows)}
        except Exception as fts_exc:
            logger.debug("FTS5 search failed (fallback to vector only): %s", fts_exc)

        # 3. Reciprocal Rank Fusion
        all_doc_ids = set(vector_rank.keys()) | set(fts_rank.keys())
        rrf_k = 60  # RRF constant
        scores: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            v_rank = vector_rank.get(doc_id, len(vector_rank) + rrf_k)
            f_rank = fts_rank.get(doc_id, len(fts_rank) + rrf_k)
            scores[doc_id] = (
                vector_weight / (rrf_k + v_rank)
                + keyword_weight / (rrf_k + f_rank)
            )

        # 4. Map doc_ids back to RetrievedChunk objects, re-ranked by RRF score
        chunk_map: Dict[str, RetrievedChunk] = {r.doc_id: r for r in vector_results}

        # Fetch any FTS-only results not in vector results
        fts_only = set(fts_rank.keys()) - set(vector_rank.keys())
        if fts_only:
            try:
                conn = self._get_conn()
                placeholders = ",".join("?" * len(fts_only))
                rows = conn.execute(
                    f"SELECT doc_id, content, ticker, form_type, year, quarter, "
                    f"section, source_url, file_date FROM vec_docs "
                    f"WHERE doc_id IN ({placeholders})",
                    list(fts_only),
                ).fetchall()
                for row in rows:
                    chunk_map[row[0]] = RetrievedChunk(
                        doc_id=row[0], content=row[1], score=0.0,
                        ticker=row[2] or "", form_type=row[3] or "",
                        year=row[4], quarter=row[5], section=row[6],
                        source_url=row[7], file_date=row[8],
                    )
            except Exception:
                pass

        sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        results: List[RetrievedChunk] = []
        for rank, doc_id in enumerate(sorted_ids[:k]):
            if doc_id in chunk_map:
                chunk = chunk_map[doc_id]
                chunk.score = scores[doc_id]
                chunk.rank = rank
                results.append(chunk)
        return results

    # ------------------------------------------------------------------
    # Entity-linked retrieval
    # ------------------------------------------------------------------

    def entity_linked_search(
        self,
        query: str,
        entity_extractor: "FinancialEntityExtractor",
        k: int = 10,
    ) -> List[RetrievedChunk]:
        """
        Entity-aware retrieval:
          1. Extract entities (tickers, companies) from the query.
          2. For each recognised ticker, run a targeted search filtered by that ticker.
          3. Merge and de-duplicate results, prioritising entity-matched chunks.
        """
        entities = entity_extractor.extract_entities(query)

        # Resolve company names → tickers
        linked_tickers = list(entities.tickers)
        for company in entities.companies:
            ticker = entity_extractor.link_entity_to_ticker(company)
            if ticker and ticker not in linked_tickers:
                linked_tickers.append(ticker)

        if not linked_tickers:
            return self.search(query, k=k)

        all_chunks: List[RetrievedChunk] = []
        seen_ids: set = set()
        per_ticker_k = max(3, k // max(len(linked_tickers), 1))

        for ticker in linked_tickers[:5]:  # limit to 5 entities
            chunks = self.search(query, k=per_ticker_k, filter={"ticker": ticker})
            for chunk in chunks:
                if chunk.doc_id not in seen_ids:
                    chunk.score *= 1.1  # boost entity-matched chunks by 10%
                    seen_ids.add(chunk.doc_id)
                    all_chunks.append(chunk)

        # Fill remaining slots with generic search if needed
        if len(all_chunks) < k:
            generic = self.search(query, k=k * 2)
            for chunk in generic:
                if chunk.doc_id not in seen_ids:
                    seen_ids.add(chunk.doc_id)
                    all_chunks.append(chunk)

        all_chunks.sort(key=lambda c: c.score, reverse=True)
        for rank, chunk in enumerate(all_chunks[:k]):
            chunk.rank = rank
        return all_chunks[:k]


# ---------------------------------------------------------------------------
# Standalone TF-IDF index helpers (sklearn-free, pure numpy)
# ---------------------------------------------------------------------------

@dataclass
class TFIDFIndex:
    """Lightweight in-memory TF-IDF index for small corpora."""
    documents: List[str]
    doc_ids: List[str]
    vocab: Dict[str, int]
    idf: np.ndarray           # (vocab_size,)
    tfidf_matrix: np.ndarray  # (n_docs, vocab_size)

    # SQLite FTS5 database path (set when build_index persists to SQLite)
    fts_db_path: Optional[str] = None


def build_index(
    documents: List[str],
    doc_ids: Optional[List[str]] = None,
    persist_path: Optional[str] = None,
) -> TFIDFIndex:
    """
    Build a TF-IDF index over a list of document strings.

    Parameters
    ----------
    documents  : list of text strings (the corpus)
    doc_ids    : optional list of IDs; defaults to "doc_{i}"
    persist_path : if given, also store in SQLite FTS5 at this path

    Returns
    -------
    TFIDFIndex ready for use with :func:`query`.
    """
    if not documents:
        empty_vocab: Dict[str, int] = {}
        empty_idf = np.zeros(0, dtype=np.float32)
        empty_mat = np.zeros((0, 0), dtype=np.float32)
        return TFIDFIndex([], [], empty_vocab, empty_idf, empty_mat)

    if doc_ids is None:
        doc_ids = [f"doc_{i}" for i in range(len(documents))]

    # Tokenise
    def _tok(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    tokenised = [_tok(d) for d in documents]
    n = len(documents)

    # Document frequency
    df: Counter = Counter()
    for tokens in tokenised:
        df.update(set(tokens))

    # Vocabulary (top 16 384 by DF)
    vocab_tokens = [tok for tok, _ in df.most_common(16384)]
    vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    v = len(vocab)

    # IDF
    idf = np.zeros(v, dtype=np.float32)
    for tok, idx in vocab.items():
        idf[idx] = math.log((n + 1) / (df[tok] + 1)) + 1.0

    # TF-IDF matrix
    tfidf_matrix = np.zeros((n, v), dtype=np.float32)
    for i, tokens in enumerate(tokenised):
        tf: Counter = Counter(tokens)
        total = max(len(tokens), 1)
        for tok, cnt in tf.items():
            idx = vocab.get(tok)
            if idx is not None:
                tfidf_matrix[i, idx] = (cnt / total) * idf[idx]

    # L2-normalise rows
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    tfidf_matrix /= norms

    fts_db_path: Optional[str] = None

    # Optionally persist to SQLite FTS5
    if persist_path:
        fts_db_path = persist_path
        _SQLITE_VEC_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(persist_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fts_docs (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_index
                USING fts5(doc_id UNINDEXED, content, content='fts_docs', content_rowid='rowid')
            """)
            rows = [(did, doc) for did, doc in zip(doc_ids, documents)]
            conn.executemany(
                "INSERT OR REPLACE INTO fts_docs (doc_id, content) VALUES (?, ?)", rows
            )
            conn.executemany(
                "INSERT OR REPLACE INTO fts_index (rowid, doc_id, content) "
                "SELECT rowid, doc_id, content FROM fts_docs WHERE doc_id = ?",
                [(did,) for did in doc_ids],
            )
            conn.commit()
        except Exception as exc:
            logger.debug("FTS5 persist failed: %s", exc)
        finally:
            conn.close()

    return TFIDFIndex(
        documents=documents,
        doc_ids=doc_ids,
        vocab=vocab,
        idf=idf,
        tfidf_matrix=tfidf_matrix,
        fts_db_path=fts_db_path,
    )


@dataclass
class Chunk:
    """A retrieved text chunk from :func:`query`."""
    doc_id: str
    content: str
    score: float
    rank: int = 0


def tfidf_query(
    index: TFIDFIndex,
    question: str,
    top_k: int = 5,
) -> List[Chunk]:
    """
    Cosine-similarity search over a :class:`TFIDFIndex`.

    Parameters
    ----------
    index    : TFIDFIndex built by :func:`build_index`
    question : query string
    top_k    : number of results to return

    Returns
    -------
    List of :class:`Chunk` objects sorted by descending similarity.
    """
    if not index.documents or not index.vocab:
        return []

    # Vectorise the query using the index vocabulary/IDF
    q_tokens = re.findall(r"[a-z0-9]+", question.lower())
    q_vec = np.zeros(len(index.vocab), dtype=np.float32)
    tf: Counter = Counter(q_tokens)
    total = max(len(q_tokens), 1)
    for tok, cnt in tf.items():
        idx = index.vocab.get(tok)
        if idx is not None:
            q_vec[idx] = (cnt / total) * index.idf[idx]

    # L2-normalise query
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm > 0:
        q_vec /= q_norm

    # Cosine similarity (dot product since rows are L2-normalised)
    scores = index.tfidf_matrix @ q_vec  # (n_docs,)

    # Top-k selection
    top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-min(top_k, len(scores)):]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results: List[Chunk] = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:
            results.append(Chunk(
                doc_id=index.doc_ids[idx],
                content=index.documents[idx],
                score=float(scores[idx]),
                rank=rank,
            ))
    return results


# ---------------------------------------------------------------------------
# FinancialDocumentChunker
# ---------------------------------------------------------------------------

class FinancialDocumentChunker:
    """
    Financial-aware chunking:
      - Respects SEC Item boundaries
      - Keeps markdown tables intact
      - Never cuts within a sentence containing numbers
      - Preserves speaker turns in earnings calls
    """

    _TABLE_PATTERN = re.compile(r"(\|.+\|[\r\n]+)+", re.MULTILINE)
    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\$\d])")
    _ITEM_HEADER = re.compile(
        r"(?:^|\n)(Item\s+\d+[A-C]?\.?\s+[^\n]{2,80})\n", re.IGNORECASE
    )
    _SPEAKER_PATTERN = re.compile(
        r"^([A-Z][a-zA-Z\s\-]+(?:,\s*[A-Z][a-z]+)?)\s*[-–]\s*", re.MULTILINE
    )

    def __init__(
        self,
        chunk_size: int = _CHUNK_SIZE_TOKENS,
        overlap_sentences: int = _CHUNK_OVERLAP_SENTENCES,
    ):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    # ------------------------------------------------------------------
    def chunk_10k(
        self, text: str, ticker: str, year: int
    ) -> List[Document]:
        """Split by Item headers first, then further split large items."""
        docs: List[Document] = []
        sections = self._split_by_items(text)

        for section_name, section_text in sections:
            sub_chunks = self._split_section(section_text, max_tokens=self.chunk_size)
            for idx, chunk_text in enumerate(sub_chunks):
                if not chunk_text.strip():
                    continue
                doc_id = self._make_id(ticker, "10-K", str(year), section_name, idx)
                docs.append(
                    Document(
                        doc_id=doc_id,
                        content=chunk_text.strip(),
                        ticker=ticker,
                        form_type="10-K",
                        year=year,
                        section=section_name,
                        metadata={"chunk_index": idx},
                    )
                )
        return docs

    def chunk_earnings_call(
        self, text: str, ticker: str, quarter: str
    ) -> List[Document]:
        """Split by speaker/section, preserving speaker turns."""
        docs: List[Document] = []

        # Detect prepared remarks vs Q&A
        qa_start = self._find_qa_boundary(text)
        prepared = text[:qa_start] if qa_start > 0 else text
        qa_section = text[qa_start:] if qa_start > 0 else ""

        for section_label, section_text in [
            ("Prepared Remarks", prepared),
            ("Q&A", qa_section),
        ]:
            if not section_text.strip():
                continue
            turns = self._split_speaker_turns(section_text)
            if not turns:
                turns = [("UNKNOWN", section_text)]

            for turn_idx, (speaker, turn_text) in enumerate(turns):
                chunks = self._split_section(turn_text, max_tokens=self.chunk_size)
                for chunk_idx, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():
                        continue
                    doc_id = self._make_id(
                        ticker, "earnings_call", quarter,
                        f"{section_label}_{speaker}", turn_idx * 100 + chunk_idx,
                    )
                    docs.append(
                        Document(
                            doc_id=doc_id,
                            content=f"[{speaker}] {chunk_text.strip()}",
                            ticker=ticker,
                            form_type="earnings_call",
                            quarter=quarter,
                            section=section_label,
                            metadata={"speaker": speaker, "turn_index": turn_idx},
                        )
                    )
        return docs

    def chunk_news(
        self, articles: List[Dict[str, Any]]
    ) -> List[Document]:
        """One chunk per article."""
        docs = []
        for idx, article in enumerate(articles):
            title = article.get("title", "")
            body = article.get("body", article.get("summary", ""))
            content = f"{title}\n\n{body}".strip()
            if not content:
                continue
            ticker = article.get("ticker", "")
            doc_id = self._make_id(ticker, "news", article.get("date", ""), "article", idx)
            docs.append(
                Document(
                    doc_id=doc_id,
                    content=content,
                    ticker=ticker,
                    form_type="news",
                    section="news_article",
                    source_url=article.get("url"),
                    file_date=article.get("date"),
                    metadata={"source": article.get("source", "")},
                )
            )
        return docs

    def chunk_generic(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        ticker: str = "",
        form_type: str = "generic",
    ) -> List[Document]:
        """Character-based fallback chunker."""
        docs = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                doc_id = self._make_id(ticker, form_type, "generic", "chunk", idx)
                docs.append(
                    Document(
                        doc_id=doc_id,
                        content=chunk_text,
                        ticker=ticker,
                        form_type=form_type,
                        metadata={"chunk_index": idx},
                    )
                )
            start = end - overlap
            idx += 1
        return docs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_by_items(self, text: str) -> List[Tuple[str, str]]:
        """Split 10-K text into (item_name, text) pairs."""
        matches = list(self._ITEM_HEADER.finditer(text))
        if not matches:
            return [("Full Document", text)]

        sections = []
        for i, match in enumerate(matches):
            item_name = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((item_name, text[start:end]))

        # Preamble before first item
        if matches[0].start() > 100:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.insert(0, ("Preamble", preamble))

        return sections

    def _split_section(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split a section into chunks, preserving tables and numeric sentences."""
        # Extract and protect tables
        tables: List[str] = []
        protected = text

        def _table_placeholder(m: re.Match) -> str:
            idx = len(tables)
            tables.append(m.group(0))
            return f"\n__TABLE_{idx}__\n"

        protected = self._TABLE_PATTERN.sub(_table_placeholder, protected)

        # Split into sentences
        raw_sentences = self._SENTENCE_END.split(protected)
        # Never cut inside a sentence containing numbers
        sentences: List[str] = []
        buf = ""
        for s in raw_sentences:
            if _DOLLAR_PATTERN.search(s) or _PCT_PATTERN.search(s):
                # Numeric sentence — keep whole
                if buf:
                    sentences.append(buf.strip())
                    buf = ""
                sentences.append(s.strip())
            else:
                buf += " " + s
                # Estimate tokens (rough: 4 chars/token)
                if len(buf) > max_tokens * 4:
                    sentences.append(buf.strip())
                    buf = ""
        if buf.strip():
            sentences.append(buf.strip())

        # Group sentences into chunks
        chunks = []
        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            sent_len = len(sentence) // 4  # rough token count
            if current_len + sent_len > max_tokens and current:
                chunk_text = " ".join(current)
                # Restore tables
                for tidx, table in enumerate(tables):
                    chunk_text = chunk_text.replace(f"__TABLE_{tidx}__", table)
                chunks.append(chunk_text)
                # Overlap: keep last N sentences
                current = current[-self.overlap_sentences:]
                current_len = sum(len(s) // 4 for s in current)
            current.append(sentence)
            current_len += sent_len

        if current:
            chunk_text = " ".join(current)
            for tidx, table in enumerate(tables):
                chunk_text = chunk_text.replace(f"__TABLE_{tidx}__", table)
            chunks.append(chunk_text)

        return chunks

    def _find_qa_boundary(self, text: str) -> int:
        """Find where Q&A section begins in earnings call transcript."""
        patterns = [
            r"(?i)question.and.answer",
            r"(?i)q&a\s+session",
            r"(?i)open.{0,10}questions",
            r"(?i)operator.{0,30}question",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.start()
        return 0

    def _split_speaker_turns(self, text: str) -> List[Tuple[str, str]]:
        """Split text into (speaker, text) turns."""
        matches = list(self._SPEAKER_PATTERN.finditer(text))
        if len(matches) < 2:
            return []
        turns = []
        for i, match in enumerate(matches):
            speaker = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            turn_text = text[start:end].strip()
            if turn_text:
                turns.append((speaker, turn_text))
        return turns

    @staticmethod
    def _make_id(*parts: Any) -> str:
        raw = "_".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# FinancialEntityExtractor
# ---------------------------------------------------------------------------

class FinancialEntityExtractor:
    """
    Extract structured financial entities and numeric facts from text chunks.
    """

    _COMPANY_PATTERN = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}(?:\s+(?:Inc|Corp|Ltd|LLC|Co|Group|Holdings|Technologies|Systems|Solutions|Financial|Capital|Partners|Ventures))?\.?)\b"
    )
    _DOLLAR_FULL = re.compile(
        r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*"
        r"(trillion|billion|million|thousand|T|B|M|K)?",
        re.IGNORECASE,
    )
    _PCT_FULL = re.compile(r"(\d+(?:\.\d+)?)\s*(?:percent|%|pct\.?)", re.IGNORECASE)
    _DATE_FULL = re.compile(
        r"\b(Q[1-4]\s*20\d{2}|FY\s*20\d{2}|fiscal\s+(?:year\s+)?20\d{2}"
        r"|(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+20\d{2})\b",
        re.IGNORECASE,
    )
    _GROWTH_PATTERN = re.compile(
        r"(revenue|sales|earnings|ebitda|margin|profit|loss|income|growth|"
        r"decline|decrease|increase)\s+(?:grew|fell|increased|decreased|"
        r"rose|declined|dropped|surged|improved)\s+(\d+(?:\.\d+)?)\s*(?:%|percent)",
        re.IGNORECASE,
    )
    _DOLLAR_UNIT_MAP = {
        "trillion": 1e12, "T": 1e12,
        "billion": 1e9, "B": 1e9,
        "million": 1e6, "M": 1e6,
        "thousand": 1e3, "K": 1e3,
    }

    def extract_entities(self, text: str) -> FinancialEntities:
        companies = list(
            dict.fromkeys(m.group(1) for m in self._COMPANY_PATTERN.finditer(text))
        )[:20]

        tickers: List[str] = []
        for m in _TICKER_PATTERN.finditer(text):
            sym = m.group(1)
            if 2 <= len(sym) <= 5 and sym not in {"THE", "AND", "FOR", "NOT", "BUT"}:
                tickers.append(sym)
        tickers = list(dict.fromkeys(tickers))[:10]

        dollar_amounts: List[Tuple[float, str]] = []
        for m in self._DOLLAR_FULL.finditer(text):
            val_str = m.group(1).replace(",", "")
            unit = m.group(2) or ""
            multiplier = self._DOLLAR_UNIT_MAP.get(unit, self._DOLLAR_UNIT_MAP.get(unit.upper(), 1.0))
            try:
                val = float(val_str) * multiplier
                dollar_amounts.append((val, unit))
            except ValueError:
                pass

        percentages = [float(m.group(1)) for m in self._PCT_FULL.finditer(text)][:20]
        dates = [m.group(1) for m in self._DATE_FULL.finditer(text)][:15]

        lower_text = text.lower()
        metrics = [kw for kw in _METRIC_KEYWORDS if kw in lower_text]

        return FinancialEntities(
            companies=companies,
            tickers=tickers,
            dollar_amounts=dollar_amounts,
            percentages=percentages,
            dates=dates,
            metrics=metrics,
        )

    def extract_numeric_facts(self, text: str) -> List[NumericFact]:
        facts: List[NumericFact] = []

        # Dollar-amount facts
        for m in self._DOLLAR_FULL.finditer(text):
            val_str = m.group(1).replace(",", "")
            unit = m.group(2) or "USD"
            multiplier = self._DOLLAR_UNIT_MAP.get(unit, self._DOLLAR_UNIT_MAP.get(unit.upper(), 1.0))
            try:
                val = float(val_str) * multiplier
            except ValueError:
                continue
            context_start = max(0, m.start() - 80)
            context_end = min(len(text), m.end() + 80)
            ctx = text[context_start:context_end]
            metric = self._infer_metric_from_context(ctx)
            period = self._extract_period_from_context(ctx)
            facts.append(
                NumericFact(
                    metric=metric,
                    value=val,
                    unit="USD",
                    context=ctx,
                    confidence=0.7,
                    period=period,
                )
            )

        # Growth/change facts
        for m in self._GROWTH_PATTERN.finditer(text):
            metric_word = m.group(1).lower()
            pct_val = float(m.group(2))
            context_start = max(0, m.start() - 40)
            context_end = min(len(text), m.end() + 40)
            ctx = text[context_start:context_end]
            facts.append(
                NumericFact(
                    metric=f"{metric_word}_growth",
                    value=pct_val,
                    unit="%",
                    context=ctx,
                    confidence=0.85,
                    period=self._extract_period_from_context(ctx),
                )
            )

        # Percentage facts (standalone)
        for m in self._PCT_FULL.finditer(text):
            pct_val = float(m.group(1))
            context_start = max(0, m.start() - 80)
            context_end = min(len(text), m.end() + 80)
            ctx = text[context_start:context_end]
            metric = self._infer_metric_from_context(ctx)
            if metric != "financial_metric":
                facts.append(
                    NumericFact(
                        metric=metric,
                        value=pct_val,
                        unit="%",
                        context=ctx,
                        confidence=0.6,
                        period=self._extract_period_from_context(ctx),
                    )
                )

        return facts[:30]

    def link_entity_to_ticker(self, entity_name: str) -> Optional[str]:
        """Fuzzy match company name to ticker symbol."""
        _KNOWN: Dict[str, str] = {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
            "facebook": "META", "nvidia": "NVDA", "tesla": "TSLA",
            "netflix": "NFLX", "salesforce": "CRM", "oracle": "ORCL",
            "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM",
            "broadcom": "AVGO", "cisco": "CSCO", "ibm": "IBM",
            "paypal": "PYPL", "block": "SQ", "square": "SQ",
            "shopify": "SHOP", "zoom": "ZM", "snowflake": "SNOW",
            "palantir": "PLTR", "robinhood": "HOOD", "coinbase": "COIN",
            "jpmorgan": "JPM", "bank of america": "BAC", "wells fargo": "WFC",
            "goldman sachs": "GS", "morgan stanley": "MS",
        }
        name_lower = entity_name.lower().strip()
        for key, ticker in _KNOWN.items():
            if key in name_lower or name_lower in key:
                return ticker
        return None

    # ------------------------------------------------------------------
    def _infer_metric_from_context(self, ctx: str) -> str:
        lower = ctx.lower()
        for kw in sorted(_METRIC_KEYWORDS, key=len, reverse=True):
            if kw in lower:
                return kw.replace(" ", "_")
        return "financial_metric"

    def _extract_period_from_context(self, ctx: str) -> Optional[str]:
        m = _DATE_PATTERN.search(ctx)
        return m.group(0) if m else None


# ---------------------------------------------------------------------------
# CitationTracker
# ---------------------------------------------------------------------------

class CitationTracker:
    """Format citations, generate cited answers, validate numeric facts."""

    def format_citation(self, chunk: RetrievedChunk) -> str:
        parts = [chunk.ticker.upper() if chunk.ticker else "UNK"]

        if chunk.form_type == "10-K":
            year_str = f"FY{chunk.year}" if chunk.year else ""
            parts.append(f"10-K {year_str}")
        elif chunk.form_type == "10-Q":
            qstr = chunk.quarter or (f"FY{chunk.year}" if chunk.year else "")
            parts.append(f"10-Q {qstr}")
        elif chunk.form_type == "8-K":
            parts.append(f"8-K {chunk.file_date or ''}")
        elif chunk.form_type == "earnings_call":
            parts.append(f"Earnings Call {chunk.quarter or chunk.year or ''}")
        elif chunk.form_type == "news":
            parts.append(f"News {chunk.file_date or ''}")
        else:
            parts.append(chunk.form_type or "Filing")

        if chunk.section:
            parts.append(chunk.section)

        return "[" + ", ".join(p for p in parts if p) + "]"

    def generate_answer_with_citations(
        self,
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> CitedAnswer:
        citations = [self.format_citation(c) for c in chunks]
        # Deduplicate
        seen: set = set()
        unique_citations: List[str] = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                unique_citations.append(c)

        return CitedAnswer(
            question="",
            answer=answer,
            citations=unique_citations,
            source_chunks=chunks,
            confidence=sum(c.score for c in chunks) / max(len(chunks), 1),
        )

    def validate_numeric_fact(
        self,
        fact: NumericFact,
        source_chunks: List[RetrievedChunk],
    ) -> bool:
        """Check that the extracted numeric value appears in at least one source chunk."""
        val_str = str(fact.value)
        # Also check rounded version
        rounded = str(round(fact.value))
        for chunk in source_chunks:
            if val_str in chunk.content or rounded in chunk.content:
                return True
        return False


# ---------------------------------------------------------------------------
# LlamaIndexAdapter (optional)
# ---------------------------------------------------------------------------

class LlamaIndexAdapter:
    """
    Optional PDF ingestion via LlamaIndex.
    Falls back to plain text extraction if LlamaIndex not installed.
    """

    def __init__(self):
        self._available = False
        try:
            from llama_index.core import VectorStoreIndex, SimpleDirectoryReader  # noqa: F401
            self._available = True
            logger.info("LlamaIndexAdapter: LlamaIndex available")
        except ImportError:
            logger.info("LlamaIndexAdapter: LlamaIndex not installed, using text fallback")

    def index_pdf(self, path: str) -> Optional[Any]:
        if not self._available:
            return self._fallback_pdf_read(path)
        try:
            from llama_index.core import SimpleDirectoryReader

            reader = SimpleDirectoryReader(input_files=[path])
            docs = reader.load_data()
            logger.info("LlamaIndex: loaded %d nodes from %s", len(docs), path)
            return docs
        except Exception as exc:
            logger.warning("LlamaIndex PDF load failed: %s", exc)
            return self._fallback_pdf_read(path)

    @staticmethod
    def _fallback_pdf_read(path: str) -> str:
        """Try pdfplumber → pdfminer → raw bytes in priority order."""
        # 1. pdfplumber: best table and layout-aware extraction
        try:
            import pdfplumber  # type: ignore
            text_parts: list = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    # Also extract tables as TSV if present
                    for table in page.extract_tables():
                        for row in table:
                            row_text = "\t".join(str(cell or "") for cell in row)
                            if row_text.strip():
                                text_parts.append(row_text)
                    if page_text.strip():
                        text_parts.append(page_text)
            result = "\n".join(text_parts)
            if result.strip():
                logger.info("LlamaIndexAdapter: extracted %d chars via pdfplumber", len(result))
                return result
        except ImportError:
            logger.debug("pdfplumber not available, trying pdfminer")
        except Exception as exc:
            logger.debug("pdfplumber failed: %s", exc)

        # 2. pdfminer fallback
        try:
            from pdfminer.high_level import extract_text  # type: ignore
            result = extract_text(path)
            if result and result.strip():
                logger.info("LlamaIndexAdapter: extracted %d chars via pdfminer", len(result))
                return result
        except ImportError:
            logger.debug("pdfminer not available")
        except Exception as exc:
            logger.debug("pdfminer failed: %s", exc)

        # 3. Raw bytes last resort — strip PDF binary noise with regex
        try:
            with open(path, "rb") as f:
                raw = f.read()
            text = raw.decode("latin-1", errors="replace")
            # Extract readable ASCII runs from PDF binary
            readable = re.findall(r"[\x20-\x7E]{6,}", text)
            result = "\n".join(readable)
            logger.info("LlamaIndexAdapter: raw-bytes fallback, %d chars", len(result))
            return result
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# EDGAR fetching utilities
# ---------------------------------------------------------------------------

class _EDGARFetcher:
    """Minimal EDGAR fetcher for 10-K, 10-Q, 8-K text filings."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _get(self, url: str, timeout: int = 20) -> Optional[requests.Response]:
        try:
            resp = self._session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            logger.warning("EDGAR fetch failed [%s]: %s", url, exc)
            return None

    def resolve_cik(self, ticker: str) -> Optional[str]:
        url = f"{_EDGAR_BASE}/submissions/CIK{ticker.upper().zfill(10)}.json"
        # Try tickers.json mapping first
        try:
            resp = self._get("https://www.sec.gov/files/company_tickers.json", timeout=15)
            if resp:
                data = resp.json()
                for entry in data.values():
                    if entry.get("ticker", "").upper() == ticker.upper():
                        return str(entry["cik_str"]).zfill(10)
        except Exception:
            pass
        return None

    def list_filings(
        self, cik: str, form_type: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
        resp = self._get(url)
        if not resp:
            return []
        try:
            data = resp.json()
            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            results = []
            for form, date, acc in zip(forms, dates, accessions):
                if form == form_type:
                    results.append({"form": form, "date": date, "accession": acc})
                    if len(results) >= limit:
                        break
            return results
        except Exception as exc:
            logger.warning("EDGAR list_filings failed: %s", exc)
            return []

    def fetch_filing_text(self, cik: str, accession: str) -> str:
        acc_fmt = accession.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{acc_fmt}/{accession}-index.htm"
        )
        resp = self._get(index_url)
        if not resp:
            return ""
        # Find the main document
        htm_links = re.findall(
            r'href="(/Archives/edgar/data/[^"]+\.(?:htm|txt))"',
            resp.text,
            re.IGNORECASE,
        )
        for link in htm_links[:3]:
            doc_resp = self._get(f"https://www.sec.gov{link}")
            if doc_resp and len(doc_resp.text) > 1000:
                return self._clean_html(doc_resp.text)
        return ""

    @staticmethod
    def _clean_html(html_text: str) -> str:
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html_text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s{3,}", "\n\n", text)
        return text.strip()

    def fetch_earnings_call_transcript(self, ticker: str, quarter: str) -> str:
        """
        Try to fetch earnings transcript from free sources.
        Primary: EDGAR 8-K (earnings release), fallback: empty string.
        """
        cik = self.resolve_cik(ticker)
        if not cik:
            return ""
        filings = self.list_filings(cik, "8-K", limit=20)
        if not filings:
            return ""
        # Return text of most recent 8-K as proxy
        for filing in filings[:5]:
            text = self.fetch_filing_text(cik, filing["accession"])
            if len(text) > 500:
                return text
        return ""


# ---------------------------------------------------------------------------
# FinancialRAGPipeline
# ---------------------------------------------------------------------------

class FinancialRAGPipeline:
    """
    Complete RAG pipeline for financial documents with citation tracking,
    multi-document synthesis, company comparison, and portfolio Q&A.
    """

    def __init__(
        self,
        vector_store: Optional[PersistentVectorStore] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.vector_store = vector_store or PersistentVectorStore(
            embedding_engine=self.embedding_engine
        )
        self.chunker = FinancialDocumentChunker()
        self.entity_extractor = FinancialEntityExtractor()
        self.citation_tracker = CitationTracker()
        self.llama_adapter = LlamaIndexAdapter()
        self.edgar = _EDGARFetcher()
        self._anthropic_client = None
        self._try_init_anthropic()

    def _try_init_anthropic(self) -> None:
        try:
            import anthropic

            self._anthropic_client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
            logger.info("FinancialRAGPipeline: Anthropic client initialized")
        except ImportError:
            logger.info("FinancialRAGPipeline: anthropic SDK not installed, using extractive fallback")
        except Exception as exc:
            logger.warning("FinancialRAGPipeline: Anthropic init failed: %s", exc)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_ticker(
        self,
        ticker: str,
        years: int = 3,
        forms: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        if forms is None:
            forms = ["10-K", "10-Q", "8-K", "earnings_call"]

        indexed: Dict[str, int] = {}
        cik = self.edgar.resolve_cik(ticker)
        if not cik:
            logger.warning("Could not resolve CIK for %s", ticker)
            return indexed

        current_year = datetime.now().year

        for form_type in forms:
            if form_type == "earnings_call":
                # Fetch via 8-K proxy
                text = self.edgar.fetch_earnings_call_transcript(ticker, "latest")
                if text:
                    quarter = f"Q{(datetime.now().month - 1) // 3 + 1} {current_year}"
                    docs = self.chunker.chunk_earnings_call(text, ticker, quarter)
                    self.vector_store.add_documents(docs)
                    indexed["earnings_call"] = indexed.get("earnings_call", 0) + len(docs)
                continue

            filings = self.edgar.list_filings(cik, form_type, limit=years * 4)
            total_docs = 0
            for filing in filings:
                text = self.edgar.fetch_filing_text(cik, filing["accession"])
                if not text or len(text) < 200:
                    continue

                filing_year = int(filing["date"][:4]) if filing.get("date") else current_year

                if form_type == "10-K":
                    docs = self.chunker.chunk_10k(text, ticker, filing_year)
                else:
                    docs = self.chunker.chunk_generic(
                        text, ticker=ticker, form_type=form_type
                    )
                    for doc in docs:
                        doc.year = filing_year
                        doc.file_date = filing.get("date")

                self.vector_store.add_documents(docs)
                total_docs += len(docs)
                time.sleep(0.2)  # EDGAR rate limit

            indexed[form_type] = total_docs
            logger.info("Indexed %s %s: %d chunks", ticker, form_type, total_docs)

        return indexed

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        tickers: Optional[List[str]] = None,
        form_types: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> CitedAnswer:
        # Build filter
        filt: Dict[str, Any] = {}
        if tickers and len(tickers) == 1:
            filt["ticker"] = tickers[0]
        if form_types and len(form_types) == 1:
            filt["form_type"] = form_types[0]

        # Retrieve chunks
        chunks = self.vector_store.search(question, k=top_k * 2, filter=filt or None)

        # Filter by multiple tickers if needed
        if tickers and len(tickers) > 1:
            chunks = [c for c in chunks if c.ticker in tickers]

        # Filter by multiple form types if needed
        if form_types and len(form_types) > 1:
            chunks = [c for c in chunks if c.form_type in form_types]

        # Re-rank: boost chunks that contain numeric facts matching question intent
        chunks = self._rerank_by_numeric_relevance(question, chunks)
        chunks = chunks[:top_k]

        if not chunks:
            return CitedAnswer(
                question=question,
                answer="No relevant documents found in the index.",
                confidence=0.0,
            )

        # Extract numeric facts
        all_facts: List[NumericFact] = []
        for chunk in chunks:
            facts = self.entity_extractor.extract_numeric_facts(chunk.content)
            for fact in facts:
                fact.ticker = chunk.ticker
            all_facts.extend(facts)

        # Generate answer
        answer = self._generate_answer(question, chunks)

        # Build cited answer
        cited = self.citation_tracker.generate_answer_with_citations(answer, chunks)
        cited.question = question
        cited.numeric_facts = all_facts[:15]
        return cited

    # ------------------------------------------------------------------
    # Multi-document synthesis
    # ------------------------------------------------------------------

    def multi_doc_synthesis(
        self,
        questions: List[str],
        tickers: List[str],
    ) -> Dict[str, CitedAnswer]:
        results: Dict[str, CitedAnswer] = {}
        for question in questions:
            results[question] = self.query(question, tickers=tickers, top_k=8)
        return results

    # ------------------------------------------------------------------
    # Company comparison
    # ------------------------------------------------------------------

    def compare_companies(
        self,
        ticker1: str,
        ticker2: str,
        aspect: str,
    ) -> CompanyComparison:
        question = f"What is the {aspect} for {ticker1} vs {ticker2}?"

        ans1 = self.query(aspect, tickers=[ticker1], top_k=6)
        ans2 = self.query(aspect, tickers=[ticker2], top_k=6)

        narrative = self._generate_comparison_narrative(
            ticker1, ticker2, aspect, ans1, ans2
        )
        all_citations = list(dict.fromkeys(ans1.citations + ans2.citations))

        return CompanyComparison(
            ticker1=ticker1,
            ticker2=ticker2,
            aspect=aspect,
            company1_facts=ans1.numeric_facts,
            company2_facts=ans2.numeric_facts,
            narrative=narrative,
            citations=all_citations,
        )

    # ------------------------------------------------------------------
    # Portfolio Q&A
    # ------------------------------------------------------------------

    def portfolio_qa(
        self,
        holdings: Dict[str, float],  # ticker → weight (0-1)
        question: str,
    ) -> CitedAnswer:
        tickers = list(holdings.keys())
        total_weight = sum(holdings.values())

        all_chunks: List[RetrievedChunk] = []
        for ticker, weight in holdings.items():
            filt: Dict[str, Any] = {"ticker": ticker}
            chunks = self.vector_store.search(question, k=6, filter=filt)
            # Score-weight by portfolio allocation
            normalized_weight = weight / max(total_weight, 1.0)
            for chunk in chunks:
                chunk.score *= normalized_weight
            all_chunks.extend(chunks)

        # Sort by weighted score
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        top_chunks = all_chunks[:12]

        # Synthesize portfolio-level answer
        portfolio_desc = ", ".join(
            f"{t} ({w * 100:.1f}%)" for t, w in sorted(
                holdings.items(), key=lambda x: x[1], reverse=True
            )
        )
        synthesis_question = (
            f"For a portfolio with holdings [{portfolio_desc}]: {question}"
        )
        answer = self._generate_answer(synthesis_question, top_chunks)

        cited = self.citation_tracker.generate_answer_with_citations(answer, top_chunks)
        cited.question = question
        cited.numeric_facts = []
        for chunk in top_chunks:
            facts = self.entity_extractor.extract_numeric_facts(chunk.content)
            for fact in facts:
                fact.ticker = chunk.ticker
            cited.numeric_facts.extend(facts)
        cited.numeric_facts = cited.numeric_facts[:20]
        return cited

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rerank_by_numeric_relevance(
        self,
        question: str,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Boost chunks that contain metric keywords from the question."""
        question_lower = question.lower()
        boost_keywords = [kw for kw in _METRIC_KEYWORDS if kw in question_lower]
        if not boost_keywords:
            return sorted(chunks, key=lambda c: c.score, reverse=True)

        def _score(chunk: RetrievedChunk) -> float:
            base = chunk.score
            text_lower = chunk.content.lower()
            boost = sum(0.05 for kw in boost_keywords if kw in text_lower)
            # Extra boost for numeric content
            if _DOLLAR_PATTERN.search(chunk.content) or _PCT_PATTERN.search(chunk.content):
                boost += 0.1
            return base + boost

        for chunk in chunks:
            chunk.score = _score(chunk)
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def _build_context(self, chunks: List[RetrievedChunk], max_chars: int = 8000) -> str:
        parts: List[str] = []
        total = 0
        for chunk in chunks:
            citation = self.citation_tracker.format_citation(chunk)
            part = f"{citation}\n{chunk.content}\n"
            if total + len(part) > max_chars:
                break
            parts.append(part)
            total += len(part)
        return "\n---\n".join(parts)

    def _generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
    ) -> str:
        context = self._build_context(chunks)

        if self._anthropic_client and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                prompt = (
                    "You are a financial analyst assistant. Using only the provided "
                    "source documents, answer the question concisely and accurately. "
                    "Include specific numbers and cite sources.\n\n"
                    f"SOURCES:\n{context}\n\n"
                    f"QUESTION: {question}\n\n"
                    "ANSWER:"
                )
                import anthropic

                resp = self._anthropic_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
            except Exception as exc:
                logger.warning("Claude API call failed: %s", exc)

        # Extractive fallback: return most relevant sentences
        return self._extractive_answer(question, chunks)

    def _extractive_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
    ) -> str:
        """Extract key sentences from top chunks as answer."""
        question_words = set(re.findall(r"[a-z]+", question.lower()))
        all_sentences: List[Tuple[float, str, str]] = []  # (score, ticker, sentence)

        for chunk in chunks[:5]:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.content)
            for sent in sentences:
                sent_lower = sent.lower()
                sent_words = set(re.findall(r"[a-z]+", sent_lower))
                overlap = len(question_words & sent_words) / max(len(question_words), 1)
                # Prefer sentences with numbers
                num_boost = 0.3 if (_DOLLAR_PATTERN.search(sent) or _PCT_PATTERN.search(sent)) else 0.0
                score = overlap + num_boost + chunk.score * 0.1
                if len(sent.strip()) > 20:
                    citation = self.citation_tracker.format_citation(chunk)
                    all_sentences.append((score, citation, sent.strip()))

        all_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sents = all_sentences[:5]

        if not top_sents:
            return chunks[0].content[:500] if chunks else "No relevant information found."

        answer_parts = []
        for _, citation, sent in top_sents:
            answer_parts.append(f"{sent} {citation}")

        return " ".join(answer_parts)

    def _generate_comparison_narrative(
        self,
        ticker1: str,
        ticker2: str,
        aspect: str,
        ans1: CitedAnswer,
        ans2: CitedAnswer,
    ) -> str:
        if self._anthropic_client and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic

                prompt = (
                    f"Compare {ticker1} and {ticker2} on {aspect}.\n\n"
                    f"{ticker1} data: {ans1.answer}\n\n"
                    f"{ticker2} data: {ans2.answer}\n\n"
                    "Provide a concise comparison with specific numbers."
                )
                resp = self._anthropic_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
            except Exception:
                pass

        return (
            f"{ticker1}: {ans1.answer[:400]}\n\n"
            f"{ticker2}: {ans2.answer[:400]}"
        )


# ---------------------------------------------------------------------------
# FinancialRAGEngine (orchestrator)
# ---------------------------------------------------------------------------

class FinancialRAGEngine:
    """
    Top-level orchestrator for the financial RAG system.
    Provides bulk indexing, Q&A, and index management.
    """

    def __init__(self, collection_name: str = "financial_docs_v3"):
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = PersistentVectorStore(
            collection_name=collection_name,
            embedding_engine=self.embedding_engine,
        )
        self.pipeline = FinancialRAGPipeline(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine,
        )
        self._indexed_tickers: set = set()

    # ------------------------------------------------------------------
    def index_universe(
        self,
        tickers: List[str],
        years: int = 2,
        forms: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        results: Dict[str, Dict[str, int]] = {}
        for ticker in tickers:
            logger.info("Indexing %s...", ticker)
            indexed = self.pipeline.index_ticker(ticker, years=years, forms=forms)
            results[ticker] = indexed
            self._indexed_tickers.add(ticker)
        return results

    def ask(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CitedAnswer:
        tickers = None
        form_types = None
        if context:
            tickers = context.get("tickers")
            form_types = context.get("form_types")
        return self.pipeline.query(question, tickers=tickers, form_types=form_types)

    def get_collection_stats(self) -> Dict[str, Any]:
        stats = self.vector_store.get_collection_stats()
        stats["indexed_tickers"] = sorted(self._indexed_tickers)
        return stats

    def export_index(self, path: str) -> None:
        """Export SQLite vector store to a backup location."""
        import shutil

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.vector_store._backend == "chromadb":
            shutil.copytree(str(_CHROMA_PATH), str(dest / "chroma_db"), dirs_exist_ok=True)
            logger.info("Exported ChromaDB index to %s", dest / "chroma_db")
        else:
            src = _SQLITE_VEC_PATH
            if src.exists():
                shutil.copy2(str(src), str(dest / "vec_store.db"))
                logger.info("Exported SQLite index to %s", dest / "vec_store.db")

    # Delegation helpers
    def compare_companies(
        self, ticker1: str, ticker2: str, aspect: str
    ) -> CompanyComparison:
        return self.pipeline.compare_companies(ticker1, ticker2, aspect)

    def portfolio_qa(
        self, holdings: Dict[str, float], question: str
    ) -> CitedAnswer:
        return self.pipeline.portfolio_qa(holdings, question)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

financial_rag_v3_router = APIRouter(prefix="/frag/v3", tags=["financial-rag-v3"])

_engine_instance: Optional[FinancialRAGEngine] = None


def _get_engine() -> FinancialRAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FinancialRAGEngine()
    return _engine_instance


class IndexRequest(BaseModel):
    tickers: List[str]
    years: int = 2
    forms: Optional[List[str]] = None


class QueryRequest(BaseModel):
    question: str
    tickers: Optional[List[str]] = None
    form_types: Optional[List[str]] = None
    top_k: int = 10


class CompareRequest(BaseModel):
    ticker1: str
    ticker2: str
    aspect: str


class PortfolioQARequest(BaseModel):
    holdings: Dict[str, float]
    question: str


@financial_rag_v3_router.post("/index")
async def index_universe(req: IndexRequest, background: BackgroundTasks):
    engine = _get_engine()
    background.add_task(engine.index_universe, req.tickers, req.years, req.forms)
    return {"status": "indexing started", "tickers": req.tickers}


@financial_rag_v3_router.post("/query")
async def query(req: QueryRequest):
    engine = _get_engine()
    ans = engine.ask(
        req.question,
        context={"tickers": req.tickers, "form_types": req.form_types},
    )
    return {
        "question": ans.question,
        "answer": ans.answer,
        "citations": ans.citations,
        "confidence": ans.confidence,
        "numeric_facts": [
            {
                "metric": f.metric,
                "value": f.value,
                "unit": f.unit,
                "period": f.period,
                "ticker": f.ticker,
            }
            for f in ans.numeric_facts
        ],
    }


@financial_rag_v3_router.post("/compare")
async def compare(req: CompareRequest):
    engine = _get_engine()
    comp = engine.compare_companies(req.ticker1, req.ticker2, req.aspect)
    return {
        "ticker1": comp.ticker1,
        "ticker2": comp.ticker2,
        "aspect": comp.aspect,
        "narrative": comp.narrative,
        "citations": comp.citations,
        "company1_facts_count": len(comp.company1_facts),
        "company2_facts_count": len(comp.company2_facts),
    }


@financial_rag_v3_router.post("/portfolio")
async def portfolio_qa(req: PortfolioQARequest):
    engine = _get_engine()
    ans = engine.portfolio_qa(req.holdings, req.question)
    return {
        "question": ans.question,
        "answer": ans.answer,
        "citations": ans.citations,
        "confidence": ans.confidence,
    }


@financial_rag_v3_router.get("/stats")
async def get_stats():
    engine = _get_engine()
    return engine.get_collection_stats()


# ---------------------------------------------------------------------------
# Main — demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = FinancialRAGEngine()

    print("\n=== SENTINEL Financial RAG V3 Demo ===\n")

    # Index AAPL and MSFT
    print("Indexing AAPL and MSFT (10-K + earnings calls)...")
    results = engine.index_universe(
        ["AAPL", "MSFT"],
        years=2,
        forms=["10-K", "earnings_call"],
    )
    for ticker, indexed in results.items():
        total = sum(indexed.values())
        print(f"  {ticker}: {total} chunks indexed ({indexed})")

    # Portfolio Q&A
    print("\n--- Portfolio Q&A ---")
    holdings = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
    portfolio_q = "What are the main risks across my portfolio?"
    port_ans = engine.portfolio_qa(holdings, portfolio_q)
    print(f"Q: {portfolio_q}")
    print(f"A: {port_ans.answer[:600]}")
    print("Citations:", port_ans.citations[:5])

    # General Q&A
    print("\n--- General Q&A ---")
    question = "What were the main growth drivers?"
    ans = engine.ask(question, context={"tickers": ["AAPL", "MSFT"]})
    print(f"Q: {question}")
    print(f"A: {ans.answer[:600]}")
    print("Citations:", ans.citations[:5])
    if ans.numeric_facts:
        print("Key facts:")
        for fact in ans.numeric_facts[:3]:
            print(f"  {fact.ticker} {fact.metric}: {fact.value} {fact.unit} ({fact.period})")

    # Company comparison
    print("\n--- Company Comparison: AAPL vs MSFT gross margin ---")
    comp = engine.compare_companies("AAPL", "MSFT", "gross margin trends")
    print(comp.narrative[:600])
    print("Sources:", comp.citations[:4])

    # Stats
    print("\n--- Index Stats ---")
    stats = engine.get_collection_stats()
    print(json.dumps(stats, indent=2))
