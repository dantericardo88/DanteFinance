"""RAG over financial documents — pgvector + LlamaIndex-style pipeline.

dim_051: Retrieval-Augmented Generation over EDGAR filings (10-K, 10-Q, 8-K),
earnings transcripts, and news articles. Targets score raise from 6 → 9+.

Architecture:
  DocumentIngestionPipeline  — fetch EDGAR filing → chunk → embed → pgvector upsert
  RAGQueryEngine             — dense + hybrid search, LLM synthesis (Claude Haiku)
  DocumentIndex              — table DDL, inventory, maintenance
  EmbeddingModel             — lazy sentence-transformers wrapper (384-dim MiniLM)
  rag_router                 — FastAPI router exposing all capabilities

Storage table: ``financial_documents`` (vector(384), metadata JSONB, pgvector IVFFlat).

Usage::
    from sentinel.sai.rag_financial_docs import DocumentIngestionPipeline, RAGQueryEngine

    pipeline = DocumentIngestionPipeline(db_url, user_agent="SENTINEL ...")
    await pipeline.ingest_10k("0000320193", ticker="AAPL", years=3)

    engine = RAGQueryEngine(db_url)
    answer = await engine.answer_question("What are Apple's main risk factors?", ticker="AAPL")
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_USER_AGENT = "SENTINEL financial-terminal richard.porras@realempanada.com"
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# ── Embedding model ────────────────────────────────────────────────────────────

class EmbeddingModel:
    """Lazy-loaded sentence-transformers wrapper (all-MiniLM-L6-v2, 384-dim).

    The model is only instantiated on first encode() call (~200 MB download on
    cold start). Subsequent calls reuse the cached instance for the process
    lifetime. Matches the embedding space used by ``news_articles.embedding``.
    """

    _instance: Optional[Any] = None

    @classmethod
    def _get(cls) -> Any:
        if cls._instance is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading all-MiniLM-L6-v2 for financial docs RAG...")
                cls._instance = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Embedding model ready")
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers not installed — run: "
                    "pip install sentence-transformers"
                ) from exc
        return cls._instance

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return (N, 384) float32 array."""
        model = self._get()
        return model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


_embedding_model = EmbeddingModel()


# ── DDL ────────────────────────────────────────────────────────────────────────

_FINANCIAL_DOCS_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS financial_documents (
    id          BIGSERIAL PRIMARY KEY,
    ticker      VARCHAR(20),
    cik         VARCHAR(10),
    doc_type    VARCHAR(30)  NOT NULL,
    section     VARCHAR(100),
    chunk_text  TEXT         NOT NULL,
    embedding   vector(384),
    metadata    JSONB        NOT NULL DEFAULT '{}',
    ingested_at TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_fd_ticker
    ON financial_documents (ticker)
    WHERE ticker IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_fd_doc_type
    ON financial_documents (doc_type);

CREATE INDEX IF NOT EXISTS ix_fd_cik
    ON financial_documents (cik)
    WHERE cik IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_fd_trgm_chunk
    ON financial_documents USING GIN (chunk_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS ix_fd_meta
    ON financial_documents USING GIN (metadata);
"""

_IVFFLAT_DDL = """
CREATE INDEX IF NOT EXISTS ix_fd_embedding_ivfflat
    ON financial_documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = {lists})
"""

# ── Engine registry ────────────────────────────────────────────────────────────

_engines: dict[str, AsyncEngine] = {}
_schema_ready: set[str] = set()


def _get_engine(db_url: str) -> AsyncEngine:
    if db_url not in _engines:
        _engines[db_url] = create_async_engine(
            db_url, pool_size=5, max_overflow=10, pool_pre_ping=True
        )
    return _engines[db_url]


async def _ensure_schema(db_url: str) -> None:
    if db_url in _schema_ready:
        return
    engine = _get_engine(db_url)
    async with engine.begin() as conn:
        for stmt in _FINANCIAL_DOCS_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    await conn.execute(text(stmt))
                except Exception as exc:
                    logger.debug("DDL stmt skipped", stmt=stmt[:60], error=str(exc))
    _schema_ready.add(db_url)
    logger.info("financial_documents schema ready")


# ── Helpers: EDGAR fetch ───────────────────────────────────────────────────────

async def _edgar_get(url: str, host: str = "data.sec.gov", timeout: int = 30) -> dict:
    headers = {**_HEADERS, "Host": host}
    await asyncio.sleep(0.12)  # EDGAR rate limit ~10 req/s
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _edgar_get_text(url: str, timeout: int = 60) -> str:
    await asyncio.sleep(0.12)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()
        return resp.text


async def _resolve_cik(ticker: str) -> Optional[str]:
    try:
        data = await _edgar_get(EDGAR_TICKERS_URL, host="www.sec.gov")
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning("CIK resolution failed", ticker=ticker, error=str(exc))
    return None


async def _fetch_submissions(cik: str) -> dict:
    return await _edgar_get(f"{EDGAR_BASE}/submissions/CIK{cik}.json")


def _list_filings(submissions: dict, form_type: str, limit: int = 20) -> list[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    periods = recent.get("reportDate", [])

    results = []
    for i, form in enumerate(forms):
        if form != form_type:
            continue
        results.append({
            "form_type": form,
            "filing_date": dates[i] if i < len(dates) else None,
            "accession": accessions[i] if i < len(accessions) else None,
            "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
            "period": periods[i] if i < len(periods) else None,
            "cik": submissions.get("cik", ""),
        })
        if len(results) >= limit:
            break
    return results


async def _fetch_filing_text(cik: str, accession: str, primary_doc: str) -> str:
    acc_clean = accession.replace("-", "")
    cik_stripped = cik.lstrip("0") or "0"
    url = f"{EDGAR_ARCHIVES}/{cik_stripped}/{acc_clean}/{primary_doc}"
    return await _edgar_get_text(url)


def _strip_html(text: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    for entity, char in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                          ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(entity, char)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()


# ── Section extractors ─────────────────────────────────────────────────────────

_SECTION_PATTERNS: dict[str, tuple[str, str]] = {
    "Risk Factors": (
        r"ITEM\s+1A[\.\s]*RISK\s+FACTORS",
        r"ITEM\s+1B[\.\s]|ITEM\s+2[\.\s]",
    ),
    "MD&A": (
        r"ITEM\s+7[\.\s]*MANAGEMENT[''`]?S?\s+DISCUSSION",
        r"ITEM\s+7A[\.\s]|ITEM\s+8[\.\s]",
    ),
    "Business": (
        r"ITEM\s+1[\.\s]*BUSINESS",
        r"ITEM\s+1A[\.\s]|ITEM\s+2[\.\s]",
    ),
    "Financial Statements": (
        r"ITEM\s+8[\.\s]*FINANCIAL\s+STATEMENTS",
        r"ITEM\s+9[\.\s]",
    ),
}


def _extract_section(text: str, section_name: str) -> str:
    """Extract named section from filing text using regex anchors."""
    patterns = _SECTION_PATTERNS.get(section_name)
    if not patterns:
        return text[:6000]

    start_pat, end_pat = patterns
    m_start = re.search(start_pat, text, re.IGNORECASE)
    if not m_start:
        return ""

    start = m_start.start()
    m_end = re.search(end_pat, text[start + 200:], re.IGNORECASE)
    end = (start + 200 + m_end.start()) if m_end else min(start + 40000, len(text))
    return text[start:end]


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_document(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Sliding-window word-count chunking with sentence-boundary awareness.

    Args:
        text:        Raw text to chunk.
        chunk_size:  Approximate chunk size in whitespace tokens (≈ BPE tokens).
        overlap:     Token overlap between successive chunks for context continuity.

    Returns:
        List of text chunk strings.
    """
    # Sentence-split first to avoid cutting mid-sentence
    sentences = re.split(r"(?<=[.!?])\s+", text)
    words: list[str] = []
    for sent in sentences:
        words.extend(sent.split())

    if not words:
        return []

    chunks: list[str] = []
    start = 0
    total = len(words)
    while start < total:
        end = min(start + chunk_size, total)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == total:
            break
        start = end - overlap
    return chunks


# ── Vector helpers ─────────────────────────────────────────────────────────────

def _vec_str(embedding: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"


async def _embed_in_executor(texts: list[str]) -> list[list[float]]:
    """Run embedding in thread pool to avoid blocking async event loop."""
    loop = asyncio.get_event_loop()
    arr: np.ndarray = await loop.run_in_executor(
        None, _embedding_model.encode, texts
    )
    return [row.tolist() for row in arr]


# ── EmbeddingModel public API ──────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Synchronous single-text embedding (384-dim). Lazy model load."""
    return _embedding_model.encode([text])[0].tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embedding — more efficient than calling embed_text N times."""
    if not texts:
        return []
    return [row.tolist() for row in _embedding_model.encode(texts)]


# ── DocumentIndex ──────────────────────────────────────────────────────────────

class DocumentIndex:
    """Schema management and maintenance utilities for financial_documents table."""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    async def create_tables(self) -> None:
        """Create financial_documents table with pgvector index if not exists."""
        await _ensure_schema(self.db_url)

    async def build_ivfflat_index(self) -> None:
        """Build IVFFlat approximate nearest-neighbour index.

        Only effective after > 100 rows. Safe to call repeatedly (IF NOT EXISTS).
        """
        engine = _get_engine(self.db_url)
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT COUNT(*) FROM financial_documents WHERE embedding IS NOT NULL")
            )
            count = result.scalar() or 0

        if count < 100:
            logger.warning("Too few embeddings for IVFFlat index", count=count)
            return

        lists = max(10, min(int(count / 39), 1000))
        async with engine.begin() as conn:
            await conn.execute(text(_IVFFLAT_DDL.format(lists=lists)))
        logger.info("IVFFlat index built/verified", lists=lists, rows=count)

    async def get_document_inventory(
        self, ticker: Optional[str] = None
    ) -> pd.DataFrame:
        """Return a DataFrame summarising indexed documents per ticker/doc_type."""
        await _ensure_schema(self.db_url)
        engine = _get_engine(self.db_url)

        where = "WHERE ticker = :ticker" if ticker else ""
        params = {"ticker": ticker.upper()} if ticker else {}

        sql = text(f"""
            SELECT
                ticker,
                cik,
                doc_type,
                section,
                COUNT(*) AS chunk_count,
                MAX(ingested_at) AS last_ingested
            FROM financial_documents
            {where}
            GROUP BY ticker, cik, doc_type, section
            ORDER BY ticker, doc_type, section
        """)

        async with engine.connect() as conn:
            rows = (await conn.execute(sql, params)).fetchall()

        if not rows:
            return pd.DataFrame(columns=["ticker", "cik", "doc_type", "section",
                                         "chunk_count", "last_ingested"])
        return pd.DataFrame([dict(r._mapping) for r in rows])

    async def delete_stale_documents(
        self, ticker: str, older_than_days: int = 90
    ) -> int:
        """Delete documents older than threshold for a ticker. Returns deleted count."""
        await _ensure_schema(self.db_url)
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        engine = _get_engine(self.db_url)
        async with engine.begin() as conn:
            result = await conn.execute(
                text("""
                    DELETE FROM financial_documents
                    WHERE ticker = :ticker AND ingested_at < :cutoff
                """),
                {"ticker": ticker.upper(), "cutoff": cutoff},
            )
            count = result.rowcount
        logger.info("Stale documents deleted", ticker=ticker, count=count)
        return count

    async def rebuild_index(self, ticker: str, pipeline: "DocumentIngestionPipeline") -> dict:
        """Delete and re-ingest all EDGAR filings for a ticker."""
        await self.delete_stale_documents(ticker, older_than_days=0)
        result_10k = await pipeline.ingest_10k(ticker=ticker, years=3)
        result_10q = await pipeline.ingest_10q(ticker=ticker, years=2)
        return {"ticker": ticker, "10k_filings": len(result_10k), "10q_filings": len(result_10q)}


# ── DocumentIngestionPipeline ──────────────────────────────────────────────────

class DocumentIngestionPipeline:
    """Full ingest pipeline: EDGAR download → section extract → chunk → embed → pgvector."""

    def __init__(self, db_url: str, user_agent: str = _USER_AGENT) -> None:
        self.db_url = db_url
        self._headers = {**_HEADERS, "User-Agent": user_agent}
        self._index = DocumentIndex(db_url)

    async def _upsert_chunks(
        self,
        ticker: Optional[str],
        cik: Optional[str],
        doc_type: str,
        section: str,
        chunks: list[str],
        metadata: dict,
    ) -> int:
        """Embed chunks and upsert into financial_documents. Returns inserted count."""
        if not chunks:
            return 0

        await _ensure_schema(self.db_url)
        embeddings = await _embed_in_executor(chunks)

        engine = _get_engine(self.db_url)
        upsert_sql = text("""
            INSERT INTO financial_documents
                (ticker, cik, doc_type, section, chunk_text, embedding, metadata)
            VALUES
                (:ticker, :cik, :doc_type, :section, :chunk_text, :embedding::vector, :metadata)
            ON CONFLICT DO NOTHING
        """)

        meta_json = json.dumps(metadata)
        async with engine.begin() as conn:
            for chunk, emb in zip(chunks, embeddings):
                await conn.execute(upsert_sql, {
                    "ticker": ticker.upper() if ticker else None,
                    "cik": cik,
                    "doc_type": doc_type,
                    "section": section,
                    "chunk_text": chunk,
                    "embedding": _vec_str(emb),
                    "metadata": meta_json,
                })

        logger.debug("Chunks upserted", doc_type=doc_type, section=section, n=len(chunks))
        return len(chunks)

    async def ingest_edgar_filing(
        self, accession_number: str, cik: str, form_type: str
    ) -> dict:
        """Download a single EDGAR filing by accession number, chunk, embed, store.

        Returns dict with {accession_number, doc_type, chunks_ingested, sections}.
        """
        cik = cik.zfill(10)
        try:
            submissions = await _fetch_submissions(cik)
        except Exception as exc:
            return {"error": str(exc), "accession_number": accession_number}

        # Find primary document for this accession
        recent = submissions.get("filings", {}).get("recent", {})
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        filing_dates = recent.get("filingDate", [])

        primary_doc = None
        filing_date_str = None
        for i, acc in enumerate(accessions):
            if acc == accession_number:
                primary_doc = primary_docs[i] if i < len(primary_docs) else None
                filing_date_str = filing_dates[i] if i < len(filing_dates) else None
                break

        if not primary_doc:
            return {"error": "Primary document not found", "accession_number": accession_number}

        try:
            raw = await _fetch_filing_text(cik, accession_number, primary_doc)
        except Exception as exc:
            return {"error": f"Filing fetch failed: {exc}", "accession_number": accession_number}

        clean = _strip_html(raw) if "<" in raw[:500] else raw
        ticker = submissions.get("tickers", [None])[0]

        total_chunks = 0
        sections_done: list[str] = []
        for section_name in ["Risk Factors", "MD&A", "Business", "Financial Statements"]:
            section_text = _extract_section(clean, section_name)
            if not section_text:
                continue
            chunks = chunk_document(section_text)
            meta = {
                "accession_number": accession_number,
                "form_type": form_type,
                "filing_date": filing_date_str,
            }
            n = await self._upsert_chunks(ticker, cik, form_type, section_name, chunks, meta)
            total_chunks += n
            sections_done.append(section_name)

        logger.info("Filing ingested", accession=accession_number, chunks=total_chunks)
        return {
            "accession_number": accession_number,
            "doc_type": form_type,
            "chunks_ingested": total_chunks,
            "sections": sections_done,
        }

    async def ingest_10k(
        self,
        cik: str = None,
        ticker: str = None,
        years: int = 3,
    ) -> list[dict]:
        """Download last N annual 10-K reports, extract key sections, embed, store.

        Args:
            cik:    Zero-padded 10-digit CIK. Resolved from ticker if omitted.
            ticker: Ticker symbol. Used to resolve CIK if cik is None.
            years:  Number of most-recent annual filings to ingest.

        Returns:
            List of per-filing result dicts.
        """
        if not cik and ticker:
            cik = await _resolve_cik(ticker)
        if not cik:
            return [{"error": f"CIK not found for ticker={ticker}"}]

        cik = cik.zfill(10)
        try:
            submissions = await _fetch_submissions(cik)
        except Exception as exc:
            return [{"error": str(exc)}]

        filings = _list_filings(submissions, "10-K", limit=years)
        if not ticker:
            ticker = (submissions.get("tickers") or [None])[0]

        results = []
        for filing in filings:
            accession = filing.get("accession")
            primary_doc = filing.get("primary_doc")
            period = filing.get("period", "")
            filing_date = filing.get("filing_date", "")

            if not accession or not primary_doc:
                results.append({"error": "Missing accession/doc", "period": period})
                continue

            try:
                raw = await _fetch_filing_text(cik, accession, primary_doc)
            except Exception as exc:
                results.append({"error": str(exc), "period": period})
                continue

            clean = _strip_html(raw) if "<" in raw[:1000] else raw
            filing_chunks = 0
            sections_done = []

            for section_name in ["Risk Factors", "MD&A", "Business", "Financial Statements"]:
                section_text = _extract_section(clean, section_name)
                if not section_text:
                    continue
                chunks = chunk_document(section_text)
                meta = {
                    "form_type": "10-K",
                    "accession_number": accession,
                    "period": period,
                    "filing_date": filing_date,
                }
                n = await self._upsert_chunks(ticker, cik, "10-K", section_name, chunks, meta)
                filing_chunks += n
                sections_done.append(section_name)

            results.append({
                "period": period,
                "filing_date": filing_date,
                "accession": accession,
                "chunks_ingested": filing_chunks,
                "sections": sections_done,
            })
            logger.info("10-K ingested", ticker=ticker, period=period, chunks=filing_chunks)

        return results

    async def ingest_10q(self, cik: str = None, ticker: str = None, years: int = 2) -> list[dict]:
        """Download last N*4 quarterly 10-Q reports, extract key sections, embed, store."""
        if not cik and ticker:
            cik = await _resolve_cik(ticker)
        if not cik:
            return [{"error": f"CIK not found for ticker={ticker}"}]

        cik = cik.zfill(10)
        try:
            submissions = await _fetch_submissions(cik)
        except Exception as exc:
            return [{"error": str(exc)}]

        filings = _list_filings(submissions, "10-Q", limit=years * 4)
        if not ticker:
            ticker = (submissions.get("tickers") or [None])[0]

        results = []
        for filing in filings:
            accession = filing.get("accession")
            primary_doc = filing.get("primary_doc")
            period = filing.get("period", "")

            if not accession or not primary_doc:
                continue

            try:
                raw = await _fetch_filing_text(cik, accession, primary_doc)
            except Exception as exc:
                results.append({"error": str(exc), "period": period})
                continue

            clean = _strip_html(raw) if "<" in raw[:1000] else raw
            filing_chunks = 0
            sections_done = []

            for section_name in ["MD&A", "Risk Factors"]:
                section_text = _extract_section(clean, section_name)
                if not section_text:
                    continue
                chunks = chunk_document(section_text)
                meta = {
                    "form_type": "10-Q",
                    "accession_number": accession,
                    "period": period,
                    "filing_date": filing.get("filing_date", ""),
                }
                n = await self._upsert_chunks(ticker, cik, "10-Q", section_name, chunks, meta)
                filing_chunks += n
                sections_done.append(section_name)

            results.append({
                "period": period,
                "accession": accession,
                "chunks_ingested": filing_chunks,
                "sections": sections_done,
            })

        return results

    async def ingest_earnings_transcript(
        self, ticker: str, quarter: Optional[str] = None
    ) -> dict:
        """Ingest earnings call transcript.

        Primary source: EDGAR 8-K Item 7.01 (Regulation FD) / Exhibit 99.1 press releases.
        Extracts call text and stores as doc_type='earnings_call'.

        Args:
            ticker:  Ticker symbol.
            quarter: Optional 'Q1 2024' filter; uses most recent if omitted.

        Returns:
            Ingest result dict.
        """
        cik = await _resolve_cik(ticker)
        if not cik:
            return {"error": f"CIK not found: {ticker}"}

        submissions = await _fetch_submissions(cik)
        # 8-K filings contain earnings press releases (Exhibit 99.1 + Items 7.01 / 8.01)
        filings = _list_filings(submissions, "8-K", limit=12)
        if not filings:
            return {"error": "No 8-K filings found", "ticker": ticker}

        target_filing = filings[0]  # most recent
        if quarter:
            # Try to find matching quarter by period
            for f in filings:
                period = f.get("period", "")
                if period and quarter.replace(" ", "") in period.replace("-", ""):
                    target_filing = f
                    break

        accession = target_filing.get("accession")
        primary_doc = target_filing.get("primary_doc")
        if not accession or not primary_doc:
            return {"error": "No primary document", "ticker": ticker}

        try:
            raw = await _fetch_filing_text(cik, accession, primary_doc)
        except Exception as exc:
            return {"error": str(exc), "ticker": ticker}

        clean = _strip_html(raw) if "<" in raw[:500] else raw
        # 8-K typically has press release as the body
        chunks = chunk_document(clean[:20000])
        meta = {
            "form_type": "8-K",
            "accession_number": accession,
            "period": target_filing.get("period", ""),
            "filing_date": target_filing.get("filing_date", ""),
        }
        n = await self._upsert_chunks(ticker, cik, "earnings_call", "Press Release", chunks, meta)
        return {
            "ticker": ticker,
            "quarter": quarter or target_filing.get("period"),
            "accession": accession,
            "chunks_ingested": n,
        }

    async def ingest_news_article(self, url: str, ticker: Optional[str] = None) -> dict:
        """Fetch a public news article by URL, extract text, embed, and store.

        Args:
            url:    Public URL of the article to ingest.
            ticker: Optional associated ticker for filtering.

        Returns:
            Ingest result dict.
        """
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
                resp.raise_for_status()
                raw = resp.text
        except Exception as exc:
            return {"error": str(exc), "url": url}

        clean = _strip_html(raw)
        # Rough content extraction: skip nav/header boilerplate
        paragraphs = [p.strip() for p in clean.split("\n") if len(p.strip()) > 80]
        article_text = " ".join(paragraphs)[:15000]

        if not article_text:
            return {"error": "No extractable text", "url": url}

        chunks = chunk_document(article_text)
        meta = {"url": url, "fetched_at": datetime.utcnow().isoformat()}
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        n = await self._upsert_chunks(ticker, None, "news_article", "Article", chunks, meta)
        return {
            "url": url,
            "ticker": ticker,
            "doc_id": url_hash,
            "chunks_ingested": n,
        }


# ── RAGQueryEngine ─────────────────────────────────────────────────────────────

class RAGQueryEngine:
    """Semantic and hybrid retrieval over financial_documents.

    Provides dense (pgvector cosine) + hybrid (pgvector + pg_trgm trigram) search
    plus LLM synthesis via Claude Haiku for question answering.
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def _make_filters(
        self,
        ticker: Optional[str],
        doc_types: Optional[list[str]],
        extra_clauses: list[str] = None,
    ) -> tuple[str, dict]:
        clauses: list[str] = list(extra_clauses or [])
        params: dict = {}
        if ticker:
            clauses.append("ticker = :ticker")
            params["ticker"] = ticker.upper()
        if doc_types:
            clauses.append("doc_type = ANY(:doc_types)")
            params["doc_types"] = doc_types
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    async def search(
        self,
        query: str,
        ticker: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        """Pure dense (pgvector cosine) search over financial_documents.

        Args:
            query:               Natural language query.
            ticker:              Optional ticker filter.
            doc_types:           Optional list of doc_type filters.
            top_k:               Maximum results.
            similarity_threshold: Minimum cosine similarity (0–1).

        Returns:
            List of result dicts ordered by similarity descending.
        """
        await _ensure_schema(self.db_url)
        q_embs = await _embed_in_executor([query])
        vec = _vec_str(q_embs[0])

        distance_threshold = 1 - similarity_threshold
        where, params = self._make_filters(
            ticker, doc_types,
            extra_clauses=["embedding IS NOT NULL",
                           f"(embedding <=> '{vec}'::vector) < :dist_thresh"]
        )
        params.update({"vec": vec, "top_k": top_k, "dist_thresh": distance_threshold})

        sql = text(f"""
            SELECT
                id, ticker, cik, doc_type, section, chunk_text, metadata,
                1 - (embedding <=> :vec::vector) AS similarity_score
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
                "chunk_text": r.chunk_text,
                "doc_type": r.doc_type,
                "section": r.section,
                "ticker": r.ticker,
                "similarity_score": round(float(r.similarity_score), 4),
                "metadata": r.metadata if isinstance(r.metadata, dict) else json.loads(r.metadata or "{}"),
            }
            for r in rows
        ]

    async def hybrid_search(
        self,
        query: str,
        ticker: Optional[str] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Hybrid dense + trigram search combining pgvector and pg_trgm.

        Scores: 0.7 * cosine_similarity + 0.3 * trigram_similarity, then re-ranks.
        Provides higher recall than pure vector search for financial acronyms and
        exact company/product names.

        Args:
            query:  Natural language or keyword query.
            ticker: Optional ticker filter.
            top_k:  Maximum results after re-ranking.

        Returns:
            List of result dicts ordered by combined score descending.
        """
        await _ensure_schema(self.db_url)
        q_embs = await _embed_in_executor([query])
        vec = _vec_str(q_embs[0])

        where, params = self._make_filters(ticker, None, extra_clauses=["embedding IS NOT NULL"])
        params.update({"vec": vec, "query": query, "top_k": top_k * 3})

        # Fetch over-sample for re-ranking
        sql = text(f"""
            SELECT
                id, ticker, doc_type, section, chunk_text, metadata,
                1 - (embedding <=> :vec::vector) AS dense_score,
                similarity(chunk_text, :query) AS trgm_score
            FROM financial_documents
            {where}
            ORDER BY embedding <=> :vec::vector
            LIMIT :top_k
        """)

        engine = _get_engine(self.db_url)
        async with engine.connect() as conn:
            rows = (await conn.execute(sql, params)).fetchall()

        results = []
        for r in rows:
            dense = float(r.dense_score)
            trgm = float(r.trgm_score)
            combined = 0.7 * dense + 0.3 * trgm
            results.append({
                "chunk_text": r.chunk_text,
                "doc_type": r.doc_type,
                "section": r.section,
                "ticker": r.ticker,
                "dense_score": round(dense, 4),
                "trgm_score": round(trgm, 4),
                "combined_score": round(combined, 4),
                "metadata": r.metadata if isinstance(r.metadata, dict) else json.loads(r.metadata or "{}"),
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k]

    async def get_context_for_llm(
        self,
        query: str,
        ticker: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> str:
        """Retrieve top search results formatted as numbered context passages for LLM prompts.

        Includes source metadata citations: [Source: AAPL 10-K 2024, Section: Risk Factors].

        Args:
            query:      Query string.
            ticker:     Optional ticker filter.
            max_tokens: Approximate token budget (1 token ≈ 4 chars).

        Returns:
            Formatted context string ready for LLM system/user prompt injection.
        """
        results = await self.search(query, ticker=ticker, top_k=15)
        context_parts = []
        char_budget = max_tokens * 4

        for i, r in enumerate(results, start=1):
            meta = r.get("metadata", {})
            form_type = meta.get("form_type", r["doc_type"])
            period = meta.get("period", "")
            source_label = f"{r['ticker'] or 'N/A'} {form_type} {period}".strip()
            header = f"[{i}] [Source: {source_label}, Section: {r['section']}]"
            passage = f"{header}\n{r['chunk_text']}"

            if sum(len(p) for p in context_parts) + len(passage) > char_budget:
                break
            context_parts.append(passage)

        return "\n\n".join(context_parts)

    async def answer_question(
        self,
        query: str,
        ticker: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> dict:
        """RAG question answering: retrieve context → call Claude Haiku → return answer.

        Graceful fallback: if no Anthropic API key is configured, returns search
        results with a 'No LLM configured' note instead of raising.

        Args:
            query:  Natural language question.
            ticker: Optional ticker to narrow retrieval.
            model:  Anthropic model ID. Defaults to claude-haiku-4-5-20251001.

        Returns:
            {answer, sources, confidence, model_used}
        """
        context = await self.get_context_for_llm(query, ticker=ticker, max_tokens=3500)
        sources = await self.search(query, ticker=ticker, top_k=5)

        settings = get_settings()
        api_key = settings.anthropic_api_key

        if not api_key:
            return {
                "answer": "No LLM configured — set ANTHROPIC_API_KEY for AI synthesis.",
                "sources": sources,
                "confidence": 0.0,
                "model_used": None,
                "context_passages": len(sources),
            }

        system_prompt = (
            "You are a senior financial analyst assistant. Using ONLY the document excerpts "
            "provided, answer the question with precision and cite excerpt numbers [N] where "
            "relevant. If the context does not contain enough information, say so explicitly."
        )
        user_prompt = (
            f"DOCUMENT EXCERPTS:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER (cite sources as [1], [2], etc.):"
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
                answer_text = msg.content[0].text if msg.content else ""
                tokens = msg.usage.input_tokens + msg.usage.output_tokens
                return answer_text, tokens

            answer, tokens = await loop.run_in_executor(None, _call)
            confidence = min(1.0, len(sources) / 10.0) if sources else 0.0

            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 2),
                "model_used": model,
                "tokens_used": tokens,
                "context_passages": len(sources),
            }

        except Exception as exc:
            logger.error("LLM synthesis failed", error=str(exc))
            return {
                "answer": f"LLM error: {exc}",
                "sources": sources,
                "confidence": 0.0,
                "model_used": model,
            }

    async def compare_across_filings(
        self,
        query: str,
        ticker: str,
        years: list[int],
    ) -> dict:
        """Answer the same question using each year's filings separately.

        Useful for trend analysis: 'What guidance did management give?' across 2022/2023/2024.

        Args:
            query:  Question to answer for each year.
            ticker: Ticker to search.
            years:  List of calendar years to compare.

        Returns:
            {ticker, query, comparisons: [{year, answer, sources}]}
        """
        comparisons = []
        for year in sorted(years):
            # Filter by filing period year via metadata
            await _ensure_schema(self.db_url)
            q_embs = await _embed_in_executor([query])
            vec = _vec_str(q_embs[0])

            sql = text("""
                SELECT chunk_text, doc_type, section, metadata,
                       1 - (embedding <=> :vec::vector) AS sim
                FROM financial_documents
                WHERE ticker = :ticker
                  AND embedding IS NOT NULL
                  AND (metadata->>'period' LIKE :year_pattern
                       OR metadata->>'filing_date' LIKE :year_pattern)
                ORDER BY embedding <=> :vec::vector
                LIMIT 8
            """)
            engine = _get_engine(self.db_url)
            async with engine.connect() as conn:
                rows = (await conn.execute(sql, {
                    "vec": vec,
                    "ticker": ticker.upper(),
                    "year_pattern": f"{year}%",
                })).fetchall()

            if not rows:
                comparisons.append({"year": year, "answer": "No data indexed for this year.", "sources": []})
                continue

            context = "\n\n".join(
                f"[{i+1}] [{r.doc_type} | {r.section}]\n{r.chunk_text}"
                for i, r in enumerate(rows)
            )
            sources = [{"chunk_text": r.chunk_text[:200], "doc_type": r.doc_type} for r in rows]

            settings = get_settings()
            if not settings.anthropic_api_key:
                comparisons.append({"year": year, "answer": "No LLM configured.", "sources": sources})
                continue

            try:
                import anthropic
                loop = asyncio.get_event_loop()

                def _call(ctx=context) -> str:
                    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                    msg = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=512,
                        messages=[{
                            "role": "user",
                            "content": f"EXCERPTS ({year}):\n{ctx}\n\nQUESTION: {query}\n\nANSWER:",
                        }],
                    )
                    return msg.content[0].text if msg.content else ""

                answer = await loop.run_in_executor(None, _call)
            except Exception as exc:
                answer = f"Error: {exc}"

            comparisons.append({"year": year, "answer": answer, "sources": sources})

        return {"ticker": ticker, "query": query, "comparisons": comparisons}

    async def extract_structured_data(
        self,
        ticker: str,
        fields: list[str],
    ) -> dict:
        """Use RAG to extract specific structured fields from indexed documents.

        Supported fields: guidance_revenue, guidance_eps, risk_factors_count,
        management_tone, revenue_trend, margin_commentary.

        Args:
            ticker: Ticker to query.
            fields: List of field names to extract.

        Returns:
            Dict mapping field name → extracted value.
        """
        field_queries = {
            "guidance_revenue": "What is the revenue guidance or outlook provided by management?",
            "guidance_eps": "What is the EPS or earnings per share guidance provided?",
            "risk_factors_count": "List the main risk factors facing the company",
            "management_tone": "What is the overall management tone: optimistic, cautious, or neutral?",
            "revenue_trend": "What is the trend in revenue growth year over year?",
            "margin_commentary": "What does management say about operating or gross margins?",
        }

        extracted = {}
        for field in fields:
            q = field_queries.get(field, f"What is the {field.replace('_', ' ')}?")
            result = await self.answer_question(q, ticker=ticker)
            extracted[field] = {
                "value": result.get("answer", ""),
                "sources_count": len(result.get("sources", [])),
                "confidence": result.get("confidence", 0.0),
            }

        return {"ticker": ticker, "fields": extracted}


# ── FastAPI Router ─────────────────────────────────────────────────────────────

rag_router = APIRouter(prefix="/api/rag", tags=["RAG Financial Docs"])


def _get_db_url() -> str:
    return get_settings().database_url


class IngestRequest(BaseModel):
    cik: Optional[str] = None
    years: int = 3


class AskRequest(BaseModel):
    query: str
    ticker: Optional[str] = None
    model: str = "claude-haiku-4-5-20251001"


class CompareRequest(BaseModel):
    query: str
    years: list[int] = [2022, 2023, 2024]


@rag_router.post("/ingest/{ticker}", summary="Ingest latest EDGAR filings for a ticker")
async def ingest_ticker(ticker: str, body: IngestRequest = None):
    """Download and index the latest 10-K and 10-Q filings for a ticker.

    Uses EDGAR as the primary source. Embeddings are stored in pgvector for
    subsequent semantic search.
    """
    if body is None:
        body = IngestRequest()
    db_url = _get_db_url()
    pipeline = DocumentIngestionPipeline(db_url)
    k_results = await pipeline.ingest_10k(cik=body.cik, ticker=ticker, years=body.years)
    q_results = await pipeline.ingest_10q(cik=body.cik, ticker=ticker, years=max(1, body.years - 1))
    total_chunks = sum(r.get("chunks_ingested", 0) for r in k_results + q_results)
    return {
        "ticker": ticker.upper(),
        "10k_filings": len(k_results),
        "10q_filings": len(q_results),
        "total_chunks": total_chunks,
        "detail": {"10k": k_results, "10q": q_results},
    }


@rag_router.get("/search", summary="Semantic search over financial documents")
async def search_documents(
    q: str = Query(..., description="Natural language search query"),
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    doc_type: Optional[str] = Query(None, description="Filter by doc_type (10-K, 10-Q, 8-K)"),
    top_k: int = Query(10, ge=1, le=50),
    hybrid: bool = Query(False, description="Use hybrid vector+trigram search"),
):
    """Semantic search over indexed financial documents using pgvector.

    Set hybrid=true to enable pg_trgm fallback for improved recall on
    exact financial terms and acronyms.
    """
    db_url = _get_db_url()
    engine = RAGQueryEngine(db_url)
    doc_types = [doc_type] if doc_type else None

    if hybrid:
        results = await engine.hybrid_search(q, ticker=ticker, top_k=top_k)
    else:
        results = await engine.search(q, ticker=ticker, doc_types=doc_types, top_k=top_k)

    return {"query": q, "count": len(results), "results": results}


@rag_router.post("/ask", summary="RAG question answering with LLM synthesis")
async def ask_question(body: AskRequest):
    """Answer a financial question using RAG + Claude Haiku.

    Retrieves relevant document chunks, builds a grounded context prompt, and
    calls the configured Anthropic model. Falls back gracefully if no API key.
    """
    db_url = _get_db_url()
    engine = RAGQueryEngine(db_url)
    result = await engine.answer_question(body.query, ticker=body.ticker, model=body.model)
    return result


@rag_router.get("/inventory", summary="List all indexed documents")
async def document_inventory(
    ticker: Optional[str] = Query(None, description="Filter inventory by ticker"),
):
    """Return a summary of indexed documents: ticker, doc_type, section, chunk count."""
    db_url = _get_db_url()
    index = DocumentIndex(db_url)
    df = await index.get_document_inventory(ticker=ticker)
    if df.empty:
        return {"count": 0, "documents": []}
    return {"count": len(df), "documents": df.to_dict(orient="records")}


@rag_router.get("/compare/{ticker}", summary="Multi-year filing comparison")
async def compare_filings(
    ticker: str,
    query: str = Query("What guidance did management provide?"),
    years: str = Query("2022,2023,2024", description="Comma-separated list of years"),
):
    """Compare answers to the same question across multiple years of filings.

    Useful for trend analysis of management guidance, risk factor evolution,
    and margin commentary over time.
    """
    year_list = [int(y.strip()) for y in years.split(",") if y.strip().isdigit()]
    if not year_list:
        raise HTTPException(status_code=422, detail="years must be comma-separated integers")
    db_url = _get_db_url()
    engine = RAGQueryEngine(db_url)
    return await engine.compare_across_filings(query, ticker=ticker, years=year_list)
