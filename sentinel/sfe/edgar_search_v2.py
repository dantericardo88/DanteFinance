"""
EDGAR Full-Text Search V2 — Advanced Textual Intelligence (dim_033, target 9).

Builds on edgar_full_text_search.py with:
  - AdvancedEFTSSearcher: Boolean/proximity search, field-specific queries, regex extraction
  - FilingContentExtractor: Deep 10-K/10-Q parsing, section extraction, table parsing,
    sentiment analysis per section
  - SemanticSearchEngine: sentence-transformers embeddings, SQLite vector store, BM25+semantic hybrid
  - FilingAlertSystem: real-time EDGAR RSS monitoring, watchlist, 8-K material event parsing

Public API
----------
AdvancedEFTSSearcher
    boolean_search(query, forms, days)        -> SearchResponse
    proximity_search(term1, term2, within_n, forms, days) -> SearchResponse
    field_search(entity, query, forms, days)  -> SearchResponse
    regex_extract(text, patterns)             -> dict[str, list[str]]

FilingContentExtractor
    extract_filing(cik, accession)            -> FilingContent
    extract_section(cik, accession, section)  -> str
    extract_tables(cik, accession)            -> list[dict]
    extract_financials(cik, accession)        -> dict
    section_sentiment(cik, accession)         -> dict[str, SectionSentiment]

SemanticSearchEngine
    index_filing(cik, accession, text)        -> int (segments indexed)
    semantic_search(query, n)                 -> list[SemanticResult]
    hybrid_search(query, n)                   -> list[SemanticResult]

FilingAlertSystem
    register_watch(ticker, cik, form_types)   -> str (watch_id)
    poll_feed(max_new)                        -> list[AlertEvent]
    get_alerts(watch_id)                      -> list[AlertEvent]

edgar_v2_router — FastAPI router, prefix /api/edgar/v2
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
import struct
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
from pydantic import BaseModel, Field

# Re-use helpers from the base module
from sentinel.sfe.edgar_full_text_search import (
    EDGARFullTextSearch,
    SearchQuery,
    SearchResponse,
    SearchResult,
    MaterialEvent,
    IntelligenceFeed,
    EFTS_BASE,
    EDGAR_BASE,
    EDGAR_ARCHIVES,
    EDGAR_BROWSE,
    _HEADERS,
    _TIMEOUT,
    _RATE_DELAY,
    _MAX_EFTS_HITS,
    _EFTS_SOURCE,
    _strip_html,
    _extract_section,
    _query_words,
    _count_mentions,
    _best_entity_name,
    _extract_cik,
    _SECTION_HEADERS,
    _fetch_text,
    SEARCH_TEMPLATES,
)

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DB_DIR  = Path(__file__).parent.parent / "data" / "db"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_EDGAR_DB = _DB_DIR / "edgar_search_v2.db"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EDGAR_RSS_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type={form_type}&dateb=&owner=include&count=40&output=atom"
)

_8K_MATERIAL_ITEMS = {
    "1.01": "Entry into Material Agreement",
    "1.02": "Termination of Material Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition",
    "2.02": "Results of Operations",
    "2.04": "Triggering Events",
    "2.05": "Cost Associated with Exit",
    "2.06": "Material Impairments",
    "3.01": "Delisting Notice",
    "4.01": "Auditor Changes",
    "4.02": "Non-Reliance Restatement",
    "5.01": "Change of Control",
    "5.02": "Departure of Directors/Officers",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements",
}

# Regex patterns for financial data extraction
_FINANCIAL_PATTERNS: dict[str, str] = {
    "revenue":          r"(?:revenue|net revenue|total revenue)[^\d]*\$([\d,\.]+)\s*(million|billion)?",
    "eps":              r"(?:earnings per share|EPS|diluted EPS)[^\d]*\$([\d\.]+)",
    "net_income":       r"net income[^\d]*\$([\d,\.]+)\s*(million|billion)?",
    "gross_margin":     r"gross (?:margin|profit margin)[^\d]*([\d\.]+)\s*%",
    "guidance_high":    r"guidance[^\d]*\$([\d,\.]+)\s*(?:to|-)\s*\$([\d,\.]+)",
    "shares_out":       r"(?:shares outstanding|weighted.average shares)[^\d]*([\d,\.]+)",
    "debt":             r"(?:total debt|long.term debt)[^\d]*\$([\d,\.]+)\s*(million|billion)?",
    "cash":             r"(?:cash and cash equivalents)[^\d]*\$([\d,\.]+)\s*(million|billion)?",
    "operating_income": r"(?:operating income|income from operations)[^\d]*\$([\d,\.]+)\s*(million|billion)?",
}

# Positive/negative word lists for simple lexicon sentiment
_POSITIVE_WORDS = {
    "growth", "increase", "record", "strong", "exceeded", "outperform", "improve",
    "expansion", "profitability", "momentum", "confident", "opportunity", "robust",
    "accelerate", "gain", "success", "positive", "ahead", "beat", "recovery",
}
_NEGATIVE_WORDS = {
    "decline", "decrease", "loss", "risk", "headwind", "challenge", "uncertainty",
    "pressure", "concern", "adverse", "weak", "difficult", "impairment", "default",
    "breach", "litigation", "investigation", "restatement", "going concern", "failure",
}

# Extended 10-K section headers
_EXTENDED_SECTION_HEADERS: dict[str, list[str]] = {
    "business":                 ["Item 1.", "Item 1 ", "ITEM 1 ", "ITEM 1."],
    "risk_factors":             ["Item 1A", "ITEM 1A", "Risk Factors"],
    "unresolved_staff_comments": ["Item 1B", "ITEM 1B"],
    "properties":               ["Item 2.", "ITEM 2 "],
    "legal_proceedings":        ["Item 3", "ITEM 3"],
    "mine_safety":              ["Item 4.", "ITEM 4 "],
    "market_info":              ["Item 5.", "ITEM 5 "],
    "selected_financials":      ["Item 6.", "ITEM 6 "],
    "mda":                      ["Item 7.", "ITEM 7 ", "MD&A", "Management's Discussion"],
    "market_risk":              ["Item 7A", "ITEM 7A"],
    "financial_statements":     ["Item 8.", "ITEM 8 "],
    "controls":                 ["Item 9A", "ITEM 9A"],
    "other_info":               ["Item 9B", "ITEM 9B"],
    "risk_governance":          ["Item 10.", "ITEM 10"],
    "exec_compensation":        ["Item 11.", "ITEM 11"],
    "security_ownership":       ["Item 12.", "ITEM 12"],
    "certain_relationships":    ["Item 13.", "ITEM 13"],
    "principal_accountant":     ["Item 14.", "ITEM 14"],
    "exhibits":                 ["Item 15.", "ITEM 15"],
}


# ---------------------------------------------------------------------------
# Pydantic models (V2-specific)
# ---------------------------------------------------------------------------


class BooleanSearchQuery(BaseModel):
    must:      list[str] = Field(default_factory=list, description="AND terms (all must appear)")
    should:    list[str] = Field(default_factory=list, description="OR terms (at least one)")
    must_not:  list[str] = Field(default_factory=list, description="NOT terms (none must appear)")
    form_types: list[str] = Field(default_factory=list)
    start_date: Optional[date] = None
    end_date:   Optional[date] = None
    limit:      int = 50


class ProximityQuery(BaseModel):
    term1:      str
    term2:      str
    within_n:   int = 50                  # words
    form_types: list[str] = Field(default_factory=list)
    days_back:  int = 30
    limit:      int = 50


class FieldSearchQuery(BaseModel):
    entity_name:  Optional[str] = None
    query:        str
    form_types:   list[str] = Field(default_factory=list)
    days_back:    int = 90
    limit:        int = 50


class SectionSentiment(BaseModel):
    section:         str
    text_length:     int
    positive_count:  int
    negative_count:  int
    sentiment_score: float     # -1 to +1
    tone:            str       # "positive" | "neutral" | "negative"
    top_positive:    list[str]
    top_negative:    list[str]
    excerpt:         Optional[str] = None


class FilingContent(BaseModel):
    cik:           str
    accession:     str
    entity_name:   Optional[str] = None
    form_type:     Optional[str] = None
    filed_date:    Optional[date] = None
    sections:      dict[str, str]    = Field(default_factory=dict)
    financials:    dict[str, Any]    = Field(default_factory=dict)
    sentiment:     dict[str, Any]    = Field(default_factory=dict)
    tables_count:  int = 0


class SemanticResult(BaseModel):
    segment_id:    str
    cik:           str
    accession:     str
    entity_name:   Optional[str] = None
    section:       Optional[str] = None
    text:          str
    keyword_score: float    # BM25-like score
    semantic_score: float   # cosine similarity
    hybrid_score:  float    # 70% keyword + 30% semantic
    filed_date:    Optional[str] = None


class WatchEntry(BaseModel):
    watch_id:   str
    ticker:     str
    cik:        Optional[str] = None
    form_types: list[str]
    created_at: datetime
    last_polled: Optional[datetime] = None


class AlertEvent(BaseModel):
    alert_id:    str
    watch_id:    str
    ticker:      str
    cik:         Optional[str] = None
    form_type:   str
    filed_date:  date
    entity_name: str
    accession:   str
    url:         str
    headline:    str
    item_codes:  list[str]    # 8-K item codes
    severity:    str
    detected_at: datetime


# ---------------------------------------------------------------------------
# SQLite schema for V2
# ---------------------------------------------------------------------------


def _init_edgar_db(db_path: Path = _EDGAR_DB) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS filing_segments (
            segment_id   TEXT PRIMARY KEY,
            cik          TEXT NOT NULL,
            accession    TEXT NOT NULL,
            entity_name  TEXT,
            form_type    TEXT,
            section      TEXT,
            text         TEXT NOT NULL,
            embedding    BLOB,           -- serialized float32 array
            filed_date   TEXT,
            indexed_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_seg_cik ON filing_segments(cik);
        CREATE INDEX IF NOT EXISTS idx_seg_entity ON filing_segments(entity_name);
        CREATE INDEX IF NOT EXISTS idx_seg_date ON filing_segments(filed_date);

        CREATE TABLE IF NOT EXISTS watchlist (
            watch_id    TEXT PRIMARY KEY,
            ticker      TEXT NOT NULL,
            cik         TEXT,
            form_types  TEXT,           -- JSON list
            created_at  TEXT DEFAULT (datetime('now')),
            last_polled TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id    TEXT PRIMARY KEY,
            watch_id    TEXT NOT NULL,
            ticker      TEXT,
            cik         TEXT,
            form_type   TEXT,
            filed_date  TEXT,
            entity_name TEXT,
            accession   TEXT,
            url         TEXT,
            headline    TEXT,
            item_codes  TEXT,           -- JSON list
            severity    TEXT,
            detected_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_watch ON alerts(watch_id);
        CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker);
        """
    )
    conn.commit()
    conn.close()


_init_edgar_db()


def _db_conn(db_path: Path = _EDGAR_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ---------------------------------------------------------------------------
# Embedding helpers (sentence-transformers with graceful fallback)
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_EMBED_DIM   = 384


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("edgar_v2: sentence-transformers loaded (all-MiniLM-L6-v2)")
    except ImportError:
        logger.warning("edgar_v2: sentence-transformers not installed — semantic search disabled")
        _EMBED_MODEL = None
    return _EMBED_MODEL


def _embed_text(text: str) -> Optional[bytes]:
    """Return a bytes-serialized float32 embedding or None if unavailable."""
    model = _get_embed_model()
    if model is None:
        return None
    try:
        vec = model.encode(text[:512], normalize_embeddings=True)
        return struct.pack(f"{len(vec)}f", *vec)
    except Exception as exc:
        logger.debug("edgar_v2: embed_text failed", error=str(exc))
        return None


def _cosine(a_bytes: bytes, b_bytes: bytes) -> float:
    """Cosine similarity between two serialized float32 vectors."""
    n = len(a_bytes) // 4
    if n == 0:
        return 0.0
    a = struct.unpack(f"{n}f", a_bytes)
    b = struct.unpack(f"{n}f", b_bytes)
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# BM25-lite scorer
# ---------------------------------------------------------------------------


class _BM25Lite:
    """
    Minimal BM25 scorer over a list of text documents.
    k1=1.5, b=0.75 — standard OKAPI BM25 defaults.
    """

    K1 = 1.5
    B  = 0.75

    def __init__(self, corpus: list[str]) -> None:
        self._corpus = corpus
        self._n      = len(corpus)
        self._tok    = [self._tokenize(d) for d in corpus]
        avg_dl       = sum(len(t) for t in self._tok) / max(self._n, 1)
        self._avg_dl = avg_dl
        # Document frequency
        self._df: Counter = Counter()
        for doc_toks in self._tok:
            self._df.update(set(doc_toks))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z]{2,}", text.lower())

    def score(self, query: str, doc_idx: int) -> float:
        q_toks  = self._tokenize(query)
        d_toks  = self._tok[doc_idx]
        dl      = len(d_toks)
        tf_map  = Counter(d_toks)
        score   = 0.0
        for term in q_toks:
            tf  = tf_map.get(term, 0)
            df  = self._df.get(term, 0)
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            tfc = (tf * (self.K1 + 1)) / (tf + self.K1 * (1 - self.B + self.B * dl / max(self._avg_dl, 1)))
            score += idf * tfc
        return score

    def rank(self, query: str, top_n: int = 10) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self._n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# ---------------------------------------------------------------------------
# AdvancedEFTSSearcher
# ---------------------------------------------------------------------------


class AdvancedEFTSSearcher:
    """
    Enhanced EDGAR EFTS search with Boolean operators, proximity, and field search.
    Wraps EDGARFullTextSearch for base functionality.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._base    = EDGARFullTextSearch(timeout=timeout)
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Boolean search
    # ------------------------------------------------------------------

    def boolean_search(self, bq: BooleanSearchQuery) -> SearchResponse:
        """
        Translate must/should/must_not into an EFTS query string.
        EFTS uses Elasticsearch query syntax.
        """
        parts: list[str] = []

        # AND terms
        for term in bq.must:
            t = term.strip()
            if " " in t:
                parts.append(f'"{t}"')
            else:
                parts.append(t)

        # OR group
        if bq.should:
            or_parts = []
            for term in bq.should:
                t = term.strip()
                or_parts.append(f'"{t}"' if " " in t else t)
            parts.append(f"({' OR '.join(or_parts)})")

        # NOT terms
        for term in bq.must_not:
            t = term.strip()
            parts.append(f'NOT "{t}"' if " " in t else f"NOT {t}")

        query_str = " AND ".join(parts) if parts else "*"

        q = SearchQuery(
            query=query_str,
            form_types=bq.form_types,
            start_date=bq.start_date,
            end_date=bq.end_date or date.today(),
            limit=bq.limit,
        )
        return asyncio.run(self._base.search(q))

    # ------------------------------------------------------------------
    # Proximity search
    # ------------------------------------------------------------------

    def proximity_search(self, pq: ProximityQuery) -> SearchResponse:
        """
        Search for filings where term1 and term2 appear within N words of each other.
        EFTS doesn't support native proximity, so we: (1) run a broad AND search,
        (2) fetch filing text, (3) filter by actual proximity.
        """
        today = date.today()
        broad_q = SearchQuery(
            query=f'"{pq.term1}" AND "{pq.term2}"',
            form_types=pq.form_types,
            start_date=today - timedelta(days=pq.days_back),
            end_date=today,
            limit=min(pq.limit * 4, _MAX_EFTS_HITS),
        )
        broad_resp = asyncio.run(self._base.search(broad_q))

        # Post-filter by actual text proximity
        filtered: list[SearchResult] = []

        async def _check_all() -> None:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                for r in broad_resp.results:
                    if len(filtered) >= pq.limit:
                        break
                    text = await _fetch_text(client, r.file_url)
                    if self._check_proximity(text, pq.term1, pq.term2, pq.within_n):
                        filtered.append(r)
                    await asyncio.sleep(_RATE_DELAY)

        asyncio.run(_check_all())

        logger.info(
            "edgar_v2: proximity_search",
            term1=pq.term1, term2=pq.term2, within_n=pq.within_n,
            broad_hits=len(broad_resp.results), proximity_matches=len(filtered),
        )
        return SearchResponse(
            query=broad_q,
            total_hits=len(filtered),
            results=filtered,
        )

    @staticmethod
    def _check_proximity(text: str, term1: str, term2: str, within_n: int) -> bool:
        """Return True if term1 and term2 appear within within_n words in text."""
        if not text:
            return False
        words   = re.findall(r"\S+", text.lower())
        t1_low  = term1.lower()
        t2_low  = term2.lower()
        positions1 = [i for i, w in enumerate(words) if t1_low in w]
        positions2 = [i for i, w in enumerate(words) if t2_low in w]
        for p1 in positions1:
            for p2 in positions2:
                if abs(p1 - p2) <= within_n:
                    return True
        return False

    # ------------------------------------------------------------------
    # Field-specific search
    # ------------------------------------------------------------------

    def field_search(self, fsq: FieldSearchQuery) -> SearchResponse:
        """
        Search for query within filings from a specific entity.
        Combines entity name and keyword into a structured EFTS query.
        """
        today = date.today()
        if fsq.entity_name:
            query_str = f'"{fsq.entity_name}" AND ({fsq.query})'
        else:
            query_str = fsq.query

        q = SearchQuery(
            query=query_str,
            form_types=fsq.form_types,
            start_date=today - timedelta(days=fsq.days_back),
            end_date=today,
            entity_name=fsq.entity_name,
            limit=fsq.limit,
        )
        return asyncio.run(self._base.search(q))

    # ------------------------------------------------------------------
    # Regex extraction
    # ------------------------------------------------------------------

    def regex_extract(
        self,
        text: str,
        patterns: Optional[dict[str, str]] = None,
    ) -> dict[str, list[str]]:
        """
        Apply regex patterns to text and return all matches.
        Defaults to _FINANCIAL_PATTERNS if no custom patterns supplied.
        """
        pats = patterns or _FINANCIAL_PATTERNS
        results: dict[str, list[str]] = {}
        clean  = _strip_html(text)
        for name, pattern in pats.items():
            try:
                matches = re.findall(pattern, clean, re.IGNORECASE)
                # Flatten tuples
                flat = []
                for m in matches:
                    if isinstance(m, tuple):
                        flat.append(" ".join(p for p in m if p))
                    else:
                        flat.append(str(m))
                results[name] = flat
            except re.error as exc:
                logger.debug("edgar_v2: regex_extract pattern error", name=name, error=str(exc))
                results[name] = []
        return results

    # ------------------------------------------------------------------
    # Multi-keyword convenience
    # ------------------------------------------------------------------

    def multi_keyword_search(
        self,
        keywords: list[str],
        operator: str = "AND",
        form_types: Optional[list[str]] = None,
        days_back: int = 30,
        limit: int = 50,
    ) -> SearchResponse:
        """Search for multiple keywords combined with AND/OR."""
        op   = operator.upper().strip()
        if op not in ("AND", "OR"):
            raise ValueError(f"operator must be 'AND' or 'OR', got '{operator}'")

        quoted = [f'"{kw}"' if " " in kw else kw for kw in keywords]
        query  = f" {op} ".join(quoted)

        today = date.today()
        q = SearchQuery(
            query=query,
            form_types=form_types or [],
            start_date=today - timedelta(days=days_back),
            end_date=today,
            limit=limit,
        )
        return asyncio.run(self._base.search(q))


# ---------------------------------------------------------------------------
# FilingContentExtractor
# ---------------------------------------------------------------------------


class FilingContentExtractor:
    """
    Downloads and deeply parses EDGAR 10-K / 10-Q filings.
    Extracts named sections, HTML tables, financial metrics, and per-section sentiment.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Master extract
    # ------------------------------------------------------------------

    def extract_filing(self, cik: str, accession: str) -> FilingContent:
        """
        Full extraction of a filing: all sections, financials, sentiment.
        """
        return asyncio.run(self._async_extract(cik, accession))

    async def _async_extract(self, cik: str, accession: str) -> FilingContent:
        acc_nd    = accession.replace("-", "")
        cik_plain = str(cik).lstrip("0") or cik

        # Try to resolve the primary document URL
        primary_url = await self._resolve_primary_doc(cik_plain, acc_nd)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            raw = await _fetch_text(client, primary_url) if primary_url else ""
            if not raw:
                # Fallback: index page
                index_url = f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nd}/{acc_nd}-index.htm"
                raw = await _fetch_text(client, index_url)

        clean = _strip_html(raw)

        # Extract all sections
        sections: dict[str, str] = {}
        for section_key, headers in _EXTENDED_SECTION_HEADERS.items():
            for header in headers:
                idx = clean.find(header)
                if idx >= 0:
                    # Find end at next item header
                    next_re = re.compile(r"\bItem\s+\d+[A-Z]?\b", re.IGNORECASE)
                    end_idx = len(clean)
                    for m in next_re.finditer(clean, idx + len(header) + 10):
                        end_idx = m.start()
                        break
                    section_text = clean[idx:end_idx].strip()
                    if len(section_text) > 100:
                        sections[section_key] = section_text[:10_000]
                        break

        # Financial metric extraction
        financials = self._extract_financials_from_text(clean)

        # Sentiment per section
        sentiment = {}
        for sec_name, sec_text in sections.items():
            if sec_name in ("mda", "risk_factors", "business"):
                s = self._compute_sentiment(sec_name, sec_text)
                sentiment[sec_name] = s.model_dump()

        # Count HTML tables (basic)
        table_count = len(re.findall(r"<table", raw, re.IGNORECASE))

        # Try to get entity metadata from filing index JSON
        entity_name, form_type, filed_date = await self._get_filing_meta(cik_plain, acc_nd)

        return FilingContent(
            cik=cik,
            accession=accession,
            entity_name=entity_name,
            form_type=form_type,
            filed_date=filed_date,
            sections=sections,
            financials=financials,
            sentiment=sentiment,
            tables_count=table_count,
        )

    async def _resolve_primary_doc(self, cik: str, acc_nd: str) -> Optional[str]:
        """Fetch the filing index and return the primary .htm document URL."""
        index_json_url = f"{EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{EDGAR_ARCHIVES}/{cik}/{acc_nd}/{acc_nd}-index.json",
                    headers=_HEADERS,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    docs = data.get("documents", [])
                    for doc in docs:
                        if doc.get("type", "").upper() in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                            fname = doc.get("filename", "")
                            if fname:
                                return f"{EDGAR_ARCHIVES}/{cik}/{acc_nd}/{fname}"
        except Exception:
            pass
        # Fallback: guess primary doc name
        return f"{EDGAR_ARCHIVES}/{cik}/{acc_nd}/{acc_nd}.htm"

    async def _get_filing_meta(
        self, cik: str, acc_nd: str
    ) -> tuple[Optional[str], Optional[str], Optional[date]]:
        """Retrieve entity name, form type, filed date from EDGAR index JSON."""
        url = f"{EDGAR_ARCHIVES}/{cik}/{acc_nd}/{acc_nd}-index.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=_HEADERS)
                if resp.status_code == 200:
                    data = resp.json()
                    entity  = data.get("entityName")
                    form    = data.get("formType")
                    fd_str  = data.get("filedAt", data.get("dateFiled", ""))[:10]
                    fd      = date.fromisoformat(fd_str) if fd_str else None
                    return entity, form, fd
        except Exception:
            pass
        return None, None, None

    # ------------------------------------------------------------------
    # Section extraction (async)
    # ------------------------------------------------------------------

    def extract_section(self, cik: str, accession: str, section: str) -> str:
        """
        Extract a single named section from a filing.
        section: one of the keys in _EXTENDED_SECTION_HEADERS (e.g. 'mda', 'risk_factors')
        """
        content = self.extract_filing(cik, accession)
        return content.sections.get(section, "")

    # ------------------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------------------

    def extract_tables(self, cik: str, accession: str) -> list[dict]:
        """
        Download filing HTML and parse <table> elements into dicts.
        Returns list of {headers: [...], rows: [[...], ...], table_index: N}.
        """
        return asyncio.run(self._async_extract_tables(cik, accession))

    async def _async_extract_tables(self, cik: str, accession: str) -> list[dict]:
        cik_plain = str(cik).lstrip("0") or cik
        acc_nd    = accession.replace("-", "")
        url       = f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nd}/{acc_nd}.htm"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp_text = await _fetch_text(client, url)

        if not resp_text:
            return []

        # Use regex-based table parser (no lxml dependency required)
        tables: list[dict] = []
        table_pattern = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
        row_pattern   = re.compile(r"<tr[^>]*>(.*?)</tr>",    re.IGNORECASE | re.DOTALL)
        cell_pattern  = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)

        for t_idx, table_match in enumerate(table_pattern.finditer(resp_text)):
            table_html = table_match.group(1)
            rows_raw   = row_pattern.findall(table_html)
            parsed_rows: list[list[str]] = []

            for row_html in rows_raw:
                cells = [
                    _strip_html(c).strip()
                    for c in cell_pattern.findall(row_html)
                    if _strip_html(c).strip()
                ]
                if cells:
                    parsed_rows.append(cells)

            if not parsed_rows:
                continue

            # Heuristic: first row with all non-numeric content → header
            headers = parsed_rows[0] if parsed_rows else []
            data_rows = parsed_rows[1:] if len(parsed_rows) > 1 else []

            # Only keep tables with financial-looking content (numbers)
            has_numbers = any(
                re.search(r"\d", cell)
                for row in data_rows
                for cell in row
            )
            if not has_numbers:
                continue

            tables.append({
                "table_index": t_idx,
                "headers":     headers,
                "rows":        data_rows[:50],      # cap at 50 data rows
                "row_count":   len(data_rows),
                "col_count":   len(headers),
            })

            if len(tables) >= 20:   # cap total tables returned
                break

        return tables

    # ------------------------------------------------------------------
    # Financial metric extraction
    # ------------------------------------------------------------------

    def extract_financials(self, cik: str, accession: str) -> dict[str, Any]:
        """Extract key financial metrics from filing text using regex patterns."""
        content = self.extract_filing(cik, accession)
        return content.financials

    def _extract_financials_from_text(self, text: str) -> dict[str, Any]:
        searcher = AdvancedEFTSSearcher()
        raw      = searcher.regex_extract(text, _FINANCIAL_PATTERNS)

        result: dict[str, Any] = {}
        for metric, matches in raw.items():
            if not matches:
                continue
            # Take first match; attempt numeric parse
            val_str = matches[0].replace(",", "").strip()
            try:
                # Handle "million" / "billion" suffix
                if "billion" in val_str.lower():
                    val_str = re.sub(r"\s*billion.*", "", val_str, flags=re.IGNORECASE)
                    result[metric] = float(val_str) * 1e9
                elif "million" in val_str.lower():
                    val_str = re.sub(r"\s*million.*", "", val_str, flags=re.IGNORECASE)
                    result[metric] = float(val_str) * 1e6
                else:
                    result[metric] = float(val_str)
            except ValueError:
                result[metric] = matches[0]

        return result

    # ------------------------------------------------------------------
    # Per-section sentiment
    # ------------------------------------------------------------------

    def section_sentiment(self, cik: str, accession: str) -> dict[str, SectionSentiment]:
        """Compute tone/sentiment for each extracted section."""
        content = self.extract_filing(cik, accession)
        result: dict[str, SectionSentiment] = {}
        for sec_name, sec_text in content.sections.items():
            result[sec_name] = self._compute_sentiment(sec_name, sec_text)
        return result

    def _compute_sentiment(self, section: str, text: str) -> SectionSentiment:
        """Lexicon-based sentiment for a section of filing text."""
        words    = re.findall(r"[a-zA-Z]+", text.lower())
        word_set = set(words)

        pos_hits = sorted(_POSITIVE_WORDS & word_set)
        neg_hits = sorted(_NEGATIVE_WORDS & word_set)

        # Multi-word phrase matching
        for phrase in ("going concern", "reduction in force", "net loss", "unable to"):
            if phrase in text.lower():
                neg_hits.append(phrase)

        pos_count = sum(text.lower().count(w) for w in _POSITIVE_WORDS)
        neg_count = sum(text.lower().count(w) for w in _NEGATIVE_WORDS)
        total     = pos_count + neg_count

        score = (pos_count - neg_count) / max(total, 1)
        tone  = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"

        # Brief excerpt from section
        excerpt = text[:300].replace("\n", " ").strip() if text else None

        return SectionSentiment(
            section=section,
            text_length=len(text),
            positive_count=pos_count,
            negative_count=neg_count,
            sentiment_score=round(score, 4),
            tone=tone,
            top_positive=pos_hits[:10],
            top_negative=neg_hits[:10],
            excerpt=excerpt,
        )


# ---------------------------------------------------------------------------
# SemanticSearchEngine
# ---------------------------------------------------------------------------


class SemanticSearchEngine:
    """
    Semantic search over EDGAR filing segments using sentence-transformers.
    Stores embeddings in SQLite as serialized float32 arrays.
    Falls back to BM25-only if sentence-transformers is unavailable.
    """

    _SEGMENT_MAX_CHARS = 800
    _SEGMENT_OVERLAP   = 100

    def __init__(self, db_path: Path = _EDGAR_DB) -> None:
        self._db_path  = db_path
        self._extractor = FilingContentExtractor()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_filing(
        self,
        cik: str,
        accession: str,
        entity_name: Optional[str] = None,
        form_type:   Optional[str] = None,
        filed_date:  Optional[str] = None,
    ) -> int:
        """
        Extract filing text, segment it, embed each segment, and store in SQLite.
        Returns number of segments indexed.
        """
        content = self._extractor.extract_filing(cik, accession)
        all_text = "\n\n".join(
            f"[{section}]\n{text}"
            for section, text in content.sections.items()
            if text
        )

        if not all_text:
            logger.warning("edgar_v2: index_filing — no text extracted", cik=cik, acc=accession)
            return 0

        segments = self._segment_text(all_text)
        ent_name = entity_name or content.entity_name or "Unknown"
        f_type   = form_type   or content.form_type or "Unknown"
        f_date   = filed_date  or (content.filed_date.isoformat() if content.filed_date else None)

        # Detect section for each segment
        section_labels = self._label_sections(segments, content.sections)

        inserted = 0
        with _db_conn(self._db_path) as conn:
            for i, seg in enumerate(segments):
                seg_id    = hashlib.sha256(f"{cik}|{accession}|{i}|{seg[:50]}".encode()).hexdigest()[:20]
                embedding = _embed_text(seg)
                section   = section_labels.get(i)

                conn.execute(
                    """INSERT OR IGNORE INTO filing_segments
                       (segment_id, cik, accession, entity_name, form_type,
                        section, text, embedding, filed_date)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (seg_id, cik, accession, ent_name, f_type,
                     section, seg, embedding, f_date),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1

        logger.info("edgar_v2: index_filing", cik=cik, acc=accession, segments=inserted)
        return inserted

    def _segment_text(self, text: str) -> list[str]:
        """Split text into overlapping segments of _SEGMENT_MAX_CHARS."""
        segments: list[str] = []
        step = self._SEGMENT_MAX_CHARS - self._SEGMENT_OVERLAP
        for start in range(0, len(text), step):
            seg = text[start: start + self._SEGMENT_MAX_CHARS].strip()
            if len(seg) > 50:
                segments.append(seg)
        return segments

    def _label_sections(
        self, segments: list[str], sections: dict[str, str]
    ) -> dict[int, str]:
        """Best-effort: match each segment to a section by text prefix."""
        labels: dict[int, str] = {}
        for i, seg in enumerate(segments):
            for sec_name, sec_text in sections.items():
                if sec_text and seg[:100] in sec_text:
                    labels[i] = sec_name
                    break
        return labels

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def semantic_search(self, query: str, n: int = 10) -> list[SemanticResult]:
        """
        Find the n most semantically similar filing segments to query.
        Falls back to keyword-only if embeddings unavailable.
        """
        q_emb = _embed_text(query)

        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM filing_segments ORDER BY indexed_at DESC LIMIT 2000"
            ).fetchall()

        if not rows:
            return []

        texts = [r["text"] for r in rows]

        # BM25 scores (always computed)
        bm25    = _BM25Lite(texts)
        bm25_ranks = {idx: score for idx, score in bm25.rank(query, top_n=len(texts))}

        # Normalise BM25 scores to 0-1
        max_bm25 = max(bm25_ranks.values(), default=1.0) or 1.0

        results: list[SemanticResult] = []
        for i, row in enumerate(rows):
            kw_score = bm25_ranks.get(i, 0.0) / max_bm25

            # Semantic score
            sem_score = 0.0
            if q_emb and row["embedding"]:
                try:
                    sem_score = max(0.0, _cosine(q_emb, bytes(row["embedding"])))
                except Exception:
                    sem_score = 0.0

            hybrid = kw_score * 0.70 + sem_score * 0.30

            if hybrid < 0.01:
                continue

            results.append(
                SemanticResult(
                    segment_id=row["segment_id"],
                    cik=row["cik"],
                    accession=row["accession"],
                    entity_name=row["entity_name"],
                    section=row["section"],
                    text=row["text"],
                    keyword_score=round(kw_score, 4),
                    semantic_score=round(sem_score, 4),
                    hybrid_score=round(hybrid, 4),
                    filed_date=row["filed_date"],
                )
            )

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:n]

    def hybrid_search(self, query: str, n: int = 10) -> list[SemanticResult]:
        """Alias for semantic_search — BM25+semantic hybrid is always used."""
        return self.semantic_search(query, n=n)

    # ------------------------------------------------------------------
    # Index stats
    # ------------------------------------------------------------------

    def index_stats(self) -> dict[str, Any]:
        with _db_conn(self._db_path) as conn:
            total   = conn.execute("SELECT COUNT(*) FROM filing_segments").fetchone()[0]
            filings = conn.execute("SELECT COUNT(DISTINCT accession) FROM filing_segments").fetchone()[0]
            entities = conn.execute("SELECT COUNT(DISTINCT entity_name) FROM filing_segments").fetchone()[0]
            has_emb  = conn.execute(
                "SELECT COUNT(*) FROM filing_segments WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        return {
            "total_segments":      total,
            "unique_filings":      filings,
            "unique_entities":     entities,
            "segments_with_embed": has_emb,
            "embed_coverage_pct":  round(has_emb / max(total, 1) * 100, 1),
            "embed_model":         "all-MiniLM-L6-v2" if _get_embed_model() else "unavailable",
        }


# ---------------------------------------------------------------------------
# FilingAlertSystem
# ---------------------------------------------------------------------------


class FilingAlertSystem:
    """
    Real-time EDGAR filing alert system.
    Monitors EDGAR RSS feeds for new filings matching a watchlist.
    Parses 8-K filings to extract material item codes.
    """

    def __init__(self, db_path: Path = _EDGAR_DB, timeout: float = _TIMEOUT) -> None:
        self._db_path = db_path
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Watchlist management
    # ------------------------------------------------------------------

    def register_watch(
        self,
        ticker:     str,
        cik:        Optional[str] = None,
        form_types: Optional[list[str]] = None,
    ) -> str:
        """
        Register a company + filing type combination to watch.
        Returns a watch_id for subsequent polling.
        """
        watch_id   = hashlib.md5(f"{ticker}|{cik}|{json.dumps(sorted(form_types or []))}".encode()).hexdigest()[:16]
        form_json  = json.dumps(form_types or ["8-K", "10-Q", "10-K"])
        now        = datetime.utcnow().isoformat()

        with _db_conn(self._db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO watchlist (watch_id, ticker, cik, form_types, created_at)
                   VALUES (?,?,?,?,?)""",
                (watch_id, ticker.upper(), cik, form_json, now),
            )
        logger.info("edgar_v2: watch registered", ticker=ticker, watch_id=watch_id)
        return watch_id

    def list_watches(self) -> list[WatchEntry]:
        with _db_conn(self._db_path) as conn:
            rows = conn.execute("SELECT * FROM watchlist ORDER BY created_at DESC").fetchall()
        return [
            WatchEntry(
                watch_id=r["watch_id"],
                ticker=r["ticker"],
                cik=r["cik"],
                form_types=json.loads(r["form_types"] or "[]"),
                created_at=datetime.fromisoformat(r["created_at"]),
                last_polled=datetime.fromisoformat(r["last_polled"]) if r["last_polled"] else None,
            )
            for r in rows
        ]

    def remove_watch(self, watch_id: str) -> bool:
        with _db_conn(self._db_path) as conn:
            conn.execute("DELETE FROM watchlist WHERE watch_id = ?", (watch_id,))
        return True

    # ------------------------------------------------------------------
    # RSS feed polling
    # ------------------------------------------------------------------

    def poll_feed(self, max_new: int = 40) -> list[AlertEvent]:
        """
        Poll EDGAR RSS for all form types across all watchlist entries.
        Store new filings as alerts and return them.
        """
        return asyncio.run(self._async_poll_feed(max_new))

    async def _async_poll_feed(self, max_new: int) -> list[AlertEvent]:
        watches = self.list_watches()
        if not watches:
            return []

        # Gather unique form types across all watches
        all_form_types: set[str] = set()
        for w in watches:
            all_form_types.update(w.form_types)

        new_alerts: list[AlertEvent] = []

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for form_type in all_form_types:
                url = _EDGAR_RSS_URL.format(form_type=urllib.parse.quote(form_type))
                try:
                    resp = await client.get(url, headers={**_HEADERS, "Accept": "application/atom+xml"})
                    if resp.status_code != 200:
                        continue
                    entries = self._parse_rss(resp.text, form_type)
                    await asyncio.sleep(_RATE_DELAY)
                except Exception as exc:
                    logger.warning("edgar_v2: poll_feed RSS error", form=form_type, error=str(exc))
                    continue

                for entry in entries[:max_new]:
                    alerts = self._match_entry_to_watches(entry, watches)
                    new_alerts.extend(alerts)

        # Persist and deduplicate
        saved: list[AlertEvent] = []
        with _db_conn(self._db_path) as conn:
            for alert in new_alerts:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO alerts
                           (alert_id, watch_id, ticker, cik, form_type, filed_date,
                            entity_name, accession, url, headline, item_codes, severity, detected_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            alert.alert_id, alert.watch_id, alert.ticker,
                            alert.cik, alert.form_type, alert.filed_date.isoformat(),
                            alert.entity_name, alert.accession, alert.url,
                            alert.headline, json.dumps(alert.item_codes),
                            alert.severity, alert.detected_at.isoformat(),
                        ),
                    )
                    if conn.execute("SELECT changes()").fetchone()[0]:
                        saved.append(alert)
                except Exception as exc:
                    logger.debug("edgar_v2: alert persist error", error=str(exc))

        # Update last_polled for all watches
        now = datetime.utcnow().isoformat()
        with _db_conn(self._db_path) as conn:
            for w in watches:
                conn.execute(
                    "UPDATE watchlist SET last_polled = ? WHERE watch_id = ?",
                    (now, w.watch_id),
                )

        logger.info("edgar_v2: poll_feed", new_alerts=len(saved), watches=len(watches))
        return saved

    def _parse_rss(self, xml_text: str, form_type: str) -> list[dict]:
        """Parse EDGAR Atom RSS feed into a list of filing dicts."""
        entries: list[dict] = []
        try:
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                title_el   = entry.find("atom:title", ns)
                updated_el = entry.find("atom:updated", ns)
                link_el    = entry.find("atom:link", ns)
                summary_el = entry.find("atom:summary", ns)

                title   = title_el.text   if title_el   is not None else ""
                updated = updated_el.text if updated_el is not None else ""
                link    = link_el.get("href", "") if link_el is not None else ""
                summary = summary_el.text if summary_el is not None else ""

                # Parse company name and CIK from summary
                cik_match    = re.search(r"CIK\s*=\s*(\d+)", summary or title or "", re.IGNORECASE)
                entity_match = re.search(r"Company:\s*([^\n]+)", summary or "", re.IGNORECASE)
                acc_match    = re.search(r"accession-number[^\d]*([\d-]+)", link or "", re.IGNORECASE)

                cik         = cik_match.group(1)    if cik_match    else None
                entity_name = entity_match.group(1).strip() if entity_match else (title or "Unknown")
                accession   = acc_match.group(1)    if acc_match    else ""

                # Filed date
                filed_date: Optional[date] = None
                if updated:
                    try:
                        filed_date = datetime.fromisoformat(updated[:10]).date()
                    except ValueError:
                        pass
                filed_date = filed_date or date.today()

                # Build EDGAR URL
                if cik and accession:
                    acc_nd  = accession.replace("-", "")
                    doc_url = f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{acc_nd}/"
                else:
                    doc_url = link

                entries.append({
                    "entity_name": entity_name,
                    "cik":         cik,
                    "form_type":   form_type,
                    "filed_date":  filed_date,
                    "accession":   accession,
                    "url":         doc_url,
                    "summary":     summary or "",
                })
        except ET.ParseError as exc:
            logger.warning("edgar_v2: RSS parse error", error=str(exc))
        return entries

    def _match_entry_to_watches(
        self, entry: dict, watches: list[WatchEntry]
    ) -> list[AlertEvent]:
        """Check if an RSS entry matches any watchlist entry; create AlertEvents."""
        alerts: list[AlertEvent] = []
        entity  = (entry.get("entity_name") or "").lower()
        cik     = entry.get("cik") or ""
        f_type  = entry.get("form_type", "")

        for watch in watches:
            ticker_match  = watch.ticker.lower() in entity or (cik and cik == (watch.cik or ""))
            form_match    = not watch.form_types or f_type in watch.form_types
            if not (ticker_match and form_match):
                continue

            # Parse 8-K item codes from summary
            item_codes: list[str] = []
            severity  = "low"
            summary   = entry.get("summary", "")
            if f_type == "8-K":
                item_codes = self._extract_8k_items(summary)
                severity   = self._classify_8k_severity(item_codes)

            headline = (
                f"[{f_type}] {entry.get('entity_name', 'Unknown')} filed "
                f"on {entry.get('filed_date')} — items: {', '.join(item_codes) or 'n/a'}"
            )

            alert_id = hashlib.md5(
                f"{watch.watch_id}|{entry.get('accession')}|{f_type}".encode()
            ).hexdigest()[:16]

            filed_date = entry.get("filed_date")
            if not isinstance(filed_date, date):
                filed_date = date.today()

            alerts.append(
                AlertEvent(
                    alert_id=alert_id,
                    watch_id=watch.watch_id,
                    ticker=watch.ticker,
                    cik=cik or watch.cik,
                    form_type=f_type,
                    filed_date=filed_date,
                    entity_name=entry.get("entity_name", "Unknown"),
                    accession=entry.get("accession", ""),
                    url=entry.get("url", ""),
                    headline=headline,
                    item_codes=item_codes,
                    severity=severity,
                    detected_at=datetime.utcnow(),
                )
            )

        return alerts

    @staticmethod
    def _extract_8k_items(text: str) -> list[str]:
        """Extract 8-K item codes from filing text/summary."""
        return re.findall(r"\b(\d+\.\d+)\b", text)

    @staticmethod
    def _classify_8k_severity(item_codes: list[str]) -> str:
        high_items = {"1.03", "2.02", "4.02", "2.06", "4.01", "3.01", "5.01"}
        med_items  = {"1.01", "1.02", "2.01", "2.04", "2.05", "5.02"}
        codes_set  = set(item_codes)
        if codes_set & high_items:
            return "high"
        if codes_set & med_items:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Alert retrieval
    # ------------------------------------------------------------------

    def get_alerts(
        self,
        watch_id:  Optional[str] = None,
        ticker:    Optional[str] = None,
        since:     Optional[date] = None,
        limit:     int = 100,
    ) -> list[AlertEvent]:
        """Retrieve stored alerts, optionally filtered."""
        conditions: list[str] = []
        params: list[Any] = []

        if watch_id:
            conditions.append("watch_id = ?")
            params.append(watch_id)
        if ticker:
            conditions.append("UPPER(ticker) = UPPER(?)")
            params.append(ticker)
        if since:
            conditions.append("filed_date >= ?")
            params.append(since.isoformat())

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM alerts {where} ORDER BY detected_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()

        events: list[AlertEvent] = []
        for r in rows:
            try:
                events.append(
                    AlertEvent(
                        alert_id=r["alert_id"],
                        watch_id=r["watch_id"],
                        ticker=r["ticker"] or "",
                        cik=r["cik"],
                        form_type=r["form_type"] or "",
                        filed_date=date.fromisoformat(r["filed_date"]),
                        entity_name=r["entity_name"] or "Unknown",
                        accession=r["accession"] or "",
                        url=r["url"] or "",
                        headline=r["headline"] or "",
                        item_codes=json.loads(r["item_codes"] or "[]"),
                        severity=r["severity"] or "low",
                        detected_at=datetime.fromisoformat(r["detected_at"]),
                    )
                )
            except Exception as exc:
                logger.debug("edgar_v2: get_alerts row error", error=str(exc))

        return events


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
    from fastapi.responses import JSONResponse

    edgar_v2_router = APIRouter(
        prefix="/api/edgar/v2",
        tags=["EDGAR Full-Text Search V2"],
    )

    _searcher   = AdvancedEFTSSearcher()
    _extractor  = FilingContentExtractor()
    _semantic   = SemanticSearchEngine()
    _alert_sys  = FilingAlertSystem()

    # ------------------------------------------------------------------
    # POST /edgar/v2/search
    # ------------------------------------------------------------------

    @edgar_v2_router.post("/search")
    def api_v2_search(bq: BooleanSearchQuery):
        """
        Advanced Boolean search: specify must/should/must_not keyword lists
        plus optional form_types and date range.
        """
        try:
            resp = _searcher.boolean_search(bq)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return resp.model_dump()

    # ------------------------------------------------------------------
    # POST /edgar/v2/proximity-search
    # ------------------------------------------------------------------

    @edgar_v2_router.post("/proximity-search")
    def api_v2_proximity(pq: ProximityQuery):
        """
        Proximity search: find filings where term1 and term2 appear
        within within_n words of each other.
        """
        try:
            resp = _searcher.proximity_search(pq)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return resp.model_dump()

    # ------------------------------------------------------------------
    # GET /edgar/v2/semantic-search
    # ------------------------------------------------------------------

    @edgar_v2_router.get("/semantic-search")
    def api_v2_semantic_search(
        q:    str = Query(..., description="Natural-language query"),
        n:    int = Query(10, ge=1, le=50),
        mode: str = Query("hybrid", enum=["hybrid", "semantic", "keyword"]),
    ):
        """
        Semantic search over indexed filing segments.
        Hybrid mode (default): 70% BM25 keyword + 30% semantic embedding.
        """
        try:
            results = _semantic.hybrid_search(q, n=n)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return [r.model_dump() for r in results]

    # ------------------------------------------------------------------
    # GET /edgar/v2/extract/{cik}/{accession}
    # ------------------------------------------------------------------

    @edgar_v2_router.get("/extract/{cik}/{accession}")
    def api_v2_extract(
        cik:       str,
        accession: str,
        section:   Optional[str] = Query(None, description="Specific section key (e.g. 'mda', 'risk_factors')"),
        include_financials: bool = Query(True),
        include_sentiment:  bool = Query(True),
        include_tables:     bool = Query(False),
    ):
        """
        Extract content from a specific EDGAR filing.
        Returns sections, financials, and sentiment analysis.
        """
        try:
            content = _extractor.extract_filing(cik, accession)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Filing extraction failed: {exc}")

        result: dict[str, Any] = {
            "cik":        cik,
            "accession":  accession,
            "entity_name": content.entity_name,
            "form_type":  content.form_type,
            "filed_date": str(content.filed_date) if content.filed_date else None,
        }

        if section:
            result["section"] = content.sections.get(section, "")
        else:
            result["sections"] = {k: v[:3000] for k, v in content.sections.items()}

        if include_financials:
            result["financials"] = content.financials

        if include_sentiment:
            result["sentiment"] = content.sentiment

        if include_tables:
            try:
                result["tables"] = _extractor.extract_tables(cik, accession)
            except Exception:
                result["tables"] = []

        return result

    # ------------------------------------------------------------------
    # POST /edgar/v2/alerts/register
    # ------------------------------------------------------------------

    class WatchRequest(BaseModel):
        ticker:     str
        cik:        Optional[str] = None
        form_types: list[str]    = Field(default=["8-K", "10-Q", "10-K"])

    @edgar_v2_router.post("/alerts/register")
    def api_v2_register_alert(req: WatchRequest):
        """Register a company for real-time filing alerts."""
        watch_id = _alert_sys.register_watch(
            ticker=req.ticker, cik=req.cik, form_types=req.form_types
        )
        return {"watch_id": watch_id, "ticker": req.ticker, "form_types": req.form_types}

    # ------------------------------------------------------------------
    # GET /edgar/v2/alerts
    # ------------------------------------------------------------------

    @edgar_v2_router.get("/alerts")
    def api_v2_get_alerts(
        watch_id: Optional[str] = Query(None),
        ticker:   Optional[str] = Query(None),
        limit:    int           = Query(50, ge=1, le=500),
    ):
        """Retrieve stored filing alerts."""
        alerts = _alert_sys.get_alerts(watch_id=watch_id, ticker=ticker, limit=limit)
        return [a.model_dump() for a in alerts]

    # ------------------------------------------------------------------
    # GET /edgar/v2/feed
    # ------------------------------------------------------------------

    @edgar_v2_router.get("/feed")
    def api_v2_feed(
        max_new:    int  = Query(40, ge=1, le=200),
        background: bool = Query(False, description="Run poll in background"),
    ):
        """
        Poll EDGAR RSS feeds for all watchlist entries.
        Returns newly detected filings.
        """
        try:
            alerts = _alert_sys.poll_feed(max_new=max_new)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Feed poll failed: {exc}")
        return {
            "polled_at":    datetime.utcnow().isoformat(),
            "new_alerts":   len(alerts),
            "alerts":       [a.model_dump() for a in alerts],
        }

    # ------------------------------------------------------------------
    # GET /edgar/v2/watchlist
    # ------------------------------------------------------------------

    @edgar_v2_router.get("/watchlist")
    def api_v2_watchlist():
        """List all registered watchlist entries."""
        watches = _alert_sys.list_watches()
        return [w.model_dump() for w in watches]

    # ------------------------------------------------------------------
    # DELETE /edgar/v2/watchlist/{watch_id}
    # ------------------------------------------------------------------

    @edgar_v2_router.delete("/watchlist/{watch_id}")
    def api_v2_remove_watch(watch_id: str):
        """Remove a watchlist entry."""
        _alert_sys.remove_watch(watch_id)
        return {"status": "removed", "watch_id": watch_id}

    # ------------------------------------------------------------------
    # POST /edgar/v2/index
    # ------------------------------------------------------------------

    class IndexRequest(BaseModel):
        cik:         str
        accession:   str
        entity_name: Optional[str] = None
        form_type:   Optional[str] = None
        filed_date:  Optional[str] = None

    @edgar_v2_router.post("/index")
    def api_v2_index_filing(req: IndexRequest):
        """
        Extract and index a filing into the semantic search store.
        Segments the filing, computes embeddings (if available), and stores in SQLite.
        """
        try:
            n = _semantic.index_filing(
                cik=req.cik,
                accession=req.accession,
                entity_name=req.entity_name,
                form_type=req.form_type,
                filed_date=req.filed_date,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"segments_indexed": n, "cik": req.cik, "accession": req.accession}

    # ------------------------------------------------------------------
    # GET /edgar/v2/index-stats
    # ------------------------------------------------------------------

    @edgar_v2_router.get("/index-stats")
    def api_v2_index_stats():
        """Return statistics about the semantic search index."""
        return _semantic.index_stats()

    # ------------------------------------------------------------------
    # GET /edgar/v2/regex-extract
    # ------------------------------------------------------------------

    @edgar_v2_router.get("/regex-extract/{cik}/{accession}")
    def api_v2_regex_extract(
        cik:       str,
        accession: str,
        section:   str = Query("mda", description="Section to extract from"),
    ):
        """
        Apply financial regex patterns to a specific filing section.
        Returns extracted revenue, EPS, guidance, etc.
        """
        text = _extractor.extract_section(cik, accession, section)
        if not text:
            return {"cik": cik, "accession": accession, "section": section, "extracted": {}}
        extracted = _searcher.regex_extract(text)
        return {"cik": cik, "accession": accession, "section": section, "extracted": extracted}

    # ------------------------------------------------------------------
    # GET /edgar/v2/material-events
    # ------------------------------------------------------------------

    @edgar_v2_router.get("/material-events")
    def api_v2_material_events(
        days_back:  int = Query(7, ge=1, le=30),
        min_severity: str = Query("low", enum=["low", "medium", "high"]),
    ):
        """
        Scan EDGAR for material corporate events using pre-built search templates.
        Returns a prioritised, deduplicated feed.
        """
        from sentinel.sfe.edgar_full_text_search import EDGARFullTextSearch
        engine = EDGARFullTextSearch()
        try:
            feed = asyncio.run(engine.monitor_material_events(days_back=days_back))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        sev_rank = {"low": 0, "medium": 1, "high": 2}
        min_rank = sev_rank.get(min_severity, 0)
        filtered = [
            e.model_dump() for e in feed.events
            if sev_rank.get(e.severity, 0) >= min_rank
        ]
        return {
            "as_of":        feed.as_of.isoformat(),
            "total_events": len(feed.events),
            "filtered":     len(filtered),
            "event_counts": feed.event_counts,
            "top_entities": feed.top_entities,
            "events":       filtered,
        }

except ImportError:
    edgar_v2_router = None   # type: ignore[assignment]
    logger.warning("FastAPI not available — edgar_v2_router not registered")


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def search_boolean(
    must:      list[str],
    should:    list[str] = None,
    must_not:  list[str] = None,
    form_types: list[str] = None,
    days_back: int = 30,
) -> SearchResponse:
    """Quick Boolean EDGAR search."""
    bq = BooleanSearchQuery(
        must=must,
        should=should or [],
        must_not=must_not or [],
        form_types=form_types or [],
        start_date=date.today() - timedelta(days=days_back),
        end_date=date.today(),
    )
    return AdvancedEFTSSearcher().boolean_search(bq)


def extract_filing_content(cik: str, accession: str) -> FilingContent:
    """Extract complete filing content including sections, financials, sentiment."""
    return FilingContentExtractor().extract_filing(cik, accession)


def semantic_search_filings(query: str, n: int = 10) -> list[dict]:
    """Search indexed filing segments semantically."""
    engine = SemanticSearchEngine()
    results = engine.semantic_search(query, n=n)
    return [r.model_dump() for r in results]


def register_filing_watch(ticker: str, cik: Optional[str] = None) -> str:
    """Register a ticker for real-time EDGAR filing alerts."""
    return FilingAlertSystem().register_watch(ticker=ticker, cik=cik)


def poll_filing_alerts() -> list[dict]:
    """Poll EDGAR RSS for new filings matching the watchlist."""
    alerts = FilingAlertSystem().poll_feed()
    return [a.model_dump() for a in alerts]
