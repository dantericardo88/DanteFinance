"""
Earnings call / news corpus RAG engine V3 — Dimension #057 (score 6 → 9).

Production RAG system over earnings call transcripts, SEC filings, and
financial news.  All data sources are free.  ChromaDB, sentence-transformers,
and the Anthropic SDK are optional — the module works without them using a
pure-numpy TF-IDF retrieval fallback.

dim_057 — Earnings call / news corpus (RAG)

Key capabilities:
  - EarningsCallCollector: EDGAR 8-K + Motley Fool + Seeking Alpha transcripts
  - SECFilingCorpusBuilder: 10-K sections (1A/7/7A), 8-K corpus, DEF 14A proxy
  - NewsCorpusBuilder: GDELT (free, no key) + Yahoo Finance RSS + Reuters RSS
  - LocalVectorStore: pure-numpy TF-IDF cosine retrieval (no external vector DB)
  - ChromaDBAdapter: optional persistent chroma with sentence-transformers
  - FinancialQASystem: RAG → Claude (or extractive fallback)
  - EarningsAnalytics: tone, KPI extraction, guidance delta, call quality score
  - EarningsRAGEngine: full orchestrator

FastAPI router: earnings_rag_v3_router
  POST /rag/v3/index
  POST /rag/v3/query
  GET  /rag/v3/brief/{ticker}/{quarter}
  POST /rag/v3/screen

Usage::

    from sentinel.sai.earnings_rag_v3 import EarningsRAGEngine
    engine = EarningsRAGEngine()
    engine.index_ticker("AAPL", years=2)
    ans = engine.query("AAPL", "What guidance did management give for 2025?")
    print(ans.answer)
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
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

try:
    from sentinel.core.config import get_settings
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidf
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSection:
    title: str
    speaker: str
    role: str
    text: str
    start_idx: int = 0


@dataclass
class EarningsTranscript:
    ticker: str
    quarter: str
    year: int
    date: str
    source: str
    title: str
    full_text: str
    prepared_remarks: List[TranscriptSection] = field(default_factory=list)
    qa_sections: List[TranscriptSection] = field(default_factory=list)
    url: str = ""
    filing_date: str = ""
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.full_text.split())


@dataclass
class SECDocument:
    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    title: str
    text: str
    url: str
    section: str = ""
    cik: str = ""


@dataclass
class NewsArticle:
    title: str
    url: str
    source: str
    published: str
    snippet: str
    full_text: str = ""
    ticker: str = ""
    sentiment: float = 0.0


@dataclass
class Document:
    doc_id: str
    ticker: str
    source: str
    doc_type: str
    date: str
    title: str
    text: str
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    chunk_total: int = 1


@dataclass
class RetrievedDocument:
    document: Document
    score: float
    rank: int


@dataclass
class RAGAnswer:
    question: str
    answer: str
    ticker: str
    sources: List[RetrievedDocument]
    confidence: float
    method: str
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ToneAnalysis:
    ticker: str
    quarter: str
    positive_score: float
    negative_score: float
    uncertainty_score: float
    forward_looking_count: int
    sentiment_compound: float
    dominant_tone: str
    top_positive_phrases: List[str] = field(default_factory=list)
    top_negative_phrases: List[str] = field(default_factory=list)


@dataclass
class KPI:
    name: str
    value: float
    unit: str
    context: str
    quarter: str = ""
    yoy_change: Optional[float] = None


@dataclass
class GuidanceDelta:
    ticker: str
    metric: str
    prior_guidance: str
    current_guidance: str
    direction: str
    magnitude_pct: Optional[float]
    commentary: str


@dataclass
class EarningsBrief:
    ticker: str
    quarter: str
    summary: str
    key_metrics: List[KPI]
    tone: ToneAnalysis
    guidance_highlights: List[str]
    risk_mentions: List[str]
    analyst_concerns: List[str]


@dataclass
class GuidanceExtraction:
    ticker: str
    revenue_guidance: str
    eps_guidance: str
    margin_guidance: str
    capex_guidance: str
    other: List[str]
    raw_quotes: List[str]


# ---------------------------------------------------------------------------
# Simple TF-IDF (pure numpy, no sklearn)
# ---------------------------------------------------------------------------

class SimpleTFIDF:
    """Pure-numpy TF-IDF implementation — fallback when sklearn absent."""

    def __init__(self, max_features: int = 8000, ngram_max: int = 2):
        self.max_features = max_features
        self.ngram_max = ngram_max
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self._stop_words = frozenset(
            "a an the is are was were be been being have has had do does did "
            "will would could should may might shall can to of in on at for "
            "with by from as this that these those it its he she we you they "
            "i me my our us your their him her them who what which how when "
            "where why all any each few more most other some such no nor not "
            "or but and if while".split()
        )

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        tokens = [t for t in tokens if t not in self._stop_words and len(t) > 1]
        ngrams: List[str] = list(tokens)
        if self.ngram_max >= 2:
            ngrams += [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return ngrams

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        # Count term frequencies per document
        doc_tfs: List[Counter] = []
        all_terms: Counter = Counter()
        for text in texts:
            toks = self._tokenize(text)
            tf = Counter(toks)
            doc_tfs.append(tf)
            all_terms.update(tf.keys())

        # Select top-N by document frequency
        top_terms = [t for t, _ in all_terms.most_common(self.max_features)]
        self.vocab = {term: idx for idx, term in enumerate(top_terms)}

        N = len(texts)
        V = len(self.vocab)
        matrix = np.zeros((N, V), dtype=np.float32)

        for i, (tf, text) in enumerate(zip(doc_tfs, texts)):
            n_words = max(1, sum(tf.values()))
            for term, count in tf.items():
                if term in self.vocab:
                    matrix[i, self.vocab[term]] = count / n_words

        # IDF
        df = (matrix > 0).sum(axis=0) + 1
        self.idf = np.log((N + 1) / df) + 1.0

        matrix = matrix * self.idf
        # L2 normalise
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms
        return self.matrix

    def transform(self, query: str) -> np.ndarray:
        toks = self._tokenize(query)
        tf = Counter(toks)
        n_words = max(1, sum(tf.values()))
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for term, count in tf.items():
            if term in self.vocab:
                vec[self.vocab[term]] = count / n_words
        vec = vec * self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ---------------------------------------------------------------------------
# Local Vector Store
# ---------------------------------------------------------------------------

class LocalVectorStore:
    """TF-IDF cosine-similarity document store; no external dependencies."""

    def __init__(self):
        self.documents: List[Document] = []
        self._texts: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._tfidf = None
        self._dirty = True

    def add_documents(self, docs: List[Document]) -> None:
        self.documents.extend(docs)
        self._texts.extend(d.title + " " + d.text for d in docs)
        self._dirty = True

    def _rebuild_index(self) -> None:
        if not self._texts:
            return
        if _SKLEARN_AVAILABLE:
            self._tfidf = SklearnTfidf(
                max_features=10_000, ngram_range=(1, 2), sublinear_tf=True
            )
            self._matrix = self._tfidf.fit_transform(self._texts).toarray().astype(np.float32)
        else:
            self._tfidf = SimpleTFIDF(max_features=10_000)
            self._matrix = self._tfidf.fit_transform(self._texts)
        self._dirty = False

    def search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        if self._dirty:
            self._rebuild_index()
        if self._matrix is None or len(self._matrix) == 0:
            return []

        if _SKLEARN_AVAILABLE:
            qvec = self._tfidf.transform([query]).toarray()[0].astype(np.float32)
        else:
            qvec = self._tfidf.transform(query)

        scores = self._matrix @ qvec
        top_k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievedDocument(
                document=self.documents[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))
        return results

    def __len__(self) -> int:
        return len(self.documents)


# ---------------------------------------------------------------------------
# ChromaDB Adapter (optional)
# ---------------------------------------------------------------------------

class ChromaDBAdapter:
    """Persistent vector store using ChromaDB + optional sentence-transformers."""

    def __init__(self, persist_dir: str = "sentinel/data/chroma_db"):
        self._available = _CHROMA_AVAILABLE
        self._client = None
        self._collections: Dict[str, Any] = {}
        self._embedder = None
        self._fallback = LocalVectorStore()
        self.persist_dir = persist_dir

        if self._available:
            try:
                self._client = chromadb.PersistentClient(path=persist_dir)
                if _ST_AVAILABLE:
                    self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("ChromaDBAdapter: using sentence-transformers embeddings")
                else:
                    logger.info("ChromaDBAdapter: using default chroma embeddings")
            except Exception as exc:
                logger.warning("ChromaDB init failed (%s); falling back to LocalVectorStore", exc)
                self._available = False

    def _get_collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name)
        return self._collections[name]

    def add_documents(self, docs: List[Document], collection: str = "default") -> None:
        if not self._available:
            self._fallback.add_documents(docs)
            return
        col = self._get_collection(collection)
        texts = [d.title + " " + d.text[:2000] for d in docs]
        ids = [d.doc_id for d in docs]
        metadatas = [{"ticker": d.ticker, "source": d.source, "date": d.date,
                      "doc_type": d.doc_type, "url": d.url} for d in docs]
        if self._embedder:
            embeddings = self._embedder.encode(texts).tolist()
            col.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        else:
            col.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def search(self, query: str, k: int = 5, collection: str = "default") -> List[RetrievedDocument]:
        if not self._available:
            return self._fallback.search(query, k)
        col = self._get_collection(collection)
        if self._embedder:
            qemb = self._embedder.encode([query]).tolist()
            results = col.query(query_embeddings=qemb, n_results=k)
        else:
            results = col.query(query_texts=[query], n_results=k)
        retrieved = []
        docs_list = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for rank, (text, meta, dist) in enumerate(zip(docs_list, metas, dists)):
            score = max(0.0, 1.0 - dist)
            doc = Document(
                doc_id=f"chroma_{rank}",
                ticker=meta.get("ticker", ""),
                source=meta.get("source", "chroma"),
                doc_type=meta.get("doc_type", ""),
                date=meta.get("date", ""),
                title="",
                text=text,
                url=meta.get("url", ""),
            )
            retrieved.append(RetrievedDocument(document=doc, score=score, rank=rank + 1))
        return retrieved


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
})

def _get(url: str, timeout: int = 20, params: dict = None, headers: dict = None,
         retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(2 ** attempt)
        except Exception as exc:
            logger.debug("GET %s attempt %d failed: %s", url, attempt + 1, exc)
            time.sleep(1)
    return None


def _clean_html(raw: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_doc_id(ticker: str, source: str, date: str, idx: int = 0) -> str:
    raw = f"{ticker}:{source}:{date}:{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Earnings Call Collector
# ---------------------------------------------------------------------------

_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_CIK_LOOKUP = "https://www.sec.gov/cgi-bin/browse-edgar"

_PREPARED_PATTERNS = re.compile(
    r"(prepared remarks?|opening remarks?|management discussion|presentation)",
    re.I,
)
_QA_PATTERNS = re.compile(
    r"(question[- ]and[- ]answer|q&a session|analyst question|operator:)",
    re.I,
)
_SPEAKER_RE = re.compile(
    r"^([A-Z][a-zA-Z .'-]{2,40})\s*[:\-]\s*", re.M
)
_DOLLAR_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)\b", re.I
)
_PCT_RE = re.compile(r"([\d.]+)\s*%")
_GUIDANCE_KEYWORDS = frozenset(
    "guidance outlook expect anticipate project forecast target range full-year "
    "fiscal next quarter revenue earnings EPS margin".split()
)


def _get_cik(ticker: str) -> Optional[str]:
    """Return CIK for ticker via EDGAR company search."""
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    r = _get(url, params={"company": ticker, "CIK": ticker, "action": "getcompany",
                          "type": "10-K", "dateb": "", "owner": "include",
                          "count": "10", "search_text": "", "output": "atom"})
    if r is None:
        return None
    m = re.search(r"CIK=(\d+)", r.text)
    if m:
        return m.group(1).lstrip("0")

    # Try JSON ticker map
    r2 = _get("https://www.sec.gov/files/company_tickers.json")
    if r2:
        try:
            data = r2.json()
            for _, entry in data.items():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"])
        except Exception:
            pass
    return None


def _fetch_submissions(cik: str) -> Optional[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    r = _get(url, headers={"User-Agent": "DanteFinance research@dante.finance"})
    if r:
        try:
            return r.json()
        except Exception:
            pass
    return None


def _parse_transcript_sections(text: str) -> Tuple[List[TranscriptSection], List[TranscriptSection]]:
    """Split transcript into prepared remarks and Q&A sections."""
    prepared: List[TranscriptSection] = []
    qa: List[TranscriptSection] = []

    # Find split point
    qa_start = len(text)
    for m in _QA_PATTERNS.finditer(text):
        qa_start = m.start()
        break

    prepared_text = text[:qa_start]
    qa_text = text[qa_start:]

    def parse_blocks(raw: str, sections: List[TranscriptSection], mode: str) -> None:
        # Split by speaker labels
        parts = re.split(r"\n([A-Z][a-zA-Z .'-]{2,40})\s*[:\-]\s*", raw)
        i = 0
        while i < len(parts):
            chunk = parts[i].strip()
            speaker = ""
            role = ""
            if i + 1 < len(parts) and re.match(r"^[A-Z][a-zA-Z .'-]{2,40}$", parts[i + 1] if i + 1 < len(parts) else ""):
                speaker = parts[i + 1]
                chunk = parts[i + 2] if i + 2 < len(parts) else ""
                i += 3
            else:
                i += 1

            # Infer role
            lower_chunk = chunk.lower()
            if any(w in speaker.lower() for w in ["operator", "moderator"]):
                role = "Operator"
            elif any(w in lower_chunk[:100] for w in ["chief executive", "ceo", "president and ceo"]):
                role = "CEO"
            elif any(w in lower_chunk[:100] for w in ["chief financial", "cfo", "finance officer"]):
                role = "CFO"
            elif any(w in lower_chunk[:100] for w in ["analyst", "research", "securities", "capital"]):
                role = "Analyst"

            if chunk.strip():
                sections.append(TranscriptSection(
                    title=mode,
                    speaker=speaker,
                    role=role,
                    text=chunk.strip(),
                    start_idx=raw.find(chunk),
                ))

    parse_blocks(prepared_text, prepared, "Prepared Remarks")
    parse_blocks(qa_text, qa, "Q&A")
    return prepared, qa


class EarningsCallCollector:
    """Collect earnings call transcripts from EDGAR 8-K and free web sources."""

    MOTLEY_FOOL_BASE = "https://www.fool.com/earnings-call-transcripts/"
    SEEKING_ALPHA_API = "https://seekingalpha.com/api/v3/articles"
    EDGAR_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self, cache_dir: str = "sentinel/data/transcripts_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, quarter: str) -> Path:
        safe = re.sub(r"[^\w]", "_", f"{ticker}_{quarter}")
        return self.cache_dir / f"{safe}.json"

    def _load_cache(self, ticker: str, quarter: str) -> Optional[EarningsTranscript]:
        p = self._cache_path(ticker, quarter)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return EarningsTranscript(**{
                    k: v for k, v in data.items()
                    if k in EarningsTranscript.__dataclass_fields__
                })
            except Exception:
                pass
        return None

    def _save_cache(self, transcript: EarningsTranscript) -> None:
        p = self._cache_path(transcript.ticker, transcript.quarter)
        try:
            import dataclasses
            p.write_text(
                json.dumps(dataclasses.asdict(transcript), default=str, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Cache save failed: %s", exc)

    def fetch_transcript(
        self, ticker: str, quarter: str = None
    ) -> Optional[EarningsTranscript]:
        """Fetch single earnings transcript. quarter: 'Q4 2024'."""
        if quarter is None:
            quarter = "Q4 2024"

        cached = self._load_cache(ticker, quarter)
        if cached:
            return cached

        transcript = None
        # Try sources in order of reliability
        for fetcher in [
            self._fetch_edgar_8k,
            self._fetch_motley_fool,
            self._fetch_seeking_alpha,
        ]:
            try:
                transcript = fetcher(ticker, quarter)
                if transcript and len(transcript.full_text) > 500:
                    break
            except Exception as exc:
                logger.debug("Fetcher %s failed for %s %s: %s",
                             fetcher.__name__, ticker, quarter, exc)

        if transcript:
            self._save_cache(transcript)
        return transcript

    def _fetch_edgar_8k(self, ticker: str, quarter: str) -> Optional[EarningsTranscript]:
        """Search EDGAR EFTS for 8-K filings containing earnings call text."""
        # Parse quarter
        m = re.match(r"Q(\d)\s+(\d{4})", quarter)
        q_num = int(m.group(1)) if m else 4
        year = int(m.group(2)) if m else 2024

        # Rough date windows per quarter
        q_ends = {1: ("01-01", "05-30"), 2: ("04-01", "08-30"),
                  3: ("07-01", "11-30"), 4: ("10-01", "03-30")}
        start_s, end_s = q_ends.get(q_num, ("01-01", "12-31"))
        start_dt = f"{year}-{start_s}" if q_num != 4 else f"{year}-{start_s}"
        end_dt = f"{year}-{end_s}" if q_num != 4 else f"{year + 1}-{end_s}"

        params = {
            "q": f'"{ticker}" "earnings call"',
            "forms": "8-K",
            "dateRange": "custom",
            "startdt": start_dt,
            "enddt": end_dt,
            "hits.hits.total.value": 1,
            "hits.hits._source.period_of_report": 1,
        }
        r = _get(self.EDGAR_EFTS_BASE, params=params,
                 headers={"User-Agent": "DanteFinance research@dante.finance"})
        if not r:
            return None

        try:
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
        except Exception:
            return None

        for hit in hits[:5]:
            src = hit.get("_source", {})
            accession = src.get("accession_no", "").replace("-", "")
            cik = src.get("entity_id", "")
            if not accession or not cik:
                continue

            # Fetch filing index
            idx_url = (f"https://www.sec.gov/Archives/edgar/full-index/"
                       f"cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
                       f"&type=8-K&dateb=&owner=include&count=10&output=atom")
            filing_url = (f"https://www.sec.gov/Archives/edgar/data/"
                          f"{cik}/{accession[:10]}-{accession[10:12]}-"
                          f"{accession[12:]}-index.htm")

            r2 = _get(filing_url,
                      headers={"User-Agent": "DanteFinance research@dante.finance"})
            if not r2:
                continue

            # Find exhibit text file
            txt_links = re.findall(r'href="(/Archives/edgar/[^"]+\.txt)"', r2.text, re.I)
            htm_links = re.findall(r'href="(/Archives/edgar/[^"]+\.htm)"', r2.text, re.I)
            doc_links = txt_links + htm_links

            for link in doc_links[:3]:
                doc_url = "https://www.sec.gov" + link
                r3 = _get(doc_url,
                          headers={"User-Agent": "DanteFinance research@dante.finance"})
                if not r3:
                    continue
                text = _clean_html(r3.text) if "<html" in r3.text.lower() else r3.text
                if "earnings call" in text.lower() and len(text) > 2000:
                    prepared, qa = _parse_transcript_sections(text)
                    return EarningsTranscript(
                        ticker=ticker,
                        quarter=quarter,
                        year=year,
                        date=src.get("file_date", ""),
                        source="EDGAR-8K",
                        title=f"{ticker} {quarter} Earnings Call",
                        full_text=text[:50_000],
                        prepared_remarks=prepared,
                        qa_sections=qa,
                        url=doc_url,
                        filing_date=src.get("file_date", ""),
                    )
        return None

    def _fetch_motley_fool(self, ticker: str, quarter: str) -> Optional[EarningsTranscript]:
        """Scrape Motley Fool earnings transcript index."""
        search_url = f"https://www.fool.com/search/#q={ticker}+earnings+call+transcript&s=site"
        # Try direct pattern URL
        q_m = re.match(r"Q(\d)\s+(\d{4})", quarter)
        if not q_m:
            return None
        q_num, year = int(q_m.group(1)), int(q_m.group(2))

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Referer": "https://www.fool.com/",
        }

        # Try index page
        r = _get(self.MOTLEY_FOOL_BASE, headers=headers, timeout=25)
        if not r:
            return None

        # Find links mentioning our ticker
        links = re.findall(
            r'href="(/earnings-call-transcripts/[^"]+)"', r.text
        )
        ticker_links = [
            l for l in links
            if ticker.lower() in l.lower()
        ]

        for link in ticker_links[:5]:
            full_url = "https://www.fool.com" + link
            r2 = _get(full_url, headers=headers, timeout=25)
            if not r2:
                continue
            text = _clean_html(r2.text)
            if len(text) < 2000:
                continue
            # Check quarter match
            if str(year) in text and f"Q{q_num}" in text:
                prepared, qa = _parse_transcript_sections(text)
                date_m = re.search(r"(\w+ \d+, \d{4})", text)
                date_str = date_m.group(1) if date_m else ""
                return EarningsTranscript(
                    ticker=ticker,
                    quarter=quarter,
                    year=year,
                    date=date_str,
                    source="MotleyFool",
                    title=f"{ticker} {quarter} Earnings Call Transcript",
                    full_text=text[:50_000],
                    prepared_remarks=prepared,
                    qa_sections=qa,
                    url=full_url,
                )
        return None

    def _fetch_seeking_alpha(self, ticker: str, quarter: str) -> Optional[EarningsTranscript]:
        """Try Seeking Alpha transcript API (public endpoint, may need headers)."""
        q_m = re.match(r"Q(\d)\s+(\d{4})", quarter)
        if not q_m:
            return None
        year = int(q_m.group(2))

        params = {
            "filter[category]": "earnings",
            "filter[since]": int((datetime(year, 1, 1)).timestamp()),
            "filter[until]": int((datetime(year, 12, 31)).timestamp()),
            "filter[tag]": ticker,
            "include": "author,primaryTickers",
            "page[size]": 5,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
            "Accept": "application/json",
            "Referer": "https://seekingalpha.com/",
        }
        r = _get(self.SEEKING_ALPHA_API, params=params, headers=headers, timeout=20)
        if not r:
            return None
        try:
            data = r.json()
            articles = data.get("data", [])
        except Exception:
            return None

        for article in articles[:3]:
            attr = article.get("attributes", {})
            title = attr.get("title", "")
            if "transcript" not in title.lower() and "earnings call" not in title.lower():
                continue
            slug = attr.get("slug", "")
            if not slug:
                continue
            url = f"https://seekingalpha.com/article/{slug}"
            r2 = _get(url, headers=headers, timeout=25)
            if not r2:
                continue
            text = _clean_html(r2.text)
            if len(text) < 1000:
                continue
            prepared, qa = _parse_transcript_sections(text)
            return EarningsTranscript(
                ticker=ticker,
                quarter=quarter,
                year=year,
                date=attr.get("publishOn", "")[:10],
                source="SeekingAlpha",
                title=title,
                full_text=text[:50_000],
                prepared_remarks=prepared,
                qa_sections=qa,
                url=url,
            )
        return None

    def fetch_recent_transcripts(
        self, ticker: str, n: int = 8
    ) -> List[EarningsTranscript]:
        """Fetch last n quarters of transcripts."""
        results: List[EarningsTranscript] = []
        current = datetime.utcnow()
        q = ((current.month - 1) // 3) + 1
        year = current.year

        for _ in range(n):
            quarter = f"Q{q} {year}"
            t = self.fetch_transcript(ticker, quarter)
            if t:
                results.append(t)
            q -= 1
            if q == 0:
                q = 4
                year -= 1
            time.sleep(0.5)
        return results

    def bulk_fetch(
        self, tickers: List[str], quarters: int = 4
    ) -> Dict[str, List[EarningsTranscript]]:
        result: Dict[str, List[EarningsTranscript]] = {}
        for ticker in tickers:
            logger.info("Bulk fetch: %s (%d quarters)", ticker, quarters)
            result[ticker] = self.fetch_recent_transcripts(ticker, n=quarters)
            time.sleep(1)
        return result


# ---------------------------------------------------------------------------
# SEC Filing Corpus Builder
# ---------------------------------------------------------------------------

_ITEM_PATTERNS = {
    "1A": re.compile(r"item\s+1a[\.\s]+(risk factors?)", re.I),
    "7": re.compile(r"item\s+7[\.\s]+(management.{0,30}discussion)", re.I),
    "7A": re.compile(r"item\s+7a[\.\s]+(quantitative.{0,40}market risk)", re.I),
    "1": re.compile(r"item\s+1[\.\s]+(business)", re.I),
    "8": re.compile(r"item\s+8[\.\s]+(financial statements?)", re.I),
}


class SECFilingCorpusBuilder:
    """Download and parse SEC filings for a ticker into searchable documents."""

    def __init__(self):
        self._headers = {"User-Agent": "DanteFinance research@dante.finance"}

    def _get_cik(self, ticker: str) -> Optional[str]:
        return _get_cik(ticker)

    def fetch_10k_sections(
        self, ticker: str, sections: List[str] = None
    ) -> Dict[str, str]:
        """Fetch and parse specified 10-K sections."""
        if sections is None:
            sections = ["1A", "7", "7A"]

        cik = self._get_cik(ticker)
        if not cik:
            logger.warning("Could not resolve CIK for %s", ticker)
            return {}

        subs = _fetch_submissions(cik)
        if not subs:
            return {}

        # Find latest 10-K
        filings = subs.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])

        acc_no = None
        for form, acc, date in zip(forms, accessions, dates):
            if form in ("10-K", "10-K/A"):
                acc_no = acc
                break
        if not acc_no:
            return {}

        acc_clean = acc_no.replace("-", "")
        idx_url = (f"https://www.sec.gov/Archives/edgar/data/"
                   f"{cik}/{acc_clean}/{acc_no}-index.htm")
        r = _get(idx_url, headers=self._headers)
        if not r:
            return {}

        # Find primary document
        doc_links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"', r.text, re.I)
        primary = None
        for link in doc_links[:5]:
            if "10k" in link.lower() or "annual" in link.lower() or cik in link:
                primary = link
                break
        if not primary and doc_links:
            primary = doc_links[0]
        if not primary:
            return {}

        doc_url = "https://www.sec.gov" + primary
        r2 = _get(doc_url, headers=self._headers, timeout=60)
        if not r2:
            return {}

        full_text = _clean_html(r2.text)
        return self._extract_sections(full_text, sections)

    def _extract_sections(self, text: str, sections: List[str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        # Find positions of each item
        positions: List[Tuple[str, int, int]] = []
        for sec_id, pattern in _ITEM_PATTERNS.items():
            if sec_id not in sections and "all" not in sections:
                continue
            for m in pattern.finditer(text):
                positions.append((sec_id, m.start(), m.end()))
                break

        positions.sort(key=lambda x: x[1])

        for i, (sec_id, start, end) in enumerate(positions):
            # Extract until next section or 20k words
            next_start = positions[i + 1][1] if i + 1 < len(positions) else start + 40_000
            snippet = text[end:next_start][:40_000].strip()
            result[sec_id] = snippet

        return result

    def fetch_8k_corpus(
        self, ticker: str, n_filings: int = 20
    ) -> List[SECDocument]:
        cik = self._get_cik(ticker)
        if not cik:
            return []

        subs = _fetch_submissions(cik)
        if not subs:
            return []

        filings = subs.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])
        primary_docs = filings.get("primaryDocument", [])

        docs: List[SECDocument] = []
        count = 0
        for form, acc, date, pdoc in zip(forms, accessions, dates, primary_docs):
            if form != "8-K":
                continue
            if count >= n_filings:
                break

            acc_clean = acc.replace("-", "")
            doc_url = (f"https://www.sec.gov/Archives/edgar/data/"
                       f"{cik}/{acc_clean}/{pdoc}")
            r = _get(doc_url, headers=self._headers, timeout=30)
            if not r:
                count += 1
                continue
            text = _clean_html(r.text)
            if len(text) < 100:
                count += 1
                continue

            docs.append(SECDocument(
                ticker=ticker,
                form_type="8-K",
                filing_date=date,
                accession_number=acc,
                title=f"{ticker} 8-K {date}",
                text=text[:20_000],
                url=doc_url,
                cik=cik,
            ))
            count += 1
            time.sleep(0.3)
        return docs

    def fetch_proxy_highlights(self, ticker: str) -> str:
        """Fetch DEF 14A proxy — exec compensation narrative."""
        cik = self._get_cik(ticker)
        if not cik:
            return ""

        subs = _fetch_submissions(cik)
        if not subs:
            return ""

        filings = subs.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])
        dates = filings.get("filingDate", [])

        for form, acc, pdoc, date in zip(forms, accessions, primary_docs, dates):
            if form != "DEF 14A":
                continue
            acc_clean = acc.replace("-", "")
            doc_url = (f"https://www.sec.gov/Archives/edgar/data/"
                       f"{cik}/{acc_clean}/{pdoc}")
            r = _get(doc_url, headers=self._headers, timeout=30)
            if not r:
                return ""
            text = _clean_html(r.text)
            # Extract compensation discussion section
            m = re.search(
                r"(compensation discussion and analysis.*?)(?:item \d|$)",
                text, re.I | re.S
            )
            if m:
                return m.group(1)[:10_000]
            return text[:10_000]
        return ""

    def build_ticker_corpus(self, ticker: str) -> List[Document]:
        docs: List[Document] = []

        # 10-K sections
        sections = self.fetch_10k_sections(ticker, sections=["1A", "7", "7A"])
        section_names = {"1A": "Risk Factors", "7": "MD&A", "7A": "Market Risk"}
        for sec_id, text in sections.items():
            for i, chunk in enumerate(_chunk_text(text)):
                docs.append(Document(
                    doc_id=_make_doc_id(ticker, f"10K-{sec_id}", "latest", i),
                    ticker=ticker,
                    source="EDGAR-10K",
                    doc_type="10-K",
                    date=datetime.utcnow().strftime("%Y-%m-%d"),
                    title=f"{ticker} 10-K {section_names.get(sec_id, sec_id)}",
                    text=chunk,
                    metadata={"section": sec_id},
                    chunk_index=i,
                ))

        # 8-K filings
        filings_8k = self.fetch_8k_corpus(ticker, n_filings=10)
        for filing in filings_8k:
            for i, chunk in enumerate(_chunk_text(filing.text)):
                docs.append(Document(
                    doc_id=_make_doc_id(ticker, "8K", filing.filing_date, i),
                    ticker=ticker,
                    source="EDGAR-8K",
                    doc_type="8-K",
                    date=filing.filing_date,
                    title=filing.title,
                    text=chunk,
                    url=filing.url,
                    chunk_index=i,
                ))

        # Proxy
        proxy_text = self.fetch_proxy_highlights(ticker)
        if proxy_text:
            for i, chunk in enumerate(_chunk_text(proxy_text)):
                docs.append(Document(
                    doc_id=_make_doc_id(ticker, "DEF14A", "latest", i),
                    ticker=ticker,
                    source="EDGAR-DEF14A",
                    doc_type="Proxy",
                    date=datetime.utcnow().strftime("%Y-%m-%d"),
                    title=f"{ticker} DEF 14A Executive Compensation",
                    text=chunk,
                    chunk_index=i,
                ))

        return docs


# ---------------------------------------------------------------------------
# News Corpus Builder
# ---------------------------------------------------------------------------

_GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
_REUTERS_RSS = "https://feeds.reuters.com/reuters/businessNews"


class NewsCorpusBuilder:
    """Build news corpus from GDELT, Yahoo Finance RSS, and Reuters RSS."""

    def fetch_gdelt_articles(
        self, query: str, days: int = 30
    ) -> List[NewsArticle]:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        params = {
            "query": query + " sourcelang:english",
            "mode": "ArtList",
            "maxrecords": 50,
            "format": "json",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
            "sort": "DateDesc",
        }
        r = _get(_GDELT_BASE, params=params, timeout=30)
        if not r:
            return []
        try:
            data = r.json()
        except Exception:
            return []

        articles: List[NewsArticle] = []
        for item in data.get("articles", []):
            articles.append(NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                source=item.get("domain", "gdelt"),
                published=item.get("seendate", "")[:8],
                snippet=item.get("title", ""),
                ticker=query.split()[0],
            ))
        return articles

    def fetch_rss_articles(
        self, ticker: str, days: int = 14
    ) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        cutoff = datetime.utcnow() - timedelta(days=days)

        feeds = [
            _YAHOO_RSS.format(ticker=ticker),
            _REUTERS_RSS,
        ]

        for feed_url in feeds:
            r = _get(feed_url, timeout=20)
            if not r:
                continue
            try:
                root = ET.fromstring(r.text)
            except Exception:
                # Try cleaning the XML
                cleaned = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", r.text)
                try:
                    root = ET.fromstring(cleaned)
                except Exception:
                    continue

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)
            for item in items[:30]:
                title_el = item.find("title") or item.find("atom:title", ns)
                link_el = item.find("link") or item.find("atom:link", ns)
                desc_el = item.find("description") or item.find("atom:summary", ns)
                pub_el = item.find("pubDate") or item.find("atom:updated", ns)

                title = title_el.text if title_el is not None else ""
                link = link_el.text if link_el is not None else ""
                if link_el is not None and not link:
                    link = link_el.get("href", "")
                desc = _clean_html(desc_el.text or "") if desc_el is not None else ""
                pub = pub_el.text if pub_el is not None else ""

                if ticker.upper() not in title.upper() and ticker.upper() not in desc.upper():
                    continue

                articles.append(NewsArticle(
                    title=title,
                    url=link,
                    source=feed_url.split("/")[2],
                    published=pub[:10],
                    snippet=desc[:500],
                    ticker=ticker,
                ))
        return articles

    def _fetch_article_text(self, url: str) -> str:
        """Best-effort article body fetch."""
        r = _get(url, timeout=20)
        if not r:
            return ""
        text = _clean_html(r.text)
        # Heuristic: take paragraph-dense region
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 80]
        return " ".join(paragraphs[:30])[:5_000]

    def build_news_corpus(
        self, ticker: str, days: int = 90
    ) -> List[Document]:
        docs: List[Document] = []

        # GDELT
        gdelt_articles = self.fetch_gdelt_articles(f"{ticker} stock", days=days)
        # RSS
        rss_articles = self.fetch_rss_articles(ticker, days=min(days, 30))

        all_articles = gdelt_articles + rss_articles
        seen_urls: set = set()

        for article in all_articles:
            if article.url in seen_urls:
                continue
            seen_urls.add(article.url)

            text = article.snippet
            if not text and article.url:
                text = self._fetch_article_text(article.url)
                time.sleep(0.3)

            if not text:
                continue

            for i, chunk in enumerate(_chunk_text(text, chunk_size=400)):
                docs.append(Document(
                    doc_id=_make_doc_id(ticker, article.source, article.published, i),
                    ticker=ticker,
                    source=article.source,
                    doc_type="news",
                    date=article.published,
                    title=article.title,
                    text=chunk,
                    url=article.url,
                    metadata={"sentiment": article.sentiment},
                    chunk_index=i,
                ))

        return docs


# ---------------------------------------------------------------------------
# Earnings Analytics
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = frozenset(
    "strong growth record beat exceeded outperform robust accelerate expand "
    "improve gain momentum opportunity confident positive upside benefited "
    "increased higher growth delivered exceptional strength outstanding "
    "above raised upgraded favourable".split()
)
_NEGATIVE_WORDS = frozenset(
    "decline decrease lower below miss disappoint challenge headwind difficult "
    "pressure weak shrink slow cautious concern uncertain risk volatile "
    "unfavourable reduced downgrade deteriorate impair loss shortfall".split()
)
_UNCERTAINTY_WORDS = frozenset(
    "expect anticipate may could might should around approximately roughly "
    "estimate guidance target range potential possibly likely".split()
)
_FORWARD_RE = re.compile(
    r"\b(expect|anticipate|forecast|project|guidance|outlook|next quarter|"
    r"next year|fiscal \d{4}|FY\d{2,4}|full.?year|going forward)\b", re.I
)


class EarningsAnalytics:
    """Sentiment, KPI, guidance analytics over EarningsTranscript objects."""

    def __init__(self):
        self._vader = None
        if _VADER_AVAILABLE:
            try:
                self._vader = SentimentIntensityAnalyzer()
            except Exception:
                pass

    def extract_management_tone(
        self, transcript: EarningsTranscript
    ) -> ToneAnalysis:
        # Use prepared remarks only for management tone
        mgmt_text = " ".join(
            s.text for s in transcript.prepared_remarks
            if s.role in ("CEO", "CFO", "") or not s.role
        )
        if not mgmt_text:
            mgmt_text = transcript.full_text[:10_000]

        words = mgmt_text.lower().split()
        total = max(1, len(words))

        pos_count = sum(1 for w in words if w.rstrip(".,;:!?") in _POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w.rstrip(".,;:!?") in _NEGATIVE_WORDS)
        unc_count = sum(1 for w in words if w.rstrip(".,;:!?") in _UNCERTAINTY_WORDS)

        pos_score = pos_count / total
        neg_score = neg_count / total
        unc_score = unc_count / total
        compound = (pos_score - neg_score) / max(pos_score + neg_score, 0.001)

        if _VADER_AVAILABLE and self._vader:
            try:
                vs = self._vader.polarity_scores(mgmt_text[:5000])
                compound = vs["compound"]
            except Exception:
                pass

        fwd_count = len(_FORWARD_RE.findall(mgmt_text))

        dominant = "Neutral"
        if pos_score > neg_score * 1.5:
            dominant = "Positive"
        elif neg_score > pos_score * 1.5:
            dominant = "Negative"
        elif unc_score > 0.03:
            dominant = "Cautious"

        # Extract top positive/negative phrases (5-word windows)
        top_pos: List[str] = []
        top_neg: List[str] = []
        word_list = mgmt_text.split()
        for i, w in enumerate(word_list):
            clean = w.rstrip(".,;:!?").lower()
            window = " ".join(word_list[max(0, i - 2):i + 3])
            if clean in _POSITIVE_WORDS and len(top_pos) < 5:
                top_pos.append(window)
            elif clean in _NEGATIVE_WORDS and len(top_neg) < 5:
                top_neg.append(window)

        return ToneAnalysis(
            ticker=transcript.ticker,
            quarter=transcript.quarter,
            positive_score=round(pos_score, 4),
            negative_score=round(neg_score, 4),
            uncertainty_score=round(unc_score, 4),
            forward_looking_count=fwd_count,
            sentiment_compound=round(compound, 4),
            dominant_tone=dominant,
            top_positive_phrases=top_pos,
            top_negative_phrases=top_neg,
        )

    def extract_kpis(self, transcript: EarningsTranscript) -> List[KPI]:
        text = transcript.full_text
        kpis: List[KPI] = []
        seen: set = set()

        # Dollar amounts with context
        for m in _DOLLAR_RE.finditer(text):
            raw_val = m.group(1).replace(",", "")
            unit = m.group(2).upper()
            multiplier = {"BILLION": 1e9, "B": 1e9, "MILLION": 1e6, "M": 1e6,
                          "THOUSAND": 1e3, "K": 1e3}.get(unit, 1)
            value = float(raw_val) * multiplier

            context_start = max(0, m.start() - 80)
            context = text[context_start:m.end() + 40].strip()

            # Infer KPI name
            name = "Revenue"
            lower_ctx = context.lower()
            if "revenue" in lower_ctx or "sales" in lower_ctx:
                name = "Revenue"
            elif "ebitda" in lower_ctx:
                name = "EBITDA"
            elif "operating income" in lower_ctx or "operating profit" in lower_ctx:
                name = "Operating Income"
            elif "net income" in lower_ctx or "net profit" in lower_ctx:
                name = "Net Income"
            elif "gross" in lower_ctx:
                name = "Gross Profit"
            elif "capex" in lower_ctx or "capital expenditure" in lower_ctx:
                name = "CapEx"
            elif "dividend" in lower_ctx:
                name = "Dividend"
            elif "buyback" in lower_ctx or "repurchase" in lower_ctx:
                name = "Share Buyback"
            else:
                name = "Financial Metric"

            key = f"{name}:{value:.0f}"
            if key in seen:
                continue
            seen.add(key)

            kpis.append(KPI(
                name=name,
                value=value,
                unit="USD",
                context=context,
                quarter=transcript.quarter,
            ))

        # Percentage metrics
        for m in _PCT_RE.finditer(text):
            pct = float(m.group(1))
            if pct > 200 or pct < 0:
                continue
            context_start = max(0, m.start() - 60)
            context = text[context_start:m.end() + 30].strip()
            lower_ctx = context.lower()

            name = "Percentage"
            if "margin" in lower_ctx:
                if "gross" in lower_ctx:
                    name = "Gross Margin"
                elif "operating" in lower_ctx:
                    name = "Operating Margin"
                elif "net" in lower_ctx:
                    name = "Net Margin"
                else:
                    name = "Margin"
            elif "growth" in lower_ctx:
                name = "Growth Rate"
            elif "yield" in lower_ctx:
                name = "Yield"
            else:
                continue

            key = f"{name}:{pct:.1f}"
            if key in seen:
                continue
            seen.add(key)

            kpis.append(KPI(
                name=name,
                value=pct,
                unit="%",
                context=context,
                quarter=transcript.quarter,
            ))

        return kpis[:30]

    def detect_guidance_change(
        self,
        current: EarningsTranscript,
        prior: EarningsTranscript,
    ) -> List[GuidanceDelta]:
        deltas: List[GuidanceDelta] = []

        for metric in ["Revenue", "EPS", "Gross Margin", "Operating Margin"]:
            curr_kpis = [k for k in self.extract_kpis(current) if k.name == metric]
            prior_kpis = [k for k in self.extract_kpis(prior) if k.name == metric]
            if not curr_kpis or not prior_kpis:
                continue

            curr_val = curr_kpis[0].value
            prior_val = prior_kpis[0].value
            if prior_val == 0:
                continue

            pct_change = (curr_val - prior_val) / abs(prior_val) * 100
            direction = "raised" if pct_change > 2 else "lowered" if pct_change < -2 else "maintained"

            deltas.append(GuidanceDelta(
                ticker=current.ticker,
                metric=metric,
                prior_guidance=f"{prior_val:,.0f} {prior_kpis[0].unit}",
                current_guidance=f"{curr_val:,.0f} {curr_kpis[0].unit}",
                direction=direction,
                magnitude_pct=round(pct_change, 2),
                commentary=curr_kpis[0].context[:200],
            ))
        return deltas

    # ------------------------------------------------------------------
    # dim_057 additions
    # ------------------------------------------------------------------

    def extract_guidance_sentences(self, text: str) -> List[str]:
        """Filter sentences containing guidance language near a number.

        A sentence qualifies if it contains at least one guidance keyword
        ("expect", "anticipate", "guide", "outlook", "project") AND
        a number appears within 10 words of that keyword.

        Returns a deduplicated list of qualifying sentences.
        """
        _GUIDANCE_KWS = re.compile(
            r"\b(expect|anticipate|guid\w*|outlook|project\w*|forecast\w*|target)\b",
            re.IGNORECASE,
        )
        _NUMBER_NEAR = re.compile(
            r"\b\d+(?:[.,]\d+)?(?:\s*(?:billion|million|thousand|percent|%|bps?|B|M|K|x))?\b",
            re.IGNORECASE,
        )

        # Sentence splitter
        sentence_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_re.split(text.strip())

        qualifying: List[str] = []
        seen: set = set()

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue

            # Find each guidance keyword position
            kw_match = _GUIDANCE_KWS.search(sent)
            if not kw_match:
                continue

            # Check whether a number appears within 10 words of the keyword
            kw_pos = kw_match.start()
            words = sent.split()
            kw_word_idx: Optional[int] = None
            char_count = 0
            for i, w in enumerate(words):
                if char_count >= kw_pos:
                    kw_word_idx = i
                    break
                char_count += len(w) + 1  # +1 for space

            if kw_word_idx is None:
                kw_word_idx = 0

            window_start = max(0, kw_word_idx - 10)
            window_end   = min(len(words), kw_word_idx + 11)
            window_text  = " ".join(words[window_start:window_end])

            if _NUMBER_NEAR.search(window_text):
                key = sent[:80]
                if key not in seen:
                    seen.add(key)
                    qualifying.append(sent)

        return qualifying

    def compute_tone_shift_score(
        self,
        current_text: str,
        prior_text: str,
    ) -> Dict[str, float]:
        """Compare guidance sentiment between current and prior quarter.

        Sentiment score per text:
            s = (pos_count - neg_count) / max(pos_count + neg_count, 1)

        Returns:
        {
            "current_score":  float in [-1, 1],
            "prior_score":    float in [-1, 1],
            "delta":          current_score - prior_score,
            "direction":      "improving" | "deteriorating" | "stable",
        }
        """
        _POS = frozenset([
            "strong", "growth", "increase", "beat", "exceeded", "record",
            "confident", "positive", "upside", "robust", "accelerat", "improve",
            "outperform", "momentum", "opportunity", "raised", "above",
        ])
        _NEG = frozenset([
            "decline", "decrease", "miss", "below", "concern", "headwind",
            "pressure", "uncertain", "cautious", "challenge", "disappoint",
            "weak", "lower", "reduce", "deteriorat", "volatile", "difficult",
        ])

        def _score(txt: str) -> float:
            words = txt.lower().split()
            pos = sum(1 for w in words if w.rstrip(".,;:!?") in _POS)
            neg = sum(1 for w in words if w.rstrip(".,;:!?") in _NEG)
            total = pos + neg
            return round((pos - neg) / max(total, 1), 4)

        current_score = _score(current_text)
        prior_score   = _score(prior_text)
        delta         = round(current_score - prior_score, 4)

        direction = "stable"
        if delta > 0.05:
            direction = "improving"
        elif delta < -0.05:
            direction = "deteriorating"

        return {
            "current_score": current_score,
            "prior_score":   prior_score,
            "delta":         delta,
            "direction":     direction,
        }

    def build_earnings_timeline(
        self,
        transcripts: List[EarningsTranscript],
    ) -> List[Dict[str, Any]]:
        """Build a timeline of earnings events with beat/miss/in-line labels.

        For each transcript, extract the first revenue KPI and compare it to
        any guidance mentioned in the same text.  Label:
          - "beat"    if actual > guidance by > 1%
          - "miss"    if actual < guidance by > 1%
          - "in-line" otherwise

        Returns a list of dicts sorted by date ascending:
        [{
            "ticker": str,
            "quarter": str,
            "date": str,
            "label": "beat" | "miss" | "in-line" | "unknown",
            "revenue_kpi": float | None,
            "guidance_kpi": float | None,
            "word_count": int,
        }]
        """
        _GUIDANCE_RE = re.compile(
            r"(?:guidance|expect|project|forecast|anticipate)[^\d]{0,40}"
            r"\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)?",
            re.IGNORECASE,
        )
        _ACTUAL_RE = re.compile(
            r"(?:revenue|sales|net revenue)[^\d]{0,30}"
            r"\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)?",
            re.IGNORECASE,
        )

        def _parse_val(num_str: str, unit_str: str) -> Optional[float]:
            try:
                val = float(num_str.replace(",", ""))
                unit = (unit_str or "").lower()
                mult = {"billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6}.get(unit, 1.0)
                return val * mult
            except (ValueError, TypeError):
                return None

        timeline: List[Dict[str, Any]] = []
        for t in transcripts:
            text = t.full_text

            # Extract first actual revenue mention
            actual_val: Optional[float] = None
            am = _ACTUAL_RE.search(text)
            if am:
                actual_val = _parse_val(am.group(1), am.group(2))

            # Extract first guidance revenue mention
            guidance_val: Optional[float] = None
            gm = _GUIDANCE_RE.search(text)
            if gm:
                guidance_val = _parse_val(gm.group(1), gm.group(2))

            # Label
            label = "unknown"
            if actual_val is not None and guidance_val is not None and guidance_val > 0:
                ratio = actual_val / guidance_val
                if ratio > 1.01:
                    label = "beat"
                elif ratio < 0.99:
                    label = "miss"
                else:
                    label = "in-line"

            timeline.append({
                "ticker":       t.ticker,
                "quarter":      t.quarter,
                "date":         t.date,
                "label":        label,
                "revenue_kpi":  actual_val,
                "guidance_kpi": guidance_val,
                "word_count":   t.word_count,
            })

        # Sort by date ascending (ISO date strings sort lexicographically)
        timeline.sort(key=lambda x: x["date"])
        return timeline

    def compute_call_quality_score(self, transcript: EarningsTranscript) -> float:
        """Score 0-100 based on call completeness and detail."""
        score = 0.0

        # Word count: 2000+ words = good call
        wc = transcript.word_count
        score += min(25.0, wc / 100)

        # Has Q&A
        if transcript.qa_sections:
            score += 20.0

        # Forward-looking statements
        fwd_count = len(_FORWARD_RE.findall(transcript.full_text))
        score += min(20.0, fwd_count * 2)

        # KPI mentions
        kpis = self.extract_kpis(transcript)
        score += min(20.0, len(kpis) * 2)

        # Specific numbers (dollar amounts)
        dollar_count = len(_DOLLAR_RE.findall(transcript.full_text))
        score += min(15.0, dollar_count * 1.5)

        return round(min(100.0, score), 1)


# ---------------------------------------------------------------------------
# Financial QA System
# ---------------------------------------------------------------------------

_EXTRACTIVE_SENT_RE = re.compile(r"[.!?]+\s+")


def _extractive_answer(question: str, chunks: List[RetrievedDocument]) -> str:
    """Return the best sentence from top retrieved chunk."""
    if not chunks:
        return "No relevant information found."

    top_text = chunks[0].document.text
    sentences = _EXTRACTIVE_SENT_RE.split(top_text)
    if not sentences:
        return top_text[:500]

    q_words = set(question.lower().split())
    best_sent = max(sentences, key=lambda s: len(q_words & set(s.lower().split())))
    return best_sent.strip()


class FinancialQASystem:
    """RAG: retrieve → augment → generate (Claude or extractive fallback)."""

    def __init__(
        self,
        vector_store: LocalVectorStore = None,
        use_claude: bool = True,
    ):
        self.vector_store = vector_store or LocalVectorStore()
        self.use_claude = use_claude and _ANTHROPIC_AVAILABLE
        self._claude_client = None
        if self.use_claude:
            try:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if api_key:
                    self._claude_client = anthropic.Anthropic(api_key=api_key)
            except Exception as exc:
                logger.warning("Claude client init failed: %s", exc)
                self.use_claude = False

    def _format_context(self, chunks: List[RetrievedDocument]) -> str:
        lines: List[str] = []
        for r in chunks:
            doc = r.document
            lines.append(
                f"[Source: {doc.source} | {doc.ticker} | {doc.date} | score={r.score:.3f}]\n"
                f"{doc.text}\n"
            )
        return "\n---\n".join(lines)

    def answer(
        self,
        question: str,
        ticker: str = None,
        corpus: str = "all",
        k: int = 8,
    ) -> RAGAnswer:
        chunks = self.vector_store.search(question, k=k)

        if ticker:
            chunks = [c for c in chunks if not ticker or c.document.ticker == ticker] or chunks

        if corpus != "all":
            type_map = {
                "earnings": ["earnings", "transcript"],
                "filings": ["10-K", "8-K", "Proxy"],
                "news": ["news"],
            }
            allowed = type_map.get(corpus, [])
            filtered = [c for c in chunks if c.document.doc_type in allowed]
            chunks = filtered or chunks

        context = self._format_context(chunks)

        if self.use_claude and self._claude_client:
            ans_text, method = self._claude_answer(question, context)
        else:
            ans_text = _extractive_answer(question, chunks)
            method = "extractive"

        confidence = float(chunks[0].score) if chunks else 0.0

        return RAGAnswer(
            question=question,
            answer=ans_text,
            ticker=ticker or "",
            sources=chunks,
            confidence=round(confidence, 4),
            method=method,
        )

    def _claude_answer(self, question: str, context: str) -> Tuple[str, str]:
        prompt = (
            f"You are a financial analyst assistant. Using ONLY the provided context, "
            f"answer the question concisely with specific data points and cite sources.\n\n"
            f"Context:\n{context[:12_000]}\n\n"
            f"Question: {question}\n\n"
            f"Answer (include specific numbers, dates, and source citations):"
        )
        try:
            msg = self._claude_client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text, "claude"
        except Exception as exc:
            logger.warning("Claude API call failed: %s", exc)
            return _extractive_answer(question, []), "extractive-fallback"

    def extract_guidance(self, ticker: str) -> GuidanceExtraction:
        guidance_q = f"{ticker} revenue guidance outlook forward fiscal year EPS"
        chunks = self.vector_store.search(guidance_q, k=10)

        raw_quotes: List[str] = []
        for c in chunks:
            fwd_matches = _FORWARD_RE.findall(c.document.text)
            if fwd_matches:
                raw_quotes.append(c.document.text[:300])

        def _extract_metric(keyword: str) -> str:
            for c in chunks:
                if keyword.lower() in c.document.text.lower():
                    m = re.search(
                        rf"{keyword}[^.]*\$?[\d,.]+ ?(billion|million|%)?",
                        c.document.text, re.I
                    )
                    if m:
                        return m.group(0)[:200]
            return "Not found"

        return GuidanceExtraction(
            ticker=ticker,
            revenue_guidance=_extract_metric("revenue"),
            eps_guidance=_extract_metric("EPS"),
            margin_guidance=_extract_metric("margin"),
            capex_guidance=_extract_metric("capex"),
            other=[],
            raw_quotes=raw_quotes[:5],
        )

    def get_risk_factors(self, ticker: str) -> List[str]:
        q = f"{ticker} risk factors threats regulatory competition"
        chunks = self.vector_store.search(q, k=10)
        risks: List[str] = []
        risk_re = re.compile(
            r"(risk|threat|challenge|uncertainty|headwind)[^.]{10,200}\.", re.I
        )
        for c in chunks:
            for m in risk_re.finditer(c.document.text):
                txt = m.group(0).strip()
                if txt not in risks:
                    risks.append(txt)
                if len(risks) >= 10:
                    break
            if len(risks) >= 10:
                break
        return risks

    def compare_tickers(
        self, ticker1: str, ticker2: str, question: str
    ) -> str:
        ans1 = self.answer(question, ticker=ticker1)
        ans2 = self.answer(question, ticker=ticker2)
        return (
            f"=== {ticker1} ===\n{ans1.answer}\n\n"
            f"=== {ticker2} ===\n{ans2.answer}"
        )


# ---------------------------------------------------------------------------
# Earnings RAG Engine (Orchestrator)
# ---------------------------------------------------------------------------

class EarningsRAGEngine:
    """Top-level orchestrator: index tickers, answer questions, generate briefs."""

    def __init__(
        self,
        use_chroma: bool = True,
        chroma_dir: str = "sentinel/data/chroma_db",
        cache_dir: str = "sentinel/data/transcripts_cache",
    ):
        self.collector = EarningsCallCollector(cache_dir=cache_dir)
        self.sec_builder = SECFilingCorpusBuilder()
        self.news_builder = NewsCorpusBuilder()
        self.analytics = EarningsAnalytics()

        # Vector store
        if use_chroma and _CHROMA_AVAILABLE:
            self._store_backend: Any = ChromaDBAdapter(persist_dir=chroma_dir)
            logger.info("EarningsRAGEngine: using ChromaDB backend")
        else:
            self._store_backend = LocalVectorStore()
            logger.info("EarningsRAGEngine: using LocalVectorStore (numpy TF-IDF)")

        self._local_store = LocalVectorStore()
        self.qa = FinancialQASystem(vector_store=self._local_store)

        self._indexed_tickers: set = set()
        self._corpus_stats: List[dict] = []

    def _add_docs(self, docs: List[Document]) -> None:
        self._local_store.add_documents(docs)
        if hasattr(self._store_backend, "add_documents"):
            try:
                self._store_backend.add_documents(docs)
            except Exception as exc:
                logger.debug("ChromaDB add failed: %s", exc)

    def index_ticker(self, ticker: str, years: int = 3) -> None:
        """Fetch and index all earnings, SEC, and news docs for ticker."""
        logger.info("Indexing %s (%d years)...", ticker, years)
        all_docs: List[Document] = []

        # Transcripts
        n_quarters = years * 4
        transcripts = self.collector.fetch_recent_transcripts(ticker, n=n_quarters)
        for t in transcripts:
            for i, chunk in enumerate(_chunk_text(t.full_text)):
                all_docs.append(Document(
                    doc_id=_make_doc_id(ticker, "transcript", t.quarter, i),
                    ticker=ticker,
                    source=t.source,
                    doc_type="earnings",
                    date=t.date,
                    title=f"{ticker} {t.quarter} Earnings Call",
                    text=chunk,
                    url=t.url,
                    metadata={"quarter": t.quarter, "year": t.year},
                    chunk_index=i,
                ))

        # SEC filings
        sec_docs = self.sec_builder.build_ticker_corpus(ticker)
        all_docs.extend(sec_docs)

        # News
        news_docs = self.news_builder.build_news_corpus(ticker, days=years * 365)
        all_docs.extend(news_docs)

        self._add_docs(all_docs)
        self._indexed_tickers.add(ticker)

        self._corpus_stats.append({
            "ticker": ticker,
            "transcript_count": len(transcripts),
            "sec_doc_count": len(sec_docs),
            "news_doc_count": len(news_docs),
            "total_docs": len(all_docs),
            "indexed_at": datetime.utcnow().isoformat(),
        })
        logger.info(
            "Indexed %s: %d transcripts, %d SEC docs, %d news chunks",
            ticker, len(transcripts), len(sec_docs), len(news_docs),
        )

    def query(self, ticker: str, question: str) -> RAGAnswer:
        if ticker not in self._indexed_tickers:
            logger.info("Auto-indexing %s...", ticker)
            self.index_ticker(ticker, years=2)
        return self.qa.answer(question, ticker=ticker)

    def get_earnings_brief(
        self, ticker: str, quarter: str
    ) -> Optional[EarningsBrief]:
        transcript = self.collector.fetch_transcript(ticker, quarter)
        if not transcript:
            return None

        tone = self.analytics.extract_management_tone(transcript)
        kpis = self.analytics.extract_kpis(transcript)

        # Guidance highlights
        guidance_q = self.qa.extract_guidance(ticker)
        guidance_highlights = [
            g for g in [
                guidance_q.revenue_guidance,
                guidance_q.eps_guidance,
                guidance_q.margin_guidance,
            ] if g and g != "Not found"
        ]

        # Risk mentions from transcript
        risk_re = re.compile(r"(risk|headwind|challenge|pressure)[^.]{5,150}\.", re.I)
        risks = [m.group(0).strip() for m in risk_re.finditer(transcript.full_text)][:5]

        # Analyst concerns from Q&A
        analyst_concerns: List[str] = []
        for section in transcript.qa_sections:
            if section.role == "Analyst" and len(section.text) > 50:
                # Take first sentence
                first_sent = section.text.split(".")[0]
                analyst_concerns.append(first_sent.strip())
                if len(analyst_concerns) >= 5:
                    break

        # Summary
        summary = (
            f"{ticker} {quarter} earnings call. "
            f"Management tone: {tone.dominant_tone}. "
            f"KPIs mentioned: {len(kpis)}. "
            f"Forward-looking statements: {tone.forward_looking_count}."
        )

        return EarningsBrief(
            ticker=ticker,
            quarter=quarter,
            summary=summary,
            key_metrics=kpis[:10],
            tone=tone,
            guidance_highlights=guidance_highlights,
            risk_mentions=risks,
            analyst_concerns=analyst_concerns,
        )

    def compare_quarter_narratives(
        self, ticker: str, q1: str, q2: str
    ) -> str:
        t1 = self.collector.fetch_transcript(ticker, q1)
        t2 = self.collector.fetch_transcript(ticker, q2)
        if not t1 or not t2:
            return f"Could not fetch one or both transcripts for {ticker}: {q1}, {q2}"

        tone1 = self.analytics.extract_management_tone(t1)
        tone2 = self.analytics.extract_management_tone(t2)
        kpis1 = self.analytics.extract_kpis(t1)
        kpis2 = self.analytics.extract_kpis(t2)
        deltas = self.analytics.detect_guidance_change(t2, t1)

        lines = [
            f"Narrative Comparison: {ticker} — {q1} vs {q2}",
            "=" * 60,
            f"{q1} Tone: {tone1.dominant_tone} "
            f"(pos={tone1.positive_score:.3f}, neg={tone1.negative_score:.3f})",
            f"{q2} Tone: {tone2.dominant_tone} "
            f"(pos={tone2.positive_score:.3f}, neg={tone2.negative_score:.3f})",
            f"\nKPIs {q1}: {len(kpis1)} metrics found",
            f"KPIs {q2}: {len(kpis2)} metrics found",
        ]
        if deltas:
            lines.append("\nGuidance Changes:")
            for d in deltas:
                lines.append(
                    f"  {d.metric}: {d.prior_guidance} → {d.current_guidance} "
                    f"({d.direction}, {d.magnitude_pct:+.1f}%)"
                )
        return "\n".join(lines)

    def screen_by_narrative(
        self, tickers: List[str], criteria: str
    ) -> List[str]:
        """Return tickers whose earnings narratives match the criteria."""
        matches: List[str] = []
        for ticker in tickers:
            if ticker not in self._indexed_tickers:
                self.index_ticker(ticker, years=1)
            ans = self.qa.answer(criteria, ticker=ticker)
            if ans.confidence > 0.1 and ans.answer and "not found" not in ans.answer.lower():
                matches.append(ticker)
        return matches

    def export_corpus_stats(self) -> pd.DataFrame:
        if not self._corpus_stats:
            return pd.DataFrame()
        return pd.DataFrame(self._corpus_stats)


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

earnings_rag_v3_router = APIRouter(prefix="/rag/v3", tags=["earnings-rag-v3"])

_engine_singleton: Optional[EarningsRAGEngine] = None


def _get_engine() -> EarningsRAGEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = EarningsRAGEngine()
    return _engine_singleton


class IndexRequest(BaseModel):
    ticker: str
    years: int = 2


class QueryRequest(BaseModel):
    ticker: str
    question: str
    corpus: str = "all"


class ScreenRequest(BaseModel):
    tickers: List[str]
    criteria: str


@earnings_rag_v3_router.post("/index")
def api_index(req: IndexRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_get_engine().index_ticker, req.ticker, req.years)
    return {"status": "indexing_started", "ticker": req.ticker}


@earnings_rag_v3_router.post("/query")
def api_query(req: QueryRequest):
    eng = _get_engine()
    ans = eng.query(req.ticker, req.question)
    return {
        "answer": ans.answer,
        "confidence": ans.confidence,
        "method": ans.method,
        "sources": [
            {"source": r.document.source, "date": r.document.date,
             "score": r.score, "text": r.document.text[:200]}
            for r in ans.sources[:5]
        ],
    }


@earnings_rag_v3_router.get("/brief/{ticker}/{quarter}")
def api_brief(ticker: str, quarter: str):
    eng = _get_engine()
    brief = eng.get_earnings_brief(ticker, quarter.replace("-", " "))
    if not brief:
        raise HTTPException(404, f"No transcript found for {ticker} {quarter}")
    import dataclasses
    return dataclasses.asdict(brief)


@earnings_rag_v3_router.post("/screen")
def api_screen(req: ScreenRequest):
    eng = _get_engine()
    matches = eng.screen_by_narrative(req.tickers, req.criteria)
    return {"matches": matches, "criteria": req.criteria}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    logger.info("EarningsRAGEngine — indexing AAPL for last 2 years...")

    engine = EarningsRAGEngine(use_chroma=False)
    engine.index_ticker("AAPL", years=2)

    question = "What guidance did management give for 2025?"
    logger.info("Query: %s", question)
    answer = engine.query("AAPL", question)

    print("\n" + "=" * 70)
    print(f"Question: {answer.question}")
    print(f"Method:   {answer.method}")
    print(f"Confidence: {answer.confidence:.3f}")
    print(f"\nAnswer:\n{answer.answer}")
    if answer.sources:
        print(f"\nTop source: [{answer.sources[0].document.source}] "
              f"{answer.sources[0].document.title[:80]}")

    stats = engine.export_corpus_stats()
    if not stats.empty:
        print(f"\nCorpus stats:\n{stats.to_string(index=False)}")
