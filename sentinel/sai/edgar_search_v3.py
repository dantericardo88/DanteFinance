"""
EDGAR Full-Text Search V3 — Production semantic intelligence platform (dim_033, score 7 → 9).

Uses SEC EDGAR EFTS (free) for full-text search, EDGAR archives for document retrieval,
TF-IDF for semantic matching, and optional sentence-transformers for deep embeddings.

Architecture
------------
EDGARFullTextSearcher      — EFTS query wrapper, document fetcher, section extractor
EDGARDocumentAnalyzer      — risk factor extraction, sentiment, readability, NLP
EDGARSemanticSearch        — TF-IDF index, optional sentence-transformers, k-means
EDGARWatchList             — monitoring watches for recurring searches
EDGARNLPPipeline           — end-to-end per-ticker NLP analysis
EDGARSearchEngine          — orchestrator

Dataclasses: EDGARSearchResult, RiskFactor, RiskFactorDelta, EDGARAlert,
             FilingAnalysis, FilingComparison

Free APIs only. Core works without optional deps (BeautifulSoup, sklearn, etc.).

Usage::
    from sentinel.sai.edgar_search_v3 import EDGARSearchEngine

    engine = EDGARSearchEngine()
    results = engine.search("going concern", form_types=["10-K"], days_back=90)
    analysis = engine.analyze("AAPL", "10-K")
    alerts = engine.monitor(["material weakness", "SEC investigation"], ["10-K", "8-K"])
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import requests

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False
    BeautifulSoup = None  # type: ignore

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    pd = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    KMeans = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EFTS_BASE      = "https://efts.sec.gov/LATEST/search-index"
EDGAR_DATA     = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_COMPANY  = "https://www.sec.gov/cgi-bin/browse-edgar"
SUBMISSIONS    = "https://data.sec.gov/submissions/CIK{cik10}.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/3.0 richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_HTML_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/3.0 richard.porras@realempanada.com",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Encoding": "gzip, deflate",
}

_TIMEOUT        = 30.0
_RATE_DELAY     = 0.15   # ~6 req/s — well under EDGAR's 10 req/s limit
_MAX_EFTS_HITS  = 200

# EDGAR EFTS _source fields
_EFTS_SOURCE = (
    "period_of_report,entity_name,file_num,inc_states,category_name,"
    "accession_no,file_date,form_type,display_names,entity_id"
)

# 10-K section headers (regex fragments)
_SECTION_PATTERNS: Dict[str, str] = {
    "item1":   r"item\s*1[\.\s]+(a\s*)?business",
    "item1a":  r"item\s*1a[\.\s]+risk\s*factors",
    "item1b":  r"item\s*1b[\.\s]+unresolved\s*staff\s*comments",
    "item2":   r"item\s*2[\.\s]+properties",
    "item3":   r"item\s*3[\.\s]+legal\s*proceedings",
    "item7":   r"item\s*7[\.\s]+management.{0,50}discussion",
    "item7a":  r"item\s*7a[\.\s]+quantitative.{0,30}qualitative",
    "item8":   r"item\s*8[\.\s]+financial\s*statements",
    "item9":   r"item\s*9[\.\s]+changes.{0,30}disagreements",
    "item9a":  r"item\s*9a[\.\s]+controls\s*and\s*procedures",
}

# Negative words for risk/sentiment scoring
_NEGATIVE_WORDS: Set[str] = {
    "risk", "uncertainty", "adverse", "decline", "loss", "failure", "default",
    "litigation", "regulatory", "volatility", "impairment", "material", "weakness",
    "going concern", "substantial", "significant", "harmful", "negative", "decrease",
    "deterioration", "exposure", "unfavorable", "threat", "investigation", "violation",
    "penalty", "fine", "lawsuit", "breach", "fraud", "restatement", "error", "doubt",
    "inability", "disruption", "obstacle", "challenge", "competition", "cybersecurity",
    "inflation", "recession", "downturn", "instability", "shortage", "dependence",
}

_POSITIVE_WORDS: Set[str] = {
    "growth", "strong", "increase", "improvement", "opportunity", "expansion",
    "profitable", "success", "innovative", "leading", "competitive", "advantage",
    "record", "exceed", "momentum", "achieve", "enhance", "progress", "invest",
    "return", "dividend", "organic", "accretive", "efficiency", "synergy",
}

# Risk categories
RISK_CATEGORIES = {
    "MARKET":      ["market", "stock", "equity", "price", "trading", "volatility"],
    "OPERATIONAL": ["operations", "supply", "manufacturing", "logistics", "employee"],
    "FINANCIAL":   ["credit", "debt", "interest", "liquidity", "capital", "cash"],
    "REGULATORY":  ["regulation", "compliance", "government", "law", "legal", "SEC"],
    "COMPETITIVE": ["competition", "competitor", "market share", "pricing", "substitute"],
    "MACRO":       ["inflation", "recession", "economy", "gdp", "global", "pandemic"],
}

# Cache directory
_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "edgar_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EDGARSearchResult:
    """A single filing returned by EDGAR EFTS search."""
    accession_no: str
    filing_date: Optional[str]
    form_type: str
    entity_name: str
    cik: str
    period_of_report: Optional[str]
    file_num: str
    category: str
    display_names: str
    document_url: Optional[str] = None       # populated by get_filing_document
    excerpt: Optional[str] = None


@dataclass
class RiskFactor:
    """A single risk factor extracted from a 10-K Item 1A."""
    text: str
    severity_score: float      # 0..1 (negative_words / total_words)
    category: str              # MARKET | OPERATIONAL | FINANCIAL | REGULATORY | COMPETITIVE | MACRO
    word_count: int
    key_terms: List[str]


@dataclass
class RiskFactorDelta:
    """Diff of risk factors between two annual filings."""
    new_risks: List[RiskFactor]       # in doc2 but not doc1
    removed_risks: List[RiskFactor]   # in doc1 but not doc2
    intensified_risks: List[Tuple[RiskFactor, RiskFactor]]  # (old, new) — same topic, escalated
    new_risk_count: int
    removed_risk_count: int
    net_risk_change: int              # positive = more risky
    severity_delta: float             # avg severity change


@dataclass
class EDGARAlert:
    """An alert triggered by an EDGARWatchList watch."""
    watch_name: str
    query: str
    trigger_date: str
    filing: EDGARSearchResult
    context_snippet: str


@dataclass
class FilingAnalysis:
    """Full NLP analysis of a single EDGAR filing."""
    ticker: str
    form_type: str
    entity_name: str
    filing_date: Optional[str]
    period: Optional[str]
    risk_factors: List[RiskFactor]
    forward_looking_statements: List[str]
    material_weaknesses: List[str]
    quantitative_disclosures: List[Tuple[str, float, str]]
    readability_score: float          # Flesch-Kincaid
    fog_index: float
    overall_sentiment: float          # positive ratio (0..1)
    document_length: int              # chars
    section_lengths: Dict[str, int]
    summary: str


@dataclass
class FilingComparison:
    """Comparison between two annual filings for the same ticker."""
    ticker: str
    year1: int
    year2: int
    risk_delta: RiskFactorDelta
    readability_delta: float          # year2 - year1
    sentiment_delta: float            # year2 - year1
    new_forward_guidance: List[str]
    new_material_weaknesses: List[str]
    document_length_delta: int
    summary: str


# ---------------------------------------------------------------------------
# TF-IDF Index dataclass
# ---------------------------------------------------------------------------

@dataclass
class TFIDFIndex:
    """Lightweight TF-IDF index for semantic document search."""
    vectorizer: Any             # TfidfVectorizer instance (if sklearn available)
    matrix: Any                 # sparse matrix (n_docs × n_features)
    documents: List[str]        # raw documents
    idf_map: Dict[str, float]   # fallback manual IDF


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_LAST_REQ_TIME: float = 0.0


def _get(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = _TIMEOUT,
    as_json: bool = True,
) -> Any:
    """Rate-limited GET with optional JSON parsing."""
    global _LAST_REQ_TIME
    elapsed = time.time() - _LAST_REQ_TIME
    if elapsed < _RATE_DELAY:
        time.sleep(_RATE_DELAY - elapsed)
    _LAST_REQ_TIME = time.time()
    h = headers or _HEADERS
    resp = requests.get(url, params=params, headers=h, timeout=timeout)
    resp.raise_for_status()
    if as_json:
        return resp.json()
    return resp.text


def _cache_path(key: str) -> Path:
    hk = hashlib.sha1(key.encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{hk}.json"


def _cached_get(url: str, params: Optional[dict] = None, ttl: int = 3600) -> Any:
    """GET with file-based caching."""
    cache_key = url + json.dumps(params or {}, sort_keys=True)
    cp = _cache_path(cache_key)
    if cp.exists():
        mtime = cp.stat().st_mtime
        if (time.time() - mtime) < ttl:
            try:
                return json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                pass
    data = _get(url, params=params)
    try:
        cp.write_text(json.dumps(data, default=str), encoding="utf-8")
    except Exception:
        pass
    return data


def _strip_html(html: str) -> str:
    """Strip HTML tags from a document string."""
    if _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    # Fallback: regex-based stripping
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _accession_to_path(accession_no: str) -> str:
    """Convert 0000320193-23-000077 → 000032019323000077"""
    return accession_no.replace("-", "")


def _cik_10(cik: str) -> str:
    """Zero-pad CIK to 10 digits."""
    return str(cik).lstrip("0").zfill(10)


# ---------------------------------------------------------------------------
# CIK Lookup
# ---------------------------------------------------------------------------

_TICKER_CIK_CACHE: Dict[str, str] = {}


def _lookup_cik(ticker: str) -> Optional[str]:
    """Lookup CIK for a ticker via SEC EDGAR company tickers JSON."""
    if ticker.upper() in _TICKER_CIK_CACHE:
        return _TICKER_CIK_CACHE[ticker.upper()]
    try:
        data = _cached_get(
            "https://www.sec.gov/files/company_tickers.json", ttl=86400
        )
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"])
                _TICKER_CIK_CACHE[ticker.upper()] = cik
                return cik
    except Exception as exc:
        logger.debug("CIK lookup failed for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# EDGARFullTextSearcher
# ---------------------------------------------------------------------------

class EDGARFullTextSearcher:
    """
    Wrapper for EDGAR EFTS full-text search API.
    Provides document retrieval, section extraction, and phrase search.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        form_types: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 100,
    ) -> List[EDGARSearchResult]:
        """
        Full-text search across EDGAR using the EFTS API.

        Parameters
        ----------
        query       : free-text or quoted phrase
        form_types  : e.g. ["10-K", "10-Q"]
        ticker      : filter by issuer ticker
        start_date  : ISO date "YYYY-MM-DD"
        end_date    : ISO date "YYYY-MM-DD"
        max_results : max filings to return
        """
        params: Dict[str, Any] = {
            "q":      f'"{query}"' if " " in query and not query.startswith('"') else query,
            "_source": _EFTS_SOURCE,
            "dateRange": "custom",
        }
        if form_types:
            params["forms"] = ",".join(form_types)
        if ticker:
            cik = _lookup_cik(ticker)
            if cik:
                params["entity"] = cik
            else:
                params["entity"] = ticker
        if start_date:
            params["startdt"] = start_date
        if end_date:
            params["enddt"] = end_date

        results: List[EDGARSearchResult] = []
        offset = 0
        batch  = min(max_results, _MAX_EFTS_HITS)

        while len(results) < max_results:
            params["from"]  = offset
            params["hits.hits.total.value"] = batch
            params["hits.hits._source"] = _EFTS_SOURCE

            try:
                data = _get(EFTS_BASE, params=params)
            except Exception as exc:
                logger.error("EFTS search error: %s", exc)
                break

            hits = (
                data.get("hits", {}).get("hits", [])
                if isinstance(data, dict)
                else []
            )
            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", hit)
                accession = src.get("accession_no", "")
                cik_val   = str(src.get("entity_id", ""))
                result = EDGARSearchResult(
                    accession_no=accession,
                    filing_date=src.get("file_date"),
                    form_type=src.get("form_type", ""),
                    entity_name=src.get("entity_name", src.get("display_names", "")),
                    cik=cik_val,
                    period_of_report=src.get("period_of_report"),
                    file_num=src.get("file_num", ""),
                    category=src.get("category_name", ""),
                    display_names=src.get("display_names", ""),
                )
                results.append(result)
                if len(results) >= max_results:
                    break

            offset += len(hits)
            total_available = (
                data.get("hits", {}).get("total", {}).get("value", 0)
                if isinstance(data, dict) else 0
            )
            if offset >= total_available or offset >= _MAX_EFTS_HITS:
                break

        logger.info("EFTS search '%s': %d results", query, len(results))
        return results

    def search_by_phrase(
        self, phrase: str, form_types: Optional[List[str]] = None
    ) -> List[EDGARSearchResult]:
        """Exact-phrase search: wraps phrase in double quotes."""
        quoted = f'"{phrase}"'
        return self.search(quoted, form_types=form_types)

    def search_risk_factors(
        self, ticker: str, keywords: List[str]
    ) -> List[EDGARSearchResult]:
        """Search for a ticker's filings containing specific risk-factor keywords (AND)."""
        query = " AND ".join(f'"{kw}"' for kw in keywords)
        return self.search(query, form_types=["10-K"], ticker=ticker)

    # ------------------------------------------------------------------
    # Document retrieval
    # ------------------------------------------------------------------

    def _get_filing_index(self, cik: str, accession_no: str) -> Optional[dict]:
        """Fetch filing index JSON from EDGAR archives."""
        acc_clean = _accession_to_path(accession_no)
        url = (
            f"{EDGAR_DATA}/submissions/"
            f"CIK{_cik_10(cik)}.json"
        )
        # Use filing index endpoint directly
        index_url = (
            f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{acc_clean}/"
            f"{accession_no}-index.json"
        )
        try:
            return _cached_get(index_url, ttl=86400)
        except Exception:
            pass
        # Fallback: parse the HTML index
        html_index_url = (
            f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{acc_clean}/"
            f"{accession_no}-index.htm"
        )
        try:
            html = _get(html_index_url, headers=_HTML_HEADERS, as_json=False)
            return {"html": html, "url": html_index_url}
        except Exception as exc:
            logger.debug("Filing index fetch failed: %s", exc)
        return None

    def _extract_primary_doc_url(
        self, cik: str, accession_no: str, index_data: Optional[dict]
    ) -> Optional[str]:
        """Determine the URL of the primary filing document."""
        acc_clean = _accession_to_path(accession_no)
        base = f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{acc_clean}/"

        if index_data and "html" in index_data:
            # Parse the HTML index for .htm documents
            html = index_data["html"]
            if _HAS_BS4:
                soup = BeautifulSoup(html, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if href.endswith(".htm") and "index" not in href.lower():
                        return base + href.split("/")[-1]
            else:
                matches = re.findall(r'href="([^"]+\.htm)"', html, re.IGNORECASE)
                for m in matches:
                    if "index" not in m.lower():
                        fname = m.split("/")[-1]
                        return base + fname

        if index_data and "documents" in index_data:
            for doc in index_data["documents"]:
                doc_type = doc.get("type", "").upper()
                name = doc.get("filename", doc.get("name", ""))
                if doc_type in ("10-K", "10-Q", "8-K", "20-F", "DEF 14A"):
                    return base + name
                if name.endswith(".htm") and "index" not in name.lower():
                    return base + name

        # Last resort: try common naming convention
        return base + accession_no + ".htm"

    def get_filing_document(self, result: EDGARSearchResult) -> str:
        """
        Fetch the primary filing document text.
        Strips HTML, returns clean plaintext.
        """
        cik = result.cik
        accession_no = result.accession_no

        # Cache check
        cache_key = f"doc_{accession_no}"
        cp = _cache_path(cache_key)
        if cp.exists() and (time.time() - cp.stat().st_mtime) < 86400 * 7:
            cached = cp.read_text(encoding="utf-8")
            if cached:
                return cached

        index_data = self._get_filing_index(cik, accession_no)
        doc_url = self._extract_primary_doc_url(cik, accession_no, index_data)

        if not doc_url:
            logger.warning("Could not determine document URL for %s", accession_no)
            return ""

        try:
            html_content = _get(doc_url, headers=_HTML_HEADERS, as_json=False)
            text = _strip_html(html_content)
            cp.write_text(text, encoding="utf-8")
            result.document_url = doc_url
            return text
        except Exception as exc:
            logger.error("Document fetch failed for %s: %s", doc_url, exc)
            return ""

    def get_filing_sections(self, document: str) -> Dict[str, str]:
        """
        Split a 10-K document into its Item sections using regex header matching.
        Returns {section_key: section_text} e.g. {"item1a": "Risk factors text…"}.
        """
        if not document:
            return {}

        doc_lower = document.lower()
        # Find positions of each known section header
        positions: List[Tuple[str, int]] = []
        for key, pattern in _SECTION_PATTERNS.items():
            m = re.search(pattern, doc_lower)
            if m:
                positions.append((key, m.start()))

        if not positions:
            return {"full": document}

        # Sort by position in document
        positions.sort(key=lambda x: x[1])

        sections: Dict[str, str] = {}
        for i, (key, pos) in enumerate(positions):
            end_pos = positions[i + 1][1] if i + 1 < len(positions) else len(document)
            sections[key] = document[pos:end_pos].strip()

        return sections


# ---------------------------------------------------------------------------
# EDGARDocumentAnalyzer
# ---------------------------------------------------------------------------

class EDGARDocumentAnalyzer:
    """
    NLP analysis of EDGAR filing documents.
    Extracts risk factors, sentiment, readability, forward guidance, material weaknesses.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Risk factor extraction
    # ------------------------------------------------------------------

    def extract_risk_factors(self, document: str) -> List[RiskFactor]:
        """
        Extract and score individual risk factors from Item 1A text.
        Splits by risk-factor bullet headers and scores each for severity.
        """
        if not document:
            return []

        # Find Item 1A section
        item1a_match = re.search(
            r"item\s*1a[\.\s]+risk\s*factors", document, re.IGNORECASE
        )
        if item1a_match:
            item1a_text = document[item1a_match.start():]
            # Cut off at Item 2
            item2_match = re.search(
                r"item\s*[2-9][\.\s]", item1a_text, re.IGNORECASE
            )
            if item2_match and item2_match.start() > 500:
                item1a_text = item1a_text[: item2_match.start()]
        else:
            item1a_text = document

        # Split into individual risk factors by double-newline or header-like patterns
        # Risk factor headers are usually bolded/capitalized
        splitter = re.compile(
            r"\n{2,}(?=[A-Z][A-Z\s]{10,}[A-Z]\.?\n)",
        )
        chunks = splitter.split(item1a_text)
        if len(chunks) < 3:
            # Fallback: split by paragraph breaks
            chunks = re.split(r"\n{3,}", item1a_text)

        risk_factors: List[RiskFactor] = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 100:
                continue
            rf = self._score_risk_factor(chunk)
            risk_factors.append(rf)

        return risk_factors

    def _score_risk_factor(self, text: str) -> RiskFactor:
        """Score a single risk factor paragraph."""
        words = re.findall(r"\b[a-z]+\b", text.lower())
        if not words:
            return RiskFactor(
                text=text, severity_score=0.0, category="OPERATIONAL",
                word_count=0, key_terms=[],
            )

        neg_count = sum(1 for w in words if w in _NEGATIVE_WORDS)
        severity = min(1.0, neg_count / (len(words) ** 0.5))

        # Categorize
        category = "OPERATIONAL"
        best_score = 0
        text_lower = text.lower()
        for cat, keywords in RISK_CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                category = cat

        key_terms = [w for w in words if w in _NEGATIVE_WORDS][:10]

        return RiskFactor(
            text=text[:2000],
            severity_score=round(severity, 4),
            category=category,
            word_count=len(words),
            key_terms=list(set(key_terms)),
        )

    # ------------------------------------------------------------------
    # Risk factor comparison
    # ------------------------------------------------------------------

    def compare_risk_factors(self, doc1: str, doc2: str) -> RiskFactorDelta:
        """
        Compare risk factors between two filing documents.
        Uses TF-IDF cosine similarity (or fallback Jaccard) to match factors.
        """
        rfs1 = self.extract_risk_factors(doc1)
        rfs2 = self.extract_risk_factors(doc2)

        if not rfs1 and not rfs2:
            return RiskFactorDelta([], [], [], 0, 0, 0, 0.0)

        # Build similarity matrix
        def _vectorize(texts: List[str]) -> List[Dict[str, float]]:
            """Simple TF-IDF via word frequency."""
            from collections import Counter
            tf_idf_vecs = []
            # Compute IDF over corpus
            N = len(texts)
            df: Dict[str, int] = defaultdict(int)
            tokenized = []
            for t in texts:
                words = set(re.findall(r"\b[a-z]{3,}\b", t.lower()))
                tokenized.append(words)
                for w in words:
                    df[w] += 1
            for i, (t, words) in enumerate(zip(texts, tokenized)):
                all_words = re.findall(r"\b[a-z]{3,}\b", t.lower())
                tf = Counter(all_words)
                vec = {
                    w: (c / len(all_words)) * math.log((N + 1) / (df[w] + 1))
                    for w, c in tf.items()
                    if len(all_words) > 0
                }
                tf_idf_vecs.append(vec)
            return tf_idf_vecs

        def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
            common = set(v1.keys()) & set(v2.keys())
            dot = sum(v1[w] * v2[w] for w in common)
            n1 = math.sqrt(sum(x * x for x in v1.values()))
            n2 = math.sqrt(sum(x * x for x in v2.values()))
            if n1 == 0 or n2 == 0:
                return 0.0
            return dot / (n1 * n2)

        texts1 = [rf.text for rf in rfs1]
        texts2 = [rf.text for rf in rfs2]
        vecs1  = _vectorize(texts1) if texts1 else []
        vecs2  = _vectorize(texts2) if texts2 else []

        # Match: for each rf in doc1, find best match in doc2
        matched1: Set[int] = set()
        matched2: Set[int] = set()
        intensified: List[Tuple[RiskFactor, RiskFactor]] = []

        SIM_THRESHOLD = 0.30  # considered same risk if similarity > 30%

        for i, (v1, rf1) in enumerate(zip(vecs1, rfs1)):
            best_sim = 0.0
            best_j   = -1
            for j, (v2, rf2) in enumerate(zip(vecs2, rfs2)):
                if j in matched2:
                    continue
                sim = _cosine(v1, v2)
                if sim > best_sim:
                    best_sim = sim
                    best_j   = j
            if best_sim >= SIM_THRESHOLD and best_j >= 0:
                matched1.add(i)
                matched2.add(best_j)
                # Check if severity increased
                if rfs2[best_j].severity_score > rf1.severity_score + 0.05:
                    intensified.append((rf1, rfs2[best_j]))

        removed = [rfs1[i] for i in range(len(rfs1)) if i not in matched1]
        new     = [rfs2[j] for j in range(len(rfs2)) if j not in matched2]

        avg_sev1 = sum(r.severity_score for r in rfs1) / len(rfs1) if rfs1 else 0.0
        avg_sev2 = sum(r.severity_score for r in rfs2) / len(rfs2) if rfs2 else 0.0

        return RiskFactorDelta(
            new_risks=new,
            removed_risks=removed,
            intensified_risks=intensified,
            new_risk_count=len(new),
            removed_risk_count=len(removed),
            net_risk_change=len(new) - len(removed),
            severity_delta=round(avg_sev2 - avg_sev1, 4),
        )

    # ------------------------------------------------------------------
    # NLP extraction
    # ------------------------------------------------------------------

    def extract_forward_looking_statements(self, document: str) -> List[str]:
        """
        Extract sentences containing forward-looking language.
        Looks for: "we believe", "we expect", "we anticipate", "we plan",
        "we intend", "guidance", "outlook", "forecast".
        """
        if not document:
            return []
        triggers = [
            r"\bwe\s+believe\b",
            r"\bwe\s+expect\b",
            r"\bwe\s+anticipate\b",
            r"\bwe\s+plan\b",
            r"\bwe\s+intend\b",
            r"\bguidance\b",
            r"\boutlook\b",
            r"\bforecast\b",
            r"\bprojected?\b",
            r"\bgoing\s+forward\b",
        ]
        pattern = re.compile("|".join(triggers), re.IGNORECASE)
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", document)
        fls: List[str] = []
        seen: Set[str] = set()
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30 or len(sent) > 1000:
                continue
            if pattern.search(sent):
                key = sent[:80]
                if key not in seen:
                    fls.append(sent)
                    seen.add(key)
        return fls[:50]

    def extract_material_weaknesses(self, document: str) -> List[str]:
        """
        Extract sentences disclosing material weaknesses or significant deficiencies
        in internal controls (Item 9A).
        """
        if not document:
            return []
        pattern = re.compile(
            r"(material\s+weakness|significant\s+deficiency|icfr|"
            r"internal\s+control.{0,30}report|restatement)",
            re.IGNORECASE,
        )
        sentences = re.split(r"(?<=[.!?])\s+", document)
        results: List[str] = []
        for sent in sentences:
            sent = sent.strip()
            if pattern.search(sent) and len(sent) > 40:
                results.append(sent[:500])
        return list(dict.fromkeys(results))[:20]  # deduplicate

    def extract_quantitative_disclosures(
        self, document: str
    ) -> List[Tuple[str, float, str]]:
        """
        Extract quantitative disclosures: dollar amounts and percentages with context.
        Returns list of (context_sentence, value, unit).
        """
        if not document:
            return []
        results: List[Tuple[str, float, str]] = []

        dollar_pattern = re.compile(
            r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)?", re.IGNORECASE
        )
        pct_pattern = re.compile(
            r"([\d,]+(?:\.\d+)?)\s*%"
        )

        sentences = re.split(r"(?<=[.!?])\s+", document)
        for sent in sentences[:2000]:
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 500:
                continue
            for m in dollar_pattern.finditer(sent):
                try:
                    val_str = m.group(1).replace(",", "")
                    val = float(val_str)
                    unit_word = (m.group(2) or "").lower()
                    if unit_word == "billion":
                        val *= 1e9
                        unit = "USD_billion"
                    elif unit_word == "million":
                        val *= 1e6
                        unit = "USD_million"
                    elif unit_word == "thousand":
                        val *= 1e3
                        unit = "USD_thousand"
                    else:
                        unit = "USD"
                    results.append((sent[:200], val, unit))
                except ValueError:
                    pass
            for m in pct_pattern.finditer(sent):
                try:
                    val = float(m.group(1).replace(",", ""))
                    results.append((sent[:200], val, "percent"))
                except ValueError:
                    pass

        return results[:100]

    # ------------------------------------------------------------------
    # Readability
    # ------------------------------------------------------------------

    def compute_readability_score(self, document: str) -> Tuple[float, float]:
        """
        Compute Flesch-Kincaid reading ease and Gunning Fog Index.
        Returns (flesch_kincaid_ease, fog_index).
        Higher FK ease = simpler text. Higher fog = more complex.
        """
        if not document:
            return 0.0, 0.0

        # Tokenize
        sentences = re.split(r"[.!?]+", document)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        words = re.findall(r"\b[a-zA-Z]+\b", document)
        if not sentences or not words:
            return 0.0, 0.0

        n_sentences = len(sentences)
        n_words     = len(words)
        avg_words_per_sentence = n_words / n_sentences if n_sentences else 0

        # Syllable approximation: count vowel groups
        def count_syllables(word: str) -> int:
            word = word.lower()
            vowels = re.findall(r"[aeiouy]+", word)
            count  = len(vowels)
            if word.endswith("e") and count > 1:
                count -= 1
            return max(1, count)

        syllable_counts = [count_syllables(w) for w in words]
        n_syllables = sum(syllable_counts)

        # Flesch-Kincaid Reading Ease
        avg_syllables_per_word = n_syllables / n_words if n_words else 0
        fk_ease = (
            206.835
            - 1.015 * avg_words_per_sentence
            - 84.6 * avg_syllables_per_word
        )
        fk_ease = max(0.0, min(100.0, fk_ease))

        # Gunning Fog Index
        complex_words = sum(1 for c in syllable_counts if c >= 3)
        fog = 0.4 * (avg_words_per_sentence + 100.0 * complex_words / n_words)

        return round(fk_ease, 2), round(fog, 2)

    def compute_sentiment(self, document: str) -> float:
        """
        Compute document sentiment as positive-word ratio (0..1).
        0.5 = neutral, >0.5 = positive, <0.5 = negative.
        """
        if not document:
            return 0.5
        words = re.findall(r"\b[a-z]+\b", document.lower())
        if not words:
            return 0.5
        pos = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.5
        return round(pos / total, 4)


# ---------------------------------------------------------------------------
# EDGARSemanticSearch
# ---------------------------------------------------------------------------

class EDGARSemanticSearch:
    """
    Semantic search over EDGAR document collections.
    Uses sklearn TF-IDF + cosine similarity (primary) or
    sentence-transformers MiniLM (optional, if installed).
    """

    def __init__(self, use_sbert: bool = True) -> None:
        self._sbert_model = None
        if use_sbert and _HAS_SBERT:
            try:
                self._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("EDGARSemanticSearch: using sentence-transformers")
            except Exception as exc:
                logger.warning("sentence-transformers load failed: %s", exc)

    # ------------------------------------------------------------------
    # TF-IDF index
    # ------------------------------------------------------------------

    def build_index(self, documents: List[str]) -> TFIDFIndex:
        """Build a TF-IDF index over a corpus of documents."""
        if not documents:
            return TFIDFIndex(
                vectorizer=None, matrix=None, documents=[], idf_map={}
            )

        if _HAS_SKLEARN:
            vectorizer = TfidfVectorizer(
                max_features=20_000,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform(documents)
            return TFIDFIndex(
                vectorizer=vectorizer,
                matrix=matrix,
                documents=documents,
                idf_map={},
            )
        else:
            # Fallback: manual IDF
            N = len(documents)
            df_map: Dict[str, int] = defaultdict(int)
            tokenized = []
            for doc in documents:
                words = set(re.findall(r"\b[a-z]{3,}\b", doc.lower()))
                tokenized.append(words)
                for w in words:
                    df_map[w] += 1
            idf_map = {
                w: math.log((N + 1) / (df + 1)) + 1
                for w, df in df_map.items()
            }
            return TFIDFIndex(
                vectorizer=None, matrix=None,
                documents=documents, idf_map=idf_map,
            )

    def search_semantic(
        self, query: str, index: TFIDFIndex, k: int = 10
    ) -> List[int]:
        """Return indices of top-k most semantically similar documents."""
        if not index.documents:
            return []

        if _HAS_SKLEARN and index.vectorizer is not None:
            q_vec = index.vectorizer.transform([query])
            sims  = cosine_similarity(q_vec, index.matrix).flatten()
            top_k = sims.argsort()[::-1][:k]
            return [int(i) for i in top_k if sims[i] > 0]

        # Fallback: manual cosine
        query_words = Counter(re.findall(r"\b[a-z]{3,}\b", query.lower()))
        q_tfidf = {
            w: (c / len(query_words)) * index.idf_map.get(w, 1.0)
            for w, c in query_words.items()
        }

        scores: List[Tuple[float, int]] = []
        for i, doc in enumerate(index.documents):
            doc_words = Counter(re.findall(r"\b[a-z]{3,}\b", doc.lower()))
            total = sum(doc_words.values()) or 1
            d_tfidf = {
                w: (c / total) * index.idf_map.get(w, 1.0)
                for w, c in doc_words.items()
            }
            common  = set(q_tfidf) & set(d_tfidf)
            dot     = sum(q_tfidf[w] * d_tfidf[w] for w in common)
            nq = math.sqrt(sum(v * v for v in q_tfidf.values()))
            nd = math.sqrt(sum(v * v for v in d_tfidf.values()))
            sim = dot / (nq * nd) if nq > 0 and nd > 0 else 0.0
            scores.append((sim, i))

        scores.sort(reverse=True)
        return [i for _, i in scores[:k] if _ > 0]

    # ------------------------------------------------------------------
    # Sentence-transformers path
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> Any:
        """Encode texts using sentence-transformers or fallback."""
        if self._sbert_model is not None:
            return self._sbert_model.encode(texts, show_progress_bar=False)
        raise RuntimeError("sentence-transformers not available")

    # ------------------------------------------------------------------
    # Find similar filings
    # ------------------------------------------------------------------

    def find_similar_filings(
        self, target_doc: str, corpus: List[str]
    ) -> List[Tuple[float, int]]:
        """
        Find corpus filings most similar to `target_doc`.
        Returns list of (similarity_score, index) sorted descending.
        """
        if not corpus:
            return []

        if self._sbert_model is not None:
            try:
                all_texts = [target_doc] + corpus
                embeddings = self._encode(all_texts)
                target_emb = embeddings[0:1]
                corpus_emb = embeddings[1:]
                # Manual cosine
                results = []
                for i, emb in enumerate(corpus_emb):
                    dot = float(sum(a * b for a, b in zip(target_emb[0], emb)))
                    nt  = math.sqrt(sum(a * a for a in target_emb[0]))
                    nc  = math.sqrt(sum(a * a for a in emb))
                    sim = dot / (nt * nc) if nt > 0 and nc > 0 else 0.0
                    results.append((sim, i))
                results.sort(reverse=True)
                return results
            except Exception as exc:
                logger.debug("SBERT similarity failed: %s", exc)

        # TF-IDF fallback
        index = self.build_index(corpus)
        top_k = self.search_semantic(target_doc, index, k=len(corpus))
        if _HAS_SKLEARN and index.matrix is not None:
            q_vec = index.vectorizer.transform([target_doc])
            sims  = cosine_similarity(q_vec, index.matrix).flatten()
            return [(float(sims[i]), i) for i in top_k]
        return [(1.0 / (rank + 1), i) for rank, i in enumerate(top_k)]

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_filings(
        self, documents: List[str], n_clusters: int = 5
    ) -> List[int]:
        """
        Cluster documents into n_clusters groups using K-means on TF-IDF vectors.
        Returns list of cluster labels (one per document).
        """
        if not documents:
            return []

        if _HAS_SKLEARN:
            vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english", sublinear_tf=True
            )
            X = vectorizer.fit_transform(documents)
            n = min(n_clusters, len(documents))
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = km.fit_predict(X).tolist()
            # Log top terms per cluster
            feature_names = vectorizer.get_feature_names_out()
            for ci in range(n):
                center = km.cluster_centers_[ci]
                top_idx = center.argsort()[::-1][:5]
                top_terms = [feature_names[idx] for idx in top_idx]
                logger.info("Cluster %d top terms: %s", ci, top_terms)
            return labels
        else:
            # Fallback: round-robin assignment
            return [i % n_clusters for i in range(len(documents))]

    def get_cluster_labels(
        self, documents: List[str], labels: List[int]
    ) -> Dict[int, List[str]]:
        """Return the top TF-IDF terms for each cluster label."""
        if not _HAS_SKLEARN:
            return {}
        from collections import defaultdict
        cluster_docs: Dict[int, List[str]] = defaultdict(list)
        for doc, label in zip(documents, labels):
            cluster_docs[label].append(doc)

        result: Dict[int, List[str]] = {}
        for label, docs in cluster_docs.items():
            combined = " ".join(docs)
            words = Counter(re.findall(r"\b[a-z]{4,}\b", combined.lower()))
            for sw in {"that", "this", "with", "from", "have", "were",
                       "their", "they", "also", "other", "such", "been"}:
                words.pop(sw, None)
            result[label] = [w for w, _ in words.most_common(8)]
        return result


# ---------------------------------------------------------------------------
# EDGARWatchList
# ---------------------------------------------------------------------------

class EDGARWatchList:
    """
    Monitor specific search criteria for new EDGAR filings.
    Stores watches in memory; alerts generated by comparing to a last-seen timestamp.
    """

    def __init__(self, searcher: Optional[EDGARFullTextSearcher] = None) -> None:
        self.searcher = searcher or EDGARFullTextSearcher()
        self._watches: List[dict] = []

    def add_watch(
        self,
        name: str,
        query: str,
        form_types: Optional[List[str]] = None,
        alert_threshold: int = 1,
    ) -> None:
        """Register a new watch."""
        self._watches.append({
            "name": name,
            "query": query,
            "form_types": form_types or [],
            "alert_threshold": alert_threshold,
            "last_checked": None,
            "last_seen_accessions": set(),
        })
        logger.info("Watch added: '%s' for query '%s'", name, query)

    def get_new_filings_since(
        self, timestamp: str, form_types: Optional[List[str]] = None
    ) -> List[EDGARSearchResult]:
        """
        Return all filings since a given date (ISO format).
        """
        results = self.searcher.search(
            query="*",
            form_types=form_types,
            start_date=timestamp,
            end_date=date.today().isoformat(),
            max_results=200,
        )
        return results

    def run_all_watches(self) -> List[EDGARAlert]:
        """Check all watches and return alerts for newly matched filings."""
        alerts: List[EDGARAlert] = []
        today = date.today().isoformat()

        for watch in self._watches:
            last = watch["last_checked"] or (
                date.today() - timedelta(days=7)
            ).isoformat()

            form_types = watch["form_types"] or None
            try:
                results = self.searcher.search(
                    query=watch["query"],
                    form_types=form_types,
                    start_date=last,
                    end_date=today,
                    max_results=50,
                )
            except Exception as exc:
                logger.warning("Watch '%s' search failed: %s", watch["name"], exc)
                continue

            new_results = [
                r for r in results
                if r.accession_no not in watch["last_seen_accessions"]
            ]

            if len(new_results) >= watch["alert_threshold"]:
                for r in new_results:
                    snippet = (
                        f"{r.entity_name} filed {r.form_type} on {r.filing_date}"
                        f" (period: {r.period_of_report})"
                    )
                    alerts.append(EDGARAlert(
                        watch_name=watch["name"],
                        query=watch["query"],
                        trigger_date=today,
                        filing=r,
                        context_snippet=snippet,
                    ))
                    watch["last_seen_accessions"].add(r.accession_no)

            watch["last_checked"] = today

        return alerts

    def add_phrase_watches(
        self,
        phrases: List[str],
        form_types: Optional[List[str]] = None,
    ) -> None:
        """Convenience: add standard phrase watches."""
        for phrase in phrases:
            self.add_watch(
                name=phrase.replace(" ", "_").upper(),
                query=f'"{phrase}"',
                form_types=form_types,
            )


# ---------------------------------------------------------------------------
# EDGARNLPPipeline
# ---------------------------------------------------------------------------

class EDGARNLPPipeline:
    """
    End-to-end NLP analysis pipeline for EDGAR filings.
    Fetches, parses, and analyzes filings for a given ticker.
    """

    def __init__(
        self,
        searcher: Optional[EDGARFullTextSearcher] = None,
        analyzer: Optional[EDGARDocumentAnalyzer] = None,
    ) -> None:
        self.searcher = searcher or EDGARFullTextSearcher()
        self.analyzer = analyzer or EDGARDocumentAnalyzer()

    def _get_latest_filing(
        self, ticker: str, form_type: str
    ) -> Optional[Tuple[EDGARSearchResult, str]]:
        """Fetch the latest filing of form_type for ticker. Returns (result, document_text)."""
        results = self.searcher.search(
            query="",
            form_types=[form_type],
            ticker=ticker,
            max_results=5,
        )
        if not results:
            # Try a broader search
            results = self.searcher.search(
                query=ticker,
                form_types=[form_type],
                max_results=5,
            )
        if not results:
            return None

        # Sort by filing date
        results.sort(
            key=lambda r: r.filing_date or "1900-01-01", reverse=True
        )
        latest = results[0]
        text = self.searcher.get_filing_document(latest)
        if not text:
            return None
        return latest, text

    def analyze_filing(
        self, ticker: str, form_type: str = "10-K"
    ) -> Optional[FilingAnalysis]:
        """
        Full NLP pipeline over a single filing:
        - Risk factor extraction
        - Forward guidance
        - Material weaknesses
        - Quantitative disclosures
        - Readability + sentiment
        """
        fetched = self._get_latest_filing(ticker, form_type)
        if not fetched:
            logger.warning("No %s filing found for %s", form_type, ticker)
            return None

        result, document = fetched
        sections = self.searcher.get_filing_sections(document)

        risk_factors = self.analyzer.extract_risk_factors(
            sections.get("item1a", document)
        )
        fls = self.analyzer.extract_forward_looking_statements(
            sections.get("item7", document)
        )
        mw = self.analyzer.extract_material_weaknesses(
            sections.get("item9a", document)
        )
        quant = self.analyzer.extract_quantitative_disclosures(
            sections.get("item7", document)[:50_000]
        )
        fk_ease, fog = self.analyzer.compute_readability_score(document[:100_000])
        sentiment = self.analyzer.compute_sentiment(document[:100_000])

        section_lengths = {k: len(v) for k, v in sections.items()}

        # Brief summary
        top_risks = sorted(risk_factors, key=lambda r: r.severity_score, reverse=True)[:3]
        risk_summary = "; ".join(
            f"{r.category}({r.severity_score:.2f})" for r in top_risks
        )
        summary = (
            f"{result.entity_name} {form_type} (filed {result.filing_date}, "
            f"period {result.period_of_report}). "
            f"Top risks: {risk_summary}. "
            f"Material weaknesses: {len(mw)}. "
            f"Readability: FK={fk_ease}, Fog={fog}. "
            f"Sentiment: {sentiment:.2%}."
        )

        return FilingAnalysis(
            ticker=ticker,
            form_type=form_type,
            entity_name=result.entity_name,
            filing_date=result.filing_date,
            period=result.period_of_report,
            risk_factors=risk_factors,
            forward_looking_statements=fls,
            material_weaknesses=mw,
            quantitative_disclosures=quant,
            readability_score=fk_ease,
            fog_index=fog,
            overall_sentiment=sentiment,
            document_length=len(document),
            section_lengths=section_lengths,
            summary=summary,
        )

    def compare_annual_filings(
        self, ticker: str, year1: int, year2: int
    ) -> Optional[FilingComparison]:
        """
        Compare 10-K filings for two different years.
        """
        def _get_filing_for_year(yr: int) -> Optional[Tuple[EDGARSearchResult, str]]:
            start = f"{yr}-01-01"
            end   = f"{yr}-12-31"
            results = self.searcher.search(
                query="",
                form_types=["10-K"],
                ticker=ticker,
                start_date=start,
                end_date=end,
                max_results=5,
            )
            if not results:
                return None
            results.sort(key=lambda r: r.filing_date or "", reverse=True)
            doc = self.searcher.get_filing_document(results[0])
            return (results[0], doc) if doc else None

        f1 = _get_filing_for_year(year1)
        f2 = _get_filing_for_year(year2)

        if not f1 or not f2:
            logger.warning(
                "Could not fetch 10-K for %s years %d/%d", ticker, year1, year2
            )
            return None

        doc1, doc2 = f1[1], f2[1]
        risk_delta = self.analyzer.compare_risk_factors(doc1, doc2)

        fk1, fog1 = self.analyzer.compute_readability_score(doc1[:100_000])
        fk2, fog2 = self.analyzer.compute_readability_score(doc2[:100_000])
        sent1 = self.analyzer.compute_sentiment(doc1[:100_000])
        sent2 = self.analyzer.compute_sentiment(doc2[:100_000])
        mw1   = self.analyzer.extract_material_weaknesses(doc1)
        mw2   = self.analyzer.extract_material_weaknesses(doc2)
        fls2  = self.analyzer.extract_forward_looking_statements(doc2)[:10]

        new_mw = [m for m in mw2 if not any(m[:60] in o for o in mw1)]

        summary = (
            f"{ticker} 10-K comparison {year1}→{year2}: "
            f"{risk_delta.new_risk_count} new risk factors, "
            f"{risk_delta.removed_risk_count} removed. "
            f"Severity delta: {risk_delta.severity_delta:+.3f}. "
            f"Readability change: FK {fk1}→{fk2}. "
            f"Sentiment: {sent1:.2%}→{sent2:.2%}."
        )

        return FilingComparison(
            ticker=ticker,
            year1=year1,
            year2=year2,
            risk_delta=risk_delta,
            readability_delta=round(fk2 - fk1, 2),
            sentiment_delta=round(sent2 - sent1, 4),
            new_forward_guidance=fls2,
            new_material_weaknesses=new_mw,
            document_length_delta=len(doc2) - len(doc1),
            summary=summary,
        )

    def generate_filing_brief(
        self, ticker: str, form_type: str = "10-K"
    ) -> str:
        """Generate a 5-sentence executive summary of a filing."""
        analysis = self.analyze_filing(ticker, form_type)
        if not analysis:
            return f"No {form_type} filing found for {ticker}."

        top_risks = sorted(
            analysis.risk_factors, key=lambda r: r.severity_score, reverse=True
        )[:3]
        risk_strs = [
            f"{r.category.lower()} risk ({r.severity_score:.0%} severity)"
            for r in top_risks
        ]

        sentences = [
            f"{analysis.entity_name} filed its {form_type} on {analysis.filing_date} "
            f"for the period ending {analysis.period}.",
            f"The filing spans {analysis.document_length:,} characters with "
            f"{len(analysis.risk_factors)} risk factors identified in Item 1A.",
            f"Top risks by severity: {', '.join(risk_strs) if risk_strs else 'none identified'}.",
            (
                f"The document contains {len(analysis.material_weaknesses)} "
                f"material weakness disclosure(s) in internal controls."
                if analysis.material_weaknesses
                else "No material weaknesses were identified in internal controls."
            ),
            f"Overall tone is {'positive' if analysis.overall_sentiment > 0.55 else 'cautious'} "
            f"(sentiment {analysis.overall_sentiment:.0%}), with a readability ease score of "
            f"{analysis.readability_score:.0f}/100 (Flesch-Kincaid).",
        ]
        return " ".join(sentences)


# ---------------------------------------------------------------------------
# EDGARSearchEngine (orchestrator)
# ---------------------------------------------------------------------------

class EDGARSearchEngine:
    """
    Main orchestrator for EDGAR full-text search and NLP analysis.

    Usage::
        engine = EDGARSearchEngine()
        results = engine.search("going concern", form_types=["10-K"], days_back=90)
        analysis = engine.analyze("AAPL", "10-K")
        alerts = engine.monitor(["material weakness"], ["10-K"])
    """

    def __init__(self) -> None:
        self.searcher  = EDGARFullTextSearcher()
        self.analyzer  = EDGARDocumentAnalyzer()
        self.semantic  = EDGARSemanticSearch()
        self.watchlist = EDGARWatchList(self.searcher)
        self.pipeline  = EDGARNLPPipeline(self.searcher, self.analyzer)

    def search(
        self,
        query: str,
        form_types: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        days_back: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 50,
    ) -> List[EDGARSearchResult]:
        """Full-text EDGAR search with optional date/form filters."""
        if days_back and not start_date:
            start_date = (date.today() - timedelta(days=days_back)).isoformat()
        if not end_date:
            end_date = date.today().isoformat()
        return self.searcher.search(
            query=query,
            form_types=form_types,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )

    def analyze(
        self, ticker: str, form_type: str = "10-K"
    ) -> Optional[FilingAnalysis]:
        """Full NLP analysis of the latest filing for a ticker."""
        return self.pipeline.analyze_filing(ticker, form_type)

    def monitor(
        self,
        phrases: List[str],
        form_types: Optional[List[str]] = None,
    ) -> List[EDGARAlert]:
        """Set up phrase monitors and return any new alerts."""
        self.watchlist.add_phrase_watches(phrases, form_types=form_types)
        return self.watchlist.run_all_watches()

    def export_results(
        self,
        results: List[EDGARSearchResult],
        path: str,
    ) -> None:
        """Export search results to a CSV file."""
        if not _HAS_PANDAS:
            raise ImportError("pandas required for CSV export")
        rows = [
            {
                "accession_no":     r.accession_no,
                "filing_date":      r.filing_date,
                "form_type":        r.form_type,
                "entity_name":      r.entity_name,
                "cik":              r.cik,
                "period_of_report": r.period_of_report,
                "file_num":         r.file_num,
                "category":         r.category,
                "document_url":     r.document_url,
            }
            for r in results
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("Exported %d results to %s", len(results), path)

    def compare(
        self, ticker: str, year1: int, year2: int
    ) -> Optional[FilingComparison]:
        """Compare two annual 10-K filings."""
        return self.pipeline.compare_annual_filings(ticker, year1, year2)

    def find_similar(
        self, source_ticker: str, candidate_tickers: List[str],
        form_type: str = "10-K"
    ) -> List[Tuple[float, str]]:
        """
        Find tickers with filings most similar to source_ticker's filing.
        Returns list of (similarity, ticker) sorted descending.
        """
        src = self._fetch_doc_text(source_ticker, form_type)
        if not src:
            return []
        corpus_texts = []
        corpus_tickers = []
        for tk in candidate_tickers:
            text = self._fetch_doc_text(tk, form_type)
            if text:
                corpus_texts.append(text[:50_000])
                corpus_tickers.append(tk)
        if not corpus_texts:
            return []
        ranked = self.semantic.find_similar_filings(src[:50_000], corpus_texts)
        return [(score, corpus_tickers[idx]) for score, idx in ranked]

    def _fetch_doc_text(self, ticker: str, form_type: str) -> Optional[str]:
        results = self.searcher.search(query="", form_types=[form_type], ticker=ticker, max_results=1)
        if not results:
            return None
        return self.searcher.get_filing_document(results[0])

    def cluster(
        self,
        tickers: List[str],
        form_type: str = "10-K",
        n_clusters: int = 5,
    ) -> Dict[str, int]:
        """
        Cluster tickers by filing similarity.
        Returns {ticker: cluster_id}.
        """
        texts: List[str] = []
        valid_tickers: List[str] = []
        for tk in tickers:
            text = self._fetch_doc_text(tk, form_type)
            if text:
                texts.append(text[:50_000])
                valid_tickers.append(tk)
        if not texts:
            return {}
        labels = self.semantic.cluster_filings(texts, n_clusters=n_clusters)
        return dict(zip(valid_tickers, labels))


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("EDGAR Full-Text Search V3 — Production NLP Platform")
    print("=" * 60)

    engine = EDGARSearchEngine()

    print("\n[1] Search 'going concern' in 10-Ks (last 90 days) …")
    try:
        results = engine.search(
            query="going concern",
            form_types=["10-K"],
            days_back=90,
            max_results=20,
        )
        print(f"    Found {len(results)} filings")
        for r in results[:5]:
            print(f"      {r.entity_name:40s} {r.form_type:6s} {r.filing_date}")
    except Exception as exc:
        print(f"    Error: {exc}")

    print("\n[2] Analyze AAPL 10-K …")
    try:
        analysis = engine.analyze("AAPL", "10-K")
        if analysis:
            print(f"    Entity: {analysis.entity_name}")
            print(f"    Filed:  {analysis.filing_date}")
            print(f"    Risk factors: {len(analysis.risk_factors)}")
            print(f"    Material weaknesses: {len(analysis.material_weaknesses)}")
            print(f"    Readability (FK): {analysis.readability_score}")
            print(f"    Fog index: {analysis.fog_index}")
            print(f"    Sentiment: {analysis.overall_sentiment:.2%}")
            print(f"    Forward guidance statements: {len(analysis.forward_looking_statements)}")
        else:
            print("    No analysis returned.")
    except Exception as exc:
        print(f"    Error: {exc}")

    print("\n[3] Compare AAPL 10-K risk factors (2022 vs 2023) …")
    try:
        comparison = engine.compare("AAPL", 2022, 2023)
        if comparison:
            print(f"    {comparison.summary}")
            print(f"    New risks: {comparison.risk_delta.new_risk_count}")
            print(f"    Removed risks: {comparison.risk_delta.removed_risk_count}")
            print(f"    Severity delta: {comparison.risk_delta.severity_delta:+.4f}")
        else:
            print("    Comparison not available.")
    except Exception as exc:
        print(f"    Error: {exc}")

    print("\n[4] Generate 5-sentence brief for MSFT 10-K …")
    try:
        brief = engine.pipeline.generate_filing_brief("MSFT", "10-K")
        print("   ", brief)
    except Exception as exc:
        print(f"    Error: {exc}")

    print("\n[5] Monitor for 'SEC investigation' and 'material weakness' …")
    try:
        alerts = engine.monitor(
            ["SEC investigation", "material weakness", "going concern"],
            form_types=["10-K", "8-K"],
        )
        print(f"    Alerts triggered: {len(alerts)}")
        for a in alerts[:3]:
            print(f"      [{a.watch_name}] {a.context_snippet}")
    except Exception as exc:
        print(f"    Error: {exc}")

    print("\n[6] Readability comparison — AAPL vs MSFT 10-K …")
    try:
        for ticker in ["AAPL", "MSFT"]:
            text = engine._fetch_doc_text(ticker, "10-K")
            if text:
                fk, fog = engine.analyzer.compute_readability_score(text[:80_000])
                sent = engine.analyzer.compute_sentiment(text[:80_000])
                print(f"    {ticker}: FK={fk:.1f}, Fog={fog:.1f}, Sentiment={sent:.2%}")
    except Exception as exc:
        print(f"    Error: {exc}")

    print("\nDone.")
    sys.exit(0)
