"""
Document Summarizer V3 — Comprehensive financial document summarization system.

dim_055: LLM document summarization — score 6 → 9

Architecture:
  ClaudeAPIClient            — Anthropic SDK wrapper with prompt caching, retry, fallback
  ExtractiveSummarizer       — TF-IDF sentence scoring, NER, number extraction (no LLM)
  SECFilingSummarizer        — 10-K / 10-Q / 8-K / DEF 14A section extraction + summarization
  EarningsCallSummarizer     — Transcript fetch (SEC 8-K / Motley Fool), management tone,
                               guidance extraction, surprise language detection
  NewsArticleSummarizer      — GDELT API multi-doc summary, sentiment timeline, catalyst detect
  PortfolioNarrativeGenerator — Claude-generated portfolio/risk/trade narratives
  DocumentSummarizationEngine — Orchestrator: full ticker package, batch, Q&A over docs

Data sources (all free):
  - EDGAR EFTS search: https://efts.sec.gov/LATEST/search-index?...
  - EDGAR submissions API: https://data.sec.gov/submissions/CIK{cik}.json
  - EDGAR filing text: https://www.sec.gov/Archives/edgar/...
  - GDELT: https://api.gdeltproject.org/api/v2/doc/doc?...
  - Motley Fool transcripts: https://www.fool.com/earnings-call-transcripts/

Usage::
    engine = DocumentSummarizationEngine()
    summary = engine.summarize_ticker("AAPL", depth="brief")
    print(summary.executive_brief)
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import string
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin

import numpy as np
import pandas as pd
import requests

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Optional: Anthropic SDK
# ---------------------------------------------------------------------------
try:
    from anthropic import Anthropic, APIStatusError, APIConnectionError, RateLimitError
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None  # type: ignore[assignment,misc]
    APIStatusError = Exception  # type: ignore[assignment,misc,misc]
    APIConnectionError = Exception  # type: ignore[assignment,misc]
    RateLimitError = Exception  # type: ignore[assignment,misc]
    _ANTHROPIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EDGAR_BASE        = "https://data.sec.gov"
EDGAR_ARCHIVES    = "https://www.sec.gov/Archives/edgar/full-index"
EFTS_SEARCH       = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

GDELT_DOC_API = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query={query}&mode=ArtList&maxrecords={n}&format=json&timespan={days}d"
)

MOTLEY_FOOL_TRANSCRIPTS = "https://www.fool.com/earnings-call-transcripts/?page={page}"

_EDGAR_HEADERS = {
    "User-Agent": "SENTINEL/3.0 research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html",
}

_REQUEST_TIMEOUT = 30
_MAX_CHUNK_CHARS = 12_000    # ~4k tokens for Haiku
_MAX_CONTEXT_CHARS = 80_000  # ~26k tokens

HAIKU_MODEL  = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FilingSummary:
    ticker: str
    filing_type: str       # 10-K, 10-Q, 8-K, DEF 14A
    period: str            # e.g. "2024", "Q3 2024"
    accession_number: str
    filed_date: str
    executive_summary: str = ""
    key_metrics: Dict[str, str] = field(default_factory=dict)
    risks: List[str] = field(default_factory=list)
    outlook: str = ""
    full_text_length: int = 0
    model_used: str = ""


@dataclass
class RiskSummary:
    ticker: str
    period: str
    top_risks: List[Dict[str, Any]] = field(default_factory=list)   # [{text, score, category}]
    risk_categories: Dict[str, int] = field(default_factory=dict)
    total_risk_factors: int = 0
    emphasis_score: float = 0.0   # mention density
    narrative: str = ""


@dataclass
class MDASummary:
    ticker: str
    period: str
    revenue_drivers: List[str] = field(default_factory=list)
    margin_trends: str = ""
    forward_outlook: str = ""
    management_tone: str = "neutral"   # positive / neutral / negative
    key_numbers: List[Tuple[str, float, str]] = field(default_factory=list)  # (label, val, unit)
    narrative: str = ""


@dataclass
class EarningsSummary:
    ticker: str
    quarter: str
    management_tone: str = "neutral"
    revenue_guidance: str = ""
    eps_guidance: str = ""
    top_concerns: List[str] = field(default_factory=list)
    bull_points: List[str] = field(default_factory=list)
    bear_points: List[str] = field(default_factory=list)
    surprise_language: List[str] = field(default_factory=list)
    narrative: str = ""


@dataclass
class GuidanceData:
    revenue_guide: str = ""
    eps_guide: str = ""
    revenue_low: Optional[float] = None
    revenue_high: Optional[float] = None
    eps_low: Optional[float] = None
    eps_high: Optional[float] = None
    confidence: float = 0.5
    fiscal_year: str = ""


@dataclass
class NewsArticle:
    url: str
    title: str
    source: str
    published: str
    seendate: str = ""
    language: str = "English"
    domain: str = ""
    snippet: str = ""
    sentiment: float = 0.0   # -1 to 1


@dataclass
class Catalyst:
    ticker: str
    event_type: str      # earnings, M&A, regulatory, macro, product
    date: str
    headline: str
    impact: str          # bullish / bearish / neutral
    magnitude: float     # 0-1
    source_url: str = ""


@dataclass
class RiskDelta:
    ticker: str
    year1: int
    year2: int
    new_risks: List[str] = field(default_factory=list)
    removed_risks: List[str] = field(default_factory=list)
    intensified_risks: List[str] = field(default_factory=list)
    narrative: str = ""


@dataclass
class ComprehensiveSummary:
    ticker: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    executive_brief: str = ""
    filing_summary: Optional[FilingSummary] = None
    earnings_summary: Optional[EarningsSummary] = None
    news_summary: str = ""
    risk_summary: Optional[RiskSummary] = None
    mda_summary: Optional[MDASummary] = None
    catalysts: List[Catalyst] = field(default_factory=list)
    sentiment_trend: str = "neutral"
    depth: str = "brief"


# ---------------------------------------------------------------------------
# ClaudeAPIClient
# ---------------------------------------------------------------------------

class ClaudeAPIClient:
    """
    Anthropic Claude API wrapper with prompt caching, retry logic,
    and graceful fallback to extractive summarization if SDK unavailable.
    """

    SYSTEM_PROMPT_FINANCE = (
        "You are SENTINEL, an institutional financial analyst AI. "
        "Provide precise, data-driven analysis. Use exact numbers where available. "
        "Flag uncertainty explicitly. Format outputs as requested — concise prose "
        "unless structured JSON is specified. Do not hallucinate data."
    )

    def __init__(self,
                 model_fast: str = HAIKU_MODEL,
                 model_deep: str = SONNET_MODEL,
                 max_retries: int = 4,
                 api_key: Optional[str] = None):
        self.model_fast = model_fast
        self.model_deep = model_deep
        self.max_retries = max_retries
        self._fallback = ExtractiveSummarizer()

        if _ANTHROPIC_AVAILABLE:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if key:
                self._client = Anthropic(api_key=key)
                self._available = True
            else:
                logger.warning("ANTHROPIC_API_KEY not set — using extractive fallback")
                self._client = None
                self._available = False
        else:
            self._client = None
            self._available = False

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def _call(self, messages: List[Dict], model: str,
              system: str, max_tokens: int = 1024) -> str:
        """Make API call with exponential backoff retry."""
        if not self._available:
            return ""

        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=[{
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=messages,
                )
                return response.content[0].text
            except RateLimitError:
                wait = 2 ** attempt + 1
                logger.warning("Rate limit hit; waiting %ds (attempt %d/%d)",
                               wait, attempt + 1, self.max_retries)
                time.sleep(wait)
            except APIConnectionError as exc:
                logger.warning("Connection error: %s (attempt %d)", exc, attempt + 1)
                time.sleep(2 ** attempt)
            except APIStatusError as exc:
                logger.error("API status error %s: %s", exc.status_code, exc.message)
                if exc.status_code in (400, 401, 403):
                    break  # don't retry auth/format errors
                time.sleep(2 ** attempt)
            except Exception as exc:
                logger.error("Unexpected API error: %s", exc)
                break
        return ""

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def summarize(self, text: str, context: str = "",
                  mode: str = "brief") -> str:
        """
        Summarize text.
        mode: "brief" (3-5 sentences), "detailed" (paragraph), "bullets" (bullet list)
        """
        if not self._available:
            return self._fallback.summarize(text, n_sentences=5)

        # Truncate if needed
        text = text[:_MAX_CHUNK_CHARS]

        mode_instructions = {
            "brief":    "Summarize in 3-5 concise sentences.",
            "detailed": "Write a thorough 2-3 paragraph analysis.",
            "bullets":  "List 5-8 key points as bullet points.",
        }.get(mode, "Summarize in 3-5 sentences.")

        context_str = f"\nContext: {context}\n" if context else ""
        prompt = f"{context_str}\n{mode_instructions}\n\nText:\n{text}"

        result = self._call(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_fast if mode == "brief" else self.model_deep,
            system=self.SYSTEM_PROMPT_FINANCE,
            max_tokens=512 if mode == "brief" else 1200,
        )
        return result or self._fallback.summarize(text, n_sentences=5)

    def extract_structured(self, text: str, schema: Dict) -> Dict:
        """Extract structured data from text per provided JSON schema."""
        if not self._available:
            return {}

        text = text[:_MAX_CHUNK_CHARS]
        schema_str = json.dumps(schema, indent=2)
        prompt = (
            f"Extract the following structured data from the text below. "
            f"Return ONLY valid JSON matching this schema:\n{schema_str}\n\n"
            f"Text:\n{text}"
        )
        result = self._call(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_fast,
            system=self.SYSTEM_PROMPT_FINANCE,
            max_tokens=1024,
        )
        try:
            # Extract JSON from response (may have surrounding text)
            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {}

    def compare_documents(self, doc1: str, doc2: str, focus: str) -> str:
        """Compare two documents, focusing on the given aspect."""
        if not self._available:
            return "Document comparison requires Claude API."

        # Truncate each document
        d1 = doc1[:_MAX_CHUNK_CHARS // 2]
        d2 = doc2[:_MAX_CHUNK_CHARS // 2]
        prompt = (
            f"Compare these two documents focusing on: {focus}\n\n"
            f"Document 1:\n{d1}\n\nDocument 2:\n{d2}\n\n"
            f"Highlight: (1) What's new in Doc2 vs Doc1, (2) What changed, "
            f"(3) What was removed. Be concise and specific."
        )
        return self._call(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_deep,
            system=self.SYSTEM_PROMPT_FINANCE,
            max_tokens=1000,
        ) or "Comparison unavailable."

    def answer_question(self, context: str, question: str) -> str:
        """Q&A over a document context."""
        if not self._available:
            return "Q&A requires Claude API."

        context = context[:_MAX_CONTEXT_CHARS]
        prompt = (
            f"Answer the following question using ONLY the provided context. "
            f"If the answer is not in the context, say 'Not found in document.'\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        return self._call(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_fast,
            system=self.SYSTEM_PROMPT_FINANCE,
            max_tokens=512,
        ) or "Answer unavailable."

    def map_reduce_summarize(self, text: str, chunk_size: int = _MAX_CHUNK_CHARS,
                              final_mode: str = "detailed") -> str:
        """
        Map-reduce summarization for very long documents.
        1. Split into chunks → summarize each (map)
        2. Concatenate chunk summaries → final summary (reduce)
        """
        if not self._available:
            return self._fallback.summarize(text, n_sentences=8)

        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            s = self.summarize(chunk, mode="brief")
            chunk_summaries.append(f"[Chunk {i+1}]: {s}")
            time.sleep(0.2)  # gentle rate limiting

        combined = "\n".join(chunk_summaries)
        return self.summarize(combined, mode=final_mode)


# ---------------------------------------------------------------------------
# ExtractiveSummarizer — no LLM required
# ---------------------------------------------------------------------------

class ExtractiveSummarizer:
    """
    TF-IDF sentence scoring + regex NER.
    Used as fallback when Claude API is unavailable.
    """

    # Simple stop words for TF-IDF
    STOP_WORDS = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","are","was","were","be","been","being","have","has","had","do",
        "does","did","will","would","could","should","may","might","shall",
        "this","that","these","those","it","its","we","our","us","they","their",
        "he","she","his","her","i","you","your","which","who","what","when",
        "where","how","not","no","from","by","as","if","so","than","then","into",
    }

    # NER patterns
    ORG_PATTERN     = re.compile(r'\b([A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co|Group|Holdings|Bancorp|Technologies|Services|Financial|Capital|Partners|Enterprises|Solutions|Systems|Industries|International|Pharmaceuticals|Therapeutics|Biosciences)\.?)\b')
    MONEY_PATTERN   = re.compile(r'\$[\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion|thousand|M|B|T|K)?', re.IGNORECASE)
    DATE_PATTERN    = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b(?:Q[1-4])\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b')
    PERCENT_PATTERN = re.compile(r'[-+]?\d+(?:\.\d+)?%')
    NUMBER_PATTERN  = re.compile(
        r'(?:(?:\$|USD|EUR|GBP)?\s*)([\d,]+(?:\.\d+)?)\s*'
        r'(million|billion|trillion|thousand|percent|bp|bps|ppt|x|times)?',
        re.IGNORECASE
    )

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if t not in self.STOP_WORDS and len(t) > 2]

    def _sentence_tokenize(self, text: str) -> List[str]:
        # Simple sentence splitter
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _compute_tfidf(self, sentences: List[str]) -> Dict[str, float]:
        """Compute TF-IDF weights for each token across sentences."""
        # TF: term frequency per sentence
        n = len(sentences)
        if n == 0:
            return {}

        # Count document frequency
        df: Counter = Counter()
        tf_list = []
        for sent in sentences:
            tokens = self._tokenize(sent)
            tf = Counter(tokens)
            tf_list.append(tf)
            for tok in set(tokens):
                df[tok] += 1

        # Compute TF-IDF per token (averaged across sentences)
        tfidf: Dict[str, float] = {}
        for tok, freq in df.items():
            idf = math.log((n + 1) / (freq + 1)) + 1
            avg_tf = sum(tf.get(tok, 0) for tf in tf_list) / n
            tfidf[tok] = avg_tf * idf

        return tfidf

    def summarize(self, text: str, n_sentences: int = 5) -> str:
        """Extract top N sentences by TF-IDF score."""
        sentences = self._sentence_tokenize(text)
        if not sentences:
            return text[:500]
        if len(sentences) <= n_sentences:
            return " ".join(sentences)

        tfidf = self._compute_tfidf(sentences)
        if not tfidf:
            return " ".join(sentences[:n_sentences])

        scores = []
        for sent in sentences:
            tokens = self._tokenize(sent)
            score = sum(tfidf.get(t, 0) for t in tokens) / (len(tokens) + 1)
            scores.append(score)

        # Get top-N indices preserving original order
        top_indices = sorted(
            sorted(range(len(scores)), key=lambda i: -scores[i])[:n_sentences]
        )
        return " ".join(sentences[i] for i in top_indices)

    def extract_key_phrases(self, text: str, n: int = 10) -> List[str]:
        """Extract top N key phrases (2-gram and 3-gram)."""
        sentences = self._sentence_tokenize(text)
        tokens_list = [self._tokenize(s) for s in sentences]
        all_tokens = [t for tl in tokens_list for t in tl]

        # Build 2-grams and 3-grams
        bigrams  = [" ".join(all_tokens[i:i+2]) for i in range(len(all_tokens)-1)]
        trigrams = [" ".join(all_tokens[i:i+3]) for i in range(len(all_tokens)-2)]

        phrase_counts: Counter = Counter(bigrams + trigrams)

        # Filter to phrases with count >= 2 or high tfidf
        tfidf = self._compute_tfidf(sentences)
        scored = {}
        for phrase, cnt in phrase_counts.items():
            if cnt < 2:
                continue
            words = phrase.split()
            score = cnt * sum(tfidf.get(w, 0) for w in words) / len(words)
            scored[phrase] = score

        top = sorted(scored, key=lambda p: -scored[p])[:n]
        return top

    def extract_numbers(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Extract (context, number, unit) triples from text.
        Attempts to capture financial figures with surrounding context.
        """
        results = []
        for match in self.NUMBER_PATTERN.finditer(text):
            raw_num = match.group(1).replace(",", "")
            unit = (match.group(2) or "").lower()
            try:
                val = float(raw_num)
            except ValueError:
                continue
            if val == 0:
                continue

            # Multiplier
            multiplier = {
                "thousand": 1e3, "million": 1e6, "billion": 1e9,
                "trillion": 1e12, "m": 1e6, "b": 1e9, "t": 1e12, "k": 1e3,
            }.get(unit, 1)
            val *= multiplier

            # Context window: 60 chars before match
            start = max(0, match.start() - 60)
            context = text[start:match.start()].strip().split(".")[-1]
            results.append((context.strip(), val, unit))
        return results[:50]  # limit

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Manual regex NER: ORG, MONEY, DATE, PERCENT."""
        return {
            "ORG":     list({m.group(1) for m in self.ORG_PATTERN.finditer(text)}),
            "MONEY":   list({m.group() for m in self.MONEY_PATTERN.finditer(text)}),
            "DATE":    list({m.group() for m in self.DATE_PATTERN.finditer(text)}),
            "PERCENT": list({m.group() for m in self.PERCENT_PATTERN.finditer(text)}),
        }

    # ------------------------------------------------------------------
    # dim_055 additions
    # ------------------------------------------------------------------

    def compute_text_density_score(self, text: str) -> float:
        """Compute information density as unique_words / total_words.

        Higher score → more substantive text (less repetition).
        Returns a float in [0, 1]. Empty text returns 0.0.

        Identity: density = len(set(words)) / len(words)
        """
        if not text or not text.strip():
            return 0.0
        # Normalize: lowercase, split on whitespace
        words = text.lower().split()
        if not words:
            return 0.0
        unique_words = set(words)
        density = len(unique_words) / len(words)
        return round(density, 6)

    def extract_key_numbers(self, text: str) -> Dict[str, List[Dict]]:
        """Extract all dollar amounts, percentages, and basis points from text.

        Returns a structured dict:
        {
            "dollar_amounts": [{"raw": "$89 billion", "value": 89e9, "context": "..."}],
            "percentages":    [{"raw": "12%", "value": 0.12, "context": "..."}],
            "basis_points":   [{"raw": "25 bps", "value": 25, "context": "..."}],
        }
        """
        result: Dict[str, List[Dict]] = {
            "dollar_amounts": [],
            "percentages":    [],
            "basis_points":   [],
        }

        # Dollar amounts: $X [billion|million|thousand|B|M|K]?
        dollar_re = re.compile(
            r'\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|trillion|B|M|T|K)?',
            re.IGNORECASE,
        )
        for m in dollar_re.finditer(text):
            raw_num = m.group(1).replace(",", "")
            try:
                val = float(raw_num)
            except ValueError:
                continue
            unit = (m.group(2) or "").lower()
            mult = {"billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6,
                    "trillion": 1e12, "t": 1e12, "thousand": 1e3, "k": 1e3}.get(unit, 1.0)
            val *= mult
            ctx_start = max(0, m.start() - 50)
            ctx_end   = min(len(text), m.end() + 30)
            result["dollar_amounts"].append({
                "raw":     m.group(0),
                "value":   val,
                "context": text[ctx_start:ctx_end].strip(),
            })

        # Percentages: X% or X percent
        pct_re = re.compile(
            r'([-+]?\d+(?:\.\d+)?)\s*(?:%|percent\b)',
            re.IGNORECASE,
        )
        for m in pct_re.finditer(text):
            try:
                val = float(m.group(1)) / 100.0
            except ValueError:
                continue
            ctx_start = max(0, m.start() - 50)
            ctx_end   = min(len(text), m.end() + 30)
            result["percentages"].append({
                "raw":     m.group(0),
                "value":   val,
                "context": text[ctx_start:ctx_end].strip(),
            })

        # Basis points: X bps / X bp / X basis points
        bps_re = re.compile(
            r'([-+]?\d+(?:\.\d+)?)\s*(?:bps?|basis\s+points?)',
            re.IGNORECASE,
        )
        for m in bps_re.finditer(text):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            ctx_start = max(0, m.start() - 50)
            ctx_end   = min(len(text), m.end() + 30)
            result["basis_points"].append({
                "raw":     m.group(0),
                "value":   val,
                "context": text[ctx_start:ctx_end].strip(),
            })

        return result

    def compute_sentiment_arc(self, text: str) -> Dict[str, Any]:
        """Divide document into thirds; score each for sentiment direction.

        Returns:
        {
            "thirds": [
                {"label": "beginning", "pos": p, "neg": n, "score": s},
                {"label": "middle",    "pos": p, "neg": n, "score": s},
                {"label": "end",       "pos": p, "neg": n, "score": s},
            ],
            "arc": "improving" | "deteriorating" | "stable" | "mixed",
            "delta": end_score - beginning_score,
        }

        score per third = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        arc is "improving" if end_score > beginning_score + 0.05, etc.
        """
        _POS = [
            "growth", "increase", "strong", "record", "expand", "improve",
            "positive", "outperform", "exceed", "accelerat", "gain", "beat",
            "robust", "momentum", "confident", "opportunity", "upside",
        ]
        _NEG = [
            "decline", "decrease", "challenging", "headwind", "pressure",
            "weakness", "miss", "deteriorat", "uncertain", "concern", "risk",
            "warn", "loss", "below", "disappoint", "difficult", "volatile",
        ]

        words = text.split()
        n = len(words)
        if n == 0:
            return {"thirds": [], "arc": "stable", "delta": 0.0}

        third = max(1, n // 3)
        segments = [
            ("beginning", words[:third]),
            ("middle",    words[third: 2 * third]),
            ("end",       words[2 * third:]),
        ]

        thirds_data = []
        for label, seg in segments:
            seg_lower = " ".join(seg).lower()
            pos = sum(1 for p in _POS if p in seg_lower)
            neg = sum(1 for p in _NEG if p in seg_lower)
            total = pos + neg
            score = (pos - neg) / max(total, 1)
            thirds_data.append({"label": label, "pos": pos, "neg": neg, "score": round(score, 4)})

        begin_score = thirds_data[0]["score"]
        end_score   = thirds_data[2]["score"]
        delta       = round(end_score - begin_score, 4)

        if delta > 0.05:
            arc = "improving"
        elif delta < -0.05:
            arc = "deteriorating"
        elif abs(thirds_data[1]["score"] - begin_score) > 0.1:
            arc = "mixed"
        else:
            arc = "stable"

        return {"thirds": thirds_data, "arc": arc, "delta": delta}


# ---------------------------------------------------------------------------
# SEC EDGAR utilities
# ---------------------------------------------------------------------------

class _EDGARClient:
    """Low-level EDGAR fetcher with caching."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_EDGAR_HEADERS)
        self._cache: Dict[str, Any] = {}

    def get_cik(self, ticker: str) -> Optional[str]:
        """Resolve ticker → CIK via EDGAR company search."""
        cache_key = f"cik_{ticker}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2000-01-01&forms=10-K"
            r = self._session.get(url, timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                cik = hits[0].get("_source", {}).get("entity_id", "")
                if cik:
                    cik = cik.zfill(10)
                    self._cache[cache_key] = cik
                    return cik
        except Exception:
            pass

        # Fallback: company_tickers.json
        try:
            r = self._session.get(
                "https://www.sec.gov/files/company_tickers.json",
                timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cache[cache_key] = cik
                    return cik
        except Exception:
            pass
        return None

    def get_submissions(self, cik: str) -> Optional[Dict]:
        """Fetch submissions JSON for a CIK."""
        cache_key = f"sub_{cik}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            url = EDGAR_SUBMISSIONS.format(cik=cik)
            r = self._session.get(url, timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            self._cache[cache_key] = data
            return data
        except Exception as exc:
            logger.warning("EDGAR submissions fetch failed for CIK %s: %s", cik, exc)
            return None

    def find_filings(self, cik: str, form_type: str,
                     year: Optional[int] = None,
                     max_results: int = 5) -> List[Dict]:
        """Return list of filings matching form_type (optionally filtered by year)."""
        subs = self.get_submissions(cik)
        if not subs:
            return []

        filings_data = subs.get("filings", {}).get("recent", {})
        if not filings_data:
            return []

        forms      = filings_data.get("form", [])
        dates      = filings_data.get("filingDate", [])
        accessions = filings_data.get("accessionNumber", [])
        documents  = filings_data.get("primaryDocument", [])

        results = []
        for form, date_str, acc, doc in zip(forms, dates, accessions, documents):
            if form != form_type:
                continue
            if year is not None:
                try:
                    if int(date_str[:4]) != year and int(date_str[:4]) != year + 1:
                        continue
                except (ValueError, IndexError):
                    continue
            results.append({
                "form": form,
                "date": date_str,
                "accession": acc,
                "document": doc,
            })
            if len(results) >= max_results:
                break
        return results

    def fetch_filing_text(self, cik: str, accession: str,
                           document: str, max_chars: int = 200_000) -> str:
        """Download filing text from EDGAR archives."""
        acc_clean = accession.replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/full-index/"
               f"../../Archives/edgar/data/{int(cik)}/{acc_clean}/{document}")
        # More reliable path
        url2 = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{acc_clean}/{document}")
        for target_url in [url2, url]:
            try:
                r = self._session.get(target_url, timeout=_REQUEST_TIMEOUT)
                if r.status_code == 200:
                    text = r.text[:max_chars]
                    return _strip_html(text)
            except Exception:
                continue
        return ""

    def efts_search(self, ticker: str, form_type: str,
                    year: Optional[int] = None) -> List[Dict]:
        """EDGAR EFTS full-text search."""
        year_str = f"{year}-01-01&enddt={year}-12-31" if year else ""
        url = (f"{EFTS_SEARCH}?q=&forms={form_type}"
               f"&dateRange=custom&startdt={year_str if year else '2000-01-01'}"
               f"&entity={ticker}")
        try:
            r = self._session.get(url, timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
            return [h.get("_source", {}) for h in hits[:10]]
        except Exception as exc:
            logger.warning("EFTS search failed: %s", exc)
            return []


def _strip_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    # Remove script/style blocks
    text = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', ' ', text,
                  flags=re.DOTALL | re.IGNORECASE)
    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>') \
               .replace('&nbsp;', ' ').replace('&#39;', "'").replace('&quot;', '"')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# SEC section extractor
# ---------------------------------------------------------------------------

class _SECSectionExtractor:
    """Extract labeled sections from SEC filing text using regex patterns."""

    SECTION_PATTERNS = {
        "risk_factors": [
            re.compile(r'ITEM\s+1A[\.\s]+RISK\s+FACTORS', re.IGNORECASE),
            re.compile(r'Item\s+1A\.\s*Risk\s+Factors', re.IGNORECASE),
        ],
        "mda": [
            re.compile(r"ITEM\s+7[\.\s]+MANAGEMENT'?S?\s+DISCUSSION", re.IGNORECASE),
            re.compile(r"Item\s+7\.\s*Management'?s?\s+Discussion", re.IGNORECASE),
        ],
        "financial_statements": [
            re.compile(r'ITEM\s+8[\.\s]+FINANCIAL\s+STATEMENTS', re.IGNORECASE),
            re.compile(r'Item\s+8\.\s*Financial\s+Statements', re.IGNORECASE),
        ],
        "quantitative_disclosures": [
            re.compile(r'ITEM\s+(?:7A|3)[\.\s]+QUANTITATIVE', re.IGNORECASE),
        ],
        "business": [
            re.compile(r'ITEM\s+1[\.\s]+BUSINESS', re.IGNORECASE),
        ],
    }

    # Next-section markers for boundary detection
    NEXT_SECTION_PATTERNS = [
        re.compile(r'ITEM\s+\d+[A-Z]?[\.\s]+[A-Z]', re.IGNORECASE),
        re.compile(r'Item\s+\d+[A-Z]?\.', re.IGNORECASE),
    ]

    def extract_section(self, text: str, section: str,
                         max_chars: int = 40_000) -> str:
        """Extract a named section from filing text."""
        patterns = self.SECTION_PATTERNS.get(section, [])
        if not patterns:
            return ""

        start_pos = None
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                start_pos = match.start()
                break

        if start_pos is None:
            return ""

        # Find end: next item heading
        search_text = text[start_pos + 50:]
        end_pos = len(search_text)

        for pattern in self.NEXT_SECTION_PATTERNS:
            match = pattern.search(search_text)
            if match and match.start() < end_pos:
                end_pos = match.start()

        section_text = text[start_pos: start_pos + 50 + end_pos]
        return section_text[:max_chars]

    def extract_10q_sections(self, text: str) -> Dict[str, str]:
        return {
            "mda": self.extract_section(text, "mda"),
            "quantitative_disclosures": self.extract_section(text, "quantitative_disclosures"),
        }

    def extract_10k_sections(self, text: str) -> Dict[str, str]:
        return {
            "business":           self.extract_section(text, "business"),
            "risk_factors":       self.extract_section(text, "risk_factors"),
            "mda":                self.extract_section(text, "mda"),
            "financial_statements": self.extract_section(text, "financial_statements"),
        }


# ---------------------------------------------------------------------------
# SECFilingSummarizer
# ---------------------------------------------------------------------------

class SECFilingSummarizer:
    """Fetch and summarize SEC filings (10-K, 10-Q, 8-K, DEF 14A)."""

    def __init__(self, claude: Optional[ClaudeAPIClient] = None):
        self._claude = claude or ClaudeAPIClient()
        self._edgar  = _EDGARClient()
        self._sec_extractor = _SECSectionExtractor()
        self._extractive = ExtractiveSummarizer()

    # ------------------------------------------------------------------
    # 10-K
    # ------------------------------------------------------------------

    def fetch_and_summarize_10k(self, ticker: str,
                                  year: Optional[int] = None) -> FilingSummary:
        """Fetch latest 10-K for ticker and summarize key sections."""
        year = year or (datetime.now().year - 1)
        cik = self._edgar.get_cik(ticker)
        if not cik:
            logger.warning("Could not resolve CIK for %s", ticker)
            return FilingSummary(ticker=ticker, filing_type="10-K",
                                 period=str(year), accession_number="",
                                 filed_date="", executive_summary="CIK not found.")

        filings = self._edgar.find_filings(cik, "10-K", year=year, max_results=3)
        if not filings:
            # Try without year filter
            filings = self._edgar.find_filings(cik, "10-K", max_results=3)
        if not filings:
            return FilingSummary(ticker=ticker, filing_type="10-K",
                                 period=str(year), accession_number="",
                                 filed_date="", executive_summary="No 10-K found.")

        filing = filings[0]
        text = self._edgar.fetch_filing_text(
            cik, filing["accession"], filing["document"])

        if not text:
            return FilingSummary(
                ticker=ticker, filing_type="10-K",
                period=str(year),
                accession_number=filing["accession"],
                filed_date=filing["date"],
                executive_summary="Failed to fetch filing text.",
            )

        # Extract sections
        sections = self._sec_extractor.extract_10k_sections(text)

        # Summarize risk factors
        rf_text = sections.get("risk_factors", "")
        rf_summary = self.summarize_risk_factors(rf_text) if rf_text else None

        # Summarize MD&A
        mda_text = sections.get("mda", "")
        mda_summary = self.summarize_mda(mda_text) if mda_text else None

        # Executive summary: combine
        brief_parts = []
        if rf_summary and rf_summary.narrative:
            brief_parts.append(f"RISKS: {rf_summary.narrative}")
        if mda_summary and mda_summary.narrative:
            brief_parts.append(f"OPERATIONS: {mda_summary.narrative}")

        if not brief_parts:
            # Fall back to full-text extractive summary
            exec_summary = self._extractive.summarize(text[:20_000], n_sentences=6)
        else:
            exec_summary = " | ".join(brief_parts)

        # Key metrics from numbers
        numbers = self._extractive.extract_numbers(mda_text or text[:10_000])
        key_metrics = {f"figure_{i+1}": f"{ctx}: {val:,.0f} {unit}"
                       for i, (ctx, val, unit) in enumerate(numbers[:8])}

        return FilingSummary(
            ticker=ticker,
            filing_type="10-K",
            period=str(year),
            accession_number=filing["accession"],
            filed_date=filing["date"],
            executive_summary=exec_summary,
            key_metrics=key_metrics,
            risks=rf_summary.top_risks[:5] if rf_summary else [],
            outlook=mda_summary.forward_outlook if mda_summary else "",
            full_text_length=len(text),
            model_used=HAIKU_MODEL if self._claude._available else "extractive",
        )

    def summarize_risk_factors(self, text: str) -> RiskSummary:
        """
        Analyze risk factors section:
        - Count risk paragraphs
        - Rank by emphasis (mention count × proximity to start)
        - Classify by category
        """
        if not text:
            return RiskSummary(ticker="", period="")

        # Split into individual risks (delimited by header-like patterns)
        risk_pattern = re.compile(
            r'(?:^|\n)\s*([A-Z][A-Z\s]{5,60})\s*\n',
            re.MULTILINE
        )
        risk_sections = risk_pattern.split(text)

        # Category keywords
        categories = {
            "macro":       ["interest rate", "inflation", "recession", "economic", "gdp", "federal reserve"],
            "regulatory":  ["regulation", "compliance", "sec", "government", "law", "legal", "litigation"],
            "competitive": ["competition", "market share", "competitor", "pricing pressure"],
            "operational": ["supply chain", "operations", "manufacturing", "labor", "workforce"],
            "technology":  ["cybersecurity", "data breach", "technology", "ai", "cloud", "digital"],
            "financial":   ["debt", "liquidity", "credit", "capital", "cash flow", "covenant"],
            "geopolitical":["geopolit", "international", "china", "russia", "tariff", "trade war"],
        }

        risk_cats: Dict[str, int] = defaultdict(int)
        top_risks: List[Dict[str, Any]] = []

        # Score by position (earlier = more emphasized) and mention count
        chunks = [text[i:i+500] for i in range(0, min(len(text), 40_000), 500)]
        for i, chunk in enumerate(chunks[:50]):
            chunk_lower = chunk.lower()
            for cat, keywords in categories.items():
                if any(kw in chunk_lower for kw in keywords):
                    risk_cats[cat] += 1

        # Extract top risk paragraphs by extractive method
        top_risk_texts = self._extractive.summarize(text, n_sentences=8).split(". ")
        for j, rt in enumerate(top_risk_texts[:10]):
            if len(rt) > 30:
                cat = "other"
                rt_lower = rt.lower()
                for c, kws in categories.items():
                    if any(kw in rt_lower for kw in kws):
                        cat = c
                        break
                score = 1.0 / (j + 1)  # positional weight
                top_risks.append({"text": rt, "score": score, "category": cat})

        # Claude narrative
        if self._claude._available and text:
            narrative = self._claude.summarize(
                text[:8_000],
                context="This is the Risk Factors section of a 10-K SEC filing.",
                mode="bullets",
            )
        else:
            narrative = self._extractive.summarize(text, n_sentences=5)

        return RiskSummary(
            ticker="",
            period="",
            top_risks=top_risks[:10],
            risk_categories=dict(risk_cats),
            total_risk_factors=len(top_risks),
            emphasis_score=float(len(top_risks)) / max(1, len(text) // 1000),
            narrative=narrative,
        )

    def summarize_mda(self, text: str) -> MDASummary:
        """Summarize Management's Discussion & Analysis section."""
        if not text:
            return MDASummary(ticker="", period="")

        # Extract numbers for financial context
        numbers = self._extractive.extract_numbers(text)
        key_numbers = [(ctx, val, unit) for ctx, val, unit in numbers[:10]
                       if val > 100_000]  # filter small noise values

        # Revenue driver extraction (heuristic)
        revenue_pattern = re.compile(
            r'(?:revenue|net sales|total revenues?).*?(?:\.|$)', re.IGNORECASE)
        rev_drivers = [m.group()[:200] for m in revenue_pattern.finditer(text)][:5]

        # Margin trend detection
        margin_pattern = re.compile(
            r'(?:gross margin|operating margin|net margin|EBITDA margin).*?(?:\.|$)',
            re.IGNORECASE)
        margin_text = " ".join(m.group()[:200] for m in margin_pattern.finditer(text))[:500]

        # Forward outlook
        outlook_pattern = re.compile(
            r'(?:expect|anticipate|guidance|forecast|outlook|project|full.year).*?(?:\.|$)',
            re.IGNORECASE)
        outlook_sentences = [m.group()[:200] for m in outlook_pattern.finditer(text)][:4]
        outlook_text = " ".join(outlook_sentences)

        # Management tone (simple keyword count)
        pos_words = ["growth", "increase", "strong", "record", "expand", "improve",
                     "positive", "outperform", "exceed", "accelerat"]
        neg_words = ["decline", "decrease", "challenging", "headwind", "pressure",
                     "weakness", "miss", "deteriorat", "uncertain", "concern"]
        text_lower = text.lower()
        pos_count = sum(text_lower.count(w) for w in pos_words)
        neg_count = sum(text_lower.count(w) for w in neg_words)
        if pos_count > neg_count * 1.5:
            tone = "positive"
        elif neg_count > pos_count * 1.5:
            tone = "negative"
        else:
            tone = "neutral"

        # Claude narrative
        if self._claude._available and text:
            narrative = self._claude.summarize(
                text[:8_000],
                context="This is the MD&A section of a 10-K SEC filing. Focus on revenue, margins, and outlook.",
                mode="detailed",
            )
        else:
            narrative = self._extractive.summarize(text, n_sentences=6)

        return MDASummary(
            ticker="",
            period="",
            revenue_drivers=rev_drivers,
            margin_trends=margin_text,
            forward_outlook=outlook_text[:500],
            management_tone=tone,
            key_numbers=key_numbers,
            narrative=narrative,
        )

    def compare_annual_risk_factors(self, ticker: str,
                                     year1: int, year2: int) -> RiskDelta:
        """Compare risk factors between two annual 10-K filings."""
        summary1 = self.fetch_and_summarize_10k(ticker, year1)
        summary2 = self.fetch_and_summarize_10k(ticker, year2)

        risks1_set = {r["text"][:80] if isinstance(r, dict) else r[:80]
                      for r in summary1.risks}
        risks2_set = {r["text"][:80] if isinstance(r, dict) else r[:80]
                      for r in summary2.risks}

        new_risks     = [r for r in risks2_set if r not in risks1_set]
        removed_risks = [r for r in risks1_set if r not in risks2_set]

        if self._claude._available:
            compare_context = (
                f"Year {year1} risks: {'; '.join(list(risks1_set)[:10])}\n"
                f"Year {year2} risks: {'; '.join(list(risks2_set)[:10])}"
            )
            narrative = self._claude.summarize(
                compare_context,
                context=f"Compare risk factor changes for {ticker} from {year1} to {year2}.",
                mode="brief",
            )
        else:
            narrative = (f"New risks: {len(new_risks)}; "
                         f"Removed risks: {len(removed_risks)}")

        return RiskDelta(
            ticker=ticker,
            year1=year1,
            year2=year2,
            new_risks=new_risks[:10],
            removed_risks=removed_risks[:10],
            intensified_risks=[],
            narrative=narrative,
        )


# ---------------------------------------------------------------------------
# EarningsCallSummarizer
# ---------------------------------------------------------------------------

class EarningsCallSummarizer:
    """
    Fetch and summarize earnings call transcripts.
    Sources: SEC 8-K (Item 7.01), Motley Fool free transcripts.
    """

    def __init__(self, claude: Optional[ClaudeAPIClient] = None):
        self._claude = claude or ClaudeAPIClient()
        self._edgar  = _EDGARClient()
        self._extractive = ExtractiveSummarizer()
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 SENTINEL/3.0",
            "Accept": "text/html,application/xhtml+xml",
        })

    def fetch_transcript(self, ticker: str, quarter: str) -> str:
        """
        Fetch earnings call transcript text.
        quarter: e.g. "Q4 2024"
        """
        # Try SEC 8-K first
        text = self._fetch_8k_transcript(ticker, quarter)
        if text and len(text) > 200:
            return text

        # Try Motley Fool
        text = self._fetch_motley_fool(ticker, quarter)
        if text and len(text) > 200:
            return text

        return ""

    def _fetch_8k_transcript(self, ticker: str, quarter: str) -> str:
        """Look for transcript text in 8-K Item 7.01 filings."""
        # Parse quarter
        q_match = re.match(r'Q(\d)\s+(\d{4})', quarter)
        if not q_match:
            return ""
        q_num = int(q_match.group(1))
        year  = int(q_match.group(2))

        # Estimate fiscal quarter end month
        q_month = {1: 3, 2: 6, 3: 9, 4: 12}.get(q_num, 3)
        target_year = year if q_month <= 12 else year + 1

        cik = self._edgar.get_cik(ticker)
        if not cik:
            return ""

        filings = self._edgar.find_filings(cik, "8-K", year=target_year, max_results=10)

        for filing in filings:
            text = self._edgar.fetch_filing_text(
                cik, filing["accession"], filing["document"], max_chars=100_000)
            # Check for earnings call transcript indicators
            if ("earnings call" in text.lower() or
                    "conference call" in text.lower() or
                    "operator:" in text.lower() or
                    "analyst:" in text.lower()):
                return text
        return ""

    def _fetch_motley_fool(self, ticker: str, quarter: str) -> str:
        """Scrape Motley Fool earnings transcript listing for ticker/quarter."""
        try:
            url = MOTLEY_FOOL_TRANSCRIPTS.format(page=1)
            r = self._session.get(url, timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            text = r.text
            # Look for links containing the ticker symbol
            link_pattern = re.compile(
                rf'href="(/[^"]*{re.escape(ticker.lower())}[^"]*transcript[^"]*)"',
                re.IGNORECASE
            )
            matches = link_pattern.findall(text)
            if not matches:
                return ""
            # Fetch first match
            transcript_url = "https://www.fool.com" + matches[0]
            r2 = self._session.get(transcript_url, timeout=_REQUEST_TIMEOUT)
            r2.raise_for_status()
            return _strip_html(r2.text)[:80_000]
        except Exception as exc:
            logger.debug("Motley Fool scrape failed for %s: %s", ticker, exc)
            return ""

    def summarize_transcript(self, text: str) -> EarningsSummary:
        """Extract tone, guidance, concerns, and bull/bear debate from transcript."""
        ticker = ""
        quarter = ""

        # Management section: typically prepared remarks before Q&A
        qa_start = -1
        for marker in ["question-and-answer", "q&a session", "question and answer",
                        "question:", "analyst:"]:
            idx = text.lower().find(marker)
            if idx > 0:
                qa_start = idx
                break

        prepared = text[:qa_start] if qa_start > 0 else text[:len(text)//2]
        qa_text   = text[qa_start:] if qa_start > 0 else text[len(text)//2:]

        # Management tone
        pos_words = ["exceeded", "record", "strong", "growth", "positive", "momentum",
                     "confident", "outperform", "robust", "accelerat"]
        neg_words = ["disappointing", "challenging", "headwind", "pressure", "uncertain",
                     "cautious", "miss", "below", "concern", "deteriorat"]
        pl = sum(text.lower().count(w) for w in pos_words)
        nl = sum(text.lower().count(w) for w in neg_words)
        tone = "positive" if pl > nl * 1.3 else ("negative" if nl > pl * 1.3 else "neutral")

        # Guidance extraction
        guidance = self.extract_guidance(text)

        # Surprise language
        surprises = self.detect_surprise_language(text)

        # Q&A concerns (simple: find analyst questions)
        concern_pattern = re.compile(r'(?:analyst|operator).*?(?:\?|\.)', re.IGNORECASE)
        concerns = [m.group()[:200] for m in concern_pattern.finditer(qa_text)][:6]

        # Bull / bear extraction from full text
        bull_terms = ["opportunity", "upside", "growth driver", "expansion", "market share",
                      "innovation", "tailwind", "launch", "new product"]
        bear_terms = ["risk", "headwind", "competitive pressure", "margin compression",
                      "cost increase", "supply chain", "regulation", "litigation"]

        bull_points = []
        bear_points = []
        sentences = self._extractive._sentence_tokenize(text)
        for sent in sentences[:200]:
            sl = sent.lower()
            if any(t in sl for t in bull_terms):
                bull_points.append(sent[:200])
            elif any(t in sl for t in bear_terms):
                bear_points.append(sent[:200])

        # Claude narrative
        if self._claude._available and text:
            narrative = self._claude.summarize(
                prepared[:6_000],
                context="This is an earnings call transcript prepared remarks section. "
                        "Summarize management's key messages, guidance, and tone.",
                mode="bullets",
            )
        else:
            narrative = self._extractive.summarize(prepared, n_sentences=6)

        return EarningsSummary(
            ticker=ticker,
            quarter=quarter,
            management_tone=tone,
            revenue_guidance=guidance.revenue_guide,
            eps_guidance=guidance.eps_guide,
            top_concerns=concerns[:5],
            bull_points=list(dict.fromkeys(bull_points))[:5],
            bear_points=list(dict.fromkeys(bear_points))[:5],
            surprise_language=surprises[:10],
            narrative=narrative,
        )

    def extract_guidance(self, text: str) -> GuidanceData:
        """Extract revenue and EPS guidance ranges from transcript."""
        guidance = GuidanceData()

        # Revenue guidance patterns
        rev_patterns = [
            re.compile(
                r'(?:revenue|sales).*?guidance.*?\$?([\d,.]+)\s*(?:to|and|-)\s*\$?([\d,.]+)\s*(billion|million|B|M)?',
                re.IGNORECASE),
            re.compile(
                r'(?:expect|anticipate|guide).*?(?:revenue|sales).*?\$?([\d,.]+)\s*(billion|million|B|M)',
                re.IGNORECASE),
        ]
        for pattern in rev_patterns:
            m = pattern.search(text)
            if m:
                try:
                    groups = m.groups()
                    guidance.revenue_guide = m.group()[:200]
                    unit_mult = 1e9 if any("b" in str(g).lower() for g in groups if g) else 1e6
                    vals = [float(str(g).replace(",","")) * unit_mult
                            for g in groups[:2] if g and re.match(r'[\d,]+', str(g))]
                    if len(vals) >= 2:
                        guidance.revenue_low, guidance.revenue_high = sorted(vals[:2])
                    elif len(vals) == 1:
                        guidance.revenue_low = guidance.revenue_high = vals[0]
                except (ValueError, IndexError):
                    pass
                break

        # EPS guidance patterns
        eps_patterns = [
            re.compile(
                r'(?:EPS|earnings per share).*?\$([\d.]+)\s*(?:to|and|-)\s*\$([\d.]+)',
                re.IGNORECASE),
            re.compile(
                r'(?:expect|guide).*?(?:EPS|earnings per share).*?\$([\d.]+)',
                re.IGNORECASE),
        ]
        for pattern in eps_patterns:
            m = pattern.search(text)
            if m:
                guidance.eps_guide = m.group()[:200]
                try:
                    groups = m.groups()
                    vals = [float(g) for g in groups if g]
                    if len(vals) >= 2:
                        guidance.eps_low, guidance.eps_high = sorted(vals[:2])
                    elif vals:
                        guidance.eps_low = guidance.eps_high = vals[0]
                except (ValueError, IndexError):
                    pass
                break

        # FY marker
        fy_match = re.search(r'(?:fiscal|full.year)\s+(\d{4})', text, re.IGNORECASE)
        if fy_match:
            guidance.fiscal_year = fy_match.group(1)

        guidance.confidence = 0.8 if guidance.revenue_guide or guidance.eps_guide else 0.3
        return guidance

    def detect_surprise_language(self, text: str) -> List[str]:
        """Detect language indicating surprise (positive or negative)."""
        patterns = [
            "better than expected", "ahead of expectations", "exceeded guidance",
            "beat estimates", "upside surprise",
            "below expectations", "missed estimates", "fell short",
            "headwind", "cautious", "uncertain", "challenging environment",
            "macro pressure", "weaker than anticipated", "disappointing",
            "stronger than anticipated", "outperformed", "above consensus",
        ]
        found = []
        text_lower = text.lower()
        for phrase in patterns:
            if phrase in text_lower:
                # Get context
                idx = text_lower.find(phrase)
                snippet = text[max(0, idx-30):min(len(text), idx+len(phrase)+50)]
                found.append(snippet.strip())
        return found


# ---------------------------------------------------------------------------
# NewsArticleSummarizer
# ---------------------------------------------------------------------------

class NewsArticleSummarizer:
    """
    Fetch and summarize financial news from GDELT.
    Compute sentiment timelines and detect catalysts.
    """

    POSITIVE_WORDS = [
        "beat", "exceed", "surge", "rally", "jump", "record", "win", "upgrade",
        "growth", "profit", "gain", "rise", "outperform", "boost", "strong",
    ]
    NEGATIVE_WORDS = [
        "miss", "decline", "fall", "drop", "loss", "downgrade", "warn", "cut",
        "weak", "concern", "risk", "sell-off", "crash", "investigation", "lawsuit",
    ]

    def __init__(self, claude: Optional[ClaudeAPIClient] = None):
        self._claude = claude or ClaudeAPIClient()
        self._extractive = ExtractiveSummarizer()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SENTINEL/3.0"})

    def fetch_news(self, query: str, days: int = 7,
                   max_records: int = 25) -> List[NewsArticle]:
        """Fetch articles from GDELT API."""
        encoded = quote_plus(query)
        url = GDELT_DOC_API.format(query=encoded, n=max_records, days=days)
        articles = []
        try:
            r = self._session.get(url, timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            for item in data.get("articles", [])[:max_records]:
                sentiment = self._score_sentiment(item.get("title", "") + " " +
                                                  item.get("seendate", ""))
                articles.append(NewsArticle(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    source=item.get("domain", ""),
                    published=item.get("seendate", ""),
                    seendate=item.get("seendate", ""),
                    language=item.get("language", "English"),
                    domain=item.get("domain", ""),
                    snippet="",
                    sentiment=sentiment,
                ))
        except Exception as exc:
            logger.warning("GDELT fetch failed for query '%s': %s", query, exc)
        return articles

    def _score_sentiment(self, text: str) -> float:
        """Simple word-count sentiment: -1 to 1."""
        tl = text.lower()
        pos = sum(tl.count(w) for w in self.POSITIVE_WORDS)
        neg = sum(tl.count(w) for w in self.NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def summarize_news_cluster(self, articles: List[NewsArticle]) -> str:
        """Multi-document summary of a cluster of news articles."""
        if not articles:
            return "No articles found."

        # Combine titles and snippets
        combined = "\n".join(
            f"- {a.title} ({a.source}, {a.published[:10]})"
            for a in articles[:20]
        )

        if self._claude._available:
            return self._claude.summarize(
                combined,
                context="These are financial news article titles. Synthesize the main themes and market implications.",
                mode="detailed",
            )
        return self._extractive.summarize(combined, n_sentences=5)

    def extract_sentiment_timeline(self, ticker: str,
                                    days: int = 30) -> pd.DataFrame:
        """
        Build a daily sentiment timeline for a ticker over the past N days.
        Returns DataFrame: date, n_articles, avg_sentiment, positive_pct, negative_pct
        """
        articles = self.fetch_news(f'"{ticker}" stock', days=days, max_records=50)

        if not articles:
            return pd.DataFrame(columns=["date", "n_articles", "avg_sentiment",
                                         "positive_pct", "negative_pct"])

        rows: Dict[str, List[float]] = defaultdict(list)
        for art in articles:
            date_str = art.published[:8] if len(art.published) >= 8 else "unknown"
            try:
                date_str = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            rows[date_str].append(art.sentiment)

        records = []
        for date_str, sentiments in sorted(rows.items()):
            arr = np.array(sentiments)
            records.append({
                "date":         date_str,
                "n_articles":   len(arr),
                "avg_sentiment": float(arr.mean()),
                "positive_pct": float((arr > 0.1).mean()),
                "negative_pct": float((arr < -0.1).mean()),
            })
        return pd.DataFrame(records)

    def detect_catalyst(self, ticker: str) -> List[Catalyst]:
        """
        Detect significant news events for a ticker in the past 30 days.
        Classifies by event type and estimated market impact.
        """
        articles = self.fetch_news(ticker, days=30, max_records=50)
        catalysts = []

        event_patterns = {
            "earnings":    ["earnings", "quarterly results", "eps", "revenue report",
                            "beat", "miss", "guidance"],
            "M&A":         ["merger", "acquisition", "deal", "buyout", "takeover",
                            "acquire", "bid for"],
            "regulatory":  ["fda", "sec", "doj", "ftc", "antitrust", "investigation",
                            "fine", "penalty", "settlement"],
            "macro":       ["fed", "inflation", "interest rate", "gdp", "employment",
                            "economic", "recession"],
            "product":     ["launch", "product", "approval", "partnership", "contract",
                            "new product", "patent"],
        }

        for art in articles:
            title_lower = art.title.lower()
            event_type = "general"
            for etype, keywords in event_patterns.items():
                if any(kw in title_lower for kw in keywords):
                    event_type = etype
                    break

            # Only flag high-sentiment articles as catalysts
            if abs(art.sentiment) > 0.3 or event_type in ("earnings", "M&A", "regulatory"):
                impact = "bullish" if art.sentiment > 0.1 else (
                    "bearish" if art.sentiment < -0.1 else "neutral")
                magnitude = min(1.0, abs(art.sentiment) + 0.3)
                date_str = art.published[:10] if len(art.published) >= 10 else ""

                catalysts.append(Catalyst(
                    ticker=ticker,
                    event_type=event_type,
                    date=date_str,
                    headline=art.title[:200],
                    impact=impact,
                    magnitude=magnitude,
                    source_url=art.url,
                ))

        # Deduplicate and sort by magnitude
        seen = set()
        unique_catalysts = []
        for c in sorted(catalysts, key=lambda x: -x.magnitude):
            key = c.headline[:60]
            if key not in seen:
                seen.add(key)
                unique_catalysts.append(c)
        return unique_catalysts[:10]


# ---------------------------------------------------------------------------
# PortfolioNarrativeGenerator
# ---------------------------------------------------------------------------

class PortfolioNarrativeGenerator:
    """Generate natural-language portfolio narratives using Claude."""

    def __init__(self, claude: Optional[ClaudeAPIClient] = None):
        self._claude = claude or ClaudeAPIClient()
        self._extractive = ExtractiveSummarizer()

    def generate_portfolio_summary(self,
                                    holdings: pd.DataFrame,
                                    performance: pd.DataFrame) -> str:
        """
        Generate a portfolio summary narrative.
        holdings: DataFrame with columns [ticker, weight, sector, return_ytd]
        performance: DataFrame with columns [date, portfolio_return, benchmark_return]
        """
        # Build context text
        context_parts = []

        if not holdings.empty:
            top_holdings = holdings.nlargest(5, "weight") if "weight" in holdings.columns else holdings.head(5)
            context_parts.append("Portfolio Holdings:")
            for _, row in top_holdings.iterrows():
                ticker = row.get("ticker", "N/A")
                weight = row.get("weight", 0)
                ret    = row.get("return_ytd", 0)
                context_parts.append(f"  {ticker}: {weight:.1%} weight, {ret:+.1%} YTD")

        if not performance.empty:
            if "portfolio_return" in performance.columns:
                total_ret = performance["portfolio_return"].sum()
                context_parts.append(f"\nTotal Portfolio Return: {total_ret:+.1%}")
            if "benchmark_return" in performance.columns:
                bench_ret = performance["benchmark_return"].sum()
                alpha = total_ret - bench_ret if "portfolio_return" in performance.columns else 0
                context_parts.append(f"Benchmark Return: {bench_ret:+.1%}")
                context_parts.append(f"Alpha: {alpha:+.1%}")

        context_str = "\n".join(context_parts)

        if self._claude._available:
            return self._claude.summarize(
                context_str,
                context="Write a concise 3-sentence portfolio performance narrative for an institutional investor.",
                mode="brief",
            )
        return context_str

    def generate_risk_narrative(self, var_data: Dict, stress_results: Dict) -> str:
        """Generate risk narrative from VaR and stress test data."""
        context = (
            f"VaR (95%): {var_data.get('var_95', 'N/A')}\n"
            f"VaR (99%): {var_data.get('var_99', 'N/A')}\n"
            f"Expected Shortfall: {var_data.get('cvar', 'N/A')}\n"
            f"Stress Scenarios:\n"
        )
        for scenario, loss in stress_results.items():
            context += f"  {scenario}: {loss}\n"

        if self._claude._available:
            return self._claude.summarize(
                context,
                context="Summarize portfolio risk in plain English for a CIO briefing.",
                mode="brief",
            )
        return context

    def generate_trade_rationale(self, trade: Dict, context: Dict) -> str:
        """Generate trade rationale narrative."""
        trade_str = (
            f"Trade: {trade.get('action','BUY')} {trade.get('size',0):,} "
            f"shares of {trade.get('ticker','N/A')} @ ${trade.get('price',0):.2f}\n"
            f"Thesis: {trade.get('thesis','')}\n"
            f"Target: ${trade.get('target',0):.2f} | Stop: ${trade.get('stop',0):.2f}\n"
            f"Context: {context.get('regime','')}, {context.get('sector_momentum','')}"
        )
        if self._claude._available:
            return self._claude.summarize(
                trade_str,
                context="Write a 2-sentence institutional trade rationale.",
                mode="brief",
            )
        return trade_str

    def generate_watchlist_brief(self, tickers: List[str]) -> str:
        """Generate a one-liner per ticker using GDELT news."""
        news_client = NewsArticleSummarizer(self._claude)
        briefs = []
        for ticker in tickers[:10]:  # limit
            articles = news_client.fetch_news(ticker, days=7, max_records=3)
            if articles:
                top = articles[0]
                sentiment_str = ("↑" if top.sentiment > 0.1 else
                                 ("↓" if top.sentiment < -0.1 else "→"))
                briefs.append(f"{ticker}: {sentiment_str} {top.title[:100]}")
            else:
                briefs.append(f"{ticker}: No recent news.")
        return "\n".join(briefs)


# ---------------------------------------------------------------------------
# DocumentSummarizationEngine — orchestrator
# ---------------------------------------------------------------------------

class DocumentSummarizationEngine:
    """
    Full-stack financial document summarization.
    Coordinates SEC filings, earnings calls, news, and Claude synthesis.
    """

    def __init__(self, model_fast: str = HAIKU_MODEL,
                 model_deep: str = SONNET_MODEL):
        self._claude     = ClaudeAPIClient(model_fast=model_fast, model_deep=model_deep)
        self._sec        = SECFilingSummarizer(self._claude)
        self._earnings   = EarningsCallSummarizer(self._claude)
        self._news       = NewsArticleSummarizer(self._claude)
        self._portfolio  = PortfolioNarrativeGenerator(self._claude)
        self._extractive = ExtractiveSummarizer()

    # ------------------------------------------------------------------
    # Core: full ticker package
    # ------------------------------------------------------------------

    def summarize_ticker(self, ticker: str,
                          depth: str = "brief") -> ComprehensiveSummary:
        """
        Produce a comprehensive summary for a ticker:
        1. Latest 10-K risk factors + MD&A
        2. Most recent earnings call
        3. Last 7 days of news + catalysts
        4. Claude synthesis of everything
        """
        logger.info("Summarizing ticker: %s (depth=%s)", ticker, depth)
        summary = ComprehensiveSummary(ticker=ticker, depth=depth)

        # 1. Filing summary
        try:
            summary.filing_summary = self._sec.fetch_and_summarize_10k(ticker)
            if summary.filing_summary:
                rf_text = summary.filing_summary.executive_summary
                risk_sum = self._sec.summarize_risk_factors(rf_text)
                risk_sum.ticker = ticker
                summary.risk_summary = risk_sum
        except Exception as exc:
            logger.warning("Filing summary failed for %s: %s", ticker, exc)

        # 2. Earnings call (last available quarter)
        try:
            year = datetime.now().year
            month = datetime.now().month
            q = (month - 1) // 3 + 1
            quarter = f"Q{q} {year}"
            transcript = self._earnings.fetch_transcript(ticker, quarter)
            if not transcript:
                # Try previous quarter
                pq = q - 1 if q > 1 else 4
                py = year if q > 1 else year - 1
                transcript = self._earnings.fetch_transcript(ticker, f"Q{pq} {py}")
            if transcript:
                earnings_sum = self._earnings.summarize_transcript(transcript)
                earnings_sum.ticker = ticker
                earnings_sum.quarter = quarter
                summary.earnings_summary = earnings_sum
        except Exception as exc:
            logger.warning("Earnings summary failed for %s: %s", ticker, exc)

        # 3. News
        try:
            articles = self._news.fetch_news(f'"{ticker}" stock', days=7)
            summary.news_summary = self._news.summarize_news_cluster(articles)
            summary.catalysts = self._news.detect_catalyst(ticker)

            # Sentiment trend
            sentiments = [a.sentiment for a in articles]
            if sentiments:
                avg_s = np.mean(sentiments)
                summary.sentiment_trend = (
                    "positive" if avg_s > 0.1 else
                    ("negative" if avg_s < -0.1 else "neutral")
                )
        except Exception as exc:
            logger.warning("News summary failed for %s: %s", ticker, exc)

        # 4. Executive brief (Claude synthesis)
        try:
            parts = []
            if summary.filing_summary:
                parts.append(f"10-K: {summary.filing_summary.executive_summary[:500]}")
            if summary.earnings_summary:
                parts.append(f"Earnings: {summary.earnings_summary.narrative[:500]}")
            if summary.news_summary:
                parts.append(f"News: {summary.news_summary[:300]}")

            combined = "\n".join(parts)
            if combined:
                summary.executive_brief = self._claude.summarize(
                    combined,
                    context=f"Write a 3-sentence institutional-grade brief for {ticker}.",
                    mode="brief",
                ) or self._extractive.summarize(combined, n_sentences=3)
        except Exception as exc:
            logger.warning("Executive brief failed for %s: %s", ticker, exc)
            summary.executive_brief = f"Summary for {ticker} generated with limited data."

        logger.info("Ticker summary complete: %s", ticker)
        return summary

    # ------------------------------------------------------------------
    # Batch summarization
    # ------------------------------------------------------------------

    def batch_summarize(self, tickers: List[str],
                         workers: int = 3) -> Dict[str, ComprehensiveSummary]:
        """Summarize multiple tickers in parallel."""
        results: Dict[str, ComprehensiveSummary] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ticker = {
                executor.submit(self.summarize_ticker, ticker): ticker
                for ticker in tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result()
                    logger.info("Batch: completed %s", ticker)
                except Exception as exc:
                    logger.error("Batch: %s failed: %s", ticker, exc)
                    results[ticker] = ComprehensiveSummary(
                        ticker=ticker,
                        executive_brief=f"Summary failed: {exc}",
                    )
        return results

    # ------------------------------------------------------------------
    # Sector brief
    # ------------------------------------------------------------------

    def generate_sector_brief(self, sector: str,
                               tickers: List[str]) -> str:
        """Summarize a sector using news for a list of representative tickers."""
        articles = self._news.fetch_news(f'"{sector}" stocks OR ETF', days=14, max_records=30)
        for ticker in tickers[:5]:
            ticker_articles = self._news.fetch_news(ticker, days=7, max_records=10)
            articles.extend(ticker_articles)

        cluster_summary = self._news.summarize_news_cluster(articles[:40])

        if self._claude._available:
            return self._claude.summarize(
                cluster_summary,
                context=f"Write a sector brief for the {sector} sector covering key themes, "
                        f"risks, and opportunities for institutional investors.",
                mode="detailed",
            )
        return cluster_summary

    # ------------------------------------------------------------------
    # Q&A over document
    # ------------------------------------------------------------------

    def qa_over_document(self, doc_path: str,
                          questions: List[str]) -> Dict[str, str]:
        """
        Load a document from disk and answer a list of questions.
        Falls back to extractive method if Claude unavailable.
        """
        path = Path(doc_path)
        if not path.exists():
            return {q: f"File not found: {doc_path}" for q in questions}

        text = path.read_text(encoding="utf-8", errors="replace")[:_MAX_CONTEXT_CHARS]

        answers = {}
        for question in questions:
            if self._claude._available:
                ans = self._claude.answer_question(text, question)
            else:
                # Extractive: find most relevant sentence
                sentences = self._extractive._sentence_tokenize(text)
                q_tokens = set(self._extractive._tokenize(question))
                best_sent = max(
                    sentences,
                    key=lambda s: len(q_tokens & set(self._extractive._tokenize(s))),
                    default="",
                )
                ans = best_sent or "Not found in document."
            answers[question] = ans
        return answers

    # ------------------------------------------------------------------
    # Convenience: CIK lookup passthrough
    # ------------------------------------------------------------------

    def get_cik(self, ticker: str) -> Optional[str]:
        return self._sec._edgar.get_cik(ticker)


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("SENTINEL Document Summarizer V3")
    print("=" * 60)

    engine = DocumentSummarizationEngine()

    # 1. AAPL 10-K risk factors
    print("\n[1] AAPL 10-K Risk Factors")
    print("-" * 40)
    try:
        sec = SECFilingSummarizer()
        filing = sec.fetch_and_summarize_10k("AAPL", year=2023)
        print(f"Filing date:    {filing.filed_date}")
        print(f"Accession:      {filing.accession_number}")
        print(f"Text length:    {filing.full_text_length:,} chars")
        print(f"\nTop risks:")
        for i, risk in enumerate(filing.risks[:3], 1):
            if isinstance(risk, dict):
                print(f"  {i}. {risk.get('text','')[:120]}")
            else:
                print(f"  {i}. {str(risk)[:120]}")
        print(f"\nExecutive summary:\n  {filing.executive_summary[:400]}")
    except Exception as e:
        print(f"  Error: {e}")

    # 2. Earnings call
    print("\n[2] AAPL Last Earnings Call Transcript")
    print("-" * 40)
    try:
        earner = EarningsCallSummarizer()
        year = datetime.now().year
        q = (datetime.now().month - 1) // 3 + 1
        quarter = f"Q{max(1,q-1)} {year}"
        transcript = earner.fetch_transcript("AAPL", quarter)
        if transcript:
            summary = earner.summarize_transcript(transcript)
            print(f"Quarter:       {quarter}")
            print(f"Tone:          {summary.management_tone}")
            print(f"Revenue guide: {summary.revenue_guidance[:100] or 'N/A'}")
            print(f"EPS guide:     {summary.eps_guidance[:100] or 'N/A'}")
            print(f"Surprise lang: {summary.surprise_language[:2]}")
            print(f"\nNarrative:\n  {summary.narrative[:400]}")
        else:
            print(f"  No transcript found for AAPL {quarter}")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. Three-sentence brief
    print("\n[3] AAPL 3-Sentence Brief")
    print("-" * 40)
    try:
        brief_summary = engine.summarize_ticker("AAPL", depth="brief")
        print(brief_summary.executive_brief or "(no brief generated)")
        if brief_summary.catalysts:
            print(f"\nTop catalyst: {brief_summary.catalysts[0].headline[:120]}")
        print(f"News sentiment: {brief_summary.sentiment_trend}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Done.")
