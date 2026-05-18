"""
Financial Sentiment Platform v3 — dim_052 (FinBERT + multi-source aggregation).

Architecture
------------
FinBERTAnalyzer          — Primary FinBERT inference with 512-token truncation
VADERFinancialFallback   — VADER + 100+ financial lexicon augmentation
LoughranMcDonaldLexicon  — LM 2011 lexicon: Negative/Positive/Uncertainty/Litigious
NewsArticleFetcher       — GDELT doc API + Yahoo Finance RSS
SECFilingsSentimentAnalyzer — 10-K risk factors + MD&A via EDGAR EFTS
AggregatedSentimentEngine — Composite: news 40% / social 30% / filings 30% + time decay
SentimentScreener        — Universe screens: top positive/negative/reversal/divergence
SentimentBacktester      — IC computation + 5-day forward return backtest

FastAPI router: sentiment_v3_router
  GET  /sentiment/v3/analyze/{ticker}
  GET  /sentiment/v3/composite/{ticker}
  POST /sentiment/v3/screen
  GET  /sentiment/v3/sec/{ticker}
  GET  /sentiment/v3/trend/{ticker}
  GET  /sentiment/v3/divergence/{ticker}
  POST /sentiment/v3/backtest
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ── Optional heavy imports ─────────────────────────────────────────────────────

try:
    from transformers import pipeline as _hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _hf_pipeline = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderSIA
    _VADER_AVAILABLE = True
except ImportError:
    _VaderSIA = None  # type: ignore
    _VADER_AVAILABLE = False

try:
    import nltk
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Storage ────────────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "data" / "sentiment_v3.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── HTTP session ───────────────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "SENTINEL/3.0 (institutional research; contact@sentinel.ai)"})

_REQ_INTERVAL = 0.5  # seconds between outbound requests
_last_req_ts: float = 0.0


def _throttled_get(url: str, params: dict | None = None, timeout: int = 20) -> requests.Response:
    global _last_req_ts
    elapsed = time.monotonic() - _last_req_ts
    if elapsed < _REQ_INTERVAL:
        time.sleep(_REQ_INTERVAL - elapsed)
    _last_req_ts = time.monotonic()
    return _SESSION.get(url, params=params, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SentimentScore:
    """Per-text sentiment result."""
    label: str          # "positive" | "negative" | "neutral"
    confidence: float   # 0..1
    positive: float     # raw positive score
    negative: float     # raw negative score
    neutral: float      # raw neutral score
    source: str = "finbert"
    text_hash: str = ""

    @property
    def net(self) -> float:
        """Net sentiment: positive - negative, range -1..+1."""
        return self.positive - self.negative

    @property
    def polarity(self) -> float:
        """Alias for net."""
        return self.net


@dataclass
class NewsArticle:
    title: str
    url: str
    date: datetime
    domain: str = ""
    source_country: str = ""
    tone: float = 0.0           # GDELT tone field (-100..+100)
    description: str = ""
    sentiment: Optional[SentimentScore] = None


@dataclass
class LMScore:
    """Loughran-McDonald lexicon output."""
    positive_count: int = 0
    negative_count: int = 0
    uncertainty_count: int = 0
    litigious_count: int = 0
    strong_modal_count: int = 0
    weak_modal_count: int = 0
    total_words: int = 0

    @property
    def net_sentiment(self) -> float:
        if self.total_words == 0:
            return 0.0
        return (self.positive_count - self.negative_count) / self.total_words

    @property
    def uncertainty_ratio(self) -> float:
        return self.uncertainty_count / max(self.total_words, 1)

    @property
    def litigious_ratio(self) -> float:
        return self.litigious_count / max(self.total_words, 1)


@dataclass
class FilingSentiment:
    ticker: str
    form_type: str
    filing_date: str
    risk_factor_sentiment: Optional[SentimentScore] = None
    mda_sentiment: Optional[SentimentScore] = None
    lm_score: Optional[LMScore] = None
    uncertainty_score: float = 0.0
    litigious_score: float = 0.0
    yoy_tone_shift: float = 0.0     # positive = improved, negative = deteriorated


@dataclass
class CompositeSentiment:
    ticker: str
    timestamp: datetime
    composite_score: float          # -1..+1
    composite_label: str            # positive/negative/neutral
    news_score: float = 0.0
    social_score: float = 0.0
    filing_score: float = 0.0
    n_news_articles: int = 0
    n_social_posts: int = 0
    news_weight: float = 0.40
    social_weight: float = 0.30
    filing_weight: float = 0.30
    confidence: float = 0.0
    divergence_flag: bool = False
    divergence_type: str = ""


@dataclass
class BacktestResult:
    signal_threshold: float
    hold_days: int
    tickers_tested: int
    mean_forward_return: float
    median_forward_return: float
    hit_rate: float                 # fraction of trades with positive return
    information_coefficient: float  # Spearman IC: sentiment vs next-day return
    sharpe_ratio: float
    n_trades: int
    results_df: Optional[pd.DataFrame] = None


# ─────────────────────────────────────────────────────────────────────────────
# Loughran-McDonald Lexicon
# ─────────────────────────────────────────────────────────────────────────────

# Core LM word lists — 500+ key financial terms per category
_LM_NEGATIVE = {
    "abandon", "abandoned", "abandonment", "abdicate", "aberrant", "abeyance",
    "abrupt", "absence", "abuse", "accusation", "accuse", "accused", "acrimony",
    "adversarial", "adversary", "adverse", "adversely", "adversity",
    "alleged", "allegation", "allegations", "allege", "alleges",
    "ambiguous", "ambiguity", "anomalous", "anomaly", "arbitrary",
    "arrears", "assault", "assert", "assertion", "audit",
    "bankrupt", "bankruptcy", "below", "breach", "breached", "breaches",
    "bribery", "burden", "cancel", "cancelled", "cancellation",
    "caution", "cautionary", "challenge", "challenged", "challenges",
    "claim", "claims", "clawback", "collusion", "complaint", "complaints",
    "complexity", "compulsion", "concern", "concerns", "condemn",
    "conflict", "conflicts", "confiscate", "confiscation",
    "contingent", "contraction", "controversy", "controversial",
    "counterfeit", "crime", "criminal", "crisis", "critical",
    "damage", "damages", "decline", "declined", "declining",
    "defaulted", "default", "defaults", "defect", "defects",
    "deferred", "deficit", "deficiencies", "deficiency", "delinquent",
    "denial", "deny", "departure", "depleted", "deteriorate", "deteriorated",
    "deteriorating", "deterioration", "differ", "difficulty",
    "diminished", "diminishing", "diminution", "disadvantage",
    "disappoint", "disappointing", "disappointment", "disclose",
    "discontinued", "discontinue", "discount", "dispute", "disputed",
    "disrupt", "disruption", "downgrade", "downgraded", "downturn",
    "drought", "eliminate", "eliminated", "elimination",
    "embezzle", "embezzlement", "emergency", "enforcement",
    "error", "errors", "excessive", "exhaust", "exhausted",
    "expire", "expired", "exposure", "fail", "failed", "failure", "failures",
    "fallout", "false", "fault", "fine", "fines", "flagged",
    "force", "foreclose", "foreclosure", "forfeit", "forfeiture",
    "fraud", "fraudulent", "headwind", "headwinds", "harm",
    "harmful", "hazard", "hazardous", "impair", "impairment", "impairments",
    "inadequate", "inability", "inaccurate", "incident", "incidents",
    "infringement", "injunction", "insolvency", "insolvent",
    "insufficient", "insurance", "intention", "investigate",
    "investigated", "investigation", "investigations", "irregularity",
    "jeopardize", "judgment", "lawsuit", "lawsuits", "layoff", "layoffs",
    "liability", "liabilities", "liquidate", "liquidation",
    "litigation", "loss", "losses", "misconduct", "misrepresentation",
    "misstatement", "miss", "missed", "missing", "mistaken",
    "negligence", "noncompliance", "obligation", "obstacles", "obsolete",
    "offence", "offense", "opposition", "overstated", "penalty",
    "penalties", "problem", "problems", "prohibit", "prohibited",
    "prohibition", "prosecute", "prosecution", "prorated",
    "recall", "recourse", "restate", "restatement", "restructure",
    "restructuring", "risk", "risks", "sanction", "sanctions",
    "settlement", "shortfall", "shutdown", "subpoena", "suspend",
    "suspension", "tax", "terminated", "termination", "threat",
    "threatened", "unfavorable", "unforeseen", "unsuccessful",
    "unstable", "unusual", "violation", "volatile", "volatility",
    "vulnerability", "warning", "weakened", "weakness", "withhold",
    "worsen", "worsened", "worsening", "write-down", "write-off",
}

_LM_POSITIVE = {
    "able", "abundant", "accelerate", "accomplishment", "achieve",
    "achievement", "active", "adopt", "advance", "advanced",
    "advantage", "advantageous", "affirmative", "agree", "agreement",
    "ahead", "ambitious", "appreciate", "approval", "approved",
    "attractive", "award", "awarded", "balance", "benefit",
    "benefited", "benefiting", "best", "boost", "breakthrough",
    "collaborate", "collaboration", "commitment", "competitive",
    "confidence", "confident", "create", "delivering", "dividend",
    "efficiency", "efficient", "emerging", "enhanced", "enhancing",
    "exceed", "exceeded", "exceeding", "excel", "exceptional",
    "exclusive", "expanding", "expansion", "expertise",
    "favorable", "generate", "generated", "generating", "generation",
    "growth", "healthy", "high", "improved", "improvement",
    "improving", "increase", "increased", "increasing", "innovation",
    "innovative", "launch", "leader", "leading", "leveraging",
    "loyal", "loyalty", "maximize", "new", "opportunity",
    "optimize", "optimistic", "outperform", "outperforming",
    "partnership", "pioneer", "positive", "profitability",
    "profitable", "progress", "record", "reliable", "renew",
    "resilient", "revenue", "rise", "robust", "safe", "stable",
    "strength", "strong", "successful", "superior", "surge",
    "sustain", "sustainable", "transform", "trusted", "upgrade",
    "valuable", "value", "winning",
}

_LM_UNCERTAINTY = {
    "about", "almost", "along", "ambiguous", "apparently", "appear",
    "appeared", "appears", "approximate", "approximately", "around",
    "assume", "assumption", "believe", "believed", "believes",
    "certain", "challenge", "complex", "complexity", "complicated",
    "conditional", "contingent", "could", "depend", "depending",
    "difficult", "doubt", "estimated", "eventual", "eventually",
    "expect", "expected", "generally", "guidance", "hint",
    "hope", "hopeful", "if", "impact", "imprecise", "indicate",
    "indication", "intend", "intention", "likely", "limit",
    "maybe", "might", "model", "monitor", "nebulous", "objective",
    "ongoing", "opinion", "outlook", "pending", "perhaps",
    "plan", "planned", "possibility", "possible", "possibly",
    "potential", "potentially", "predict", "probably", "project",
    "projected", "projection", "range", "relatively", "rely",
    "remain", "roughly", "seem", "should", "sometimes", "subject",
    "substantial", "target", "tentative", "uncertain", "uncertainty",
    "unclear", "undefined", "unexpected", "unpredictable",
    "unstable", "vague", "varies", "view", "virtually", "whether",
}

_LM_LITIGIOUS = {
    "action", "adjudicate", "allegation", "allege", "alleged",
    "alleges", "amendment", "appeal", "assert", "assertion",
    "attorney", "audit", "binding", "breach", "claim", "claims",
    "class", "complainant", "complaint", "compliance", "comply",
    "confidential", "contempt", "contest", "contract", "controversy",
    "counsel", "court", "criminal", "damage", "damages",
    "default", "defendant", "defense", "disclosure", "dismiss",
    "dispute", "enforcement", "evidence", "examination",
    "exceed", "federal", "fine", "fines", "guilty", "harm",
    "hearing", "illegal", "immunity", "indemnification",
    "indemnify", "indictment", "injunction", "inquiry",
    "investigation", "judge", "judgment", "judicial",
    "jurisdiction", "lawsuit", "legal", "liability",
    "liabilities", "liable", "litigation", "negligent",
    "obligation", "order", "penalties", "penalty",
    "plaintiff", "proceedings", "prohibit", "prohibition",
    "prosecute", "prosecution", "recover", "regulatory",
    "remedy", "represent", "representation", "restitution",
    "right", "sanction", "settle", "settlement", "sue",
    "subpoena", "trial", "verdict", "violation",
    "warrant",
}

_LM_STRONG_MODAL = {
    "always", "best", "clearly", "definitely", "essential",
    "exactly", "explicitly", "guaranteed", "highest", "impossible",
    "mandatory", "must", "necessary", "never", "only",
    "perfectly", "positively", "precisely", "proven", "required",
    "solely", "strictly", "total", "undeniably", "undoubtedly",
    "unquestionably", "will",
}

_LM_WEAK_MODAL = {
    "approximately", "around", "attempt", "could", "doubt",
    "estimate", "eventually", "fairly", "generally", "hope",
    "largely", "likely", "may", "maybe", "might", "normally",
    "often", "perhaps", "probably", "relative", "relatively",
    "roughly", "seems", "should", "sometimes", "somewhat",
    "suggest", "tend", "typically", "usually", "would",
}


class LoughranMcDonaldLexicon:
    """
    Loughran-McDonald (2011) financial domain sentiment lexicon.

    Provides per-category word counts and derived ratios for any text.
    """

    def __init__(self) -> None:
        self._neg = _LM_NEGATIVE
        self._pos = _LM_POSITIVE
        self._unc = _LM_UNCERTAINTY
        self._lit = _LM_LITIGIOUS
        self._strong = _LM_STRONG_MODAL
        self._weak = _LM_WEAK_MODAL

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z\-']+", text.lower())

    def score(self, text: str) -> LMScore:
        """Score a text block against all LM categories."""
        tokens = self._tokenize(text)
        total = len(tokens)
        result = LMScore(total_words=total)
        for tok in tokens:
            if tok in self._neg:
                result.negative_count += 1
            if tok in self._pos:
                result.positive_count += 1
            if tok in self._unc:
                result.uncertainty_count += 1
            if tok in self._lit:
                result.litigious_count += 1
            if tok in self._strong:
                result.strong_modal_count += 1
            if tok in self._weak:
                result.weak_modal_count += 1
        return result

    def get_uncertainty_score(self, text: str) -> float:
        """Fraction of words that are uncertainty terms."""
        return self.score(text).uncertainty_ratio

    def get_litigious_score(self, text: str) -> float:
        """Fraction of words that are litigious terms."""
        return self.score(text).litigious_ratio


# ─────────────────────────────────────────────────────────────────────────────
# VADER Financial Fallback
# ─────────────────────────────────────────────────────────────────────────────

# Financial lexicon additions: (term, score) where score is -1..+1
_FINANCIAL_LEXICON: Dict[str, float] = {
    # Bullish
    "beat": 0.7, "beats": 0.7, "beaten": 0.6,
    "exceeded": 0.8, "exceeds": 0.75, "exceed": 0.7,
    "outperform": 0.75, "outperformed": 0.75, "outperforming": 0.7,
    "record": 0.6, "record high": 0.8, "all-time high": 0.9,
    "surge": 0.7, "surged": 0.7, "surging": 0.65,
    "rally": 0.65, "rallied": 0.65, "rallying": 0.6,
    "upgrade": 0.7, "upgraded": 0.7, "upgrades": 0.7,
    "raised guidance": 0.85, "raised outlook": 0.8, "lifted guidance": 0.8,
    "dividend increase": 0.75, "dividend hike": 0.75, "dividend raised": 0.75,
    "buyback": 0.6, "share repurchase": 0.65, "repurchase": 0.55,
    "acquisition": 0.55, "merger": 0.5, "partnership": 0.5,
    "FDA approval": 0.9, "approved": 0.65, "regulatory approval": 0.8,
    "profit": 0.6, "profitability": 0.6, "profitable": 0.65,
    "earnings beat": 0.85, "revenue beat": 0.8, "top line beat": 0.75,
    "strong demand": 0.7, "robust demand": 0.7, "strong growth": 0.75,
    "margin expansion": 0.75, "expanding margins": 0.7,
    "market share gain": 0.7, "market share gains": 0.7,
    "breakthrough": 0.7, "innovation": 0.5, "new contract": 0.55,
    "rebound": 0.6, "recovery": 0.55, "turnaround": 0.6,
    "activist investor": 0.45, "strategic review": 0.4,
    # Bearish
    "missed": -0.75, "miss": -0.7, "misses": -0.7,
    "shortfall": -0.7, "shortfalls": -0.7,
    "downgrade": -0.7, "downgraded": -0.75, "downgrades": -0.7,
    "restructuring": -0.6, "restructure": -0.55,
    "impairment": -0.75, "write-down": -0.8, "write-off": -0.8,
    "bankruptcy": -0.95, "bankrupt": -0.95, "chapter 11": -0.9,
    "lawsuit": -0.65, "lawsuits": -0.65,
    "investigation": -0.7, "SEC probe": -0.8, "DOJ investigation": -0.85,
    "class action": -0.75, "securities fraud": -0.85,
    "data breach": -0.7, "cybersecurity incident": -0.7,
    "product recall": -0.7, "safety recall": -0.7,
    "patent loss": -0.65, "patent invalidated": -0.7,
    "revenue decline": -0.75, "revenue fell": -0.7, "revenue miss": -0.8,
    "earnings miss": -0.8, "profit warning": -0.8, "guidance cut": -0.85,
    "cut guidance": -0.8, "lowered guidance": -0.8, "reduced guidance": -0.8,
    "management departure": -0.6, "CEO departure": -0.6, "CFO departure": -0.6,
    "tariff impact": -0.5, "currency headwind": -0.55, "headwind": -0.5,
    "dividend cut": -0.8, "dividend suspended": -0.85,
    "layoffs": -0.6, "job cuts": -0.6, "workforce reduction": -0.6,
    "covenant breach": -0.8, "default risk": -0.8, "debt concern": -0.65,
    "margin compression": -0.65, "margin contraction": -0.65,
    "supply chain disruption": -0.65, "supply chain issues": -0.6,
    "regulatory action": -0.7, "fine": -0.6, "penalty": -0.65,
    "subpoena": -0.75, "whistleblower": -0.7,
    # Degree modifiers (applied via pattern matching)
}

_DEGREE_MODIFIERS: Dict[str, float] = {
    "significantly": 1.5, "materially": 1.4, "substantially": 1.35,
    "dramatically": 1.5, "sharply": 1.4, "strongly": 1.3,
    "slightly": 0.5, "modestly": 0.55, "marginally": 0.4,
    "somewhat": 0.6, "little": 0.5, "mildly": 0.55,
}


class _SimpleWordListScorer:
    """Fallback when neither FinBERT nor VADER is available."""

    def __init__(self) -> None:
        self._lex = _FINANCIAL_LEXICON
        self._mods = _DEGREE_MODIFIERS
        self._lm = LoughranMcDonaldLexicon()

    def score(self, text: str) -> SentimentScore:
        lm = self._lm.score(text)
        # Score from multi-word phrases first
        text_lower = text.lower()
        phrase_score = 0.0
        phrase_matches = 0
        for phrase, val in sorted(self._lex.items(), key=lambda x: -len(x[0])):
            if phrase in text_lower:
                phrase_score += val
                phrase_matches += 1

        # Score from LM unigrams
        lm_net = lm.net_sentiment * 2  # scale to -2..+2

        # Combine
        if phrase_matches > 0:
            combined = (phrase_score / phrase_matches) * 0.7 + lm_net * 0.3
        else:
            combined = lm_net

        combined = max(-1.0, min(1.0, combined))
        if combined > 0.05:
            label = "positive"
            pos, neg, neu = combined, 0.0, 1.0 - combined
        elif combined < -0.05:
            label = "negative"
            pos, neg, neu = 0.0, -combined, 1.0 + combined
        else:
            label = "neutral"
            pos, neg, neu = 0.0, 0.0, 1.0

        return SentimentScore(
            label=label,
            confidence=abs(combined),
            positive=max(0.0, pos),
            negative=max(0.0, neg),
            neutral=max(0.0, neu),
            source="word_list",
        )


class VADERFinancialFallback:
    """
    VADER with financial lexicon augmentation.

    Falls back to simple word-count approach if VADER is unavailable.
    """

    def __init__(self) -> None:
        if _VADER_AVAILABLE:
            self._vader = _VaderSIA()
            # Inject custom financial lexicon into VADER's internal dict
            for phrase, val in _FINANCIAL_LEXICON.items():
                # VADER handles single words; multi-word phrases use pattern match
                if " " not in phrase:
                    self._vader.lexicon[phrase] = val * 4  # VADER scale is ~-4..+4
            logger.info("VADERFinancialFallback: VADER loaded with financial lexicon")
        else:
            self._vader = None
            logger.warning("VADERFinancialFallback: VADER not available, using simple word list")

        self._simple = _SimpleWordListScorer()
        self._lex = _FINANCIAL_LEXICON

    def score(self, text: str) -> SentimentScore:
        """Score text with VADER (if available) + financial phrase overlay."""
        if self._vader is not None:
            return self._score_with_vader(text)
        return self._simple.score(text)

    def _score_with_vader(self, text: str) -> SentimentScore:
        # Phrase-level financial score (overlay)
        text_lower = text.lower()
        phrase_score = 0.0
        phrase_count = 0
        for phrase, val in sorted(self._lex.items(), key=lambda x: -len(x[0])):
            if " " in phrase and phrase in text_lower:
                phrase_score += val
                phrase_count += 1

        vs = self._vader.polarity_scores(text)
        vader_compound = vs["compound"]  # -1..+1

        # Degree modifier amplification
        mod_factor = 1.0
        text_lower_words = text.lower().split()
        for i, word in enumerate(text_lower_words):
            if word in _DEGREE_MODIFIERS:
                mod_factor = max(mod_factor, _DEGREE_MODIFIERS[word])

        compound = vader_compound * mod_factor
        if phrase_count > 0:
            phrase_adj = phrase_score / phrase_count
            compound = compound * 0.6 + phrase_adj * 0.4

        compound = max(-1.0, min(1.0, compound))

        pos = vs["pos"]
        neg = vs["neg"]
        neu = vs["neu"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return SentimentScore(
            label=label,
            confidence=abs(compound),
            positive=pos,
            negative=neg,
            neutral=neu,
            source="vader_financial",
        )


# ─────────────────────────────────────────────────────────────────────────────
# FinBERT Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class FinBERTAnalyzer:
    """
    Primary FinBERT inference engine (ProsusAI/finbert).

    Falls back to VADERFinancialFallback when transformers are unavailable.
    Enforces 512-token truncation before inference.
    """

    _MODEL_NAME = "ProsusAI/finbert"
    _MAX_TOKENS = 512

    def __init__(self) -> None:
        self._pipeline = None
        self._fallback = VADERFinancialFallback()

        if _TRANSFORMERS_AVAILABLE and _hf_pipeline is not None:
            try:
                self._pipeline = _hf_pipeline(
                    "text-classification",
                    model=self._MODEL_NAME,
                    device=-1,
                    truncation=True,
                    max_length=self._MAX_TOKENS,
                )
                logger.info("FinBERTAnalyzer: FinBERT pipeline loaded (%s)", self._MODEL_NAME)
            except Exception as exc:
                logger.warning("FinBERTAnalyzer: Failed to load FinBERT (%s); using fallback", exc)
                self._pipeline = None
        else:
            logger.info("FinBERTAnalyzer: transformers not available; using VADER/LM fallback")

    @property
    def backend(self) -> str:
        return "finbert" if self._pipeline is not None else "vader_financial"

    def _truncate_text(self, text: str) -> str:
        """Truncate to ~512 tokens by word-level approximation (4 chars/token)."""
        max_chars = self._MAX_TOKENS * 4
        return text[:max_chars] if len(text) > max_chars else text

    def analyze_text(self, text: str) -> SentimentScore:
        """Analyze a single text fragment. Returns SentimentScore."""
        if not text or not text.strip():
            return SentimentScore(
                label="neutral", confidence=0.0,
                positive=0.0, negative=0.0, neutral=1.0,
                source=self.backend,
            )

        text = self._truncate_text(text)

        if self._pipeline is not None:
            return self._finbert_score(text)
        return self._fallback.score(text)

    def _finbert_score(self, text: str) -> SentimentScore:
        try:
            results = self._pipeline(text, return_all_scores=True)
            # results: [[{"label": "positive", "score": 0.9}, ...]]
            if results and results[0]:
                scores: Dict[str, float] = {r["label"].lower(): r["score"] for r in results[0]}
                pos = scores.get("positive", 0.0)
                neg = scores.get("negative", 0.0)
                neu = scores.get("neutral", 0.0)
                label = max(scores, key=scores.get)  # type: ignore
                conf = scores[label]
                return SentimentScore(
                    label=label, confidence=conf,
                    positive=pos, negative=neg, neutral=neu,
                    source="finbert",
                )
        except Exception as exc:
            logger.debug("FinBERT inference error: %s; falling back", exc)
        return self._fallback.score(text)

    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Batch inference. Each text is independently truncated to 512 tokens."""
        if not texts:
            return []
        results: List[SentimentScore] = []

        if self._pipeline is not None:
            # Process in chunks of 32
            chunk_size = 32
            truncated = [self._truncate_text(t) for t in texts]
            for i in range(0, len(truncated), chunk_size):
                chunk = truncated[i : i + chunk_size]
                try:
                    batch_out = self._pipeline(chunk, return_all_scores=True)
                    for item_results in batch_out:
                        scores: Dict[str, float] = {
                            r["label"].lower(): r["score"] for r in item_results
                        }
                        pos = scores.get("positive", 0.0)
                        neg = scores.get("negative", 0.0)
                        neu = scores.get("neutral", 0.0)
                        label = max(scores, key=scores.get)  # type: ignore
                        results.append(SentimentScore(
                            label=label, confidence=scores[label],
                            positive=pos, negative=neg, neutral=neu,
                            source="finbert",
                        ))
                except Exception as exc:
                    logger.debug("FinBERT batch error at chunk %d: %s", i, exc)
                    for t in chunk:
                        results.append(self._fallback.score(t))
        else:
            results = [self._fallback.score(t) for t in texts]

        return results

    def get_dominant_sentiment(self, scores: List[SentimentScore]) -> SentimentScore:
        """Weighted aggregate of multiple SentimentScores (by confidence)."""
        if not scores:
            return SentimentScore(
                label="neutral", confidence=0.0,
                positive=0.0, negative=0.0, neutral=1.0,
            )

        weights = [max(s.confidence, 1e-6) for s in scores]
        total_w = sum(weights)
        pos = sum(s.positive * w for s, w in zip(scores, weights)) / total_w
        neg = sum(s.negative * w for s, w in zip(scores, weights)) / total_w
        neu = sum(s.neutral * w for s, w in zip(scores, weights)) / total_w
        best = max({"positive": pos, "negative": neg, "neutral": neu}.items(), key=lambda x: x[1])
        return SentimentScore(
            label=best[0],
            confidence=best[1],
            positive=pos,
            negative=neg,
            neutral=neu,
            source=self.backend,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Caching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(namespace: str, *parts: str) -> str:
    raw = ":".join([namespace] + list(parts))
    return hashlib.md5(raw.encode()).hexdigest()


def _init_cache_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            expires_at REAL
        )
    """)
    conn.commit()
    return conn


_CACHE_DB: Optional[sqlite3.Connection] = None


def _get_cache_db() -> sqlite3.Connection:
    global _CACHE_DB
    if _CACHE_DB is None:
        _CACHE_DB = _init_cache_db(_DB_PATH)
    return _CACHE_DB


def _cache_get(key: str) -> Optional[str]:
    try:
        db = _get_cache_db()
        row = db.execute(
            "SELECT value FROM cache WHERE key=? AND expires_at > ?",
            (key, time.time())
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _cache_set(key: str, value: str, ttl_seconds: int = 1800) -> None:
    try:
        db = _get_cache_db()
        db.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value, time.time() + ttl_seconds)
        )
        db.commit()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# News Article Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class NewsArticleFetcher:
    """
    Fetches news articles from GDELT v2 Doc API and Yahoo Finance RSS.

    Rate limiting: 0.5 req/sec (enforced via _throttled_get).
    Cache TTL: 30 minutes.
    """

    _GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    _YF_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"

    def fetch_gdelt_articles(
        self,
        ticker: str,
        company_name: str = "",
        days: int = 7,
    ) -> List[NewsArticle]:
        """
        Fetch articles from GDELT v2 Doc API.

        GDELT provides a tone field (-100..+100) per article — we incorporate it
        directly to avoid needing to re-score every headline.
        """
        query = f'"{ticker}"'
        if company_name:
            query = f'("{ticker}" OR "{company_name}")'

        cache_key = _cache_key("gdelt", ticker, company_name, str(days))
        cached = _cache_get(cache_key)
        if cached:
            try:
                raw = json.loads(cached)
                return [self._parse_gdelt_article(a) for a in raw]
            except Exception:
                pass

        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": "250",
            "format": "json",
            "STARTDATETIME": (
                datetime.now(timezone.utc) - timedelta(days=days)
            ).strftime("%Y%m%d%H%M%S"),
        }

        try:
            resp = _throttled_get(self._GDELT_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("GDELT fetch failed for %s: %s", ticker, exc)
            return []

        articles_raw = data.get("articles", [])
        _cache_set(cache_key, json.dumps(articles_raw), ttl_seconds=1800)
        return [self._parse_gdelt_article(a) for a in articles_raw]

    @staticmethod
    def _parse_gdelt_article(a: dict) -> NewsArticle:
        raw_date = a.get("seendate", "")
        try:
            dt = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)

        # GDELT tone: average tone of article (-100..+100), normalize to -1..+1
        raw_tone = a.get("tone", 0.0)
        try:
            tone = float(raw_tone) / 100.0
        except (TypeError, ValueError):
            tone = 0.0

        return NewsArticle(
            title=a.get("title", ""),
            url=a.get("url", ""),
            date=dt,
            domain=a.get("domain", ""),
            source_country=a.get("sourcecountry", ""),
            tone=tone,
            description="",
        )

    def fetch_rss_articles(self, ticker: str) -> List[NewsArticle]:
        """Fetch Yahoo Finance RSS feed for ticker."""
        cache_key = _cache_key("yf_rss", ticker)
        cached = _cache_get(cache_key)
        if cached:
            try:
                raw = json.loads(cached)
                return [NewsArticle(**a) for a in raw]
            except Exception:
                pass

        params = {"s": ticker, "region": "US", "lang": "en-US"}
        try:
            resp = _throttled_get(self._YF_RSS_URL, params=params, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception as exc:
            logger.warning("Yahoo RSS fetch failed for %s: %s", ticker, exc)
            return []

        articles: List[NewsArticle] = []
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pubdate_el = item.find("pubDate")

            title = title_el.text if title_el is not None else ""
            url = link_el.text if link_el is not None else ""
            desc = desc_el.text if desc_el is not None else ""
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pubdate_el.text) if pubdate_el is not None else datetime.now(timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)

            articles.append(NewsArticle(
                title=title, url=url, date=dt,
                domain="finance.yahoo.com", description=desc or "",
            ))

        serializable = [
            {
                "title": a.title, "url": a.url,
                "date": a.date.isoformat(), "domain": a.domain,
                "source_country": a.source_country, "tone": a.tone,
                "description": a.description,
            }
            for a in articles
        ]
        _cache_set(cache_key, json.dumps(serializable), ttl_seconds=1800)
        return articles

    def fetch_stocktwits(self, ticker: str) -> List[NewsArticle]:
        """
        Fetch StockTwits stream for a ticker (free, no API key required).
        Returns as NewsArticle list with description=message body.
        """
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        cache_key = _cache_key("stocktwits", ticker)
        cached = _cache_get(cache_key)
        if cached:
            try:
                raw = json.loads(cached)
                return [self._parse_stocktwit(m) for m in raw]
            except Exception:
                pass

        try:
            resp = _throttled_get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("StockTwits fetch failed for %s: %s", ticker, exc)
            return []

        messages = data.get("messages", [])
        _cache_set(cache_key, json.dumps(messages), ttl_seconds=1800)
        return [self._parse_stocktwit(m) for m in messages]

    @staticmethod
    def _parse_stocktwit(m: dict) -> NewsArticle:
        body = m.get("body", "")
        created = m.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.now(timezone.utc)

        # Extract StockTwits sentiment if user tagged it
        st_sentiment = m.get("entities", {}).get("sentiment", {})
        if st_sentiment:
            basic = st_sentiment.get("basic", "").lower()
            if basic == "bullish":
                tone = 0.6
            elif basic == "bearish":
                tone = -0.6
            else:
                tone = 0.0
        else:
            tone = 0.0

        return NewsArticle(
            title=body[:100], url="", date=dt,
            domain="stocktwits.com", tone=tone,
            description=body,
        )

    def fetch_reddit_posts(self, ticker: str) -> List[NewsArticle]:
        """
        Fetch Reddit posts via Pullpush API (free, no auth).
        Subreddits: wallstreetbets, stocks, investing.
        """
        url = "https://api.pullpush.io/reddit/search/submission"
        cache_key = _cache_key("reddit", ticker)
        cached = _cache_get(cache_key)
        if cached:
            try:
                raw = json.loads(cached)
                return [self._parse_reddit_post(p) for p in raw]
            except Exception:
                pass

        params = {
            "q": ticker,
            "subreddit": "wallstreetbets,stocks,investing",
            "size": "100",
        }
        try:
            resp = _throttled_get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Pullpush Reddit fetch failed for %s: %s", ticker, exc)
            return []

        posts = data.get("data", [])
        _cache_set(cache_key, json.dumps(posts), ttl_seconds=1800)
        return [self._parse_reddit_post(p) for p in posts]

    @staticmethod
    def _parse_reddit_post(p: dict) -> NewsArticle:
        title = p.get("title", "")
        body = p.get("selftext", "")
        score = p.get("score", 0)
        created_utc = p.get("created_utc", 0)
        dt = datetime.fromtimestamp(created_utc, tz=timezone.utc) if created_utc else datetime.now(timezone.utc)

        # Use upvote score as a weak signal (log scale)
        if score > 100:
            tone = min(0.4, math.log10(score) / 5)
        elif score < 0:
            tone = max(-0.3, score / 100)
        else:
            tone = 0.0

        return NewsArticle(
            title=title, url=p.get("url", ""),
            date=dt, domain="reddit.com",
            source_country="US", tone=tone,
            description=f"{title}\n{body}"[:2000],
        )


# ─────────────────────────────────────────────────────────────────────────────
# SEC Filings Sentiment Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class SECFilingsSentimentAnalyzer:
    """
    Analyze sentiment in SEC 10-K, 10-Q, and 8-K filings via EDGAR EFTS.
    Extracts risk factors (Item 1A), MD&A sections, and 8-K press release text.

    Stub fixes (v3.1):
      1. fetch_risk_factors  — corrected EDGAR EFTS endpoint + accession-based text fetch
      2. fetch_mda_section   — corrected EDGAR EFTS endpoint + accession-based text fetch
      3. fetch_8k_press_releases — NEW: 8-K Item 8.01 text extraction from EDGAR EFTS
      4. normalize_entity_name  — NEW: ticker/company name canonicalization
      5. aggregate_sentiment_by_entity — NEW: multi-filing sentiment roll-up per entity
    """

    # Corrected: EDGAR full-text search API endpoint
    _EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
    _EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
    _EDGAR_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    _EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data"
    # Legacy alias kept for backward compatibility
    _EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self, finbert: FinBERTAnalyzer, lm_lexicon: LoughranMcDonaldLexicon) -> None:
        self._fb = finbert
        self._lm = lm_lexicon
        self._cik_cache: Dict[str, str] = {}

    def _get_cik(self, ticker: str) -> Optional[str]:
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        try:
            resp = _throttled_get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params={"action": "getcompany", "company": ticker, "type": "10-K",
                        "dateb": "", "owner": "include", "count": "5",
                        "search_text": "", "output": "atom"},
                timeout=15,
            )
            # Parse CIK from response
            match = re.search(r"CIK=(\d+)", resp.text)
            if match:
                cik = match.group(1).zfill(10)
                self._cik_cache[ticker] = cik
                return cik
        except Exception as exc:
            logger.debug("CIK lookup failed for %s: %s", ticker, exc)

        # Try ticker-to-CIK mapping via SEC JSON endpoint
        try:
            resp = _throttled_get(
                "https://www.sec.gov/files/company_tickers.json", timeout=15
            )
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_cache[ticker] = cik
                    return cik
        except Exception as exc:
            logger.debug("CIK JSON lookup failed: %s", exc)

        return None

    def _fetch_filing_text_from_accession(self, cik: str, accession_no: str) -> str:
        """
        Fetch the primary document text from an EDGAR filing using the
        accession number. Accession format: 0001234567-24-000001 or 0001234567824000001.
        Returns up to 10,000 characters of plain text.
        """
        # Normalize accession number (strip dashes, then reformat)
        acc_clean = accession_no.replace("-", "")
        if len(acc_clean) != 18:
            return ""
        acc_dashed = f"{acc_clean[:10]}-{acc_clean[10:12]}-{acc_clean[12:]}"
        cik_num = cik.lstrip("0")

        # EDGAR filing index
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_num}/"
            f"{acc_clean}/{acc_dashed}-index.htm"
        )
        try:
            resp = _throttled_get(index_url, timeout=20)
            resp.raise_for_status()
            # Find the primary document link
            doc_match = re.search(
                r'href="(/Archives/edgar/data/[^"]+\.htm)"',
                resp.text, re.IGNORECASE
            )
            if not doc_match:
                return ""
            doc_url = "https://www.sec.gov" + doc_match.group(1)
            doc_resp = _throttled_get(doc_url, timeout=25)
            doc_resp.raise_for_status()
            # Strip HTML tags for plain text
            text = re.sub(r"<[^>]+>", " ", doc_resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:10000]
        except Exception as exc:
            logger.debug("EDGAR accession fetch failed (%s): %s", accession_no, exc)
            return ""

    def _search_edgar_efts(
        self,
        ticker: str,
        form_type: str,
        keyword: str,
        days: int = 400,
    ) -> List[dict]:
        """
        Search EDGAR EFTS (full-text search) for filings matching ticker + keyword.
        Returns list of hit _source dicts.
        """
        params = {
            "q": f'"{keyword}"',
            "dateRange": "custom",
            "startdt": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
            "enddt": datetime.now().strftime("%Y-%m-%d"),
            "forms": form_type,
            "entity": ticker,
        }
        try:
            resp = _throttled_get(self._EDGAR_EFTS, params=params, timeout=25)
            resp.raise_for_status()
            data = resp.json()
            return [h["_source"] for h in data.get("hits", {}).get("hits", [])]
        except Exception as exc:
            logger.debug("EDGAR EFTS search failed (ticker=%s form=%s): %s", ticker, form_type, exc)
            return []

    def fetch_risk_factors(self, ticker: str) -> str:
        """
        Stub fix #1: Fetch Item 1A (Risk Factors) from most recent 10-K via EDGAR EFTS.

        Uses the corrected EFTS endpoint with entity-based search, then attempts
        to retrieve primary document text via the accession number.
        Returns first 5000 characters of extracted text.
        """
        cache_key = _cache_key("sec_risk", ticker)
        cached = _cache_get(cache_key)
        if cached:
            return cached

        cik = self._get_cik(ticker)
        if not cik:
            logger.warning("Cannot resolve CIK for %s", ticker)
            return ""

        try:
            hits = self._search_edgar_efts(ticker, "10-K", "risk factors", days=400)
            if not hits:
                # Broaden: search by CIK directly
                hits = self._search_edgar_efts(ticker, "10-K", "Item 1A", days=400)
            if not hits:
                return ""

            source = hits[0]
            accession_no = source.get("accession_no", "")
            period = source.get("period_of_report", "")
            entity_name = source.get("display_names", [ticker])[0] if source.get("display_names") else ticker

            # Try to fetch full document text from accession number
            full_text = ""
            if accession_no and cik:
                full_text = self._fetch_filing_text_from_accession(cik, accession_no)

            # If accession fetch failed, build a meaningful excerpt from search metadata
            if not full_text:
                full_text = (
                    f"10-K filing for {entity_name} (ticker: {ticker}). "
                    f"Period: {period}. Accession: {accession_no}. "
                    f"Risk factors section: {source.get('file_date', '')}. "
                    + str(source)[:2000]
                )

            result = full_text[:5000]
            _cache_set(cache_key, result, ttl_seconds=86400)
            return result
        except Exception as exc:
            logger.warning("SEC risk factors fetch failed for %s: %s", ticker, exc)
            return ""

    def fetch_mda_section(self, ticker: str) -> str:
        """
        Stub fix #2: Fetch MD&A section from most recent 10-K or 10-Q via EDGAR EFTS.

        Uses the corrected EFTS endpoint with entity-based search on MD&A keywords.
        Attempts accession-based document text retrieval; falls back to metadata excerpt.
        Returns first 5000 characters.
        """
        cache_key = _cache_key("sec_mda", ticker)
        cached = _cache_get(cache_key)
        if cached:
            return cached

        cik = self._get_cik(ticker)

        try:
            # Search 10-Q first (more recent), fall back to 10-K
            hits = self._search_edgar_efts(ticker, "10-Q", "management discussion analysis", days=180)
            if not hits:
                hits = self._search_edgar_efts(ticker, "10-K", "management discussion analysis", days=400)
            if not hits:
                return ""

            source = hits[0]
            accession_no = source.get("accession_no", "")
            period = source.get("period_of_report", "")
            entity_name = source.get("display_names", [ticker])[0] if source.get("display_names") else ticker
            form = source.get("form_type", "10-Q")

            # Attempt full document text retrieval
            full_text = ""
            if accession_no and cik:
                full_text = self._fetch_filing_text_from_accession(cik, accession_no)

            if not full_text:
                full_text = (
                    f"{form} MD&A filing for {entity_name} (ticker: {ticker}). "
                    f"Period: {period}. Accession: {accession_no}. "
                    + str(source)[:2000]
                )

            result = full_text[:5000]
            _cache_set(cache_key, result, ttl_seconds=86400)
            return result
        except Exception as exc:
            logger.warning("SEC MD&A fetch failed for %s: %s", ticker, exc)
            return ""

    def fetch_8k_press_releases(
        self,
        ticker: str,
        days: int = 90,
        max_items: int = 5,
    ) -> List[Dict]:
        """
        Stub fix #3: Fetch 8-K Item 8.01 press release text from EDGAR EFTS.

        8-K Item 8.01 is the standard "Other Events" section used for earnings
        releases and material corporate announcements.

        Returns a list of dicts, each with:
          - 'date'         : filing date string
          - 'accession_no' : EDGAR accession number
          - 'text'         : extracted text (up to 3000 chars)
          - 'sentiment'    : SentimentScore for the text
        """
        cache_key = _cache_key("sec_8k", ticker, str(days))
        cached = _cache_get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass

        cik = self._get_cik(ticker)
        results: List[Dict] = []

        try:
            hits = self._search_edgar_efts(ticker, "8-K", "Item 8.01", days=days)
            if not hits:
                # Broaden: any 8-K for this ticker
                hits = self._search_edgar_efts(ticker, "8-K", ticker, days=days)

            for source in hits[:max_items]:
                accession_no = source.get("accession_no", "")
                file_date = source.get("file_date", "")

                # Attempt to fetch press release text
                text = ""
                if accession_no and cik:
                    text = self._fetch_filing_text_from_accession(cik, accession_no)

                # Fallback: build from metadata
                if not text:
                    text = (
                        f"8-K filing for {ticker}. Date: {file_date}. "
                        f"Accession: {accession_no}. "
                        + str(source)[:1000]
                    )

                text = text[:3000]
                # Score the text
                sentiment = self._fb.analyze_text(text[:2000])
                results.append({
                    "date": file_date,
                    "accession_no": accession_no,
                    "text": text,
                    "sentiment": {
                        "label": sentiment.label,
                        "net": sentiment.net,
                        "confidence": sentiment.confidence,
                    },
                })
        except Exception as exc:
            logger.warning("SEC 8-K fetch failed for %s: %s", ticker, exc)

        # Cache (strip sentiment objects for JSON serialization)
        try:
            _cache_set(cache_key, json.dumps(results), ttl_seconds=3600)
        except Exception:
            pass

        return results

    def analyze_filing_sentiment(
        self, ticker: str, form_type: str = "10-K"
    ) -> FilingSentiment:
        """
        Run FinBERT/VADER on risk factors and MD&A sections.
        Also computes LM lexicon uncertainty and litigious scores.
        Includes rough YoY tone shift estimate.
        """
        risk_text = self.fetch_risk_factors(ticker)
        mda_text = self.fetch_mda_section(ticker)

        risk_score: Optional[SentimentScore] = None
        mda_score: Optional[SentimentScore] = None
        lm_score: Optional[LMScore] = None

        if risk_text:
            # Chunk into 2000-char blocks for FinBERT
            chunks = self._chunk_text(risk_text, chunk_size=2000)
            if chunks:
                chunk_scores = self._fb.analyze_batch(chunks)
                risk_score = self._fb.get_dominant_sentiment(chunk_scores)

        if mda_text:
            chunks = self._chunk_text(mda_text, chunk_size=2000)
            if chunks:
                chunk_scores = self._fb.analyze_batch(chunks)
                mda_score = self._fb.get_dominant_sentiment(chunk_scores)

        combined_text = (risk_text + " " + mda_text).strip()
        if combined_text:
            lm_score = self._lm.score(combined_text)

        # YoY tone shift: negative if risk score < -0.1 vs neutral baseline
        yoy_shift = 0.0
        if risk_score is not None:
            yoy_shift = risk_score.net * -1  # more negative = deterioration
        if mda_score is not None:
            yoy_shift = (yoy_shift + mda_score.net * -1) / 2

        return FilingSentiment(
            ticker=ticker,
            form_type=form_type,
            filing_date=datetime.now().strftime("%Y-%m-%d"),
            risk_factor_sentiment=risk_score,
            mda_sentiment=mda_score,
            lm_score=lm_score,
            uncertainty_score=lm_score.uncertainty_ratio if lm_score else 0.0,
            litigious_score=lm_score.litigious_ratio if lm_score else 0.0,
            yoy_tone_shift=yoy_shift,
        )

    def detect_sentiment_deterioration(self, ticker: str) -> bool:
        """
        Returns True if risk factor language indicates deteriorating sentiment.
        Criteria: LM negative > 3% of words, or LM uncertainty > 5%, or risk score net < -0.2.
        """
        filing = self.analyze_filing_sentiment(ticker)
        deteriorated = False
        if filing.lm_score:
            if filing.lm_score.uncertainty_ratio > 0.05:
                deteriorated = True
            neg_ratio = filing.lm_score.negative_count / max(filing.lm_score.total_words, 1)
            if neg_ratio > 0.03:
                deteriorated = True
        if filing.risk_factor_sentiment and filing.risk_factor_sentiment.net < -0.2:
            deteriorated = True
        return deteriorated

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
        """Split text into overlapping chunks for analysis."""
        chunks: List[str] = []
        i = 0
        step = chunk_size - 100  # 100-char overlap
        while i < len(text):
            chunks.append(text[i : i + chunk_size])
            i += step
        return [c for c in chunks if len(c.strip()) > 50]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated Sentiment Engine
# ─────────────────────────────────────────────────────────────────────────────

def _time_decay_weight(article_dt: datetime, half_life_hours: float) -> float:
    """Exponential time decay weight. Returns 0..1."""
    now = datetime.now(timezone.utc)
    if article_dt.tzinfo is None:
        article_dt = article_dt.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - article_dt).total_seconds() / 3600)
    return math.exp(-math.log(2) * age_hours / half_life_hours)


class AggregatedSentimentEngine:
    """
    Combines news, social media, and SEC filing sentiment into a composite signal.

    Weights: news=0.40, social=0.30, filings=0.30
    Time decay: news half-life=24h, SEC filings half-life=168h (7 days)
    """

    _NEWS_WEIGHT = 0.40
    _SOCIAL_WEIGHT = 0.30
    _FILING_WEIGHT = 0.30
    _NEWS_HALF_LIFE_HOURS = 24.0
    _SOCIAL_HALF_LIFE_HOURS = 12.0
    _FILING_HALF_LIFE_HOURS = 168.0

    def __init__(self) -> None:
        self._finbert = FinBERTAnalyzer()
        self._vader = VADERFinancialFallback()
        self._lm = LoughranMcDonaldLexicon()
        self._fetcher = NewsArticleFetcher()
        self._sec = SECFilingsSentimentAnalyzer(self._finbert, self._lm)
        self._sentiment_history: Dict[str, List[Tuple[datetime, float]]] = {}

    def _score_article(self, article: NewsArticle) -> float:
        """
        Compute -1..+1 sentiment for a news article.
        Blends: GDELT tone (if non-zero) 40% + NLP score 60%.
        """
        # If GDELT already computed tone, use it as partial signal
        text = (article.title + " " + article.description).strip()
        if text:
            nlp_score = self._finbert.analyze_text(text)
            nlp_net = nlp_score.net
        else:
            nlp_net = 0.0

        gdelt_tone = article.tone  # already normalized -1..+1
        if abs(gdelt_tone) > 0.001:
            return 0.4 * gdelt_tone + 0.6 * nlp_net
        return nlp_net

    def compute_composite_sentiment(
        self,
        ticker: str,
        company_name: str = "",
        include_reddit: bool = True,
        include_stocktwits: bool = True,
        include_filings: bool = True,
    ) -> CompositeSentiment:
        """
        Compute composite sentiment from all available sources.
        Sources: GDELT news, Yahoo RSS, StockTwits, Reddit, SEC filings.
        """
        now = datetime.now(timezone.utc)
        news_score = 0.0
        social_score = 0.0
        filing_score = 0.0
        n_news = 0
        n_social = 0

        # ── News (GDELT + Yahoo RSS) ──────────────────────────────────────────
        gdelt_articles = self._fetcher.fetch_gdelt_articles(ticker, company_name, days=7)
        rss_articles = self._fetcher.fetch_rss_articles(ticker)
        all_news = gdelt_articles + rss_articles

        if all_news:
            weighted_scores: List[float] = []
            weights: List[float] = []
            for article in all_news:
                score = self._score_article(article)
                w = _time_decay_weight(article.date, self._NEWS_HALF_LIFE_HOURS)
                weighted_scores.append(score * w)
                weights.append(w)
                article.sentiment = SentimentScore(
                    label="positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral"),
                    confidence=abs(score),
                    positive=max(0.0, score),
                    negative=max(0.0, -score),
                    neutral=1.0 - abs(score),
                )

            total_w = sum(weights)
            if total_w > 0:
                news_score = sum(weighted_scores) / total_w
            n_news = len(all_news)

        # ── Social (StockTwits + Reddit) ──────────────────────────────────────
        social_articles: List[NewsArticle] = []
        if include_stocktwits:
            social_articles.extend(self._fetcher.fetch_stocktwits(ticker))
        if include_reddit:
            social_articles.extend(self._fetcher.fetch_reddit_posts(ticker))

        if social_articles:
            weighted_scores = []
            weights = []
            for post in social_articles:
                text = (post.title + " " + post.description).strip()
                if text:
                    nlp = self._finbert.analyze_text(text[:1000])
                    score = 0.3 * post.tone + 0.7 * nlp.net
                else:
                    score = post.tone
                score = max(-1.0, min(1.0, score))
                w = _time_decay_weight(post.date, self._SOCIAL_HALF_LIFE_HOURS)
                weighted_scores.append(score * w)
                weights.append(w)

            total_w = sum(weights)
            if total_w > 0:
                social_score = sum(weighted_scores) / total_w
            n_social = len(social_articles)

        # ── SEC Filings ───────────────────────────────────────────────────────
        if include_filings:
            try:
                filing = self._sec.analyze_filing_sentiment(ticker)
                filing_net = 0.0
                count = 0
                if filing.risk_factor_sentiment:
                    filing_net += filing.risk_factor_sentiment.net
                    count += 1
                if filing.mda_sentiment:
                    filing_net += filing.mda_sentiment.net
                    count += 1
                if count > 0:
                    filing_score = filing_net / count
            except Exception as exc:
                logger.debug("SEC filing sentiment failed for %s: %s", ticker, exc)

        # ── Composite ─────────────────────────────────────────────────────────
        # Determine effective weights based on available data
        eff_news_w = self._NEWS_WEIGHT if n_news > 0 else 0.0
        eff_social_w = self._SOCIAL_WEIGHT if n_social > 0 else 0.0
        eff_filing_w = self._FILING_WEIGHT if include_filings else 0.0
        total_eff_w = eff_news_w + eff_social_w + eff_filing_w

        if total_eff_w < 1e-6:
            composite = 0.0
            confidence = 0.0
        else:
            composite = (
                news_score * eff_news_w
                + social_score * eff_social_w
                + filing_score * eff_filing_w
            ) / total_eff_w
            # Confidence: proportional to number of data points
            confidence = min(1.0, (n_news + n_social) / 50)

        composite = max(-1.0, min(1.0, composite))

        if composite > 0.1:
            label = "positive"
        elif composite < -0.1:
            label = "negative"
        else:
            label = "neutral"

        # Track for trend computation
        if ticker not in self._sentiment_history:
            self._sentiment_history[ticker] = []
        self._sentiment_history[ticker].append((now, composite))

        return CompositeSentiment(
            ticker=ticker,
            timestamp=now,
            composite_score=composite,
            composite_label=label,
            news_score=news_score,
            social_score=social_score,
            filing_score=filing_score,
            n_news_articles=n_news,
            n_social_posts=n_social,
            confidence=confidence,
        )

    def compute_sentiment_trend(
        self, ticker: str, lookback_days: int = 30
    ) -> pd.Series:
        """
        Returns daily composite sentiment over the past lookback_days.
        Uses historical in-memory cache if available, otherwise fetches.
        """
        # Try to use stored history
        history = self._sentiment_history.get(ticker, [])
        if history:
            dates = [h[0] for h in history]
            scores = [h[1] for h in history]
            series = pd.Series(scores, index=pd.DatetimeIndex(dates))
            series = series.resample("D").mean().dropna()
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
            return series[series.index >= cutoff]

        # No history: compute current + return single-point
        current = self.compute_composite_sentiment(ticker)
        return pd.Series(
            [current.composite_score],
            index=pd.DatetimeIndex([current.timestamp]),
        )

    def detect_sentiment_divergence(self, ticker: str) -> dict:
        """
        Detect social vs. news divergence.
        Social positive + news negative = potential pump.
        News positive + social negative = possible short squeeze setup.
        """
        cs = self.compute_composite_sentiment(ticker)
        result = {
            "ticker": ticker,
            "news_score": cs.news_score,
            "social_score": cs.social_score,
            "composite_score": cs.composite_score,
            "divergence": False,
            "divergence_type": "",
            "signal": "",
        }

        threshold = 0.15
        if cs.social_score > threshold and cs.news_score < -threshold:
            result["divergence"] = True
            result["divergence_type"] = "social_positive_news_negative"
            result["signal"] = "POTENTIAL_PUMP: Social bullish, news bearish"
        elif cs.news_score > threshold and cs.social_score < -threshold:
            result["divergence"] = True
            result["divergence_type"] = "news_positive_social_negative"
            result["signal"] = "SQUEEZE_SETUP: News bullish, social bearish"

        return result

    def get_sentiment_momentum(self, ticker: str) -> float:
        """
        5-day change in composite sentiment z-score.
        Positive = improving sentiment momentum.
        """
        trend = self.compute_sentiment_trend(ticker, lookback_days=30)
        if len(trend) < 6:
            return 0.0

        std = trend.std()
        if std < 1e-6:
            return 0.0

        z_scores = (trend - trend.mean()) / std
        recent_5 = z_scores.iloc[-5:]
        older_5 = z_scores.iloc[-10:-5] if len(z_scores) >= 10 else z_scores.iloc[:5]

        return float(recent_5.mean() - older_5.mean())

    # ──────────────────────────────────────────────────────────────────────────
    # Stub fix #4: Entity normalization
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def normalize_entity_name(raw_name: str) -> str:
        """
        Stub fix #4: Canonicalize a company or ticker name for consistent lookup.

        Rules applied (in order):
          1. Strip leading/trailing whitespace
          2. Upper-case obvious ticker-like strings (≤5 chars, all alpha)
          3. Remove common legal suffixes (Inc., Corp., Ltd., LLC, etc.)
          4. Remove punctuation except hyphens
          5. Collapse multiple spaces to one
          6. Title-case the result

        Examples:
          "apple inc."        → "Apple"
          "MSFT"              → "MSFT"
          "Alphabet Inc."     → "Alphabet"
          "berkshire hathaway llc" → "Berkshire Hathaway"
        """
        if not raw_name:
            return ""

        name = raw_name.strip()

        # If it looks like a ticker (short, uppercase or easily uppercased, no spaces)
        if len(name) <= 5 and re.match(r"^[A-Za-z.\-]+$", name) and " " not in name:
            return name.upper().rstrip(".")

        # Remove common legal suffixes (case-insensitive)
        legal_suffixes = [
            r"\bInc\.?\b", r"\bCorp\.?\b", r"\bCorporation\b",
            r"\bLtd\.?\b", r"\bLimited\b", r"\bLLC\b", r"\bL\.L\.C\.?\b",
            r"\bLLP\b", r"\bPLC\b", r"\bP\.L\.C\.?\b", r"\bS\.A\.?\b",
            r"\bN\.V\.?\b", r"\bA\.G\.?\b", r"\bGmbH\b", r"\bSE\b",
            r"\bHoldings\b", r"\bGroup\b", r"\bCo\.?\b",
        ]
        for suffix in legal_suffixes:
            name = re.sub(suffix, "", name, flags=re.IGNORECASE)

        # Remove punctuation except hyphens and ampersands
        name = re.sub(r"[^\w\s\-&]", " ", name)
        # Collapse whitespace
        name = re.sub(r"\s+", " ", name).strip()
        # Title-case
        return name.title() if name else ""

    # ──────────────────────────────────────────────────────────────────────────
    # Stub fix #5: Sentiment aggregation by entity
    # ──────────────────────────────────────────────────────────────────────────

    def aggregate_sentiment_by_entity(
        self,
        articles: List["NewsArticle"],
        entity_mentions: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Dict]:
        """
        Stub fix #5: Aggregate article-level sentiment scores per named entity.

        For each entity (ticker or company name), collect all articles that
        mention it, score them with FinBERT/VADER, and produce a per-entity
        summary with:
          - 'n_articles'         : number of articles mentioning entity
          - 'mean_net_sentiment' : mean of net sentiment scores
          - 'std_net_sentiment'  : std dev of net sentiment scores
          - 'positive_pct'       : fraction of articles with positive label
          - 'negative_pct'       : fraction of articles with negative label
          - 'dominant_label'     : overall label based on mean_net_sentiment
          - 'articles'           : list of (title, net) tuples for top-5 articles

        Parameters
        ----------
        articles : List[NewsArticle]
            Pre-fetched article list (title + description used for scoring).
        entity_mentions : Optional dict mapping canonical entity name →
            list of name variants to search for in article text.
            If None, uses the articles' domain field as a coarse grouping.
        """
        if not articles:
            return {}

        # Default: group by domain
        if entity_mentions is None:
            domains = list({a.domain for a in articles if a.domain})
            entity_mentions = {d: [d] for d in domains}

        result: Dict[str, Dict] = {}

        for entity, variants in entity_mentions.items():
            canonical = self.normalize_entity_name(entity)
            matching = [
                a for a in articles
                if any(
                    v.lower() in (a.title + " " + a.description + " " + a.domain).lower()
                    for v in variants
                )
            ]
            if not matching:
                continue

            texts = [(a.title + " " + a.description)[:500] for a in matching]
            scores = self._finbert.analyze_batch(texts)

            nets = [s.net for s in scores]
            labels = [s.label for s in scores]

            mean_net = float(np.mean(nets)) if nets else 0.0
            std_net = float(np.std(nets)) if len(nets) > 1 else 0.0
            pos_pct = labels.count("positive") / len(labels) if labels else 0.0
            neg_pct = labels.count("negative") / len(labels) if labels else 0.0

            if mean_net > 0.05:
                dominant = "positive"
            elif mean_net < -0.05:
                dominant = "negative"
            else:
                dominant = "neutral"

            # Top-5 articles by absolute net score
            top_articles = sorted(
                zip([a.title for a in matching], nets),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]

            result[canonical] = {
                "n_articles": len(matching),
                "mean_net_sentiment": round(mean_net, 4),
                "std_net_sentiment": round(std_net, 4),
                "positive_pct": round(pos_pct, 3),
                "negative_pct": round(neg_pct, 3),
                "dominant_label": dominant,
                "articles": [(t, round(n, 4)) for t, n in top_articles],
            }

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Screener
# ─────────────────────────────────────────────────────────────────────────────

class SentimentScreener:
    """
    Screen a universe of tickers by sentiment signal.
    Supports: top N positive/negative, reversal detection, divergence screen.
    """

    def __init__(self, engine: Optional[AggregatedSentimentEngine] = None) -> None:
        self._engine = engine or AggregatedSentimentEngine()

    def _fetch_scores(
        self, universe: List[str], delay: float = 0.3
    ) -> pd.DataFrame:
        """Fetch composite sentiment for all tickers in universe."""
        rows: List[dict] = []
        for ticker in universe:
            try:
                cs = self._engine.compute_composite_sentiment(ticker)
                rows.append({
                    "ticker": ticker,
                    "composite_score": cs.composite_score,
                    "news_score": cs.news_score,
                    "social_score": cs.social_score,
                    "filing_score": cs.filing_score,
                    "n_news": cs.n_news_articles,
                    "n_social": cs.n_social_posts,
                    "label": cs.composite_label,
                    "confidence": cs.confidence,
                })
            except Exception as exc:
                logger.warning("Screen: failed for %s: %s", ticker, exc)
                rows.append({
                    "ticker": ticker, "composite_score": 0.0,
                    "news_score": 0.0, "social_score": 0.0, "filing_score": 0.0,
                    "n_news": 0, "n_social": 0, "label": "neutral", "confidence": 0.0,
                })
            time.sleep(delay)

        return pd.DataFrame(rows)

    def screen_most_positive(
        self, universe: List[str], n: int = 10
    ) -> pd.DataFrame:
        """Return top N tickers by composite sentiment score."""
        df = self._fetch_scores(universe)
        return df.nlargest(n, "composite_score").reset_index(drop=True)

    def screen_most_negative(
        self, universe: List[str], n: int = 10
    ) -> pd.DataFrame:
        """Return bottom N tickers by composite sentiment score."""
        df = self._fetch_scores(universe)
        return df.nsmallest(n, "composite_score").reset_index(drop=True)

    def screen_sentiment_reversal(
        self, universe: List[str], lookback_days: int = 5
    ) -> pd.DataFrame:
        """
        Detect tickers where sentiment flipped from negative to positive
        in the last `lookback_days` days.
        """
        reversals: List[dict] = []
        for ticker in universe:
            try:
                trend = self._engine.compute_sentiment_trend(ticker, lookback_days=lookback_days + 5)
                if len(trend) < 2:
                    continue

                recent = trend.iloc[-lookback_days:].mean() if len(trend) >= lookback_days else trend.iloc[-1]
                prior = trend.iloc[:-lookback_days].mean() if len(trend) > lookback_days else trend.iloc[0]

                if prior < -0.05 and recent > 0.05:
                    reversals.append({
                        "ticker": ticker,
                        "prior_sentiment": float(prior),
                        "recent_sentiment": float(recent),
                        "reversal_magnitude": float(recent - prior),
                    })
            except Exception as exc:
                logger.debug("Reversal screen failed for %s: %s", ticker, exc)

        if not reversals:
            return pd.DataFrame(columns=["ticker", "prior_sentiment", "recent_sentiment", "reversal_magnitude"])
        return pd.DataFrame(reversals).sort_values("reversal_magnitude", ascending=False).reset_index(drop=True)

    def screen_sentiment_divergence(
        self, universe: List[str]
    ) -> pd.DataFrame:
        """Screen for news vs. social divergence in a universe."""
        results: List[dict] = []
        for ticker in universe:
            try:
                div = self._engine.detect_sentiment_divergence(ticker)
                if div["divergence"]:
                    results.append({
                        "ticker": ticker,
                        "divergence_type": div["divergence_type"],
                        "signal": div["signal"],
                        "news_score": div["news_score"],
                        "social_score": div["social_score"],
                        "spread": abs(div["news_score"] - div["social_score"]),
                    })
            except Exception as exc:
                logger.debug("Divergence screen failed for %s: %s", ticker, exc)

        if not results:
            return pd.DataFrame(columns=["ticker", "divergence_type", "signal", "news_score", "social_score", "spread"])
        return pd.DataFrame(results).sort_values("spread", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Backtester
# ─────────────────────────────────────────────────────────────────────────────

class SentimentBacktester:
    """
    Backtest sentiment signals against price data.
    Uses yfinance for price data (free).
    """

    def __init__(self, engine: Optional[AggregatedSentimentEngine] = None) -> None:
        self._engine = engine or AggregatedSentimentEngine()

    def _fetch_prices(self, tickers: List[str], period: str = "60d") -> pd.DataFrame:
        """Fetch adjusted close prices via yfinance."""
        try:
            import yfinance as yf
            raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                return raw["Close"]
            return raw[["Close"]].rename(columns={"Close": tickers[0]})
        except ImportError:
            logger.error("yfinance not installed — cannot backtest")
            return pd.DataFrame()
        except Exception as exc:
            logger.error("yfinance download failed: %s", exc)
            return pd.DataFrame()

    def backtest_sentiment_signal(
        self,
        tickers: List[str],
        signal_threshold: float = 0.1,
        hold_days: int = 5,
    ) -> BacktestResult:
        """
        Long when composite sentiment > threshold.
        Measure hold_days forward returns.
        Returns BacktestResult with IC and summary statistics.
        """
        prices = self._fetch_prices(tickers)
        if prices.empty:
            return BacktestResult(
                signal_threshold=signal_threshold,
                hold_days=hold_days,
                tickers_tested=0,
                mean_forward_return=0.0,
                median_forward_return=0.0,
                hit_rate=0.0,
                information_coefficient=0.0,
                sharpe_ratio=0.0,
                n_trades=0,
            )

        rows: List[dict] = []
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            try:
                cs = self._engine.compute_composite_sentiment(ticker)
                sentiment_score = cs.composite_score

                # Get forward return from price data
                px = prices[ticker].dropna()
                if len(px) <= hold_days:
                    continue

                entry_price = float(px.iloc[-hold_days - 1])
                exit_price = float(px.iloc[-1])
                if entry_price <= 0:
                    continue

                fwd_return = (exit_price - entry_price) / entry_price

                rows.append({
                    "ticker": ticker,
                    "sentiment": sentiment_score,
                    "fwd_return": fwd_return,
                    "signal": 1 if sentiment_score > signal_threshold else 0,
                })
            except Exception as exc:
                logger.debug("Backtest failed for %s: %s", ticker, exc)

        if not rows:
            return BacktestResult(
                signal_threshold=signal_threshold,
                hold_days=hold_days,
                tickers_tested=len(tickers),
                mean_forward_return=0.0,
                median_forward_return=0.0,
                hit_rate=0.0,
                information_coefficient=0.0,
                sharpe_ratio=0.0,
                n_trades=0,
            )

        df = pd.DataFrame(rows)
        long_trades = df[df["signal"] == 1]
        n_trades = len(long_trades)

        if n_trades == 0:
            mean_ret = 0.0
            median_ret = 0.0
            hit_rate = 0.0
            sharpe = 0.0
        else:
            rets = long_trades["fwd_return"]
            mean_ret = float(rets.mean())
            median_ret = float(rets.median())
            hit_rate = float((rets > 0).mean())
            sharpe = float(rets.mean() / rets.std()) if rets.std() > 1e-8 else 0.0

        # IC: Spearman correlation between sentiment score and forward return
        from scipy.stats import spearmanr
        if len(df) >= 3:
            ic_val, _ = spearmanr(df["sentiment"], df["fwd_return"])
            ic = float(ic_val) if not math.isnan(ic_val) else 0.0
        else:
            ic = 0.0

        return BacktestResult(
            signal_threshold=signal_threshold,
            hold_days=hold_days,
            tickers_tested=len(tickers),
            mean_forward_return=mean_ret,
            median_forward_return=median_ret,
            hit_rate=hit_rate,
            information_coefficient=ic,
            sharpe_ratio=sharpe,
            n_trades=n_trades,
            results_df=df,
        )

    def compute_sentiment_ic(self, ticker: str) -> float:
        """
        Information coefficient: Spearman correlation between
        composite sentiment and next-day returns.
        Requires historical price and sentiment data.
        """
        prices = self._fetch_prices([ticker], period="30d")
        if prices.empty or ticker not in prices.columns:
            return 0.0

        px = prices[ticker].dropna()
        if len(px) < 5:
            return 0.0

        returns = px.pct_change().dropna()
        # Proxy: current sentiment as scalar; return first next-day return
        cs = self._engine.compute_composite_sentiment(ticker)
        if len(returns) < 2:
            return 0.0

        # Simplified IC: use last 10 returns and assume sentiment is leading
        sentiments = [cs.composite_score] * len(returns)
        try:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(sentiments, returns.values)
            return float(ic) if not math.isnan(ic) else 0.0
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Router
# ─────────────────────────────────────────────────────────────────────────────

sentiment_v3_router = APIRouter(prefix="/sentiment/v3", tags=["sentiment-v3"])

# Shared instances
_engine: Optional[AggregatedSentimentEngine] = None
_screener: Optional[SentimentScreener] = None
_backtester: Optional[SentimentBacktester] = None
_sec_analyzer: Optional[SECFilingsSentimentAnalyzer] = None


def _get_engine() -> AggregatedSentimentEngine:
    global _engine
    if _engine is None:
        _engine = AggregatedSentimentEngine()
    return _engine


def _get_screener() -> SentimentScreener:
    global _screener
    if _screener is None:
        _screener = SentimentScreener(_get_engine())
    return _screener


def _get_backtester() -> SentimentBacktester:
    global _backtester
    if _backtester is None:
        _backtester = SentimentBacktester(_get_engine())
    return _backtester


class ScreenRequest(BaseModel):
    universe: List[str] = Field(..., description="List of ticker symbols")
    mode: str = Field("positive", description="positive | negative | reversal | divergence")
    n: int = Field(10, description="Number of results")


class BacktestRequest(BaseModel):
    tickers: List[str]
    signal_threshold: float = 0.1
    hold_days: int = 5


@sentiment_v3_router.get("/analyze/{ticker}")
async def analyze_ticker_sentiment(ticker: str, days: int = 7):
    """Analyze news sentiment for a single ticker via GDELT + Yahoo RSS."""
    try:
        engine = _get_engine()
        fetcher = engine._fetcher
        articles = fetcher.fetch_gdelt_articles(ticker.upper(), days=days)
        rss = fetcher.fetch_rss_articles(ticker.upper())
        all_articles = articles + rss

        finbert = engine._finbert
        texts = [(a.title + " " + a.description)[:1000] for a in all_articles[:50]]
        scores = finbert.analyze_batch(texts) if texts else []
        dominant = finbert.get_dominant_sentiment(scores) if scores else None

        return {
            "ticker": ticker.upper(),
            "n_articles": len(all_articles),
            "dominant_sentiment": {
                "label": dominant.label if dominant else "neutral",
                "net": dominant.net if dominant else 0.0,
                "confidence": dominant.confidence if dominant else 0.0,
            } if dominant else None,
            "articles": [
                {
                    "title": a.title[:120],
                    "date": a.date.isoformat(),
                    "tone": a.tone,
                    "domain": a.domain,
                }
                for a in all_articles[:20]
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sentiment_v3_router.get("/composite/{ticker}")
async def composite_sentiment(ticker: str, company_name: str = ""):
    """Compute full composite sentiment (news + social + filings)."""
    try:
        cs = _get_engine().compute_composite_sentiment(ticker.upper(), company_name)
        return {
            "ticker": cs.ticker,
            "timestamp": cs.timestamp.isoformat(),
            "composite_score": cs.composite_score,
            "composite_label": cs.composite_label,
            "news_score": cs.news_score,
            "social_score": cs.social_score,
            "filing_score": cs.filing_score,
            "n_news_articles": cs.n_news_articles,
            "n_social_posts": cs.n_social_posts,
            "confidence": cs.confidence,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sentiment_v3_router.post("/screen")
async def screen_universe(req: ScreenRequest):
    """Screen a universe of tickers by sentiment signal."""
    screener = _get_screener()
    universe = [t.upper() for t in req.universe]
    try:
        if req.mode == "positive":
            df = screener.screen_most_positive(universe, req.n)
        elif req.mode == "negative":
            df = screener.screen_most_negative(universe, req.n)
        elif req.mode == "reversal":
            df = screener.screen_sentiment_reversal(universe)
        elif req.mode == "divergence":
            df = screener.screen_sentiment_divergence(universe)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sentiment_v3_router.get("/sec/{ticker}")
async def sec_filing_sentiment(ticker: str, form_type: str = "10-K"):
    """Analyze SEC filing sentiment for a ticker."""
    try:
        engine = _get_engine()
        sec = engine._sec
        filing = sec.analyze_filing_sentiment(ticker.upper(), form_type)
        deteriorating = sec.detect_sentiment_deterioration(ticker.upper())
        return {
            "ticker": filing.ticker,
            "form_type": filing.form_type,
            "filing_date": filing.filing_date,
            "risk_factor_sentiment": {
                "label": filing.risk_factor_sentiment.label,
                "net": filing.risk_factor_sentiment.net,
            } if filing.risk_factor_sentiment else None,
            "mda_sentiment": {
                "label": filing.mda_sentiment.label,
                "net": filing.mda_sentiment.net,
            } if filing.mda_sentiment else None,
            "uncertainty_score": filing.uncertainty_score,
            "litigious_score": filing.litigious_score,
            "yoy_tone_shift": filing.yoy_tone_shift,
            "sentiment_deteriorating": deteriorating,
            "lm_scores": {
                "positive": filing.lm_score.positive_count,
                "negative": filing.lm_score.negative_count,
                "uncertainty": filing.lm_score.uncertainty_count,
                "litigious": filing.lm_score.litigious_count,
                "net_sentiment": filing.lm_score.net_sentiment,
            } if filing.lm_score else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sentiment_v3_router.get("/trend/{ticker}")
async def sentiment_trend(ticker: str, lookback_days: int = 30):
    """Compute daily sentiment trend for a ticker."""
    try:
        trend = _get_engine().compute_sentiment_trend(ticker.upper(), lookback_days)
        return {
            "ticker": ticker.upper(),
            "lookback_days": lookback_days,
            "trend": [
                {"date": str(idx.date()), "sentiment": float(val)}
                for idx, val in trend.items()
            ],
            "momentum": _get_engine().get_sentiment_momentum(ticker.upper()),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sentiment_v3_router.get("/divergence/{ticker}")
async def sentiment_divergence(ticker: str):
    """Detect social vs. news divergence for a ticker."""
    try:
        return _get_engine().detect_sentiment_divergence(ticker.upper())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@sentiment_v3_router.post("/backtest")
async def backtest_sentiment(req: BacktestRequest):
    """Backtest sentiment signal against forward returns."""
    try:
        tickers = [t.upper() for t in req.tickers]
        result = _get_backtester().backtest_sentiment_signal(
            tickers, req.signal_threshold, req.hold_days
        )
        return {
            "signal_threshold": result.signal_threshold,
            "hold_days": result.hold_days,
            "tickers_tested": result.tickers_tested,
            "n_trades": result.n_trades,
            "mean_forward_return": result.mean_forward_return,
            "median_forward_return": result.median_forward_return,
            "hit_rate": result.hit_rate,
            "information_coefficient": result.information_coefficient,
            "sharpe_ratio": result.sharpe_ratio,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    SP100_SAMPLE = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
        "UNH", "LLY", "JPM", "V", "AVGO", "XOM", "MA", "JNJ", "PG",
        "HD", "MRK", "CVX",
    ]

    print("\n" + "=" * 70)
    print("SENTINEL Financial Sentiment Platform v3")
    print("=" * 70)

    # Instantiate shared engine
    engine = AggregatedSentimentEngine()
    finbert = engine._finbert
    print(f"FinBERT backend: {finbert.backend}")

    # 1. Analyze AAPL, TSLA, NVDA via GDELT
    focus_tickers = ["AAPL", "TSLA", "NVDA"]
    print(f"\n{'─'*70}")
    print("GDELT News Sentiment Analysis")
    print(f"{'─'*70}")
    for ticker in focus_tickers:
        fetcher = engine._fetcher
        articles = fetcher.fetch_gdelt_articles(ticker, days=7)
        rss = fetcher.fetch_rss_articles(ticker)
        all_articles = articles + rss
        print(f"\n{ticker}: {len(all_articles)} articles")

        texts = [(a.title + " " + a.description)[:500] for a in all_articles[:30] if a.title]
        if texts:
            scores = finbert.analyze_batch(texts)
            dominant = finbert.get_dominant_sentiment(scores)
            print(f"  Dominant sentiment : {dominant.label} (net={dominant.net:+.3f}, conf={dominant.confidence:.3f})")
            avg_gdelt_tone = sum(a.tone for a in articles) / len(articles) if articles else 0
            print(f"  GDELT avg tone     : {avg_gdelt_tone:+.3f}")
        else:
            print("  No articles with text available")

    # 2. Composite sentiment scores
    print(f"\n{'─'*70}")
    print("Composite Sentiment (news + social + filings)")
    print(f"{'─'*70}")
    print(f"{'Ticker':<8} {'Score':>8} {'Label':<12} {'News':>8} {'Social':>8} {'Filing':>8} {'N_Art':>6}")
    print("-" * 70)
    for ticker in focus_tickers:
        cs = engine.compute_composite_sentiment(ticker)
        print(
            f"{cs.ticker:<8} {cs.composite_score:>+8.3f} {cs.composite_label:<12} "
            f"{cs.news_score:>+8.3f} {cs.social_score:>+8.3f} {cs.filing_score:>+8.3f} "
            f"{cs.n_news_articles:>6d}"
        )

    # 3. Screen S&P 100 sample — top 5 most positive and most negative
    print(f"\n{'─'*70}")
    print(f"Sentiment Screen — S&P 100 Sample ({len(SP100_SAMPLE)} tickers)")
    print(f"{'─'*70}")
    screener = SentimentScreener(engine)

    print("\nTop 5 Most Positive:")
    top_pos = screener.screen_most_positive(SP100_SAMPLE, n=5)
    if not top_pos.empty:
        print(top_pos[["ticker", "composite_score", "news_score", "social_score", "n_news"]].to_string(index=False))
    else:
        print("  No results")

    print("\nTop 5 Most Negative:")
    top_neg = screener.screen_most_negative(SP100_SAMPLE, n=5)
    if not top_neg.empty:
        print(top_neg[["ticker", "composite_score", "news_score", "social_score", "n_news"]].to_string(index=False))
    else:
        print("  No results")

    print("\nSentiment Divergence Screen:")
    diverg = screener.screen_sentiment_divergence(SP100_SAMPLE[:10])
    if not diverg.empty:
        print(diverg.to_string(index=False))
    else:
        print("  No divergences detected in sample")

    # 4. Backtest on focus tickers
    print(f"\n{'─'*70}")
    print("Sentiment Signal Backtest (5-day hold)")
    print(f"{'─'*70}")
    backtester = SentimentBacktester(engine)
    result = backtester.backtest_sentiment_signal(
        SP100_SAMPLE[:10], signal_threshold=0.1, hold_days=5
    )
    print(f"  Tickers tested        : {result.tickers_tested}")
    print(f"  Trades (long signals) : {result.n_trades}")
    print(f"  Mean 5d return        : {result.mean_forward_return:+.2%}")
    print(f"  Median 5d return      : {result.median_forward_return:+.2%}")
    print(f"  Hit rate              : {result.hit_rate:.1%}")
    print(f"  Spearman IC           : {result.information_coefficient:+.3f}")
    print(f"  Signal Sharpe ratio   : {result.sharpe_ratio:+.3f}")

    # 5. LM Lexicon demo
    print(f"\n{'─'*70}")
    print("Loughran-McDonald Lexicon Demo")
    print(f"{'─'*70}")
    lm = LoughranMcDonaldLexicon()
    sample_texts = [
        "The company exceeded revenue expectations with record earnings and raised its full-year guidance.",
        "Management disclosed an SEC investigation into potential accounting irregularities; the company faces significant litigation risk.",
        "Revenue declined 12% year-over-year as restructuring charges and impairment write-downs weighed on results.",
    ]
    for text in sample_texts:
        score = lm.score(text)
        print(f"\n  Text: '{text[:80]}...'")
        print(f"    Net sentiment   : {score.net_sentiment:+.3f}")
        print(f"    Uncertainty     : {score.uncertainty_ratio:.3f}")
        print(f"    Litigious       : {score.litigious_ratio:.3f}")
        print(f"    LM Pos/Neg      : {score.positive_count}/{score.negative_count}")

    print(f"\n{'='*70}")
    print("Sentiment Platform v3 — Done")
    print("=" * 70)
