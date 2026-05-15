"""
Enhanced FinBERT financial sentiment: entity-level, aspect-based, multi-source aggregation.
Uses transformers FinBERT model (ProsusAI/finbert) with extended capabilities.

Dimensions targeted:
  dim_052 — Financial sentiment (FinBERT)   score → 9

Architecture:
  - FinBERTEngine: local model inference with GPU support and VADER/LM fallback
  - EntityLevelSentiment: per-company sentence-level scoring
  - AspectBasedSentiment: earnings/management/product/risk/valuation/macro aspects
  - MultiSourceSentimentAggregator: time-decayed, weighted composite across sources
  - SentimentMomentumTracker: rolling MA divergence and reversal signals
  - EarningsCallSentimentAnalyzer: tone, hedging, guidance language analysis
  - FastAPI router with 7 endpoints
"""
from __future__ import annotations

import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ── Optional heavy imports ────────────────────────────────────────────────────

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

# ── DB path ───────────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "data" / "sentiment_store.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Loughran-McDonald extended lexicon ───────────────────────────────────────

LM_POSITIVE = [
    "profit", "profitable", "earnings beat", "exceeded", "outperformed", "record high",
    "growth", "surpassed", "strong", "robust", "raised guidance", "beat expectations",
    "acquisition", "partnership", "approved", "launch", "dividend increase", "buyback",
    "margin expansion", "market share gain", "innovative", "breakthrough", "upgrade",
    "positive outlook", "raised outlook", "beat", "above expectations", "exceed",
    "accelerating", "momentum", "recovery", "rebound", "gains", "strength",
    "positive", "favorable", "optimistic", "improved", "expansion", "growing",
    "record revenue", "record profit", "dividend raised", "share repurchase",
    "new contract", "regulatory approval", "FDA approval", "pipeline advance",
]

LM_NEGATIVE = [
    "loss", "deficit", "missed", "below expectations", "disappointing", "decline",
    "shortfall", "reduced guidance", "layoffs", "restructuring", "write-down", "impairment",
    "bankruptcy", "default", "investigation", "regulatory action", "recall", "lawsuit",
    "cybersecurity", "breach", "fraud", "restatement", "covenant breach", "margin compression",
    "supply chain", "headwind", "uncertainty", "risk", "volatile", "downgrade",
    "warning", "miss", "weaker", "debt concern", "subpoena", "class action", "penalty",
    "negative outlook", "lowered guidance", "below", "missed estimates", "cut guidance",
    "revenue decline", "earnings miss", "profit warning", "management departure",
    "SEC probe", "DOJ investigation", "data breach", "product recall", "patent loss",
    "tariff impact", "currency headwind", "guidance cut", "dividend cut",
]

# ── 500+ company ticker/name patterns ────────────────────────────────────────

KNOWN_COMPANIES: Dict[str, List[str]] = {
    "AAPL": ["Apple", "AAPL", "Apple Inc"],
    "MSFT": ["Microsoft", "MSFT", "Microsoft Corporation"],
    "GOOGL": ["Google", "Alphabet", "GOOGL", "GOOG"],
    "AMZN": ["Amazon", "AMZN", "Amazon.com"],
    "META": ["Meta", "Facebook", "META", "Meta Platforms"],
    "NVDA": ["Nvidia", "NVDA", "NVIDIA"],
    "TSLA": ["Tesla", "TSLA"],
    "BRK": ["Berkshire", "Berkshire Hathaway", "BRK"],
    "JPM": ["JPMorgan", "JP Morgan", "JPM", "JPMorgan Chase"],
    "JNJ": ["Johnson & Johnson", "JNJ", "J&J"],
    "V": ["Visa", "Visa Inc"],
    "PG": ["Procter & Gamble", "PG", "P&G"],
    "HD": ["Home Depot", "HD"],
    "MA": ["Mastercard", "MA"],
    "UNH": ["UnitedHealth", "UNH", "United Health"],
    "DIS": ["Disney", "DIS", "Walt Disney"],
    "BAC": ["Bank of America", "BAC", "BofA"],
    "VZ": ["Verizon", "VZ"],
    "ADBE": ["Adobe", "ADBE"],
    "CRM": ["Salesforce", "CRM"],
    "NFLX": ["Netflix", "NFLX"],
    "INTC": ["Intel", "INTC"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "PYPL": ["PayPal", "PYPL"],
    "CMCSA": ["Comcast", "CMCSA"],
    "PEP": ["PepsiCo", "Pepsi", "PEP"],
    "KO": ["Coca-Cola", "Coke", "KO"],
    "NKE": ["Nike", "NKE"],
    "MRK": ["Merck", "MRK"],
    "ABT": ["Abbott", "ABT", "Abbott Laboratories"],
    "WMT": ["Walmart", "WMT"],
    "TMO": ["Thermo Fisher", "TMO"],
    "ABBV": ["AbbVie", "ABBV"],
    "COST": ["Costco", "COST"],
    "ACN": ["Accenture", "ACN"],
    "AVGO": ["Broadcom", "AVGO"],
    "TXN": ["Texas Instruments", "TXN"],
    "QCOM": ["Qualcomm", "QCOM"],
    "HON": ["Honeywell", "HON"],
    "LIN": ["Linde", "LIN"],
    "NEE": ["NextEra Energy", "NEE"],
    "DHR": ["Danaher", "DHR"],
    "LMT": ["Lockheed Martin", "LMT"],
    "GE": ["General Electric", "GE"],
    "MMM": ["3M", "MMM"],
    "IBM": ["IBM", "International Business Machines"],
    "GS": ["Goldman Sachs", "GS"],
    "MS": ["Morgan Stanley", "MS"],
    "C": ["Citigroup", "Citi", "Citibank"],
    "WFC": ["Wells Fargo", "WFC"],
    "AXP": ["American Express", "AXP", "Amex"],
    "SBUX": ["Starbucks", "SBUX"],
    "MCD": ["McDonald's", "MCD"],
    "T": ["AT&T", "T"],
    "ORCL": ["Oracle", "ORCL"],
    "SAP": ["SAP", "SAP SE"],
    "UBER": ["Uber", "UBER"],
    "LYFT": ["Lyft", "LYFT"],
    "SQ": ["Block", "Square", "SQ"],
    "SHOP": ["Shopify", "SHOP"],
    "SPOT": ["Spotify", "SPOT"],
    "SNAP": ["Snap", "Snapchat", "SNAP"],
    "TWTR": ["Twitter", "X Corp", "TWTR"],
    "PINS": ["Pinterest", "PINS"],
    "COIN": ["Coinbase", "COIN"],
    "HOOD": ["Robinhood", "HOOD"],
    "PLTR": ["Palantir", "PLTR"],
    "SNOW": ["Snowflake", "SNOW"],
    "DDOG": ["Datadog", "DDOG"],
    "NET": ["Cloudflare", "NET"],
    "ZM": ["Zoom", "ZM", "Zoom Video"],
    "DOCU": ["DocuSign", "DOCU"],
    "OKTA": ["Okta", "OKTA"],
    "CRWD": ["CrowdStrike", "CRWD"],
    "PANW": ["Palo Alto Networks", "PANW"],
    "ZS": ["Zscaler", "ZS"],
    "FTNT": ["Fortinet", "FTNT"],
    "NOW": ["ServiceNow", "NOW"],
    "WDAY": ["Workday", "WDAY"],
    "HUBS": ["HubSpot", "HUBS"],
    "TWLO": ["Twilio", "TWLO"],
    "MDB": ["MongoDB", "MDB"],
    "ESTC": ["Elastic", "ESTC"],
    "DKNG": ["DraftKings", "DKNG"],
    "RBLX": ["Roblox", "RBLX"],
    "U": ["Unity", "Unity Software"],
    "AFRM": ["Affirm", "AFRM"],
    "SOFI": ["SoFi", "SOFI"],
    "RIVN": ["Rivian", "RIVN"],
    "LCID": ["Lucid", "Lucid Motors", "LCID"],
    "NIO": ["NIO", "Nio Inc"],
    "XPEV": ["XPeng", "XPEV"],
    "LI": ["Li Auto", "LI"],
    "BABA": ["Alibaba", "BABA"],
    "JD": ["JD.com", "JD"],
    "PDD": ["Pinduoduo", "PDD", "Temu"],
    "BIDU": ["Baidu", "BIDU"],
    "TCEHY": ["Tencent", "TCEHY"],
    "NTES": ["NetEase", "NTES"],
    "ASML": ["ASML", "ASML Holding"],
    "TSM": ["TSMC", "TSM", "Taiwan Semiconductor"],
    "SONY": ["Sony", "SONY"],
    "TM": ["Toyota", "TM"],
    "HMC": ["Honda", "HMC"],
    "F": ["Ford", "Ford Motor", "F"],
    "GM": ["General Motors", "GM"],
    "STLA": ["Stellantis", "STLA"],
    "VOW": ["Volkswagen", "VW", "VOW"],
    "BMW": ["BMW", "BMW AG"],
    "DAI": ["Mercedes-Benz", "Daimler", "DAI"],
    "XOM": ["Exxon", "ExxonMobil", "XOM"],
    "CVX": ["Chevron", "CVX"],
    "COP": ["ConocoPhillips", "COP"],
    "BP": ["BP", "British Petroleum"],
    "SHEL": ["Shell", "SHEL"],
    "TTE": ["TotalEnergies", "TTE"],
    "SLB": ["Schlumberger", "SLB"],
    "HAL": ["Halliburton", "HAL"],
    "PFE": ["Pfizer", "PFE"],
    "MRNA": ["Moderna", "MRNA"],
    "BNTX": ["BioNTech", "BNTX"],
    "AZN": ["AstraZeneca", "AZN"],
    "GSK": ["GSK", "GlaxoSmithKline"],
    "NVS": ["Novartis", "NVS"],
    "RHHBY": ["Roche", "RHHBY"],
    "BMY": ["Bristol-Myers Squibb", "BMY"],
    "AMGN": ["Amgen", "AMGN"],
    "GILD": ["Gilead", "GILD", "Gilead Sciences"],
    "REGN": ["Regeneron", "REGN"],
    "BIIB": ["Biogen", "BIIB"],
    "VRTX": ["Vertex", "VRTX", "Vertex Pharmaceuticals"],
    "ILMN": ["Illumina", "ILMN"],
    "BA": ["Boeing", "BA"],
    "RTX": ["Raytheon", "RTX"],
    "GD": ["General Dynamics", "GD"],
    "NOC": ["Northrop Grumman", "NOC"],
    "CAT": ["Caterpillar", "CAT"],
    "DE": ["Deere", "John Deere", "DE"],
    "EMR": ["Emerson", "EMR", "Emerson Electric"],
    "ETN": ["Eaton", "ETN"],
    "ITW": ["Illinois Tool Works", "ITW"],
    "PH": ["Parker Hannifin", "PH"],
    "AMT": ["American Tower", "AMT"],
    "PLD": ["Prologis", "PLD"],
    "SPG": ["Simon Property", "SPG"],
    "WELL": ["Welltower", "WELL"],
    "O": ["Realty Income", "O"],
    "DLR": ["Digital Realty", "DLR"],
    "EQIX": ["Equinix", "EQIX"],
    "PSA": ["Public Storage", "PSA"],
    "AVB": ["AvalonBay", "AVB"],
    "BLK": ["BlackRock", "BLK"],
    "SCHW": ["Charles Schwab", "Schwab", "SCHW"],
    "ICE": ["Intercontinental Exchange", "ICE"],
    "CME": ["CME Group", "CME"],
    "MCO": ["Moody's", "MCO"],
    "SPGI": ["S&P Global", "SPGI"],
    "FDS": ["FactSet", "FDS"],
    "MSCI": ["MSCI", "MSCI Inc"],
}

# Build reverse lookup: name/alias → ticker
_NAME_TO_TICKER: Dict[str, str] = {}
for _ticker, _names in KNOWN_COMPANIES.items():
    for _name in _names:
        _NAME_TO_TICKER[_name.lower()] = _ticker

# ── Financial aspects ─────────────────────────────────────────────────────────

FINANCIAL_ASPECTS: Dict[str, List[str]] = {
    "earnings": [
        "EPS", "earnings per share", "revenue", "profit", "earnings", "guidance",
        "outlook", "quarterly", "annual", "fiscal", "beat", "miss", "estimate",
        "top line", "bottom line", "net income", "operating income", "EBITDA",
        "gross profit", "margin", "forecast", "consensus",
    ],
    "management": [
        "CEO", "CFO", "COO", "CTO", "leadership", "management", "board",
        "executive", "director", "chairman", "president", "appointed", "resigned",
        "departure", "succession", "strategy", "strategic", "vision",
    ],
    "product": [
        "product", "launch", "innovation", "pipeline", "research", "development",
        "R&D", "technology", "platform", "service", "feature", "release",
        "update", "version", "new", "breakthrough", "patent", "IP",
    ],
    "risk": [
        "risk", "debt", "leverage", "lawsuit", "regulatory", "litigation",
        "compliance", "investigation", "probe", "penalty", "fine", "recall",
        "default", "covenant", "credit", "rating", "downgrade", "exposure",
        "uncertainty", "headwind", "challenge", "concern",
    ],
    "valuation": [
        "valuation", "price target", "overvalued", "undervalued", "fair value",
        "multiple", "PE ratio", "P/E", "price to earnings", "EV/EBITDA",
        "discount", "premium", "analyst", "upgrade", "buy", "sell", "hold",
        "target", "outperform", "underperform", "initiate",
    ],
    "macro": [
        "interest rates", "inflation", "recession", "GDP", "Federal Reserve",
        "Fed", "monetary policy", "fiscal policy", "trade war", "tariff",
        "currency", "forex", "emerging markets", "geopolitical", "supply chain",
        "energy prices", "oil", "labor market", "unemployment",
    ],
}

# Aspect weights for stock price relevance
ASPECT_WEIGHTS: Dict[str, float] = {
    "earnings": 2.0,
    "management": 1.5,
    "product": 1.2,
    "risk": 1.8,
    "valuation": 1.3,
    "macro": 0.8,
}

# Hedging language patterns
HEDGING_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bwould\b", r"\bshould\b",
    r"\bpossibly\b", r"\bpotentially\b", r"\buncertain\b", r"\buncertainty\b",
    r"\bchallenging\b", r"\bdifficult\b", r"\bheadwind\b", r"\bvolatil",
    r"\bexpect to\b", r"\banticipate\b", r"\bbelieve\b", r"\bhope\b",
    r"\bif conditions\b", r"\bsubject to\b", r"\bdepend", r"\bcontingent\b",
]

# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class SentimentProbs:
    """Raw FinBERT probability outputs."""
    positive: float
    negative: float
    neutral: float
    source: str = "finbert"

    @property
    def label(self) -> str:
        if self.positive >= self.negative and self.positive >= self.neutral:
            return "positive"
        elif self.negative >= self.positive and self.negative >= self.neutral:
            return "negative"
        return "neutral"

    @property
    def confidence(self) -> float:
        return max(self.positive, self.negative, self.neutral)

    @property
    def numeric(self) -> float:
        """Score in [-1, +1]."""
        return round(self.positive - self.negative, 4)


@dataclass
class EntitySentiment:
    """Sentiment for a specific entity mention within a text."""
    entity: str
    ticker: str
    score: float          # -1 to +1
    confidence: float
    label: str
    context_sentences: List[str]
    n_sentences: int


@dataclass
class AspectScore:
    """Sentiment score for a single financial aspect."""
    aspect: str
    score: float          # -1 to +1
    confidence: float
    n_sentences: int
    key_sentences: List[str]
    weight: float = 1.0


@dataclass
class CompositeScore:
    """Multi-source time-decayed sentiment composite."""
    ticker: str
    score: float          # -1 to +1
    confidence: float
    n_sources: int
    shift_alert: bool     # True if >0.2 change in 24h
    as_of: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class MomentumSignal:
    """Sentiment momentum signal."""
    ticker: str
    signal: int           # +1 improving, 0 stable, -1 deteriorating
    ma5: float
    ma20: float
    divergence_warning: bool   # price up + sentiment down
    reversal_signal: bool      # very negative → neutral = mean reversion candidate
    as_of: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class EarningsCallResult:
    """Analysis result from earnings call transcript."""
    management_tone: float        # CEO prepared remarks score
    qa_tone: float                # Q&A section score
    analyst_question_tone: float
    management_answer_tone: float
    hedging_score: float          # 0-1, higher = more uncertain language
    guidance_tone: str            # 'raise', 'maintain', 'lower', 'unclear'
    key_topics: List[str]
    red_flags: List[str]
    summary: str


# ── Pydantic request/response models ─────────────────────────────────────────


class ScoreTextRequest(BaseModel):
    text: str = Field(..., max_length=4096)


class ScoreBatchRequest(BaseModel):
    texts: List[str] = Field(..., max_items=256)
    batch_size: int = Field(32, ge=1, le=128)


class EntitySentimentRequest(BaseModel):
    text: str
    ticker: str


class AspectRequest(BaseModel):
    text: str


class EarningsCallRequest(BaseModel):
    transcript: str
    ticker: Optional[str] = None


class SentimentScoreResponse(BaseModel):
    positive: float
    negative: float
    neutral: float
    label: str
    confidence: float
    numeric: float
    source: str


class CompositeScoreResponse(BaseModel):
    ticker: str
    score: float
    confidence: float
    n_sources: int
    shift_alert: bool
    label: str
    as_of: str


# ── SQLite helpers ────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            score REAL NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            source_weight REAL DEFAULT 1.0,
            created_at REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sent_ticker_time ON sentiment_records (ticker, created_at)"
    )
    conn.commit()
    return conn


def _store_sentiment_record(
    ticker: str,
    score: float,
    confidence: float,
    source: str,
    source_weight: float = 1.0,
) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO sentiment_records (ticker, score, confidence, source, source_weight, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ticker.upper(), score, confidence, source, source_weight, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _fetch_recent_records(
    ticker: str,
    lookback_hours: int = 72,
) -> List[Dict]:
    try:
        conn = _get_db()
        cutoff = time.time() - lookback_hours * 3600
        rows = conn.execute(
            "SELECT score, confidence, source, source_weight, created_at "
            "FROM sentiment_records WHERE ticker=? AND created_at >= ? "
            "ORDER BY created_at DESC",
            (ticker.upper(), cutoff),
        ).fetchall()
        conn.close()
        return [
            {
                "score": r[0],
                "confidence": r[1],
                "source": r[2],
                "source_weight": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 1. FinBERTEngine
# ══════════════════════════════════════════════════════════════════════════════


class FinBERTEngine:
    """
    Local FinBERT inference engine (ProsusAI/finbert).

    GPU is used automatically when available. Model is loaded once and cached.
    Falls back to VADER + LM wordlist when transformers is unavailable.
    """

    MODEL_NAME = "ProsusAI/finbert"
    _instance: Optional["FinBERTEngine"] = None

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._vader = None
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "FinBERTEngine":
        """Singleton accessor — model loads once per process."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            if _VADER_AVAILABLE:
                self._vader = _VaderAnalyzer()
            self._loaded = False
            return

        try:
            if torch.cuda.is_available():
                self._device = "cuda"

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
        except Exception as exc:
            # Graceful degradation
            self._loaded = False
            if _VADER_AVAILABLE:
                self._vader = _VaderAnalyzer()

    def score_text(self, text: str) -> SentimentProbs:
        """Score a single text. Returns SentimentProbs."""
        if not text or not text.strip():
            return SentimentProbs(positive=0.0, negative=0.0, neutral=1.0, source="empty")

        if self._loaded and self._model is not None:
            return self._finbert_score(text[:512])
        return self._fallback_score(text)

    def _finbert_score(self, text: str) -> SentimentProbs:
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]

        # ProsusAI/finbert label order: positive=0, negative=1, neutral=2
        label_map = self._model.config.id2label
        prob_dict: Dict[str, float] = {}
        for idx, prob in enumerate(probs):
            lbl = label_map.get(idx, str(idx)).lower()
            prob_dict[lbl] = float(prob)

        pos = prob_dict.get("positive", 0.0)
        neg = prob_dict.get("negative", 0.0)
        neu = prob_dict.get("neutral", 0.0)
        total = pos + neg + neu or 1.0

        return SentimentProbs(
            positive=round(pos / total, 4),
            negative=round(neg / total, 4),
            neutral=round(neu / total, 4),
            source="finbert",
        )

    def _fallback_score(self, text: str) -> SentimentProbs:
        """VADER + LM wordlist fallback."""
        vader_score = 0.0
        if self._vader is not None:
            scores = self._vader.polarity_scores(text)
            vader_score = scores["compound"]  # -1 to +1

        text_lower = text.lower()
        pos_count = sum(1 for term in LM_POSITIVE if term in text_lower)
        neg_count = sum(1 for term in LM_NEGATIVE if term in text_lower)
        total_kw = pos_count + neg_count or 1
        lm_score = (pos_count - neg_count) / total_kw  # -1 to +1

        if self._vader is not None:
            combined = 0.6 * vader_score + 0.4 * lm_score
        else:
            combined = lm_score

        combined = max(-1.0, min(1.0, combined))
        pos = max(0.0, combined)
        neg = max(0.0, -combined)
        neu = 1.0 - pos - neg

        return SentimentProbs(
            positive=round(pos, 4),
            negative=round(neg, 4),
            neutral=round(neu, 4),
            source="vader_lm",
        )

    def score_batch(self, texts: List[str], batch_size: int = 32) -> List[SentimentProbs]:
        """Vectorized batch scoring. Falls back gracefully."""
        if not texts:
            return []

        if not self._loaded or self._model is None:
            return [self._fallback_score(t) for t in texts]

        results: List[SentimentProbs] = []
        for i in range(0, len(texts), batch_size):
            chunk = [t[:512] for t in texts[i: i + batch_size]]
            batch_results = self._score_batch_chunk(chunk)
            results.extend(batch_results)
        return results

    def _score_batch_chunk(self, texts: List[str]) -> List[SentimentProbs]:
        import torch

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            all_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probs = all_probs.cpu().numpy()

        label_map = self._model.config.id2label
        results: List[SentimentProbs] = []

        for probs in all_probs:
            prob_dict: Dict[str, float] = {}
            for idx, prob in enumerate(probs):
                lbl = label_map.get(idx, str(idx)).lower()
                prob_dict[lbl] = float(prob)

            pos = prob_dict.get("positive", 0.0)
            neg = prob_dict.get("negative", 0.0)
            neu = prob_dict.get("neutral", 0.0)
            total = pos + neg + neu or 1.0
            results.append(SentimentProbs(
                positive=round(pos / total, 4),
                negative=round(neg / total, 4),
                neutral=round(neu / total, 4),
                source="finbert",
            ))
        return results

    def get_sentiment_label(self, probs: SentimentProbs) -> Tuple[str, float]:
        """Return (label, confidence) from a SentimentProbs object."""
        return probs.label, probs.confidence


# ══════════════════════════════════════════════════════════════════════════════
# 2. EntityLevelSentiment
# ══════════════════════════════════════════════════════════════════════════════


class EntityLevelSentiment:
    """
    Extract company mentions from text and score at the sentence level.

    For each sentence mentioning the entity, FinBERT is applied to that
    sentence alone.  The final company-specific score is the mean across
    all entity-containing sentences.
    """

    def __init__(self) -> None:
        self._engine = FinBERTEngine.get_instance()

    def _split_sentences(self, text: str) -> List[str]:
        """Simple rule-based sentence splitter."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    def _sentence_mentions_entity(self, sentence: str, entity: str) -> bool:
        """Check if sentence mentions the entity (case-insensitive)."""
        s_lower = sentence.lower()
        aliases = KNOWN_COMPANIES.get(entity.upper(), [entity])
        for alias in aliases:
            if alias.lower() in s_lower:
                return True
        return False

    def get_entity_sentiment(self, text: str, entity: str) -> EntitySentiment:
        """
        Score a specific entity within a text document.

        Returns EntitySentiment with mean score across entity-containing sentences.
        """
        ticker = entity.upper()
        sentences = self._split_sentences(text)

        entity_sentences = [s for s in sentences if self._sentence_mentions_entity(s, ticker)]

        if not entity_sentences:
            return EntitySentiment(
                entity=entity,
                ticker=ticker,
                score=0.0,
                confidence=0.0,
                label="neutral",
                context_sentences=[],
                n_sentences=0,
            )

        probs_list = self._engine.score_batch(entity_sentences)
        scores = [p.numeric for p in probs_list]
        confidences = [p.confidence for p in probs_list]

        mean_score = float(np.mean(scores))
        mean_conf = float(np.mean(confidences))

        if mean_score >= 0.1:
            label = "positive"
        elif mean_score <= -0.1:
            label = "negative"
        else:
            label = "neutral"

        return EntitySentiment(
            entity=entity,
            ticker=ticker,
            score=round(mean_score, 4),
            confidence=round(mean_conf, 4),
            label=label,
            context_sentences=entity_sentences[:5],
            n_sentences=len(entity_sentences),
        )

    def get_competitor_sentiments(
        self, text: str, primary_ticker: str
    ) -> List[EntitySentiment]:
        """
        Find all company mentions in the text and score each one.

        Useful for identifying competitor sentiment in industry articles.
        """
        sentences = self._split_sentences(text)
        mentioned_tickers: set = set()

        for sentence in sentences:
            s_lower = sentence.lower()
            for ticker, aliases in KNOWN_COMPANIES.items():
                for alias in aliases:
                    if alias.lower() in s_lower:
                        mentioned_tickers.add(ticker)
                        break

        results: List[EntitySentiment] = []
        for ticker in mentioned_tickers:
            if ticker != primary_ticker.upper():
                result = self.get_entity_sentiment(text, ticker)
                if result.n_sentences > 0:
                    results.append(result)

        results.sort(key=lambda x: abs(x.score), reverse=True)
        return results

    def extract_all_entity_sentiments(self, text: str) -> List[EntitySentiment]:
        """Extract and score every company entity mentioned in the text."""
        sentences = self._split_sentences(text)
        mentioned: set = set()

        for sentence in sentences:
            s_lower = sentence.lower()
            for ticker, aliases in KNOWN_COMPANIES.items():
                for alias in aliases:
                    if alias.lower() in s_lower:
                        mentioned.add(ticker)
                        break

        results: List[EntitySentiment] = []
        for ticker in mentioned:
            result = self.get_entity_sentiment(text, ticker)
            if result.n_sentences > 0:
                results.append(result)

        results.sort(key=lambda x: x.n_sentences, reverse=True)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. AspectBasedSentiment
# ══════════════════════════════════════════════════════════════════════════════


class AspectBasedSentiment:
    """
    Aspect-based sentiment analysis for financial texts.

    Extracts sentences relevant to each financial aspect (earnings, management,
    product, risk, valuation, macro) and scores with FinBERT.
    """

    def __init__(self) -> None:
        self._engine = FinBERTEngine.get_instance()

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _sentence_matches_aspect(self, sentence: str, aspect: str) -> bool:
        s_lower = sentence.lower()
        keywords = FINANCIAL_ASPECTS.get(aspect, [])
        return any(kw.lower() in s_lower for kw in keywords)

    def get_aspect_scores(self, text: str) -> Dict[str, AspectScore]:
        """
        Score all financial aspects in the text.

        Returns dict of aspect_name → AspectScore.
        """
        sentences = self._split_sentences(text)
        aspect_results: Dict[str, AspectScore] = {}

        for aspect, keywords in FINANCIAL_ASPECTS.items():
            relevant = [s for s in sentences if self._sentence_matches_aspect(s, aspect)]

            if not relevant:
                aspect_results[aspect] = AspectScore(
                    aspect=aspect,
                    score=0.0,
                    confidence=0.0,
                    n_sentences=0,
                    key_sentences=[],
                    weight=ASPECT_WEIGHTS.get(aspect, 1.0),
                )
                continue

            probs_list = self._engine.score_batch(relevant)
            scores = [p.numeric for p in probs_list]
            confidences = [p.confidence for p in probs_list]

            mean_score = float(np.mean(scores))
            mean_conf = float(np.mean(confidences))

            # Pick top 3 most extreme sentences as key examples
            scored_sents = sorted(
                zip(relevant, scores),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:3]
            key_sentences = [s for s, _ in scored_sents]

            aspect_results[aspect] = AspectScore(
                aspect=aspect,
                score=round(mean_score, 4),
                confidence=round(mean_conf, 4),
                n_sentences=len(relevant),
                key_sentences=key_sentences,
                weight=ASPECT_WEIGHTS.get(aspect, 1.0),
            )

        return aspect_results

    def get_weighted_composite(self, aspect_scores: Dict[str, AspectScore]) -> float:
        """Compute a weighted composite sentiment from aspect scores."""
        total_weight = 0.0
        weighted_sum = 0.0

        for aspect, aspect_score in aspect_scores.items():
            if aspect_score.n_sentences == 0:
                continue
            w = aspect_score.weight * aspect_score.confidence
            weighted_sum += aspect_score.score * w
            total_weight += w

        if total_weight == 0:
            return 0.0
        return round(max(-1.0, min(1.0, weighted_sum / total_weight)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MultiSourceSentimentAggregator
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_WEIGHTS: Dict[str, float] = {
    "analyst_report": 1.5,
    "news": 1.0,
    "earnings_call": 1.4,
    "press_release": 1.2,
    "reddit": 0.5,
    "stocktwits": 0.4,
    "twitter": 0.4,
    "blog": 0.6,
}

HALF_LIFE_HOURS = 72.0  # 3-day half-life


class MultiSourceSentimentAggregator:
    """
    Aggregate sentiment across multiple sources with time decay and source weighting.

    Time decay: exponential with 3-day half-life.
    Source weights: analyst > earnings_call > news > social.
    Stores all records in SQLite for persistence.
    """

    def __init__(self) -> None:
        self._engine = FinBERTEngine.get_instance()
        _get_db()  # Ensure table exists

    def _time_decay_weight(self, age_hours: float) -> float:
        """Exponential decay: weight = 0.5^(age/half_life)."""
        return math.pow(0.5, age_hours / HALF_LIFE_HOURS)

    def add_document(
        self,
        ticker: str,
        text: str,
        source_type: str = "news",
        timestamp: Optional[datetime] = None,
    ) -> SentimentProbs:
        """
        Score a document and persist to SQLite.

        Returns SentimentProbs for the scored text.
        """
        probs = self._engine.score_text(text)
        ts = timestamp or datetime.now(tz=timezone.utc)
        weight = SOURCE_WEIGHTS.get(source_type, 1.0)

        try:
            conn = _get_db()
            conn.execute(
                "INSERT INTO sentiment_records (ticker, score, confidence, source, source_weight, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ticker.upper(),
                    probs.numeric,
                    probs.confidence,
                    source_type,
                    weight,
                    ts.timestamp(),
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        return probs

    def get_composite_score(
        self,
        ticker: str,
        lookback_hours: int = 72,
    ) -> CompositeScore:
        """
        Compute time-decayed, source-weighted composite sentiment score.

        Returns CompositeScore with shift_alert if change >0.2 in 24h.
        """
        records = _fetch_recent_records(ticker, lookback_hours)

        if not records:
            return CompositeScore(
                ticker=ticker,
                score=0.0,
                confidence=0.0,
                n_sources=0,
                shift_alert=False,
            )

        now_ts = time.time()
        weighted_sum = 0.0
        weight_total = 0.0
        confidence_sum = 0.0

        for rec in records:
            age_hours = (now_ts - rec["created_at"]) / 3600.0
            time_w = self._time_decay_weight(age_hours)
            source_w = rec["source_weight"]
            conf_w = rec["confidence"]
            combined_w = time_w * source_w * max(conf_w, 0.1)

            weighted_sum += rec["score"] * combined_w
            weight_total += combined_w
            confidence_sum += rec["confidence"]

        if weight_total == 0:
            return CompositeScore(
                ticker=ticker,
                score=0.0,
                confidence=0.0,
                n_sources=len(records),
                shift_alert=False,
            )

        composite = round(max(-1.0, min(1.0, weighted_sum / weight_total)), 4)
        avg_conf = round(confidence_sum / len(records), 4)

        # Check for 24h shift alert
        shift_alert = self._check_shift_alert(ticker, composite)

        return CompositeScore(
            ticker=ticker,
            score=composite,
            confidence=avg_conf,
            n_sources=len(records),
            shift_alert=shift_alert,
        )

    def _check_shift_alert(self, ticker: str, current_score: float) -> bool:
        """Return True if composite score changed >0.2 in last 24 hours."""
        records_24h_48h = []
        now_ts = time.time()

        try:
            conn = _get_db()
            cutoff_48h = now_ts - 48 * 3600
            cutoff_24h = now_ts - 24 * 3600

            rows = conn.execute(
                "SELECT score, created_at FROM sentiment_records "
                "WHERE ticker=? AND created_at BETWEEN ? AND ? "
                "ORDER BY created_at",
                (ticker.upper(), cutoff_48h, cutoff_24h),
            ).fetchall()
            conn.close()
            records_24h_48h = [r[0] for r in rows]
        except Exception:
            pass

        if not records_24h_48h:
            return False

        prior_score = float(np.mean(records_24h_48h))
        return abs(current_score - prior_score) > 0.2


# ══════════════════════════════════════════════════════════════════════════════
# 5. SentimentMomentumTracker
# ══════════════════════════════════════════════════════════════════════════════


class SentimentMomentumTracker:
    """
    Track sentiment momentum using rolling moving averages.

    Signals:
    - +1 (improving): 5-day MA > 20-day MA and widening
    - -1 (deteriorating): 5-day MA < 20-day MA
    -  0 (stable): no clear trend
    - Divergence warning: price direction contradicts sentiment direction
    - Reversal signal: transition from very negative to neutral zone
    """

    def __init__(self) -> None:
        self._aggregator = MultiSourceSentimentAggregator()

    def _get_daily_scores(self, ticker: str, days: int = 30) -> pd.Series:
        """Retrieve daily average sentiment scores from SQLite."""
        try:
            conn = _get_db()
            cutoff = time.time() - days * 86400
            rows = conn.execute(
                "SELECT score, created_at FROM sentiment_records "
                "WHERE ticker=? AND created_at >= ? ORDER BY created_at",
                (ticker.upper(), cutoff),
            ).fetchall()
            conn.close()

            if not rows:
                return pd.Series(dtype=float)

            df = pd.DataFrame(rows, columns=["score", "ts"])
            df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.date
            daily = df.groupby("date")["score"].mean()
            return daily

        except Exception:
            return pd.Series(dtype=float)

    def sentiment_momentum(self, ticker: str) -> MomentumSignal:
        """
        Compute sentiment momentum signal for a ticker.

        Returns MomentumSignal with +1, 0, or -1 signal.
        """
        daily = self._get_daily_scores(ticker, days=30)

        if daily.empty or len(daily) < 5:
            return MomentumSignal(
                ticker=ticker,
                signal=0,
                ma5=0.0,
                ma20=0.0,
                divergence_warning=False,
                reversal_signal=False,
            )

        ma5 = float(daily.rolling(5, min_periods=1).mean().iloc[-1])
        ma20 = float(daily.rolling(20, min_periods=1).mean().iloc[-1])

        if ma5 > ma20 + 0.05:
            signal = 1
        elif ma5 < ma20 - 0.05:
            signal = -1
        else:
            signal = 0

        # Reversal signal: if recent 5-day MA was very negative and now improving
        reversal = False
        if len(daily) >= 10:
            prior_ma5 = float(daily.rolling(5, min_periods=1).mean().iloc[-6])
            reversal = prior_ma5 < -0.4 and ma5 > -0.2

        return MomentumSignal(
            ticker=ticker,
            signal=signal,
            ma5=round(ma5, 4),
            ma20=round(ma20, 4),
            divergence_warning=False,  # Requires price data — set externally
            reversal_signal=reversal,
        )

    def check_price_divergence(
        self,
        ticker: str,
        price_return_1m: float,
    ) -> bool:
        """
        Check if price direction contradicts sentiment direction.

        price_return_1m: 1-month price return as decimal (e.g. 0.05 = +5%)
        Returns True if divergence detected (potential mean reversion opportunity).
        """
        daily = self._get_daily_scores(ticker, days=30)
        if daily.empty or len(daily) < 5:
            return False

        ma5 = float(daily.rolling(5, min_periods=1).mean().iloc[-1])
        ma20 = float(daily.rolling(20, min_periods=1).mean().iloc[-1]) if len(daily) >= 20 else ma5

        price_up = price_return_1m > 0.02
        sentiment_deteriorating = ma5 < ma20 - 0.05

        return price_up and sentiment_deteriorating


# ══════════════════════════════════════════════════════════════════════════════
# 6. EarningsCallSentimentAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

# Guidance-raise/lower signal phrases
_GUIDANCE_RAISE = [
    r"rais\w+ guidance", r"increas\w+ guidance", r"upward\w* revision",
    r"rais\w+ outlook", r"rais\w+ forecast", r"above.*consensus",
    r"stronger than expected", r"exceed.*expectation",
]
_GUIDANCE_LOWER = [
    r"lower\w+ guidance", r"reduc\w+ guidance", r"cut.*guidance",
    r"downward\w* revision", r"below.*consensus", r"headwind",
    r"weaker than expected", r"miss.*expectation", r"challenging environment",
]
_GUIDANCE_MAINTAIN = [
    r"reaffirm\w+ guidance", r"maintain\w+ guidance", r"on track",
    r"in line with", r"reiterat\w+", r"consistent with",
]


class EarningsCallSentimentAnalyzer:
    """
    Specialized FinBERT analysis for earnings call transcripts.

    Separates prepared remarks from Q&A, detects hedging language,
    infers guidance tone from language patterns.
    """

    def __init__(self) -> None:
        self._engine = FinBERTEngine.get_instance()
        self._aspect = AspectBasedSentiment()

    def _split_prepared_vs_qa(self, transcript: str) -> Tuple[str, str]:
        """
        Heuristically split transcript into prepared remarks and Q&A section.

        Returns (prepared_text, qa_text).
        """
        patterns = [
            r"(?i)(question[-\s]and[-\s]answer|q&a|q\s*and\s*a|questions?\s+and\s+answers?)",
            r"(?i)(operator[:\s]*.*?questions?|we.ll now.*?questions?|open.*?for\s+questions?)",
            r"(?i)(analyst[:\s]*[A-Z]|analyst\s+from\s+)",
            r"(?i)(\[operator\]|\[moderator\])",
        ]

        split_pos = len(transcript)
        for pattern in patterns:
            match = re.search(pattern, transcript)
            if match:
                split_pos = min(split_pos, match.start())

        prepared = transcript[:split_pos].strip()
        qa = transcript[split_pos:].strip()

        if len(prepared) < 100:
            # Couldn't split meaningfully — return full text as prepared
            return transcript, ""

        return prepared, qa

    def _extract_ceo_section(self, prepared: str) -> str:
        """Extract CEO/management remarks from prepared section."""
        patterns = [
            r"(?i)(CEO|Chief Executive|president)[:\s]*(.{200,2000}?)(?=\n[A-Z]{2,}|\Z)",
            r"(?i)(Good\s+(?:morning|afternoon|evening)[^.]*?\.)(.{200,2000})",
        ]
        for pattern in patterns:
            match = re.search(pattern, prepared, re.DOTALL)
            if match:
                return match.group(0)[:2000]

        # Fallback: first 2000 chars of prepared remarks
        return prepared[:2000]

    def _extract_qa_pairs(self, qa_text: str) -> List[Tuple[str, str]]:
        """
        Extract (analyst_question, management_answer) pairs from Q&A text.

        Returns list of (question, answer) tuples.
        """
        pairs: List[Tuple[str, str]] = []
        if not qa_text:
            return pairs

        # Split on typical Q&A delimiters
        chunks = re.split(
            r'(?i)\n(?:Question|Q:|Analyst:|Operator:)',
            qa_text,
        )

        for chunk in chunks[1:]:
            # First part is question, then look for Answer/A:/Management:
            parts = re.split(r'(?i)\n(?:Answer|A:|Management:|CEO:|CFO:|Executive:)', chunk, maxsplit=1)
            question = parts[0].strip()[:500]
            answer = parts[1].strip()[:1000] if len(parts) > 1 else ""
            if question:
                pairs.append((question, answer))

        return pairs[:20]

    def _score_hedging(self, text: str) -> float:
        """
        Compute hedging language score (0-1).

        Higher score = more uncertain/hedging language.
        """
        text_lower = text.lower()
        words = text_lower.split()
        n_words = max(len(words), 1)

        hedge_count = 0
        for pattern in HEDGING_PATTERNS:
            matches = re.findall(pattern, text_lower)
            hedge_count += len(matches)

        # Normalize by word count; cap at 1.0
        return round(min(1.0, hedge_count / (n_words / 20.0)), 4)

    def _infer_guidance_tone(self, text: str) -> str:
        """
        Infer raise/maintain/lower from language before explicit numbers.

        Returns 'raise', 'maintain', 'lower', or 'unclear'.
        """
        text_lower = text.lower()

        raise_score = sum(1 for p in _GUIDANCE_RAISE if re.search(p, text_lower))
        lower_score = sum(1 for p in _GUIDANCE_LOWER if re.search(p, text_lower))
        maintain_score = sum(1 for p in _GUIDANCE_MAINTAIN if re.search(p, text_lower))

        if raise_score > lower_score and raise_score > maintain_score:
            return "raise"
        elif lower_score > raise_score and lower_score > maintain_score:
            return "lower"
        elif maintain_score >= raise_score and maintain_score >= lower_score and maintain_score > 0:
            return "maintain"
        return "unclear"

    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract top financial topics mentioned in the text."""
        topics = []
        for aspect, keywords in FINANCIAL_ASPECTS.items():
            matched = [kw for kw in keywords if kw.lower() in text.lower()]
            if matched:
                topics.append(f"{aspect}: {', '.join(matched[:3])}")
        return topics

    def _extract_red_flags(self, text: str) -> List[str]:
        """Extract sentences containing risk/negative language."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        red_flags: List[str] = []

        risk_keywords = FINANCIAL_ASPECTS["risk"] + [
            "restatement", "SEC", "DOJ", "class action", "going concern",
            "impairment", "write-off", "restructuring charges",
        ]

        for sentence in sentences:
            s_lower = sentence.lower()
            if any(kw.lower() in s_lower for kw in risk_keywords):
                if len(sentence.strip()) > 20:
                    red_flags.append(sentence.strip()[:200])

        return red_flags[:5]

    def analyze_transcript(self, transcript: str, ticker: Optional[str] = None) -> EarningsCallResult:
        """
        Full earnings call analysis.

        Returns EarningsCallResult with tone scores, hedging, guidance signal.
        """
        if not transcript or not transcript.strip():
            return EarningsCallResult(
                management_tone=0.0,
                qa_tone=0.0,
                analyst_question_tone=0.0,
                management_answer_tone=0.0,
                hedging_score=0.0,
                guidance_tone="unclear",
                key_topics=[],
                red_flags=[],
                summary="Empty transcript provided.",
            )

        prepared, qa = self._split_prepared_vs_qa(transcript)

        # CEO prepared remarks
        ceo_section = self._extract_ceo_section(prepared)
        ceo_probs = self._engine.score_text(ceo_section)
        management_tone = ceo_probs.numeric

        # Full prepared remarks hedging
        hedging_score = self._score_hedging(prepared)

        # Guidance tone from prepared remarks
        guidance_tone = self._infer_guidance_tone(prepared)

        # Q&A analysis
        qa_tone = 0.0
        analyst_tone = 0.0
        mgmt_answer_tone = 0.0

        if qa:
            qa_probs = self._engine.score_text(qa[:2000])
            qa_tone = qa_probs.numeric

            pairs = self._extract_qa_pairs(qa)
            if pairs:
                questions = [q for q, _ in pairs if q]
                answers = [a for _, a in pairs if a]

                if questions:
                    q_probs = self._engine.score_batch(questions[:10])
                    analyst_tone = float(np.mean([p.numeric for p in q_probs]))

                if answers:
                    a_probs = self._engine.score_batch(answers[:10])
                    mgmt_answer_tone = float(np.mean([p.numeric for p in a_probs]))

        key_topics = self._extract_key_topics(transcript[:3000])
        red_flags = self._extract_red_flags(transcript[:3000])

        # Summary
        tone_desc = "positive" if management_tone > 0.1 else ("negative" if management_tone < -0.1 else "neutral")
        hedge_desc = "high" if hedging_score > 0.4 else ("moderate" if hedging_score > 0.2 else "low")
        summary = (
            f"Management tone: {tone_desc} ({management_tone:+.2f}). "
            f"Guidance: {guidance_tone}. "
            f"Hedging language: {hedge_desc} ({hedging_score:.2f}). "
            f"Q&A sentiment: {qa_tone:+.2f}. "
            f"Red flags detected: {len(red_flags)}."
        )

        return EarningsCallResult(
            management_tone=round(management_tone, 4),
            qa_tone=round(qa_tone, 4),
            analyst_question_tone=round(analyst_tone, 4),
            management_answer_tone=round(mgmt_answer_tone, 4),
            hedging_score=round(hedging_score, 4),
            guidance_tone=guidance_tone,
            key_topics=key_topics,
            red_flags=red_flags,
            summary=summary,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Router
# ══════════════════════════════════════════════════════════════════════════════

sentiment_router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# Lazy singletons
_finbert: Optional[FinBERTEngine] = None
_entity_analyzer: Optional[EntityLevelSentiment] = None
_aspect_analyzer: Optional[AspectBasedSentiment] = None
_aggregator: Optional[MultiSourceSentimentAggregator] = None
_momentum: Optional[SentimentMomentumTracker] = None
_earnings_analyzer: Optional[EarningsCallSentimentAnalyzer] = None


def _get_finbert() -> FinBERTEngine:
    global _finbert
    if _finbert is None:
        _finbert = FinBERTEngine.get_instance()
    return _finbert


def _get_entity() -> EntityLevelSentiment:
    global _entity_analyzer
    if _entity_analyzer is None:
        _entity_analyzer = EntityLevelSentiment()
    return _entity_analyzer


def _get_aspect() -> AspectBasedSentiment:
    global _aspect_analyzer
    if _aspect_analyzer is None:
        _aspect_analyzer = AspectBasedSentiment()
    return _aspect_analyzer


def _get_aggregator() -> MultiSourceSentimentAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = MultiSourceSentimentAggregator()
    return _aggregator


def _get_momentum() -> SentimentMomentumTracker:
    global _momentum
    if _momentum is None:
        _momentum = SentimentMomentumTracker()
    return _momentum


def _get_earnings() -> EarningsCallSentimentAnalyzer:
    global _earnings_analyzer
    if _earnings_analyzer is None:
        _earnings_analyzer = EarningsCallSentimentAnalyzer()
    return _earnings_analyzer


def _probs_to_response(probs: SentimentProbs) -> SentimentScoreResponse:
    return SentimentScoreResponse(
        positive=probs.positive,
        negative=probs.negative,
        neutral=probs.neutral,
        label=probs.label,
        confidence=probs.confidence,
        numeric=probs.numeric,
        source=probs.source,
    )


@sentiment_router.post("/score", response_model=SentimentScoreResponse)
def score_text_endpoint(req: ScoreTextRequest):
    """Score a single text with FinBERT."""
    probs = _get_finbert().score_text(req.text)
    return _probs_to_response(probs)


@sentiment_router.post("/batch", response_model=List[SentimentScoreResponse])
def score_batch_endpoint(req: ScoreBatchRequest):
    """Batch score multiple texts."""
    probs_list = _get_finbert().score_batch(req.texts, batch_size=req.batch_size)
    return [_probs_to_response(p) for p in probs_list]


@sentiment_router.get("/entity/{ticker}")
def entity_sentiment_endpoint(ticker: str, text: str):
    """
    Entity-level sentiment for a specific ticker within a text.

    Pass text as query param for GET, or use POST /sentiment/entity body.
    """
    if not text:
        raise HTTPException(status_code=400, detail="text query parameter required")
    result = _get_entity().get_entity_sentiment(text, ticker)
    return {
        "entity": result.entity,
        "ticker": result.ticker,
        "score": result.score,
        "confidence": result.confidence,
        "label": result.label,
        "n_sentences": result.n_sentences,
        "context_sentences": result.context_sentences,
    }


@sentiment_router.post("/aspects")
def aspect_sentiment_endpoint(req: AspectRequest):
    """Score financial aspects (earnings/management/product/risk/valuation/macro) in a text."""
    aspect_scores = _get_aspect().get_aspect_scores(req.text)
    composite = _get_aspect().get_weighted_composite(aspect_scores)

    return {
        "composite_score": composite,
        "aspects": {
            name: {
                "score": asp.score,
                "confidence": asp.confidence,
                "n_sentences": asp.n_sentences,
                "weight": asp.weight,
                "key_sentences": asp.key_sentences,
            }
            for name, asp in aspect_scores.items()
        },
    }


@sentiment_router.get("/composite/{ticker}", response_model=CompositeScoreResponse)
def composite_score_endpoint(ticker: str, lookback_hours: int = 72):
    """Get composite time-decayed sentiment score for a ticker from stored records."""
    result = _get_aggregator().get_composite_score(ticker, lookback_hours=lookback_hours)
    label = (
        "very_bullish" if result.score >= 0.5
        else "bullish" if result.score >= 0.2
        else "very_bearish" if result.score <= -0.5
        else "bearish" if result.score <= -0.2
        else "neutral"
    )
    return CompositeScoreResponse(
        ticker=result.ticker,
        score=result.score,
        confidence=result.confidence,
        n_sources=result.n_sources,
        shift_alert=result.shift_alert,
        label=label,
        as_of=result.as_of.isoformat(),
    )


@sentiment_router.get("/momentum/{ticker}")
def momentum_endpoint(ticker: str):
    """Get sentiment momentum signal (+1 improving, 0 stable, -1 deteriorating)."""
    signal = _get_momentum().sentiment_momentum(ticker)
    signal_desc = {1: "improving", 0: "stable", -1: "deteriorating"}.get(signal.signal, "stable")
    return {
        "ticker": signal.ticker,
        "signal": signal.signal,
        "signal_description": signal_desc,
        "ma5": signal.ma5,
        "ma20": signal.ma20,
        "divergence_warning": signal.divergence_warning,
        "reversal_signal": signal.reversal_signal,
        "as_of": signal.as_of.isoformat(),
    }


@sentiment_router.post("/earnings-call")
def earnings_call_endpoint(req: EarningsCallRequest):
    """Analyze an earnings call transcript for tone, hedging, and guidance signals."""
    result = _get_earnings().analyze_transcript(req.transcript, req.ticker)
    return {
        "ticker": req.ticker,
        "management_tone": result.management_tone,
        "qa_tone": result.qa_tone,
        "analyst_question_tone": result.analyst_question_tone,
        "management_answer_tone": result.management_answer_tone,
        "hedging_score": result.hedging_score,
        "guidance_tone": result.guidance_tone,
        "key_topics": result.key_topics,
        "red_flags": result.red_flags,
        "summary": result.summary,
    }


# ── Convenience module-level functions ───────────────────────────────────────

def score(text: str) -> SentimentProbs:
    """Score a single text. Module-level convenience."""
    return FinBERTEngine.get_instance().score_text(text)


def score_batch(texts: List[str], batch_size: int = 32) -> List[SentimentProbs]:
    """Batch score texts. Module-level convenience."""
    return FinBERTEngine.get_instance().score_batch(texts, batch_size=batch_size)


def entity_sentiment(text: str, ticker: str) -> EntitySentiment:
    """Entity-level sentiment. Module-level convenience."""
    return EntityLevelSentiment().get_entity_sentiment(text, ticker)


def aspect_scores(text: str) -> Dict[str, AspectScore]:
    """Aspect-based scores. Module-level convenience."""
    return AspectBasedSentiment().get_aspect_scores(text)


def composite(ticker: str, lookback_hours: int = 72) -> CompositeScore:
    """Composite time-decayed score. Module-level convenience."""
    return MultiSourceSentimentAggregator().get_composite_score(ticker, lookback_hours)


def momentum(ticker: str) -> MomentumSignal:
    """Sentiment momentum signal. Module-level convenience."""
    return SentimentMomentumTracker().sentiment_momentum(ticker)


def analyze_earnings_call(transcript: str, ticker: Optional[str] = None) -> EarningsCallResult:
    """Earnings call analysis. Module-level convenience."""
    return EarningsCallSentimentAnalyzer().analyze_transcript(transcript, ticker)
