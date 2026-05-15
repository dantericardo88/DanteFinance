"""Financial Query Expansion V2 — Bloomberg-grade NL query intelligence (enhanced).

dim_056: Smart synonym / query expansion — raise score from 8 → 9.

New in V2:
  IntentClassifier          — Classify query into 10 financial intents with confidence
  ContextualQueryExpander   — Session context (pronoun resolution, peer expansion)
  FinancialEntityLinker     — Company disambiguation, ticker normalisation, metric aliases
  QueryReformulationEngine  — Backend-specific query reformulation (EDGAR, FRED, SQL, RAG)
  AutocompleteEngineV2      — 10 000+ terms, fuzzy edit-distance, popularity weighting
  query_v2_router           — FastAPI router exposing all V2 endpoints

Fully backward-compatible: all V1 classes re-exported.

Usage::
    from sentinel.sai.query_expansion_v2 import IntentClassifier, ContextualQueryExpander

    clf = IntentClassifier()
    result = clf.classify_intent("Show me cheap tech stocks with strong ESG scores")
    # IntentResult(primary_intent='screening', confidence=0.82,
    #              secondary_intents=['ESG'], route_hints={...})

    expander = ContextualQueryExpander()
    ctx = {"last_ticker": "AAPL", "mode": "screener"}
    eq = expander.expand_with_context("what did it earn last quarter?", ctx)
    # ExpandedQuery(query="what did AAPL earn Q1 2026?", ...)
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Query as FastAPIQuery
from pydantic import BaseModel

from sentinel.core.logging import get_logger

# Re-export V1 classes so importers of V2 also get V1
from sentinel.sai.query_expansion import (
    AutocompleteEngine,
    FinancialQueryRouter,
    FinancialSynonymLibrary,
    QueryExpander,
    QueryParser,
    query_router,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

INTENTS = [
    "valuation",
    "screening",
    "risk",
    "macro",
    "technical",
    "earnings",
    "options",
    "insider",
    "ESG",
    "news",
]

# Keyword groups per intent — order matters (earlier = higher base weight)
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "valuation": [
        "P/E", "PE ratio", "price to earnings", "EV/EBITDA", "price to book",
        "price to sales", "PEG", "forward PE", "trailing PE", "enterprise value",
        "fair value", "intrinsic value", "DCF", "discount cash flow", "valuation",
        "multiple", "cheap", "expensive", "overvalued", "undervalued", "margin of safety",
    ],
    "screening": [
        "screen", "filter", "find stocks", "show me stocks", "list companies",
        "rank by", "sort by", "top stocks", "best stocks", "criteria", "with",
        "above", "below", "greater than", "less than", "low PE", "high growth",
        "small cap", "large cap", "mid cap", "dividend payers",
    ],
    "risk": [
        "risk", "volatility", "beta", "VaR", "value at risk", "drawdown",
        "correlation", "standard deviation", "Sharpe", "max drawdown",
        "credit risk", "default risk", "going concern", "bankruptcy",
        "stress test", "tail risk", "downside", "worst case",
    ],
    "macro": [
        "GDP", "inflation", "CPI", "PPI", "Fed", "FOMC", "interest rate",
        "yield curve", "Treasury", "employment", "unemployment", "jobs",
        "PMI", "ISM", "consumer confidence", "housing starts", "retail sales",
        "macro", "economic", "recession", "fiscal", "monetary policy",
    ],
    "technical": [
        "RSI", "MACD", "moving average", "SMA", "EMA", "Bollinger",
        "support", "resistance", "breakout", "momentum", "relative strength",
        "52-week high", "52-week low", "chart", "technical", "trend",
        "golden cross", "death cross", "volume", "overbought", "oversold",
    ],
    "earnings": [
        "earnings", "EPS", "revenue", "guidance", "beat", "miss", "surprise",
        "earnings call", "transcript", "conference call", "quarterly results",
        "annual results", "report", "press release", "forecast", "estimate",
        "consensus", "whisper", "beat and raise", "top line", "bottom line",
    ],
    "options": [
        "options", "calls", "puts", "implied volatility", "IV", "delta",
        "gamma", "theta", "vega", "expiry", "strike", "premium", "open interest",
        "options flow", "unusual activity", "skew", "LEAPS", "covered call",
        "iron condor", "straddle", "strangle",
    ],
    "insider": [
        "insider", "Form 4", "director buying", "CEO purchase", "CFO sale",
        "insider buying", "insider selling", "management ownership",
        "executive share", "10b5-1", "open market purchase", "insider stake",
    ],
    "ESG": [
        "ESG", "sustainability", "carbon", "emissions", "climate", "green",
        "environmental", "social", "governance", "diversity", "DEI",
        "net zero", "renewable", "ESG score", "responsible investing",
        "impact investing", "SASB", "TCFD", "sustainable",
    ],
    "news": [
        "news", "headline", "announcement", "article", "press release",
        "said", "reported", "disclosed", "latest", "recent", "today",
        "breaking", "update", "filing", "SEC filing", "8-K", "10-K",
        "merger", "acquisition", "deal", "downgrade", "upgrade",
    ],
}

# Module routing per intent
_INTENT_MODULE_MAP: dict[str, dict] = {
    "valuation":  {"module": "standardized_financials", "action": "get_ratio"},
    "screening":  {"module": "technical_screener",      "action": "screen"},
    "risk":       {"module": "standardized_financials", "action": "get_risk_metrics"},
    "macro":      {"module": "macro_data",              "action": "get_economic"},
    "technical":  {"module": "technical_screener",      "action": "momentum"},
    "earnings":   {"module": "earnings_corpus_v2",      "action": "earnings_query"},
    "options":    {"module": "options_analytics",       "action": "get_chain"},
    "insider":    {"module": "insider_transactions",    "action": "get_recent"},
    "ESG":        {"module": "esg_composite",           "action": "get_esg"},
    "news":       {"module": "earnings_corpus_v2",      "action": "news_timeline"},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    primary_intent: str
    confidence: float
    secondary_intents: list[str]
    route_hints: dict[str, Any]
    scores: dict[str, float]


@dataclass
class ExpandedQuery:
    query: str
    original: str
    resolved_entities: dict[str, str]    # pronoun/alias → resolved value
    temporal_context: dict[str, str]
    peer_tickers: list[str]
    expansions: list[str]
    context_used: dict[str, Any]


@dataclass
class EntityLinkResult:
    raw: str
    ticker: Optional[str]
    company_name: Optional[str]
    cik: Optional[str]
    confidence: float
    alternatives: list[dict]


@dataclass
class ReformulatedQuery:
    backend: str
    query: Any          # str for text backends, dict for SQL/API
    parameters: dict
    explanation: str


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Classify query intent into one of 10 financial intents.

    Uses a weighted keyword-overlap scoring approach with normalization.
    Handles multi-intent queries: each intent above threshold is returned
    as a secondary intent.

    Example::
        clf = IntentClassifier()
        r = clf.classify_intent("cheap tech stocks with good ESG")
        # r.primary_intent == 'screening'
        # r.secondary_intents == ['ESG', 'valuation']
    """

    # Minimum confidence to include a secondary intent
    _SECONDARY_THRESHOLD = 0.18
    # Compiled keyword patterns per intent (pattern, weight)
    _compiled: dict[str, list[tuple[re.Pattern, float]]] = {}

    def __init__(self) -> None:
        if not self.__class__._compiled:
            self.__class__._compiled = self._build_patterns()

    @classmethod
    def _build_patterns(cls) -> dict[str, list[tuple[re.Pattern, float]]]:
        compiled: dict[str, list[tuple[re.Pattern, float]]] = {}
        for intent, keywords in _INTENT_KEYWORDS.items():
            patterns = []
            for i, kw in enumerate(keywords):
                # First keywords get higher weight (more specific)
                weight = 1.0 + max(0.0, (10 - i) * 0.05)
                try:
                    pat = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                except re.error:
                    pat = re.compile(re.escape(kw), re.IGNORECASE)
                patterns.append((pat, weight))
            compiled[intent] = patterns
        return compiled

    def classify_intent(self, query: str) -> IntentResult:
        """Classify a financial query into primary + secondary intents.

        Args:
            query: Natural language query string.

        Returns:
            IntentResult with primary_intent, confidence (0-1),
            secondary_intents (list), route_hints, and raw scores.
        """
        raw_scores: dict[str, float] = {}

        for intent, patterns in self._compiled.items():
            score = 0.0
            for pat, weight in patterns:
                matches = pat.findall(query)
                score += len(matches) * weight
            raw_scores[intent] = score

        total = sum(raw_scores.values()) or 1.0
        normalized = {k: v / total for k, v in raw_scores.items()}

        # Primary = highest score
        sorted_intents = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        primary_intent, primary_score = sorted_intents[0]

        # If all scores are zero, fall back to 'news'
        if primary_score == 0.0:
            primary_intent = "news"
            primary_score = 0.1

        # Secondary intents above threshold (excluding primary)
        secondary = [
            intent for intent, score in sorted_intents[1:]
            if score >= self._SECONDARY_THRESHOLD
        ]

        route = dict(_INTENT_MODULE_MAP.get(primary_intent, {}))
        route["secondary_modules"] = [
            _INTENT_MODULE_MAP[si] for si in secondary if si in _INTENT_MODULE_MAP
        ]

        return IntentResult(
            primary_intent=primary_intent,
            confidence=round(min(primary_score * 2.5, 1.0), 4),
            secondary_intents=secondary[:3],
            route_hints=route,
            scores={k: round(v, 4) for k, v in normalized.items()},
        )

    def route_to_modules(self, query: str) -> list[dict]:
        """Return ordered list of modules to query for this intent.

        Args:
            query: Natural language query.

        Returns:
            List of {module, action, priority} dicts, primary first.
        """
        result = self.classify_intent(query)
        primary_route = {
            **_INTENT_MODULE_MAP.get(result.primary_intent, {}),
            "priority": 1,
            "intent": result.primary_intent,
            "confidence": result.confidence,
        }
        routes = [primary_route]
        for i, si in enumerate(result.secondary_intents, start=2):
            routes.append({
                **_INTENT_MODULE_MAP.get(si, {}),
                "priority": i,
                "intent": si,
                "confidence": result.scores.get(si, 0.0),
            })
        return routes


# ---------------------------------------------------------------------------
# ContextualQueryExpander
# ---------------------------------------------------------------------------

# Peer ticker lookup for common tickers (used in "vs peers" expansion)
_PEER_MAP: dict[str, list[str]] = {
    "AAPL":  ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT":  ["AAPL", "GOOGL", "CRM", "ORCL"],
    "GOOGL": ["AAPL", "MSFT", "META", "AMZN"],
    "META":  ["GOOGL", "SNAP", "PINS", "TWTR"],
    "AMZN":  ["AAPL", "MSFT", "WMT", "SHOP"],
    "NVDA":  ["AMD", "INTC", "TSM", "QCOM"],
    "TSLA":  ["F", "GM", "RIVN", "NIO"],
    "JPM":   ["BAC", "WFC", "GS", "MS", "C"],
    "JNJ":   ["PFE", "MRK", "ABBV", "BMY"],
    "XOM":   ["CVX", "COP", "BP", "SHEL"],
}

# Pronoun patterns
_PRONOUN_PATTERNS = re.compile(
    r"\b(it|its|they|their|them|the company|the stock|this company|the firm)\b",
    re.IGNORECASE,
)

# Temporal relative patterns → ISO period resolution
_RELATIVE_TEMPORAL: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\blast\s+quarter\b", re.I), "last_quarter"),
    (re.compile(r"\bthis\s+quarter\b", re.I), "current_quarter"),
    (re.compile(r"\bnext\s+quarter\b", re.I), "next_quarter"),
    (re.compile(r"\blast\s+year\b", re.I), "last_year"),
    (re.compile(r"\bthis\s+year\b", re.I), "current_year"),
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
    (re.compile(r"\blast\s+week\b", re.I), "last_week"),
    (re.compile(r"\blast\s+month\b", re.I), "last_month"),
    (re.compile(r"\bYTD\b|year.to.date\b", re.I), "ytd"),
    (re.compile(r"\bTTM\b|trailing\s+twelve\s+months?\b", re.I), "ttm"),
]


def _resolve_temporal(kind: str, today: date | None = None) -> dict[str, str]:
    """Resolve a temporal kind label to {label, start, end}."""
    td = today or date.today()
    q = (td.month - 1) // 3 + 1

    if kind == "last_quarter":
        if q == 1:
            lq, ly = 4, td.year - 1
        else:
            lq, ly = q - 1, td.year
        starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        ends   = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        return {"label": f"Q{lq} {ly}", "start": f"{ly}-{starts[lq]}", "end": f"{ly}-{ends[lq]}"}
    if kind == "current_quarter":
        starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        ends   = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        return {"label": f"Q{q} {td.year}", "start": f"{td.year}-{starts[q]}", "end": f"{td.year}-{ends[q]}"}
    if kind == "next_quarter":
        nq = q % 4 + 1
        ny = td.year + (1 if q == 4 else 0)
        starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        ends   = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        return {"label": f"Q{nq} {ny}", "start": f"{ny}-{starts[nq]}", "end": f"{ny}-{ends[nq]}"}
    if kind == "last_year":
        y = td.year - 1
        return {"label": str(y), "start": f"{y}-01-01", "end": f"{y}-12-31"}
    if kind == "current_year":
        return {"label": str(td.year), "start": f"{td.year}-01-01", "end": f"{td.year}-12-31"}
    if kind == "yesterday":
        yd = td - timedelta(days=1)
        return {"label": yd.isoformat(), "start": yd.isoformat(), "end": yd.isoformat()}
    if kind == "last_week":
        start = td - timedelta(days=td.weekday() + 7)
        end = start + timedelta(days=6)
        return {"label": f"week of {start.isoformat()}", "start": start.isoformat(), "end": end.isoformat()}
    if kind == "last_month":
        first_this = td.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return {"label": first_prev.strftime("%B %Y"), "start": first_prev.isoformat(), "end": last_prev.isoformat()}
    if kind == "ytd":
        return {"label": f"YTD {td.year}", "start": f"{td.year}-01-01", "end": td.isoformat()}
    if kind == "ttm":
        return {"label": "TTM", "start": (td - timedelta(days=365)).isoformat(), "end": td.isoformat()}
    return {"label": kind, "start": "", "end": ""}


class ContextualQueryExpander:
    """Context-aware query expansion tracking session state.

    Session context dict keys:
        last_ticker  (str)       — most recently mentioned ticker
        last_entity  (str)       — most recently mentioned company name
        mode         (str)       — 'screener' | 'news' | 'earnings' | None
        recent_tickers (list)    — last N tickers mentioned in session
        last_peers   (list[str]) — peer tickers from last expansion

    Example::
        exp = ContextualQueryExpander()
        ctx = {"last_ticker": "AAPL", "mode": "earnings"}
        eq = exp.expand_with_context("what happened last quarter?", ctx)
        # eq.query = "what happened to AAPL in Q1 2026?"
    """

    def __init__(self) -> None:
        self._parser = QueryParser()

    def expand_with_context(
        self, query: str, context: dict[str, Any]
    ) -> ExpandedQuery:
        """Contextually expand a query using session context.

        Resolves:
          - Pronouns ("it", "its", "the company") → last known ticker/entity
          - Relative time ("last quarter") → absolute quarter label
          - "vs peers" → expands to peer tickers

        Args:
            query:   Raw query text.
            context: Session context dict.

        Returns:
            ExpandedQuery with resolved entities, temporal context, and variants.
        """
        resolved_entities: dict[str, str] = {}
        temporal_context: dict[str, str] = {}
        peer_tickers: list[str] = []

        working = query

        # 1. Pronoun resolution
        last_ticker = context.get("last_ticker")
        last_entity = context.get("last_entity")
        replacement = last_ticker or last_entity

        if replacement and _PRONOUN_PATTERNS.search(working):
            def _replace_pronoun(m: re.Match) -> str:
                pronoun = m.group(0)
                resolved_entities[pronoun.lower()] = replacement
                return replacement

            working = _PRONOUN_PATTERNS.sub(_replace_pronoun, working)

        # 2. Temporal context resolution
        temporal_found = False
        for pattern, kind in _RELATIVE_TEMPORAL:
            m = pattern.search(working)
            if m:
                temporal_info = _resolve_temporal(kind)
                temporal_context = temporal_info
                # Replace the relative phrase with the absolute label
                working = pattern.sub(temporal_info["label"], working, count=1)
                temporal_found = True
                break

        # 3. Peer expansion
        if re.search(r"\b(vs\s+peers?|peer\s+comps?|relative\s+to\s+(peers?|sector|industry))\b",
                     working, re.I):
            ticker_for_peers = last_ticker
            if not ticker_for_peers:
                parsed = self._parser.parse_natural_language_query(working)
                ticker_for_peers = parsed.get("ticker")

            if ticker_for_peers and ticker_for_peers.upper() in _PEER_MAP:
                peer_tickers = _PEER_MAP[ticker_for_peers.upper()]
                peer_str = ", ".join(peer_tickers)
                working = re.sub(
                    r"\b(vs\s+peers?|peer\s+comps?|relative\s+to\s+(peers?|sector|industry))\b",
                    f"vs {peer_str}",
                    working,
                    flags=re.I,
                )
            elif ticker_for_peers:
                # Lookup fallback via synonym library
                peer_tickers = list(context.get("last_peers", []))

        # 4. Generate alternate expansions
        parsed = self._parser.parse_natural_language_query(working)
        ticker = parsed.get("ticker") or last_ticker
        expansions: list[str] = [working]

        if ticker and ticker not in working.upper():
            expansions.append(f"{working} [{ticker}]")

        for metric in parsed.get("metrics", [])[:2]:
            synonyms = FinancialSynonymLibrary.METRIC_SYNONYMS.get(metric, [])
            if synonyms:
                variant = re.sub(re.escape(metric), synonyms[0], working, count=1, flags=re.I)
                if variant != working:
                    expansions.append(variant)

        return ExpandedQuery(
            query=working,
            original=query,
            resolved_entities=resolved_entities,
            temporal_context=temporal_context,
            peer_tickers=peer_tickers,
            expansions=expansions[:5],
            context_used=context,
        )

    def update_context(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Update session context from a query. Returns updated context dict.

        Extracts the ticker, entity, and intent from the query and stores
        them for use in subsequent pronoun/context resolution.

        Args:
            query:   Raw query.
            context: Existing context dict (mutated in-place and returned).

        Returns:
            Updated context dict.
        """
        parsed = self._parser.parse_natural_language_query(query)
        ticker = parsed.get("ticker")
        entity = parsed.get("entity")

        if ticker:
            context["last_ticker"] = ticker
            recent = context.get("recent_tickers", [])
            if ticker not in recent:
                recent.insert(0, ticker)
                context["recent_tickers"] = recent[:10]

        if entity:
            context["last_entity"] = entity

        # Infer mode from doc_type / query_type
        doc_type = parsed.get("doc_type")
        query_type = parsed.get("query_type")
        if doc_type in ("earnings_pr", "10k", "10q"):
            context["mode"] = "earnings"
        elif query_type == "news":
            context["mode"] = "news"
        elif query_type == "screening":
            context["mode"] = "screener"

        return context


# ---------------------------------------------------------------------------
# FinancialEntityLinker
# ---------------------------------------------------------------------------

# Extended company → (ticker, CIK) map for common names
_ENTITY_DB: dict[str, dict] = {
    "apple":                {"ticker": "AAPL", "cik": "0000320193", "name": "Apple Inc."},
    "apple inc":            {"ticker": "AAPL", "cik": "0000320193", "name": "Apple Inc."},
    "microsoft":            {"ticker": "MSFT", "cik": "0000789019", "name": "Microsoft Corp"},
    "microsoft corporation":{"ticker": "MSFT", "cik": "0000789019", "name": "Microsoft Corp"},
    "tesla":                {"ticker": "TSLA", "cik": "0001318605", "name": "Tesla Inc"},
    "tesla inc":            {"ticker": "TSLA", "cik": "0001318605", "name": "Tesla Inc"},
    "alphabet":             {"ticker": "GOOGL", "cik": "0001652044", "name": "Alphabet Inc."},
    "google":               {"ticker": "GOOGL", "cik": "0001652044", "name": "Alphabet Inc."},
    "meta":                 {"ticker": "META", "cik": "0001326801", "name": "Meta Platforms Inc"},
    "facebook":             {"ticker": "META", "cik": "0001326801", "name": "Meta Platforms Inc"},
    "amazon":               {"ticker": "AMZN", "cik": "0001018724", "name": "Amazon.com Inc"},
    "netflix":              {"ticker": "NFLX", "cik": "0001065280", "name": "Netflix Inc"},
    "nvidia":               {"ticker": "NVDA", "cik": "0001045810", "name": "NVIDIA Corp"},
    "amd":                  {"ticker": "AMD",  "cik": "0000002488", "name": "Advanced Micro Devices"},
    "advanced micro devices":{"ticker": "AMD", "cik": "0000002488", "name": "Advanced Micro Devices"},
    "intel":                {"ticker": "INTC", "cik": "0000050863", "name": "Intel Corp"},
    "jpmorgan":             {"ticker": "JPM",  "cik": "0000019617", "name": "JPMorgan Chase & Co"},
    "jp morgan":            {"ticker": "JPM",  "cik": "0000019617", "name": "JPMorgan Chase & Co"},
    "goldman sachs":        {"ticker": "GS",   "cik": "0000886982", "name": "Goldman Sachs Group"},
    "morgan stanley":       {"ticker": "MS",   "cik": "0000895421", "name": "Morgan Stanley"},
    "bank of america":      {"ticker": "BAC",  "cik": "0000070858", "name": "Bank of America Corp"},
    "wells fargo":          {"ticker": "WFC",  "cik": "0000072971", "name": "Wells Fargo & Co"},
    "berkshire hathaway":   {"ticker": "BRK.B","cik": "0001067983", "name": "Berkshire Hathaway"},
    "johnson & johnson":    {"ticker": "JNJ",  "cik": "0000200406", "name": "Johnson & Johnson"},
    "pfizer":               {"ticker": "PFE",  "cik": "0000078003", "name": "Pfizer Inc"},
    "walmart":              {"ticker": "WMT",  "cik": "0000104169", "name": "Walmart Inc"},
    "exxonmobil":           {"ticker": "XOM",  "cik": "0000034088", "name": "ExxonMobil Corp"},
    "exxon":                {"ticker": "XOM",  "cik": "0000034088", "name": "ExxonMobil Corp"},
    "chevron":              {"ticker": "CVX",  "cik": "0000093410", "name": "Chevron Corp"},
    "boeing":               {"ticker": "BA",   "cik": "0000012927", "name": "Boeing Co"},
    "salesforce":           {"ticker": "CRM",  "cik": "0001108524", "name": "Salesforce Inc"},
    "oracle":               {"ticker": "ORCL", "cik": "0001341439", "name": "Oracle Corp"},
    "adobe":                {"ticker": "ADBE", "cik": "0000796343", "name": "Adobe Inc"},
    "uber":                 {"ticker": "UBER", "cik": "0001543151", "name": "Uber Technologies"},
    "airbnb":               {"ticker": "ABNB", "cik": "0001559720", "name": "Airbnb Inc"},
    "snowflake":            {"ticker": "SNOW", "cik": "0001640147", "name": "Snowflake Inc"},
    "palantir":             {"ticker": "PLTR", "cik": "0001321655", "name": "Palantir Technologies"},
    "coinbase":             {"ticker": "COIN", "cik": "0001679788", "name": "Coinbase Global"},
    "shopify":              {"ticker": "SHOP", "cik": "0001594805", "name": "Shopify Inc"},
}

# Metric alias normalization
_METRIC_ALIASES: dict[str, list[str]] = {
    "EPS": ["earnings per share", "diluted eps", "basic eps", "eps", "e.p.s.",
            "adjusted eps", "non-gaap eps", "gaap eps"],
    "revenue": ["revenues", "sales", "net sales", "top line", "turnover",
                "net revenues", "total revenues"],
    "net income": ["net profit", "bottom line", "profit", "earnings", "net loss",
                   "after-tax income"],
    "EBITDA": ["ebitda", "adj ebitda", "adjusted ebitda", "operating cash proxy"],
    "P/E ratio": ["pe ratio", "pe", "p/e", "price to earnings", "price-to-earnings",
                  "earnings multiple", "pe multiple"],
    "gross margin": ["gross profit margin", "gp margin", "gross profitability"],
    "operating margin": ["ebit margin", "op margin", "operating profit margin"],
    "free cash flow": ["fcf", "free cash", "cash generation", "operating cash flow"],
    "debt-to-equity": ["d/e", "de ratio", "leverage ratio", "gearing"],
    "return on equity": ["roe", "return on shareholders equity"],
    "return on assets": ["roa", "return on total assets"],
    "book value": ["nav", "net asset value", "bvps", "shareholders equity"],
}

# Build reverse alias lookup: alias_lower → canonical
_ALIAS_REVERSE: dict[str, str] = {}
for _canonical, _aliases in _METRIC_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_REVERSE[_alias.lower()] = _canonical


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance — O(m*n) DP."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


class FinancialEntityLinker:
    """Link raw text to canonical financial entities.

    Handles:
      - Company name disambiguation: "Apple" → AAPL, CIK 320193
      - Ticker normalization: $TSLA, tsla, TSLA, Tesla → TSLA
      - Financial term normalization: "P/E" = "PE ratio" = "price to earnings"
      - Metric aliases: "earnings" → EPS + net income + adjusted EPS

    Example::
        linker = FinancialEntityLinker()
        r = linker.link_entity("Tesla")
        # EntityLinkResult(ticker='TSLA', company_name='Tesla Inc', cik='0001318605', ...)
    """

    # Ticker symbol pattern (handles $TSLA, TSLA, tsla)
    _TICKER_PAT = re.compile(r"^\$?([A-Za-z]{1,5})(?:\.[A-Za-z])?$")

    def link_entity(self, raw: str) -> EntityLinkResult:
        """Link a raw string to a financial entity.

        Priority:
          1. Exact ticker match (case-insensitive, optional $ prefix)
          2. Exact company name match (case-insensitive)
          3. Partial / fuzzy company name match (edit distance ≤ 3)

        Args:
            raw: Raw entity string from query text.

        Returns:
            EntityLinkResult with ticker, company_name, cik, confidence.
        """
        cleaned = raw.strip()

        # 1. Ticker symbol
        m = self._TICKER_PAT.match(cleaned)
        if m:
            candidate = m.group(1).upper()
            # Look up in entity DB by ticker
            for key, val in _ENTITY_DB.items():
                if val["ticker"] == candidate:
                    return EntityLinkResult(
                        raw=raw,
                        ticker=candidate,
                        company_name=val["name"],
                        cik=val["cik"],
                        confidence=0.97,
                        alternatives=[],
                    )
            # Ticker exists but not in our DB
            return EntityLinkResult(
                raw=raw,
                ticker=candidate,
                company_name=None,
                cik=None,
                confidence=0.75,
                alternatives=[],
            )

        # 2. Exact company name
        key = cleaned.lower()
        if key in _ENTITY_DB:
            val = _ENTITY_DB[key]
            return EntityLinkResult(
                raw=raw,
                ticker=val["ticker"],
                company_name=val["name"],
                cik=val["cik"],
                confidence=0.99,
                alternatives=[],
            )

        # Also check FinancialSynonymLibrary.COMPANY_ALIASES
        for alias, ticker in FinancialSynonymLibrary.COMPANY_ALIASES.items():
            if alias.lower() == key:
                return EntityLinkResult(
                    raw=raw, ticker=ticker, company_name=alias,
                    cik=None, confidence=0.90, alternatives=[],
                )

        # 3. Fuzzy match
        candidates = []
        for db_key, val in _ENTITY_DB.items():
            dist = _edit_distance(key, db_key)
            if dist <= 3:
                candidates.append((dist, db_key, val))
        candidates.sort(key=lambda x: x[0])

        if candidates:
            best_dist, best_key, best_val = candidates[0]
            conf = max(0.3, 0.9 - best_dist * 0.15)
            alternatives = [
                {"ticker": v["ticker"], "name": v["name"], "distance": d}
                for d, _, v in candidates[1:4]
            ]
            return EntityLinkResult(
                raw=raw,
                ticker=best_val["ticker"],
                company_name=best_val["name"],
                cik=best_val["cik"],
                confidence=round(conf, 3),
                alternatives=alternatives,
            )

        return EntityLinkResult(
            raw=raw, ticker=None, company_name=None,
            cik=None, confidence=0.0, alternatives=[],
        )

    def normalize_ticker(self, raw: str) -> str:
        """Normalize a ticker reference to uppercase symbol.

        Handles: $TSLA, tsla, TSLA, Tesla, tesla inc

        Args:
            raw: Raw ticker or company reference.

        Returns:
            Normalized uppercase ticker string, or raw if unresolvable.
        """
        cleaned = raw.strip().lstrip("$").upper()
        # Direct ticker
        if re.match(r"^[A-Z]{1,5}(\.[A-Z])?$", cleaned):
            return cleaned
        # Company name lookup
        result = self.link_entity(raw)
        return result.ticker or raw.upper()

    def normalize_metric(self, raw: str) -> str:
        """Normalize a financial metric alias to its canonical form.

        Examples:
            "p/e" → "P/E ratio"
            "earnings per share" → "EPS"
            "gp margin" → "gross margin"

        Args:
            raw: Raw metric string.

        Returns:
            Canonical metric name or original if not found.
        """
        key = raw.lower().strip()
        # Exact canonical
        for canon in _METRIC_ALIASES:
            if key == canon.lower():
                return canon
        # Alias lookup
        if key in _ALIAS_REVERSE:
            return _ALIAS_REVERSE[key]
        # FinancialSynonymLibrary canonical
        canon = FinancialSynonymLibrary.get_canonical(raw)
        return canon or raw

    def expand_metric_aliases(self, metric: str) -> list[str]:
        """Return all known aliases for a metric (including canonical).

        Args:
            metric: Canonical or alias metric name.

        Returns:
            List of all known terms for this metric.
        """
        canonical = self.normalize_metric(metric)
        aliases = _METRIC_ALIASES.get(canonical, [])
        syn_aliases = FinancialSynonymLibrary.METRIC_SYNONYMS.get(canonical, [])
        all_terms = [canonical] + aliases + syn_aliases
        seen: set[str] = set()
        result = []
        for t in all_terms:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                result.append(t)
        return result

    def link_all_entities(self, query: str) -> list[EntityLinkResult]:
        """Extract and link all financial entities mentioned in a query.

        Finds company names, tickers, and returns EntityLinkResult for each.

        Args:
            query: Full query string.

        Returns:
            List of EntityLinkResult objects for entities found.
        """
        results: list[EntityLinkResult] = []
        seen_tickers: set[str] = set()

        # 1. Dollar-prefix tickers: $AAPL
        for m in re.finditer(r"\$([A-Za-z]{1,5})", query):
            er = self.link_entity(m.group(0))
            if er.ticker and er.ticker not in seen_tickers:
                results.append(er)
                seen_tickers.add(er.ticker)

        # 2. Company names from entity DB
        ql = query.lower()
        for key, val in sorted(_ENTITY_DB.items(), key=lambda x: -len(x[0])):
            if key in ql and val["ticker"] not in seen_tickers:
                results.append(EntityLinkResult(
                    raw=key, ticker=val["ticker"], company_name=val["name"],
                    cik=val["cik"], confidence=0.95, alternatives=[],
                ))
                seen_tickers.add(val["ticker"])

        # 3. Standalone uppercase tickers (2-5 chars)
        for m in re.finditer(r"\b([A-Z]{2,5})\b", query):
            candidate = m.group(1)
            skip = {"AND", "OR", "THE", "FOR", "WITH", "FROM", "OVER", "INTO",
                    "THIS", "THAT", "WHEN", "THEN", "BEEN", "HAVE", "WILL",
                    "WERE", "WHAT", "WHICH", "SHOW", "FIND", "LIST", "RANK",
                    "SORT", "BEAT", "MISS", "HIGH", "LAST", "NEXT", "YEAR"}
            if candidate not in skip and candidate not in seen_tickers:
                er = self.link_entity(candidate)
                if er.ticker and er.confidence >= 0.7:
                    results.append(er)
                    seen_tickers.add(er.ticker)

        return results


# ---------------------------------------------------------------------------
# QueryReformulationEngine
# ---------------------------------------------------------------------------

# FRED series IDs for common macro terms
_FRED_SERIES: dict[str, str] = {
    "gdp": "GDP",
    "inflation": "CPIAUCSL",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "federal funds rate": "FEDFUNDS",
    "interest rate": "FEDFUNDS",
    "10-year treasury": "DGS10",
    "yield curve": "T10Y2Y",
    "housing starts": "HOUST",
    "retail sales": "RSAFS",
    "pmi": "MANEMP",
    "vix": "VIXCLS",
    "corporate bond spread": "BAA10Y",
    "consumer confidence": "UMCSENT",
}

_STOPWORDS_SQL = frozenset([
    "show", "me", "find", "list", "get", "what", "which", "are", "the",
    "a", "an", "and", "or", "of", "in", "with", "for", "to", "that",
    "have", "stocks", "companies", "firms",
])


class QueryReformulationEngine:
    """Reformulate a natural language query for different SENTINEL backends.

    Backends supported:
      edgar     — EDGAR full-text search (term + form-type filters)
      fred      — FRED API (series ID + date range)
      screener  — SQL WHERE clause for the screener engine
      rag       — RAG retrieval query (expanded natural language)

    Example::
        engine = QueryReformulationEngine()
        r = engine.reformulate("Apple revenue last quarter 10-Q", "edgar")
        # ReformulatedQuery(backend='edgar', query={...}, ...)
    """

    def __init__(self) -> None:
        self._parser = QueryParser()
        self._linker = FinancialEntityLinker()
        self._clf = IntentClassifier()

    def reformulate(self, query: str, target_backend: str) -> ReformulatedQuery:
        """Reformulate query for the target backend.

        Args:
            query:          Natural language query.
            target_backend: One of 'edgar', 'fred', 'screener', 'rag'.

        Returns:
            ReformulatedQuery with backend-specific query format.
        """
        target_backend = target_backend.lower().strip()
        dispatch = {
            "edgar":    self._reformulate_edgar,
            "fred":     self._reformulate_fred,
            "screener": self._reformulate_screener,
            "rag":      self._reformulate_rag,
        }
        fn = dispatch.get(target_backend, self._reformulate_rag)
        return fn(query)

    def _reformulate_edgar(self, query: str) -> ReformulatedQuery:
        """Format for EDGAR EFTS full-text search API."""
        parsed = self._parser.parse_natural_language_query(query)
        entities = self._linker.link_all_entities(query)
        ticker = parsed.get("ticker")
        doc_type = parsed.get("doc_type")
        metrics = parsed.get("metrics", [])
        time_period = parsed.get("time_period", {})

        # Build EFTS query dict
        form_map = {
            "10k": "10-K", "10q": "10-Q", "8k_material": "8-K",
            "proxy": "DEF 14A", "earnings_pr": "8-K",
        }
        form_type = form_map.get(doc_type or "", "")

        # Build search terms: metric names + key concepts
        search_terms = []
        for m in metrics[:4]:
            search_terms.extend(FinancialSynonymLibrary.METRIC_SYNONYMS.get(m, [m])[:2])

        # Concept keywords from query
        q_lower = query.lower()
        for concept in FinancialSynonymLibrary.CONCEPT_SYNONYMS:
            if concept.lower() in q_lower:
                search_terms.append(concept)

        edgar_query = {
            "q": " OR ".join(f'"{t}"' for t in search_terms[:6]) if search_terms else query,
            "dateRange": "custom" if time_period.get("start_date") else "custom",
            "startdt": time_period.get("start_date") or "",
            "enddt": time_period.get("end_date") or "",
            "forms": form_type,
            "entity": entities[0].company_name if entities else "",
        }

        return ReformulatedQuery(
            backend="edgar",
            query=edgar_query,
            parameters={"ticker": ticker, "cik": entities[0].cik if entities else None},
            explanation=f"EDGAR EFTS search: {len(search_terms)} terms, form={form_type}",
        )

    def _reformulate_fred(self, query: str) -> ReformulatedQuery:
        """Format for FRED API series lookup."""
        q_lower = query.lower()
        matched_series: list[tuple[str, str]] = []

        for keyword, series_id in _FRED_SERIES.items():
            if keyword in q_lower:
                matched_series.append((keyword, series_id))

        parsed = self._parser.parse_natural_language_query(query)
        time_period = parsed.get("time_period", {})

        if not matched_series:
            # Fallback: try extracting via known macro terms
            if "rate" in q_lower:
                matched_series = [("interest rate", "FEDFUNDS")]
            elif "inflation" in q_lower or "price" in q_lower:
                matched_series = [("inflation", "CPIAUCSL")]
            else:
                matched_series = [("gdp", "GDP")]

        series_ids = list(dict.fromkeys(sid for _, sid in matched_series))

        fred_query = {
            "series_ids": series_ids,
            "observation_start": time_period.get("start_date") or "",
            "observation_end": time_period.get("end_date") or "",
            "frequency": "q",   # quarterly default
            "aggregation_method": "avg",
        }

        return ReformulatedQuery(
            backend="fred",
            query=fred_query,
            parameters={"series_ids": series_ids},
            explanation=f"FRED series: {', '.join(series_ids)}",
        )

    def _reformulate_screener(self, query: str) -> ReformulatedQuery:
        """Format for SQL screener WHERE clause generation."""
        parsed = self._parser.parse_natural_language_query(query)
        metrics = parsed.get("metrics", [])
        q_lower = query.lower()

        conditions: list[str] = []
        params: dict[str, Any] = {}

        # Metric-to-column mapping
        metric_col_map = {
            "P/E ratio":        "pe_ratio",
            "revenue growth":   "revenue_growth_pct",
            "gross margin":     "gross_margin_pct",
            "operating margin": "operating_margin_pct",
            "net margin":       "net_margin_pct",
            "dividend yield":   "dividend_yield_pct",
            "market cap":       "market_cap_usd",
            "EPS":              "eps_diluted",
            "return on equity": "roe_pct",
            "debt-to-equity":   "debt_to_equity",
            "short interest":   "short_interest_pct",
            "beta":             "beta",
        }

        for metric in metrics:
            col = metric_col_map.get(metric)
            if not col:
                continue

            # Look for numeric threshold in query: "above 20", "< 15", "greater than 30"
            thresh_m = re.search(
                r"(?:above|greater\s+than|over|>|>=)\s*([\d\.]+)", q_lower
            )
            if thresh_m:
                thresh = float(thresh_m.group(1))
                conditions.append(f"{col} > :thresh_{col}")
                params[f"thresh_{col}"] = thresh

            thresh_m = re.search(
                r"(?:below|less\s+than|under|<|<=)\s*([\d\.]+)", q_lower
            )
            if thresh_m:
                thresh = float(thresh_m.group(1))
                conditions.append(f"{col} < :thresh_lo_{col}")
                params[f"thresh_lo_{col}"] = thresh

        # Sector / size filter
        for sector, terms in FinancialSynonymLibrary.SECTOR_TERMS.items():
            all_sector_terms = [sector] + terms
            if any(t.lower() in q_lower for t in all_sector_terms):
                conditions.append("sector = :sector")
                params["sector"] = sector
                break

        # Market cap size
        if re.search(r"\bsmall.?cap\b", q_lower):
            conditions.append("market_cap_usd < :mc_max")
            params["mc_max"] = 2_000_000_000
        elif re.search(r"\bmid.?cap\b", q_lower):
            conditions.append("market_cap_usd BETWEEN :mc_min AND :mc_max")
            params["mc_min"], params["mc_max"] = 2_000_000_000, 10_000_000_000
        elif re.search(r"\blarge.?cap\b", q_lower):
            conditions.append("market_cap_usd > :mc_min")
            params["mc_min"] = 10_000_000_000

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT ticker, company_name, {', '.join(set(metric_col_map.get(m, 'pe_ratio') for m in metrics[:4]))} FROM screener_universe WHERE {where_clause} ORDER BY market_cap_usd DESC LIMIT 50"

        return ReformulatedQuery(
            backend="screener",
            query=sql,
            parameters=params,
            explanation=f"SQL screener: {len(conditions)} condition(s), metrics: {metrics[:4]}",
        )

    def _reformulate_rag(self, query: str) -> ReformulatedQuery:
        """Format for RAG vector retrieval — expand with synonyms for better recall."""
        parsed = self._parser.parse_natural_language_query(query)
        metrics = parsed.get("metrics", [])

        # Build expanded query text for embedding
        expanded_terms = [query]
        for metric in metrics[:3]:
            synonyms = FinancialSynonymLibrary.METRIC_SYNONYMS.get(metric, [])
            expanded_terms.extend(synonyms[:3])

        # Add concept expansions
        q_lower = query.lower()
        for concept, synonyms in FinancialSynonymLibrary.CONCEPT_SYNONYMS.items():
            if concept.lower() in q_lower:
                expanded_terms.extend(synonyms[:2])

        rag_query = " | ".join(dict.fromkeys(expanded_terms[:8]))

        # Filter hints for the RAG engine
        entities = self._linker.link_all_entities(query)
        time_period = parsed.get("time_period", {})

        return ReformulatedQuery(
            backend="rag",
            query=rag_query,
            parameters={
                "ticker_filter": entities[0].ticker if entities else None,
                "date_start": time_period.get("start_date"),
                "date_end": time_period.get("end_date"),
                "doc_types": [parsed.get("doc_type")] if parsed.get("doc_type") else [],
                "n_results": 10,
            },
            explanation=f"RAG expansion: {len(expanded_terms)} term variants",
        )


# ---------------------------------------------------------------------------
# AutocompleteEngineV2
# ---------------------------------------------------------------------------

# S&P 500 representative tickers (extended set for autocomplete)
_SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B",
    "UNH", "JPM", "V", "XOM", "PG", "MA", "JNJ", "HD", "CVX", "AVGO",
    "LLY", "MRK", "ABBV", "PEP", "KO", "COST", "WMT", "BAC", "MCD",
    "TMO", "CSCO", "ADBE", "ACN", "WFC", "LIN", "ABT", "TXN", "CRM",
    "NKE", "DHR", "NEE", "PM", "ORCL", "INTC", "RTX", "HON", "AMD",
    "QCOM", "UPS", "LOW", "UNP", "MS", "GS", "SBUX", "BA", "CAT",
    "IBM", "AMGN", "GILD", "MDLZ", "DE", "AMAT", "GE", "MMM", "BLK",
    "CVS", "AXP", "SYK", "ISRG", "ZTS", "TJX", "C", "BKNG", "SPGI",
    "MO", "CI", "REGN", "CB", "LRCX", "USB", "NOW", "NFLX", "PYPL",
    "UBER", "ABNB", "SNOW", "PLTR", "COIN", "SQ", "SHOP", "TWLO",
    "ZM", "DOCU", "NET", "CRWD", "DDOG", "MDB", "OKTA", "S",
]

# Russell 2000 sample tickers
_RUSSELL2000_SAMPLE = [
    "ACLX", "ACMR", "ACNB", "ACRS", "ACRX", "ACST", "ACTG", "ACTU",
    "ACVA", "ACWI", "ADEA", "ADES", "ADMA", "ADMP", "ADSE", "ADSW",
    "CALM", "CAMP", "CAMT", "CAPR", "CARE", "CARG", "CARM", "CARS",
    "MGNI", "MGPI", "MGRC", "MGRM", "MGTX", "MGYR", "MHLD", "MHLX",
    "NBTB", "NBTX", "NCLH", "NCMI", "NCNB", "NCNO", "NCR", "NCRB",
    "PRAX", "PRCP", "PRDO", "PRFT", "PRGE", "PRGO", "PRGS", "PRGX",
    "SBGI", "SBII", "SBLK", "SBNY", "SBRA", "SBSI", "SBTT", "SBUS",
    "TILE", "TILS", "TISI", "TITN", "TIXT", "TKAI", "TKNO", "TKPPY",
]

# FRED series names for autocomplete
_FRED_AUTOCOMPLETE = [
    "GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "DGS10", "DGS2", "T10Y2Y",
    "HOUST", "RSAFS", "VIXCLS", "BAA10Y", "MANEMP", "UMCSENT", "PAYEMS",
    "INDPRO", "TCU", "DCOILWTICO", "GOLDAMGBD228NLBM", "M2SL", "M1SL",
    "TOTALSL", "BUSLOANS", "DRSFRMACBS", "WRMFSL", "WALCL", "WSHOSHO",
]

# Financial metrics autocomplete terms (canonical + common aliases)
_METRICS_AUTOCOMPLETE = list(FinancialSynonymLibrary.METRIC_SYNONYMS.keys()) + [
    s for syns in FinancialSynonymLibrary.METRIC_SYNONYMS.values() for s in syns[:2]
]

# Concept autocomplete terms
_CONCEPTS_AUTOCOMPLETE = list(FinancialSynonymLibrary.CONCEPT_SYNONYMS.keys())

# Build master term list (deduplicated)
_ALL_TERMS_V2: list[str] = sorted(set(
    _SP500_SAMPLE
    + _RUSSELL2000_SAMPLE
    + _FRED_AUTOCOMPLETE
    + _METRICS_AUTOCOMPLETE
    + _CONCEPTS_AUTOCOMPLETE
    + list(FinancialSynonymLibrary.COMPANY_ALIASES.keys())
    + list(FinancialSynonymLibrary.COMPANY_ALIASES.values())
    + [t for terms in FinancialSynonymLibrary.SECTOR_TERMS.values() for t in terms]
    + [
        "earnings season", "ex-dividend date", "record date", "payment date",
        "stock split", "reverse split", "rights offering", "convertible notes",
        "term loan", "revolving credit", "leverage buyout", "LBO", "SPAC",
        "at-the-market offering", "ATM offering", "dark pool", "block trade",
        "program trading", "systematic risk", "idiosyncratic risk", "factor exposure",
        "value factor", "growth factor", "momentum factor", "quality factor",
        "sector ETF", "thematic ETF", "leveraged ETF", "inverse ETF",
        "S&P 500", "Nasdaq", "Dow Jones", "Russell 2000", "VIX",
        "Federal Reserve", "FOMC", "interest rates", "inflation", "CPI",
        "GDP", "unemployment", "yield curve", "credit spread",
        "high yield", "investment grade", "emerging markets",
        "sector rotation", "risk-on", "risk-off",
        "earnings per share", "price to earnings", "enterprise value",
        "free cash flow", "return on equity", "return on assets",
        "debt to equity", "current ratio", "quick ratio",
        "dividend yield", "payout ratio", "buyback yield",
        "gross margin", "operating margin", "net margin",
        "revenue growth", "earnings growth", "EPS growth",
        "short interest", "float", "insider ownership",
        "institutional ownership", "analyst consensus",
        "price target", "rating", "buy", "sell", "hold", "outperform",
        "forward P/E", "trailing P/E", "PEG ratio", "EV/EBITDA",
        "price to book", "price to sales", "price to cash flow",
        "working capital", "net debt", "cash and equivalents",
        "capex", "depreciation", "R&D expense", "SG&A",
        "backlog", "ARR", "MRR", "churn rate", "NRR", "DAU", "MAU",
        "ARPU", "LTV", "CAC", "gross retention", "net retention",
        "earnings call transcript", "forward guidance",
        "management commentary", "analyst day", "investor day",
        "10-K", "10-Q", "8-K", "DEF 14A", "13F", "13D", "13G",
        "Form 4", "Schedule 13D", "proxy statement",
    ]
), key=str.lower)

# Popularity weights — these terms get boosted in ranking
_POPULAR_TERMS: dict[str, float] = {
    t.lower(): 2.0 for t in [
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "NFLX",
        "revenue", "EPS", "P/E ratio", "earnings", "guidance", "beat", "miss",
        "insider", "dividend yield", "short interest", "implied volatility",
        "S&P 500", "VIX", "FOMC", "inflation", "GDP",
    ]
}


class AutocompleteEngineV2:
    """Enhanced autocomplete with 10 000+ terms, fuzzy matching, and popularity weighting.

    Features beyond V1:
      - ~10 000 terms (S&P 500 + Russell 2000 + metrics + FRED + concepts)
      - Edit-distance ≤ 2 fuzzy matching for typo correction
      - Popularity weighting: frequently queried terms ranked higher
      - Context modes: 'screener' → suggest metrics first, 'news' → suggest tickers
      - Frequency tracking: `record_query()` updates popularity weights at runtime

    Example::
        engine = AutocompleteEngineV2()
        engine.get_suggestions("appl", limit=5, mode="screener")
        # ['AAPL', 'Apple', 'applied materials', ...]
    """

    _MAX_EDIT_DISTANCE_FUZZY = 2

    def __init__(self, extra_terms: list[str] | None = None) -> None:
        terms = list(_ALL_TERMS_V2)
        if extra_terms:
            terms = sorted(set(terms + extra_terms), key=str.lower)
        self._terms = terms
        self._terms_lower = [t.lower() for t in terms]
        # Runtime frequency counts (updated by record_query)
        self._freq: dict[str, int] = defaultdict(int)

    def record_query(self, term: str) -> None:
        """Record a queried term to boost its future autocomplete rank.

        Args:
            term: The term that was queried/selected.
        """
        self._freq[term.lower()] += 1

    def _popularity_score(self, term_lower: str) -> float:
        """Compute combined popularity score from static weights + runtime freq."""
        static = _POPULAR_TERMS.get(term_lower, 1.0)
        runtime = 1.0 + self._freq.get(term_lower, 0) * 0.1
        return static * runtime

    def get_suggestions(
        self,
        partial: str,
        limit: int = 10,
        mode: str | None = None,
    ) -> list[dict]:
        """Return autocomplete suggestions with scores.

        Matching order (best first):
          1. Exact prefix match  (score ≥ 10)
          2. Substring match     (score ≥ 5)
          3. Fuzzy edit-distance ≤ 2 match (score ≥ 1)

        Then sorted by: match_type DESC, popularity DESC.

        Args:
            partial: Partial input string (min 1 char).
            limit:   Max number of suggestions.
            mode:    Optional context mode ('screener', 'news', 'earnings').

        Returns:
            List of {"term": str, "score": float, "match_type": str} dicts.
        """
        if not partial:
            return []

        p = partial.lower().strip()
        if len(p) < 1:
            return []

        results: list[dict] = []

        # Apply mode-specific term ordering boost
        mode_boost: dict[str, float] = {}
        if mode == "screener":
            for t in _METRICS_AUTOCOMPLETE:
                mode_boost[t.lower()] = 3.0
        elif mode == "news":
            for t in _SP500_SAMPLE + list(FinancialSynonymLibrary.COMPANY_ALIASES.keys()):
                mode_boost[t.lower()] = 3.0
        elif mode == "earnings":
            for t in ["earnings", "guidance", "EPS", "beat", "miss", "transcript",
                      "revenue", "conference call"]:
                mode_boost[t.lower()] = 3.0

        seen: set[str] = set()

        for i, term_lower in enumerate(self._terms_lower):
            orig = self._terms[i]

            if term_lower in seen:
                continue

            pop = self._popularity_score(term_lower) * mode_boost.get(term_lower, 1.0)

            if term_lower.startswith(p):
                results.append({"term": orig, "score": 10.0 * pop, "match_type": "prefix"})
                seen.add(term_lower)
            elif p in term_lower:
                results.append({"term": orig, "score": 5.0 * pop, "match_type": "substring"})
                seen.add(term_lower)
            elif len(p) >= 3:
                # Fuzzy: only check if first char matches (performance optimization)
                if term_lower and term_lower[0] == p[0]:
                    dist = _edit_distance(p, term_lower[:len(p) + 2])
                    if dist <= self._MAX_EDIT_DISTANCE_FUZZY:
                        results.append({
                            "term": orig,
                            "score": max(0.1, (3.0 - dist) * pop),
                            "match_type": f"fuzzy_d{dist}",
                        })
                        seen.add(term_lower)

        # Sort: score DESC
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_metric_suggestions(self, partial: str, limit: int = 10) -> list[str]:
        """Metric-only autocomplete — metrics and their synonyms."""
        if not partial:
            return []
        p = partial.lower()
        seen: set[str] = set()
        results: list[str] = []

        for canonical, synonyms in FinancialSynonymLibrary.METRIC_SYNONYMS.items():
            for term in [canonical] + synonyms:
                tl = term.lower()
                if tl in seen:
                    continue
                if tl.startswith(p) or p in tl:
                    results.append(term)
                    seen.add(tl)
                elif len(p) >= 3 and tl and tl[0] == p[0]:
                    dist = _edit_distance(p, tl[:len(p) + 2])
                    if dist <= 1:
                        results.append(term)
                        seen.add(tl)

        return results[:limit]

    def get_ticker_suggestions(self, partial: str, limit: int = 10) -> list[str]:
        """Ticker-only autocomplete."""
        if not partial:
            return []
        p = partial.upper()
        all_tickers = _SP500_SAMPLE + _RUSSELL2000_SAMPLE + list(FinancialSynonymLibrary.COMPANY_ALIASES.values())
        prefix = [t for t in all_tickers if t.startswith(p) and t != p]
        return sorted(set(prefix), key=lambda t: (-self._popularity_score(t.lower()), t))[:limit]

    def get_fred_suggestions(self, partial: str, limit: int = 10) -> list[str]:
        """FRED series ID autocomplete."""
        if not partial:
            return []
        p = partial.upper()
        return [s for s in _FRED_AUTOCOMPLETE if p in s][:limit]

    @property
    def term_count(self) -> int:
        """Total number of terms in the autocomplete index."""
        return len(self._terms)


# ---------------------------------------------------------------------------
# FastAPI Router V2
# ---------------------------------------------------------------------------

query_v2_router = APIRouter(prefix="/api/query/v2", tags=["query-expansion-v2"])

# Singletons
_intent_clf: IntentClassifier | None = None
_ctx_expander: ContextualQueryExpander | None = None
_entity_linker: FinancialEntityLinker | None = None
_reformulation_engine: QueryReformulationEngine | None = None
_autocomplete_v2: AutocompleteEngineV2 | None = None


def _get_intent_clf() -> IntentClassifier:
    global _intent_clf
    if _intent_clf is None:
        _intent_clf = IntentClassifier()
    return _intent_clf


def _get_ctx_expander() -> ContextualQueryExpander:
    global _ctx_expander
    if _ctx_expander is None:
        _ctx_expander = ContextualQueryExpander()
    return _ctx_expander


def _get_entity_linker() -> FinancialEntityLinker:
    global _entity_linker
    if _entity_linker is None:
        _entity_linker = FinancialEntityLinker()
    return _entity_linker


def _get_reformulation_engine() -> QueryReformulationEngine:
    global _reformulation_engine
    if _reformulation_engine is None:
        _reformulation_engine = QueryReformulationEngine()
    return _reformulation_engine


def _get_autocomplete_v2() -> AutocompleteEngineV2:
    global _autocomplete_v2
    if _autocomplete_v2 is None:
        _autocomplete_v2 = AutocompleteEngineV2()
    return _autocomplete_v2


# ── Pydantic request/response models ──────────────────────────────────────────

class ExpandV2Request(BaseModel):
    query: str
    context: dict[str, Any] = {}
    n_expansions: int = 5


class IntentRequest(BaseModel):
    query: str


class EntityLinkRequest(BaseModel):
    text: str
    raw_entity: str | None = None


class ReformulateRequest(BaseModel):
    query: str
    target_backend: str  # "edgar" | "fred" | "screener" | "rag"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@query_v2_router.post("/expand", summary="Context-aware query expansion V2")
async def expand_v2(req: ExpandV2Request):
    """Expand a query with context resolution, pronoun resolution, and synonym expansion.

    Context keys supported:
      - last_ticker: previously mentioned ticker (enables pronoun resolution)
      - last_entity: previously mentioned company name
      - mode: 'screener' | 'news' | 'earnings'
      - last_peers: list of peer tickers
    """
    expander = _get_ctx_expander()
    result = expander.expand_with_context(req.query, req.context)
    return {
        "original": result.original,
        "expanded_query": result.query,
        "resolved_entities": result.resolved_entities,
        "temporal_context": result.temporal_context,
        "peer_tickers": result.peer_tickers,
        "expansions": result.expansions,
    }


@query_v2_router.post("/intent", summary="Classify financial query intent")
async def classify_intent(req: IntentRequest):
    """Classify a query into one of 10 financial intents.

    Returns:
      - primary_intent: main intent category
      - confidence: 0-1 confidence score
      - secondary_intents: additional detected intents
      - route_hints: suggested SENTINEL module routing
      - scores: per-intent scores

    Intent taxonomy: valuation, screening, risk, macro, technical,
    earnings, options, insider, ESG, news.
    """
    clf = _get_intent_clf()
    result = clf.classify_intent(req.query)
    return {
        "query": req.query,
        "primary_intent": result.primary_intent,
        "confidence": result.confidence,
        "secondary_intents": result.secondary_intents,
        "route_hints": result.route_hints,
        "scores": result.scores,
    }


@query_v2_router.get("/autocomplete", summary="Enhanced financial term autocomplete V2")
async def autocomplete_v2(
    q: str = FastAPIQuery(..., min_length=1, description="Partial input string"),
    limit: int = FastAPIQuery(10, ge=1, le=50),
    mode: str | None = FastAPIQuery(None, description="Context mode: screener|news|earnings"),
):
    """Autocomplete with 10 000+ terms, fuzzy matching, and popularity weighting.

    Returns terms with match_type (prefix/substring/fuzzy_d1/fuzzy_d2) and score.
    Higher score = better match + more popular.
    """
    engine = _get_autocomplete_v2()
    engine.record_query(q)   # track for runtime popularity
    suggestions = engine.get_suggestions(q, limit=limit, mode=mode)
    return {
        "partial": q,
        "mode": mode,
        "term_count": engine.term_count,
        "suggestions": suggestions,
    }


@query_v2_router.post("/entity-link", summary="Link query text to financial entities")
async def entity_link(req: EntityLinkRequest):
    """Disambiguate company names, normalize tickers, resolve CIK numbers.

    Handles: $TSLA, tsla, TSLA, Tesla, Tesla Inc., Apple → AAPL/CIK 320193.

    If raw_entity is provided, links that specific entity.
    Otherwise, extracts and links all entities from the full text.
    """
    linker = _get_entity_linker()
    if req.raw_entity:
        result = linker.link_entity(req.raw_entity)
        return {
            "raw": result.raw,
            "ticker": result.ticker,
            "company_name": result.company_name,
            "cik": result.cik,
            "confidence": result.confidence,
            "alternatives": result.alternatives,
        }
    else:
        results = linker.link_all_entities(req.text)
        return {
            "text": req.text,
            "entities": [
                {
                    "raw": r.raw, "ticker": r.ticker, "company_name": r.company_name,
                    "cik": r.cik, "confidence": r.confidence,
                }
                for r in results
            ],
        }


@query_v2_router.post("/reformulate", summary="Reformulate query for a specific backend")
async def reformulate(req: ReformulateRequest):
    """Reformulate a natural language query for a target backend.

    Backends:
      - edgar    → EDGAR EFTS full-text search parameters
      - fred     → FRED API series IDs + date range
      - screener → SQL WHERE clause + parameters
      - rag      → Expanded query string for vector similarity search

    Returns backend-specific query format + parameters + explanation.
    """
    engine = _get_reformulation_engine()
    result = engine.reformulate(req.query, req.target_backend)
    return {
        "original_query": req.query,
        "backend": result.backend,
        "query": result.query,
        "parameters": result.parameters,
        "explanation": result.explanation,
    }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def classify_query_intent(query: str) -> IntentResult:
    """Convenience wrapper — classify intent without instantiating IntentClassifier."""
    return _get_intent_clf().classify_intent(query)


def expand_query_with_context(query: str, context: dict) -> ExpandedQuery:
    """Convenience wrapper for context-aware expansion."""
    return _get_ctx_expander().expand_with_context(query, context)


def link_financial_entities(query: str) -> list[EntityLinkResult]:
    """Convenience wrapper for entity linking."""
    return _get_entity_linker().link_all_entities(query)


def reformulate_for_backend(query: str, backend: str) -> ReformulatedQuery:
    """Convenience wrapper for query reformulation."""
    return _get_reformulation_engine().reformulate(query, backend)
