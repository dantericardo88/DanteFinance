"""
Natural Language Screener V3 — dim_076: NL screener (score 6 → 9).

Converts plain-English screening queries to structured filters, fetches
fundamental data from EDGAR XBRL (via FundamentalDataLayerV3 if available,
else direct EDGAR companyfacts API), and returns ranked results.

Architecture
------------
  QueryParser            — rule-based NL → ScreenerQuery
  ClaudeQueryEnhancer    — Claude API refinement (optional, anthropic SDK)
  MetricFetcher          — EDGAR XBRL + yfinance price metrics
  NLScreenerExecutor     — apply filters, sort, rank, explain
  NLScreenerEngine       — orchestrator, presets, save/load
  QuerySuggestionEngine  — autocomplete, SQL/Bloomberg translation

Data sources
------------
  Primary:   EDGAR XBRL companyfacts API (free, no key required)
             https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  Prices:    yfinance (momentum, beta, returns only)
  Universe:  S&P 500 from Wikipedia; static fallback
  Optional:  sentinel.sai.fundamental_data_layer_v3.FundamentalDataLayerV3

Public API
----------
  engine = NLScreenerEngine()
  result = engine.screen("profitable tech companies with P/E under 20")
  result = engine.run_preset("deep value")
  engine.save_screen("profitable tech companies with P/E under 20", "my_screen")
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from sentinel.sai.fundamental_data_layer_v3 import FundamentalDataLayerV3
    HAS_FDL = True
except ImportError:
    HAS_FDL = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EDGAR_CIK_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&search_text=&output=atom"
_EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_MAX_WORKERS = 8
_HTTP_HEADERS = {"User-Agent": "SENTINEL/3.0 research@sentinel.finance"}
_SCREENS_DIR = Path.home() / ".sentinel" / "screens"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScreenerFilter:
    metric: str
    operator: str            # "<", ">", "<=", ">=", "==", "between", "not_null"
    value: Union[float, Tuple[float, float], None] = None
    sector: Optional[str] = None
    market_cap_tier: Optional[str] = None


@dataclass
class ScreenerQuery:
    raw_query: str
    filters: List[ScreenerFilter] = field(default_factory=list)
    sort_by: Optional[str] = None
    sort_desc: bool = True
    limit: int = 25
    sectors: List[str] = field(default_factory=list)
    market_cap_min: Optional[float] = None
    market_cap_max: Optional[float] = None
    requires_dividend: bool = False
    requires_profitable: bool = False
    exclude_financials: bool = False


@dataclass
class ScreenerMatch:
    ticker: str
    company_name: str
    sector: str
    market_cap: float
    metrics: Dict[str, float]
    matched_filters: List[str]
    score: float
    explanation: str


@dataclass
class ScreenerResult:
    query: ScreenerQuery
    matches: List[ScreenerMatch]
    total_universe: int
    timestamp: datetime
    execution_time_s: float
    sector_breakdown: Dict[str, int] = field(default_factory=dict)
    summary: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for m in self.matches:
            row = {"ticker": m.ticker, "company": m.company_name,
                   "sector": m.sector, "market_cap": m.market_cap,
                   "score": m.score, "explanation": m.explanation}
            row.update(m.metrics)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NL → Metric mapping (50+ terms)
# ---------------------------------------------------------------------------

_METRIC_ALIASES: Dict[str, str] = {
    # Valuation
    "p/e": "pe_ratio", "pe": "pe_ratio", "price to earnings": "pe_ratio",
    "price-to-earnings": "pe_ratio", "earnings multiple": "pe_ratio",
    "p/e ratio": "pe_ratio", "price earnings": "pe_ratio",

    "p/b": "pb_ratio", "pb": "pb_ratio", "price to book": "pb_ratio",
    "price-to-book": "pb_ratio", "book value multiple": "pb_ratio",

    "p/s": "ps_ratio", "ps": "ps_ratio", "price to sales": "ps_ratio",
    "price-to-sales": "ps_ratio",

    "ev/ebitda": "ev_ebitda", "enterprise value to ebitda": "ev_ebitda",
    "ev ebitda": "ev_ebitda",
    "ev/sales": "ev_sales", "enterprise value to sales": "ev_sales",

    "peg": "peg_ratio", "peg ratio": "peg_ratio",
    "earnings yield": "earnings_yield",
    "fcf yield": "fcf_yield", "free cash flow yield": "fcf_yield",

    # Dividends
    "dividend yield": "dividend_yield", "yield": "dividend_yield",
    "div yield": "dividend_yield",
    "payout": "payout_ratio", "payout ratio": "payout_ratio",
    "dividend payout": "payout_ratio",
    "buyback yield": "buyback_yield", "repurchase yield": "buyback_yield",
    "shareholder yield": "shareholder_yield",

    # Margins
    "gross margin": "gross_margin", "gross profit margin": "gross_margin",
    "operating margin": "operating_margin", "ebit margin": "operating_margin",
    "net margin": "net_margin", "profit margin": "net_margin",
    "net profit margin": "net_margin",
    "ebitda margin": "ebitda_margin",

    # Returns
    "roe": "roe", "return on equity": "roe",
    "roa": "roa", "return on assets": "roa",
    "roic": "roic", "return on invested capital": "roic",
    "roce": "roce", "return on capital employed": "roce",

    # Growth
    "revenue growth": "revenue_growth_yoy", "sales growth": "revenue_growth_yoy",
    "top line growth": "revenue_growth_yoy", "top-line growth": "revenue_growth_yoy",
    "revenue growth yoy": "revenue_growth_yoy",
    "eps growth": "eps_growth_yoy", "earnings growth": "eps_growth_yoy",
    "fcf growth": "fcf_growth_yoy", "free cash flow growth": "fcf_growth_yoy",
    "revenue cagr": "revenue_cagr_5y", "sales cagr": "revenue_cagr_5y",

    # Balance sheet
    "debt": "debt_to_equity", "d/e": "debt_to_equity",
    "debt to equity": "debt_to_equity", "leverage": "debt_to_equity",
    "current ratio": "current_ratio", "quick ratio": "quick_ratio",
    "interest coverage": "interest_coverage",
    "net debt": "net_debt", "net debt ebitda": "net_debt_ebitda",

    # Cash
    "cash": "cash_and_equivalents", "free cash flow": "free_cash_flow",
    "fcf": "free_cash_flow", "operating cash flow": "operating_cash_flow",
    "capex": "capex",

    # Market metrics
    "market cap": "market_cap", "market capitalization": "market_cap",
    "enterprise value": "enterprise_value", "ev": "enterprise_value",

    # Momentum / price
    "momentum": "momentum_12m", "12m momentum": "momentum_12m",
    "1 year momentum": "momentum_12m", "price performance": "momentum_12m",
    "6m momentum": "momentum_6m", "6 month momentum": "momentum_6m",
    "3m momentum": "momentum_3m", "1m momentum": "momentum_1m",
    "return": "return_52w", "52 week return": "return_52w",
    "52w return": "return_52w", "ytd return": "return_ytd",
    "ytd": "return_ytd",
    "beta": "beta", "volatility": "volatility_1y",

    # Quality
    "piotroski": "piotroski_score", "f-score": "piotroski_score",
    "altman z": "altman_z", "z score": "altman_z",

    # Income statement
    "revenue": "revenue", "sales": "revenue", "top line": "revenue",
    "earnings": "net_income", "net income": "net_income", "profit": "net_income",
    "ebitda": "ebitda", "eps": "eps",
    "total assets": "total_assets", "assets": "total_assets",
    "shares": "shares_outstanding",
}

_SECTOR_ALIASES: Dict[str, str] = {
    "tech": "Information Technology", "technology": "Information Technology",
    "software": "Information Technology", "semiconductor": "Information Technology",
    "semis": "Information Technology",
    "financial": "Financials", "financials": "Financials",
    "banks": "Financials", "bank": "Financials",
    "healthcare": "Health Care", "health care": "Health Care",
    "pharma": "Health Care", "biotech": "Health Care", "medical": "Health Care",
    "energy": "Energy", "oil": "Energy", "gas": "Energy",
    "consumer": "Consumer Discretionary", "retail": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples", "staples": "Consumer Staples",
    "industrials": "Industrials", "industrial": "Industrials",
    "utilities": "Utilities", "utility": "Utilities",
    "real estate": "Real Estate", "reit": "Real Estate",
    "materials": "Materials", "mining": "Materials",
    "communication": "Communication Services",
    "telecom": "Communication Services", "media": "Communication Services",
}

# ---------------------------------------------------------------------------
# SIC code mapping — sector name → representative SIC code ranges
# Used for EDGAR XBRL and DuckDB queries that filter by SIC.
# ---------------------------------------------------------------------------

_SECTOR_SIC_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "Information Technology": [
        (3570, 3579),   # Computer & Office Equipment
        (3670, 3679),   # Electronic Components
        (3812, 3812),   # Defense Electronics
        (3820, 3829),   # Measuring Instruments
        (7370, 7379),   # Computer Programming, Data Processing
        (7372, 7372),   # Prepackaged Software
    ],
    "Health Care": [
        (2830, 2836),   # Drugs
        (5047, 5047),   # Medical & Hospital Equipment
        (7389, 7389),   # Services-Health Services
        (8000, 8099),   # Health Services
        (8011, 8099),
    ],
    "Financials": [
        (6000, 6099),   # Banks
        (6100, 6199),   # Non-bank Credit
        (6200, 6299),   # Security & Commodity Brokers
        (6300, 6399),   # Insurance
        (6400, 6499),   # Insurance Agents
        (6500, 6599),   # Real Estate Holding
        (6700, 6799),   # Holding Companies
    ],
    "Energy": [
        (1300, 1399),   # Oil & Gas Extraction
        (1311, 1311),
        (2910, 2919),   # Petroleum Refining
        (5172, 5172),   # Petroleum Products Wholesale
    ],
    "Consumer Discretionary": [
        (5200, 5299),   # Retail Stores
        (5300, 5399),
        (5600, 5699),
        (5900, 5999),
        (7011, 7011),   # Hotels
        (7812, 7812),   # Motion Picture
    ],
    "Consumer Staples": [
        (2000, 2099),   # Food
        (2100, 2199),   # Tobacco
        (2800, 2829),   # Chemicals
        (5140, 5149),   # Groceries Wholesale
        (5400, 5499),   # Food Stores
        (5912, 5912),   # Drug Stores
    ],
    "Industrials": [
        (3400, 3499),   # Fabricated Metal
        (3500, 3569),   # Industrial Machinery
        (3700, 3769),   # Transportation Equipment
        (4500, 4599),   # Air Transportation
        (7500, 7599),   # Auto Repair
    ],
    "Utilities": [
        (4900, 4999),   # Electric, Gas & Sanitary Services
    ],
    "Real Estate": [
        (6500, 6552),   # Real Estate
        (6726, 6726),   # Investment Offices
    ],
    "Materials": [
        (1000, 1499),   # Mining
        (2600, 2699),   # Paper
        (2800, 2899),   # Chemicals
        (3300, 3399),   # Primary Metals
    ],
    "Communication Services": [
        (4800, 4899),   # Communications
        (4830, 4833),
        (7372, 7375),   # Internet
        (4813, 4813),   # Telephone
    ],
}


def sector_to_sic_codes(sector: str) -> List[int]:
    """Return a list of SIC codes for a given GICS sector name."""
    ranges = _SECTOR_SIC_RANGES.get(sector, [])
    codes: List[int] = []
    for lo, hi in ranges:
        codes.extend(range(lo, hi + 1))
    return codes


def sector_alias_to_sic_codes(alias: str) -> List[int]:
    """Convert a sector alias (e.g. 'tech') to SIC codes via the alias map."""
    canonical = _SECTOR_ALIASES.get(alias.lower().strip())
    if not canonical:
        return []
    return sector_to_sic_codes(canonical)


_COMPARISON_OPS: List[Tuple[re.Pattern, str]] = [
    # Note: <= / >= must be checked before < / > to avoid partial matches
    (re.compile(r"<=|=<|\bat\s+most\b|\bno\s+more\s+than\b"), "<="),
    (re.compile(r">=|=>\b|\bat\s+least\b|\bno\s+less\s+than\b"), ">="),
    (re.compile(r"\bunder\b|\bbelow\b|\bless\s+than\b|(?<![=!<>])<(?![=])"), "<"),
    (re.compile(r"\bover\b|\babove\b|\bmore\s+than\b|\bgreater\s+than\b|(?<![=!<>])>(?![=])"), ">"),
    (re.compile(r"\bexceeds?\b|\bexceeding\b"), ">"),
    (re.compile(r"\bexceeded\s+by\b|\bdown\s+more\s+than\b"), "<"),
]

_SORT_SUPERLATIVES = {
    "highest": (None, True), "largest": (None, True), "biggest": (None, True),
    "most": (None, True), "top": (None, True), "best": (None, True),
    "lowest": (None, False), "smallest": (None, False), "least": (None, False),
    "worst": (None, False), "cheapest": (None, False),
}

_MARKET_CAP_TIERS: Dict[str, Tuple[float, float]] = {
    "mega": (200e9, float("inf")),
    "large": (10e9, 200e9),
    "mid": (2e9, 10e9),
    "small": (300e6, 2e9),
    "micro": (50e6, 300e6),
    "nano": (0, 50e6),
}

_MARKET_CAP_ALIASES: Dict[str, str] = {
    "large cap": "large", "large-cap": "large", "large caps": "large",
    "mid cap": "mid", "mid-cap": "mid", "mid caps": "mid",
    "small cap": "small", "small-cap": "small", "small caps": "small",
    "micro cap": "micro", "micro-cap": "micro",
    "mega cap": "mega", "mega-cap": "mega",
}

_SPECIAL_KEYWORDS: Dict[str, ScreenerFilter] = {
    "profitable": ScreenerFilter(metric="net_income", operator=">", value=0),
    "dividend": ScreenerFilter(metric="dividend_yield", operator=">", value=0),
    "dividend payer": ScreenerFilter(metric="dividend_yield", operator=">", value=0),
    "dividend payers": ScreenerFilter(metric="dividend_yield", operator=">", value=0),
    "buyback": ScreenerFilter(metric="buyback_yield", operator=">", value=0),
    "no debt": ScreenerFilter(metric="debt_to_equity", operator="<", value=0.1),
    "debt free": ScreenerFilter(metric="debt_to_equity", operator="<", value=0.1),
    "beaten down": ScreenerFilter(metric="return_52w", operator="<", value=-0.2),
    "undervalued": ScreenerFilter(metric="pe_ratio", operator="<", value=15),
    "momentum": ScreenerFilter(metric="momentum_12m", operator=">", value=0.15),
    "high quality": ScreenerFilter(metric="roe", operator=">", value=0.15),
    "strong balance sheet": ScreenerFilter(metric="debt_to_equity", operator="<", value=0.5),
    "value": ScreenerFilter(metric="pb_ratio", operator="<", value=2.0),
    "growth": ScreenerFilter(metric="revenue_growth_yoy", operator=">", value=0.10),
    "high growth": ScreenerFilter(metric="revenue_growth_yoy", operator=">", value=0.20),
    "cheap": ScreenerFilter(metric="pe_ratio", operator="<", value=12),
    "expensive": ScreenerFilter(metric="pe_ratio", operator=">", value=30),
    "high margin": ScreenerFilter(metric="net_margin", operator=">", value=0.15),
    "cash rich": ScreenerFilter(metric="free_cash_flow", operator=">", value=0),
    "low debt": ScreenerFilter(metric="debt_to_equity", operator="<", value=0.5),
    "high debt": ScreenerFilter(metric="debt_to_equity", operator=">", value=2.0),
    "high yield": ScreenerFilter(metric="dividend_yield", operator=">", value=0.04),
    "buying back stock": ScreenerFilter(metric="buyback_yield", operator=">", value=0.02),
    "buying back shares": ScreenerFilter(metric="buyback_yield", operator=">", value=0.02),
}

_NUMBER_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(%|x|times|percent|billion|million|b|m|k)?", re.IGNORECASE
)

_PERCENT_SUFFIXES = {"%", "percent"}
_BILLION_SUFFIXES = {"billion", "b"}
_MILLION_SUFFIXES = {"million", "m"}
_TRILLION_SUFFIXES = {"trillion", "t"}


def _parse_number(s: str) -> Optional[float]:
    """Parse a numeric string like '20%', '3.5x', '2B', '$15' to a float."""
    s = s.strip().lstrip("$").strip()
    m = _NUMBER_PATTERN.match(s)
    if not m:
        return None
    val = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix in _PERCENT_SUFFIXES:
        val = val / 100.0
    elif suffix in _BILLION_SUFFIXES:
        val = val * 1e9
    elif suffix in _MILLION_SUFFIXES:
        val = val * 1e6
    elif suffix in _TRILLION_SUFFIXES:
        val = val * 1e12
    return val


# ---------------------------------------------------------------------------
# QueryParser
# ---------------------------------------------------------------------------

class QueryParser:
    """
    Rule-based natural language query parser.
    Converts plain English to ScreenerQuery with typed filters.
    """

    def parse(self, query: str) -> ScreenerQuery:
        """Parse a natural language query into a structured ScreenerQuery."""
        q = query.lower().strip()
        filters: List[ScreenerFilter] = []
        sectors: List[str] = []
        market_cap_min: Optional[float] = None
        market_cap_max: Optional[float] = None
        sort_by: Optional[str] = None
        sort_desc: bool = True
        limit: int = 25
        requires_dividend = False
        requires_profitable = False

        # Extract limit hints ("top 10", "top 50")
        limit_m = re.search(r"\btop\s+(\d+)\b", q)
        if limit_m:
            limit = int(limit_m.group(1))

        # Sectors
        for alias, canonical in _SECTOR_ALIASES.items():
            if alias in q and canonical not in sectors:
                sectors.append(canonical)

        # Market cap tiers
        for alias, tier in _MARKET_CAP_ALIASES.items():
            if alias in q:
                lo, hi = _MARKET_CAP_TIERS[tier]
                market_cap_min = lo
                market_cap_max = hi if hi < float("inf") else None

        # Explicit market cap numbers ("market cap over $5B", "below 2 billion")
        mc_pattern = re.compile(
            r"market\s+cap\s+(?:of\s+)?(?:over|above|under|below|>|<)?\s*"
            r"\$?(\d+(?:\.\d+)?)\s*(billion|million|b|m)?",
            re.IGNORECASE
        )
        mc_m = mc_pattern.search(q)
        if mc_m:
            mc_val = float(mc_m.group(1))
            mc_suf = (mc_m.group(2) or "b").lower()
            mc_val *= 1e9 if mc_suf.startswith("b") else 1e6
            mc_context = q[max(0, mc_m.start() - 10):mc_m.end() + 5]
            if any(w in mc_context for w in ["over", "above", ">"]):
                market_cap_min = mc_val
            else:
                market_cap_max = mc_val

        # Special keyword filters
        for kw, filt in _SPECIAL_KEYWORDS.items():
            if kw in q:
                # Avoid duplicate metrics
                if not any(f.metric == filt.metric and f.operator == filt.operator
                           for f in filters):
                    import copy
                    filters.append(copy.copy(filt))
                if kw in ("profitable",):
                    requires_profitable = True
                if kw in ("dividend", "dividend payer", "dividend payers", "high yield"):
                    requires_dividend = True

        # Sort superlatives
        for sup, (_, desc) in _SORT_SUPERLATIVES.items():
            if sup in q:
                sort_desc = desc
                # Find the metric being sorted
                for alias in sorted(_METRIC_ALIASES.keys(), key=len, reverse=True):
                    if alias in q:
                        sort_by = _METRIC_ALIASES[alias]
                        break
                break

        # Metric + operator + value patterns
        filters += self._extract_metric_filters(q)

        # Deduplicate filters (prefer more specific ones)
        filters = self._deduplicate_filters(filters)

        # Requires profitable / dividend from filters
        for f in filters:
            if f.metric == "net_income" and f.operator in (">", ">=") and f.value == 0:
                requires_profitable = True
            if f.metric == "dividend_yield" and f.operator in (">", ">="):
                requires_dividend = True

        return ScreenerQuery(
            raw_query=query,
            filters=filters,
            sort_by=sort_by,
            sort_desc=sort_desc,
            limit=limit,
            sectors=sectors,
            market_cap_min=market_cap_min,
            market_cap_max=market_cap_max,
            requires_dividend=requires_dividend,
            requires_profitable=requires_profitable,
        )

    _BETWEEN_PATTERN = re.compile(
        r"(\w[\w\s/\-]*?)\s+(?:is\s+)?between\s+"
        r"\$?(\d+(?:\.\d+)?)\s*(%|x|times|percent|billion|million|b|m|k)?"
        r"\s+and\s+"
        r"\$?(\d+(?:\.\d+)?)\s*(%|x|times|percent|billion|million|b|m|k)?",
        re.IGNORECASE,
    )

    @staticmethod
    def _normalise_val(raw: str, suf: str) -> float:
        """Parse raw number string + optional suffix to float."""
        val = float(raw)
        suf_lower = (suf or "").lower()
        if suf_lower in _PERCENT_SUFFIXES:
            val /= 100.0
        elif suf_lower in _BILLION_SUFFIXES:
            val *= 1e9
        elif suf_lower in _MILLION_SUFFIXES:
            val *= 1e6
        elif suf_lower in _TRILLION_SUFFIXES:
            val *= 1e12
        return val

    def _extract_metric_filters(self, q: str) -> List[ScreenerFilter]:
        """
        Proper token-aware parser producing typed ScreenerFilter nodes.

        Parse tree overview
        -------------------
        For each metric alias found in the query we examine a ±60-char context
        window and classify into one of three filter patterns:

          1. RANGE    : <metric> between <lo> and <hi>   → operator='between'
          2. COMPARE  : <metric> <op> <value>             → operator in <,>,<=,>=,==
          3. NOT_NULL : <metric> is present / has         → operator='not_null'

        The result is a list of :class:`ScreenerFilter` objects (the parse-tree
        leaf nodes) that are later de-duplicated and SQL-translated.
        """
        filters: List[ScreenerFilter] = []
        number_re = re.compile(
            r"\$?(\d+(?:\.\d+)?)\s*(trillion|billion|million|t|b|m|k|%|x|times|percent)?",
            re.IGNORECASE,
        )

        # ── Pattern 1: RANGE ("PE between 10 and 20") ──────────────────────
        for m in self._BETWEEN_PATTERN.finditer(q):
            metric_phrase = m.group(1).strip().lower()
            lo_raw, lo_suf = m.group(2), m.group(3) or ""
            hi_raw, hi_suf = m.group(4), m.group(5) or ""

            # Match the metric phrase against aliases (longest first)
            matched_metric: Optional[str] = None
            for alias in sorted(_METRIC_ALIASES.keys(), key=len, reverse=True):
                if alias in metric_phrase or metric_phrase in alias:
                    matched_metric = _METRIC_ALIASES[alias]
                    break
            if not matched_metric:
                continue

            try:
                lo = self._normalise_val(lo_raw, lo_suf)
                hi = self._normalise_val(hi_raw, hi_suf)
                if lo > hi:
                    lo, hi = hi, lo
                filters.append(ScreenerFilter(
                    metric=matched_metric, operator="between", value=(lo, hi)
                ))
            except (ValueError, TypeError):
                pass

        # ── Pattern 2 & 3: COMPARE / NOT_NULL (existing metric alias scan) ─
        # Split query into clauses at conjunctions so operators don't bleed
        # across clause boundaries (e.g. "PE < 20 and dividend yield > 2%").
        _CLAUSE_SEP = re.compile(r"\s+(?:and|or|but|,)\s+", re.IGNORECASE)
        clauses: List[Tuple[int, str]] = []  # (offset_in_q, clause_text)
        prev = 0
        for sep in _CLAUSE_SEP.finditer(q):
            clauses.append((prev, q[prev:sep.start()]))
            prev = sep.end()
        clauses.append((prev, q[prev:]))

        processed_metrics: set = set()

        for alias in sorted(_METRIC_ALIASES.keys(), key=len, reverse=True):
            if alias not in q:
                continue
            # Require alias to appear as a complete word/phrase (not a substring of another word)
            if not re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", q):
                continue
            metric = _METRIC_ALIASES[alias]

            # Skip if already captured by range parser or processed
            if any(f.metric == metric and f.operator == "between" for f in filters):
                continue
            if metric in processed_metrics:
                continue

            # Find position using word-boundary aware search
            m_pos = re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", q)
            if not m_pos:
                continue
            alias_pos_in_q = m_pos.start()
            clause_text = q  # fallback: whole query
            clause_offset = 0
            for c_off, c_text in clauses:
                if alias in c_text:
                    # Verify the position matches
                    local_pos = c_text.find(alias)
                    if c_off + local_pos == alias_pos_in_q:
                        clause_text = c_text
                        clause_offset = c_off
                        break

            alias_local = alias_pos_in_q - clause_offset

            # Pattern 3: NOT_NULL ("has dividend", "pays dividend")
            prefix_in_clause = clause_text[:alias_local]
            if re.search(r"\b(?:has|pays?|showing)\b", prefix_in_clause):
                if not number_re.search(clause_text):
                    filters.append(ScreenerFilter(metric=metric, operator="not_null", value=None))
                    processed_metrics.add(metric)
                    continue

            op: Optional[str] = None
            value: Optional[float] = None

            # Find operator within THIS clause only
            for op_re, op_str in _COMPARISON_OPS:
                if op_re.search(clause_text):
                    op = op_str
                    break

            if op is None:
                continue

            # Find the number closest to the alias in this clause
            # Prefer numbers AFTER the alias, then before
            num_matches = list(number_re.finditer(clause_text))
            alias_end_local = alias_local + len(alias)
            best: Optional[Tuple[float, float]] = None  # (distance, parsed_val)

            for nm in num_matches:
                nm_start = nm.start()
                # Skip if number overlaps with alias text
                if alias_local <= nm_start < alias_end_local:
                    continue
                raw_val = nm.group(1)
                suf = (nm.group(2) or "").lower()
                try:
                    val = self._normalise_val(raw_val, suf)
                except Exception:
                    continue
                dist = abs(nm_start - alias_end_local)
                if best is None or dist < best[0]:
                    best = (dist, val)

            if best is not None:
                value = best[1]
                if metric == "dividend_yield" and value > 1:
                    value /= 100.0
                filters.append(ScreenerFilter(metric=metric, operator=op, value=value))
                processed_metrics.add(metric)

        return filters

    def _deduplicate_filters(self, filters: List[ScreenerFilter]) -> List[ScreenerFilter]:
        """Remove redundant filters keeping the most restrictive."""
        seen: Dict[str, ScreenerFilter] = {}
        for f in filters:
            key = f.metric
            if key not in seen:
                seen[key] = f
            else:
                existing = seen[key]
                # Keep more restrictive for same direction
                if f.operator in ("<", "<=") and existing.operator in ("<", "<="):
                    if f.value < existing.value:
                        seen[key] = f
                elif f.operator in (">", ">=") and existing.operator in (">", ">="):
                    if f.value > existing.value:
                        seen[key] = f
                else:
                    # Different direction — add as separate entry
                    seen[f"{key}_{f.operator}"] = f
        return list(seen.values())

    def parse_relative_query(self, query: str) -> dict:
        """Parse a relative/ranked query like "top 10 by PE".

        Extracts:
          - limit    : integer N from "top N" / "bottom N"
          - sort_by  : canonical metric name
          - ascending: True for "bottom N" / "lowest", False for "top N" / "highest"

        Examples:
          "top 10 by PE"          → {limit:10, sort_by:"pe_ratio",  ascending:False}
          "bottom 5 by dividend"  → {limit:5,  sort_by:"dividend_yield", ascending:True}
          "top 20 ROE stocks"     → {limit:20, sort_by:"roe", ascending:False}

        Returns:
            dict with keys: limit (int), sort_by (str|None), ascending (bool)
        """
        q = query.lower().strip()

        # Extract limit
        limit = 25  # default
        ascending = False  # default: top = descending

        # "top N" → highest first (descending)
        top_m = re.search(r"\btop\s+(\d+)\b", q)
        if top_m:
            limit = int(top_m.group(1))
            ascending = False

        # "bottom N" → lowest first (ascending)
        bot_m = re.search(r"\bbottom\s+(\d+)\b", q)
        if bot_m:
            limit = int(bot_m.group(1))
            ascending = True

        # Superlative direction overrides
        if any(w in q for w in ("lowest", "smallest", "cheapest", "worst", "least")):
            ascending = True
        elif any(w in q for w in ("highest", "largest", "biggest", "best", "most")):
            ascending = False

        # Metric — longest alias match wins
        sort_by: Optional[str] = None
        for alias in sorted(_METRIC_ALIASES.keys(), key=len, reverse=True):
            if alias in q:
                sort_by = _METRIC_ALIASES[alias]
                break

        return {"limit": limit, "sort_by": sort_by, "ascending": ascending}

    def expand_sector_synonyms(self, term: str) -> Optional[str]:
        """Map a sector synonym/alias to its canonical GICS sector name.

        Covers common informal names like "tech", "technology", "software"
        all mapping to GICS sector code 45 (Information Technology).

        Args:
            term: Free-text sector name or alias.

        Returns:
            Canonical GICS sector string, or None if no match found.

        Examples:
            "tech"       → "Information Technology"
            "technology" → "Information Technology"
            "software"   → "Information Technology"
            "banks"      → "Financials"
            "pharma"     → "Health Care"
        """
        term_lower = term.lower().strip()
        # Direct alias lookup first
        if term_lower in _SECTOR_ALIASES:
            return _SECTOR_ALIASES[term_lower]
        # Partial match fallback (term appears inside an alias key)
        for alias, canonical in _SECTOR_ALIASES.items():
            if term_lower in alias or alias in term_lower:
                return canonical
        return None

    def generate_screener_explanation(self, query: "ScreenerQuery") -> str:
        """Generate a human-readable explanation of an active screener.

        Produces natural-language output like:
          "Showing stocks with PE < 20 AND dividend yield > 2%"

        Args:
            query: A ScreenerQuery (parsed or constructed programmatically).

        Returns:
            Human-readable explanation string.
        """
        parts: list[str] = []

        if query.sectors:
            parts.append(f"Sector: {' or '.join(query.sectors)}")

        if query.market_cap_min is not None or query.market_cap_max is not None:
            lo = f"${query.market_cap_min/1e9:.1f}B" if query.market_cap_min else "any"
            hi = f"${query.market_cap_max/1e9:.1f}B" if query.market_cap_max else "any"
            parts.append(f"Market cap {lo} – {hi}")

        if query.requires_profitable:
            parts.append("must be profitable")

        if query.requires_dividend:
            parts.append("must pay a dividend")

        # Format each filter in plain English
        _OP_WORDS = {
            "<": "<", "<=": "≤", ">": ">", ">=": "≥", "==": "=",
        }
        for f in query.filters:
            # Reverse-lookup a readable metric name
            readable = f.metric.replace("_", " ")
            if f.operator == "between" and isinstance(f.value, tuple):
                parts.append(f"{readable} between {f.value[0]} and {f.value[1]}")
            elif f.operator == "not_null":
                parts.append(f"{readable} is present")
            else:
                op_str = _OP_WORDS.get(f.operator, f.operator)
                val = f.value
                # Format percentages
                if f.metric in {
                    "dividend_yield", "net_margin", "gross_margin",
                    "operating_margin", "roe", "roa", "roic",
                    "revenue_growth_yoy", "fcf_yield", "payout_ratio",
                } and isinstance(val, float) and abs(val) < 10:
                    val_str = f"{val:.0%}"
                else:
                    val_str = str(val)
                parts.append(f"{readable} {op_str} {val_str}")

        if query.sort_by:
            direction = "descending" if query.sort_desc else "ascending"
            parts.append(f"sorted by {query.sort_by.replace('_', ' ')} ({direction})")

        base = f"Showing top {query.limit} stocks"
        if parts:
            return base + " with " + " AND ".join(parts)
        return base

    def explain_parse(self, query: str) -> str:
        """Return human-readable explanation of how the query was interpreted."""
        sq = self.parse(query)
        lines = [f'Query: "{query}"', "Interpreted as:"]
        if sq.sectors:
            lines.append(f"  Sectors: {', '.join(sq.sectors)}")
        if sq.market_cap_min or sq.market_cap_max:
            lo = f"${sq.market_cap_min/1e9:.1f}B" if sq.market_cap_min else "any"
            hi = f"${sq.market_cap_max/1e9:.1f}B" if sq.market_cap_max else "any"
            lines.append(f"  Market cap: {lo} to {hi}")
        if sq.requires_dividend:
            lines.append("  Must pay dividends")
        if sq.requires_profitable:
            lines.append("  Must be profitable (net income > 0)")
        for f in sq.filters:
            if f.operator == "between":
                lines.append(f"  {f.metric} between {f.value[0]} and {f.value[1]}")
            else:
                lines.append(f"  {f.metric} {f.operator} {f.value}")
        if sq.sort_by:
            direction = "descending" if sq.sort_desc else "ascending"
            lines.append(f"  Sort by {sq.sort_by} ({direction})")
        lines.append(f"  Return top {sq.limit} results")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClaudeQueryEnhancer
# ---------------------------------------------------------------------------

class ClaudeQueryEnhancer:
    """
    Optionally use the Claude API to refine complex/ambiguous queries.
    Falls back to the rule-based parser if anthropic SDK is not available.
    """

    def __init__(self):
        self._client = None
        if HAS_ANTHROPIC:
            try:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    self._client = anthropic.Anthropic(api_key=api_key)
            except Exception as exc:
                logger.debug("Claude API init failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def enhance_parse(self, query: str,
                      base_parse: ScreenerQuery) -> ScreenerQuery:
        """
        Use Claude to resolve ambiguity and add implied filters.
        Falls back to base_parse if unavailable.
        """
        if not self.available:
            return base_parse

        # Build prompt
        metric_list = sorted(set(_METRIC_ALIASES.values()))
        system_prompt = (
            "You are a financial stock screener assistant. "
            "Given a natural language query and a partial parse, "
            "return a JSON object with additional filters. "
            "Only output valid JSON, no commentary. "
            "Available metrics: " + ", ".join(metric_list[:40])
        )
        user_prompt = (
            f"Query: {query}\n"
            f"Already parsed filters: {[f.__dict__ for f in base_parse.filters]}\n"
            "Return JSON with keys: filters (list of {{metric, operator, value}}), "
            "sectors (list), requires_profitable (bool), requires_dividend (bool). "
            "Only add filters that are clearly implied but not already present."
        )
        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text.strip()
            # Strip markdown code fences
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            parsed = json.loads(text)

            # Merge filters
            existing_metrics = {f.metric for f in base_parse.filters}
            for raw_f in parsed.get("filters", []):
                metric = raw_f.get("metric")
                if metric and metric not in existing_metrics:
                    try:
                        base_parse.filters.append(ScreenerFilter(
                            metric=metric,
                            operator=raw_f.get("operator", ">"),
                            value=raw_f.get("value"),
                        ))
                        existing_metrics.add(metric)
                    except Exception:
                        pass

            for s in parsed.get("sectors", []):
                if s not in base_parse.sectors:
                    base_parse.sectors.append(s)
            if parsed.get("requires_profitable"):
                base_parse.requires_profitable = True
            if parsed.get("requires_dividend"):
                base_parse.requires_dividend = True

        except Exception as exc:
            logger.debug("Claude enhancement failed: %s", exc)

        return base_parse

    def generate_query_suggestions(self, partial: str) -> List[str]:
        """Generate autocomplete suggestions using Claude."""
        if not self.available:
            return []
        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Complete this stock screener query: '{partial}'. "
                        "Return 5 complete query suggestions as a JSON list of strings."
                    )
                }],
            )
            text = response.content[0].text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            suggestions = json.loads(text)
            return suggestions if isinstance(suggestions, list) else []
        except Exception:
            return []


# ---------------------------------------------------------------------------
# EDGAR CIK resolver and facts fetcher
# ---------------------------------------------------------------------------

_cik_cache: Dict[str, str] = {}

def _load_cik_map() -> Dict[str, str]:
    """Download SEC company_tickers.json and build ticker → CIK map."""
    global _cik_cache
    if _cik_cache:
        return _cik_cache
    try:
        resp = requests.get(_EDGAR_TICKERS_URL, headers=_HTTP_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        mapping: Dict[str, str] = {}
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if ticker:
                mapping[ticker] = cik
        _cik_cache = mapping
        return mapping
    except Exception as exc:
        logger.warning("Could not load CIK map: %s", exc)
        return {}


def resolve_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to zero-padded 10-digit CIK."""
    mapping = _load_cik_map()
    return mapping.get(ticker.upper())


def fetch_edgar_facts(cik: str) -> Optional[dict]:
    """Fetch EDGAR XBRL companyfacts for a given CIK."""
    url = _EDGAR_FACTS_URL.format(cik=cik)
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("EDGAR facts fetch failed for CIK %s: %s", cik, exc)
        return None


def _latest_annual_value(facts: dict, concept: str,
                         taxonomy: str = "us-gaap") -> Optional[float]:
    """Extract the most recent annual value for an XBRL concept."""
    try:
        units = facts["facts"][taxonomy][concept]["units"]
        # Prefer USD, then shares, then pure
        for unit_key in ["USD", "shares", "pure"]:
            if unit_key not in units:
                continue
            data = units[unit_key]
            annual = [d for d in data
                      if d.get("form") in ("10-K", "10-K/A")
                      and d.get("val") is not None]
            if not annual:
                annual = [d for d in data if d.get("val") is not None]
            if annual:
                annual.sort(key=lambda d: d.get("end", ""), reverse=True)
                return float(annual[0]["val"])
    except (KeyError, IndexError, TypeError):
        pass
    return None


def _ltm_value(facts: dict, concept: str,
               taxonomy: str = "us-gaap") -> Optional[float]:
    """Compute last-twelve-months (LTM) value by summing four recent quarters."""
    try:
        units = facts["facts"][taxonomy][concept]["units"]
        for unit_key in ["USD", "shares"]:
            if unit_key not in units:
                continue
            data = [d for d in units[unit_key]
                    if d.get("form") in ("10-Q", "10-K")
                    and d.get("val") is not None]
            # Deduplicate by end date
            seen: set = set()
            deduped = []
            for d in sorted(data, key=lambda x: x.get("end", ""), reverse=True):
                key = d["end"]
                if key not in seen:
                    seen.add(key)
                    deduped.append(d)
            if len(deduped) >= 4:
                ltm = sum(float(d["val"]) for d in deduped[:4])
                return ltm
    except (KeyError, IndexError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# MetricFetcher
# ---------------------------------------------------------------------------

class MetricFetcher:
    """
    Fetch all required screener metrics for a universe of tickers.
    Uses FundamentalDataLayerV3 if available, else direct EDGAR XBRL.
    """

    # EDGAR concept mappings for each metric
    _EDGAR_CONCEPTS: Dict[str, List[Tuple[str, str]]] = {
        "revenue": [
            ("us-gaap", "Revenues"),
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "SalesRevenueNet"),
        ],
        "net_income": [
            ("us-gaap", "NetIncomeLoss"),
            ("us-gaap", "ProfitLoss"),
        ],
        "gross_profit": [
            ("us-gaap", "GrossProfit"),
        ],
        "operating_income": [
            ("us-gaap", "OperatingIncomeLoss"),
        ],
        "ebitda": [
            ("us-gaap", "EarningsBeforeInterestTaxesDepreciationAndAmortization"),
        ],
        "total_assets": [
            ("us-gaap", "Assets"),
        ],
        "total_equity": [
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ],
        "total_debt": [
            ("us-gaap", "LongTermDebt"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
        ],
        "cash_and_equivalents": [
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            ("us-gaap", "CashCashEquivalentsAndShortTermInvestments"),
        ],
        "shares_outstanding": [
            ("us-gaap", "CommonStockSharesOutstanding"),
            ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
        ],
        "dividends_paid": [
            ("us-gaap", "PaymentsOfDividendsCommonStock"),
            ("us-gaap", "DividendsPaid"),
        ],
        "capex": [
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
            ("us-gaap", "CapitalExpendituresIncurredButNotYetPaid"),
        ],
        "operating_cash_flow": [
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ],
        "interest_expense": [
            ("us-gaap", "InterestExpense"),
            ("us-gaap", "InterestAndDebtExpense"),
        ],
        "eps": [
            ("us-gaap", "EarningsPerShareBasic"),
            ("us-gaap", "EarningsPerShareDiluted"),
        ],
    }

    def __init__(self):
        self._cache: Dict[str, Dict[str, float]] = {}
        self._price_cache: Dict[str, Dict[str, float]] = {}
        self._sector_cache: Dict[str, str] = {}
        self._name_cache: Dict[str, str] = {}

    def _fetch_edgar_metrics(self, ticker: str) -> Dict[str, float]:
        """Fetch fundamental metrics from EDGAR XBRL companyfacts."""
        cik = resolve_cik(ticker)
        if not cik:
            return {}
        facts = fetch_edgar_facts(cik)
        if not facts:
            return {}

        raw: Dict[str, float] = {}
        for metric, concepts in self._EDGAR_CONCEPTS.items():
            for taxonomy, concept in concepts:
                val = _latest_annual_value(facts, concept, taxonomy)
                if val is None:
                    val = _ltm_value(facts, concept, taxonomy)
                if val is not None:
                    raw[metric] = val
                    break

        return raw

    def _fetch_price_metrics(self, ticker: str) -> Dict[str, float]:
        """Fetch price-based metrics from yfinance."""
        if not HAS_YF:
            return {}
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2y")
            if hist.empty:
                return {}
            close = hist["Close"]
            now_price = float(close.iloc[-1])
            metrics: Dict[str, float] = {}

            def pct_change(days: int) -> float:
                if len(close) < days:
                    return float("nan")
                return float((close.iloc[-1] / close.iloc[-days]) - 1)

            metrics["momentum_1m"] = pct_change(21)
            metrics["momentum_3m"] = pct_change(63)
            metrics["momentum_6m"] = pct_change(126)
            metrics["momentum_12m"] = pct_change(252)
            metrics["return_52w"] = pct_change(252)

            # YTD
            jan1 = close.index[close.index.year == datetime.now().year][0] if any(
                close.index.year == datetime.now().year) else close.index[0]
            ytd_base = float(close[close.index >= jan1].iloc[0])
            metrics["return_ytd"] = (now_price - ytd_base) / ytd_base

            # Beta vs SPY
            try:
                spy = yf.Ticker("SPY").history(period="1y")["Close"]
                min_len = min(len(close), len(spy))
                if min_len > 20:
                    r_stock = close.tail(min_len).pct_change().dropna()
                    r_spy = spy.tail(min_len).pct_change().dropna()
                    min_len2 = min(len(r_stock), len(r_spy))
                    cov = float(np.cov(r_stock.tail(min_len2).values,
                                      r_spy.tail(min_len2).values)[0][1])
                    var = float(np.var(r_spy.tail(min_len2).values, ddof=1))
                    metrics["beta"] = cov / var if var > 0 else 1.0
            except Exception:
                metrics["beta"] = 1.0

            # Volatility (annualized)
            daily_ret = close.pct_change().dropna()
            if len(daily_ret) >= 20:
                metrics["volatility_1y"] = float(daily_ret.tail(252).std() * math.sqrt(252))

            metrics["price"] = now_price

            # Info-based metrics
            info = {}
            try:
                info = t.info or {}
            except Exception:
                pass

            for key, ikey in [("market_cap", "marketCap"),
                               ("dividend_yield", "dividendYield"),
                               ("pe_ratio", "trailingPE"),
                               ("pb_ratio", "priceToBook"),
                               ("ps_ratio", "priceToSalesTrailing12Months"),
                               ("peg_ratio", "pegRatio"),
                               ("buyback_yield", "buybackYield"),
                               ("short_interest", "shortPercentOfFloat")]:
                val = info.get(ikey)
                if val is not None:
                    try:
                        metrics[key] = float(val)
                    except Exception:
                        pass

            sector = info.get("sector", "")
            name = info.get("longName", ticker)
            if sector:
                self._sector_cache[ticker] = sector
            if name:
                self._name_cache[ticker] = name

            return metrics
        except Exception as exc:
            logger.debug("Price metrics failed for %s: %s", ticker, exc)
            return {}

    def _compute_derived_metrics(self, ticker: str,
                                 raw: Dict[str, float],
                                 price_m: Dict[str, float]) -> Dict[str, float]:
        """Compute derived/ratio metrics from raw EDGAR + price data."""
        m: Dict[str, float] = {}
        m.update(raw)
        m.update(price_m)

        price = m.get("price", float("nan"))
        shares = m.get("shares_outstanding", float("nan"))

        # Market cap (if not from yfinance)
        if "market_cap" not in m and not math.isnan(price) and not math.isnan(shares):
            m["market_cap"] = price * shares

        mkt_cap = m.get("market_cap", float("nan"))

        # Revenue-based metrics
        revenue = m.get("revenue", float("nan"))
        net_income = m.get("net_income", float("nan"))
        gross_profit = m.get("gross_profit", float("nan"))
        total_assets = m.get("total_assets", float("nan"))
        total_equity = m.get("total_equity", float("nan"))
        total_debt = m.get("total_debt", float("nan"))
        ocf = m.get("operating_cash_flow", float("nan"))
        capex = m.get("capex", float("nan"))
        interest = m.get("interest_expense", float("nan"))

        def _safe_div(a, b):
            if math.isnan(a) or math.isnan(b) or b == 0:
                return float("nan")
            return a / b

        # Margins
        m["gross_margin"] = _safe_div(gross_profit, revenue)
        m["net_margin"] = _safe_div(net_income, revenue)

        # FCF
        if not math.isnan(ocf) and not math.isnan(capex):
            m["free_cash_flow"] = ocf - capex
        elif not math.isnan(ocf):
            m["free_cash_flow"] = ocf
        fcf = m.get("free_cash_flow", float("nan"))

        # Returns
        m["roe"] = _safe_div(net_income, total_equity)
        m["roa"] = _safe_div(net_income, total_assets)

        # Debt/equity
        m["debt_to_equity"] = _safe_div(total_debt, total_equity)

        # Net debt
        cash = m.get("cash_and_equivalents", float("nan"))
        if not math.isnan(total_debt) and not math.isnan(cash):
            m["net_debt"] = total_debt - cash

        # Interest coverage
        operating_income = m.get("operating_income", float("nan"))
        m["interest_coverage"] = _safe_div(operating_income, interest)

        # P/E if not from yfinance
        if "pe_ratio" not in m and not math.isnan(mkt_cap) and not math.isnan(net_income) and net_income > 0:
            m["pe_ratio"] = _safe_div(mkt_cap, net_income)

        # P/S
        if "ps_ratio" not in m and not math.isnan(mkt_cap) and not math.isnan(revenue) and revenue > 0:
            m["ps_ratio"] = _safe_div(mkt_cap, revenue)

        # P/B
        if "pb_ratio" not in m:
            book = m.get("total_equity", float("nan"))
            m["pb_ratio"] = _safe_div(mkt_cap, book)

        # EV
        net_debt = m.get("net_debt", float("nan"))
        if not math.isnan(mkt_cap) and not math.isnan(net_debt):
            m["enterprise_value"] = mkt_cap + net_debt
        ev = m.get("enterprise_value", float("nan"))

        # FCF yield
        m["fcf_yield"] = _safe_div(fcf, mkt_cap)

        # Payout ratio
        dividends = m.get("dividends_paid", float("nan"))
        m["payout_ratio"] = _safe_div(abs(dividends) if not math.isnan(dividends) else float("nan"),
                                       net_income)

        # Earnings yield
        m["earnings_yield"] = _safe_div(net_income, mkt_cap)

        # Shareholder yield (dividend + buyback)
        dy = m.get("dividend_yield", float("nan"))
        by = m.get("buyback_yield", float("nan"))
        if not math.isnan(dy) and not math.isnan(by):
            m["shareholder_yield"] = dy + by

        # ROIC proxy: net_income / (total_equity + total_debt)
        total_cap = (total_equity if not math.isnan(total_equity) else 0) + \
                    (total_debt if not math.isnan(total_debt) else 0)
        m["roic"] = _safe_div(net_income, total_cap) if total_cap > 0 else float("nan")

        return m

    def _fetch_single(self, ticker: str) -> Dict[str, float]:
        if ticker in self._cache:
            return self._cache[ticker]

        # Try FundamentalDataLayerV3 first
        if HAS_FDL:
            try:
                fdl = FundamentalDataLayerV3()
                metrics_df = fdl.batch_metrics([ticker], list(self._EDGAR_CONCEPTS.keys()))
                if not metrics_df.empty and ticker in metrics_df.index:
                    raw = metrics_df.loc[ticker].to_dict()
                    price_m = self._fetch_price_metrics(ticker)
                    combined = self._compute_derived_metrics(ticker, raw, price_m)
                    self._cache[ticker] = combined
                    return combined
            except Exception:
                pass

        # Fallback: direct EDGAR + yfinance
        raw = self._fetch_edgar_metrics(ticker)
        price_m = self._fetch_price_metrics(ticker)
        combined = self._compute_derived_metrics(ticker, raw, price_m)
        self._cache[ticker] = combined
        return combined

    def fetch_metrics(self, tickers: List[str],
                      metrics: List[str]) -> pd.DataFrame:
        """
        Fetch all required metrics for a list of tickers in parallel.
        Returns a DataFrame indexed by ticker.
        """
        rows: Dict[str, Dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futures = {ex.submit(self._fetch_single, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    data = future.result(timeout=30)
                    rows[ticker] = data
                except Exception as exc:
                    logger.debug("Metric fetch failed for %s: %s", ticker, exc)
                    rows[ticker] = {}

        df = pd.DataFrame.from_dict(rows, orient="index")
        # Ensure requested metrics columns exist
        for m in metrics:
            if m not in df.columns:
                df[m] = float("nan")
        return df

    def get_sector(self, ticker: str) -> str:
        return self._sector_cache.get(ticker, "Unknown")

    def get_name(self, ticker: str) -> str:
        return self._name_cache.get(ticker, ticker)


# ---------------------------------------------------------------------------
# NLScreenerExecutor
# ---------------------------------------------------------------------------

def _apply_filter(series: pd.Series, filt: ScreenerFilter) -> pd.Series:
    """Apply a single ScreenerFilter to a pandas Series, return boolean mask."""
    if filt.operator == "<":
        return series < filt.value
    elif filt.operator == "<=":
        return series <= filt.value
    elif filt.operator == ">":
        return series > filt.value
    elif filt.operator == ">=":
        return series >= filt.value
    elif filt.operator == "==":
        return series == filt.value
    elif filt.operator == "between" and isinstance(filt.value, tuple):
        lo, hi = filt.value
        return (series >= lo) & (series <= hi)
    elif filt.operator == "not_null":
        return series.notna() & (series != 0)
    return pd.Series(True, index=series.index)


class NLScreenerExecutor:
    """
    Execute a parsed ScreenerQuery against a universe of tickers.
    """

    def __init__(self):
        self.fetcher = MetricFetcher()

    def run(self, query: ScreenerQuery,
            universe: List[str] = None) -> ScreenerResult:
        """Full execution: fetch metrics → filter → sort → explain."""
        t0 = time.time()
        if universe is None:
            universe = _get_sp500_tickers()

        # Determine which metrics are needed
        needed_metrics = set()
        for f in query.filters:
            needed_metrics.add(f.metric)
        if query.sort_by:
            needed_metrics.add(query.sort_by)
        if query.requires_dividend:
            needed_metrics.add("dividend_yield")
        if query.requires_profitable:
            needed_metrics.add("net_income")
        needed_metrics.update([
            "market_cap", "pe_ratio", "revenue_growth_yoy", "net_margin",
            "roe", "debt_to_equity", "dividend_yield",
        ])
        needed_metrics = list(needed_metrics)

        df = self.fetcher.fetch_metrics(universe, needed_metrics)

        # Sector filter
        if query.sectors:
            sector_mask = pd.Series(False, index=df.index)
            for ticker in df.index:
                sec = self.fetcher.get_sector(ticker)
                for wanted_sec in query.sectors:
                    if wanted_sec.lower() in sec.lower():
                        sector_mask[ticker] = True
            df = df[sector_mask]

        # Market cap filter
        if "market_cap" in df.columns:
            if query.market_cap_min is not None:
                df = df[df["market_cap"].fillna(0) >= query.market_cap_min]
            if query.market_cap_max is not None:
                df = df[df["market_cap"].fillna(float("inf")) <= query.market_cap_max]

        # Profitability filter
        if query.requires_profitable and "net_income" in df.columns:
            df = df[df["net_income"].fillna(0) > 0]

        # Dividend filter
        if query.requires_dividend and "dividend_yield" in df.columns:
            df = df[df["dividend_yield"].fillna(0) > 0]

        # Apply metric filters
        matched_filters_per_ticker: Dict[str, List[str]] = {t: [] for t in df.index}
        for filt in query.filters:
            col = filt.metric
            if col not in df.columns:
                # Compute revenue_growth_yoy on the fly if missing
                if col == "revenue_growth_yoy":
                    df[col] = float("nan")
                else:
                    df[col] = float("nan")
                continue
            non_null = df[col].notna()
            mask = _apply_filter(df.loc[non_null, col], filt)
            drop_tickers = set(df.index[non_null][~mask].tolist())
            keep_tickers = set(df.index[non_null][mask].tolist())
            for t in keep_tickers:
                matched_filters_per_ticker[t].append(
                    f"{col} {filt.operator} {filt.value}"
                )
            df = df.drop(index=list(drop_tickers), errors="ignore")

        # Score: count matched non-null metrics as quality signal
        scores: Dict[str, float] = {}
        for ticker in df.index:
            non_nan = df.loc[ticker].notna().sum()
            scores[ticker] = float(non_nan)

        df["_score"] = pd.Series(scores)

        # Sort
        if query.sort_by and query.sort_by in df.columns:
            df = df.sort_values(query.sort_by, ascending=not query.sort_desc,
                                na_position="last")
        else:
            df = df.sort_values("_score", ascending=False, na_position="last")

        df = df.head(query.limit)

        # Build ScreenerMatch objects
        matches: List[ScreenerMatch] = []
        for ticker in df.index:
            row = df.loc[ticker]
            row_dict = row.to_dict()
            row_dict.pop("_score", None)
            mkt = float(row_dict.get("market_cap", float("nan")))
            sector = self.fetcher.get_sector(ticker)
            name = self.fetcher.get_name(ticker)
            matched = matched_filters_per_ticker.get(ticker, [])
            explanation = self._explain_match(ticker, row_dict, query, matched)
            matches.append(ScreenerMatch(
                ticker=ticker,
                company_name=name,
                sector=sector,
                market_cap=mkt,
                metrics={k: v for k, v in row_dict.items()
                         if isinstance(v, (int, float)) and not math.isnan(v)},
                matched_filters=matched,
                score=float(scores.get(ticker, 0)),
                explanation=explanation,
            ))

        sector_breakdown = self.get_sector_breakdown_from_matches(matches)
        execution_time = time.time() - t0

        result = ScreenerResult(
            query=query,
            matches=matches,
            total_universe=len(universe),
            timestamp=datetime.utcnow(),
            execution_time_s=execution_time,
            sector_breakdown=sector_breakdown,
            summary=self._build_summary(query, matches, len(universe)),
        )
        return result

    def _explain_match(self, ticker: str, metrics: Dict[str, float],
                       query: ScreenerQuery,
                       matched: List[str]) -> str:
        parts = []
        pe = metrics.get("pe_ratio")
        if pe and not math.isnan(pe):
            parts.append(f"P/E {pe:.1f}x")
        rev_gr = metrics.get("revenue_growth_yoy")
        if rev_gr and not math.isnan(rev_gr):
            parts.append(f"rev growth {rev_gr:.0%}")
        nm = metrics.get("net_margin")
        if nm and not math.isnan(nm):
            parts.append(f"net margin {nm:.0%}")
        dy = metrics.get("dividend_yield")
        if dy and not math.isnan(dy) and dy > 0:
            parts.append(f"div yield {dy:.1%}")
        roe = metrics.get("roe")
        if roe and not math.isnan(roe):
            parts.append(f"ROE {roe:.0%}")
        de = metrics.get("debt_to_equity")
        if de and not math.isnan(de):
            parts.append(f"D/E {de:.1f}x")
        if parts:
            return f"{ticker}: " + " | ".join(parts)
        return f"{ticker}: passes all filters"

    def _build_summary(self, query: ScreenerQuery,
                       matches: List[ScreenerMatch],
                       universe_size: int) -> str:
        return (
            f"Found {len(matches)} matches from {universe_size}-ticker universe "
            f"for query: '{query.raw_query}'. "
            f"Top result: {matches[0].ticker if matches else 'N/A'}."
        )

    def get_sector_breakdown(self, result: ScreenerResult) -> Dict[str, int]:
        return result.sector_breakdown

    def get_sector_breakdown_from_matches(self,
                                          matches: List[ScreenerMatch]) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}
        for m in matches:
            sec = m.sector or "Unknown"
            breakdown[sec] = breakdown.get(sec, 0) + 1
        return breakdown

    def explain_results(self, result: ScreenerResult) -> str:
        lines = [f"Results for: '{result.query.raw_query}'",
                 f"Total matches: {len(result.matches)} / {result.total_universe} universe",
                 ""]
        for i, m in enumerate(result.matches[:10], 1):
            lines.append(f"{i:2}. {m.ticker:6} | {m.company_name[:30]:30} "
                         f"| {m.sector[:20]:20} | {m.explanation}")
        return "\n".join(lines)

    def compare_to_benchmark(self, result: ScreenerResult,
                             benchmark: str = "SPY") -> Dict[str, Any]:
        """Compare average metrics of screener results vs SPY."""
        if not result.matches:
            return {}
        all_metrics: Dict[str, List[float]] = {}
        for m in result.matches:
            for k, v in m.metrics.items():
                if not math.isnan(v):
                    all_metrics.setdefault(k, []).append(v)
        averages = {k: sum(vals) / len(vals) for k, vals in all_metrics.items()
                    if vals}

        # Get benchmark metrics
        bm_data = self.fetcher._fetch_single(benchmark)
        comparison: Dict[str, Any] = {}
        for metric, avg_val in averages.items():
            bm_val = bm_data.get(metric)
            if bm_val and not math.isnan(bm_val):
                comparison[metric] = {
                    "screen_avg": avg_val,
                    "benchmark": bm_val,
                    "relative": avg_val / bm_val if bm_val != 0 else float("nan"),
                }
        return comparison


# ---------------------------------------------------------------------------
# Universe helper (shared)
# ---------------------------------------------------------------------------

def _get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia; static fallback on failure."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception:
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
            "BRK-B", "UNH", "LLY", "JPM", "V", "XOM", "MA", "AVGO", "PG",
            "HD", "JNJ", "MRK", "COST", "ABBV", "CVX", "BAC", "NFLX",
            "KO", "AMD", "ORCL", "PEP", "TMO", "ADBE", "INTC", "MCD",
            "CRM", "DIS", "ACN", "WMT", "CSCO", "ABT", "VZ", "DHR",
            "PM", "TXN", "AMGN", "NKE", "NEE", "RTX", "BMY", "QCOM", "HON",
        ]


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, ScreenerQuery] = {
    "deep value": ScreenerQuery(
        raw_query="deep value",
        filters=[
            ScreenerFilter("pe_ratio", "<", 10),
            ScreenerFilter("pb_ratio", "<", 1),
            ScreenerFilter("net_income", ">", 0),
            ScreenerFilter("debt_to_equity", "<", 0.5),
        ],
        sort_by="pe_ratio",
        sort_desc=False,
        limit=25,
        requires_profitable=True,
    ),
    "growth at reasonable price": ScreenerQuery(
        raw_query="growth at reasonable price (GARP)",
        filters=[
            ScreenerFilter("peg_ratio", "<", 1.5),
            ScreenerFilter("revenue_growth_yoy", ">", 0.15),
            ScreenerFilter("net_income", ">", 0),
        ],
        sort_by="peg_ratio",
        sort_desc=False,
        limit=25,
        requires_profitable=True,
    ),
    "garp": ScreenerQuery(
        raw_query="GARP",
        filters=[
            ScreenerFilter("peg_ratio", "<", 1.5),
            ScreenerFilter("revenue_growth_yoy", ">", 0.15),
        ],
        sort_by="peg_ratio",
        sort_desc=False,
        limit=25,
    ),
    "dividend aristocrats": ScreenerQuery(
        raw_query="dividend aristocrats",
        filters=[
            ScreenerFilter("dividend_yield", ">", 0.02),
            ScreenerFilter("payout_ratio", "<", 0.60),
            ScreenerFilter("net_income", ">", 0),
        ],
        sort_by="dividend_yield",
        sort_desc=True,
        limit=25,
        requires_dividend=True,
        requires_profitable=True,
    ),
    "momentum leaders": ScreenerQuery(
        raw_query="momentum leaders",
        filters=[
            ScreenerFilter("momentum_12m", ">", 0.10),
            ScreenerFilter("momentum_6m", ">", 0.05),
        ],
        sort_by="momentum_12m",
        sort_desc=True,
        limit=25,
    ),
    "quality compounders": ScreenerQuery(
        raw_query="quality compounders",
        filters=[
            ScreenerFilter("roic", ">", 0.15),
            ScreenerFilter("gross_margin", ">", 0.40),
            ScreenerFilter("debt_to_equity", "<", 1.0),
            ScreenerFilter("net_income", ">", 0),
        ],
        sort_by="roic",
        sort_desc=True,
        limit=25,
        requires_profitable=True,
    ),
    "turnaround candidates": ScreenerQuery(
        raw_query="turnaround candidates",
        filters=[
            ScreenerFilter("revenue_growth_yoy", ">", 0.10),
            ScreenerFilter("momentum_6m", ">", 0.0),
        ],
        sort_by="revenue_growth_yoy",
        sort_desc=True,
        limit=25,
    ),
    "small cap value": ScreenerQuery(
        raw_query="small cap value",
        filters=[
            ScreenerFilter("pb_ratio", "<", 1.5),
            ScreenerFilter("net_income", ">", 0),
        ],
        sort_by="pb_ratio",
        sort_desc=False,
        limit=25,
        market_cap_min=300e6,
        market_cap_max=2e9,
        requires_profitable=True,
    ),
}


# ---------------------------------------------------------------------------
# QuerySuggestionEngine
# ---------------------------------------------------------------------------

class QuerySuggestionEngine:
    """Autocomplete, popular screens, Bloomberg BQL translation, SQL translation."""

    _POPULAR_SCREENS = [
        "profitable tech companies with P/E under 20",
        "dividend payers with yield over 3% and payout below 60%",
        "high momentum small caps",
        "value stocks beaten down more than 30% with strong balance sheet",
        "profitable companies buying back stock",
        "growth stocks with 20%+ revenue growth",
        "undervalued financials",
        "large cap quality compounders with ROE over 20%",
        "cheap energy stocks with dividend yield over 4%",
        "biotech companies with strong balance sheet and no debt",
        "consumer staples with dividend yield over 2%",
        "semiconductor stocks with high gross margin",
    ]

    _COMPLETIONS = [
        "profitable tech companies with P/E under {n}",
        "dividend payers with yield over {n}%",
        "revenue growth over {n}%",
        "net margin above {n}%",
        "debt to equity below {n}",
        "ROE over {n}%",
        "P/E ratio under {n}",
        "P/B below {n}",
        "ROIC above {n}%",
        "free cash flow yield over {n}%",
    ]

    def suggest(self, partial: str) -> List[str]:
        """Return autocomplete suggestions for a partial query."""
        partial_lower = partial.lower().strip()
        suggestions = []
        for screen in self._POPULAR_SCREENS:
            if partial_lower in screen.lower():
                suggestions.append(screen)
        if len(suggestions) < 5:
            for screen in self._POPULAR_SCREENS:
                words = partial_lower.split()
                if any(w in screen.lower() for w in words if len(w) > 3):
                    if screen not in suggestions:
                        suggestions.append(screen)
        return suggestions[:8]

    def get_popular_screens(self) -> List[str]:
        return list(self._POPULAR_SCREENS)

    def translate_to_bloomberg(self, query: ScreenerQuery) -> str:
        """Translate a ScreenerQuery to approximate Bloomberg BQL syntax."""
        lines = ["// Bloomberg BQL equivalent", "LET(",
                 "  universe=MEMBERS('SPX Index')"]
        if query.sectors:
            for sec in query.sectors:
                lines.append(f"  // sector filter: GICS_SECTOR_NAME() == '{sec}'")
        filters_bql = []
        for f in query.filters:
            bql_metric = _metric_to_bql(f.metric)
            if f.operator == "between" and isinstance(f.value, tuple):
                filters_bql.append(
                    f"AND({bql_metric}>={f.value[0]}, {bql_metric}<={f.value[1]})")
            else:
                filters_bql.append(f"{bql_metric}{f.operator}{f.value}")
        if filters_bql:
            lines.append(f"  // filters: {', '.join(filters_bql)}")
        sort_m = _metric_to_bql(query.sort_by) if query.sort_by else "PE_RATIO()"
        direction = "DESC" if query.sort_desc else "ASC"
        lines.extend([
            ")",
            f"GET({sort_m})",
            f"WITH(universe=universe)",
            f"ORDER({sort_m} {direction})",
            f"LIMIT({query.limit})",
        ])
        return "\n".join(lines)

    def translate_to_sql(self, query: ScreenerQuery) -> str:
        """Translate a ScreenerQuery to SQL for educational purposes."""
        where_clauses = []
        if query.sectors:
            sector_list = ", ".join(f"'{s}'" for s in query.sectors)
            where_clauses.append(f"sector IN ({sector_list})")
        if query.market_cap_min:
            where_clauses.append(f"market_cap >= {query.market_cap_min:.0f}")
        if query.market_cap_max:
            where_clauses.append(f"market_cap <= {query.market_cap_max:.0f}")
        if query.requires_profitable:
            where_clauses.append("net_income > 0")
        if query.requires_dividend:
            where_clauses.append("dividend_yield > 0")
        for f in query.filters:
            col = f.metric
            if f.operator == "between" and isinstance(f.value, tuple):
                where_clauses.append(f"{col} BETWEEN {f.value[0]} AND {f.value[1]}")
            else:
                where_clauses.append(f"{col} {f.operator} {f.value}")

        sort_col = query.sort_by or "market_cap"
        direction = "DESC" if query.sort_desc else "ASC"
        where_str = "\n  AND ".join(where_clauses) if where_clauses else "1=1"
        return (
            f"SELECT ticker, company_name, sector, market_cap,\n"
            f"       pe_ratio, revenue_growth_yoy, net_margin, roe,\n"
            f"       debt_to_equity, dividend_yield\n"
            f"FROM fundamentals\n"
            f"WHERE {where_str}\n"
            f"ORDER BY {sort_col} {direction}\n"
            f"LIMIT {query.limit};"
        )


def _metric_to_bql(metric: Optional[str]) -> str:
    """Map metric code to Bloomberg BQL function."""
    mapping = {
        "pe_ratio": "PE_RATIO()", "pb_ratio": "PX_TO_BOOK_RATIO()",
        "ps_ratio": "PX_TO_SALES_RATIO()", "ev_ebitda": "EV_TO_T12M_EBITDA()",
        "dividend_yield": "DVD_YLD()", "roe": "RETURN_ON_EQY()",
        "roa": "RETURN_ON_ASSET()", "roic": "RETURN_ON_INV_CAPITAL()",
        "net_margin": "PROF_MARGIN()", "gross_margin": "GROSS_MARGIN()",
        "revenue_growth_yoy": "SALES_GROWTH()", "debt_to_equity": "TOT_DEBT_TO_EQY()",
        "market_cap": "CUR_MKT_CAP()", "momentum_12m": "TOT_RETURN_12M()",
        "peg_ratio": "PEG_RATIO()",
    }
    return mapping.get(metric or "", f"{metric}()" if metric else "MARKET_CAP()")


# ---------------------------------------------------------------------------
# NLScreenerEngine (orchestrator)
# ---------------------------------------------------------------------------

class NLScreenerEngine:
    """
    Main entry point for the natural language stock screener.
    Orchestrates: parse → enhance → fetch → filter → rank → explain.
    """

    def __init__(self):
        self.parser = QueryParser()
        self.enhancer = ClaudeQueryEnhancer()
        self.executor = NLScreenerExecutor()
        self.suggestions = QuerySuggestionEngine()
        _SCREENS_DIR.mkdir(parents=True, exist_ok=True)

    def screen(self, query: str,
               universe: List[str] = None) -> ScreenerResult:
        """
        Full pipeline: parse → optionally enhance → fetch → filter → sort → explain.
        """
        # Check for preset triggers
        preset_name = self._match_preset(query)
        if preset_name:
            logger.info("Matched preset: %s", preset_name)
            sq = _PRESETS[preset_name]
        else:
            sq = self.parser.parse(query)
            if self.enhancer.available:
                sq = self.enhancer.enhance_parse(query, sq)

        return self.executor.run(sq, universe)

    def _match_preset(self, query: str) -> Optional[str]:
        q = query.lower().strip()
        for preset_name in _PRESETS:
            if preset_name in q or q == preset_name:
                return preset_name
        return None

    def get_preset_list(self) -> List[str]:
        return sorted(_PRESETS.keys())

    def run_preset(self, name: str,
                   universe: List[str] = None) -> ScreenerResult:
        name_lower = name.lower().strip()
        if name_lower not in _PRESETS:
            # Fuzzy match
            for k in _PRESETS:
                if name_lower in k or k in name_lower:
                    name_lower = k
                    break
            else:
                raise ValueError(f"Unknown preset '{name}'. "
                                 f"Available: {self.get_preset_list()}")
        sq = _PRESETS[name_lower]
        return self.executor.run(sq, universe)

    def save_screen(self, query: str, name: str) -> None:
        """Persist a custom screen query by name."""
        path = _SCREENS_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump({"query": query, "saved_at": datetime.utcnow().isoformat()}, f)
        logger.info("Saved screen '%s' to %s", name, path)

    def load_screen(self, name: str) -> str:
        """Load a previously saved screen query."""
        path = _SCREENS_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No saved screen named '{name}'")
        with open(path) as f:
            data = json.load(f)
        return data["query"]

    def list_saved_screens(self) -> List[str]:
        """List all saved custom screens."""
        return [p.stem for p in _SCREENS_DIR.glob("*.json")]

    def explain_query(self, query: str) -> str:
        """Return a human-readable explanation of how a query is parsed."""
        return self.parser.explain_parse(query)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 72)
    print("SENTINEL Natural Language Screener V3 — dim_076 demo")
    print("=" * 72)

    engine = NLScreenerEngine()
    parser = QueryParser()
    suggestion_engine = QuerySuggestionEngine()

    EXAMPLE_QUERIES = [
        "profitable tech companies with P/E under 20",
        "dividend payers with yield over 3% and payout below 60%",
        "value stocks beaten down more than 30% with strong balance sheet",
        "growth stocks with revenue growth over 20%",
        "large cap quality companies with ROE over 15% and low debt",
    ]

    # Small universe for speed in demo
    DEMO_UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "JPM", "JNJ", "KO", "PEP", "XOM", "CVX", "PG", "WMT",
        "V", "MA", "BAC", "MRK", "ABBV", "AMD", "ORCL", "INTC",
        "NFLX", "DIS", "CSCO", "VZ", "T", "NKE", "MCD",
    ]

    print("\n[1] Query parser explanation examples:")
    print("-" * 60)
    for q in EXAMPLE_QUERIES[:3]:
        print(parser.explain_parse(q))
        print()

    print("[2] Autocomplete suggestions for 'profitable':")
    print("-" * 60)
    suggestions = suggestion_engine.suggest("profitable")
    for s in suggestions[:5]:
        print(f"  - {s}")

    print("\n[3] Running 5 example NL queries against demo universe...")
    print("    (fetching live data — may take 30-60s)")
    print("-" * 60)

    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        print(f"\n  Query {i}: \"{query}\"")
        try:
            result = engine.screen(query, universe=DEMO_UNIVERSE)
            print(f"  Matches: {len(result.matches)} | "
                  f"Universe: {result.total_universe} | "
                  f"Time: {result.execution_time_s:.1f}s")
            print(f"  Sector breakdown: {result.sector_breakdown}")
            df = result.to_dataframe()
            if not df.empty:
                cols = [c for c in ["ticker", "company", "sector", "market_cap",
                                    "pe_ratio", "net_margin", "roe", "dividend_yield",
                                    "momentum_12m"] if c in df.columns]
                pd.set_option("display.max_columns", 10)
                pd.set_option("display.width", 120)
                pd.set_option("display.float_format", "{:.2f}".format)
                print(df[cols].head(10).to_string(index=False))
        except Exception as exc:
            print(f"  Error: {exc}")

    print("\n[4] Bloomberg BQL translation example:")
    print("-" * 60)
    sq = parser.parse("profitable tech companies with P/E under 20 and ROE over 15%")
    print(suggestion_engine.translate_to_bloomberg(sq))

    print("\n[5] SQL translation example:")
    print("-" * 60)
    print(suggestion_engine.translate_to_sql(sq))

    print("\n[6] Available presets:")
    print("-" * 60)
    for p in engine.get_preset_list():
        print(f"  - {p}")

    print("\n[7] Running 'deep value' preset...")
    print("-" * 60)
    try:
        preset_result = engine.run_preset("deep value", universe=DEMO_UNIVERSE)
        print(f"  Deep value matches: {len(preset_result.matches)}")
        print(engine.executor.explain_results(preset_result))
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\nDone.")
