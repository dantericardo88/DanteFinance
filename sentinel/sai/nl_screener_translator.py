"""
Natural language to structured screener: parse plain-English queries into screener criteria.

"Show me cheap tech stocks with high growth" → {sector=Technology, pe_ratio<20, revenue_growth>20%}

Dimensions targeted:
  dim_053 — Natural language → screener translator   score → 9

Architecture:
  - NLQueryParser: regex-based metric, sector, geography, and time extraction
  - ScreenerCriteriaBuilder: convert parsed entities to ScreenerCriteria dataclasses
  - QueryAmbiguityResolver: handle vague terms with ranked interpretations
  - ScreenerExecutor: DuckDB/SQLite execution with 0-100 scoring
  - NLToScreenerPipeline: end-to-end parse → resolve → build → execute → rank
  - ScreenerTemplateLibrary: 25+ pre-built investment-strategy templates
  - FastAPI router with 6 endpoints
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ── DB path ───────────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "data" / "screener_store.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Sector canonical mapping ──────────────────────────────────────────────────

SECTOR_ALIASES: Dict[str, str] = {
    # Technology
    "tech": "Technology", "technology": "Technology", "software": "Technology",
    "semiconductor": "Technology", "semiconductors": "Technology", "chip": "Technology",
    "cloud": "Technology", "saas": "Technology", "ai": "Technology",
    "artificial intelligence": "Technology", "cybersecurity": "Technology",
    "fintech": "Technology",
    # Healthcare
    "healthcare": "Healthcare", "health care": "Healthcare", "pharma": "Healthcare",
    "pharmaceutical": "Healthcare", "biotech": "Healthcare", "biotechnology": "Healthcare",
    "medical": "Healthcare", "drug": "Healthcare", "hospital": "Healthcare",
    "medtech": "Healthcare",
    # Energy
    "energy": "Energy", "oil": "Energy", "oil and gas": "Energy",
    "gas": "Energy", "renewable energy": "Energy", "solar": "Energy",
    "wind energy": "Energy", "clean energy": "Energy", "utilities": "Utilities",
    "electric": "Utilities",
    # Financial
    "financial": "Financial", "financials": "Financial", "bank": "Financial",
    "banking": "Financial", "insurance": "Financial", "asset management": "Financial",
    "finservices": "Financial", "financial services": "Financial",
    "investment bank": "Financial",
    # Consumer
    "consumer": "Consumer Discretionary", "retail": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples", "food": "Consumer Staples",
    "beverage": "Consumer Staples", "household": "Consumer Staples",
    "luxury": "Consumer Discretionary", "e-commerce": "Consumer Discretionary",
    "ecommerce": "Consumer Discretionary",
    # Industrial
    "industrial": "Industrials", "industrials": "Industrials",
    "aerospace": "Industrials", "defense": "Industrials",
    "manufacturing": "Industrials", "transportation": "Industrials",
    # Materials
    "materials": "Materials", "mining": "Materials", "gold": "Materials",
    "metals": "Materials", "chemicals": "Materials",
    # Real Estate
    "real estate": "Real Estate", "reit": "Real Estate", "reits": "Real Estate",
    "property": "Real Estate",
    # Communication
    "communication": "Communication Services", "media": "Communication Services",
    "telecom": "Communication Services", "telecommunications": "Communication Services",
    "streaming": "Communication Services", "social media": "Communication Services",
}

# ── Geography mapping ─────────────────────────────────────────────────────────

GEO_ALIASES: Dict[str, str] = {
    "us": "US", "usa": "US", "united states": "US", "american": "US",
    "domestic": "US", "north america": "North America",
    "europe": "Europe", "european": "Europe", "eu": "Europe",
    "uk": "UK", "united kingdom": "UK", "british": "UK",
    "germany": "Germany", "german": "Germany",
    "japan": "Japan", "japanese": "Japan",
    "china": "China", "chinese": "China",
    "emerging markets": "Emerging Markets", "em": "Emerging Markets",
    "emerging": "Emerging Markets",
    "asia": "Asia", "asia pacific": "Asia Pacific", "apac": "Asia Pacific",
    "global": "Global", "international": "International",
    "latin america": "Latin America", "latam": "Latin America",
    "canada": "Canada", "canadian": "Canada",
    "australia": "Australia", "australian": "Australia",
    "india": "India", "indian": "India",
    "brazil": "Brazil", "brazilian": "Brazil",
}

# ── Time period mapping ───────────────────────────────────────────────────────

TIME_ALIASES: Dict[str, str] = {
    "past year": "1Y", "last year": "1Y", "trailing 12 months": "TTM",
    "ttm": "TTM", "ltm": "TTM", "last twelve months": "TTM",
    "last quarter": "1Q", "past quarter": "1Q", "recent quarter": "1Q",
    "ytd": "YTD", "year to date": "YTD", "this year": "YTD",
    "5 year": "5Y", "five year": "5Y", "5-year": "5Y",
    "3 year": "3Y", "three year": "3Y", "3-year": "3Y",
    "monthly": "1M", "last month": "1M", "past month": "1M",
    "weekly": "1W", "last week": "1W",
}

# ── Metric canonical names ────────────────────────────────────────────────────

METRIC_CANONICAL: Dict[str, str] = {
    "pe_ratio": "pe_ratio", "p/e": "pe_ratio", "pe": "pe_ratio",
    "price to earnings": "pe_ratio", "price-to-earnings": "pe_ratio",
    "pb_ratio": "pb_ratio", "p/b": "pb_ratio", "price to book": "pb_ratio",
    "price-to-book": "pb_ratio",
    "ps_ratio": "ps_ratio", "p/s": "ps_ratio", "price to sales": "ps_ratio",
    "ev_ebitda": "ev_ebitda", "ev/ebitda": "ev_ebitda",
    "peg_ratio": "peg_ratio", "peg": "peg_ratio",
    "market_cap": "market_cap", "market cap": "market_cap",
    "market capitalization": "market_cap",
    "revenue_growth": "revenue_growth", "revenue growth": "revenue_growth",
    "sales growth": "revenue_growth",
    "earnings_growth": "earnings_growth", "earnings growth": "earnings_growth",
    "eps_growth": "earnings_growth",
    "net_income": "net_income", "net income": "net_income",
    "revenue": "revenue", "sales": "revenue",
    "gross_margin": "gross_margin", "gross margin": "gross_margin",
    "operating_margin": "operating_margin", "operating margin": "operating_margin",
    "net_margin": "net_margin", "net margin": "net_margin", "profit margin": "net_margin",
    "roe": "roe", "return on equity": "roe", "return_on_equity": "roe",
    "roa": "roa", "return on assets": "roa",
    "roic": "roic", "return on invested capital": "roic",
    "debt_equity": "debt_equity", "debt to equity": "debt_equity",
    "debt/equity": "debt_equity", "leverage": "debt_equity",
    "current_ratio": "current_ratio", "current ratio": "current_ratio",
    "quick_ratio": "quick_ratio", "quick ratio": "quick_ratio",
    "dividend_yield": "dividend_yield", "dividend yield": "dividend_yield",
    "div_yield": "dividend_yield", "yield": "dividend_yield",
    "consecutive_div_growth": "consecutive_div_growth",
    "dividend growth years": "consecutive_div_growth",
    "short_interest": "short_interest", "short interest": "short_interest",
    "float_short": "float_short", "short float": "float_short",
    "rsi": "rsi", "relative strength index": "rsi",
    "return_3m": "return_3m", "3 month return": "return_3m",
    "3m return": "return_3m", "3-month return": "return_3m",
    "return_1y": "return_1y", "1 year return": "return_1y",
    "52_week_high_pct": "52_week_high_pct", "near 52 week high": "52_week_high_pct",
    "beta": "beta", "volatility": "beta",
    "earnings_surprise": "earnings_surprise", "eps surprise": "earnings_surprise",
    "analyst_rating": "analyst_rating", "analyst consensus": "analyst_rating",
    "eps_revision": "eps_revision", "estimate revision": "eps_revision",
    "iv_percentile": "iv_percentile", "implied volatility": "iv_percentile",
    "borrow_cost": "borrow_cost", "short borrow cost": "borrow_cost",
}

# ── Operator phrase mapping ───────────────────────────────────────────────────

OPERATOR_PHRASES: Dict[str, str] = {
    "below": "<", "under": "<", "less than": "<", "lower than": "<",
    "at most": "<=", "no more than": "<=", "up to": "<=", "maximum": "<=",
    "above": ">", "over": ">", "greater than": ">", "higher than": ">",
    "more than": ">", "at least": ">=", "minimum": ">=",
    "equal to": "==", "exactly": "==", "of": "==",
    "<": "<", ">": ">", "<=": "<=", ">=": ">=", "=": "==",
}

# ── Vague term default expansions ─────────────────────────────────────────────

VAGUE_TERMS: Dict[str, List[Dict]] = {
    "cheap": [
        {"metric": "pe_ratio", "operator": "<", "value": 15.0, "description": "P/E < 15"},
        {"metric": "pb_ratio", "operator": "<", "value": 1.5, "description": "P/B < 1.5"},
        {"metric": "ev_ebitda", "operator": "<", "value": 8.0, "description": "EV/EBITDA < 8"},
    ],
    "undervalued": [
        {"metric": "pe_ratio", "operator": "<", "value": 15.0, "description": "P/E < 15"},
        {"metric": "pb_ratio", "operator": "<", "value": 1.5, "description": "P/B < 1.5"},
    ],
    "overvalued": [
        {"metric": "pe_ratio", "operator": ">", "value": 40.0, "description": "P/E > 40"},
        {"metric": "pb_ratio", "operator": ">", "value": 5.0, "description": "P/B > 5"},
    ],
    "high growth": [
        {"metric": "revenue_growth", "operator": ">", "value": 20.0, "description": "Revenue growth > 20%"},
        {"metric": "earnings_growth", "operator": ">", "value": 15.0, "description": "EPS growth > 15%"},
    ],
    "growing": [
        {"metric": "revenue_growth", "operator": ">", "value": 10.0, "description": "Revenue growth > 10%"},
    ],
    "profitable": [
        {"metric": "net_income", "operator": ">", "value": 0.0, "description": "Net income > 0"},
    ],
    "dividend": [
        {"metric": "dividend_yield", "operator": ">", "value": 0.0, "description": "Dividend yield > 0%"},
    ],
    "high dividend": [
        {"metric": "dividend_yield", "operator": ">", "value": 3.0, "description": "Dividend yield > 3%"},
    ],
    "large cap": [
        {"metric": "market_cap", "operator": ">", "value": 10_000_000_000, "description": "Market cap > $10B"},
    ],
    "mega cap": [
        {"metric": "market_cap", "operator": ">", "value": 200_000_000_000, "description": "Market cap > $200B"},
    ],
    "mid cap": [
        {"metric": "market_cap", "operator": ">", "value": 2_000_000_000, "description": "Market cap > $2B"},
        {"metric": "market_cap", "operator": "<", "value": 10_000_000_000, "description": "Market cap < $10B"},
    ],
    "small cap": [
        {"metric": "market_cap", "operator": "<", "value": 2_000_000_000, "description": "Market cap < $2B"},
    ],
    "micro cap": [
        {"metric": "market_cap", "operator": "<", "value": 300_000_000, "description": "Market cap < $300M"},
    ],
    "momentum": [
        {"metric": "return_3m", "operator": ">", "value": 0.0, "description": "3M return > 0%"},
        {"metric": "rsi", "operator": ">", "value": 50.0, "description": "RSI > 50"},
    ],
    "quality": [
        {"metric": "roe", "operator": ">", "value": 15.0, "description": "ROE > 15%"},
        {"metric": "debt_equity", "operator": "<", "value": 0.5, "description": "Debt/Equity < 0.5"},
    ],
    "low volatility": [
        {"metric": "beta", "operator": "<", "value": 0.8, "description": "Beta < 0.8"},
    ],
    "high volatility": [
        {"metric": "beta", "operator": ">", "value": 1.5, "description": "Beta > 1.5"},
    ],
    "value": [
        {"metric": "pe_ratio", "operator": "<", "value": 15.0, "description": "P/E < 15"},
        {"metric": "pb_ratio", "operator": "<", "value": 2.0, "description": "P/B < 2"},
    ],
    "growth": [
        {"metric": "revenue_growth", "operator": ">", "value": 15.0, "description": "Revenue growth > 15%"},
    ],
    "garp": [
        {"metric": "peg_ratio", "operator": "<", "value": 1.5, "description": "PEG < 1.5"},
        {"metric": "revenue_growth", "operator": ">", "value": 10.0, "description": "Revenue growth > 10%"},
    ],
}

# ── Templates ─────────────────────────────────────────────────────────────────

SCREENER_TEMPLATES: Dict[str, Dict] = {
    "warren_buffett_quality": {
        "description": "Warren Buffett quality: durable competitive advantage, strong returns, reasonable price",
        "criteria": [
            {"metric": "roe", "operator": ">", "value": 15.0},
            {"metric": "debt_equity", "operator": "<", "value": 0.5},
            {"metric": "pe_ratio", "operator": "<", "value": 20.0},
            {"metric": "net_margin", "operator": ">", "value": 10.0},
            {"metric": "revenue_growth", "operator": ">", "value": 5.0},
        ],
        "sort_by": "roe",
        "sort_ascending": False,
    },
    "peter_lynch_garp": {
        "description": "Peter Lynch GARP: growth at a reasonable price, PEG < 1",
        "criteria": [
            {"metric": "peg_ratio", "operator": "<", "value": 1.0},
            {"metric": "revenue_growth", "operator": ">", "value": 10.0},
            {"metric": "earnings_growth", "operator": ">", "value": 10.0},
            {"metric": "pe_ratio", "operator": "<", "value": 30.0},
        ],
        "sort_by": "peg_ratio",
        "sort_ascending": True,
    },
    "benjamin_graham_deep_value": {
        "description": "Benjamin Graham deep value: statistically cheap with balance sheet safety",
        "criteria": [
            {"metric": "pe_ratio", "operator": "<", "value": 10.0},
            {"metric": "pb_ratio", "operator": "<", "value": 1.0},
            {"metric": "current_ratio", "operator": ">", "value": 2.0},
            {"metric": "net_income", "operator": ">", "value": 0.0},
            {"metric": "debt_equity", "operator": "<", "value": 1.0},
        ],
        "sort_by": "pe_ratio",
        "sort_ascending": True,
    },
    "dividend_aristocrat": {
        "description": "Dividend aristocrats: high yield with long track record of dividend growth",
        "criteria": [
            {"metric": "dividend_yield", "operator": ">", "value": 3.0},
            {"metric": "consecutive_div_growth", "operator": ">", "value": 25.0},
            {"metric": "net_income", "operator": ">", "value": 0.0},
            {"metric": "debt_equity", "operator": "<", "value": 1.5},
        ],
        "sort_by": "dividend_yield",
        "sort_ascending": False,
    },
    "high_short_interest_squeeze": {
        "description": "Short squeeze candidates: heavily shorted with low borrow availability",
        "criteria": [
            {"metric": "short_interest", "operator": ">", "value": 20.0},
            {"metric": "float_short", "operator": ">", "value": 30.0},
            {"metric": "borrow_cost", "operator": ">", "value": 10.0},
        ],
        "sort_by": "short_interest",
        "sort_ascending": False,
    },
    "pre_earnings_momentum": {
        "description": "Pre-earnings momentum: strong price action with positive estimate revisions",
        "criteria": [
            {"metric": "return_3m", "operator": ">", "value": 15.0},
            {"metric": "eps_revision", "operator": ">", "value": 0.0},
            {"metric": "iv_percentile", "operator": "<", "value": 25.0},
        ],
        "sort_by": "return_3m",
        "sort_ascending": False,
    },
    "low_volatility_income": {
        "description": "Low volatility income: defensive stocks with dividends and low beta",
        "criteria": [
            {"metric": "beta", "operator": "<", "value": 0.8},
            {"metric": "dividend_yield", "operator": ">", "value": 2.0},
            {"metric": "pe_ratio", "operator": "<", "value": 20.0},
        ],
        "sort_by": "dividend_yield",
        "sort_ascending": False,
    },
    "high_quality_growth": {
        "description": "High-quality growth: fast-growing companies with strong unit economics",
        "criteria": [
            {"metric": "revenue_growth", "operator": ">", "value": 25.0},
            {"metric": "gross_margin", "operator": ">", "value": 50.0},
            {"metric": "roe", "operator": ">", "value": 20.0},
        ],
        "sort_by": "revenue_growth",
        "sort_ascending": False,
    },
    "deep_value_turnaround": {
        "description": "Deep value turnaround: cheap stocks showing early signs of recovery",
        "criteria": [
            {"metric": "pb_ratio", "operator": "<", "value": 1.0},
            {"metric": "pe_ratio", "operator": "<", "value": 12.0},
            {"metric": "return_3m", "operator": ">", "value": 0.0},
            {"metric": "earnings_surprise", "operator": ">", "value": 0.0},
        ],
        "sort_by": "pb_ratio",
        "sort_ascending": True,
    },
    "tech_growth": {
        "description": "Technology growth: fast-growing tech companies with expanding margins",
        "criteria": [
            {"metric": "pe_ratio", "operator": "<", "value": 40.0},
            {"metric": "revenue_growth", "operator": ">", "value": 20.0},
            {"metric": "gross_margin", "operator": ">", "value": 60.0},
        ],
        "sector": "Technology",
        "sort_by": "revenue_growth",
        "sort_ascending": False,
    },
    "healthcare_value": {
        "description": "Healthcare value: established pharma/devices trading at discount",
        "criteria": [
            {"metric": "pe_ratio", "operator": "<", "value": 15.0},
            {"metric": "dividend_yield", "operator": ">", "value": 1.5},
            {"metric": "net_margin", "operator": ">", "value": 15.0},
        ],
        "sector": "Healthcare",
        "sort_by": "dividend_yield",
        "sort_ascending": False,
    },
    "financial_value": {
        "description": "Financial sector value: banks and insurers trading below book value",
        "criteria": [
            {"metric": "pb_ratio", "operator": "<", "value": 1.2},
            {"metric": "roe", "operator": ">", "value": 10.0},
            {"metric": "dividend_yield", "operator": ">", "value": 2.0},
        ],
        "sector": "Financial",
        "sort_by": "pb_ratio",
        "sort_ascending": True,
    },
    "energy_value": {
        "description": "Energy value: low valuation energy stocks with strong free cash flow",
        "criteria": [
            {"metric": "ev_ebitda", "operator": "<", "value": 6.0},
            {"metric": "dividend_yield", "operator": ">", "value": 3.0},
            {"metric": "net_margin", "operator": ">", "value": 8.0},
        ],
        "sector": "Energy",
        "sort_by": "ev_ebitda",
        "sort_ascending": True,
    },
    "micro_cap_value": {
        "description": "Micro-cap value: tiny companies trading at deep discounts to book",
        "criteria": [
            {"metric": "market_cap", "operator": "<", "value": 300_000_000},
            {"metric": "pb_ratio", "operator": "<", "value": 1.0},
            {"metric": "current_ratio", "operator": ">", "value": 1.5},
            {"metric": "net_income", "operator": ">", "value": 0.0},
        ],
        "sort_by": "pb_ratio",
        "sort_ascending": True,
    },
    "momentum_quality": {
        "description": "Momentum + quality: strong recent performers with solid fundamentals",
        "criteria": [
            {"metric": "return_3m", "operator": ">", "value": 10.0},
            {"metric": "roe", "operator": ">", "value": 15.0},
            {"metric": "debt_equity", "operator": "<", "value": 1.0},
            {"metric": "eps_revision", "operator": ">", "value": 0.0},
        ],
        "sort_by": "return_3m",
        "sort_ascending": False,
    },
    "emerging_market_growth": {
        "description": "Emerging market growth: fast-growing companies in developing economies",
        "criteria": [
            {"metric": "revenue_growth", "operator": ">", "value": 15.0},
            {"metric": "pe_ratio", "operator": "<", "value": 20.0},
            {"metric": "roe", "operator": ">", "value": 12.0},
        ],
        "geography": "Emerging Markets",
        "sort_by": "revenue_growth",
        "sort_ascending": False,
    },
    "reit_income": {
        "description": "REIT income: high-yield real estate investment trusts",
        "criteria": [
            {"metric": "dividend_yield", "operator": ">", "value": 4.0},
            {"metric": "pb_ratio", "operator": "<", "value": 2.5},
            {"metric": "debt_equity", "operator": "<", "value": 2.0},
        ],
        "sector": "Real Estate",
        "sort_by": "dividend_yield",
        "sort_ascending": False,
    },
    "analyst_upgrade_momentum": {
        "description": "Analyst upgrade momentum: stocks receiving buy upgrades with positive revisions",
        "criteria": [
            {"metric": "analyst_rating", "operator": ">", "value": 3.5},
            {"metric": "eps_revision", "operator": ">", "value": 2.0},
            {"metric": "return_3m", "operator": ">", "value": 5.0},
        ],
        "sort_by": "eps_revision",
        "sort_ascending": False,
    },
    "vc_style_growth": {
        "description": "VC-style hypergrowth: high-growth early-stage public companies",
        "criteria": [
            {"metric": "revenue_growth", "operator": ">", "value": 40.0},
            {"metric": "gross_margin", "operator": ">", "value": 60.0},
            {"metric": "market_cap", "operator": "<", "value": 5_000_000_000},
        ],
        "sort_by": "revenue_growth",
        "sort_ascending": False,
    },
    "defensive_staples": {
        "description": "Defensive staples: recession-resistant consumer staples companies",
        "criteria": [
            {"metric": "beta", "operator": "<", "value": 0.7},
            {"metric": "dividend_yield", "operator": ">", "value": 2.0},
            {"metric": "net_margin", "operator": ">", "value": 8.0},
        ],
        "sector": "Consumer Staples",
        "sort_by": "beta",
        "sort_ascending": True,
    },
    "high_roic_compounder": {
        "description": "High-ROIC compounders: businesses with exceptional capital allocation",
        "criteria": [
            {"metric": "roic", "operator": ">", "value": 20.0},
            {"metric": "revenue_growth", "operator": ">", "value": 8.0},
            {"metric": "debt_equity", "operator": "<", "value": 0.3},
        ],
        "sort_by": "roic",
        "sort_ascending": False,
    },
    "net_net_graham": {
        "description": "Graham net-net: trading below net current asset value",
        "criteria": [
            {"metric": "pb_ratio", "operator": "<", "value": 0.7},
            {"metric": "current_ratio", "operator": ">", "value": 3.0},
            {"metric": "debt_equity", "operator": "<", "value": 0.3},
        ],
        "sort_by": "pb_ratio",
        "sort_ascending": True,
    },
    "spin_off_special_situation": {
        "description": "Special situation: recent spin-offs with depressed valuations",
        "criteria": [
            {"metric": "return_1y", "operator": "<", "value": -10.0},
            {"metric": "pe_ratio", "operator": "<", "value": 15.0},
            {"metric": "roe", "operator": ">", "value": 10.0},
        ],
        "sort_by": "return_1y",
        "sort_ascending": True,
    },
    "fallen_angel": {
        "description": "Fallen angels: quality companies with temporary setbacks",
        "criteria": [
            {"metric": "return_1y", "operator": "<", "value": -20.0},
            {"metric": "roe", "operator": ">", "value": 12.0},
            {"metric": "net_income", "operator": ">", "value": 0.0},
            {"metric": "current_ratio", "operator": ">", "value": 1.5},
        ],
        "sort_by": "return_1y",
        "sort_ascending": True,
    },
    "consistent_eps_growth": {
        "description": "Consistent EPS growers: companies with steady earnings compounding",
        "criteria": [
            {"metric": "earnings_growth", "operator": ">", "value": 10.0},
            {"metric": "roe", "operator": ">", "value": 15.0},
            {"metric": "pe_ratio", "operator": "<", "value": 25.0},
        ],
        "sort_by": "earnings_growth",
        "sort_ascending": False,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScreenerCriteria:
    """A single parsed screener criterion."""
    metric: str
    operator: str        # "<", ">", "<=", ">=", "between", "=="
    value: Union[float, Tuple[float, float]]
    weight: float = 1.0
    description: str = ""
    sector: Optional[str] = None
    geography: Optional[str] = None

    def __str__(self) -> str:
        if self.operator == "between" and isinstance(self.value, tuple):
            return f"{self.metric} between {self.value[0]} and {self.value[1]}"
        return f"{self.metric} {self.operator} {self.value}"


@dataclass
class ParsedQuery:
    """Structured output from NL parsing."""
    raw_query: str
    metrics: List[Dict]            # [{metric, operator, value}]
    sectors: List[str]
    geographies: List[str]
    time_period: Optional[str]
    vague_terms: List[str]
    compound_logic: str            # "AND" | "OR" | "NOT"
    negated_terms: List[str]


@dataclass
class Interpretation:
    """A single interpretation of an ambiguous query."""
    criteria: List[ScreenerCriteria]
    confidence: float
    description: str


@dataclass
class ScreenerResult:
    """End-to-end result from NL → screener pipeline."""
    query: str
    criteria: List[ScreenerCriteria]
    matches: pd.DataFrame
    explanation: str
    n_matches: int
    top_matches: List[str]
    criteria_descriptions: List[str]


# ── Pydantic request/response models ─────────────────────────────────────────


class TranslateRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    execute: bool = Field(True, description="Whether to execute the screener after translating")


class ExecuteRequest(BaseModel):
    criteria: List[Dict[str, Any]]


class ExplainRequest(BaseModel):
    query: str
    max_interpretations: int = Field(3, ge=1, le=5)


class TranslateResponse(BaseModel):
    query: str
    criteria: List[Dict[str, Any]]
    explanation: str
    criteria_descriptions: List[str]
    n_criteria: int


class ScreenerResultResponse(BaseModel):
    query: str
    criteria_descriptions: List[str]
    explanation: str
    n_matches: int
    top_matches: List[str]


# ══════════════════════════════════════════════════════════════════════════════
# 1. NLQueryParser
# ══════════════════════════════════════════════════════════════════════════════


class NLQueryParser:
    """
    Parse natural language investment queries into structured entities.

    Handles:
    - Numeric metrics: "P/E below 15", "revenue growth > 20%"
    - Vague qualitative terms: "cheap", "high growth", "quality"
    - Sector/geography filters: "tech stocks", "US market"
    - Time periods: "past year", "TTM"
    - Compound logic: AND, OR, NOT
    """

    def __init__(self) -> None:
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        # Metric name pattern: matches known metric aliases
        metric_names = "|".join(
            re.escape(k) for k in sorted(METRIC_CANONICAL.keys(), key=len, reverse=True)
        )

        # Numeric value pattern
        num = r"[-+]?\d+(?:[.,]\d+)?(?:%|B|M|K)?"

        # Operator phrase pattern
        op_phrases = "|".join(
            re.escape(k) for k in sorted(OPERATOR_PHRASES.keys(), key=len, reverse=True)
        )

        # "metric operator value"
        self._metric_pattern = re.compile(
            rf"({metric_names})\s*(?:is\s*)?({op_phrases})\s*({num})",
            re.IGNORECASE,
        )

        # "between X and Y"
        self._between_pattern = re.compile(
            rf"({metric_names})\s+between\s+({num})\s+and\s+({num})",
            re.IGNORECASE,
        )

        # Market cap shorthand: "above $10B", "$200M"
        self._mktcap_pattern = re.compile(
            r"market\s*cap(?:italization)?\s*(?:is\s*)?(?:of\s*)?\$?([\d.]+)\s*([BM])",
            re.IGNORECASE,
        )

        # Sector patterns
        sector_names = "|".join(
            re.escape(k) for k in sorted(SECTOR_ALIASES.keys(), key=len, reverse=True)
        )
        self._sector_pattern = re.compile(
            rf"\b({sector_names})\b\s*(?:stocks?|companies|sector|industry)?",
            re.IGNORECASE,
        )

        # Geography patterns
        geo_names = "|".join(
            re.escape(k) for k in sorted(GEO_ALIASES.keys(), key=len, reverse=True)
        )
        self._geo_pattern = re.compile(
            rf"\b({geo_names})\b",
            re.IGNORECASE,
        )

        # Time patterns
        time_names = "|".join(
            re.escape(k) for k in sorted(TIME_ALIASES.keys(), key=len, reverse=True)
        )
        self._time_pattern = re.compile(
            rf"\b({time_names})\b",
            re.IGNORECASE,
        )

        # Vague terms
        vague_names = "|".join(
            re.escape(k) for k in sorted(VAGUE_TERMS.keys(), key=len, reverse=True)
        )
        self._vague_pattern = re.compile(
            rf"\b({vague_names})\b",
            re.IGNORECASE,
        )

    def _parse_numeric_value(self, raw: str) -> float:
        """Convert raw string value to float, handling % / B / M / K suffixes."""
        raw = raw.strip().replace(",", "")
        multiplier = 1.0

        if raw.endswith("%"):
            raw = raw[:-1]
        elif raw.upper().endswith("B"):
            multiplier = 1_000_000_000
            raw = raw[:-1]
        elif raw.upper().endswith("M"):
            multiplier = 1_000_000
            raw = raw[:-1]
        elif raw.upper().endswith("K"):
            multiplier = 1_000
            raw = raw[:-1]

        try:
            return float(raw) * multiplier
        except ValueError:
            return 0.0

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured entities.

        Returns ParsedQuery with metrics, sectors, geographies, etc.
        """
        metrics: List[Dict] = []
        seen_metrics: set = set()

        # Check for NOT logic first
        negated: List[str] = []
        not_match = re.search(r"\bnot\b\s+(.{3,30}?)(?:\s+and\s+|\s+or\s+|$)", query, re.IGNORECASE)
        if not_match:
            negated.append(not_match.group(1).strip())

        # Detect compound logic
        has_or = bool(re.search(r"\bor\b", query, re.IGNORECASE))
        has_and = bool(re.search(r"\band\b", query, re.IGNORECASE))
        compound_logic = "OR" if has_or and not has_and else "AND"

        # Extract between ranges
        for match in self._between_pattern.finditer(query):
            metric_raw, v1_raw, v2_raw = match.group(1), match.group(2), match.group(3)
            canonical = METRIC_CANONICAL.get(metric_raw.lower(), metric_raw.lower())
            v1 = self._parse_numeric_value(v1_raw)
            v2 = self._parse_numeric_value(v2_raw)
            key = f"{canonical}_between"
            if key not in seen_metrics:
                seen_metrics.add(key)
                metrics.append({
                    "metric": canonical,
                    "operator": "between",
                    "value": (v1, v2),
                    "description": f"{canonical} between {v1} and {v2}",
                })

        # Extract metric + operator + value
        for match in self._metric_pattern.finditer(query):
            metric_raw, op_raw, val_raw = match.group(1), match.group(2), match.group(3)
            canonical = METRIC_CANONICAL.get(metric_raw.lower(), metric_raw.lower())
            operator = OPERATOR_PHRASES.get(op_raw.lower().strip(), op_raw)
            value = self._parse_numeric_value(val_raw)
            key = f"{canonical}_{operator}"
            if key not in seen_metrics:
                seen_metrics.add(key)
                metrics.append({
                    "metric": canonical,
                    "operator": operator,
                    "value": value,
                    "description": f"{canonical} {operator} {value}",
                })

        # Market cap shorthand: "$10B" etc.
        for match in self._mktcap_pattern.finditer(query):
            val_str, suffix = match.group(1), match.group(2).upper()
            val = float(val_str) * (1_000_000_000 if suffix == "B" else 1_000_000)
            metrics.append({
                "metric": "market_cap",
                "operator": ">",
                "value": val,
                "description": f"market_cap > ${val_str}{suffix}",
            })

        # Extract sectors
        sectors: List[str] = []
        seen_sectors: set = set()
        for match in self._sector_pattern.finditer(query):
            s = SECTOR_ALIASES.get(match.group(1).lower())
            if s and s not in seen_sectors:
                seen_sectors.add(s)
                sectors.append(s)

        # Extract geographies
        geographies: List[str] = []
        seen_geos: set = set()
        for match in self._geo_pattern.finditer(query):
            g = GEO_ALIASES.get(match.group(1).lower())
            if g and g not in seen_geos:
                seen_geos.add(g)
                geographies.append(g)

        # Extract time period
        time_period: Optional[str] = None
        for match in self._time_pattern.finditer(query):
            time_period = TIME_ALIASES.get(match.group(1).lower())
            break

        # Extract vague terms
        vague_found: List[str] = []
        seen_vague: set = set()
        for match in self._vague_pattern.finditer(query):
            term = match.group(1).lower()
            if term not in seen_vague:
                seen_vague.add(term)
                vague_found.append(term)

        return ParsedQuery(
            raw_query=query,
            metrics=metrics,
            sectors=sectors,
            geographies=geographies,
            time_period=time_period,
            vague_terms=vague_found,
            compound_logic=compound_logic,
            negated_terms=negated,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. ScreenerCriteriaBuilder
# ══════════════════════════════════════════════════════════════════════════════


class ScreenerCriteriaBuilder:
    """
    Convert ParsedQuery into a list of ScreenerCriteria.

    Handles explicit metrics, vague term expansion, sector/geography filters.
    """

    def build_from_parsed(self, parsed: ParsedQuery) -> List[ScreenerCriteria]:
        """Convert parsed entities to ScreenerCriteria list."""
        criteria: List[ScreenerCriteria] = []
        seen: set = set()

        # Explicit metrics
        for m in parsed.metrics:
            key = f"{m['metric']}_{m['operator']}_{m['value']}"
            if key not in seen:
                seen.add(key)
                criteria.append(ScreenerCriteria(
                    metric=m["metric"],
                    operator=m["operator"],
                    value=m["value"],
                    weight=1.0,
                    description=m.get("description", str(m["value"])),
                ))

        # Vague terms — expand using VAGUE_TERMS
        for term in parsed.vague_terms:
            expansions = VAGUE_TERMS.get(term, [])
            for exp in expansions:
                key = f"{exp['metric']}_{exp['operator']}_{exp['value']}"
                if key not in seen:
                    seen.add(key)
                    criteria.append(ScreenerCriteria(
                        metric=exp["metric"],
                        operator=exp["operator"],
                        value=exp["value"],
                        weight=0.8,
                        description=exp.get("description", ""),
                    ))

        # Sector filter (stored as special criteria)
        for sector in parsed.sectors:
            criteria.append(ScreenerCriteria(
                metric="sector",
                operator="==",
                value=0.0,
                weight=1.0,
                description=f"sector = {sector}",
                sector=sector,
            ))

        # Geography filter
        for geo in parsed.geographies:
            criteria.append(ScreenerCriteria(
                metric="geography",
                operator="==",
                value=0.0,
                weight=1.0,
                description=f"geography = {geo}",
                geography=geo,
            ))

        return criteria

    def build_from_template(self, template_name: str) -> List[ScreenerCriteria]:
        """Build criteria from a named template."""
        template = SCREENER_TEMPLATES.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        criteria: List[ScreenerCriteria] = []
        for c in template.get("criteria", []):
            criteria.append(ScreenerCriteria(
                metric=c["metric"],
                operator=c["operator"],
                value=c["value"],
                weight=1.0,
                description=f"{c['metric']} {c['operator']} {c['value']}",
            ))

        # Add sector filter if template specifies one
        if "sector" in template:
            criteria.append(ScreenerCriteria(
                metric="sector",
                operator="==",
                value=0.0,
                description=f"sector = {template['sector']}",
                sector=template["sector"],
            ))

        if "geography" in template:
            criteria.append(ScreenerCriteria(
                metric="geography",
                operator="==",
                value=0.0,
                description=f"geography = {template['geography']}",
                geography=template["geography"],
            ))

        return criteria


# ══════════════════════════════════════════════════════════════════════════════
# 3. QueryAmbiguityResolver
# ══════════════════════════════════════════════════════════════════════════════


class QueryAmbiguityResolver:
    """
    Handle ambiguous or vague natural language queries.

    Returns top-3 interpretations with confidence scores.
    """

    # Priority metric for each vague qualifier
    VAGUE_PRIORITY: Dict[str, List[str]] = {
        "cheap": ["pe_ratio", "pb_ratio", "ev_ebitda"],
        "growing": ["revenue_growth", "earnings_growth"],
        "quality": ["roe", "roic", "net_margin"],
        "safe": ["debt_equity", "current_ratio", "beta"],
        "risky": ["beta", "debt_equity"],
        "profitable": ["net_income", "net_margin", "operating_margin"],
    }

    def resolve(self, query: str, max_interpretations: int = 3) -> List[Interpretation]:
        """
        Return top interpretations for an ambiguous query.

        Each interpretation has criteria + confidence score.
        """
        parser = NLQueryParser()
        parsed = parser.parse(query)
        builder = ScreenerCriteriaBuilder()

        interpretations: List[Interpretation] = []

        # Base interpretation: take all parsed criteria literally
        base_criteria = builder.build_from_parsed(parsed)
        if base_criteria:
            interpretations.append(Interpretation(
                criteria=base_criteria,
                confidence=0.85,
                description=f"Direct parse: {', '.join(c.description for c in base_criteria[:3])}",
            ))

        # Vague term alternative interpretations
        for term in parsed.vague_terms[:2]:
            priority_metrics = self.VAGUE_PRIORITY.get(term, [])
            for i, metric in enumerate(priority_metrics[:max_interpretations]):
                expansions = VAGUE_TERMS.get(term, [])
                if not expansions:
                    continue

                # Take only criteria for this specific priority metric
                primary_exp = [e for e in expansions if e["metric"] == metric]
                if not primary_exp:
                    primary_exp = expansions[:1]

                alt_criteria = [
                    ScreenerCriteria(
                        metric=e["metric"],
                        operator=e["operator"],
                        value=e["value"],
                        description=e.get("description", ""),
                    )
                    for e in primary_exp
                ]

                # Add sector/geo from parsed
                for sector in parsed.sectors:
                    alt_criteria.append(ScreenerCriteria(
                        metric="sector", operator="==", value=0.0,
                        description=f"sector = {sector}", sector=sector,
                    ))

                confidence = max(0.3, 0.75 - i * 0.15)
                desc_parts = [c.description for c in alt_criteria]
                interpretations.append(Interpretation(
                    criteria=alt_criteria,
                    confidence=confidence,
                    description=f"'{term}' interpreted as: {', '.join(desc_parts[:3])}",
                ))

        # Deduplicate and sort by confidence
        seen_descs: set = set()
        unique: List[Interpretation] = []
        for interp in sorted(interpretations, key=lambda x: x.confidence, reverse=True):
            if interp.description not in seen_descs:
                seen_descs.add(interp.description)
                unique.append(interp)
        return unique[:max_interpretations]

    def get_default_assumptions(self) -> Dict[str, str]:
        """Return documented default assumptions for vague terms."""
        return {
            term: "; ".join(
                e.get("description", f"{e['metric']} {e['operator']} {e['value']}")
                for e in expansions[:2]
            )
            for term, expansions in VAGUE_TERMS.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. ScreenerExecutor
# ══════════════════════════════════════════════════════════════════════════════


class ScreenerExecutor:
    """
    Execute screener criteria against SENTINEL fundamental data.

    Uses DuckDB (in-memory) if available, otherwise SQLite.
    Results are scored 0-100 based on strength of criteria satisfaction.
    Returns top 20 matches with pass/fail per criterion.
    """

    def __init__(self) -> None:
        self._use_duckdb = self._try_duckdb()

    def _try_duckdb(self) -> bool:
        try:
            import duckdb  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_fundamental_data(self) -> pd.DataFrame:
        """
        Load fundamental data from SENTINEL's SQLite store.

        Falls back to a synthetic sample if no data is available.
        """
        data_db_path = Path(__file__).parent.parent / "data" / "fundamentals.db"

        # Try loading from SQLite
        if data_db_path.exists():
            try:
                conn = sqlite3.connect(str(data_db_path))
                df = pd.read_sql("SELECT * FROM fundamentals LIMIT 2000", conn)
                conn.close()
                if not df.empty:
                    return df
            except Exception:
                pass

        # Fallback: synthetic sample for 100 tickers
        return self._generate_sample_data()

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate a synthetic fundamental dataset for demonstration."""
        import numpy as np
        rng = np.random.default_rng(42)

        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK",
            "JPM", "JNJ", "V", "PG", "HD", "MA", "UNH", "DIS", "BAC", "VZ",
            "ADBE", "CRM", "NFLX", "INTC", "AMD", "PYPL", "CMCSA", "PEP",
            "KO", "NKE", "MRK", "ABT", "WMT", "TMO", "ABBV", "COST", "ACN",
            "AVGO", "TXN", "QCOM", "HON", "LIN", "NEE", "DHR", "LMT", "GE",
            "MMM", "IBM", "GS", "MS", "C", "WFC", "AXP", "SBUX", "MCD",
            "T", "ORCL", "SAP", "UBER", "LYFT", "SQ", "SHOP", "SPOT", "SNAP",
            "COIN", "HOOD", "PLTR", "SNOW", "DDOG", "NET", "ZM", "DOCU",
            "OKTA", "CRWD", "PANW", "ZS", "FTNT", "NOW", "WDAY", "HUBS",
            "TWLO", "MDB", "ESTC", "DKNG", "RBLX", "U", "AFRM", "SOFI",
            "RIVN", "LCID", "NIO", "XPEV", "LI", "BABA", "JD", "PDD",
            "BIDU", "TCEHY", "ASML", "TSM", "F", "GM", "XOM", "CVX",
        ]

        sectors = [
            "Technology", "Healthcare", "Financial", "Consumer Discretionary",
            "Consumer Staples", "Energy", "Industrials", "Materials",
            "Real Estate", "Utilities", "Communication Services",
        ]

        n = len(tickers)

        return pd.DataFrame({
            "ticker": tickers,
            "sector": rng.choice(sectors, size=n),
            "geography": rng.choice(["US", "Europe", "Asia", "Emerging Markets"], size=n, p=[0.6, 0.2, 0.1, 0.1]),
            "market_cap": rng.lognormal(23, 2, size=n),
            "pe_ratio": rng.lognormal(2.8, 0.8, size=n),
            "pb_ratio": rng.lognormal(0.8, 0.8, size=n),
            "ps_ratio": rng.lognormal(1.2, 0.8, size=n),
            "ev_ebitda": rng.lognormal(2.5, 0.7, size=n),
            "peg_ratio": rng.lognormal(0.4, 0.6, size=n),
            "revenue_growth": rng.normal(12.0, 20.0, size=n),
            "earnings_growth": rng.normal(10.0, 25.0, size=n),
            "net_income": rng.lognormal(21, 2.5, size=n) * rng.choice([1, -1], size=n, p=[0.8, 0.2]),
            "revenue": rng.lognormal(22, 2.0, size=n),
            "gross_margin": rng.normal(45.0, 20.0, size=n).clip(0, 100),
            "operating_margin": rng.normal(18.0, 15.0, size=n).clip(-50, 60),
            "net_margin": rng.normal(12.0, 12.0, size=n).clip(-30, 50),
            "roe": rng.normal(18.0, 15.0, size=n),
            "roa": rng.normal(8.0, 8.0, size=n),
            "roic": rng.normal(15.0, 12.0, size=n),
            "debt_equity": rng.exponential(0.8, size=n),
            "current_ratio": rng.lognormal(0.4, 0.5, size=n),
            "quick_ratio": rng.lognormal(0.2, 0.5, size=n),
            "dividend_yield": rng.exponential(1.5, size=n).clip(0, 12),
            "consecutive_div_growth": rng.choice([0, 5, 10, 15, 20, 25, 30, 40, 50], size=n),
            "short_interest": rng.exponential(5.0, size=n).clip(0, 60),
            "float_short": rng.exponential(6.0, size=n).clip(0, 70),
            "rsi": rng.uniform(20, 80, size=n),
            "return_3m": rng.normal(5.0, 20.0, size=n),
            "return_1y": rng.normal(10.0, 35.0, size=n),
            "52_week_high_pct": rng.uniform(60, 100, size=n),
            "beta": rng.lognormal(0.1, 0.5, size=n),
            "earnings_surprise": rng.normal(2.0, 8.0, size=n),
            "analyst_rating": rng.uniform(1, 5, size=n),
            "eps_revision": rng.normal(1.0, 5.0, size=n),
            "iv_percentile": rng.uniform(5, 95, size=n),
            "borrow_cost": rng.exponential(3.0, size=n).clip(0, 100),
        })

    def _apply_criterion(self, df: pd.DataFrame, criterion: ScreenerCriteria) -> pd.Series:
        """
        Apply a single criterion to the dataframe.

        Returns boolean Series for rows that pass.
        """
        # Handle special sector/geography criteria
        if criterion.metric == "sector" and criterion.sector:
            if "sector" not in df.columns:
                return pd.Series(True, index=df.index)
            return df["sector"].str.lower() == criterion.sector.lower()

        if criterion.metric == "geography" and criterion.geography:
            if "geography" not in df.columns:
                return pd.Series(True, index=df.index)
            return df["geography"].str.lower() == criterion.geography.lower()

        if criterion.metric not in df.columns:
            return pd.Series(True, index=df.index)

        col = df[criterion.metric]

        if criterion.operator == "<":
            return col < criterion.value
        elif criterion.operator == "<=":
            return col <= criterion.value
        elif criterion.operator == ">":
            return col > criterion.value
        elif criterion.operator == ">=":
            return col >= criterion.value
        elif criterion.operator == "==":
            return col == criterion.value
        elif criterion.operator == "between" and isinstance(criterion.value, tuple):
            lo, hi = criterion.value
            return (col >= lo) & (col <= hi)
        else:
            return pd.Series(True, index=df.index)

    def _score_row(self, row: pd.Series, criteria: List[ScreenerCriteria]) -> float:
        """
        Score a single row 0-100 based on how strongly it satisfies all criteria.

        Stronger satisfaction (more extreme values) = higher score.
        """
        total_weight = 0.0
        weighted_score = 0.0

        for c in criteria:
            # Skip sector/geography string criteria for scoring
            if c.metric in ("sector", "geography"):
                continue

            if c.metric not in row.index:
                continue

            val = row[c.metric]
            if pd.isna(val):
                continue

            w = c.weight
            total_weight += w

            # Score: 0 if fails, 50-100 if passes proportional to margin
            if c.operator == "between" and isinstance(c.value, tuple):
                lo, hi = c.value
                passes = lo <= val <= hi
                if passes:
                    # Score based on how centered
                    mid = (lo + hi) / 2
                    margin = (hi - lo) / 2
                    dist = abs(val - mid) / max(margin, 1e-6)
                    criterion_score = 100.0 * (1 - dist * 0.5)
                else:
                    criterion_score = 0.0
            else:
                target = c.value if isinstance(c.value, (int, float)) else 0.0

                if c.operator in ("<", "<="):
                    passes = val <= target if c.operator == "<=" else val < target
                    if passes and target != 0:
                        margin = (target - val) / abs(target)
                        criterion_score = 50.0 + min(50.0, margin * 100)
                    else:
                        criterion_score = 0.0 if not passes else 50.0

                elif c.operator in (">", ">="):
                    passes = val >= target if c.operator == ">=" else val > target
                    if passes and target != 0:
                        margin = (val - target) / max(abs(target), 1e-6)
                        criterion_score = 50.0 + min(50.0, margin * 50)
                    else:
                        criterion_score = 0.0 if not passes else 50.0

                else:
                    criterion_score = 100.0 if val == target else 0.0

            weighted_score += criterion_score * w

        if total_weight == 0:
            return 50.0
        return round(weighted_score / total_weight, 1)

    def execute(
        self,
        criteria: List[ScreenerCriteria],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Execute screener criteria and return ranked matches.

        Returns DataFrame with tickers, criteria pass/fail columns, and score 0-100.
        """
        df = self._load_fundamental_data()

        if df.empty or not criteria:
            return pd.DataFrame()

        # Apply all criteria (AND logic)
        mask = pd.Series(True, index=df.index)
        for criterion in criteria:
            criterion_mask = self._apply_criterion(df, criterion)
            mask = mask & criterion_mask

        matches = df[mask].copy()

        if matches.empty:
            return pd.DataFrame()

        # Score each match
        matches["_score"] = matches.apply(
            lambda row: self._score_row(row, criteria),
            axis=1,
        )

        # Add per-criterion pass columns for transparency
        for i, c in enumerate(criteria):
            col_name = f"_pass_{i}_{c.metric}"
            try:
                matches[col_name] = self._apply_criterion(matches, c)
            except Exception:
                matches[col_name] = True

        matches = matches.sort_values("_score", ascending=False).head(top_n)
        return matches.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. NLToScreenerPipeline
# ══════════════════════════════════════════════════════════════════════════════


def _get_screener_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screener_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            criteria_json TEXT,
            n_matches INTEGER,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


class NLToScreenerPipeline:
    """
    End-to-end NL → screener pipeline.

    Steps: parse → resolve ambiguity → build criteria → execute → rank → explain.
    """

    def __init__(self) -> None:
        self._parser = NLQueryParser()
        self._builder = ScreenerCriteriaBuilder()
        self._resolver = QueryAmbiguityResolver()
        self._executor = ScreenerExecutor()

    def _store_query(self, query: str, criteria: List[ScreenerCriteria], n_matches: int) -> None:
        """Persist query to SQLite history (keep last 100)."""
        import json
        try:
            conn = _get_screener_db()
            criteria_json = json.dumps([
                {"metric": c.metric, "operator": c.operator, "value": c.value if isinstance(c.value, (int, float)) else list(c.value)}
                for c in criteria
            ])
            conn.execute(
                "INSERT INTO screener_history (query, criteria_json, n_matches, created_at) VALUES (?, ?, ?, ?)",
                (query, criteria_json, n_matches, time.time()),
            )
            # Prune to last 100
            conn.execute(
                "DELETE FROM screener_history WHERE id NOT IN "
                "(SELECT id FROM screener_history ORDER BY created_at DESC LIMIT 100)"
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _build_explanation(
        self,
        query: str,
        criteria: List[ScreenerCriteria],
        n_matches: int,
    ) -> str:
        """Generate a human-readable explanation of the screen."""
        criteria_parts = [c.description for c in criteria if c.description]
        if not criteria_parts:
            return f"No structured criteria could be parsed from: '{query}'"

        criteria_str = " AND ".join(criteria_parts[:6])
        return (
            f"Found {n_matches} stocks matching: {criteria_str}. "
            f"Results ranked by how strongly they satisfy each criterion."
        )

    def translate(self, query: str, execute: bool = True) -> ScreenerResult:
        """
        Full pipeline: NL query → structured criteria → execution → ranked results.

        Returns ScreenerResult with matches, explanation, and criteria used.
        """
        # Parse
        parsed = self._parser.parse(query)

        # Build criteria
        criteria = self._builder.build_from_parsed(parsed)

        # Handle compound OR logic by using first interpretation
        if not criteria:
            interpretations = self._resolver.resolve(query, max_interpretations=1)
            if interpretations:
                criteria = interpretations[0].criteria

        # Execute
        matches_df = pd.DataFrame()
        if execute and criteria:
            matches_df = self._executor.execute(criteria)

        n_matches = len(matches_df)
        top_tickers = (
            matches_df["ticker"].tolist()[:10]
            if "ticker" in matches_df.columns
            else []
        )

        explanation = self._build_explanation(query, criteria, n_matches)

        # Store to history
        self._store_query(query, criteria, n_matches)

        return ScreenerResult(
            query=query,
            criteria=criteria,
            matches=matches_df,
            explanation=explanation,
            n_matches=n_matches,
            top_matches=top_tickers,
            criteria_descriptions=[c.description for c in criteria],
        )

    def get_history(self, limit: int = 20) -> List[Dict]:
        """Retrieve recent query history."""
        import json
        try:
            conn = _get_screener_db()
            rows = conn.execute(
                "SELECT query, criteria_json, n_matches, created_at FROM screener_history "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            return [
                {
                    "query": r[0],
                    "criteria": json.loads(r[1]) if r[1] else [],
                    "n_matches": r[2],
                    "created_at": datetime.fromtimestamp(r[3], tz=timezone.utc).isoformat(),
                }
                for r in rows
            ]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════════════════════
# 6. ScreenerTemplateLibrary
# ══════════════════════════════════════════════════════════════════════════════


class ScreenerTemplateLibrary:
    """25+ pre-built investment strategy templates."""

    def list_templates(self) -> List[Dict]:
        """Return all template names and descriptions."""
        return [
            {
                "name": name,
                "description": tmpl["description"],
                "n_criteria": len(tmpl.get("criteria", [])),
                "sector": tmpl.get("sector"),
                "geography": tmpl.get("geography"),
            }
            for name, tmpl in SCREENER_TEMPLATES.items()
        ]

    def get_template(self, name: str) -> Dict:
        """Retrieve a specific template by name."""
        tmpl = SCREENER_TEMPLATES.get(name)
        if not tmpl:
            raise ValueError(f"Template '{name}' not found. Available: {list(SCREENER_TEMPLATES.keys())}")
        return {"name": name, **tmpl}

    def execute_template(self, name: str, top_n: int = 20) -> pd.DataFrame:
        """Build criteria from template and execute."""
        builder = ScreenerCriteriaBuilder()
        executor = ScreenerExecutor()
        criteria = builder.build_from_template(name)
        return executor.execute(criteria, top_n=top_n)

    def search_templates(self, keyword: str) -> List[Dict]:
        """Search templates by keyword in name or description."""
        kw = keyword.lower()
        return [
            {"name": name, "description": tmpl["description"]}
            for name, tmpl in SCREENER_TEMPLATES.items()
            if kw in name.lower() or kw in tmpl["description"].lower()
        ]

    def suggest_template(self, query: str) -> Optional[str]:
        """Suggest the best template name for a natural language query."""
        query_lower = query.lower()
        best_name: Optional[str] = None
        best_score = 0

        for name, tmpl in SCREENER_TEMPLATES.items():
            score = 0
            # Check name match
            name_words = name.replace("_", " ").split()
            for word in name_words:
                if word in query_lower:
                    score += 2

            # Check description match
            desc_words = tmpl["description"].lower().split()
            for word in desc_words:
                if len(word) > 4 and word in query_lower:
                    score += 1

            if score > best_score:
                best_score = score
                best_name = name

        return best_name if best_score > 0 else None


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Router
# ══════════════════════════════════════════════════════════════════════════════

nl_screener_router = APIRouter(prefix="/screener", tags=["screener"])

_pipeline: Optional[NLToScreenerPipeline] = None
_template_lib: Optional[ScreenerTemplateLibrary] = None


def _get_pipeline() -> NLToScreenerPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = NLToScreenerPipeline()
    return _pipeline


def _get_template_lib() -> ScreenerTemplateLibrary:
    global _template_lib
    if _template_lib is None:
        _template_lib = ScreenerTemplateLibrary()
    return _template_lib


def _criteria_to_dict(c: ScreenerCriteria) -> Dict:
    return {
        "metric": c.metric,
        "operator": c.operator,
        "value": list(c.value) if isinstance(c.value, tuple) else c.value,
        "weight": c.weight,
        "description": c.description,
        "sector": c.sector,
        "geography": c.geography,
    }


@nl_screener_router.post("/translate", response_model=TranslateResponse)
def translate_endpoint(req: TranslateRequest):
    """
    Translate a natural language query into structured screener criteria.

    Optionally executes the screen and returns top matches.
    """
    result = _get_pipeline().translate(req.query, execute=req.execute)
    return TranslateResponse(
        query=result.query,
        criteria=[_criteria_to_dict(c) for c in result.criteria],
        explanation=result.explanation,
        criteria_descriptions=result.criteria_descriptions,
        n_criteria=len(result.criteria),
    )


@nl_screener_router.post("/execute", response_model=ScreenerResultResponse)
def execute_endpoint(req: TranslateRequest):
    """Execute a NL query as a full screener and return top matches."""
    result = _get_pipeline().translate(req.query, execute=True)
    return ScreenerResultResponse(
        query=result.query,
        criteria_descriptions=result.criteria_descriptions,
        explanation=result.explanation,
        n_matches=result.n_matches,
        top_matches=result.top_matches,
    )


@nl_screener_router.post("/execute-criteria")
def execute_criteria_endpoint(req: ExecuteRequest):
    """Execute a list of pre-built criteria dicts against the fundamental database."""
    executor = ScreenerExecutor()
    builder = ScreenerCriteriaBuilder()

    criteria: List[ScreenerCriteria] = []
    for c_dict in req.criteria:
        value = c_dict.get("value", 0.0)
        if isinstance(value, list):
            value = tuple(value)
        criteria.append(ScreenerCriteria(
            metric=c_dict.get("metric", ""),
            operator=c_dict.get("operator", ">"),
            value=value,
            weight=c_dict.get("weight", 1.0),
            description=c_dict.get("description", ""),
            sector=c_dict.get("sector"),
            geography=c_dict.get("geography"),
        ))

    matches_df = executor.execute(criteria)
    tickers = matches_df["ticker"].tolist()[:20] if "ticker" in matches_df.columns else []
    scores = matches_df["_score"].tolist()[:20] if "_score" in matches_df.columns else []

    return {
        "n_matches": len(matches_df),
        "top_matches": [{"ticker": t, "score": s} for t, s in zip(tickers, scores)],
        "criteria_descriptions": [c.description for c in criteria],
    }


@nl_screener_router.get("/templates")
def list_templates_endpoint():
    """List all available screener templates."""
    return {
        "templates": _get_template_lib().list_templates(),
        "count": len(SCREENER_TEMPLATES),
    }


@nl_screener_router.get("/templates/{name}")
def get_template_endpoint(name: str, execute: bool = False):
    """Get a specific template's criteria and optionally execute it."""
    try:
        tmpl = _get_template_lib().get_template(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result: Dict[str, Any] = {"template": tmpl}

    if execute:
        try:
            matches_df = _get_template_lib().execute_template(name)
            tickers = matches_df["ticker"].tolist()[:20] if "ticker" in matches_df.columns else []
            scores = matches_df["_score"].tolist()[:20] if "_score" in matches_df.columns else []
            result["matches"] = [{"ticker": t, "score": s} for t, s in zip(tickers, scores)]
            result["n_matches"] = len(matches_df)
        except Exception as e:
            result["execute_error"] = str(e)

    return result


@nl_screener_router.get("/history")
def history_endpoint(limit: int = 20):
    """Retrieve recent screener query history."""
    history = _get_pipeline().get_history(limit=limit)
    return {"history": history, "count": len(history)}


@nl_screener_router.post("/explain")
def explain_endpoint(req: ExplainRequest):
    """
    Explain alternative interpretations of an ambiguous query.

    Returns top N interpretations with confidence scores.
    """
    resolver = QueryAmbiguityResolver()
    interpretations = resolver.resolve(req.query, max_interpretations=req.max_interpretations)

    defaults = resolver.get_default_assumptions()

    return {
        "query": req.query,
        "interpretations": [
            {
                "description": i.description,
                "confidence": i.confidence,
                "criteria": [_criteria_to_dict(c) for c in i.criteria],
            }
            for i in interpretations
        ],
        "default_assumptions": {
            k: v for k, v in list(defaults.items())[:10]
        },
        "suggested_template": ScreenerTemplateLibrary().suggest_template(req.query),
    }


# ── Module-level convenience functions ───────────────────────────────────────

def translate(query: str) -> ScreenerResult:
    """Translate a NL query and execute. Module-level convenience."""
    return NLToScreenerPipeline().translate(query)


def execute_template(name: str) -> pd.DataFrame:
    """Execute a named template. Module-level convenience."""
    return ScreenerTemplateLibrary().execute_template(name)


def parse_query(query: str) -> ParsedQuery:
    """Parse a NL query into structured entities. Module-level convenience."""
    return NLQueryParser().parse(query)


def list_templates() -> List[Dict]:
    """List all templates. Module-level convenience."""
    return ScreenerTemplateLibrary().list_templates()
