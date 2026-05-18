"""Financial Query Expander V3 — Institutional NLP query intelligence engine.

dim_056: Smart synonym / query expansion  (score 5 → 9)

V3 additions over V2:
  FinancialOntology       — 200+ financial term synonyms, 50+ ticker aliases, metric aliases
  QueryExpander           — expand() returning ExpandedQuery with confidence scoring
  SemanticSearch          — TF-IDF cosine similarity over financial corpus
  AutocompleteEngine      — prefix trie with context-aware suggestions
  QueryParser             — full NL → StructuredQuery with filters, universe, intent
  find_ticker()           — SEC EDGAR company search fuzzy matching
  CLI demo in __main__

Free APIs only. No paid data. Core works without optional deps.

Usage::
    from sentinel.sai.query_expander_v3 import QueryExpander, QueryParser, AutocompleteEngine

    qe = QueryExpander()
    result = qe.expand("show me tech stocks with low PE and high FCF yield")
    print(result.expanded_terms)

    qp = QueryParser()
    sq = qp.parse("tech stocks with PE under 20 and revenue growth > 10%")
    print(sq.filters)
"""
from __future__ import annotations

import logging
import math
import re
import time
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP helper with retry + rate limiting
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "SENTINEL:FinancialTerminal:3.0 (research; contact@sentinel.ai)",
    "Accept": "application/json",
})

_LAST_EDGAR_REQ: float = 0.0
_EDGAR_MIN_INTERVAL = 0.11  # ~9 req/sec max, EDGAR allows 10


def _edgar_get(url: str, params: Optional[dict] = None, timeout: int = 10) -> dict:
    """Rate-limited GET to SEC EDGAR."""
    global _LAST_EDGAR_REQ
    elapsed = time.monotonic() - _LAST_EDGAR_REQ
    if elapsed < _EDGAR_MIN_INTERVAL:
        time.sleep(_EDGAR_MIN_INTERVAL - elapsed)
    _LAST_EDGAR_REQ = time.monotonic()
    try:
        resp = _SESSION.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("EDGAR request failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# FinancialOntology — 200+ term mappings
# ---------------------------------------------------------------------------

class FinancialOntology:
    """Hardcoded financial ontology: synonyms, metric aliases, ticker aliases.

    Design: every canonical term maps to a list of surface-form aliases.
    The reverse mapping (alias → canonical) is built at init time.
    """

    # ── Core income statement ─────────────────────────────────────────────
    METRIC_SYNONYMS: Dict[str, List[str]] = {
        "total_revenue": [
            "revenue", "revenues", "sales", "net sales", "top line", "net revenue",
            "total revenues", "gross revenue", "total sales", "turnover",
            "income from operations", "operating revenue",
        ],
        "net_income": [
            "earnings", "net earnings", "profits", "bottom line", "net profit",
            "after-tax income", "net loss", "income attributable to shareholders",
            "attributable earnings", "eps denominator",
        ],
        "eps": [
            "earnings per share", "EPS", "diluted EPS", "basic EPS",
            "earnings per diluted share", "diluted earnings per share",
        ],
        "ebitda": [
            "EBITDA", "earnings before interest taxes depreciation amortization",
            "operating cash flow proxy", "adj ebitda", "adjusted ebitda",
            "ebitda margin proxy",
        ],
        "ebit": [
            "operating income", "operating profit", "EBIT",
            "earnings before interest and taxes", "operating earnings",
        ],
        "gross_profit": [
            "gross profit", "gross income", "gross margin dollars",
        ],
        "gross_margin": [
            "gross profit margin", "GP margin", "gross margin %",
            "gross profitability", "gross profit percentage", "product margin",
        ],
        "operating_margin": [
            "EBIT margin", "operating profit margin", "op margin",
            "operating income margin", "EBIT %", "operating leverage",
        ],
        "net_margin": [
            "net profit margin", "profit margin", "bottom-line margin",
            "net income margin", "return on sales",
        ],
        "cost_of_goods_sold": [
            "COGS", "cost of revenue", "cost of sales", "direct costs",
            "cost of goods", "production costs",
        ],

        # ── Cash flow ─────────────────────────────────────────────────────
        "free_cash_flow": [
            "FCF", "free cash flow", "levered FCF", "unlevered FCF",
            "cash from operations minus capex", "owner earnings",
        ],
        "operating_cash_flow": [
            "cash from operations", "CFO", "operating CF",
            "cash flow from operating activities",
        ],
        "capex": [
            "capital expenditure", "capital expenditures", "PP&E additions",
            "property plant equipment spend", "maintenance capex", "growth capex",
        ],
        "fcf_yield": [
            "free cash flow yield", "FCF yield", "cash flow yield",
            "cash return on price",
        ],

        # ── Balance sheet ─────────────────────────────────────────────────
        "total_assets": [
            "assets", "total assets", "balance sheet total",
        ],
        "total_debt": [
            "debt", "total borrowings", "financial debt", "interest-bearing debt",
            "net debt gross", "long term debt plus short term debt",
        ],
        "net_debt": [
            "net debt", "debt net of cash", "financial leverage net",
        ],
        "cash_and_equivalents": [
            "cash", "cash and cash equivalents", "cash on hand",
            "liquid assets", "short term investments",
        ],
        "shareholders_equity": [
            "equity", "book value", "net assets", "stockholders equity",
            "owners equity", "book value of equity",
        ],

        # ── Valuation ratios ──────────────────────────────────────────────
        "pe_ratio": [
            "price to earnings", "P/E", "PE", "PE ratio", "p/e ratio",
            "earnings multiple", "valuation multiple", "price earnings ratio",
        ],
        "pb_ratio": [
            "price to book", "P/B", "PB", "price book ratio",
            "market to book", "price to book value",
        ],
        "ps_ratio": [
            "price to sales", "P/S", "PS", "price sales ratio",
            "revenue multiple", "price to revenue",
        ],
        "ev_ebitda": [
            "EV/EBITDA", "enterprise value to EBITDA", "EV multiple",
            "enterprise multiple", "EV to EBITDA",
        ],
        "ev_revenue": [
            "EV/Revenue", "enterprise value to revenue", "EV to sales",
        ],
        "peg_ratio": [
            "PEG", "price earnings growth", "P/E to growth",
            "growth-adjusted PE",
        ],
        "dividend_yield": [
            "yield", "div yield", "dividend yield %",
            "income yield", "distribution yield",
        ],

        # ── Returns ───────────────────────────────────────────────────────
        "return_on_equity": [
            "ROE", "return on equity", "equity return",
            "return on shareholders equity",
        ],
        "return_on_assets": [
            "ROA", "return on assets", "asset return",
        ],
        "return_on_invested_capital": [
            "ROIC", "return on invested capital", "return on capital",
            "capital return", "invested capital return",
        ],
        "return_on_capital_employed": [
            "ROCE", "return on capital employed",
        ],

        # ── Leverage & liquidity ──────────────────────────────────────────
        "debt_to_equity": [
            "D/E", "debt equity ratio", "leverage ratio", "gearing",
            "financial leverage", "debt to equity ratio",
        ],
        "debt_to_ebitda": [
            "leverage", "net leverage", "debt multiple",
            "debt to EBITDA", "D/EBITDA",
        ],
        "current_ratio": [
            "liquidity ratio", "current liquidity", "working capital ratio",
        ],
        "quick_ratio": [
            "acid test", "quick liquidity", "liquid ratio",
        ],
        "interest_coverage": [
            "interest coverage ratio", "ICR", "times interest earned",
            "EBIT to interest",
        ],

        # ── Growth metrics ────────────────────────────────────────────────
        "revenue_growth_yoy": [
            "revenue growth", "sales growth", "top-line growth",
            "revenue increase", "revenue expansion", "YoY revenue growth",
        ],
        "eps_growth_yoy": [
            "earnings growth", "EPS growth", "profit growth",
            "earnings increase",
        ],
        "ebitda_growth_yoy": [
            "EBITDA growth", "cash earnings growth",
        ],

        # ── Market / technical ────────────────────────────────────────────
        "market_cap": [
            "market capitalisation", "market cap", "market value",
            "equity market cap", "market capitalisation",
        ],
        "enterprise_value": [
            "EV", "enterprise value", "total firm value",
            "market cap plus net debt",
        ],
        "price_momentum": [
            "momentum", "relative strength", "price performance",
            "price trend", "stock momentum",
        ],
        "rsi": [
            "RSI", "relative strength index", "overbought oversold",
            "14-day RSI",
        ],
        "rate_of_change": [
            "ROC", "rate of change", "price rate of change",
            "momentum indicator",
        ],
        "volume": [
            "trading volume", "shares traded", "average volume",
            "daily volume", "turnover volume",
        ],
        "short_interest": [
            "short interest", "short float", "short ratio",
            "days to cover", "borrow rate",
        ],
        "beta": [
            "market beta", "systematic risk", "market sensitivity",
            "correlation to market",
        ],

        # ── ESG ───────────────────────────────────────────────────────────
        "esg_score": [
            "ESG", "ESG rating", "sustainability score",
            "environmental social governance", "sustainability rating",
        ],
        "carbon_emissions": [
            "CO2 emissions", "greenhouse gas", "GHG", "carbon footprint",
            "scope 1 emissions", "scope 2 emissions",
        ],

        # ── M&A ───────────────────────────────────────────────────────────
        "acquisition_premium": [
            "takeover premium", "M&A premium", "deal premium",
            "control premium",
        ],
        "synergies": [
            "cost synergies", "revenue synergies", "merger synergies",
            "deal synergies", "cost savings",
        ],
        "goodwill": [
            "intangibles", "goodwill impairment", "M&A goodwill",
            "acquisition intangibles",
        ],

        # ── Fixed income ──────────────────────────────────────────────────
        "yield_to_maturity": [
            "YTM", "bond yield", "redemption yield",
            "yield to maturity",
        ],
        "credit_spread": [
            "spread", "OAS", "option adjusted spread",
            "credit risk premium", "bond spread",
        ],
        "duration": [
            "modified duration", "Macaulay duration", "interest rate sensitivity",
            "DV01 proxy",
        ],
        "credit_rating": [
            "rating", "bond rating", "issuer rating",
            "S&P rating", "Moody's rating",
        ],

        # ── Options ───────────────────────────────────────────────────────
        "implied_volatility": [
            "IV", "implied vol", "options implied volatility",
            "options pricing vol",
        ],
        "delta": [
            "option delta", "hedge ratio", "delta hedging",
        ],
        "gamma": [
            "option gamma", "gamma exposure", "GEX",
        ],
        "theta": [
            "time decay", "option theta", "daily decay",
        ],
        "vega": [
            "vol sensitivity", "option vega", "volatility dollar",
        ],
        "put_call_ratio": [
            "P/C ratio", "put call ratio", "options sentiment indicator",
        ],

        # ── Commodities ───────────────────────────────────────────────────
        "spot_price": [
            "commodity price", "spot", "cash price",
        ],
        "futures_price": [
            "front month futures", "commodity futures", "forward price",
        ],
        "basis": [
            "cash futures basis", "commodity basis",
        ],

        # ── Macro ─────────────────────────────────────────────────────────
        "gdp_growth": [
            "GDP", "economic growth", "output growth",
            "real GDP growth", "nominal GDP",
        ],
        "inflation": [
            "CPI", "consumer price index", "inflation rate",
            "price level", "PCE", "core inflation",
        ],
        "interest_rate": [
            "fed funds rate", "policy rate", "base rate",
            "central bank rate", "benchmark rate",
        ],
        "unemployment": [
            "unemployment rate", "jobless rate", "labor market slack",
            "nonfarm payrolls", "NFP",
        ],

        # ── Analyst / sentiment ───────────────────────────────────────────
        "analyst_rating": [
            "analyst recommendation", "buy rating", "sell rating",
            "consensus rating", "street consensus",
        ],
        "price_target": [
            "PT", "target price", "analyst price target",
            "street price target", "consensus target",
        ],
        "earnings_surprise": [
            "earnings beat", "earnings miss", "surprise factor",
            "estimate beat", "EPS surprise",
        ],
    }

    # ── Ticker aliases (company name → ticker) ────────────────────────────
    TICKER_ALIASES: Dict[str, str] = {
        # FAANGM + mega cap tech
        "apple": "AAPL",
        "apple inc": "AAPL",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "amazon": "AMZN",
        "netflix": "NFLX",
        "microsoft": "MSFT",
        "nvidia": "NVDA",
        "tesla": "TSLA",
        "salesforce": "CRM",
        "adobe": "ADBE",
        "intel": "INTC",
        "amd": "AMD",
        "advanced micro devices": "AMD",
        "qualcomm": "QCOM",
        "broadcom": "AVGO",
        "texas instruments": "TXN",
        "applied materials": "AMAT",
        "lam research": "LRCX",
        "kla": "KLAC",
        # Finance
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "goldman sachs": "GS",
        "goldman": "GS",
        "morgan stanley": "MS",
        "bank of america": "BAC",
        "wells fargo": "WFC",
        "citigroup": "C",
        "citi": "C",
        "blackrock": "BLK",
        "berkshire": "BRK.B",
        "berkshire hathaway": "BRK.B",
        "visa": "V",
        "mastercard": "MA",
        "paypal": "PYPL",
        "american express": "AXP",
        "amex": "AXP",
        # Healthcare
        "johnson and johnson": "JNJ",
        "j&j": "JNJ",
        "pfizer": "PFE",
        "moderna": "MRNA",
        "unitedhealth": "UNH",
        "abbvie": "ABBV",
        "eli lilly": "LLY",
        "lilly": "LLY",
        "merck": "MRK",
        "bristol myers": "BMY",
        "bristol-myers squibb": "BMY",
        "amgen": "AMGN",
        "gilead": "GILD",
        # Consumer
        "walmart": "WMT",
        "target": "TGT",
        "costco": "COST",
        "nike": "NKE",
        "starbucks": "SBUX",
        "mcdonalds": "MCD",
        "mcdonald's": "MCD",
        "coca cola": "KO",
        "pepsi": "PEP",
        "pepsico": "PEP",
        "procter and gamble": "PG",
        "p&g": "PG",
        "colgate": "CL",
        # Energy
        "exxon": "XOM",
        "exxonmobil": "XOM",
        "chevron": "CVX",
        "conocophillips": "COP",
        "schlumberger": "SLB",
        "halliburton": "HAL",
        # Industrial
        "general electric": "GE",
        "ge": "GE",
        "boeing": "BA",
        "caterpillar": "CAT",
        "deere": "DE",
        "john deere": "DE",
        "3m": "MMM",
        "honeywell": "HON",
        "union pacific": "UNP",
        # Telecom/Media
        "at&t": "T",
        "verizon": "VZ",
        "comcast": "CMCSA",
        "disney": "DIS",
        "walt disney": "DIS",
        "spotify": "SPOT",
        # Cloud / SaaS
        "snowflake": "SNOW",
        "datadog": "DDOG",
        "crowdstrike": "CRWD",
        "palantir": "PLTR",
        "servicenow": "NOW",
        "workday": "WDAY",
        "twilio": "TWLO",
        "shopify": "SHOP",
        "square": "SQ",
        "block": "SQ",
        "uber": "UBER",
        "lyft": "LYFT",
        "airbnb": "ABNB",
        "doordash": "DASH",
    }

    # ── Metric abbreviation aliases ───────────────────────────────────────
    METRIC_ABBREVIATIONS: Dict[str, str] = {
        "fcf": "free_cash_flow",
        "roe": "return_on_equity",
        "roa": "return_on_assets",
        "roic": "return_on_invested_capital",
        "roce": "return_on_capital_employed",
        "ev": "enterprise_value",
        "mktcap": "market_cap",
        "ytm": "yield_to_maturity",
        "dps": "dividends_per_share",
        "bvps": "book_value_per_share",
        "cfo": "operating_cash_flow",
        "capex": "capex",
        "pe": "pe_ratio",
        "pb": "pb_ratio",
        "ps": "ps_ratio",
        "peg": "peg_ratio",
        "iv": "implied_volatility",
        "esg": "esg_score",
        "rev": "total_revenue",
        "ni": "net_income",
        "gm": "gross_margin",
        "om": "operating_margin",
        "d/e": "debt_to_equity",
        "cr": "current_ratio",
        "qr": "quick_ratio",
        "ic": "interest_coverage",
        "pt": "price_target",
    }

    # ── Concept synonyms (non-metric) ─────────────────────────────────────
    CONCEPT_SYNONYMS: Dict[str, List[str]] = {
        "liquidity": [
            "current ratio", "quick ratio", "cash ratio", "working capital",
            "liquid assets", "ability to pay",
        ],
        "leverage": [
            "debt to equity", "debt ratio", "financial leverage", "gearing",
            "net leverage", "balance sheet risk",
        ],
        "momentum": [
            "price momentum", "relative strength", "RSI", "rate of change",
            "trend following", "price trend",
        ],
        "value": [
            "undervalued", "cheap", "discount to peers", "low multiple",
            "value investing", "Graham number",
        ],
        "growth": [
            "high growth", "hypergrowth", "revenue acceleration",
            "earnings growth", "top-line expansion",
        ],
        "quality": [
            "high quality", "ROIC", "durable moat", "competitive advantage",
            "franchise value",
        ],
        "dividend": [
            "income stock", "yield", "payout", "distribution",
            "dividend aristocrat", "dividend grower",
        ],
        "buyback": [
            "share repurchase", "stock buyback", "repurchase program",
            "capital return",
        ],
        "squeeze": [
            "short squeeze", "gamma squeeze", "short covering",
            "high short interest",
        ],
        "merger": [
            "M&A", "acquisition", "takeover", "deal", "merger",
            "strategic combination",
        ],
        "ipo": [
            "initial public offering", "new listing", "float",
            "going public", "direct listing",
        ],
        "spinoff": [
            "spin-off", "carve-out", "divestiture", "separation",
        ],
        "bankruptcy": [
            "chapter 11", "insolvency", "restructuring", "chapter 7",
            "default", "distressed",
        ],
        "insider_buying": [
            "insider purchase", "director buying", "Form 4",
            "insider activity",
        ],
        "catalyst": [
            "upcoming catalyst", "event driven", "near term catalyst",
            "binary event",
        ],
    }

    # ── Sector SIC ranges ─────────────────────────────────────────────────
    SECTOR_SIC: Dict[str, Tuple[int, int]] = {
        "technology": (3570, 3579),
        "software": (7370, 7379),
        "semiconductors": (3670, 3679),
        "financial": (6000, 6999),
        "banking": (6000, 6099),
        "insurance": (6300, 6399),
        "healthcare": (8000, 8099),
        "pharmaceuticals": (2830, 2839),
        "biotech": (2836, 2836),
        "energy": (1300, 1499),
        "oil gas": (1311, 1311),
        "utilities": (4900, 4999),
        "consumer discretionary": (5200, 5999),
        "retail": (5200, 5999),
        "consumer staples": (2000, 2199),
        "industrials": (3400, 3499),
        "aerospace defense": (3760, 3769),
        "transportation": (4000, 4799),
        "real estate": (6500, 6599),
        "materials": (1000, 1499),
        "mining": (1000, 1499),
        "telecom": (4800, 4899),
        "media": (7800, 7999),
    }

    def __init__(self) -> None:
        # Build reverse lookup: surface form → canonical metric
        self._alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in self.METRIC_SYNONYMS.items():
            for alias in aliases:
                self._alias_to_canonical[alias.lower()] = canonical
            # canonical maps to itself
            self._alias_to_canonical[canonical.lower()] = canonical
        for abbr, canonical in self.METRIC_ABBREVIATIONS.items():
            self._alias_to_canonical[abbr.lower()] = canonical

        # Ticker alias reverse (name → ticker already; build lower-keyed version)
        self._ticker_map: Dict[str, str] = {
            k.lower(): v for k, v in self.TICKER_ALIASES.items()
        }

    def canonical_metric(self, surface: str) -> Optional[str]:
        """Return canonical metric name for a surface form, or None."""
        return self._alias_to_canonical.get(surface.strip().lower())

    def get_synonyms(self, canonical: str) -> List[str]:
        """Return all surface forms for a canonical metric."""
        return self.METRIC_SYNONYMS.get(canonical, [])

    def resolve_ticker(self, name: str) -> Optional[str]:
        """Return ticker for a company name (local lookup only)."""
        return self._ticker_map.get(name.strip().lower())

    def all_terms(self) -> List[str]:
        """Flat list of every known term (for TF-IDF corpus)."""
        terms: List[str] = []
        for canon, aliases in self.METRIC_SYNONYMS.items():
            terms.append(canon)
            terms.extend(aliases)
        for concept, aliases in self.CONCEPT_SYNONYMS.items():
            terms.append(concept)
            terms.extend(aliases)
        terms.extend(self.TICKER_ALIASES.keys())
        terms.extend(self.METRIC_ABBREVIATIONS.keys())
        return list(dict.fromkeys(terms))  # deduplicate preserving order


# ---------------------------------------------------------------------------
# Dataclasses & Enums
# ---------------------------------------------------------------------------

class QueryIntent(str, Enum):
    SCREENING = "SCREENING"
    CHARTING = "CHARTING"
    FUNDAMENTAL = "FUNDAMENTAL"
    MACRO = "MACRO"
    SENTIMENT = "SENTIMENT"
    COMPARISON = "COMPARISON"
    ALERT = "ALERT"
    UNKNOWN = "UNKNOWN"


@dataclass
class ExpandedQuery:
    original: str
    expanded_terms: List[str]
    synonyms_used: Dict[str, List[str]]
    intent: QueryIntent
    detected_tickers: List[str]
    confidence: float
    normalized_query: str


@dataclass
class SearchResult:
    item: str
    score: float
    rank: int


@dataclass
class Suggestion:
    text: str
    type: str       # "ticker" | "metric" | "concept" | "timeframe" | "operator"
    score: float
    description: str = ""


@dataclass
class Filter:
    metric: str
    op: str         # "lt" | "gt" | "lte" | "gte" | "eq" | "between" | "neq"
    value: Any
    value2: Optional[Any] = None  # for "between"
    raw: str = ""


@dataclass
class StructuredQuery:
    filters: List[Filter]
    universe: Optional[str]
    intent: QueryIntent
    tickers: List[str]
    time_period: Optional[str]
    raw: str
    normalized: str


# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------

# Keyword lists per intent.  Each entry is (keyword, weight).
# High-signal discriminators get weight > 1 so they decisively tip the score
# even when a broad intent (FUNDAMENTAL) accumulates many low-signal hits.
_INTENT_KEYWORDS: Dict[QueryIntent, List[Tuple[str, int]]] = {
    QueryIntent.SCREENING: [
        ("screen", 2), ("filter", 2), ("find stocks", 3), ("find companies", 3),
        ("show me", 2), ("stocks with", 3), ("companies with", 3),
        ("where", 1), ("under", 1), ("over", 1), ("above", 1),
        ("below", 1), ("list", 1), ("rank", 2), ("sort", 2), ("top", 1), ("bottom", 1),
    ],
    QueryIntent.CHARTING: [
        ("chart", 3), ("plot", 3), ("graph", 3), ("show price", 3),
        ("price history", 3), ("historical", 2), ("candlestick", 3),
        ("compare prices", 3), ("overlay", 2), ("technical", 2),
    ],
    QueryIntent.FUNDAMENTAL: [
        ("revenue", 1), ("earnings", 1), ("balance sheet", 2),
        ("income statement", 2), ("cash flow", 1), ("fundamentals", 2),
        ("financials", 2), ("what did", 2), ("report", 1), ("quarter", 1),
        ("annual", 1), ("10-k", 2), ("10-q", 2), ("8-k", 2),
    ],
    QueryIntent.MACRO: [
        ("gdp", 2), ("inflation", 2), ("fed", 2), ("central bank", 3),
        ("interest rate", 2), ("unemployment", 2), ("macro", 3),
        ("economic", 2), ("cpi", 2), ("pce", 2), ("jobs", 1),
        ("payrolls", 2), ("yield curve", 3), ("treasury", 2), ("recession", 2),
    ],
    QueryIntent.SENTIMENT: [
        ("sentiment", 3), ("reddit", 3), ("wsb", 3), ("wallstreetbets", 3),
        ("social", 2), ("twitter", 3), ("mention", 2), ("bullish", 2),
        ("bearish", 2), ("fear", 1), ("greed", 1), ("news", 1),
    ],
    QueryIntent.COMPARISON: [
        # High weights: these keywords are unambiguous comparison signals
        ("vs", 4), ("versus", 4), ("compare", 4), ("relative to", 4),
        ("peer", 3), ("benchmark", 3), ("against", 3),
        ("better than", 3), ("worse than", 3),
    ],
    QueryIntent.ALERT: [
        ("alert", 4), ("notify", 4), ("if price", 4), ("set alert", 4),
        ("trigger", 3), ("watch", 2), ("monitor", 3), ("track", 2),
    ],
}


class QueryExpander:
    """Expand a natural language financial query using the FinancialOntology."""

    def __init__(self) -> None:
        self.ontology = FinancialOntology()

    def tokenize_financial(self, text: str) -> List[str]:
        """Tokenise financial text, handling hyphens, abbreviations, tickers."""
        # Normalise unicode (e.g. fancy quotes)
        text = unicodedata.normalize("NFKD", text)
        # Preserve hyphenated financial terms (e.g. year-over-year, price-to-earnings)
        # Split on whitespace and punctuation except hyphen and slash
        tokens: List[str] = []
        # First pass: split on spaces
        for raw in re.split(r"\s+", text.strip()):
            # Strip trailing punctuation but keep internal hyphens/slashes
            raw = raw.strip(",.!?;:\"'()[]{}")
            if not raw:
                continue
            # Keep compound tokens like P/E, D/E, year-over-year as single tokens
            if re.match(r"^[A-Z]{1,5}(/[A-Z]{1,5})?$", raw):
                # Looks like ticker or ratio abbreviation
                tokens.append(raw)
            elif "/" in raw and len(raw) <= 10:
                # Short ratio like P/E, EV/EBITDA — keep whole
                tokens.append(raw.lower())
            else:
                tokens.append(raw.lower())
        return tokens

    def detect_intent(self, query: str) -> QueryIntent:
        """Classify query into one of the QueryIntent categories.

        Each keyword carries an explicit weight so high-signal discriminators
        (e.g. 'vs', 'compare', 'chart') beat broad low-signal terms
        (e.g. 'revenue', 'earnings') even when the latter accumulate many hits.
        """
        q_lower = query.lower()
        scores: Dict[QueryIntent, int] = {intent: 0 for intent in QueryIntent}
        for intent, kw_weights in _INTENT_KEYWORDS.items():
            for kw, weight in kw_weights:
                if kw.lower() in q_lower:
                    scores[intent] += weight
        scores[QueryIntent.UNKNOWN] = 0
        best = max(scores, key=lambda i: scores[i])
        if scores[best] == 0:
            return QueryIntent.UNKNOWN
        return best

    def normalize_metric_name(self, name: str) -> str:
        """Return canonical snake_case metric name for any surface form."""
        canonical = self.ontology.canonical_metric(name)
        if canonical:
            return canonical
        # Fallback: slugify
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug

    def find_ticker(self, company_name: str) -> Optional[str]:
        """Find ticker for a company name via local lookup then EDGAR fallback."""
        local = self.ontology.resolve_ticker(company_name)
        if local:
            return local
        return _edgar_company_search(company_name)

    # ------------------------------------------------------------------
    # dim_056 additions
    # ------------------------------------------------------------------

    def compute_query_specificity(self, query: str) -> float:
        """Compute entropy of term distribution as a query specificity score.

        Higher entropy → broader, less specific query (many unique terms at low frequency).
        Lower entropy → more focused query (fewer, more repeated terms).

        H = -sum(p_i * log2(p_i))  where p_i = count(term_i) / total_terms

        Returns entropy in bits (float >= 0).
        """
        tokens = self.tokenize_financial(query)
        if not tokens:
            return 0.0
        counts: Counter = Counter(tokens)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 6)

    def build_concept_graph(self) -> Dict[str, List[str]]:
        """Build a dict mapping each financial concept to its 3-hop related terms.

        The graph is built from the ontology:
          - 1-hop: direct synonyms from METRIC_SYNONYMS and CONCEPT_SYNONYMS
          - 2-hop: synonyms of synonyms (terms sharing the same canonical)
          - 3-hop: concepts that appear together in CONCEPT_SYNONYMS lists

        Returns: {concept: [related_term_1, related_term_2, ...]}
        All concepts and metrics are included (deduplicated).
        """
        graph: Dict[str, List[str]] = {}

        # 1-hop: canonical → all its surface forms (direct synonyms)
        all_items: Dict[str, List[str]] = {}
        all_items.update(self.ontology.METRIC_SYNONYMS)
        all_items.update(self.ontology.CONCEPT_SYNONYMS)

        for canonical, aliases in all_items.items():
            related: List[str] = list(aliases[:8])  # up to 8 direct synonyms

            # 2-hop: other canonicals that share alias tokens
            canonical_tokens = set(canonical.lower().replace("_", " ").split())
            for other_canonical, other_aliases in all_items.items():
                if other_canonical == canonical:
                    continue
                other_tokens = set(other_canonical.lower().replace("_", " ").split())
                # Token overlap → related
                if canonical_tokens & other_tokens:
                    related.append(other_canonical.replace("_", " "))

            # 3-hop: concept_synonyms items that include any of our aliases
            our_alias_set = {a.lower() for a in aliases}
            for concept, concept_aliases in self.ontology.CONCEPT_SYNONYMS.items():
                if concept == canonical:
                    continue
                if any(ca.lower() in our_alias_set for ca in concept_aliases):
                    related.append(concept)

            # Deduplicate and exclude self
            seen: set = {canonical, canonical.replace("_", " ")}
            deduped: List[str] = []
            for r in related:
                if r not in seen:
                    seen.add(r)
                    deduped.append(r)

            graph[canonical] = deduped[:15]  # cap at 15 per node

        return graph

    def rank_expansion_terms(
        self,
        query: str,
        candidates: List[str],
        k1: float = 1.5,
    ) -> List[Tuple[str, float]]:
        """Rank candidate expansion terms using a BM25-style relevance formula.

        For each candidate term, score = tf × (k1 + 1) / (tf + k1)
        where tf = number of times the term (or its tokens) appear in the query.

        This is the BM25 term-frequency component without IDF and length normalization
        (both are fixed since we're scoring single terms against a single query).

        Returns list of (term, score) sorted descending by score.
        k1=1.5 is the standard BM25 default.
        """
        query_tokens = Counter(self.tokenize_financial(query))

        scored: List[Tuple[str, float]] = []
        for candidate in candidates:
            cand_tokens = self.tokenize_financial(candidate)
            if not cand_tokens:
                scored.append((candidate, 0.0))
                continue

            # tf = sum of query counts for all tokens in this candidate
            tf = sum(query_tokens.get(tok, 0) for tok in cand_tokens)

            # BM25 TF component
            bm25_score = tf * (k1 + 1) / (tf + k1)

            scored.append((candidate, round(bm25_score, 6)))

        # Sort descending; stable sort preserves insertion order for ties
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def expand(self, query: str) -> ExpandedQuery:
        """Expand a query into all related financial terms with confidence scoring."""
        intent = self.detect_intent(query)
        tokens = self.tokenize_financial(query)

        expanded_terms: List[str] = []
        synonyms_used: Dict[str, List[str]] = {}
        detected_tickers: List[str] = []

        # Detect explicit tickers (all-caps 2-5 chars)
        ticker_pattern = re.compile(r"\b([A-Z]{2,5})\b")
        for m in ticker_pattern.finditer(query):
            t = m.group(1)
            if t not in {"PE", "EV", "DD", "IV", "OR", "AND", "NOT", "BY",
                         "FOR", "IN", "OF", "TO", "AT", "VS", "YOY", "QOQ",
                         "FCF", "ROE", "ROA", "EPS", "TTM", "LTM", "NTM"}:
                detected_tickers.append(t)

        # Resolve company names in tokens
        for token in tokens:
            ticker = self.ontology.resolve_ticker(token)
            if ticker and ticker not in detected_tickers:
                detected_tickers.append(ticker)

        # Expand each token via ontology
        all_expanded: List[str] = list(tokens)
        for token in tokens:
            canonical = self.ontology.canonical_metric(token)
            if canonical:
                syns = self.ontology.get_synonyms(canonical)
                if syns:
                    synonyms_used[token] = syns[:4]  # top 4 synonyms
                    all_expanded.extend(syns[:4])
            # Check concept synonyms
            for concept, aliases in self.ontology.CONCEPT_SYNONYMS.items():
                if token == concept or token in [a.lower() for a in aliases]:
                    synonyms_used.setdefault(token, []).extend(aliases[:3])
                    all_expanded.extend(aliases[:3])

        # Deduplicate preserving order
        seen: set = set()
        for term in all_expanded:
            if term not in seen:
                seen.add(term)
                expanded_terms.append(term)

        # Build normalized query (replace abbreviations with canonical forms)
        normalized = query
        for abbr, canonical in self.ontology.METRIC_ABBREVIATIONS.items():
            pattern = re.compile(rf"\b{re.escape(abbr)}\b", re.IGNORECASE)
            normalized = pattern.sub(canonical.replace("_", " "), normalized)

        # Confidence: based on how many terms we recognized
        recognized = sum(1 for t in tokens if self.ontology.canonical_metric(t))
        confidence = min(0.95, 0.5 + (recognized / max(len(tokens), 1)) * 0.5)

        return ExpandedQuery(
            original=query,
            expanded_terms=expanded_terms,
            synonyms_used=synonyms_used,
            intent=intent,
            detected_tickers=detected_tickers,
            confidence=round(confidence, 3),
            normalized_query=normalized,
        )


# ---------------------------------------------------------------------------
# EDGAR Company Search
# ---------------------------------------------------------------------------

def _edgar_company_search(company_name: str) -> Optional[str]:
    """Search SEC EDGAR for a company name and return the best ticker match."""
    url = "https://efts.sec.gov/LATEST/search-index?q={}&dateRange=custom&startdt=2020-01-01&enddt=2026-12-31&forms=10-K"
    # Use the EDGAR company search endpoint (no key needed)
    search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "company": company_name,
        "CIK": "",
        "type": "10-K",
        "dateb": "",
        "owner": "include",
        "count": "5",
        "search_text": "",
        "action": "getcompany",
        "output": "atom",
    }
    try:
        resp = _SESSION.get(search_url, params=params, timeout=8)
        if resp.status_code != 200:
            return None
        # Parse Atom XML
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)
        if not entries:
            return None
        # First entry is usually the best match
        # <company-name> is in the content or title
        title = entries[0].find("atom:title", ns)
        if title is not None and title.text:
            # Extract ticker from CIK URL — try the category element
            category = entries[0].find("atom:category", ns)
            if category is not None:
                label = category.get("label", "")
                # label often contains the ticker
                m = re.search(r"\(([A-Z]{1,5})\)", label)
                if m:
                    return m.group(1)
    except Exception as exc:
        logger.debug("EDGAR company search failed for %r: %s", company_name, exc)
    return None


# ---------------------------------------------------------------------------
# TF-IDF Semantic Search
# ---------------------------------------------------------------------------

def _tokenize_simple(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser for TF-IDF."""
    return re.findall(r"[a-z0-9]+(?:[_\-][a-z0-9]+)*", text.lower())


class TFIDFIndex:
    """Minimal TF-IDF implementation with cosine similarity; no external deps."""

    def __init__(self) -> None:
        self._docs: List[str] = []
        self._tfidf: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}

    def fit(self, documents: List[str]) -> None:
        """Compute TF-IDF for a list of string documents."""
        self._docs = documents
        n = len(documents)
        # Term frequency per document
        tfs: List[Counter] = []
        for doc in documents:
            tokens = _tokenize_simple(doc)
            tf: Counter = Counter(tokens)
            total = max(sum(tf.values()), 1)
            tfs.append(Counter({t: c / total for t, c in tf.items()}))

        # Document frequency
        df: Counter = Counter()
        for tf in tfs:
            df.update(tf.keys())

        # IDF with smoothing
        self._idf = {
            term: math.log((n + 1) / (count + 1)) + 1
            for term, count in df.items()
        }

        # TF-IDF weighted vectors
        self._tfidf = [
            {term: tf_val * self._idf.get(term, 1.0) for term, tf_val in tf.items()}
            for tf in tfs
        ]

    def _query_vec(self, query: str) -> Dict[str, float]:
        tokens = _tokenize_simple(query)
        tf: Counter = Counter(tokens)
        total = max(sum(tf.values()), 1)
        return {
            term: (count / total) * self._idf.get(term, 1.0)
            for term, count in tf.items()
        }

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by relevance."""
        qvec = self._query_vec(query)
        scores = [
            (i, self._cosine(qvec, dvec))
            for i, dvec in enumerate(self._tfidf)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scores[:top_k] if s[1] > 0]


class SemanticSearch:
    """TF-IDF semantic search over a financial term corpus."""

    def __init__(self) -> None:
        self.ontology = FinancialOntology()
        self._index = TFIDFIndex()
        self._corpus: List[str] = []
        self._built = False

    def _build_corpus(self) -> List[str]:
        """Build a flat corpus from the ontology for TF-IDF fitting."""
        corpus: List[str] = []
        for canonical, aliases in self.ontology.METRIC_SYNONYMS.items():
            # Combine canonical + all aliases as one "document" per metric
            doc = canonical.replace("_", " ") + " " + " ".join(aliases)
            corpus.append(doc)
        for concept, aliases in self.ontology.CONCEPT_SYNONYMS.items():
            doc = concept + " " + " ".join(aliases)
            corpus.append(doc)
        # Add ticker company names
        for name, ticker in self.ontology.TICKER_ALIASES.items():
            corpus.append(f"{name} {ticker}")
        return corpus

    def _ensure_built(self) -> None:
        if not self._built:
            self._corpus = self._build_corpus()
            self._index.fit(self._corpus)
            self._built = True

    def search(self, query: str, candidates: List[str], top_k: int = 10) -> List[SearchResult]:
        """Search a custom list of candidates using TF-IDF cosine similarity."""
        if not candidates:
            return []
        index = TFIDFIndex()
        index.fit(candidates)
        results_raw = index.query(query, top_k=top_k)
        return [
            SearchResult(item=candidates[i], score=round(score, 4), rank=rank + 1)
            for rank, (i, score) in enumerate(results_raw)
        ]

    def rank_by_relevance(self, query: str, documents: List[dict]) -> List[dict]:
        """Rank a list of dicts by relevance to query, using 'text' or 'title' key."""
        if not documents:
            return documents
        texts = [
            d.get("text", d.get("title", d.get("content", str(d))))
            for d in documents
        ]
        index = TFIDFIndex()
        index.fit(texts)
        results_raw = index.query(query, top_k=len(documents))
        scored = [(i, score) for i, score in results_raw]
        # Append unscored docs at end
        scored_indices = {i for i, _ in scored}
        for i in range(len(documents)):
            if i not in scored_indices:
                scored.append((i, 0.0))
        ranked = []
        for i, score in scored:
            doc = dict(documents[i])
            doc["_relevance_score"] = round(score, 4)
            ranked.append(doc)
        return ranked

    def search_ontology(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search the built-in ontology corpus."""
        self._ensure_built()
        results_raw = self._index.query(query, top_k=top_k)
        return [
            SearchResult(item=self._corpus[i], score=round(score, 4), rank=rank + 1)
            for rank, (i, score) in enumerate(results_raw)
        ]


# ---------------------------------------------------------------------------
# Trie for Autocomplete
# ---------------------------------------------------------------------------

class _TrieNode:
    __slots__ = ("children", "is_terminal", "payload")

    def __init__(self) -> None:
        self.children: Dict[str, "_TrieNode"] = {}
        self.is_terminal: bool = False
        self.payload: Optional[Dict] = None


class _Trie:
    def __init__(self) -> None:
        self.root = _TrieNode()

    def insert(self, word: str, payload: Optional[Dict] = None) -> None:
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.is_terminal = True
        if payload:
            node.payload = payload

    def _collect(self, node: _TrieNode, prefix: str, results: List[Tuple[str, Optional[Dict]]]) -> None:
        if node.is_terminal:
            results.append((prefix, node.payload))
        for ch, child in sorted(node.children.items()):
            self._collect(child, prefix + ch, results)

    def search_prefix(self, prefix: str, max_results: int = 20) -> List[Tuple[str, Optional[Dict]]]:
        node = self.root
        for ch in prefix.lower():
            if ch not in node.children:
                return []
            node = node.children[ch]
        results: List[Tuple[str, Optional[Dict]]] = []
        self._collect(node, prefix.lower(), results)
        return results[:max_results]


# ---------------------------------------------------------------------------
# AutocompleteEngine
# ---------------------------------------------------------------------------

_TIMEFRAME_TERMS = [
    "1d", "5d", "1m", "3m", "6m", "1y", "2y", "5y", "10y", "ytd",
    "intraday", "daily", "weekly", "monthly", "quarterly", "annual",
    "trailing twelve months", "TTM", "LTM", "NTM",
]

_OPERATOR_TERMS = [
    "greater than", "less than", "equal to", "between", "above", "below",
    "at least", "at most", "not equal to",
]

_CONTEXT_METRICS: Dict[QueryIntent, List[str]] = {
    QueryIntent.SCREENING: [
        "pe_ratio", "ev_ebitda", "revenue_growth_yoy", "return_on_equity",
        "debt_to_equity", "free_cash_flow", "gross_margin", "net_margin",
        "market_cap", "eps_growth_yoy", "dividend_yield",
    ],
    QueryIntent.CHARTING: _TIMEFRAME_TERMS,
    QueryIntent.FUNDAMENTAL: [
        "total_revenue", "net_income", "ebitda", "eps", "operating_cash_flow",
        "capex", "shareholders_equity", "total_debt", "gross_profit",
    ],
    QueryIntent.MACRO: [
        "gdp_growth", "inflation", "interest_rate", "unemployment",
        "yield_to_maturity", "credit_spread",
    ],
    QueryIntent.SENTIMENT: [
        "analyst_rating", "price_target", "earnings_surprise",
        "esg_score", "put_call_ratio",
    ],
    QueryIntent.COMPARISON: [
        "pe_ratio", "ev_ebitda", "revenue_growth_yoy", "return_on_equity",
        "gross_margin",
    ],
}


class AutocompleteEngine:
    """Prefix-trie autocomplete with context-aware suggestion ordering."""

    def __init__(self) -> None:
        self.ontology = FinancialOntology()
        self._trie = _Trie()
        self._built = False

    def _build(self) -> None:
        if self._built:
            return
        # Insert all metrics
        for canonical, aliases in self.ontology.METRIC_SYNONYMS.items():
            self._trie.insert(canonical.replace("_", " "),
                              {"type": "metric", "canonical": canonical, "score": 1.0})
            for alias in aliases:
                self._trie.insert(alias,
                                  {"type": "metric", "canonical": canonical, "score": 0.8})
        # Insert abbreviations
        for abbr, canonical in self.ontology.METRIC_ABBREVIATIONS.items():
            self._trie.insert(abbr, {"type": "metric", "canonical": canonical, "score": 0.9})
        # Insert tickers and company names
        for name, ticker in self.ontology.TICKER_ALIASES.items():
            self._trie.insert(ticker, {"type": "ticker", "name": name, "score": 1.0})
            self._trie.insert(name, {"type": "ticker", "ticker": ticker, "score": 0.85})
        # Insert concepts
        for concept in self.ontology.CONCEPT_SYNONYMS:
            self._trie.insert(concept, {"type": "concept", "score": 0.7})
        # Insert timeframes
        for tf in _TIMEFRAME_TERMS:
            self._trie.insert(tf, {"type": "timeframe", "score": 0.6})
        # Insert operators
        for op in _OPERATOR_TERMS:
            self._trie.insert(op, {"type": "operator", "score": 0.5})
        self._built = True

    def suggest(
        self,
        prefix: str,
        context: QueryIntent = QueryIntent.UNKNOWN,
        max_suggestions: int = 10,
    ) -> List[Suggestion]:
        """Return context-aware suggestions for a given prefix."""
        self._build()
        raw = self._trie.search_prefix(prefix, max_results=50)
        suggestions: List[Suggestion] = []
        context_boost = set(_CONTEXT_METRICS.get(context, []))

        for term, payload in raw:
            if payload is None:
                continue
            item_type = payload.get("type", "unknown")
            base_score = payload.get("score", 0.5)

            # Context boost
            canonical = payload.get("canonical", term)
            if canonical in context_boost or term in context_boost:
                base_score = min(1.0, base_score + 0.15)

            desc = ""
            if item_type == "metric":
                desc = f"Metric: {canonical.replace('_', ' ')}"
            elif item_type == "ticker":
                ticker = payload.get("ticker", "")
                name = payload.get("name", "")
                desc = f"Ticker: {ticker or term.upper()} — {name}"
            elif item_type == "concept":
                desc = f"Concept: {term}"
            elif item_type == "timeframe":
                desc = f"Timeframe: {term}"
            elif item_type == "operator":
                desc = f"Operator: {term}"

            suggestions.append(Suggestion(
                text=term,
                type=item_type,
                score=round(base_score, 3),
                description=desc,
            ))

        # Sort: context-boosted first, then by score, then alphabetically
        suggestions.sort(key=lambda s: (-s.score, s.text))
        return suggestions[:max_suggestions]


# ---------------------------------------------------------------------------
# QueryParser — NL → StructuredQuery
# ---------------------------------------------------------------------------

# Operator patterns
_OP_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:greater than or equal to|>=|≥|at least)\s*", "gte"),
    (r"(?:less than or equal to|<=|≤|at most)\s*", "lte"),
    (r"(?:greater than|>|above|over|more than)\s*", "gt"),
    (r"(?:less than|<|below|under|fewer than)\s*", "lt"),
    (r"(?:equal to|=|equals|is)\s*", "eq"),
    (r"(?:not equal to|!=|<>|is not)\s*", "neq"),
    (r"between\s*", "between"),
]

# Number parsing: handles "20", "20%", "1.5B", "500M", "0.10"
_NUMBER_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([MBKT]|billion|million|trillion|thousand|%|percent)?",
    re.IGNORECASE,
)

_SECTOR_KEYWORDS: Dict[str, str] = {
    "tech": "technology", "technology": "technology",
    "software": "software", "semiconductor": "semiconductors",
    "chip": "semiconductors", "financial": "financial",
    "bank": "banking", "health": "healthcare", "pharma": "pharmaceuticals",
    "biotech": "biotech", "energy": "energy", "oil": "oil gas",
    "utility": "utilities", "retail": "retail", "consumer": "consumer discretionary",
    "industrial": "industrials", "defense": "aerospace defense",
    "transport": "transportation", "real estate": "real estate", "reit": "real estate",
    "telecom": "telecom", "media": "media", "mining": "materials",
}


def _parse_number(s: str) -> Optional[float]:
    """Parse financial number string like '20%', '1.5B', '500M'."""
    m = _NUMBER_PATTERN.match(s.strip())
    if not m:
        return None
    val = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    if suffix in ("B", "BILLION"):
        val *= 1e9
    elif suffix in ("M", "MILLION"):
        val *= 1e6
    elif suffix in ("T", "TRILLION"):
        val *= 1e12
    elif suffix in ("K", "THOUSAND"):
        val *= 1e3
    elif suffix in ("%", "PERCENT"):
        val /= 100.0
    return val


def _detect_metric_in_phrase(phrase: str, ontology: FinancialOntology) -> Optional[str]:
    """Try to match a multi-word phrase to a canonical metric."""
    # Try progressively shorter sub-phrases (greedy longest match)
    words = phrase.lower().split()
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            candidate = " ".join(words[start: start + length])
            canonical = ontology.canonical_metric(candidate)
            if canonical:
                return canonical
    return None


class QueryParser:
    """Parse natural language financial queries into StructuredQuery."""

    def __init__(self) -> None:
        self.ontology = FinancialOntology()
        self.expander = QueryExpander()

    def parse(self, query: str) -> StructuredQuery:
        """Main entry point: parse NL query into StructuredQuery."""
        intent = self.expander.detect_intent(query)
        tickers = self._extract_tickers(query)
        universe = self._extract_universe(query)
        time_period = self._extract_time_period(query)
        filters = self._extract_filters(query)
        normalized = self._normalize(query)

        return StructuredQuery(
            filters=filters,
            universe=universe,
            intent=intent,
            tickers=tickers,
            time_period=time_period,
            raw=query,
            normalized=normalized,
        )

    def _extract_tickers(self, query: str) -> List[str]:
        """Extract tickers: explicit UPPER tokens + company name resolution."""
        tickers: List[str] = []
        seen: set = set()
        # Explicit tickers
        for m in re.finditer(r"\b([A-Z]{2,5})\b", query):
            t = m.group(1)
            if t not in {"PE", "EV", "DD", "IV", "OR", "AND", "IN", "TO",
                         "FCF", "ROE", "ROA", "EPS", "TTM", "LTM", "NTM",
                         "ESG", "CEO", "IPO", "YOY", "QOQ", "YTD", "USD",
                         "GDP", "CPI", "PCE", "NFP", "FED", "SEC", "EPS"}:
                if t not in seen:
                    tickers.append(t)
                    seen.add(t)
        # Company name resolution
        q_lower = query.lower()
        for name, ticker in self.ontology.TICKER_ALIASES.items():
            if name in q_lower and ticker not in seen:
                tickers.append(ticker)
                seen.add(ticker)
        return tickers

    def _extract_universe(self, query: str) -> Optional[str]:
        """Detect sector/universe from query text."""
        q_lower = query.lower()
        for keyword, sector in _SECTOR_KEYWORDS.items():
            if keyword in q_lower:
                return sector
        return None

    def _extract_time_period(self, query: str) -> Optional[str]:
        """Extract time period references."""
        patterns = [
            (r"\blast\s+(\d+)\s+years?\b", lambda m: f"{m.group(1)}y"),
            (r"\blast\s+(\d+)\s+quarters?\b", lambda m: f"{m.group(1)}q"),
            (r"\byear\s+over\s+year\b", lambda _: "yoy"),
            (r"\bttm\b|\btrailing\s+twelve\s+months\b", lambda _: "ttm"),
            (r"\bltm\b|\blast\s+twelve\s+months\b", lambda _: "ltm"),
            (r"\bntm\b|\bnext\s+twelve\s+months\b", lambda _: "ntm"),
            (r"\bq[1-4]\s+\d{4}\b", lambda m: m.group(0)),
            (r"\bfy\d{4}\b|\bfiscal\s+year\s+\d{4}\b", lambda m: m.group(0)),
        ]
        q_lower = query.lower()
        for pattern, formatter in patterns:
            m = re.search(pattern, q_lower)
            if m:
                try:
                    return formatter(m)
                except Exception:
                    return m.group(0)
        return None

    def _extract_filters(self, query: str) -> List[Filter]:
        """Extract Filter(metric, op, value) from query text."""
        filters: List[Filter] = []
        q = query.lower()

        # Handle "between X and Y" for metric
        between_pattern = re.compile(
            r"([\w\s/]+?)\s+between\s+(-?\d+(?:\.\d+)?(?:\s*[MBKT%])?)\s+and\s+(-?\d+(?:\.\d+)?(?:\s*[MBKT%])?)",
            re.IGNORECASE,
        )
        for m in between_pattern.finditer(q):
            metric_phrase = m.group(1).strip()
            val1_str = m.group(2).strip()
            val2_str = m.group(3).strip()
            canonical = _detect_metric_in_phrase(metric_phrase, self.ontology)
            if canonical:
                v1 = _parse_number(val1_str)
                v2 = _parse_number(val2_str)
                if v1 is not None and v2 is not None:
                    filters.append(Filter(
                        metric=canonical, op="between",
                        value=min(v1, v2), value2=max(v1, v2),
                        raw=m.group(0),
                    ))

        # Handle "metric OP value" patterns
        op_segment_pattern = re.compile(
            r"([\w\s/]+?)\s*"
            r"(>=|<=|>|<|=|!=|"
            r"(?:greater than or equal to|less than or equal to|"
            r"greater than|less than|equal to|not equal to|"
            r"above|below|over|under|at least|at most|more than|fewer than))"
            r"\s*(-?\d+(?:\.\d+)?(?:\s*[MBKT%b](?:illion|illion)?)?)",
            re.IGNORECASE,
        )
        for m in op_segment_pattern.finditer(q):
            metric_phrase = m.group(1).strip()
            op_text = m.group(2).strip().lower()
            val_str = m.group(3).strip()

            canonical = _detect_metric_in_phrase(metric_phrase, self.ontology)
            if not canonical:
                continue

            # Map op_text to op code
            op_code = "eq"
            op_map = {
                ">=": "gte", "greater than or equal to": "gte", "at least": "gte",
                "<=": "lte", "less than or equal to": "lte", "at most": "lte",
                ">": "gt", "greater than": "gt", "above": "gt", "over": "gt", "more than": "gt",
                "<": "lt", "less than": "lt", "below": "lt", "under": "lt", "fewer than": "lt",
                "=": "eq", "equal to": "eq",
                "!=": "neq", "not equal to": "neq",
            }
            for op_str, code in op_map.items():
                if op_text == op_str or op_text.startswith(op_str):
                    op_code = code
                    break

            val = _parse_number(val_str)
            if val is None:
                continue

            # Check for duplicate (same metric already added via "between")
            already = any(f.metric == canonical for f in filters)
            if not already:
                filters.append(Filter(
                    metric=canonical, op=op_code, value=val, raw=m.group(0),
                ))

        return filters

    def _normalize(self, query: str) -> str:
        """Return a cleaned, normalized form of the query."""
        q = query.strip()
        # Expand abbreviations
        for abbr, canonical in self.ontology.METRIC_ABBREVIATIONS.items():
            q = re.sub(rf"\b{re.escape(abbr)}\b", canonical.replace("_", " "), q, flags=re.IGNORECASE)
        return q


# ---------------------------------------------------------------------------
# Convenience facade
# ---------------------------------------------------------------------------

class FinancialNLPEngine:
    """Unified facade combining expand, parse, search, and autocomplete."""

    def __init__(self) -> None:
        self.ontology = FinancialOntology()
        self.expander = QueryExpander()
        self.parser = QueryParser()
        self.search = SemanticSearch()
        self.autocomplete = AutocompleteEngine()

    def process(self, query: str) -> Dict[str, Any]:
        """Full pipeline: expand + parse + ontology search."""
        expanded = self.expander.expand(query)
        structured = self.parser.parse(query)
        ontology_hits = self.search.search_ontology(query, top_k=5)
        suggestions = self.autocomplete.suggest(
            query.split()[-1] if query.split() else "",
            context=expanded.intent,
            max_suggestions=5,
        )
        return {
            "expanded": expanded,
            "structured": structured,
            "ontology_hits": ontology_hits,
            "suggestions": suggestions,
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("SENTINEL Query Expander V3 — Demo")
    print("=" * 70)

    engine = FinancialNLPEngine()

    demo_queries = [
        "show me tech stocks with PE under 20 and revenue growth > 10%",
        "AAPL earnings last quarter vs MSFT",
        "find companies with FCF yield above 5% and low debt to equity",
        "nvidia options implied volatility vs historical",
        "macro outlook: inflation, fed rate, and yield curve",
        "WSB high short interest stocks with momentum",
        "Apple revenue growth year over year",
        "EV/EBITDA between 5 and 15 for healthcare sector",
    ]

    expander = QueryExpander()
    parser = QueryParser()
    autocomplete = AutocompleteEngine()

    for q in demo_queries:
        print(f"\nQuery: {q!r}")
        print("-" * 60)

        # Expand
        eq = expander.expand(q)
        print(f"  Intent:   {eq.intent.value}")
        print(f"  Tickers:  {eq.detected_tickers}")
        print(f"  Confidence: {eq.confidence}")
        print(f"  Synonyms used: {list(eq.synonyms_used.keys())[:4]}")

        # Parse
        sq = parser.parse(q)
        print(f"  Universe: {sq.universe}")
        print(f"  Filters:")
        for f in sq.filters:
            val2_str = f" and {f.value2}" if f.value2 is not None else ""
            print(f"    {f.metric} {f.op} {f.value}{val2_str}")
        print(f"  Time period: {sq.time_period}")

    # Autocomplete demo
    print("\n" + "=" * 70)
    print("Autocomplete demo:")
    for prefix, ctx in [("rev", QueryIntent.SCREENING), ("earn", QueryIntent.FUNDAMENTAL),
                         ("AAPL", QueryIntent.CHARTING), ("pe", QueryIntent.SCREENING)]:
        suggs = autocomplete.suggest(prefix, context=ctx, max_suggestions=5)
        print(f"\n  Prefix={prefix!r} ctx={ctx.value}:")
        for s in suggs:
            print(f"    [{s.type:10s}] {s.text!r:35s} score={s.score}  {s.description}")

    # Semantic search demo
    print("\n" + "=" * 70)
    print("Semantic search demo:")
    searcher = SemanticSearch()
    candidates = [
        "PE ratio price to earnings valuation",
        "revenue sales top line growth",
        "return on equity ROE profitability",
        "debt leverage gearing balance sheet",
        "free cash flow FCF yield",
        "implied volatility options pricing",
    ]
    results = searcher.search("cheap stocks with good cash generation", candidates, top_k=3)
    for r in results:
        print(f"  Rank {r.rank}: score={r.score:.4f}  {r.item!r}")

    print("\nDone.")
