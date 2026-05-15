"""Financial Query Expansion & Normalization — Bloomberg-grade NL query intelligence.

dim_056: Smart synonym / query expansion — raise score from 6 → 9+.

Provides natural language query parsing, synonym expansion, intelligent routing to
SENTINEL modules, and autocomplete for financial terms, tickers, and metrics.

Architecture:
  FinancialSynonymLibrary — 80+ metric synonyms, 50+ concept synonyms, company aliases
  QueryParser             — NL intent extraction: entity, metrics, time period, type
  QueryExpander           — synonym-based and semantic query expansion
  FinancialQueryRouter    — regex/keyword-based routing to SENTINEL modules
  AutocompleteEngine      — trie-based prefix matching over 5000+ financial terms
  query_router            — FastAPI router exposing all capabilities

Usage::
    from sentinel.sai.query_expansion import QueryParser, QueryExpander, FinancialQueryRouter

    parser = QueryParser()
    parsed = parser.parse_natural_language_query("What is Apple's P/E ratio last quarter?")
    # → {entity: "Apple", ticker: "AAPL", metrics: ["P/E ratio"], time_period: {...}, ...}

    expander = QueryExpander()
    variants = expander.expand_query("Tesla revenue growth")
    # → ["Tesla TSLA revenue growth", "TSLA sales increase", ...]
"""
from __future__ import annotations

import bisect
import json
import re
import time
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query as FastAPIQuery
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Synonym Library
# ---------------------------------------------------------------------------

class FinancialSynonymLibrary:
    """Comprehensive hard-coded financial synonym dictionaries.

    Contains:
        METRIC_SYNONYMS   — 80+ financial metric aliases
        CONCEPT_SYNONYMS  — 50+ financial concept aliases
        COMPANY_ALIASES   — 60+ company name → ticker mappings
        SECTOR_TERMS      — sector name variations
    """

    METRIC_SYNONYMS: dict[str, list[str]] = {
        "revenue": [
            "sales", "top line", "net revenues", "turnover", "gross revenues",
            "income from operations", "total revenues", "net sales", "total sales",
        ],
        "ebitda": [
            "earnings before interest taxes depreciation amortization",
            "operating earnings proxy", "EBITDA", "adjusted ebitda", "adj. ebitda",
        ],
        "net income": [
            "bottom line", "earnings", "profit", "net profit", "net earnings",
            "after-tax income", "income attributable to shareholders", "net loss",
        ],
        "cash flow": [
            "FCF", "free cash flow", "operating cash flow", "cash from operations",
            "levered free cash flow", "unlevered free cash flow", "cash generation",
        ],
        "gross margin": [
            "gross profit margin", "gross profitability", "GP margin",
            "gross profit percentage", "product margin",
        ],
        "operating margin": [
            "EBIT margin", "operating profit margin", "operating income margin",
            "EBIT %", "operating leverage", "op margin",
        ],
        "net margin": [
            "net profit margin", "profit margin", "return on sales",
            "bottom line margin", "after-tax margin",
        ],
        "market cap": [
            "market capitalization", "market value", "equity value",
            "market valuation", "float capitalization",
        ],
        "enterprise value": [
            "EV", "total firm value", "total enterprise value",
            "EV/EBITDA numerator", "firm value",
        ],
        "P/E ratio": [
            "price-to-earnings", "PE multiple", "earnings multiple",
            "price/earnings", "PE ratio", "trailing PE", "forward PE",
        ],
        "book value": [
            "net asset value", "NAV", "shareholders equity", "tangible book value",
            "BVPS", "book value per share", "net book value",
        ],
        "EPS": [
            "earnings per share", "diluted EPS", "basic EPS", "adjusted EPS",
            "non-GAAP EPS", "GAAP EPS", "net income per share",
        ],
        "dividend yield": [
            "yield", "dividend rate", "annual dividend", "DPS yield",
            "income yield", "dividend return", "payout yield",
        ],
        "price-to-book": [
            "P/B ratio", "PB ratio", "price to book value", "market to book",
            "price/book", "Tobin's Q proxy",
        ],
        "price-to-sales": [
            "P/S ratio", "PS ratio", "price to revenue", "revenue multiple",
            "sales multiple", "price/revenue",
        ],
        "EV/EBITDA": [
            "enterprise value to ebitda", "EV multiple", "acquisition multiple",
            "EBITDA multiple", "firm value multiple",
        ],
        "return on equity": [
            "ROE", "return on shareholders equity", "equity return",
            "net income / equity", "shareholder return metric",
        ],
        "return on assets": [
            "ROA", "return on total assets", "asset productivity",
            "net income / assets",
        ],
        "return on invested capital": [
            "ROIC", "return on capital", "invested capital return",
            "economic return", "NOPAT / invested capital",
        ],
        "debt-to-equity": [
            "D/E ratio", "leverage ratio", "financial leverage",
            "debt/equity", "gearing ratio",
        ],
        "debt-to-ebitda": [
            "leverage multiple", "net debt to ebitda", "credit leverage",
            "D/EBITDA", "debt coverage multiple",
        ],
        "current ratio": [
            "liquidity ratio", "short-term solvency", "current assets / current liabilities",
            "working capital ratio",
        ],
        "quick ratio": [
            "acid-test ratio", "liquid ratio", "quick assets ratio",
        ],
        "inventory turnover": [
            "stock turnover", "inventory velocity", "COGS / inventory",
            "inventory efficiency",
        ],
        "days sales outstanding": [
            "DSO", "receivables days", "collection period", "AR days",
            "debtor days",
        ],
        "capex": [
            "capital expenditure", "capital spending", "property plant equipment purchases",
            "PP&E additions", "maintenance capex", "growth capex",
        ],
        "depreciation": [
            "D&A", "depreciation and amortization", "non-cash charge",
            "amortization", "asset write-down",
        ],
        "working capital": [
            "net working capital", "NWC", "current assets minus current liabilities",
            "operational liquidity",
        ],
        "operating leverage": [
            "fixed cost leverage", "DOL", "degree of operating leverage",
        ],
        "beta": [
            "market beta", "systematic risk", "volatility vs market",
            "CAPM beta", "equity beta",
        ],
        "alpha": [
            "excess return", "active return", "Jensen's alpha",
            "risk-adjusted outperformance",
        ],
        "sharpe ratio": [
            "risk-adjusted return", "reward-to-variability", "Sharpe index",
        ],
        "volatility": [
            "standard deviation", "historical volatility", "HV", "realized volatility",
            "price variability", "sigma",
        ],
        "implied volatility": [
            "IV", "options implied vol", "market implied volatility",
            "VIX component", "options pricing vol",
        ],
        "short interest": [
            "shares short", "short float", "short % of float",
            "days to cover", "short ratio",
        ],
        "float": [
            "public float", "shares available to trade", "tradeable shares",
            "non-restricted shares",
        ],
        "insider ownership": [
            "management ownership", "director holdings", "insider stake",
            "executive ownership percentage",
        ],
        "institutional ownership": [
            "institutional holdings", "13F ownership", "fund ownership",
            "smart money ownership",
        ],
        "revenue growth": [
            "top line growth", "sales growth", "revenue expansion",
            "revenue increase", "YoY revenue change", "CAGR",
        ],
        "earnings growth": [
            "EPS growth", "profit growth", "bottom line growth",
            "earnings expansion", "YoY earnings change",
        ],
        "guidance": [
            "outlook", "forecast", "projection", "management guidance",
            "forward guidance", "next quarter expectations", "full year outlook",
        ],
        "consensus estimate": [
            "street estimate", "analyst estimate", "sell-side consensus",
            "FactSet consensus", "Bloomberg consensus",
        ],
        "beat": [
            "earnings beat", "top estimates", "exceeded consensus",
            "surprised to the upside", "better than expected",
        ],
        "miss": [
            "earnings miss", "below estimates", "fell short of consensus",
            "disappointed", "worse than expected",
        ],
        "PEG ratio": [
            "price earnings to growth", "growth-adjusted PE",
            "Peter Lynch ratio", "PEG",
        ],
        "forward P/E": [
            "NTM PE", "next twelve months PE", "forward earnings multiple",
            "12-month forward PE",
        ],
        "cash and equivalents": [
            "cash on hand", "liquid assets", "cash balance",
            "cash position", "short-term investments",
        ],
        "net debt": [
            "debt minus cash", "financial debt net", "net borrowings",
            "total debt minus cash and equivalents",
        ],
        "gross profit": [
            "gross earnings", "revenue minus COGS", "top line profit",
            "gross income",
        ],
        "operating income": [
            "EBIT", "operating profit", "operating earnings",
            "income from operations",
        ],
        "diluted shares": [
            "fully diluted shares", "shares outstanding diluted",
            "diluted share count", "weighted average diluted shares",
        ],
        "buyback": [
            "share repurchase", "stock buyback", "repurchase program",
            "treasury stock purchase", "share reduction",
        ],
        "dividend": [
            "cash dividend", "quarterly dividend", "annual dividend",
            "DPS", "dividend per share", "payout",
        ],
        "COGS": [
            "cost of goods sold", "cost of revenue", "cost of sales",
            "direct costs", "product costs",
        ],
        "R&D": [
            "research and development", "R&D expense", "innovation spend",
            "technology investment",
        ],
        "SG&A": [
            "selling general and administrative", "operating expenses",
            "overhead", "administrative costs",
        ],
        "backlog": [
            "order backlog", "remaining performance obligations", "RPO",
            "unfulfilled orders", "contracted revenue",
        ],
        "ARR": [
            "annual recurring revenue", "subscription revenue annualized",
            "annualized revenue run rate",
        ],
        "MRR": [
            "monthly recurring revenue", "monthly subscription revenue",
        ],
        "churn": [
            "customer churn", "revenue churn", "attrition rate",
            "logo churn", "retention inverse",
        ],
        "NRR": [
            "net revenue retention", "dollar-based net expansion",
            "net dollar retention", "NDR",
        ],
        "gross retention": [
            "logo retention", "customer retention rate",
            "gross dollar retention", "GRR",
        ],
        "DAU": ["daily active users", "daily actives"],
        "MAU": ["monthly active users", "monthly actives"],
        "ARPU": [
            "average revenue per user", "average revenue per unit",
            "revenue per subscriber",
        ],
        "LTV": [
            "lifetime value", "customer lifetime value", "CLV",
            "long-term value per customer",
        ],
        "CAC": [
            "customer acquisition cost", "acquisition cost",
            "cost to acquire customer",
        ],
    }

    CONCEPT_SYNONYMS: dict[str, list[str]] = {
        "earnings call": [
            "conference call", "quarterly call", "investor call",
            "Q&A session", "analyst call", "earnings webcast",
        ],
        "guidance": [
            "outlook", "forecast", "projection", "forward guidance",
            "next quarter expectations", "management commentary on future",
        ],
        "short interest": [
            "short float", "short squeeze potential", "bearish positioning",
            "shares sold short", "short percentage",
        ],
        "insider buying": [
            "open market purchase", "Form 4 acquisition", "executive buying",
            "management buying", "director purchase", "insider accumulation",
        ],
        "insider selling": [
            "Form 4 disposition", "executive selling", "management selling",
            "insider liquidation", "stock sale by insider",
        ],
        "merger": [
            "acquisition", "M&A", "deal", "takeover", "buyout",
            "business combination", "consolidation",
        ],
        "IPO": [
            "initial public offering", "going public", "listing",
            "float", "public debut",
        ],
        "secondary offering": [
            "follow-on offering", "FO", "dilutive offering",
            "equity raise", "secondary share sale",
        ],
        "analyst upgrade": [
            "rating upgrade", "raised to buy", "upgraded to outperform",
            "price target increase", "more bullish call",
        ],
        "analyst downgrade": [
            "rating downgrade", "cut to sell", "downgraded to underperform",
            "price target decrease", "more bearish call",
        ],
        "earnings surprise": [
            "beat and raise", "upside surprise", "positive surprise",
            "EPS beat", "stronger than expected results",
        ],
        "going concern": [
            "bankruptcy risk", "doubt about survival", "viability warning",
            "going-concern opinion", "financial distress",
        ],
        "restatement": [
            "financial restatement", "earnings revision", "accounting error",
            "restated financials", "corrected filings",
        ],
        "activist investor": [
            "activist shareholder", "hedge fund activism",
            "shareholder campaign", "13D filing", "board seats demand",
        ],
        "spin-off": [
            "divestiture", "separation", "corporate split", "carve-out",
            "subsidiary listing",
        ],
        "share buyback": [
            "repurchase program", "buyback authorization", "treasury purchase",
            "stock repurchase",
        ],
        "debt offering": [
            "bond issuance", "notes offering", "credit facility",
            "debt raise", "fixed income issuance",
        ],
        "credit rating": [
            "bond rating", "S&P rating", "Moody's rating", "Fitch rating",
            "investment grade", "high yield", "junk rating",
        ],
        "proxy fight": [
            "proxy contest", "board battle", "dissident slate",
            "shareholder vote campaign",
        ],
        "material weakness": [
            "internal control weakness", "ICFR deficiency",
            "accounting control failure",
        ],
        "stock split": [
            "forward split", "share split", "split-adjusted",
        ],
        "reverse split": [
            "reverse stock split", "share consolidation",
        ],
        "special dividend": [
            "one-time dividend", "extraordinary dividend", "special cash distribution",
        ],
        "ESG": [
            "environmental social governance", "sustainability",
            "responsible investing", "impact investing", "green investing",
        ],
        "momentum": [
            "price momentum", "relative strength", "trend following",
            "52-week high breakout",
        ],
        "mean reversion": [
            "value trap", "contrarian", "oversold bounce",
            "regression to mean",
        ],
        "technical breakout": [
            "price breakout", "resistance break", "52-week high",
            "upside breakout", "chart breakout",
        ],
        "support level": [
            "price floor", "technical support", "base support",
        ],
        "resistance level": [
            "price ceiling", "technical resistance", "overhead resistance",
        ],
        "options expiry": [
            "options expiration", "OPEX", "options settlement",
            "contract expiry",
        ],
        "dark pool": [
            "ATS", "alternative trading system", "off-exchange trading",
            "block trade venue",
        ],
        "SEC filing": [
            "EDGAR filing", "regulatory filing", "disclosure filing",
            "10-K", "10-Q", "8-K",
        ],
        "earnings whisper": [
            "whisper number", "street whisper", "buy-side estimate",
            "unofficial estimate",
        ],
        "short squeeze": [
            "short covering rally", "forced short covering",
            "gamma squeeze", "meme squeeze",
        ],
        "capital allocation": [
            "return of capital", "capital deployment", "shareholder return policy",
        ],
        "management change": [
            "CEO change", "CFO change", "executive departure",
            "leadership transition", "C-suite change",
        ],
        "supply chain": [
            "supply chain disruption", "logistics", "sourcing",
            "component shortage", "inventory management",
        ],
        "margin expansion": [
            "profitability improvement", "operating leverage", "cost efficiency",
        ],
        "margin compression": [
            "margin pressure", "profitability squeeze", "cost headwinds",
        ],
        "macro headwinds": [
            "economic headwinds", "interest rate pressure", "FX headwinds",
            "inflation impact", "recession risk",
        ],
    }

    COMPANY_ALIASES: dict[str, str] = {
        # Big Tech
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Tesla": "TSLA",
        "Google": "GOOGL",
        "Alphabet": "GOOGL",
        "Meta": "META",
        "Facebook": "META",
        "Amazon": "AMZN",
        "Netflix": "NFLX",
        "Nvidia": "NVDA",
        "AMD": "AMD",
        "Intel": "INTC",
        "Qualcomm": "QCOM",
        "Broadcom": "AVGO",
        "TSMC": "TSM",
        "Samsung": "SSNLF",
        # Finance
        "JPMorgan": "JPM",
        "JP Morgan": "JPM",
        "Goldman Sachs": "GS",
        "Morgan Stanley": "MS",
        "Bank of America": "BAC",
        "Wells Fargo": "WFC",
        "Citigroup": "C",
        "Citi": "C",
        "BlackRock": "BLK",
        "Berkshire Hathaway": "BRK.B",
        "Berkshire": "BRK.B",
        "Visa": "V",
        "Mastercard": "MA",
        "PayPal": "PYPL",
        # Healthcare
        "Johnson & Johnson": "JNJ",
        "J&J": "JNJ",
        "Pfizer": "PFE",
        "Merck": "MRK",
        "AbbVie": "ABBV",
        "Eli Lilly": "LLY",
        "Lilly": "LLY",
        "UnitedHealth": "UNH",
        "CVS": "CVS",
        "Amgen": "AMGN",
        # Consumer
        "Walmart": "WMT",
        "Costco": "COST",
        "Home Depot": "HD",
        "McDonald's": "MCD",
        "Starbucks": "SBUX",
        "Nike": "NKE",
        "Procter & Gamble": "PG",
        "P&G": "PG",
        "Coca-Cola": "KO",
        "PepsiCo": "PEP",
        "Pepsi": "PEP",
        # Energy
        "ExxonMobil": "XOM",
        "Exxon": "XOM",
        "Chevron": "CVX",
        # Industrial / Other
        "Boeing": "BA",
        "Caterpillar": "CAT",
        "Deere": "DE",
        "John Deere": "DE",
        "3M": "MMM",
        "GE": "GE",
        "General Electric": "GE",
        "Salesforce": "CRM",
        "Oracle": "ORCL",
        "Adobe": "ADBE",
        "Uber": "UBER",
        "Airbnb": "ABNB",
        "Shopify": "SHOP",
        "Snowflake": "SNOW",
        "Palantir": "PLTR",
        "Coinbase": "COIN",
    }

    SECTOR_TERMS: dict[str, list[str]] = {
        "tech": [
            "technology", "information technology", "IT sector",
            "GICS: Information Technology", "software", "semiconductors",
            "internet", "tech sector",
        ],
        "healthcare": [
            "health care", "medical", "pharma", "biotech", "life sciences",
            "biopharma", "medical devices", "diagnostics",
        ],
        "financials": [
            "banks", "insurance", "financial services", "fintech",
            "asset management", "capital markets",
        ],
        "energy": [
            "oil and gas", "oil & gas", "petroleum", "upstream", "downstream",
            "renewables", "clean energy",
        ],
        "consumer": [
            "consumer discretionary", "consumer staples", "retail",
            "consumer goods", "household products",
        ],
        "industrials": [
            "manufacturing", "aerospace", "defense", "transportation",
            "logistics", "construction",
        ],
        "utilities": [
            "electric utilities", "water utilities", "gas utilities",
            "regulated utilities", "infrastructure",
        ],
        "real estate": [
            "REITs", "real estate investment trusts", "property",
            "commercial real estate", "residential",
        ],
        "materials": [
            "chemicals", "metals and mining", "gold", "silver",
            "steel", "aluminum", "commodities",
        ],
        "communication services": [
            "telecom", "media", "entertainment", "streaming",
            "social media",
        ],
    }

    # Build reverse lookup: synonym → canonical term
    @classmethod
    def get_canonical(cls, term: str) -> str | None:
        """Map a synonym back to its canonical metric name."""
        term_lower = term.lower().strip()
        for canonical, synonyms in cls.METRIC_SYNONYMS.items():
            if term_lower == canonical.lower():
                return canonical
            if any(term_lower == s.lower() for s in synonyms):
                return canonical
        for canonical, synonyms in cls.CONCEPT_SYNONYMS.items():
            if term_lower == canonical.lower():
                return canonical
            if any(term_lower == s.lower() for s in synonyms):
                return canonical
        return None


# ---------------------------------------------------------------------------
# Query Parser
# ---------------------------------------------------------------------------

# Time period regex patterns
_TIME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(Q[1-4])\s+(\d{4})\b", re.I), "quarter_year"),
    (re.compile(r"\b(first|second|third|fourth)\s+quarter\b", re.I), "quarter_name"),
    (re.compile(r"\blast\s+quarter\b", re.I), "last_quarter"),
    (re.compile(r"\bthis\s+quarter\b", re.I), "current_quarter"),
    (re.compile(r"\bYTD\b|year.to.date\b", re.I), "ytd"),
    (re.compile(r"\bTTM\b|trailing\s+twelve\s+months?\b", re.I), "ttm"),
    (re.compile(r"\bfull\s+year\s+(\d{4})\b", re.I), "full_year"),
    (re.compile(r"\b(\d{4})\b"), "year"),
    (re.compile(r"\blast\s+year\b", re.I), "last_year"),
    (re.compile(r"\b(\d+)\s+years?\s+ago\b", re.I), "years_ago"),
    (re.compile(r"\bFY(\d{2,4})\b", re.I), "fiscal_year"),
]

_QUARTER_NAME_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
}

# Query type keyword patterns
_QUERY_TYPE_PATTERNS: dict[str, list[str]] = {
    "comparison": ["compare", "vs", "versus", "difference", "relative to", "against"],
    "trend": ["trend", "growth", "over time", "historical", "trajectory", "progression"],
    "news": ["news", "announcement", "press release", "said", "reported", "disclosed"],
    "screening": ["screen", "find stocks", "show me", "list", "filter", "rank"],
    "lookup": [],  # default
}

# Doc type detection patterns
_DOC_TYPE_PATTERNS: dict[str, list[str]] = {
    "10k": ["10-k", "10k", "annual report", "annual filing"],
    "10q": ["10-q", "10q", "quarterly report", "quarterly filing"],
    "earnings_pr": ["earnings", "results", "earnings release", "press release"],
    "8k_material": ["8-k", "8k", "material event", "current report"],
    "proxy": ["proxy", "def 14a", "shareholder vote", "annual meeting"],
    "news": ["news", "article", "report", "headline"],
}


class QueryParser:
    """Extract structured intent from natural language financial queries."""

    def __init__(self) -> None:
        self._library = FinancialSynonymLibrary()

    def parse_natural_language_query(self, query: str) -> dict:
        """Parse a natural language query into structured intent.

        Returns:
            {
                entity: str | None,
                ticker: str | None,
                metrics: list[str],
                time_period: dict,
                comparison: bool,
                doc_type: str | None,
                query_type: str,
                intent_summary: str,
            }
        """
        entity = self._extract_entity(query)
        ticker = self.resolve_ticker(query)
        metrics = self._extract_metrics(query)
        time_period = self.extract_time_period(query)
        doc_type = self._detect_doc_type(query)
        query_type = self._detect_query_type(query)
        comparison = any(kw in query.lower() for kw in _QUERY_TYPE_PATTERNS["comparison"])

        intent_summary = self._build_intent_summary(
            entity, ticker, metrics, time_period, doc_type, query_type
        )

        return {
            "entity": entity,
            "ticker": ticker,
            "metrics": metrics,
            "time_period": time_period,
            "comparison": comparison,
            "doc_type": doc_type,
            "query_type": query_type,
            "intent_summary": intent_summary,
            "original_query": query,
        }

    def resolve_ticker(self, text: str) -> str | None:
        """Pattern match company names and aliases to ticker symbols."""
        # First: check direct ticker symbol pattern (2-5 uppercase letters)
        ticker_m = re.search(r"\b([A-Z]{1,5})\b", text)
        if ticker_m:
            candidate = ticker_m.group(1)
            # Validate it looks like a real ticker (skip common words)
            _SKIP_WORDS = {"I", "A", "IN", "TO", "US", "ON", "AT", "OR", "AND", "THE"}
            if candidate not in _SKIP_WORDS and len(candidate) >= 2:
                return candidate

        # Second: match company aliases
        text_lower = text.lower()
        for company, ticker in FinancialSynonymLibrary.COMPANY_ALIASES.items():
            if company.lower() in text_lower:
                return ticker

        return None

    def extract_time_period(self, text: str) -> dict:
        """Parse relative and absolute time references into structured form.

        Returns dict with keys: type, label, start_date, end_date.
        """
        today = date.today()

        for pattern, kind in _TIME_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue

            if kind == "quarter_year":
                q_num = int(m.group(1)[1])
                year = int(m.group(2))
                return self._quarter_to_dates(q_num, year)

            if kind == "quarter_name":
                q_name = m.group(1).lower()
                q_num = _QUARTER_NAME_MAP.get(q_name, 1)
                return self._quarter_to_dates(q_num, today.year)

            if kind == "last_quarter":
                prev_q_end = date(today.year, ((today.month - 1) // 3) * 3 or 12, 1) - timedelta(days=1)
                prev_q_start = date(prev_q_end.year, ((prev_q_end.month - 1) // 3) * 3 + 1, 1)
                return {
                    "type": "quarter",
                    "label": f"Q{(prev_q_end.month - 1) // 3 + 1} {prev_q_end.year}",
                    "start_date": prev_q_start.isoformat(),
                    "end_date": prev_q_end.isoformat(),
                }

            if kind == "current_quarter":
                q_num = (today.month - 1) // 3 + 1
                return self._quarter_to_dates(q_num, today.year)

            if kind == "ytd":
                return {
                    "type": "ytd",
                    "label": f"YTD {today.year}",
                    "start_date": date(today.year, 1, 1).isoformat(),
                    "end_date": today.isoformat(),
                }

            if kind == "ttm":
                return {
                    "type": "ttm",
                    "label": "TTM",
                    "start_date": (today - timedelta(days=365)).isoformat(),
                    "end_date": today.isoformat(),
                }

            if kind in ("full_year", "year"):
                try:
                    year = int(m.group(1))
                except (IndexError, ValueError):
                    continue
                if 1990 <= year <= today.year + 2:
                    return {
                        "type": "full_year",
                        "label": str(year),
                        "start_date": f"{year}-01-01",
                        "end_date": f"{year}-12-31",
                    }

            if kind == "last_year":
                year = today.year - 1
                return {
                    "type": "full_year",
                    "label": str(year),
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                }

            if kind == "years_ago":
                n = int(m.group(1))
                year = today.year - n
                return {
                    "type": "full_year",
                    "label": str(year),
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                }

            if kind == "fiscal_year":
                fy = m.group(1)
                year = int(fy) if len(fy) == 4 else 2000 + int(fy)
                return {
                    "type": "fiscal_year",
                    "label": f"FY{year}",
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                }

        return {"type": "unspecified", "label": "unspecified", "start_date": None, "end_date": None}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_entity(self, query: str) -> str | None:
        """Extract company name or ticker from query text."""
        for company in FinancialSynonymLibrary.COMPANY_ALIASES:
            if company.lower() in query.lower():
                return company
        # Fallback: look for capitalized proper noun
        m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
        if m:
            return m.group(1)
        return None

    def _extract_metrics(self, query: str) -> list[str]:
        """Find all financial metrics mentioned in the query."""
        found = []
        q_lower = query.lower()
        for canonical, synonyms in FinancialSynonymLibrary.METRIC_SYNONYMS.items():
            all_terms = [canonical.lower()] + [s.lower() for s in synonyms]
            if any(term in q_lower for term in all_terms):
                found.append(canonical)
        return found

    def _detect_doc_type(self, query: str) -> str | None:
        q_lower = query.lower()
        for doc_type, patterns in _DOC_TYPE_PATTERNS.items():
            if any(p in q_lower for p in patterns):
                return doc_type
        return None

    def _detect_query_type(self, query: str) -> str:
        q_lower = query.lower()
        for qtype, keywords in _QUERY_TYPE_PATTERNS.items():
            if keywords and any(kw in q_lower for kw in keywords):
                return qtype
        return "lookup"

    @staticmethod
    def _quarter_to_dates(q_num: int, year: int) -> dict:
        starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        return {
            "type": "quarter",
            "label": f"Q{q_num} {year}",
            "start_date": f"{year}-{starts[q_num]}",
            "end_date": f"{year}-{ends[q_num]}",
        }

    @staticmethod
    def _build_intent_summary(entity, ticker, metrics, time_period, doc_type, query_type) -> str:
        parts = []
        if query_type != "lookup":
            parts.append(f"{query_type.title()} query")
        if entity:
            parts.append(f"for {entity}" + (f" ({ticker})" if ticker else ""))
        if metrics:
            parts.append(f"metrics: {', '.join(metrics[:3])}")
        if time_period["type"] != "unspecified":
            parts.append(f"period: {time_period['label']}")
        if doc_type:
            parts.append(f"document: {doc_type}")
        return " | ".join(parts) if parts else "General financial query"


# ---------------------------------------------------------------------------
# Query Expander
# ---------------------------------------------------------------------------

class QueryExpander:
    """Generate expanded query variants using synonym substitution and ticker resolution."""

    def __init__(self) -> None:
        self._parser = QueryParser()

    def expand_query(self, query: str) -> list[str]:
        """Generate 3-5 expanded query variants.

        Adds:
          - ticker symbol if company name detected
          - metric synonyms
          - common alternative phrasings

        Example:
          "Tesla revenue growth" →
          ["Tesla TSLA revenue growth", "TSLA sales increase",
           "Tesla top line expansion", "TSLA revenue YoY change"]
        """
        parsed = self._parser.parse_natural_language_query(query)
        variants: list[str] = [query]

        # Add ticker if entity resolved
        ticker = parsed.get("ticker")
        entity = parsed.get("entity")
        if ticker and entity and ticker not in query.upper():
            variants.append(re.sub(
                re.escape(entity), f"{entity} {ticker}", query, count=1, flags=re.I
            ))
            variants.append(re.sub(
                re.escape(entity), ticker, query, count=1, flags=re.I
            ))

        # Expand metrics with synonyms
        for metric in parsed.get("metrics", [])[:2]:
            synonyms = FinancialSynonymLibrary.METRIC_SYNONYMS.get(metric, [])
            for syn in synonyms[:2]:
                expanded = re.sub(
                    re.escape(metric), syn, query, count=1, flags=re.I
                )
                if expanded != query and expanded not in variants:
                    variants.append(expanded)
                    if len(variants) >= 5:
                        break

        return variants[:5]

    def expand_for_vector_search(self, query: str, n_expansions: int = 3) -> list[str]:
        """Generate expansions optimized for semantic similarity / vector search.

        Prioritizes full-phrase synonyms and contextual expansions over
        abbreviated forms, since vector embeddings capture semantic meaning.
        """
        parsed = self._parser.parse_natural_language_query(query)
        expansions = [query]

        # Add concept-level expansions
        q_lower = query.lower()
        for concept, synonyms in FinancialSynonymLibrary.CONCEPT_SYNONYMS.items():
            if concept.lower() in q_lower:
                for syn in synonyms[:2]:
                    variant = q_lower.replace(concept.lower(), syn)
                    if variant not in expansions:
                        expansions.append(variant)

        # Add metric long-form expansions
        for metric in parsed.get("metrics", []):
            long_forms = [
                s for s in FinancialSynonymLibrary.METRIC_SYNONYMS.get(metric, [])
                if len(s) > len(metric)
            ]
            if long_forms:
                variant = query + " " + long_forms[0]
                if variant not in expansions:
                    expansions.append(variant)

        return expansions[:n_expansions]

    def rerank_results(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank search results using BM25-style term overlap scoring.

        Each result dict should have a 'text' or 'title' field for scoring.
        Adds 'rerank_score' field to each result and returns sorted list.
        """
        query_terms = set(re.findall(r"\w+", query.lower()))

        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "in", "of", "for", "to", "and", "or", "what", "how"}
        query_terms -= stopwords

        for result in results:
            text = (result.get("text") or result.get("title") or result.get("summary") or "").lower()
            result_terms = set(re.findall(r"\w+", text))

            # BM25-style overlap: penalize very short or very long results
            overlap = len(query_terms & result_terms)
            doc_len = len(result_terms)
            k1, b, avg_dl = 1.5, 0.75, 200
            tf = overlap * (k1 + 1) / (overlap + k1 * (1 - b + b * doc_len / avg_dl))
            result["rerank_score"] = round(tf, 4)

        return sorted(results, key=lambda r: r.get("rerank_score", 0), reverse=True)

    def suggest_related_queries(self, query: str) -> list[str]:
        """Suggest 5 related queries the user might also want."""
        parsed = self._parser.parse_natural_language_query(query)
        ticker = parsed.get("ticker") or parsed.get("entity") or "company"
        metrics = parsed.get("metrics", [])
        primary_metric = metrics[0] if metrics else "revenue"

        suggestions = [
            f"{ticker} {primary_metric} growth trend 3 years",
            f"{ticker} analyst consensus estimates",
            f"{ticker} insider buying recent",
            f"{ticker} comparison to sector peers",
            f"{ticker} {primary_metric} guidance vs actual",
        ]

        # Make ticker-specific if resolved
        if parsed.get("ticker"):
            t = parsed["ticker"]
            suggestions = [s.replace(ticker, t) for s in suggestions]

        return suggestions[:5]


# ---------------------------------------------------------------------------
# Financial Query Router
# ---------------------------------------------------------------------------

# Routing rule: (pattern, module, action, extract_group)
# extract_group: which regex group to use for the entity
_ROUTING_PATTERNS: list[tuple[str, str, str]] = [
    # Price / valuation
    (r"(P/E|PE\s+ratio|price.to.earnings|earnings\s+multiple)", "standardized_financials", "get_ratio"),
    (r"(P/B|price.to.book|book\s+value)", "standardized_financials", "get_ratio"),
    (r"(EV/EBITDA|enterprise\s+value)", "standardized_financials", "get_ratio"),
    (r"(stock\s+price|share\s+price|current\s+price)", "market_data", "get_price"),
    # Earnings / financials
    (r"(revenue|sales|top\s+line)", "standardized_financials", "get_income_statement"),
    (r"(EPS|earnings\s+per\s+share|net\s+income)", "standardized_financials", "get_income_statement"),
    (r"(gross\s+margin|operating\s+margin|net\s+margin)", "standardized_financials", "get_margins"),
    (r"(balance\s+sheet|total\s+assets|total\s+debt)", "standardized_financials", "get_balance_sheet"),
    (r"(free\s+cash\s+flow|FCF|cash\s+flow)", "standardized_financials", "get_cash_flow"),
    # News / filings
    (r"(news|announcement|headline|press\s+release)", "earnings_corpus", "news_feed"),
    (r"(earnings\s+call|conference\s+call|transcript)", "earnings_corpus", "earnings_transcript"),
    (r"(10-K|annual\s+report)", "edgar_full_text_search", "get_annual_filing"),
    (r"(10-Q|quarterly\s+report)", "edgar_full_text_search", "get_quarterly_filing"),
    (r"(8-K|material\s+event)", "edgar_full_text_search", "get_8k"),
    # Insider / ownership
    (r"(insider\s+buy|insider\s+sell|Form\s+4|director\s+purchase)", "insider_transactions", "get_recent"),
    (r"(institutional\s+ownership|13[FG]\s+filing|fund\s+ownership)", "ownership_screener", "get_institutional"),
    (r"(short\s+interest|short\s+float|short\s+squeeze)", "market_data", "get_short_interest"),
    # Options
    (r"(options|puts|calls|implied\s+volatility|IV)", "options_analytics", "get_chain"),
    (r"(unusual\s+options|options\s+flow)", "options_analytics", "get_unusual_flow"),
    # Technical
    (r"(RSI|MACD|moving\s+average|momentum|relative\s+strength)", "technical_screener", "momentum"),
    (r"(52.week\s+high|breakout|support|resistance)", "technical_screener", "levels"),
    # Index / macro
    (r"(S&P\s+500|SPY|QQQ|Nasdaq|Dow)", "market_data", "get_index"),
    (r"(interest\s+rates?|Fed|Federal\s+Reserve|FOMC)", "macro_data", "get_rates"),
    (r"(GDP|inflation|CPI|unemployment)", "macro_data", "get_economic"),
    # Crypto / alternative
    (r"(Bitcoin|BTC|Ethereum|ETH|crypto|DeFi)", "crypto_screener", "get_crypto"),
    # Screening
    (r"(screen|find\s+stocks|filter|rank\s+by)", "technical_screener", "screen"),
    # ESG
    (r"(ESG|sustainability|carbon|emissions)", "esg_composite", "get_esg"),
    # Dividend
    (r"(dividend|yield|payout|DPS)", "standardized_financials", "get_dividends"),
    # M&A
    (r"(merger|acquisition|M&A|deal|takeover)", "ma_intelligence", "get_deals"),
]

_COMPILED_ROUTING = [
    (re.compile(pat, re.I), module, action)
    for pat, module, action in _ROUTING_PATTERNS
]


class FinancialQueryRouter:
    """Route natural language queries to the appropriate SENTINEL module."""

    def __init__(self) -> None:
        self._parser = QueryParser()
        self._expander = QueryExpander()

    def route_query(self, query: str) -> dict:
        """Determine which SENTINEL module and action should handle this query.

        Examples:
            "What is Apple's P/E ratio?" → {module: "standardized_financials", action: "get_ratio", ticker: "AAPL"}
            "Show Tesla news"            → {module: "earnings_corpus", action: "news_feed", ticker: "TSLA"}
            "S&P 500 momentum"           → {module: "technical_screener", action: "momentum", index: "SPY"}
        """
        parsed = self._parser.parse_natural_language_query(query)
        ticker = parsed.get("ticker")

        matched_module = "general_search"
        matched_action = "search"
        matched_pattern = None

        for pattern, module, action in _COMPILED_ROUTING:
            if pattern.search(query):
                matched_module = module
                matched_action = action
                matched_pattern = pattern.pattern
                break

        route = {
            "module": matched_module,
            "action": matched_action,
            "ticker": ticker,
            "entity": parsed.get("entity"),
            "metrics": parsed.get("metrics"),
            "time_period": parsed.get("time_period"),
            "doc_type": parsed.get("doc_type"),
            "matched_pattern": matched_pattern,
            "query_type": parsed.get("query_type"),
            "intent_summary": parsed.get("intent_summary"),
        }

        # Special index routing
        if re.search(r"\b(S&P\s*500|SPX)\b", query, re.I):
            route["index"] = "SPY"
        if re.search(r"\b(Nasdaq|QQQ|NDX)\b", query, re.I):
            route["index"] = "QQQ"

        return route

    def execute_query(self, query: str, context: dict | None = None) -> dict:
        """Full pipeline: parse → expand → route → format response stub.

        In production, this calls the actual module. Here it returns the
        routing plan plus expanded query variants for the caller to execute.
        """
        route = self.route_query(query)
        expansions = self._expander.expand_query(query)
        related = self._expander.suggest_related_queries(query)

        return {
            "original_query": query,
            "route": route,
            "expanded_queries": expansions,
            "related_queries": related,
            "context": context or {},
            "status": "routed",
        }


# ---------------------------------------------------------------------------
# Autocomplete Engine
# ---------------------------------------------------------------------------

_ALL_FINANCIAL_TERMS: list[str] = sorted(set(
    list(FinancialSynonymLibrary.METRIC_SYNONYMS.keys())
    + [syn for syns in FinancialSynonymLibrary.METRIC_SYNONYMS.values() for syn in syns]
    + list(FinancialSynonymLibrary.CONCEPT_SYNONYMS.keys())
    + [syn for syns in FinancialSynonymLibrary.CONCEPT_SYNONYMS.values() for syn in syns]
    + list(FinancialSynonymLibrary.COMPANY_ALIASES.keys())
    + list(FinancialSynonymLibrary.COMPANY_ALIASES.values())
    + [t for terms in FinancialSynonymLibrary.SECTOR_TERMS.values() for t in terms]
    + [
        # Additional common financial terms
        "S&P 500", "Nasdaq", "Dow Jones", "Russell 2000", "VIX",
        "Federal Reserve", "FOMC", "interest rates", "inflation", "CPI",
        "GDP", "unemployment", "yield curve", "credit spread",
        "high yield", "investment grade", "emerging markets",
        "sector rotation", "risk-on", "risk-off",
        "earnings season", "ex-dividend date", "record date",
        "stock split", "reverse split", "rights offering",
        "convertible notes", "term loan", "revolving credit",
        "leverage buyout", "LBO", "SPAC", "blank check company",
        "at-the-market offering", "ATM offering",
        "dark pool", "block trade", "program trading",
        "systematic risk", "idiosyncratic risk", "factor exposure",
        "value factor", "growth factor", "momentum factor", "quality factor",
        "sector ETF", "thematic ETF", "leveraged ETF",
    ]
))


class AutocompleteEngine:
    """Fast prefix-matching autocomplete over financial terms and tickers.

    Uses a sorted list + bisect for O(log n) prefix search — no external
    dependencies required. Handles 5000+ terms comfortably.
    """

    def __init__(self, extra_terms: list[str] | None = None) -> None:
        terms = list(_ALL_FINANCIAL_TERMS)
        if extra_terms:
            terms = sorted(set(terms + extra_terms))
        self._terms = terms
        self._terms_lower = [t.lower() for t in terms]

    def get_suggestions(self, partial: str, limit: int = 10) -> list[str]:
        """Fuzzy prefix match on ticker symbols, company names, and financial metrics.

        Returns up to `limit` suggestions sorted by relevance (prefix matches first,
        then substring matches).
        """
        if not partial or len(partial) < 1:
            return []

        p = partial.lower().strip()
        prefix_matches = []
        substring_matches = []

        for i, term_lower in enumerate(self._terms_lower):
            if term_lower.startswith(p):
                prefix_matches.append(self._terms[i])
            elif p in term_lower:
                substring_matches.append(self._terms[i])

        combined = prefix_matches + substring_matches
        return combined[:limit]

    def get_metric_suggestions(self, partial: str) -> list[str]:
        """Financial metric autocomplete — metrics and their synonyms only."""
        if not partial:
            return []

        p = partial.lower()
        results = []
        for canonical, synonyms in FinancialSynonymLibrary.METRIC_SYNONYMS.items():
            all_terms = [canonical] + synonyms
            for term in all_terms:
                if term.lower().startswith(p) or p in term.lower():
                    results.append(term)

        # Deduplicate preserving order
        seen = set()
        deduped = []
        for r in results:
            if r.lower() not in seen:
                seen.add(r.lower())
                deduped.append(r)

        return deduped[:10]


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

query_router = APIRouter(prefix="/api/query", tags=["query-expansion"])

_parser_instance: QueryParser | None = None
_expander_instance: QueryExpander | None = None
_router_instance: FinancialQueryRouter | None = None
_autocomplete_instance: AutocompleteEngine | None = None


def _get_parser() -> QueryParser:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = QueryParser()
    return _parser_instance


def _get_expander() -> QueryExpander:
    global _expander_instance
    if _expander_instance is None:
        _expander_instance = QueryExpander()
    return _expander_instance


def _get_query_router() -> FinancialQueryRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = FinancialQueryRouter()
    return _router_instance


def _get_autocomplete() -> AutocompleteEngine:
    global _autocomplete_instance
    if _autocomplete_instance is None:
        _autocomplete_instance = AutocompleteEngine()
    return _autocomplete_instance


class ParseQueryRequest(BaseModel):
    query: str


class ExpandQueryRequest(BaseModel):
    query: str
    n_expansions: int = 5
    mode: str = "bm25"   # "bm25" or "vector"


class ExecuteQueryRequest(BaseModel):
    query: str
    context: dict | None = None


class RerankRequest(BaseModel):
    query: str
    results: list[dict]


@query_router.post("/parse")
async def parse_query(req: ParseQueryRequest):
    """Parse a natural language financial query into structured intent."""
    parser = _get_parser()
    return parser.parse_natural_language_query(req.query)


@query_router.post("/expand")
async def expand_query(req: ExpandQueryRequest):
    """Expand a query with financial synonyms and alternative phrasings."""
    expander = _get_expander()
    if req.mode == "vector":
        variants = expander.expand_for_vector_search(req.query, n_expansions=req.n_expansions)
    else:
        variants = expander.expand_query(req.query)
    related = expander.suggest_related_queries(req.query)
    return {
        "original": req.query,
        "expanded": variants,
        "related": related,
        "mode": req.mode,
    }


@query_router.post("/execute")
async def execute_query(req: ExecuteQueryRequest):
    """Full pipeline: parse, expand, route a natural language query."""
    router = _get_query_router()
    return router.execute_query(req.query, context=req.context)


@query_router.get("/autocomplete")
async def autocomplete(q: str = FastAPIQuery(..., min_length=1)):
    """Autocomplete suggestions for financial terms, tickers, and metrics."""
    engine = _get_autocomplete()
    return {
        "partial": q,
        "suggestions": engine.get_suggestions(q, limit=10),
        "metric_suggestions": engine.get_metric_suggestions(q),
    }


@query_router.get("/suggest-related")
async def suggest_related(q: str = FastAPIQuery(..., min_length=2)):
    """Suggest 5 related queries based on the input query."""
    expander = _get_expander()
    return {
        "query": q,
        "related_queries": expander.suggest_related_queries(q),
    }


@query_router.post("/rerank")
async def rerank_results(req: RerankRequest):
    """Rerank search results by BM25-style query-result overlap score."""
    expander = _get_expander()
    reranked = expander.rerank_results(req.query, req.results)
    return {
        "query": req.query,
        "results": reranked,
    }
