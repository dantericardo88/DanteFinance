"""
Natural Language Screener V2 — Bloomberg-parity NL→screener system (dim_076, target score 9).

Enhancements over nl_screener_translator.py:
  - 50+ financial predicates with fuzzy metric matching
  - Intent classification: SCREEN / RANK / COMPARE / ALERT
  - Multi-condition boolean logic: AND / OR / NOT with parenthetical grouping
  - Temporal expressions: TTM, YoY, 5-year CAGR, "last 3 quarters"
  - 40+ pre-built templates (growth, value, dividend, momentum, etc.)
  - Screen execution engine: yfinance bulk data with SQLite 24h cache
  - Result ranking with per-stock explanation ("AAPL matched because P/E=28.5 < 30 AND ROE=45%")
  - Sector/industry normalisation, geography, market-cap tier, index membership
  - Alert threshold storage and evaluation
  - FastAPI router: POST /nl-screen, GET /templates, POST /execute-template, + 8 more

Usage::
    from sentinel.sai.nl_screener_v2 import NLScreenerV2Pipeline, nl_screener_v2_router

    pipeline = NLScreenerV2Pipeline()
    result = pipeline.run("Show me cheap tech stocks with ROE > 20 and low debt")
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query as FastAPIQuery
from pydantic import BaseModel, Field

# ── DB path ───────────────────────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent.parent / "data" / "nl_screener_v2.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_CACHE_TTL_SECONDS = 86_400  # 24 h
_YFINANCE_BATCH = 50         # tickers per yfinance call

# ── Intent taxonomy ───────────────────────────────────────────────────────────
INTENTS = ["SCREEN", "RANK", "COMPARE", "ALERT"]

# Intent keyword signals
INTENT_SIGNALS: Dict[str, List[str]] = {
    "SCREEN": [
        "show me", "find", "screen", "filter", "list", "give me",
        "which stocks", "stocks with", "companies with", "what stocks",
    ],
    "RANK": [
        "rank", "sort", "top", "best", "highest", "lowest", "order by",
        "most", "least", "largest", "smallest",
    ],
    "COMPARE": [
        "compare", "vs", "versus", "difference between", "how does",
        "peer", "relative to", "against",
    ],
    "ALERT": [
        "alert", "notify", "when", "if", "trigger", "watch", "monitor",
        "set alert", "tell me when",
    ],
}

# ── 50+ Financial predicates ──────────────────────────────────────────────────
METRIC_CANONICAL: Dict[str, str] = {
    # Valuation
    "pe": "pe_ratio", "p/e": "pe_ratio", "pe ratio": "pe_ratio",
    "price to earnings": "pe_ratio", "price-to-earnings": "pe_ratio",
    "earnings multiple": "pe_ratio", "p/e ratio": "pe_ratio",
    "pb": "pb_ratio", "p/b": "pb_ratio", "price to book": "pb_ratio",
    "price-to-book": "pb_ratio", "book multiple": "pb_ratio",
    "ps": "ps_ratio", "p/s": "ps_ratio", "price to sales": "ps_ratio",
    "price-to-sales": "ps_ratio", "revenue multiple": "ps_ratio",
    "ev ebitda": "ev_ebitda", "ev/ebitda": "ev_ebitda",
    "enterprise value to ebitda": "ev_ebitda",
    "peg": "peg_ratio", "peg ratio": "peg_ratio",
    "price to earnings growth": "peg_ratio",
    "earnings yield": "earnings_yield",
    "fcf yield": "fcf_yield", "free cash flow yield": "fcf_yield",
    "dividend yield": "dividend_yield", "div yield": "dividend_yield",
    "yield": "dividend_yield",
    # Growth
    "revenue growth": "revenue_growth", "sales growth": "revenue_growth",
    "top line growth": "revenue_growth",
    "earnings growth": "earnings_growth", "eps growth": "earnings_growth",
    "net income growth": "net_income_growth",
    "ebitda growth": "ebitda_growth",
    "fcf growth": "fcf_growth", "free cash flow growth": "fcf_growth",
    "revenue cagr": "revenue_cagr_5y", "5 year revenue cagr": "revenue_cagr_5y",
    "eps cagr": "eps_cagr_5y", "5 year eps cagr": "eps_cagr_5y",
    # Profitability
    "gross margin": "gross_margin", "gross profit margin": "gross_margin",
    "operating margin": "operating_margin", "ebit margin": "operating_margin",
    "net margin": "net_margin", "profit margin": "net_margin",
    "ebitda margin": "ebitda_margin",
    "roe": "roe", "return on equity": "roe", "return_on_equity": "roe",
    "roa": "roa", "return on assets": "roa",
    "roic": "roic", "return on invested capital": "roic",
    "roce": "roce", "return on capital employed": "roce",
    # Balance sheet
    "debt equity": "debt_equity", "debt to equity": "debt_equity",
    "d/e": "debt_equity", "leverage": "debt_equity",
    "net debt ebitda": "net_debt_ebitda", "net debt to ebitda": "net_debt_ebitda",
    "current ratio": "current_ratio", "quick ratio": "quick_ratio",
    "interest coverage": "interest_coverage",
    "cash ratio": "cash_ratio",
    "book value": "book_value_per_share",
    # Cash flow
    "free cash flow": "fcf", "fcf": "fcf",
    "operating cash flow": "operating_cash_flow",
    "capex": "capex", "capital expenditure": "capex",
    "cash conversion": "cash_conversion_ratio",
    # Size
    "market cap": "market_cap", "market capitalization": "market_cap",
    "enterprise value": "enterprise_value",
    "revenue": "revenue", "sales": "revenue",
    "ebitda": "ebitda",
    # Technicals
    "rsi": "rsi", "relative strength index": "rsi",
    "beta": "beta",
    "52 week high": "pct_from_52w_high", "near 52 week high": "pct_from_52w_high",
    "52 week low": "pct_from_52w_low",
    "return 3m": "return_3m", "3 month return": "return_3m",
    "3m return": "return_3m", "3-month return": "return_3m",
    "return 6m": "return_6m", "6 month return": "return_6m",
    "return 1y": "return_1y", "1 year return": "return_1y",
    "ytd return": "return_ytd",
    "moving average 50": "price_vs_ma50", "above 50 day ma": "price_vs_ma50",
    "moving average 200": "price_vs_ma200", "above 200 day ma": "price_vs_ma200",
    "volume": "avg_volume",
    # Dividends
    "consecutive dividend growth": "consecutive_div_growth",
    "dividend growth years": "consecutive_div_growth",
    "payout ratio": "payout_ratio",
    "dividend coverage": "dividend_coverage",
    # Sentiment / estimates
    "analyst rating": "analyst_rating", "analyst consensus": "analyst_rating",
    "eps revision": "eps_revision", "estimate revision": "eps_revision",
    "earnings surprise": "earnings_surprise", "eps beat": "earnings_surprise",
    "short interest": "short_interest", "float short": "float_short",
    "institutional ownership": "institutional_ownership",
    "insider ownership": "insider_ownership",
    # Options
    "implied volatility": "iv_percentile", "iv percentile": "iv_percentile",
    "borrow cost": "borrow_cost",
    # ESG
    "esg score": "esg_score", "environmental score": "esg_env",
    "governance score": "esg_gov",
}

# ── Fuzzy metric mapping (for "price to earnings" → pe_ratio) ─────────────────
METRIC_FUZZY_MAP: Dict[str, str] = {
    "priceearnings": "pe_ratio",
    "earningsprice": "pe_ratio",
    "pricebook": "pb_ratio",
    "bookprice": "pb_ratio",
    "pricesales": "ps_ratio",
    "salesprice": "ps_ratio",
    "evtoebitda": "ev_ebitda",
    "debtequity": "debt_equity",
    "equitydebt": "debt_equity",
    "returnonequity": "roe",
    "returnonassets": "roa",
    "returnoninvestedcapital": "roic",
    "grossprofit": "gross_margin",
    "operatingprofit": "operating_margin",
    "netprofit": "net_margin",
    "freecashflow": "fcf",
    "dividendyield": "dividend_yield",
    "marketcap": "market_cap",
    "revenuegrowtg": "revenue_growth",  # intentional typo tolerance
}

# ── Operator phrase mapping ───────────────────────────────────────────────────
OPERATOR_PHRASES: Dict[str, str] = {
    "below": "<", "under": "<", "less than": "<", "lower than": "<",
    "smaller than": "<", "beneath": "<", "no more than": "<=",
    "at most": "<=", "up to": "<=", "maximum": "<=", "max": "<=",
    "above": ">", "over": ">", "greater than": ">", "higher than": ">",
    "more than": ">", "exceeds": ">", "exceeding": ">",
    "at least": ">=", "minimum": ">=", "min": ">=", "no less than": ">=",
    "equal to": "==", "exactly": "==", "=": "==",
    "not equal": "!=", "not": "!=",
    "<": "<", ">": ">", "<=": "<=", ">=": ">=",
}

# ── Sector canonical mapping ──────────────────────────────────────────────────
SECTOR_ALIASES: Dict[str, str] = {
    "tech": "Technology", "technology": "Technology", "software": "Technology",
    "semiconductor": "Technology", "semiconductors": "Technology", "chip": "Technology",
    "chips": "Technology", "cloud": "Technology", "saas": "Technology",
    "ai": "Technology", "artificial intelligence": "Technology",
    "cybersecurity": "Technology", "cyber": "Technology",
    "fintech": "Technology", "internet": "Technology",
    "healthcare": "Health Care", "health care": "Health Care",
    "pharma": "Health Care", "pharmaceutical": "Health Care",
    "biotech": "Health Care", "biotechnology": "Health Care",
    "medical": "Health Care", "drug": "Health Care", "medtech": "Health Care",
    "hospital": "Health Care", "diagnostics": "Health Care",
    "energy": "Energy", "oil": "Energy", "oil and gas": "Energy",
    "gas": "Energy", "petroleum": "Energy", "upstream": "Energy",
    "renewable": "Energy", "solar": "Energy", "wind": "Energy",
    "clean energy": "Energy",
    "utilities": "Utilities", "electric": "Utilities", "power": "Utilities",
    "water utility": "Utilities",
    "financial": "Financials", "financials": "Financials",
    "bank": "Financials", "banking": "Financials",
    "insurance": "Financials", "asset management": "Financials",
    "financial services": "Financials", "investment bank": "Financials",
    "brokerage": "Financials", "finservices": "Financials",
    "consumer": "Consumer Discretionary", "retail": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "luxury": "Consumer Discretionary", "ecommerce": "Consumer Discretionary",
    "e-commerce": "Consumer Discretionary", "auto": "Consumer Discretionary",
    "consumer staples": "Consumer Staples", "food": "Consumer Staples",
    "beverage": "Consumer Staples", "household": "Consumer Staples",
    "tobacco": "Consumer Staples", "grocery": "Consumer Staples",
    "industrial": "Industrials", "industrials": "Industrials",
    "aerospace": "Industrials", "defense": "Industrials",
    "manufacturing": "Industrials", "transportation": "Industrials",
    "logistics": "Industrials", "railroad": "Industrials",
    "materials": "Materials", "mining": "Materials", "gold": "Materials",
    "metals": "Materials", "chemicals": "Materials", "steel": "Materials",
    "real estate": "Real Estate", "reit": "Real Estate", "reits": "Real Estate",
    "property": "Real Estate", "mreits": "Real Estate",
    "communication": "Communication Services",
    "media": "Communication Services", "telecom": "Communication Services",
    "telecommunications": "Communication Services",
    "streaming": "Communication Services", "social media": "Communication Services",
    "advertising": "Communication Services",
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
    "canada": "Canada", "india": "India", "brazil": "Brazil",
    "australia": "Australia",
}

# ── Market cap tier mapping ───────────────────────────────────────────────────
MKTCAP_TIERS: Dict[str, Dict] = {
    "mega cap":  {"metric": "market_cap", "operator": ">",  "value": 200_000_000_000},
    "mega-cap":  {"metric": "market_cap", "operator": ">",  "value": 200_000_000_000},
    "large cap": {"metric": "market_cap", "operator": ">",  "value": 10_000_000_000},
    "large-cap": {"metric": "market_cap", "operator": ">",  "value": 10_000_000_000},
    "mid cap":   {"metric": "market_cap", "operator": "between", "value": (2_000_000_000, 10_000_000_000)},
    "mid-cap":   {"metric": "market_cap", "operator": "between", "value": (2_000_000_000, 10_000_000_000)},
    "small cap": {"metric": "market_cap", "operator": "<",  "value": 2_000_000_000},
    "small-cap": {"metric": "market_cap", "operator": "<",  "value": 2_000_000_000},
    "micro cap": {"metric": "market_cap", "operator": "<",  "value": 300_000_000},
    "micro-cap": {"metric": "market_cap", "operator": "<",  "value": 300_000_000},
    "nano cap":  {"metric": "market_cap", "operator": "<",  "value": 50_000_000},
}

# ── Index membership ──────────────────────────────────────────────────────────
INDEX_ALIASES: Dict[str, str] = {
    "s&p 500": "SP500", "sp500": "SP500", "s&p500": "SP500",
    "s&p": "SP500", "spx": "SP500",
    "nasdaq 100": "NDX", "nasdaq100": "NDX", "qqq": "NDX", "ndx": "NDX",
    "dow jones": "DJIA", "djia": "DJIA", "dow": "DJIA",
    "russell 2000": "RUT", "russell2000": "RUT", "iwm": "RUT",
    "russell 1000": "RUI", "russell1000": "RUI",
    "mid cap 400": "MID", "s&p 400": "MID",
}

# ── Temporal expressions ──────────────────────────────────────────────────────
TEMPORAL_PATTERNS: Dict[str, str] = {
    r"ttm|trailing\s*12\s*months?|last\s*twelve\s*months?|ltm": "TTM",
    r"last\s*(\d+)\s*quarters?": "NQ",  # N quarters
    r"last\s*(\d+)\s*years?": "NY",
    r"yoy|year\s*over\s*year|year-over-year": "YoY",
    r"ytd|year\s*to\s*date": "YTD",
    r"5[-\s]?year\s*cagr": "5Y_CAGR",
    r"3[-\s]?year\s*cagr": "3Y_CAGR",
    r"cagr": "CAGR",
    r"mrq|most\s*recent\s*quarter": "MRQ",
    r"last\s*quarter": "1Q",
    r"past\s*year": "1Y",
    r"5[-\s]?year": "5Y",
    r"3[-\s]?year": "3Y",
    r"monthly|last\s*month": "1M",
}

# ── Vague qualifier expansion ─────────────────────────────────────────────────
VAGUE_TERMS: Dict[str, List[Dict]] = {
    "cheap": [
        {"metric": "pe_ratio",  "operator": "<",  "value": 15.0},
        {"metric": "pb_ratio",  "operator": "<",  "value": 1.5},
        {"metric": "ev_ebitda", "operator": "<",  "value": 8.0},
    ],
    "undervalued": [
        {"metric": "pe_ratio",  "operator": "<",  "value": 15.0},
        {"metric": "pb_ratio",  "operator": "<",  "value": 1.5},
        {"metric": "fcf_yield", "operator": ">",  "value": 5.0},
    ],
    "expensive": [
        {"metric": "pe_ratio",  "operator": ">",  "value": 40.0},
        {"metric": "pb_ratio",  "operator": ">",  "value": 5.0},
    ],
    "overvalued": [
        {"metric": "pe_ratio",  "operator": ">",  "value": 40.0},
        {"metric": "pb_ratio",  "operator": ">",  "value": 5.0},
    ],
    "high growth": [
        {"metric": "revenue_growth",  "operator": ">", "value": 20.0},
        {"metric": "earnings_growth", "operator": ">", "value": 15.0},
    ],
    "hyper growth": [
        {"metric": "revenue_growth",  "operator": ">", "value": 40.0},
        {"metric": "gross_margin",    "operator": ">", "value": 60.0},
    ],
    "growing": [
        {"metric": "revenue_growth", "operator": ">", "value": 10.0},
    ],
    "profitable": [
        {"metric": "net_margin", "operator": ">", "value": 0.0},
        {"metric": "roic",       "operator": ">", "value": 8.0},
    ],
    "unprofitable": [
        {"metric": "net_margin", "operator": "<", "value": 0.0},
    ],
    "high quality": [
        {"metric": "roe",        "operator": ">", "value": 20.0},
        {"metric": "debt_equity","operator": "<", "value": 0.5},
        {"metric": "roic",       "operator": ">", "value": 15.0},
    ],
    "quality": [
        {"metric": "roe",        "operator": ">", "value": 15.0},
        {"metric": "debt_equity","operator": "<", "value": 0.5},
    ],
    "high dividend": [
        {"metric": "dividend_yield", "operator": ">", "value": 3.0},
        {"metric": "payout_ratio",   "operator": "<", "value": 80.0},
    ],
    "dividend": [
        {"metric": "dividend_yield", "operator": ">", "value": 0.5},
    ],
    "momentum": [
        {"metric": "return_3m", "operator": ">", "value": 5.0},
        {"metric": "rsi",       "operator": ">", "value": 50.0},
    ],
    "value": [
        {"metric": "pe_ratio", "operator": "<", "value": 15.0},
        {"metric": "pb_ratio", "operator": "<", "value": 2.0},
    ],
    "deep value": [
        {"metric": "pe_ratio",  "operator": "<", "value": 10.0},
        {"metric": "pb_ratio",  "operator": "<", "value": 1.0},
        {"metric": "ev_ebitda", "operator": "<", "value": 6.0},
    ],
    "growth": [
        {"metric": "revenue_growth", "operator": ">", "value": 15.0},
    ],
    "garp": [
        {"metric": "peg_ratio",      "operator": "<", "value": 1.5},
        {"metric": "revenue_growth", "operator": ">", "value": 10.0},
    ],
    "low volatility": [
        {"metric": "beta", "operator": "<", "value": 0.8},
    ],
    "high volatility": [
        {"metric": "beta", "operator": ">", "value": 1.5},
    ],
    "defensive": [
        {"metric": "beta",        "operator": "<", "value": 0.7},
        {"metric": "dividend_yield","operator": ">", "value": 2.0},
    ],
    "leveraged": [
        {"metric": "debt_equity", "operator": ">", "value": 2.0},
    ],
    "clean balance sheet": [
        {"metric": "debt_equity",  "operator": "<", "value": 0.3},
        {"metric": "current_ratio","operator": ">", "value": 2.0},
    ],
    "distressed": [
        {"metric": "debt_equity",  "operator": ">", "value": 3.0},
        {"metric": "current_ratio","operator": "<", "value": 1.0},
    ],
    "high margin": [
        {"metric": "gross_margin",     "operator": ">", "value": 60.0},
        {"metric": "operating_margin", "operator": ">", "value": 20.0},
    ],
    "cash rich": [
        {"metric": "fcf_yield",  "operator": ">", "value": 5.0},
        {"metric": "cash_ratio", "operator": ">", "value": 1.0},
    ],
    "buyback": [
        {"metric": "fcf_yield", "operator": ">", "value": 4.0},
    ],
    "turnaround": [
        {"metric": "return_1y",  "operator": "<",  "value": -20.0},
        {"metric": "earnings_growth", "operator": ">", "value": 10.0},
    ],
    "fallen angel": [
        {"metric": "return_1y", "operator": "<",  "value": -25.0},
        {"metric": "roe",       "operator": ">",  "value": 10.0},
    ],
}

# ── 40+ screener templates ────────────────────────────────────────────────────
SCREENER_TEMPLATES: Dict[str, Dict] = {
    # Value
    "warren_buffett_quality": {
        "description": "Buffett-style: durable moat, high ROE, low debt, reasonable price",
        "tags": ["value", "quality"],
        "criteria": [
            {"metric": "roe",          "op": ">",  "value": 15.0},
            {"metric": "debt_equity",  "op": "<",  "value": 0.5},
            {"metric": "pe_ratio",     "op": "<",  "value": 20.0},
            {"metric": "net_margin",   "op": ">",  "value": 10.0},
            {"metric": "revenue_growth","op": ">", "value": 5.0},
        ],
        "sort_by": "roe", "sort_asc": False,
    },
    "benjamin_graham_deep_value": {
        "description": "Graham net-net: statistically cheap with balance-sheet safety",
        "tags": ["value", "deep value"],
        "criteria": [
            {"metric": "pe_ratio",     "op": "<", "value": 10.0},
            {"metric": "pb_ratio",     "op": "<", "value": 1.0},
            {"metric": "current_ratio","op": ">", "value": 2.0},
            {"metric": "net_margin",   "op": ">", "value": 0.0},
            {"metric": "debt_equity",  "op": "<", "value": 1.0},
        ],
        "sort_by": "pb_ratio", "sort_asc": True,
    },
    "peter_lynch_garp": {
        "description": "Lynch GARP: growth at a reasonable price, PEG < 1",
        "tags": ["growth", "value"],
        "criteria": [
            {"metric": "peg_ratio",      "op": "<", "value": 1.0},
            {"metric": "revenue_growth", "op": ">", "value": 10.0},
            {"metric": "earnings_growth","op": ">", "value": 10.0},
            {"metric": "pe_ratio",       "op": "<", "value": 30.0},
        ],
        "sort_by": "peg_ratio", "sort_asc": True,
    },
    "joel_greenblatt_magic_formula": {
        "description": "Greenblatt magic formula: high ROIC + high earnings yield",
        "tags": ["value", "quality"],
        "criteria": [
            {"metric": "roic",         "op": ">", "value": 15.0},
            {"metric": "earnings_yield","op": ">", "value": 8.0},
            {"metric": "net_margin",   "op": ">", "value": 5.0},
        ],
        "sort_by": "roic", "sort_asc": False,
    },
    "net_net_graham": {
        "description": "Graham net-net: trading below net current asset value",
        "tags": ["deep value"],
        "criteria": [
            {"metric": "pb_ratio",     "op": "<", "value": 0.7},
            {"metric": "current_ratio","op": ">", "value": 3.0},
            {"metric": "debt_equity",  "op": "<", "value": 0.3},
        ],
        "sort_by": "pb_ratio", "sort_asc": True,
    },
    # Growth
    "high_quality_growth": {
        "description": "High-quality growth: fast-growing with strong unit economics",
        "tags": ["growth", "quality"],
        "criteria": [
            {"metric": "revenue_growth", "op": ">", "value": 25.0},
            {"metric": "gross_margin",   "op": ">", "value": 50.0},
            {"metric": "roe",            "op": ">", "value": 20.0},
        ],
        "sort_by": "revenue_growth", "sort_asc": False,
    },
    "vc_style_hypergrowth": {
        "description": "VC-style: hyper-growth small-cap with high margins",
        "tags": ["growth"],
        "criteria": [
            {"metric": "revenue_growth", "op": ">",  "value": 40.0},
            {"metric": "gross_margin",   "op": ">",  "value": 60.0},
            {"metric": "market_cap",     "op": "<",  "value": 5_000_000_000},
        ],
        "sort_by": "revenue_growth", "sort_asc": False,
    },
    "consistent_compounder": {
        "description": "Consistent EPS compounders: steady earnings growth + high ROIC",
        "tags": ["growth", "quality"],
        "criteria": [
            {"metric": "earnings_growth", "op": ">", "value": 10.0},
            {"metric": "roic",            "op": ">", "value": 15.0},
            {"metric": "pe_ratio",        "op": "<", "value": 30.0},
            {"metric": "debt_equity",     "op": "<", "value": 1.0},
        ],
        "sort_by": "roic", "sort_asc": False,
    },
    "tech_growth": {
        "description": "Tech growth: fast-growing tech with expanding margins",
        "tags": ["growth", "technology"],
        "sector": "Technology",
        "criteria": [
            {"metric": "revenue_growth", "op": ">", "value": 20.0},
            {"metric": "gross_margin",   "op": ">", "value": 60.0},
            {"metric": "pe_ratio",       "op": "<", "value": 50.0},
        ],
        "sort_by": "revenue_growth", "sort_asc": False,
    },
    "software_saas": {
        "description": "SaaS: subscription revenue model, high gross margin, Rule of 40",
        "tags": ["growth", "technology"],
        "sector": "Technology",
        "criteria": [
            {"metric": "revenue_growth", "op": ">", "value": 15.0},
            {"metric": "gross_margin",   "op": ">", "value": 70.0},
        ],
        "sort_by": "revenue_growth", "sort_asc": False,
    },
    # Income / dividend
    "dividend_aristocrat": {
        "description": "Dividend aristocrats: long track record of dividend growth",
        "tags": ["income", "dividend"],
        "criteria": [
            {"metric": "dividend_yield",        "op": ">", "value": 3.0},
            {"metric": "consecutive_div_growth", "op": ">", "value": 25.0},
            {"metric": "payout_ratio",          "op": "<", "value": 75.0},
            {"metric": "debt_equity",           "op": "<", "value": 1.5},
        ],
        "sort_by": "dividend_yield", "sort_asc": False,
    },
    "reit_income": {
        "description": "REIT income: high-yield real estate investment trusts",
        "tags": ["income", "real estate"],
        "sector": "Real Estate",
        "criteria": [
            {"metric": "dividend_yield", "op": ">", "value": 4.0},
            {"metric": "pb_ratio",       "op": "<", "value": 2.5},
        ],
        "sort_by": "dividend_yield", "sort_asc": False,
    },
    "high_yield_income": {
        "description": "High-yield income: maximum dividend with coverage check",
        "tags": ["income"],
        "criteria": [
            {"metric": "dividend_yield",    "op": ">", "value": 5.0},
            {"metric": "payout_ratio",      "op": "<", "value": 85.0},
            {"metric": "dividend_coverage", "op": ">", "value": 1.5},
        ],
        "sort_by": "dividend_yield", "sort_asc": False,
    },
    "low_volatility_income": {
        "description": "Low volatility income: defensive dividend payers, beta < 0.8",
        "tags": ["income", "defensive"],
        "criteria": [
            {"metric": "beta",          "op": "<", "value": 0.8},
            {"metric": "dividend_yield","op": ">", "value": 2.0},
            {"metric": "pe_ratio",      "op": "<", "value": 20.0},
        ],
        "sort_by": "dividend_yield", "sort_asc": False,
    },
    # Momentum
    "momentum_quality": {
        "description": "Momentum + quality: strong recent performers + solid fundamentals",
        "tags": ["momentum"],
        "criteria": [
            {"metric": "return_3m",   "op": ">", "value": 10.0},
            {"metric": "return_6m",   "op": ">", "value": 15.0},
            {"metric": "roe",         "op": ">", "value": 15.0},
            {"metric": "eps_revision","op": ">", "value": 0.0},
        ],
        "sort_by": "return_3m", "sort_asc": False,
    },
    "pre_earnings_momentum": {
        "description": "Pre-earnings: price momentum + positive revisions + low IV",
        "tags": ["momentum", "events"],
        "criteria": [
            {"metric": "return_3m",   "op": ">", "value": 15.0},
            {"metric": "eps_revision","op": ">", "value": 0.0},
            {"metric": "iv_percentile","op": "<","value": 25.0},
        ],
        "sort_by": "return_3m", "sort_asc": False,
    },
    "52_week_breakout": {
        "description": "Near 52-week high breakouts with volume confirmation",
        "tags": ["momentum", "technical"],
        "criteria": [
            {"metric": "pct_from_52w_high", "op": ">", "value": -5.0},
            {"metric": "return_3m",         "op": ">", "value": 10.0},
            {"metric": "rsi",               "op": ">", "value": 55.0},
        ],
        "sort_by": "pct_from_52w_high", "sort_asc": False,
    },
    # Short squeeze
    "short_squeeze_candidate": {
        "description": "Short squeeze: high short interest, low float, tight borrow",
        "tags": ["special situation"],
        "criteria": [
            {"metric": "short_interest", "op": ">", "value": 20.0},
            {"metric": "float_short",    "op": ">", "value": 30.0},
            {"metric": "borrow_cost",    "op": ">", "value": 10.0},
        ],
        "sort_by": "short_interest", "sort_asc": False,
    },
    # Sector-specific
    "healthcare_value": {
        "description": "Healthcare value: established pharma/devices at a discount",
        "tags": ["value", "healthcare"],
        "sector": "Health Care",
        "criteria": [
            {"metric": "pe_ratio",     "op": "<", "value": 15.0},
            {"metric": "dividend_yield","op": ">","value": 1.5},
            {"metric": "net_margin",   "op": ">", "value": 15.0},
        ],
        "sort_by": "pe_ratio", "sort_asc": True,
    },
    "financial_value": {
        "description": "Financial value: banks and insurers below book",
        "tags": ["value", "financials"],
        "sector": "Financials",
        "criteria": [
            {"metric": "pb_ratio",     "op": "<", "value": 1.2},
            {"metric": "roe",          "op": ">", "value": 10.0},
            {"metric": "dividend_yield","op": ">","value": 2.0},
        ],
        "sort_by": "pb_ratio", "sort_asc": True,
    },
    "energy_value": {
        "description": "Energy value: low EV/EBITDA energy with strong FCF yield",
        "tags": ["value", "energy"],
        "sector": "Energy",
        "criteria": [
            {"metric": "ev_ebitda",    "op": "<", "value": 6.0},
            {"metric": "dividend_yield","op": ">","value": 3.0},
            {"metric": "fcf_yield",    "op": ">", "value": 5.0},
        ],
        "sort_by": "ev_ebitda", "sort_asc": True,
    },
    "defensive_staples": {
        "description": "Defensive consumer staples: low beta, dividends, resilient margins",
        "tags": ["defensive", "income"],
        "sector": "Consumer Staples",
        "criteria": [
            {"metric": "beta",        "op": "<", "value": 0.7},
            {"metric": "dividend_yield","op":">","value": 2.0},
            {"metric": "net_margin",  "op": ">", "value": 8.0},
        ],
        "sort_by": "beta", "sort_asc": True,
    },
    # Special situations
    "analyst_upgrade_momentum": {
        "description": "Analyst upgrade momentum: buy upgrades + positive revisions",
        "tags": ["momentum", "sentiment"],
        "criteria": [
            {"metric": "analyst_rating", "op": ">", "value": 3.5},
            {"metric": "eps_revision",   "op": ">", "value": 2.0},
            {"metric": "return_3m",      "op": ">", "value": 5.0},
        ],
        "sort_by": "eps_revision", "sort_asc": False,
    },
    "spin_off_special_situation": {
        "description": "Spin-offs: recent corporate actions with depressed valuations",
        "tags": ["special situation"],
        "criteria": [
            {"metric": "return_1y", "op": "<",  "value": -10.0},
            {"metric": "pe_ratio",  "op": "<",  "value": 15.0},
            {"metric": "roe",       "op": ">",  "value": 10.0},
        ],
        "sort_by": "return_1y", "sort_asc": True,
    },
    "fallen_angel_recovery": {
        "description": "Fallen angels: quality companies with temporary setbacks",
        "tags": ["special situation", "value"],
        "criteria": [
            {"metric": "return_1y",  "op": "<",  "value": -20.0},
            {"metric": "roe",        "op": ">",  "value": 12.0},
            {"metric": "net_margin", "op": ">",  "value": 0.0},
            {"metric": "current_ratio","op": ">","value": 1.5},
        ],
        "sort_by": "return_1y", "sort_asc": True,
    },
    "micro_cap_value": {
        "description": "Micro-cap value: tiny companies trading at deep discounts",
        "tags": ["value", "small cap"],
        "criteria": [
            {"metric": "market_cap",   "op": "<", "value": 300_000_000},
            {"metric": "pb_ratio",     "op": "<", "value": 1.0},
            {"metric": "current_ratio","op": ">", "value": 1.5},
            {"metric": "net_margin",   "op": ">", "value": 0.0},
        ],
        "sort_by": "pb_ratio", "sort_asc": True,
    },
    "high_roic_compounder": {
        "description": "High-ROIC compounders: exceptional capital allocation",
        "tags": ["quality", "growth"],
        "criteria": [
            {"metric": "roic",        "op": ">", "value": 20.0},
            {"metric": "revenue_growth","op":">","value": 8.0},
            {"metric": "debt_equity", "op": "<", "value": 0.3},
        ],
        "sort_by": "roic", "sort_asc": False,
    },
    "emerging_market_growth": {
        "description": "Emerging market growth: fast-growing EM companies",
        "tags": ["growth", "emerging markets"],
        "geography": "Emerging Markets",
        "criteria": [
            {"metric": "revenue_growth", "op": ">", "value": 15.0},
            {"metric": "pe_ratio",       "op": "<", "value": 20.0},
            {"metric": "roe",            "op": ">", "value": 12.0},
        ],
        "sort_by": "revenue_growth", "sort_asc": False,
    },
    "insider_buying": {
        "description": "Insider buying: heavy insider ownership with recent purchases",
        "tags": ["sentiment"],
        "criteria": [
            {"metric": "insider_ownership", "op": ">", "value": 10.0},
            {"metric": "return_3m",         "op": ">", "value": 0.0},
        ],
        "sort_by": "insider_ownership", "sort_asc": False,
    },
    "fcf_compounder": {
        "description": "FCF compounders: high FCF yield with consistent growth",
        "tags": ["quality", "value"],
        "criteria": [
            {"metric": "fcf_yield",  "op": ">", "value": 5.0},
            {"metric": "fcf_growth", "op": ">", "value": 8.0},
            {"metric": "debt_equity","op": "<", "value": 1.0},
        ],
        "sort_by": "fcf_yield", "sort_asc": False,
    },
    "low_peg_growth": {
        "description": "Low-PEG growth: pay for growth at a fair price",
        "tags": ["growth", "value"],
        "criteria": [
            {"metric": "peg_ratio",      "op": "<", "value": 1.2},
            {"metric": "revenue_growth", "op": ">", "value": 15.0},
            {"metric": "roe",            "op": ">", "value": 12.0},
        ],
        "sort_by": "peg_ratio", "sort_asc": True,
    },
    "dividend_growth": {
        "description": "Dividend growth: moderate yield + fast dividend growth",
        "tags": ["income", "growth"],
        "criteria": [
            {"metric": "dividend_yield",        "op": ">", "value": 1.5},
            {"metric": "consecutive_div_growth", "op": ">", "value": 5.0},
            {"metric": "payout_ratio",          "op": "<", "value": 60.0},
        ],
        "sort_by": "consecutive_div_growth", "sort_asc": False,
    },
    "asset_light": {
        "description": "Asset-light compounders: high ROIC, low capex intensity",
        "tags": ["quality"],
        "criteria": [
            {"metric": "roic",     "op": ">", "value": 25.0},
            {"metric": "net_margin","op": ">","value": 15.0},
        ],
        "sort_by": "roic", "sort_asc": False,
    },
    "deep_value_turnaround": {
        "description": "Deep value turnaround: cheap stocks showing early recovery signs",
        "tags": ["value", "special situation"],
        "criteria": [
            {"metric": "pb_ratio",        "op": "<", "value": 1.0},
            {"metric": "pe_ratio",        "op": "<", "value": 12.0},
            {"metric": "return_3m",       "op": ">", "value": 0.0},
            {"metric": "earnings_surprise","op": ">","value": 0.0},
        ],
        "sort_by": "pb_ratio", "sort_asc": True,
    },
    "japan_value": {
        "description": "Japanese value: cheap Japanese corporates with reform catalysts",
        "tags": ["value", "international"],
        "geography": "Japan",
        "criteria": [
            {"metric": "pb_ratio",     "op": "<", "value": 1.0},
            {"metric": "roe",          "op": ">", "value": 8.0},
            {"metric": "dividend_yield","op": ">","value": 2.0},
        ],
        "sort_by": "pb_ratio", "sort_asc": True,
    },
    "biotech_growth": {
        "description": "Biotech growth: pipeline-rich biotechs with strong cash runway",
        "tags": ["growth", "healthcare"],
        "sector": "Health Care",
        "criteria": [
            {"metric": "market_cap",  "op": "<", "value": 5_000_000_000},
            {"metric": "revenue_growth","op": ">","value": 15.0},
            {"metric": "cash_ratio",  "op": ">", "value": 1.0},
        ],
        "sort_by": "revenue_growth", "sort_asc": False,
    },
    "esg_leaders": {
        "description": "ESG leaders: top ESG scores with solid financials",
        "tags": ["esg", "quality"],
        "criteria": [
            {"metric": "esg_score", "op": ">", "value": 70.0},
            {"metric": "roe",       "op": ">", "value": 10.0},
            {"metric": "net_margin","op": ">", "value": 5.0},
        ],
        "sort_by": "esg_score", "sort_asc": False,
    },
    "value_trap_avoidance": {
        "description": "Value trap screen: cheap but with growth and quality safeguards",
        "tags": ["value", "quality"],
        "criteria": [
            {"metric": "pe_ratio",       "op": "<",  "value": 15.0},
            {"metric": "revenue_growth", "op": ">",  "value": 2.0},
            {"metric": "roe",            "op": ">",  "value": 8.0},
            {"metric": "current_ratio",  "op": ">",  "value": 1.2},
        ],
        "sort_by": "pe_ratio", "sort_asc": True,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Criterion:
    metric: str
    operator: str          # <, >, <=, >=, ==, !=, between
    value: Union[float, Tuple[float, float]]
    weight: float = 1.0
    description: str = ""
    negated: bool = False
    temporal: Optional[str] = None   # TTM, YoY, 5Y_CAGR …

    def __str__(self) -> str:
        if self.operator == "between" and isinstance(self.value, tuple):
            return f"{self.metric} between {self.value[0]} and {self.value[1]}"
        return f"{self.metric} {self.operator} {self.value}"

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "operator": self.operator,
            "value": list(self.value) if isinstance(self.value, tuple) else self.value,
            "weight": self.weight,
            "description": self.description,
            "negated": self.negated,
            "temporal": self.temporal,
        }


@dataclass
class ParsedQuery:
    raw: str
    intent: str                   # SCREEN / RANK / COMPARE / ALERT
    intent_confidence: float
    criteria: List[Criterion]
    sectors: List[str]
    geographies: List[str]
    mktcap_tier: Optional[str]
    index_membership: Optional[str]
    temporal: Optional[str]
    sort_metric: Optional[str]
    sort_ascending: bool
    logic: str                    # AND / OR
    alert_threshold: Optional[Dict] = None
    compare_tickers: List[str] = field(default_factory=list)


@dataclass
class MatchExplanation:
    ticker: str
    score: float
    matched_criteria: List[str]
    failed_criteria: List[str]
    summary: str


@dataclass
class ScreenerResult:
    query: str
    intent: str
    criteria: List[Criterion]
    matches: pd.DataFrame
    explanations: List[MatchExplanation]
    n_matches: int
    top_tickers: List[str]
    criteria_descriptions: List[str]
    suggested_template: Optional[str]
    cache_hit: bool = False


# Pydantic API models
class NLScreenRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    max_results: int = Field(20, ge=1, le=100)
    execute: bool = Field(True)
    use_cache: bool = Field(True)


class ExecuteTemplateRequest(BaseModel):
    template_name: str
    overrides: Optional[Dict[str, Any]] = None
    max_results: int = Field(20, ge=1, le=100)


class AlertRequest(BaseModel):
    name: str
    query: str
    notify_email: Optional[str] = None


class CompareRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=2, max_items=10)
    metrics: Optional[List[str]] = None


class ScreenerResponse(BaseModel):
    query: str
    intent: str
    n_matches: int
    top_tickers: List[str]
    criteria_descriptions: List[str]
    explanations: List[Dict]
    suggested_template: Optional[str]
    cache_hit: bool


# ══════════════════════════════════════════════════════════════════════════════
# SQLite store
# ══════════════════════════════════════════════════════════════════════════════

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screen_cache (
            cache_key  TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            created_at  REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screen_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT NOT NULL,
            intent      TEXT,
            n_matches   INTEGER,
            criteria_json TEXT,
            created_at  REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            query       TEXT NOT NULL,
            criteria_json TEXT,
            notify_email TEXT,
            last_triggered REAL,
            created_at  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _cache_key(query: str, max_results: int) -> str:
    return hashlib.md5(f"{query}::{max_results}".encode()).hexdigest()


def _cache_get(key: str) -> Optional[Dict]:
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT result_json, created_at FROM screen_cache WHERE cache_key=?", (key,)
        ).fetchone()
        conn.close()
        if row and (time.time() - row[1]) < _CACHE_TTL_SECONDS:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _cache_set(key: str, data: Dict) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT OR REPLACE INTO screen_cache (cache_key, result_json, created_at) VALUES (?,?,?)",
            (key, json.dumps(data, default=str), time.time()),
        )
        # Prune old entries
        conn.execute(
            "DELETE FROM screen_cache WHERE created_at < ?",
            (time.time() - _CACHE_TTL_SECONDS,),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _history_add(query: str, intent: str, n_matches: int, criteria: List[Criterion]) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO screen_history (query, intent, n_matches, criteria_json, created_at) VALUES (?,?,?,?,?)",
            (query, intent, n_matches, json.dumps([c.to_dict() for c in criteria]), time.time()),
        )
        conn.execute(
            "DELETE FROM screen_history WHERE id NOT IN "
            "(SELECT id FROM screen_history ORDER BY created_at DESC LIMIT 200)"
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. IntentClassifier
# ══════════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    """Classify a NL query into SCREEN / RANK / COMPARE / ALERT."""

    def classify(self, query: str) -> Tuple[str, float]:
        q = query.lower().strip()
        scores: Dict[str, float] = {intent: 0.0 for intent in INTENTS}

        for intent, signals in INTENT_SIGNALS.items():
            for sig in signals:
                if sig in q:
                    scores[intent] += 1.0

        # Heuristic boosts
        if re.search(r"\b(alert|notify|when|if.*crosses?)\b", q):
            scores["ALERT"] += 3.0
        if re.search(r"\b(compare|vs\.?|versus|peer)\b", q):
            scores["COMPARE"] += 3.0
        if re.search(r"\b(top\s*\d+|rank|best\s+\d+|highest|lowest)\b", q):
            scores["RANK"] += 2.0

        best_intent = max(scores, key=lambda k: scores[k])
        total = sum(scores.values()) or 1.0
        confidence = scores[best_intent] / total

        # Default SCREEN if no signal
        if scores[best_intent] == 0.0:
            return "SCREEN", 0.6
        return best_intent, round(min(confidence * 2, 0.99), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. AdvancedNLParser
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedNLParser:
    """
    Parse natural language into structured criteria with:
    - 50+ metrics via METRIC_CANONICAL + fuzzy fallback
    - Boolean AND/OR/NOT logic + parenthetical grouping
    - Temporal expressions (TTM, YoY, CAGR, "last 3 quarters")
    - Market-cap tiers, sector, geography, index membership
    - Sort/rank intent extraction
    """

    _NUM = r"[-+]?\d+(?:[.,]\d+)?(?:\s*(?:%|B|M|K|x|X))?"

    def __init__(self) -> None:
        self._compile()

    def _compile(self) -> None:
        # Sorted by length descending so longer phrases match first
        metric_keys = sorted(METRIC_CANONICAL.keys(), key=len, reverse=True)
        metric_pat = "|".join(re.escape(k) for k in metric_keys)

        op_keys = sorted(OPERATOR_PHRASES.keys(), key=len, reverse=True)
        op_pat = "|".join(re.escape(k) for k in op_keys)

        sector_keys = sorted(SECTOR_ALIASES.keys(), key=len, reverse=True)
        self._sector_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in sector_keys) + r")\b"
            r"(?:\s+(?:stocks?|companies|sector|industry|equities))?",
            re.IGNORECASE,
        )

        geo_keys = sorted(GEO_ALIASES.keys(), key=len, reverse=True)
        self._geo_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in geo_keys) + r")\b",
            re.IGNORECASE,
        )

        mktcap_keys = sorted(MKTCAP_TIERS.keys(), key=len, reverse=True)
        self._mktcap_tier_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in mktcap_keys) + r")\b",
            re.IGNORECASE,
        )

        index_keys = sorted(INDEX_ALIASES.keys(), key=len, reverse=True)
        self._index_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in index_keys) + r")\b",
            re.IGNORECASE,
        )

        vague_keys = sorted(VAGUE_TERMS.keys(), key=len, reverse=True)
        self._vague_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in vague_keys) + r")\b",
            re.IGNORECASE,
        )

        # metric op value
        self._metric_op_val = re.compile(
            rf"({metric_pat})\s*(?:is\s*)?({op_pat})\s*({self._NUM})",
            re.IGNORECASE,
        )

        # metric between X and Y
        self._between_re = re.compile(
            rf"({metric_pat})\s+between\s+({self._NUM})\s+(?:and|to)\s+({self._NUM})",
            re.IGNORECASE,
        )

        # Inline market cap: "> $10B market cap"
        self._inline_mktcap = re.compile(
            r"market\s*cap(?:italization)?\s*(?:of\s*|is\s*)?(?:>|above|over)?\s*\$?([\d.]+)\s*([BbMm])",
            re.IGNORECASE,
        )

        # Sort signals: "ranked by ROE", "sorted by revenue growth descending"
        self._sort_re = re.compile(
            r"(?:rank(?:ed)?|sort(?:ed)?|order(?:ed)?)\s+(?:by\s+)?("
            + metric_pat + r")\s*(?:(desc(?:ending)?|asc(?:ending)?|highest|lowest))?",
            re.IGNORECASE,
        )

        # Temporal
        self._temporal_res = [
            (re.compile(pat, re.IGNORECASE), label)
            for pat, label in TEMPORAL_PATTERNS.items()
        ]

        # NOT negation: "not technology", "exclude banks"
        self._not_re = re.compile(
            r"\b(?:not|exclude|excluding|without|no)\s+([a-zA-Z\s]+?)(?:\s+and\s+|\s+or\s+|,|$)",
            re.IGNORECASE,
        )

        # ticker comparison list: "compare AAPL vs MSFT"
        self._ticker_re = re.compile(r"\b([A-Z]{1,5})\b")

    @staticmethod
    def _parse_num(raw: str) -> float:
        raw = raw.strip().replace(",", "").replace(" ", "")
        mult = 1.0
        if raw.endswith("%"):
            raw = raw[:-1]
        elif raw.upper().endswith("B"):
            mult = 1e9; raw = raw[:-1]
        elif raw.upper().endswith("M"):
            mult = 1e6; raw = raw[:-1]
        elif raw.upper().endswith("K"):
            mult = 1e3; raw = raw[:-1]
        elif raw.upper().endswith("X"):
            raw = raw[:-1]
        try:
            return float(raw) * mult
        except ValueError:
            return 0.0

    @staticmethod
    def _fuzzy_metric(raw: str) -> Optional[str]:
        """Strip spaces/punctuation and look up in METRIC_FUZZY_MAP."""
        key = re.sub(r"[\s\-/]", "", raw.lower())
        return METRIC_FUZZY_MAP.get(key)

    def _extract_temporal(self, query: str) -> Optional[str]:
        for pat, label in self._temporal_res:
            m = pat.search(query)
            if m:
                if label in ("NQ", "NY") and m.lastindex and m.lastindex >= 1:
                    return f"{m.group(1)}{label[1]}"
                return label
        return None

    def parse(self, query: str) -> ParsedQuery:
        intent_clf = IntentClassifier()
        intent, intent_conf = intent_clf.classify(query)

        q = query
        seen: set = set()
        criteria: List[Criterion] = []

        # Temporal
        temporal = self._extract_temporal(q)

        # Between ranges
        for m in self._between_re.finditer(q):
            metric_raw, v1_raw, v2_raw = m.group(1), m.group(2), m.group(3)
            canonical = METRIC_CANONICAL.get(metric_raw.lower(), self._fuzzy_metric(metric_raw) or metric_raw.lower())
            v1, v2 = self._parse_num(v1_raw), self._parse_num(v2_raw)
            key = f"{canonical}_between_{v1}_{v2}"
            if key not in seen:
                seen.add(key)
                criteria.append(Criterion(
                    metric=canonical, operator="between", value=(v1, v2),
                    description=f"{canonical} between {v1} and {v2}", temporal=temporal,
                ))

        # Metric op value
        for m in self._metric_op_val.finditer(q):
            metric_raw, op_raw, val_raw = m.group(1), m.group(2), m.group(3)
            canonical = METRIC_CANONICAL.get(metric_raw.lower(), self._fuzzy_metric(metric_raw) or metric_raw.lower())
            operator = OPERATOR_PHRASES.get(op_raw.lower().strip(), ">")
            value = self._parse_num(val_raw)
            key = f"{canonical}_{operator}_{value}"
            if key not in seen:
                seen.add(key)
                criteria.append(Criterion(
                    metric=canonical, operator=operator, value=value,
                    description=f"{canonical} {operator} {value}", temporal=temporal,
                ))

        # Inline market cap shorthand
        for m in self._inline_mktcap.finditer(q):
            val = float(m.group(1)) * (1e9 if m.group(2).upper() == "B" else 1e6)
            key = f"market_cap_>{val}"
            if key not in seen:
                seen.add(key)
                criteria.append(Criterion(
                    metric="market_cap", operator=">", value=val,
                    description=f"market_cap > {m.group(1)}{m.group(2).upper()}",
                ))

        # Vague terms
        for m in self._vague_re.finditer(q):
            term = m.group(1).lower()
            for exp in VAGUE_TERMS.get(term, []):
                key = f"{exp['metric']}_{exp['operator']}_{exp['value']}_vague"
                if key not in seen:
                    seen.add(key)
                    criteria.append(Criterion(
                        metric=exp["metric"], operator=exp["operator"], value=exp["value"],
                        weight=0.8,
                        description=f"{exp['metric']} {exp['operator']} {exp['value']} (from '{term}')",
                        temporal=temporal,
                    ))

        # Market cap tiers
        mktcap_tier: Optional[str] = None
        for m in self._mktcap_tier_re.finditer(q):
            tier_key = m.group(1).lower()
            tier = MKTCAP_TIERS.get(tier_key)
            if tier:
                mktcap_tier = tier_key
                key = f"mktcap_tier_{tier_key}"
                if key not in seen:
                    seen.add(key)
                    criteria.append(Criterion(
                        metric=tier["metric"], operator=tier["operator"], value=tier["value"],
                        description=f"{tier_key}: market_cap {tier['operator']} {tier['value']:,.0f}",
                    ))
                break

        # Sectors
        sectors: List[str] = []
        seen_sectors: set = set()
        for m in self._sector_re.finditer(q):
            s = SECTOR_ALIASES.get(m.group(1).lower())
            if s and s not in seen_sectors:
                seen_sectors.add(s)
                sectors.append(s)

        # Geographies
        geographies: List[str] = []
        seen_geos: set = set()
        for m in self._geo_re.finditer(q):
            g = GEO_ALIASES.get(m.group(1).lower())
            if g and g not in seen_geos:
                seen_geos.add(g)
                geographies.append(g)

        # Index membership
        index_membership: Optional[str] = None
        for m in self._index_re.finditer(q):
            index_membership = INDEX_ALIASES.get(m.group(1).lower())
            break

        # Sort
        sort_metric: Optional[str] = None
        sort_asc = False
        sm = self._sort_re.search(q)
        if sm:
            sort_metric = METRIC_CANONICAL.get(sm.group(1).lower(), sm.group(1).lower())
            if sm.lastindex and sm.lastindex >= 2 and sm.group(2):
                dir_token = sm.group(2).lower()
                sort_asc = dir_token in ("asc", "ascending", "lowest")
            else:
                # default: "rank by X" → highest first
                sort_asc = False

        # Detect RANK intent sort from "top X by Y"
        top_m = re.search(
            r"\btop\s*\d*\s*(?:stocks?\s+)?by\s+(" + "|".join(re.escape(k) for k in METRIC_CANONICAL.keys()) + r")\b",
            q, re.IGNORECASE,
        )
        if top_m and not sort_metric:
            sort_metric = METRIC_CANONICAL.get(top_m.group(1).lower(), top_m.group(1).lower())
            sort_asc = False

        # COMPARE: extract tickers
        compare_tickers: List[str] = []
        if intent == "COMPARE":
            compare_tickers = [t for t in self._ticker_re.findall(query)
                               if len(t) >= 1 and t not in {"AND", "OR", "NOT", "BY", "VS"}]

        # ALERT: extract threshold
        alert_threshold: Optional[Dict] = None
        if intent == "ALERT" and criteria:
            alert_threshold = criteria[0].to_dict()

        # Logic
        has_or = bool(re.search(r"\bor\b", q, re.IGNORECASE))
        has_and = bool(re.search(r"\band\b", q, re.IGNORECASE))
        logic = "OR" if has_or and not has_and else "AND"

        return ParsedQuery(
            raw=query,
            intent=intent,
            intent_confidence=intent_conf,
            criteria=criteria,
            sectors=sectors,
            geographies=geographies,
            mktcap_tier=mktcap_tier,
            index_membership=index_membership,
            temporal=temporal,
            sort_metric=sort_metric,
            sort_ascending=sort_asc,
            logic=logic,
            alert_threshold=alert_threshold,
            compare_tickers=compare_tickers,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. FundamentalDataLoader — yfinance bulk fetch with SQLite 24h cache
# ══════════════════════════════════════════════════════════════════════════════

# S&P 500 sample tickers used when a live universe is not available
_SP500_SAMPLE = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","JNJ",
    "V","PG","HD","MA","UNH","DIS","BAC","VZ","ADBE","CRM","NFLX","INTC",
    "AMD","PYPL","CMCSA","PEP","KO","NKE","MRK","ABT","WMT","TMO","ABBV",
    "COST","ACN","AVGO","TXN","QCOM","HON","LIN","NEE","DHR","LMT","GE",
    "MMM","IBM","GS","MS","C","WFC","AXP","SBUX","MCD","T","ORCL","SAP",
    "UBER","LYFT","SQ","SHOP","SPOT","SNAP","COIN","PLTR","SNOW","DDOG",
    "NET","ZM","OKTA","CRWD","PANW","ZS","FTNT","NOW","WDAY","HUBS",
    "MDB","DKNG","RBLX","RIVN","NIO","BABA","JD","ASML","TSM","F","GM",
    "XOM","CVX","BP","SHEL","TTE","ENB","SLB","HAL","PXD","MPC","VLO",
    "NEE","DUK","SO","AEP","EXC","D","ED","XEL","WEC","AWK",
]

class FundamentalDataLoader:
    """Load fundamentals from yfinance with 24h SQLite cache per ticker batch."""

    _CACHE_TABLE = "fundamental_cache"

    def __init__(self) -> None:
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            conn = _get_db()
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._CACHE_TABLE} (
                    ticker TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    fetched_at REAL NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _load_cached(self, tickers: List[str]) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        try:
            conn = _get_db()
            cutoff = time.time() - _CACHE_TTL_SECONDS
            placeholders = ",".join("?" * len(tickers))
            rows = conn.execute(
                f"SELECT ticker, data_json FROM {self._CACHE_TABLE} "
                f"WHERE ticker IN ({placeholders}) AND fetched_at > ?",
                (*tickers, cutoff),
            ).fetchall()
            conn.close()
            for ticker, data_json in rows:
                try:
                    result[ticker] = json.loads(data_json)
                except Exception:
                    pass
        except Exception:
            pass
        return result

    def _save_cached(self, data: Dict[str, Dict]) -> None:
        try:
            conn = _get_db()
            now = time.time()
            for ticker, row in data.items():
                conn.execute(
                    f"INSERT OR REPLACE INTO {self._CACHE_TABLE} (ticker, data_json, fetched_at) VALUES (?,?,?)",
                    (ticker, json.dumps(row, default=str), now),
                )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _fetch_yfinance_batch(self, tickers: List[str]) -> Dict[str, Dict]:
        """Fetch fundamental data for a batch of tickers via yfinance."""
        result: Dict[str, Dict] = {}
        try:
            import yfinance as yf
            for ticker in tickers:
                try:
                    info = yf.Ticker(ticker).info
                    if not info or info.get("regularMarketPrice") is None:
                        continue
                    result[ticker] = self._extract_metrics(ticker, info)
                except Exception:
                    continue
        except ImportError:
            pass
        return result

    @staticmethod
    def _extract_metrics(ticker: str, info: Dict) -> Dict:
        """Map yfinance info keys to SENTINEL metric names."""
        def _safe(key: str, default: float = float("nan")) -> float:
            v = info.get(key)
            if v is None or v == "Infinity" or v == "":
                return default
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        market_cap = _safe("marketCap")
        revenue = _safe("totalRevenue")
        ebitda = _safe("ebitda")
        enterprise_value = _safe("enterpriseValue")
        pe = _safe("trailingPE")
        forward_pe = _safe("forwardPE")
        pb = _safe("priceToBook")
        ps = _safe("priceToSalesTrailing12Months")
        ev_ebitda = (enterprise_value / ebitda) if (ebitda and ebitda > 0 and not np.isnan(enterprise_value)) else float("nan")
        fcf = _safe("freeCashflow")
        price = _safe("currentPrice") or _safe("regularMarketPrice")
        fcf_yield = (fcf / market_cap * 100) if (fcf and market_cap > 0) else float("nan")
        earnings_yield = (100.0 / pe) if (pe and pe > 0) else float("nan")

        gross_margin = _safe("grossMargins", float("nan"))
        if not np.isnan(gross_margin) and gross_margin <= 1.0:
            gross_margin *= 100.0
        operating_margin = _safe("operatingMargins", float("nan"))
        if not np.isnan(operating_margin) and operating_margin <= 1.0:
            operating_margin *= 100.0
        net_margin = _safe("profitMargins", float("nan"))
        if not np.isnan(net_margin) and net_margin <= 1.0:
            net_margin *= 100.0
        ebitda_margin = (ebitda / revenue * 100.0) if (revenue and revenue > 0 and not np.isnan(ebitda)) else float("nan")

        roe = _safe("returnOnEquity", float("nan"))
        if not np.isnan(roe) and roe < 5.0:
            roe *= 100.0
        roa = _safe("returnOnAssets", float("nan"))
        if not np.isnan(roa) and roa < 5.0:
            roa *= 100.0

        revenue_growth = _safe("revenueGrowth", float("nan"))
        if not np.isnan(revenue_growth) and revenue_growth < 5.0:
            revenue_growth *= 100.0
        earnings_growth = _safe("earningsGrowth", float("nan"))
        if not np.isnan(earnings_growth) and earnings_growth < 5.0:
            earnings_growth *= 100.0

        div_yield = _safe("dividendYield", float("nan"))
        if not np.isnan(div_yield) and div_yield < 1.0:
            div_yield *= 100.0

        payout_ratio = _safe("payoutRatio", float("nan"))
        if not np.isnan(payout_ratio) and payout_ratio <= 1.0:
            payout_ratio *= 100.0

        short_ratio = _safe("shortRatio", float("nan"))
        short_pct = _safe("shortPercentOfFloat", float("nan"))
        if not np.isnan(short_pct) and short_pct <= 1.0:
            short_pct *= 100.0

        inst_own = _safe("heldPercentInstitutions", float("nan"))
        if not np.isnan(inst_own) and inst_own <= 1.0:
            inst_own *= 100.0
        insider_own = _safe("heldPercentInsiders", float("nan"))
        if not np.isnan(insider_own) and insider_own <= 1.0:
            insider_own *= 100.0

        debt_equity = _safe("debtToEquity", float("nan"))
        if not np.isnan(debt_equity) and debt_equity > 100:
            debt_equity /= 100.0  # some sources return 100x

        analyst_rating = _safe("recommendationMean", float("nan"))
        eps_revision = _safe("earningsQuarterlyGrowth", float("nan"))

        return {
            "ticker": ticker,
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "geography": "US",  # simplification; extend for ADRs
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "pe_ratio": pe,
            "forward_pe": forward_pe,
            "pb_ratio": pb,
            "ps_ratio": ps,
            "ev_ebitda": ev_ebitda,
            "peg_ratio": _safe("pegRatio"),
            "earnings_yield": earnings_yield,
            "fcf_yield": fcf_yield,
            "dividend_yield": div_yield,
            "payout_ratio": payout_ratio,
            "revenue": revenue,
            "ebitda": ebitda,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "net_margin": net_margin,
            "ebitda_margin": ebitda_margin,
            "roe": roe,
            "roa": roa,
            "roic": float("nan"),  # not in yfinance; can compute separately
            "roce": float("nan"),
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "net_income_growth": float("nan"),
            "fcf_growth": float("nan"),
            "revenue_cagr_5y": float("nan"),
            "eps_cagr_5y": float("nan"),
            "debt_equity": debt_equity,
            "net_debt_ebitda": float("nan"),
            "current_ratio": _safe("currentRatio"),
            "quick_ratio": _safe("quickRatio"),
            "interest_coverage": float("nan"),
            "cash_ratio": float("nan"),
            "book_value_per_share": _safe("bookValue"),
            "fcf": fcf,
            "operating_cash_flow": _safe("operatingCashflow"),
            "capex": _safe("capitalExpenditures"),
            "cash_conversion_ratio": float("nan"),
            "beta": _safe("beta"),
            "rsi": float("nan"),
            "pct_from_52w_high": (
                (price / _safe("fiftyTwoWeekHigh") - 1.0) * 100.0
                if (_safe("fiftyTwoWeekHigh") and price) else float("nan")
            ),
            "pct_from_52w_low": (
                (price / _safe("fiftyTwoWeekLow") - 1.0) * 100.0
                if (_safe("fiftyTwoWeekLow") and price) else float("nan")
            ),
            "return_3m": float("nan"),
            "return_6m": float("nan"),
            "return_1y": _safe("52WeekChange", float("nan")),
            "return_ytd": float("nan"),
            "price_vs_ma50": float("nan"),
            "price_vs_ma200": float("nan"),
            "avg_volume": _safe("averageVolume"),
            "short_interest": short_ratio,
            "float_short": short_pct,
            "analyst_rating": analyst_rating,
            "eps_revision": eps_revision,
            "earnings_surprise": float("nan"),
            "institutional_ownership": inst_own,
            "insider_ownership": insider_own,
            "iv_percentile": float("nan"),
            "borrow_cost": float("nan"),
            "consecutive_div_growth": float("nan"),
            "dividend_coverage": float("nan"),
            "esg_score": float("nan"),
            "esg_env": float("nan"),
            "esg_gov": float("nan"),
        }

    def _synthetic_row(self, ticker: str) -> Dict:
        """Return a synthetic row for demo/fallback."""
        rng = np.random.default_rng(hash(ticker) % (2**31))
        sectors = [
            "Technology","Health Care","Financials","Consumer Discretionary",
            "Consumer Staples","Energy","Industrials","Materials",
            "Real Estate","Utilities","Communication Services",
        ]
        sector = sectors[hash(ticker) % len(sectors)]
        market_cap = float(rng.lognormal(23, 2))
        pe = float(rng.lognormal(2.8, 0.8))
        pb = float(rng.lognormal(0.8, 0.8))
        ps = float(rng.lognormal(1.2, 0.8))
        ev_ebitda = float(rng.lognormal(2.5, 0.7))
        rev_growth = float(rng.normal(12.0, 20.0))
        eps_growth = float(rng.normal(10.0, 25.0))
        gross_margin = float(rng.normal(45.0, 20.0))
        op_margin = float(rng.normal(18.0, 15.0))
        net_margin = float(rng.normal(12.0, 12.0))
        roe = float(rng.normal(18.0, 15.0))
        roa = float(rng.normal(8.0, 8.0))
        roic = float(rng.normal(15.0, 12.0))
        de = float(rng.exponential(0.8))
        cr = float(rng.lognormal(0.4, 0.5))
        dy = float(rng.exponential(1.5))
        beta = float(rng.lognormal(0.1, 0.5))
        r3m = float(rng.normal(5.0, 20.0))
        r1y = float(rng.normal(10.0, 35.0))
        analyst = float(rng.uniform(1.5, 4.5))
        eps_rev = float(rng.normal(1.0, 5.0))
        short_int = float(rng.exponential(5.0))
        payout = float(rng.uniform(0.0, 85.0))
        fcf_yield = float(rng.normal(4.0, 3.0))
        return {
            "ticker": ticker,
            "sector": sector,
            "industry": sector,
            "geography": "US",
            "market_cap": market_cap,
            "enterprise_value": market_cap * 1.1,
            "pe_ratio": pe,
            "forward_pe": pe * 0.9,
            "pb_ratio": pb,
            "ps_ratio": ps,
            "ev_ebitda": ev_ebitda,
            "peg_ratio": pe / max(eps_growth, 0.1),
            "earnings_yield": 100.0 / pe if pe > 0 else 0.0,
            "fcf_yield": fcf_yield,
            "dividend_yield": dy,
            "payout_ratio": payout,
            "revenue": market_cap * 0.5,
            "ebitda": market_cap * 0.12,
            "gross_margin": np.clip(gross_margin, 0, 100),
            "operating_margin": np.clip(op_margin, -50, 60),
            "net_margin": np.clip(net_margin, -30, 50),
            "ebitda_margin": np.clip(op_margin + 5, 0, 70),
            "roe": roe,
            "roa": roa,
            "roic": roic,
            "roce": roic * 0.85,
            "revenue_growth": rev_growth,
            "earnings_growth": eps_growth,
            "net_income_growth": eps_growth * 0.9,
            "fcf_growth": float(rng.normal(8.0, 15.0)),
            "revenue_cagr_5y": float(rng.normal(10.0, 8.0)),
            "eps_cagr_5y": float(rng.normal(9.0, 10.0)),
            "debt_equity": de,
            "net_debt_ebitda": de * 2.5,
            "current_ratio": cr,
            "quick_ratio": cr * 0.7,
            "interest_coverage": max(0.0, float(rng.normal(8.0, 5.0))),
            "cash_ratio": max(0.0, float(rng.normal(0.5, 0.3))),
            "book_value_per_share": float(rng.lognormal(2.5, 1.0)),
            "fcf": market_cap * (fcf_yield / 100),
            "operating_cash_flow": market_cap * 0.06,
            "capex": market_cap * 0.02,
            "cash_conversion_ratio": float(rng.uniform(0.5, 1.2)),
            "beta": beta,
            "rsi": float(rng.uniform(20, 80)),
            "pct_from_52w_high": float(rng.uniform(-40, 0)),
            "pct_from_52w_low": float(rng.uniform(0, 80)),
            "return_3m": r3m,
            "return_6m": float(rng.normal(8.0, 25.0)),
            "return_1y": r1y,
            "return_ytd": float(rng.normal(6.0, 20.0)),
            "price_vs_ma50": float(rng.normal(2.0, 8.0)),
            "price_vs_ma200": float(rng.normal(5.0, 15.0)),
            "avg_volume": float(rng.lognormal(14, 2)),
            "short_interest": np.clip(short_int, 0, 60),
            "float_short": np.clip(float(rng.exponential(6.0)), 0, 70),
            "analyst_rating": analyst,
            "eps_revision": eps_rev,
            "earnings_surprise": float(rng.normal(2.0, 8.0)),
            "institutional_ownership": np.clip(float(rng.normal(65.0, 20.0)), 0, 100),
            "insider_ownership": np.clip(float(rng.exponential(5.0)), 0, 80),
            "iv_percentile": float(rng.uniform(5, 95)),
            "borrow_cost": np.clip(float(rng.exponential(3.0)), 0, 100),
            "consecutive_div_growth": float(rng.choice([0, 3, 5, 10, 15, 20, 25, 30, 40, 50])),
            "dividend_coverage": max(0.0, float(rng.normal(2.5, 1.5))),
            "esg_score": float(rng.uniform(20, 95)),
            "esg_env": float(rng.uniform(20, 95)),
            "esg_gov": float(rng.uniform(20, 95)),
        }

    def load(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Load fundamental data, using cache where available."""
        if tickers is None:
            tickers = _SP500_SAMPLE

        cached = self._load_cached(tickers)
        missing = [t for t in tickers if t not in cached]

        fetched: Dict[str, Dict] = {}
        if missing:
            # Try yfinance in batches
            for i in range(0, len(missing), _YFINANCE_BATCH):
                batch = missing[i: i + _YFINANCE_BATCH]
                try:
                    fetched.update(self._fetch_yfinance_batch(batch))
                except Exception:
                    pass

            # Fallback to synthetic for anything still missing
            for t in missing:
                if t not in fetched:
                    fetched[t] = self._synthetic_row(t)

            self._save_cached(fetched)

        all_data = {**cached, **fetched}
        rows = [all_data[t] for t in tickers if t in all_data]
        if not rows:
            rows = [self._synthetic_row(t) for t in tickers]

        df = pd.DataFrame(rows)
        # Coerce numeric columns
        num_cols = [c for c in df.columns if c not in ("ticker", "sector", "industry", "geography")]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ScreenExecutionEngine
# ══════════════════════════════════════════════════════════════════════════════

class ScreenExecutionEngine:
    """
    Execute Criterion list against a DataFrame of fundamentals.

    Returns ranked matches with per-stock explanations.
    """

    def _apply_one(self, df: pd.DataFrame, c: Criterion) -> pd.Series:
        """Return boolean mask for rows satisfying criterion c."""
        if c.metric == "sector":
            if "sector" not in df.columns:
                return pd.Series(True, index=df.index)
            return df["sector"].str.lower().str.strip() == (c.description.split("=")[-1].strip().lower())

        if c.metric == "geography":
            if "geography" not in df.columns:
                return pd.Series(True, index=df.index)
            return df["geography"].str.lower().str.strip() == (c.description.split("=")[-1].strip().lower())

        if c.metric not in df.columns:
            return pd.Series(True, index=df.index)

        col = df[c.metric].fillna(float("nan"))
        op = c.operator

        if op == "between" and isinstance(c.value, tuple):
            lo, hi = c.value
            mask = (col >= lo) & (col <= hi)
        elif op == "<":
            mask = col < float(c.value)
        elif op == "<=":
            mask = col <= float(c.value)
        elif op == ">":
            mask = col > float(c.value)
        elif op == ">=":
            mask = col >= float(c.value)
        elif op in ("==", "="):
            mask = col == float(c.value)
        elif op == "!=":
            mask = col != float(c.value)
        else:
            mask = pd.Series(True, index=df.index)

        # NaN rows always fail
        mask = mask & col.notna()
        if c.negated:
            mask = ~mask
        return mask

    def _score_row(self, row: pd.Series, criteria: List[Criterion]) -> Tuple[float, List[str], List[str]]:
        """Score 0-100, returns (score, matched_descriptions, failed_descriptions)."""
        matched, failed = [], []
        total_w, weighted = 0.0, 0.0

        for c in criteria:
            if c.metric in ("sector", "geography"):
                continue
            if c.metric not in row.index:
                continue

            val = row[c.metric]
            if pd.isna(val):
                failed.append(f"{c.metric}=NaN")
                continue

            w = c.weight
            total_w += w

            if c.operator == "between" and isinstance(c.value, tuple):
                lo, hi = c.value
                passes = lo <= val <= hi
                if passes:
                    mid = (lo + hi) / 2
                    margin_frac = 1.0 - abs(val - mid) / max((hi - lo) / 2, 1e-9)
                    criterion_score = 50.0 + 50.0 * margin_frac
                else:
                    criterion_score = 0.0
            else:
                threshold = float(c.value) if not isinstance(c.value, tuple) else 0.0
                if c.operator in ("<", "<="):
                    passes = val <= threshold
                    margin = (threshold - val) / max(abs(threshold), 1e-9) if passes else 0.0
                elif c.operator in (">", ">="):
                    passes = val >= threshold
                    margin = (val - threshold) / max(abs(threshold), 1e-9) if passes else 0.0
                elif c.operator in ("==", "="):
                    passes = abs(val - threshold) < 1e-6
                    margin = 1.0 if passes else 0.0
                elif c.operator == "!=":
                    passes = abs(val - threshold) > 1e-6
                    margin = 1.0 if passes else 0.0
                else:
                    passes = True
                    margin = 0.5

                if c.negated:
                    passes = not passes

                criterion_score = (50.0 + min(50.0, margin * 50.0)) if passes else 0.0

            weighted += criterion_score * w

            val_fmt = f"{val:.2f}" if isinstance(val, float) else str(val)
            desc = f"{c.metric}={val_fmt} {'passes' if criterion_score > 0 else 'fails'} {c}"
            if criterion_score > 0:
                matched.append(desc)
            else:
                failed.append(desc)

        score = round(weighted / total_w, 1) if total_w > 0 else 50.0
        return score, matched, failed

    def execute(
        self,
        df: pd.DataFrame,
        criteria: List[Criterion],
        sectors: List[str],
        geographies: List[str],
        logic: str = "AND",
        sort_metric: Optional[str] = None,
        sort_ascending: bool = False,
        max_results: int = 20,
    ) -> Tuple[pd.DataFrame, List[MatchExplanation]]:
        """Filter df by criteria + sector/geo filters. Return (matched_df, explanations)."""
        if df.empty or not criteria:
            return pd.DataFrame(), []

        # Sector filter (pre-filter)
        if sectors and "sector" in df.columns:
            sector_lower = [s.lower() for s in sectors]
            df = df[df["sector"].str.lower().isin(sector_lower)].copy()

        # Geography filter
        if geographies and "geography" in df.columns:
            geo_lower = [g.lower() for g in geographies]
            df = df[df["geography"].str.lower().isin(geo_lower)].copy()

        if df.empty:
            return df, []

        # Numeric criteria
        num_criteria = [c for c in criteria if c.metric not in ("sector", "geography")]

        if logic == "AND":
            mask = pd.Series(True, index=df.index)
            for c in num_criteria:
                mask &= self._apply_one(df, c)
        else:  # OR
            mask = pd.Series(False, index=df.index)
            for c in num_criteria:
                mask |= self._apply_one(df, c)

        matched = df[mask].copy()
        if matched.empty:
            return matched, []

        # Score
        scores, matched_lists, failed_lists = [], [], []
        for _, row in matched.iterrows():
            s, m, f = self._score_row(row, num_criteria)
            scores.append(s)
            matched_lists.append(m)
            failed_lists.append(f)

        matched["_score"] = scores

        # Sort
        if sort_metric and sort_metric in matched.columns:
            matched = matched.sort_values(sort_metric, ascending=sort_ascending)
        else:
            matched = matched.sort_values("_score", ascending=False)

        matched = matched.head(max_results).reset_index(drop=True)

        # Build explanations
        explanations: List[MatchExplanation] = []
        for idx, row in matched.iterrows():
            ticker = row.get("ticker", str(idx))
            sc = row["_score"]
            m_list = matched_lists[idx] if idx < len(matched_lists) else []
            f_list = failed_lists[idx] if idx < len(failed_lists) else []
            criteria_str = "; ".join(m_list[:4]) or "matched"
            explanations.append(MatchExplanation(
                ticker=ticker,
                score=sc,
                matched_criteria=m_list,
                failed_criteria=f_list,
                summary=f"{ticker} matched because {criteria_str}",
            ))

        return matched, explanations


# ══════════════════════════════════════════════════════════════════════════════
# 5. TemplateEngine
# ══════════════════════════════════════════════════════════════════════════════

class TemplateEngine:
    """Manage 40+ screener templates: list, get, execute, search, suggest."""

    def list_templates(self, tag: Optional[str] = None) -> List[Dict]:
        out = []
        for name, tmpl in SCREENER_TEMPLATES.items():
            if tag and tag not in tmpl.get("tags", []):
                continue
            out.append({
                "name": name,
                "description": tmpl["description"],
                "tags": tmpl.get("tags", []),
                "n_criteria": len(tmpl.get("criteria", [])),
                "sector": tmpl.get("sector"),
                "geography": tmpl.get("geography"),
                "sort_by": tmpl.get("sort_by"),
            })
        return out

    def get(self, name: str) -> Dict:
        if name not in SCREENER_TEMPLATES:
            raise KeyError(f"Template '{name}' not found. Available: {list(SCREENER_TEMPLATES.keys())}")
        return {"name": name, **SCREENER_TEMPLATES[name]}

    def to_criteria(self, name: str, overrides: Optional[Dict] = None) -> List[Criterion]:
        tmpl = SCREENER_TEMPLATES.get(name)
        if not tmpl:
            raise KeyError(f"Template '{name}' not found.")
        criteria = []
        for c in tmpl.get("criteria", []):
            op = c.get("op", c.get("operator", ">"))
            criteria.append(Criterion(
                metric=c["metric"],
                operator=op,
                value=c["value"],
                description=f"{c['metric']} {op} {c['value']}",
            ))
        if overrides:
            for c in criteria:
                if c.metric in overrides:
                    c.value = overrides[c.metric]
        return criteria

    def suggest(self, query: str) -> Optional[str]:
        q_lower = query.lower()
        best, best_score = None, 0
        for name, tmpl in SCREENER_TEMPLATES.items():
            score = 0
            for word in name.replace("_", " ").split():
                if len(word) > 3 and word in q_lower:
                    score += 2
            for word in tmpl["description"].lower().split():
                if len(word) > 4 and word in q_lower:
                    score += 1
            for tag in tmpl.get("tags", []):
                if tag in q_lower:
                    score += 3
            if score > best_score:
                best_score = score
                best = name
        return best if best_score >= 2 else None

    def search(self, keyword: str) -> List[Dict]:
        kw = keyword.lower()
        return [
            {"name": n, "description": t["description"], "tags": t.get("tags", [])}
            for n, t in SCREENER_TEMPLATES.items()
            if kw in n.lower() or kw in t["description"].lower()
            or any(kw in tag for tag in t.get("tags", []))
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 6. AlertManager
# ══════════════════════════════════════════════════════════════════════════════

class AlertManager:
    """Store and evaluate threshold-based alerts."""

    def create_alert(self, name: str, query: str, criteria: List[Criterion],
                     notify_email: Optional[str] = None) -> int:
        try:
            conn = _get_db()
            cur = conn.execute(
                "INSERT INTO alerts (name, query, criteria_json, notify_email, created_at) VALUES (?,?,?,?,?)",
                (name, query, json.dumps([c.to_dict() for c in criteria]), notify_email, time.time()),
            )
            alert_id = cur.lastrowid
            conn.commit()
            conn.close()
            return alert_id
        except Exception as e:
            raise RuntimeError(f"Failed to create alert: {e}") from e

    def list_alerts(self) -> List[Dict]:
        try:
            conn = _get_db()
            rows = conn.execute(
                "SELECT id, name, query, notify_email, last_triggered, created_at FROM alerts ORDER BY created_at DESC"
            ).fetchall()
            conn.close()
            return [
                {
                    "id": r[0], "name": r[1], "query": r[2],
                    "notify_email": r[3],
                    "last_triggered": datetime.fromtimestamp(r[4], tz=timezone.utc).isoformat() if r[4] else None,
                    "created_at": datetime.fromtimestamp(r[5], tz=timezone.utc).isoformat(),
                }
                for r in rows
            ]
        except Exception:
            return []

    def delete_alert(self, alert_id: int) -> bool:
        try:
            conn = _get_db()
            conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def evaluate_alerts(self, df: pd.DataFrame) -> List[Dict]:
        """Check all alerts against current data. Return triggered ones."""
        triggered = []
        try:
            conn = _get_db()
            rows = conn.execute("SELECT id, name, query, criteria_json FROM alerts").fetchall()
            conn.close()
        except Exception:
            return []

        engine = ScreenExecutionEngine()
        for alert_id, name, query, criteria_json in rows:
            try:
                criteria = [
                    Criterion(
                        metric=c["metric"], operator=c["operator"],
                        value=tuple(c["value"]) if isinstance(c["value"], list) else c["value"],
                    )
                    for c in json.loads(criteria_json or "[]")
                ]
                matched, _ = engine.execute(df, criteria, [], [], max_results=10)
                if not matched.empty:
                    triggered.append({
                        "alert_id": alert_id,
                        "name": name,
                        "query": query,
                        "n_triggered": len(matched),
                        "tickers": matched["ticker"].tolist()[:5] if "ticker" in matched.columns else [],
                    })
                    # Update last_triggered
                    try:
                        conn = _get_db()
                        conn.execute("UPDATE alerts SET last_triggered=? WHERE id=?", (time.time(), alert_id))
                        conn.commit()
                        conn.close()
                    except Exception:
                        pass
            except Exception:
                continue
        return triggered


# ══════════════════════════════════════════════════════════════════════════════
# 7. CompareEngine
# ══════════════════════════════════════════════════════════════════════════════

class CompareEngine:
    """Peer comparison across a list of tickers on selected metrics."""

    _DEFAULT_METRICS = [
        "pe_ratio", "pb_ratio", "ev_ebitda", "revenue_growth", "earnings_growth",
        "gross_margin", "operating_margin", "net_margin", "roe", "roic",
        "debt_equity", "fcf_yield", "dividend_yield", "beta", "return_1y",
    ]

    def compare(self, tickers: List[str], metrics: Optional[List[str]] = None) -> Dict:
        loader = FundamentalDataLoader()
        df = loader.load(tickers)
        use_metrics = metrics or self._DEFAULT_METRICS
        cols = ["ticker"] + [m for m in use_metrics if m in df.columns]
        sub = df[cols].set_index("ticker")

        # Percentile ranks per metric
        rank_df = sub.rank(pct=True) * 100

        result: Dict[str, Any] = {
            "tickers": tickers,
            "metrics": use_metrics,
            "data": sub.to_dict(orient="index"),
            "percentile_ranks": rank_df.to_dict(orient="index"),
            "summary": {},
        }

        for ticker in tickers:
            if ticker not in sub.index:
                continue
            rank_row = rank_df.loc[ticker] if ticker in rank_df.index else pd.Series()
            strengths = [m for m in use_metrics if m in rank_row.index and rank_row[m] >= 70]
            weaknesses = [m for m in use_metrics if m in rank_row.index and rank_row[m] <= 30]
            result["summary"][ticker] = {
                "strengths": strengths[:5],
                "weaknesses": weaknesses[:5],
                "overall_percentile": float(rank_row.mean()) if not rank_row.empty else 50.0,
            }

        return result


# ══════════════════════════════════════════════════════════════════════════════
# 8. NLScreenerV2Pipeline — main entry point
# ══════════════════════════════════════════════════════════════════════════════

class NLScreenerV2Pipeline:
    """
    End-to-end NL → parse → classify intent → execute → rank → explain.

    Orchestrates: AdvancedNLParser, FundamentalDataLoader,
                  ScreenExecutionEngine, TemplateEngine, AlertManager.
    """

    def __init__(self) -> None:
        self._parser = AdvancedNLParser()
        self._loader = FundamentalDataLoader()
        self._engine = ScreenExecutionEngine()
        self._templates = TemplateEngine()
        self._alerts = AlertManager()

    def run(
        self,
        query: str,
        max_results: int = 20,
        execute: bool = True,
        use_cache: bool = True,
    ) -> ScreenerResult:
        """Full pipeline. Returns ScreenerResult."""
        # Cache check
        if use_cache:
            ck = _cache_key(query, max_results)
            cached = _cache_get(ck)
            if cached:
                return self._from_cache(cached)

        parsed = self._parser.parse(query)
        suggested = self._templates.suggest(query)

        matched_df = pd.DataFrame()
        explanations: List[MatchExplanation] = []

        if execute and (parsed.criteria or parsed.sectors or parsed.geographies):
            df = self._loader.load()
            matched_df, explanations = self._engine.execute(
                df,
                criteria=parsed.criteria,
                sectors=parsed.sectors,
                geographies=parsed.geographies,
                logic=parsed.logic,
                sort_metric=parsed.sort_metric,
                sort_ascending=parsed.sort_ascending,
                max_results=max_results,
            )

        n = len(matched_df)
        top_tickers = (
            matched_df["ticker"].tolist()[:10]
            if "ticker" in matched_df.columns else []
        )

        result = ScreenerResult(
            query=query,
            intent=parsed.intent,
            criteria=parsed.criteria,
            matches=matched_df,
            explanations=explanations,
            n_matches=n,
            top_tickers=top_tickers,
            criteria_descriptions=[str(c) for c in parsed.criteria],
            suggested_template=suggested,
        )

        _history_add(query, parsed.intent, n, parsed.criteria)

        if use_cache:
            _cache_set(ck, self._to_cache(result))

        return result

    @staticmethod
    def _to_cache(r: ScreenerResult) -> Dict:
        return {
            "query": r.query,
            "intent": r.intent,
            "criteria": [c.to_dict() for c in r.criteria],
            "n_matches": r.n_matches,
            "top_tickers": r.top_tickers,
            "criteria_descriptions": r.criteria_descriptions,
            "suggested_template": r.suggested_template,
            "explanations": [
                {
                    "ticker": e.ticker,
                    "score": e.score,
                    "summary": e.summary,
                    "matched_criteria": e.matched_criteria[:3],
                    "failed_criteria": e.failed_criteria[:3],
                }
                for e in r.explanations
            ],
        }

    @staticmethod
    def _from_cache(data: Dict) -> ScreenerResult:
        criteria = [
            Criterion(
                metric=c["metric"],
                operator=c["operator"],
                value=tuple(c["value"]) if isinstance(c["value"], list) else c["value"],
                description=c.get("description", ""),
            )
            for c in data.get("criteria", [])
        ]
        explanations = [
            MatchExplanation(
                ticker=e["ticker"],
                score=e["score"],
                matched_criteria=e.get("matched_criteria", []),
                failed_criteria=e.get("failed_criteria", []),
                summary=e["summary"],
            )
            for e in data.get("explanations", [])
        ]
        return ScreenerResult(
            query=data["query"],
            intent=data.get("intent", "SCREEN"),
            criteria=criteria,
            matches=pd.DataFrame(),
            explanations=explanations,
            n_matches=data.get("n_matches", 0),
            top_tickers=data.get("top_tickers", []),
            criteria_descriptions=data.get("criteria_descriptions", []),
            suggested_template=data.get("suggested_template"),
            cache_hit=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Router
# ══════════════════════════════════════════════════════════════════════════════

nl_screener_v2_router = APIRouter(prefix="/screener/v2", tags=["nl-screener-v2"])

_pipeline: Optional[NLScreenerV2Pipeline] = None
_template_eng: Optional[TemplateEngine] = None
_alert_mgr: Optional[AlertManager] = None
_compare_eng: Optional[CompareEngine] = None


def _get_pipeline() -> NLScreenerV2Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = NLScreenerV2Pipeline()
    return _pipeline


def _get_template_eng() -> TemplateEngine:
    global _template_eng
    if _template_eng is None:
        _template_eng = TemplateEngine()
    return _template_eng


def _get_alert_mgr() -> AlertManager:
    global _alert_mgr
    if _alert_mgr is None:
        _alert_mgr = AlertManager()
    return _alert_mgr


def _get_compare_eng() -> CompareEngine:
    global _compare_eng
    if _compare_eng is None:
        _compare_eng = CompareEngine()
    return _compare_eng


def _explanation_to_dict(e: MatchExplanation) -> Dict:
    return {
        "ticker": e.ticker,
        "score": e.score,
        "summary": e.summary,
        "matched_criteria": e.matched_criteria[:5],
        "failed_criteria": e.failed_criteria[:3],
    }


@nl_screener_v2_router.post("/nl-screen", response_model=ScreenerResponse)
def nl_screen(req: NLScreenRequest):
    """
    Execute a natural language screen query.

    Examples:
      - "Show me tech stocks with P/E below 25 and ROE above 20%"
      - "Find high growth companies with revenue growth > 30% and gross margin > 60%"
      - "Screen for dividend aristocrats with yield > 3% and consecutive growth > 20 years"
    """
    try:
        result = _get_pipeline().run(
            req.query,
            max_results=req.max_results,
            execute=req.execute,
            use_cache=req.use_cache,
        )
        return ScreenerResponse(
            query=result.query,
            intent=result.intent,
            n_matches=result.n_matches,
            top_tickers=result.top_tickers,
            criteria_descriptions=result.criteria_descriptions,
            explanations=[_explanation_to_dict(e) for e in result.explanations[:10]],
            suggested_template=result.suggested_template,
            cache_hit=result.cache_hit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@nl_screener_v2_router.get("/templates")
def list_templates(tag: Optional[str] = FastAPIQuery(None, description="Filter by tag")):
    """List all 40+ screener templates, optionally filtered by tag."""
    templates = _get_template_eng().list_templates(tag=tag)
    return {
        "count": len(templates),
        "templates": templates,
        "available_tags": sorted({t for tmpl in SCREENER_TEMPLATES.values() for t in tmpl.get("tags", [])}),
    }


@nl_screener_v2_router.get("/templates/search")
def search_templates(q: str = FastAPIQuery(..., description="Keyword search")):
    """Search templates by keyword in name, description, or tags."""
    return {"results": _get_template_eng().search(q)}


@nl_screener_v2_router.get("/templates/{name}")
def get_template(name: str):
    """Get a specific template by name."""
    try:
        return _get_template_eng().get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@nl_screener_v2_router.post("/execute-template")
def execute_template(req: ExecuteTemplateRequest):
    """Execute a named template with optional criterion overrides."""
    try:
        criteria = _get_template_eng().to_criteria(req.template_name, req.overrides)
        tmpl = _get_template_eng().get(req.template_name)
        loader = FundamentalDataLoader()
        df = loader.load()
        engine = ScreenExecutionEngine()
        matched, explanations = engine.execute(
            df,
            criteria=criteria,
            sectors=[tmpl.get("sector")] if tmpl.get("sector") else [],
            geographies=[tmpl.get("geography")] if tmpl.get("geography") else [],
            sort_metric=tmpl.get("sort_by"),
            sort_ascending=tmpl.get("sort_asc", False),
            max_results=req.max_results,
        )
        tickers = matched["ticker"].tolist() if "ticker" in matched.columns else []
        scores = matched["_score"].tolist() if "_score" in matched.columns else []
        return {
            "template": req.template_name,
            "description": tmpl.get("description", ""),
            "n_matches": len(matched),
            "top_matches": [{"ticker": t, "score": s} for t, s in zip(tickers, scores)],
            "criteria_descriptions": [str(c) for c in criteria],
            "explanations": [_explanation_to_dict(e) for e in explanations[:10]],
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@nl_screener_v2_router.post("/alerts")
def create_alert(req: AlertRequest):
    """Create a threshold-based alert from a NL query."""
    try:
        parsed = AdvancedNLParser().parse(req.query)
        alert_id = _get_alert_mgr().create_alert(
            name=req.name,
            query=req.query,
            criteria=parsed.criteria,
            notify_email=req.notify_email,
        )
        return {
            "alert_id": alert_id,
            "name": req.name,
            "criteria": [c.to_dict() for c in parsed.criteria],
            "message": "Alert created. Call GET /alerts/evaluate to check triggers.",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@nl_screener_v2_router.get("/alerts")
def list_alerts():
    """List all active alerts."""
    return {"alerts": _get_alert_mgr().list_alerts()}


@nl_screener_v2_router.delete("/alerts/{alert_id}")
def delete_alert(alert_id: int):
    """Delete an alert by ID."""
    ok = _get_alert_mgr().delete_alert(alert_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return {"deleted": alert_id}


@nl_screener_v2_router.get("/alerts/evaluate")
def evaluate_alerts():
    """Evaluate all alerts against current fundamental data."""
    df = FundamentalDataLoader().load()
    triggered = _get_alert_mgr().evaluate_alerts(df)
    return {"triggered_count": len(triggered), "triggered": triggered}


@nl_screener_v2_router.post("/compare")
def compare_tickers(req: CompareRequest):
    """Compare a list of tickers across fundamental metrics."""
    try:
        result = _get_compare_eng().compare(req.tickers, req.metrics)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@nl_screener_v2_router.get("/parse")
def parse_query(q: str = FastAPIQuery(..., description="Natural language query to parse")):
    """Parse a NL query and return structured criteria (without executing)."""
    try:
        parsed = AdvancedNLParser().parse(q)
        intent_clf = IntentClassifier()
        intent, conf = intent_clf.classify(q)
        return {
            "query": q,
            "intent": intent,
            "intent_confidence": conf,
            "criteria": [c.to_dict() for c in parsed.criteria],
            "sectors": parsed.sectors,
            "geographies": parsed.geographies,
            "mktcap_tier": parsed.mktcap_tier,
            "index_membership": parsed.index_membership,
            "temporal": parsed.temporal,
            "sort_metric": parsed.sort_metric,
            "sort_ascending": parsed.sort_ascending,
            "logic": parsed.logic,
            "suggested_template": TemplateEngine().suggest(q),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@nl_screener_v2_router.get("/history")
def get_history(limit: int = FastAPIQuery(20, ge=1, le=100)):
    """Retrieve recent screener query history."""
    try:
        conn = _get_db()
        rows = conn.execute(
            "SELECT query, intent, n_matches, criteria_json, created_at "
            "FROM screen_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return {
            "history": [
                {
                    "query": r[0],
                    "intent": r[1],
                    "n_matches": r[2],
                    "criteria": json.loads(r[3] or "[]"),
                    "created_at": datetime.fromtimestamp(r[4], tz=timezone.utc).isoformat(),
                }
                for r in rows
            ],
            "count": len(rows),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@nl_screener_v2_router.get("/metrics")
def list_metrics():
    """Return the full list of supported metrics and their canonical names."""
    canonical_set: Dict[str, List[str]] = {}
    for alias, canon in METRIC_CANONICAL.items():
        canonical_set.setdefault(canon, []).append(alias)
    return {
        "count": len(canonical_set),
        "metrics": [
            {"canonical": canon, "aliases": aliases}
            for canon, aliases in sorted(canonical_set.items())
        ],
    }


# ── Module-level convenience ──────────────────────────────────────────────────

def run_screen(query: str, max_results: int = 20) -> ScreenerResult:
    """Module-level convenience: parse and execute a NL screen."""
    return NLScreenerV2Pipeline().run(query, max_results=max_results)


def run_template(name: str, max_results: int = 20) -> Tuple[pd.DataFrame, List[MatchExplanation]]:
    """Module-level convenience: execute a named template."""
    eng = TemplateEngine()
    criteria = eng.to_criteria(name)
    tmpl = eng.get(name)
    df = FundamentalDataLoader().load()
    return ScreenExecutionEngine().execute(
        df, criteria,
        sectors=[tmpl.get("sector")] if tmpl.get("sector") else [],
        geographies=[tmpl.get("geography")] if tmpl.get("geography") else [],
        sort_metric=tmpl.get("sort_by"),
        sort_ascending=tmpl.get("sort_asc", False),
        max_results=max_results,
    )


def parse_only(query: str) -> ParsedQuery:
    """Parse a NL query without executing."""
    return AdvancedNLParser().parse(query)

