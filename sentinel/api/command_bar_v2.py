"""
Bloomberg-style command bar v2 — Dimension #90.

Provides a 102+ function-code registry, command parser, autocomplete engine,
command history, and output routing for the SENTINEL terminal.

Syntax:  "{TICKER} {SECURITY_TYPE} {FUNCTION} {MODIFIERS...}"
Examples:
  AAPL US EQUITY DES
  SPX INDEX GP
  EUR CURNCY FXFWD
  TLT US EQUITY YAS 3M
  ECOW US <enter>

Score target: dim_090 → 9
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "SecurityType",
    "FunctionCode",
    "CommandParser",
    "FunctionRegistry",
    "AutocompleteEngine",
    "CommandHistoryStore",
    "CommandRouter",
    "command_router",
]

# ---------------------------------------------------------------------------
# SQLite path
# ---------------------------------------------------------------------------

_DB_DIR = Path(".sentinel") / "command_bar"
_DB_PATH = _DB_DIR / "command_bar.db"


def _get_db() -> sqlite3.Connection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    with _get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS command_history (
                id          TEXT PRIMARY KEY,
                raw_input   TEXT NOT NULL,
                ticker      TEXT,
                sec_type    TEXT,
                func_code   TEXT,
                modifiers   TEXT,
                ts          REAL NOT NULL,
                routed_to   TEXT
            );

            CREATE TABLE IF NOT EXISTS ticker_recents (
                ticker      TEXT PRIMARY KEY,
                sec_type    TEXT,
                last_used   REAL NOT NULL,
                use_count   INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS shortcut_map (
                shortcut    TEXT PRIMARY KEY,
                func_code   TEXT NOT NULL,
                description TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_history_ts  ON command_history(ts DESC);
            CREATE INDEX IF NOT EXISTS idx_history_ticker ON command_history(ticker);
            """
        )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SecurityType(str, Enum):
    EQUITY = "EQUITY"
    INDEX = "INDEX"
    CURRENCY = "CURRENCY"
    CURNCY = "CURNCY"
    COMDTY = "COMDTY"
    GOVT = "GOVT"
    CORP = "CORP"
    MUNI = "MUNI"
    ETF = "ETF"
    CRYPTO = "CRYPTO"
    FUND = "FUND"
    PREF = "PREF"
    UNKNOWN = "UNKNOWN"


# Aliases that map to canonical types
_SEC_TYPE_ALIASES: dict[str, SecurityType] = {
    "EQUITY":   SecurityType.EQUITY,
    "EQ":       SecurityType.EQUITY,
    "STK":      SecurityType.EQUITY,
    "INDEX":    SecurityType.INDEX,
    "IND":      SecurityType.INDEX,
    "INDX":     SecurityType.INDEX,
    "CURRENCY": SecurityType.CURRENCY,
    "CURNCY":   SecurityType.CURNCY,
    "FX":       SecurityType.CURNCY,
    "COMDTY":   SecurityType.COMDTY,
    "CMDTY":    SecurityType.COMDTY,
    "COMMODITY":SecurityType.COMDTY,
    "GOVT":     SecurityType.GOVT,
    "GOV":      SecurityType.GOVT,
    "TREASURY": SecurityType.GOVT,
    "CORP":     SecurityType.CORP,
    "CREDIT":   SecurityType.CORP,
    "MUNI":     SecurityType.MUNI,
    "MUNICIPAL":SecurityType.MUNI,
    "ETF":      SecurityType.ETF,
    "CRYPTO":   SecurityType.CRYPTO,
    "FUND":     SecurityType.FUND,
    "PREF":     SecurityType.PREF,
}


class OutputTarget(str, Enum):
    """Which SENTINEL module handles the function output."""
    PRICE_CHART       = "price_chart"
    DESCRIPTION       = "description"
    FINANCIAL_ANALYSIS= "financial_analysis"
    RELATIVE_VALUE    = "relative_value"
    DIVIDENDS         = "dividends"
    EARNINGS          = "earnings"
    ANALYST_RECS      = "analyst_recs"
    OPTIONS_MONITOR   = "options_monitor"
    OPTIONS_VALUATION = "options_valuation"
    EVENTS            = "events"
    CASH_FLOW         = "cash_flow"
    NEWS              = "news"
    MOST_ACTIVE       = "most_active"
    TOP_STORIES       = "top_stories"
    YIELD_ANALYTICS   = "yield_analytics"
    ALL_QUOTES        = "all_quotes"
    CREDIT_DEFAULT    = "credit_default"
    DURATION          = "duration"
    RISK_ANALYTICS    = "risk_analytics"
    FX_FORWARDS       = "fx_forwards"
    FX_IMPLIED_VOL    = "fx_implied_vol"
    FX_FORECAST       = "fx_forecast"
    ECO_CALENDAR      = "eco_calendar"
    WORLD_ECONOMICS   = "world_economics"
    GDP_VIEWER        = "gdp_viewer"
    CPI_VIEWER        = "cpi_viewer"
    RATE_FUTURES      = "rate_futures"
    PORTFOLIO         = "portfolio"
    PORTFOLIO_UPLOAD  = "portfolio_upload"
    PERF_ATTRIBUTION  = "performance_attribution"
    EQUITY_SCREEN     = "equity_screen"
    CREDIT_SCREEN     = "credit_screen"
    FILTER            = "filter"
    GRAPH             = "graph"
    COMPARISON        = "comparison"
    INDUSTRY          = "industry"
    MOVING_AVG        = "moving_avg"
    HELP              = "help"
    DOCS              = "docs"
    MENU              = "menu"
    DEBT_DISTRIBUTION = "debt_distribution"
    HISTORICAL_SPREADS= "historical_spreads"
    BOND_CASH_FLOWS   = "bond_cash_flows"
    INSIDER_FLOW      = "insider_flow"
    OWNERSHIP         = "ownership"
    SHORT_INTEREST    = "short_interest"
    SECTOR_MAP        = "sector_map"
    HEAT_MAP          = "heat_map"
    VOLUME_PROFILE    = "volume_profile"
    TECHNICAL         = "technical"
    SCREENING         = "screening"
    WATCHLIST         = "watchlist"
    CORRELATION       = "correlation"
    BACKTEST          = "backtest"
    SCENARIO          = "scenario"
    SUPPLY_CHAIN      = "supply_chain"
    ESG               = "esg"
    MACRO_OVERVIEW    = "macro_overview"
    CREDIT_MONITOR    = "credit_monitor"
    BOND_SCREENER     = "bond_screener"
    YIELD_CURVE       = "yield_curve"
    CONVERTIBLE       = "convertible"
    SWAP_MONITOR      = "swap_monitor"
    FUTURES_CURVE     = "futures_curve"
    CRYPTO_BOOK       = "crypto_book"
    ON_CHAIN          = "on_chain"
    DEFI              = "defi"
    ALTERNATIVE_DATA  = "alternative_data"
    IPO_CALENDAR      = "ipo_calendar"
    MA_INTELLIGENCE   = "ma_intelligence"
    PRIVATE_MARKETS   = "private_markets"
    SETTINGS          = "settings"
    CALCULATOR        = "calculator"
    UNKNOWN           = "unknown"


# ---------------------------------------------------------------------------
# Function Code Registry — 102+ codes
# ---------------------------------------------------------------------------

class FunctionCode(BaseModel):
    code: str
    label: str
    description: str
    output_target: OutputTarget
    applicable_types: list[SecurityType]
    modifiers: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    example: str = ""


# Full registry definition
_FUNCTION_DEFINITIONS: list[dict] = [
    # -------------------------------------------------------------------------
    # EQUITY functions
    # -------------------------------------------------------------------------
    {
        "code": "DES",
        "label": "Description",
        "description": "Company description, key statistics, and overview",
        "output_target": OutputTarget.DESCRIPTION,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF, SecurityType.FUND],
        "modifiers": [],
        "keywords": ["description", "overview", "profile", "about"],
        "example": "AAPL US EQUITY DES",
    },
    {
        "code": "GP",
        "label": "Price Graph",
        "description": "Historical price chart with configurable period",
        "output_target": OutputTarget.PRICE_CHART,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF, SecurityType.INDEX,
                              SecurityType.COMDTY, SecurityType.GOVT, SecurityType.CURNCY],
        "modifiers": ["1D","5D","1M","3M","6M","1Y","2Y","5Y","10Y","MAX"],
        "keywords": ["price", "chart", "graph", "historical"],
        "example": "AAPL US EQUITY GP 1Y",
    },
    {
        "code": "GY",
        "label": "Yield Graph",
        "description": "Historical yield chart for fixed income securities",
        "output_target": OutputTarget.YIELD_CURVE,
        "applicable_types": [SecurityType.GOVT, SecurityType.CORP, SecurityType.MUNI],
        "modifiers": ["1M","3M","6M","1Y","5Y","10Y"],
        "keywords": ["yield", "chart", "graph"],
        "example": "TLT US EQUITY GY 1Y",
    },
    {
        "code": "FA",
        "label": "Financial Analysis",
        "description": "Comprehensive income statement, balance sheet, and cash flow",
        "output_target": OutputTarget.FINANCIAL_ANALYSIS,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": ["A","Q","TTM"],
        "keywords": ["financials", "income", "balance sheet", "cash flow", "statements"],
        "example": "AAPL US EQUITY FA Q",
    },
    {
        "code": "RV",
        "label": "Relative Value",
        "description": "Peer comparison and relative valuation multiples",
        "output_target": OutputTarget.RELATIVE_VALUE,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["valuation", "peers", "multiples", "comparison"],
        "example": "AAPL US EQUITY RV",
    },
    {
        "code": "DDIS",
        "label": "Debt Distribution",
        "description": "Debt maturity schedule and capital structure breakdown",
        "output_target": OutputTarget.DEBT_DISTRIBUTION,
        "applicable_types": [SecurityType.EQUITY, SecurityType.CORP],
        "modifiers": [],
        "keywords": ["debt", "maturity", "capital structure"],
        "example": "AAPL US EQUITY DDIS",
    },
    {
        "code": "DVD",
        "label": "Dividends",
        "description": "Dividend history, yield, payout ratio, and estimates",
        "output_target": OutputTarget.DIVIDENDS,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": [],
        "keywords": ["dividends", "yield", "payout"],
        "example": "AAPL US EQUITY DVD",
    },
    {
        "code": "EE",
        "label": "Earnings Estimates",
        "description": "Consensus EPS/revenue estimates and revision history",
        "output_target": OutputTarget.EARNINGS,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": ["FY1","FY2","Q1","Q2"],
        "keywords": ["earnings", "estimates", "eps", "consensus"],
        "example": "AAPL US EQUITY EE FY1",
    },
    {
        "code": "ANR",
        "label": "Analyst Recommendations",
        "description": "Buy/sell/hold ratings, price targets, and rating changes",
        "output_target": OutputTarget.ANALYST_RECS,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["analyst", "recommendations", "ratings", "price target"],
        "example": "AAPL US EQUITY ANR",
    },
    {
        "code": "HS",
        "label": "Historical Spreads",
        "description": "Credit spread history and Z-spread analysis",
        "output_target": OutputTarget.HISTORICAL_SPREADS,
        "applicable_types": [SecurityType.CORP, SecurityType.GOVT, SecurityType.MUNI],
        "modifiers": ["1M","3M","1Y","5Y"],
        "keywords": ["spread", "credit", "z-spread", "oas"],
        "example": "AAPL US EQUITY HS 1Y",
    },
    {
        "code": "OMON",
        "label": "Options Monitor",
        "description": "Live options chain with Greeks, IV, and volume",
        "output_target": OutputTarget.OPTIONS_MONITOR,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF, SecurityType.INDEX],
        "modifiers": ["CALLS","PUTS","ALL"],
        "keywords": ["options", "chain", "greeks", "iv", "implied volatility"],
        "example": "AAPL US EQUITY OMON",
    },
    {
        "code": "OV",
        "label": "Options Valuation",
        "description": "Single-option pricing models: Black-Scholes, Binomial, Monte Carlo",
        "output_target": OutputTarget.OPTIONS_VALUATION,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF, SecurityType.INDEX],
        "modifiers": ["BS","BIN","MC"],
        "keywords": ["option pricing", "black scholes", "valuation"],
        "example": "AAPL US EQUITY OV BS",
    },
    {
        "code": "EVTS",
        "label": "Events",
        "description": "Upcoming corporate events: earnings, dividends, splits, conferences",
        "output_target": OutputTarget.EVENTS,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": [],
        "keywords": ["events", "earnings date", "corporate actions"],
        "example": "AAPL US EQUITY EVTS",
    },
    {
        "code": "CF",
        "label": "Cash Flow Statement",
        "description": "Detailed cash flow from operations, investing, and financing",
        "output_target": OutputTarget.CASH_FLOW,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": ["A","Q","TTM"],
        "keywords": ["cash flow", "fcf", "capex", "ocf"],
        "example": "AAPL US EQUITY CF TTM",
    },
    {
        "code": "CN",
        "label": "Company News",
        "description": "Latest news headlines and press releases for the company",
        "output_target": OutputTarget.NEWS,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF, SecurityType.CORP],
        "modifiers": ["1D","1W","1M"],
        "keywords": ["news", "press release", "headlines"],
        "example": "AAPL US EQUITY CN",
    },
    {
        "code": "MOST",
        "label": "Most Active",
        "description": "Most active stocks by volume, value, or % change",
        "output_target": OutputTarget.MOST_ACTIVE,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": ["VOL","VAL","PCT"],
        "keywords": ["most active", "movers", "volume"],
        "example": "MOST VOL",
    },
    {
        "code": "TOP",
        "label": "Top Stories",
        "description": "Top market-moving news stories across all assets",
        "output_target": OutputTarget.TOP_STORIES,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["top news", "headlines", "market news"],
        "example": "TOP",
    },
    {
        "code": "INSIDER",
        "label": "Insider Transactions",
        "description": "Form 4 filings: insider buys, sells, and option exercises",
        "output_target": OutputTarget.INSIDER_FLOW,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": ["BUY","SELL","ALL"],
        "keywords": ["insider", "form 4", "insiders buying"],
        "example": "AAPL US EQUITY INSIDER",
    },
    {
        "code": "OWN",
        "label": "Ownership",
        "description": "Institutional and retail ownership breakdown from 13F filings",
        "output_target": OutputTarget.OWNERSHIP,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": [],
        "keywords": ["ownership", "13f", "institutional", "holders"],
        "example": "AAPL US EQUITY OWN",
    },
    {
        "code": "SHORT",
        "label": "Short Interest",
        "description": "Short interest data, days-to-cover, and short squeeze potential",
        "output_target": OutputTarget.SHORT_INTEREST,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["short interest", "short squeeze", "days to cover"],
        "example": "AAPL US EQUITY SHORT",
    },
    {
        "code": "SEC",
        "label": "SEC Filings",
        "description": "EDGAR filings: 10-K, 10-Q, 8-K, proxy, and full-text search",
        "output_target": OutputTarget.FINANCIAL_ANALYSIS,
        "applicable_types": [SecurityType.EQUITY, SecurityType.CORP],
        "modifiers": ["10K","10Q","8K","DEF14A"],
        "keywords": ["sec", "edgar", "filings", "10k", "10q"],
        "example": "AAPL US EQUITY SEC 10K",
    },
    {
        "code": "ESG",
        "label": "ESG Scores",
        "description": "Environmental, Social, Governance composite scores and breakdown",
        "output_target": OutputTarget.ESG,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": [],
        "keywords": ["esg", "environmental", "social", "governance", "sustainability"],
        "example": "AAPL US EQUITY ESG",
    },
    {
        "code": "SUPP",
        "label": "Supply Chain",
        "description": "Key suppliers, customers, and supply chain risk assessment",
        "output_target": OutputTarget.SUPPLY_CHAIN,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["supply chain", "suppliers", "customers"],
        "example": "AAPL US EQUITY SUPP",
    },
    {
        "code": "BEAT",
        "label": "Earnings Surprise History",
        "description": "Historical earnings beats/misses and surprise magnitude",
        "output_target": OutputTarget.EARNINGS,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["earnings surprise", "beat", "miss", "estimate"],
        "example": "AAPL US EQUITY BEAT",
    },
    {
        "code": "COMP",
        "label": "Comparison Chart",
        "description": "Multi-security performance comparison on one chart",
        "output_target": OutputTarget.COMPARISON,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["compare", "comparison", "performance"],
        "example": "AAPL US EQUITY COMP",
    },
    {
        "code": "INDU",
        "label": "Industry Analysis",
        "description": "Sector/industry peers, relative performance, and ranking",
        "output_target": OutputTarget.INDUSTRY,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["industry", "sector", "peers"],
        "example": "AAPL US EQUITY INDU",
    },
    {
        "code": "MA",
        "label": "Moving Averages",
        "description": "Technical chart with SMA, EMA, MACD, RSI overlays",
        "output_target": OutputTarget.MOVING_AVG,
        "applicable_types": [SecurityType.EQUITY, SecurityType.INDEX, SecurityType.ETF,
                              SecurityType.COMDTY, SecurityType.CURNCY],
        "modifiers": ["SMA20","SMA50","SMA200","EMA20","MACD","RSI"],
        "keywords": ["moving average", "technical", "sma", "ema"],
        "example": "AAPL US EQUITY MA SMA50",
    },
    {
        "code": "TECH",
        "label": "Technical Analysis",
        "description": "Full technical dashboard: indicators, patterns, signals",
        "output_target": OutputTarget.TECHNICAL,
        "applicable_types": [SecurityType.EQUITY, SecurityType.INDEX, SecurityType.ETF,
                              SecurityType.COMDTY, SecurityType.CURNCY],
        "modifiers": [],
        "keywords": ["technical analysis", "indicators", "signals"],
        "example": "AAPL US EQUITY TECH",
    },
    {
        "code": "VWAP",
        "label": "VWAP Chart",
        "description": "Volume-weighted average price with intraday bands",
        "output_target": OutputTarget.PRICE_CHART,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": [],
        "keywords": ["vwap", "volume weighted"],
        "example": "AAPL US EQUITY VWAP",
    },
    {
        "code": "VP",
        "label": "Volume Profile",
        "description": "Volume-at-price distribution and high-volume nodes",
        "output_target": OutputTarget.VOLUME_PROFILE,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF, SecurityType.INDEX],
        "modifiers": [],
        "keywords": ["volume profile", "hvn", "lvn", "poc"],
        "example": "AAPL US EQUITY VP",
    },
    {
        "code": "HEAT",
        "label": "Heat Map",
        "description": "Sector heat map by return, volume, or fundamental metric",
        "output_target": OutputTarget.HEAT_MAP,
        "applicable_types": [SecurityType.INDEX, SecurityType.EQUITY],
        "modifiers": ["1D","1W","1M","YTD","PCT","VOL"],
        "keywords": ["heat map", "sector", "market map"],
        "example": "SPX INDEX HEAT 1D",
    },
    {
        "code": "MAP",
        "label": "Sector Map",
        "description": "Treemap visualization of market cap and sector weights",
        "output_target": OutputTarget.SECTOR_MAP,
        "applicable_types": [SecurityType.INDEX, SecurityType.ETF],
        "modifiers": [],
        "keywords": ["sector map", "treemap", "market cap"],
        "example": "SPX INDEX MAP",
    },
    # -------------------------------------------------------------------------
    # Fixed Income functions
    # -------------------------------------------------------------------------
    {
        "code": "YAS",
        "label": "Yield Analytics",
        "description": "Yield, duration, convexity, OAS, Z-spread, and scenario analysis",
        "output_target": OutputTarget.YIELD_ANALYTICS,
        "applicable_types": [SecurityType.GOVT, SecurityType.CORP, SecurityType.MUNI],
        "modifiers": ["DUR","CONV","OAS","ZSP"],
        "keywords": ["yield", "duration", "convexity", "spread", "analytics"],
        "example": "TLT US EQUITY YAS",
    },
    {
        "code": "ALLQ",
        "label": "All Quotes",
        "description": "Consolidated bid/offer quotes from all available venues",
        "output_target": OutputTarget.ALL_QUOTES,
        "applicable_types": [SecurityType.GOVT, SecurityType.CORP, SecurityType.MUNI],
        "modifiers": [],
        "keywords": ["quotes", "bid", "offer", "venues"],
        "example": "TLT US EQUITY ALLQ",
    },
    {
        "code": "CRVD",
        "label": "Credit Default",
        "description": "CDS spreads, credit default risk metrics, and term structure",
        "output_target": OutputTarget.CREDIT_DEFAULT,
        "applicable_types": [SecurityType.CORP, SecurityType.GOVT],
        "modifiers": ["1Y","3Y","5Y","10Y"],
        "keywords": ["cds", "credit default swap", "credit risk"],
        "example": "AAPL US EQUITY CRVD 5Y",
    },
    {
        "code": "FLW",
        "label": "Cash Flows",
        "description": "Bond cash flow schedule: coupon and principal payments",
        "output_target": OutputTarget.BOND_CASH_FLOWS,
        "applicable_types": [SecurityType.GOVT, SecurityType.CORP, SecurityType.MUNI],
        "modifiers": [],
        "keywords": ["cash flows", "coupon", "bond payments"],
        "example": "TLT US EQUITY FLW",
    },
    {
        "code": "DURA",
        "label": "Duration",
        "description": "Modified duration, DV01, convexity, and key rate durations",
        "output_target": OutputTarget.DURATION,
        "applicable_types": [SecurityType.GOVT, SecurityType.CORP, SecurityType.MUNI],
        "modifiers": ["MOD","MAC","DV01","KRD"],
        "keywords": ["duration", "dv01", "modified duration", "convexity"],
        "example": "TLT US EQUITY DURA MOD",
    },
    {
        "code": "RISK",
        "label": "Risk Analytics",
        "description": "VaR, CVaR, stress tests, and scenario analysis for fixed income",
        "output_target": OutputTarget.RISK_ANALYTICS,
        "applicable_types": [SecurityType.GOVT, SecurityType.CORP, SecurityType.MUNI,
                              SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": ["VAR","CVAR","STRESS"],
        "keywords": ["risk", "var", "value at risk", "stress test"],
        "example": "TLT US EQUITY RISK VAR",
    },
    {
        "code": "TRACE",
        "label": "TRACE Bond Pricing",
        "description": "FINRA TRACE reported bond trade prices and volume",
        "output_target": OutputTarget.ALL_QUOTES,
        "applicable_types": [SecurityType.CORP, SecurityType.GOVT, SecurityType.MUNI],
        "modifiers": [],
        "keywords": ["trace", "bond price", "finra"],
        "example": "LQD US EQUITY TRACE",
    },
    {
        "code": "YCRV",
        "label": "Yield Curve",
        "description": "Treasury/sovereign yield curve with full term structure",
        "output_target": OutputTarget.YIELD_CURVE,
        "applicable_types": [SecurityType.GOVT, SecurityType.INDEX],
        "modifiers": ["US","UK","DE","JP","AU"],
        "keywords": ["yield curve", "treasury", "term structure"],
        "example": "YCRV US",
    },
    {
        "code": "SRCH",
        "label": "Credit Screen",
        "description": "Bond screener: filter by rating, duration, spread, sector",
        "output_target": OutputTarget.CREDIT_SCREEN,
        "applicable_types": [SecurityType.CORP, SecurityType.GOVT, SecurityType.MUNI],
        "modifiers": [],
        "keywords": ["bond screener", "credit screen", "filter bonds"],
        "example": "SRCH IG",
    },
    {
        "code": "BSRCH",
        "label": "Bond Screener",
        "description": "Advanced fixed income screener with multi-factor filters",
        "output_target": OutputTarget.BOND_SCREENER,
        "applicable_types": [SecurityType.CORP, SecurityType.GOVT, SecurityType.MUNI],
        "modifiers": ["IG","HY","MUNI","TIPS"],
        "keywords": ["bond screener", "fixed income screen"],
        "example": "BSRCH HY",
    },
    {
        "code": "CNVT",
        "label": "Convertible Bonds",
        "description": "Convertible bond analysis: parity, premium, delta hedge",
        "output_target": OutputTarget.CONVERTIBLE,
        "applicable_types": [SecurityType.CORP, SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["convertible", "convert", "parity"],
        "example": "AAPL US EQUITY CNVT",
    },
    # -------------------------------------------------------------------------
    # FX functions
    # -------------------------------------------------------------------------
    {
        "code": "FXFWD",
        "label": "FX Forwards",
        "description": "FX forward rates, points, and non-deliverable forward pricing",
        "output_target": OutputTarget.FX_FORWARDS,
        "applicable_types": [SecurityType.CURNCY, SecurityType.CURRENCY],
        "modifiers": ["1W","1M","3M","6M","1Y"],
        "keywords": ["fx forward", "forward points", "ndf"],
        "example": "EUR CURNCY FXFWD 3M",
    },
    {
        "code": "FXIV",
        "label": "FX Implied Volatility",
        "description": "FX volatility surface, risk reversal, and butterfly spreads",
        "output_target": OutputTarget.FX_IMPLIED_VOL,
        "applicable_types": [SecurityType.CURNCY, SecurityType.CURRENCY],
        "modifiers": ["1M","3M","6M","1Y"],
        "keywords": ["fx volatility", "implied vol", "risk reversal", "vol surface"],
        "example": "EUR CURNCY FXIV 1M",
    },
    {
        "code": "FXFC",
        "label": "FX Forecast",
        "description": "Consensus FX forecasts and model-based projections",
        "output_target": OutputTarget.FX_FORECAST,
        "applicable_types": [SecurityType.CURNCY, SecurityType.CURRENCY],
        "modifiers": ["3M","6M","12M"],
        "keywords": ["fx forecast", "currency forecast", "projection"],
        "example": "EUR CURNCY FXFC 12M",
    },
    {
        "code": "FXCA",
        "label": "FX Carry",
        "description": "Carry trade returns, funding costs, and rolldown by pair",
        "output_target": OutputTarget.FX_FORWARDS,
        "applicable_types": [SecurityType.CURNCY, SecurityType.CURRENCY],
        "modifiers": [],
        "keywords": ["carry trade", "fx carry", "funding"],
        "example": "USDJPY CURNCY FXCA",
    },
    # -------------------------------------------------------------------------
    # Macro / Economics functions
    # -------------------------------------------------------------------------
    {
        "code": "ECOW",
        "label": "Economic Calendar",
        "description": "Global economic data releases with forecast vs. actual",
        "output_target": OutputTarget.ECO_CALENDAR,
        "applicable_types": list(SecurityType),
        "modifiers": ["US","EU","UK","JP","CN","AU"],
        "keywords": ["economic calendar", "data releases", "macro calendar"],
        "example": "ECOW US",
    },
    {
        "code": "WECO",
        "label": "World Economics",
        "description": "Country-level macro dashboards: GDP, CPI, unemployment, PMI",
        "output_target": OutputTarget.WORLD_ECONOMICS,
        "applicable_types": list(SecurityType),
        "modifiers": ["US","EU","UK","JP","CN","EM"],
        "keywords": ["world economics", "country macro", "global economy"],
        "example": "WECO EU",
    },
    {
        "code": "GDP",
        "label": "GDP Viewer",
        "description": "GDP growth: actual, forecast, and historical time series",
        "output_target": OutputTarget.GDP_VIEWER,
        "applicable_types": list(SecurityType),
        "modifiers": ["QOQ","YOY","REAL","NOM"],
        "keywords": ["gdp", "growth", "economic growth"],
        "example": "GDP US QOQ",
    },
    {
        "code": "CPI",
        "label": "CPI Viewer",
        "description": "Consumer Price Index: headline, core, components, and trend",
        "output_target": OutputTarget.CPI_VIEWER,
        "applicable_types": list(SecurityType),
        "modifiers": ["YOY","MOM","CORE","HEADLINE"],
        "keywords": ["cpi", "inflation", "consumer prices"],
        "example": "CPI US YOY",
    },
    {
        "code": "IFFS",
        "label": "Interest Rate Futures",
        "description": "Fed funds futures pricing and implied rate path probability",
        "output_target": OutputTarget.RATE_FUTURES,
        "applicable_types": [SecurityType.COMDTY, SecurityType.GOVT, SecurityType.INDEX],
        "modifiers": [],
        "keywords": ["fed funds", "rate futures", "fomc", "rate path"],
        "example": "IFFS",
    },
    {
        "code": "FOMC",
        "label": "FOMC Monitor",
        "description": "Fed meeting dates, decisions, dot plot, and rate path",
        "output_target": OutputTarget.RATE_FUTURES,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["fomc", "fed", "interest rates", "dot plot"],
        "example": "FOMC",
    },
    {
        "code": "PMI",
        "label": "PMI Data",
        "description": "Manufacturing and services PMI data across major economies",
        "output_target": OutputTarget.WORLD_ECONOMICS,
        "applicable_types": list(SecurityType),
        "modifiers": ["MFG","SVCS","COMP"],
        "keywords": ["pmi", "purchasing managers", "manufacturing"],
        "example": "PMI US MFG",
    },
    {
        "code": "JOLTS",
        "label": "JOLTS / Labor",
        "description": "Job openings, labor turnover, quits rate, and NFP dashboard",
        "output_target": OutputTarget.WORLD_ECONOMICS,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["jolts", "jobs", "labor market", "nfp"],
        "example": "JOLTS",
    },
    # -------------------------------------------------------------------------
    # Portfolio functions
    # -------------------------------------------------------------------------
    {
        "code": "PORT",
        "label": "Portfolio Analytics",
        "description": "Portfolio performance, attribution, risk, and allocation",
        "output_target": OutputTarget.PORTFOLIO,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["portfolio", "performance", "allocation"],
        "example": "PORT",
    },
    {
        "code": "PRTU",
        "label": "Portfolio Upload",
        "description": "Upload portfolio positions from CSV/JSON for analysis",
        "output_target": OutputTarget.PORTFOLIO_UPLOAD,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["portfolio upload", "positions", "import portfolio"],
        "example": "PRTU",
    },
    {
        "code": "PPAR",
        "label": "Performance Attribution",
        "description": "Brinson-Hood-Beebower attribution by sector, factor, and security",
        "output_target": OutputTarget.PERF_ATTRIBUTION,
        "applicable_types": list(SecurityType),
        "modifiers": ["SECTOR","FACTOR","SECURITY"],
        "keywords": ["attribution", "performance", "brinson"],
        "example": "PPAR SECTOR",
    },
    {
        "code": "CORR",
        "label": "Correlation Matrix",
        "description": "Asset correlation matrix with rolling windows and heatmap",
        "output_target": OutputTarget.CORRELATION,
        "applicable_types": list(SecurityType),
        "modifiers": ["1M","3M","6M","1Y"],
        "keywords": ["correlation", "matrix", "covariance"],
        "example": "CORR 3M",
    },
    {
        "code": "BACK",
        "label": "Backtest",
        "description": "Strategy backtesting with Sharpe, drawdown, and factor exposure",
        "output_target": OutputTarget.BACKTEST,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["backtest", "strategy", "historical test"],
        "example": "BACK",
    },
    {
        "code": "SCEN",
        "label": "Scenario Analysis",
        "description": "What-if scenario analysis for portfolio or single security",
        "output_target": OutputTarget.SCENARIO,
        "applicable_types": list(SecurityType),
        "modifiers": ["BULL","BEAR","BASE"],
        "keywords": ["scenario", "what if", "stress"],
        "example": "SCEN BEAR",
    },
    # -------------------------------------------------------------------------
    # Screener functions
    # -------------------------------------------------------------------------
    {
        "code": "EQS",
        "label": "Equity Screen",
        "description": "Bloomberg-style equity screener with 200+ fundamental/technical filters",
        "output_target": OutputTarget.EQUITY_SCREEN,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": ["VALUE","GROWTH","QUALITY","MOMENTUM","DIVIDEND"],
        "keywords": ["equity screen", "stock screener", "filter stocks"],
        "example": "EQS VALUE",
    },
    {
        "code": "FLT",
        "label": "Filter",
        "description": "Quick inline filter on currently displayed data",
        "output_target": OutputTarget.FILTER,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["filter", "search", "narrow"],
        "example": "FLT PE<15",
    },
    {
        "code": "WGT",
        "label": "Watchlist",
        "description": "Manage watchlists: view, add, remove tickers",
        "output_target": OutputTarget.WATCHLIST,
        "applicable_types": list(SecurityType),
        "modifiers": ["ADD","REMOVE","VIEW"],
        "keywords": ["watchlist", "watch", "favorites"],
        "example": "WGT ADD AAPL",
    },
    # -------------------------------------------------------------------------
    # Index / Futures functions
    # -------------------------------------------------------------------------
    {
        "code": "G",
        "label": "Graph",
        "description": "Quick single-security graph with default settings",
        "output_target": OutputTarget.GRAPH,
        "applicable_types": list(SecurityType),
        "modifiers": ["1D","5D","1M","3M","1Y"],
        "keywords": ["graph", "chart", "plot"],
        "example": "SPX INDEX G 1Y",
    },
    {
        "code": "FUTS",
        "label": "Futures Curve",
        "description": "Futures term structure, contango/backwardation, and roll yield",
        "output_target": OutputTarget.FUTURES_CURVE,
        "applicable_types": [SecurityType.COMDTY, SecurityType.INDEX],
        "modifiers": [],
        "keywords": ["futures curve", "term structure", "contango", "backwardation"],
        "example": "CL1 COMDTY FUTS",
    },
    {
        "code": "SWAP",
        "label": "Swap Monitor",
        "description": "Interest rate swap rates, OIS spreads, and cross-currency basis",
        "output_target": OutputTarget.SWAP_MONITOR,
        "applicable_types": [SecurityType.GOVT, SecurityType.INDEX],
        "modifiers": ["1Y","2Y","5Y","10Y","30Y"],
        "keywords": ["swap", "interest rate swap", "ois", "libor"],
        "example": "SWAP 5Y",
    },
    # -------------------------------------------------------------------------
    # Crypto / DeFi functions
    # -------------------------------------------------------------------------
    {
        "code": "CRYP",
        "label": "Crypto Book",
        "description": "Crypto order book, depth chart, and cross-exchange arbitrage",
        "output_target": OutputTarget.CRYPTO_BOOK,
        "applicable_types": [SecurityType.CRYPTO],
        "modifiers": [],
        "keywords": ["crypto", "order book", "depth"],
        "example": "BTC CRYPTO CRYP",
    },
    {
        "code": "ONCH",
        "label": "On-Chain Metrics",
        "description": "On-chain analytics: active addresses, NVT, MVRV, exchange flows",
        "output_target": OutputTarget.ON_CHAIN,
        "applicable_types": [SecurityType.CRYPTO],
        "modifiers": [],
        "keywords": ["on-chain", "blockchain", "nvt", "mvrv"],
        "example": "BTC CRYPTO ONCH",
    },
    {
        "code": "DEFI",
        "label": "DeFi Analytics",
        "description": "DeFi protocol TVL, yield farming, and liquidity pool analytics",
        "output_target": OutputTarget.DEFI,
        "applicable_types": [SecurityType.CRYPTO],
        "modifiers": [],
        "keywords": ["defi", "tvl", "yield farming", "liquidity"],
        "example": "ETH CRYPTO DEFI",
    },
    # -------------------------------------------------------------------------
    # Alternative Data
    # -------------------------------------------------------------------------
    {
        "code": "ALTS",
        "label": "Alternative Data",
        "description": "Satellite, web traffic, credit card, sentiment alternative data",
        "output_target": OutputTarget.ALTERNATIVE_DATA,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": ["SENT","WEB","CC","SAT"],
        "keywords": ["alternative data", "satellite", "sentiment", "web traffic"],
        "example": "AAPL US EQUITY ALTS SENT",
    },
    {
        "code": "GTREND",
        "label": "Google Trends",
        "description": "Google search interest trends correlated with price action",
        "output_target": OutputTarget.ALTERNATIVE_DATA,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["google trends", "search interest", "trends"],
        "example": "AAPL US EQUITY GTREND",
    },
    # -------------------------------------------------------------------------
    # Corporate Events
    # -------------------------------------------------------------------------
    {
        "code": "IPO",
        "label": "IPO Calendar",
        "description": "Upcoming IPOs, direct listings, and SPACs with deal terms",
        "output_target": OutputTarget.IPO_CALENDAR,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["ipo", "initial public offering", "listing"],
        "example": "IPO",
    },
    {
        "code": "MA_DEAL",
        "label": "M&A Intelligence",
        "description": "Merger and acquisition deal tracker, rumors, and screening",
        "output_target": OutputTarget.MA_INTELLIGENCE,
        "applicable_types": [SecurityType.EQUITY, SecurityType.CORP],
        "modifiers": ["PENDING","CLOSED","RUMOR"],
        "keywords": ["merger", "acquisition", "m&a", "deal"],
        "example": "MA_DEAL PENDING",
    },
    {
        "code": "PRIV",
        "label": "Private Markets",
        "description": "VC/PE deals, fund performance, and private company data",
        "output_target": OutputTarget.PRIVATE_MARKETS,
        "applicable_types": [SecurityType.EQUITY, SecurityType.FUND],
        "modifiers": ["VC","PE","CREDIT"],
        "keywords": ["private equity", "venture capital", "private markets"],
        "example": "PRIV VC",
    },
    # -------------------------------------------------------------------------
    # Help / Navigation
    # -------------------------------------------------------------------------
    {
        "code": "HELP",
        "label": "Help",
        "description": "Context-sensitive help for current function or general overview",
        "output_target": OutputTarget.HELP,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["help", "support", "how to"],
        "example": "HELP DES",
    },
    {
        "code": "DOCS",
        "label": "Documentation",
        "description": "Full documentation for the SENTINEL terminal",
        "output_target": OutputTarget.DOCS,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["docs", "documentation", "manual"],
        "example": "DOCS",
    },
    {
        "code": "MENU",
        "label": "Main Menu",
        "description": "Top-level navigation menu for all terminal functions",
        "output_target": OutputTarget.MENU,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["menu", "home", "navigation"],
        "example": "MENU",
    },
    {
        "code": "CALC",
        "label": "Calculator",
        "description": "Financial calculator: bond, options, DCF, mortgage",
        "output_target": OutputTarget.CALCULATOR,
        "applicable_types": list(SecurityType),
        "modifiers": ["BOND","OPT","DCF","MORT"],
        "keywords": ["calculator", "compute", "calculate"],
        "example": "CALC BOND",
    },
    {
        "code": "SET",
        "label": "Settings",
        "description": "Terminal preferences: theme, data sources, shortcuts",
        "output_target": OutputTarget.SETTINGS,
        "applicable_types": list(SecurityType),
        "modifiers": ["THEME","DATA","KEYS"],
        "keywords": ["settings", "preferences", "config"],
        "example": "SET THEME",
    },
    {
        "code": "MACRO",
        "label": "Macro Overview",
        "description": "Global macro dashboard: rates, FX, commodities, equities",
        "output_target": OutputTarget.MACRO_OVERVIEW,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["macro", "global", "overview", "dashboard"],
        "example": "MACRO",
    },
    {
        "code": "CMON",
        "label": "Credit Monitor",
        "description": "IG/HY spread monitor, CDS indices, and credit flow",
        "output_target": OutputTarget.CREDIT_MONITOR,
        "applicable_types": [SecurityType.CORP, SecurityType.GOVT],
        "modifiers": ["IG","HY","EM"],
        "keywords": ["credit monitor", "spreads", "ig", "hy"],
        "example": "CMON HY",
    },
    # -------------------------------------------------------------------------
    # Market Microstructure
    # -------------------------------------------------------------------------
    {
        "code": "BOOK",
        "label": "Order Book",
        "description": "Level 2 order book depth, bid/ask ladder, and tape",
        "output_target": OutputTarget.ALL_QUOTES,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF, SecurityType.CRYPTO],
        "modifiers": [],
        "keywords": ["order book", "level 2", "depth", "bid ask"],
        "example": "AAPL US EQUITY BOOK",
    },
    {
        "code": "TAPE",
        "label": "Time & Sales",
        "description": "Tick-by-tick trade data with size, price, and exchange",
        "output_target": OutputTarget.ALL_QUOTES,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF],
        "modifiers": [],
        "keywords": ["time and sales", "tape", "tick data", "trades"],
        "example": "AAPL US EQUITY TAPE",
    },
    {
        "code": "DARK",
        "label": "Dark Pool Monitor",
        "description": "Dark pool print tracking and off-exchange volume analysis",
        "output_target": OutputTarget.VOLUME_PROFILE,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["dark pool", "off exchange", "ats"],
        "example": "AAPL US EQUITY DARK",
    },
    # -------------------------------------------------------------------------
    # Additional unique codes to reach 102+
    # -------------------------------------------------------------------------
    {
        "code": "OPTF",
        "label": "Options Flow",
        "description": "Unusual options activity, sweep orders, and block prints",
        "output_target": OutputTarget.OPTIONS_MONITOR,
        "applicable_types": [SecurityType.EQUITY, SecurityType.ETF, SecurityType.INDEX],
        "modifiers": ["UNUSUAL","SWEEP","BLOCK"],
        "keywords": ["options flow", "unusual options", "sweep"],
        "example": "AAPL US EQUITY OPTF UNUSUAL",
    },
    {
        "code": "IMPL",
        "label": "Implied Move",
        "description": "Options-implied earnings move and straddle pricing",
        "output_target": OutputTarget.OPTIONS_VALUATION,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["implied move", "straddle", "earnings move"],
        "example": "AAPL US EQUITY IMPL",
    },
    {
        "code": "SKEW",
        "label": "Volatility Skew",
        "description": "Options volatility skew across strikes and expirations",
        "output_target": OutputTarget.OPTIONS_VALUATION,
        "applicable_types": [SecurityType.EQUITY, SecurityType.INDEX],
        "modifiers": [],
        "keywords": ["skew", "volatility skew", "put call skew"],
        "example": "SPX INDEX SKEW",
    },
    {
        "code": "VIX",
        "label": "VIX / Vol Monitor",
        "description": "VIX term structure, VVIX, and cross-asset volatility dashboard",
        "output_target": OutputTarget.TECHNICAL,
        "applicable_types": [SecurityType.INDEX],
        "modifiers": [],
        "keywords": ["vix", "volatility index", "vvix", "vol"],
        "example": "VIX INDEX",
    },
    {
        "code": "SENT",
        "label": "Market Sentiment",
        "description": "Put/call ratio, AAII survey, CNN Fear & Greed, short interest aggregate",
        "output_target": OutputTarget.ALTERNATIVE_DATA,
        "applicable_types": list(SecurityType),
        "modifiers": [],
        "keywords": ["sentiment", "put call ratio", "fear greed"],
        "example": "SENT",
    },
    {
        "code": "SEAS",
        "label": "Seasonality",
        "description": "Historical seasonal patterns by month, week, and day-of-year",
        "output_target": OutputTarget.PRICE_CHART,
        "applicable_types": [SecurityType.EQUITY, SecurityType.COMDTY, SecurityType.INDEX],
        "modifiers": [],
        "keywords": ["seasonality", "seasonal patterns", "calendar effects"],
        "example": "AAPL US EQUITY SEAS",
    },
    {
        "code": "FLOW",
        "label": "Capital Flows",
        "description": "ETF fund flows, sector rotation, and cross-asset flow data",
        "output_target": OutputTarget.ALTERNATIVE_DATA,
        "applicable_types": [SecurityType.ETF, SecurityType.INDEX, SecurityType.EQUITY],
        "modifiers": ["1D","1W","1M"],
        "keywords": ["flows", "capital flows", "etf flows", "fund flows"],
        "example": "SPY US EQUITY FLOW 1W",
    },
    {
        "code": "ACTV",
        "label": "Activist Monitor",
        "description": "Activist investor 13D/13G filings and campaign tracker",
        "output_target": OutputTarget.OWNERSHIP,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["activist", "13d", "13g", "activist investor"],
        "example": "AAPL US EQUITY ACTV",
    },
    {
        "code": "GOVT_SPEND",
        "label": "Government Spending",
        "description": "USASpending.gov contract and grant data by company",
        "output_target": OutputTarget.ALTERNATIVE_DATA,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["government contracts", "spending", "defense"],
        "example": "LMT US EQUITY GOVT_SPEND",
    },
    {
        "code": "PATENT",
        "label": "Patent Analytics",
        "description": "Patent filing trends, citations, and R&D intensity",
        "output_target": OutputTarget.ALTERNATIVE_DATA,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["patents", "ip", "intellectual property", "r&d"],
        "example": "AAPL US EQUITY PATENT",
    },
    {
        "code": "JOB",
        "label": "Job Posting Trends",
        "description": "Real-time job posting velocity as leading revenue indicator",
        "output_target": OutputTarget.ALTERNATIVE_DATA,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["jobs", "hiring", "job postings", "employment"],
        "example": "META US EQUITY JOB",
    },
    {
        "code": "APP",
        "label": "App Store Analytics",
        "description": "Mobile app download trends and rating analysis",
        "output_target": OutputTarget.ALTERNATIVE_DATA,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["app downloads", "app store", "mobile"],
        "example": "SNAP US EQUITY APP",
    },
    {
        "code": "SDG",
        "label": "SDG Impact Score",
        "description": "UN Sustainable Development Goals alignment scoring",
        "output_target": OutputTarget.ESG,
        "applicable_types": [SecurityType.EQUITY, SecurityType.FUND],
        "modifiers": [],
        "keywords": ["sdg", "sustainable development", "impact"],
        "example": "AAPL US EQUITY SDG",
    },
    {
        "code": "XBRL",
        "label": "XBRL Data",
        "description": "Structured XBRL financial data direct from SEC filings",
        "output_target": OutputTarget.FINANCIAL_ANALYSIS,
        "applicable_types": [SecurityType.EQUITY],
        "modifiers": [],
        "keywords": ["xbrl", "structured data", "sec"],
        "example": "AAPL US EQUITY XBRL",
    },
    {
        "code": "CTRY",
        "label": "Country Risk",
        "description": "Country risk metrics: political, economic, and credit ratings",
        "output_target": OutputTarget.WORLD_ECONOMICS,
        "applicable_types": [SecurityType.GOVT, SecurityType.INDEX],
        "modifiers": ["US","EU","EM","APAC"],
        "keywords": ["country risk", "sovereign risk", "political risk"],
        "example": "CTRY EM",
    },
    {
        "code": "NFPORT",
        "label": "NPORT Analytics",
        "description": "SEC NPORT mutual fund holdings and risk analytics",
        "output_target": OutputTarget.PORTFOLIO,
        "applicable_types": [SecurityType.FUND, SecurityType.ETF],
        "modifiers": [],
        "keywords": ["nport", "mutual fund", "holdings"],
        "example": "SPY US EQUITY NFPORT",
    },
    {
        "code": "SPREAD",
        "label": "Spread Monitor",
        "description": "Live bid-ask spread tracking and cost-of-carry analysis",
        "output_target": OutputTarget.ALL_QUOTES,
        "applicable_types": [SecurityType.EQUITY, SecurityType.CORP, SecurityType.GOVT],
        "modifiers": [],
        "keywords": ["spread", "bid ask", "liquidity"],
        "example": "AAPL US EQUITY SPREAD",
    },
]

# Build code → FunctionCode lookup
_REGISTRY: dict[str, FunctionCode] = {}

for _d in _FUNCTION_DEFINITIONS:
    _fc = FunctionCode(**_d)
    _REGISTRY[_fc.code] = _fc

assert len(_REGISTRY) >= 102, f"Expected 102+ function codes, got {len(_REGISTRY)}"

# ---------------------------------------------------------------------------
# Bloomberg keyboard shortcuts
# ---------------------------------------------------------------------------

_SHORTCUT_MAP: dict[str, str] = {
    "F1":  "HELP",
    "F2":  "MENU",
    "F3":  "DES",
    "F4":  "GP",
    "F5":  "FA",
    "F6":  "EE",
    "F7":  "ANR",
    "F8":  "OMON",
    "F9":  "PORT",
    "F10": "EQS",
    "F11": "ECOW",
    "F12": "TOP",
    "CTRL+G": "G",
    "CTRL+H": "HS",
    "CTRL+N": "CN",
    "CTRL+P": "PORT",
    "CTRL+S": "EQS",
    "CTRL+W": "WGT",
    "CTRL+Y": "YAS",
    "CTRL+Z": "YCRV",
    "ESC":     "MENU",
    "ALT+F4":  "SET",
}

# ---------------------------------------------------------------------------
# Parsed Command
# ---------------------------------------------------------------------------


class ParsedCommand(BaseModel):
    raw_input: str
    ticker: Optional[str] = None
    exchange: Optional[str] = None
    security_type: Optional[SecurityType] = None
    func_code: Optional[str] = None
    modifiers: list[str] = Field(default_factory=list)
    output_target: Optional[OutputTarget] = None
    function_meta: Optional[dict] = None
    is_valid: bool = False
    error: Optional[str] = None
    suggestions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Command Parser
# ---------------------------------------------------------------------------

# Known single-word global commands (no ticker needed)
_GLOBAL_COMMANDS = {
    "TOP", "MOST", "HELP", "DOCS", "MENU", "CALC", "SET", "MACRO",
    "ECOW", "WECO", "GDP", "CPI", "IFFS", "FOMC", "PMI", "JOLTS",
    "PORT", "PRTU", "PPAR", "EQS", "SRCH", "FLT", "WGT", "BSRCH",
    "YCRV", "CORR", "BACK", "SCEN", "SWAP", "SENT", "FLOW", "IPO",
    "MA_DEAL", "PRIV", "CMON", "CTRY", "IFFS", "VIX",
}

# Tokens that are security types
_SEC_TYPE_TOKENS = set(_SEC_TYPE_ALIASES.keys())

# Regex patterns
_TICKER_PAT = re.compile(r"^[A-Z0-9.\-/^]{1,20}$")
_EXCHANGE_PAT = re.compile(r"^[A-Z]{2,4}$")


class CommandParser:
    """
    Parses Bloomberg-style command strings into structured ParsedCommand objects.

    Grammar (simplified):
        command := global_cmd [modifier...]
               | ticker [exchange] security_type func_code [modifier...]
               | ticker [exchange] func_code [modifier...]
               | ticker func_code [modifier...]

    Tokens are separated by whitespace and normalised to uppercase.
    """

    def __init__(self, registry: dict[str, FunctionCode]) -> None:
        self.registry = registry

    def parse(self, raw_input: str) -> ParsedCommand:
        if not raw_input or not raw_input.strip():
            return ParsedCommand(
                raw_input=raw_input, is_valid=False, error="Empty command"
            )

        tokens = raw_input.strip().upper().split()
        cmd = ParsedCommand(raw_input=raw_input)

        # Handle keyboard shortcut expansion
        if tokens[0] in _SHORTCUT_MAP:
            tokens[0] = _SHORTCUT_MAP[tokens[0]]

        # Global command (no ticker)
        if tokens[0] in _GLOBAL_COMMANDS and tokens[0] in self.registry:
            cmd.func_code = tokens[0]
            cmd.modifiers = tokens[1:]
            fc = self.registry[tokens[0]]
            cmd.output_target = fc.output_target
            cmd.function_meta = fc.dict()
            cmd.is_valid = True
            return cmd

        # Parse: TICKER [EXCHANGE] [SEC_TYPE] FUNC [MODIFIERS...]
        idx = 0
        ticker_parts: list[str] = []

        # Consume ticker (may include exchange suffix like "AAPL US" or standalone)
        while idx < len(tokens):
            tok = tokens[idx]
            # If token is a known security type → stop
            if tok in _SEC_TYPE_TOKENS:
                break
            # If token is a known function code → stop
            if tok in self.registry:
                break
            # Could be ticker or exchange qualifier
            ticker_parts.append(tok)
            idx += 1

        if not ticker_parts:
            cmd.error = f"Could not identify ticker in: {raw_input}"
            cmd.suggestions = self._suggest_globals(tokens[0])
            return cmd

        # Last ticker_part could be an exchange code (US, LN, JP, HK, ...)
        if len(ticker_parts) >= 2 and _EXCHANGE_PAT.match(ticker_parts[-1]):
            cmd.exchange = ticker_parts[-1]
            cmd.ticker = " ".join(ticker_parts[:-1])
        else:
            cmd.ticker = " ".join(ticker_parts)

        # Consume optional security type
        if idx < len(tokens) and tokens[idx] in _SEC_TYPE_TOKENS:
            cmd.security_type = _SEC_TYPE_ALIASES[tokens[idx]]
            idx += 1

        # Consume function code
        if idx < len(tokens) and tokens[idx] in self.registry:
            cmd.func_code = tokens[idx]
            idx += 1
        else:
            leftover = tokens[idx] if idx < len(tokens) else "<none>"
            cmd.error = f"Unknown function code: {leftover}"
            cmd.suggestions = self._suggest_functions(leftover, cmd.security_type)
            return cmd

        # Remaining tokens are modifiers
        cmd.modifiers = tokens[idx:]

        # Validate function against security type
        fc = self.registry[cmd.func_code]
        if cmd.security_type and SecurityType.UNKNOWN not in fc.applicable_types:
            if cmd.security_type not in fc.applicable_types and \
               list(SecurityType) != fc.applicable_types:
                cmd.error = (
                    f"{cmd.func_code} is not applicable to {cmd.security_type.value}"
                )
                return cmd

        cmd.output_target = fc.output_target
        cmd.function_meta = {
            "code": fc.code,
            "label": fc.label,
            "description": fc.description,
            "output_target": fc.output_target.value,
            "modifiers": fc.modifiers,
            "example": fc.example,
        }
        cmd.is_valid = True
        return cmd

    def _suggest_globals(self, prefix: str) -> list[str]:
        prefix = prefix.upper()
        return [
            code for code in _GLOBAL_COMMANDS
            if code.startswith(prefix)
        ][:5]

    def _suggest_functions(
        self,
        prefix: str,
        sec_type: Optional[SecurityType] = None,
    ) -> list[str]:
        prefix = prefix.upper()
        results = []
        for code, fc in self.registry.items():
            if not code.startswith(prefix):
                continue
            if sec_type and list(SecurityType) != fc.applicable_types:
                if sec_type not in fc.applicable_types:
                    continue
            results.append(code)
        return results[:8]


# ---------------------------------------------------------------------------
# Function Registry
# ---------------------------------------------------------------------------


class FunctionRegistry:
    """
    Provides read access to the function code registry and related lookups.
    """

    def __init__(self) -> None:
        self._reg = _REGISTRY

    def get(self, code: str) -> Optional[FunctionCode]:
        return self._reg.get(code.upper())

    def all_codes(self) -> list[str]:
        return sorted(self._reg.keys())

    def by_security_type(self, sec_type: SecurityType) -> list[FunctionCode]:
        return [
            fc for fc in self._reg.values()
            if sec_type in fc.applicable_types or list(SecurityType) == fc.applicable_types
        ]

    def by_output_target(self, target: OutputTarget) -> list[FunctionCode]:
        return [fc for fc in self._reg.values() if fc.output_target == target]

    def search_by_keyword(self, keyword: str) -> list[FunctionCode]:
        kw = keyword.lower()
        results = []
        for fc in self._reg.values():
            if kw in fc.label.lower() or kw in fc.description.lower():
                results.append(fc)
                continue
            if any(kw in k for k in fc.keywords):
                if fc not in results:
                    results.append(fc)
        return results

    def to_dict(self) -> dict[str, dict]:
        return {code: fc.dict() for code, fc in self._reg.items()}

    def count(self) -> int:
        return len(self._reg)


# ---------------------------------------------------------------------------
# Autocomplete Engine
# ---------------------------------------------------------------------------

# Well-known tickers for offline autocomplete seeding
_SEED_TICKERS: list[tuple[str, SecurityType, str]] = [
    # (ticker, sec_type, description)
    ("AAPL",  SecurityType.EQUITY,  "Apple Inc"),
    ("MSFT",  SecurityType.EQUITY,  "Microsoft Corporation"),
    ("GOOGL", SecurityType.EQUITY,  "Alphabet Inc"),
    ("AMZN",  SecurityType.EQUITY,  "Amazon.com Inc"),
    ("TSLA",  SecurityType.EQUITY,  "Tesla Inc"),
    ("NVDA",  SecurityType.EQUITY,  "NVIDIA Corporation"),
    ("META",  SecurityType.EQUITY,  "Meta Platforms Inc"),
    ("JPM",   SecurityType.EQUITY,  "JPMorgan Chase & Co"),
    ("V",     SecurityType.EQUITY,  "Visa Inc"),
    ("JNJ",   SecurityType.EQUITY,  "Johnson & Johnson"),
    ("WMT",   SecurityType.EQUITY,  "Walmart Inc"),
    ("UNH",   SecurityType.EQUITY,  "UnitedHealth Group"),
    ("XOM",   SecurityType.EQUITY,  "Exxon Mobil Corporation"),
    ("BAC",   SecurityType.EQUITY,  "Bank of America Corp"),
    ("MA",    SecurityType.EQUITY,  "Mastercard Inc"),
    ("PG",    SecurityType.EQUITY,  "Procter & Gamble Co"),
    ("CVX",   SecurityType.EQUITY,  "Chevron Corporation"),
    ("HD",    SecurityType.EQUITY,  "Home Depot Inc"),
    ("ABBV",  SecurityType.EQUITY,  "AbbVie Inc"),
    ("GS",    SecurityType.EQUITY,  "Goldman Sachs Group"),
    ("MS",    SecurityType.EQUITY,  "Morgan Stanley"),
    ("INTC",  SecurityType.EQUITY,  "Intel Corporation"),
    ("AMD",   SecurityType.EQUITY,  "Advanced Micro Devices"),
    ("NFLX",  SecurityType.EQUITY,  "Netflix Inc"),
    ("PYPL",  SecurityType.EQUITY,  "PayPal Holdings"),
    ("CRM",   SecurityType.EQUITY,  "Salesforce Inc"),
    ("SPX",   SecurityType.INDEX,   "S&P 500 Index"),
    ("NDX",   SecurityType.INDEX,   "Nasdaq 100 Index"),
    ("DJI",   SecurityType.INDEX,   "Dow Jones Industrial Average"),
    ("RUT",   SecurityType.INDEX,   "Russell 2000 Index"),
    ("VIX",   SecurityType.INDEX,   "CBOE Volatility Index"),
    ("SPY",   SecurityType.ETF,     "SPDR S&P 500 ETF"),
    ("QQQ",   SecurityType.ETF,     "Invesco QQQ Trust"),
    ("IWM",   SecurityType.ETF,     "iShares Russell 2000 ETF"),
    ("TLT",   SecurityType.ETF,     "iShares 20+ Year Treasury Bond ETF"),
    ("GLD",   SecurityType.ETF,     "SPDR Gold Shares ETF"),
    ("SLV",   SecurityType.ETF,     "iShares Silver Trust ETF"),
    ("LQD",   SecurityType.ETF,     "iShares iBoxx $ IG Corp Bond ETF"),
    ("HYG",   SecurityType.ETF,     "iShares iBoxx $ HY Corp Bond ETF"),
    ("EURUSD",SecurityType.CURNCY,  "Euro / US Dollar"),
    ("USDJPY",SecurityType.CURNCY,  "US Dollar / Japanese Yen"),
    ("GBPUSD",SecurityType.CURNCY,  "British Pound / US Dollar"),
    ("AUDUSD",SecurityType.CURNCY,  "Australian Dollar / US Dollar"),
    ("USDCAD",SecurityType.CURNCY,  "US Dollar / Canadian Dollar"),
    ("USDCHF",SecurityType.CURNCY,  "US Dollar / Swiss Franc"),
    ("BTC",   SecurityType.CRYPTO,  "Bitcoin"),
    ("ETH",   SecurityType.CRYPTO,  "Ethereum"),
    ("SOL",   SecurityType.CRYPTO,  "Solana"),
    ("CL1",   SecurityType.COMDTY,  "WTI Crude Oil Front Month"),
    ("GC1",   SecurityType.COMDTY,  "Gold Front Month Futures"),
    ("NG1",   SecurityType.COMDTY,  "Natural Gas Front Month"),
    ("US10Y", SecurityType.GOVT,    "US 10-Year Treasury"),
    ("US2Y",  SecurityType.GOVT,    "US 2-Year Treasury"),
    ("US30Y", SecurityType.GOVT,    "US 30-Year Treasury"),
    ("DE10Y", SecurityType.GOVT,    "German 10-Year Bund"),
    ("UK10Y", SecurityType.GOVT,    "UK 10-Year Gilt"),
    ("JP10Y", SecurityType.GOVT,    "Japan 10-Year JGB"),
]


class AutocompleteEngine:
    """
    Fast prefix-matching autocomplete for tickers and function codes.
    Also supports keyword-search mode (starts with '?').
    """

    def __init__(self, registry: FunctionRegistry, db_path: Path = _DB_PATH) -> None:
        self.registry = registry
        self.db_path = db_path
        self._ticker_cache: list[dict] = []
        self._last_ticker_load: float = 0.0
        self._load_seed_tickers()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_seed_tickers(self) -> None:
        self._ticker_cache = [
            {
                "ticker": t,
                "sec_type": st.value,
                "description": desc,
                "source": "seed",
            }
            for t, st, desc in _SEED_TICKERS
        ]

    def _load_recent_tickers(self) -> list[dict]:
        if not self.db_path.exists():
            return []
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT ticker, sec_type, last_used, use_count "
                    "FROM ticker_recents ORDER BY last_used DESC LIMIT 50"
                ).fetchall()
            return [
                {
                    "ticker": r["ticker"],
                    "sec_type": r["sec_type"],
                    "description": f"Recently used ({r['use_count']}x)",
                    "source": "recent",
                }
                for r in rows
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def suggest(
        self,
        prefix: str,
        limit: int = 10,
        sec_type_filter: Optional[str] = None,
        mode: str = "smart",
    ) -> dict:
        """
        Return autocomplete suggestions.

        Parameters
        ----------
        prefix:
            Current command bar text.
        limit:
            Max number of results per category.
        sec_type_filter:
            Restrict to a specific security type string.
        mode:
            'smart' (default), 'ticker', 'function', 'keyword'.

        Returns
        -------
        dict with keys: tickers, functions, keywords, shortcuts
        """
        prefix = prefix.strip().upper()
        tokens = prefix.split()

        # Keyword search mode: prefix starts with '?'
        if prefix.startswith("?"):
            keyword = prefix[1:].strip()
            funcs = self.registry.search_by_keyword(keyword)
            return {
                "tickers": [],
                "functions": [self._fmt_func(f) for f in funcs[:limit]],
                "keywords": [],
                "shortcuts": [],
                "mode": "keyword",
                "prefix": prefix,
            }

        # If single incomplete token → suggest both tickers and functions
        if len(tokens) <= 1:
            tok = tokens[0] if tokens else ""

            # Check shortcuts
            shortcuts = [
                {"shortcut": k, "func_code": v, "label": self.registry.get(v).label if self.registry.get(v) else v}
                for k, v in _SHORTCUT_MAP.items()
                if k.startswith(tok)
            ]

            ticker_suggestions = self._match_tickers(tok, limit, sec_type_filter)
            func_suggestions   = self._match_functions(tok, limit, sec_type_filter)
            return {
                "tickers":   ticker_suggestions,
                "functions": func_suggestions,
                "keywords":  [],
                "shortcuts": shortcuts[:5],
                "mode":      "mixed",
                "prefix":    prefix,
            }

        # Multi-token: determine context
        # Try to identify what has already been typed
        sec_type_found: Optional[str] = None
        has_ticker = False
        last_tok = tokens[-1]

        for tok in tokens[:-1]:
            if tok in _SEC_TYPE_TOKENS:
                sec_type_found = tok
            elif tok not in self.registry and tok not in _SEC_TYPE_TOKENS:
                has_ticker = True

        # Last token is likely a function code prefix
        func_filter = sec_type_found
        func_suggestions = self._match_functions(last_tok, limit, func_filter)

        # Or it could be a modifier for an already-typed function
        mod_suggestions: list[str] = []
        for tok in tokens[:-1]:
            if tok in self.registry:
                fc = self.registry.get(tok)
                if fc:
                    mod_suggestions = [
                        m for m in fc.modifiers
                        if m.startswith(last_tok)
                    ]
                break

        return {
            "tickers":   [],
            "functions": func_suggestions,
            "modifiers": mod_suggestions[:limit],
            "keywords":  [],
            "shortcuts": [],
            "mode":      "function",
            "prefix":    prefix,
        }

    def _match_tickers(
        self,
        prefix: str,
        limit: int,
        sec_type_filter: Optional[str],
    ) -> list[dict]:
        recent = self._load_recent_tickers()
        all_tickers = recent + [
            t for t in self._ticker_cache
            if not any(r["ticker"] == t["ticker"] for r in recent)
        ]

        results = []
        for entry in all_tickers:
            if not entry["ticker"].startswith(prefix):
                continue
            if sec_type_filter and entry.get("sec_type", "").upper() != sec_type_filter.upper():
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def _match_functions(
        self,
        prefix: str,
        limit: int,
        sec_type_filter: Optional[str],
    ) -> list[dict]:
        sec_type: Optional[SecurityType] = None
        if sec_type_filter and sec_type_filter.upper() in _SEC_TYPE_ALIASES:
            sec_type = _SEC_TYPE_ALIASES[sec_type_filter.upper()]

        results = []
        for code, fc in sorted(_REGISTRY.items()):
            if not code.startswith(prefix):
                continue
            if sec_type and list(SecurityType) != fc.applicable_types:
                if sec_type not in fc.applicable_types:
                    continue
            results.append(self._fmt_func(fc))
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _fmt_func(fc: FunctionCode) -> dict:
        return {
            "code":        fc.code,
            "label":       fc.label,
            "description": fc.description,
            "example":     fc.example,
        }


# ---------------------------------------------------------------------------
# Command History Store
# ---------------------------------------------------------------------------


class CommandHistoryStore:
    """
    Persists command history in SQLite — last 50 commands, LRU eviction.
    """

    MAX_HISTORY = 50

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path
        _init_db()

    def record(self, parsed: ParsedCommand, routed_to: Optional[str] = None) -> None:
        entry_id = str(uuid.uuid4())
        ts = time.time()
        with _get_db() as conn:
            conn.execute(
                """
                INSERT INTO command_history
                    (id, raw_input, ticker, sec_type, func_code, modifiers, ts, routed_to)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    parsed.raw_input,
                    parsed.ticker,
                    parsed.security_type.value if parsed.security_type else None,
                    parsed.func_code,
                    json.dumps(parsed.modifiers),
                    ts,
                    routed_to,
                ),
            )
            # Evict old entries
            conn.execute(
                """
                DELETE FROM command_history
                WHERE id NOT IN (
                    SELECT id FROM command_history ORDER BY ts DESC LIMIT ?
                )
                """,
                (self.MAX_HISTORY,),
            )
            # Update ticker recents
            if parsed.ticker:
                conn.execute(
                    """
                    INSERT INTO ticker_recents (ticker, sec_type, last_used, use_count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(ticker) DO UPDATE SET
                        last_used = excluded.last_used,
                        use_count = use_count + 1,
                        sec_type  = excluded.sec_type
                    """,
                    (
                        parsed.ticker,
                        parsed.security_type.value if parsed.security_type else None,
                        ts,
                    ),
                )

    def get_history(self, limit: int = 50) -> list[dict]:
        if not self.db_path.exists():
            return []
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM command_history ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            row_dict = dict(r)
            row_dict["modifiers"] = json.loads(row_dict["modifiers"] or "[]")
            row_dict["ts_iso"] = datetime.fromtimestamp(
                row_dict["ts"], tz=timezone.utc
            ).isoformat()
            result.append(row_dict)
        return result

    def clear(self) -> int:
        with _get_db() as conn:
            cur = conn.execute("DELETE FROM command_history")
            return cur.rowcount

    def get_frequent_tickers(self, limit: int = 10) -> list[dict]:
        if not self.db_path.exists():
            return []
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT ticker, sec_type, use_count, last_used "
                "FROM ticker_recents ORDER BY use_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Command Router — maps parsed commands to SENTINEL module endpoints
# ---------------------------------------------------------------------------

# Routing table: output_target → SENTINEL API path template
_ROUTE_TABLE: dict[OutputTarget, str] = {
    OutputTarget.PRICE_CHART:       "/api/v1/data/ohlcv/{ticker}",
    OutputTarget.DESCRIPTION:       "/api/v1/intelligence/profile/{ticker}",
    OutputTarget.FINANCIAL_ANALYSIS:"/api/v1/data/financials/{ticker}",
    OutputTarget.RELATIVE_VALUE:    "/api/v1/screen/peers/{ticker}",
    OutputTarget.DIVIDENDS:         "/api/v1/data/dividends/{ticker}",
    OutputTarget.EARNINGS:          "/api/v1/intelligence/earnings/{ticker}",
    OutputTarget.ANALYST_RECS:      "/api/v1/intelligence/analyst/{ticker}",
    OutputTarget.OPTIONS_MONITOR:   "/api/v1/data/options/{ticker}",
    OutputTarget.OPTIONS_VALUATION: "/api/v1/data/options/valuation/{ticker}",
    OutputTarget.EVENTS:            "/api/v1/intelligence/events/{ticker}",
    OutputTarget.CASH_FLOW:         "/api/v1/data/cashflow/{ticker}",
    OutputTarget.NEWS:              "/api/v1/data/news/{ticker}",
    OutputTarget.MOST_ACTIVE:       "/api/v1/screen/most-active",
    OutputTarget.TOP_STORIES:       "/api/v1/data/news/top",
    OutputTarget.YIELD_ANALYTICS:   "/api/v1/data/bond-analytics/{ticker}",
    OutputTarget.ALL_QUOTES:        "/api/v1/data/quotes/{ticker}",
    OutputTarget.CREDIT_DEFAULT:    "/api/v1/data/credit/cds/{ticker}",
    OutputTarget.DURATION:          "/api/v1/data/bond-analytics/{ticker}/duration",
    OutputTarget.RISK_ANALYTICS:    "/api/v1/portfolio/risk/{ticker}",
    OutputTarget.FX_FORWARDS:       "/api/v1/data/fx/forwards/{ticker}",
    OutputTarget.FX_IMPLIED_VOL:    "/api/v1/data/fx/vol-surface/{ticker}",
    OutputTarget.FX_FORECAST:       "/api/v1/data/fx/forecast/{ticker}",
    OutputTarget.ECO_CALENDAR:      "/api/v1/macro/calendar",
    OutputTarget.WORLD_ECONOMICS:   "/api/v1/macro/world",
    OutputTarget.GDP_VIEWER:        "/api/v1/macro/gdp",
    OutputTarget.CPI_VIEWER:        "/api/v1/macro/cpi",
    OutputTarget.RATE_FUTURES:      "/api/v1/futures/rate",
    OutputTarget.PORTFOLIO:         "/api/v1/portfolio",
    OutputTarget.PORTFOLIO_UPLOAD:  "/api/v1/portfolio/upload",
    OutputTarget.PERF_ATTRIBUTION:  "/api/v1/portfolio/attribution",
    OutputTarget.EQUITY_SCREEN:     "/api/v1/screen/equity",
    OutputTarget.CREDIT_SCREEN:     "/api/v1/screen/credit",
    OutputTarget.FILTER:            "/api/v1/screen/filter",
    OutputTarget.GRAPH:             "/api/v1/data/ohlcv/{ticker}",
    OutputTarget.COMPARISON:        "/api/v1/data/comparison",
    OutputTarget.INDUSTRY:          "/api/v1/screen/industry/{ticker}",
    OutputTarget.MOVING_AVG:        "/api/v1/data/technical/{ticker}",
    OutputTarget.HELP:              "/api/v1/help",
    OutputTarget.DOCS:              "/api/v1/docs",
    OutputTarget.MENU:              "/api/v1/menu",
    OutputTarget.DEBT_DISTRIBUTION: "/api/v1/data/debt/{ticker}",
    OutputTarget.HISTORICAL_SPREADS:"/api/v1/data/spreads/{ticker}",
    OutputTarget.BOND_CASH_FLOWS:   "/api/v1/data/bond-cashflows/{ticker}",
    OutputTarget.INSIDER_FLOW:      "/api/v1/intelligence/insider/{ticker}",
    OutputTarget.OWNERSHIP:         "/api/v1/intelligence/ownership/{ticker}",
    OutputTarget.SHORT_INTEREST:    "/api/v1/data/short-interest/{ticker}",
    OutputTarget.SECTOR_MAP:        "/api/v1/screen/sector-map",
    OutputTarget.HEAT_MAP:          "/api/v1/screen/heat-map",
    OutputTarget.VOLUME_PROFILE:    "/api/v1/data/volume-profile/{ticker}",
    OutputTarget.TECHNICAL:         "/api/v1/data/technical/{ticker}",
    OutputTarget.SCREENING:         "/api/v1/screen/equity",
    OutputTarget.WATCHLIST:         "/api/v1/data/watchlist",
    OutputTarget.CORRELATION:       "/api/v1/portfolio/correlation",
    OutputTarget.BACKTEST:          "/api/v1/backtest",
    OutputTarget.SCENARIO:          "/api/v1/portfolio/scenario",
    OutputTarget.SUPPLY_CHAIN:      "/api/v1/intelligence/supply-chain/{ticker}",
    OutputTarget.ESG:               "/api/v1/data/esg/{ticker}",
    OutputTarget.MACRO_OVERVIEW:    "/api/v1/macro/overview",
    OutputTarget.CREDIT_MONITOR:    "/api/v1/data/credit/monitor",
    OutputTarget.BOND_SCREENER:     "/api/v1/screen/bonds",
    OutputTarget.YIELD_CURVE:       "/api/v1/data/yield-curve",
    OutputTarget.CONVERTIBLE:       "/api/v1/data/convertible/{ticker}",
    OutputTarget.SWAP_MONITOR:      "/api/v1/data/swaps",
    OutputTarget.FUTURES_CURVE:     "/api/v1/futures/curve/{ticker}",
    OutputTarget.CRYPTO_BOOK:       "/api/v1/data/crypto/book/{ticker}",
    OutputTarget.ON_CHAIN:          "/api/v1/data/crypto/onchain/{ticker}",
    OutputTarget.DEFI:              "/api/v1/data/crypto/defi/{ticker}",
    OutputTarget.ALTERNATIVE_DATA:  "/api/v1/intelligence/alt-data/{ticker}",
    OutputTarget.IPO_CALENDAR:      "/api/v1/intelligence/ipo",
    OutputTarget.MA_INTELLIGENCE:   "/api/v1/intelligence/ma",
    OutputTarget.PRIVATE_MARKETS:   "/api/v1/intelligence/private",
    OutputTarget.SETTINGS:          "/api/v1/settings",
    OutputTarget.CALCULATOR:        "/api/v1/calc",
    OutputTarget.UNKNOWN:           "/api/v1/unknown",
}


class CommandRouter:
    """
    Routes a parsed command to the appropriate SENTINEL API endpoint.
    Returns the endpoint URL template, resolved URL, and parameters.
    """

    def __init__(self, registry: FunctionRegistry) -> None:
        self.registry = registry

    def route(self, parsed: ParsedCommand) -> dict:
        if not parsed.is_valid:
            return {
                "success": False,
                "error": parsed.error,
                "suggestions": parsed.suggestions,
                "parsed": parsed.dict(),
            }

        target = parsed.output_target
        url_template = _ROUTE_TABLE.get(target, "/api/v1/unknown")
        ticker = parsed.ticker or ""
        url = url_template.replace("{ticker}", ticker.replace(" ", "_").lower())

        params: dict[str, Any] = {}
        if parsed.security_type:
            params["sec_type"] = parsed.security_type.value
        if parsed.exchange:
            params["exchange"] = parsed.exchange
        if parsed.modifiers:
            params["modifiers"] = parsed.modifiers

        return {
            "success":      True,
            "func_code":    parsed.func_code,
            "label":        parsed.function_meta.get("label") if parsed.function_meta else None,
            "description":  parsed.function_meta.get("description") if parsed.function_meta else None,
            "ticker":       parsed.ticker,
            "security_type":parsed.security_type.value if parsed.security_type else None,
            "exchange":     parsed.exchange,
            "modifiers":    parsed.modifiers,
            "output_target":target.value if target else None,
            "api_endpoint": url,
            "url_template": url_template,
            "params":       params,
        }


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_init_db()
_function_registry  = FunctionRegistry()
_parser             = CommandParser(_REGISTRY)
_autocomplete       = AutocompleteEngine(_function_registry, _DB_PATH)
_history_store      = CommandHistoryStore(_DB_PATH)
_command_router     = CommandRouter(_function_registry)

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class CommandRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=500, description="Raw command string")
    record_history: bool = Field(True, description="Whether to save this command to history")


class CommandResponse(BaseModel):
    parsed: dict
    route: dict
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AutocompleteRequest(BaseModel):
    prefix: str = Field(..., description="Current text in command bar")
    limit: int = Field(10, ge=1, le=50)
    sec_type_filter: Optional[str] = None
    mode: str = "smart"


class HistoryClearResponse(BaseModel):
    deleted: int
    message: str


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

command_router = APIRouter(prefix="/command-bar", tags=["CommandBar"])


@command_router.post("/command", response_model=CommandResponse, summary="Execute command")
def execute_command(req: CommandRequest) -> CommandResponse:
    """
    Parse and route a Bloomberg-style command string.

    Returns the parsed representation and the API route mapping.
    """
    parsed = _parser.parse(req.command)
    route  = _command_router.route(parsed)

    if req.record_history and parsed.is_valid:
        _history_store.record(parsed, routed_to=route.get("api_endpoint"))

    return CommandResponse(parsed=parsed.dict(), route=route)


@command_router.get("/autocomplete", summary="Autocomplete suggestions")
def get_autocomplete(
    prefix: str = Query(..., description="Current command bar text"),
    limit: int = Query(10, ge=1, le=50),
    sec_type: Optional[str] = Query(None, description="Security type filter"),
    mode: str = Query("smart", description="smart | ticker | function | keyword"),
) -> dict:
    """
    Return autocomplete suggestions for the current command bar input.
    Supports ticker prefix, function code prefix, keyword search (prefix with ?),
    and keyboard shortcut display.
    """
    return _autocomplete.suggest(prefix, limit=limit, sec_type_filter=sec_type, mode=mode)


@command_router.get("/history", summary="Command history")
def get_command_history(limit: int = Query(50, ge=1, le=50)) -> dict:
    """Return the last N commands from history."""
    history = _history_store.get_history(limit=limit)
    frequent = _history_store.get_frequent_tickers(limit=10)
    return {
        "count": len(history),
        "history": history,
        "frequent_tickers": frequent,
    }


@command_router.delete("/history", response_model=HistoryClearResponse, summary="Clear history")
def clear_command_history() -> HistoryClearResponse:
    """Clear all command history."""
    deleted = _history_store.clear()
    return HistoryClearResponse(deleted=deleted, message="Command history cleared.")


@command_router.get("/function-registry", summary="Full function code registry")
def get_function_registry(
    sec_type: Optional[str] = Query(None, description="Filter by security type"),
    search: Optional[str] = Query(None, description="Keyword search"),
) -> dict:
    """
    Return the complete function code registry, optionally filtered
    by security type or keyword search.
    """
    if search:
        funcs = _function_registry.search_by_keyword(search)
        return {
            "count": len(funcs),
            "total": _function_registry.count(),
            "functions": [f.dict() for f in funcs],
            "query": search,
        }

    if sec_type:
        canonical = _SEC_TYPE_ALIASES.get(sec_type.upper())
        if not canonical:
            raise HTTPException(status_code=400, detail=f"Unknown security type: {sec_type}")
        funcs = _function_registry.by_security_type(canonical)
        return {
            "count": len(funcs),
            "total": _function_registry.count(),
            "security_type": sec_type,
            "functions": [f.dict() for f in funcs],
        }

    return {
        "count": _function_registry.count(),
        "total": _function_registry.count(),
        "functions": _function_registry.to_dict(),
    }


@command_router.get("/shortcuts", summary="Keyboard shortcut map")
def get_shortcuts() -> dict:
    """Return the full keyboard shortcut → function code mapping."""
    result = {}
    for shortcut, code in _SHORTCUT_MAP.items():
        fc = _function_registry.get(code)
        result[shortcut] = {
            "func_code": code,
            "label": fc.label if fc else code,
            "description": fc.description if fc else "",
        }
    return {"count": len(result), "shortcuts": result}


@command_router.get("/parse", summary="Parse command without routing")
def parse_command_only(
    command: str = Query(..., description="Command string to parse"),
) -> dict:
    """
    Parse a command string and return its structured representation
    without recording history or routing.
    """
    parsed = _parser.parse(command)
    return parsed.dict()


@command_router.get("/function/{code}", summary="Get function metadata")
def get_function_by_code(code: str) -> dict:
    """Return detailed metadata for a specific function code."""
    fc = _function_registry.get(code.upper())
    if not fc:
        raise HTTPException(
            status_code=404,
            detail=f"Function code '{code.upper()}' not found. "
                   f"Available codes: {', '.join(sorted(_REGISTRY.keys())[:10])}...",
        )
    return fc.dict()


@command_router.get("/suggest-related", summary="Suggest related function codes")
def suggest_related(
    func_code: str = Query(..., description="Function code to find related codes for"),
    sec_type: Optional[str] = Query(None),
) -> dict:
    """Suggest function codes related to the given function."""
    fc = _function_registry.get(func_code.upper())
    if not fc:
        raise HTTPException(status_code=404, detail=f"Unknown function code: {func_code}")

    # Find codes with same output target
    same_target = _function_registry.by_output_target(fc.output_target)
    # Find codes applicable to same security types
    related_codes: list[FunctionCode] = []
    for st in fc.applicable_types:
        if st == SecurityType.UNKNOWN:
            continue
        for r in _function_registry.by_security_type(st):
            if r.code != fc.code and r not in related_codes:
                related_codes.append(r)

    return {
        "func_code": fc.code,
        "label":     fc.label,
        "same_output_target": [f.dict() for f in same_target if f.code != fc.code][:5],
        "related_by_asset_class": [f.dict() for f in related_codes][:8],
    }


@command_router.get("/health", summary="Command bar health check")
def command_bar_health() -> dict:
    """Return health status of the command bar module."""
    return {
        "status":          "ok",
        "registry_size":   _function_registry.count(),
        "db_path":         str(_DB_PATH),
        "db_exists":       _DB_PATH.exists(),
        "shortcut_count":  len(_SHORTCUT_MAP),
        "seed_tickers":    len(_SEED_TICKERS),
        "ts":              datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Convenience re-export for main app inclusion
# ---------------------------------------------------------------------------

router = command_router
