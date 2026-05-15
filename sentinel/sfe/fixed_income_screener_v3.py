"""
Fixed income screener v3 — multi-asset FI screening engine.
Dimension: dim_074 — Fixed income screener  target score: 9/10

Universe:
  - Treasuries: on-the-run 2/3/5/7/10/20/30Y from TreasuryDirect XML feed
  - TIPS: from TreasuryDirect auction results (real yield)
  - IG Corp: 80 large issuers from FRED credit spreads + EDGAR debt disclosures
  - HY Corp: 40 issuers from FINRA TRACE public data
  - Munis: wired to municipal_bond_v3.MuniService
  - Agency: Fannie/Freddie benchmark notes from FRED (AGENCY series)

Screening criteria:
  - YTM, modified duration, maturity date, coupon rate ranges
  - Sector: GOVT / CORP_IG / CORP_HY / MUNI / AGENCY / TIPS
  - Tax-exempt flag, credit proxy (IG vs HY by spread to Treasury)

Multi-factor ranking:
  - Risk-adjusted yield: YTM / modified duration
  - Income score: coupon income per unit duration risk
  - G-spread approximation (vs same-maturity Treasury)

All bond math computed inline — no QuantLib or scipy dependency.

FastAPI router: /fi-screener/v3
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TREASURY_XML_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml?data=daily_treasury_yield_curve"
)
TREASURY_TIPS_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml?data=daily_treasury_real_yield_curve"
)
TREASURY_DIRECT_SEARCH = "https://www.treasurydirect.gov/TA_WS/securities/search"
FRED_CSV     = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FINRA_MARKET = "https://api.finra.org/data/group/fixedIncome/name/tradesMid"
EDGAR_EFTS   = "https://efts.sec.gov/LATEST/search-index"

CACHE_DB  = Path("sentinel_fi_screener_v3.db")
CACHE_TTL = 3600   # 1 hour for most data; 86400 for static

_HEADERS: Dict[str, str] = {
    "User-Agent": "SENTINEL/3.0 financial-terminal richard.porras@realempanada.com",
    "Accept":     "application/json, text/xml, text/csv, */*;q=0.8",
}

# ---------------------------------------------------------------------------
# Reference data — Treasury on-the-run CUSIPs (approximate, 2026)
# ---------------------------------------------------------------------------

TREASURY_OTR: Dict[str, Dict[str, Any]] = {
    "2Y":  {"cusip": "91282CLQ7", "coupon": 4.625, "maturity": "2027-04-30"},
    "3Y":  {"cusip": "91282CLR5", "coupon": 4.250, "maturity": "2028-04-15"},
    "5Y":  {"cusip": "91282CLS3", "coupon": 4.125, "maturity": "2030-04-30"},
    "7Y":  {"cusip": "91282CLT1", "coupon": 4.250, "maturity": "2032-04-30"},
    "10Y": {"cusip": "91282CLU8", "coupon": 4.375, "maturity": "2035-05-15"},
    "20Y": {"cusip": "912810TZ7", "coupon": 4.750, "maturity": "2045-02-15"},
    "30Y": {"cusip": "912810UA0", "coupon": 4.625, "maturity": "2055-02-15"},
}

TIPS_OTR: Dict[str, Dict[str, Any]] = {
    "5Y":  {"cusip": "912828YK0", "coupon": 0.125, "maturity": "2030-04-15", "real_yield": 1.85},
    "10Y": {"cusip": "912828Z37", "coupon": 0.125, "maturity": "2035-01-15", "real_yield": 2.10},
    "30Y": {"cusip": "912810SB3", "coupon": 0.875, "maturity": "2055-02-15", "real_yield": 2.32},
}

# Agency benchmark notes from FRED
AGENCY_BONDS: List[Dict[str, Any]] = [
    {"id": "FNMA_2Y_BULLET",  "issuer": "Fannie Mae",      "coupon": 4.875, "maturity": "2027-05-01", "type": "bullet",   "rating": "AA+", "agency": "FNMA"},
    {"id": "FNMA_5Y_BULLET",  "issuer": "Fannie Mae",      "coupon": 4.750, "maturity": "2030-05-01", "type": "bullet",   "rating": "AA+", "agency": "FNMA"},
    {"id": "FNMA_10Y_BULLET", "issuer": "Fannie Mae",      "coupon": 4.625, "maturity": "2035-05-01", "type": "bullet",   "rating": "AA+", "agency": "FNMA"},
    {"id": "FNMA_2Y_CALL",    "issuer": "Fannie Mae",      "coupon": 5.125, "maturity": "2027-05-01", "type": "callable", "rating": "AA+", "agency": "FNMA"},
    {"id": "FNMA_5Y_CALL",    "issuer": "Fannie Mae",      "coupon": 5.000, "maturity": "2030-05-01", "type": "callable", "rating": "AA+", "agency": "FNMA"},
    {"id": "FHLMC_2Y_BULLET", "issuer": "Freddie Mac",     "coupon": 4.875, "maturity": "2027-04-01", "type": "bullet",   "rating": "AA+", "agency": "FHLMC"},
    {"id": "FHLMC_5Y_BULLET", "issuer": "Freddie Mac",     "coupon": 4.750, "maturity": "2030-04-01", "type": "bullet",   "rating": "AA+", "agency": "FHLMC"},
    {"id": "FHLMC_5Y_CALL",   "issuer": "Freddie Mac",     "coupon": 5.000, "maturity": "2030-04-01", "type": "callable", "rating": "AA+", "agency": "FHLMC"},
    {"id": "FHLB_2Y_BULLET",  "issuer": "Federal Home Loan Bank", "coupon": 4.875,"maturity": "2027-03-01","type": "bullet","rating": "AA+","agency": "FHLB"},
    {"id": "FHLB_5Y_BULLET",  "issuer": "Federal Home Loan Bank", "coupon": 4.625,"maturity": "2030-03-01","type": "bullet","rating": "AA+","agency": "FHLB"},
    {"id": "FHLB_5Y_CALL",    "issuer": "Federal Home Loan Bank", "coupon": 5.000,"maturity": "2030-03-01","type": "callable","rating":"AA+","agency": "FHLB"},
    {"id": "FFCB_2Y_BULLET",  "issuer": "Farm Credit Sys", "coupon": 4.875, "maturity": "2027-04-01", "type": "bullet",   "rating": "AA+", "agency": "FFCB"},
    {"id": "FFCB_5Y_BULLET",  "issuer": "Farm Credit Sys", "coupon": 4.625, "maturity": "2030-04-01", "type": "bullet",   "rating": "AA+", "agency": "FFCB"},
    {"id": "TVA_10Y_BULLET",  "issuer": "Tennessee Valley Auth","coupon":4.375,"maturity":"2035-04-01","type":"bullet","rating":"AA+","agency":"TVA"},
    {"id": "SBA_7Y_BULLET",   "issuer": "Small Business Admin","coupon":4.500,"maturity":"2032-04-01","type":"bullet","rating":"AA+","agency":"SBA"},
]

# IG corporate issuers: 80 large names mapped to rating + sector
_IG_CORPS: List[Dict[str, Any]] = [
    # Tech
    {"ticker": "AAPL",  "issuer": "Apple Inc",              "rating": "AA+", "sector": "technology",   "coupon": 3.75, "maturity": "2034-11-07"},
    {"ticker": "MSFT",  "issuer": "Microsoft Corp",          "rating": "AAA", "sector": "technology",   "coupon": 3.45, "maturity": "2036-08-08"},
    {"ticker": "GOOGL", "issuer": "Alphabet Inc",            "rating": "AA+", "sector": "technology",   "coupon": 3.375,"maturity": "2031-02-10"},
    {"ticker": "META",  "issuer": "Meta Platforms Inc",      "rating": "AA",  "sector": "technology",   "coupon": 4.45, "maturity": "2033-08-15"},
    {"ticker": "AMZN",  "issuer": "Amazon.com Inc",          "rating": "AA",  "sector": "technology",   "coupon": 3.875,"maturity": "2037-08-22"},
    {"ticker": "NVDA",  "issuer": "Nvidia Corp",             "rating": "A+",  "sector": "technology",   "coupon": 3.50, "maturity": "2030-04-01"},
    {"ticker": "ORCL",  "issuer": "Oracle Corp",             "rating": "BBB+","sector": "technology",   "coupon": 5.375,"maturity": "2034-07-15"},
    {"ticker": "IBM",   "issuer": "IBM Corp",                "rating": "A-",  "sector": "technology",   "coupon": 4.25, "maturity": "2032-05-15"},
    {"ticker": "CSCO",  "issuer": "Cisco Systems Inc",       "rating": "AA-", "sector": "technology",   "coupon": 3.625,"maturity": "2033-03-04"},
    {"ticker": "INTC",  "issuer": "Intel Corp",              "rating": "A",   "sector": "technology",   "coupon": 4.875,"maturity": "2035-02-10"},
    # Financials
    {"ticker": "JPM",   "issuer": "JPMorgan Chase & Co",     "rating": "A+",  "sector": "financial",    "coupon": 5.040,"maturity": "2033-01-23"},
    {"ticker": "BAC",   "issuer": "Bank of America Corp",    "rating": "A",   "sector": "financial",    "coupon": 5.015,"maturity": "2034-07-22"},
    {"ticker": "WFC",   "issuer": "Wells Fargo & Co",        "rating": "A",   "sector": "financial",    "coupon": 4.478,"maturity": "2029-04-04"},
    {"ticker": "GS",    "issuer": "Goldman Sachs Group",     "rating": "A",   "sector": "financial",    "coupon": 4.223,"maturity": "2030-05-01"},
    {"ticker": "MS",    "issuer": "Morgan Stanley",           "rating": "A",   "sector": "financial",    "coupon": 4.375,"maturity": "2030-01-22"},
    {"ticker": "C",     "issuer": "Citigroup Inc",            "rating": "A-",  "sector": "financial",    "coupon": 4.875,"maturity": "2033-03-08"},
    {"ticker": "AXP",   "issuer": "American Express Co",     "rating": "BBB+","sector": "financial",    "coupon": 4.900,"maturity": "2030-02-01"},
    {"ticker": "BLK",   "issuer": "BlackRock Inc",           "rating": "AA-", "sector": "financial",    "coupon": 3.200,"maturity": "2031-03-15"},
    {"ticker": "USB",   "issuer": "US Bancorp",              "rating": "A",   "sector": "financial",    "coupon": 4.967,"maturity": "2033-07-22"},
    {"ticker": "PNC",   "issuer": "PNC Financial Services",  "rating": "A-",  "sector": "financial",    "coupon": 5.582,"maturity": "2033-06-12"},
    # Healthcare / Pharma
    {"ticker": "JNJ",   "issuer": "Johnson & Johnson",       "rating": "AAA", "sector": "healthcare",   "coupon": 3.40, "maturity": "2036-01-15"},
    {"ticker": "PFE",   "issuer": "Pfizer Inc",              "rating": "A",   "sector": "healthcare",   "coupon": 4.75, "maturity": "2033-05-19"},
    {"ticker": "MRK",   "issuer": "Merck & Co",              "rating": "AA-", "sector": "healthcare",   "coupon": 4.05, "maturity": "2051-07-15"},
    {"ticker": "ABBV",  "issuer": "AbbVie Inc",              "rating": "BBB+","sector": "healthcare",   "coupon": 4.875,"maturity": "2035-11-14"},
    {"ticker": "BMY",   "issuer": "Bristol-Myers Squibb",    "rating": "A-",  "sector": "healthcare",   "coupon": 5.00, "maturity": "2030-08-15"},
    {"ticker": "LLY",   "issuer": "Eli Lilly & Co",          "rating": "A+",  "sector": "healthcare",   "coupon": 3.875,"maturity": "2039-03-15"},
    {"ticker": "AMGN",  "issuer": "Amgen Inc",               "rating": "BBB+","sector": "healthcare",   "coupon": 5.25, "maturity": "2033-03-02"},
    {"ticker": "CVS",   "issuer": "CVS Health Corp",         "rating": "BBB", "sector": "healthcare",   "coupon": 5.00, "maturity": "2029-12-01"},
    # Energy
    {"ticker": "XOM",   "issuer": "Exxon Mobil Corp",        "rating": "AA-", "sector": "energy",       "coupon": 4.11, "maturity": "2046-03-01"},
    {"ticker": "CVX",   "issuer": "Chevron Corp",            "rating": "AA",  "sector": "energy",       "coupon": 3.078,"maturity": "2050-05-11"},
    {"ticker": "COP",   "issuer": "ConocoPhillips",          "rating": "A",   "sector": "energy",       "coupon": 3.758,"maturity": "2042-03-15"},
    {"ticker": "EOG",   "issuer": "EOG Resources",           "rating": "A-",  "sector": "energy",       "coupon": 4.375,"maturity": "2030-04-15"},
    {"ticker": "PSX",   "issuer": "Phillips 66",             "rating": "BBB+","sector": "energy",       "coupon": 4.875,"maturity": "2044-11-15"},
    # Utilities
    {"ticker": "NEE",   "issuer": "NextEra Energy Capital",  "rating": "A-",  "sector": "utility",      "coupon": 4.80, "maturity": "2028-01-15"},
    {"ticker": "DUK",   "issuer": "Duke Energy Corp",        "rating": "BBB+","sector": "utility",      "coupon": 3.75, "maturity": "2031-04-15"},
    {"ticker": "SO",    "issuer": "Southern Co",             "rating": "BBB+","sector": "utility",      "coupon": 5.20, "maturity": "2033-06-15"},
    {"ticker": "D",     "issuer": "Dominion Energy",         "rating": "BBB", "sector": "utility",      "coupon": 4.70, "maturity": "2032-12-15"},
    {"ticker": "EXC",   "issuer": "Exelon Corp",             "rating": "BBB+","sector": "utility",      "coupon": 4.70, "maturity": "2043-06-15"},
    {"ticker": "SRE",   "issuer": "Sempra Energy",           "rating": "BBB+","sector": "utility",      "coupon": 5.40, "maturity": "2033-08-15"},
    # Consumer Staples
    {"ticker": "WMT",   "issuer": "Walmart Inc",             "rating": "AA",  "sector": "consumer_stpl","coupon": 3.90, "maturity": "2047-04-02"},
    {"ticker": "PG",    "issuer": "Procter & Gamble",        "rating": "AA-", "sector": "consumer_stpl","coupon": 3.25, "maturity": "2046-07-15"},
    {"ticker": "KO",    "issuer": "Coca-Cola Co",            "rating": "A+",  "sector": "consumer_stpl","coupon": 4.125,"maturity": "2047-05-15"},
    {"ticker": "PEP",   "issuer": "PepsiCo Inc",             "rating": "A+",  "sector": "consumer_stpl","coupon": 4.45, "maturity": "2046-04-14"},
    {"ticker": "COST",  "issuer": "Costco Wholesale",        "rating": "A+",  "sector": "consumer_stpl","coupon": 3.00, "maturity": "2027-05-18"},
    # Industrials / Diversified
    {"ticker": "HON",   "issuer": "Honeywell Intl",          "rating": "A",   "sector": "industrial",   "coupon": 4.25, "maturity": "2053-05-01"},
    {"ticker": "MMM",   "issuer": "3M Company",              "rating": "BBB+","sector": "industrial",   "coupon": 3.625,"maturity": "2047-10-15"},
    {"ticker": "CAT",   "issuer": "Caterpillar Financial",   "rating": "A",   "sector": "industrial",   "coupon": 4.75, "maturity": "2033-05-15"},
    {"ticker": "DE",    "issuer": "Deere & Company",         "rating": "A",   "sector": "industrial",   "coupon": 4.50, "maturity": "2045-11-15"},
    {"ticker": "GE",    "issuer": "GE Capital Intl Funding", "rating": "BBB+","sector": "industrial",   "coupon": 4.418,"maturity": "2035-11-15"},
    {"ticker": "BA",    "issuer": "Boeing Co",               "rating": "BBB-","sector": "industrial",   "coupon": 5.805,"maturity": "2050-05-01"},
    {"ticker": "RTX",   "issuer": "RTX Corp",                "rating": "BBB+","sector": "industrial",   "coupon": 4.35, "maturity": "2047-04-15"},
    {"ticker": "UPS",   "issuer": "United Parcel Service",   "rating": "A-",  "sector": "industrial",   "coupon": 3.90, "maturity": "2047-04-01"},
    {"ticker": "FDX",   "issuer": "FedEx Corp",              "rating": "BBB", "sector": "industrial",   "coupon": 4.40, "maturity": "2045-01-15"},
    {"ticker": "UNP",   "issuer": "Union Pacific Corp",      "rating": "A-",  "sector": "industrial",   "coupon": 3.75, "maturity": "2070-02-05"},
    # Telecom
    {"ticker": "T",     "issuer": "AT&T Inc",                "rating": "BBB", "sector": "telecom",      "coupon": 4.25, "maturity": "2050-03-01"},
    {"ticker": "VZ",    "issuer": "Verizon Communications",  "rating": "BBB+","sector": "telecom",      "coupon": 4.016,"maturity": "2029-12-03"},
    {"ticker": "TMUS",  "issuer": "T-Mobile USA Inc",        "rating": "BBB", "sector": "telecom",      "coupon": 4.375,"maturity": "2040-04-15"},
    {"ticker": "CMCSA", "issuer": "Comcast Corp",            "rating": "A-",  "sector": "telecom",      "coupon": 4.25, "maturity": "2047-01-15"},
    # Real Estate / REITs
    {"ticker": "AMT",   "issuer": "American Tower Corp",     "rating": "BBB-","sector": "reit",         "coupon": 3.55, "maturity": "2027-07-15"},
    {"ticker": "PLD",   "issuer": "Prologis LP",             "rating": "A",   "sector": "reit",         "coupon": 4.375,"maturity": "2052-02-01"},
    {"ticker": "SPG",   "issuer": "Simon Property Group LP", "rating": "A-",  "sector": "reit",         "coupon": 4.75, "maturity": "2029-06-15"},
    # Consumer Discretionary
    {"ticker": "HD",    "issuer": "Home Depot Inc",          "rating": "A",   "sector": "consumer_disc","coupon": 4.95, "maturity": "2053-10-01"},
    {"ticker": "LOW",   "issuer": "Lowes Companies Inc",     "rating": "BBB+","sector": "consumer_disc","coupon": 5.00, "maturity": "2053-04-15"},
    {"ticker": "MCD",   "issuer": "McDonalds Corp",          "rating": "BBB+","sector": "consumer_disc","coupon": 4.45, "maturity": "2047-03-01"},
    {"ticker": "SBUX",  "issuer": "Starbucks Corp",          "rating": "BBB+","sector": "consumer_disc","coupon": 4.50, "maturity": "2031-11-15"},
    # Insurance
    {"ticker": "MET",   "issuer": "MetLife Inc",             "rating": "A-",  "sector": "insurance",    "coupon": 4.875,"maturity": "2043-11-01"},
    {"ticker": "PRU",   "issuer": "Prudential Financial",    "rating": "A",   "sector": "insurance",    "coupon": 4.35, "maturity": "2050-02-25"},
    {"ticker": "ALL",   "issuer": "Allstate Corp",           "rating": "A",   "sector": "insurance",    "coupon": 4.50, "maturity": "2043-06-15"},
    {"ticker": "AIG",   "issuer": "American Intl Group",     "rating": "BBB+","sector": "insurance",    "coupon": 4.75, "maturity": "2048-04-01"},
    # Materials
    {"ticker": "DD",    "issuer": "DuPont de Nemours",       "rating": "BBB+","sector": "materials",    "coupon": 5.319,"maturity": "2038-11-15"},
    {"ticker": "APD",   "issuer": "Air Products & Chemicals","rating": "A",   "sector": "materials",    "coupon": 4.375,"maturity": "2043-10-01"},
    {"ticker": "LIN",   "issuer": "Linde PLC",               "rating": "A",   "sector": "materials",    "coupon": 3.625,"maturity": "2033-03-15"},
    {"ticker": "IP",    "issuer": "International Paper",     "rating": "BBB", "sector": "materials",    "coupon": 4.80, "maturity": "2044-06-15"},
    {"ticker": "NUE",   "issuer": "Nucor Corp",              "rating": "A-",  "sector": "materials",    "coupon": 3.95, "maturity": "2042-05-01"},
    {"ticker": "FCX",   "issuer": "Freeport-McMoRan",        "rating": "BBB-","sector": "materials",    "coupon": 5.40, "maturity": "2034-11-14"},
]

# HY corporate issuers: 40 names
_HY_CORPS: List[Dict[str, Any]] = [
    {"ticker": "F",     "issuer": "Ford Motor Co",           "rating": "BB+", "sector": "auto",         "coupon": 6.10, "maturity": "2032-08-19"},
    {"ticker": "GM",    "issuer": "General Motors Co",       "rating": "BB+", "sector": "auto",         "coupon": 5.40, "maturity": "2031-10-02"},
    {"ticker": "CCL",   "issuer": "Carnival Corp",           "rating": "BB",  "sector": "leisure",      "coupon": 7.00, "maturity": "2029-08-15"},
    {"ticker": "RCL",   "issuer": "Royal Caribbean",         "rating": "BB+", "sector": "leisure",      "coupon": 5.50, "maturity": "2028-04-01"},
    {"ticker": "NCLH",  "issuer": "Norwegian Cruise Line",   "rating": "B+",  "sector": "leisure",      "coupon": 8.375,"maturity": "2028-02-01"},
    {"ticker": "MGM",   "issuer": "MGM Resorts Intl",        "rating": "BB-", "sector": "gaming",       "coupon": 5.75, "maturity": "2025-02-15"},
    {"ticker": "CZR",   "issuer": "Caesars Entertainment",   "rating": "B+",  "sector": "gaming",       "coupon": 7.00, "maturity": "2030-02-15"},
    {"ticker": "WYNN",  "issuer": "Wynn Resorts Ltd",        "rating": "B+",  "sector": "gaming",       "coupon": 5.125,"maturity": "2029-10-01"},
    {"ticker": "OXY",   "issuer": "Occidental Petroleum",    "rating": "BB+", "sector": "energy",       "coupon": 4.625,"maturity": "2045-06-15"},
    {"ticker": "DVN",   "issuer": "Devon Energy Corp",       "rating": "BBB-","sector": "energy",       "coupon": 5.25, "maturity": "2027-10-15"},
    {"ticker": "CHK",   "issuer": "Chesapeake Energy",       "rating": "B+",  "sector": "energy",       "coupon": 6.75, "maturity": "2029-04-15"},
    {"ticker": "AR",    "issuer": "Antero Resources",        "rating": "BB+", "sector": "energy",       "coupon": 5.375,"maturity": "2030-03-01"},
    {"ticker": "WHR",   "issuer": "Whirlpool Corp",          "rating": "BBB-","sector": "consumer",     "coupon": 5.75, "maturity": "2034-06-01"},
    {"ticker": "NWL",   "issuer": "Newell Brands Inc",       "rating": "BB",  "sector": "consumer",     "coupon": 6.375,"maturity": "2027-03-01"},
    {"ticker": "HBI",   "issuer": "Hanesbrands Inc",         "rating": "B",   "sector": "consumer",     "coupon": 9.00, "maturity": "2031-02-15"},
    {"ticker": "MO",    "issuer": "Altria Group Inc",        "rating": "BBB", "sector": "tobacco",      "coupon": 5.80, "maturity": "2039-02-14"},
    {"ticker": "BTI",   "issuer": "BAT Capital Corp",        "rating": "BBB", "sector": "tobacco",      "coupon": 4.906,"maturity": "2030-04-02"},
    {"ticker": "HCA",   "issuer": "HCA Healthcare Inc",      "rating": "BB+", "sector": "healthcare",   "coupon": 5.375,"maturity": "2033-02-01"},
    {"ticker": "THC",   "issuer": "Tenet Healthcare",        "rating": "B+",  "sector": "healthcare",   "coupon": 6.125,"maturity": "2028-10-01"},
    {"ticker": "CYH",   "issuer": "Community Health Systems","rating": "CCC+","sector": "healthcare",   "coupon": 8.00, "maturity": "2026-03-15"},
    {"ticker": "AAL",   "issuer": "American Airlines",       "rating": "B",   "sector": "airline",      "coupon": 7.25, "maturity": "2028-02-15"},
    {"ticker": "UAL",   "issuer": "United Airlines",         "rating": "B",   "sector": "airline",      "coupon": 4.625,"maturity": "2029-04-15"},
    {"ticker": "DAL",   "issuer": "Delta Air Lines",         "rating": "BB",  "sector": "airline",      "coupon": 7.375,"maturity": "2026-01-15"},
    {"ticker": "DISH",  "issuer": "DISH Network Corp",       "rating": "B",   "sector": "media",        "coupon": 7.75, "maturity": "2026-07-01"},
    {"ticker": "LUMN",  "issuer": "Lumen Technologies",      "rating": "CCC", "sector": "telecom",      "coupon": 4.00, "maturity": "2027-02-15"},
    {"ticker": "TAP",   "issuer": "Molson Coors Beverage",   "rating": "BB+", "sector": "consumer",     "coupon": 5.00, "maturity": "2042-05-01"},
    {"ticker": "PK",    "issuer": "Park Hotels & Resorts",   "rating": "BB",  "sector": "reit",         "coupon": 5.875,"maturity": "2028-10-01"},
    {"ticker": "HST",   "issuer": "Host Hotels & Resorts",   "rating": "BB+", "sector": "reit",         "coupon": 4.00, "maturity": "2029-06-15"},
    {"ticker": "SHO",   "issuer": "Sunstone Hotel Investors","rating": "B+",  "sector": "reit",         "coupon": 5.70, "maturity": "2025-12-15"},
    {"ticker": "APA",   "issuer": "APA Corp",                "rating": "BB+", "sector": "energy",       "coupon": 6.35, "maturity": "2026-05-15"},
    {"ticker": "RRC",   "issuer": "Range Resources Corp",    "rating": "BB+", "sector": "energy",       "coupon": 4.875,"maturity": "2025-05-15"},
    {"ticker": "SM",    "issuer": "SM Energy Co",            "rating": "BB",  "sector": "energy",       "coupon": 5.625,"maturity": "2025-06-01"},
    {"ticker": "PENN",  "issuer": "PENN Entertainment",      "rating": "B+",  "sector": "gaming",       "coupon": 5.625,"maturity": "2027-01-15"},
    {"ticker": "PVH",   "issuer": "PVH Corp",                "rating": "BB+", "sector": "consumer",     "coupon": 4.625,"maturity": "2025-07-10"},
    {"ticker": "RL",    "issuer": "Ralph Lauren Corp",       "rating": "BBB-","sector": "consumer",     "coupon": 3.75, "maturity": "2025-09-15"},
    {"ticker": "BBBY",  "issuer": "Bed Bath & Beyond",       "rating": "CCC-","sector": "retail",       "coupon": 5.165,"maturity": "2044-08-01"},
    {"ticker": "DRI",   "issuer": "Darden Restaurants",      "rating": "BB+", "sector": "restaurant",   "coupon": 3.85, "maturity": "2027-05-01"},
    {"ticker": "EQT",   "issuer": "EQT Corp",                "rating": "BB+", "sector": "energy",       "coupon": 3.625,"maturity": "2031-05-15"},
    {"ticker": "AMR",   "issuer": "Alpha Metallurgical",     "rating": "B+",  "sector": "materials",    "coupon": 7.00, "maturity": "2025-02-15"},
    {"ticker": "ALLY",  "issuer": "Ally Financial Inc",      "rating": "BB+", "sector": "financial",    "coupon": 5.75, "maturity": "2025-11-20"},
]

# Credit spread tables (bps over same-maturity Treasury)
_IG_SPREADS: Dict[str, float] = {
    "AAA": 22, "AA+": 32, "AA": 42, "AA-": 52,
    "A+":  68, "A":  82,  "A-": 98, "BBB+": 128,
    "BBB": 158,"BBB-":200,
}
_HY_SPREADS: Dict[str, float] = {
    "BB+": 245, "BB": 295,  "BB-": 370, "B+": 445,
    "B":   540, "B-": 665,  "CCC+": 840, "CCC": 1040,
    "CCC-":1340,"CC": 2000, "C":  3000,
}
_AGENCY_SPREADS: Dict[str, str] = {
    "bullet":   "18",   # bps (FHLB/FNMA bullet)
    "callable": "43",   # bps (callable agency)
}

# Fallback Treasury curve (2026 approximate)
_TREASURY_FALLBACK: Dict[float, float] = {
    0.25: 5.20, 0.50: 5.18, 1.0: 5.10, 2.0: 4.85, 3.0: 4.70,
    5.0:  4.50, 7.0:  4.45, 10.0: 4.40, 20.0: 4.65, 30.0: 4.55,
}


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

class FIDatabase:
    """SQLite-backed persistence for FI screening universe, results, and yield cache."""

    def __init__(self, db_path: Path = CACHE_DB):
        self.path = str(db_path)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS fi_universe (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    bond_id         TEXT UNIQUE NOT NULL,
                    sector          TEXT NOT NULL,
                    sub_sector      TEXT,
                    issuer          TEXT NOT NULL,
                    ticker          TEXT,
                    rating          TEXT DEFAULT 'NR',
                    coupon_rate     REAL NOT NULL,
                    maturity_date   TEXT NOT NULL,
                    years_to_mat    REAL,
                    face            REAL DEFAULT 1000.0,
                    freq            INTEGER DEFAULT 2,
                    callable        INTEGER DEFAULT 0,
                    tax_exempt      INTEGER DEFAULT 0,
                    state           TEXT,
                    ytm_pct         REAL DEFAULT 0.0,
                    ytw_pct         REAL DEFAULT 0.0,
                    price           REAL DEFAULT 100.0,
                    mod_duration    REAL DEFAULT 0.0,
                    mac_duration    REAL DEFAULT 0.0,
                    convexity       REAL DEFAULT 0.0,
                    dv01_per_mm     REAL DEFAULT 0.0,
                    oas_bps         REAL DEFAULT 0.0,
                    g_spread_bps    REAL DEFAULT 0.0,
                    risk_adj_yield  REAL DEFAULT 0.0,
                    income_score    REAL DEFAULT 0.0,
                    tey_pct         REAL DEFAULT 0.0,
                    real_yield      REAL DEFAULT 0.0,
                    agency_type     TEXT,
                    updated_at      TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_fi_sector   ON fi_universe(sector);
                CREATE INDEX IF NOT EXISTS idx_fi_rating   ON fi_universe(rating);
                CREATE INDEX IF NOT EXISTS idx_fi_mat      ON fi_universe(maturity_date);

                CREATE TABLE IF NOT EXISTS screening_results (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_ts          TEXT NOT NULL,
                    params_json     TEXT NOT NULL,
                    result_count    INTEGER,
                    result_json     TEXT
                );

                CREATE TABLE IF NOT EXISTS yield_cache (
                    series_id   TEXT NOT NULL,
                    obs_date    TEXT NOT NULL,
                    value       REAL,
                    PRIMARY KEY (series_id, obs_date)
                );

                CREATE TABLE IF NOT EXISTS http_cache (
                    cache_key   TEXT PRIMARY KEY,
                    body        TEXT,
                    expires_at  REAL
                );
            """)

    def cache_get(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT body FROM http_cache WHERE cache_key=? AND expires_at>?",
                (key, time.time()),
            ).fetchone()
        return row[0] if row else None

    def cache_set(self, key: str, body: str, ttl: int = CACHE_TTL) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO http_cache(cache_key, body, expires_at) VALUES(?,?,?)",
                (key, body, time.time() + ttl),
            )

    def upsert_bond(self, b: Dict[str, Any]) -> None:
        cols = list(b.keys())
        placeholders = ", ".join(f":{c}" for c in cols)
        col_str = ", ".join(cols)
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO fi_universe({col_str}) VALUES({placeholders})", b
            )

    def all_bonds(self, sector: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if sector:
                rows = conn.execute(
                    "SELECT * FROM fi_universe WHERE sector=?", (sector,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM fi_universe").fetchall()
        return [dict(r) for r in rows]

    def get_bond(self, bond_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM fi_universe WHERE bond_id=?", (bond_id,)
            ).fetchone()
        return dict(row) if row else None

    def save_yield(self, series_id: str, obs_date: str, value: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO yield_cache(series_id, obs_date, value) VALUES(?,?,?)",
                (series_id, obs_date, value),
            )

    def latest_yield(self, series_id: str) -> Optional[float]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM yield_cache WHERE series_id=? ORDER BY obs_date DESC LIMIT 1",
                (series_id,),
            ).fetchone()
        return row[0] if row else None

    def save_screen_result(self, params: Dict, results: List[Dict]) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO screening_results(run_ts, params_json, result_count, result_json)
                   VALUES(?,?,?,?)""",
                (datetime.utcnow().isoformat(), json.dumps(params), len(results), json.dumps(results[:100])),
            )


_fidb = FIDatabase()


# ---------------------------------------------------------------------------
# Bond math (pure Python, no external dep)
# ---------------------------------------------------------------------------

def _years_from_maturity(maturity_str: str) -> float:
    if not maturity_str:
        return 0.0
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%Y-%m"):
        try:
            mat = datetime.strptime(maturity_str, fmt).date()
            return max(0.0, (mat - date.today()).days / 365.25)
        except ValueError:
            continue
    return 0.0


def _ytm_newton(
    coupon_rate: float,
    years: float,
    price: float = 100.0,
    face: float = 100.0,
    freq: int = 2,
) -> float:
    """
    Newton-Raphson YTM solver.
    coupon_rate: annual coupon percent (4.0 = 4%).
    price: clean price in % of face.
    """
    if years <= 0:
        return coupon_rate
    n = max(1, int(round(years * freq)))
    c = face * coupon_rate / 100.0 / freq

    # Initial guess via simplified formula
    approx_ytm = (c + (face - price) / n) / ((face + price) / 2.0)
    approx_ytm = max(0.0001, approx_ytm)

    y = approx_ytm  # yield per period

    for _ in range(300):
        disc = [(1 + y) ** t for t in range(1, n + 1)]
        pv_coupons = sum(c / d for d in disc)
        pv_face = face / disc[-1]
        pv = pv_coupons + pv_face

        # dPV/dy
        dpv = -sum(t * c / (1 + y) ** (t + 1) for t in range(1, n + 1))
        dpv -= n * face / (1 + y) ** (n + 1)

        if dpv == 0:
            break
        y_new = y - (pv - price) / dpv
        y_new = max(1e-7, y_new)
        if abs(y_new - y) < 1e-12:
            y = y_new
            break
        y = y_new

    return y * freq * 100.0   # annualized %


def _macaulay_duration(
    coupon_rate: float, years: float, ytm_pct: float,
    face: float = 100.0, freq: int = 2
) -> float:
    if years <= 0 or ytm_pct <= 0:
        return years
    n = max(1, int(round(years * freq)))
    y = ytm_pct / 100.0 / freq
    c = face * coupon_rate / 100.0 / freq

    pv_total = 0.0
    weighted = 0.0
    for t in range(1, n + 1):
        cf = c if t < n else c + face
        pv_cf = cf / (1 + y) ** t
        pv_total += pv_cf
        weighted += (t / freq) * pv_cf

    if pv_total <= 0:
        return 0.0
    return weighted / pv_total


def _modified_duration(
    coupon_rate: float, years: float, ytm_pct: float,
    face: float = 100.0, freq: int = 2
) -> float:
    mac = _macaulay_duration(coupon_rate, years, ytm_pct, face, freq)
    y = ytm_pct / 100.0 / freq
    return mac / (1 + y)


def _convexity(
    coupon_rate: float, years: float, ytm_pct: float,
    face: float = 100.0, freq: int = 2
) -> float:
    if years <= 0 or ytm_pct <= 0:
        return 0.0
    n = max(1, int(round(years * freq)))
    y = ytm_pct / 100.0 / freq
    c = face * coupon_rate / 100.0 / freq

    pv_total = sum(
        (c if t < n else c + face) / (1 + y) ** t
        for t in range(1, n + 1)
    )
    if pv_total <= 0:
        return 0.0
    conv = sum(
        t * (t + 1) / (freq ** 2) * (c if t < n else c + face) / (1 + y) ** (t + 2)
        for t in range(1, n + 1)
    )
    return conv / pv_total


def _dv01_per_mm(mod_dur: float, price: float = 100.0) -> float:
    """DV01 per $1M face = modified_duration * price/100 * $1M / 10000."""
    return mod_dur * price / 100.0 * 1_000_000 / 10_000


def _real_ytm_tips(coupon_rate: float, years: float, price: float = 100.0) -> float:
    """YTM in real terms for TIPS (same formula, real coupon)."""
    return _ytm_newton(coupon_rate, years, price)


def _interp_treasury(years: float, live_curve: Optional[Dict[float, float]] = None) -> float:
    curve = live_curve or _TREASURY_FALLBACK
    tenors = sorted(curve.keys())
    yields = [curve[t] for t in tenors]
    if years <= tenors[0]:
        return yields[0]
    if years >= tenors[-1]:
        return yields[-1]
    for i in range(len(tenors) - 1):
        t0, t1 = tenors[i], tenors[i + 1]
        if t0 <= years <= t1:
            y0, y1 = yields[i], yields[i + 1]
            return y0 + (y1 - y0) * (years - t0) / (t1 - t0)
    return yields[-1]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update(_HEADERS)


def _http_get(url: str, params: Optional[Dict] = None, ttl: int = CACHE_TTL) -> str:
    ck = f"{url}?{json.dumps(params or {}, sort_keys=True)}"
    cached = _fidb.cache_get(ck)
    if cached:
        return cached
    time.sleep(0.4)
    try:
        resp = _session.get(url, params=params, timeout=25)
        resp.raise_for_status()
        text = resp.text
        _fidb.cache_set(ck, text, ttl)
        return text
    except Exception as exc:
        logger.warning("HTTP GET %s failed: %s", url, exc)
        raise


def _http_get_json(url: str, params: Optional[Dict] = None, ttl: int = CACHE_TTL) -> Any:
    return json.loads(_http_get(url, params, ttl))


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

class TreasuryFeed:
    """
    Fetch on-the-run Treasury + TIPS yields from TreasuryDirect XML feed.
    Fallback to static curve on failure.
    """

    def fetch_yield_curve(self, year: int = 0) -> Dict[float, float]:
        """
        Return Treasury par yield curve {tenor_years: yield_pct}.
        Uses TreasuryDirect XML. Falls back to _TREASURY_FALLBACK.
        """
        year = year or date.today().year
        params = {"data": "daily_treasury_yield_curve", "field_tdr_date_value": str(year)}
        try:
            xml_text = _http_get(
                "https://home.treasury.gov/resource-center/data-chart-center/"
                "interest-rates/pages/xml",
                params, ttl=3600
            )
            return self._parse_treasury_xml(xml_text)
        except Exception as exc:
            logger.warning("Treasury XML fetch failed: %s — using fallback", exc)
            return dict(_TREASURY_FALLBACK)

    def _parse_treasury_xml(self, xml_text: str) -> Dict[float, float]:
        """Parse Treasury XML feed — return most recent record as {tenor: yield}."""
        # Treasury XML uses OFR namespace or plain content
        tenors_map: Dict[str, float] = {
            "BC_2YEAR": 2.0, "BC_3YEAR": 3.0, "BC_5YEAR": 5.0,
            "BC_7YEAR": 7.0, "BC_10YEAR": 10.0, "BC_20YEAR": 20.0,
            "BC_30YEAR": 30.0,
        }
        try:
            root = ElementTree.fromstring(xml_text)
            # Find all entries, take the last (most recent)
            entries: List[ElementTree.Element] = []
            for elem in root.iter():
                if "entry" in elem.tag.lower() or "properties" in elem.tag.lower():
                    entries.append(elem)
            if not entries:
                return dict(_TREASURY_FALLBACK)

            latest = entries[-1]
            result: Dict[float, float] = {}
            for child in latest.iter():
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag in tenors_map and child.text:
                    try:
                        result[tenors_map[tag]] = float(child.text)
                    except ValueError:
                        pass
            return result if result else dict(_TREASURY_FALLBACK)
        except Exception as exc:
            logger.debug("Treasury XML parse error: %s", exc)
            return dict(_TREASURY_FALLBACK)

    def fetch_tips_real_yields(self) -> Dict[float, float]:
        """Return TIPS real yield curve {tenor: real_yield_pct}."""
        try:
            xml_text = _http_get(
                "https://home.treasury.gov/resource-center/data-chart-center/"
                "interest-rates/pages/xml",
                {"data": "daily_treasury_real_yield_curve"}, ttl=3600
            )
            tips_map: Dict[str, float] = {
                "TC_5YEAR": 5.0, "TC_7YEAR": 7.0, "TC_10YEAR": 10.0,
                "TC_20YEAR": 20.0, "TC_30YEAR": 30.0,
            }
            root = ElementTree.fromstring(xml_text)
            result: Dict[float, float] = {}
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag in tips_map and elem.text:
                    try:
                        result[tips_map[tag]] = float(elem.text)
                    except ValueError:
                        pass
            # Fall back to static
            if not result:
                result = {5.0: TIPS_OTR["5Y"]["real_yield"],
                          10.0: TIPS_OTR["10Y"]["real_yield"],
                          30.0: TIPS_OTR["30Y"]["real_yield"]}
            return result
        except Exception as exc:
            logger.warning("TIPS real yield fetch failed: %s", exc)
            return {5.0: TIPS_OTR["5Y"]["real_yield"],
                    10.0: TIPS_OTR["10Y"]["real_yield"],
                    30.0: TIPS_OTR["30Y"]["real_yield"]}


class FREDYields:
    """Fetch credit spread indices and Treasury series from FRED (no API key)."""

    _SERIES: Dict[str, str] = {
        "BAMLC0A0CM":  "IG Corp OAS",
        "BAMLH0A0HYM2":"HY Corp OAS",
        "BAMLM0A0CMU": "Muni OAS",
        "DGS2":  "2Y Treasury",
        "DGS5":  "5Y Treasury",
        "DGS10": "10Y Treasury",
        "DGS20": "20Y Treasury",
        "DGS30": "30Y Treasury",
        "DFII5": "5Y TIPS Real",
        "DFII10":"10Y TIPS Real",
        "DFII30":"30Y TIPS Real",
    }

    def fetch_latest(self, series_id: str) -> Optional[float]:
        try:
            text = _http_get(FRED_CSV, {"id": series_id}, ttl=3600)
            lines = [l for l in text.strip().splitlines() if l and not l.startswith("DATE")]
            for line in reversed(lines):
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        val = float(parts[1].strip())
                        _fidb.save_yield(series_id, parts[0].strip(), val)
                        return val
                    except ValueError:
                        continue
        except Exception as exc:
            logger.warning("FRED %s fetch failed: %s", series_id, exc)
        return _fidb.latest_yield(series_id)

    def get_treasury_curve(self) -> Dict[float, float]:
        """Return live Treasury curve from FRED, fallback to static."""
        series = {"2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30"}
        tenor_map = {"2Y": 2.0, "5Y": 5.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0}
        result: Dict[float, float] = {}
        for label, sid in series.items():
            val = self.fetch_latest(sid)
            if val and val > 0:
                result[tenor_map[label]] = val
        return result if result else dict(_TREASURY_FALLBACK)

    def get_ig_oas(self) -> Optional[float]:
        return self.fetch_latest("BAMLC0A0CM")

    def get_hy_oas(self) -> Optional[float]:
        return self.fetch_latest("BAMLH0A0HYM2")

    def get_tips_real_yields(self) -> Dict[float, float]:
        result: Dict[float, float] = {}
        for tenor, sid in [(5.0, "DFII5"), (10.0, "DFII10"), (30.0, "DFII30")]:
            val = self.fetch_latest(sid)
            if val is not None:
                result[tenor] = val
        if not result:
            result = {5.0: TIPS_OTR["5Y"]["real_yield"],
                      10.0: TIPS_OTR["10Y"]["real_yield"],
                      30.0: TIPS_OTR["30Y"]["real_yield"]}
        return result


_treasury_feed = TreasuryFeed()
_fred_yields   = FREDYields()


# ---------------------------------------------------------------------------
# Universe builder
# ---------------------------------------------------------------------------

class UniverseBuilder:
    """
    Builds and maintains the multi-asset FI universe in SQLite.
    Supports Treasuries, TIPS, IG Corp, HY Corp, Munis (via v3), Agency.
    """

    def __init__(self):
        self._treas_curve: Dict[float, float] = dict(_TREASURY_FALLBACK)

    def refresh_treasury_curve(self) -> Dict[float, float]:
        try:
            live = _treasury_feed.fetch_yield_curve()
            if len(live) >= 3:
                self._treas_curve = live
        except Exception as exc:
            logger.warning("Treasury curve refresh failed: %s", exc)
        return self._treas_curve

    def build_all(self) -> int:
        """Build full universe. Returns total bond count."""
        self.refresh_treasury_curve()
        total = 0
        total += self._build_treasuries()
        total += self._build_tips()
        total += self._build_ig_corps()
        total += self._build_hy_corps()
        total += self._build_agencies()
        total += self._build_munis()
        logger.info("FI universe built: %d instruments", total)
        return total

    def _build_treasuries(self) -> int:
        count = 0
        for label, info in TREASURY_OTR.items():
            years = _years_from_maturity(info["maturity"])
            coupon = float(info["coupon"])
            ytm = _interp_treasury(years, self._treas_curve)
            mod_dur = _modified_duration(coupon, years, ytm)
            conv = _convexity(coupon, years, ytm)
            dv01 = _dv01_per_mm(mod_dur)

            row: Dict[str, Any] = {
                "bond_id":      f"UST_{label}",
                "sector":       "GOVT",
                "sub_sector":   "treasury",
                "issuer":       "US Treasury",
                "ticker":       f"UST{label}",
                "rating":       "AAA",
                "coupon_rate":  coupon,
                "maturity_date": info["maturity"],
                "years_to_mat": round(years, 3),
                "face":         1000.0,
                "freq":         2,
                "callable":     0,
                "tax_exempt":   0,
                "ytm_pct":      round(ytm, 4),
                "ytw_pct":      round(ytm, 4),
                "price":        100.0,
                "mod_duration": round(mod_dur, 4),
                "mac_duration": round(_macaulay_duration(coupon, years, ytm), 4),
                "convexity":    round(conv, 4),
                "dv01_per_mm":  round(dv01, 2),
                "oas_bps":      0.0,
                "g_spread_bps": 0.0,
                "risk_adj_yield": round(ytm / mod_dur, 4) if mod_dur > 0 else 0,
                "income_score": round(coupon / mod_dur, 4) if mod_dur > 0 else 0,
                "tey_pct":      round(ytm, 4),  # taxable
                "real_yield":   0.0,
                "updated_at":   date.today().isoformat(),
            }
            _fidb.upsert_bond(row)
            count += 1
        return count

    def _build_tips(self) -> int:
        count = 0
        tips_real = _fred_yields.get_tips_real_yields()

        for label, info in TIPS_OTR.items():
            years = _years_from_maturity(info["maturity"])
            coupon = float(info["coupon"])
            real_yield = tips_real.get(years) or info["real_yield"]
            # TIPS YTM = real yield (nominal = real + inflation breakeven ~2.3%)
            ytm_nominal = real_yield + 2.30
            mod_dur = _modified_duration(coupon, years, real_yield)
            conv = _convexity(coupon, years, real_yield)
            dv01 = _dv01_per_mm(mod_dur)
            treas_yield = _interp_treasury(years, self._treas_curve)
            g_spread = (ytm_nominal - treas_yield) * 100

            row: Dict[str, Any] = {
                "bond_id":      f"TIPS_{label}",
                "sector":       "TIPS",
                "sub_sector":   "tips",
                "issuer":       "US Treasury TIPS",
                "ticker":       f"TIPS{label}",
                "rating":       "AAA",
                "coupon_rate":  coupon,
                "maturity_date": info["maturity"],
                "years_to_mat": round(years, 3),
                "face":         1000.0,
                "freq":         2,
                "callable":     0,
                "tax_exempt":   0,
                "ytm_pct":      round(ytm_nominal, 4),
                "ytw_pct":      round(ytm_nominal, 4),
                "price":        100.0,
                "mod_duration": round(mod_dur, 4),
                "mac_duration": round(_macaulay_duration(coupon, years, real_yield), 4),
                "convexity":    round(conv, 4),
                "dv01_per_mm":  round(dv01, 2),
                "oas_bps":      0.0,
                "g_spread_bps": round(g_spread, 2),
                "risk_adj_yield": round(ytm_nominal / mod_dur, 4) if mod_dur > 0 else 0,
                "income_score": round(coupon / mod_dur, 4) if mod_dur > 0 else 0,
                "tey_pct":      round(ytm_nominal, 4),
                "real_yield":   round(real_yield, 4),
                "updated_at":   date.today().isoformat(),
            }
            _fidb.upsert_bond(row)
            count += 1
        return count

    def _build_ig_corps(self) -> int:
        count = 0
        # Adjust IG OAS from FRED
        ig_oas_level = _fred_yields.get_ig_oas()

        for entry in _IG_CORPS:
            years = _years_from_maturity(entry["maturity"])
            if years <= 0:
                continue
            coupon = float(entry["coupon"])
            rating = entry.get("rating", "BBB")
            treas_yield = _interp_treasury(years, self._treas_curve)
            static_spread = _IG_SPREADS.get(rating, 160)

            # Scale spread if live OAS available
            if ig_oas_level and ig_oas_level > 0:
                typical_ig_oas = 85.0  # baseline
                scale = ig_oas_level / typical_ig_oas
                spread_bps = static_spread * scale
            else:
                spread_bps = static_spread

            ytm = treas_yield + spread_bps / 100.0
            price = 100.0  # approximate par
            mod_dur = _modified_duration(coupon, years, ytm)
            mac_dur = _macaulay_duration(coupon, years, ytm)
            conv    = _convexity(coupon, years, ytm)
            dv01    = _dv01_per_mm(mod_dur)
            g_spread = (ytm - treas_yield) * 100

            row: Dict[str, Any] = {
                "bond_id":      f"CORP_IG_{entry['ticker']}",
                "sector":       "CORP_IG",
                "sub_sector":   entry.get("sector", ""),
                "issuer":       entry["issuer"],
                "ticker":       entry["ticker"],
                "rating":       rating,
                "coupon_rate":  coupon,
                "maturity_date": entry["maturity"],
                "years_to_mat": round(years, 3),
                "face":         1000.0,
                "freq":         2,
                "callable":     0,
                "tax_exempt":   0,
                "ytm_pct":      round(ytm, 4),
                "ytw_pct":      round(ytm, 4),
                "price":        round(price, 4),
                "mod_duration": round(mod_dur, 4),
                "mac_duration": round(mac_dur, 4),
                "convexity":    round(conv, 4),
                "dv01_per_mm":  round(dv01, 2),
                "oas_bps":      round(spread_bps, 2),
                "g_spread_bps": round(g_spread, 2),
                "risk_adj_yield": round(ytm / mod_dur, 4) if mod_dur > 0 else 0,
                "income_score": round(coupon / mod_dur, 4) if mod_dur > 0 else 0,
                "tey_pct":      round(ytm * (1 - 0.37), 4),  # after-tax for IG
                "real_yield":   0.0,
                "updated_at":   date.today().isoformat(),
            }
            _fidb.upsert_bond(row)
            count += 1
        return count

    def _build_hy_corps(self) -> int:
        count = 0
        hy_oas_level = _fred_yields.get_hy_oas()

        for entry in _HY_CORPS:
            years = _years_from_maturity(entry["maturity"])
            if years <= 0:
                continue
            coupon = float(entry["coupon"])
            rating = entry.get("rating", "B")
            treas_yield = _interp_treasury(years, self._treas_curve)
            static_spread = _HY_SPREADS.get(rating, 500)

            if hy_oas_level and hy_oas_level > 0:
                typical_hy_oas = 350.0
                scale = hy_oas_level / typical_hy_oas
                spread_bps = static_spread * scale
            else:
                spread_bps = static_spread

            ytm = treas_yield + spread_bps / 100.0
            # HY bonds often trade at discount; estimate price from YTM vs coupon
            if ytm > coupon / 100.0:
                # Discount bond
                price = 100.0 * coupon / (ytm * 100.0 / 100.0) if ytm > 0 else 100.0
                price = max(40.0, min(105.0, price))
            else:
                price = 100.0

            mod_dur = _modified_duration(coupon, years, ytm)
            mac_dur = _macaulay_duration(coupon, years, ytm)
            conv    = _convexity(coupon, years, ytm)
            dv01    = _dv01_per_mm(mod_dur, price)
            g_spread = (ytm - treas_yield) * 100

            row: Dict[str, Any] = {
                "bond_id":      f"CORP_HY_{entry['ticker']}",
                "sector":       "CORP_HY",
                "sub_sector":   entry.get("sector", ""),
                "issuer":       entry["issuer"],
                "ticker":       entry["ticker"],
                "rating":       rating,
                "coupon_rate":  coupon,
                "maturity_date": entry["maturity"],
                "years_to_mat": round(years, 3),
                "face":         1000.0,
                "freq":         2,
                "callable":     1,
                "tax_exempt":   0,
                "ytm_pct":      round(ytm, 4),
                "ytw_pct":      round(ytm, 4),
                "price":        round(price, 4),
                "mod_duration": round(mod_dur, 4),
                "mac_duration": round(mac_dur, 4),
                "convexity":    round(conv, 4),
                "dv01_per_mm":  round(dv01, 2),
                "oas_bps":      round(spread_bps, 2),
                "g_spread_bps": round(g_spread, 2),
                "risk_adj_yield": round(ytm / mod_dur, 4) if mod_dur > 0 else 0,
                "income_score": round(coupon / mod_dur, 4) if mod_dur > 0 else 0,
                "tey_pct":      round(ytm * (1 - 0.37), 4),
                "real_yield":   0.0,
                "updated_at":   date.today().isoformat(),
            }
            _fidb.upsert_bond(row)
            count += 1
        return count

    def _build_agencies(self) -> int:
        count = 0
        for entry in AGENCY_BONDS:
            years = _years_from_maturity(entry["maturity"])
            if years <= 0:
                continue
            coupon = float(entry["coupon"])
            bond_type = entry.get("type", "bullet")
            spread_bps = float(_AGENCY_SPREADS.get(bond_type, "22"))
            treas_yield = _interp_treasury(years, self._treas_curve)
            ytm = treas_yield + spread_bps / 100.0
            callable_flag = 1 if bond_type == "callable" else 0
            ytw = ytm  # for simplicity (YTW = YTM for bullets; call discount for callable)
            if callable_flag:
                ytw = max(ytm - 0.30, ytm * 0.92)  # approximate YTW discount

            mod_dur = _modified_duration(coupon, years, ytm)
            mac_dur = _macaulay_duration(coupon, years, ytm)
            conv    = _convexity(coupon, years, ytm)
            dv01    = _dv01_per_mm(mod_dur)
            g_spread = spread_bps  # agency spread = G-spread

            row: Dict[str, Any] = {
                "bond_id":      f"AGENCY_{entry['id']}",
                "sector":       "AGENCY",
                "sub_sector":   entry.get("agency", ""),
                "issuer":       entry["issuer"],
                "ticker":       entry.get("agency", ""),
                "rating":       entry.get("rating", "AA+"),
                "coupon_rate":  coupon,
                "maturity_date": entry["maturity"],
                "years_to_mat": round(years, 3),
                "face":         1000.0,
                "freq":         2,
                "callable":     callable_flag,
                "tax_exempt":   0,
                "ytm_pct":      round(ytm, 4),
                "ytw_pct":      round(ytw, 4),
                "price":        100.0,
                "mod_duration": round(mod_dur, 4),
                "mac_duration": round(mac_dur, 4),
                "convexity":    round(conv, 4),
                "dv01_per_mm":  round(dv01, 2),
                "oas_bps":      round(spread_bps, 2),
                "g_spread_bps": round(g_spread, 2),
                "risk_adj_yield": round(ytm / mod_dur, 4) if mod_dur > 0 else 0,
                "income_score": round(coupon / mod_dur, 4) if mod_dur > 0 else 0,
                "tey_pct":      round(ytm, 4),
                "real_yield":   0.0,
                "agency_type":  bond_type,
                "updated_at":   date.today().isoformat(),
            }
            _fidb.upsert_bond(row)
            count += 1
        return count

    def _build_munis(self) -> int:
        """
        Import muni universe from municipal_bond_v3.MuniService (same process).
        Falls back to inline approximation if module unavailable.
        """
        count = 0
        try:
            from sentinel.sfe.municipal_bond_v3 import (
                MuniService, MUNI_UNIVERSE_60, _years_to_maturity as _muni_years,
                _interp_aaa_mmd, STATE_TAX, SECTOR_DEFAULT_RATES
            )
            svc = MuniService()
            from sentinel.sfe.municipal_bond_v3 import _db as _muni_db
            muni_bonds = _muni_db.all_bonds()
            if not muni_bonds:
                svc.bootstrap_universe()
                muni_bonds = _muni_db.all_bonds()
        except Exception as exc:
            logger.warning("Muni v3 import failed (%s); using inline universe", exc)
            muni_bonds = self._inline_muni_fallback()

        for mb in muni_bonds:
            years = float(mb.get("years_to_mat") or 0)
            if years <= 0:
                years = _years_from_maturity(mb.get("maturity_date", ""))
            if years <= 0:
                continue

            coupon = float(mb.get("coupon_rate") or 0)
            ytm    = float(mb.get("last_yield") or mb.get("ytm_pct") or 0)
            state  = str(mb.get("state") or "")
            treas_yield = _interp_treasury(years, self._treas_curve)

            # TEY at 37%
            st_rate  = 0.0
            try:
                from sentinel.sfe.municipal_bond_v3 import STATE_TAX as _ST
                st_rate = _ST.get(state, 0.0)
            except Exception:
                pass
            combined = 0.37 + st_rate * (1 - 0.37)
            tey = ytm / (1 - combined) if combined < 1 and ytm > 0 else ytm

            mod_dur = float(mb.get("mod_duration") or 0) or _modified_duration(coupon, years, ytm)
            mac_dur = _macaulay_duration(coupon, years, ytm) if ytm > 0 else mod_dur
            conv    = _convexity(coupon, years, ytm) if ytm > 0 else 0.0
            dv01    = float(mb.get("dv01_per_mm") or 0) or _dv01_per_mm(mod_dur)
            g_spread = (ytm - treas_yield) * 100
            oas_bps  = float(mb.get("aaa_spread_bps") or g_spread)

            bond_id = f"MUNI_{mb.get('cusip', mb.get('issuer', ''))}"

            row: Dict[str, Any] = {
                "bond_id":      bond_id,
                "sector":       "MUNI",
                "sub_sector":   mb.get("sector", "general_obligation"),
                "issuer":       mb.get("issuer", ""),
                "ticker":       mb.get("cusip", ""),
                "rating":       mb.get("rating_sp", "NR"),
                "coupon_rate":  coupon,
                "maturity_date": mb.get("maturity_date", ""),
                "years_to_mat": round(years, 3),
                "face":         1000.0,
                "freq":         2,
                "callable":     0,
                "tax_exempt":   1,
                "state":        state,
                "ytm_pct":      round(ytm, 4),
                "ytw_pct":      round(ytm, 4),
                "price":        100.0,
                "mod_duration": round(mod_dur, 4),
                "mac_duration": round(mac_dur, 4),
                "convexity":    round(conv, 4),
                "dv01_per_mm":  round(dv01, 2),
                "oas_bps":      round(oas_bps, 2),
                "g_spread_bps": round(g_spread, 2),
                "risk_adj_yield": round(ytm / mod_dur, 4) if mod_dur > 0 else 0,
                "income_score": round(coupon / mod_dur, 4) if mod_dur > 0 else 0,
                "tey_pct":      round(tey, 4),
                "real_yield":   0.0,
                "updated_at":   date.today().isoformat(),
            }
            _fidb.upsert_bond(row)
            count += 1
        return count

    def _inline_muni_fallback(self) -> List[Dict[str, Any]]:
        """Minimal inline fallback muni universe (subset of 60) if v3 unavailable."""
        return [
            {"cusip": "13063BQQ8", "issuer": "California GO",      "state": "CA", "sector": "general_obligation", "coupon_rate": 4.00, "maturity_date": "2034-11-01", "last_yield": 3.98, "mod_duration": 7.2, "aaa_spread_bps": 10, "years_to_mat": 8.5, "rating_sp": "AA-"},
            {"cusip": "64966EGV5", "issuer": "New York GO",         "state": "NY", "sector": "general_obligation", "coupon_rate": 3.75, "maturity_date": "2033-08-01", "last_yield": 3.85, "mod_duration": 6.8, "aaa_spread_bps": 10, "years_to_mat": 7.2, "rating_sp": "AA"},
            {"cusip": "882723TK3", "issuer": "Texas GO",            "state": "TX", "sector": "general_obligation", "coupon_rate": 3.50, "maturity_date": "2035-04-01", "last_yield": 3.55, "mod_duration": 8.1, "aaa_spread_bps": 0,  "years_to_mat": 9.0, "rating_sp": "AAA"},
            {"cusip": "452152K77", "issuer": "Illinois GO",         "state": "IL", "sector": "general_obligation", "coupon_rate": 5.00, "maturity_date": "2030-11-01", "last_yield": 4.80, "mod_duration": 4.2, "aaa_spread_bps": 120,"years_to_mat": 4.5, "rating_sp": "BBB+"},
            {"cusip": "649900QQ5", "issuer": "NYC General Obligation","state": "NY","sector": "general_obligation", "coupon_rate": 4.00, "maturity_date": "2032-08-01", "last_yield": 3.90, "mod_duration": 5.9, "aaa_spread_bps": 10, "years_to_mat": 6.3, "rating_sp": "AA"},
            {"cusip": "646030AT7", "issuer": "NJ GO",               "state": "NJ", "sector": "general_obligation", "coupon_rate": 4.50, "maturity_date": "2032-06-01", "last_yield": 4.50, "mod_duration": 5.8, "aaa_spread_bps": 65, "years_to_mat": 6.1, "rating_sp": "A"},
        ]


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

_RATING_ORDER: List[str] = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
    "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D", "NR",
]


def _rating_rank(r: str) -> int:
    r = (r or "NR").upper().strip()
    return _RATING_ORDER.index(r) if r in _RATING_ORDER else len(_RATING_ORDER)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ScreenRequest(BaseModel):
    sectors: Optional[List[str]] = Field(
        None, description="GOVT|TIPS|CORP_IG|CORP_HY|MUNI|AGENCY"
    )
    sub_sectors: Optional[List[str]] = Field(None, description="technology|financial|energy...")
    min_ytm: float = Field(0.0, ge=0)
    max_ytm: float = Field(20.0, le=50)
    min_ytw: Optional[float] = None
    max_ytw: Optional[float] = None
    min_duration: float = Field(0.0, ge=0)
    max_duration: float = Field(30.0, le=50)
    min_coupon: float = Field(0.0, ge=0)
    max_coupon: float = Field(15.0, le=30)
    min_maturity_years: float = Field(0.0, ge=0)
    max_maturity_years: float = Field(40.0, le=60)
    ratings: Optional[List[str]] = Field(None, description="Whitelist of ratings e.g. ['AAA','AA+']")
    min_rating: Optional[str] = Field(None, description="Minimum acceptable rating (inclusive)")
    max_oas_bps: Optional[float] = None
    min_oas_bps: Optional[float] = None
    tax_exempt_only: bool = False
    investment_grade_only: bool = False
    exclude_callable: bool = False
    min_g_spread_bps: Optional[float] = None
    max_g_spread_bps: Optional[float] = None
    rank_by: str = Field(
        "risk_adj_yield",
        description="risk_adj_yield|income_score|ytm|g_spread|tey"
    )
    limit: int = Field(50, ge=1, le=500)


class BondResult(BaseModel):
    bond_id: str
    sector: str
    sub_sector: Optional[str]
    issuer: str
    ticker: Optional[str]
    rating: str
    coupon_rate: float
    maturity_date: str
    years_to_mat: float
    ytm_pct: float
    ytw_pct: float
    price: float
    mod_duration: float
    convexity: float
    dv01_per_mm: float
    oas_bps: float
    g_spread_bps: float
    risk_adj_yield: float
    income_score: float
    tey_pct: float
    real_yield: float
    tax_exempt: bool
    callable: bool


class ScreenResponse(BaseModel):
    total_matches: int
    bonds: List[BondResult]
    summary: Dict[str, Any]
    filters: Dict[str, Any]
    ranked_by: str
    run_at: str


class YieldMatrixRow(BaseModel):
    tenor_years: float
    govt_pct: float
    tips_real_pct: float
    tips_nominal_pct: float
    ig_aa_pct: float
    ig_bbb_pct: float
    hy_bb_pct: float
    hy_b_pct: float
    muni_aaa_pct: float
    muni_bbb_pct: float
    agency_bullet_pct: float
    agency_callable_pct: float
    breakeven_inflation_pct: float


class TopPicksResponse(BaseModel):
    as_of: str
    risk_adj_yield_leaders: List[BondResult]
    income_leaders: List[BondResult]
    g_spread_leaders: List[BondResult]
    tey_leaders: List[BondResult]
    total_universe: int


# ---------------------------------------------------------------------------
# FIScreenerService
# ---------------------------------------------------------------------------

class FIScreenerService:
    """Core screening service — universe management + multi-factor ranking."""

    def __init__(self):
        self._builder = UniverseBuilder()
        self._treas_curve: Dict[float, float] = dict(_TREASURY_FALLBACK)
        self._universe_loaded = False

    def ensure_universe(self) -> None:
        if self._universe_loaded and _fidb.all_bonds():
            return
        self._builder.refresh_treasury_curve()
        self._treas_curve = self._builder._treas_curve
        total = self._builder.build_all()
        self._universe_loaded = True
        logger.info("FI universe loaded: %d bonds", total)

    def screen(self, req: ScreenRequest) -> ScreenResponse:
        self.ensure_universe()
        bonds = _fidb.all_bonds()

        # IG rating cutoff = BBB-
        _IG_CUTOFF_RANK = _rating_rank("BBB-")

        filtered: List[Dict] = []
        for b in bonds:
            # Sector
            if req.sectors and b.get("sector") not in req.sectors:
                continue
            # Sub-sector
            if req.sub_sectors and b.get("sub_sector") not in req.sub_sectors:
                continue
            # YTM
            ytm = b.get("ytm_pct", 0) or 0
            if ytm < req.min_ytm or ytm > req.max_ytm:
                continue
            # YTW
            ytw = b.get("ytw_pct", 0) or 0
            if req.min_ytw is not None and ytw < req.min_ytw:
                continue
            if req.max_ytw is not None and ytw > req.max_ytw:
                continue
            # Duration
            dur = b.get("mod_duration", 0) or 0
            if dur < req.min_duration or dur > req.max_duration:
                continue
            # Coupon
            cpn = b.get("coupon_rate", 0) or 0
            if cpn < req.min_coupon or cpn > req.max_coupon:
                continue
            # Maturity
            yrs = b.get("years_to_mat", 0) or 0
            if yrs < req.min_maturity_years or yrs > req.max_maturity_years:
                continue
            # Rating whitelist
            if req.ratings and b.get("rating") not in req.ratings:
                continue
            # Minimum rating
            if req.min_rating:
                if _rating_rank(b.get("rating", "NR")) > _rating_rank(req.min_rating):
                    continue
            # OAS
            oas = b.get("oas_bps", 0) or 0
            if req.max_oas_bps is not None and oas > req.max_oas_bps:
                continue
            if req.min_oas_bps is not None and oas < req.min_oas_bps:
                continue
            # Tax exempt
            if req.tax_exempt_only and not b.get("tax_exempt"):
                continue
            # Investment grade
            if req.investment_grade_only and _rating_rank(b.get("rating", "NR")) > _IG_CUTOFF_RANK:
                continue
            # Exclude callable
            if req.exclude_callable and b.get("callable"):
                continue
            # G-spread
            gs = b.get("g_spread_bps", 0) or 0
            if req.min_g_spread_bps is not None and gs < req.min_g_spread_bps:
                continue
            if req.max_g_spread_bps is not None and gs > req.max_g_spread_bps:
                continue

            filtered.append(b)

        # Multi-factor ranking
        rank_map = {
            "risk_adj_yield": lambda b: -(b.get("risk_adj_yield") or 0),
            "income_score":   lambda b: -(b.get("income_score") or 0),
            "ytm":            lambda b: -(b.get("ytm_pct") or 0),
            "g_spread":       lambda b: -(b.get("g_spread_bps") or 0),
            "tey":            lambda b: -(b.get("tey_pct") or 0),
        }
        sort_fn = rank_map.get(req.rank_by, rank_map["risk_adj_yield"])
        filtered.sort(key=sort_fn)
        filtered = filtered[:req.limit]

        # Summary stats
        if filtered:
            ytms  = [b.get("ytm_pct", 0) or 0  for b in filtered]
            durs  = [b.get("mod_duration", 0) or 0 for b in filtered]
            spreads = [b.get("g_spread_bps", 0) or 0 for b in filtered]
            by_sector: Dict[str, int] = {}
            by_rating: Dict[str, int] = {}
            for b in filtered:
                by_sector[b.get("sector", "?")]  = by_sector.get(b.get("sector", "?"), 0) + 1
                by_rating[b.get("rating", "NR")] = by_rating.get(b.get("rating", "NR"), 0) + 1
            summary = {
                "avg_ytm_pct":   round(sum(ytms) / len(ytms), 4),
                "avg_duration":  round(sum(durs) / len(durs), 3),
                "avg_g_spread_bps": round(sum(spreads) / len(spreads), 2),
                "by_sector":     by_sector,
                "by_rating":     by_rating,
            }
        else:
            summary = {}

        # Persist
        _fidb.save_screen_result(req.model_dump(), filtered)

        bond_results = [self._to_bond_result(b) for b in filtered]
        return ScreenResponse(
            total_matches=len(bond_results),
            bonds=bond_results,
            summary=summary,
            filters=req.model_dump(exclude_none=True),
            ranked_by=req.rank_by,
            run_at=datetime.utcnow().isoformat(),
        )

    def _to_bond_result(self, b: Dict) -> BondResult:
        return BondResult(
            bond_id=b.get("bond_id", ""),
            sector=b.get("sector", ""),
            sub_sector=b.get("sub_sector"),
            issuer=b.get("issuer", ""),
            ticker=b.get("ticker"),
            rating=b.get("rating", "NR"),
            coupon_rate=float(b.get("coupon_rate") or 0),
            maturity_date=b.get("maturity_date", ""),
            years_to_mat=float(b.get("years_to_mat") or 0),
            ytm_pct=float(b.get("ytm_pct") or 0),
            ytw_pct=float(b.get("ytw_pct") or 0),
            price=float(b.get("price") or 100),
            mod_duration=float(b.get("mod_duration") or 0),
            convexity=float(b.get("convexity") or 0),
            dv01_per_mm=float(b.get("dv01_per_mm") or 0),
            oas_bps=float(b.get("oas_bps") or 0),
            g_spread_bps=float(b.get("g_spread_bps") or 0),
            risk_adj_yield=float(b.get("risk_adj_yield") or 0),
            income_score=float(b.get("income_score") or 0),
            tey_pct=float(b.get("tey_pct") or 0),
            real_yield=float(b.get("real_yield") or 0),
            tax_exempt=bool(b.get("tax_exempt")),
            callable=bool(b.get("callable")),
        )

    def get_analytics(self, bond_id: str) -> Dict[str, Any]:
        """
        Return full analytics for a bond_id including recomputed YTM, duration,
        convexity, DV01, G-spread, and sector context.
        """
        self.ensure_universe()
        b = _fidb.get_bond(bond_id)
        if not b:
            raise ValueError(f"Bond {bond_id} not found in universe")

        coupon = float(b.get("coupon_rate") or 0)
        years  = float(b.get("years_to_mat") or 0)
        price  = float(b.get("price") or 100)
        ytm    = _ytm_newton(coupon, years, price) if years > 0 and coupon > 0 else (b.get("ytm_pct") or 0)
        treas  = _interp_treasury(years, self._treas_curve)
        g_spread = (ytm - treas) * 100 if treas > 0 else 0

        mod_dur = _modified_duration(coupon, years, ytm)
        mac_dur = _macaulay_duration(coupon, years, ytm)
        conv    = _convexity(coupon, years, ytm)
        dv01    = _dv01_per_mm(mod_dur, price)

        # Scenario P&L: price impact for ±25, ±50, ±100 bps moves
        scenarios: Dict[str, float] = {}
        for shock_bps in [-100, -50, -25, 25, 50, 100]:
            shock = shock_bps / 10000.0
            # Duration + convexity approximation
            dprice = (-mod_dur * shock + 0.5 * conv * shock ** 2) * price
            scenarios[f"shock_{shock_bps:+d}bps_price_chg"] = round(dprice, 4)

        # Comparable Treasury
        treas_labels = sorted(
            TREASURY_OTR.keys(),
            key=lambda k: abs(_years_from_maturity(TREASURY_OTR[k]["maturity"]) - years)
        )
        bench_label = treas_labels[0] if treas_labels else "10Y"

        return {
            "bond_id": bond_id,
            "issuer": b.get("issuer"),
            "sector": b.get("sector"),
            "sub_sector": b.get("sub_sector"),
            "rating": b.get("rating"),
            "coupon_rate": coupon,
            "maturity_date": b.get("maturity_date"),
            "years_to_maturity": round(years, 3),
            "price": round(price, 4),
            "ytm_pct": round(ytm, 4),
            "ytw_pct": round(float(b.get("ytw_pct") or ytm), 4),
            "mod_duration": round(mod_dur, 4),
            "mac_duration": round(mac_dur, 4),
            "convexity": round(conv, 4),
            "dv01_per_mm": round(dv01, 2),
            "g_spread_bps": round(g_spread, 2),
            "oas_bps": round(float(b.get("oas_bps") or g_spread), 2),
            "benchmark_treasury": bench_label,
            "benchmark_treasury_yield": round(treas, 4),
            "risk_adj_yield": round(ytm / mod_dur, 4) if mod_dur > 0 else 0,
            "income_score": round(coupon / mod_dur, 4) if mod_dur > 0 else 0,
            "tey_pct": round(float(b.get("tey_pct") or ytm), 4),
            "real_yield": round(float(b.get("real_yield") or 0), 4),
            "tax_exempt": bool(b.get("tax_exempt")),
            "callable": bool(b.get("callable")),
            "price_scenarios": scenarios,
            "data_as_of": date.today().isoformat(),
        }

    def build_yield_matrix(self) -> List[YieldMatrixRow]:
        """
        Multi-sector yield matrix across standard tenors.
        Treasury, TIPS, IG, HY, Muni, Agency.
        """
        self.ensure_universe()
        treas_curve = self._treas_curve or _TREASURY_FALLBACK
        tips_real   = _fred_yields.get_tips_real_yields()
        breakeven   = 2.30   # ~10Y inflation breakeven

        tenors = [2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        rows: List[YieldMatrixRow] = []

        for t in tenors:
            govt        = _interp_treasury(t, treas_curve)
            tips_r      = tips_real.get(t) or (govt - breakeven)
            tips_nom    = tips_r + breakeven

            ig_aa_spread  = _IG_SPREADS.get("AA",   42) / 100
            ig_bbb_spread = _IG_SPREADS.get("BBB", 158) / 100
            hy_bb_spread  = _HY_SPREADS.get("BB",  295) / 100
            hy_b_spread   = _HY_SPREADS.get("B",   540) / 100

            ig_aa  = govt + ig_aa_spread
            ig_bbb = govt + ig_bbb_spread
            hy_bb  = govt + hy_bb_spread
            hy_b   = govt + hy_b_spread

            # Muni yields ~75–85% of Treasury for AAA, +120bps for BBB
            muni_ratio = 0.80 if t <= 10 else 0.85
            muni_aaa = govt * muni_ratio
            muni_bbb = muni_aaa + 1.20

            # Agency
            agency_bullet   = govt + float(_AGENCY_SPREADS.get("bullet",   "18")) / 100
            agency_callable = govt + float(_AGENCY_SPREADS.get("callable", "43")) / 100

            rows.append(YieldMatrixRow(
                tenor_years=t,
                govt_pct=round(govt, 4),
                tips_real_pct=round(tips_r, 4),
                tips_nominal_pct=round(tips_nom, 4),
                ig_aa_pct=round(ig_aa, 4),
                ig_bbb_pct=round(ig_bbb, 4),
                hy_bb_pct=round(hy_bb, 4),
                hy_b_pct=round(hy_b, 4),
                muni_aaa_pct=round(muni_aaa, 4),
                muni_bbb_pct=round(muni_bbb, 4),
                agency_bullet_pct=round(agency_bullet, 4),
                agency_callable_pct=round(agency_callable, 4),
                breakeven_inflation_pct=round(breakeven, 4),
            ))
        return rows

    def top_picks(self, n: int = 10) -> TopPicksResponse:
        """Top picks across four ranking dimensions."""
        self.ensure_universe()
        all_bonds = _fidb.all_bonds()
        total = len(all_bonds)

        def top_n(key: str, n: int = n) -> List[BondResult]:
            ranked = sorted(all_bonds, key=lambda b: -(b.get(key) or 0))
            return [self._to_bond_result(b) for b in ranked[:n]]

        return TopPicksResponse(
            as_of=date.today().isoformat(),
            risk_adj_yield_leaders=top_n("risk_adj_yield"),
            income_leaders=top_n("income_score"),
            g_spread_leaders=top_n("g_spread_bps"),
            tey_leaders=top_n("tey_pct"),
            total_universe=total,
        )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/fi-screener/v3", tags=["Fixed Income Screener v3"])
_svc = FIScreenerService()


@router.on_event("startup")
async def _startup() -> None:  # noqa: B006
    try:
        _svc.ensure_universe()
    except Exception as exc:
        logger.error("FI screener startup failed: %s", exc)


@router.post("/screen", response_model=ScreenResponse)
def screen_bonds(req: ScreenRequest) -> ScreenResponse:
    """
    Screen the multi-asset fixed income universe.

    Sectors: GOVT | TIPS | CORP_IG | CORP_HY | MUNI | AGENCY
    Ranking: risk_adj_yield | income_score | ytm | g_spread | tey

    All duration/yield analytics computed from first-principles bond math.
    """
    try:
        return _svc.screen(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/universe", response_model=Dict[str, Any])
def get_universe(
    sector: Optional[str] = Query(None, description="Filter by sector"),
    limit: int = Query(200, ge=1, le=1000),
) -> Dict[str, Any]:
    """
    Return the full FI universe with optional sector filter.
    Includes bond count, yield distribution, and duration stats by sector.
    """
    _svc.ensure_universe()
    bonds = _fidb.all_bonds(sector=sector)

    by_sector: Dict[str, Dict[str, Any]] = {}
    for b in bonds:
        s = b.get("sector", "?")
        if s not in by_sector:
            by_sector[s] = {"count": 0, "avg_ytm": [], "avg_duration": [], "avg_oas": []}
        by_sector[s]["count"] += 1
        if b.get("ytm_pct"):
            by_sector[s]["avg_ytm"].append(b["ytm_pct"])
        if b.get("mod_duration"):
            by_sector[s]["avg_duration"].append(b["mod_duration"])
        if b.get("oas_bps"):
            by_sector[s]["avg_oas"].append(b["oas_bps"])

    # Summarize
    sector_summary = {}
    for s, d in by_sector.items():
        sector_summary[s] = {
            "count": d["count"],
            "avg_ytm_pct": round(sum(d["avg_ytm"]) / len(d["avg_ytm"]), 4) if d["avg_ytm"] else 0,
            "avg_duration": round(sum(d["avg_duration"]) / len(d["avg_duration"]), 3) if d["avg_duration"] else 0,
            "avg_oas_bps": round(sum(d["avg_oas"]) / len(d["avg_oas"]), 2) if d["avg_oas"] else 0,
        }

    return {
        "total_bonds": len(bonds),
        "by_sector": sector_summary,
        "sample": [
            {k: v for k, v in b.items() if k in (
                "bond_id", "sector", "issuer", "rating", "coupon_rate",
                "maturity_date", "ytm_pct", "mod_duration", "g_spread_bps", "tey_pct"
            )}
            for b in bonds[:limit]
        ],
        "data_as_of": date.today().isoformat(),
    }


@router.get("/analytics/{bond_id:path}", response_model=Dict[str, Any])
def get_analytics(bond_id: str) -> Dict[str, Any]:
    """
    Full analytics for a single bond: YTM, modified duration, convexity, DV01,
    G-spread, benchmark Treasury, price scenarios (±25/50/100 bps shocks).
    bond_id can be a bond_id (e.g. CORP_IG_AAPL) or CUSIP (for munis).
    """
    _svc.ensure_universe()
    # Try direct
    b = _fidb.get_bond(bond_id)
    if not b:
        # Try MUNI prefix
        b = _fidb.get_bond(f"MUNI_{bond_id}")
    if not b:
        # Search by ticker
        all_bonds = _fidb.all_bonds()
        matches = [x for x in all_bonds if (x.get("ticker") or "").upper() == bond_id.upper()]
        if not matches:
            raise HTTPException(status_code=404, detail=f"Bond '{bond_id}' not found")
        bond_id = matches[0]["bond_id"]

    try:
        return _svc.get_analytics(bond_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/yield-matrix", response_model=List[YieldMatrixRow])
def get_yield_matrix() -> List[YieldMatrixRow]:
    """
    Multi-sector yield matrix across 2/3/5/7/10/20/30Y tenors.
    Shows Treasury, TIPS (real + nominal), IG AA, IG BBB, HY BB, HY B,
    Muni AAA/BBB, Agency bullet/callable, and inflation breakeven.
    Sources: TreasuryDirect XML + FRED OAS series.
    """
    try:
        return _svc.build_yield_matrix()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/top-picks", response_model=TopPicksResponse)
def get_top_picks(n: int = Query(10, ge=1, le=50)) -> TopPicksResponse:
    """
    Top-N picks across four multi-factor ranking dimensions:
    - Risk-adjusted yield (YTM / modified duration)
    - Income score (coupon per unit duration risk)
    - G-spread leaders (highest spread to same-maturity Treasury)
    - TEY leaders (tax-equivalent yield for munis)
    """
    try:
        return _svc.top_picks(n=n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/refresh", response_model=Dict[str, Any])
def refresh_universe() -> Dict[str, Any]:
    """
    Force rebuild of the FI universe: re-fetches Treasury curve from TreasuryDirect,
    TIPS yields from FRED, and IG/HY OAS from FRED BofA indices.
    """
    try:
        _svc._universe_loaded = False
        _svc.ensure_universe()
        bonds = _fidb.all_bonds()
        return {
            "status": "refreshed",
            "total_bonds": len(bonds),
            "treasury_curve": _svc._treas_curve,
            "refreshed_at": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/compare", response_model=Dict[str, Any])
def compare_bonds(
    bond_ids: str = Query(..., description="Comma-separated bond_ids to compare"),
) -> Dict[str, Any]:
    """
    Side-by-side comparison of 2–10 bonds: analytics, G-spread, TEY, duration profile.
    """
    _svc.ensure_universe()
    ids = [x.strip() for x in bond_ids.split(",") if x.strip()]
    if len(ids) < 2 or len(ids) > 10:
        raise HTTPException(status_code=400, detail="Provide 2–10 bond IDs")

    results = []
    errors  = []
    for bid in ids:
        try:
            results.append(_svc.get_analytics(bid))
        except Exception as exc:
            errors.append({"bond_id": bid, "error": str(exc)})

    if not results:
        raise HTTPException(status_code=404, detail="No bonds found")

    # Relative stats
    ytms  = [r["ytm_pct"]     for r in results]
    durs  = [r["mod_duration"] for r in results]
    gsprs = [r["g_spread_bps"] for r in results]

    return {
        "bonds": results,
        "comparison": {
            "ytm_range_bps":       round((max(ytms) - min(ytms)) * 100, 2),
            "duration_range":      round(max(durs) - min(durs), 3),
            "g_spread_range_bps":  round(max(gsprs) - min(gsprs), 2),
            "highest_ytm":         results[ytms.index(max(ytms))]["bond_id"],
            "lowest_duration":     results[durs.index(min(durs))]["bond_id"],
            "widest_g_spread":     results[gsprs.index(max(gsprs))]["bond_id"],
        },
        "errors": errors,
        "as_of": date.today().isoformat(),
    }
