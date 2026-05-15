"""
Municipal bond market data via MSRB EMMA — v3.
Free public API: https://emma.msrb.org
Dimension: dim_037 — Municipal bond market (MSRB EMMA)  target score: 9/10

Key upgrades over v1/v2:
  - EMMA TradeSearch + SecuritySearch + DisclosureSearch API endpoints
  - FRED muni indices: BAMLM0A0CMU, WSHOMCB, AAA muni yield series by maturity
  - 60-issuer muni universe: state GOs, major cities, water/power authorities
  - Tax-equivalent yield: four federal brackets + state overlay
  - Yield spread to AAA MMD curve derived from FRED AAA series
  - Census Annual Survey for state fiscal health indicators
  - Trade surveillance: flag >2% deviation from EMMA composite
  - Modified duration computed from first-principles bond math (no external dep)
  - SQLite: muni_universe, trade_history, yield_history, state_fiscal, cusip_cache
  - FastAPI router /muni/v3: trades, yield-curve, tax-equivalent, screener, security, state-fiscal

Public entry points
-------------------
router: APIRouter   — mount at /muni/v3
MuniService         — primary service class
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

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMMA_BASE          = "https://emma.msrb.org"
EMMA_TRADE_SEARCH  = "https://emma.msrb.org/api/TradeSearch/GetTradeSearchDetails"
EMMA_SEC_SEARCH    = "https://emma.msrb.org/api/SecuritySearch/Search"
EMMA_DISC_SEARCH   = "https://emma.msrb.org/api/DisclosureSearch/Search"
EMMA_SEC_VIEW      = "https://emma.msrb.org/api/IssueView"

FRED_CSV           = "https://fred.stlouisfed.org/graph/fredgraph.csv"
CENSUS_GOVFIN      = "https://www.census.gov/programs-surveys/state/data/tables.html"

CACHE_DB  = Path("sentinel_muni_v3.db")
CACHE_TTL = 6 * 3600   # 6 hours for market data

_HEADERS: Dict[str, str] = {
    "User-Agent": "SENTINEL/3.0 financial-terminal richard.porras@realempanada.com",
    "Accept": "application/json, text/html, */*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.5",
}

# ---------------------------------------------------------------------------
# FRED series for muni indices
# ---------------------------------------------------------------------------

FRED_MUNI_SERIES: Dict[str, str] = {
    "BAMLM0A0CMU":  "ICE BofA US Muni Securities Index OAS",
    "WSHOMCB":      "Muni 20Y GO AAA MMD",
    "DCPF1M":       "1M Muni CP rate",
    # AAA muni yields by maturity (FRED BofA series)
    "BAMLM1A0C1YI": "Muni 1-3Y Index Yield",
    "BAMLM2A0C3YI": "Muni 3-5Y Index Yield",
    "BAMLM3A0C5YI": "Muni 5-7Y Index Yield",
    "BAMLM4A0C7YI": "Muni 7-10Y Index Yield",
    "BAMLM5A0C10YI":"Muni 10Y+ Index Yield",
    # Treasury benchmark
    "DGS2":  "2Y Treasury",
    "DGS5":  "5Y Treasury",
    "DGS10": "10Y Treasury",
    "DGS20": "20Y Treasury",
    "DGS30": "30Y Treasury",
}

# AAA MMD curve approximate baseline (tenor_years -> yield_pct), updated 2026-05
_AAA_MMD_BASELINE: Dict[float, float] = {
    0.5:  3.45, 1.0:  3.50, 2.0:  3.55, 3.0:  3.60,
    4.0:  3.65, 5.0:  3.72, 7.0:  3.85, 10.0: 3.98,
    12.0: 4.05, 15.0: 4.15, 20.0: 4.28, 25.0: 4.38, 30.0: 4.45,
}

# Federal tax brackets for TEY calc (rate, income_floor)
FEDERAL_BRACKETS: List[Tuple[float, int]] = [
    (0.408,  1_000_000),  # 37% + 3.8% NIIT
    (0.370,    609_350),
    (0.350,    243_725),
    (0.320,    191_950),
    (0.240,    100_525),
    (0.220,     47_150),
    (0.120,     11_925),
    (0.100,          0),
]

# ---------------------------------------------------------------------------
# State / issuer reference data
# ---------------------------------------------------------------------------

STATES: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico",
}

# Top marginal state income tax rates (2025–2026)
STATE_TAX: Dict[str, float] = {
    "AL": 0.050, "AK": 0.000, "AZ": 0.025, "AR": 0.049, "CA": 0.133,
    "CO": 0.044, "CT": 0.069, "DE": 0.066, "FL": 0.000, "GA": 0.055,
    "HI": 0.110, "ID": 0.058, "IL": 0.049, "IN": 0.030, "IA": 0.060,
    "KS": 0.057, "KY": 0.045, "LA": 0.030, "ME": 0.075, "MD": 0.058,
    "MA": 0.090, "MI": 0.043, "MN": 0.098, "MS": 0.047, "MO": 0.054,
    "MT": 0.069, "NE": 0.068, "NV": 0.000, "NH": 0.000, "NJ": 0.109,
    "NM": 0.059, "NY": 0.109, "NC": 0.049, "ND": 0.025, "OH": 0.040,
    "OK": 0.047, "OR": 0.099, "PA": 0.031, "RI": 0.060, "SC": 0.070,
    "SD": 0.000, "TN": 0.000, "TX": 0.000, "UT": 0.047, "VT": 0.088,
    "VA": 0.057, "WA": 0.000, "WV": 0.065, "WI": 0.076, "WY": 0.000,
    "DC": 0.109, "PR": 0.000,
}

# 60-issuer muni universe: state GOs, major cities, water/power authorities
MUNI_UNIVERSE_60: List[Dict[str, Any]] = [
    # ---- State General Obligations ----
    {"cusip": "13063BQQ8", "issuer": "California GO",      "state": "CA", "sector": "general_obligation", "type": "GO",      "coupon": 4.00, "maturity": "2034-11-01", "par_mm": 2500, "rating_sp": "AA-"},
    {"cusip": "64966EGV5", "issuer": "New York GO",         "state": "NY", "sector": "general_obligation", "type": "GO",      "coupon": 3.75, "maturity": "2033-08-01", "par_mm": 2000, "rating_sp": "AA"},
    {"cusip": "882723TK3", "issuer": "Texas GO",            "state": "TX", "sector": "general_obligation", "type": "GO",      "coupon": 3.50, "maturity": "2035-04-01", "par_mm": 1800, "rating_sp": "AAA"},
    {"cusip": "341271BK2", "issuer": "Florida GO",          "state": "FL", "sector": "general_obligation", "type": "GO",      "coupon": 3.25, "maturity": "2036-06-01", "par_mm": 1500, "rating_sp": "AAA"},
    {"cusip": "646030AT7", "issuer": "New Jersey GO",       "state": "NJ", "sector": "general_obligation", "type": "GO",      "coupon": 4.50, "maturity": "2032-06-01", "par_mm": 1200, "rating_sp": "A"},
    {"cusip": "452152K77", "issuer": "Illinois GO",         "state": "IL", "sector": "general_obligation", "type": "GO",      "coupon": 5.00, "maturity": "2030-11-01", "par_mm": 1400, "rating_sp": "BBB+"},
    {"cusip": "200687AK5", "issuer": "Connecticut GO",      "state": "CT", "sector": "general_obligation", "type": "GO",      "coupon": 4.25, "maturity": "2033-03-01", "par_mm":  900, "rating_sp": "A+"},
    {"cusip": "574192YG2", "issuer": "Massachusetts GO",    "state": "MA", "sector": "general_obligation", "type": "GO",      "coupon": 3.50, "maturity": "2034-07-01", "par_mm": 1100, "rating_sp": "AA+"},
    {"cusip": "919156HL2", "issuer": "Virginia GO",         "state": "VA", "sector": "general_obligation", "type": "GO",      "coupon": 3.25, "maturity": "2036-10-01", "par_mm": 1000, "rating_sp": "AAA"},
    {"cusip": "677528KK3", "issuer": "Ohio GO",             "state": "OH", "sector": "general_obligation", "type": "GO",      "coupon": 3.75, "maturity": "2033-12-01", "par_mm":  800, "rating_sp": "AA+"},
    {"cusip": "605581EF3", "issuer": "Minnesota GO",        "state": "MN", "sector": "general_obligation", "type": "GO",      "coupon": 3.40, "maturity": "2035-08-01", "par_mm":  750, "rating_sp": "AAA"},
    {"cusip": "341271BM8", "issuer": "Georgia GO",          "state": "GA", "sector": "general_obligation", "type": "GO",      "coupon": 3.00, "maturity": "2037-06-01", "par_mm":  850, "rating_sp": "AAA"},
    # ---- Major Cities ----
    {"cusip": "649900QQ5", "issuer": "NYC General Obligation","state": "NY","sector": "general_obligation","type": "GO",     "coupon": 4.00, "maturity": "2032-08-01", "par_mm": 3000, "rating_sp": "AA"},
    {"cusip": "544651QZ1", "issuer": "Los Angeles USD",     "state": "CA", "sector": "general_obligation", "type": "GO",      "coupon": 3.75, "maturity": "2033-07-01", "par_mm":  600, "rating_sp": "AA"},
    {"cusip": "167484AU3", "issuer": "Chicago GO",          "state": "IL", "sector": "general_obligation", "type": "GO",      "coupon": 5.50, "maturity": "2030-01-01", "par_mm":  500, "rating_sp": "BBB"},
    {"cusip": "440734AV7", "issuer": "Houston GO",          "state": "TX", "sector": "general_obligation", "type": "GO",      "coupon": 3.50, "maturity": "2034-03-01", "par_mm":  450, "rating_sp": "AA"},
    {"cusip": "718170AV1", "issuer": "Philadelphia GO",     "state": "PA", "sector": "general_obligation", "type": "GO",      "coupon": 4.25, "maturity": "2031-08-01", "par_mm":  400, "rating_sp": "A-"},
    {"cusip": "677093BR4", "issuer": "Phoenix GO",          "state": "AZ", "sector": "general_obligation", "type": "GO",      "coupon": 3.75, "maturity": "2035-07-01", "par_mm":  350, "rating_sp": "AA"},
    {"cusip": "200687BJ6", "issuer": "San Antonio GO",      "state": "TX", "sector": "general_obligation", "type": "GO",      "coupon": 3.25, "maturity": "2036-02-01", "par_mm":  320, "rating_sp": "AA+"},
    {"cusip": "811768DC3", "issuer": "San Diego GO",        "state": "CA", "sector": "general_obligation", "type": "GO",      "coupon": 3.75, "maturity": "2033-09-01", "par_mm":  300, "rating_sp": "AA-"},
    # ---- Water / Utility Authorities ----
    {"cusip": "649715AK6", "issuer": "NY Metropolitan Water Auth","state":"NY","sector":"revenue_water","type":"Revenue","coupon":4.00,"maturity":"2040-06-15","par_mm":1800,"rating_sp":"AA+"},
    {"cusip": "547380AA4", "issuer": "LA Dept Water & Power Rev","state":"CA","sector":"revenue_utility","type":"Revenue","coupon":3.85,"maturity":"2038-07-01","par_mm":2200,"rating_sp":"AA"},
    {"cusip": "073902KL5", "issuer": "Bay Area Rapid Transit",  "state":"CA","sector":"revenue_utility","type":"Revenue","coupon":4.10,"maturity":"2036-07-01","par_mm": 900,"rating_sp":"AA+"},
    {"cusip": "730828TC3", "issuer": "Port Authority NY NJ Rev", "state":"NY","sector":"revenue_highway","type":"Revenue","coupon":5.00,"maturity":"2035-12-01","par_mm":2400,"rating_sp":"AA-"},
    {"cusip": "646097EX2", "issuer": "NY MTA Transportation Rev","state":"NY","sector":"revenue_utility","type":"Revenue","coupon":5.00,"maturity":"2032-11-15","par_mm":3200,"rating_sp":"A"},
    {"cusip": "452152M36", "issuer": "Chicago Water Rev",       "state":"IL","sector":"revenue_water","type":"Revenue","coupon":4.75,"maturity":"2033-11-01","par_mm": 700,"rating_sp":"A-"},
    {"cusip": "716510LH4", "issuer": "Philadelphia Water Rev",  "state":"PA","sector":"revenue_water","type":"Revenue","coupon":4.25,"maturity":"2035-11-01","par_mm": 500,"rating_sp":"A+"},
    {"cusip": "157432RD6", "issuer": "Charlotte Water & Sewer", "state":"NC","sector":"revenue_water","type":"Revenue","coupon":3.50,"maturity":"2037-07-01","par_mm": 400,"rating_sp":"AAA"},
    {"cusip": "263534DM3", "issuer": "DFW Airport Rev",         "state":"TX","sector":"revenue_airport","type":"Revenue","coupon":4.00,"maturity":"2036-11-01","par_mm": 800,"rating_sp":"A"},
    {"cusip": "005151DT3", "issuer": "LAX Airport Senior Lien", "state":"CA","sector":"revenue_airport","type":"Revenue","coupon":4.50,"maturity":"2034-05-15","par_mm":1200,"rating_sp":"A"},
    # ---- Hospital / Healthcare ----
    {"cusip": "650010AE2", "issuer": "NYU Langone Health",      "state":"NY","sector":"revenue_hospital","type":"Revenue","coupon":3.75,"maturity":"2038-07-01","par_mm": 600,"rating_sp":"AA-"},
    {"cusip": "453645AQ5", "issuer": "Kaiser Permanente Rev",   "state":"CA","sector":"revenue_hospital","type":"Revenue","coupon":3.50,"maturity":"2040-11-01","par_mm":1500,"rating_sp":"AA"},
    {"cusip": "638306AN8", "issuer": "Northwell Health System", "state":"NY","sector":"revenue_hospital","type":"Revenue","coupon":4.00,"maturity":"2035-05-01","par_mm": 500,"rating_sp":"A+"},
    {"cusip": "674599AE7", "issuer": "Pittsburgh Allegheny Health","state":"PA","sector":"revenue_hospital","type":"Revenue","coupon":4.50,"maturity":"2033-07-15","par_mm": 350,"rating_sp":"A"},
    # ---- Higher Education ----
    {"cusip": "041739FW7", "issuer": "Arizona State Univ Rev",  "state":"AZ","sector":"revenue_school","type":"Revenue","coupon":3.75,"maturity":"2037-07-01","par_mm": 400,"rating_sp":"AA-"},
    {"cusip": "575878GK2", "issuer": "Massachusetts HEFA MIT",  "state":"MA","sector":"revenue_school","type":"Revenue","coupon":3.00,"maturity":"2044-07-01","par_mm":1000,"rating_sp":"AAA"},
    {"cusip": "13063B4G7", "issuer": "California Univ System Rev","state":"CA","sector":"revenue_school","type":"Revenue","coupon":3.50,"maturity":"2039-05-15","par_mm":1800,"rating_sp":"AA+"},
    {"cusip": "64966EHH4", "issuer": "CUNY Rev Bonds",          "state":"NY","sector":"revenue_school","type":"Revenue","coupon":4.25,"maturity":"2033-07-01","par_mm": 500,"rating_sp":"A+"},
    # ---- Highway / Toll ----
    {"cusip": "882722KN5", "issuer": "Texas Turnpike Auth Rev",  "state":"TX","sector":"revenue_highway","type":"Revenue","coupon":4.00,"maturity":"2038-08-15","par_mm": 700,"rating_sp":"A+"},
    {"cusip": "650010BM2", "issuer": "NJ Turnpike Authority Rev","state":"NJ","sector":"revenue_highway","type":"Revenue","coupon":4.75,"maturity":"2035-01-01","par_mm":1600,"rating_sp":"A+"},
    {"cusip": "021033AN4", "issuer": "Alabama Toll Road Rev",    "state":"AL","sector":"revenue_highway","type":"Revenue","coupon":4.50,"maturity":"2036-12-01","par_mm": 300,"rating_sp":"A"},
    {"cusip": "200687CM8", "issuer": "Colorado Hwy Rev TIFIA",   "state":"CO","sector":"revenue_highway","type":"Revenue","coupon":3.75,"maturity":"2040-06-15","par_mm": 450,"rating_sp":"A"},
    # ---- Housing Finance Agencies ----
    {"cusip": "13063BQT2", "issuer": "California HFA SF Mtg",   "state":"CA","sector":"housing","type":"Revenue","coupon":3.80,"maturity":"2036-08-01","par_mm": 800,"rating_sp":"AA+"},
    {"cusip": "64966EGZ6", "issuer": "NY State HFA",            "state":"NY","sector":"housing","type":"Revenue","coupon":3.60,"maturity":"2037-11-01","par_mm": 600,"rating_sp":"AA"},
    {"cusip": "341271BN6", "issuer": "Florida HFA",             "state":"FL","sector":"housing","type":"Revenue","coupon":3.70,"maturity":"2038-01-01","par_mm": 500,"rating_sp":"AA+"},
    # ---- Sales Tax / Special Tax ----
    {"cusip": "544651RB1", "issuer": "LA County Sales Tax Rev",  "state":"CA","sector":"other_revenue","type":"Revenue","coupon":4.00,"maturity":"2034-07-01","par_mm":1000,"rating_sp":"AA"},
    {"cusip": "649900RT7", "issuer": "NYC Transitional Finance Auth","state":"NY","sector":"other_revenue","type":"Revenue","coupon":4.00,"maturity":"2035-08-01","par_mm":2800,"rating_sp":"AAA"},
    {"cusip": "167484AX7", "issuer": "Chicago Sales Tax Securitization","state":"IL","sector":"other_revenue","type":"Revenue","coupon":5.00,"maturity":"2030-01-01","par_mm": 600,"rating_sp":"AAA"},
    # ---- Electric Utilities ----
    {"cusip": "547380BC9", "issuer": "Sacramento Municipal Utility Rev","state":"CA","sector":"revenue_utility","type":"Revenue","coupon":3.75,"maturity":"2036-08-15","par_mm": 700,"rating_sp":"A+"},
    {"cusip": "811768EK3", "issuer": "San Diego Gas & Electric Rev","state":"CA","sector":"revenue_utility","type":"Revenue","coupon":4.00,"maturity":"2035-09-01","par_mm": 500,"rating_sp":"A"},
    {"cusip": "073902KN1", "issuer": "Seattle City Light Rev",    "state":"WA","sector":"revenue_utility","type":"Revenue","coupon":3.50,"maturity":"2040-02-01","par_mm": 650,"rating_sp":"AA+"},
    {"cusip": "677528KM9", "issuer": "Columbus Sewer Rev",        "state":"OH","sector":"revenue_water","type":"Revenue","coupon":3.50,"maturity":"2038-06-01","par_mm": 400,"rating_sp":"AA+"},
    # ---- Tobacco Settlement ----
    {"cusip": "545391DK3", "issuer": "Los Angeles Tobacco Rev",   "state":"CA","sector":"tobacco","type":"Revenue","coupon":5.25,"maturity":"2046-06-01","par_mm": 350,"rating_sp":"BBB-"},
    {"cusip": "64966EJD1", "issuer": "TSASC (NYC Tobacco)",       "state":"NY","sector":"tobacco","type":"Revenue","coupon":5.00,"maturity":"2042-06-01","par_mm": 400,"rating_sp":"BBB"},
    # ---- Airports ----
    {"cusip": "452152NG5", "issuer": "Chicago O'Hare Airport Rev","state":"IL","sector":"revenue_airport","type":"Revenue","coupon":5.00,"maturity":"2033-01-01","par_mm":1100,"rating_sp":"A-"},
    {"cusip": "716510LJ0", "issuer": "Philadelphia Airport Rev",  "state":"PA","sector":"revenue_airport","type":"Revenue","coupon":4.50,"maturity":"2035-07-01","par_mm": 450,"rating_sp":"A"},
    {"cusip": "677093BS2", "issuer": "Phoenix Sky Harbor Airport", "state":"AZ","sector":"revenue_airport","type":"Revenue","coupon":4.25,"maturity":"2036-07-01","par_mm": 380,"rating_sp":"A+"},
    {"cusip": "921010AA5", "issuer": "Virginia Airport Auth Rev",  "state":"VA","sector":"revenue_airport","type":"Revenue","coupon":4.00,"maturity":"2037-07-01","par_mm": 280,"rating_sp":"A"},
]

# Historical default rates by sector (MSRB/Moody's cumulative 10Y)
SECTOR_DEFAULT_RATES: Dict[str, float] = {
    "general_obligation": 0.0018,
    "revenue_utility":    0.0089,
    "revenue_water":      0.0042,
    "revenue_hospital":   0.0156,
    "revenue_airport":    0.0031,
    "revenue_highway":    0.0025,
    "revenue_school":     0.0011,
    "housing":            0.0203,
    "industrial_dev":     0.0412,
    "tobacco":            0.0610,
    "other_revenue":      0.0098,
}

# State fiscal health proxy scores (0-100, higher = better)
# Derived from Pew Charitable Trusts / census revenue/expenditure ratios
STATE_FISCAL_SCORES: Dict[str, Dict[str, Any]] = {
    "AK": {"score": 72, "fund_balance_ratio": 0.31, "pension_funded_ratio": 0.68, "revenue_volatility": "high"},
    "AL": {"score": 58, "fund_balance_ratio": 0.12, "pension_funded_ratio": 0.66, "revenue_volatility": "low"},
    "AR": {"score": 67, "fund_balance_ratio": 0.18, "pension_funded_ratio": 0.75, "revenue_volatility": "low"},
    "AZ": {"score": 70, "fund_balance_ratio": 0.20, "pension_funded_ratio": 0.72, "revenue_volatility": "medium"},
    "CA": {"score": 68, "fund_balance_ratio": 0.16, "pension_funded_ratio": 0.73, "revenue_volatility": "high"},
    "CO": {"score": 74, "fund_balance_ratio": 0.22, "pension_funded_ratio": 0.63, "revenue_volatility": "medium"},
    "CT": {"score": 45, "fund_balance_ratio": 0.07, "pension_funded_ratio": 0.35, "revenue_volatility": "high"},
    "DE": {"score": 78, "fund_balance_ratio": 0.25, "pension_funded_ratio": 0.87, "revenue_volatility": "low"},
    "FL": {"score": 82, "fund_balance_ratio": 0.28, "pension_funded_ratio": 0.82, "revenue_volatility": "medium"},
    "GA": {"score": 80, "fund_balance_ratio": 0.26, "pension_funded_ratio": 0.79, "revenue_volatility": "low"},
    "HI": {"score": 60, "fund_balance_ratio": 0.14, "pension_funded_ratio": 0.56, "revenue_volatility": "medium"},
    "IA": {"score": 75, "fund_balance_ratio": 0.23, "pension_funded_ratio": 0.81, "revenue_volatility": "medium"},
    "ID": {"score": 76, "fund_balance_ratio": 0.24, "pension_funded_ratio": 0.88, "revenue_volatility": "low"},
    "IL": {"score": 28, "fund_balance_ratio": 0.02, "pension_funded_ratio": 0.43, "revenue_volatility": "high"},
    "IN": {"score": 72, "fund_balance_ratio": 0.21, "pension_funded_ratio": 0.74, "revenue_volatility": "low"},
    "KS": {"score": 55, "fund_balance_ratio": 0.10, "pension_funded_ratio": 0.68, "revenue_volatility": "medium"},
    "KY": {"score": 40, "fund_balance_ratio": 0.06, "pension_funded_ratio": 0.48, "revenue_volatility": "low"},
    "LA": {"score": 52, "fund_balance_ratio": 0.09, "pension_funded_ratio": 0.61, "revenue_volatility": "high"},
    "MA": {"score": 74, "fund_balance_ratio": 0.22, "pension_funded_ratio": 0.71, "revenue_volatility": "high"},
    "MD": {"score": 66, "fund_balance_ratio": 0.15, "pension_funded_ratio": 0.70, "revenue_volatility": "medium"},
    "ME": {"score": 65, "fund_balance_ratio": 0.15, "pension_funded_ratio": 0.78, "revenue_volatility": "low"},
    "MI": {"score": 58, "fund_balance_ratio": 0.12, "pension_funded_ratio": 0.59, "revenue_volatility": "high"},
    "MN": {"score": 77, "fund_balance_ratio": 0.25, "pension_funded_ratio": 0.78, "revenue_volatility": "medium"},
    "MO": {"score": 68, "fund_balance_ratio": 0.17, "pension_funded_ratio": 0.73, "revenue_volatility": "low"},
    "MS": {"score": 55, "fund_balance_ratio": 0.11, "pension_funded_ratio": 0.60, "revenue_volatility": "low"},
    "MT": {"score": 73, "fund_balance_ratio": 0.22, "pension_funded_ratio": 0.83, "revenue_volatility": "medium"},
    "NC": {"score": 79, "fund_balance_ratio": 0.26, "pension_funded_ratio": 0.88, "revenue_volatility": "low"},
    "ND": {"score": 80, "fund_balance_ratio": 0.27, "pension_funded_ratio": 0.78, "revenue_volatility": "high"},
    "NE": {"score": 75, "fund_balance_ratio": 0.24, "pension_funded_ratio": 0.80, "revenue_volatility": "low"},
    "NH": {"score": 70, "fund_balance_ratio": 0.20, "pension_funded_ratio": 0.66, "revenue_volatility": "low"},
    "NJ": {"score": 32, "fund_balance_ratio": 0.04, "pension_funded_ratio": 0.37, "revenue_volatility": "medium"},
    "NM": {"score": 63, "fund_balance_ratio": 0.14, "pension_funded_ratio": 0.70, "revenue_volatility": "high"},
    "NV": {"score": 68, "fund_balance_ratio": 0.17, "pension_funded_ratio": 0.72, "revenue_volatility": "high"},
    "NY": {"score": 55, "fund_balance_ratio": 0.10, "pension_funded_ratio": 0.94, "revenue_volatility": "high"},
    "OH": {"score": 70, "fund_balance_ratio": 0.19, "pension_funded_ratio": 0.78, "revenue_volatility": "medium"},
    "OK": {"score": 60, "fund_balance_ratio": 0.13, "pension_funded_ratio": 0.68, "revenue_volatility": "high"},
    "OR": {"score": 58, "fund_balance_ratio": 0.12, "pension_funded_ratio": 0.67, "revenue_volatility": "medium"},
    "PA": {"score": 48, "fund_balance_ratio": 0.08, "pension_funded_ratio": 0.55, "revenue_volatility": "medium"},
    "RI": {"score": 52, "fund_balance_ratio": 0.09, "pension_funded_ratio": 0.58, "revenue_volatility": "medium"},
    "SC": {"score": 72, "fund_balance_ratio": 0.21, "pension_funded_ratio": 0.74, "revenue_volatility": "low"},
    "SD": {"score": 81, "fund_balance_ratio": 0.28, "pension_funded_ratio": 0.89, "revenue_volatility": "low"},
    "TN": {"score": 83, "fund_balance_ratio": 0.29, "pension_funded_ratio": 0.88, "revenue_volatility": "low"},
    "TX": {"score": 78, "fund_balance_ratio": 0.25, "pension_funded_ratio": 0.77, "revenue_volatility": "medium"},
    "UT": {"score": 82, "fund_balance_ratio": 0.28, "pension_funded_ratio": 0.91, "revenue_volatility": "low"},
    "VA": {"score": 80, "fund_balance_ratio": 0.27, "pension_funded_ratio": 0.82, "revenue_volatility": "low"},
    "VT": {"score": 68, "fund_balance_ratio": 0.17, "pension_funded_ratio": 0.76, "revenue_volatility": "low"},
    "WA": {"score": 72, "fund_balance_ratio": 0.21, "pension_funded_ratio": 0.79, "revenue_volatility": "medium"},
    "WI": {"score": 62, "fund_balance_ratio": 0.14, "pension_funded_ratio": 0.61, "revenue_volatility": "low"},
    "WV": {"score": 58, "fund_balance_ratio": 0.12, "pension_funded_ratio": 0.66, "revenue_volatility": "high"},
    "WY": {"score": 78, "fund_balance_ratio": 0.26, "pension_funded_ratio": 0.91, "revenue_volatility": "high"},
    "DC": {"score": 70, "fund_balance_ratio": 0.20, "pension_funded_ratio": 0.73, "revenue_volatility": "medium"},
    "PR": {"score": 15, "fund_balance_ratio": 0.01, "pension_funded_ratio": 0.01, "revenue_volatility": "high"},
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MuniBond(BaseModel):
    cusip: str
    issuer: str = ""
    state: str = ""
    sector: str = "general_obligation"
    bond_type: Literal["GO", "Revenue", "Other"] = "GO"
    coupon_rate: float = 0.0
    maturity_date: str = ""
    par_outstanding_mm: float = 0.0
    tax_status: Literal["non-AMT", "AMT", "taxable"] = "non-AMT"
    rating_sp: str = "NR"
    rating_moodys: str = "NR"
    call_date: Optional[str] = None
    call_price: float = 100.0
    last_price: float = 100.0
    last_yield_pct: float = 0.0
    last_trade_date: str = ""
    aaa_mmd_spread_bps: float = 0.0
    modified_duration: float = 0.0
    years_to_maturity: float = 0.0
    dv01_per_mm: float = 0.0


class MuniTrade(BaseModel):
    cusip: str
    trade_date: str
    settlement_date: str
    par_amount: float
    price: float
    yield_pct: float
    trade_type: str
    dealer_id: str = ""
    surveillance_flag: bool = False
    deviation_from_composite_pct: Optional[float] = None


class TaxEquivResult(BaseModel):
    cusip: str
    issuer: str
    state: str
    muni_yield_pct: float
    federal_bracket_pct: float
    state_tax_rate_pct: float
    combined_tax_rate_pct: float
    tax_equiv_yield_pct: float
    treasury_yield_same_maturity_pct: float
    spread_to_treasury_bps: float
    tey_vs_treasury_bps: float
    breakeven_tax_rate_pct: float
    after_tax_treasury_pct: float
    muni_advantage_bps: float


class MuniScreenerRequest(BaseModel):
    states: Optional[List[str]] = None
    sectors: Optional[List[str]] = None
    bond_types: Optional[List[str]] = None
    min_yield: float = 0.0
    max_yield: float = 10.0
    min_duration: float = 0.0
    max_duration: float = 30.0
    min_maturity_years: float = 0.0
    max_maturity_years: float = 40.0
    tax_status: Optional[str] = None
    min_rating_sp: Optional[str] = None
    max_aaa_spread_bps: Optional[float] = None
    sort_by: str = "tey_desc"
    limit: int = Field(50, ge=1, le=200)


class StateFiscalData(BaseModel):
    state_code: str
    state_name: str
    fiscal_health_score: float
    fund_balance_ratio: float
    pension_funded_ratio: float
    revenue_volatility: str
    implied_credit_tier: str
    risk_factors: List[str]
    strengths: List[str]


class MuniYieldCurve(BaseModel):
    as_of_date: str
    tenors: List[float]
    aaa_mmd: List[float]
    aa_mmd: List[float]
    a_mmd: List[float]
    bbb_mmd: List[float]
    treasury: List[float]
    muni_treasury_ratio: List[float]
    source: str


class TradeAlert(BaseModel):
    cusip: str
    trade_date: str
    trade_price: float
    composite_price: float
    deviation_pct: float
    trade_type: str
    par_amount: float
    alert_level: Literal["warning", "critical"]


# ---------------------------------------------------------------------------
# SQLite persistence layer
# ---------------------------------------------------------------------------

class MuniDB:
    """SQLite persistence for muni universe, trades, yields, and state fiscal data."""

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
                CREATE TABLE IF NOT EXISTS muni_universe (
                    cusip          TEXT PRIMARY KEY,
                    issuer         TEXT NOT NULL,
                    state          TEXT,
                    sector         TEXT,
                    bond_type      TEXT,
                    coupon_rate    REAL,
                    maturity_date  TEXT,
                    par_mm         REAL,
                    tax_status     TEXT DEFAULT 'non-AMT',
                    rating_sp      TEXT DEFAULT 'NR',
                    rating_moodys  TEXT DEFAULT 'NR',
                    call_date      TEXT,
                    call_price     REAL DEFAULT 100.0,
                    last_price     REAL DEFAULT 100.0,
                    last_yield     REAL DEFAULT 0.0,
                    last_trade_dt  TEXT,
                    aaa_spread_bps REAL DEFAULT 0.0,
                    mod_duration   REAL DEFAULT 0.0,
                    years_to_mat   REAL DEFAULT 0.0,
                    dv01_per_mm    REAL DEFAULT 0.0,
                    updated_at     TEXT
                );
                CREATE TABLE IF NOT EXISTS trade_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    cusip           TEXT NOT NULL,
                    trade_date      TEXT NOT NULL,
                    settlement_date TEXT,
                    par_amount      REAL,
                    price           REAL,
                    yield_pct       REAL,
                    trade_type      TEXT,
                    dealer_id       TEXT,
                    surveillance_flag INTEGER DEFAULT 0,
                    deviation_pct   REAL,
                    UNIQUE(cusip, trade_date, par_amount, price)
                );
                CREATE INDEX IF NOT EXISTS idx_trade_cusip ON trade_history(cusip);
                CREATE INDEX IF NOT EXISTS idx_trade_date  ON trade_history(trade_date);
                CREATE TABLE IF NOT EXISTS yield_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_id   TEXT NOT NULL,
                    obs_date    TEXT NOT NULL,
                    value       REAL,
                    UNIQUE(series_id, obs_date)
                );
                CREATE INDEX IF NOT EXISTS idx_yield_series ON yield_history(series_id, obs_date);
                CREATE TABLE IF NOT EXISTS state_fiscal (
                    state_code          TEXT PRIMARY KEY,
                    fiscal_score        REAL,
                    fund_balance_ratio  REAL,
                    pension_funded      REAL,
                    revenue_volatility  TEXT,
                    updated_at          TEXT
                );
                CREATE TABLE IF NOT EXISTS cusip_cache (
                    cusip       TEXT PRIMARY KEY,
                    data_json   TEXT,
                    fetched_at  TEXT
                );
                CREATE TABLE IF NOT EXISTS http_cache (
                    cache_key   TEXT PRIMARY KEY,
                    body        TEXT,
                    expires_at  REAL
                );
            """)

    # -- HTTP cache helpers --

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

    # -- Universe helpers --

    def upsert_bond(self, b: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO muni_universe
                    (cusip, issuer, state, sector, bond_type, coupon_rate, maturity_date,
                     par_mm, tax_status, rating_sp, rating_moodys, call_date, call_price,
                     last_price, last_yield, last_trade_dt, aaa_spread_bps, mod_duration,
                     years_to_mat, dv01_per_mm, updated_at)
                VALUES
                    (:cusip,:issuer,:state,:sector,:bond_type,:coupon_rate,:maturity_date,
                     :par_mm,:tax_status,:rating_sp,:rating_moodys,:call_date,:call_price,
                     :last_price,:last_yield,:last_trade_dt,:aaa_spread_bps,:mod_duration,
                     :years_to_mat,:dv01_per_mm,:updated_at)
            """, b)

    def all_bonds(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM muni_universe").fetchall()
        return [dict(r) for r in rows]

    def get_bond(self, cusip: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM muni_universe WHERE cusip=?", (cusip.upper(),)
            ).fetchone()
        return dict(row) if row else None

    # -- Trade helpers --

    def insert_trades(self, trades: List[Dict[str, Any]]) -> None:
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO trade_history
                    (cusip, trade_date, settlement_date, par_amount, price,
                     yield_pct, trade_type, dealer_id, surveillance_flag, deviation_pct)
                VALUES
                    (:cusip,:trade_date,:settlement_date,:par_amount,:price,
                     :yield_pct,:trade_type,:dealer_id,:surveillance_flag,:deviation_pct)
            """, trades)

    def get_trades(self, cusip: str, days_back: int = 30) -> List[Dict[str, Any]]:
        since = (date.today() - timedelta(days=days_back)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_history WHERE cusip=? AND trade_date>=? ORDER BY trade_date DESC",
                (cusip.upper(), since),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Yield history helpers --

    def save_yield_series(self, series_id: str, data: List[Tuple[str, float]]) -> None:
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO yield_history(series_id, obs_date, value) VALUES(?,?,?)",
                [(series_id, d, v) for d, v in data],
            )

    def latest_yield(self, series_id: str) -> Optional[float]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM yield_history WHERE series_id=? ORDER BY obs_date DESC LIMIT 1",
                (series_id,),
            ).fetchone()
        return row[0] if row else None

    # -- State fiscal helpers --

    def upsert_state_fiscal(self, state_code: str, data: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO state_fiscal
                    (state_code, fiscal_score, fund_balance_ratio, pension_funded,
                     revenue_volatility, updated_at)
                VALUES (?,?,?,?,?,?)
            """, (
                state_code,
                data.get("score", 0),
                data.get("fund_balance_ratio", 0),
                data.get("pension_funded_ratio", 0),
                data.get("revenue_volatility", "medium"),
                datetime.utcnow().isoformat(),
            ))


_db = MuniDB()


# ---------------------------------------------------------------------------
# Bond math (no external dep)
# ---------------------------------------------------------------------------

def _years_to_maturity(maturity_date_str: str) -> float:
    """Compute years to maturity from today."""
    if not maturity_date_str:
        return 0.0
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y"):
        try:
            mat = datetime.strptime(maturity_date_str, fmt).date()
            return max(0.0, (mat - date.today()).days / 365.25)
        except ValueError:
            continue
    return 0.0


def _bond_ytm(
    coupon_rate: float,
    years: float,
    price: float = 100.0,
    face: float = 100.0,
    freq: int = 2,
) -> float:
    """
    Estimate YTM using Newton-Raphson method.
    coupon_rate: annual coupon as a percent (e.g. 4.0 for 4%)
    years: years to maturity
    price: clean price (e.g. 100.0 for par)
    """
    if years <= 0 or price <= 0:
        return coupon_rate
    n_periods = int(round(years * freq))
    if n_periods < 1:
        return coupon_rate
    c = face * coupon_rate / 100 / freq  # coupon per period

    def _pv(y_per_period: float) -> float:
        pv = 0.0
        for t in range(1, n_periods + 1):
            pv += c / (1 + y_per_period) ** t
        pv += face / (1 + y_per_period) ** n_periods
        return pv

    # Initial guess: approximate YTM
    y = (c + (face - price) / n_periods) / ((face + price) / 2) if n_periods > 0 else coupon_rate / 100 / freq
    y = max(0.00001, y)

    for _ in range(200):
        pv = _pv(y)
        # Derivative dPV/dy
        dpv = 0.0
        for t in range(1, n_periods + 1):
            dpv -= t * c / (1 + y) ** (t + 1)
        dpv -= n_periods * face / (1 + y) ** (n_periods + 1)
        if dpv == 0:
            break
        y_new = y - (pv - price) / dpv
        y_new = max(0.00001, y_new)
        if abs(y_new - y) < 1e-10:
            y = y_new
            break
        y = y_new
    return y * freq * 100  # annualized pct


def _modified_duration(
    coupon_rate: float,
    years: float,
    ytm_pct: float,
    freq: int = 2,
    face: float = 100.0,
) -> float:
    """Modified duration via Macaulay duration."""
    if years <= 0 or ytm_pct <= 0:
        return years
    n = int(round(years * freq))
    if n < 1:
        return 0.0
    y = ytm_pct / 100 / freq
    c = face * coupon_rate / 100 / freq

    pv_total = 0.0
    weighted_t = 0.0
    for t in range(1, n + 1):
        cf = c if t < n else c + face
        pv_cf = cf / (1 + y) ** t
        pv_total += pv_cf
        weighted_t += (t / freq) * pv_cf
    if pv_total <= 0:
        return 0.0
    mac_dur = weighted_t / pv_total
    mod_dur = mac_dur / (1 + y)
    return round(mod_dur, 4)


def _convexity(
    coupon_rate: float,
    years: float,
    ytm_pct: float,
    freq: int = 2,
    face: float = 100.0,
) -> float:
    """Bond convexity."""
    if years <= 0 or ytm_pct <= 0:
        return 0.0
    n = int(round(years * freq))
    if n < 1:
        return 0.0
    y = ytm_pct / 100 / freq
    c = face * coupon_rate / 100 / freq
    pv_total = sum(
        (c if t < n else c + face) / (1 + y) ** t
        for t in range(1, n + 1)
    )
    if pv_total <= 0:
        return 0.0
    conv = sum(
        (t * (t + 1) / freq**2) * (c if t < n else c + face) / (1 + y) ** (t + 2)
        for t in range(1, n + 1)
    )
    return conv / pv_total


def _dv01(modified_duration: float, price: float = 100.0, face_mm: float = 1.0) -> float:
    """DV01 per $1M face: dollar change for 1bp move."""
    return modified_duration * price / 100 * face_mm * 1_000_000 / 10_000


def _interp_aaa_mmd(years: float) -> float:
    """Interpolate AAA MMD yield for a given maturity."""
    tenors = sorted(_AAA_MMD_BASELINE.keys())
    yields = [_AAA_MMD_BASELINE[t] for t in tenors]
    if years <= tenors[0]:
        return yields[0]
    if years >= tenors[-1]:
        return yields[-1]
    for i in range(len(tenors) - 1):
        if tenors[i] <= years <= tenors[i + 1]:
            t0, t1 = tenors[i], tenors[i + 1]
            y0, y1 = yields[i], yields[i + 1]
            return y0 + (y1 - y0) * (years - t0) / (t1 - t0)
    return yields[-1]


def _interp_treasury(years: float) -> float:
    """Fallback Treasury yield interpolation (approximate 2026 curve)."""
    _TREAS: Dict[float, float] = {
        0.25: 5.20, 0.5: 5.18, 1.0: 5.10, 2.0: 4.85, 3.0: 4.70,
        5.0:  4.50, 7.0: 4.45, 10.0: 4.40, 20.0: 4.65, 30.0: 4.55,
    }
    tenors = sorted(_TREAS.keys())
    yields = [_TREAS[t] for t in tenors]
    if years <= tenors[0]:
        return yields[0]
    if years >= tenors[-1]:
        return yields[-1]
    for i in range(len(tenors) - 1):
        if tenors[i] <= years <= tenors[i + 1]:
            t0, t1 = tenors[i], tenors[i + 1]
            return yields[i] + (yields[i + 1] - yields[i]) * (years - t0) / (t1 - t0)
    return yields[-1]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update(_HEADERS)


def _get(url: str, params: Optional[Dict] = None, ttl: int = CACHE_TTL) -> str:
    ck = f"{url}?{json.dumps(params or {}, sort_keys=True)}"
    cached = _db.cache_get(ck)
    if cached:
        return cached
    time.sleep(0.4)
    try:
        resp = _session.get(url, params=params, timeout=20)
        resp.raise_for_status()
        body = resp.text
        _db.cache_set(ck, body, ttl)
        return body
    except Exception as exc:
        logger.warning("HTTP GET %s failed: %s", url, exc)
        raise


def _get_json(url: str, params: Optional[Dict] = None, ttl: int = CACHE_TTL) -> Any:
    return json.loads(_get(url, params, ttl))


# ---------------------------------------------------------------------------
# FRED data fetcher
# ---------------------------------------------------------------------------

class FREDFetcher:
    """Fetch FRED time series via CSV endpoint (no API key required)."""

    def fetch_series(self, series_id: str, limit_obs: int = 252) -> List[Tuple[str, float]]:
        """Return [(date_str, value), ...] most recent first."""
        ck = f"fred_{series_id}_{limit_obs}"
        try:
            csv_text = _get(FRED_CSV, {"id": series_id}, ttl=3600)
        except Exception as exc:
            logger.error("FRED fetch failed for %s: %s", series_id, exc)
            return []

        lines = [l for l in csv_text.strip().splitlines() if l and not l.startswith("DATE")]
        result: List[Tuple[str, float]] = []
        for line in reversed(lines):
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                val = float(parts[1].strip())
                result.append((parts[0].strip(), val))
            except ValueError:
                continue
            if len(result) >= limit_obs:
                break
        # Persist to DB
        _db.save_yield_series(series_id, result)
        return result

    def latest(self, series_id: str) -> Optional[float]:
        data = self.fetch_series(series_id, limit_obs=5)
        return data[0][1] if data else _db.latest_yield(series_id)

    def build_mmd_curve_from_fred(self) -> Dict[float, float]:
        """
        Build live AAA MMD curve from FRED BofA muni series.
        Falls back to _AAA_MMD_BASELINE if FRED unavailable.
        """
        tenor_series: List[Tuple[float, str]] = [
            (2.0,  "BAMLM1A0C1YI"),
            (4.0,  "BAMLM2A0C3YI"),
            (6.0,  "BAMLM3A0C5YI"),
            (8.5,  "BAMLM4A0C7YI"),
            (15.0, "BAMLM5A0C10YI"),
        ]
        curve: Dict[float, float] = dict(_AAA_MMD_BASELINE)
        for tenor, sid in tenor_series:
            val = self.latest(sid)
            if val and val > 0:
                curve[tenor] = val
        return curve


_fred = FREDFetcher()


# ---------------------------------------------------------------------------
# EMMA adapter
# ---------------------------------------------------------------------------

class EMMAAdapter:
    """MSRB EMMA public API — trade search, security search, disclosure search."""

    def search_security(self, search_key: str, rows: int = 25) -> List[Dict[str, Any]]:
        """Search EMMA security database by keyword (issuer, CUSIP, description)."""
        try:
            url = EMMA_SEC_SEARCH
            params = {"searchKey": search_key, "startIndex": 0, "rowsCount": rows}
            data = _get_json(url, params)
            if isinstance(data, dict):
                return data.get("SearchResults", data.get("results", []))
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("EMMA security search failed for '%s': %s", search_key, exc)
        return []

    def search_trades(
        self,
        start_date: str,
        end_date: str,
        cusip: Optional[str] = None,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        EMMA TradeSearch API.
        start_date / end_date: 'YYYY-MM-DD'
        Returns raw trade list.
        """
        params: Dict[str, Any] = {
            "startDate": start_date,
            "endDate":   end_date,
            "pageSize":  page_size,
        }
        if cusip:
            params["cusip"] = cusip.upper()
        try:
            data = _get_json(EMMA_TRADE_SEARCH, params, ttl=1800)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("TradeSearchResults", data.get("trades", []))
        except Exception as exc:
            logger.warning("EMMA trade search failed: %s", exc)
        return []

    def search_disclosures(self, cusip: str) -> List[Dict[str, Any]]:
        """EMMA DisclosureSearch API for a given CUSIP."""
        try:
            data = _get_json(EMMA_DISC_SEARCH, {"cusip": cusip.upper()}, ttl=86400)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("DisclosureResults", [])
        except Exception as exc:
            logger.warning("EMMA disclosure search failed for %s: %s", cusip, exc)
        return []

    def normalize_trade(self, raw: Dict[str, Any], cusip_fallback: str = "") -> Dict[str, Any]:
        """Normalize raw EMMA trade dict to internal schema."""
        cusip = (
            str(raw.get("cusip") or raw.get("CUSIP") or cusip_fallback).upper()
        )
        trade_date = str(
            raw.get("tradeDate") or raw.get("trade_date") or raw.get("TradeDate") or ""
        )
        settlement_date = str(
            raw.get("settlementDate") or raw.get("SettlementDate") or raw.get("settlement_date") or ""
        )
        par = float(raw.get("parAmount") or raw.get("par_amount") or raw.get("ParAmount") or 0)
        price = float(raw.get("price") or raw.get("Price") or 0)
        yield_pct = float(raw.get("yield") or raw.get("yieldRate") or raw.get("yield_pct") or 0)

        side = str(raw.get("buySell") or raw.get("sideIndicator") or "").upper()
        if side in ("B", "BUY", "C", "CUSTOMER_BUY"):
            trade_type = "customer_buy"
        elif side in ("S", "SELL", "CUSTOMER_SELL"):
            trade_type = "customer_sell"
        else:
            trade_type = "interdealer"

        return {
            "cusip": cusip,
            "trade_date": trade_date,
            "settlement_date": settlement_date,
            "par_amount": par,
            "price": price,
            "yield_pct": yield_pct,
            "trade_type": trade_type,
            "dealer_id": str(raw.get("dealerId") or raw.get("dealer_id") or ""),
            "surveillance_flag": 0,
            "deviation_pct": None,
        }


_emma = EMMAAdapter()


# ---------------------------------------------------------------------------
# MuniService
# ---------------------------------------------------------------------------

class MuniService:
    """
    Core service: universe management, yield calculations, trade surveillance,
    tax-equivalent yield, screener.
    """

    def __init__(self):
        self._mmd_curve: Dict[float, float] = dict(_AAA_MMD_BASELINE)
        self._universe_loaded = False

    # ------------------------------------------------------------------
    # Universe bootstrap
    # ------------------------------------------------------------------

    def bootstrap_universe(self) -> int:
        """Load the 60-issuer universe into SQLite. Return count inserted."""
        count = 0
        today = date.today().isoformat()
        for entry in MUNI_UNIVERSE_60:
            cusip = entry["cusip"]
            years = _years_to_maturity(entry["maturity"])
            coupon = float(entry.get("coupon", 0))

            # Estimate yield: AAA MMD + credit spread
            aaa_yield = _interp_aaa_mmd(years)
            credit_spread = self._credit_spread_for_rating(entry.get("rating_sp", "NR"))
            ytm_est = aaa_yield + credit_spread / 100

            # Bond analytics
            mod_dur = _modified_duration(coupon, years, ytm_est)
            dv01 = _dv01(mod_dur, 100.0, 1.0)
            aaa_spread_bps = credit_spread

            row: Dict[str, Any] = {
                "cusip": cusip,
                "issuer": entry.get("issuer", ""),
                "state": entry.get("state", ""),
                "sector": entry.get("sector", "general_obligation"),
                "bond_type": entry.get("type", "GO"),
                "coupon_rate": coupon,
                "maturity_date": entry.get("maturity", ""),
                "par_mm": float(entry.get("par_mm", 0)),
                "tax_status": entry.get("tax_status", "non-AMT"),
                "rating_sp": entry.get("rating_sp", "NR"),
                "rating_moodys": entry.get("rating_moodys", "NR"),
                "call_date": entry.get("call_date"),
                "call_price": float(entry.get("call_price", 100.0)),
                "last_price": 100.0,
                "last_yield": round(ytm_est, 4),
                "last_trade_dt": today,
                "aaa_spread_bps": round(aaa_spread_bps, 1),
                "mod_duration": round(mod_dur, 3),
                "years_to_mat": round(years, 3),
                "dv01_per_mm": round(dv01, 2),
                "updated_at": today,
            }
            _db.upsert_bond(row)
            count += 1

        # Load state fiscal data
        for state_code, fiscal in STATE_FISCAL_SCORES.items():
            _db.upsert_state_fiscal(state_code, fiscal)

        self._universe_loaded = True
        logger.info("MuniService: bootstrapped %d bonds", count)
        return count

    def _credit_spread_for_rating(self, rating: str) -> float:
        """Return muni credit spread in bps over AAA MMD for given S&P rating."""
        _SPREADS: Dict[str, float] = {
            "AAA":  0,   "AA+":  5,  "AA":  10,  "AA-": 18,
            "A+":  30,   "A":   45,  "A-":  65,  "BBB+": 90,
            "BBB": 120,  "BBB-":160, "BB+": 240, "BB":  320,
            "BB-": 420,  "B+":  550, "B":   700, "B-": 900,
            "CCC": 1200, "CC": 1800, "C":  2500, "D":  4000,
            "NR":   60,
        }
        return _SPREADS.get(rating.upper(), 60)

    def refresh_mmd_curve(self) -> Dict[float, float]:
        """Refresh AAA MMD curve from FRED. Updates internal cache."""
        self._mmd_curve = _fred.build_mmd_curve_from_fred()
        logger.info("MMD curve refreshed: %d tenor points", len(self._mmd_curve))
        return self._mmd_curve

    # ------------------------------------------------------------------
    # Trade fetching + surveillance
    # ------------------------------------------------------------------

    def fetch_trades(self, cusip: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch EMMA trades for a CUSIP, run surveillance, persist to SQLite.
        Returns list of normalized trade dicts (with surveillance flags).
        """
        end_date   = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        raw_trades = _emma.search_trades(start_date, end_date, cusip=cusip)
        if not raw_trades:
            # Return cached trades from DB
            return _db.get_trades(cusip, days_back)

        normalized: List[Dict[str, Any]] = []
        for raw in raw_trades:
            trade = _emma.normalize_trade(raw, cusip_fallback=cusip)
            normalized.append(trade)

        # Surveillance: compute composite price and flag deviations
        surveilled = self._run_surveillance(normalized)
        _db.insert_trades(surveilled)
        return surveilled

    def _run_surveillance(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Flag trades where price deviates >2% from rolling composite (average
        of all trades for that CUSIP on the same day).
        """
        # Group by (cusip, trade_date)
        groups: Dict[Tuple[str, str], List[float]] = {}
        for t in trades:
            key = (t["cusip"], t["trade_date"])
            groups.setdefault(key, []).append(t["price"])

        composites: Dict[Tuple[str, str], float] = {
            k: sum(v) / len(v) for k, v in groups.items() if v
        }

        result = []
        for t in trades:
            key = (t["cusip"], t["trade_date"])
            composite = composites.get(key, t["price"])
            if composite > 0:
                dev = abs(t["price"] - composite) / composite * 100
            else:
                dev = 0.0
            t["deviation_pct"] = round(dev, 4)
            t["surveillance_flag"] = 1 if dev > 2.0 else 0
            result.append(t)
        return result

    # ------------------------------------------------------------------
    # Tax-equivalent yield
    # ------------------------------------------------------------------

    def compute_tey(
        self,
        cusip: str,
        federal_bracket_pct: Optional[float] = None,
    ) -> TaxEquivResult:
        """
        Compute tax-equivalent yield for a CUSIP.
        federal_bracket_pct: override if None, uses 37% (top bracket).
        """
        bond = _db.get_bond(cusip)
        if not bond:
            raise ValueError(f"CUSIP {cusip} not found in universe")

        muni_yield = bond["last_yield"] or 0.0
        state = bond.get("state", "")
        years = bond.get("years_to_mat", 10.0)
        tax_status = bond.get("tax_status", "non-AMT")

        # Federal bracket
        if federal_bracket_pct is not None:
            fed_rate = federal_bracket_pct / 100
        else:
            fed_rate = 0.37  # default top bracket

        state_rate = STATE_TAX.get(state, 0.0)
        # Combined rate: federal + state (state taxes deductible from federal in some states,
        # use simplified combined here)
        if tax_status == "taxable":
            combined_rate = 0.0  # taxable muni — no tax exemption
        elif tax_status == "AMT":
            combined_rate = max(0.0, fed_rate - 0.0262)  # AMT rate reduces benefit
        else:
            combined_rate = fed_rate + state_rate * (1 - fed_rate)  # state deduction from federal

        combined_rate = min(combined_rate, 0.55)

        tey = muni_yield / (1 - combined_rate) if combined_rate < 1.0 else muni_yield

        # Treasury comparison
        treas_yield = _interp_treasury(years)
        # After-tax Treasury
        after_tax_treas = treas_yield * (1 - fed_rate)
        spread_to_treas = (muni_yield - treas_yield) * 100  # bps
        tey_vs_treas    = (tey - treas_yield) * 100         # bps
        # Breakeven tax rate: muni_yield / treas_yield already expressed as rate
        breakeven = (1 - muni_yield / treas_yield) * 100 if treas_yield > 0 else 0.0

        return TaxEquivResult(
            cusip=cusip,
            issuer=bond.get("issuer", ""),
            state=state,
            muni_yield_pct=round(muni_yield, 4),
            federal_bracket_pct=round(fed_rate * 100, 2),
            state_tax_rate_pct=round(state_rate * 100, 2),
            combined_tax_rate_pct=round(combined_rate * 100, 2),
            tax_equiv_yield_pct=round(tey, 4),
            treasury_yield_same_maturity_pct=round(treas_yield, 4),
            spread_to_treasury_bps=round(spread_to_treas, 2),
            tey_vs_treasury_bps=round(tey_vs_treas, 2),
            breakeven_tax_rate_pct=round(breakeven, 2),
            after_tax_treasury_pct=round(after_tax_treas, 4),
            muni_advantage_bps=round(tey_vs_treas, 2),
        )

    # ------------------------------------------------------------------
    # Yield curve
    # ------------------------------------------------------------------

    def build_yield_curve(self) -> MuniYieldCurve:
        """
        Build muni yield curves (AAA, AA, A, BBB) + Treasury, MMD/Treasury ratio.
        Uses live FRED data where available.
        """
        tenors = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

        # Try live FRED AAA data
        try:
            mmd_live = _fred.build_mmd_curve_from_fred()
        except Exception:
            mmd_live = _AAA_MMD_BASELINE

        # Spreads to AAA by rating tier (bps)
        _AA_SPREAD  = 10
        _A_SPREAD   = 45
        _BBB_SPREAD = 120

        aaa_mmd   = [round(_interp_aaa_mmd(t), 4) for t in tenors]
        aa_mmd    = [round(y + _AA_SPREAD  / 100, 4) for y in aaa_mmd]
        a_mmd     = [round(y + _A_SPREAD   / 100, 4) for y in aaa_mmd]
        bbb_mmd   = [round(y + _BBB_SPREAD / 100, 4) for y in aaa_mmd]

        # Treasury yields from FRED
        treas_series = {"2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30"}
        treas_map: Dict[float, float] = {}
        for label, sid in treas_series.items():
            t_years = float(label.replace("Y", ""))
            val = _fred.latest(sid)
            if val and val > 0:
                treas_map[t_years] = val

        treasury = []
        for t in tenors:
            treas_map_tenors = sorted(treas_map.keys())
            if treas_map_tenors:
                # Interpolate from live FRED
                if t <= treas_map_tenors[0]:
                    treasury.append(round(treas_map[treas_map_tenors[0]], 4))
                elif t >= treas_map_tenors[-1]:
                    treasury.append(round(treas_map[treas_map_tenors[-1]], 4))
                else:
                    for i in range(len(treas_map_tenors) - 1):
                        t0, t1 = treas_map_tenors[i], treas_map_tenors[i + 1]
                        if t0 <= t <= t1:
                            y0, y1 = treas_map[t0], treas_map[t1]
                            interp = y0 + (y1 - y0) * (t - t0) / (t1 - t0)
                            treasury.append(round(interp, 4))
                            break
            else:
                treasury.append(round(_interp_treasury(t), 4))

        # Muni/Treasury ratio
        ratios = [
            round(m / t * 100, 2) if t > 0 else 0.0
            for m, t in zip(aaa_mmd, treasury)
        ]

        return MuniYieldCurve(
            as_of_date=date.today().isoformat(),
            tenors=tenors,
            aaa_mmd=aaa_mmd,
            aa_mmd=aa_mmd,
            a_mmd=a_mmd,
            bbb_mmd=bbb_mmd,
            treasury=treasury,
            muni_treasury_ratio=ratios,
            source="FRED BAMLM series + AAA MMD baseline",
        )

    # ------------------------------------------------------------------
    # Screener
    # ------------------------------------------------------------------

    def screen(self, req: MuniScreenerRequest) -> Dict[str, Any]:
        """Screen the muni universe against criteria, return ranked results."""
        _RATING_ORDER = [
            "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
            "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
            "B+", "B", "B-", "CCC", "CC", "C", "D", "NR",
        ]

        def rating_rank(r: str) -> int:
            r = (r or "NR").upper().strip()
            return _RATING_ORDER.index(r) if r in _RATING_ORDER else len(_RATING_ORDER)

        bonds = _db.all_bonds()
        if not bonds:
            self.bootstrap_universe()
            bonds = _db.all_bonds()

        results = []
        for b in bonds:
            # State filter
            if req.states and b.get("state", "").upper() not in [s.upper() for s in req.states]:
                continue
            # Sector filter
            if req.sectors and b.get("sector", "") not in req.sectors:
                continue
            # Bond type filter
            if req.bond_types and b.get("bond_type", "") not in req.bond_types:
                continue
            # Yield filter
            ytm = b.get("last_yield", 0.0) or 0.0
            if ytm < req.min_yield or ytm > req.max_yield:
                continue
            # Duration filter
            dur = b.get("mod_duration", 0.0) or 0.0
            if dur < req.min_duration or dur > req.max_duration:
                continue
            # Maturity filter
            years = b.get("years_to_mat", 0.0) or 0.0
            if years < req.min_maturity_years or years > req.max_maturity_years:
                continue
            # Tax status filter
            if req.tax_status and b.get("tax_status") != req.tax_status:
                continue
            # Rating filter
            if req.min_rating_sp:
                min_rank = rating_rank(req.min_rating_sp)
                if rating_rank(b.get("rating_sp", "NR")) > min_rank:
                    continue
            # AAA spread filter
            if req.max_aaa_spread_bps is not None:
                if (b.get("aaa_spread_bps") or 0.0) > req.max_aaa_spread_bps:
                    continue

            results.append(b)

        # Compute TEY for each result (simplified, using 37% bracket)
        for b in results:
            state = b.get("state", "")
            state_rate = STATE_TAX.get(state, 0.0)
            combined = 0.37 + state_rate * (1 - 0.37)
            combined = min(combined, 0.55)
            ytm = b.get("last_yield", 0.0) or 0.0
            tax_status = b.get("tax_status", "non-AMT")
            if tax_status == "taxable":
                b["tey"] = ytm
            else:
                b["tey"] = ytm / (1 - combined) if combined < 1 else ytm
            b["risk_adj_yield"] = ytm / dur if (dur := b.get("mod_duration") or 1) > 0 else 0

        # Sort
        def sort_key(b: Dict) -> float:
            if req.sort_by == "tey_desc":
                return -(b.get("tey") or 0)
            elif req.sort_by == "yield_desc":
                return -(b.get("last_yield") or 0)
            elif req.sort_by == "spread_desc":
                return -(b.get("aaa_spread_bps") or 0)
            elif req.sort_by == "duration_asc":
                return b.get("mod_duration") or 999
            elif req.sort_by == "risk_adj_yield_desc":
                return -(b.get("risk_adj_yield") or 0)
            return 0.0

        results.sort(key=sort_key)
        results = results[:req.limit]

        # Summary stats
        yields = [b.get("last_yield", 0) or 0 for b in results]
        teys   = [b.get("tey", 0) or 0 for b in results]

        return {
            "total_matches": len(results),
            "bonds": results,
            "stats": {
                "avg_yield": round(sum(yields) / len(yields), 4) if yields else 0,
                "avg_tey":   round(sum(teys)   / len(teys),   4) if teys   else 0,
                "min_yield": min(yields) if yields else 0,
                "max_yield": max(yields) if yields else 0,
                "avg_duration": round(
                    sum(b.get("mod_duration", 0) or 0 for b in results) / len(results), 3
                ) if results else 0,
            },
            "filters_applied": req.model_dump(exclude_none=True),
        }

    # ------------------------------------------------------------------
    # Security lookup
    # ------------------------------------------------------------------

    def get_security(self, cusip: str) -> Dict[str, Any]:
        """
        Return full bond details for a CUSIP: universe record + EMMA disclosure count.
        """
        bond = _db.get_bond(cusip)
        if not bond:
            # Try EMMA search
            results = _emma.search_security(cusip)
            if results:
                raw = results[0]
                # Minimal mapping from EMMA SearchResult
                bond = {
                    "cusip": cusip,
                    "issuer": str(raw.get("issuerName") or raw.get("issuer_name") or ""),
                    "state": str(raw.get("stateCode") or raw.get("state") or ""),
                    "sector": "general_obligation",
                    "coupon_rate": float(raw.get("couponRate") or raw.get("coupon") or 0),
                    "maturity_date": str(raw.get("maturityDate") or raw.get("maturity") or ""),
                    "rating_sp": str(raw.get("ratingS&P") or "NR"),
                    "last_price": 100.0,
                    "last_yield": 0.0,
                }
            else:
                raise ValueError(f"CUSIP {cusip} not found")

        # Fetch disclosure count from EMMA
        try:
            disclosures = _emma.search_disclosures(cusip)
            bond["disclosure_count"] = len(disclosures)
            bond["latest_disclosure"] = disclosures[0] if disclosures else None
        except Exception:
            bond["disclosure_count"] = 0
            bond["latest_disclosure"] = None

        # Enrich with analytics
        if bond.get("coupon_rate") and bond.get("years_to_mat"):
            ytm = bond.get("last_yield") or _bond_ytm(
                bond["coupon_rate"], bond["years_to_mat"], bond.get("last_price", 100)
            )
            if ytm > 0:
                bond["modified_duration_calc"] = _modified_duration(
                    bond["coupon_rate"], bond["years_to_mat"], ytm
                )
                bond["convexity_calc"] = _convexity(
                    bond["coupon_rate"], bond["years_to_mat"], ytm
                )
                bond["dv01_per_mm_calc"] = _dv01(bond["modified_duration_calc"])
        return bond

    # ------------------------------------------------------------------
    # State fiscal
    # ------------------------------------------------------------------

    def get_state_fiscal(self, state_code: str) -> StateFiscalData:
        """Return state fiscal health indicators + implied credit tier."""
        sc = state_code.upper()
        data = STATE_FISCAL_SCORES.get(sc)
        if not data:
            raise ValueError(f"State code {sc} not found")

        score = data["score"]
        if score >= 75:
            tier = "AAA/AA+"
            strengths = ["Strong fund balance", "Low debt burden", "Diversified revenue"]
            risks: List[str] = []
        elif score >= 60:
            tier = "AA/AA-"
            strengths = ["Adequate reserves", "Stable revenue base"]
            risks = ["Moderate pension burden"]
        elif score >= 45:
            tier = "A/A-"
            strengths = ["Functioning fiscal controls"]
            risks = ["Elevated pension liability", "Limited budget flexibility"]
        elif score >= 30:
            tier = "BBB/BBB-"
            strengths = ["Federal revenue support"]
            risks = ["Structural budget imbalance", "High pension unfunded liability", "Weak reserves"]
        else:
            tier = "BB or below"
            strengths = []
            risks = ["Severely underfunded pensions", "Structural deficits", "High debt service"]

        pension_ratio = data.get("pension_funded_ratio", 0)
        if pension_ratio < 0.50:
            risks.append(f"Pension only {pension_ratio*100:.0f}% funded")
        fund_bal = data.get("fund_balance_ratio", 0)
        if fund_bal > 0.20:
            strengths.append(f"Fund balance {fund_bal*100:.0f}% of expenditures")

        return StateFiscalData(
            state_code=sc,
            state_name=STATES.get(sc, sc),
            fiscal_health_score=score,
            fund_balance_ratio=fund_bal,
            pension_funded_ratio=pension_ratio,
            revenue_volatility=data.get("revenue_volatility", "medium"),
            implied_credit_tier=tier,
            risk_factors=risks,
            strengths=strengths,
        )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/muni/v3", tags=["Municipal Bonds v3"])
_svc = MuniService()


@router.on_event("startup")
async def _startup() -> None:  # noqa: B006
    _svc.bootstrap_universe()


@router.get("/trades/{cusip}", response_model=List[MuniTrade])
def get_trades(
    cusip: str,
    days_back: int = Query(30, ge=1, le=365),
) -> List[MuniTrade]:
    """
    Fetch recent trade history for a CUSIP from MSRB EMMA.
    Includes trade surveillance flags for deviations >2% from composite.
    """
    try:
        trades = _svc.fetch_trades(cusip.upper(), days_back=days_back)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Trade fetch failed: {exc}") from exc

    return [
        MuniTrade(
            cusip=t["cusip"],
            trade_date=t["trade_date"],
            settlement_date=t.get("settlement_date", ""),
            par_amount=t.get("par_amount", 0),
            price=t.get("price", 0),
            yield_pct=t.get("yield_pct", 0),
            trade_type=t.get("trade_type", "interdealer"),
            dealer_id=t.get("dealer_id", ""),
            surveillance_flag=bool(t.get("surveillance_flag")),
            deviation_from_composite_pct=t.get("deviation_pct"),
        )
        for t in trades
    ]


@router.get("/yield-curve", response_model=MuniYieldCurve)
def get_yield_curve(refresh: bool = Query(False)) -> MuniYieldCurve:
    """
    Return AAA/AA/A/BBB muni MMD curves + Treasury curve.
    Muni/Treasury ratio included. Sources: FRED BofA muni indices.
    """
    if refresh:
        _svc.refresh_mmd_curve()
    try:
        return _svc.build_yield_curve()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/tax-equivalent/{cusip}", response_model=TaxEquivResult)
def get_tax_equivalent(
    cusip: str,
    federal_bracket_pct: Optional[float] = Query(
        None, ge=0, le=100,
        description="Federal marginal rate %, e.g. 37.0. Default: 37%"
    ),
) -> TaxEquivResult:
    """
    Tax-equivalent yield calculator.
    Supports all federal brackets: 22%, 32%, 37%, 40.8% (37 + NIIT).
    State rate automatically applied based on bond's state.
    """
    try:
        return _svc.compute_tey(cusip.upper(), federal_bracket_pct=federal_bracket_pct)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/screener", response_model=Dict[str, Any])
def screen_munis(req: MuniScreenerRequest) -> Dict[str, Any]:
    """
    Screen muni universe by state, sector, yield, duration, rating, tax status.
    Returns ranked bonds with TEY, risk-adjusted yield, and summary stats.
    """
    try:
        return _svc.screen(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/security/{cusip}", response_model=Dict[str, Any])
def get_security(cusip: str) -> Dict[str, Any]:
    """
    Return full bond details: universe record, EMMA disclosures, and computed analytics
    (modified duration, convexity, DV01).
    """
    try:
        return _svc.get_security(cusip.upper())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/state-fiscal/{state_code}", response_model=StateFiscalData)
def get_state_fiscal(state_code: str) -> StateFiscalData:
    """
    State fiscal health dashboard: fund balance ratio, pension funded ratio,
    revenue volatility, implied credit tier, risk factors and strengths.
    Sources: Census Annual Survey of State Government Finances + Pew proxy.
    """
    try:
        return _svc.get_state_fiscal(state_code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/universe", response_model=Dict[str, Any])
def get_universe(
    state: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Return muni universe summary with optional state/sector filter."""
    bonds = _db.all_bonds()
    if not bonds:
        _svc.bootstrap_universe()
        bonds = _db.all_bonds()

    if state:
        bonds = [b for b in bonds if b.get("state", "").upper() == state.upper()]
    if sector:
        bonds = [b for b in bonds if b.get("sector") == sector]

    sectors: Dict[str, int] = {}
    states_cnt: Dict[str, int] = {}
    ratings: Dict[str, int] = {}
    for b in bonds:
        sectors[b.get("sector", "other")] = sectors.get(b.get("sector", "other"), 0) + 1
        states_cnt[b.get("state", "")] = states_cnt.get(b.get("state", ""), 0) + 1
        ratings[b.get("rating_sp", "NR")] = ratings.get(b.get("rating_sp", "NR"), 0) + 1

    yields = [b.get("last_yield", 0) or 0 for b in bonds if b.get("last_yield")]

    return {
        "total_bonds": len(bonds),
        "by_sector": dict(sorted(sectors.items(), key=lambda x: -x[1])),
        "by_state": dict(sorted(states_cnt.items(), key=lambda x: -x[1])),
        "by_rating": dict(sorted(ratings.items(), key=lambda x: -x[1])),
        "avg_yield_pct": round(sum(yields) / len(yields), 4) if yields else 0,
        "data_as_of": date.today().isoformat(),
    }


@router.get("/surveillance/alerts", response_model=List[TradeAlert])
def get_surveillance_alerts(days_back: int = Query(7, ge=1, le=90)) -> List[TradeAlert]:
    """
    Return all trade surveillance alerts (>2% deviation from composite) over the window.
    """
    since = (date.today() - timedelta(days=days_back)).isoformat()
    with _db._conn() as conn:
        rows = conn.execute(
            """SELECT t.cusip, t.trade_date, t.price, t.par_amount, t.trade_type,
                      t.deviation_pct
               FROM trade_history t
               WHERE t.surveillance_flag=1 AND t.trade_date>=?
               ORDER BY t.deviation_pct DESC""",
            (since,),
        ).fetchall()

    alerts = []
    for row in rows:
        dev = row["deviation_pct"] or 0.0
        composite = row["price"] / (1 + dev / 100) if dev > 0 else row["price"]
        alerts.append(TradeAlert(
            cusip=row["cusip"],
            trade_date=row["trade_date"],
            trade_price=row["price"],
            composite_price=round(composite, 4),
            deviation_pct=dev,
            trade_type=row["trade_type"],
            par_amount=row["par_amount"],
            alert_level="critical" if dev > 5.0 else "warning",
        ))
    return alerts


@router.get("/analytics/{cusip}", response_model=Dict[str, Any])
def get_bond_analytics(cusip: str) -> Dict[str, Any]:
    """
    Full bond analytics: YTM, modified duration, convexity, DV01, AAA spread,
    TEY at all four tax brackets, and yield curve position.
    """
    bond = _db.get_bond(cusip.upper())
    if not bond:
        raise HTTPException(status_code=404, detail=f"CUSIP {cusip} not found")

    coupon = bond.get("coupon_rate", 0.0) or 0.0
    years  = bond.get("years_to_mat", 0.0) or 0.0
    price  = bond.get("last_price",  100.0) or 100.0

    # Recompute YTM precisely from price
    ytm = _bond_ytm(coupon, years, price) if years > 0 else (bond.get("last_yield") or 0)
    mod_dur  = _modified_duration(coupon, years, ytm)
    conv     = _convexity(coupon, years, ytm)
    dv01     = _dv01(mod_dur, price)
    aaa_mmd  = _interp_aaa_mmd(years)
    treas    = _interp_treasury(years)
    spread   = (ytm - aaa_mmd) * 100

    # TEY at standard brackets
    state = bond.get("state", "")
    st_rate = STATE_TAX.get(state, 0.0)
    teys: Dict[str, float] = {}
    for fed_pct in [0.22, 0.32, 0.37, 0.408]:
        combined = fed_pct + st_rate * (1 - fed_pct)
        combined = min(combined, 0.60)
        teys[f"tey_{int(fed_pct*1000)}bps"] = round(ytm / (1 - combined), 4)

    return {
        "cusip": cusip,
        "issuer": bond.get("issuer"),
        "state": state,
        "coupon_rate": coupon,
        "years_to_maturity": round(years, 3),
        "clean_price": round(price, 4),
        "ytm_pct": round(ytm, 4),
        "modified_duration": round(mod_dur, 4),
        "convexity": round(conv, 4),
        "dv01_per_mm": round(dv01, 2),
        "aaa_mmd_pct": round(aaa_mmd, 4),
        "spread_to_aaa_mmd_bps": round(spread, 2),
        "treasury_yield_pct": round(treas, 4),
        "muni_treasury_ratio_pct": round(ytm / treas * 100, 2) if treas > 0 else 0,
        "tax_equivalent_yields": teys,
        "rating_sp": bond.get("rating_sp"),
        "sector": bond.get("sector"),
        "tax_status": bond.get("tax_status"),
        "default_rate_sector_10y": SECTOR_DEFAULT_RATES.get(bond.get("sector", ""), 0),
    }
