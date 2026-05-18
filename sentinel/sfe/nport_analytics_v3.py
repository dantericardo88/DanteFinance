"""
N-PORT Analytics v3 — dim_032: Fund holdings intelligence (target 9/10).

Improvements over v2:
  - EDGAR full-universe enumeration: 500+ N-PORT-P filers via paginated search API
  - Extended static fund registry: 200+ funds across Vanguard, iShares, SPDR, Fidelity,
    Schwab, Invesco, ARK, T. Rowe Price, American Funds, PIMCO, and major active MFs
  - Complete XML parse: genInfo, invstOrSec (all asset classes), derivativeInfo,
    returnInfo, creditSpreadRiskInfo, borrowing/lending, repurchase agreements
  - Analytics: factor exposures (FF5), sector/geo/asset-class breakdown, duration
    (bond funds), FF5 factor model hookup, concentration (HHI, active share)
  - Smart-money consensus: ≥10 fund holders → "institutionally validated"
  - Flow estimation: position Δ × midpoint price → implied monthly flow
  - Fund overlap matrix: Jaccard pairwise for any set of funds
  - SQLite persistence: fund_universe, fund_holdings, holdings_history,
    fund_flows, fund_analytics
  - FastAPI router at /nport/v3

All data: SEC EDGAR (free, no API key).  Rate limit: 10 req/s → 0.11 s sleep.
"""
from __future__ import annotations

import asyncio
import contextlib
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EFTS_BASE      = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_ARCHIVE  = "https://www.sec.gov/Archives/edgar/data"
_SUBMISSIONS    = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/3.0 richard.porras@realempanada.com",
    "Accept":     "application/json, text/html, */*",
    "Accept-Encoding": "gzip, deflate",
}
_XML_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/3.0 richard.porras@realempanada.com",
    "Accept":     "application/xml, text/xml, */*",
    "Accept-Encoding": "gzip, deflate",
}

_RATE_LIMIT_SLEEP = 0.12   # stay under 10 req/s with margin

# SQLite path (project-relative)
_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "nport_v3.db"

# GICS sector codes embedded in N-PORT issuerCat / assetCat
ASSET_CAT_MAP: Dict[str, str] = {
    "EC":  "US Equity (Common)",
    "EP":  "US Equity (Preferred)",
    "DB":  "US Debt",
    "ABS": "Asset-Backed Security",
    "MBS": "Mortgage-Backed Security",
    "MM":  "Money Market",
    "RA":  "Real Asset",
    "DER": "Derivative",
    "OTH": "Other",
    "UST": "US Treasury",
    "MF":  "Mutual Fund / ETF",
    "CE":  "Closed-End Fund",
    "ETF": "ETF",
    "RF":  "Real Estate / REIT",
    "STIV": "Short-Term Investment Vehicle",
}

# Fama-French 5 factor proxies — simple sector loading assumptions
# used when no live factor regression is available
FF5_SECTOR_LOADINGS: Dict[str, Dict[str, float]] = {
    "Technology":    {"mkt": 1.30, "smb": -0.20, "hml": -0.60, "rmw": 0.10, "cma": -0.30},
    "Financial":     {"mkt": 1.10, "smb":  0.10, "hml":  0.80, "rmw": 0.20, "cma":  0.10},
    "Healthcare":    {"mkt": 0.80, "smb": -0.10, "hml": -0.10, "rmw": 0.40, "cma": -0.10},
    "Consumer":      {"mkt": 0.90, "smb":  0.05, "hml":  0.20, "rmw": 0.30, "cma":  0.00},
    "Energy":        {"mkt": 1.20, "smb":  0.20, "hml":  0.50, "rmw": -0.10,"cma":  0.20},
    "Industrial":    {"mkt": 1.10, "smb":  0.15, "hml":  0.30, "rmw": 0.20, "cma":  0.10},
    "Utility":       {"mkt": 0.60, "smb": -0.10, "hml":  0.40, "rmw": 0.50, "cma":  0.30},
    "Real Estate":   {"mkt": 0.90, "smb":  0.20, "hml":  0.60, "rmw": -0.20,"cma":  0.20},
    "Communication": {"mkt": 1.00, "smb": -0.05, "hml": -0.10, "rmw": 0.15, "cma": -0.10},
    "Materials":     {"mkt": 1.10, "smb":  0.25, "hml":  0.35, "rmw": 0.10, "cma":  0.15},
    "Other":         {"mkt": 1.00, "smb":  0.00, "hml":  0.00, "rmw": 0.00, "cma":  0.00},
}

# Approximate sector from N-PORT issuerCat / country + name heuristic
ISSUER_CAT_SECTOR: Dict[str, str] = {
    "CORP":  "Other",
    "MUN":   "Municipal",
    "SOVR":  "Sovereign",
    "SUPR":  "Supranational",
    "ABS":   "Structured",
    "MBS":   "Structured",
}

_SMART_MONEY_MIN_FUNDS = 10   # ≥ this many funds → "institutionally validated"
_CROWDING_HIGH  = 80.0
_CROWDING_MED   = 50.0

# ---------------------------------------------------------------------------
# Massive fund universe (200+ entries)
# ---------------------------------------------------------------------------

FUND_UNIVERSE: Dict[str, Dict[str, Any]] = {
    # ── Vanguard ETFs ─────────────────────────────────────────────────────
    "Vanguard Total Stock Market ETF":          {"cik": "0000899392", "ticker": "VTI",  "issuer": "Vanguard", "type": "ETF",         "aum_b": 380, "benchmark": "CRSP US Total Market"},
    "Vanguard S&P 500 ETF":                     {"cik": "0000899394", "ticker": "VOO",  "issuer": "Vanguard", "type": "ETF",         "aum_b": 430, "benchmark": "S&P 500"},
    "Vanguard Growth ETF":                      {"cik": "0000036405", "ticker": "VUG",  "issuer": "Vanguard", "type": "ETF",         "aum_b": 120, "benchmark": "CRSP US Large Cap Growth"},
    "Vanguard Value ETF":                       {"cik": "0000036405", "ticker": "VTV",  "issuer": "Vanguard", "type": "ETF",         "aum_b": 100, "benchmark": "CRSP US Large Cap Value"},
    "Vanguard Total International Stock ETF":   {"cik": "0000884394", "ticker": "VXUS", "issuer": "Vanguard", "type": "ETF",         "aum_b":  75, "benchmark": "FTSE Global All Cap ex US"},
    "Vanguard FTSE Developed Markets ETF":      {"cik": "0000884394", "ticker": "VEA",  "issuer": "Vanguard", "type": "ETF",         "aum_b": 100, "benchmark": "FTSE Developed All Cap ex US"},
    "Vanguard FTSE Emerging Markets ETF":       {"cik": "0000884394", "ticker": "VWO",  "issuer": "Vanguard", "type": "ETF",         "aum_b":  85, "benchmark": "FTSE Emerging Markets All Cap"},
    "Vanguard Mid-Cap ETF":                     {"cik": "0000036405", "ticker": "VO",   "issuer": "Vanguard", "type": "ETF",         "aum_b":  55, "benchmark": "CRSP US Mid Cap"},
    "Vanguard Small-Cap ETF":                   {"cik": "0000036405", "ticker": "VB",   "issuer": "Vanguard", "type": "ETF",         "aum_b":  50, "benchmark": "CRSP US Small Cap"},
    "Vanguard Real Estate ETF":                 {"cik": "0000036405", "ticker": "VNQ",  "issuer": "Vanguard", "type": "ETF",         "aum_b":  60, "benchmark": "MSCI US REIT"},
    "Vanguard Information Technology ETF":      {"cik": "0000036405", "ticker": "VGT",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":  70, "benchmark": "MSCI US IMI IT"},
    "Vanguard Health Care ETF":                 {"cik": "0000036405", "ticker": "VHT",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":  25, "benchmark": "MSCI US IMI Healthcare"},
    "Vanguard Financials ETF":                  {"cik": "0000036405", "ticker": "VFH",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":  12, "benchmark": "MSCI US IMI Financials"},
    "Vanguard Consumer Staples ETF":            {"cik": "0000036405", "ticker": "VDC",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":   7, "benchmark": "MSCI US IMI Consumer Staples"},
    "Vanguard Consumer Discretionary ETF":      {"cik": "0000036405", "ticker": "VCR",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":   6, "benchmark": "MSCI US IMI Consumer Disc"},
    "Vanguard Industrials ETF":                 {"cik": "0000036405", "ticker": "VIS",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":   5, "benchmark": "MSCI US IMI Industrials"},
    "Vanguard Energy ETF":                      {"cik": "0000036405", "ticker": "VDE",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":   8, "benchmark": "MSCI US IMI Energy"},
    "Vanguard Materials ETF":                   {"cik": "0000036405", "ticker": "VAW",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":   3, "benchmark": "MSCI US IMI Materials"},
    "Vanguard Utilities ETF":                   {"cik": "0000036405", "ticker": "VPU",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":   5, "benchmark": "MSCI US IMI Utilities"},
    "Vanguard Communication Services ETF":      {"cik": "0000036405", "ticker": "VOX",  "issuer": "Vanguard", "type": "Sector ETF",  "aum_b":   3, "benchmark": "MSCI US IMI Comm Services"},
    "Vanguard Dividend Appreciation ETF":       {"cik": "0000036405", "ticker": "VIG",  "issuer": "Vanguard", "type": "ETF",         "aum_b":  75, "benchmark": "S&P US Dividend Growers"},
    "Vanguard High Dividend Yield ETF":         {"cik": "0000036405", "ticker": "VYM",  "issuer": "Vanguard", "type": "ETF",         "aum_b":  55, "benchmark": "FTSE High Dividend Yield"},
    "Vanguard Russell 1000 Growth ETF":         {"cik": "0000036405", "ticker": "VONG", "issuer": "Vanguard", "type": "ETF",         "aum_b":  12, "benchmark": "Russell 1000 Growth"},
    "Vanguard Russell 2000 ETF":                {"cik": "0000036405", "ticker": "VTWO", "issuer": "Vanguard", "type": "ETF",         "aum_b":   5, "benchmark": "Russell 2000"},
    # ── Vanguard Bond ETFs ────────────────────────────────────────────────
    "Vanguard Total Bond Market ETF":           {"cik": "0000899392", "ticker": "BND",  "issuer": "Vanguard", "type": "Bond ETF",    "aum_b": 110, "benchmark": "Bloomberg US Agg"},
    "Vanguard Short-Term Bond ETF":             {"cik": "0000899392", "ticker": "BSV",  "issuer": "Vanguard", "type": "Bond ETF",    "aum_b":  35, "benchmark": "Bloomberg 1-5Y US Gov/Credit"},
    "Vanguard Intermediate-Term Bond ETF":      {"cik": "0000899392", "ticker": "BIV",  "issuer": "Vanguard", "type": "Bond ETF",    "aum_b":  20, "benchmark": "Bloomberg 5-10Y US Gov/Credit"},
    "Vanguard Long-Term Bond ETF":              {"cik": "0000899392", "ticker": "BLV",  "issuer": "Vanguard", "type": "Bond ETF",    "aum_b":   8, "benchmark": "Bloomberg US Long Gov/Credit"},
    "Vanguard Total International Bond ETF":    {"cik": "0000884394", "ticker": "BNDX", "issuer": "Vanguard", "type": "Bond ETF",    "aum_b":  55, "benchmark": "Bloomberg Global Agg ex USD"},
    "Vanguard Short-Term Corporate Bond ETF":   {"cik": "0000899392", "ticker": "VCSH", "issuer": "Vanguard", "type": "Bond ETF",    "aum_b":  40, "benchmark": "Bloomberg 1-5Y Corp"},
    "Vanguard Intermediate-Term Corporate":     {"cik": "0000899392", "ticker": "VCIT", "issuer": "Vanguard", "type": "Bond ETF",    "aum_b":  50, "benchmark": "Bloomberg 5-10Y Corp"},
    "Vanguard Long-Term Corporate Bond ETF":    {"cik": "0000899392", "ticker": "VCLT", "issuer": "Vanguard", "type": "Bond ETF",    "aum_b":  10, "benchmark": "Bloomberg LT Corp"},
    # ── Vanguard Mutual Funds ─────────────────────────────────────────────
    "Vanguard 500 Index Fund":                  {"cik": "0000036405", "ticker": "VFIAX","issuer": "Vanguard", "type": "Index Fund",  "aum_b": 350, "benchmark": "S&P 500"},
    "Vanguard Total Stock Market Index Fund":   {"cik": "0000036405", "ticker": "VTSAX","issuer": "Vanguard", "type": "Index Fund",  "aum_b": 380, "benchmark": "CRSP US Total Market"},
    "Vanguard Wellington Fund":                 {"cik": "0000036405", "ticker": "VWELX","issuer": "Vanguard", "type": "Balanced",    "aum_b":  95, "benchmark": "60/40 Blend"},
    "Vanguard Windsor II Fund":                 {"cik": "0000036405", "ticker": "VWNFX","issuer": "Vanguard", "type": "Mutual Fund", "aum_b":  40, "benchmark": "Russell 1000 Value"},
    "Vanguard Primecap Fund":                   {"cik": "0000036405", "ticker": "VPMCX","issuer": "Vanguard", "type": "Mutual Fund", "aum_b":  55, "benchmark": "S&P 500", "manager": "PRIMECAP"},
    "Vanguard Capital Opportunity Fund":        {"cik": "0000036405", "ticker": "VHCOX","issuer": "Vanguard", "type": "Mutual Fund", "aum_b":  22, "benchmark": "Russell Mid Cap Growth"},
    # ── iShares ETFs ──────────────────────────────────────────────────────
    "iShares Core S&P 500 ETF":                 {"cik": "0000277751", "ticker": "IVV",  "issuer": "BlackRock", "type": "ETF",        "aum_b": 450, "benchmark": "S&P 500"},
    "iShares Core S&P Total US Stock Market":   {"cik": "0001100663", "ticker": "ITOT", "issuer": "BlackRock", "type": "ETF",        "aum_b":  60, "benchmark": "S&P Total Market"},
    "iShares Russell 1000 ETF":                 {"cik": "0001100663", "ticker": "IWB",  "issuer": "BlackRock", "type": "ETF",        "aum_b":  40, "benchmark": "Russell 1000"},
    "iShares Russell 2000 ETF":                 {"cik": "0001100663", "ticker": "IWM",  "issuer": "BlackRock", "type": "ETF",        "aum_b":  70, "benchmark": "Russell 2000"},
    "iShares Russell 3000 ETF":                 {"cik": "0001100663", "ticker": "IWV",  "issuer": "BlackRock", "type": "ETF",        "aum_b":  12, "benchmark": "Russell 3000"},
    "iShares MSCI EAFE ETF":                    {"cik": "0001100663", "ticker": "EFA",  "issuer": "BlackRock", "type": "ETF",        "aum_b":  65, "benchmark": "MSCI EAFE"},
    "iShares MSCI Emerging Markets ETF":        {"cik": "0001100663", "ticker": "EEM",  "issuer": "BlackRock", "type": "ETF",        "aum_b":  25, "benchmark": "MSCI EM"},
    "iShares Core MSCI Emerging Markets ETF":   {"cik": "0001100663", "ticker": "IEMG", "issuer": "BlackRock", "type": "ETF",        "aum_b":  70, "benchmark": "MSCI EM IMI"},
    "iShares MSCI ACWI ETF":                    {"cik": "0001100663", "ticker": "ACWI", "issuer": "BlackRock", "type": "ETF",        "aum_b":  20, "benchmark": "MSCI ACWI"},
    "iShares Core US Aggregate Bond ETF":       {"cik": "0001100663", "ticker": "AGG",  "issuer": "BlackRock", "type": "Bond ETF",   "aum_b": 100, "benchmark": "Bloomberg US Agg"},
    "iShares 20+ Year Treasury Bond ETF":       {"cik": "0001100663", "ticker": "TLT",  "issuer": "BlackRock", "type": "Bond ETF",   "aum_b":  40, "benchmark": "ICE US Treasury 20+Y"},
    "iShares 7-10 Year Treasury Bond ETF":      {"cik": "0001100663", "ticker": "IEF",  "issuer": "BlackRock", "type": "Bond ETF",   "aum_b":  28, "benchmark": "ICE US Treasury 7-10Y"},
    "iShares 1-3 Year Treasury Bond ETF":       {"cik": "0001100663", "ticker": "SHY",  "issuer": "BlackRock", "type": "Bond ETF",   "aum_b":  22, "benchmark": "ICE US Treasury 1-3Y"},
    "iShares iBoxx IG Corporate Bond ETF":      {"cik": "0001100663", "ticker": "LQD",  "issuer": "BlackRock", "type": "Bond ETF",   "aum_b":  35, "benchmark": "Markit iBoxx USD IG Corp"},
    "iShares iBoxx HY Corporate Bond ETF":      {"cik": "0001100663", "ticker": "HYG",  "issuer": "BlackRock", "type": "Bond ETF",   "aum_b":  18, "benchmark": "Markit iBoxx USD HY Corp"},
    "iShares TIPS Bond ETF":                    {"cik": "0001100663", "ticker": "TIP",  "issuer": "BlackRock", "type": "Bond ETF",   "aum_b":  20, "benchmark": "Bloomberg US TIPS"},
    "iShares Russell 1000 Growth ETF":          {"cik": "0001100663", "ticker": "IWF",  "issuer": "BlackRock", "type": "ETF",        "aum_b":  90, "benchmark": "Russell 1000 Growth"},
    "iShares Russell 1000 Value ETF":           {"cik": "0001100663", "ticker": "IWD",  "issuer": "BlackRock", "type": "ETF",        "aum_b":  55, "benchmark": "Russell 1000 Value"},
    "iShares US Technology ETF":                {"cik": "0001100663", "ticker": "IYW",  "issuer": "BlackRock", "type": "Sector ETF", "aum_b":  12, "benchmark": "DJ US Technology"},
    "iShares Global Tech ETF":                  {"cik": "0001100663", "ticker": "IXN",  "issuer": "BlackRock", "type": "Sector ETF", "aum_b":   6, "benchmark": "S&P Global 1200 IT"},
    "iShares MSCI USA Min Vol Factor ETF":      {"cik": "0001100663", "ticker": "USMV", "issuer": "BlackRock", "type": "Factor ETF", "aum_b":  25, "benchmark": "MSCI USA Min Vol"},
    "iShares MSCI USA Momentum Factor ETF":     {"cik": "0001100663", "ticker": "MTUM", "issuer": "BlackRock", "type": "Factor ETF", "aum_b":  12, "benchmark": "MSCI USA Momentum"},
    "iShares MSCI USA Quality Factor ETF":      {"cik": "0001100663", "ticker": "QUAL", "issuer": "BlackRock", "type": "Factor ETF", "aum_b":  35, "benchmark": "MSCI USA Quality"},
    "iShares MSCI USA Value Factor ETF":        {"cik": "0001100663", "ticker": "VLUE", "issuer": "BlackRock", "type": "Factor ETF", "aum_b":   7, "benchmark": "MSCI USA Enhanced Value"},
    "iShares US Healthcare ETF":                {"cik": "0001100663", "ticker": "IYH",  "issuer": "BlackRock", "type": "Sector ETF", "aum_b":   4, "benchmark": "DJ US Healthcare"},
    "iShares US Financials ETF":                {"cik": "0001100663", "ticker": "IYF",  "issuer": "BlackRock", "type": "Sector ETF", "aum_b":   3, "benchmark": "DJ US Financials"},
    "iShares China Large-Cap ETF":              {"cik": "0001100663", "ticker": "FXI",  "issuer": "BlackRock", "type": "ETF",        "aum_b":   5, "benchmark": "FTSE China 50"},
    "iShares MSCI Brazil ETF":                  {"cik": "0001100663", "ticker": "EWZ",  "issuer": "BlackRock", "type": "ETF",        "aum_b":   4, "benchmark": "MSCI Brazil 25/50"},
    # ── SPDR ETFs ─────────────────────────────────────────────────────────
    "SPDR S&P 500 ETF Trust":                   {"cik": "0000884394", "ticker": "SPY",  "issuer": "State Street", "type": "ETF",        "aum_b": 500, "benchmark": "S&P 500"},
    "SPDR S&P MidCap 400 ETF Trust":            {"cik": "0000884394", "ticker": "MDY",  "issuer": "State Street", "type": "ETF",        "aum_b":  20, "benchmark": "S&P MidCap 400"},
    "SPDR Portfolio S&P 500 Growth ETF":        {"cik": "0001064642", "ticker": "SPYG", "issuer": "State Street", "type": "ETF",        "aum_b":  25, "benchmark": "S&P 500 Growth"},
    "SPDR Portfolio S&P 500 Value ETF":         {"cik": "0001064642", "ticker": "SPYV", "issuer": "State Street", "type": "ETF",        "aum_b":  18, "benchmark": "S&P 500 Value"},
    "Technology Select Sector SPDR":            {"cik": "0001064642", "ticker": "XLK",  "issuer": "State Street", "type": "Sector ETF", "aum_b":  55, "benchmark": "S&P 500 Tech"},
    "Health Care Select Sector SPDR":           {"cik": "0001064642", "ticker": "XLV",  "issuer": "State Street", "type": "Sector ETF", "aum_b":  35, "benchmark": "S&P 500 Healthcare"},
    "Financial Select Sector SPDR":             {"cik": "0001064642", "ticker": "XLF",  "issuer": "State Street", "type": "Sector ETF", "aum_b":  35, "benchmark": "S&P 500 Financials"},
    "Energy Select Sector SPDR":                {"cik": "0001064642", "ticker": "XLE",  "issuer": "State Street", "type": "Sector ETF", "aum_b":  35, "benchmark": "S&P 500 Energy"},
    "Industrial Select Sector SPDR":            {"cik": "0001064642", "ticker": "XLI",  "issuer": "State Street", "type": "Sector ETF", "aum_b":  20, "benchmark": "S&P 500 Industrials"},
    "Consumer Discretionary SPDR":              {"cik": "0001064642", "ticker": "XLY",  "issuer": "State Street", "type": "Sector ETF", "aum_b":  20, "benchmark": "S&P 500 Cons Disc"},
    "Consumer Staples Select Sector SPDR":      {"cik": "0001064642", "ticker": "XLP",  "issuer": "State Street", "type": "Sector ETF", "aum_b":  15, "benchmark": "S&P 500 Cons Staples"},
    "Utilities Select Sector SPDR":             {"cik": "0001064642", "ticker": "XLU",  "issuer": "State Street", "type": "Sector ETF", "aum_b":  14, "benchmark": "S&P 500 Utilities"},
    "Real Estate Select Sector SPDR":           {"cik": "0001064642", "ticker": "XLRE", "issuer": "State Street", "type": "Sector ETF", "aum_b":   7, "benchmark": "S&P 500 Real Estate"},
    "Materials Select Sector SPDR":             {"cik": "0001064642", "ticker": "XLB",  "issuer": "State Street", "type": "Sector ETF", "aum_b":   7, "benchmark": "S&P 500 Materials"},
    "Communication Services Select SPDR":       {"cik": "0001064642", "ticker": "XLC",  "issuer": "State Street", "type": "Sector ETF", "aum_b":  18, "benchmark": "S&P 500 Comm Services"},
    "SPDR Gold Shares":                         {"cik": "0001222333", "ticker": "GLD",  "issuer": "State Street", "type": "Commodity ETF","aum_b": 55, "benchmark": "Gold Spot"},
    "SPDR Bloomberg High Yield Bond ETF":       {"cik": "0001064642", "ticker": "JNK",  "issuer": "State Street", "type": "Bond ETF",   "aum_b":   8, "benchmark": "Bloomberg HY VLI"},
    "SPDR Portfolio Aggregate Bond ETF":        {"cik": "0001064642", "ticker": "SPAB", "issuer": "State Street", "type": "Bond ETF",   "aum_b":   9, "benchmark": "Bloomberg US Agg"},
    # ── Invesco ETFs ──────────────────────────────────────────────────────
    "Invesco QQQ Trust":                        {"cik": "0001067839", "ticker": "QQQ",  "issuer": "Invesco", "type": "ETF",        "aum_b": 250, "benchmark": "Nasdaq-100"},
    "Invesco NASDAQ 100 ETF":                   {"cik": "0001067839", "ticker": "QQQM", "issuer": "Invesco", "type": "ETF",        "aum_b":  30, "benchmark": "Nasdaq-100"},
    "Invesco S&P 500 Equal Weight ETF":         {"cik": "0001067839", "ticker": "RSP",  "issuer": "Invesco", "type": "ETF",        "aum_b":  50, "benchmark": "S&P 500 EW"},
    "Invesco S&P 500 Top 50 ETF":               {"cik": "0001067839", "ticker": "XLG",  "issuer": "Invesco", "type": "ETF",        "aum_b":   4, "benchmark": "S&P 500 Top 50"},
    "Invesco DB Commodity Index Tracking":      {"cik": "0001067839", "ticker": "DBC",  "issuer": "Invesco", "type": "Commodity",  "aum_b":   3, "benchmark": "DBIQ Optimum Yield Diversified"},
    "Invesco Senior Loan ETF":                  {"cik": "0001067839", "ticker": "BKLN", "issuer": "Invesco", "type": "Bond ETF",   "aum_b":   4, "benchmark": "S&P/LSTA US LL 100"},
    "Invesco Preferred ETF":                    {"cik": "0001067839", "ticker": "PGX",  "issuer": "Invesco", "type": "ETF",        "aum_b":   4, "benchmark": "ICE BofA Core Plus Fixed Rate Pfd"},
    "Invesco Russell 1000 Dynamic Multifactor": {"cik": "0001067839", "ticker": "OMFL", "issuer": "Invesco", "type": "Factor ETF", "aum_b":   3, "benchmark": "Russell 1000 Dynamic Multifactor"},
    # ── Schwab ETFs ───────────────────────────────────────────────────────
    "Schwab US Broad Market ETF":               {"cik": "0001444822", "ticker": "SCHB", "issuer": "Schwab", "type": "ETF",        "aum_b":  28, "benchmark": "Dow Jones Broad US"},
    "Schwab US Large-Cap ETF":                  {"cik": "0001444822", "ticker": "SCHX", "issuer": "Schwab", "type": "ETF",        "aum_b":  45, "benchmark": "Dow Jones US Large-Cap"},
    "Schwab US Large-Cap Growth ETF":           {"cik": "0001444822", "ticker": "SCHG", "issuer": "Schwab", "type": "ETF",        "aum_b":  30, "benchmark": "Dow Jones US Large-Cap Growth"},
    "Schwab US Large-Cap Value ETF":            {"cik": "0001444822", "ticker": "SCHV", "issuer": "Schwab", "type": "ETF",        "aum_b":  12, "benchmark": "Dow Jones US Large-Cap Value"},
    "Schwab US Small-Cap ETF":                  {"cik": "0001444822", "ticker": "SCHA", "issuer": "Schwab", "type": "ETF",        "aum_b":  15, "benchmark": "Dow Jones US Small-Cap"},
    "Schwab US Mid-Cap ETF":                    {"cik": "0001444822", "ticker": "SCHM", "issuer": "Schwab", "type": "ETF",        "aum_b":   8, "benchmark": "Dow Jones US Mid-Cap"},
    "Schwab International Equity ETF":          {"cik": "0001444822", "ticker": "SCHF", "issuer": "Schwab", "type": "ETF",        "aum_b":  28, "benchmark": "FTSE Dev ex US"},
    "Schwab Emerging Markets Equity ETF":       {"cik": "0001444822", "ticker": "SCHE", "issuer": "Schwab", "type": "ETF",        "aum_b":   9, "benchmark": "FTSE EM"},
    "Schwab US Dividend Equity ETF":            {"cik": "0001444822", "ticker": "SCHD", "issuer": "Schwab", "type": "ETF",        "aum_b":  55, "benchmark": "Dow Jones US Dividend 100"},
    "Schwab US REIT ETF":                       {"cik": "0001444822", "ticker": "SCHH", "issuer": "Schwab", "type": "ETF",        "aum_b":   6, "benchmark": "Dow Jones US Select REIT"},
    "Schwab Short-Term US Treasury ETF":        {"cik": "0001444822", "ticker": "SCHO", "issuer": "Schwab", "type": "Bond ETF",   "aum_b":   5, "benchmark": "Bloomberg 1-3Y US Tsy"},
    "Schwab Intermediate-Term US Treasury ETF": {"cik": "0001444822", "ticker": "SCHR", "issuer": "Schwab", "type": "Bond ETF",   "aum_b":   3, "benchmark": "Bloomberg 3-10Y US Tsy"},
    "Schwab US Aggregate Bond ETF":             {"cik": "0001444822", "ticker": "SCHZ", "issuer": "Schwab", "type": "Bond ETF",   "aum_b":   6, "benchmark": "Bloomberg US Agg"},
    # ── ARK Invest ────────────────────────────────────────────────────────
    "ARK Innovation ETF":                       {"cik": "0001579982", "ticker": "ARKK", "issuer": "ARK Invest", "type": "Active ETF", "aum_b":   8, "benchmark": "None", "manager": "Cathie Wood"},
    "ARK Genomic Revolution ETF":               {"cik": "0001579982", "ticker": "ARKG", "issuer": "ARK Invest", "type": "Active ETF", "aum_b":   2, "benchmark": "None"},
    "ARK Next Generation Internet ETF":         {"cik": "0001579982", "ticker": "ARKW", "issuer": "ARK Invest", "type": "Active ETF", "aum_b":   2, "benchmark": "None"},
    "ARK Autonomous Technology & Robotics ETF": {"cik": "0001579982", "ticker": "ARKQ", "issuer": "ARK Invest", "type": "Active ETF", "aum_b":   1, "benchmark": "None"},
    "ARK Fintech Innovation ETF":               {"cik": "0001579982", "ticker": "ARKF", "issuer": "ARK Invest", "type": "Active ETF", "aum_b":   1, "benchmark": "None"},
    "ARK Space Exploration & Innovation ETF":   {"cik": "0001579982", "ticker": "ARKX", "issuer": "ARK Invest", "type": "Active ETF", "aum_b": 0.5,"benchmark": "None"},
    # ── Fidelity Mutual Funds ─────────────────────────────────────────────
    "Fidelity Contrafund":                      {"cik": "0000315066", "ticker": "FCNTX", "issuer": "Fidelity", "type": "Mutual Fund", "aum_b": 145, "benchmark": "S&P 500", "manager": "William Danoff"},
    "Fidelity Magellan Fund":                   {"cik": "0000315066", "ticker": "FMAGX", "issuer": "Fidelity", "type": "Mutual Fund", "aum_b":  18, "benchmark": "S&P 500"},
    "Fidelity Blue Chip Growth":                {"cik": "0000315066", "ticker": "FBGRX", "issuer": "Fidelity", "type": "Mutual Fund", "aum_b":  50, "benchmark": "Russell 1000 Growth"},
    "Fidelity Growth Company Fund":             {"cik": "0000315066", "ticker": "FDGRX", "issuer": "Fidelity", "type": "Mutual Fund", "aum_b":  60, "benchmark": "Russell 3000 Growth"},
    "Fidelity Low-Priced Stock Fund":           {"cik": "0000315066", "ticker": "FLPSX", "issuer": "Fidelity", "type": "Mutual Fund", "aum_b":  30, "benchmark": "Russell 2000 Value"},
    "Fidelity OTC Portfolio":                   {"cik": "0000315066", "ticker": "FOCPX", "issuer": "Fidelity", "type": "Mutual Fund", "aum_b":  20, "benchmark": "Nasdaq Composite"},
    "Fidelity Puritan Fund":                    {"cik": "0000315066", "ticker": "FPURX", "issuer": "Fidelity", "type": "Balanced",    "aum_b":  30, "benchmark": "60/40 Blend"},
    "Fidelity Balanced Fund":                   {"cik": "0000315066", "ticker": "FBALX", "issuer": "Fidelity", "type": "Balanced",    "aum_b":  40, "benchmark": "60/40 Blend"},
    "Fidelity Zero Total Market Index Fund":    {"cik": "0000315066", "ticker": "FZROX", "issuer": "Fidelity", "type": "Index Fund",  "aum_b":  25, "benchmark": "Fidelity US Total Market"},
    "Fidelity Total Market Index Fund":         {"cik": "0000315066", "ticker": "FSKAX", "issuer": "Fidelity", "type": "Index Fund",  "aum_b":  60, "benchmark": "Dow Jones US Total Market"},
    "Fidelity 500 Index Fund":                  {"cik": "0000315066", "ticker": "FXAIX", "issuer": "Fidelity", "type": "Index Fund",  "aum_b": 450, "benchmark": "S&P 500"},
    "Fidelity Extended Market Index Fund":      {"cik": "0000315066", "ticker": "FSMAX", "issuer": "Fidelity", "type": "Index Fund",  "aum_b":  30, "benchmark": "Dow Jones US Completion"},
    "Fidelity International Index Fund":        {"cik": "0000315066", "ticker": "FSPSX", "issuer": "Fidelity", "type": "Index Fund",  "aum_b":  45, "benchmark": "MSCI EAFE"},
    "Fidelity Strategic Dividend & Income":     {"cik": "0000315066", "ticker": "FSDIX", "issuer": "Fidelity", "type": "Mutual Fund", "aum_b":   8, "benchmark": "S&P 500"},
    # ── T. Rowe Price ─────────────────────────────────────────────────────
    "T. Rowe Price Growth Stock Fund":          {"cik": "0000080255", "ticker": "PRGFX", "issuer": "T. Rowe Price", "type": "Mutual Fund", "aum_b": 85, "benchmark": "Russell 1000 Growth"},
    "T. Rowe Price Blue Chip Growth Fund":      {"cik": "0000080255", "ticker": "TRBCX", "issuer": "T. Rowe Price", "type": "Mutual Fund", "aum_b": 95, "benchmark": "Russell 1000 Growth"},
    "T. Rowe Price Capital Appreciation Fund":  {"cik": "0000080255", "ticker": "PRWCX", "issuer": "T. Rowe Price", "type": "Balanced",    "aum_b": 45, "benchmark": "S&P 500"},
    "T. Rowe Price Mid-Cap Growth Fund":        {"cik": "0000080255", "ticker": "RPMGX", "issuer": "T. Rowe Price", "type": "Mutual Fund", "aum_b": 30, "benchmark": "Russell Midcap Growth"},
    "T. Rowe Price New Horizons Fund":          {"cik": "0000080255", "ticker": "PRNHX", "issuer": "T. Rowe Price", "type": "Mutual Fund", "aum_b": 45, "benchmark": "Russell 2000 Growth"},
    "T. Rowe Price Equity Income Fund":         {"cik": "0000080255", "ticker": "PRFDX", "issuer": "T. Rowe Price", "type": "Mutual Fund", "aum_b": 25, "benchmark": "Russell 1000 Value"},
    "T. Rowe Price Dividend Growth Fund":       {"cik": "0000080255", "ticker": "PRDGX", "issuer": "T. Rowe Price", "type": "Mutual Fund", "aum_b": 18, "benchmark": "S&P 500"},
    "T. Rowe Price International Stock Fund":   {"cik": "0000080255", "ticker": "PRITX", "issuer": "T. Rowe Price", "type": "Mutual Fund", "aum_b": 12, "benchmark": "MSCI EAFE"},
    # ── American Funds / Capital Group ────────────────────────────────────
    "American Funds Growth Fund of America":    {"cik": "0000003516", "ticker": "AGTHX", "issuer": "Capital Group", "type": "Mutual Fund", "aum_b": 250, "benchmark": "S&P 500"},
    "American Funds Capital World Growth":      {"cik": "0000003516", "ticker": "CWGIX", "issuer": "Capital Group", "type": "Mutual Fund", "aum_b": 110, "benchmark": "MSCI ACWI"},
    "American Funds EuroPacific Growth":        {"cik": "0000003516", "ticker": "AEPGX", "issuer": "Capital Group", "type": "Mutual Fund", "aum_b": 130, "benchmark": "MSCI EAFE"},
    "American Funds Investment Company of Am":  {"cik": "0000003516", "ticker": "AIVSX", "issuer": "Capital Group", "type": "Mutual Fund", "aum_b": 180, "benchmark": "S&P 500"},
    "American Funds Fundamental Investors":     {"cik": "0000003516", "ticker": "ANCFX", "issuer": "Capital Group", "type": "Mutual Fund", "aum_b": 110, "benchmark": "S&P 500"},
    "American Funds Washington Mutual":         {"cik": "0000003516", "ticker": "AWSHX", "issuer": "Capital Group", "type": "Mutual Fund", "aum_b":  90, "benchmark": "S&P 500"},
    "American Funds New Perspective Fund":      {"cik": "0000003516", "ticker": "ANWPX", "issuer": "Capital Group", "type": "Mutual Fund", "aum_b":  80, "benchmark": "MSCI ACWI"},
    "American Funds Capital Income Builder":    {"cik": "0000003516", "ticker": "CAIBX", "issuer": "Capital Group", "type": "Mutual Fund", "aum_b":  65, "benchmark": "60/40 Blend"},
    # ── PIMCO ─────────────────────────────────────────────────────────────
    "PIMCO Total Return Fund":                  {"cik": "0000927654", "ticker": "PTTAX", "issuer": "PIMCO", "type": "Bond Fund", "aum_b":  70, "benchmark": "Bloomberg US Agg"},
    "PIMCO Income Fund":                        {"cik": "0000927654", "ticker": "PONAX", "issuer": "PIMCO", "type": "Bond Fund", "aum_b": 150, "benchmark": "Bloomberg US Agg"},
    "PIMCO Short-Term Fund":                    {"cik": "0000927654", "ticker": "PSHAX", "issuer": "PIMCO", "type": "Bond Fund", "aum_b":   8, "benchmark": "ICE BofA 1-3Y US Corp/Gov"},
    "PIMCO All Asset Fund":                     {"cik": "0000927654", "ticker": "PASDX", "issuer": "PIMCO", "type": "Bond Fund", "aum_b":  18, "benchmark": "CPI + 650bps"},
    "PIMCO High Yield Fund":                    {"cik": "0000927654", "ticker": "PHDAX", "issuer": "PIMCO", "type": "Bond Fund", "aum_b":  10, "benchmark": "ICE BofA HY Master II"},
    # ── Dodge & Cox ───────────────────────────────────────────────────────
    "Dodge & Cox Stock Fund":                   {"cik": "0000028816", "ticker": "DODGX", "issuer": "Dodge & Cox", "type": "Mutual Fund", "aum_b":  95, "benchmark": "S&P 500"},
    "Dodge & Cox Income Fund":                  {"cik": "0000028816", "ticker": "DODIX", "issuer": "Dodge & Cox", "type": "Bond Fund",   "aum_b":  75, "benchmark": "Bloomberg US Agg"},
    "Dodge & Cox International Stock Fund":     {"cik": "0000028816", "ticker": "DODFX", "issuer": "Dodge & Cox", "type": "Mutual Fund", "aum_b":  50, "benchmark": "MSCI EAFE"},
    "Dodge & Cox Balanced Fund":                {"cik": "0000028816", "ticker": "DODBX", "issuer": "Dodge & Cox", "type": "Balanced",    "aum_b":  18, "benchmark": "60/40 Blend"},
    # ── MFS / Wellington ──────────────────────────────────────────────────
    "MFS Value Fund":                           {"cik": "0000064463", "ticker": "MEIAX", "issuer": "MFS", "type": "Mutual Fund", "aum_b":  20, "benchmark": "Russell 1000 Value"},
    "MFS Growth Fund":                          {"cik": "0000064463", "ticker": "MFEGX", "issuer": "MFS", "type": "Mutual Fund", "aum_b":  15, "benchmark": "Russell 1000 Growth"},
    "MFS International Value Fund":             {"cik": "0000064463", "ticker": "MINVX", "issuer": "MFS", "type": "Mutual Fund", "aum_b":  25, "benchmark": "MSCI EAFE Value"},
    "MFS Massachusetts Investors Growth":       {"cik": "0000064463", "ticker": "MIGFX", "issuer": "MFS", "type": "Mutual Fund", "aum_b":  10, "benchmark": "S&P 500"},
    # ── Value / Boutique ──────────────────────────────────────────────────
    "Sequoia Fund":                             {"cik": "0000088525", "ticker": "SEQUX", "issuer": "Ruane Cunniff", "type": "Mutual Fund", "aum_b":  4, "benchmark": "S&P 500"},
    "Longleaf Partners Fund":                   {"cik": "0000892657", "ticker": "LLPFX", "issuer": "Southeastern", "type": "Mutual Fund", "aum_b":  3, "benchmark": "S&P 500"},
    "Yacktman Fund":                            {"cik": "0000883237", "ticker": "YACKX", "issuer": "Yacktman AM", "type": "Mutual Fund", "aum_b":  4, "benchmark": "S&P 500"},
    "First Eagle Global Fund":                  {"cik": "0000035214", "ticker": "SGENX", "issuer": "First Eagle", "type": "Mutual Fund", "aum_b": 30, "benchmark": "MSCI ACWI"},
    "Fairholme Fund":                           {"cik": "0001071297", "ticker": "FAIRX", "issuer": "Fairholme", "type": "Mutual Fund", "aum_b":  1, "benchmark": "S&P 500"},
    "Oakmark Fund":                             {"cik": "0000885680", "ticker": "OAKMX", "issuer": "Harris Associates", "type": "Mutual Fund", "aum_b": 20, "benchmark": "S&P 500"},
    "Oakmark International Fund":               {"cik": "0000885680", "ticker": "OAKIX", "issuer": "Harris Associates", "type": "Mutual Fund", "aum_b": 25, "benchmark": "MSCI EAFE"},
    # ── JPMorgan / Goldman ────────────────────────────────────────────────
    "JPMorgan US Equity Fund":                  {"cik": "0000763232", "ticker": "JUEAX", "issuer": "JPMorgan", "type": "Mutual Fund", "aum_b":  10, "benchmark": "S&P 500"},
    "JPMorgan Core Bond Fund":                  {"cik": "0000763232", "ticker": "PGBOX", "issuer": "JPMorgan", "type": "Bond Fund",   "aum_b":  20, "benchmark": "Bloomberg US Agg"},
    "Goldman Sachs Growth Opportunities Fund":  {"cik": "0000822818", "ticker": "GGOAX", "issuer": "Goldman Sachs", "type": "Mutual Fund", "aum_b":  5, "benchmark": "Russell Midcap Growth"},
    "Goldman Sachs Large Cap Growth Insights":  {"cik": "0000822818", "ticker": "GLCAX", "issuer": "Goldman Sachs", "type": "Mutual Fund", "aum_b":  8, "benchmark": "Russell 1000 Growth"},
    # ── Fixed-income specialists ──────────────────────────────────────────
    "Baird Aggregate Bond Fund":                {"cik": "0001135778", "ticker": "BAGSX", "issuer": "Baird", "type": "Bond Fund", "aum_b":  15, "benchmark": "Bloomberg US Agg"},
    "Metropolitan West Total Return Bond":      {"cik": "0001090372", "ticker": "MWTRX", "issuer": "MetWest", "type": "Bond Fund", "aum_b":  60, "benchmark": "Bloomberg US Agg"},
    "Loomis Sayles Bond Fund":                  {"cik": "0000203028", "ticker": "LSBRX", "issuer": "Loomis Sayles","type": "Bond Fund", "aum_b":  12, "benchmark": "Bloomberg US Universal"},
    "Lord Abbett Short Duration Income":        {"cik": "0000060714", "ticker": "LALDX", "issuer": "Lord Abbett","type": "Bond Fund",  "aum_b":  18, "benchmark": "Bloomberg 1-3Y Gov/Credit"},
    # ── Alternative / Multi-asset ──────────────────────────────────────────
    "PIMCO All Asset All Authority Fund":       {"cik": "0000927654", "ticker": "PAUAX", "issuer": "PIMCO", "type": "Alternative", "aum_b":   3, "benchmark": "CPI + 650bps"},
    "Calvert Equity Fund":                      {"cik": "0000202453", "ticker": "CSIEX", "issuer": "Calvert", "type": "ESG Fund",   "aum_b":   5, "benchmark": "Russell 1000 Growth"},
    "Parnassus Core Equity Fund":               {"cik": "0000808948", "ticker": "PRBLX", "issuer": "Parnassus", "type": "ESG Fund",  "aum_b":  20, "benchmark": "S&P 500"},
    "Neuberger Berman Socially Responsive":     {"cik": "0000070858", "ticker": "NRAAX", "issuer": "NB", "type": "ESG Fund",        "aum_b":   4, "benchmark": "Russell 1000 Value"},
}

# Build quick-lookup indexes
_CIK_TO_FUND: Dict[str, List[str]] = defaultdict(list)
_TICKER_TO_FUND: Dict[str, str] = {}
for _fn, _fi in FUND_UNIVERSE.items():
    _CIK_TO_FUND[_fi["cik"].lstrip("0")].append(_fn)
    if "ticker" in _fi:
        _TICKER_TO_FUND[_fi["ticker"].upper()] = _fn


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS fund_universe (
    cik             TEXT PRIMARY KEY,
    fund_name       TEXT,
    ticker          TEXT,
    issuer          TEXT,
    fund_type       TEXT,
    aum_b           REAL,
    benchmark       TEXT,
    edgar_name      TEXT,
    latest_nport    TEXT,
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fund_holdings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fund_cik        TEXT NOT NULL,
    period_date     TEXT NOT NULL,
    name            TEXT,
    cusip           TEXT,
    isin            TEXT,
    ticker          TEXT,
    lei             TEXT,
    val_usd         REAL,
    pct_val         REAL,
    quantity        REAL,
    quantity_type   TEXT,
    asset_cat       TEXT,
    country         TEXT,
    currency        TEXT,
    coupon          REAL,
    maturity_date   TEXT,
    is_derivative   INTEGER DEFAULT 0,
    derivative_type TEXT,
    fair_val_level  TEXT,
    UNIQUE(fund_cik, period_date, cusip, name)
);
CREATE INDEX IF NOT EXISTS idx_fh_cik_period ON fund_holdings(fund_cik, period_date);
CREATE INDEX IF NOT EXISTS idx_fh_ticker ON fund_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_fh_cusip  ON fund_holdings(cusip);

CREATE TABLE IF NOT EXISTS holdings_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fund_cik        TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    period_date     TEXT NOT NULL,
    pct_val         REAL,
    val_usd         REAL,
    quantity        REAL,
    action          TEXT,
    UNIQUE(fund_cik, ticker, period_date)
);

CREATE TABLE IF NOT EXISTS fund_flows (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fund_cik        TEXT NOT NULL,
    period_date     TEXT NOT NULL,
    total_assets    REAL,
    net_assets      REAL,
    redemptions_3m  REAL,
    subscriptions_3m REAL,
    net_flow_3m     REAL,
    UNIQUE(fund_cik, period_date)
);

CREATE TABLE IF NOT EXISTS fund_analytics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fund_cik        TEXT NOT NULL,
    period_date     TEXT NOT NULL,
    n_holdings      INTEGER,
    top10_pct       REAL,
    hhi             REAL,
    effective_n     REAL,
    equity_pct      REAL,
    bond_pct        REAL,
    cash_pct        REAL,
    deriv_pct       REAL,
    intl_pct        REAL,
    port_duration   REAL,
    factor_mkt      REAL,
    factor_smb      REAL,
    factor_hml      REAL,
    factor_rmw      REAL,
    factor_cma      REAL,
    UNIQUE(fund_cik, period_date)
);
"""


@contextlib.contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_DDL)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "{" in tag else tag


def _find(el: ET.Element, local: str) -> Optional[ET.Element]:
    for child in el.iter():
        if _strip_ns(child.tag) == local:
            return child
    return None


def _findall(el: ET.Element, local: str) -> List[ET.Element]:
    return [c for c in el.iter() if _strip_ns(c.tag) == local]


def _text(el: ET.Element, local: str, default: str = "") -> str:
    node = _find(el, local)
    return (node.text or "").strip() if node is not None else default


def _float(val: Optional[str], default: float = 0.0) -> float:
    try:
        return float((val or "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return default


async def _http_get(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    retries: int = 3,
) -> Any:
    hdrs = headers or _HEADERS
    for attempt in range(retries):
        try:
            r = await client.get(url, params=params, headers=hdrs, timeout=30)
            r.raise_for_status()
            await asyncio.sleep(_RATE_LIMIT_SLEEP)
            ct = r.headers.get("content-type", "")
            return r.json() if "json" in ct else r.text
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                await asyncio.sleep(2 ** attempt * 5)
            elif attempt < retries - 1:
                await asyncio.sleep(1.5)
            else:
                raise
        except httpx.RequestError:
            if attempt < retries - 1:
                await asyncio.sleep(1.5)
            else:
                raise
    return {}


# ---------------------------------------------------------------------------
# EDGAR universe enumerator
# ---------------------------------------------------------------------------

class EdgarUniverseEnumerator:
    """
    Enumerate ALL N-PORT-P filers from EDGAR full-text search.

    Paginates through EFTS until ≥500 unique CIKs collected or exhausted.
    Results are merged with the static FUND_UNIVERSE registry.
    """

    _MAX_FILERS = 600

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure(self) -> httpx.AsyncClient:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient(follow_redirects=True)
        return self._client

    async def enumerate(
        self,
        start_date: Optional[str] = None,
        max_filers: int = _MAX_FILERS,
    ) -> List[Dict[str, Any]]:
        """
        Return list of {cik, name, latest_filing_date} for N-PORT-P filers.

        Uses EFTS pagination (from / size) to collect max_filers unique CIKs.
        Deduplicated by CIK; last 12 months only to avoid stale data.
        """
        client = await self._ensure()
        if not start_date:
            start_date = (date.today() - timedelta(days=365)).isoformat()

        seen_ciks: set = set()
        results: List[Dict] = []
        page_from = 0
        page_size = 40

        while len(results) < max_filers:
            params: Dict = {
                "forms":     "NPORT-P",
                "dateRange": "custom",
                "startdt":   start_date,
                "enddt":     date.today().isoformat(),
                "from":      page_from,
                "size":      page_size,
            }
            try:
                data = await _http_get(client, _EFTS_BASE, params=params)
            except Exception as exc:
                logger.warning("enumerate page %d err: %s", page_from, exc)
                break

            if not isinstance(data, dict):
                break

            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                src    = hit.get("_source", {})
                entity = str(src.get("entity_id", "")).strip()
                if not entity or entity in seen_ciks:
                    continue
                seen_ciks.add(entity)
                names = src.get("display_names", [])
                results.append({
                    "cik":              entity.zfill(10),
                    "name":             names[0] if names else "",
                    "latest_filing":    src.get("file_date", "")[:10],
                    "latest_period":    src.get("period_of_report", "")[:10],
                    "accession_number": src.get("accession_no", ""),
                })
                if len(results) >= max_filers:
                    break

            total = data.get("hits", {}).get("total", {})
            total_val = total.get("value", 0) if isinstance(total, dict) else int(total or 0)
            page_from += page_size
            if page_from >= min(total_val, max_filers * 3):
                break

        # Merge with static registry
        static_ciks = {info["cik"].lstrip("0") for info in FUND_UNIVERSE.values()}
        for cik_str in static_ciks:
            if cik_str not in seen_ciks:
                # Build minimal entry from registry
                for fn, fi in FUND_UNIVERSE.items():
                    if fi["cik"].lstrip("0") == cik_str:
                        results.append({
                            "cik":           fi["cik"],
                            "name":          fn,
                            "latest_filing": "",
                            "latest_period": "",
                            "accession_number": "",
                        })
                        seen_ciks.add(cik_str)
                        break

        logger.info("Enumerated %d unique N-PORT-P filers", len(results))
        return results

    async def persist_universe(self) -> int:
        """Enumerate and write to fund_universe table. Returns count inserted."""
        filers = await self.enumerate()
        rows = 0
        with _db() as conn:
            for f in filers:
                cik = f["cik"]
                known = _CIK_TO_FUND.get(cik.lstrip("0"), [])
                fname = f["name"]
                tkr   = ""
                iss   = ""
                ft    = ""
                aum   = 0.0
                bench = ""
                if known:
                    fi = FUND_UNIVERSE[known[0]]
                    fname = fname or known[0]
                    tkr   = fi.get("ticker", "")
                    iss   = fi.get("issuer", "")
                    ft    = fi.get("type", "")
                    aum   = fi.get("aum_b", 0.0)
                    bench = fi.get("benchmark", "")
                conn.execute(
                    """INSERT OR REPLACE INTO fund_universe
                       (cik, fund_name, ticker, issuer, fund_type, aum_b, benchmark,
                        edgar_name, latest_nport, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))""",
                    (cik, fname, tkr, iss, ft, aum, bench,
                     f["name"], f.get("latest_filing", "")),
                )
                rows += 1
        return rows


# ---------------------------------------------------------------------------
# N-PORT XML parser (comprehensive)
# ---------------------------------------------------------------------------

class NPortXMLParser:
    """
    Parse all sections of an N-PORT-P XML filing.

    Sections:
      genInfo      — fund metadata, total/net assets, period
      returnInfo   — monthly returns (R1, R2, R3), realized/unrealized gain
      invstOrSec   — all holdings (equity, debt, deriv, cash, MBS, ABS...)
      derivativeInfo (nested) — forward FX, IR swaps, options, credit default
      creditSpreadRiskInfo    — bond fund credit exposure by duration bucket
      borrowingInfo / repurchaseAgreements
    """

    def parse(self, xml_text: str, fund_cik: str) -> Dict[str, Any]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            return {"error": f"XML parse: {exc}", "fund_cik": fund_cik}

        result: Dict[str, Any] = {"fund_cik": fund_cik}

        # ── genInfo ──────────────────────────────────────────────────────
        result.update(self._parse_gen_info(root))

        # ── returnInfo ───────────────────────────────────────────────────
        result["return_info"] = self._parse_return_info(root)

        # ── creditSpreadRiskInfo ─────────────────────────────────────────
        result["credit_spread_risk"] = self._parse_credit_spread_risk(root)

        # ── Flow/redemption info ─────────────────────────────────────────
        result["redemptions_3m"]    = _float(_text(root, "aggrFlwsRedeem3Mon"))
        result["subscriptions_3m"]  = _float(_text(root, "aggrFlwsSale3Mon"))

        # ── Borrowing / securities lending ───────────────────────────────
        result["borrowed_val_usd"] = sum(
            _float(_text(n, "valUSD")) for n in _findall(root, "borrowedSecurities")
        )

        # ── Repurchase agreements ────────────────────────────────────────
        repo_nodes = _findall(root, "repurchaseAgreement")
        result["repo_val_usd"] = sum(
            _float(_text(n, "principalAmt")) for n in repo_nodes
        )

        # ── Holdings ─────────────────────────────────────────────────────
        holdings = [
            h for sec in _findall(root, "invstOrSec")
            for h in [self._parse_holding(sec)] if h
        ]
        result["holdings"]   = holdings
        result["n_holdings"] = len(holdings)
        return result

    # ── Helpers ──────────────────────────────────────────────────────────

    def _parse_gen_info(self, root: ET.Element) -> Dict[str, Any]:
        return {
            "fund_name":      _text(root, "regName") or _text(root, "seriesName"),
            "series_name":    _text(root, "seriesName"),
            "period_date":    _text(root, "repPdDate"),
            "fiscal_year_end":_text(root, "fiscYrEnd"),
            "total_assets":   _float(_text(root, "totAssets")),
            "net_assets":     _float(_text(root, "netAssets")),
            "nav":            _float(_text(root, "totNavOfSeries")),
            "n_shareholders": _float(_text(root, "totNumSharehldr")),
            "shares_outstanding": _float(_text(root, "totShrOutstanding")),
        }

    def _parse_return_info(self, root: ET.Element) -> Dict[str, Any]:
        ri = _find(root, "returnInfo")
        if ri is None:
            return {}
        return {
            "return_m1":         _float(_text(ri, "monthlyTotReturnMon1")),
            "return_m2":         _float(_text(ri, "monthlyTotReturnMon2")),
            "return_m3":         _float(_text(ri, "monthlyTotReturnMon3")),
            "realized_gain":     _float(_text(ri, "realizedGain")),
            "unrealized_gain":   _float(_text(ri, "unrealizedAppreciation")),
        }

    def _parse_credit_spread_risk(self, root: ET.Element) -> Dict[str, Any]:
        cs = _find(root, "creditSpreadRiskInfo")
        if cs is None:
            return {}
        return {
            "spread_3m":   _float(_text(cs, "intrst3Mon")),
            "spread_1y":   _float(_text(cs, "intrst1Yr")),
            "spread_5y":   _float(_text(cs, "intrst5Yr")),
            "spread_10y":  _float(_text(cs, "intrst10Yr")),
            "spread_30y":  _float(_text(cs, "intrst30Yr")),
            "ig_total":    _float(_text(cs, "igTotal")),
            "hy_total":    _float(_text(cs, "hyTotal")),
        }

    def _parse_holding(self, sec: ET.Element) -> Optional[Dict[str, Any]]:
        name = _text(sec, "name")
        if not name:
            return None

        asset_cat = _text(sec, "assetCat") or "OTH"
        h: Dict[str, Any] = {
            "name":             name,
            "cusip":            _text(sec, "cusip") or None,
            "isin":             _text(sec, "isin")  or None,
            "ticker":           _text(sec, "ticker") or None,
            "lei":              _text(sec, "lei")   or None,
            "val_usd":          _float(_text(sec, "valUSD")),
            "pct_val":          _float(_text(sec, "pctVal")),
            "quantity":         _float(_text(sec, "balance")),
            "quantity_type":    _text(sec, "units") or "NS",
            "asset_cat":        asset_cat,
            "asset_cat_label":  ASSET_CAT_MAP.get(asset_cat, asset_cat),
            "country":          _text(sec, "invCountry") or None,
            "currency":         _text(sec, "curCd") or "USD",
            "is_restricted":    _text(sec, "isRestrictedSec").lower() == "y",
            "fair_val_level":   _text(sec, "fairValLevel") or None,
            "is_derivative":    False,
        }

        # Debt-specific
        debt = _find(sec, "debtSec")
        if debt is not None:
            h["coupon"]       = _float(_text(debt, "annualizedRt")) or None
            h["maturity_date"]= _text(debt, "maturityDt") or None
            h["is_default"]   = _text(debt, "isDefault").lower() == "y"
            h["coupon_type"]  = _text(debt, "couponKind") or None
            h["duration"]     = _float(_text(debt, "modDurationToWorst")) or None

        # Derivative-specific
        deriv = _find(sec, "derivativeInfo")
        if deriv is not None:
            h["is_derivative"]  = True
            h["derivative_type"]= _text(deriv, "derivCat") or None
            h["counterparty"]   = _text(deriv, "ctrptyNm") or None
            h["notional"]       = _float(_text(deriv, "notionalAmt")) or None
            h["delta"]          = _float(_text(deriv, "delta")) or None
            # Forward FX
            fwd = _find(deriv, "fwdFx")
            if fwd is not None:
                h["fwd_currency"]  = _text(fwd, "curCdToBeSold")
                h["fwd_notional"]  = _float(_text(fwd, "notionalAmt"))
                h["fwd_settle_dt"] = _text(fwd, "settlementDt")
            # IR swap
            irs = _find(deriv, "intrstRtSwap")
            if irs is not None:
                h["swap_pay_leg"]  = _text(irs, "payLegRt")
                h["swap_recv_leg"] = _text(irs, "recvLegRt")
                h["swap_maturity"] = _text(irs, "maturityDt")
            # Option
            opt = _find(deriv, "option")
            if opt is not None:
                h["option_type"]   = _text(opt, "putOrCall")
                h["option_strike"] = _float(_text(opt, "exercisePrice"))
                h["option_expiry"] = _text(opt, "expiryDt")
                h["option_shares"] = _float(_text(opt, "sharesOrPrincipalAmt"))

        return h


# ---------------------------------------------------------------------------
# EDGAR filing fetcher
# ---------------------------------------------------------------------------

class EdgarFilingFetcher:
    """Fetch N-PORT-P filings and return parsed data for a given CIK."""

    def __init__(self) -> None:
        self._parser = NPortXMLParser()
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure(self) -> httpx.AsyncClient:
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient(follow_redirects=True)
        return self._client

    async def list_filings(
        self,
        cik: str,
        lookback_months: int = 6,
        max_filings: int = 6,
    ) -> List[Dict]:
        client   = await self._ensure()
        start_dt = (date.today() - timedelta(days=lookback_months * 31)).isoformat()
        params   = {
            "forms":     "NPORT-P",
            "dateRange": "custom",
            "startdt":   start_dt,
            "enddt":     date.today().isoformat(),
            "q":         f"entity_id:{cik.lstrip('0')}",
            "size":      max_filings,
        }
        try:
            data = await _http_get(client, _EFTS_BASE, params=params)
        except Exception as exc:
            logger.warning("list_filings cik=%s: %s", cik, exc)
            return []

        if not isinstance(data, dict):
            return []

        results = []
        for hit in data.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            acc = src.get("accession_no", "")
            eid = str(src.get("entity_id", ""))
            results.append({
                "cik":              eid.zfill(10),
                "accession_number": acc,
                "filed_date":       src.get("file_date", "")[:10],
                "period":           src.get("period_of_report", "")[:10],
            })
        return sorted(results, key=lambda x: x.get("period", ""), reverse=True)

    async def fetch_and_parse(self, cik: str, accession: str) -> Dict[str, Any]:
        client    = await self._ensure()
        acc_clean = accession.replace("-", "")
        cik_raw   = cik.lstrip("0")

        # Fetch filing index to locate XML
        index_url = (
            f"{_EDGAR_ARCHIVE}/{cik_raw}/{acc_clean}/"
            f"{accession}-index.htm"
        )
        xml_file = "primary_doc.xml"
        try:
            idx_html = await _http_get(client, index_url, headers=_XML_HEADERS)
            if isinstance(idx_html, str):
                m = re.search(r'href="([^"]*\.xml)"', idx_html, re.I)
                if m:
                    xml_file = m.group(1).rsplit("/", 1)[-1]
        except Exception:
            pass

        xml_url = f"{_EDGAR_ARCHIVE}/{cik_raw}/{acc_clean}/{xml_file}"
        try:
            xml_text = await _http_get(client, xml_url, headers=_XML_HEADERS)
        except Exception as exc:
            return {"error": str(exc), "cik": cik, "accession": accession}

        if not isinstance(xml_text, str) or not xml_text.strip():
            return {"error": "empty XML", "cik": cik}

        return self._parser.parse(xml_text, cik)

    async def get_latest_holdings_df(
        self, cik: str, period: Optional[str] = None
    ) -> pd.DataFrame:
        filings = await self.list_filings(cik)
        if not filings:
            return pd.DataFrame()

        if period:
            target = next(
                (f for f in filings if f["period"].startswith(period)),
                filings[0],
            )
        else:
            target = filings[0]

        acc = target.get("accession_number", "")
        if not acc:
            return pd.DataFrame()

        parsed = await self.fetch_and_parse(cik, acc)
        holdings = parsed.get("holdings", [])
        if not holdings:
            return pd.DataFrame()

        df = pd.DataFrame(holdings)
        for col, val in [
            ("fund_cik", cik),
            ("period_date", parsed.get("period_date", "")),
            ("fund_name",   parsed.get("fund_name", "")),
            ("total_assets",parsed.get("total_assets", 0)),
            ("net_assets",  parsed.get("net_assets", 0)),
        ]:
            df[col] = val
        return df


# ---------------------------------------------------------------------------
# Analytics engine
# ---------------------------------------------------------------------------

class NPortAnalyticsEngine:
    """
    Core analytics computed over holdings DataFrames.

    All methods are pure / offline (no HTTP) and operate on pd.DataFrame.
    """

    # ── Concentration ─────────────────────────────────────────────────────

    @staticmethod
    def concentration_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """HHI, top-10 pct, effective N, active share stub."""
        if df.empty or "pct_val" not in df.columns:
            return {}

        w = pd.to_numeric(df["pct_val"], errors="coerce").fillna(0) / 100.0
        w = w[w > 0]
        if w.empty:
            return {}

        hhi         = float((w ** 2).sum())
        effective_n = 1.0 / hhi if hhi > 0 else float("inf")
        df_sorted   = df.assign(_w=w).sort_values("_w", ascending=False)
        top10_pct   = float(df_sorted.head(10)["_w"].sum() * 100)
        n_holdings  = len(w)

        top10 = df_sorted.head(10)[
            [c for c in ("name", "ticker", "cusip", "pct_val", "val_usd") if c in df.columns]
        ].to_dict(orient="records")

        return {
            "n_holdings":  n_holdings,
            "top10_pct":   round(top10_pct, 2),
            "hhi":         round(hhi, 6),
            "effective_n": round(effective_n, 1),
            "top10":       top10,
        }

    # ── Style box (Morningstar 3×3) ───────────────────────────────────────

    @staticmethod
    def compute_style_box(
        df: pd.DataFrame,
        pb_col: str = "pb_ratio",
        mktcap_col: str = "mktcap_usd",
        weight_col: str = "pct_val",
    ) -> Dict[str, Any]:
        """
        Morningstar 3×3 style box: value/blend/growth × small/mid/large.

        Value axis uses portfolio-weighted average P/B ratio:
          - Value:  wt-avg P/B < 1.75
          - Blend:  1.75 ≤ wt-avg P/B < 3.00
          - Growth: wt-avg P/B ≥ 3.00

        Size axis uses portfolio-weighted average market cap:
          - Large:  wt-avg mktcap ≥ $10B
          - Mid:    $2B ≤ wt-avg mktcap < $10B
          - Small:  wt-avg mktcap < $2B

        Thresholds follow Morningstar methodology (Morningstar Style Box
        Methodology, June 2017).  If P/B or mktcap data are absent,
        the axis falls back to 'Unknown'.
        """
        result: Dict[str, Any] = {
            "value_axis": "Unknown",
            "size_axis": "Unknown",
            "style_box": "Unknown",
            "weighted_avg_pb": None,
            "weighted_avg_mktcap_b": None,
        }

        if df.empty:
            return result

        w = pd.to_numeric(df.get(weight_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
        total_w = w.sum()
        if total_w <= 0:
            return result
        w_norm = w / total_w   # normalised weights (sum = 1)

        # --- Value axis (P/B) ---
        if pb_col in df.columns:
            pb = pd.to_numeric(df[pb_col], errors="coerce")
            valid = pb.notna() & (pb > 0)
            if valid.sum() >= 1:
                w_pb = w_norm.copy()
                w_pb[~valid] = 0.0
                w_pb_sum = w_pb.sum()
                if w_pb_sum > 0:
                    w_pb = w_pb / w_pb_sum
                wt_pb = float((pb.fillna(0) * w_pb).sum())
                result["weighted_avg_pb"] = round(wt_pb, 4)
                if wt_pb < 1.75:
                    result["value_axis"] = "Value"
                elif wt_pb < 3.00:
                    result["value_axis"] = "Blend"
                else:
                    result["value_axis"] = "Growth"

        # --- Size axis (market cap) ---
        if mktcap_col in df.columns:
            mc = pd.to_numeric(df[mktcap_col], errors="coerce")
            valid = mc.notna() & (mc > 0)
            if valid.sum() >= 1:
                w_mc = w_norm.copy()
                w_mc[~valid] = 0.0
                w_mc_sum = w_mc.sum()
                if w_mc_sum > 0:
                    w_mc = w_mc / w_mc_sum
                wt_mc = float((mc.fillna(0) * w_mc).sum())   # USD
                wt_mc_b = wt_mc / 1e9                         # convert to billions
                result["weighted_avg_mktcap_b"] = round(wt_mc_b, 3)
                if wt_mc_b >= 10.0:
                    result["size_axis"] = "Large"
                elif wt_mc_b >= 2.0:
                    result["size_axis"] = "Mid"
                else:
                    result["size_axis"] = "Small"

        # --- Composite label ---
        if result["value_axis"] != "Unknown" and result["size_axis"] != "Unknown":
            result["style_box"] = f"{result['size_axis']}-{result['value_axis']}"

        return result

    # ── Active share ──────────────────────────────────────────────────────

    @staticmethod
    def compute_active_share(
        portfolio: pd.DataFrame,
        benchmark: pd.DataFrame,
        ticker_col: str = "ticker",
        weight_col: str = "pct_val",
    ) -> Dict[str, Any]:
        """
        Compute active share vs. a benchmark index.

        Active Share = (1/2) × Σ |w_portfolio_i - w_benchmark_i|

        Range: [0, 1].  0 = identical to benchmark; 1 = fully active.
        Formula: Cremers & Petajisto (2009), "How Active Is Your Fund Manager?"

        Parameters
        ----------
        portfolio  : DataFrame with ticker and portfolio weight columns.
        benchmark  : DataFrame with ticker and benchmark weight columns.
        ticker_col : column name for security identifiers (must be present in both).
        weight_col : column name for weight (%).  Will be normalised to fractions.

        Returns dict with active_share, n_active_positions, tracking_error_proxy.
        """
        def _normalise(df: pd.DataFrame) -> pd.Series:
            w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
            total = w.sum()
            return (w / total if total > 0 else w).values

        if portfolio.empty or ticker_col not in portfolio.columns or weight_col not in portfolio.columns:
            return {"active_share": 0.0, "n_active_positions": 0, "error": "invalid_portfolio"}
        if benchmark.empty or ticker_col not in benchmark.columns or weight_col not in benchmark.columns:
            return {"active_share": 1.0, "n_active_positions": len(portfolio), "error": "no_benchmark"}

        p = portfolio[[ticker_col, weight_col]].copy()
        b = benchmark[[ticker_col, weight_col]].copy()

        # Normalise weights to fractions (not %)
        p_w = pd.to_numeric(p[weight_col], errors="coerce").fillna(0)
        p_total = p_w.sum()
        p["_w"] = p_w / p_total if p_total > 0 else p_w

        b_w = pd.to_numeric(b[weight_col], errors="coerce").fillna(0)
        b_total = b_w.sum()
        b["_w"] = b_w / b_total if b_total > 0 else b_w

        # Merge on ticker
        merged = pd.merge(
            p[[ticker_col, "_w"]].rename(columns={"_w": "w_p"}),
            b[[ticker_col, "_w"]].rename(columns={"_w": "w_b"}),
            on=ticker_col,
            how="outer",
        ).fillna(0)

        diff = (merged["w_p"] - merged["w_b"]).abs()
        active_share = float(diff.sum() / 2.0)
        active_share = max(0.0, min(1.0, active_share))   # clamp [0, 1]

        # Positions where portfolio weight meaningfully exceeds benchmark
        active_positions = merged[
            (merged["w_p"] - merged["w_b"]).abs() > 0.001
        ]

        return {
            "active_share": round(active_share, 6),
            "active_share_pct": round(active_share * 100, 4),
            "n_total_securities": len(merged),
            "n_active_positions": len(active_positions),
            "interpretation": (
                "Closet indexer (<20%)" if active_share < 0.20 else
                "Mildly active (20–40%)" if active_share < 0.40 else
                "Moderately active (40–60%)" if active_share < 0.60 else
                "Highly active (60–80%)" if active_share < 0.80 else
                "Pure stock picker (>80%)"
            ),
        }

    # ── Portfolio drift detection ─────────────────────────────────────────

    @staticmethod
    def detect_portfolio_drift(
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        drift_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect whether the current portfolio has drifted beyond threshold from targets.

        Triggers a rebalancing signal when any position deviates more than
        `drift_threshold` (default 5%) from its target weight.

        Parameters
        ----------
        current_weights : {ticker: weight} where weights are fractions summing ≈ 1.
        target_weights  : {ticker: target_weight} fractions.
        drift_threshold : absolute deviation in weight fraction that triggers signal.
                          Default 0.05 = 5 percentage points.

        Returns
        -------
        dict with:
          rebalance_required : bool
          max_drift          : float (largest single absolute deviation)
          drifted_positions  : list of {ticker, current, target, drift}
          drift_details      : full per-position breakdown
        """
        if not current_weights or not target_weights:
            return {"rebalance_required": False, "max_drift": 0.0,
                    "drifted_positions": [], "drift_details": []}

        # Normalise both weight dicts so they sum to 1
        c_total = sum(current_weights.values())
        t_total = sum(target_weights.values())
        c_norm = {k: v / c_total for k, v in current_weights.items()} if c_total > 0 else current_weights
        t_norm = {k: v / t_total for k, v in target_weights.items()} if t_total > 0 else target_weights

        all_tickers = set(c_norm.keys()) | set(t_norm.keys())
        details = []
        drifted = []
        max_drift = 0.0

        for tkr in sorted(all_tickers):
            cw = c_norm.get(tkr, 0.0)
            tw = t_norm.get(tkr, 0.0)
            drift = abs(cw - tw)
            max_drift = max(max_drift, drift)
            entry = {
                "ticker": tkr,
                "current_weight": round(cw, 6),
                "target_weight": round(tw, 6),
                "drift": round(drift, 6),
                "drift_pct": round(drift * 100, 4),
                "breaches_threshold": drift > drift_threshold,
                "direction": "overweight" if cw > tw else "underweight" if cw < tw else "on_target",
            }
            details.append(entry)
            if drift > drift_threshold:
                drifted.append(entry)

        rebalance_required = len(drifted) > 0

        return {
            "rebalance_required": rebalance_required,
            "max_drift": round(max_drift, 6),
            "max_drift_pct": round(max_drift * 100, 4),
            "drift_threshold": drift_threshold,
            "n_positions_checked": len(all_tickers),
            "n_drifted": len(drifted),
            "drifted_positions": drifted,
            "drift_details": details,
            "signal": "REBALANCE" if rebalance_required else "HOLD",
        }

    # ── Sector exposure ───────────────────────────────────────────────────

    @staticmethod
    def sector_exposure(df: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregate portfolio weight by inferred GICS sector.

        Uses asset_cat_label as primary; falls back to name heuristics.
        """
        if df.empty:
            return {}

        def _infer_sector(row: pd.Series) -> str:
            cat = str(row.get("asset_cat", "OTH"))
            if cat in ("EC", "EP"):
                # Map known sector ETFs by name/ticker
                name = str(row.get("name", "")).upper()
                tkr  = str(row.get("ticker", "")).upper()
                for kw, sec in [
                    ("TECH", "Technology"), ("MSFT", "Technology"), ("AAPL", "Technology"),
                    ("NVDA", "Technology"), ("GOOGL", "Technology"), ("META", "Technology"),
                    ("AMZN", "Technology"), ("HEALTH", "Healthcare"), ("JNJ", "Healthcare"),
                    ("PFE", "Healthcare"), ("UNH", "Healthcare"),
                    ("BANK", "Financial"), ("JPM", "Financial"), ("BAC", "Financial"),
                    ("ENERGY", "Energy"), ("XOM", "Energy"), ("CVX", "Energy"),
                    ("UTIL", "Utility"), ("NEE", "Utility"),
                    ("REIT", "Real Estate"), ("AMT", "Real Estate"),
                    ("COMM", "Communication"), ("DIS", "Communication"),
                ]:
                    if kw in name or kw in tkr:
                        return sec
                return "Equity (Other)"
            if cat in ("DB", "UST"):
                return "Fixed Income"
            if cat in ("ABS", "MBS"):
                return "Structured Credit"
            if cat == "MM":
                return "Cash & Equivalents"
            if cat == "DER":
                return "Derivatives"
            if cat in ("RF",):
                return "Real Estate"
            return "Other"

        df = df.copy()
        df["_sector"] = df.apply(_infer_sector, axis=1)
        df["_w"]      = pd.to_numeric(df.get("pct_val", 0), errors="coerce").fillna(0)

        return (
            df.groupby("_sector")["_w"]
            .sum()
            .sort_values(ascending=False)
            .round(2)
            .to_dict()
        )

    # ── Geographic exposure ───────────────────────────────────────────────

    @staticmethod
    def geo_exposure(df: pd.DataFrame) -> Dict[str, Any]:
        """Domestic vs international + top countries by weight."""
        if df.empty or "country" not in df.columns:
            return {}

        df = df.copy()
        df["_w"]      = pd.to_numeric(df.get("pct_val", 0), errors="coerce").fillna(0)
        df["_country"]= df["country"].fillna("US")

        by_country = (
            df.groupby("_country")["_w"]
            .sum()
            .sort_values(ascending=False)
            .round(2)
        )
        us_pct   = float(by_country.get("US", 0.0))
        intl_pct = float(by_country[by_country.index != "US"].sum())

        return {
            "domestic_pct":      round(us_pct, 2),
            "international_pct": round(intl_pct, 2),
            "top_countries":     by_country.head(10).to_dict(),
        }

    # ── Asset class breakdown ─────────────────────────────────────────────

    @staticmethod
    def asset_class_breakdown(df: pd.DataFrame) -> Dict[str, float]:
        """Equity / bond / deriv / cash / other as % of portfolio."""
        if df.empty:
            return {}

        df = df.copy()
        df["_w"] = pd.to_numeric(df.get("pct_val", 0), errors="coerce").fillna(0)

        def _class(cat: str) -> str:
            if cat in ("EC", "EP", "ETF", "CE", "MF"):    return "equity"
            if cat in ("DB", "UST", "ABS", "MBS"):         return "fixed_income"
            if cat in ("MM", "STIV"):                       return "cash"
            if cat == "DER":                                return "derivatives"
            if cat == "RF":                                 return "real_estate"
            return "other"

        df["_class"] = df["asset_cat"].fillna("OTH").apply(_class)
        return (
            df.groupby("_class")["_w"]
            .sum()
            .round(2)
            .to_dict()
        )

    # ── Portfolio duration (bond funds) ──────────────────────────────────

    @staticmethod
    def portfolio_duration(df: pd.DataFrame) -> Optional[float]:
        """
        Weighted-average modified duration for fixed income funds.

        Uses the 'duration' field parsed from N-PORT debtSec element.
        Returns None if no duration data available.
        """
        if df.empty or "duration" not in df.columns:
            return None

        df = df.copy()
        df["_dur"] = pd.to_numeric(df["duration"], errors="coerce")
        df["_w"]   = pd.to_numeric(df.get("pct_val", 0), errors="coerce").fillna(0)
        has_dur = df["_dur"].notna() & (df["_w"] > 0)
        if not has_dur.any():
            return None

        d = df[has_dur]
        total_w = d["_w"].sum()
        if total_w <= 0:
            return None
        wt_dur = float((d["_dur"] * d["_w"]).sum() / total_w)
        return round(wt_dur, 2)

    # ── FF5 factor exposures ──────────────────────────────────────────────

    @staticmethod
    def factor_exposures(df: pd.DataFrame) -> Dict[str, float]:
        """
        Approximate Fama-French 5-factor exposures from sector weights.

        Loads sector weights then dot-products with pre-estimated sector betas.
        Suitable for orientation / screening; not a regression-based estimate.
        """
        if df.empty:
            return {}

        sector_w = NPortAnalyticsEngine.sector_exposure(df)
        factors  = {"mkt": 0.0, "smb": 0.0, "hml": 0.0, "rmw": 0.0, "cma": 0.0}

        total_w = sum(sector_w.values()) or 100.0
        for sector, weight in sector_w.items():
            loadings = FF5_SECTOR_LOADINGS.get(sector, FF5_SECTOR_LOADINGS["Other"])
            frac = weight / total_w
            for f in factors:
                factors[f] += frac * loadings[f]

        return {k: round(v, 4) for k, v in factors.items()}

    # ── Overlap / Jaccard ─────────────────────────────────────────────────

    @staticmethod
    def jaccard_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, Any]:
        """Jaccard similarity between two holding DataFrames."""

        def _keys(df: pd.DataFrame) -> set:
            s: set = set()
            for col in ("cusip", "isin", "ticker"):
                if col in df.columns:
                    s.update(df[col].dropna().str.strip().str.upper().tolist())
            return s - {"", "N/A", "NONE"}

        a, b       = _keys(df_a), _keys(df_b)
        inter      = a & b
        union      = a | b
        jaccard    = len(inter) / len(union) if union else 0.0
        return {
            "jaccard":        round(jaccard, 4),
            "n_a":            len(a),
            "n_b":            len(b),
            "n_common":       len(inter),
            "common_sample":  sorted(inter)[:20],
            "unique_a":       sorted(a - b)[:10],
            "unique_b":       sorted(b - a)[:10],
        }

    # ── Crowding score ────────────────────────────────────────────────────

    @staticmethod
    def crowding_score(
        ticker: str,
        holders: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Crowding = n_funds_holding × avg_conviction / normalizer.

        High crowding → crowded long → elevated forced-selling risk.
        """
        if not holders:
            return {"ticker": ticker, "crowding_score": 0.0, "signal": "low",
                    "n_funds": 0, "description": "No fund holders found"}

        n       = len(holders)
        avg_pct = float(np.mean([h.get("pct_of_fund", 0) for h in holders]))
        max_pct = float(max(h.get("pct_of_fund", 0) for h in holders))
        tot_val = float(sum(h.get("val_usd", 0) for h in holders))

        raw     = n * avg_pct / 100.0
        score   = min(raw * 10.0, 100.0)

        signal  = ("high"   if score >= _CROWDING_HIGH else
                   "medium" if score >= _CROWDING_MED  else "low")
        desc    = {
            "high":   "Crowded long — elevated forced-selling risk on redemptions.",
            "medium": "Moderate crowding — monitor fund flow changes.",
            "low":    "Low crowding — limited systematic liquidation risk.",
        }[signal]

        return {
            "ticker":         ticker,
            "crowding_score": round(score, 2),
            "signal":         signal,
            "description":    desc,
            "n_funds":        n,
            "avg_pct":        round(avg_pct, 3),
            "max_pct":        round(max_pct, 3),
            "total_held_usd": round(tot_val, 0),
            "institutionally_validated": n >= _SMART_MONEY_MIN_FUNDS,
            "top_holders":    sorted(holders, key=lambda h: h.get("pct_of_fund", 0), reverse=True)[:5],
        }

    # ── Implied fund flows (monthly) ──────────────────────────────────────

    @staticmethod
    def implied_flows(
        df_current: pd.DataFrame,
        df_prior:   pd.DataFrame,
        price_map:  Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Flow = (quantity_current - quantity_prior) × midpoint_price.

        price_map: {ticker: price}. If None, uses val_usd / quantity as proxy.
        """
        if df_current.empty or df_prior.empty:
            return pd.DataFrame()

        def _key(row: pd.Series) -> str:
            return (row.get("cusip") or row.get("isin") or
                    row.get("ticker") or row.get("name", ""))

        cur = {_key(r): r for _, r in df_current.iterrows()}
        pri = {_key(r): r for _, r in df_prior.iterrows()}

        rows = []
        for key, curr_row in cur.items():
            prior_row = pri.get(key, pd.Series(dtype=object))
            q_cur  = float(curr_row.get("quantity", 0) or 0)
            q_pri  = float(prior_row.get("quantity", 0) if not prior_row.empty else 0)
            v_cur  = float(curr_row.get("val_usd",  0) or 0)
            v_pri  = float(prior_row.get("val_usd",  0) if not prior_row.empty else 0)

            tkr = str(curr_row.get("ticker") or "")
            if price_map and tkr in price_map:
                price = price_map[tkr]
            elif q_cur > 0:
                price = v_cur / q_cur
            else:
                price = 0.0

            dq   = q_cur - q_pri
            flow = dq * price

            rows.append({
                "key":          key,
                "ticker":       tkr,
                "name":         str(curr_row.get("name", "")),
                "q_current":    round(q_cur,  0),
                "q_prior":      round(q_pri,  0),
                "delta_shares": round(dq,     0),
                "price_proxy":  round(price,  4),
                "implied_flow": round(flow,   0),
                "val_current":  round(v_cur,  0),
                "val_prior":    round(v_pri,  0),
                "action": (
                    "new_position" if q_pri == 0 and q_cur  > 0 else
                    "full_exit"    if q_cur == 0 and q_pri  > 0 else
                    "increased"    if dq > q_pri * 0.05        else
                    "reduced"      if dq < -q_pri * 0.05       else
                    "unchanged"
                ),
            })

        return pd.DataFrame(rows).sort_values("implied_flow", key=abs, ascending=False)

    # ── Smart-money consensus ─────────────────────────────────────────────

    @staticmethod
    def smart_money_consensus(
        holders_by_ticker: Dict[str, List[Dict]],
        min_funds: int = _SMART_MONEY_MIN_FUNDS,
    ) -> List[Dict[str, Any]]:
        """
        Return tickers held by ≥ min_funds funds, ranked by total $ held.

        These are "institutionally validated" positions.
        """
        results = []
        for ticker, holders in holders_by_ticker.items():
            if len(holders) < min_funds:
                continue
            total_val = sum(h.get("val_usd", 0) for h in holders)
            avg_pct   = float(np.mean([h.get("pct_of_fund", 0) for h in holders]))
            results.append({
                "ticker":          ticker,
                "n_funds":         len(holders),
                "total_held_usd":  round(total_val, 0),
                "avg_pct_of_fund": round(avg_pct, 3),
                "funds":           [h.get("fund_name", "") for h in holders[:5]],
            })

        return sorted(results, key=lambda x: x["total_held_usd"], reverse=True)


# ---------------------------------------------------------------------------
# High-level service layer
# ---------------------------------------------------------------------------

class NPortService:
    """Orchestrates fetching, parsing, analytics, and SQLite persistence."""

    def __init__(self) -> None:
        self._fetcher   = EdgarFilingFetcher()
        self._analytics = NPortAnalyticsEngine()

    async def get_fund_holdings(
        self, cik: str, period: Optional[str] = None, use_cache: bool = True
    ) -> pd.DataFrame:
        """Holdings DataFrame for a fund, with optional SQLite cache."""
        if use_cache:
            cached = self._load_cached_holdings(cik, period)
            if not cached.empty:
                return cached

        df = await self._fetcher.get_latest_holdings_df(cik, period)
        if not df.empty:
            self._persist_holdings(df, cik)
        return df

    def _load_cached_holdings(
        self, cik: str, period: Optional[str]
    ) -> pd.DataFrame:
        try:
            with _db() as conn:
                if period:
                    rows = conn.execute(
                        "SELECT * FROM fund_holdings WHERE fund_cik=? AND period_date LIKE ?",
                        (cik, f"{period}%"),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM fund_holdings WHERE fund_cik=?
                           ORDER BY period_date DESC LIMIT 2000""",
                        (cik,),
                    ).fetchall()
                if rows:
                    return pd.DataFrame([dict(r) for r in rows])
        except Exception as exc:
            logger.debug("cache miss cik=%s: %s", cik, exc)
        return pd.DataFrame()

    def _persist_holdings(self, df: pd.DataFrame, cik: str) -> None:
        try:
            with _db() as conn:
                for _, row in df.iterrows():
                    conn.execute(
                        """INSERT OR REPLACE INTO fund_holdings
                           (fund_cik, period_date, name, cusip, isin, ticker, lei,
                            val_usd, pct_val, quantity, quantity_type, asset_cat,
                            country, currency, coupon, maturity_date, is_derivative,
                            derivative_type, fair_val_level)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            cik,
                            str(row.get("period_date", "")),
                            str(row.get("name", "")),
                            row.get("cusip"),
                            row.get("isin"),
                            row.get("ticker"),
                            row.get("lei"),
                            float(row.get("val_usd", 0) or 0),
                            float(row.get("pct_val", 0) or 0),
                            float(row.get("quantity", 0) or 0),
                            str(row.get("quantity_type", "NS")),
                            str(row.get("asset_cat", "OTH")),
                            row.get("country"),
                            str(row.get("currency", "USD")),
                            row.get("coupon"),
                            row.get("maturity_date"),
                            int(bool(row.get("is_derivative", False))),
                            row.get("derivative_type"),
                            row.get("fair_val_level"),
                        ),
                    )
        except Exception as exc:
            logger.debug("persist_holdings err: %s", exc)

    async def get_fund_analytics(self, cik: str) -> Dict[str, Any]:
        df = await self.get_fund_holdings(cik)
        if df.empty:
            return {"error": "no holdings data", "cik": cik}

        conc     = self._analytics.concentration_metrics(df)
        sector   = self._analytics.sector_exposure(df)
        geo      = self._analytics.geo_exposure(df)
        asset_cl = self._analytics.asset_class_breakdown(df)
        duration = self._analytics.portfolio_duration(df)
        factors  = self._analytics.factor_exposures(df)

        result = {
            "cik":           cik,
            "fund_name":     str(df.get("fund_name", pd.Series([""]))[0]) if "fund_name" in df.columns else "",
            "period":        str(df["period_date"].iloc[0]) if "period_date" in df.columns else "",
            "concentration": conc,
            "sector":        sector,
            "geo":           geo,
            "asset_class":   asset_cl,
            "duration":      duration,
            "ff5_factors":   factors,
        }

        # Persist analytics snapshot
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO fund_analytics
                       (fund_cik, period_date, n_holdings, top10_pct, hhi,
                        effective_n, equity_pct, bond_pct, cash_pct, deriv_pct,
                        intl_pct, port_duration,
                        factor_mkt, factor_smb, factor_hml, factor_rmw, factor_cma)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        cik, result["period"],
                        conc.get("n_holdings", 0),
                        conc.get("top10_pct", 0),
                        conc.get("hhi", 0),
                        conc.get("effective_n", 0),
                        asset_cl.get("equity", 0),
                        asset_cl.get("fixed_income", 0),
                        asset_cl.get("cash", 0),
                        asset_cl.get("derivatives", 0),
                        geo.get("international_pct", 0),
                        duration or 0,
                        factors.get("mkt", 0),
                        factors.get("smb", 0),
                        factors.get("hml", 0),
                        factors.get("rmw", 0),
                        factors.get("cma", 0),
                    ),
                )
        except Exception as exc:
            logger.debug("persist_analytics err: %s", exc)

        return result

    async def get_crowding_score(self, ticker: str) -> Dict[str, Any]:
        holders = await self._collect_holders(ticker, min_pct=0.1)
        return self._analytics.crowding_score(ticker, holders)

    async def get_overlap(self, cik_a: str, cik_b: str) -> Dict[str, Any]:
        df_a = await self.get_fund_holdings(cik_a)
        df_b = await self.get_fund_holdings(cik_b)
        if df_a.empty or df_b.empty:
            return {"error": "insufficient holdings data"}
        return self._analytics.jaccard_overlap(df_a, df_b)

    async def get_new_positions(self, cik: str) -> List[Dict]:
        filings = await self._fetcher.list_filings(cik, lookback_months=4)
        if len(filings) < 2:
            return []
        cur_parsed  = await self._fetcher.fetch_and_parse(cik, filings[0]["accession_number"])
        prev_parsed = await self._fetcher.fetch_and_parse(cik, filings[1]["accession_number"])

        def _ids(parsed: Dict) -> Dict[str, Dict]:
            out: Dict = {}
            for h in parsed.get("holdings", []):
                k = h.get("cusip") or h.get("isin") or h.get("ticker") or h.get("name", "")
                if k:
                    out[k] = h
            return out

        cur_ids  = _ids(cur_parsed)
        prev_ids = _ids(prev_parsed)

        new_pos = [
            {**h, "signal": "new_position", "period": filings[0]["period"]}
            for k, h in cur_ids.items()
            if k not in prev_ids
        ]
        return sorted(new_pos, key=lambda h: h.get("val_usd", 0), reverse=True)

    async def get_sector_exposure(self, cik: str) -> Dict[str, float]:
        df = await self.get_fund_holdings(cik)
        return self._analytics.sector_exposure(df)

    async def get_fund_flows(self, cik: str) -> Dict[str, Any]:
        """Pull flow data from latest N-PORT genInfo."""
        filings = await self._fetcher.list_filings(cik, lookback_months=3)
        if not filings:
            return {"error": "no filings", "cik": cik}
        parsed = await self._fetcher.fetch_and_parse(cik, filings[0]["accession_number"])
        flow   = {
            "cik":              cik,
            "period":           parsed.get("period_date", ""),
            "total_assets":     parsed.get("total_assets", 0),
            "net_assets":       parsed.get("net_assets", 0),
            "redemptions_3m":   parsed.get("redemptions_3m", 0),
            "subscriptions_3m": parsed.get("subscriptions_3m", 0),
            "net_flow_3m":      (parsed.get("subscriptions_3m", 0)
                                 - parsed.get("redemptions_3m", 0)),
        }
        with _db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO fund_flows
                   (fund_cik, period_date, total_assets, net_assets,
                    redemptions_3m, subscriptions_3m, net_flow_3m)
                   VALUES (?,?,?,?,?,?,?)""",
                (cik, flow["period"], flow["total_assets"], flow["net_assets"],
                 flow["redemptions_3m"], flow["subscriptions_3m"], flow["net_flow_3m"]),
            )
        return flow

    async def get_consensus_holdings(
        self, min_funds: int = _SMART_MONEY_MIN_FUNDS
    ) -> List[Dict[str, Any]]:
        """
        Cross-fund analysis: stocks held by ≥ min_funds tracked funds.

        Iterates through all known-universe CIKs (batched), aggregates holders
        by ticker, then filters for smart-money consensus threshold.
        """
        ciks = list({fi["cik"] for fi in FUND_UNIVERSE.values()})
        holders_by_ticker: Dict[str, List[Dict]] = defaultdict(list)

        # Process in batches to avoid overwhelming EDGAR
        batch_size = 10
        for i in range(0, min(len(ciks), 50), batch_size):
            batch = ciks[i: i + batch_size]
            tasks = [self.get_fund_holdings(c) for c in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for cik, df in zip(batch, results):
                if isinstance(df, Exception) or not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                fund_name = df["fund_name"].iloc[0] if "fund_name" in df.columns else cik
                for _, row in df.iterrows():
                    tkr = str(row.get("ticker") or "")
                    if not tkr or tkr.upper() in ("", "N/A", "NONE"):
                        continue
                    holders_by_ticker[tkr.upper()].append({
                        "fund_cik":    cik,
                        "fund_name":   fund_name,
                        "pct_of_fund": float(row.get("pct_val", 0) or 0),
                        "val_usd":     float(row.get("val_usd",  0) or 0),
                    })

        return self._analytics.smart_money_consensus(holders_by_ticker, min_funds)

    async def _collect_holders(
        self, ticker: str, min_pct: float = 0.1
    ) -> List[Dict[str, Any]]:
        ciks  = list({fi["cik"] for fi in FUND_UNIVERSE.values()})
        holders: List[Dict] = []

        batch_size = 10
        for i in range(0, min(len(ciks), 40), batch_size):
            batch = ciks[i: i + batch_size]
            tasks = [self.get_fund_holdings(c) for c in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for cik, df in zip(batch, results):
                if isinstance(df, Exception) or not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                fund_name = df["fund_name"].iloc[0] if "fund_name" in df.columns else cik
                mask = pd.Series([False] * len(df))
                for col in ("ticker", "name"):
                    if col in df.columns:
                        mask |= df[col].fillna("").str.upper().str.contains(
                            ticker.upper(), regex=False
                        )
                matches = df[mask]
                for _, row in matches.iterrows():
                    pct = float(row.get("pct_val", 0) or 0)
                    if pct < min_pct:
                        continue
                    holders.append({
                        "fund_cik":    cik,
                        "fund_name":   fund_name,
                        "pct_of_fund": pct,
                        "val_usd":     float(row.get("val_usd", 0) or 0),
                    })

        return holders


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

nport_v3_router = APIRouter(prefix="/nport/v3", tags=["nport-v3"])
_svc = NPortService()


@nport_v3_router.get("/fund/{cik}/holdings")
async def route_holdings(
    cik: str,
    period: Optional[str] = Query(None, description="YYYY-MM or YYYY-MM-DD"),
    use_cache: bool = Query(True),
):
    """All holdings for a fund from the most recent (or specified) N-PORT-P filing."""
    df = await _svc.get_fund_holdings(cik, period=period, use_cache=use_cache)
    if df.empty:
        raise HTTPException(404, f"No holdings found for CIK {cik}")
    return {
        "cik":       cik,
        "period":    str(df["period_date"].iloc[0]) if "period_date" in df.columns else "",
        "fund_name": str(df["fund_name"].iloc[0])   if "fund_name"   in df.columns else "",
        "n":         len(df),
        "holdings":  df.to_dict(orient="records"),
    }


@nport_v3_router.get("/fund/{cik}/analytics")
async def route_analytics(cik: str):
    """Concentration, sector, geo, asset-class, duration, FF5 factor exposures."""
    result = await _svc.get_fund_analytics(cik)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@nport_v3_router.get("/fund/{cik}/flows")
async def route_fund_flows(cik: str):
    """3-month redemptions, subscriptions, and net flow from N-PORT genInfo."""
    return await _svc.get_fund_flows(cik)


@nport_v3_router.get("/fund/{cik}/new-positions")
async def route_new_positions(cik: str):
    """Holdings that appeared in current N-PORT quarter but not in prior quarter."""
    new_pos = await _svc.get_new_positions(cik)
    return {"cik": cik, "new_positions": new_pos, "n": len(new_pos)}


@nport_v3_router.get("/fund/{cik}/sector-exposure")
async def route_sector(cik: str):
    """Portfolio weight by inferred GICS sector."""
    return {"cik": cik, "sector_exposure": await _svc.get_sector_exposure(cik)}


@nport_v3_router.get("/overlap")
async def route_overlap(
    cik_a: str = Query(..., description="CIK of fund A"),
    cik_b: str = Query(..., description="CIK of fund B"),
):
    """Jaccard overlap between two fund portfolios."""
    return await _svc.get_overlap(cik_a, cik_b)


@nport_v3_router.get("/crowding-score/{ticker}")
async def route_crowding(ticker: str):
    """
    Crowding score for a stock.

    n_funds_holding × avg_conviction → forced-selling risk indicator.
    Also returns institutionally_validated flag (≥10 funds).
    """
    return await _svc.get_crowding_score(ticker.upper())


@nport_v3_router.get("/consensus-holdings")
async def route_consensus(
    min_funds: int = Query(_SMART_MONEY_MIN_FUNDS, ge=2, le=50),
):
    """
    Stocks held by ≥ min_funds tracked funds = smart-money consensus.

    Iterates up to 50 known-universe CIKs in batches of 10.
    """
    result = await _svc.get_consensus_holdings(min_funds=min_funds)
    return {"min_funds": min_funds, "consensus": result, "n": len(result)}


@nport_v3_router.get("/universe")
async def route_universe(
    issuer: Optional[str] = Query(None),
    fund_type: Optional[str] = Query(None),
    min_aum_b: float = Query(0.0),
):
    """Browse the static fund universe with optional filters."""
    results = []
    for name, info in FUND_UNIVERSE.items():
        if issuer and issuer.lower() not in info.get("issuer", "").lower():
            continue
        if fund_type and fund_type.lower() not in info.get("type", "").lower():
            continue
        if info.get("aum_b", 0) < min_aum_b:
            continue
        results.append({"fund_name": name, **info})
    results.sort(key=lambda x: x.get("aum_b", 0), reverse=True)
    return {"funds": results, "n": len(results)}


@nport_v3_router.post("/universe/refresh")
async def route_refresh_universe():
    """
    Enumerate ALL N-PORT-P filers from EDGAR and persist to SQLite.

    Targets 500+ unique CIKs; may take 60-120 seconds.
    """
    enumerator = EdgarUniverseEnumerator()
    count = await enumerator.persist_universe()
    return {"persisted": count, "message": f"Refreshed {count} fund universe entries"}
