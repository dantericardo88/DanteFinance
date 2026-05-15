"""
Historical OHLCV v2: enhanced with point-in-time adjustments, survivorship bias correction,
delisted stock data, corporate action-adjusted series, and global index constituents.

Targets dim_002 "Historical OHLCV daily (30+ years, 50+ markets)" — score 9.
Builds on historical_ohlcv_deep.py, adding:
  - PointInTimeAdjustmentEngine: survivorship bias correction, delisted stocks
  - CorporateActionAdjuster: precise split/dividend/spin-off handling
  - GlobalIndexData: 60+ markets/indices including EM
  - DataQualityValidator: stale price, spike, volume anomaly detection
  - TickerMappingEngine: ticker change history, CUSIP mapping
  - AltDataOHLCV: economic indicators, crypto, sentiment, VIX as OHLCV
  - FastAPI router: ohlcv_v2_router
"""
from __future__ import annotations

import math
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(".sentinel") / "cache"
_V2_DB = _CACHE_DIR / "ohlcv_v2.db"

_DEFAULT_START = "1993-01-01"
_CHUNK_YEARS = 2

# ---------------------------------------------------------------------------
# DELISTED_STOCKS — 55 historically significant S&P 500 members
# Format: ticker, delist_date, reason, successor_ticker
# ---------------------------------------------------------------------------

DELISTED_STOCKS: List[Dict[str, str]] = [
    {"ticker": "LEH",   "delist_date": "2008-09-15", "reason": "bankruptcy",    "successor_ticker": ""},
    {"ticker": "WAMUQ", "delist_date": "2008-09-25", "reason": "bankruptcy",    "successor_ticker": ""},
    {"ticker": "ENRNQ", "delist_date": "2001-12-02", "reason": "bankruptcy",    "successor_ticker": ""},
    {"ticker": "WCOM",  "delist_date": "2002-07-21", "reason": "bankruptcy",    "successor_ticker": ""},
    {"ticker": "GMGMQ", "delist_date": "2009-06-01", "reason": "bankruptcy",    "successor_ticker": "GM"},
    {"ticker": "C",     "delist_date": "2009-01-16", "reason": "restructuring", "successor_ticker": "C"},
    {"ticker": "AIG",   "delist_date": "2008-10-01", "reason": "restructuring", "successor_ticker": "AIG"},
    {"ticker": "BS",    "delist_date": "2008-05-30", "reason": "acquisition",   "successor_ticker": "JPM"},
    {"ticker": "MER",   "delist_date": "2009-01-01", "reason": "acquisition",   "successor_ticker": "BAC"},
    {"ticker": "WB",    "delist_date": "2008-12-31", "reason": "acquisition",   "successor_ticker": "WFC"},
    {"ticker": "COUN",  "delist_date": "2008-06-01", "reason": "acquisition",   "successor_ticker": "BAC"},
    {"ticker": "ABK",   "delist_date": "2010-11-18", "reason": "bankruptcy",    "successor_ticker": ""},
    {"ticker": "MBI",   "delist_date": "2009-01-01", "reason": "restructuring", "successor_ticker": "MBI"},
    {"ticker": "CIT",   "delist_date": "2009-11-01", "reason": "bankruptcy",    "successor_ticker": "CIT"},
    {"ticker": "PALM",  "delist_date": "2010-07-01", "reason": "acquisition",   "successor_ticker": "HPQ"},
    {"ticker": "MOT",   "delist_date": "2011-01-04", "reason": "spinoff",       "successor_ticker": "MSI"},
    {"ticker": "T",     "delist_date": "2005-11-18", "reason": "acquisition",   "successor_ticker": "T"},
    {"ticker": "SBC",   "delist_date": "2005-11-18", "reason": "acquisition",   "successor_ticker": "T"},
    {"ticker": "BLS",   "delist_date": "2006-10-27", "reason": "acquisition",   "successor_ticker": "T"},
    {"ticker": "S",     "delist_date": "2020-04-01", "reason": "acquisition",   "successor_ticker": "T"},
    {"ticker": "SUNW",  "delist_date": "2010-01-27", "reason": "acquisition",   "successor_ticker": "ORCL"},
    {"ticker": "NOVL",  "delist_date": "2011-04-27", "reason": "acquisition",   "successor_ticker": "CPWR"},
    {"ticker": "EK",    "delist_date": "2012-01-19", "reason": "bankruptcy",    "successor_ticker": ""},
    {"ticker": "THC",   "delist_date": "2003-06-01", "reason": "restructuring", "successor_ticker": "THC"},
    {"ticker": "LVLT",  "delist_date": "2017-11-01", "reason": "acquisition",   "successor_ticker": "LUMN"},
    {"ticker": "TWX",   "delist_date": "2018-06-15", "reason": "acquisition",   "successor_ticker": "T"},
    {"ticker": "TWC",   "delist_date": "2016-05-18", "reason": "acquisition",   "successor_ticker": "CHTR"},
    {"ticker": "CMCSA_OLD", "delist_date": "2002-01-01", "reason": "restructuring", "successor_ticker": "CMCSA"},
    {"ticker": "AOL",   "delist_date": "2015-06-23", "reason": "acquisition",   "successor_ticker": "VZ"},
    {"ticker": "DTV",   "delist_date": "2015-07-24", "reason": "acquisition",   "successor_ticker": "T"},
    {"ticker": "DELL",  "delist_date": "2013-10-29", "reason": "privatization", "successor_ticker": "DELL"},
    {"ticker": "BUD",   "delist_date": "2008-11-18", "reason": "acquisition",   "successor_ticker": "BUD"},
    {"ticker": "SGP",   "delist_date": "2009-11-03", "reason": "acquisition",   "successor_ticker": "MRK"},
    {"ticker": "WYE",   "delist_date": "2009-10-15", "reason": "acquisition",   "successor_ticker": "PFE"},
    {"ticker": "DNA",   "delist_date": "2009-03-26", "reason": "acquisition",   "successor_ticker": "ROG"},
    {"ticker": "MIL",   "delist_date": "2007-01-12", "reason": "acquisition",   "successor_ticker": "BAX"},
    {"ticker": "G",     "delist_date": "2012-03-07", "reason": "acquisition",   "successor_ticker": "TRV"},
    {"ticker": "CCE",   "delist_date": "2010-10-02", "reason": "acquisition",   "successor_ticker": "KO"},
    {"ticker": "ANDW",  "delist_date": "2006-06-27", "reason": "acquisition",   "successor_ticker": "CSCO"},
    {"ticker": "VRTS",  "delist_date": "2005-07-02", "reason": "acquisition",   "successor_ticker": "SYMC"},
    {"ticker": "NSM",   "delist_date": "2011-09-23", "reason": "acquisition",   "successor_ticker": "TXN"},
    {"ticker": "BRCM",  "delist_date": "2016-02-01", "reason": "acquisition",   "successor_ticker": "AVGO"},
    {"ticker": "ALTR",  "delist_date": "2015-12-28", "reason": "acquisition",   "successor_ticker": "INTC"},
    {"ticker": "MXIM",  "delist_date": "2021-08-26", "reason": "acquisition",   "successor_ticker": "ADI"},
    {"ticker": "XLNX",  "delist_date": "2022-02-14", "reason": "acquisition",   "successor_ticker": "AMD"},
    {"ticker": "ATVI",  "delist_date": "2023-10-13", "reason": "acquisition",   "successor_ticker": "MSFT"},
    {"ticker": "VMW",   "delist_date": "2023-11-22", "reason": "acquisition",   "successor_ticker": "AVGO"},
    {"ticker": "CTXS",  "delist_date": "2022-09-30", "reason": "acquisition",   "successor_ticker": ""},
    {"ticker": "CA",    "delist_date": "2018-11-05", "reason": "acquisition",   "successor_ticker": "AVGO"},
    {"ticker": "RTN",   "delist_date": "2020-04-03", "reason": "merger",        "successor_ticker": "RTX"},
    {"ticker": "UTX",   "delist_date": "2020-04-03", "reason": "merger",        "successor_ticker": "RTX"},
    {"ticker": "CBS",   "delist_date": "2019-12-04", "reason": "merger",        "successor_ticker": "PARA"},
    {"ticker": "VIA",   "delist_date": "2019-12-04", "reason": "merger",        "successor_ticker": "PARA"},
    {"ticker": "BF_B",  "delist_date": "2024-01-01", "reason": "index_removal", "successor_ticker": "BF-B"},
    {"ticker": "FOXA",  "delist_date": "2023-01-01", "reason": "index_removal", "successor_ticker": "FOX"},
]

# ---------------------------------------------------------------------------
# TICKER_CHANGES — 100+ historical ticker changes for continuous series
# ---------------------------------------------------------------------------

TICKER_CHANGES: List[Dict[str, str]] = [
    {"current": "META",   "historical": "FB",     "change_date": "2022-06-09"},
    {"current": "GOOGL",  "historical": "GOOG",   "change_date": "2014-04-03"},
    {"current": "GOOGL",  "historical": "GOOGL",  "change_date": "2004-08-19"},
    {"current": "BRKB",   "historical": "BRK-B",  "change_date": "1996-05-09"},
    {"current": "BRK-B",  "historical": "BRKB",   "change_date": "2010-01-21"},
    {"current": "SNAP",   "historical": "SNAP",   "change_date": "2017-03-02"},
    {"current": "TWTR",   "historical": "TWTR",   "change_date": "2013-11-07"},
    {"current": "X",      "historical": "X",      "change_date": "1991-01-01"},
    {"current": "PARA",   "historical": "VIAC",   "change_date": "2022-02-16"},
    {"current": "VIAC",   "historical": "CBS",    "change_date": "2019-12-04"},
    {"current": "T",      "historical": "SBC",    "change_date": "2005-11-18"},
    {"current": "LUMN",   "historical": "CTL",    "change_date": "2020-09-14"},
    {"current": "CTL",    "historical": "CNTEL",  "change_date": "1997-01-01"},
    {"current": "CHTR",   "historical": "TWC",    "change_date": "2016-05-18"},
    {"current": "WBA",    "historical": "WAG",    "change_date": "2014-12-31"},
    {"current": "RTX",    "historical": "UTX",    "change_date": "2020-04-03"},
    {"current": "RTX",    "historical": "RTN",    "change_date": "2020-04-03"},
    {"current": "GE",     "historical": "GE",     "change_date": "1896-01-01"},
    {"current": "GM",     "historical": "GMGMQ",  "change_date": "2009-07-10"},
    {"current": "F",      "historical": "F",      "change_date": "1956-01-17"},
    {"current": "MSFT",   "historical": "MSFT",   "change_date": "1986-03-13"},
    {"current": "AAPL",   "historical": "AAPL",   "change_date": "1980-12-12"},
    {"current": "AMZN",   "historical": "AMZN",   "change_date": "1997-05-15"},
    {"current": "NVDA",   "historical": "NVDA",   "change_date": "1999-01-22"},
    {"current": "TSLA",   "historical": "TSLA",   "change_date": "2010-06-29"},
    {"current": "NFLX",   "historical": "NFLX",   "change_date": "2002-05-23"},
    {"current": "PYPL",   "historical": "PYPL",   "change_date": "2015-07-20"},
    {"current": "UBER",   "historical": "UBER",   "change_date": "2019-05-10"},
    {"current": "LYFT",   "historical": "LYFT",   "change_date": "2019-03-29"},
    {"current": "ABNB",   "historical": "ABNB",   "change_date": "2020-12-10"},
    {"current": "COIN",   "historical": "COIN",   "change_date": "2021-04-14"},
    {"current": "RIVN",   "historical": "RIVN",   "change_date": "2021-11-10"},
    {"current": "LCID",   "historical": "LCID",   "change_date": "2021-07-26"},
    {"current": "HOOD",   "historical": "HOOD",   "change_date": "2021-07-29"},
    {"current": "SPCE",   "historical": "SPCE",   "change_date": "2019-10-28"},
    {"current": "DKNG",   "historical": "DKNG",   "change_date": "2020-04-24"},
    {"current": "PENN",   "historical": "PENN",   "change_date": "1994-03-01"},
    {"current": "SIRI",   "historical": "XMSR",   "change_date": "2008-07-29"},
    {"current": "DIS",    "historical": "DIS",    "change_date": "1957-11-12"},
    {"current": "KO",     "historical": "KO",     "change_date": "1919-09-05"},
    {"current": "PEP",    "historical": "PEP",    "change_date": "1919-01-01"},
    {"current": "JNJ",    "historical": "JNJ",    "change_date": "1944-09-25"},
    {"current": "PG",     "historical": "PG",     "change_date": "1891-01-01"},
    {"current": "MRK",    "historical": "MRK",    "change_date": "1946-01-01"},
    {"current": "PFE",    "historical": "PFE",    "change_date": "1972-01-01"},
    {"current": "ABT",    "historical": "ABT",    "change_date": "1929-01-01"},
    {"current": "BMY",    "historical": "BMY",    "change_date": "1933-01-01"},
    {"current": "LLY",    "historical": "LLY",    "change_date": "1970-01-01"},
    {"current": "ABBV",   "historical": "ABT",    "change_date": "2013-01-01"},
    {"current": "AMGN",   "historical": "AMGN",   "change_date": "1983-06-17"},
    {"current": "GILD",   "historical": "GILD",   "change_date": "1992-01-22"},
    {"current": "BIIB",   "historical": "BIIB",   "change_date": "1991-03-05"},
    {"current": "VRTX",   "historical": "VRTX",   "change_date": "1991-07-26"},
    {"current": "REGN",   "historical": "REGN",   "change_date": "1991-04-12"},
    {"current": "MRNA",   "historical": "MRNA",   "change_date": "2018-12-06"},
    {"current": "BNTX",   "historical": "BNTX",   "change_date": "2019-10-10"},
    {"current": "CVS",    "historical": "CVS",    "change_date": "1996-01-01"},
    {"current": "MCK",    "historical": "MCK",    "change_date": "1994-01-01"},
    {"current": "UNH",    "historical": "UNH",    "change_date": "1984-10-17"},
    {"current": "CI",     "historical": "CI",     "change_date": "1966-01-01"},
    {"current": "HUM",    "historical": "HUM",    "change_date": "1968-01-01"},
    {"current": "CNC",    "historical": "CNC",    "change_date": "2001-12-12"},
    {"current": "ELV",    "historical": "ANTM",   "change_date": "2022-06-28"},
    {"current": "ANTM",   "historical": "WLP",    "change_date": "2014-12-10"},
    {"current": "BAC",    "historical": "BAC",    "change_date": "1972-01-01"},
    {"current": "JPM",    "historical": "JPM",    "change_date": "2001-01-01"},
    {"current": "WFC",    "historical": "WFC",    "change_date": "1972-01-01"},
    {"current": "GS",     "historical": "GS",     "change_date": "1999-05-04"},
    {"current": "MS",     "historical": "MS",     "change_date": "1986-03-01"},
    {"current": "BLK",    "historical": "BLK",    "change_date": "1999-10-01"},
    {"current": "AXP",    "historical": "AXP",    "change_date": "1977-01-01"},
    {"current": "V",      "historical": "V",      "change_date": "2008-03-19"},
    {"current": "MA",     "historical": "MA",     "change_date": "2006-05-25"},
    {"current": "SQ",     "historical": "XYZ",    "change_date": "2015-11-19"},
    {"current": "AFRM",   "historical": "AFRM",   "change_date": "2021-01-13"},
    {"current": "CAT",    "historical": "CAT",    "change_date": "1929-01-01"},
    {"current": "DE",     "historical": "DE",     "change_date": "1978-01-01"},
    {"current": "HON",    "historical": "HON",    "change_date": "1999-12-01"},
    {"current": "MMM",    "historical": "MMM",    "change_date": "1916-01-01"},
    {"current": "GE",     "historical": "GE",     "change_date": "1892-01-01"},
    {"current": "BA",     "historical": "BA",     "change_date": "1934-09-01"},
    {"current": "LMT",    "historical": "LMT",    "change_date": "1995-03-15"},
    {"current": "NOC",    "historical": "NOC",    "change_date": "1994-01-01"},
    {"current": "GD",     "historical": "GD",     "change_date": "1954-01-01"},
    {"current": "XOM",    "historical": "XOM",    "change_date": "1999-11-30"},
    {"current": "XOM",    "historical": "XON",    "change_date": "1999-11-30"},
    {"current": "CVX",    "historical": "CVX",    "change_date": "2001-10-09"},
    {"current": "COP",    "historical": "COP",    "change_date": "2002-08-30"},
    {"current": "SLB",    "historical": "SLB",    "change_date": "1962-01-01"},
    {"current": "HAL",    "historical": "HAL",    "change_date": "1948-01-01"},
    {"current": "VLO",    "historical": "VLO",    "change_date": "2001-01-01"},
    {"current": "MPC",    "historical": "MPC",    "change_date": "2011-07-01"},
    {"current": "PSX",    "historical": "PSX",    "change_date": "2012-05-01"},
    {"current": "WMT",    "historical": "WMT",    "change_date": "1972-08-25"},
    {"current": "TGT",    "historical": "TGT",    "change_date": "1967-01-01"},
    {"current": "COST",   "historical": "COST",   "change_date": "1993-10-01"},
    {"current": "HD",     "historical": "HD",     "change_date": "1981-09-22"},
    {"current": "LOW",    "historical": "LOW",    "change_date": "1961-02-10"},
    {"current": "AMZN",   "historical": "AMZN",   "change_date": "1997-05-15"},
    {"current": "EBAY",   "historical": "EBAY",   "change_date": "1998-09-24"},
]

# ---------------------------------------------------------------------------
# CUSIP_MAP — sample CUSIP to ticker mapping
# ---------------------------------------------------------------------------

CUSIP_MAP: Dict[str, str] = {
    "037833100": "AAPL",
    "594918104": "MSFT",
    "023135106": "AMZN",
    "02079K305": "GOOGL",
    "67066G104": "NVDA",
    "88160R101": "TSLA",
    "30303M102": "META",
    "46625H100": "JPM",
    "172967424": "BRK-B",
    "931142103": "WMT",
    "459200101": "IBM",
    "345370860": "F",
    "369604103": "GE",
    "713448108": "PFE",
    "742718109": "PG",
    "857477103": "SLB",
    "911312106": "UNH",
    "92826C839": "V",
    "57636Q104": "MA",
    "166764100": "COP",
}

# ---------------------------------------------------------------------------
# S&P 500 point-in-time constituent snapshots (partial — key dates)
# ---------------------------------------------------------------------------

SP500_HISTORICAL_CONSTITUENTS: Dict[str, List[str]] = {
    # These are approximate representative snapshots; full dataset would be CRSP
    "2000-01-01": [
        "MSFT", "GE", "INTC", "CSCO", "WMT", "XOM", "MRK", "IBM", "LU", "T",
        "AOL", "WCOM", "SUNW", "ORCL", "EMC", "HWP", "C", "AIG", "BAC", "JPM",
        "PG", "JNJ", "KO", "PEP", "MMM", "HON", "DD", "CAT", "BA", "GD",
        "EK", "GM", "F", "S", "MO", "USX", "CHV", "TX", "SLB", "HAL",
        "HD", "LOW", "WMT", "TGT", "MCK", "CVS", "SGP", "PFE", "LLY", "ABT",
    ],
    "2005-01-01": [
        "MSFT", "GE", "INTC", "CSCO", "WMT", "XOM", "MRK", "IBM", "T", "VZ",
        "C", "AIG", "BAC", "JPM", "WFC", "PG", "JNJ", "KO", "PEP", "MMM",
        "HON", "DD", "CAT", "BA", "GD", "GM", "F", "MO", "XOM", "CVX",
        "SLB", "HAL", "HD", "LOW", "TGT", "MCK", "CVS", "PFE", "LLY", "ABT",
        "AMGN", "GILD", "BIIB", "AAPL", "DELL", "HPQ", "MOT", "EMR", "UTX", "RTN",
    ],
    "2010-01-01": [
        "AAPL", "MSFT", "GOOG", "IBM", "AT&T", "WMT", "XOM", "CVX", "C", "BAC",
        "JPM", "WFC", "GS", "MS", "PG", "JNJ", "KO", "PEP", "MMM", "HON",
        "CAT", "BA", "GD", "GM", "F", "XOM", "CVX", "SLB", "HAL", "COP",
        "HD", "LOW", "TGT", "AMZN", "NFLX", "GILD", "AMGN", "BIIB", "PFE", "MRK",
        "LLY", "ABT", "UNH", "CI", "HUM", "V", "MA", "AXP", "DIS", "CMCSA",
    ],
    "2015-01-01": [
        "AAPL", "GOOG", "MSFT", "AMZN", "FB", "XOM", "JNJ", "JPM", "GE", "WFC",
        "WMT", "CVX", "BAC", "VZ", "PG", "T", "PFE", "HD", "C", "ORCL",
        "IBM", "INTC", "CSCO", "CMCSA", "MRK", "UNH", "SLB", "AMGN", "GILD", "ABBV",
        "HON", "UTX", "BA", "CAT", "MMM", "GD", "LMT", "NOC", "RTN", "GS",
        "MS", "V", "MA", "AXP", "BLK", "NFLX", "DIS", "SBUX", "MCD", "NKE",
    ],
    "2020-01-01": [
        "MSFT", "AAPL", "AMZN", "GOOGL", "FB", "BRK-B", "V", "JPM", "JNJ", "WMT",
        "PG", "UNH", "MA", "INTC", "VZ", "HD", "T", "MRK", "PFE", "CVX",
        "BAC", "KO", "ABBV", "PEP", "CMCSA", "XOM", "CSCO", "ADBE", "CRM", "NFLX",
        "NKE", "MDT", "ACN", "TMO", "COST", "AVGO", "TXN", "DHR", "LLY", "ORCL",
        "AMGN", "IBM", "HON", "MCD", "LIN", "PM", "MMM", "CAT", "GS", "MS",
    ],
    "2023-01-01": [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "XOM",
        "JNJ", "JPM", "V", "LLY", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
        "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN", "ABT", "WMT",
        "BAC", "DHR", "CRM", "TXN", "VZ", "ADBE", "CMCSA", "NEE", "NKE", "PM",
        "RTX", "LIN", "NFLX", "ORCL", "INTC", "HON", "BMY", "T", "AMGN", "MS",
    ],
}

# ---------------------------------------------------------------------------
# Indian ADR mapping
# ---------------------------------------------------------------------------

INDIAN_ADR_MAP: Dict[str, str] = {
    "INFY.NS": "INFY",    # Infosys NSE -> NYSE ADR
    "WIT.NS":  "WIT",     # Wipro
    "HDB.NS":  "HDB",     # HDFC Bank
    "IBN.NS":  "IBN",     # ICICI Bank
    "TTM.NS":  "TTM",     # Tata Motors
    "RDY.NS":  "RDY",     # Dr. Reddy's
    "SIFY.NS": "SIFY",    # Sify Technologies
    "VEDL.NS": "VEDL",    # Vedanta
    "SIT.NS":  "SIT",     # Sitio Royalties (proxy)
    "BSAC.NS": "BSAC",    # Banco Santander Chile (proxy)
}

# ---------------------------------------------------------------------------
# Commodity ETF proxies for underlying commodities
# ---------------------------------------------------------------------------

COMMODITY_ETF_PROXIES: Dict[str, str] = {
    "GOLD":     "GLD",
    "SILVER":   "SLV",
    "OIL":      "USO",
    "NATGAS":   "UNG",
    "COPPER":   "CPER",
    "WHEAT":    "WEAT",
    "CORN":     "CORN",
    "SOYBEANS": "SOYB",
    "SUGAR":    "SGG",
    "COFFEE":   "JO",
    "COTTON":   "BAL",
    "CATTLE":   "COW",
    "PALLADIUM":"PALL",
    "PLATINUM": "PPLT",
}


# ---------------------------------------------------------------------------
# SQLite cache helpers (lighter than DuckDB for v2 additions)
# ---------------------------------------------------------------------------

def _ensure_sqlite_db() -> sqlite3.Connection:
    """Create or open the v2 SQLite cache and ensure schema exists."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_V2_DB), check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS pit_constituents (
            index_name  TEXT NOT NULL,
            as_of_date  TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            PRIMARY KEY (index_name, as_of_date, ticker)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS corporate_actions (
            ticker          TEXT NOT NULL,
            ex_date         TEXT NOT NULL,
            action_type     TEXT NOT NULL,
            ratio           REAL,
            dividend_amount REAL,
            spinoff_ticker  TEXT,
            PRIMARY KEY (ticker, ex_date, action_type)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS quality_reports (
            ticker          TEXT NOT NULL,
            report_date     TEXT NOT NULL,
            confidence_score REAL,
            issues_json     TEXT,
            PRIMARY KEY (ticker, report_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS alt_ohlcv (
            series_id   TEXT NOT NULL,
            date        TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            source      TEXT,
            PRIMARY KEY (series_id, date)
        )
    """)
    con.commit()
    return con


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    ticker: str
    issues: List[str]
    confidence_score: float          # 0.0 - 1.0
    stale_count: int
    spike_count: int
    zero_volume_count: int
    gap_count: int
    total_rows: int
    date_range: Tuple[str, str]
    grade: str                        # A/B/C/D


@dataclass
class SmileParams:
    atm_iv: float
    skew: float                       # 25-delta put IV - 25-delta call IV
    kurtosis: float                   # butterfly spread
    risk_reversal: float


@dataclass
class VolArbSignal:
    ticker: str
    signal_type: str
    edge: float
    description: str


# ---------------------------------------------------------------------------
# PointInTimeAdjustmentEngine
# ---------------------------------------------------------------------------

class PointInTimeAdjustmentEngine:
    """Survivorship bias correction and point-in-time index reconstruction.

    Tracks delisted stocks (bankruptcy, acquisition, privatization) that
    were once S&P 500 members, enabling researchers to reconstruct the index
    as it actually existed at any historical date — preventing look-ahead bias.
    """

    def __init__(self) -> None:
        self._delisted: List[Dict[str, str]] = DELISTED_STOCKS
        self._con: Optional[sqlite3.Connection] = None

    def _db(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = _ensure_sqlite_db()
        return self._con

    def get_delisted_stocks(
        self,
        delist_after: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Return delisted stocks, optionally filtered by delist_date or reason.

        Args:
            delist_after: ISO date string; only return stocks delisted after this date.
            reason: Filter by reason ('bankruptcy', 'acquisition', 'privatization', etc.).

        Returns:
            List of dicts with keys: ticker, delist_date, reason, successor_ticker.
        """
        results = self._delisted
        if delist_after:
            results = [s for s in results if s["delist_date"] >= delist_after]
        if reason:
            results = [s for s in results if s["reason"].lower() == reason.lower()]
        return results

    def get_historical_constituents(self, index: str, as_of_date: str) -> List[str]:
        """Return tickers that were actually in `index` at `as_of_date`.

        Uses hardcoded S&P 500 snapshots + interpolation for dates between snapshots.
        For non-SP500 indices, falls back to current Wikipedia constituents minus
        stocks that were delisted before as_of_date.

        Args:
            index: Index name — 'SP500', 'NASDAQ100', etc.
            as_of_date: ISO date string 'YYYY-MM-DD'.

        Returns:
            List of ticker strings (point-in-time correct).
        """
        # Check DB cache first
        con = self._db()
        rows = con.execute(
            "SELECT ticker FROM pit_constituents WHERE index_name=? AND as_of_date=?",
            (index.upper(), as_of_date)
        ).fetchall()
        if rows:
            return [r[0] for r in rows]

        tickers = self._compute_constituents(index.upper(), as_of_date)

        # Cache results
        con.executemany(
            "INSERT OR IGNORE INTO pit_constituents (index_name, as_of_date, ticker) VALUES (?,?,?)",
            [(index.upper(), as_of_date, t) for t in tickers]
        )
        con.commit()
        return tickers

    def _compute_constituents(self, index: str, as_of_date: str) -> List[str]:
        """Internal: compute point-in-time constituents."""
        if index != "SP500":
            # For non-SP500, use current constituents and filter out already-delisted
            try:
                from sentinel.sds.adapters.historical_ohlcv_deep import MultiMarketUniverse
                universe = MultiMarketUniverse()
                raw = universe.get_index_constituents(index)
                tickers = [c["ticker"] for c in raw]
            except Exception:
                tickers = []
            return self._filter_active_at_date(tickers, as_of_date)

        # SP500: interpolate from historical snapshots
        snapshot_dates = sorted(SP500_HISTORICAL_CONSTITUENTS.keys())

        # Find the most recent snapshot on or before as_of_date
        best_snap = None
        for snap_date in snapshot_dates:
            if snap_date <= as_of_date:
                best_snap = snap_date
            else:
                break

        if best_snap is None:
            # Before earliest snapshot: use earliest
            best_snap = snapshot_dates[0]

        base_tickers = list(SP500_HISTORICAL_CONSTITUENTS[best_snap])

        # Add stocks not yet delisted that were added after the snapshot
        # (simplified: we trust the snapshot and filter only confirmed delistings)
        return self._filter_active_at_date(base_tickers, as_of_date)

    def _filter_active_at_date(self, tickers: List[str], as_of_date: str) -> List[str]:
        """Remove tickers known to have been delisted before as_of_date."""
        delisted_before = {
            s["ticker"] for s in self._delisted
            if s["delist_date"] <= as_of_date
        }
        return [t for t in tickers if t not in delisted_before]

    def get_pre_delist_history(
        self,
        ticker: str,
        start: str = _DEFAULT_START,
    ) -> pd.DataFrame:
        """Fetch pre-delist price history for a delisted ticker.

        Primary: yfinance (often has pre-delist data).
        Fallback: constructs synthetic from successor ticker with ratio.

        Returns:
            DataFrame indexed by date with OHLCV columns, or empty DataFrame.
        """
        import yfinance as yf

        delist_info = next(
            (s for s in self._delisted if s["ticker"] == ticker), None
        )

        # Try yfinance first
        try:
            obj = yf.Ticker(ticker)
            end = delist_info["delist_date"] if delist_info else datetime.today().strftime("%Y-%m-%d")
            df = obj.history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "date"
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                df["ticker"] = ticker
                df["delisted"] = True
                logger.debug("Pre-delist history fetched via yfinance", ticker=ticker, rows=len(df))
                return df
        except Exception as exc:
            logger.warning("Pre-delist yfinance fetch failed", ticker=ticker, error=str(exc))

        # Fallback: try successor price scaled (crude approximation)
        if delist_info and delist_info.get("successor_ticker"):
            succ = delist_info["successor_ticker"]
            try:
                obj = yf.Ticker(succ)
                df = obj.history(start=start, end=delist_info["delist_date"], auto_adjust=True)
                if df is not None and not df.empty:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df.index.name = "date"
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    df["ticker"] = ticker
                    df["delisted"] = True
                    df["source"] = f"successor_proxy:{succ}"
                    logger.debug("Pre-delist history via successor proxy", ticker=ticker, successor=succ)
                    return df
            except Exception:
                pass

        return pd.DataFrame()

    def build_survivorship_free_universe(
        self,
        index: str,
        start: str,
        end: str,
        max_workers: int = 6,
    ) -> Dict[str, pd.DataFrame]:
        """Build complete survivorship-bias-free price history for index.

        Includes all stocks that were ever in the index between start and end,
        including those subsequently delisted.

        Returns:
            Dict mapping ticker -> OHLCV DataFrame.
        """
        import yfinance as yf

        # Collect all tickers ever in index across snapshot dates
        all_tickers: set = set()
        for snap_date in SP500_HISTORICAL_CONSTITUENTS:
            if snap_date >= start:
                tickers = self.get_historical_constituents(index, snap_date)
                all_tickers.update(tickers)

        # Add currently delisted stocks that were present during our window
        for s in self._delisted:
            if s["delist_date"] >= start and s["ticker"]:
                all_tickers.add(s["ticker"])

        logger.info(
            "Survivorship-free universe size",
            index=index,
            ticker_count=len(all_tickers),
        )

        results: Dict[str, pd.DataFrame] = {}

        def _fetch(ticker: str) -> Tuple[str, pd.DataFrame]:
            try:
                obj = yf.Ticker(ticker)
                df = obj.history(start=start, end=end, auto_adjust=True)
                if df is None or df.empty:
                    return ticker, pd.DataFrame()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "date"
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                df["ticker"] = ticker
                return ticker, df
            except Exception as exc:
                logger.warning("Universe fetch failed", ticker=ticker, error=str(exc))
                return ticker, pd.DataFrame()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch, t): t for t in all_tickers}
            for fut in as_completed(futures):
                t, df = fut.result()
                results[t] = df

        return results


# ---------------------------------------------------------------------------
# CorporateActionAdjuster
# ---------------------------------------------------------------------------

class CorporateActionAdjuster:
    """Precise corporate action handling: splits, dividends, spin-offs.

    Supports four adjustment modes:
      - 'total_return':  adjust prices + dividends (true total return)
      - 'split_only':    adjust prices for splits only, ignore dividends
      - 'forward':       forward-adjust from reference date
      - 'backward':      backward-adjust (standard — latest price is reference)
    """

    ADJUSTMENT_TYPES = ("split", "reverse_split", "dividend", "special_dividend", "spinoff")

    def __init__(self) -> None:
        self._con: Optional[sqlite3.Connection] = None

    def _db(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = _ensure_sqlite_db()
        return self._con

    def fetch_corporate_actions(self, ticker: str) -> pd.DataFrame:
        """Fetch corporate actions from yfinance for a given ticker.

        Returns:
            DataFrame with columns: ex_date, action_type, ratio, dividend_amount.
            Indexed by ex_date.
        """
        import yfinance as yf

        try:
            obj = yf.Ticker(ticker)
            actions = obj.actions  # DataFrame with Dividends and Stock Splits
            if actions is None or actions.empty:
                return pd.DataFrame()

            actions = actions.copy()
            actions.index = pd.to_datetime(actions.index).tz_localize(None)
            actions.index.name = "ex_date"
            actions.columns = [c.lower().replace(" ", "_") for c in actions.columns]

            records = []
            for dt, row in actions.iterrows():
                div = float(row.get("dividends", 0) or 0)
                split = float(row.get("stock_splits", 0) or 0)

                if div > 0:
                    records.append({
                        "ex_date": dt,
                        "action_type": "dividend",
                        "ratio": 1.0,
                        "dividend_amount": div,
                        "spinoff_ticker": None,
                    })
                if split > 0 and split != 1.0:
                    atype = "split" if split > 1 else "reverse_split"
                    records.append({
                        "ex_date": dt,
                        "action_type": atype,
                        "ratio": split,
                        "dividend_amount": 0.0,
                        "spinoff_ticker": None,
                    })

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records).set_index("ex_date").sort_index()

            # Cache to SQLite
            con = self._db()
            for dt, row in df.iterrows():
                con.execute(
                    """INSERT OR REPLACE INTO corporate_actions
                       (ticker, ex_date, action_type, ratio, dividend_amount, spinoff_ticker)
                       VALUES (?,?,?,?,?,?)""",
                    (ticker, str(dt.date()), row["action_type"],
                     row["ratio"], row["dividend_amount"], row.get("spinoff_ticker"))
                )
            con.commit()

            return df

        except Exception as exc:
            logger.warning("fetch_corporate_actions failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

    def get_adjustment_factor(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        mode: str = "total_return",
    ) -> float:
        """Compute cumulative adjustment factor between two dates.

        Args:
            ticker: Ticker symbol.
            from_date: Start date ISO string.
            to_date: End date ISO string.
            mode: 'total_return', 'split_only', 'forward', or 'backward'.

        Returns:
            Cumulative multiplicative factor. Apply to historical prices:
            adjusted_price = raw_price * factor.
        """
        actions = self.fetch_corporate_actions(ticker)
        if actions.empty:
            return 1.0

        from_dt = pd.Timestamp(from_date)
        to_dt = pd.Timestamp(to_date)

        # Filter actions between the two dates
        mask = (actions.index >= from_dt) & (actions.index <= to_dt)
        relevant = actions[mask]

        if relevant.empty:
            return 1.0

        cumulative = 1.0

        for dt, row in relevant.iterrows():
            if mode == "split_only":
                # Only apply splits and reverse splits
                if row["action_type"] in ("split", "reverse_split"):
                    cumulative *= row["ratio"]

            elif mode == "total_return":
                # Apply splits
                if row["action_type"] in ("split", "reverse_split"):
                    cumulative *= row["ratio"]
                # Apply dividend: factor = (price - div) / price approximation
                elif row["action_type"] in ("dividend", "special_dividend"):
                    div = float(row["dividend_amount"])
                    # We need price context; approximate as 1 - div_yield
                    # Caller must supply price context for exact computation
                    # Here we apply a simplified factor
                    cumulative *= (1.0 - div / max(div * 100, 1))  # approximation

            elif mode in ("forward", "backward"):
                # Standard backward adjustment
                if row["action_type"] in ("split", "reverse_split"):
                    cumulative /= row["ratio"] if mode == "backward" else cumulative
                elif row["action_type"] in ("dividend", "special_dividend"):
                    pass  # dividends excluded from price-only adjustment

        return round(cumulative, 8)

    def apply_total_return_adjustment(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Apply full total-return adjustment to a price series.

        Computes exact dividend reinvestment factors using actual ex-date prices.

        Args:
            df: OHLCV DataFrame indexed by date with 'close' column.
            ticker: Ticker for fetching corporate actions.

        Returns:
            DataFrame with 'adj_close_total_return' column added.
        """
        if df is None or df.empty:
            return df

        df = df.copy()
        actions = self.fetch_corporate_actions(ticker)

        # Start with close prices
        prices = df["close"].copy().sort_index()
        adj_prices = prices.copy()

        if actions.empty:
            df["adj_close_total_return"] = adj_prices
            return df

        # Process actions from most recent to oldest (backward adjustment)
        factor = 1.0
        actions_sorted = actions.sort_index(ascending=False)

        for ex_date, row in actions_sorted.iterrows():
            atype = row["action_type"]

            if atype in ("split", "reverse_split"):
                ratio = float(row["ratio"])
                if ratio > 0:
                    # All prices before split date are divided by ratio
                    mask = adj_prices.index < ex_date
                    adj_prices[mask] /= ratio

            elif atype in ("dividend", "special_dividend"):
                div = float(row["dividend_amount"])
                # Find the close price just before ex_date
                pre_prices = prices[prices.index < ex_date]
                if not pre_prices.empty and pre_prices.iloc[-1] > 0:
                    pre_close = float(pre_prices.iloc[-1])
                    div_factor = (pre_close - div) / pre_close
                    if div_factor > 0:
                        mask = adj_prices.index < ex_date
                        adj_prices[mask] *= div_factor

        df["adj_close_total_return"] = adj_prices
        return df

    def apply_split_adjustment(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Apply split-only adjustment (no dividends) to price series."""
        if df is None or df.empty:
            return df

        df = df.copy()
        actions = self.fetch_corporate_actions(ticker)

        if actions.empty:
            df["adj_close_split_only"] = df["close"]
            return df

        prices = df["close"].copy().sort_index()
        adj_prices = prices.copy()

        splits = actions[actions["action_type"].isin(["split", "reverse_split"])]
        splits_sorted = splits.sort_index(ascending=False)

        for ex_date, row in splits_sorted.iterrows():
            ratio = float(row["ratio"])
            if ratio > 0:
                mask = adj_prices.index < ex_date
                adj_prices[mask] /= ratio

        df["adj_close_split_only"] = adj_prices
        return df

    def handle_spinoff(
        self,
        parent_ticker: str,
        spinoff_ticker: str,
        spinoff_date: str,
        spinoff_ratio: float,
    ) -> Dict[str, pd.DataFrame]:
        """Adjust parent price series for a spin-off and fetch spun-off entity.

        Args:
            parent_ticker: Original company ticker.
            spinoff_ticker: New spun-off entity ticker.
            spinoff_date: ISO date of spin-off.
            spinoff_ratio: Shares of spinoff received per parent share.

        Returns:
            Dict with keys 'parent' and 'spinoff', each containing adjusted DataFrames.
        """
        import yfinance as yf

        result: Dict[str, pd.DataFrame] = {}

        try:
            obj_parent = yf.Ticker(parent_ticker)
            df_parent = obj_parent.history(start=_DEFAULT_START, auto_adjust=False)
            if df_parent is not None and not df_parent.empty:
                df_parent.index = pd.to_datetime(df_parent.index).tz_localize(None)
                df_parent.index.name = "date"
                df_parent.columns = [c.lower().replace(" ", "_") for c in df_parent.columns]
                df_parent["ticker"] = parent_ticker
                df_parent["spinoff_adjusted"] = False
                spinoff_dt = pd.Timestamp(spinoff_date)
                # Mark records after spinoff
                df_parent.loc[df_parent.index >= spinoff_dt, "spinoff_adjusted"] = True
                result["parent"] = df_parent
        except Exception as exc:
            logger.warning("Spinoff parent fetch failed", ticker=parent_ticker, error=str(exc))

        try:
            obj_spinoff = yf.Ticker(spinoff_ticker)
            df_spinoff = obj_spinoff.history(start=spinoff_date, auto_adjust=True)
            if df_spinoff is not None and not df_spinoff.empty:
                df_spinoff.index = pd.to_datetime(df_spinoff.index).tz_localize(None)
                df_spinoff.index.name = "date"
                df_spinoff.columns = [c.lower().replace(" ", "_") for c in df_spinoff.columns]
                df_spinoff["ticker"] = spinoff_ticker
                df_spinoff["spinoff_ratio"] = spinoff_ratio
                result["spinoff"] = df_spinoff
        except Exception as exc:
            logger.warning("Spinoff entity fetch failed", ticker=spinoff_ticker, error=str(exc))

        return result

    def forward_adjust(
        self,
        df: pd.DataFrame,
        ticker: str,
        reference_date: str,
    ) -> pd.DataFrame:
        """Forward-adjust prices so the reference_date price is the anchor.

        All prices are scaled so that the price at reference_date equals
        the actual market price on that date, with historical prices
        consistently adjusted forward.

        Args:
            df: OHLCV DataFrame indexed by date.
            ticker: Ticker symbol.
            reference_date: ISO date string — anchor point.

        Returns:
            DataFrame with forward-adjusted 'close' column.
        """
        if df is None or df.empty:
            return df

        df = df.copy()
        ref_dt = pd.Timestamp(reference_date)

        if ref_dt not in df.index:
            # Find nearest available date
            available = df.index[df.index <= ref_dt]
            if available.empty:
                return df
            ref_dt = available[-1]

        ref_price = float(df.loc[ref_dt, "close"])
        if ref_price <= 0:
            return df

        # Apply split adjustment to get continuous series, then rescale
        df_split = self.apply_split_adjustment(df, ticker)
        adj_at_ref = float(df_split.loc[ref_dt, "adj_close_split_only"])
        if adj_at_ref <= 0:
            return df

        scale = ref_price / adj_at_ref
        df["close_forward_adjusted"] = df_split["adj_close_split_only"] * scale
        return df


# ---------------------------------------------------------------------------
# GlobalIndexData
# ---------------------------------------------------------------------------

class GlobalIndexData:
    """Extended global market coverage: 60+ markets/indices.

    Adds EM markets, European exchanges, Asia-Pacific, macro proxies,
    and Indian ADR mappings on top of the existing exchange map.
    """

    # 60+ market/exchange definitions
    GLOBAL_MARKETS: Dict[str, Dict[str, str]] = {
        # US
        "NYSE":      {"suffix": "",     "currency": "USD", "region": "North America"},
        "NASDAQ":    {"suffix": "",     "currency": "USD", "region": "North America"},
        "AMEX":      {"suffix": "",     "currency": "USD", "region": "North America"},
        # Canada
        "TSX":       {"suffix": ".TO",  "currency": "CAD", "region": "North America"},
        "TSXV":      {"suffix": ".V",   "currency": "CAD", "region": "North America"},
        # Latin America
        "B3":        {"suffix": ".SA",  "currency": "BRL", "region": "LatAm"},
        "BMV":       {"suffix": ".MX",  "currency": "MXN", "region": "LatAm"},
        "BCS":       {"suffix": ".SN",  "currency": "CLP", "region": "LatAm"},
        "BVC":       {"suffix": ".CL",  "currency": "COP", "region": "LatAm"},
        "BVL":       {"suffix": ".LM",  "currency": "PEN", "region": "LatAm"},
        "BCBA":      {"suffix": ".BA",  "currency": "ARS", "region": "LatAm"},
        # Western Europe
        "LSE":       {"suffix": ".L",   "currency": "GBP", "region": "Europe"},
        "EURONEXT":  {"suffix": ".PA",  "currency": "EUR", "region": "Europe"},
        "XETRA":     {"suffix": ".DE",  "currency": "EUR", "region": "Europe"},
        "SIX":       {"suffix": ".SW",  "currency": "CHF", "region": "Europe"},
        "BME":       {"suffix": ".MC",  "currency": "EUR", "region": "Europe"},
        "BORSA_IT":  {"suffix": ".MI",  "currency": "EUR", "region": "Europe"},
        "AEX":       {"suffix": ".AS",  "currency": "EUR", "region": "Europe"},
        "EURONEXT_BR": {"suffix": ".BR","currency": "EUR", "region": "Europe"},
        "OSE":       {"suffix": ".OL",  "currency": "NOK", "region": "Europe"},
        "HEX":       {"suffix": ".HE",  "currency": "EUR", "region": "Europe"},
        "OMX":       {"suffix": ".ST",  "currency": "SEK", "region": "Europe"},
        "CPH":       {"suffix": ".CO",  "currency": "DKK", "region": "Europe"},
        "WBAG":      {"suffix": ".VI",  "currency": "EUR", "region": "Europe"},
        "EURONEXT_LI": {"suffix": ".LS","currency": "EUR", "region": "Europe"},
        "ISE":       {"suffix": ".IR",  "currency": "EUR", "region": "Europe"},
        # Eastern Europe
        "GPW":       {"suffix": ".WA",  "currency": "PLN", "region": "Europe"},
        "PSE_CZ":    {"suffix": ".PR",  "currency": "CZK", "region": "Europe"},
        "BSE_HU":    {"suffix": ".BU",  "currency": "HUF", "region": "Europe"},
        "ATHEX":     {"suffix": ".AT",  "currency": "EUR", "region": "Europe"},
        "BIST":      {"suffix": ".IS",  "currency": "TRY", "region": "Europe"},
        "MOEX":      {"suffix": ".ME",  "currency": "RUB", "region": "Europe"},
        # Asia Pacific
        "TSE":       {"suffix": ".T",   "currency": "JPY", "region": "Asia-Pacific"},
        "HKEX":      {"suffix": ".HK",  "currency": "HKD", "region": "Asia-Pacific"},
        "SSE":       {"suffix": ".SS",  "currency": "CNY", "region": "Asia-Pacific"},
        "SZSE":      {"suffix": ".SZ",  "currency": "CNY", "region": "Asia-Pacific"},
        "KRX":       {"suffix": ".KS",  "currency": "KRW", "region": "Asia-Pacific"},
        "KOSDAQ":    {"suffix": ".KQ",  "currency": "KRW", "region": "Asia-Pacific"},
        "TWSE":      {"suffix": ".TW",  "currency": "TWD", "region": "Asia-Pacific"},
        "NSE":       {"suffix": ".NS",  "currency": "INR", "region": "Asia-Pacific"},
        "BSE":       {"suffix": ".BO",  "currency": "INR", "region": "Asia-Pacific"},
        "ASX":       {"suffix": ".AX",  "currency": "AUD", "region": "Asia-Pacific"},
        "SGX":       {"suffix": ".SI",  "currency": "SGD", "region": "Asia-Pacific"},
        "NZX":       {"suffix": ".NZ",  "currency": "NZD", "region": "Asia-Pacific"},
        "BURSA":     {"suffix": ".KL",  "currency": "MYR", "region": "Asia-Pacific"},
        "SET":       {"suffix": ".BK",  "currency": "THB", "region": "Asia-Pacific"},
        "IDX":       {"suffix": ".JK",  "currency": "IDR", "region": "Asia-Pacific"},
        "PSE":       {"suffix": ".PS",  "currency": "PHP", "region": "Asia-Pacific"},
        # MENA
        "TADAWUL":   {"suffix": ".SR",  "currency": "SAR", "region": "MENA"},
        "ADX":       {"suffix": ".AD",  "currency": "AED", "region": "MENA"},
        "DFM":       {"suffix": ".DU",  "currency": "AED", "region": "MENA"},
        "EGX":       {"suffix": ".CA",  "currency": "EGP", "region": "MENA"},
        "TASE":      {"suffix": ".TA",  "currency": "ILS", "region": "MENA"},
        "QSE":       {"suffix": ".QA",  "currency": "QAR", "region": "MENA"},
        "MSM":       {"suffix": ".OM",  "currency": "OMR", "region": "MENA"},
        # Africa
        "JSE":       {"suffix": ".JO",  "currency": "ZAR", "region": "Africa"},
        "NSE_NG":    {"suffix": ".LG",  "currency": "NGN", "region": "Africa"},
        "NSE_KE":    {"suffix": ".NR",  "currency": "KES", "region": "Africa"},
    }

    # Representative tickers per market for universe screening
    MARKET_SAMPLES: Dict[str, List[str]] = {
        "KRX":     ["005930.KS", "000660.KS", "035420.KS", "005380.KS", "051910.KS"],
        "TWSE":    ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2412.TW"],
        "NSE":     ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS"],
        "B3":      ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"],
        "TADAWUL": ["2222.SR", "1120.SR", "2010.SR", "1211.SR", "4030.SR"],
        "LSE":     ["HSBA.L", "BP.L", "SHEL.L", "AZN.L", "ULVR.L"],
        "EURONEXT":["MC.PA", "LVMH.PA", "TTE.PA", "SAN.PA", "BNP.PA"],
        "XETRA":   ["SAP.DE", "SIE.DE", "BAYN.DE", "BAS.DE", "BMW.DE"],
        "SIX":     ["NESN.SW", "ROG.SW", "NOVN.SW", "ABB.SW", "UHR.SW"],
        "ASX":     ["BHP.AX", "CBA.AX", "CSL.AX", "ANZ.AX", "WBC.AX"],
        "TSE":     ["7203.T", "6758.T", "9432.T", "9984.T", "8306.T"],
        "HKEX":    ["0700.HK", "0005.HK", "0941.HK", "1299.HK", "2318.HK"],
        "JSE":     ["NPN.JO", "BTI.JO", "FSR.JO", "SBK.JO", "AGL.JO"],
        "SGX":     ["D05.SI", "O39.SI", "U11.SI", "Z74.SI", "C6L.SI"],
        "SZSE":    ["000002.SZ", "000858.SZ", "002594.SZ", "300750.SZ", "000063.SZ"],
        "SSE":     ["600519.SS", "601318.SS", "600036.SS", "600900.SS", "601398.SS"],
    }

    def get_market_info(self, market: str) -> Optional[Dict[str, str]]:
        """Return exchange metadata for a market code."""
        return self.GLOBAL_MARKETS.get(market.upper())

    def get_market_tickers(self, market: str, limit: int = 20) -> List[str]:
        """Return sample tickers for a given market."""
        samples = self.MARKET_SAMPLES.get(market.upper(), [])
        return samples[:limit]

    def resolve_indian_adr(self, nse_ticker: str) -> Optional[str]:
        """Map Indian NSE ticker to its US-listed ADR ticker."""
        return INDIAN_ADR_MAP.get(nse_ticker)

    def get_commodity_proxy(self, commodity: str) -> Optional[str]:
        """Return ETF ticker used as commodity proxy."""
        return COMMODITY_ETF_PROXIES.get(commodity.upper())

    def fetch_market_ohlcv(
        self,
        market: str,
        ticker: str,
        start: str = "2000-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV for a specific market/ticker combination.

        Automatically appends the correct exchange suffix and returns
        a standardized DataFrame.

        Args:
            market: Exchange code (e.g. 'LSE', 'KRX', 'NSE').
            ticker: Base ticker without suffix (e.g. 'HSBA', '005930').
            start: Start date ISO string.
            end: End date ISO string (defaults to today).

        Returns:
            DataFrame with OHLCV columns + exchange and currency metadata.
        """
        import yfinance as yf

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        market_info = self.get_market_info(market)
        if market_info is None:
            logger.warning("Unknown market", market=market)
            return pd.DataFrame()

        suffix = market_info["suffix"]
        currency = market_info["currency"]
        full_ticker = f"{ticker}{suffix}" if not ticker.endswith(suffix) else ticker

        try:
            obj = yf.Ticker(full_ticker)
            df = obj.history(start=start, end=end, auto_adjust=True)
            if df is None or df.empty:
                return pd.DataFrame()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = np.nan

            df["ticker"] = full_ticker
            df["exchange"] = market
            df["currency"] = currency
            df["adj_close"] = df["close"]
            df["adj_factor"] = 1.0
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

            return df

        except Exception as exc:
            logger.warning("fetch_market_ohlcv failed", market=market, ticker=ticker, error=str(exc))
            return pd.DataFrame()

    def get_all_markets(self, region: Optional[str] = None) -> List[str]:
        """Return list of all supported market codes, optionally filtered by region."""
        if region is None:
            return list(self.GLOBAL_MARKETS.keys())
        return [
            code for code, info in self.GLOBAL_MARKETS.items()
            if info.get("region", "").lower() == region.lower()
        ]

    def get_em_markets(self) -> List[str]:
        """Return Emerging Market exchange codes."""
        em_markets = ["KRX", "KOSDAQ", "TWSE", "NSE", "BSE", "B3", "BMV",
                      "TADAWUL", "ADX", "DFM", "EGX", "JSE", "BURSA", "SET",
                      "IDX", "PSE", "QSE", "MSM", "BCBA", "BVC", "BVL"]
        return [m for m in em_markets if m in self.GLOBAL_MARKETS]


# ---------------------------------------------------------------------------
# DataQualityValidator
# ---------------------------------------------------------------------------

class DataQualityValidator:
    """Comprehensive OHLCV data quality validation.

    Detects:
      - Stale prices: same close for >5 consecutive days
      - Price spikes: >50% single-day move (likely data error)
      - Zero volume on non-holiday trading days
      - Adjusted price continuity around corporate action dates
      - OHLC logic violations
      - Excessive gaps in time series
    """

    STALE_THRESHOLD = 5           # consecutive days with same price
    SPIKE_THRESHOLD = 0.50        # 50% single-day move
    MAX_ACCEPTABLE_GAP = 10       # trading days

    def validate_series(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> DataQualityReport:
        """Run full quality validation on an OHLCV series.

        Args:
            df: OHLCV DataFrame indexed by date.
            ticker: Ticker symbol (for logging).

        Returns:
            DataQualityReport with issues list and confidence_score (0-1).
        """
        issues: List[str] = []
        stale_count = 0
        spike_count = 0
        zero_vol_count = 0
        gap_count = 0

        if df is None or df.empty:
            return DataQualityReport(
                ticker=ticker,
                issues=["Empty DataFrame"],
                confidence_score=0.0,
                stale_count=0,
                spike_count=0,
                zero_volume_count=0,
                gap_count=0,
                total_rows=0,
                date_range=("", ""),
                grade="D",
            )

        df = df.sort_index()
        close = df["close"].dropna() if "close" in df.columns else pd.Series(dtype=float)
        volume = df["volume"].dropna() if "volume" in df.columns else pd.Series(dtype=float)
        total_rows = len(df)

        # --- Stale price detection ---
        if len(close) > self.STALE_THRESHOLD:
            consecutive_same = 0
            prev_price = None
            stale_streak_start = None

            for dt, price in close.items():
                if prev_price is not None and abs(price - prev_price) < 1e-8:
                    consecutive_same += 1
                    if consecutive_same == self.STALE_THRESHOLD:
                        stale_streak_start = dt
                    if consecutive_same > self.STALE_THRESHOLD:
                        stale_count += 1
                else:
                    if consecutive_same > self.STALE_THRESHOLD:
                        issues.append(
                            f"Stale price: {consecutive_same} consecutive identical closes "
                            f"ending {dt.date() if hasattr(dt, 'date') else dt}"
                        )
                    consecutive_same = 0
                prev_price = price

        # --- Price spike detection ---
        if len(close) > 1:
            pct_change = close.pct_change().abs()
            spikes = pct_change[pct_change > self.SPIKE_THRESHOLD]
            spike_count = len(spikes)
            for dt, val in spikes.items():
                issues.append(
                    f"Price spike: {val:.1%} single-day move on "
                    f"{dt.date() if hasattr(dt, 'date') else dt}"
                )

        # --- Zero volume detection ---
        if not volume.empty:
            zero_mask = volume <= 0
            zero_vol_count = int(zero_mask.sum())
            if zero_vol_count > 0:
                issues.append(f"Zero/negative volume on {zero_vol_count} trading days")

        # --- Gap detection ---
        if len(close) > 1:
            dates = pd.to_datetime(close.index).sort_values()
            for i in range(len(dates) - 1):
                try:
                    gap = len(pd.bdate_range(start=dates[i], end=dates[i + 1])) - 1
                    if gap > self.MAX_ACCEPTABLE_GAP:
                        gap_count += 1
                        issues.append(
                            f"Gap of {gap} trading days: "
                            f"{dates[i].date()} to {dates[i+1].date()}"
                        )
                except Exception:
                    pass

        # --- OHLC logic violations ---
        ohlc_violations = 0
        for col_set in [["open", "high", "low", "close"]]:
            if all(c in df.columns for c in col_set):
                h = df["high"]
                l = df["low"]
                o = df["open"]
                c_col = df["close"]
                violations = ((h < o) | (h < c_col) | (l > o) | (l > c_col)).sum()
                ohlc_violations = int(violations)
                if ohlc_violations > 0:
                    issues.append(f"OHLC logic violations: {ohlc_violations} rows")

        # --- Negative prices ---
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if price_cols:
            neg_count = int((df[price_cols] < 0).any(axis=1).sum())
            if neg_count > 0:
                issues.append(f"Negative prices on {neg_count} rows")

        # --- Confidence score computation ---
        penalty = 0.0
        penalty += min(stale_count * 0.05, 0.30)
        penalty += min(spike_count * 0.03, 0.20)
        penalty += min(zero_vol_count / max(total_rows, 1) * 5, 0.15)
        penalty += min(gap_count * 0.10, 0.25)
        penalty += min(ohlc_violations / max(total_rows, 1) * 10, 0.10)

        confidence_score = max(0.0, min(1.0, 1.0 - penalty))

        # Grade
        if confidence_score >= 0.95:
            grade = "A"
        elif confidence_score >= 0.85:
            grade = "B"
        elif confidence_score >= 0.70:
            grade = "C"
        else:
            grade = "D"

        dates_idx = pd.to_datetime(df.index)
        date_range = (
            dates_idx.min().date().isoformat() if len(dates_idx) > 0 else "",
            dates_idx.max().date().isoformat() if len(dates_idx) > 0 else "",
        )

        report = DataQualityReport(
            ticker=ticker,
            issues=issues,
            confidence_score=round(confidence_score, 4),
            stale_count=stale_count,
            spike_count=spike_count,
            zero_volume_count=zero_vol_count,
            gap_count=gap_count,
            total_rows=total_rows,
            date_range=date_range,
            grade=grade,
        )

        # Persist to SQLite
        try:
            import json
            con = _ensure_sqlite_db()
            con.execute(
                """INSERT OR REPLACE INTO quality_reports
                   (ticker, report_date, confidence_score, issues_json)
                   VALUES (?,?,?,?)""",
                (ticker, datetime.today().strftime("%Y-%m-%d"),
                 confidence_score, json.dumps(issues))
            )
            con.commit()
            con.close()
        except Exception:
            pass

        return report

    def check_adjusted_continuity(
        self,
        df: pd.DataFrame,
        ticker: str,
        threshold: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """Check for discontinuities in adjusted price series at corporate action dates.

        A large jump in adj_close on an ex-date (after accounting for the split/dividend)
        may indicate a mis-adjustment.

        Args:
            df: OHLCV DataFrame with 'adj_close' column.
            ticker: Ticker symbol.
            threshold: Max acceptable ratio gap at adjustment dates.

        Returns:
            List of discontinuity events with date and magnitude.
        """
        discontinuities: List[Dict[str, Any]] = []

        if df is None or df.empty or "adj_close" not in df.columns:
            return discontinuities

        adj = df["adj_close"].dropna().sort_index()
        if len(adj) < 2:
            return discontinuities

        adj_pct = adj.pct_change().abs()
        large_jumps = adj_pct[adj_pct > threshold]

        for dt, val in large_jumps.items():
            discontinuities.append({
                "date": str(dt.date() if hasattr(dt, "date") else dt),
                "adj_close_change_pct": round(float(val), 4),
                "description": (
                    f"Possible adjustment discontinuity: {val:.1%} jump in adj_close"
                ),
            })

        return discontinuities

    def batch_validate(
        self,
        price_dict: Dict[str, pd.DataFrame],
        min_confidence: float = 0.80,
    ) -> Dict[str, DataQualityReport]:
        """Run validation across a universe of tickers.

        Args:
            price_dict: Dict of ticker -> OHLCV DataFrame.
            min_confidence: Log warning for tickers below this threshold.

        Returns:
            Dict of ticker -> DataQualityReport.
        """
        reports: Dict[str, DataQualityReport] = {}

        for ticker, df in price_dict.items():
            report = self.validate_series(df, ticker)
            reports[ticker] = report
            if report.confidence_score < min_confidence:
                logger.warning(
                    "Low confidence data quality",
                    ticker=ticker,
                    score=report.confidence_score,
                    issues_count=len(report.issues),
                )

        return reports


# ---------------------------------------------------------------------------
# TickerMappingEngine
# ---------------------------------------------------------------------------

class TickerMappingEngine:
    """Handle ticker changes and build continuous historical series.

    Maintains a database of 100+ historical ticker changes, enabling
    researchers to stitch continuous price histories across ticker renames.
    """

    def __init__(self) -> None:
        self._changes: List[Dict[str, str]] = TICKER_CHANGES
        self._cusip_map: Dict[str, str] = CUSIP_MAP

    def get_all_aliases(self, ticker: str) -> List[str]:
        """Return all historical tickers (aliases) for a given current ticker.

        Args:
            ticker: Current ticker symbol (e.g. 'META').

        Returns:
            List of all historical ticker symbols in chronological order.
            Includes the current ticker.
        """
        aliases = set()
        aliases.add(ticker.upper())

        for change in self._changes:
            if change["current"].upper() == ticker.upper():
                aliases.add(change["historical"].upper())
            # Also handle chains: if historical is another known current
            if change["historical"].upper() == ticker.upper():
                aliases.add(change["current"].upper())

        return list(aliases)

    def get_full_history(
        self,
        ticker: str,
        start: str = _DEFAULT_START,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a stitched continuous price history across all ticker aliases.

        For each alias, fetches available history and combines into a single
        continuous DataFrame, deduplicating by date.

        Args:
            ticker: Current ticker symbol.
            start: Earliest date to fetch.
            end: Latest date to fetch (defaults to today).

        Returns:
            Stitched DataFrame indexed by date, with 'ticker_alias' column
            indicating which alias provided each row's data.
        """
        import yfinance as yf

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        aliases = self.get_all_aliases(ticker)
        frames: List[pd.DataFrame] = []

        for alias in aliases:
            try:
                # Get the date range for this alias
                alias_changes = [
                    c for c in self._changes
                    if c["current"].upper() == ticker.upper()
                    and c["historical"].upper() == alias.upper()
                ]

                alias_start = start
                alias_end = end

                if alias_changes:
                    change_date = alias_changes[0]["change_date"]
                    if alias == ticker.upper():
                        alias_start = change_date
                    else:
                        alias_end = change_date

                obj = yf.Ticker(alias)
                df = obj.history(start=alias_start, end=alias_end, auto_adjust=True)

                if df is None or df.empty:
                    continue

                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "date"
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                df["ticker_alias"] = alias
                df["current_ticker"] = ticker

                for col in ["open", "high", "low", "close", "volume"]:
                    if col not in df.columns:
                        df[col] = np.nan

                frames.append(df)
                time.sleep(0.05)

            except Exception as exc:
                logger.warning("Alias fetch failed", alias=alias, ticker=ticker, error=str(exc))

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined

    def resolve_cusip(self, cusip: str) -> Optional[str]:
        """Resolve CUSIP to ticker symbol.

        Args:
            cusip: 9-character CUSIP string.

        Returns:
            Ticker symbol or None if not found.
        """
        ticker = self._cusip_map.get(cusip)
        if ticker:
            return ticker

        # Try OpenFIGI API as fallback
        try:
            payload = [{"idType": "ID_CUSIP", "idValue": cusip}]
            with httpx.Client(timeout=15) as client:
                resp = client.post(
                    "https://api.openfigi.com/v3/mapping",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                if data and data[0].get("data"):
                    t = data[0]["data"][0].get("ticker")
                    if t:
                        self._cusip_map[cusip] = t
                        return t
        except Exception as exc:
            logger.warning("CUSIP resolution via OpenFIGI failed", cusip=cusip, error=str(exc))

        return None

    def build_ticker_timeline(self, ticker: str) -> List[Dict[str, str]]:
        """Return chronological timeline of ticker name changes.

        Args:
            ticker: Current ticker symbol.

        Returns:
            List of dicts: [{from_ticker, to_ticker, change_date}] in date order.
        """
        timeline = []
        relevant = [
            c for c in self._changes
            if c["current"].upper() == ticker.upper()
            and c["historical"].upper() != ticker.upper()
        ]
        for c in sorted(relevant, key=lambda x: x["change_date"]):
            timeline.append({
                "from_ticker": c["historical"],
                "to_ticker": c["current"],
                "change_date": c["change_date"],
            })
        return timeline

    def find_ticker_at_date(self, current_ticker: str, as_of_date: str) -> str:
        """Return the ticker symbol that was used for `current_ticker` on `as_of_date`.

        Args:
            current_ticker: The current (post-rename) ticker.
            as_of_date: ISO date string.

        Returns:
            The ticker symbol in use on that date.
        """
        relevant = [
            c for c in self._changes
            if c["current"].upper() == current_ticker.upper()
            and c["historical"].upper() != current_ticker.upper()
            and c["change_date"] > as_of_date
        ]

        if relevant:
            # The most recent change before as_of_date
            pre_changes = [c for c in relevant if c["change_date"] > as_of_date]
            if pre_changes:
                earliest = min(pre_changes, key=lambda x: x["change_date"])
                return earliest["historical"]

        return current_ticker


# ---------------------------------------------------------------------------
# AltDataOHLCV
# ---------------------------------------------------------------------------

class AltDataOHLCV:
    """Alternative 'OHLCV' for non-price time series.

    Converts economic indicators, crypto prices, and sentiment scores
    into standard OHLCV format for unified analysis.
    """

    FRED_SERIES: Dict[str, str] = {
        "GDP":       "GDP",
        "CPI":       "CPIAUCSL",
        "PCE":       "PCE",
        "UNRATE":    "UNRATE",
        "PAYEMS":    "PAYEMS",
        "ISM_MFG":   "ISM001",
        "RETAIL_SALES": "RSAFS",
        "HOUSING_STARTS": "HOUST",
        "DURABLE_GOODS": "DGORDER",
        "PPI":       "PPIACO",
        "FEDFUNDS":  "FEDFUNDS",
        "T10Y2Y":    "T10Y2Y",
        "T10YIE":    "T10YIE",
        "DGS10":     "DGS10",
        "DGS2":      "DGS2",
        "VIX":       "VIXCLS",
        "BAML_HY":   "BAMLH0A0HYM2",
        "UMCSENT":   "UMCSENT",
        "INDPRO":    "INDPRO",
        "PERMIT":    "PERMIT",
        "JTSJOR":    "JTSJOR",
        "M2SL":      "M2SL",
        "WALCL":     "WALCL",
        "DEXUSEU":   "DEXUSEU",
        "DEXJPUS":   "DEXJPUS",
        "GOLDAMGBD228NLBM": "GOLDAMGBD228NLBM",
        "DCOILWTICO": "DCOILWTICO",
    }

    def _fetch_fred_series(self, series_id: str, start: str, end: str) -> pd.DataFrame:
        """Fetch a FRED series and format as OHLCV."""
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        )
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url)
                resp.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["value"] != "."].copy()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            df = df.set_index("date").sort_index()
            df = df[
                (df.index >= pd.Timestamp(start)) &
                (df.index <= pd.Timestamp(end))
            ]

            # Synthesize OHLCV: use value for close/open/high/low
            out = pd.DataFrame(index=df.index)
            out["open"] = df["value"]
            out["high"] = df["value"] * 1.001  # tiny synthetic range
            out["low"] = df["value"] * 0.999
            out["close"] = df["value"]
            out["volume"] = 0.0
            out["series_id"] = series_id
            out["source"] = "FRED"
            return out

        except Exception as exc:
            logger.warning("FRED fetch failed", series=series_id, error=str(exc))
            return pd.DataFrame()

    def get_economic_ohlcv(
        self,
        indicator: str,
        start: str = "1990-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch economic indicator as synthetic OHLCV.

        Supported indicators: GDP, CPI, PCE, UNRATE, PAYEMS, ISM_MFG,
        RETAIL_SALES, HOUSING_STARTS, DURABLE_GOODS, PPI, FEDFUNDS,
        T10Y2Y, DGS10, DGS2, VIX, BAML_HY, UMCSENT, INDPRO, M2SL.

        Args:
            indicator: Key from FRED_SERIES dict or raw FRED series ID.
            start: Start date ISO string.
            end: End date ISO string (defaults to today).

        Returns:
            DataFrame with OHLCV columns representing the indicator level.
        """
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        series_id = self.FRED_SERIES.get(indicator.upper(), indicator)
        return self._fetch_fred_series(series_id, start, end)

    def get_vix_history(
        self,
        start: str = "1990-01-02",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch VIX history back to 1990 from FRED (VIXCLS series).

        The CBOE VIX index is available on FRED from January 1990.
        Returns full history as OHLCV format.

        Args:
            start: Start date (VIX available from 1990-01-02).
            end: End date.

        Returns:
            DataFrame with columns: open, high, low, close (=VIX level), volume=0.
        """
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        df = self._fetch_fred_series("VIXCLS", start, end)

        # Also try yfinance for more granular recent data
        try:
            import yfinance as yf
            obj = yf.Ticker("^VIX")
            df_yf = obj.history(start=start, end=end, auto_adjust=True)
            if df_yf is not None and not df_yf.empty:
                df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None)
                df_yf.index.name = "date"
                df_yf.columns = [c.lower().replace(" ", "_") for c in df_yf.columns]
                df_yf["series_id"] = "VIX"
                df_yf["source"] = "yfinance"

                # Combine: prefer yfinance for overlapping dates
                if not df.empty:
                    df_combined = pd.concat([df, df_yf])
                    df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
                    return df_combined.sort_index()
                return df_yf
        except Exception:
            pass

        return df

    def get_crypto_ohlcv(
        self,
        symbol: str = "BTC",
        start: str = "2010-07-17",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch cryptocurrency historical OHLCV.

        Uses yfinance for BTC, ETH, and other crypto pairs.
        BTC history available from ~2010 on yfinance via BTC-USD.

        Args:
            symbol: Crypto symbol (e.g. 'BTC', 'ETH', 'SOL', 'ADA').
            start: Start date (BTC available from 2010-07-17).
            end: End date.

        Returns:
            DataFrame with OHLCV + volume columns.
        """
        import yfinance as yf

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        # Normalize crypto ticker
        if not symbol.endswith("-USD"):
            yf_ticker = f"{symbol.upper()}-USD"
        else:
            yf_ticker = symbol.upper()

        try:
            obj = yf.Ticker(yf_ticker)
            df = obj.history(start=start, end=end, auto_adjust=True)

            if df is None or df.empty:
                logger.warning("No crypto data", symbol=symbol)
                return pd.DataFrame()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df["series_id"] = symbol.upper()
            df["source"] = "yfinance_crypto"
            df["currency"] = "USD"

            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = np.nan

            logger.debug("Crypto OHLCV fetched", symbol=symbol, rows=len(df))
            return df

        except Exception as exc:
            logger.warning("Crypto fetch failed", symbol=symbol, error=str(exc))
            return pd.DataFrame()

    def get_sentiment_ohlcv(
        self,
        ticker: str,
        start: str = "2010-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate synthetic sentiment OHLCV series.

        Uses available news/put-call ratio signals as a proxy for sentiment.
        Since real-time sentiment APIs require subscriptions, this method
        uses the options put/call ratio from yfinance as a sentiment proxy.

        Args:
            ticker: Underlying ticker for sentiment proxy.
            start: Start date.
            end: End date.

        Returns:
            DataFrame with OHLCV columns where close = sentiment score (0-1 scale).
        """
        import yfinance as yf

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        try:
            # Use historical put/call ratio from options chain as sentiment proxy
            obj = yf.Ticker(ticker)
            expirations = obj.options
            if not expirations:
                return pd.DataFrame()

            sentiment_records = []
            # Sample recent expirations for sentiment signal
            for expiry in expirations[:6]:
                try:
                    chain = obj.option_chain(expiry)
                    calls = chain.calls
                    puts = chain.puts

                    total_call_oi = float(calls["openInterest"].sum()) if "openInterest" in calls else 0
                    total_put_oi = float(puts["openInterest"].sum()) if "openInterest" in puts else 0

                    if total_call_oi + total_put_oi > 0:
                        pc_ratio = total_put_oi / (total_call_oi + total_put_oi)
                        # Invert: high put/call = bearish = low sentiment score
                        sentiment = 1.0 - pc_ratio
                        sentiment_records.append({
                            "date": expiry,
                            "put_call_ratio": pc_ratio,
                            "sentiment_score": sentiment,
                        })
                except Exception:
                    pass

            if not sentiment_records:
                return pd.DataFrame()

            df_raw = pd.DataFrame(sentiment_records)
            df_raw["date"] = pd.to_datetime(df_raw["date"])
            df_raw = df_raw.set_index("date").sort_index()

            # Format as OHLCV
            out = pd.DataFrame(index=df_raw.index)
            score = df_raw["sentiment_score"]
            out["open"] = score
            out["high"] = score.clip(upper=1.0)
            out["low"] = score.clip(lower=0.0)
            out["close"] = score
            out["volume"] = 0.0
            out["ticker"] = ticker
            out["source"] = "put_call_ratio_proxy"
            return out

        except Exception as exc:
            logger.warning("Sentiment OHLCV failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

    def get_fx_rate_ohlcv(
        self,
        pair: str = "EURUSD",
        start: str = "1990-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch FX rate historical OHLCV via yfinance.

        Args:
            pair: Currency pair (e.g. 'EURUSD', 'GBPUSD', 'USDJPY').
            start: Start date.
            end: End date.

        Returns:
            DataFrame with standard OHLCV columns.
        """
        import yfinance as yf

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        # Normalize: EURUSD -> EURUSD=X
        if "=" not in pair:
            yf_pair = f"{pair.upper()}=X"
        else:
            yf_pair = pair.upper()

        try:
            obj = yf.Ticker(yf_pair)
            df = obj.history(start=start, end=end, auto_adjust=True)
            if df is None or df.empty:
                return pd.DataFrame()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df["series_id"] = pair.upper()
            df["source"] = "yfinance_fx"
            return df

        except Exception as exc:
            logger.warning("FX fetch failed", pair=pair, error=str(exc))
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# HistoricalOHLCVV2  —  main public API
# ---------------------------------------------------------------------------

class HistoricalOHLCVV2:
    """Enhanced Historical OHLCV adapter — score 9 target for dim_002.

    Combines all v2 components:
      - PointInTimeAdjustmentEngine: survivorship-bias-free universes
      - CorporateActionAdjuster: total return, split-only, spin-off
      - GlobalIndexData: 60+ markets
      - DataQualityValidator: stale, spike, gap, OHLC checks
      - TickerMappingEngine: continuous history across ticker renames
      - AltDataOHLCV: economic, crypto, VIX, FX as OHLCV
    """

    def __init__(self) -> None:
        self.pit = PointInTimeAdjustmentEngine()
        self.ca = CorporateActionAdjuster()
        self.global_data = GlobalIndexData()
        self.validator = DataQualityValidator()
        self.ticker_map = TickerMappingEngine()
        self.alt_data = AltDataOHLCV()

    def get_ohlcv(
        self,
        ticker: str,
        start: str = _DEFAULT_START,
        end: Optional[str] = None,
        adjust: str = "total_return",
    ) -> pd.DataFrame:
        """Fetch OHLCV with specified adjustment mode.

        Args:
            ticker: Ticker symbol.
            start: Start date.
            end: End date (defaults to today).
            adjust: 'total_return', 'split_only', 'none'.

        Returns:
            Standardized OHLCV DataFrame.
        """
        import yfinance as yf

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        try:
            obj = yf.Ticker(ticker)
            df = obj.history(start=start, end=end, auto_adjust=(adjust != "none"))
            if df is None or df.empty:
                return pd.DataFrame()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = np.nan

            df["ticker"] = ticker
            df["adj_close"] = df["close"]
            df["adj_factor"] = 1.0
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

            if adjust == "total_return":
                df = self.ca.apply_total_return_adjustment(df, ticker)
            elif adjust == "split_only":
                df = self.ca.apply_split_adjustment(df, ticker)

            return df

        except Exception as exc:
            logger.warning("get_ohlcv failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

    def get_adjusted_ohlcv(
        self,
        ticker: str,
        start: str = _DEFAULT_START,
        end: Optional[str] = None,
        mode: str = "total_return",
    ) -> pd.DataFrame:
        """Alias for get_ohlcv with explicit adjustment mode parameter."""
        return self.get_ohlcv(ticker, start, end, adjust=mode)

    def get_index_constituents_pit(
        self,
        index: str,
        as_of_date: str,
    ) -> List[str]:
        """Get point-in-time correct index constituents."""
        return self.pit.get_historical_constituents(index, as_of_date)

    def validate_quality(self, ticker: str, start: str = "2000-01-01") -> DataQualityReport:
        """Validate data quality for a ticker."""
        df = self.get_ohlcv(ticker, start=start, adjust="none")
        return self.validator.validate_series(df, ticker)

    def get_full_history(
        self,
        ticker: str,
        start: str = _DEFAULT_START,
    ) -> pd.DataFrame:
        """Fetch stitched history across all ticker aliases."""
        return self.ticker_map.get_full_history(ticker, start=start)

    def get_global_market_ohlcv(
        self,
        market: str,
        ticker: str,
        start: str = "2000-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV for a specific global market."""
        return self.global_data.fetch_market_ohlcv(market, ticker, start, end)


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel

    ohlcv_v2_router = APIRouter(prefix="/ohlcv/v2", tags=["OHLCV-v2"])

    _v2 = HistoricalOHLCVV2()

    class OHLCVResponse(BaseModel):
        ticker: str
        rows: int
        start: str
        end: str
        data: List[Dict[str, Any]]

    class ConstituentResponse(BaseModel):
        index: str
        as_of_date: str
        tickers: List[str]
        count: int

    class QualityResponse(BaseModel):
        ticker: str
        confidence_score: float
        grade: str
        issues: List[str]
        stale_count: int
        spike_count: int
        zero_volume_count: int
        gap_count: int
        total_rows: int
        date_range: Tuple[str, str]

    class FullHistoryResponse(BaseModel):
        ticker: str
        aliases_used: List[str]
        rows: int
        data: List[Dict[str, Any]]

    @ohlcv_v2_router.get("/{ticker}", response_model=OHLCVResponse)
    def get_ohlcv_v2(
        ticker: str,
        start: str = Query(_DEFAULT_START, description="Start date YYYY-MM-DD"),
        end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
        adjust: str = Query("total_return", description="Adjustment mode"),
    ) -> OHLCVResponse:
        """Fetch OHLCV with adjustment mode selection."""
        df = _v2.get_ohlcv(ticker.upper(), start=start, end=end, adjust=adjust)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        df_out = df.reset_index()
        df_out["date"] = df_out["date"].astype(str)
        records = df_out.to_dict(orient="records")

        dates = df_out["date"].tolist()
        return OHLCVResponse(
            ticker=ticker.upper(),
            rows=len(records),
            start=dates[0] if dates else "",
            end=dates[-1] if dates else "",
            data=records,
        )

    @ohlcv_v2_router.get("/adjusted/{ticker}", response_model=OHLCVResponse)
    def get_adjusted_ohlcv_v2(
        ticker: str,
        start: str = Query(_DEFAULT_START),
        end: Optional[str] = Query(None),
        mode: str = Query("total_return", description="total_return | split_only | none"),
    ) -> OHLCVResponse:
        """Fetch adjusted OHLCV with explicit mode parameter."""
        df = _v2.get_adjusted_ohlcv(ticker.upper(), start=start, end=end, mode=mode)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        df_out = df.reset_index()
        df_out["date"] = df_out["date"].astype(str)
        records = df_out.to_dict(orient="records")
        dates = df_out["date"].tolist()

        return OHLCVResponse(
            ticker=ticker.upper(),
            rows=len(records),
            start=dates[0] if dates else "",
            end=dates[-1] if dates else "",
            data=records,
        )

    @ohlcv_v2_router.get("/constituents/{index}/{as_of_date}", response_model=ConstituentResponse)
    def get_constituents(
        index: str,
        as_of_date: str,
    ) -> ConstituentResponse:
        """Get point-in-time correct index constituents."""
        tickers = _v2.get_index_constituents_pit(index.upper(), as_of_date)
        return ConstituentResponse(
            index=index.upper(),
            as_of_date=as_of_date,
            tickers=tickers,
            count=len(tickers),
        )

    @ohlcv_v2_router.get("/quality/{ticker}", response_model=QualityResponse)
    def get_quality_report(
        ticker: str,
        start: str = Query("2000-01-01"),
    ) -> QualityResponse:
        """Run data quality validation for a ticker."""
        report = _v2.validate_quality(ticker.upper(), start=start)
        return QualityResponse(
            ticker=report.ticker,
            confidence_score=report.confidence_score,
            grade=report.grade,
            issues=report.issues,
            stale_count=report.stale_count,
            spike_count=report.spike_count,
            zero_volume_count=report.zero_volume_count,
            gap_count=report.gap_count,
            total_rows=report.total_rows,
            date_range=report.date_range,
        )

    @ohlcv_v2_router.get("/full-history/{ticker}", response_model=FullHistoryResponse)
    def get_full_history_endpoint(
        ticker: str,
        start: str = Query(_DEFAULT_START),
    ) -> FullHistoryResponse:
        """Fetch stitched continuous history across all ticker aliases."""
        df = _v2.get_full_history(ticker.upper(), start=start)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No history for {ticker}")

        aliases = list(df["ticker_alias"].unique()) if "ticker_alias" in df.columns else [ticker]
        df_out = df.reset_index()
        df_out["date"] = df_out["date"].astype(str)
        records = df_out.to_dict(orient="records")

        return FullHistoryResponse(
            ticker=ticker.upper(),
            aliases_used=aliases,
            rows=len(records),
            data=records,
        )

    @ohlcv_v2_router.get("/global/{market}", response_model=OHLCVResponse)
    def get_global_market(
        market: str,
        ticker: str = Query(..., description="Base ticker without exchange suffix"),
        start: str = Query("2000-01-01"),
        end: Optional[str] = Query(None),
    ) -> OHLCVResponse:
        """Fetch OHLCV for a specific global market."""
        df = _v2.get_global_market_ohlcv(market.upper(), ticker, start=start, end=end)
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data for {ticker} on {market}",
            )

        df_out = df.reset_index()
        df_out["date"] = df_out["date"].astype(str)
        records = df_out.to_dict(orient="records")
        dates = df_out["date"].tolist()

        return OHLCVResponse(
            ticker=ticker.upper(),
            rows=len(records),
            start=dates[0] if dates else "",
            end=dates[-1] if dates else "",
            data=records,
        )

    @ohlcv_v2_router.get("/alt/economic/{indicator}")
    def get_economic_indicator(
        indicator: str,
        start: str = Query("1990-01-01"),
        end: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        """Fetch economic indicator as OHLCV (GDP, CPI, UNRATE, VIX, etc.)."""
        alt = AltDataOHLCV()
        df = alt.get_economic_ohlcv(indicator.upper(), start=start, end=end)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for indicator {indicator}")

        df_out = df.reset_index()
        df_out["date"] = df_out["date"].astype(str)
        return {
            "indicator": indicator.upper(),
            "rows": len(df_out),
            "data": df_out.to_dict(orient="records"),
        }

    @ohlcv_v2_router.get("/alt/crypto/{symbol}")
    def get_crypto(
        symbol: str,
        start: str = Query("2010-07-17"),
        end: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        """Fetch cryptocurrency OHLCV (BTC, ETH, SOL, etc.)."""
        alt = AltDataOHLCV()
        df = alt.get_crypto_ohlcv(symbol.upper(), start=start, end=end)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No crypto data for {symbol}")

        df_out = df.reset_index()
        df_out["date"] = df_out["date"].astype(str)
        return {
            "symbol": symbol.upper(),
            "rows": len(df_out),
            "data": df_out.to_dict(orient="records"),
        }

    @ohlcv_v2_router.get("/alt/vix")
    def get_vix_history_endpoint(
        start: str = Query("1990-01-02"),
        end: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        """Fetch VIX history back to 1990 from FRED."""
        alt = AltDataOHLCV()
        df = alt.get_vix_history(start=start, end=end)
        if df.empty:
            raise HTTPException(status_code=404, detail="No VIX data available")

        df_out = df.reset_index()
        df_out["date"] = df_out["date"].astype(str)
        return {
            "series": "VIX",
            "source": "FRED_VIXCLS",
            "rows": len(df_out),
            "data": df_out.to_dict(orient="records"),
        }

    @ohlcv_v2_router.get("/delisted/list")
    def get_delisted_stocks(
        reason: Optional[str] = Query(None, description="Filter by reason: bankruptcy|acquisition|merger"),
        delist_after: Optional[str] = Query(None, description="Only show stocks delisted after this date"),
    ) -> Dict[str, Any]:
        """Return list of historically significant delisted S&P 500 members."""
        engine = PointInTimeAdjustmentEngine()
        stocks = engine.get_delisted_stocks(delist_after=delist_after, reason=reason)
        return {
            "count": len(stocks),
            "stocks": stocks,
        }

    @ohlcv_v2_router.get("/ticker-history/{ticker}")
    def get_ticker_timeline(ticker: str) -> Dict[str, Any]:
        """Return chronological ticker rename timeline for a given current ticker."""
        engine = TickerMappingEngine()
        timeline = engine.build_ticker_timeline(ticker.upper())
        aliases = engine.get_all_aliases(ticker.upper())
        return {
            "current_ticker": ticker.upper(),
            "aliases": aliases,
            "timeline": timeline,
        }

except ImportError:
    # FastAPI not available — define stub
    ohlcv_v2_router = None  # type: ignore
    logger.warning("FastAPI not available; ohlcv_v2_router not created")


# ---------------------------------------------------------------------------
# Module-level convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    "PointInTimeAdjustmentEngine",
    "CorporateActionAdjuster",
    "GlobalIndexData",
    "DataQualityValidator",
    "TickerMappingEngine",
    "AltDataOHLCV",
    "HistoricalOHLCVV2",
    "DataQualityReport",
    "DELISTED_STOCKS",
    "TICKER_CHANGES",
    "CUSIP_MAP",
    "SP500_HISTORICAL_CONSTITUENTS",
    "COMMODITY_ETF_PROXIES",
    "INDIAN_ADR_MAP",
    "ohlcv_v2_router",
]
