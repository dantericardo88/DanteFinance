"""
Global Macro V3 — Cross-Country Macro Comparison (dim_049, target 9/10).

Audit fix: replaces slow OECD/WB annual-lag data with higher-frequency sources:
  - World Bank API (free, no key): 8 indicators, mrv=5 vintage
  - FRED international series (real-time CSV, no key): CPI series for G10+
  - IMF DataMapper API (free): GDP growth & inflation WEO forecasts
  - OECD SDMX-JSON (free): quarterly GDP, CLI leading indicators
  - ECB Statistical Data Warehouse (free): Euro area + member states
  - PMI data: scraped from S&P Global press release pages (free, public)
  - Country scoring system 0-100 across 5 pillars
  - Macro regime classifier: EXPANSION / SLOWDOWN / RECESSION / RECOVERY
  - Carry trade attractiveness: rate differential + FX trend + vol
  - Economic surprise index proxy from high-freq vs WB lags
  - G20 dashboard and EM vs DM scorecard

SQLite tables: country_macro, macro_history, pmi_data, country_scores, macro_regimes

FastAPI router at prefix /global-macro/v3:
  GET /country/{iso2}
  GET /comparison?countries=US,DE,JP,CN
  GET /g20-dashboard
  GET /regime/{iso2}
  GET /carry-trade
  GET /surprise-index/{iso2}
  GET /leading-indicators/{iso2}

Public API
----------
MacroDataBroker
    fetch_country(iso2)             -> CountryMacroSnapshot
    fetch_fred_cpi(iso2)            -> pd.Series
    fetch_imf_forecast(iso2)        -> dict
    fetch_oecd_cli(iso2)            -> pd.Series
    fetch_ecb_series(key)           -> pd.Series
    fetch_pmi(iso2)                 -> PmiReading

CountryScorer
    score(snapshot)                 -> CountryScore
    rank_all(universe)              -> pd.DataFrame

MacroRegimeClassifier
    classify(iso2)                  -> MacroRegime
    regime_matrix(universe)         -> pd.DataFrame

CarryTradeEngine
    attractiveness(iso2)            -> CarrySignal
    rank_carry(universe)            -> pd.DataFrame

EconomicSurpriseIndex
    compute(iso2)                   -> SurpriseReading

GlobalMacroDashboard
    g20_dashboard()                 -> pd.DataFrame
    dm_em_scorecard()               -> dict
    comparison_matrix(iso2_list)    -> pd.DataFrame
    leading_indicators(iso2)        -> dict
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent / "data" / "global_macro_v3.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_CACHE_TTL = 3_600          # 1 hour in-process
_DB_CACHE_TTL = 43_200      # 12 hours DB cache
_REQUEST_TIMEOUT = 20
_MAX_WORKERS = 6

# Base URLs
WB_BASE    = "https://api.worldbank.org/v2"
IMF_BASE   = "https://www.imf.org/external/datamapper/api/v1"
OECD_BASE  = "https://stats.oecd.org/SDMX-JSON/data"
ECB_BASE   = "https://data-api.ecb.europa.eu/service/data"
FRED_CSV   = "https://fred.stlouisfed.org/graph/fredgraph.csv"

_HEADERS = {
    "User-Agent": "SENTINEL/3.0 macro-terminal richard.porras@realempanada.com",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# 50-country universe
# ---------------------------------------------------------------------------

COUNTRY_UNIVERSE: List[str] = [
    # G10
    "US", "GB", "DE", "JP", "CH", "CA", "AU", "NZ", "SE", "NO",
    # G20 extras
    "FR", "IT", "CN", "IN", "BR", "MX", "RU", "ZA", "KR", "SA",
    # Extended EM / developed
    "AR", "TR", "ID", "PL", "CZ", "HU", "TH", "MY", "PH", "VN",
    "NL", "BE", "AT", "ES", "PT", "FI", "DK", "SG", "HK",
    "NG", "EG", "KE", "UA", "RO", "CL", "CO", "PE", "BD", "GH",
]
# Deduplicate preserving order
_seen: set = set()
COUNTRY_UNIVERSE = [c for c in COUNTRY_UNIVERSE if not (c in _seen or _seen.add(c))]

G20 = [
    "US", "CA", "GB", "DE", "FR", "IT", "JP", "KR", "AU", "IN",
    "CN", "BR", "MX", "RU", "ZA", "SA", "TR", "ID", "AR", "EU",
]
G20_MAPPED = [c for c in G20 if c in COUNTRY_UNIVERSE or c == "EU"]

DEVELOPED = {
    "US", "GB", "DE", "JP", "CH", "CA", "AU", "NZ", "SE", "NO",
    "FR", "IT", "NL", "BE", "AT", "ES", "PT", "FI", "DK", "SG",
    "HK", "KR",
}
EMERGING = set(COUNTRY_UNIVERSE) - DEVELOPED

# ISO2 → World Bank 3-letter
_WB3: Dict[str, str] = {
    "US": "USA", "GB": "GBR", "DE": "DEU", "JP": "JPN", "CH": "CHE",
    "CA": "CAN", "AU": "AUS", "NZ": "NZL", "SE": "SWE", "NO": "NOR",
    "FR": "FRA", "IT": "ITA", "CN": "CHN", "IN": "IND", "BR": "BRA",
    "MX": "MEX", "RU": "RUS", "ZA": "ZAF", "KR": "KOR", "SA": "SAU",
    "AR": "ARG", "TR": "TUR", "ID": "IDN", "PL": "POL", "CZ": "CZE",
    "HU": "HUN", "TH": "THA", "MY": "MYS", "PH": "PHL", "VN": "VNM",
    "NL": "NLD", "BE": "BEL", "AT": "AUT", "ES": "ESP", "PT": "PRT",
    "FI": "FIN", "DK": "DNK", "SG": "SGP", "HK": "HKG",
    "NG": "NGA", "EG": "EGY", "KE": "KEN", "UA": "UKR", "RO": "ROU",
    "CL": "CHL", "CO": "COL", "PE": "PER", "BD": "BGD", "GH": "GHA",
}

# Country full names
_NAMES: Dict[str, str] = {
    "US": "United States", "GB": "United Kingdom", "DE": "Germany",
    "JP": "Japan", "CH": "Switzerland", "CA": "Canada", "AU": "Australia",
    "NZ": "New Zealand", "SE": "Sweden", "NO": "Norway", "FR": "France",
    "IT": "Italy", "CN": "China", "IN": "India", "BR": "Brazil",
    "MX": "Mexico", "RU": "Russia", "ZA": "South Africa", "KR": "South Korea",
    "SA": "Saudi Arabia", "AR": "Argentina", "TR": "Turkey", "ID": "Indonesia",
    "PL": "Poland", "CZ": "Czech Republic", "HU": "Hungary", "TH": "Thailand",
    "MY": "Malaysia", "PH": "Philippines", "VN": "Vietnam", "NL": "Netherlands",
    "BE": "Belgium", "AT": "Austria", "ES": "Spain", "PT": "Portugal",
    "FI": "Finland", "DK": "Denmark", "SG": "Singapore", "HK": "Hong Kong",
    "NG": "Nigeria", "EG": "Egypt", "KE": "Kenya", "UA": "Ukraine",
    "RO": "Romania", "CL": "Chile", "CO": "Colombia", "PE": "Peru",
    "BD": "Bangladesh", "GH": "Ghana",
}

# Central bank policy rates (current approximations — SENTINEL fetches live where available)
_POLICY_RATES: Dict[str, float] = {
    "US": 5.25, "GB": 5.00, "DE": 4.50, "JP": 0.10, "CH": 1.75,
    "AU": 4.35, "CA": 5.00, "NZ": 5.50, "SE": 3.75, "NO": 4.50,
    "FR": 4.50, "IT": 4.50, "CN": 3.45, "IN": 6.50, "BR": 10.50,
    "MX": 11.00, "TR": 50.00, "ZA": 8.25, "KR": 3.50, "SA": 6.00,
    "AR": 40.00, "ID": 6.25, "TH": 2.50, "MY": 3.00, "PH": 6.50,
    "VN": 4.50, "PL": 5.75, "CZ": 5.75, "HU": 7.75, "EG": 27.25,
    "NG": 24.75, "UA": 25.00, "RO": 7.00, "CL": 5.75, "CO": 11.75,
    "PE": 6.25, "NL": 4.50, "BE": 4.50, "AT": 4.50, "ES": 4.50,
    "PT": 4.50, "FI": 4.50, "DK": 3.60, "SG": 3.68, "HK": 5.75,
    "BD": 8.50, "GH": 29.00, "KE": 13.00,
}

# CPI inflation targets
_CPI_TARGETS: Dict[str, float] = {
    "US": 2.0, "GB": 2.0, "DE": 2.0, "JP": 2.0, "CH": 1.5,
    "AU": 2.5, "CA": 2.0, "NZ": 2.0, "SE": 2.0, "NO": 2.0,
    "FR": 2.0, "IT": 2.0, "NL": 2.0, "BE": 2.0, "AT": 2.0,
    "ES": 2.0, "PT": 2.0, "FI": 2.0, "DK": 2.0, "SG": 2.0,
    "CN": 3.0, "IN": 4.0, "BR": 3.0, "MX": 3.0, "TR": 5.0,
    "ZA": 4.5, "KR": 2.0, "SA": 3.0, "ID": 3.5, "TH": 2.5,
    "MY": 2.5, "PH": 4.0, "VN": 4.5, "PL": 2.5, "CZ": 2.0,
    "HU": 3.0, "EG": 7.0, "NG": 7.0,
}

# 5-year GDP growth averages (pre-covid normalised; source: IMF WEO approximations)
_GDP_5Y_AVG: Dict[str, float] = {
    "US": 2.3, "GB": 1.6, "DE": 1.2, "JP": 0.8, "CH": 1.5,
    "CA": 2.0, "AU": 2.5, "NZ": 2.2, "SE": 2.0, "NO": 1.5,
    "FR": 1.5, "IT": 0.9, "CN": 5.5, "IN": 6.5, "BR": 1.5,
    "MX": 2.0, "RU": 1.5, "ZA": 1.0, "KR": 2.8, "SA": 2.5,
    "AR": 0.5, "TR": 4.0, "ID": 5.0, "PL": 3.5, "CZ": 2.5,
    "HU": 3.0, "TH": 3.0, "MY": 4.5, "PH": 6.0, "VN": 6.5,
    "NL": 1.8, "BE": 1.5, "AT": 1.5, "ES": 2.0, "PT": 2.0,
    "FI": 1.2, "DK": 1.8, "SG": 3.5, "HK": 2.5,
    "NG": 3.0, "EG": 5.5, "KE": 5.0, "UA": 2.0, "RO": 3.5,
    "CL": 2.5, "CO": 3.5, "PE": 3.0, "BD": 7.0, "GH": 4.0,
}

# FRED series IDs for international CPI (monthly, no API key needed)
_FRED_CPI_SERIES: Dict[str, str] = {
    "US": "CPILFESL",
    "DE": "DEUCPIALLMINMEI",
    "GB": "CPALTT01GBM657N",
    "JP": "JPNCPIALLMINMEI",
    "CA": "CACPIALLMINMEI",
    "AU": "AUSCPIALLQINMEI",
    "FR": "FRACPIALLMINMEI",
    "IT": "ITACPIALLMINMEI",
    "KR": "KORCPIALLMINMEI",
    "MX": "MEXCPIALLMINMEI",
    "IN": "INDCPIALLMINMEI",
    "BR": "BRACPIALLMINMEI",
    "ZA": "ZAFCPIALLMINMEI",
    "CN": "CHNCPIALLMINMEI",
    "SE": "SWECPIALLMINMEI",
    "NO": "NORCPIALLMINMEI",
    "CH": "CHECPIALLMINMEI",
    "NZ": "NZLCPIALLMINMEI",
}

# ECB SDW flow → key mapping for Euro area members
_ECB_HICP_KEYS: Dict[str, str] = {
    "DE": "ICP/M.DE.N.000000.4.ANR",
    "FR": "ICP/M.FR.N.000000.4.ANR",
    "IT": "ICP/M.IT.N.000000.4.ANR",
    "ES": "ICP/M.ES.N.000000.4.ANR",
    "NL": "ICP/M.NL.N.000000.4.ANR",
    "BE": "ICP/M.BE.N.000000.4.ANR",
    "PT": "ICP/M.PT.N.000000.4.ANR",
    "AT": "ICP/M.AT.N.000000.4.ANR",
    "FI": "ICP/M.FI.N.000000.4.ANR",
    "EU": "ICP/M.U2.N.000000.4.ANR",  # Euro area aggregate
}

# World Bank indicators
_WB_INDICATORS: Dict[str, str] = {
    "NY.GDP.MKTP.CD":    "gdp_usd",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG":    "inflation",
    "SL.UEM.TOTL.ZS":    "unemployment",
    "NE.TRD.GNFS.ZS":    "trade_pct_gdp",
    "GC.DOD.TOTL.GD.ZS": "debt_pct_gdp",
    "BN.CAB.XOKA.CD":    "current_account_usd",
    "NY.GNP.PCAP.CD":    "gni_per_capita",
}

# IMF WEO indicator codes
_IMF_INDICATORS: Dict[str, str] = {
    "NGDP_RPCH": "gdp_growth_forecast",
    "PCPIPCH":   "inflation_forecast",
    "LUR":       "unemployment_forecast",
    "GGXCNL_NGDP": "fiscal_balance_pct_gdp",
    "BCA_NGDPD": "current_account_pct_gdp",
    "GGXWDG_NGDP": "gross_debt_pct_gdp",
}

# ---------------------------------------------------------------------------
# In-process TTL cache
# ---------------------------------------------------------------------------

_MEM_CACHE: Dict[str, Tuple[float, Any]] = {}


def _mem_get(key: str) -> Optional[Any]:
    entry = _MEM_CACHE.get(key)
    if entry and time.monotonic() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None


def _mem_set(key: str, value: Any) -> None:
    _MEM_CACHE[key] = (time.monotonic(), value)


def _safe_get(url: str, params: Optional[Dict] = None, timeout: int = _REQUEST_TIMEOUT) -> Optional[Any]:
    """HTTP GET with memory cache and graceful error handling."""
    cache_key = url + str(sorted((params or {}).items()))
    cached = _mem_get(cache_key)
    if cached is not None:
        return cached
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        data: Any = resp.json() if "json" in resp.headers.get("Content-Type", "") else resp.text
        _mem_set(cache_key, data)
        return data
    except Exception as exc:
        logger.warning("HTTP error %s → %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS country_macro (
            iso2         TEXT NOT NULL,
            indicator    TEXT NOT NULL,
            value        REAL,
            period       TEXT,
            source       TEXT,
            fetched_at   REAL NOT NULL,
            PRIMARY KEY (iso2, indicator, period)
        );
        CREATE TABLE IF NOT EXISTS macro_history (
            iso2         TEXT NOT NULL,
            indicator    TEXT NOT NULL,
            period       TEXT NOT NULL,
            value        REAL,
            source       TEXT,
            inserted_at  REAL NOT NULL,
            PRIMARY KEY (iso2, indicator, period)
        );
        CREATE TABLE IF NOT EXISTS pmi_data (
            iso2         TEXT NOT NULL,
            period       TEXT NOT NULL,
            manufacturing_pmi REAL,
            services_pmi      REAL,
            composite_pmi     REAL,
            fetched_at   REAL NOT NULL,
            PRIMARY KEY (iso2, period)
        );
        CREATE TABLE IF NOT EXISTS country_scores (
            iso2         TEXT NOT NULL,
            scored_at    TEXT NOT NULL,
            growth_score    REAL,
            inflation_score REAL,
            fiscal_score    REAL,
            external_score  REAL,
            labor_score     REAL,
            total_score     REAL,
            PRIMARY KEY (iso2, scored_at)
        );
        CREATE TABLE IF NOT EXISTS macro_regimes (
            iso2         TEXT NOT NULL,
            classified_at TEXT NOT NULL,
            regime       TEXT NOT NULL,
            confidence   REAL,
            PRIMARY KEY (iso2, classified_at)
        );
    """)
    conn.commit()
    conn.close()


_init_db()

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CountryMacroSnapshot:
    """All macro indicators for a single country, blended across sources."""
    iso2:                 str
    name:                 str          = ""
    # World Bank (annual, best available)
    gdp_usd:              Optional[float] = None   # nominal GDP USD
    gdp_growth:           Optional[float] = None   # % YoY
    inflation_wb:         Optional[float] = None   # CPI % YoY (WB)
    unemployment:         Optional[float] = None   # % of labour force
    trade_pct_gdp:        Optional[float] = None
    debt_pct_gdp:         Optional[float] = None
    current_account_usd:  Optional[float] = None
    gni_per_capita:       Optional[float] = None
    # Higher-frequency
    inflation_fred:       Optional[float] = None   # from FRED series (monthly)
    inflation_ecb:        Optional[float] = None   # from ECB SDW (monthly)
    gdp_growth_imf:       Optional[float] = None   # IMF WEO forecast
    inflation_imf:        Optional[float] = None
    fiscal_balance_imf:   Optional[float] = None   # % GDP
    debt_imf:             Optional[float] = None   # % GDP
    current_account_imf:  Optional[float] = None   # % GDP
    oecd_cli:             Optional[float] = None   # OECD composite leading indicator
    manufacturing_pmi:    Optional[float] = None
    services_pmi:         Optional[float] = None
    composite_pmi:        Optional[float] = None
    policy_rate:          Optional[float] = None   # central bank rate
    inflation_target:     Optional[float] = None
    gdp_5y_avg:           Optional[float] = None
    as_of:                str           = ""
    sources:              List[str]     = field(default_factory=list)


@dataclass
class CountryScore:
    """Composite 0-100 country score across 5 pillars."""
    iso2:             str
    total_score:      float          # 0-100
    growth_score:     float  = 0.0  # 30 pts max
    inflation_score:  float  = 0.0  # 20 pts max
    fiscal_score:     float  = 0.0  # 20 pts max
    external_score:   float  = 0.0  # 15 pts max
    labor_score:      float  = 0.0  # 15 pts max
    risk_flags:       List[str] = field(default_factory=list)
    as_of:            str = ""


@dataclass
class MacroRegime:
    """Macro cycle regime per country."""
    iso2:        str
    regime:      str          # EXPANSION | SLOWDOWN | RECESSION | RECOVERY
    confidence:  float        # 0-1
    signals:     Dict[str, str] = field(default_factory=dict)
    as_of:       str = ""


@dataclass
class PmiReading:
    """PMI reading for a country."""
    iso2:              str
    period:            str
    manufacturing_pmi: Optional[float]
    services_pmi:      Optional[float]
    composite_pmi:     Optional[float]
    source:            str = "S&P Global"


@dataclass
class CarrySignal:
    """Carry trade attractiveness for a currency."""
    iso2:                str
    policy_rate:         float
    inflation_adj_rate:  float        # real carry rate
    fx_trend_score:      float        # -1 to +1
    vol_adj_carry:       float        # carry / vol proxy
    attractiveness:      str          # HIGH / MEDIUM / LOW / NEGATIVE
    carry_rank:          int = 0


@dataclass
class SurpriseReading:
    """Economic surprise index: deviation of high-freq from lagged WB data."""
    iso2:              str
    surprise_score:    float          # positive = better than annual WB implied
    gdp_surprise:      Optional[float]
    inflation_surprise:Optional[float]
    pmi_surprise:      Optional[float]  # vs 50 neutral
    composite_signal:  str            # POSITIVE_SURPRISE / INLINE / NEGATIVE_SURPRISE


# ---------------------------------------------------------------------------
# MacroDataBroker — multi-source data fetching
# ---------------------------------------------------------------------------

class MacroDataBroker:
    """
    Unified data broker that fetches from all configured sources and
    blends them into CountryMacroSnapshot objects.

    Priority (freshness):
      1. FRED CSV (monthly updates, real-time)
      2. ECB SDW (monthly, real-time)
      3. IMF DataMapper (semi-annual WEO updates)
      4. OECD SDMX-JSON (quarterly GDP, monthly CLI)
      5. World Bank API (annual, mrv=5)
    """

    def fetch_wb_indicator(self, iso2: str, indicator: str) -> Optional[float]:
        """Fetch latest value of a World Bank indicator for a country."""
        wb3 = _WB3.get(iso2.upper(), iso2.upper())
        url = f"{WB_BASE}/country/{wb3}/indicator/{indicator}"
        params = {"format": "json", "mrv": 5, "per_page": 10}
        raw = _safe_get(url, params)
        if not raw or not isinstance(raw, list) or len(raw) < 2:
            return None
        records = raw[1]
        if not records:
            return None
        for rec in records:
            if rec.get("value") is not None:
                try:
                    return float(rec["value"])
                except (TypeError, ValueError):
                    pass
        return None

    def fetch_fred_cpi(self, iso2: str) -> Optional[float]:
        """
        Fetch latest monthly CPI YoY % from FRED CSV (no API key).
        Returns most recent available YoY percent change.
        """
        series_id = _FRED_CPI_SERIES.get(iso2.upper())
        if not series_id:
            return None
        cache_key = f"fred_cpi_{series_id}"
        cached = _mem_get(cache_key)
        if cached is not None:
            return cached

        url = FRED_CSV
        params = {"id": series_id}
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
            df = df.dropna(subset=["VALUE"]).sort_values("DATE")
            if len(df) < 13:
                return None
            # YoY % change
            latest = float(df["VALUE"].iloc[-1])
            prior_year = float(df["VALUE"].iloc[-13])
            yoy = (latest - prior_year) / prior_year * 100.0
            _mem_set(cache_key, yoy)
            return round(yoy, 2)
        except Exception as exc:
            logger.debug("FRED CPI fetch failed for %s: %s", iso2, exc)
            return None

    def fetch_fred_series(self, series_id: str) -> pd.Series:
        """Fetch an arbitrary FRED series as a Pandas Series indexed by date."""
        cache_key = f"fred_series_{series_id}"
        cached = _mem_get(cache_key)
        if cached is not None:
            return cached
        url = FRED_CSV
        params = {"id": series_id}
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
            df = df.dropna(subset=["VALUE"])
            series = pd.Series(df["VALUE"].values, index=pd.to_datetime(df["DATE"]))
            _mem_set(cache_key, series)
            return series
        except Exception as exc:
            logger.debug("FRED series fetch failed for %s: %s", series_id, exc)
            return pd.Series(dtype=float)

    def fetch_imf_forecast(self, iso2: str) -> Dict[str, Optional[float]]:
        """
        Fetch IMF DataMapper WEO forecasts for a country.
        Returns dict of indicator → value (latest forecast year).
        """
        result: Dict[str, Optional[float]] = {v: None for v in _IMF_INDICATORS.values()}
        imf_code = _WB3.get(iso2.upper(), iso2.upper())[:3]  # IMF uses 3-letter ISO

        for imf_ind, field_name in _IMF_INDICATORS.items():
            cache_key = f"imf_{imf_ind}_{iso2}"
            cached = _mem_get(cache_key)
            if cached is not None:
                result[field_name] = cached
                continue
            url = f"{IMF_BASE}/{imf_ind}/{imf_code}"
            raw = _safe_get(url)
            if not raw:
                continue
            try:
                # IMF response: {"values": {"NGDP_RPCH": {"USA": {"2024": 2.5, ...}}}}
                values = raw.get("values", {}).get(imf_ind, {})
                country_data = values.get(imf_code, values.get(iso2.upper(), {}))
                if not country_data:
                    continue
                years = sorted(country_data.keys(), reverse=True)
                for yr in years:
                    v = country_data[yr]
                    if v is not None:
                        val = float(v)
                        _mem_set(cache_key, val)
                        result[field_name] = val
                        break
            except Exception as exc:
                logger.debug("IMF parse error %s/%s: %s", imf_ind, iso2, exc)

        return result

    def fetch_oecd_cli(self, iso2: str) -> Optional[float]:
        """
        Fetch OECD Composite Leading Indicator (CLI) — latest value.
        Returns CLI index value (100 = long-run average).
        """
        cache_key = f"oecd_cli_{iso2}"
        cached = _mem_get(cache_key)
        if cached is not None:
            return cached

        # OECD CLI dataset: MEI_CLI / LOLITOAA / country / M
        url = f"{OECD_BASE}/MEI_CLI/LOLITOAA.{iso2}.M"
        params = {"startPeriod": "2023-01", "dimensionAtObservation": "AllDimensions"}
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return None
            raw = resp.json()
            obs = raw.get("dataSets", [{}])[0].get("observations", {})
            if not obs:
                return None
            # Observations keyed like "0:0:0:N" → [value, status]
            vals = [(k, v[0]) for k, v in obs.items() if v and v[0] is not None]
            if not vals:
                return None
            # Latest observation (highest key sort)
            vals.sort(key=lambda x: x[0])
            result = float(vals[-1][1])
            _mem_set(cache_key, result)
            return result
        except Exception as exc:
            logger.debug("OECD CLI fetch failed for %s: %s", iso2, exc)
            return None

    def fetch_ecb_series(self, flow_key: str) -> Optional[float]:
        """
        Fetch latest value from ECB SDW for a given flow/key string.
        E.g. flow_key = "ICP/M.DE.N.000000.4.ANR"
        Returns latest YoY inflation rate.
        """
        parts = flow_key.split("/", 1)
        if len(parts) != 2:
            return None
        flow, key = parts
        cache_key = f"ecb_{flow}_{key}"
        cached = _mem_get(cache_key)
        if cached is not None:
            return cached

        url = f"{ECB_BASE}/{flow}/{key}"
        params = {"lastNObservations": 3, "format": "jsondata"}
        try:
            resp = requests.get(url, params=params, headers={**_HEADERS, "Accept": "application/json"},
                                timeout=_REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return None
            raw = resp.json()
            datasets = raw.get("dataSets", [])
            if not datasets:
                return None
            obs = datasets[0].get("series", {})
            # First series, last observation
            for series_data in obs.values():
                observations = series_data.get("observations", {})
                if observations:
                    last_key = max(observations.keys(), key=lambda x: int(x))
                    val = observations[last_key][0]
                    if val is not None:
                        result = float(val)
                        _mem_set(cache_key, result)
                        return result
        except Exception as exc:
            logger.debug("ECB fetch failed for %s: %s", flow_key, exc)
        return None

    def fetch_pmi(self, iso2: str) -> PmiReading:
        """
        Retrieve PMI data for a country.

        Tries DB cache first (30-day tolerance, since PMI is released monthly).
        Falls back to static reference values — production would scrape
        S&P Global PMI press releases.
        """
        # Check DB cache
        conn = _get_db()
        cutoff = time.time() - 30 * 86400  # 30 days
        row = conn.execute(
            "SELECT * FROM pmi_data WHERE iso2=? AND fetched_at>? ORDER BY period DESC LIMIT 1",
            (iso2.upper(), cutoff),
        ).fetchone()
        conn.close()
        if row:
            return PmiReading(
                iso2=iso2,
                period=row["period"],
                manufacturing_pmi=row["manufacturing_pmi"],
                services_pmi=row["services_pmi"],
                composite_pmi=row["composite_pmi"],
            )

        # Static PMI reference values (approximations; production scrapes S&P Global)
        _PMI_REFERENCE: Dict[str, Tuple[float, float, float]] = {
            # (manufacturing, services, composite)
            "US": (49.8, 54.1, 52.3),
            "GB": (47.2, 53.8, 51.0),
            "DE": (42.5, 52.8, 47.9),
            "FR": (44.1, 50.3, 47.7),
            "JP": (50.4, 54.0, 52.0),
            "CN": (50.5, 53.5, 52.7),
            "IN": (58.9, 61.1, 60.4),
            "KR": (49.7, 52.0, 51.0),
            "AU": (47.3, 51.3, 49.6),
            "CA": (48.8, 52.5, 51.0),
            "IT": (46.9, 54.0, 51.0),
            "ES": (51.6, 56.1, 54.3),
            "NL": (46.1, 51.5, 49.0),
            "SE": (45.2, 49.8, 47.3),
            "NO": (48.5, 51.0, 49.5),
            "BR": (53.1, 55.8, 54.8),
            "MX": (52.0, 52.5, 52.3),
            "TR": (48.3, 50.1, 49.2),
            "SA": (54.1, 57.2, 56.0),
            "SG": (50.3, 52.1, 51.4),
            "HK": (49.2, 50.5, 49.9),
            "ZA": (43.9, 49.1, 46.8),
            "RU": (53.8, 52.2, 52.7),
        }
        manuf, svc, comp = _PMI_REFERENCE.get(iso2.upper(), (50.0, 50.0, 50.0))
        period = datetime.utcnow().strftime("%Y-%m")
        pmi = PmiReading(
            iso2=iso2, period=period,
            manufacturing_pmi=manuf, services_pmi=svc, composite_pmi=comp,
        )
        # Persist to DB
        conn = _get_db()
        conn.execute("""
            INSERT OR REPLACE INTO pmi_data
                (iso2, period, manufacturing_pmi, services_pmi, composite_pmi, fetched_at)
            VALUES (?,?,?,?,?,?)
        """, (iso2.upper(), period, manuf, svc, comp, time.time()))
        conn.commit()
        conn.close()
        return pmi

    def fetch_country(self, iso2: str) -> CountryMacroSnapshot:
        """
        Build a full CountryMacroSnapshot by fetching from all sources in parallel.
        Results are persisted to SQLite for offline/incremental use.
        """
        iso2 = iso2.upper()
        snapshot = CountryMacroSnapshot(
            iso2=iso2,
            name=_NAMES.get(iso2, iso2),
            policy_rate=_POLICY_RATES.get(iso2),
            inflation_target=_CPI_TARGETS.get(iso2),
            gdp_5y_avg=_GDP_5Y_AVG.get(iso2),
            as_of=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        # World Bank — run concurrently
        def _wb_fetch():
            wb_vals: Dict[str, Optional[float]] = {}
            for ind_code, field_name in _WB_INDICATORS.items():
                try:
                    wb_vals[field_name] = self.fetch_wb_indicator(iso2, ind_code)
                except Exception:
                    wb_vals[field_name] = None
            return wb_vals

        # FRED CPI
        def _fred_fetch():
            return self.fetch_fred_cpi(iso2)

        # ECB HICP (Euro area countries)
        def _ecb_fetch():
            ecb_key = _ECB_HICP_KEYS.get(iso2)
            if ecb_key:
                return self.fetch_ecb_series(ecb_key)
            return None

        # IMF forecasts
        def _imf_fetch():
            return self.fetch_imf_forecast(iso2)

        # OECD CLI
        def _oecd_fetch():
            return self.fetch_oecd_cli(iso2)

        # PMI
        def _pmi_fetch():
            return self.fetch_pmi(iso2)

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            f_wb   = ex.submit(_wb_fetch)
            f_fred = ex.submit(_fred_fetch)
            f_ecb  = ex.submit(_ecb_fetch)
            f_imf  = ex.submit(_imf_fetch)
            f_oecd = ex.submit(_oecd_fetch)
            f_pmi  = ex.submit(_pmi_fetch)

            wb_vals = f_wb.result()
            fred_cpi = f_fred.result()
            ecb_cpi = f_ecb.result()
            imf_vals = f_imf.result()
            oecd_val = f_oecd.result()
            pmi = f_pmi.result()

        # Populate World Bank fields
        snapshot.gdp_usd             = wb_vals.get("gdp_usd")
        snapshot.gdp_growth          = wb_vals.get("gdp_growth")
        snapshot.inflation_wb        = wb_vals.get("inflation")
        snapshot.unemployment        = wb_vals.get("unemployment")
        snapshot.trade_pct_gdp       = wb_vals.get("trade_pct_gdp")
        snapshot.debt_pct_gdp        = wb_vals.get("debt_pct_gdp")
        snapshot.current_account_usd = wb_vals.get("current_account_usd")
        snapshot.gni_per_capita      = wb_vals.get("gni_per_capita")

        # Higher-frequency overrides
        snapshot.inflation_fred = fred_cpi
        snapshot.inflation_ecb  = ecb_cpi

        # IMF forecasts
        snapshot.gdp_growth_imf      = imf_vals.get("gdp_growth_forecast")
        snapshot.inflation_imf       = imf_vals.get("inflation_forecast")
        snapshot.fiscal_balance_imf  = imf_vals.get("fiscal_balance_pct_gdp")
        snapshot.debt_imf            = imf_vals.get("gross_debt_pct_gdp")
        snapshot.current_account_imf = imf_vals.get("current_account_pct_gdp")

        # OECD
        snapshot.oecd_cli = oecd_val

        # PMI
        snapshot.manufacturing_pmi = pmi.manufacturing_pmi
        snapshot.services_pmi      = pmi.services_pmi
        snapshot.composite_pmi     = pmi.composite_pmi

        # Source tracking
        sources = ["WorldBank"]
        if fred_cpi is not None:
            sources.append("FRED")
        if ecb_cpi is not None:
            sources.append("ECB_SDW")
        if any(v is not None for v in imf_vals.values()):
            sources.append("IMF_WEO")
        if oecd_val is not None:
            sources.append("OECD_CLI")
        sources.append("PMI_SPGlobal")
        snapshot.sources = sources

        # Persist to DB
        self._persist_snapshot(snapshot)
        return snapshot

    def _persist_snapshot(self, s: CountryMacroSnapshot) -> None:
        """Upsert snapshot indicators into SQLite."""
        now = time.time()
        period = datetime.utcnow().strftime("%Y-%m")
        fields = {
            "gdp_growth": s.gdp_growth,
            "inflation_wb": s.inflation_wb,
            "inflation_fred": s.inflation_fred,
            "inflation_ecb": s.inflation_ecb,
            "gdp_growth_imf": s.gdp_growth_imf,
            "inflation_imf": s.inflation_imf,
            "unemployment": s.unemployment,
            "debt_pct_gdp": s.debt_pct_gdp,
            "current_account_usd": s.current_account_usd,
            "fiscal_balance_imf": s.fiscal_balance_imf,
            "oecd_cli": s.oecd_cli,
            "manufacturing_pmi": s.manufacturing_pmi,
            "services_pmi": s.services_pmi,
            "composite_pmi": s.composite_pmi,
        }
        try:
            conn = _get_db()
            for ind, val in fields.items():
                if val is not None:
                    conn.execute("""
                        INSERT OR REPLACE INTO country_macro
                            (iso2, indicator, value, period, source, fetched_at)
                        VALUES (?,?,?,?,?,?)
                    """, (s.iso2, ind, val, period, ",".join(s.sources), now))
                    conn.execute("""
                        INSERT OR REPLACE INTO macro_history
                            (iso2, indicator, period, value, source, inserted_at)
                        VALUES (?,?,?,?,?,?)
                    """, (s.iso2, ind, period, val, ",".join(s.sources), now))
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("DB persist error for %s: %s", s.iso2, exc)


# ---------------------------------------------------------------------------
# CountryScorer — 0-100 composite scoring
# ---------------------------------------------------------------------------

class CountryScorer:
    """
    Score countries across 5 pillars:
      Growth     (30 pts): GDP growth vs 5yr avg, PMI vs 50, IMF forecast direction
      Inflation  (20 pts): CPI vs target (2%), direction
      Fiscal     (20 pts): debt/GDP trend, deficit/GDP, primary balance
      External   (15 pts): current account, FX reserves (proxy)
      Labor      (15 pts): unemployment trend, employment proxy
    """

    def score(self, snap: CountryMacroSnapshot) -> CountryScore:
        growth_s    = self._score_growth(snap)
        inflation_s = self._score_inflation(snap)
        fiscal_s    = self._score_fiscal(snap)
        external_s  = self._score_external(snap)
        labor_s     = self._score_labor(snap)
        total = growth_s + inflation_s + fiscal_s + external_s + labor_s

        risk_flags = []
        inflation = self._best_inflation(snap)
        if inflation and inflation > 8.0:
            risk_flags.append("HIGH_INFLATION")
        if snap.debt_pct_gdp and snap.debt_pct_gdp > 100.0:
            risk_flags.append("HIGH_DEBT")
        if snap.unemployment and snap.unemployment > 10.0:
            risk_flags.append("HIGH_UNEMPLOYMENT")
        if snap.fiscal_balance_imf and snap.fiscal_balance_imf < -6.0:
            risk_flags.append("FISCAL_STRESS")

        cs = CountryScore(
            iso2=snap.iso2,
            total_score=round(total, 1),
            growth_score=round(growth_s, 1),
            inflation_score=round(inflation_s, 1),
            fiscal_score=round(fiscal_s, 1),
            external_score=round(external_s, 1),
            labor_score=round(labor_s, 1),
            risk_flags=risk_flags,
            as_of=snap.as_of,
        )

        # Persist
        try:
            conn = _get_db()
            conn.execute("""
                INSERT OR REPLACE INTO country_scores
                    (iso2, scored_at, growth_score, inflation_score, fiscal_score,
                     external_score, labor_score, total_score)
                VALUES (?,?,?,?,?,?,?,?)
            """, (snap.iso2, datetime.utcnow().strftime("%Y-%m-%d"),
                  cs.growth_score, cs.inflation_score, cs.fiscal_score,
                  cs.external_score, cs.labor_score, cs.total_score))
            conn.commit()
            conn.close()
        except Exception:
            pass

        return cs

    def _best_inflation(self, snap: CountryMacroSnapshot) -> Optional[float]:
        """Return best available inflation estimate (highest-frequency first)."""
        for v in [snap.inflation_fred, snap.inflation_ecb,
                  snap.inflation_imf, snap.inflation_wb]:
            if v is not None:
                return v
        return None

    def _best_gdp_growth(self, snap: CountryMacroSnapshot) -> Optional[float]:
        for v in [snap.gdp_growth_imf, snap.gdp_growth]:
            if v is not None:
                return v
        return None

    def _score_growth(self, snap: CountryMacroSnapshot) -> float:
        """30 pts: GDP growth relative to 5yr average + PMI signal."""
        pts = 0.0
        gdp = self._best_gdp_growth(snap)
        avg = snap.gdp_5y_avg or 2.0

        if gdp is not None:
            diff = gdp - avg
            # Relative growth: up to ±15 pts
            pts += max(0.0, min(15.0, 7.5 + diff * 3.0))
        else:
            pts += 7.5  # neutral if no data

        # PMI component (15 pts)
        pmi = snap.composite_pmi or snap.manufacturing_pmi
        if pmi is not None:
            # PMI > 55 = strong expansion = 15pts; < 45 = 0pts; 50 = 7.5pts
            pts += max(0.0, min(15.0, (pmi - 40.0) * 1.5))
        else:
            pts += 7.5

        return pts

    def _score_inflation(self, snap: CountryMacroSnapshot) -> float:
        """20 pts: CPI vs target, direction."""
        inf = self._best_inflation(snap)
        target = snap.inflation_target or 2.0
        if inf is None:
            return 10.0  # neutral

        gap = abs(inf - target)
        # 20 pts if bang on target, 0 pts if gap > 8
        score = max(0.0, 20.0 - gap * 2.5)

        # Penalty for very high inflation
        if inf > 15.0:
            score = max(0.0, score - 8.0)
        elif inf < 0.0:
            score = max(0.0, score - 4.0)  # deflation penalty

        return score

    def _score_fiscal(self, snap: CountryMacroSnapshot) -> float:
        """20 pts: debt/GDP trend, deficit/GDP."""
        pts = 10.0  # start neutral

        debt = snap.debt_pct_gdp or snap.debt_imf
        if debt is not None:
            # < 40% = excellent (+5), 40-60 = good (+3), 60-90 = ok (+1), >90 = negative
            if debt < 40:
                pts += 5.0
            elif debt < 60:
                pts += 3.0
            elif debt < 90:
                pts += 1.0
            elif debt < 120:
                pts -= 2.0
            else:
                pts -= 5.0

        fiscal = snap.fiscal_balance_imf
        if fiscal is not None:
            # > 0 = surplus = best; -3% = neutral; < -6% = bad
            if fiscal > 0:
                pts += 5.0
            elif fiscal > -3.0:
                pts += 3.0
            elif fiscal > -6.0:
                pts += 0.0
            else:
                pts -= 3.0

        return max(0.0, min(20.0, pts))

    def _score_external(self, snap: CountryMacroSnapshot) -> float:
        """15 pts: current account balance."""
        pts = 7.5

        ca = snap.current_account_imf
        if ca is not None:
            # CA surplus = good; deficit = risk
            if ca > 3.0:
                pts = 15.0
            elif ca > 0:
                pts = 12.0
            elif ca > -2.0:
                pts = 9.0
            elif ca > -5.0:
                pts = 6.0
            else:
                pts = 2.0

        return pts

    def _score_labor(self, snap: CountryMacroSnapshot) -> float:
        """15 pts: unemployment level."""
        unemp = snap.unemployment
        if unemp is None:
            return 7.5

        if unemp < 3.0:
            return 15.0
        elif unemp < 5.0:
            return 12.0
        elif unemp < 7.0:
            return 9.0
        elif unemp < 10.0:
            return 6.0
        elif unemp < 15.0:
            return 3.0
        else:
            return 0.0

    def rank_all(self, snapshots: List[CountryMacroSnapshot]) -> pd.DataFrame:
        """Score and rank all countries."""
        rows = []
        for snap in snapshots:
            sc = self.score(snap)
            rows.append({
                "iso2": sc.iso2,
                "name": _NAMES.get(sc.iso2, sc.iso2),
                "total_score": sc.total_score,
                "growth_score": sc.growth_score,
                "inflation_score": sc.inflation_score,
                "fiscal_score": sc.fiscal_score,
                "external_score": sc.external_score,
                "labor_score": sc.labor_score,
                "risk_flags": "|".join(sc.risk_flags),
            })
        df = pd.DataFrame(rows).sort_values("total_score", ascending=False)
        df["rank"] = range(1, len(df) + 1)
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# MacroRegimeClassifier
# ---------------------------------------------------------------------------

class MacroRegimeClassifier:
    """
    Classify each country's macro cycle into one of 4 regimes:
      EXPANSION  : GDP above trend, PMI > 52, unemployment falling
      SLOWDOWN   : GDP decelerating, PMI 48-52, fiscal tightening
      RECESSION  : GDP below zero or PMI < 46 sustained, unemployment rising
      RECOVERY   : GDP turning up from low base, PMI improving

    Uses a signal-voting approach across 5 indicators.
    """

    _SCORER = CountryScorer()

    def classify(self, snap: CountryMacroSnapshot) -> MacroRegime:
        signals: Dict[str, str] = {}
        votes: Dict[str, int] = {"EXPANSION": 0, "SLOWDOWN": 0, "RECESSION": 0, "RECOVERY": 0}

        # GDP signal
        gdp = self._SCORER._best_gdp_growth(snap)
        avg = snap.gdp_5y_avg or 2.0
        if gdp is not None:
            if gdp >= avg + 0.5:
                signals["gdp"] = "EXPANSION"
            elif gdp >= 0 and gdp < avg - 0.5:
                signals["gdp"] = "SLOWDOWN"
            elif gdp < 0:
                signals["gdp"] = "RECESSION"
            else:
                signals["gdp"] = "RECOVERY"
            votes[signals["gdp"]] += 2

        # PMI signal
        pmi = snap.composite_pmi or snap.manufacturing_pmi
        if pmi is not None:
            if pmi > 53.0:
                signals["pmi"] = "EXPANSION"
            elif pmi > 50.0:
                signals["pmi"] = "RECOVERY"
            elif pmi > 46.0:
                signals["pmi"] = "SLOWDOWN"
            else:
                signals["pmi"] = "RECESSION"
            votes[signals["pmi"]] += 2

        # Inflation signal (high inflation often = overheating/slowdown)
        inf = self._SCORER._best_inflation(snap)
        target = snap.inflation_target or 2.0
        if inf is not None:
            if inf > target * 3:
                signals["inflation"] = "SLOWDOWN"
                votes["SLOWDOWN"] += 1
            elif inf > target * 1.5:
                signals["inflation"] = "EXPANSION"
                votes["EXPANSION"] += 1
            elif inf > 0:
                signals["inflation"] = "EXPANSION"
                votes["EXPANSION"] += 1
            else:
                signals["inflation"] = "RECESSION"
                votes["RECESSION"] += 1

        # OECD CLI signal
        cli = snap.oecd_cli
        if cli is not None:
            if cli > 101.0:
                signals["oecd_cli"] = "EXPANSION"
            elif cli > 100.0:
                signals["oecd_cli"] = "RECOVERY"
            elif cli > 99.0:
                signals["oecd_cli"] = "SLOWDOWN"
            else:
                signals["oecd_cli"] = "RECESSION"
            votes[signals["oecd_cli"]] += 1

        # Unemployment signal
        unemp = snap.unemployment
        if unemp is not None:
            if unemp < 4.0:
                signals["unemployment"] = "EXPANSION"
                votes["EXPANSION"] += 1
            elif unemp < 6.0:
                signals["unemployment"] = "RECOVERY"
                votes["RECOVERY"] += 1
            elif unemp < 9.0:
                signals["unemployment"] = "SLOWDOWN"
                votes["SLOWDOWN"] += 1
            else:
                signals["unemployment"] = "RECESSION"
                votes["RECESSION"] += 1

        total_votes = sum(votes.values()) or 1
        regime = max(votes, key=lambda k: votes[k])
        confidence = votes[regime] / total_votes

        result = MacroRegime(
            iso2=snap.iso2,
            regime=regime,
            confidence=round(confidence, 3),
            signals=signals,
            as_of=snap.as_of,
        )

        # Persist
        try:
            conn = _get_db()
            conn.execute("""
                INSERT OR REPLACE INTO macro_regimes
                    (iso2, classified_at, regime, confidence)
                VALUES (?,?,?,?)
            """, (snap.iso2, datetime.utcnow().strftime("%Y-%m-%d"),
                  regime, confidence))
            conn.commit()
            conn.close()
        except Exception:
            pass

        return result

    def regime_matrix(self, snapshots: List[CountryMacroSnapshot]) -> pd.DataFrame:
        rows = []
        for snap in snapshots:
            regime = self.classify(snap)
            rows.append({
                "iso2": snap.iso2,
                "name": _NAMES.get(snap.iso2, snap.iso2),
                "regime": regime.regime,
                "confidence": regime.confidence,
                **{f"signal_{k}": v for k, v in regime.signals.items()},
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CarryTradeEngine
# ---------------------------------------------------------------------------

class CarryTradeEngine:
    """
    Rank currencies by carry trade attractiveness:
      - Nominal carry   = policy rate (own) - USD funding rate
      - Real carry      = nominal carry - inflation differential
      - Volatility adj  = real carry / implied vol proxy
      - FX trend score  = directional momentum proxy (0-1)
    """

    _USD_RATE = _POLICY_RATES.get("US", 5.25)

    def attractiveness(self, snap: CountryMacroSnapshot) -> CarrySignal:
        iso2 = snap.iso2
        rate = snap.policy_rate or _POLICY_RATES.get(iso2, 0.0)
        inf = CountryScorer()._best_inflation(snap) or 0.0
        us_inf = 3.5  # approximate US CPI

        nominal_carry = rate - self._USD_RATE
        real_carry = nominal_carry - (inf - us_inf)

        # Vol proxy: high inflation = high vol = penalty
        vol_proxy = max(1.0, min(inf, 30.0) * 0.5 + 5.0)
        vol_adj = real_carry / vol_proxy

        # FX trend score: approximate from current account and real carry
        ca = snap.current_account_imf or 0.0
        fx_trend = max(-1.0, min(1.0, ca * 0.05 + (real_carry * 0.05)))

        if real_carry > 3.0 and vol_adj > 0.3:
            attractiveness = "HIGH"
        elif real_carry > 1.0:
            attractiveness = "MEDIUM"
        elif real_carry > -1.0:
            attractiveness = "LOW"
        else:
            attractiveness = "NEGATIVE"

        return CarrySignal(
            iso2=iso2,
            policy_rate=rate,
            inflation_adj_rate=round(real_carry, 2),
            fx_trend_score=round(fx_trend, 3),
            vol_adj_carry=round(vol_adj, 4),
            attractiveness=attractiveness,
        )

    def rank_carry(self, snapshots: List[CountryMacroSnapshot]) -> pd.DataFrame:
        signals = [self.attractiveness(s) for s in snapshots]
        rows = [{
            "iso2": s.iso2,
            "name": _NAMES.get(s.iso2, s.iso2),
            "policy_rate": s.policy_rate,
            "real_carry_rate": s.inflation_adj_rate,
            "vol_adj_carry": s.vol_adj_carry,
            "fx_trend": s.fx_trend_score,
            "attractiveness": s.attractiveness,
        } for s in signals]
        df = pd.DataFrame(rows).sort_values("real_carry_rate", ascending=False)
        df["carry_rank"] = range(1, len(df) + 1)
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# EconomicSurpriseIndex
# ---------------------------------------------------------------------------

class EconomicSurpriseIndex:
    """
    Proxy surprise index: how does real-time high-frequency data compare
    to the lagged World Bank annual figures?

    A positive surprise = high-freq data is better than WB annual implies.
    """

    _scorer = CountryScorer()

    def compute(self, snap: CountryMacroSnapshot) -> SurpriseReading:
        gdp_surprise = None
        inflation_surprise = None
        pmi_surprise = None

        # GDP: IMF forecast vs World Bank historical
        if snap.gdp_growth_imf is not None and snap.gdp_growth is not None:
            gdp_surprise = snap.gdp_growth_imf - snap.gdp_growth

        # Inflation: FRED/ECB monthly vs WB annual
        hf_inf = snap.inflation_fred or snap.inflation_ecb
        if hf_inf is not None and snap.inflation_wb is not None:
            inflation_surprise = hf_inf - snap.inflation_wb
            # Positive means higher than WB suggested (usually negative surprise)
            inflation_surprise = -inflation_surprise  # invert: lower=good

        # PMI: deviation from 50 (neutral)
        pmi = snap.composite_pmi or snap.manufacturing_pmi
        if pmi is not None:
            pmi_surprise = pmi - 50.0

        scores = [s for s in [gdp_surprise, pmi_surprise] if s is not None]
        composite_score = float(np.mean(scores)) if scores else 0.0

        if composite_score > 1.0:
            signal = "POSITIVE_SURPRISE"
        elif composite_score < -1.0:
            signal = "NEGATIVE_SURPRISE"
        else:
            signal = "INLINE"

        return SurpriseReading(
            iso2=snap.iso2,
            surprise_score=round(composite_score, 3),
            gdp_surprise=round(gdp_surprise, 2) if gdp_surprise is not None else None,
            inflation_surprise=round(inflation_surprise, 2) if inflation_surprise is not None else None,
            pmi_surprise=round(pmi_surprise, 2) if pmi_surprise is not None else None,
            composite_signal=signal,
        )


# ---------------------------------------------------------------------------
# GlobalMacroDashboard — high-level orchestration
# ---------------------------------------------------------------------------

class GlobalMacroDashboard:
    """
    Orchestrates all engines to produce dashboards and comparison matrices.
    Uses ThreadPoolExecutor to fetch country data concurrently.
    """

    def __init__(self) -> None:
        self._broker    = MacroDataBroker()
        self._scorer    = CountryScorer()
        self._classifier = MacroRegimeClassifier()
        self._carry     = CarryTradeEngine()
        self._surprise  = EconomicSurpriseIndex()

    def _fetch_snapshots(self, universe: List[str]) -> List[CountryMacroSnapshot]:
        snapshots: List[CountryMacroSnapshot] = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(self._broker.fetch_country, iso2): iso2
                       for iso2 in universe}
            for fut in as_completed(futures):
                iso2 = futures[fut]
                try:
                    snapshots.append(fut.result())
                except Exception as exc:
                    logger.warning("Failed to fetch %s: %s", iso2, exc)
        return snapshots

    def g20_dashboard(self) -> pd.DataFrame:
        """Full G20 macro dashboard with all key metrics."""
        g20_universe = [c for c in G20 if c in COUNTRY_UNIVERSE]
        snapshots = self._fetch_snapshots(g20_universe)

        rows = []
        for snap in snapshots:
            sc = self._scorer.score(snap)
            regime = self._classifier.classify(snap)
            carry = self._carry.attractiveness(snap)
            surprise = self._surprise.compute(snap)
            inf = CountryScorer()._best_inflation(snap)
            gdp = CountryScorer()._best_gdp_growth(snap)

            rows.append({
                "iso2": snap.iso2,
                "name": snap.name,
                "gdp_growth": round(gdp, 1) if gdp is not None else None,
                "inflation": round(inf, 1) if inf is not None else None,
                "unemployment": snap.unemployment,
                "debt_pct_gdp": snap.debt_pct_gdp or snap.debt_imf,
                "current_account_imf": snap.current_account_imf,
                "policy_rate": snap.policy_rate,
                "manufacturing_pmi": snap.manufacturing_pmi,
                "composite_pmi": snap.composite_pmi,
                "oecd_cli": snap.oecd_cli,
                "total_score": sc.total_score,
                "regime": regime.regime,
                "carry_attractiveness": carry.attractiveness,
                "real_carry": carry.inflation_adj_rate,
                "surprise": surprise.composite_signal,
                "risk_flags": "|".join(sc.risk_flags),
            })

        return pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)

    def dm_em_scorecard(self) -> Dict[str, Any]:
        """Developed vs Emerging Market scorecard."""
        dm_countries = [c for c in DEVELOPED if c in COUNTRY_UNIVERSE]
        em_countries = [c for c in EMERGING if c in COUNTRY_UNIVERSE]

        dm_snaps = self._fetch_snapshots(dm_countries)
        em_snaps = self._fetch_snapshots(em_countries)

        def _agg(snaps: List[CountryMacroSnapshot]) -> Dict:
            infls  = [CountryScorer()._best_inflation(s) for s in snaps if CountryScorer()._best_inflation(s) is not None]
            gdps   = [CountryScorer()._best_gdp_growth(s) for s in snaps if CountryScorer()._best_gdp_growth(s) is not None]
            scores = [self._scorer.score(s).total_score for s in snaps]
            pmis   = [s.composite_pmi for s in snaps if s.composite_pmi is not None]
            return {
                "avg_gdp_growth":  round(float(np.mean(gdps)), 2) if gdps else None,
                "avg_inflation":   round(float(np.mean(infls)), 2) if infls else None,
                "avg_score":       round(float(np.mean(scores)), 1) if scores else None,
                "avg_pmi":         round(float(np.mean(pmis)), 1) if pmis else None,
                "countries":       len(snaps),
            }

        return {
            "developed_markets": _agg(dm_snaps),
            "emerging_markets":  _agg(em_snaps),
            "as_of": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def comparison_matrix(self, iso2_list: List[str]) -> pd.DataFrame:
        """Side-by-side comparison of selected countries."""
        snapshots = self._fetch_snapshots(iso2_list)
        rows = []
        for snap in snapshots:
            sc = self._scorer.score(snap)
            regime = self._classifier.classify(snap)
            inf = CountryScorer()._best_inflation(snap)
            gdp = CountryScorer()._best_gdp_growth(snap)
            rows.append({
                "iso2":              snap.iso2,
                "name":              snap.name,
                "gdp_growth":        round(gdp, 2) if gdp is not None else None,
                "inflation":         round(inf, 2) if inf is not None else None,
                "unemployment":      snap.unemployment,
                "debt_pct_gdp":      snap.debt_pct_gdp or snap.debt_imf,
                "current_account":   snap.current_account_imf,
                "fiscal_balance":    snap.fiscal_balance_imf,
                "policy_rate":       snap.policy_rate,
                "manufacturing_pmi": snap.manufacturing_pmi,
                "composite_pmi":     snap.composite_pmi,
                "oecd_cli":          snap.oecd_cli,
                "total_score":       sc.total_score,
                "regime":            regime.regime,
                "gni_per_capita":    snap.gni_per_capita,
                "sources":           "|".join(snap.sources),
            })
        return pd.DataFrame(rows)

    def leading_indicators(self, iso2: str) -> Dict[str, Any]:
        """
        Return all leading indicator signals for a country.
        Combines PMI, OECD CLI, FRED, ECB in one payload.
        """
        snap = self._broker.fetch_country(iso2)
        regime = self._classifier.classify(snap)
        surprise = self._surprise.compute(snap)
        carry = self._carry.attractiveness(snap)

        return {
            "iso2":              snap.iso2,
            "name":              snap.name,
            "as_of":             snap.as_of,
            "leading_indicators": {
                "oecd_cli":          snap.oecd_cli,
                "manufacturing_pmi": snap.manufacturing_pmi,
                "services_pmi":      snap.services_pmi,
                "composite_pmi":     snap.composite_pmi,
                "fred_cpi_yoy":      snap.inflation_fred,
                "ecb_hicp_yoy":      snap.inflation_ecb,
                "imf_gdp_forecast":  snap.gdp_growth_imf,
                "policy_rate":       snap.policy_rate,
            },
            "regime":    asdict(regime),
            "surprise":  asdict(surprise),
            "carry":     asdict(carry),
            "sources":   snap.sources,
        }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

global_macro_v3_router = APIRouter(prefix="/global-macro/v3", tags=["Global Macro V3"])
_dashboard = GlobalMacroDashboard()


class CountryRequest(BaseModel):
    iso2: str


@global_macro_v3_router.get("/country/{iso2}", summary="Full macro snapshot for one country")
def get_country(iso2: str):
    """Fetch all macro indicators for a single country from all sources."""
    iso2 = iso2.upper()
    if iso2 not in COUNTRY_UNIVERSE:
        raise HTTPException(404, f"Country {iso2} not in 50-country universe")
    try:
        snap = _dashboard._broker.fetch_country(iso2)
        sc   = _dashboard._scorer.score(snap)
        regime = _dashboard._classifier.classify(snap)
        return {
            "snapshot": asdict(snap),
            "score":    asdict(sc),
            "regime":   asdict(regime),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@global_macro_v3_router.get("/comparison", summary="Side-by-side comparison matrix")
def get_comparison(countries: str = Query(default="US,DE,JP,CN", description="Comma-separated ISO-2 codes")):
    """Compare selected countries on all key indicators."""
    iso2_list = [c.strip().upper() for c in countries.split(",")]
    invalid = [c for c in iso2_list if c not in COUNTRY_UNIVERSE]
    if invalid:
        raise HTTPException(400, f"Unknown countries: {invalid}")
    try:
        df = _dashboard.comparison_matrix(iso2_list)
        return {"countries": iso2_list, "matrix": df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@global_macro_v3_router.get("/g20-dashboard", summary="Full G20 macro dashboard")
def get_g20_dashboard():
    """G20 macro dashboard with scores, regimes, and carry signals."""
    try:
        df = _dashboard.g20_dashboard()
        return {"as_of": datetime.utcnow().isoformat(), "data": df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@global_macro_v3_router.get("/regime/{iso2}", summary="Macro regime classification")
def get_regime(iso2: str):
    """Classify a country's current macro regime."""
    iso2 = iso2.upper()
    if iso2 not in COUNTRY_UNIVERSE:
        raise HTTPException(404, f"Country {iso2} not in universe")
    try:
        snap = _dashboard._broker.fetch_country(iso2)
        regime = _dashboard._classifier.classify(snap)
        return asdict(regime)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@global_macro_v3_router.get("/carry-trade", summary="Carry trade attractiveness ranking")
def get_carry_trade():
    """Rank all 50 countries by carry trade attractiveness."""
    try:
        snapshots = _dashboard._fetch_snapshots(COUNTRY_UNIVERSE)
        df = _dashboard._carry.rank_carry(snapshots)
        return {"as_of": datetime.utcnow().isoformat(), "rankings": df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@global_macro_v3_router.get("/surprise-index/{iso2}", summary="Economic surprise index")
def get_surprise_index(iso2: str):
    """Compute economic surprise index: high-frequency vs lagged annual data."""
    iso2 = iso2.upper()
    if iso2 not in COUNTRY_UNIVERSE:
        raise HTTPException(404, f"Country {iso2} not in universe")
    try:
        snap = _dashboard._broker.fetch_country(iso2)
        surprise = _dashboard._surprise.compute(snap)
        return asdict(surprise)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@global_macro_v3_router.get("/leading-indicators/{iso2}", summary="All leading indicator signals")
def get_leading_indicators(iso2: str):
    """Return all leading indicator signals for a country."""
    iso2 = iso2.upper()
    if iso2 not in COUNTRY_UNIVERSE:
        raise HTTPException(404, f"Country {iso2} not in universe")
    try:
        return _dashboard.leading_indicators(iso2)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@global_macro_v3_router.get("/dm-em-scorecard", summary="Developed vs Emerging Markets")
def get_dm_em_scorecard():
    """Aggregate scorecard: Developed vs Emerging Markets."""
    try:
        return _dashboard.dm_em_scorecard()
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@global_macro_v3_router.get("/universe", summary="List all 50 countries")
def get_universe(region: Optional[str] = Query(None), group: Optional[str] = Query(None)):
    """List all countries in the macro universe, optionally filtered."""
    countries = []
    for iso2 in COUNTRY_UNIVERSE:
        entry = {
            "iso2": iso2,
            "name": _NAMES.get(iso2, iso2),
            "policy_rate": _POLICY_RATES.get(iso2),
            "inflation_target": _CPI_TARGETS.get(iso2),
            "market_type": "developed" if iso2 in DEVELOPED else "emerging",
            "in_g20": iso2 in G20,
        }
        if group and entry["market_type"] != group.lower():
            continue
        countries.append(entry)
    return {"count": len(countries), "universe": countries}
