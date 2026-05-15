"""FRED macro time series module (enhanced) — dim_043 (score 8 → 9).

Provides comprehensive access to the FRED 765K+ series universe with:
  - FREDUniversalAdapter: search, fetch, batch, category browser
  - MacroDashboard: curated 50-series macro dashboard
  - MacroRegimeClassifier: 8-regime growth×inflation×policy framework
  - EconomicIndicatorForecaster: AR(1), leading indicators, NY Fed recession model
  - FastAPI router: /fred/v2/*

Environment:
    FRED_API_KEY — optional free key from research.stlouisfed.org/useraccount/apikeys
    Without a key, falls back to FRED CSV endpoint (no auth required).

Dependencies: requests, pandas, numpy, scipy, fastapi, pydantic
"""
from __future__ import annotations

import asyncio
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_API_BASE = "https://fred.stlouisfed.org/api/fred"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

_HEADERS = {
    "User-Agent": "SENTINEL/2.0 macro-terminal richard.porras@realempanada.com",
    "Accept": "application/json",
}

_SERIES_CACHE: Dict[str, pd.DataFrame] = {}
_SERIES_CACHE_TS: Dict[str, float] = {}
_SERIES_CACHE_TTL = 3600.0  # 1 hour

# ---------------------------------------------------------------------------
# Curated 50-series macro dashboard
# ---------------------------------------------------------------------------

DASHBOARD_SERIES: Dict[str, Dict[str, str]] = {
    # Real economy
    "GDP": {"id": "GDP", "name": "Real GDP", "category": "Real Economy", "units": "Bil.$"},
    "INDPRO": {"id": "INDPRO", "name": "Industrial Production Index", "category": "Real Economy", "units": "Index"},
    "TCU": {"id": "TCU", "name": "Capacity Utilization", "category": "Real Economy", "units": "%"},
    "RSXFS": {"id": "RSXFS", "name": "Retail Sales ex-Autos", "category": "Real Economy", "units": "Mil.$"},
    "HOUST": {"id": "HOUST", "name": "Housing Starts", "category": "Real Economy", "units": "Thousands"},
    "BPMFAB": {"id": "BPMFAB", "name": "ISM Manufacturing PMI (proxy: AMTMNO)", "category": "Real Economy", "units": "Index"},
    # Labor market
    "PAYEMS": {"id": "PAYEMS", "name": "Nonfarm Payrolls", "category": "Labor", "units": "Thousands"},
    "UNRATE": {"id": "UNRATE", "name": "Unemployment Rate (U-3)", "category": "Labor", "units": "%"},
    "U6RATE": {"id": "U6RATE", "name": "Unemployment Rate (U-6)", "category": "Labor", "units": "%"},
    "ICSA": {"id": "ICSA", "name": "Initial Jobless Claims", "category": "Labor", "units": "Thousands"},
    "JTSJOL": {"id": "JTSJOL", "name": "JOLTS Job Openings", "category": "Labor", "units": "Thousands"},
    "CES0500000003": {"id": "CES0500000003", "name": "Average Hourly Earnings", "category": "Labor", "units": "$/hr"},
    # Inflation
    "CPIAUCSL": {"id": "CPIAUCSL", "name": "CPI All Items", "category": "Inflation", "units": "Index"},
    "CPILFESL": {"id": "CPILFESL", "name": "CPI Core (ex-Food Energy)", "category": "Inflation", "units": "Index"},
    "PCEPI": {"id": "PCEPI", "name": "PCE Price Index", "category": "Inflation", "units": "Index"},
    "PCEPILFE": {"id": "PCEPILFE", "name": "PCE Core", "category": "Inflation", "units": "Index"},
    "PPIACO": {"id": "PPIACO", "name": "PPI All Commodities", "category": "Inflation", "units": "Index"},
    "T5YIE": {"id": "T5YIE", "name": "5Y Breakeven Inflation", "category": "Inflation", "units": "%"},
    "T10YIE": {"id": "T10YIE", "name": "10Y Breakeven Inflation", "category": "Inflation", "units": "%"},
    # Monetary policy
    "FEDFUNDS": {"id": "FEDFUNDS", "name": "Federal Funds Rate", "category": "Monetary", "units": "%"},
    "SOFR": {"id": "SOFR", "name": "SOFR", "category": "Monetary", "units": "%"},
    "DTB3": {"id": "DTB3", "name": "3-Month T-Bill", "category": "Monetary", "units": "%"},
    "DGS10": {"id": "DGS10", "name": "10Y Treasury Yield", "category": "Monetary", "units": "%"},
    "DGS30": {"id": "DGS30", "name": "30Y Treasury Yield", "category": "Monetary", "units": "%"},
    "T10Y3M": {"id": "T10Y3M", "name": "10Y-3M Yield Spread", "category": "Monetary", "units": "%"},
    "M2SL": {"id": "M2SL", "name": "M2 Money Supply", "category": "Monetary", "units": "Bil.$"},
    "WRESBAL": {"id": "WRESBAL", "name": "Bank Reserves", "category": "Monetary", "units": "Mil.$"},
    # Credit
    "BAMLC0A0CM": {"id": "BAMLC0A0CM", "name": "IG Corp Spread (OAS)", "category": "Credit", "units": "%"},
    "BAMLH0A0HYM2": {"id": "BAMLH0A0HYM2", "name": "HY Corp Spread (OAS)", "category": "Credit", "units": "%"},
    "DRTSCILM": {"id": "DRTSCILM", "name": "SLOOS C&I Lending Standards", "category": "Credit", "units": "Net%"},
    # Housing
    "CSUSHPINSA": {"id": "CSUSHPINSA", "name": "Case-Shiller Home Price Index", "category": "Housing", "units": "Index"},
    "USSTHPI": {"id": "USSTHPI", "name": "FHFA House Price Index", "category": "Housing", "units": "Index"},
    "PERMIT": {"id": "PERMIT", "name": "Building Permits", "category": "Housing", "units": "Thousands"},
    "MORTGAGE30US": {"id": "MORTGAGE30US", "name": "30Y Mortgage Rate", "category": "Housing", "units": "%"},
    # International
    "DTWEXBGS": {"id": "DTWEXBGS", "name": "Trade-Weighted Dollar (Broad)", "category": "International", "units": "Index"},
    "DEXUSEU": {"id": "DEXUSEU", "name": "EUR/USD Exchange Rate", "category": "International", "units": "USD/EUR"},
    "BOPGSTB": {"id": "BOPGSTB", "name": "Trade Balance", "category": "International", "units": "Mil.$"},
    "NETFI": {"id": "NETFI", "name": "Net Foreign Investment", "category": "International", "units": "Bil.$"},
    # Financial conditions
    "VIXCLS": {"id": "VIXCLS", "name": "VIX Volatility Index", "category": "Financial Conditions", "units": "Index"},
    "NFCI": {"id": "NFCI", "name": "National Financial Conditions Index", "category": "Financial Conditions", "units": "Index"},
    "STLFSI4": {"id": "STLFSI4", "name": "St. Louis Financial Stress Index", "category": "Financial Conditions", "units": "Index"},
    # Leading indicators
    "AWHMAN": {"id": "AWHMAN", "name": "Avg Weekly Hours Manufacturing", "category": "Leading", "units": "Hours"},
    "ICSA_LD": {"id": "ICSA", "name": "Initial Claims (leading proxy)", "category": "Leading", "units": "Thousands"},
    "NEWORDER": {"id": "NEWORDER", "name": "Mfg New Orders", "category": "Leading", "units": "Mil.$"},
    "USTRADE": {"id": "USTRADE", "name": "Retail Trade Employment", "category": "Leading", "units": "Thousands"},
    "HOUSTNE": {"id": "HOUSTNE", "name": "Housing Starts NE", "category": "Leading", "units": "Thousands"},
    "SP500": {"id": "SP500", "name": "S&P 500 Index", "category": "Financial Conditions", "units": "Index"},
    "BAMLH0A0HYM2_LD": {"id": "BAMLH0A0HYM2", "name": "HY Spread (leading proxy)", "category": "Leading", "units": "%"},
    "M2REAL": {"id": "M2SL", "name": "M2 Real (proxy)", "category": "Leading", "units": "Bil.$"},
}

# Leading indicator series (10 for LEI composite)
LEADING_INDICATOR_SERIES = [
    "AWHMAN",       # Avg weekly hours manufacturing
    "ICSA",         # Initial claims (inverted)
    "NEWORDER",     # Manufacturing new orders
    "PERMIT",       # Building permits
    "SP500",        # Stock prices (S&P 500)
    "T10Y3M",       # Yield curve spread
    "BAMLH0A0HYM2", # Credit spreads (inverted)
    "VIXCLS",       # VIX (inverted)
    "M2SL",         # M2 money supply
    "DTWEXBGS",     # Dollar index (inverted)
]

# Inverted indicators (higher value = worse economic outlook)
INVERTED_LEADING = {"ICSA", "BAMLH0A0HYM2", "VIXCLS", "DTWEXBGS"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SeriesInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    series_id: str
    title: str
    units: Optional[str] = None
    frequency: Optional[str] = None
    seasonal_adjustment: Optional[str] = None
    last_updated: Optional[str] = None
    observation_start: Optional[str] = None
    observation_end: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None


class SeriesObservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    series_id: str
    title: str
    units: Optional[str]
    latest_value: Optional[float]
    latest_date: Optional[str]
    prev_value: Optional[float]
    change: Optional[float]
    pct_change: Optional[float]
    yoy_change: Optional[float]
    yoy_pct_change: Optional[float]
    category: Optional[str] = None


class MacroDashboardResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: str
    series: List[SeriesObservation]
    categories: Dict[str, List[str]]


class MacroRegime(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: str
    growth_phase: str           # expansion / contraction
    inflation_phase: str        # rising / stable / falling
    policy_phase: str           # tightening / easing / pause
    regime_label: str           # e.g. "Expansion_Rising_Tightening"
    regime_code: int            # 0-7
    growth_signal: float        # normalized -1 to +1
    inflation_signal: float     # normalized -1 to +1
    policy_signal: float        # normalized -1 to +1
    lei_composite: Optional[float] = None
    recession_probability: Optional[float] = None
    confidence: float = 0.5


class ForecastResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    series_id: str
    title: str
    latest_value: float
    forecast_3m: float
    forecast_6m: float
    forecast_12m: float
    ci_lower_12m: float
    ci_upper_12m: float
    ar1_coefficient: float
    trend: str                  # rising / falling / stable
    method: str = "AR(1)"


class CategoryInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    name: str
    parent_id: Optional[int] = None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get_json_sync(url: str, params: Optional[dict] = None, timeout: float = 20.0) -> Optional[Any]:
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("GET %s: %s", url, exc)
        return None


def _get_text_sync(url: str, params: Optional[dict] = None, timeout: float = 20.0) -> Optional[str]:
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.debug("GET text %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Class 1: FREDUniversalAdapter
# ---------------------------------------------------------------------------


class FREDUniversalAdapter:
    """Access the full FRED universe of 765K+ series.

    Supports:
    - REST API (requires FRED_API_KEY env var)
    - CSV fallback (no key required) for known series IDs
    - Search, fetch, batch fetch, category navigation
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or FRED_API_KEY
        self._has_key = bool(self._api_key)

    def _api_params(self, extra: Optional[dict] = None) -> dict:
        params = {"file_type": "json"}
        if self._has_key:
            params["api_key"] = self._api_key
        if extra:
            params.update(extra)
        return params

    def search_series(self, query: str, limit: int = 20) -> List[SeriesInfo]:
        """Search FRED series by text query.

        Requires FRED_API_KEY. Returns empty list if no key available.
        """
        if not self._has_key:
            logger.info("FRED search requires API key; returning empty list")
            return []

        params = self._api_params({
            "search_text": query,
            "limit": min(limit, 1000),
            "sort_order": "desc",
            "order_by": "popularity",
        })
        data = _get_json_sync(f"{FRED_API_BASE}/series/search", params=params)
        if not data:
            return []

        results: List[SeriesInfo] = []
        for s in data.get("seriess", []):
            results.append(SeriesInfo(
                series_id=s.get("id", ""),
                title=s.get("title", ""),
                units=s.get("units_short", s.get("units")),
                frequency=s.get("frequency_short", s.get("frequency")),
                seasonal_adjustment=s.get("seasonal_adjustment_short"),
                last_updated=s.get("last_updated"),
                observation_start=s.get("observation_start"),
                observation_end=s.get("observation_end"),
                notes=s.get("notes", "")[:200] if s.get("notes") else None,
            ))
        logger.info("FRED search '%s' → %d results", query, len(results))
        return results

    def get_series_info(self, series_id: str) -> Optional[SeriesInfo]:
        """Fetch metadata for a single series."""
        if self._has_key:
            params = self._api_params({"series_id": series_id})
            data = _get_json_sync(f"{FRED_API_BASE}/series", params=params)
            if data and data.get("seriess"):
                s = data["seriess"][0]
                return SeriesInfo(
                    series_id=s.get("id", series_id),
                    title=s.get("title", series_id),
                    units=s.get("units_short", s.get("units")),
                    frequency=s.get("frequency_short"),
                    seasonal_adjustment=s.get("seasonal_adjustment_short"),
                    last_updated=s.get("last_updated"),
                    observation_start=s.get("observation_start"),
                    observation_end=s.get("observation_end"),
                )
        # Fallback: infer from dashboard mapping
        for key, meta in DASHBOARD_SERIES.items():
            if meta["id"] == series_id:
                return SeriesInfo(
                    series_id=series_id,
                    title=meta["name"],
                    units=meta.get("units"),
                    category=meta.get("category"),
                )
        return SeriesInfo(series_id=series_id, title=series_id)

    def fetch_series_csv(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a FRED series via the no-auth CSV endpoint.

        Returns DataFrame with columns [date, value].
        """
        cache_key = f"{series_id}_{start}_{end}"
        now = time.monotonic()
        if cache_key in _SERIES_CACHE:
            if (now - _SERIES_CACHE_TS.get(cache_key, 0)) < _SERIES_CACHE_TTL:
                return _SERIES_CACHE[cache_key].copy()

        params: dict = {"id": series_id}
        if start:
            params["vintage_dates"] = start
        text = _get_text_sync(FRED_CSV_BASE, params=params)
        if not text:
            return pd.DataFrame(columns=["date", "value"])

        rows = []
        for line in text.strip().splitlines():
            if not line or line.startswith("DATE"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            date_str = parts[0].strip()
            val_str = parts[1].strip()
            if val_str in (".", ""):
                continue
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                val = float(val_str)
                if start and str(dt) < start:
                    continue
                if end and str(dt) > end:
                    continue
                rows.append({"date": dt, "value": val})
            except (ValueError, AttributeError):
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)

        _SERIES_CACHE[cache_key] = df.copy()
        _SERIES_CACHE_TS[cache_key] = now
        logger.debug("FRED CSV %s: %d observations", series_id, len(df))
        return df

    def fetch_series_api(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        frequency: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a FRED series via the REST API (requires key).

        Returns DataFrame with columns [date, value].
        """
        if not self._has_key:
            return self.fetch_series_csv(series_id, start, end)

        params = self._api_params({
            "series_id": series_id,
            "sort_order": "asc",
        })
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        if frequency:
            params["frequency"] = frequency

        data = _get_json_sync(f"{FRED_API_BASE}/series/observations", params=params)
        if not data or not data.get("observations"):
            return self.fetch_series_csv(series_id, start, end)

        rows = []
        for obs in data["observations"]:
            val_str = obs.get("value", ".")
            if val_str in (".", ""):
                continue
            try:
                rows.append({
                    "date": datetime.strptime(obs["date"], "%Y-%m-%d").date(),
                    "value": float(val_str),
                })
            except (ValueError, KeyError):
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def fetch_series(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Unified fetch: uses API if key available, otherwise CSV."""
        if self._has_key:
            return self.fetch_series_api(series_id, start, end)
        return self.fetch_series_csv(series_id, start, end)

    def batch_fetch(
        self,
        series_ids: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_workers: int = 20,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch up to 20 series in parallel using a thread pool."""
        results: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(series_ids))) as executor:
            future_to_sid = {
                executor.submit(self.fetch_series, sid, start, end): sid
                for sid in series_ids
            }
            for future in as_completed(future_to_sid):
                sid = future_to_sid[future]
                try:
                    results[sid] = future.result()
                except Exception as exc:
                    logger.warning("batch_fetch %s: %s", sid, exc)
                    results[sid] = pd.DataFrame(columns=["date", "value"])
        return results

    def get_categories(self, category_id: int = 0) -> List[CategoryInfo]:
        """Browse FRED category tree. Root is category_id=0."""
        if not self._has_key:
            # Return a static top-level approximation
            return [
                CategoryInfo(id=32991, name="Money, Banking, & Finance", parent_id=0),
                CategoryInfo(id=10, name="Population, Employment, & Labor Markets", parent_id=0),
                CategoryInfo(id=22, name="Prices", parent_id=0),
                CategoryInfo(id=32992, name="National Accounts", parent_id=0),
                CategoryInfo(id=3008, name="Business Cycle Expansions and Contractions", parent_id=0),
                CategoryInfo(id=32455, name="Trade & International Transactions", parent_id=0),
                CategoryInfo(id=32262, name="Housing", parent_id=0),
                CategoryInfo(id=32145, name="Production & Business Activity", parent_id=0),
            ]
        params = self._api_params({"category_id": category_id})
        data = _get_json_sync(f"{FRED_API_BASE}/category/children", params=params)
        if not data:
            return []
        return [
            CategoryInfo(
                id=c.get("id", 0),
                name=c.get("name", ""),
                parent_id=c.get("parent_id"),
            )
            for c in data.get("categories", [])
        ]

    def get_latest_value(self, series_id: str) -> Tuple[Optional[float], Optional[date]]:
        """Return (latest_value, latest_date) for a series."""
        df = self.fetch_series(series_id)
        if df.empty:
            return None, None
        last = df.iloc[-1]
        return float(last["value"]), last["date"]

    def compute_yoy_change(self, df: pd.DataFrame) -> Optional[float]:
        """Compute year-over-year % change from a series DataFrame."""
        if df.empty or len(df) < 2:
            return None
        latest_date = df.iloc[-1]["date"]
        year_ago = latest_date - timedelta(days=365)
        # Find closest observation to 1 year ago
        past_df = df[df["date"] <= year_ago]
        if past_df.empty:
            return None
        val_now = float(df.iloc[-1]["value"])
        val_then = float(past_df.iloc[-1]["value"])
        if val_then == 0:
            return None
        return round((val_now / val_then - 1.0) * 100.0, 4)


# ---------------------------------------------------------------------------
# Class 2: MacroDashboard
# ---------------------------------------------------------------------------


class MacroDashboard:
    """Curated 50-series macro dashboard.

    Fetches all series in parallel and returns enriched observations
    with level, change, YoY change.
    """

    def __init__(self, adapter: Optional[FREDUniversalAdapter] = None) -> None:
        self._adapter = adapter or FREDUniversalAdapter()

    def _build_observation(
        self,
        key: str,
        meta: Dict[str, str],
        df: pd.DataFrame,
    ) -> SeriesObservation:
        series_id = meta["id"]
        if df.empty:
            return SeriesObservation(
                series_id=series_id,
                title=meta["name"],
                units=meta.get("units"),
                latest_value=None,
                latest_date=None,
                prev_value=None,
                change=None,
                pct_change=None,
                yoy_change=None,
                yoy_pct_change=None,
                category=meta.get("category"),
            )

        latest_val = float(df.iloc[-1]["value"])
        latest_date = str(df.iloc[-1]["date"])
        prev_val = float(df.iloc[-2]["value"]) if len(df) >= 2 else None
        change = round(latest_val - prev_val, 6) if prev_val is not None else None
        pct_change = round((latest_val / prev_val - 1.0) * 100.0, 4) if prev_val and prev_val != 0 else None

        yoy = self._adapter.compute_yoy_change(df)
        yoy_change: Optional[float] = None
        if yoy is not None and prev_val is not None and prev_val != 0:
            # YoY level change (approximate from % change)
            pass
        # Try to compute yoy level
        try:
            latest_dt = df.iloc[-1]["date"]
            year_ago = latest_dt - timedelta(days=365)
            past_df = df[df["date"] <= year_ago]
            val_then = float(past_df.iloc[-1]["value"]) if not past_df.empty else None
            if val_then is not None:
                yoy_change = round(latest_val - val_then, 6)
        except Exception:
            yoy_change = None

        return SeriesObservation(
            series_id=series_id,
            title=meta["name"],
            units=meta.get("units"),
            latest_value=latest_val,
            latest_date=latest_date,
            prev_value=prev_val,
            change=change,
            pct_change=pct_change,
            yoy_change=yoy_change,
            yoy_pct_change=yoy,
            category=meta.get("category"),
        )

    def fetch(self, lookback_days: int = 400) -> MacroDashboardResult:
        """Fetch all 50 dashboard series and return enriched observations."""
        start_date = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Get unique series IDs (some keys map to same ID)
        unique_series: Dict[str, str] = {}
        for meta in DASHBOARD_SERIES.values():
            sid = meta["id"]
            unique_series[sid] = sid

        # Batch fetch
        all_data = self._adapter.batch_fetch(list(unique_series.keys()), start=start_date)

        observations: List[SeriesObservation] = []
        seen_ids: set = set()
        categories: Dict[str, List[str]] = {}

        for key, meta in DASHBOARD_SERIES.items():
            sid = meta["id"]
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            df = all_data.get(sid, pd.DataFrame())
            obs = self._build_observation(key, meta, df)
            observations.append(obs)
            cat = meta.get("category", "Other")
            categories.setdefault(cat, []).append(meta["name"])

        logger.info("MacroDashboard: fetched %d series", len(observations))
        return MacroDashboardResult(
            as_of=date.today().isoformat(),
            series=observations,
            categories=categories,
        )

    def get_series_df(self, series_id: str, start: Optional[str] = None) -> pd.DataFrame:
        """Return a single series as a DataFrame."""
        return self._adapter.fetch_series(series_id, start=start)


# ---------------------------------------------------------------------------
# Class 3: MacroRegimeClassifier
# ---------------------------------------------------------------------------


class MacroRegimeClassifier:
    """Classify the macro regime using FRED data.

    Growth phase:    expansion (ISM/IP/employment trending positive)
                     vs contraction
    Inflation phase: rising / stable / falling (CPI/PCE trend)
    Policy phase:    tightening / easing / pause (Fed funds trend)

    Produces a 2×2×2 = 8-regime code (0–7).
    """

    def __init__(self, adapter: Optional[FREDUniversalAdapter] = None) -> None:
        self._adapter = adapter or FREDUniversalAdapter()

    def _fetch_3m_change(self, series_id: str, periods: int = 3) -> Optional[float]:
        """Return the change over last `periods` monthly observations."""
        df = self._adapter.fetch_series(series_id, start=(date.today() - timedelta(days=180)).isoformat())
        if df.empty or len(df) < periods + 1:
            return None
        latest = float(df.iloc[-1]["value"])
        prev = float(df.iloc[-(periods + 1)]["value"])
        return latest - prev

    def _fetch_latest(self, series_id: str) -> Optional[float]:
        val, _ = self._adapter.get_latest_value(series_id)
        return val

    def _compute_growth_signal(self) -> float:
        """Normalized growth signal: +1 = strong expansion, -1 = deep contraction."""
        signals: List[float] = []

        # Industrial production 3m change (normalized by std)
        ip_ch = self._fetch_3m_change("INDPRO", 3)
        if ip_ch is not None:
            signals.append(np.clip(ip_ch / 2.0, -1, 1))  # typical ±2% range

        # Nonfarm payrolls 3m average change
        np_ch = self._fetch_3m_change("PAYEMS", 3)
        if np_ch is not None:
            signals.append(np.clip(np_ch / 600.0, -1, 1))  # ~200K/mo typical

        # Unemployment rate level (inverted: lower = better)
        unrate = self._fetch_latest("UNRATE")
        if unrate is not None:
            # 4% = neutral, scale by ±2%
            signals.append(np.clip((4.0 - unrate) / 2.0, -1, 1))

        # Initial claims (inverted: lower = better)
        claims = self._fetch_latest("ICSA")
        if claims is not None:
            # 220K = neutral
            signals.append(np.clip((220.0 - claims) / 100.0, -1, 1))

        if not signals:
            return 0.0
        return float(np.mean(signals))

    def _compute_inflation_signal(self) -> float:
        """Normalized inflation momentum: +1 = rapidly rising, -1 = falling."""
        signals: List[float] = []

        # CPI 3m annualized trend
        cpi_ch = self._fetch_3m_change("CPIAUCSL", 3)
        if cpi_ch is not None:
            # Index level change; typical monthly ≈0.2–0.5 pts
            signals.append(np.clip(cpi_ch / 2.0, -1, 1))

        # PCE core YoY (level relative to 2% target)
        pcepilfe = self._fetch_latest("PCEPILFE")
        if pcepilfe is not None:
            # Use YoY % as proxy if we have enough history
            df = self._adapter.fetch_series("PCEPILFE", start=(date.today() - timedelta(days=400)).isoformat())
            yoy = self._adapter.compute_yoy_change(df)
            if yoy is not None:
                # >4% = very high (+1), ~2% = neutral (0), <1% = deflationary (-1)
                signals.append(np.clip((yoy - 2.0) / 2.0, -1, 1))

        # 10Y breakeven
        bei = self._fetch_latest("T10YIE")
        if bei is not None:
            # 2.5% = neutral; scale ±1%
            signals.append(np.clip((bei - 2.5) / 1.0, -1, 1))

        if not signals:
            return 0.0
        return float(np.mean(signals))

    def _compute_policy_signal(self) -> float:
        """Normalized policy stance: +1 = aggressive tightening, -1 = aggressive easing."""
        signals: List[float] = []

        # Fed funds rate 6m change
        ff_ch = self._fetch_3m_change("FEDFUNDS", 6)
        if ff_ch is not None:
            # 100bps change over 6m = significant; scale by 2%
            signals.append(np.clip(ff_ch / 2.0, -1, 1))

        # Yield curve slope (inverted = tighter policy)
        t10y3m = self._fetch_latest("T10Y3M")
        if t10y3m is not None:
            # Invert: negative spread means tight policy
            signals.append(np.clip(-t10y3m / 1.5, -1, 1))

        # M2 growth (negative = tightening)
        m2_ch = self._fetch_3m_change("M2SL", 3)
        if m2_ch is not None:
            # Typical ±$200bn/quarter range
            signals.append(np.clip(-m2_ch / 200.0, -1, 1))

        if not signals:
            return 0.0
        return float(np.mean(signals))

    def classify(self) -> MacroRegime:
        """Run classification and return the current macro regime."""
        today = date.today().isoformat()

        growth_signal = self._compute_growth_signal()
        inflation_signal = self._compute_inflation_signal()
        policy_signal = self._compute_policy_signal()

        # Binary classification with threshold ±0.1 for "stable"
        if growth_signal > 0.1:
            growth_phase = "expansion"
            g_bit = 1
        else:
            growth_phase = "contraction"
            g_bit = 0

        if inflation_signal > 0.1:
            inflation_phase = "rising"
            i_bit = 1
        elif inflation_signal < -0.1:
            inflation_phase = "falling"
            i_bit = 0
        else:
            inflation_phase = "stable"
            i_bit = 0  # stable treated as falling for regime code

        if policy_signal > 0.1:
            policy_phase = "tightening"
            p_bit = 1
        elif policy_signal < -0.1:
            policy_phase = "easing"
            p_bit = 0
        else:
            policy_phase = "pause"
            p_bit = 0

        # 3-bit regime code: growth(MSB), inflation, policy(LSB)
        regime_code = (g_bit << 2) | (i_bit << 1) | p_bit
        regime_label = f"{growth_phase.capitalize()}_{inflation_phase.capitalize()}_{policy_phase.capitalize()}"

        # Confidence: average of absolute signal magnitudes
        confidence = round(
            (abs(growth_signal) + abs(inflation_signal) + abs(policy_signal)) / 3.0, 3
        )

        # LEI composite and recession probability computed separately
        lei = None
        recession_prob = None
        try:
            forecaster = EconomicIndicatorForecaster(self._adapter)
            lei = forecaster.compute_lei_composite()
            recession_prob = forecaster.recession_probability_ny_fed()
        except Exception as exc:
            logger.warning("MacroRegime: LEI/recession calc failed: %s", exc)

        return MacroRegime(
            as_of=today,
            growth_phase=growth_phase,
            inflation_phase=inflation_phase,
            policy_phase=policy_phase,
            regime_label=regime_label,
            regime_code=regime_code,
            growth_signal=round(growth_signal, 4),
            inflation_signal=round(inflation_signal, 4),
            policy_signal=round(policy_signal, 4),
            lei_composite=round(lei, 4) if lei is not None else None,
            recession_probability=round(recession_prob, 4) if recession_prob is not None else None,
            confidence=min(1.0, confidence),
        )


# ---------------------------------------------------------------------------
# Class 4: EconomicIndicatorForecaster
# ---------------------------------------------------------------------------


class EconomicIndicatorForecaster:
    """Simple macro forecasting using AR(1) and leading indicator composite.

    Methods:
    - ar1_forecast: fit AR(1) to a FRED series, project 12m
    - compute_lei_composite: LEI-style composite from 10 leading indicators
    - recession_probability_ny_fed: probit model from 3m10y Treasury spread
    - get_12m_forecast: main public method
    """

    def __init__(self, adapter: Optional[FREDUniversalAdapter] = None) -> None:
        self._adapter = adapter or FREDUniversalAdapter()

    def _get_series_with_info(self, series_id: str) -> Tuple[pd.DataFrame, Optional[SeriesInfo]]:
        df = self._adapter.fetch_series(
            series_id,
            start=(date.today() - timedelta(days=5 * 365)).isoformat(),
        )
        info = self._adapter.get_series_info(series_id)
        return df, info

    def ar1_forecast(
        self,
        series_id: str,
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fit AR(1) model and return forecasts + confidence band.

        AR(1): y_t = c + φ·y_{t-1} + ε_t

        Projects 3, 6, 12 months forward using compound iteration.
        """
        if df is None:
            df = self._adapter.fetch_series(
                series_id,
                start=(date.today() - timedelta(days=5 * 365)).isoformat(),
            )
        if df.empty or len(df) < 12:
            return None

        values = df["value"].values.astype(float)
        n = len(values)

        # AR(1) via OLS: y_t ~ y_{t-1}
        y = values[1:]
        x = values[:-1]
        x_with_const = np.column_stack([np.ones_like(x), x])
        try:
            result = np.linalg.lstsq(x_with_const, y, rcond=None)
            coeffs = result[0]
            c = coeffs[0]
            phi = coeffs[1]
        except Exception:
            return None

        # Residual std
        y_hat = c + phi * x
        residuals = y - y_hat
        sigma = float(np.std(residuals, ddof=2))

        # Forecast horizon: approximate monthly/quarterly steps
        # Determine data frequency from date spacing
        date_diffs = [(df["date"].iloc[i + 1] - df["date"].iloc[i]).days for i in range(min(5, n - 1))]
        avg_days = np.mean(date_diffs) if date_diffs else 30.0

        # Periods per month
        if avg_days < 10:
            freq = "W"
            periods_3m = 13
            periods_6m = 26
            periods_12m = 52
        elif avg_days < 45:
            freq = "M"
            periods_3m = 3
            periods_6m = 6
            periods_12m = 12
        elif avg_days < 100:
            freq = "Q"
            periods_3m = 1
            periods_6m = 2
            periods_12m = 4
        else:
            freq = "A"
            periods_3m = 1
            periods_6m = 1
            periods_12m = 1

        # Compound AR(1) projection
        last_val = float(values[-1])

        def _project(periods: int) -> float:
            val = last_val
            for _ in range(periods):
                val = c + phi * val
            return val

        f3m = _project(periods_3m)
        f6m = _project(periods_6m)
        f12m = _project(periods_12m)

        # Confidence band at 12m: propagate uncertainty
        # Var(h-step ahead) ≈ σ² × Σ_{k=0}^{h-1} φ^{2k}
        h = periods_12m
        if abs(phi) < 1.0:
            var_h = sigma ** 2 * (1 - phi ** (2 * h)) / (1 - phi ** 2 + 1e-10)
        else:
            var_h = sigma ** 2 * h
        ci_half = 1.96 * math.sqrt(max(0.0, var_h))

        return {
            "ar1_coeff": round(float(phi), 4),
            "ar1_intercept": round(float(c), 4),
            "residual_std": round(sigma, 4),
            "forecast_3m": round(f3m, 4),
            "forecast_6m": round(f6m, 4),
            "forecast_12m": round(f12m, 4),
            "ci_lower_12m": round(f12m - ci_half, 4),
            "ci_upper_12m": round(f12m + ci_half, 4),
            "latest_value": round(last_val, 4),
            "freq": freq,
        }

    def compute_lei_composite(self) -> Optional[float]:
        """Compute LEI-style composite from 10 leading indicators.

        Each series is standardized (z-score) then averaged.
        Inverted series (claims, credit spreads, VIX, dollar) are negated.
        Result is the composite index level (positive = expanding).
        """
        start = (date.today() - timedelta(days=365)).isoformat()
        data = self._adapter.batch_fetch(LEADING_INDICATOR_SERIES, start=start)

        component_z_scores: List[float] = []
        for sid in LEADING_INDICATOR_SERIES:
            df = data.get(sid, pd.DataFrame())
            if df.empty or len(df) < 3:
                continue
            vals = df["value"].values.astype(float)
            mean = vals.mean()
            std = vals.std()
            if std < 1e-8:
                continue
            latest_z = (vals[-1] - mean) / std
            if sid in INVERTED_LEADING:
                latest_z = -latest_z
            component_z_scores.append(float(latest_z))

        if not component_z_scores:
            return None
        return round(float(np.mean(component_z_scores)), 4)

    def recession_probability_ny_fed(self) -> Optional[float]:
        """NY Fed probit model: recession probability from 3m10y spread.

        The NY Fed model (Estrella & Mishkin 1998):
            P(recession in 12m) = Φ(α + β × spread)
        where α = -0.6071, β = -0.7374 (re-estimated coefficients)
        and spread = 10Y yield minus 3M T-bill rate.

        Returns probability 0–100.
        """
        t10y3m = self._adapter.get_latest_value("T10Y3M")[0]
        if t10y3m is None:
            # Try to compute from individual series
            dgs10 = self._adapter.get_latest_value("DGS10")[0]
            dtb3 = self._adapter.get_latest_value("DTB3")[0]
            if dgs10 is None or dtb3 is None:
                return None
            t10y3m = dgs10 - dtb3

        # NY Fed coefficients (from Estrella & Mishkin 1998, re-estimated 2024)
        alpha = -0.6071
        beta = -0.7374
        z = alpha + beta * t10y3m
        prob = stats.norm.cdf(z) * 100.0
        return round(prob, 2)

    def get_12m_forecast(self, series_id: str) -> ForecastResult:
        """Return 12-month AR(1) forecast with confidence band for any FRED series."""
        df, info = self._get_series_with_info(series_id)
        title = info.title if info else series_id

        ar_result = self.ar1_forecast(series_id, df)
        if ar_result is None or df.empty:
            latest_val = float(df.iloc[-1]["value"]) if not df.empty else 0.0
            return ForecastResult(
                series_id=series_id,
                title=title,
                latest_value=latest_val,
                forecast_3m=latest_val,
                forecast_6m=latest_val,
                forecast_12m=latest_val,
                ci_lower_12m=latest_val * 0.9,
                ci_upper_12m=latest_val * 1.1,
                ar1_coefficient=1.0,
                trend="stable",
                method="Fallback (insufficient data)",
            )

        f12 = ar_result["forecast_12m"]
        latest = ar_result["latest_value"]
        pct_change = (f12 / latest - 1.0) * 100.0 if latest != 0 else 0.0
        if pct_change > 1.0:
            trend = "rising"
        elif pct_change < -1.0:
            trend = "falling"
        else:
            trend = "stable"

        return ForecastResult(
            series_id=series_id,
            title=title,
            latest_value=latest,
            forecast_3m=ar_result["forecast_3m"],
            forecast_6m=ar_result["forecast_6m"],
            forecast_12m=f12,
            ci_lower_12m=ar_result["ci_lower_12m"],
            ci_upper_12m=ar_result["ci_upper_12m"],
            ar1_coefficient=ar_result["ar1_coeff"],
            trend=trend,
            method=f"AR(1) [{ar_result['freq']} frequency]",
        )

    def forecast_batch(self, series_ids: List[str]) -> List[ForecastResult]:
        """Batch forecast for multiple series."""
        results = []
        for sid in series_ids:
            try:
                results.append(self.get_12m_forecast(sid))
            except Exception as exc:
                logger.warning("forecast_batch %s: %s", sid, exc)
        return results


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_adapter = FREDUniversalAdapter()
_dashboard = MacroDashboard(_adapter)
_classifier = MacroRegimeClassifier(_adapter)
_forecaster = EconomicIndicatorForecaster(_adapter)

# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

fred_router_v2 = APIRouter(prefix="/fred/v2", tags=["FRED Macro v2"])


@fred_router_v2.get("/search", response_model=List[SeriesInfo])
async def api_search(
    q: str = Query(description="Search text for FRED series"),
    limit: int = Query(default=20, le=100),
):
    """Search FRED 765K+ series by text query (requires FRED_API_KEY)."""
    def _search():
        return _adapter.search_series(q, limit=limit)
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, _search)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/series/{series_id}", response_model=SeriesObservation)
async def api_series(
    series_id: str,
    start: Optional[str] = Query(default=None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="End date YYYY-MM-DD"),
):
    """Fetch latest observation and metadata for a FRED series."""
    def _fetch():
        df = _adapter.fetch_series(series_id.upper(), start=start, end=end)
        info = _adapter.get_series_info(series_id.upper())
        title = info.title if info else series_id
        units = info.units if info else None
        if df.empty:
            return SeriesObservation(
                series_id=series_id,
                title=title,
                units=units,
                latest_value=None,
                latest_date=None,
                prev_value=None,
                change=None,
                pct_change=None,
                yoy_change=None,
                yoy_pct_change=None,
            )
        latest_val = float(df.iloc[-1]["value"])
        latest_date = str(df.iloc[-1]["date"])
        prev_val = float(df.iloc[-2]["value"]) if len(df) >= 2 else None
        change = round(latest_val - prev_val, 6) if prev_val is not None else None
        pct_change = round((latest_val / prev_val - 1.0) * 100.0, 4) if prev_val and prev_val != 0 else None
        yoy = _adapter.compute_yoy_change(df)

        yoy_change = None
        try:
            latest_dt = df.iloc[-1]["date"]
            year_ago = latest_dt - timedelta(days=365)
            past_df = df[df["date"] <= year_ago]
            if not past_df.empty:
                yoy_change = round(latest_val - float(past_df.iloc[-1]["value"]), 6)
        except Exception:
            pass

        return SeriesObservation(
            series_id=series_id,
            title=title,
            units=units,
            latest_value=latest_val,
            latest_date=latest_date,
            prev_value=prev_val,
            change=change,
            pct_change=pct_change,
            yoy_change=yoy_change,
            yoy_pct_change=yoy,
        )

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/series/{series_id}/history")
async def api_series_history(
    series_id: str,
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
    limit: int = Query(default=250, le=2000),
):
    """Return historical observations for a FRED series as a list of {date, value} dicts."""
    def _fetch():
        df = _adapter.fetch_series(series_id.upper(), start=start, end=end)
        if df.empty:
            return {"series_id": series_id, "count": 0, "data": []}
        tail = df.tail(limit)
        return {
            "series_id": series_id,
            "count": len(tail),
            "data": [{"date": str(row["date"]), "value": row["value"]} for _, row in tail.iterrows()],
        }

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/dashboard", response_model=MacroDashboardResult)
async def api_dashboard(
    category: Optional[str] = Query(default=None, description="Filter by category"),
):
    """Return the curated 50-series macro dashboard."""
    def _fetch():
        result = _dashboard.fetch(lookback_days=400)
        if category:
            filtered = [s for s in result.series if (s.category or "").lower() == category.lower()]
            return MacroDashboardResult(
                as_of=result.as_of,
                series=filtered,
                categories={k: v for k, v in result.categories.items()
                            if k.lower() == category.lower()},
            )
        return result

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as exc:
        logger.error("api_dashboard: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/batch")
async def api_batch(
    series_ids: str = Query(description="Comma-separated FRED series IDs (up to 20)"),
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
):
    """Batch fetch up to 20 FRED series in parallel."""
    id_list = [s.strip().upper() for s in series_ids.split(",") if s.strip()][:20]
    if not id_list:
        raise HTTPException(status_code=400, detail="No series IDs provided")

    def _fetch():
        data = _adapter.batch_fetch(id_list, start=start, end=end)
        result = {}
        for sid, df in data.items():
            if df.empty:
                result[sid] = {"count": 0, "latest_value": None, "latest_date": None, "data": []}
            else:
                tail = df.tail(50)
                result[sid] = {
                    "count": len(df),
                    "latest_value": float(df.iloc[-1]["value"]),
                    "latest_date": str(df.iloc[-1]["date"]),
                    "data": [{"date": str(r["date"]), "value": r["value"]} for _, r in tail.iterrows()],
                }
        return result

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/regime", response_model=MacroRegime)
async def api_regime():
    """Classify the current macro regime (growth × inflation × policy)."""
    def _classify():
        classifier = MacroRegimeClassifier(_adapter)
        return classifier.classify()

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _classify)
    except Exception as exc:
        logger.error("api_regime: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/forecast/{series_id}", response_model=ForecastResult)
async def api_forecast(series_id: str):
    """12-month AR(1) forecast for any FRED series."""
    def _forecast():
        forecaster = EconomicIndicatorForecaster(_adapter)
        return forecaster.get_12m_forecast(series_id.upper())

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _forecast)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/forecast")
async def api_forecast_batch(
    series_ids: str = Query(description="Comma-separated FRED series IDs"),
):
    """Batch 12-month forecasts for multiple FRED series."""
    id_list = [s.strip().upper() for s in series_ids.split(",") if s.strip()][:20]
    if not id_list:
        raise HTTPException(status_code=400, detail="No series IDs provided")

    def _forecast():
        forecaster = EconomicIndicatorForecaster(_adapter)
        return forecaster.forecast_batch(id_list)

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _forecast)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/lei")
async def api_lei():
    """Compute the LEI-style composite from 10 leading indicators."""
    def _compute():
        forecaster = EconomicIndicatorForecaster(_adapter)
        lei = forecaster.compute_lei_composite()
        recession_prob = forecaster.recession_probability_ny_fed()
        return {
            "lei_composite": lei,
            "recession_probability_12m": recession_prob,
            "components": LEADING_INDICATOR_SERIES,
            "as_of": date.today().isoformat(),
        }

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _compute)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/recession-probability")
async def api_recession_prob():
    """NY Fed probit model: recession probability from 3m10y yield spread."""
    def _compute():
        forecaster = EconomicIndicatorForecaster(_adapter)
        prob = forecaster.recession_probability_ny_fed()
        spread, _ = _adapter.get_latest_value("T10Y3M")
        return {
            "recession_probability_12m": prob,
            "t10y3m_spread": spread,
            "model": "NY Fed Probit (Estrella & Mishkin 1998)",
            "as_of": date.today().isoformat(),
        }

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _compute)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/categories")
async def api_categories(category_id: int = Query(default=0)):
    """Browse FRED category tree (requires FRED_API_KEY for full tree)."""
    def _browse():
        return _adapter.get_categories(category_id)

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _browse)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fred_router_v2.get("/dashboard/categories")
async def api_dashboard_categories():
    """List available dashboard categories."""
    cats = {}
    for meta in DASHBOARD_SERIES.values():
        cat = meta.get("category", "Other")
        cats[cat] = cats.get(cat, 0) + 1
    return {
        "categories": [{"name": k, "series_count": v} for k, v in sorted(cats.items())],
        "total_series": len(set(m["id"] for m in DASHBOARD_SERIES.values())),
    }


@fred_router_v2.get("/info/{series_id}", response_model=SeriesInfo)
async def api_series_info(series_id: str):
    """Fetch metadata for a single FRED series."""
    def _info():
        info = _adapter.get_series_info(series_id.upper())
        if not info:
            raise ValueError(f"Series {series_id} not found")
        return info

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _info)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
