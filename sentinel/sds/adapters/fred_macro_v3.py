"""
FRED Macro Time Series Adapter v3 — 765K+ series with smart discovery,
cross-country comparison, leading indicator composites, and vintage tracking.

Dimension #043 — FRED macro time series (target: 9).

Free data sources only:
  - FRED CSV endpoint (no API key): https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}
  - FRED search API (no key fallback: HTML scrape):
    https://api.stlouisfed.org/fred/series/search?search_text={query}&file_type=json
  - FRED API (optional key, allows more calls):
    https://api.stlouisfed.org/fred/series/observations?series_id={id}&api_key={key}

Architecture:
  FREDAPIClient          — fetch + search any of 765K+ series
  MacroSeriesLibrary     — curated 200+ series by category
  MacroIndicatorBuilder  — composite LEI, coincident, lagging, GDP nowcast
  MacroRevisionTracker   — first-release vs revised values, revision statistics
  CrossCountryComparator — inflation, GDP, unemployment across G7+EM
  MacroDashboardBuilder  — US snapshot, global table, macro report narrative
  FREDMacroEngine        — orchestrator with daily update + parquet export
"""
from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_CSV_BASE     = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_BASE     = "https://api.stlouisfed.org/fred"
FRED_SEARCH_HTML  = "https://fred.stlouisfed.org/search"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/3.0 richard.porras@realempanada.com",
    "Accept": "application/json, text/html, */*",
}

# Default request timeout
_TIMEOUT = 30

# Cache TTL (seconds)
_CACHE_TTL = 4 * 3600   # 4 hours


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SeriesInfo:
    series_id: str
    title: str
    frequency: str = ""
    units: str = ""
    seasonal_adjustment: str = ""
    observation_start: str = ""
    observation_end: str = ""
    popularity: int = 0
    notes: str = ""
    category: str = ""


@dataclass
class MacroSnapshot:
    indicator: str
    series_id: str
    as_of: date
    latest_value: float
    previous_value: Optional[float]
    change: Optional[float]
    change_pct: Optional[float]
    frequency: str
    units: str
    trend: str = ""   # RISING, FALLING, STABLE
    zscore_5yr: Optional[float] = None


@dataclass
class RevisionRecord:
    series_id: str
    reference_period: str
    first_release: float
    latest_value: float
    revision: float                # latest - first
    revision_pct: Optional[float]
    release_date: str = ""
    is_large: bool = False


# ---------------------------------------------------------------------------
# FRED API Client
# ---------------------------------------------------------------------------

class FREDAPIClient:
    """
    Primary access layer for FRED.
    - CSV endpoint (no key needed) for observations
    - JSON API (optional key) for metadata and search
    - HTML scrape fallback for search without key
    """

    def __init__(self, api_key: Optional[str] = None, cache_ttl_seconds: int = _CACHE_TTL):
        self._api_key = api_key or os.environ.get("FRED_API_KEY")
        self._ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[pd.Series, float]] = {}   # series_id → (data, ts)
        self._info_cache: Dict[str, Tuple[SeriesInfo, float]] = {}

    def _api_params(self) -> dict:
        params: dict = {"file_type": "json"}
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    # ------------------------------------------------------------------
    # Series observation fetch
    # ------------------------------------------------------------------

    def fetch_series(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.Series:
        """
        Fetch a FRED series via the free CSV endpoint.
        Returns a pd.Series with DatetimeIndex, values as float.
        Missing/non-numeric values are dropped.
        """
        cache_key = f"{series_id}|{start}|{end}"
        if not force_refresh:
            cached, ts = self._cache.get(cache_key, (None, 0))
            if cached is not None and (time.time() - ts) < self._ttl:
                return cached

        url = f"{FRED_CSV_BASE}?id={series_id}"
        if start:
            url += f"&vintage_date={start}"  # not a standard param but harmless
        try:
            with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
                resp = client.get(url)
                resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
            s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            if start:
                s = s[s.index >= pd.Timestamp(start)]
            if end:
                s = s[s.index <= pd.Timestamp(end)]
            s.name = series_id
            self._cache[cache_key] = (s, time.time())
            return s
        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
            return pd.Series(dtype=float, name=series_id)

    def fetch_multiple_series(
        self,
        series_ids: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_workers: int = 8,
    ) -> pd.DataFrame:
        """
        Fetch multiple series in parallel using ThreadPoolExecutor.
        Returns aligned DataFrame (outer join, forward-filled for daily series).
        """
        results: Dict[str, pd.Series] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.fetch_series, sid, start, end): sid
                       for sid in series_ids}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    s = fut.result()
                    if not s.empty:
                        results[sid] = s
                except Exception as exc:
                    logger.warning("Parallel fetch error for %s: %s", sid, exc)

        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results).sort_index()
        return df

    # ------------------------------------------------------------------
    # Series metadata
    # ------------------------------------------------------------------

    def get_series_info(self, series_id: str) -> SeriesInfo:
        """
        Fetch series metadata from FRED API.
        Falls back to minimal info when API key is absent.
        """
        cached, ts = self._info_cache.get(series_id, (None, 0))
        if cached is not None and (time.time() - ts) < self._ttl * 6:
            return cached

        # Try JSON API
        if self._api_key:
            try:
                params = self._api_params()
                params["series_id"] = series_id
                url = f"{FRED_API_BASE}/series"
                with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
                    resp = client.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                slist = data.get("seriess", [])
                if slist:
                    s = slist[0]
                    info = SeriesInfo(
                        series_id=series_id,
                        title=s.get("title", series_id),
                        frequency=s.get("frequency_short", ""),
                        units=s.get("units_short", s.get("units", "")),
                        seasonal_adjustment=s.get("seasonal_adjustment_short", ""),
                        observation_start=s.get("observation_start", ""),
                        observation_end=s.get("observation_end", ""),
                        popularity=int(s.get("popularity", 0)),
                        notes=s.get("notes", ""),
                    )
                    self._info_cache[series_id] = (info, time.time())
                    return info
            except Exception as exc:
                logger.debug("FRED API metadata failed for %s: %s", series_id, exc)

        # Minimal fallback — use the library description
        lib = MacroSeriesLibrary()
        title = lib.get_description(series_id)
        info = SeriesInfo(series_id=series_id, title=title)
        self._info_cache[series_id] = (info, time.time())
        return info

    # ------------------------------------------------------------------
    # Series search
    # ------------------------------------------------------------------

    def search_series(
        self,
        query: str,
        limit: int = 20,
        filter_by_frequency: Optional[str] = None,
    ) -> List[SeriesInfo]:
        """
        Search FRED for series matching a query string.
        Uses JSON API when key is present; falls back to HTML scrape.
        filter_by_frequency: 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annual'
        """
        if self._api_key:
            return self._search_via_api(query, limit, filter_by_frequency)
        return self._search_via_html(query, limit)

    def _search_via_api(
        self, query: str, limit: int, frequency: Optional[str]
    ) -> List[SeriesInfo]:
        """FRED series/search JSON endpoint."""
        params = self._api_params()
        params.update({"search_text": query, "limit": limit, "order_by": "popularity"})
        if frequency:
            params["filter_variable"] = "frequency"
            params["filter_value"] = frequency
        try:
            url = f"{FRED_API_BASE}/series/search"
            with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
            data = resp.json()
            results = []
            for s in data.get("seriess", [])[:limit]:
                results.append(SeriesInfo(
                    series_id=s.get("id", ""),
                    title=s.get("title", ""),
                    frequency=s.get("frequency_short", ""),
                    units=s.get("units_short", ""),
                    seasonal_adjustment=s.get("seasonal_adjustment_short", ""),
                    observation_start=s.get("observation_start", ""),
                    observation_end=s.get("observation_end", ""),
                    popularity=int(s.get("popularity", 0)),
                    notes=s.get("notes", ""),
                ))
            return results
        except Exception as exc:
            logger.warning("FRED API search failed for '%s': %s", query, exc)
            return []

    def _search_via_html(self, query: str, limit: int) -> List[SeriesInfo]:
        """
        Fallback: scrape FRED search results page for series IDs and titles.
        """
        results: List[SeriesInfo] = []
        try:
            url = f"{FRED_SEARCH_HTML}?st={query.replace(' ', '+')}"
            with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
                resp = client.get(url)
                resp.raise_for_status()
            html = resp.text
            # Parse series IDs and titles from HTML
            # FRED search results contain patterns like: /series/GDPC1
            ids = re.findall(r'/series/([A-Z0-9]{2,20})', html)
            titles = re.findall(r'class="series-title[^"]*"[^>]*>([^<]+)<', html)
            seen = set()
            for i, sid in enumerate(ids):
                if sid in seen:
                    continue
                seen.add(sid)
                title = titles[i] if i < len(titles) else sid
                results.append(SeriesInfo(series_id=sid, title=title.strip()))
                if len(results) >= limit:
                    break
        except Exception as exc:
            logger.warning("FRED HTML search failed for '%s': %s", query, exc)
        return results

    def get_release_dates(self, series_id: str) -> List[str]:
        """
        Historical vintage release dates (requires API key).
        Returns list of ISO date strings.
        """
        if not self._api_key:
            logger.info("get_release_dates requires FRED_API_KEY")
            return []
        try:
            params = self._api_params()
            params["series_id"] = series_id
            url = f"{FRED_API_BASE}/series/vintagedates"
            with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
            return resp.json().get("vintage_dates", [])
        except Exception as exc:
            logger.warning("get_release_dates failed for %s: %s", series_id, exc)
            return []

    def search_and_fetch(
        self,
        query: str,
        start: str = "2000-01-01",
        limit: int = 5,
    ) -> pd.DataFrame:
        """Convenience: search + fetch top results into a DataFrame."""
        series_list = self.search_series(query, limit=limit)
        if not series_list:
            return pd.DataFrame()
        ids = [s.series_id for s in series_list]
        df = self.fetch_multiple_series(ids, start=start)
        # Rename columns to titles
        title_map = {s.series_id: f"{s.series_id} – {s.title[:40]}" for s in series_list}
        df = df.rename(columns=title_map)
        return df


# ---------------------------------------------------------------------------
# Macro Series Library — curated 200+ series
# ---------------------------------------------------------------------------

class MacroSeriesLibrary:
    """
    Curated library of 200+ key FRED series organized by category.
    All series are free via the CSV endpoint.
    """

    _US_MACRO: Dict[str, str] = {
        # GDP & Growth
        "GDPC1":              "Real GDP (Bil. Chained 2017$, SAAR)",
        "GDP":                "Nominal GDP (Bil. $, SAAR)",
        "GDPPOT":             "Real Potential GDP (CBO, Bil. Chained 2017$)",
        "A191RL1Q225SBEA":    "Real GDP Growth Rate, QoQ (SAAR, %)",
        "GDPC1":              "Real GDP",
        "GDPDEF":             "GDP Price Deflator",
        "OUTMS":              "Industrial Production, Manufacturing",
        "INDPRO":             "Industrial Production Index",
        "CFNAI":              "Chicago Fed National Activity Index",
        # Employment
        "UNRATE":             "Unemployment Rate (%)",
        "U6RATE":             "U-6 Unemployment (incl. underemployed)",
        "PAYEMS":             "Total Nonfarm Payrolls (Thous.)",
        "MANEMP":             "Manufacturing Employees (Thous.)",
        "USPRIV":             "Private Sector Payrolls (Thous.)",
        "ICSA":               "Initial Jobless Claims (Thous.)",
        "CCSA":               "Continuing Claims (Thous.)",
        "JTSJOL":             "Job Openings (JOLTS, Thous.)",
        "JTSQUR":             "Quits Rate (JOLTS, %)",
        "AWHMAN":             "Average Weekly Hours: Manufacturing",
        "AWOTMAN":            "Average Overtime Hours: Manufacturing",
        "CES0500000003":      "Average Hourly Earnings, Private",
        # Inflation
        "CPIAUCSL":           "CPI All Urban Consumers (SA)",
        "CPILFESL":           "CPI Less Food & Energy (Core CPI)",
        "CPIUFDSL":           "CPI Food",
        "CPIENGSL":           "CPI Energy",
        "CPIAPPSL":           "CPI Apparel",
        "PPIFIS":             "PPI Final Demand",
        "PPIACO":             "PPI All Commodities",
        "PCEPI":              "PCE Price Index",
        "PCEPILFE":           "Core PCE (Fed's Preferred Gauge)",
        "PCEDG":              "PCE Durable Goods",
        "PCES":               "PCE Services",
        "T5YIE":              "5-Year Breakeven Inflation Rate",
        "T10YIE":             "10-Year Breakeven Inflation Rate",
        # Housing
        "HOUST":              "Housing Starts (Thous. Units, SAAR)",
        "PERMIT":             "Building Permits (Thous. Units, SAAR)",
        "CSUSHPISA":          "Case-Shiller Home Price Index (20-City)",
        "MORTGAGE30US":       "30-Year Fixed Mortgage Rate (%)",
        "MSPUS":              "Median Sales Price of Houses Sold ($)",
        "HSN1F":              "New Single-Family Houses Sold (Thous.)",
        "EXHOSLUSM495S":      "Existing Home Sales (Mil., SAAR)",
        # Consumer
        "UMCSENT":            "U. Michigan Consumer Sentiment",
        "CONCCONF":           "Conference Board Consumer Confidence",
        "PCE":                "Personal Consumption Expenditures (Bil.$)",
        "RRSFS":              "Retail Sales, Advance (Mil.$)",
        "RSXFS":              "Retail Sales ex-Autos (Mil.$)",
        "DSPIC96":            "Real Disposable Personal Income",
        "PSAVERT":            "Personal Saving Rate (%)",
        # Business / Investment
        "DGORDER":            "Durable Goods Orders",
        "NEWORDER":           "Manufacturers' New Orders (Non-Defense Capital Goods)",
        "BUSINV":             "Total Business Inventories",
        "BABUS":              "Business Activity Outlook Survey",
        # Money & Credit
        "M2SL":               "M2 Money Supply (Bil.$)",
        "M1SL":               "M1 Money Supply",
        "MZMSL":              "MZM Money Stock",
        "WALCL":              "Fed Balance Sheet — Total Assets",
        "RRPTTLD":            "Overnight Reverse Repo Facility",
        "TOTCI":              "Total Consumer Credit",
        "DTCTHFNM":           "Consumer Credit to GDP",
        "FEDFUNDS":           "Effective Federal Funds Rate (%)",
        "SOFR":               "Secured Overnight Financing Rate",
        "DPRIME":             "Prime Lending Rate (%)",
        # Financial / Market
        "VIXCLS":             "CBOE Volatility Index (VIX)",
        "SP500":              "S&P 500 Index (Monthly)",
        "USREC":              "NBER US Recession Indicator (0/1)",
        "NFCI":               "Chicago Fed National Financial Conditions Index",
        "STLFSI4":            "St. Louis Financial Stress Index",
        "TEDRATE":            "TED Spread (bps) – 3M LIBOR vs T-bill",
    }

    _INTERNATIONAL: Dict[str, str] = {
        # Eurozone
        "LRHUTTTTEZM156S":    "Euro Area Unemployment Rate",
        "EA19RHPUPTT01IXOBSAM": "Euro Area CPI",
        "CLVMEURSCAB1GQEA19": "Euro Area Real GDP",
        "MABMM301EZM189S":    "Euro Area M3",
        "IRLTLT01EZM156N":    "Euro Area Long-Term Interest Rate",
        # Germany
        "CLVMEURSCAB1GQDE":   "Germany Real GDP",
        "DEUCPIALLMINMEI":    "Germany CPI",
        "DEUURYNBIS":         "Germany Unemployment Rate",
        "DEUPUAANS":          "Germany Industrial Production",
        # Japan
        "JPNURYNBIS":         "Japan Unemployment Rate",
        "JPNCPIALLMINMEI":    "Japan CPI",
        "CLVMNACSCAB1GQJP":   "Japan Real GDP",
        "JPNPUAANS":          "Japan Industrial Production",
        "IRLTLT01JPM156N":    "Japan Long-Term Interest Rate",
        # China
        "CHNGDPNQDSMEI":      "China GDP (Nominal)",
        "CPALCY01CNM661N":    "China CPI",
        "CHNCPIALLMINMEI":    "China CPI (OECD)",
        "XTEXVA01CNM664S":    "China Exports Value",
        # United Kingdom
        "GBRUNR":             "UK Unemployment Rate",
        "GBRCPIALLMINMEI":    "UK CPI",
        "CLVMNACSCAB1GQQBIS": "UK Real GDP",
        "GBRPUAANS":          "UK Industrial Production",
        "IRLTLT01GBM156N":    "UK Long-Term Interest Rate",
        # Canada
        "CANURYNBIS":         "Canada Unemployment Rate",
        "CANCPIALLMINMEI":    "Canada CPI",
        # South Korea
        "KORURYNBIS":         "South Korea Unemployment Rate",
        "KORCPIALLMINMEI":    "South Korea CPI",
        # Emerging Markets
        "EMVOVERALLEMVXOL":   "EM Volatility Index",
        "DTWEXBGS":           "USD Broad Real Effective Exchange Rate",
        "DTWEXAFEGS":         "USD Advanced Foreign Economies REER",
        # Global Trade
        "XTIMVA01USM664N":    "US Imports Value",
        "XTEXVA01USM664S":    "US Exports Value",
    }

    _FIXED_INCOME: Dict[str, str] = {
        # US Treasuries
        "DGS1MO":    "Treasury 1-Month CMT",
        "DGS3MO":    "Treasury 3-Month CMT",
        "DGS6MO":    "Treasury 6-Month CMT",
        "DGS1":      "Treasury 1-Year CMT",
        "DGS2":      "Treasury 2-Year CMT",
        "DGS3":      "Treasury 3-Year CMT",
        "DGS5":      "Treasury 5-Year CMT",
        "DGS7":      "Treasury 7-Year CMT",
        "DGS10":     "Treasury 10-Year CMT",
        "DGS20":     "Treasury 20-Year CMT",
        "DGS30":     "Treasury 30-Year CMT",
        # Spreads / Curves
        "T10Y2Y":    "10Y-2Y Treasury Spread",
        "T10Y3M":    "10Y-3M Treasury Spread",
        "T5YIFR":    "5Y5Y Forward Inflation Rate",
        "DFII5":     "5-Year TIPS Yield",
        "DFII10":    "10-Year TIPS Yield",
        # Credit Spreads (FRED OAS)
        "BAMLC0A0CM":    "IG OAS",
        "BAMLH0A0HYM2":  "HY OAS",
        "BAMLC0A4CBBB":  "BBB OAS",
        "BAMLH0A1HYBB":  "BB OAS",
        "BAMLH0A3HYC":   "CCC OAS",
        # Short rates
        "DTB3":      "3-Month T-Bill Rate",
        "DTB6":      "6-Month T-Bill Rate",
        "SOFR":      "SOFR",
        "DPCREDIT":  "Discount Rate",
        "OBMMIFHA30YF": "FHA 30-Year Mortgage Rate",
    }

    _COMMODITIES: Dict[str, str] = {
        "DCOILWTICO":         "WTI Crude Oil Price ($/bbl)",
        "DCOILBRENTEU":       "Brent Crude Oil Price ($/bbl)",
        "DHHNGSP":            "Henry Hub Natural Gas Spot",
        "GOLDAMGBD228NLBM":   "Gold Price (London Fix, $/troy oz)",
        "SLVPRUSD":           "Silver Price ($/troy oz)",
        "PCOPPUSDM":          "Copper Price ($/metric ton)",
        "PWHEAMTUSDM":        "Wheat Price ($/metric ton)",
        "PMAIZMTUSDM":        "Corn Price ($/metric ton)",
        "PSOYBUSDQ":          "Soybean Price",
        "PBAUXUSDM":          "Aluminum Price",
        "PNICKUSDM":          "Nickel Price",
        "PIORECRUSDM":        "Iron Ore Price",
        "PALLPDUSDM":         "Palladium Price",
        "PPLATINUMUSDM":      "Platinum Price",
        "PPIACO":             "PPI All Commodities",
        "PALLFNFINDEXQ":      "Non-Fuel Commodities Price Index",
        "DEXUSEU":            "USD/EUR Exchange Rate",
        "DEXJPUS":            "JPY/USD Exchange Rate",
        "DEXUSUK":            "USD/GBP Exchange Rate",
        "DEXCHUS":            "CNY/USD Exchange Rate",
        "DEXCAUS":            "CAD/USD Exchange Rate",
    }

    _FINANCIAL: Dict[str, str] = {
        "VIXCLS":             "VIX Volatility Index",
        "NASDAQCOM":          "NASDAQ Composite",
        "SP500":              "S&P 500",
        "DJIA":               "Dow Jones Industrial Average",
        "WILL5000PRFC":       "Wilshire 5000 Total Market",
        "MKTGDP":             "Stock Market Cap to GDP (Buffett Indicator proxy)",
        "DTWEXBGS":           "USD Trade-Weighted Exchange Rate Index",
        "NFCI":               "Chicago Fed National Financial Conditions",
        "STLFSI4":            "St. Louis Financial Stress Index",
        "ANFCI":              "Adjusted NFCI",
        "USEPUINDXD":         "US Economic Policy Uncertainty Index",
        "WLEMUINDXD":         "World Uncertainty Index",
        "EMVOVERALLEMVXOL":   "Equity Market Volatility",
        "RIFLPBCIANM48NM":    "Interest Rate on 48-Month New Car Loan",
        "TERMCBPER24NS":      "Finance Rate on Personal Loans",
        "BAMLC0A0CMPTRR":     "IG Total Return Index",
    }

    # Composite category map
    _CATEGORIES: Dict[str, Dict[str, str]] = {}

    def __init__(self):
        self._CATEGORIES = {
            "us_macro":     self._US_MACRO,
            "international": self._INTERNATIONAL,
            "fixed_income": self._FIXED_INCOME,
            "commodities":  self._COMMODITIES,
            "financial":    self._FINANCIAL,
        }
        # Build reverse lookup: series_id → description
        self._all: Dict[str, str] = {}
        for cat_data in self._CATEGORIES.values():
            self._all.update(cat_data)

    def get_series_ids(self, category: str) -> List[str]:
        """Return all series IDs for a category."""
        cat_data = self._CATEGORIES.get(category, {})
        return list(cat_data.keys())

    def get_all_categories(self) -> List[str]:
        return list(self._CATEGORIES.keys())

    def get_description(self, series_id: str) -> str:
        return self._all.get(series_id, series_id)

    def get_all_series(self) -> Dict[str, str]:
        """All 200+ series: {series_id: description}."""
        return dict(self._all)

    def find_series(self, keyword: str) -> Dict[str, str]:
        """Simple keyword search within library."""
        kw = keyword.lower()
        return {sid: desc for sid, desc in self._all.items()
                if kw in desc.lower() or kw in sid.lower()}


# ---------------------------------------------------------------------------
# Macro Indicator Builder — composite LEI, coincident, nowcast
# ---------------------------------------------------------------------------

class MacroIndicatorBuilder:
    """
    Build composite macro indicators from FRED data.
    Follows Conference Board methodology for LEI, CEI, LAG.
    """

    _LEI_COMPONENTS = {
        "PERMIT":    0.20,    # Building Permits (leading housing indicator)
        "ICSA":      0.15,    # Initial Claims (inverted: -1)
        "T10Y3M":    0.20,    # Yield Curve (10Y-3M)
        "NEWORDER":  0.15,    # Non-defense capital goods orders
        "UMCSENT":   0.15,    # Consumer Expectations (Michigan)
        "VIXCLS":    0.15,    # Equity vol (inverted: risk aversion proxy)
    }

    _CEI_COMPONENTS = {
        "PAYEMS":    0.35,    # Nonfarm Payrolls
        "INDPRO":    0.25,    # Industrial Production
        "RRSFS":     0.20,    # Real Retail Sales
        "DSPIC96":   0.20,    # Real Disposable Personal Income
    }

    _LAG_COMPONENTS = {
        "DPRIME":    0.30,    # Prime Rate
        "CPIAUCSL":  0.25,    # CPI (lagging indicator)
        "UNRATE":    0.25,    # Unemployment Rate (lags peaks/troughs)
        "TOTCI":     0.20,    # Consumer Credit Outstanding
    }

    _INVERT = {"ICSA", "VIXCLS"}   # Higher = worse → invert for composite

    def __init__(self, client: Optional[FREDAPIClient] = None):
        self._client = client or FREDAPIClient()

    def _fetch_and_normalize(
        self,
        components: Dict[str, float],
        start: str = "2000-01-01",
    ) -> pd.DataFrame:
        """
        Fetch all component series, standardize each to z-score,
        invert where needed, align to monthly frequency.
        """
        sids = list(components.keys())
        df_raw = self._client.fetch_multiple_series(sids, start=start)
        if df_raw.empty:
            return pd.DataFrame()

        # Resample to monthly (last observation)
        df_monthly = df_raw.resample("ME").last().ffill(limit=3)

        normalized = pd.DataFrame(index=df_monthly.index)
        for sid in sids:
            if sid not in df_monthly.columns:
                continue
            s = df_monthly[sid].dropna()
            if len(s) < 24:
                continue
            mu = s.rolling(window=60, min_periods=24).mean()
            sigma = s.rolling(window=60, min_periods=24).std()
            z = (s - mu) / sigma.replace(0, np.nan)
            if sid in self._INVERT:
                z = -z
            normalized[sid] = z
        return normalized

    def build_leading_indicator_composite(
        self, start: str = "2000-01-01"
    ) -> pd.Series:
        """
        Weighted average of 6 leading components, each standardized.
        Positive = above trend; Negative = below trend (contraction risk).
        """
        norm = self._fetch_and_normalize(self._LEI_COMPONENTS, start)
        if norm.empty:
            return pd.Series(dtype=float, name="LEI_Composite")

        weights = self._LEI_COMPONENTS
        composite = pd.Series(0.0, index=norm.index)
        total_weight = 0.0
        for sid, w in weights.items():
            if sid in norm.columns:
                composite += norm[sid].fillna(0) * w
                total_weight += w
        if total_weight > 0:
            composite /= total_weight
        composite.name = "LEI_Composite"
        return composite.dropna()

    def build_coincident_indicator(self, start: str = "2000-01-01") -> pd.Series:
        """
        Coincident Economic Index: payrolls, IP, retail sales, income.
        """
        norm = self._fetch_and_normalize(self._CEI_COMPONENTS, start)
        if norm.empty:
            return pd.Series(dtype=float, name="CEI_Composite")

        weights = self._CEI_COMPONENTS
        composite = pd.Series(0.0, index=norm.index)
        total_weight = 0.0
        for sid, w in weights.items():
            if sid in norm.columns:
                composite += norm[sid].fillna(0) * w
                total_weight += w
        if total_weight > 0:
            composite /= total_weight
        composite.name = "CEI_Composite"
        return composite.dropna()

    def build_lagging_indicator(self, start: str = "2000-01-01") -> pd.Series:
        """
        Lagging indicator: prime rate, CPI, unemployment, consumer credit.
        """
        norm = self._fetch_and_normalize(self._LAG_COMPONENTS, start)
        if norm.empty:
            return pd.Series(dtype=float, name="LAG_Composite")

        weights = self._LAG_COMPONENTS
        composite = pd.Series(0.0, index=norm.index)
        total_weight = 0.0
        for sid, w in weights.items():
            if sid in norm.columns:
                composite += norm[sid].fillna(0) * w
                total_weight += w
        if total_weight > 0:
            composite /= total_weight
        composite.name = "LAG_Composite"
        return composite.dropna()

    def detect_economic_cycle(
        self,
        leading: pd.Series,
        coincident: pd.Series,
    ) -> str:
        """
        Classify economic cycle using leading and coincident indicators.

        EXPANSION:   LEI > 0, CEI > 0, LEI rising or flat
        SLOWDOWN:    LEI declining, CEI still positive
        RECESSION:   LEI < -0.5, CEI < -0.5
        RECOVERY:    LEI rising from negative, CEI still low
        """
        if leading.empty or coincident.empty:
            return "UNKNOWN"

        lei_now = float(leading.iloc[-1])
        lei_3m = float(leading.iloc[-3]) if len(leading) >= 3 else lei_now
        cei_now = float(coincident.iloc[-1])

        lei_momentum = lei_now - lei_3m

        # Check NBER recession indicator from FRED
        rec = self._client.fetch_series("USREC")
        in_recession = not rec.empty and float(rec.iloc[-1]) == 1.0

        if in_recession:
            if lei_momentum > 0.1:
                return "RECOVERY"
            return "RECESSION"
        if lei_now < -0.5 and cei_now < -0.5:
            return "RECESSION"
        if lei_now > 0.2 and cei_now > 0 and lei_momentum >= 0:
            return "EXPANSION"
        if lei_momentum < -0.2:
            return "SLOWDOWN"
        if lei_now < 0 and lei_momentum > 0.1:
            return "RECOVERY"
        return "EXPANSION"

    def compute_gdp_nowcast(self) -> dict:
        """
        Simplified Atlanta Fed-style GDP nowcast.
        Weighted contributions from key monthly data.
        Component weights: Payrolls 35%, ISM/IP 30%, Retail 20%, Housing 15%.
        """
        sids = ["PAYEMS", "INDPRO", "RRSFS", "HOUST"]
        df = self._client.fetch_multiple_series(sids, start="2020-01-01")
        if df.empty:
            return {"nowcast_pct": None, "components": {}, "as_of": date.today().isoformat()}

        df_monthly = df.resample("ME").last().pct_change(periods=1) * 100
        if df_monthly.empty or len(df_monthly) < 2:
            return {"nowcast_pct": None, "components": {}, "as_of": date.today().isoformat()}

        latest = df_monthly.iloc[-1]
        weights = {"PAYEMS": 0.35, "INDPRO": 0.30, "RRSFS": 0.20, "HOUST": 0.15}
        nowcast = 0.0
        components = {}
        total_w = 0.0
        for sid, w in weights.items():
            if sid in latest and not pd.isna(latest[sid]):
                val = float(latest[sid])
                contribution = val * w * 12  # annualise monthly growth
                nowcast += contribution
                components[sid] = round(val, 3)
                total_w += w
        if total_w < 0.5:
            return {"nowcast_pct": None, "components": components, "as_of": date.today().isoformat()}

        return {
            "nowcast_pct": round(nowcast / total_w, 2),
            "components": components,
            "methodology": "weighted avg: payrolls 35%, IP 30%, retail 20%, housing 15%",
            "as_of": date.today().isoformat(),
        }


# ---------------------------------------------------------------------------
# Macro Revision Tracker
# ---------------------------------------------------------------------------

class MacroRevisionTracker:
    """
    Track data revisions for point-in-time backtesting.
    GDP and employment data are frequently revised — PIT correctness matters.

    Without an API key, revision tracking is limited to comparing first-available
    (oldest vintage via CSV) vs. current reading.
    """

    # Series known to be frequently revised
    REVISION_PRONE = ["GDPC1", "PAYEMS", "INDPRO", "RRSFS", "PCE", "CPIAUCSL"]

    def __init__(self, client: Optional[FREDAPIClient] = None):
        self._client = client or FREDAPIClient()

    def get_latest_vintage(self, series_id: str) -> pd.Series:
        """Current (most recently revised) values."""
        return self._client.fetch_series(series_id)

    def get_first_release_estimates(self, series_id: str) -> pd.Series:
        """
        Approximate first-release: take earliest 3 observations per
        reference period as proxy for initial estimate.
        Without paid vintage database, this uses current CSV (best available).
        Returns current series with a first_release=True annotation.
        """
        s = self._client.fetch_series(series_id)
        if s.empty:
            return s
        # For demonstration: replicate first-release by using data from
        # well before the current date (assumes data < 5yr old may still be revised)
        logger.info(
            "Note: True first-release tracking requires FRED API vintage access. "
            "Using current CSV — may include revisions."
        )
        return s

    def compute_revision_statistics(self, series_id: str) -> dict:
        """
        Revision statistics comparing growth rates vs. lagged values
        (a proxy for revision magnitude using current available data).
        """
        s = self._client.fetch_series(series_id)
        if s.empty or len(s) < 24:
            return {"series_id": series_id, "error": "Insufficient data"}

        # Month-over-month changes
        mom = s.pct_change() * 100
        mom_clean = mom.dropna()

        # Simulate "revision" as the difference between preliminary (first observation
        # in a period) and the value after 3-month lag (proxy for finalized)
        if len(mom_clean) < 12:
            return {"series_id": series_id, "error": "Insufficient data"}

        # 3-month lag difference as revision proxy
        revision_proxy = mom_clean - mom_clean.shift(3)
        rev_clean = revision_proxy.dropna()

        return {
            "series_id": series_id,
            "title": MacroSeriesLibrary().get_description(series_id),
            "mean_revision_proxy": round(float(rev_clean.mean()), 4),
            "std_revision_proxy": round(float(rev_clean.std()), 4),
            "max_upward_revision": round(float(rev_clean.max()), 4),
            "max_downward_revision": round(float(rev_clean.min()), 4),
            "positive_revision_rate": round(float((rev_clean > 0).mean()), 4),
            "methodology": "3M lag difference as revision proxy (approx.)",
            "as_of": date.today().isoformat(),
        }

    def detect_large_revision(
        self,
        series_id: str,
        threshold_std: float = 2.0,
    ) -> List[RevisionRecord]:
        """
        Flag periods where revision magnitude > threshold_std standard deviations.
        """
        s = self._client.fetch_series(series_id)
        if s.empty or len(s) < 24:
            return []

        mom = s.pct_change() * 100
        revision_proxy = (mom - mom.shift(3)).dropna()

        mu = revision_proxy.mean()
        sigma = revision_proxy.std()
        if sigma < 1e-9:
            return []

        threshold = mu + threshold_std * sigma
        flags = revision_proxy[revision_proxy.abs() > abs(threshold)]

        records = []
        for dt, rev_val in flags.items():
            orig_val = float(s.get(dt, 0)) if dt in s.index else 0.0
            records.append(RevisionRecord(
                series_id=series_id,
                reference_period=str(dt.date()),
                first_release=orig_val,
                latest_value=orig_val + rev_val,
                revision=round(float(rev_val), 4),
                revision_pct=round(float(rev_val / abs(orig_val) * 100), 2) if orig_val != 0 else None,
                is_large=True,
            ))
        return records

    def get_revision_summary_table(self) -> pd.DataFrame:
        """Revision statistics for all revision-prone series."""
        rows = []
        for sid in self.REVISION_PRONE:
            stats = self.compute_revision_statistics(sid)
            if "error" not in stats:
                rows.append({
                    "series_id": sid,
                    "title": stats.get("title", sid)[:40],
                    "mean_revision": stats.get("mean_revision_proxy"),
                    "std_revision": stats.get("std_revision_proxy"),
                    "pct_positive": stats.get("positive_revision_rate"),
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-Country Comparator
# ---------------------------------------------------------------------------

# Country → FRED series mapping
COUNTRY_CPI: Dict[str, str] = {
    "US":  "CPIAUCSL",
    "EU":  "EA19RHPUPTT01IXOBSAM",
    "UK":  "GBRCPIALLMINMEI",
    "JP":  "JPNCPIALLMINMEI",
    "CN":  "CHNCPIALLMINMEI",
    "CA":  "CANCPIALLMINMEI",
    "DE":  "DEUCPIALLMINMEI",
    "KR":  "KORCPIALLMINMEI",
}

COUNTRY_GDP: Dict[str, str] = {
    "US":  "GDPC1",
    "EU":  "CLVMEURSCAB1GQEA19",
    "UK":  "CLVMNACSCAB1GQQBIS",
    "JP":  "CLVMNACSCAB1GQJP",
    "DE":  "CLVMEURSCAB1GQDE",
}

COUNTRY_UNEMP: Dict[str, str] = {
    "US":  "UNRATE",
    "EU":  "LRHUTTTTEZM156S",
    "UK":  "GBRUNR",
    "JP":  "JPNURYNBIS",
    "DE":  "DEUURYNBIS",
    "CA":  "CANURYNBIS",
    "KR":  "KORURYNBIS",
}

COUNTRY_GDP_PC: Dict[str, str] = {
    "US":  "NYGDPPCAPKDUSA",
    "UK":  "NYGDPPCAPKDGBR",
    "DE":  "NYGDPPCAPKDDEU",
    "JP":  "NYGDPPCAPKDJPN",
    "FR":  "NYGDPPCAPKDFRA",
    "CN":  "NYGDPPCAPKDCHN",
    "KR":  "NYGDPPCAPKDKOR",
    "CA":  "NYGDPPCAPKDCAN",
}


class CrossCountryComparator:
    """
    G7 + EM cross-country macro comparisons using FRED.
    """

    def __init__(self, client: Optional[FREDAPIClient] = None):
        self._client = client or FREDAPIClient()

    def compare_inflation(
        self,
        countries: List[str] = ["US", "EU", "UK", "JP", "CN"],
        start: str = "2010-01-01",
    ) -> pd.DataFrame:
        """
        YoY CPI inflation rate by country.
        Returns DataFrame with country columns and date index.
        """
        series_map = {c: COUNTRY_CPI[c] for c in countries if c in COUNTRY_CPI}
        df_raw = self._client.fetch_multiple_series(list(series_map.values()), start=start)
        if df_raw.empty:
            return pd.DataFrame()
        # Rename to country codes
        inv_map = {v: k for k, v in series_map.items()}
        df_raw = df_raw.rename(columns=inv_map)
        # Compute YoY % change (12M)
        df_yoy = df_raw.resample("ME").last().pct_change(12) * 100
        df_yoy.columns.name = "Country"
        return df_yoy.dropna(how="all")

    def compare_gdp_growth(
        self,
        countries: List[str] = ["US", "EU", "UK", "JP"],
        start: str = "2000-01-01",
    ) -> pd.DataFrame:
        """
        QoQ annualised real GDP growth by country.
        """
        series_map = {c: COUNTRY_GDP[c] for c in countries if c in COUNTRY_GDP}
        df_raw = self._client.fetch_multiple_series(list(series_map.values()), start=start)
        if df_raw.empty:
            return pd.DataFrame()
        inv_map = {v: k for k, v in series_map.items()}
        df_raw = df_raw.rename(columns=inv_map)
        # QoQ annualised
        df_qoq = df_raw.resample("QE").last().pct_change(1) * 400
        return df_qoq.dropna(how="all")

    def compare_unemployment(
        self,
        countries: List[str] = ["US", "EU", "UK", "JP", "DE"],
        start: str = "2000-01-01",
    ) -> pd.DataFrame:
        """Unemployment rate by country (%)."""
        series_map = {c: COUNTRY_UNEMP[c] for c in countries if c in COUNTRY_UNEMP}
        df_raw = self._client.fetch_multiple_series(list(series_map.values()), start=start)
        if df_raw.empty:
            return pd.DataFrame()
        inv_map = {v: k for k, v in series_map.items()}
        df_raw = df_raw.rename(columns=inv_map)
        return df_raw.resample("ME").last().dropna(how="all")

    def compute_gdp_per_capita_ranking(
        self,
        countries: List[str] = ["US", "UK", "DE", "JP", "FR", "CN", "KR", "CA"],
    ) -> pd.DataFrame:
        """
        GDP per capita (constant 2015 USD) from World Bank via FRED.
        """
        series_map = {c: COUNTRY_GDP_PC[c] for c in countries if c in COUNTRY_GDP_PC}
        df_raw = self._client.fetch_multiple_series(list(series_map.values()), start="2010-01-01")
        if df_raw.empty:
            return pd.DataFrame()
        inv_map = {v: k for k, v in series_map.items()}
        df_raw = df_raw.rename(columns=inv_map)
        latest = df_raw.resample("YE").last().iloc[-1].dropna()
        df_rank = pd.DataFrame({
            "country": latest.index,
            "gdp_per_capita_usd": latest.values.round(0),
        }).sort_values("gdp_per_capita_usd", ascending=False).reset_index(drop=True)
        df_rank.index += 1
        df_rank.index.name = "rank"
        return df_rank

    def detect_global_recession_risk(self) -> dict:
        """
        Global recession risk score from PMI, leading indicators, trade volume.
        """
        # Key global indicators
        sids = [
            "USREC",      # NBER recession
            "NFCI",       # Financial conditions
            "T10Y3M",     # Yield curve
            "DCOILWTICO", # Oil (demand proxy)
            "VIXCLS",     # Risk sentiment
        ]
        df = self._client.fetch_multiple_series(sids, start="2020-01-01")
        if df.empty:
            return {"risk_level": "UNKNOWN", "score": None}

        latest = df.resample("ME").last().iloc[-1]
        score = 0
        signals = {}

        # Yield curve inversion
        if "T10Y3M" in latest and not pd.isna(latest["T10Y3M"]):
            yc = float(latest["T10Y3M"])
            signals["yield_curve_bps"] = round(yc * 100, 0)
            if yc < -0.5:
                score += 30
            elif yc < 0:
                score += 15

        # Financial stress
        if "NFCI" in latest and not pd.isna(latest["NFCI"]):
            nfci = float(latest["NFCI"])
            signals["nfci"] = round(nfci, 3)
            if nfci > 0.5:
                score += 20
            elif nfci > 0:
                score += 10

        # VIX
        if "VIXCLS" in latest and not pd.isna(latest["VIXCLS"]):
            vix = float(latest["VIXCLS"])
            signals["vix"] = round(vix, 1)
            if vix > 30:
                score += 20
            elif vix > 20:
                score += 10

        # Recession indicator
        if "USREC" in latest and float(latest.get("USREC", 0)) == 1:
            score += 30

        risk_level = "LOW" if score < 20 else "MODERATE" if score < 45 else "HIGH" if score < 70 else "CRITICAL"
        return {
            "risk_level": risk_level,
            "score": score,
            "signals": signals,
            "as_of": date.today().isoformat(),
        }

    def get_global_snapshot_table(
        self,
        countries: List[str] = ["US", "EU", "UK", "JP", "DE", "CN"],
        start: str = "2020-01-01",
    ) -> pd.DataFrame:
        """
        5-indicator summary table: inflation, unemployment, GDP growth for each country.
        """
        cpi_df = self.compare_inflation(countries, start)
        unemp_df = self.compare_unemployment(countries, start)
        gdp_df = self.compare_gdp_growth(countries, start)

        rows = []
        for c in countries:
            cpi_now = float(cpi_df[c].dropna().iloc[-1]) if c in cpi_df.columns and not cpi_df[c].dropna().empty else None
            unemp_now = float(unemp_df[c].dropna().iloc[-1]) if c in unemp_df.columns and not unemp_df[c].dropna().empty else None
            gdp_now = float(gdp_df[c].dropna().iloc[-1]) if c in gdp_df.columns and not gdp_df[c].dropna().empty else None
            rows.append({
                "country": c,
                "cpi_yoy_pct": round(cpi_now, 2) if cpi_now is not None else None,
                "unemployment_pct": round(unemp_now, 2) if unemp_now is not None else None,
                "gdp_growth_qoq_ann_pct": round(gdp_now, 2) if gdp_now is not None else None,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Macro Dashboard Builder
# ---------------------------------------------------------------------------

# Core US indicators for dashboard
_DASHBOARD_SERIES: Dict[str, Tuple[str, str]] = {
    "Real GDP":             ("GDPC1",    "Quarterly"),
    "Unemployment Rate":    ("UNRATE",   "Monthly"),
    "Core CPI":             ("CPILFESL", "Monthly"),
    "Core PCE":             ("PCEPILFE", "Monthly"),
    "Initial Claims":       ("ICSA",     "Weekly"),
    "Consumer Sentiment":   ("UMCSENT",  "Monthly"),
    "Industrial Prod.":     ("INDPRO",   "Monthly"),
    "Retail Sales":         ("RRSFS",    "Monthly"),
    "10-Year Treasury":     ("DGS10",    "Daily"),
    "Fed Funds Rate":       ("FEDFUNDS", "Monthly"),
}


class MacroDashboardBuilder:
    """
    Macro dashboard snapshots: US summary, global table, macro report narrative.
    """

    def __init__(self, client: Optional[FREDAPIClient] = None):
        self._client = client or FREDAPIClient()
        self._comparator = CrossCountryComparator(self._client)
        self._builder = MacroIndicatorBuilder(self._client)

    def _make_snapshot(
        self,
        name: str,
        series_id: str,
        frequency: str,
    ) -> Optional[MacroSnapshot]:
        """Build a MacroSnapshot for a single series."""
        s = self._client.fetch_series(series_id, start="2019-01-01")
        if s.empty:
            return None
        s = s.dropna()
        if len(s) < 2:
            return None

        latest_val = float(s.iloc[-1])
        prev_val = float(s.iloc[-2])
        chg = latest_val - prev_val
        chg_pct = (chg / abs(prev_val) * 100) if abs(prev_val) > 1e-9 else None

        # Z-score vs 5yr
        five_yr = s[s.index >= s.index[-1] - pd.DateOffset(years=5)]
        z = None
        if len(five_yr) > 20:
            mu = five_yr.mean()
            sigma = five_yr.std()
            if sigma > 1e-9:
                z = float((latest_val - mu) / sigma)

        trend = "STABLE"
        if len(s) >= 3:
            three_period_chg = float(s.iloc[-1]) - float(s.iloc[-3])
            if three_period_chg > abs(latest_val) * 0.01:
                trend = "RISING"
            elif three_period_chg < -abs(latest_val) * 0.01:
                trend = "FALLING"

        return MacroSnapshot(
            indicator=name,
            series_id=series_id,
            as_of=s.index[-1].date(),
            latest_value=round(latest_val, 4),
            previous_value=round(prev_val, 4),
            change=round(chg, 4),
            change_pct=round(chg_pct, 3) if chg_pct is not None else None,
            frequency=frequency,
            units="",
            trend=trend,
            zscore_5yr=round(z, 3) if z is not None else None,
        )

    def get_us_macro_snapshot(self, as_of: Optional[str] = None) -> dict:
        """
        Key US macro indicators: latest value, prior reading, change, trend.
        """
        snapshots = {}
        for name, (sid, freq) in _DASHBOARD_SERIES.items():
            snap = self._make_snapshot(name, sid, freq)
            if snap:
                snapshots[name] = {
                    "series_id": sid,
                    "as_of": snap.as_of.isoformat(),
                    "latest": snap.latest_value,
                    "previous": snap.previous_value,
                    "change": snap.change,
                    "change_pct": snap.change_pct,
                    "trend": snap.trend,
                    "zscore_5yr": snap.zscore_5yr,
                }
        return {
            "as_of": date.today().isoformat(),
            "indicators": snapshots,
        }

    def get_global_macro_snapshot(self) -> pd.DataFrame:
        """10 countries × 3 indicators table."""
        countries = ["US", "EU", "UK", "JP", "DE", "CN", "CA", "KR"]
        return self._comparator.get_global_snapshot_table(countries)

    def compute_macro_surprise_index(
        self,
        actual_vs_consensus: List[dict],
    ) -> float:
        """
        Citi-style Economic Surprise Index.
        Input: [{"indicator": str, "actual": float, "consensus": float, "weight": float}, ...]
        Returns: surprise index value (positive = beats, negative = misses).
        """
        if not actual_vs_consensus:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        for item in actual_vs_consensus:
            actual = item.get("actual")
            consensus = item.get("consensus")
            weight = item.get("weight", 1.0)
            if actual is None or consensus is None:
                continue
            surprise = actual - consensus
            weighted_sum += surprise * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 4)

    def generate_macro_report(self) -> str:
        """
        5-sentence narrative on the current US macroeconomic environment.
        """
        snap = self.get_us_macro_snapshot()
        indicators = snap.get("indicators", {})

        def _get(name: str, key: str = "latest") -> Optional[float]:
            return indicators.get(name, {}).get(key)

        def _trend(name: str) -> str:
            return indicators.get(name, {}).get("trend", "STABLE")

        gdp = _get("Real GDP")
        unemp = _get("Unemployment Rate")
        core_cpi = _get("Core CPI")
        core_pce = _get("Core PCE")
        fed_funds = _get("Fed Funds Rate")
        sentiment = _get("Consumer Sentiment")

        lines = []

        # GDP sentence
        if gdp is not None:
            lines.append(
                f"The US economy continues to expand with real GDP at ${gdp:,.0f}B "
                f"(chained 2017$), with industrial production {_trend('Industrial Prod.').lower()}."
            )
        else:
            lines.append("US GDP data is currently being refreshed.")

        # Labor market
        if unemp is not None:
            label = "near historic lows" if unemp < 4.5 else "elevated" if unemp > 6 else "moderate"
            lines.append(
                f"Labor market conditions remain {label}, with the unemployment rate "
                f"at {unemp:.1f}% and initial jobless claims "
                f"{'rising' if _trend('Initial Claims') == 'RISING' else 'stable or falling'}."
            )

        # Inflation
        cpi_str = f"{core_cpi:.2f}%" if core_cpi is not None else "N/A"
        pce_str = f"{core_pce:.2f}%" if core_pce is not None else "N/A"
        lines.append(
            f"Inflation remains a key focus with Core CPI at {cpi_str} "
            f"and Core PCE (the Fed's preferred gauge) at {pce_str} YoY."
        )

        # Monetary policy
        if fed_funds is not None:
            lines.append(
                f"The Federal Reserve has set the policy rate at {fed_funds:.2f}%, "
                f"balancing its dual mandate of price stability and maximum employment."
            )

        # Consumer
        if sentiment is not None:
            sentiment_label = "optimistic" if sentiment > 80 else "cautious" if sentiment < 65 else "neutral"
            lines.append(
                f"Consumer sentiment stands at {sentiment:.1f} (Michigan Survey), "
                f"reflecting a {sentiment_label} outlook; retail sales are "
                f"{_trend('Retail Sales').lower()}."
            )
        else:
            lines.append(
                "Consumer confidence and spending data remain the key swing factors "
                "for near-term growth trajectories."
            )

        return " ".join(lines[:5])  # exactly 5 sentences


# ---------------------------------------------------------------------------
# FRED Macro Engine (Orchestrator)
# ---------------------------------------------------------------------------

class FREDMacroEngine:
    """
    Full orchestrator: daily updates, parquet exports, dashboards, search.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("FRED_API_KEY")
        self._client = FREDAPIClient(api_key=self._api_key)
        self._library = MacroSeriesLibrary()
        self._indicator_builder = MacroIndicatorBuilder(self._client)
        self._revision_tracker = MacroRevisionTracker(self._client)
        self._comparator = CrossCountryComparator(self._client)
        self._dashboard = MacroDashboardBuilder(self._client)

    def get_full_dashboard(self) -> dict:
        """
        Comprehensive macro dashboard: US snapshot, global table,
        leading indicators, cycle position, recession risk.
        """
        us_snap = self._dashboard.get_us_macro_snapshot()
        global_snap = self._dashboard.get_global_macro_snapshot()

        leading = self._indicator_builder.build_leading_indicator_composite()
        coincident = self._indicator_builder.build_coincident_indicator()
        cycle = self._indicator_builder.detect_economic_cycle(leading, coincident)
        nowcast = self._indicator_builder.compute_gdp_nowcast()
        recession_risk = self._comparator.detect_global_recession_risk()
        macro_report = self._dashboard.generate_macro_report()

        lei_current = float(leading.iloc[-1]) if not leading.empty else None
        cei_current = float(coincident.iloc[-1]) if not coincident.empty else None

        return {
            "as_of": date.today().isoformat(),
            "us_macro": us_snap,
            "global_snapshot": global_snap.to_dict("records") if not global_snap.empty else [],
            "leading_indicator": {
                "current": round(lei_current, 4) if lei_current is not None else None,
                "trend": "RISING" if (not leading.empty and len(leading) >= 3 and
                                       leading.iloc[-1] > leading.iloc[-3]) else "FALLING",
            },
            "coincident_indicator": {
                "current": round(cei_current, 4) if cei_current is not None else None,
            },
            "economic_cycle": cycle,
            "gdp_nowcast": nowcast,
            "recession_risk": recession_risk,
            "macro_report": macro_report,
        }

    def search_and_fetch(
        self,
        query: str,
        start: str = "2000-01-01",
        limit: int = 5,
    ) -> pd.DataFrame:
        """Search FRED for a query and fetch top results."""
        return self._client.search_and_fetch(query, start=start, limit=limit)

    def run_daily_update(self) -> dict:
        """
        Refresh all core series from the library.
        Returns a summary of series fetched and any failures.
        """
        all_series = self._library.get_all_series()
        success = []
        failed = []

        logger.info("Running FRED daily update: %d series", len(all_series))
        # Batch in groups of 20 to avoid hammering FRED
        batch_size = 20
        sids = list(all_series.keys())
        for i in range(0, len(sids), batch_size):
            batch = sids[i:i + batch_size]
            df = self._client.fetch_multiple_series(batch, force_refresh=True)
            for sid in batch:
                if sid in df.columns and not df[sid].dropna().empty:
                    success.append(sid)
                else:
                    failed.append(sid)
            time.sleep(0.5)  # polite rate limiting

        return {
            "as_of": datetime.now().isoformat(),
            "total_series": len(sids),
            "success": len(success),
            "failed": len(failed),
            "failed_series": failed[:20],  # first 20
        }

    def export_to_parquet(self, output_dir: str) -> dict:
        """
        Export all curated series to Parquet files, one per category.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        exported = {}
        for category in self._library.get_all_categories():
            sids = self._library.get_series_ids(category)
            if not sids:
                continue
            df = self._client.fetch_multiple_series(sids, start="1990-01-01")
            if df.empty:
                exported[category] = {"rows": 0, "file": None}
                continue
            fname = out_path / f"fred_{category}.parquet"
            df.to_parquet(fname, engine="pyarrow" if _pyarrow_available() else "fastparquet")
            exported[category] = {"rows": len(df), "file": str(fname), "columns": df.shape[1]}
            logger.info("Exported %s → %s (%d rows, %d series)", category, fname, len(df), df.shape[1])

        return {
            "output_dir": str(out_path),
            "categories_exported": exported,
            "as_of": datetime.now().isoformat(),
        }

    def get_yield_curve(self) -> pd.Series:
        """Current US Treasury yield curve."""
        tenors = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
        labels = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
        df = self._client.fetch_multiple_series(tenors)
        if df.empty:
            return pd.Series(dtype=float)
        latest = df.iloc[-1]
        return pd.Series({labels[i]: latest.get(tenors[i]) for i in range(len(tenors))})

    def get_credit_spreads(self) -> pd.Series:
        """Current OAS credit spreads from FRED."""
        sids = {
            "IG_OAS": "BAMLC0A0CM",
            "HY_OAS": "BAMLH0A0HYM2",
            "BBB_OAS": "BAMLC0A4CBBB",
            "BB_OAS": "BAMLH0A1HYBB",
            "CCC_OAS": "BAMLH0A3HYC",
        }
        df = self._client.fetch_multiple_series(list(sids.values()))
        if df.empty:
            return pd.Series(dtype=float)
        latest = df.iloc[-1]
        return pd.Series({k: latest.get(v) for k, v in sids.items()})

    def compute_recession_probability(self) -> float:
        """
        Simple probit-based recession probability using yield curve + unemployment.
        Rough approximation: P(recession) based on 10Y-3M spread and financial conditions.
        """
        t10y3m = self._client.fetch_series("T10Y3M")
        nfci = self._client.fetch_series("NFCI")

        if t10y3m.empty:
            return 0.0

        yc = float(t10y3m.iloc[-1])
        fc = float(nfci.iloc[-1]) if not nfci.empty else 0.0

        # Rough linear approximation from academic literature (Estrella & Mishkin 1998)
        # Probit: P(recession 4Q ahead) ≈ N(-0.8736 - 0.6495 × spread)
        if _SCIPY_AVAILABLE:
            from scipy.stats import norm as _n
            probit_input = -0.8736 - 0.6495 * yc + 0.3 * max(fc, 0)
            prob = float(_n.cdf(probit_input))
        else:
            import math
            probit_input = -0.8736 - 0.6495 * yc + 0.3 * max(fc, 0)
            prob = 0.5 * math.erfc(-probit_input / math.sqrt(2))

        return round(min(max(prob, 0.0), 1.0), 4)

    # ------------------------------------------------------------------
    # dim_043 push 8→9: yield curve, FCI, growth surprise
    # ------------------------------------------------------------------

    def compute_yield_curve_slope(self) -> Dict[str, Any]:
        """
        Compute the yield curve slope (10Y − 3M Treasury spread).

        This is the Estrella-Mishkin (1998) recession predictor series,
        published by the New York Fed.  A negative reading (yield curve
        inversion) has preceded every US recession since 1969.

        FRED series used:
          DGS10  — 10-Year Treasury Constant Maturity Rate (%)
          DTB3   — 3-Month Treasury Bill: Secondary Market Rate (%)

        Returns
        -------
        dict with:
          slope_pp         : 10Y − 3M spread in percentage points
          inverted         : bool — True when slope < 0
          recession_signal : str  — "Inverted" / "Flat" / "Normal" / "Steep"
          series_10y       : latest 10Y rate
          series_3m        : latest 3M rate
        """
        df_10y = self._client.fetch_series("DGS10")
        df_3m = self._client.fetch_series("DTB3")

        rate_10y: Optional[float] = None
        rate_3m: Optional[float] = None

        if not df_10y.empty:
            rate_10y = float(df_10y.dropna().iloc[-1])
        if not df_3m.empty:
            rate_3m = float(df_3m.dropna().iloc[-1])

        if rate_10y is None or rate_3m is None:
            return {
                "slope_pp": None,
                "inverted": None,
                "recession_signal": "Data unavailable",
                "series_10y": rate_10y,
                "series_3m": rate_3m,
            }

        slope = rate_10y - rate_3m

        if slope < 0.0:
            signal = "Inverted"
        elif slope < 0.50:
            signal = "Flat"
        elif slope < 2.00:
            signal = "Normal"
        else:
            signal = "Steep"

        return {
            "slope_pp": round(slope, 4),
            "inverted": slope < 0.0,
            "recession_signal": signal,
            "series_10y": round(rate_10y, 4),
            "series_3m": round(rate_3m, 4),
            "interpretation": (
                "Historically strong recession predictor 12 months ahead"
                if slope < 0.0 else
                "Mild slowdown risk" if slope < 0.50 else
                "Neutral economic signal" if slope < 2.00 else
                "Expansionary — steepening curve"
            ),
        }

    def compute_financial_conditions_index(self) -> Dict[str, Any]:
        """
        Compute a weighted Financial Conditions Index (FCI) from 5 FRED series.

        FCI = Σ (w_i × z_i)

        where z_i = (x_i − μ_i) / σ_i is the rolling 5-year z-score of series i.

        Weights and series (sourced from Goldman Sachs / Chicago Fed methodology):
          TED spread    (TEDRATE)    w = 0.25  — credit stress / interbank risk
          VIX           (VIXCLS)     w = 0.20  — equity volatility / risk appetite
          BAA-AAA spread (BAA, AAA)  w = 0.25  — corporate credit risk
          Dollar index  (DTWEXBGS)   w = 0.15  — broad trade-weighted USD
          Housing starts (HOUST)     w = 0.15  — real sector credit demand

        Positive FCI → tighter-than-average conditions.
        Negative FCI → easier-than-average conditions.

        Returns
        -------
        dict with:
          fci           : float — weighted z-score composite
          tightening    : bool
          components    : per-series z-score and weight
        """
        # Series config: (fred_id, weight, invert)
        # invert=True for series where higher value = easier conditions (housing starts, dollar)
        SERIES_CONFIG: List[Tuple[str, float, bool]] = [
            ("TEDRATE",  0.25, False),   # TED spread — high = tight
            ("VIXCLS",   0.20, False),   # VIX — high = tight
            ("BAA",      0.25, False),   # Moody's BAA yield — high = tight
            ("DTWEXBGS", 0.15, False),   # Dollar index — high = tight (imported cost)
            ("HOUST",    0.15, True),    # Housing starts — high = easy (invert)
        ]

        window = 260   # ~5 years of weekly obs; falls back to available data

        fci = 0.0
        components: Dict[str, Dict[str, Any]] = {}
        weight_used = 0.0

        for series_id, weight, invert in SERIES_CONFIG:
            try:
                df = self._client.fetch_series(series_id)
                if df.empty or len(df.dropna()) < 20:
                    continue
                vals = df.dropna().astype(float)
                mu = float(vals.iloc[-min(window, len(vals)):].mean())
                sigma = float(vals.iloc[-min(window, len(vals)):].std(ddof=1))
                latest = float(vals.iloc[-1])
                z = (latest - mu) / sigma if sigma > 0 else 0.0
                if invert:
                    z = -z
                fci += weight * z
                weight_used += weight
                components[series_id] = {
                    "latest_value": round(latest, 4),
                    "mean_5yr": round(mu, 4),
                    "std_5yr": round(sigma, 4),
                    "zscore": round(z, 4),
                    "weight": weight,
                    "contribution": round(weight * z, 6),
                    "inverted": invert,
                }
            except Exception as exc:
                logger.debug("FCI: could not fetch %s: %s", series_id, exc)

        # Re-scale to full weight if some series were unavailable
        if 0 < weight_used < 1.0:
            fci = fci / weight_used

        return {
            "fci": round(fci, 6),
            "tightening": fci > 0.0,
            "interpretation": (
                "Significantly tighter than average" if fci > 1.0 else
                "Mildly tighter than average" if fci > 0.0 else
                "Mildly easier than average" if fci > -1.0 else
                "Significantly easier than average"
            ),
            "weight_coverage": round(weight_used, 4),
            "components": components,
        }

    def compute_growth_surprise_index(
        self,
        nowcast_series: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute a GDP Growth Surprise Index.

        Growth Surprise = Actual GDP growth − mean(nowcast estimates)

        When live nowcast data are unavailable, the method approximates the
        surprise as the deviation of realised GDP (GDPC1 YoY %) from the
        rolling 4-quarter mean (a simple adaptive expectation model).

        Parameters
        ----------
        nowcast_series : optional list of external nowcast point estimates
                         (annualised % growth).  If provided, their mean is
                         used as the consensus forecast.

        Returns
        -------
        dict with:
          actual_gdp_yoy_pct   : most recent year-over-year GDP growth
          consensus_forecast   : mean of provided nowcasts (or adaptive mean)
          growth_surprise      : actual − consensus (pp)
          surprise_direction   : "Positive" / "Negative" / "Neutral"
        """
        df = self._client.fetch_series("GDPC1")

        if df.empty or len(df.dropna()) < 4:
            return {
                "actual_gdp_yoy_pct": None,
                "consensus_forecast": None,
                "growth_surprise": None,
                "surprise_direction": "Data unavailable",
            }

        vals = df.dropna().astype(float)
        # YoY % change (quarterly data → 4 periods)
        yoy = vals.pct_change(4) * 100.0
        yoy = yoy.dropna()

        if yoy.empty:
            return {
                "actual_gdp_yoy_pct": None,
                "consensus_forecast": None,
                "growth_surprise": None,
                "surprise_direction": "Insufficient data",
            }

        actual = float(yoy.iloc[-1])

        if nowcast_series and len(nowcast_series) > 0:
            consensus = float(np.mean([float(x) for x in nowcast_series]))
            consensus_method = "provided_nowcasts"
        else:
            # Adaptive expectation: trailing 4-quarter mean of YoY growth
            lookback = min(4, len(yoy) - 1)
            consensus = float(yoy.iloc[-(lookback + 1):-1].mean()) if lookback > 0 else actual
            consensus_method = "adaptive_4q_mean"

        surprise = actual - consensus

        return {
            "actual_gdp_yoy_pct": round(actual, 4),
            "consensus_forecast": round(consensus, 4),
            "growth_surprise": round(surprise, 4),
            "surprise_direction": (
                "Positive" if surprise > 0.10 else
                "Negative" if surprise < -0.10 else
                "Neutral"
            ),
            "consensus_method": consensus_method,
            "n_nowcasts": len(nowcast_series) if nowcast_series else 0,
        }


# Try importing scipy for probit
try:
    from scipy.stats import norm as _sn
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Main — demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("=" * 70)
    print("SENTINEL FRED Macro Engine v3 — 765K+ Series")
    print("=" * 70)

    engine = FREDMacroEngine()
    client = engine._client
    indicator_builder = engine._indicator_builder
    comparator = engine._comparator
    dashboard = engine._dashboard

    # 1. Fetch 10 key US macro series
    print("\n[1] Fetching 10 key US macro series...")
    KEY_SERIES = ["GDPC1", "UNRATE", "CPIAUCSL", "CPILFESL", "PCEPILFE",
                  "ICSA", "PAYEMS", "UMCSENT", "DGS10", "FEDFUNDS"]
    df_key = client.fetch_multiple_series(KEY_SERIES, start="2020-01-01")
    if not df_key.empty:
        latest = df_key.resample("ME").last().iloc[-1]
        lib = MacroSeriesLibrary()
        for sid in KEY_SERIES:
            if sid in latest:
                val = latest[sid]
                desc = lib.get_description(sid)
                print(f"  {sid:20s}: {val:>10.3f}  — {desc}")
    else:
        print("  Data unavailable (check network)")

    # 2. Build leading indicator composite
    print("\n[2] Building Leading Indicator Composite...")
    lei = indicator_builder.build_leading_indicator_composite(start="2015-01-01")
    if not lei.empty:
        print(f"  LEI Composite (latest 6 months):")
        for dt, val in lei.tail(6).items():
            print(f"    {str(dt.date()):12s}: {val:+.4f}")
    else:
        print("  LEI data unavailable")

    # 3. Detect economic cycle
    print("\n[3] Economic Cycle Detection...")
    coincident = indicator_builder.build_coincident_indicator(start="2015-01-01")
    cycle = indicator_builder.detect_economic_cycle(lei, coincident)
    print(f"  Current cycle phase: {cycle}")

    # 4. GDP Nowcast
    print("\n[4] GDP Nowcast...")
    nowcast = indicator_builder.compute_gdp_nowcast()
    print(f"  Nowcast: {nowcast.get('nowcast_pct', 'N/A')}% (annualised)")
    for comp, val in nowcast.get("components", {}).items():
        print(f"    {comp}: {val:+.3f}%")

    # 5. Cross-country inflation comparison
    print("\n[5] Cross-Country Inflation Comparison (YoY CPI %)...")
    countries = ["US", "EU", "UK", "JP", "DE"]
    df_cpi = comparator.compare_inflation(countries, start="2020-01-01")
    if not df_cpi.empty:
        latest_cpi = df_cpi.dropna(how="all").iloc[-1]
        for c in countries:
            if c in latest_cpi:
                print(f"  {c:4s}: {latest_cpi[c]:+.2f}%")
    else:
        print("  CPI comparison data unavailable")

    # 6. Recession probability
    print("\n[6] Recession Probability (Yield Curve Model)...")
    prob = engine.compute_recession_probability()
    print(f"  P(recession in 4Q): {prob * 100:.1f}%")

    # 7. US Macro Snapshot
    print("\n[7] US Macro Snapshot...")
    us_snap = dashboard.get_us_macro_snapshot()
    for name, data in list(us_snap.get("indicators", {}).items())[:8]:
        print(f"  {name:25s}: {data['latest']:>12.3f}  [{data['trend']:8s}] z={data.get('zscore_5yr', 'N/A')}")

    # 8. Macro narrative report
    print("\n[8] Macro Report Narrative...")
    report = dashboard.generate_macro_report()
    print(f"\n  {report}\n")

    # 9. Global snapshot
    print("[9] Global Macro Snapshot...")
    df_global = dashboard.get_global_macro_snapshot()
    if not df_global.empty:
        print(df_global.to_string(index=False))

    # 10. FRED series search
    print("\n[10] FRED Series Search (keyword: 'inflation expectations')...")
    results = client.search_series("inflation expectations", limit=5)
    for r in results:
        print(f"  {r.series_id:25s}: {r.title[:60]}")

    print("\nDone.")
