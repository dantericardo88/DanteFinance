"""
cftc_cot_v3.py — Comprehensive CFTC COT Positioning Analysis Platform.

dim_046: CFTC COT positioning — score 7 → 9

Architecture:
  COTDataDownloader      — Download legacy, disaggregated, TFF from CFTC.gov
  COTSignalEngine        — COT Index, extremes, commercial hedger, speculator crowding
  COTMarketCoverage      — 100+ market catalog with CFTC code mapping
  COTPortfolioAnalyzer   — Portfolio-level COT dashboard and risk appetite index
  COTAlertSystem         — Extreme positioning alerts with reversal detection
  COTEngine              — Orchestrator: full dashboard, signals, weekly update

Free data only: CFTC.gov public files (CSV / zip).
No API keys required. Parquet with CSV fallback for caching.
"""
from __future__ import annotations

import io
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deps — guarded
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CFTC_BASE = "https://www.cftc.gov/files/dea/history"
_CFTC_LATEST_DISAGG = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
_CFTC_LATEST_LEGACY = "https://www.cftc.gov/dea/newcot/deafut.txt"
_CFTC_LATEST_TFF    = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"

_TIMEOUT = 30
_HEADERS = {"User-Agent": "SENTINEL/3.0 (financial-terminal; research)"}

_CACHE_DIR = Path("sentinel/data")
_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours for latest files

# Disaggregated column spec — CFTC CSV format
_DISAGG_COLS = {
    "market":          "Market_and_Exchange_Names",
    "date":            "As_of_Date_In_Form_YYYY-MM-DD",
    "cftc_code":       "CFTC_Commodity_Code",
    "oi":              "Open_Interest_All",
    # Producer / Merchant (commercials)
    "comm_long":       "Prod_Merc_Positions_Long_All",
    "comm_short":      "Prod_Merc_Positions_Short_All",
    # Swap Dealers
    "swap_long":       "Swap_Positions_Long_All",
    "swap_short":      "Swap_Positions_Short_All",
    "swap_spread":     "Swap__Positions_Spread_All",
    # Managed Money (large specs / non-commercial equivalent)
    "mm_long":         "M_Money_Positions_Long_All",
    "mm_short":        "M_Money_Positions_Short_All",
    "mm_spread":       "M_Money_Positions_Spread_All",
    # Other Reportable
    "other_long":      "Other_Rept_Positions_Long_All",
    "other_short":     "Other_Rept_Positions_Short_All",
    "other_spread":    "Other_Rept_Positions_Spread_All",
    # Non-reportable (small specs)
    "nonrep_long":     "NonRept_Positions_Long_All",
    "nonrep_short":    "NonRept_Positions_Short_All",
    # Change columns
    "oi_change":       "Change_in_Open_Interest_All",
    "mm_long_change":  "Change_in_M_Money_Long_All",
    "mm_short_change": "Change_in_M_Money_Short_All",
}

# Legacy COT column spec (broader market coverage, simpler trader breakdown)
_LEGACY_COLS = {
    "market":          "Market_and_Exchange_Names",
    "date":            "As_of_Date_In_Form_YYYY-MM-DD",
    "cftc_code":       "CFTC_Commodity_Code",
    "oi":              "Open_Interest_All",
    "nc_long":         "NonComm_Positions_Long_All",
    "nc_short":        "NonComm_Positions_Short_All",
    "nc_spread":       "NonComm_Positions_Spreading_All",
    "comm_long":       "Comm_Positions_Long_All",
    "comm_short":      "Comm_Positions_Short_All",
    "nonrep_long":     "NonRept_Positions_Long_All",
    "nonrep_short":    "NonRept_Positions_Short_All",
    "oi_change":       "Change_in_Open_Interest_All",
    "nc_long_change":  "Change_in_NonComm_Long_All",
    "nc_short_change": "Change_in_NonComm_Short_All",
}

# Annual zip URL templates — CFTC archives
_DISAGG_YEAR_URL = _CFTC_BASE + "/fut_disagg_txt_{year}.zip"
_LEGACY_YEAR_URL = _CFTC_BASE + "/annual_{year}.zip"
_TFF_YEAR_URL    = _CFTC_BASE + "/traders_fin_fut_txt_{year}.zip"

_START_YEARS = {
    "disaggregated": 2006,
    "legacy":        1986,
    "tff":           2010,
}

# ---------------------------------------------------------------------------
# Market catalog — CFTC codes and common names
# ---------------------------------------------------------------------------

_GRAINS = {
    "Corn":          "002602",
    "Soybeans":      "005602",
    "Wheat CBOT":    "001602",
    "Wheat KCBT":    "001612",
    "Soybean Oil":   "007601",
    "Soybean Meal":  "026603",
    "Oats":          "004603",
    "Rough Rice":    "039601",
}

_SOFTS = {
    "Sugar #11":     "080732",
    "Coffee":        "083731",
    "Cocoa":         "073732",
    "Cotton #2":     "033661",
    "Orange Juice":  "040701",
    "Lumber":        "058644",
    "Milk":          "052641",
    "Feeder Cattle": "061641",
    "Live Cattle":   "057642",
    "Lean Hogs":     "054642",
}

_ENERGY = {
    "Crude Oil WTI":  "067651",
    "Natural Gas":    "023651",
    "RBOB Gasoline":  "111659",
    "Heating Oil":    "022651",
    "Brent Crude":    "06765T",
    "Ethanol":        "039601",
    "Propane":        "065517",
}

_METALS = {
    "Gold":           "088691",
    "Silver":         "084691",
    "Copper":         "085692",
    "Platinum":       "076651",
    "Palladium":      "075651",
}

_EQUITY = {
    "S&P 500 E-Mini":     "13874+",
    "NASDAQ-100 E-Mini":  "20974+",
    "Dow Jones E-Mini":   "12460+",
    "Russell 2000 E-Mini":"239742",
    "S&P 500 VIX":        "1170E1",
    "Nikkei 225":         "240741",
    "S&P 500":            "138741",
}

_RATES = {
    "10-Year T-Note":   "043602",
    "5-Year T-Note":    "044601",
    "2-Year T-Note":    "042601",
    "30-Year T-Bond":   "020601",
    "3-Month Eurodollar":"132741",
    "30-Day Fed Funds": "045601",
    "Ultra 10-Year":    "43874+",
    "Ultra T-Bond":     "02074+",
}

_FX = {
    "Euro FX":          "099741",
    "Japanese Yen":     "097741",
    "British Pound":    "096742",
    "Swiss Franc":      "092741",
    "Canadian Dollar":  "090741",
    "Australian Dollar":"232741",
    "New Zealand Dollar":"112741",
    "Mexican Peso":     "095741",
    "Brazilian Real":   "102741",
    "South Korean Won": "146021",
    "Russian Ruble":    "089741",
    "Bitcoin":          "133741",
}

_ALL_MARKETS: Dict[str, str] = {
    **_GRAINS, **_SOFTS, **_ENERGY, **_METALS,
    **_EQUITY, **_RATES, **_FX,
}

_CATEGORY_MAP: Dict[str, Dict[str, str]] = {
    "grains":    _GRAINS,
    "softs":     _SOFTS,
    "energy":    _ENERGY,
    "metals":    _METALS,
    "equity":    _EQUITY,
    "rates":     _RATES,
    "fx":        _FX,
    "livestock": {k: v for k, v in _SOFTS.items()
                  if k in ("Live Cattle", "Lean Hogs", "Feeder Cattle")},
    "commodities": {**_GRAINS, **_SOFTS, **_ENERGY, **_METALS},
    "financial": {**_EQUITY, **_RATES, **_FX},
}

# Alias map — common short names to canonical names
_MARKET_ALIASES: Dict[str, str] = {
    "es":          "S&P 500 E-Mini",
    "sp500":       "S&P 500 E-Mini",
    "s&p":         "S&P 500 E-Mini",
    "spx":         "S&P 500 E-Mini",
    "nq":          "NASDAQ-100 E-Mini",
    "nasdaq":      "NASDAQ-100 E-Mini",
    "ym":          "Dow Jones E-Mini",
    "dow":         "Dow Jones E-Mini",
    "rty":         "Russell 2000 E-Mini",
    "russell":     "Russell 2000 E-Mini",
    "vix":         "S&P 500 VIX",
    "zn":          "10-Year T-Note",
    "10y":         "10-Year T-Note",
    "zt":          "2-Year T-Note",
    "zf":          "5-Year T-Note",
    "zb":          "30-Year T-Bond",
    "gc":          "Gold",
    "gold":        "Gold",
    "si":          "Silver",
    "silver":      "Silver",
    "hg":          "Copper",
    "copper":      "Copper",
    "cl":          "Crude Oil WTI",
    "crude":       "Crude Oil WTI",
    "oil":         "Crude Oil WTI",
    "ng":          "Natural Gas",
    "gas":         "Natural Gas",
    "rb":          "RBOB Gasoline",
    "ho":          "Heating Oil",
    "eur":         "Euro FX",
    "euro":        "Euro FX",
    "6e":          "Euro FX",
    "jpy":         "Japanese Yen",
    "6j":          "Japanese Yen",
    "gbp":         "British Pound",
    "6b":          "British Pound",
    "chf":         "Swiss Franc",
    "6s":          "Swiss Franc",
    "cad":         "Canadian Dollar",
    "6c":          "Canadian Dollar",
    "aud":         "Australian Dollar",
    "6a":          "Australian Dollar",
    "nzd":         "New Zealand Dollar",
    "mxp":         "Mexican Peso",
    "brl":         "Brazilian Real",
    "corn":        "Corn",
    "zc":          "Corn",
    "soybeans":    "Soybeans",
    "zs":          "Soybeans",
    "wheat":       "Wheat CBOT",
    "zw":          "Wheat CBOT",
    "sugar":       "Sugar #11",
    "coffee":      "Coffee",
    "cocoa":       "Cocoa",
    "cotton":      "Cotton #2",
    "oj":          "Orange Juice",
    "cattle":      "Live Cattle",
    "hogs":        "Lean Hogs",
}


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_path(report_type: str, year: Optional[int] = None,
                suffix: str = "parquet") -> Path:
    if year:
        return _CACHE_DIR / f"cot_{report_type}_{year}.{suffix}"
    return _CACHE_DIR / f"cot_{report_type}_latest.{suffix}"


def _try_read_cache(path: Path, max_age_s: int = _CACHE_TTL_SECONDS) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > max_age_s:
        return None
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.warning("Cache read failed %s: %s", path, exc)
        return None


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    _ensure_cache_dir()
    try:
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.debug("Cached %d rows → %s", len(df), path)
    except Exception as exc:
        logger.warning("Cache write failed %s: %s", path, exc)


def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return first candidate column name that exists in df (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PositionChange:
    """Week-over-week significant position change."""
    market:             str
    trader_type:        str
    prior_net:          float
    current_net:        float
    change_pct:         float
    is_significant:     bool        # > 20% move in net position
    direction:          str         # "increasing_long", "increasing_short", "flipping_long", "flipping_short"
    as_of:              str
    alert_text:         str = ""


@dataclass
class COTSignal:
    """Full COT signal bundle for a single market."""
    market:                 str
    cftc_code:              str
    as_of:                  str
    report_type:            str
    # Net positions
    mm_net:                 float = 0.0   # managed money (large specs)
    comm_net:               float = 0.0   # commercials / hedgers
    nonrep_net:             float = 0.0   # small specs
    open_interest:          float = 0.0
    # COT Index
    cot_index:              float = np.nan   # 0–100, 52-week window
    cot_index_52w:          float = np.nan
    cot_index_5yr:          float = np.nan
    # Signals
    extreme_signal:         str = "NEUTRAL"  # EXTREME_LONG / EXTREME_SHORT / NEUTRAL
    commercial_signal:      str = "NEUTRAL"  # smart-money direction
    speculator_signal:      str = "NEUTRAL"  # crowding
    risk_appetite:          str = "NEUTRAL"
    # Scores
    contrarian_score:       float = 0.0   # 0–100
    crowding_score:         float = 0.0   # 0–100
    # OI trend
    oi_trend:               float = 0.0   # 4-week OI change %
    # Summary
    narrative:              str = ""


@dataclass
class COTAlert:
    """Positioning extreme or reversal alert."""
    market:         str
    alert_type:     str   # "EXTREME_LONG", "EXTREME_SHORT", "REVERSAL_SIGNAL", "CROWDING"
    severity:       str   # "HIGH", "MEDIUM", "LOW"
    cot_index:      float
    description:    str
    as_of:          str
    contrarian:     bool = False


# ---------------------------------------------------------------------------
# COTDataDownloader
# ---------------------------------------------------------------------------

class COTDataDownloader:
    """
    Download and parse CFTC Commitments of Traders reports.

    Supports three report types:
      - 'disaggregated' : Disaggregated Futures & Options (best coverage, 2006+)
      - 'legacy'        : Legacy Futures Only (broadest history, 1986+)
      - 'tff'           : Traders in Financial Futures (financial markets, 2010+)

    All data from CFTC.gov public files — no API key required.
    """

    _REPORT_URLS = {
        "disaggregated": _CFTC_LATEST_DISAGG,
        "legacy":        _CFTC_LATEST_LEGACY,
        "tff":           _CFTC_LATEST_TFF,
    }
    _YEAR_URL_TEMPLATES = {
        "disaggregated": _DISAGG_YEAR_URL,
        "legacy":        _LEGACY_YEAR_URL,
        "tff":           _TFF_YEAR_URL,
    }

    def __init__(self, cache_dir: Optional[Path] = None, timeout: int = _TIMEOUT):
        self.cache_dir = cache_dir or _CACHE_DIR
        self.timeout = timeout
        _ensure_cache_dir()

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, url: str) -> bytes:
        """Download bytes from URL with retry."""
        for attempt in range(3):
            try:
                r = requests.get(url, headers=_HEADERS, timeout=self.timeout)
                r.raise_for_status()
                return r.content
            except requests.RequestException as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning("GET %s failed (attempt %d): %s — retrying in %ds",
                               url, attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"All retries exhausted for {url}")

    def _get_zip_csv(self, url: str) -> str:
        """Download zip and return content of the first .txt/.csv file."""
        raw = self._get(url)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            txt_files = [n for n in names if n.lower().endswith((".txt", ".csv"))]
            if not txt_files:
                raise ValueError(f"No .txt/.csv in zip from {url}. Files: {names}")
            # Prefer the file with "all" in the name, else first
            target = next((n for n in txt_files if "all" in n.lower()), txt_files[0])
            with zf.open(target) as fh:
                return fh.read().decode("latin-1", errors="replace")

    # ------------------------------------------------------------------
    # Public: fetch latest weekly data
    # ------------------------------------------------------------------

    def fetch_latest_cot(self, report_type: str = "disaggregated") -> pd.DataFrame:
        """
        Download the current week's COT file from CFTC.gov.

        Returns parsed DataFrame with standardised columns.
        Cache result for _CACHE_TTL_SECONDS to avoid repeated downloads.
        """
        report_type = report_type.lower()
        if report_type not in self._REPORT_URLS:
            raise ValueError(f"Unknown report_type: {report_type}. "
                             f"Choose from {list(self._REPORT_URLS)}")

        cache_path = _cache_path(report_type, suffix="parquet")
        cached = _try_read_cache(cache_path, max_age_s=_CACHE_TTL_SECONDS)
        if cached is not None:
            logger.info("Latest COT %s from cache (%d rows)", report_type, len(cached))
            return cached

        url = self._REPORT_URLS[report_type]
        logger.info("Downloading latest COT %s from %s", report_type, url)
        try:
            content = self._get(url).decode("latin-1", errors="replace")
        except Exception as exc:
            logger.error("Failed to download latest COT: %s", exc)
            raise

        df = self.parse_cot_csv(content, report_type=report_type)
        _write_cache(df, cache_path)
        logger.info("Latest COT %s: %d rows, date range %s",
                    report_type, len(df), df["as_of_date"].max() if "as_of_date" in df.columns else "?")
        return df

    # ------------------------------------------------------------------
    # Public: fetch historical annual data
    # ------------------------------------------------------------------

    def fetch_historical_cot(self, year: int,
                             report_type: str = "disaggregated") -> pd.DataFrame:
        """
        Download a specific annual COT file from CFTC historical archive.

        Files are in zip format containing one large CSV/TXT.
        Results cached as parquet per year.
        """
        report_type = report_type.lower()
        cache_p = _cache_path(report_type, year=year, suffix="parquet")
        cached = _try_read_cache(cache_p, max_age_s=365 * 86400)  # historical won't change
        if cached is not None:
            return cached

        tmpl = self._YEAR_URL_TEMPLATES.get(report_type)
        if not tmpl:
            raise ValueError(f"Unknown report_type: {report_type}")

        url = tmpl.format(year=year)
        logger.info("Downloading COT %s year %d from %s", report_type, year, url)
        try:
            content = self._get_zip_csv(url)
        except Exception as exc:
            logger.warning("Could not fetch COT %s %d: %s", report_type, year, exc)
            return pd.DataFrame()

        df = self.parse_cot_csv(content, report_type=report_type)
        if not df.empty:
            _write_cache(df, cache_p)
        return df

    # ------------------------------------------------------------------
    # Public: build full multi-year history
    # ------------------------------------------------------------------

    def fetch_full_history(self, start_year: Optional[int] = None,
                           report_type: str = "disaggregated") -> pd.DataFrame:
        """
        Assemble full COT history by downloading every annual file since start_year.

        start_year defaults to the first available year for each report type.
        Returns combined DataFrame sorted by as_of_date.
        """
        report_type = report_type.lower()
        if start_year is None:
            start_year = _START_YEARS.get(report_type, 2006)

        current_year = datetime.now().year
        frames: List[pd.DataFrame] = []

        for yr in range(start_year, current_year):
            df_yr = self.fetch_historical_cot(yr, report_type=report_type)
            if not df_yr.empty:
                frames.append(df_yr)

        # Always add latest (current year)
        try:
            df_latest = self.fetch_latest_cot(report_type=report_type)
            if not df_latest.empty:
                frames.append(df_latest)
        except Exception as exc:
            logger.warning("Could not fetch latest COT: %s", exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        if "as_of_date" in combined.columns:
            combined["as_of_date"] = pd.to_datetime(combined["as_of_date"], errors="coerce")
            combined = combined.sort_values("as_of_date").drop_duplicates(
                subset=["as_of_date", "cftc_code"], keep="last"
            )
        return combined

    # ------------------------------------------------------------------
    # Public: CSV parser
    # ------------------------------------------------------------------

    def parse_cot_csv(self, content: str,
                      report_type: str = "disaggregated") -> pd.DataFrame:
        """
        Parse CFTC COT text/CSV content.

        CFTC files are comma-delimited CSVs with a header row.
        Normalises column names and computes net position fields.
        """
        try:
            df = pd.read_csv(io.StringIO(content), low_memory=False)
        except Exception as exc:
            logger.error("CSV parse error: %s", exc)
            return pd.DataFrame()

        if df.empty:
            return df

        # Normalise column names — strip spaces
        df.columns = [c.strip() for c in df.columns]

        # Find date column (varies across report vintages)
        date_col = _find_col(df,
            "As_of_Date_In_Form_YYYY-MM-DD",
            "As of Date in Form YYYY-MM-DD",
            "Report_Date_as_YYYY-MM-DD",
            "YYYY-MM-DD",
        )
        if date_col:
            df = df.rename(columns={date_col: "as_of_date"})

        # Find market name column
        mkt_col = _find_col(df, "Market_and_Exchange_Names", "Market and Exchange Names")
        if mkt_col and mkt_col != "market_name":
            df = df.rename(columns={mkt_col: "market_name"})

        # Find CFTC code column
        code_col = _find_col(df, "CFTC_Commodity_Code", "CFTC Commodity Code",
                             "CFTC_Market_Code", "Commodity_Code")
        if code_col and code_col != "cftc_code":
            df = df.rename(columns={code_col: "cftc_code"})

        # Standardise OI
        oi_col = _find_col(df, "Open_Interest_All", "Open Interest (All)")
        if oi_col and oi_col != "open_interest":
            df = df.rename(columns={oi_col: "open_interest"})

        # Report-type-specific column normalisation
        if report_type == "disaggregated":
            df = self._normalise_disagg(df)
        elif report_type == "legacy":
            df = self._normalise_legacy(df)
        elif report_type == "tff":
            df = self._normalise_tff(df)

        # Ensure as_of_date is datetime
        if "as_of_date" in df.columns:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")

        df["report_type"] = report_type
        return df

    # ------------------------------------------------------------------
    # Internal normalisers
    # ------------------------------------------------------------------

    def _normalise_disagg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise disaggregated report columns to standard names."""
        renames: Dict[str, str] = {}
        # Managed money (large specs)
        for old, new in [
            ("M_Money_Positions_Long_All",  "mm_long"),
            ("M_Money_Positions_Short_All", "mm_short"),
            ("M_Money_Positions_Spread_All","mm_spread"),
            # Producer/merchant = commercials
            ("Prod_Merc_Positions_Long_All",  "comm_long"),
            ("Prod_Merc_Positions_Short_All", "comm_short"),
            # Swap dealers
            ("Swap_Positions_Long_All",      "swap_long"),
            ("Swap_Positions_Short_All",     "swap_short"),
            ("Swap__Positions_Spread_All",   "swap_spread"),
            # Other reportable
            ("Other_Rept_Positions_Long_All",  "other_long"),
            ("Other_Rept_Positions_Short_All", "other_short"),
            # Non-reportable (small specs)
            ("NonRept_Positions_Long_All",  "nonrep_long"),
            ("NonRept_Positions_Short_All", "nonrep_short"),
            # Changes
            ("Change_in_Open_Interest_All",  "oi_change"),
            ("Change_in_M_Money_Long_All",   "mm_long_change"),
            ("Change_in_M_Money_Short_All",  "mm_short_change"),
            ("Change_in_Prod_Merc_Long_All", "comm_long_change"),
            ("Change_in_Prod_Merc_Short_All","comm_short_change"),
        ]:
            col = _find_col(df, old)
            if col:
                renames[col] = new

        df = df.rename(columns=renames)

        for col in ["mm_long","mm_short","comm_long","comm_short",
                    "swap_long","swap_short","other_long","other_short",
                    "nonrep_long","nonrep_short","open_interest"]:
            if col in df.columns:
                df[col] = _safe_num(df[col])
            else:
                df[col] = 0.0

        # Compute nets
        df["mm_net"]     = df.get("mm_long", 0) - df.get("mm_short", 0)
        df["comm_net"]   = df.get("comm_long", 0) - df.get("comm_short", 0)
        df["swap_net"]   = df.get("swap_long", 0) - df.get("swap_short", 0)
        df["other_net"]  = df.get("other_long", 0) - df.get("other_short", 0)
        df["nonrep_net"] = df.get("nonrep_long", 0) - df.get("nonrep_short", 0)

        return df

    def _normalise_legacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise legacy COT columns (non-commercial = large specs)."""
        renames: Dict[str, str] = {}
        for old, new in [
            ("NonComm_Positions_Long_All",       "mm_long"),
            ("NonComm_Positions_Short_All",      "mm_short"),
            ("NonComm_Positions_Spreading_All",  "mm_spread"),
            ("Comm_Positions_Long_All",          "comm_long"),
            ("Comm_Positions_Short_All",         "comm_short"),
            ("NonRept_Positions_Long_All",       "nonrep_long"),
            ("NonRept_Positions_Short_All",      "nonrep_short"),
            ("Change_in_Open_Interest_All",      "oi_change"),
            ("Change_in_NonComm_Long_All",       "mm_long_change"),
            ("Change_in_NonComm_Short_All",      "mm_short_change"),
        ]:
            col = _find_col(df, old)
            if col:
                renames[col] = new

        df = df.rename(columns=renames)

        for col in ["mm_long","mm_short","comm_long","comm_short",
                    "nonrep_long","nonrep_short","open_interest"]:
            if col in df.columns:
                df[col] = _safe_num(df[col])
            else:
                df[col] = 0.0

        df["mm_net"]     = df.get("mm_long", 0) - df.get("mm_short", 0)
        df["comm_net"]   = df.get("comm_long", 0) - df.get("comm_short", 0)
        df["nonrep_net"] = df.get("nonrep_long", 0) - df.get("nonrep_short", 0)
        # No swap dealer breakdown in legacy
        df["swap_net"]  = 0.0
        df["other_net"] = 0.0

        return df

    def _normalise_tff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise Traders in Financial Futures columns."""
        renames: Dict[str, str] = {}
        for old, new in [
            # Dealer intermediary → use as comm proxy
            ("Dealer_Positions_Long_All",     "comm_long"),
            ("Dealer_Positions_Short_All",    "comm_short"),
            # Asset manager → large spec / long-only
            ("Asset_Mgr_Positions_Long_All",  "mm_long"),
            ("Asset_Mgr_Positions_Short_All", "mm_short"),
            # Leveraged funds → hedge fund proxy
            ("Lev_Money_Positions_Long_All",  "lev_long"),
            ("Lev_Money_Positions_Short_All", "lev_short"),
            ("Other_Rept_Positions_Long_All", "other_long"),
            ("Other_Rept_Positions_Short_All","other_short"),
            ("NonRept_Positions_Long_All",    "nonrep_long"),
            ("NonRept_Positions_Short_All",   "nonrep_short"),
        ]:
            col = _find_col(df, old)
            if col:
                renames[col] = new

        df = df.rename(columns=renames)

        for col in ["mm_long","mm_short","comm_long","comm_short",
                    "lev_long","lev_short","other_long","other_short",
                    "nonrep_long","nonrep_short","open_interest"]:
            if col in df.columns:
                df[col] = _safe_num(df[col])
            else:
                df[col] = 0.0

        df["mm_net"]     = df.get("mm_long", 0) - df.get("mm_short", 0)
        df["comm_net"]   = df.get("comm_long", 0) - df.get("comm_short", 0)
        df["lev_net"]    = df.get("lev_long", 0) - df.get("lev_short", 0)
        df["other_net"]  = df.get("other_long", 0) - df.get("other_short", 0)
        df["nonrep_net"] = df.get("nonrep_long", 0) - df.get("nonrep_short", 0)
        df["swap_net"]   = 0.0

        return df


# ---------------------------------------------------------------------------
# COTMarketCoverage
# ---------------------------------------------------------------------------

class COTMarketCoverage:
    """
    Market catalog with CFTC code mapping for 100+ futures markets.

    Provides name→code lookup, category browsing, and alias resolution.
    """

    def get_market_code(self, market_name: str) -> Optional[str]:
        """Map common market name or alias to CFTC code."""
        # Exact match first
        if market_name in _ALL_MARKETS:
            return _ALL_MARKETS[market_name]
        # Alias lookup (case-insensitive)
        alias_key = market_name.lower().strip()
        if alias_key in _MARKET_ALIASES:
            canonical = _MARKET_ALIASES[alias_key]
            return _ALL_MARKETS.get(canonical)
        # Partial match
        for name, code in _ALL_MARKETS.items():
            if market_name.lower() in name.lower():
                return code
        return None

    def get_market_name(self, code: str) -> Optional[str]:
        """Reverse lookup: CFTC code → canonical market name."""
        for name, c in _ALL_MARKETS.items():
            if c == code:
                return name
        return None

    def resolve_market(self, market_input: str) -> Tuple[str, str]:
        """
        Resolve any market input (name, alias, code) to (canonical_name, cftc_code).
        Raises ValueError if not found.
        """
        # Could be a CFTC code already
        if market_input in _ALL_MARKETS.values():
            name = self.get_market_name(market_input) or market_input
            return name, market_input

        code = self.get_market_code(market_input)
        if code is None:
            raise ValueError(f"Unknown market: '{market_input}'. "
                             f"Use list_all_markets() to see available markets.")
        name = self.get_market_name(code) or market_input
        return name, code

    def list_all_markets(self) -> List[str]:
        """Return sorted list of all canonical market names."""
        return sorted(_ALL_MARKETS.keys())

    def get_market_by_category(self, category: str) -> List[str]:
        """
        Return market names for a given category.

        Categories: grains, softs, energy, metals, equity, rates, fx,
                    livestock, commodities, financial
        """
        cat = category.lower()
        if cat not in _CATEGORY_MAP:
            raise ValueError(f"Unknown category: '{cat}'. "
                             f"Available: {sorted(_CATEGORY_MAP.keys())}")
        return sorted(_CATEGORY_MAP[cat].keys())

    def list_categories(self) -> List[str]:
        return sorted(_CATEGORY_MAP.keys())

    def get_cftc_codes_for_category(self, category: str) -> Dict[str, str]:
        """Return {market_name: cftc_code} for a category."""
        cat = category.lower()
        return dict(_CATEGORY_MAP.get(cat, {}))


# ---------------------------------------------------------------------------
# COTSignalEngine
# ---------------------------------------------------------------------------

class COTSignalEngine:
    """
    Compute trading signals from COT position data.

    Implements COT Index, extremes, hedger signals, speculator crowding,
    position change detection, and OI trend analysis.
    """

    def __init__(self, history_df: Optional[pd.DataFrame] = None):
        """
        history_df: full historical COT DataFrame (all markets, all dates).
        If None, signals requiring history will return NaN / NEUTRAL.
        """
        self._history = history_df
        self._coverage = COTMarketCoverage()

    def _get_market_history(self, market: str,
                            trader_type: str = "mm") -> Optional[pd.Series]:
        """
        Extract net position series for a market from history_df.

        Returns pd.Series indexed by as_of_date, sorted ascending.
        """
        if self._history is None or self._history.empty:
            return None

        name, code = self._coverage.resolve_market(market)

        # Filter by CFTC code
        mask = self._history["cftc_code"].astype(str).str.strip() == code.strip()
        sub = self._history[mask].copy()

        if sub.empty:
            # Try by market name substring
            if "market_name" in self._history.columns:
                mask2 = self._history["market_name"].str.contains(
                    name.split()[0], case=False, na=False
                )
                sub = self._history[mask2].copy()

        if sub.empty:
            return None

        net_col = f"{trader_type}_net"
        if net_col not in sub.columns:
            logger.warning("Column %s not found for market %s", net_col, market)
            return None

        sub = sub.sort_values("as_of_date")
        series = pd.to_numeric(sub[net_col], errors="coerce").dropna()
        series.index = pd.to_datetime(sub.loc[series.index, "as_of_date"])
        return series

    def compute_net_position(self, df: pd.DataFrame,
                             trader_type: str = "non_commercial") -> pd.Series:
        """
        Compute net position (long - short) for specified trader type.

        trader_type: "non_commercial" / "mm", "commercial" / "comm",
                     "non_reportable" / "nonrep", "swap", "leveraged"
        """
        _type_map = {
            "non_commercial": "mm",
            "large_spec":     "mm",
            "managed_money":  "mm",
            "mm":             "mm",
            "commercial":     "comm",
            "hedger":         "comm",
            "comm":           "comm",
            "non_reportable": "nonrep",
            "small_spec":     "nonrep",
            "nonrep":         "nonrep",
            "swap":           "swap",
            "leveraged":      "lev",
            "lev":            "lev",
        }
        prefix = _type_map.get(trader_type.lower(), "mm")
        long_col  = f"{prefix}_long"
        short_col = f"{prefix}_short"

        if long_col in df.columns and short_col in df.columns:
            return _safe_num(df[long_col]) - _safe_num(df[short_col])

        # Fall back to pre-computed net col
        net_col = f"{prefix}_net"
        if net_col in df.columns:
            return _safe_num(df[net_col])

        logger.warning("No columns for trader_type=%s in DataFrame", trader_type)
        return pd.Series(dtype=float)

    def compute_cot_index(self, market: str,
                          lookback_weeks: int = 52,
                          trader_type: str = "mm") -> float:
        """
        Compute COT Index for a market over a rolling lookback window.

        COT Index = (current_net - min_net) / (max_net - min_net) × 100
        Range 0–100: 0 = maximum net short, 100 = maximum net long.

        Returns NaN if insufficient history.
        """
        series = self._get_market_history(market, trader_type)
        if series is None or len(series) < 4:
            return float("nan")

        window = series.iloc[-lookback_weeks:]
        current = window.iloc[-1]
        mn = window.min()
        mx = window.max()
        rng = mx - mn
        if rng == 0:
            return 50.0
        return float(np.clip((current - mn) / rng * 100, 0, 100))

    def compute_cot_extremes(self, market: str,
                             threshold: float = 10.0,
                             trader_type: str = "mm") -> str:
        """
        Classify COT positioning as EXTREME_LONG, EXTREME_SHORT, or NEUTRAL.

        EXTREME_LONG  (COT index > 100 - threshold): contrarian bearish
        EXTREME_SHORT (COT index < threshold):        contrarian bullish
        """
        idx = self.compute_cot_index(market, trader_type=trader_type)
        if np.isnan(idx):
            return "NEUTRAL"
        if idx > (100 - threshold):
            return "EXTREME_LONG"
        if idx < threshold:
            return "EXTREME_SHORT"
        return "NEUTRAL"

    def compute_commercial_hedger_signal(self, market: str) -> float:
        """
        Commercial hedger signal (0–100 COT index using comm_net).

        Commercials hedge their underlying exposure, so they are:
          - Net long → underlying prices are low / near bottom (bullish signal)
          - Net short → they are selling into high prices (bearish signal)

        Returns commercial COT index (0=comm max short, 100=comm max long).
        """
        return self.compute_cot_index(market, trader_type="comm")

    def compute_speculator_crowding(self, market: str) -> float:
        """
        Speculator crowding score (COT index for large specs / mm).

        High score (>80) = specs very net long = crowded long = risk of unwind.
        Low score (<20)  = specs very net short = crowded short = squeeze risk.

        Returns COT index 0–100.
        """
        return self.compute_cot_index(market, trader_type="mm")

    def detect_position_change(self, market: str,
                               current_week: pd.Series,
                               prior_week: pd.Series,
                               threshold_pct: float = 20.0,
                               trader_type: str = "mm") -> PositionChange:
        """
        Detect significant week-over-week net position change.

        Significant = net position changed by > threshold_pct% of open interest
        or more than 20% of the prior net (whichever is more sensitive).
        """
        net_col = f"{trader_type}_net"
        prior_net   = float(prior_week.get(net_col, 0) or 0)
        current_net = float(current_week.get(net_col, 0) or 0)
        as_of = str(current_week.get("as_of_date", "unknown"))

        if prior_net == 0:
            change_pct = 0.0
        else:
            change_pct = (current_net - prior_net) / abs(prior_net) * 100

        is_significant = abs(change_pct) > threshold_pct

        # Determine direction
        if current_net > 0 and prior_net > 0:
            direction = "increasing_long" if current_net > prior_net else "decreasing_long"
        elif current_net < 0 and prior_net < 0:
            direction = "increasing_short" if current_net < prior_net else "decreasing_short"
        elif current_net > 0 and prior_net <= 0:
            direction = "flipping_long"
            is_significant = True
        elif current_net < 0 and prior_net >= 0:
            direction = "flipping_short"
            is_significant = True
        else:
            direction = "flat"

        alert_text = ""
        if is_significant:
            alert_text = (
                f"{market}: {trader_type.upper()} net {direction} "
                f"({prior_net:+,.0f} → {current_net:+,.0f}, "
                f"{change_pct:+.1f}% change)"
            )

        return PositionChange(
            market=market,
            trader_type=trader_type,
            prior_net=prior_net,
            current_net=current_net,
            change_pct=change_pct,
            is_significant=is_significant,
            direction=direction,
            as_of=as_of,
            alert_text=alert_text,
        )

    def compute_open_interest_trend(self, market: str,
                                    n_weeks: int = 4) -> float:
        """
        Compute n-week OI % change.

        Rising OI + price up  = strong uptrend (confirms rally)
        Rising OI + price dn  = strong downtrend (confirms selloff)
        Falling OI + movement = trend exhaustion / unwinding

        Returns pct change in OI over n_weeks (can be negative).
        """
        if self._history is None or self._history.empty:
            return 0.0

        name, code = self._coverage.resolve_market(market)
        mask = self._history["cftc_code"].astype(str).str.strip() == code.strip()
        sub = self._history[mask].sort_values("as_of_date")

        if "open_interest" not in sub.columns or len(sub) < n_weeks + 1:
            return 0.0

        oi = _safe_num(sub["open_interest"])
        if len(oi) < n_weeks + 1:
            return 0.0

        prior = float(oi.iloc[-(n_weeks + 1)])
        current = float(oi.iloc[-1])
        if prior == 0:
            return 0.0
        return (current - prior) / abs(prior) * 100

    def build_signal(self, market: str,
                     current_row: Optional[pd.Series] = None) -> COTSignal:
        """
        Build a full COTSignal for a market using available history.
        """
        try:
            name, code = self._coverage.resolve_market(market)
        except ValueError:
            name, code = market, "UNKNOWN"

        # COT indexes
        mm_idx   = self.compute_cot_index(market, lookback_weeks=52, trader_type="mm")
        comm_idx = self.compute_commercial_hedger_signal(market)

        # Signals
        extreme = self.compute_cot_extremes(market, threshold=10.0)
        crowding_score = mm_idx if not np.isnan(mm_idx) else 50.0
        contrarian_score = 0.0

        # Contrarian score: how extreme relative to threshold
        if not np.isnan(mm_idx):
            if mm_idx > 90:
                contrarian_score = mm_idx - 90  # 0-10
            elif mm_idx < 10:
                contrarian_score = 10 - mm_idx  # 0-10
            contrarian_score = min(contrarian_score * 10, 100)

        # Commercial hedger signal
        comm_signal = "NEUTRAL"
        if not np.isnan(comm_idx):
            if comm_idx > 70:
                comm_signal = "BULLISH_HEDGER"   # hedgers net long = underpriced asset
            elif comm_idx < 30:
                comm_signal = "BEARISH_HEDGER"   # hedgers net short = overpriced

        # Spec signal
        spec_signal = "NEUTRAL"
        if not np.isnan(mm_idx):
            if mm_idx > 80:
                spec_signal = "CROWDED_LONG"     # risk of unwind
            elif mm_idx < 20:
                spec_signal = "CROWDED_SHORT"    # squeeze risk

        # OI trend
        oi_trend = self.compute_open_interest_trend(market, n_weeks=4)

        # Extract current values from row if provided
        mm_net = comm_net = nonrep_net = oi = 0.0
        as_of = datetime.now().strftime("%Y-%m-%d")
        if current_row is not None:
            mm_net     = float(current_row.get("mm_net", 0) or 0)
            comm_net   = float(current_row.get("comm_net", 0) or 0)
            nonrep_net = float(current_row.get("nonrep_net", 0) or 0)
            oi         = float(current_row.get("open_interest", 0) or 0)
            if "as_of_date" in current_row.index:
                as_of = str(current_row["as_of_date"])[:10]

        # Narrative
        parts = []
        if extreme == "EXTREME_LONG":
            parts.append(f"Specs at EXTREME LONG ({mm_idx:.0f}/100) — contrarian bearish")
        elif extreme == "EXTREME_SHORT":
            parts.append(f"Specs at EXTREME SHORT ({mm_idx:.0f}/100) — contrarian bullish")
        if comm_signal != "NEUTRAL":
            parts.append(f"Commercial signal: {comm_signal}")
        if oi_trend > 5:
            parts.append(f"OI rising +{oi_trend:.1f}% (4w) — trend confirmation")
        elif oi_trend < -5:
            parts.append(f"OI falling {oi_trend:.1f}% (4w) — potential exhaustion")

        return COTSignal(
            market=name,
            cftc_code=code,
            as_of=as_of,
            report_type="disaggregated",
            mm_net=mm_net,
            comm_net=comm_net,
            nonrep_net=nonrep_net,
            open_interest=oi,
            cot_index=mm_idx,
            cot_index_52w=mm_idx,
            cot_index_5yr=self.compute_cot_index(market, lookback_weeks=260),
            extreme_signal=extreme,
            commercial_signal=comm_signal,
            speculator_signal=spec_signal,
            contrarian_score=contrarian_score,
            crowding_score=crowding_score,
            oi_trend=oi_trend,
            narrative=" | ".join(parts) if parts else "No extreme positioning detected",
        )

    def compute_cot_index_series(self, market: str,
                                 lookback_weeks: int = 52,
                                 trader_type: str = "mm") -> pd.Series:
        """
        Compute rolling COT Index time series.

        Returns pd.Series of COT Index values (0-100) indexed by date.
        Useful for charting and historical backtesting.
        """
        series = self._get_market_history(market, trader_type)
        if series is None or len(series) < lookback_weeks:
            return pd.Series(dtype=float)

        def _idx(window: pd.Series) -> float:
            mn, mx = window.min(), window.max()
            rng = mx - mn
            if rng == 0:
                return 50.0
            return float(np.clip((window.iloc[-1] - mn) / rng * 100, 0, 100))

        return series.rolling(window=lookback_weeks, min_periods=max(4, lookback_weeks // 4)).apply(
            lambda w: _idx(pd.Series(w)), raw=False
        )

    def rank_markets_by_cot_index(self, markets: List[str],
                                  trader_type: str = "mm") -> pd.DataFrame:
        """
        Rank a list of markets by COT index (descending).

        Returns DataFrame with columns: market, cot_index, extreme_signal.
        """
        rows = []
        for mkt in markets:
            try:
                idx = self.compute_cot_index(mkt, trader_type=trader_type)
                ext = self.compute_cot_extremes(mkt, trader_type=trader_type)
                rows.append({"market": mkt, "cot_index": idx, "extreme_signal": ext})
            except Exception:
                pass

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("cot_index", ascending=False)
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# COTPortfolioAnalyzer
# ---------------------------------------------------------------------------

class COTPortfolioAnalyzer:
    """
    Portfolio-level COT analysis: dashboards, risk appetite, crowding.
    """

    def __init__(self, signal_engine: COTSignalEngine):
        self._engine = signal_engine
        self._coverage = COTMarketCoverage()

    def _build_dashboard(self, category: str) -> pd.DataFrame:
        """Build COT dashboard for all markets in a category."""
        markets = self._coverage.get_market_by_category(category)
        rows = []
        for mkt in markets:
            try:
                mm_idx   = self._engine.compute_cot_index(mkt, trader_type="mm")
                comm_idx = self._engine.compute_cot_index(mkt, trader_type="comm")
                extreme  = self._engine.compute_cot_extremes(mkt)
                oi_trend = self._engine.compute_open_interest_trend(mkt)
                rows.append({
                    "market":        mkt,
                    "mm_cot_index":  round(mm_idx, 1) if not np.isnan(mm_idx) else None,
                    "comm_cot_index":round(comm_idx, 1) if not np.isnan(comm_idx) else None,
                    "extreme":       extreme,
                    "oi_trend_4w":   round(oi_trend, 2),
                    "spec_bias":     "LONG" if not np.isnan(mm_idx) and mm_idx > 55
                                     else "SHORT" if not np.isnan(mm_idx) and mm_idx < 45
                                     else "NEUTRAL",
                })
            except Exception as exc:
                logger.debug("Dashboard skipping %s: %s", mkt, exc)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("mm_cot_index",
                                              ascending=False, na_position="last")

    def get_commodity_positioning_dashboard(self) -> pd.DataFrame:
        """
        COT dashboard for all commodity markets.

        Includes grains, energy, metals, softs.
        """
        frames = []
        for cat in ["grains", "energy", "metals", "softs"]:
            df = self._build_dashboard(cat)
            if not df.empty:
                df["category"] = cat
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_financial_positioning_dashboard(self) -> pd.DataFrame:
        """
        COT dashboard for financial futures: equity, rates, FX.
        """
        frames = []
        for cat in ["equity", "rates", "fx"]:
            df = self._build_dashboard(cat)
            if not df.empty:
                df["category"] = cat
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def compute_risk_appetite_index(self) -> float:
        """
        Risk appetite composite (0–100).

        Combines:
          - S&P 500 E-Mini speculator positioning (40% weight)
          - VIX positioning — inverse (20% weight)
          - 10-Year T-Note positioning — inverse for risk appetite (20% weight)
          - EUR/USD speculator positioning (20% weight)

        > 60 = RISK_ON
        < 40 = RISK_OFF
        40–60 = NEUTRAL
        """
        components: List[Tuple[str, float, float]] = [
            ("S&P 500 E-Mini",   0.40, 1.0),   # high spec long = risk on
            ("S&P 500 VIX",     0.20, -1.0),   # high VIX spec long = risk off (inverse)
            ("10-Year T-Note",  0.20, -1.0),   # high T-Note spec long = risk off (flight-to-quality)
            ("Euro FX",         0.20, 1.0),    # EUR long = USD bearish = risk on
        ]

        weighted_sum = 0.0
        total_weight = 0.0

        for market, weight, direction in components:
            try:
                idx = self._engine.compute_cot_index(market, trader_type="mm")
                if not np.isnan(idx):
                    # Direction: +1 means high index = risk on, -1 means inverse
                    adjusted = idx if direction > 0 else (100 - idx)
                    weighted_sum += adjusted * weight
                    total_weight += weight
            except Exception:
                pass

        if total_weight == 0:
            return 50.0

        return float(np.clip(weighted_sum / total_weight, 0, 100))

    def get_most_crowded_longs(self, n: int = 10) -> pd.DataFrame:
        """
        Return n markets with highest COT index (most crowded long positioning).
        """
        all_markets = self._coverage.list_all_markets()
        df = self._engine.rank_markets_by_cot_index(all_markets, trader_type="mm")
        if df.empty:
            return df
        return df.head(n)

    def get_most_extreme_shorts(self, n: int = 10) -> pd.DataFrame:
        """
        Return n markets with lowest COT index (most crowded short / extreme bearish).
        """
        all_markets = self._coverage.list_all_markets()
        df = self._engine.rank_markets_by_cot_index(all_markets, trader_type="mm")
        if df.empty:
            return df
        return df.tail(n).sort_values("cot_index")

    def compute_commodity_sector_bias(self) -> Dict[str, Any]:
        """
        Aggregate COT positioning by commodity sector.

        Returns dict with sector → {avg_cot_index, bias, markets_extreme_long,
                                    markets_extreme_short}
        """
        result: Dict[str, Any] = {}
        for sector in ["grains", "energy", "metals", "softs", "livestock"]:
            markets = self._coverage.get_market_by_category(sector)
            indices = []
            extreme_longs = []
            extreme_shorts = []
            for mkt in markets:
                try:
                    idx = self._engine.compute_cot_index(mkt, trader_type="mm")
                    if not np.isnan(idx):
                        indices.append(idx)
                        if idx > 80:
                            extreme_longs.append(mkt)
                        elif idx < 20:
                            extreme_shorts.append(mkt)
                except Exception:
                    pass

            if indices:
                avg = float(np.mean(indices))
                bias = "BULLISH" if avg > 60 else "BEARISH" if avg < 40 else "NEUTRAL"
            else:
                avg = float("nan")
                bias = "INSUFFICIENT_DATA"

            result[sector] = {
                "avg_cot_index":       round(avg, 1) if not np.isnan(avg) else None,
                "bias":                bias,
                "markets_extreme_long": extreme_longs,
                "markets_extreme_short":extreme_shorts,
                "n_markets":           len(indices),
            }

        return result

    def get_full_dashboard(self) -> pd.DataFrame:
        """Return combined commodity + financial COT dashboard."""
        comm = self.get_commodity_positioning_dashboard()
        fin  = self.get_financial_positioning_dashboard()
        if comm.empty and fin.empty:
            return pd.DataFrame()
        return pd.concat([comm, fin], ignore_index=True)

    def get_risk_appetite_label(self) -> str:
        """Return RISK_ON / RISK_OFF / NEUTRAL label."""
        score = self.compute_risk_appetite_index()
        if score > 60:
            return f"RISK_ON ({score:.1f})"
        if score < 40:
            return f"RISK_OFF ({score:.1f})"
        return f"NEUTRAL ({score:.1f})"


# ---------------------------------------------------------------------------
# COTAlertSystem
# ---------------------------------------------------------------------------

class COTAlertSystem:
    """
    Generate COT positioning alerts for extreme readings and reversals.
    """

    def __init__(self, signal_engine: COTSignalEngine,
                 extreme_threshold: float = 10.0,
                 reversal_threshold: float = 20.0):
        self._engine = signal_engine
        self._coverage = COTMarketCoverage()
        self._extreme_threshold = extreme_threshold
        self._reversal_threshold = reversal_threshold

    def check_extreme_alerts(self,
                             markets: Optional[List[str]] = None) -> List[COTAlert]:
        """
        Check all (or specified) markets for extreme COT positioning.

        Returns list of COTAlert objects sorted by severity.
        """
        if markets is None:
            markets = self._coverage.list_all_markets()

        alerts: List[COTAlert] = []

        for mkt in markets:
            try:
                mm_idx = self._engine.compute_cot_index(mkt, trader_type="mm")
                if np.isnan(mm_idx):
                    continue

                as_of = datetime.now().strftime("%Y-%m-%d")

                if mm_idx > (100 - self._extreme_threshold):
                    severity = "HIGH" if mm_idx > 95 else "MEDIUM"
                    alerts.append(COTAlert(
                        market=mkt,
                        alert_type="EXTREME_LONG",
                        severity=severity,
                        cot_index=mm_idx,
                        description=(
                            f"{mkt}: Large specs at EXTREME LONG ({mm_idx:.1f}/100). "
                            f"Contrarian bearish signal — crowded positioning risk."
                        ),
                        as_of=as_of,
                        contrarian=True,
                    ))

                elif mm_idx < self._extreme_threshold:
                    severity = "HIGH" if mm_idx < 5 else "MEDIUM"
                    alerts.append(COTAlert(
                        market=mkt,
                        alert_type="EXTREME_SHORT",
                        severity=severity,
                        cot_index=mm_idx,
                        description=(
                            f"{mkt}: Large specs at EXTREME SHORT ({mm_idx:.1f}/100). "
                            f"Contrarian bullish signal — short squeeze risk."
                        ),
                        as_of=as_of,
                        contrarian=True,
                    ))

            except Exception as exc:
                logger.debug("Alert check skipping %s: %s", mkt, exc)

        # Sort: HIGH first, then by extremeness
        alerts.sort(key=lambda a: (
            0 if a.severity == "HIGH" else 1,
            a.cot_index if a.alert_type == "EXTREME_SHORT" else (100 - a.cot_index),
        ))

        return alerts

    def check_reversal_signals(self,
                               markets: Optional[List[str]] = None) -> List[COTAlert]:
        """
        Detect REVERSAL_SIGNAL: position shifted significantly from extreme in one week.

        Uses the most recent two observations from history.
        """
        if self._engine._history is None or self._engine._history.empty:
            return []

        if markets is None:
            markets = self._coverage.list_all_markets()

        alerts: List[COTAlert] = []

        for mkt in markets:
            try:
                name, code = self._coverage.resolve_market(mkt)
                mask = (self._engine._history["cftc_code"]
                        .astype(str).str.strip() == code.strip())
                sub = self._engine._history[mask].sort_values("as_of_date")

                if len(sub) < 3 or "mm_net" not in sub.columns:
                    continue

                # Build rolling COT index for last 3 weeks
                series = pd.to_numeric(sub["mm_net"], errors="coerce").dropna()
                if len(series) < 8:
                    continue

                window = series.iloc[-52:]  # 52-week window for index
                mn, mx = window.min(), window.max()
                rng = mx - mn
                if rng == 0:
                    continue

                prior_net   = float(series.iloc[-2])
                current_net = float(series.iloc[-1])

                prior_idx   = float(np.clip((prior_net   - mn) / rng * 100, 0, 100))
                current_idx = float(np.clip((current_net - mn) / rng * 100, 0, 100))

                # Was extreme last week, now moving toward neutral
                was_extreme_long  = prior_idx > (100 - self._extreme_threshold)
                was_extreme_short = prior_idx < self._extreme_threshold
                moved_from_long   = was_extreme_long  and current_idx < (100 - 30)
                moved_from_short  = was_extreme_short and current_idx > 30

                as_of = str(sub["as_of_date"].iloc[-1])[:10]

                if moved_from_long:
                    alerts.append(COTAlert(
                        market=mkt,
                        alert_type="REVERSAL_SIGNAL",
                        severity="HIGH",
                        cot_index=current_idx,
                        description=(
                            f"{mkt}: REVERSAL from extreme long "
                            f"({prior_idx:.0f} → {current_idx:.0f}). "
                            f"Specs unwinding long positions — bearish follow-through."
                        ),
                        as_of=as_of,
                        contrarian=False,
                    ))

                elif moved_from_short:
                    alerts.append(COTAlert(
                        market=mkt,
                        alert_type="REVERSAL_SIGNAL",
                        severity="HIGH",
                        cot_index=current_idx,
                        description=(
                            f"{mkt}: REVERSAL from extreme short "
                            f"({prior_idx:.0f} → {current_idx:.0f}). "
                            f"Short covering rally — bullish follow-through."
                        ),
                        as_of=as_of,
                        contrarian=False,
                    ))

            except Exception as exc:
                logger.debug("Reversal check skipping %s: %s", mkt, exc)

        return alerts

    def get_weekly_summary(self) -> str:
        """
        Generate a text summary of this week's COT highlights.
        """
        lines = [f"=== COT Weekly Alert Summary ({date.today()}) ===", ""]

        extreme_alerts = self.check_extreme_alerts()
        reversal_alerts = self.check_reversal_signals()

        # High-severity extremes
        high_alerts = [a for a in extreme_alerts if a.severity == "HIGH"]
        if high_alerts:
            lines.append("--- HIGH SEVERITY EXTREMES ---")
            for a in high_alerts[:10]:
                lines.append(f"  [{a.alert_type}] {a.description}")
        else:
            lines.append("--- No high-severity extremes this week ---")

        lines.append("")

        if reversal_alerts:
            lines.append("--- REVERSAL SIGNALS ---")
            for a in reversal_alerts[:5]:
                lines.append(f"  {a.description}")
        else:
            lines.append("--- No reversal signals detected ---")

        lines.append("")
        lines.append(f"Total alerts: {len(extreme_alerts)} extreme, "
                     f"{len(reversal_alerts)} reversal")

        return "\n".join(lines)

    def get_all_alerts(self) -> List[COTAlert]:
        """Return combined extreme + reversal alerts."""
        extremes  = self.check_extreme_alerts()
        reversals = self.check_reversal_signals()
        return extremes + reversals


# ---------------------------------------------------------------------------
# COTEngine — Orchestrator
# ---------------------------------------------------------------------------

class COTEngine:
    """
    Top-level orchestrator for CFTC COT analysis.

    Coordinates data download, signal computation, portfolio analysis,
    and alerting. Entry point for SENTINEL integration.

    Usage:
        engine = COTEngine()
        engine.run_weekly_update()
        dashboard = engine.get_full_dashboard()
        signal = engine.get_signal("Gold")
    """

    def __init__(self, report_type: str = "disaggregated",
                 cache_dir: Optional[Path] = None):
        self.report_type = report_type
        self._downloader = COTDataDownloader(cache_dir=cache_dir)
        self._coverage   = COTMarketCoverage()
        self._history_df: Optional[pd.DataFrame] = None
        self._signal_engine: Optional[COTSignalEngine] = None
        self._portfolio:     Optional[COTPortfolioAnalyzer] = None
        self._alerts:        Optional[COTAlertSystem] = None

    def _ensure_data(self) -> None:
        """Lazy-load data and initialise engines."""
        if self._history_df is not None:
            return
        try:
            self._history_df = self._downloader.fetch_latest_cot(
                report_type=self.report_type
            )
        except Exception as exc:
            logger.error("Could not load COT data: %s", exc)
            self._history_df = pd.DataFrame()

        self._signal_engine = COTSignalEngine(history_df=self._history_df)
        self._portfolio     = COTPortfolioAnalyzer(self._signal_engine)
        self._alerts        = COTAlertSystem(self._signal_engine)

    def run_weekly_update(self) -> pd.DataFrame:
        """
        Download latest COT data and refresh all signals.

        Resets cached data to force a fresh download.
        """
        self._history_df = None  # invalidate cache
        # Remove on-disk latest cache so downloader fetches fresh
        for suffix in ("parquet", "csv"):
            p = _cache_path(self.report_type, suffix=suffix)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

        self._ensure_data()
        logger.info("COT weekly update complete: %d rows",
                    len(self._history_df) if self._history_df is not None else 0)
        return self._history_df or pd.DataFrame()

    def get_signal(self, market: str) -> COTSignal:
        """
        Get a full COTSignal for a single market.
        """
        self._ensure_data()
        assert self._signal_engine is not None

        # Get latest row for this market if history available
        current_row = None
        if self._history_df is not None and not self._history_df.empty:
            try:
                name, code = self._coverage.resolve_market(market)
                mask = (self._history_df["cftc_code"]
                        .astype(str).str.strip() == code.strip())
                sub = self._history_df[mask].sort_values("as_of_date")
                if not sub.empty:
                    current_row = sub.iloc[-1]
            except Exception:
                pass

        return self._signal_engine.build_signal(market, current_row=current_row)

    def get_full_dashboard(self) -> pd.DataFrame:
        """
        Return complete COT dashboard: all markets, all categories.
        """
        self._ensure_data()
        assert self._portfolio is not None
        return self._portfolio.get_full_dashboard()

    def get_alerts(self) -> List[COTAlert]:
        """Return all current COT alerts."""
        self._ensure_data()
        assert self._alerts is not None
        return self._alerts.get_all_alerts()

    def get_weekly_summary(self) -> str:
        """Generate weekly COT text summary."""
        self._ensure_data()
        assert self._alerts is not None
        return self._alerts.get_weekly_summary()

    def get_risk_appetite(self) -> str:
        """Return RISK_ON / RISK_OFF / NEUTRAL with score."""
        self._ensure_data()
        assert self._portfolio is not None
        return self._portfolio.get_risk_appetite_label()

    def get_commodity_sector_bias(self) -> Dict[str, Any]:
        """Return commodity sector positioning bias."""
        self._ensure_data()
        assert self._portfolio is not None
        return self._portfolio.compute_commodity_sector_bias()

    def export_signals(self, path: str) -> None:
        """
        Export all market COT signals to a CSV file.

        Columns: market, cot_index, extreme_signal, commercial_signal,
                 speculator_signal, contrarian_score, oi_trend, narrative
        """
        self._ensure_data()
        assert self._signal_engine is not None

        all_markets = self._coverage.list_all_markets()
        rows = []
        for mkt in all_markets:
            try:
                sig = self._signal_engine.build_signal(mkt)
                rows.append({
                    "market":            sig.market,
                    "cftc_code":         sig.cftc_code,
                    "as_of":             sig.as_of,
                    "mm_net":            sig.mm_net,
                    "comm_net":          sig.comm_net,
                    "open_interest":     sig.open_interest,
                    "cot_index_52w":     sig.cot_index_52w,
                    "cot_index_5yr":     sig.cot_index_5yr,
                    "extreme_signal":    sig.extreme_signal,
                    "commercial_signal": sig.commercial_signal,
                    "speculator_signal": sig.speculator_signal,
                    "contrarian_score":  sig.contrarian_score,
                    "crowding_score":    sig.crowding_score,
                    "oi_trend":          sig.oi_trend,
                    "narrative":         sig.narrative,
                })
            except Exception as exc:
                logger.debug("Export skipping %s: %s", mkt, exc)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("COT signals exported to %s (%d rows)", path, len(df))

    def get_most_crowded_longs(self, n: int = 10) -> pd.DataFrame:
        """Top n most crowded long positions."""
        self._ensure_data()
        assert self._portfolio is not None
        return self._portfolio.get_most_crowded_longs(n=n)

    def get_most_extreme_shorts(self, n: int = 10) -> pd.DataFrame:
        """Top n most extreme short positions."""
        self._ensure_data()
        assert self._portfolio is not None
        return self._portfolio.get_most_extreme_shorts(n=n)

    def get_report_type(self) -> str:
        return self.report_type

    def get_coverage(self) -> COTMarketCoverage:
        return self._coverage

    def get_signal_engine(self) -> Optional[COTSignalEngine]:
        self._ensure_data()
        return self._signal_engine

    def get_portfolio_analyzer(self) -> Optional[COTPortfolioAnalyzer]:
        self._ensure_data()
        return self._portfolio

    def get_alert_system(self) -> Optional[COTAlertSystem]:
        self._ensure_data()
        return self._alerts


# ---------------------------------------------------------------------------
# Convenience functions for standalone / script usage
# ---------------------------------------------------------------------------

def _print_signal(sig: COTSignal) -> None:
    """Pretty-print a COTSignal."""
    print(f"\n{'─'*60}")
    print(f"  Market: {sig.market}  ({sig.cftc_code})  [{sig.as_of}]")
    print(f"  MM Net:          {sig.mm_net:>+12,.0f}  contracts")
    print(f"  Commercial Net:  {sig.comm_net:>+12,.0f}  contracts")
    print(f"  Open Interest:   {sig.open_interest:>12,.0f}  contracts")
    print(f"  COT Index 52w:   {sig.cot_index_52w:>8.1f}  / 100")
    print(f"  COT Index 5yr:   {sig.cot_index_5yr:>8.1f}  / 100")
    print(f"  Extreme Signal:  {sig.extreme_signal}")
    print(f"  Commercial Sig:  {sig.commercial_signal}")
    print(f"  Spec Signal:     {sig.speculator_signal}")
    print(f"  Contrarian:      {sig.contrarian_score:.0f} / 100")
    print(f"  OI Trend (4w):   {sig.oi_trend:>+.1f}%")
    print(f"  Narrative: {sig.narrative}")


def _print_sector_bias(bias: Dict[str, Any]) -> None:
    """Pretty-print commodity sector bias."""
    print(f"\n{'─'*60}")
    print("  COMMODITY SECTOR COT BIAS")
    print(f"  {'Sector':<15} {'Avg COT':>9} {'Bias':<10} {'Extreme Longs'}")
    print(f"  {'─'*15} {'─'*9} {'─'*10} {'─'*20}")
    for sector, data in bias.items():
        avg = f"{data['avg_cot_index']:.1f}" if data["avg_cot_index"] is not None else "N/A"
        longs = ", ".join(data["markets_extreme_long"][:3]) or "none"
        print(f"  {sector:<15} {avg:>9} {data['bias']:<10} {longs}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SENTINEL COT ENGINE v3 — CFTC Positioning Analysis      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    engine = COTEngine(report_type="disaggregated")

    print("\n[1/5] Downloading latest CFTC COT data...")
    try:
        engine.run_weekly_update()
        print("      OK")
    except Exception as exc:
        print(f"      ERROR: {exc}")
        sys.exit(1)

    print("\n[2/5] Computing COT signals for key markets...")
    key_markets = [
        "S&P 500 E-Mini",
        "Gold",
        "Crude Oil WTI",
        "Euro FX",
        "10-Year T-Note",
        "Japanese Yen",
    ]
    for mkt in key_markets:
        try:
            sig = engine.get_signal(mkt)
            _print_signal(sig)
        except Exception as exc:
            print(f"  [{mkt}] Error: {exc}")

    print("\n\n[3/5] Risk Appetite Index...")
    print(f"  {engine.get_risk_appetite()}")

    print("\n[4/5] Commodity Sector Bias...")
    try:
        bias = engine.get_commodity_sector_bias()
        _print_sector_bias(bias)
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[5/5] COT Extreme Alerts...")
    try:
        alerts = engine.get_alerts()
        high_alerts = [a for a in alerts if a.severity == "HIGH"]
        if high_alerts:
            for a in high_alerts[:5]:
                print(f"  [{a.severity}] {a.alert_type}: {a.description[:80]}")
        else:
            print("  No high-severity alerts.")
        print(f"\n  Total alerts: {len(alerts)}")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n\n=== COT Weekly Summary ===")
    print(engine.get_weekly_summary())

    print("\nDone.")
