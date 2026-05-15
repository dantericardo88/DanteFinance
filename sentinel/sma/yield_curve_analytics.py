"""
Yield curve spread analytics: 2s10s, 2s30s, 3m10y, butterfly, carry trades.
Free data: FRED Treasury rates, Nelson-Siegel fitting, regime classification.

Covers:
  - All Treasury maturities (1M through 30Y) from FRED free CSV endpoints
  - 2s10s, 2s30s, 3m10y, 5s30s, 10s30s, TED-spread-proxy spreads
  - 2s5s10s and 2s10s30s butterfly spreads, condor
  - NY Fed recession probability (probit model)
  - Bull steepener / bear flattener / carry trade signals
  - Nelson-Siegel curve fitting (level, slope, curvature)
  - Regime classification: inverted, flat, normal, steep
  - Global curve comparison vs Bund, Gilt, JGB
  - FastAPI router with 7 endpoints
"""
from __future__ import annotations

import math
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from scipy import optimize
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/csv,text/html,*/*",
}
_TIMEOUT = 30

_DB_PATH = Path(__file__).parent.parent.parent / ".danteforge" / "yield_curve_cache.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Treasury FRED series IDs (no API key required for CSV endpoint)
TREASURY_SERIES: dict[str, str] = {
    "1M":  "DGS1MO",
    "3M":  "DGS3MO",
    "6M":  "DGS6MO",
    "1Y":  "DGS1",
    "2Y":  "DGS2",
    "3Y":  "DGS3",
    "5Y":  "DGS5",
    "7Y":  "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

# Maturities in years for curve fitting
MATURITIES_YR: dict[str, float] = {
    "1M":  1 / 12,
    "3M":  0.25,
    "6M":  0.5,
    "1Y":  1.0,
    "2Y":  2.0,
    "3Y":  3.0,
    "5Y":  5.0,
    "7Y":  7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

# International yield curve FRED series (longer-term rates)
INTL_SERIES: dict[str, str] = {
    "DE_10Y": "IRLTLT01DEM156N",   # Germany Bund 10Y
    "GB_10Y": "IRLTLT01GBM156N",   # UK Gilt 10Y
    "JP_10Y": "IRLTLT01JPM156N",   # Japan JGB 10Y
    "CA_10Y": "IRLTLT01CAM156N",   # Canada 10Y
    "AU_10Y": "IRLTLT01AUM156N",   # Australia 10Y
    "FR_10Y": "IRLTLT01FRM156N",   # France OAT 10Y
    "IT_10Y": "IRLTLT01ITM156N",   # Italy BTP 10Y
    "SOFR":   "SOFR",              # SOFR overnight rate (TED-spread proxy)
}

# Curve regime thresholds (based on 10Y-2Y spread, basis points)
REGIME_THRESHOLDS = {
    "deeply_inverted": -100,   # bp
    "inverted": -10,
    "flat": 25,
    "normal": 100,
    "steep": 200,
}

# NY Fed recession model coefficients (Estrella & Mishkin 1996)
# P(recession in 12M) = Φ(a + b * spread_3m10y)
# spread in percentage points
NY_FED_ALPHA = -0.5381
NY_FED_BETA = -0.6762

# Historical NBER recession periods for calibration reference
NBER_RECESSIONS: list[tuple[date, date]] = [
    (date(1960, 4, 1),  date(1961, 2, 1)),
    (date(1969, 12, 1), date(1970, 11, 1)),
    (date(1973, 11, 1), date(1975, 3, 1)),
    (date(1980, 1, 1),  date(1980, 7, 1)),
    (date(1981, 7, 1),  date(1982, 11, 1)),
    (date(1990, 7, 1),  date(1991, 3, 1)),
    (date(2001, 3, 1),  date(2001, 11, 1)),
    (date(2007, 12, 1), date(2009, 6, 1)),
    (date(2020, 2, 1),  date(2020, 4, 1)),
]


# ---------------------------------------------------------------------------
# DB cache
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS curve_cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            ts REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spread_history (
            spread_name TEXT,
            as_of TEXT,
            value_bp REAL,
            PRIMARY KEY (spread_name, as_of)
        )
    """)
    conn.commit()
    return conn


def _cache_get(key: str, ttl: float = 3600.0) -> Optional[str]:
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT value, ts FROM curve_cache WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        if row and (time.time() - row[1]) < ttl:
            return row[0]
    except Exception:
        pass
    return None


def _cache_set(key: str, value: str) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT OR REPLACE INTO curve_cache (key, value, ts) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _store_spread(spread_name: str, as_of: str, value_bp: float) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT OR REPLACE INTO spread_history (spread_name, as_of, value_bp) VALUES (?, ?, ?)",
            (spread_name, as_of, value_bp),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _load_spread_history(spread_name: str, days: int = 365 * 10) -> pd.DataFrame:
    try:
        conn = _get_db()
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        df = pd.read_sql_query(
            "SELECT as_of, value_bp FROM spread_history WHERE spread_name = ? AND as_of >= ? ORDER BY as_of",
            conn,
            params=(spread_name, cutoff),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["as_of", "value_bp"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

CurveRegime = Literal["deeply_inverted", "inverted", "flat", "normal", "steep", "unknown"]
TradingSignalType = Literal[
    "bull_steepener", "bear_flattener", "bull_flattener", "bear_steepener",
    "carry_long_belly", "duration_neutral_barbell", "belly_rich", "belly_cheap",
    "neutral"
]


class TenorRate(BaseModel):
    model_config = ConfigDict(frozen=True)
    tenor: str
    series_id: str
    rate: float          # percent
    as_of: date


class CurveSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    rates: dict[str, float]       # tenor -> rate in %
    missing_tenors: list[str]
    regime: CurveRegime
    slope_2s10s: float            # bp
    curvature_2s5s10s: float      # bp (butterfly)


class SpreadRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    spread_name: str
    description: str
    current_bp: float
    one_week_ago_bp: Optional[float] = None
    one_month_ago_bp: Optional[float] = None
    one_year_ago_bp: Optional[float] = None
    percentile_10y: Optional[float] = None
    direction: Literal["widening", "narrowing", "stable"]
    signal: str


class ButterflyResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    butterfly_2s5s10s: float     # bp
    butterfly_2s10s30s: float    # bp
    condor: float                # bp
    butterfly_2s5s10s_interpretation: str
    butterfly_2s10s30s_interpretation: str
    condor_interpretation: str
    curvature_percentile_10y: Optional[float] = None


class RecessionProbability(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    spread_3m10y_bp: float
    probability_12m: float       # 0–1
    probability_12m_pct: float   # 0–100
    confidence_interval_low: float
    confidence_interval_high: float
    signal: Literal["elevated", "moderate", "low"]
    model: str
    historical_note: str


class CurveTradingSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    signal_type: TradingSignalType
    rationale: str
    conviction: Literal["high", "medium", "low"]
    leg_long: str        # what to buy
    leg_short: str       # what to sell / hedge
    carry_bps_per_year: float
    duration_neutral: bool


class NelsonSiegelParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    beta_0: float        # level
    beta_1: float        # slope
    beta_2: float        # curvature
    lambda_: float       # shape parameter
    fit_rmse: float      # fitting error in bp
    level: float         # 10Y yield proxy
    slope: float         # 10Y - 3M in bp
    curvature: float     # 2×5Y - 2Y - 10Y in bp


class GlobalCurveComparison(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    us_10y: float
    de_10y: Optional[float]
    gb_10y: Optional[float]
    jp_10y: Optional[float]
    ca_10y: Optional[float]
    au_10y: Optional[float]
    us_de_spread: Optional[float]   # bp (US - Germany)
    us_gb_spread: Optional[float]   # bp (US - UK)
    us_jp_spread: Optional[float]   # bp (US - Japan)
    fx_carry_signals: dict[str, str]
    global_term_premium_proxy: float  # avg 10Y - 3M across all


class SpreadDashboardResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    spreads: list[SpreadRecord]
    curve_regime: CurveRegime
    curve_snapshot: CurveSnapshot


# ---------------------------------------------------------------------------
# TreasuryCurveData — FRED free CSV fetcher
# ---------------------------------------------------------------------------

class TreasuryCurveData:
    """
    Fetch US Treasury yield curve data from FRED free CSV endpoints.
    No API key required. SQLite cache with 1h TTL.
    """

    def __init__(self, cache_ttl_seconds: float = 3600.0):
        self._ttl = cache_ttl_seconds

    def _fetch_series(self, series_id: str, start_date: Optional[date] = None) -> pd.Series:
        """
        Fetch a single FRED series as pd.Series indexed by date.
        Returns float values; missing values (.) become NaN.
        """
        cache_key = f"fred_series_{series_id}_{start_date or 'all'}"
        cached = _cache_get(cache_key, ttl=self._ttl)
        if cached:
            import io
            return pd.read_csv(io.StringIO(cached), index_col=0, squeeze=False).iloc[:, 0]

        url = f"{FRED_BASE}?id={series_id}"
        if start_date:
            url += f"&vintage_date={start_date.isoformat()}"

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            raw = resp.text
        except Exception as exc:
            logger.warning(f"_fetch_series {series_id}: {exc}")
            return pd.Series(dtype=float)

        try:
            import io
            df = pd.read_csv(io.StringIO(raw), index_col=0)
            df.index = pd.to_datetime(df.index)
            df.columns = [series_id]
            # Replace "." with NaN and convert
            df.replace(".", np.nan, inplace=True)
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            series = df[series_id].dropna()
            _cache_set(cache_key, series.to_csv())
            return series
        except Exception as exc:
            logger.warning(f"_fetch_series parse {series_id}: {exc}")
            return pd.Series(dtype=float)

    def fetch_curve(
        self,
        as_of: Optional[date] = None,
        lookback_days: int = 10,
    ) -> dict[str, float]:
        """
        Fetch all Treasury tenors for a given date.
        Tries the as_of date first; if any tenors missing, looks back up to lookback_days.
        Returns dict: tenor -> rate in %.
        """
        target_date = as_of or date.today()
        cache_key = f"curve_{target_date.isoformat()}"
        cached = _cache_get(cache_key, ttl=self._ttl)
        if cached:
            import json
            return json.loads(cached)

        rates: dict[str, float] = {}
        for tenor, series_id in TREASURY_SERIES.items():
            series = self._fetch_series(series_id)
            if series.empty:
                continue
            # Find nearest date <= target_date within lookback_days
            ts_target = pd.Timestamp(target_date)
            past = series[series.index <= ts_target]
            if past.empty:
                continue
            # Check it's not too stale
            nearest_date = past.index[-1]
            if (ts_target - nearest_date).days > lookback_days:
                continue
            val = float(past.iloc[-1])
            if not math.isnan(val):
                rates[tenor] = round(val, 4)

        if rates:
            import json
            _cache_set(cache_key, json.dumps(rates))

        return rates

    def fetch_history(
        self,
        tenor: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.Series:
        """Fetch historical rates for a single tenor."""
        if tenor not in TREASURY_SERIES:
            raise ValueError(f"Unknown tenor: {tenor}. Valid: {list(TREASURY_SERIES.keys())}")
        series = self._fetch_series(TREASURY_SERIES[tenor], start_date)
        if end_date:
            series = series[series.index <= pd.Timestamp(end_date)]
        if start_date:
            series = series[series.index >= pd.Timestamp(start_date)]
        return series

    def build_curve_dataframe(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Build a full DataFrame of all Treasury tenors over a date range.
        Columns: tenor names, rows: dates.
        """
        frames: dict[str, pd.Series] = {}
        for tenor, series_id in TREASURY_SERIES.items():
            s = self._fetch_series(series_id, start_date)
            if not s.empty:
                if start_date:
                    s = s[s.index >= pd.Timestamp(start_date)]
                if end_date:
                    s = s[s.index <= pd.Timestamp(end_date)]
                frames[tenor] = s

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.sort_index(inplace=True)
        return df

    def fetch_intl_rates(self, as_of: Optional[date] = None) -> dict[str, float]:
        """Fetch international 10Y rates from FRED."""
        target_date = as_of or date.today()
        cache_key = f"intl_rates_{target_date.isoformat()}"
        cached = _cache_get(cache_key, ttl=self._ttl)
        if cached:
            import json
            return json.loads(cached)

        rates: dict[str, float] = {}
        for label, series_id in INTL_SERIES.items():
            series = self._fetch_series(series_id)
            if series.empty:
                continue
            ts_target = pd.Timestamp(target_date)
            past = series[series.index <= ts_target]
            if past.empty:
                continue
            # Monthly series — allow up to 45 days stale
            nearest = past.index[-1]
            if (ts_target - nearest).days > 45:
                continue
            val = float(past.iloc[-1])
            if not math.isnan(val):
                rates[label] = round(val, 4)

        if rates:
            import json
            _cache_set(cache_key, json.dumps(rates))

        return rates

    def _rate_n_periods_ago(
        self, series: pd.Series, n_days: int, as_of: date
    ) -> Optional[float]:
        """Get rate approximately n trading days ago."""
        target = pd.Timestamp(as_of) - timedelta(days=n_days)
        past = series[series.index <= target]
        if past.empty:
            return None
        # Allow ±5 days
        nearest = past.index[-1]
        if (target - nearest).days > 10:
            return None
        val = float(past.iloc[-1])
        return round(val, 4) if not math.isnan(val) else None

    def classify_regime(self, rates: dict[str, float]) -> CurveRegime:
        """Classify curve regime based on 2s10s spread."""
        r2y = rates.get("2Y")
        r10y = rates.get("10Y")
        if r2y is None or r10y is None:
            return "unknown"
        spread_bp = (r10y - r2y) * 100
        if spread_bp < REGIME_THRESHOLDS["deeply_inverted"]:
            return "deeply_inverted"
        if spread_bp < REGIME_THRESHOLDS["inverted"]:
            return "inverted"
        if spread_bp < REGIME_THRESHOLDS["flat"]:
            return "flat"
        if spread_bp < REGIME_THRESHOLDS["normal"]:
            return "normal"
        if spread_bp < REGIME_THRESHOLDS["steep"]:
            return "steep"
        return "steep"


# ---------------------------------------------------------------------------
# SpreadDashboard
# ---------------------------------------------------------------------------

class SpreadDashboard:
    """
    Compute all key yield curve spreads with historical context.
    """

    SPREAD_DEFINITIONS: dict[str, tuple[str, str, str]] = {
        # name: (long_tenor, short_tenor, description)
        "2s10s":  ("10Y", "2Y",  "2Y-10Y spread — most-watched recession indicator"),
        "2s30s":  ("30Y", "2Y",  "2Y-30Y spread — long-end steepness"),
        "3m10y":  ("10Y", "3M",  "3M-10Y spread — NY Fed recession model input"),
        "5s30s":  ("30Y", "5Y",  "5Y-30Y spread — long-end carry"),
        "10s30s": ("30Y", "10Y", "10Y-30Y spread — ultra-long premium"),
        "1y2y":   ("2Y",  "1Y",  "1Y-2Y spread — near-term policy expectations"),
        "2y5y":   ("5Y",  "2Y",  "2Y-5Y spread — medium-term steepener"),
        "5y10y":  ("10Y", "5Y",  "5Y-10Y spread — medium long-end"),
    }

    def __init__(self, curve_data: TreasuryCurveData):
        self._data = curve_data

    def _spread_bp(
        self, rates: dict[str, float], long_tenor: str, short_tenor: str
    ) -> Optional[float]:
        r_long = rates.get(long_tenor)
        r_short = rates.get(short_tenor)
        if r_long is None or r_short is None:
            return None
        return round((r_long - r_short) * 100, 2)

    def _spread_signal(self, spread_name: str, current_bp: float) -> str:
        """Generate a brief signal description for each spread."""
        if spread_name == "3m10y":
            if current_bp < -100:
                return "Deeply inverted: historically precedes recession by 6-18M"
            if current_bp < 0:
                return "Inverted: NY Fed model signals elevated recession risk"
            if current_bp < 50:
                return "Flat-to-slightly positive: neutral signal"
            return "Positive: benign recession risk signal"

        if spread_name == "2s10s":
            if current_bp < -50:
                return "Deeply inverted: strong recession warning"
            if current_bp < 0:
                return "Inverted: bear-flattening or early recession signal"
            if current_bp < 30:
                return "Flat: market sees little term premium"
            if current_bp > 150:
                return "Steep: market expects rate cuts or strong growth recovery"
            return "Normal steepening: positive for banks"

        if spread_name in ("2s30s", "5s30s"):
            if current_bp < 0:
                return "Inverted long end: strong rate cut expectations"
            if current_bp > 100:
                return "Steep: term premium elevated, reflation trade"
            return "Normal long-end premium"

        return f"Current: {current_bp:.1f}bp"

    def _direction(self, current: float, one_month_ago: Optional[float]) -> str:
        if one_month_ago is None:
            return "stable"
        delta = current - one_month_ago
        if delta > 5:
            return "widening"
        if delta < -5:
            return "narrowing"
        return "stable"

    def compute_all_spreads(self, as_of: Optional[date] = None) -> list[SpreadRecord]:
        """Compute all key spreads for a given date."""
        target = as_of or date.today()
        rates = self._data.fetch_curve(target)

        results: list[SpreadRecord] = []
        for spread_name, (long_t, short_t, desc) in self.SPREAD_DEFINITIONS.items():
            current_bp = self._spread_bp(rates, long_t, short_t)
            if current_bp is None:
                continue

            # Historical context from DB
            history = _load_spread_history(spread_name, days=365 * 10)

            one_week_ago = None
            one_month_ago = None
            one_year_ago = None
            percentile = None

            if not history.empty:
                history["as_of"] = pd.to_datetime(history["as_of"])
                history = history.sort_values("as_of")

                def _hist_val(delta_days: int) -> Optional[float]:
                    target_ts = pd.Timestamp(target) - timedelta(days=delta_days)
                    past = history[history["as_of"] <= target_ts]
                    if past.empty:
                        return None
                    nearest = past.iloc[-1]
                    if (target_ts - nearest["as_of"]).days > delta_days // 2 + 5:
                        return None
                    return round(float(nearest["value_bp"]), 2)

                one_week_ago = _hist_val(7)
                one_month_ago = _hist_val(30)
                one_year_ago = _hist_val(365)

                all_vals = history["value_bp"].dropna()
                if len(all_vals) > 10:
                    percentile = round(float((all_vals < current_bp).mean() * 100), 1)

            # Store today's value
            _store_spread(spread_name, target.isoformat(), current_bp)

            results.append(SpreadRecord(
                spread_name=spread_name,
                description=desc,
                current_bp=current_bp,
                one_week_ago_bp=one_week_ago,
                one_month_ago_bp=one_month_ago,
                one_year_ago_bp=one_year_ago,
                percentile_10y=percentile,
                direction=self._direction(current_bp, one_month_ago),
                signal=self._spread_signal(spread_name, current_bp),
            ))

        return results

    def ted_spread_proxy(self, as_of: Optional[date] = None) -> Optional[float]:
        """
        TED spread proxy: 3M Treasury vs SOFR.
        """
        target = as_of or date.today()
        rates = self._data.fetch_curve(target)
        intl = self._data.fetch_intl_rates(target)
        sofr = intl.get("SOFR")
        t3m = rates.get("3M")
        if sofr is None or t3m is None:
            return None
        return round((t3m - sofr) * 100, 2)

    def snapshot(self, as_of: Optional[date] = None) -> SpreadDashboardResult:
        """Full spread dashboard snapshot."""
        target = as_of or date.today()
        rates = self._data.fetch_curve(target)
        spreads = self.compute_all_spreads(target)
        regime = self._data.classify_regime(rates)

        r2 = rates.get("2Y", 0.0)
        r10 = rates.get("10Y", 0.0)
        r2y_r10y_spread = (r10 - r2) * 100 if r2 and r10 else 0.0

        r5 = rates.get("5Y", 0.0)
        butterfly = (r5 - 0.5 * (r2 + r10)) * 100 if r5 and r2 and r10 else 0.0

        curve_snap = CurveSnapshot(
            as_of=target,
            rates=rates,
            missing_tenors=[t for t in TREASURY_SERIES if t not in rates],
            regime=regime,
            slope_2s10s=round(r2y_r10y_spread, 2),
            curvature_2s5s10s=round(butterfly, 2),
        )

        return SpreadDashboardResult(
            as_of=target,
            spreads=spreads,
            curve_regime=regime,
            curve_snapshot=curve_snap,
        )


# ---------------------------------------------------------------------------
# ButterflySpread
# ---------------------------------------------------------------------------

class ButterflySpread:
    """
    Butterfly and condor spread computation.
    """

    def __init__(self, curve_data: TreasuryCurveData):
        self._data = curve_data

    def compute(self, as_of: Optional[date] = None) -> ButterflyResult:
        """Compute all butterfly spreads for a given date."""
        target = as_of or date.today()
        rates = self._data.fetch_curve(target)

        r2 = rates.get("2Y")
        r5 = rates.get("5Y")
        r10 = rates.get("10Y")
        r30 = rates.get("30Y")

        # 2s5s10s butterfly: 5Y - 0.5*(2Y + 10Y)
        btfly_2s5s10s: float = 0.0
        if r2 is not None and r5 is not None and r10 is not None:
            btfly_2s5s10s = round((r5 - 0.5 * (r2 + r10)) * 100, 2)

        def _interpret_2s5s10s(val: float) -> str:
            if val > 30:
                return f"Strongly humped at 5Y (+{val:.1f}bp): 5Y expensive vs wings"
            if val > 10:
                return f"Moderately humped at 5Y (+{val:.1f}bp): belly rich"
            if val < -30:
                return f"Tent-shaped, deeply negative ({val:.1f}bp): 5Y cheap vs wings"
            if val < -10:
                return f"Mildly tent-shaped ({val:.1f}bp): belly cheap"
            return f"Flat butterfly ({val:.1f}bp): no notable curvature"

        # 2s10s30s butterfly: 10Y - 0.5*(2Y + 30Y)
        btfly_2s10s30s: float = 0.0
        if r2 is not None and r10 is not None and r30 is not None:
            btfly_2s10s30s = round((r10 - 0.5 * (r2 + r30)) * 100, 2)

        def _interpret_2s10s30s(val: float) -> str:
            if val > 20:
                return f"10Y hump ({val:.1f}bp): belly of curve elevated vs extremes"
            if val < -20:
                return f"10Y trough ({val:.1f}bp): belly of curve depressed vs extremes"
            return f"Flat 2s10s30s butterfly ({val:.1f}bp)"

        # Condor: (30Y-10Y) - (10Y-2Y)
        condor: float = 0.0
        if r2 is not None and r10 is not None and r30 is not None:
            condor = round(((r30 - r10) - (r10 - r2)) * 100, 2)

        def _interpret_condor(val: float) -> str:
            if val > 30:
                return f"Positive condor ({val:.1f}bp): long end steeper than short-to-mid"
            if val < -30:
                return f"Negative condor ({val:.1f}bp): long end flatter than short-to-mid"
            return f"Balanced condor ({val:.1f}bp)"

        # 10Y percentile from history
        history = _load_spread_history("curvature_2s5s10s", days=365 * 10)
        _store_spread("curvature_2s5s10s", target.isoformat(), btfly_2s5s10s)
        percentile = None
        if not history.empty:
            all_vals = history["value_bp"].dropna()
            if len(all_vals) > 10:
                percentile = round(float((all_vals < btfly_2s5s10s).mean() * 100), 1)

        return ButterflyResult(
            as_of=target,
            butterfly_2s5s10s=btfly_2s5s10s,
            butterfly_2s10s30s=btfly_2s10s30s,
            condor=condor,
            butterfly_2s5s10s_interpretation=_interpret_2s5s10s(btfly_2s5s10s),
            butterfly_2s10s30s_interpretation=_interpret_2s10s30s(btfly_2s10s30s),
            condor_interpretation=_interpret_condor(condor),
            curvature_percentile_10y=percentile,
        )

    def history(self, days: int = 252) -> pd.DataFrame:
        """Return historical 2s5s10s butterfly values."""
        df = _load_spread_history("curvature_2s5s10s", days=days)
        return df.rename(columns={"value_bp": "butterfly_2s5s10s_bp"})


# ---------------------------------------------------------------------------
# RecessionProbabilityModel — NY Fed probit
# ---------------------------------------------------------------------------

class RecessionProbabilityModel:
    """
    NY Fed recession probability model (Estrella & Mishkin 1996).
    P(recession in 12M) = Φ(a + b × spread_3m10y)
    spread_3m10y in percentage points.
    """

    def __init__(self, alpha: float = NY_FED_ALPHA, beta: float = NY_FED_BETA):
        self.alpha = alpha
        self.beta = beta

    def probability(self, spread_3m10y_pct: float) -> float:
        """Compute recession probability from 3M-10Y spread in percentage points."""
        z = self.alpha + self.beta * spread_3m10y_pct
        return float(norm.cdf(z))

    def confidence_interval(
        self,
        spread_3m10y_pct: float,
        se_alpha: float = 0.1,
        se_beta: float = 0.07,
    ) -> tuple[float, float]:
        """
        Bootstrap-style confidence interval via parameter uncertainty.
        Uses ±1.96 SE on alpha and beta from original paper.
        """
        z_central = self.alpha + self.beta * spread_3m10y_pct
        z_se = math.sqrt(se_alpha**2 + (se_beta * spread_3m10y_pct) ** 2)
        z_low = z_central - 1.96 * z_se
        z_high = z_central + 1.96 * z_se
        return float(norm.cdf(z_low)), float(norm.cdf(z_high))

    def compute(self, curve_data: TreasuryCurveData, as_of: Optional[date] = None) -> RecessionProbability:
        """Compute recession probability using live curve data."""
        target = as_of or date.today()
        rates = curve_data.fetch_curve(target)
        r3m = rates.get("3M")
        r10y = rates.get("10Y")

        if r3m is None or r10y is None:
            spread_bp = 0.0
            spread_pct = 0.0
        else:
            spread_pct = r10y - r3m       # in pp
            spread_bp = spread_pct * 100

        prob = self.probability(spread_pct)
        ci_low, ci_high = self.confidence_interval(spread_pct)

        if prob >= 0.40:
            signal: Literal["elevated", "moderate", "low"] = "elevated"
        elif prob >= 0.20:
            signal = "moderate"
        else:
            signal = "low"

        # Historical note: what was recession probability before past recessions?
        notes = {
            "elevated": (
                "Probability ≥40%: historically consistent with recession within 12M. "
                "Pre-2008 peak reached 50%+; pre-2001 exceeded 45%."
            ),
            "moderate": (
                "Probability 20-40%: elevated but not at historical alarm levels. "
                "Monitor for sustained inversion deepening."
            ),
            "low": (
                "Probability <20%: consistent with expansion. "
                "NY Fed model shows no near-term recession signal."
            ),
        }

        return RecessionProbability(
            as_of=target,
            spread_3m10y_bp=round(spread_bp, 2),
            probability_12m=round(prob, 4),
            probability_12m_pct=round(prob * 100, 2),
            confidence_interval_low=round(ci_low * 100, 2),
            confidence_interval_high=round(ci_high * 100, 2),
            signal=signal,
            model="NY Fed probit: P = Φ(-0.5381 - 0.6762 × spread_3m10y_pct)",
            historical_note=notes[signal],
        )

    def historical_probabilities(
        self,
        curve_data: TreasuryCurveData,
        start_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Compute recession probability time series from historical spread data."""
        s10 = curve_data.fetch_history("10Y", start_date=start_date)
        s3m = curve_data.fetch_history("3M", start_date=start_date)
        df = pd.DataFrame({"10Y": s10, "3M": s3m}).dropna()
        df["spread_pct"] = df["10Y"] - df["3M"]
        df["recession_prob"] = df["spread_pct"].apply(self.probability)
        df["spread_bp"] = df["spread_pct"] * 100
        return df[["spread_bp", "recession_prob"]]


# ---------------------------------------------------------------------------
# Nelson-Siegel curve fitting
# ---------------------------------------------------------------------------

def _nelson_siegel(tau: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float) -> np.ndarray:
    """
    Nelson-Siegel yield curve model.
    r(t) = β₀ + β₁ × (1 - e^(-λt))/(λt) + β₂ × [(1 - e^(-λt))/(λt) - e^(-λt)]
    """
    # Avoid division by zero for tau ~ 0
    lam_tau = lam * tau
    factor1 = np.where(lam_tau > 1e-8, (1 - np.exp(-lam_tau)) / lam_tau, 1.0)
    factor2 = factor1 - np.exp(-lam_tau)
    return beta0 + beta1 * factor1 + beta2 * factor2


def fit_nelson_siegel(rates: dict[str, float]) -> NelsonSiegelParams:
    """
    Fit Nelson-Siegel model to available Treasury rates.
    Returns fitted parameters and fit quality.
    """
    today = date.today()
    available = {t: r for t, r in rates.items() if t in MATURITIES_YR}
    if len(available) < 4:
        return NelsonSiegelParams(
            as_of=today,
            beta_0=0.0, beta_1=0.0, beta_2=0.0, lambda_=0.5,
            fit_rmse=99.0, level=0.0, slope=0.0, curvature=0.0,
        )

    taus = np.array([MATURITIES_YR[t] for t in available], dtype=float)
    yields = np.array([r for r in available.values()], dtype=float)

    try:
        p0 = [yields.max(), yields.min() - yields.max(), 0.0, 0.5]
        bounds = ([0, -10, -10, 0.01], [20, 10, 10, 5.0])
        popt, _ = optimize.curve_fit(
            _nelson_siegel, taus, yields, p0=p0, bounds=bounds, maxfev=5000
        )
        beta0, beta1, beta2, lam = popt
        fitted = _nelson_siegel(taus, *popt)
        rmse = float(np.sqrt(np.mean((fitted - yields) ** 2))) * 100  # bp

    except Exception as exc:
        logger.warning(f"Nelson-Siegel fit failed: {exc}")
        beta0 = float(np.mean(yields))
        beta1, beta2, lam = 0.0, 0.0, 0.5
        rmse = 99.0

    level = rates.get("10Y", beta0)
    r10 = rates.get("10Y", 0.0)
    r3m = rates.get("3M", 0.0)
    slope = (r10 - r3m) * 100 if r10 and r3m else 0.0

    r2 = rates.get("2Y", 0.0)
    r5 = rates.get("5Y", 0.0)
    curvature = (r5 - 0.5 * (r2 + r10)) * 100 if r2 and r5 and r10 else 0.0

    return NelsonSiegelParams(
        as_of=today,
        beta_0=round(beta0, 4),
        beta_1=round(beta1, 4),
        beta_2=round(beta2, 4),
        lambda_=round(lam, 4),
        fit_rmse=round(rmse, 3),
        level=round(level, 4),
        slope=round(slope, 2),
        curvature=round(curvature, 2),
    )


# ---------------------------------------------------------------------------
# CurveTradingSignals
# ---------------------------------------------------------------------------

def _roll_down_carry(short_rate: float, long_rate: float, short_maturity: float, long_maturity: float) -> float:
    """
    Estimate roll-down carry in basis points per year for a position.
    As bond ages from long_maturity to short_maturity, it picks up:
    carry ≈ (long_rate - short_rate) / (long_maturity - short_maturity) × short_maturity
    (rough approximation ignoring convexity)
    """
    if long_maturity <= short_maturity:
        return 0.0
    slope = (long_rate - short_rate) / (long_maturity - short_maturity)
    return round(slope * short_maturity * 100, 2)


class CurveTradingSignals:
    """
    Generate trading signals from yield curve shape.
    """

    def __init__(self, curve_data: TreasuryCurveData):
        self._data = curve_data

    def _current_fed_stance(self, spread_3m10y_bp: float) -> str:
        """Infer Fed stance from curve shape."""
        if spread_3m10y_bp < -100:
            return "aggressively_tightened"
        if spread_3m10y_bp < 0:
            return "tightened"
        if spread_3m10y_bp < 50:
            return "neutral"
        return "easing_or_loose"

    def generate_signals(self, as_of: Optional[date] = None) -> list[CurveTradingSignal]:
        """Generate all curve trading signals."""
        target = as_of or date.today()
        rates = self._data.fetch_curve(target)

        signals: list[CurveTradingSignal] = []

        r2 = rates.get("2Y", 0.0)
        r5 = rates.get("5Y", 0.0)
        r10 = rates.get("10Y", 0.0)
        r30 = rates.get("30Y", 0.0)
        r3m = rates.get("3M", 0.0)

        spread_2s10s = (r10 - r2) * 100 if r2 and r10 else 0.0
        spread_3m10y = (r10 - r3m) * 100 if r3m and r10 else 0.0
        spread_5s30s = (r30 - r5) * 100 if r5 and r30 else 0.0

        # 1. Bull Steepener — Fed cutting into weak economy
        if spread_2s10s < 20 and spread_3m10y < 0:
            carry = _roll_down_carry(r2, r10, 2.0, 10.0)
            signals.append(CurveTradingSignal(
                as_of=target,
                signal_type="bull_steepener",
                rationale=(
                    f"2s10s flat at {spread_2s10s:.0f}bp, 3m10y inverted at {spread_3m10y:.0f}bp. "
                    "Inverted curve + Fed likely to cut → buy 10Y vs sell 2Y. "
                    "Long-end rates fall more as Fed cuts; short-end anchored by policy."
                ),
                conviction="high" if spread_3m10y < -100 else "medium",
                leg_long="10Y Treasury",
                leg_short="2Y Treasury (or receive 2Y swap)",
                carry_bps_per_year=carry,
                duration_neutral=False,
            ))

        # 2. Bear Flattener — Fed hiking
        if spread_2s10s > 100 and r2 < r3m:
            carry = _roll_down_carry(r5, r30, 5.0, 30.0)
            signals.append(CurveTradingSignal(
                as_of=target,
                signal_type="bear_flattener",
                rationale=(
                    f"2s10s steep at {spread_2s10s:.0f}bp with Fed hiking cycle. "
                    "Sell long-end (10-30Y), buy short-end protection. "
                    "Short-end rates rise faster with rate hikes; long-end lags."
                ),
                conviction="high" if spread_2s10s > 150 else "medium",
                leg_long="2Y Treasury (short duration)",
                leg_short="10Y or 30Y Treasury (long duration)",
                carry_bps_per_year=-carry,
                duration_neutral=True,
            ))

        # 3. Carry trade — long steep part of curve
        if r5 and r10 and r2:
            spread_2s5s = (r5 - r2) * 100
            if spread_2s5s > 50 or spread_5s30s > 50:
                steep_segment = "5Y-10Y" if spread_2s10s > spread_5s30s else "5Y-30Y"
                carry = _roll_down_carry(r5, r10, 5.0, 10.0) if spread_2s10s > spread_5s30s else _roll_down_carry(r5, r30, 5.0, 30.0)
                signals.append(CurveTradingSignal(
                    as_of=target,
                    signal_type="carry_long_belly",
                    rationale=(
                        f"Steepest segment {steep_segment}: carry + roll-down favors holding belly. "
                        f"Estimated carry: ~{carry:.0f}bp/yr. Finance with short-end."
                    ),
                    conviction="medium",
                    leg_long=f"{steep_segment} segment",
                    leg_short="2Y or 3M (financing leg)",
                    carry_bps_per_year=carry,
                    duration_neutral=False,
                ))

        # 4. Duration-neutral barbell: 2Y + 30Y vs 10Y
        if r2 and r10 and r30:
            barbell_rate = 0.5 * (r2 + r30)
            richness = (r10 - barbell_rate) * 100  # positive → 10Y rich vs barbell
            if richness > 20:
                signals.append(CurveTradingSignal(
                    as_of=target,
                    signal_type="duration_neutral_barbell",
                    rationale=(
                        f"10Y is {richness:.1f}bp rich vs duration-neutral barbell of 2Y+30Y. "
                        "Sell 10Y, buy barbell (2Y+30Y) for duration-neutral trade. "
                        "Positive carry from long 30Y vs short 10Y."
                    ),
                    conviction="medium" if richness > 30 else "low",
                    leg_long="2Y + 30Y barbell",
                    leg_short="10Y Treasury (belly sell)",
                    carry_bps_per_year=round(richness * 0.5, 1),
                    duration_neutral=True,
                ))
            elif richness < -20:
                signals.append(CurveTradingSignal(
                    as_of=target,
                    signal_type="belly_cheap",
                    rationale=(
                        f"10Y is {abs(richness):.1f}bp cheap vs barbell. "
                        "Buy 10Y belly, sell wings (2Y+30Y). Positive carry from yield pickup."
                    ),
                    conviction="medium",
                    leg_long="10Y Treasury (belly)",
                    leg_short="2Y + 30Y barbell (wings)",
                    carry_bps_per_year=round(abs(richness) * 0.5, 1),
                    duration_neutral=True,
                ))

        # 5. 5Y belly richness vs 2Y+10Y barbell
        if r2 and r5 and r10:
            belly_richness = (r5 - 0.5 * (r2 + r10)) * 100
            if belly_richness > 15:
                signals.append(CurveTradingSignal(
                    as_of=target,
                    signal_type="belly_rich",
                    rationale=(
                        f"5Y belly {belly_richness:.1f}bp rich vs 2Y+10Y wings. "
                        "Sell 5Y, buy wings (2Y+10Y) as butterfly trade."
                    ),
                    conviction="low",
                    leg_long="2Y + 10Y barbell",
                    leg_short="5Y Treasury",
                    carry_bps_per_year=round(-belly_richness * 0.3, 1),
                    duration_neutral=True,
                ))

        if not signals:
            signals.append(CurveTradingSignal(
                as_of=target,
                signal_type="neutral",
                rationale=(
                    f"Current 2s10s: {spread_2s10s:.1f}bp. "
                    "No strong curve signal detected. Monitor for regime change."
                ),
                conviction="low",
                leg_long="N/A",
                leg_short="N/A",
                carry_bps_per_year=0.0,
                duration_neutral=True,
            ))

        return signals

    def barbell_vs_bullet(self, rates: dict[str, float]) -> dict[str, float]:
        """Compute barbell vs bullet (10Y) metrics."""
        r2 = rates.get("2Y", 0.0)
        r10 = rates.get("10Y", 0.0)
        r30 = rates.get("30Y", 0.0)
        barbell = 0.5 * (r2 + r30)
        richness = (r10 - barbell) * 100
        return {
            "barbell_rate": round(barbell, 4),
            "bullet_10y": round(r10, 4),
            "belly_richness_bp": round(richness, 2),
            "barbell_trade": "sell belly, buy wings" if richness > 15 else "buy belly, sell wings" if richness < -15 else "neutral",
        }


# ---------------------------------------------------------------------------
# GlobalCurveComparison
# ---------------------------------------------------------------------------

class GlobalCurveComparison:
    """
    Compare US Treasury yields to international government bond yields.
    """

    def __init__(self, curve_data: TreasuryCurveData):
        self._data = curve_data

    def compute(self, as_of: Optional[date] = None) -> GlobalCurveComparison:
        target = as_of or date.today()
        us_rates = self._data.fetch_curve(target)
        intl_rates = self._data.fetch_intl_rates(target)

        us_10y = us_rates.get("10Y", 0.0)
        de_10y = intl_rates.get("DE_10Y")
        gb_10y = intl_rates.get("GB_10Y")
        jp_10y = intl_rates.get("JP_10Y")
        ca_10y = intl_rates.get("CA_10Y")
        au_10y = intl_rates.get("AU_10Y")

        def spread_bp(a: float, b: Optional[float]) -> Optional[float]:
            if b is None:
                return None
            return round((a - b) * 100, 2)

        us_de = spread_bp(us_10y, de_10y)
        us_gb = spread_bp(us_10y, gb_10y)
        us_jp = spread_bp(us_10y, jp_10y)

        # FX carry signals based on rate differentials
        fx_signals: dict[str, str] = {}
        if us_de is not None:
            if us_de > 150:
                fx_signals["EURUSD"] = f"USD favored (US-DE spread +{us_de:.0f}bp): bearish EURUSD"
            elif us_de < 50:
                fx_signals["EURUSD"] = f"EUR favored (tight US-DE spread {us_de:.0f}bp): bullish EURUSD"
            else:
                fx_signals["EURUSD"] = f"Neutral (US-DE: {us_de:.0f}bp)"

        if us_gb is not None:
            if us_gb > 100:
                fx_signals["GBPUSD"] = f"USD favored (US-UK +{us_gb:.0f}bp): bearish GBPUSD"
            elif us_gb < 0:
                fx_signals["GBPUSD"] = f"GBP favored (UK yields higher by {abs(us_gb):.0f}bp): bullish GBPUSD"
            else:
                fx_signals["GBPUSD"] = f"Neutral (US-UK: {us_gb:.0f}bp)"

        if us_jp is not None:
            if us_jp > 300:
                fx_signals["USDJPY"] = f"Yen carry trade viable (US-JP +{us_jp:.0f}bp): bullish USDJPY"
            else:
                fx_signals["USDJPY"] = f"Carry compressed (US-JP: {us_jp:.0f}bp)"

        # Global term premium proxy: average 10Y-3M across available countries
        us_3m = us_rates.get("3M", 0.0)
        term_premia: list[float] = []
        if us_10y and us_3m:
            term_premia.append((us_10y - us_3m) * 100)
        if de_10y:
            term_premia.append((de_10y - 0.0) * 100)  # ECB deposit rate proxy
        if gb_10y:
            term_premia.append(gb_10y * 100 - 500)   # rough BOE base rate offset
        global_tp = round(float(np.mean(term_premia)), 2) if term_premia else 0.0

        return GlobalCurveComparison(
            as_of=target,
            us_10y=round(us_10y, 4),
            de_10y=round(de_10y, 4) if de_10y else None,
            gb_10y=round(gb_10y, 4) if gb_10y else None,
            jp_10y=round(jp_10y, 4) if jp_10y else None,
            ca_10y=round(ca_10y, 4) if ca_10y else None,
            au_10y=round(au_10y, 4) if au_10y else None,
            us_de_spread=us_de,
            us_gb_spread=us_gb,
            us_jp_spread=us_jp,
            fx_carry_signals=fx_signals,
            global_term_premium_proxy=global_tp,
        )


# ---------------------------------------------------------------------------
# YieldCurveAnalytics — main orchestrator
# ---------------------------------------------------------------------------

class YieldCurveAnalytics:
    """
    Full yield curve analytics suite.
    """

    def __init__(self):
        self.data = TreasuryCurveData()
        self.spreads = SpreadDashboard(self.data)
        self.butterfly = ButterflySpread(self.data)
        self.recession = RecessionProbabilityModel()
        self.signals = CurveTradingSignals(self.data)
        self.global_compare = GlobalCurveComparison(self.data)

    def full_snapshot(self, as_of: Optional[date] = None) -> dict[str, Any]:
        """Run all analytics and return consolidated result."""
        target = as_of or date.today()
        rates = self.data.fetch_curve(target)
        regime = self.data.classify_regime(rates)

        spread_dashboard = self.spreads.snapshot(target)
        btfly = self.butterfly.compute(target)
        rec_prob = self.recession.compute(self.data, target)
        trade_signals = self.signals.generate_signals(target)
        global_comp = self.global_compare.compute(target)
        ns_params = fit_nelson_siegel(rates)

        return {
            "as_of": target.isoformat(),
            "curve_regime": regime,
            "rates": rates,
            "spread_dashboard": spread_dashboard.model_dump(mode="json"),
            "butterfly": btfly.model_dump(mode="json"),
            "recession_probability": rec_prob.model_dump(mode="json"),
            "trading_signals": [s.model_dump(mode="json") for s in trade_signals],
            "global_comparison": global_comp.model_dump(mode="json"),
            "nelson_siegel": ns_params.model_dump(mode="json"),
        }

    def spread_history_df(
        self, spread_name: str, days: int = 365
    ) -> pd.DataFrame:
        """Return historical spread data as DataFrame."""
        df = _load_spread_history(spread_name, days=days)
        if df.empty:
            return pd.DataFrame(columns=["date", "spread_bp"])
        df = df.rename(columns={"as_of": "date", "value_bp": "spread_bp"})
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def percentile_table(self, as_of: Optional[date] = None) -> dict[str, Any]:
        """Return percentile rankings for all spreads."""
        spreads = self.spreads.compute_all_spreads(as_of)
        return {
            s.spread_name: {
                "current_bp": s.current_bp,
                "percentile_10y": s.percentile_10y,
                "direction": s.direction,
            }
            for s in spreads
        }

    def curve_series(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict[str, Any]:
        """Return full curve DataFrame as dict for API serialisation."""
        df = self.data.build_curve_dataframe(start_date, end_date)
        if df.empty:
            return {}
        return {
            "dates": [d.isoformat() for d in df.index],
            "tenors": list(df.columns),
            "data": {col: df[col].round(4).tolist() for col in df.columns},
        }


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

yield_curve_router = APIRouter(prefix="/curve", tags=["yield-curve"])

_analytics = YieldCurveAnalytics()


@yield_curve_router.get("/spreads")
def get_spreads(
    as_of: Optional[str] = Query(default=None, description="Date YYYY-MM-DD, defaults to today"),
) -> dict[str, Any]:
    """Get all key yield curve spreads with historical context."""
    target = _parse_date_param(as_of)
    snapshot = _analytics.spreads.snapshot(target)
    return snapshot.model_dump(mode="json")


@yield_curve_router.get("/butterfly")
def get_butterfly(
    as_of: Optional[str] = Query(default=None, description="Date YYYY-MM-DD"),
) -> dict[str, Any]:
    """Get butterfly and condor spread metrics."""
    target = _parse_date_param(as_of)
    result = _analytics.butterfly.compute(target)
    return result.model_dump(mode="json")


@yield_curve_router.get("/recession-prob")
def get_recession_probability(
    as_of: Optional[str] = Query(default=None, description="Date YYYY-MM-DD"),
) -> dict[str, Any]:
    """Get NY Fed recession probability model output."""
    target = _parse_date_param(as_of)
    result = _analytics.recession.compute(_analytics.data, target)
    return result.model_dump(mode="json")


@yield_curve_router.get("/regime")
def get_curve_regime(
    as_of: Optional[str] = Query(default=None, description="Date YYYY-MM-DD"),
) -> dict[str, Any]:
    """Get current yield curve regime classification and Nelson-Siegel fit."""
    target = _parse_date_param(as_of)
    rates = _analytics.data.fetch_curve(target)
    regime = _analytics.data.classify_regime(rates)
    ns = fit_nelson_siegel(rates)
    r2 = rates.get("2Y", 0.0)
    r10 = rates.get("10Y", 0.0)
    r3m = rates.get("3M", 0.0)

    return {
        "as_of": target.isoformat(),
        "regime": regime,
        "spread_2s10s_bp": round((r10 - r2) * 100, 2) if r2 and r10 else None,
        "spread_3m10y_bp": round((r10 - r3m) * 100, 2) if r3m and r10 else None,
        "rates": rates,
        "nelson_siegel": ns.model_dump(mode="json"),
        "regime_thresholds": REGIME_THRESHOLDS,
    }


@yield_curve_router.get("/signals")
def get_trading_signals(
    as_of: Optional[str] = Query(default=None, description="Date YYYY-MM-DD"),
) -> dict[str, Any]:
    """Get curve trading signals (steepeners, flatteners, carry, butterfly trades)."""
    target = _parse_date_param(as_of)
    sigs = _analytics.signals.generate_signals(target)
    rates = _analytics.data.fetch_curve(target)
    barbell = _analytics.signals.barbell_vs_bullet(rates)
    return {
        "as_of": target.isoformat(),
        "signals": [s.model_dump(mode="json") for s in sigs],
        "barbell_vs_bullet": barbell,
    }


@yield_curve_router.get("/global-comparison")
def get_global_comparison(
    as_of: Optional[str] = Query(default=None, description="Date YYYY-MM-DD"),
) -> dict[str, Any]:
    """Get global yield curve comparison vs Bunds, Gilts, JGBs."""
    target = _parse_date_param(as_of)
    result = _analytics.global_compare.compute(target)
    return result.model_dump(mode="json")


@yield_curve_router.get("/history/{spread_name}")
def get_spread_history(
    spread_name: str,
    days: int = Query(default=365, ge=30, le=365 * 15, description="Days of history"),
) -> dict[str, Any]:
    """Get historical spread data for a named spread."""
    valid_spreads = list(SpreadDashboard.SPREAD_DEFINITIONS.keys()) + [
        "curvature_2s5s10s", "curvature_2s10s30s"
    ]
    if spread_name not in valid_spreads:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown spread: {spread_name}. Valid: {valid_spreads}",
        )
    df = _analytics.spread_history_df(spread_name, days)
    if df.empty:
        return {"spread_name": spread_name, "days": days, "data": [], "message": "No cached history"}

    values = df["spread_bp"].round(2).tolist()
    dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in df["date"]]
    return {
        "spread_name": spread_name,
        "days": days,
        "count": len(values),
        "current_bp": values[-1] if values else None,
        "min_bp": round(min(values), 2) if values else None,
        "max_bp": round(max(values), 2) if values else None,
        "mean_bp": round(float(np.mean(values)), 2) if values else None,
        "dates": dates,
        "values": values,
    }


@yield_curve_router.get("/full")
def get_full_snapshot(
    as_of: Optional[str] = Query(default=None, description="Date YYYY-MM-DD"),
) -> dict[str, Any]:
    """Run all yield curve analytics in one call."""
    target = _parse_date_param(as_of)
    return _analytics.full_snapshot(target)


@yield_curve_router.get("/nelson-siegel")
def get_nelson_siegel(
    as_of: Optional[str] = Query(default=None, description="Date YYYY-MM-DD"),
) -> dict[str, Any]:
    """Fit Nelson-Siegel model to current curve and return parameters."""
    target = _parse_date_param(as_of)
    rates = _analytics.data.fetch_curve(target)
    ns = fit_nelson_siegel(rates)

    # Generate a smooth fitted curve for visualisation
    tenors_plot = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    fitted = _nelson_siegel(tenors_plot, ns.beta_0, ns.beta_1, ns.beta_2, ns.lambda_)

    return {
        "as_of": target.isoformat(),
        "parameters": ns.model_dump(mode="json"),
        "fitted_curve": {
            "maturities_yr": tenors_plot.round(4).tolist(),
            "yields_pct": fitted.round(4).tolist(),
        },
        "actual_rates": rates,
    }


@yield_curve_router.get("/percentiles")
def get_percentile_table(
    as_of: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    """Get historical percentile rankings for all spreads."""
    target = _parse_date_param(as_of)
    return {
        "as_of": target.isoformat(),
        "percentiles": _analytics.percentile_table(target),
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_date_param(date_str: Optional[str]) -> date:
    if not date_str:
        return date.today()
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str}. Use YYYY-MM-DD")


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_current_curve(as_of: Optional[date] = None) -> dict[str, float]:
    """Fetch current Treasury yield curve. Returns tenor -> rate (%)."""
    return TreasuryCurveData().fetch_curve(as_of)


def get_spread_dashboard(as_of: Optional[date] = None) -> SpreadDashboardResult:
    """Get full spread dashboard."""
    data = TreasuryCurveData()
    return SpreadDashboard(data).snapshot(as_of)


def get_recession_probability(as_of: Optional[date] = None) -> RecessionProbability:
    """Get NY Fed recession probability."""
    data = TreasuryCurveData()
    return RecessionProbabilityModel().compute(data, as_of)


def get_trading_signals(as_of: Optional[date] = None) -> list[CurveTradingSignal]:
    """Get curve trading signals."""
    data = TreasuryCurveData()
    return CurveTradingSignals(data).generate_signals(as_of)


def get_global_comparison(as_of: Optional[date] = None) -> GlobalCurveComparison:
    """Get international yield curve comparison."""
    data = TreasuryCurveData()
    return GlobalCurveComparison(data).compute(as_of)


def full_analytics(as_of: Optional[date] = None) -> dict[str, Any]:
    """Run all yield curve analytics in one call."""
    return YieldCurveAnalytics().full_snapshot(as_of)
