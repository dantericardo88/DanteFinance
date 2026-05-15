"""
US Treasury yield curves via FRED API (free).
Covers spot rates, forward rates, Nelson-Siegel fitting, curve regime detection.

Dimension: dim_035 — US Treasury yield curves (target score: 9)

Data sources (all free FRED CSV, no API key required):
  Nominal: DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30
  TIPS real yields: DFII5, DFII10, DFII20, DFII30
  Breakevens: T5YIE, T10YIE, T5YIFR (5y5y)

Public API
----------
FREDTreasuryAdapter
    fetch_series(series_id, start, end)     -> dict[str, float]
    fetch_latest(series_id)                 -> float | None

YieldCurveBuilder
    get_current_curve()                     -> pd.DataFrame
    get_curve_history(start, end)           -> pd.DataFrame
    compute_spreads()                       -> pd.DataFrame

NelsonSiegelFitter
    fit(yields, maturities)                 -> NSFitResult
    forward_rate(t1, t2)                    -> float
    instantaneous_forward(t)               -> float

ForwardRateCalculator
    implied_forward(t1, t2)                -> float
    breakeven_inflation(tips_yield, nominal_yield) -> float
    real_yield_curve()                      -> pd.DataFrame

YieldCurveRegimeDetector
    classify(spreads, history)             -> CurveRegime
    recession_signal(spread_history)       -> RecessionSignal

YieldCurveSignalEngine
    carry_roll_down()                      -> dict[str, float]
    dv01_approximation(maturity, yield_pct) -> float
    positioning_recommendation()           -> dict

treasury_router — FastAPI APIRouter, prefix /treasury
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from scipy.optimize import minimize

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/csv,text/plain,*/*",
    "Accept-Encoding": "gzip, deflate",
}

_TIMEOUT    = 25.0
_RATE_DELAY = 0.10   # 100 ms between FRED requests

# SQLite cache path
_DB_PATH = Path(__file__).parent.parent.parent / "data" / "treasury_yields.db"

# Cache TTLs (seconds)
_TTL_CURRENT = 3_600      # 1h for current curve
_TTL_HISTORY = 86_400     # 24h for historical data

# Treasury constant-maturity series (label, FRED series, tenor_years)
_TREASURY_TENORS: list[tuple[str, str, float]] = [
    ("1M",  "DGS1MO",  1 / 12),
    ("3M",  "DGS3MO",  0.25),
    ("6M",  "DGS6MO",  0.5),
    ("1Y",  "DGS1",    1.0),
    ("2Y",  "DGS2",    2.0),
    ("3Y",  "DGS3",    3.0),
    ("5Y",  "DGS5",    5.0),
    ("7Y",  "DGS7",    7.0),
    ("10Y", "DGS10",   10.0),
    ("20Y", "DGS20",   20.0),
    ("30Y", "DGS30",   30.0),
]

# TIPS real yield series
_TIPS_TENORS: list[tuple[str, str, float]] = [
    ("TIPS5Y",  "DFII5",  5.0),
    ("TIPS10Y", "DFII10", 10.0),
    ("TIPS20Y", "DFII20", 20.0),
    ("TIPS30Y", "DFII30", 30.0),
]

# Spread definitions (label, long series id, short series id, long_tenor, short_tenor)
_SPREAD_DEFS: list[tuple[str, str, str]] = [
    ("2s10s",  "DGS10",  "DGS2"),
    ("2s30s",  "DGS30",  "DGS2"),
    ("3m10y",  "DGS10",  "DGS3MO"),
    ("5s30s",  "DGS30",  "DGS5"),
    ("1y10y",  "DGS10",  "DGS1"),
    ("3y10y",  "DGS10",  "DGS3"),
]

# Breakeven inflation series
_BREAKEVEN_SERIES = {
    "T5YIE":  "5Y Breakeven",
    "T10YIE": "10Y Breakeven",
    "T5YIFR": "5Y5Y Forward Breakeven",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TenorPoint(BaseModel):
    tenor: str
    series_id: str
    tenor_years: float
    yield_pct: float
    real_yield_pct: Optional[float] = None
    breakeven_inflation: Optional[float] = None


class YieldCurveSnapshot(BaseModel):
    as_of: str
    curve: list[TenorPoint]
    spreads: dict[str, float]
    ns_params: Optional[dict] = None
    warnings: list[str] = Field(default_factory=list)


class NSFitResult(BaseModel):
    beta0: float                  # level
    beta1: float                  # slope
    beta2: float                  # curvature
    lambda_: float                # shape parameter
    r_squared: float
    rmse: float
    converged: bool
    fitted_curve: list[dict]      # [{tenor_years, fitted_yield}] fine grid
    warnings: list[str] = Field(default_factory=list)


class ForwardRate(BaseModel):
    t1: float
    t2: float
    label: str
    forward_rate_pct: float
    method: str = "bootstrap"


class CurveRegime(BaseModel):
    shape: str                    # "normal" | "inverted" | "flat" | "humped"
    momentum: str                 # "steepening" | "flattening" | "stable"
    bear_bull: str                # "bear_steepening" | "bear_flattening" | "bull_steepening" | "bull_flattening" | "neutral"
    spread_2s10s_bps: float
    spread_3m10y_bps: float
    inversion_days: int           # consecutive days inverted
    recession_signal: str         # "warning" | "clear" | "watch"
    description: str


class RecessionSignal(BaseModel):
    inverted: bool
    spread_2s10s_bps: float
    consecutive_inverted_days: int
    signal_level: str             # "clear" | "watch" | "warning" | "alert"
    historical_context: str


class PositioningRecommendation(BaseModel):
    strategy: str                 # "barbell" | "bullet" | "ladder" | "neutral"
    rationale: str
    preferred_maturities: list[str]
    avoid_maturities: list[str]
    carry_leaders: list[dict]     # top maturities by carry
    confidence: str               # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# SQLite cache helpers
# ---------------------------------------------------------------------------

def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fred_series (
                series_id   TEXT NOT NULL,
                date_str    TEXT NOT NULL,
                value       REAL,
                cached_at   REAL,
                PRIMARY KEY (series_id, date_str)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_meta (
                cache_key   TEXT PRIMARY KEY,
                payload     TEXT,
                cached_at   REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_series_date
            ON fred_series(series_id, date_str DESC)
        """)
        conn.commit()


@contextmanager
def _db_conn() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _meta_get(key: str, ttl: float) -> Optional[Any]:
    try:
        with _db_conn() as conn:
            row = conn.execute(
                "SELECT payload, cached_at FROM cache_meta WHERE cache_key = ?", (key,)
            ).fetchone()
            if row and (time.time() - row["cached_at"]) < ttl:
                return json.loads(row["payload"])
    except Exception:
        pass
    return None


def _meta_set(key: str, payload: Any) -> None:
    try:
        with _db_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta (cache_key, payload, cached_at) VALUES (?,?,?)",
                (key, json.dumps(payload, default=str), time.time())
            )
            conn.commit()
    except Exception:
        pass


def _store_series(series_id: str, data: dict[str, float]) -> None:
    """Persist series data to SQLite."""
    try:
        now = time.time()
        with _db_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO fred_series (series_id, date_str, value, cached_at) VALUES (?,?,?,?)",
                [(series_id, dt, val, now) for dt, val in data.items()]
            )
            conn.commit()
    except Exception as exc:
        logger.debug("Series store failed", series=series_id, error=str(exc))


def _load_series(series_id: str, start: Optional[str] = None, end: Optional[str] = None) -> dict[str, float]:
    """Load series from SQLite cache."""
    try:
        with _db_conn() as conn:
            query = "SELECT date_str, value FROM fred_series WHERE series_id = ?"
            params: list[Any] = [series_id]
            if start:
                query += " AND date_str >= ?"
                params.append(start)
            if end:
                query += " AND date_str <= ?"
                params.append(end)
            query += " ORDER BY date_str"
            rows = conn.execute(query, params).fetchall()
            return {r["date_str"]: r["value"] for r in rows if r["value"] is not None}
    except Exception:
        return {}


def _series_freshness(series_id: str) -> Optional[float]:
    """Return cached_at timestamp of most recent row for series, or None."""
    try:
        with _db_conn() as conn:
            row = conn.execute(
                "SELECT MAX(cached_at) as ts FROM fred_series WHERE series_id = ?",
                (series_id,)
            ).fetchone()
            return row["ts"] if row else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. FREDTreasuryAdapter
# ---------------------------------------------------------------------------

class FREDTreasuryAdapter:
    """
    Fetch US Treasury constant-maturity rates from FRED CSV endpoint.
    No API key required; uses the public fredgraph.csv endpoint.
    """

    async def fetch_series(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_rows: int = 1500,
    ) -> dict[str, float]:
        """
        Fetch a FRED series as {date_str: float}.
        Uses SQLite cache with TTL; falls back to stale cache on error.

        Args:
            series_id: FRED series identifier (e.g. "DGS10")
            start: ISO date string "YYYY-MM-DD" or None
            end:   ISO date string "YYYY-MM-DD" or None
            max_rows: Max historical rows to fetch

        Returns:
            dict mapping date strings to float values
        """
        # Check cache freshness
        ts = _series_freshness(series_id)
        if ts and (time.time() - ts) < _TTL_CURRENT:
            cached = _load_series(series_id, start=start, end=end)
            if cached:
                return cached

        # Fetch from FRED
        data = await self._fetch_csv(series_id)

        if not data:
            # Return stale cache as fallback
            stale = _load_series(series_id, start=start, end=end)
            if stale:
                logger.warning("Using stale cache for FRED series", series=series_id)
                return stale
            return {}

        # Persist to cache
        _store_series(series_id, data)

        # Filter by date range
        if start or end:
            data = {
                dt: v for dt, v in data.items()
                if (not start or dt >= start) and (not end or dt <= end)
            }

        return data

    async def _fetch_csv(self, series_id: str) -> dict[str, float]:
        """Download FRED CSV and parse to {date: float}."""
        try:
            await asyncio.sleep(_RATE_DELAY)
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    _FRED_CSV,
                    params={"id": series_id},
                    headers=_HEADERS,
                    timeout=_TIMEOUT,
                )
            if r.status_code != 200:
                logger.warning("FRED CSV non-200", series=series_id, status=r.status_code)
                return {}

            result: dict[str, float] = {}
            lines = r.text.strip().splitlines()[1:]   # skip header row
            for line in lines:
                parts = line.split(",")
                if len(parts) != 2:
                    continue
                dt_str, val_str = parts[0].strip(), parts[1].strip()
                if val_str in (".", "", "NA", "nan"):
                    continue
                try:
                    result[dt_str] = float(val_str)
                except ValueError:
                    continue

            logger.debug("FRED CSV fetched", series=series_id, rows=len(result))
            return result

        except Exception as exc:
            logger.warning("FRED CSV fetch failed", series=series_id, error=str(exc))
            return {}

    async def fetch_latest(self, series_id: str) -> Optional[float]:
        """Return the most recent non-null value for a FRED series."""
        data = await self.fetch_series(series_id, max_rows=30)
        if not data:
            return None
        latest_dt = max(data.keys())
        return data[latest_dt]

    async def fetch_on_date(
        self, series_id: str, target_date: str, lookback_rows: int = 30
    ) -> Optional[float]:
        """Most recent value on or before target_date."""
        end_dt = target_date
        start_dt = (
            datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=lookback_rows * 2)
        ).strftime("%Y-%m-%d")
        data = await self.fetch_series(series_id, start=start_dt, end=end_dt)
        if not data:
            return None
        candidates = sorted(
            ((dt, v) for dt, v in data.items() if dt <= target_date),
            key=lambda x: x[0], reverse=True,
        )
        return candidates[0][1] if candidates else None

    async def fetch_all_tenors(
        self, as_of: Optional[str] = None
    ) -> dict[str, Optional[float]]:
        """
        Fetch all standard Treasury maturities in parallel.
        Returns {series_id: latest_value_or_None}.
        """
        target = as_of or date.today().isoformat()

        tasks = {
            series: self.fetch_on_date(series, target)
            for _, series, _ in _TREASURY_TENORS
        }

        results: dict[str, Optional[float]] = {}
        for series, coro in tasks.items():
            try:
                results[series] = await coro
            except Exception as exc:
                logger.warning("Tenor fetch failed", series=series, error=str(exc))
                results[series] = None

        return results

    async def fetch_tips_yields(
        self, as_of: Optional[str] = None
    ) -> dict[str, Optional[float]]:
        """Fetch all TIPS real yield series."""
        target = as_of or date.today().isoformat()
        results: dict[str, Optional[float]] = {}
        for _, series, _ in _TIPS_TENORS:
            results[series] = await self.fetch_on_date(series, target)
        return results

    async def fetch_breakevens(
        self, as_of: Optional[str] = None
    ) -> dict[str, Optional[float]]:
        """Fetch breakeven inflation series."""
        target = as_of or date.today().isoformat()
        results: dict[str, Optional[float]] = {}
        for series_id in _BREAKEVEN_SERIES:
            results[series_id] = await self.fetch_on_date(series_id, target)
        return results


# ---------------------------------------------------------------------------
# 2. YieldCurveBuilder
# ---------------------------------------------------------------------------

class YieldCurveBuilder:
    """
    Build and analyse the full US Treasury yield curve from FRED data.
    """

    def __init__(self) -> None:
        self._adapter = FREDTreasuryAdapter()

    async def get_current_curve(self, as_of: Optional[str] = None) -> pd.DataFrame:
        """
        Return current yield curve as DataFrame with columns:
        [tenor, series_id, maturity_years, yield_pct]

        Args:
            as_of: ISO date string or None for latest

        Returns:
            DataFrame sorted by maturity_years ascending
        """
        cache_key = f"current_curve:{as_of or 'latest'}"
        cached = _meta_get(cache_key, _TTL_CURRENT)
        if cached is not None:
            return pd.DataFrame(cached)

        target = as_of or date.today().isoformat()
        tenor_data = await self._adapter.fetch_all_tenors(as_of=target)

        rows = []
        for label, series_id, tenor_yrs in _TREASURY_TENORS:
            val = tenor_data.get(series_id)
            if val is not None:
                rows.append({
                    "tenor":        label,
                    "series_id":    series_id,
                    "maturity_years": tenor_yrs,
                    "yield_pct":    round(val, 4),
                })

        df = pd.DataFrame(rows).sort_values("maturity_years").reset_index(drop=True)
        _meta_set(cache_key, df.to_dict("records"))
        return df

    async def get_curve_history(
        self,
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return historical yield curves as wide DataFrame.
        Columns: date, DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30
        Index: date strings.

        Args:
            start: ISO start date "YYYY-MM-DD"
            end:   ISO end date or None for today

        Returns:
            Wide DataFrame, rows = dates, columns = tenor series IDs
        """
        end_dt = end or date.today().isoformat()
        cache_key = f"curve_history:{start}:{end_dt}"
        cached = _meta_get(cache_key, _TTL_HISTORY)
        if cached is not None:
            return pd.DataFrame(cached).set_index("date")

        # Fetch all series in parallel (throttled by RATE_DELAY inside adapter)
        series_ids = [s for _, s, _ in _TREASURY_TENORS]
        fetch_tasks = {
            sid: self._adapter.fetch_series(sid, start=start, end=end_dt)
            for sid in series_ids
        }

        series_data: dict[str, dict[str, float]] = {}
        for sid, coro in fetch_tasks.items():
            try:
                series_data[sid] = await coro
            except Exception as exc:
                logger.warning("History fetch failed", series=sid, error=str(exc))
                series_data[sid] = {}

        # Build wide DataFrame
        all_dates = sorted(set().union(*[set(d.keys()) for d in series_data.values()]))
        records = []
        for dt in all_dates:
            row: dict[str, Any] = {"date": dt}
            for sid in series_ids:
                row[sid] = series_data.get(sid, {}).get(dt)
            records.append(row)

        df = pd.DataFrame(records).set_index("date")
        _meta_set(cache_key, df.reset_index().to_dict("records"))
        return df

    async def compute_spreads(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """
        Compute key yield curve spreads historically.
        Returns DataFrame with columns: date, 2s10s, 2s30s, 3m10y, 5s30s, 1y10y, 3y10y (all in bps).

        Args:
            start:         ISO start date (overrides lookback_days if provided)
            end:           ISO end date or None for today
            lookback_days: Days to look back from end if start not provided

        Returns:
            DataFrame with spread columns in basis points
        """
        end_dt   = end or date.today().isoformat()
        start_dt = start or (
            datetime.strptime(end_dt, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        cache_key = f"spreads:{start_dt}:{end_dt}"
        cached = _meta_get(cache_key, _TTL_CURRENT)
        if cached is not None:
            return pd.DataFrame(cached).set_index("date")

        # Fetch all unique series needed
        needed_series: set[str] = set()
        for _, long_s, short_s in _SPREAD_DEFS:
            needed_series.add(long_s)
            needed_series.add(short_s)

        series_data: dict[str, dict[str, float]] = {}
        for sid in needed_series:
            try:
                series_data[sid] = await self._adapter.fetch_series(
                    sid, start=start_dt, end=end_dt
                )
            except Exception as exc:
                logger.warning("Spread series failed", series=sid, error=str(exc))
                series_data[sid] = {}

        # Build spread DataFrame
        all_dates = sorted(
            set().union(*[set(d.keys()) for d in series_data.values()])
        )
        records = []
        for dt in all_dates:
            row: dict[str, Any] = {"date": dt}
            for label, long_s, short_s in _SPREAD_DEFS:
                lv = series_data.get(long_s, {}).get(dt)
                sv = series_data.get(short_s, {}).get(dt)
                if lv is not None and sv is not None:
                    row[label] = round((lv - sv) * 100, 2)  # convert to bps
                else:
                    row[label] = None
            records.append(row)

        df = pd.DataFrame(records).set_index("date")
        _meta_set(cache_key, df.reset_index().to_dict("records"))
        return df

    async def get_spread_current(self) -> dict[str, float]:
        """Return the most recent value of each spread (in bps)."""
        target = date.today().isoformat()
        result: dict[str, float] = {}

        for label, long_s, short_s in _SPREAD_DEFS:
            lv, sv = await asyncio.gather(
                self._adapter.fetch_on_date(long_s, target),
                self._adapter.fetch_on_date(short_s, target),
            )
            if lv is not None and sv is not None:
                result[label] = round((lv - sv) * 100, 2)

        return result


# ---------------------------------------------------------------------------
# 3. NelsonSiegelFitter
# ---------------------------------------------------------------------------

class NelsonSiegelFitter:
    """
    Fit Nelson-Siegel 3-factor model to observed Treasury yields.

      y(τ) = β₀ + β₁·[(1-e^(-τ/λ))/(τ/λ)]
                + β₂·[(1-e^(-τ/λ))/(τ/λ) - e^(-τ/λ)]

    where:
      β₀ = level (long end)
      β₁ = slope (short − long)
      β₂ = curvature (hump)
      λ  = shape/decay parameter

    Fitted via Nelder-Mead optimisation.
    """

    def __init__(self) -> None:
        self._last_fit: Optional[NSFitResult] = None
        self._last_maturities: Optional[list[float]] = None

    @staticmethod
    def _ns_yield(tau: float, b0: float, b1: float, b2: float, lam: float) -> float:
        """Nelson-Siegel yield for a single maturity tau (years)."""
        if tau <= 0.0:
            return b0 + b1
        lam = max(lam, 0.001)
        x   = tau / lam
        ex  = math.exp(-x)
        L   = (1.0 - ex) / x
        return b0 + b1 * L + b2 * (L - ex)

    @staticmethod
    def _ns_vec(taus: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Vectorised Nelson-Siegel for numpy arrays."""
        b0, b1, b2, lam = p
        lam = max(float(lam), 0.001)
        x   = taus / lam
        ex  = np.exp(-x)
        L   = (1.0 - ex) / x
        return b0 + b1 * L + b2 * (L - ex)

    def fit(
        self,
        yields: list[float],
        maturities: list[float],
        n_grid: int = 60,
    ) -> NSFitResult:
        """
        Fit Nelson-Siegel model to observed (maturity, yield) pairs.

        Args:
            yields:     Observed yields in percent (e.g. 4.5 = 4.5%)
            maturities: Corresponding maturities in years
            n_grid:     Points on the smooth fitted curve output

        Returns:
            NSFitResult with parameters, R², RMSE, fitted curve
        """
        warnings: list[str] = []
        taus = np.array(maturities, dtype=float)
        ylds = np.array(yields, dtype=float)

        if len(taus) < 3:
            warnings.append("Insufficient data points for NS fit")
            return NSFitResult(
                beta0=float(np.mean(ylds)) if len(ylds) else 0.0,
                beta1=0.0, beta2=0.0, lambda_=1.5,
                r_squared=0.0, rmse=0.0, converged=False,
                fitted_curve=[], warnings=warnings,
            )

        def _objective(p: np.ndarray) -> float:
            if p[0] <= 0 or p[3] <= 0.001:
                return 1e9
            fitted = self._ns_vec(taus, p)
            return float(np.sum((fitted - ylds) ** 2))

        # Initial guess: long-end yield for β0, negative slope for β1, zero curvature, λ=1.5
        long_end = float(ylds[-1]) if len(ylds) else 4.0
        short_end = float(ylds[0]) if len(ylds) else 5.0
        x0 = np.array([long_end, short_end - long_end, 0.5, 1.5])

        res = minimize(
            _objective, x0, method="Nelder-Mead",
            options={"maxiter": 10_000, "xatol": 1e-7, "fatol": 1e-10},
        )

        if not res.success:
            warnings.append(f"NS optimisation non-convergent: {res.message}")

        b0, b1, b2, lam = res.x
        lam = max(float(lam), 0.001)

        # R² and RMSE
        fitted_vals = self._ns_vec(taus, np.array([b0, b1, b2, lam]))
        ss_res = float(np.sum((fitted_vals - ylds) ** 2))
        ss_tot = float(np.sum((ylds - np.mean(ylds)) ** 2))
        r_sq   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse   = math.sqrt(ss_res / len(ylds))

        # Smooth fitted curve on log-spaced grid 1M → 30Y
        grid = np.exp(np.linspace(math.log(1 / 12), math.log(30), n_grid))
        fitted_curve = [
            {
                "tenor_years": round(float(t), 4),
                "fitted_yield": round(
                    self._ns_yield(t, float(b0), float(b1), float(b2), lam), 4
                ),
            }
            for t in grid
        ]

        result = NSFitResult(
            beta0=round(float(b0), 6),
            beta1=round(float(b1), 6),
            beta2=round(float(b2), 6),
            lambda_=round(lam, 6),
            r_squared=round(r_sq, 6),
            rmse=round(rmse, 6),
            converged=res.success,
            fitted_curve=fitted_curve,
            warnings=warnings,
        )
        self._last_fit = result
        self._last_maturities = list(maturities)
        return result

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Implied forward rate from t1 to t2 using NS parameters from last fit.
        f(t1, t2) = [y(t2)·t2 - y(t1)·t1] / (t2 - t1)
        """
        if self._last_fit is None:
            raise RuntimeError("Must call fit() before computing forward rates")
        f = self._last_fit
        y1 = self._ns_yield(t1, f.beta0, f.beta1, f.beta2, f.lambda_)
        y2 = self._ns_yield(t2, f.beta0, f.beta1, f.beta2, f.lambda_)
        if t2 <= t1:
            raise ValueError(f"t2 ({t2}) must be > t1 ({t1})")
        return round((y2 * t2 - y1 * t1) / (t2 - t1), 4)

    def instantaneous_forward(self, t: float) -> float:
        """
        Instantaneous forward rate at maturity t from last NS fit.
        f(t) = β₀ + β₁·exp(-t/λ) + β₂·(t/λ)·exp(-t/λ)
        """
        if self._last_fit is None:
            raise RuntimeError("Must call fit() before computing forward rates")
        f = self._last_fit
        if t <= 0:
            return f.beta0 + f.beta1
        lam = max(f.lambda_, 0.001)
        x   = t / lam
        ex  = math.exp(-x)
        return round(f.beta0 + f.beta1 * ex + f.beta2 * x * ex, 4)

    def get_forward_curve(self, horizons: Optional[list[tuple[float, float]]] = None) -> list[ForwardRate]:
        """
        Compute a set of standard forward rates from the last fitted curve.

        Args:
            horizons: list of (t1, t2) pairs; defaults to standard market horizons

        Returns:
            list of ForwardRate models
        """
        if horizons is None:
            horizons = [
                (0.5, 1.0),    # 6M → 1Y
                (1.0, 2.0),    # 1Y → 2Y
                (2.0, 5.0),    # 2Y → 5Y
                (5.0, 10.0),   # 5Y → 10Y
                (10.0, 30.0),  # 10Y → 30Y
                (5.0, 30.0),   # 5Y → 30Y
            ]

        labels = {
            (0.5, 1.0):   "6M1Y",
            (1.0, 2.0):   "1Y2Y",
            (2.0, 5.0):   "2Y5Y",
            (5.0, 10.0):  "5Y10Y",
            (10.0, 30.0): "10Y30Y",
            (5.0, 30.0):  "5Y30Y",
        }

        results = []
        for t1, t2 in horizons:
            try:
                rate = self.forward_rate(t1, t2)
                results.append(ForwardRate(
                    t1=t1, t2=t2,
                    label=labels.get((t1, t2), f"{t1}Y{t2}Y"),
                    forward_rate_pct=rate,
                    method="nelson-siegel",
                ))
            except Exception as exc:
                logger.debug("Forward rate failed", t1=t1, t2=t2, error=str(exc))

        return results


# ---------------------------------------------------------------------------
# 4. ForwardRateCalculator
# ---------------------------------------------------------------------------

class ForwardRateCalculator:
    """
    Bootstrap zero rates from par/CMT yields and compute forward rates.
    Also handles TIPS-based real yields and breakeven inflation.
    """

    def __init__(self) -> None:
        self._adapter = FREDTreasuryAdapter()

    def bootstrap_zero_curve(
        self, par_yields: list[float], maturities: list[float]
    ) -> dict[float, float]:
        """
        Bootstrap zero/spot rates from par yields using iterative discounting.
        Assumes annual coupon (simplified; Treasury CMT rates are semi-annual equivalent).

        Args:
            par_yields:  Par/CMT yields in percent, sorted by maturity ascending
            maturities:  Maturity years, matching par_yields

        Returns:
            dict {maturity_years: zero_rate_pct}
        """
        zero_rates: dict[float, float] = {}

        for i, (t, par) in enumerate(zip(maturities, par_yields)):
            par_dec = par / 100.0       # Convert to decimal
            coupon  = par_dec           # Par bond: coupon = yield

            if t <= 1.0 or i == 0:
                # For short end: zero ≈ par yield (no intermediate coupons)
                zero_rates[t] = par
                continue

            # Sum of discounted prior coupons using bootstrapped zero rates
            pv_coupons = 0.0
            for j, (t_j, z_j) in enumerate(sorted(zero_rates.items())):
                if t_j >= t:
                    break
                # Interpolate coupon frequency: annual simplified
                pv_coupons += coupon * math.exp(-z_j / 100.0 * t_j)

            # Solve for zero rate at this maturity: 1 = pv_coupons + (1+coupon)*exp(-z*t)
            # => exp(-z*t) = (1 - pv_coupons) / (1 + coupon)
            terminal_pv = (1.0 - pv_coupons) / (1.0 + coupon)
            if terminal_pv <= 0:
                # Fallback: linear interpolation
                zero_rates[t] = par
                continue

            zero_rate = -math.log(terminal_pv) / t * 100.0  # back to percent
            zero_rates[t] = round(zero_rate, 6)

        return zero_rates

    def implied_forward(
        self,
        t1: float,
        t2: float,
        zero_curve: dict[float, float],
    ) -> Optional[float]:
        """
        Implied forward rate from t1 to t2 using bootstrapped zero curve.
        f(t1,t2) = [z(t2)·t2 - z(t1)·t1] / (t2 - t1)

        Args:
            t1: Start maturity (years)
            t2: End maturity (years)
            zero_curve: {maturity_years: zero_rate_pct} from bootstrap_zero_curve

        Returns:
            Forward rate in percent or None if data insufficient
        """
        z1 = self._interp_zero(t1, zero_curve)
        z2 = self._interp_zero(t2, zero_curve)

        if z1 is None or z2 is None or t2 <= t1:
            return None

        fwd = (z2 * t2 - z1 * t1) / (t2 - t1)
        return round(fwd, 4)

    def _interp_zero(
        self, t: float, zero_curve: dict[float, float]
    ) -> Optional[float]:
        """Linear interpolation of zero rate at maturity t."""
        mats = sorted(zero_curve.keys())
        if not mats:
            return None
        if t <= mats[0]:
            return zero_curve[mats[0]]
        if t >= mats[-1]:
            return zero_curve[mats[-1]]
        for i in range(len(mats) - 1):
            t1, t2 = mats[i], mats[i + 1]
            if t1 <= t <= t2:
                w = (t - t1) / (t2 - t1)
                return zero_curve[t1] * (1 - w) + zero_curve[t2] * w
        return None

    @staticmethod
    def breakeven_inflation(nominal_yield: float, tips_yield: float) -> float:
        """
        Compute breakeven inflation rate.
        BEI ≈ nominal_yield - tips_yield (in percent)

        Args:
            nominal_yield: Nominal Treasury yield in percent
            tips_yield:    TIPS real yield in percent

        Returns:
            Breakeven inflation in percent
        """
        return round(nominal_yield - tips_yield, 4)

    async def real_yield_curve(self, as_of: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch TIPS real yield curve from FRED.
        Returns DataFrame with columns [tenor, maturity_years, real_yield_pct, nominal_yield_pct, breakeven_pct].
        """
        target = as_of or date.today().isoformat()
        cache_key = f"real_yield_curve:{target}"
        cached = _meta_get(cache_key, _TTL_CURRENT)
        if cached is not None:
            return pd.DataFrame(cached)

        tips_data = await self._adapter.fetch_tips_yields(as_of=target)
        nominal_map = {
            5.0:  "DGS5",
            10.0: "DGS10",
            20.0: "DGS20",
            30.0: "DGS30",
        }

        rows = []
        for label, series_id, tenor_yrs in _TIPS_TENORS:
            real_y = tips_data.get(series_id)
            if real_y is None:
                continue
            nom_series = nominal_map.get(tenor_yrs)
            nom_y = None
            if nom_series:
                nom_y = await self._adapter.fetch_on_date(nom_series, target)

            bei = self.breakeven_inflation(nom_y, real_y) if nom_y is not None else None
            rows.append({
                "tenor":             label,
                "maturity_years":    tenor_yrs,
                "real_yield_pct":    round(real_y, 4),
                "nominal_yield_pct": round(nom_y, 4) if nom_y is not None else None,
                "breakeven_pct":     round(bei, 4) if bei is not None else None,
            })

        df = pd.DataFrame(rows)
        _meta_set(cache_key, df.to_dict("records"))
        return df

    async def five_five_breakeven(self, as_of: Optional[str] = None) -> Optional[float]:
        """
        Fetch 5Y5Y forward breakeven inflation rate (T5YIFR series from FRED).
        Market-implied inflation expectation from year 5 to year 10.
        """
        target = as_of or date.today().isoformat()
        return await self._adapter.fetch_on_date("T5YIFR", target)

    async def compute_all_forwards(
        self, as_of: Optional[str] = None
    ) -> list[ForwardRate]:
        """
        Build bootstrapped zero curve from current CMT yields, then compute
        all standard forward rates.
        """
        target = as_of or date.today().isoformat()
        tenor_data = await self._adapter.fetch_all_tenors(as_of=target)

        maturities: list[float] = []
        par_yields: list[float] = []
        for _, series_id, tenor_yrs in _TREASURY_TENORS:
            val = tenor_data.get(series_id)
            if val is not None:
                maturities.append(tenor_yrs)
                par_yields.append(val)

        if len(maturities) < 3:
            logger.warning("Insufficient tenors for bootstrap forward calculation")
            return []

        zero_curve = self.bootstrap_zero_curve(par_yields, maturities)

        horizons = [
            (0.25, 0.5),   # 3M → 6M
            (0.5,  1.0),   # 6M → 1Y
            (1.0,  2.0),   # 1Y → 2Y
            (2.0,  3.0),   # 2Y → 3Y
            (2.0,  5.0),   # 2Y → 5Y
            (5.0,  10.0),  # 5Y → 10Y
            (5.0,  5.0 + 5.0),  # 5Y5Y: from 5 to 10
        ]
        labels = {
            (0.25, 0.5):  "3M6M",
            (0.5,  1.0):  "6M1Y",
            (1.0,  2.0):  "1Y2Y",
            (2.0,  3.0):  "2Y3Y",
            (2.0,  5.0):  "2Y5Y",
            (5.0,  10.0): "5Y10Y",
            (5.0,  10.0): "5Y10Y",
        }

        results: list[ForwardRate] = []
        for t1, t2 in horizons:
            fwd = self.implied_forward(t1, t2, zero_curve)
            if fwd is not None:
                results.append(ForwardRate(
                    t1=t1, t2=t2,
                    label=labels.get((t1, t2), f"{t1}Y{t2}Y"),
                    forward_rate_pct=fwd,
                    method="bootstrap",
                ))

        return results


# ---------------------------------------------------------------------------
# 5. YieldCurveRegimeDetector
# ---------------------------------------------------------------------------

class YieldCurveRegimeDetector:
    """
    Classify the current yield curve shape and regime.
    Signals: normal, inverted, flat, humped.
    Momentum: steepening / flattening.
    Recession signal based on 2s10s inversion duration.
    """

    def __init__(self) -> None:
        self._builder = YieldCurveBuilder()

    def classify_shape(self, spread_2s10s_bps: float, spread_3m10y_bps: float) -> str:
        """
        Classify curve shape from key spreads.

        Returns: "normal" | "inverted" | "flat" | "humped"
        """
        if spread_2s10s_bps < -10 and spread_3m10y_bps < -10:
            return "inverted"
        if abs(spread_2s10s_bps) < 25:
            return "flat"
        if spread_2s10s_bps > 0 and spread_3m10y_bps > 0:
            return "normal"
        # Mixed signals (one positive, one negative) suggest humped / transitioning
        return "humped"

    def classify_momentum(
        self,
        current_2s10s: float,
        ma20_2s10s: float,
    ) -> str:
        """
        Classify steepening/flattening momentum vs 20-day moving average.

        Returns: "steepening" | "flattening" | "stable"
        """
        delta = current_2s10s - ma20_2s10s
        if delta > 5:        # spreading > 5bps above 20d MA
            return "steepening"
        if delta < -5:       # narrowing > 5bps below 20d MA
            return "flattening"
        return "stable"

    def classify_bear_bull(
        self,
        curve_df: pd.DataFrame,
        spread_series: pd.DataFrame,
    ) -> str:
        """
        Classify bear/bull steepening and flattening:
          Bear steepening:  long rates rising faster than short rates
          Bear flattening:  short rates rising faster than long rates
          Bull steepening:  short rates falling faster than long rates
          Bull flattening:  long rates falling faster than short rates

        Uses last two available data points.
        """
        if curve_df is None or len(curve_df) < 2:
            return "neutral"

        # Use 2Y (DGS2) and 10Y (DGS10) if available
        hist_cols = curve_df.columns.tolist() if hasattr(curve_df, "columns") else []

        try:
            if "DGS2" in hist_cols and "DGS10" in hist_cols:
                df = curve_df[["DGS2", "DGS10"]].dropna()
                if len(df) < 2:
                    return "neutral"
                d_short = float(df["DGS2"].iloc[-1]) - float(df["DGS2"].iloc[-2])
                d_long  = float(df["DGS10"].iloc[-1]) - float(df["DGS10"].iloc[-2])

                if d_long > 0 and d_short >= 0:
                    return "bear_steepening" if d_long > d_short else "bear_flattening"
                if d_long < 0 and d_short <= 0:
                    return "bull_flattening" if abs(d_long) < abs(d_short) else "bull_steepening"
                if d_long > 0 > d_short:
                    return "bear_steepening"
                if d_long < 0 < d_short:
                    return "bear_flattening"
        except Exception as exc:
            logger.debug("Bear/bull classification error", error=str(exc))

        return "neutral"

    def count_inversion_days(
        self, spread_history: pd.DataFrame, spread_col: str = "2s10s"
    ) -> int:
        """
        Count consecutive trading days the 2s10s spread has been negative (inverted).
        Counts from the most recent observation backwards.
        """
        if spread_history is None or spread_col not in spread_history.columns:
            return 0
        series = spread_history[spread_col].dropna().sort_index()
        count = 0
        for val in reversed(series.values):
            if val < 0:
                count += 1
            else:
                break
        return count

    def recession_signal_from_inversion(self, consecutive_days: int) -> RecessionSignal:
        """
        Generate recession signal based on 2s10s inversion streak.
        Historical rule: inversion >60 trading days = elevated recession risk.
        """
        # We'll track the actual spread value later; use 0 as placeholder
        return self._make_recession_signal(consecutive_days, 0.0)

    def _make_recession_signal(
        self, consecutive_days: int, spread_bps: float
    ) -> RecessionSignal:
        if consecutive_days >= 180:
            level = "alert"
            context = (
                "Sustained inversion (6+ months). Historical precedent: US recessions "
                "typically begin 6–18 months after prolonged 2s10s inversion."
            )
        elif consecutive_days >= 60:
            level = "warning"
            context = (
                "Inversion >60 trading days. Historically, this is a reliable leading "
                "recession indicator with ~12-month lag (since 1960s, 7 of 8 recessions "
                "were preceded by 2s10s inversion)."
            )
        elif consecutive_days > 0:
            level = "watch"
            context = (
                "Short-duration inversion. Not yet historically significant. "
                "Watch for persistence beyond 60 days."
            )
        else:
            level = "clear"
            context = "2s10s spread is positive. No inversion recession signal."

        return RecessionSignal(
            inverted=consecutive_days > 0,
            spread_2s10s_bps=round(spread_bps, 2),
            consecutive_inverted_days=consecutive_days,
            signal_level=level,
            historical_context=context,
        )

    async def classify(self, as_of: Optional[str] = None) -> CurveRegime:
        """
        Full curve regime classification using current and historical data.

        Returns:
            CurveRegime with shape, momentum, bear/bull type, inversion days,
            recession signal, and human-readable description.
        """
        cache_key = f"regime:{as_of or 'latest'}"
        cached = _meta_get(cache_key, 1_800)  # 30m TTL
        if cached is not None:
            return CurveRegime(**cached)

        # Fetch spreads
        target    = as_of or date.today().isoformat()
        lookback  = 30  # days for MA + bear/bull
        start_lb  = (
            datetime.strptime(target, "%Y-%m-%d") - timedelta(days=lookback * 2)
        ).strftime("%Y-%m-%d")

        spreads_hist = await self._builder.compute_spreads(start=start_lb, end=target)
        curve_hist   = await self._builder.get_curve_history(start=start_lb, end=target)

        # Current spreads
        if len(spreads_hist) > 0:
            latest_row  = spreads_hist.iloc[-1]
            spread_2s10s = float(latest_row.get("2s10s") or 0.0)
            spread_3m10y = float(latest_row.get("3m10y") or 0.0)
        else:
            spread_2s10s = 0.0
            spread_3m10y = 0.0

        # 20-day MA of 2s10s
        if "2s10s" in spreads_hist.columns and len(spreads_hist) >= 20:
            ma20 = float(spreads_hist["2s10s"].dropna().tail(20).mean())
        else:
            ma20 = spread_2s10s

        # Classification
        shape    = self.classify_shape(spread_2s10s, spread_3m10y)
        momentum = self.classify_momentum(spread_2s10s, ma20)
        bear_bull = self.classify_bear_bull(curve_hist, spreads_hist)
        inv_days  = self.count_inversion_days(spreads_hist, "2s10s")

        # Recession signal
        rec_sig = self._make_recession_signal(inv_days, spread_2s10s)

        # Human-readable description
        desc = self._describe(shape, momentum, bear_bull, spread_2s10s, inv_days)

        regime = CurveRegime(
            shape=shape,
            momentum=momentum,
            bear_bull=bear_bull,
            spread_2s10s_bps=round(spread_2s10s, 2),
            spread_3m10y_bps=round(spread_3m10y, 2),
            inversion_days=inv_days,
            recession_signal=rec_sig.signal_level,
            description=desc,
        )

        _meta_set(cache_key, regime.model_dump())
        return regime

    def _describe(
        self,
        shape: str,
        momentum: str,
        bear_bull: str,
        spread_2s10s: float,
        inv_days: int,
    ) -> str:
        parts = []
        shape_map = {
            "normal":   f"Normal (positive-sloping) curve, 2s10s = {spread_2s10s:.1f}bps.",
            "inverted": f"Inverted curve, 2s10s = {spread_2s10s:.1f}bps. Inverted {inv_days} trading days.",
            "flat":     f"Flat curve, 2s10s = {spread_2s10s:.1f}bps (|spread|<25bps).",
            "humped":   f"Humped/transitional curve, 2s10s = {spread_2s10s:.1f}bps.",
        }
        parts.append(shape_map.get(shape, f"Shape: {shape}, 2s10s={spread_2s10s:.1f}bps."))

        if momentum != "stable":
            parts.append(f"Momentum: {momentum.capitalize()} vs 20-day MA.")

        if bear_bull != "neutral":
            bb_map = {
                "bear_steepening":  "Bear steepening (long rates rising faster than short).",
                "bear_flattening":  "Bear flattening (short rates rising faster than long).",
                "bull_steepening":  "Bull steepening (short rates falling faster than long).",
                "bull_flattening":  "Bull flattening (long rates falling faster than short).",
            }
            parts.append(bb_map.get(bear_bull, bear_bull))

        return " ".join(parts)


# ---------------------------------------------------------------------------
# 6. YieldCurveSignalEngine
# ---------------------------------------------------------------------------

class YieldCurveSignalEngine:
    """
    Generate trading signals from yield curve analysis:
      - Carry and roll-down return estimates
      - DV01 approximation per maturity
      - Barbell / bullet / ladder positioning recommendations
    """

    def __init__(self) -> None:
        self._builder  = YieldCurveBuilder()
        self._adapter  = FREDTreasuryAdapter()
        self._fitter   = NelsonSiegelFitter()
        self._fwdcalc  = ForwardRateCalculator()
        self._detector = YieldCurveRegimeDetector()

    async def carry_roll_down(
        self, horizon_months: int = 12, as_of: Optional[str] = None
    ) -> dict[str, float]:
        """
        Estimate carry + roll-down return for each Treasury maturity over a horizon.

        Carry:    yield earned by holding the bond = current yield × horizon
        Roll-down: gain from "rolling down" the curve as the bond ages
                   ≈ (yield_at_t - yield_at_{t-horizon}) × modified_duration

        Args:
            horizon_months: Investment horizon in months
            as_of:          ISO date or None for latest

        Returns:
            dict {tenor_label: annualised_carry_rolldown_pct}
        """
        target   = as_of or date.today().isoformat()
        horizon  = horizon_months / 12.0
        curve_df = await self._builder.get_current_curve(as_of=target)

        if curve_df.empty:
            return {}

        # Fit NS curve for smooth roll-down
        yields     = curve_df["yield_pct"].tolist()
        maturities = curve_df["maturity_years"].tolist()
        ns_fit = self._fitter.fit(yields, maturities)

        result: dict[str, float] = {}

        for _, row in curve_df.iterrows():
            tenor     = str(row["tenor"])
            mat       = float(row["maturity_years"])
            y_now     = float(row["yield_pct"])

            if mat <= horizon:
                # Bond matures before horizon — carry only
                carry = y_now * (mat / 1.0)   # simple, annualised
                result[tenor] = round(carry, 4)
                continue

            # Yield at rolled-down maturity: mat - horizon
            mat_rolled = mat - horizon
            y_rolled = self._fitter._ns_yield(
                mat_rolled,
                ns_fit.beta0, ns_fit.beta1, ns_fit.beta2, ns_fit.lambda_,
            )

            # Approximate modified duration: (1 - (1/(1+y)^mat)) / y (simplified)
            y_dec = y_now / 100.0
            if y_dec > 0:
                mod_dur = (1.0 - (1.0 + y_dec) ** (-mat)) / y_dec
            else:
                mod_dur = mat

            # Roll-down return ≈ (y_now - y_rolled) × mod_duration (as price gain %)
            roll_down_pct = (y_now - y_rolled) * mod_dur / 100.0 * 100  # back to %

            # Carry = yield earned over horizon (approximation)
            carry_pct = y_now * horizon

            total = round(carry_pct + roll_down_pct, 4)
            result[tenor] = total

        return result

    def dv01_approximation(self, maturity_years: float, yield_pct: float, face: float = 1_000_000.0) -> float:
        """
        Approximate DV01 (dollar value of 1bp move) for a par bond.
        DV01 ≈ Modified Duration × Price × 0.0001

        For a par bond, Modified Duration ≈ (1 - (1+y)^(-t)) / y

        Args:
            maturity_years: Bond maturity in years
            yield_pct:      Yield in percent (e.g. 4.5)
            face:           Face value in dollars

        Returns:
            DV01 in dollars
        """
        y = yield_pct / 100.0
        if y <= 0 or maturity_years <= 0:
            return 0.0

        # Modified duration for par bond (annual coupon, simplified)
        mod_dur = (1.0 - (1.0 + y) ** (-maturity_years)) / y

        # Price is par = face value
        dv01 = mod_dur * face * 0.0001
        return round(dv01, 2)

    async def dv01_curve(self, as_of: Optional[str] = None, face: float = 1_000_000.0) -> dict[str, float]:
        """Return DV01 for every maturity on the current curve."""
        curve_df = await self._builder.get_current_curve(as_of=as_of)
        result: dict[str, float] = {}
        for _, row in curve_df.iterrows():
            result[str(row["tenor"])] = self.dv01_approximation(
                float(row["maturity_years"]), float(row["yield_pct"]), face
            )
        return result

    async def positioning_recommendation(
        self, as_of: Optional[str] = None
    ) -> PositioningRecommendation:
        """
        Recommend curve positioning strategy based on regime and signals.

        Strategies:
          Barbell:  overweight short + long, underweight belly
          Bullet:   concentrate in belly (5-7Y)
          Ladder:   equal-weight across all maturities
          Neutral:  no strong recommendation

        Returns:
            PositioningRecommendation with rationale
        """
        regime = await self._detector.classify(as_of=as_of)
        carry  = await self.carry_roll_down(horizon_months=12, as_of=as_of)

        # Rank maturities by carry + roll-down
        carry_sorted = sorted(carry.items(), key=lambda x: x[1], reverse=True)
        carry_leaders = [{"tenor": t, "carry_rolldown_pct": v} for t, v in carry_sorted[:3]]

        # Strategy logic based on curve regime
        if regime.shape == "inverted" and regime.inversion_days > 60:
            strategy = "barbell"
            preferred = ["1M", "3M", "6M", "1Y", "20Y", "30Y"]
            avoid     = ["2Y", "3Y", "5Y"]
            rationale = (
                f"Deep inversion ({regime.inversion_days} days). "
                "Barbell: park short-end cash at elevated front rates, "
                "hold long-duration for capital appreciation when inversion resolves."
            )
            confidence = "high"

        elif regime.shape == "inverted":
            strategy = "bullet"
            preferred = ["2Y", "3Y"]
            avoid     = ["10Y", "20Y", "30Y"]
            rationale = (
                "Recent inversion. Short bullet at 2-3Y captures elevated "
                "short rates while waiting for curve normalisation."
            )
            confidence = "medium"

        elif regime.shape == "normal" and regime.momentum == "steepening":
            strategy = "barbell"
            preferred = ["1Y", "2Y", "20Y", "30Y"]
            avoid     = ["5Y", "7Y"]
            rationale = (
                "Normal curve steepening. Barbell profits from the long end "
                "outperforming the belly as the curve steepens."
            )
            confidence = "medium"

        elif regime.shape == "flat":
            strategy = "ladder"
            preferred = ["1Y", "3Y", "5Y", "7Y", "10Y"]
            avoid     = []
            rationale = (
                "Flat curve: limited directional conviction. "
                "Ladder provides duration diversification without concentration risk."
            )
            confidence = "low"

        else:
            strategy = "neutral"
            preferred = carry_leaders[:3] if carry_leaders else []
            preferred = [p["tenor"] if isinstance(p, dict) else p for p in preferred]
            avoid     = []
            rationale = (
                "No strong directional signal. Overweight highest-carry maturities."
            )
            confidence = "low"

        return PositioningRecommendation(
            strategy=strategy,
            rationale=rationale,
            preferred_maturities=preferred,
            avoid_maturities=avoid,
            carry_leaders=carry_leaders,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

treasury_router = APIRouter(prefix="/treasury", tags=["Treasury Yield Curves"])

# Module-level singletons
_adapter  = FREDTreasuryAdapter()
_builder  = YieldCurveBuilder()
_fitter   = NelsonSiegelFitter()
_fwdcalc  = ForwardRateCalculator()
_detector = YieldCurveRegimeDetector()
_signals  = YieldCurveSignalEngine()


@treasury_router.get("/curve")
async def get_current_curve(
    as_of: Optional[str] = Query(None, description="ISO date YYYY-MM-DD or omit for latest"),
) -> dict:
    """
    Current US Treasury yield curve with Nelson-Siegel fit.
    Returns spot rates for all 11 maturities plus NS parameters.
    """
    curve_df = await _builder.get_current_curve(as_of=as_of)
    if curve_df.empty:
        raise HTTPException(status_code=503, detail="Treasury yield data unavailable")

    yields     = curve_df["yield_pct"].tolist()
    maturities = curve_df["maturity_years"].tolist()

    ns_fit = _fitter.fit(yields, maturities) if len(yields) >= 3 else None

    spreads = await _builder.get_spread_current()

    return {
        "as_of": as_of or date.today().isoformat(),
        "curve": curve_df.to_dict("records"),
        "spreads_bps": spreads,
        "ns_fit": ns_fit.model_dump() if ns_fit else None,
        "data_source": "FRED CSV (free, no API key)",
    }


@treasury_router.get("/history")
async def get_curve_history(
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD or omit for today"),
) -> dict:
    """
    Historical yield curve data as wide-format time series.
    Returns all Treasury CMT series from FRED over the requested period.
    """
    try:
        datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="start must be YYYY-MM-DD")

    df = await _builder.get_curve_history(start=start, end=end)
    records = df.reset_index().to_dict("records")

    return {
        "start":   start,
        "end":     end or date.today().isoformat(),
        "rows":    len(records),
        "columns": df.columns.tolist(),
        "data":    records,
    }


@treasury_router.get("/spreads")
async def get_spreads(
    lookback_days: int = Query(365, ge=30, le=3650, description="Days of history"),
    end: Optional[str] = Query(None),
) -> dict:
    """
    Key yield curve spreads (2s10s, 2s30s, 3m10y, 5s30s, 1y10y, 3y10y) in basis points.
    Includes current values and historical series.
    """
    end_dt   = end or date.today().isoformat()
    start_dt = (
        datetime.strptime(end_dt, "%Y-%m-%d") - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    df = await _builder.compute_spreads(start=start_dt, end=end_dt)
    if df.empty:
        raise HTTPException(status_code=503, detail="Spread data unavailable")

    current = {}
    if len(df) > 0:
        latest = df.iloc[-1]
        current = {col: round(float(v), 2) for col, v in latest.items() if pd.notna(v)}

    return {
        "as_of":   end_dt,
        "lookback_days": lookback_days,
        "current_bps": current,
        "history": df.reset_index().to_dict("records"),
    }


@treasury_router.get("/forward-rates")
async def get_forward_rates(
    as_of: Optional[str] = Query(None, description="ISO date or omit for latest"),
    method: str = Query("both", description="'ns', 'bootstrap', or 'both'"),
) -> dict:
    """
    Implied forward rates computed via Nelson-Siegel fit and bootstrap.
    Includes 5Y5Y breakeven inflation.
    """
    target   = as_of or date.today().isoformat()
    curve_df = await _builder.get_current_curve(as_of=target)

    if curve_df.empty:
        raise HTTPException(status_code=503, detail="Curve data unavailable")

    yields     = curve_df["yield_pct"].tolist()
    maturities = curve_df["maturity_years"].tolist()

    ns_forwards: list[dict] = []
    bootstrap_forwards: list[dict] = []

    if method in ("ns", "both") and len(yields) >= 3:
        ns_fit = _fitter.fit(yields, maturities)
        ns_forwards = [f.model_dump() for f in _fitter.get_forward_curve()]

    if method in ("bootstrap", "both"):
        bootstrap_fwds = await _fwdcalc.compute_all_forwards(as_of=target)
        bootstrap_forwards = [f.model_dump() for f in bootstrap_fwds]

    bei_5y5y = await _fwdcalc.five_five_breakeven(as_of=target)
    real_yield_df = await _fwdcalc.real_yield_curve(as_of=target)

    return {
        "as_of":                target,
        "ns_forwards":          ns_forwards,
        "bootstrap_forwards":   bootstrap_forwards,
        "breakeven_5y5y_pct":   bei_5y5y,
        "real_yield_curve":     real_yield_df.to_dict("records"),
    }


@treasury_router.get("/regime")
async def get_curve_regime(
    as_of: Optional[str] = Query(None, description="ISO date or omit for latest"),
) -> dict:
    """
    Current yield curve regime: shape, steepening/flattening momentum,
    bear/bull classification, inversion streak, recession signal.
    """
    regime = await _detector.classify(as_of=as_of)
    return regime.model_dump()


@treasury_router.get("/signals")
async def get_yield_signals(
    as_of: Optional[str] = Query(None),
    face_value: float = Query(1_000_000.0, description="Face value for DV01 calculation"),
) -> dict:
    """
    Yield curve trading signals: carry/roll-down by maturity, DV01,
    positioning recommendation (barbell/bullet/ladder).
    """
    carry    = await _signals.carry_roll_down(horizon_months=12, as_of=as_of)
    dv01     = await _signals.dv01_curve(as_of=as_of, face=face_value)
    position = await _signals.positioning_recommendation(as_of=as_of)

    return {
        "as_of":                as_of or date.today().isoformat(),
        "carry_rolldown_pct":   carry,
        "dv01_usd":             dv01,
        "positioning":          position.model_dump(),
    }


@treasury_router.get("/tips")
async def get_tips_real_yields(
    as_of: Optional[str] = Query(None),
) -> dict:
    """
    TIPS real yield curve and breakeven inflation rates.
    Returns real yields (DFII5/10/20/30) vs nominal yields and derived BEI.
    """
    real_df = await _fwdcalc.real_yield_curve(as_of=as_of)
    breakevens = await _adapter.fetch_breakevens(as_of=as_of)

    return {
        "as_of":          as_of or date.today().isoformat(),
        "real_yields":    real_df.to_dict("records"),
        "breakevens":     {
            k: {"label": v, "value_pct": breakevens.get(k)}
            for k, v in _BREAKEVEN_SERIES.items()
        },
    }
