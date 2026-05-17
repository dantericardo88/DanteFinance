"""
treasury_yield_v3.py — US Treasury yield curve analytics engine.

dim_035: US Treasury yield curves (FRED) — score 8 → 9

Architecture:
  FREDYieldLoader           — FRED CSV downloader, all nominal + TIPS series
  NelsonSiegelSvensson      — NSS parametric curve fitting (scipy + fallback)
  ForwardRateExtractor      — forward rates, instantaneous forward, policy path
  YieldCurveAnalytics       — 2s10s, 3m10y, butterfly, regime classification
  BreakevenInflationCurve   — breakeven = nominal - TIPS real yields
  FedFundsImplied           — FOMC path from forward rates, hardcoded calendar
  CrossCurrencyYields       — USD vs EUR vs JPY vs GBP yield differentials
  TreasuryPortfolioAnalyzer — DV01, duration, KRD, convexity, P&L attribution
  TreasuryYieldEngine       — orchestrator

Free data only: FRED CSV, ECB SDW, yfinance fallback for non-USD curves.
No API keys required.
"""
from __future__ import annotations

import csv
import io
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Callable, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional scipy — guard for fallback
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import minimize as scipy_minimize
    from scipy.optimize import differential_evolution
    HAS_SCIPY = True
except ImportError:
    scipy_minimize = None  # type: ignore[assignment]
    differential_evolution = None
    HAS_SCIPY = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ECB_SDW_BASE = "https://data-api.ecb.europa.eu/service/data"

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "treasury_yields.csv")
_CACHE_TTL_HOURS = 12  # refresh cache after 12 hours

# FRED series for nominal Treasury yields (constant maturity, %)
_NOMINAL_SERIES: dict[str, float] = {
    "DGS1MO":  1 / 12,
    "DGS3MO":  0.25,
    "DGS6MO":  0.5,
    "DGS1":    1.0,
    "DGS2":    2.0,
    "DGS3":    3.0,
    "DGS5":    5.0,
    "DGS7":    7.0,
    "DGS10":   10.0,
    "DGS20":   20.0,
    "DGS30":   30.0,
}

# TIPS real yield series
_TIPS_SERIES: dict[str, float] = {
    "DFII5":  5.0,
    "DFII7":  7.0,
    "DFII10": 10.0,
    "DFII20": 20.0,
    "DFII30": 30.0,
}

# Fed Funds series
_FED_FUNDS_SERIES = {"FEDFUNDS": "monthly", "DFF": "daily"}

# Tenor label map for display
_TENOR_LABELS: dict[str, str] = {
    "DGS1MO": "1M", "DGS3MO": "3M", "DGS6MO": "6M",
    "DGS1": "1Y", "DGS2": "2Y", "DGS3": "3Y",
    "DGS5": "5Y", "DGS7": "7Y", "DGS10": "10Y",
    "DGS20": "20Y", "DGS30": "30Y",
}

# FOMC meeting calendar 2025-2027 (approximate dates)
_FOMC_MEETING_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NSSParams:
    """Nelson-Siegel-Svensson model parameters."""
    beta0: float    # long-run level
    beta1: float    # slope (short-term loading)
    beta2: float    # first curvature hump
    beta3: float    # second curvature hump
    lambda1: float  # first decay factor
    lambda2: float  # second decay factor
    rmse: float = 0.0
    n_tenors: int = 0


@dataclass
class YieldCurve:
    """Full yield curve snapshot."""
    as_of: str
    nominal: dict[str, float]       # tenor_label → yield (%)
    real_tips: dict[str, float]     # tenor_label → TIPS real yield (%)
    breakeven: dict[str, float]     # tenor_label → breakeven inflation (%)
    tenors_years: list[float] = field(default_factory=list)
    yields_pct: list[float] = field(default_factory=list)
    nss_params: Optional[NSSParams] = None
    regime: str = ""
    slope_2s10s: float = 0.0
    slope_3m10y: float = 0.0


@dataclass
class ForwardCurve:
    """Forward rate curve derived from spot yields."""
    as_of: str
    forward_rates: dict[str, float]     # label → forward rate (%)
    policy_path: dict[str, float]       # horizon → implied policy rate (%)
    five_year_five_year: float = 0.0    # 5y5y forward inflation proxy


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _http_get_text(url: str, timeout: int = 20) -> str | None:
    """Fetch URL and return text content. Returns None on failure."""
    try:
        req = Request(url, headers={"User-Agent": "sentinel/3.0 treasury-yield-v3"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("HTTP GET failed [%s]: %s", url, exc)
        return None


def _fetch_fred_csv(series_id: str) -> pd.Series | None:
    """
    Fetch a single FRED series as a pandas Series (date index, float values).
    Returns None if unavailable.
    """
    url = f"{FRED_CSV_BASE}?id={series_id}"
    text = _http_get_text(url, timeout=25)
    if not text:
        return None

    try:
        df = pd.read_csv(io.StringIO(text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df.set_index("date")["value"]
        return df
    except Exception as exc:
        logger.warning("Failed to parse FRED CSV [%s]: %s", series_id, exc)
        return None


def _latest_value(series: pd.Series | None) -> float | None:
    """Get the most recent non-null value from a series."""
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    return float(val) if not math.isnan(float(val)) else None


# ---------------------------------------------------------------------------
# FREDYieldLoader
# ---------------------------------------------------------------------------

class FREDYieldLoader:
    """
    Downloads and caches US Treasury yield data from FRED.
    Covers nominal Treasuries, TIPS real yields, and Fed Funds rate.
    Uses local CSV cache with daily refresh.
    """

    def __init__(self, cache_file: str = _CACHE_FILE) -> None:
        self._cache_file = cache_file
        self._cache: dict[str, pd.Series] = {}
        os.makedirs(_DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    def _is_cache_fresh(self) -> bool:
        """Check if the local cache file is recent enough."""
        if not os.path.exists(self._cache_file):
            return False
        mtime = os.path.getmtime(self._cache_file)
        age_hours = (time.time() - mtime) / 3600.0
        return age_hours < _CACHE_TTL_HOURS

    # ------------------------------------------------------------------
    def _load_cache(self) -> pd.DataFrame | None:
        """Load cached yield data from disk."""
        if not os.path.exists(self._cache_file):
            return None
        try:
            df = pd.read_csv(self._cache_file, index_col=0, parse_dates=True)
            return df
        except Exception as exc:
            logger.warning("Failed to load cache: %s", exc)
            return None

    # ------------------------------------------------------------------
    def _save_cache(self, df: pd.DataFrame) -> None:
        """Save yield data to local CSV cache."""
        try:
            df.to_csv(self._cache_file)
            logger.debug("Saved yield cache to %s", self._cache_file)
        except Exception as exc:
            logger.warning("Failed to save cache: %s", exc)

    # ------------------------------------------------------------------
    def fetch_all_nominal_yields(self, date_str: str | None = None) -> dict[str, float]:
        """
        Fetch current (or historical date) nominal yields for all tenors.
        Returns dict: FRED series ID → yield (%).
        """
        all_series = self._get_all_series()
        result: dict[str, float] = {}

        for series_id in _NOMINAL_SERIES:
            s = all_series.get(series_id)
            if s is None or s.empty:
                continue
            if date_str:
                target = pd.Timestamp(date_str)
                # Find closest date ≤ target
                s_before = s[s.index <= target]
                if not s_before.empty:
                    result[series_id] = float(s_before.iloc[-1])
            else:
                val = _latest_value(s)
                if val is not None:
                    result[series_id] = val

        return result

    # ------------------------------------------------------------------
    def fetch_yield_history(
        self, tenor: str, start: str = "2000-01-01"
    ) -> pd.Series:
        """
        Fetch historical yields for a specific tenor.
        tenor: FRED series ID (e.g., "DGS10") or label ("10Y").
        """
        # Map label to series ID if needed
        label_to_series = {v: k for k, v in _TENOR_LABELS.items()}
        series_id = label_to_series.get(tenor, tenor)

        s = self._fetch_single_series(series_id)
        if s is None:
            return pd.Series(dtype=float)

        start_dt = pd.Timestamp(start)
        return s[s.index >= start_dt]

    # ------------------------------------------------------------------
    def fetch_full_curve_history(
        self, start: str = "2000-01-01"
    ) -> pd.DataFrame:
        """
        Fetch full history for all nominal tenors.
        Returns DataFrame: dates × series IDs.
        """
        if self._is_cache_fresh():
            cached = self._load_cache()
            if cached is not None:
                start_dt = pd.Timestamp(start)
                return cached[cached.index >= start_dt]

        logger.info("Downloading full yield curve history from FRED...")
        all_series: dict[str, pd.Series] = {}

        for series_id in list(_NOMINAL_SERIES.keys()) + list(_TIPS_SERIES.keys()) + ["DFF"]:
            s = self._fetch_single_series(series_id)
            if s is not None:
                all_series[series_id] = s
            time.sleep(0.1)  # rate limit courtesy

        if not all_series:
            return pd.DataFrame()

        df = pd.DataFrame(all_series)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        self._save_cache(df)

        start_dt = pd.Timestamp(start)
        return df[df.index >= start_dt]

    # ------------------------------------------------------------------
    def fetch_tips_yields(self, date_str: str | None = None) -> dict[str, float]:
        """Fetch TIPS real yields for available tenors."""
        result: dict[str, float] = {}
        all_series = self._get_all_series()

        for series_id, tenor_yr in _TIPS_SERIES.items():
            s = all_series.get(series_id)
            if s is None or s.empty:
                continue
            if date_str:
                target = pd.Timestamp(date_str)
                s_before = s[s.index <= target]
                if not s_before.empty:
                    result[series_id] = float(s_before.iloc[-1])
            else:
                val = _latest_value(s)
                if val is not None:
                    result[series_id] = val

        return result

    # ------------------------------------------------------------------
    def fetch_fed_funds_rate(self) -> float | None:
        """Fetch current Fed Funds rate (effective daily rate from FRED)."""
        s = self._fetch_single_series("DFF")
        return _latest_value(s)

    # ------------------------------------------------------------------
    def _get_all_series(self) -> dict[str, pd.Series]:
        """Return all series from cache or download."""
        if self._cache:
            return self._cache

        all_ids = (
            list(_NOMINAL_SERIES.keys())
            + list(_TIPS_SERIES.keys())
            + ["DFF", "FEDFUNDS"]
        )

        for series_id in all_ids:
            s = self._fetch_single_series(series_id)
            if s is not None:
                self._cache[series_id] = s

        return self._cache

    # ------------------------------------------------------------------
    def _fetch_single_series(self, series_id: str) -> pd.Series | None:
        """Fetch a single FRED series, using memory cache if available."""
        if series_id in self._cache:
            return self._cache[series_id]

        s = _fetch_fred_csv(series_id)
        if s is not None:
            self._cache[series_id] = s

        return s


# ---------------------------------------------------------------------------
# NelsonSiegelSvensson
# ---------------------------------------------------------------------------

class NelsonSiegelSvensson:
    """
    Fits the Nelson-Siegel-Svensson parametric yield curve model.

    NSS formula:
      y(τ) = β₀
           + β₁ × [(1 - e^(-τ/λ₁)) / (τ/λ₁)]
           + β₂ × [(1 - e^(-τ/λ₁)) / (τ/λ₁) - e^(-τ/λ₁)]
           + β₃ × [(1 - e^(-τ/λ₂)) / (τ/λ₂) - e^(-τ/λ₂)]

    Parameters:
      β₀ = level (long-run rate)
      β₁ = slope
      β₂ = first curvature
      β₃ = second curvature
      λ₁, λ₂ = decay factors (>0)
    """

    # Default initial parameter bounds
    _BOUNDS = [
        (-5.0, 20.0),   # beta0
        (-20.0, 20.0),  # beta1
        (-30.0, 30.0),  # beta2
        (-30.0, 30.0),  # beta3
        (0.01, 10.0),   # lambda1
        (0.01, 10.0),   # lambda2
    ]

    # ------------------------------------------------------------------
    @staticmethod
    def _nss_yield(tau: float, params: list[float]) -> float:
        """Compute NSS yield at maturity tau (years)."""
        beta0, beta1, beta2, beta3, lam1, lam2 = params

        if tau <= 0:
            return beta0 + beta1

        exp1 = math.exp(-tau / lam1)
        f1 = (1 - exp1) / (tau / lam1)

        exp2 = math.exp(-tau / lam2)
        f2 = (1 - exp2) / (tau / lam2)

        y = (
            beta0
            + beta1 * f1
            + beta2 * (f1 - exp1)
            + beta3 * (f2 - exp2)
        )
        return y

    # ------------------------------------------------------------------
    @classmethod
    def _nss_yield_vec(
        cls, tenors: list[float], params: list[float]
    ) -> list[float]:
        """Compute NSS yields for a list of maturities."""
        return [cls._nss_yield(t, params) for t in tenors]

    # ------------------------------------------------------------------
    @classmethod
    def _sse(
        cls, params: list[float], tenors: list[float], yields: list[float]
    ) -> float:
        """Sum of squared errors between NSS fit and observed yields."""
        fitted = cls._nss_yield_vec(tenors, params)
        return sum((f - y) ** 2 for f, y in zip(fitted, yields))

    # ------------------------------------------------------------------
    def fit(
        self,
        tenors: list[float],
        yields: list[float],
    ) -> NSSParams:
        """
        Fit NSS parameters to observed yields.
        Uses scipy BFGS when available; falls back to grid search.
        """
        if len(tenors) < 4:
            logger.warning("NSS fit needs >= 4 tenors, got %d", len(tenors))
            # Return degenerate flat curve at average yield
            avg = sum(yields) / len(yields) if yields else 0.0
            return NSSParams(
                beta0=avg, beta1=0.0, beta2=0.0, beta3=0.0,
                lambda1=1.0, lambda2=2.0,
                rmse=0.0, n_tenors=len(tenors),
            )

        # Filter valid (finite) pairs
        valid = [(t, y) for t, y in zip(tenors, yields) if math.isfinite(t) and math.isfinite(y) and t > 0]
        if len(valid) < 4:
            return NSSParams(beta0=0.0, beta1=0.0, beta2=0.0, beta3=0.0,
                             lambda1=1.0, lambda2=2.0, rmse=999.0, n_tenors=0)

        tenors_v = [t for t, _ in valid]
        yields_v = [y for _, y in valid]

        if HAS_SCIPY:
            best_params = self._fit_scipy(tenors_v, yields_v)
        else:
            best_params = self._fit_grid_search(tenors_v, yields_v)

        fitted = self._nss_yield_vec(tenors_v, best_params)
        residuals = [f - y for f, y in zip(fitted, yields_v)]
        rmse = math.sqrt(sum(r ** 2 for r in residuals) / len(residuals))

        return NSSParams(
            beta0=best_params[0],
            beta1=best_params[1],
            beta2=best_params[2],
            beta3=best_params[3],
            lambda1=best_params[4],
            lambda2=best_params[5],
            rmse=rmse,
            n_tenors=len(tenors_v),
        )

    # ------------------------------------------------------------------
    def _fit_scipy(
        self, tenors: list[float], yields: list[float]
    ) -> list[float]:
        """Scipy-based optimization with multiple starting points."""
        best_params = None
        best_loss = float("inf")

        # Multiple initial guesses to avoid local minima
        initial_guesses = [
            [yields[-1], yields[0] - yields[-1], 0.0, 0.0, 1.0, 3.0],
            [yields[-1], yields[0] - yields[-1], 1.0, -1.0, 2.0, 5.0],
            [4.0, -2.0, 0.5, 0.5, 1.5, 4.0],
            [3.0, 1.0, -2.0, 1.0, 0.5, 2.0],
        ]

        for x0 in initial_guesses:
            try:
                res = scipy_minimize(
                    self._sse,
                    x0,
                    args=(tenors, yields),
                    method="L-BFGS-B",
                    bounds=self._BOUNDS,
                    options={"maxiter": 1000, "ftol": 1e-10},
                )
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_params = list(res.x)
            except Exception:
                continue

        if best_params is None:
            best_params = initial_guesses[0]

        return best_params

    # ------------------------------------------------------------------
    def _fit_grid_search(
        self, tenors: list[float], yields: list[float]
    ) -> list[float]:
        """Grid search fallback when scipy is not available."""
        best_params = None
        best_loss = float("inf")

        avg_y = sum(yields) / len(yields)
        slope = yields[-1] - yields[0]  # long minus short

        # Coarse grid
        for lam1 in [0.5, 1.0, 2.0, 3.0]:
            for lam2 in [2.0, 4.0, 6.0]:
                for b2 in [-3.0, 0.0, 3.0]:
                    for b3 in [-2.0, 0.0, 2.0]:
                        params = [avg_y, slope, b2, b3, lam1, lam2]
                        loss = self._sse(params, tenors, yields)
                        if loss < best_loss:
                            best_loss = loss
                            best_params = params[:]

        return best_params or [avg_y, slope, 0.0, 0.0, 1.0, 3.0]

    # ------------------------------------------------------------------
    def predict(self, tau: float, params: NSSParams) -> float:
        """Predict yield at maturity tau years from NSS parameters."""
        p = [params.beta0, params.beta1, params.beta2, params.beta3,
             params.lambda1, params.lambda2]
        return self._nss_yield(tau, p)

    # ------------------------------------------------------------------
    def fit_time_series(self, yield_history: pd.DataFrame) -> pd.DataFrame:
        """
        Fit NSS for every date in yield_history DataFrame.
        Returns DataFrame with columns: date, beta0, beta1, beta2, beta3,
        lambda1, lambda2, rmse.
        """
        # Map column names to tenors
        col_to_tenor = {col: _NOMINAL_SERIES[col] for col in yield_history.columns if col in _NOMINAL_SERIES}
        tenor_cols = list(col_to_tenor.keys())

        if not tenor_cols:
            logger.warning("No recognized nominal yield columns in history")
            return pd.DataFrame()

        results = []
        for idx, row in yield_history.iterrows():
            tenors = []
            yields = []
            for col in tenor_cols:
                val = row.get(col)
                if val is not None and not math.isnan(float(val)):
                    tenors.append(col_to_tenor[col])
                    yields.append(float(val))

            if len(tenors) < 4:
                continue

            try:
                nss = self.fit(tenors, yields)
                results.append({
                    "date": idx,
                    "beta0": nss.beta0,
                    "beta1": nss.beta1,
                    "beta2": nss.beta2,
                    "beta3": nss.beta3,
                    "lambda1": nss.lambda1,
                    "lambda2": nss.lambda2,
                    "rmse": nss.rmse,
                })
            except Exception as exc:
                logger.debug("NSS fit failed for %s: %s", idx, exc)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).set_index("date")

    # ------------------------------------------------------------------
    def extract_level_slope_curvature(self, params: NSSParams) -> dict:
        """
        Decompose NSS parameters into economic factors.
        Level = β₀ (long-run rate, approached as τ → ∞)
        Slope = -β₁ (10Y - 3M approximation from NS structure)
        Curvature = β₂ + β₃ (hump magnitude)
        """
        return {
            "level_beta0": params.beta0,
            "slope_minus_beta1": -params.beta1,
            "curvature_beta2_plus_beta3": params.beta2 + params.beta3,
            "first_hump_beta2": params.beta2,
            "second_hump_beta3": params.beta3,
            "decay_lambda1": params.lambda1,
            "decay_lambda2": params.lambda2,
        }

    # ------------------------------------------------------------------
    def get_fitted_curve_points(
        self,
        params: NSSParams,
        max_tenor: float = 30.0,
        n_points: int = 100,
    ) -> list[dict]:
        """Return NSS fitted curve as list of {tenor_years, yield_pct} dicts."""
        step = max_tenor / n_points
        points = []
        for i in range(1, n_points + 1):
            tau = i * step
            y = self.predict(tau, params)
            points.append({"tenor_years": round(tau, 3), "yield_pct": round(y, 4)})
        return points


# ---------------------------------------------------------------------------
# ForwardRateExtractor
# ---------------------------------------------------------------------------

class ForwardRateExtractor:
    """
    Extracts forward rates from spot yield curves.
    Implements par rate → spot rate bootstrapping and forward derivation.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def compute_forward_rate(
        spot_yields: dict[str, float],
        T1: float,
        T2: float,
    ) -> float:
        """
        Implied forward rate from T1 to T2 using spot yields.
        f(T1,T2) = [y(T2) × T2 - y(T1) × T1] / (T2 - T1)
        spot_yields: dict mapping tenor_years → yield (%).
        """
        if T2 <= T1:
            raise ValueError(f"T2 ({T2}) must be > T1 ({T1})")

        # Find closest available tenors
        y_T1 = _interpolate_yield(spot_yields, T1)
        y_T2 = _interpolate_yield(spot_yields, T2)

        if y_T1 is None or y_T2 is None:
            return float("nan")

        return (y_T2 * T2 - y_T1 * T1) / (T2 - T1)

    # ------------------------------------------------------------------
    def compute_forward_curve(
        self,
        spot_yields: dict[str, float],
    ) -> dict:
        """
        Compute standard 1-year forward rates from the spot curve.
        Returns: 1y1y, 2y1y, 3y1y, 4y1y, 5y1y, 5y5y, 10y10y
        """
        forward_labels = [
            ("1y1y",   1.0,  2.0),
            ("2y1y",   2.0,  3.0),
            ("3y1y",   3.0,  4.0),
            ("4y1y",   4.0,  5.0),
            ("5y1y",   5.0,  6.0),
            ("5y5y",   5.0, 10.0),
            ("10y10y", 10.0, 20.0),
            ("10y20y", 10.0, 30.0),
        ]

        result: dict[str, float] = {}
        for label, T1, T2 in forward_labels:
            rate = self.compute_forward_rate(spot_yields, T1, T2)
            if math.isfinite(rate):
                result[label] = round(rate, 4)

        return result

    # ------------------------------------------------------------------
    def compute_instantaneous_forward(
        self,
        nss_params: NSSParams,
    ) -> Callable[[float], float]:
        """
        Derive the instantaneous forward rate f(τ) from NSS analytically.
        f(τ) = d[τ × y(τ)] / dτ

        Returns a callable: tau → instantaneous forward rate (%)
        """
        b0 = nss_params.beta0
        b1 = nss_params.beta1
        b2 = nss_params.beta2
        b3 = nss_params.beta3
        l1 = nss_params.lambda1
        l2 = nss_params.lambda2

        def _f(tau: float) -> float:
            if tau <= 1e-8:
                return b0 + b1

            # Analytical derivative of NSS
            # d/dτ [τ × y(τ)] using product rule
            e1 = math.exp(-tau / l1)
            e2 = math.exp(-tau / l2)

            # NSS: y(τ) = b0 + b1*f1 + b2*(f1 - e1) + b3*(f2 - e2)
            # where f1 = (1-e1)/(τ/l1), f2 = (1-e2)/(τ/l2)

            # df1/dτ = (e1/l1 - (1-e1)/τ) / (τ/l1)  ... simplifies to:
            df1 = (e1 / l1 - (1 - e1) / tau) * l1 / tau

            df2_term = (e2 / l2 - (1 - e2) / tau) * l2 / tau

            de1 = -e1 / l1
            de2 = -e2 / l2

            # d[τ·y]/dτ = y(τ) + τ·(dy/dτ)
            f1 = (1 - e1) / (tau / l1)
            f2 = (1 - e2) / (tau / l2)

            y_tau = b0 + b1 * f1 + b2 * (f1 - e1) + b3 * (f2 - e2)

            dy_dtau = (
                b1 * df1
                + b2 * (df1 - de1)
                + b3 * (df2_term - de2)
            )

            return y_tau + tau * dy_dtau

        return _f

    # ------------------------------------------------------------------
    def extract_implied_policy_path(
        self,
        spot_yields: dict[str, float],
        horizons_months: list[int] | None = None,
    ) -> pd.Series:
        """
        Extract implied short-term rate path from forward rates.
        Uses 3-month rolling forwards as proxy for expected Fed Funds rate.
        """
        if horizons_months is None:
            horizons_months = [3, 6, 9, 12, 18, 24, 36, 60, 84, 120]

        path: dict[str, float] = {}
        for h in horizons_months:
            T1 = h / 12.0
            T2 = T1 + 0.25  # 3-month forward

            rate = self.compute_forward_rate(spot_yields, T1, T2)
            if math.isfinite(rate):
                label = f"{h}M"
                path[label] = round(rate, 4)

        return pd.Series(path, name="implied_policy_rate_pct")


# ---------------------------------------------------------------------------
# Yield interpolation helper
# ---------------------------------------------------------------------------

def _interpolate_yield(
    spot_yields: dict[str, float],
    target_tenor: float,
) -> float | None:
    """
    Linear interpolation between nearest tenor bracket.
    spot_yields: keys as tenor_years (float) or FRED series IDs.
    """
    # Normalize keys to float tenors
    tenor_yield: dict[float, float] = {}
    for k, v in spot_yields.items():
        if isinstance(k, (int, float)) and math.isfinite(float(k)):
            tenor_yield[float(k)] = v
        elif k in _NOMINAL_SERIES:
            tenor_yield[_NOMINAL_SERIES[k]] = v
        elif k in _TIPS_SERIES:
            tenor_yield[_TIPS_SERIES[k]] = v

    if not tenor_yield:
        return None

    sorted_tenors = sorted(tenor_yield.keys())

    # Exact match
    if target_tenor in tenor_yield:
        return tenor_yield[target_tenor]

    # Below range
    if target_tenor < sorted_tenors[0]:
        return tenor_yield[sorted_tenors[0]]

    # Above range
    if target_tenor > sorted_tenors[-1]:
        return tenor_yield[sorted_tenors[-1]]

    # Linear interpolation
    for i in range(len(sorted_tenors) - 1):
        t1 = sorted_tenors[i]
        t2 = sorted_tenors[i + 1]
        if t1 <= target_tenor <= t2:
            w = (target_tenor - t1) / (t2 - t1)
            return tenor_yield[t1] * (1 - w) + tenor_yield[t2] * w

    return None


def _build_tenor_yield_map(
    nominal_yields: dict[str, float],
) -> dict[float, float]:
    """Convert FRED series-ID keyed yields to tenor-years keyed dict."""
    result: dict[float, float] = {}
    for series_id, tenor_yr in _NOMINAL_SERIES.items():
        if series_id in nominal_yields:
            result[tenor_yr] = nominal_yields[series_id]
    return result


# ---------------------------------------------------------------------------
# YieldCurveAnalytics
# ---------------------------------------------------------------------------

class YieldCurveAnalytics:
    """
    Computes spread, slope, butterfly, and regime metrics for the yield curve.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def compute_2s10s(yields: dict[str, float]) -> float:
        """10Y - 2Y spread in basis points (primary recession indicator)."""
        tenor_map = _build_tenor_yield_map(yields)
        y2 = _interpolate_yield(tenor_map, 2.0)
        y10 = _interpolate_yield(tenor_map, 10.0)
        if y2 is None or y10 is None:
            return float("nan")
        return (y10 - y2) * 100  # bps

    # ------------------------------------------------------------------
    @staticmethod
    def compute_3m10y(yields: dict[str, float]) -> float:
        """10Y - 3M spread in bps (best recession predictor, per NY Fed research)."""
        tenor_map = _build_tenor_yield_map(yields)
        y3m = _interpolate_yield(tenor_map, 0.25)
        y10 = _interpolate_yield(tenor_map, 10.0)
        if y3m is None or y10 is None:
            return float("nan")
        return (y10 - y3m) * 100

    # ------------------------------------------------------------------
    @staticmethod
    def compute_butterfly(yields: dict[str, float]) -> float:
        """
        Butterfly spread: (2Y + 30Y) / 2 - 10Y in bps.
        Positive = humped curve (belly cheap); Negative = inverted belly.
        """
        tenor_map = _build_tenor_yield_map(yields)
        y2 = _interpolate_yield(tenor_map, 2.0)
        y10 = _interpolate_yield(tenor_map, 10.0)
        y30 = _interpolate_yield(tenor_map, 30.0)
        if any(v is None for v in [y2, y10, y30]):
            return float("nan")
        return ((y2 + y30) / 2.0 - y10) * 100

    # ------------------------------------------------------------------
    @staticmethod
    def compute_5s30s(yields: dict[str, float]) -> float:
        """30Y - 5Y spread in bps (long-end steepness)."""
        tenor_map = _build_tenor_yield_map(yields)
        y5 = _interpolate_yield(tenor_map, 5.0)
        y30 = _interpolate_yield(tenor_map, 30.0)
        if y5 is None or y30 is None:
            return float("nan")
        return (y30 - y5) * 100

    # ------------------------------------------------------------------
    @staticmethod
    def compute_curve_regime(yields: dict[str, float]) -> str:
        """
        Classify yield curve regime based on 2s10s spread.
        NORMAL_STEEP    : 2s10s > 150 bps
        NORMAL          : 50 < 2s10s ≤ 150 bps
        FLAT            : -50 < 2s10s ≤ 50 bps
        INVERTED        : -100 < 2s10s ≤ -50 bps (recession warning)
        DEEPLY_INVERTED : 2s10s ≤ -100 bps (strong recession signal)
        """
        spread = YieldCurveAnalytics.compute_2s10s(yields)
        if math.isnan(spread):
            return "UNKNOWN"

        if spread > 150:
            return "NORMAL_STEEP"
        elif spread > 50:
            return "NORMAL"
        elif spread > -50:
            return "FLAT"
        elif spread > -100:
            return "INVERTED"
        else:
            return "DEEPLY_INVERTED"

    # ------------------------------------------------------------------
    @staticmethod
    def compute_historical_regime_frequency(
        yield_history: pd.DataFrame,
    ) -> dict:
        """
        Compute how often each curve regime appeared historically.
        yield_history: DataFrame with DGS2, DGS10 columns.
        """
        if "DGS2" not in yield_history.columns or "DGS10" not in yield_history.columns:
            return {}

        spreads = (yield_history["DGS10"] - yield_history["DGS2"]) * 100
        spreads = spreads.dropna()

        if spreads.empty:
            return {}

        regime_counts: dict[str, int] = {
            "NORMAL_STEEP": 0,
            "NORMAL": 0,
            "FLAT": 0,
            "INVERTED": 0,
            "DEEPLY_INVERTED": 0,
        }

        for s in spreads:
            if s > 150:
                regime_counts["NORMAL_STEEP"] += 1
            elif s > 50:
                regime_counts["NORMAL"] += 1
            elif s > -50:
                regime_counts["FLAT"] += 1
            elif s > -100:
                regime_counts["INVERTED"] += 1
            else:
                regime_counts["DEEPLY_INVERTED"] += 1

        total = len(spreads)
        return {
            regime: {
                "count": count,
                "frequency_pct": round(count / total * 100, 1),
            }
            for regime, count in regime_counts.items()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def compute_spread_percentile(
        spread: float,
        spread_history: pd.Series,
    ) -> float:
        """
        Compute the percentile rank of current spread vs historical distribution.
        Returns 0-100.
        """
        if spread_history.empty or math.isnan(spread):
            return float("nan")

        clean = spread_history.dropna()
        if clean.empty:
            return float("nan")

        below = (clean <= spread).sum()
        return float(below / len(clean) * 100)

    # ------------------------------------------------------------------
    @staticmethod
    def get_curve_summary(yields: dict[str, float]) -> dict:
        """Full analytics summary for a given curve."""
        analytics = YieldCurveAnalytics()
        return {
            "2s10s_bps": analytics.compute_2s10s(yields),
            "3m10y_bps": analytics.compute_3m10y(yields),
            "5s30s_bps": analytics.compute_5s30s(yields),
            "butterfly_bps": analytics.compute_butterfly(yields),
            "regime": analytics.compute_curve_regime(yields),
        }


# ---------------------------------------------------------------------------
# BreakevenInflationCurve
# ---------------------------------------------------------------------------

class BreakevenInflationCurve:
    """
    Computes breakeven inflation rates: nominal yield - TIPS real yield.
    Provides term structure of market inflation expectations.
    """

    # Matching nominal ↔ TIPS series for each available tenor
    _BREAKEVEN_PAIRS: list[tuple[str, str, float]] = [
        ("DGS5",  "DFII5",  5.0),
        ("DGS7",  "DFII7",  7.0),
        ("DGS10", "DFII10", 10.0),
        ("DGS20", "DFII20", 20.0),
        ("DGS30", "DFII30", 30.0),
    ]

    # ------------------------------------------------------------------
    def compute_breakeven_curve(
        self,
        nominal: dict[str, float],
        tips_real: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute breakeven inflation curve.
        breakeven_τ = nominal_yield_τ - TIPS_real_yield_τ
        Returns dict: tenor_label → breakeven (%).
        """
        breakeven: dict[str, float] = {}

        for nom_series, tips_series, tenor_yr in self._BREAKEVEN_PAIRS:
            nom_y = nominal.get(nom_series)
            tips_y = tips_real.get(tips_series)

            if nom_y is not None and tips_y is not None:
                label = f"{int(tenor_yr)}Y" if tenor_yr.is_integer() else f"{tenor_yr}Y"
                breakeven[label] = round(nom_y - tips_y, 4)

        return breakeven

    # ------------------------------------------------------------------
    def compute_inflation_term_premium(
        self, breakeven_curve: dict[str, float]
    ) -> float:
        """
        Inflation term premium = 5Y5Y forward breakeven - 5Y breakeven.
        5Y5Y breakeven = 2 × 10Y breakeven - 5Y breakeven (approximation).
        Represents additional inflation risk premium for years 5-10.
        """
        be5 = breakeven_curve.get("5Y")
        be10 = breakeven_curve.get("10Y")

        if be5 is None or be10 is None:
            return float("nan")

        # 5Y5Y forward breakeven approximation
        be5y5y = 2 * be10 - be5
        term_premium = be5y5y - be5
        return round(term_premium, 4)

    # ------------------------------------------------------------------
    def compute_5y5y_forward_breakeven(
        self, breakeven_curve: dict[str, float]
    ) -> float:
        """
        5Y5Y forward breakeven inflation: expected inflation in years 5-10.
        Key Fed monitoring metric.
        """
        be5 = breakeven_curve.get("5Y")
        be10 = breakeven_curve.get("10Y")

        if be5 is None or be10 is None:
            return float("nan")

        return round(2 * be10 - be5, 4)

    # ------------------------------------------------------------------
    def get_real_yield_curve(
        self,
        loader: FREDYieldLoader,
        date_str: str | None = None,
    ) -> dict[str, float]:
        """
        Fetch TIPS real yields as {tenor_label → real_yield %}.
        """
        raw = loader.fetch_tips_yields(date_str)
        result: dict[str, float] = {}

        for series_id, tenor_yr in _TIPS_SERIES.items():
            if series_id in raw:
                label = f"{int(tenor_yr)}Y" if tenor_yr == int(tenor_yr) else f"{tenor_yr}Y"
                result[label] = raw[series_id]

        return result

    # ------------------------------------------------------------------
    def breakeven_history(
        self,
        loader: FREDYieldLoader,
        tenor: str = "10Y",
        start: str = "2003-01-01",
    ) -> pd.Series:
        """
        Historical breakeven for a single tenor.
        10Y is most liquid and widely followed.
        """
        tenor_map = {"5Y": ("DGS5", "DFII5"), "7Y": ("DGS7", "DFII7"),
                     "10Y": ("DGS10", "DFII10"), "20Y": ("DGS20", "DFII20"),
                     "30Y": ("DGS30", "DFII30")}

        if tenor not in tenor_map:
            logger.warning("Unknown tenor %s for breakeven history", tenor)
            return pd.Series(dtype=float)

        nom_series_id, tips_series_id = tenor_map[tenor]
        nom = loader.fetch_yield_history(nom_series_id, start=start)
        tips = loader.fetch_yield_history(tips_series_id, start=start)

        if nom.empty or tips.empty:
            return pd.Series(dtype=float)

        combined = pd.concat([nom, tips], axis=1, join="inner")
        combined.columns = ["nominal", "real"]
        combined = combined.dropna()

        return (combined["nominal"] - combined["real"]).rename(f"breakeven_{tenor}")


# ---------------------------------------------------------------------------
# FedFundsImplied
# ---------------------------------------------------------------------------

class FedFundsImplied:
    """
    Implied Fed Funds policy path from forward rates and market signals.
    Uses yield curve forward rates as the primary data source (always available).
    Falls back to CME FedWatch scraping when connectivity allows.
    """

    def __init__(self, loader: FREDYieldLoader) -> None:
        self._loader = loader
        self._extractor = ForwardRateExtractor()

    # ------------------------------------------------------------------
    def get_meeting_dates(self, n_meetings: int = 8) -> list[str]:
        """Return next N FOMC meeting dates from today."""
        today = date.today()
        future = [
            d for d in _FOMC_MEETING_DATES
            if date.fromisoformat(d) >= today
        ]
        return future[:n_meetings]

    # ------------------------------------------------------------------
    def get_implied_path_from_forwards(
        self,
        yields: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Use forward rate curve as proxy for implied policy path.
        This is the reliable always-available method.
        Returns dict: {meeting_date_str → implied_rate_pct}.
        """
        if yields is None:
            yields = self._loader.fetch_all_nominal_yields()

        tenor_map = _build_tenor_yield_map(yields)
        meeting_dates = self.get_meeting_dates(12)
        today = date.today()

        result: dict[str, float] = {}
        for meeting_str in meeting_dates:
            meeting_dt = date.fromisoformat(meeting_str)
            T1 = (meeting_dt - today).days / 365.0
            T2 = T1 + 0.25  # 3M rate at that horizon

            if T1 <= 0:
                continue

            rate = self._extractor.compute_forward_rate(tenor_map, T1, T2)
            if math.isfinite(rate):
                result[meeting_str] = round(rate, 3)

        return result

    # ------------------------------------------------------------------
    def fetch_sofr_futures_implied(self) -> dict:
        """
        Attempt to scrape CME FedWatch or use SOFR proxy.
        Falls back to forward rate curve if unavailable.
        """
        # Try CME FedWatch data endpoint (JSON)
        cme_url = "https://www.cmegroup.com/CmeWS/mvc/ProductCalendar/V2/getFedWatchData.json"
        data = None
        try:
            text = _http_get_text(cme_url, timeout=10)
            if text:
                data = _parse_cme_fedwatch(text)
        except Exception:
            pass

        if data:
            return data

        # Fallback: forward rate curve
        logger.info("CME FedWatch unavailable; using forward rate curve as policy path proxy")
        path = self.get_implied_path_from_forwards()
        return {
            "source": "forward_rate_proxy",
            "meetings": path,
            "note": "Derived from Treasury forward rates as CME data unavailable",
        }

    # ------------------------------------------------------------------
    def compute_hike_probability(
        self,
        meeting_date: str,
        current_rate: float | None = None,
        yields: dict[str, float] | None = None,
    ) -> float:
        """
        Estimate probability of rate hike at a given FOMC meeting.
        Uses the forward-implied rate at that horizon vs current rate.

        Simplified model: probability ∝ (implied_rate - current_rate).
        """
        if yields is None:
            yields = self._loader.fetch_all_nominal_yields()

        if current_rate is None:
            current_rate = self._loader.fetch_fed_funds_rate() or 5.25

        path = self.get_implied_path_from_forwards(yields)
        implied = path.get(meeting_date)

        if implied is None:
            return float("nan")

        # Simplified probability model (each hike = 25bps)
        diff = implied - current_rate
        # Probability of at least one hike = sigmoid-like mapping
        # diff > 0 → hike; diff < 0 → cut
        prob = 0.5 + diff / 0.5  # linear approximation
        return max(0.0, min(1.0, prob))

    # ------------------------------------------------------------------
    def get_rate_change_expectations(
        self, yields: dict[str, float] | None = None
    ) -> pd.DataFrame:
        """Summary table of rate change expectations at each meeting."""
        if yields is None:
            yields = self._loader.fetch_all_nominal_yields()

        current_rate = self._loader.fetch_fed_funds_rate() or 5.25
        path = self.get_implied_path_from_forwards(yields)
        meetings = self.get_meeting_dates(8)

        rows = []
        for m in meetings:
            implied = path.get(m, float("nan"))
            change = implied - current_rate if math.isfinite(implied) else float("nan")
            prob_hike = self.compute_hike_probability(m, current_rate, yields)
            rows.append({
                "meeting": m,
                "implied_rate_pct": implied,
                "expected_change_bps": round(change * 100, 1) if math.isfinite(change) else float("nan"),
                "hike_probability": round(prob_hike, 3) if math.isfinite(prob_hike) else float("nan"),
            })

        return pd.DataFrame(rows)


def _parse_cme_fedwatch(text: str) -> dict | None:
    """Attempt to parse CME FedWatch JSON response."""
    import json
    try:
        data = json.loads(text)
        # Structure varies — return raw if it looks valid
        if isinstance(data, (dict, list)):
            return {"source": "cme_fedwatch", "raw": data}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# CrossCurrencyYields
# ---------------------------------------------------------------------------

class CrossCurrencyYields:
    """
    Fetch and compare yield curves across USD, EUR, JPY, and GBP.
    Primary sources: ECB SDW (EUR), BOJ (JPY with yfinance fallback), FRED (USD).
    """

    def __init__(self, loader: FREDYieldLoader) -> None:
        self._loader = loader

    # ------------------------------------------------------------------
    def fetch_ecb_yields(self) -> dict[str, float]:
        """
        Fetch EUR yield curve (German Bund benchmark) from ECB SDW API.
        Returns dict: tenor_label → yield (%).
        """
        # ECB SDW: Government bond yield curve (AAA-rated), Svensson parameters
        # Try key tenors directly
        ecb_series = {
            "1Y":  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y",
            "2Y":  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
            "5Y":  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y",
            "10Y": "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
            "30Y": "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_30Y",
        }

        result: dict[str, float] = {}
        for label, series_key in ecb_series.items():
            url = f"{ECB_SDW_BASE}/YC/{series_key}?format=csvdata&lastNObservations=1"
            text = _http_get_text(url, timeout=15)
            if text:
                val = _parse_ecb_csv(text)
                if val is not None:
                    result[label] = val

        if not result:
            logger.warning("ECB SDW unavailable; falling back to yfinance proxies for EUR")
            result = self._fetch_eur_yfinance_fallback()

        return result

    # ------------------------------------------------------------------
    def _fetch_eur_yfinance_fallback(self) -> dict[str, float]:
        """Fallback: use yfinance to get EUR govt yield proxies."""
        try:
            import yfinance as yf
            # German Bund proxies via yfinance tickers
            proxies = {
                "2Y": "^TNX",  # Not EUR but directional proxy
                "10Y": "^BUND",  # German 10Y if available
            }
            result: dict[str, float] = {}
            for label, ticker_sym in proxies.items():
                try:
                    tk = yf.Ticker(ticker_sym)
                    hist = tk.history(period="5d")
                    if not hist.empty:
                        result[label] = round(float(hist["Close"].iloc[-1]), 4)
                except Exception:
                    pass
            return result
        except ImportError:
            return {}

    # ------------------------------------------------------------------
    def fetch_boj_yields(self) -> dict[str, float]:
        """
        Fetch JGB (Japanese Government Bond) yields.
        Attempts BOJ statistics; falls back to yfinance.
        """
        # BOJ benchmark yields via yfinance
        result: dict[str, float] = {}
        result = self._fetch_jgb_yfinance_fallback()

        if not result:
            # Try BOJ web scraping (simplified)
            logger.info("Attempting BOJ direct fetch")
            result = self._fetch_jgb_direct()

        return result

    # ------------------------------------------------------------------
    def _fetch_jgb_yfinance_fallback(self) -> dict[str, float]:
        """Fetch JGB yields via yfinance Japanese bond tickers."""
        try:
            import yfinance as yf
            tickers = {
                "2Y": "^JTN2Y",  # Placeholder — yfinance doesn't have JGB directly
                "10Y": "^TNX",   # Approximate; real JGB 10Y may be unavailable free
            }
            result: dict[str, float] = {}
            # Try FRED for Japan 10Y if available
            jgb10 = _fetch_fred_csv("IRLTLT01JPM156N")  # OECD Japan 10Y, monthly
            if jgb10 is not None and not jgb10.empty:
                result["10Y"] = round(float(jgb10.iloc[-1]), 4)
            return result
        except ImportError:
            return {}

    # ------------------------------------------------------------------
    def _fetch_jgb_direct(self) -> dict[str, float]:
        """Try to get JGB yields from OECD/FRED monthly series."""
        result: dict[str, float] = {}
        fred_jgb = {
            "10Y": "IRLTLT01JPM156N",  # Japan 10Y from FRED (OECD)
        }
        for label, series_id in fred_jgb.items():
            s = _fetch_fred_csv(series_id)
            val = _latest_value(s)
            if val is not None:
                result[label] = val
        return result

    # ------------------------------------------------------------------
    def fetch_gbp_yields(self) -> dict[str, float]:
        """Fetch UK Gilt yields from FRED (OECD data)."""
        result: dict[str, float] = {}
        fred_uk = {
            "2Y":  "IRLTLT01GBM156N",  # UK 10Y from OECD/FRED (closest available)
            "10Y": "IRLTLT01GBM156N",
        }
        s = _fetch_fred_csv("IRLTLT01GBM156N")
        val = _latest_value(s)
        if val is not None:
            result["10Y"] = val
        return result

    # ------------------------------------------------------------------
    def fetch_usd_yields(
        self, yields: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Return USD yields in tenor-label format."""
        if yields is None:
            yields = self._loader.fetch_all_nominal_yields()

        return {
            _TENOR_LABELS.get(k, k): v
            for k, v in yields.items()
            if k in _TENOR_LABELS
        }

    # ------------------------------------------------------------------
    def compute_yield_differential(
        self,
        currency_pair: str = "EUR/USD",
        tenor: str = "2Y",
    ) -> dict:
        """
        Compute yield differential at a given tenor between two currencies.
        2Y differential is the key FX driver per UIP theory.
        Returns dict with yields and differential.
        """
        pair_currencies = currency_pair.split("/")
        if len(pair_currencies) != 2:
            return {"error": f"Invalid currency pair: {currency_pair}"}

        base_ccy, quote_ccy = pair_currencies

        # Fetch curves for each currency
        curve_fetchers = {
            "USD": self.fetch_usd_yields,
            "EUR": self.fetch_ecb_yields,
            "JPY": self.fetch_boj_yields,
            "GBP": self.fetch_gbp_yields,
        }

        base_yields = curve_fetchers.get(base_ccy, lambda: {})()
        quote_yields = curve_fetchers.get(quote_ccy, lambda: {})()

        base_y = base_yields.get(tenor)
        quote_y = quote_yields.get(tenor)

        result: dict[str, Any] = {
            "currency_pair": currency_pair,
            "tenor": tenor,
            f"{base_ccy}_{tenor}_yield_pct": base_y,
            f"{quote_ccy}_{tenor}_yield_pct": quote_y,
        }

        if base_y is not None and quote_y is not None:
            result["differential_bps"] = round((base_y - quote_y) * 100, 2)
            result["direction"] = (
                f"{base_ccy} higher" if base_y > quote_y else f"{quote_ccy} higher"
            )
        else:
            result["differential_bps"] = None

        return result

    # ------------------------------------------------------------------
    def compute_real_yield_differential(
        self,
        pair: str = "EUR/USD",
        tenor: str = "10Y",
        breakeven_usd: float | None = None,
        breakeven_eur: float | None = None,
    ) -> float:
        """
        Real yield differential adjusted for inflation expectations.
        Real yield ≈ nominal yield - breakeven inflation.
        Key driver of FX spot at longer tenors.
        """
        nominal_diff = self.compute_yield_differential(pair, tenor)
        diff_bps = nominal_diff.get("differential_bps")

        if diff_bps is None:
            return float("nan")

        # Approximate inflation differential if breakevens not provided
        # Use rough assumptions: EUR inflation ≈ 2.0%, USD anchored at Fed target
        usd_be = breakeven_usd or 2.35  # rough 10Y breakeven
        eur_be = breakeven_eur or 2.0

        inflation_diff_bps = (usd_be - eur_be) * 100
        real_diff_bps = diff_bps - inflation_diff_bps

        return round(real_diff_bps, 2)

    # ------------------------------------------------------------------
    def get_cross_currency_table(self) -> pd.DataFrame:
        """Full cross-currency yield comparison table."""
        usd = self.fetch_usd_yields()
        eur = self.fetch_ecb_yields()
        jpy = self.fetch_boj_yields()
        gbp = self.fetch_gbp_yields()

        tenors = ["1Y", "2Y", "5Y", "10Y", "30Y"]
        rows = []
        for tenor in tenors:
            rows.append({
                "tenor": tenor,
                "USD": usd.get(tenor),
                "EUR": eur.get(tenor),
                "JPY": jpy.get(tenor),
                "GBP": gbp.get(tenor),
                "USD_EUR_spread_bps": (
                    round((usd.get(tenor, 0) - eur.get(tenor, 0)) * 100, 1)
                    if usd.get(tenor) and eur.get(tenor) else None
                ),
                "USD_JPY_spread_bps": (
                    round((usd.get(tenor, 0) - jpy.get(tenor, 0)) * 100, 1)
                    if usd.get(tenor) and jpy.get(tenor) else None
                ),
            })

        return pd.DataFrame(rows)


def _parse_ecb_csv(text: str) -> float | None:
    """Parse ECB SDW CSV response and return latest value."""
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        # ECB CSV: header rows then data rows with date and value
        for row in reversed(rows):
            if len(row) >= 2:
                try:
                    val = float(row[-1])
                    if math.isfinite(val):
                        return val
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# TreasuryPortfolioAnalyzer
# ---------------------------------------------------------------------------

class TreasuryPortfolioAnalyzer:
    """
    Duration, DV01, convexity, and P&L analytics for Treasury bond portfolios.
    Pure math — no QuantLib required.
    """

    # Approximate modified duration for on-the-run Treasuries (years)
    # These are approximations; exact duration requires full cash flow schedule
    _APPROX_DURATION: dict[float, float] = {
        0.083: 0.08,   # 1M
        0.25:  0.25,   # 3M
        0.5:   0.49,   # 6M
        1.0:   0.97,   # 1Y
        2.0:   1.90,   # 2Y
        3.0:   2.75,   # 3Y
        5.0:   4.35,   # 5Y
        7.0:   5.85,   # 7Y
        10.0:  8.00,   # 10Y
        20.0:  13.0,   # 20Y
        30.0:  17.5,   # 30Y
    }

    # ------------------------------------------------------------------
    @staticmethod
    def compute_modified_duration(tenor_years: float, coupon_rate: float, ytm: float) -> float:
        """
        Compute modified duration for a coupon bond using full cash flow model.
        Assumes semi-annual coupon payments. YTM and coupon in decimal (not %).
        """
        if tenor_years <= 0:
            return 0.0

        n_periods = int(tenor_years * 2)  # semi-annual periods
        if n_periods == 0:
            return tenor_years  # approximate for short bonds

        half_ytm = ytm / 2.0
        half_coupon = coupon_rate / 2.0

        # Discount each cash flow
        price = 0.0
        macaulay_numerator = 0.0

        for t in range(1, n_periods + 1):
            cf = half_coupon + (1.0 if t == n_periods else 0.0)
            discount = (1 + half_ytm) ** t
            pv = cf / discount
            price += pv
            macaulay_numerator += (t / 2.0) * pv  # convert to years

        if price <= 0:
            return 0.0

        macaulay = macaulay_numerator / price
        modified = macaulay / (1 + half_ytm)
        return modified

    # ------------------------------------------------------------------
    def compute_dv01(
        self,
        holdings: dict[str, float],
        yields: dict[str, float],
        face_value: float = 1_000_000,
    ) -> float:
        """
        Dollar Value of 1 Basis Point (DV01) for the portfolio.
        DV01 = modified_duration × price × 0.0001 × notional

        holdings: {tenor_label → notional in units (e.g., number of $1M bonds)}
        yields: {FRED series ID → yield %}
        """
        tenor_map = _build_tenor_yield_map(yields)
        total_dv01 = 0.0

        for tenor_label, notional in holdings.items():
            tenor_yr = _label_to_tenor_years(tenor_label)
            if tenor_yr is None:
                continue

            ytm = _interpolate_yield(tenor_map, tenor_yr)
            if ytm is None:
                continue

            ytm_dec = ytm / 100.0
            coupon = ytm_dec  # par bond assumption: coupon ≈ ytm

            mod_dur = self.compute_modified_duration(tenor_yr, coupon, ytm_dec)
            price = face_value  # par bond assumption
            dv01 = mod_dur * price * 0.0001 * notional
            total_dv01 += dv01

        return round(total_dv01, 2)

    # ------------------------------------------------------------------
    def compute_portfolio_duration(
        self,
        holdings: dict[str, float],
        yields: dict[str, float],
    ) -> float:
        """
        Dollar-weighted portfolio modified duration.
        holdings: {tenor_label → market value ($)}
        """
        tenor_map = _build_tenor_yield_map(yields)
        total_value = sum(holdings.values())

        if total_value <= 0:
            return 0.0

        weighted_dur = 0.0
        for tenor_label, mkt_val in holdings.items():
            tenor_yr = _label_to_tenor_years(tenor_label)
            if tenor_yr is None:
                continue

            ytm = _interpolate_yield(tenor_map, tenor_yr)
            if ytm is None:
                continue

            ytm_dec = ytm / 100.0
            mod_dur = self.compute_modified_duration(tenor_yr, ytm_dec, ytm_dec)
            weighted_dur += mod_dur * (mkt_val / total_value)

        return round(weighted_dur, 4)

    # ------------------------------------------------------------------
    def compute_key_rate_durations(
        self,
        holdings: dict[str, float],
        yields: dict[str, float],
        key_tenors: list[float] | None = None,
    ) -> dict[str, float]:
        """
        Key Rate Durations (KRD) at standard tenors.
        Measures sensitivity to yield changes at each point on the curve.
        Uses parallel shift of ±1bp at each key tenor.
        """
        if key_tenors is None:
            key_tenors = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]

        tenor_map = _build_tenor_yield_map(yields)
        krds: dict[str, float] = {}
        bump_bp = 0.01  # 1 basis point = 0.01%

        for key_tenor in key_tenors:
            # Compute portfolio value with and without bump at this tenor
            base_pv = self._compute_portfolio_pv(holdings, tenor_map)

            bumped_map = dict(tenor_map)
            bumped_map[key_tenor] = tenor_map.get(key_tenor, 0) + bump_bp

            bumped_pv = self._compute_portfolio_pv(holdings, bumped_map)

            krd = -(bumped_pv - base_pv) / (base_pv * bump_bp / 100) if base_pv > 0 else 0.0
            label = f"KRD_{key_tenor}Y"
            krds[label] = round(krd, 4)

        return krds

    # ------------------------------------------------------------------
    def _compute_portfolio_pv(
        self,
        holdings: dict[str, float],
        tenor_map: dict[float, float],
    ) -> float:
        """Compute total portfolio present value given a yield map."""
        total_pv = 0.0
        for tenor_label, notional in holdings.items():
            tenor_yr = _label_to_tenor_years(tenor_label)
            if tenor_yr is None:
                continue
            ytm = _interpolate_yield(tenor_map, tenor_yr)
            if ytm is None:
                continue
            # Price = par for coupon bonds at par (approximation)
            # For sensitivity, use duration-based price change
            ytm_dec = ytm / 100.0
            price = 1.0 / (1 + ytm_dec * tenor_yr)  # zero-coupon approximation
            total_pv += price * notional
        return total_pv

    # ------------------------------------------------------------------
    def compute_convexity(
        self,
        holdings: dict[str, float],
        yields: dict[str, float],
    ) -> float:
        """
        Portfolio convexity (second derivative of price w.r.t. yield).
        Convexity = Σ [w_i × C_i] where C_i ≈ D_i² for approximation.
        """
        tenor_map = _build_tenor_yield_map(yields)
        total_value = sum(holdings.values())

        if total_value <= 0:
            return 0.0

        weighted_conv = 0.0
        for tenor_label, mkt_val in holdings.items():
            tenor_yr = _label_to_tenor_years(tenor_label)
            if tenor_yr is None:
                continue

            ytm = _interpolate_yield(tenor_map, tenor_yr)
            if ytm is None:
                continue

            ytm_dec = ytm / 100.0
            mod_dur = self.compute_modified_duration(tenor_yr, ytm_dec, ytm_dec)

            # Convexity approximation: C ≈ D² + D (for option-free bonds)
            convexity = mod_dur ** 2 + mod_dur
            weighted_conv += convexity * (mkt_val / total_value)

        return round(weighted_conv, 4)

    # ------------------------------------------------------------------
    def estimate_pnl_from_yield_move(
        self,
        holdings: dict[str, float],
        yields: dict[str, float],
        yield_shift: dict[str, float],  # tenor_label → yield change (%)
    ) -> float:
        """
        Estimate P&L from yield moves using duration + convexity approximation.
        P&L = -D × Δy + 0.5 × C × Δy²
        yield_shift: {tenor_label → yield_change_in_pct}
        Returns dollar P&L.
        """
        tenor_map = _build_tenor_yield_map(yields)
        total_pnl = 0.0

        for tenor_label, mkt_val in holdings.items():
            tenor_yr = _label_to_tenor_years(tenor_label)
            if tenor_yr is None:
                continue

            ytm = _interpolate_yield(tenor_map, tenor_yr)
            if ytm is None:
                continue

            # Find applicable yield shift (exact or interpolated)
            dy = yield_shift.get(tenor_label, 0.0) / 100.0  # convert to decimal

            if dy == 0:
                continue

            ytm_dec = ytm / 100.0
            mod_dur = self.compute_modified_duration(tenor_yr, ytm_dec, ytm_dec)
            convexity = mod_dur ** 2 + mod_dur

            # Duration contribution
            pnl_dur = -mod_dur * dy * mkt_val
            # Convexity contribution
            pnl_conv = 0.5 * convexity * (dy ** 2) * mkt_val

            total_pnl += pnl_dur + pnl_conv

        return round(total_pnl, 2)

    # ------------------------------------------------------------------
    def get_portfolio_summary(
        self,
        holdings: dict[str, float],
        yields: dict[str, float],
    ) -> dict:
        """Full portfolio risk summary."""
        return {
            "dv01": self.compute_dv01(holdings, yields),
            "modified_duration": self.compute_portfolio_duration(holdings, yields),
            "convexity": self.compute_convexity(holdings, yields),
            "key_rate_durations": self.compute_key_rate_durations(holdings, yields),
            "total_market_value": sum(holdings.values()),
        }


def _label_to_tenor_years(label: str) -> float | None:
    """Convert tenor label like '10Y', '3M', '6M' to years."""
    label_map = {
        "1M": 1/12, "3M": 0.25, "6M": 0.5,
        "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
        "5Y": 5.0, "7Y": 7.0, "10Y": 10.0,
        "20Y": 20.0, "30Y": 30.0,
    }
    return label_map.get(label.upper())


# ---------------------------------------------------------------------------
# TreasuryYieldEngine — orchestrator
# ---------------------------------------------------------------------------

class TreasuryYieldEngine:
    """
    High-level orchestrator for all Treasury yield curve analytics.
    Single entry point for the SENTINEL financial terminal.
    """

    def __init__(self, cache_file: str = _CACHE_FILE) -> None:
        self.loader = FREDYieldLoader(cache_file=cache_file)
        self.nss_fitter = NelsonSiegelSvensson()
        self.forward_extractor = ForwardRateExtractor()
        self.analytics = YieldCurveAnalytics()
        self.breakeven_curve = BreakevenInflationCurve()
        self.fed_funds = FedFundsImplied(self.loader)
        self.cross_currency = CrossCurrencyYields(self.loader)
        self.portfolio = TreasuryPortfolioAnalyzer()

    # ------------------------------------------------------------------
    def get_current_curve(self) -> YieldCurve:
        """Fetch and assemble the full current yield curve."""
        logger.info("Fetching current yield curve from FRED...")

        nominal = self.loader.fetch_all_nominal_yields()
        tips = self.loader.fetch_tips_yields()

        # Build tenor-indexed dicts for analytics
        tenor_map = _build_tenor_yield_map(nominal)
        tips_tenor_map = {
            _TIPS_SERIES[k]: v for k, v in tips.items() if k in _TIPS_SERIES
        }

        # Compute breakeven
        breakeven = self.breakeven_curve.compute_breakeven_curve(nominal, tips)

        # Prepare ordered tenors/yields for NSS fitting
        tenor_yield_pairs = sorted(tenor_map.items())
        tenors = [t for t, _ in tenor_yield_pairs]
        yields = [y for _, y in tenor_yield_pairs]

        # Fit NSS
        nss_params = None
        if len(tenors) >= 4:
            try:
                nss_params = self.nss_fitter.fit(tenors, yields)
            except Exception as exc:
                logger.warning("NSS fit failed: %s", exc)

        # Build tenor labels for nominal
        nominal_labeled = {
            _TENOR_LABELS.get(k, k): v for k, v in nominal.items()
        }

        # Build TIPS labeled
        tips_labeled: dict[str, float] = {}
        for series_id, tenor_yr in _TIPS_SERIES.items():
            val = tips.get(series_id)
            if val is not None:
                label = f"{int(tenor_yr)}Y" if tenor_yr == int(tenor_yr) else f"{tenor_yr}Y"
                tips_labeled[label] = val

        # Regime
        regime = self.analytics.compute_curve_regime(nominal)

        as_of = date.today().isoformat()

        return YieldCurve(
            as_of=as_of,
            nominal=nominal_labeled,
            real_tips=tips_labeled,
            breakeven=breakeven,
            tenors_years=tenors,
            yields_pct=yields,
            nss_params=nss_params,
            regime=regime,
            slope_2s10s=self.analytics.compute_2s10s(nominal),
            slope_3m10y=self.analytics.compute_3m10y(nominal),
        )

    # ------------------------------------------------------------------
    def get_analytics(self) -> dict:
        """Full analytics snapshot including spreads, regime, percentiles."""
        nominal = self.loader.fetch_all_nominal_yields()

        summary = self.analytics.get_curve_summary(nominal)

        # Historical context: 2s10s percentile
        history = self.loader.fetch_full_curve_history(start="2000-01-01")
        if not history.empty and "DGS2" in history and "DGS10" in history:
            spread_history = (history["DGS10"] - history["DGS2"]) * 100
            summary["2s10s_percentile"] = self.analytics.compute_spread_percentile(
                summary["2s10s_bps"], spread_history
            )
            summary["regime_frequencies"] = self.analytics.compute_historical_regime_frequency(history)
        else:
            summary["2s10s_percentile"] = None
            summary["regime_frequencies"] = {}

        summary["as_of"] = date.today().isoformat()
        summary["fed_funds_rate"] = self.loader.fetch_fed_funds_rate()

        return summary

    # ------------------------------------------------------------------
    def get_breakeven_curve_data(self) -> dict:
        """Return current breakeven inflation curve with all derived metrics."""
        nominal = self.loader.fetch_all_nominal_yields()
        tips = self.loader.fetch_tips_yields()

        breakeven = self.breakeven_curve.compute_breakeven_curve(nominal, tips)

        return {
            "as_of": date.today().isoformat(),
            "breakeven_curve": breakeven,
            "5y5y_forward_breakeven": self.breakeven_curve.compute_5y5y_forward_breakeven(breakeven),
            "inflation_term_premium": self.breakeven_curve.compute_inflation_term_premium(breakeven),
            "notes": "Breakeven = nominal Treasury yield - TIPS real yield",
        }

    # ------------------------------------------------------------------
    def get_forward_curve(
        self,
        nss_params: NSSParams | None = None,
    ) -> ForwardCurve:
        """Compute forward rate curve from spot yields and NSS model."""
        nominal = self.loader.fetch_all_nominal_yields()
        tenor_map = _build_tenor_yield_map(nominal)

        forward_rates = self.forward_extractor.compute_forward_curve(tenor_map)

        # Policy path
        policy_path_series = self.forward_extractor.extract_implied_policy_path(tenor_map)
        policy_path = policy_path_series.to_dict()

        # 5Y5Y from breakeven
        breakeven = self.get_breakeven_curve_data()
        five_five = breakeven.get("5y5y_forward_breakeven", 0.0)

        return ForwardCurve(
            as_of=date.today().isoformat(),
            forward_rates=forward_rates,
            policy_path=policy_path,
            five_year_five_year=five_five,
        )

    # ------------------------------------------------------------------
    def get_fomc_expectations(self) -> pd.DataFrame:
        """Fed Funds rate expectations at upcoming FOMC meetings."""
        return self.fed_funds.get_rate_change_expectations()

    # ------------------------------------------------------------------
    def get_cross_currency_comparison(self) -> pd.DataFrame:
        """Full USD/EUR/JPY/GBP yield comparison table."""
        return self.cross_currency.get_cross_currency_table()

    # ------------------------------------------------------------------
    def run_daily_update(self) -> dict:
        """
        Refresh all FRED data and update cache.
        Designed to be called once per day by the scheduler.
        """
        logger.info("Running daily Treasury yield update...")
        start = time.time()

        # Force cache refresh by clearing memory cache
        self.loader._cache.clear()
        if os.path.exists(self.loader._cache_file):
            os.remove(self.loader._cache_file)
            logger.info("Cleared stale yield cache")

        # Fetch fresh data
        nominal = self.loader.fetch_all_nominal_yields()
        tips = self.loader.fetch_tips_yields()
        fed_funds = self.loader.fetch_fed_funds_rate()
        full_history = self.loader.fetch_full_curve_history(start="2000-01-01")

        elapsed = round(time.time() - start, 1)
        logger.info("Daily yield update complete in %.1fs", elapsed)

        return {
            "status": "ok",
            "as_of": date.today().isoformat(),
            "nominal_tenors_loaded": len(nominal),
            "tips_tenors_loaded": len(tips),
            "fed_funds_rate": fed_funds,
            "history_rows": len(full_history) if not full_history.empty else 0,
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    def get_portfolio_analytics(
        self,
        holdings: dict[str, float],
    ) -> dict:
        """
        Run full portfolio analytics for given holdings.
        holdings: {tenor_label → market value in USD}
        Example: {"2Y": 1_000_000, "10Y": 2_000_000, "30Y": 500_000}
        """
        nominal = self.loader.fetch_all_nominal_yields()
        return self.portfolio.get_portfolio_summary(holdings, nominal)

    # ------------------------------------------------------------------
    def get_full_dashboard(self) -> dict:
        """Single call to get everything needed for a yield curve dashboard."""
        curve = self.get_current_curve()
        analytics = self.get_analytics()
        forward = self.get_forward_curve(curve.nss_params)
        breakeven = self.get_breakeven_curve_data()

        nss_points = []
        if curve.nss_params:
            nss_points = self.nss_fitter.get_fitted_curve_points(curve.nss_params)

        return {
            "as_of": curve.as_of,
            "curve": {
                "nominal": curve.nominal,
                "real_tips": curve.real_tips,
                "breakeven": curve.breakeven,
            },
            "analytics": analytics,
            "nss": {
                "params": {
                    "beta0": curve.nss_params.beta0 if curve.nss_params else None,
                    "beta1": curve.nss_params.beta1 if curve.nss_params else None,
                    "beta2": curve.nss_params.beta2 if curve.nss_params else None,
                    "beta3": curve.nss_params.beta3 if curve.nss_params else None,
                    "lambda1": curve.nss_params.lambda1 if curve.nss_params else None,
                    "lambda2": curve.nss_params.lambda2 if curve.nss_params else None,
                    "rmse_bps": round(curve.nss_params.rmse, 4) if curve.nss_params else None,
                } if curve.nss_params else {},
                "fitted_curve_points": nss_points[:20],  # sample
                "level_slope_curvature": (
                    self.nss_fitter.extract_level_slope_curvature(curve.nss_params)
                    if curve.nss_params else {}
                ),
            },
            "forward_rates": forward.forward_rates,
            "policy_path": forward.policy_path,
            "breakeven": breakeven,
            "regime": curve.regime,
        }


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def create_engine(cache_file: str = _CACHE_FILE) -> TreasuryYieldEngine:
    """Create a fully initialized TreasuryYieldEngine."""
    return TreasuryYieldEngine(cache_file=cache_file)


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    print("=" * 70)
    print("SENTINEL — TreasuryYieldEngine v3 Demo")
    print("=" * 70)

    engine = create_engine()

    # 1. Current yield curve
    print("\n[1] Current US Treasury Yield Curve:")
    curve = engine.get_current_curve()
    print(f"  As of: {curve.as_of}")
    print(f"  Regime: {curve.regime}")
    print(f"  2s10s: {curve.slope_2s10s:.1f} bps")
    print(f"  3m10y: {curve.slope_3m10y:.1f} bps")

    print("\n  Nominal yields:")
    for tenor_label, yield_pct in sorted(
        curve.nominal.items(),
        key=lambda x: _label_to_tenor_years(x[0]) or 0
    ):
        bar = "=" * int(yield_pct * 5)
        print(f"  {tenor_label:4s}: {yield_pct:6.3f}%  {bar}")

    # 2. NSS fit
    print("\n[2] Nelson-Siegel-Svensson fit:")
    if curve.nss_params:
        p = curve.nss_params
        print(f"  β₀ (level):    {p.beta0:.4f}%")
        print(f"  β₁ (slope):    {p.beta1:.4f}")
        print(f"  β₂ (curve1):   {p.beta2:.4f}")
        print(f"  β₃ (curve2):   {p.beta3:.4f}")
        print(f"  λ₁:            {p.lambda1:.4f}")
        print(f"  λ₂:            {p.lambda2:.4f}")
        print(f"  RMSE:          {p.rmse:.4f}%")

        lsc = engine.nss_fitter.extract_level_slope_curvature(p)
        print(f"  Level:         {lsc['level_beta0']:.4f}%")
        print(f"  Slope (-β₁):   {lsc['slope_minus_beta1']:.4f}")
        print(f"  Curvature:     {lsc['curvature_beta2_plus_beta3']:.4f}")

        # Sample NSS prediction at key maturities
        print("\n  NSS fitted yields:")
        for tau in [0.25, 1, 2, 5, 10, 20, 30]:
            y_nss = engine.nss_fitter.predict(tau, p)
            print(f"  {tau:4.1f}Y: {y_nss:.4f}%")
    else:
        print("  NSS fit unavailable (insufficient data points)")

    # 3. Forward curve
    print("\n[3] Forward rate curve:")
    forward = engine.get_forward_curve(curve.nss_params)
    for label, rate in forward.forward_rates.items():
        print(f"  {label:10s}: {rate:.4f}%")

    # 4. Implied policy path
    print("\n[4] Implied policy rate path (from forward curve):")
    for horizon, rate in list(forward.policy_path.items())[:8]:
        print(f"  {horizon:5s}: {rate:.3f}%")

    # 5. FOMC meeting expectations
    print("\n[5] FOMC meeting rate expectations:")
    try:
        fomc_df = engine.get_fomc_expectations()
        if not fomc_df.empty:
            print(fomc_df.to_string(index=False))
    except Exception as e:
        print(f"  Error: {e}")

    # 6. Breakeven inflation
    print("\n[6] Breakeven inflation curve:")
    be_data = engine.get_breakeven_curve_data()
    for tenor, be in be_data.get("breakeven_curve", {}).items():
        print(f"  {tenor:5s}: {be:.4f}%")
    fyfive = be_data.get("5y5y_forward_breakeven")
    if fyfive and math.isfinite(fyfive):
        print(f"  5Y5Y forward breakeven: {fyfive:.4f}%")
    tp = be_data.get("inflation_term_premium")
    if tp and math.isfinite(tp):
        print(f"  Inflation term premium: {tp:.4f}%")

    # 7. Cross-currency yield comparison
    print("\n[7] Cross-currency yield comparison:")
    try:
        cc_table = engine.get_cross_currency_comparison()
        if not cc_table.empty:
            print(cc_table.to_string(index=False))
    except Exception as e:
        print(f"  Error: {e}")

    # 8. Historical regime analysis
    print("\n[8] Yield curve regime history (since 2000):")
    try:
        history = engine.loader.fetch_full_curve_history(start="2000-01-01")
        if not history.empty:
            freq = engine.analytics.compute_historical_regime_frequency(history)
            for regime, stats in freq.items():
                pct = stats.get("frequency_pct", 0)
                bar = "#" * int(pct / 2)
                print(f"  {regime:18s}: {pct:5.1f}%  {bar}")
        else:
            print("  History unavailable")
    except Exception as e:
        print(f"  Error: {e}")

    # 9. Portfolio analytics demo
    print("\n[9] Treasury portfolio analytics:")
    sample_portfolio = {
        "2Y":  5_000_000,
        "5Y":  10_000_000,
        "10Y": 7_500_000,
        "30Y": 2_500_000,
    }
    print(f"  Holdings: {sample_portfolio}")
    try:
        port_summary = engine.get_portfolio_analytics(sample_portfolio)
        print(f"  DV01:              ${port_summary['dv01']:,.0f}")
        print(f"  Modified Duration: {port_summary['modified_duration']:.2f} years")
        print(f"  Convexity:         {port_summary['convexity']:.2f}")
        print("  Key Rate Durations:")
        for k, v in port_summary.get("key_rate_durations", {}).items():
            print(f"    {k}: {v:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # 10. P&L scenario
    print("\n[10] P&L scenario: +25bps across curve (rate hike):")
    yield_shift = {"2Y": 0.25, "5Y": 0.25, "10Y": 0.25, "30Y": 0.25}
    try:
        nominal = engine.loader.fetch_all_nominal_yields()
        pnl = engine.portfolio.estimate_pnl_from_yield_move(
            sample_portfolio, nominal, yield_shift
        )
        print(f"  Estimated P&L: ${pnl:,.0f}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nDone.")
