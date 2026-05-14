"""
Yield curve analytics — Dimension #44.

US Treasury spot rates, Nelson-Siegel fit, implied forward rates, slope signals,
recession classification, OAS credit spreads, and historical slope trend.

Data sources (all free, FRED CSV):
  Treasury: GS1M GS3M GS6M GS1 GS2 GS5 GS7 GS10 GS20 GS30
  TIPS real yields: DFII5 DFII7 DFII10 DFII20 DFII30
  OAS: BAMLC0A0CM (IG), BAMLH0A0HYM2 (HY)
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, datetime
from typing import Optional

import httpx
import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 20.0
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# (label, FRED series, tenor_years)
_TREASURY_TENORS: list[tuple[str, str, float]] = [
    ("1M",  "GS1M",  1 / 12),
    ("3M",  "GS3M",  0.25),
    ("6M",  "GS6M",  0.5),
    ("1Y",  "GS1",   1.0),
    ("2Y",  "GS2",   2.0),
    ("5Y",  "GS5",   5.0),
    ("7Y",  "GS7",   7.0),
    ("10Y", "GS10",  10.0),
    ("20Y", "GS20",  20.0),
    ("30Y", "GS30",  30.0),
]

# (FRED series, tenor_years)
_TIPS_SERIES: list[tuple[str, float]] = [
    ("DFII5",  5.0), ("DFII7",  7.0), ("DFII10", 10.0),
    ("DFII20", 20.0), ("DFII30", 30.0),
]

_OAS_IG_SERIES  = "BAMLC0A0CM"
_OAS_HY_SERIES  = "BAMLH0A0HYM2"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TenorPoint(BaseModel):
    tenor: str                         # "1M","3M","6M","1Y","2Y","5Y","7Y","10Y","20Y","30Y"
    yield_pct: float
    real_yield_pct: Optional[float] = None   # TIPS-based if available


class YieldCurveResult(BaseModel):
    as_of: str
    curve: list[TenorPoint]
    slope_10y2y: float          # bps
    slope_10y3m: float          # bps
    inversion_flag: bool
    recession_signal: str       # "inverted"|"flat"|"normal"|"steep"
    fitted_curve: list[dict]    # [{tenor_years, fitted_yield}]
    ns_params: dict             # {beta0, beta1, beta2, lambda_}
    forward_rates: dict[str, float]
    slope_history: list[dict]   # [{date, slope_10y2y}]
    oas_ig_bps: Optional[float] = None
    oas_hy_bps: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FRED CSV helpers
# ---------------------------------------------------------------------------

async def _fred_series_history(
    client: httpx.AsyncClient, series_id: str, rows: int = 150,
) -> dict[str, float]:
    """Fetch FRED CSV → {date_str: float}. Missing dots excluded."""
    try:
        r = await client.get(_FRED_CSV, params={"id": series_id}, timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.warning("FRED CSV non-200", series=series_id, status=r.status_code)
            return {}
        lines = r.text.strip().splitlines()[1:]   # skip header
        lines = lines[-rows:] if len(lines) > rows else lines
        result: dict[str, float] = {}
        for line in lines:
            parts = line.split(",")
            if len(parts) != 2:
                continue
            dt_str, val_str = parts[0].strip(), parts[1].strip()
            if val_str in (".", "", "NA"):
                continue
            try:
                result[dt_str] = float(val_str)
            except ValueError:
                continue
        return result
    except Exception as exc:
        logger.warning("FRED fetch failed", series=series_id, error=str(exc))
        return {}


async def _fred_value_on_date(
    client: httpx.AsyncClient, series_id: str, target_date: str, rows: int = 30,
) -> Optional[float]:
    """Most recent value on or before target_date."""
    history = await _fred_series_history(client, series_id, rows=rows)
    candidates = sorted(
        ((dt, v) for dt, v in history.items() if dt <= target_date),
        key=lambda x: x[0], reverse=True,
    )
    return candidates[0][1] if candidates else None


# ---------------------------------------------------------------------------
# Nelson-Siegel fitting
# ---------------------------------------------------------------------------

def _ns_yield(tau: float, b0: float, b1: float, b2: float, lam: float) -> float:
    """Scalar Nelson-Siegel: y(τ) = β0 + β1·L(τ/λ) + β2·[L(τ/λ) − exp(−τ/λ)]"""
    if tau <= 0.0:
        return b0 + b1
    x = tau / lam
    ex = math.exp(-x)
    L = (1.0 - ex) / x
    return b0 + b1 * L + b2 * (L - ex)


def _ns_vec(taus: np.ndarray, p: np.ndarray) -> np.ndarray:
    b0, b1, b2, lam = p
    lam = max(lam, 0.01)
    x = taus / lam
    ex = np.exp(-x)
    L = (1.0 - ex) / x
    return b0 + b1 * L + b2 * (L - ex)


def fit_nelson_siegel(tenors_years: list[float], yields_pct: list[float]) -> dict:
    """Fit NS to observed curve via Nelder-Mead. Returns param dict + _warnings."""
    taus = np.array(tenors_years, dtype=float)
    ylds = np.array(yields_pct, dtype=float)

    def obj(p: np.ndarray) -> float:
        if p[0] <= 0 or p[3] <= 0.001:
            return 1e9
        return float(np.sum((_ns_vec(taus, p) - ylds) ** 2))

    long_end = float(np.max(ylds)) if len(ylds) else 4.0
    x0 = np.array([long_end, float(ylds[0]) - long_end if len(ylds) else -0.5, 0.5, 1.5])
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})

    warn: list[str] = []
    if not res.success:
        warn.append(f"NS fit non-convergence: {res.message}")
        logger.warning("NS fit", message=res.message)

    b0, b1, b2, lam = res.x
    return {
        "beta0": round(float(b0), 6), "beta1": round(float(b1), 6),
        "beta2": round(float(b2), 6), "lambda_": round(max(float(lam), 0.01), 6),
        "_warnings": warn,
    }


def build_fitted_curve(ns: dict, n: int = 50) -> list[dict]:
    """Smooth NS curve, log-spaced 1M→30Y."""
    b0, b1, b2, lam = ns["beta0"], ns["beta1"], ns["beta2"], ns["lambda_"]
    tenors = np.exp(np.linspace(math.log(1 / 12), math.log(30), n))
    return [
        {"tenor_years": round(float(t), 4), "fitted_yield": round(_ns_yield(t, b0, b1, b2, lam), 4)}
        for t in tenors
    ]


# ---------------------------------------------------------------------------
# Forward rates
# ---------------------------------------------------------------------------

def compute_forward_rates(tenors_years: list[float], yields_pct: list[float]) -> dict[str, float]:
    """
    Implied forwards via bootstrapping: f(t1,t2) = (Y2·t2 − Y1·t1)/(t2−t1).
    Linear interpolation for off-grid tenors.
    """
    def interp(t: float) -> Optional[float]:
        if not tenors_years:
            return None
        if t <= tenors_years[0]:
            return yields_pct[0]
        if t >= tenors_years[-1]:
            return yields_pct[-1]
        for i in range(len(tenors_years) - 1):
            t1, t2 = tenors_years[i], tenors_years[i + 1]
            if t1 <= t <= t2:
                w = (t - t1) / (t2 - t1)
                return yields_pct[i] * (1 - w) + yields_pct[i + 1] * w
        return None

    horizons = {
        "6M_fwd_6M": (0.5, 1.0), "1Y_fwd_1Y": (1.0, 2.0),
        "2Y_fwd_2Y": (2.0, 4.0), "5Y_fwd_5Y": (5.0, 10.0),
    }
    result: dict[str, float] = {}
    for label, (t1, t2) in horizons.items():
        y1, y2 = interp(t1), interp(t2)
        if y1 is not None and y2 is not None:
            result[label] = round((y2 * t2 - y1 * t1) / (t2 - t1), 4)
    return result


# ---------------------------------------------------------------------------
# Slope history
# ---------------------------------------------------------------------------

async def _fetch_slope_history(client: httpx.AsyncClient, days: int) -> list[dict]:
    """Daily 10Y-2Y slope (bps) for last `days` dates, ascending."""
    rows = days + 30
    gs2_data, gs10_data = await asyncio.gather(
        _fred_series_history(client, "GS2",  rows=rows),
        _fred_series_history(client, "GS10", rows=rows),
    )
    common = sorted(set(gs2_data) & set(gs10_data))[-days:]
    return [{"date": dt, "slope_10y2y": round((gs10_data[dt] - gs2_data[dt]) * 100, 2)}
            for dt in common]


# ---------------------------------------------------------------------------
# Recession signal
# ---------------------------------------------------------------------------

def classify_recession_signal(slope_bps: float) -> str:
    if slope_bps < 0:
        return "inverted"
    if slope_bps < 30:
        return "flat"
    if slope_bps > 200:
        return "steep"
    return "normal"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def get_yield_curve(
    as_of: Optional[str] = None,
    include_history_days: int = 90,
) -> YieldCurveResult:
    """
    Fetch and analyse the US Treasury yield curve.

    Args:
        as_of:                ISO date (YYYY-MM-DD); None = today.
        include_history_days: Days of 10Y-2Y slope history to include.

    Returns:
        YieldCurveResult with spot curve, NS fit, forward rates, OAS spreads,
        and slope history.
    """
    warnings_: list[str] = []

    if as_of is None:
        as_of_date = date.today().isoformat()
    else:
        try:
            datetime.fromisoformat(as_of)
            as_of_date = as_of
        except ValueError:
            warnings_.append(f"Invalid as_of '{as_of}', using today.")
            as_of_date = date.today().isoformat()

    rows_needed = include_history_days + 30

    async with httpx.AsyncClient() as client:
        treasury_tasks = {
            label: _fred_value_on_date(client, series, as_of_date, rows=rows_needed)
            for label, series, _ in _TREASURY_TENORS
        }
        tips_tasks = {
            series: _fred_value_on_date(client, series, as_of_date, rows=60)
            for series, _ in _TIPS_SERIES
        }

        all_keys = (list(treasury_tasks) + list(tips_tasks) + ["OAS_IG", "OAS_HY"])
        all_coros = (
            list(treasury_tasks.values()) + list(tips_tasks.values())
            + [_fred_value_on_date(client, _OAS_IG_SERIES, as_of_date, rows=60),
               _fred_value_on_date(client, _OAS_HY_SERIES, as_of_date, rows=60)]
        )
        gathered = await asyncio.gather(
            _fetch_slope_history(client, include_history_days),
            *all_coros,
            return_exceptions=True,
        )

    slope_history_raw, *values_raw = gathered

    if isinstance(slope_history_raw, Exception):
        warnings_.append(f"Slope history failed: {slope_history_raw}")
        slope_history: list[dict] = []
    else:
        slope_history = slope_history_raw  # type: ignore[assignment]

    value_map: dict[str, Optional[float]] = {}
    for key, val in zip(all_keys, values_raw):
        if isinstance(val, Exception):
            logger.warning("Series fetch failed", series=key, error=str(val))
            value_map[key] = None
        else:
            value_map[key] = val  # type: ignore[assignment]

    # TIPS lookup: tenor_years → real yield
    tips_lookup: dict[float, float] = {
        tenor_yr: value_map[series]  # type: ignore[assignment]
        for series, tenor_yr in _TIPS_SERIES
        if value_map.get(series) is not None
    }

    def _nearest_tips(t: float) -> Optional[float]:
        if not tips_lookup:
            return None
        k = min(tips_lookup, key=lambda x: abs(x - t))
        return round(tips_lookup[k], 4) if abs(k - t) <= 3.0 else None

    # Spot curve
    curve: list[TenorPoint] = []
    valid_tenors: list[float] = []
    valid_yields: list[float] = []

    for label, series, tenor_yr in _TREASURY_TENORS:
        val = value_map.get(label)
        if val is None:
            warnings_.append(f"Missing {label} ({series}) on {as_of_date}")
            continue
        curve.append(TenorPoint(
            tenor=label, yield_pct=round(val, 4), real_yield_pct=_nearest_tips(tenor_yr),
        ))
        valid_tenors.append(tenor_yr)
        valid_yields.append(val)

    if len(curve) < 3:
        warnings_.append(f"Only {len(curve)} tenors — analytics degraded.")

    # Slopes & signals
    ybl: dict[str, float] = {tp.tenor: tp.yield_pct for tp in curve}
    y10, y2, y3m = ybl.get("10Y"), ybl.get("2Y"), ybl.get("3M")

    slope_10y2y = round((y10 - y2) * 100, 2) if y10 is not None and y2 is not None else 0.0
    slope_10y3m = round((y10 - y3m) * 100, 2) if y10 is not None and y3m is not None else 0.0

    if y10 is None or y2 is None:
        warnings_.append("Cannot compute 10Y-2Y slope — missing rate(s).")
    if y10 is None or y3m is None:
        warnings_.append("Cannot compute 10Y-3M slope — missing rate(s).")

    inversion_flag = bool(y10 is not None and y2 is not None and y10 < y2)
    recession_signal = classify_recession_signal(slope_10y2y)

    # Nelson-Siegel
    ns_params: dict = {}
    fitted_curve: list[dict] = []
    if len(valid_tenors) >= 3:
        ns_raw = fit_nelson_siegel(valid_tenors, valid_yields)
        warnings_.extend(ns_raw.pop("_warnings", []))
        ns_params = ns_raw
        fitted_curve = build_fitted_curve(ns_params)
    else:
        warnings_.append("Need >= 3 tenors for Nelson-Siegel fit.")

    # Forward rates
    forward_rates: dict[str, float] = (
        compute_forward_rates(valid_tenors, valid_yields) if len(valid_tenors) >= 2
        else {}
    )
    if not forward_rates:
        warnings_.append("Insufficient data for forward rate calculation.")

    # OAS spreads (FRED reports as percent; multiply by 100 → bps)
    raw_ig = value_map.get("OAS_IG")
    raw_hy = value_map.get("OAS_HY")
    oas_ig_bps = round(raw_ig * 100, 1) if raw_ig is not None else None
    oas_hy_bps = round(raw_hy * 100, 1) if raw_hy is not None else None
    if oas_ig_bps is None:
        warnings_.append("IG OAS unavailable.")
    if oas_hy_bps is None:
        warnings_.append("HY OAS unavailable.")

    logger.info(
        "Yield curve assembled",
        as_of=as_of_date, tenors=len(curve),
        slope_10y2y_bps=slope_10y2y, signal=recession_signal,
    )

    return YieldCurveResult(
        as_of=as_of_date,
        curve=curve,
        slope_10y2y=slope_10y2y,
        slope_10y3m=slope_10y3m,
        inversion_flag=inversion_flag,
        recession_signal=recession_signal,
        fitted_curve=fitted_curve,
        ns_params=ns_params,
        forward_rates=forward_rates,
        slope_history=slope_history,
        oas_ig_bps=oas_ig_bps,
        oas_hy_bps=oas_hy_bps,
        warnings=warnings_,
    )
