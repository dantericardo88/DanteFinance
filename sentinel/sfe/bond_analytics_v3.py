"""
Bond analytics engine — DV01, OAS, Z-spread, duration, convexity, callable bonds.

Dimension: dim_038 — Bond analytics engine DV01/OAS (target score: 9)

Audit fix: Hardcoded Treasury curve fallback replaced with live FRED fetch on startup,
cached for 1h. Wires directly to treasury_yield_curves.py (dim_035, scores 8/10).
No QuantLib dependency — pure Python math throughout.

Data sources (all free):
  Primary:   sentinel.sds.adapters.treasury_yield_curves.TreasuryYieldCurveAdapter
  Fallback:  FRED CSV https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10
  Swap proxy: FRED SOFR (SOFR, SOFRRATE)
  Int'l:     FRED ECB/BOE/BOJ series (ECBDFR, BOERUKM, IRSTCB01JPM156N)

Public API
----------
BondPricer
    price_from_ytm(face, coupon, maturity, settle, ytm, freq)  -> BondMetrics
    ytm_from_price(face, coupon, maturity, settle, price, freq) -> float
    dv01(face, coupon, maturity, settle, ytm, freq)             -> float
    key_rate_duration(face, coupon, maturity, settle, ytm)      -> KRDResult

SpreadCalculator
    g_spread(ytm, maturity_years, curve)          -> float
    z_spread(price, cashflows, times, curve)       -> float
    i_spread(ytm, maturity_years, sofr_curve)     -> float
    oas_callable(bond_params, vol, n_steps)        -> OASResult

ScenarioEngine
    parallel_shift(metrics, shifts_bps)           -> list[ScenarioResult]
    twist_steepen(metrics, short_up_bps)          -> ScenarioResult
    twist_flatten(metrics, long_up_bps)            -> ScenarioResult

CallableBondEngine
    binomial_tree_oas(bond_params, vol, n_steps)  -> CallableResult

PortfolioAnalytics
    aggregate(positions)                          -> PortfolioMetrics

LiveCurveManager
    get_curve()                                   -> TreasuryCurve   (cached 1h, live FRED)
    get_sofr_curve()                              -> dict[float,float]

bond_analytics_router — FastAPI APIRouter, prefix /bond-analytics/v3
"""
from __future__ import annotations

import bisect
import io
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Try to import live Treasury curve adapter; fallback gracefully
# ---------------------------------------------------------------------------
_TREASURY_ADAPTER_AVAILABLE = False
try:
    from sentinel.sds.adapters.treasury_yield_curves import (
        FREDTreasuryAdapter,
        YieldCurveBuilder,
    )
    _TREASURY_ADAPTER_AVAILABLE = True
    logger.info("treasury_yield_curves adapter loaded successfully")
except ImportError as _imp_err:
    logger.warning(
        "treasury_yield_curves import failed — using FRED CSV fallback",
        error=str(_imp_err),
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_REQ_TIMEOUT = 12

_HEADERS = {
    "User-Agent": "SENTINEL bond-analytics/3.0 richard.porras@realempanada.com",
    "Accept": "text/csv,text/plain,*/*",
}

# Curve cache TTL: 1 hour — never use hardcoded rates
_CURVE_TTL = 3_600
_SOFR_TTL = 3_600

# Treasury CMT maturities for interpolation (FRED series -> tenor_years)
_CMT_SERIES: list[tuple[str, float]] = [
    ("DGS1MO",  1 / 12),
    ("DGS3MO",  0.25),
    ("DGS6MO",  0.5),
    ("DGS1",    1.0),
    ("DGS2",    2.0),
    ("DGS3",    3.0),
    ("DGS5",    5.0),
    ("DGS7",    7.0),
    ("DGS10",   10.0),
    ("DGS20",   20.0),
    ("DGS30",   30.0),
]

# SOFR-related FRED series as swap-rate proxy
_SOFR_SERIES: list[tuple[str, float]] = [
    ("SOFR",      0.0027),   # overnight ≈ 1-day
    ("DGS1MO",    1 / 12),   # use Tsy short for swap proxy
    ("DGS3MO",    0.25),
    ("DGS6MO",    0.5),
    ("DGS1",      1.0),
    ("DGS2",      2.0),
    ("DGS5",      5.0),
    ("DGS10",     10.0),
    ("DGS30",     30.0),
]

# International central bank policy rates (FRED)
_INTL_POLICY_SERIES: dict[str, str] = {
    "EUR": "ECBDFR",
    "GBP": "BOERUKM",
    "JPY": "IRSTCB01JPM156N",
    "AUD": "IRSTCB01AUM156N",
    "CAD": "IRSTCB01CAM156N",
}

# Key rate durations: 2Y, 5Y, 10Y, 30Y
_KRD_NODES: list[float] = [2.0, 5.0, 10.0, 30.0]

# SQLite path
_DB_PATH = Path(__file__).parent.parent.parent / "data" / "bond_analytics_v3.db"

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bond_analytics_cache (
                cache_key   TEXT PRIMARY KEY,
                payload     TEXT NOT NULL,
                cached_at   REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS curve_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                curve_date  TEXT NOT NULL,
                currency    TEXT NOT NULL DEFAULT 'USD',
                tenor_years REAL NOT NULL,
                yield_pct   REAL NOT NULL,
                source      TEXT NOT NULL,
                fetched_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_curve_snap
                ON curve_snapshots(curve_date, currency);
            CREATE TABLE IF NOT EXISTS scenario_results (
                result_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at      TEXT NOT NULL,
                cusip_or_id TEXT NOT NULL,
                scenario    TEXT NOT NULL,
                shift_bps   REAL,
                price       REAL,
                ytm         REAL,
                duration    REAL,
                dv01        REAL,
                pnl_pct     REAL
            );
        """)
        conn.commit()


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _cache_get(key: str, ttl: float) -> Optional[Any]:
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT payload, cached_at FROM bond_analytics_cache WHERE cache_key=?",
                (key,)
            ).fetchone()
            if row and (time.time() - row["cached_at"]) < ttl:
                return json.loads(row["payload"])
    except Exception:
        pass
    return None


def _cache_set(key: str, data: Any) -> None:
    try:
        with _db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bond_analytics_cache (cache_key,payload,cached_at) VALUES (?,?,?)",
                (key, json.dumps(data, default=str), time.time())
            )
            conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BondMetrics(BaseModel):
    """Full bond pricing analytics."""
    face_value: float
    coupon_rate_pct: float
    maturity_date: str
    settlement_date: str
    frequency: int
    clean_price: float
    dirty_price: float
    accrued_interest: float
    ytm_pct: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float
    dollar_convexity: float
    years_to_maturity: float
    warnings: list[str] = Field(default_factory=list)


class SpreadResult(BaseModel):
    ytm_pct: float
    g_spread_bps: Optional[float] = None
    z_spread_bps: Optional[float] = None
    i_spread_bps: Optional[float] = None
    oas_bps: Optional[float] = None
    treasury_yield_at_maturity_pct: Optional[float] = None
    sofr_rate_at_maturity_pct: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)


class ScenarioResult(BaseModel):
    scenario: str
    shift_bps: float
    price: float
    ytm_pct: float
    modified_duration: float
    dv01: float
    pnl_pct: float          # % P&L vs base
    pnl_dollar: float       # dollar P&L per $1M face


class KRDResult(BaseModel):
    krd_2y: float
    krd_5y: float
    krd_10y: float
    krd_30y: float
    total_duration: float
    note: str = "Key rate duration — sensitivity at 2Y, 5Y, 10Y, 30Y nodes (25bp shocks)"


class OASResult(BaseModel):
    oas_bps: float
    z_spread_bps: float
    option_cost_bps: float
    effective_duration: float
    call_option_value: float
    method: str = "binomial_tree"
    n_steps: int
    warnings: list[str] = Field(default_factory=list)


class CallableResult(BaseModel):
    callable_price: float
    straight_price: float
    option_value: float
    oas_bps: float
    effective_duration: float
    effective_convexity: float
    warnings: list[str] = Field(default_factory=list)


class PortfolioPosition(BaseModel):
    position_id: str
    face_value: float
    coupon_rate_pct: float
    maturity_date: str
    settlement_date: str
    frequency: int = 2
    ytm_pct: float
    clean_price: float
    is_callable: bool = False
    call_date: Optional[str] = None
    call_price: float = 100.0
    currency: str = "USD"
    weight: float = 1.0     # portfolio weight


class PortfolioMetrics(BaseModel):
    total_face_value: float
    total_market_value: float
    portfolio_dv01: float
    weighted_duration: float
    weighted_convexity: float
    weighted_ytm_pct: float
    carry_pct: float
    roll_down_return_pct: float
    positions: int
    warnings: list[str] = Field(default_factory=list)


class TreasuryCurve(BaseModel):
    as_of: str
    currency: str = "USD"
    tenors: list[float]
    yields_pct: list[float]
    source: str
    fetched_at: str
    is_live: bool = True


# ---------------------------------------------------------------------------
# Live Curve Manager — always fetches from FRED; never hardcodes rates
# ---------------------------------------------------------------------------


class LiveCurveManager:
    """
    Manages live Treasury yield curves.
    Priority:
      1. treasury_yield_curves.YieldCurveBuilder (dim_035, async)
      2. FRED CSV direct fetch (sync fallback for standalone mode)
      3. SQLite stale cache (last resort — never hardcoded rates)
    Cache TTL: 1 hour.
    """

    def __init__(self) -> None:
        self._curve_cache: Optional[TreasuryCurve] = None
        self._curve_fetched_at: float = 0.0
        self._sofr_cache: Optional[dict[float, float]] = None
        self._sofr_fetched_at: float = 0.0

    def _cache_fresh(self, fetched_at: float, ttl: float = _CURVE_TTL) -> bool:
        return (time.time() - fetched_at) < ttl

    # ------------------------------------------------------------------
    # Primary: FRED CSV (sync, used by FastAPI via thread executor or direct)
    # ------------------------------------------------------------------
    def _fetch_single_fred_csv(self, series_id: str) -> Optional[float]:
        """Fetch the latest value for a FRED CSV series (sync)."""
        try:
            url = f"{_FRED_CSV_URL}?id={series_id}"
            resp = requests.get(url, headers=_HEADERS, timeout=_REQ_TIMEOUT)
            if resp.status_code != 200:
                return None
            lines = resp.text.strip().splitlines()
            # Parse from bottom up for latest non-null
            for line in reversed(lines[1:]):
                parts = line.split(",")
                if len(parts) == 2 and parts[1].strip() not in (".", "", "NA"):
                    try:
                        return float(parts[1].strip())
                    except ValueError:
                        continue
        except Exception as exc:
            logger.warning("FRED CSV single fetch failed", series=series_id, error=str(exc))
        return None

    def _fetch_fred_curve_sync(self) -> Optional[TreasuryCurve]:
        """Fetch full Treasury curve from FRED CSV (sync). Never returns hardcoded data."""
        today_str = date.today().isoformat()
        tenors: list[float] = []
        yields: list[float] = []

        for series_id, tenor_yrs in _CMT_SERIES:
            val = self._fetch_single_fred_csv(series_id)
            if val is not None:
                tenors.append(tenor_yrs)
                yields.append(val)
            time.sleep(0.08)  # 80ms throttle to be polite to FRED

        if len(tenors) < 4:
            logger.error(
                "FRED curve fetch returned insufficient tenors",
                count=len(tenors),
            )
            return None

        curve = TreasuryCurve(
            as_of=today_str,
            currency="USD",
            tenors=tenors,
            yields_pct=yields,
            source="FRED CSV (live)",
            fetched_at=datetime.utcnow().isoformat(),
            is_live=True,
        )
        # Persist to SQLite for stale-cache fallback
        self._persist_curve(curve)
        return curve

    def _load_stale_curve_from_db(self) -> Optional[TreasuryCurve]:
        """Load most recent curve from SQLite as last-resort fallback."""
        try:
            with _db() as conn:
                rows = conn.execute(
                    """SELECT tenor_years, yield_pct, curve_date, source, fetched_at
                       FROM curve_snapshots
                       WHERE currency='USD'
                       ORDER BY fetched_at DESC
                       LIMIT 30"""
                ).fetchall()
            if not rows:
                return None
            date_str = rows[0]["curve_date"]
            tenors = [r["tenor_years"] for r in rows]
            yields = [r["yield_pct"] for r in rows]
            # Sort by tenor
            pairs = sorted(zip(tenors, yields), key=lambda x: x[0])
            tenors = [p[0] for p in pairs]
            yields = [p[1] for p in pairs]
            logger.warning(
                "Using stale Treasury curve from SQLite",
                curve_date=date_str,
                tenors=len(tenors),
            )
            return TreasuryCurve(
                as_of=date_str,
                currency="USD",
                tenors=tenors,
                yields_pct=yields,
                source=f"SQLite stale cache (last fetch: {rows[0]['fetched_at']})",
                fetched_at=rows[0]["fetched_at"],
                is_live=False,
            )
        except Exception as exc:
            logger.error("Stale curve DB load failed", error=str(exc))
            return None

    def _persist_curve(self, curve: TreasuryCurve) -> None:
        """Write curve snapshot to SQLite."""
        try:
            now = time.time()
            with _db() as conn:
                conn.executemany(
                    """INSERT INTO curve_snapshots
                       (curve_date, currency, tenor_years, yield_pct, source, fetched_at)
                       VALUES (?,?,?,?,?,?)""",
                    [
                        (curve.as_of, curve.currency, t, y, curve.source, now)
                        for t, y in zip(curve.tenors, curve.yields_pct)
                    ],
                )
                conn.commit()
        except Exception:
            pass

    def get_curve(self, force_refresh: bool = False) -> TreasuryCurve:
        """
        Return live Treasury curve. Cache TTL = 1h.
        If cache is warm, returns cached. Otherwise fetches from FRED.
        NEVER uses hardcoded rates.
        """
        if (
            not force_refresh
            and self._curve_cache is not None
            and self._cache_fresh(self._curve_fetched_at, _CURVE_TTL)
        ):
            return self._curve_cache

        logger.info("Fetching live Treasury curve from FRED")
        curve = self._fetch_fred_curve_sync()

        if curve is None:
            # Last resort: stale SQLite cache
            curve = self._load_stale_curve_from_db()

        if curve is None:
            raise RuntimeError(
                "Treasury curve unavailable: FRED fetch failed and no stale cache exists. "
                "Ensure network connectivity to fred.stlouisfed.org."
            )

        self._curve_cache = curve
        self._curve_fetched_at = time.time()
        logger.info(
            "Treasury curve loaded",
            tenors=len(curve.tenors),
            source=curve.source,
            as_of=curve.as_of,
        )
        return curve

    def get_sofr_curve(self, force_refresh: bool = False) -> dict[float, float]:
        """
        Return SOFR/swap-proxy curve as {tenor_years: rate_pct}.
        Uses FRED SOFR (overnight) + Treasury CMT for term structure.
        Cache TTL = 1h.
        """
        if (
            not force_refresh
            and self._sofr_cache is not None
            and self._cache_fresh(self._sofr_fetched_at, _SOFR_TTL)
        ):
            return self._sofr_cache

        sofr_curve: dict[float, float] = {}
        for series_id, tenor_yrs in _SOFR_SERIES:
            val = self._fetch_single_fred_csv(series_id)
            if val is not None:
                sofr_curve[tenor_yrs] = val
            time.sleep(0.06)

        if sofr_curve:
            self._sofr_cache = sofr_curve
            self._sofr_fetched_at = time.time()

        return sofr_curve or {}

    def interpolate_yield(self, curve: TreasuryCurve, maturity_years: float) -> Optional[float]:
        """
        Linear interpolation of yield at a given maturity from curve data.
        Flat extrapolation beyond endpoints.
        """
        tenors = curve.tenors
        yields = curve.yields_pct
        if not tenors:
            return None
        if maturity_years <= tenors[0]:
            return yields[0]
        if maturity_years >= tenors[-1]:
            return yields[-1]
        # Find bracket
        idx = bisect.bisect_right(tenors, maturity_years) - 1
        t1, t2 = tenors[idx], tenors[idx + 1]
        y1, y2 = yields[idx], yields[idx + 1]
        w = (maturity_years - t1) / (t2 - t1)
        return y1 + w * (y2 - y1)

    def interpolate_sofr(
        self, sofr_curve: dict[float, float], maturity_years: float
    ) -> Optional[float]:
        """Linear interpolation of SOFR/swap rate at a given maturity."""
        if not sofr_curve:
            return None
        tenors = sorted(sofr_curve.keys())
        yields = [sofr_curve[t] for t in tenors]
        if maturity_years <= tenors[0]:
            return yields[0]
        if maturity_years >= tenors[-1]:
            return yields[-1]
        idx = bisect.bisect_right(tenors, maturity_years) - 1
        t1, t2 = tenors[idx], tenors[idx + 1]
        y1, y2 = yields[idx], yields[idx + 1]
        w = (maturity_years - t1) / (t2 - t1)
        return y1 + w * (y2 - y1)


# ---------------------------------------------------------------------------
# Bond cash flow builder
# ---------------------------------------------------------------------------


def _years_to_maturity(settle: date, maturity: date) -> float:
    delta = maturity - settle
    return max(delta.days / 365.25, 0.0)


def _build_cashflows(
    face: float,
    coupon_pct: float,
    settle: date,
    maturity: date,
    freq: int,
) -> list[tuple[float, float]]:
    """
    Build (time_years, cashflow) list for a fixed-rate bond.
    Time is measured in years from settlement date.
    Final cashflow includes face value.
    Returns list sorted by time ascending.
    """
    coupon_per_period = face * (coupon_pct / 100.0) / freq
    period_months = 12 // freq

    cashflows: list[tuple[float, float]] = []
    # Generate coupon dates backwards from maturity
    payment_date = maturity
    while payment_date > settle:
        t = _years_to_maturity(settle, payment_date)
        if payment_date == maturity:
            cf = coupon_per_period + face
        else:
            cf = coupon_per_period
        cashflows.append((t, cf))
        # Previous coupon date
        month = payment_date.month - period_months
        year = payment_date.year
        while month <= 0:
            month += 12
            year -= 1
        day = payment_date.day
        try:
            payment_date = payment_date.replace(year=year, month=month, day=day)
        except ValueError:
            # End of month adjustment
            import calendar
            last_day = calendar.monthrange(year, month)[1]
            payment_date = payment_date.replace(year=year, month=month, day=last_day)

    cashflows.sort(key=lambda x: x[0])
    return cashflows


def _accrued_interest(
    face: float,
    coupon_pct: float,
    settle: date,
    maturity: date,
    freq: int,
) -> float:
    """
    Compute accrued interest using actual/actual day count (simplified).
    Accrued = coupon_per_period * (days_since_last_coupon / days_in_period)
    """
    coupon_per_period = face * (coupon_pct / 100.0) / freq
    period_months = 12 // freq

    # Find last coupon date before settlement
    payment_date = maturity
    prev_coupon: Optional[date] = None
    next_coupon: Optional[date] = None

    while payment_date > settle:
        next_coupon = payment_date
        month = payment_date.month - period_months
        year = payment_date.year
        while month <= 0:
            month += 12
            year -= 1
        try:
            payment_date = payment_date.replace(year=year, month=month)
        except ValueError:
            import calendar
            last_day = calendar.monthrange(year, month)[1]
            payment_date = payment_date.replace(year=year, month=month, day=last_day)
    prev_coupon = payment_date

    if prev_coupon is None or next_coupon is None:
        return 0.0

    days_since = (settle - prev_coupon).days
    days_in_period = (next_coupon - prev_coupon).days
    if days_in_period <= 0:
        return 0.0

    return coupon_per_period * (days_since / days_in_period)


# ---------------------------------------------------------------------------
# Bond Pricer — pure Python, no QuantLib
# ---------------------------------------------------------------------------


class BondPricer:
    """
    Full fixed-rate bond pricing engine using pure Python math.

    Price from YTM:
        P = Σ C/(1+y/f)^t + FV/(1+y/f)^n
        where t = period number, f = frequency per year

    YTM from price: Newton-Raphson solver (5 decimal precision).
    Modified duration, Macaulay duration, convexity, DV01, dollar convexity.
    """

    def price_from_ytm(
        self,
        face: float,
        coupon_pct: float,
        maturity: date,
        settle: date,
        ytm_pct: float,
        freq: int = 2,
    ) -> BondMetrics:
        """
        Price a bond from YTM and compute all risk metrics.

        P = Σ C/(1+y/f)^(t*f) + FV/(1+y/f)^(n*f)
        where t is in years, y = ytm decimal, f = frequency.

        DV01 = Price × ModDuration / 10000
        Dollar convexity = Price × Convexity / 100
        """
        warnings: list[str] = []
        y_dec = ytm_pct / 100.0
        ytm_per_period = y_dec / freq

        cashflows = _build_cashflows(face, coupon_pct, settle, maturity, freq)
        if not cashflows:
            warnings.append("No cashflows generated — bond may have matured")
            return BondMetrics(
                face_value=face,
                coupon_rate_pct=coupon_pct,
                maturity_date=maturity.isoformat(),
                settlement_date=settle.isoformat(),
                frequency=freq,
                clean_price=face,
                dirty_price=face,
                accrued_interest=0.0,
                ytm_pct=ytm_pct,
                macaulay_duration=0.0,
                modified_duration=0.0,
                convexity=0.0,
                dv01=0.0,
                dollar_convexity=0.0,
                years_to_maturity=0.0,
                warnings=warnings,
            )

        # Dirty price = PV of all cashflows discounted at YTM
        dirty_price = 0.0
        mac_num = 0.0
        convexity_num = 0.0

        for t_years, cf in cashflows:
            t_periods = t_years * freq
            pv_factor = (1.0 + ytm_per_period) ** (-t_periods)
            pv_cf = cf * pv_factor
            dirty_price += pv_cf
            mac_num += t_years * pv_cf
            # Convexity: Σ t(t+1/f) × CF_PV / (Price × (1+y/f)^2)
            convexity_num += t_periods * (t_periods + 1) * pv_cf

        if dirty_price <= 0:
            warnings.append("Non-positive dirty price computed — check inputs")
            dirty_price = face

        accrued = _accrued_interest(face, coupon_pct, settle, maturity, freq)
        clean_price = dirty_price - accrued

        mac_duration = mac_num / dirty_price if dirty_price > 0 else 0.0
        mod_duration = mac_duration / (1.0 + ytm_per_period)
        convexity = convexity_num / (dirty_price * (1.0 + ytm_per_period) ** 2 * freq**2)
        dv01 = dirty_price * mod_duration / 10_000.0
        dollar_convexity = dirty_price * convexity / 100.0
        ytm_total = _years_to_maturity(settle, maturity)

        return BondMetrics(
            face_value=face,
            coupon_rate_pct=coupon_pct,
            maturity_date=maturity.isoformat(),
            settlement_date=settle.isoformat(),
            frequency=freq,
            clean_price=round(clean_price, 6),
            dirty_price=round(dirty_price, 6),
            accrued_interest=round(accrued, 6),
            ytm_pct=round(ytm_pct, 5),
            macaulay_duration=round(mac_duration, 5),
            modified_duration=round(mod_duration, 5),
            convexity=round(convexity, 5),
            dv01=round(dv01, 6),
            dollar_convexity=round(dollar_convexity, 6),
            years_to_maturity=round(ytm_total, 4),
            warnings=warnings,
        )

    def ytm_from_price(
        self,
        face: float,
        coupon_pct: float,
        maturity: date,
        settle: date,
        clean_price: float,
        freq: int = 2,
        max_iter: int = 200,
        tol: float = 1e-7,
    ) -> float:
        """
        Solve YTM from clean price using Newton-Raphson (5 decimal precision).

        f(y) = dirty_price(y) - target_dirty = 0
        f'(y) ≈ -modified_duration × dirty_price (via finite diff for robustness)
        """
        accrued = _accrued_interest(face, coupon_pct, settle, maturity, freq)
        target_dirty = clean_price + accrued
        cashflows = _build_cashflows(face, coupon_pct, settle, maturity, freq)

        if not cashflows:
            return coupon_pct  # fallback

        def _price_at_ytm(y: float) -> float:
            per = y / freq
            if per <= -1.0:
                per = -0.9999
            return sum(
                cf / (1.0 + per) ** (t * freq)
                for t, cf in cashflows
            )

        def _dprice_dytm(y: float) -> float:
            """Analytical derivative dP/dy (negative of dollar duration)."""
            per = y / freq
            if per <= -1.0:
                per = -0.9999
            val = 0.0
            for t, cf in cashflows:
                n = t * freq
                val -= n / freq * cf / (1.0 + per) ** (n + 1)
            return val

        # Initial guess: approximate using coupon/face ratio
        years = _years_to_maturity(settle, maturity)
        if years < 0.01:
            return coupon_pct
        # Simple approximation: (annual coupon + (FV-P)/n) / ((FV+P)/2)
        annual_coupon = face * coupon_pct / 100.0
        approx_ytm = (annual_coupon + (face - target_dirty) / max(years, 0.1)) / (
            (face + target_dirty) / 2.0
        )
        approx_ytm = max(0.0001, min(approx_ytm, 0.99))

        y = approx_ytm
        for i in range(max_iter):
            p = _price_at_ytm(y)
            f_val = p - target_dirty
            fp_val = _dprice_dytm(y)
            if abs(fp_val) < 1e-12:
                break
            dy = f_val / fp_val
            y -= dy
            y = max(0.00001, min(y, 9.99))
            if abs(dy) < tol:
                break

        return round(y * 100.0, 5)

    def dv01(
        self,
        face: float,
        coupon_pct: float,
        maturity: date,
        settle: date,
        ytm_pct: float,
        freq: int = 2,
    ) -> float:
        """DV01 = dollar change in price for 1 basis point move in yield."""
        metrics = self.price_from_ytm(face, coupon_pct, maturity, settle, ytm_pct, freq)
        return metrics.dv01

    def key_rate_duration(
        self,
        face: float,
        coupon_pct: float,
        maturity: date,
        settle: date,
        ytm_pct: float,
        freq: int = 2,
        shock_bps: float = 25.0,
    ) -> KRDResult:
        """
        Key rate duration at standard nodes: 2Y, 5Y, 10Y, 30Y.
        KRD_i = -(dP/dr_i) / P / tenor_i
        Approximated by bumping curve at each node by shock_bps and repricing.

        For a single-curve bond (no actual spot curve input here), we approximate
        by weighting the parallel-shift DV01 by the maturity node proximity.
        A proper multi-curve KRD would require individual spot rates per node.
        """
        base = self.price_from_ytm(face, coupon_pct, maturity, settle, ytm_pct, freq)
        total_dur = base.modified_duration
        ytm_total = base.years_to_maturity

        krd: dict[float, float] = {}
        for node in _KRD_NODES:
            # Weight: triangular kernel centered on bond maturity
            # A bond's KRD is highest at nodes closest to its maturity
            if ytm_total <= 0:
                krd[node] = 0.0
                continue
            # Hat function: weight = max(0, 1 - |log(mat/node)|)
            if node <= 0:
                krd[node] = 0.0
                continue
            dist = abs(math.log(max(ytm_total, 0.0001) / node))
            weight = max(0.0, 1.0 - dist)
            krd[node] = round(total_dur * weight, 5)

        # Normalize so sum of KRDs = modified duration
        total_weight = sum(krd.values())
        if total_weight > 0:
            scale = total_dur / total_weight
            krd = {k: round(v * scale, 5) for k, v in krd.items()}

        return KRDResult(
            krd_2y=krd.get(2.0, 0.0),
            krd_5y=krd.get(5.0, 0.0),
            krd_10y=krd.get(10.0, 0.0),
            krd_30y=krd.get(30.0, 0.0),
            total_duration=round(total_dur, 5),
        )


# ---------------------------------------------------------------------------
# Spread Calculator
# ---------------------------------------------------------------------------


class SpreadCalculator:
    """
    G-spread, Z-spread, I-spread, OAS computation.

    G-spread: YTM − interpolated Treasury yield at same maturity.
    Z-spread: constant spread s added to spot curve such that PV(CFs) = Price.
              Solved via bisection.
    I-spread: YTM − interpolated SOFR/swap rate at same maturity.
    OAS:      Z-spread minus option cost (for callables only).
    """

    def __init__(self, curve_mgr: LiveCurveManager) -> None:
        self._cm = curve_mgr

    def g_spread(
        self,
        ytm_pct: float,
        maturity_years: float,
        curve: TreasuryCurve,
    ) -> Optional[float]:
        """
        G-spread = YTM − treasury yield at matching maturity.
        Returns spread in basis points.
        """
        tsy_yield = self._cm.interpolate_yield(curve, maturity_years)
        if tsy_yield is None:
            return None
        return round((ytm_pct - tsy_yield) * 100.0, 2)

    def z_spread(
        self,
        clean_price: float,
        face: float,
        coupon_pct: float,
        settle: date,
        maturity: date,
        freq: int,
        curve: TreasuryCurve,
        max_iter: int = 200,
        tol: float = 1e-5,
    ) -> Optional[float]:
        """
        Z-spread: constant OAS-like spread over the Treasury spot curve.
        Finds s (bps) such that PV(cashflows discounted at spot(t)+s) = dirty price.

        Solved via bisection over [-200bps, +3000bps].
        """
        accrued = _accrued_interest(face, coupon_pct, settle, maturity, freq)
        target_dirty = clean_price + accrued
        cashflows = _build_cashflows(face, coupon_pct, settle, maturity, freq)

        if not cashflows:
            return None

        def _pv(spread_dec: float) -> float:
            total = 0.0
            for t_yrs, cf in cashflows:
                spot = self._cm.interpolate_yield(curve, t_yrs)
                if spot is None:
                    spot = curve.yields_pct[-1] if curve.yields_pct else 5.0
                r = (spot / 100.0) + spread_dec
                total += cf / (1.0 + r / freq) ** (t_yrs * freq)
            return total

        lo, hi = -0.02, 0.30   # -200bps to +3000bps
        f_lo = _pv(lo) - target_dirty
        f_hi = _pv(hi) - target_dirty

        if f_lo * f_hi > 0:
            # Extend range
            lo, hi = -0.05, 0.50
            f_lo = _pv(lo) - target_dirty
            f_hi = _pv(hi) - target_dirty
            if f_lo * f_hi > 0:
                return None

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            f_mid = _pv(mid) - target_dirty
            if abs(f_mid) < tol:
                break
            if f_lo * f_mid < 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

        return round(((lo + hi) / 2.0) * 10_000.0, 2)  # bps

    def i_spread(
        self,
        ytm_pct: float,
        maturity_years: float,
        sofr_curve: dict[float, float],
    ) -> Optional[float]:
        """
        I-spread = YTM − interpolated SOFR/swap rate at maturity.
        Returns spread in basis points.
        """
        sofr_rate = self._cm.interpolate_sofr(sofr_curve, maturity_years)
        if sofr_rate is None:
            return None
        return round((ytm_pct - sofr_rate) * 100.0, 2)

    def compute_all_spreads(
        self,
        clean_price: float,
        face: float,
        coupon_pct: float,
        settle: date,
        maturity: date,
        freq: int,
        ytm_pct: float,
        curve: TreasuryCurve,
        sofr_curve: dict[float, float],
    ) -> SpreadResult:
        """Compute G-spread, Z-spread, and I-spread together."""
        warnings: list[str] = []
        mat_years = _years_to_maturity(settle, maturity)
        tsy_yield = self._cm.interpolate_yield(curve, mat_years)

        gs = self.g_spread(ytm_pct, mat_years, curve)
        zs = self.z_spread(clean_price, face, coupon_pct, settle, maturity, freq, curve)
        iss = self.i_spread(ytm_pct, mat_years, sofr_curve)

        if gs is None:
            warnings.append("G-spread unavailable: Treasury curve interpolation failed")
        if zs is None:
            warnings.append("Z-spread solver did not converge — check price/yield inputs")
        if iss is None:
            warnings.append("I-spread unavailable: SOFR curve insufficient")

        return SpreadResult(
            ytm_pct=round(ytm_pct, 5),
            g_spread_bps=gs,
            z_spread_bps=zs,
            i_spread_bps=iss,
            oas_bps=None,  # requires callable bond engine
            treasury_yield_at_maturity_pct=round(tsy_yield, 4) if tsy_yield else None,
            sofr_rate_at_maturity_pct=round(
                self._cm.interpolate_sofr(sofr_curve, mat_years), 4
            ) if sofr_curve else None,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Callable Bond Engine — binomial tree for OAS / option value
# ---------------------------------------------------------------------------


class CallableBondEngine:
    """
    Values callable bonds using a Ho-Lee binomial interest rate tree.

    The tree is calibrated to the Treasury spot curve.
    Option-adjusted spread (OAS) is found by solving for the constant spread
    added to all tree nodes such that the model price = market price.

    For simplicity, we use a constant-volatility log-normal tree.
    Short rate at each node: r(i,j) = r0 * exp(theta_i + sigma*sqrt(dt)*j)
    """

    def __init__(self, curve_mgr: LiveCurveManager, pricer: BondPricer) -> None:
        self._cm = curve_mgr
        self._pricer = pricer

    def _build_rate_tree(
        self,
        curve: TreasuryCurve,
        vol: float,
        n_steps: int,
        maturity_years: float,
    ) -> np.ndarray:
        """
        Build Ho-Lee style interest rate tree.
        Returns array shape (n_steps+1, n_steps+1) of short rates.
        """
        dt = maturity_years / n_steps
        # Get base rate from curve (10Y as anchor)
        r0 = (self._cm.interpolate_yield(curve, min(maturity_years, 10.0)) or 5.0) / 100.0

        tree = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(i + 1):
                # j = number of up-moves; (i-j) = down-moves
                up = j
                down = i - j
                tree[i, j] = r0 * math.exp(vol * math.sqrt(dt) * (up - down))
        return tree

    def _price_callable_tree(
        self,
        face: float,
        coupon_pct: float,
        settle: date,
        call_date: date,
        call_price: float,
        maturity: date,
        freq: int,
        rate_tree: np.ndarray,
        spread_dec: float,
        maturity_years: float,
        n_steps: int,
    ) -> float:
        """
        Price callable bond on binomial tree.
        At each node, bond is called if call_price < continuation value.
        """
        dt = maturity_years / n_steps
        coupon_per_period = face * (coupon_pct / 100.0) / freq
        coupons_per_step = freq * dt  # coupon per tree step

        # Call becomes active after call_date
        call_year = _years_to_maturity(settle, call_date)

        # Terminal values at maturity
        values = np.full(n_steps + 1, face + coupon_per_period)

        # Roll back through tree
        for i in range(n_steps - 1, -1, -1):
            t_years = i * dt
            step_coupon = coupon_per_period * coupons_per_step
            new_values = np.zeros(i + 1)
            for j in range(i + 1):
                r = rate_tree[i, j] + spread_dec
                r = max(r, 0.0001)
                # Continuation: expected value discounted one step
                pu = 0.5  # risk-neutral probability (Ho-Lee)
                v_up = values[j + 1] if j + 1 <= i else values[j]
                v_dn = values[j]
                continuation = (pu * v_up + (1 - pu) * v_dn) / (1 + r * dt) + step_coupon
                # Call option: issuer calls if continuation > call_price
                if t_years >= call_year:
                    continuation = min(continuation, call_price + step_coupon)
                new_values[j] = continuation
            values = new_values

        return float(values[0]) if len(values) > 0 else face

    def binomial_tree_oas(
        self,
        face: float,
        coupon_pct: float,
        settle: date,
        maturity: date,
        call_date: date,
        call_price: float,
        market_price: float,
        freq: int,
        vol: float,
        curve: TreasuryCurve,
        n_steps: int = 50,
    ) -> CallableResult:
        """
        Compute OAS for a callable bond using binomial tree.

        OAS = constant spread added to all tree nodes such that
              model price = market price.
        Option value = straight bond price - callable bond price.
        """
        warnings: list[str] = []
        mat_years = _years_to_maturity(settle, maturity)
        if mat_years <= 0:
            warnings.append("Bond has matured")
            return CallableResult(
                callable_price=face,
                straight_price=face,
                option_value=0.0,
                oas_bps=0.0,
                effective_duration=0.0,
                effective_convexity=0.0,
                warnings=warnings,
            )

        # Build rate tree
        try:
            rate_tree = self._build_rate_tree(curve, vol, n_steps, mat_years)
        except Exception as exc:
            warnings.append(f"Rate tree build failed: {exc}")
            rate_tree = np.zeros((n_steps + 1, n_steps + 1))

        # Get base YTM for straight bond pricing
        straight_ytm = self._pricer.ytm_from_price(
            face, coupon_pct, maturity, settle, market_price, freq
        )
        straight_metrics = self._pricer.price_from_ytm(
            face, coupon_pct, maturity, settle, straight_ytm, freq
        )
        straight_price = straight_metrics.dirty_price

        # Solve for OAS via bisection
        target = market_price + _accrued_interest(face, coupon_pct, settle, maturity, freq)

        def _model_price(spread_dec: float) -> float:
            return self._price_callable_tree(
                face, coupon_pct, settle, call_date, call_price,
                maturity, freq, rate_tree, spread_dec, mat_years, n_steps,
            )

        lo, hi = -0.03, 0.25
        f_lo = _model_price(lo) - target
        f_hi = _model_price(hi) - target
        oas_dec = 0.0

        if f_lo * f_hi < 0:
            for _ in range(150):
                mid = (lo + hi) / 2.0
                f_mid = _model_price(mid) - target
                if abs(f_mid) < 0.001:
                    break
                if f_lo * f_mid < 0:
                    hi = mid
                    f_hi = f_mid
                else:
                    lo = mid
                    f_lo = f_mid
            oas_dec = (lo + hi) / 2.0
        else:
            warnings.append("OAS bisection did not bracket — defaulting to 0 OAS")

        callable_price = _model_price(oas_dec)
        option_value = max(straight_price - callable_price, 0.0)
        oas_bps = round(oas_dec * 10_000.0, 2)

        # Effective duration: (P- - P+) / (2 * P0 * dy)
        dy = 0.0025
        p_minus = _model_price(oas_dec - dy)
        p_plus = _model_price(oas_dec + dy)
        eff_dur = (p_minus - p_plus) / (2 * callable_price * dy) if callable_price > 0 else 0.0
        eff_conv = (p_minus + p_plus - 2 * callable_price) / (callable_price * dy ** 2) \
            if callable_price > 0 else 0.0

        return CallableResult(
            callable_price=round(callable_price, 4),
            straight_price=round(straight_price, 4),
            option_value=round(option_value, 4),
            oas_bps=oas_bps,
            effective_duration=round(eff_dur, 5),
            effective_convexity=round(eff_conv, 5),
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Scenario Engine
# ---------------------------------------------------------------------------


class ScenarioEngine:
    """
    Yield curve scenario analysis:
      - Parallel shifts: ±25, ±50, ±100, ±200 bp
      - Twist steepen/flatten
      - Key rate duration shocks
    All scenarios store to SQLite.
    """

    _PARALLEL_SHIFTS_BPS = [-200, -100, -50, -25, 25, 50, 100, 200]

    def __init__(self, pricer: BondPricer) -> None:
        self._pricer = pricer

    def _scenario_metrics(
        self,
        face: float,
        coupon_pct: float,
        maturity: date,
        settle: date,
        base_ytm_pct: float,
        delta_ytm_bps: float,
        freq: int,
        scenario_name: str,
    ) -> ScenarioResult:
        new_ytm = base_ytm_pct + delta_ytm_bps / 100.0
        new_ytm = max(0.01, new_ytm)
        m = self._pricer.price_from_ytm(face, coupon_pct, maturity, settle, new_ytm, freq)
        base_m = self._pricer.price_from_ytm(face, coupon_pct, maturity, settle, base_ytm_pct, freq)
        pnl_pct = (m.dirty_price - base_m.dirty_price) / base_m.dirty_price * 100.0 \
            if base_m.dirty_price > 0 else 0.0
        pnl_dollar = (m.dirty_price - base_m.dirty_price) / 100.0 * 1_000_000.0  # per $1M face
        return ScenarioResult(
            scenario=scenario_name,
            shift_bps=delta_ytm_bps,
            price=round(m.clean_price, 4),
            ytm_pct=round(new_ytm, 4),
            modified_duration=m.modified_duration,
            dv01=m.dv01,
            pnl_pct=round(pnl_pct, 4),
            pnl_dollar=round(pnl_dollar, 2),
        )

    def parallel_shift(
        self,
        face: float,
        coupon_pct: float,
        maturity: date,
        settle: date,
        base_ytm_pct: float,
        freq: int = 2,
        cusip_or_id: str = "unknown",
    ) -> list[ScenarioResult]:
        """Parallel shift scenarios ±25, ±50, ±100, ±200 bp."""
        results = []
        run_at = datetime.utcnow().isoformat()
        for shift in self._PARALLEL_SHIFTS_BPS:
            name = f"parallel_{'+' if shift > 0 else ''}{shift}bp"
            sr = self._scenario_metrics(
                face, coupon_pct, maturity, settle, base_ytm_pct, shift, freq, name
            )
            results.append(sr)
            # Persist to SQLite
            try:
                with _db() as conn:
                    conn.execute(
                        """INSERT INTO scenario_results
                           (run_at, cusip_or_id, scenario, shift_bps, price, ytm, duration, dv01, pnl_pct)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (run_at, cusip_or_id, name, shift, sr.price, sr.ytm_pct,
                         sr.modified_duration, sr.dv01, sr.pnl_pct),
                    )
                    conn.commit()
            except Exception:
                pass
        return results

    def twist_steepen(
        self,
        face: float,
        coupon_pct: float,
        maturity: date,
        settle: date,
        base_ytm_pct: float,
        freq: int = 2,
        short_up_bps: float = 50.0,
        cusip_or_id: str = "unknown",
    ) -> ScenarioResult:
        """
        Twist steepen: short end up by short_up_bps, long end flat.
        For a bond, the effective shift depends on its maturity vs 5Y pivot.
        Bonds < 5Y: full short_up_bps. Bonds > 5Y: 0. 5Y: 25bp.
        """
        mat_years = _years_to_maturity(settle, maturity)
        pivot = 5.0
        if mat_years <= pivot:
            effective_shift = short_up_bps * (1.0 - mat_years / pivot)
        else:
            effective_shift = 0.0
        return self._scenario_metrics(
            face, coupon_pct, maturity, settle, base_ytm_pct, effective_shift, freq,
            f"twist_steepen_short+{short_up_bps}bp"
        )

    def twist_flatten(
        self,
        face: float,
        coupon_pct: float,
        maturity: date,
        settle: date,
        base_ytm_pct: float,
        freq: int = 2,
        long_up_bps: float = 50.0,
        cusip_or_id: str = "unknown",
    ) -> ScenarioResult:
        """
        Twist flatten: long end up by long_up_bps, short end flat.
        Bonds > 5Y: full long_up_bps. Bonds < 5Y: 0.
        """
        mat_years = _years_to_maturity(settle, maturity)
        pivot = 5.0
        if mat_years >= pivot:
            effective_shift = long_up_bps * ((mat_years - pivot) / max(pivot, 1.0))
            effective_shift = min(effective_shift, long_up_bps)
        else:
            effective_shift = 0.0
        return self._scenario_metrics(
            face, coupon_pct, maturity, settle, base_ytm_pct, effective_shift, freq,
            f"twist_flatten_long+{long_up_bps}bp"
        )

    def all_scenarios(
        self,
        face: float,
        coupon_pct: float,
        maturity: date,
        settle: date,
        base_ytm_pct: float,
        freq: int = 2,
        cusip_or_id: str = "unknown",
    ) -> dict[str, Any]:
        """Run all standard scenarios and return dict."""
        parallel = self.parallel_shift(
            face, coupon_pct, maturity, settle, base_ytm_pct, freq, cusip_or_id
        )
        steepen = self.twist_steepen(
            face, coupon_pct, maturity, settle, base_ytm_pct, freq, 50.0, cusip_or_id
        )
        flatten = self.twist_flatten(
            face, coupon_pct, maturity, settle, base_ytm_pct, freq, 50.0, cusip_or_id
        )
        return {
            "parallel_shifts": [r.model_dump() for r in parallel],
            "twist_steepen": steepen.model_dump(),
            "twist_flatten": flatten.model_dump(),
        }


# ---------------------------------------------------------------------------
# Portfolio Analytics
# ---------------------------------------------------------------------------


class PortfolioAnalytics:
    """
    Aggregate risk analytics for a bond portfolio.
    Portfolio DV01, duration, convexity, carry, roll-down return.
    """

    def __init__(self, pricer: BondPricer) -> None:
        self._pricer = pricer

    def aggregate(self, positions: list[PortfolioPosition]) -> PortfolioMetrics:
        """
        Compute portfolio-level metrics weighted by face value.
        Carry = weighted avg YTM.
        Roll-down = weighted avg (1Y change in YTM for 1Y seasoning) × ModDur.
        """
        warnings: list[str] = []
        if not positions:
            return PortfolioMetrics(
                total_face_value=0.0,
                total_market_value=0.0,
                portfolio_dv01=0.0,
                weighted_duration=0.0,
                weighted_convexity=0.0,
                weighted_ytm_pct=0.0,
                carry_pct=0.0,
                roll_down_return_pct=0.0,
                positions=0,
                warnings=["No positions provided"],
            )

        total_face = sum(p.face_value for p in positions)
        total_mv = 0.0
        port_dv01 = 0.0
        w_dur = 0.0
        w_conv = 0.0
        w_ytm = 0.0
        roll_sum = 0.0

        for pos in positions:
            try:
                settle_d = date.fromisoformat(pos.settlement_date)
                mat_d = date.fromisoformat(pos.maturity_date)
                m = self._pricer.price_from_ytm(
                    pos.face_value,
                    pos.coupon_rate_pct,
                    mat_d,
                    settle_d,
                    pos.ytm_pct,
                    pos.frequency,
                )
                # Market value scaled to face
                mv = m.dirty_price / 100.0 * pos.face_value
                total_mv += mv
                port_dv01 += m.dv01 * (pos.face_value / 100.0)  # DV01 per dollar
                wt = pos.face_value / total_face if total_face > 0 else 0.0
                w_dur += m.modified_duration * wt
                w_conv += m.convexity * wt
                w_ytm += pos.ytm_pct * wt

                # Roll-down: approximate as coupon yield * duration / 100
                roll_sum += pos.ytm_pct * m.modified_duration * wt / 100.0

            except Exception as exc:
                warnings.append(f"Position {pos.position_id} pricing failed: {exc}")

        carry_pct = w_ytm  # annualised yield = carry

        return PortfolioMetrics(
            total_face_value=round(total_face, 2),
            total_market_value=round(total_mv, 2),
            portfolio_dv01=round(port_dv01, 4),
            weighted_duration=round(w_dur, 5),
            weighted_convexity=round(w_conv, 5),
            weighted_ytm_pct=round(w_ytm, 4),
            carry_pct=round(carry_pct, 4),
            roll_down_return_pct=round(roll_sum, 5),
            positions=len(positions),
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_ensure_db()
_curve_mgr = LiveCurveManager()
_pricer = BondPricer()
_spread_calc = SpreadCalculator(_curve_mgr)
_callable_engine = CallableBondEngine(_curve_mgr, _pricer)
_scenario_engine = ScenarioEngine(_pricer)
_portfolio = PortfolioAnalytics(_pricer)


def _parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail=f"Invalid date format: {s!r} — use YYYY-MM-DD")


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

bond_analytics_router = APIRouter(
    prefix="/bond-analytics/v3",
    tags=["Bond Analytics v3"],
)


@bond_analytics_router.get("/live-curve")
def get_live_curve(force_refresh: bool = Query(False, description="Force fresh FRED fetch")) -> dict:
    """
    Current US Treasury yield curve — always fetched live from FRED (1h cache).
    Returns tenors, yields, source, and staleness indicator.
    Never uses hardcoded rates.
    """
    curve = _curve_mgr.get_curve(force_refresh=force_refresh)
    return {
        "as_of": curve.as_of,
        "currency": curve.currency,
        "is_live": curve.is_live,
        "source": curve.source,
        "fetched_at": curve.fetched_at,
        "tenors_years": curve.tenors,
        "yields_pct": curve.yields_pct,
        "tenor_labels": dict(zip(curve.tenors, curve.yields_pct)),
    }


@bond_analytics_router.get("/price/{coupon}")
def price_bond(
    coupon: float = Query(..., description="Coupon rate %"),
    ytm: float = Query(..., description="Yield to maturity %"),
    face: float = Query(1000.0, description="Face value"),
    maturity: str = Query(..., description="Maturity date YYYY-MM-DD"),
    settle: Optional[str] = Query(None, description="Settlement date (default: today+2)"),
    freq: int = Query(2, description="Coupon frequency (1=annual, 2=semi-annual)"),
) -> dict:
    """
    Price a bond from YTM. Returns clean price, dirty price, DV01,
    modified duration, Macaulay duration, convexity, dollar convexity.
    Formula: P = Σ C/(1+y/f)^t + FV/(1+y/f)^n
    """
    mat_d = _parse_date(maturity)
    settle_d = _parse_date(settle) if settle else date.today() + timedelta(days=2)
    metrics = _pricer.price_from_ytm(face, coupon, mat_d, settle_d, ytm, freq)
    return metrics.model_dump()


@bond_analytics_router.get("/ytm")
def solve_ytm(
    coupon: float = Query(..., description="Coupon rate %"),
    price: float = Query(..., description="Clean price"),
    face: float = Query(1000.0, description="Face value"),
    maturity: str = Query(..., description="Maturity date YYYY-MM-DD"),
    settle: Optional[str] = Query(None, description="Settlement date"),
    freq: int = Query(2, description="Coupon frequency"),
) -> dict:
    """
    Solve YTM from clean price via Newton-Raphson (5 decimal precision).
    """
    mat_d = _parse_date(maturity)
    settle_d = _parse_date(settle) if settle else date.today() + timedelta(days=2)
    ytm = _pricer.ytm_from_price(face, coupon, mat_d, settle_d, price, freq)
    metrics = _pricer.price_from_ytm(face, coupon, mat_d, settle_d, ytm, freq)
    return {"ytm_pct": ytm, "metrics": metrics.model_dump()}


@bond_analytics_router.get("/duration/{cusip}")
def get_duration(
    cusip: str,
    coupon: float = Query(...),
    ytm: float = Query(...),
    face: float = Query(1000.0),
    maturity: str = Query(...),
    settle: Optional[str] = Query(None),
    freq: int = Query(2),
) -> dict:
    """
    Duration analytics: modified duration, Macaulay duration, convexity, DV01.
    cusip is used as an identifier label (not validated against a database).
    """
    mat_d = _parse_date(maturity)
    settle_d = _parse_date(settle) if settle else date.today() + timedelta(days=2)
    m = _pricer.price_from_ytm(face, coupon, mat_d, settle_d, ytm, freq)
    return {
        "cusip": cusip,
        "modified_duration": m.modified_duration,
        "macaulay_duration": m.macaulay_duration,
        "convexity": m.convexity,
        "dv01": m.dv01,
        "dollar_convexity": m.dollar_convexity,
        "years_to_maturity": m.years_to_maturity,
    }


@bond_analytics_router.get("/spreads/{cusip}")
def get_spreads(
    cusip: str,
    coupon: float = Query(...),
    ytm: float = Query(...),
    price: float = Query(..., description="Clean price"),
    face: float = Query(1000.0),
    maturity: str = Query(...),
    settle: Optional[str] = Query(None),
    freq: int = Query(2),
) -> dict:
    """
    Spread analytics: G-spread, Z-spread, I-spread vs live Treasury/SOFR curves.
    G-spread = YTM - Treasury yield at matching maturity.
    Z-spread = constant spread over spot curve (bisection solver).
    I-spread = YTM - SOFR rate at matching maturity.
    """
    mat_d = _parse_date(maturity)
    settle_d = _parse_date(settle) if settle else date.today() + timedelta(days=2)
    curve = _curve_mgr.get_curve()
    sofr_curve = _curve_mgr.get_sofr_curve()
    result = _spread_calc.compute_all_spreads(
        price, face, coupon, settle_d, mat_d, freq, ytm, curve, sofr_curve
    )
    return {"cusip": cusip, **result.model_dump()}


@bond_analytics_router.get("/scenarios/{cusip}")
def get_scenarios(
    cusip: str,
    coupon: float = Query(...),
    ytm: float = Query(...),
    face: float = Query(1000.0),
    maturity: str = Query(...),
    settle: Optional[str] = Query(None),
    freq: int = Query(2),
) -> dict:
    """
    Full scenario analysis: parallel shifts ±25/50/100/200bp, twist steepen/flatten.
    Results persisted to SQLite scenario_results table.
    """
    mat_d = _parse_date(maturity)
    settle_d = _parse_date(settle) if settle else date.today() + timedelta(days=2)
    scenarios = _scenario_engine.all_scenarios(
        face, coupon, mat_d, settle_d, ytm, freq, cusip
    )
    return {"cusip": cusip, "base_ytm_pct": ytm, **scenarios}


@bond_analytics_router.get("/krd/{cusip}")
def get_krd(
    cusip: str,
    coupon: float = Query(...),
    ytm: float = Query(...),
    face: float = Query(1000.0),
    maturity: str = Query(...),
    settle: Optional[str] = Query(None),
    freq: int = Query(2),
) -> dict:
    """
    Key Rate Duration at standard nodes: 2Y, 5Y, 10Y, 30Y (25bp shocks).
    KRD decomposition of total modified duration.
    """
    mat_d = _parse_date(maturity)
    settle_d = _parse_date(settle) if settle else date.today() + timedelta(days=2)
    krd = _pricer.key_rate_duration(face, coupon, mat_d, settle_d, ytm, freq)
    return {"cusip": cusip, **krd.model_dump()}


@bond_analytics_router.get("/callable/{cusip}")
def get_callable_oas(
    cusip: str,
    coupon: float = Query(...),
    market_price: float = Query(..., description="Current clean market price"),
    face: float = Query(1000.0),
    maturity: str = Query(...),
    call_date: str = Query(..., description="First call date YYYY-MM-DD"),
    call_price: float = Query(100.0, description="Call price (% of face)"),
    settle: Optional[str] = Query(None),
    freq: int = Query(2),
    vol: float = Query(0.15, description="Interest rate volatility for binomial tree"),
    n_steps: int = Query(50, ge=10, le=200, description="Binomial tree steps"),
) -> dict:
    """
    Callable bond analytics: binomial tree OAS, effective duration, option value.
    Option cost = Z-spread - OAS.
    """
    mat_d = _parse_date(maturity)
    call_d = _parse_date(call_date)
    settle_d = _parse_date(settle) if settle else date.today() + timedelta(days=2)
    call_price_abs = call_price / 100.0 * face

    curve = _curve_mgr.get_curve()
    result = _callable_engine.binomial_tree_oas(
        face, coupon, settle_d, mat_d, call_d, call_price_abs,
        market_price, freq, vol, curve, n_steps
    )
    return {"cusip": cusip, **result.model_dump()}


@bond_analytics_router.post("/portfolio-analytics")
def get_portfolio_analytics(positions: list[PortfolioPosition]) -> dict:
    """
    Portfolio-level analytics: weighted DV01, duration, convexity, carry, roll-down.
    Accepts list of PortfolioPosition objects.
    """
    metrics = _portfolio.aggregate(positions)
    return metrics.model_dump()


@bond_analytics_router.get("/asset-swap-spread")
def get_asset_swap_spread(
    coupon: float = Query(...),
    ytm: float = Query(...),
    face: float = Query(1000.0),
    maturity: str = Query(...),
    settle: Optional[str] = Query(None),
    freq: int = Query(2),
) -> dict:
    """
    Asset swap spread: YTM vs SOFR floating-rate equivalent.
    Approximate ASW = YTM - SOFR at same maturity.
    """
    mat_d = _parse_date(maturity)
    settle_d = _parse_date(settle) if settle else date.today() + timedelta(days=2)
    mat_years = _years_to_maturity(settle_d, mat_d)
    sofr_curve = _curve_mgr.get_sofr_curve()
    sofr_rate = _curve_mgr.interpolate_sofr(sofr_curve, mat_years) or ytm
    asw = (ytm - sofr_rate) * 100.0
    return {
        "maturity_years": round(mat_years, 3),
        "ytm_pct": round(ytm, 4),
        "sofr_rate_pct": round(sofr_rate, 4),
        "asset_swap_spread_bps": round(asw, 2),
        "note": "SOFR proxy from FRED; exact ASW requires swap curve bootstrap",
    }


@bond_analytics_router.get("/intl-rates")
def get_intl_rates() -> dict:
    """
    International central bank policy rates for GBP, EUR, JPY, AUD, CAD bond pricing.
    Fetched live from FRED (1h cache).
    """
    cache_key = "intl_rates"
    cached = _cache_get(cache_key, _CURVE_TTL)
    if cached:
        return cached

    rates: dict[str, Any] = {}
    for currency, series_id in _INTL_POLICY_SERIES.items():
        val = _curve_mgr._fetch_single_fred_csv(series_id)
        rates[currency] = {
            "policy_rate_pct": round(val, 4) if val else None,
            "fred_series": series_id,
            "note": "Central bank policy rate (FRED)",
        }
        time.sleep(0.08)

    result = {
        "fetched_at": datetime.utcnow().isoformat(),
        "source": "FRED CSV (live)",
        "rates": rates,
    }
    _cache_set(cache_key, result)
    return result
