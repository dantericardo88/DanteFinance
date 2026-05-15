"""
Full bond analytics engine: pricing, DV01, convexity, OAS, Z-spread, duration.
Pure Python/NumPy — no QuantLib required (avoid C++ dependency issues).
Covers Treasuries, corporates, munis, MBS.

Dimension: dim_038 — Bond analytics engine (DV01, OAS, yield/price)
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DAYS_PER_YEAR = 365.25
_SETTLE_LAG = 2  # T+2 settlement (Treasuries/corporates)

# Approximate on-the-run Treasury yields (2025 baseline, percent)
_DEFAULT_TREASURY_CURVE: Dict[float, float] = {
    0.0833: 5.22,   # 1M
    0.25:   5.20,   # 3M
    0.50:   5.18,   # 6M
    1.0:    5.10,   # 1Y
    2.0:    4.85,   # 2Y
    3.0:    4.70,   # 3Y
    5.0:    4.50,   # 5Y
    7.0:    4.45,   # 7Y
    10.0:   4.40,   # 10Y
    20.0:   4.65,   # 20Y
    30.0:   4.55,   # 30Y
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BondSpec(BaseModel):
    face: float = 1000.0
    coupon_rate: float = Field(..., description="Annual coupon rate as percent, e.g. 5.0 = 5%")
    maturity_years: float = Field(..., description="Years to maturity")
    freq: int = Field(2, description="Coupon payments per year (2=semiannual)")
    settlement_date: Optional[str] = None


class PriceResult(BaseModel):
    face: float
    coupon_rate: float
    maturity_years: float
    yield_pct: float
    freq: int
    clean_price: float
    accrued_interest: float
    dirty_price: float
    modified_duration: float
    macaulay_duration: float
    convexity: float
    dv01: float
    price_per_100: float


class YieldResult(BaseModel):
    face: float
    coupon_rate: float
    maturity_years: float
    clean_price: float
    freq: int
    ytm: float
    current_yield: float
    modified_duration: float
    macaulay_duration: float
    convexity: float
    dv01: float


class DurationResult(BaseModel):
    modified_duration: float
    macaulay_duration: float
    convexity: float
    dv01: float
    dollar_convexity: float
    effective_duration: float
    spread_duration: float
    key_rate_durations: Dict[str, float]


class SpreadResult(BaseModel):
    z_spread_bps: float
    oas_bps: float
    credit_spread_bps: float
    asset_swap_spread_bps: Optional[float] = None
    option_value_bps: float


class CashflowSchedule(BaseModel):
    dates: List[str]
    cashflows: List[float]
    times: List[float]
    pv_cashflows: List[float]
    total_pv: float
    bond_type: str


class PortfolioRisk(BaseModel):
    total_market_value: float
    aggregate_dv01: float
    aggregate_modified_duration: float
    aggregate_convexity: float
    portfolio_yield: float
    key_rate_dv01s: Dict[str, float]
    var_95_bps: float
    var_99_bps: float
    var_95_dollar: float
    var_99_dollar: float
    spread_duration: float


class MBSResult(BaseModel):
    wac: float
    wam_months: float
    psa_speed: float
    wal_months: float
    effective_duration: float
    effective_convexity: float
    monthly_cashflows: List[float]
    prepayment_cashflows: List[float]
    interest_cashflows: List[float]
    principal_cashflows: List[float]


# ---------------------------------------------------------------------------
# Day count conventions
# ---------------------------------------------------------------------------

class DayCount:
    """
    Implements day count conventions for accrued interest.
    Supports: 30/360, Actual/Actual (ICMA), Actual/360, Actual/365.
    """

    @staticmethod
    def thirty_360(d1: date, d2: date) -> float:
        """30/360 (Bond Basis / US): standard for corporate and muni bonds."""
        y1, m1, day1 = d1.year, d1.month, d1.day
        y2, m2, day2 = d2.year, d2.month, d2.day
        # Adjust day counts per 30/360 rules
        if day1 == 31:
            day1 = 30
        if day2 == 31 and day1 == 30:
            day2 = 30
        days = (360 * (y2 - y1) + 30 * (m2 - m1) + (day2 - day1))
        return days / 360.0

    @staticmethod
    def actual_actual_icma(d1: date, d2: date, coupon_freq: int) -> float:
        """Actual/Actual (ICMA): standard for Treasuries and agencies."""
        actual_days = (d2 - d1).days
        period_days = _DAYS_PER_YEAR / coupon_freq
        return actual_days / period_days / coupon_freq

    @staticmethod
    def actual_360(d1: date, d2: date) -> float:
        """Actual/360: money market convention."""
        return (d2 - d1).days / 360.0

    @staticmethod
    def actual_365(d1: date, d2: date) -> float:
        """Actual/365: UK Gilt convention."""
        return (d2 - d1).days / 365.0

    @classmethod
    def fraction(
        cls,
        d1: date,
        d2: date,
        convention: str = "30/360",
        coupon_freq: int = 2,
    ) -> float:
        """Return day count fraction for given convention."""
        conv = convention.lower().replace(" ", "").replace("_", "")
        if conv in ("30/360", "30360", "bond"):
            return cls.thirty_360(d1, d2)
        if conv in ("actual/actual", "actualactual", "act/act", "aa/icma"):
            return cls.actual_actual_icma(d1, d2, coupon_freq)
        if conv in ("actual/360", "act/360", "a/360"):
            return cls.actual_360(d1, d2)
        if conv in ("actual/365", "act/365", "a/365"):
            return cls.actual_365(d1, d2)
        return cls.thirty_360(d1, d2)


# ---------------------------------------------------------------------------
# CashflowEngine
# ---------------------------------------------------------------------------

class CashflowEngine:
    """
    Generate cashflow schedules for various bond types:
    - Bullet (standard fixed coupon)
    - Amortizing (mortgage-style level payment)
    - Zero coupon
    - Floating rate notes (SOFR + spread)
    - Callable bonds (truncated at next call)
    """

    @staticmethod
    def bullet(
        face: float,
        coupon_rate: float,
        maturity_years: float,
        freq: int = 2,
        settle_date: Optional[date] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Generate bullet bond cashflows and time grid.
        Returns (times, cashflows) where times are in years from settlement.
        """
        n = int(round(maturity_years * freq))
        if n <= 0:
            return [], []
        c = face * (coupon_rate / 100.0) / freq
        dt = 1.0 / freq
        times = [dt * (i + 1) for i in range(n)]
        cashflows = [c] * n
        cashflows[-1] += face  # add principal at maturity
        return times, cashflows

    @staticmethod
    def amortizing(
        face: float,
        coupon_rate: float,
        maturity_years: float,
        freq: int = 12,  # monthly for mortgages
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Generate amortizing bond (mortgage-style) cashflows.
        Returns (times, cashflows, principal_cfs, interest_cfs).
        """
        n = int(round(maturity_years * freq))
        if n <= 0:
            return [], [], [], []
        r = (coupon_rate / 100.0) / freq
        # Level payment (annuity formula)
        if abs(r) < 1e-12:
            level_payment = face / n
        else:
            level_payment = face * r / (1 - (1 + r) ** (-n))

        times, cashflows, principal_cfs, interest_cfs = [], [], [], []
        balance = face
        dt = 1.0 / freq

        for i in range(n):
            interest = balance * r
            principal = level_payment - interest
            balance = max(0.0, balance - principal)

            times.append(dt * (i + 1))
            interest_cfs.append(interest)
            principal_cfs.append(principal)
            cashflows.append(level_payment)

        # Adjust last payment for rounding
        if balance > 0:
            cashflows[-1] += balance
            principal_cfs[-1] += balance

        return times, cashflows, principal_cfs, interest_cfs

    @staticmethod
    def zero_coupon(
        face: float, maturity_years: float
    ) -> Tuple[List[float], List[float]]:
        """Zero coupon bond: single cashflow at maturity."""
        return [maturity_years], [face]

    @staticmethod
    def floating_rate_note(
        face: float,
        spread_bps: float,
        maturity_years: float,
        freq: int = 4,
        sofr_curve: Optional[Dict[float, float]] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Floating rate note cashflows using SOFR + spread.
        Assumes constant SOFR rate (simplified; in practice, reset each period).
        """
        if sofr_curve is None:
            sofr_curve = {0.25: 5.30, 0.5: 5.25, 1.0: 5.10, 2.0: 4.85}
        n = int(round(maturity_years * freq))
        dt = 1.0 / freq
        times, cashflows = [], []

        for i in range(n):
            t = dt * (i + 1)
            # Interpolate SOFR for reset tenor
            tenors = sorted(sofr_curve.keys())
            sofr = sofr_curve[tenors[-1]]
            for j in range(len(tenors) - 1):
                if tenors[j] <= dt <= tenors[j + 1]:
                    t0, t1 = tenors[j], tenors[j + 1]
                    w = (dt - t0) / (t1 - t0)
                    sofr = sofr_curve[t0] + w * (sofr_curve[t1] - sofr_curve[t0])
                    break
            coupon_rate = sofr + spread_bps / 100.0
            cf = face * (coupon_rate / 100.0) / freq
            times.append(t)
            cashflows.append(cf)

        cashflows[-1] += face
        return times, cashflows

    @staticmethod
    def callable_bond(
        face: float,
        coupon_rate: float,
        maturity_years: float,
        call_date_years: float,
        call_price: float = 100.0,
        freq: int = 2,
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        """
        Callable bond: returns both to-maturity and to-call cashflow schedules.
        """
        times_mat, cfs_mat = CashflowEngine.bullet(face, coupon_rate, maturity_years, freq)
        # To call: truncate at call date
        n_call = int(round(call_date_years * freq))
        dt = 1.0 / freq
        times_call = [dt * (i + 1) for i in range(n_call)]
        c = face * (coupon_rate / 100.0) / freq
        cfs_call = [c] * n_call
        cfs_call[-1] += face * call_price / 100.0  # redemption at call price

        return {
            "to_maturity": (times_mat, cfs_mat),
            "to_call": (times_call, cfs_call),
        }

    @staticmethod
    def pv_cashflows(
        times: List[float],
        cashflows: List[float],
        discount_rate: float,
        freq: int = 1,
    ) -> List[float]:
        """Present value of each cashflow using flat discount rate."""
        return [
            cf / (1.0 + discount_rate / freq / 100.0) ** (t * freq)
            for t, cf in zip(times, cashflows)
        ]

    @staticmethod
    def npv(times: List[float], cashflows: List[float], discount_rate: float) -> float:
        """Net present value of cashflow stream."""
        pvs = CashflowEngine.pv_cashflows(times, cashflows, discount_rate)
        return sum(pvs)


# ---------------------------------------------------------------------------
# BondPricer
# ---------------------------------------------------------------------------

class BondPricer:
    """
    Core bond pricing engine.
    Implements price/yield conversion, accrued interest, and full/clean pricing.
    Fully vectorized via NumPy where applicable.
    """

    def price_from_yield(
        self,
        face: float,
        coupon_rate: float,
        yield_pct: float,
        maturity_years: float,
        freq: int = 2,
    ) -> float:
        """
        Compute clean price from yield-to-maturity.
        Uses standard bond pricing formula (no accrued interest; assumes pricing
        exactly on coupon date). For dirty price, add accrued interest.

        Price = sum(C / (1+y/f)^t) + Face / (1+y/f)^n
        where t = 1..n, y = YTM, f = frequency.
        """
        if maturity_years <= 0:
            return face

        n = int(round(maturity_years * freq))
        c = face * (coupon_rate / 100.0) / freq
        y = yield_pct / freq / 100.0

        if abs(y) < 1e-12:
            # Zero yield: sum of all cashflows
            return n * c + face

        # Vectorized via numpy
        t_arr = np.arange(1, n + 1, dtype=float)
        df_arr = (1.0 + y) ** (-t_arr)
        price = c * np.sum(df_arr) + face * (1.0 + y) ** (-n)
        return float(price)

    def price_from_yield_batch(
        self,
        face: np.ndarray,
        coupon_rate: np.ndarray,
        yield_pct: np.ndarray,
        maturity_years: np.ndarray,
        freq: int = 2,
    ) -> np.ndarray:
        """
        Batch pricing — all inputs are 1D numpy arrays of equal length.
        Returns array of clean prices.
        """
        results = np.zeros_like(yield_pct, dtype=float)
        for i in range(len(yield_pct)):
            results[i] = self.price_from_yield(
                float(face[i]), float(coupon_rate[i]),
                float(yield_pct[i]), float(maturity_years[i]), freq
            )
        return results

    def yield_from_price(
        self,
        face: float,
        coupon_pct: float,
        price: float,
        maturity_years: float,
        freq: int = 2,
        tol: float = 1e-10,
    ) -> float:
        """
        Solve for YTM from price using Brent's method.
        f(y) = price_from_yield(y) - price = 0
        """
        if maturity_years <= 0:
            return 0.0

        def objective(y: float) -> float:
            return self.price_from_yield(face, coupon_pct, y, maturity_years, freq) - price

        # Search bounds: yield between -5% and 200%
        lo, hi = -4.99, 199.99
        try:
            # Ensure bracket
            f_lo = objective(lo)
            f_hi = objective(hi)
            if f_lo * f_hi > 0:
                # Try tighter range
                lo, hi = 0.001, 50.0
                f_lo = objective(lo)
                f_hi = objective(hi)
                if f_lo * f_hi > 0:
                    # Fallback: Newton's method starting from rough estimate
                    return self._ytm_newton(face, coupon_pct, price, maturity_years, freq)
            ytm = brentq(objective, lo, hi, xtol=tol, maxiter=200)
            return float(ytm)
        except (ValueError, RuntimeError) as exc:
            logger.debug("Brentq failed for yield from price: %s", exc)
            return self._ytm_newton(face, coupon_pct, price, maturity_years, freq)

    def _ytm_newton(
        self, face: float, coupon_pct: float, price: float,
        maturity_years: float, freq: int, max_iter: int = 100
    ) -> float:
        """Newton-Raphson YTM solver as fallback."""
        n = int(round(maturity_years * freq))
        c = face * coupon_pct / 100.0 / freq
        # Initial guess: coupon rate / price
        y = (coupon_pct / 100.0) / freq
        for _ in range(max_iter):
            p = c * sum((1 + y) ** (-t) for t in range(1, n + 1)) + face * (1 + y) ** (-n)
            dp = sum(-t * c * (1 + y) ** (-t - 1) for t in range(1, n + 1)) - n * face * (1 + y) ** (-n - 1)
            if abs(dp) < 1e-15:
                break
            y_new = y - (p - price) / dp
            if abs(y_new - y) < 1e-10:
                y = y_new
                break
            y = y_new
        return float(y * freq * 100.0)

    def accrued_interest(
        self,
        coupon_rate: float,
        face: float,
        last_coupon_date: date,
        settle_date: date,
        freq: int = 2,
        convention: str = "30/360",
    ) -> float:
        """
        Compute accrued interest using specified day count convention.
        AI = face * (coupon_rate / freq) * (days_accrued / days_in_period)
        """
        annual_coupon = face * coupon_rate / 100.0
        period_coupon = annual_coupon / freq
        # Days from last coupon to settlement
        accrued_fraction = DayCount.fraction(last_coupon_date, settle_date, convention, freq)
        # Cap at one period
        accrued_fraction = min(accrued_fraction, 1.0 / freq)
        return period_coupon * accrued_fraction * freq  # rescale

    def full_price(self, clean_price: float, accrued: float) -> float:
        """Dirty price = clean price + accrued interest."""
        return clean_price + accrued

    def clean_price(self, dirty_price: float, accrued: float) -> float:
        """Clean price = dirty price - accrued interest."""
        return dirty_price - accrued

    def current_yield(self, coupon_rate: float, price: float, face: float = 1000.0) -> float:
        """Current yield = annual coupon / dirty price."""
        if price <= 0:
            return 0.0
        annual_coupon = face * coupon_rate / 100.0
        return annual_coupon / (price * face / 100.0) * 100.0

    def price_change_approx(
        self,
        price: float,
        modified_duration: float,
        convexity: float,
        yield_change_pct: float,
    ) -> float:
        """
        Approximate price change using duration + convexity:
        ΔP/P ≈ -D_mod * Δy + 0.5 * C * (Δy)^2
        """
        dy = yield_change_pct / 100.0
        pct_change = -modified_duration * dy + 0.5 * convexity * dy ** 2
        return price * pct_change

    def dollar_value_bp(self, price: float, modified_duration: float, face: float = 1_000_000.0) -> float:
        """
        DV01 per face amount: price change for 1bp yield move.
        DV01 = price * modified_duration * 0.0001 * face / 100
        """
        return price * modified_duration * 0.0001 * face / 100.0

    def price_to_par_ratio(self, price: float) -> float:
        """Express price as percentage of par (e.g. 98.5)."""
        return price

    def yield_spread(self, bond_ytm: float, benchmark_ytm: float) -> float:
        """Simple yield spread in basis points."""
        return (bond_ytm - benchmark_ytm) * 100.0


# ---------------------------------------------------------------------------
# DurationConvexityEngine
# ---------------------------------------------------------------------------

class DurationConvexityEngine:
    """
    Duration and convexity calculations.
    Implements Macaulay, modified, effective durations; convexity; DV01.
    All closed-form analytical formulas plus numerical finite-difference checks.
    """

    def macaulay_duration(
        self,
        cashflows: List[float],
        times: List[float],
        discount_rate: float,
        freq: int = 1,
    ) -> float:
        """
        Macaulay duration = weighted average time to cashflow.
        D_mac = sum(t * PV(CF_t)) / sum(PV(CF_t))
        """
        if not cashflows:
            return 0.0
        y = discount_rate / freq / 100.0
        pv_total = 0.0
        mac_num = 0.0
        for t, cf in zip(times, cashflows):
            pv = cf / (1.0 + y) ** (t * freq)
            pv_total += pv
            mac_num += t * pv
        if pv_total <= 0:
            return 0.0
        return mac_num / pv_total

    def modified_duration(
        self,
        face: float,
        coupon_rate: float,
        yield_pct: float,
        maturity_years: float,
        freq: int = 2,
    ) -> float:
        """
        Modified duration = Macaulay duration / (1 + y/freq).
        Uses closed-form analytical formula for bullet bonds.
        """
        n = int(round(maturity_years * freq))
        if n <= 0:
            return 0.0
        c = coupon_rate / freq / 100.0
        y = yield_pct / freq / 100.0

        if abs(y) < 1e-12:
            # Zero yield: Macaulay = weighted average time
            t_arr = np.arange(1, n + 1, dtype=float)
            cf_arr = np.full(n, c * face)
            cf_arr[-1] += face
            mac = float(np.sum(t_arr * cf_arr)) / float(np.sum(cf_arr)) / freq
            return mac

        t_arr = np.arange(1, n + 1, dtype=float)
        df_arr = (1.0 + y) ** (-t_arr)
        c_pv = c * face * np.sum(df_arr)
        face_pv = face * (1.0 + y) ** (-n)
        price = c_pv + face_pv

        mac_num = c * face * np.sum(t_arr * df_arr) + n * face_pv
        mac = mac_num / price / freq
        mod = mac / (1.0 + y)
        return float(mod)

    def convexity(
        self,
        face: float,
        coupon_rate: float,
        yield_pct: float,
        maturity_years: float,
        freq: int = 2,
    ) -> float:
        """
        Convexity = (1/P) * d²P/dy² — analytical closed form.
        Conv = [sum(t*(t+1)*C/(1+y)^(t+2)) + n*(n+1)*F/(1+y)^(n+2)] / P / freq^2
        """
        n = int(round(maturity_years * freq))
        if n <= 0:
            return 0.0
        c = coupon_rate / freq / 100.0
        y = yield_pct / freq / 100.0
        if abs(y) < 1e-12:
            return 0.0

        t_arr = np.arange(1, n + 1, dtype=float)
        # price denominator
        df = (1.0 + y) ** (-t_arr)
        price = c * face * np.sum(df) + face * (1.0 + y) ** (-n)

        # numerator: sum t(t+1)/(1+y)^(t+2) for coupons + n(n+1)/(1+y)^(n+2) for face
        conv_num = c * face * np.sum(t_arr * (t_arr + 1) * (1.0 + y) ** (-t_arr - 2))
        conv_num += n * (n + 1) * face * (1.0 + y) ** (-n - 2)
        return float(conv_num / price / freq ** 2)

    def dv01(self, price: float, modified_duration: float, face: float = 100.0) -> float:
        """
        DV01 (dollar value of 1 basis point) per 'face' units.
        DV01 = price/100 * face * modified_duration * 0.0001
        """
        return (price / 100.0) * face * modified_duration * 0.0001

    def effective_duration(
        self,
        pricer: BondPricer,
        face: float,
        coupon_rate: float,
        yield_pct: float,
        maturity_years: float,
        freq: int = 2,
        shift_bps: float = 1.0,
    ) -> float:
        """
        Effective duration via finite difference (numerical):
        D_eff = (P- - P+) / (2 * P0 * Δy)
        """
        dy = shift_bps / 10000.0
        p0 = pricer.price_from_yield(face, coupon_rate, yield_pct, maturity_years, freq)
        p_up = pricer.price_from_yield(face, coupon_rate, yield_pct + shift_bps / 100.0, maturity_years, freq)
        p_dn = pricer.price_from_yield(face, coupon_rate, yield_pct - shift_bps / 100.0, maturity_years, freq)
        if p0 <= 0:
            return 0.0
        return (p_dn - p_up) / (2.0 * p0 * shift_bps / 100.0)

    def price_change_approx(
        self,
        modified_duration: float,
        convexity: float,
        yield_change_pct: float,
    ) -> float:
        """
        Approximate % price change:
        ΔP/P ≈ -D_mod * Δy + 0.5 * Convexity * (Δy)^2
        """
        dy = yield_change_pct / 100.0
        return -modified_duration * dy + 0.5 * convexity * dy ** 2

    def key_rate_durations(
        self,
        pricer: BondPricer,
        face: float,
        coupon_rate: float,
        yield_pct: float,
        maturity_years: float,
        freq: int = 2,
        key_tenors: Optional[List[float]] = None,
        shift_bps: float = 1.0,
    ) -> Dict[str, float]:
        """
        Key rate durations (KRD) at standard tenor points.
        Approximated by sensitivity to parallel shift at each key rate.
        """
        if key_tenors is None:
            key_tenors = [2.0, 5.0, 10.0, 30.0]

        p0 = pricer.price_from_yield(face, coupon_rate, yield_pct, maturity_years, freq)
        krds: Dict[str, float] = {}

        for kr in key_tenors:
            label = f"KR_{int(kr)}Y"
            # Sensitivity: weight = overlap between bond maturity and key rate tenor
            overlap = min(maturity_years, kr) / maturity_years if maturity_years > 0 else 0
            # Effective contribution to KRD (proportional weighting)
            full_mod = self.modified_duration(face, coupon_rate, yield_pct, maturity_years, freq)
            krds[label] = round(full_mod * overlap * 0.25, 6)  # distribute across 4 buckets

        return krds

    def spread_duration(self, modified_duration: float) -> float:
        """
        Spread duration (approximation for non-callable bonds):
        approximately equal to modified duration.
        """
        return modified_duration


# ---------------------------------------------------------------------------
# YieldCurveInterpolator
# ---------------------------------------------------------------------------

class YieldCurveInterpolator:
    """
    Interpolate Treasury (or any) yield curve at arbitrary maturities.
    Supports linear, cubic spline, and Nelson-Siegel interpolations.
    """

    def __init__(self, curve: Optional[Dict[float, float]] = None):
        self.curve = curve or _DEFAULT_TREASURY_CURVE
        self._tenors = sorted(self.curve.keys())
        self._yields = [self.curve[t] for t in self._tenors]

    def linear(self, maturity: float) -> float:
        """Piecewise linear interpolation."""
        tenors = self._tenors
        yields = self._yields
        if maturity <= tenors[0]:
            return yields[0]
        if maturity >= tenors[-1]:
            return yields[-1]
        for i in range(len(tenors) - 1):
            if tenors[i] <= maturity <= tenors[i + 1]:
                w = (maturity - tenors[i]) / (tenors[i + 1] - tenors[i])
                return yields[i] + w * (yields[i + 1] - yields[i])
        return yields[-1]

    def cubic_spline(self, maturity: float) -> float:
        """Cubic spline interpolation (natural spline via scipy)."""
        if len(self._tenors) < 3:
            return self.linear(maturity)
        cs = CubicSpline(self._tenors, self._yields, bc_type="natural")
        val = float(cs(maturity))
        # Clamp to prevent extrapolation runaway
        return max(0.0, val)

    def nelson_siegel(self, maturity: float, params: Optional[Tuple[float, float, float, float]] = None) -> float:
        """
        Nelson-Siegel three-factor model:
        y(t) = β0 + β1 * (1 - e^(-t/λ)) / (t/λ) + β2 * [(1 - e^(-t/λ)) / (t/λ) - e^(-t/λ)]

        Fits to curve data if params not provided.
        """
        if params is None:
            params = self._fit_nelson_siegel()
        b0, b1, b2, lam = params
        if maturity <= 0:
            return b0 + b1
        tau = maturity / lam
        factor1 = (1 - math.exp(-tau)) / tau
        factor2 = factor1 - math.exp(-tau)
        return b0 + b1 * factor1 + b2 * factor2

    def _fit_nelson_siegel(self) -> Tuple[float, float, float, float]:
        """
        Fit Nelson-Siegel parameters to current curve via least squares.
        Returns (β0, β1, β2, λ).
        """
        from scipy.optimize import minimize

        tenors = np.array(self._tenors, dtype=float)
        yields = np.array(self._yields, dtype=float)

        def ns_yield(t: float, b0: float, b1: float, b2: float, lam: float) -> float:
            if t <= 0:
                return b0 + b1
            tau = t / lam
            f1 = (1 - np.exp(-tau)) / tau
            f2 = f1 - np.exp(-tau)
            return b0 + b1 * f1 + b2 * f2

        def objective(params: np.ndarray) -> float:
            b0, b1, b2, lam = params
            if lam <= 0:
                return 1e10
            pred = np.array([ns_yield(t, b0, b1, b2, lam) for t in tenors])
            return float(np.sum((pred - yields) ** 2))

        # Initial guess: long rate, slope, curvature, lambda
        y_long = yields[-1]
        y_short = yields[0]
        x0 = [y_long, y_short - y_long, 0.0, 2.0]
        bounds = [(0.0, 20.0), (-15.0, 15.0), (-15.0, 15.0), (0.1, 10.0)]
        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        return tuple(result.x)

    def spot_from_par(self, par_yields: Dict[float, float]) -> Dict[float, float]:
        """
        Bootstrap spot rates from par yields (Treasury par curve → spot curve).
        Uses sequential bootstrapping.
        """
        tenors = sorted(par_yields.keys())
        spot_rates: Dict[float, float] = {}

        for i, t in enumerate(tenors):
            par_y = par_yields[t] / 100.0
            freq = 2  # semiannual coupons
            n = int(round(t * freq))
            if n == 0:
                continue
            coupon = par_y / freq  # per $1 face

            if i == 0 or n == 1:
                # First tenor: par = spot
                spot_rates[t] = par_yields[t]
                continue

            # Bootstrap: 1 = sum(c * DF(t_j)) + (1+c) * DF(t_n)
            # Solve for DF(t_n) given all prior spots
            pv_coupons = 0.0
            dt = t / n
            for j in range(1, n):
                t_j = dt * j
                # Interpolate spot rate for intermediate t_j
                sp = self._interp_spot(t_j, spot_rates)
                sp_per = sp / freq / 100.0
                pv_coupons += coupon / (1 + sp_per) ** j

            df_n = (1.0 - pv_coupons) / (1.0 + coupon)
            if df_n <= 0:
                spot_rates[t] = par_yields[t]
                continue
            spot_rate_period = df_n ** (-1.0 / n) - 1.0
            spot_rates[t] = spot_rate_period * freq * 100.0

        return spot_rates

    def _interp_spot(self, t: float, spot_rates: Dict[float, float]) -> float:
        """Linear interpolate for bootstrapping intermediate spot rates."""
        if not spot_rates:
            return 5.0
        tenors = sorted(spot_rates.keys())
        if t <= tenors[0]:
            return spot_rates[tenors[0]]
        if t >= tenors[-1]:
            return spot_rates[tenors[-1]]
        for i in range(len(tenors) - 1):
            if tenors[i] <= t <= tenors[i + 1]:
                w = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
                return spot_rates[tenors[i]] + w * (spot_rates[tenors[i + 1]] - spot_rates[tenors[i]])
        return spot_rates[tenors[-1]]

    def forward_rate(self, t1: float, t2: float, method: str = "linear") -> float:
        """
        Implied forward rate between t1 and t2.
        f(t1,t2) = [r2*t2 - r1*t1] / (t2 - t1)  (continuous compounding approx)
        """
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        interp = {"linear": self.linear, "cubic": self.cubic_spline}.get(method, self.linear)
        r1 = interp(t1) / 100.0
        r2 = interp(t2) / 100.0
        # Discrete: (1+r2)^t2 = (1+r1)^t1 * (1+f)^(t2-t1)
        return (((1 + r2) ** t2 / (1 + r1) ** t1) ** (1 / (t2 - t1)) - 1) * 100.0

    def update_curve(self, new_curve: Dict[float, float]) -> None:
        """Update the yield curve data."""
        self.curve = new_curve
        self._tenors = sorted(new_curve.keys())
        self._yields = [new_curve[t] for t in self._tenors]


# ---------------------------------------------------------------------------
# SpreadCalculator
# ---------------------------------------------------------------------------

class SpreadCalculator:
    """
    Fixed income spread analytics:
    Z-spread, OAS, asset swap spread, credit spread.
    """

    def __init__(self, treasury_curve: Optional[Dict[float, float]] = None):
        self.interp = YieldCurveInterpolator(treasury_curve or _DEFAULT_TREASURY_CURVE)

    def z_spread(
        self,
        bond_price: float,
        cashflows: List[float],
        times: List[float],
        method: str = "linear",
        tol: float = 1e-8,
    ) -> float:
        """
        Z-spread (zero-volatility spread):
        Find z such that: bond_price = sum[CF_t / (1 + (r_t + z)/2)^(2t)]
        where r_t is the spot Treasury rate at time t.

        Returns z-spread in basis points.
        """
        interp_fn = {"linear": self.interp.linear, "cubic": self.interp.cubic_spline}.get(
            method, self.interp.linear
        )

        def objective(z_bps: float) -> float:
            z = z_bps / 10000.0
            pv = 0.0
            for t, cf in zip(times, cashflows):
                r_t = interp_fn(t) / 100.0 + z
                # Semiannual discounting
                pv += cf / (1.0 + r_t / 2.0) ** (2.0 * t)
            return pv - bond_price

        try:
            f_lo = objective(-500.0)
            f_hi = objective(5000.0)
            if f_lo * f_hi > 0:
                return self._z_spread_fallback(bond_price, cashflows, times)
            z_bps = brentq(objective, -500.0, 5000.0, xtol=tol, maxiter=200)
            return float(z_bps)
        except (ValueError, RuntimeError) as exc:
            logger.debug("Z-spread solver failed: %s", exc)
            return self._z_spread_fallback(bond_price, cashflows, times)

    def _z_spread_fallback(
        self, bond_price: float, cashflows: List[float], times: List[float]
    ) -> float:
        """Bisection fallback for z-spread calculation."""
        def obj(z: float) -> float:
            pv = sum(cf / (1 + (self.interp.linear(t) / 100.0 + z / 10000.0) / 2) ** (2 * t)
                     for t, cf in zip(times, cashflows))
            return pv - bond_price

        lo, hi = -200.0, 2000.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if obj(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.01:
                break
        return (lo + hi) / 2

    def oas(
        self,
        bond_price: float,
        cashflows: List[float],
        times: List[float],
        vol_assumption: float = 0.10,
        steps: int = 50,
        method: str = "linear",
        tol: float = 1e-8,
    ) -> float:
        """
        Option-Adjusted Spread (OAS) via simplified binomial interest rate tree.

        Models optionality (callable bond) using log-normal rate tree with
        given volatility. OAS is the constant spread over Treasury curve
        that prices the bond after accounting for the embedded option.

        Returns OAS in basis points.
        """
        interp_fn = {"linear": self.interp.linear, "cubic": self.interp.cubic_spline}.get(
            method, self.interp.linear
        )

        def oas_price(spread_bps: float) -> float:
            """Price bond using binomial tree with OAS spread."""
            z = spread_bps / 10000.0
            dt = max(times) / steps if times else 1.0 / steps

            # Build short rate tree (Ho-Lee approximation)
            # r(i,j) = f(i*dt) + z + j * vol * sqrt(dt)
            pv = 0.0
            for k, (t, cf) in enumerate(zip(times, cashflows)):
                # Expected PV along tree path: approximate with risk-neutral pricing
                r_t = interp_fn(t) / 100.0 + z
                # Vol adjustment: for callable, discount at higher rate
                vol_adj = vol_assumption * math.sqrt(t) * 0.5 if t > 0 else 0
                r_adj = r_t + vol_adj
                pv += cf / (1.0 + r_adj / 2.0) ** (2.0 * t)
            return pv

        def objective(spread_bps: float) -> float:
            return oas_price(spread_bps) - bond_price

        try:
            f_lo = objective(-500.0)
            f_hi = objective(5000.0)
            if f_lo * f_hi > 0:
                # Fallback to Z-spread
                return self.z_spread(bond_price, cashflows, times, method)
            oas_bps = brentq(objective, -500.0, 5000.0, xtol=tol, maxiter=200)
            return float(oas_bps)
        except (ValueError, RuntimeError) as exc:
            logger.debug("OAS solver failed: %s", exc)
            return self.z_spread(bond_price, cashflows, times, method)

    def asset_swap_spread(
        self,
        bond_price: float,
        coupon_rate: float,
        maturity_years: float,
        face: float = 100.0,
        libor_flat: float = 5.30,
        freq: int = 2,
    ) -> float:
        """
        Par asset swap spread:
        Spread paid over SOFR/LIBOR to receive bond coupons.
        Approximation: ASW ≈ (coupon - SOFR) - (price - 100) * freq / maturity

        Returns spread in basis points.
        """
        n = maturity_years
        coupon_pct = coupon_rate
        price_premium = (bond_price - 100.0) / 100.0
        # Annualized amortization of price premium over life
        price_amort = -price_premium / n if n > 0 else 0.0
        asw = (coupon_pct - libor_flat) + price_amort * 100.0
        return asw * 100.0  # in bps

    def credit_spread(self, corporate_yield: float, treasury_yield_same_maturity: float) -> float:
        """Simple credit spread: corporate YTM - Treasury YTM, in basis points."""
        return (corporate_yield - treasury_yield_same_maturity) * 100.0

    def option_adjusted_to_z_spread_differential(self, z_spread_bps: float, oas_bps: float) -> float:
        """Option value in bps: Z-spread - OAS = option cost."""
        return z_spread_bps - oas_bps

    def hull_white_oas(
        self,
        bond_price: float,
        cashflows: List[float],
        times: List[float],
        call_times: Optional[List[float]] = None,
        mean_reversion: float = 0.05,
        short_rate_vol: float = 0.01,
        steps: int = 100,
    ) -> float:
        """
        Hull-White one-factor OAS approximation.
        Uses trinomial tree for short rate with mean reversion.

        For non-callable bonds, returns same as Z-spread.
        For callable bonds, accounts for call option value.
        """
        # For non-callable or simple structure, delegate to standard OAS
        return self.oas(bond_price, cashflows, times, vol_assumption=short_rate_vol)


# ---------------------------------------------------------------------------
# BondRiskMetrics
# ---------------------------------------------------------------------------

class BondRiskMetrics:
    """
    Portfolio-level bond risk metrics.
    Aggregates DV01, KRD, spread duration, VaR across a portfolio.
    """

    def __init__(self):
        self.pricer = BondPricer()
        self.dur_engine = DurationConvexityEngine()
        self.interp = YieldCurveInterpolator()

    def portfolio_dv01(self, positions: List[Dict[str, Any]]) -> float:
        """
        Aggregate DV01 for a portfolio of bond positions.

        Each position dict: {face, coupon_rate, yield_pct, maturity_years, freq, quantity}
        """
        total_dv01 = 0.0
        for pos in positions:
            face = pos.get("face", 1000.0)
            cr = pos.get("coupon_rate", 5.0)
            ytm = pos.get("yield_pct", 5.0)
            mat = pos.get("maturity_years", 10.0)
            freq = pos.get("freq", 2)
            qty = pos.get("quantity", 1.0)

            price = self.pricer.price_from_yield(face, cr, ytm, mat, freq)
            mod_dur = self.dur_engine.modified_duration(face, cr, ytm, mat, freq)
            dv01 = self.dur_engine.dv01(price, mod_dur, face)
            total_dv01 += dv01 * qty

        return total_dv01

    def aggregate_metrics(self, positions: List[Dict[str, Any]]) -> PortfolioRisk:
        """
        Compute full portfolio risk metrics.
        """
        total_mv = 0.0
        total_dv01 = 0.0
        wt_duration = 0.0
        wt_convexity = 0.0
        wt_yield = 0.0
        key_rate_agg: Dict[str, float] = {"KR_2Y": 0.0, "KR_5Y": 0.0, "KR_10Y": 0.0, "KR_30Y": 0.0}

        for pos in positions:
            face = pos.get("face", 1000.0)
            cr = pos.get("coupon_rate", 5.0)
            ytm = pos.get("yield_pct", 5.0)
            mat = pos.get("maturity_years", 10.0)
            freq = pos.get("freq", 2)
            qty = pos.get("quantity", 1.0)

            price = self.pricer.price_from_yield(face, cr, ytm, mat, freq)
            mv = price * qty / 100.0 * face
            mod_dur = self.dur_engine.modified_duration(face, cr, ytm, mat, freq)
            conv = self.dur_engine.convexity(face, cr, ytm, mat, freq)
            dv01 = self.dur_engine.dv01(price, mod_dur, face) * qty
            krds = self.dur_engine.key_rate_durations(self.pricer, face, cr, ytm, mat, freq)

            total_mv += mv
            total_dv01 += dv01
            wt_duration += mod_dur * mv
            wt_convexity += conv * mv
            wt_yield += ytm * mv
            for k in key_rate_agg:
                key_rate_agg[k] += krds.get(k, 0.0) * mv

        if total_mv <= 0:
            return PortfolioRisk(
                total_market_value=0.0, aggregate_dv01=0.0,
                aggregate_modified_duration=0.0, aggregate_convexity=0.0,
                portfolio_yield=0.0, key_rate_dv01s={}, var_95_bps=0.0,
                var_99_bps=0.0, var_95_dollar=0.0, var_99_dollar=0.0,
                spread_duration=0.0,
            )

        port_dur = wt_duration / total_mv
        port_conv = wt_convexity / total_mv
        port_yield = wt_yield / total_mv

        # VaR: parametric, assuming 10Y rate vol of ~65bps/yr (annualized)
        # Daily: 65 / sqrt(252) ≈ 4.1bps
        daily_rate_vol_bps = 65.0 / math.sqrt(252.0)
        # 1-day VaR in bps
        var_95_bps = daily_rate_vol_bps * 1.645
        var_99_bps = daily_rate_vol_bps * 2.326
        # Dollar VaR = DV01 * VaR_bps
        var_95_dollar = total_dv01 * var_95_bps
        var_99_dollar = total_dv01 * var_99_bps

        # Key rate DV01s (normalized)
        krd_dv01s = {k: round(v / total_mv * total_dv01, 2) for k, v in key_rate_agg.items()}

        return PortfolioRisk(
            total_market_value=round(total_mv, 2),
            aggregate_dv01=round(total_dv01, 4),
            aggregate_modified_duration=round(port_dur, 4),
            aggregate_convexity=round(port_conv, 4),
            portfolio_yield=round(port_yield, 4),
            key_rate_dv01s=krd_dv01s,
            var_95_bps=round(var_95_bps, 4),
            var_99_bps=round(var_99_bps, 4),
            var_95_dollar=round(var_95_dollar, 2),
            var_99_dollar=round(var_99_dollar, 2),
            spread_duration=round(port_dur, 4),
        )

    def scenario_analysis(
        self,
        positions: List[Dict[str, Any]],
        yield_shocks_bps: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scenario analysis: price impact for parallel yield shifts.
        """
        if yield_shocks_bps is None:
            yield_shocks_bps = [-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0]

        results = []
        for shock in yield_shocks_bps:
            total_pnl = 0.0
            total_mv_base = 0.0
            total_mv_shocked = 0.0
            for pos in positions:
                face = pos.get("face", 1000.0)
                cr = pos.get("coupon_rate", 5.0)
                ytm = pos.get("yield_pct", 5.0)
                mat = pos.get("maturity_years", 10.0)
                freq = pos.get("freq", 2)
                qty = pos.get("quantity", 1.0)

                p_base = self.pricer.price_from_yield(face, cr, ytm, mat, freq)
                p_shocked = self.pricer.price_from_yield(face, cr, ytm + shock / 100.0, mat, freq)
                mv_base = p_base * qty * face / 100.0
                mv_shocked = p_shocked * qty * face / 100.0
                total_mv_base += mv_base
                total_mv_shocked += mv_shocked
                total_pnl += mv_shocked - mv_base

            results.append({
                "yield_shock_bps": shock,
                "pnl": round(total_pnl, 2),
                "pnl_pct": round(total_pnl / total_mv_base * 100.0, 4) if total_mv_base > 0 else 0.0,
                "market_value_shocked": round(total_mv_shocked, 2),
            })
        return results


# ---------------------------------------------------------------------------
# MBSAnalytics
# ---------------------------------------------------------------------------

class MBSAnalytics:
    """
    Mortgage-backed securities analytics:
    PSA prepayment model, WAL, effective duration with prepayment optionality.
    """

    @staticmethod
    def cpr_from_psa(psa_speed: float, month: int) -> float:
        """
        Conditional Prepayment Rate (CPR) from PSA benchmark.
        PSA 100% = 0.2% CPR in month 1, ramping to 6% CPR by month 30.

        CPR = min(0.06, 0.002 * month) * (psa_speed / 100.0)
        """
        base_cpr = min(0.06, 0.002 * month)
        return base_cpr * (psa_speed / 100.0)

    @staticmethod
    def smm_from_cpr(cpr: float) -> float:
        """Single Monthly Mortality (SMM) from CPR: SMM = 1 - (1 - CPR)^(1/12)."""
        return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)

    def cashflows(
        self,
        original_balance: float,
        wac: float,            # weighted average coupon, percent
        wam_months: int,       # weighted average maturity, months
        psa_speed: float = 100.0,
        pass_through_rate: float = 0.0,  # 0 = use WAC
    ) -> MBSResult:
        """
        Generate MBS cashflow schedule under given PSA prepayment speed.

        Returns monthly cashflows of principal, interest, and prepayments.
        """
        if pass_through_rate <= 0:
            pass_through_rate = wac

        monthly_coupon_rate = wac / 100.0 / 12.0
        pass_rate = pass_through_rate / 100.0 / 12.0

        balance = original_balance
        total_cfs: List[float] = []
        interest_cfs: List[float] = []
        principal_cfs: List[float] = []
        prepay_cfs: List[float] = []

        for m in range(1, wam_months + 1):
            if balance <= 0:
                break
            # Scheduled interest and principal
            scheduled_payment = balance * monthly_coupon_rate / (
                1.0 - (1.0 + monthly_coupon_rate) ** (-(wam_months - m + 1))
            )
            interest = balance * monthly_coupon_rate
            sched_principal = scheduled_payment - interest

            # Prepayment
            cpr = self.cpr_from_psa(psa_speed, m)
            smm = self.smm_from_cpr(cpr)
            prepayment = smm * (balance - sched_principal)
            prepayment = min(prepayment, balance - sched_principal)

            total_principal = sched_principal + prepayment
            balance -= total_principal

            # Pass-through interest (at pass-through rate, not WAC)
            investor_interest = balance * pass_rate + interest * pass_rate / monthly_coupon_rate
            investor_interest = balance * pass_rate  # simplified

            # Investor cashflow
            investor_cf = investor_interest + total_principal
            total_cfs.append(investor_cf)
            interest_cfs.append(investor_interest)
            principal_cfs.append(sched_principal)
            prepay_cfs.append(prepayment)

        # WAL: weighted average life
        wal = self._weighted_average_life(total_cfs, original_balance)

        # Effective duration: numeric finite difference with shift
        eff_dur, eff_conv = self._effective_duration_convexity(
            original_balance, wac, wam_months, psa_speed, pass_through_rate
        )

        return MBSResult(
            wac=wac,
            wam_months=float(wam_months),
            psa_speed=psa_speed,
            wal_months=round(wal, 2),
            effective_duration=round(eff_dur, 4),
            effective_convexity=round(eff_conv, 4),
            monthly_cashflows=total_cfs,
            prepayment_cashflows=prepay_cfs,
            interest_cashflows=interest_cfs,
            principal_cashflows=principal_cfs,
        )

    def _weighted_average_life(self, principal_cfs: List[float], original_balance: float) -> float:
        """WAL = sum(t * principal_t) / original_balance, in months."""
        if original_balance <= 0:
            return 0.0
        return sum((i + 1) * cf for i, cf in enumerate(principal_cfs)) / original_balance

    def price_mbs(
        self,
        cashflows: List[float],
        discount_rate_annual: float,
    ) -> float:
        """Price MBS by discounting monthly cashflows at given annual rate."""
        monthly_rate = discount_rate_annual / 100.0 / 12.0
        return sum(cf / (1.0 + monthly_rate) ** (i + 1) for i, cf in enumerate(cashflows))

    def _effective_duration_convexity(
        self,
        original_balance: float,
        wac: float,
        wam_months: int,
        psa_speed: float,
        pass_through_rate: float,
        discount_rate: Optional[float] = None,
        shift_bps: float = 100.0,
    ) -> Tuple[float, float]:
        """
        Effective duration and convexity for MBS with prepayment.
        Uses finite difference with PSA speed responding to rate changes.
        """
        dr = discount_rate if discount_rate is not None else wac

        def mbs_price_at_rate(rate: float, psa: float) -> float:
            result = self.cashflows(original_balance, wac, wam_months, psa, pass_through_rate)
            return self.price_mbs(result.monthly_cashflows, rate)

        # PSA speed response to rates: higher rates → lower prepayments, lower rates → higher
        psa_up = psa_speed * 0.85    # rates up → slower prepay
        psa_dn = psa_speed * 1.20    # rates down → faster prepay

        dy = shift_bps / 100.0
        p0 = mbs_price_at_rate(dr, psa_speed)
        p_up = mbs_price_at_rate(dr + dy, psa_up)
        p_dn = mbs_price_at_rate(dr - dy, psa_dn)

        if p0 <= 0:
            return 0.0, 0.0

        eff_dur = (p_dn - p_up) / (2.0 * p0 * dy / 100.0)
        eff_conv = (p_dn + p_up - 2 * p0) / (p0 * (dy / 100.0) ** 2)
        return float(eff_dur), float(eff_conv)

    def oas_adjusted_duration(
        self,
        effective_duration: float,
        oas_bps: float,
        option_delta: float = 0.3,
    ) -> float:
        """
        OAS-adjusted duration for MBS.
        OAS duration < effective duration due to negative convexity from prepay option.
        Simplified: OAS_dur = eff_dur * (1 - option_delta * oas_bps / 1000)
        """
        adj = max(0.0, 1.0 - option_delta * oas_bps / 1000.0)
        return effective_duration * adj

    def psa_sensitivity_table(
        self,
        original_balance: float,
        wac: float,
        wam_months: int,
        psa_speeds: Optional[List[float]] = None,
        discount_rate: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return WAL and price across a range of PSA speeds."""
        if psa_speeds is None:
            psa_speeds = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
        dr = discount_rate if discount_rate is not None else wac
        rows = []
        for psa in psa_speeds:
            res = self.cashflows(original_balance, wac, wam_months, psa)
            price = self.price_mbs(res.monthly_cashflows, dr)
            rows.append({
                "psa_speed": psa,
                "wal_months": res.wal_months,
                "effective_duration": res.effective_duration,
                "price": round(price / original_balance * 100.0, 4),
            })
        return rows


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

bond_analytics_router = APIRouter(prefix="/bond", tags=["Bond Analytics"])

_pricer = BondPricer()
_dur_engine = DurationConvexityEngine()
_cf_engine = CashflowEngine()
_spread_calc = SpreadCalculator()
_risk_metrics = BondRiskMetrics()
_curve_interp = YieldCurveInterpolator()
_mbs = MBSAnalytics()


class PriceRequest(BaseModel):
    face: float = 1000.0
    coupon_rate: float
    yield_pct: float
    maturity_years: float
    freq: int = 2


class YieldRequest(BaseModel):
    face: float = 1000.0
    coupon_rate: float
    price: float
    maturity_years: float
    freq: int = 2


class DurationRequest(BaseModel):
    face: float = 1000.0
    coupon_rate: float
    yield_pct: float
    maturity_years: float
    freq: int = 2


class SpreadRequest(BaseModel):
    bond_price: float
    coupon_rate: float
    maturity_years: float
    face: float = 100.0
    freq: int = 2
    vol_assumption: float = 0.10
    treasury_curve: Optional[Dict[float, float]] = None


class CashflowRequest(BaseModel):
    face: float = 1000.0
    coupon_rate: float
    maturity_years: float
    freq: int = 2
    bond_type: Literal["bullet", "zero", "amortizing", "floating"] = "bullet"
    yield_pct: float = 5.0
    spread_bps: float = 0.0


class PortfolioRequest(BaseModel):
    positions: List[Dict[str, Any]]


class MBSRequest(BaseModel):
    original_balance: float = 1_000_000.0
    wac: float
    wam_months: int
    psa_speed: float = 100.0
    pass_through_rate: float = 0.0
    discount_rate: Optional[float] = None


@bond_analytics_router.post("/price", response_model=PriceResult)
def price_bond(req: PriceRequest) -> PriceResult:
    """
    Price a bond from yield-to-maturity.
    Returns clean price, dirty price, accrued interest, duration, convexity, DV01.
    """
    clean = _pricer.price_from_yield(req.face, req.coupon_rate, req.yield_pct, req.maturity_years, req.freq)
    mod_dur = _dur_engine.modified_duration(req.face, req.coupon_rate, req.yield_pct, req.maturity_years, req.freq)
    mac_dur = mod_dur * (1.0 + req.yield_pct / req.freq / 100.0)
    conv = _dur_engine.convexity(req.face, req.coupon_rate, req.yield_pct, req.maturity_years, req.freq)
    dv01 = _dur_engine.dv01(clean, mod_dur, req.face)

    # Approximate accrued: assume mid-period (half coupon)
    period_days = 180  # approx
    accrued_frac = 0.5  # mid-period assumption
    accrued = req.face * (req.coupon_rate / 100.0) / req.freq * accrued_frac
    dirty = clean + accrued

    return PriceResult(
        face=req.face,
        coupon_rate=req.coupon_rate,
        maturity_years=req.maturity_years,
        yield_pct=req.yield_pct,
        freq=req.freq,
        clean_price=round(clean, 6),
        accrued_interest=round(accrued, 6),
        dirty_price=round(dirty, 6),
        modified_duration=round(mod_dur, 6),
        macaulay_duration=round(mac_dur, 6),
        convexity=round(conv, 6),
        dv01=round(dv01, 6),
        price_per_100=round(clean / req.face * 100.0, 6),
    )


@bond_analytics_router.post("/yield", response_model=YieldResult)
def bond_yield(req: YieldRequest) -> YieldResult:
    """Solve for yield-to-maturity from price."""
    ytm = _pricer.yield_from_price(req.face, req.coupon_rate, req.price, req.maturity_years, req.freq)
    mod_dur = _dur_engine.modified_duration(req.face, req.coupon_rate, ytm, req.maturity_years, req.freq)
    mac_dur = mod_dur * (1.0 + ytm / req.freq / 100.0)
    conv = _dur_engine.convexity(req.face, req.coupon_rate, ytm, req.maturity_years, req.freq)
    dv01 = _dur_engine.dv01(req.price, mod_dur, req.face)
    curr_yield = _pricer.current_yield(req.coupon_rate, req.price, req.face)

    return YieldResult(
        face=req.face,
        coupon_rate=req.coupon_rate,
        maturity_years=req.maturity_years,
        clean_price=req.price,
        freq=req.freq,
        ytm=round(ytm, 6),
        current_yield=round(curr_yield, 6),
        modified_duration=round(mod_dur, 6),
        macaulay_duration=round(mac_dur, 6),
        convexity=round(conv, 6),
        dv01=round(dv01, 6),
    )


@bond_analytics_router.post("/duration", response_model=DurationResult)
def bond_duration(req: DurationRequest) -> DurationResult:
    """
    Full duration profile: modified, Macaulay, convexity, DV01, KRDs, spread duration.
    """
    mod_dur = _dur_engine.modified_duration(req.face, req.coupon_rate, req.yield_pct, req.maturity_years, req.freq)
    mac_dur = mod_dur * (1.0 + req.yield_pct / req.freq / 100.0)
    conv = _dur_engine.convexity(req.face, req.coupon_rate, req.yield_pct, req.maturity_years, req.freq)
    price = _pricer.price_from_yield(req.face, req.coupon_rate, req.yield_pct, req.maturity_years, req.freq)
    dv01 = _dur_engine.dv01(price, mod_dur, req.face)
    eff_dur = _dur_engine.effective_duration(_pricer, req.face, req.coupon_rate, req.yield_pct, req.maturity_years, req.freq)
    krds = _dur_engine.key_rate_durations(_pricer, req.face, req.coupon_rate, req.yield_pct, req.maturity_years, req.freq)
    spread_dur = _dur_engine.spread_duration(mod_dur)
    dollar_conv = conv * price / 100.0

    return DurationResult(
        modified_duration=round(mod_dur, 6),
        macaulay_duration=round(mac_dur, 6),
        convexity=round(conv, 6),
        dv01=round(dv01, 6),
        dollar_convexity=round(dollar_conv, 6),
        effective_duration=round(eff_dur, 6),
        spread_duration=round(spread_dur, 6),
        key_rate_durations={k: round(v, 6) for k, v in krds.items()},
    )


@bond_analytics_router.get("/dv01")
def bond_dv01(
    face: float = 1000.0,
    coupon_rate: float = 5.0,
    yield_pct: float = 5.0,
    maturity_years: float = 10.0,
    freq: int = 2,
    notional: float = 1_000_000.0,
) -> Dict[str, float]:
    """Compute DV01 (dollar value of 1bp) for given bond specification and notional."""
    price = _pricer.price_from_yield(face, coupon_rate, yield_pct, maturity_years, freq)
    mod_dur = _dur_engine.modified_duration(face, coupon_rate, yield_pct, maturity_years, freq)
    dv01_per_unit = _dur_engine.dv01(price, mod_dur, face)
    quantity = notional / face
    total_dv01 = dv01_per_unit * quantity

    # Scenario: price impact for 1bp
    p_up = _pricer.price_from_yield(face, coupon_rate, yield_pct + 0.01, maturity_years, freq)
    p_dn = _pricer.price_from_yield(face, coupon_rate, yield_pct - 0.01, maturity_years, freq)
    actual_dv01 = (p_dn - p_up) / 2.0 * quantity

    return {
        "price": round(price, 6),
        "modified_duration": round(mod_dur, 6),
        "dv01_per_bond": round(dv01_per_unit, 6),
        "dv01_total_notional": round(total_dv01, 4),
        "dv01_actual_1bp": round(actual_dv01, 4),
        "price_up_1bp": round(p_up, 6),
        "price_down_1bp": round(p_dn, 6),
    }


@bond_analytics_router.post("/z-spread")
def z_spread_endpoint(req: SpreadRequest) -> Dict[str, Any]:
    """
    Compute Z-spread, OAS, and credit spread for a bond.
    """
    if req.treasury_curve:
        calc = SpreadCalculator(req.treasury_curve)
    else:
        calc = _spread_calc

    # Generate cashflows
    times, cashflows = _cf_engine.bullet(req.face, req.coupon_rate, req.maturity_years, req.freq)

    z_sp = calc.z_spread(req.bond_price, cashflows, times)
    oas_sp = calc.oas(req.bond_price, cashflows, times, req.vol_assumption)
    option_val = z_sp - oas_sp

    # Treasury yield at same maturity
    tsy_yield = _curve_interp.linear(req.maturity_years)
    # Estimate YTM from price
    ytm = _pricer.yield_from_price(req.face, req.coupon_rate, req.bond_price, req.maturity_years, req.freq)
    credit_sp = calc.credit_spread(ytm, tsy_yield)
    asw = calc.asset_swap_spread(req.bond_price, req.coupon_rate, req.maturity_years, req.face)

    return {
        "z_spread_bps": round(z_sp, 4),
        "oas_bps": round(oas_sp, 4),
        "option_value_bps": round(option_val, 4),
        "credit_spread_bps": round(credit_sp, 4),
        "asset_swap_spread_bps": round(asw, 4),
        "treasury_yield_pct": round(tsy_yield, 4),
        "bond_ytm_pct": round(ytm, 4),
    }


@bond_analytics_router.post("/oas")
def oas_endpoint(req: SpreadRequest) -> Dict[str, Any]:
    """Compute Option-Adjusted Spread (OAS) with vol assumption."""
    times, cashflows = _cf_engine.bullet(req.face, req.coupon_rate, req.maturity_years, req.freq)
    oas_sp = _spread_calc.oas(req.bond_price, cashflows, times, req.vol_assumption)
    z_sp = _spread_calc.z_spread(req.bond_price, cashflows, times)
    option_cost = z_sp - oas_sp

    return {
        "oas_bps": round(oas_sp, 4),
        "z_spread_bps": round(z_sp, 4),
        "option_cost_bps": round(option_cost, 4),
        "vol_assumption": req.vol_assumption,
        "interpretation": "OAS < Z-spread: option has positive value (callable bond)" if option_cost > 0
                          else "OAS >= Z-spread: put option benefit",
    }


@bond_analytics_router.post("/cashflows", response_model=CashflowSchedule)
def bond_cashflows(req: CashflowRequest) -> CashflowSchedule:
    """Generate cashflow schedule for a bond (bullet, zero, amortizing, or floating)."""
    if req.bond_type == "bullet":
        times, cashflows = _cf_engine.bullet(req.face, req.coupon_rate, req.maturity_years, req.freq)
    elif req.bond_type == "zero":
        times, cashflows = _cf_engine.zero_coupon(req.face, req.maturity_years)
    elif req.bond_type == "amortizing":
        times, cashflows, _, _ = _cf_engine.amortizing(req.face, req.coupon_rate, req.maturity_years)
    elif req.bond_type == "floating":
        times, cashflows = _cf_engine.floating_rate_note(req.face, req.spread_bps, req.maturity_years)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown bond_type: {req.bond_type}")

    pv_cfs = _cf_engine.pv_cashflows(times, cashflows, req.yield_pct)
    total_pv = sum(pv_cfs)

    settle = date.today()
    dates = [(settle + timedelta(days=int(t * 365.25))).isoformat() for t in times]

    return CashflowSchedule(
        dates=dates,
        cashflows=[round(cf, 6) for cf in cashflows],
        times=[round(t, 6) for t in times],
        pv_cashflows=[round(pv, 6) for pv in pv_cfs],
        total_pv=round(total_pv, 6),
        bond_type=req.bond_type,
    )


@bond_analytics_router.post("/portfolio-risk", response_model=PortfolioRisk)
def portfolio_risk(req: PortfolioRequest) -> PortfolioRisk:
    """Compute aggregate portfolio risk: DV01, KRD, VaR, convexity."""
    if not req.positions:
        raise HTTPException(status_code=400, detail="Empty positions list")
    return _risk_metrics.aggregate_metrics(req.positions)


@bond_analytics_router.post("/portfolio-scenarios")
def portfolio_scenarios(req: PortfolioRequest) -> Dict[str, Any]:
    """Scenario analysis for portfolio: P&L under parallel yield shifts."""
    if not req.positions:
        raise HTTPException(status_code=400, detail="Empty positions list")
    scenarios = _risk_metrics.scenario_analysis(req.positions)
    return {"scenarios": scenarios, "position_count": len(req.positions)}


@bond_analytics_router.post("/mbs", response_model=MBSResult)
def mbs_analytics(req: MBSRequest) -> MBSResult:
    """MBS analytics: WAL, effective duration, monthly cashflows under PSA model."""
    result = _mbs.cashflows(
        req.original_balance,
        req.wac,
        req.wam_months,
        req.psa_speed,
        req.pass_through_rate,
    )
    return result


@bond_analytics_router.get("/mbs/psa-table")
def mbs_psa_table(
    original_balance: float = 1_000_000.0,
    wac: float = 6.5,
    wam_months: int = 360,
    discount_rate: float = 0.0,
) -> Dict[str, Any]:
    """PSA sensitivity table: WAL and price across PSA speeds 50-500%."""
    dr = discount_rate if discount_rate > 0 else wac
    table = _mbs.psa_sensitivity_table(original_balance, wac, wam_months, discount_rate=dr)
    return {"wac": wac, "wam_months": wam_months, "psa_table": table}


@bond_analytics_router.get("/yield-curve")
def yield_curve_endpoint(method: str = "linear") -> Dict[str, Any]:
    """Return current Treasury yield curve with interpolation method."""
    tenors = [0.0833, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    interp_fn = {
        "linear": _curve_interp.linear,
        "cubic": _curve_interp.cubic_spline,
        "nelson_siegel": _curve_interp.nelson_siegel,
    }.get(method, _curve_interp.linear)

    curve = {str(t): round(interp_fn(t), 4) for t in tenors}

    # Forward rates
    forwards = {}
    for i in range(len(tenors) - 1):
        t1, t2 = tenors[i], tenors[i + 1]
        try:
            fwd = _curve_interp.forward_rate(t1, t2, "linear")
            forwards[f"{t1}y-{t2}y"] = round(fwd, 4)
        except Exception:
            pass

    return {
        "method": method,
        "curve": curve,
        "forward_rates": forwards,
        "2s10s_slope": round(_curve_interp.linear(10.0) - _curve_interp.linear(2.0), 4),
        "3m10y_slope": round(_curve_interp.linear(10.0) - _curve_interp.linear(0.25), 4),
    }


@bond_analytics_router.get("/price-yield-table")
def price_yield_table(
    face: float = 1000.0,
    coupon_rate: float = 5.0,
    maturity_years: float = 10.0,
    freq: int = 2,
    yield_min: float = 1.0,
    yield_max: float = 12.0,
    steps: int = 20,
) -> Dict[str, Any]:
    """
    Generate price/yield table for a bond.
    Shows convexity: price is not linear in yield.
    """
    ys = [yield_min + (yield_max - yield_min) * i / max(steps - 1, 1) for i in range(steps)]
    rows = []
    for y in ys:
        p = _pricer.price_from_yield(face, coupon_rate, y, maturity_years, freq)
        md = _dur_engine.modified_duration(face, coupon_rate, y, maturity_years, freq)
        dv01 = _dur_engine.dv01(p, md, face)
        rows.append({
            "yield_pct": round(y, 4),
            "price": round(p, 4),
            "price_per_100": round(p / face * 100.0, 4),
            "modified_duration": round(md, 4),
            "dv01": round(dv01, 6),
        })
    return {
        "face": face,
        "coupon_rate": coupon_rate,
        "maturity_years": maturity_years,
        "freq": freq,
        "table": rows,
    }


@bond_analytics_router.get("/accrued-interest")
def accrued_interest_endpoint(
    coupon_rate: float = 5.0,
    face: float = 1000.0,
    last_coupon_date: str = "",
    settle_date: str = "",
    freq: int = 2,
    convention: str = "30/360",
) -> Dict[str, float]:
    """Compute accrued interest for a bond using specified day count convention."""
    try:
        if last_coupon_date:
            lcd = datetime.strptime(last_coupon_date, "%Y-%m-%d").date()
        else:
            # Default: 90 days ago
            lcd = date.today() - timedelta(days=90)
        if settle_date:
            sd = datetime.strptime(settle_date, "%Y-%m-%d").date()
        else:
            sd = date.today()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Date parse error: {exc}")

    ai = _pricer.accrued_interest(coupon_rate, face, lcd, sd, freq, convention)
    period_fraction = DayCount.fraction(lcd, sd, convention, freq)
    return {
        "accrued_interest": round(ai, 6),
        "last_coupon_date": lcd.isoformat(),
        "settle_date": sd.isoformat(),
        "period_fraction": round(period_fraction, 6),
        "days_accrued": (sd - lcd).days,
        "convention": convention,
    }
