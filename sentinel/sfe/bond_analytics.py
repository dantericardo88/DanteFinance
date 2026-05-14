"""
Fixed income analytics — Dimensions 50–52.

Bond price, duration, convexity, DV01, scenario analysis, Altman Z-score.
Dims 50 (credit risk 0→5), 51 (duration/convexity 0→6), 52 (DV01 1→6).
"""
from __future__ import annotations

import asyncio
import math
from datetime import date
from typing import Optional

import numpy as np
import yfinance as yf
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BondPriceResult(BaseModel):
    face_value: float
    coupon_rate: float
    years_to_maturity: float
    yield_to_maturity: float
    frequency: int
    # Pricing
    clean_price: float        # price per $100 face
    dirty_price: float        # includes accrued interest
    accrued_interest: float
    # Duration measures
    macaulay_duration: float  # in years
    modified_duration: float  # price sensitivity per 1% yield change
    effective_duration: float # DV01-based, parallel shift
    # Convexity
    convexity: float          # second-order price sensitivity
    # DV01 (dollar value of 1bp) per $1M face
    dv01: float
    # Scenario analysis: price change % for yield shifts
    scenarios: dict[str, float]  # {"-200bps": ..., "-100bps": ..., "+100bps": ..., "+200bps": ...}
    # Spread
    credit_spread_bps: float
    warnings: list[str]


class AltmanZResult(BaseModel):
    ticker: str
    z_score: float
    classification: str           # "safe" | "grey" | "distress"
    probability_of_distress: float  # 0–1
    # Component ratios
    working_capital_to_assets: float      # X1
    retained_earnings_to_assets: float    # X2
    ebit_to_assets: float                 # X3
    market_cap_to_book_liabilities: float # X4
    sales_to_assets: float                # X5
    # Metadata
    as_of: str
    data_source: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Internal bond math helpers
# ---------------------------------------------------------------------------

def _bond_price(
    face: float,
    coupon_rate: float,
    ytm: float,
    years: float,
    freq: int,
) -> float:
    """
    Compute the full (dirty) price of a bond per $face.

    P = sum_{t=1}^{n} C/f / (1+y/f)^t  +  F / (1+y/f)^n
    where n = years * freq (total periods), C = coupon_rate * face.
    """
    n = int(round(years * freq))
    coupon = coupon_rate * face / freq
    r = ytm / freq
    if abs(r) < 1e-12:
        # Zero-yield edge case: price = sum of undiscounted cash flows
        return coupon * n + face

    periods = np.arange(1, n + 1, dtype=float)
    discount = (1.0 + r) ** periods
    pv_coupons = np.sum(coupon / discount)
    pv_face = face / ((1.0 + r) ** n)
    return pv_coupons + pv_face


def _bond_price_per100(
    coupon_rate: float,
    ytm: float,
    years: float,
    freq: int,
) -> float:
    """Price per $100 face value."""
    return _bond_price(100.0, coupon_rate, ytm, years, freq)


def _macaulay_duration(
    face: float,
    coupon_rate: float,
    ytm: float,
    years: float,
    freq: int,
) -> float:
    """
    Macaulay duration: weighted average time of cash flows (in years).

    D_mac = sum(t_i * PV(CF_i)) / Price
    where t_i is time in years for period i.
    """
    n = int(round(years * freq))
    coupon = coupon_rate * face / freq
    r = ytm / freq

    price = _bond_price(face, coupon_rate, ytm, years, freq)
    if price <= 0:
        return 0.0

    periods = np.arange(1, n + 1, dtype=float)
    times_years = periods / freq

    if abs(r) < 1e-12:
        cf = np.full(n, coupon)
        cf[-1] += face
        weighted = np.sum(times_years * cf)
        return weighted / price

    discount = (1.0 + r) ** periods
    pv_coupons = coupon / discount
    pv_face_arr = np.zeros(n)
    pv_face_arr[-1] = face / ((1.0 + r) ** n)
    pv_cf = pv_coupons + pv_face_arr

    return float(np.sum(times_years * pv_cf) / price)


def _convexity(
    face: float,
    coupon_rate: float,
    ytm: float,
    years: float,
    freq: int,
) -> float:
    """
    Convexity: second-order price sensitivity.

    C = sum(t_i*(t_i + 1/f) * PV(CF_i)) / (Price * (1+y/f)^2)
    """
    n = int(round(years * freq))
    coupon = coupon_rate * face / freq
    r = ytm / freq
    price = _bond_price(face, coupon_rate, ytm, years, freq)

    if price <= 0:
        return 0.0

    periods = np.arange(1, n + 1, dtype=float)
    times_years = periods / freq

    if abs(r) < 1e-12:
        cf = np.full(n, coupon)
        cf[-1] += face
        weights = times_years * (times_years + 1.0 / freq)
        return float(np.sum(weights * cf) / price)

    discount = (1.0 + r) ** periods
    pv_coupons = coupon / discount
    pv_face_arr = np.zeros(n)
    pv_face_arr[-1] = face / ((1.0 + r) ** n)
    pv_cf = pv_coupons + pv_face_arr

    weights = times_years * (times_years + 1.0 / freq)
    denom = price * (1.0 + r) ** 2
    return float(np.sum(weights * pv_cf) / denom)


# ---------------------------------------------------------------------------
# Yield curve interpolation helper
# ---------------------------------------------------------------------------

async def _fetch_ytm_from_curve(years_to_maturity: float) -> tuple[float, list[str]]:
    """
    Fetch Treasury yield curve and linearly interpolate a spot rate
    for the given maturity. Returns (rate_decimal, warnings).
    """
    warnings: list[str] = []
    try:
        from sentinel.sfe.yield_curve import get_yield_curve  # local import to avoid circulars
        curve_result = await get_yield_curve()
        tenors = [(tp.tenor, tp.yield_pct) for tp in curve_result.curve]
        if not tenors:
            warnings.append("Yield curve returned no tenor points; defaulting YTM to 4.5%")
            return 0.045, warnings

        # Build (years, yield_decimal) pairs
        _TENOR_TO_YEARS: dict[str, float] = {
            "1M": 1 / 12, "3M": 0.25, "6M": 0.5,
            "1Y": 1.0, "2Y": 2.0, "5Y": 5.0, "7Y": 7.0,
            "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
        }
        pairs = [
            (_TENOR_TO_YEARS[t], y / 100.0)  # FRED returns pct, convert to decimal
            for t, y in tenors
            if t in _TENOR_TO_YEARS and y is not None
        ]
        if not pairs:
            warnings.append("No usable tenor points from yield curve; defaulting YTM to 4.5%")
            return 0.045, warnings

        pairs.sort(key=lambda x: x[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]

        # Clamp to range
        if years_to_maturity <= xs[0]:
            return ys[0], warnings
        if years_to_maturity >= xs[-1]:
            return ys[-1], warnings

        # Linear interpolation
        rate = float(np.interp(years_to_maturity, xs, ys))
        return rate, warnings

    except Exception as exc:
        warnings.append(f"Yield curve fetch failed ({exc}); defaulting YTM to 4.5%")
        return 0.045, warnings


# ---------------------------------------------------------------------------
# Public: compute_bond_price_analytics
# ---------------------------------------------------------------------------

async def compute_bond_price_analytics(
    face_value: float = 1000.0,
    coupon_rate: float = 0.05,
    years_to_maturity: float = 10.0,
    yield_to_maturity: Optional[float] = None,
    frequency: int = 2,
    credit_spread_bps: float = 0.0,
) -> BondPriceResult:
    """
    Compute comprehensive bond price analytics.

    If yield_to_maturity is None, the Treasury spot rate is fetched from
    the FRED-backed yield curve and interpolated at years_to_maturity.
    credit_spread_bps is added on top of the risk-free rate.
    """
    warnings: list[str] = []

    # -----------------------------------------------------------------------
    # Determine YTM
    # -----------------------------------------------------------------------
    if yield_to_maturity is None:
        rf_rate, curve_warns = await _fetch_ytm_from_curve(years_to_maturity)
        warnings.extend(curve_warns)
        ytm = rf_rate + credit_spread_bps / 10_000.0
    else:
        ytm = yield_to_maturity + credit_spread_bps / 10_000.0

    if ytm <= 0:
        warnings.append(f"Effective YTM {ytm:.4%} is non-positive; results may be unreliable")

    if frequency not in (1, 2, 4, 12):
        warnings.append(f"Unusual coupon frequency {frequency}; common values are 1, 2, 4, 12")

    # -----------------------------------------------------------------------
    # Price per $100 face (normalised)
    # -----------------------------------------------------------------------
    dirty_per100 = _bond_price_per100(coupon_rate, ytm, years_to_maturity, frequency)
    # Assume issuance date — zero accrued interest
    accrued_interest = 0.0
    clean_per100 = dirty_per100 - accrued_interest

    # -----------------------------------------------------------------------
    # Duration and convexity (computed on $100 face for normalisation)
    # -----------------------------------------------------------------------
    mac_dur = _macaulay_duration(100.0, coupon_rate, ytm, years_to_maturity, frequency)
    mod_dur = mac_dur / (1.0 + ytm / frequency)

    # Effective duration via central finite difference on price (1bp = 0.0001)
    bump = 0.0001
    p_up   = _bond_price_per100(coupon_rate, ytm + bump, years_to_maturity, frequency)
    p_down = _bond_price_per100(coupon_rate, ytm - bump, years_to_maturity, frequency)
    base_p = dirty_per100
    if abs(base_p) > 1e-12:
        eff_dur = (p_down - p_up) / (2.0 * base_p * bump)
    else:
        eff_dur = 0.0
        warnings.append("Bond price is near zero; effective duration unreliable")

    conv = _convexity(100.0, coupon_rate, ytm, years_to_maturity, frequency)

    # -----------------------------------------------------------------------
    # DV01 per $1M face
    # DV01 = modified_duration * (dirty_price / 100) * face_amount * 0.0001
    # For $1M: = modified_duration * (dirty_per100 / 100) * 1_000_000 * 0.0001
    #         = modified_duration * dirty_per100 * 10_000 / 1_000_000 * 1_000_000
    #         = modified_duration * dirty_per100 * 10_000 / 1e6 * 1e6
    # Simplified: mod_dur * (dirty_per100/100) * 1e6 * 0.0001
    # -----------------------------------------------------------------------
    dv01 = mod_dur * (dirty_per100 / 100.0) * 1_000_000.0 * 0.0001

    # -----------------------------------------------------------------------
    # Scenario analysis: yield shifts of ±100bps and ±200bps
    # Express as % change in clean price from base
    # -----------------------------------------------------------------------
    def _pct_change(shift_bps: float) -> float:
        shifted_ytm = ytm + shift_bps / 10_000.0
        new_price = _bond_price_per100(coupon_rate, shifted_ytm, years_to_maturity, frequency)
        if abs(clean_per100) > 1e-12:
            return (new_price - clean_per100) / clean_per100 * 100.0
        return 0.0

    scenarios: dict[str, float] = {
        "-200bps": round(_pct_change(-200), 4),
        "-100bps": round(_pct_change(-100), 4),
        "+100bps": round(_pct_change(+100), 4),
        "+200bps": round(_pct_change(+200), 4),
    }

    return BondPriceResult(
        face_value=face_value,
        coupon_rate=coupon_rate,
        years_to_maturity=years_to_maturity,
        yield_to_maturity=ytm,
        frequency=frequency,
        clean_price=round(clean_per100, 6),
        dirty_price=round(dirty_per100, 6),
        accrued_interest=round(accrued_interest, 6),
        macaulay_duration=round(mac_dur, 6),
        modified_duration=round(mod_dur, 6),
        effective_duration=round(eff_dur, 6),
        convexity=round(conv, 6),
        dv01=round(dv01, 2),
        scenarios=scenarios,
        credit_spread_bps=credit_spread_bps,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal: yfinance data fetch for Altman Z
# ---------------------------------------------------------------------------

def _safe_loc(df, key: str, col_idx: int = 0) -> Optional[float]:
    """
    Try to retrieve a scalar from a pandas DataFrame row by index label.
    Returns None on KeyError or IndexError.
    """
    try:
        val = df.loc[key].iloc[col_idx]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return float(val)
    except (KeyError, IndexError, TypeError):
        return None


def _fetch_yfinance_data(ticker: str) -> dict:
    """Synchronous yfinance calls — run inside asyncio.to_thread."""
    tk = yf.Ticker(ticker)
    return {
        "balance_sheet": tk.balance_sheet,
        "income_stmt": tk.income_stmt,
        "info": tk.info,
    }


# ---------------------------------------------------------------------------
# Public: compute_altman_z
# ---------------------------------------------------------------------------

async def compute_altman_z(ticker: str) -> AltmanZResult:
    """
    Compute Altman Z-score for a publicly traded company.

    Uses Altman (1968) original model for public manufacturing firms:
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Classification:
        Z > 2.99  → "safe"
        1.81–2.99 → "grey"
        Z < 1.81  → "distress"

    Data fetched from yfinance (free). Missing line items are warned and
    treated as 0.0 (conservative, will push score toward distress).
    """
    ticker = ticker.upper().strip()
    warnings: list[str] = []

    # -----------------------------------------------------------------------
    # Fetch yfinance data in thread pool
    # -----------------------------------------------------------------------
    try:
        data = await asyncio.to_thread(_fetch_yfinance_data, ticker)
    except Exception as exc:
        warnings.append(f"yfinance fetch failed: {exc}")
        data = {"balance_sheet": None, "income_stmt": None, "info": {}}

    bs  = data.get("balance_sheet")
    inc = data.get("income_stmt")
    info = data.get("info") or {}

    # -----------------------------------------------------------------------
    # Helper: extract with fallbacks
    # -----------------------------------------------------------------------
    def _bs(key: str, fallback_keys: list[str] | None = None) -> Optional[float]:
        if bs is None:
            return None
        val = _safe_loc(bs, key)
        if val is None and fallback_keys:
            for fk in fallback_keys:
                val = _safe_loc(bs, fk)
                if val is not None:
                    break
        return val

    def _inc(key: str, fallback_keys: list[str] | None = None) -> Optional[float]:
        if inc is None:
            return None
        val = _safe_loc(inc, key)
        if val is None and fallback_keys:
            for fk in fallback_keys:
                val = _safe_loc(inc, fk)
                if val is not None:
                    break
        return val

    # -----------------------------------------------------------------------
    # Retrieve line items
    # -----------------------------------------------------------------------
    current_assets      = _bs("Current Assets")
    current_liabilities = _bs("Current Liabilities")
    total_assets        = _bs("Total Assets")
    retained_earnings   = _bs("Retained Earnings")
    total_liabilities   = _bs(
        "Total Liabilities Net Minority Interest",
        ["Total Liabilities"],
    )
    ebit = _inc("EBIT", ["Operating Income", "Operating Income or Loss"])
    revenue = _inc("Total Revenue", ["Revenue"])
    market_cap = info.get("marketCap")

    # -----------------------------------------------------------------------
    # Warn and default missing values
    # -----------------------------------------------------------------------
    def _require(val: Optional[float], name: str) -> float:
        if val is None:
            warnings.append(f"Missing '{name}' from yfinance; treated as 0.0 (conservative)")
            return 0.0
        return val

    current_assets      = _require(current_assets,      "Current Assets")
    current_liabilities = _require(current_liabilities, "Current Liabilities")
    total_assets        = _require(total_assets,         "Total Assets")
    retained_earnings   = _require(retained_earnings,    "Retained Earnings")
    total_liabilities   = _require(total_liabilities,   "Total Liabilities Net Minority Interest")
    ebit                = _require(ebit,                 "EBIT")
    revenue             = _require(revenue,              "Total Revenue")

    if market_cap is None:
        warnings.append("Missing 'marketCap' from yfinance info; treated as 0.0 (conservative)")
        market_cap = 0.0
    else:
        market_cap = float(market_cap)

    # Guard against zero total assets (prevents division by zero)
    if abs(total_assets) < 1.0:
        warnings.append("Total Assets is zero or near-zero; Z-score is unreliable")
        total_assets = 1.0

    # -----------------------------------------------------------------------
    # Compute Altman Z ratios
    # -----------------------------------------------------------------------
    working_capital = current_assets - current_liabilities

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities if abs(total_liabilities) > 1.0 else 0.0
    x5 = revenue / total_assets

    if abs(total_liabilities) <= 1.0:
        warnings.append(
            "Total Liabilities is zero or near-zero; X4 (market_cap / liabilities) set to 0.0"
        )

    z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    # -----------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------
    if z_score > 2.99:
        classification = "safe"
    elif z_score >= 1.81:
        classification = "grey"
    else:
        classification = "distress"

    # Probability of distress: sigmoid centred at the distress threshold (1.81)
    # Higher Z → lower probability of distress
    prob_distress = 1.0 / (1.0 + math.exp(z_score - 1.81))

    return AltmanZResult(
        ticker=ticker,
        z_score=round(z_score, 4),
        classification=classification,
        probability_of_distress=round(prob_distress, 4),
        working_capital_to_assets=round(x1, 6),
        retained_earnings_to_assets=round(x2, 6),
        ebit_to_assets=round(x3, 6),
        market_cap_to_book_liabilities=round(x4, 6),
        sales_to_assets=round(x5, 6),
        as_of=date.today().isoformat(),
        data_source="yfinance (balance_sheet, income_stmt, info)",
        warnings=warnings,
    )
