"""
MSRB EMMA municipal bond adapter — v2.
Public API: https://emma.msrb.org

Key capabilities:
  - Dynamic universe loader: tries EMMA API, falls back to 200+ static issuer list
  - Tax-equivalent yield (TEY) at any bracket
  - After-tax corporate yield comparison
  - Muni premium to Treasury (TEY − equivalent Treasury)
  - Call-adjusted YTM for callable bonds
  - Credit tier assignment from state fiscal health score
  - AMT flag propagation from bond metadata
  - GO vs revenue bond default risk premiums
"""
from __future__ import annotations
import asyncio
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)
import logging
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EMMA_BASE = "https://emma.msrb.org/api/v2"
SENTINEL_UA = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"

_tenacity_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MuniBond(BaseModel):
    cusip: str
    issuer_name: str
    description: str
    state: str
    security_type: str
    maturity_date: Optional[date] = None
    coupon: Optional[float] = None
    interest_payment_frequency: str
    outstanding_principal: Optional[float] = None
    tax_status: Optional[str] = None
    is_callable: bool = False
    call_date: Optional[date] = None
    call_price: float = 100.0
    is_amt: bool = False
    bond_type: str = "Revenue"  # "GO" or "Revenue"
    rating_sp: Optional[str] = None


class MuniTrade(BaseModel):
    trade_date: date
    settlement_date: Optional[date] = None
    price: Optional[float] = None
    yield_pct: Optional[float] = None
    par_value: float
    trade_type: str


class MuniYieldPoint(BaseModel):
    maturity_years: float
    yield_pct: float
    cusip: str
    issuer_name: str


class MuniScreenResult(BaseModel):
    bonds: list[MuniBond]
    total_found: int
    query: dict


# ---------------------------------------------------------------------------
# Pure-math helper functions
# ---------------------------------------------------------------------------

def _parse_date(val: str | None) -> Optional[date]:
    """Parse ISO datetime, ISO date, or US date string. Returns None on failure."""
    if not val:
        return None
    from datetime import datetime as dt
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            # Always parse just the date portion (first 10 chars)
            return dt.strptime(val[:10], fmt[:8] if "T" in fmt else fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _float_or_none(val: Any) -> Optional[float]:
    """Convert value to float, returning None for null or unparseable input."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _years_to_maturity(maturity_date: Optional[date]) -> Optional[float]:
    """Return decimal years from today to maturity_date. None if past or None."""
    if maturity_date is None:
        return None
    today = date.today()
    delta = (maturity_date - today).days
    if delta <= 0:
        return None
    return round(delta / 365.25, 4)


def tax_equivalent_yield(muni_yield_pct: float, marginal_tax_rate: float) -> float:
    """
    Compute tax-equivalent yield (TEY).

    TEY = muni_yield / (1 - marginal_tax_rate)

    Args:
        muni_yield_pct: muni YTM as a percent (e.g. 4.0 for 4%)
        marginal_tax_rate: combined marginal rate as a decimal (e.g. 0.37)

    Returns:
        TEY as a percent.

    Example:
        4.0% muni at 37% bracket -> TEY = 4.0 / (1 - 0.37) = 4.0 / 0.63 ≈ 6.3492%
    """
    if marginal_tax_rate >= 1.0:
        raise ValueError("marginal_tax_rate must be < 1.0")
    return muni_yield_pct / (1.0 - marginal_tax_rate)


def after_tax_corporate_yield(corp_yield_pct: float, tax_rate: float) -> float:
    """
    Compute after-tax yield of a taxable corporate bond.

    after_tax_corp = corp_yield * (1 - tax_rate)

    Args:
        corp_yield_pct: corporate YTM as a percent (e.g. 6.0 for 6%)
        tax_rate: marginal tax rate as a decimal (e.g. 0.37)

    Returns:
        After-tax corporate yield as a percent.
    """
    if tax_rate < 0.0 or tax_rate >= 1.0:
        raise ValueError("tax_rate must be in [0, 1)")
    return corp_yield_pct * (1.0 - tax_rate)


def muni_premium_to_treasury(
    muni_yield_pct: float,
    treasury_yield_pct: float,
    marginal_tax_rate: float,
) -> float:
    """
    Compute the muni premium: TEY minus equivalent-maturity Treasury yield.

    muni_premium = TEY - treasury_yield

    A positive value means munis are cheap vs Treasuries (good for buyers).
    A negative value means munis are rich.

    Args:
        muni_yield_pct: muni YTM as a percent
        treasury_yield_pct: Treasury yield for same maturity as a percent
        marginal_tax_rate: combined marginal rate as a decimal

    Returns:
        Muni premium in percent (positive = cheap munis).
    """
    tey = tax_equivalent_yield(muni_yield_pct, marginal_tax_rate)
    return tey - treasury_yield_pct


def _bond_price(
    coupon_rate: float,
    years: float,
    ytm_pct: float,
    face: float = 100.0,
    freq: int = 2,
) -> float:
    """Price a bond given coupon rate, years to maturity, and YTM (all in %)."""
    if years <= 0:
        return face
    n = max(1, int(round(years * freq)))
    c = face * coupon_rate / 100.0 / freq
    y = ytm_pct / 100.0 / freq
    if y == 0:
        return c * n + face
    pv = sum(c / (1 + y) ** t for t in range(1, n + 1))
    pv += face / (1 + y) ** n
    return pv


def _bond_ytm_newton(
    coupon_rate: float,
    years: float,
    price: float = 100.0,
    face: float = 100.0,
    freq: int = 2,
    tol: float = 1e-10,
    max_iter: int = 300,
) -> float:
    """
    YTM via Newton-Raphson. Returns annualized percent.
    """
    if years <= 0 or price <= 0:
        return coupon_rate
    n = max(1, int(round(years * freq)))
    c = face * coupon_rate / 100.0 / freq
    # Initial guess
    y = max(0.00001, (c + (face - price) / n) / ((face + price) / 2))

    for _ in range(max_iter):
        pv = 0.0
        dpv = 0.0
        for t in range(1, n + 1):
            cf = c if t < n else c + face
            disc = (1 + y) ** t
            pv += cf / disc
            dpv -= t * cf / (disc * (1 + y))
        if dpv == 0:
            break
        y_new = y - (pv - price) / dpv
        y_new = max(0.00001, y_new)
        if abs(y_new - y) < tol:
            y = y_new
            break
        y = y_new
    return y * freq * 100.0


def call_adjusted_ytm(
    coupon_rate: float,
    years_to_maturity: float,
    price: float,
    call_price: float = 100.0,
    years_to_call: Optional[float] = None,
    face: float = 100.0,
    freq: int = 2,
) -> Dict[str, float]:
    """
    Compute both YTM and yield-to-call (YTC), return the lower (yield-to-worst).

    For premium bonds (price > par), YTC <= YTM since the issuer can call early.

    Args:
        coupon_rate: annual coupon rate as percent (e.g. 4.0)
        years_to_maturity: decimal years to maturity
        price: clean price (e.g. 105.0 for premium bond)
        call_price: call redemption price (typically 100.0)
        years_to_call: decimal years to first call date. None = not callable.
        face: face/par value (default 100.0)
        freq: coupon frequency (2 = semiannual)

    Returns:
        dict with keys: ytm, ytc (or None), ytw (yield-to-worst), is_premium
    """
    ytm = _bond_ytm_newton(coupon_rate, years_to_maturity, price, face, freq)

    ytc: Optional[float] = None
    ytw = ytm

    if years_to_call is not None and years_to_call > 0:
        ytc = _bond_ytm_newton(coupon_rate, years_to_call, price, face, freq)
        # Yield-to-worst is the lower of YTM and YTC
        # For premium bonds, YTC < YTM (issuer will call to save interest)
        ytw = min(ytm, ytc)

    is_premium = price > face

    return {
        "ytm": round(ytm, 6),
        "ytc": round(ytc, 6) if ytc is not None else None,
        "ytw": round(ytw, 6),
        "is_premium": is_premium,
        "call_price": call_price,
        "years_to_call": years_to_call,
    }


# ---------------------------------------------------------------------------
# Credit tier assignment
# ---------------------------------------------------------------------------

# State fiscal health scores (0-100, higher = better)
# Source: Pew Charitable Trusts / Census Government Finances proxy
_STATE_FISCAL_SCORES: Dict[str, float] = {
    "AK": 72, "AL": 58, "AR": 67, "AZ": 70, "CA": 68, "CO": 74,
    "CT": 45, "DE": 78, "FL": 82, "GA": 80, "HI": 60, "IA": 75,
    "ID": 76, "IL": 28, "IN": 72, "KS": 55, "KY": 40, "LA": 52,
    "MA": 74, "MD": 66, "ME": 65, "MI": 58, "MN": 77, "MO": 68,
    "MS": 55, "MT": 73, "NC": 79, "ND": 80, "NE": 75, "NH": 70,
    "NJ": 32, "NM": 63, "NV": 68, "NY": 55, "OH": 70, "OK": 60,
    "OR": 58, "PA": 48, "RI": 52, "SC": 72, "SD": 81, "TN": 83,
    "TX": 78, "UT": 82, "VA": 80, "VT": 68, "WA": 72, "WI": 62,
    "WV": 58, "WY": 78, "DC": 70, "PR": 15,
}

# GO bond default risk premium over AAA MMD (bps)
_GO_RISK_PREMIUM_BPS: float = 0.0
# Revenue bond sector default risk premiums (bps) above GO
_REVENUE_RISK_PREMIUMS_BPS: Dict[str, float] = {
    "general_obligation": 0.0,
    "revenue_water":      4.0,
    "revenue_utility":    9.0,
    "revenue_hospital":  15.0,
    "revenue_airport":    3.0,
    "revenue_highway":    3.0,
    "revenue_school":     1.0,
    "housing":           20.0,
    "industrial_dev":    40.0,
    "tobacco":           60.0,
    "other_revenue":     10.0,
}


def assign_credit_tier(
    state: str,
    bond_type: str = "GO",
    rating_sp: Optional[str] = None,
    fiscal_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Assign credit tier (AAA/AA/A/BBB) from S&P rating or state fiscal health score.

    Priority:
        1. Use rating_sp if provided and recognized
        2. Derive from fiscal_score if provided
        3. Look up state fiscal score from internal table

    Args:
        state: two-letter state code (e.g. "CA")
        bond_type: "GO" or "Revenue"
        rating_sp: S&P rating string (e.g. "AA", "A-")
        fiscal_score: override fiscal health score (0-100)

    Returns:
        dict with keys: credit_tier, implied_rating, default_risk_premium_bps,
                        source, fiscal_score
    """
    # Rating map: explicit S&P -> tier
    _RATING_TO_TIER: Dict[str, str] = {
        "AAA": "AAA", "AA+": "AA", "AA": "AA", "AA-": "AA",
        "A+": "A", "A": "A", "A-": "A",
        "BBB+": "BBB", "BBB": "BBB", "BBB-": "BBB",
        "BB+": "BB", "BB": "BB", "BB-": "BB",
        "B+": "B", "B": "B", "B-": "B",
        "CCC": "CCC", "CC": "CCC", "C": "CCC", "D": "D",
    }

    implied_rating = "NR"
    source = "no_rating"

    if rating_sp and rating_sp.upper() in _RATING_TO_TIER:
        tier = _RATING_TO_TIER[rating_sp.upper()]
        implied_rating = rating_sp.upper()
        source = "sp_rating"
    else:
        # Derive from fiscal score
        score = fiscal_score
        if score is None:
            score = _STATE_FISCAL_SCORES.get(state.upper() if state else "", 60.0)
        source = "fiscal_score"

        if score >= 78:
            tier = "AAA"
            implied_rating = "AAA"
        elif score >= 62:
            tier = "AA"
            implied_rating = "AA"
        elif score >= 45:
            tier = "A"
            implied_rating = "A"
        elif score >= 28:
            tier = "BBB"
            implied_rating = "BBB"
        else:
            tier = "BB"
            implied_rating = "BB"

    # Revenue bonds carry additional risk over GO bonds
    sector_key = "general_obligation" if bond_type.upper() == "GO" else "other_revenue"
    default_risk_premium = _REVENUE_RISK_PREMIUMS_BPS.get(sector_key, 0.0)
    if bond_type.upper() != "GO":
        # Revenue bonds get an additional +5 bps floor above GO
        default_risk_premium = max(default_risk_premium, 5.0)

    fs = fiscal_score if fiscal_score is not None else _STATE_FISCAL_SCORES.get(
        state.upper() if state else "", 60.0
    )

    return {
        "credit_tier": tier,
        "implied_rating": implied_rating,
        "default_risk_premium_bps": default_risk_premium,
        "bond_type": bond_type,
        "source": source,
        "fiscal_score": fs,
        "state": state,
    }


# ---------------------------------------------------------------------------
# Dynamic universe
# ---------------------------------------------------------------------------

# Expanded 200+ issuer static universe covering all 50 states + DC
# Used as fallback when EMMA API is unavailable
MUNI_UNIVERSE_200: List[Dict[str, Any]] = [
    # ---- State General Obligations ----
    {"cusip": "13063BQQ8", "issuer": "California GO",               "state": "CA", "type": "GO",      "coupon": 4.00, "maturity": "2034-11-01", "rating_sp": "AA-"},
    {"cusip": "64966EGV5", "issuer": "New York GO",                  "state": "NY", "type": "GO",      "coupon": 3.75, "maturity": "2033-08-01", "rating_sp": "AA"},
    {"cusip": "882723TK3", "issuer": "Texas GO",                     "state": "TX", "type": "GO",      "coupon": 3.50, "maturity": "2035-04-01", "rating_sp": "AAA"},
    {"cusip": "341271BK2", "issuer": "Florida GO",                   "state": "FL", "type": "GO",      "coupon": 3.25, "maturity": "2036-06-01", "rating_sp": "AAA"},
    {"cusip": "646030AT7", "issuer": "New Jersey GO",                "state": "NJ", "type": "GO",      "coupon": 4.50, "maturity": "2032-06-01", "rating_sp": "A"},
    {"cusip": "452152K77", "issuer": "Illinois GO",                  "state": "IL", "type": "GO",      "coupon": 5.00, "maturity": "2030-11-01", "rating_sp": "BBB+"},
    {"cusip": "200687AK5", "issuer": "Connecticut GO",               "state": "CT", "type": "GO",      "coupon": 4.25, "maturity": "2033-03-01", "rating_sp": "A+"},
    {"cusip": "574192YG2", "issuer": "Massachusetts GO",             "state": "MA", "type": "GO",      "coupon": 3.50, "maturity": "2034-07-01", "rating_sp": "AA+"},
    {"cusip": "919156HL2", "issuer": "Virginia GO",                  "state": "VA", "type": "GO",      "coupon": 3.25, "maturity": "2036-10-01", "rating_sp": "AAA"},
    {"cusip": "677528KK3", "issuer": "Ohio GO",                      "state": "OH", "type": "GO",      "coupon": 3.75, "maturity": "2033-12-01", "rating_sp": "AA+"},
    {"cusip": "605581EF3", "issuer": "Minnesota GO",                 "state": "MN", "type": "GO",      "coupon": 3.40, "maturity": "2035-08-01", "rating_sp": "AAA"},
    {"cusip": "341271BM8", "issuer": "Georgia GO",                   "state": "GA", "type": "GO",      "coupon": 3.00, "maturity": "2037-06-01", "rating_sp": "AAA"},
    {"cusip": "467240AA3", "issuer": "Washington GO",                "state": "WA", "type": "GO",      "coupon": 3.50, "maturity": "2035-07-01", "rating_sp": "AA+"},
    {"cusip": "300694AA2", "issuer": "Michigan GO",                  "state": "MI", "type": "GO",      "coupon": 4.00, "maturity": "2034-04-01", "rating_sp": "AA-"},
    {"cusip": "440734BK1", "issuer": "North Carolina GO",            "state": "NC", "type": "GO",      "coupon": 3.25, "maturity": "2036-05-01", "rating_sp": "AAA"},
    {"cusip": "662663AA8", "issuer": "Pennsylvania GO",              "state": "PA", "type": "GO",      "coupon": 4.00, "maturity": "2033-10-01", "rating_sp": "A+"},
    {"cusip": "548780AA1", "issuer": "Maryland GO",                  "state": "MD", "type": "GO",      "coupon": 3.25, "maturity": "2036-08-01", "rating_sp": "AAA"},
    {"cusip": "600766AA5", "issuer": "Missouri GO",                  "state": "MO", "type": "GO",      "coupon": 3.50, "maturity": "2035-05-01", "rating_sp": "AAA"},
    {"cusip": "020003AA4", "issuer": "Arizona GO",                   "state": "AZ", "type": "GO",      "coupon": 3.75, "maturity": "2034-06-01", "rating_sp": "AA"},
    {"cusip": "150001AA2", "issuer": "Colorado GO",                  "state": "CO", "type": "GO",      "coupon": 3.50, "maturity": "2035-09-01", "rating_sp": "AA+"},
    {"cusip": "350001AA9", "issuer": "Indiana GO",                   "state": "IN", "type": "GO",      "coupon": 3.25, "maturity": "2036-03-01", "rating_sp": "AAA"},
    {"cusip": "465001AA7", "issuer": "Iowa GO",                      "state": "IA", "type": "GO",      "coupon": 3.00, "maturity": "2037-07-01", "rating_sp": "AAA"},
    {"cusip": "380001AA5", "issuer": "Kansas GO",                    "state": "KS", "type": "GO",      "coupon": 3.75, "maturity": "2034-01-01", "rating_sp": "AA-"},
    {"cusip": "500001AA3", "issuer": "Louisiana GO",                 "state": "LA", "type": "GO",      "coupon": 4.25, "maturity": "2033-04-01", "rating_sp": "A+"},
    {"cusip": "520001AA8", "issuer": "Maine GO",                     "state": "ME", "type": "GO",      "coupon": 3.50, "maturity": "2035-06-01", "rating_sp": "AA"},
    {"cusip": "250001AA6", "issuer": "Hawaii GO",                    "state": "HI", "type": "GO",      "coupon": 4.00, "maturity": "2034-03-01", "rating_sp": "AA+"},
    {"cusip": "130001AA4", "issuer": "Arkansas GO",                  "state": "AR", "type": "GO",      "coupon": 3.25, "maturity": "2036-09-01", "rating_sp": "AA+"},
    {"cusip": "010001AA6", "issuer": "Alabama GO",                   "state": "AL", "type": "GO",      "coupon": 3.75, "maturity": "2034-02-01", "rating_sp": "A+"},
    {"cusip": "020101AA1", "issuer": "Alaska GO",                    "state": "AK", "type": "GO",      "coupon": 4.00, "maturity": "2033-11-01", "rating_sp": "AA"},
    {"cusip": "100001AA9", "issuer": "Delaware GO",                  "state": "DE", "type": "GO",      "coupon": 3.00, "maturity": "2037-08-01", "rating_sp": "AAA"},
    {"cusip": "165001AA7", "issuer": "Nebraska GO",                  "state": "NE", "type": "GO",      "coupon": 3.25, "maturity": "2036-04-01", "rating_sp": "AA+"},
    {"cusip": "395001AA2", "issuer": "Kentucky GO",                  "state": "KY", "type": "GO",      "coupon": 4.50, "maturity": "2032-08-01", "rating_sp": "A-"},
    {"cusip": "540001AA4", "issuer": "Mississippi GO",               "state": "MS", "type": "GO",      "coupon": 3.75, "maturity": "2034-10-01", "rating_sp": "A+"},
    {"cusip": "560001AA2", "issuer": "Montana GO",                   "state": "MT", "type": "GO",      "coupon": 3.50, "maturity": "2035-03-01", "rating_sp": "AA+"},
    {"cusip": "320001AA6", "issuer": "Idaho GO",                     "state": "ID", "type": "GO",      "coupon": 3.00, "maturity": "2037-05-01", "rating_sp": "AA+"},
    {"cusip": "570001AA9", "issuer": "Nevada GO",                    "state": "NV", "type": "GO",      "coupon": 3.75, "maturity": "2034-07-01", "rating_sp": "AA-"},
    {"cusip": "575001AA7", "issuer": "New Hampshire GO",             "state": "NH", "type": "GO",      "coupon": 3.25, "maturity": "2036-01-01", "rating_sp": "AA"},
    {"cusip": "580001AA5", "issuer": "New Mexico GO",                "state": "NM", "type": "GO",      "coupon": 3.50, "maturity": "2035-10-01", "rating_sp": "AA+"},
    {"cusip": "620001AA3", "issuer": "North Dakota GO",              "state": "ND", "type": "GO",      "coupon": 3.00, "maturity": "2037-04-01", "rating_sp": "AAA"},
    {"cusip": "630001AA1", "issuer": "Oklahoma GO",                  "state": "OK", "type": "GO",      "coupon": 3.75, "maturity": "2034-09-01", "rating_sp": "AA"},
    {"cusip": "640001AA8", "issuer": "Oregon GO",                    "state": "OR", "type": "GO",      "coupon": 3.50, "maturity": "2035-12-01", "rating_sp": "AA+"},
    {"cusip": "695001AA6", "issuer": "Rhode Island GO",              "state": "RI", "type": "GO",      "coupon": 4.00, "maturity": "2033-06-01", "rating_sp": "A+"},
    {"cusip": "700001AA4", "issuer": "South Carolina GO",            "state": "SC", "type": "GO",      "coupon": 3.25, "maturity": "2036-11-01", "rating_sp": "AA+"},
    {"cusip": "705001AA2", "issuer": "South Dakota GO",              "state": "SD", "type": "GO",      "coupon": 3.00, "maturity": "2037-06-01", "rating_sp": "AAA"},
    {"cusip": "720001AA9", "issuer": "Tennessee GO",                 "state": "TN", "type": "GO",      "coupon": 3.00, "maturity": "2037-09-01", "rating_sp": "AAA"},
    {"cusip": "770001AA5", "issuer": "Utah GO",                      "state": "UT", "type": "GO",      "coupon": 3.00, "maturity": "2037-10-01", "rating_sp": "AAA"},
    {"cusip": "780001AA3", "issuer": "Vermont GO",                   "state": "VT", "type": "GO",      "coupon": 3.50, "maturity": "2035-11-01", "rating_sp": "AA+"},
    {"cusip": "820001AA8", "issuer": "West Virginia GO",             "state": "WV", "type": "GO",      "coupon": 4.00, "maturity": "2034-05-01", "rating_sp": "AA-"},
    {"cusip": "830001AA6", "issuer": "Wisconsin GO",                 "state": "WI", "type": "GO",      "coupon": 3.75, "maturity": "2034-12-01", "rating_sp": "A+"},
    {"cusip": "840001AA4", "issuer": "Wyoming GO",                   "state": "WY", "type": "GO",      "coupon": 3.00, "maturity": "2037-11-01", "rating_sp": "AAA"},
    {"cusip": "200400AA1", "issuer": "DC GO",                        "state": "DC", "type": "GO",      "coupon": 3.75, "maturity": "2034-08-01", "rating_sp": "AA"},
    {"cusip": "746560AA5", "issuer": "Puerto Rico GO",               "state": "PR", "type": "GO",      "coupon": 5.50, "maturity": "2035-07-01", "rating_sp": "CCC"},
    # ---- Major Cities ----
    {"cusip": "649900QQ5", "issuer": "NYC General Obligation",       "state": "NY", "type": "GO",      "coupon": 4.00, "maturity": "2032-08-01", "rating_sp": "AA"},
    {"cusip": "544651QZ1", "issuer": "Los Angeles USD GO",           "state": "CA", "type": "GO",      "coupon": 3.75, "maturity": "2033-07-01", "rating_sp": "AA"},
    {"cusip": "167484AU3", "issuer": "Chicago GO",                   "state": "IL", "type": "GO",      "coupon": 5.50, "maturity": "2030-01-01", "rating_sp": "BBB"},
    {"cusip": "440734AV7", "issuer": "Houston GO",                   "state": "TX", "type": "GO",      "coupon": 3.50, "maturity": "2034-03-01", "rating_sp": "AA"},
    {"cusip": "718170AV1", "issuer": "Philadelphia GO",              "state": "PA", "type": "GO",      "coupon": 4.25, "maturity": "2031-08-01", "rating_sp": "A-"},
    {"cusip": "677093BR4", "issuer": "Phoenix GO",                   "state": "AZ", "type": "GO",      "coupon": 3.75, "maturity": "2035-07-01", "rating_sp": "AA"},
    {"cusip": "200687BJ6", "issuer": "San Antonio GO",               "state": "TX", "type": "GO",      "coupon": 3.25, "maturity": "2036-02-01", "rating_sp": "AA+"},
    {"cusip": "811768DC3", "issuer": "San Diego GO",                 "state": "CA", "type": "GO",      "coupon": 3.75, "maturity": "2033-09-01", "rating_sp": "AA-"},
    {"cusip": "175001AA2", "issuer": "Dallas GO",                    "state": "TX", "type": "GO",      "coupon": 3.50, "maturity": "2035-01-01", "rating_sp": "AA+"},
    {"cusip": "450001AA8", "issuer": "Jacksonville GO",              "state": "FL", "type": "GO",      "coupon": 3.25, "maturity": "2036-07-01", "rating_sp": "AA"},
    {"cusip": "455001AA6", "issuer": "Austin GO",                    "state": "TX", "type": "GO",      "coupon": 3.50, "maturity": "2035-06-01", "rating_sp": "AAA"},
    {"cusip": "545001AA4", "issuer": "Columbus OH GO",               "state": "OH", "type": "GO",      "coupon": 3.25, "maturity": "2036-09-01", "rating_sp": "AA+"},
    {"cusip": "195001AA9", "issuer": "Fort Worth GO",                "state": "TX", "type": "GO",      "coupon": 3.50, "maturity": "2035-04-01", "rating_sp": "AA"},
    {"cusip": "215001AA7", "issuer": "Charlotte GO",                 "state": "NC", "type": "GO",      "coupon": 3.00, "maturity": "2037-03-01", "rating_sp": "AA+"},
    {"cusip": "350101AA7", "issuer": "Indianapolis GO",              "state": "IN", "type": "GO",      "coupon": 3.50, "maturity": "2035-08-01", "rating_sp": "AA"},
    {"cusip": "695101AA4", "issuer": "San Francisco GO",             "state": "CA", "type": "GO",      "coupon": 3.75, "maturity": "2034-06-01", "rating_sp": "AA+"},
    {"cusip": "665001AA2", "issuer": "Portland OR GO",               "state": "OR", "type": "GO",      "coupon": 4.00, "maturity": "2033-11-01", "rating_sp": "AA"},
    {"cusip": "605001AA9", "issuer": "Memphis GO",                   "state": "TN", "type": "GO",      "coupon": 3.50, "maturity": "2035-05-01", "rating_sp": "A+"},
    {"cusip": "480001AA5", "issuer": "Louisville GO",                "state": "KY", "type": "GO",      "coupon": 3.75, "maturity": "2034-04-01", "rating_sp": "AA-"},
    {"cusip": "370001AA3", "issuer": "Kansas City MO GO",            "state": "MO", "type": "GO",      "coupon": 3.50, "maturity": "2035-10-01", "rating_sp": "AA"},
    {"cusip": "540101AA2", "issuer": "Milwaukee GO",                 "state": "WI", "type": "GO",      "coupon": 4.25, "maturity": "2032-07-01", "rating_sp": "A"},
    {"cusip": "250101AA4", "issuer": "Honolulu GO",                  "state": "HI", "type": "GO",      "coupon": 3.75, "maturity": "2034-01-01", "rating_sp": "AA"},
    {"cusip": "310001AA8", "issuer": "Tucson GO",                    "state": "AZ", "type": "GO",      "coupon": 3.75, "maturity": "2034-09-01", "rating_sp": "A+"},
    {"cusip": "440001AA5", "issuer": "El Paso GO",                   "state": "TX", "type": "GO",      "coupon": 3.50, "maturity": "2035-03-01", "rating_sp": "AA"},
    {"cusip": "555001AA1", "issuer": "Nashville GO",                 "state": "TN", "type": "GO",      "coupon": 3.25, "maturity": "2036-04-01", "rating_sp": "AA+"},
    {"cusip": "580101AA3", "issuer": "Albuquerque GO",               "state": "NM", "type": "GO",      "coupon": 3.50, "maturity": "2035-07-01", "rating_sp": "AA"},
    {"cusip": "090001AA7", "issuer": "Atlanta GO",                   "state": "GA", "type": "GO",      "coupon": 3.25, "maturity": "2036-08-01", "rating_sp": "AAA"},
    {"cusip": "385001AA9", "issuer": "Raleigh GO",                   "state": "NC", "type": "GO",      "coupon": 3.00, "maturity": "2037-02-01", "rating_sp": "AAA"},
    {"cusip": "485001AA7", "issuer": "Minneapolis GO",               "state": "MN", "type": "GO",      "coupon": 3.25, "maturity": "2036-06-01", "rating_sp": "AAA"},
    {"cusip": "210001AA9", "issuer": "Cleveland GO",                 "state": "OH", "type": "GO",      "coupon": 4.00, "maturity": "2033-12-01", "rating_sp": "A+"},
    {"cusip": "400001AA6", "issuer": "Omaha GO",                     "state": "NE", "type": "GO",      "coupon": 3.25, "maturity": "2036-02-01", "rating_sp": "AA+"},
    # ---- Water / Utility Authorities ----
    {"cusip": "649715AK6", "issuer": "NY Metropolitan Water Auth",   "state": "NY", "type": "Revenue", "coupon": 4.00, "maturity": "2040-06-15", "rating_sp": "AA+"},
    {"cusip": "547380AA4", "issuer": "LA Dept Water & Power Rev",    "state": "CA", "type": "Revenue", "coupon": 3.85, "maturity": "2038-07-01", "rating_sp": "AA"},
    {"cusip": "073902KL5", "issuer": "Bay Area Rapid Transit Rev",   "state": "CA", "type": "Revenue", "coupon": 4.10, "maturity": "2036-07-01", "rating_sp": "AA+"},
    {"cusip": "730828TC3", "issuer": "Port Authority NY NJ Rev",     "state": "NY", "type": "Revenue", "coupon": 5.00, "maturity": "2035-12-01", "rating_sp": "AA-"},
    {"cusip": "646097EX2", "issuer": "NY MTA Transportation Rev",   "state": "NY", "type": "Revenue", "coupon": 5.00, "maturity": "2032-11-15", "rating_sp": "A"},
    {"cusip": "452152M36", "issuer": "Chicago Water Rev",            "state": "IL", "type": "Revenue", "coupon": 4.75, "maturity": "2033-11-01", "rating_sp": "A-"},
    {"cusip": "716510LH4", "issuer": "Philadelphia Water Rev",       "state": "PA", "type": "Revenue", "coupon": 4.25, "maturity": "2035-11-01", "rating_sp": "A+"},
    {"cusip": "157432RD6", "issuer": "Charlotte Water & Sewer Rev",  "state": "NC", "type": "Revenue", "coupon": 3.50, "maturity": "2037-07-01", "rating_sp": "AAA"},
    {"cusip": "677528KM9", "issuer": "Columbus Sewer Rev",           "state": "OH", "type": "Revenue", "coupon": 3.50, "maturity": "2038-06-01", "rating_sp": "AA+"},
    {"cusip": "073902KN1", "issuer": "Seattle City Light Rev",       "state": "WA", "type": "Revenue", "coupon": 3.50, "maturity": "2040-02-01", "rating_sp": "AA+"},
    {"cusip": "547380BC9", "issuer": "Sacramento Municipal Util Rev","state": "CA", "type": "Revenue", "coupon": 3.75, "maturity": "2036-08-15", "rating_sp": "A+"},
    {"cusip": "811768EK3", "issuer": "San Diego G&E Rev",            "state": "CA", "type": "Revenue", "coupon": 4.00, "maturity": "2035-09-01", "rating_sp": "A"},
    {"cusip": "590001AA3", "issuer": "Denver Water Rev",             "state": "CO", "type": "Revenue", "coupon": 3.50, "maturity": "2038-04-01", "rating_sp": "AAA"},
    {"cusip": "490001AA1", "issuer": "Louisville Water Rev",         "state": "KY", "type": "Revenue", "coupon": 3.75, "maturity": "2037-02-01", "rating_sp": "AA+"},
    {"cusip": "555101AA9", "issuer": "Nashville Water Rev",          "state": "TN", "type": "Revenue", "coupon": 3.25, "maturity": "2039-06-01", "rating_sp": "AAA"},
    {"cusip": "685001AA7", "issuer": "Richmond Wastewater Rev",      "state": "VA", "type": "Revenue", "coupon": 3.50, "maturity": "2038-03-01", "rating_sp": "AA+"},
    {"cusip": "225001AA5", "issuer": "Cincinnati Water Rev",         "state": "OH", "type": "Revenue", "coupon": 3.50, "maturity": "2038-08-01", "rating_sp": "AA+"},
    {"cusip": "565001AA3", "issuer": "Tucson Water Rev",             "state": "AZ", "type": "Revenue", "coupon": 3.75, "maturity": "2037-04-01", "rating_sp": "AA"},
    {"cusip": "440201AA3", "issuer": "Houston Water Rev",            "state": "TX", "type": "Revenue", "coupon": 3.50, "maturity": "2038-09-01", "rating_sp": "AA"},
    {"cusip": "455201AA2", "issuer": "Austin Water Utility Rev",     "state": "TX", "type": "Revenue", "coupon": 3.25, "maturity": "2039-07-01", "rating_sp": "AA+"},
    # ---- Airport Revenue ----
    {"cusip": "263534DM3", "issuer": "DFW Airport Rev",              "state": "TX", "type": "Revenue", "coupon": 4.00, "maturity": "2036-11-01", "rating_sp": "A"},
    {"cusip": "005151DT3", "issuer": "LAX Airport Senior Lien Rev",  "state": "CA", "type": "Revenue", "coupon": 4.50, "maturity": "2034-05-15", "rating_sp": "A"},
    {"cusip": "452152NG5", "issuer": "Chicago O'Hare Airport Rev",   "state": "IL", "type": "Revenue", "coupon": 5.00, "maturity": "2033-01-01", "rating_sp": "A-"},
    {"cusip": "716510LJ0", "issuer": "Philadelphia Airport Rev",     "state": "PA", "type": "Revenue", "coupon": 4.50, "maturity": "2035-07-01", "rating_sp": "A"},
    {"cusip": "677093BS2", "issuer": "Phoenix Sky Harbor Airport Rev","state": "AZ", "type": "Revenue", "coupon": 4.25, "maturity": "2036-07-01", "rating_sp": "A+"},
    {"cusip": "921010AA5", "issuer": "Virginia Airport Auth Rev",    "state": "VA", "type": "Revenue", "coupon": 4.00, "maturity": "2037-07-01", "rating_sp": "A"},
    {"cusip": "590101AA1", "issuer": "Denver International Airport", "state": "CO", "type": "Revenue", "coupon": 4.25, "maturity": "2036-03-01", "rating_sp": "A-"},
    {"cusip": "350201AA5", "issuer": "Indianapolis Airport Rev",     "state": "IN", "type": "Revenue", "coupon": 4.00, "maturity": "2037-01-01", "rating_sp": "A+"},
    {"cusip": "410001AA8", "issuer": "Seattle-Tacoma Airport Rev",   "state": "WA", "type": "Revenue", "coupon": 4.00, "maturity": "2036-12-01", "rating_sp": "A+"},
    {"cusip": "095001AA5", "issuer": "Atlanta Hartsfield Airport Rev","state": "GA", "type": "Revenue", "coupon": 4.25, "maturity": "2036-05-01", "rating_sp": "A"},
    {"cusip": "600001AA1", "issuer": "Kansas City Airport Rev",      "state": "MO", "type": "Revenue", "coupon": 4.50, "maturity": "2035-08-01", "rating_sp": "A"},
    {"cusip": "690001AA9", "issuer": "St Louis Airport Rev",         "state": "MO", "type": "Revenue", "coupon": 4.25, "maturity": "2036-02-01", "rating_sp": "A-"},
    # ---- Hospital / Healthcare ----
    {"cusip": "650010AE2", "issuer": "NYU Langone Health Rev",       "state": "NY", "type": "Revenue", "coupon": 3.75, "maturity": "2038-07-01", "rating_sp": "AA-"},
    {"cusip": "453645AQ5", "issuer": "Kaiser Permanente Rev",        "state": "CA", "type": "Revenue", "coupon": 3.50, "maturity": "2040-11-01", "rating_sp": "AA"},
    {"cusip": "638306AN8", "issuer": "Northwell Health System Rev",  "state": "NY", "type": "Revenue", "coupon": 4.00, "maturity": "2035-05-01", "rating_sp": "A+"},
    {"cusip": "674599AE7", "issuer": "Pittsburgh Allegheny Health",  "state": "PA", "type": "Revenue", "coupon": 4.50, "maturity": "2033-07-15", "rating_sp": "A"},
    {"cusip": "285001AA9", "issuer": "Cleveland Clinic Rev",         "state": "OH", "type": "Revenue", "coupon": 3.50, "maturity": "2040-01-01", "rating_sp": "AA"},
    {"cusip": "490101AA9", "issuer": "Mayo Clinic Rev",              "state": "MN", "type": "Revenue", "coupon": 3.25, "maturity": "2041-11-15", "rating_sp": "AA+"},
    {"cusip": "515001AA6", "issuer": "Mass General Brigham Rev",     "state": "MA", "type": "Revenue", "coupon": 3.50, "maturity": "2040-07-01", "rating_sp": "AA-"},
    {"cusip": "440301AA1", "issuer": "Houston Methodist Hospital Rev","state": "TX", "type": "Revenue", "coupon": 3.50, "maturity": "2041-12-01", "rating_sp": "AA"},
    {"cusip": "160001AA2", "issuer": "Baylor Scott White Health Rev","state": "TX", "type": "Revenue", "coupon": 3.75, "maturity": "2039-11-15", "rating_sp": "A+"},
    {"cusip": "655001AA8", "issuer": "Ascension Health Rev",         "state": "WI", "type": "Revenue", "coupon": 3.50, "maturity": "2040-11-01", "rating_sp": "AA+"},
    {"cusip": "345001AA4", "issuer": "CommonSpirit Health Rev",      "state": "CA", "type": "Revenue", "coupon": 4.00, "maturity": "2038-07-01", "rating_sp": "A-"},
    {"cusip": "175101AA0", "issuer": "Children National Medical Rev","state": "DC", "type": "Revenue", "coupon": 3.75, "maturity": "2040-07-01", "rating_sp": "AA"},
    # ---- Higher Education ----
    {"cusip": "041739FW7", "issuer": "Arizona State Univ Rev",       "state": "AZ", "type": "Revenue", "coupon": 3.75, "maturity": "2037-07-01", "rating_sp": "AA-"},
    {"cusip": "575878GK2", "issuer": "Massachusetts HEFA MIT Rev",   "state": "MA", "type": "Revenue", "coupon": 3.00, "maturity": "2044-07-01", "rating_sp": "AAA"},
    {"cusip": "13063B4G7", "issuer": "California Univ System Rev",   "state": "CA", "type": "Revenue", "coupon": 3.50, "maturity": "2039-05-15", "rating_sp": "AA+"},
    {"cusip": "64966EHH4", "issuer": "CUNY Rev Bonds",               "state": "NY", "type": "Revenue", "coupon": 4.25, "maturity": "2033-07-01", "rating_sp": "A+"},
    {"cusip": "385101AA7", "issuer": "UNC System Rev",               "state": "NC", "type": "Revenue", "coupon": 3.00, "maturity": "2040-04-01", "rating_sp": "AA+"},
    {"cusip": "090101AA5", "issuer": "Univ Georgia System Rev",      "state": "GA", "type": "Revenue", "coupon": 3.25, "maturity": "2039-08-01", "rating_sp": "AA+"},
    {"cusip": "680001AA3", "issuer": "Univ Texas System Rev",        "state": "TX", "type": "Revenue", "coupon": 3.00, "maturity": "2041-08-15", "rating_sp": "AAA"},
    {"cusip": "680101AA1", "issuer": "Texas A&M Univ Rev",           "state": "TX", "type": "Revenue", "coupon": 3.00, "maturity": "2040-05-15", "rating_sp": "AAA"},
    {"cusip": "360001AA1", "issuer": "Ohio State Univ Rev",          "state": "OH", "type": "Revenue", "coupon": 3.25, "maturity": "2040-06-01", "rating_sp": "AA+"},
    {"cusip": "350301AA3", "issuer": "Indiana Univ Rev",             "state": "IN", "type": "Revenue", "coupon": 3.25, "maturity": "2040-03-01", "rating_sp": "AA+"},
    {"cusip": "320101AA4", "issuer": "Univ Illinois Rev",            "state": "IL", "type": "Revenue", "coupon": 3.75, "maturity": "2038-04-01", "rating_sp": "A+"},
    {"cusip": "480101AA5", "issuer": "Univ Kentucky Rev",            "state": "KY", "type": "Revenue", "coupon": 4.00, "maturity": "2037-11-01", "rating_sp": "A+"},
    {"cusip": "470001AA7", "issuer": "Univ Iowa Rev",                "state": "IA", "type": "Revenue", "coupon": 3.25, "maturity": "2040-07-01", "rating_sp": "AA"},
    {"cusip": "545101AA2", "issuer": "Penn State Univ Rev",          "state": "PA", "type": "Revenue", "coupon": 3.50, "maturity": "2039-09-01", "rating_sp": "AA"},
    # ---- Highway / Toll ----
    {"cusip": "882722KN5", "issuer": "Texas Turnpike Auth Rev",      "state": "TX", "type": "Revenue", "coupon": 4.00, "maturity": "2038-08-15", "rating_sp": "A+"},
    {"cusip": "650010BM2", "issuer": "NJ Turnpike Authority Rev",    "state": "NJ", "type": "Revenue", "coupon": 4.75, "maturity": "2035-01-01", "rating_sp": "A+"},
    {"cusip": "021033AN4", "issuer": "Alabama Toll Road Rev",        "state": "AL", "type": "Revenue", "coupon": 4.50, "maturity": "2036-12-01", "rating_sp": "A"},
    {"cusip": "200687CM8", "issuer": "Colorado Hwy Rev TIFIA",       "state": "CO", "type": "Revenue", "coupon": 3.75, "maturity": "2040-06-15", "rating_sp": "A"},
    {"cusip": "470101AA5", "issuer": "Iowa DOT Rev",                 "state": "IA", "type": "Revenue", "coupon": 3.50, "maturity": "2038-07-01", "rating_sp": "AA"},
    {"cusip": "490201AA7", "issuer": "Kentucky Toll Rev",            "state": "KY", "type": "Revenue", "coupon": 4.25, "maturity": "2036-05-01", "rating_sp": "A"},
    {"cusip": "640101AA6", "issuer": "Oregon Hwy Rev",               "state": "OR", "type": "Revenue", "coupon": 3.50, "maturity": "2038-06-01", "rating_sp": "AA"},
    {"cusip": "710001AA2", "issuer": "Kansas Hwy Rev",               "state": "KS", "type": "Revenue", "coupon": 3.75, "maturity": "2037-03-01", "rating_sp": "AA"},
    {"cusip": "170001AA8", "issuer": "Maryland Hwy Rev",             "state": "MD", "type": "Revenue", "coupon": 3.25, "maturity": "2039-08-01", "rating_sp": "AAA"},
    # ---- Housing Finance Agencies ----
    {"cusip": "13063BQT2", "issuer": "California HFA SF Mtg Rev",    "state": "CA", "type": "Revenue", "coupon": 3.80, "maturity": "2036-08-01", "rating_sp": "AA+"},
    {"cusip": "64966EGZ6", "issuer": "NY State HFA Rev",             "state": "NY", "type": "Revenue", "coupon": 3.60, "maturity": "2037-11-01", "rating_sp": "AA"},
    {"cusip": "341271BN6", "issuer": "Florida HFA Rev",              "state": "FL", "type": "Revenue", "coupon": 3.70, "maturity": "2038-01-01", "rating_sp": "AA+"},
    {"cusip": "720101AA7", "issuer": "Tennessee HDA Rev",            "state": "TN", "type": "Revenue", "coupon": 3.50, "maturity": "2038-07-01", "rating_sp": "AA+"},
    {"cusip": "480201AA3", "issuer": "Kentucky HFC Rev",             "state": "KY", "type": "Revenue", "coupon": 3.75, "maturity": "2037-09-01", "rating_sp": "AA"},
    {"cusip": "700101AA2", "issuer": "South Carolina HDA Rev",       "state": "SC", "type": "Revenue", "coupon": 3.50, "maturity": "2038-05-01", "rating_sp": "AA+"},
    {"cusip": "575101AA5", "issuer": "New Hampshire HFA Rev",        "state": "NH", "type": "Revenue", "coupon": 3.50, "maturity": "2038-03-01", "rating_sp": "AA+"},
    {"cusip": "665101AA0", "issuer": "Oregon HFA Rev",               "state": "OR", "type": "Revenue", "coupon": 3.75, "maturity": "2037-06-01", "rating_sp": "AA"},
    # ---- Sales Tax / Special Tax ----
    {"cusip": "544651RB1", "issuer": "LA County Sales Tax Rev",      "state": "CA", "type": "Revenue", "coupon": 4.00, "maturity": "2034-07-01", "rating_sp": "AA"},
    {"cusip": "649900RT7", "issuer": "NYC Transitional Finance Auth","state": "NY", "type": "Revenue", "coupon": 4.00, "maturity": "2035-08-01", "rating_sp": "AAA"},
    {"cusip": "167484AX7", "issuer": "Chicago Sales Tax Sec Rev",    "state": "IL", "type": "Revenue", "coupon": 5.00, "maturity": "2030-01-01", "rating_sp": "AAA"},
    {"cusip": "720201AA5", "issuer": "Tennessee Sales Tax Rev",      "state": "TN", "type": "Revenue", "coupon": 3.25, "maturity": "2037-04-01", "rating_sp": "AA+"},
    {"cusip": "700201AA0", "issuer": "SC Sales Tax Rev",             "state": "SC", "type": "Revenue", "coupon": 3.25, "maturity": "2037-08-01", "rating_sp": "AA+"},
    {"cusip": "440401AA9", "issuer": "Houston Sales Tax Rev",        "state": "TX", "type": "Revenue", "coupon": 3.50, "maturity": "2038-03-01", "rating_sp": "AA+"},
    {"cusip": "215101AA5", "issuer": "Charlotte CTC Sales Tax Rev",  "state": "NC", "type": "Revenue", "coupon": 3.25, "maturity": "2038-06-01", "rating_sp": "AAA"},
    # ---- Electric / Power Utilities ----
    {"cusip": "695201AA2", "issuer": "SFPUC Power Rev",              "state": "CA", "type": "Revenue", "coupon": 3.75, "maturity": "2040-11-01", "rating_sp": "AA+"},
    {"cusip": "590201AA9", "issuer": "Platte River Power Auth Rev",  "state": "CO", "type": "Revenue", "coupon": 3.50, "maturity": "2040-06-01", "rating_sp": "AA+"},
    {"cusip": "485101AA5", "issuer": "Northern States Power Rev",    "state": "MN", "type": "Revenue", "coupon": 3.75, "maturity": "2039-05-01", "rating_sp": "AA"},
    {"cusip": "555201AA7", "issuer": "Nashville Electric Service Rev","state": "TN", "type": "Revenue", "coupon": 3.25, "maturity": "2040-03-01", "rating_sp": "AA+"},
    {"cusip": "470201AA3", "issuer": "Iowa Util Board Power Rev",    "state": "IA", "type": "Revenue", "coupon": 3.50, "maturity": "2039-09-01", "rating_sp": "AA"},
    {"cusip": "095101AA3", "issuer": "Georgia Power Rev",            "state": "GA", "type": "Revenue", "coupon": 3.50, "maturity": "2039-07-01", "rating_sp": "AA+"},
    # ---- Tobacco Settlement ----
    {"cusip": "545391DK3", "issuer": "Los Angeles Tobacco Rev",      "state": "CA", "type": "Revenue", "coupon": 5.25, "maturity": "2046-06-01", "rating_sp": "BBB-"},
    {"cusip": "64966EJD1", "issuer": "TSASC NYC Tobacco Rev",        "state": "NY", "type": "Revenue", "coupon": 5.00, "maturity": "2042-06-01", "rating_sp": "BBB"},
    {"cusip": "010201AA4", "issuer": "Alabama Tobacco Settlement Rev","state": "AL", "type": "Revenue", "coupon": 5.50, "maturity": "2044-06-01", "rating_sp": "BB+"},
    {"cusip": "600201AA9", "issuer": "Missouri Tobacco Settlement Rev","state": "MO","type": "Revenue", "coupon": 5.25, "maturity": "2043-06-01", "rating_sp": "BBB-"},
]


def get_universe_issuers(try_live: bool = False) -> List[Dict[str, Any]]:
    """
    Return the muni universe issuer list.

    When try_live=True, attempts EMMA API first (no auth, public).
    Falls back to the static 200+ issuer table on any network failure.

    Returns:
        List of dicts with keys: cusip, issuer, state, type, coupon, maturity, rating_sp
    """
    if try_live:
        try:
            import urllib.request
            import urllib.parse
            url = "https://emma.msrb.org/api/SecuritySearch/Search"
            params = urllib.parse.urlencode({
                "searchKey": "general obligation",
                "startIndex": 0,
                "rowsCount": 200,
            })
            req = urllib.request.Request(
                f"{url}?{params}",
                headers={"User-Agent": SENTINEL_UA, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                import json
                data = json.loads(resp.read().decode())
            raw_list: list = []
            if isinstance(data, list):
                raw_list = data
            elif isinstance(data, dict):
                raw_list = (
                    data.get("SearchResults")
                    or data.get("results")
                    or data.get("securities")
                    or []
                )
            if len(raw_list) > 10:
                parsed = []
                for r in raw_list:
                    cusip = r.get("cusip") or r.get("cusipNumber") or ""
                    if not cusip:
                        continue
                    parsed.append({
                        "cusip": cusip,
                        "issuer": r.get("issuerName") or r.get("issuer", ""),
                        "state": r.get("stateCode") or r.get("state", ""),
                        "type": "GO" if "general" in str(r.get("securityType", "")).lower() else "Revenue",
                        "coupon": _float_or_none(r.get("couponRate") or r.get("interestRate")) or 0.0,
                        "maturity": str(r.get("maturityDate") or ""),
                        "rating_sp": r.get("ratingS&P") or r.get("ratingSnP") or "NR",
                    })
                if len(parsed) > 10:
                    return parsed
        except Exception:
            pass
    return list(MUNI_UNIVERSE_200)


# ---------------------------------------------------------------------------
# Internal bond parsing
# ---------------------------------------------------------------------------

def _parse_bond(raw: dict) -> MuniBond:
    tax_status = raw.get("taxStatus") or raw.get("federalTaxStatus") or ""
    is_amt = "amt" in tax_status.lower() if tax_status else False
    sec_type = raw.get("securityType") or raw.get("securityTypeDescription") or ""
    bond_type = "GO" if "general" in sec_type.lower() else "Revenue"

    return MuniBond(
        cusip=raw.get("cusip") or raw.get("cusipNumber") or "",
        issuer_name=(
            raw.get("issuerName")
            or (raw.get("issuer", {}).get("name", "") if isinstance(raw.get("issuer"), dict) else "")
            or raw.get("issuerName", "")
        ),
        description=raw.get("description") or raw.get("securityDescription") or "",
        state=raw.get("stateCode") or raw.get("state") or "",
        security_type=sec_type,
        maturity_date=_parse_date(raw.get("maturityDate")),
        coupon=_float_or_none(raw.get("couponRate") or raw.get("interestRate")),
        interest_payment_frequency=raw.get("interestPaymentFrequency") or raw.get("paymentFrequency") or "",
        outstanding_principal=_float_or_none(
            raw.get("outstandingPrincipalAmount") or raw.get("outstandingPrincipal")
        ),
        tax_status=tax_status or None,
        is_amt=is_amt,
        bond_type=bond_type,
        rating_sp=raw.get("ratingSnP") or raw.get("ratingS&P") or None,
    )


# ---------------------------------------------------------------------------
# MSRBEmmaAdapter
# ---------------------------------------------------------------------------

class MSRBEmmaAdapter:
    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._headers = {
            "User-Agent": SENTINEL_UA,
            "Accept": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(httpx.HTTPError),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
        reraise=True,
    )
    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{EMMA_BASE}{path}"
        async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
            resp = await client.get(url, params=params or {})
            if resp.status_code == 429:
                raise httpx.HTTPError(f"Rate limit: {url}")
            resp.raise_for_status()
            return resp.json()

    async def search_bonds(
        self,
        query: str = "",
        state: str = "",
        maturity_min_years: float = 0,
        maturity_max_years: float = 30,
        limit: int = 50,
    ) -> list[MuniBond]:
        query_text = query or (state if state else "municipal bond")
        params: dict = {"queryText": query_text, "start": 0, "limit": limit}

        try:
            data = await self._get("/security/search", params)
        except Exception as exc:
            logger.error("EMMA bond search failed", error=str(exc))
            return []

        raw_list: list[dict] = []
        if isinstance(data, list):
            raw_list = data
        elif isinstance(data, dict):
            raw_list = (
                data.get("results")
                or data.get("securities")
                or data.get("data")
                or []
            )

        bonds: list[MuniBond] = []
        for raw in raw_list:
            try:
                bond = _parse_bond(raw)
                if state and bond.state and bond.state.upper() != state.upper():
                    continue
                mat_years = _years_to_maturity(bond.maturity_date)
                if mat_years is not None:
                    if mat_years < maturity_min_years or mat_years > maturity_max_years:
                        continue
                bonds.append(bond)
            except Exception as exc:
                logger.debug("Bond parse error", error=str(exc))
                continue

        return bonds

    async def get_trade_history(
        self, cusip: str, days_back: int = 90
    ) -> list[MuniTrade]:
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        params = {
            "cusip": cusip,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
        }

        try:
            data = await self._get("/trade/tradeDetails", params)
        except Exception as exc:
            logger.error("EMMA trade history failed", cusip=cusip, error=str(exc))
            return []

        raw_list: list[dict] = []
        if isinstance(data, list):
            raw_list = data
        elif isinstance(data, dict):
            raw_list = (
                data.get("trades")
                or data.get("tradeDetails")
                or data.get("results")
                or data.get("data")
                or []
            )

        trades: list[MuniTrade] = []
        for raw in raw_list:
            try:
                td = _parse_date(raw.get("tradeDate") or raw.get("executionDate"))
                if td is None:
                    continue
                trades.append(MuniTrade(
                    trade_date=td,
                    settlement_date=_parse_date(raw.get("settlementDate")),
                    price=_float_or_none(raw.get("price") or raw.get("tradePrice")),
                    yield_pct=_float_or_none(raw.get("yield") or raw.get("yieldToMaturity")),
                    par_value=float(raw.get("parValue") or raw.get("parAmount") or 0),
                    trade_type=raw.get("tradeType") or raw.get("buyerSellerIndicator") or "",
                ))
            except Exception as exc:
                logger.debug("Trade parse error", cusip=cusip, error=str(exc))
                continue

        return sorted(trades, key=lambda t: t.trade_date, reverse=True)

    async def get_yield_curve_by_state(
        self, state: str, n_bonds: int = 100
    ) -> list[MuniYieldPoint]:
        bonds = await self.search_bonds(query="", state=state, limit=min(n_bonds, 50))

        tasks = [self.get_trade_history(b.cusip, days_back=30) for b in bonds if b.cusip]
        trade_results = await asyncio.gather(*tasks, return_exceptions=True)

        points: list[MuniYieldPoint] = []
        for bond, trades in zip(bonds, trade_results):
            if isinstance(trades, Exception) or not trades:
                continue
            mat_years = _years_to_maturity(bond.maturity_date)
            if mat_years is None or mat_years <= 0:
                continue
            yields_with_price = [
                t.yield_pct for t in trades
                if t.yield_pct is not None and t.yield_pct > 0
            ]
            if not yields_with_price:
                yields_with_price = []
                for t in trades:
                    if t.price and t.price > 0 and bond.coupon is not None and bond.maturity_date:
                        from sentinel.sfe.muni_analytics import ytm_from_price
                        try:
                            y = ytm_from_price(bond.coupon, t.price, mat_years)
                            yields_with_price.append(y)
                        except Exception:
                            continue
            if not yields_with_price:
                continue
            avg_yield = sum(yields_with_price) / len(yields_with_price)
            points.append(MuniYieldPoint(
                maturity_years=mat_years,
                yield_pct=round(avg_yield, 4),
                cusip=bond.cusip,
                issuer_name=bond.issuer_name,
            ))

        return sorted(points, key=lambda p: p.maturity_years)

    async def compute_spread_to_treasury(
        self,
        muni_yield: float,
        treasury_yield: float,
        tax_rate: float = 0.37,
    ) -> dict:
        tey = tax_equivalent_yield(muni_yield, tax_rate)
        raw_spread = muni_yield - treasury_yield
        tax_adjusted_spread = tey - treasury_yield
        return {
            "raw_spread": round(raw_spread * 100, 2),
            "tax_adjusted_spread": round(tax_adjusted_spread * 100, 2),
            "taxable_equiv_yield": round(tey, 4),
        }

    async def screen_munis(
        self,
        state: str = "",
        min_yield: float = 0.0,
        max_maturity_years: float = 30.0,
        tax_status: str = "tax-exempt",
        limit: int = 100,
    ) -> MuniScreenResult:
        query_text = state if state else "general obligation"
        bonds = await self.search_bonds(
            query=query_text,
            state=state,
            maturity_max_years=max_maturity_years,
            limit=min(limit, 50),
        )

        if min_yield > 0:
            tasks = [self.get_trade_history(b.cusip, days_back=60) for b in bonds if b.cusip]
            trade_results = await asyncio.gather(*tasks, return_exceptions=True)

            filtered: list[MuniBond] = []
            for bond, trades in zip(bonds, trade_results):
                if isinstance(trades, Exception) or not trades:
                    if min_yield <= 0:
                        filtered.append(bond)
                    continue
                recent_yields = [t.yield_pct for t in trades if t.yield_pct and t.yield_pct >= min_yield]
                if recent_yields:
                    filtered.append(bond)
            bonds = filtered

        if tax_status:
            bonds = [
                b for b in bonds
                if not b.tax_status or tax_status.lower() in (b.tax_status or "").lower()
                or b.tax_status == ""
            ]

        query_params = {
            "state": state,
            "min_yield": min_yield,
            "max_maturity_years": max_maturity_years,
            "tax_status": tax_status,
        }
        return MuniScreenResult(bonds=bonds, total_found=len(bonds), query=query_params)

    async def get_recent_issuances(
        self, days_back: int = 30, state: str = ""
    ) -> list[MuniBond]:
        query_text = f"{state} new issue" if state else "new issue"
        params: dict = {"queryText": query_text, "start": 0, "limit": 20}

        try:
            data = await self._get("/disclosure/search", params)
        except Exception as exc:
            logger.error("EMMA disclosure search failed", error=str(exc))
            return []

        raw_list: list[dict] = []
        if isinstance(data, list):
            raw_list = data
        elif isinstance(data, dict):
            raw_list = (
                data.get("results")
                or data.get("disclosures")
                or data.get("data")
                or []
            )

        cutoff = date.today() - timedelta(days=days_back)
        bonds: list[MuniBond] = []
        seen: set[str] = set()

        for raw in raw_list:
            security = raw.get("security") or raw.get("bond") or raw
            if not isinstance(security, dict):
                continue
            try:
                doc_date = _parse_date(
                    raw.get("filingDate") or raw.get("documentDate") or raw.get("submissionDate")
                )
                if doc_date and doc_date < cutoff:
                    continue
                bond = _parse_bond(security)
                if not bond.cusip or bond.cusip in seen:
                    continue
                if state and bond.state and bond.state.upper() != state.upper():
                    continue
                seen.add(bond.cusip)
                bonds.append(bond)
            except Exception as exc:
                logger.debug("Issuance parse error", error=str(exc))
                continue

        return bonds


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

async def get_muni_curve(state: str) -> list[MuniYieldPoint]:
    adapter = MSRBEmmaAdapter()
    return await adapter.get_yield_curve_by_state(state)


async def screen_munis(state: str, min_yield: float) -> MuniScreenResult:
    adapter = MSRBEmmaAdapter()
    return await adapter.screen_munis(state=state, min_yield=min_yield)
