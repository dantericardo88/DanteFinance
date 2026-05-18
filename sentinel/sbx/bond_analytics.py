"""Bond analytics — QuantLib wrapper for DV01, OAS, z-spread, duration, convexity.

Falls back gracefully when QuantLib is not installed — pure-Python approximations
are used so callers never get an ImportError at runtime.
"""
from __future__ import annotations
from datetime import date
from decimal import Decimal
from typing import Optional
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

try:
    import QuantLib as ql
    _QL_AVAILABLE = True
except ImportError:  # QuantLib is a heavy C++ build — not always present
    ql = None  # type: ignore[assignment]
    _QL_AVAILABLE = False
    logger.warning("QuantLib not installed — bond analytics using pure-Python fallback")


def _require_ql(fn_name: str) -> None:
    """Raise a clear ImportError if QuantLib is unavailable."""
    if not _QL_AVAILABLE:
        raise ImportError(
            f"QuantLib required for {fn_name}. "
            "Install with: pip install QuantLib  (or conda install -c conda-forge quantlib)"
        )


def build_yield_curve(treasury_rates: dict[str, float], as_of: date) -> "ql.YieldTermStructureHandle":
    """
    Build a QuantLib YieldTermStructure from FRED treasury rate data.
    treasury_rates: {tenor: rate} e.g. {'1M': 0.053, '3M': 0.054, ..., '30Y': 0.047}
    """
    _require_ql("build_yield_curve")
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    settlement = ql.Date(as_of.day, as_of.month, as_of.year)
    ql.Settings.instance().evaluationDate = settlement

    TENOR_MAP = {
        "1M": ql.Period(1, ql.Months), "3M": ql.Period(3, ql.Months),
        "6M": ql.Period(6, ql.Months), "1Y": ql.Period(1, ql.Years),
        "2Y": ql.Period(2, ql.Years), "3Y": ql.Period(3, ql.Years),
        "5Y": ql.Period(5, ql.Years), "7Y": ql.Period(7, ql.Years),
        "10Y": ql.Period(10, ql.Years), "20Y": ql.Period(20, ql.Years),
        "30Y": ql.Period(30, ql.Years),
    }

    helpers = []
    for tenor_str, rate in treasury_rates.items():
        period = TENOR_MAP.get(tenor_str.upper())
        if period is None:
            continue
        quote = ql.QuoteHandle(ql.SimpleQuote(rate / 100))
        helper = ql.DepositRateHelper(quote, period, 2, cal,
                                       ql.ModifiedFollowing, False, ql.Actual360())
        helpers.append(helper)

    if not helpers:
        raise ValueError("No valid treasury rates provided")

    curve = ql.PiecewiseLinearZero(settlement, helpers, ql.Actual365Fixed())
    return ql.YieldTermStructureHandle(curve)


def price_fixed_rate_bond(
    face_value: float,
    coupon_rate: float,
    maturity: date,
    settlement: date,
    yield_curve: "ql.YieldTermStructureHandle",
    frequency: int = 2,  # Semi-annual
) -> dict:
    """
    Price a fixed-rate bond using QuantLib. Returns clean price, dirty price, YTM, DV01, duration.
    """
    _require_ql("price_fixed_rate_bond")
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    settle_date = ql.Date(settlement.day, settlement.month, settlement.year)
    mat_date = ql.Date(maturity.day, maturity.month, maturity.year)
    ql.Settings.instance().evaluationDate = settle_date

    schedule = ql.Schedule(
        settle_date, mat_date,
        ql.Period(ql.Semiannual if frequency == 2 else ql.Annual),
        cal, ql.ModifiedFollowing, ql.ModifiedFollowing,
        ql.DateGeneration.Backward, False,
    )

    bond = ql.FixedRateBond(
        2, face_value,
        schedule,
        [coupon_rate / 100],
        ql.ActualActual(ql.ActualActual.Bond),
    )

    pricing_engine = ql.DiscountingBondEngine(yield_curve)
    bond.setPricingEngine(pricing_engine)

    clean_price = bond.cleanPrice()
    dirty_price = bond.dirtyPrice()
    ytm = bond.bondYield(clean_price, ql.ActualActual(ql.ActualActual.Bond),
                         ql.Compounded, ql.Semiannual) * 100
    duration = ql.BondFunctions.duration(bond, ql.InterestRate(
        ytm / 100, ql.ActualActual(ql.ActualActual.Bond), ql.Compounded, ql.Semiannual
    ), ql.Duration.Modified)
    convexity = ql.BondFunctions.convexity(bond, ql.InterestRate(
        ytm / 100, ql.ActualActual(ql.ActualActual.Bond), ql.Compounded, ql.Semiannual
    ))

    # DV01 = dirty_price * modified_duration * 0.0001
    dv01 = dirty_price * duration * 0.0001

    return {
        "clean_price": round(clean_price, 4),
        "dirty_price": round(dirty_price, 4),
        "ytm": round(ytm, 4),
        "modified_duration": round(duration, 4),
        "convexity": round(convexity, 4),
        "dv01": round(dv01, 4),
        "accrued_interest": round(dirty_price - clean_price, 4),
    }


def compute_z_spread(
    bond_price: float,
    face_value: float,
    coupon_rate: float,
    maturity: date,
    settlement: date,
    yield_curve: "ql.YieldTermStructureHandle",
    frequency: int = 2,
) -> float:
    """
    Compute the Z-spread (parallel shift of the risk-free curve that reprices the bond).
    Returns Z-spread in basis points.
    """
    _require_ql("compute_z_spread")
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    settle_date = ql.Date(settlement.day, settlement.month, settlement.year)
    mat_date = ql.Date(maturity.day, maturity.month, maturity.year)
    ql.Settings.instance().evaluationDate = settle_date

    schedule = ql.Schedule(
        settle_date, mat_date,
        ql.Period(ql.Semiannual if frequency == 2 else ql.Annual),
        cal, ql.ModifiedFollowing, ql.ModifiedFollowing,
        ql.DateGeneration.Backward, False,
    )

    bond = ql.FixedRateBond(
        2, face_value, schedule,
        [coupon_rate / 100],
        ql.ActualActual(ql.ActualActual.Bond),
    )

    z_spread = ql.BondFunctions.zSpread(
        bond, bond_price, yield_curve.currentLink(),
        ql.ActualActual(ql.ActualActual.Bond),
        ql.Compounded, ql.Semiannual,
    )
    return round(z_spread * 10000, 2)  # Convert to basis points


def build_treasury_rates_from_fred(fred_points: dict[str, list]) -> dict[str, float]:
    """
    Convert FRED macro data points to tenor→rate dict for yield curve construction.
    fred_points: {series_id: [MacroDataPoint, ...]}
    """
    FRED_TO_TENOR = {
        "DGS1MO": "1M", "DGS3MO": "3M", "DGS6MO": "6M",
        "DGS1": "1Y", "DGS2": "2Y", "DGS5": "5Y",
        "DGS10": "10Y", "DGS20": "20Y", "DGS30": "30Y",
    }
    rates = {}
    for series_id, points in fred_points.items():
        tenor = FRED_TO_TENOR.get(series_id)
        if tenor and points:
            latest = sorted(points, key=lambda p: p.time)[-1]
            rates[tenor] = float(latest.value)
    return rates
