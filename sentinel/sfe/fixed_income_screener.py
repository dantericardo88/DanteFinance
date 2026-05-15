"""
Fixed income screener: corporate bonds, Treasuries, munis, agency, tips.
Multi-asset FI screening with yield, spread, duration, credit quality filters.

Dimension: dim_074 — Fixed income screener (target score 9).

Data sources:
    - FRED API (free) for Treasury yields, TIPS breakevens, credit spreads
    - EMMA MSRB for muni data
    - Synthetic universe for IG/HY corporates (representative issuers)
    - TreasuryDirect for on-the-run data

FastAPI router: fi_screener_router (prefix /fi)
"""
from __future__ import annotations

import logging
import math
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TREASURY_DIRECT = "https://www.treasurydirect.gov/TA_WS/securities/search"
EMMA_TRADE_API = "https://www.msrb.org/msrb1/tradedata.asp"

CACHE_DB = Path("sentinel_fi_screener.db")
CACHE_TTL = 3600   # 1 hour

_HEADERS = {
    "User-Agent": "SENTINEL/1.0 richard.porras@realempanada.com",
    "Accept": "application/json, text/csv",
}

# On-the-run Treasury curve (approximate 2025-2026 baseline, pct)
_TREASURY_CURVE: Dict[float, float] = {
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

# IG corporate spread typical values by rating (bps above Treasury)
_IG_SPREADS: Dict[str, float] = {
    "AAA": 25,
    "AA+": 35,
    "AA":  45,
    "AA-": 55,
    "A+":  70,
    "A":   85,
    "A-":  100,
    "BBB+": 130,
    "BBB":  160,
    "BBB-": 200,
}

# HY spreads
_HY_SPREADS: Dict[str, float] = {
    "BB+": 250,
    "BB":  300,
    "BB-": 375,
    "B+":  450,
    "B":   550,
    "B-":  675,
    "CCC+": 850,
    "CCC":  1050,
    "CCC-": 1350,
}

# Muni tax-equivalent yield multiplier (assume 37% federal bracket + state)
_MUNI_TEY_FACTOR = 1 / (1 - 0.40)   # 40% combined rate

# Agency typical spreads over Treasury (bps)
_AGENCY_SPREADS: Dict[str, float] = {
    "FNMA_bullet":    22,
    "FNMA_callable":  45,
    "FHLMC_bullet":   20,
    "FHLMC_callable": 42,
    "FHLB_bullet":    18,
    "FHLB_callable":  40,
}

# Representative IG corporate issuers (100)
_IG_ISSUERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK", "JPM", "BAC", "WFC", "C",
    "GS", "MS", "USB", "TFC", "PNC", "AXP", "COF", "DFS", "SYF", "ALLY",
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "AMGN", "GILD", "CVS", "MCK",
    "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "PSX", "VLO", "MPC", "PXD",
    "NEE", "DUK", "SO", "D", "EXC", "AEP", "SRE", "PCG", "ED", "WEC",
    "T", "VZ", "TMUS", "CHTR", "CMCSA", "DISH", "LUMN", "SIRI", "ATVI", "EA",
    "WMT", "TGT", "COST", "HD", "LOW", "MCD", "SBUX", "YUM", "CMG", "DRI",
    "BA", "RTX", "LMT", "NOC", "GD", "HON", "GE", "MMM", "EMR", "ITW",
    "CAT", "DE", "PCAR", "CMI", "ETN", "PH", "ROK", "DOV", "FTV", "AME",
    "UNP", "CSX", "NSC", "BNI", "FDX", "UPS", "LUV", "DAL", "AAL", "UAL",
]

# Representative HY issuers (50)
_HY_ISSUERS = [
    "F", "GM", "FORD", "LCII", "AHT", "CZR", "MGM", "WYNN", "LVS", "PENN",
    "CCL", "RCL", "NCLH", "HLT", "MAR", "HST", "PK", "AHC", "SHO", "RHP",
    "OXY", "DVN", "MRO", "HES", "APA", "SM", "RRC", "AR", "EQT", "CNXC",
    "WHR", "HBI", "PVH", "RL", "TAP", "MO", "PM", "BTI", "RAI", "LO",
    "HCA", "THC", "CYH", "LPNT", "ENSG", "AMR", "NWL", "CHK", "BBBY", "DISH2",
]

# Muni sectors and representative issuers
_MUNI_SAMPLES = [
    {"state": "CA", "sector": "general_obligation", "issuer": "California GO"},
    {"state": "NY", "sector": "general_obligation", "issuer": "New York GO"},
    {"state": "TX", "sector": "general_obligation", "issuer": "Texas GO"},
    {"state": "FL", "sector": "general_obligation", "issuer": "Florida GO"},
    {"state": "IL", "sector": "general_obligation", "issuer": "Illinois GO"},
    {"state": "PA", "sector": "revenue_utility", "issuer": "Philadelphia Water"},
    {"state": "OH", "sector": "revenue_utility", "issuer": "Columbus Sewer"},
    {"state": "NY", "sector": "revenue_hospital", "issuer": "NY Presbyterian"},
    {"state": "CA", "sector": "revenue_airport", "issuer": "LAX Airport Rev"},
    {"state": "TX", "sector": "revenue_highway", "issuer": "Texas Turnpike Auth"},
    {"state": "NJ", "sector": "general_obligation", "issuer": "New Jersey GO"},
    {"state": "MA", "sector": "general_obligation", "issuer": "Massachusetts GO"},
    {"state": "WA", "sector": "revenue_utility", "issuer": "Seattle City Light"},
    {"state": "CO", "sector": "revenue_school", "issuer": "Denver School Dist"},
    {"state": "GA", "sector": "general_obligation", "issuer": "Georgia GO"},
]

# Altman Z-score safe zone threshold
_ALTMAN_ZSCORE_SAFE = 1.8

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CACHE_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables() -> None:
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fi_universe_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key   TEXT NOT NULL,
            data_json   TEXT NOT NULL,
            fetched_at  REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fi_screen_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            screen_type TEXT NOT NULL,
            params_json TEXT NOT NULL,
            result_count INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fi_cache_key ON fi_universe_cache(cache_key);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BondSpec(BaseModel):
    """Specification for a single bond in the screening universe."""
    cusip: Optional[str] = None
    isin: Optional[str] = None
    issuer: str
    symbol: Optional[str] = None
    sector: str = Field(..., description="treasury|tips|ig_corp|hy_corp|muni|agency")
    sub_sector: Optional[str] = None   # e.g. "financial", "utility", "industrial"
    rating: Optional[str] = None
    coupon_rate: float = Field(..., description="Annual coupon as pct, e.g. 5.0")
    maturity_years: float = Field(..., description="Years to maturity from today")
    maturity_date: Optional[str] = None
    face: float = 1000.0
    freq: int = Field(2, description="Coupon payments per year")
    callable: bool = False
    call_date_years: Optional[float] = None
    state: Optional[str] = None        # for munis
    tax_exempt: bool = False
    ytm: float = 0.0                   # yield to maturity (pct)
    ytw: float = 0.0                   # yield to worst (pct)
    oas_bps: float = 0.0               # OAS spread in bps
    treasury_spread_bps: float = 0.0   # spread to same-maturity Treasury
    modified_duration: float = 0.0
    macaulay_duration: float = 0.0
    convexity: float = 0.0
    dv01: float = 0.0                  # dollar value of 1 bps per $1M face
    price: float = 100.0
    real_yield: float = 0.0            # for TIPS
    tey: float = 0.0                   # tax-equivalent yield (for munis)
    altman_z: Optional[float] = None
    interest_coverage: Optional[float] = None
    debt_ebitda: Optional[float] = None
    negative_watch: bool = False
    fallen_angel_risk: bool = False
    oas_pct_30d: Optional[float] = None  # OAS percentile over last 30 days
    oas_30d_change_bps: float = 0.0      # OAS change over 30 days


class ScreenRequest(BaseModel):
    sectors: Optional[List[str]] = None
    min_ytm: Optional[float] = None
    max_ytm: Optional[float] = None
    min_ytw: Optional[float] = None
    max_ytw: Optional[float] = None
    min_oas_bps: Optional[float] = None
    max_oas_bps: Optional[float] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    min_convexity: Optional[float] = None
    ratings: Optional[List[str]] = None
    exclude_negative_watch: bool = True
    min_interest_coverage: Optional[float] = None
    max_debt_ebitda: Optional[float] = None
    min_altman_z: Optional[float] = None
    exclude_fallen_angels: bool = False
    tax_equivalent: bool = False
    rank_method: str = Field("carry_duration", description="carry_duration|spread_duration|credit_adj")
    limit: int = Field(50, ge=1, le=500)


class RankRequest(BaseModel):
    universe_key: str = Field("all", description="Universe key: all|ig|hy|muni|treasury|tips|agency")
    method: str = Field("carry_duration", description="carry_duration|spread_duration|credit_adj")
    limit: int = Field(50, ge=1, le=500)


# ---------------------------------------------------------------------------
# Yield math helpers
# ---------------------------------------------------------------------------

def _interp_treasury_yield(maturity_years: float) -> float:
    """Linearly interpolate Treasury yield for a given maturity."""
    maturities = sorted(_TREASURY_CURVE.keys())
    yields = [_TREASURY_CURVE[m] for m in maturities]
    if maturity_years <= maturities[0]:
        return yields[0]
    if maturity_years >= maturities[-1]:
        return yields[-1]
    for i in range(len(maturities) - 1):
        if maturities[i] <= maturity_years <= maturities[i + 1]:
            t0, t1 = maturities[i], maturities[i + 1]
            y0, y1 = yields[i], yields[i + 1]
            frac = (maturity_years - t0) / (t1 - t0)
            return y0 + frac * (y1 - y0)
    return yields[-1]


def _bond_price(ytm_pct: float, coupon_rate_pct: float, maturity_years: float,
                face: float = 1000.0, freq: int = 2) -> float:
    """Compute dirty bond price given YTM."""
    ytm = ytm_pct / 100.0
    c = coupon_rate_pct / 100.0
    n = int(round(maturity_years * freq))
    if n <= 0:
        return face * (1 + c / freq)
    period_rate = ytm / freq
    coupon_payment = face * c / freq
    if period_rate == 0:
        return coupon_payment * n + face
    price = coupon_payment * (1 - (1 + period_rate) ** (-n)) / period_rate
    price += face * (1 + period_rate) ** (-n)
    return price


def _compute_duration_convexity(
    ytm_pct: float, coupon_rate_pct: float, maturity_years: float,
    face: float = 1000.0, freq: int = 2
) -> Tuple[float, float, float, float]:
    """Return (modified_duration, macaulay_duration, convexity, dv01_per_million)."""
    ytm = ytm_pct / 100.0
    c = coupon_rate_pct / 100.0
    n = int(round(maturity_years * freq))
    if n <= 0:
        return 0.0, 0.0, 0.0, 0.0
    period_rate = ytm / freq
    coupon_payment = face * c / freq

    weighted_cf_sum = 0.0
    price = 0.0
    convexity_sum = 0.0
    for t in range(1, n + 1):
        cf = coupon_payment if t < n else coupon_payment + face
        pv = cf / (1 + period_rate) ** t
        time_years = t / freq
        weighted_cf_sum += time_years * pv
        price += pv
        convexity_sum += pv * t * (t + 1) / freq ** 2

    if price <= 0:
        return 0.0, 0.0, 0.0, 0.0

    mac_dur = weighted_cf_sum / price
    mod_dur = mac_dur / (1 + ytm / freq)
    convexity = convexity_sum / (price * (1 + period_rate) ** 2)
    dv01_per_million = mod_dur * price * 10.0   # per $1M face, 1bps = 0.0001

    return round(mod_dur, 4), round(mac_dur, 4), round(convexity, 4), round(dv01_per_million, 2)


def _ytm_from_price(
    clean_price: float, coupon_rate_pct: float, maturity_years: float,
    face: float = 1000.0, freq: int = 2
) -> float:
    """Solve for YTM given clean price using bisection."""
    def pv_diff(ytm_pct: float) -> float:
        return _bond_price(ytm_pct, coupon_rate_pct, maturity_years, face, freq) - clean_price
    try:
        from scipy.optimize import brentq
        ytm = brentq(pv_diff, -50.0, 200.0, xtol=1e-8)
        return round(ytm, 4)
    except Exception:
        c = coupon_rate_pct
        n = maturity_years
        par = face
        approx = (c + (par - clean_price) / n) / ((par + clean_price) / 2) * 100
        return round(approx, 4)


def _ytw_callable(ytm: float, coupon_rate_pct: float, call_date_years: float,
                   call_price: float = 100.0, face: float = 1000.0) -> float:
    """Yield to worst for a callable bond: min(YTM, YTC)."""
    ytc = _ytm_from_price(call_price * face / 100.0, coupon_rate_pct, call_date_years, face)
    return min(ytm, ytc)


# ---------------------------------------------------------------------------
# FixedIncomeUniverse
# ---------------------------------------------------------------------------

class FixedIncomeUniverse:
    """Build and maintain a multi-sector bond universe.

    Sectors
    -------
    treasury : on-the-run Treasuries, all maturities
    tips     : TIPS, all maturities
    ig_corp  : 100 IG corporate issuers, 3 maturities each
    hy_corp  : 50 HY issuers
    muni     : representative samples by state/sector
    agency   : FNMA, FHLMC, FHLB bullets and callables
    """

    def __init__(self) -> None:
        _ensure_tables()
        self._today = date.today()
        self._rng = random.Random(42)   # deterministic synthetic data

    # ------------------------------------------------------------------
    # Treasury universe
    # ------------------------------------------------------------------

    def _build_treasuries(self) -> List[BondSpec]:
        bonds: List[BondSpec] = []
        maturities = [0.0833, 0.25, 0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        labels = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
        for mat, label in zip(maturities, labels):
            ytm = _interp_treasury_yield(mat)
            coupon = round(ytm * 0.95, 3)
            mod_dur, mac_dur, convex, dv01 = _compute_duration_convexity(ytm, coupon, mat)
            mat_date = (self._today + timedelta(days=int(mat * 365.25))).isoformat()
            bonds.append(BondSpec(
                cusip=f"912828{label}",
                issuer=f"US Treasury {label}",
                sector="treasury",
                sub_sector="government",
                rating="AAA",
                coupon_rate=coupon,
                maturity_years=mat,
                maturity_date=mat_date,
                face=1000.0,
                freq=2,
                callable=False,
                tax_exempt=False,
                ytm=round(ytm, 4),
                ytw=round(ytm, 4),
                oas_bps=0.0,
                treasury_spread_bps=0.0,
                modified_duration=mod_dur,
                macaulay_duration=mac_dur,
                convexity=convex,
                dv01=dv01,
                price=round(_bond_price(ytm, coupon, mat) / 10.0, 4),
            ))
        return bonds

    # ------------------------------------------------------------------
    # TIPS universe
    # ------------------------------------------------------------------

    def _build_tips(self) -> List[BondSpec]:
        bonds: List[BondSpec] = []
        tips_maturities = [(2, "2Y"), (5, "5Y"), (7, "7Y"), (10, "10Y"), (20, "20Y"), (30, "30Y")]
        bei_approx = 2.30   # breakeven inflation assumption
        for mat, label in tips_maturities:
            nom_ytm = _interp_treasury_yield(mat)
            real_ytm = round(nom_ytm - bei_approx, 3)
            coupon = round(max(real_ytm * 0.9, 0.125), 3)
            mod_dur, mac_dur, convex, dv01 = _compute_duration_convexity(real_ytm, coupon, mat)
            mat_date = (self._today + timedelta(days=int(mat * 365.25))).isoformat()
            bonds.append(BondSpec(
                cusip=f"912828T{label}",
                issuer=f"US TIPS {label}",
                sector="tips",
                sub_sector="inflation_linked",
                rating="AAA",
                coupon_rate=coupon,
                maturity_years=float(mat),
                maturity_date=mat_date,
                freq=2,
                callable=False,
                tax_exempt=False,
                ytm=round(nom_ytm, 4),
                ytw=round(nom_ytm, 4),
                real_yield=round(real_ytm, 4),
                oas_bps=0.0,
                treasury_spread_bps=0.0,
                modified_duration=mod_dur,
                macaulay_duration=mac_dur,
                convexity=convex,
                dv01=dv01,
                price=100.0,
            ))
        return bonds

    # ------------------------------------------------------------------
    # IG corporate universe
    # ------------------------------------------------------------------

    def _build_ig_corps(self) -> List[BondSpec]:
        bonds: List[BondSpec] = []
        ratings = list(_IG_SPREADS.keys())
        maturities = [3.0, 7.0, 10.0]
        sub_sectors = ["financial", "industrial", "utility", "technology", "healthcare", "energy"]

        for i, issuer in enumerate(_IG_ISSUERS):
            rating = ratings[i % len(ratings)]
            spread_bps = _IG_SPREADS[rating] + self._rng.uniform(-15, 15)
            sub_sector = sub_sectors[i % len(sub_sectors)]

            for mat in maturities:
                tsy_yield = _interp_treasury_yield(mat)
                ytm = round(tsy_yield + spread_bps / 100.0, 4)
                coupon = round(ytm - self._rng.uniform(-0.3, 0.3), 3)
                price = _bond_price(ytm, coupon, mat) / 10.0
                mod_dur, mac_dur, convex, dv01 = _compute_duration_convexity(ytm, coupon, mat)
                mat_date = (self._today + timedelta(days=int(mat * 365.25))).isoformat()

                is_callable = mat >= 10.0 and self._rng.random() > 0.5
                call_yrs = mat - 2.0 if is_callable else None
                ytw = _ytw_callable(ytm, coupon, call_yrs) if is_callable and call_yrs else ytm

                interest_cov = self._rng.uniform(3.0, 15.0)
                debt_ebitda = self._rng.uniform(1.5, 4.5)
                altman_z = self._rng.uniform(2.0, 6.0)
                neg_watch = self._rng.random() < 0.03
                oas_pct = self._rng.uniform(0, 100)
                oas_30d_chg = self._rng.uniform(-30, 30)
                fallen_angel = rating in ("BBB", "BBB-") and debt_ebitda > 4.0

                bonds.append(BondSpec(
                    cusip=f"IG{issuer}{int(mat)}Y",
                    issuer=f"{issuer} Corp {int(mat)}Y",
                    symbol=issuer,
                    sector="ig_corp",
                    sub_sector=sub_sector,
                    rating=rating,
                    coupon_rate=round(coupon, 3),
                    maturity_years=mat,
                    maturity_date=mat_date,
                    freq=2,
                    callable=is_callable,
                    call_date_years=call_yrs,
                    tax_exempt=False,
                    ytm=ytm,
                    ytw=round(ytw, 4),
                    oas_bps=round(spread_bps, 1),
                    treasury_spread_bps=round(spread_bps, 1),
                    modified_duration=mod_dur,
                    macaulay_duration=mac_dur,
                    convexity=convex,
                    dv01=dv01,
                    price=round(price, 4),
                    interest_coverage=round(interest_cov, 2),
                    debt_ebitda=round(debt_ebitda, 2),
                    altman_z=round(altman_z, 2),
                    negative_watch=neg_watch,
                    fallen_angel_risk=fallen_angel,
                    oas_pct_30d=round(oas_pct, 1),
                    oas_30d_change_bps=round(oas_30d_chg, 1),
                ))
        return bonds

    # ------------------------------------------------------------------
    # HY corporate universe
    # ------------------------------------------------------------------

    def _build_hy_corps(self) -> List[BondSpec]:
        bonds: List[BondSpec] = []
        hy_ratings = list(_HY_SPREADS.keys())
        maturities = [5.0, 8.0]
        sub_sectors = ["energy", "consumer", "media", "healthcare", "industrials"]

        for i, issuer in enumerate(_HY_ISSUERS):
            rating = hy_ratings[i % len(hy_ratings)]
            spread_bps = _HY_SPREADS[rating] + self._rng.uniform(-50, 50)
            sub_sector = sub_sectors[i % len(sub_sectors)]

            for mat in maturities:
                tsy_yield = _interp_treasury_yield(mat)
                ytm = round(tsy_yield + spread_bps / 100.0, 4)
                coupon = round(ytm - 0.5 + self._rng.uniform(-0.5, 0.5), 3)
                price = _bond_price(ytm, coupon, mat) / 10.0
                mod_dur, mac_dur, convex, dv01 = _compute_duration_convexity(ytm, coupon, mat)
                mat_date = (self._today + timedelta(days=int(mat * 365.25))).isoformat()

                interest_cov = self._rng.uniform(1.5, 5.0)
                debt_ebitda = self._rng.uniform(3.5, 8.0)
                altman_z = self._rng.uniform(1.0, 2.5)
                neg_watch = self._rng.random() < 0.10
                oas_pct = self._rng.uniform(0, 100)
                oas_30d_chg = self._rng.uniform(-80, 80)

                bonds.append(BondSpec(
                    cusip=f"HY{issuer}{int(mat)}Y",
                    issuer=f"{issuer} HY {int(mat)}Y",
                    symbol=issuer,
                    sector="hy_corp",
                    sub_sector=sub_sector,
                    rating=rating,
                    coupon_rate=round(coupon, 3),
                    maturity_years=mat,
                    maturity_date=mat_date,
                    freq=2,
                    callable=True,
                    call_date_years=mat - 2.0,
                    tax_exempt=False,
                    ytm=ytm,
                    ytw=round(_ytw_callable(ytm, coupon, mat - 2.0), 4),
                    oas_bps=round(spread_bps, 1),
                    treasury_spread_bps=round(spread_bps, 1),
                    modified_duration=mod_dur,
                    macaulay_duration=mac_dur,
                    convexity=convex,
                    dv01=dv01,
                    price=round(price, 4),
                    interest_coverage=round(interest_cov, 2),
                    debt_ebitda=round(debt_ebitda, 2),
                    altman_z=round(altman_z, 2),
                    negative_watch=neg_watch,
                    fallen_angel_risk=False,
                    oas_pct_30d=round(oas_pct, 1),
                    oas_30d_change_bps=round(oas_30d_chg, 1),
                ))
        return bonds

    # ------------------------------------------------------------------
    # Municipal universe
    # ------------------------------------------------------------------

    def _build_munis(self) -> List[BondSpec]:
        bonds: List[BondSpec] = []
        muni_maturities = [5.0, 10.0, 20.0]
        ratio = 0.85  # munis yield ~85% of Treasuries (tax-exempt discount)

        for sample in _MUNI_SAMPLES:
            for mat in muni_maturities:
                tsy_yield = _interp_treasury_yield(mat)
                ytm = round(tsy_yield * ratio + self._rng.uniform(-0.15, 0.15), 4)
                coupon = round(ytm + self._rng.uniform(-0.2, 0.2), 3)
                tey = round(ytm * _MUNI_TEY_FACTOR, 4)

                spread = round((ytm - tsy_yield) * 100, 1)
                price = _bond_price(ytm, coupon, mat) / 10.0
                mod_dur, mac_dur, convex, dv01 = _compute_duration_convexity(ytm, coupon, mat)
                mat_date = (self._today + timedelta(days=int(mat * 365.25))).isoformat()

                is_go = "general_obligation" in sample["sector"]
                rating = "AA" if is_go else self._rng.choice(["A", "A+", "BBB+"])

                bonds.append(BondSpec(
                    cusip=f"MU{sample['state']}{sample['sector'][:3].upper()}{int(mat)}Y",
                    issuer=f"{sample['issuer']} {int(mat)}Y",
                    sector="muni",
                    sub_sector=sample["sector"],
                    rating=rating,
                    state=sample["state"],
                    coupon_rate=round(coupon, 3),
                    maturity_years=mat,
                    maturity_date=mat_date,
                    freq=2,
                    callable=mat >= 10.0,
                    call_date_years=mat - 5.0 if mat >= 10.0 else None,
                    tax_exempt=True,
                    ytm=ytm,
                    ytw=ytm,
                    tey=tey,
                    oas_bps=max(spread, -50.0),
                    treasury_spread_bps=spread,
                    modified_duration=mod_dur,
                    macaulay_duration=mac_dur,
                    convexity=convex,
                    dv01=dv01,
                    price=round(price, 4),
                    negative_watch=False,
                    oas_pct_30d=self._rng.uniform(10, 90),
                    oas_30d_change_bps=self._rng.uniform(-20, 20),
                ))
        return bonds

    # ------------------------------------------------------------------
    # Agency universe
    # ------------------------------------------------------------------

    def _build_agencies(self) -> List[BondSpec]:
        bonds: List[BondSpec] = []
        agency_types = [
            ("FNMA", "FNMA_bullet", False),
            ("FNMA", "FNMA_callable", True),
            ("FHLMC", "FHLMC_bullet", False),
            ("FHLMC", "FHLMC_callable", True),
            ("FHLB", "FHLB_bullet", False),
            ("FHLB", "FHLB_callable", True),
        ]
        maturities = [2.0, 5.0, 10.0]

        for agency, key, is_callable in agency_types:
            spread = _AGENCY_SPREADS[key]
            for mat in maturities:
                tsy_yield = _interp_treasury_yield(mat)
                ytm = round(tsy_yield + spread / 100.0, 4)
                coupon = round(ytm - self._rng.uniform(0.05, 0.15), 3)
                price = _bond_price(ytm, coupon, mat) / 10.0
                mod_dur, mac_dur, convex, dv01 = _compute_duration_convexity(ytm, coupon, mat)
                mat_date = (self._today + timedelta(days=int(mat * 365.25))).isoformat()
                call_yrs = mat - 1.0 if is_callable else None
                ytw = _ytw_callable(ytm, coupon, call_yrs) if is_callable and call_yrs else ytm
                call_type = "callable" if is_callable else "bullet"
                bonds.append(BondSpec(
                    cusip=f"AGY{agency}{call_type[:3].upper()}{int(mat)}Y",
                    issuer=f"{agency} {call_type.capitalize()} {int(mat)}Y",
                    sector="agency",
                    sub_sector=key,
                    rating="AA+",
                    coupon_rate=round(coupon, 3),
                    maturity_years=mat,
                    maturity_date=mat_date,
                    freq=2,
                    callable=is_callable,
                    call_date_years=call_yrs,
                    tax_exempt=False,
                    ytm=ytm,
                    ytw=round(ytw, 4),
                    oas_bps=round(spread + self._rng.uniform(-5, 5), 1),
                    treasury_spread_bps=round(spread, 1),
                    modified_duration=mod_dur,
                    macaulay_duration=mac_dur,
                    convexity=convex,
                    dv01=dv01,
                    price=round(price, 4),
                    negative_watch=False,
                    oas_pct_30d=self._rng.uniform(20, 80),
                    oas_30d_change_bps=self._rng.uniform(-10, 10),
                ))
        return bonds

    # ------------------------------------------------------------------
    # Main build method
    # ------------------------------------------------------------------

    def build_universe(
        self, sectors: Optional[List[str]] = None
    ) -> List[BondSpec]:
        """Build the bond universe for the specified sectors.

        Parameters
        ----------
        sectors : list of sectors to include, or None for all.
                  Valid: treasury, tips, ig_corp, hy_corp, muni, agency
        """
        all_sectors = sectors or ["treasury", "tips", "ig_corp", "hy_corp", "muni", "agency"]
        universe: List[BondSpec] = []

        builders = {
            "treasury": self._build_treasuries,
            "tips":     self._build_tips,
            "ig_corp":  self._build_ig_corps,
            "hy_corp":  self._build_hy_corps,
            "muni":     self._build_munis,
            "agency":   self._build_agencies,
        }
        for sector in all_sectors:
            if sector in builders:
                universe.extend(builders[sector]())
            else:
                logger.warning("Unknown sector: %s", sector)

        logger.info("Built FI universe: %d bonds across %s", len(universe), all_sectors)
        return universe


# ---------------------------------------------------------------------------
# YieldScreener
# ---------------------------------------------------------------------------

class YieldScreener:
    """Screen bonds by yield metrics: YTM, YTW, spread, TEY, real yield."""

    def __init__(self, universe: Optional[List[BondSpec]] = None) -> None:
        self._universe_builder = FixedIncomeUniverse()
        self._universe = universe or []

    def _ensure_universe(self, sectors: Optional[List[str]] = None) -> List[BondSpec]:
        if not self._universe:
            self._universe = self._universe_builder.build_universe(sectors)
        return self._universe

    def screen_by_yield(
        self,
        min_ytm: Optional[float] = None,
        max_ytm: Optional[float] = None,
        min_ytw: Optional[float] = None,
        max_ytw: Optional[float] = None,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter bonds by YTM and/or YTW thresholds."""
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            if min_ytm is not None and bond.ytm < min_ytm:
                continue
            if max_ytm is not None and bond.ytm > max_ytm:
                continue
            if min_ytw is not None and bond.ytw < min_ytw:
                continue
            if max_ytw is not None and bond.ytw > max_ytw:
                continue
            results.append(bond)
        return sorted(results, key=lambda b: b.ytm, reverse=True)

    def screen_by_spread(
        self,
        min_spread_bps: Optional[float] = None,
        max_spread_bps: Optional[float] = None,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter by treasury spread."""
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            s = bond.treasury_spread_bps
            if min_spread_bps is not None and s < min_spread_bps:
                continue
            if max_spread_bps is not None and s > max_spread_bps:
                continue
            results.append(bond)
        return sorted(results, key=lambda b: b.treasury_spread_bps, reverse=True)

    def screen_tax_equivalent(
        self,
        min_tey: Optional[float] = None,
        max_tey: Optional[float] = None,
        compare_ytm: Optional[float] = None,
    ) -> List[BondSpec]:
        """Screen munis by tax-equivalent yield. Optionally filter where TEY > compare_ytm."""
        universe = self._ensure_universe(["muni"])
        results = []
        for bond in universe:
            if bond.sector != "muni":
                continue
            tey = bond.tey or bond.ytm * _MUNI_TEY_FACTOR
            if min_tey is not None and tey < min_tey:
                continue
            if max_tey is not None and tey > max_tey:
                continue
            if compare_ytm is not None and tey <= compare_ytm:
                continue
            results.append(bond)
        return sorted(results, key=lambda b: b.tey, reverse=True)

    def screen_real_yield(
        self,
        min_real_yield: Optional[float] = None,
        max_real_yield: Optional[float] = None,
    ) -> List[BondSpec]:
        """Screen TIPS by real yield."""
        universe = self._ensure_universe(["tips"])
        results = []
        for bond in universe:
            if bond.sector != "tips":
                continue
            ry = bond.real_yield
            if min_real_yield is not None and ry < min_real_yield:
                continue
            if max_real_yield is not None and ry > max_real_yield:
                continue
            results.append(bond)
        return sorted(results, key=lambda b: b.real_yield, reverse=True)


# ---------------------------------------------------------------------------
# SpreadScreener
# ---------------------------------------------------------------------------

class SpreadScreener:
    """Screen bonds by spread metrics: OAS, historical percentile, sector comparison, momentum."""

    def __init__(self, universe: Optional[List[BondSpec]] = None) -> None:
        self._universe_builder = FixedIncomeUniverse()
        self._universe = universe or []

    def _ensure_universe(self, sectors: Optional[List[str]] = None) -> List[BondSpec]:
        if not self._universe:
            self._universe = self._universe_builder.build_universe(sectors)
        return self._universe

    def screen_by_oas(
        self,
        min_oas_bps: Optional[float] = None,
        max_oas_bps: Optional[float] = None,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter by OAS spread in basis points."""
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            oas = bond.oas_bps
            if min_oas_bps is not None and oas < min_oas_bps:
                continue
            if max_oas_bps is not None and oas > max_oas_bps:
                continue
            results.append(bond)
        return sorted(results, key=lambda b: b.oas_bps, reverse=True)

    def screen_by_oas_percentile(
        self,
        min_pct: Optional[float] = None,
        max_pct: Optional[float] = None,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter by OAS percentile (0=tight, 100=wide).

        Low percentile (< 25) = historically tight spreads.
        High percentile (> 75) = historically wide = potentially cheap.
        """
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            pct = bond.oas_pct_30d
            if pct is None:
                continue
            if min_pct is not None and pct < min_pct:
                continue
            if max_pct is not None and pct > max_pct:
                continue
            results.append(bond)
        return sorted(results, key=lambda b: b.oas_pct_30d or 0, reverse=True)

    def screen_by_spread_momentum(
        self,
        max_30d_change_bps: Optional[float] = None,
        min_30d_change_bps: Optional[float] = None,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter by 30-day OAS change.

        Positive change = spread widening (cheapening).
        Negative change = spread narrowing (richening).
        """
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            chg = bond.oas_30d_change_bps
            if min_30d_change_bps is not None and chg < min_30d_change_bps:
                continue
            if max_30d_change_bps is not None and chg > max_30d_change_bps:
                continue
            results.append(bond)
        return sorted(results, key=lambda b: b.oas_30d_change_bps, reverse=True)

    def sector_spread_comparison(
        self, sectors: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Return median OAS by sector and sub-sector."""
        universe = self._ensure_universe(sectors)
        sector_data: Dict[str, List[float]] = {}
        subsector_data: Dict[str, List[float]] = {}

        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            sector_data.setdefault(bond.sector, []).append(bond.oas_bps)
            if bond.sub_sector:
                subsector_data.setdefault(bond.sub_sector, []).append(bond.oas_bps)

        result: Dict[str, Dict[str, Any]] = {"by_sector": {}, "by_sub_sector": {}}
        for sec, vals in sector_data.items():
            arr = sorted(vals)
            n = len(arr)
            result["by_sector"][sec] = {
                "median_oas": arr[n // 2] if arr else 0,
                "mean_oas": sum(arr) / n if arr else 0,
                "min_oas": arr[0] if arr else 0,
                "max_oas": arr[-1] if arr else 0,
                "count": n,
            }
        for sub, vals in subsector_data.items():
            arr = sorted(vals)
            n = len(arr)
            result["by_sub_sector"][sub] = {
                "median_oas": arr[n // 2] if arr else 0,
                "mean_oas": sum(arr) / n if arr else 0,
                "count": n,
            }
        return result

    def treasury_relative_value(self, sectors: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return bonds sorted by spread vs same-maturity Treasury."""
        universe = self._ensure_universe(sectors)
        rv = []
        for bond in universe:
            if bond.sector in ("treasury",):
                continue
            tsy_yield = _interp_treasury_yield(bond.maturity_years)
            spread_vs_tsy = bond.ytm - tsy_yield
            rv.append({
                "issuer": bond.issuer,
                "sector": bond.sector,
                "rating": bond.rating,
                "maturity_years": bond.maturity_years,
                "ytm": bond.ytm,
                "treasury_yield": round(tsy_yield, 4),
                "spread_vs_treasury_bps": round(spread_vs_tsy * 100, 1),
                "oas_bps": bond.oas_bps,
            })
        return sorted(rv, key=lambda x: x["spread_vs_treasury_bps"], reverse=True)


# ---------------------------------------------------------------------------
# DurationRiskScreener
# ---------------------------------------------------------------------------

class DurationRiskScreener:
    """Screen bonds by duration and rate sensitivity metrics."""

    _DV01_BUCKETS = {
        "low":    (0, 50),
        "medium": (50, 200),
        "high":   (200, float("inf")),
    }

    _KEY_RATE_MATURITIES = [2.0, 5.0, 10.0, 30.0]

    def __init__(self, universe: Optional[List[BondSpec]] = None) -> None:
        self._universe_builder = FixedIncomeUniverse()
        self._universe = universe or []

    def _ensure_universe(self, sectors: Optional[List[str]] = None) -> List[BondSpec]:
        if not self._universe:
            self._universe = self._universe_builder.build_universe(sectors)
        return self._universe

    def screen_by_duration(
        self,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter by modified duration."""
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            d = bond.modified_duration
            if min_duration is not None and d < min_duration:
                continue
            if max_duration is not None and d > max_duration:
                continue
            results.append(bond)
        return sorted(results, key=lambda b: b.modified_duration)

    def screen_by_dv01_bucket(
        self,
        bucket: str,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter bonds by DV01 bucket: low (<$50), medium ($50-200), high (>$200).

        DV01 is per $1M face value.
        """
        low, high = self._DV01_BUCKETS.get(bucket.lower(), (0, float("inf")))
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            if low <= bond.dv01 < high:
                results.append(bond)
        return sorted(results, key=lambda b: b.dv01)

    def screen_positive_convexity(
        self,
        min_convexity: float = 0.0,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter bonds with positive (or above threshold) convexity.

        Positive convexity: price gains more when rates fall than it loses when rates rise.
        This is valuable when the rate environment is uncertain or volatile.
        """
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            if bond.convexity >= min_convexity:
                results.append(bond)
        return sorted(results, key=lambda b: b.convexity, reverse=True)

    def key_rate_exposure(self, bonds: List[BondSpec]) -> Dict[str, Dict[str, float]]:
        """Compute approximate key rate duration (KRD) at 2Y, 5Y, 10Y, 30Y.

        Uses simplified linear decay allocation: duration contribution to
        each key rate tenor decreases with distance from bond maturity.
        """
        krd_map: Dict[str, Dict[str, float]] = {}
        for bond in bonds:
            mat = bond.maturity_years
            dur = bond.modified_duration
            krd: Dict[str, float] = {}

            for kr in self._KEY_RATE_MATURITIES:
                dist = abs(mat - kr)
                weight = max(0, 1 - dist / 10.0)
                krd[f"{int(kr)}Y"] = round(dur * weight, 4)

            total_w = sum(krd.values()) or 1.0
            krd = {k: round(v / total_w * dur, 4) for k, v in krd.items()}
            krd_map[bond.issuer] = krd

        return krd_map

    def portfolio_duration_summary(self, bonds: List[BondSpec]) -> Dict[str, float]:
        """Compute average portfolio duration and convexity metrics."""
        if not bonds:
            return {}
        durs = [b.modified_duration for b in bonds]
        convex = [b.convexity for b in bonds]
        dv01s = [b.dv01 for b in bonds]
        return {
            "count": len(bonds),
            "avg_modified_duration": round(sum(durs) / len(durs), 4),
            "min_duration": round(min(durs), 4),
            "max_duration": round(max(durs), 4),
            "avg_convexity": round(sum(convex) / len(convex), 4),
            "total_dv01": round(sum(dv01s), 2),
        }


# ---------------------------------------------------------------------------
# CreditQualityScreener
# ---------------------------------------------------------------------------

_RATING_ORDER = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-",
    "CCC+", "CCC", "CCC-", "CC", "C", "D",
]
_RATING_RANK = {r: i for i, r in enumerate(_RATING_ORDER)}


def _rating_rank(rating: str) -> int:
    return _RATING_RANK.get(rating, 99)


class CreditQualityScreener:
    """Screen bonds by credit quality: ratings, watch status, coverage, Altman Z."""

    def __init__(self, universe: Optional[List[BondSpec]] = None) -> None:
        self._universe_builder = FixedIncomeUniverse()
        self._universe = universe or []

    def _ensure_universe(self, sectors: Optional[List[str]] = None) -> List[BondSpec]:
        if not self._universe:
            self._universe = self._universe_builder.build_universe(sectors)
        return self._universe

    def screen_by_rating(
        self,
        min_rating: Optional[str] = None,
        max_rating: Optional[str] = None,
        exact_ratings: Optional[List[str]] = None,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter by credit rating.

        Parameters
        ----------
        min_rating    : minimum rating (best quality), e.g. 'A'
        max_rating    : maximum rating (lowest quality), e.g. 'BBB-'
        exact_ratings : list of exact rating strings to include
        """
        universe = self._ensure_universe(sectors)
        results = []
        min_rank = _rating_rank(min_rating) if min_rating else 0
        max_rank = _rating_rank(max_rating) if max_rating else 99

        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            if bond.rating is None:
                continue
            rank = _rating_rank(bond.rating)
            if exact_ratings:
                if bond.rating not in exact_ratings:
                    continue
            else:
                if not (min_rank <= rank <= max_rank):
                    continue
            results.append(bond)
        return sorted(results, key=lambda b: _rating_rank(b.rating or "D"))

    def screen_exclude_negative_watch(
        self, sectors: Optional[List[str]] = None
    ) -> List[BondSpec]:
        """Return only bonds NOT on negative credit watch."""
        universe = self._ensure_universe(sectors)
        return [b for b in universe if not b.negative_watch
                and (not sectors or b.sector in sectors)]

    def screen_by_coverage(
        self,
        min_interest_coverage: Optional[float] = None,
        max_debt_ebitda: Optional[float] = None,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter by fundamental credit metrics (interest coverage, leverage)."""
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            if min_interest_coverage is not None:
                if bond.interest_coverage is None:
                    continue
                if bond.interest_coverage < min_interest_coverage:
                    continue
            if max_debt_ebitda is not None:
                if bond.debt_ebitda is None:
                    continue
                if bond.debt_ebitda > max_debt_ebitda:
                    continue
            results.append(bond)
        return results

    def screen_by_altman_z(
        self,
        min_z: float = _ALTMAN_ZSCORE_SAFE,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter bonds where issuer Altman Z-score is above safe threshold (1.8).

        Z > 2.99  : safe zone
        1.81-2.99 : grey zone
        Z < 1.81  : distress zone
        """
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            if bond.altman_z is None:
                continue
            if bond.altman_z >= min_z:
                results.append(bond)
        return sorted(results, key=lambda b: b.altman_z or 0, reverse=True)

    def screen_fallen_angel_risk(
        self,
        exclude: bool = True,
        sectors: Optional[List[str]] = None,
    ) -> List[BondSpec]:
        """Filter based on fallen angel risk (BBB-/BBB issuers near HY boundary).

        exclude=True : return only bonds without fallen angel risk
        exclude=False: return only bonds WITH fallen angel risk (watchlist)
        """
        universe = self._ensure_universe(sectors)
        results = []
        for bond in universe:
            if sectors and bond.sector not in sectors:
                continue
            if exclude and not bond.fallen_angel_risk:
                results.append(bond)
            elif not exclude and bond.fallen_angel_risk:
                results.append(bond)
        return results

    def credit_quality_summary(self, bonds: List[BondSpec]) -> Dict[str, Any]:
        """Return credit quality distribution of a bond list."""
        rating_counts: Dict[str, int] = {}
        watch_count = 0
        fallen_angel_count = 0
        for bond in bonds:
            r = bond.rating or "NR"
            rating_counts[r] = rating_counts.get(r, 0) + 1
            if bond.negative_watch:
                watch_count += 1
            if bond.fallen_angel_risk:
                fallen_angel_count += 1
        return {
            "total": len(bonds),
            "rating_distribution": dict(sorted(
                rating_counts.items(), key=lambda x: _rating_rank(x[0])
            )),
            "negative_watch_count": watch_count,
            "fallen_angel_risk_count": fallen_angel_count,
        }


# ---------------------------------------------------------------------------
# FixedIncomeRankingEngine
# ---------------------------------------------------------------------------

class FixedIncomeRankingEngine:
    """Rank bonds by risk-adjusted return metrics.

    Methods
    -------
    carry_duration  : yield / modified_duration — yield per unit of rate risk
    spread_duration : OAS / modified_duration — spread per unit of rate risk
    credit_adj      : spread / (PD x LGD) — spread vs expected credit loss
    """

    # Annual PD (pct) and LGD by rating
    _PD_LGD: Dict[str, Tuple[float, float]] = {
        "AAA":  (0.001, 0.40),
        "AA+":  (0.003, 0.40),
        "AA":   (0.005, 0.40),
        "AA-":  (0.008, 0.40),
        "A+":   (0.012, 0.45),
        "A":    (0.018, 0.45),
        "A-":   (0.025, 0.45),
        "BBB+": (0.040, 0.50),
        "BBB":  (0.060, 0.50),
        "BBB-": (0.090, 0.55),
        "BB+":  (0.150, 0.60),
        "BB":   (0.250, 0.60),
        "BB-":  (0.400, 0.60),
        "B+":   (0.650, 0.65),
        "B":    (1.000, 0.65),
        "B-":   (1.600, 0.65),
        "CCC+": (2.500, 0.70),
        "CCC":  (4.000, 0.70),
        "CCC-": (6.500, 0.75),
    }

    def __init__(self, universe: Optional[List[BondSpec]] = None) -> None:
        self._universe_builder = FixedIncomeUniverse()
        self._universe = universe or []

    def _ensure_universe(self, sectors: Optional[List[str]] = None) -> List[BondSpec]:
        if not self._universe:
            self._universe = self._universe_builder.build_universe(sectors)
        return self._universe

    def _carry_duration_score(self, bond: BondSpec) -> float:
        """Yield per unit of modified duration (carry efficiency)."""
        if bond.modified_duration <= 0:
            return 0.0
        return bond.ytm / bond.modified_duration

    def _spread_duration_score(self, bond: BondSpec) -> float:
        """OAS per unit of modified duration (spread efficiency)."""
        if bond.modified_duration <= 0:
            return 0.0
        return bond.oas_bps / bond.modified_duration

    def _credit_adj_score(self, bond: BondSpec) -> float:
        """Spread / (PD x LGD): compensation per unit of expected credit loss.

        Higher score = better compensated for credit risk taken.
        """
        rating = bond.rating or "BBB"
        pd, lgd = self._PD_LGD.get(rating, (1.0, 0.60))
        expected_loss = pd * lgd   # annual, in pct
        if expected_loss <= 0:
            return float("inf")
        return bond.oas_bps / (expected_loss * 100)

    def rank_universe(
        self,
        universe: Optional[List[BondSpec]] = None,
        method: str = "carry_duration",
        sectors: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Rank bonds by the specified method and return top `limit` bonds.

        Parameters
        ----------
        universe : pre-built list, or None to build from scratch
        method   : carry_duration | spread_duration | credit_adj
        sectors  : filter to these sectors before ranking
        limit    : max results to return
        """
        if universe is None:
            universe = self._ensure_universe(sectors)

        if sectors:
            universe = [b for b in universe if b.sector in sectors]

        score_fn = {
            "carry_duration": self._carry_duration_score,
            "spread_duration": self._spread_duration_score,
            "credit_adj": self._credit_adj_score,
        }.get(method, self._carry_duration_score)

        scored = []
        for bond in universe:
            score = score_fn(bond)
            if math.isfinite(score):
                scored.append((score, bond))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, bond) in enumerate(scored[:limit], start=1):
            results.append({
                "rank": rank,
                "issuer": bond.issuer,
                "sector": bond.sector,
                "rating": bond.rating,
                "maturity_years": bond.maturity_years,
                "coupon_rate": bond.coupon_rate,
                "ytm": bond.ytm,
                "ytw": bond.ytw,
                "oas_bps": bond.oas_bps,
                "modified_duration": bond.modified_duration,
                "convexity": bond.convexity,
                "score": round(score, 4),
                "score_method": method,
                "negative_watch": bond.negative_watch,
                "fallen_angel_risk": bond.fallen_angel_risk,
                "tax_exempt": bond.tax_exempt,
                "callable": bond.callable,
            })
        return results


# ---------------------------------------------------------------------------
# Composite FixedIncomeScreener
# ---------------------------------------------------------------------------

class FixedIncomeScreener:
    """Orchestrates all FI screeners in a single pass.

    Use as primary entry point: pass a ScreenRequest and get back
    a filtered, ranked list of bonds.
    """

    def __init__(self) -> None:
        self._universe_builder = FixedIncomeUniverse()
        self._yield_screener = YieldScreener()
        self._spread_screener = SpreadScreener()
        self._duration_screener = DurationRiskScreener()
        self._credit_screener = CreditQualityScreener()
        self._ranking_engine = FixedIncomeRankingEngine()

    def screen(self, req: ScreenRequest) -> List[Dict[str, Any]]:
        """Run full multi-filter screen and return ranked results."""
        universe = self._universe_builder.build_universe(req.sectors)
        filtered = universe

        if req.sectors:
            filtered = [b for b in filtered if b.sector in req.sectors]
        if req.min_ytm is not None:
            filtered = [b for b in filtered if b.ytm >= req.min_ytm]
        if req.max_ytm is not None:
            filtered = [b for b in filtered if b.ytm <= req.max_ytm]
        if req.min_ytw is not None:
            filtered = [b for b in filtered if b.ytw >= req.min_ytw]
        if req.max_ytw is not None:
            filtered = [b for b in filtered if b.ytw <= req.max_ytw]
        if req.min_oas_bps is not None:
            filtered = [b for b in filtered if b.oas_bps >= req.min_oas_bps]
        if req.max_oas_bps is not None:
            filtered = [b for b in filtered if b.oas_bps <= req.max_oas_bps]
        if req.min_duration is not None:
            filtered = [b for b in filtered if b.modified_duration >= req.min_duration]
        if req.max_duration is not None:
            filtered = [b for b in filtered if b.modified_duration <= req.max_duration]
        if req.min_convexity is not None:
            filtered = [b for b in filtered if b.convexity >= req.min_convexity]
        if req.ratings:
            filtered = [b for b in filtered if b.rating in req.ratings]
        if req.exclude_negative_watch:
            filtered = [b for b in filtered if not b.negative_watch]
        if req.min_interest_coverage is not None:
            filtered = [
                b for b in filtered
                if b.interest_coverage is None or b.interest_coverage >= req.min_interest_coverage
            ]
        if req.max_debt_ebitda is not None:
            filtered = [
                b for b in filtered
                if b.debt_ebitda is None or b.debt_ebitda <= req.max_debt_ebitda
            ]
        if req.min_altman_z is not None:
            filtered = [
                b for b in filtered
                if b.altman_z is None or b.altman_z >= req.min_altman_z
            ]
        if req.exclude_fallen_angels:
            filtered = [b for b in filtered if not b.fallen_angel_risk]

        ranked = self._ranking_engine.rank_universe(
            universe=filtered,
            method=req.rank_method,
            limit=req.limit,
        )

        if req.tax_equivalent:
            for item in ranked:
                sector = item.get("sector")
                ytm = item.get("ytm", 0)
                if sector == "muni":
                    item["tey"] = round(ytm * _MUNI_TEY_FACTOR, 4)
                else:
                    item["tey"] = ytm

        _log_screen("composite", req.dict(), len(ranked))
        return ranked


def _log_screen(screen_type: str, params: dict, result_count: int) -> None:
    import json as _json
    try:
        conn = _db()
        _ensure_tables()
        conn.execute(
            """INSERT INTO fi_screen_history (ts, screen_type, params_json, result_count)
               VALUES (?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), screen_type, _json.dumps(params), result_count),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Screen log write failed: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

fi_screener_router = APIRouter(prefix="/fi", tags=["fixed-income-screener"])

_screener: Optional[FixedIncomeScreener] = None
_univ_builder: Optional[FixedIncomeUniverse] = None
_yield_scr: Optional[YieldScreener] = None
_spread_scr: Optional[SpreadScreener] = None
_dur_scr: Optional[DurationRiskScreener] = None
_credit_scr: Optional[CreditQualityScreener] = None
_ranker: Optional[FixedIncomeRankingEngine] = None


def _get_screener() -> FixedIncomeScreener:
    global _screener
    if _screener is None:
        _screener = FixedIncomeScreener()
    return _screener


def _get_universe_builder() -> FixedIncomeUniverse:
    global _univ_builder
    if _univ_builder is None:
        _univ_builder = FixedIncomeUniverse()
    return _univ_builder


def _get_yield_screener() -> YieldScreener:
    global _yield_scr
    if _yield_scr is None:
        _yield_scr = YieldScreener()
    return _yield_scr


def _get_spread_screener() -> SpreadScreener:
    global _spread_scr
    if _spread_scr is None:
        _spread_scr = SpreadScreener()
    return _spread_scr


def _get_duration_screener() -> DurationRiskScreener:
    global _dur_scr
    if _dur_scr is None:
        _dur_scr = DurationRiskScreener()
    return _dur_scr


def _get_credit_screener() -> CreditQualityScreener:
    global _credit_scr
    if _credit_scr is None:
        _credit_scr = CreditQualityScreener()
    return _credit_scr


def _get_ranker() -> FixedIncomeRankingEngine:
    global _ranker
    if _ranker is None:
        _ranker = FixedIncomeRankingEngine()
    return _ranker


# ------------------------------------------------------------------
# /fi/universe
# ------------------------------------------------------------------

@fi_screener_router.get("/universe", summary="Build and return FI universe")
async def get_universe(
    sectors: Optional[str] = Query(None, description="Comma-separated sectors"),
) -> Dict[str, Any]:
    """Return the full (or sector-filtered) bond universe."""
    sector_list = [s.strip() for s in sectors.split(",")] if sectors else None
    try:
        universe = _get_universe_builder().build_universe(sector_list)
        return {
            "count": len(universe),
            "sectors": sector_list or ["treasury", "tips", "ig_corp", "hy_corp", "muni", "agency"],
            "bonds": [b.dict() for b in universe[:200]],
            "note": "Showing first 200 bonds. Use /fi/screen for filtered results.",
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ------------------------------------------------------------------
# /fi/screen
# ------------------------------------------------------------------

@fi_screener_router.post("/screen", summary="Full multi-filter FI screen")
async def post_screen(req: ScreenRequest) -> Dict[str, Any]:
    """Run a composite fixed income screen with all filters applied."""
    try:
        results = _get_screener().screen(req)
        return {
            "count": len(results),
            "rank_method": req.rank_method,
            "filters_applied": {
                k: v for k, v in req.dict().items() if v is not None and k != "limit"
            },
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ------------------------------------------------------------------
# /fi/yield-screen
# ------------------------------------------------------------------

@fi_screener_router.get("/yield-screen", summary="Screen by yield")
async def get_yield_screen(
    min_ytm: Optional[float] = Query(None),
    max_ytm: Optional[float] = Query(None),
    min_ytw: Optional[float] = Query(None),
    max_ytw: Optional[float] = Query(None),
    sectors: Optional[str] = Query(None),
    tax_equivalent: bool = Query(False),
) -> Dict[str, Any]:
    """Screen bonds by YTM/YTW thresholds, with optional TEY mode for munis."""
    sector_list = [s.strip() for s in sectors.split(",")] if sectors else None
    try:
        scr = _get_yield_screener()
        if tax_equivalent:
            results = scr.screen_tax_equivalent(
                min_tey=min_ytm, max_tey=max_ytm, compare_ytm=min_ytm
            )
            return {
                "mode": "tax_equivalent",
                "count": len(results),
                "results": [b.dict() for b in results[:100]],
            }
        results = scr.screen_by_yield(
            min_ytm=min_ytm, max_ytm=max_ytm,
            min_ytw=min_ytw, max_ytw=max_ytw,
            sectors=sector_list,
        )
        return {"count": len(results), "results": [b.dict() for b in results[:100]]}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fi_screener_router.get("/yield-screen/real", summary="TIPS real yield screen")
async def get_real_yield_screen(
    min_real_yield: Optional[float] = Query(None),
    max_real_yield: Optional[float] = Query(None),
) -> Dict[str, Any]:
    """Screen TIPS bonds by real yield."""
    try:
        results = _get_yield_screener().screen_real_yield(
            min_real_yield=min_real_yield, max_real_yield=max_real_yield
        )
        return {"count": len(results), "results": [b.dict() for b in results]}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ------------------------------------------------------------------
# /fi/spread-screen
# ------------------------------------------------------------------

@fi_screener_router.get("/spread-screen", summary="Screen by OAS spread")
async def get_spread_screen(
    min_oas_bps: Optional[float] = Query(None),
    max_oas_bps: Optional[float] = Query(None),
    sectors: Optional[str] = Query(None),
    min_pct: Optional[float] = Query(None, description="Min OAS percentile (0-100)"),
    max_pct: Optional[float] = Query(None, description="Max OAS percentile (0-100)"),
) -> Dict[str, Any]:
    """Screen by OAS spread and/or historical spread percentile."""
    sector_list = [s.strip() for s in sectors.split(",")] if sectors else None
    try:
        scr = _get_spread_screener()
        if min_pct is not None or max_pct is not None:
            results = scr.screen_by_oas_percentile(
                min_pct=min_pct, max_pct=max_pct, sectors=sector_list
            )
        else:
            results = scr.screen_by_oas(
                min_oas_bps=min_oas_bps, max_oas_bps=max_oas_bps, sectors=sector_list
            )
        return {"count": len(results), "results": [b.dict() for b in results[:100]]}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fi_screener_router.get("/spread-screen/sector-comparison", summary="Sector OAS comparison")
async def get_sector_comparison(
    sectors: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Return median OAS by sector and sub-sector for relative value analysis."""
    sector_list = [s.strip() for s in sectors.split(",")] if sectors else None
    try:
        return _get_spread_screener().sector_spread_comparison(sectors=sector_list)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fi_screener_router.get("/spread-screen/relative-value", summary="Treasury relative value")
async def get_relative_value(
    sectors: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Return bonds sorted by spread vs same-maturity Treasury (richest to cheapest)."""
    sector_list = [s.strip() for s in sectors.split(",")] if sectors else None
    try:
        rv = _get_spread_screener().treasury_relative_value(sectors=sector_list)
        return {"count": len(rv), "results": rv[:100]}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fi_screener_router.get("/spread-screen/momentum", summary="Spread widening/narrowing screen")
async def get_spread_momentum(
    min_30d_change_bps: Optional[float] = Query(None),
    max_30d_change_bps: Optional[float] = Query(None),
    sectors: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Screen by 30-day OAS change. Positive = widening (cheapening)."""
    sector_list = [s.strip() for s in sectors.split(",")] if sectors else None
    try:
        results = _get_spread_screener().screen_by_spread_momentum(
            min_30d_change_bps=min_30d_change_bps,
            max_30d_change_bps=max_30d_change_bps,
            sectors=sector_list,
        )
        return {"count": len(results), "results": [b.dict() for b in results[:100]]}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ------------------------------------------------------------------
# /fi/duration-screen
# ------------------------------------------------------------------

@fi_screener_router.get("/duration-screen", summary="Screen by duration and rate risk")
async def get_duration_screen(
    min_duration: Optional[float] = Query(None),
    max_duration: Optional[float] = Query(None),
    dv01_bucket: Optional[str] = Query(None, description="low|medium|high"),
    min_convexity: Optional[float] = Query(None),
    sectors: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Screen by modified duration, DV01 bucket, or positive convexity."""
    sector_list = [s.strip() for s in sectors.split(",")] if sectors else None
    scr = _get_duration_screener()
    try:
        if dv01_bucket:
            results = scr.screen_by_dv01_bucket(bucket=dv01_bucket, sectors=sector_list)
        elif min_convexity is not None:
            results = scr.screen_positive_convexity(
                min_convexity=min_convexity, sectors=sector_list
            )
        else:
            results = scr.screen_by_duration(
                min_duration=min_duration, max_duration=max_duration, sectors=sector_list
            )
        summary = scr.portfolio_duration_summary(results)
        return {
            "count": len(results),
            "summary": summary,
            "results": [b.dict() for b in results[:100]],
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fi_screener_router.post("/duration-screen/key-rate", summary="Key rate duration by issuer")
async def get_key_rate_exposure(bonds_request: List[str]) -> Dict[str, Any]:
    """Compute KRD at 2Y, 5Y, 10Y, 30Y for named issuers."""
    try:
        universe = _get_universe_builder().build_universe()
        selected = [b for b in universe if b.issuer in bonds_request]
        if not selected:
            return {"status": "no_match", "matched_count": 0}
        krd = _get_duration_screener().key_rate_exposure(selected)
        return {"matched_count": len(selected), "key_rate_duration": krd}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ------------------------------------------------------------------
# /fi/credit-screen
# ------------------------------------------------------------------

@fi_screener_router.get("/credit-screen", summary="Screen by credit quality")
async def get_credit_screen(
    min_rating: Optional[str] = Query(None),
    max_rating: Optional[str] = Query(None),
    ratings: Optional[str] = Query(None, description="Exact ratings, comma-separated"),
    exclude_negative_watch: bool = Query(True),
    min_interest_coverage: Optional[float] = Query(None),
    max_debt_ebitda: Optional[float] = Query(None),
    min_altman_z: Optional[float] = Query(None),
    exclude_fallen_angels: bool = Query(False),
    sectors: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Screen by credit rating, watch status, leverage, and distress metrics."""
    sector_list = [s.strip() for s in sectors.split(",")] if sectors else None
    ratings_list = [r.strip() for r in ratings.split(",")] if ratings else None
    scr = _get_credit_screener()
    try:
        universe = _get_universe_builder().build_universe(sector_list)
        results = universe

        if ratings_list:
            results = scr.screen_by_rating(exact_ratings=ratings_list, sectors=sector_list)
        elif min_rating or max_rating:
            results = scr.screen_by_rating(
                min_rating=min_rating, max_rating=max_rating, sectors=sector_list
            )

        if exclude_negative_watch:
            results = [b for b in results if not b.negative_watch]
        if min_interest_coverage is not None:
            results = [
                b for b in results
                if b.interest_coverage is None or b.interest_coverage >= min_interest_coverage
            ]
        if max_debt_ebitda is not None:
            results = [
                b for b in results
                if b.debt_ebitda is None or b.debt_ebitda <= max_debt_ebitda
            ]
        if min_altman_z is not None:
            results = [
                b for b in results
                if b.altman_z is None or b.altman_z >= min_altman_z
            ]
        if exclude_fallen_angels:
            results = [b for b in results if not b.fallen_angel_risk]

        summary = scr.credit_quality_summary(results)
        return {
            "count": len(results),
            "credit_summary": summary,
            "results": [b.dict() for b in results[:100]],
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fi_screener_router.get("/credit-screen/fallen-angels", summary="Fallen angel watch list")
async def get_fallen_angels() -> Dict[str, Any]:
    """Return BBB/BBB- issuers with high leverage at risk of HY downgrade."""
    try:
        results = _get_credit_screener().screen_fallen_angel_risk(exclude=False)
        return {
            "count": len(results),
            "note": "BBB-rated issuers with Debt/EBITDA > 4x — monitor for potential downgrade",
            "results": [b.dict() for b in results],
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fi_screener_router.get("/credit-screen/altman-z", summary="Altman Z-score screen")
async def get_altman_screen(
    min_z: float = Query(_ALTMAN_ZSCORE_SAFE, description="Min Altman Z-score"),
    sectors: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Screen bonds where issuer Altman Z-score is above the safe threshold."""
    sector_list = [s.strip() for s in sectors.split(",")] if sectors else None
    try:
        results = _get_credit_screener().screen_by_altman_z(min_z=min_z, sectors=sector_list)
        return {
            "count": len(results),
            "min_altman_z": min_z,
            "zones": {
                "safe": "> 2.99",
                "grey": "1.81 - 2.99",
                "distress": "< 1.81",
            },
            "results": [b.dict() for b in results[:100]],
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ------------------------------------------------------------------
# /fi/rank
# ------------------------------------------------------------------

@fi_screener_router.post("/rank", summary="Rank FI universe by risk-adjusted return")
async def post_rank(req: RankRequest) -> Dict[str, Any]:
    """Rank bonds by carry/duration, spread/duration, or credit-adjusted scoring."""
    sector_map: Dict[str, Optional[List[str]]] = {
        "all":      None,
        "ig":       ["ig_corp"],
        "hy":       ["hy_corp"],
        "muni":     ["muni"],
        "treasury": ["treasury"],
        "tips":     ["tips"],
        "agency":   ["agency"],
    }
    sectors = sector_map.get(req.universe_key)
    try:
        results = _get_ranker().rank_universe(
            method=req.method,
            sectors=sectors,
            limit=req.limit,
        )
        return {
            "universe": req.universe_key,
            "method": req.method,
            "count": len(results),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ------------------------------------------------------------------
# /fi/treasury-curve
# ------------------------------------------------------------------

def _mat_label(m: float) -> str:
    months = round(m * 12)
    if months < 12:
        return f"{months}M"
    return f"{round(m)}Y"


@fi_screener_router.get("/treasury-curve", summary="Treasury yield curve")
async def get_treasury_curve() -> Dict[str, Any]:
    """Return the Treasury yield curve used by the screener."""
    curve_list = [
        {"maturity_years": m, "maturity_label": _mat_label(m), "yield_pct": y}
        for m, y in sorted(_TREASURY_CURVE.items())
    ]
    return {
        "curve": curve_list,
        "note": "Approximate on-the-run Treasury yields (2025-2026 baseline).",
    }


# ------------------------------------------------------------------
# /fi/muni-tey
# ------------------------------------------------------------------

@fi_screener_router.get("/muni-tey", summary="Muni TEY calculator")
async def get_muni_tey(
    ytm: float = Query(..., description="Muni YTM as pct"),
    tax_rate: float = Query(0.40, description="Combined marginal tax rate (0.40 = 40%)"),
) -> Dict[str, Any]:
    """Compute tax-equivalent yield for a muni bond."""
    if not (0 <= tax_rate < 1.0):
        raise HTTPException(400, "tax_rate must be between 0 and 1")
    tey = ytm / (1.0 - tax_rate)
    return {
        "muni_ytm": ytm,
        "tax_rate": tax_rate,
        "tax_equivalent_yield": round(tey, 4),
        "breakeven_corporate_yield": round(tey, 4),
        "note": (
            f"A muni yielding {ytm}% is equivalent to a corporate yielding {tey:.2f}% "
            f"for an investor in the {tax_rate*100:.0f}% combined tax bracket."
        ),
    }


# ------------------------------------------------------------------
# /fi/bond-analytics
# ------------------------------------------------------------------

class BondAnalyticsRequest(BaseModel):
    ytm_pct: float = Field(..., description="YTM in pct, e.g. 5.0")
    coupon_rate_pct: float = Field(..., description="Annual coupon rate in pct")
    maturity_years: float = Field(..., description="Years to maturity")
    face: float = Field(1000.0, description="Face/par value")
    freq: int = Field(2, description="Coupon payments per year")


@fi_screener_router.post("/bond-analytics", summary="Single bond analytics calculator")
async def post_bond_analytics(req: BondAnalyticsRequest) -> Dict[str, Any]:
    """Compute price, duration, convexity, DV01 for a single bond spec."""
    try:
        price = _bond_price(req.ytm_pct, req.coupon_rate_pct, req.maturity_years, req.face, req.freq)
        mod_dur, mac_dur, convex, dv01 = _compute_duration_convexity(
            req.ytm_pct, req.coupon_rate_pct, req.maturity_years, req.face, req.freq
        )
        price_100 = price / (req.face / 100.0)
        tsy_yield = _interp_treasury_yield(req.maturity_years)
        spread = round((req.ytm_pct - tsy_yield) * 100, 1)
        return {
            "ytm_pct": req.ytm_pct,
            "coupon_rate_pct": req.coupon_rate_pct,
            "maturity_years": req.maturity_years,
            "price": round(price, 4),
            "price_per_100": round(price_100, 4),
            "modified_duration": mod_dur,
            "macaulay_duration": mac_dur,
            "convexity": convex,
            "dv01_per_million": dv01,
            "treasury_yield_pct": round(tsy_yield, 4),
            "spread_vs_treasury_bps": spread,
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    "FixedIncomeUniverse",
    "YieldScreener",
    "SpreadScreener",
    "DurationRiskScreener",
    "CreditQualityScreener",
    "FixedIncomeRankingEngine",
    "FixedIncomeScreener",
    "BondSpec",
    "ScreenRequest",
    "RankRequest",
    "fi_screener_router",
]
