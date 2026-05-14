"""Credit spread term structure — Z-spread computation, maturity-bucketed credit curves,
spread vs treasury benchmark, migration alerts. SENTINEL wave-17c.

Data sources:
  - FINRA TRACE API (free, no auth): bond issuances search + aggregate quote data
  - FRED (St. Louis Fed): benchmark Treasury rates by tenor
  - yfinance (fallback): synthetic bond proxy via equity fundamentals

Key entry points:
  - get_credit_spread_profile(ticker) -> IssuerCreditProfile
  - screen_credit_spreads(tickers)   -> CreditSpreadScreen
"""
from __future__ import annotations

import asyncio
import io
import logging
from datetime import date, datetime
from typing import Any, Optional

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINRA_API_BASE = "https://api.finra.org/data/group/fixedIncome"
FINRA_BOND_SEARCH = (
    "https://services.finra.org/sysapi/api/issuances/active"
    "?fields=issueid,issueName,cusip,couponRate,maturityDate,moodysRating,spRating,issuerName"
    "&pageSize=50&issuerName={ticker}"
)

FRED_TREASURY = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

TREASURY_SERIES: dict[str, str] = {
    "1Y": "DGS1",
    "2Y": "DGS2",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "30Y": "DGS30",
}

# Tenor midpoints in years for bucket matching
_TENOR_YEARS: dict[str, float] = {
    "1Y": 1.0,
    "2Y": 2.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "30Y": 30.0,
}

_HTTP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "SENTINEL/1.0 research@sentinel.local",
}

# Ticker → FINRA issuer name fragment for better match rate
_TICKER_ISSUER_MAP: dict[str, str] = {
    "AAPL": "APPLE INC",
    "MSFT": "MICROSOFT",
    "AMZN": "AMAZON",
    "GOOGL": "ALPHABET",
    "GOOG": "ALPHABET",
    "META": "META PLATFORMS",
    "TSLA": "TESLA",
    "JPM": "JPMORGAN",
    "BAC": "BANK OF AMERICA",
    "GS": "GOLDMAN SACHS",
    "MS": "MORGAN STANLEY",
    "WFC": "WELLS FARGO",
    "C": "CITIGROUP",
    "IBM": "INTERNATIONAL BUSINESS MACH",
    "GE": "GENERAL ELECTRIC",
    "F": "FORD MOTOR",
    "GM": "GENERAL MOTORS",
    "T": "AT&T",
    "VZ": "VERIZON",
    "CVX": "CHEVRON",
    "XOM": "EXXON MOBIL",
    "JNJ": "JOHNSON & JOHNSON",
    "PFE": "PFIZER",
    "KO": "COCA-COLA",
    "PEP": "PEPSICO",
    "HD": "HOME DEPOT",
    "WMT": "WALMART",
    "DIS": "WALT DISNEY",
    "NFLX": "NETFLIX",
    "INTC": "INTEL",
    "AMD": "ADVANCED MICRO",
}

_CALLABLE_KEYWORDS = ("callable", "call", "mtns", "medium term note")
_CALLABLE_OAS_HAIRCUT_BPS = 50.0  # proxy option cost for callable bonds

# Migration alert threshold
_MIGRATION_ALERT_BPS = 300.0

# Inverted curve threshold
_INVERTED_CURVE_BPS = -50.0

# Bucket boundaries in days
_SHORT_DAYS = 730    # ≤2Y
_MEDIUM_DAYS = 2555  # ≤7Y

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TreasuryCurve(BaseModel):
    """Live Treasury par yields keyed by tenor label; rates stored as decimals."""
    model_config = ConfigDict(frozen=True)

    as_of: date
    rates: dict[str, float]  # e.g. {"1Y": 0.0510, "2Y": 0.0480, ...}


class BondSpread(BaseModel):
    """Spread analytics for a single corporate bond relative to Treasury benchmark."""
    model_config = ConfigDict(frozen=True)

    cusip: str
    issuer: str
    coupon: float
    maturity_date: str          # ISO-8601 string for frozen-model JSON compat
    maturity_bucket: str        # "short" | "medium" | "long" | "unknown"
    ytm: float                  # yield to maturity, decimal
    treasury_yield: float       # matched Treasury benchmark yield, decimal
    z_spread: float             # (ytm - treasury_yield) × 10_000 in bps
    oas_proxy: float            # z_spread adjusted for embedded options (bps)
    rating_moody: str
    rating_sp: str
    credit_tier: str            # IG_HG | IG | HY_BB | HY_B | HY_CCC | NR
    is_callable: bool
    data_quality: str           # "live" | "synthetic"


class SpreadCurve(BaseModel):
    """Maturity-bucketed credit spread curve for a single issuer."""
    model_config = ConfigDict(frozen=True)

    issuer: str
    ticker: str
    short_spread_bps: Optional[float]   # mean z_spread for ≤2Y bucket
    medium_spread_bps: Optional[float]  # mean z_spread for 2-7Y bucket
    long_spread_bps: Optional[float]    # mean z_spread for >7Y bucket
    curve_slope_bps: Optional[float]    # long - short (negative → inverted)
    avg_spread_bps: float               # overall mean across all bonds
    credit_tier: str
    spread_trend: Optional[str]         # "widening" | "tightening" | "stable" | None
    migration_alert: bool               # True if avg_spread > _MIGRATION_ALERT_BPS
    num_bonds: int


class IssuerCreditProfile(BaseModel):
    """Full credit analytics for a single ticker."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    spread_curve: Optional[SpreadCurve]
    bonds: list[BondSpread]
    sector_comparison: str
    warnings: list[str]


class SpreadScreenRow(BaseModel):
    """Compact row for the credit screen summary table."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    avg_spread_bps: float
    curve_slope: Optional[float]
    credit_tier: str
    migration_alert: bool
    num_bonds: int


class CreditSpreadScreen(BaseModel):
    """Aggregated credit spread screen across multiple tickers."""
    model_config = ConfigDict(frozen=True)

    tickers_screened: list[str]
    results: list[IssuerCreditProfile]
    wide_spread_issuers: list[str]
    inverted_curve_issuers: list[str]
    migration_alerts: list[str]
    avg_ig_spread_bps: Optional[float]
    avg_hy_spread_bps: Optional[float]
    as_of: date
    warnings: list[str]

    @property
    def screen_rows(self) -> list[SpreadScreenRow]:
        rows: list[SpreadScreenRow] = []
        for profile in self.results:
            if profile.spread_curve is None:
                continue
            rows.append(
                SpreadScreenRow(
                    ticker=profile.ticker,
                    avg_spread_bps=profile.spread_curve.avg_spread_bps,
                    curve_slope=profile.spread_curve.curve_slope_bps,
                    credit_tier=profile.spread_curve.credit_tier,
                    migration_alert=profile.spread_curve.migration_alert,
                    num_bonds=profile.spread_curve.num_bonds,
                )
            )
        return sorted(rows, key=lambda r: r.avg_spread_bps, reverse=True)


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def _maturity_bucket(maturity_date_str: str) -> str:
    """Classify a bond maturity date string into short / medium / long.

    Accepts YYYY-MM-DD or MM/DD/YYYY. Returns 'unknown' on parse failure.
    """
    if not maturity_date_str:
        return "unknown"
    parsed: Optional[date] = None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(maturity_date_str.strip(), fmt).date()
            break
        except (ValueError, AttributeError):
            continue
    if parsed is None:
        return "unknown"
    days_to_mat = (parsed - date.today()).days
    if days_to_mat <= 0:
        return "unknown"
    if days_to_mat <= _SHORT_DAYS:
        return "short"
    if days_to_mat <= _MEDIUM_DAYS:
        return "medium"
    return "long"


def _years_to_maturity(maturity_date_str: str) -> float:
    """Return years to maturity; 0.0 on parse error or already matured."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%d/%m/%Y"):
        try:
            mat = datetime.strptime(maturity_date_str.strip(), fmt).date()
            return max(0.0, (mat - date.today()).days / 365.25)
        except (ValueError, AttributeError):
            continue
    return 0.0


def _ytm_proxy(coupon: float, face: float, price: float, years_to_maturity: float) -> float:
    """Approximate YTM via bond pricing shortcut.

    Formula: (annual_coupon + (face - price) / years) / ((face + price) / 2)
    Returns coupon / face if years_to_maturity is essentially zero.
    """
    if years_to_maturity < 0.01:
        return coupon / face if face > 0 else 0.0
    annual_coupon = coupon
    numerator = annual_coupon + (face - price) / years_to_maturity
    denominator = (face + price) / 2.0
    return numerator / denominator if denominator > 0 else 0.0


def _z_spread_bps(ytm: float, treasury_yield: float) -> float:
    """Compute simple Z-spread proxy in basis points, floored at -100 bps."""
    spread = (ytm - treasury_yield) * 10_000.0
    return max(spread, -100.0)


def _credit_tier(moody: str, sp: str) -> str:
    """Map agency ratings to internal credit tier labels.

    Checks both agencies, returns the less conservative (lower risk) tier
    when they disagree.
    """
    tier_rank = {
        "IG_HG": 0,
        "IG": 1,
        "HY_BB": 2,
        "HY_B": 3,
        "HY_CCC": 4,
        "NR": 5,
    }

    def _parse_one(rating: str) -> str:
        if not rating:
            return "NR"
        r = rating.strip().upper()
        # Moody's convention
        if r.startswith("AAA") or r.startswith("AA"):
            return "IG_HG"
        if r.startswith("A") and not r.startswith("AA"):
            return "IG"
        if r.startswith("BAA") or r.startswith("BBB"):
            return "IG"
        if r.startswith("BA") or r.startswith("BB"):
            return "HY_BB"
        if r.startswith("B") and not r.startswith("BA") and not r.startswith("BB"):
            return "HY_B"
        if r.startswith("CAA") or r.startswith("CCC"):
            return "HY_CCC"
        # S&P / Fitch convention overlaps via initial chars above; handle NR / WR
        return "NR"

    tier_m = _parse_one(moody)
    tier_s = _parse_one(sp)
    # Return better (lower rank = less risky)
    if tier_rank[tier_m] <= tier_rank[tier_s]:
        return tier_m
    return tier_s


def _is_callable(description: str) -> bool:
    desc_lower = (description or "").lower()
    return any(kw in desc_lower for kw in _CALLABLE_KEYWORDS)


def _nearest_treasury_tenor(years: float, rates: dict[str, float]) -> tuple[str, float]:
    """Return (tenor_label, rate) for the Treasury tenor closest to `years`."""
    best_label = "10Y"
    best_delta = float("inf")
    for label, tenor_yrs in _TENOR_YEARS.items():
        if label not in rates:
            continue
        delta = abs(tenor_yrs - years)
        if delta < best_delta:
            best_delta = delta
            best_label = label
    return best_label, rates.get(best_label, 0.04)


def _dominant_credit_tier(bonds: list[BondSpread]) -> str:
    """Return the most common credit tier among bonds, excluding NR if possible."""
    if not bonds:
        return "NR"
    from collections import Counter
    counts = Counter(b.credit_tier for b in bonds)
    # prefer any rated tier over NR
    for tier, _ in counts.most_common():
        if tier != "NR":
            return tier
    return "NR"


# ---------------------------------------------------------------------------
# SpreadCurve builder
# ---------------------------------------------------------------------------


def _build_spread_curve(ticker: str, issuer: str, bonds: list[BondSpread]) -> Optional[SpreadCurve]:
    """Compute a maturity-bucketed SpreadCurve from a list of BondSpread objects.

    Returns None only when bonds list is completely empty.
    """
    if not bonds:
        return None

    buckets: dict[str, list[float]] = {"short": [], "medium": [], "long": []}
    all_spreads: list[float] = []

    for bond in bonds:
        if bond.maturity_bucket in buckets:
            buckets[bond.maturity_bucket].append(bond.z_spread)
        all_spreads.append(bond.z_spread)

    def bucket_mean(key: str) -> Optional[float]:
        vals = buckets[key]
        if not vals:
            return None
        return float(np.mean(vals))

    short_bps = bucket_mean("short")
    medium_bps = bucket_mean("medium")
    long_bps = bucket_mean("long")
    avg_spread = float(np.mean(all_spreads)) if all_spreads else 0.0

    # Curve slope: long - short; negative = inverted
    if long_bps is not None and short_bps is not None:
        slope = long_bps - short_bps
    elif long_bps is not None and medium_bps is not None:
        slope = long_bps - medium_bps
    else:
        slope = None

    migration_alert = avg_spread > _MIGRATION_ALERT_BPS
    credit_tier = _dominant_credit_tier(bonds)

    return SpreadCurve(
        issuer=issuer,
        ticker=ticker,
        short_spread_bps=round(short_bps, 2) if short_bps is not None else None,
        medium_spread_bps=round(medium_bps, 2) if medium_bps is not None else None,
        long_spread_bps=round(long_bps, 2) if long_bps is not None else None,
        curve_slope_bps=round(slope, 2) if slope is not None else None,
        avg_spread_bps=round(avg_spread, 2),
        credit_tier=credit_tier,
        spread_trend=None,   # historical comparison requires stored prior data
        migration_alert=migration_alert,
        num_bonds=len(bonds),
    )


# ---------------------------------------------------------------------------
# FRED Treasury curve fetch
# ---------------------------------------------------------------------------


async def _fetch_fred_series(series_id: str, client: httpx.AsyncClient) -> Optional[float]:
    """Fetch a single FRED CSV series and return the latest non-null value."""
    url = FRED_TREASURY.format(series_id=series_id)
    try:
        resp = await client.get(url, headers={"Accept": "text/csv"}, timeout=20.0)
        if resp.status_code != 200:
            logger.warning("fred.series status=%d series=%s", resp.status_code, series_id)
            return None
        text = resp.text.strip()
        lines = text.splitlines()
        # Lines: DATE,VALUE header then data rows
        for line in reversed(lines):
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            val_str = parts[1].strip()
            if val_str in (".", "", "NA"):
                continue
            try:
                return float(val_str)
            except ValueError:
                continue
        return None
    except Exception as exc:
        logger.warning("fred.series error series=%s exc=%s", series_id, exc)
        return None


async def _fetch_treasury_curve(client: httpx.AsyncClient) -> TreasuryCurve:
    """Fetch all Treasury benchmark rates from FRED in parallel.

    Rates returned in decimal form (e.g. 5.10% → 0.0510).
    Missing tenors are omitted from the rates dict.
    """
    labels = list(TREASURY_SERIES.keys())
    series_ids = [TREASURY_SERIES[lbl] for lbl in labels]

    raw_values = await asyncio.gather(
        *[_fetch_fred_series(sid, client) for sid in series_ids],
        return_exceptions=False,
    )

    rates: dict[str, float] = {}
    for label, raw in zip(labels, raw_values):
        if raw is not None:
            rates[label] = round(raw / 100.0, 6)  # percent → decimal

    if not rates:
        # Hard-coded fallback so the module never returns an empty curve
        logger.warning("treasury_curve: all FRED fetches failed, using static fallback")
        rates = {
            "1Y": 0.0510, "2Y": 0.0480, "5Y": 0.0460,
            "7Y": 0.0455, "10Y": 0.0450, "30Y": 0.0465,
        }

    logger.info("treasury_curve fetched tenors=%s", list(rates.keys()))
    return TreasuryCurve(as_of=date.today(), rates=rates)


# ---------------------------------------------------------------------------
# FINRA bond search
# ---------------------------------------------------------------------------


async def _search_finra_bonds(ticker: str, client: httpx.AsyncClient) -> list[dict]:
    """Search FINRA issuances API for corporate bonds by issuer ticker.

    Tries two name forms: mapped full name and raw ticker. Limits to 20 bonds.
    Falls back to the TRACE aggregates endpoint if the issuances API fails.
    Returns empty list (not an exception) when totally unavailable.
    """
    issuer_name = _TICKER_ISSUER_MAP.get(ticker.upper(), ticker.upper())
    bonds: list[dict] = []

    for name_attempt in [issuer_name, ticker.upper()]:
        if bonds:
            break
        url = FINRA_BOND_SEARCH.format(ticker=name_attempt)
        try:
            resp = await client.get(url, headers=_HTTP_HEADERS, timeout=25.0)
            if resp.status_code in (403, 404, 504):
                logger.debug("finra.issuances http=%d ticker=%s", resp.status_code, ticker)
                continue
            if resp.status_code != 200:
                logger.warning("finra.issuances http=%d ticker=%s", resp.status_code, ticker)
                continue
            data = resp.json()
            records: list[dict] = data if isinstance(data, list) else data.get("data", [])
            bonds = [r for r in records if r.get("cusip")]
        except Exception as exc:
            logger.debug("finra.issuances exc ticker=%s exc=%s", ticker, exc)

    if not bonds:
        # Fallback: try TRACE aggregates endpoint
        bonds = await _search_finra_trace_aggregates(issuer_name, client)

    return bonds[:20]


async def _search_finra_trace_aggregates(issuer_name: str, client: httpx.AsyncClient) -> list[dict]:
    """Secondary fallback: TRACE aggregates endpoint filtered by issuer name."""
    url = f"{FINRA_API_BASE}/name/traceAggregates"
    fields = (
        "cusip,issuerName,securityDescription,lastSaleYield,"
        "spreadToBenchmark,coupon,maturityDate,tradeCount"
    )
    params = {
        "limit": 20,
        "offset": 0,
        "fields": fields,
        "filter": f"issuerName=={issuer_name}",
    }
    try:
        resp = await client.get(url, params=params, headers=_HTTP_HEADERS, timeout=25.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        records: list[dict] = data if isinstance(data, list) else data.get("data", [])
        # Normalise field names to match issuances schema
        normalised: list[dict] = []
        for r in records:
            if not r.get("cusip"):
                continue
            normalised.append({
                "cusip": r.get("cusip", ""),
                "issueName": r.get("securityDescription", ""),
                "issuerName": r.get("issuerName", issuer_name),
                "couponRate": r.get("coupon"),
                "maturityDate": r.get("maturityDate"),
                "moodysRating": "",
                "spRating": "",
                "_ytm_pct": r.get("lastSaleYield"),
                "_spread_bps": r.get("spreadToBenchmark"),
            })
        return normalised
    except Exception as exc:
        logger.debug("finra.trace_aggregates exc issuer=%s exc=%s", issuer_name, exc)
        return []


# ---------------------------------------------------------------------------
# yfinance synthetic fallback
# ---------------------------------------------------------------------------


async def _synthetic_bonds_from_yfinance(ticker: str) -> list[dict]:
    """Build a best-effort synthetic corporate bond from yfinance equity data.

    Used when FINRA TRACE returns no bonds. Maps equity fundamentals to a
    proxy bond: forward earnings yield as YTM, typical IG/HY tenor buckets.
    Returns a list with one synthetic bond dict or empty list on failure.
    """

    def _fetch() -> dict:
        try:
            import yfinance as yf  # lazy import: yfinance is optional dep
            info = yf.Ticker(ticker).info or {}
            return info
        except Exception:
            return {}

    info: dict = await asyncio.to_thread(_fetch)
    if not info:
        return []

    # Derive proxy YTM from earnings yield (1/PE) or forward yield
    pe = info.get("trailingPE") or info.get("forwardPE")
    earnings_yield = (1.0 / pe) if pe and pe > 0 else None
    dividend_yield = info.get("dividendYield") or 0.0
    # Credit proxy: earnings yield or dividend yield + 2% IG spread
    ytm_proxy_pct = (earnings_yield or (dividend_yield + 0.02)) * 100.0  # percent

    long_term_debt = info.get("totalDebt") or info.get("longTermDebt") or 0
    market_cap = info.get("marketCap") or 1
    leverage_ratio = long_term_debt / market_cap if market_cap > 0 else 0

    # Derive synthetic rating proxy from leverage
    if leverage_ratio < 0.3:
        moody_proxy, sp_proxy = "Aa", "AA"
    elif leverage_ratio < 0.6:
        moody_proxy, sp_proxy = "A", "A"
    elif leverage_ratio < 1.2:
        moody_proxy, sp_proxy = "Baa", "BBB"
    elif leverage_ratio < 2.5:
        moody_proxy, sp_proxy = "Ba", "BB"
    else:
        moody_proxy, sp_proxy = "B", "B"

    issuer_name = info.get("longName") or info.get("shortName") or ticker.upper()
    today = date.today()
    # Synthetic 10-year bond
    mat_year = today.year + 10
    mat_str = f"{mat_year}-{today.month:02d}-{today.day:02d}"

    synthetic = {
        "cusip": f"SYNTH-{ticker.upper()}-10Y",
        "issueName": f"{issuer_name} Synthetic 10Y Note",
        "issuerName": issuer_name,
        "couponRate": round(max(ytm_proxy_pct - 0.25, 0.5), 2),
        "maturityDate": mat_str,
        "moodysRating": moody_proxy,
        "spRating": sp_proxy,
        "_ytm_pct": ytm_proxy_pct,
        "_synthetic": True,
    }

    logger.info(
        "synthetic_bond ticker=%s ytm_pct=%.2f rating=%s/%s",
        ticker, ytm_proxy_pct, moody_proxy, sp_proxy,
    )
    return [synthetic]


# ---------------------------------------------------------------------------
# Bond spread computation
# ---------------------------------------------------------------------------


def _bond_record_to_spread(
    rec: dict,
    treasury_curve: TreasuryCurve,
    issuer_fallback: str,
) -> Optional[BondSpread]:
    """Convert a raw FINRA/synthetic bond record into a BondSpread.

    Returns None if insufficient data (no maturity, no YTM derivable).
    """
    cusip: str = rec.get("cusip", "UNKNOWN")
    issuer: str = rec.get("issuerName", issuer_fallback)
    coupon_raw = rec.get("couponRate") or rec.get("coupon") or 0.0
    try:
        coupon = float(coupon_raw)
    except (TypeError, ValueError):
        coupon = 0.0

    mat_str: str = rec.get("maturityDate", "")
    if not mat_str:
        return None

    years = _years_to_maturity(mat_str)
    if years <= 0.0:
        return None  # already matured

    bucket = _maturity_bucket(mat_str)
    if bucket == "unknown":
        return None

    # YTM: prefer pre-computed (from TRACE aggregates), else derive via proxy
    ytm_pct_raw = rec.get("_ytm_pct")
    if ytm_pct_raw is not None:
        try:
            ytm = float(ytm_pct_raw) / 100.0  # percent → decimal
        except (TypeError, ValueError):
            ytm = None
    else:
        ytm = None

    if ytm is None or ytm <= 0:
        # Derive from coupon at par (new issue assumption: price=100, face=100)
        face, price = 100.0, 100.0
        annual_coupon = coupon  # coupon given as % of par → numeric coupon cash flow
        ytm = _ytm_proxy(annual_coupon / 100.0, face / 100.0, price / 100.0, years)

    if ytm <= 0:
        ytm = 0.01  # absolute floor

    # Match nearest Treasury tenor
    _tenor_label, tsy_yield = _nearest_treasury_tenor(years, treasury_curve.rates)
    z_spread = _z_spread_bps(ytm, tsy_yield)

    # OAS proxy: subtract callable option cost
    description = rec.get("issueName", rec.get("securityDescription", ""))
    callable_bond = _is_callable(description)
    oas = z_spread - _CALLABLE_OAS_HAIRCUT_BPS if callable_bond else z_spread
    oas = max(oas, -100.0)

    moody_r = str(rec.get("moodysRating") or "").strip()
    sp_r = str(rec.get("spRating") or "").strip()
    tier = _credit_tier(moody_r, sp_r)

    is_synthetic = bool(rec.get("_synthetic", False))
    data_quality = "synthetic" if is_synthetic else "live"

    return BondSpread(
        cusip=cusip,
        issuer=issuer,
        coupon=coupon,
        maturity_date=mat_str,
        maturity_bucket=bucket,
        ytm=round(ytm, 6),
        treasury_yield=round(tsy_yield, 6),
        z_spread=round(z_spread, 2),
        oas_proxy=round(oas, 2),
        rating_moody=moody_r or "NR",
        rating_sp=sp_r or "NR",
        credit_tier=tier,
        is_callable=callable_bond,
        data_quality=data_quality,
    )


# ---------------------------------------------------------------------------
# Core profile builder
# ---------------------------------------------------------------------------


async def get_credit_spread_profile(ticker: str) -> IssuerCreditProfile:
    """Compute full credit spread analytics for a single issuer ticker.

    Fetches Treasury curve and FINRA bond data concurrently. Falls back to
    yfinance synthetic bond when TRACE returns no results. Always returns a
    valid profile — never raises silently.

    Args:
        ticker: Equity ticker used as issuer search key (e.g. "AAPL", "T").

    Returns:
        IssuerCreditProfile with spread_curve, bonds, sector_comparison, warnings.
    """
    warnings: list[str] = []
    ticker_upper = ticker.upper()

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Concurrent: treasury curve + FINRA bond search
        treasury_task = asyncio.create_task(_fetch_treasury_curve(client))
        finra_task = asyncio.create_task(_search_finra_bonds(ticker_upper, client))

        treasury_curve, raw_bonds = await asyncio.gather(
            treasury_task, finra_task, return_exceptions=False
        )

    issuer_label = _TICKER_ISSUER_MAP.get(ticker_upper, ticker_upper)

    # Fallback: yfinance synthetic bond
    if not raw_bonds:
        logger.info("credit_profile: no FINRA bonds, trying yfinance fallback ticker=%s", ticker_upper)
        warnings.append(
            f"No FINRA TRACE bonds found for '{ticker_upper}'. "
            "Using synthetic bond derived from yfinance fundamentals."
        )
        raw_bonds = await _synthetic_bonds_from_yfinance(ticker_upper)
        if not raw_bonds:
            warnings.append(
                f"yfinance fallback also returned no data for '{ticker_upper}'. "
                "Profile contains no bond analytics."
            )

    # Convert raw records to BondSpread objects
    bond_spreads: list[BondSpread] = []
    skipped = 0
    for rec in raw_bonds:
        bs = _bond_record_to_spread(rec, treasury_curve, issuer_label)
        if bs is not None:
            bond_spreads.append(bs)
        else:
            skipped += 1

    if skipped:
        warnings.append(f"Skipped {skipped} bonds with insufficient maturity/YTM data.")

    # Spread curve
    spread_curve = _build_spread_curve(ticker_upper, issuer_label, bond_spreads)

    # Sector comparison text
    if spread_curve is not None:
        avg_bps = spread_curve.avg_spread_bps
        sector_comparison = (
            f"Avg {ticker_upper} spread: {avg_bps:.0f}bps; "
            "IG avg ~100bps; HY avg ~350bps. "
        )
        if avg_bps < 150:
            sector_comparison += f"{ticker_upper} trades in line with investment-grade peers."
        elif avg_bps < 300:
            sector_comparison += f"{ticker_upper} spread is elevated — high-yield territory approaching."
        else:
            sector_comparison += f"{ticker_upper} spread signals high-yield / distressed risk."
    else:
        sector_comparison = f"No spread data available for {ticker_upper}."

    if spread_curve and spread_curve.migration_alert:
        warnings.append(
            f"MIGRATION ALERT: {ticker_upper} avg spread {spread_curve.avg_spread_bps:.0f}bps "
            f"exceeds {_MIGRATION_ALERT_BPS:.0f}bps HY trigger."
        )

    if spread_curve and spread_curve.curve_slope_bps is not None:
        if spread_curve.curve_slope_bps < _INVERTED_CURVE_BPS:
            warnings.append(
                f"INVERTED CURVE: {ticker_upper} long spread below short spread "
                f"(slope={spread_curve.curve_slope_bps:.0f}bps)."
            )

    data_qualities = {b.data_quality for b in bond_spreads}
    if "synthetic" in data_qualities:
        warnings.append("One or more bonds use synthetic/estimated data — treat analytically.")

    return IssuerCreditProfile(
        ticker=ticker_upper,
        spread_curve=spread_curve,
        bonds=bond_spreads,
        sector_comparison=sector_comparison,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Multi-ticker screener
# ---------------------------------------------------------------------------


async def screen_credit_spreads(tickers: list[str]) -> CreditSpreadScreen:
    """Screen credit spreads across multiple issuer tickers concurrently.

    Fetches all IssuerCreditProfiles in parallel and aggregates into a
    CreditSpreadScreen with sector averages and alert lists.

    Args:
        tickers: List of equity ticker symbols to screen.

    Returns:
        CreditSpreadScreen with results, alert lists, IG/HY averages.
    """
    if not tickers:
        return CreditSpreadScreen(
            tickers_screened=[],
            results=[],
            wide_spread_issuers=[],
            inverted_curve_issuers=[],
            migration_alerts=[],
            avg_ig_spread_bps=None,
            avg_hy_spread_bps=None,
            as_of=date.today(),
            warnings=["No tickers provided to screen_credit_spreads."],
        )

    tickers_upper = [t.upper() for t in tickers]
    screen_warnings: list[str] = []

    # Fetch all profiles concurrently; catch individual failures
    async def _safe_profile(ticker: str) -> Optional[IssuerCreditProfile]:
        try:
            return await get_credit_spread_profile(ticker)
        except Exception as exc:
            screen_warnings.append(f"Failed to fetch profile for {ticker}: {exc}")
            logger.error("screen.profile ticker=%s error=%s", ticker, exc)
            return None

    raw_results = await asyncio.gather(
        *[_safe_profile(t) for t in tickers_upper], return_exceptions=False
    )
    results: list[IssuerCreditProfile] = [r for r in raw_results if r is not None]

    # Aggregate alert lists
    wide_spread_issuers: list[str] = []
    inverted_curve_issuers: list[str] = []
    migration_alerts: list[str] = []

    ig_spreads: list[float] = []
    hy_spreads: list[float] = []

    _IG_TIERS = {"IG_HG", "IG"}
    _HY_TIERS = {"HY_BB", "HY_B", "HY_CCC"}

    for profile in results:
        sc = profile.spread_curve
        if sc is None:
            continue

        if sc.avg_spread_bps > _MIGRATION_ALERT_BPS:
            wide_spread_issuers.append(profile.ticker)

        if sc.migration_alert:
            migration_alerts.append(profile.ticker)

        if sc.curve_slope_bps is not None and sc.curve_slope_bps < _INVERTED_CURVE_BPS:
            inverted_curve_issuers.append(profile.ticker)

        if sc.credit_tier in _IG_TIERS:
            ig_spreads.append(sc.avg_spread_bps)
        elif sc.credit_tier in _HY_TIERS:
            hy_spreads.append(sc.avg_spread_bps)

    avg_ig = float(np.mean(ig_spreads)) if ig_spreads else None
    avg_hy = float(np.mean(hy_spreads)) if hy_spreads else None

    if avg_ig is not None:
        avg_ig = round(avg_ig, 2)
    if avg_hy is not None:
        avg_hy = round(avg_hy, 2)

    if not results:
        screen_warnings.append("All ticker fetches failed — screen returned no results.")

    logger.info(
        "credit_spread_screen tickers=%d results=%d wide=%d inverted=%d migration=%d "
        "avg_ig=%.1f avg_hy=%s",
        len(tickers_upper), len(results),
        len(wide_spread_issuers), len(inverted_curve_issuers), len(migration_alerts),
        avg_ig or 0.0, f"{avg_hy:.1f}" if avg_hy else "N/A",
    )

    return CreditSpreadScreen(
        tickers_screened=tickers_upper,
        results=results,
        wide_spread_issuers=wide_spread_issuers,
        inverted_curve_issuers=inverted_curve_issuers,
        migration_alerts=migration_alerts,
        avg_ig_spread_bps=avg_ig,
        avg_hy_spread_bps=avg_hy,
        as_of=date.today(),
        warnings=screen_warnings,
    )
