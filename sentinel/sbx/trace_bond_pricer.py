"""FINRA TRACE corporate bond pricing engine — dim_036 (score 5 → 9).

Combines FINRA TRACE aggregate API for market prices, FRED for the Treasury
discounting curve, and pure-Python analytics (YTM, z-spread, duration,
convexity, DV01) to produce a full BondPricingResult for any CUSIP or issuer.

Public API
----------
price_bond(cusip, issuer)          → BondPricingResult
issuer_curve(issuer)               → CreditCurve
screen_bonds(min_yield, …)         → list[BondPricingResult]

Async context-manager not required; all helpers create short-lived clients.
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINRA_FIXED = "https://api.finra.org/data/group/fixedIncome"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
}

# FRED series IDs for the Treasury par-yield curve
SWAP_RATES: dict[str, str] = {
    "3M": "DTB3",
    "6M": "DTB6",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

# Tenor strings → fractional years (used for curve interpolation)
_TENOR_YEARS: dict[str, float] = {
    "3M": 0.25,
    "6M": 0.50,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

# Issuer name overrides: equity ticker → FINRA issuer name fragment
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
    "PFE": "PFIZER",
    "JNJ": "JOHNSON & JOHNSON",
    "V": "VISA",
    "MA": "MASTERCARD",
    "UNH": "UNITEDHEALTH",
}

_AGGREGATE_FIELDS = (
    "cusip,issuerName,securityDescription,lastSalePrice,lastSaleYield,"
    "lastSaleDate,spreadToBenchmark,coupon,maturityDate,tradeCount"
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class BondTrade(BaseModel):
    model_config = ConfigDict(frozen=True)

    cusip: str
    trade_date: date
    settlement_date: Optional[date] = None
    price: float                   # clean price per $100 face
    yield_pct: Optional[float] = None
    par_value: float = 0.0         # notional traded
    trade_type: str = ""           # D=dealer-customer, I=interdealer, C=customer
    report_side: str = ""          # B=buy, S=sell
    is_as_of: bool = False


class BondPricingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    cusip: str
    issuer_name: str
    coupon: Optional[float] = None
    maturity_date: Optional[date] = None
    # Market prices
    last_trade_price: Optional[float] = None
    last_trade_date: Optional[date] = None
    last_trade_yield: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    mid_price: Optional[float] = None
    bid_ask_spread_bps: Optional[float] = None
    # Analytics
    ytm: Optional[float] = None
    ytw: Optional[float] = None
    current_yield: Optional[float] = None
    duration_modified: Optional[float] = None
    convexity: Optional[float] = None
    dv01: Optional[float] = None          # per $1MM face
    oas: Optional[float] = None           # bps
    z_spread: Optional[float] = None      # bps
    # Meta
    rating_sp: Optional[str] = None
    rating_moody: Optional[str] = None
    amount_outstanding: Optional[float] = None
    is_callable: bool = False
    liquidity_score: float = 5.0          # 0–10


class BondTradeHistory(BaseModel):
    model_config = ConfigDict(frozen=True)

    cusip: str
    issuer: str
    trades: list[BondTrade]
    avg_price_30d: Optional[float] = None
    price_range_30d: Optional[tuple[float, float]] = None
    total_volume_30d: Optional[float] = None
    trade_count_30d: int = 0
    avg_bid_ask_spread_bps: Optional[float] = None


class CreditCurve(BaseModel):
    model_config = ConfigDict(frozen=True)

    issuer_name: str
    as_of: date
    # Each point: {maturity_years, ytm, cusip, price, spread_bps}
    curve_points: list[dict]
    interpolated_5y_ytm: Optional[float] = None
    interpolated_10y_ytm: Optional[float] = None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_DATE_FMTS = ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d")


def _parse_date(raw: Any) -> Optional[date]:
    if not raw:
        return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(str(raw).strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _parse_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _parse_decimal(raw: Any) -> Optional[Decimal]:
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, TypeError):
        return None


def _years_to_maturity(mat: date, as_of: Optional[date] = None) -> float:
    today = as_of or date.today()
    return max(0.0, (mat - today).days / 365.25)


# ---------------------------------------------------------------------------
# Pure-Python bond math
# ---------------------------------------------------------------------------


def _bond_cashflows(
    coupon: float, maturity_years: float, freq: int = 2
) -> list[tuple[float, float]]:
    """Return [(t_years, cash_flow)] for a standard fixed-rate bond.

    Face value normalised to 100. Coupon is annual rate (percent, e.g. 4.5).
    """
    n = max(1, round(maturity_years * freq))
    c = coupon / freq  # semi-annual coupon per $100
    dt = 1.0 / freq
    cashflows = []
    for i in range(1, n + 1):
        t = i * dt
        cf = c + (100.0 if i == n else 0.0)
        cashflows.append((t, cf))
    return cashflows


def _bond_price(coupon: float, ytm: float, maturity_years: float, freq: int = 2) -> float:
    """Theoretical bond price given YTM (both as annual percent)."""
    r = ytm / 100.0 / freq
    cashflows = _bond_cashflows(coupon, maturity_years, freq)
    if abs(r) < 1e-10:
        return sum(cf for _, cf in cashflows)
    return sum(cf / (1 + r) ** (t * freq) for t, cf in cashflows)


def compute_ytm(
    coupon: float, price: float, maturity_years: float, freq: int = 2
) -> float:
    """Newton-Raphson yield-to-maturity solver.

    Args:
        coupon: Annual coupon rate in percent (e.g. 4.5 for 4.5%).
        price: Clean price per $100 face.
        maturity_years: Time to maturity in fractional years.
        freq: Coupon frequency per year (2 = semi-annual).

    Returns:
        YTM in percent (e.g. 5.23).
    """
    if maturity_years <= 0 or price <= 0:
        return 0.0

    # Initial guess: approximate by current yield + par effect
    approx = (coupon + (100.0 - price) / max(maturity_years, 0.5)) / ((price + 100.0) / 2.0)
    ytm = max(0.001, min(approx * 100.0, 30.0))  # percent

    for _ in range(100):
        p = _bond_price(coupon, ytm, maturity_years, freq)
        r = ytm / 100.0 / freq
        cashflows = _bond_cashflows(coupon, maturity_years, freq)
        # dP/dr (w.r.t. periodic rate)
        dp_dr = -sum(
            t * freq * cf / (1 + r) ** (t * freq + 1)
            for t, cf in cashflows
        )
        dp_dytm = dp_dr / (100.0 * freq)  # chain rule: dr/dytm = 1/(100*freq)
        delta_p = p - price
        if abs(delta_p) < 1e-8:
            break
        if abs(dp_dytm) < 1e-12:
            break
        ytm -= delta_p / dp_dytm
        ytm = max(0.001, min(ytm, 50.0))

    return round(ytm, 6)


def compute_duration(
    coupon: float, ytm: float, maturity_years: float, freq: int = 2
) -> tuple[float, float, float]:
    """Return (Macaulay duration, modified duration, convexity).

    All in years. Convexity is the second-order price sensitivity.
    """
    r = ytm / 100.0 / freq
    cashflows = _bond_cashflows(coupon, maturity_years, freq)
    price = _bond_price(coupon, ytm, maturity_years, freq)

    if price <= 0 or abs(r + 1) < 1e-10:
        return 0.0, 0.0, 0.0

    mac_num = 0.0
    convex_num = 0.0
    for t, cf in cashflows:
        pv = cf / (1 + r) ** (t * freq)
        mac_num += t * pv
        convex_num += t * freq * (t * freq + 1) * pv / (1 + r) ** 2

    macaulay = mac_num / price
    modified = macaulay / (1 + r)
    convexity = convex_num / (price * freq**2)
    return round(macaulay, 4), round(modified, 4), round(convexity, 4)


def estimate_bid_ask(ytm: float, liquidity_score: float) -> float:
    """Estimate bid-ask spread in basis points from YTM and liquidity score.

    Investment grade (ytm < 5%): 25–50 bps base.
    High yield (ytm >= 5%): 50–150 bps base.
    Liquidity score 0–10: higher = tighter spread.
    """
    if ytm < 5.0:
        base_low, base_high = 15.0, 50.0
    else:
        base_low, base_high = 50.0, 200.0
    # Interpolate: score 10 → tightest, score 0 → widest
    frac = max(0.0, min(1.0, liquidity_score / 10.0))
    return round(base_low + (1.0 - frac) * (base_high - base_low), 1)


def _interp_curve(curve: dict[float, float], years: float) -> Optional[float]:
    """Linear interpolation across a {maturity_years → rate} curve dict."""
    tenors = sorted(curve)
    if not tenors:
        return None
    if years <= tenors[0]:
        return curve[tenors[0]]
    if years >= tenors[-1]:
        return curve[tenors[-1]]
    for i in range(len(tenors) - 1):
        t0, t1 = tenors[i], tenors[i + 1]
        if t0 <= years <= t1:
            w = (years - t0) / (t1 - t0)
            return curve[t0] + w * (curve[t1] - curve[t0])
    return None


# ---------------------------------------------------------------------------
# Treasury curve cache (module-level, 1-hour TTL)
# ---------------------------------------------------------------------------

_treasury_cache: dict[str, float] = {}
_treasury_cache_ts: float = 0.0
_TREASURY_CACHE_TTL = 3600.0  # seconds

# Static fallback in case FRED is unreachable (approximate current rates)
_TREASURY_FALLBACK: dict[str, float] = {
    "3M": 5.30,
    "6M": 5.25,
    "1Y": 5.10,
    "2Y": 4.80,
    "3Y": 4.70,
    "5Y": 4.60,
    "7Y": 4.55,
    "10Y": 4.50,
    "20Y": 4.70,
    "30Y": 4.65,
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


async def _get_json(url: str, params: Optional[dict] = None, timeout: float = 30.0) -> Any:
    """GET with tenacity retry, returns parsed JSON."""
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    ):
        with attempt:
            async with httpx.AsyncClient(
                headers=_HEADERS, timeout=timeout, follow_redirects=True
            ) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 429:
                    await asyncio.sleep(5)
                    resp.raise_for_status()
                resp.raise_for_status()
                return resp.json()


async def _get_text(url: str, params: Optional[dict] = None, timeout: float = 20.0) -> str:
    """GET returning raw text (for FRED CSV)."""
    async with httpx.AsyncClient(
        headers=_HEADERS, timeout=timeout, follow_redirects=True
    ) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.text


# ---------------------------------------------------------------------------
# FINRA TRACE helpers
# ---------------------------------------------------------------------------


def _bond_from_record(rec: dict) -> dict:
    """Normalise a TRACE aggregate JSON record to a flat dict."""
    return {
        "cusip": rec.get("cusip", ""),
        "issuer_name": rec.get("issuerName", rec.get("issuer_name", "")),
        "description": rec.get("securityDescription", rec.get("description", "")),
        "coupon": _parse_float(rec.get("coupon")),
        "maturity_date": _parse_date(rec.get("maturityDate", rec.get("maturity_date"))),
        "last_price": _parse_float(rec.get("lastSalePrice", rec.get("last_sale_price"))),
        "last_yield": _parse_float(rec.get("lastSaleYield", rec.get("last_sale_yield"))),
        "spread_to_benchmark": _parse_float(
            rec.get("spreadToBenchmark", rec.get("spread_to_benchmark"))
        ),
        "last_sale_date": _parse_date(rec.get("lastSaleDate", rec.get("last_sale_date"))),
        "trade_count": rec.get("tradeCount", rec.get("trade_count")),
    }


async def _fetch_trace_aggregates(
    filter_expr: str, limit: int = 50
) -> list[dict]:
    """Fetch TRACE aggregate records from the FINRA API."""
    url = f"{FINRA_FIXED}/name/traceAggregates"
    params = {
        "limit": limit,
        "offset": 0,
        "fields": _AGGREGATE_FIELDS,
        "filter": filter_expr,
        "sortFields": ["-lastSaleDate"],
    }
    try:
        data = await _get_json(url, params=params)
    except Exception as exc:
        logger.warning("trace._fetch_aggregates error filter=%s: %s", filter_expr, exc)
        return []

    records: list[dict] = data if isinstance(data, list) else data.get("data", [])
    return [_bond_from_record(r) for r in records if r.get("cusip")]


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class TRACEBondPricer:
    """Full-stack corporate bond pricing engine backed by FINRA TRACE and FRED.

    Usage (no context manager required):
        pricer = TRACEBondPricer()
        result = await pricer.price_bond("594918BP8", "MICROSOFT")
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Treasury curve
    # ------------------------------------------------------------------

    async def get_treasury_curve(self) -> dict[str, float]:
        """Fetch Treasury par yields from FRED. Cached for 1 hour.

        Returns a dict keyed by tenor string (e.g. '10Y') with rate in percent.
        """
        global _treasury_cache, _treasury_cache_ts

        now = time.monotonic()
        if _treasury_cache and (now - _treasury_cache_ts) < _TREASURY_CACHE_TTL:
            return dict(_treasury_cache)

        curve: dict[str, float] = {}
        for tenor, series_id in SWAP_RATES.items():
            try:
                text = await _get_text(FRED_CSV, params={"id": series_id})
                # CSV: DATE,VALUE — last non-"." row
                rate: Optional[float] = None
                for line in reversed(text.strip().splitlines()):
                    if "," not in line or line.startswith("DATE"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2 and parts[1].strip() != ".":
                        try:
                            rate = float(parts[1].strip())
                            break
                        except ValueError:
                            continue
                if rate is not None:
                    curve[tenor] = rate
            except Exception as exc:
                logger.warning("fred fetch error series=%s: %s", series_id, exc)

        if not curve:
            logger.warning("FRED unreachable, using static Treasury fallback curve")
            curve = dict(_TREASURY_FALLBACK)

        _treasury_cache = dict(curve)
        _treasury_cache_ts = now
        logger.info("treasury_curve loaded tenors=%d", len(curve))
        return curve

    def _curve_years(self, curve: dict[str, float]) -> dict[float, float]:
        """Convert tenor strings to fractional years."""
        return {
            _TENOR_YEARS[k]: v
            for k, v in curve.items()
            if k in _TENOR_YEARS
        }

    # ------------------------------------------------------------------
    # Z-spread
    # ------------------------------------------------------------------

    async def compute_z_spread(
        self,
        coupon: float,
        maturity_date: date,
        price: float,
        treasury_curve: Optional[dict[str, float]] = None,
    ) -> float:
        """Parallel shift z-spread to Treasury curve, in basis points.

        Binary search for z such that:
            Σ CF_t / (1 + (r_t + z/10000)/2)^(t*2) = price
        where r_t is interpolated from the Treasury curve at time t.
        """
        today = date.today()
        maturity_years = _years_to_maturity(maturity_date, today)
        if maturity_years <= 0 or price <= 0:
            return 0.0

        if treasury_curve is None:
            treasury_curve = await self.get_treasury_curve()

        curve_y = self._curve_years(treasury_curve)
        if not curve_y:
            return 0.0

        cashflows = _bond_cashflows(coupon, maturity_years)

        def _dcf(z_bps: float) -> float:
            z = z_bps / 10000.0
            total = 0.0
            for t, cf in cashflows:
                r_t = (_interp_curve(curve_y, t) or 4.5) / 100.0
                disc = (1 + (r_t + z) / 2) ** (t * 2)
                total += cf / disc
            return total

        # Binary search: z in [-500, 3000] bps
        lo, hi = -500.0, 3000.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            p = _dcf(mid)
            if p > price:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < 0.01:
                break

        return round((lo + hi) / 2.0, 2)

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    async def get_trade_history(
        self, cusip: str, days_back: int = 30
    ) -> BondTradeHistory:
        """Fetch TRACE aggregate history for a CUSIP.

        FINRA's public API returns daily aggregates; we collect the latest
        records and compute 30-day statistics.
        """
        records = await _fetch_trace_aggregates(f"cusip=={cusip}", limit=days_back)

        trades: list[BondTrade] = []
        issuer = ""
        for rec in records:
            if not rec.get("last_price") or not rec.get("last_sale_date"):
                continue
            if rec.get("issuer_name"):
                issuer = rec["issuer_name"]
            trades.append(
                BondTrade(
                    cusip=cusip,
                    trade_date=rec["last_sale_date"],
                    price=rec["last_price"],
                    yield_pct=rec.get("last_yield"),
                    par_value=0.0,
                )
            )

        prices = [t.price for t in trades]
        hist = BondTradeHistory(
            cusip=cusip,
            issuer=issuer,
            trades=trades,
            avg_price_30d=round(sum(prices) / len(prices), 4) if prices else None,
            price_range_30d=(min(prices), max(prices)) if prices else None,
            trade_count_30d=len(trades),
        )
        logger.info(
            "trace_history cusip=%s records=%d", cusip, len(trades)
        )
        return hist

    # ------------------------------------------------------------------
    # Single bond pricer
    # ------------------------------------------------------------------

    async def price_bond(
        self,
        cusip: str,
        issuer_name: str = "",
        coupon: Optional[float] = None,
        maturity_date: Optional[date] = None,
    ) -> BondPricingResult:
        """Price a single bond by CUSIP.

        Pulls TRACE aggregate data, computes YTM, z-spread, modified
        duration, convexity, DV01, and estimated bid/ask.
        """
        # Fetch from TRACE
        records = await _fetch_trace_aggregates(f"cusip=={cusip}", limit=5)
        rec = records[0] if records else {}

        # Override from TRACE if not provided
        issuer = issuer_name or rec.get("issuer_name", "")
        cpn = coupon if coupon is not None else rec.get("coupon")
        mat = maturity_date or rec.get("maturity_date")

        last_price = rec.get("last_price")
        last_yield = rec.get("last_yield")
        last_date = rec.get("last_sale_date")
        spread_raw = rec.get("spread_to_benchmark")
        trade_count = rec.get("trade_count") or 0

        # Liquidity score: 0–10 based on trade frequency
        liquidity = min(10.0, max(0.0, math.log1p(float(trade_count or 0)) * 1.5))

        # Analytics
        ytm: Optional[float] = None
        mod_dur: Optional[float] = None
        convexity: Optional[float] = None
        dv01: Optional[float] = None
        z_spread: Optional[float] = None
        oas: Optional[float] = None
        current_yield_val: Optional[float] = None
        bid_price: Optional[float] = None
        ask_price: Optional[float] = None
        mid_price: Optional[float] = None
        bid_ask_bps: Optional[float] = None

        treasury_curve: Optional[dict[str, float]] = None

        if last_price and cpn is not None and mat is not None:
            mat_years = _years_to_maturity(mat)
            if mat_years > 0:
                ytm = last_yield or compute_ytm(cpn, last_price, mat_years)
                _, mod_dur, convexity = compute_duration(cpn, ytm, mat_years)
                # DV01 per $1MM face: price sensitivity to 1bp shift
                dv01 = round(mod_dur * last_price / 100.0 * 10_000.0, 2)
                current_yield_val = round(cpn / last_price * 100.0, 4) if last_price else None

                # Z-spread
                try:
                    treasury_curve = await self.get_treasury_curve()
                    z_spread = await self.compute_z_spread(cpn, mat, last_price, treasury_curve)
                    # OAS ≈ Z-spread for bullet bonds (no optionality adjustment)
                    oas = z_spread
                except Exception as exc:
                    logger.warning("z_spread computation failed: %s", exc)

                # Bid-ask
                bid_ask_bps = estimate_bid_ask(ytm, liquidity)
                # Convert bps spread to price using modified duration
                price_spread = mod_dur * (bid_ask_bps / 10000.0) * last_price / 2.0
                bid_price = round(last_price - price_spread, 3)
                ask_price = round(last_price + price_spread, 3)
                mid_price = round(last_price, 3)

        result = BondPricingResult(
            cusip=cusip,
            issuer_name=issuer,
            coupon=cpn,
            maturity_date=mat,
            last_trade_price=last_price,
            last_trade_date=last_date,
            last_trade_yield=last_yield,
            bid_price=bid_price,
            ask_price=ask_price,
            mid_price=mid_price,
            bid_ask_spread_bps=bid_ask_bps,
            ytm=ytm,
            current_yield=current_yield_val,
            duration_modified=mod_dur,
            convexity=convexity,
            dv01=dv01,
            z_spread=z_spread,
            oas=oas,
            liquidity_score=round(liquidity, 2),
        )
        logger.info(
            "price_bond cusip=%s price=%s ytm=%s z_spread=%s",
            cusip, last_price, ytm, z_spread,
        )
        return result

    # ------------------------------------------------------------------
    # Issuer credit curve
    # ------------------------------------------------------------------

    async def get_issuer_curve(
        self,
        issuer_name: str,
        cusip_list: Optional[list[str]] = None,
    ) -> CreditCurve:
        """Build an issuer credit curve from all TRACE bonds for the issuer.

        Returns interpolated YTMs at 5Y and 10Y standard tenors.
        """
        today = date.today()

        if cusip_list:
            # Price each CUSIP individually
            tasks = [self.price_bond(c, issuer_name) for c in cusip_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            pricing_results = [r for r in results if isinstance(r, BondPricingResult)]
        else:
            # Search by issuer name via TRACE
            issuer_key = _TICKER_ISSUER_MAP.get(issuer_name.upper(), issuer_name.upper())
            records = await _fetch_trace_aggregates(
                f"issuerName=={issuer_key}", limit=50
            )
            if not records and issuer_key != issuer_name.upper():
                records = await _fetch_trace_aggregates(
                    f"issuerName=={issuer_name.upper()}", limit=50
                )

            treasury_curve = await self.get_treasury_curve()
            curve_y = self._curve_years(treasury_curve)

            points: list[dict] = []
            for rec in records:
                mat = rec.get("maturity_date")
                if not mat:
                    continue
                yrs = _years_to_maturity(mat, today)
                if yrs <= 0:
                    continue
                cpn = rec.get("coupon") or 0.0
                price = rec.get("last_price")
                ytm_val = rec.get("last_yield")
                if price and cpn and not ytm_val:
                    ytm_val = compute_ytm(cpn, price, yrs)
                if ytm_val is None:
                    continue
                t_rate = _interp_curve(curve_y, yrs) or 4.5
                spread = round((ytm_val - t_rate) * 100.0, 1)
                points.append({
                    "maturity_years": round(yrs, 2),
                    "ytm": round(ytm_val, 4),
                    "cusip": rec.get("cusip", ""),
                    "price": price,
                    "spread_bps": spread,
                    "coupon": cpn,
                })

            points.sort(key=lambda p: p["maturity_years"])
            curve_pts_dict = {p["maturity_years"]: p["ytm"] for p in points}
            interp5 = _interp_curve(curve_pts_dict, 5.0)
            interp10 = _interp_curve(curve_pts_dict, 10.0)

            return CreditCurve(
                issuer_name=issuer_name,
                as_of=today,
                curve_points=points,
                interpolated_5y_ytm=round(interp5, 4) if interp5 else None,
                interpolated_10y_ytm=round(interp10, 4) if interp10 else None,
            )

        # From CUSIP list path
        points = []
        treasury_curve2 = await self.get_treasury_curve()
        curve_y2 = self._curve_years(treasury_curve2)
        for r in pricing_results:
            if r.maturity_date is None or r.ytm is None:
                continue
            yrs = _years_to_maturity(r.maturity_date, today)
            if yrs <= 0:
                continue
            t_rate = _interp_curve(curve_y2, yrs) or 4.5
            spread = round((r.ytm - t_rate) * 100.0, 1)
            points.append({
                "maturity_years": round(yrs, 2),
                "ytm": r.ytm,
                "cusip": r.cusip,
                "price": r.last_trade_price,
                "spread_bps": spread,
                "coupon": r.coupon,
            })

        points.sort(key=lambda p: p["maturity_years"])
        curve_pts_dict = {p["maturity_years"]: p["ytm"] for p in points}
        interp5 = _interp_curve(curve_pts_dict, 5.0)
        interp10 = _interp_curve(curve_pts_dict, 10.0)

        logger.info("issuer_curve issuer=%s points=%d", issuer_name, len(points))
        return CreditCurve(
            issuer_name=issuer_name,
            as_of=today,
            curve_points=points,
            interpolated_5y_ytm=round(interp5, 4) if interp5 else None,
            interpolated_10y_ytm=round(interp10, 4) if interp10 else None,
        )

    # ------------------------------------------------------------------
    # Bond screener
    # ------------------------------------------------------------------

    async def screen_bonds(
        self,
        min_yield: float = 0.0,
        max_yield: float = 15.0,
        min_maturity_years: float = 0.5,
        max_maturity_years: float = 30.0,
        rating: Optional[str] = None,
        issuer_filter: Optional[str] = None,
        limit: int = 100,
    ) -> list[BondPricingResult]:
        """Screen the TRACE universe and return bonds matching criteria.

        Args:
            min_yield / max_yield: YTM filter in percent.
            min/max_maturity_years: Maturity window.
            rating: "IG" (investment grade, yield < 6%), "HY" (high yield, yield >= 6%),
                    or a specific label (informational only — TRACE has no ratings).
            issuer_filter: Partial issuer name to pre-filter TRACE results.
            limit: Maximum number of results returned.
        """
        today = date.today()

        filter_expr = "tradeCount>0"
        if issuer_filter:
            issuer_key = _TICKER_ISSUER_MAP.get(issuer_filter.upper(), issuer_filter.upper())
            filter_expr = f"issuerName=={issuer_key}"

        records = await _fetch_trace_aggregates(filter_expr, limit=min(limit * 3, 300))

        treasury_curve = await self.get_treasury_curve()
        curve_y = self._curve_years(treasury_curve)

        results: list[BondPricingResult] = []
        for rec in records:
            cpn = rec.get("coupon")
            mat = rec.get("maturity_date")
            price = rec.get("last_price")
            last_yield_raw = rec.get("last_yield")

            if not rec.get("cusip"):
                continue
            if mat is None:
                continue

            yrs = _years_to_maturity(mat, today)
            if yrs < min_maturity_years or yrs > max_maturity_years:
                continue

            ytm_val: Optional[float] = last_yield_raw
            if ytm_val is None and price and cpn:
                try:
                    ytm_val = compute_ytm(cpn, price, yrs)
                except Exception:
                    pass

            if ytm_val is None:
                continue
            if not (min_yield <= ytm_val <= max_yield):
                continue

            # Rating proxy: IG <6%, HY >=6%
            if rating == "IG" and ytm_val >= 6.0:
                continue
            if rating == "HY" and ytm_val < 6.0:
                continue

            trade_count = int(rec.get("trade_count") or 0)
            liquidity = min(10.0, max(0.0, math.log1p(float(trade_count)) * 1.5))

            mod_dur = convexity = dv01 = z_spread_val = oas_val = None
            bid_price = ask_price = bid_ask_bps = current_yield_val = None

            if cpn and price and yrs > 0:
                try:
                    _, mod_dur, convexity = compute_duration(cpn, ytm_val, yrs)
                    dv01 = round(mod_dur * price / 100.0 * 10_000.0, 2)
                    current_yield_val = round(cpn / price * 100.0, 4)
                    z_spread_val = await self.compute_z_spread(cpn, mat, price, treasury_curve)
                    oas_val = z_spread_val
                    bid_ask_bps = estimate_bid_ask(ytm_val, liquidity)
                    price_spread = mod_dur * (bid_ask_bps / 10000.0) * price / 2.0
                    bid_price = round(price - price_spread, 3)
                    ask_price = round(price + price_spread, 3)
                except Exception as exc:
                    logger.debug("screen analytics error cusip=%s: %s", rec["cusip"], exc)

            results.append(
                BondPricingResult(
                    cusip=rec["cusip"],
                    issuer_name=rec.get("issuer_name", ""),
                    coupon=cpn,
                    maturity_date=mat,
                    last_trade_price=price,
                    last_trade_date=rec.get("last_sale_date"),
                    last_trade_yield=last_yield_raw,
                    bid_price=bid_price,
                    ask_price=ask_price,
                    mid_price=round(price, 3) if price else None,
                    bid_ask_spread_bps=bid_ask_bps,
                    ytm=round(ytm_val, 4),
                    current_yield=current_yield_val,
                    duration_modified=mod_dur,
                    convexity=convexity,
                    dv01=dv01,
                    z_spread=z_spread_val,
                    oas=oas_val,
                    liquidity_score=round(liquidity, 2),
                )
            )
            if len(results) >= limit:
                break

        logger.info(
            "screen_bonds min_yield=%s max_yield=%s results=%d",
            min_yield, max_yield, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Public sync wrappers for analytics (no I/O)
    # ------------------------------------------------------------------

    def compute_ytm(
        self, coupon: float, price: float, maturity_years: float, freq: int = 2
    ) -> float:
        """Module-level compute_ytm as an instance method."""
        return compute_ytm(coupon, price, maturity_years, freq)

    def compute_duration(
        self, coupon: float, ytm: float, maturity_years: float, freq: int = 2
    ) -> tuple[float, float, float]:
        """Returns (macaulay_duration, modified_duration, convexity)."""
        return compute_duration(coupon, ytm, maturity_years, freq)

    def estimate_bid_ask(self, ytm: float, liquidity_score: float) -> float:
        """Estimate bid-ask spread in basis points."""
        return estimate_bid_ask(ytm, liquidity_score)


# ---------------------------------------------------------------------------
# Module-level convenience coroutines
# ---------------------------------------------------------------------------


async def price_bond(cusip: str, issuer: str = "") -> BondPricingResult:
    """Price a single bond by CUSIP. Convenience one-shot wrapper."""
    return await TRACEBondPricer().price_bond(cusip, issuer)


async def issuer_curve(issuer: str) -> CreditCurve:
    """Build a credit curve for an issuer by name or equity ticker."""
    return await TRACEBondPricer().get_issuer_curve(issuer)


async def screen_bonds(
    min_yield: float = 4.0,
    max_yield: float = 15.0,
    rating: Optional[str] = None,
    limit: int = 50,
) -> list[BondPricingResult]:
    """Screen bonds by yield range and optional rating proxy."""
    return await TRACEBondPricer().screen_bonds(
        min_yield=min_yield, max_yield=max_yield, rating=rating, limit=limit
    )
