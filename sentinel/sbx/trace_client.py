"""FINRA TRACE corporate bond pricing client — free 15-minute delayed OTC bond data.

Dimension #36 in SENTINEL competitive matrix. TRACE is the only free source of
OTC corporate bond transaction data with price, yield, spread and volume.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BondQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    cusip: str
    issuer_name: str
    description: str
    coupon: float | None
    maturity_date: date | None
    last_price: Decimal | None       # clean price (% of par)
    last_yield: float | None         # YTM %
    spread_to_benchmark: float | None  # OAS in bps
    last_sale_date: date | None
    trade_count: int | None
    source: str = "finra_trace"


class CreditCurve(BaseModel):
    """Simplified credit curve: yield by maturity for an issuer's bonds."""
    model_config = ConfigDict(frozen=True)

    issuer: str
    as_of: date
    points: list[dict]  # [{maturity_years, yield, spread, cusip}] sorted by maturity


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_DATE_FMTS = ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d")


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _parse_decimal(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, TypeError):
        return None


def _parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _parse_int(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _years_to_maturity(mat: date, as_of: date) -> float:
    return max(0.0, (mat - as_of).days / 365.25)


def _bond_from_record(rec: dict) -> BondQuote:
    """Map a FINRA TRACE aggregates JSON record to BondQuote."""
    return BondQuote(
        cusip=rec.get("cusip", ""),
        issuer_name=rec.get("issuerName", rec.get("issuer_name", "")),
        description=rec.get("securityDescription", rec.get("description", "")),
        coupon=_parse_float(rec.get("coupon")),
        maturity_date=_parse_date(rec.get("maturityDate", rec.get("maturity_date"))),
        last_price=_parse_decimal(rec.get("lastSalePrice", rec.get("last_sale_price"))),
        last_yield=_parse_float(rec.get("lastSaleYield", rec.get("last_sale_yield"))),
        spread_to_benchmark=_parse_float(
            rec.get("spreadToBenchmark", rec.get("spread_to_benchmark"))
        ),
        last_sale_date=_parse_date(
            rec.get("lastSaleDate", rec.get("last_sale_date"))
        ),
        trade_count=_parse_int(rec.get("tradeCount", rec.get("trade_count"))),
    )


# ---------------------------------------------------------------------------
# Treasury yield interpolation (benchmark for spread calc)
# Approximate par yields; updated quarterly — good enough for curve building.
# ---------------------------------------------------------------------------

_TREASURY_APPROX: dict[float, float] = {
    0.25: 5.30,
    0.50: 5.25,
    1.0:  5.10,
    2.0:  4.80,
    3.0:  4.70,
    5.0:  4.60,
    7.0:  4.55,
    10.0: 4.50,
    20.0: 4.70,
    30.0: 4.65,
}


def _interp_treasury(years: float) -> float:
    """Linear interpolation of approximate Treasury par yield for a given maturity."""
    tenors = sorted(_TREASURY_APPROX)
    if years <= tenors[0]:
        return _TREASURY_APPROX[tenors[0]]
    if years >= tenors[-1]:
        return _TREASURY_APPROX[tenors[-1]]
    for i in range(len(tenors) - 1):
        t0, t1 = tenors[i], tenors[i + 1]
        if t0 <= years <= t1:
            w = (years - t0) / (t1 - t0)
            return _TREASURY_APPROX[t0] + w * (_TREASURY_APPROX[t1] - _TREASURY_APPROX[t0])
    return 4.50  # fallback


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

# Issuer name overrides: map common equity tickers to FINRA issuer name fragments
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
}

_AGGREGATE_FIELDS = (
    "cusip,issuerName,securityDescription,lastSalePrice,lastSaleYield,"
    "lastSaleDate,spreadToBenchmark,coupon,maturityDate,tradeCount"
)

_TRADE_FIELDS = (
    "cusip,transactionDate,transactionTime,lastSalePrice,lastSaleYield,"
    "quantity,side,tradeStatus"
)


class TRACEClient:
    API_BASE = "https://api.finra.org/data/group/fixedIncome"
    HEADERS = {
        "Accept": "application/json",
        "User-Agent": "SENTINEL/1.0 research@example.com",
    }

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> TRACEClient:
        self._client = httpx.AsyncClient(
            headers=self.HEADERS,
            timeout=self._timeout,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("TRACEClient must be used as an async context manager")
        return self._client

    # ------------------------------------------------------------------
    # Internal request helper with tenacity retries
    # ------------------------------------------------------------------

    async def _get(self, url: str, params: dict | None = None) -> Any:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                resp = await self._http().get(url, params=params)
                if resp.status_code == 429:
                    logger.warning("trace_client: rate-limited, backing off")
                    await asyncio.sleep(5)
                    resp.raise_for_status()
                resp.raise_for_status()
                return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_bonds(
        self, issuer_ticker: str, limit: int = 20
    ) -> list[BondQuote]:
        """Search TRACE aggregates for bonds by issuer equity ticker.

        Tries an exact issuer name lookup first, then falls back to the
        known ticker→issuer mapping, and finally a bare upper-cased ticker
        fragment search.
        """
        issuer_name = _TICKER_ISSUER_MAP.get(issuer_ticker.upper(), issuer_ticker.upper())
        results = await self._fetch_aggregates_by_issuer(issuer_name, limit)

        # Fallback: try the raw ticker if the mapped name returned nothing
        if not results and issuer_name != issuer_ticker.upper():
            results = await self._fetch_aggregates_by_issuer(issuer_ticker.upper(), limit)

        logger.info(
            "trace.search_bonds ticker=%s issuer=%s found=%d",
            issuer_ticker, issuer_name, len(results),
        )
        return results

    async def _fetch_aggregates_by_issuer(
        self, issuer_name: str, limit: int
    ) -> list[BondQuote]:
        """Fetch TRACE aggregate records filtered by issuer name."""
        url = f"{self.API_BASE}/name/traceAggregates"
        params = {
            "limit": limit,
            "offset": 0,
            "fields": _AGGREGATE_FIELDS,
            "filter": f"issuerName=={issuer_name}",
        }
        try:
            data = await self._get(url, params=params)
        except Exception as exc:
            logger.warning("trace._fetch_aggregates_by_issuer error: %s", exc)
            return []

        records: list[dict] = data if isinstance(data, list) else data.get("data", [])
        return [_bond_from_record(r) for r in records if r.get("cusip")]

    async def get_bond_by_cusip(self, cusip: str) -> BondQuote | None:
        """Fetch single bond aggregate details by CUSIP."""
        url = f"{self.API_BASE}/name/traceAggregates"
        params = {
            "limit": 1,
            "offset": 0,
            "fields": _AGGREGATE_FIELDS,
            "filter": f"cusip=={cusip}",
        }
        try:
            data = await self._get(url, params=params)
        except Exception as exc:
            logger.warning("trace.get_bond_by_cusip cusip=%s error: %s", cusip, exc)
            return None

        records: list[dict] = data if isinstance(data, list) else data.get("data", [])
        if not records:
            logger.debug("trace.get_bond_by_cusip cusip=%s not found", cusip)
            return None
        return _bond_from_record(records[0])

    async def get_investment_grade_universe(self, limit: int = 100) -> list[BondQuote]:
        """Fetch the most-actively-traded corporate bonds from TRACE aggregates.

        FINRA does not expose an explicit IG filter; we approximate by
        requesting high-trade-count bonds and returning them sorted by
        trade_count descending.
        """
        url = f"{self.API_BASE}/name/traceAggregates"
        params = {
            "limit": limit,
            "offset": 0,
            "fields": _AGGREGATE_FIELDS,
            "sortFields": ["-tradeCount"],
        }
        try:
            data = await self._get(url, params=params)
        except Exception as exc:
            logger.error("trace.get_investment_grade_universe error: %s", exc)
            return []

        records: list[dict] = data if isinstance(data, list) else data.get("data", [])
        quotes = [_bond_from_record(r) for r in records if r.get("cusip")]
        quotes.sort(key=lambda q: q.trade_count or 0, reverse=True)
        logger.info("trace.get_investment_grade_universe fetched=%d", len(quotes))
        return quotes

    async def build_credit_curve(self, issuer_ticker: str) -> CreditCurve | None:
        """Build an issuer credit curve from all available TRACE bonds.

        Sorts by maturity, then for each bond computes spread as
        yield − interpolated Treasury par yield (simple OAS proxy).
        Returns None when fewer than 2 bonds with valid yield+maturity found.
        """
        bonds = await self.search_bonds(issuer_ticker, limit=50)

        today = date.today()
        points: list[dict] = []
        for bond in bonds:
            if bond.maturity_date is None or bond.last_yield is None:
                continue
            years = _years_to_maturity(bond.maturity_date, today)
            if years <= 0:
                continue
            treasury_yield = _interp_treasury(years)
            spread_bps = (bond.last_yield - treasury_yield) * 100.0
            points.append(
                {
                    "maturity_years": round(years, 2),
                    "yield": bond.last_yield,
                    "spread_bps": round(spread_bps, 1),
                    "spread_to_benchmark": bond.spread_to_benchmark,
                    "cusip": bond.cusip,
                    "coupon": bond.coupon,
                    "last_price": float(bond.last_price) if bond.last_price else None,
                }
            )

        if len(points) < 2:
            logger.info(
                "trace.build_credit_curve ticker=%s insufficient bonds (%d)",
                issuer_ticker, len(points),
            )
            return None

        points.sort(key=lambda p: p["maturity_years"])
        logger.info(
            "trace.build_credit_curve ticker=%s points=%d", issuer_ticker, len(points)
        )
        return CreditCurve(issuer=issuer_ticker.upper(), as_of=today, points=points)

    async def get_recent_trades(self, cusip: str, limit: int = 50) -> list[dict]:
        """Fetch individual TRACE trade reports (real OTC transaction data) for a CUSIP.

        Uses the traceAggregates endpoint filtered by CUSIP; individual tick
        data is available only on the FINRA professional feed, so this returns
        the most granular public data (daily aggregates) available via the free API.
        """
        url = f"{self.API_BASE}/name/traceAggregates"
        params = {
            "limit": limit,
            "offset": 0,
            "fields": _TRADE_FIELDS + ",cusip,issuerName,maturityDate,tradeCount",
            "filter": f"cusip=={cusip}",
            "sortFields": ["-lastSaleDate"],
        }
        try:
            data = await self._get(url, params=params)
        except Exception as exc:
            logger.warning("trace.get_recent_trades cusip=%s error: %s", cusip, exc)
            return []

        records: list[dict] = data if isinstance(data, list) else data.get("data", [])
        logger.info("trace.get_recent_trades cusip=%s records=%d", cusip, len(records))
        return records

    async def health_check(self) -> bool:
        """Verify FINRA TRACE API is accessible by fetching a single aggregate record."""
        url = f"{self.API_BASE}/name/traceAggregates"
        params = {"limit": 1, "offset": 0, "fields": "cusip"}
        try:
            data = await self._get(url, params=params)
            ok = bool(data)
            logger.info("trace.health_check ok=%s", ok)
            return ok
        except Exception as exc:
            logger.error("trace.health_check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

async def get_corporate_bonds(issuer_ticker: str) -> list[BondQuote]:
    """Convenience wrapper — create client, search, close."""
    async with TRACEClient() as client:
        return await client.search_bonds(issuer_ticker)
