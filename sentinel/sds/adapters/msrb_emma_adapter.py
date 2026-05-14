from __future__ import annotations
import asyncio
from datetime import date, timedelta
from typing import Optional
import httpx
from pydantic import BaseModel, Field
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


def _parse_date(val: str | None) -> Optional[date]:
    if not val:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            from datetime import datetime as dt
            return dt.strptime(val[:10], fmt[:8] if "T" in fmt else fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _parse_bond(raw: dict) -> MuniBond:
    return MuniBond(
        cusip=raw.get("cusip") or raw.get("cusipNumber") or "",
        issuer_name=raw.get("issuerName") or raw.get("issuer", {}).get("name", "") if isinstance(raw.get("issuer"), dict) else raw.get("issuerName", ""),
        description=raw.get("description") or raw.get("securityDescription") or "",
        state=raw.get("stateCode") or raw.get("state") or "",
        security_type=raw.get("securityType") or raw.get("securityTypeDescription") or "",
        maturity_date=_parse_date(raw.get("maturityDate")),
        coupon=_float_or_none(raw.get("couponRate") or raw.get("interestRate")),
        interest_payment_frequency=raw.get("interestPaymentFrequency") or raw.get("paymentFrequency") or "",
        outstanding_principal=_float_or_none(raw.get("outstandingPrincipalAmount") or raw.get("outstandingPrincipal")),
        tax_status=raw.get("taxStatus") or raw.get("federalTaxStatus"),
    )


def _float_or_none(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _years_to_maturity(maturity_date: Optional[date]) -> Optional[float]:
    if maturity_date is None:
        return None
    today = date.today()
    delta = (maturity_date - today).days
    if delta <= 0:
        return None
    return round(delta / 365.25, 4)


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
        tey = muni_yield / (1 - tax_rate)
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


async def get_muni_curve(state: str) -> list[MuniYieldPoint]:
    adapter = MSRBEmmaAdapter()
    return await adapter.get_yield_curve_by_state(state)


async def screen_munis(state: str, min_yield: float) -> MuniScreenResult:
    adapter = MSRBEmmaAdapter()
    return await adapter.screen_munis(state=state, min_yield=min_yield)
