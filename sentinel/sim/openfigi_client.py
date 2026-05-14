"""OpenFIGI client — free CUSIP/ISIN/SEDOL/ticker→FIGI resolver. 25K req/min, MIT-licensed.

Three-tier lookup cache (fastest to slowest):
  L1 — in-process dict          (zero latency, lost on restart)
  L2 — PostgreSQL instruments   (sub-ms, survives restarts, free)
  L3 — OpenFIGI API             (network call, 25K/min free tier)

Pass an AsyncSession to ticker_to_figi / bulk_resolve_tickers to activate L2.
Without a session, falls back to L1 → L3 (original behaviour).
"""
from __future__ import annotations
import asyncio
from typing import Optional
import httpx
from sentinel.core.types import Instrument, AssetClass
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_SEARCH_URL = "https://api.openfigi.com/v3/search"
BATCH_SIZE = 100  # Max per request


class OpenFIGIClient:
    """Resolves identifiers to FIGI. L1 in-memory → L2 PostgreSQL → L3 API."""

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["X-OPENFIGI-APIKEY"] = api_key
        self._cache: dict[str, Optional[dict]] = {}  # L1

    # ── L2: PostgreSQL instruments table ──────────────────────────────────────

    async def _l2_get(self, ticker: str, session) -> Optional[str]:
        """Check instruments table for a pre-resolved FIGI."""
        from sqlalchemy import text
        try:
            result = await session.execute(
                text("SELECT figi FROM instruments WHERE ticker = :ticker LIMIT 1"),
                {"ticker": ticker},
            )
            row = result.fetchone()
            return row[0] if row else None
        except Exception as exc:
            logger.warning("L2 cache read failed", ticker=ticker, error=str(exc))
            return None

    async def _l2_put(self, ticker: str, record: dict, session) -> None:
        """Write-through: persist resolved instrument to PostgreSQL."""
        from sentinel.sds.repository import upsert_instrument
        instrument = self.figi_record_to_instrument(record, ticker=ticker)
        if instrument.figi:
            try:
                await upsert_instrument(
                    figi=instrument.figi,
                    ticker=instrument.ticker or ticker,
                    session=session,
                    name=instrument.name,
                    exchange=instrument.exchange,
                    asset_class=instrument.asset_class.value,
                    currency=instrument.currency or "USD",
                )
            except Exception as exc:
                logger.warning("L2 cache write failed", ticker=ticker, error=str(exc))

    # ── L3: OpenFIGI API ──────────────────────────────────────────────────────

    async def map_identifiers(self, jobs: list[dict]) -> list[list[dict]]:
        """Batch map identifiers to FIGI records via OpenFIGI API."""
        results: list[list[dict]] = []
        for chunk_start in range(0, len(jobs), BATCH_SIZE):
            chunk = jobs[chunk_start : chunk_start + BATCH_SIZE]
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(OPENFIGI_URL, json=chunk, headers=self._headers)
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data:
                        results.append(item.get("data", []))
            except Exception as exc:
                logger.error("OpenFIGI map error", error=str(exc))
                results.extend([[] for _ in chunk])
            await asyncio.sleep(0.05)
        return results

    # ── Public resolution methods ─────────────────────────────────────────────

    async def ticker_to_figi(
        self,
        ticker: str,
        exch_code: str = "US",
        security_type: str = "Common Stock",
        session=None,
    ) -> Optional[str]:
        """Resolve a US equity ticker to its composite FIGI.

        Args:
            session: AsyncSession — enables L2 PostgreSQL cache.
                     If None, uses L1 → L3 only.
        """
        cache_key = f"{ticker}:{exch_code}:{security_type}"

        # L1 — in-process dict
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached.get("figi") if cached else None

        # L2 — PostgreSQL instruments table
        if session is not None:
            figi = await self._l2_get(ticker, session)
            if figi:
                self._cache[cache_key] = {"figi": figi, "compositeFIGI": figi}
                logger.debug("FIGI resolved from L2", ticker=ticker, figi=figi)
                return figi

        # L3 — OpenFIGI API
        jobs = [{"idType": "TICKER", "idValue": ticker, "exchCode": exch_code,
                 "securityType": security_type}]
        results = await self.map_identifiers(jobs)
        if results and results[0]:
            record = results[0][0]
            self._cache[cache_key] = record
            if session is not None:
                await self._l2_put(ticker, record, session)
            figi = record.get("compositeFIGI") or record.get("figi")
            logger.debug("FIGI resolved from L3 API", ticker=ticker, figi=figi)
            return figi

        self._cache[cache_key] = None
        return None

    async def isin_to_figi(self, isin: str) -> Optional[str]:
        """Resolve ISIN to composite FIGI."""
        if isin in self._cache:
            cached = self._cache[isin]
            return cached.get("compositeFIGI") if cached else None
        jobs = [{"idType": "ID_ISIN", "idValue": isin}]
        results = await self.map_identifiers(jobs)
        if results and results[0]:
            record = results[0][0]
            self._cache[isin] = record
            return record.get("compositeFIGI") or record.get("figi")
        self._cache[isin] = None
        return None

    async def cusip_to_figi(self, cusip: str) -> Optional[str]:
        """Resolve CUSIP to composite FIGI."""
        if cusip in self._cache:
            cached = self._cache[cusip]
            return cached.get("compositeFIGI") if cached else None
        jobs = [{"idType": "ID_CUSIP", "idValue": cusip}]
        results = await self.map_identifiers(jobs)
        if results and results[0]:
            record = results[0][0]
            self._cache[cusip] = record
            return record.get("compositeFIGI") or record.get("figi")
        self._cache[cusip] = None
        return None

    async def bulk_resolve_tickers(
        self,
        tickers: list[str],
        exch_code: str = "US",
        security_type: str = "Common Stock",
        session=None,
    ) -> dict[str, Optional[str]]:
        """Bulk resolve tickers → {ticker: figi}.

        With session: checks L2 for each uncached ticker before batching L3 calls.
        """
        uncached = [t for t in tickers
                    if f"{t}:{exch_code}:{security_type}" not in self._cache]

        # L2 check for uncached tickers
        if session is not None and uncached:
            l2_hits: list[str] = []
            for t in uncached:
                figi = await self._l2_get(t, session)
                if figi:
                    key = f"{t}:{exch_code}:{security_type}"
                    self._cache[key] = {"figi": figi, "compositeFIGI": figi}
                    l2_hits.append(t)
            uncached = [t for t in uncached
                        if f"{t}:{exch_code}:{security_type}" not in self._cache]

        # L3 API for remaining misses
        if uncached:
            jobs = [{"idType": "TICKER", "idValue": t, "exchCode": exch_code,
                     "securityType": security_type}
                    for t in uncached]
            results = await self.map_identifiers(jobs)
            for ticker, recs in zip(uncached, results):
                key = f"{ticker}:{exch_code}:{security_type}"
                if recs:
                    self._cache[key] = recs[0]
                    if session is not None:
                        await self._l2_put(ticker, recs[0], session)
                else:
                    self._cache[key] = None

        out = {}
        for t in tickers:
            key = f"{t}:{exch_code}:{security_type}"
            cached = self._cache.get(key)
            out[t] = cached.get("compositeFIGI") or cached.get("figi") if cached else None
        return out

    async def search(
        self,
        query: str,
        security_type: Optional[str] = None,
        market_sec_des: Optional[str] = None,
        exch_code: Optional[str] = None,
    ) -> list[dict]:
        """Full-text search across OpenFIGI instrument universe."""
        payload: dict = {"query": query}
        if security_type:
            payload["securityType"] = security_type
        if market_sec_des:
            payload["marketSecDes"] = market_sec_des
        if exch_code:
            payload["exchCode"] = exch_code
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(OPENFIGI_SEARCH_URL, json=payload,
                                         headers=self._headers)
                resp.raise_for_status()
                data = resp.json()
                return data.get("data", [])
        except Exception as exc:
            logger.error("OpenFIGI search error", query=query, error=str(exc))
            return []

    def figi_record_to_instrument(self, record: dict, ticker: str = "") -> Instrument:
        """Convert an OpenFIGI record to an Instrument domain object."""
        sec_type = record.get("securityType", "")
        asset_class = _infer_asset_class(sec_type)
        return Instrument(
            figi=record.get("compositeFIGI") or record.get("figi", ""),
            ticker=record.get("ticker") or ticker,
            name=record.get("name", ""),
            exchange=record.get("exchCode", ""),
            asset_class=asset_class,
            currency=record.get("marketSector", ""),
            security_type=sec_type,
        )


def _infer_asset_class(security_type: str) -> AssetClass:
    st = security_type.lower()
    if "equity" in st or "common" in st or "preferred" in st:
        return AssetClass.EQUITY
    if "etf" in st or "fund" in st:
        return AssetClass.ETF
    if "option" in st:
        return AssetClass.OPTION
    if "future" in st or "warrant" in st:
        return AssetClass.FUTURE
    if "bond" in st or "note" in st or "bill" in st:
        return AssetClass.FIXED_INCOME
    if "crypto" in st or "digital" in st:
        return AssetClass.CRYPTO
    if "currency" in st or "fx" in st:
        return AssetClass.FX
    return AssetClass.EQUITY
