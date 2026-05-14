"""EDGAR adapter — SEC filing metadata and XBRL companyfacts. Rate limited to 10 req/sec."""
from __future__ import annotations
import asyncio
import json
from datetime import datetime
from typing import Optional
import httpx
from sentinel.core.types import DataHealthEvent
from sentinel.sds.base_adapter import BaseAdapter
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class EDGARAdapter(BaseAdapter):
    name = "edgar"
    rate_limit_per_min = 600  # 10/sec = 600/min; enforced via _throttle with burst

    def __init__(self, user_agent: str = "Sentinel sentinel@example.com") -> None:
        super().__init__()
        self._headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        self._ticker_to_cik: dict[str, str] = {}
        self._cik_to_ticker: dict[str, str] = {}
        self._loaded = False

    async def fetch_ohlcv(self, ticker, start, end, interval="1d", figi=None):
        return []  # EDGAR doesn't provide price data

    async def load_company_tickers(self) -> dict[str, str]:
        """Load all ticker→CIK mappings from bulk SEC file. Call once at startup."""
        if self._loaded:
            return self._ticker_to_cik
        resp = await self._get(
            COMPANY_TICKERS_URL,
            headers={**self._headers, "Host": "www.sec.gov"},
        )
        data = resp.json()
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            name = entry.get("title", "")
            if ticker and cik:
                self._ticker_to_cik[ticker] = cik
                self._cik_to_ticker[cik] = ticker
        self._loaded = True
        logger.info("EDGAR tickers loaded", count=len(self._ticker_to_cik))
        return self._ticker_to_cik

    def ticker_to_cik(self, ticker: str) -> Optional[str]:
        return self._ticker_to_cik.get(ticker.upper())

    async def fetch_submissions(self, cik: str) -> dict:
        """Fetch filing submissions for a company."""
        cik = cik.zfill(10)
        await self._throttle()
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        try:
            resp = await self._get(url, headers=self._headers)
            return resp.json()
        except Exception as exc:
            logger.error("EDGAR submissions error", cik=cik, error=str(exc))
            return {}

    async def fetch_companyfacts(self, cik: str) -> dict:
        """Fetch all XBRL facts for a company — the core of financial data."""
        cik = cik.zfill(10)
        await self._throttle()
        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            resp = await self._get(url, headers=self._headers)
            return resp.json()
        except Exception as exc:
            logger.error("EDGAR companyfacts error", cik=cik, error=str(exc))
            return {}

    async def fetch_companyconcept(
        self, cik: str, taxonomy: str, concept: str
    ) -> dict:
        """Fetch a single XBRL concept history for a company."""
        cik = cik.zfill(10)
        await self._throttle()
        url = f"{EDGAR_BASE}/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
        try:
            resp = await self._get(url, headers=self._headers)
            return resp.json()
        except Exception as exc:
            logger.error("EDGAR concept error", cik=cik, concept=concept, error=str(exc))
            return {}

    async def fetch_frames(
        self, taxonomy: str, concept: str, unit: str, period: str
    ) -> dict:
        """Cross-sectional snapshot — all companies reporting a concept in a period.
        period format: CY2024Q1I (instant) or CY2024Q1 (duration).
        """
        await self._throttle()
        url = f"{EDGAR_BASE}/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json"
        try:
            resp = await self._get(url, headers=self._headers)
            return resp.json()
        except Exception as exc:
            logger.error("EDGAR frames error", concept=concept, period=period, error=str(exc))
            return {}

    async def fulltext_search(
        self,
        query: str,
        form_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        hits: int = 10,
    ) -> list[dict]:
        """Full-text search across all EDGAR filings post-2001."""
        await self._throttle()
        params: dict = {"q": f'"{query}"', "dateRange": "custom", "hits.hits._source": "file_date,period_of_report,entity_name,file_num,form_type"}
        if form_type:
            params["forms"] = form_type
        if date_from:
            params["startdt"] = date_from
        if date_to:
            params["enddt"] = date_to
        url = f"{EDGAR_SEARCH}/LATEST/search-index"
        try:
            resp = await self._get(url, params=params, headers={**self._headers, "Host": "efts.sec.gov"})
            data = resp.json()
            return data.get("hits", {}).get("hits", [])[:hits]
        except Exception as exc:
            logger.error("EDGAR fulltext search error", query=query, error=str(exc))
            return []

    async def fetch_recent_filings(
        self, cik: str, form_type: Optional[str] = None, limit: int = 40
    ) -> list[dict]:
        """Get recent filings for a company, optionally filtered by form type."""
        submissions = await self.fetch_submissions(cik)
        filings = submissions.get("filings", {}).get("recent", {})
        if not filings:
            return []

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        urls = filings.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form_type and form != form_type:
                continue
            results.append({
                "form_type": form,
                "filing_date": dates[i] if i < len(dates) else None,
                "accession": accessions[i] if i < len(accessions) else None,
                "primary_doc": urls[i] if i < len(urls) else None,
                "cik": cik,
            })
            if len(results) >= limit:
                break
        return results

    async def health_check(self) -> DataHealthEvent:
        try:
            tickers = await self.load_company_tickers()
            if len(tickers) > 1000:
                return self._ok_event()
            return self._error_event(f"Only {len(tickers)} tickers loaded")
        except Exception as exc:
            return self._error_event(str(exc))
