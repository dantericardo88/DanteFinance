"""
EDGAR full-text search (EFTS) — Sentinel Intelligence Layer.

Uses the SEC's EDGAR Full-Text Search System (EFTS) to search the full text
of all SEC filings. No API key required; rate-limited to ~10 req/sec.

EFTS base: https://efts.sec.gov/LATEST/search-index

Typical use-cases:
  - Search for a keyword across all 10-K filings in the past year.
  - Find 8-K filings for a ticker mentioning "acquisition" or "restatement".
  - Surface recent management-change or material-event disclosures.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_ARCHIVE = "https://www.sec.gov"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 20.0
_DEFAULT_FORMS = ["10-K", "10-Q", "8-K", "DEF 14A"]

# Module-level ticker→company-name cache
_ticker_name_cache: dict[str, str] = {}
_cache_loaded = False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EdgarSearchResult(BaseModel):
    accession: str
    ticker: Optional[str] = None
    company_name: str
    cik: str
    form_type: str
    filed_date: date
    period_of_report: Optional[date] = None
    description: str = ""
    url: str
    relevant_excerpts: list[str] = Field(default_factory=list)


class EdgarSearchResponse(BaseModel):
    query: str
    total_hits: int
    results: list[EdgarSearchResult]
    form_types_searched: list[str]
    date_range: str
    search_url: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _load_ticker_names(client: httpx.AsyncClient) -> None:
    """Populate _ticker_name_cache from EDGAR company_tickers.json."""
    global _cache_loaded
    if _cache_loaded:
        return
    try:
        resp = await client.get(_COMPANY_TICKERS_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data: dict = resp.json()
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            title = str(entry.get("title", "")).strip()
            if ticker and title:
                _ticker_name_cache[ticker] = title
        _cache_loaded = True
        logger.debug("edgar_search: loaded %d company names", len(_ticker_name_cache))
    except Exception as exc:
        logger.warning("edgar_search: ticker name cache load failed: %s", exc)


def _build_efts_url(
    query: str,
    form_types: list[str],
    start_date: date,
    end_date: date,
    entity_name: Optional[str] = None,
) -> str:
    """Construct the EFTS search URL with correct query encoding."""
    # EFTS wants the query phrase wrapped in double-quotes for exact-phrase search
    q_encoded = quote(f'"{query}"')
    forms_encoded = ",".join(form_types)

    url = (
        f"{_EFTS_BASE}"
        f"?q={q_encoded}"
        f"&forms={quote(forms_encoded)}"
        f"&dateRange=custom"
        f"&startdt={start_date.isoformat()}"
        f"&enddt={end_date.isoformat()}"
    )
    if entity_name:
        url += f"&entity={quote(entity_name)}"

    return url


def _parse_hit(hit: dict, ticker: Optional[str]) -> Optional[EdgarSearchResult]:
    """Convert one EFTS hit dict into an EdgarSearchResult."""
    src = hit.get("_source", {})
    if not src:
        return None

    accession = src.get("accession_no") or hit.get("_id", "")
    company_name = src.get("entity_name", "")
    if not company_name:
        display_names = src.get("display_names", [])
        if display_names and isinstance(display_names, list):
            dn = display_names[0]
            company_name = dn.get("entity", "") if isinstance(dn, dict) else str(dn)

    cik = str(src.get("cik", "")).zfill(10)
    form_type = src.get("form_type", "")

    file_date_str = src.get("file_date", "")
    try:
        filed_date = date.fromisoformat(file_date_str[:10])
    except (ValueError, TypeError):
        return None

    period_str = src.get("period_of_report", "")
    period_of_report: Optional[date] = None
    if period_str:
        try:
            period_of_report = date.fromisoformat(period_str[:10])
        except (ValueError, TypeError):
            pass

    # Build filing URL
    file_name = src.get("file_name", "")
    accession_nodash = accession.replace("-", "")
    cik_plain = cik.lstrip("0") or cik
    if file_name:
        url = f"{_EDGAR_ARCHIVE}/Archives/edgar/data/{cik_plain}/{accession_nodash}/{file_name}"
    else:
        url = f"{_EDGAR_ARCHIVE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik_plain}&type={quote(form_type)}&dateb=&owner=include&count=40"

    # Relevant excerpts from highlight field
    highlights = hit.get("highlight", {})
    excerpts: list[str] = []
    for _field, snippets in highlights.items():
        if isinstance(snippets, list):
            excerpts.extend(str(s) for s in snippets)
    excerpts = excerpts[:5]  # cap at 5

    description = src.get("description", src.get("form_type", ""))

    return EdgarSearchResult(
        accession=accession,
        ticker=ticker,
        company_name=company_name,
        cik=cik,
        form_type=form_type,
        filed_date=filed_date,
        period_of_report=period_of_report,
        description=description,
        url=url,
        relevant_excerpts=excerpts,
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

async def search_edgar(
    query: str,
    form_types: Optional[list[str]] = None,
    days_back: int = 365,
    ticker: Optional[str] = None,
    limit: int = 20,
) -> EdgarSearchResponse:
    """
    Full-text EDGAR search using EFTS.

    Args:
        query: The phrase to search for (e.g. "climate risk", "going concern").
        form_types: List of SEC form types to restrict to. Defaults to
                    ["10-K", "10-Q", "8-K", "DEF 14A"].
        days_back: Number of calendar days to look back from today.
        ticker: If given, restrict results to this company by resolving
                its name from company_tickers.json and filtering via entity.
        limit: Maximum number of results to return (max 20 per EFTS page).

    Returns:
        EdgarSearchResponse with parsed results.
    """
    if form_types is None:
        form_types = _DEFAULT_FORMS

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    date_range = f"{start_date.isoformat()} to {end_date.isoformat()}"

    entity_name: Optional[str] = None
    if ticker:
        async with httpx.AsyncClient() as client:
            await _load_ticker_names(client)
        entity_name = _ticker_name_cache.get(ticker.upper())

    search_url = _build_efts_url(query, form_types, start_date, end_date, entity_name)

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(search_url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            data: dict = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("edgar_search: HTTP %d for query '%s': %s", exc.response.status_code, query, exc)
            return EdgarSearchResponse(
                query=query,
                total_hits=0,
                results=[],
                form_types_searched=form_types,
                date_range=date_range,
                search_url=search_url,
            )
        except Exception as exc:
            logger.error("edgar_search: request failed for query '%s': %s", query, exc)
            return EdgarSearchResponse(
                query=query,
                total_hits=0,
                results=[],
                form_types_searched=form_types,
                date_range=date_range,
                search_url=search_url,
            )

    hits_wrapper = data.get("hits", {})
    total_hits: int = 0
    total_obj = hits_wrapper.get("total", {})
    if isinstance(total_obj, dict):
        total_hits = int(total_obj.get("value", 0))
    elif isinstance(total_obj, int):
        total_hits = total_obj

    raw_hits: list[dict] = hits_wrapper.get("hits", [])

    results: list[EdgarSearchResult] = []
    for hit in raw_hits[:limit]:
        parsed = _parse_hit(hit, ticker)
        if parsed is not None:
            results.append(parsed)

    logger.info(
        "edgar_search: query='%s' forms=%s hits=%d returned=%d",
        query, form_types, total_hits, len(results),
    )

    return EdgarSearchResponse(
        query=query,
        total_hits=total_hits,
        results=results,
        form_types_searched=form_types,
        date_range=date_range,
        search_url=search_url,
    )


async def search_company_filings(
    ticker: str,
    query: str,
    form_type: str = "10-K",
) -> EdgarSearchResponse:
    """
    Convenience wrapper: search filings of a single company for a query phrase.

    Args:
        ticker: Company ticker symbol (e.g. "AAPL").
        query: Full-text phrase to search for within filings.
        form_type: SEC form type to restrict to.

    Returns:
        EdgarSearchResponse filtered to the given company.
    """
    return await search_edgar(
        query=query,
        form_types=[form_type],
        days_back=1825,   # 5 years back for single-company searches
        ticker=ticker,
        limit=20,
    )


async def get_recent_8k_events(
    ticker: str,
    days_back: int = 90,
) -> EdgarSearchResponse:
    """
    Retrieve recent 8-K material-event filings for a ticker.

    8-K filings cover earnings, M&A, management changes, bankruptcy, and
    other material events. This function returns all 8-K filings (and their
    amendments, 8-K/A) for a company in the given lookback window.

    Args:
        ticker: Company ticker symbol (e.g. "MSFT").
        days_back: Lookback window in calendar days.

    Returns:
        EdgarSearchResponse with 8-K results sorted newest-first.
    """
    # Use the ticker symbol as the search phrase so EFTS can match company name.
    # We also pass the entity filter via ticker resolution.
    async with httpx.AsyncClient() as client:
        await _load_ticker_names(client)

    company_name = _ticker_name_cache.get(ticker.upper(), ticker)
    # Search for the company name in 8-K filings; a short company name is
    # usually unique enough for EFTS to filter correctly.
    response = await search_edgar(
        query=company_name,
        form_types=["8-K", "8-K/A"],
        days_back=days_back,
        ticker=ticker,
        limit=20,
    )

    # Sort newest first
    response.results.sort(key=lambda r: r.filed_date, reverse=True)

    logger.info(
        "edgar_search: 8-K events for %s: %d results (days_back=%d)",
        ticker, len(response.results), days_back,
    )
    return response
