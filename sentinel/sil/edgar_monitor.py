"""EDGAR real-time filing monitor — Sentinel Intelligence Layer.

Polls EFTS for recent filings by form type or ticker. Equivalent to Bloomberg
filing alerts at zero cost. Cache: ~/.sentinel/edgar_monitor.json.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_HEADERS = {
    "User-Agent": "SENTINEL/1.0 research@sentinel.ai",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 15.0
_CACHE_PATH = Path.home() / ".sentinel" / "edgar_monitor.json"

MONITORED_FORMS: dict[str, str] = {
    "8-K":    "Material event (earnings, M&A, management change, etc.)",
    "10-K":   "Annual report",
    "10-Q":   "Quarterly report",
    "DEF 14A": "Proxy statement (shareholder vote)",
    "SC 13D": "Activist investor (>5% stake + intent to influence)",
    "SC 13G": "Passive institutional investor (>5% stake)",
    "4":      "Insider transaction (Form 4)",
    "S-1":    "IPO registration",
    "424B4":  "Prospectus (offering)",
    "6-K":    "Foreign private issuer material event",
    "20-F":   "Foreign private issuer annual report",
}

# Keywords that escalate an 8-K to "high" priority
_HIGH_KEYWORDS = frozenset({"merger", "acquisition", "acquire", "acquires", "buyout",
                             "takeover", "management change", "ceo", "resign"})


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EDGARFiling(BaseModel):
    entity_name: str
    entity_id: str
    form_type: str
    file_date: str
    period_of_report: Optional[str] = None
    description: str
    filing_url: str
    is_new: bool = True


class FilingAlert(BaseModel):
    filing: EDGARFiling
    alert_reason: str
    priority: str  # "high" | "medium" | "low"


class MonitorResult(BaseModel):
    filings: list[EDGARFiling]
    alerts: list[FilingAlert]
    total_found: int
    new_count: int
    form_type_counts: dict[str, int]
    as_of: str
    warnings: list[str] = Field(default_factory=list)


class WatchlistMonitorResult(BaseModel):
    ticker: str
    entity_id: Optional[str] = None
    filings: list[EDGARFiling]
    alerts: list[FilingAlert]
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Persistence helpers (sync, called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, bool]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("edgar_monitor.cache_load_failed", path=str(_CACHE_PATH), error=str(exc))
        return {}


def _save_cache(cache: dict[str, bool]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cache_key(entity_id: str, form_type: str, file_date: str) -> str:
    return f"{entity_id}_{form_type}_{file_date}"


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def _filing_url(entity_id: str, form_type: str, accession_num: Optional[str] = None) -> str:
    if accession_num:
        acc = accession_num.replace("-", "")
        cik = entity_id.lstrip("0")
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
    return (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={entity_id}"
        f"&type={quote(form_type)}&dateb=&owner=include&count=10"
    )


# ---------------------------------------------------------------------------
# EFTS helpers
# ---------------------------------------------------------------------------

def _efts_url(
    form_types: list[str],
    start_date: date,
    end_date: date,
    ticker_query: Optional[str] = None,
    limit: int = 50,
) -> str:
    forms_param = quote(",".join(form_types))
    url = (
        f"{_EFTS_BASE}"
        f"?forms={forms_param}"
        f"&dateRange=custom"
        f"&startdt={start_date.isoformat()}"
        f"&enddt={end_date.isoformat()}"
        f"&hits.hits.total=true"
        f"&hits.hits._source=period_of_report,display_names,entity_id,file_date,form_type,file_num,accession_no"
    )
    if ticker_query:
        url += f"&q={quote(chr(34) + ticker_query + chr(34))}"
    return url


def _parse_hits(raw_hits: list[dict], cache: dict[str, bool]) -> tuple[list[EDGARFiling], dict[str, bool]]:
    """Parse EFTS hit list into EDGARFiling objects, marking which are new."""
    filings: list[EDGARFiling] = []
    for hit in raw_hits:
        src = hit.get("_source", {})
        if not src:
            continue

        display_names = src.get("display_names", [])
        entity_name = ""
        entity_id = str(src.get("entity_id", ""))
        if display_names and isinstance(display_names, list):
            dn = display_names[0]
            if isinstance(dn, dict):
                entity_name = dn.get("name", "") or dn.get("entity", "")
                if not entity_id:
                    entity_id = str(dn.get("entity_id", ""))
            else:
                entity_name = str(dn)

        form_type = src.get("form_type", "")
        file_date = src.get("file_date", "")
        period = src.get("period_of_report")
        accession = src.get("accession_no")

        if not (entity_id and form_type and file_date):
            continue

        key = _cache_key(entity_id, form_type, file_date)
        is_new = key not in cache
        if is_new:
            cache[key] = True

        filing = EDGARFiling(
            entity_name=entity_name or f"CIK {entity_id}",
            entity_id=entity_id,
            form_type=form_type,
            file_date=file_date[:10],
            period_of_report=period[:10] if period else None,
            description=MONITORED_FORMS.get(form_type, form_type),
            filing_url=_filing_url(entity_id, form_type, accession),
            is_new=is_new,
        )
        filings.append(filing)
    return filings, cache


# ---------------------------------------------------------------------------
# Alert priority logic
# ---------------------------------------------------------------------------

def _classify_alert(filing: EDGARFiling) -> Optional[FilingAlert]:
    ft = filing.form_type
    name_lower = filing.entity_name.lower()

    if ft == "SC 13D":
        return FilingAlert(
            filing=filing,
            alert_reason="Activist investor disclosed >5% stake with intent to influence",
            priority="high",
        )
    if ft == "S-1":
        return FilingAlert(
            filing=filing,
            alert_reason="IPO registration statement filed",
            priority="high",
        )
    if ft == "8-K":
        for kw in _HIGH_KEYWORDS:
            if kw in name_lower or kw in filing.description.lower():
                return FilingAlert(
                    filing=filing,
                    alert_reason=f"Material 8-K possibly related to: {kw}",
                    priority="high",
                )
        return FilingAlert(
            filing=filing,
            alert_reason="Material event 8-K filed",
            priority="low",
        )
    if ft in ("10-K", "10-Q", "DEF 14A"):
        return FilingAlert(
            filing=filing,
            alert_reason=f"Periodic report filed: {ft}",
            priority="medium",
        )
    if ft == "4":
        return FilingAlert(
            filing=filing,
            alert_reason="Insider transaction reported (Form 4)",
            priority="medium",
        )
    if ft in ("SC 13G", "6-K"):
        return FilingAlert(
            filing=filing,
            alert_reason=f"{MONITORED_FORMS.get(ft, ft)} filed",
            priority="low",
        )
    return None


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------

async def _resolve_cik(ticker: str, client: httpx.AsyncClient) -> Optional[str]:
    """Resolve a ticker to its EDGAR CIK using EFTS."""
    url = (
        f"{_EFTS_BASE}"
        f"?q={quote(chr(34) + ticker.upper() + chr(34))}"
        f"&forms=10-K"
        f"&hits.hits._source=entity_id,display_names"
    )
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            src = hits[0].get("_source", {})
            entity_id = str(src.get("entity_id", ""))
            if entity_id:
                return entity_id
            display_names = src.get("display_names", [])
            if display_names and isinstance(display_names[0], dict):
                return str(display_names[0].get("entity_id", ""))
    except Exception as exc:
        logger.warning("edgar_monitor.cik_resolve_failed", ticker=ticker, error=str(exc))
    return None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def get_recent_filings(
    form_types: list[str] | None = None,
    days_back: int = 1,
    limit: int = 50,
) -> MonitorResult:
    """Fetch recent EDGAR filings across all companies for specified form types."""
    if form_types is None:
        form_types = list(MONITORED_FORMS.keys())

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    warnings: list[str] = []

    url = _efts_url(form_types, start_date, end_date, limit=limit)

    cache = await asyncio.to_thread(_load_cache)

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            warnings.append(f"EFTS HTTP {exc.response.status_code}: {exc}")
            logger.error("edgar_monitor.http_error", status=exc.response.status_code, url=url)
            data = {}
        except Exception as exc:
            warnings.append(f"EFTS request failed: {exc}")
            logger.error("edgar_monitor.request_failed", error=str(exc))
            data = {}

    hits_wrapper = data.get("hits", {})
    total_obj = hits_wrapper.get("total", {})
    total_found = int(total_obj.get("value", 0)) if isinstance(total_obj, dict) else int(total_obj or 0)
    raw_hits: list[dict] = hits_wrapper.get("hits", [])[:limit]

    filings, cache = _parse_hits(raw_hits, cache)
    await asyncio.to_thread(_save_cache, cache)

    new_count = sum(1 for f in filings if f.is_new)
    form_type_counts: dict[str, int] = {}
    for f in filings:
        form_type_counts[f.form_type] = form_type_counts.get(f.form_type, 0) + 1

    alerts: list[FilingAlert] = []
    for f in filings:
        if f.is_new:
            alert = _classify_alert(f)
            if alert:
                alerts.append(alert)

    logger.info(
        "edgar_monitor.get_recent_filings",
        total_found=total_found,
        returned=len(filings),
        new=new_count,
        alerts=len(alerts),
    )

    return MonitorResult(
        filings=filings,
        alerts=alerts,
        total_found=total_found,
        new_count=new_count,
        form_type_counts=form_type_counts,
        as_of=datetime.utcnow().isoformat(),
        warnings=warnings,
    )


async def _fetch_ticker_filings(
    ticker: str,
    form_types: list[str],
    start_date: date,
    end_date: date,
    client: httpx.AsyncClient,
    cache: dict[str, bool],
) -> WatchlistMonitorResult:
    """Fetch filings for a single ticker (used internally by monitor_watchlist)."""
    warnings: list[str] = []
    as_of = datetime.utcnow().isoformat()

    entity_id = await _resolve_cik(ticker, client)
    if not entity_id:
        warnings.append(f"Could not resolve CIK for ticker {ticker}")
        return WatchlistMonitorResult(
            ticker=ticker,
            entity_id=None,
            filings=[],
            alerts=[],
            as_of=as_of,
            warnings=warnings,
        )

    url = _efts_url(form_types, start_date, end_date, ticker_query=ticker)

    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        warnings.append(f"EFTS request failed for {ticker}: {exc}")
        logger.warning("edgar_monitor.ticker_fetch_failed", ticker=ticker, error=str(exc))
        data = {}

    raw_hits: list[dict] = data.get("hits", {}).get("hits", [])

    # Filter to this entity only to avoid cross-ticker noise from text search
    filtered_hits = [
        h for h in raw_hits
        if str(h.get("_source", {}).get("entity_id", "")) == entity_id
    ]
    filings, cache = _parse_hits(filtered_hits, cache)

    alerts: list[FilingAlert] = []
    for f in filings:
        if f.is_new:
            alert = _classify_alert(f)
            if alert:
                alerts.append(alert)

    logger.info(
        "edgar_monitor.ticker_result",
        ticker=ticker,
        entity_id=entity_id,
        filings=len(filings),
        alerts=len(alerts),
    )

    return WatchlistMonitorResult(
        ticker=ticker,
        entity_id=entity_id,
        filings=filings,
        alerts=alerts,
        as_of=as_of,
        warnings=warnings,
    )


async def monitor_watchlist(
    tickers: list[str],
    form_types: list[str] | None = None,
    days_back: int = 7,
) -> list[WatchlistMonitorResult]:
    """Monitor specific tickers for new filings. Returns one result per ticker."""
    if form_types is None:
        form_types = list(MONITORED_FORMS.keys())

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    cache = await asyncio.to_thread(_load_cache)

    async with httpx.AsyncClient() as client:
        tasks = [
            _fetch_ticker_filings(ticker.upper(), form_types, start_date, end_date, client, cache)
            for ticker in tickers
        ]
        results: list[WatchlistMonitorResult] = await asyncio.gather(*tasks)

    await asyncio.to_thread(_save_cache, cache)
    return results


async def get_insider_transactions(
    ticker: str,
    days_back: int = 30,
    limit: int = 20,
) -> list[EDGARFiling]:
    """Get Form 4 insider transactions for a specific ticker."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    warnings: list[str] = []

    cache = await asyncio.to_thread(_load_cache)

    async with httpx.AsyncClient() as client:
        entity_id = await _resolve_cik(ticker.upper(), client)
        if not entity_id:
            warnings.append(f"Could not resolve CIK for {ticker}")
            logger.warning("edgar_monitor.insider_cik_failed", ticker=ticker)
            return []

        url = _efts_url(["4"], start_date, end_date, ticker_query=ticker.upper(), limit=limit)

        try:
            resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("edgar_monitor.insider_fetch_failed", ticker=ticker, error=str(exc))
            return []

    raw_hits: list[dict] = data.get("hits", {}).get("hits", [])[:limit]
    filtered = [
        h for h in raw_hits
        if str(h.get("_source", {}).get("entity_id", "")) == entity_id
    ]
    filings, cache = _parse_hits(filtered, cache)
    await asyncio.to_thread(_save_cache, cache)

    logger.info(
        "edgar_monitor.insider_transactions",
        ticker=ticker,
        entity_id=entity_id,
        count=len(filings),
    )
    return filings
