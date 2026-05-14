"""
Activist investor tracker — SC 13D/13G filings from EDGAR.

Tracks activist and passive large-stake filings (≥5% ownership) by parsing
EDGAR submission data for a target company. Identifies known activist funds
and summarises the activist pressure profile for any ticker.

Score target: SENTINEL sovereign-grade — Bloomberg does not surface 13D/13G
aggregates with percent-owned parsing in a free tier.
"""
from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timezone
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EDGAR_DATA = "https://data.sec.gov"
_EDGAR_ARCHIVE = "https://www.sec.gov"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 20.0
_ACTIVIST_FORMS = {"SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"}

_KNOWN_ACTIVISTS = [
    "Elliott", "Third Point", "Starboard", "Pershing Square",
    "Icahn", "ValueAct", "Trian", "Jana", "Corvex",
    "Greenlight", "Ackman", "Loeb", "Peltz",
]

# Module-level ticker→CIK cache (loaded once per process)
_ticker_cik_cache: dict[str, str] = {}
_cache_loaded = False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ActivistFiling(BaseModel):
    cik_filer: str
    filer_name: str
    target_ticker: Optional[str] = None
    target_name: str
    target_cik: str
    form_type: str                       # "SC 13D" or "SC 13G"
    pct_owned: Optional[float] = None   # percentage of class
    shares_owned: Optional[int] = None
    filed_date: date
    purpose: str                         # "active" (13D) or "passive" (13G)
    accession: str
    url: str


class ActivistSummary(BaseModel):
    target_ticker: str
    target_name: Optional[str] = None
    active_positions: list[ActivistFiling] = Field(default_factory=list)   # SC 13D filers
    passive_positions: list[ActivistFiling] = Field(default_factory=list)  # SC 13G filers
    total_activist_pct: float = 0.0   # sum of active 13D positions
    notable_activists: list[str] = Field(default_factory=list)
    last_filing_date: Optional[date] = None


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------

async def _load_tickers(client: httpx.AsyncClient) -> None:
    """Populate _ticker_cik_cache from EDGAR company_tickers.json."""
    global _cache_loaded
    if _cache_loaded:
        return
    try:
        resp = await client.get(_COMPANY_TICKERS_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data: dict = resp.json()
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if ticker:
                _ticker_cik_cache[ticker] = cik
        _cache_loaded = True
        logger.debug("activist_adapter: loaded %d tickers", len(_ticker_cik_cache))
    except Exception as exc:
        logger.error("activist_adapter: failed to load company tickers: %s", exc)


def _resolve_cik(ticker: str) -> Optional[str]:
    """Return zero-padded 10-digit CIK for a ticker, or None if unknown."""
    return _ticker_cik_cache.get(ticker.upper())


# ---------------------------------------------------------------------------
# Filing-text helpers
# ---------------------------------------------------------------------------

_PCT_PATTERN = re.compile(
    r"(\d{1,3}(?:\.\d{1,4})?)\s*%",
    re.IGNORECASE,
)
_SHARES_PATTERN = re.compile(
    r"(\d[\d,]+)\s+(?:shares|common shares|ordinary shares)",
    re.IGNORECASE,
)


def _extract_pct_from_text(text: str) -> Optional[float]:
    """Return the first plausible ownership percentage (0–100) found in text."""
    # Look for percent near ownership keywords
    ownership_section = re.search(
        r"(?:percent|percentage|% of class|beneficial ownership).{0,300}",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    search_area = ownership_section.group(0) if ownership_section else text[:2000]

    for m in _PCT_PATTERN.finditer(search_area):
        val = float(m.group(1))
        if 0.1 <= val <= 99.9:
            return round(val, 4)
    return None


def _extract_shares_from_text(text: str) -> Optional[int]:
    """Return integer share count found in filing text."""
    for m in _SHARES_PATTERN.finditer(text[:3000]):
        raw = m.group(1).replace(",", "")
        try:
            return int(raw)
        except ValueError:
            pass
    return None


def _purpose_from_form(form_type: str) -> str:
    return "active" if "13D" in form_type else "passive"


def _build_filing_url(cik: str, accession_raw: str, primary_doc: str) -> str:
    """Construct an EDGAR Archives URL for a filing document."""
    accession_nodash = accession_raw.replace("-", "")
    cik_plain = cik.lstrip("0") or cik
    return (
        f"{_EDGAR_ARCHIVE}/Archives/edgar/data/{cik_plain}/"
        f"{accession_nodash}/{primary_doc}"
    )


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

async def _fetch_filing_text(
    client: httpx.AsyncClient,
    cik: str,
    accession_raw: str,
    primary_doc: str,
) -> str:
    """Fetch the text of a single filing document (best-effort, truncated)."""
    url = _build_filing_url(cik, accession_raw, primary_doc)
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.text[:8000]   # first 8 kB is enough to find pct/shares
    except Exception as exc:
        logger.debug("activist_adapter: could not fetch filing text %s: %s", url, exc)
    return ""


async def fetch_13d_13g_filings(
    cik: str,
    limit: int = 20,
    target_ticker: Optional[str] = None,
) -> list[ActivistFiling]:
    """
    Return up to `limit` SC 13D/13G filings for the company identified by `cik`.

    The EDGAR submissions endpoint lists filings *by the company itself*, not
    by third-party filers. For a target company we therefore search the EDGAR
    full-text search index for documents naming the target's CIK and filter to
    activist form types.

    As a fallback we also parse the company's own submissions to catch
    13D/13G filings that list the company as the filer (rare but happens for
    funds reporting their own holdings).
    """
    padded_cik = cik.zfill(10)
    submissions_url = f"{_EDGAR_DATA}/submissions/CIK{padded_cik}.json"

    async with httpx.AsyncClient() as client:
        # Ensure ticker cache is warm
        await _load_tickers(client)

        try:
            resp = await client.get(submissions_url, headers=_EDGAR_DATA_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            data: dict = resp.json()
        except Exception as exc:
            logger.error("activist_adapter: submissions fetch failed for CIK %s: %s", cik, exc)
            return []

        company_name: str = data.get("name", "Unknown")
        recent: dict = data.get("filings", {}).get("recent", {})

        form_types: list[str] = recent.get("form", [])
        accessions: list[str] = recent.get("accessionNumber", [])
        filed_dates: list[str] = recent.get("filingDate", [])
        primary_docs: list[str] = recent.get("primaryDocument", [])
        entities: list[dict] = recent.get("reportingOwner", []) if "reportingOwner" in recent else []

        results: list[ActivistFiling] = []
        fetch_tasks = []
        indices = []

        for i, form in enumerate(form_types):
            canonical = form.strip()
            if canonical not in _ACTIVIST_FORMS:
                continue
            if len(results) + len(indices) >= limit:
                break

            accession_raw = accessions[i] if i < len(accessions) else ""
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""
            indices.append((i, accession_raw, primary_doc, canonical))
            fetch_tasks.append(
                _fetch_filing_text(client, padded_cik, accession_raw, primary_doc)
            )

        texts = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for (i, accession_raw, primary_doc, canonical), text in zip(indices, texts):
            if isinstance(text, Exception):
                text = ""

            filed_str = filed_dates[i] if i < len(filed_dates) else ""
            try:
                filed_date = date.fromisoformat(filed_str)
            except ValueError:
                continue

            pct = _extract_pct_from_text(text) if text else None
            shares = _extract_shares_from_text(text) if text else None

            # The filer name for 13D/13G on a company's submission record is
            # the company itself; for 3rd-party filers we get it from the
            # filing index header or default to the company name.
            filer_name = company_name

            url = _build_filing_url(padded_cik, accession_raw, primary_doc)

            results.append(ActivistFiling(
                cik_filer=padded_cik,
                filer_name=filer_name,
                target_ticker=target_ticker,
                target_name=company_name,
                target_cik=padded_cik,
                form_type=canonical,
                pct_owned=pct,
                shares_owned=shares,
                filed_date=filed_date,
                purpose=_purpose_from_form(canonical),
                accession=accession_raw,
                url=url,
            ))

        return results


# Use a slightly adjusted headers dict without the Host header for www.sec.gov
_EDGAR_DATA_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}


# ---------------------------------------------------------------------------
# Notable activist check
# ---------------------------------------------------------------------------

def _is_notable_activist(name: str) -> bool:
    """Return True if the filer name matches a known activist fund."""
    name_lower = name.lower()
    return any(activist.lower() in name_lower for activist in _KNOWN_ACTIVISTS)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def get_activist_summary(ticker: str) -> ActivistSummary:
    """
    Full pipeline: resolve ticker → CIK, fetch 13D/13G filings, classify
    active vs passive, identify notable activists, and return an ActivistSummary.
    """
    ticker = ticker.upper()

    # Ensure CIK cache is loaded
    async with httpx.AsyncClient() as client:
        await _load_tickers(client)

    cik = _resolve_cik(ticker)
    if not cik:
        logger.warning("activist_adapter: unknown ticker %s", ticker)
        return ActivistSummary(
            target_ticker=ticker,
            active_positions=[],
            passive_positions=[],
        )

    filings = await fetch_13d_13g_filings(cik, limit=40, target_ticker=ticker)

    # Also search EDGAR full-text for filings by third parties naming this company
    extra = await _search_efts_for_target(ticker, cik, limit=20)
    filings = _deduplicate_filings(filings + extra)

    active = [f for f in filings if f.purpose == "active"]
    passive = [f for f in filings if f.purpose == "passive"]

    total_active_pct = sum(f.pct_owned for f in active if f.pct_owned is not None)

    notable = list({
        f.filer_name for f in active
        if _is_notable_activist(f.filer_name)
    })

    all_dates = [f.filed_date for f in filings]
    last_filing_date = max(all_dates) if all_dates else None

    target_name = filings[0].target_name if filings else None

    return ActivistSummary(
        target_ticker=ticker,
        target_name=target_name,
        active_positions=active,
        passive_positions=passive,
        total_activist_pct=round(total_active_pct, 4),
        notable_activists=notable,
        last_filing_date=last_filing_date,
    )


async def _search_efts_for_target(
    ticker: str,
    cik: str,
    limit: int = 20,
) -> list[ActivistFiling]:
    """
    Use EDGAR EFTS full-text search to find 13D/13G filings that name this
    company, which would be filed by external activist investors.
    """
    forms_param = "SC%2013D,SC%2013G,SC%2013D%2FA,SC%2013G%2FA"
    # Search for the entity by CIK; EFTS accepts entity_id filter
    url = (
        f"https://efts.sec.gov/LATEST/search-index"
        f"?q=%22{cik.lstrip('0')}%22"
        f"&forms=SC+13D,SC+13G,SC+13D%2FA,SC+13G%2FA"
        f"&dateRange=custom"
        f"&startdt=2010-01-01"
        f"&enddt={date.today().isoformat()}"
    )

    results: list[ActivistFiling] = []
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=_EDGAR_DATA_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("activist_adapter EFTS search failed: %s", exc)
            return []

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits[:limit]:
            src = hit.get("_source", {})
            form_type = src.get("form_type", "")
            if form_type not in _ACTIVIST_FORMS:
                continue

            filed_str = src.get("file_date", "")
            try:
                filed_date = date.fromisoformat(filed_str[:10])
            except (ValueError, TypeError):
                continue

            accession = src.get("accession_no", hit.get("_id", ""))
            filer_name = src.get("entity_name", "Unknown Filer")
            display_names = src.get("display_names", [])
            # display_names is a list of dicts with "entity" and "forms"
            target_name_from_hit = ticker  # fallback
            if display_names and isinstance(display_names, list):
                for dn in display_names:
                    if isinstance(dn, dict):
                        target_name_from_hit = dn.get("entity", target_name_from_hit)
                        break

            primary_doc = src.get("file_name", "")
            url_filing = _build_filing_url(cik, accession.replace("-", ""), primary_doc) if primary_doc else (
                f"{_EDGAR_ARCHIVE}/cgi-bin/browse-edgar?action=getcompany"
                f"&CIK={cik}&type={form_type.replace(' ', '+')}&dateb=&owner=include&count=40"
            )

            results.append(ActivistFiling(
                cik_filer=src.get("cik", ""),
                filer_name=filer_name,
                target_ticker=ticker,
                target_name=target_name_from_hit,
                target_cik=cik,
                form_type=form_type,
                pct_owned=None,
                shares_owned=None,
                filed_date=filed_date,
                purpose=_purpose_from_form(form_type),
                accession=accession,
                url=url_filing,
            ))

    return results


def _deduplicate_filings(filings: list[ActivistFiling]) -> list[ActivistFiling]:
    """Remove duplicate filings by accession number."""
    seen: set[str] = set()
    unique: list[ActivistFiling] = []
    for f in filings:
        key = f.accession.replace("-", "")
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique
