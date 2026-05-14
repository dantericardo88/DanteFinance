"""
M&A Deal Flow Screener — Dimension #25 (target 4/10).

Screens EDGAR filings for M&A activity: mergers, acquisitions, spinoffs,
and divestitures via EDGAR EFTS full-text search.  No API key required.

Public API
----------
screen_ma_deals(days_back, min_value_billions, sector, deal_type, limit) → MAScreenResult
get_ma_profile(ticker) → MATargetProfile
"""
from __future__ import annotations

import asyncio
import re
from datetime import date, timedelta
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
_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 20.0

# (query_phrase, form_types, deal_type_hint)
_MA_QUERIES: list[tuple[str, list[str], str]] = [
    ("merger agreement",               ["8-K"],      "merger"),
    ("agreement and plan of merger",   ["8-K"],      "merger"),
    ("acquisition agreement",          ["8-K"],      "acquisition"),
    ("definitive agreement to acquire",["8-K"],      "acquisition"),
    ("spin-off",                       ["8-K"],      "spinoff"),
    ("definitive agreement to sell",   ["8-K"],      "divestiture"),
    ("merger agreement",               ["DEFM14A"],  "merger"),
    ("tender offer",                   ["SC TO-T"],  "acquisition"),
]

_DEFENSE_MECHANISMS = [
    "poison pill", "staggered board", "supermajority",
    "classified board", "rights plan",
]

# Lightweight sector → keywords map for entity-name / description filtering
_SECTOR_KW: dict[str, list[str]] = {
    "technology":    ["software", "semiconductor", "tech", "computing", "cloud"],
    "healthcare":    ["pharma", "biotech", "medical", "health", "therapeutics"],
    "financials":    ["bank", "insurance", "financial", "capital"],
    "energy":        ["energy", "oil", "gas", "petroleum", "pipeline", "renewable"],
    "industrials":   ["industrial", "manufacturing", "aerospace", "defense", "logistics"],
    "materials":     ["chemicals", "mining", "metals", "materials"],
    "consumer":      ["retail", "consumer", "food", "beverage", "apparel"],
    "real estate":   ["reit", "real estate", "property", "realty"],
    "utilities":     ["utility", "utilities", "electric", "water"],
    "communication": ["media", "telecom", "broadcasting", "entertainment"],
}

_cik_cache: dict[str, str] = {}
_cik_cache_loaded = False

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MADeal(BaseModel):
    acquirer: Optional[str] = None
    target: Optional[str] = None
    ticker: Optional[str] = None
    deal_type: str                      # "merger"|"acquisition"|"spinoff"|"divestiture"|"unknown"
    deal_value_billions: Optional[float] = None
    filing_date: str                    # ISO date
    status: str                         # "announced"|"pending"|"completed"|"terminated"
    description: str
    form_type: str
    accession_number: str
    edgar_url: str


class MAScreenResult(BaseModel):
    deals: list[MADeal]
    total_found: int
    days_back: int
    filters_applied: dict
    as_of: str
    warnings: list[str]


class MATargetProfile(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    recent_deals: list[MADeal] = Field(default_factory=list)
    is_acquisition_target: bool = False
    activist_interest: bool = False
    defense_mechanisms: list[str] = Field(default_factory=list)
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers: extraction / classification
# ---------------------------------------------------------------------------

_VALUE_RE = re.compile(
    r'\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B\b|M\b)', re.IGNORECASE
)


def _extract_deal_value(text: str) -> Optional[float]:
    best: Optional[float] = None
    for m in _VALUE_RE.finditer(text):
        try:
            amount = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        val = amount if m.group(2).lower() in ("billion", "b") else amount / 1_000.0
        if best is None or val > best:
            best = val
    return round(best, 3) if best is not None else None


def _classify_deal_type(form_type: str, text: str) -> str:
    t, ft = text.lower(), form_type.upper()
    if ft in ("DEFM14A", "DEF14A") or "plan of merger" in t or "merger agreement" in t:
        return "merger"
    if ft in ("SC TO-T", "SC TO-I") or "tender offer" in t:
        return "acquisition"
    if "spin-off" in t or "spinoff" in t or "spin off" in t:
        return "spinoff"
    if "divestiture" in t or "sale of assets" in t or "definitive agreement to sell" in t:
        return "divestiture"
    if "acquisition agreement" in t or "definitive agreement to acquire" in t:
        return "acquisition"
    return "unknown"


def _detect_status(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in ("consummated", "completed", "closed", "transaction closed")):
        return "completed"
    if any(kw in t for kw in ("terminated", "withdrawn", "abandoned")):
        return "terminated"
    if "definitive agreement" in t or "announced" in t:
        return "announced"
    return "pending"


def _matches_sector(company_name: str, description: str, sector: str) -> bool:
    keywords = _SECTOR_KW.get(sector.lower(), [])
    if not keywords:
        return True
    combined = (company_name + " " + description).lower()
    return any(kw in combined for kw in keywords)


def _accession_to_url(accession_no: str, cik: str) -> str:
    acc_nodash = accession_no.replace("-", "")
    cik_plain = cik.lstrip("0") or cik
    return f"{_EDGAR_ARCHIVE}/data/{cik_plain}/{acc_nodash}/"


# ---------------------------------------------------------------------------
# EFTS networking
# ---------------------------------------------------------------------------


async def _efts_query(
    client: httpx.AsyncClient,
    query: str,
    forms: list[str],
    start_date: date,
    end_date: date,
    warnings: list[str],
) -> list[dict]:
    url = (
        f"{_EFTS_BASE}"
        f"?q={quote(chr(34) + query + chr(34))}"
        f"&forms={quote(','.join(forms))}"
        f"&dateRange=custom"
        f"&startdt={start_date.isoformat()}"
        f"&enddt={end_date.isoformat()}"
    )
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("hits", {}).get("hits", [])
    except httpx.HTTPStatusError as exc:
        warnings.append(f"EFTS HTTP {exc.response.status_code} for query '{query}' forms={forms}")
        logger.warning("ma_screener: EFTS %d query='%s'", exc.response.status_code, query)
    except Exception as exc:
        warnings.append(f"EFTS request failed for query '{query}': {exc}")
        logger.warning("ma_screener: EFTS error query='%s': %s", query, exc)
    return []


def _parse_hit_to_deal(hit: dict, hint: str) -> Optional[MADeal]:
    src = hit.get("_source", {})
    accession_no = src.get("accession_no") or hit.get("_id", "")
    if not accession_no:
        return None

    company_name = src.get("entity_name", "")
    if not company_name:
        display_names = src.get("display_names", [])
        if display_names and isinstance(display_names, list):
            dn = display_names[0]
            company_name = dn.get("entity", "") if isinstance(dn, dict) else str(dn)

    cik = str(src.get("cik", "")).zfill(10)
    form_type = src.get("form_type", "8-K")

    try:
        filing_date = date.fromisoformat(src.get("file_date", "")[:10]).isoformat()
    except (ValueError, TypeError):
        return None

    description = (src.get("description", "") + " " + src.get("biz_descs", "")).strip()
    if not description:
        description = f"{form_type} filing by {company_name}"

    deal_type = _classify_deal_type(form_type, description + " " + hint)
    if deal_type == "unknown":
        deal_type = _classify_deal_type(form_type, hint)

    return MADeal(
        acquirer=None,
        target=company_name or None,
        ticker=cik or None,
        deal_type=deal_type,
        deal_value_billions=_extract_deal_value(description),
        filing_date=filing_date,
        status=_detect_status(description),
        description=description[:300].strip(),
        form_type=form_type,
        accession_number=accession_no,
        edgar_url=_accession_to_url(accession_no, cik),
    )


# ---------------------------------------------------------------------------
# CIK resolution + submissions fetch
# ---------------------------------------------------------------------------


async def _load_cik_map(client: httpx.AsyncClient) -> None:
    global _cik_cache_loaded
    if _cik_cache_loaded:
        return
    try:
        resp = await client.get(_COMPANY_TICKERS_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        for entry in resp.json().values():
            ticker = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", entry.get("cik", ""))).zfill(10)
            if ticker and cik:
                _cik_cache[ticker] = cik
        _cik_cache_loaded = True
        logger.debug("ma_screener: loaded %d ticker→CIK entries", len(_cik_cache))
    except Exception as exc:
        logger.warning("ma_screener: CIK map load failed: %s", exc)


async def _resolve_cik(client: httpx.AsyncClient, ticker: str) -> Optional[str]:
    await _load_cik_map(client)
    return _cik_cache.get(ticker.upper())


async def _fetch_submissions(
    client: httpx.AsyncClient, cik_padded: str, warnings: list[str]
) -> dict:
    url = f"{_SUBMISSIONS_BASE}/CIK{cik_padded}.json"
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        warnings.append(f"Submissions HTTP {exc.response.status_code} for CIK {cik_padded}")
    except Exception as exc:
        warnings.append(f"Submissions fetch failed for CIK {cik_padded}: {exc}")
    return {}


def _recent_filings_by_form(
    submissions: dict, form_types: set[str], cutoff_date: date
) -> list[dict]:
    block = submissions.get("filings", {}).get("recent", {})
    if not block:
        return []
    forms = block.get("form", [])
    dates = block.get("filingDate", [])
    accessions = block.get("accessionNumber", [])
    cik = str(submissions.get("cik", "")).zfill(10)
    name = submissions.get("name", "")
    results = []
    for i, form in enumerate(forms):
        if form.upper() not in {f.upper() for f in form_types}:
            continue
        try:
            filed = date.fromisoformat((dates[i] if i < len(dates) else "")[:10])
        except (ValueError, TypeError):
            continue
        if filed < cutoff_date:
            continue
        results.append({
            "form": form,
            "filing_date": dates[i][:10] if i < len(dates) else "",
            "accession_no": accessions[i] if i < len(accessions) else "",
            "cik": cik,
            "company_name": name,
        })
    return results


async def _find_defense_mechanisms(
    client: httpx.AsyncClient, company_name: str
) -> list[str]:
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=730)

    async def _check(mechanism: str) -> Optional[str]:
        url = (
            f"{_EFTS_BASE}"
            f"?q={quote(chr(34) + mechanism + chr(34) + ' ' + company_name)}"
            f"&forms=DEF14A"
            f"&dateRange=custom"
            f"&startdt={start_dt.isoformat()}&enddt={end_dt.isoformat()}"
        )
        try:
            resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            return mechanism if hits else None
        except Exception as exc:
            logger.debug("ma_screener: defense mech '%s' error: %s", mechanism, exc)
        return None

    results = await asyncio.gather(*[_check(m) for m in _DEFENSE_MECHANISMS])
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def screen_ma_deals(
    days_back: int = 30,
    min_value_billions: Optional[float] = None,
    sector: Optional[str] = None,
    deal_type: Optional[str] = None,
    limit: int = 25,
) -> MAScreenResult:
    """
    Screen EDGAR filings for recent M&A activity.

    Args:
        days_back:           Lookback window in calendar days (default 30).
        min_value_billions:  Minimum deal value filter (billions).
        sector:              GICS sector keyword filter (e.g. "technology").
        deal_type:           "merger" | "acquisition" | "spinoff" | "divestiture" | None.
        limit:               Maximum deals returned (default 25).

    Returns:
        MAScreenResult with deduplicated, filtered deals sorted newest-first.
    """
    warnings_: list[str] = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    queries_to_run = _MA_QUERIES
    if deal_type is not None:
        filtered = [(q, f, h) for q, f, h in _MA_QUERIES if h == deal_type.lower()]
        if filtered:
            queries_to_run = filtered
        else:
            warnings_.append(f"Unknown deal_type '{deal_type}'. Returning all types.")

    async with httpx.AsyncClient() as client:
        raw_results: list[list[dict]] = await asyncio.gather(*[
            _efts_query(client, q, forms, start_date, end_date, warnings_)
            for q, forms, _ in queries_to_run
        ])

    seen: set[str] = set()
    deals: list[MADeal] = []

    for (_query, _forms, hint), hits in zip(queries_to_run, raw_results):
        for hit in hits:
            src = hit.get("_source", {})
            acc = src.get("accession_no") or hit.get("_id", "")
            if not acc or acc in seen:
                continue
            seen.add(acc)

            deal = _parse_hit_to_deal(hit, hint)
            if deal is None:
                continue
            if deal_type is not None and deal.deal_type != deal_type.lower():
                continue
            if min_value_billions is not None and (
                deal.deal_value_billions is None
                or deal.deal_value_billions < min_value_billions
            ):
                continue
            if sector is not None and not _matches_sector(
                deal.target or "", deal.description, sector
            ):
                continue
            deals.append(deal)

    deals.sort(key=lambda d: d.filing_date, reverse=True)
    total_found = len(deals)
    deals = deals[:limit]

    logger.info(
        "ma_screener: screen_ma_deals total=%d returned=%d days_back=%d",
        total_found, len(deals), days_back,
    )
    return MAScreenResult(
        deals=deals,
        total_found=total_found,
        days_back=days_back,
        filters_applied={
            "days_back": days_back,
            "min_value_billions": min_value_billions,
            "sector": sector,
            "deal_type": deal_type,
            "limit": limit,
        },
        as_of=date.today().isoformat(),
        warnings=warnings_,
    )


async def get_ma_profile(ticker: str) -> MATargetProfile:
    """
    Build an M&A target profile for a company by ticker symbol.

    Resolves the EDGAR CIK, inspects recent filings for deal activity, and
    signals acquisition-target status, activist interest, and takeover defenses.

    Args:
        ticker: Exchange ticker (e.g. "AAPL").

    Returns:
        MATargetProfile with deal history and takeover-readiness signals.
    """
    warnings_: list[str] = []
    ticker_upper = ticker.upper()
    today = date.today()

    async with httpx.AsyncClient() as client:
        cik = await _resolve_cik(client, ticker_upper)
        if cik is None:
            warnings_.append(
                f"Could not resolve CIK for '{ticker_upper}'. Profile incomplete."
            )
            return MATargetProfile(
                ticker=ticker_upper,
                as_of=today.isoformat(),
                warnings=warnings_,
            )
        submissions = await _fetch_submissions(client, cik, warnings_)

    company_name: Optional[str] = submissions.get("name")
    cutoff_90d = today - timedelta(days=90)
    cutoff_365d = today - timedelta(days=365)

    deal_forms = {"8-K", "8-K/A", "DEFM14A", "SC TO-T", "SC TO-I"}
    activist_forms = {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}
    target_forms = {"DEFM14A", "SC TO-T", "SC TO-I", "SC TO-T/A"}

    recent_deal_filings = _recent_filings_by_form(submissions, deal_forms, cutoff_365d)
    activist_filings = _recent_filings_by_form(submissions, activist_forms, cutoff_90d)
    target_signal_filings = _recent_filings_by_form(submissions, target_forms, cutoff_90d)

    cik_plain = cik.lstrip("0") or cik
    recent_deals: list[MADeal] = []
    for f in recent_deal_filings[:20]:
        form = f.get("form", "8-K")
        acc = f.get("accession_no", "")
        acc_nodash = acc.replace("-", "")
        if form.upper() in ("DEFM14A",):
            dtype, status = "merger", "announced"
        elif form.upper() in ("SC TO-T", "SC TO-I", "SC TO-T/A"):
            dtype, status = "acquisition", "announced"
        else:
            dtype, status = "acquisition", "pending"
        recent_deals.append(MADeal(
            acquirer=None,
            target=company_name,
            ticker=ticker_upper,
            deal_type=dtype,
            deal_value_billions=None,
            filing_date=f.get("filing_date", today.isoformat()),
            status=status,
            description=f"{form} filing by {company_name or ticker_upper}",
            form_type=form,
            accession_number=acc,
            edgar_url=f"{_EDGAR_ARCHIVE}/data/{cik_plain}/{acc_nodash}/",
        ))

    defense_mechs: list[str] = []
    if company_name:
        async with httpx.AsyncClient() as client:
            defense_mechs = await _find_defense_mechanisms(client, company_name)
    else:
        warnings_.append("Company name unavailable; skipping defense mechanism scan.")

    return MATargetProfile(
        ticker=ticker_upper,
        company_name=company_name,
        recent_deals=recent_deals,
        is_acquisition_target=len(target_signal_filings) > 0,
        activist_interest=len(activist_filings) > 0,
        defense_mechanisms=defense_mechs,
        as_of=today.isoformat(),
        warnings=warnings_,
    )
