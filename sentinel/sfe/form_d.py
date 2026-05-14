"""
Form D / Private Market Intelligence — SEC EDGAR.

Parses Reg D exempt-offering filings to surface private-company fundraising:
VC raises, PE deals, hedge fund launches, startup rounds — 100% free public data.

Public API
----------
get_company_form_d(company_name, limit)                                 -> list[FormDFiling]
screen_private_market(query, state, min_amount_mm, fund_type,
                      days_back, limit)                                 -> PrivateMarketScreen
"""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
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

_EFTS_BASE     = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
_HEADERS = {
    "User-Agent": "SENTINEL/1.0 research@sentinel.ai",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 15.0
_NS = "http://www.sec.gov/xmlschema/formd"   # Form D XML namespace

_EXEMPTION_MAP: dict[str, str] = {
    "06b": "506b", "6b": "506b", "06c": "506c", "6c": "506c",
}

_FUND_TYPE_ALIASES: dict[str, str] = {
    "hedge": "Hedge Fund",
    "venture": "Venture Capital Fund",
    "private equity": "Private Equity Fund",
    "real estate": "Real Estate Fund",
    "other": "Other Investment Fund",
}

_EFTS_SOURCE = (
    "period_of_report,display_names,entity_id,"
    "file_date,form_type,biz_location,accession_no,file_num"
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FormDFiling(BaseModel):
    company_name: str
    cik: str
    file_date: str
    period_of_report: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    total_offering_amount: Optional[float] = None   # USD
    amount_sold: Optional[float] = None             # USD
    offering_type: str = "Unknown"                  # "Equity" | "Debt" | "Fund"
    fund_type: Optional[str] = None                 # "Hedge Fund" | "VC" | "PE" | None
    industry: Optional[str] = None
    exemption_type: Optional[str] = None            # "506b" | "506c"
    is_amendment: bool = False
    key_persons: list[str] = Field(default_factory=list)
    filing_url: str


class PrivateMarketScreen(BaseModel):
    query: str
    total_found: int
    filings: list[FormDFiling]
    total_capital_raised: Optional[float] = None    # sum of amount_sold
    by_state: dict[str, int]
    by_offering_type: dict[str, int]
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _safe_float(v: object) -> Optional[float]:
    try: return float(v)  # type: ignore[arg-type]
    except Exception: return None


async def _get_json(client: httpx.AsyncClient, url: str, warnings: list[str], label: str) -> dict:
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code == 429:
            warnings.append(f"{label}: rate-limited (429); partial result.")
            return {}
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except httpx.HTTPStatusError as exc:
        logger.warning("form_d: HTTP error", label=label, status=exc.response.status_code)
        warnings.append(f"{label}: HTTP {exc.response.status_code}")
    except Exception as exc:
        logger.warning("form_d: request failed", label=label, error=str(exc))
        warnings.append(f"{label}: {exc}")
    return {}


async def _get_text(client: httpx.AsyncClient, url: str, label: str) -> str:
    hdrs = {**_HEADERS, "Accept": "application/xml,text/html,*/*"}
    try:
        resp = await client.get(url, headers=hdrs, timeout=_TIMEOUT)
        if resp.status_code in (200,):
            return resp.text
    except Exception as exc:
        logger.debug("form_d: text fetch failed", label=label, error=str(exc))
    return ""


# ---------------------------------------------------------------------------
# EFTS search
# ---------------------------------------------------------------------------


def _efts_url_by_name(company_name: str, limit: int) -> str:
    return (
        f"{_EFTS_BASE}?q={quote(chr(34)+company_name+chr(34))}&forms=D"
        f"&hits.hits._source={_EFTS_SOURCE}&hits.hits.total=true&hits.hits.size={limit}"
    )


def _efts_url_recent(start_dt: str, end_dt: str, limit: int, query: str | None) -> str:
    q_part = f"&q={quote(chr(34)+query+chr(34))}" if query else ""
    return (
        f"{_EFTS_BASE}?forms=D{q_part}"
        f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}"
        f"&hits.hits._source={_EFTS_SOURCE}&hits.hits.total=true&hits.hits.size={limit}"
    )


async def _efts_search(client: httpx.AsyncClient, url: str, warnings: list[str], label: str) -> list[dict]:
    data = await _get_json(client, url, warnings, label)
    return data.get("hits", {}).get("hits", [])


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def _xml_text(root: ET.Element, tag: str) -> Optional[str]:
    el = root.find(f".//{{{_NS}}}{tag}") or root.find(f".//{tag}")
    return el.text.strip() if el is not None and el.text else None


def _parse_form_d_xml(xml_raw: str) -> dict:
    result: dict = {}
    try:
        root = ET.fromstring(xml_raw)
    except ET.ParseError:
        return result

    gt = lambda tag: _xml_text(root, tag)  # noqa: E731

    result["total_offering_amount"] = _safe_float(gt("totalOfferingAmount"))
    result["amount_sold"]           = _safe_float(gt("totalAmountSold") or gt("amountSold"))
    result["is_amendment"]          = (gt("isAmendment") or "").lower() == "true"

    # Offering type
    if gt("isPooledInvestmentFundType") == "true":
        offering_type = "Fund"
    elif gt("isDebtType") == "true":
        offering_type = "Debt"
    elif gt("isEquityType") == "true":
        offering_type = "Equity"
    else:
        offering_type = "Unknown"
    result["offering_type"] = offering_type

    result["fund_type"] = gt("investmentFundType")
    result["industry"]  = gt("industryGroupType")

    exemption_raw = (gt("exemptionsAndExclusions") or gt("federalExemptionsExclusions") or "").lower()
    for raw_key, mapped in _EXEMPTION_MAP.items():
        if raw_key in exemption_raw:
            result["exemption_type"] = mapped
            break

    persons: list[str] = []
    for el in root.iter():
        if "relatedPerson" in el.tag or "RelatedPerson" in el.tag:
            fn_el = el.find(f"{{{_NS}}}firstName") or el.find("firstName")
            ln_el = el.find(f"{{{_NS}}}lastName")  or el.find("lastName")
            fn = (fn_el.text or "").strip() if fn_el is not None else ""
            ln = (ln_el.text or "").strip() if ln_el is not None else ""
            full = f"{fn} {ln}".strip()
            if full and full not in persons:
                persons.append(full)
    result["key_persons"] = persons[:10]

    result["state"] = gt("stateOrCountry") or gt("issuerState")
    result["city"]  = gt("city") or gt("issuerCity")
    return result


# ---------------------------------------------------------------------------
# Source dict → helpers
# ---------------------------------------------------------------------------


def _entity_name(src: dict) -> str:
    display = src.get("display_names") or []
    if display and isinstance(display, list):
        first = display[0]
        return (first.get("entity") or first.get("name") or "") if isinstance(first, dict) else str(first)
    return src.get("entity_name") or src.get("issuerName") or "Unknown"


def _cik(src: dict) -> str:
    return str(src.get("entity_id") or src.get("cik") or "").zfill(10)


def _acc(src: dict) -> str:
    return src.get("accession_no") or src.get("file_num") or ""


def _extract_state_city(src: dict) -> tuple[Optional[str], Optional[str]]:
    biz = src.get("biz_location") or ""
    if isinstance(biz, str) and biz:
        parts = [p.strip() for p in biz.split(",")]
        if len(parts) >= 2:
            return parts[-1][:2].upper() or None, parts[0] or None
        if len(parts) == 1 and len(parts[0]) == 2:
            return parts[0].upper(), None
    elif isinstance(biz, list) and biz:
        first = biz[0]
        if isinstance(first, dict):
            return first.get("state") or first.get("stateOrCountryDescription"), first.get("city")
        return str(first)[:2].upper(), None
    return None, None


def _build_filing(src: dict, xd: dict | None = None) -> FormDFiling:
    cik_val    = _cik(src)
    acc_val    = _acc(src)
    acc_nd     = acc_val.replace("-", "") if acc_val else ""
    cik_plain  = cik_val.lstrip("0") or cik_val
    filing_url = (f"{_EDGAR_ARCHIVE}/{cik_plain}/{acc_nd}/"
                  if acc_nd else f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_val}&type=D")

    state, city = _extract_state_city(src)
    x = xd or {}
    if x.get("state"): state = x["state"]
    if x.get("city"):  city  = x["city"]

    fund_type = x.get("fund_type")
    if fund_type:
        low = fund_type.lower()
        for alias, canonical in _FUND_TYPE_ALIASES.items():
            if alias in low:
                fund_type = canonical
                break

    return FormDFiling(
        company_name=_entity_name(src),
        cik=cik_val,
        file_date=str(src.get("file_date", ""))[:10],
        period_of_report=str(src.get("period_of_report", ""))[:10] or None,
        state=state, city=city,
        total_offering_amount=x.get("total_offering_amount"),
        amount_sold=x.get("amount_sold"),
        offering_type=x.get("offering_type", "Unknown"),
        fund_type=fund_type or None,
        industry=x.get("industry"),
        exemption_type=x.get("exemption_type"),
        is_amendment=bool(x.get("is_amendment", False)),
        key_persons=x.get("key_persons", []),
        filing_url=filing_url,
    )


# ---------------------------------------------------------------------------
# XML enrichment
# ---------------------------------------------------------------------------


async def _enrich(client: httpx.AsyncClient, filing: FormDFiling, src: dict, warnings: list[str]) -> FormDFiling:
    acc_val = _acc(src)
    if not acc_val:
        return filing
    acc_nd    = acc_val.replace("-", "")
    cik_plain = filing.cik.lstrip("0") or filing.cik
    folder    = f"{_EDGAR_ARCHIVE}/{cik_plain}/{acc_nd}/"
    dash_acc  = f"{acc_nd[:10]}-{acc_nd[10:12]}-{acc_nd[12:]}" if len(acc_nd) >= 18 else acc_nd

    xml_raw = ""
    for suffix in ("primary_doc.xml", f"{dash_acc}.xml", "formd.xml"):
        xml_raw = await _get_text(client, f"{folder}{suffix}", f"XML/{acc_nd}")
        if xml_raw and "<" in xml_raw:
            break

    if not xml_raw:
        warnings.append(f"XML unavailable for {filing.company_name} ({acc_nd}); metadata only.")
        return filing

    xd = _parse_form_d_xml(xml_raw)
    if not xd:
        warnings.append(f"XML parse failed for {filing.company_name} ({acc_nd}).")
        return filing

    return _build_filing(src, xd)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _match_state(f: FormDFiling, state: str) -> bool:
    return (f.state or "").upper() == state.upper()


def _match_amount(f: FormDFiling, min_mm: float) -> bool:
    v = f.amount_sold or f.total_offering_amount
    return v is not None and v >= min_mm * 1_000_000


def _match_fund_type(f: FormDFiling, fund_type: str) -> bool:
    return fund_type.lower() in (f.fund_type or "").lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_company_form_d(company_name: str, limit: int = 5) -> list[FormDFiling]:
    """Look up Form D filings for a specific company (VC-backed startup, fund, etc.)."""
    warnings_: list[str] = []
    url = _efts_url_by_name(company_name, min(limit * 3, 20))

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        hits = await _efts_search(client, url, warnings_, f"EFTS-D/{company_name}")
        if not hits:
            logger.info("form_d: no hits", company=company_name)
            return []

        pairs: list[tuple[FormDFiling, dict]] = []
        for hit in hits[:limit * 2]:
            src = hit.get("_source", {})
            if src:
                pairs.append((_build_filing(src), src))

        enriched = list(await asyncio.gather(*[
            _enrich(client, f, src, warnings_) for f, src in pairs
        ]))

    results = sorted(enriched, key=lambda f: f.file_date, reverse=True)[:limit]
    logger.info("form_d: get_company_form_d", company=company_name, returned=len(results))
    return results


async def screen_private_market(
    query: str | None = None,
    state: str | None = None,
    min_amount_mm: float | None = None,
    fund_type: str | None = None,
    days_back: int = 30,
    limit: int = 25,
) -> PrivateMarketScreen:
    """Screen recent Form D filings. Monitor new VC raises, hedge fund launches, PE deals."""
    warnings_: list[str] = []
    today    = date.today()
    start_dt = (today - timedelta(days=days_back)).isoformat()
    end_dt   = today.isoformat()
    fetch_n  = min(limit * 4, 100)
    url      = _efts_url_recent(start_dt, end_dt, fetch_n, query)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        hits = await _efts_search(client, url, warnings_, "EFTS-D-screen")

        candidates: list[tuple[FormDFiling, dict]] = []
        for hit in hits:
            src = hit.get("_source", {})
            if not src:
                continue
            f = _build_filing(src)
            if state and not _match_state(f, state):
                continue
            candidates.append((f, src))

        enrich_n = min(len(candidates), limit * 3)
        enriched: list[FormDFiling] = list(await asyncio.gather(*[
            _enrich(client, f, src, warnings_) for f, src in candidates[:enrich_n]
        ]))

    filtered = [
        f for f in enriched
        if (min_amount_mm is None or _match_amount(f, min_amount_mm))
        and (fund_type is None or _match_fund_type(f, fund_type))
    ]
    filtered.sort(key=lambda f: f.file_date, reverse=True)
    page = filtered[:limit]

    raised_vals = [f.amount_sold for f in page if f.amount_sold is not None]
    total_raised: Optional[float] = sum(raised_vals) if raised_vals else None

    by_state: dict[str, int] = {}
    by_type:  dict[str, int] = {}
    for f in page:
        s = f.state or "Unknown"
        by_state[s] = by_state.get(s, 0) + 1
        t = f.offering_type or "Unknown"
        by_type[t]  = by_type.get(t, 0) + 1

    logger.info(
        "form_d: screen_private_market",
        query=query, state=state, days_back=days_back,
        hits=len(hits), filtered=len(filtered), returned=len(page),
    )
    return PrivateMarketScreen(
        query=query or f"Form D filings last {days_back}d",
        total_found=len(filtered),
        filings=page,
        total_capital_raised=total_raised,
        by_state=by_state,
        by_offering_type=by_type,
        as_of=today.isoformat(),
        warnings=warnings_,
    )
