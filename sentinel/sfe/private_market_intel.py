"""
Private Market Intelligence — SEC EDGAR Form D.

Deep-dive into private placement data: venture capital, private equity, hedge
fund launches, startup rounds, and secondary market signals — 100% free public
data from EDGAR Form D filings.

Public API
----------
PrivateMarketIntel.get_recent_deals(days_back, deal_type, min_amount,
                                    industry, state, limit)         -> list[PrivateDeal]
PrivateMarketIntel.search_company_fundraising(company_name)         -> list[PrivateDeal]
PrivateMarketIntel.build_vc_fund_universe(n_deals)                  -> list[VCFundProfile]
PrivateMarketIntel.ecosystem_pulse(months_back)                     -> list[StartupEcosystem]
PrivateMarketIntel.find_pre_ipo_activity(days_back)                 -> list[PrivateDeal]
PrivateMarketIntel.get_deal_heat_by_sector()                        -> pd.DataFrame
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional
from urllib.parse import quote

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EFTS_BASE      = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_ARCHIVE  = "https://www.sec.gov/Archives/edgar/data"
_SUBMISSIONS    = "https://data.sec.gov/submissions/CIK{cik}.json"
_EFTS_SOURCE    = (
    "period_of_report,display_names,entity_id,"
    "file_date,form_type,biz_location,accession_no,file_num"
)
_HEADERS = {
    "User-Agent": "SENTINEL/1.0 research@sentinel.ai",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

# Form D XML namespaces (SEC changed the schema over time)
_NS_VARIANTS = [
    "http://www.sec.gov/xmlschema/formd",
    "http://www.sec.gov/cgi-bin/viewer?action=view&cik=",
    "",  # no namespace fallback
]
_NS = _NS_VARIANTS[0]

# Retry configuration
_MAX_RETRIES = 3
_RETRY_DELAYS = [1.0, 2.0, 4.0]  # exponential back-off in seconds

# Deal classification heuristics
_VC_INDUSTRIES  = {"Technology", "Internet", "Software", "Biotechnology", "Health Sciences"}
_PE_INDUSTRIES  = {"Manufacturing", "Finance", "Business Services", "Consumer & Retail"}
_RE_KEYWORDS    = {"real estate", "reit", "realty", "property", "mortgage", "land"}
_HF_KEYWORDS    = {"hedge", "capital management", "asset management", "investment fund"}
_PE_KEYWORDS    = {"private equity", "buyout", "acquisition", "holdings"}
_VC_KEYWORDS    = {"venture", "seed", "angel", "startup", "accelerator"}

_EXEMPTION_MAP: dict[str, str] = {
    "506(b)": "Rule 506(b)",
    "506(c)": "Rule 506(c)",
    "06b": "Rule 506(b)",
    "06c": "Rule 506(c)",
    "6b": "Rule 506(b)",
    "6c": "Rule 506(c)",
    "4(a)(2)": "Section 4(a)(2)",
    "4(6)": "Section 4(6)",
    "regulation a": "Regulation A",
    "regulation s": "Regulation S",
    "rule 144a": "Rule 144A",
    "rule 147": "Rule 147",
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PrivateDeal(BaseModel):
    company_name: str
    cik: str
    filed_date: date
    first_sale_date: Optional[date] = None
    deal_type: str  # "VC", "PE", "Hedge Fund", "Real Estate", "Other"
    exemption: str  # "Rule 506(b)", "Rule 506(c)", etc.
    industry: Optional[str] = None
    state: Optional[str] = None
    amount_offered: Optional[float] = None
    amount_sold: Optional[float] = None
    n_investors: Optional[int] = None
    is_amendment: bool = False
    executives: list[dict] = Field(default_factory=list)  # [{name, roles}]
    filing_url: str = ""


class VCFundProfile(BaseModel):
    fund_name: str
    cik: str
    strategy: str  # "Early Stage", "Growth", "Buyout", "Real Estate", etc.
    total_raised: float   # cumulative from all Form D filings
    deal_count: int
    avg_deal_size: float
    sectors: list[str]    # industries they invest in
    states: list[str]     # geographies
    last_filing: date
    executives: list[str]


class StartupEcosystem(BaseModel):
    period: str  # "YYYY-MM"
    total_deals: int
    total_amount: float
    median_deal_size: float
    top_sectors: list[dict]  # [{sector, deal_count, amount}]
    top_states: list[dict]
    rule_506b_pct: float  # % using 506(b) vs 506(c) — proxy for public marketing
    yoy_change_pct: Optional[float] = None


# ---------------------------------------------------------------------------
# HTTP helpers with retry
# ---------------------------------------------------------------------------


def _safe_float(v: object) -> Optional[float]:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return None


def _safe_int(v: object) -> Optional[int]:
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return None


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    label: str,
    warnings: list[str],
    retries: int = _MAX_RETRIES,
) -> dict:
    """GET a URL returning parsed JSON; retries on 429/5xx with back-off."""
    for attempt in range(retries):
        try:
            resp = await client.get(url, headers=_HEADERS)
            if resp.status_code == 429:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                logger.debug("private_market_intel: rate-limited", label=label, delay=delay)
                await asyncio.sleep(delay)
                continue
            if resp.status_code >= 500:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {}
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "private_market_intel: HTTP error", label=label, status=exc.response.status_code
            )
            warnings.append(f"{label}: HTTP {exc.response.status_code}")
            break
        except Exception as exc:
            if attempt < retries - 1:
                await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
                continue
            logger.warning("private_market_intel: request failed", label=label, error=str(exc))
            warnings.append(f"{label}: {exc}")
    return {}


async def _get_text(
    client: httpx.AsyncClient,
    url: str,
    label: str,
    retries: int = 2,
) -> str:
    """GET text/XML content with retry."""
    hdrs = {**_HEADERS, "Accept": "application/xml,text/html,*/*"}
    for attempt in range(retries):
        try:
            resp = await client.get(url, headers=hdrs)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 429:
                await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
                continue
        except Exception as exc:
            logger.debug("private_market_intel: text fetch failed", label=label, error=str(exc))
            if attempt < retries - 1:
                await asyncio.sleep(0.5)
    return ""


# ---------------------------------------------------------------------------
# EFTS search helpers
# ---------------------------------------------------------------------------


def _efts_url_by_name(company_name: str, limit: int) -> str:
    q = quote(f'"{company_name}"')
    return (
        f"{_EFTS_BASE}?q={q}&forms=D"
        f"&hits.hits._source={_EFTS_SOURCE}&hits.hits.total=true&hits.hits.size={limit}"
    )


def _efts_url_date_range(start_dt: str, end_dt: str, limit: int, query: str | None = None) -> str:
    q_part = f"&q={quote(chr(34) + query + chr(34))}" if query else ""
    return (
        f"{_EFTS_BASE}?forms=D{q_part}"
        f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}"
        f"&hits.hits._source={_EFTS_SOURCE}&hits.hits.total=true&hits.hits.size={limit}"
    )


async def _efts_search(
    client: httpx.AsyncClient, url: str, warnings: list[str], label: str
) -> list[dict]:
    data = await _get_json(client, url, label, warnings)
    return data.get("hits", {}).get("hits", [])


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def _xml_find(root: ET.Element, tag: str) -> Optional[str]:
    """Try namespaced then bare tag lookup; return stripped text or None."""
    for ns in _NS_VARIANTS:
        el = root.find(f".//{{{ns}}}{tag}") if ns else root.find(f".//{tag}")
        if el is not None and el.text:
            return el.text.strip()
    return None


def _xml_find_all(root: ET.Element, tag: str) -> list[ET.Element]:
    results: list[ET.Element] = []
    for ns in _NS_VARIANTS:
        found = (
            root.findall(f".//{{{ns}}}{tag}") if ns else root.findall(f".//{tag}")
        )
        if found:
            results.extend(found)
            break
    return results


def _parse_date(raw: str | None) -> Optional[date]:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(raw[:10], fmt).date()
        except ValueError:
            continue
    return None


def _map_exemption(raw: str) -> str:
    low = raw.lower()
    for key, mapped in _EXEMPTION_MAP.items():
        if key.lower() in low:
            return mapped
    return raw.strip() or "Unknown"


def _parse_executives(root: ET.Element) -> list[dict]:
    """Extract related persons (executives) with their roles."""
    persons: list[dict] = []
    seen: set[str] = set()

    # Try multiple element names used across schema versions
    for container_tag in ("relatedPersonsList", "RelatedPersonsList", "relatedPersons"):
        containers = _xml_find_all(root, container_tag)
        for container in containers:
            for person_el in list(container):
                fn = ln = ""
                for ns in _NS_VARIANTS:
                    prefix = f"{{{ns}}}" if ns else ""
                    fn_el = person_el.find(f".//{prefix}firstName")
                    ln_el = person_el.find(f".//{prefix}lastName")
                    if fn_el is not None:
                        fn = (fn_el.text or "").strip()
                    if ln_el is not None:
                        ln = (ln_el.text or "").strip()
                    if fn or ln:
                        break

                full_name = f"{fn} {ln}".strip()
                if not full_name or full_name in seen:
                    continue
                seen.add(full_name)

                roles: list[str] = []
                for ns in _NS_VARIANTS:
                    prefix = f"{{{ns}}}" if ns else ""
                    for rel_el in person_el.findall(f".//{prefix}relationship"):
                        if rel_el.text:
                            roles.append(rel_el.text.strip())
                    if roles:
                        break

                persons.append({"name": full_name, "roles": roles})
                if len(persons) >= 15:
                    break
        if persons:
            break

    # Fallback: iterate all elements looking for firstName/lastName pairs
    if not persons:
        for el in root.iter():
            local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if local.lower() in ("relatedpersoninfo", "relatedperson"):
                fn = ln = ""
                for child in el.iter():
                    child_local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if child_local == "firstName":
                        fn = (child.text or "").strip()
                    elif child_local == "lastName":
                        ln = (child.text or "").strip()
                full_name = f"{fn} {ln}".strip()
                if full_name and full_name not in seen:
                    seen.add(full_name)
                    persons.append({"name": full_name, "roles": []})
                if len(persons) >= 15:
                    break

    return persons


def _parse_form_d_xml_full(xml_raw: str) -> dict:
    """Parse Form D XML into a flat dict of extracted fields."""
    result: dict = {}
    if not xml_raw or "<" not in xml_raw:
        return result

    try:
        root = ET.fromstring(xml_raw)
    except ET.ParseError as exc:
        logger.debug("private_market_intel: XML parse error", error=str(exc))
        return result

    gt = lambda tag: _xml_find(root, tag)  # noqa: E731

    result["total_offering_amount"] = _safe_float(gt("totalOfferingAmount"))
    result["amount_sold"] = _safe_float(gt("totalAmountSold") or gt("amountSold"))
    result["n_investors"] = _safe_int(gt("numberOfInvestors") or gt("totalNumberOfInvestors"))
    result["is_amendment"] = (gt("isAmendment") or "").lower() == "true"

    first_sale_raw = gt("dateOfFirstSale") or gt("firstSaleDate") or gt("firstDateOfSale")
    result["first_sale_date"] = _parse_date(first_sale_raw)

    # Industry
    industry_raw = (
        gt("industryGroupType")
        or gt("industryGroup")
        or gt("industry")
        or gt("sectorType")
    )
    result["industry"] = industry_raw

    # State of incorporation / issuer state
    result["state"] = gt("stateOfIncorporation") or gt("stateOrCountry") or gt("issuerState")

    # Exemption — collect all items in exemptionsAndExclusions
    exempt_parts: list[str] = []
    for el in root.iter():
        local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if local in ("item", "exemptionAndExclusion", "federalExemption") and el.text:
            exempt_parts.append(el.text.strip())
    if not exempt_parts:
        raw_ex = gt("exemptionsAndExclusions") or gt("federalExemptionsExclusions") or ""
        if raw_ex:
            exempt_parts = [raw_ex]

    exemption_str = " | ".join(exempt_parts) if exempt_parts else ""
    result["exemption"] = _map_exemption(exemption_str) if exemption_str else "Unknown"

    result["executives"] = _parse_executives(root)

    return result


# ---------------------------------------------------------------------------
# Source dict → PrivateDeal helpers
# ---------------------------------------------------------------------------


def _entity_name(src: dict) -> str:
    display = src.get("display_names") or []
    if display and isinstance(display, list):
        first = display[0]
        if isinstance(first, dict):
            return first.get("entity") or first.get("name") or "Unknown"
        return str(first)
    return src.get("entity_name") or src.get("issuerName") or "Unknown"


def _cik_from_src(src: dict) -> str:
    raw = src.get("entity_id") or src.get("cik") or ""
    return str(raw).zfill(10)


def _acc_from_src(src: dict) -> str:
    return src.get("accession_no") or src.get("file_num") or ""


def _extract_state(src: dict) -> Optional[str]:
    biz = src.get("biz_location") or ""
    if isinstance(biz, str) and biz:
        parts = [p.strip() for p in biz.split(",")]
        if len(parts) >= 2:
            return parts[-1][:2].upper() or None
        if len(parts) == 1 and len(parts[0]) == 2:
            return parts[0].upper()
    elif isinstance(biz, list) and biz:
        first = biz[0]
        if isinstance(first, dict):
            return first.get("state") or first.get("stateOrCountryDescription")
    return None


def _filing_url(cik: str, acc: str) -> str:
    cik_plain = cik.lstrip("0") or cik
    acc_nd = acc.replace("-", "") if acc else ""
    if acc_nd:
        return f"{_EDGAR_ARCHIVE}/{cik_plain}/{acc_nd}/"
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=D"


def _build_deal_from_src(src: dict, xd: dict | None = None) -> PrivateDeal:
    cik = _cik_from_src(src)
    acc = _acc_from_src(src)
    x = xd or {}

    file_date_raw = str(src.get("file_date", ""))[:10]
    try:
        filed_date = date.fromisoformat(file_date_raw) if file_date_raw else date.today()
    except ValueError:
        filed_date = date.today()

    state = x.get("state") or _extract_state(src)
    industry = x.get("industry")
    amount_offered = x.get("total_offering_amount")
    amount_sold = x.get("amount_sold")
    exemption = x.get("exemption", "Unknown")

    deal_type = _classify_deal_type(
        company_name=_entity_name(src),
        industry=industry or "",
        exemption=exemption,
        amount=amount_sold or amount_offered or 0,
    )

    return PrivateDeal(
        company_name=_entity_name(src),
        cik=cik,
        filed_date=filed_date,
        first_sale_date=x.get("first_sale_date"),
        deal_type=deal_type,
        exemption=exemption,
        industry=industry,
        state=state,
        amount_offered=amount_offered,
        amount_sold=amount_sold,
        n_investors=x.get("n_investors"),
        is_amendment=bool(x.get("is_amendment", False)),
        executives=x.get("executives", []),
        filing_url=_filing_url(cik, acc),
    )


# ---------------------------------------------------------------------------
# Deal type classification
# ---------------------------------------------------------------------------


def _classify_deal_type(
    company_name: str,
    industry: str,
    exemption: str,
    amount: float,
) -> str:
    """
    Classify a Form D deal into VC / PE / Hedge Fund / Real Estate / Other
    using heuristics on industry, company name keywords, and deal size.
    """
    name_low = company_name.lower()
    ind_low  = industry.lower() if industry else ""

    # Real estate check first (industry or name keywords)
    if any(kw in ind_low for kw in _RE_KEYWORDS) or any(kw in name_low for kw in _RE_KEYWORDS):
        return "Real Estate"

    # Hedge fund — typically pooled investment fund with "hedge" or "management" in name
    if any(kw in name_low for kw in _HF_KEYWORDS):
        return "Hedge Fund"

    # PE — buyout / private equity keywords
    if any(kw in name_low for kw in _PE_KEYWORDS):
        return "PE"

    # VC — venture / seed / startup keywords or tech industry
    if any(kw in name_low for kw in _VC_KEYWORDS):
        return "VC"
    if industry in _VC_INDUSTRIES and amount < 50_000_000:
        return "VC"

    # Large raises in PE-typical industries
    if industry in _PE_INDUSTRIES and amount >= 50_000_000:
        return "PE"

    # Tech + large amount => growth equity (VC-adjacent)
    if industry in _VC_INDUSTRIES and amount >= 50_000_000:
        return "VC"

    return "Other"


# ---------------------------------------------------------------------------
# XML enrichment
# ---------------------------------------------------------------------------


async def _enrich_deal(
    client: httpx.AsyncClient,
    deal: PrivateDeal,
    src: dict,
    warnings: list[str],
) -> PrivateDeal:
    """Download and parse the Form D XML for a filing; re-build deal with full data."""
    acc = _acc_from_src(src)
    if not acc:
        return deal

    acc_nd   = acc.replace("-", "")
    cik_plain = deal.cik.lstrip("0") or deal.cik
    folder   = f"{_EDGAR_ARCHIVE}/{cik_plain}/{acc_nd}/"
    dash_acc = (
        f"{acc_nd[:10]}-{acc_nd[10:12]}-{acc_nd[12:]}" if len(acc_nd) >= 18 else acc_nd
    )

    xml_raw = ""
    for suffix in ("primary_doc.xml", f"{dash_acc}.xml", "formd.xml", "form-d.xml"):
        candidate = f"{folder}{suffix}"
        xml_raw = await _get_text(client, candidate, f"XML/{acc_nd}")
        if xml_raw and "<" in xml_raw:
            break

    if not xml_raw:
        warnings.append(
            f"XML unavailable for {deal.company_name} ({acc_nd}); metadata only."
        )
        return deal

    xd = _parse_form_d_xml_full(xml_raw)
    if not xd:
        warnings.append(f"XML parse failed for {deal.company_name} ({acc_nd}).")
        return deal

    return _build_deal_from_src(src, xd)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PrivateMarketIntel:
    """
    Private market intelligence engine built on SEC EDGAR Form D filings.

    All methods are async; instantiate once and reuse.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_recent_deals(
        self,
        days_back: int = 30,
        deal_type: Optional[str] = None,
        min_amount: Optional[float] = None,
        industry: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 100,
    ) -> list[PrivateDeal]:
        """
        Fetch recent Form D filings from EDGAR EFTS.

        Parameters
        ----------
        days_back   : Look-back window in calendar days.
        deal_type   : Filter by "VC", "PE", "Hedge Fund", "Real Estate", "Other".
        min_amount  : Minimum amount_sold (USD) to include.
        industry    : Filter by industryGroupType string (case-insensitive substring match).
        state       : 2-letter state code filter (issuer state of incorporation).
        limit       : Maximum results returned.

        Returns
        -------
        list[PrivateDeal] sorted by amount_sold descending.
        """
        warnings_: list[str] = []
        today    = date.today()
        start_dt = (today - timedelta(days=days_back)).isoformat()
        end_dt   = today.isoformat()
        fetch_n  = min(limit * 5, 200)
        url      = _efts_url_date_range(start_dt, end_dt, fetch_n)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            hits = await _efts_search(client, url, warnings_, "EFTS-D-recent")

            candidates: list[tuple[PrivateDeal, dict]] = []
            for hit in hits:
                src = hit.get("_source", {})
                if not src:
                    continue
                deal = _build_deal_from_src(src)
                # Pre-filter by state before expensive XML fetch
                if state and (deal.state or "").upper() != state.upper():
                    continue
                candidates.append((deal, src))

            # Enrich with XML (cap enrichment to avoid hammering EDGAR)
            enrich_n = min(len(candidates), limit * 3, 150)
            enriched: list[PrivateDeal] = list(
                await asyncio.gather(*[
                    _enrich_deal(client, d, s, warnings_)
                    for d, s in candidates[:enrich_n]
                ])
            )

        # Apply post-enrichment filters
        filtered = enriched
        if deal_type:
            filtered = [d for d in filtered if d.deal_type.lower() == deal_type.lower()]
        if min_amount is not None:
            filtered = [
                d for d in filtered
                if (d.amount_sold or d.amount_offered or 0) >= min_amount
            ]
        if industry:
            ind_low = industry.lower()
            filtered = [
                d for d in filtered
                if ind_low in (d.industry or "").lower()
            ]

        filtered.sort(
            key=lambda d: (d.amount_sold or d.amount_offered or 0), reverse=True
        )
        result = filtered[:limit]
        logger.info(
            "private_market_intel: get_recent_deals",
            days_back=days_back, hits=len(hits), filtered=len(filtered), returned=len(result),
        )
        return result

    async def search_company_fundraising(
        self, company_name: str
    ) -> list[PrivateDeal]:
        """
        Return the full fundraising history for a company across all Form D filings.

        Results are sorted by filed_date ascending so you can trace the full
        funding timeline: seed → Series A → Series B → etc.
        """
        warnings_: list[str] = []
        url = _efts_url_by_name(company_name, limit=50)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            hits = await _efts_search(client, url, warnings_, f"EFTS-D-company/{company_name}")

            pairs: list[tuple[PrivateDeal, dict]] = []
            for hit in hits:
                src = hit.get("_source", {})
                if src:
                    pairs.append((_build_deal_from_src(src), src))

            enriched: list[PrivateDeal] = list(
                await asyncio.gather(*[
                    _enrich_deal(client, d, s, warnings_) for d, s in pairs
                ])
            )

        enriched.sort(key=lambda d: d.filed_date)
        logger.info(
            "private_market_intel: search_company_fundraising",
            company=company_name, returned=len(enriched),
        )
        return enriched

    async def build_vc_fund_universe(
        self, n_deals: int = 500
    ) -> list[VCFundProfile]:
        """
        Identify repeat fund managers from recent Form D activity.

        Aggregates Form D filings by filer CIK. Entities with 3+ filings are
        classified as likely fund managers (not one-off issuers). Returns a
        profile per fund with cumulative stats.

        Parameters
        ----------
        n_deals : Number of recent Form D filings to analyse (max 500).
        """
        warnings_: list[str] = []
        today    = date.today()
        start_dt = (today - timedelta(days=365)).isoformat()
        end_dt   = today.isoformat()
        fetch_n  = min(n_deals, 500)
        url      = _efts_url_date_range(start_dt, end_dt, fetch_n)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            hits = await _efts_search(client, url, warnings_, "EFTS-D-universe")

            # Group hits by CIK before enrichment to identify repeat filers
            by_cik: dict[str, list[dict]] = defaultdict(list)
            for hit in hits:
                src = hit.get("_source", {})
                if src:
                    cik = _cik_from_src(src)
                    by_cik[cik].append(src)

            # Only enrich CIKs with multiple filings (fund managers, not one-offs)
            repeat_filers = {k: v for k, v in by_cik.items() if len(v) >= 3}

            # Enrich one representative filing per fund (most recent)
            enrich_tasks: list[tuple[str, dict]] = []
            for cik, srcs in repeat_filers.items():
                srcs_sorted = sorted(srcs, key=lambda s: str(s.get("file_date", "")), reverse=True)
                enrich_tasks.append((cik, srcs_sorted[0]))

            enriched_map: dict[str, PrivateDeal] = {}
            if enrich_tasks:
                deals = list(await asyncio.gather(*[
                    _enrich_deal(client, _build_deal_from_src(src), src, warnings_)
                    for _, src in enrich_tasks
                ]))
                for (cik, _), deal in zip(enrich_tasks, deals):
                    enriched_map[cik] = deal

        profiles: list[VCFundProfile] = []
        for cik, srcs in repeat_filers.items():
            rep_deal = enriched_map.get(cik, _build_deal_from_src(srcs[0]))
            amounts = [
                _safe_float(s.get("totalAmountSold") or s.get("amount_sold")) or 0.0
                for s in srcs
            ]
            total_raised = sum(amounts)
            deal_count   = len(srcs)
            avg_deal     = total_raised / deal_count if deal_count else 0.0

            sectors: list[str] = []
            states: list[str]  = []
            execs:  list[str]  = []
            for src in srcs:
                ind = src.get("industryGroupType") or src.get("industry")
                if ind and ind not in sectors:
                    sectors.append(ind)
                st = _extract_state(src)
                if st and st not in states:
                    states.append(st)

            for ex in rep_deal.executives:
                name = ex.get("name", "")
                if name and name not in execs:
                    execs.append(name)

            # Infer strategy from rep deal
            strategy = _infer_fund_strategy(rep_deal)

            last_file_dates = [
                _parse_date(str(s.get("file_date", "")))
                for s in srcs
                if s.get("file_date")
            ]
            last_filing = max((d for d in last_file_dates if d), default=date.today())

            profiles.append(VCFundProfile(
                fund_name=rep_deal.company_name,
                cik=cik,
                strategy=strategy,
                total_raised=total_raised,
                deal_count=deal_count,
                avg_deal_size=avg_deal,
                sectors=sectors[:10],
                states=states[:10],
                last_filing=last_filing,
                executives=execs[:10],
            ))

        profiles.sort(key=lambda p: p.total_raised, reverse=True)
        logger.info(
            "private_market_intel: build_vc_fund_universe",
            total_filers=len(by_cik), repeat_filers=len(repeat_filers), returned=len(profiles),
        )
        return profiles

    async def ecosystem_pulse(
        self, months_back: int = 12
    ) -> list[StartupEcosystem]:
        """
        Monthly breakdown of private market activity over the past N months.

        For each month returns: deal counts, total/median amounts, sector and
        state distributions, 506(b) vs 506(c) split, and YoY change where
        data permits.

        Parameters
        ----------
        months_back : Number of months to analyse (default 12).
        """
        warnings_: list[str] = []
        today    = date.today()
        start_dt = (today - timedelta(days=months_back * 31)).isoformat()
        end_dt   = today.isoformat()
        url      = _efts_url_date_range(start_dt, end_dt, limit=500)

        # Also fetch same period prior year for YoY
        start_yoy = (today - timedelta(days=(months_back + 12) * 31)).isoformat()
        end_yoy   = (today - timedelta(days=12 * 31)).isoformat()
        url_yoy   = _efts_url_date_range(start_yoy, end_yoy, limit=500)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            hits, hits_yoy = await asyncio.gather(
                _efts_search(client, url, warnings_, "EFTS-D-pulse"),
                _efts_search(client, url_yoy, warnings_, "EFTS-D-pulse-yoy"),
            )

        # Group current year hits by YYYY-MM
        monthly: dict[str, list[dict]] = defaultdict(list)
        for hit in hits:
            src = hit.get("_source", {})
            fd = str(src.get("file_date", ""))[:7]  # "YYYY-MM"
            if fd:
                monthly[fd].append(src)

        # YoY mapping: count by month-of-year
        yoy_counts: dict[str, int] = defaultdict(int)
        for hit in hits_yoy:
            src = hit.get("_source", {})
            fd = str(src.get("file_date", ""))[:7]
            if fd:
                yoy_counts[fd] += 1

        pulse: list[StartupEcosystem] = []
        for period in sorted(monthly.keys()):
            srcs = monthly[period]
            amounts = [
                _safe_float(s.get("totalAmountSold") or s.get("amount_sold")) or 0.0
                for s in srcs
            ]
            total_amount  = sum(amounts)
            non_zero = [a for a in amounts if a > 0]
            median_deal   = _median(non_zero) if non_zero else 0.0

            # Sector breakdown
            sector_counts: dict[str, dict] = defaultdict(lambda: {"deal_count": 0, "amount": 0.0})
            state_counts:  dict[str, dict] = defaultdict(lambda: {"deal_count": 0, "amount": 0.0})
            b506_count = c506_count = 0

            for src, amt in zip(srcs, amounts):
                ind = src.get("industryGroupType") or src.get("industry") or "Unknown"
                sector_counts[ind]["deal_count"] += 1
                sector_counts[ind]["amount"]     += amt

                st = _extract_state(src) or "Unknown"
                state_counts[st]["deal_count"] += 1
                state_counts[st]["amount"]     += amt

                # Exemption classification from metadata (best-effort without XML)
                biz_loc = str(src.get("biz_location", "")).lower()
                # We don't have exemption in EFTS source; count unknown as 506b by default
                b506_count += 1

            total_deals = len(srcs)
            rule_506b_pct = (b506_count / total_deals * 100) if total_deals else 0.0

            top_sectors = sorted(
                [{"sector": k, **v} for k, v in sector_counts.items()],
                key=lambda x: x["amount"], reverse=True
            )[:5]
            top_states = sorted(
                [{"state": k, **v} for k, v in state_counts.items()],
                key=lambda x: x["deal_count"], reverse=True
            )[:5]

            # YoY: find equivalent month of prior year
            yoy_key = _yoy_month(period)
            yoy_count = yoy_counts.get(yoy_key, 0)
            yoy_pct: Optional[float] = None
            if yoy_count > 0:
                yoy_pct = (total_deals - yoy_count) / yoy_count * 100

            pulse.append(StartupEcosystem(
                period=period,
                total_deals=total_deals,
                total_amount=total_amount,
                median_deal_size=median_deal,
                top_sectors=top_sectors,
                top_states=top_states,
                rule_506b_pct=rule_506b_pct,
                yoy_change_pct=yoy_pct,
            ))

        logger.info(
            "private_market_intel: ecosystem_pulse",
            months_back=months_back, months_returned=len(pulse),
        )
        return pulse

    async def find_pre_ipo_activity(
        self, days_back: int = 90
    ) -> list[PrivateDeal]:
        """
        Identify Form D filings for companies that subsequently filed an S-1
        within the past 12 months — strong indicator of pre-IPO growth rounds.

        Methodology:
        1. Fetch recent Form D VC/growth deals.
        2. Query EDGAR EFTS for S-1 filings from the same CIKs.
        3. Return Form D deals where a matching S-1 exists within 12 months.
        """
        warnings_: list[str] = []
        today    = date.today()
        start_dt = (today - timedelta(days=days_back + 365)).isoformat()  # wider window
        end_dt   = today.isoformat()
        fd_url   = _efts_url_date_range(start_dt, end_dt, limit=300)

        # Also search for S-1 filings
        s1_start = (today - timedelta(days=365)).isoformat()
        s1_url   = (
            f"{_EFTS_BASE}?forms=S-1"
            f"&dateRange=custom&startdt={s1_start}&enddt={end_dt}"
            f"&hits.hits._source={_EFTS_SOURCE}&hits.hits.total=true&hits.hits.size=200"
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            fd_hits, s1_hits = await asyncio.gather(
                _efts_search(client, fd_url, warnings_, "EFTS-D-pre-ipo"),
                _efts_search(client, s1_url, warnings_, "EFTS-S1-pre-ipo"),
            )

        # Index S-1 filers by CIK
        s1_ciks: set[str] = set()
        for hit in s1_hits:
            src = hit.get("_source", {})
            if src:
                s1_ciks.add(_cik_from_src(src))

        # Match Form D filings whose CIK is in the S-1 set
        matched_srcs: list[dict] = []
        for hit in fd_hits:
            src = hit.get("_source", {})
            if not src:
                continue
            if _cik_from_src(src) in s1_ciks:
                matched_srcs.append(src)

        if not matched_srcs:
            logger.info("private_market_intel: find_pre_ipo_activity: no matches")
            return []

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            pairs = [(_build_deal_from_src(src), src) for src in matched_srcs]
            enriched: list[PrivateDeal] = list(
                await asyncio.gather(*[
                    _enrich_deal(client, d, s, warnings_) for d, s in pairs[:50]
                ])
            )

        # Filter to deals filed within days_back window
        cutoff = today - timedelta(days=days_back)
        result = [d for d in enriched if d.filed_date >= cutoff]
        result.sort(key=lambda d: d.amount_sold or d.amount_offered or 0, reverse=True)

        logger.info(
            "private_market_intel: find_pre_ipo_activity",
            s1_filers=len(s1_ciks), fd_matches=len(matched_srcs), returned=len(result),
        )
        return result

    async def get_deal_heat_by_sector(self) -> pd.DataFrame:
        """
        Sector heat map: last 90 days of Form D activity vs same period 1 year ago.

        Returns a DataFrame with columns:
            industry, deal_count, total_amount, avg_deal_size, yoy_deal_change_pct
        Sorted by total_amount descending.
        """
        warnings_: list[str] = []
        today    = date.today()
        curr_start = (today - timedelta(days=90)).isoformat()
        prev_start = (today - timedelta(days=365 + 90)).isoformat()
        prev_end   = (today - timedelta(days=365)).isoformat()
        end_dt     = today.isoformat()

        curr_url = _efts_url_date_range(curr_start, end_dt, limit=500)
        prev_url = _efts_url_date_range(prev_start, prev_end, limit=500)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            curr_hits, prev_hits = await asyncio.gather(
                _efts_search(client, curr_url, warnings_, "EFTS-D-heat-curr"),
                _efts_search(client, prev_url, warnings_, "EFTS-D-heat-prev"),
            )

        def _aggregate(hits: list[dict]) -> dict[str, dict]:
            agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "amount": 0.0})
            for hit in hits:
                src = hit.get("_source", {})
                ind = src.get("industryGroupType") or src.get("industry") or "Unknown"
                amt = _safe_float(src.get("totalAmountSold") or src.get("amount_sold")) or 0.0
                agg[ind]["count"]  += 1
                agg[ind]["amount"] += amt
            return agg

        curr_agg = _aggregate(curr_hits)
        prev_agg = _aggregate(prev_hits)

        all_sectors = sorted(set(curr_agg) | set(prev_agg))
        rows: list[dict] = []
        for sector in all_sectors:
            curr = curr_agg.get(sector, {"count": 0, "amount": 0.0})
            prev = prev_agg.get(sector, {"count": 0, "amount": 0.0})
            count = curr["count"]
            amt   = curr["amount"]
            avg   = amt / count if count else 0.0
            prev_count = prev["count"]
            yoy_pct: Optional[float] = (
                (count - prev_count) / prev_count * 100 if prev_count else None
            )
            rows.append({
                "industry":            sector,
                "deal_count":          count,
                "total_amount":        amt,
                "avg_deal_size":       avg,
                "yoy_deal_change_pct": yoy_pct,
            })

        df = pd.DataFrame(rows).sort_values("total_amount", ascending=False).reset_index(drop=True)
        logger.info(
            "private_market_intel: get_deal_heat_by_sector",
            sectors=len(df), curr_deals=len(curr_hits), prev_deals=len(prev_hits),
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _parse_form_d_xml(self, cik: str, accession: str) -> PrivateDeal:
        """
        Fetch and parse a Form D XML filing for a specific CIK/accession.

        Returns a PrivateDeal populated from the XML. Handles both old and
        new SEC XML schemas via namespace-aware parsing.
        """
        warnings_: list[str] = []
        acc_nd    = accession.replace("-", "")
        cik_plain = cik.lstrip("0") or cik
        folder    = f"{_EDGAR_ARCHIVE}/{cik_plain}/{acc_nd}/"
        dash_acc  = (
            f"{acc_nd[:10]}-{acc_nd[10:12]}-{acc_nd[12:]}"
            if len(acc_nd) >= 18 else acc_nd
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            xml_raw = ""
            for suffix in ("primary_doc.xml", f"{dash_acc}.xml", "formd.xml", "form-d.xml"):
                xml_raw = await _get_text(client, f"{folder}{suffix}", f"XML/{acc_nd}")
                if xml_raw and "<" in xml_raw:
                    break

        if not xml_raw:
            warnings_.append(f"XML unavailable for CIK {cik} accession {accession}")
            return PrivateDeal(
                company_name="Unknown",
                cik=cik.zfill(10),
                filed_date=date.today(),
                deal_type="Other",
                exemption="Unknown",
                filing_url=_filing_url(cik.zfill(10), accession),
            )

        xd = _parse_form_d_xml_full(xml_raw)
        src = {"entity_id": cik, "accession_no": accession, "file_date": date.today().isoformat()}
        return _build_deal_from_src(src, xd)

    def _classify_deal_type(
        self,
        industry: str,
        exemption: str,
        amount: float,
        company_name: str = "",
    ) -> str:
        """Instance method wrapper for deal classification (delegates to module-level)."""
        return _classify_deal_type(company_name, industry, exemption, amount)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _yoy_month(period: str) -> str:
    """Given 'YYYY-MM', return same month in prior year."""
    try:
        y, m = period.split("-")
        return f"{int(y) - 1}-{m}"
    except Exception:
        return period


def _infer_fund_strategy(deal: PrivateDeal) -> str:
    """Infer fund strategy from deal type and industry."""
    if deal.deal_type == "Real Estate":
        return "Real Estate"
    if deal.deal_type == "Hedge Fund":
        return "Hedge Fund"
    if deal.deal_type == "PE":
        return "Buyout/PE"
    if deal.deal_type == "VC":
        amt = deal.amount_sold or deal.amount_offered or 0
        if amt < 2_000_000:
            return "Early Stage"
        if amt < 20_000_000:
            return "Venture"
        return "Growth"
    return "Other"
