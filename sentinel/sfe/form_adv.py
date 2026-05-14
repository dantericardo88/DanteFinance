"""
Form ADV / RIA Adviser Intelligence — SEC IAPD.

Fetches registered investment adviser profiles (AUM, clients, fees, etc.)
from the SEC Investment Adviser Public Disclosure system.

Public API
----------
get_ria_profile(firm_name) -> RIAProfile
screen_rias(query, min_aum_billions, max_aum_billions, state, limit) -> RIAScreenResult
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Optional
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_IAPD_SEARCH = "https://api.adviserinfo.sec.gov/search/firm"
_IAPD_FIRM   = "https://api.adviserinfo.sec.gov/firm"
_EFTS_BASE   = "https://efts.sec.gov/LATEST/search-index"
_HEADERS = {
    "User-Agent": "SENTINEL/1.0 research@sentinel.ai",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 15.0

_AUM_KEYS = ("totalRegulatoryAssets", "regulatoryAssetsUnderManagement", "raum", "totalAssets")

_CLIENT_CATEGORIES: list[tuple[str, str]] = [
    ("individuals (other than", "Individuals"),
    ("high net worth", "High Net Worth"),
    ("banking or thrift", "Banks/Thrifts"),
    ("investment compan", "Investment Companies"),
    ("pooled investment", "Pooled Vehicles"),
    ("pension and profit", "Pension Funds"),
    ("charitable", "Charities"),
    ("state or municipal", "Government"),
    ("other investment adviser", "Other Advisers"),
    ("insurance", "Insurance"),
    ("sovereign wealth", "Sovereign Wealth"),
    ("corporations or other", "Corporations"),
]

_STYLE_KEYWORDS: list[tuple[str, str]] = [
    ("equity", "Equity"), ("fixed income", "Fixed Income"), ("bonds", "Fixed Income"),
    ("options", "Options"), ("derivatives", "Derivatives"), ("real estate", "Real Estate"),
    ("alternative", "Alternatives"), ("hedge", "Hedge Fund"), ("etf", "ETF"),
    ("mutual fund", "Mutual Fund"), ("private equity", "Private Equity"),
    ("venture", "Venture Capital"), ("esg", "ESG"), ("quantitative", "Quantitative"),
    ("passive", "Passive/Index"), ("index", "Passive/Index"),
    ("commodities", "Commodities"), ("crypto", "Digital Assets"),
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ClientType(BaseModel):
    category: str
    count: Optional[int] = None
    pct_of_clients: Optional[float] = None


class FeeStructure(BaseModel):
    pct_of_aum: bool = False
    hourly: bool = False
    fixed_fee: bool = False
    performance_based: bool = False
    subscription: bool = False


class RIAProfile(BaseModel):
    name: str
    crd_number: Optional[str] = None
    sec_number: Optional[str] = None
    aum_usd: Optional[float] = None
    aum_discretionary_usd: Optional[float] = None
    num_clients: Optional[int] = None
    num_employees: Optional[int] = None
    num_advisers: Optional[int] = None
    primary_state: Optional[str] = None
    registration_date: Optional[str] = None
    latest_adv_date: Optional[str] = None
    client_types: list[ClientType] = Field(default_factory=list)
    fee_structure: Optional[FeeStructure] = None
    has_discretion: Optional[bool] = None
    investment_styles: list[str] = Field(default_factory=list)
    custodians: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    as_of: str


class RIAScreenResult(BaseModel):
    query: str
    total_found: int
    profiles: list[RIAProfile]
    as_of: str


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _safe_int(v: object) -> Optional[int]:
    try: return int(v)  # type: ignore[arg-type]
    except Exception: return None

def _safe_float(v: object) -> Optional[float]:
    try: return float(v)  # type: ignore[arg-type]
    except Exception: return None

def _deep_get(d: dict, *keys: str) -> object:
    cur: object = d
    for k in keys:
        if not isinstance(cur, dict): return None
        cur = cur.get(k)
    return cur


async def _get_json(client: httpx.AsyncClient, url: str, warnings: list[str], label: str) -> dict | list:
    """GET a URL, returning parsed JSON or empty dict/list on error."""
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code == 429:
            warnings.append(f"{label}: rate-limited (429); partial result.")
            return {}
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("form_adv: HTTP error", label=label, status=exc.response.status_code)
        warnings.append(f"{label}: HTTP {exc.response.status_code}")
    except Exception as exc:
        logger.warning("form_adv: request failed", label=label, error=str(exc))
        warnings.append(f"{label}: {exc}")
    return {}


async def _iapd_search(client: httpx.AsyncClient, name: str, nrows: int, warnings: list[str]) -> list[dict]:
    url = f"{_IAPD_SEARCH}?query={quote(name)}&hl=true&nrows={nrows}&start=0&r=25&np=0&comptypes=130"
    data = await _get_json(client, url, warnings, "IAPD-search")
    return data.get("hits", {}).get("hits", []) if isinstance(data, dict) else []


async def _iapd_detail(client: httpx.AsyncClient, crd: str, warnings: list[str]) -> dict:
    data = await _get_json(client, f"{_IAPD_FIRM}/{crd}", warnings, f"IAPD-firm/{crd}")
    return data if isinstance(data, dict) else {}


async def _efts_adv(client: httpx.AsyncClient, name: str, warnings: list[str]) -> list[dict]:
    url = (f"{_EFTS_BASE}?q={quote(chr(34)+name+chr(34))}"
           f"&forms=ADV&dateRange=custom&startdt=2023-01-01")
    data = await _get_json(client, url, warnings, "EFTS-ADV")
    return data.get("hits", {}).get("hits", []) if isinstance(data, dict) else []


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_aum(adv: dict) -> tuple[Optional[float], Optional[float]]:
    total: Optional[float] = None
    disc: Optional[float] = None
    item5f = _deep_get(adv, "Part1A", "Item5F") or {}
    if isinstance(item5f, dict):
        for k in ("TotalRegulatoryAssets", "raum", "totalRaum", "AmountTotal"):
            if item5f.get(k) is not None and total is None:
                total = _safe_float(item5f[k])
        disc = _safe_float(item5f.get("AmountDiscretionary") or item5f.get("discretionaryRaum"))
    if total is None:
        for k in _AUM_KEYS:
            if adv.get(k) is not None:
                total = _safe_float(adv[k]); break
    # Amounts reported in millions → convert to dollars
    if total is not None and total < 100_000: total = total * 1_000_000
    if disc is not None and disc < 100_000: disc = disc * 1_000_000
    return total, disc


def _parse_client_types(adv: dict) -> list[ClientType]:
    item5d = _deep_get(adv, "Part1A", "Item5D") or {}
    if not isinstance(item5d, dict): return []
    raw = item5d.get("clientTypes") or item5d.get("ClientTypes") or []
    results: list[ClientType] = []
    for entry in (raw if isinstance(raw, list) else []):
        if not isinstance(entry, dict): continue
        label = (entry.get("clientType") or entry.get("Category") or entry.get("label") or "").lower()
        friendly = label.title()
        for prefix, nice in _CLIENT_CATEGORIES:
            if prefix in label: friendly = nice; break
        results.append(ClientType(
            category=friendly,
            count=_safe_int(entry.get("count") or entry.get("numberOfClients")),
            pct_of_clients=_safe_float(entry.get("percentage") or entry.get("pct")),
        ))
    return results


def _parse_fee_structure(adv: dict) -> Optional[FeeStructure]:
    item5e = _deep_get(adv, "Part1A", "Item5E") or {}
    raw = (item5e.get("compensationTypes") or item5e.get("CompensationArrangements")
           or adv.get("compensationTypes") or []) if isinstance(item5e, dict) else []
    if not raw: return None
    s = str(raw).lower()
    return FeeStructure(
        pct_of_aum="percentage" in s or "aum" in s or "assets under management" in s,
        hourly="hourly" in s,
        fixed_fee="fixed" in s or "flat" in s,
        performance_based="performance" in s or "incentive" in s,
        subscription="subscription" in s or "retainer" in s,
    )


def _parse_styles(adv: dict) -> list[str]:
    combined = " ".join([
        str(adv.get("advisoryServices", "")),
        str(_deep_get(adv, "Part2", "advisoryServices") or ""),
        str(adv.get("investmentStrategies", "")),
        str(adv.get("description", "")),
    ]).lower()
    seen: set[str] = set()
    return [label for kw, label in _STYLE_KEYWORDS if kw in combined and not seen.add(label)]  # type: ignore[func-returns-value]


def _parse_custodians(adv: dict) -> list[str]:
    raw = _deep_get(adv, "Part1A", "custodians") or adv.get("custodians") or []
    if not isinstance(raw, list): return []
    names: list[str] = []
    for e in raw:
        n = (e.get("name") or e.get("custodianName") or "") if isinstance(e, dict) else str(e)
        if n and n not in names: names.append(n)
    return names


def _build_profile(src: dict, detail: dict, warnings: list[str]) -> RIAProfile:
    today = date.today().isoformat()
    name = (src.get("org_nm") or src.get("firm_name") or src.get("firmName")
            or detail.get("firmName") or "Unknown")
    crd = str(src.get("org_pk") or src.get("crd_nb") or detail.get("crdNumber") or "") or None

    sec_number: Optional[str] = None
    for reg in (detail.get("registrations") or []):
        if isinstance(reg, dict) and "SEC" in str(reg.get("regAuthority", "")):
            sec_number = str(reg.get("regNumber") or reg.get("registrationNumber") or "") or None
            break

    basic = detail.get("basicInfo") or detail.get("firmInfo") or {}
    adv   = detail.get("adv") or {}
    total_aum, disc_aum = _parse_aum(adv)

    reg_raw = basic.get("registrationDate") or basic.get("initialFilingDate") or src.get("registration_dt")
    latest_raw = adv.get("filingDate") or adv.get("latestFilingDate") or detail.get("latestFilingDate") or src.get("latest_filing_date")

    disc_raw = (_deep_get(adv, "Part1A", "Item5F", "AmountDiscretionary")
                or adv.get("hasDiscretion") or adv.get("discretionaryClients"))
    has_disc: Optional[bool] = None
    if isinstance(disc_raw, bool): has_disc = disc_raw
    elif isinstance(disc_raw, (int, float)): has_disc = disc_raw > 0
    elif isinstance(disc_raw, str): has_disc = disc_raw.lower() in ("yes", "true", "y")

    return RIAProfile(
        name=str(name),
        crd_number=crd,
        sec_number=sec_number,
        aum_usd=total_aum,
        aum_discretionary_usd=disc_aum,
        num_clients=_safe_int(
            _deep_get(adv, "Part1A", "Item5D", "totalClients")
            or _deep_get(adv, "Part1A", "Item5D", "numberOfClients")
            or adv.get("numberOfClients") or basic.get("numClients")),
        num_employees=_safe_int(basic.get("numEmployees") or basic.get("totalEmployees")
                                or _deep_get(adv, "Part1A", "Item5B1")),
        num_advisers=_safe_int(basic.get("numAdvisers") or basic.get("registeredRepresentatives")
                               or _deep_get(adv, "Part1A", "Item5B2")),
        primary_state=basic.get("stateCode") or basic.get("state") or src.get("st") or None,
        registration_date=str(reg_raw)[:10] if reg_raw else None,
        latest_adv_date=str(latest_raw)[:10] if latest_raw else None,
        client_types=_parse_client_types(adv),
        fee_structure=_parse_fee_structure(adv),
        has_discretion=has_disc,
        investment_styles=_parse_styles(adv),
        custodians=_parse_custodians(adv),
        warnings=warnings,
        as_of=today,
    )


def _build_from_efts(hit: dict, warnings: list[str]) -> RIAProfile:
    src = hit.get("_source", {})
    display_names = src.get("display_names") or []
    name = ""
    if display_names and isinstance(display_names, list):
        first = display_names[0]
        name = first.get("entity", "") if isinstance(first, dict) else str(first)
    name = name or src.get("entity_name") or "Unknown"
    latest_raw = src.get("period_of_report") or src.get("file_date")
    warnings.append("Limited data: built from EDGAR EFTS fallback; IAPD detail unavailable.")
    return RIAProfile(
        name=str(name),
        sec_number=str(src.get("entity_id") or "")[:20] or None,
        latest_adv_date=str(latest_raw)[:10] if latest_raw else None,
        warnings=list(warnings),
        as_of=date.today().isoformat(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_ria_profile(firm_name: str) -> RIAProfile:
    """
    Look up a single RIA by firm name — returns the top match profile.

    Tries the IAPD REST API first; falls back to EDGAR EFTS ADV filing search.

    Raises:
        ValueError: If no RIA is found matching the given name.
    """
    warnings_: list[str] = []
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        hits = await _iapd_search(client, firm_name, nrows=5, warnings=warnings_)
        if hits:
            src = hits[0].get("_source") or hits[0]
            crd = str(src.get("org_pk") or src.get("crd_nb") or "")
            detail = await _iapd_detail(client, crd, warnings_) if crd else {}
            if not crd:
                warnings_.append("CRD number not found in IAPD search; detail incomplete.")
            profile = _build_profile(src, detail, warnings_)
            logger.info("form_adv: profile via IAPD", name=profile.name, crd=profile.crd_number)
            return profile

        logger.info("form_adv: IAPD empty; trying EFTS", query=firm_name)
        efts_hits = await _efts_adv(client, firm_name, warnings_)
        if efts_hits:
            profile = _build_from_efts(efts_hits[0], warnings_)
            logger.info("form_adv: profile via EFTS fallback", name=profile.name)
            return profile

    raise ValueError(f"No RIA found matching '{firm_name}'")


async def screen_rias(
    query: str,
    min_aum_billions: float | None = None,
    max_aum_billions: float | None = None,
    state: str | None = None,
    limit: int = 10,
) -> RIAScreenResult:
    """
    Screen RIAs by name query, AUM range, and/or primary state.

    Fetches top IAPD matches, enriches with firm detail in parallel,
    then applies AUM/state filters. Falls back to EDGAR EFTS if IAPD is empty.

    Returns:
        RIAScreenResult with matching profiles sorted by AUM descending.
    """
    today = date.today().isoformat()
    warnings_: list[str] = []
    fetch_n = max(limit * 3, 25)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        hits = await _iapd_search(client, query, nrows=fetch_n, warnings=warnings_)

        if not hits:
            logger.info("form_adv: screen_rias no IAPD hits; EFTS fallback", query=query)
            efts_hits = await _efts_adv(client, query, warnings_)
            profiles = [_build_from_efts(h, []) for h in efts_hits[:limit]]
            return RIAScreenResult(query=query, total_found=len(profiles), profiles=profiles, as_of=today)

        top_hits = hits[:min(fetch_n, 15)]

        async def _enrich(hit: dict) -> RIAProfile:
            src = hit.get("_source") or hit
            crd = str(src.get("org_pk") or src.get("crd_nb") or "")
            w: list[str] = []
            detail = await _iapd_detail(client, crd, w) if crd else {}
            if not crd: w.append("CRD unavailable; minimal profile.")
            return _build_profile(src, detail, w)

        all_profiles = list(await asyncio.gather(*[_enrich(h) for h in top_hits]))

    filtered: list[RIAProfile] = []
    for p in all_profiles:
        if state and (p.primary_state or "").upper() != state.upper(): continue
        if min_aum_billions and (p.aum_usd is None or p.aum_usd < min_aum_billions * 1e9): continue
        if max_aum_billions and p.aum_usd is not None and p.aum_usd > max_aum_billions * 1e9: continue
        filtered.append(p)

    filtered.sort(key=lambda p: p.aum_usd or -1.0, reverse=True)
    profiles = filtered[:limit]
    logger.info("form_adv: screen_rias done", query=query, total=len(filtered), returned=len(profiles))
    return RIAScreenResult(query=query, total_found=len(filtered), profiles=profiles, as_of=today)
