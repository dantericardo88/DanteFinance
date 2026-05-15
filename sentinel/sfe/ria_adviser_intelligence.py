"""
Form ADV / RIA adviser intelligence via SEC IAPD and EDGAR.
Covers registered investment advisers, AUM tracking, CRD numbers, Form ADV Part 1/2.

Dimension: dim_034 — Form ADV / RIA adviser intelligence (target score: 9)

Data sources (all free):
  SEC IAPD: https://api.adviserinfo.sec.gov/search/firm
  EDGAR EFTS: https://efts.sec.gov/LATEST/search-index
  SEC EDGAR data: https://data.sec.gov/

Public API
----------
IAPDAdapter
    search_firms(query, limit)              -> list[dict]
    get_firm_detail(crd)                    -> dict
    get_recent_adv_filings(days_back)       -> list[dict]

FormADVParser
    parse_adv_filing(accession, cik)        -> ADVParsedData
    extract_from_xml(xml_text)              -> ADVParsedData

RIAUniverse
    get_top_by_aum(n)                       -> list[RIARecord]
    get_by_crd(crd)                         -> RIARecord | None
    search(name)                            -> list[RIARecord]
    get_state_breakdown()                   -> dict[str, int]
    refresh(force)                          -> None

AdviserSignalEngine
    aum_growth_yoy(crd)                     -> float | None
    client_concentration(crd)               -> dict
    compensation_alignment_score(crd)       -> float
    disciplinary_flag(crd)                  -> bool
    get_new_launches(days_back)             -> list[RIARecord]

ETFOwnershipViaRIA
    institutional_bias_score(ticker)        -> float
    retail_vs_institutional(ticker)         -> dict

ria_router — FastAPI APIRouter, prefix /ria
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Optional
from urllib.parse import quote, urlencode

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IAPD_SEARCH   = "https://api.adviserinfo.sec.gov/search/firm"
_IAPD_FIRM     = "https://api.adviserinfo.sec.gov/firm"
_EFTS_BASE     = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_DATA    = "https://data.sec.gov"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

_TIMEOUT      = 30.0
_RATE_DELAY   = 0.12   # 120 ms between SEC requests
_MAX_RETRIES  = 3
_RETRY_DELAYS = [1.0, 2.0, 4.0]

# SQLite path for RIA universe cache
_DB_PATH = Path(__file__).parent.parent / "data" / "ria_adviser.db"

# Cache TTLs (seconds)
_TTL_FIRM_DETAIL = 86_400       # 24h for firm detail
_TTL_UNIVERSE    = 86_400       # 24h for universe
_TTL_SIGNALS     = 3_600        # 1h for signal data

# AUM bucket labels
_AUM_BUCKETS: list[tuple[str, float, float]] = [
    ("<$100M",       0,       100e6),
    ("$100M–$500M",  100e6,   500e6),
    ("$500M–$1B",    500e6,   1e9),
    ("$1B–$5B",      1e9,     5e9),
    ("$5B–$50B",     5e9,     50e9),
    (">$50B",        50e9,    float("inf")),
]

# Compensation type keywords from ADV text
_COMP_TYPES: list[tuple[str, str]] = [
    ("percentage of assets under management", "AUM-based fee"),
    ("percentage of aum",                    "AUM-based fee"),
    ("hourly charge",                        "Hourly charges"),
    ("fixed fee",                            "Fixed fee"),
    ("subscription fee",                     "Subscription fee"),
    ("performance-based fee",                "Performance fees"),
    ("performance fees",                     "Performance fees"),
    ("commission",                           "Commissions"),
    ("wrap fee",                             "Wrap fee program"),
    ("other",                                "Other"),
]

# Advisory service keywords
_SERVICE_KEYWORDS: list[tuple[str, str]] = [
    ("financial planning",           "Financial planning"),
    ("portfolio management",         "Portfolio management"),
    ("pension consulting",           "Pension consulting"),
    ("selection of other advisers",  "Manager selection"),
    ("publication of periodicals",   "Research/publications"),
    ("educational seminars",         "Educational seminars"),
    ("market timing",                "Market timing"),
    ("educational seminars",         "Educational seminars"),
]

# Business type keywords
_BUSINESS_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("broker-dealer",                        "Broker-dealer"),
    ("registered representative",            "Registered representative"),
    ("futures commission merchant",          "FCM"),
    ("commodity pool operator",              "CPO"),
    ("commodity trading advisor",            "CTA"),
    ("real estate broker",                   "Real estate broker"),
    ("insurance company or agency",          "Insurance"),
    ("bank",                                 "Bank"),
    ("trust company",                        "Trust company"),
    ("accounting firm",                      "Accounting firm"),
    ("law firm",                             "Law firm"),
    ("pension consultant",                   "Pension consultant"),
    ("mutual fund company",                  "Mutual fund company"),
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RIARecord(BaseModel):
    crd_number: str
    firm_name: str
    sec_number: Optional[str] = None
    aum: Optional[float] = None               # total regulatory AUM (USD)
    aum_discretionary: Optional[float] = None
    aum_nondiscretionary: Optional[float] = None
    pct_discretionary: Optional[float] = None
    num_clients: Optional[int] = None
    num_employees: Optional[int] = None
    num_investment_advisers: Optional[int] = None
    state: Optional[str] = None
    registration_date: Optional[str] = None
    last_amended: Optional[str] = None
    has_disciplinary_history: bool = False
    compensation_types: dict[str, bool] = Field(default_factory=dict)
    advisory_services: list[str] = Field(default_factory=list)
    business_types: list[str] = Field(default_factory=list)
    cached_at: Optional[float] = None


class ADVParsedData(BaseModel):
    crd_number: Optional[str] = None
    firm_name: Optional[str] = None
    total_aum: Optional[float] = None
    aum_discretionary: Optional[float] = None
    aum_nondiscretionary: Optional[float] = None
    num_clients: Optional[int] = None
    pct_discretionary: Optional[float] = None
    compensation_types: dict[str, bool] = Field(default_factory=dict)
    advisory_services: list[str] = Field(default_factory=list)
    business_types: list[str] = Field(default_factory=list)
    disciplinary_history: bool = False
    disciplinary_details: Optional[str] = None
    filing_date: Optional[str] = None
    accession_number: Optional[str] = None
    parse_warnings: list[str] = Field(default_factory=list)


class AUMGrowthSignal(BaseModel):
    crd_number: str
    firm_name: Optional[str] = None
    aum_current: Optional[float] = None
    aum_prior: Optional[float] = None
    growth_pct: Optional[float] = None        # YoY %
    filing_current: Optional[str] = None
    filing_prior: Optional[str] = None
    signal: str = "neutral"                   # "strong_growth" | "growth" | "neutral" | "decline"


class ConcentrationSignal(BaseModel):
    crd_number: str
    pct_discretionary: Optional[float] = None
    pct_nondiscretionary: Optional[float] = None
    concentration_score: float = 0.0          # 0–100, higher = more discretionary


class CompensationAlignmentScore(BaseModel):
    crd_number: str
    score: float                              # 0–100
    has_aum_fee: bool = False
    has_performance_fee: bool = False
    has_commission: bool = False
    alignment_label: str = "unknown"         # "aligned" | "mixed" | "misaligned" | "unknown"


class NewLaunchRecord(BaseModel):
    crd_number: str
    firm_name: str
    registration_date: Optional[str] = None
    filing_date: Optional[str] = None
    state: Optional[str] = None
    aum: Optional[float] = None
    days_since_filing: int = 0


class ETFOwnershipProfile(BaseModel):
    ticker: str
    institutional_bias_score: float           # 0–100
    institutional_count: int = 0
    retail_count: int = 0
    retail_vs_institutional_ratio: float = 0.0
    top_institutional_owners: list[dict] = Field(default_factory=list)
    data_source: str = "13F/EDGAR"


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _ensure_db() -> None:
    """Create database tables if they don't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ria_firms (
                crd_number      TEXT PRIMARY KEY,
                firm_name       TEXT,
                sec_number      TEXT,
                aum             REAL,
                aum_discretionary REAL,
                aum_nondiscretionary REAL,
                pct_discretionary REAL,
                num_clients     INTEGER,
                num_employees   INTEGER,
                num_investment_advisers INTEGER,
                state           TEXT,
                registration_date TEXT,
                last_amended    TEXT,
                has_disciplinary_history INTEGER DEFAULT 0,
                compensation_types TEXT,   -- JSON
                advisory_services TEXT,    -- JSON
                business_types TEXT,       -- JSON
                cached_at       REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS adv_cache (
                cache_key   TEXT PRIMARY KEY,
                payload     TEXT,           -- JSON
                cached_at   REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ria_aum ON ria_firms(aum DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ria_state ON ria_firms(state)
        """)
        conn.commit()


@contextmanager
def _db_conn() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _cache_get(cache_key: str, ttl: float) -> Optional[Any]:
    """Return cached payload if fresh, else None."""
    try:
        with _db_conn() as conn:
            row = conn.execute(
                "SELECT payload, cached_at FROM adv_cache WHERE cache_key = ?",
                (cache_key,)
            ).fetchone()
            if row and (time.time() - row["cached_at"]) < ttl:
                return json.loads(row["payload"])
    except Exception as exc:
        logger.debug("Cache get failed", key=cache_key, error=str(exc))
    return None


def _cache_set(cache_key: str, payload: Any) -> None:
    """Store payload in cache."""
    try:
        with _db_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO adv_cache (cache_key, payload, cached_at) VALUES (?,?,?)",
                (cache_key, json.dumps(payload, default=str), time.time())
            )
            conn.commit()
    except Exception as exc:
        logger.debug("Cache set failed", key=cache_key, error=str(exc))


def _upsert_ria(record: RIARecord) -> None:
    try:
        with _db_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ria_firms
                    (crd_number, firm_name, sec_number, aum, aum_discretionary,
                     aum_nondiscretionary, pct_discretionary, num_clients,
                     num_employees, num_investment_advisers, state,
                     registration_date, last_amended, has_disciplinary_history,
                     compensation_types, advisory_services, business_types, cached_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                record.crd_number,
                record.firm_name,
                record.sec_number,
                record.aum,
                record.aum_discretionary,
                record.aum_nondiscretionary,
                record.pct_discretionary,
                record.num_clients,
                record.num_employees,
                record.num_investment_advisers,
                record.state,
                record.registration_date,
                record.last_amended,
                int(record.has_disciplinary_history),
                json.dumps(record.compensation_types),
                json.dumps(record.advisory_services),
                json.dumps(record.business_types),
                time.time(),
            ))
            conn.commit()
    except Exception as exc:
        logger.warning("RIA upsert failed", crd=record.crd_number, error=str(exc))


def _row_to_record(row: sqlite3.Row) -> RIARecord:
    return RIARecord(
        crd_number=row["crd_number"],
        firm_name=row["firm_name"] or "",
        sec_number=row["sec_number"],
        aum=row["aum"],
        aum_discretionary=row["aum_discretionary"],
        aum_nondiscretionary=row["aum_nondiscretionary"],
        pct_discretionary=row["pct_discretionary"],
        num_clients=row["num_clients"],
        num_employees=row["num_employees"],
        num_investment_advisers=row["num_investment_advisers"],
        state=row["state"],
        registration_date=row["registration_date"],
        last_amended=row["last_amended"],
        has_disciplinary_history=bool(row["has_disciplinary_history"]),
        compensation_types=json.loads(row["compensation_types"] or "{}"),
        advisory_services=json.loads(row["advisory_services"] or "[]"),
        business_types=json.loads(row["business_types"] or "[]"),
        cached_at=row["cached_at"],
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[dict] = None,
    retries: int = _MAX_RETRIES,
) -> Optional[dict]:
    """GET with retry and rate limiting."""
    for attempt in range(retries):
        try:
            await asyncio.sleep(_RATE_DELAY)
            r = await client.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)] * 2
                logger.warning("IAPD rate-limited", url=url, wait=wait)
                await asyncio.sleep(wait)
                continue
            logger.warning("HTTP non-200", url=url, status=r.status_code)
            return None
        except Exception as exc:
            if attempt < retries - 1:
                await asyncio.sleep(_RETRY_DELAYS[attempt])
            else:
                logger.error("HTTP request failed", url=url, error=str(exc))
    return None


async def _get_text(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[dict] = None,
) -> Optional[str]:
    """GET returning raw text."""
    try:
        await asyncio.sleep(_RATE_DELAY)
        r = await client.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.text
        return None
    except Exception as exc:
        logger.warning("Text fetch failed", url=url, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# 1. IAPDAdapter
# ---------------------------------------------------------------------------

class IAPDAdapter:
    """
    SEC IAPD public API adapter.
    Fetches adviser firm data, AUM, CRD numbers, Form ADV filings.
    """

    async def search_firms(
        self,
        query: str,
        limit: int = 25,
    ) -> list[dict]:
        """
        Search for RIA firms by name via IAPD API.
        Returns raw adviser info dicts.
        """
        cache_key = f"iapd_search:{hashlib.md5(f'{query}{limit}'.encode()).hexdigest()}"
        cached = _cache_get(cache_key, _TTL_FIRM_DETAIL)
        if cached is not None:
            return cached

        params = {
            "query": query,
            "dataType": "FirmInfo",
            "noDataPull": "true",
        }

        async with httpx.AsyncClient() as client:
            data = await _get_json(client, _IAPD_SEARCH, params=params)

        if not data:
            return []

        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits[:limit]:
            src = hit.get("_source", {})
            firm_info = src.get("FirmInfo", {})
            results.append({
                "crd_number": str(firm_info.get("FirmCRD", "")),
                "firm_name":  firm_info.get("FirmName", ""),
                "sec_number": firm_info.get("SecNumber", ""),
                "state":      firm_info.get("StateCode", ""),
                "has_disciplinary_history": firm_info.get("InvstAdvDisclosureFlag", "N") == "Y",
            })

        _cache_set(cache_key, results)
        return results

    async def get_firm_detail(self, crd: str) -> dict:
        """
        Fetch full Form ADV Part 1 data for a specific CRD number.
        Returns dict with AUM, employees, clients, fee structures.
        """
        cache_key = f"iapd_firm:{crd}"
        cached = _cache_get(cache_key, _TTL_FIRM_DETAIL)
        if cached is not None:
            return cached

        url = f"{_IAPD_FIRM}/{crd}/iadv/formadv/formadvsummary"

        async with httpx.AsyncClient() as client:
            data = await _get_json(client, url)

        if not data:
            # Fallback: try the full IAPD firm endpoint
            async with httpx.AsyncClient() as client:
                data = await _get_json(client, f"{_IAPD_FIRM}/{crd}")

        if not data:
            return {"crd_number": crd, "error": "Not found"}

        result = self._parse_firm_detail(crd, data)
        _cache_set(cache_key, result)
        return result

    def _parse_firm_detail(self, crd: str, data: dict) -> dict:
        """Parse IAPD API response into standardised dict."""
        # The IAPD API nests differently per endpoint; handle both shapes
        firm = data.get("firmSummary", data.get("FirmInfo", data))

        # AUM (in millions from IAPD, need to normalise)
        raw_aum = (
            firm.get("totalRegulatoryAssets")
            or firm.get("TotalRegulatoryAssets")
            or firm.get("AUMTotal")
            or 0.0
        )
        try:
            aum = float(raw_aum)
            # IAPD often reports in dollars already; if tiny, assume millions
            if 0 < aum < 1e6:
                aum *= 1_000_000
        except (TypeError, ValueError):
            aum = None

        raw_disc = (
            firm.get("discretionaryAssets")
            or firm.get("DiscretionaryAssets")
            or 0.0
        )
        try:
            aum_disc = float(raw_disc)
            if 0 < aum_disc < 1e6:
                aum_disc *= 1_000_000
        except (TypeError, ValueError):
            aum_disc = None

        raw_nondisc = (
            firm.get("nonDiscretionaryAssets")
            or firm.get("NonDiscretionaryAssets")
            or 0.0
        )
        try:
            aum_nondisc = float(raw_nondisc)
            if 0 < aum_nondisc < 1e6:
                aum_nondisc *= 1_000_000
        except (TypeError, ValueError):
            aum_nondisc = None

        # Pct discretionary
        pct_disc = None
        if aum and aum > 0 and aum_disc is not None:
            pct_disc = round(aum_disc / aum * 100, 2)

        num_clients = (
            firm.get("numberOfAccounts")
            or firm.get("NumAccounts")
            or firm.get("totalAccounts")
        )
        num_employees = (
            firm.get("totalEmployees")
            or firm.get("TotalEmployees")
            or firm.get("numberOfEmployees")
        )

        registration_date = (
            firm.get("registrationDate")
            or firm.get("RegistrationDate")
        )
        last_amended = (
            firm.get("lastAmendedDate")
            or firm.get("LastAmendedDate")
            or firm.get("filingDate")
        )

        # Disciplinary
        disc_flag = (
            firm.get("disclosureFlag")
            or firm.get("DisclosureFlag")
            or firm.get("InvstAdvDisclosureFlag")
            or "N"
        )
        has_disc = str(disc_flag).upper() in ("Y", "YES", "TRUE", "1")

        # State
        state = (
            firm.get("stateCode")
            or firm.get("StateCode")
            or firm.get("mainOfficeStateCode")
            or ""
        )

        # SEC number
        sec_number = (
            firm.get("secNumber")
            or firm.get("SecNumber")
            or ""
        )

        # Fee / compensation types (list from IAPD)
        comp_raw = (
            firm.get("compensationTypes")
            or firm.get("feeTypes")
            or []
        )
        comp_types: dict[str, bool] = {}
        if isinstance(comp_raw, list):
            for c in comp_raw:
                if isinstance(c, str):
                    comp_types[c] = True
                elif isinstance(c, dict):
                    label = c.get("type") or c.get("Type") or str(c)
                    comp_types[label] = True

        return {
            "crd_number": crd,
            "sec_number": sec_number,
            "firm_name": (
                firm.get("firmName")
                or firm.get("FirmName")
                or ""
            ),
            "aum": aum,
            "aum_discretionary": aum_disc,
            "aum_nondiscretionary": aum_nondisc,
            "pct_discretionary": pct_disc,
            "num_clients": _to_int(num_clients),
            "num_employees": _to_int(num_employees),
            "state": state,
            "registration_date": _clean_date(registration_date),
            "last_amended": _clean_date(last_amended),
            "has_disciplinary_history": has_disc,
            "compensation_types": comp_types,
        }

    async def get_recent_adv_filings(self, days_back: int = 30) -> list[dict]:
        """
        Fetch recent Form ADV filings from EDGAR EFTS.
        Returns list of filing metadata dicts.
        """
        cache_key = f"efts_adv_recent:{days_back}"
        cached = _cache_get(cache_key, 3_600)  # 1h TTL for recent filings
        if cached is not None:
            return cached

        start_dt = (date.today() - timedelta(days=days_back)).isoformat()
        end_dt   = date.today().isoformat()

        params = {
            "q": '"form adv"',
            "dateRange": "custom",
            "startdt": start_dt,
            "enddt": end_dt,
            "forms": "ADV",
            "_source": "period_of_report,display_names,entity_id,file_date,form_type,accession_no,file_num",
            "from": "0",
            "size": "100",
        }

        async with httpx.AsyncClient() as client:
            data = await _get_json(client, _EFTS_BASE, params=params)

        if not data:
            return []

        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            src = hit.get("_source", {})
            names = src.get("display_names", [])
            firm_name = names[0] if names else ""
            results.append({
                "firm_name":    firm_name,
                "entity_id":    src.get("entity_id", ""),
                "file_date":    src.get("file_date", ""),
                "form_type":    src.get("form_type", ""),
                "accession_no": src.get("accession_no", ""),
                "file_num":     src.get("file_num", ""),
                "period":       src.get("period_of_report", ""),
            })

        _cache_set(cache_key, results)
        return results

    async def get_adv_history_for_crd(self, crd: str) -> list[dict]:
        """
        Fetch all historical ADV filings for a CRD number via EDGAR.
        Returns filing list sorted oldest→newest.
        """
        cache_key = f"adv_history_crd:{crd}"
        cached = _cache_get(cache_key, _TTL_FIRM_DETAIL)
        if cached is not None:
            return cached

        url = f"{_EDGAR_DATA}/submissions/CIK{crd.zfill(10)}.json"

        async with httpx.AsyncClient() as client:
            data = await _get_json(client, url)

        if not data:
            return []

        filings = data.get("filings", {}).get("recent", {})
        forms     = filings.get("form", [])
        dates     = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        periods    = filings.get("reportDate", [])

        results = []
        for form, dt, acc, period in zip(forms, dates, accessions, periods):
            if "ADV" in str(form).upper():
                results.append({
                    "form_type":    form,
                    "file_date":    dt,
                    "accession_no": acc,
                    "period":       period,
                    "crd_number":   crd,
                })

        results.sort(key=lambda x: x["file_date"])
        _cache_set(cache_key, results)
        return results


# ---------------------------------------------------------------------------
# 2. FormADVParser
# ---------------------------------------------------------------------------

class FormADVParser:
    """
    Parse Form ADV Part 1A from EDGAR XBRL/XML/text filings.
    Extracts: total_aum, num_clients, pct_discretionary, compensation_types,
    advisory_services, business_types, disciplinary_history.
    """

    async def parse_adv_filing(
        self, accession_no: str, cik: str
    ) -> ADVParsedData:
        """
        Download and parse an ADV filing from EDGAR archives.
        Tries XBRL first, falls back to raw text extraction.
        """
        cache_key = f"adv_parsed:{accession_no}"
        cached = _cache_get(cache_key, _TTL_UNIVERSE)
        if cached is not None:
            return ADVParsedData(**cached)

        cik_clean = cik.zfill(10)
        acc_clean = accession_no.replace("-", "")
        base_url = f"{_EDGAR_ARCHIVES}/{cik_clean}/{acc_clean}"
        index_url = f"{base_url}/{accession_no}-index.htm"

        async with httpx.AsyncClient() as client:
            # Try index page to find the primary XML document
            index_text = await _get_text(client, index_url)
            xml_url = None

            if index_text:
                # Look for ADV XML/XBRL document links
                for pattern in [r'href="([^"]*adv[^"]*\.xml)"', r'href="([^"]*\.xml)"']:
                    m = re.search(pattern, index_text, re.IGNORECASE)
                    if m:
                        xml_url = f"{base_url}/{m.group(1)}"
                        break

            xml_text = None
            if xml_url:
                xml_text = await _get_text(client, xml_url)

            # If no XML found, try the primary document as text
            if not xml_text:
                # Try direct .htm or .txt document
                for ext in [".htm", ".txt", "-primary.xml"]:
                    doc_url = f"{base_url}/{accession_no}{ext}"
                    xml_text = await _get_text(client, doc_url)
                    if xml_text:
                        break

        result: ADVParsedData
        if xml_text and xml_text.strip().startswith("<"):
            result = self.extract_from_xml(xml_text)
        elif xml_text:
            result = self._extract_from_text(xml_text)
        else:
            result = ADVParsedData(parse_warnings=["Filing text not retrieved"])

        result.accession_number = accession_no
        _cache_set(cache_key, result.model_dump())
        return result

    def extract_from_xml(self, xml_text: str) -> ADVParsedData:
        """
        Parse ADV XML/XBRL. Handles both XBRL namespace and plain XML shapes
        produced by the SEC EDGAR ADV renderer.
        """
        warnings: list[str] = []
        data = ADVParsedData()

        try:
            # Strip default namespace for easier querying
            clean = re.sub(r'\sxmlns(?::\w+)?="[^"]+"', "", xml_text)
            root = ET.fromstring(clean)
        except ET.ParseError as exc:
            warnings.append(f"XML parse error: {exc}")
            return ADVParsedData(parse_warnings=warnings)

        def _find_text(*tags: str) -> Optional[str]:
            for tag in tags:
                el = root.find(f".//{tag}")
                if el is not None and el.text:
                    return el.text.strip()
            return None

        def _find_float(*tags: str) -> Optional[float]:
            t = _find_text(*tags)
            if t:
                try:
                    return float(t.replace(",", "").replace("$", ""))
                except ValueError:
                    pass
            return None

        # CRD / firm name
        data.crd_number = _find_text("CrdNumber", "CRDNumber", "crdNumber", "FirmCRD")
        data.firm_name  = _find_text("FirmName", "firmName", "CompanyName", "EntityName")

        # AUM fields (SEC XBRL uses different tag names across filings)
        total_aum = _find_float(
            "TotalRegulatoryAssetsUnderMgmt", "TotalRegulatoryAssets",
            "TotalAUM", "totalAUM", "RegulatoryAUM",
        )
        if total_aum and total_aum < 1e6:
            total_aum *= 1_000_000
        data.total_aum = total_aum

        disc_aum = _find_float(
            "DiscretionaryRegAssets", "DiscretionaryAUM", "discretionaryAUM",
            "TotalDiscretionaryAssets",
        )
        if disc_aum and disc_aum < 1e6:
            disc_aum *= 1_000_000
        data.aum_discretionary = disc_aum

        nondisc_aum = _find_float(
            "NondiscretionaryRegAssets", "NondiscretionaryAUM", "nondiscretionaryAUM",
            "TotalNonDiscretionaryAssets",
        )
        if nondisc_aum and nondisc_aum < 1e6:
            nondisc_aum *= 1_000_000
        data.aum_nondiscretionary = nondisc_aum

        # Pct discretionary
        if data.total_aum and data.total_aum > 0 and data.aum_discretionary is not None:
            data.pct_discretionary = round(data.aum_discretionary / data.total_aum * 100, 2)
        else:
            pct_raw = _find_float("PctDiscretionary", "pctDiscretionary")
            if pct_raw:
                data.pct_discretionary = pct_raw

        # Num clients
        raw_clients = _find_text(
            "NumberOfClients", "NumClients", "numberOfAccounts",
            "TotalNumberOfAccounts", "TotalClients",
        )
        data.num_clients = _to_int(raw_clients)

        # Compensation types — look for checkbox elements
        comp_section = root.find(".//CompensationArrangements") or root
        comp_types: dict[str, bool] = {}
        for kw, label in _COMP_TYPES:
            # Search for tag text or attribute values containing keyword
            found = False
            for el in comp_section.iter():
                text_check = (el.text or "").lower()
                tag_check  = el.tag.lower()
                if kw.lower() in text_check or kw.lower() in tag_check:
                    found = True
                    break
            if found:
                comp_types[label] = True
        data.compensation_types = comp_types

        # Advisory services
        services: list[str] = []
        for kw, label in _SERVICE_KEYWORDS:
            for el in root.iter():
                if kw.lower() in (el.text or "").lower():
                    if label not in services:
                        services.append(label)
                    break
        data.advisory_services = services

        # Business types
        biz_types: list[str] = []
        for kw, label in _BUSINESS_TYPE_KEYWORDS:
            for el in root.iter():
                if kw.lower() in (el.text or "").lower():
                    if label not in biz_types:
                        biz_types.append(label)
                    break
        data.business_types = biz_types

        # Disciplinary history — look for any positive disclosure flag
        disc_text = _find_text(
            "DisclosureFlag", "HasDisciplinaryHistory",
            "DisciplinaryHistory", "IsDisciplinaryFlag",
        )
        if disc_text:
            data.disciplinary_history = str(disc_text).upper() in ("Y", "YES", "TRUE", "1")

        # Look for disclosure details
        disc_detail_el = root.find(".//DisciplinaryDisclosures")
        if disc_detail_el is not None:
            details = " ".join(
                (el.text or "").strip()
                for el in disc_detail_el.iter()
                if el.text and el.text.strip()
            )[:1000]
            if details:
                data.disciplinary_details = details
                data.disciplinary_history = True

        data.parse_warnings = warnings
        return data

    def _extract_from_text(self, text: str) -> ADVParsedData:
        """
        Fallback: extract key fields via regex from raw ADV text/HTML.
        """
        data = ADVParsedData()
        warnings: list[str] = ["Using text fallback parser"]
        low = text.lower()

        # AUM via regex
        aum_patterns = [
            r'regulatory\s+assets[^$\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)?',
            r'total\s+aum[^$\d]*\$?([\d,]+(?:\.\d+)?)',
            r'assets\s+under\s+management[^$\d]*\$?([\d,]+(?:\.\d+)?)',
        ]
        for pat in aum_patterns:
            m = re.search(pat, low)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    # Guess scale from context
                    if "billion" in low[max(0, m.start()-50):m.end()+50]:
                        val *= 1e9
                    elif "million" in low[max(0, m.start()-50):m.end()+50]:
                        val *= 1e6
                    elif val < 1e6:
                        val *= 1_000_000
                    data.total_aum = val
                    break
                except ValueError:
                    pass

        # Client count
        client_m = re.search(r'(?:number\s+of\s+clients|total\s+clients)[^:\d]*:?\s*([\d,]+)', low)
        if client_m:
            data.num_clients = _to_int(client_m.group(1))

        # Compensation types
        comp_types: dict[str, bool] = {}
        for kw, label in _COMP_TYPES:
            if kw.lower() in low:
                comp_types[label] = True
        data.compensation_types = comp_types

        # Advisory services
        services: list[str] = []
        for kw, label in _SERVICE_KEYWORDS:
            if kw.lower() in low:
                services.append(label)
        data.advisory_services = list(dict.fromkeys(services))

        # Business types
        biz_types: list[str] = []
        for kw, label in _BUSINESS_TYPE_KEYWORDS:
            if kw.lower() in low:
                biz_types.append(label)
        data.business_types = list(dict.fromkeys(biz_types))

        # Disciplinary
        disc_keywords = ["disciplinary history", "legal or disciplinary", "criminal action",
                         "regulatory action", "civil judicial action"]
        has_disc_keyword = any(kw in low for kw in disc_keywords)
        # Heuristic: if any of these appear with "yes" nearby, flag it
        if has_disc_keyword:
            context_window = 200
            for kw in disc_keywords:
                idx = low.find(kw)
                if idx != -1:
                    window = low[max(0, idx-50):idx + context_window]
                    if re.search(r'\byes\b|\b(?:has|had)\s+(?:a\s+)?(?:disciplinary|criminal)', window):
                        data.disciplinary_history = True
                        break

        data.parse_warnings = warnings
        return data


# ---------------------------------------------------------------------------
# 3. RIAUniverse
# ---------------------------------------------------------------------------

class RIAUniverse:
    """
    Universe of top RIAs by AUM with SQLite caching (24h TTL).
    """

    def __init__(self) -> None:
        _ensure_db()
        self._iapd = IAPDAdapter()
        self._parser = FormADVParser()

    def _is_stale(self) -> bool:
        """True if the universe cache is older than 24h or empty."""
        try:
            with _db_conn() as conn:
                row = conn.execute(
                    "SELECT MIN(cached_at) as oldest FROM ria_firms"
                ).fetchone()
                if not row or not row["oldest"]:
                    return True
                return (time.time() - row["oldest"]) > _TTL_UNIVERSE
        except Exception:
            return True

    def _count(self) -> int:
        try:
            with _db_conn() as conn:
                return conn.execute("SELECT COUNT(*) FROM ria_firms").fetchone()[0]
        except Exception:
            return 0

    async def refresh(self, force: bool = False) -> None:
        """
        Populate RIA universe from IAPD + EDGAR ADV filings.
        Fetches top firms sorted by AUM from IAPD search categories.
        """
        if not force and not self._is_stale() and self._count() > 0:
            logger.info("RIA universe fresh, skipping refresh")
            return

        logger.info("Refreshing RIA universe...")

        # Use EFTS to find the latest ADV filings over past 365 days
        # and pull firm details for top registrants
        start_dt = (date.today() - timedelta(days=365)).isoformat()
        end_dt   = date.today().isoformat()

        params = {
            "forms": "ADV",
            "dateRange": "custom",
            "startdt": start_dt,
            "enddt": end_dt,
            "_source": "display_names,entity_id,file_date,form_type,accession_no",
            "from": "0",
            "size": "100",
        }

        async with httpx.AsyncClient() as client:
            data = await _get_json(client, _EFTS_BASE, params=params)

        if not data:
            logger.warning("RIA universe: EFTS returned no data")
            return

        hits = data.get("hits", {}).get("hits", [])
        crds_seen: set[str] = set()

        tasks = []
        for hit in hits[:50]:  # limit to 50 to stay within rate limits
            src = hit.get("_source", {})
            entity_id = str(src.get("entity_id", ""))
            if entity_id and entity_id not in crds_seen:
                crds_seen.add(entity_id)
                tasks.append(self._fetch_and_store(entity_id))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Also seed from known large firms via common searches
        seed_queries = ["Vanguard", "Fidelity", "BlackRock", "Schwab", "Pimco",
                        "Capital Group", "Wellington", "Dimensional", "T. Rowe"]
        for q in seed_queries:
            firms = await self._iapd.search_firms(q, limit=5)
            for firm in firms:
                crd = firm.get("crd_number", "")
                if crd and crd not in crds_seen:
                    crds_seen.add(crd)
                    await self._fetch_and_store(crd)

        logger.info("RIA universe refresh complete", firms=self._count())

    async def _fetch_and_store(self, crd: str) -> None:
        """Fetch firm detail and upsert into DB."""
        try:
            detail = await self._iapd.get_firm_detail(crd)
            record = RIARecord(
                crd_number=detail.get("crd_number", crd),
                firm_name=detail.get("firm_name", ""),
                sec_number=detail.get("sec_number", ""),
                aum=detail.get("aum"),
                aum_discretionary=detail.get("aum_discretionary"),
                aum_nondiscretionary=detail.get("aum_nondiscretionary"),
                pct_discretionary=detail.get("pct_discretionary"),
                num_clients=detail.get("num_clients"),
                num_employees=detail.get("num_employees"),
                state=detail.get("state", ""),
                registration_date=detail.get("registration_date", ""),
                last_amended=detail.get("last_amended", ""),
                has_disciplinary_history=detail.get("has_disciplinary_history", False),
                compensation_types=detail.get("compensation_types", {}),
                advisory_services=detail.get("advisory_services", []),
                business_types=detail.get("business_types", []),
            )
            _upsert_ria(record)
        except Exception as exc:
            logger.debug("Fetch/store failed", crd=crd, error=str(exc))

    def get_top_by_aum(self, n: int = 50) -> list[RIARecord]:
        """Return top n RIAs by AUM, descending."""
        with _db_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM ria_firms WHERE aum IS NOT NULL ORDER BY aum DESC LIMIT ?",
                (n,)
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_by_crd(self, crd: str) -> Optional[RIARecord]:
        """Return a single RIA record by CRD number."""
        with _db_conn() as conn:
            row = conn.execute(
                "SELECT * FROM ria_firms WHERE crd_number = ?", (crd,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def search(self, name: str) -> list[RIARecord]:
        """Full-text search on firm name (case-insensitive LIKE)."""
        with _db_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM ria_firms WHERE LOWER(firm_name) LIKE ? ORDER BY aum DESC LIMIT 25",
                (f"%{name.lower()}%",)
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_state_breakdown(self) -> dict[str, int]:
        """Return {state_code: firm_count} for all registered states."""
        with _db_conn() as conn:
            rows = conn.execute(
                "SELECT state, COUNT(*) as cnt FROM ria_firms WHERE state IS NOT NULL "
                "GROUP BY state ORDER BY cnt DESC"
            ).fetchall()
        return {r["state"]: r["cnt"] for r in rows if r["state"]}

    def get_aum_distribution(self) -> dict[str, int]:
        """Return AUM bucket distribution across the universe."""
        with _db_conn() as conn:
            rows = conn.execute(
                "SELECT aum FROM ria_firms WHERE aum IS NOT NULL"
            ).fetchall()
        aums = [r["aum"] for r in rows]
        dist: dict[str, int] = {label: 0 for label, _, _ in _AUM_BUCKETS}
        for aum in aums:
            for label, lo, hi in _AUM_BUCKETS:
                if lo <= aum < hi:
                    dist[label] += 1
                    break
        return dist


# ---------------------------------------------------------------------------
# 4. AdviserSignalEngine
# ---------------------------------------------------------------------------

class AdviserSignalEngine:
    """
    Compute investment-grade signals from RIA data.
    """

    def __init__(self) -> None:
        self._iapd   = IAPDAdapter()
        self._parser = FormADVParser()
        self._universe = RIAUniverse()

    async def aum_growth_yoy(self, crd: str) -> AUMGrowthSignal:
        """
        Compute AUM growth YoY by comparing the two most recent ADV filings.
        Requires at least 2 filings in EDGAR history.
        """
        history = await self._iapd.get_adv_history_for_crd(crd)
        if len(history) < 2:
            return AUMGrowthSignal(
                crd_number=crd,
                signal="insufficient_data",
            )

        # Most recent and prior filing
        latest  = history[-1]
        prior   = history[-2]

        async def _parse_filing(f: dict) -> Optional[float]:
            try:
                parsed = await self._parser.parse_adv_filing(
                    f["accession_no"], crd
                )
                return parsed.total_aum
            except Exception:
                return None

        aum_now, aum_then = await asyncio.gather(
            _parse_filing(latest),
            _parse_filing(prior),
        )

        # Fallback: use IAPD current detail for latest
        if aum_now is None:
            detail = await self._iapd.get_firm_detail(crd)
            aum_now  = detail.get("aum")
            firm_name = detail.get("firm_name", "")
        else:
            detail = await self._iapd.get_firm_detail(crd)
            firm_name = detail.get("firm_name", "")

        growth_pct: Optional[float] = None
        if aum_now is not None and aum_then and aum_then > 0:
            growth_pct = round((aum_now - aum_then) / aum_then * 100, 2)

        signal = "neutral"
        if growth_pct is not None:
            if growth_pct >= 25:
                signal = "strong_growth"
            elif growth_pct >= 10:
                signal = "growth"
            elif growth_pct <= -10:
                signal = "decline"

        return AUMGrowthSignal(
            crd_number=crd,
            firm_name=firm_name,
            aum_current=aum_now,
            aum_prior=aum_then,
            growth_pct=growth_pct,
            filing_current=latest.get("file_date"),
            filing_prior=prior.get("file_date"),
            signal=signal,
        )

    async def client_concentration(self, crd: str) -> ConcentrationSignal:
        """
        Analyse client concentration: pct discretionary vs nondiscretionary.
        High discretionary = adviser has more control.
        """
        record = self._universe.get_by_crd(crd)
        if record:
            pct_d = record.pct_discretionary
        else:
            detail = await self._iapd.get_firm_detail(crd)
            pct_d = detail.get("pct_discretionary")

        if pct_d is None:
            return ConcentrationSignal(crd_number=crd, concentration_score=0.0)

        pct_nd = round(100.0 - pct_d, 2) if pct_d is not None else None
        score  = round(pct_d, 2) if pct_d is not None else 0.0

        return ConcentrationSignal(
            crd_number=crd,
            pct_discretionary=pct_d,
            pct_nondiscretionary=pct_nd,
            concentration_score=score,
        )

    async def compensation_alignment_score(self, crd: str) -> CompensationAlignmentScore:
        """
        Score compensation alignment:
          AUM-based fees → aligned (high score)
          Commissions → potential conflict (low score)
          Performance fees → mixed
        Score 0–100.
        """
        record = self._universe.get_by_crd(crd)
        if record:
            comp = record.compensation_types
        else:
            detail = await self._iapd.get_firm_detail(crd)
            comp = detail.get("compensation_types", {})

        has_aum    = "AUM-based fee" in comp
        has_perf   = "Performance fees" in comp
        has_comm   = "Commissions" in comp
        has_fixed  = "Fixed fee" in comp
        has_hourly = "Hourly charges" in comp

        score = 50.0  # base
        if has_aum:
            score += 30.0
        if has_perf:
            score += 10.0   # not penalised, adds upside alignment
        if has_comm:
            score -= 25.0   # conflict of interest
        if has_fixed or has_hourly:
            score += 5.0    # neutral/positive

        score = max(0.0, min(100.0, score))

        if score >= 75:
            label = "aligned"
        elif score >= 50:
            label = "mixed"
        else:
            label = "misaligned"

        return CompensationAlignmentScore(
            crd_number=crd,
            score=round(score, 1),
            has_aum_fee=has_aum,
            has_performance_fee=has_perf,
            has_commission=has_comm,
            alignment_label=label,
        )

    async def disciplinary_flag(self, crd: str) -> bool:
        """Return True if the adviser has any disciplinary history."""
        record = self._universe.get_by_crd(crd)
        if record:
            return record.has_disciplinary_history
        detail = await self._iapd.get_firm_detail(crd)
        return detail.get("has_disciplinary_history", False)

    async def get_new_launches(self, days_back: int = 90) -> list[NewLaunchRecord]:
        """
        Find recently registered RIAs (ADV filings within `days_back` days).
        New launches signal emerging managers.
        """
        cache_key = f"new_launches:{days_back}"
        cached = _cache_get(cache_key, 3_600)
        if cached is not None:
            return [NewLaunchRecord(**r) for r in cached]

        filings = await self._iapd.get_recent_adv_filings(days_back=days_back)
        today = date.today()
        results: list[NewLaunchRecord] = []

        for f in filings:
            file_date_str = f.get("file_date", "")
            if not file_date_str:
                continue
            try:
                file_date = date.fromisoformat(file_date_str[:10])
            except ValueError:
                continue

            days_since = (today - file_date).days
            if days_since > days_back:
                continue

            entity_id = f.get("entity_id", "")
            # Avoid excessive API calls — use cached data if available
            record = self._universe.get_by_crd(entity_id) if entity_id else None

            results.append(NewLaunchRecord(
                crd_number=entity_id,
                firm_name=f.get("firm_name", ""),
                filing_date=file_date_str,
                aum=record.aum if record else None,
                state=record.state if record else None,
                days_since_filing=days_since,
            ))

        _cache_set(cache_key, [r.model_dump() for r in results])
        return results

    async def composite_signals(self, crd: str) -> dict:
        """
        Compute all signals for an adviser and return combined dict.
        """
        growth, concentration, alignment, disciplinary = await asyncio.gather(
            self.aum_growth_yoy(crd),
            self.client_concentration(crd),
            self.compensation_alignment_score(crd),
            asyncio.coroutine(lambda: self.disciplinary_flag(crd))(),  # type: ignore[arg-type]
        )
        return {
            "aum_growth":       growth.model_dump(),
            "concentration":    concentration.model_dump(),
            "alignment":        alignment.model_dump(),
            "has_disciplinary": disciplinary,
        }


# ---------------------------------------------------------------------------
# 5. ETFOwnershipViaRIA
# ---------------------------------------------------------------------------

class ETFOwnershipViaRIA:
    """
    Cross-reference ETF/fund 13F holdings with RIA adviser type.
    Computes institutional bias score and retail/institutional ratio.
    """

    async def get_13f_holders(self, ticker: str) -> list[dict]:
        """
        Fetch institutional holders from EDGAR 13F filings for a ticker.
        Uses EFTS to find filings mentioning the ticker.
        """
        cache_key = f"13f_holders:{ticker.upper()}"
        cached = _cache_get(cache_key, _TTL_SIGNALS)
        if cached is not None:
            return cached

        params = {
            "q": f'"{ticker.upper()}"',
            "forms": "13F-HR",
            "dateRange": "custom",
            "startdt": (date.today() - timedelta(days=180)).isoformat(),
            "enddt": date.today().isoformat(),
            "_source": "display_names,entity_id,file_date,accession_no",
            "from": "0",
            "size": "50",
        }

        async with httpx.AsyncClient() as client:
            data = await _get_json(client, _EFTS_BASE, params=params)

        if not data:
            return []

        holders: list[dict] = []
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            names = src.get("display_names", [])
            holders.append({
                "name":        names[0] if names else "Unknown",
                "crd":         str(src.get("entity_id", "")),
                "file_date":   src.get("file_date", ""),
                "accession":   src.get("accession_no", ""),
            })

        _cache_set(cache_key, holders)
        return holders

    def _classify_holder(self, holder_name: str) -> str:
        """
        Classify a holder as institutional or retail based on name patterns.
        """
        name_low = holder_name.lower()
        institutional_patterns = [
            "capital management", "asset management", "investment management",
            "fund management", "advisors", "advisers", "partners", "capital llc",
            "capital lp", "investments llc", "investments lp", "portfolio management",
            "pension", "endowment", "foundation", "insurance", "trust company",
            "bank", "securities", "financial group", "wealth management",
            "hedge fund", "private equity", "sovereign", "family office",
        ]
        retail_patterns = [
            "individual", "retail", "personal", "family trust", "self-directed",
        ]

        for pat in retail_patterns:
            if pat in name_low:
                return "retail"
        for pat in institutional_patterns:
            if pat in name_low:
                return "institutional"
        return "institutional"  # default unknown to institutional (13F filers are institutions)

    async def institutional_bias_score(self, ticker: str) -> float:
        """
        Compute institutional bias score 0–100.
        100 = entirely institutional, 0 = entirely retail.
        """
        holders = await self.get_13f_holders(ticker)
        if not holders:
            return 50.0  # neutral if no data

        inst_count   = sum(1 for h in holders if self._classify_holder(h["name"]) == "institutional")
        retail_count = len(holders) - inst_count

        total = len(holders)
        if total == 0:
            return 50.0

        return round(inst_count / total * 100, 2)

    async def retail_vs_institutional(self, ticker: str) -> ETFOwnershipProfile:
        """
        Full ownership profile: counts, ratio, top institutional owners.
        """
        holders = await self.get_13f_holders(ticker)

        inst_holders   = [h for h in holders if self._classify_holder(h["name"]) == "institutional"]
        retail_holders = [h for h in holders if self._classify_holder(h["name"]) == "retail"]

        total = len(holders)
        bias_score = round(len(inst_holders) / total * 100, 2) if total else 50.0

        ratio = (
            len(inst_holders) / len(retail_holders)
            if retail_holders
            else float(len(inst_holders))
        )

        return ETFOwnershipProfile(
            ticker=ticker.upper(),
            institutional_bias_score=bias_score,
            institutional_count=len(inst_holders),
            retail_count=len(retail_holders),
            retail_vs_institutional_ratio=round(ratio, 2),
            top_institutional_owners=inst_holders[:10],
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _clean_date(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    # Try to parse and normalise to YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return s[:10] if s else None


# ---------------------------------------------------------------------------
# 6. FastAPI router
# ---------------------------------------------------------------------------

ria_router = APIRouter(prefix="/ria", tags=["RIA / Form ADV"])

# Module-level singletons
_iapd    = IAPDAdapter()
_universe = RIAUniverse()
_signals  = AdviserSignalEngine()
_ownership = ETFOwnershipViaRIA()


@ria_router.get("/top")
async def get_top_rias(n: int = Query(50, ge=1, le=500)) -> dict:
    """
    Return top n RIAs by AUM from the cached universe.
    Triggers a background refresh if the cache is stale.
    """
    if _universe._is_stale():
        asyncio.create_task(_universe.refresh())

    records = _universe.get_top_by_aum(n)
    return {
        "count": len(records),
        "firms": [r.model_dump() for r in records],
    }


@ria_router.get("/search")
async def search_rias(
    q: str = Query(..., min_length=2, description="Firm name search"),
    limit: int = Query(25, ge=1, le=100),
) -> dict:
    """
    Search RIA firms by name. Searches cached DB first, falls back to IAPD API.
    """
    # Local DB first
    records = _universe.search(q)
    if records:
        return {"count": len(records), "firms": [r.model_dump() for r in records]}

    # Fall back to live IAPD search
    raw = await _iapd.search_firms(q, limit=limit)
    return {"count": len(raw), "firms": raw, "source": "iapd_live"}


@ria_router.get("/adviser/{crd}")
async def get_adviser(crd: str) -> dict:
    """
    Full adviser profile for a given CRD number.
    Combines IAPD firm detail with cached universe data.
    """
    # Try DB first
    record = _universe.get_by_crd(crd)
    if record and record.cached_at and (time.time() - record.cached_at) < _TTL_FIRM_DETAIL:
        return record.model_dump()

    # Fetch fresh
    detail = await _iapd.get_firm_detail(crd)
    if not detail or detail.get("error"):
        raise HTTPException(status_code=404, detail=f"Adviser CRD {crd} not found")

    return detail


@ria_router.get("/state-breakdown")
async def get_state_breakdown() -> dict:
    """
    Return RIA registration counts grouped by state.
    """
    breakdown = _universe.get_state_breakdown()
    return {
        "total_states": len(breakdown),
        "breakdown": breakdown,
    }


@ria_router.get("/new-launches")
async def get_new_launches(
    days_back: int = Query(90, ge=1, le=365, description="Days to look back for new filings"),
) -> dict:
    """
    List recently registered RIAs (potential new manager launches).
    """
    launches = await _signals.get_new_launches(days_back=days_back)
    return {
        "count": len(launches),
        "days_back": days_back,
        "launches": [r.model_dump() for r in launches],
    }


@ria_router.get("/signals/{crd}")
async def get_adviser_signals(crd: str) -> dict:
    """
    Compute all investment signals for an adviser: AUM growth, compensation
    alignment, client concentration, disciplinary flags.
    """
    growth, concentration, alignment, disciplinary = await asyncio.gather(
        _signals.aum_growth_yoy(crd),
        _signals.client_concentration(crd),
        _signals.compensation_alignment_score(crd),
        _signals.disciplinary_flag(crd),
    )
    return {
        "crd": crd,
        "aum_growth":       growth.model_dump(),
        "concentration":    concentration.model_dump(),
        "alignment":        alignment.model_dump(),
        "has_disciplinary": disciplinary,
    }


@ria_router.get("/etf-ownership/{ticker}")
async def get_etf_ownership(ticker: str) -> dict:
    """
    Institutional vs retail ownership profile for an ETF/stock via 13F data.
    """
    profile = await _ownership.retail_vs_institutional(ticker.upper())
    return profile.model_dump()


@ria_router.get("/aum-distribution")
async def get_aum_distribution() -> dict:
    """
    AUM bucket distribution across the RIA universe.
    """
    dist = _universe.get_aum_distribution()
    total = sum(dist.values())
    return {
        "total_firms": total,
        "distribution": dist,
    }
