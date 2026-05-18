"""
Form D Screener V3 — Full Regulation D Intelligence Engine.

dim_031: Upgrades private placement intelligence 7 → 9 over form_d.py by adding:
  - RegDExemptionAnalyzer: full Rule 504/506(b)/506(c)/RegA/RegCF analysis
  - OfferingTimelineTracker: amendment lifecycle, round closure detection
  - VCEcosystemMapper: state rankings, hub cities, emerging markets
  - FormDScreenerEnhanced: stealth mode, repeat issuers, crowdfunding, sophistication
  - PostIPOTracker: Form D → IPO conversion, pre-IPO fundraising history
  - FormDIntelligenceEngine: orchestrator with market overview, company analysis

FastAPI router: form_d_screener_v3_router
  GET  /formd/v3/overview
  GET  /formd/v3/company/{name}
  GET  /formd/v3/ecosystem/{state}
  POST /formd/v3/screen
  GET  /formd/v3/ipo-pipeline

Usage::

    from sentinel.sfe.form_d_screener_v3 import FormDIntelligenceEngine
    engine = FormDIntelligenceEngine()
    overview = engine.get_market_overview(days=30)
    print(overview)
    pipeline = engine.track_ipo_pipeline()
    print(pipeline.head(20))
"""
from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
_EDGAR_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_HEADERS = {
    "User-Agent": "SENTINEL/3.0 research@sentinel.ai",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 20.0
_NS = "http://www.sec.gov/xmlschema/formd"

# Rule exemptions
_EXEMPTION_RULES: Dict[str, Dict[str, Any]] = {
    "Rule 504": {
        "code": "504",
        "max_amount": 10_000_000,
        "accredited_required": False,
        "general_solicitation": False,
        "description": "Reg D Rule 504 — up to $10M, no accredited investor requirement",
    },
    "Rule 506(b)": {
        "code": "06b",
        "max_amount": None,
        "accredited_required": True,
        "max_non_accredited": 35,
        "general_solicitation": False,
        "description": "Reg D Rule 506(b) — unlimited, up to 35 non-accredited, no general solicitation",
    },
    "Rule 506(c)": {
        "code": "06c",
        "max_amount": None,
        "accredited_required": True,
        "max_non_accredited": 0,
        "general_solicitation": True,
        "description": "Reg D Rule 506(c) — unlimited, accredited only, general solicitation allowed",
    },
    "Regulation A (Tier 1)": {
        "code": "RegA1",
        "max_amount": 20_000_000,
        "accredited_required": False,
        "general_solicitation": True,
        "description": "Regulation A Tier 1 — up to $20M mini-IPO",
    },
    "Regulation A (Tier 2)": {
        "code": "RegA2",
        "max_amount": 75_000_000,
        "accredited_required": False,
        "general_solicitation": True,
        "description": "Regulation A Tier 2 — up to $75M mini-IPO",
    },
    "Regulation CF": {
        "code": "RegCF",
        "max_amount": 5_000_000,
        "accredited_required": False,
        "general_solicitation": True,
        "description": "Regulation Crowdfunding — up to $5M, retail investors",
    },
    "Section 4(a)(2)": {
        "code": "4a2",
        "max_amount": None,
        "accredited_required": True,
        "general_solicitation": False,
        "description": "Section 4(a)(2) — private placement catch-all",
    },
}

_STATE_ABBREVS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

_VC_HUB_ZIPS: Dict[str, str] = {
    "941": "San Francisco Bay Area",
    "940": "San Francisco Bay Area",
    "945": "San Francisco Bay Area",
    "100": "New York City",
    "101": "New York City",
    "102": "New York City",
    "021": "Boston",
    "022": "Boston",
    "787": "Austin",
    "900": "Los Angeles",
    "901": "Los Angeles",
    "980": "Seattle",
    "981": "Seattle",
    "606": "Chicago",
    "303": "Atlanta",
    "850": "Phoenix",
    "801": "Denver",
    "331": "Miami",
    "770": "Houston",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FormDFiling:
    company_name: str
    cik: str
    file_date: str
    accession_no: str
    period_of_report: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    zip_code: Optional[str] = None
    total_offering_amount: Optional[float] = None
    amount_sold: Optional[float] = None
    offering_type: str = "Unknown"
    fund_type: Optional[str] = None
    industry: Optional[str] = None
    exemption_claimed: Optional[str] = None   # raw code from XML
    exemption_normalized: Optional[str] = None # "506b", "506c", "504", etc.
    is_amendment: bool = False
    num_accredited_investors: Optional[int] = None
    num_non_accredited_investors: Optional[int] = None
    date_of_first_sale: Optional[str] = None
    more_to_come: Optional[bool] = None
    key_persons: List[str] = field(default_factory=list)
    related_persons: List[Dict[str, str]] = field(default_factory=list)
    filing_url: str = ""
    form_type: str = "D"


@dataclass
class ExemptionAnalysis:
    rule_claimed: str
    rule_normalized: str
    is_valid: bool
    flags: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    investor_type_classification: str = "UNKNOWN"
    suitability_notes: str = ""


@dataclass
class OfferingTimeline:
    company_name: str
    cik: str
    filings: List[FormDFiling] = field(default_factory=list)
    first_file_date: Optional[str] = None
    last_file_date: Optional[str] = None
    total_amendments: int = 0
    max_offering_amount: Optional[float] = None
    max_amount_sold: Optional[float] = None
    is_closed: bool = False
    duration_days: Optional[int] = None
    amendment_deltas: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EcosystemData:
    state: str
    period_days: int
    total_deals: int
    total_capital: float
    top_exemptions: Dict[str, int] = field(default_factory=dict)
    top_industries: Dict[str, int] = field(default_factory=dict)
    top_investors_types: Dict[str, int] = field(default_factory=dict)
    hub_cities: List[Dict[str, Any]] = field(default_factory=list)
    avg_deal_size: float = 0.0
    median_deal_size: float = 0.0
    yoy_growth_pct: Optional[float] = None


@dataclass
class IPOConversion:
    company_name: str
    cik: str
    first_form_d_date: str
    s1_filing_date: Optional[str] = None
    ipo_date: Optional[str] = None
    days_to_ipo: Optional[int] = None
    ticker: Optional[str] = None
    last_private_round_amount: Optional[float] = None
    s1_accession: Optional[str] = None


@dataclass
class MarketOverview:
    period_days: int
    as_of_date: str
    total_deals: int
    total_capital: float
    new_deals_506b: int
    new_deals_506c: int
    new_deals_504: int
    new_deals_other: int
    top_states: List[Tuple[str, int]] = field(default_factory=list)
    top_industries: List[Tuple[str, int]] = field(default_factory=list)
    avg_deal_size: float = 0.0
    largest_deal: Optional[Dict[str, Any]] = None
    crowdfunding_deals: int = 0


@dataclass
class CompanyAnalysis:
    company_name: str
    cik: str
    total_filings: int
    total_rounds: int
    total_raised: float
    exemptions_used: List[str] = field(default_factory=list)
    timeline: Optional[OfferingTimeline] = None
    latest_exemption_analysis: Optional[ExemptionAnalysis] = None
    ipo_conversion: Optional[IPOConversion] = None
    risk_score: float = 0.0


# ---------------------------------------------------------------------------
# EDGAR Form D fetcher
# ---------------------------------------------------------------------------

class _FormDFetcher:
    """Fetch Form D filings from EDGAR EFTS full-text search."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        try:
            resp = self._session.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("FormD fetch [%s]: %s", url, exc)
            return None

    def _get_text(self, url: str) -> str:
        try:
            resp = self._session.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            logger.warning("FormD text fetch [%s]: %s", url, exc)
            return ""

    def search_recent(
        self,
        days: int = 30,
        state: Optional[str] = None,
        form_types: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[FormDFiling]:
        """Search EDGAR EFTS for recent Form D filings."""
        if form_types is None:
            form_types = ["D", "D/A"]

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        params: Dict[str, Any] = {
            "q": "",
            "dateRange": "custom",
            "startdt": start_date.isoformat(),
            "enddt": end_date.isoformat(),
            "forms": ",".join(form_types),
            "_source": (
                "period_of_report,display_names,entity_id,"
                "file_date,form_type,biz_location,accession_no,file_num"
            ),
            "hits.hits._source": "true",
            "hits.hits.total.value": "true",
            "hits.hits.hits": str(limit),
        }
        if state:
            params["locationCode"] = state

        # Use EDGAR full-text search
        search_url = "https://efts.sec.gov/LATEST/search-index"
        data = self._get(search_url, params=params)

        # Fallback: EDGAR EFTS REST endpoint
        if not data:
            data = self._search_efts_fallback(start_date, end_date, state, limit)

        return self._parse_efts_results(data or {})

    def _search_efts_fallback(
        self,
        start_date: date,
        end_date: date,
        state: Optional[str],
        limit: int,
    ) -> Dict:
        """EDGAR EFTS v2 search fallback."""
        params: Dict[str, Any] = {
            "q": "",
            "dateRange": "custom",
            "startdt": start_date.isoformat(),
            "enddt": end_date.isoformat(),
            "forms": "D,D/A",
            "hits.hits._source": "true",
        }
        if state:
            params["locationCode"] = state

        url = "https://efts.sec.gov/LATEST/search-index"
        return self._get(url, params) or {}

    def _parse_efts_results(self, data: Dict) -> List[FormDFiling]:
        hits = data.get("hits", {}).get("hits", [])
        results: List[FormDFiling] = []

        for hit in hits:
            src = hit.get("_source", {})
            display = src.get("display_names", [{}])
            name = display[0].get("name", "Unknown") if display else "Unknown"
            cik = str(display[0].get("id", "")) if display else ""
            accession = src.get("accession_no", "")
            form_type = src.get("form_type", "D")

            biz_loc = src.get("biz_location", {})
            state = biz_loc.get("stateOrCountry", "") if isinstance(biz_loc, dict) else ""
            city = biz_loc.get("city", "") if isinstance(biz_loc, dict) else ""

            filing = FormDFiling(
                company_name=name,
                cik=cik,
                file_date=src.get("file_date", ""),
                accession_no=accession,
                period_of_report=src.get("period_of_report"),
                state=state or None,
                city=city or None,
                form_type=form_type,
                is_amendment=(form_type == "D/A"),
                filing_url=self._build_filing_url(cik, accession),
            )
            results.append(filing)

        return results

    def fetch_form_d_detail(self, cik: str, accession: str) -> Optional[FormDFiling]:
        """Fetch and parse the Form D XML for detailed data."""
        acc_clean = accession.replace("-", "")
        xml_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik) if cik.isdigit() else cik}/"
            f"{acc_clean}/primary_doc.xml"
        )
        xml_text = self._get_text(xml_url)
        if not xml_text:
            # Try the index to find the XML file
            index_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik) if cik.isdigit() else cik}/"
                f"{acc_clean}/{accession}-index.htm"
            )
            index_text = self._get_text(index_url)
            xml_links = re.findall(
                r'href="(/Archives/edgar/data/[^"]+\.xml)"', index_text
            )
            if xml_links:
                xml_text = self._get_text(f"https://www.sec.gov{xml_links[0]}")

        if not xml_text:
            return None

        return self._parse_form_d_xml(xml_text, cik, accession)

    def _parse_form_d_xml(
        self, xml_text: str, cik: str, accession: str
    ) -> Optional[FormDFiling]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("XML parse error for %s: %s", accession, exc)
            return None

        def _find(tag: str) -> Optional[str]:
            el = root.find(f".//{{{_NS}}}{tag}")
            return el.text.strip() if el is not None and el.text else None

        def _find_float(tag: str) -> Optional[float]:
            val = _find(tag)
            if val:
                try:
                    return float(val.replace(",", ""))
                except ValueError:
                    pass
            return None

        def _find_int(tag: str) -> Optional[int]:
            val = _find(tag)
            if val:
                try:
                    return int(val.replace(",", ""))
                except ValueError:
                    pass
            return None

        # Company info
        company_name = _find("entityName") or "Unknown"
        state = _find("stateOrCountry") or _find("issuerStateOrCountry")
        city = _find("city") or _find("issuerCity")
        zip_code = _find("zipCode") or _find("issuerZipCode")
        file_date = _find("submissionDate") or _find("dateOfEarliestSale") or ""

        # Offering details
        total_offering = _find_float("totalOfferingAmount")
        amount_sold = _find_float("totalAmountSold")
        date_first_sale = _find("dateOfFirstSale")

        # Exemptions
        exemption_el = root.find(f".//{{{_NS}}}exemptionClaimed")
        exemption_code = None
        if exemption_el is not None:
            exemption_code = exemption_el.text.strip() if exemption_el.text else None

        # Investors
        num_accredited = _find_int("numberAccreditedInvestors") or _find_int("numberOfPreviousInvestors")
        num_non_accredited = _find_int("numberOfNonAccreditedInvestors")

        # More to come
        more_text = _find("isAmendment")
        is_amendment = more_text and more_text.lower() == "true"

        more_to_come_text = _find("moreToCome")
        more_to_come = more_to_come_text and more_to_come_text.lower() == "true"

        # Related persons
        related_persons: List[Dict[str, str]] = []
        for rp in root.findall(f".//{{{_NS}}}relatedPersonInfo"):
            first = rp.find(f"{{{_NS}}}relatedPersonFirstName")
            last = rp.find(f"{{{_NS}}}relatedPersonLastName")
            rel = rp.find(f"{{{_NS}}}relatedPersonRelationshipList")
            if first is not None and last is not None:
                name = f"{first.text or ''} {last.text or ''}".strip()
                relationship = rel.text if rel is not None else ""
                related_persons.append({"name": name, "relationship": relationship})

        # Industry / fund type
        industry = _find("industryGroupType") or _find("industryGroup")
        fund_type = _find("typeOfFiling")
        offering_type = _find("offeringType") or "Equity"

        exemption_normalized = self._normalize_exemption(exemption_code or "")

        return FormDFiling(
            company_name=company_name,
            cik=cik,
            file_date=file_date,
            accession_no=accession,
            state=state,
            city=city,
            zip_code=zip_code,
            total_offering_amount=total_offering,
            amount_sold=amount_sold,
            offering_type=offering_type or "Equity",
            fund_type=fund_type,
            industry=industry,
            exemption_claimed=exemption_code,
            exemption_normalized=exemption_normalized,
            is_amendment=bool(is_amendment),
            num_accredited_investors=num_accredited,
            num_non_accredited_investors=num_non_accredited,
            date_of_first_sale=date_first_sale,
            more_to_come=bool(more_to_come) if more_to_come is not None else None,
            related_persons=related_persons,
            filing_url=self._build_filing_url(cik, accession),
        )

    @staticmethod
    def _normalize_exemption(code: str) -> str:
        code_lower = code.lower().replace(" ", "")
        mapping = {
            "06b": "506b", "506b": "506b", "rule506b": "506b",
            "06c": "506c", "506c": "506c", "rule506c": "506c",
            "04": "504", "504": "504", "rule504": "504",
            "rega1": "rega_tier1", "regat1": "rega_tier1",
            "rega2": "rega_tier2", "regat2": "rega_tier2",
            "regcf": "regcf", "cf": "regcf",
            "4a2": "4a2", "4(a)(2)": "4a2", "section4a2": "4a2",
        }
        for pattern, normalized in mapping.items():
            if pattern in code_lower:
                return normalized
        return code or "unknown"

    @staticmethod
    def _build_filing_url(cik: str, accession: str) -> str:
        if not cik or not accession:
            return ""
        acc_clean = accession.replace("-", "")
        try:
            cik_int = int(cik)
        except ValueError:
            cik_int = 0
        return (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
            f"{acc_clean}/{accession}-index.htm"
        )

    def search_by_company(
        self, company_name: str, include_amendments: bool = True
    ) -> List[FormDFiling]:
        """Search for all Form D filings by a specific company name."""
        form_types = ["D", "D/A"] if include_amendments else ["D"]
        params: Dict[str, Any] = {
            "q": f'"{company_name}"',
            "forms": ",".join(form_types),
            "hits.hits._source": "true",
        }
        data = self._get("https://efts.sec.gov/LATEST/search-index", params) or {}
        results = self._parse_efts_results(data)
        # Filter exact name match
        name_lower = company_name.lower()
        return [
            r for r in results if name_lower in r.company_name.lower()
        ]

    def search_s1_filings(self, company_name: str) -> List[Dict[str, Any]]:
        """Search for S-1 / S-11 registration statements by company name."""
        params: Dict[str, Any] = {
            "q": f'"{company_name}"',
            "forms": "S-1,S-11,S-1/A",
            "hits.hits._source": "true",
        }
        data = self._get("https://efts.sec.gov/LATEST/search-index", params) or {}
        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            src = hit.get("_source", {})
            display = src.get("display_names", [{}])
            name = display[0].get("name", "") if display else ""
            results.append(
                {
                    "company_name": name,
                    "form_type": src.get("form_type", ""),
                    "file_date": src.get("file_date", ""),
                    "accession_no": src.get("accession_no", ""),
                    "cik": str(display[0].get("id", "")) if display else "",
                }
            )
        return results


# ---------------------------------------------------------------------------
# RegDExemptionAnalyzer
# ---------------------------------------------------------------------------

class RegDExemptionAnalyzer:
    """
    Analyze Regulation D exemption claims: validity, suitability, risk flags.
    """

    def analyze_exemption(self, filing: FormDFiling) -> ExemptionAnalysis:
        rule = filing.exemption_normalized or "unknown"
        flags: List[str] = []
        risk = 0.0

        amount = filing.total_offering_amount or 0.0
        num_non_acc = filing.num_non_accredited_investors or 0
        num_acc = filing.num_accredited_investors or 0

        # --- Rule-specific validation ---
        if rule == "504":
            if amount > 10_000_000:
                flags.append(
                    f"VIOLATION: Amount ${amount:,.0f} exceeds $10M Rule 504 cap"
                )
                risk += 0.6

        elif rule == "506b":
            if num_non_acc > 35:
                flags.append(
                    f"VIOLATION: {num_non_acc} non-accredited investors exceeds 506(b) limit of 35"
                )
                risk += 0.7
            if num_non_acc > 0:
                flags.append(f"INFO: {num_non_acc} non-accredited investors (max 35 allowed under 506b)")

        elif rule == "506c":
            if num_non_acc > 0:
                flags.append(
                    f"SUSPICIOUS: 506(c) requires ALL accredited investors, "
                    f"but {num_non_acc} non-accredited listed"
                )
                risk += 0.8

        elif rule in ("rega_tier1",):
            if amount > 20_000_000:
                flags.append(
                    f"VIOLATION: Regulation A Tier 1 limit is $20M, amount is ${amount:,.0f}"
                )
                risk += 0.5

        elif rule in ("rega_tier2",):
            if amount > 75_000_000:
                flags.append(
                    f"VIOLATION: Regulation A Tier 2 limit is $75M, amount is ${amount:,.0f}"
                )
                risk += 0.5

        elif rule == "regcf":
            if amount > 5_000_000:
                flags.append(
                    f"VIOLATION: Regulation CF limit is $5M, amount is ${amount:,.0f}"
                )
                risk += 0.6

        # --- General risk factors ---
        if not filing.related_persons:
            flags.append("WARNING: No key persons disclosed")
            risk += 0.15

        if filing.industry is None:
            flags.append("INFO: No industry type disclosed")
            risk += 0.05

        if filing.date_of_first_sale and filing.file_date:
            try:
                sale_dt = datetime.fromisoformat(filing.date_of_first_sale)
                file_dt = datetime.fromisoformat(filing.file_date)
                lag = (file_dt - sale_dt).days
                if lag > 15:
                    flags.append(
                        f"WARNING: Filing {lag} days after first sale (>15-day deadline)"
                    )
                    risk += 0.2
            except ValueError:
                pass

        if filing.fund_type and filing.fund_type.lower() in ("hedge fund", "private equity fund"):
            risk += 0.1  # Fund filings are higher complexity

        is_valid = risk < 0.6
        investor_type = self.classify_investor_type(filing)

        suitability = self._suitability_note(rule, amount, num_acc, num_non_acc)

        return ExemptionAnalysis(
            rule_claimed=filing.exemption_claimed or rule,
            rule_normalized=rule,
            is_valid=is_valid,
            flags=flags,
            risk_score=min(risk, 1.0),
            investor_type_classification=investor_type,
            suitability_notes=suitability,
        )

    def classify_investor_type(self, filing: FormDFiling) -> str:
        """Classify the likely investor base from filing characteristics."""
        fund_type = (filing.fund_type or "").lower()
        industry = (filing.industry or "").lower()
        exemption = filing.exemption_normalized or ""
        amount = filing.total_offering_amount or 0

        if exemption == "regcf":
            return "CROWDFUNDING"

        if exemption in ("rega_tier1", "rega_tier2"):
            return "RETAIL_MINI_IPO"

        if "venture" in fund_type or "vc" in fund_type:
            return "INSTITUTIONAL_VC"

        if "hedge" in fund_type:
            return "HEDGE_FUND"

        if "private equity" in fund_type or "pe fund" in fund_type:
            return "PE"

        if "family" in fund_type:
            return "FAMILY_OFFICE"

        if amount < 1_000_000 and exemption == "506b":
            return "ANGEL"

        if amount >= 5_000_000 and exemption == "506c":
            return "INSTITUTIONAL_VC"

        if amount >= 1_000_000 and exemption == "506b":
            return "ANGEL"

        if "real estate" in fund_type or "real estate" in industry:
            return "REAL_ESTATE"

        return "STARTUP_EQUITY"

    def compute_offering_risk_score(self, filing: FormDFiling) -> float:
        """
        0–1 composite risk score based on filing characteristics.
        Higher = more risk flags / unusual filing.
        """
        risk = 0.0
        amount = filing.total_offering_amount or 0

        # First-time issuer signal (no related person history available here)
        if not filing.related_persons:
            risk += 0.15

        # Large offering with no amount sold
        if amount > 10_000_000 and (filing.amount_sold is None or filing.amount_sold == 0):
            risk += 0.1

        # No date of first sale
        if not filing.date_of_first_sale:
            risk += 0.05

        # Late filing
        if filing.file_date and filing.date_of_first_sale:
            try:
                lag = (
                    datetime.fromisoformat(filing.file_date)
                    - datetime.fromisoformat(filing.date_of_first_sale)
                ).days
                if lag > 15:
                    risk += min(lag / 100, 0.25)
            except ValueError:
                pass

        # Exemption mismatch
        analysis = self.analyze_exemption(filing)
        risk += analysis.risk_score * 0.5

        return min(risk, 1.0)

    @staticmethod
    def _suitability_note(
        rule: str,
        amount: float,
        num_acc: int,
        num_non_acc: int,
    ) -> str:
        notes = {
            "504": f"Rule 504: appropriate for amounts up to $10M. No accredited investor requirement.",
            "506b": (
                f"Rule 506(b): appropriate for unlimited raises. "
                f"Allows up to 35 non-accredited investors but no general solicitation."
            ),
            "506c": (
                f"Rule 506(c): appropriate for unlimited raises. "
                f"Requires ALL investors to be accredited. Allows general solicitation."
            ),
            "rega_tier1": f"Reg A Tier 1: mini-IPO up to $20M, open to all investors.",
            "rega_tier2": f"Reg A Tier 2: mini-IPO up to $75M, open to all investors.",
            "regcf": f"Regulation CF: crowdfunding, up to $5M, all investors eligible.",
            "4a2": f"Section 4(a)(2): private placement catch-all, sophisticated investors.",
        }
        return notes.get(rule, f"Exemption rule '{rule}' — refer to SEC regulations.")


# ---------------------------------------------------------------------------
# OfferingTimelineTracker
# ---------------------------------------------------------------------------

class OfferingTimelineTracker:
    """
    Track the full lifecycle of Form D offerings from initial to close.
    """

    def __init__(self, fetcher: Optional[_FormDFetcher] = None):
        self.fetcher = fetcher or _FormDFetcher()

    def build_offering_timeline(self, company_name: str) -> OfferingTimeline:
        """Fetch all Form D + D/A filings and build lifecycle timeline."""
        filings = self.fetcher.search_by_company(company_name, include_amendments=True)

        if not filings:
            return OfferingTimeline(company_name=company_name, cik="")

        # Sort by file date
        filings_sorted = sorted(
            filings,
            key=lambda f: f.file_date or "1900-01-01",
        )
        cik = filings_sorted[0].cik

        timeline = OfferingTimeline(
            company_name=company_name,
            cik=cik,
            filings=filings_sorted,
            first_file_date=filings_sorted[0].file_date,
            last_file_date=filings_sorted[-1].file_date,
            total_amendments=sum(1 for f in filings_sorted if f.is_amendment),
        )

        # Aggregate offering amounts
        amounts = [
            f.total_offering_amount
            for f in filings_sorted
            if f.total_offering_amount
        ]
        sold = [
            f.amount_sold for f in filings_sorted if f.amount_sold
        ]
        timeline.max_offering_amount = max(amounts) if amounts else None
        timeline.max_amount_sold = max(sold) if sold else None

        # Duration
        if timeline.first_file_date and timeline.last_file_date:
            try:
                d1 = datetime.fromisoformat(timeline.first_file_date)
                d2 = datetime.fromisoformat(timeline.last_file_date)
                timeline.duration_days = (d2 - d1).days
            except ValueError:
                pass

        # Amendment deltas
        timeline.amendment_deltas = []
        for i in range(1, len(filings_sorted)):
            delta = self.detect_amendment_changes(filings_sorted[i - 1], filings_sorted[i])
            if delta:
                timeline.amendment_deltas.append(
                    {
                        "from_date": filings_sorted[i - 1].file_date,
                        "to_date": filings_sorted[i].file_date,
                        "changes": delta,
                    }
                )

        timeline.is_closed = self.detect_round_closure(timeline)
        return timeline

    def detect_amendment_changes(
        self,
        filing1: FormDFiling,
        filing2: FormDFiling,
    ) -> List[str]:
        changes: List[str] = []

        # Amount raised
        if filing1.total_offering_amount != filing2.total_offering_amount:
            old = filing1.total_offering_amount or 0
            new = filing2.total_offering_amount or 0
            changes.append(
                f"Offering amount changed: ${old:,.0f} → ${new:,.0f}"
            )

        # Amount sold
        if filing1.amount_sold != filing2.amount_sold:
            old = filing1.amount_sold or 0
            new = filing2.amount_sold or 0
            changes.append(f"Amount sold changed: ${old:,.0f} → ${new:,.0f}")

        # Investor count
        if filing1.num_accredited_investors != filing2.num_accredited_investors:
            changes.append(
                f"Accredited investors changed: "
                f"{filing1.num_accredited_investors} → {filing2.num_accredited_investors}"
            )

        # Exemption change
        if filing1.exemption_normalized != filing2.exemption_normalized:
            changes.append(
                f"Exemption changed: {filing1.exemption_normalized} → {filing2.exemption_normalized}"
            )

        # More to come flag
        if filing1.more_to_come != filing2.more_to_come:
            changes.append(
                f"More-to-come flag changed: {filing1.more_to_come} → {filing2.more_to_come}"
            )

        return changes

    def compute_capital_raise_duration(self, timeline: OfferingTimeline) -> int:
        """Days from first file date to last amendment."""
        return timeline.duration_days or 0

    def detect_round_closure(self, timeline: OfferingTimeline) -> bool:
        """Check if last amendment signals round closure."""
        if not timeline.filings:
            return False
        last = timeline.filings[-1]
        if last.more_to_come is False:
            return True
        if (
            last.total_offering_amount
            and last.amount_sold
            and last.amount_sold >= last.total_offering_amount * 0.95
        ):
            return True
        return False

    def compute_total_raised(self, company_name: str) -> float:
        """Sum raised across all offerings for a company."""
        filings = self.fetcher.search_by_company(company_name)
        total = 0.0
        seen_offerings: set = set()
        for filing in filings:
            # Deduplicate: don't double-count amendments
            if not filing.is_amendment and filing.accession_no not in seen_offerings:
                if filing.amount_sold:
                    total += filing.amount_sold
                elif filing.total_offering_amount:
                    total += filing.total_offering_amount
                seen_offerings.add(filing.accession_no)
        return total


# ---------------------------------------------------------------------------
# VCEcosystemMapper
# ---------------------------------------------------------------------------

class VCEcosystemMapper:
    """
    Map VC/PE ecosystem from Form D data: state rankings, hub cities, trends.
    """

    def __init__(self, fetcher: Optional[_FormDFetcher] = None):
        self.fetcher = fetcher or _FormDFetcher()

    def get_ecosystem_by_state(
        self, state: str, days: int = 365
    ) -> EcosystemData:
        """Fetch all Form D filings for a state and aggregate ecosystem metrics."""
        filings = self.fetcher.search_recent(days=days, state=state, limit=500)

        if not filings:
            return EcosystemData(
                state=state, period_days=days, total_deals=0, total_capital=0.0
            )

        total_capital = sum(
            f.total_offering_amount for f in filings if f.total_offering_amount
        )
        exemption_counts: Counter = Counter()
        industry_counts: Counter = Counter()
        investor_type_counts: Counter = Counter()
        city_counts: Counter = Counter()
        amounts: List[float] = [
            f.total_offering_amount for f in filings if f.total_offering_amount
        ]

        analyzer = RegDExemptionAnalyzer()

        for filing in filings:
            if filing.exemption_normalized:
                exemption_counts[filing.exemption_normalized] += 1
            if filing.industry:
                industry_counts[filing.industry] += 1
            investor_type_counts[analyzer.classify_investor_type(filing)] += 1
            if filing.city:
                city_counts[filing.city.title()] += 1

        hub_cities: List[Dict[str, Any]] = []
        for city, count in city_counts.most_common(10):
            hub_cities.append({"city": city, "deals": count})

        return EcosystemData(
            state=state,
            period_days=days,
            total_deals=len(filings),
            total_capital=total_capital,
            top_exemptions=dict(exemption_counts.most_common(5)),
            top_industries=dict(industry_counts.most_common(10)),
            top_investors_types=dict(investor_type_counts.most_common(8)),
            hub_cities=hub_cities,
            avg_deal_size=float(np.mean(amounts)) if amounts else 0.0,
            median_deal_size=float(np.median(amounts)) if amounts else 0.0,
        )

    def get_hub_cities(self) -> pd.DataFrame:
        """Aggregate Form D ZIP codes to identify top VC hub cities."""
        filings = self.fetcher.search_recent(days=365, limit=1000)
        city_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"deals": 0, "capital": 0.0, "state": ""}
        )

        for filing in filings:
            zip3 = (filing.zip_code or "")[:3]
            hub = _VC_HUB_ZIPS.get(zip3)
            city_key = hub or filing.city or "Unknown"
            if city_key == "Unknown":
                continue
            city_data[city_key]["deals"] += 1
            city_data[city_key]["capital"] += filing.total_offering_amount or 0
            city_data[city_key]["state"] = filing.state or ""

        rows = [
            {
                "city": city,
                "total_deals": data["deals"],
                "total_capital_m": data["capital"] / 1e6,
                "state": data["state"],
            }
            for city, data in city_data.items()
            if data["deals"] >= 2
        ]
        if not rows:
            return pd.DataFrame(columns=["city", "total_deals", "total_capital_m", "state"])

        df = pd.DataFrame(rows).sort_values("total_deals", ascending=False)
        return df.reset_index(drop=True)

    def compute_state_rankings(self) -> pd.DataFrame:
        """Rank all states by VC activity (deal count + capital)."""
        filings = self.fetcher.search_recent(days=365, limit=1000)
        state_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"deals": 0, "capital": 0.0}
        )

        for filing in filings:
            st = filing.state or "Unknown"
            state_data[st]["deals"] += 1
            state_data[st]["capital"] += filing.total_offering_amount or 0

        rows = [
            {
                "state": st,
                "total_deals": data["deals"],
                "total_capital_m": data["capital"] / 1e6,
                "avg_deal_size_m": (data["capital"] / data["deals"] / 1e6)
                if data["deals"] > 0
                else 0,
            }
            for st, data in state_data.items()
            if st in _STATE_ABBREVS
        ]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("total_deals", ascending=False)
        df["rank"] = range(1, len(df) + 1)
        return df.reset_index(drop=True)

    def get_emerging_markets(self, yoy_growth_threshold: float = 0.20) -> List[str]:
        """
        Identify states with YoY deal count growth > threshold.
        Compares last 180 days vs prior 180 days.
        """
        recent = self.fetcher.search_recent(days=180, limit=500)
        prior_end = date.today() - timedelta(days=180)
        prior_start = prior_end - timedelta(days=180)

        # Fetch prior period via date range
        params: Dict[str, Any] = {
            "q": "",
            "forms": "D",
            "dateRange": "custom",
            "startdt": prior_start.isoformat(),
            "enddt": prior_end.isoformat(),
            "hits.hits._source": "true",
        }
        fetcher = _FormDFetcher()
        prior_data = fetcher._get("https://efts.sec.gov/LATEST/search-index", params) or {}
        prior_filings = fetcher._parse_efts_results(prior_data)

        recent_by_state: Counter = Counter(
            f.state for f in recent if f.state in _STATE_ABBREVS
        )
        prior_by_state: Counter = Counter(
            f.state for f in prior_filings if f.state in _STATE_ABBREVS
        )

        emerging: List[str] = []
        for state in _STATE_ABBREVS:
            recent_count = recent_by_state.get(state, 0)
            prior_count = prior_by_state.get(state, 1)  # avoid div by zero
            growth = (recent_count - prior_count) / prior_count
            if growth > yoy_growth_threshold and recent_count >= 5:
                emerging.append(state)

        return sorted(emerging)

    def compute_capital_concentration(self) -> Dict[str, float]:
        """What % of deals and capital are in top 3 states?"""
        filings = self.fetcher.search_recent(days=365, limit=500)
        state_deals: Counter = Counter()
        state_capital: Dict[str, float] = defaultdict(float)

        for filing in filings:
            if filing.state and filing.state in _STATE_ABBREVS:
                state_deals[filing.state] += 1
                state_capital[filing.state] += filing.total_offering_amount or 0

        total_deals = sum(state_deals.values())
        total_capital = sum(state_capital.values())

        top3_states = [s for s, _ in state_deals.most_common(3)]
        top3_deals = sum(state_deals[s] for s in top3_states)
        top3_capital = sum(state_capital[s] for s in top3_states)

        return {
            "top3_states": top3_states,
            "deal_concentration_pct": (top3_deals / total_deals * 100) if total_deals else 0,
            "capital_concentration_pct": (top3_capital / total_capital * 100) if total_capital else 0,
            "total_deals": total_deals,
            "total_capital": total_capital,
        }


# ---------------------------------------------------------------------------
# FormDScreenerEnhanced
# ---------------------------------------------------------------------------

class FormDScreenerEnhanced:
    """
    Advanced screener with stealth mode, repeat issuers, crowdfunding, sophistication filters.
    """

    def __init__(self, fetcher: Optional[_FormDFetcher] = None):
        self.fetcher = fetcher or _FormDFetcher()

    def _filings_to_df(self, filings: List[FormDFiling]) -> pd.DataFrame:
        if not filings:
            return pd.DataFrame()
        rows = []
        for f in filings:
            rows.append(
                {
                    "company_name": f.company_name,
                    "cik": f.cik,
                    "file_date": f.file_date,
                    "state": f.state,
                    "city": f.city,
                    "total_offering_amount": f.total_offering_amount,
                    "amount_sold": f.amount_sold,
                    "exemption": f.exemption_normalized,
                    "is_amendment": f.is_amendment,
                    "num_accredited": f.num_accredited_investors,
                    "num_non_accredited": f.num_non_accredited_investors,
                    "industry": f.industry,
                    "fund_type": f.fund_type,
                    "filing_url": f.filing_url,
                }
            )
        return pd.DataFrame(rows)

    def screen_stealth_mode(self, days: int = 30) -> pd.DataFrame:
        """
        506(b) new filings (no general solicitation) = private / stealth mode.
        These are higher signal because companies are NOT publicly advertising the raise.
        """
        filings = self.fetcher.search_recent(days=days, limit=500)
        stealth = [
            f
            for f in filings
            if f.exemption_normalized == "506b" and not f.is_amendment
        ]
        df = self._filings_to_df(stealth)
        if not df.empty:
            df = df.sort_values("total_offering_amount", ascending=False, na_position="last")
        return df.reset_index(drop=True) if not df.empty else df

    def screen_repeat_issuers(self, min_rounds: int = 3) -> pd.DataFrame:
        """Companies with 3+ Form D filings (initial, not amendments) = multi-round raisers."""
        filings = self.fetcher.search_recent(days=365 * 3, limit=1000)
        # Count by company (CIK), exclude amendments
        initial_filings = [f for f in filings if not f.is_amendment]
        cik_counts: Counter = Counter(f.cik for f in initial_filings)
        cik_info: Dict[str, str] = {f.cik: f.company_name for f in initial_filings}

        rows = [
            {
                "company_name": cik_info[cik],
                "cik": cik,
                "round_count": count,
                "stage": self._estimate_stage(count),
            }
            for cik, count in cik_counts.items()
            if count >= min_rounds
        ]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("round_count", ascending=False)
        return df.reset_index(drop=True)

    def screen_large_rounds(self, min_amount: float = 10_000_000) -> pd.DataFrame:
        """Large round screener."""
        filings = self.fetcher.search_recent(days=90, limit=500)
        large = [
            f
            for f in filings
            if (f.total_offering_amount or 0) >= min_amount and not f.is_amendment
        ]
        df = self._filings_to_df(large)
        if not df.empty:
            df = df.sort_values("total_offering_amount", ascending=False)
        return df.reset_index(drop=True) if not df.empty else df

    def screen_crowdfunding_active(self, days: int = 90) -> pd.DataFrame:
        """Regulation CF filings = retail-accessible investment tracking."""
        filings = self.fetcher.search_recent(days=days, limit=200)
        cf = [f for f in filings if f.exemption_normalized == "regcf"]
        df = self._filings_to_df(cf)
        if not df.empty:
            df = df.sort_values("file_date", ascending=False)
        return df.reset_index(drop=True) if not df.empty else df

    def screen_by_investor_sophistication(self) -> pd.DataFrame:
        """
        506(b): mixed investors → smaller deals, angel/early stage.
        506(c): accredited only → institutional-grade.
        """
        filings = self.fetcher.search_recent(days=90, limit=500)
        rows = []
        for f in filings:
            exemption = f.exemption_normalized
            if exemption not in ("506b", "506c"):
                continue
            sophistication = "INSTITUTIONAL" if exemption == "506c" else "ANGEL_MIXED"
            rows.append(
                {
                    "company_name": f.company_name,
                    "cik": f.cik,
                    "file_date": f.file_date,
                    "exemption": exemption,
                    "sophistication": sophistication,
                    "total_offering_amount": f.total_offering_amount,
                    "num_accredited": f.num_accredited_investors,
                    "num_non_accredited": f.num_non_accredited_investors,
                    "state": f.state,
                    "industry": f.industry,
                }
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values(
            ["sophistication", "total_offering_amount"],
            ascending=[True, False],
            na_position="last",
        )
        return df.reset_index(drop=True)

    def screen_rega_deals(self, days: int = 90) -> pd.DataFrame:
        """Regulation A (mini-IPO) filings — public-facing, could list on exchanges."""
        filings = self.fetcher.search_recent(days=days, limit=200)
        rega = [
            f
            for f in filings
            if f.exemption_normalized in ("rega_tier1", "rega_tier2")
        ]
        df = self._filings_to_df(rega)
        if not df.empty:
            df["tier"] = df["exemption"].map(
                {"rega_tier1": "Tier 1 (<$20M)", "rega_tier2": "Tier 2 (<$75M)"}
            )
            df = df.sort_values("total_offering_amount", ascending=False)
        return df.reset_index(drop=True) if not df.empty else df

    @staticmethod
    def _estimate_stage(round_count: int) -> str:
        if round_count == 1:
            return "Seed"
        elif round_count == 2:
            return "Series A"
        elif round_count == 3:
            return "Series B"
        elif round_count == 4:
            return "Series C"
        elif round_count == 5:
            return "Series D"
        else:
            return f"Series {chr(ord('D') + round_count - 4)}+"


# ---------------------------------------------------------------------------
# PostIPOTracker
# ---------------------------------------------------------------------------

class PostIPOTracker:
    """
    Track Form D issuers that subsequently filed S-1 / went public.
    """

    def __init__(self, fetcher: Optional[_FormDFetcher] = None):
        self.fetcher = fetcher or _FormDFetcher()

    def detect_ipo_conversion(self, company_name: str) -> Optional[IPOConversion]:
        """Search for S-1 filings by same company name."""
        s1_filings = self.fetcher.search_s1_filings(company_name)
        if not s1_filings:
            return None

        # Get Form D history
        form_d_filings = self.fetcher.search_by_company(company_name)
        if not form_d_filings:
            return None

        first_form_d = sorted(
            form_d_filings, key=lambda f: f.file_date or ""
        )[0]
        s1 = s1_filings[0]

        days_to_ipo = None
        if first_form_d.file_date and s1.get("file_date"):
            try:
                d1 = datetime.fromisoformat(first_form_d.file_date)
                d2 = datetime.fromisoformat(s1["file_date"])
                days_to_ipo = (d2 - d1).days
            except ValueError:
                pass

        last_amount = max(
            (f.total_offering_amount for f in form_d_filings if f.total_offering_amount),
            default=None,
        )

        return IPOConversion(
            company_name=company_name,
            cik=first_form_d.cik,
            first_form_d_date=first_form_d.file_date,
            s1_filing_date=s1.get("file_date"),
            days_to_ipo=days_to_ipo,
            last_private_round_amount=last_amount,
            s1_accession=s1.get("accession_no"),
        )

    def compute_time_to_ipo(self, timeline: OfferingTimeline) -> Optional[int]:
        """Days from first Form D to S-1 filing."""
        conversion = self.detect_ipo_conversion(timeline.company_name)
        return conversion.days_to_ipo if conversion else None

    def get_pre_ipo_fundraising(self, company_name: str) -> List[FormDFiling]:
        """Find pre-IPO Form D history for a given company name."""
        filings = self.fetcher.search_by_company(company_name)
        initial = [f for f in filings if not f.is_amendment]
        return sorted(initial, key=lambda f: f.file_date or "")

    def compute_ipo_premium_vs_last_private_round(self, company_name: str) -> Optional[float]:
        """
        Compute implied IPO premium vs last Form D implied valuation.
        Requires knowing IPO price (not available from EDGAR alone),
        so returns None unless we have price data.
        """
        # Without a pricing API, we can only indicate last private round size
        filings = self.get_pre_ipo_fundraising(company_name)
        if not filings:
            return None
        last = filings[-1]
        if last.total_offering_amount and last.num_accredited_investors:
            # Rough implied valuation: amount / typical VC ownership % (assume 20%)
            implied_val = last.total_offering_amount / 0.20
            return implied_val
        return None

    def scan_ipo_pipeline(self, days: int = 365, min_rounds: int = 3) -> pd.DataFrame:
        """
        Find companies with 3+ Form D rounds that haven't filed S-1 yet
        — likely approaching IPO.
        """
        screener = FormDScreenerEnhanced(self.fetcher)
        repeat_df = screener.screen_repeat_issuers(min_rounds=min_rounds)
        if repeat_df.empty:
            return pd.DataFrame()

        rows = []
        for _, row in repeat_df.iterrows():
            s1 = self.fetcher.search_s1_filings(row["company_name"])
            if not s1:
                rows.append(
                    {
                        "company_name": row["company_name"],
                        "cik": row["cik"],
                        "round_count": row["round_count"],
                        "estimated_stage": row["stage"],
                        "s1_filed": False,
                        "ipo_probability": self._estimate_ipo_probability(row["round_count"]),
                    }
                )

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("ipo_probability", ascending=False)
        return df.reset_index(drop=True)

    @staticmethod
    def _estimate_ipo_probability(round_count: int) -> float:
        """Rough heuristic: more rounds → higher IPO probability."""
        probs = {1: 0.02, 2: 0.06, 3: 0.12, 4: 0.25, 5: 0.40, 6: 0.55}
        return probs.get(round_count, min(0.70, round_count * 0.10))


# ---------------------------------------------------------------------------
# FormDIntelligenceEngine (orchestrator)
# ---------------------------------------------------------------------------

class FormDIntelligenceEngine:
    """
    Full orchestrator combining all Form D intelligence capabilities.
    """

    def __init__(self):
        self.fetcher = _FormDFetcher()
        self.analyzer = RegDExemptionAnalyzer()
        self.timeline_tracker = OfferingTimelineTracker(self.fetcher)
        self.ecosystem_mapper = VCEcosystemMapper(self.fetcher)
        self.screener = FormDScreenerEnhanced(self.fetcher)
        self.ipo_tracker = PostIPOTracker(self.fetcher)

    def get_market_overview(self, days: int = 30) -> MarketOverview:
        """Broad market overview: deal counts, capital, top states/industries."""
        filings = self.fetcher.search_recent(days=days, limit=500)

        if not filings:
            return MarketOverview(
                period_days=days,
                as_of_date=date.today().isoformat(),
                total_deals=0,
                total_capital=0.0,
                new_deals_506b=0,
                new_deals_506c=0,
                new_deals_504=0,
                new_deals_other=0,
            )

        initial = [f for f in filings if not f.is_amendment]
        total_capital = sum(
            f.total_offering_amount for f in initial if f.total_offering_amount
        )

        state_counts: Counter = Counter(
            f.state for f in initial if f.state and f.state in _STATE_ABBREVS
        )
        industry_counts: Counter = Counter(f.industry for f in initial if f.industry)

        largest = None
        max_amount = 0.0
        for f in initial:
            amt = f.total_offering_amount or 0
            if amt > max_amount:
                max_amount = amt
                largest = {
                    "company": f.company_name,
                    "amount": amt,
                    "state": f.state,
                    "exemption": f.exemption_normalized,
                }

        amounts = [f.total_offering_amount for f in initial if f.total_offering_amount]

        return MarketOverview(
            period_days=days,
            as_of_date=date.today().isoformat(),
            total_deals=len(initial),
            total_capital=total_capital,
            new_deals_506b=sum(1 for f in initial if f.exemption_normalized == "506b"),
            new_deals_506c=sum(1 for f in initial if f.exemption_normalized == "506c"),
            new_deals_504=sum(1 for f in initial if f.exemption_normalized == "504"),
            new_deals_other=sum(
                1
                for f in initial
                if f.exemption_normalized not in ("506b", "506c", "504")
            ),
            top_states=state_counts.most_common(10),
            top_industries=industry_counts.most_common(10),
            avg_deal_size=float(np.mean(amounts)) if amounts else 0.0,
            largest_deal=largest,
            crowdfunding_deals=sum(
                1 for f in initial if f.exemption_normalized == "regcf"
            ),
        )

    def analyze_company(self, company_name: str) -> CompanyAnalysis:
        """Full intelligence profile for a company."""
        filings = self.fetcher.search_by_company(company_name)
        cik = filings[0].cik if filings else ""

        initial = [f for f in filings if not f.is_amendment]
        total_raised = sum(
            f.amount_sold or f.total_offering_amount or 0 for f in initial
        )
        exemptions = list(
            dict.fromkeys(
                f.exemption_normalized for f in filings if f.exemption_normalized
            )
        )

        timeline = self.timeline_tracker.build_offering_timeline(company_name)
        latest_analysis = (
            self.analyzer.analyze_exemption(filings[-1]) if filings else None
        )
        ipo_conversion = self.ipo_tracker.detect_ipo_conversion(company_name)
        risk_score = (
            self.analyzer.compute_offering_risk_score(filings[-1]) if filings else 0.0
        )

        return CompanyAnalysis(
            company_name=company_name,
            cik=cik,
            total_filings=len(filings),
            total_rounds=len(initial),
            total_raised=total_raised,
            exemptions_used=exemptions,
            timeline=timeline,
            latest_exemption_analysis=latest_analysis,
            ipo_conversion=ipo_conversion,
            risk_score=risk_score,
        )

    def get_ecosystem_report(self, state: str) -> EcosystemData:
        return self.ecosystem_mapper.get_ecosystem_by_state(state)

    def screen_opportunities(self, criteria: Dict[str, Any]) -> pd.DataFrame:
        """
        Flexible screener. criteria keys:
          mode: stealth|repeat|large|crowdfunding|sophistication|rega
          days: int
          min_amount: float
          min_rounds: int
          state: str
        """
        mode = criteria.get("mode", "stealth")
        days = int(criteria.get("days", 30))
        min_amount = float(criteria.get("min_amount", 0))
        min_rounds = int(criteria.get("min_rounds", 3))

        if mode == "stealth":
            df = self.screener.screen_stealth_mode(days=days)
        elif mode == "repeat":
            df = self.screener.screen_repeat_issuers(min_rounds=min_rounds)
        elif mode == "large":
            df = self.screener.screen_large_rounds(
                min_amount=max(min_amount, 10_000_000)
            )
        elif mode == "crowdfunding":
            df = self.screener.screen_crowdfunding_active(days=days)
        elif mode == "sophistication":
            df = self.screener.screen_by_investor_sophistication()
        elif mode == "rega":
            df = self.screener.screen_rega_deals(days=days)
        else:
            df = self.screener.screen_stealth_mode(days=days)

        # Post-filter by state
        state_filter = criteria.get("state")
        if state_filter and not df.empty and "state" in df.columns:
            df = df[df["state"] == state_filter]

        # Post-filter by min_amount
        if min_amount > 0 and not df.empty and "total_offering_amount" in df.columns:
            df = df[df["total_offering_amount"] >= min_amount]

        return df.reset_index(drop=True) if not df.empty else df

    def track_ipo_pipeline(self) -> pd.DataFrame:
        """Form D companies approaching IPO (3+ rounds, no S-1 yet)."""
        return self.ipo_tracker.scan_ipo_pipeline(min_rounds=3)

    def get_state_comparison(
        self, states: List[str], days: int = 90
    ) -> pd.DataFrame:
        """Compare ecosystem metrics across multiple states."""
        rows = []
        for state in states:
            eco = self.get_ecosystem_report(state)
            rows.append(
                {
                    "state": state,
                    "total_deals": eco.total_deals,
                    "total_capital_m": eco.total_capital / 1e6,
                    "avg_deal_size_m": eco.avg_deal_size / 1e6,
                    "median_deal_size_m": eco.median_deal_size / 1e6,
                    "top_exemption": next(iter(eco.top_exemptions), "N/A"),
                    "top_industry": next(iter(eco.top_industries), "N/A"),
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("total_deals", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# dim_031 additions: raise velocity, Reg A+ detection, investor count signal
# ---------------------------------------------------------------------------


def compute_raise_velocity(
    total_raised: float,
    days_since_formation: int,
) -> dict:
    """
    Compute the annualised capital-raise velocity for a private company.

    Formula
    -------
    raise_velocity = (total_raised / days_since_formation) × 365

    This gives the annual run-rate of capital raised.  Higher velocity
    suggests strong investor demand and rapid scaling.

    Parameters
    ----------
    total_raised : float
        Total capital raised across all Form D filings (dollars).
    days_since_formation : int
        Calendar days from company formation (or first Form D) to today.
        Must be > 0.

    Returns
    -------
    dict with keys:
        raise_velocity_annual : float   annualised raise rate ($/year)
        total_raised          : float
        days_since_formation  : int
        daily_rate            : float   $/day
        formula               : str
    """
    if days_since_formation <= 0:
        raise ValueError("days_since_formation must be > 0")
    if total_raised < 0:
        raise ValueError("total_raised must be >= 0")

    daily_rate = total_raised / days_since_formation
    raise_velocity_annual = daily_rate * 365

    return {
        "raise_velocity_annual": round(raise_velocity_annual, 2),
        "total_raised": total_raised,
        "days_since_formation": days_since_formation,
        "daily_rate": round(daily_rate, 4),
        "formula": "raise_velocity = (total_raised / days_since_formation) × 365",
    }


def detect_regulation_a_plus(
    filing: "FormDFiling",
) -> dict:
    """
    Detect Regulation A+ (Tier 1 or Tier 2) from a Form D filing and validate
    the offering amount against the SEC caps.

    Tier 1: maximum $20,000,000 in any 12-month period.
    Tier 2: maximum $75,000,000 in any 12-month period.

    Parameters
    ----------
    filing : FormDFiling
        A parsed Form D filing.

    Returns
    -------
    dict with keys:
        is_reg_a_plus   : bool
        tier            : str | None   "Tier 1" | "Tier 2" | None
        max_amount      : float | None  SEC cap for the detected tier
        amount          : float | None  claimed offering amount
        within_cap      : bool | None   True if amount <= cap
        exemption_code  : str
    """
    exemption = filing.exemption_normalized or ""
    amount = filing.total_offering_amount

    tier: Optional[str] = None
    max_amount: Optional[float] = None
    within_cap: Optional[bool] = None

    if exemption == "rega_tier1":
        tier = "Tier 1"
        max_amount = 20_000_000.0
    elif exemption == "rega_tier2":
        tier = "Tier 2"
        max_amount = 75_000_000.0

    is_reg_a_plus = tier is not None

    if is_reg_a_plus and amount is not None and max_amount is not None:
        within_cap = amount <= max_amount

    return {
        "is_reg_a_plus": is_reg_a_plus,
        "tier": tier,
        "max_amount": max_amount,
        "amount": amount,
        "within_cap": within_cap,
        "exemption_code": exemption,
    }


def compute_investor_count_signal(
    num_investors: int,
    total_assets_under_management: float,
) -> dict:
    """
    Detect whether an offering triggers mandatory Exchange Act registration
    under Section 12(g) of the Securities Exchange Act of 1934.

    Threshold: >500 investors of record AND >$10,000,000 AUM
    → mandatory Exchange Act registration signal.

    Note: The JOBS Act raised the threshold to 2,000 investors (or 500
    non-accredited) for most issuers, but the classic 500-investor test
    remains relevant for older filings and smaller issuers that have not
    opted into the new rules.

    Parameters
    ----------
    num_investors : int
        Total number of investors in the current round (all types combined).
    total_assets_under_management : float
        Total AUM / total assets of the issuer (dollars).

    Returns
    -------
    dict with keys:
        mandatory_registration_signal : bool
        num_investors                 : int
        total_aum                     : float
        investor_threshold            : int     (500)
        aum_threshold                 : float   ($10,000,000)
        note                          : str
    """
    INVESTOR_THRESHOLD = 500
    AUM_THRESHOLD = 10_000_000.0

    triggers = (
        num_investors > INVESTOR_THRESHOLD
        and total_assets_under_management > AUM_THRESHOLD
    )

    return {
        "mandatory_registration_signal": triggers,
        "num_investors": num_investors,
        "total_aum": total_assets_under_management,
        "investor_threshold": INVESTOR_THRESHOLD,
        "aum_threshold": AUM_THRESHOLD,
        "note": (
            "Exceeds classic 500-investor / $10M AUM threshold — may trigger "
            "mandatory Exchange Act Section 12(g) registration."
            if triggers
            else
            "Below mandatory Exchange Act registration threshold."
        ),
    }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

form_d_screener_v3_router = APIRouter(prefix="/formd/v3", tags=["form-d-v3"])

_engine_instance: Optional[FormDIntelligenceEngine] = None


def _get_engine() -> FormDIntelligenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FormDIntelligenceEngine()
    return _engine_instance


class ScreenRequest(BaseModel):
    criteria: Dict[str, Any] = Field(default_factory=dict)


@form_d_screener_v3_router.get("/overview")
async def get_overview(days: int = Query(30, ge=1, le=365)):
    engine = _get_engine()
    overview = engine.get_market_overview(days=days)
    return {
        "period_days": overview.period_days,
        "as_of_date": overview.as_of_date,
        "total_deals": overview.total_deals,
        "total_capital_b": overview.total_capital / 1e9,
        "new_deals_506b": overview.new_deals_506b,
        "new_deals_506c": overview.new_deals_506c,
        "new_deals_504": overview.new_deals_504,
        "new_deals_other": overview.new_deals_other,
        "avg_deal_size_m": overview.avg_deal_size / 1e6,
        "top_states": [{"state": s, "deals": c} for s, c in overview.top_states[:10]],
        "top_industries": [{"industry": i, "deals": c} for i, c in overview.top_industries[:10]],
        "largest_deal": overview.largest_deal,
        "crowdfunding_deals": overview.crowdfunding_deals,
    }


@form_d_screener_v3_router.get("/company/{name}")
async def get_company_analysis(name: str):
    engine = _get_engine()
    try:
        analysis = engine.analyze_company(name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result: Dict[str, Any] = {
        "company_name": analysis.company_name,
        "cik": analysis.cik,
        "total_filings": analysis.total_filings,
        "total_rounds": analysis.total_rounds,
        "total_raised_m": analysis.total_raised / 1e6,
        "exemptions_used": analysis.exemptions_used,
        "risk_score": analysis.risk_score,
    }

    if analysis.latest_exemption_analysis:
        exc_analysis = analysis.latest_exemption_analysis
        result["latest_exemption"] = {
            "rule": exc_analysis.rule_normalized,
            "is_valid": exc_analysis.is_valid,
            "investor_type": exc_analysis.investor_type_classification,
            "risk_score": exc_analysis.risk_score,
            "flags": exc_analysis.flags,
        }

    if analysis.timeline:
        tl = analysis.timeline
        result["timeline"] = {
            "first_filing": tl.first_file_date,
            "last_filing": tl.last_file_date,
            "duration_days": tl.duration_days,
            "total_amendments": tl.total_amendments,
            "max_offering_amount_m": (tl.max_offering_amount or 0) / 1e6,
            "is_closed": tl.is_closed,
        }

    if analysis.ipo_conversion:
        ipo = analysis.ipo_conversion
        result["ipo_conversion"] = {
            "s1_date": ipo.s1_filing_date,
            "days_to_ipo": ipo.days_to_ipo,
        }

    return result


@form_d_screener_v3_router.get("/ecosystem/{state}")
async def get_ecosystem(
    state: str,
    days: int = Query(365, ge=30, le=730),
):
    if state.upper() not in _STATE_ABBREVS:
        raise HTTPException(status_code=400, detail=f"Invalid state abbreviation: {state}")
    engine = _get_engine()
    eco = engine.get_ecosystem_report(state.upper())
    return {
        "state": eco.state,
        "period_days": eco.period_days,
        "total_deals": eco.total_deals,
        "total_capital_b": eco.total_capital / 1e9,
        "avg_deal_size_m": eco.avg_deal_size / 1e6,
        "median_deal_size_m": eco.median_deal_size / 1e6,
        "top_exemptions": eco.top_exemptions,
        "top_industries": eco.top_industries,
        "top_investor_types": eco.top_investors_types,
        "hub_cities": eco.hub_cities[:10],
    }


@form_d_screener_v3_router.post("/screen")
async def screen(req: ScreenRequest):
    engine = _get_engine()
    df = engine.screen_opportunities(req.criteria)
    if df.empty:
        return {"results": [], "count": 0}
    return {
        "results": df.head(100).to_dict(orient="records"),
        "count": len(df),
    }


@form_d_screener_v3_router.get("/ipo-pipeline")
async def ipo_pipeline():
    engine = _get_engine()
    df = engine.track_ipo_pipeline()
    if df.empty:
        return {"pipeline": [], "count": 0}
    return {
        "pipeline": df.head(50).to_dict(orient="records"),
        "count": len(df),
    }


@form_d_screener_v3_router.get("/state-rankings")
async def state_rankings():
    engine = _get_engine()
    df = engine.ecosystem_mapper.compute_state_rankings()
    if df.empty:
        return {"rankings": []}
    return {"rankings": df.head(50).to_dict(orient="records")}


@form_d_screener_v3_router.get("/capital-concentration")
async def capital_concentration():
    engine = _get_engine()
    return engine.ecosystem_mapper.compute_capital_concentration()


# ---------------------------------------------------------------------------
# Main — demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    engine = FormDIntelligenceEngine()

    print("\n=== SENTINEL Form D Intelligence Engine V3 Demo ===\n")

    # 1. Market overview: last 30 days
    print("--- Market Overview (last 30 days) ---")
    overview = engine.get_market_overview(days=30)
    print(f"Total deals: {overview.total_deals}")
    print(f"Total capital: ${overview.total_capital / 1e9:.2f}B")
    print(f"506(b) deals: {overview.new_deals_506b}")
    print(f"506(c) deals: {overview.new_deals_506c}")
    print(f"Avg deal size: ${overview.avg_deal_size / 1e6:.2f}M")
    if overview.largest_deal:
        print(f"Largest deal: {overview.largest_deal['company']} — ${overview.largest_deal['amount'] / 1e6:.1f}M")
    print(f"Top states: {overview.top_states[:5]}")
    print(f"Crowdfunding deals: {overview.crowdfunding_deals}")

    # 2. Stealth mode 506(b) deals
    print("\n--- Stealth Mode 506(b) Deals (last 30 days) ---")
    stealth_df = engine.screen_opportunities({"mode": "stealth", "days": 30})
    print(f"Found {len(stealth_df)} stealth deals")
    if not stealth_df.empty:
        print(stealth_df[["company_name", "state", "total_offering_amount", "exemption"]].head(10).to_string(index=False))

    # 3. Ecosystem for CA + NY
    print("\n--- CA Ecosystem ---")
    ca_eco = engine.get_ecosystem_report("CA")
    print(f"CA: {ca_eco.total_deals} deals, ${ca_eco.total_capital / 1e9:.2f}B capital")
    print(f"  Top exemptions: {ca_eco.top_exemptions}")
    print(f"  Top industries: {dict(list(ca_eco.top_industries.items())[:5])}")
    print(f"  Hub cities: {ca_eco.hub_cities[:3]}")

    print("\n--- NY Ecosystem ---")
    ny_eco = engine.get_ecosystem_report("NY")
    print(f"NY: {ny_eco.total_deals} deals, ${ny_eco.total_capital / 1e9:.2f}B capital")

    # 4. Companies with 4+ rounds (Series C+ equivalent)
    print("\n--- Repeat Issuers (4+ rounds = Series C+) ---")
    repeat_df = engine.screen_opportunities({"mode": "repeat", "min_rounds": 4})
    print(f"Found {len(repeat_df)} companies with 4+ Form D rounds")
    if not repeat_df.empty:
        print(repeat_df.head(10).to_string(index=False))

    # 5. IPO pipeline
    print("\n--- IPO Pipeline (3+ rounds, no S-1 yet) ---")
    pipeline_df = engine.track_ipo_pipeline()
    print(f"Found {len(pipeline_df)} potential IPO candidates")
    if not pipeline_df.empty:
        print(pipeline_df.head(10).to_string(index=False))

    # 6. State rankings
    print("\n--- State Rankings ---")
    rankings_df = engine.ecosystem_mapper.compute_state_rankings()
    if not rankings_df.empty:
        print(rankings_df.head(10).to_string(index=False))

    # 7. Capital concentration
    print("\n--- Capital Concentration ---")
    concentration = engine.ecosystem_mapper.compute_capital_concentration()
    print(
        f"Top 3 states ({concentration.get('top3_states', [])}) account for "
        f"{concentration.get('deal_concentration_pct', 0):.1f}% of deals and "
        f"{concentration.get('capital_concentration_pct', 0):.1f}% of capital"
    )

    print("\nDone.")
