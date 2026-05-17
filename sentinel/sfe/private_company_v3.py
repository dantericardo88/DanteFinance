"""
private_company_v3.py — Private company intelligence platform for SENTINEL.

dim_097: Private company profiles (Form D)  score 6 → 9

Data sources (all free, no API key required):
  - SEC EDGAR EFTS full-text search: https://efts.sec.gov/LATEST/search-index
  - SEC EDGAR company search:        https://www.sec.gov/cgi-bin/browse-edgar
  - Form D XML filings:              https://www.sec.gov/Archives/edgar/data/{cik}/...
  - Form D bulk submissions API:     https://data.sec.gov/submissions/CIK{cik}.json
  - GDELT Project:                   https://api.gdeltproject.org (news signals)

SEC policy: max 10 req/sec, User-Agent header required.
Storage: SQLite at sentinel/data/private_company.db

Form D categories:
  - Form D: initial filing
  - Form D/A: amendment
  - Exemptions: 506(b), 506(c), 4(a)(2), Regulation A, Regulation CF
  - Industry groups: 15 standard SEC categories
  - Investment fund types: Hedge Fund, PE, VC, Real Estate, etc.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote_plus, urlencode
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_OK = False
    logger.warning("pandas not installed — DataFrame outputs will be plain lists. pip install pandas")

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment]
    _BS4_OK = False

try:
    import requests as _req_lib
    _REQUESTS_OK = True
except ImportError:
    _req_lib = None  # type: ignore[assignment]
    _REQUESTS_OK = False

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

EDGAR_EFTS      = "https://efts.sec.gov/LATEST/search-index"
EDGAR_DATA_API  = "https://data.sec.gov"
EDGAR_WWW       = "https://www.sec.gov"
RATE_SLEEP      = 0.12          # seconds between EDGAR requests (<10/sec)
MAX_EFTS_HITS   = 100           # max results per EFTS query
REQUEST_TIMEOUT = 20            # seconds

_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "SENTINEL private_company_v3 sentinel@example.com",
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

DB_PATH = Path(__file__).parent.parent / "data" / "private_company.db"

# ---------------------------------------------------------------------------
# SEC Form D industry groups (15 standard EDGAR categories)
# ---------------------------------------------------------------------------

INDUSTRY_GROUPS: Dict[str, str] = {
    "1":  "Agriculture",
    "2":  "Restaurants & Lounges",
    "3":  "Banking & Financial Services",
    "4":  "Business Services",
    "5":  "Energy",
    "6":  "Health Care",
    "7":  "Manufacturing",
    "8":  "Real Estate",
    "9":  "Retailing",
    "10": "Restaurants",
    "11": "Technology",
    "12": "Travel",
    "13": "Other",
    "14": "Pooled Investment Fund",
    "15": "Investing / Investment Management",
}

# Mapping from Form D investmentFundType values
FUND_TYPE_MAP: Dict[str, str] = {
    "Hedge Fund":                     "Hedge Fund",
    "Private Equity Fund":            "PE",
    "Venture Capital Fund":           "VC",
    "Real Estate Fund":               "Real Estate",
    "Other Investment Fund":          "Other Fund",
    "N/A":                            "N/A",
    "":                               "N/A",
}

# Exemption labels
EXEMPTION_MAP: Dict[str, str] = {
    "Rule 506(b)": "506(b)",
    "Rule 506(c)": "506(c)",
    "Section 4(a)(5)": "4(a)(5)",
    "Section 4(a)(2)": "4(a)(2)",
    "Regulation A":    "Reg A",
    "Regulation CF":   "Reg CF",
    "Rule 504":        "504",
    "Rule 505":        "505",
}

# Funding stage inference from offering amount
def _infer_stage(amount: float) -> str:
    if amount <= 0:
        return "Unknown"
    if amount < 500_000:
        return "Pre-Seed"
    if amount < 2_000_000:
        return "Seed"
    if amount < 10_000_000:
        return "Series A"
    if amount < 50_000_000:
        return "Series B"
    if amount < 200_000_000:
        return "Series C/D"
    return "Late Stage / Growth"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IssuerInfo:
    cik: str = ""
    entity_name: str = ""
    state_of_incorporation: str = ""
    state_of_inc_desc: str = ""
    jurisdiction: str = ""
    zip_code: str = ""
    issuer_type: str = ""       # "corporation", "limited partnership", etc.

@dataclass
class RelatedPerson:
    first_name: str = ""
    last_name: str = ""
    street1: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    relationship: str = ""      # "Executive Officer", "Director", "Promoter"
    clarification: str = ""

@dataclass
class OfferingDetails:
    industry_group_type: str = ""       # e.g. "Technology"
    investment_fund_type: str = ""      # "VC", "PE", "Hedge Fund"
    is_pooled_investment: bool = False
    date_of_first_sale: str = ""
    duration: str = ""                  # "Indefinite" or specific
    securities_offered: List[str] = field(default_factory=list)  # Equity, Debt, etc.
    business_combination: bool = False
    minimum_investment: float = 0.0
    sales_compensation_recipients: List[str] = field(default_factory=list)
    exemption_claimed: str = ""         # "506(b)", "506(c)", etc.
    total_offering_amount: float = 0.0
    total_amount_sold: float = 0.0
    total_remaining: float = 0.0
    clarification_of_response: str = ""
    investor_count_total: int = 0
    investor_count_accredited: int = 0
    investor_count_non_accredited: int = 0
    is_equity: bool = False
    is_debt: bool = False

@dataclass
class OfferingMetrics:
    offering_size_tier: str = ""        # Small/Mid/Large/Mega
    capital_efficiency: float = 0.0     # amount_sold / total_offering_amount
    implied_valuation: Optional[float] = None
    stage: str = ""
    is_vc_deal: bool = False
    is_pe_deal: bool = False
    is_hedge_fund: bool = False
    is_crowdfunding: bool = False
    days_since_filing: int = 0

@dataclass
class FormDFiling:
    accession_number: str = ""
    cik: str = ""
    filed_date: str = ""                # ISO date string "2024-06-15"
    form_type: str = "D"                # "D" or "D/A"
    issuer: IssuerInfo = field(default_factory=IssuerInfo)
    offering: OfferingDetails = field(default_factory=OfferingDetails)
    related_persons: List[RelatedPerson] = field(default_factory=list)
    url: str = ""
    source: str = "EDGAR"

    # convenience shortcuts (populated after parse)
    issuer_name: str = ""
    state: str = ""
    industry: str = ""
    exemption: str = ""
    total_offering_amount: float = 0.0
    amount_sold: float = 0.0
    investor_count: int = 0
    security_types: List[str] = field(default_factory=list)
    fund_type: str = ""
    date_of_first_sale: str = ""

@dataclass
class PrivateCompanyProfile:
    company_name: str = ""
    cik: str = ""
    state: str = ""
    industry: str = ""
    total_raised: float = 0.0
    filing_count: int = 0
    first_filing_date: str = ""
    last_filing_date: str = ""
    latest_amount: float = 0.0
    funding_stage: str = ""
    investors: List[str] = field(default_factory=list)
    exemptions_used: List[str] = field(default_factory=list)
    security_types: List[str] = field(default_factory=list)
    fund_type: str = ""
    filings: List[FormDFiling] = field(default_factory=list)
    fundraising_velocity: Dict[str, Any] = field(default_factory=dict)
    implied_valuation: Optional[float] = None

@dataclass
class DealCriteria:
    min_amount: float = 0.0
    max_amount: float = float("inf")
    industry: str = ""              # empty = all
    state: str = ""                 # empty = all
    exemption: str = ""             # "506(b)", "506(c)", "Reg CF", etc.
    fund_type: str = ""             # "VC", "PE", "Hedge Fund", etc.
    security_type: str = ""        # "Equity", "Debt", etc.
    days_since_filing: int = 60
    is_amendment: bool = False      # True = Form D/A only
    investor_count_min: int = 0

@dataclass
class DealDashboard:
    period_days: int = 30
    total_deals: int = 0
    total_capital_raised: float = 0.0
    avg_deal_size: float = 0.0
    median_deal_size: float = 0.0
    top_states: List[Tuple[str, int]] = field(default_factory=list)
    top_industries: List[Tuple[str, int]] = field(default_factory=list)
    exemption_breakdown: Dict[str, int] = field(default_factory=dict)
    vc_deals: int = 0
    pe_deals: int = 0
    reg_cf_deals: int = 0
    weekly_trend: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MarketTrends:
    period_days: int = 90
    deal_flow_weekly: List[Dict[str, Any]] = field(default_factory=list)
    hot_sectors: List[Dict[str, Any]] = field(default_factory=list)
    geographic_heatmap: Dict[str, int] = field(default_factory=dict)
    stage_distribution: Dict[str, int] = field(default_factory=dict)
    yoy_growth_pct: Optional[float] = None


# ---------------------------------------------------------------------------
# SQLite persistence layer
# ---------------------------------------------------------------------------

class _Database:
    """Local SQLite cache for Form D filings."""

    DDL = """
    CREATE TABLE IF NOT EXISTS form_d_filings (
        accession_number    TEXT PRIMARY KEY,
        cik                 TEXT,
        issuer_name         TEXT,
        filed_date          TEXT,
        form_type           TEXT,
        state               TEXT,
        industry            TEXT,
        exemption           TEXT,
        total_offering      REAL,
        amount_sold         REAL,
        investor_count      INTEGER,
        fund_type           TEXT,
        security_types      TEXT,
        raw_json            TEXT,
        fetched_at          TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_filed_date  ON form_d_filings(filed_date);
    CREATE INDEX IF NOT EXISTS idx_issuer      ON form_d_filings(issuer_name);
    CREATE INDEX IF NOT EXISTS idx_state       ON form_d_filings(state);
    CREATE INDEX IF NOT EXISTS idx_industry    ON form_d_filings(industry);
    CREATE INDEX IF NOT EXISTS idx_fund_type   ON form_d_filings(fund_type);
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(self.DDL)
            self._conn.commit()
        return self._conn

    def upsert_filing(self, filing: FormDFiling):
        sql = """
        INSERT OR REPLACE INTO form_d_filings
        (accession_number, cik, issuer_name, filed_date, form_type, state,
         industry, exemption, total_offering, amount_sold, investor_count,
         fund_type, security_types, raw_json, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        try:
            raw = json.dumps(asdict(filing))
        except Exception:
            raw = ""
        self.conn().execute(sql, (
            filing.accession_number,
            filing.cik,
            filing.issuer_name,
            filing.filed_date,
            filing.form_type,
            filing.state,
            filing.industry,
            filing.exemption,
            filing.total_offering_amount,
            filing.amount_sold,
            filing.investor_count,
            filing.fund_type,
            json.dumps(filing.security_types),
            raw,
            datetime.now(timezone.utc).isoformat(),
        ))
        self.conn().commit()

    def upsert_batch(self, filings: List[FormDFiling]):
        for f in filings:
            try:
                self.upsert_filing(f)
            except Exception as exc:
                logger.debug("upsert_filing failed for %s: %s", f.accession_number, exc)

    def query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        return self.conn().execute(sql, params).fetchall()

    def query_as_filings(self, sql: str, params: tuple = ()) -> List[FormDFiling]:
        rows = self.query(sql, params)
        filings = []
        for row in rows:
            try:
                raw = row["raw_json"]
                if raw:
                    d = json.loads(raw)
                    f = _dict_to_filing(d)
                    filings.append(f)
            except Exception:
                # reconstruct minimal filing from columns
                f = FormDFiling(
                    accession_number=row["accession_number"],
                    cik=row["cik"],
                    filed_date=row["filed_date"],
                    form_type=row["form_type"],
                    issuer_name=row["issuer_name"],
                    state=row["state"],
                    industry=row["industry"],
                    exemption=row["exemption"],
                    total_offering_amount=row["total_offering"] or 0.0,
                    amount_sold=row["amount_sold"] or 0.0,
                    investor_count=row["investor_count"] or 0,
                    fund_type=row["fund_type"] or "",
                    security_types=json.loads(row["security_types"] or "[]"),
                )
                filings.append(f)
        return filings

    def count(self) -> int:
        row = self.conn().execute("SELECT COUNT(*) as n FROM form_d_filings").fetchone()
        return row["n"] if row else 0


def _dict_to_filing(d: dict) -> FormDFiling:
    """Reconstruct a FormDFiling from a serialized dict."""
    issuer_d = d.get("issuer", {})
    offering_d = d.get("offering", {})
    persons_d = d.get("related_persons", [])
    issuer = IssuerInfo(**{k: issuer_d.get(k, "") for k in IssuerInfo.__dataclass_fields__})
    offering = OfferingDetails(**{
        k: offering_d.get(k, v.default if hasattr(v, "default") else
                          ([] if "list" in str(v.default_factory) else 0.0))
        for k, v in OfferingDetails.__dataclass_fields__.items()
    })
    persons = [
        RelatedPerson(**{k: p.get(k, "") for k in RelatedPerson.__dataclass_fields__})
        for p in persons_d
    ]
    f = FormDFiling(
        accession_number=d.get("accession_number", ""),
        cik=d.get("cik", ""),
        filed_date=d.get("filed_date", ""),
        form_type=d.get("form_type", "D"),
        issuer=issuer,
        offering=offering,
        related_persons=persons,
        url=d.get("url", ""),
        source=d.get("source", "EDGAR"),
        issuer_name=d.get("issuer_name", ""),
        state=d.get("state", ""),
        industry=d.get("industry", ""),
        exemption=d.get("exemption", ""),
        total_offering_amount=d.get("total_offering_amount", 0.0),
        amount_sold=d.get("amount_sold", 0.0),
        investor_count=d.get("investor_count", 0),
        security_types=d.get("security_types", []),
        fund_type=d.get("fund_type", ""),
        date_of_first_sale=d.get("date_of_first_sale", ""),
    )
    return f


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: dict = _HEADERS, timeout: int = REQUEST_TIMEOUT) -> bytes:
    """HTTP GET with EDGAR-compliant rate limiting."""
    time.sleep(RATE_SLEEP)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            logger.warning("EDGAR rate limit hit — sleeping 2s")
            time.sleep(2.0)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        raise


def _http_get_json(url: str) -> Any:
    raw = _http_get(url)
    return json.loads(raw)


def _http_get_text(url: str) -> str:
    return _http_get(url).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# FormDParser — parse SEC Form D XML
# ---------------------------------------------------------------------------

class FormDParser:
    """Parse Form D XML into structured dataclasses.

    SEC Form D XML schema: https://www.sec.gov/info/edgar/edgarfm-vol2-v54.htm
    Namespace: http://www.sec.gov/edgar/document/formd
    """

    _NS = {
        "fd": "http://www.sec.gov/edgar/document/formd",
        "ns": "http://www.sec.gov/edgar/document/formd",
    }

    def parse_xml(self, xml_content: str) -> FormDFiling:
        """Parse raw Form D XML string into a FormDFiling."""
        try:
            # SEC uses a specific namespace — strip it for simpler parsing
            xml_clean = re.sub(r' xmlns[^"]*"[^"]*"', "", xml_content)
            root = ET.fromstring(xml_clean)
        except ET.ParseError as exc:
            logger.warning("FormD XML parse error: %s", exc)
            return FormDFiling()

        filing = FormDFiling()
        filing.form_type = self._text(root, ".//submissionType") or "D"
        filing.cik = self._text(root, ".//issuerCik") or ""

        filing.issuer = self.extract_issuer_info(root)
        filing.offering = self.extract_offering_details(root)
        filing.related_persons = self.extract_related_persons(root)

        # Populate flat convenience fields
        filing.issuer_name = filing.issuer.entity_name
        filing.state = filing.issuer.state_of_incorporation
        filing.industry = filing.offering.industry_group_type
        filing.exemption = filing.offering.exemption_claimed
        filing.total_offering_amount = filing.offering.total_offering_amount
        filing.amount_sold = filing.offering.total_amount_sold
        filing.investor_count = filing.offering.investor_count_total
        filing.security_types = filing.offering.securities_offered
        filing.fund_type = FUND_TYPE_MAP.get(filing.offering.investment_fund_type, "")
        filing.date_of_first_sale = filing.offering.date_of_first_sale

        return filing

    def extract_issuer_info(self, root: ET.Element) -> IssuerInfo:
        """Extract issuer block."""
        info = IssuerInfo()
        # Multiple issuer support — take first
        issuer_el = root.find(".//issuer") or root.find(".//primaryIssuer")
        if issuer_el is None:
            issuer_el = root

        info.cik = self._text(issuer_el, "issuerCik") or self._text(root, ".//issuerCik") or ""
        info.entity_name = (
            self._text(issuer_el, "issuerName")
            or self._text(issuer_el, "entityName")
            or self._text(root, ".//entityName")
            or ""
        )
        info.state_of_incorporation = (
            self._text(issuer_el, "issuerStateOfInc")
            or self._text(issuer_el, "stateOfInc")
            or ""
        )
        info.jurisdiction = (
            self._text(issuer_el, "issuerJurisdictionOfInc")
            or info.state_of_incorporation
        )
        info.zip_code = self._text(issuer_el, ".//zipCode") or ""
        info.issuer_type = self._text(issuer_el, "issuerType") or ""
        return info

    def extract_offering_details(self, root: ET.Element) -> OfferingDetails:
        """Extract offering information block."""
        od = OfferingDetails()
        off_el = root.find(".//offeringData") or root

        # Industry group
        ig = off_el.find(".//industryGroup")
        if ig is not None:
            od.industry_group_type = (
                self._text(ig, "industryGroupType")
                or self._text(ig, "typeOfFiling")
                or ""
            )
            fund_el = ig.find(".//investmentFundInfo")
            if fund_el is not None:
                od.investment_fund_type = self._text(fund_el, "investmentFundType") or ""
                od.is_pooled_investment = True

        # Offering type
        ot_el = off_el.find(".//offeringType") or off_el
        od.is_equity = self._bool(ot_el, "isEquityType") or self._bool(ot_el, "isEquity")
        od.is_debt = self._bool(ot_el, "isDebtType") or self._bool(ot_el, "isDebt")

        # Securities type list
        for sec_el in root.findall(".//typesOfSecuritiesOffered"):
            for child in sec_el:
                label = child.tag.replace("is", "").replace("Type", "").replace("Offered", "")
                val = child.text or ""
                if val.strip().lower() in ("true", "1"):
                    od.securities_offered.append(label)

        # Date and duration
        od.date_of_first_sale = (
            self._text(off_el, ".//dateOfFirstSale")
            or self._text(off_el, ".//firstSaleDate")
            or ""
        )
        od.duration = self._text(off_el, ".//durationOfOffering") or ""

        # Exemption
        for ex_el in root.findall(".//exemptionsAndExclusions"):
            for child in ex_el:
                val = child.text or ""
                if val.strip().lower() in ("true", "1", "X"):
                    raw_label = child.tag
                    od.exemption_claimed = self._map_exemption(raw_label)
                    break
        if not od.exemption_claimed:
            # Try direct text
            ex_text = self._text(root, ".//rule506bFlagType") or ""
            if ex_text.strip().lower() in ("true", "1"):
                od.exemption_claimed = "506(b)"
            ex_text_c = self._text(root, ".//rule506cFlagType") or ""
            if ex_text_c.strip().lower() in ("true", "1"):
                od.exemption_claimed = "506(c)"
            reg_cf = self._text(root, ".//regulationCrowdfundingFlagType") or ""
            if reg_cf.strip().lower() in ("true", "1"):
                od.exemption_claimed = "Reg CF"

        # Amounts
        offer_info = root.find(".//offeringSalesAmounts") or off_el
        od.total_offering_amount = self._float(offer_info, "totalOfferingAmount")
        od.total_amount_sold = self._float(offer_info, "totalAmountSold")
        od.total_remaining = self._float(offer_info, "totalRemaining")

        # Minimum investment
        od.minimum_investment = self._float(off_el, "minimumInvestmentAccepted")

        # Investor count
        inv_el = root.find(".//salesCompensationRecipients") or root.find(".//numberSold")
        count_el = root.find(".//numbersOfInvestors") or root.find(".//salesCount")
        if count_el is not None:
            od.investor_count_total = int(self._float(count_el, "totalNumberAlreadyInvested") or 0)
            od.investor_count_accredited = int(self._float(count_el, "numberAccreditedInvestors") or 0)
            od.investor_count_non_accredited = int(self._float(count_el, "numberNonAccreditedInvestors") or 0)
            if od.investor_count_total == 0:
                od.investor_count_total = (
                    od.investor_count_accredited + od.investor_count_non_accredited
                )

        return od

    def extract_related_persons(self, root: ET.Element) -> List[RelatedPerson]:
        """Extract all related persons (officers, directors, promoters)."""
        persons = []
        for rp_el in root.findall(".//relatedPersonsList/relatedPersonInfo"):
            rp = RelatedPerson()
            name_el = rp_el.find(".//relatedPersonName")
            if name_el is not None:
                rp.first_name = self._text(name_el, "firstName") or ""
                rp.last_name = self._text(name_el, "lastName") or ""
            addr_el = rp_el.find(".//relatedPersonAddress")
            if addr_el is not None:
                rp.street1 = self._text(addr_el, "street1") or ""
                rp.city = self._text(addr_el, "city") or ""
                rp.state = self._text(addr_el, "stateOrCountry") or ""
                rp.zip_code = self._text(addr_el, "zipCode") or ""
            rel_el = rp_el.find(".//relatedPersonRelationshipList")
            if rel_el is not None:
                rels = [c.text for c in rel_el if c.text]
                rp.relationship = ", ".join(filter(None, rels))
            rp.clarification = self._text(rp_el, "relationshipClarification") or ""
            persons.append(rp)
        return persons

    def compute_offering_metrics(self, filing: FormDFiling) -> OfferingMetrics:
        """Compute derived metrics from a filing."""
        m = OfferingMetrics()
        amount = filing.total_offering_amount
        sold = filing.amount_sold

        # Size tier
        if amount < 500_000:
            m.offering_size_tier = "Micro (<500K)"
        elif amount < 2_000_000:
            m.offering_size_tier = "Small (500K-2M)"
        elif amount < 10_000_000:
            m.offering_size_tier = "Mid (2M-10M)"
        elif amount < 100_000_000:
            m.offering_size_tier = "Large (10M-100M)"
        else:
            m.offering_size_tier = "Mega (>100M)"

        m.capital_efficiency = (sold / amount) if amount > 0 else 0.0
        m.stage = _infer_stage(amount)
        m.is_vc_deal = filing.fund_type in ("VC", "Venture Capital Fund")
        m.is_pe_deal = filing.fund_type in ("PE", "Private Equity Fund")
        m.is_hedge_fund = "Hedge Fund" in filing.fund_type
        m.is_crowdfunding = filing.exemption in ("Reg CF", "Regulation CF")

        # Days since filing
        try:
            filed = datetime.fromisoformat(filing.filed_date)
            m.days_since_filing = (datetime.now(timezone.utc) - filed.replace(tzinfo=timezone.utc)).days
        except Exception:
            m.days_since_filing = 0

        return m

    # helpers
    @staticmethod
    def _text(el: ET.Element, tag: str) -> Optional[str]:
        child = el.find(tag)
        return child.text.strip() if child is not None and child.text else None

    @staticmethod
    def _bool(el: ET.Element, tag: str) -> bool:
        child = el.find(tag)
        if child is None or not child.text:
            return False
        return child.text.strip().lower() in ("true", "1", "yes", "x")

    @staticmethod
    def _float(el: ET.Element, tag: str) -> float:
        child = el.find(tag)
        if child is None or not child.text:
            return 0.0
        try:
            return float(child.text.strip().replace(",", ""))
        except ValueError:
            return 0.0

    @staticmethod
    def _map_exemption(tag: str) -> str:
        tag = tag.lower()
        if "506b" in tag or "rule506b" in tag:
            return "506(b)"
        if "506c" in tag or "rule506c" in tag:
            return "506(c)"
        if "crowdfunding" in tag or "regulationcf" in tag:
            return "Reg CF"
        if "regula" in tag and "a" in tag:
            return "Reg A"
        if "4a2" in tag or "4(a)(2)" in tag:
            return "4(a)(2)"
        if "504" in tag:
            return "504"
        if "505" in tag:
            return "505"
        return tag


# ---------------------------------------------------------------------------
# FormDCollector — fetch Form D filings from EDGAR
# ---------------------------------------------------------------------------

class FormDCollector:
    """Fetch Form D filings from SEC EDGAR EFTS and individual filing pages."""

    def __init__(self, db: Optional[_Database] = None):
        self._db = db or _Database()
        self._parser = FormDParser()

    # -----------------------------------------------------------------------
    # EFTS search
    # -----------------------------------------------------------------------

    def _efts_search(
        self,
        extra_params: Dict[str, str],
        max_hits: int = MAX_EFTS_HITS,
    ) -> List[dict]:
        """Run an EDGAR full-text search for Form D filings."""
        params: Dict[str, str] = {
            "forms": "D,D/A",
            "_source": (
                "file_date,entity_name,file_num,period_of_report,"
                "form_type,biz_location,inc_states,file_num"
            ),
            "dateRange": "custom",
        }
        params.update(extra_params)
        hits_all: List[dict] = []
        from_val = 0
        page_size = min(max_hits, 40)

        while len(hits_all) < max_hits:
            params["from"] = str(from_val)
            params["hits.hits.total.value"] = str(page_size)
            url = f"{EDGAR_EFTS}?{urlencode(params)}"
            try:
                data = _http_get_json(url)
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break
                hits_all.extend(hits)
                if len(hits) < page_size:
                    break
                from_val += page_size
            except Exception as exc:
                logger.warning("EFTS search failed: %s", exc)
                break

        return hits_all[:max_hits]

    def _hits_to_filings(self, hits: List[dict]) -> List[FormDFiling]:
        """Convert EFTS search hits to minimal FormDFiling objects."""
        filings = []
        for h in hits:
            src = h.get("_source", {})
            f = FormDFiling(
                accession_number=h.get("_id", ""),
                filed_date=src.get("file_date", ""),
                form_type=src.get("form_type", "D"),
                issuer_name=src.get("entity_name", ""),
                state=src.get("inc_states", [""])[0] if isinstance(src.get("inc_states"), list) else src.get("inc_states", ""),
                url=f"{EDGAR_WWW}/cgi-bin/browse-edgar?action=getcompany&filenum={src.get('file_num', '')}",
                source="EDGAR-EFTS",
            )
            f.issuer = IssuerInfo(entity_name=f.issuer_name, state_of_incorporation=f.state)
            filings.append(f)
        return filings

    def fetch_recent_filings(self, days: int = 30) -> List[FormDFiling]:
        """Fetch all Form D filings from the last N days."""
        start_dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_dt = datetime.now().strftime("%Y-%m-%d")
        hits = self._efts_search({
            "startdt": start_dt,
            "enddt": end_dt,
        }, max_hits=MAX_EFTS_HITS)
        filings = self._hits_to_filings(hits)
        logger.info("fetch_recent_filings: %d filings (last %d days)", len(filings), days)
        self._db.upsert_batch(filings)
        return filings

    def search_by_issuer(self, company_name: str) -> List[FormDFiling]:
        """Search Form D filings by issuer/company name."""
        hits = self._efts_search({"q": f'"{company_name}"'}, max_hits=50)
        filings = self._hits_to_filings(hits)
        return filings

    def search_by_sector(self, industry_code: str) -> List[FormDFiling]:
        """Search Form D filings by EDGAR industry group code."""
        industry_name = INDUSTRY_GROUPS.get(str(industry_code), industry_code)
        hits = self._efts_search({"q": industry_name, "forms": "D,D/A"}, max_hits=MAX_EFTS_HITS)
        return self._hits_to_filings(hits)

    def search_by_state(self, state: str) -> List[FormDFiling]:
        """Search Form D filings by state of incorporation."""
        # EFTS doesn't directly filter by inc_states, so we filter post-fetch
        hits = self._efts_search({}, max_hits=MAX_EFTS_HITS)
        filings = self._hits_to_filings(hits)
        state_up = state.upper()
        return [f for f in filings if f.state and f.state.upper() == state_up]

    def search_by_amount(self, min_amount: float, max_amount: float) -> List[FormDFiling]:
        """Search and filter Form D filings by offering amount (post-filter from DB)."""
        sql = """
        SELECT * FROM form_d_filings
        WHERE total_offering >= ? AND total_offering <= ?
        ORDER BY filed_date DESC
        LIMIT 200
        """
        rows = self._db.query(sql, (min_amount, max_amount))
        # Return structured filings from DB
        return self._db.query_as_filings(sql, (min_amount, max_amount))

    def fetch_filing_xml(self, accession_number: str) -> FormDFiling:
        """Fetch and parse the full Form D XML for an accession number.

        Accession format: 0001234567-24-001234 or 0001234567-24-001234.txt
        """
        # Normalize accession number
        acc = accession_number.replace("-", "").replace(".txt", "")
        # CIK is the first 10 digits of accession
        cik_part = acc[:10].lstrip("0") or "0"

        # Build URL to filing index
        acc_dashed = f"{acc[:10]}-{acc[10:12]}-{acc[12:]}"
        index_url = (
            f"{EDGAR_WWW}/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik_part}&type=D&dateb=&owner=include&count=5"
        )
        # Direct XML URL pattern
        xml_url = (
            f"{EDGAR_WWW}/Archives/edgar/data/{cik_part}/"
            f"{acc}/{acc_dashed}.xml"
        )
        try:
            xml_text = _http_get_text(xml_url)
            filing = self._parser.parse_xml(xml_text)
            filing.accession_number = accession_number
            filing.url = xml_url
            self._db.upsert_filing(filing)
            return filing
        except Exception as exc:
            logger.warning("fetch_filing_xml failed for %s: %s", accession_number, exc)
            # Try alternate URL pattern (no subfolder)
            try:
                alt_url = (
                    f"{EDGAR_WWW}/Archives/edgar/data/{cik_part}/"
                    f"{acc}/primary_doc.xml"
                )
                xml_text = _http_get_text(alt_url)
                filing = self._parser.parse_xml(xml_text)
                filing.accession_number = accession_number
                filing.url = alt_url
                return filing
            except Exception as exc2:
                logger.warning("alt fetch_filing_xml also failed: %s", exc2)
                return FormDFiling(accession_number=accession_number)

    def enrich_filing(self, filing: FormDFiling) -> FormDFiling:
        """Enrich a minimal EFTS-sourced filing by fetching its full XML."""
        if filing.accession_number:
            try:
                full = self.fetch_filing_xml(filing.accession_number)
                # Merge — prefer full XML data
                if full.issuer_name:
                    filing = full
            except Exception:
                pass
        return filing

    def fetch_from_submissions_api(self, cik: str) -> List[FormDFiling]:
        """Fetch all Form D filings for a CIK via submissions API."""
        cik_padded = cik.zfill(10)
        url = f"{EDGAR_DATA_API}/submissions/CIK{cik_padded}.json"
        try:
            data = _http_get_json(url)
            filings_raw = data.get("filings", {}).get("recent", {})
            forms = filings_raw.get("form", [])
            accessions = filings_raw.get("accessionNumber", [])
            dates = filings_raw.get("filingDate", [])
            result = []
            for form, acc, dt in zip(forms, accessions, dates):
                if form in ("D", "D/A"):
                    f = FormDFiling(
                        accession_number=acc,
                        cik=cik,
                        filed_date=dt,
                        form_type=form,
                        issuer_name=data.get("name", ""),
                        state=data.get("stateOfIncorporation", ""),
                    )
                    f.issuer = IssuerInfo(
                        entity_name=f.issuer_name,
                        state_of_incorporation=f.state,
                        cik=cik,
                    )
                    result.append(f)
            logger.debug("submissions API: CIK %s → %d Form D filings", cik, len(result))
            return result
        except Exception as exc:
            logger.warning("submissions_api failed for CIK %s: %s", cik, exc)
            return []


# ---------------------------------------------------------------------------
# VCPEActivityTracker
# ---------------------------------------------------------------------------

class VCPEActivityTracker:
    """VC/PE deal activity analysis built on Form D data.

    Falls back to importing vcpe_tracker_v3 if available, otherwise works
    independently from the local SQLite DB populated by FormDCollector.
    """

    def __init__(self, db: Optional[_Database] = None):
        self._db = db or _Database()
        self._collector = FormDCollector(self._db)
        # Try to import existing vcpe_tracker_v3
        try:
            import importlib
            self._vcpe_mod = importlib.import_module("sentinel.sfe.vcpe_tracker_v3")
        except Exception:
            self._vcpe_mod = None

    def _ensure_data(self, days: int):
        """Ensure we have fresh data for the given period."""
        count = self._db.count()
        if count < 10:
            logger.info("DB sparse (%d rows) — fetching fresh Form D data", count)
            self._collector.fetch_recent_filings(days=min(days, 90))

    def _rows_to_df(self, rows: List[sqlite3.Row]) -> Any:
        if not _PANDAS_OK:
            return [dict(r) for r in rows]
        records = [dict(r) for r in rows]
        return pd.DataFrame(records)

    def get_vc_activity(self, state: Optional[str] = None, days: int = 90) -> Any:
        """VC deals — equity offerings with VC fund type."""
        self._ensure_data(days)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
        SELECT issuer_name, state, industry, amount_sold, total_offering,
               investor_count, filed_date, exemption, accession_number
        FROM form_d_filings
        WHERE filed_date >= ?
          AND (fund_type LIKE '%VC%' OR fund_type LIKE '%Venture%'
               OR security_types LIKE '%Equity%')
        {state_clause}
        ORDER BY filed_date DESC
        LIMIT 500
        """
        state_clause = "AND state = ?" if state else ""
        sql = sql.format(state_clause=state_clause)
        params = (start, state.upper()) if state else (start,)
        rows = self._db.query(sql, params)
        return self._rows_to_df(rows)

    def get_pe_activity(self, state: Optional[str] = None, days: int = 90) -> Any:
        """PE deals — fund type = PE, leveraged buyout signals."""
        self._ensure_data(days)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        state_clause = "AND state = ?" if state else ""
        sql = f"""
        SELECT issuer_name, state, industry, amount_sold, total_offering,
               investor_count, filed_date, exemption, accession_number
        FROM form_d_filings
        WHERE filed_date >= ?
          AND (fund_type LIKE '%PE%' OR fund_type LIKE '%Private Equity%')
          {state_clause}
        ORDER BY amount_sold DESC
        LIMIT 500
        """
        params = (start, state.upper()) if state else (start,)
        rows = self._db.query(sql, params)
        return self._rows_to_df(rows)

    def compute_deal_flow_trend(self, days: int = 365) -> Any:
        """Weekly deal count and total amount raised over time."""
        self._ensure_data(days)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
        SELECT
            strftime('%Y-W%W', filed_date)  AS week,
            COUNT(*)                         AS deal_count,
            SUM(amount_sold)                AS total_raised,
            AVG(amount_sold)                AS avg_deal_size
        FROM form_d_filings
        WHERE filed_date >= ?
        GROUP BY week
        ORDER BY week ASC
        """
        rows = self._db.query(sql, (start,))
        return self._rows_to_df(rows)

    def get_hot_sectors(self, days: int = 90) -> Any:
        """Sectors with the most VC/equity deal activity."""
        self._ensure_data(days)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
        SELECT
            industry,
            COUNT(*)         AS deal_count,
            SUM(amount_sold) AS total_raised,
            AVG(amount_sold) AS avg_deal_size,
            MAX(amount_sold) AS max_deal
        FROM form_d_filings
        WHERE filed_date >= ?
          AND industry != ''
        GROUP BY industry
        ORDER BY deal_count DESC
        LIMIT 20
        """
        rows = self._db.query(sql, (start,))
        return self._rows_to_df(rows)

    def get_active_investors(self, days: int = 90) -> Any:
        """Most active investors (related persons appearing most frequently)."""
        self._ensure_data(days)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        # We rely on raw_json to extract related person names
        sql = """
        SELECT raw_json, filed_date FROM form_d_filings
        WHERE filed_date >= ?
        LIMIT 500
        """
        rows = self._db.query(sql, (start,))
        investor_counts: Dict[str, int] = {}
        for row in rows:
            try:
                d = json.loads(row["raw_json"] or "{}")
                for rp in d.get("related_persons", []):
                    name = f"{rp.get('first_name', '')} {rp.get('last_name', '')}".strip()
                    if name:
                        investor_counts[name] = investor_counts.get(name, 0) + 1
            except Exception:
                continue

        records = sorted(
            [{"investor_name": k, "deal_count": v} for k, v in investor_counts.items()],
            key=lambda x: x["deal_count"], reverse=True,
        )[:50]

        if _PANDAS_OK:
            return pd.DataFrame(records)
        return records

    def get_geographic_heatmap(self, days: int = 90) -> Dict[str, int]:
        """Deal counts by state."""
        self._ensure_data(days)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
        SELECT state, COUNT(*) AS deal_count
        FROM form_d_filings
        WHERE filed_date >= ? AND state != ''
        GROUP BY state
        ORDER BY deal_count DESC
        """
        rows = self._db.query(sql, (start,))
        return {row["state"]: row["deal_count"] for row in rows}


# ---------------------------------------------------------------------------
# PrivateCompanyProfiler
# ---------------------------------------------------------------------------

class PrivateCompanyProfiler:
    """Build enriched profiles for private companies from Form D history."""

    def __init__(self, db: Optional[_Database] = None):
        self._db = db or _Database()
        self._collector = FormDCollector(self._db)
        self._parser = FormDParser()

    def build_profile(self, company_name: str) -> PrivateCompanyProfile:
        """Build a comprehensive private company profile."""
        filings = self.get_funding_history(company_name)
        if not filings:
            return PrivateCompanyProfile(company_name=company_name)

        profile = PrivateCompanyProfile(company_name=company_name)
        profile.filings = filings
        profile.filing_count = len(filings)
        profile.cik = filings[0].cik or ""
        profile.state = filings[0].state or ""
        profile.industry = filings[0].industry or ""
        profile.fund_type = filings[0].fund_type or ""

        # Aggregate financial data
        amounts = [f.amount_sold for f in filings if f.amount_sold > 0]
        profile.total_raised = sum(amounts)
        profile.latest_amount = amounts[-1] if amounts else 0.0

        # Dates
        dates = sorted([f.filed_date for f in filings if f.filed_date])
        profile.first_filing_date = dates[0] if dates else ""
        profile.last_filing_date = dates[-1] if dates else ""

        # Stage
        profile.funding_stage = _infer_stage(profile.latest_amount)

        # Unique investors (from related persons)
        investor_set = set()
        for f in filings:
            for rp in f.related_persons:
                name = f"{rp.first_name} {rp.last_name}".strip()
                if name:
                    investor_set.add(name)
        profile.investors = sorted(investor_set)

        # Exemption + security types
        profile.exemptions_used = list({f.exemption for f in filings if f.exemption})
        sec_types: set = set()
        for f in filings:
            sec_types.update(f.security_types)
        profile.security_types = list(sec_types)

        # Velocity
        profile.fundraising_velocity = self.compute_fundraising_velocity(company_name, filings)

        # Implied valuation (best-effort)
        profile.implied_valuation = self._estimate_valuation_from_filings(filings)

        return profile

    def estimate_valuation(self, filing: FormDFiling) -> Optional[float]:
        """Estimate implied valuation if equity % and amount are available.

        Form D does not require equity %, so this is rarely possible.
        Returns None when insufficient data.
        """
        # Look for clarification text that mentions equity percentage
        clarification = filing.offering.clarification_of_response or ""
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", clarification)
        if pct_match and filing.amount_sold > 0:
            pct = float(pct_match.group(1)) / 100
            if 0 < pct < 1:
                return round(filing.amount_sold / pct, 2)
        return None

    def _estimate_valuation_from_filings(self, filings: List[FormDFiling]) -> Optional[float]:
        for f in reversed(filings):
            val = self.estimate_valuation(f)
            if val:
                return val
        return None

    def get_funding_history(
        self,
        company_name: str,
        filings: Optional[List[FormDFiling]] = None,
    ) -> List[FormDFiling]:
        """All Form D filings for a company, sorted by date."""
        if filings is not None:
            return filings

        # Check DB first
        sql = """
        SELECT * FROM form_d_filings
        WHERE LOWER(issuer_name) LIKE LOWER(?)
        ORDER BY filed_date ASC
        """
        db_filings = self._db.query_as_filings(sql, (f"%{company_name}%",))
        if db_filings:
            return db_filings

        # Fetch from EDGAR
        logger.info("Fetching Form D history for %s from EDGAR", company_name)
        raw_filings = self._collector.search_by_issuer(company_name)
        self._db.upsert_batch(raw_filings)
        return sorted(raw_filings, key=lambda f: f.filed_date)

    def compute_fundraising_velocity(
        self,
        company_name: str,
        filings: Optional[List[FormDFiling]] = None,
    ) -> Dict[str, Any]:
        """Compute pace of capital raises."""
        f_list = filings or self.get_funding_history(company_name)
        if not f_list:
            return {"rounds": 0, "total_raised": 0.0, "velocity": None}

        dates = [f.filed_date for f in f_list if f.filed_date]
        amounts = [f.amount_sold for f in f_list if f.amount_sold > 0]

        if len(dates) < 2:
            return {
                "rounds": len(f_list),
                "total_raised": sum(amounts),
                "velocity": None,
                "avg_days_between_rounds": None,
            }

        try:
            dt_objects = [datetime.fromisoformat(d) for d in sorted(dates)]
            gaps = [
                (dt_objects[i+1] - dt_objects[i]).days
                for i in range(len(dt_objects) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps) if gaps else None
        except Exception:
            avg_gap = None

        total_days = 0
        try:
            first = datetime.fromisoformat(sorted(dates)[0])
            last = datetime.fromisoformat(sorted(dates)[-1])
            total_days = (last - first).days
        except Exception:
            pass

        return {
            "rounds": len(f_list),
            "total_raised": sum(amounts),
            "avg_days_between_rounds": round(avg_gap, 1) if avg_gap else None,
            "total_period_days": total_days,
            "raises_per_year": round(len(f_list) / max(total_days / 365, 0.1), 2),
            "avg_round_size": round(sum(amounts) / len(amounts), 2) if amounts else None,
        }


# ---------------------------------------------------------------------------
# InvestorUniverse
# ---------------------------------------------------------------------------

class InvestorUniverse:
    """Track investors appearing as related persons in Form D filings."""

    def __init__(self, db: Optional[_Database] = None):
        self._db = db or _Database()
        self._collector = FormDCollector(self._db)

    def _get_all_related_persons(self, days: int = 365) -> List[Dict[str, Any]]:
        """Extract all related person records from cached filings."""
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = "SELECT raw_json, issuer_name, industry, state, filed_date FROM form_d_filings WHERE filed_date >= ?"
        rows = self._db.query(sql, (start,))
        records = []
        for row in rows:
            try:
                d = json.loads(row["raw_json"] or "{}")
                for rp in d.get("related_persons", []):
                    fname = rp.get("first_name", "").strip()
                    lname = rp.get("last_name", "").strip()
                    name = f"{fname} {lname}".strip()
                    if name:
                        records.append({
                            "investor_name": name,
                            "company": row["issuer_name"],
                            "industry": row["industry"],
                            "state": row["state"],
                            "filed_date": row["filed_date"],
                            "relationship": rp.get("relationship", ""),
                        })
            except Exception:
                continue
        return records

    def get_investor_portfolio(self, investor_name: str) -> Any:
        """All companies an investor appears in (Form D related persons)."""
        all_persons = self._get_all_related_persons(days=730)
        investor_lower = investor_name.lower()
        portfolio = [
            r for r in all_persons
            if investor_lower in r["investor_name"].lower()
        ]
        if _PANDAS_OK:
            return pd.DataFrame(portfolio)
        return portfolio

    def get_top_investors(self, n: int = 50, days: int = 365) -> Any:
        """Most active investors by deal count."""
        all_persons = self._get_all_related_persons(days=days)
        counts: Dict[str, int] = {}
        for r in all_persons:
            name = r["investor_name"]
            counts[name] = counts.get(name, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
        records = [{"investor_name": k, "deal_count": v} for k, v in top]
        if _PANDAS_OK:
            return pd.DataFrame(records)
        return records

    def get_investor_sector_focus(self, investor_name: str) -> Dict[str, int]:
        """Industry distribution for a given investor."""
        portfolio = self.get_investor_portfolio(investor_name)
        if _PANDAS_OK and hasattr(portfolio, "groupby"):
            if not portfolio.empty and "industry" in portfolio.columns:
                counts = portfolio["industry"].value_counts()
                return counts.to_dict()
            return {}
        sector_counts: Dict[str, int] = {}
        for r in (portfolio if isinstance(portfolio, list) else []):
            ind = r.get("industry", "")
            if ind:
                sector_counts[ind] = sector_counts.get(ind, 0) + 1
        return sector_counts

    def compute_investor_activity_score(self, investor_name: str) -> float:
        """Activity score 0-100 based on deal frequency in last 365 days."""
        portfolio = self.get_investor_portfolio(investor_name)
        if _PANDAS_OK and hasattr(portfolio, "__len__"):
            deal_count = len(portfolio)
        else:
            deal_count = len(portfolio) if isinstance(portfolio, list) else 0

        # Score: log scale, max at 50+ deals
        if deal_count == 0:
            return 0.0
        import math
        score = min(100.0, (math.log(deal_count + 1) / math.log(51)) * 100)
        return round(score, 2)


# ---------------------------------------------------------------------------
# PrivateMarketScreener
# ---------------------------------------------------------------------------

class PrivateMarketScreener:
    """Screen private company Form D deals by configurable criteria."""

    PRESETS: Dict[str, DealCriteria] = {
        "seed_rounds": DealCriteria(
            min_amount=0, max_amount=2_000_000,
            security_type="Equity", days_since_filing=60,
        ),
        "series_a": DealCriteria(
            min_amount=2_000_000, max_amount=10_000_000,
            security_type="Equity", days_since_filing=90,
        ),
        "pe_buyouts": DealCriteria(
            min_amount=50_000_000, fund_type="PE", days_since_filing=90,
        ),
        "regulation_cf": DealCriteria(
            exemption="Reg CF", days_since_filing=60,
        ),
        "hot_sectors": DealCriteria(
            days_since_filing=30,
        ),
        "geographic": DealCriteria(
            days_since_filing=60,
        ),
    }

    def __init__(self, db: Optional[_Database] = None):
        self._db = db or _Database()
        self._collector = FormDCollector(self._db)

    def screen_deals(self, criteria: DealCriteria) -> Any:
        """Screen Form D filings by criteria."""
        start = (datetime.now() - timedelta(days=criteria.days_since_filing)).strftime("%Y-%m-%d")

        clauses = ["filed_date >= ?"]
        params: List[Any] = [start]

        if criteria.min_amount > 0:
            clauses.append("total_offering >= ?")
            params.append(criteria.min_amount)

        if criteria.max_amount < float("inf"):
            clauses.append("total_offering <= ?")
            params.append(criteria.max_amount)

        if criteria.industry:
            clauses.append("LOWER(industry) LIKE LOWER(?)")
            params.append(f"%{criteria.industry}%")

        if criteria.state:
            clauses.append("UPPER(state) = UPPER(?)")
            params.append(criteria.state)

        if criteria.exemption:
            clauses.append("LOWER(exemption) LIKE LOWER(?)")
            params.append(f"%{criteria.exemption}%")

        if criteria.fund_type:
            clauses.append("LOWER(fund_type) LIKE LOWER(?)")
            params.append(f"%{criteria.fund_type}%")

        if criteria.security_type:
            clauses.append("security_types LIKE ?")
            params.append(f"%{criteria.security_type}%")

        if criteria.investor_count_min > 0:
            clauses.append("investor_count >= ?")
            params.append(criteria.investor_count_min)

        if criteria.is_amendment:
            clauses.append("form_type = 'D/A'")

        where = " AND ".join(clauses)
        sql = f"""
        SELECT issuer_name, state, industry, total_offering, amount_sold,
               investor_count, exemption, fund_type, security_types,
               filed_date, form_type, accession_number
        FROM form_d_filings
        WHERE {where}
        ORDER BY filed_date DESC
        LIMIT 500
        """
        rows = self._db.query(sql, tuple(params))
        records = [dict(r) for r in rows]

        if _PANDAS_OK:
            df = pd.DataFrame(records)
            if not df.empty:
                df["funding_stage"] = df["total_offering"].apply(
                    lambda x: _infer_stage(float(x or 0))
                )
            return df
        return records

    def run_preset(self, name: str, **kwargs) -> Any:
        """Run a named preset screen with optional overrides."""
        if name not in self.PRESETS:
            raise ValueError(
                f"Preset '{name}' not found. Available: {list(self.PRESETS.keys())}"
            )
        criteria = self.PRESETS[name]
        # Allow overrides
        for k, v in kwargs.items():
            if hasattr(criteria, k):
                setattr(criteria, k, v)
        # Ensure DB has data
        if self._db.count() < 5:
            logger.info("DB empty — seeding with recent Form D data")
            self._collector.fetch_recent_filings(days=criteria.days_since_filing)
        return self.screen_deals(criteria)


# ---------------------------------------------------------------------------
# PrivateCompanyEngine — orchestrator
# ---------------------------------------------------------------------------

class PrivateCompanyEngine:
    """Main orchestrator for all private company intelligence functions."""

    def __init__(self, db_path: Path = DB_PATH):
        self._db = _Database(db_path)
        self._collector = FormDCollector(self._db)
        self._parser = FormDParser()
        self._profiler = PrivateCompanyProfiler(self._db)
        self._screener = PrivateMarketScreener(self._db)
        self._vc_tracker = VCPEActivityTracker(self._db)
        self._investor_universe = InvestorUniverse(self._db)

    def _ensure_fresh_data(self, days: int = 30):
        """Fetch data if the local DB is sparse."""
        count = self._db.count()
        if count < 10:
            logger.info("Seeding DB from EDGAR (last %d days)...", days)
            self._collector.fetch_recent_filings(days=days)

    def get_deal_dashboard(self, days: int = 30) -> DealDashboard:
        """Overview dashboard of the private deal market."""
        self._ensure_fresh_data(days)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        total_row = self._db.query(
            "SELECT COUNT(*) as n, SUM(amount_sold) as total FROM form_d_filings WHERE filed_date >= ?",
            (start,)
        )
        n = total_row[0]["n"] if total_row else 0
        total_capital = total_row[0]["total"] or 0.0

        # Amounts for median
        amounts = [
            row["amount_sold"] or 0.0
            for row in self._db.query(
                "SELECT amount_sold FROM form_d_filings WHERE filed_date >= ? AND amount_sold > 0",
                (start,)
            )
        ]
        median_deal = float(sorted(amounts)[len(amounts)//2]) if amounts else 0.0

        # Top states
        state_rows = self._db.query(
            "SELECT state, COUNT(*) as n FROM form_d_filings WHERE filed_date >= ? AND state != '' GROUP BY state ORDER BY n DESC LIMIT 10",
            (start,)
        )
        top_states = [(r["state"], r["n"]) for r in state_rows]

        # Top industries
        ind_rows = self._db.query(
            "SELECT industry, COUNT(*) as n FROM form_d_filings WHERE filed_date >= ? AND industry != '' GROUP BY industry ORDER BY n DESC LIMIT 10",
            (start,)
        )
        top_industries = [(r["industry"], r["n"]) for r in ind_rows]

        # Exemption breakdown
        ex_rows = self._db.query(
            "SELECT exemption, COUNT(*) as n FROM form_d_filings WHERE filed_date >= ? GROUP BY exemption",
            (start,)
        )
        exemption_breakdown = {r["exemption"] or "Unknown": r["n"] for r in ex_rows}

        # VC / PE / Reg CF counts
        vc_count = self._db.query(
            "SELECT COUNT(*) as n FROM form_d_filings WHERE filed_date >= ? AND fund_type LIKE '%VC%'",
            (start,)
        )[0]["n"]
        pe_count = self._db.query(
            "SELECT COUNT(*) as n FROM form_d_filings WHERE filed_date >= ? AND fund_type LIKE '%PE%'",
            (start,)
        )[0]["n"]
        cf_count = self._db.query(
            "SELECT COUNT(*) as n FROM form_d_filings WHERE filed_date >= ? AND exemption LIKE '%CF%'",
            (start,)
        )[0]["n"]

        # Weekly trend
        trend_rows = self._db.query(
            """SELECT strftime('%Y-W%W', filed_date) AS week,
                      COUNT(*) AS deals, SUM(amount_sold) AS raised
               FROM form_d_filings WHERE filed_date >= ?
               GROUP BY week ORDER BY week""",
            (start,)
        )
        weekly_trend = [
            {"week": r["week"], "deals": r["deals"], "raised": r["raised"] or 0.0}
            for r in trend_rows
        ]

        return DealDashboard(
            period_days=days,
            total_deals=n,
            total_capital_raised=total_capital,
            avg_deal_size=total_capital / n if n > 0 else 0.0,
            median_deal_size=median_deal,
            top_states=top_states,
            top_industries=top_industries,
            exemption_breakdown=exemption_breakdown,
            vc_deals=vc_count,
            pe_deals=pe_count,
            reg_cf_deals=cf_count,
            weekly_trend=weekly_trend,
        )

    def search_company(self, name: str) -> PrivateCompanyProfile:
        """Search for a company and build its profile."""
        return self._profiler.build_profile(name)

    def run_screen(self, criteria: DealCriteria) -> Any:
        """Run a deal screen."""
        self._ensure_fresh_data(criteria.days_since_filing)
        return self._screener.screen_deals(criteria)

    def get_market_trends(self, days: int = 90) -> MarketTrends:
        """Comprehensive market trends analysis."""
        self._ensure_fresh_data(days)

        deal_flow = self._vc_tracker.compute_deal_flow_trend(days)
        hot_sectors = self._vc_tracker.get_hot_sectors(days)
        geo_heatmap = self._vc_tracker.get_geographic_heatmap(days)

        # Stage distribution
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        amount_rows = self._db.query(
            "SELECT total_offering FROM form_d_filings WHERE filed_date >= ? AND total_offering > 0",
            (start,)
        )
        stage_dist: Dict[str, int] = {}
        for row in amount_rows:
            stage = _infer_stage(row["total_offering"] or 0.0)
            stage_dist[stage] = stage_dist.get(stage, 0) + 1

        # YoY growth (compare current period vs same period 1yr ago)
        year_ago_start = (datetime.now() - timedelta(days=days + 365)).strftime("%Y-%m-%d")
        year_ago_end = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        current_count_row = self._db.query(
            "SELECT COUNT(*) as n FROM form_d_filings WHERE filed_date >= ?", (start,)
        )
        prior_count_row = self._db.query(
            "SELECT COUNT(*) as n FROM form_d_filings WHERE filed_date >= ? AND filed_date <= ?",
            (year_ago_start, year_ago_end)
        )
        current_n = current_count_row[0]["n"] if current_count_row else 0
        prior_n = prior_count_row[0]["n"] if prior_count_row else 0
        yoy_growth = (
            round((current_n - prior_n) / prior_n * 100, 2) if prior_n > 0 else None
        )

        # Serialize DataFrames to list of dicts
        def _df_to_list(obj: Any) -> List[Dict]:
            if _PANDAS_OK and hasattr(obj, "to_dict"):
                return obj.to_dict(orient="records")
            return obj if isinstance(obj, list) else []

        return MarketTrends(
            period_days=days,
            deal_flow_weekly=_df_to_list(deal_flow),
            hot_sectors=_df_to_list(hot_sectors),
            geographic_heatmap=geo_heatmap,
            stage_distribution=stage_dist,
            yoy_growth_pct=yoy_growth,
        )

    def export_deals(self, path: str, days: int = 30):
        """Export Form D filings to CSV."""
        self._ensure_fresh_data(days)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
        SELECT issuer_name, cik, state, industry, total_offering, amount_sold,
               investor_count, exemption, fund_type, security_types,
               filed_date, form_type, accession_number
        FROM form_d_filings
        WHERE filed_date >= ?
        ORDER BY filed_date DESC
        """
        rows = self._db.query(sql, (start,))
        records = [dict(r) for r in rows]
        if _PANDAS_OK:
            df = pd.DataFrame(records)
            df.to_csv(path, index=False)
            logger.info("Exported %d deals to %s", len(records), path)
        else:
            import csv
            if records:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
                logger.info("Exported %d deals to %s", len(records), path)


# ---------------------------------------------------------------------------
# Entry point — demonstration run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 65)
    print("SENTINEL — Private Company Intelligence (dim_097)")
    print("=" * 65)

    engine = PrivateCompanyEngine()

    # Step 1: fetch last 30 days of Form D filings
    print("\n[1] Fetching recent Form D filings (last 30 days)...")
    collector = engine._collector
    filings = collector.fetch_recent_filings(days=30)
    print(f"    Fetched {len(filings)} filings. DB total: {engine._db.count()}")

    # Step 2: deal dashboard
    print("\n[2] Deal Dashboard (30 days)...")
    dash = engine.get_deal_dashboard(days=30)
    print(f"    Total deals:       {dash.total_deals}")
    print(f"    Total capital:     ${dash.total_capital_raised:,.0f}")
    print(f"    Avg deal size:     ${dash.avg_deal_size:,.0f}")
    print(f"    VC deals:          {dash.vc_deals}")
    print(f"    PE deals:          {dash.pe_deals}")
    print(f"    Reg CF deals:      {dash.reg_cf_deals}")
    if dash.top_industries:
        print("    Top industries:")
        for ind, cnt in dash.top_industries[:5]:
            print(f"      {ind:30s} {cnt} deals")
    if dash.top_states:
        print("    Top states:")
        for st, cnt in dash.top_states[:5]:
            print(f"      {st:10s} {cnt} deals")

    # Step 3: hot sectors
    print("\n[3] Hot Sectors (VC activity, 90 days)...")
    vc_tracker = engine._vc_tracker
    hot = vc_tracker.get_hot_sectors(days=90)
    if _PANDAS_OK and hasattr(hot, "iterrows"):
        for _, row in hot.head(5).iterrows():
            print(f"    {row.get('industry','?'):30s} "
                  f"{int(row.get('deal_count',0)):4d} deals  "
                  f"${float(row.get('total_raised',0)):>14,.0f} raised")
    elif isinstance(hot, list):
        for row in hot[:5]:
            print(f"    {row.get('industry','?')}: {row.get('deal_count')} deals")

    # Step 4: seed rounds screener
    print("\n[4] Seed Rounds Screen (< $2M equity, last 60 days)...")
    screener = engine._screener
    try:
        seed_df = screener.run_preset("seed_rounds")
        if _PANDAS_OK and hasattr(seed_df, "__len__"):
            print(f"    Found {len(seed_df)} seed deals")
            if not seed_df.empty:
                cols = ["issuer_name", "state", "industry", "total_offering", "filed_date"]
                cols = [c for c in cols if c in seed_df.columns]
                print(seed_df[cols].head(5).to_string(index=False))
        else:
            print(f"    Found {len(seed_df)} seed deals")
    except Exception as exc:
        print(f"    Screen failed: {exc}")

    # Step 5: build a company profile (use first fetched issuer)
    sample_company = filings[0].issuer_name if filings else "Acme Corp"
    print(f"\n[5] Company Profile: {sample_company}")
    profile = engine.search_company(sample_company)
    print(f"    Total raised:     ${profile.total_raised:,.0f}")
    print(f"    Funding stage:    {profile.funding_stage}")
    print(f"    Filings:          {profile.filing_count}")
    print(f"    State:            {profile.state}")
    print(f"    Industry:         {profile.industry}")
    if profile.investors:
        print(f"    Investors:        {', '.join(profile.investors[:3])}")

    # Step 6: market trends
    print("\n[6] Market Trends (90 days)...")
    trends = engine.get_market_trends(days=90)
    print(f"    Weekly deal flow: {len(trends.deal_flow_weekly)} weeks of data")
    print(f"    Stage distribution:")
    for stage, cnt in sorted(trends.stage_distribution.items(), key=lambda x: -x[1]):
        print(f"      {stage:25s} {cnt}")
    if trends.yoy_growth_pct is not None:
        print(f"    YoY deal growth:  {trends.yoy_growth_pct:+.1f}%")

    print("\nDone.")
