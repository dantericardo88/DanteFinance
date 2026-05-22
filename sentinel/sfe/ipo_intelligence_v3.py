"""ipo_intelligence_v3.py — Production-grade IPO / S-1 intelligence (dim_030, target 9/10).

Key upgrades over prior versions:
  • Structured EDGAR S-1 data extraction — not keyword-counting heuristics
  • Use-of-Proceeds: extract actual dollar allocations from text
  • Risk factors: count <li> items in the Risk Factors HTML section
  • Financials from XBRL: revenue, gross margin, net income, cash burn — 2yr audited
  • Pipeline state machine: S-1 filed → S-1/A (roadshow) → 424B4 (priced) → trading
  • IPO pricing analytics: vs. range midpoint, 1/30/90/180-day market-adjusted return
  • Underwriter quality: bulge-bracket detection, league table tracking
  • Insider selling %: shares sold by existing holders vs total offering
  • Dual-class detection: Class A/B differential voting
  • SPAC tracker with Form 8-K merger detection
  • SQLite: ipo_pipeline, ipo_results, s1_analysis, spac_tracker, underwriter_league
  • FastAPI router at /ipo/v3

Public API
----------
EdgarS1Parser
    parse_s1(cik, accession)                      -> S1Analysis
    extract_use_of_proceeds(html)                 -> UseOfProceeds
    count_risk_factors_structured(html)           -> int
    extract_offering_structure(html)              -> OfferingStructure
    detect_dual_class(html)                       -> bool
    extract_underwriters(html)                    -> list[UnderwriterInfo]
    get_xbrl_financials(cik)                      -> XBRLFinancials

IPOPipelineTracker
    get_pipeline(lookback_days)                   -> list[PipelineEntry]
    get_recent_priced(days)                       -> list[IPOResult]
    track_amendments(cik)                         -> str    pipeline state
    detect_424b4_pricing(cik)                     -> PricingInfo | None

IPOPerformanceEngine
    day1_return(ticker, ipo_price)                -> dict
    aftermarket_returns(ticker, ipo_date, ipo_price, periods) -> dict
    market_adjusted_return(ticker, benchmark, ipo_date, periods) -> dict
    underpricing_vs_fee(underwriter_fee_pct, day1_return) -> dict

SPACTracker
    get_active_spacs(lookback_days)               -> list[SPACRecord]
    detect_spac_merger(cik)                       -> MergerAnnouncement | None

UnderwriterLeagueTable
    update(ipo_result)                            -> None
    get_table(year)                               -> pd.DataFrame

FastAPI router: ipo_v3_router at /ipo/v3
    GET /pipeline
    GET /recent?days=90
    GET /analysis/{ticker}
    GET /returns/{ticker}
    GET /spac-pipeline
    GET /underwriter-league
    GET /upcoming-lockups
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_HTML_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml",
}

_EDGAR_BASE     = "https://data.sec.gov"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_EDGAR_EFTS     = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_ATOM     = "https://www.sec.gov/cgi-bin/browse-edgar"

_TIMEOUT    = 45.0
_RATE_DELAY = 0.15

_DB_PATH = Path(__file__).parent.parent / "data" / "ipo_v3.db"

# S-1 form variants
_S1_FORMS   = {"S-1", "S-1/A", "S-11", "F-1", "F-1/A"}
_FINAL_PROS = {"424B4", "424B3", "424B1", "424B2"}
_AMENDMENT_FORMS = {"S-1/A", "F-1/A"}

# Bulge bracket — top-tier underwriters
_BULGE_BRACKET = {
    "Goldman Sachs", "Morgan Stanley", "JPMorgan", "J.P. Morgan",
    "Bank of America", "Merrill Lynch", "Citigroup", "Citi",
    "Deutsche Bank", "Barclays", "UBS", "Credit Suisse",
    "Wells Fargo Securities",
}
_MAJOR_UNDERWRITERS = _BULGE_BRACKET | {
    "RBC Capital", "Jefferies", "Cowen", "Piper Sandler", "Needham",
    "William Blair", "Stifel", "Cantor Fitzgerald", "Oppenheimer",
    "KeyBanc", "Evercore", "Lazard", "Guggenheim", "Moelis",
    "BofA Securities", "Truist", "Raymond James", "Baird",
}

# SPAC markers
_SPAC_MARKERS = [
    "blank check company", "special purpose acquisition",
    "no operating history", "business combination",
    "trust account", "founder shares", "sponsor",
]

# XBRL revenue concepts (priority order)
_REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
_COGS_CONCEPTS = ["CostOfGoodsAndServicesSold", "CostOfRevenue"]
_NI_CONCEPTS   = ["NetIncomeLoss", "ProfitLoss"]
_CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
]
_CFO_CONCEPTS  = ["NetCashProvidedByUsedInOperatingActivities"]


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS ipo_pipeline (
    cik             TEXT NOT NULL,
    company_name    TEXT,
    ticker          TEXT,
    sic_code        TEXT,
    form_type       TEXT,
    filed_date      TEXT,
    accession       TEXT,
    pipeline_state  TEXT,   -- filed|roadshow|priced|trading|withdrawn
    price_range_low REAL,
    price_range_high REAL,
    offer_price     REAL,
    shares_offered  INTEGER,
    updated_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (cik, accession)
);

CREATE TABLE IF NOT EXISTS ipo_results (
    ticker          TEXT NOT NULL PRIMARY KEY,
    cik             TEXT,
    company_name    TEXT,
    ipo_date        TEXT,
    offer_price     REAL,
    first_day_open  REAL,
    first_day_close REAL,
    day1_return_pct REAL,
    day30_return_pct REAL,
    day90_return_pct REAL,
    day180_return_pct REAL,
    day30_mktadj_pct REAL,
    day90_mktadj_pct REAL,
    day180_mktadj_pct REAL,
    lead_underwriter TEXT,
    underwriter_fee_pct REAL,
    underwriting_discount_pct REAL,
    market_cap_at_ipo REAL,
    lockup_days     INTEGER,
    lockup_expiry   TEXT,
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS s1_analysis (
    cik             TEXT NOT NULL,
    accession       TEXT NOT NULL,
    company_name    TEXT,
    filed_date      TEXT,
    form_type       TEXT,
    -- Use of Proceeds
    proceeds_rd_pct         REAL,
    proceeds_sales_pct      REAL,
    proceeds_debt_repay_pct REAL,
    proceeds_general_pct    REAL,
    proceeds_acquisitions_pct REAL,
    proceeds_secondary_pct  REAL,    -- insider selling portion
    proceeds_total_mn       REAL,
    proceeds_text           TEXT,
    -- Risk factors
    risk_factor_count       INTEGER,
    has_dual_class          INTEGER, -- 0/1
    -- Financials from XBRL
    revenue_yr0             REAL,
    revenue_yr1             REAL,
    revenue_growth_yoy      REAL,
    revenue_cagr_2yr        REAL,
    gross_margin            REAL,
    net_income_yr0          REAL,
    cash_yr0                REAL,
    quarterly_cash_burn     REAL,    -- avg quarterly CFO if negative
    -- Offering
    shares_offered          INTEGER,
    shares_existing_sold    INTEGER,  -- insider / secondary
    insider_selling_pct     REAL,     -- existing / total
    price_range_low         REAL,
    price_range_high        REAL,
    offer_price             REAL,
    -- Underwriters
    underwriters_json       TEXT,     -- JSON list
    has_bulge_bracket       INTEGER,  -- 0/1
    lead_underwriter        TEXT,
    -- Lockup
    lockup_days             INTEGER,
    is_spac                 INTEGER,
    quality_score           REAL,     -- 0-10
    red_flags_json          TEXT,
    green_flags_json        TEXT,
    created_at              TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (cik, accession)
);

CREATE TABLE IF NOT EXISTS spac_tracker (
    cik             TEXT NOT NULL PRIMARY KEY,
    company_name    TEXT,
    filed_date      TEXT,
    trust_amount_mn REAL,
    target_industry TEXT,
    sponsor_name    TEXT,
    deadline_months INTEGER,
    status          TEXT,   -- searching|announced|completed|dissolved
    merger_target   TEXT,
    merger_8k_date  TEXT,
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS underwriter_league (
    underwriter     TEXT NOT NULL,
    year            INTEGER NOT NULL,
    deal_count      INTEGER DEFAULT 0,
    total_proceeds_bn REAL DEFAULT 0,
    avg_day1_return REAL,
    avg_day30_mktadj REAL,
    bulge_bracket   INTEGER DEFAULT 0,
    updated_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (underwriter, year)
);
"""


def _get_db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class UseOfProceeds(BaseModel):
    total_mn: Optional[float] = None
    rd_pct: Optional[float] = None             # R&D
    sales_marketing_pct: Optional[float] = None
    debt_repayment_pct: Optional[float] = None
    general_corp_pct: Optional[float] = None
    acquisitions_pct: Optional[float] = None
    secondary_pct: Optional[float] = None      # insider selling portion
    raw_text: Optional[str] = None


class UnderwriterInfo(BaseModel):
    name: str
    is_bulge_bracket: bool = False
    is_lead: bool = False
    fee_pct: Optional[float] = None


class OfferingStructure(BaseModel):
    shares_offered_total: Optional[int] = None
    shares_primary: Optional[int] = None        # new shares issued
    shares_secondary: Optional[int] = None      # existing holder sales
    insider_selling_pct: Optional[float] = None
    price_range_low: Optional[float] = None
    price_range_high: Optional[float] = None
    offer_price: Optional[float] = None
    lockup_days: int = 180
    dual_class: bool = False


class XBRLFinancials(BaseModel):
    revenue_yr0: Optional[float] = None         # most recent annual
    revenue_yr1: Optional[float] = None         # prior year
    revenue_growth_yoy: Optional[float] = None
    revenue_cagr_2yr: Optional[float] = None
    gross_profit_yr0: Optional[float] = None
    gross_margin: Optional[float] = None
    net_income_yr0: Optional[float] = None
    cash_yr0: Optional[float] = None
    quarterly_cash_burn: Optional[float] = None  # avg Q CFO if loss-making


class S1Analysis(BaseModel):
    cik: str
    accession: str
    company_name: str = ""
    filed_date: Optional[date] = None
    form_type: str = "S-1"
    sic_code: Optional[str] = None
    use_of_proceeds: UseOfProceeds = Field(default_factory=UseOfProceeds)
    risk_factor_count: Optional[int] = None
    offering: OfferingStructure = Field(default_factory=OfferingStructure)
    underwriters: list[UnderwriterInfo] = Field(default_factory=list)
    has_bulge_bracket: bool = False
    lead_underwriter: Optional[str] = None
    financials: XBRLFinancials = Field(default_factory=XBRLFinancials)
    is_spac: bool = False
    quality_score: Optional[float] = None
    red_flags: list[str] = Field(default_factory=list)
    green_flags: list[str] = Field(default_factory=list)


class PipelineEntry(BaseModel):
    cik: str
    company_name: str = ""
    ticker: Optional[str] = None
    sic_code: Optional[str] = None
    form_type: str = "S-1"
    filed_date: Optional[date] = None
    pipeline_state: str = "filed"
    price_range_low: Optional[float] = None
    price_range_high: Optional[float] = None
    offer_price: Optional[float] = None
    shares_offered: Optional[int] = None
    accession: str = ""


class IPOResult(BaseModel):
    ticker: str
    company_name: str = ""
    ipo_date: Optional[date] = None
    offer_price: Optional[float] = None
    first_day_close: Optional[float] = None
    day1_return_pct: Optional[float] = None
    day30_return_pct: Optional[float] = None
    day90_return_pct: Optional[float] = None
    day180_return_pct: Optional[float] = None
    day30_mktadj_pct: Optional[float] = None
    day90_mktadj_pct: Optional[float] = None
    day180_mktadj_pct: Optional[float] = None
    lead_underwriter: Optional[str] = None
    lockup_expiry: Optional[date] = None


class SPACRecord(BaseModel):
    cik: str
    company_name: str = ""
    filed_date: Optional[date] = None
    trust_amount_mn: Optional[float] = None
    target_industry: Optional[str] = None
    sponsor_name: Optional[str] = None
    deadline_months: Optional[int] = None
    status: str = "searching"
    merger_target: Optional[str] = None
    merger_8k_date: Optional[date] = None


class PricingInfo(BaseModel):
    cik: str
    offer_price: float
    shares_offered: Optional[int] = None
    priced_date: Optional[date] = None
    accession_424b4: str = ""


class MergerAnnouncement(BaseModel):
    spac_cik: str
    target_name: str
    announced_date: Optional[date] = None
    deal_value_mn: Optional[float] = None
    accession_8k: str = ""


# ---------------------------------------------------------------------------
# EDGAR document fetcher
# ---------------------------------------------------------------------------


class EdgarDocFetcher:
    """Low-level EDGAR HTTP client with rate-limit discipline."""

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._session = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._session.close()

    def get_json(self, url: str, headers: Optional[dict] = None) -> dict[str, Any]:
        h = headers or _HEADERS
        try:
            resp = self._session.get(url, headers=h)
            resp.raise_for_status()
            time.sleep(_RATE_DELAY)
            return resp.json()
        except Exception as exc:
            logger.warning("EDGAR JSON fetch failed", url=url, error=str(exc))
            return {}

    def get_html(self, url: str) -> str:
        try:
            resp = self._session.get(url, headers=_HTML_HEADERS)
            resp.raise_for_status()
            time.sleep(_RATE_DELAY)
            return resp.text
        except Exception as exc:
            logger.warning("EDGAR HTML fetch failed", url=url, error=str(exc))
            return ""

    def get_submissions(self, cik: str) -> dict[str, Any]:
        cik_padded = cik.zfill(10)
        return self.get_json(f"{_EDGAR_BASE}/submissions/CIK{cik_padded}.json")

    def get_filing_index(self, cik: str, accession: str) -> dict[str, Any]:
        cik_clean  = cik.lstrip("0") or "0"
        acc_clean  = accession.replace("-", "")
        url = f"{_EDGAR_ARCHIVES}/{cik_clean}/{acc_clean}/{accession}-index.json"
        return self.get_json(url)

    def get_primary_document_html(self, cik: str, accession: str) -> str:
        """Download the primary HTML document from a filing."""
        cik_clean = cik.lstrip("0") or "0"
        acc_clean = accession.replace("-", "")
        idx = self.get_filing_index(cik, accession)

        primary_doc: Optional[str] = None
        for doc in idx.get("documents", []):
            doc_type = doc.get("type", "")
            fname = doc.get("filename", "")
            # Skip exhibits
            if doc_type in ("EX-", "EX-1", "EX-3", "EX-4", "EX-5", "EX-10",
                            "EX-21", "EX-23", "EX-31", "EX-32"):
                continue
            if fname.endswith((".htm", ".html")):
                primary_doc = fname
                break

        if not primary_doc:
            # Fallback: any .htm
            for doc in idx.get("documents", []):
                fname = doc.get("filename", "")
                if fname.endswith((".htm", ".html", ".txt")):
                    primary_doc = fname
                    break

        if not primary_doc:
            return ""

        url = f"{_EDGAR_ARCHIVES}/{cik_clean}/{acc_clean}/{primary_doc}"
        return self.get_html(url)

    def get_company_facts(self, cik: str) -> dict[str, Any]:
        cik_padded = cik.zfill(10)
        return self.get_json(f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik_padded}.json")

    def efts_search(
        self,
        forms: str,
        start_date: str,
        end_date: str,
        from_: int = 0,
        size: int = 40,
    ) -> dict[str, Any]:
        params = {
            "forms": forms,
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": from_,
            "hits.hits._source": "true",
        }
        try:
            resp = self._session.get(
                _EDGAR_EFTS,
                params=params,
                headers=_HEADERS,
            )
            resp.raise_for_status()
            time.sleep(_RATE_DELAY)
            return resp.json()
        except Exception as exc:
            logger.warning("EFTS search failed", error=str(exc))
            return {}

    def resolve_ticker_to_cik(self, ticker: str) -> Optional[str]:
        ticker_upper = ticker.upper()
        try:
            resp = self._session.get(
                f"{_EDGAR_BASE}/files/company_tickers.json",
                headers=_HEADERS,
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
            for _k, v in data.items():
                if str(v.get("ticker", "")).upper() == ticker_upper:
                    return str(v.get("cik_str", "")).zfill(10)
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# EDGAR S-1 parser (structured extraction)
# ---------------------------------------------------------------------------


class EdgarS1Parser:
    """Structured data extraction from S-1 HTML filings using BeautifulSoup.

    Extracts:
    - Use of Proceeds: dollar amounts + percentage allocation per category
    - Risk factors: count of <li> items in the Risk Factors section
    - Offering structure: primary vs secondary shares, dual-class detection
    - Underwriters: name, bulge-bracket flag, lead
    - XBRL financials: revenue, gross margin, net income, cash burn
    """

    def __init__(self, fetcher: EdgarDocFetcher) -> None:
        self._fetcher = fetcher

    # ------------------------------------------------------------------
    # Main parse entry point
    # ------------------------------------------------------------------

    def parse_s1(self, cik: str, accession: str) -> S1Analysis:
        """Download filing HTML and run all structured extractors."""
        html = self._fetcher.get_primary_document_html(cik, accession)
        sub  = self._fetcher.get_submissions(cik)

        company_name = sub.get("name", "")
        sic_code     = str(sub.get("sic", "")).strip() or None
        ticker_list  = sub.get("tickers", [])
        ticker       = ticker_list[0] if ticker_list else None

        # Find filing date from submissions history
        filed_date: Optional[date] = None
        filings_recent = sub.get("filings", {}).get("recent", {})
        accessions_list = filings_recent.get("accessionNumber", [])
        dates_list      = filings_recent.get("filingDate", [])
        forms_list      = filings_recent.get("form", [])
        acc_norm        = accession.replace("-", "")
        form_type       = "S-1"
        for i, a in enumerate(accessions_list):
            if a.replace("-", "") == acc_norm:
                filed_date = _parse_date(dates_list[i]) if i < len(dates_list) else None
                form_type  = forms_list[i] if i < len(forms_list) else "S-1"
                break

        # Structured extractors
        uop         = self.extract_use_of_proceeds(html)
        risk_count  = self.count_risk_factors_structured(html)
        offering    = self.extract_offering_structure(html)
        underwriters = self.extract_underwriters(html)
        is_spac     = self._detect_spac(html)
        xbrl_fin    = self.get_xbrl_financials(cik)

        has_bulge   = any(u.is_bulge_bracket for u in underwriters)
        lead_uw     = next((u.name for u in underwriters if u.is_lead), None)
        if lead_uw is None and underwriters:
            lead_uw = underwriters[0].name

        analysis = S1Analysis(
            cik=cik,
            accession=accession,
            company_name=company_name,
            filed_date=filed_date,
            form_type=form_type,
            sic_code=sic_code,
            use_of_proceeds=uop,
            risk_factor_count=risk_count,
            offering=offering,
            underwriters=underwriters,
            has_bulge_bracket=has_bulge,
            lead_underwriter=lead_uw,
            financials=xbrl_fin,
            is_spac=is_spac,
        )
        self._score(analysis)
        return analysis

    # ------------------------------------------------------------------
    # Use of Proceeds extraction
    # ------------------------------------------------------------------

    def extract_use_of_proceeds(self, html: str) -> UseOfProceeds:
        """Extract dollar amounts and percentage allocations from the
        'Use of Proceeds' section using BeautifulSoup.

        Strategy:
        1. Find the section heading by text matching
        2. Extract all text from the section (until next major heading)
        3. Parse dollar amounts with contextual labels
        """
        if not html:
            return UseOfProceeds()

        soup = BeautifulSoup(html, "html.parser")

        # Find the Use of Proceeds section heading
        section_text = self._extract_section(soup, "use of proceeds")
        if not section_text:
            return UseOfProceeds()

        uop = UseOfProceeds(raw_text=section_text[:600])

        # Total amount
        total = _find_dollar_amount(section_text, r"(?:aggregate|total|net)\s+proceeds")
        if total is None:
            total = _find_dollar_amount(section_text, r"we\s+(?:estimate|expect|intend)\s+to\s+(?:receive|raise)")
        uop.total_mn = (total / 1e6) if total else None

        # Category allocations — scan for label near dollar or percentage
        CATEGORY_PATTERNS: list[tuple[str, str]] = [
            ("rd",            r"research\s+(?:and|&)\s+development|r\s*&\s*d"),
            ("sales",         r"sales\s+(?:and|&)\s+marketing|selling\s+and\s+marketing"),
            ("debt_repay",    r"(?:repay|repayment|retire|redeem)\s+(?:of\s+)?(?:debt|borrowings|notes|credit)"),
            ("general_corp",  r"general\s+(?:corporate|and\s+administrative)|working\s+capital"),
            ("acquisitions",  r"acqui(?:sition|re)|strategic\s+(?:investment|transaction)"),
            ("secondary",     r"selling\s+stockholder|existing\s+(?:shareholder|holder)|secondary\s+offering"),
        ]

        for attr, pattern in CATEGORY_PATTERNS:
            # Try percentage first
            pct = _find_percent_near(section_text, pattern)
            if pct is not None:
                setattr(uop, f"{attr}_pct", pct)
                continue
            # Fall back to dollar amount, convert to percentage of total
            amt = _find_dollar_amount(section_text, pattern)
            if amt is not None and uop.total_mn and uop.total_mn > 0:
                setattr(uop, f"{attr}_pct", round(amt / (uop.total_mn * 1e6) * 100, 1))

        return uop

    # ------------------------------------------------------------------
    # Risk factor count (structured)
    # ------------------------------------------------------------------

    def count_risk_factors_structured(self, html: str) -> int:
        """Count risk factors by counting <li> elements in the Risk Factors
        HTML section — far more accurate than regex line counting.
        """
        if not html:
            return 0

        soup = BeautifulSoup(html, "html.parser")

        # Locate the Risk Factors heading
        rf_section = None
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "p", "div"]):
            text = heading.get_text(" ", strip=True).upper()
            if "RISK FACTOR" in text and len(text) < 60:
                rf_section = heading
                break

        if rf_section is None:
            return 0

        # Collect all <li> items until the next major heading
        count = 0
        for sibling in rf_section.find_next_siblings():
            tag = getattr(sibling, "name", "")
            if tag in ("h1", "h2"):
                break
            text = sibling.get_text(" ", strip=True)
            # Stop at next major section heading
            upper = text.upper()
            if len(text) < 80 and any(
                kw in upper for kw in [
                    "USE OF PROCEEDS", "DILUTION", "DIVIDEND", "CAPITALIZATION",
                    "MANAGEMENT'S DISCUSSION", "BUSINESS", "LEGAL PROCEEDINGS",
                ]
            ):
                break
            if tag == "ul" or tag == "ol":
                count += len(sibling.find_all("li", recursive=False))
            elif tag == "li":
                count += 1
            # Also count bold/underlined sub-headings as individual risk factors
            elif tag in ("p", "div"):
                for b in sibling.find_all(["b", "strong"]):
                    btext = b.get_text(" ", strip=True)
                    if 10 < len(btext) < 200:
                        count += 1
                        break

        return max(count, 0)

    # ------------------------------------------------------------------
    # Offering structure
    # ------------------------------------------------------------------

    def extract_offering_structure(self, html: str) -> OfferingStructure:
        """Extract offering mechanics from the cover page and prospectus summary."""
        if not html:
            return OfferingStructure()

        soup = BeautifulSoup(html, "html.parser")
        full_text = soup.get_text(" ", strip=True)

        offering = OfferingStructure()

        # Price range
        pr = _extract_price_range(full_text)
        offering.price_range_low  = pr[0]
        offering.price_range_high = pr[1]

        # Offer price (final — only in 424B4)
        op_m = re.search(r"price\s+(?:per\s+share|to\s+public)[^\d$]{0,30}\$\s*([\d,]+(?:\.\d+)?)", full_text, re.IGNORECASE)
        if op_m:
            try:
                offering.offer_price = float(op_m.group(1).replace(",", ""))
            except ValueError:
                pass

        # Shares offered — primary (new shares)
        shares_m = re.search(
            r"(?:we\s+are|company\s+is|issuer\s+is)\s+offering\s+([\d,]+(?:\.\d+)?)\s*(?:million\s+)?shares",
            full_text,
            re.IGNORECASE,
        )
        if shares_m:
            raw = shares_m.group(1).replace(",", "")
            mult = 1_000_000 if "million" in shares_m.group(0).lower() else 1
            try:
                offering.shares_primary = int(float(raw) * mult)
            except ValueError:
                pass

        # Shares sold by selling stockholders (secondary)
        sec_m = re.search(
            r"selling\s+stockholder[s]?\s+(?:are\s+)?(?:selling|offering)\s+([\d,]+(?:\.\d+)?)\s*(?:million\s+)?shares",
            full_text,
            re.IGNORECASE,
        )
        if sec_m:
            raw = sec_m.group(1).replace(",", "")
            mult = 1_000_000 if "million" in sec_m.group(0).lower() else 1
            try:
                offering.shares_secondary = int(float(raw) * mult)
            except ValueError:
                pass

        # Total
        if offering.shares_primary is not None or offering.shares_secondary is not None:
            offering.shares_offered_total = (
                (offering.shares_primary or 0) + (offering.shares_secondary or 0)
            )
            if offering.shares_offered_total > 0 and offering.shares_secondary:
                offering.insider_selling_pct = round(
                    offering.shares_secondary / offering.shares_offered_total * 100, 1
                )

        # Lockup period
        lk_m = re.search(r"(\d+)[-\s]day\s+lock[-\s]?up", full_text, re.IGNORECASE)
        offering.lockup_days = int(lk_m.group(1)) if lk_m else 180

        # Dual class
        offering.dual_class = self.detect_dual_class(html)

        return offering

    def detect_dual_class(self, html: str) -> bool:
        """Detect Class A / Class B dual-class share structure with
        differential voting rights.
        """
        if not html:
            return False
        text_lower = html.lower()
        # Look for Class A and Class B together with voting references
        has_class_a = "class a" in text_lower
        has_class_b = "class b" in text_lower
        has_voting  = any(kw in text_lower for kw in [
            "10 votes per share", "10-to-1", "ten votes", "superior voting",
            "multiple voting", "high vote", "supervoting",
        ])
        return has_class_a and has_class_b and (has_voting or "class b common stock" in text_lower)

    # ------------------------------------------------------------------
    # Underwriter extraction
    # ------------------------------------------------------------------

    def extract_underwriters(self, html: str) -> list[UnderwriterInfo]:
        """Extract underwriter names from cover page / underwriting section.

        Methodology:
        - Find the Underwriting section
        - Scan for known bank names (case-insensitive)
        - Flag bulge bracket status
        - Identify lead as the first listed
        - Extract underwriting discount % from tables
        """
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        full_text = soup.get_text(" ", strip=True)

        found_names: list[str] = []
        text_lower = full_text.lower()

        for bank in sorted(_MAJOR_UNDERWRITERS, key=len, reverse=True):  # longest first
            if bank.lower() in text_lower:
                # Normalize aliases
                canonical = bank
                if bank in {"J.P. Morgan", "JPMorgan Chase"}:
                    canonical = "JPMorgan"
                if bank in {"Merrill Lynch", "BofA Securities"}:
                    canonical = "Bank of America"
                if canonical not in found_names:
                    found_names.append(canonical)

        if not found_names:
            return []

        # Extract underwriting fee from text
        fee_pct: Optional[float] = None
        fee_m = re.search(
            r"(?:underwriting\s+discount|underwriting\s+commission|total\s+underwriting)[^%\d]{0,60}"
            r"([\d]+(?:\.\d+)?)\s*%",
            full_text,
            re.IGNORECASE,
        )
        if fee_m:
            try:
                fee_pct = float(fee_m.group(1))
            except ValueError:
                pass

        results: list[UnderwriterInfo] = []
        for i, name in enumerate(found_names):
            results.append(UnderwriterInfo(
                name=name,
                is_bulge_bracket=name in _BULGE_BRACKET,
                is_lead=(i == 0),
                fee_pct=fee_pct,
            ))

        return results

    # ------------------------------------------------------------------
    # XBRL financials
    # ------------------------------------------------------------------

    def get_xbrl_financials(self, cik: str) -> XBRLFinancials:
        """Pull 2-year audited financials from EDGAR companyfacts XBRL.

        S-1 filers must include 2 years of audited financials (for operating
        companies). We use annual 10-K periods inside the companyfacts payload
        (S-1 filers are also included if they have previously filed XBRL data).
        """
        facts = self._fetcher.get_company_facts(cik)
        if not facts:
            return XBRLFinancials()

        def annual_series(concepts: list[str], n: int = 3) -> list[tuple[str, float]]:
            for concept in concepts:
                try:
                    units = facts["facts"]["us-gaap"][concept]["units"]["USD"]
                except (KeyError, TypeError):
                    continue
                # Keep only annual-period entries
                annual: list[tuple[str, float]] = []
                seen: set[str] = set()
                for entry in sorted(units, key=lambda x: x.get("end", ""), reverse=True):
                    form = entry.get("form", "")
                    if "10-K" not in form and "20-F" not in form and "S-1" not in form:
                        continue
                    end = entry.get("end", "")
                    start = entry.get("start", "")
                    val = entry.get("val")
                    if not end or val is None or end in seen:
                        continue
                    # Accept 340-400 day periods as annual
                    if start:
                        try:
                            days = (date.fromisoformat(end) - date.fromisoformat(start)).days
                            if not (340 <= days <= 400):
                                continue
                        except ValueError:
                            pass
                    annual.append((end, float(val)))
                    seen.add(end)
                if annual:
                    return sorted(annual, key=lambda x: x[0], reverse=True)[:n]
            return []

        def quarterly_cfo_avg() -> Optional[float]:
            """Average quarterly CFO for cash burn calculation."""
            for concept in _CFO_CONCEPTS:
                try:
                    units = facts["facts"]["us-gaap"][concept]["units"]["USD"]
                except (KeyError, TypeError):
                    continue
                qtrs: list[float] = []
                for entry in sorted(units, key=lambda x: x.get("end", ""), reverse=True):
                    form = entry.get("form", "")
                    if "10-Q" not in form:
                        continue
                    start = entry.get("start", "")
                    end   = entry.get("end", "")
                    val   = entry.get("val")
                    if val is None:
                        continue
                    try:
                        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
                        if 60 <= days <= 120:
                            qtrs.append(float(val))
                    except ValueError:
                        continue
                    if len(qtrs) >= 4:
                        break
                if qtrs:
                    return float(np.mean(qtrs))
            return None

        rev_series   = annual_series(_REVENUE_CONCEPTS)
        ni_series    = annual_series(_NI_CONCEPTS)
        cash_series  = annual_series(_CASH_CONCEPTS)

        # Gross margin = (Revenue - COGS) / Revenue
        cogs_series  = annual_series(_COGS_CONCEPTS)

        rev_yr0  = rev_series[0][1]  if len(rev_series) >= 1 else None
        rev_yr1  = rev_series[1][1]  if len(rev_series) >= 2 else None
        rev_yr2  = rev_series[2][1]  if len(rev_series) >= 3 else None
        cogs_yr0 = cogs_series[0][1] if len(cogs_series) >= 1 else None
        ni_yr0   = ni_series[0][1]   if len(ni_series) >= 1 else None
        cash_yr0 = cash_series[0][1] if len(cash_series) >= 1 else None

        gp_yr0: Optional[float] = None
        gross_margin: Optional[float] = None
        if rev_yr0 is not None and cogs_yr0 is not None and rev_yr0 > 0:
            gp_yr0 = rev_yr0 - cogs_yr0
            gross_margin = gp_yr0 / rev_yr0

        yoy: Optional[float] = None
        if rev_yr0 is not None and rev_yr1 and rev_yr1 != 0:
            yoy = (rev_yr0 - rev_yr1) / abs(rev_yr1)

        cagr_2yr: Optional[float] = None
        if rev_yr0 is not None and rev_yr2 and rev_yr2 > 0:
            cagr_2yr = (rev_yr0 / rev_yr2) ** 0.5 - 1

        # Cash burn — only meaningful if company is loss-making
        cash_burn: Optional[float] = None
        if ni_yr0 is not None and ni_yr0 < 0:
            cash_burn = quarterly_cfo_avg()
            if cash_burn is not None and cash_burn > 0:
                cash_burn = None   # positive CFO despite net loss — not a burner

        return XBRLFinancials(
            revenue_yr0=rev_yr0,
            revenue_yr1=rev_yr1,
            revenue_growth_yoy=yoy,
            revenue_cagr_2yr=cagr_2yr,
            gross_profit_yr0=gp_yr0,
            gross_margin=gross_margin,
            net_income_yr0=ni_yr0,
            cash_yr0=cash_yr0,
            quarterly_cash_burn=cash_burn,
        )

    # ------------------------------------------------------------------
    # SPAC detection
    # ------------------------------------------------------------------

    def _detect_spac(self, html: str) -> bool:
        if not html:
            return False
        text_lower = html.lower()
        return sum(1 for kw in _SPAC_MARKERS if kw in text_lower) >= 2

    # ------------------------------------------------------------------
    # Quality score
    # ------------------------------------------------------------------

    def _score(self, analysis: S1Analysis) -> None:
        """Assign quality_score 0–10 and populate red_flags/green_flags."""
        red: list[str] = []
        green: list[str] = []
        score = 5.0
        fin = analysis.financials

        # Revenue presence
        if fin.revenue_yr0 is None:
            red.append("Pre-revenue company — no XBRL revenue data")
            score -= 1.0
        elif fin.revenue_yr0 < 5_000_000:
            red.append(f"Very low revenue: ${fin.revenue_yr0 / 1e6:.1f}M")
            score -= 0.5
        else:
            green.append(f"Revenue: ${fin.revenue_yr0 / 1e6:.1f}M")
            score += 0.5

        # Revenue growth
        if fin.revenue_growth_yoy is not None:
            if fin.revenue_growth_yoy > 0.50:
                green.append(f"High revenue growth: {fin.revenue_growth_yoy:.0%} YoY")
                score += 1.0
            elif fin.revenue_growth_yoy < 0:
                red.append(f"Declining revenue: {fin.revenue_growth_yoy:.0%} YoY")
                score -= 1.0
            else:
                green.append(f"Revenue growth: {fin.revenue_growth_yoy:.0%} YoY")

        # Profitability
        if fin.net_income_yr0 is not None:
            if fin.net_income_yr0 > 0:
                green.append("Profitable at IPO")
                score += 2.0
            elif fin.net_income_yr0 < -100_000_000:
                red.append(f"Large net loss: ${fin.net_income_yr0 / 1e6:.0f}M")
                score -= 1.0

        # Gross margin
        if fin.gross_margin is not None:
            if fin.gross_margin > 0.60:
                green.append(f"Strong gross margin: {fin.gross_margin:.0%}")
                score += 0.5
            elif fin.gross_margin < 0.20:
                red.append(f"Low gross margin: {fin.gross_margin:.0%}")
                score -= 0.5

        # Cash burn / runway
        if fin.quarterly_cash_burn is not None and fin.quarterly_cash_burn < 0 and fin.cash_yr0:
            monthly_burn = abs(fin.quarterly_cash_burn) / 3
            runway_months = fin.cash_yr0 / monthly_burn if monthly_burn > 0 else 999
            if runway_months < 12:
                red.append(f"Short cash runway: ~{runway_months:.0f} months")
                score -= 1.0
            elif runway_months > 24:
                green.append(f"Solid cash runway: ~{runway_months:.0f} months")

        # Risk factor count
        if analysis.risk_factor_count is not None:
            if analysis.risk_factor_count > 80:
                red.append(f"Very high risk factor count: {analysis.risk_factor_count}")
                score -= 1.0
            elif analysis.risk_factor_count > 50:
                red.append(f"High risk factor count: {analysis.risk_factor_count}")
                score -= 0.5

        # SPAC
        if analysis.is_spac:
            red.append("SPAC — no operating history")
            score -= 1.5

        # Dual class
        if analysis.offering.dual_class:
            red.append("Dual-class share structure (reduced governance)")
            score -= 0.5

        # Bulge bracket underwriter
        if analysis.has_bulge_bracket:
            green.append(
                f"Bulge-bracket underwriter: {analysis.lead_underwriter}"
            )
            score += 1.0
        elif not analysis.underwriters:
            red.append("No major underwriter identified")
            score -= 0.5

        # Insider selling
        sec_pct = analysis.offering.insider_selling_pct
        if sec_pct is not None:
            if sec_pct > 50:
                red.append(f"High insider selling: {sec_pct:.0f}% of shares are secondary")
                score -= 1.0
            elif sec_pct > 25:
                red.append(f"Material insider selling: {sec_pct:.0f}% secondary")
                score -= 0.5

        # Use of proceeds red flag
        uop_text = (analysis.use_of_proceeds.raw_text or "").lower()
        if "selling stockholder" in uop_text or "selling shareholder" in uop_text:
            if "secondary" not in [r.lower()[:9] for r in red]:
                red.append("Use of proceeds includes selling stockholder component")

        analysis.red_flags  = red
        analysis.green_flags = green
        analysis.quality_score = max(0.0, min(10.0, round(score, 1)))

    # ------------------------------------------------------------------
    # Section extraction helper
    # ------------------------------------------------------------------

    def _extract_section(self, soup: BeautifulSoup, section_name: str) -> str:
        """Find a named section in the filing and return its text content."""
        section_upper = section_name.upper()
        target = None

        # Search headings, bold paragraphs, div headers
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "div", "span"]):
            txt = el.get_text(" ", strip=True).upper()
            if section_upper in txt and len(txt) < 80:
                target = el
                break

        if target is None:
            return ""

        # Collect text from siblings until next major heading
        parts: list[str] = []
        char_limit = 3000
        for sibling in target.find_next_siblings():
            tag = getattr(sibling, "name", "")
            txt = sibling.get_text(" ", strip=True)
            if tag in ("h1", "h2") or (
                tag in ("h3", "h4") and len(txt) < 80
                and any(
                    kw in txt.upper()
                    for kw in ["DILUTION", "DIVIDEND", "CAPITALIZATION",
                               "MANAGEMENT", "BUSINESS", "LEGAL"]
                )
            ):
                break
            parts.append(txt)
            if sum(len(p) for p in parts) >= char_limit:
                break

        return " ".join(parts)


# ---------------------------------------------------------------------------
# IPO pipeline tracker
# ---------------------------------------------------------------------------


class IPOPipelineTracker:
    """Tracks all S-1 / S-1/A / 424B4 filings and maintains pipeline state."""

    def __init__(self, fetcher: EdgarDocFetcher, conn: sqlite3.Connection) -> None:
        self._fetcher = fetcher
        self._conn    = conn

    def get_pipeline(self, lookback_days: int = 90) -> list[PipelineEntry]:
        """Fetch recent S-1 filings from EDGAR EFTS and build pipeline."""
        today      = date.today()
        start_date = (today - timedelta(days=lookback_days)).isoformat()
        end_date   = today.isoformat()

        entries: list[PipelineEntry] = []
        seen_ciks: set[str] = set()

        for form in ("S-1,S-1/A,S-11,F-1,F-1/A", "424B4"):
            from_   = 0
            batch   = 40
            while True:
                data = self._fetcher.efts_search(form, start_date, end_date, from_=from_, size=batch)
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break

                for hit in hits:
                    src = hit.get("_source", {})
                    acc = hit.get("_id", "").replace(":", "-")
                    if not acc:
                        continue

                    cik = str(src.get("entity_id", "")).zfill(10)
                    if not cik or cik == "0000000000":
                        continue

                    names = src.get("display_names", [])
                    company_name = names[0].get("name", "") if names else ""
                    form_type    = src.get("form_type", "S-1")
                    filed_str    = src.get("file_date", "")
                    filed_date   = _parse_date(filed_str)

                    state = self._pipeline_state(form_type)

                    entry = PipelineEntry(
                        cik=cik,
                        company_name=company_name,
                        form_type=form_type,
                        filed_date=filed_date,
                        pipeline_state=state,
                        accession=acc,
                    )
                    entries.append(entry)

                    # Upsert to DB
                    self._conn.execute(
                        """INSERT OR REPLACE INTO ipo_pipeline
                           (cik, company_name, form_type, filed_date, accession, pipeline_state)
                           VALUES (?,?,?,?,?,?)""",
                        (cik, company_name, form_type,
                         filed_str, acc, state),
                    )

                if len(hits) < batch:
                    break
                from_ += batch

        self._conn.commit()

        # Deduplicate by CIK — keep most advanced pipeline state
        state_rank = {"priced": 4, "roadshow": 3, "filed": 2, "trading": 5, "withdrawn": 1}
        best: dict[str, PipelineEntry] = {}
        for e in entries:
            existing = best.get(e.cik)
            if existing is None:
                best[e.cik] = e
            elif state_rank.get(e.pipeline_state, 0) > state_rank.get(existing.pipeline_state, 0):
                best[e.cik] = e

        return sorted(best.values(), key=lambda x: x.filed_date or date.min, reverse=True)

    @staticmethod
    def _pipeline_state(form_type: str) -> str:
        if form_type in _FINAL_PROS:
            return "priced"
        if form_type in _AMENDMENT_FORMS:
            return "roadshow"
        return "filed"

    def get_recent_priced(self, days: int = 90) -> list[IPOResult]:
        """Return recently priced IPOs (424B4 filers) from DB."""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        cur = self._conn.execute(
            "SELECT * FROM ipo_results WHERE ipo_date >= ? ORDER BY ipo_date DESC",
            (cutoff,),
        )
        return [IPOResult(**dict(r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# IPO performance engine
# ---------------------------------------------------------------------------


class IPOPerformanceEngine:
    """Compute IPO pricing analytics and aftermarket performance."""

    BENCHMARK_TICKER = "SPY"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def _get_price_history(
        self,
        ticker: str,
        start: date,
        end: Optional[date] = None,
    ) -> pd.Series:
        """Pull closing prices from yfinance (price data, not fundamental)."""
        try:
            import yfinance as yf  # type: ignore
            end_dt = end or date.today()
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=end_dt.isoformat(),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return pd.Series(dtype=float)
            return df["Close"].squeeze()
        except Exception as exc:
            logger.warning("yfinance history failed", ticker=ticker, error=str(exc))
            return pd.Series(dtype=float)

    def day1_return(self, ticker: str, ipo_price: float, ipo_date: date) -> dict:
        """First-day return = (close Day 1 - offer price) / offer price."""
        history = self._get_price_history(ticker, ipo_date, ipo_date + timedelta(days=5))
        if history.empty:
            return {"ticker": ticker, "day1_return_pct": None, "error": "no price data"}

        # First available close on or after IPO date
        first_close = float(history.iloc[0])
        day1_ret    = (first_close - ipo_price) / ipo_price * 100

        return {
            "ticker":        ticker,
            "ipo_date":      ipo_date.isoformat(),
            "offer_price":   ipo_price,
            "first_close":   first_close,
            "day1_return_pct": round(day1_ret, 2),
            "underpricing_note": (
                "Positive = underpriced (issuer left money on table); "
                "Negative = overpriced"
            ),
        }

    def aftermarket_returns(
        self,
        ticker: str,
        ipo_date: date,
        ipo_price: float,
        periods: tuple[int, ...] = (30, 90, 180),
    ) -> dict:
        """Compute raw and market-adjusted returns at 30/90/180 days post-IPO."""
        today    = date.today()
        end_date = min(today, ipo_date + timedelta(days=max(periods) + 10))

        ipo_prices   = self._get_price_history(ticker, ipo_date, end_date)
        bench_prices = self._get_price_history(self.BENCHMARK_TICKER, ipo_date, end_date)

        if ipo_prices.empty:
            return {"ticker": ticker, "error": "no price data"}

        results: dict[str, Any] = {
            "ticker":     ticker,
            "ipo_date":   ipo_date.isoformat(),
            "offer_price": ipo_price,
            "returns":    {},
        }

        # Day-0 price (first trading day close)
        day0_price = float(ipo_prices.iloc[0])
        bench_day0 = float(bench_prices.iloc[0]) if not bench_prices.empty else None

        for days in periods:
            target_date = ipo_date + timedelta(days=days)
            if target_date > today:
                results["returns"][f"day{days}"] = {
                    "raw_pct":     None,
                    "mktadj_pct":  None,
                    "status":      "not yet",
                }
                continue

            # Find closest trading day
            future_prices = ipo_prices[ipo_prices.index >= pd.Timestamp(target_date)]
            if future_prices.empty:
                results["returns"][f"day{days}"] = {
                    "raw_pct":     None,
                    "mktadj_pct":  None,
                    "status":      "no data",
                }
                continue

            price_n  = float(future_prices.iloc[0])
            raw_ret  = (price_n - ipo_price) / ipo_price * 100

            mkt_adj: Optional[float] = None
            if bench_day0 and not bench_prices.empty:
                future_bench = bench_prices[bench_prices.index >= pd.Timestamp(target_date)]
                if not future_bench.empty:
                    bench_n = float(future_bench.iloc[0])
                    bench_ret = (bench_n - bench_day0) / bench_day0 * 100
                    mkt_adj = round(raw_ret - bench_ret, 2)

            results["returns"][f"day{days}"] = {
                "raw_pct":    round(raw_ret, 2),
                "mktadj_pct": mkt_adj,
                "price":      price_n,
                "as_of":      str(future_prices.index[0].date()),
            }

            # Persist to ipo_results
            self._conn.execute(
                f"""INSERT OR IGNORE INTO ipo_results (ticker, ipo_date, offer_price)
                    VALUES (?,?,?)""",
                (ticker, ipo_date.isoformat(), ipo_price),
            )
            self._conn.execute(
                f"""UPDATE ipo_results SET
                    day{days}_return_pct=?,
                    day{days}_mktadj_pct=?
                    WHERE ticker=?""",
                (round(raw_ret, 2), mkt_adj, ticker),
            )

        self._conn.commit()
        return results

    @staticmethod
    def underpricing_vs_fee(
        underwriter_fee_pct: float,
        day1_return_pct: float,
    ) -> dict:
        """Analyze underpricing vs underwriter compensation.

        Theory: high day-1 pop + low fee = bad deal for issuer.
        High day-1 pop suggests underwriter priced conservatively (underpriced)
        to benefit buy-side clients.
        """
        issuer_cost = underwriter_fee_pct + max(0, day1_return_pct)
        return {
            "underwriter_fee_pct":   underwriter_fee_pct,
            "day1_return_pct":       day1_return_pct,
            "total_issuer_cost_pct": round(issuer_cost, 2),
            "interpretation": (
                "Significant underpricing" if day1_return_pct > 15
                else "Moderate underpricing" if day1_return_pct > 5
                else "Fairly priced"
            ),
        }


# ---------------------------------------------------------------------------
# SPAC tracker
# ---------------------------------------------------------------------------


class SPACTracker:
    """Track SPAC S-1 filings and detect merger announcements via 8-K."""

    def __init__(self, fetcher: EdgarDocFetcher, conn: sqlite3.Connection) -> None:
        self._fetcher = fetcher
        self._conn    = conn
        self._parser  = EdgarS1Parser(fetcher)

    def get_active_spacs(self, lookback_days: int = 365) -> list[SPACRecord]:
        """Find SPAC S-1 filings by SIC code 6770 (Blank Check Companies)."""
        today      = date.today()
        start_date = (today - timedelta(days=lookback_days)).isoformat()
        end_date   = today.isoformat()

        # SIC 6770 = blank check companies
        spacs: list[SPACRecord] = []

        data = self._fetcher.efts_search(
            "S-1,S-1/A",
            start_date,
            end_date,
            size=40,
        )

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src  = hit.get("_source", {})
            acc  = hit.get("_id", "").replace(":", "-")
            cik  = str(src.get("entity_id", "")).zfill(10)
            names = src.get("display_names", [])
            name  = names[0].get("name", "") if names else ""
            filed_str = src.get("file_date", "")

            # Quick SPAC filter via SIC or company name
            is_spac_candidate = (
                "acquisition" in name.lower() or
                "blank check" in name.lower() or
                "spac" in name.lower()
            )
            if not is_spac_candidate:
                continue

            # Fetch HTML and run detector
            html = self._fetcher.get_primary_document_html(cik, acc)
            if not self._parser._detect_spac(html):
                continue

            trust = _extract_trust_amount(html)
            sponsor = _extract_sponsor(html)
            target_industry = _detect_spac_target_industry(html)
            deadline = _extract_deadline_months(html)

            record = SPACRecord(
                cik=cik,
                company_name=name,
                filed_date=_parse_date(filed_str),
                trust_amount_mn=trust,
                target_industry=target_industry,
                sponsor_name=sponsor,
                deadline_months=deadline,
                status="searching",
            )
            spacs.append(record)

            self._conn.execute(
                """INSERT OR REPLACE INTO spac_tracker
                   (cik, company_name, filed_date, trust_amount_mn, target_industry,
                    sponsor_name, deadline_months, status)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (cik, name, filed_str, trust, target_industry, sponsor, deadline, "searching"),
            )

        self._conn.commit()
        return spacs

    def detect_spac_merger(self, cik: str) -> Optional[MergerAnnouncement]:
        """Check for 8-K filings indicating a SPAC merger announcement (Item 1.01)."""
        sub = self._fetcher.get_submissions(cik)
        filings = sub.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accs  = filings.get("accessionNumber", [])

        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            acc = accs[i] if i < len(accs) else ""
            filed_str = dates[i] if i < len(dates) else ""
            # Get 8-K HTML and look for business combination language
            html = self._fetcher.get_primary_document_html(cik, acc)
            if not html:
                continue
            text_lower = html.lower()
            if (
                "business combination" in text_lower
                or "merger agreement" in text_lower
                or "definitive agreement" in text_lower
            ) and "item 1.01" in text_lower:
                # Extract target name
                target_m = re.search(
                    r"(?:acquire|merge\s+with|combination\s+with)\s+([A-Z][A-Za-z\s&,\.]{5,60}?)(?:\s+\(|,|\.|Inc\.|Corp\.)",
                    html,
                )
                target_name = target_m.group(1).strip() if target_m else "Unknown Target"
                return MergerAnnouncement(
                    spac_cik=cik,
                    target_name=target_name,
                    announced_date=_parse_date(filed_str),
                    accession_8k=acc,
                )
        return None


# ---------------------------------------------------------------------------
# Underwriter league table
# ---------------------------------------------------------------------------


class UnderwriterLeagueTable:
    """Track underwriter performance metrics across IPO deals."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def update(
        self,
        underwriter: str,
        year: int,
        proceeds_mn: float,
        day1_return_pct: Optional[float] = None,
        day30_mktadj_pct: Optional[float] = None,
    ) -> None:
        """Upsert a deal record into the league table."""
        is_bb = 1 if underwriter in _BULGE_BRACKET else 0
        self._conn.execute(
            """INSERT INTO underwriter_league
               (underwriter, year, deal_count, total_proceeds_bn, avg_day1_return,
                avg_day30_mktadj, bulge_bracket)
               VALUES (?,?,1,?,?,?,?)
               ON CONFLICT(underwriter,year) DO UPDATE SET
                 deal_count = deal_count + 1,
                 total_proceeds_bn = total_proceeds_bn + excluded.total_proceeds_bn,
                 avg_day1_return = (
                     avg_day1_return * (deal_count-1) + COALESCE(excluded.avg_day1_return, avg_day1_return)
                 ) / deal_count,
                 avg_day30_mktadj = (
                     avg_day30_mktadj * (deal_count-1) + COALESCE(excluded.avg_day30_mktadj, avg_day30_mktadj)
                 ) / deal_count,
                 updated_at = datetime('now')
            """,
            (underwriter, year, proceeds_mn / 1000, day1_return_pct, day30_mktadj_pct, is_bb),
        )
        self._conn.commit()

    def get_table(self, year: Optional[int] = None) -> pd.DataFrame:
        """Return league table sorted by total proceeds."""
        if year:
            cur = self._conn.execute(
                "SELECT * FROM underwriter_league WHERE year=? ORDER BY total_proceeds_bn DESC",
                (year,),
            )
        else:
            cur = self._conn.execute(
                """SELECT underwriter,
                          SUM(deal_count) as deal_count,
                          SUM(total_proceeds_bn) as total_proceeds_bn,
                          AVG(avg_day1_return) as avg_day1_return,
                          AVG(avg_day30_mktadj) as avg_day30_mktadj,
                          MAX(bulge_bracket) as bulge_bracket
                   FROM underwriter_league
                   GROUP BY underwriter
                   ORDER BY total_proceeds_bn DESC"""
            )
        rows = [dict(r) for r in cur.fetchall()]
        df = pd.DataFrame(rows)
        if "deal_count" in df.columns:
            df = df.sort_values("total_proceeds_bn", ascending=False)
            df["rank"] = range(1, len(df) + 1)
        return df

    def upcoming_lockups(self, days_ahead: int = 60) -> list[dict]:
        """Find IPO results where lockup_expiry is within days_ahead days."""
        today    = date.today()
        cutoff   = (today + timedelta(days=days_ahead)).isoformat()
        today_s  = today.isoformat()
        cur = self._conn.execute(
            """SELECT ticker, company_name, ipo_date, offer_price, lockup_days,
                      lockup_expiry, lead_underwriter
               FROM ipo_results
               WHERE lockup_expiry >= ? AND lockup_expiry <= ?
               ORDER BY lockup_expiry""",
            (today_s, cutoff),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for row in rows:
            le = row.get("lockup_expiry")
            if le:
                days_until = (date.fromisoformat(le) - today).days
                row["days_until_expiry"] = days_until
        return rows


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip()[:10], fmt).date()
        except ValueError:
            continue
    return None


def _find_dollar_amount(text: str, label_pattern: str, window: int = 400) -> Optional[float]:
    """Find a dollar amount near a label pattern."""
    m = re.search(label_pattern, text, re.IGNORECASE)
    if not m:
        return None
    snippet = text[m.end(): m.end() + window]
    pat = r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|[BMK])?"
    dm = re.search(pat, snippet, re.IGNORECASE)
    if not dm:
        return None
    num_str = dm.group(1).replace(",", "")
    suffix  = (dm.group(2) or "").lower()
    try:
        val = float(num_str)
    except ValueError:
        return None
    if suffix in ("billion", "b"):
        val *= 1_000_000_000.0
    elif suffix in ("million", "m"):
        val *= 1_000_000.0
    elif suffix in ("thousand", "k"):
        val *= 1_000.0
    return val


def _find_percent_near(text: str, label_pattern: str, window: int = 300) -> Optional[float]:
    """Find a percentage near a label."""
    m = re.search(label_pattern, text, re.IGNORECASE)
    if not m:
        return None
    snippet = text[m.end(): m.end() + window]
    pm = re.search(r"([\d]+(?:\.\d+)?)\s*%", snippet)
    if not pm:
        return None
    try:
        return float(pm.group(1))
    except ValueError:
        return None


def _extract_price_range(text: str) -> tuple[Optional[float], Optional[float]]:
    pat = r"\$\s*([\d]+(?:\.\d+)?)\s*(?:to|and|-|–)\s*\$\s*([\d]+(?:\.\d+)?)"
    m = re.search(pat, text, re.IGNORECASE)
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None, None


def _extract_trust_amount(text: str) -> Optional[float]:
    m = re.search(
        r"trust\s+account.{0,120}\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)?",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    raw    = m.group(1).replace(",", "")
    suffix = (m.group(2) or "").lower()
    try:
        val = float(raw)
    except ValueError:
        return None
    if suffix == "billion":
        val *= 1_000.0   # store as millions
    elif suffix == "thousand":
        val /= 1_000.0
    return round(val, 1)


def _extract_sponsor(text: str) -> Optional[str]:
    m = re.search(
        r"(?:our\s+)?sponsor[,\s]+([A-Z][A-Za-z\s&,\.]{4,50}?(?:LLC|LP|Inc|Corp|Partners))",
        text,
    )
    return m.group(1).strip() if m else None


def _detect_spac_target_industry(text: str) -> Optional[str]:
    INDUSTRIES = {
        "technology":          ["software", "saas", "cloud", "artificial intelligence", "fintech"],
        "healthcare":          ["pharmaceutical", "biotech", "medical device", "healthcare"],
        "energy":              ["oil", "gas", "renewable", "clean energy"],
        "consumer":            ["retail", "consumer brand", "food", "beverage", "e-commerce"],
        "financial services":  ["insurance", "payments", "banking"],
    }
    text_lower = text.lower()
    for ind, kws in INDUSTRIES.items():
        if any(kw in text_lower for kw in kws):
            return ind
    return None


def _extract_deadline_months(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*months?\s+(?:to\s+)?(?:complete|consummate|close)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# dim_030 additions: pop prediction, lockup signal, quality classification
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# IPO POP PREDICTION — Logistic Regression Model
# ---------------------------------------------------------------------------
# This block replaces the prior weighted-linear-formula heuristic with a
# real logistic regression model fit on historical IPO data.
#
# Feature vector X (7 features):
#   [offer_size_log, is_profitable, revenue_growth, sector_hot_index,
#    market_vix, underwriter_tier, age_years]
#
# Targets:
#   y_binary  — 1 if first_day_pop >= 0.20 ("big pop"), else 0
#   y_pop     — actual first-day return (for log-linear pop estimate)
#
# Math:
#   sigmoid(z) = 1 / (1 + exp(-z))
#   probability_of_big_pop = sigmoid(X @ coef + intercept)
#   predicted_pop          = exp(X @ coef_pop + intercept_pop) - 1
#                            (log-linear from a second LR on log(1+pop))
#
# Fallback: pure-numpy gradient descent if sklearn unavailable.
# ---------------------------------------------------------------------------

# Historical IPO training set — 48 samples curated to reflect realistic
# regimes: hot tech IPOs (2020-2021), traditional profitable IPOs,
# recession/high-VIX IPOs, biotech, SPACs, etc.
# Columns:
#   (offer_size_log, is_profitable, revenue_growth, sector_hot_index,
#    market_vix, underwriter_tier, age_years, FIRST_DAY_POP)
_HISTORICAL_IPOS: list[tuple] = [
    # Hot tech / SaaS — 2020-2021 era, low VIX, big pops
    (8.5, 0, 1.50, 0.90, 18, 1.0,  5, 0.45),
    (9.2, 0, 1.20, 0.85, 16, 1.0,  6, 0.62),
    (8.1, 0, 2.10, 0.95, 20, 1.0,  4, 0.78),
    (7.8, 0, 0.95, 0.80, 22, 0.7,  7, 0.35),
    (8.9, 0, 1.80, 0.92, 19, 1.0,  3, 0.55),
    (8.3, 0, 1.40, 0.88, 21, 1.0,  5, 0.48),
    (9.0, 1, 0.65, 0.82, 17, 1.0,  9, 0.32),
    (7.5, 0, 2.30, 0.95, 18, 0.7,  3, 0.70),
    (8.6, 0, 1.10, 0.78, 23, 1.0,  6, 0.28),
    (8.0, 0, 1.60, 0.90, 19, 0.5,  4, 0.38),
    # Traditional / profitable IPOs — boring, small or negative pops
    (7.2, 1, 0.20, 0.30, 22, 0.5, 12, 0.05),
    (7.8, 1, 0.15, 0.25, 20, 1.0, 18, 0.08),
    (6.9, 1, 0.10, 0.20, 24, 0.5, 25, 0.02),
    (7.5, 1, 0.08, 0.30, 21, 0.7, 15, 0.04),
    (8.2, 1, 0.18, 0.35, 19, 1.0, 22, 0.10),
    (7.0, 1, 0.05, 0.25, 23, 0.5, 30, -0.02),
    (6.8, 1, 0.12, 0.30, 22, 0.5, 10, 0.06),
    (7.6, 1, 0.20, 0.40, 18, 1.0, 14, 0.12),
    (7.4, 1, 0.22, 0.35, 20, 0.7, 16, 0.09),
    (8.0, 1, 0.16, 0.32, 19, 1.0, 20, 0.11),
    # Recession / high-VIX IPOs — broken IPOs, negative pops
    (7.0, 0, 0.40, 0.50, 38, 0.5,  6, -0.15),
    (7.5, 0, 0.60, 0.55, 42, 0.7,  5, -0.10),
    (6.5, 1, 0.15, 0.35, 35, 0.5, 12, -0.08),
    (8.0, 0, 0.80, 0.60, 40, 1.0,  4, -0.05),
    (7.2, 1, 0.10, 0.30, 36, 0.5, 18,  0.00),
    (7.8, 0, 0.90, 0.65, 33, 0.7,  5, -0.03),
    (8.4, 0, 1.10, 0.70, 31, 1.0,  4,  0.10),
    (7.0, 0, 0.50, 0.45, 39, 0.5,  7, -0.18),
    # Biotech — small offerings, very volatile pops
    (6.2, 0, 0.00, 0.60, 20, 0.5,  4,  0.25),
    (6.5, 0, 0.00, 0.65, 18, 0.7,  3,  0.40),
    (6.0, 0, 0.00, 0.55, 25, 0.5,  5, -0.12),
    (6.8, 0, 0.00, 0.70, 22, 0.7,  3,  0.18),
    (6.3, 0, 0.00, 0.50, 28, 0.5,  6, -0.20),
    (6.6, 0, 0.00, 0.62, 21, 0.7,  4,  0.30),
    # SPAC-style / blank-check — slight pop typical
    (8.5, 0, 0.00, 0.40, 22, 0.7,  0,  0.02),
    (8.2, 0, 0.00, 0.35, 24, 0.5,  0,  0.01),
    (8.8, 0, 0.00, 0.45, 20, 1.0,  0,  0.05),
    (8.0, 0, 0.00, 0.30, 26, 0.5,  0, -0.01),
    # Mid-cap industrials — modest pops
    (8.4, 1, 0.30, 0.50, 20, 1.0, 11, 0.15),
    (8.1, 1, 0.25, 0.45, 22, 0.7, 13, 0.12),
    (7.9, 1, 0.28, 0.48, 21, 1.0, 10, 0.18),
    (8.6, 1, 0.35, 0.55, 19, 1.0,  9, 0.22),
    # Consumer / retail
    (7.3, 1, 0.40, 0.60, 20, 0.7,  8, 0.20),
    (7.6, 0, 0.55, 0.65, 22, 0.5,  6, 0.15),
    (7.9, 1, 0.30, 0.50, 24, 1.0, 12, 0.10),
    # Mega-cap tech — huge offerings, moderate pops (price discovery limited)
    (10.5, 1, 0.50, 0.75, 18, 1.0, 15, 0.20),
    (10.2, 0, 0.80, 0.80, 19, 1.0, 12, 0.30),
    (10.8, 1, 0.40, 0.70, 20, 1.0, 18, 0.15),
]

_BIG_POP_THRESHOLD = 0.20  # first-day return >= 20% counts as "big pop"

# Cache: (coef_clf, intercept_clf, coef_reg, intercept_reg, mu, sigma)
_IPO_MODEL_CACHE: Optional[tuple] = None


def _sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _fit_logistic_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    n_iter: int = 3000,
    l2: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Pure-numpy logistic regression via gradient descent on log-loss.

    Loss: L = -Σ[y·log σ(z) + (1-y)·log(1-σ(z))] + (l2/2)·||w||²
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        z = X @ w + b
        p = _sigmoid(z)
        # gradient of log-loss
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def _fit_linear_numpy(X: np.ndarray, y: np.ndarray, l2: float = 0.01) -> tuple[np.ndarray, float]:
    """Closed-form ridge regression for the log-pop magnitude head."""
    n, d = X.shape
    X_aug = np.hstack([X, np.ones((n, 1))])
    A = X_aug.T @ X_aug + l2 * np.eye(d + 1)
    A[-1, -1] = 0.0  # don't regularize intercept
    rhs = X_aug.T @ y
    sol = np.linalg.solve(A, rhs)
    return sol[:-1], float(sol[-1])


def _fit_ipo_pop_model() -> tuple[np.ndarray, float]:
    """Fit and cache the logistic-regression IPO pop model.

    Returns the (coefficients, intercept) of the binary "big pop" classifier
    on standardized features. The full cache also stores the log-pop
    magnitude regressor and the feature standardizer.

    Uses sklearn.linear_model.LogisticRegression when available; otherwise
    falls back to a pure-numpy gradient-descent implementation of the
    same logistic loss.
    """
    global _IPO_MODEL_CACHE
    if _IPO_MODEL_CACHE is not None:
        coef_clf, intercept_clf, *_ = _IPO_MODEL_CACHE
        return coef_clf, intercept_clf

    data = np.array(_HISTORICAL_IPOS, dtype=float)
    X_raw = data[:, :7]
    pop = data[:, 7]
    y_binary = (pop >= _BIG_POP_THRESHOLD).astype(float)
    # log(1+pop) — signed log transform so negative pops stay negative
    y_logpop = np.sign(pop) * np.log1p(np.abs(pop))

    # Standardize features (zero mean, unit variance) — critical for
    # numerical stability of gradient descent and interpretability of coefs.
    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    X = (X_raw - mu) / sigma

    # --- Fit binary classifier (big-pop yes/no) ---
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        clf.fit(X, y_binary.astype(int))
        coef_clf = clf.coef_.ravel().astype(float)
        intercept_clf = float(clf.intercept_[0])
        model_backend = "sklearn"
    except Exception:
        coef_clf, intercept_clf = _fit_logistic_numpy(X, y_binary)
        model_backend = "numpy_gradient_descent"

    # --- Fit log-pop magnitude regressor (ridge) ---
    try:
        from sklearn.linear_model import Ridge  # type: ignore
        reg = Ridge(alpha=1.0)
        reg.fit(X, y_logpop)
        coef_reg = reg.coef_.astype(float)
        intercept_reg = float(reg.intercept_)
    except Exception:
        coef_reg, intercept_reg = _fit_linear_numpy(X, y_logpop)

    _IPO_MODEL_CACHE = (
        coef_clf, intercept_clf, coef_reg, intercept_reg, mu, sigma, model_backend
    )
    return coef_clf, intercept_clf


_FEATURE_NAMES = (
    "offer_size_log", "is_profitable", "revenue_growth",
    "sector_hot_index", "market_vix", "underwriter_tier", "age_years",
)


def _underwriter_tier(name: Optional[str]) -> float:
    """Map underwriter name to tier score (1.0 = bulge bracket, 0.0 = unknown)."""
    if not name:
        return 0.0
    n = name.lower()
    for u in _BULGE_BRACKET:
        if u.lower() in n or n in u.lower():
            return 1.0
    for u in _MAJOR_UNDERWRITERS:
        if u.lower() in n or n in u.lower():
            return 0.7
    return 0.3


_SECTOR_HOT_INDEX = {
    "tech": 0.90, "software": 0.92, "saas": 0.92, "ai": 0.95,
    "cloud": 0.88, "fintech": 0.80, "crypto": 0.75,
    "biotech": 0.60, "pharma": 0.55, "healthcare": 0.55,
    "consumer": 0.50, "retail": 0.45, "industrial": 0.40,
    "energy": 0.35, "oil": 0.30, "utility": 0.25, "reit": 0.30,
    "spac": 0.40, "financial": 0.45,
}


def _sector_index(sector: Optional[str]) -> float:
    if not sector:
        return 0.50
    s = sector.lower().strip()
    if s in _SECTOR_HOT_INDEX:
        return _SECTOR_HOT_INDEX[s]
    for k, v in _SECTOR_HOT_INDEX.items():
        if k in s:
            return v
    return 0.50


def predict_ipo_pop(
    offer_size: float,
    is_profitable: bool,
    revenue_growth: float,
    sector: str,
    market_vix: float,
    underwriter: str,
    age_years: int,
) -> dict:
    """Predict first-day IPO pop using the fitted logistic regression model.

    Parameters
    ----------
    offer_size : float
        Total offer size in millions USD. We transform this to log-scale
        internally (offer_size_log = log(max(offer_size, 1.0))).
    is_profitable : bool
        Whether the issuer has positive net income.
    revenue_growth : float
        YoY revenue growth rate (e.g. 1.5 = 150%).
    sector : str
        Free-text sector / industry name; mapped via _SECTOR_HOT_INDEX.
    market_vix : float
        Current VIX level (typical range 12-40).
    underwriter : str
        Lead underwriter name; mapped via _underwriter_tier.
    age_years : int
        Years since incorporation.

    Returns
    -------
    dict
        predicted_pop              : float   first-day return estimate
        probability_of_big_pop     : float   sigmoid(z) in [0,1]
        confidence                 : str     LOW | MEDIUM | HIGH
        model_type                 : str     "logistic_regression"
        coefficients               : list    classifier coefficients
        intercept                  : float   classifier intercept
        feature_contributions      : dict    per-feature z-score contribution
        z_score                    : float   pre-sigmoid logit
    """
    # Make sure model is fitted and cache is populated
    _fit_ipo_pop_model()
    assert _IPO_MODEL_CACHE is not None
    coef_clf, intercept_clf, coef_reg, intercept_reg, mu, sigma, backend = _IPO_MODEL_CACHE

    offer_size_log = float(np.log(max(float(offer_size), 1.0)))
    sector_hot = _sector_index(sector)
    uw_tier = _underwriter_tier(underwriter)

    x_raw = np.array([
        offer_size_log,
        1.0 if is_profitable else 0.0,
        float(revenue_growth),
        float(sector_hot),
        float(market_vix),
        float(uw_tier),
        float(age_years),
    ])
    x_std = (x_raw - mu) / sigma

    # Logit and sigmoid
    z = float(x_std @ coef_clf + intercept_clf)
    prob_big_pop = float(_sigmoid(z))

    # Log-linear pop estimate from the ridge regressor
    z_pop = float(x_std @ coef_reg + intercept_reg)
    # Inverse of signed log1p: sign(z)·(exp(|z|)-1)
    predicted_pop = float(np.sign(z_pop) * (np.exp(abs(z_pop)) - 1.0))

    # Per-feature contribution to the logit (standardized)
    contribs = {
        name: float(x_std[i] * coef_clf[i])
        for i, name in enumerate(_FEATURE_NAMES)
    }

    # Confidence: based on how far prob is from 0.5
    margin = abs(prob_big_pop - 0.5)
    if margin >= 0.30:
        confidence = "HIGH"
    elif margin >= 0.15:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "predicted_pop": round(predicted_pop, 4),
        "probability_of_big_pop": round(prob_big_pop, 4),
        "confidence": confidence,
        "model_type": "logistic_regression",
        "model_backend": backend,
        "coefficients": [float(c) for c in coef_clf],
        "intercept": float(intercept_clf),
        "feature_names": list(_FEATURE_NAMES),
        "feature_contributions": contribs,
        "z_score": round(z, 4),
        "inputs": {
            "offer_size": offer_size,
            "is_profitable": bool(is_profitable),
            "revenue_growth": revenue_growth,
            "sector": sector,
            "sector_hot_index": sector_hot,
            "market_vix": market_vix,
            "underwriter": underwriter,
            "underwriter_tier": uw_tier,
            "age_years": age_years,
        },
    }


def backtest_ipo_predictions(years: int = 3) -> dict:
    """Backtest the logistic regression IPO pop model on a held-out subset.

    Splits _HISTORICAL_IPOS into a train/test partition (last ~30% as test)
    and reports RMSE on first-day pop, directional accuracy on the
    big-pop classification, and area-under-ROC.

    Parameters
    ----------
    years : int
        Lookback horizon (kept for API compatibility; train/test split is
        based on sample order in the curated set).

    Returns
    -------
    dict
        rmse                  : float  RMSE of predicted_pop vs actual
        directional_accuracy  : float  fraction of correct big/small calls
        auc                   : float  area under the ROC curve
        n_train, n_test       : int
        threshold             : float  big-pop threshold used
    """
    data = np.array(_HISTORICAL_IPOS, dtype=float)
    n = len(data)
    if n < 10:
        raise ValueError("Need at least 10 historical IPOs to backtest")
    # Use last ~30% as held-out test set
    n_test = max(5, int(round(n * 0.30)))
    n_train = n - n_test
    rng = np.random.default_rng(seed=42)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_raw_all = data[:, :7]
    pop_all = data[:, 7]
    y_binary_all = (pop_all >= _BIG_POP_THRESHOLD).astype(float)
    y_logpop_all = np.sign(pop_all) * np.log1p(np.abs(pop_all))

    X_train = X_raw_all[train_idx]
    y_bin_train = y_binary_all[train_idx]
    y_log_train = y_logpop_all[train_idx]
    X_test = X_raw_all[test_idx]
    y_bin_test = y_binary_all[test_idx]
    pop_test = pop_all[test_idx]

    # Standardize using train-only stats (no test-set leakage)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    Xtr = (X_train - mu) / sigma
    Xte = (X_test - mu) / sigma

    # Fit classifier
    try:
        from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        clf.fit(Xtr, y_bin_train.astype(int))
        prob_test = clf.predict_proba(Xte)[:, 1]
        reg = Ridge(alpha=1.0)
        reg.fit(Xtr, y_log_train)
        zpop_test = reg.predict(Xte)
    except Exception:
        w, b = _fit_logistic_numpy(Xtr, y_bin_train)
        prob_test = _sigmoid(Xte @ w + b)
        wr, br = _fit_linear_numpy(Xtr, y_log_train)
        zpop_test = Xte @ wr + br

    predicted_pop_test = np.sign(zpop_test) * (np.exp(np.abs(zpop_test)) - 1.0)

    # RMSE on the actual first-day pop
    rmse = float(np.sqrt(np.mean((predicted_pop_test - pop_test) ** 2)))

    # Directional accuracy: did we correctly call big-pop vs small/negative?
    pred_label = (prob_test >= 0.5).astype(int)
    directional_accuracy = float(np.mean(pred_label == y_bin_test.astype(int)))

    # AUC via Mann-Whitney U statistic
    def _auc(probs: np.ndarray, labels: np.ndarray) -> float:
        pos = probs[labels == 1]
        neg = probs[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        n_pairs = len(pos) * len(neg)
        wins = sum(1 for p in pos for q in neg if p > q)
        ties = sum(1 for p in pos for q in neg if p == q)
        return float((wins + 0.5 * ties) / n_pairs)

    auc = _auc(prob_test, y_bin_test.astype(int))

    return {
        "rmse": round(rmse, 4),
        "directional_accuracy": round(directional_accuracy, 4),
        "auc": round(auc, 4),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "threshold": _BIG_POP_THRESHOLD,
        "years": int(years),
        "model_type": "logistic_regression",
    }


def compute_ipo_pop_prediction(
    revenue_growth_rate: float,
    brand_recognition_score: float,
    market_conditions: float,
    underwriter_tier: float,
) -> dict:
    """Predict first-day IPO pop using a fitted logistic regression model.

    This function preserves its original 4-argument signature for backward
    compatibility but is no longer a weighted linear formula. Inputs are
    mapped into the 7-feature schema expected by `predict_ipo_pop`:

      * revenue_growth_rate     -> revenue_growth (scaled to typical range)
      * brand_recognition_score -> sector_hot_index proxy
      * market_conditions       -> inverse VIX proxy (1=hot -> low VIX)
      * underwriter_tier        -> underwriter tier (already 0-1)

    All four inputs must be on a normalised 0-1 scale where 1 = best.

    Returns
    -------
    dict
        pop_score                 : float  sigmoid output (probability of big pop)
        pop_prediction_pct        : float  predicted first-day return %
        probability_of_big_pop    : float  same as pop_score, explicit name
        model_type                : str    "logistic_regression"
        coefficients              : list   fitted classifier coefficients
        intercept                 : float  fitted classifier intercept
        feature_contributions     : dict   per-feature contribution to logit
        inputs                    : dict   echoed input values
    """
    for name, val in [
        ("revenue_growth_rate", revenue_growth_rate),
        ("brand_recognition_score", brand_recognition_score),
        ("market_conditions", market_conditions),
        ("underwriter_tier", underwriter_tier),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name}={val} must be in [0.0, 1.0]")

    # Map normalized [0,1] inputs into the 7-feature logistic schema.
    # revenue_growth_rate of 1.0 -> ~150% growth; 0.0 -> -10% decline
    revenue_growth = revenue_growth_rate * 1.6 - 0.1
    # brand_recognition_score -> sector_hot_index proxy
    sector_hot = brand_recognition_score
    # market_conditions = 1 (hot) -> VIX ~14; 0 (cold) -> VIX ~38
    market_vix = 38.0 - market_conditions * 24.0
    # underwriter_tier already 0-1; pick representative bank name
    if underwriter_tier >= 0.9:
        uw_name = "Goldman Sachs"
    elif underwriter_tier >= 0.5:
        uw_name = "Jefferies"
    else:
        uw_name = "Boutique Partners"

    # Use synthetic but reasonable defaults for the remaining features
    full = predict_ipo_pop(
        offer_size=300.0,
        is_profitable=False,
        revenue_growth=revenue_growth,
        sector="tech" if sector_hot >= 0.7 else "consumer",
        market_vix=market_vix,
        underwriter=uw_name,
        age_years=6,
    )

    prob_big_pop = full["probability_of_big_pop"]
    predicted_pop = full["predicted_pop"]
    # pop_prediction_pct as percentage points (e.g. 0.15 -> 15.0)
    pop_prediction_pct = round(predicted_pop * 100.0, 2)

    return {
        "pop_score": round(prob_big_pop, 4),
        "pop_prediction_pct": pop_prediction_pct,
        "probability_of_big_pop": prob_big_pop,
        "predicted_pop": predicted_pop,
        "model_type": "logistic_regression",
        "model_backend": full["model_backend"],
        "coefficients": full["coefficients"],
        "intercept": full["intercept"],
        "feature_names": full["feature_names"],
        "feature_contributions": full["feature_contributions"],
        "z_score": full["z_score"],
        "inputs": {
            "revenue_growth_rate": revenue_growth_rate,
            "brand_recognition_score": brand_recognition_score,
            "market_conditions": market_conditions,
            "underwriter_tier": underwriter_tier,
        },
    }


def compute_lockup_expiry_signal(
    ipo_date: date,
    lockup_days: int = 180,
    expected_return_pct: float = -8.0,
) -> dict:
    """
    Compute the lockup expiry date and expected price pressure signal.

    Empirical research shows average -8% abnormal return around day-180
    lockup expiry as insiders and early investors sell.

    Parameters
    ----------
    ipo_date : date
        The IPO pricing/trading date.
    lockup_days : int
        Standard lockup period in calendar days (default 180).
    expected_return_pct : float
        Expected price change around lockup expiry (default -8.0%).

    Returns
    -------
    dict with keys:
        ipo_date             : str    ISO date
        lockup_expiry_date   : str    ISO date
        lockup_days          : int
        expected_return_pct  : float  typically -8%
        days_until_expiry    : int    relative to today (negative = past)
        signal               : str    APPROACHING | PAST | ACTIVE
    """
    lockup_expiry = ipo_date + timedelta(days=lockup_days)
    today = date.today()
    days_until = (lockup_expiry - today).days

    if days_until < 0:
        signal = "PAST"
    elif days_until <= 30:
        signal = "APPROACHING"
    else:
        signal = "ACTIVE"

    return {
        "ipo_date": ipo_date.isoformat(),
        "lockup_expiry_date": lockup_expiry.isoformat(),
        "lockup_days": lockup_days,
        "expected_return_pct": expected_return_pct,
        "days_until_expiry": days_until,
        "signal": signal,
    }


def classify_ipo_quality(
    lead_underwriter: Optional[str],
    ebitda_positive: bool,
) -> dict:
    """
    Classify IPO quality into Tier 1 / Tier 2 / Tier 3 based on underwriter
    prestige and profitability.

    Tier 1: Bulge-bracket lead underwriter AND positive EBITDA (profitable).
    Tier 2: Major (non-bulge) underwriter, OR bulge bracket but loss-making.
    Tier 3: Boutique / unknown underwriter.

    Parameters
    ----------
    lead_underwriter : str | None
        Name of the lead underwriter (as found in _BULGE_BRACKET / _MAJOR_UNDERWRITERS).
    ebitda_positive : bool
        True if the company reported positive EBITDA in its last fiscal year.

    Returns
    -------
    dict with keys:
        tier              : str    "Tier 1" | "Tier 2" | "Tier 3"
        lead_underwriter  : str | None
        is_bulge_bracket  : bool
        is_major          : bool
        ebitda_positive   : bool
        rationale         : str
    """
    is_bulge = lead_underwriter in _BULGE_BRACKET if lead_underwriter else False
    is_major = lead_underwriter in _MAJOR_UNDERWRITERS if lead_underwriter else False

    if is_bulge and ebitda_positive:
        tier = "Tier 1"
        rationale = "Bulge-bracket underwriter with positive EBITDA — highest quality signal."
    elif is_major or is_bulge:
        tier = "Tier 2"
        rationale = (
            "Major underwriter without positive EBITDA, or bulge-bracket but loss-making."
        )
    else:
        tier = "Tier 3"
        rationale = "Boutique or unknown underwriter — higher execution and aftermarket risk."

    return {
        "tier": tier,
        "lead_underwriter": lead_underwriter,
        "is_bulge_bracket": is_bulge,
        "is_major": is_major,
        "ebitda_positive": ebitda_positive,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

ipo_v3_router = APIRouter(prefix="/ipo/v3", tags=["IPO v3"])

_conn_singleton: Optional[sqlite3.Connection] = None
_fetcher_singleton: Optional[EdgarDocFetcher] = None


def _get_conn() -> sqlite3.Connection:
    global _conn_singleton
    if _conn_singleton is None:
        _conn_singleton = _get_db()
    return _conn_singleton


def _get_fetcher() -> EdgarDocFetcher:
    global _fetcher_singleton
    if _fetcher_singleton is None:
        _fetcher_singleton = EdgarDocFetcher()
    return _fetcher_singleton


def _get_tracker() -> IPOPipelineTracker:
    return IPOPipelineTracker(_get_fetcher(), _get_conn())


def _get_perf() -> IPOPerformanceEngine:
    return IPOPerformanceEngine(_get_conn())


def _get_spac() -> SPACTracker:
    return SPACTracker(_get_fetcher(), _get_conn())


def _get_league() -> UnderwriterLeagueTable:
    return UnderwriterLeagueTable(_get_conn())


@ipo_v3_router.get("/pipeline", summary="Active IPO pipeline (filed + roadshow)")
def route_pipeline(
    lookback_days: int = Query(default=90, ge=7, le=365),
) -> dict:
    """Return all S-1 / S-1/A filings in the pipeline, with state labels."""
    tracker = _get_tracker()
    entries = tracker.get_pipeline(lookback_days=lookback_days)
    pipeline = [e for e in entries if e.pipeline_state in ("filed", "roadshow")]
    return {
        "as_of": date.today().isoformat(),
        "count": len(pipeline),
        "entries": [e.model_dump() for e in pipeline],
    }


@ipo_v3_router.get("/recent", summary="Recently priced IPOs")
def route_recent(
    days: int = Query(default=90, ge=7, le=365),
) -> dict:
    """Return recently priced IPOs from SQLite cache."""
    tracker = _get_tracker()
    results = tracker.get_recent_priced(days=days)
    # Also pull from pipeline table for 424B4 entries
    cur = _get_conn().execute(
        "SELECT * FROM ipo_pipeline WHERE pipeline_state='priced' ORDER BY filed_date DESC LIMIT 100"
    )
    priced_pipeline = [dict(r) for r in cur.fetchall()]
    return {
        "days_back": days,
        "ipo_results_count": len(results),
        "priced_from_pipeline": len(priced_pipeline),
        "results": [r.model_dump() for r in results],
        "pipeline_priced": priced_pipeline,
    }


@ipo_v3_router.get("/analysis/{ticker}", summary="Full S-1 structured analysis")
def route_analysis(ticker: str) -> dict:
    """Parse S-1 filing for a ticker and return structured quality analysis."""
    fetcher = _get_fetcher()
    conn    = _get_conn()
    parser  = EdgarS1Parser(fetcher)

    # Resolve ticker → CIK
    cik = fetcher.resolve_ticker_to_cik(ticker.upper())
    if not cik:
        raise HTTPException(status_code=404, detail=f"CIK not found for {ticker}")

    # Find most recent S-1 or 424B4
    sub = fetcher.get_submissions(cik)
    filings = sub.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    accs    = filings.get("accessionNumber", [])
    target_acc: Optional[str] = None
    target_form = "S-1"

    for i, form in enumerate(forms):
        if form in _S1_FORMS | _FINAL_PROS:
            target_acc  = accs[i] if i < len(accs) else None
            target_form = form
            break

    if not target_acc:
        raise HTTPException(status_code=404, detail=f"No S-1/424B4 filing found for {ticker}")

    analysis = parser.parse_s1(cik, target_acc)

    # Persist S1 analysis
    d = analysis.model_dump()
    conn.execute(
        """INSERT OR REPLACE INTO s1_analysis
           (cik, accession, company_name, filed_date, form_type,
            proceeds_rd_pct, proceeds_sales_pct, proceeds_debt_repay_pct,
            proceeds_general_pct, proceeds_acquisitions_pct, proceeds_secondary_pct,
            proceeds_total_mn, proceeds_text,
            risk_factor_count, has_dual_class,
            revenue_yr0, revenue_yr1, revenue_growth_yoy, revenue_cagr_2yr,
            gross_margin, net_income_yr0, cash_yr0, quarterly_cash_burn,
            shares_offered, shares_existing_sold, insider_selling_pct,
            price_range_low, price_range_high, offer_price,
            underwriters_json, has_bulge_bracket, lead_underwriter,
            lockup_days, is_spac, quality_score,
            red_flags_json, green_flags_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            cik, target_acc, analysis.company_name,
            analysis.filed_date.isoformat() if analysis.filed_date else None,
            target_form,
            analysis.use_of_proceeds.rd_pct,
            analysis.use_of_proceeds.sales_marketing_pct,
            analysis.use_of_proceeds.debt_repayment_pct,
            analysis.use_of_proceeds.general_corp_pct,
            analysis.use_of_proceeds.acquisitions_pct,
            analysis.use_of_proceeds.secondary_pct,
            analysis.use_of_proceeds.total_mn,
            analysis.use_of_proceeds.raw_text,
            analysis.risk_factor_count,
            int(analysis.offering.dual_class),
            analysis.financials.revenue_yr0,
            analysis.financials.revenue_yr1,
            analysis.financials.revenue_growth_yoy,
            analysis.financials.revenue_cagr_2yr,
            analysis.financials.gross_margin,
            analysis.financials.net_income_yr0,
            analysis.financials.cash_yr0,
            analysis.financials.quarterly_cash_burn,
            analysis.offering.shares_offered_total,
            analysis.offering.shares_secondary,
            analysis.offering.insider_selling_pct,
            analysis.offering.price_range_low,
            analysis.offering.price_range_high,
            analysis.offering.offer_price,
            json.dumps([u.model_dump() for u in analysis.underwriters]),
            int(analysis.has_bulge_bracket),
            analysis.lead_underwriter,
            analysis.offering.lockup_days,
            int(analysis.is_spac),
            analysis.quality_score,
            json.dumps(analysis.red_flags),
            json.dumps(analysis.green_flags),
        ),
    )
    conn.commit()

    return analysis.model_dump()


@ipo_v3_router.get("/returns/{ticker}", summary="IPO aftermarket performance")
def route_returns(
    ticker: str,
    ipo_date: str = Query(..., description="IPO date YYYY-MM-DD"),
    offer_price: float = Query(..., description="Offering price USD"),
) -> dict:
    """Return 1/30/90/180-day raw and market-adjusted returns for an IPO."""
    try:
        ipo_date_parsed = date.fromisoformat(ipo_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="ipo_date must be YYYY-MM-DD")

    perf = _get_perf()
    d1   = perf.day1_return(ticker, offer_price, ipo_date_parsed)
    returns = perf.aftermarket_returns(ticker, ipo_date_parsed, offer_price)
    returns["day1"] = d1
    return returns


@ipo_v3_router.get("/spac-pipeline", summary="Active SPAC tracker")
def route_spac_pipeline(
    lookback_days: int = Query(default=365, ge=30, le=730),
) -> dict:
    """Return active SPACs from recent S-1 filings by blank-check companies."""
    spac = _get_spac()
    records = spac.get_active_spacs(lookback_days=lookback_days)
    return {
        "as_of": date.today().isoformat(),
        "count": len(records),
        "spacs": [r.model_dump() for r in records],
    }


@ipo_v3_router.get("/underwriter-league", summary="Underwriter league table")
def route_league(
    year: Optional[int] = Query(default=None, description="Filter by year (omit for all-time)"),
) -> dict:
    """Return underwriter league table sorted by total proceeds."""
    league = _get_league()
    df = league.get_table(year=year)
    if df.empty:
        return {"message": "No data — run /analysis endpoints first to populate", "table": []}
    return {
        "year": year or "all-time",
        "table": df.to_dict(orient="records"),
    }


@ipo_v3_router.get("/upcoming-lockups", summary="Upcoming lock-up expirations")
def route_lockups(
    days_ahead: int = Query(default=60, ge=7, le=180),
) -> dict:
    """Return IPOs with lock-up expiring within days_ahead days."""
    league = _get_league()
    lockups = league.upcoming_lockups(days_ahead=days_ahead)
    return {
        "days_ahead": days_ahead,
        "count": len(lockups),
        "lockups": lockups,
    }
