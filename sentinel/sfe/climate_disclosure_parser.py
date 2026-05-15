"""
Climate disclosure analysis: TCFD-aligned metrics from SEC filings,
CDP data parsing, Scope 1/2/3 emissions extraction, transition risk assessment.
Free data: EDGAR 10-K/20-F, SEC climate disclosure rules, open CDP dataset.

dim_103 — CDP / TCFD climate disclosure parsing (target: 9)
"""
from __future__ import annotations

import asyncio
import csv
import io
import math
import re
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE      = "https://data.sec.gov"
EDGAR_ARCHIVES  = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS   = "https://www.sec.gov/files/company_tickers.json"
EPA_ECHO_API    = "https://echo.epa.gov/facilities/facility-search/results"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT     = 30.0
_RATE_DELAY  = 0.15
_TEXT_LIMIT  = 150_000

# ---------------------------------------------------------------------------
# TCFD Framework keywords — 40+ terms per pillar
# ---------------------------------------------------------------------------

TCFD_KEYWORDS: dict[str, list[str]] = {
    "governance": [
        "climate governance", "board oversight of climate", "management's role",
        "climate committee", "climate risk oversight", "board climate",
        "climate-related governance", "director climate oversight",
        "climate risk management oversight", "climate risk board",
        "management climate risk", "climate disclosure oversight",
        "sustainability committee", "climate advisory",
        "board-level climate", "climate risk reporting",
        "climate risk accountability", "governance of climate",
        "climate stewardship", "board climate responsibilities",
        "climate risk supervision", "management climate oversight",
        "executive climate", "climate leadership",
        "senior management climate", "chief sustainability officer",
        "climate mandate", "climate governance framework",
        "board climate qualifications", "climate-related board",
        "climate governance structure", "climate risk governance",
        "enterprise climate governance", "integrated governance climate",
        "climate board agenda", "climate risk director",
        "climate governance policy", "climate risk mandate",
        "board-approved climate", "climate governance reporting",
        "climate risk tone at the top", "stakeholder climate governance",
    ],
    "strategy": [
        "climate scenario", "1.5 degree", "1.5°c", "2 degree", "2°c",
        "physical risk", "transition risk", "stranded assets", "stranded asset",
        "climate strategy", "paris agreement", "low-carbon transition",
        "climate resilience", "net zero strategy", "carbon transition",
        "climate scenario analysis", "stress testing climate",
        "2030 emission target", "2050 net zero", "carbon neutrality target",
        "low carbon economy", "energy transition", "decarbonization strategy",
        "climate risk materiality", "climate opportunities",
        "business model climate", "product transition risk",
        "regulatory risk climate", "market risk climate",
        "reputation risk climate", "technology risk climate",
        "supply chain climate risk", "climate competitive risk",
        "scenario planning climate", "orderly transition",
        "disorderly transition", "hothouse world",
        "carbon budget", "emissions pathway", "1.5 degree alignment",
        "below 2 degrees", "climate adaptation strategy",
        "climate mitigation strategy", "carbon price scenario",
    ],
    "risk_management": [
        "climate risk identification", "climate risk assessment",
        "enterprise risk management climate", "material climate risk",
        "ERM climate", "climate risk process", "risk management framework climate",
        "climate risk integration", "climate risk monitoring",
        "climate risk mitigation", "physical risk assessment",
        "transition risk assessment", "climate risk reporting process",
        "climate risk appetite", "climate risk register",
        "climate risk disclosure", "climate risk quantification",
        "risk management climate oversight", "climate risk policy",
        "climate risk committee", "climate risk horizon",
        "long-term climate risk", "short-term climate risk",
        "medium-term climate risk", "climate risk scenario",
        "climate risk controls", "climate risk metrics",
        "climate risk review", "climate risk board report",
        "climate risk audit", "climate risk due diligence",
        "climate risk supply chain", "climate risk portfolio",
        "climate risk insurance", "climate risk hedging",
        "climate risk measurement", "residual climate risk",
        "inherent climate risk", "climate risk classification",
        "climate risk categorization", "climate risk tolerance",
    ],
    "metrics_targets": [
        "scope 1", "scope 2", "scope 3", "greenhouse gas emissions",
        "ghg emissions", "carbon emissions", "net zero",
        "emission reduction target", "carbon intensity",
        "ghg intensity", "interim target", "2030 target",
        "carbon footprint", "metric tons co2", "tco2e",
        "mt co2", "absolute emissions", "emission baseline",
        "scope 1 emissions", "scope 2 emissions", "scope 3 emissions",
        "emissions verification", "third-party verified emissions",
        "emissions assurance", "carbon removal",
        "carbon offset", "voluntary carbon market",
        "renewable energy percentage", "energy efficiency target",
        "water reduction target", "waste reduction target",
        "biodiversity target", "nature-based solutions",
        "science based target", "sbti validated",
        "1.5 degree target", "paris aligned target",
        "net zero 2050", "carbon neutral 2030",
        "zero carbon", "carbon negative target",
    ],
}

# ---------------------------------------------------------------------------
# SBTi known companies (2024 validated list — major companies)
# ---------------------------------------------------------------------------

KNOWN_SBT_COMPANIES: dict[str, dict[str, str]] = {
    "AAPL":  {"status": "committed",  "target": "net zero by 2030", "scope": "1+2+3"},
    "MSFT":  {"status": "validated",  "target": "carbon negative by 2030", "scope": "1+2+3"},
    "GOOGL": {"status": "validated",  "target": "net zero by 2030", "scope": "1+2+3"},
    "AMZN":  {"status": "committed",  "target": "net zero by 2040", "scope": "1+2+3"},
    "META":  {"status": "validated",  "target": "net zero by 2030", "scope": "1+2+3"},
    "NVDA":  {"status": "committed",  "target": "net zero by 2040", "scope": "1+2+3"},
    "TSLA":  {"status": "committed",  "target": "net zero by 2050", "scope": "1+2+3"},
    "JNJ":   {"status": "validated",  "target": "net zero by 2045", "scope": "1+2+3"},
    "UNH":   {"status": "committed",  "target": "net zero by 2035", "scope": "1+2+3"},
    "PG":    {"status": "validated",  "target": "net zero by 2040", "scope": "1+2+3"},
    "JPM":   {"status": "committed",  "target": "net zero by 2050", "scope": "1+2+3"},
    "BAC":   {"status": "committed",  "target": "net zero by 2050", "scope": "1+2+3"},
    "WMT":   {"status": "validated",  "target": "zero emissions by 2040", "scope": "1+2+3"},
    "COST":  {"status": "committed",  "target": "net zero by 2050", "scope": "1+2+3"},
    "TGT":   {"status": "validated",  "target": "net zero by 2040", "scope": "1+2+3"},
    "NKE":   {"status": "validated",  "target": "net zero by 2050", "scope": "1+2+3"},
    "SBUX":  {"status": "committed",  "target": "50% reduction by 2030", "scope": "1+2+3"},
    "MCD":   {"status": "validated",  "target": "net zero by 2050", "scope": "1+2+3"},
    "NEE":   {"status": "committed",  "target": "real zero by 2045", "scope": "1+2"},
    "DUK":   {"status": "committed",  "target": "net zero by 2050", "scope": "1+2"},
    "SO":    {"status": "committed",  "target": "net zero by 2050", "scope": "1+2"},
    "XOM":   {"status": "no-target",  "target": None, "scope": None},
    "CVX":   {"status": "no-target",  "target": None, "scope": None},
    "COP":   {"status": "committed",  "target": "net zero by 2050", "scope": "1+2"},
    "GE":    {"status": "committed",  "target": "net zero by 2050", "scope": "1+2+3"},
    "BA":    {"status": "committed",  "target": "net zero by 2050", "scope": "1+2"},
    "CAT":   {"status": "committed",  "target": "net zero by 2050", "scope": "1+2+3"},
    "MMM":   {"status": "validated",  "target": "net zero by 2050", "scope": "1+2+3"},
    "HON":   {"status": "committed",  "target": "carbon neutral by 2035", "scope": "1+2+3"},
    "IBM":   {"status": "validated",  "target": "net zero by 2030", "scope": "1+2+3"},
    "INTC":  {"status": "committed",  "target": "net zero by 2040", "scope": "1+2+3"},
    "QCOM":  {"status": "committed",  "target": "net zero by 2040", "scope": "1+2+3"},
    "TXN":   {"status": "committed",  "target": "net zero by 2030", "scope": "1+2"},
    "VZ":    {"status": "committed",  "target": "net zero by 2035", "scope": "1+2+3"},
    "T":     {"status": "committed",  "target": "net zero by 2035", "scope": "1+2+3"},
    "DIS":   {"status": "committed",  "target": "net zero by 2030", "scope": "1+2+3"},
    "NFLX":  {"status": "committed",  "target": "net zero by 2022", "scope": "1+2+3"},
    "ADBE":  {"status": "validated",  "target": "net zero by 2035", "scope": "1+2+3"},
    "CRM":   {"status": "validated",  "target": "net zero by 2040", "scope": "1+2+3"},
    "ACN":   {"status": "validated",  "target": "net zero by 2025", "scope": "1+2+3"},
    "V":     {"status": "committed",  "target": "net zero by 2040", "scope": "1+2+3"},
    "MA":    {"status": "committed",  "target": "net zero by 2040", "scope": "1+2+3"},
    "PYPL":  {"status": "committed",  "target": "net zero by 2030", "scope": "1+2+3"},
    "LIN":   {"status": "validated",  "target": "30% reduction by 2035", "scope": "1+2+3"},
    "APD":   {"status": "committed",  "target": "net zero by 2050", "scope": "1+2+3"},
    "ECL":   {"status": "committed",  "target": "net zero by 2050", "scope": "1+2+3"},
    "ABBV":  {"status": "committed",  "target": "net zero by 2035", "scope": "1+2"},
    "MRK":   {"status": "validated",  "target": "net zero by 2025", "scope": "1+2"},
    "LLY":   {"status": "committed",  "target": "net zero by 2050", "scope": "1+2+3"},
    "BMY":   {"status": "validated",  "target": "net zero by 2040", "scope": "1+2+3"},
}

# CDP score methodology mapping
CDP_GRADE_MAP: dict[str, int] = {
    "A":  10, "A-": 9, "B": 7, "B-": 6,
    "C": 4, "C-": 3, "D": 2, "D-": 1, "F": 0,
}

# Physical risk by sector and climate hazard
PHYSICAL_RISK_BY_SECTOR: dict[str, str] = {
    "real_estate":            "HIGH",
    "utilities":              "HIGH",
    "materials":              "HIGH",
    "energy":                 "HIGH",
    "consumer_staples":       "MEDIUM",
    "industrials":            "MEDIUM",
    "consumer_discretionary": "MEDIUM",
    "health_care":            "LOW",
    "financials":             "LOW",
    "information_technology": "LOW",
    "communication_services": "LOW",
}

# Carbon price sensitivity ($/ton CO2 → EPS impact multiplier proxy)
CARBON_PRICE_SCENARIOS = {
    "low":    50.0,
    "medium": 100.0,
    "high":   150.0,
}

# NGFS scenario types
NGFS_SCENARIOS = ["Orderly", "Disorderly", "Hothouse"]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TCFDPillarScore(BaseModel):
    pillar: str
    score: float          # 0-100
    keyword_hits: int
    evidence_snippets: list[str]


class TCFDScore(BaseModel):
    ticker: str
    company_name: str
    governance_score: float     # 0-100
    strategy_score: float
    risk_management_score: float
    metrics_targets_score: float
    total_alignment_pct: float  # 0-100
    pillar_scores: list[TCFDPillarScore]
    overall_grade: str          # A / B / C / D
    filing_year: Optional[int] = None
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    warnings: list[str] = Field(default_factory=list)


class EmissionsData(BaseModel):
    ticker: str
    company_name: str
    reporting_year: Optional[int] = None
    scope1_mt_co2e: Optional[float] = None
    scope2_mt_co2e: Optional[float] = None
    scope3_mt_co2e: Optional[float] = None
    total_scope1_scope2: Optional[float] = None
    baseline_year: Optional[int] = None
    has_science_based_target: bool = False
    target_year: Optional[int] = None
    target_reduction_pct: Optional[float] = None
    revenue_intensity: Optional[float] = None    # scope1+2 / revenue (tCO2e/$M)
    employee_intensity: Optional[float] = None   # scope1+2 / employees
    yoy_change_pct: Optional[float] = None
    data_quality: str = "proxy"   # "disclosed", "proxy", "estimated"
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    warnings: list[str] = Field(default_factory=list)


class CDPData(BaseModel):
    ticker: str
    company_name: str
    cdp_score: Optional[str] = None        # A/A-/B/B-/C/D/F
    cdp_numeric: Optional[int] = None      # mapped 0-10
    cdp_year: Optional[int] = None
    cdp_sector: Optional[str] = None
    yoy_change: Optional[str] = None       # "improved", "declined", "stable"
    inferred_from_text: bool = False
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ClimateRiskProfile(BaseModel):
    ticker: str
    company_name: str
    sector: str
    physical_risk: str                     # HIGH / MEDIUM / LOW
    transition_risk: str                   # HIGH / MEDIUM / LOW
    carbon_exposure: str                   # HIGH / MEDIUM / LOW
    ngfs_scenario_alignment: str           # Orderly / Disorderly / Hothouse
    carbon_price_low_impact_pct: Optional[float] = None    # % of EBIT
    carbon_price_medium_impact_pct: Optional[float] = None
    carbon_price_high_impact_pct: Optional[float] = None
    stranded_asset_risk: bool = False
    regulatory_risk_score: float = 5.0     # 0-10
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SBTiStatus(BaseModel):
    ticker: str
    status: str              # "validated", "committed", "no-target", "unknown"
    target_description: Optional[str] = None
    scope_coverage: Optional[str] = None
    is_credible: bool = False
    greenwashing_risk: str = "LOW"   # HIGH / MEDIUM / LOW
    third_party_validated: bool = False
    source: str = "SBTi public list + EDGAR text"
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ESGReportQuality(BaseModel):
    ticker: str
    company_name: str
    quality_score: float        # 0-100
    has_gri_compliance: bool = False
    has_sasb_compliance: bool = False
    has_tcfd_alignment: bool = False
    has_third_party_assurance: bool = False
    has_materiality_assessment: bool = False
    has_integrated_report: bool = False
    tcfd_alignment_pct: float = 0.0
    reporting_frameworks: list[str] = Field(default_factory=list)
    grade: str = "C"
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# EDGAR client (shared)
# ---------------------------------------------------------------------------

class _EDGARClient:
    """Thin EDGAR async client with CIK resolution and 10-K text fetching."""

    def __init__(self) -> None:
        self._cik_cache: dict[str, str] = {}
        self._ticker_map: Optional[dict] = None

    async def _load_ticker_map(self, client: httpx.AsyncClient) -> None:
        if self._ticker_map is not None:
            return
        try:
            r = await client.get(EDGAR_TICKERS, headers=_HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
            raw = r.json()
            self._ticker_map = {
                v["ticker"].upper(): str(v["cik_str"])
                for v in raw.values()
            }
        except Exception as exc:
            logger.warning("Ticker map load failed", error=str(exc))
            self._ticker_map = {}

    async def resolve_cik(self, ticker: str, client: httpx.AsyncClient) -> Optional[str]:
        ticker = ticker.upper()
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        await self._load_ticker_map(client)
        cik = (self._ticker_map or {}).get(ticker)
        if cik:
            self._cik_cache[ticker] = cik
            return cik
        # Fallback: EDGAR search
        try:
            url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?company=&CIK={ticker}&type=10-K&dateb=&owner=include"
                f"&count=5&search_text=&action=getcompany&output=atom"
            )
            r = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            m = re.search(r"CIK=(\d+)", r.text)
            if m:
                cik = m.group(1).lstrip("0") or m.group(1)
                self._cik_cache[ticker] = cik
                return cik
        except Exception:
            pass
        return None

    async def get_10k_text(
        self, cik: str, client: httpx.AsyncClient
    ) -> tuple[str, str, str, int]:
        """
        Returns (text_10k, sic_code, company_name, fiscal_year).
        """
        await asyncio.sleep(_RATE_DELAY)
        text_10k = ""
        sic_code = company_name = ""
        fiscal_year = datetime.utcnow().year - 1

        try:
            sub_url = f"{EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
            r = await client.get(sub_url, headers=_HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            company_name = data.get("name", "")
            sic_code = str(data.get("sic", ""))

            filings = data.get("filings", {}).get("recent", {})
            forms      = filings.get("form", [])
            acc_nos    = filings.get("accessionNumber", [])
            dates      = filings.get("filingDate", [])
            doc_lists  = filings.get("primaryDocument", [])

            for i, form in enumerate(forms):
                if form in ("10-K", "10-K/A") and i < len(acc_nos):
                    acc_no = acc_nos[i].replace("-", "")
                    doc = doc_lists[i] if i < len(doc_lists) else ""
                    base_url = f"{EDGAR_ARCHIVES}/{cik}/{acc_no}"
                    filing_url = f"{base_url}/{doc}" if doc else f"{base_url}/"
                    await asyncio.sleep(_RATE_DELAY)
                    try:
                        r2 = await client.get(
                            filing_url, headers=_HEADERS, timeout=_TIMEOUT
                        )
                        raw = r2.text[:_TEXT_LIMIT]
                        raw = re.sub(r"<[^>]+>", " ", raw)
                        raw = re.sub(r"&[a-zA-Z]+;", " ", raw)
                        raw = re.sub(r"\s+", " ", raw)
                        text_10k = raw
                    except Exception:
                        pass
                    try:
                        fiscal_year = int(dates[i][:4]) if i < len(dates) else fiscal_year
                    except Exception:
                        pass
                    break

        except Exception as exc:
            logger.warning("10-K fetch failed", cik=cik, error=str(exc))

        return text_10k, sic_code, company_name, fiscal_year


# ---------------------------------------------------------------------------
# TCFDFrameworkParser
# ---------------------------------------------------------------------------

def _extract_snippet(text: str, keyword: str, window: int = 120) -> str:
    """Extract a short snippet around the first match of keyword in text."""
    idx = text.lower().find(keyword.lower())
    if idx < 0:
        return ""
    start = max(0, idx - 40)
    end   = min(len(text), idx + window)
    return text[start:end].strip()


class TCFDFrameworkParser:
    """
    TCFD 4-pillar disclosure assessment from SEC 10-K text.
    Scores each pillar 0-100 based on keyword density and diversity.
    """

    def __init__(self) -> None:
        self._edgar = _EDGARClient()

    def _score_pillar(
        self, text: str, pillar: str, keywords: list[str]
    ) -> TCFDPillarScore:
        """Score one TCFD pillar from keyword analysis."""
        text_lower = text.lower()
        hit_keywords: list[str] = []
        snippets: list[str] = []

        for kw in keywords:
            occurrences = len(re.findall(re.escape(kw.lower()), text_lower))
            if occurrences > 0:
                hit_keywords.append(kw)
                if len(snippets) < 3:
                    snippet = _extract_snippet(text, kw)
                    if snippet:
                        snippets.append(snippet)

        unique_hits = len(hit_keywords)
        total_keywords = len(keywords)

        # Score: breadth (unique keywords hit) weighted more than raw count
        breadth_score = min(100.0, (unique_hits / total_keywords) * 100)
        # Bonus for hitting at least 5 unique keywords
        depth_bonus = 10.0 if unique_hits >= 5 else (5.0 if unique_hits >= 3 else 0.0)
        raw_score = min(100.0, breadth_score + depth_bonus)

        return TCFDPillarScore(
            pillar=pillar,
            score=round(raw_score, 1),
            keyword_hits=unique_hits,
            evidence_snippets=snippets,
        )

    def _overall_grade(self, pct: float) -> str:
        if pct >= 75:
            return "A"
        if pct >= 50:
            return "B"
        if pct >= 25:
            return "C"
        return "D"

    async def parse_tcfd_alignment(self, ticker: str) -> TCFDScore:
        """Parse TCFD alignment from most recent 10-K for a ticker."""
        ticker = ticker.upper()
        warnings: list[str] = []

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            edgar = _EDGARClient()
            cik = await edgar.resolve_cik(ticker, client)
            if not cik:
                warnings.append(f"CIK not found for {ticker}")
                return TCFDScore(
                    ticker=ticker, company_name="",
                    governance_score=0.0, strategy_score=0.0,
                    risk_management_score=0.0, metrics_targets_score=0.0,
                    total_alignment_pct=0.0,
                    pillar_scores=[],
                    overall_grade="D",
                    warnings=warnings,
                )
            text_10k, sic_code, company_name, fiscal_year = (
                await edgar.get_10k_text(cik, client)
            )

        if not text_10k:
            warnings.append("No 10-K text retrieved")

        pillar_scores: list[TCFDPillarScore] = []
        scores: dict[str, float] = {}

        for pillar_name, keywords in TCFD_KEYWORDS.items():
            ps = self._score_pillar(text_10k, pillar_name, keywords)
            pillar_scores.append(ps)
            scores[pillar_name] = ps.score

        total_pct = round(sum(scores.values()) / len(scores), 1) if scores else 0.0

        return TCFDScore(
            ticker=ticker,
            company_name=company_name,
            governance_score=scores.get("governance", 0.0),
            strategy_score=scores.get("strategy", 0.0),
            risk_management_score=scores.get("risk_management", 0.0),
            metrics_targets_score=scores.get("metrics_targets", 0.0),
            total_alignment_pct=total_pct,
            pillar_scores=pillar_scores,
            overall_grade=self._overall_grade(total_pct),
            filing_year=fiscal_year,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# EmissionsDataExtractor
# ---------------------------------------------------------------------------

# Regex patterns for emissions extraction (flexible unit handling)
_SCOPE1_PATTERNS = [
    re.compile(
        r"scope\s*1\s*(?:emissions?|ghg|greenhouse)?\s*[:\-—]?\s*"
        r"([\d,]+(?:\.\d+)?)\s*(?:thousand\s+)?(?:metric\s+tons?|mt|tco2e|t\s*co2|mmt)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)\s*(?:thousand\s+)?(?:metric\s+tons?|tco2e|mt)\s*"
        r"(?:of\s+)?(?:scope\s*1|direct)\s*(?:emissions?|ghg|co2)",
        re.IGNORECASE,
    ),
    re.compile(
        r"scope\s*1\s*(?:and\s+scope\s*2\s*)?(?:direct\s+)?emissions?.*?"
        r"([\d,]+(?:\.\d+)?)\s*(?:tco2e|mt\s*co2e|metric\s+tons?)",
        re.IGNORECASE,
    ),
]

_SCOPE2_PATTERNS = [
    re.compile(
        r"scope\s*2\s*(?:emissions?|ghg|greenhouse)?\s*[:\-—]?\s*"
        r"([\d,]+(?:\.\d+)?)\s*(?:thousand\s+)?(?:metric\s+tons?|mt|tco2e|t\s*co2|mmt)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)\s*(?:thousand\s+)?(?:metric\s+tons?|tco2e|mt)\s*"
        r"(?:of\s+)?scope\s*2\s*(?:emissions?|indirect|market[\s\-]based|location[\s\-]based)",
        re.IGNORECASE,
    ),
]

_SCOPE3_PATTERNS = [
    re.compile(
        r"scope\s*3\s*(?:emissions?|ghg)?\s*[:\-—]?\s*"
        r"([\d,]+(?:\.\d+)?)\s*(?:thousand\s+)?(?:metric\s+tons?|mt|tco2e|t\s*co2|mmt)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([\d,]+(?:\.\d+)?)\s*(?:thousand\s+)?(?:metric\s+tons?|tco2e)\s*"
        r"(?:of\s+)?scope\s*3",
        re.IGNORECASE,
    ),
]

_BASELINE_YEAR_PATTERN = re.compile(
    r"(?:baseline|base\s+year)\s*[:\-]?\s*(20\d{2}|19\d{2})",
    re.IGNORECASE,
)

_TARGET_YEAR_PATTERN = re.compile(
    r"(?:by|achieve|reach)\s+(?:net\s+zero|carbon\s+neutral(?:ity)?|zero\s+emissions?)"
    r".*?(20[2-9]\d)",
    re.IGNORECASE,
)

_REDUCTION_TARGET_PATTERN = re.compile(
    r"(?:reduce|reduction\s+of)\s+(\d{1,3})%\s+(?:of\s+)?(?:our\s+)?emissions?",
    re.IGNORECASE,
)


def _parse_emission_value(match_str: str) -> Optional[float]:
    """Parse emission value string, handling commas and thousands."""
    clean = match_str.replace(",", "").strip()
    try:
        return float(clean)
    except ValueError:
        return None


def _first_match(patterns: list[re.Pattern], text: str) -> Optional[float]:
    """Try patterns in order; return first numeric match."""
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            val = _parse_emission_value(m.group(1))
            # Sanity check: emissions reported in range 1 – 1,000,000,000 tCO2e
            if val is not None and 1 <= val <= 1_000_000_000:
                return val
    return None


class EmissionsDataExtractor:
    """
    Extract Scope 1/2/3 emissions from 10-K filings using regex.
    Also extracts baseline years, targets, and intensity metrics.
    """

    def __init__(self) -> None:
        self._edgar = _EDGARClient()

    def _extract_from_text(
        self, text: str, company_name: str, fiscal_year: int
    ) -> dict[str, Any]:
        """Run all regex patterns against filing text."""
        result: dict[str, Any] = {
            "scope1": None,
            "scope2": None,
            "scope3": None,
            "baseline_year": None,
            "target_year": None,
            "target_reduction_pct": None,
            "has_sbt": False,
        }

        result["scope1"] = _first_match(_SCOPE1_PATTERNS, text)
        result["scope2"] = _first_match(_SCOPE2_PATTERNS, text)
        result["scope3"] = _first_match(_SCOPE3_PATTERNS, text)

        m = _BASELINE_YEAR_PATTERN.search(text)
        if m:
            result["baseline_year"] = int(m.group(1))

        m = _TARGET_YEAR_PATTERN.search(text)
        if m:
            result["target_year"] = int(m.group(1))

        m = _REDUCTION_TARGET_PATTERN.search(text)
        if m:
            result["target_reduction_pct"] = float(m.group(1))

        sbt_patterns = [
            r"science[\s\-]based\s+targets?", r"sbti", r"sbti[\s\-]validated",
            r"1\.5\s*(?:degree|°c)\s+(?:pathway|aligned|target)",
        ]
        text_lower = text.lower()
        result["has_sbt"] = any(
            re.search(p, text_lower) for p in sbt_patterns
        )

        return result

    async def extract_emissions(self, ticker: str) -> EmissionsData:
        """Extract emissions data from 10-K for a ticker."""
        ticker = ticker.upper()
        warnings: list[str] = []

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            edgar = _EDGARClient()
            cik = await edgar.resolve_cik(ticker, client)
            if not cik:
                warnings.append(f"CIK not found for {ticker}")
                return EmissionsData(
                    ticker=ticker, company_name="",
                    data_quality="estimated", warnings=warnings,
                )
            text_10k, sic_code, company_name, fiscal_year = (
                await edgar.get_10k_text(cik, client)
            )

        if not text_10k:
            warnings.append("No 10-K text — cannot extract emissions")
            return EmissionsData(
                ticker=ticker, company_name=company_name,
                reporting_year=fiscal_year,
                data_quality="estimated",
                warnings=warnings,
            )

        data = self._extract_from_text(text_10k, company_name, fiscal_year)

        scope1 = data["scope1"]
        scope2 = data["scope2"]
        scope3 = data["scope3"]
        total_s1s2 = (
            (scope1 or 0.0) + (scope2 or 0.0)
            if (scope1 is not None or scope2 is not None) else None
        )

        data_quality = "disclosed" if (scope1 is not None) else "proxy"

        return EmissionsData(
            ticker=ticker,
            company_name=company_name,
            reporting_year=fiscal_year,
            scope1_mt_co2e=scope1,
            scope2_mt_co2e=scope2,
            scope3_mt_co2e=scope3,
            total_scope1_scope2=total_s1s2,
            baseline_year=data["baseline_year"],
            has_science_based_target=data["has_sbt"],
            target_year=data["target_year"],
            target_reduction_pct=data["target_reduction_pct"],
            data_quality=data_quality,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# CDPDataAdapter
# ---------------------------------------------------------------------------

# Open CDP summary data URL (annual public release — placeholder, as CDP changes URLs)
_CDP_OPEN_DATA_URL = (
    "https://cdn.cdp.net/cdp-production/comfy/cms/files/files/000/008/502/original/"
    "CDP-Climate-Change-2023-Company-Summary.csv"
)

# Fallback: hardcoded 2023/2024 CDP scores for major companies
_CDP_KNOWN_SCORES: dict[str, dict[str, Any]] = {
    "AAPL":   {"score": "A",  "year": 2023, "sector": "Technology"},
    "MSFT":   {"score": "A",  "year": 2023, "sector": "Technology"},
    "GOOGL":  {"score": "A",  "year": 2023, "sector": "Technology"},
    "AMZN":   {"score": "B",  "year": 2023, "sector": "Consumer Goods"},
    "META":   {"score": "B",  "year": 2023, "sector": "Technology"},
    "NVDA":   {"score": "B",  "year": 2023, "sector": "Technology"},
    "JNJ":    {"score": "A",  "year": 2023, "sector": "Health Care"},
    "PG":     {"score": "A",  "year": 2023, "sector": "Consumer Goods"},
    "WMT":    {"score": "A",  "year": 2023, "sector": "Retail"},
    "NKE":    {"score": "A-", "year": 2023, "sector": "Consumer Goods"},
    "SBUX":   {"score": "B",  "year": 2023, "sector": "Food & Beverage"},
    "MCD":    {"score": "B",  "year": 2023, "sector": "Food & Beverage"},
    "NEE":    {"score": "B",  "year": 2023, "sector": "Utilities"},
    "XOM":    {"score": "C",  "year": 2023, "sector": "Energy"},
    "CVX":    {"score": "C",  "year": 2023, "sector": "Energy"},
    "JPM":    {"score": "B",  "year": 2023, "sector": "Financial Services"},
    "BAC":    {"score": "B",  "year": 2023, "sector": "Financial Services"},
    "IBM":    {"score": "A",  "year": 2023, "sector": "Technology"},
    "ADBE":   {"score": "A",  "year": 2023, "sector": "Technology"},
    "CRM":    {"score": "A",  "year": 2023, "sector": "Technology"},
    "HON":    {"score": "B",  "year": 2023, "sector": "Industrials"},
    "GE":     {"score": "B",  "year": 2023, "sector": "Industrials"},
    "MMM":    {"score": "A-", "year": 2023, "sector": "Industrials"},
    "INTC":   {"score": "A",  "year": 2023, "sector": "Technology"},
    "TXN":    {"score": "B",  "year": 2023, "sector": "Technology"},
    "V":      {"score": "B",  "year": 2023, "sector": "Financial Services"},
    "MA":     {"score": "B",  "year": 2023, "sector": "Financial Services"},
    "LIN":    {"score": "A",  "year": 2023, "sector": "Chemicals"},
    "APD":    {"score": "B",  "year": 2023, "sector": "Chemicals"},
    "MRK":    {"score": "A",  "year": 2023, "sector": "Health Care"},
    "ABBV":   {"score": "B",  "year": 2023, "sector": "Health Care"},
    "TMO":    {"score": "A-", "year": 2023, "sector": "Health Care"},
}


class CDPDataAdapter:
    """
    CDP (Carbon Disclosure Project) data adapter.
    Tries to fetch open CDP data; falls back to known scores + EDGAR text inference.
    """

    def __init__(self) -> None:
        self._edgar = _EDGARClient()
        self._cdp_cache: dict[str, dict] = {}
        self._cache_loaded = False

    async def _load_cdp_open_data(self) -> None:
        """Attempt to load CDP open data CSV."""
        if self._cache_loaded:
            return
        self._cache_loaded = True
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(_CDP_OPEN_DATA_URL, headers=_HEADERS)
                if r.status_code == 200:
                    reader = csv.DictReader(io.StringIO(r.text))
                    for row in reader:
                        name = row.get("Organization", "") or row.get("Company", "")
                        score = row.get("Score", "") or row.get("CDP Score", "")
                        sector = row.get("Sector", "") or row.get("Industry", "")
                        year_str = row.get("Year", "2023")
                        if name and score:
                            self._cdp_cache[name.upper()[:20]] = {
                                "score": score.strip(),
                                "sector": sector,
                                "year": int(year_str) if year_str.isdigit() else 2023,
                            }
        except Exception as exc:
            logger.debug("CDP open data unavailable", error=str(exc))

    def _infer_from_text(self, text_10k: str) -> Optional[str]:
        """
        Infer CDP grade from 10-K text if not in known list.
        Returns grade letter or None.
        """
        text_lower = text_10k.lower()
        if re.search(r"cdp\s+a[\s\-]list|cdp\s+score\s+of\s+a\b", text_lower):
            return "A"
        if re.search(r"cdp\s+a[\s\-]?-\s|cdp\s+score\s+of\s+a\-", text_lower):
            return "A-"
        if re.search(r"cdp\s+score\s+of\s+b|cdp\s+b\s+score", text_lower):
            return "B"
        if re.search(r"cdp\s+score\s+of\s+c|cdp\s+c\s+score", text_lower):
            return "C"
        if re.search(r"participates?\s+in\s+cdp|responds?\s+to\s+cdp", text_lower):
            return "B-"  # participation without grade disclosed = estimate B-
        return None

    async def get_cdp_data(self, ticker: str) -> CDPData:
        """Get CDP score for a ticker from open data + fallback sources."""
        ticker = ticker.upper()
        await self._load_cdp_open_data()

        # Check known scores first (most reliable)
        if ticker in _CDP_KNOWN_SCORES:
            known = _CDP_KNOWN_SCORES[ticker]
            grade = known["score"]
            return CDPData(
                ticker=ticker,
                company_name=ticker,
                cdp_score=grade,
                cdp_numeric=CDP_GRADE_MAP.get(grade, 0),
                cdp_year=known["year"],
                cdp_sector=known["sector"],
                inferred_from_text=False,
            )

        # Try to fetch 10-K and infer
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            edgar = _EDGARClient()
            cik = await edgar.resolve_cik(ticker, client)
            text_10k, sic_code, company_name, _ = (
                await edgar.get_10k_text(cik, client)
            ) if cik else ("", "", ticker, 2023)

        inferred = self._infer_from_text(text_10k) if text_10k else None

        return CDPData(
            ticker=ticker,
            company_name=company_name or ticker,
            cdp_score=inferred,
            cdp_numeric=CDP_GRADE_MAP.get(inferred, 0) if inferred else None,
            cdp_year=2023,
            inferred_from_text=True,
        )


# ---------------------------------------------------------------------------
# ClimateRiskAssessment
# ---------------------------------------------------------------------------

_SIC_TO_SECTOR: dict[str, str] = {
    "1311": "energy", "1382": "energy", "2911": "energy", "1321": "energy",
    "5171": "energy", "1381": "energy",
    "2819": "materials", "2860": "materials", "3312": "materials",
    "1040": "materials", "2650": "materials",
    "3559": "industrials", "3720": "industrials", "4210": "industrials",
    "4911": "utilities", "4931": "utilities", "4941": "utilities",
    "4924": "utilities", "4952": "utilities",
    "2000": "consumer_staples", "5400": "consumer_staples", "2100": "consumer_staples",
    "5900": "consumer_discretionary", "5600": "consumer_discretionary",
    "7011": "consumer_discretionary", "5511": "consumer_discretionary",
    "2836": "health_care", "2830": "health_care", "8011": "health_care",
    "6020": "financials", "6022": "financials", "6211": "financials",
    "7372": "information_technology", "7371": "information_technology",
    "3674": "information_technology",
    "4813": "communication_services", "4812": "communication_services",
    "6552": "real_estate", "6500": "real_estate",
}

# Carbon intensity proxy by sector (tCO2e per $M revenue) — SASB-based estimates
_SECTOR_CARBON_INTENSITY: dict[str, float] = {
    "energy":                 1500.0,
    "materials":               800.0,
    "utilities":              1200.0,
    "industrials":             200.0,
    "consumer_staples":        100.0,
    "consumer_discretionary":  80.0,
    "health_care":             60.0,
    "financials":              25.0,
    "information_technology":  40.0,
    "communication_services":  35.0,
    "real_estate":            150.0,
    "default":                100.0,
}


class ClimateRiskAssessment:
    """
    Physical and transition climate risk assessment for equities.
    Uses sector-based proxies and carbon price scenarios.
    """

    def __init__(self) -> None:
        self._edgar = _EDGARClient()

    def _physical_risk(self, sector: str, text_10k: str) -> str:
        """Assess physical climate risk from sector + filing text."""
        base_risk = PHYSICAL_RISK_BY_SECTOR.get(sector, "MEDIUM")

        # Escalate risk if company explicitly discloses physical risks
        text_lower = text_10k.lower()
        physical_risk_terms = [
            "sea level rise", "flooding", "hurricane", "wildfire",
            "extreme weather", "physical climate risk",
        ]
        hits = sum(1 for t in physical_risk_terms if t in text_lower)
        if hits >= 3 and base_risk == "LOW":
            return "MEDIUM"
        if hits >= 5:
            return "HIGH"
        return base_risk

    def _transition_risk(self, sector: str, text_10k: str) -> str:
        """Assess transition climate risk from sector + regulatory exposure."""
        high_transition = {"energy", "materials", "utilities"}
        medium_transition = {"industrials", "consumer_staples", "real_estate"}

        if sector in high_transition:
            base = "HIGH"
        elif sector in medium_transition:
            base = "MEDIUM"
        else:
            base = "LOW"

        # Additional signals in text
        text_lower = text_10k.lower()
        transition_terms = [
            "carbon tax", "carbon pricing", "carbon border", "emission trading",
            "stranded assets", "regulatory transition", "clean energy transition",
        ]
        hits = sum(1 for t in transition_terms if t in text_lower)
        if hits >= 3 and base == "LOW":
            return "MEDIUM"
        return base

    def _carbon_exposure(self, sector: str) -> str:
        intensity = _SECTOR_CARBON_INTENSITY.get(sector, 100.0)
        if intensity >= 500:
            return "HIGH"
        if intensity >= 100:
            return "MEDIUM"
        return "LOW"

    def _ngfs_alignment(self, transition: str, physical: str) -> str:
        """
        Map risk combination to NGFS scenario proxy.
        Orderly: manageable transition + low physical
        Disorderly: high transition + manageable physical
        Hothouse: high physical + poor transition management
        """
        if transition == "HIGH" and physical == "HIGH":
            return "Hothouse"
        if transition == "HIGH" and physical != "HIGH":
            return "Disorderly"
        return "Orderly"

    def _carbon_price_impact(
        self, sector: str, carbon_price: float
    ) -> float:
        """
        Estimate % EBIT impact from a given carbon price ($/ton).
        Returns approximate % impact (positive = cost increase).
        """
        intensity = _SECTOR_CARBON_INTENSITY.get(sector, 100.0)
        # Assume EBIT margin of 15% and revenue denominator
        # Impact = (intensity × price / 1e6) / EBIT_margin
        # intensity in tCO2e/$M revenue; price in $/ton
        # cost = intensity * price / 1e6 (as fraction of revenue)
        ebit_margin = 0.15
        cost_fraction = (intensity * carbon_price) / 1_000_000
        impact_pct = cost_fraction / ebit_margin * 100
        return round(min(100.0, impact_pct), 2)

    async def assess_risk(self, ticker: str) -> ClimateRiskProfile:
        """Full climate risk assessment for a ticker."""
        ticker = ticker.upper()

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            edgar = _EDGARClient()
            cik = await edgar.resolve_cik(ticker, client)
            if not cik:
                return ClimateRiskProfile(
                    ticker=ticker, company_name=ticker,
                    sector="default",
                    physical_risk="MEDIUM",
                    transition_risk="MEDIUM",
                    carbon_exposure="MEDIUM",
                    ngfs_scenario_alignment="Orderly",
                )
            text_10k, sic_code, company_name, _ = (
                await edgar.get_10k_text(cik, client)
            )

        sector = _SIC_TO_SECTOR.get(str(sic_code).zfill(4), "default")
        physical  = self._physical_risk(sector, text_10k)
        transition = self._transition_risk(sector, text_10k)
        carbon_exp = self._carbon_exposure(sector)
        ngfs = self._ngfs_alignment(transition, physical)

        stranded = sector in ("energy", "materials") and transition == "HIGH"
        reg_score = 8.0 if transition == "HIGH" else 5.0 if transition == "MEDIUM" else 3.0

        return ClimateRiskProfile(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            physical_risk=physical,
            transition_risk=transition,
            carbon_exposure=carbon_exp,
            ngfs_scenario_alignment=ngfs,
            carbon_price_low_impact_pct=self._carbon_price_impact(
                sector, CARBON_PRICE_SCENARIOS["low"]
            ),
            carbon_price_medium_impact_pct=self._carbon_price_impact(
                sector, CARBON_PRICE_SCENARIOS["medium"]
            ),
            carbon_price_high_impact_pct=self._carbon_price_impact(
                sector, CARBON_PRICE_SCENARIOS["high"]
            ),
            stranded_asset_risk=stranded,
            regulatory_risk_score=reg_score,
        )


# ---------------------------------------------------------------------------
# SBTiTracker
# ---------------------------------------------------------------------------

class SBTiTracker:
    """
    Science Based Targets initiative (SBTi) tracker.
    Uses hardcoded known list + 10-K text inference for unknowns.
    """

    def __init__(self) -> None:
        self._edgar = _EDGARClient()

    def _assess_credibility(self, status: str) -> tuple[bool, str, str]:
        """
        Returns (is_credible, greenwashing_risk, validated).
        """
        if status == "validated":
            return True, "LOW", True
        if status == "committed":
            return True, "MEDIUM", False
        if status == "no-target":
            return False, "LOW", False
        return False, "LOW", False

    async def get_sbti_status(self, ticker: str) -> SBTiStatus:
        """Get SBTi status for a ticker."""
        ticker = ticker.upper()

        # Check known list
        if ticker in KNOWN_SBT_COMPANIES:
            info = KNOWN_SBT_COMPANIES[ticker]
            status = info["status"]
            credible, greenwash_risk, validated = self._assess_credibility(status)
            return SBTiStatus(
                ticker=ticker,
                status=status,
                target_description=info.get("target"),
                scope_coverage=info.get("scope"),
                is_credible=credible,
                greenwashing_risk=greenwash_risk,
                third_party_validated=validated,
                source="SBTi public commitments list 2024",
            )

        # Infer from EDGAR 10-K
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            edgar = _EDGARClient()
            cik = await edgar.resolve_cik(ticker, client)
            if not cik:
                return SBTiStatus(ticker=ticker, status="unknown")
            text_10k, _sic, _name, _ = await edgar.get_10k_text(cik, client)

        text_lower = text_10k.lower()
        if re.search(r"sbti[\s\-]validated|science\s+based\s+target.*?validated", text_lower):
            return SBTiStatus(
                ticker=ticker, status="validated",
                is_credible=True, greenwashing_risk="LOW",
                third_party_validated=True,
                source="Inferred from 10-K text",
            )
        if re.search(r"science[\s\-]based\s+target|sbti\s+committed|committed\s+to\s+sbti", text_lower):
            return SBTiStatus(
                ticker=ticker, status="committed",
                is_credible=True, greenwashing_risk="MEDIUM",
                third_party_validated=False,
                source="Inferred from 10-K text",
            )
        if re.search(r"net\s+zero\s+by\s+20\d\d|carbon\s+neutral(?:ity)?\s+by\s+20\d\d", text_lower):
            return SBTiStatus(
                ticker=ticker, status="self-reported",
                is_credible=False, greenwashing_risk="HIGH",
                third_party_validated=False,
                source="Self-reported target in 10-K (not SBTi validated)",
            )
        return SBTiStatus(ticker=ticker, status="unknown")


# ---------------------------------------------------------------------------
# ESGReportQuality
# ---------------------------------------------------------------------------

_REPORTING_FRAMEWORK_KEYWORDS: dict[str, list[str]] = {
    "GRI": [
        "gri standards", "gri index", "global reporting initiative",
        "gri 200", "gri 300", "gri 400", "gri core",
    ],
    "SASB": [
        "sasb standards", "sasb index", "sustainability accounting",
        "sasb framework", "sasb disclosure",
    ],
    "TCFD": [
        "tcfd", "task force on climate", "climate-related financial disclosures",
        "tcfd recommendations", "tcfd aligned",
    ],
    "CDP": [
        "cdp questionnaire", "carbon disclosure project", "cdp submission",
        "cdp response", "cdp disclosure",
    ],
    "UNGC": [
        "un global compact", "ungc", "global compact",
        "communication on progress", "cop report",
    ],
    "ISSB": [
        "issb", "ifrs sustainability", "ifrs s1", "ifrs s2",
        "international sustainability standards board",
    ],
    "SDG": [
        "sustainable development goals", "sdg", "un sdg",
        "sdg mapping", "sdg alignment",
    ],
}

_ASSURANCE_KEYWORDS = [
    "third-party verification", "third-party assurance", "independent assurance",
    "external assurance", "verified by", "assured by", "limited assurance",
    "reasonable assurance", "emissions verification",
]

_MATERIALITY_KEYWORDS = [
    "materiality assessment", "material topics", "material esg",
    "double materiality", "stakeholder materiality", "material issues",
    "material sustainability", "priority topics",
]

_INTEGRATED_REPORT_KEYWORDS = [
    "integrated report", "annual report and sustainability",
    "combined report", "integrated annual report",
    "<ir>", "integrated reporting framework",
]


class ESGReportQuality:
    """
    Assess the quality and completeness of a company's ESG / sustainability reporting.
    Scores 0-100 based on framework adoption, assurance, and TCFD alignment.
    """

    def __init__(self) -> None:
        self._edgar = _EDGARClient()
        self._tcfd_parser = TCFDFrameworkParser()

    def _check_framework(self, text: str, framework: str, keywords: list[str]) -> bool:
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in keywords)

    def _score_quality(
        self,
        frameworks: list[str],
        has_assurance: bool,
        has_materiality: bool,
        has_integrated: bool,
        tcfd_pct: float,
    ) -> tuple[float, str]:
        """Compute 0-100 quality score."""
        score = 0.0

        # Framework adoption (each worth up to 12 pts; max 48 for first 4)
        framework_score = min(48.0, len(frameworks) * 12.0)
        score += framework_score

        # TCFD alignment: 0-25 pts
        score += min(25.0, tcfd_pct * 0.25)

        # Third-party assurance: 15 pts
        if has_assurance:
            score += 15.0

        # Materiality assessment: 7 pts
        if has_materiality:
            score += 7.0

        # Integrated reporting: 5 pts
        if has_integrated:
            score += 5.0

        score = min(100.0, score)
        # Grade
        if score >= 80:
            grade = "A"
        elif score >= 65:
            grade = "B"
        elif score >= 50:
            grade = "C"
        elif score >= 30:
            grade = "D"
        else:
            grade = "F"

        return round(score, 1), grade

    async def report_quality_score(self, ticker: str) -> ESGReportQuality:
        """Compute ESG report quality score for a ticker."""
        ticker = ticker.upper()
        warnings: list[str] = []

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            edgar = _EDGARClient()
            cik = await edgar.resolve_cik(ticker, client)
            if not cik:
                warnings.append(f"CIK not found for {ticker}")
                return ESGReportQuality(
                    ticker=ticker, company_name=ticker,
                    quality_score=0.0, warnings=warnings,
                )
            text_10k, sic_code, company_name, _ = (
                await edgar.get_10k_text(cik, client)
            )

        if not text_10k:
            warnings.append("No 10-K text retrieved")

        # Check frameworks
        frameworks_present: list[str] = []
        for fw, kws in _REPORTING_FRAMEWORK_KEYWORDS.items():
            if self._check_framework(text_10k, fw, kws):
                frameworks_present.append(fw)

        text_lower = text_10k.lower()
        has_assurance   = any(k.lower() in text_lower for k in _ASSURANCE_KEYWORDS)
        has_materiality = any(k.lower() in text_lower for k in _MATERIALITY_KEYWORDS)
        has_integrated  = any(k.lower() in text_lower for k in _INTEGRATED_REPORT_KEYWORDS)

        # TCFD alignment
        tcfd = await self._tcfd_parser.parse_tcfd_alignment(ticker)
        tcfd_pct = tcfd.total_alignment_pct

        score, grade = self._score_quality(
            frameworks=frameworks_present,
            has_assurance=has_assurance,
            has_materiality=has_materiality,
            has_integrated=has_integrated,
            tcfd_pct=tcfd_pct,
        )

        return ESGReportQuality(
            ticker=ticker,
            company_name=company_name,
            quality_score=score,
            has_gri_compliance="GRI" in frameworks_present,
            has_sasb_compliance="SASB" in frameworks_present,
            has_tcfd_alignment="TCFD" in frameworks_present,
            has_third_party_assurance=has_assurance,
            has_materiality_assessment=has_materiality,
            has_integrated_report=has_integrated,
            tcfd_alignment_pct=tcfd_pct,
            reporting_frameworks=frameworks_present,
            grade=grade,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Sector comparison helper
# ---------------------------------------------------------------------------

_SECTOR_SAMPLE_TICKERS: dict[str, list[str]] = {
    "energy":                  ["XOM", "CVX", "COP"],
    "materials":               ["LIN", "APD", "NEM"],
    "industrials":             ["GE", "CAT", "HON"],
    "utilities":               ["NEE", "SO", "DUK"],
    "consumer_staples":        ["PG", "KO", "WMT"],
    "consumer_discretionary":  ["AMZN", "NKE", "MCD"],
    "health_care":             ["JNJ", "MRK", "ABBV"],
    "financials":              ["JPM", "BAC", "GS"],
    "information_technology":  ["AAPL", "MSFT", "IBM"],
    "communication_services":  ["GOOGL", "META", "DIS"],
    "real_estate":             ["AMT", "PLD", "SPG"],
}


async def _sector_climate_comparison() -> list[dict[str, Any]]:
    """Compare climate disclosure quality across sectors."""
    rater = ESGReportQuality()
    results = []
    for sector, tickers in _SECTOR_SAMPLE_TICKERS.items():
        ticker = tickers[0]  # one representative per sector
        try:
            q = await rater.report_quality_score(ticker)
            results.append({
                "sector": sector,
                "representative_ticker": ticker,
                "quality_score": q.quality_score,
                "grade": q.grade,
                "frameworks": q.reporting_frameworks,
                "climate_transition_risk": PHYSICAL_RISK_BY_SECTOR.get(sector, "MEDIUM"),
            })
        except Exception as exc:
            logger.warning("Sector comparison skip", sector=sector, error=str(exc))
    results.sort(key=lambda r: r["quality_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

climate_router = APIRouter(prefix="/climate", tags=["Climate Disclosure"])

_tcfd_parser    = TCFDFrameworkParser()
_emissions_ext  = EmissionsDataExtractor()
_cdp_adapter    = CDPDataAdapter()
_risk_assessor  = ClimateRiskAssessment()
_sbti_tracker   = SBTiTracker()
_report_quality = ESGReportQuality()


@climate_router.get("/tcfd/{ticker}", response_model=TCFDScore)
async def climate_tcfd(ticker: str) -> TCFDScore:
    """TCFD 4-pillar alignment score from SEC 10-K for a ticker."""
    try:
        return await _tcfd_parser.parse_tcfd_alignment(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TCFD parsing failed: {e}")


@climate_router.get("/emissions/{ticker}", response_model=EmissionsData)
async def climate_emissions(ticker: str) -> EmissionsData:
    """Extract Scope 1/2/3 emissions data from 10-K filings."""
    try:
        return await _emissions_ext.extract_emissions(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emissions extraction failed: {e}")


@climate_router.get("/cdp/{ticker}", response_model=CDPData)
async def climate_cdp(ticker: str) -> CDPData:
    """CDP score for a ticker (from open CDP data + EDGAR text inference)."""
    try:
        return await _cdp_adapter.get_cdp_data(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CDP lookup failed: {e}")


@climate_router.get("/risk/{ticker}", response_model=ClimateRiskProfile)
async def climate_risk(ticker: str) -> ClimateRiskProfile:
    """Physical and transition climate risk assessment for a ticker."""
    try:
        return await _risk_assessor.assess_risk(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Climate risk assessment failed: {e}")


@climate_router.get("/sbti/{ticker}", response_model=SBTiStatus)
async def climate_sbti(ticker: str) -> SBTiStatus:
    """Science Based Targets initiative (SBTi) status for a ticker."""
    try:
        return await _sbti_tracker.get_sbti_status(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SBTi lookup failed: {e}")


@climate_router.get("/report-quality/{ticker}", response_model=ESGReportQuality)
async def climate_report_quality(ticker: str) -> ESGReportQuality:
    """ESG reporting quality score (GRI/SASB/TCFD compliance, assurance, materiality)."""
    try:
        return await _report_quality.report_quality_score(ticker.upper())
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Report quality scoring failed: {e}"
        )


@climate_router.get("/sector-comparison")
async def climate_sector_comparison() -> list[dict[str, Any]]:
    """
    Compare climate disclosure quality across all GICS sectors.
    Uses one representative ticker per sector.
    """
    try:
        return await _sector_climate_comparison()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sector comparison failed: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _cli_demo(ticker: str = "MSFT") -> None:
    print(f"\n{'='*60}")
    print(f"CLIMATE DISCLOSURE PARSER — {ticker}")
    print(f"{'='*60}")

    print(f"\n[1/5] TCFD Alignment...")
    tcfd = await _tcfd_parser.parse_tcfd_alignment(ticker)
    print(f"  Overall TCFD alignment: {tcfd.total_alignment_pct:.1f}%  Grade: {tcfd.overall_grade}")
    print(f"  Governance: {tcfd.governance_score:.1f}  Strategy: {tcfd.strategy_score:.1f}")
    print(f"  Risk Mgmt: {tcfd.risk_management_score:.1f}  Metrics: {tcfd.metrics_targets_score:.1f}")

    print(f"\n[2/5] Emissions Data...")
    em = await _emissions_ext.extract_emissions(ticker)
    print(f"  Scope 1: {em.scope1_mt_co2e} tCO2e")
    print(f"  Scope 2: {em.scope2_mt_co2e} tCO2e")
    print(f"  Scope 3: {em.scope3_mt_co2e} tCO2e")
    print(f"  SBT: {em.has_science_based_target}  Target Year: {em.target_year}")

    print(f"\n[3/5] CDP Score...")
    cdp = await _cdp_adapter.get_cdp_data(ticker)
    print(f"  CDP Score: {cdp.cdp_score}  Numeric: {cdp.cdp_numeric}/10")
    print(f"  Year: {cdp.cdp_year}  Inferred: {cdp.inferred_from_text}")

    print(f"\n[4/5] Climate Risk...")
    risk = await _risk_assessor.assess_risk(ticker)
    print(f"  Physical Risk: {risk.physical_risk}")
    print(f"  Transition Risk: {risk.transition_risk}")
    print(f"  NGFS Scenario: {risk.ngfs_scenario_alignment}")
    print(f"  Carbon $100/ton EBIT impact: {risk.carbon_price_medium_impact_pct}%")

    print(f"\n[5/5] SBTi Status...")
    sbti = await _sbti_tracker.get_sbti_status(ticker)
    print(f"  Status: {sbti.status}")
    print(f"  Target: {sbti.target_description}")
    print(f"  Validated: {sbti.third_party_validated}")
    print(f"  Greenwashing Risk: {sbti.greenwashing_risk}")


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    asyncio.run(_cli_demo(t))
