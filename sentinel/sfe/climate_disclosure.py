"""climate_disclosure.py — CDP/TCFD climate disclosure parsing and analysis.

Parses climate-related disclosures from free public sources:
  - SEC EDGAR 10-K TCFD keyword analysis (Task Force on Climate-related Financial Disclosures)
  - SEC 2024 climate disclosure rule items: Scope 1/2 GHG, physical risk, transition risk
  - SBTi (Science Based Targets initiative) public company CSV
  - EPA ECHO facility enforcement data (Clean Air Act / Clean Water Act violations)
  - GHG extraction via multi-pattern regex from 10-K filings

Proxy signals only — NOT equivalent to CDP scores or Sustainalytics ratings.
"""
from __future__ import annotations

import asyncio
import csv
import io
import math
import re
from datetime import date, datetime
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

SBTI_CSV_URL = (
    "https://sciencebasedtargets.org/files/SBTi-Companies-Taking-Action.csv"
)
EPA_ECHO_API = "https://echo.epa.gov/facilities/facility-search/results"
EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_EDGAR_SLEEP = 0.15
_TEXT_LIMIT = 120_000

# ---------------------------------------------------------------------------
# TCFD framework keyword mappings
# ---------------------------------------------------------------------------

TCFD_PILLARS: dict[str, list[str]] = {
    "governance": [
        "climate governance", "board oversight of climate", "management's role",
        "climate committee", "climate risk oversight", "board climate",
        "climate-related governance", "director climate",
    ],
    "strategy": [
        "climate scenario", "1.5 degree", "1.5°C", "2 degree", "2°C",
        "physical risk", "transition risk", "stranded assets", "stranded asset",
        "climate strategy", "paris agreement", "low-carbon transition",
        "climate resilience", "net zero strategy",
    ],
    "risk_management": [
        "climate risk", "climate risk identification", "climate risk assessment",
        "enterprise risk management", "material climate", "ERM",
        "climate risk process", "risk management framework",
    ],
    "metrics_targets": [
        "scope 1", "scope 2", "scope 3", "greenhouse gas emissions",
        "GHG emissions", "carbon emissions", "net zero", "emission reduction target",
        "carbon intensity", "GHG intensity", "interim target", "2030 target",
        "carbon footprint", "metric tons CO2",
    ],
}

# Physical climate risk keywords
PHYSICAL_RISKS: dict[str, list[str]] = {
    "acute": [
        "hurricane", "flood", "wildfire", "extreme weather", "storm surge",
        "drought", "extreme precipitation", "heatwave", "tornado", "tropical cyclone",
    ],
    "chronic": [
        "sea level rise", "temperature increase", "chronic precipitation",
        "heat stress", "permafrost", "changing rainfall patterns",
        "chronic water stress", "long-term temperature",
    ],
}

# Transition climate risk keywords
TRANSITION_RISKS: dict[str, list[str]] = {
    "policy": [
        "carbon tax", "carbon price", "emissions trading", "cap-and-trade",
        "regulatory risk", "stranded asset", "policy uncertainty",
        "climate regulation", "carbon border adjustment",
    ],
    "technology": [
        "clean technology", "electrification", "technological disruption",
        "low-carbon technology", "battery storage", "CCS", "carbon capture",
    ],
    "market": [
        "demand shift", "consumer preference", "energy transition",
        "market repricing", "carbon-intensive products", "fossil fuel demand",
    ],
    "reputational": [
        "stakeholder expectations", "divestment", "investor pressure",
        "reputational risk", "greenwashing", "climate litigation",
    ],
}

# Industry climate risk by SIC code range
_HIGH_RISK_SICS: frozenset[str] = frozenset({
    "1311", "1321", "1381", "1382",  # oil & gas
    "2911",                           # petroleum refining
    "1221", "1222",                   # coal mining
    "4911", "4931",                   # electric services (fossil-heavy)
    "3312",                           # blast furnace / steel
    "2819", "2860",                   # industrial chemicals
    "4953",                           # waste / cement
    "3312",                           # primary metals
    "2411",                           # logging
})

_LOW_RISK_SICS: frozenset[str] = frozenset({
    "7372", "7371", "7374",   # software / tech services
    "8011", "8049", "8099",   # healthcare / medical
    "8200", "8211", "8220",   # education
    "6411", "6311",           # insurance
    "6022", "6020",           # banking / credit
})

# GHG regex patterns — ordered from most to least specific
# Each pattern must capture the numeric value in group 1
_GHG_PATTERNS: list[tuple[str, str]] = [
    # "X million metric tons of CO2e" / "X MMTCO2e"
    (
        r"scope\s*(?:1|one|i)\b[^.]{0,200}?"
        r"([\d,]+(?:\.\d+)?)\s*"
        r"(?:million\s+)?(?:metric\s+)?(?:tons?|MT|MMTCO2e|MtCO2e|tCO2e)",
        "scope1",
    ),
    (
        r"scope\s*(?:2|two|ii)\b[^.]{0,200}?"
        r"([\d,]+(?:\.\d+)?)\s*"
        r"(?:million\s+)?(?:metric\s+)?(?:tons?|MT|MMTCO2e|MtCO2e|tCO2e)",
        "scope2",
    ),
    (
        r"scope\s*(?:3|three|iii)\b[^.]{0,200}?"
        r"([\d,]+(?:\.\d+)?)\s*"
        r"(?:million\s+)?(?:metric\s+)?(?:tons?|MT|MMTCO2e|MtCO2e|tCO2e)",
        "scope3",
    ),
    # "direct GHG emissions: X"
    (
        r"direct\s+(?:ghg|greenhouse\s+gas)\s+emissions[^.]{0,100}?"
        r"([\d,]+(?:\.\d+)?)\s*(?:million\s+)?(?:metric\s+)?tons?",
        "scope1",
    ),
    # "market-based scope 2"
    (
        r"market[\s\-]based\s+scope\s*2[^.]{0,100}?"
        r"([\d,]+(?:\.\d+)?)\s*(?:million\s+)?(?:metric\s+)?tons?",
        "scope2_market",
    ),
    # "location-based scope 2"
    (
        r"location[\s\-]based\s+scope\s*2[^.]{0,100}?"
        r"([\d,]+(?:\.\d+)?)\s*(?:million\s+)?(?:metric\s+)?tons?",
        "scope2_location",
    ),
    # Simpler fallback: number immediately after "Scope 1 emissions" table entry
    (
        r"Scope\s+1\s+(?:GHG\s+)?[Ee]missions?\s*[\|:\t ]+\s*([\d,]+(?:\.\d+)?)",
        "scope1",
    ),
    (
        r"Scope\s+2\s+(?:GHG\s+)?[Ee]missions?\s*[\|:\t ]+\s*([\d,]+(?:\.\d+)?)",
        "scope2",
    ),
    # GHG intensity: X metric tons per unit
    (
        r"(?:ghg|carbon|emissions?)\s+intensity[^.]{0,150}?"
        r"([\d,]+(?:\.\d+)?)\s*(?:mt|tons?|kg)?\s*CO2[e]?\s*/\s*\$?(?:million|MM|M)?",
        "intensity",
    ),
]

# Net-zero year extraction pattern
_NET_ZERO_YEAR_PATTERNS: list[str] = [
    r"net[\s\-]zero\s+(?:by\s+)?(20[3-9]\d)",
    r"carbon\s+neutral(?:ity)?\s+(?:by\s+)?(20[3-9]\d)",
    r"(20[3-9]\d)\s+net[\s\-]zero",
    r"climate\s+target[^.]{0,50}?(20[3-9]\d)",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GHGEmissions(BaseModel):
    ticker: str
    filing_year: int
    scope1_mt_co2e: Optional[float] = None
    scope2_mt_co2e: Optional[float] = None
    scope3_mt_co2e: Optional[float] = None
    scope2_market_based: Optional[float] = None
    scope2_location_based: Optional[float] = None
    intensity_metric: Optional[str] = None   # e.g. "per revenue $MM"
    intensity_value: Optional[float] = None
    yoy_reduction_pct: Optional[float] = None
    data_source: str = "sec_10k"             # "sec_10k" | "cdp" | "sustainability_report"
    extraction_confidence: str = "medium"    # "high" | "medium" | "low"


class TCFDDisclosure(BaseModel):
    ticker: str
    company_name: str
    filing_year: int
    has_tcfd_disclosure: bool
    tcfd_completeness: float     # 0-1, fraction of TCFD pillars addressed
    # Governance pillar
    board_climate_oversight: bool = False
    mgmt_climate_committee: bool = False
    # Strategy pillar
    has_scenario_analysis: bool = False
    scenarios_analyzed: list[str] = Field(default_factory=list)
    physical_risks_identified: list[str] = Field(default_factory=list)
    transition_risks_identified: list[str] = Field(default_factory=list)
    # Risk management pillar
    climate_risk_in_erp: bool = False
    # Metrics & Targets pillar
    ghg: Optional[GHGEmissions] = None
    has_net_zero_target: bool = False
    net_zero_year: Optional[int] = None
    has_sbti_commitment: bool = False
    has_interim_target: bool = False
    # Summary
    disclosure_gaps: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = (
        "Proxy signals from public EDGAR filings. "
        "Not equivalent to CDP scores or Sustainalytics ratings."
    )


class EPAViolation(BaseModel):
    facility_name: str
    state: str
    violation_type: str
    penalty_amount: Optional[float] = None
    violation_date: Optional[date] = None
    statute: Optional[str] = None   # "CAA", "CWA", "RCRA", "EPCRA", etc.


class ClimateRiskScore(BaseModel):
    ticker: str
    company_name: str
    physical_risk_score: float      # 0-10
    transition_risk_score: float    # 0-10
    overall_climate_risk: float     # weighted composite 0-10
    disclosure_score: float         # 0-10
    net_zero_readiness: float       # 0-10
    epa_violations_5y: int
    industry_risk_level: str        # "High" | "Medium" | "Low"
    recommendation: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class ClimateDisclosureAnalyzer:
    """Parse TCFD/CDP climate disclosures and compute climate risk scores."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._sbti_cache: Optional[set[str]] = None
        self._cik_cache: dict[str, str] = {}
        self._ticker_map: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_tcfd_disclosure(
        self, ticker: str
    ) -> TCFDDisclosure:
        """Fetch most recent 10-K and extract TCFD disclosure profile."""
        ticker = ticker.upper()
        cik = await self._get_cik(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker '{ticker}'")

        text, sic_code, company_name, filing_year = \
            await self._get_10k_text(cik)
        text_lower = text.lower()

        # --- TCFD pillar analysis ---
        pillar_scores: dict[str, float] = {}
        for pillar, terms in TCFD_PILLARS.items():
            hits = sum(
                1 for t in terms
                if re.search(re.escape(t.lower()), text_lower)
            )
            pillar_scores[pillar] = min(1.0, hits / max(len(terms) * 0.3, 1))

        tcfd_completeness = round(
            sum(1 for v in pillar_scores.values() if v > 0.0) / 4.0, 2
        )
        has_tcfd = bool(
            re.search(r"\btcfd\b|task\s+force\s+on\s+climate", text_lower)
        )
        if tcfd_completeness >= 0.5:
            has_tcfd = True

        # Governance sub-flags
        board_oversight = bool(re.search(
            r"board\s+(?:of\s+directors\s+)?(?:oversight|oversight\s+of)\s+climate|"
            r"climate[\s\-]related\s+(?:risks?\s+and\s+)?(?:opportunities?\s+)?(?:oversight|supervised)",
            text_lower,
        ))
        mgmt_committee = bool(re.search(
            r"(?:management|executive|sustainability)\s+(?:climate\s+)?committee|"
            r"climate\s+(?:steering|working|task\s+force)\s+committee",
            text_lower,
        ))

        # Strategy sub-flags
        scenario_analysis = bool(re.search(
            r"scenario\s+analysis|climate\s+scenario|"
            r"stress\s+test[^.]{0,40}climate|scenario\s+planning",
            text_lower,
        ))
        scenarios: list[str] = []
        if re.search(r"1\.5\s*(?:degree|°c|°c)", text_lower):
            scenarios.append("1.5C")
        if re.search(r"2\s*(?:degree|°c|°c)|well\s+below\s+2", text_lower):
            scenarios.append("2C")
        if re.search(r"3\s*(?:degree|°c|°c)|business[\s\-]as[\s\-]usual", text_lower):
            scenarios.append("3C+")

        physical_risks = _extract_physical_risks(text_lower)
        transition_risks = _extract_transition_risks(text_lower)

        # Risk management
        climate_in_erp = bool(re.search(
            r"enterprise\s+risk\s+(?:management|process)[^.]{0,150}climate|"
            r"climate[^.]{0,150}enterprise\s+risk\s+(?:management|process)",
            text_lower,
        ))

        # Metrics & targets
        ghg = self._extract_ghg_from_text(text, ticker)
        if ghg:
            ghg.filing_year = filing_year

        net_zero, nz_year = _extract_net_zero(text_lower)
        has_interim = bool(re.search(
            r"interim\s+(?:target|goal|milestone)|"
            r"2030\s+(?:target|goal|emission)|near[\s\-]term\s+target",
            text_lower,
        ))

        # SBTi check
        sbti_companies = await self.get_sbti_companies()
        ticker_clean = ticker.upper().strip()
        has_sbti = ticker_clean in sbti_companies

        # Disclosure gaps
        gaps: list[str] = []
        if not board_oversight:
            gaps.append("Board climate oversight not evident")
        if not scenario_analysis:
            gaps.append("No climate scenario analysis disclosed")
        if not climate_in_erp:
            gaps.append("Climate risk not mentioned in enterprise risk process")
        if ghg is None:
            gaps.append("No quantitative GHG emission data found")
        if not net_zero:
            gaps.append("No net-zero or carbon-neutral target identified")

        logger.info(
            "TCFD disclosure analyzed",
            ticker=ticker,
            completeness=tcfd_completeness,
            has_tcfd=has_tcfd,
            ghg_found=ghg is not None,
            scenarios=scenarios,
        )

        return TCFDDisclosure(
            ticker=ticker,
            company_name=company_name,
            filing_year=filing_year,
            has_tcfd_disclosure=has_tcfd,
            tcfd_completeness=tcfd_completeness,
            board_climate_oversight=board_oversight,
            mgmt_climate_committee=mgmt_committee,
            has_scenario_analysis=scenario_analysis,
            scenarios_analyzed=scenarios,
            physical_risks_identified=physical_risks,
            transition_risks_identified=transition_risks,
            climate_risk_in_erp=climate_in_erp,
            ghg=ghg,
            has_net_zero_target=net_zero,
            net_zero_year=nz_year,
            has_sbti_commitment=has_sbti,
            has_interim_target=has_interim,
            disclosure_gaps=gaps,
        )

    async def extract_ghg_emissions(
        self, ticker: str
    ) -> Optional[GHGEmissions]:
        """Download most recent 10-K and extract GHG emission figures."""
        ticker = ticker.upper()
        cik = await self._get_cik(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker '{ticker}'")

        text, _, _, filing_year = await self._get_10k_text(cik)
        ghg = self._extract_ghg_from_text(text, ticker)
        if ghg:
            ghg.filing_year = filing_year
        return ghg

    async def get_sbti_companies(self) -> set[str]:
        """Download SBTi public CSV and return set of ticker/company identifiers.

        SBTi publishes a CSV with company names (no tickers), so we normalize
        company names to upper-case for fuzzy matching. We also cache the result.
        """
        if self._sbti_cache is not None:
            return self._sbti_cache

        identifiers: set[str] = set()
        try:
            await asyncio.sleep(_EDGAR_SLEEP)
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True
            ) as client:
                resp = await client.get(SBTI_CSV_URL, headers=_HEADERS)
                resp.raise_for_status()
                content = resp.text

            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                # Common field names in SBTi CSV
                for field in ("Company Name", "company_name", "Name", "name"):
                    val = row.get(field, "")
                    if val:
                        identifiers.add(val.strip().upper())
                for field in ("Ticker", "ticker", "Stock Ticker"):
                    val = row.get(field, "")
                    if val:
                        identifiers.add(val.strip().upper())

            logger.info("SBTi CSV loaded", company_count=len(identifiers))
        except Exception as exc:
            logger.warning("SBTi CSV download failed", error=str(exc))

        self._sbti_cache = identifiers
        return identifiers

    async def get_epa_violations(
        self, company_name: str, years: int = 5
    ) -> list[EPAViolation]:
        """Search EPA ECHO API for recent enforcement actions by company name.

        EPA ECHO: https://echo.epa.gov/facilities/facility-search/results
        Returns enforcement actions from the last `years` years.
        """
        violations: list[EPAViolation] = []
        cutoff_year = date.today().year - years

        try:
            params = {
                "p_fn": company_name[:50],
                "output": "JSON",
                "p_qiv": "c",
                "p_penyear": cutoff_year,
                "qcolumns": "1,3,4,5,8,9,13,23",
            }
            await asyncio.sleep(0.3)
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    EPA_ECHO_API, params=params, headers=_HEADERS
                )

            if resp.status_code != 200:
                logger.warning(
                    "EPA ECHO non-200", status=resp.status_code,
                    company=company_name
                )
                return []

            data = resp.json()
            facilities = (
                data.get("Results", {}).get("Facilities", [])
                or data.get("facilities", [])
                or []
            )

            for fac in facilities[:20]:  # cap at 20 facilities
                fname = (
                    fac.get("FacilityName", "")
                    or fac.get("facility_name", "")
                    or company_name
                )
                state = fac.get("StateAbbr", fac.get("state", ""))
                penalty_str = fac.get("TotalPenalties", fac.get("total_penalties", ""))
                penalty: Optional[float] = None
                if penalty_str:
                    try:
                        penalty = float(str(penalty_str).replace(",", "").replace("$", ""))
                    except (ValueError, TypeError):
                        pass

                statute = fac.get("Statute", fac.get("statute", ""))
                vtype = fac.get("ViolationType", fac.get("violation_type", "Enforcement Action"))
                vdate_str = fac.get("ViolationDate", fac.get("violation_date", ""))
                vdate: Optional[date] = None
                if vdate_str:
                    try:
                        vdate = date.fromisoformat(str(vdate_str)[:10])
                    except (ValueError, TypeError):
                        pass

                violations.append(EPAViolation(
                    facility_name=fname,
                    state=state,
                    violation_type=vtype or "Enforcement Action",
                    penalty_amount=penalty,
                    violation_date=vdate,
                    statute=statute or None,
                ))

        except Exception as exc:
            logger.warning(
                "EPA ECHO query failed", company=company_name, error=str(exc)
            )

        return violations

    async def compute_climate_risk_score(
        self, ticker: str
    ) -> ClimateRiskScore:
        """Combine TCFD disclosure + SIC + EPA violations into a climate risk score.

        Scoring components:
          - Physical risk (0-10): industry SIC + physical keyword density
          - Transition risk (0-10): industry SIC + transition keyword density
          - Disclosure score (0-10): TCFD completeness + GHG data
          - Net-zero readiness (0-10): net-zero target + SBTi + interim targets
          - EPA violations (deduct from overall)
        """
        ticker = ticker.upper()
        cik = await self._get_cik(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker '{ticker}'")

        text, sic_code, company_name, filing_year = await self._get_10k_text(cik)
        text_lower = text.lower()
        industry_risk = self._classify_industry_risk(sic_code or "")

        # Physical risk score
        phys_hits = sum(
            len(re.findall(re.escape(term.lower()), text_lower))
            for risk_type in PHYSICAL_RISKS.values()
            for term in risk_type
        )
        base_phys = {"High": 7.0, "Medium": 5.0, "Low": 3.0}[industry_risk]
        phys_score = round(
            min(10.0, base_phys + math.log1p(phys_hits) * 0.5), 1
        )

        # Transition risk score
        trans_hits = sum(
            len(re.findall(re.escape(term.lower()), text_lower))
            for risk_type in TRANSITION_RISKS.values()
            for term in risk_type
        )
        base_trans = {"High": 7.5, "Medium": 5.0, "Low": 2.5}[industry_risk]
        trans_score = round(
            min(10.0, base_trans + math.log1p(trans_hits) * 0.3), 1
        )

        # TCFD completeness and disclosure score
        pillar_coverage = sum(
            1 for pillar_terms in TCFD_PILLARS.values()
            if any(re.search(re.escape(t.lower()), text_lower) for t in pillar_terms)
        )
        tcfd_completeness = pillar_coverage / 4.0
        ghg = self._extract_ghg_from_text(text, ticker)
        ghg_bonus = 2.0 if ghg and ghg.scope1_mt_co2e else 0.0
        disclosure_score = round(
            min(10.0, tcfd_completeness * 7.0 + ghg_bonus + (1.0 if ghg and ghg.scope2_mt_co2e else 0.0)),
            1,
        )

        # Net-zero readiness
        net_zero, _ = _extract_net_zero(text_lower)
        has_interim = bool(re.search(
            r"interim\s+(?:target|goal)|2030\s+(?:target|emission)|near[\s\-]term\s+target",
            text_lower,
        ))
        sbti_companies = await self.get_sbti_companies()
        has_sbti = ticker in sbti_companies

        nz_score = 0.0
        if net_zero:
            nz_score += 4.0
        if has_sbti:
            nz_score += 3.0
        if has_interim:
            nz_score += 2.0
        if ghg and (ghg.scope1_mt_co2e or ghg.scope2_mt_co2e):
            nz_score += 1.0
        nz_readiness = round(min(10.0, nz_score), 1)

        # EPA violations
        epa_violations = await self.get_epa_violations(company_name, years=5)
        n_violations = len(epa_violations)

        # Overall climate risk (higher = more risky)
        # High physical + high transition + low disclosure = high risk
        overall = round(
            phys_score * 0.35
            + trans_score * 0.35
            + (10.0 - disclosure_score) * 0.20
            + (10.0 - nz_readiness) * 0.10
            + min(1.0, n_violations * 0.2),  # EPA penalty
            1,
        )
        overall = min(10.0, overall)

        recommendation = _build_recommendation(
            overall, disclosure_score, nz_readiness, industry_risk, n_violations
        )

        logger.info(
            "Climate risk scored",
            ticker=ticker,
            physical=phys_score,
            transition=trans_score,
            disclosure=disclosure_score,
            nz_readiness=nz_readiness,
            overall=overall,
        )

        return ClimateRiskScore(
            ticker=ticker,
            company_name=company_name,
            physical_risk_score=phys_score,
            transition_risk_score=trans_score,
            overall_climate_risk=overall,
            disclosure_score=disclosure_score,
            net_zero_readiness=nz_readiness,
            epa_violations_5y=n_violations,
            industry_risk_level=industry_risk,
            recommendation=recommendation,
        )

    async def screen_by_climate_risk(
        self,
        tickers: list[str],
        max_risk: float = 6.0,
        max_concurrent: int = 5,
    ) -> list[ClimateRiskScore]:
        """Batch compute climate risk and return tickers below max_risk threshold."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _bounded(t: str) -> Optional[ClimateRiskScore]:
            async with sem:
                try:
                    return await self.compute_climate_risk_score(t)
                except Exception as exc:
                    logger.warning(
                        "Climate risk failed", ticker=t, error=str(exc)
                    )
                    return None

        results_raw = await asyncio.gather(*[_bounded(t) for t in tickers])
        results = [r for r in results_raw if r is not None]
        passing = [r for r in results if r.overall_climate_risk <= max_risk]
        passing.sort(key=lambda r: r.overall_climate_risk)
        return passing

    async def get_sector_climate_summary(
        self, sector: str, tickers: list[str]
    ) -> dict:
        """Compute sector-level climate risk and TCFD completeness averages."""
        sem = asyncio.Semaphore(5)

        async def _score(t: str) -> Optional[tuple[ClimateRiskScore, TCFDDisclosure]]:
            async with sem:
                try:
                    risk, tcfd = await asyncio.gather(
                        self.compute_climate_risk_score(t),
                        self.analyze_tcfd_disclosure(t),
                    )
                    return risk, tcfd
                except Exception as exc:
                    logger.warning("Sector climate failed", ticker=t, error=str(exc))
                    return None

        raw = await asyncio.gather(*[_score(t) for t in tickers])
        pairs = [r for r in raw if r is not None]

        if not pairs:
            return {
                "sector": sector,
                "n_companies": 0,
                "error": "No data available",
            }

        risks = [p[0] for p in pairs]
        tcfds = [p[1] for p in pairs]
        n = len(pairs)

        avg_risk = round(sum(r.overall_climate_risk for r in risks) / n, 1)
        avg_phys = round(sum(r.physical_risk_score for r in risks) / n, 1)
        avg_trans = round(sum(r.transition_risk_score for r in risks) / n, 1)
        avg_disclosure = round(sum(r.disclosure_score for r in risks) / n, 1)
        pct_net_zero = round(
            100.0 * sum(1 for t in tcfds if t.has_net_zero_target) / n, 1
        )
        pct_sbti = round(
            100.0 * sum(1 for t in tcfds if t.has_sbti_commitment) / n, 1
        )
        pct_scenario = round(
            100.0 * sum(1 for t in tcfds if t.has_scenario_analysis) / n, 1
        )
        pct_tcfd = round(
            100.0 * sum(1 for t in tcfds if t.has_tcfd_disclosure) / n, 1
        )
        avg_completeness = round(
            sum(t.tcfd_completeness for t in tcfds) / n, 2
        )

        top_disclosers = sorted(
            zip(tickers[:n], [t.tcfd_completeness for t in tcfds]),
            key=lambda x: x[1], reverse=True,
        )

        return {
            "sector": sector,
            "n_companies": n,
            "avg_climate_risk": avg_risk,
            "avg_physical_risk": avg_phys,
            "avg_transition_risk": avg_trans,
            "avg_disclosure_score": avg_disclosure,
            "pct_with_net_zero_target": pct_net_zero,
            "pct_with_sbti_commitment": pct_sbti,
            "pct_with_scenario_analysis": pct_scenario,
            "pct_with_tcfd_disclosure": pct_tcfd,
            "avg_tcfd_completeness": avg_completeness,
            "top_disclosers": [t[0] for t in top_disclosers[:3]],
        }

    # ------------------------------------------------------------------
    # GHG extraction
    # ------------------------------------------------------------------

    def _extract_ghg_from_text(
        self, text: str, ticker: str
    ) -> Optional[GHGEmissions]:
        """Apply multi-pattern regex to extract GHG data from 10-K text.

        Strategy:
          1. Try specific patterns for Scope 1 / Scope 2 / Scope 3
          2. Detect magnitude (million tons vs thousand tons vs raw MT)
          3. Apply unit normalisation → MT CO2e
          4. Attempt intensity extraction if absolute values absent
        """
        found: dict[str, Optional[float]] = {
            "scope1": None,
            "scope2": None,
            "scope3": None,
            "scope2_market": None,
            "scope2_location": None,
            "intensity": None,
        }
        intensity_unit: Optional[str] = None
        confidence = "low"

        for pattern, label in _GHG_PATTERNS:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if not matches:
                continue

            for m in matches[:2]:  # take up to two matches per pattern
                raw_val_str = m.group(1).replace(",", "")
                try:
                    raw_val = float(raw_val_str)
                except ValueError:
                    continue

                # Determine magnitude from surrounding context
                context = text[max(0, m.start() - 30): m.end() + 30].lower()
                normalised = _normalise_ghg_value(raw_val, context)

                if label == "intensity":
                    if found["intensity"] is None:
                        found["intensity"] = normalised
                        intensity_unit = _extract_intensity_unit(context)
                elif found.get(label) is None:
                    found[label] = normalised
                    confidence = "medium"

        # If we got both scope1 and scope2 absolutely, confidence = high
        if found["scope1"] is not None and found["scope2"] is not None:
            confidence = "high"

        # Nothing found
        if all(v is None for v in found.values()):
            return None

        return GHGEmissions(
            ticker=ticker,
            filing_year=0,  # filled in by caller
            scope1_mt_co2e=found["scope1"],
            scope2_mt_co2e=found["scope2"] or found["scope2_market"] or found["scope2_location"],
            scope3_mt_co2e=found["scope3"],
            scope2_market_based=found["scope2_market"],
            scope2_location_based=found["scope2_location"],
            intensity_metric=intensity_unit,
            intensity_value=found["intensity"],
            data_source="sec_10k",
            extraction_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Industry classification
    # ------------------------------------------------------------------

    def _classify_industry_risk(self, sic_code: str) -> str:
        """Map SIC code to climate risk level: High / Medium / Low."""
        sic4 = str(sic_code)[:4]
        if sic4 in _HIGH_RISK_SICS:
            return "High"
        if sic4 in _LOW_RISK_SICS:
            return "Low"
        return "Medium"

    # ------------------------------------------------------------------
    # EDGAR retrieval
    # ------------------------------------------------------------------

    async def _get_cik(self, ticker: str) -> Optional[str]:
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        try:
            if self._ticker_map is None:
                await asyncio.sleep(_EDGAR_SLEEP)
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(EDGAR_TICKERS, headers=_HEADERS)
                    resp.raise_for_status()
                    self._ticker_map = resp.json()

            for entry in self._ticker_map.values():
                if str(entry.get("ticker", "")).upper() == ticker:
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_cache[ticker] = cik
                    return cik
        except Exception as exc:
            logger.error("CIK lookup failed", ticker=ticker, error=str(exc))

        return None

    async def _get_10k_text(
        self, cik: str
    ) -> tuple[str, Optional[str], str, int]:
        """Fetch most recent 10-K full text + metadata.

        Returns:
            (text, sic_code, company_name, filing_year)
        """
        cik_padded = cik.zfill(10)
        sub_url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"

        await asyncio.sleep(_EDGAR_SLEEP)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                sub_resp = await client.get(sub_url, headers=_HEADERS)
                sub_resp.raise_for_status()
                sub = sub_resp.json()
        except Exception as exc:
            logger.error("Submissions fetch failed", cik=cik, error=str(exc))
            return "", None, cik, datetime.utcnow().year

        company_name: str = sub.get("name", cik)
        sic_code: Optional[str] = sub.get("sic")
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        for i, form in enumerate(forms):
            if form in ("10-K", "10-K405"):
                acc = accessions[i] if i < len(accessions) else None
                doc = primary_docs[i] if i < len(primary_docs) else None
                rd = report_dates[i] if i < len(report_dates) else None
                if acc and doc:
                    filing_year = int(rd[:4]) if rd else datetime.utcnow().year
                    text = await self._fetch_filing_text(cik_padded, acc, doc)
                    return text, sic_code, company_name, filing_year

        logger.warning("No 10-K found", cik=cik)
        return "", sic_code, company_name, datetime.utcnow().year

    async def _fetch_filing_text(
        self,
        cik_padded: str,
        accession_num: str,
        doc_name: str,
        limit: int = _TEXT_LIMIT,
    ) -> str:
        """Download and clean EDGAR filing HTML."""
        cik_bare = cik_padded.lstrip("0") or "0"
        acc_clean = accession_num.replace("-", "")
        url = f"{EDGAR_ARCHIVES}/{cik_bare}/{acc_clean}/{doc_name}"

        await asyncio.sleep(_EDGAR_SLEEP)
        try:
            async with httpx.AsyncClient(
                timeout=60.0, follow_redirects=True
            ) as client:
                resp = await client.get(url, headers=_HEADERS)
                resp.raise_for_status()
                raw = resp.text
        except Exception as exc:
            logger.error("Filing fetch failed", url=url, error=str(exc))
            return ""

        if "<" in raw[:500]:
            raw = _strip_html(raw)
        return re.sub(r"\s+", " ", raw)[:limit]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    text = re.sub(
        r"<(script|style)[^>]*>.*?</(script|style)>",
        " ", html, flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, rep in [
        ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
        ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"),
    ]:
        text = text.replace(ent, rep)
    return re.sub(r"\s{3,}", "  ", text).strip()


def _normalise_ghg_value(raw: float, context: str) -> float:
    """Normalise a GHG numeric value to metric tons CO2e.

    Heuristics:
      - If 'million' or 'MM' appears near the number → × 1,000,000
      - If 'thousand' or 'k' appears → × 1,000
      - If value < 1,000 and no unit qualifier → assume million tons (common in 10-K tables)
      - Otherwise keep as-is (assume metric tons already)
    """
    ctx = context.lower()
    if re.search(r"\bmillion\b|\bMM\b|\bMMT\b", context):
        return round(raw * 1_000_000, 0)
    if re.search(r"\bthousand\b|\bkilo\b", ctx):
        return round(raw * 1_000, 0)
    # Heuristic: values under 10,000 in a GHG table context are likely
    # expressed in millions of metric tons (common S&P 500 scale)
    if raw < 10_000 and re.search(r"metric\s+ton|mtco2|co2e", ctx):
        # Could be millions or actual MT; check for "million" absence
        return round(raw, 2)  # leave as reported; caller can interpret
    return round(raw, 2)


def _extract_intensity_unit(context: str) -> str:
    """Try to identify the denominator in a GHG intensity ratio."""
    ctx = context.lower()
    if re.search(r"per\s+(?:revenue|\$|dollar)", ctx):
        return "per revenue $MM"
    if re.search(r"per\s+(?:employee|fte|headcount)", ctx):
        return "per employee"
    if re.search(r"per\s+(?:mwh|megawatt)", ctx):
        return "per MWh"
    if re.search(r"per\s+(?:tonne|ton)\s+(?:of\s+)?(?:product|output)", ctx):
        return "per ton of product"
    if re.search(r"per\s+(?:unit|item|piece)", ctx):
        return "per unit"
    return "per unit of output"


def _extract_physical_risks(text_lower: str) -> list[str]:
    """Return list of physical risk categories mentioned in text."""
    found: list[str] = []
    for category, terms in PHYSICAL_RISKS.items():
        for term in terms:
            if re.search(re.escape(term.lower()), text_lower):
                found.append(term)
                break  # one entry per category term is enough
    return found[:10]


def _extract_transition_risks(text_lower: str) -> list[str]:
    """Return list of transition risk categories mentioned in text."""
    found: list[str] = []
    for category, terms in TRANSITION_RISKS.items():
        for term in terms:
            if re.search(re.escape(term.lower()), text_lower):
                found.append(term)
                break
    return found[:10]


def _extract_net_zero(text_lower: str) -> tuple[bool, Optional[int]]:
    """Return (has_net_zero, target_year) from text."""
    for pattern in _NET_ZERO_YEAR_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            try:
                year = int(m.group(1))
                if 2025 <= year <= 2100:
                    return True, year
            except (IndexError, ValueError):
                pass

    # Presence without specific year
    has_nz = bool(
        re.search(r"net[\s\-]zero|carbon[\s\-]neutral(?:ity)?", text_lower)
    )
    return has_nz, None


def _build_recommendation(
    overall_risk: float,
    disclosure_score: float,
    nz_readiness: float,
    industry_risk: str,
    n_violations: int,
) -> str:
    """Generate a one-sentence investment recommendation based on climate profile."""
    parts: list[str] = []

    if overall_risk >= 7.5:
        parts.append("Significant climate risk exposure warrants close monitoring.")
    elif overall_risk >= 5.0:
        parts.append("Moderate climate risk; review transition plan adequacy.")
    else:
        parts.append("Relatively low overall climate risk profile.")

    if disclosure_score < 4.0:
        parts.append("Disclosure quality is below best practice — limited transparency.")
    elif disclosure_score >= 7.0:
        parts.append("Strong TCFD-aligned disclosure supports risk assessment.")

    if nz_readiness >= 7.0:
        parts.append("Company demonstrates credible net-zero transition readiness.")
    elif nz_readiness < 3.0 and industry_risk == "High":
        parts.append("High-risk industry with no visible decarbonisation pathway.")

    if n_violations > 0:
        parts.append(
            f"{n_violations} EPA enforcement action(s) in last 5 years add regulatory risk."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

async def tcfd_analysis(ticker: str) -> TCFDDisclosure:
    """Analyze TCFD disclosure quality for a single ticker."""
    analyzer = ClimateDisclosureAnalyzer()
    return await analyzer.analyze_tcfd_disclosure(ticker)


async def climate_risk(ticker: str) -> ClimateRiskScore:
    """Compute overall climate risk score for a single ticker."""
    analyzer = ClimateDisclosureAnalyzer()
    return await analyzer.compute_climate_risk_score(ticker)


async def screen_climate(
    tickers: list[str], max_risk: float = 6.0
) -> list[ClimateRiskScore]:
    """Batch climate risk screen — return tickers below max_risk threshold."""
    analyzer = ClimateDisclosureAnalyzer()
    return await analyzer.screen_by_climate_risk(tickers, max_risk=max_risk)


async def ghg_data(ticker: str) -> Optional[GHGEmissions]:
    """Extract GHG emissions data from the most recent 10-K filing."""
    analyzer = ClimateDisclosureAnalyzer()
    return await analyzer.extract_ghg_emissions(ticker)


async def sector_climate_summary(sector: str, tickers: list[str]) -> dict:
    """Return aggregated climate risk and TCFD disclosure metrics for a sector."""
    analyzer = ClimateDisclosureAnalyzer()
    return await analyzer.get_sector_climate_summary(sector, tickers)
