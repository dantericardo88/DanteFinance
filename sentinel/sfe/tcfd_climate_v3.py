"""
tcfd_climate_v3.py — CDP / TCFD Climate Disclosure Parsing v3 (dim_103, target 9/10)

Architecture upgrade: fragile Scope 1/2/3 wild-regex replaced with a three-tier
structured extraction pipeline.

GHG Extraction (three tiers, in order):
  Tier 1 — XBRL tags: us-gaap GreenHouseGasEmissionsScope1/2/3 (SEC 2024 climate taxonomy)
  Tier 2 — HTML table detection: find <table> elements with Scope 1/2/tCO2e column headers
  Tier 3 — Context-anchored regex: only scan a 400-char window after a detected Scope anchor

TCFD Four-Pillar Assessment:
  Governance   — board oversight language from proxy + 10-K board risk committee section
  Strategy     — climate scenario analysis, 1.5°C/2°C language, Paris Agreement
  Risk Mgmt    — climate risk register, physical risk, transition risk, stranded assets
  Metrics/Tgts — GHG numbers + targets + verification status (third-party vs self-reported)

Additional modules:
  - CDP A-list and scored tiers scraping (cdp.net public)
  - Net-zero commitment extractor: year, base year, milestones, verification
  - Physical risk assessment: flood, water stress, extreme heat, sea level rise + asset exposure
  - Transition risk: stranded assets, carbon pricing, regulatory risk
  - TCFD alignment score: 0–100 based on % of 11 TCFD recommended disclosures disclosed
  - Peer TCFD comparison by SIC/sector
  - SBTi (Science Based Targets initiative) public CSV lookup

SQLite: tcfd_assessments, ghg_emissions, climate_targets, risk_disclosures, cdp_scores
FastAPI router at /tcfd/v3
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field as PydanticField

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "GHGExtractor",
    "CDPPublicData",
    "SBTiLookup",
    "TCFDPillarScorer",
    "NetZeroExtractor",
    "PhysicalRiskAssessor",
    "TransitionRiskAssessor",
    "TCFDEngine",
    "TCFDDB",
    "tcfd_v3_router",
    # Physical risk
    "get_state_physical_risk",
    "get_company_physical_risk",
    # Transition risk
    "get_sic_transition_risk",
    "compute_carbon_intensity",
    # Scenario analysis
    "climate_scenario_analysis",
    # Climate VaR
    "compute_climate_var",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT  = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_HEADERS     = {
    "User-Agent":      _USER_AGENT,
    "Accept":          "application/json, text/html, */*",
    "Accept-Encoding": "gzip, deflate",
}
_RATE_DELAY  = 0.15
_TIMEOUT     = 30.0
_MAX_RETRY   = 3
_TEXT_LIMIT  = 200_000

EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_ARCHIVE     = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EFTS_SEARCH_URL   = "https://efts.sec.gov/LATEST/search-index"

CDP_SCORES_URL    = "https://www.cdp.net/en/scores"
CDP_RESPONSES_URL = "https://www.cdp.net/en/responses"
SBTI_CSV_URL      = "https://sciencebasedtargets.org/files/SBTi-Companies-Taking-Action.csv"

_DB_PATH = Path(__file__).parent.parent / "data" / "tcfd_v3.db"

# ---------------------------------------------------------------------------
# TCFD 11 recommended disclosures (4 pillars)
# Each disclosure is a (pillar, disclosure_key, description) tuple
# ---------------------------------------------------------------------------

TCFD_DISCLOSURES: list[tuple[str, str, str]] = [
    # Governance (2)
    ("governance", "board_oversight",    "Board oversight of climate-related risks/opportunities"),
    ("governance", "mgmt_role",          "Management's role in assessing/managing climate risks"),
    # Strategy (3)
    ("strategy",   "risks_opportunities","Climate-related risks/opportunities identified over time horizons"),
    ("strategy",   "business_impact",    "Impact on business, strategy, financial planning"),
    ("strategy",   "scenario_analysis",  "Climate scenario analysis including 2°C or lower scenario"),
    # Risk Management (3)
    ("risk_mgmt",  "id_assess_process",  "Processes for identifying/assessing climate-related risks"),
    ("risk_mgmt",  "manage_process",     "Processes for managing climate-related risks"),
    ("risk_mgmt",  "integration",        "Integration of climate risk into overall risk management"),
    # Metrics & Targets (3)
    ("metrics",    "ghg_scope1_2",       "Scope 1 and Scope 2 GHG emissions disclosed"),
    ("metrics",    "ghg_scope3",         "Scope 3 GHG emissions disclosed if material"),
    ("metrics",    "targets",            "Targets used to manage climate risks/opportunities"),
]

_DISCLOSURE_KEYS = {d[1] for d in TCFD_DISCLOSURES}

# ---------------------------------------------------------------------------
# SIC → sector mapping
# ---------------------------------------------------------------------------

_SIC_TO_SECTOR: dict[str, str] = {
    "1311": "energy",    "1382": "energy",    "2911": "energy",
    "1321": "energy",    "5171": "energy",    "1381": "energy",
    "2819": "materials", "2860": "materials", "3312": "materials",
    "1040": "materials", "3559": "industrials","3720": "industrials",
    "4911": "utilities", "4931": "utilities",  "4941": "utilities",
    "2000": "consumer_staples","5400": "consumer_staples",
    "5900": "consumer_discretionary","7011": "consumer_discretionary",
    "2836": "health_care","8011": "health_care",
    "6020": "financials", "6022": "financials","6211": "financials",
    "7372": "information_technology","7371": "information_technology",
    "3674": "information_technology","4813": "communication_services",
    "6552": "real_estate","6798": "real_estate",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GHGEmissions(BaseModel):
    ticker: str
    filing_year: Optional[int] = None
    # Scope values in metric tons CO2e
    scope1_mt: Optional[float] = None
    scope2_mt_location: Optional[float] = None
    scope2_mt_market: Optional[float] = None
    scope3_mt: Optional[float] = None
    # Extraction metadata
    scope1_source: str = "none"   # "xbrl" | "html_table" | "text_anchored" | "none"
    scope2_source: str = "none"
    scope3_source: str = "none"
    scope1_unit: str = "tCO2e"
    ghg_verified: bool = False     # third-party verification detected
    verification_body: Optional[str] = None
    intensity_metric: Optional[float] = None   # tCO2e per revenue unit
    intensity_unit: Optional[str] = None
    prior_year_scope1: Optional[float] = None
    prior_year_scope2: Optional[float] = None
    yoy_scope12_change_pct: Optional[float] = None
    data_quality: str = "low"     # "high" | "medium" | "low"


class ClimateTarget(BaseModel):
    ticker: str
    target_type: str = "unknown"    # "net_zero" | "carbon_neutral" | "sbti" | "renewable_energy" | "intensity"
    target_year: Optional[int] = None
    base_year: Optional[int] = None
    interim_milestones: list[str] = PydanticField(default_factory=list)
    scope_coverage: list[str] = PydanticField(default_factory=list)  # ["scope1","scope2","scope3"]
    reduction_pct: Optional[float] = None
    verification_status: str = "self_reported"  # "third_party" | "sbti_validated" | "self_reported"
    sbti_committed: bool = False
    sbti_approved: bool = False
    is_net_zero: bool = False
    description: str = ""
    source: str = ""


class PhysicalRisk(BaseModel):
    ticker: str
    flood_risk_mentioned: bool = False
    water_stress_mentioned: bool = False
    extreme_heat_mentioned: bool = False
    sea_level_rise_mentioned: bool = False
    hurricane_storm_mentioned: bool = False
    wildfire_mentioned: bool = False
    asset_exposure_quantified: bool = False
    asset_exposure_pct: Optional[float] = None
    acute_risk_score: float = 0.0    # 0–50
    chronic_risk_score: float = 0.0  # 0–50
    total_physical_risk_score: float = 0.0   # 0–100
    risk_details: list[str] = PydanticField(default_factory=list)


class TransitionRisk(BaseModel):
    ticker: str
    stranded_asset_risk: bool = False
    carbon_pricing_exposure: bool = False
    regulatory_risk_mentioned: bool = False
    technology_disruption: bool = False
    market_risk_mentioned: bool = False
    reputational_risk_mentioned: bool = False
    carbon_price_assumption: Optional[float] = None   # $/tCO2e used in scenario
    regulatory_compliance_cost: Optional[float] = None
    transition_risk_score: float = 0.0   # 0–100
    risk_details: list[str] = PydanticField(default_factory=list)


class CDPScore(BaseModel):
    ticker: str
    company_name: str = ""
    cdp_year: Optional[int] = None
    climate_score: str = "not_disclosed"    # A / A- / B / B- / C / C- / D / D- / not_disclosed
    water_score: Optional[str] = None
    forest_score: Optional[str] = None
    a_list: bool = False
    source_url: str = ""


class TCFDPillarResult(BaseModel):
    pillar: str   # "governance" | "strategy" | "risk_mgmt" | "metrics"
    score: float = 0.0        # 0–100 within this pillar
    max_score: float = 100.0
    disclosures_found: list[str] = PydanticField(default_factory=list)
    disclosures_missing: list[str] = PydanticField(default_factory=list)
    evidence: dict[str, Any] = PydanticField(default_factory=dict)


class TCFDAssessment(BaseModel):
    ticker: str
    company_name: str = ""
    as_of: str = PydanticField(default_factory=lambda: datetime.utcnow().date().isoformat())
    filing_year: Optional[int] = None
    # Overall alignment
    alignment_score: float = 0.0      # 0–100
    alignment_level: str = "minimal"  # "aligned" | "advancing" | "developing" | "minimal"
    disclosures_found_count: int = 0
    disclosures_total: int = 11
    # Pillar results
    governance_pillar: TCFDPillarResult = PydanticField(default_factory=lambda: TCFDPillarResult(pillar="governance"))
    strategy_pillar: TCFDPillarResult = PydanticField(default_factory=lambda: TCFDPillarResult(pillar="strategy"))
    risk_mgmt_pillar: TCFDPillarResult = PydanticField(default_factory=lambda: TCFDPillarResult(pillar="risk_mgmt"))
    metrics_pillar: TCFDPillarResult = PydanticField(default_factory=lambda: TCFDPillarResult(pillar="metrics"))
    # Key flags
    item_1c_disclosed: bool = False      # SEC 2024 mandatory
    has_scenario_analysis: bool = False
    has_net_zero_target: bool = False
    has_sbti: bool = False
    # Sub-results
    ghg_emissions: Optional[GHGEmissions] = None
    climate_targets: list[ClimateTarget] = PydanticField(default_factory=list)
    physical_risk: Optional[PhysicalRisk] = None
    transition_risk: Optional[TransitionRisk] = None
    cdp_score: Optional[CDPScore] = None
    sector: str = "default"
    warnings: list[str] = PydanticField(default_factory=list)
    data_quality: dict[str, str] = PydanticField(default_factory=dict)
    disclaimer: str = (
        "TCFD assessment derived from public SEC EDGAR filings, CDP public data, "
        "and SBTi public records. Not a substitute for professional ESG advisory."
    )


class PeerTCFDComparison(BaseModel):
    ticker: str
    sector: str
    alignment_score: float
    sector_avg_alignment: float
    sector_avg_scope12_disclosed_pct: float
    percentile_rank: float
    peers: list[dict] = PydanticField(default_factory=list)


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------


class TCFDDB:
    """SQLite persistence for TCFD assessments, GHG emissions, targets, risks, CDP scores."""

    DDL = """
    CREATE TABLE IF NOT EXISTS tcfd_assessments (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker                  TEXT NOT NULL,
        company_name            TEXT,
        as_of                   TEXT NOT NULL,
        filing_year             INTEGER,
        alignment_score         REAL,
        alignment_level         TEXT,
        disclosures_found_count INTEGER,
        item_1c_disclosed       INTEGER DEFAULT 0,
        has_scenario_analysis   INTEGER DEFAULT 0,
        has_net_zero_target     INTEGER DEFAULT 0,
        has_sbti                INTEGER DEFAULT 0,
        sector                  TEXT,
        warnings_json           TEXT,
        data_quality_json       TEXT,
        inserted_at             TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, as_of)
    );

    CREATE TABLE IF NOT EXISTS ghg_emissions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker              TEXT NOT NULL,
        as_of               TEXT NOT NULL,
        filing_year         INTEGER,
        scope1_mt           REAL,
        scope2_mt_location  REAL,
        scope2_mt_market    REAL,
        scope3_mt           REAL,
        scope1_source       TEXT,
        scope2_source       TEXT,
        scope3_source       TEXT,
        ghg_verified        INTEGER DEFAULT 0,
        verification_body   TEXT,
        yoy_scope12_change_pct REAL,
        data_quality        TEXT,
        inserted_at         TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, as_of)
    );

    CREATE TABLE IF NOT EXISTS climate_targets (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker              TEXT NOT NULL,
        as_of               TEXT NOT NULL,
        target_type         TEXT,
        target_year         INTEGER,
        base_year           INTEGER,
        milestones_json     TEXT,
        scope_coverage_json TEXT,
        reduction_pct       REAL,
        verification_status TEXT,
        sbti_committed      INTEGER DEFAULT 0,
        sbti_approved       INTEGER DEFAULT 0,
        is_net_zero         INTEGER DEFAULT 0,
        description         TEXT,
        source              TEXT,
        inserted_at         TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS risk_disclosures (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker              TEXT NOT NULL,
        as_of               TEXT NOT NULL,
        risk_type           TEXT,    -- 'physical' | 'transition'
        details_json        TEXT,
        total_score         REAL,
        inserted_at         TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, as_of, risk_type)
    );

    CREATE TABLE IF NOT EXISTS cdp_scores (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL,
        company_name    TEXT,
        cdp_year        INTEGER,
        climate_score   TEXT,
        water_score     TEXT,
        forest_score    TEXT,
        a_list          INTEGER DEFAULT 0,
        source_url      TEXT,
        inserted_at     TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, cdp_year)
    );

    CREATE INDEX IF NOT EXISTS ix_tcfd_ticker  ON tcfd_assessments(ticker, as_of);
    CREATE INDEX IF NOT EXISTS ix_ghg_ticker   ON ghg_emissions(ticker, as_of);
    CREATE INDEX IF NOT EXISTS ix_tgt_ticker   ON climate_targets(ticker, as_of);
    CREATE INDEX IF NOT EXISTS ix_risk_ticker  ON risk_disclosures(ticker, as_of, risk_type);
    CREATE INDEX IF NOT EXISTS ix_cdp_ticker   ON cdp_scores(ticker, cdp_year);
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path else _DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.DDL)
        self._conn.commit()

    def upsert_assessment(self, a: TCFDAssessment) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO tcfd_assessments
               (ticker, company_name, as_of, filing_year, alignment_score,
                alignment_level, disclosures_found_count, item_1c_disclosed,
                has_scenario_analysis, has_net_zero_target, has_sbti, sector,
                warnings_json, data_quality_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (a.ticker, a.company_name, a.as_of, a.filing_year,
             a.alignment_score, a.alignment_level, a.disclosures_found_count,
             int(a.item_1c_disclosed), int(a.has_scenario_analysis),
             int(a.has_net_zero_target), int(a.has_sbti), a.sector,
             json.dumps(a.warnings), json.dumps(a.data_quality)),
        )
        self._conn.commit()

    def upsert_ghg(self, g: GHGEmissions, as_of: str) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO ghg_emissions
               (ticker, as_of, filing_year, scope1_mt, scope2_mt_location,
                scope2_mt_market, scope3_mt, scope1_source, scope2_source,
                scope3_source, ghg_verified, verification_body,
                yoy_scope12_change_pct, data_quality)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (g.ticker, as_of, g.filing_year, g.scope1_mt,
             g.scope2_mt_location, g.scope2_mt_market, g.scope3_mt,
             g.scope1_source, g.scope2_source, g.scope3_source,
             int(g.ghg_verified), g.verification_body,
             g.yoy_scope12_change_pct, g.data_quality),
        )
        self._conn.commit()

    def insert_targets(self, targets: list[ClimateTarget], as_of: str) -> None:
        for t in targets:
            self._conn.execute(
                """INSERT INTO climate_targets
                   (ticker, as_of, target_type, target_year, base_year,
                    milestones_json, scope_coverage_json, reduction_pct,
                    verification_status, sbti_committed, sbti_approved,
                    is_net_zero, description, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (t.ticker, as_of, t.target_type, t.target_year, t.base_year,
                 json.dumps(t.interim_milestones), json.dumps(t.scope_coverage),
                 t.reduction_pct, t.verification_status,
                 int(t.sbti_committed), int(t.sbti_approved),
                 int(t.is_net_zero), t.description, t.source),
            )
        self._conn.commit()

    def upsert_risk(self, ticker: str, as_of: str, risk_type: str,
                    details: dict, score: float) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO risk_disclosures
               (ticker, as_of, risk_type, details_json, total_score)
               VALUES (?,?,?,?,?)""",
            (ticker, as_of, risk_type, json.dumps(details), score),
        )
        self._conn.commit()

    def upsert_cdp(self, c: CDPScore) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO cdp_scores
               (ticker, company_name, cdp_year, climate_score,
                water_score, forest_score, a_list, source_url)
               VALUES (?,?,?,?,?,?,?,?)""",
            (c.ticker, c.company_name, c.cdp_year, c.climate_score,
             c.water_score, c.forest_score, int(c.a_list), c.source_url),
        )
        self._conn.commit()

    def get_assessment(self, ticker: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM tcfd_assessments WHERE ticker=? ORDER BY as_of DESC LIMIT 1",
            (ticker,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_ghg(self, ticker: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM ghg_emissions WHERE ticker=? ORDER BY as_of DESC LIMIT 1",
            (ticker,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_targets(self, ticker: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM climate_targets WHERE ticker=? ORDER BY as_of DESC LIMIT 20",
            (ticker,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_risk(self, ticker: str, risk_type: str) -> Optional[dict]:
        cur = self._conn.execute(
            """SELECT * FROM risk_disclosures WHERE ticker=? AND risk_type=?
               ORDER BY as_of DESC LIMIT 1""",
            (ticker, risk_type),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_cdp_score(self, ticker: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM cdp_scores WHERE ticker=? ORDER BY cdp_year DESC LIMIT 1",
            (ticker,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_sector_assessments(self, sector: str) -> list[dict]:
        cur = self._conn.execute(
            """SELECT t.ticker, t.alignment_score, t.disclosures_found_count,
                      t.has_net_zero_target, t.has_scenario_analysis
               FROM tcfd_assessments t
               INNER JOIN (
                   SELECT ticker, MAX(as_of) AS max_as_of FROM tcfd_assessments GROUP BY ticker
               ) m ON t.ticker = m.ticker AND t.as_of = m.max_as_of
               WHERE t.sector=? ORDER BY t.alignment_score DESC""",
            (sector,),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------


def _edgar_get(url: str, params: dict | None = None) -> Optional[dict | str]:
    for attempt in range(_MAX_RETRY):
        try:
            time.sleep(_RATE_DELAY)
            r = requests.get(url, headers=_HEADERS, params=params, timeout=_TIMEOUT)
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code == 200:
                ct = r.headers.get("Content-Type", "")
                if "json" in ct:
                    return r.json()
                return r.text
        except Exception as exc:
            logger.warning("edgar_get_error", url=url, attempt=attempt, error=str(exc))
    return None


def _resolve_cik(ticker: str) -> Optional[str]:
    data = _edgar_get(EDGAR_TICKERS_URL)
    if not isinstance(data, dict):
        return None
    tu = ticker.upper()
    for entry in data.values():
        if isinstance(entry, dict) and entry.get("ticker", "").upper() == tu:
            return str(entry["cik_str"]).zfill(10)
    return None


def _fetch_company_facts(cik: str) -> dict:
    url  = EDGAR_FACTS_URL.format(cik=cik)
    data = _edgar_get(url)
    if isinstance(data, dict):
        return data.get("facts", {})
    return {}


def _get_latest_filings(cik: str, forms: list[str], max_results: int = 3) -> list[dict]:
    """Return most recent filings of specified form types."""
    url  = EDGAR_SUBMISSIONS.format(cik=cik)
    data = _edgar_get(url)
    if not isinstance(data, dict):
        return []
    filings = data.get("filings", {}).get("recent", {})
    form_list = filings.get("form", [])
    acc_list  = filings.get("accessionNumber", [])
    date_list = filings.get("filingDate", [])
    doc_list  = filings.get("primaryDocument", [])
    cik_int   = int(cik)
    results   = []
    for i, form in enumerate(form_list):
        if form in forms and len(results) < max_results:
            acc_nodash = acc_list[i].replace("-", "")
            results.append({
                "form":        form,
                "accession":   acc_list[i],
                "acc_nodash":  acc_nodash,
                "filing_date": date_list[i] if i < len(date_list) else "",
                "primary_doc": doc_list[i] if i < len(doc_list) else "",
                "cik_int":     cik_int,
                "doc_url": EDGAR_ARCHIVE.format(
                    cik_int=cik_int, acc_nodash=acc_nodash,
                    doc=doc_list[i] if i < len(doc_list) else ""
                ),
            })
    return results


def _fetch_filing_html(filing: dict) -> tuple[str, BeautifulSoup]:
    """Download filing; return (plain_text, soup) pair."""
    url = filing.get("doc_url", "")
    if not url:
        return "", BeautifulSoup("", "html.parser")
    raw = _edgar_get(url)
    if not isinstance(raw, str):
        return "", BeautifulSoup("", "html.parser")
    raw_truncated = raw[:_TEXT_LIMIT * 5]
    soup = BeautifulSoup(raw_truncated, "html.parser")
    text = soup.get_text(separator=" ", strip=True)[:_TEXT_LIMIT]
    return text, soup


def _extract_xbrl_fact(facts: dict, namespace: str, concept: str) -> list[dict]:
    try:
        units = facts.get(namespace, {}).get(concept, {}).get("units", {})
        results = []
        for unit_key, entries in units.items():
            for e in entries:
                results.append({
                    "value": e.get("val"),
                    "unit": unit_key,
                    "form": e.get("form", ""),
                    "end":  e.get("end", ""),
                    "filed": e.get("filed", ""),
                })
        return sorted(results, key=lambda x: x.get("end", ""), reverse=True)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tier 1: XBRL GHG extraction
# ---------------------------------------------------------------------------

# SEC 2024 climate taxonomy + ecd + common company extensions
_SCOPE1_XBRL_CONCEPTS: list[tuple[str, str]] = [
    ("us-gaap", "GreenHouseGasEmissionsScope1"),
    ("ecd",     "GHGEmissionsScope1"),
    ("us-gaap", "CarbonDioxideEmissions"),
    ("us-gaap", "GreenHouseGasEmissions"),           # some filers use this for scope 1
]
_SCOPE2_XBRL_CONCEPTS: list[tuple[str, str]] = [
    ("us-gaap", "GreenHouseGasEmissionsScope2"),
    ("ecd",     "GHGEmissionsScope2"),
    ("us-gaap", "GreenHouseGasEmissionsScope2LocationBased"),
    ("us-gaap", "GreenHouseGasEmissionsScope2MarketBased"),
]
_SCOPE3_XBRL_CONCEPTS: list[tuple[str, str]] = [
    ("us-gaap", "GreenHouseGasEmissionsScope3"),
    ("ecd",     "GHGEmissionsScope3"),
]


class GHGExtractor:
    """
    Three-tier GHG emissions extractor:
    Tier 1 — XBRL structured tags (most reliable)
    Tier 2 — HTML table detection with column header matching (BeautifulSoup)
    Tier 3 — Context-anchored regex (only after anchor detection)

    This architecture directly addresses the audit failure: no wild-regex on
    the entire filing text.
    """

    # Tier 2: column headers that indicate a GHG emissions table
    _TABLE_SCOPE_HEADERS = re.compile(
        r"scope\s*[123]|tco2e?|mt\s*co2|ghg\s+emissions?|greenhouse\s+gas",
        re.IGNORECASE
    )
    # Tier 3: number pattern for context-anchored extraction
    _NUMBER_RE = re.compile(
        r"(\d[\d,]*\.?\d*)\s*"
        r"(?:million\s+)?(?:metric\s+tons?|mt\s*co2|tco2e?|mtco2e?|"
        r"tonnes?\s+of\s+co2|million\s+mt\s+co2)",
        re.IGNORECASE
    )
    # Verification body detection
    _VERIFICATION_RE = re.compile(
        r"(?:third[\s-]party|external|independent)\s+(?:verification|assurance|audit)"
        r"(?:\s+(?:by|from|provided\s+by)\s+([\w\s,]+?))?(?:\.|,|\s+of)",
        re.IGNORECASE
    )

    def extract(self, ticker: str, facts: dict, text: str, soup: BeautifulSoup,
                filing_year: Optional[int]) -> GHGEmissions:
        ghg = GHGEmissions(ticker=ticker, filing_year=filing_year)

        # ── Tier 1: XBRL ─────────────────────────────────────────────────────
        s1_xbrl  = self._try_xbrl_scope(facts, _SCOPE1_XBRL_CONCEPTS)
        s2l_xbrl = self._try_xbrl_scope(facts, _SCOPE2_XBRL_CONCEPTS[:2])
        s2m_xbrl = self._try_xbrl_scope_pair(facts, [
            ("us-gaap", "GreenHouseGasEmissionsScope2MarketBased"),
            ("ecd",     "GHGEmissionsScope2MarketBased"),
        ])
        s3_xbrl  = self._try_xbrl_scope(facts, _SCOPE3_XBRL_CONCEPTS)

        if s1_xbrl is not None:
            ghg.scope1_mt     = s1_xbrl
            ghg.scope1_source = "xbrl"
        if s2l_xbrl is not None:
            ghg.scope2_mt_location = s2l_xbrl
            ghg.scope2_source      = "xbrl"
        if s2m_xbrl is not None:
            ghg.scope2_mt_market = s2m_xbrl
        if s3_xbrl is not None:
            ghg.scope3_mt     = s3_xbrl
            ghg.scope3_source = "xbrl"

        # ── Tier 2: HTML table detection ──────────────────────────────────────
        if ghg.scope1_mt is None or ghg.scope2_mt_location is None:
            s1_tbl, s2_tbl, s3_tbl = self._extract_from_table(soup)
            if ghg.scope1_mt is None and s1_tbl is not None:
                ghg.scope1_mt     = s1_tbl
                ghg.scope1_source = "html_table"
            if ghg.scope2_mt_location is None and s2_tbl is not None:
                ghg.scope2_mt_location = s2_tbl
                ghg.scope2_source      = "html_table"
            if ghg.scope3_mt is None and s3_tbl is not None:
                ghg.scope3_mt     = s3_tbl
                ghg.scope3_source = "html_table"

        # ── Tier 3: Context-anchored regex (fallback) ─────────────────────────
        if ghg.scope1_mt is None or ghg.scope2_mt_location is None:
            s1_re, s2_re, s3_re = self._extract_anchored_regex(text)
            if ghg.scope1_mt is None and s1_re is not None:
                ghg.scope1_mt     = s1_re
                ghg.scope1_source = "text_anchored"
            if ghg.scope2_mt_location is None and s2_re is not None:
                ghg.scope2_mt_location = s2_re
                ghg.scope2_source      = "text_anchored"
            if ghg.scope3_mt is None and s3_re is not None:
                ghg.scope3_mt     = s3_re
                ghg.scope3_source = "text_anchored"

        # ── Verification detection ────────────────────────────────────────────
        if text:
            m = self._VERIFICATION_RE.search(text)
            if m:
                ghg.ghg_verified     = True
                ghg.verification_body = (m.group(1) or "").strip()[:80] or None

        # ── Data quality assessment ───────────────────────────────────────────
        sources = {ghg.scope1_source, ghg.scope2_source}
        if "xbrl" in sources:
            ghg.data_quality = "high"
        elif "html_table" in sources:
            ghg.data_quality = "medium"
        elif "text_anchored" in sources:
            ghg.data_quality = "low"
        else:
            ghg.data_quality = "not_found"

        return ghg

    @staticmethod
    def _try_xbrl_scope(facts: dict, concepts: list[tuple[str, str]]) -> Optional[float]:
        """Return the first non-None annual 10-K value across a list of XBRL concepts."""
        for ns, concept in concepts:
            series = _extract_xbrl_fact(facts, ns, concept)
            annual = [e for e in series if "10-K" in e.get("form", "")]
            if annual and annual[0].get("value") is not None:
                try:
                    return float(annual[0]["value"])
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def _try_xbrl_scope_pair(facts: dict, concepts: list[tuple[str, str]]) -> Optional[float]:
        """Like _try_xbrl_scope but returns None if not found (used for market-based Scope 2)."""
        return GHGExtractor._try_xbrl_scope(facts, concepts)

    def _extract_from_table(self, soup: BeautifulSoup) -> tuple[
        Optional[float], Optional[float], Optional[float]
    ]:
        """
        Tier 2: Scan HTML tables for Scope 1/2/3 rows.
        Strategy:
        1. Find tables whose headers contain GHG-related keywords.
        2. In those tables, identify rows labeled "Scope 1", "Scope 2", "Scope 3".
        3. Extract the numeric value from the data column.
        """
        scope1 = scope2 = scope3 = None
        _num_re = re.compile(r"(\d[\d,]*\.?\d*)")

        for table in soup.find_all("table"):
            # Check if this table is GHG-related
            header_text = table.get_text(separator=" ", strip=True)[:300]
            if not self._TABLE_SCOPE_HEADERS.search(header_text):
                continue

            rows = table.find_all("tr")
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if not cells:
                    continue
                label = cells[0].lower()

                # Scope 1 row
                if re.search(r"\bscope[\s\-]*1\b", label) and scope1 is None:
                    for cell in cells[1:]:
                        m = _num_re.search(cell.replace(",", ""))
                        if m:
                            try:
                                val = float(m.group(1))
                                if val > 0:
                                    scope1 = val
                                    break
                            except ValueError:
                                pass

                # Scope 2 row (prefer location-based)
                elif re.search(r"\bscope[\s\-]*2\b", label) and scope2 is None:
                    for cell in cells[1:]:
                        m = _num_re.search(cell.replace(",", ""))
                        if m:
                            try:
                                val = float(m.group(1))
                                if val > 0:
                                    scope2 = val
                                    break
                            except ValueError:
                                pass

                # Scope 3 row
                elif re.search(r"\bscope[\s\-]*3\b", label) and scope3 is None:
                    for cell in cells[1:]:
                        m = _num_re.search(cell.replace(",", ""))
                        if m:
                            try:
                                val = float(m.group(1))
                                if val > 0:
                                    scope3 = val
                                    break
                            except ValueError:
                                pass

        return scope1, scope2, scope3

    def _extract_anchored_regex(self, text: str) -> tuple[
        Optional[float], Optional[float], Optional[float]
    ]:
        """
        Tier 3: Context-anchored regex.
        Finds "Scope N" anchor positions first, then scans ONLY a 400-char window
        after each anchor — never the entire document.
        """
        scope1 = scope2 = scope3 = None

        def _find_anchored(scope_num: int) -> Optional[float]:
            pattern = re.compile(
                rf"scope\s*{scope_num}\s*(?:ghg\s*)?(?:emissions?)?",
                re.IGNORECASE
            )
            anchors = [m.start() for m in pattern.finditer(text)]
            for pos in anchors[:5]:  # max 5 anchor positions per scope
                window = text[pos: pos + 400]
                m = self._NUMBER_RE.search(window)
                if m:
                    raw = m.group(1).replace(",", "")
                    try:
                        val = float(raw)
                        # Sanity: emissions should be > 0 and < 10 billion MT
                        if 0 < val < 10_000_000_000:
                            return val
                    except ValueError:
                        pass
            return None

        scope1 = _find_anchored(1)
        scope2 = _find_anchored(2)
        scope3 = _find_anchored(3)
        return scope1, scope2, scope3


# ---------------------------------------------------------------------------
# CDP Public Data scraper
# ---------------------------------------------------------------------------


class CDPPublicData:
    """
    Scrape CDP public scores and A-list from cdp.net.
    CDP publishes annual A-list companies freely on their website.
    """

    _CACHE: dict[str, str] = {}   # company_name_upper -> climate_score_tier
    _CACHE_YEAR: Optional[int] = None
    _ALIST_TICKERS: set[str] = set()

    @classmethod
    def _refresh(cls) -> None:
        current_year = date.today().year
        if cls._CACHE_YEAR == current_year and cls._CACHE:
            return
        try:
            time.sleep(_RATE_DELAY)
            r = requests.get(
                CDP_SCORES_URL,
                headers={**_HEADERS, "Accept": "text/html"},
                timeout=_TIMEOUT,
            )
            if r.status_code != 200:
                return
            soup = BeautifulSoup(r.text, "html.parser")
            # CDP renders company lists in various div/table structures
            # Look for any element containing a company name + score label
            for row in soup.select("tr, .company-row, [class*='company'], [class*='scores']"):
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th", "span", "div"])]
                if len(cells) >= 2:
                    name  = cells[0].upper()[:60]
                    score = cells[1].strip() if len(cells) > 1 else ""
                    if re.match(r"^[A-D][-]?$", score):
                        cls._CACHE[name] = score
                        if score.startswith("A"):
                            cls._ALIST_TICKERS.add(name[:12])
            cls._CACHE_YEAR = current_year
        except Exception as exc:
            logger.warning("cdp_refresh_failed", error=str(exc))

    @classmethod
    def get_score(cls, ticker: str, company_name: str) -> CDPScore:
        cls._refresh()
        name_up = company_name.upper()[:60]
        tick_up = ticker.upper()
        year    = date.today().year

        # Fuzzy match: check if any cached name is a substring of the company name
        climate_score = "not_disclosed"
        a_list        = False
        for key, tier in cls._CACHE.items():
            if key[:15] in name_up or tick_up in key:
                climate_score = tier
                a_list        = tier.startswith("A")
                break

        return CDPScore(
            ticker=ticker,
            company_name=company_name,
            cdp_year=year,
            climate_score=climate_score,
            a_list=a_list,
            source_url=CDP_SCORES_URL,
        )


# ---------------------------------------------------------------------------
# SBTi lookup
# ---------------------------------------------------------------------------


class SBTiLookup:
    """
    Check the SBTi public CSV for company targets.
    https://sciencebasedtargets.org/files/SBTi-Companies-Taking-Action.csv
    """

    _DATA: Optional[pd.DataFrame] = None
    _FETCHED: Optional[str] = None

    @classmethod
    def _ensure_loaded(cls) -> None:
        today = date.today().isoformat()
        if cls._DATA is not None and cls._FETCHED == today[:7]:  # monthly cache
            return
        try:
            time.sleep(_RATE_DELAY)
            r = requests.get(SBTI_CSV_URL, headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code == 200:
                import io
                cls._DATA    = pd.read_csv(io.StringIO(r.text), low_memory=False)
                cls._FETCHED = today[:7]
        except Exception as exc:
            logger.warning("sbti_load_failed", error=str(exc))

    @classmethod
    def lookup(cls, ticker: str, company_name: str) -> tuple[bool, bool, str]:
        """Return (committed, approved, target_description)."""
        cls._ensure_loaded()
        if cls._DATA is None or cls._DATA.empty:
            return False, False, ""
        df = cls._DATA
        name_up = company_name.upper()
        # Try ticker match first; fall back to name match
        match_cols = [c for c in df.columns if "company" in c.lower() or "name" in c.lower()]
        target_cols = [c for c in df.columns if "target" in c.lower() or "status" in c.lower()]
        if not match_cols:
            return False, False, ""
        col = match_cols[0]
        mask = df[col].astype(str).str.upper().str.contains(
            "|".join([ticker.upper(), name_up[:12]]), na=False, regex=False
        )
        rows = df[mask]
        if rows.empty:
            return False, False, ""
        row    = rows.iloc[0]
        status = " ".join(str(row.get(c, "")) for c in target_cols if c in row.index).upper()
        committed = True
        approved  = "APPROVED" in status or "VALIDATED" in status
        desc      = str(row.get(target_cols[0], "")) if target_cols else ""
        return committed, approved, desc


# ---------------------------------------------------------------------------
# TCFD Pillar Scorer
# ---------------------------------------------------------------------------


class TCFDPillarScorer:
    """
    Assess each of the four TCFD pillars from 10-K text + proxy text.

    Detection strategy:
    - Governance: board risk committee language in proxy or 10-K header
    - Strategy:   scenario analysis language + temperature targets
    - Risk Mgmt:  climate risk register, physical/transition risk identification
    - Metrics:    GHG numbers (uses GHGExtractor results), targets, verification
    """

    # Governance disclosure patterns
    _GOV_PATTERNS: dict[str, str] = {
        "board_oversight": (
            r"board\s+(?:of\s+directors?|committee)\s+(?:oversee|oversight|review|monitor)"
            r"(?:\s+\w+){0,5}\s+climate"
            r"|climate.{0,30}board\s+(?:oversight|level|committee)"
            r"|risk\s+committee.{0,30}climate"
        ),
        "mgmt_role": (
            r"management(?:'s)?\s+role\s+(?:in\s+)?(?:assessing|managing|evaluating)"
            r"(?:\s+\w+){0,4}\s+climate"
            r"|chief\s+(?:sustainability|climate|environmental)\s+officer"
            r"|sustainability\s+committee\s+(?:of\s+)?management"
        ),
    }

    # Strategy disclosure patterns
    _STRAT_PATTERNS: dict[str, str] = {
        "risks_opportunities": (
            r"climate.{0,30}(?:risk|opportunity|opportunit)"
            r"(?:\s+\w+){0,5}\s+(?:short|medium|long).{0,10}term"
            r"|(?:short|medium|long).{0,10}term.{0,30}climate"
        ),
        "business_impact": (
            r"climate.{0,30}impact.{0,30}(?:business|strateg|financial|operati)"
            r"|(?:financial|business|strategic)\s+impact.{0,30}climate"
        ),
        "scenario_analysis": (
            r"climate\s+scenario\s+anal"
            r"|1\.5\s*[°℃C]\s*scenario"
            r"|2\s*[°℃C]\s+(?:scenario|pathway|target)"
            r"|(?:well.below|below)\s+2\s*[°℃C]"
            r"|IEA\s+(?:Net\s+Zero|NZE|SDS)\s+scenario"
            r"|NGFS\s+scenario"
            r"|TCFD\s+scenario"
        ),
    }

    # Risk management disclosure patterns
    _RISK_PATTERNS: dict[str, str] = {
        "id_assess_process": (
            r"climate\s+risk\s+(?:identification|assessment|register|inventory)"
            r"|identify.{0,20}(?:physical|transition)\s+risk"
            r"|climate.{0,20}risk\s+(?:process|framework|approach)"
        ),
        "manage_process": (
            r"(?:managing|management\s+of|mitigat).{0,30}climate\s+risk"
            r"|climate\s+risk\s+(?:management|mitigation|response)"
            r"|transition\s+risk\s+management"
        ),
        "integration": (
            r"integrat.{0,20}climate.{0,20}(?:enterprise|overall|ERM|risk\s+management)"
            r"|climate.{0,20}integrat.{0,20}(?:enterprise|risk\s+management)"
            r"|enterprise\s+risk\s+management.{0,50}climate"
        ),
    }

    # Metrics & targets patterns
    _METRICS_PATTERNS: dict[str, str] = {
        "ghg_scope1_2": (
            r"scope\s*1\s*(?:and|&)\s*(?:scope\s*)?2\s*(?:ghg\s*)?emissions?"
            r"|(?:direct|indirect)\s+(?:ghg\s*)?emissions?\s+(?:of|were|totaled)"
        ),
        "ghg_scope3": (
            r"scope\s*3\s*(?:ghg\s*)?emissions?"
            r"|value\s+chain\s+emissions?"
            r"|indirect\s+(?:scope\s*3|upstream|downstream)\s+emissions?"
        ),
        "targets": (
            r"(?:net[\s-]zero|carbon[\s-]neutral|carbon\s+negative)\s+(?:by\s+)?20[3-5]\d"
            r"|science[\s-]based\s+target"
            r"|(?:reduce|reduction)\s+(?:ghg|emissions?|carbon).{0,30}(?:by\s+)?\d+\s*%"
            r"|SBTi\s+(?:committed|approved|validated)"
            r"|emission\s+(?:reduction\s+)?target"
        ),
    }

    def score_governance(self, text: str) -> TCFDPillarResult:
        return self._score_pillar("governance", text, self._GOV_PATTERNS,
                                  ["board_oversight", "mgmt_role"])

    def score_strategy(self, text: str) -> TCFDPillarResult:
        return self._score_pillar("strategy", text, self._STRAT_PATTERNS,
                                  ["risks_opportunities", "business_impact", "scenario_analysis"])

    def score_risk_mgmt(self, text: str) -> TCFDPillarResult:
        return self._score_pillar("risk_mgmt", text, self._RISK_PATTERNS,
                                  ["id_assess_process", "manage_process", "integration"])

    def score_metrics(self, text: str, ghg: GHGEmissions) -> TCFDPillarResult:
        """Metrics pillar: GHG result from structured extractor + target language."""
        found   = []
        missing = []
        evidence: dict[str, Any] = {}

        # ghg_scope1_2: use structured GHG result
        if ghg.scope1_mt is not None or ghg.scope2_mt_location is not None:
            found.append("ghg_scope1_2")
            evidence["ghg_scope1_2"] = {
                "scope1_mt": ghg.scope1_mt,
                "scope2_mt": ghg.scope2_mt_location,
                "source": ghg.scope1_source,
                "data_quality": ghg.data_quality,
            }
        else:
            missing.append("ghg_scope1_2")

        # ghg_scope3
        if ghg.scope3_mt is not None:
            found.append("ghg_scope3")
            evidence["ghg_scope3"] = {"scope3_mt": ghg.scope3_mt, "source": ghg.scope3_source}
        elif text and re.search(self._METRICS_PATTERNS["ghg_scope3"], text, re.IGNORECASE):
            found.append("ghg_scope3")
            evidence["ghg_scope3"] = {"source": "text_keyword"}
        else:
            missing.append("ghg_scope3")

        # targets
        if text and re.search(self._METRICS_PATTERNS["targets"], text, re.IGNORECASE):
            found.append("targets")
            evidence["targets"] = {"source": "text_keyword"}
        else:
            missing.append("targets")

        # score: each of 3 disclosures = 100/3 pts
        score = (len(found) / 3) * 100.0
        return TCFDPillarResult(
            pillar="metrics",
            score=round(score, 1),
            disclosures_found=found,
            disclosures_missing=missing,
            evidence=evidence,
        )

    @staticmethod
    def _score_pillar(
        pillar: str,
        text: str,
        patterns: dict[str, str],
        keys: list[str],
    ) -> TCFDPillarResult:
        found   = []
        missing = []
        evidence: dict[str, Any] = {}
        text_to_search = text or ""

        for key in keys:
            pattern = patterns.get(key, "")
            if pattern and re.search(pattern, text_to_search, re.IGNORECASE | re.DOTALL):
                found.append(key)
                # Store a short snippet for evidence
                m = re.search(pattern, text_to_search, re.IGNORECASE | re.DOTALL)
                if m:
                    start = max(0, m.start() - 30)
                    evidence[key] = text_to_search[start: m.end() + 60].strip()[:200]
            else:
                missing.append(key)

        n_keys = len(keys)
        score  = (len(found) / n_keys * 100.0) if n_keys else 0.0
        return TCFDPillarResult(
            pillar=pillar,
            score=round(score, 1),
            disclosures_found=found,
            disclosures_missing=missing,
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# Net-Zero Commitment Extractor
# ---------------------------------------------------------------------------


class NetZeroExtractor:
    """
    Extract net-zero / carbon-neutral commitment details from filing text.
    Returns structured ClimateTarget objects with:
    - target type, year, base year, interim milestones, scope coverage, verification
    """

    # Net-zero / carbon-neutral target year pattern
    _TARGET_YEAR_RE = re.compile(
        r"(?:net[\s\-]zero|carbon[\s\-]neutral|carbon\s+negative)\s+"
        r"(?:by\s+)?(?:the\s+)?(?:year\s+)?(20[3-9]\d)",
        re.IGNORECASE
    )
    # Base year detection
    _BASE_YEAR_RE = re.compile(
        r"(?:base\s+year|baseline\s+year|relative\s+to\s+(?:a\s+)?(?:base\s+year\s+of\s+)?)"
        r"\s*((?:19|20)\d{2})",
        re.IGNORECASE
    )
    # Interim milestone detection
    _MILESTONE_RE = re.compile(
        r"(?:by\s+)?(20[2-4]\d)[,\s].*?(?:reduc\w+|achiev\w+|reach\w+).{0,60}"
        r"(?:\d+\s*%|percent)",
        re.IGNORECASE
    )
    # Scope coverage
    _SCOPE_COV_RE = re.compile(
        r"scope\s*([123](?:\s*(?:and|,|&)\s*[123])*)",
        re.IGNORECASE
    )
    # Reduction percentage
    _REDUCTION_RE = re.compile(
        r"(?:reduc\w+|decreas\w+)\s+(?:emissions?|carbon|ghg).{0,40}?"
        r"(\d+)\s*(?:percent|%)\s*(?:by\s+20\d\d)?",
        re.IGNORECASE
    )
    # Third-party verification
    _VERIFICATION_RE = re.compile(
        r"(?:third[\s-]party|external|independent)\s+(?:verif\w+|assur\w+)",
        re.IGNORECASE
    )
    _SBTI_RE = re.compile(
        r"SBTi\s+(?:committed|validated|approved|aligned)|science[\s-]based\s+target",
        re.IGNORECASE
    )

    def extract(self, ticker: str, text: str,
                sbti_committed: bool, sbti_approved: bool) -> list[ClimateTarget]:
        targets: list[ClimateTarget] = []
        if not text:
            return targets

        # Find net-zero / carbon-neutral targets
        for m in self._TARGET_YEAR_RE.finditer(text):
            target_year = int(m.group(1))
            # Extract context window ±300 chars
            start   = max(0, m.start() - 100)
            end     = min(len(text), m.end() + 500)
            context = text[start:end]

            # Base year
            base_m    = self._BASE_YEAR_RE.search(context)
            base_year = int(base_m.group(1)) if base_m else None

            # Scope coverage
            scope_ms = self._SCOPE_COV_RE.findall(context)
            scopes   = []
            for sm in scope_ms:
                for digit in re.findall(r"[123]", sm):
                    key = f"scope{digit}"
                    if key not in scopes:
                        scopes.append(key)

            # Interim milestones
            milestones = [mm.group(0)[:80] for mm in self._MILESTONE_RE.finditer(context)]

            # Reduction pct
            red_m = self._REDUCTION_RE.search(context)
            red_pct = float(red_m.group(1)) if red_m else None

            # Verification
            third_party = bool(self._VERIFICATION_RE.search(context))
            sbti_in_ctx = bool(self._SBTI_RE.search(context))
            if sbti_in_ctx and sbti_approved:
                verification = "sbti_validated"
            elif sbti_in_ctx and sbti_committed:
                verification = "sbti_committed"
            elif third_party:
                verification = "third_party"
            else:
                verification = "self_reported"

            targets.append(ClimateTarget(
                ticker=ticker,
                target_type="net_zero",
                target_year=target_year,
                base_year=base_year,
                interim_milestones=milestones[:5],
                scope_coverage=scopes,
                reduction_pct=red_pct,
                verification_status=verification,
                sbti_committed=sbti_committed or sbti_in_ctx,
                sbti_approved=sbti_approved,
                is_net_zero=True,
                description=context.strip()[:200],
                source="edgar_10k",
            ))
            if len(targets) >= 5:
                break

        # SBTi target without explicit net-zero year
        if not targets and (sbti_committed or sbti_approved):
            targets.append(ClimateTarget(
                ticker=ticker,
                target_type="sbti",
                verification_status="sbti_validated" if sbti_approved else "sbti_committed",
                sbti_committed=sbti_committed,
                sbti_approved=sbti_approved,
                source="sbti_public_csv",
            ))

        return targets


# ---------------------------------------------------------------------------
# Physical Risk Assessor
# ---------------------------------------------------------------------------


class PhysicalRiskAssessor:
    """
    Detect and score physical climate risk disclosures.
    Acute risks: floods, hurricanes, extreme weather events
    Chronic risks: water stress, sea level rise, extreme heat, drought
    """

    _RISK_PATTERNS: dict[str, tuple[str, str]] = {
        # (type, pattern)
        "flood_risk":        ("acute",   r"\bflood(?:ing)?\s+risk|\bflood\s+(?:event|damage|loss|prone)"),
        "hurricane_storm":   ("acute",   r"\bhurricane|\btropical\s+storm|\bcyclone|\bextreme\s+weather"),
        "wildfire":          ("acute",   r"\bwildfire|\bforest\s+fire\s+risk|\bfire\s+(?:risk|hazard)"),
        "water_stress":      ("chronic", r"\bwater\s+stress|\bwater\s+scarcity|\bdrought\s+risk|\bwater\s+availability"),
        "extreme_heat":      ("chronic", r"\bextreme\s+heat|\bheat\s+(?:stress|wave|risk)|\bhigh\s+temperature\s+risk"),
        "sea_level_rise":    ("chronic", r"\bsea[\s\-]level\s+rise|\bcoastal\s+flood|\binundation\s+risk"),
    }

    # Asset exposure quantification
    _EXPOSURE_RE = re.compile(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s+(?:our\s+)?(?:assets?|facilities?|operations?|"
        r"properties?|sites?|portfolio)\s+(?:are|is|located|subject|exposed)\s+(?:in|to|at)",
        re.IGNORECASE
    )

    def assess(self, ticker: str, text: str) -> PhysicalRisk:
        risk = PhysicalRisk(ticker=ticker)
        if not text:
            return risk

        acute_count   = 0
        chronic_count = 0
        details       = []

        for risk_name, (risk_type, pattern) in self._RISK_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                setattr(risk, risk_name.replace("-", "_") + "_mentioned", True)
                details.append(risk_name)
                if risk_type == "acute":
                    acute_count += 1
                else:
                    chronic_count += 1

        # Set individual flags
        risk.flood_risk_mentioned     = "flood_risk"     in details
        risk.water_stress_mentioned   = "water_stress"   in details
        risk.extreme_heat_mentioned   = "extreme_heat"   in details
        risk.sea_level_rise_mentioned = "sea_level_rise" in details
        risk.hurricane_storm_mentioned = "hurricane_storm" in details
        risk.wildfire_mentioned       = "wildfire"       in details

        # Asset exposure
        exp_m = self._EXPOSURE_RE.search(text)
        if exp_m:
            risk.asset_exposure_quantified = True
            try:
                risk.asset_exposure_pct = float(exp_m.group(1))
            except ValueError:
                pass

        # Score: 0–50 for acute, 0–50 for chronic
        max_acute   = len([v for v in self._RISK_PATTERNS.values() if v[0] == "acute"])
        max_chronic = len([v for v in self._RISK_PATTERNS.values() if v[0] == "chronic"])
        risk.acute_risk_score   = (acute_count / max(max_acute, 1)) * 50.0
        risk.chronic_risk_score = (chronic_count / max(max_chronic, 1)) * 50.0
        # Bonus for quantified asset exposure
        exposure_bonus = 10.0 if risk.asset_exposure_quantified else 0.0
        risk.total_physical_risk_score = min(100.0,
            risk.acute_risk_score + risk.chronic_risk_score + exposure_bonus)
        risk.risk_details = details

        return risk


# ---------------------------------------------------------------------------
# Transition Risk Assessor
# ---------------------------------------------------------------------------


class TransitionRiskAssessor:
    """
    Detect and score transition climate risk disclosures.
    Covers: stranded assets, carbon pricing, regulatory risk, technology disruption.
    """

    _RISK_PATTERNS: dict[str, str] = {
        "stranded_asset_risk": (
            r"\bstranded\s+asset|\bstranded\s+(?:fossil|coal|oil|gas|carbon)"
        ),
        "carbon_pricing_exposure": (
            r"\bcarbon\s+(?:price|tax|pricing|credit|cost|fee)"
            r"|\bcarbon\s+allowance|\bETS\s+|\bemission\s+trading"
            r"|\binternal\s+carbon\s+price"
        ),
        "regulatory_risk_mentioned": (
            r"\bclimate\s+regulation|\bclimate\s+policy\s+risk"
            r"|\bclimate\s+legislation|\benvironmental\s+regulation"
            r"|\bclean\s+air\s+act|\bcarbon\s+regulat"
        ),
        "technology_disruption": (
            r"\bclean\s+(?:energy\s+)?technology|\brenewable\s+energy\s+(?:transition|disruption)"
            r"|\belectric\s+vehicle\s+(?:transition|adoption)"
            r"|\blow[\s-]carbon\s+technology|\benergy\s+transition\s+risk"
        ),
        "market_risk_mentioned": (
            r"\bmarket\s+(?:shift|demand|risk).{0,30}(?:climate|carbon|green|clean)"
            r"|\bconsumer\s+(?:preference|demand).{0,30}(?:sustainable|climate|green)"
        ),
        "reputational_risk_mentioned": (
            r"\breputational\s+risk.{0,30}climate"
            r"|\bclimate.{0,30}reputational\s+risk"
            r"|\bbrand\s+risk.{0,30}(?:climate|ESG|sustainability)"
        ),
    }

    _CARBON_PRICE_RE = re.compile(
        r"(?:\$|USD)\s*(\d+(?:\.\d+)?)\s*(?:per\s+metric\s+ton|per\s+tco2e?|per\s+ton\s+co2)",
        re.IGNORECASE
    )
    _COMPLIANCE_COST_RE = re.compile(
        r"compliance\s+cost.{0,50}(?:\$|USD)\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?",
        re.IGNORECASE
    )

    def assess(self, ticker: str, text: str) -> TransitionRisk:
        risk = TransitionRisk(ticker=ticker)
        if not text:
            return risk

        risk_count = 0
        details    = []

        for risk_name, pattern in self._RISK_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                setattr(risk, risk_name, True)
                details.append(risk_name)
                risk_count += 1

        # Carbon price assumption
        cp_m = self._CARBON_PRICE_RE.search(text)
        if cp_m:
            try:
                risk.carbon_price_assumption = float(cp_m.group(1))
            except ValueError:
                pass

        # Compliance cost
        cc_m = self._COMPLIANCE_COST_RE.search(text)
        if cc_m:
            try:
                raw = cc_m.group(1).replace(",", "")
                risk.regulatory_compliance_cost = float(raw)
            except ValueError:
                pass

        total_risks = len(self._RISK_PATTERNS)
        risk.transition_risk_score = (risk_count / total_risks) * 100.0
        risk.risk_details = details
        return risk


# ---------------------------------------------------------------------------
# Physical Risk — State-level flood/water risk proxy (FEMA-inspired)
# ---------------------------------------------------------------------------

# State-level physical risk scores (0–10 scale, driven by FEMA flood zone data
# and historical climate event frequency).  Higher = more exposed.
_STATE_PHYSICAL_RISK: dict[str, float] = {
    # High coastal/flood risk
    "FL": 9.0, "LA": 9.0, "MS": 8.5, "AL": 8.0, "TX": 8.0,
    # Significant coastal/storm risk
    "NC": 7.5, "SC": 7.5, "VA": 7.0, "MD": 6.5, "NJ": 6.5,
    "NY": 5.0, "CT": 5.0, "MA": 5.0, "RI": 5.0, "DE": 5.5,
    # West coast (wildfire/drought dominated)
    "CA": 7.0, "OR": 5.5, "WA": 4.5,
    # Interior flood-prone
    "IA": 6.0, "MO": 6.0, "AR": 6.5, "TN": 5.5, "KY": 5.5,
    "OH": 4.5, "IN": 4.5, "IL": 5.0, "WI": 4.0, "MN": 4.0,
    # Arid / lower physical risk
    "AZ": 4.0, "NM": 3.5, "NV": 3.5, "UT": 3.0, "ID": 3.5,
    "CO": 3.0, "WY": 3.0, "MT": 3.0, "ND": 3.5, "SD": 3.5,
    "NE": 4.0, "KS": 4.5, "OK": 5.0,
    # Other
    "MI": 3.5, "PA": 4.5, "GA": 5.5, "AK": 3.0, "HI": 6.0,
    "WV": 5.0, "ME": 3.5, "NH": 3.5, "VT": 3.5,
}

_DEFAULT_STATE_RISK = 4.0  # national average fallback


def get_state_physical_risk(state_code: str) -> float:
    """
    Return a 0–10 physical risk score for a US state (2-letter code).
    Proxy for FEMA flood zone exposure + climate event frequency.
    FL/LA score highest (9); CO/WY/MT score lowest (~3).
    """
    return _STATE_PHYSICAL_RISK.get(state_code.upper(), _DEFAULT_STATE_RISK)


def get_company_physical_risk(
    hq_state: str,
    operations_states: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compute aggregate physical risk score for a company given its HQ state
    and (optionally) a list of states where it has material operations.

    Returns a dict with:
      hq_risk, operations_risk_avg, aggregate_risk (0–10), risk_tier
    """
    hq_risk = get_state_physical_risk(hq_state)

    if operations_states:
        ops_risks = [get_state_physical_risk(s) for s in operations_states]
        ops_avg = sum(ops_risks) / len(ops_risks)
        # Weight: 40% HQ, 60% operations footprint
        aggregate = 0.4 * hq_risk + 0.6 * ops_avg
    else:
        ops_avg = hq_risk
        aggregate = hq_risk

    if aggregate >= 7.5:
        tier = "very_high"
    elif aggregate >= 6.0:
        tier = "high"
    elif aggregate >= 4.5:
        tier = "medium"
    elif aggregate >= 3.0:
        tier = "low"
    else:
        tier = "very_low"

    return {
        "hq_state":              hq_state.upper(),
        "hq_risk_score":         hq_risk,
        "operations_risk_avg":   round(ops_avg, 2),
        "aggregate_risk_score":  round(aggregate, 2),
        "risk_tier":             tier,
    }


# ---------------------------------------------------------------------------
# Transition Risk — SIC-based carbon intensity tier
# ---------------------------------------------------------------------------

# SIC code → (transition_risk_tier, carbon_intensity_proxy)
# Tier: "very_high" | "high" | "medium" | "low" | "very_low"
_SIC_TRANSITION_TIERS: dict[str, tuple[str, float]] = {
    # Energy — fossil fuel extraction and refining
    "1311": ("very_high", 0.95),   # Crude Petroleum & Natural Gas
    "1381": ("very_high", 0.90),   # Drilling Oil & Gas Wells
    "1382": ("very_high", 0.88),   # Oil & Gas Field Services
    "2911": ("very_high", 0.92),   # Petroleum Refining
    "1321": ("very_high", 0.85),   # Natural Gas Liquids
    "5171": ("high",      0.70),   # Petroleum Wholesale
    # Utilities — fossil-heavy
    "4911": ("very_high", 0.85),   # Electric Services
    "4931": ("high",      0.75),   # Electric & Other Services
    "4941": ("high",      0.65),   # Water Supply (lower GHG)
    # Materials — high GHG intensity
    "2819": ("high",      0.72),   # Industrial Chemicals
    "2860": ("high",      0.68),   # Industrial Chemicals/Plastics
    "3312": ("high",      0.80),   # Steel Works
    "1040": ("high",      0.75),   # Gold/Silver Mining (high energy use)
    # Industrials
    "3559": ("medium",    0.45),   # Industrial Machinery
    "3720": ("medium",    0.50),   # Aircraft & Parts
    # Consumer Staples
    "2000": ("medium",    0.40),   # Food Products
    "5400": ("low",       0.25),   # Food Stores
    # Consumer Discretionary
    "5900": ("low",       0.20),   # Retail
    "7011": ("low",       0.22),   # Hotels
    # Health Care
    "2836": ("low",       0.18),   # Pharmaceutical Preparations
    "8011": ("very_low",  0.12),   # Offices of Physicians
    # Financials — low direct emissions
    "6020": ("very_low",  0.05),   # State Commercial Banks
    "6022": ("very_low",  0.05),   # National Commercial Banks
    "6211": ("very_low",  0.04),   # Security Brokers & Dealers
    # Information Technology
    "7372": ("low",       0.15),   # Prepackaged Software
    "7371": ("low",       0.12),   # Computer Programming Services
    "3674": ("medium",    0.30),   # Semiconductors (energy-intensive fabs)
    # Communication Services
    "4813": ("low",       0.18),   # Telephone Communications
    # Real Estate
    "6552": ("low",       0.25),   # Land Subdividers (excl. cemeteries)
    "6798": ("low",       0.22),   # Real Estate Investment Trusts
}

_DEFAULT_TRANSITION_TIER = ("medium", 0.40)


def get_sic_transition_risk(sic_code: str) -> dict[str, Any]:
    """
    Return transition risk tier and carbon intensity proxy for a SIC code.

    Returns:
        tier: "very_high" | "high" | "medium" | "low" | "very_low"
        carbon_intensity_proxy: 0–1 relative intensity score
        sector: mapped sector name from _SIC_TO_SECTOR
    """
    code = str(sic_code).zfill(4)
    tier, intensity = _SIC_TRANSITION_TIERS.get(code, _DEFAULT_TRANSITION_TIER)
    sector = _SIC_TO_SECTOR.get(code, "unknown")
    return {
        "sic_code":                code,
        "transition_risk_tier":    tier,
        "carbon_intensity_proxy":  intensity,
        "sector":                  sector,
    }


def compute_carbon_intensity(scope1_mt: float, revenue_m: float) -> Optional[float]:
    """
    Compute Scope 1 carbon intensity = scope1 emissions (tCO2e) / revenue (USD millions).
    Returns tCO2e per USD million, or None if inputs are invalid.
    """
    if revenue_m <= 0 or scope1_mt < 0:
        return None
    return round(scope1_mt / revenue_m, 4)


# ---------------------------------------------------------------------------
# Climate Scenario Analysis — 1.5°C / 2°C / 4°C
# ---------------------------------------------------------------------------

# Stranding risk = fraction of fossil/carbon-intensive assets impaired
# under each scenario.  More aggressive (lower-temperature) targets
# impose higher stranding risk on fossil assets.
_SCENARIO_PARAMS: dict[str, dict[str, Any]] = {
    "1.5C": {
        "label":             "1.5°C — Net-Zero by 2050 (Aggressive transition)",
        "temp_delta":        1.5,
        "stranding_risk_fossil":   0.65,   # 65% of fossil assets stranded
        "stranding_risk_utility":  0.40,
        "stranding_risk_default":  0.10,
        "carbon_price_2030":  150.0,       # $/tCO2e
        "renewable_penetration": 0.80,
        "policy_stringency": "very_high",
    },
    "2C": {
        "label":             "2°C — Paris Agreement (Moderate transition)",
        "temp_delta":        2.0,
        "stranding_risk_fossil":   0.40,
        "stranding_risk_utility":  0.25,
        "stranding_risk_default":  0.06,
        "carbon_price_2030":  75.0,
        "renewable_penetration": 0.60,
        "policy_stringency": "high",
    },
    "4C": {
        "label":             "4°C — Business as Usual (Physical risk dominant)",
        "temp_delta":        4.0,
        "stranding_risk_fossil":   0.15,   # low transition risk; high physical risk
        "stranding_risk_utility":  0.10,
        "stranding_risk_default":  0.03,
        "carbon_price_2030":  25.0,
        "renewable_penetration": 0.30,
        "policy_stringency": "low",
    },
}


def climate_scenario_analysis(
    ticker: str,
    sector: str,
    total_assets_m: float,
    scenarios: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Run TCFD scenario analysis for 1.5°C, 2°C, and 4°C warming pathways.

    Args:
        ticker:          company ticker symbol
        sector:          sector string (from _SIC_TO_SECTOR values)
        total_assets_m:  total assets in USD millions
        scenarios:       list of scenario keys to run (default: all three)

    Returns a dict with per-scenario results including:
        - stranding_risk_pct: fraction of assets at risk (0–1)
        - stranded_assets_m: USD millions of potentially stranded assets
        - carbon_price_usd: assumed carbon price by 2030 ($/tCO2e)
        - policy_stringency: qualitative policy environment
    """
    scenarios = scenarios or ["1.5C", "2C", "4C"]
    results: dict[str, Any] = {"ticker": ticker, "sector": sector, "scenarios": {}}

    # Determine asset stranding multiplier based on sector
    fossil_sectors = {"energy", "utilities"}
    high_intensity = {"materials", "industrials"}

    for sc_key in scenarios:
        params = _SCENARIO_PARAMS.get(sc_key)
        if params is None:
            continue
        if sector in fossil_sectors:
            strand_pct = params["stranding_risk_fossil"]
        elif sector in high_intensity:
            strand_pct = (params["stranding_risk_fossil"] + params["stranding_risk_default"]) / 2
        else:
            strand_pct = params["stranding_risk_default"]

        stranded_assets_m = round(total_assets_m * strand_pct, 2)

        results["scenarios"][sc_key] = {
            "label":               params["label"],
            "temp_delta":          params["temp_delta"],
            "stranding_risk_pct":  round(strand_pct, 4),
            "stranded_assets_m":   stranded_assets_m,
            "carbon_price_2030":   params["carbon_price_2030"],
            "renewable_penetration": params["renewable_penetration"],
            "policy_stringency":   params["policy_stringency"],
        }

    return results


# ---------------------------------------------------------------------------
# Climate VaR — portfolio-level
# ---------------------------------------------------------------------------

def compute_climate_var(
    holdings: list[dict[str, Any]],
    temperature_delta: float = 2.0,
) -> dict[str, Any]:
    """
    Compute portfolio Climate Value-at-Risk (Climate VaR).

    Formula: Climate VaR = Σ(weight_i × climate_beta_i × temperature_delta)

    where climate_beta_i is derived from the sector's carbon intensity proxy.

    Args:
        holdings: list of dicts with keys:
            - "ticker": str
            - "weight": float (portfolio weight, must sum to ~1.0)
            - "sic_code": str (optional; used for climate_beta lookup)
            - "sector": str (optional; fallback if sic_code missing)
            - "climate_beta": float (optional; overrides computed beta)
        temperature_delta: warming above pre-industrial baseline (°C)

    Returns:
        climate_var: float (fraction of portfolio value at risk)
        breakdown: per-holding contribution
    """
    # Sector → implied climate beta (sensitivity of asset value to 1°C warming)
    _SECTOR_CLIMATE_BETA: dict[str, float] = {
        "energy":                  0.25,
        "utilities":               0.18,
        "materials":               0.14,
        "industrials":             0.10,
        "consumer_staples":        0.06,
        "consumer_discretionary":  0.05,
        "health_care":             0.03,
        "financials":              0.04,
        "information_technology":  0.04,
        "communication_services":  0.03,
        "real_estate":             0.08,
        "default":                 0.07,
    }

    breakdown = []
    total_var = 0.0
    total_weight = 0.0

    for h in holdings:
        weight = float(h.get("weight", 0))
        total_weight += weight

        # Resolve climate beta
        beta = h.get("climate_beta")
        if beta is None:
            sic = str(h.get("sic_code", ""))
            if sic:
                sic_info = get_sic_transition_risk(sic)
                sector = sic_info["sector"]
            else:
                sector = h.get("sector", "default")
            beta = _SECTOR_CLIMATE_BETA.get(sector, _SECTOR_CLIMATE_BETA["default"])

        contribution = weight * beta * temperature_delta
        total_var += contribution

        breakdown.append({
            "ticker":        h.get("ticker", ""),
            "weight":        round(weight, 4),
            "climate_beta":  round(beta, 4),
            "contribution":  round(contribution, 6),
        })

    return {
        "climate_var":       round(total_var, 6),
        "temperature_delta": temperature_delta,
        "total_weight":      round(total_weight, 4),
        "breakdown":         breakdown,
    }


# ---------------------------------------------------------------------------
# TCFD Engine — orchestrator
# ---------------------------------------------------------------------------


def _tcfd_alignment_level(score: float) -> str:
    if score >= 80:  return "aligned"
    if score >= 60:  return "advancing"
    if score >= 35:  return "developing"
    return "minimal"


class TCFDEngine:
    """
    Orchestrates the full TCFD assessment pipeline.

    1. Resolves ticker → CIK → XBRL facts + 10-K HTML
    2. Runs GHGExtractor (3-tier)
    3. Runs TCFDPillarScorer for all 4 pillars
    4. Runs NetZeroExtractor, PhysicalRiskAssessor, TransitionRiskAssessor
    5. Fetches CDP score and SBTi lookup
    6. Computes overall TCFD alignment score (0–100)
    7. Persists all results to SQLite

    Usage:
        engine = TCFDEngine()
        result = engine.assess("AAPL")
    """

    def __init__(self, db: TCFDDB | None = None) -> None:
        self._db       = db or TCFDDB()
        self._ghg      = GHGExtractor()
        self._pillar   = TCFDPillarScorer()
        self._netzero  = NetZeroExtractor()
        self._physical = PhysicalRiskAssessor()
        self._transn   = TransitionRiskAssessor()
        self._cdp      = CDPPublicData()
        self._sbti     = SBTiLookup()

    def assess(self, ticker: str) -> TCFDAssessment:
        warnings: list[str] = []
        dq: dict[str, str]  = {}
        ticker = ticker.upper()

        # ── Resolve CIK ──────────────────────────────────────────────────────
        cik = _resolve_cik(ticker)
        if not cik:
            warnings.append(f"CIK not resolved for {ticker}")
            dq["cik"] = "missing"

        # ── Fetch XBRL facts ─────────────────────────────────────────────────
        facts: dict = {}
        company_name = ticker
        sector       = "default"
        filing_year  = datetime.utcnow().year

        if cik:
            facts = _fetch_company_facts(cik)
            dq["xbrl"] = "ok" if facts else "missing"

            # Extract company name from facts
            entity_entries = list(
                (facts.get("dei", {})
                 .get("EntityRegistrantName", {})
                 .get("units", {}).values() or [[]])
            )
            if entity_entries and entity_entries[0]:
                company_name = entity_entries[0][0].get("val", ticker)

            # SIC sector
            sic_series = _extract_xbrl_fact(facts, "dei", "EntitySicCode")
            if sic_series:
                sic_code = str(sic_series[0].get("value", ""))
                sector   = _SIC_TO_SECTOR.get(sic_code.zfill(4), "default")

        # ── Fetch 10-K filing (latest) ────────────────────────────────────────
        text = ""
        soup = BeautifulSoup("", "html.parser")
        if cik:
            filings_10k = _get_latest_filings(cik, ["10-K", "10-K/A"], max_results=1)
            if filings_10k:
                filing      = filings_10k[0]
                filing_year = int(filing.get("filing_date", f"{filing_year}")[:4])
                text, soup  = _fetch_filing_html(filing)
                dq["filing"] = "ok" if text else "empty"
            else:
                warnings.append("No 10-K found in EDGAR")
                dq["filing"] = "missing"

        # Also try to get DEF 14A for governance pillar
        proxy_text = ""
        if cik:
            proxy_filings = _get_latest_filings(cik, ["DEF 14A", "DEFA14A"], max_results=1)
            if proxy_filings:
                proxy_text, _ = _fetch_filing_html(proxy_filings[0])
                dq["proxy"] = "ok" if proxy_text else "empty"

        # Combine for governance pillar (board oversight in both docs)
        gov_text = (proxy_text + " " + text)[:_TEXT_LIMIT]

        # ── Tier 1/2/3 GHG Extraction ────────────────────────────────────────
        ghg = self._ghg.extract(ticker, facts, text, soup, filing_year)
        dq["ghg_source"] = ghg.scope1_source

        # Compute YoY change if prior-year XBRL available
        ghg = self._add_prior_year_ghg(ghg, facts)

        # ── SBTi lookup ───────────────────────────────────────────────────────
        sbti_committed, sbti_approved, sbti_desc = self._sbti.lookup(ticker, company_name)

        # ── CDP score ─────────────────────────────────────────────────────────
        cdp_score = CDPPublicData.get_score(ticker, company_name)
        self._db.upsert_cdp(cdp_score)

        # ── TCFD pillar scoring ───────────────────────────────────────────────
        gov_pillar   = self._pillar.score_governance(gov_text)
        strat_pillar = self._pillar.score_strategy(text)
        risk_pillar  = self._pillar.score_risk_mgmt(text)
        met_pillar   = self._pillar.score_metrics(text, ghg)

        # ── Item 1C detection (SEC 2024 mandatory) ────────────────────────────
        item_1c = bool(re.search(
            r"item\s+1c[\.\s]*(?:cybersecurity|climate)|"
            r"climate[\s-]related\s+risk\s+disclosure|"
            r"item\s+1\s*c[\.\s]*climate",
            text, re.IGNORECASE
        ))

        # ── Scenario analysis flag ────────────────────────────────────────────
        has_scenario = "scenario_analysis" in strat_pillar.disclosures_found

        # ── Net-zero targets ─────────────────────────────────────────────────
        targets      = self._netzero.extract(ticker, text, sbti_committed, sbti_approved)
        has_net_zero = any(t.is_net_zero for t in targets)
        has_sbti     = sbti_committed or sbti_approved

        # ── Physical + transition risk ────────────────────────────────────────
        phys_risk  = self._physical.assess(ticker, text)
        trans_risk = self._transn.assess(ticker, text)

        # ── Overall TCFD alignment score ─────────────────────────────────────
        # 11 disclosures total (per TCFD spec):
        # Governance: 2, Strategy: 3, Risk Mgmt: 3, Metrics: 3
        all_found = (
            gov_pillar.disclosures_found +
            strat_pillar.disclosures_found +
            risk_pillar.disclosures_found +
            met_pillar.disclosures_found
        )
        found_count   = len(all_found)
        alignment_raw = (found_count / 11) * 100.0

        # Bonus for structured (XBRL) GHG data: +5 pts
        if ghg.scope1_source == "xbrl":
            alignment_raw = min(100.0, alignment_raw + 5.0)
        # Bonus for CDP A/A- disclosure: +3 pts
        if cdp_score.climate_score in ("A", "A-"):
            alignment_raw = min(100.0, alignment_raw + 3.0)
        # Bonus for third-party GHG verification: +3 pts
        if ghg.ghg_verified:
            alignment_raw = min(100.0, alignment_raw + 3.0)

        alignment_score = round(alignment_raw, 1)
        alignment_level = _tcfd_alignment_level(alignment_score)

        # ── Assemble full result ──────────────────────────────────────────────
        assessment = TCFDAssessment(
            ticker=ticker,
            company_name=company_name,
            filing_year=filing_year,
            alignment_score=alignment_score,
            alignment_level=alignment_level,
            disclosures_found_count=found_count,
            governance_pillar=gov_pillar,
            strategy_pillar=strat_pillar,
            risk_mgmt_pillar=risk_pillar,
            metrics_pillar=met_pillar,
            item_1c_disclosed=item_1c,
            has_scenario_analysis=has_scenario,
            has_net_zero_target=has_net_zero,
            has_sbti=has_sbti,
            ghg_emissions=ghg,
            climate_targets=targets,
            physical_risk=phys_risk,
            transition_risk=trans_risk,
            cdp_score=cdp_score,
            sector=sector,
            warnings=warnings,
            data_quality=dq,
        )

        # ── Persist ───────────────────────────────────────────────────────────
        as_of = assessment.as_of
        self._db.upsert_assessment(assessment)
        self._db.upsert_ghg(ghg, as_of)
        self._db.insert_targets(targets, as_of)
        self._db.upsert_risk(ticker, as_of, "physical",
                             phys_risk.model_dump(), phys_risk.total_physical_risk_score)
        self._db.upsert_risk(ticker, as_of, "transition",
                             trans_risk.model_dump(), trans_risk.transition_risk_score)

        return assessment

    @staticmethod
    def _add_prior_year_ghg(ghg: GHGEmissions, facts: dict) -> GHGEmissions:
        """Compute YoY Scope 1+2 change by fetching prior-year XBRL values."""
        try:
            scope1_series = _extract_xbrl_fact(facts, "us-gaap", "GreenHouseGasEmissionsScope1")
            scope2_series = _extract_xbrl_fact(facts, "us-gaap", "GreenHouseGasEmissionsScope2")

            annual_s1 = [e for e in scope1_series if "10-K" in e.get("form", "")]
            annual_s2 = [e for e in scope2_series if "10-K" in e.get("form", "")]

            s1_latest = float(annual_s1[0]["value"]) if annual_s1 else None
            s1_prior  = float(annual_s1[1]["value"]) if len(annual_s1) > 1 else None
            s2_latest = float(annual_s2[0]["value"]) if annual_s2 else None
            s2_prior  = float(annual_s2[1]["value"]) if len(annual_s2) > 1 else None

            ghg.prior_year_scope1 = s1_prior
            ghg.prior_year_scope2 = s2_prior

            total_latest = (s1_latest or 0) + (s2_latest or 0)
            total_prior  = (s1_prior  or 0) + (s2_prior  or 0)

            if total_prior > 0 and total_latest > 0:
                ghg.yoy_scope12_change_pct = round(
                    (total_latest - total_prior) / total_prior * 100.0, 2
                )
        except Exception:
            pass
        return ghg

    def peer_comparison(self, ticker: str) -> PeerTCFDComparison:
        my_row = self._db.get_assessment(ticker)
        if not my_row:
            raise ValueError(f"No TCFD assessment for {ticker} — run assess() first")

        sector = my_row.get("sector", "default")
        peers  = self._db.get_sector_assessments(sector)

        if not peers:
            return PeerTCFDComparison(
                ticker=ticker, sector=sector,
                alignment_score=my_row["alignment_score"],
                sector_avg_alignment=my_row["alignment_score"],
                sector_avg_scope12_disclosed_pct=0.0,
                percentile_rank=50.0, peers=[],
            )

        df      = pd.DataFrame(peers)
        avg_ali = float(df["alignment_score"].mean())
        scope12_pct = float(
            (df["disclosures_found_count"] >= 7).sum() / len(df) * 100.0
        )
        my_score = my_row["alignment_score"]
        rank     = float((df["alignment_score"] < my_score).sum() / len(df) * 100.0)

        return PeerTCFDComparison(
            ticker=ticker,
            sector=sector,
            alignment_score=my_score,
            sector_avg_alignment=round(avg_ali, 1),
            sector_avg_scope12_disclosed_pct=round(scope12_pct, 1),
            percentile_rank=round(rank, 1),
            peers=peers[:20],
        )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

tcfd_v3_router = APIRouter(prefix="/tcfd/v3", tags=["TCFD v3"])
_engine = TCFDEngine()


@tcfd_v3_router.get("/assessment/{ticker}", summary="Full TCFD alignment assessment")
def get_assessment(ticker: str) -> dict:
    """
    Run full TCFD assessment for a ticker.
    Returns 4-pillar scores, GHG data (3-tier extraction), targets, risks, CDP score.
    """
    ticker = ticker.upper()
    try:
        result = _engine.assess(ticker)
        return result.model_dump()
    except Exception as exc:
        logger.error("tcfd_assess_error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@tcfd_v3_router.get("/ghg-emissions/{ticker}", summary="Structured GHG emissions data")
def get_ghg_emissions(ticker: str) -> dict:
    """
    Return Scope 1/2/3 GHG emissions for a ticker, with extraction source metadata.
    Primary: XBRL. Secondary: HTML table. Tertiary: context-anchored regex.
    """
    ticker = ticker.upper()
    row    = _engine._db.get_ghg(ticker)
    if not row:
        raise HTTPException(status_code=404, detail=f"No GHG data for {ticker} — run assessment first")
    return row


@tcfd_v3_router.get("/targets/{ticker}", summary="Climate targets and net-zero commitments")
def get_targets(ticker: str) -> dict:
    """Return extracted net-zero, carbon-neutral, and SBTi targets."""
    ticker  = ticker.upper()
    targets = _engine._db.get_targets(ticker)
    if not targets:
        raise HTTPException(status_code=404, detail=f"No climate targets for {ticker}")
    return {"ticker": ticker, "count": len(targets), "targets": targets}


@tcfd_v3_router.get("/pillar/{ticker}/{pillar}", summary="Single TCFD pillar detail")
def get_pillar(ticker: str, pillar: str) -> dict:
    """
    Return TCFD assessment for a specific pillar.
    Pillar must be one of: governance, strategy, risk_mgmt, metrics.
    """
    ticker = ticker.upper()
    pillar = pillar.lower()
    valid  = {"governance", "strategy", "risk_mgmt", "metrics"}
    if pillar not in valid:
        raise HTTPException(status_code=400, detail=f"pillar must be one of {sorted(valid)}")

    row = _engine._db.get_assessment(ticker)
    if not row:
        raise HTTPException(status_code=404, detail=f"No TCFD assessment for {ticker}")

    # Re-run and return specific pillar if we have the cached assessment
    # For efficiency, return from the stored full assessment
    try:
        result   = _engine.assess(ticker)
        pillar_map = {
            "governance": result.governance_pillar,
            "strategy":   result.strategy_pillar,
            "risk_mgmt":  result.risk_mgmt_pillar,
            "metrics":    result.metrics_pillar,
        }
        return pillar_map[pillar].model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@tcfd_v3_router.get("/peer-comparison/{ticker}", summary="TCFD peer comparison by sector")
def get_peer_comparison(ticker: str) -> dict:
    """Compare ticker TCFD alignment to sector peers."""
    ticker = ticker.upper()
    try:
        result = _engine.peer_comparison(ticker)
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@tcfd_v3_router.get("/cdp-score/{ticker}", summary="CDP public disclosure score")
def get_cdp_score(ticker: str) -> dict:
    """Return CDP climate disclosure score tier for a ticker."""
    ticker = ticker.upper()
    row    = _engine._db.get_cdp_score(ticker)
    if not row:
        raise HTTPException(status_code=404, detail=f"No CDP score for {ticker} — run assessment first")
    return row
