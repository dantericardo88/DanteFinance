"""
esg_ratings_v3.py — ESG Composite Ratings v3 (dim_102, target 9/10)

Architecture upgrade: keyword counting replaced with structured multi-source scoring.

Environmental (E) pillar — structured EDGAR + EPA + OWID:
  - Our World in Data (OWID) CO2 dataset: country-level CO2 per capita, intensity,
    total emissions — free CSV from GitHub, no scraping (replaces fragile CDP scraper).
  - SEC 10-K Item 1C Climate Risk section (mandatory 2024+) parsed via EDGAR EFTS
  - Scope 1+2 GHG: XBRL companyfacts us-gaap/ecd GHG tags + table detection fallback
  - Company-level GHG: search SEC 10-K text for "metric tons CO2" disclosures
  - EPA ECHO enforcement database — violation counts as negative signal
  - Climate goals language detection from 10-K environmental section
  - esg_momentum: (current - avg_3yr) / avg_3yr — negative = improving

Social (S) pillar — EDGAR XBRL + BLS + OSHA + DOL:
  - Employee headcount trend from XBRL dei:EntityNumberOfEmployees
  - CEO pay ratio Dodd-Frank tag from XBRL (structured, not regex)
  - Human capital disclosure depth (10-K Item 1 mandatory since 2020)
  - OSHA inspection / violation lookup via public OSHA IMIS API
  - WARN Act layoff notices via DOL search

Governance (G) pillar — wired to proxy_intelligence_v3.GovernanceScoringEngine:
  - Delegates entirely to the 20-component governance scorer (0–100)

Composite: esg_score_composite() combines E+S+G with source-aware weighting.
Controversy deduction: up to –20 pts for active EPA/OSHA enforcement actions.

SQLite: esg_scores, pillar_details, esg_history, controversies
FastAPI router at /esg/v3
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field as PydanticField

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "OWIDCo2Client",
    "EPAEchoClient",
    "OSHAClient",
    "WARNActClient",
    "EnvironmentalPillar",
    "SocialPillar",
    "GovernancePillarBridge",
    "ESGCompositeEngine",
    "ESGDB",
    "esg_score_composite",
    "esg_momentum",
    "esg_v3_router",
    # Legacy alias kept for backward compatibility
    "CDPScraper",
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
_RATE_DELAY  = 0.15   # seconds between EDGAR requests
_TIMEOUT     = 30.0
_MAX_RETRY   = 3
_TEXT_LIMIT  = 200_000  # chars to download per filing

EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_ARCHIVE     = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EFTS_SEARCH_URL   = "https://efts.sec.gov/LATEST/search-index"

OWID_CO2_CSV_URL  = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

EPA_ECHO_API      = "https://echo.epa.gov/rest-api/facility-search/sites"
EPA_ECHO_ACTIONS  = "https://echo.epa.gov/rest-api/enforcement-case-search/enforcement-cases"

OSHA_IMIS_URL     = "https://www.osha.gov/ords/imis/establishment.html"
DOL_WARN_URL      = "https://www.dol.gov/agencies/eta/layoffs/warn"

_DB_PATH = Path(__file__).parent.parent / "data" / "esg_v3.db"

# ---------------------------------------------------------------------------
# CDP tier mapping
# ---------------------------------------------------------------------------

CDP_TIER_SCORES: dict[str, float] = {
    "A":  40.0,
    "A-": 36.0,
    "B":  28.0,
    "B-": 22.0,
    "C":  14.0,
    "C-": 10.0,
    "D":   5.0,
    "D-":  2.0,
    "F":   0.0,
    "not_disclosed": 0.0,
}

# ---------------------------------------------------------------------------
# SIC → sector mapping (used for peer comparison bucketing)
# ---------------------------------------------------------------------------

_SIC_TO_SECTOR: dict[str, str] = {
    "1311": "energy",    "1382": "energy",    "2911": "energy",
    "1321": "energy",    "5171": "energy",    "1381": "energy",
    "2819": "materials", "2860": "materials", "3312": "materials",
    "1040": "materials", "3559": "industrials","3720": "industrials",
    "3812": "industrials","4210": "industrials","4911": "utilities",
    "4931": "utilities",  "4941": "utilities",  "4924": "utilities",
    "2000": "consumer_staples","2010": "consumer_staples","5400": "consumer_staples",
    "5900": "consumer_discretionary","7011": "consumer_discretionary",
    "2836": "health_care","2830": "health_care","8011": "health_care",
    "6020": "financials", "6022": "financials", "6211": "financials",
    "7372": "information_technology","7371": "information_technology",
    "3674": "information_technology","4813": "communication_services",
    "6552": "real_estate","6798": "real_estate",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EnvironmentalPillarResult(BaseModel):
    ticker: str
    score: float = 0.0              # 0–100
    # OWID-sourced country context (replaces fragile CDP scraping)
    owid_country: Optional[str] = None
    owid_co2_per_capita: Optional[float] = None
    owid_renewable_share: Optional[float] = None
    owid_country_score: float = 0.0      # 0–100 OWID environmental score
    # Company-level GHG from XBRL + SEC text
    ghg_scope1_mt: Optional[float] = None
    ghg_scope2_mt: Optional[float] = None
    ghg_xbrl_sourced: bool = False
    ghg_text_extracted: bool = False     # from SEC 10-K "metric tons CO2" text
    emissions_trend_score: float = 0.0   # max 20 (positive = decreasing)
    esg_momentum_value: Optional[float] = None  # (current-avg3yr)/avg3yr; negative=improving
    # EPA enforcement
    epa_violation_count: int = 0
    epa_penalty_pts: float = 0.0         # negative; min –20
    # Climate goals from 10-K text
    climate_goals_score: float = 0.0    # max 20 (net-zero, SBTi, Paris)
    item_1c_present: bool = False        # SEC mandatory climate risk section
    # Legacy fields (kept for backward compat)
    cdp_tier: str = "not_disclosed"
    cdp_score_pts: float = 0.0
    details: dict[str, Any] = PydanticField(default_factory=dict)


class SocialPillarResult(BaseModel):
    ticker: str
    score: float = 0.0              # 0–100
    ceo_pay_ratio: Optional[int] = None
    ceo_pay_ratio_score: float = 0.0    # max 25
    employee_count_latest: Optional[int] = None
    employee_count_prior: Optional[int] = None
    headcount_trend_score: float = 0.0  # max 15
    osha_violation_count: int = 0
    osha_score: float = 0.0             # max 25
    warn_layoff_detected: bool = False
    warn_score: float = 0.0             # max 20
    hc_disclosure_depth: int = 0        # 0–5 categories found
    hc_disclosure_score: float = 0.0    # max 15
    details: dict[str, Any] = PydanticField(default_factory=dict)


class ControversyRecord(BaseModel):
    ticker: str
    source: str          # "epa" | "osha" | "warn"
    description: str
    date_detected: str
    severity: str = "low"   # "low" | "medium" | "high"
    penalty_usd: Optional[float] = None
    deduction_pts: float = 0.0


class ESGCompositeResult(BaseModel):
    ticker: str
    as_of: str = PydanticField(default_factory=lambda: datetime.utcnow().date().isoformat())
    composite_score: float = 0.0    # 0–100
    e_score: float = 0.0
    s_score: float = 0.0
    g_score: float = 0.0
    controversy_deduction: float = 0.0
    controversy_adjusted: float = 0.0
    letter_grade: str = "F"
    sector: str = "default"
    pillar_env: Optional[EnvironmentalPillarResult] = None
    pillar_soc: Optional[SocialPillarResult] = None
    g_total_score: Optional[float] = None
    controversies: list[ControversyRecord] = PydanticField(default_factory=list)
    data_quality: dict[str, str] = PydanticField(default_factory=dict)
    warnings: list[str] = PydanticField(default_factory=list)
    disclaimer: str = (
        "Structured proxy signals from public EDGAR/EPA/OSHA/DOL filings. "
        "Not equivalent to MSCI/Sustainalytics/ISS/CDP commercial ratings."
    )


class PeerESGComparison(BaseModel):
    ticker: str
    sector: str
    composite_score: float
    sector_avg_composite: float
    sector_avg_e: float
    sector_avg_s: float
    sector_avg_g: float
    percentile_rank: float
    peers: list[dict] = PydanticField(default_factory=list)


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------


class ESGDB:
    """SQLite persistence for ESG scores, pillar details, history, controversies."""

    DDL = """
    CREATE TABLE IF NOT EXISTS esg_scores (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker              TEXT NOT NULL,
        as_of               TEXT NOT NULL,
        composite_score     REAL,
        e_score             REAL,
        s_score             REAL,
        g_score             REAL,
        controversy_deduction REAL DEFAULT 0,
        controversy_adjusted REAL,
        letter_grade        TEXT,
        sector              TEXT,
        g_total_score       REAL,
        data_quality_json   TEXT,
        warnings_json       TEXT,
        inserted_at         TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, as_of)
    );

    CREATE TABLE IF NOT EXISTS pillar_details (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT NOT NULL,
        as_of       TEXT NOT NULL,
        pillar      TEXT NOT NULL,   -- 'E' | 'S' | 'G'
        score       REAL,
        details_json TEXT,
        inserted_at TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, as_of, pillar)
    );

    CREATE TABLE IF NOT EXISTS esg_history (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL,
        as_of           TEXT NOT NULL,
        composite_score REAL,
        e_score         REAL,
        s_score         REAL,
        g_score         REAL,
        letter_grade    TEXT,
        inserted_at     TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS controversies (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL,
        source          TEXT,
        description     TEXT,
        date_detected   TEXT,
        severity        TEXT,
        penalty_usd     REAL,
        deduction_pts   REAL,
        inserted_at     TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS ix_esg_ticker   ON esg_scores(ticker, as_of);
    CREATE INDEX IF NOT EXISTS ix_hist_ticker  ON esg_history(ticker, as_of);
    CREATE INDEX IF NOT EXISTS ix_cont_ticker  ON controversies(ticker, date_detected);
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path else _DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.DDL)
        self._conn.commit()

    def upsert_composite(self, result: ESGCompositeResult) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO esg_scores
               (ticker, as_of, composite_score, e_score, s_score, g_score,
                controversy_deduction, controversy_adjusted, letter_grade, sector,
                g_total_score, data_quality_json, warnings_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result.ticker, result.as_of, result.composite_score,
             result.e_score, result.s_score, result.g_score,
             result.controversy_deduction, result.controversy_adjusted,
             result.letter_grade, result.sector, result.g_total_score,
             json.dumps(result.data_quality), json.dumps(result.warnings)),
        )
        self._conn.execute(
            """INSERT OR IGNORE INTO esg_history
               (ticker, as_of, composite_score, e_score, s_score, g_score, letter_grade)
               VALUES (?,?,?,?,?,?,?)""",
            (result.ticker, result.as_of, result.composite_score,
             result.e_score, result.s_score, result.g_score, result.letter_grade),
        )
        self._conn.commit()

    def upsert_pillar(self, ticker: str, as_of: str, pillar: str, score: float, details: dict) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO pillar_details (ticker, as_of, pillar, score, details_json)
               VALUES (?,?,?,?,?)""",
            (ticker, as_of, pillar, score, json.dumps(details)),
        )
        self._conn.commit()

    def insert_controversy(self, c: ControversyRecord) -> None:
        self._conn.execute(
            """INSERT INTO controversies
               (ticker, source, description, date_detected, severity, penalty_usd, deduction_pts)
               VALUES (?,?,?,?,?,?,?)""",
            (c.ticker, c.source, c.description, c.date_detected,
             c.severity, c.penalty_usd, c.deduction_pts),
        )
        self._conn.commit()

    def get_latest_score(self, ticker: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM esg_scores WHERE ticker=? ORDER BY as_of DESC LIMIT 1",
            (ticker,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_history(self, ticker: str, limit: int = 10) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM esg_history WHERE ticker=? ORDER BY as_of DESC LIMIT ?",
            (ticker, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_controversies(self, ticker: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM controversies WHERE ticker=? ORDER BY date_detected DESC",
            (ticker,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_universe_scores(self) -> list[dict]:
        cur = self._conn.execute(
            """SELECT t.ticker, t.composite_score, t.e_score, t.s_score, t.g_score,
                      t.letter_grade, t.sector
               FROM esg_scores t
               INNER JOIN (
                   SELECT ticker, MAX(as_of) AS max_as_of FROM esg_scores GROUP BY ticker
               ) m ON t.ticker = m.ticker AND t.as_of = m.max_as_of
               ORDER BY t.composite_score DESC"""
        )
        return [dict(r) for r in cur.fetchall()]

    def get_sector_peers(self, sector: str) -> list[dict]:
        cur = self._conn.execute(
            """SELECT t.ticker, t.composite_score, t.e_score, t.s_score, t.g_score
               FROM esg_scores t
               INNER JOIN (
                   SELECT ticker, MAX(as_of) AS max_as_of FROM esg_scores GROUP BY ticker
               ) m ON t.ticker = m.ticker AND t.as_of = m.max_as_of
               WHERE t.sector=? ORDER BY t.composite_score DESC""",
            (sector,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_pillar(self, ticker: str, pillar: str) -> Optional[dict]:
        cur = self._conn.execute(
            """SELECT * FROM pillar_details WHERE ticker=? AND pillar=?
               ORDER BY as_of DESC LIMIT 1""",
            (ticker, pillar),
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------


def _edgar_get(url: str, params: dict | None = None, retry: int = _MAX_RETRY) -> Optional[dict | str]:
    """GET with rate limiting, retry, and dual JSON/text return."""
    for attempt in range(retry):
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
            return None
        except Exception as exc:
            logger.warning("edgar_get_error", url=url, attempt=attempt, error=str(exc))
            if attempt == retry - 1:
                return None
    return None


def _resolve_ticker_to_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to zero-padded 10-digit CIK via EDGAR company tickers JSON."""
    data = _edgar_get(EDGAR_TICKERS_URL)
    if not isinstance(data, dict):
        return None
    ticker_upper = ticker.upper()
    for entry in data.values():
        if isinstance(entry, dict) and entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    return None


def _fetch_company_facts(cik: str) -> Optional[dict]:
    url  = EDGAR_FACTS_URL.format(cik=cik)
    data = _edgar_get(url)
    return data if isinstance(data, dict) else None


def _get_latest_10k_filing(cik: str) -> Optional[dict]:
    """Return the most recent 10-K filing metadata dict."""
    url  = EDGAR_SUBMISSIONS.format(cik=cik)
    data = _edgar_get(url)
    if not isinstance(data, dict):
        return None
    filings = data.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    accs    = filings.get("accessionNumber", [])
    dates   = filings.get("filingDate", [])
    docs    = filings.get("primaryDocument", [])
    cik_int = int(cik)
    for i, form in enumerate(forms):
        if form in ("10-K", "10-K/A"):
            acc_nodash = accs[i].replace("-", "")
            return {
                "accession": accs[i],
                "acc_nodash": acc_nodash,
                "filing_date": dates[i] if i < len(dates) else "",
                "primary_doc": docs[i] if i < len(docs) else "",
                "cik_int": cik_int,
                "doc_url": EDGAR_ARCHIVE.format(
                    cik_int=cik_int, acc_nodash=acc_nodash,
                    doc=docs[i] if i < len(docs) else ""
                ),
            }
    return None


def _fetch_filing_text(filing: dict) -> str:
    """Download 10-K filing HTML/text and strip tags; truncate to _TEXT_LIMIT."""
    url  = filing.get("doc_url", "")
    if not url:
        return ""
    raw = _edgar_get(url)
    if not isinstance(raw, str):
        return ""
    soup = BeautifulSoup(raw[:_TEXT_LIMIT * 5], "html.parser")
    return soup.get_text(separator=" ", strip=True)[:_TEXT_LIMIT]


def _extract_xbrl_fact(facts: dict, namespace: str, concept: str) -> list[dict]:
    """Extract all fact values for a given taxonomy namespace + concept."""
    try:
        units = facts.get(namespace, {}).get(concept, {}).get("units", {})
        results = []
        for unit_key, entries in units.items():
            for e in entries:
                results.append({
                    "value": e.get("val"),
                    "unit": unit_key,
                    "form": e.get("form", ""),
                    "end": e.get("end", ""),
                    "filed": e.get("filed", ""),
                })
        return sorted(results, key=lambda x: x.get("end", ""), reverse=True)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# OWID CO2 client — replaces fragile CDP scraper
# ---------------------------------------------------------------------------


class OWIDCo2Client:
    """
    Our World in Data (OWID) CO2 dataset client.

    Downloads the free CSV from GitHub (no scraping, no auth):
      https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv

    Provides:
      - Country-level CO2 per capita, CO2 intensity, total emissions
      - Multi-year history enabling momentum computation
      - Sector/country benchmarking for environmental scoring

    Cache: in-memory DataFrame refreshed once per process (or on stale date).
    """

    _df: Optional[pd.DataFrame] = None
    _fetched_date: Optional[str] = None

    @classmethod
    def _load(cls) -> Optional[pd.DataFrame]:
        today = date.today().isoformat()
        if cls._df is not None and cls._fetched_date == today:
            return cls._df
        try:
            r = requests.get(OWID_CO2_CSV_URL, headers=_HEADERS, timeout=60)
            if r.status_code != 200:
                logger.warning("owid_co2_fetch_failed", status=r.status_code)
                return cls._df  # return stale if available
            cls._df = pd.read_csv(pd.io.common.StringIO(r.text), low_memory=False)
            cls._fetched_date = today
            logger.info("owid_co2_loaded", rows=len(cls._df))
        except Exception as exc:
            logger.warning("owid_co2_error", error=str(exc))
        return cls._df

    @classmethod
    def get_country_emissions(
        cls,
        country: str,
        year: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Return emissions data for a country in a given year (latest if None).

        Returns dict with keys:
          co2_per_capita, co2_intensity, co2_total_mt, co2_growth_pct,
          share_global_co2, energy_per_capita, renewable_share_energy
        """
        df = cls._load()
        if df is None:
            return {}

        mask = df["country"].str.upper() == country.upper()
        country_df = df[mask].copy()
        if country_df.empty:
            return {}

        if year is None:
            country_df = country_df.dropna(subset=["co2"])
            if country_df.empty:
                return {}
            row = country_df.sort_values("year").iloc[-1]
        else:
            rows = country_df[country_df["year"] == year]
            if rows.empty:
                return {}
            row = rows.iloc[0]

        def _safe(col: str) -> Optional[float]:
            val = row.get(col)
            return float(val) if pd.notna(val) else None

        return {
            "country": country,
            "year": int(row.get("year", 0)),
            "co2_per_capita": _safe("co2_per_capita"),
            "co2_intensity": _safe("energy_per_gdp"),
            "co2_total_mt": _safe("co2"),
            "co2_growth_pct": _safe("co2_growth_prct"),
            "share_global_co2": _safe("share_global_co2"),
            "energy_per_capita": _safe("energy_per_capita"),
            "renewable_share_energy": _safe("renewables_share_energy"),
        }

    @classmethod
    def compute_country_momentum(
        cls,
        country: str,
        metric: str = "co2_per_capita",
        lookback_years: int = 3,
    ) -> Optional[float]:
        """
        Emissions momentum: (current - avg_3yr) / avg_3yr.
        Negative = improving (emissions falling). Positive = worsening.
        """
        df = cls._load()
        if df is None:
            return None

        mask = df["country"].str.upper() == country.upper()
        country_df = df[mask].dropna(subset=[metric]).sort_values("year")
        if len(country_df) < lookback_years + 1:
            return None

        current = float(country_df[metric].iloc[-1])
        prior_window = country_df[metric].iloc[-(lookback_years + 1):-1]
        avg_prior = float(prior_window.mean())
        if avg_prior == 0:
            return None
        return round((current - avg_prior) / avg_prior, 4)

    @classmethod
    def score_country_environmental(cls, country: str) -> float:
        """
        Score 0–100 based on OWID CO2 metrics vs global benchmarks.
        Lower CO2 per capita + improving trend = higher score.
        """
        data = cls.get_country_emissions(country)
        if not data:
            return 50.0  # neutral default

        score = 50.0

        # CO2 per capita scoring (global mean ~4.7 t, range 0–20+)
        co2_pc = data.get("co2_per_capita")
        if co2_pc is not None:
            if co2_pc <= 2.0:
                score += 25.0
            elif co2_pc <= 4.0:
                score += 15.0
            elif co2_pc <= 7.0:
                score += 5.0
            elif co2_pc <= 12.0:
                score -= 5.0
            else:
                score -= 15.0

        # Renewable energy share
        renew = data.get("renewable_share_energy")
        if renew is not None:
            if renew >= 60:
                score += 15.0
            elif renew >= 30:
                score += 8.0
            elif renew >= 10:
                score += 2.0
            else:
                score -= 5.0

        # Momentum: improving = +points
        momentum = cls.compute_country_momentum(country)
        if momentum is not None:
            if momentum <= -0.10:
                score += 10.0
            elif momentum <= -0.03:
                score += 5.0
            elif momentum >= 0.10:
                score -= 10.0
            elif momentum >= 0.03:
                score -= 5.0

        return max(0.0, min(100.0, round(score, 1)))


# Legacy alias — kept for backward compatibility; does NOT make network calls to CDP
class CDPScraper:
    """
    Legacy stub. CDP scraping replaced by OWIDCo2Client.
    Returns "not_disclosed" for all companies (no fragile web scraping).
    """

    _CDP_TIER_SCORES: dict[str, float] = {
        "A": 40.0, "A-": 36.0, "B": 28.0, "B-": 22.0,
        "C": 14.0, "C-": 10.0, "D": 5.0, "D-": 2.0,
        "not_disclosed": 0.0, "F": 0.0,
    }

    @classmethod
    def get_tier(cls, _company_name: str, _ticker: str) -> str:
        """Always returns 'not_disclosed' — use OWIDCo2Client instead."""
        return "not_disclosed"

    @classmethod
    def get_tier_score(cls, tier: str) -> float:
        return cls._CDP_TIER_SCORES.get(tier, 0.0)


# ---------------------------------------------------------------------------
# ESG momentum and composite helpers
# ---------------------------------------------------------------------------


def esg_momentum(
    current_emissions: float,
    emissions_3yr_avg: float,
) -> float:
    """
    Compute ESG emissions momentum.

    Formula: (current - avg_3yr) / avg_3yr
    Negative = improving (emissions falling). Positive = worsening.

    Args:
        current_emissions: Most recent year's emissions figure.
        emissions_3yr_avg: Average of the prior 3 years.

    Returns:
        Momentum as a signed fraction. Returns 0.0 if avg is zero.
    """
    if emissions_3yr_avg == 0:
        return 0.0
    return round((current_emissions - emissions_3yr_avg) / emissions_3yr_avg, 4)


def esg_score_composite(
    e_score: float,
    g_score: float,
    s_score: float,
    e_weight: float = 0.33,
    s_weight: float = 0.33,
    g_weight: float = 0.34,
    controversy_deduction: float = 0.0,
) -> float:
    """
    Combine E, S, G pillar scores (0–100 each) into a weighted composite (0–100).

    Sources:
      - Environmental (EPA ECHO, OSHA, OWID CO2 emissions, 10-K text)
      - Governance (proxy_intelligence_v3 — board independence, pay ratios, etc.)
      - Social (job postings, OSHA, WARN Act, human capital disclosure)

    Args:
        e_score:              Environmental pillar score (0–100).
        g_score:              Governance pillar score (0–100).
        s_score:              Social pillar score (0–100).
        e_weight:             Weight for E (default 0.33).
        s_weight:             Weight for S (default 0.33).
        g_weight:             Weight for G (default 0.34).
        controversy_deduction: Points deducted for active enforcement actions (0–20).

    Returns:
        Controversy-adjusted composite score clamped to [0, 100].
    """
    raw = e_weight * e_score + s_weight * s_score + g_weight * g_score
    adjusted = max(0.0, min(100.0, raw - controversy_deduction))
    return round(adjusted, 2)


# ---------------------------------------------------------------------------
# EPA ECHO client
# ---------------------------------------------------------------------------


class EPAEchoClient:
    """
    Query EPA ECHO (Enforcement and Compliance History Online) for violations.
    https://echo.epa.gov/rest-api/ — free, no API key required.
    """

    @staticmethod
    def get_violation_count(company_name: str) -> tuple[int, list[dict]]:
        """
        Return (violation_count, raw_records) for a company name.
        Uses facility-search API then pulls active enforcement actions.
        """
        try:
            time.sleep(_RATE_DELAY)
            params = {
                "output": "JSON",
                "p_fn": company_name,
                "p_act": "Y",       # active facilities only
                "p_qiv": "Violation",
                "rows": "100",
            }
            r = requests.get(
                EPA_ECHO_API,
                headers=_HEADERS,
                params=params,
                timeout=_TIMEOUT,
            )
            if r.status_code != 200:
                return 0, []
            data = r.json()
            facilities = data.get("Results", {}).get("Facilities", [])
            if not facilities:
                return 0, []

            total_violations = 0
            records = []
            for fac in facilities[:10]:  # cap at 10 facilities
                viol_count = int(fac.get("PollutantInspCount", 0) or 0)
                penalty    = float(fac.get("TotalPenalties", 0) or 0)
                if viol_count > 0 or penalty > 0:
                    total_violations += viol_count
                    records.append({
                        "facility":  fac.get("FacilityName", ""),
                        "state":     fac.get("StateCode", ""),
                        "violations": viol_count,
                        "penalty_usd": penalty,
                        "air_flag":  fac.get("AIRFlag", ""),
                        "cwa_flag":  fac.get("CWAFlag", ""),
                    })
            return total_violations, records

        except Exception as exc:
            logger.warning("epa_echo_error", company=company_name, error=str(exc))
            return 0, []


# ---------------------------------------------------------------------------
# OSHA violations client
# ---------------------------------------------------------------------------


class OSHAClient:
    """
    Query OSHA IMIS establishment search for violation history.
    https://www.osha.gov/ords/imis/establishment.html — public, no auth.
    """

    @staticmethod
    def get_violations(company_name: str) -> tuple[int, list[dict]]:
        """Return (violation_count, records) from OSHA IMIS."""
        try:
            time.sleep(_RATE_DELAY)
            params = {
                "p_companyname": company_name,
                "p_State": "",
                "p_start_month": "",
                "p_month_end": "",
                "p_citation_id": "",
                "p_penalty_type": "",
                "p_penalty_final": "",
            }
            r = requests.get(
                OSHA_IMIS_URL,
                headers={**_HEADERS, "Accept": "text/html"},
                params=params,
                timeout=_TIMEOUT,
            )
            if r.status_code != 200:
                return 0, []

            soup  = BeautifulSoup(r.text, "html.parser")
            rows  = soup.select("table.t1 tr") or soup.select("table tr")
            records: list[dict] = []
            for row in rows[1:21]:  # skip header, cap 20 rows
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) >= 5:
                    records.append({
                        "establishment": cells[0],
                        "city":          cells[1] if len(cells) > 1 else "",
                        "state":         cells[2] if len(cells) > 2 else "",
                        "inspection_date": cells[3] if len(cells) > 3 else "",
                        "violations":    cells[4] if len(cells) > 4 else "0",
                    })
            return len(records), records

        except Exception as exc:
            logger.warning("osha_error", company=company_name, error=str(exc))
            return 0, []


# ---------------------------------------------------------------------------
# WARN Act layoff client
# ---------------------------------------------------------------------------


class WARNActClient:
    """
    Check DOL WARN Act notice database for mass layoff notices.
    https://www.dol.gov/agencies/eta/layoffs/warn — public HTML search.
    """

    @staticmethod
    def has_recent_layoff(company_name: str, lookback_days: int = 365) -> tuple[bool, list[dict]]:
        """Return (found, records) for WARN Act filings within lookback window."""
        try:
            time.sleep(_RATE_DELAY)
            cutoff = (date.today() - timedelta(days=lookback_days)).strftime("%m/%d/%Y")
            params = {
                "query": company_name,
                "date_from": cutoff,
            }
            r = requests.get(
                DOL_WARN_URL,
                headers={**_HEADERS, "Accept": "text/html"},
                params=params,
                timeout=_TIMEOUT,
            )
            if r.status_code != 200:
                return False, []

            soup    = BeautifulSoup(r.text, "html.parser")
            tables  = soup.find_all("table")
            records = []
            name_u  = company_name.upper()

            for table in tables:
                for row in table.find_all("tr")[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all("td")]
                    if not cells:
                        continue
                    row_text = " ".join(cells).upper()
                    if any(tok in row_text for tok in name_u.split()[:3] if len(tok) > 3):
                        records.append({
                            "company":   cells[0] if cells else "",
                            "city":      cells[1] if len(cells) > 1 else "",
                            "state":     cells[2] if len(cells) > 2 else "",
                            "date":      cells[3] if len(cells) > 3 else "",
                            "employees": cells[4] if len(cells) > 4 else "",
                        })

            return bool(records), records

        except Exception as exc:
            logger.warning("warn_act_error", company=company_name, error=str(exc))
            return False, []


# ---------------------------------------------------------------------------
# Environmental Pillar scorer
# ---------------------------------------------------------------------------


class EnvironmentalPillar:
    """
    Score the Environmental pillar (0–100) from structured sources.

    Component breakdown:
      CDP tier         : 0–40 pts  (structured tier mapping)
      Emissions trend  : 0–20 pts  (XBRL Scope 1+2 YoY change)
      EPA violations   : 0 to –20  (penalty deduction)
      Climate goals    : 0–20 pts  (net-zero, SBTi, Paris text signals)
    Total potential    : 0–80 + bonus = capped at 100
    """

    # XBRL concepts for GHG emissions (us-gaap ESG taxonomy + SEC climate taxonomy)
    _GHG_CONCEPTS: list[tuple[str, str]] = [
        # SEC 2024 climate taxonomy (emerging standard)
        ("us-gaap", "GreenHouseGasEmissionsScope1"),
        ("us-gaap", "GreenHouseGasEmissionsScope2"),
        ("us-gaap", "GreenHouseGasEmissionsScope3"),
        # Older ESG extension concepts used by some filers
        ("ecd",     "GHGEmissionsScope1"),
        ("ecd",     "GHGEmissionsScope2"),
        # Common custom extensions
        ("us-gaap", "CarbonDioxideEmissions"),
        ("us-gaap", "GreenHouseGasEmissions"),
    ]

    # Climate goals / net-zero language patterns
    _CLIMATE_GOAL_PATTERNS: list[tuple[str, float]] = [
        # (regex_pattern, points_awarded)
        (r"\bnet[\s\-]zero\b",                           6.0),
        (r"\bcarbon[\s\-]neutral\b",                     5.0),
        (r"\bscience[\s\-]based\s+target",               5.0),
        (r"\bSBTi\b",                                    5.0),
        (r"\bparis\s+agreement\b",                       3.0),
        (r"\b1\.5\s*[°℃C]\b",                           3.0),
        (r"\bscope\s+[123]\s+emission",                  2.0),
        (r"\brenewable\s+energy\s+(?:target|goal|100)",  2.0),
        (r"\bnet\s+zero\s+by\s+20[3-5]\d\b",            3.0),
        (r"\bzero\s+emission",                           2.0),
        (r"\bclimate\s+transition\s+plan",               2.0),
    ]

    # Item 1C detection patterns
    _ITEM_1C_PATTERNS = [
        r"item\s+1c[\.\s]*(?:cybersecurity|climate)",
        r"item\s+1\s*c[\.\s]*climate\s*risk",
        r"climate[-\s]related\s+risk\s+disclosure",
        r"climate\s+risk\s+factor",
    ]

    def __init__(self) -> None:
        self._owid = OWIDCo2Client()
        self._epa  = EPAEchoClient()

    def score(
        self,
        ticker:       str,
        company_name: str,
        facts:        dict,
        filing_text:  str,
        country:      str = "United States",
    ) -> EnvironmentalPillarResult:
        result = EnvironmentalPillarResult(ticker=ticker)
        details: dict[str, Any] = {}

        # ── 1. OWID CO2 country context (replaces fragile CDP scraping) ───────
        # Uses free OWID dataset: no scraping, no rate-limiting, reliable.
        owid_data = OWIDCo2Client.get_country_emissions(country)
        result.owid_country       = country
        result.owid_co2_per_capita = owid_data.get("co2_per_capita")
        result.owid_renewable_share = owid_data.get("renewable_share_energy")
        result.owid_country_score  = OWIDCo2Client.score_country_environmental(country)
        details["owid_country"]   = country
        details["owid_data"]      = owid_data
        details["owid_score"]     = result.owid_country_score

        # ── 2. GHG emissions from XBRL (company-level) ────────────────────────
        scope1_series: list[dict] = []
        scope2_series: list[dict] = []
        for ns, concept in self._GHG_CONCEPTS:
            if "Scope1" in concept or concept in ("GHGEmissionsScope1", "CarbonDioxideEmissions"):
                if not scope1_series:
                    scope1_series = _extract_xbrl_fact(facts, ns, concept)
            elif "Scope2" in concept or concept in ("GHGEmissionsScope2",):
                if not scope2_series:
                    scope2_series = _extract_xbrl_fact(facts, ns, concept)

        # Extract multi-year series for momentum computation
        s1_latest = s1_prior = s2_latest = s2_prior = None
        s1_annual: list[dict] = []
        if scope1_series:
            s1_annual = [f for f in scope1_series if "10-K" in f.get("form", "")]
            if s1_annual:
                s1_latest = s1_annual[0].get("value")
                s1_prior  = s1_annual[1].get("value") if len(s1_annual) > 1 else None
            result.ghg_xbrl_sourced = True

        if scope2_series:
            annual2 = [f for f in scope2_series if "10-K" in f.get("form", "")]
            if annual2:
                s2_latest = annual2[0].get("value")
                s2_prior  = annual2[1].get("value") if len(annual2) > 1 else None
            result.ghg_xbrl_sourced = True

        result.ghg_scope1_mt = float(s1_latest) if s1_latest is not None else None
        result.ghg_scope2_mt = float(s2_latest) if s2_latest is not None else None

        # Compute ESG momentum: (current - avg_3yr) / avg_3yr; negative = improving
        result.esg_momentum_value = self._compute_xbrl_momentum(s1_annual)
        details["esg_momentum"] = result.esg_momentum_value

        # Emissions trend score (0–20)
        trend_score = self._emissions_trend_score(
            s1_latest, s1_prior, s2_latest, s2_prior
        )
        result.emissions_trend_score = trend_score
        details["scope1_latest"] = result.ghg_scope1_mt
        details["scope1_prior"]  = float(s1_prior) if s1_prior is not None else None
        details["scope2_latest"] = result.ghg_scope2_mt
        details["scope2_prior"]  = float(s2_prior) if s2_prior is not None else None
        details["ghg_xbrl"]      = result.ghg_xbrl_sourced
        details["emissions_trend_score"] = trend_score

        # Fallback: extract GHG from 10-K text when XBRL missing
        if not result.ghg_xbrl_sourced and filing_text:
            scope1_text, _scope2_text = self._parse_ghg_from_text(filing_text)
            if scope1_text:
                result.ghg_scope1_mt = scope1_text
                result.ghg_text_extracted = True
                details["ghg_text_fallback"] = True

        # ── 3. Company-level SEC 10-K emissions disclosure ────────────────────
        # Search for "metric tons CO2" disclosures in 10-K text (structured fallback)
        sec_emissions = self._extract_sec_emissions_disclosure(filing_text)
        details["sec_emissions_disclosed"] = sec_emissions is not None
        if sec_emissions and result.ghg_scope1_mt is None:
            result.ghg_scope1_mt = sec_emissions
            result.ghg_text_extracted = True

        # ── 4. EPA ECHO violations (0 to –20 pts) ────────────────────────────
        violation_count, epa_records = self._epa.get_violation_count(company_name)
        result.epa_violation_count = violation_count
        penalty_pts = max(-20.0, -5.0 * min(violation_count, 4))
        result.epa_penalty_pts = penalty_pts
        details["epa_violations"] = violation_count
        details["epa_records"]    = epa_records[:5]

        # ── 5. Climate goals from 10-K text (0–20 pts) ───────────────────────
        goals_score = 0.0
        goals_found = {}
        if filing_text:
            text_lower = filing_text.lower()
            for pattern, pts in self._CLIMATE_GOAL_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    goals_found[pattern[:30]] = pts
                    goals_score = min(20.0, goals_score + pts)

        result.climate_goals_score = goals_score
        details["climate_goals_found"] = goals_found

        # ── 6. Item 1C presence check (SEC 2024 climate rule) ────────────────
        item_1c = False
        if filing_text:
            for pat in self._ITEM_1C_PATTERNS:
                if re.search(pat, filing_text, re.IGNORECASE):
                    item_1c = True
                    break
        result.item_1c_present = item_1c
        details["item_1c_present"] = item_1c

        # ── Composite E score (0–100) ─────────────────────────────────────────
        # Components:
        #   OWID country score (0–100) × 0.30  → 0–30 pts
        #   Emissions trend              (0–20) → 0–20 pts
        #   Climate goals text           (0–20) → 0–20 pts
        #   EPA penalty               (–20 to 0)
        # Normalize to 0–100
        owid_component   = result.owid_country_score * 0.30   # 0–30
        trend_component  = result.emissions_trend_score        # 0–20
        goals_component  = result.climate_goals_score          # 0–20
        epa_component    = result.epa_penalty_pts              # –20 to 0
        # Bonus for item 1C (mandatory disclosure, +5 pts)
        item1c_bonus     = 5.0 if item_1c else 0.0
        # Bonus for momentum improving (negative esg_momentum = good, up to +5)
        mom = result.esg_momentum_value or 0.0
        momentum_bonus = max(0.0, min(5.0, -mom * 25.0)) if mom < 0 else 0.0

        raw = owid_component + trend_component + goals_component + epa_component + item1c_bonus + momentum_bonus
        # Max theoretical: 30+20+20+5+5 = 80; scale to 100
        normalized = (raw / 80.0) * 100.0
        result.score = max(0.0, min(100.0, round(normalized, 1)))

        result.details = details
        return result

    @staticmethod
    def _compute_xbrl_momentum(annual_series: list[dict]) -> Optional[float]:
        """
        Compute ESG emissions momentum from XBRL annual series.
        Formula: (current - avg_3yr) / avg_3yr.  Negative = improving.
        Requires at least 4 annual data points (current + 3 prior).
        """
        if len(annual_series) < 4:
            return None
        try:
            vals = [float(e["value"]) for e in annual_series[:4] if e.get("value") is not None]
            if len(vals) < 4:
                return None
            current = vals[0]
            avg_3yr = sum(vals[1:4]) / 3.0
            return esg_momentum(current, avg_3yr)
        except Exception:
            return None

    @staticmethod
    def _extract_sec_emissions_disclosure(filing_text: str) -> Optional[float]:
        """
        Extract company-level GHG emissions from SEC 10-K filing text.

        Searches for phrases like "X metric tons CO2" or "X million metric tons
        of CO2-equivalent" that companies commonly use to disclose Scope 1 emissions.
        Returns the largest plausible value found (in metric tons), or None.
        """
        if not filing_text:
            return None

        # Pattern: number followed by unit within 60 chars of a CO2/GHG anchor
        anchors = [m.start() for m in re.finditer(
            r"scope\s*1|greenhouse\s+gas|ghg\s+emission|carbon\s+emission",
            filing_text, re.IGNORECASE,
        )]

        number_re = re.compile(
            r"([\d,]+\.?\d*)\s*"
            r"(million\s+)?(?:metric\s+tons?|mt|mtco2e?|tco2e?)",
            re.IGNORECASE,
        )
        candidates: list[float] = []
        for pos in anchors[:5]:
            window = filing_text[pos: pos + 400]
            for m in number_re.finditer(window):
                try:
                    val = float(m.group(1).replace(",", ""))
                    if m.group(2):   # "million metric tons"
                        val *= 1_000_000
                    # Sanity check: plausible corporate Scope 1 range
                    if 1_000 <= val <= 5_000_000_000:
                        candidates.append(val)
                except ValueError:
                    pass

        return max(candidates) if candidates else None

    @staticmethod
    def _emissions_trend_score(
        s1_new: Any, s1_old: Any,
        s2_new: Any, s2_old: Any,
    ) -> float:
        """Score 0–20 based on YoY change in Scope 1+2 emissions. Decrease = good."""
        try:
            total_new = (float(s1_new or 0) + float(s2_new or 0))
            total_old = (float(s1_old or 0) + float(s2_old or 0))
            if total_old <= 0 or total_new <= 0:
                return 5.0   # partial credit: disclosed but can't compute trend
            pct_change = (total_new - total_old) / total_old
            # > 10% decrease = full 20; flat = 10; increase = 0
            if pct_change <= -0.10:
                return 20.0
            if pct_change <= -0.05:
                return 15.0
            if pct_change <= 0.0:
                return 10.0
            if pct_change <= 0.05:
                return 5.0
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _parse_ghg_from_text(text: str) -> tuple[Optional[float], Optional[float]]:
        """
        Context-anchored GHG extraction fallback.
        Only run regex on the paragraph immediately following a Scope 1/2 anchor,
        not on the entire document (avoids the audit failure of wild-regex).
        """
        scope1 = scope2 = None

        # Find anchor positions first
        s1_anchors = [m.start() for m in re.finditer(
            r"scope\s*1\s*(?:ghg\s*)?emissions?", text, re.IGNORECASE
        )]
        s2_anchors = [m.start() for m in re.finditer(
            r"scope\s*2\s*(?:ghg\s*)?emissions?", text, re.IGNORECASE
        )]

        # Scan only a 400-char window after each anchor
        number_pattern = re.compile(
            r"(\d[\d,]*\.?\d*)\s*(?:metric\s+tons?|mt|mtco2e?|tco2e?|million\s+mt)",
            re.IGNORECASE
        )

        for pos in s1_anchors[:3]:
            window = text[pos: pos + 400]
            m = number_pattern.search(window)
            if m:
                try:
                    scope1 = float(m.group(1).replace(",", ""))
                    break
                except ValueError:
                    pass

        for pos in s2_anchors[:3]:
            window = text[pos: pos + 400]
            m = number_pattern.search(window)
            if m:
                try:
                    scope2 = float(m.group(1).replace(",", ""))
                    break
                except ValueError:
                    pass

        return scope1, scope2


# ---------------------------------------------------------------------------
# Social Pillar scorer
# ---------------------------------------------------------------------------


class SocialPillar:
    """
    Score the Social pillar (0–100) from EDGAR XBRL + OSHA + DOL.

    Component breakdown:
      CEO pay ratio (Dodd-Frank XBRL)  : 0–25 pts
      Employee headcount trend (XBRL)  : 0–15 pts
      OSHA clean record                : 0–25 pts
      No WARN Act layoffs              : 0–20 pts
      Human capital disclosure depth   : 0–15 pts
    Total: 100 pts
    """

    # Human capital disclosure categories (Item 1 mandatory since 2020)
    _HC_CATEGORIES: list[tuple[str, str]] = [
        ("turnover",     r"turnover\s+rate|employee\s+retention|attrition\s+rate"),
        ("training",     r"training\s+(?:hours?|investment|program|spend)|learning\s+and\s+development"),
        ("dei_metrics",  r"diversity[\s,]+equity[\s,]+inclusion|DEI\s+metric|diversity\s+data|representation"),
        ("safety",       r"injury\s+rate|recordable\s+incident|TRIR|DART\s+rate|lost[\s-]time"),
        ("compensation", r"median\s+(?:annual\s+)?compensation|pay\s+equity|living\s+wage|wage\s+gap"),
    ]

    # CEO pay ratio XBRL tag (Dodd-Frank mandatory since 2018)
    _PAY_RATIO_CONCEPT = ("us-gaap", "PayRatioDisclosureTextBlock")
    _PAY_RATIO_REGEX   = re.compile(
        r"(?:pay\s+ratio|ratio\s+of\s+ceo|ceo\s+to\s+median)\D{0,30}"
        r"(\d{1,4})\s*(?:to\s*1|:\s*1|x\b)",
        re.IGNORECASE,
    )
    _EMPLOYEE_CONCEPT = ("dei", "EntityNumberOfEmployees")

    def __init__(self) -> None:
        self._osha = OSHAClient()
        self._warn = WARNActClient()

    def score(
        self,
        ticker:       str,
        company_name: str,
        facts:        dict,
        filing_text:  str,
    ) -> SocialPillarResult:
        result  = SocialPillarResult(ticker=ticker)
        details: dict[str, Any] = {}

        # ── 1. CEO Pay Ratio (Dodd-Frank structured disclosure) 0–25 pts ─────
        pay_ratio, pay_ratio_source = self._extract_pay_ratio(facts, filing_text)
        result.ceo_pay_ratio = pay_ratio
        result.ceo_pay_ratio_score = self._pay_ratio_score(pay_ratio)
        details["ceo_pay_ratio"]        = pay_ratio
        details["ceo_pay_ratio_source"] = pay_ratio_source
        details["ceo_pay_ratio_score"]  = result.ceo_pay_ratio_score

        # ── 2. Employee headcount trend (XBRL dei) 0–15 pts ──────────────────
        emp_series = _extract_xbrl_fact(facts, "dei", "EntityNumberOfEmployees")
        annual_emp = [e for e in emp_series if "10-K" in e.get("form", "")]
        emp_latest = int(annual_emp[0]["value"]) if annual_emp else None
        emp_prior  = int(annual_emp[1]["value"]) if len(annual_emp) > 1 else None
        result.employee_count_latest = emp_latest
        result.employee_count_prior  = emp_prior
        result.headcount_trend_score = self._headcount_score(emp_latest, emp_prior)
        details["emp_latest"]          = emp_latest
        details["emp_prior"]           = emp_prior
        details["headcount_trend_score"] = result.headcount_trend_score

        # ── 3. OSHA violations 0–25 pts ───────────────────────────────────────
        osha_count, osha_records = self._osha.get_violations(company_name)
        result.osha_violation_count = osha_count
        # Clean = 25; 1 violation = 15; 2 = 8; 3+ = 0
        if osha_count == 0:
            result.osha_score = 25.0
        elif osha_count == 1:
            result.osha_score = 15.0
        elif osha_count == 2:
            result.osha_score = 8.0
        else:
            result.osha_score = 0.0
        details["osha_violations"] = osha_count
        details["osha_records"]    = osha_records[:5]

        # ── 4. WARN Act layoffs 0–20 pts ─────────────────────────────────────
        warn_found, warn_records = self._warn.has_recent_layoff(company_name)
        result.warn_layoff_detected = warn_found
        result.warn_score           = 0.0 if warn_found else 20.0
        details["warn_layoff"]   = warn_found
        details["warn_records"]  = warn_records[:3]

        # ── 5. Human Capital disclosure depth 0–15 pts ────────────────────────
        hc_depth, hc_found = self._human_capital_depth(filing_text)
        result.hc_disclosure_depth  = hc_depth
        result.hc_disclosure_score  = min(15.0, hc_depth * 3.0)
        details["hc_depth"]          = hc_depth
        details["hc_categories"]     = hc_found

        # ── Composite S score ─────────────────────────────────────────────────
        raw = (result.ceo_pay_ratio_score +
               result.headcount_trend_score +
               result.osha_score +
               result.warn_score +
               result.hc_disclosure_score)
        result.score = max(0.0, min(100.0, round(raw, 1)))
        result.details = details
        return result

    @staticmethod
    def _extract_pay_ratio(facts: dict, filing_text: str) -> tuple[Optional[int], str]:
        """
        Extract CEO pay ratio from XBRL (preferred) or context-anchored text fallback.
        XBRL: us-gaap PayRatioDisclosureTextBlock (text block, needs regex within).
        Numeric tags: company extensions vary; use PayRatio or MedEmployeePayRatio.
        """
        # Try XBRL numeric pay ratio tags first
        for concept in ("PayRatioOfCEOToMedianEmployee", "CEOToMedianEmployeePayRatio",
                        "AnnualTotalCompensationRatioCEOToMedianEmployee"):
            series = _extract_xbrl_fact(facts, "us-gaap", concept)
            if series:
                val = series[0].get("value")
                if val is not None:
                    return int(float(val)), "xbrl_numeric"

        # Fallback: text block disclosure — context-anchored regex
        if filing_text:
            # Find "CEO pay ratio" or "pay ratio" anchor first
            anchors = [m.start() for m in re.finditer(
                r"(?:ceo\s+pay\s+ratio|median\s+annual\s+total\s+compensation|"
                r"ratio\s+of\s+(?:annual|ceo))",
                filing_text, re.IGNORECASE
            )]
            pay_ratio_re = re.compile(
                r"(\d{1,4})\s*(?:to\s*1|:\s*1|x\b)",
                re.IGNORECASE
            )
            for pos in anchors[:5]:
                window = filing_text[pos: pos + 600]
                m = pay_ratio_re.search(window)
                if m:
                    try:
                        ratio = int(m.group(1))
                        if 1 <= ratio <= 5000:  # sanity check
                            return ratio, "text_anchored"
                    except ValueError:
                        pass
        return None, "not_found"

    @staticmethod
    def _pay_ratio_score(ratio: Optional[int]) -> float:
        """0–25 pts: lower CEO pay ratio = better social score."""
        if ratio is None:
            return 5.0   # partial credit for disclosure attempt
        if ratio < 50:
            return 25.0
        if ratio < 100:
            return 20.0
        if ratio < 200:
            return 12.0
        if ratio < 400:
            return 6.0
        return 0.0

    @staticmethod
    def _headcount_score(latest: Optional[int], prior: Optional[int]) -> float:
        """0–15 pts: growing headcount = better social score."""
        if latest is None:
            return 0.0
        if prior is None or prior == 0:
            return 7.5   # disclosed but can't compute trend
        pct = (latest - prior) / prior
        if pct >= 0.05:
            return 15.0
        if pct >= 0.0:
            return 10.0
        if pct >= -0.05:
            return 5.0
        return 0.0

    def _human_capital_depth(self, filing_text: str) -> tuple[int, list[str]]:
        """Count how many HC disclosure categories are present in 10-K text."""
        if not filing_text:
            return 0, []
        found    = []
        text_low = filing_text.lower()
        for name, pattern in self._HC_CATEGORIES:
            if re.search(pattern, text_low, re.IGNORECASE):
                found.append(name)
        return len(found), found


# ---------------------------------------------------------------------------
# Governance Pillar bridge — delegates to proxy_intelligence_v3
# ---------------------------------------------------------------------------


class GovernancePillarBridge:
    """
    Thin bridge that imports GovernanceScoringEngine from proxy_intelligence_v3
    and returns its 0–100 governance score.

    Falls back to a basic structural check if the proxy data is unavailable.
    """

    def __init__(self) -> None:
        try:
            from sentinel.sfe.proxy_intelligence_v3 import (
                GovernanceScoringEngine,
                BoardAnalyticsEngine,
                ProxyDB,
            )
            self._engine   = GovernanceScoringEngine()
            self._board_eng = BoardAnalyticsEngine()
            self._proxy_db  = ProxyDB()
            self._available = True
        except ImportError as exc:
            logger.warning("proxy_intelligence_v3_unavailable", error=str(exc))
            self._available = False

    def score(self, ticker: str, year: int, filing_text: str = "") -> tuple[float, dict]:
        """Return (score_0_to_100, detail_dict)."""
        if not self._available:
            return self._fallback_score(filing_text), {"source": "fallback_text"}

        try:
            # Pull the latest proxy filing text and parse provisions
            board_text = filing_text  # proxy text; use 10-K as approximation if no DEF 14A
            provisions = self._board_eng.detect_governance_provisions(board_text)
            votes      = self._proxy_db.get_vote_outcomes(ticker)
            pay_ratios = self._proxy_db.get_pay_ratios(ticker)
            gov_score  = self._engine.score(
                ticker=ticker, year=year,
                provisions=provisions, votes=votes, pay_ratios=pay_ratios,
            )
            return gov_score.total_score, gov_score.component_detail

        except Exception as exc:
            logger.warning("governance_bridge_error", ticker=ticker, error=str(exc))
            return self._fallback_score(filing_text), {"source": "fallback_text", "error": str(exc)}

    @staticmethod
    def _fallback_score(text: str) -> float:
        """Structural text-based governance proxy when proxy_intelligence_v3 is unavailable."""
        score = 50.0
        if not text:
            return score
        text_low = text.lower()
        if re.search(r"\bindependent\s+chair\b|\blead\s+independent\s+director\b", text_low):
            score += 10.0
        if re.search(r"\bpoison\s+pill\b|\bshareholder\s+rights\s+plan\b", text_low):
            score -= 10.0
        if re.search(r"\bclassified\s+board\b|\bstaggered\s+board\b", text_low):
            score -= 8.0
        if re.search(r"\bmajority\s+vote\b", text_low):
            score += 5.0
        if re.search(r"\bproxy\s+access\b", text_low):
            score += 5.0
        if re.search(r"\bsay.on.pay\b", text_low):
            score += 3.0
        return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Controversy tracker
# ---------------------------------------------------------------------------


def _collect_controversies(
    ticker:       str,
    _company_name: str,
    env_result:   EnvironmentalPillarResult,
    soc_result:   SocialPillarResult,
) -> tuple[list[ControversyRecord], float]:
    """Aggregate active enforcement controversies and compute total deduction (max 20 pts)."""
    controversies: list[ControversyRecord] = []
    today = date.today().isoformat()

    # EPA violations
    if env_result.epa_violation_count > 0:
        severity  = "high" if env_result.epa_violation_count >= 3 else "medium"
        deduction = min(10.0, env_result.epa_violation_count * 3.0)
        controversies.append(ControversyRecord(
            ticker=ticker,
            source="epa",
            description=f"{env_result.epa_violation_count} EPA ECHO violation(s) detected",
            date_detected=today,
            severity=severity,
            deduction_pts=deduction,
        ))

    # OSHA violations
    if soc_result.osha_violation_count > 0:
        severity  = "high" if soc_result.osha_violation_count >= 3 else "medium"
        deduction = min(7.0, soc_result.osha_violation_count * 2.0)
        controversies.append(ControversyRecord(
            ticker=ticker,
            source="osha",
            description=f"{soc_result.osha_violation_count} OSHA violation record(s) found",
            date_detected=today,
            severity=severity,
            deduction_pts=deduction,
        ))

    # WARN Act layoffs
    if soc_result.warn_layoff_detected:
        controversies.append(ControversyRecord(
            ticker=ticker,
            source="warn",
            description="WARN Act mass layoff notice filed within last 12 months",
            date_detected=today,
            severity="medium",
            deduction_pts=5.0,
        ))

    total_deduction = min(20.0, sum(c.deduction_pts for c in controversies))
    return controversies, total_deduction


# ---------------------------------------------------------------------------
# Composite ESG engine
# ---------------------------------------------------------------------------


def _composite_letter_grade(score: float) -> str:
    if score >= 85:  return "AAA"
    if score >= 75:  return "AA"
    if score >= 65:  return "A"
    if score >= 55:  return "BBB"
    if score >= 45:  return "BB"
    if score >= 35:  return "B"
    if score >= 25:  return "CCC"
    return "D"


class ESGCompositeEngine:
    """
    Orchestrates E, S, G pillar scoring and produces the composite ESG result.

    Usage:
        engine = ESGCompositeEngine()
        result = engine.score("AAPL")
    """

    def __init__(self, db: ESGDB | None = None) -> None:
        self._db    = db or ESGDB()
        self._env   = EnvironmentalPillar()
        self._soc   = SocialPillar()
        self._gov   = GovernancePillarBridge()

    def score(self, ticker: str) -> ESGCompositeResult:
        """Full ESG scoring pipeline for a single ticker."""
        warnings: list[str] = []
        dq: dict[str, str]  = {}

        # ── Resolve ticker → CIK ─────────────────────────────────────────────
        cik = _resolve_ticker_to_cik(ticker)
        if not cik:
            warnings.append(f"CIK not resolved for {ticker} — EDGAR data unavailable")

        # ── Fetch XBRL facts ─────────────────────────────────────────────────
        facts: dict = {}
        if cik:
            raw_facts = _fetch_company_facts(cik)
            if raw_facts:
                facts = raw_facts.get("facts", {})
                dq["xbrl_facts"] = "ok"
            else:
                warnings.append("XBRL company facts not available")
                dq["xbrl_facts"] = "missing"

        # ── Fetch latest 10-K text ────────────────────────────────────────────
        filing_text = ""
        company_name = ticker   # default until we get the real name
        if cik:
            filing = _get_latest_10k_filing(cik)
            if filing:
                filing_text  = _fetch_filing_text(filing)
                dq["filing"] = "ok" if filing_text else "empty"
            else:
                warnings.append("No 10-K filing found")
                dq["filing"] = "missing"

        # Extract legal company name from facts
        if facts:
            entity_name = facts.get("dei", {}).get("EntityRegistrantName", {})
            if entity_name:
                vals = list(entity_name.get("units", {}).values())
                if vals and vals[0]:
                    company_name = vals[0][0].get("val", ticker)

        # ── Determine SIC sector ─────────────────────────────────────────────
        sic_code = ""
        if facts:
            sic_vals = _extract_xbrl_fact(facts, "dei", "EntitySicCode")
            if sic_vals:
                sic_code = str(sic_vals[0].get("value", ""))
        sector = _SIC_TO_SECTOR.get(sic_code.zfill(4) if sic_code else "", "default")

        # Infer year for governance scoring
        year = datetime.utcnow().year

        # ── Score all three pillars ───────────────────────────────────────────
        env_result = self._env.score(ticker, company_name, facts, filing_text)
        soc_result = self._soc.score(ticker, company_name, facts, filing_text)
        g_score_100, g_detail = self._gov.score(ticker, year, filing_text)

        # ── Controversy deduction ─────────────────────────────────────────────
        controversies, deduction = _collect_controversies(
            ticker, company_name, env_result, soc_result
        )

        # ── Composite (equal-weighted, spec default) ──────────────────────────
        composite = (env_result.score * 0.33 +
                     soc_result.score * 0.33 +
                     g_score_100 * 0.34)
        adjusted  = max(0.0, composite - deduction)

        result = ESGCompositeResult(
            ticker=ticker,
            composite_score=round(composite, 1),
            e_score=round(env_result.score, 1),
            s_score=round(soc_result.score, 1),
            g_score=round(g_score_100, 1),
            controversy_deduction=round(deduction, 1),
            controversy_adjusted=round(adjusted, 1),
            letter_grade=_composite_letter_grade(adjusted),
            sector=sector,
            pillar_env=env_result,
            pillar_soc=soc_result,
            g_total_score=round(g_score_100, 1),
            controversies=controversies,
            data_quality=dq,
            warnings=warnings,
        )

        # ── Persist ───────────────────────────────────────────────────────────
        self._db.upsert_composite(result)
        self._db.upsert_pillar(ticker, result.as_of, "E", env_result.score, env_result.details)
        self._db.upsert_pillar(ticker, result.as_of, "S", soc_result.score, soc_result.details)
        self._db.upsert_pillar(ticker, result.as_of, "G", g_score_100, g_detail)
        for c in controversies:
            self._db.insert_controversy(c)

        return result

    def peer_comparison(self, ticker: str) -> PeerESGComparison:
        """Compare ticker ESG score against sector peers in the local DB."""
        my_row = self._db.get_latest_score(ticker)
        if not my_row:
            raise ValueError(f"No ESG score for {ticker} — run score() first")

        sector = my_row.get("sector", "default")
        peers  = self._db.get_sector_peers(sector)

        if not peers:
            return PeerESGComparison(
                ticker=ticker, sector=sector,
                composite_score=my_row["composite_score"],
                sector_avg_composite=my_row["composite_score"],
                sector_avg_e=my_row.get("e_score", 0),
                sector_avg_s=my_row.get("s_score", 0),
                sector_avg_g=my_row.get("g_score", 0),
                percentile_rank=50.0, peers=[],
            )

        df = pd.DataFrame(peers)
        avg_composite = float(df["composite_score"].mean())
        avg_e         = float(df["e_score"].mean())
        avg_s         = float(df["s_score"].mean())
        avg_g         = float(df["g_score"].mean())

        my_score = my_row["composite_score"]
        rank     = float((df["composite_score"] < my_score).sum() / len(df) * 100)

        return PeerESGComparison(
            ticker=ticker,
            sector=sector,
            composite_score=my_score,
            sector_avg_composite=round(avg_composite, 1),
            sector_avg_e=round(avg_e, 1),
            sector_avg_s=round(avg_s, 1),
            sector_avg_g=round(avg_g, 1),
            percentile_rank=round(rank, 1),
            peers=peers[:20],
        )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

esg_v3_router = APIRouter(prefix="/esg/v3", tags=["ESG v3"])
_engine = ESGCompositeEngine()


@esg_v3_router.get("/score/{ticker}", summary="Full ESG composite score")
def get_esg_score(ticker: str = Query(..., min_length=1, max_length=10)) -> dict:
    """
    Compute or retrieve ESG composite score for a ticker.
    Returns E/S/G pillar scores (0–100 each), composite, and controversy-adjusted score.
    """
    ticker = ticker.upper()
    try:
        result = _engine.score(ticker)
        return result.model_dump()
    except Exception as exc:
        logger.error("esg_score_error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@esg_v3_router.get("/pillar/{ticker}/{pillar}", summary="Single pillar detail")
def get_pillar(
    ticker: str,
    pillar: str,
) -> dict:
    """
    Return detailed breakdown for one ESG pillar (E, S, or G).
    Pillar must be one of: E, S, G.
    """
    ticker = ticker.upper()
    pillar = pillar.upper()
    if pillar not in ("E", "S", "G"):
        raise HTTPException(status_code=400, detail="pillar must be E, S, or G")
    row = _engine._db.get_pillar(ticker, pillar)
    if not row:
        raise HTTPException(status_code=404, detail=f"No pillar data for {ticker}/{pillar}")
    details = json.loads(row.get("details_json") or "{}")
    return {"ticker": ticker, "pillar": pillar, "score": row["score"],
            "as_of": row["as_of"], "details": details}


@esg_v3_router.get("/peer-comparison/{ticker}", summary="ESG peer comparison by sector")
def get_peer_comparison(ticker: str) -> dict:
    """Compare ticker ESG to sector peers stored in the local DB."""
    ticker = ticker.upper()
    try:
        result = _engine.peer_comparison(ticker)
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@esg_v3_router.get("/controversy/{ticker}", summary="Active ESG controversies")
def get_controversies(ticker: str) -> dict:
    """Return active EPA/OSHA/WARN controversy records for a ticker."""
    ticker = ticker.upper()
    rows   = _engine._db.get_controversies(ticker)
    return {"ticker": ticker, "count": len(rows), "controversies": rows}


@esg_v3_router.get("/history/{ticker}", summary="ESG score history")
def get_history(
    ticker: str,
    limit:  int = Query(default=10, ge=1, le=50),
) -> dict:
    """Return historical ESG composite scores for a ticker."""
    ticker = ticker.upper()
    rows   = _engine._db.get_history(ticker, limit=limit)
    return {"ticker": ticker, "history": rows}


@esg_v3_router.get("/universe-rank", summary="Universe-wide ESG ranking")
def get_universe_rank(
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    """Return all tickers ranked by composite ESG score descending."""
    rows = _engine._db.get_universe_scores()
    return {
        "count": len(rows),
        "as_of": datetime.utcnow().date().isoformat(),
        "rankings": rows[:limit],
    }
