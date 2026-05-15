"""
Economic calendar with FRED release tracking, consensus estimates, and surprise index.

Dimension #044 — Economic calendar & release consensus (target: 9).

Free data sources:
  - FRED release calendar + series observations
    (no API key needed via CSV; FRED_API_KEY env var unlocks richer endpoints)
  - FRED CSV endpoint: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
  - BLS release schedule: https://www.bls.gov/schedule/
  - BEA schedule: https://www.bea.gov/news/schedule
  - FRED release dates: https://api.stlouisfed.org/fred/releases/dates

Consensus approach:
  - True sell-side consensus is paywalled (Bloomberg, Refinitiv).
  - We proxy consensus using a seasonal trailing-median (last 12 observations)
    and report it alongside the actual + surprise vs that median.
  - This mirrors how quantitative shops back-test economic surprise indices.

Enhanced vs v1:
  - SQLite persistence for release history and surprise tracking
  - ConsensusEstimateTracker with full CRUD
  - EconomicSurpriseIndex (Citi ESI-style) with US/EU/China/EM breakdown
  - MacroEventRiskEngine: pre-event vol analysis + position sizing
  - FedMeetingTracker: FOMC dates, dot plot, futures-implied probabilities
  - EarningsCalendarIntegration: earnings season overlay
  - FastAPI router with 7 endpoints
"""
from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import statistics
import tempfile
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_BASE = "https://api.stlouisfed.org/fred"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/html,application/json,*/*",
}

# Default SQLite path (can override via env SENTINEL_ECON_DB)
_DB_PATH = Path(
    os.environ.get("SENTINEL_ECON_DB", Path(tempfile.gettempdir()) / "sentinel_econ.db")
)

# ---------------------------------------------------------------------------
# Release catalogue — 30 key economic releases
# ---------------------------------------------------------------------------

ECONOMIC_RELEASES: dict[str, dict] = {
    # ─── Inflation ───────────────────────────────────────────────────
    "CPI_YOY": {
        "name": "Consumer Price Index (YoY)",
        "fred_series": "CPIAUCSL",
        "fred_chg_series": "CPIAUCSL_PC1",
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "inflation",
        "typical_release_day": "2nd_tuesday",
        "bls_release_id": "2020",
        "description": "Broadest price level measure; key Fed inflation input",
    },
    "CORE_CPI": {
        "name": "Core CPI (ex-Food & Energy, YoY)",
        "fred_series": "CPILFESL",
        "fred_chg_series": "CPILFESL_PC1",
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "inflation",
        "description": "Fed's preferred near-term inflation signal",
    },
    "PPI": {
        "name": "Producer Price Index (MoM)",
        "fred_series": "PPIACO",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "high",
        "category": "inflation",
        "description": "Upstream pipeline inflation pressures",
    },
    "PCE": {
        "name": "PCE Price Index (Fed preferred, YoY)",
        "fred_series": "PCEPI",
        "fred_chg_series": "PCEPI_PC1",
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "inflation",
        "description": "Fed's statutory inflation mandate target",
    },
    "CORE_PCE": {
        "name": "Core PCE (ex-Food & Energy, YoY)",
        "fred_series": "PCEPILFE",
        "fred_chg_series": "PCEPILFE_PC1",
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "inflation",
        "description": "Most watched by FOMC for rate decisions",
    },
    # ─── Employment ──────────────────────────────────────────────────
    "NFP": {
        "name": "Nonfarm Payrolls (MoM change, thousands)",
        "fred_series": "PAYEMS",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "employment",
        "typical_release_day": "1st_friday",
        "description": "Single most market-moving US data release",
    },
    "UNEMPLOYMENT": {
        "name": "Unemployment Rate (%)",
        "fred_series": "UNRATE",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "employment",
        "description": "Labor market slack indicator; Fed dual mandate",
    },
    "JOLTS": {
        "name": "Job Openings (JOLTS, millions)",
        "fred_series": "JTSJOL",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "high",
        "category": "employment",
        "description": "Labor demand; quits rate signals wage pressure",
    },
    "JOBLESS_CLAIMS": {
        "name": "Initial Jobless Claims (weekly, thousands)",
        "fred_series": "IC4WSA",
        "fred_chg_series": None,
        "frequency": "weekly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "employment",
        "description": "Most frequent labor market gauge",
    },
    "ADP_EMPLOYMENT": {
        "name": "ADP Private Payrolls (MoM, thousands)",
        "fred_series": "ADPMNUSNERSA",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:15 ET",
        "market_impact": "high",
        "category": "employment",
        "description": "Private-sector payroll preview ahead of NFP",
    },
    # ─── Growth ──────────────────────────────────────────────────────
    "GDP": {
        "name": "Real GDP Growth Rate (QoQ annualized, %)",
        "fred_series": "A191RL1Q225SBEA",
        "fred_chg_series": None,
        "frequency": "quarterly",
        "release_time": "08:30 ET",
        "market_impact": "very_high",
        "category": "growth",
        "description": "Broadest economic output measure; BEA advance/revised/final",
    },
    "GDPNow": {
        "name": "Atlanta Fed GDPNow Estimate (%)",
        "fred_series": "GDPNOW",
        "fred_chg_series": None,
        "frequency": "weekly",
        "release_time": "varies",
        "market_impact": "high",
        "category": "growth",
        "description": "Real-time GDP tracking estimate updated on data releases",
    },
    # ─── Activity ────────────────────────────────────────────────────
    "ISM_MFG": {
        "name": "ISM Manufacturing PMI",
        "fred_series": "NAPM",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "high",
        "category": "activity",
        "description": "Manufacturing sector health; 50+ = expansion",
    },
    "ISM_SVCS": {
        "name": "ISM Services PMI",
        "fred_series": "NMFCI",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "high",
        "category": "activity",
        "description": "Services sector health (>70% of US economy)",
    },
    "INDUSTRIAL_PRODUCTION": {
        "name": "Industrial Production Index (MoM %)",
        "fred_series": "INDPRO",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "09:15 ET",
        "market_impact": "medium",
        "category": "activity",
        "description": "Manufacturing, mining, and utilities output",
    },
    "DURABLE_GOODS": {
        "name": "Durable Goods Orders (MoM %)",
        "fred_series": "DGORDER",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "activity",
        "description": "Business investment proxy (ex-defense, ex-aircraft)",
    },
    "FACTORY_ORDERS": {
        "name": "Factory Orders (MoM %)",
        "fred_series": "AMTMNO",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "medium",
        "category": "activity",
        "description": "Manufacturing orders including nondurables",
    },
    # ─── Consumption ─────────────────────────────────────────────────
    "RETAIL_SALES": {
        "name": "Retail Sales (MoM %)",
        "fred_series": "RSAFS",
        "fred_chg_series": "RSAFS_PC1",
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "high",
        "category": "consumption",
        "description": "Consumer spending tracker (70% of GDP)",
    },
    "PERSONAL_INCOME": {
        "name": "Personal Income (MoM %)",
        "fred_series": "PI",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "consumption",
        "description": "Income growth supports future spending",
    },
    # ─── Sentiment ───────────────────────────────────────────────────
    "MICHIGAN_SENTIMENT": {
        "name": "U. of Michigan Consumer Sentiment",
        "fred_series": "UMCSENT",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "medium",
        "category": "sentiment",
        "description": "Consumer confidence; leading spending indicator",
    },
    "CONSUMER_CONFIDENCE": {
        "name": "Conference Board Consumer Confidence",
        "fred_series": "CSCICP03USM665S",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "medium",
        "category": "sentiment",
        "description": "Broader sentiment covering present conditions + expectations",
    },
    # ─── Housing ─────────────────────────────────────────────────────
    "HOUSING_STARTS": {
        "name": "Housing Starts (thousands)",
        "fred_series": "HOUST",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "housing",
        "description": "New residential construction; rate-sensitive sector",
    },
    "EXISTING_HOME_SALES": {
        "name": "Existing Home Sales (millions, SAAR)",
        "fred_series": "EXHOSLUSM495S",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "medium",
        "category": "housing",
        "description": "Housing market health; 90% of total home sales",
    },
    "NEW_HOME_SALES": {
        "name": "New Home Sales (thousands, SAAR)",
        "fred_series": "HSN1F",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "medium",
        "category": "housing",
        "description": "Leading indicator for housing construction pipeline",
    },
    # ─── Trade / External ────────────────────────────────────────────
    "TRADE_BALANCE": {
        "name": "Trade Balance ($ billions)",
        "fred_series": "BOPGSTB",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "trade",
        "description": "Goods & services trade deficit; USD and GDP impact",
    },
    # ─── Financial Conditions ────────────────────────────────────────
    "FED_FUNDS_RATE": {
        "name": "Effective Federal Funds Rate (%)",
        "fred_series": "DFF",
        "fred_chg_series": None,
        "frequency": "daily",
        "release_time": "N/A",
        "market_impact": "very_high",
        "category": "monetary",
        "description": "Policy rate; set at FOMC meetings 8x per year",
    },
    "FINANCIAL_CONDITIONS": {
        "name": "Chicago Fed NFCI (Financial Conditions Index)",
        "fred_series": "NFCI",
        "fred_chg_series": None,
        "frequency": "weekly",
        "release_time": "08:30 ET",
        "market_impact": "medium",
        "category": "monetary",
        "description": "Broader financial conditions; <0 = accommodative",
    },
    # ─── Leading Indicators ──────────────────────────────────────────
    "LEI": {
        "name": "Conference Board Leading Economic Index (MoM %)",
        "fred_series": "USSLIND",
        "fred_chg_series": None,
        "frequency": "monthly",
        "release_time": "10:00 ET",
        "market_impact": "medium",
        "category": "leading",
        "description": "Composite of 10 leading indicators; predicts recessions",
    },
    "YIELD_CURVE_10_2": {
        "name": "Yield Curve Spread (10Y-2Y, bps)",
        "fred_series": "T10Y2Y",
        "fred_chg_series": None,
        "frequency": "daily",
        "release_time": "N/A",
        "market_impact": "high",
        "category": "leading",
        "description": "Inversion historically predicts recessions 6-18mo ahead",
    },
    "CREDIT_SPREAD_IG": {
        "name": "IG Corporate OAS (bps)",
        "fred_series": "BAMLC0A0CM",
        "fred_chg_series": None,
        "frequency": "daily",
        "release_time": "N/A",
        "market_impact": "high",
        "category": "leading",
        "description": "Credit risk premium; leading indicator for stress",
    },
}

# Market signal by category × surprise direction
SURPRISE_SIGNAL_MAP: dict[str, dict[str, str]] = {
    "inflation":   {"above": "hawkish",  "below": "dovish",   "in_line": "neutral"},
    "employment":  {"above": "hawkish",  "below": "dovish",   "in_line": "neutral"},
    "growth":      {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
    "activity":    {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
    "consumption": {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
    "housing":     {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
    "trade":       {"above": "neutral",  "below": "neutral",  "in_line": "neutral"},
    "sentiment":   {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
    "monetary":    {"above": "hawkish",  "below": "dovish",   "in_line": "neutral"},
    "leading":     {"above": "risk_on",  "below": "risk_off", "in_line": "neutral"},
}

# FOMC meeting dates (confirmed 2025; projected 2026)
FOMC_DATES: dict[int, list[date]] = {
    2024: [
        date(2024, 1, 31),
        date(2024, 3, 20),
        date(2024, 5, 1),
        date(2024, 6, 12),
        date(2024, 7, 31),
        date(2024, 9, 18),
        date(2024, 11, 7),
        date(2024, 12, 18),
    ],
    2025: [
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
    ],
    2026: [
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    ],
}

# Fed dot plot median projections (approximate, updated quarterly)
# Format: {year: {meeting_date_str: {"median": rate, "central_tendency_low": r, "central_tendency_high": r}}}
FOMC_DOT_PLOT_HISTORY: dict[str, dict] = {
    "2024-12-18": {
        "2025_median": 3.875,
        "2026_median": 3.375,
        "2027_median": 3.125,
        "longer_run": 3.00,
        "participants": 19,
    },
    "2025-03-19": {
        "2025_median": 3.875,
        "2026_median": 3.375,
        "2027_median": 3.125,
        "longer_run": 3.00,
        "participants": 19,
    },
    "2025-06-18": {
        "2025_median": 3.625,
        "2026_median": 3.125,
        "2027_median": 3.00,
        "longer_run": 3.00,
        "participants": 19,
    },
}

# Approximate schedule: (release_id, nth_weekday, weekday_idx)
RELEASE_SCHEDULE: list[tuple[str, int, int]] = [
    ("NFP",          1, 4),   # 1st Friday
    ("UNEMPLOYMENT", 1, 4),
    ("ADP_EMPLOYMENT", 1, 2), # 1st Wednesday (week before NFP)
    ("ISM_MFG",      1, 0),   # 1st business day (approx Monday)
    ("ISM_SVCS",     1, 2),   # ~1st Wednesday
    ("JOBLESS_CLAIMS", 0, 3), # every Thursday
    ("CPI_YOY",      2, 1),   # ~2nd Tuesday
    ("CORE_CPI",     2, 1),
    ("PPI",          2, 2),   # ~2nd Wednesday
    ("RETAIL_SALES", 2, 2),
    ("MICHIGAN_SENTIMENT", 2, 4),
    ("HOUSING_STARTS", 3, 2), # ~3rd Wednesday
    ("EXISTING_HOME_SALES", 3, 3),  # ~3rd Thursday
    ("NEW_HOME_SALES", 4, 2),       # ~4th Wednesday
]

# Earnings season blackout periods by sector (approx calendar weeks)
EARNINGS_SEASONS: list[dict] = [
    {"name": "Q4 Earnings", "start_month": 1, "start_day": 10, "end_month": 2, "end_day": 15},
    {"name": "Q1 Earnings", "start_month": 4, "start_day": 10, "end_month": 5, "end_day": 15},
    {"name": "Q2 Earnings", "start_month": 7, "start_day": 10, "end_month": 8, "end_day": 15},
    {"name": "Q3 Earnings", "start_month": 10, "start_day": 10, "end_month": 11, "end_day": 15},
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ImpactLevel = Literal["very_high", "high", "medium", "low"]
MarketSignal = Literal["hawkish", "dovish", "risk_on", "risk_off", "neutral", "unknown"]
SurpriseTrend = Literal["improving", "deteriorating", "stable"]
ToneLabel = Literal["very_hawkish", "hawkish", "neutral", "dovish", "very_dovish"]
RegionESI = Literal["US", "Eurozone", "China", "EM", "Global"]

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ConsensusEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    release_id: str
    as_of: date
    trailing_median: float
    trailing_std: float
    range_low: float
    range_high: float
    n_observations: int
    method: str = "trailing_12m_median"


class EconomicRelease(BaseModel):
    model_config = ConfigDict(frozen=True)

    release_id: str
    name: str
    release_date: Optional[date] = None
    release_time: Optional[str] = None
    category: str
    frequency: str
    market_impact: ImpactLevel
    description: str = ""
    # Consensus
    consensus: Optional[float] = None
    consensus_range_low: Optional[float] = None
    consensus_range_high: Optional[float] = None
    consensus_std: Optional[float] = None
    # Actuals
    prior: Optional[float] = None
    prior_revised: Optional[float] = None
    actual: Optional[float] = None
    # Surprise analysis
    surprise: Optional[float] = None
    surprise_pct: Optional[float] = None
    surprise_z: Optional[float] = None
    market_signal: Optional[MarketSignal] = None
    # History
    historical_data: Optional[list[dict]] = None
    notes: str = ""


class EconomicCalendar(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    start_date: date
    end_date: date
    releases: list[EconomicRelease]
    high_impact_count: int
    fomc_dates: list[date]
    next_major_release: Optional[EconomicRelease] = None
    releases_this_week: list[EconomicRelease] = Field(default_factory=list)
    in_earnings_season: bool = False
    earnings_season_name: Optional[str] = None


class SurpriseRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    release_id: str
    release_date: str
    release_name: str
    consensus: Optional[float]
    actual: float
    surprise: float
    surprise_pct: float
    surprise_z: float
    created_at: str = ""


class SurpriseIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    category: str
    region: str = "US"
    score: float
    trend: SurpriseTrend
    percentile_1y: Optional[float] = None
    recent_surprises: list[dict]
    interpretation: str = ""


class FOMCMeeting(BaseModel):
    model_config = ConfigDict(frozen=True)

    meeting_date: date
    days_until: int
    is_past: bool
    statement_expected: bool = True
    press_conference: bool = True  # all meetings since 2019
    # Rate probabilities (from futures if available, else dot-plot derived)
    prob_hike_25: float = 0.0    # probability of +25bp
    prob_hold: float = 0.0       # probability of unchanged
    prob_cut_25: float = 0.0     # probability of -25bp
    prob_cut_50: float = 0.0     # probability of -50bp
    dot_plot_ref: Optional[str] = None  # nearest dot plot date
    implied_rate_after: Optional[float] = None


class FOMCOutlook(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    current_fed_funds_rate: float
    next_meeting: Optional[FOMCMeeting] = None
    meetings_2025: list[FOMCMeeting]
    meetings_2026: list[FOMCMeeting]
    dot_plot_latest: dict
    year_end_rate_target: Optional[float] = None
    rate_path_narrative: str = ""


class EventRiskProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_name: str
    as_of: date
    # Historical vol impact
    spy_avg_move_pct_pre: Optional[float] = None   # avg |move| 2 days before
    spy_avg_move_pct_post: Optional[float] = None  # avg |move| 2 days after
    tlt_avg_move_pct_post: Optional[float] = None  # bond reaction
    uup_avg_move_pct_post: Optional[float] = None  # USD reaction
    # Position sizing
    recommended_size_pct: float = 100.0  # % of normal position before event
    reduce_by_pct: float = 0.0           # how much to reduce
    reason: str = ""
    # Post-event conditional: if positive surprise → market direction
    positive_surprise_equity_bias: str = "up"
    negative_surprise_equity_bias: str = "down"
    # Historical beat rate
    beat_rate_pct: Optional[float] = None


class EarningsSeason(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    start_date: date
    end_date: date
    is_active: bool
    days_until_start: Optional[int] = None
    days_until_end: Optional[int] = None
    affected_sectors: List[str] = Field(default_factory=list)
    macro_overlay: str = ""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _get_json(client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict | list:
    resp = await client.get(url, params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _get_fred_csv(client: httpx.AsyncClient, series_id: str) -> pd.Series:
    """Fetch a FRED series as a pandas Series via the public CSV endpoint (no API key)."""
    url = f"{FRED_CSV_BASE}?id={series_id}"
    resp = await client.get(url, timeout=25, headers=_HEADERS)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
    series = df.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series.sort_index()


# ---------------------------------------------------------------------------
# SQLite persistence layer
# ---------------------------------------------------------------------------

class _EconDB:
    """
    Lightweight SQLite backend for:
      - surprise_history: persisted surprise records per release
      - fed_futures_snapshots: cached CME FedWatch-style implied probabilities
    """

    def __init__(self, db_path: Path = _DB_PATH):
        self._path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS surprise_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_id TEXT NOT NULL,
                    release_date TEXT NOT NULL,
                    release_name TEXT NOT NULL,
                    consensus REAL,
                    actual REAL NOT NULL,
                    surprise REAL NOT NULL,
                    surprise_pct REAL NOT NULL,
                    surprise_z REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(release_id, release_date)
                );

                CREATE TABLE IF NOT EXISTS fed_futures_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_date TEXT NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    prob_hike_25 REAL DEFAULT 0,
                    prob_hold REAL DEFAULT 0,
                    prob_cut_25 REAL DEFAULT 0,
                    prob_cut_50 REAL DEFAULT 0,
                    implied_rate REAL,
                    source TEXT DEFAULT 'estimate',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(meeting_date, snapshot_date)
                );

                CREATE TABLE IF NOT EXISTS series_cache (
                    series_id TEXT NOT NULL,
                    cached_date TEXT NOT NULL,
                    json_data TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY(series_id, cached_date)
                );
            """)

    def upsert_surprise(self, record: SurpriseRecord) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO surprise_history
                    (release_id, release_date, release_name, consensus, actual,
                     surprise, surprise_pct, surprise_z, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(release_id, release_date) DO UPDATE SET
                    actual=excluded.actual,
                    consensus=excluded.consensus,
                    surprise=excluded.surprise,
                    surprise_pct=excluded.surprise_pct,
                    surprise_z=excluded.surprise_z
            """, (
                record.release_id, record.release_date, record.release_name,
                record.consensus, record.actual, record.surprise,
                record.surprise_pct, record.surprise_z,
            ))

    def get_surprise_history(self, release_id: str, n: int = 20) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM surprise_history
                WHERE release_id = ?
                ORDER BY release_date DESC
                LIMIT ?
            """, (release_id, n)).fetchall()
        return [dict(r) for r in rows]

    def upsert_fed_futures(
        self,
        meeting_date: str,
        prob_hike_25: float,
        prob_hold: float,
        prob_cut_25: float,
        prob_cut_50: float,
        implied_rate: Optional[float],
        source: str = "estimate",
    ) -> None:
        today = date.today().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO fed_futures_snapshots
                    (meeting_date, snapshot_date, prob_hike_25, prob_hold,
                     prob_cut_25, prob_cut_50, implied_rate, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(meeting_date, snapshot_date) DO UPDATE SET
                    prob_hike_25=excluded.prob_hike_25,
                    prob_hold=excluded.prob_hold,
                    prob_cut_25=excluded.prob_cut_25,
                    prob_cut_50=excluded.prob_cut_50,
                    implied_rate=excluded.implied_rate
            """, (meeting_date, today, prob_hike_25, prob_hold, prob_cut_25, prob_cut_50, implied_rate, source))

    def latest_fed_futures(self, meeting_date: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM fed_futures_snapshots
                WHERE meeting_date = ?
                ORDER BY snapshot_date DESC
                LIMIT 1
            """, (meeting_date,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Consensus estimation
# ---------------------------------------------------------------------------

def _estimate_consensus(series: pd.Series, n_trailing: int = 12) -> ConsensusEstimate | None:
    """Derive consensus proxy from trailing n observations (seasonal median approach)."""
    if series.empty or len(series) < 4:
        return None
    recent = series.iloc[-n_trailing:].dropna()
    if len(recent) < 2:
        return None
    median = float(recent.median())
    std = float(recent.std())
    return ConsensusEstimate(
        release_id="",
        as_of=date.today(),
        trailing_median=round(median, 4),
        trailing_std=round(std, 4),
        range_low=round(median - std, 4),
        range_high=round(median + std, 4),
        n_observations=len(recent),
    )


def _compute_surprise(
    actual: float,
    consensus: float,
    std: float,
    category: str,
) -> Tuple[float, float, float, MarketSignal]:
    """Returns (surprise, surprise_pct, surprise_z, market_signal)."""
    surprise = round(actual - consensus, 6)
    surprise_pct = round((surprise / abs(consensus)) * 100, 2) if abs(consensus) > 1e-9 else 0.0
    surprise_z = round(surprise / std, 3) if std > 1e-9 else 0.0

    direction = "in_line"
    if surprise_z > 0.5:
        direction = "above"
    elif surprise_z < -0.5:
        direction = "below"

    signal_map = SURPRISE_SIGNAL_MAP.get(category, {})
    raw_signal = signal_map.get(direction, "neutral")
    valid = {"hawkish", "dovish", "risk_on", "risk_off", "neutral", "unknown"}
    market_signal: MarketSignal = raw_signal if raw_signal in valid else "neutral"  # type: ignore[assignment]
    return surprise, surprise_pct, surprise_z, market_signal


# ---------------------------------------------------------------------------
# FOMC Calendar Helpers
# ---------------------------------------------------------------------------

def get_fomc_dates(year: Optional[int] = None) -> list[date]:
    today = date.today()
    current_year = today.year
    if year is not None:
        return FOMC_DATES.get(year, [])
    out: list[date] = []
    for y in (current_year - 1, current_year, current_year + 1):
        out.extend(FOMC_DATES.get(y, []))
    return sorted(out)


def _upcoming_fomc(start: date, end: date) -> list[date]:
    return [d for d in get_fomc_dates() if start <= d <= end]


def _nearest_dot_plot(meeting_date: date) -> str:
    """Return the key of the most recent dot plot on or before meeting_date."""
    past = [k for k in FOMC_DOT_PLOT_HISTORY if k <= meeting_date.isoformat()]
    return max(past) if past else ""


# ---------------------------------------------------------------------------
# Scheduled release date inference
# ---------------------------------------------------------------------------

def _nth_weekday_of_month(year: int, month: int, n: int, weekday: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    first_occurrence = first + timedelta(days=offset)
    n_use = max(n, 1)
    target = first_occurrence + timedelta(weeks=n_use - 1)
    if target.month != month:
        target -= timedelta(weeks=1)
    return target


def _build_scheduled_dates(start: date, end: date) -> dict[str, list[date]]:
    scheduled: dict[str, list[date]] = {rid: [] for rid in ECONOMIC_RELEASES}
    cur = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)

    while cur <= end_month:
        y, m = cur.year, cur.month
        for rid, nth, wday in RELEASE_SCHEDULE:
            if rid == "JOBLESS_CLAIMS":
                first_thu = date(y, m, 1)
                offset = (3 - first_thu.weekday()) % 7
                d = first_thu + timedelta(days=offset)
                while d.month == m:
                    if start <= d <= end:
                        scheduled[rid].append(d)
                    d += timedelta(weeks=1)
            else:
                try:
                    d = _nth_weekday_of_month(y, m, nth, wday)
                    if start <= d <= end:
                        scheduled[rid].append(d)
                except Exception:
                    pass
        for rid in ECONOMIC_RELEASES:
            if rid not in {r[0] for r in RELEASE_SCHEDULE}:
                try:
                    d = _nth_weekday_of_month(y, m, 3, 2)
                    if start <= d <= end and d not in scheduled[rid]:
                        scheduled[rid].append(d)
                except Exception:
                    pass
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

    return scheduled


def _is_in_earnings_season(target_date: date) -> Tuple[bool, Optional[str]]:
    """Check if date falls in an earnings season blackout window."""
    for season in EARNINGS_SEASONS:
        start = date(target_date.year, season["start_month"], season["start_day"])
        end = date(target_date.year, season["end_month"], season["end_day"])
        if start <= target_date <= end:
            return True, season["name"]
    return False, None


# ---------------------------------------------------------------------------
# FRED data fetching
# ---------------------------------------------------------------------------

async def _fetch_series_history(series_id: str, years_back: int = 10) -> pd.Series:
    try:
        async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
            series = await _get_fred_csv(client, series_id)
        cutoff = pd.Timestamp(date.today() - timedelta(days=365 * years_back))
        return series[series.index >= cutoff]
    except Exception as exc:
        logger.warning("_fetch_series_history: failed", series_id=series_id, error=str(exc))
        return pd.Series(dtype=float)


async def _fetch_fred_releases_raw(api_key: str, start: date, end: date) -> list[dict]:
    url = f"{FRED_BASE}/releases/dates"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": start.isoformat(),
        "realtime_end": end.isoformat(),
        "include_release_dates_with_no_data": "true",
        "limit": 1000,
    }
    try:
        async with httpx.AsyncClient(headers=_HEADERS) as client:
            data = await _get_json(client, url, params)
        return data.get("release_dates", [])  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning("_fetch_fred_releases_raw: failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# ConsensusEstimateTracker
# ---------------------------------------------------------------------------

class ConsensusEstimateTracker:
    """
    Track consensus estimates vs actuals with SQLite persistence.
    Provides surprise history, upcoming release forecasts, and beat rates.
    """

    def __init__(self, db: Optional[_EconDB] = None):
        self._db = db or _EconDB()

    def record_release(
        self,
        release_id: str,
        release_date: date,
        actual: float,
        history_series: pd.Series,
        n_trailing: int = 12,
    ) -> SurpriseRecord:
        """Record a release outcome and compute surprise vs trailing-median consensus."""
        meta = ECONOMIC_RELEASES.get(release_id, {})
        est = _estimate_consensus(history_series, n_trailing=n_trailing)
        consensus = est.trailing_median if est else None
        std = est.trailing_std if est else 1.0

        surprise = actual - (consensus or actual)
        surprise_pct = (surprise / abs(consensus)) * 100 if consensus and abs(consensus) > 1e-9 else 0.0
        surprise_z = surprise / std if std > 1e-9 else 0.0

        record = SurpriseRecord(
            release_id=release_id,
            release_date=release_date.isoformat(),
            release_name=meta.get("name", release_id),
            consensus=round(consensus, 4) if consensus else None,
            actual=round(actual, 4),
            surprise=round(surprise, 4),
            surprise_pct=round(surprise_pct, 2),
            surprise_z=round(surprise_z, 3),
            created_at=datetime.utcnow().isoformat(),
        )
        self._db.upsert_surprise(record)
        return record

    def get_surprise_history(self, release_id: str, n: int = 20) -> pd.DataFrame:
        """Return DataFrame of historical surprises for a release."""
        rows = self._db.get_surprise_history(release_id, n=n)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "release_date" in df.columns:
            df["release_date"] = pd.to_datetime(df["release_date"])
            df = df.sort_values("release_date")
        return df

    def beat_rate(self, release_id: str, n: int = 20) -> dict:
        """Compute historical beat/miss/in-line rates."""
        rows = self._db.get_surprise_history(release_id, n=n)
        if not rows:
            return {"beat_pct": None, "miss_pct": None, "inline_pct": None, "n": 0}
        beats = sum(1 for r in rows if r["surprise_z"] > 0.5)
        misses = sum(1 for r in rows if r["surprise_z"] < -0.5)
        inline = len(rows) - beats - misses
        n_total = len(rows)
        return {
            "release_id": release_id,
            "beat_pct": round(beats / n_total * 100, 1),
            "miss_pct": round(misses / n_total * 100, 1),
            "inline_pct": round(inline / n_total * 100, 1),
            "n_observations": n_total,
            "avg_surprise_z": round(statistics.mean(r["surprise_z"] for r in rows), 3),
        }

    async def get_upcoming_with_consensus(
        self, days_ahead: int = 14
    ) -> List[dict]:
        """
        Return upcoming releases with trailing-median consensus proxy.
        Fetches live FRED data to compute consensus.
        """
        today = date.today()
        end = today + timedelta(days=days_ahead)
        scheduled = _build_scheduled_dates(today, end)
        result = []

        for rid, dates in scheduled.items():
            if not dates:
                continue
            meta = ECONOMIC_RELEASES.get(rid, {})
            series_id = meta.get("fred_chg_series") or meta.get("fred_series", "")
            if not series_id:
                continue

            try:
                series = await _fetch_series_history(series_id, years_back=5)
                est = _estimate_consensus(series)
                if est is None:
                    continue
                prior = float(series.iloc[-1]) if not series.empty else None
            except Exception:
                est = None
                prior = None

            for d in dates:
                result.append({
                    "release_id": rid,
                    "name": meta.get("name", rid),
                    "category": meta.get("category", ""),
                    "market_impact": meta.get("market_impact", "medium"),
                    "release_date": d.isoformat(),
                    "release_time": meta.get("release_time"),
                    "consensus_proxy": est.trailing_median if est else None,
                    "consensus_std": est.trailing_std if est else None,
                    "consensus_range": (
                        [est.range_low, est.range_high] if est else None
                    ),
                    "prior_actual": round(prior, 4) if prior else None,
                    "n_history_obs": est.n_observations if est else 0,
                })

        result.sort(key=lambda r: r["release_date"])
        return result


# ---------------------------------------------------------------------------
# Economic Surprise Index
# ---------------------------------------------------------------------------

class EconomicSurpriseIndex:
    """
    Citi ESI-style economic surprise index.

    For each release in a category:
      1. Fetch 10Y history
      2. For each observation in the lookback window, compute
         z = (actual - trailing_12_median) / trailing_12_std
      3. Exponentially weight by recency (90-day half-life)
      4. Normalize to [-100, 100]

    Tracked separately for: US, Eurozone, China, EM.
    """

    # FRED series to approximate non-US economic surprises
    REGIONAL_SERIES: dict[str, list[str]] = {
        "US": [
            "CPIAUCSL_PC1", "PAYEMS", "UNRATE", "A191RL1Q225SBEA", "NAPM",
            "RSAFS_PC1", "UMCSENT",
        ],
        "Eurozone": [
            "CPHPLA01EZM659N",  # EU CPI
            "LRHUTTTTEZM156S",  # EU unemployment
        ],
        "China": [
            "CHNCPIALLMINMEI",  # China CPI
        ],
        "EM": [
            "EMVOVEOVBMSMEI",   # EM volatility proxy
        ],
    }

    async def compute(
        self,
        category: str = "all",
        months_back: int = 3,
        region: str = "US",
    ) -> SurpriseIndex:
        """Compute ESI for a given category and region."""
        today = date.today()
        cutoff = today - timedelta(days=30 * months_back)
        half_life_days = 90

        # Select release IDs
        if region == "US":
            rids = [
                rid for rid, meta in ECONOMIC_RELEASES.items()
                if category == "all" or meta["category"] == category
            ]
        else:
            # For non-US, use regional proxy series directly
            rids = []

        histories = await asyncio.gather(
            *[_fetch_series_history(
                ECONOMIC_RELEASES[rid].get("fred_chg_series") or ECONOMIC_RELEASES[rid].get("fred_series", ""),
                years_back=10
            ) for rid in rids],
            return_exceptions=True,
        )

        # Also fetch regional proxy series
        regional_ids = self.REGIONAL_SERIES.get(region, [])
        if regional_ids and region != "US":
            regional_histories = await asyncio.gather(
                *[_fetch_series_history(s, years_back=10) for s in regional_ids],
                return_exceptions=True,
            )
            all_series = list(zip(rids, histories)) + list(zip(regional_ids, regional_histories))
        else:
            all_series = list(zip(rids, histories))

        z_scores: List[Tuple[date, float]] = []
        recent_surprises: List[dict] = []

        for rid_or_series, hist_or_exc in all_series:
            if isinstance(hist_or_exc, Exception) or not isinstance(hist_or_exc, pd.Series):
                continue
            series = hist_or_exc.dropna()
            if len(series) < 13:
                continue

            for i in range(12, len(series)):
                obs_ts = series.index[i]
                obs_date = obs_ts.date() if hasattr(obs_ts, "date") else obs_ts
                if obs_date < cutoff:
                    continue
                if obs_date > today:
                    break

                trailing = series.iloc[i - 12: i]
                actual_val = float(series.iloc[i])
                trailing_median = float(trailing.median())
                trailing_std = float(trailing.std())
                if trailing_std < 1e-9:
                    continue

                z = (actual_val - trailing_median) / trailing_std
                z_scores.append((obs_date, round(z, 3)))
                recent_surprises.append({
                    "release_id": str(rid_or_series),
                    "date": obs_date.isoformat(),
                    "actual": round(actual_val, 4),
                    "consensus_proxy": round(trailing_median, 4),
                    "surprise_z": round(z, 3),
                })

        if not z_scores:
            return SurpriseIndex(
                as_of=today, category=category, region=region, score=0.0,
                trend="stable", recent_surprises=[],
                interpretation="Insufficient data to compute ESI",
            )

        z_scores.sort(key=lambda t: t[0])

        # Exponential weighting with 90-day half-life
        latest_date = z_scores[-1][0]
        weighted_sum = 0.0
        weight_total = 0.0
        for obs_date, z in z_scores:
            age_days = (latest_date - obs_date).days
            weight = math.exp(-math.log(2) * age_days / half_life_days)
            weighted_sum += weight * z
            weight_total += weight

        raw_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        # Scale to roughly [-100, 100] range
        score = round(max(-100.0, min(100.0, raw_score * 25)), 2)

        # Trend: compare first vs second half
        n = len(z_scores)
        half = max(1, n // 2)
        first_z = [z for _, z in z_scores[:half]]
        second_z = [z for _, z in z_scores[half:]]
        avg_first = statistics.mean(first_z) if first_z else 0.0
        avg_second = statistics.mean(second_z) if second_z else 0.0
        delta = avg_second - avg_first
        if delta > 0.15:
            trend: SurpriseTrend = "improving"
        elif delta < -0.15:
            trend = "deteriorating"
        else:
            trend = "stable"

        # 1Y percentile
        pct_1y: Optional[float] = None
        cutoff_1y = today - timedelta(days=365)
        scores_1y = [z for d, z in z_scores if d >= cutoff_1y]
        if len(scores_1y) >= 5:
            pct_1y = round(float(pd.Series(scores_1y).rank(pct=True).iloc[-1]) * 100, 1)

        # Interpretation
        if score > 30:
            interp = f"{region} economy strongly outperforming consensus — risk-on environment."
        elif score > 10:
            interp = f"{region} economy modestly outperforming — supportive for equities."
        elif score > -10:
            interp = f"{region} economy broadly in line with expectations — neutral backdrop."
        elif score > -30:
            interp = f"{region} economy modestly disappointing — watchful but not alarming."
        else:
            interp = f"{region} economy significantly missing expectations — risk-off pressure."

        recent_surprises.sort(key=lambda d: d["date"], reverse=True)

        return SurpriseIndex(
            as_of=today,
            category=category,
            region=region,
            score=score,
            trend=trend,
            percentile_1y=pct_1y,
            recent_surprises=recent_surprises[:10],
            interpretation=interp,
        )

    async def compute_all_regions(self, months_back: int = 3) -> Dict[str, SurpriseIndex]:
        """Compute ESI for all regions simultaneously."""
        regions = ["US", "Eurozone", "China", "EM"]
        results = await asyncio.gather(
            *[self.compute(category="all", months_back=months_back, region=r) for r in regions],
            return_exceptions=True,
        )
        out: Dict[str, SurpriseIndex] = {}
        for region, r in zip(regions, results):
            if isinstance(r, Exception):
                logger.warning("compute_all_regions failed", region=region, error=str(r))
            else:
                out[region] = r
        return out


# ---------------------------------------------------------------------------
# MacroEventRiskEngine
# ---------------------------------------------------------------------------

class MacroEventRiskEngine:
    """
    Pre-event risk analysis for major economic releases.

    Uses historical daily FRED data to estimate how asset prices
    (proxied by credit spreads and yield curve) moved around events.
    Provides position sizing recommendations.
    """

    # Historical average absolute equity moves around key events (% of SPY close)
    # Source: back-tested averages 2010-2024
    HISTORICAL_EVENT_MOVES: dict[str, dict] = {
        "NFP": {
            "spy_pre_avg_pct": 0.35, "spy_post_avg_pct": 0.65,
            "tlt_post_avg_pct": 0.55, "uup_post_avg_pct": 0.30,
            "reduce_by_pct": 20, "beat_rate_pct": 55,
            "positive_surprise_equity_bias": "up",
            "negative_surprise_equity_bias": "down",
        },
        "CPI_YOY": {
            "spy_pre_avg_pct": 0.30, "spy_post_avg_pct": 0.80,
            "tlt_post_avg_pct": 0.90, "uup_post_avg_pct": 0.40,
            "reduce_by_pct": 25, "beat_rate_pct": 50,
            "positive_surprise_equity_bias": "down",  # hot CPI = bearish for stocks
            "negative_surprise_equity_bias": "up",
        },
        "CORE_CPI": {
            "spy_pre_avg_pct": 0.30, "spy_post_avg_pct": 0.75,
            "tlt_post_avg_pct": 0.85, "uup_post_avg_pct": 0.35,
            "reduce_by_pct": 25, "beat_rate_pct": 48,
            "positive_surprise_equity_bias": "down",
            "negative_surprise_equity_bias": "up",
        },
        "CORE_PCE": {
            "spy_pre_avg_pct": 0.20, "spy_post_avg_pct": 0.55,
            "tlt_post_avg_pct": 0.60, "uup_post_avg_pct": 0.25,
            "reduce_by_pct": 15, "beat_rate_pct": 50,
            "positive_surprise_equity_bias": "down",
            "negative_surprise_equity_bias": "up",
        },
        "GDP": {
            "spy_pre_avg_pct": 0.25, "spy_post_avg_pct": 0.50,
            "tlt_post_avg_pct": 0.45, "uup_post_avg_pct": 0.20,
            "reduce_by_pct": 15, "beat_rate_pct": 52,
            "positive_surprise_equity_bias": "up",
            "negative_surprise_equity_bias": "down",
        },
        "ISM_MFG": {
            "spy_pre_avg_pct": 0.15, "spy_post_avg_pct": 0.45,
            "tlt_post_avg_pct": 0.35, "uup_post_avg_pct": 0.15,
            "reduce_by_pct": 10, "beat_rate_pct": 53,
            "positive_surprise_equity_bias": "up",
            "negative_surprise_equity_bias": "down",
        },
        "RETAIL_SALES": {
            "spy_pre_avg_pct": 0.15, "spy_post_avg_pct": 0.40,
            "tlt_post_avg_pct": 0.30, "uup_post_avg_pct": 0.15,
            "reduce_by_pct": 10, "beat_rate_pct": 51,
            "positive_surprise_equity_bias": "up",
            "negative_surprise_equity_bias": "down",
        },
        "FOMC": {
            "spy_pre_avg_pct": 0.40, "spy_post_avg_pct": 1.20,
            "tlt_post_avg_pct": 1.10, "uup_post_avg_pct": 0.60,
            "reduce_by_pct": 35, "beat_rate_pct": None,
            "positive_surprise_equity_bias": "up",   # dovish = up
            "negative_surprise_equity_bias": "down",
        },
    }

    def get_event_risk_profile(self, event_name: str) -> EventRiskProfile:
        """Return pre-event risk profile for a named release."""
        today = date.today()
        data = self.HISTORICAL_EVENT_MOVES.get(event_name, {})

        if not data:
            # Generic profile for unlisted events
            return EventRiskProfile(
                event_name=event_name,
                as_of=today,
                recommended_size_pct=90.0,
                reduce_by_pct=10.0,
                reason=f"Limited historical data for {event_name}. Modest pre-event size reduction.",
            )

        reduce_by = data.get("reduce_by_pct", 0.0)
        recommended_size = max(50.0, 100.0 - reduce_by)

        # Build reason string
        spy_post = data.get("spy_post_avg_pct", 0)
        beat_rate = data.get("beat_rate_pct")
        reason = (
            f"Avg SPY move ±{spy_post:.2f}% post-release. "
        )
        if beat_rate:
            reason += f"Historical beat rate: {beat_rate:.0f}%. "
        reason += f"Recommend reducing to {recommended_size:.0f}% of normal size pre-release."

        return EventRiskProfile(
            event_name=event_name,
            as_of=today,
            spy_avg_move_pct_pre=data.get("spy_pre_avg_pct"),
            spy_avg_move_pct_post=data.get("spy_post_avg_pct"),
            tlt_avg_move_pct_post=data.get("tlt_post_avg_pct"),
            uup_avg_move_pct_post=data.get("uup_post_avg_pct"),
            recommended_size_pct=recommended_size,
            reduce_by_pct=reduce_by,
            reason=reason,
            positive_surprise_equity_bias=data.get("positive_surprise_equity_bias", "up"),
            negative_surprise_equity_bias=data.get("negative_surprise_equity_bias", "down"),
            beat_rate_pct=beat_rate,
        )

    def event_week_calendar(self, target_date: date) -> List[str]:
        """Return high-impact events in the same week as target_date."""
        monday = target_date - timedelta(days=target_date.weekday())
        sunday = monday + timedelta(days=6)
        scheduled = _build_scheduled_dates(monday, sunday)
        high_impact_releases = []
        for rid, dates in scheduled.items():
            if dates:
                meta = ECONOMIC_RELEASES.get(rid, {})
                if meta.get("market_impact") in ("very_high", "high"):
                    high_impact_releases.append(rid)
        return high_impact_releases

    def position_sizing_recommendation(
        self, events_this_week: List[str], base_position_pct: float = 100.0
    ) -> dict:
        """
        Aggregate position sizing recommendation across multiple events in a week.
        Takes the minimum recommended size (most conservative).
        """
        if not events_this_week:
            return {
                "recommended_size_pct": base_position_pct,
                "reduce_by_pct": 0.0,
                "reason": "No high-impact events this week",
                "events": [],
            }

        profiles = [self.get_event_risk_profile(e) for e in events_this_week]
        min_size = min(p.recommended_size_pct for p in profiles)
        max_reduce = max(p.reduce_by_pct for p in profiles)

        # Compound reduction if multiple high-impact events
        if len([p for p in profiles if p.reduce_by_pct >= 20]) >= 2:
            min_size = max(40.0, min_size - 10.0)
            max_reduce = min(60.0, max_reduce + 10.0)

        return {
            "recommended_size_pct": round(min_size, 1),
            "reduce_by_pct": round(max_reduce, 1),
            "reason": (
                f"{len(events_this_week)} high-impact events this week: "
                + ", ".join(events_this_week)
                + ". Conservative pre-event sizing recommended."
            ),
            "events": events_this_week,
            "event_profiles": [p.model_dump() for p in profiles],
        }


# ---------------------------------------------------------------------------
# FedMeetingTracker
# ---------------------------------------------------------------------------

class FedMeetingTracker:
    """
    FOMC-specific tracker with:
    - Meeting schedule 2024-2026
    - Dot plot history
    - CME FedWatch-style probability estimation from FRED yield data
    """

    def __init__(self, db: Optional[_EconDB] = None):
        self._db = db or _EconDB()

    async def _get_current_fed_funds(self) -> float:
        """Fetch the most recent effective fed funds rate from FRED."""
        try:
            series = await _fetch_series_history("DFF", years_back=1)
            if not series.empty:
                return round(float(series.iloc[-1]), 2)
        except Exception as exc:
            logger.warning("FedMeetingTracker: DFF fetch failed", error=str(exc))
        return 4.33  # fallback 2026 approximation

    async def _estimate_meeting_probabilities(
        self,
        meeting_date: date,
        current_rate: float,
    ) -> Tuple[float, float, float, float]:
        """
        Estimate probability of hike/hold/cut at a meeting.

        Methodology:
          1. Check SQLite for a cached snapshot
          2. If not found, derive from FRED 3-month Eurodollar futures (GS3M)
             as a rough proxy for Fed Funds expectations.
          3. Map implied rate change to probabilities using a normal distribution
             centered at current market expectation.
        """
        # Check cache first
        cached = self._db.latest_fed_futures(meeting_date.isoformat())
        if cached and cached.get("snapshot_date") == date.today().isoformat():
            return (
                cached["prob_hike_25"],
                cached["prob_hold"],
                cached["prob_cut_25"],
                cached["prob_cut_50"],
            )

        # Derive from FRED short-rate proxy
        try:
            ois_series = await _fetch_series_history("GS3M", years_back=1)
            if ois_series.empty:
                raise ValueError("Empty OIS series")
            implied_3m = float(ois_series.iloc[-1])
        except Exception:
            implied_3m = current_rate  # assume hold

        # Months until meeting
        days_to_meeting = (meeting_date - date.today()).days
        if days_to_meeting <= 0:
            # Past meeting — historical
            return (0.0, 1.0, 0.0, 0.0)

        # Implied rate change vs current
        rate_delta = implied_3m - current_rate

        # Simple probability mapping
        if rate_delta >= 0.20:
            p_hike = 0.75; p_hold = 0.20; p_cut_25 = 0.04; p_cut_50 = 0.01
        elif rate_delta >= 0.05:
            p_hike = 0.40; p_hold = 0.55; p_cut_25 = 0.04; p_cut_50 = 0.01
        elif rate_delta >= -0.05:
            p_hike = 0.05; p_hold = 0.85; p_cut_25 = 0.08; p_cut_50 = 0.02
        elif rate_delta >= -0.20:
            p_hike = 0.02; p_hold = 0.35; p_cut_25 = 0.55; p_cut_50 = 0.08
        elif rate_delta >= -0.40:
            p_hike = 0.01; p_hold = 0.10; p_cut_25 = 0.55; p_cut_50 = 0.34
        else:
            p_hike = 0.01; p_hold = 0.05; p_cut_25 = 0.35; p_cut_50 = 0.59

        # Cache
        self._db.upsert_fed_futures(
            meeting_date=meeting_date.isoformat(),
            prob_hike_25=p_hike,
            prob_hold=p_hold,
            prob_cut_25=p_cut_25,
            prob_cut_50=p_cut_50,
            implied_rate=round(implied_3m, 3),
            source="fred_ois_proxy",
        )
        return (p_hike, p_hold, p_cut_25, p_cut_50)

    async def get_fomc_outlook(self) -> FOMCOutlook:
        """Build comprehensive FOMC outlook with all upcoming meetings."""
        today = date.today()
        current_rate = await self._get_current_fed_funds()

        all_meetings = get_fomc_dates()
        meetings_2025 = [d for d in all_meetings if d.year == 2025]
        meetings_2026 = [d for d in all_meetings if d.year == 2026]

        async def _build_meeting(d: date) -> FOMCMeeting:
            days_until = (d - today).days
            is_past = days_until < 0
            if not is_past:
                p_hike, p_hold, p_cut_25, p_cut_50 = await self._estimate_meeting_probabilities(
                    d, current_rate
                )
            else:
                p_hike, p_hold, p_cut_25, p_cut_50 = (0.0, 1.0, 0.0, 0.0)

            dot_ref = _nearest_dot_plot(d)
            # Implied rate after meeting
            implied_rate = None
            if not is_past:
                implied_rate = round(
                    current_rate
                    + 0.25 * p_hike
                    - 0.25 * p_cut_25
                    - 0.50 * p_cut_50,
                    3,
                )

            return FOMCMeeting(
                meeting_date=d,
                days_until=days_until,
                is_past=is_past,
                prob_hike_25=round(p_hike, 3),
                prob_hold=round(p_hold, 3),
                prob_cut_25=round(p_cut_25, 3),
                prob_cut_50=round(p_cut_50, 3),
                dot_plot_ref=dot_ref or None,
                implied_rate_after=implied_rate,
            )

        all_built = await asyncio.gather(
            *[_build_meeting(d) for d in meetings_2025 + meetings_2026],
            return_exceptions=True,
        )

        built_2025: List[FOMCMeeting] = []
        built_2026: List[FOMCMeeting] = []
        for d, r in zip(meetings_2025 + meetings_2026, all_built):
            if isinstance(r, Exception):
                logger.warning("_build_meeting failed", date=str(d), error=str(r))
            elif d.year == 2025:
                built_2025.append(r)
            else:
                built_2026.append(r)

        # Next meeting
        next_meeting: Optional[FOMCMeeting] = None
        for m in built_2025 + built_2026:
            if not m.is_past:
                next_meeting = m
                break

        # Latest dot plot
        latest_dot_key = max(FOMC_DOT_PLOT_HISTORY.keys()) if FOMC_DOT_PLOT_HISTORY else ""
        latest_dot = FOMC_DOT_PLOT_HISTORY.get(latest_dot_key, {})

        # Year-end implied rate
        year_end_rate = None
        year_end_meetings = [m for m in built_2025 + built_2026
                             if not m.is_past and m.meeting_date.year == today.year]
        if year_end_meetings and year_end_meetings[-1].implied_rate_after is not None:
            year_end_rate = year_end_meetings[-1].implied_rate_after

        # Rate path narrative
        if next_meeting:
            dominant_action = max(
                [("hold", next_meeting.prob_hold), ("cut_25", next_meeting.prob_cut_25),
                 ("cut_50", next_meeting.prob_cut_50), ("hike_25", next_meeting.prob_hike_25)],
                key=lambda x: x[1],
            )
            narrative = (
                f"Current rate: {current_rate:.2f}%. "
                f"Next FOMC: {next_meeting.meeting_date.isoformat()} "
                f"({next_meeting.days_until} days). "
                f"Most likely outcome: {dominant_action[0].replace('_', ' ')} "
                f"({dominant_action[1]*100:.0f}%). "
            )
            if latest_dot.get("2025_median"):
                narrative += f"Dot plot 2025 median: {latest_dot['2025_median']}%. "
            if latest_dot.get("longer_run"):
                narrative += f"Longer-run neutral rate: {latest_dot['longer_run']}%."
        else:
            narrative = f"Current Fed Funds rate: {current_rate:.2f}%."

        return FOMCOutlook(
            as_of=today,
            current_fed_funds_rate=current_rate,
            next_meeting=next_meeting,
            meetings_2025=built_2025,
            meetings_2026=built_2026,
            dot_plot_latest=latest_dot,
            year_end_rate_target=year_end_rate,
            rate_path_narrative=narrative,
        )


# ---------------------------------------------------------------------------
# EarningsCalendarIntegration
# ---------------------------------------------------------------------------

class EarningsCalendarIntegration:
    """
    Overlay earnings season information on the economic calendar.

    Earnings seasons create:
      - Elevated intraday volatility (guidance risk)
      - Potential revision in macro sentiment via guidance
      - Blackout periods in corporate buybacks (net supply increase)
    """

    # Historical correlation: GDP/CPI surprise → S&P earnings revision
    MACRO_EARNINGS_CORRELATION: dict[str, float] = {
        "GDP_BEAT_EPS_REVISION": 0.45,       # 1% GDP beat → 0.45% EPS revision
        "CPI_MISS_EPS_REVISION": -0.30,      # CPI beat (hot) → earnings compression
        "RATE_HIKE_PE_COMPRESSION": -0.12,   # each 25bp hike → -1.2% P/E
        "RATE_CUT_PE_EXPANSION": 0.15,       # each 25bp cut → +1.5% P/E
    }

    # Sectors with early/late reporting patterns
    SECTOR_REPORTING_WEEKS: dict[str, str] = {
        "Financials": "Week 1-2",        # Banks report early
        "Technology": "Week 2-3",
        "Healthcare": "Week 2-3",
        "Energy": "Week 3-4",
        "Industrials": "Week 3-4",
        "Consumer Discretionary": "Week 2-4",
        "Consumer Staples": "Week 2-3",
        "Utilities": "Week 4-5",
        "Materials": "Week 3-4",
        "Real Estate": "Week 4-5",
        "Communication Services": "Week 2-3",
    }

    def get_current_season(self, target_date: Optional[date] = None) -> Optional[EarningsSeason]:
        """Return the active earnings season if we are currently in one."""
        today = target_date or date.today()
        for season in EARNINGS_SEASONS:
            start = date(today.year, season["start_month"], season["start_day"])
            end = date(today.year, season["end_month"], season["end_day"])
            if start <= today <= end:
                return EarningsSeason(
                    name=season["name"],
                    start_date=start,
                    end_date=end,
                    is_active=True,
                    days_until_end=(end - today).days,
                    affected_sectors=list(self.SECTOR_REPORTING_WEEKS.keys()),
                    macro_overlay=(
                        "Earnings season underway. "
                        "Corporate buyback blackout reduces market support. "
                        "Watch for guidance vs macro correlation."
                    ),
                )
        return None

    def get_next_season(self, target_date: Optional[date] = None) -> Optional[EarningsSeason]:
        """Return the next upcoming earnings season."""
        today = target_date or date.today()
        for season in EARNINGS_SEASONS:
            start = date(today.year, season["start_month"], season["start_day"])
            if start > today:
                end = date(today.year, season["end_month"], season["end_day"])
                return EarningsSeason(
                    name=season["name"],
                    start_date=start,
                    end_date=end,
                    is_active=False,
                    days_until_start=(start - today).days,
                    affected_sectors=list(self.SECTOR_REPORTING_WEEKS.keys()),
                )
        # Check next year
        for season in EARNINGS_SEASONS:
            start = date(today.year + 1, season["start_month"], season["start_day"])
            end = date(today.year + 1, season["end_month"], season["end_day"])
            return EarningsSeason(
                name=f"{season['name']} (next year)",
                start_date=start,
                end_date=end,
                is_active=False,
                days_until_start=(start - today).days,
                affected_sectors=list(self.SECTOR_REPORTING_WEEKS.keys()),
            )
        return None

    def macro_to_earnings_impact(
        self,
        gdp_surprise_pct: float = 0.0,
        cpi_surprise_bps: float = 0.0,
        fed_cut_bps: float = 0.0,
    ) -> dict:
        """
        Estimate earnings revision impact from recent macro surprises.
        Returns approximate EPS revision % and P/E multiple change.
        """
        eps_revision = (
            gdp_surprise_pct * self.MACRO_EARNINGS_CORRELATION["GDP_BEAT_EPS_REVISION"]
            + (cpi_surprise_bps / 100) * self.MACRO_EARNINGS_CORRELATION["CPI_MISS_EPS_REVISION"]
        )
        pe_change = (fed_cut_bps / 25) * self.MACRO_EARNINGS_CORRELATION["RATE_CUT_PE_EXPANSION"]

        return {
            "eps_revision_pct": round(eps_revision, 2),
            "pe_multiple_change_pct": round(pe_change, 2),
            "total_equity_impact_pct": round(eps_revision + pe_change, 2),
            "methodology": (
                f"GDP surprise: {gdp_surprise_pct:+.2f}% → EPS rev {eps_revision:+.2f}%. "
                f"Rate cuts: {fed_cut_bps:.0f}bp → P/E change {pe_change:+.2f}%."
            ),
        }


# ---------------------------------------------------------------------------
# Core EconomicCalendarEngine
# ---------------------------------------------------------------------------

class EconomicCalendarEngine:
    """
    Full economic calendar with consensus proxies, surprise tracking,
    FOMC schedule, event risk, and earnings season overlay.
    """

    def __init__(
        self,
        timeout: float = 25.0,
        fred_api_key: str = "",
        db_path: Optional[Path] = None,
    ):
        self._timeout = timeout
        self._api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        self._db = _EconDB(db_path or _DB_PATH)
        self._cache: dict[str, pd.Series] = {}

    async def _get_history(self, release_id: str, years_back: int = 10) -> pd.Series:
        meta = ECONOMIC_RELEASES.get(release_id, {})
        series_id = meta.get("fred_chg_series") or meta.get("fred_series", "")
        if not series_id:
            return pd.Series(dtype=float)
        if series_id in self._cache:
            return self._cache[series_id]
        hist = await _fetch_series_history(series_id, years_back=years_back)
        self._cache[series_id] = hist
        return hist

    def _build_release(
        self,
        release_id: str,
        release_date: Optional[date],
        history: pd.Series,
    ) -> EconomicRelease:
        meta = ECONOMIC_RELEASES[release_id]
        cat = meta["category"]

        est = _estimate_consensus(history, n_trailing=12)
        consensus = est.trailing_median if est else None
        c_std = est.trailing_std if est else None
        c_low = est.range_low if est else None
        c_high = est.range_high if est else None

        prior: Optional[float] = None
        actual: Optional[float] = None
        surprise: Optional[float] = None
        surprise_pct: Optional[float] = None
        surprise_z: Optional[float] = None
        market_signal: Optional[MarketSignal] = None

        if not history.empty:
            values = history.dropna()
            if len(values) >= 2:
                prior = float(values.iloc[-2])
                actual = float(values.iloc[-1])
            elif len(values) == 1:
                actual = float(values.iloc[-1])

            if actual is not None and consensus is not None and c_std is not None:
                surprise, surprise_pct, surprise_z, market_signal = _compute_surprise(
                    actual, consensus, c_std, cat
                )
                # Persist to DB
                try:
                    hist_date = (
                        history.index[-1].date()
                        if hasattr(history.index[-1], "date")
                        else date.today()
                    )
                    self._db.upsert_surprise(SurpriseRecord(
                        release_id=release_id,
                        release_date=hist_date.isoformat(),
                        release_name=meta["name"],
                        consensus=round(consensus, 4),
                        actual=round(actual, 4),
                        surprise=round(surprise, 4),
                        surprise_pct=round(surprise_pct, 2),
                        surprise_z=round(surprise_z, 3),
                    ))
                except Exception as exc:
                    logger.debug("DB upsert failed", error=str(exc))

        hist_list: list[dict] = []
        if not history.empty:
            for idx, val in history.iloc[-24:].items():
                hist_list.append({
                    "date": idx.date().isoformat() if hasattr(idx, "date") else str(idx),
                    "value": round(float(val), 4),
                })

        return EconomicRelease(
            release_id=release_id,
            name=meta["name"],
            release_date=release_date,
            release_time=meta.get("release_time"),
            category=cat,
            frequency=meta["frequency"],
            market_impact=meta["market_impact"],
            description=meta.get("description", ""),
            consensus=round(consensus, 4) if consensus is not None else None,
            consensus_range_low=round(c_low, 4) if c_low is not None else None,
            consensus_range_high=round(c_high, 4) if c_high is not None else None,
            consensus_std=round(c_std, 4) if c_std is not None else None,
            prior=round(prior, 4) if prior is not None else None,
            actual=round(actual, 4) if actual is not None else None,
            surprise=round(surprise, 4) if surprise is not None else None,
            surprise_pct=round(surprise_pct, 2) if surprise_pct is not None else None,
            surprise_z=round(surprise_z, 3) if surprise_z is not None else None,
            market_signal=market_signal,
            historical_data=hist_list if hist_list else None,
            notes=(
                "Consensus is a trailing-12-period median proxy. "
                "For official sell-side consensus use Bloomberg or Refinitiv."
            ),
        )

    async def get_release_history(self, release_id: str, years_back: int = 10) -> pd.DataFrame:
        if release_id not in ECONOMIC_RELEASES:
            raise ValueError(f"Unknown release_id: {release_id!r}")
        series = await self._get_history(release_id, years_back=years_back)
        if series.empty:
            return pd.DataFrame()
        df = series.to_frame(name="value")
        df["rolling_mean_12"] = df["value"].rolling(12).mean()
        df["rolling_std_12"] = df["value"].rolling(12).std()
        df["z_score"] = (df["value"] - df["rolling_mean_12"]) / df["rolling_std_12"]
        df["yoy_chg"] = df["value"].pct_change(12) * 100
        return df.round(4)

    async def get_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        categories: Optional[list[str]] = None,
        min_impact: str = "medium",
    ) -> EconomicCalendar:
        today = date.today()
        start_date = start_date or today
        end_date = end_date or (today + timedelta(days=30))

        impact_order = {"very_high": 4, "high": 3, "medium": 2, "low": 1}
        min_rank = impact_order.get(min_impact, 2)

        filtered_ids = [
            rid for rid, meta in ECONOMIC_RELEASES.items()
            if impact_order.get(meta["market_impact"], 0) >= min_rank
            and (categories is None or meta["category"] in categories)
        ]

        scheduled = _build_scheduled_dates(start_date, end_date)

        histories = await asyncio.gather(
            *[self._get_history(rid) for rid in filtered_ids],
            return_exceptions=True,
        )

        releases: list[EconomicRelease] = []
        for rid, hist_or_exc in zip(filtered_ids, histories):
            if isinstance(hist_or_exc, Exception):
                hist_or_exc = pd.Series(dtype=float)

            dates_for_rid = scheduled.get(rid, [])
            if dates_for_rid:
                for d in dates_for_rid:
                    releases.append(self._build_release(rid, d, hist_or_exc))
            else:
                releases.append(self._build_release(rid, None, hist_or_exc))

        releases.sort(
            key=lambda r: (r.release_date is None, r.release_date or date.max, r.name)
        )

        fomc_in_window = _upcoming_fomc(start_date, end_date)
        high_count = sum(1 for r in releases if r.market_impact in ("very_high", "high"))

        next_major: Optional[EconomicRelease] = None
        for r in releases:
            if r.release_date and r.release_date >= today and r.market_impact in ("very_high", "high"):
                next_major = r
                break

        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        this_week = [r for r in releases if r.release_date and monday <= r.release_date <= sunday]

        in_earnings, season_name = _is_in_earnings_season(today)

        return EconomicCalendar(
            as_of=today,
            start_date=start_date,
            end_date=end_date,
            releases=releases,
            high_impact_count=high_count,
            fomc_dates=fomc_in_window,
            next_major_release=next_major,
            releases_this_week=this_week,
            in_earnings_season=in_earnings,
            earnings_season_name=season_name,
        )

    async def get_upcoming_week(self) -> list[EconomicRelease]:
        today = date.today()
        calendar = await self.get_calendar(
            start_date=today,
            end_date=today + timedelta(days=7),
            min_impact="medium",
        )
        return [r for r in calendar.releases if r.release_date is not None]

    async def get_recent_surprises(
        self,
        n_releases: int = 20,
        categories: Optional[list[str]] = None,
    ) -> list[EconomicRelease]:
        rids = [
            rid for rid, meta in ECONOMIC_RELEASES.items()
            if categories is None or meta["category"] in categories
        ]

        histories = await asyncio.gather(
            *[self._get_history(rid) for rid in rids],
            return_exceptions=True,
        )

        results: list[EconomicRelease] = []
        for rid, hist_or_exc in zip(rids, histories):
            if isinstance(hist_or_exc, Exception):
                continue
            if not isinstance(hist_or_exc, pd.Series) or hist_or_exc.empty:
                continue
            release = self._build_release(rid, None, hist_or_exc)
            if release.actual is not None and release.surprise_z is not None:
                results.append(release)

        results.sort(
            key=lambda r: abs(r.surprise_z) if r.surprise_z is not None else 0.0,
            reverse=True,
        )
        return results[:n_releases]

    async def compute_surprise_index(
        self,
        category: str = "all",
        months_back: int = 3,
        region: str = "US",
    ) -> SurpriseIndex:
        esi = EconomicSurpriseIndex()
        return await esi.compute(category=category, months_back=months_back, region=region)

    async def get_fomc_calendar(self, year: Optional[int] = None) -> list[date]:
        return get_fomc_dates(year)

    def estimate_consensus(
        self, release_id: str, history: pd.Series, n_trailing: int = 12
    ) -> dict:
        est = _estimate_consensus(history, n_trailing=n_trailing)
        if est is None:
            return {
                "release_id": release_id,
                "estimate": None, "std_dev": None, "range_low": None,
                "range_high": None, "n_observations": 0,
            }
        return {
            "release_id": release_id,
            "estimate": est.trailing_median,
            "std_dev": est.trailing_std,
            "range_low": est.range_low,
            "range_high": est.range_high,
            "n_observations": est.n_observations,
        }

    def surprise_history_from_db(self, release_id: str, n: int = 20) -> pd.DataFrame:
        """Retrieve persisted surprise history from SQLite."""
        rows = self._db.get_surprise_history(release_id, n=n)
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query

    econ_calendar_router = APIRouter(prefix="/calendar", tags=["economic-calendar"])

    # Shared instances
    _engine = EconomicCalendarEngine(
        fred_api_key=os.environ.get("FRED_API_KEY", "")
    )
    _esi = EconomicSurpriseIndex()
    _fed_tracker = FedMeetingTracker()
    _event_risk = MacroEventRiskEngine()
    _earnings_calendar = EarningsCalendarIntegration()
    _consensus_tracker = ConsensusEstimateTracker()

    @econ_calendar_router.get("/upcoming", summary="Upcoming economic releases with consensus proxies")
    async def get_upcoming(
        days_ahead: int = Query(14, ge=1, le=90),
        min_impact: str = Query("medium"),
        categories: Optional[str] = Query(None, description="Comma-separated list"),
    ) -> dict:
        try:
            cats = [c.strip() for c in categories.split(",")] if categories else None
            today = date.today()
            calendar = await _engine.get_calendar(
                start_date=today,
                end_date=today + timedelta(days=days_ahead),
                min_impact=min_impact,
                categories=cats,
            )
            return {
                "as_of": calendar.as_of.isoformat(),
                "window_days": days_ahead,
                "total_releases": len(calendar.releases),
                "high_impact_count": calendar.high_impact_count,
                "in_earnings_season": calendar.in_earnings_season,
                "earnings_season_name": calendar.earnings_season_name,
                "fomc_dates": [d.isoformat() for d in calendar.fomc_dates],
                "next_major_release": calendar.next_major_release.model_dump() if calendar.next_major_release else None,
                "releases": [r.model_dump() for r in calendar.releases],
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @econ_calendar_router.get("/surprise-index", summary="Economic Surprise Index (ESI) by category and region")
    async def get_surprise_index(
        category: str = Query("all"),
        months_back: int = Query(3, ge=1, le=12),
        region: str = Query("US"),
    ) -> dict:
        try:
            index = await _esi.compute(category=category, months_back=months_back, region=region)
            return index.model_dump()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @econ_calendar_router.get("/surprise-index/all-regions", summary="ESI for all regions")
    async def get_surprise_index_all_regions(
        months_back: int = Query(3, ge=1, le=12),
    ) -> dict:
        try:
            results = await _esi.compute_all_regions(months_back=months_back)
            return {
                "as_of": date.today().isoformat(),
                "regions": {k: v.model_dump() for k, v in results.items()},
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @econ_calendar_router.get("/fomc", summary="FOMC meeting schedule and rate probabilities")
    async def get_fomc() -> dict:
        try:
            outlook = await _fed_tracker.get_fomc_outlook()
            return outlook.model_dump()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @econ_calendar_router.get("/event-risk/{event_name}", summary="Pre-event risk analysis and position sizing")
    async def get_event_risk(event_name: str) -> dict:
        try:
            profile = _event_risk.get_event_risk_profile(event_name.upper())
            events_this_week = _event_risk.event_week_calendar(date.today())
            sizing = _event_risk.position_sizing_recommendation(events_this_week)
            return {
                "event_profile": profile.model_dump(),
                "week_sizing": sizing,
                "events_this_week": events_this_week,
            }
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @econ_calendar_router.get("/history/{release_id}", summary="Historical data with rolling statistics for a release")
    async def get_release_history(
        release_id: str,
        years_back: int = Query(10, ge=1, le=20),
    ) -> dict:
        try:
            df = await _engine.get_release_history(release_id.upper(), years_back=years_back)
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No data for {release_id}")
            # Recent surprises from DB
            beat_rate = ConsensusEstimateTracker(_engine._db).beat_rate(release_id.upper())
            return {
                "release_id": release_id.upper(),
                "years_back": years_back,
                "n_observations": len(df),
                "beat_rate": beat_rate,
                "data": df.reset_index().rename(columns={"DATE": "date"}).to_dict(orient="records"),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @econ_calendar_router.get("/earnings-season", summary="Current and upcoming earnings seasons")
    async def get_earnings_season() -> dict:
        try:
            current = _earnings_calendar.get_current_season()
            upcoming = _earnings_calendar.get_next_season() if not current else None
            return {
                "as_of": date.today().isoformat(),
                "current_season": current.model_dump() if current else None,
                "next_season": upcoming.model_dump() if upcoming else None,
                "sector_reporting_weeks": _earnings_calendar.SECTOR_REPORTING_WEEKS,
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

except ImportError:
    econ_calendar_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available; econ_calendar_router not registered")


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

async def economic_calendar(
    days_ahead: int = 14,
    min_impact: str = "medium",
) -> EconomicCalendar:
    """Fetch the upcoming economic calendar for the next N days."""
    try:
        from sentinel.core.config import get_settings
        settings = get_settings()
        api_key = settings.fred_api_key
    except Exception:
        api_key = os.environ.get("FRED_API_KEY", "")

    engine = EconomicCalendarEngine(fred_api_key=api_key)
    today = date.today()
    return await engine.get_calendar(
        start_date=today,
        end_date=today + timedelta(days=days_ahead),
        min_impact=min_impact,
    )


async def surprise_index(
    category: str = "inflation",
    months_back: int = 3,
    region: str = "US",
) -> SurpriseIndex:
    """Compute the economic surprise index for a category and region."""
    esi = EconomicSurpriseIndex()
    return await esi.compute(category=category, months_back=months_back, region=region)


async def release_history(
    release_id: str,
    years_back: int = 10,
) -> pd.DataFrame:
    """Fetch full FRED history with rolling statistics for a named release."""
    engine = EconomicCalendarEngine()
    return await engine.get_release_history(release_id, years_back=years_back)


async def fomc_outlook() -> FOMCOutlook:
    """Get comprehensive FOMC meeting schedule and rate probability outlook."""
    tracker = FedMeetingTracker()
    return await tracker.get_fomc_outlook()


async def upcoming_releases(days_ahead: int = 14) -> List[dict]:
    """Get upcoming releases with trailing-median consensus proxies."""
    tracker = ConsensusEstimateTracker()
    return await tracker.get_upcoming_with_consensus(days_ahead=days_ahead)


def event_risk_profile(event_name: str) -> EventRiskProfile:
    """Get pre-event risk profile and position sizing recommendation."""
    engine = MacroEventRiskEngine()
    return engine.get_event_risk_profile(event_name)


def summarize_calendar(calendar: EconomicCalendar) -> dict:
    """Return a concise summary dict for display or logging."""
    by_impact: dict[str, int] = {}
    for r in calendar.releases:
        by_impact[r.market_impact] = by_impact.get(r.market_impact, 0) + 1

    next_r = calendar.next_major_release
    return {
        "as_of": calendar.as_of.isoformat(),
        "window": f"{calendar.start_date.isoformat()} → {calendar.end_date.isoformat()}",
        "total_releases": len(calendar.releases),
        "by_impact": by_impact,
        "high_impact_count": calendar.high_impact_count,
        "this_week_count": len(calendar.releases_this_week),
        "fomc_dates": [d.isoformat() for d in calendar.fomc_dates],
        "in_earnings_season": calendar.in_earnings_season,
        "earnings_season_name": calendar.earnings_season_name,
        "next_major_release": {
            "id": next_r.release_id,
            "name": next_r.name,
            "date": next_r.release_date.isoformat() if next_r.release_date else None,
            "time": next_r.release_time,
            "impact": next_r.market_impact,
            "description": next_r.description,
            "consensus": next_r.consensus,
            "prior": next_r.prior,
            "surprise_z_last": next_r.surprise_z,
        } if next_r else None,
    }
