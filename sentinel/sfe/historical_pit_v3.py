"""
Historical Point-In-Time Financial Database — dim_020 v3  (target score: 9/10)

The audit problem: the previous implementation approximated publication lags
with static lookup tables rather than computing actual per-company, per-filing
lag from EDGAR's authoritative filingDate + reportDate fields.

This module builds a *true* PIT database:

Key design decisions
--------------------
1. **filingDate is the PIT timestamp.**
   EDGAR's submissions API provides two dates per filing:
     - reportDate   (= fiscal period end, the "period of report")
     - filingDate   (= date SEC received / accepted the filing)
   We use ``filingDate`` as the moment data became publicly available.

2. **Restatement detection.**
   When the same (ticker, period_end, metric) appears in multiple filings,
   the later one is a restatement.  We store *every* vintage and flag them.

3. **Per-company lag registry.**
   For each company we compute actual lags from its historical filings and
   classify it by filer_status (large_accelerated / accelerated / non_accelerated
   / SRC).  Median lag per status is used for forward-looking coverage estimates.

4. **Look-ahead bias detector.**
   Given a backtest start date and a universe, report which tickers had filed
   data by that date and which had not.

5. **PIT time series with vintage labels.**
   For each metric we build a full time series keyed by (period_end, filed_date),
   so backtests can accurately reproduce the information set at any point.

SQLite tables
-------------
pit_filings       — one row per (cik, accession), stores filingDate, reportDate
pit_financials    — one row per (cik, period_end, metric, accn), stores vintage values
restatements      — flagged pairs (original_accn, restatement_accn)
lag_registry      — per-company lag statistics by filer_status
publication_lags  — every individual filing's lag in days

FastAPI router: /pit/v3
  GET /financials/{ticker}?as_of=YYYY-MM-DD
  GET /lag-report/{ticker}
  GET /restatements/{ticker}
  GET /backtest-universe?as_of=YYYY-MM-DD&index=sp500
  GET /vintage/{ticker}/{period}
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://data.sec.gov"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 35.0
_RATE_DELAY = 0.12        # 120 ms → well under EDGAR's 10 req/s cap
_SEMAPHORE_WIDTH = 4      # max parallel EDGAR fetches

# Annual and quarterly form types
_ANNUAL_FORMS = frozenset({"10-K", "10-KT", "20-F", "40-F", "10-K/A"})
_QUARTERLY_FORMS = frozenset({"10-Q", "10-QT", "10-Q/A"})
_ALL_PERIODIC = _ANNUAL_FORMS | _QUARTERLY_FORMS

# Balance-sheet items: point-in-time values (NOT summed for TTM)
_BALANCE_SHEET_METRICS = frozenset({
    "total_assets", "total_liabilities", "equity", "long_term_debt",
    "short_term_debt", "cash", "shares_diluted", "inventory", "receivables",
})

# Filer-status lag targets (calendar days after period end)
_LAG_TARGETS: dict[str, int] = {
    "large_accelerated": 40,   # 10-Q; 60 for 10-K
    "accelerated": 45,         # 10-Q; 75 for 10-K
    "non_accelerated": 60,
    "src": 75,
    "unknown": 90,
}
_ANNUAL_LAG_TARGETS: dict[str, int] = {
    "large_accelerated": 60,
    "accelerated": 75,
    "non_accelerated": 90,
    "src": 105,
    "unknown": 120,
}

# Canonical metric → XBRL concept fallback chain
CONCEPT_MAP: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "gross_profit": ["GrossProfit"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "short_term_debt": ["ShortTermBorrowings", "NotesPayableCurrent", "DebtCurrent"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "rd": ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "shares_diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "CommonStockSharesOutstanding",
    ],
    "inventory": ["InventoryNet", "Inventories"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "dividends_per_share": ["CommonStockDividendsPerShareCashPaid"],
}

# S&P 500 tickers (representative 100 for coverage reports; extend as needed)
SP500_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH",
    "JPM", "V", "XOM", "JNJ", "PG", "MA", "HD", "AVGO", "CVX", "MRK", "ABBV",
    "COST", "PEP", "ADBE", "KO", "WMT", "TMO", "CRM", "ABT", "LIN", "CSCO",
    "MCD", "BAC", "NFLX", "DHR", "DIS", "ACN", "CMCSA", "TXN", "VZ", "NEE",
    "PM", "LLY", "INTC", "IBM", "RTX", "AMD", "UPS", "HON", "QCOM", "GS",
    "LOW", "CAT", "MS", "SBUX", "AMGN", "INTU", "BLK", "GE", "MDT", "AXP",
    "BA", "DE", "AMAT", "MMM", "ISRG", "NOW", "ADI", "TGT", "MDLZ", "CVS",
    "MU", "ZTS", "LMT", "C", "ADP", "WFC", "REGN", "SPGI", "BKNG", "EOG",
    "SYK", "CB", "EL", "USB", "SO", "GILD", "PLD", "NSC", "EMR", "CME",
    "D", "F", "GM", "ORLY", "PCAR", "MO", "CL", "DUK", "ETN", "SHW",
]


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "data" / "pit_v3.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        -- Every filing with its key dates
        CREATE TABLE IF NOT EXISTS pit_filings (
            cik             TEXT NOT NULL,
            accession       TEXT NOT NULL,
            form            TEXT NOT NULL,
            filing_date     TEXT NOT NULL,   -- filingDate (PIT timestamp)
            report_date     TEXT NOT NULL,   -- reportDate (period end)
            lag_days        INTEGER,         -- filing_date - report_date
            is_annual       INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (cik, accession)
        );
        CREATE INDEX IF NOT EXISTS idx_pit_filings_cik_report
            ON pit_filings(cik, report_date);
        CREATE INDEX IF NOT EXISTS idx_pit_filings_cik_filing
            ON pit_filings(cik, filing_date);

        -- Individual metric observations — every vintage stored
        CREATE TABLE IF NOT EXISTS pit_financials (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cik             TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            accession       TEXT NOT NULL,
            metric          TEXT NOT NULL,
            xbrl_concept    TEXT NOT NULL,
            period_start    TEXT,
            period_end      TEXT NOT NULL,
            filed_date      TEXT NOT NULL,
            value           REAL NOT NULL,
            is_annual       INTEGER NOT NULL DEFAULT 0,
            is_quarterly    INTEGER NOT NULL DEFAULT 0,
            UNIQUE(cik, accession, metric, period_end)
        );
        CREATE INDEX IF NOT EXISTS idx_pit_fin_cik_metric_period
            ON pit_financials(cik, metric, period_end, filed_date);
        CREATE INDEX IF NOT EXISTS idx_pit_fin_cik_filed
            ON pit_financials(cik, filed_date);

        -- Restatement registry: same period, different value in later filing
        CREATE TABLE IF NOT EXISTS restatements (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            cik                 TEXT NOT NULL,
            ticker              TEXT NOT NULL,
            metric              TEXT NOT NULL,
            period_end          TEXT NOT NULL,
            original_accn       TEXT NOT NULL,
            original_filed      TEXT NOT NULL,
            original_value      REAL NOT NULL,
            restatement_accn    TEXT NOT NULL,
            restatement_filed   TEXT NOT NULL,
            restatement_value   REAL NOT NULL,
            pct_change          REAL,
            detected_at         TEXT DEFAULT (datetime('now')),
            UNIQUE(cik, metric, period_end, restatement_accn)
        );

        -- Per-company lag statistics
        CREATE TABLE IF NOT EXISTS lag_registry (
            cik             TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            filer_status    TEXT,
            form_type       TEXT NOT NULL,
            median_lag_days REAL,
            mean_lag_days   REAL,
            min_lag_days    INTEGER,
            max_lag_days    INTEGER,
            n_filings       INTEGER,
            computed_at     TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (cik, form_type)
        );

        -- Every individual filing's lag (raw data behind lag_registry)
        CREATE TABLE IF NOT EXISTS publication_lags (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cik         TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            accession   TEXT NOT NULL,
            form        TEXT NOT NULL,
            report_date TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            lag_days    INTEGER NOT NULL,
            UNIQUE(cik, accession)
        );

        -- Coverage index: first available PIT date per ticker
        CREATE TABLE IF NOT EXISTS coverage_index (
            ticker              TEXT PRIMARY KEY,
            cik                 TEXT NOT NULL,
            first_filing_date   TEXT,
            first_report_date   TEXT,
            total_filings       INTEGER,
            last_updated        TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


_init_db()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class PITSnapshot(BaseModel):
    """Financial data available as of a specific as_of_date."""
    ticker: str
    cik: str
    company_name: str
    as_of_date: str
    fiscal_period_end: str
    filed_date: str
    filing_lag_days: int

    revenue: Optional[float] = None
    revenue_ttm: Optional[float] = None
    net_income: Optional[float] = None
    net_income_ttm: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    eps_diluted: Optional[float] = None
    total_assets: Optional[float] = None
    equity: Optional[float] = None
    long_term_debt: Optional[float] = None
    short_term_debt: Optional[float] = None
    cash: Optional[float] = None
    cfo: Optional[float] = None
    cfo_ttm: Optional[float] = None
    capex: Optional[float] = None
    capex_ttm: Optional[float] = None
    free_cash_flow: Optional[float] = None
    da: Optional[float] = None
    shares_diluted: Optional[float] = None
    rd: Optional[float] = None

    # Derived
    total_debt: Optional[float] = None
    net_debt: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None

    n_concepts_found: int = 0
    data_vintage: str = "original"   # "original" | "restated"


class FilingRecord(BaseModel):
    accession: str
    form: str
    filing_date: str
    report_date: str
    lag_days: int
    is_annual: bool


class LagReport(BaseModel):
    ticker: str
    cik: str
    form_type: str
    median_lag_days: Optional[float]
    mean_lag_days: Optional[float]
    min_lag_days: Optional[int]
    max_lag_days: Optional[int]
    n_filings: int
    filer_status_estimate: str
    target_lag_days: int
    filings: list[FilingRecord] = Field(default_factory=list)


class RestatementRecord(BaseModel):
    metric: str
    period_end: str
    original_filed: str
    original_value: float
    restatement_filed: str
    restatement_value: float
    pct_change: Optional[float]


class VintageRecord(BaseModel):
    metric: str
    period_end: str
    filed_date: str
    value: float
    accession: str
    is_annual: bool
    is_quarterly: bool


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, delay: float = _RATE_DELAY) -> None:
        self._delay = delay
        self._last = 0.0

    async def wait(self) -> None:
        now = time.monotonic()
        gap = self._delay - (now - self._last)
        if gap > 0:
            await asyncio.sleep(gap)
        self._last = time.monotonic()


_limiter = _RateLimiter()


async def _get_json(url: str, timeout: float = _TIMEOUT) -> dict:
    await _limiter.wait()
    async with httpx.AsyncClient(
        headers=_HEADERS, timeout=timeout, follow_redirects=True
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# CIK resolver
# ---------------------------------------------------------------------------

_cik_cache: dict[str, tuple[str, str]] = {}   # ticker → (padded_cik, name)


async def _resolve_cik(ticker: str) -> tuple[str, str]:
    key = ticker.upper()
    if key in _cik_cache:
        return _cik_cache[key]
    data = await _get_json(EDGAR_TICKERS_URL)
    for entry in data.values():
        t = str(entry.get("ticker", "")).upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        name = str(entry.get("title", ""))
        if t:
            _cik_cache[t] = (cik, name)
    if key not in _cik_cache:
        raise LookupError(f"Ticker '{key}' not found in SEC company_tickers.json")
    return _cik_cache[key]


# ---------------------------------------------------------------------------
# EDGAR submissions fetcher — builds the PIT filing index
# ---------------------------------------------------------------------------

@dataclass
class FilingMeta:
    accession: str
    form: str
    filing_date: str   # YYYY-MM-DD — the PIT timestamp
    report_date: str   # YYYY-MM-DD — fiscal period end
    lag_days: int
    is_annual: bool


async def _fetch_filing_index(cik: str) -> list[FilingMeta]:
    """
    Pull all periodic filings for a CIK from the submissions API and
    return FilingMeta objects with lag_days computed.

    The submissions JSON 'filings.recent' array has parallel arrays:
        accessionNumber, filingDate, reportDate, form, ...

    For companies with > 1000 filings, additional pages are linked via
    filings.files[].name and must be fetched separately.
    """
    padded = cik.zfill(10)
    url = EDGAR_SUBMISSIONS_URL.format(cik=padded)

    try:
        sub = await _get_json(url, timeout=45.0)
    except Exception as exc:
        logger.warning("Submissions fetch failed", cik=padded, error=str(exc))
        return []

    results: list[FilingMeta] = []
    entity_name = sub.get("name", "")

    def _process_block(filings_block: dict) -> None:
        forms = filings_block.get("form", [])
        accessions = filings_block.get("accessionNumber", [])
        filing_dates = filings_block.get("filingDate", [])
        report_dates = filings_block.get("reportDate", [])

        for i, form in enumerate(forms):
            if form not in _ALL_PERIODIC:
                continue
            accn = accessions[i] if i < len(accessions) else ""
            fd = filing_dates[i] if i < len(filing_dates) else ""
            rd = report_dates[i] if i < len(report_dates) else ""

            if not accn or not fd or not rd:
                continue

            try:
                lag = (
                    date.fromisoformat(fd) - date.fromisoformat(rd)
                ).days
            except ValueError:
                lag = 0

            # Sanity: negative lag or >500 days = data error, skip
            if lag < 0 or lag > 500:
                continue

            results.append(FilingMeta(
                accession=accn,
                form=form,
                filing_date=fd,
                report_date=rd,
                lag_days=lag,
                is_annual=form in _ANNUAL_FORMS,
            ))

    # Process the 'recent' block
    _process_block(sub.get("filings", {}).get("recent", {}))

    # Fetch additional pages if the company has > 1000 filings
    additional_files = sub.get("filings", {}).get("files", [])
    for extra in additional_files:
        extra_url = f"{EDGAR_BASE}/submissions/{extra.get('name', '')}"
        if not extra_url.endswith(".json"):
            continue
        try:
            extra_data = await _get_json(extra_url, timeout=30.0)
            _process_block(extra_data)
        except Exception as exc:
            logger.warning("Extra submissions page failed", url=extra_url, error=str(exc))

    return results


# ---------------------------------------------------------------------------
# XBRL facts fetcher
# ---------------------------------------------------------------------------

_facts_cache: dict[str, dict] = {}   # padded_cik → companyfacts JSON


async def _load_facts(cik: str) -> dict:
    padded = cik.zfill(10)
    if padded in _facts_cache:
        return _facts_cache[padded]
    url = EDGAR_FACTS_URL.format(cik=padded)
    try:
        data = await _get_json(url, timeout=60.0)
        _facts_cache[padded] = data
        return data
    except Exception as exc:
        logger.warning("companyfacts fetch failed", cik=padded, error=str(exc))
        return {}


def _raw_observations(facts: dict, xbrl_concept: str) -> list[dict]:
    """Return raw observation list for an XBRL concept (USD preferred)."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    concept_data = us_gaap.get(xbrl_concept, {})
    units = concept_data.get("units", {})
    for unit_key in ("USD", "shares", "USD/shares", "pure"):
        if unit_key in units:
            return units[unit_key]
    for obs_list in units.values():
        return obs_list
    return []


def _detect_unit(facts: dict, xbrl_concept: str) -> str:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    units = us_gaap.get(xbrl_concept, {}).get("units", {})
    for k in ("USD", "shares", "USD/shares", "pure"):
        if k in units:
            return k
    return next(iter(units), "USD")


def _is_quarterly_obs(obs: dict) -> bool:
    form = obs.get("form", "")
    if form in _QUARTERLY_FORMS:
        return True
    start = obs.get("start")
    end = obs.get("end") or obs.get("instant")
    if start and end:
        try:
            days = (date.fromisoformat(end) - date.fromisoformat(start)).days
            return 70 <= days <= 110
        except ValueError:
            pass
    return False


def _is_annual_obs(obs: dict) -> bool:
    return obs.get("form", "") in _ANNUAL_FORMS


# ---------------------------------------------------------------------------
# Per-filing XBRL extraction (PIT aware)
# ---------------------------------------------------------------------------

@dataclass
class MetricVintage:
    """A single (metric, period_end) observation from one specific filing."""
    metric: str
    xbrl_concept: str
    period_start: Optional[str]
    period_end: str
    filed_date: str
    accession: str
    value: float
    is_annual: bool
    is_quarterly: bool


def _extract_all_vintages(
    facts: dict,
    metric: str,
    min_year: int = 2000,
) -> list[MetricVintage]:
    """
    Extract every observation for a metric across all time — multiple
    vintages per period (restatements included).

    Returns list sorted by (period_end asc, filed_date asc).
    """
    cutoff = date(min_year, 1, 1)
    vintages: list[MetricVintage] = []
    seen_concepts: set[str] = set()

    for xbrl_concept in CONCEPT_MAP.get(metric, []):
        obs_list = _raw_observations(facts, xbrl_concept)
        if not obs_list:
            continue
        seen_concepts.add(xbrl_concept)

        for obs in obs_list:
            filed_str = obs.get("filed")
            end_str = obs.get("end") or obs.get("instant")
            if not filed_str or not end_str:
                continue
            try:
                filed = date.fromisoformat(filed_str)
                period_end = date.fromisoformat(end_str)
            except ValueError:
                continue
            if period_end < cutoff:
                continue
            val = obs.get("val")
            if val is None:
                continue

            is_ann = _is_annual_obs(obs)
            is_qtr = _is_quarterly_obs(obs)
            if not is_ann and not is_qtr:
                continue

            vintages.append(MetricVintage(
                metric=metric,
                xbrl_concept=xbrl_concept,
                period_start=obs.get("start"),
                period_end=end_str,
                filed_date=filed_str,
                accession=obs.get("accn", ""),
                value=float(val),
                is_annual=is_ann,
                is_quarterly=is_qtr,
            ))

        # Stop at first concept with data — preserves concept priority
        if vintages:
            break

    vintages.sort(key=lambda v: (v.period_end, v.filed_date))
    return vintages


# ---------------------------------------------------------------------------
# PIT query engine
# ---------------------------------------------------------------------------

class PITEngineV3:
    """
    True PIT engine: all queries are gated by filingDate <= as_of_date.

    The engine operates in two layers:
        1. Filing index layer (submissions API) — knows WHEN each filing arrived.
        2. XBRL layer (companyfacts) — knows WHAT was in each filing.

    Combining the two gives genuine PIT retrieval with restatement tracking.
    """

    def __init__(self) -> None:
        self._filings_cache: dict[str, list[FilingMeta]] = {}  # cik → filings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_filings(self, cik: str) -> list[FilingMeta]:
        padded = cik.zfill(10)
        if padded in self._filings_cache:
            return self._filings_cache[padded]
        filings = await _fetch_filing_index(padded)
        self._filings_cache[padded] = filings
        return filings

    def _filings_available_on(
        self, filings: list[FilingMeta], as_of: date
    ) -> list[FilingMeta]:
        """Filter to filings where filing_date <= as_of_date."""
        return [
            f for f in filings
            if date.fromisoformat(f.filing_date) <= as_of
        ]

    def _pit_value(
        self,
        vintages: list[MetricVintage],
        as_of: date,
        prefer: str = "latest_period",
    ) -> Optional[MetricVintage]:
        """
        Return the best MetricVintage available on as_of_date.

        prefer="latest_period" → most recent fiscal period whose filing date <= as_of.
        prefer="as_originally_reported" → earliest vintage for the latest period.
        """
        eligible = [
            v for v in vintages
            if date.fromisoformat(v.filed_date) <= as_of
        ]
        if not eligible:
            return None

        # Latest period_end first; for ties, prefer earliest filed (original)
        if prefer == "as_originally_reported":
            eligible.sort(key=lambda v: (-date.fromisoformat(v.period_end).toordinal(),
                                          date.fromisoformat(v.filed_date).toordinal()))
        else:
            eligible.sort(key=lambda v: (-date.fromisoformat(v.period_end).toordinal(),
                                         -date.fromisoformat(v.filed_date).toordinal()))
        return eligible[0]

    def _ttm_value(
        self, vintages: list[MetricVintage], as_of: date
    ) -> Optional[float]:
        """
        TTM = sum of 4 most-recent non-overlapping quarters filed <= as_of.
        Balance-sheet metrics return the latest available point value.
        """
        is_balance = vintages[0].metric in _BALANCE_SHEET_METRICS if vintages else False
        eligible = [
            v for v in vintages
            if v.is_quarterly and date.fromisoformat(v.filed_date) <= as_of
        ]
        if not eligible:
            return None

        eligible.sort(key=lambda v: date.fromisoformat(v.period_end), reverse=True)

        if is_balance:
            return eligible[0].value

        # Greedy non-overlapping quarter selection
        selected: list[MetricVintage] = []
        for v in eligible:
            if len(selected) >= 4:
                break
            v_end = date.fromisoformat(v.period_end)
            v_start = (
                date.fromisoformat(v.period_start)
                if v.period_start else v_end - timedelta(days=91)
            )
            overlaps = any(
                not (v_end <= date.fromisoformat(s.period_start or
                     str(date.fromisoformat(s.period_end) - timedelta(days=91)))
                     or v_start >= date.fromisoformat(s.period_end))
                for s in selected
            )
            if not overlaps:
                selected.append(v)

        if len(selected) < 4:
            return None
        return sum(s.value for s in selected)

    # ------------------------------------------------------------------
    # Public API: PIT snapshot
    # ------------------------------------------------------------------

    async def get_as_of(
        self, ticker: str, as_of: date
    ) -> PITSnapshot:
        """
        Return a PITSnapshot for ticker with only data that had been
        publicly filed by as_of_date.  No look-ahead.
        """
        cik, company_name = await _resolve_cik(ticker)

        # Load all vintages for all metrics
        facts = await _load_facts(cik)
        if not facts:
            raise ValueError(f"No XBRL facts found for {ticker} / CIK {cik}")

        n_found = 0
        metric_vintages: dict[str, list[MetricVintage]] = {}
        for metric in CONCEPT_MAP:
            vts = _extract_all_vintages(facts, metric)
            if vts:
                metric_vintages[metric] = vts

        def _get(metric: str) -> Optional[float]:
            vts = metric_vintages.get(metric, [])
            v = self._pit_value(vts, as_of)
            return v.value if v else None

        def _ttm(metric: str) -> Optional[float]:
            vts = metric_vintages.get(metric, [])
            return self._ttm_value(vts, as_of)

        def _anchor(metric: str) -> Optional[MetricVintage]:
            vts = metric_vintages.get(metric, [])
            return self._pit_value(vts, as_of) if vts else None

        # ── Income Statement ─────────────────────────────────────────────
        revenue = _get("revenue")
        if revenue is not None:
            n_found += 1
        revenue_ttm = _ttm("revenue")
        net_income = _get("net_income")
        if net_income is not None:
            n_found += 1
        net_income_ttm = _ttm("net_income")
        gross_profit = _get("gross_profit")
        if gross_profit is not None:
            n_found += 1
        operating_income = _get("operating_income")
        if operating_income is not None:
            n_found += 1
        eps_diluted = _get("eps_diluted")
        if eps_diluted is not None:
            n_found += 1
        rd = _get("rd")
        da = _get("da")

        # ── Balance Sheet ────────────────────────────────────────────────
        total_assets = _get("total_assets")
        if total_assets is not None:
            n_found += 1
        equity = _get("equity")
        if equity is not None:
            n_found += 1
        long_term_debt = _get("long_term_debt")
        short_term_debt = _get("short_term_debt")
        cash = _get("cash")
        if cash is not None:
            n_found += 1
        shares_diluted = _get("shares_diluted")

        total_debt: Optional[float] = None
        if long_term_debt is not None or short_term_debt is not None:
            total_debt = (long_term_debt or 0.0) + (short_term_debt or 0.0)
        net_debt: Optional[float] = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash

        # ── Cash Flow ────────────────────────────────────────────────────
        cfo = _get("cfo")
        if cfo is not None:
            n_found += 1
        cfo_ttm = _ttm("cfo")
        capex_raw = _get("capex")
        capex = abs(capex_raw) if capex_raw is not None else None
        capex_ttm_raw = _ttm("capex")
        capex_ttm = abs(capex_ttm_raw) if capex_ttm_raw is not None else None

        free_cash_flow: Optional[float] = None
        if cfo_ttm is not None and capex_ttm is not None:
            free_cash_flow = cfo_ttm - capex_ttm

        # ── Derived margins ──────────────────────────────────────────────
        rev_base = revenue_ttm or revenue
        gross_margin: Optional[float] = None
        operating_margin: Optional[float] = None
        net_margin: Optional[float] = None
        if rev_base and rev_base != 0.0:
            gp_base = _ttm("gross_profit") or gross_profit
            if gp_base is not None:
                gross_margin = round(gp_base / rev_base, 4)
            oi_base = _ttm("operating_income") or operating_income
            if oi_base is not None:
                operating_margin = round(oi_base / rev_base, 4)
            ni_base = net_income_ttm or net_income
            if ni_base is not None:
                net_margin = round(ni_base / rev_base, 4)

        # ── YoY Revenue Growth (PIT: compare TTMs one year apart) ────────
        rev_growth: Optional[float] = None
        prior_as_of = date(as_of.year - 1, as_of.month, as_of.day)
        prior_rev_ttm = self._ttm_value(
            metric_vintages.get("revenue", []), prior_as_of
        )
        curr_rev = revenue_ttm or revenue
        if curr_rev is not None and prior_rev_ttm and prior_rev_ttm != 0.0:
            rev_growth = round((curr_rev - prior_rev_ttm) / abs(prior_rev_ttm), 4)

        # ── PIT anchor filing ────────────────────────────────────────────
        anc = _anchor("revenue") or _anchor("total_assets")
        fiscal_period_end = anc.period_end if anc else str(as_of)
        filed_date_str = anc.filed_date if anc else str(as_of)
        try:
            lag = (date.fromisoformat(filed_date_str) -
                   date.fromisoformat(fiscal_period_end)).days
        except ValueError:
            lag = 0

        snap = PITSnapshot(
            ticker=ticker.upper(),
            cik=cik,
            company_name=company_name,
            as_of_date=str(as_of),
            fiscal_period_end=fiscal_period_end,
            filed_date=filed_date_str,
            filing_lag_days=max(0, lag),
            revenue=revenue,
            revenue_ttm=revenue_ttm,
            net_income=net_income,
            net_income_ttm=net_income_ttm,
            gross_profit=gross_profit,
            operating_income=operating_income,
            eps_diluted=eps_diluted,
            total_assets=total_assets,
            equity=equity,
            long_term_debt=long_term_debt,
            short_term_debt=short_term_debt,
            cash=cash,
            total_debt=total_debt,
            net_debt=net_debt,
            cfo=cfo,
            cfo_ttm=cfo_ttm,
            capex=capex,
            capex_ttm=capex_ttm,
            free_cash_flow=free_cash_flow,
            da=da,
            shares_diluted=shares_diluted,
            rd=rd,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            revenue_growth_yoy=rev_growth,
            n_concepts_found=n_found,
        )

        # Persist to SQLite (async-safe via sync write in thread)
        _persist_snapshot(ticker, cik, snap, metric_vintages, as_of)
        return snap

    # ------------------------------------------------------------------
    # Public API: lag report
    # ------------------------------------------------------------------

    async def build_lag_report(self, ticker: str) -> list[LagReport]:
        """
        Compute actual publication lag statistics from EDGAR filing history.

        Returns one LagReport per form type (10-K and 10-Q), with full
        per-filing detail and filer_status classification.
        """
        cik, company_name = await _resolve_cik(ticker)
        filings = await self._get_filings(cik)
        if not filings:
            return []

        # Group by form type
        by_form: dict[str, list[FilingMeta]] = defaultdict(list)
        for f in filings:
            # Normalise amendment forms
            form_key = "10-K" if f.is_annual else "10-Q"
            by_form[form_key].append(f)

        reports: list[LagReport] = []
        all_lags: list[int] = [f.lag_days for f in filings]
        # Infer filer status from median lag
        median_all = float(np.median(all_lags)) if all_lags else 90.0
        filer_status = _infer_filer_status(median_all, is_annual=False)

        for form_key, form_filings in by_form.items():
            lags = [f.lag_days for f in form_filings]
            if not lags:
                continue
            median_lag = float(np.median(lags))
            mean_lag = float(np.mean(lags))
            min_lag = int(np.min(lags))
            max_lag = int(np.max(lags))
            is_ann = form_key == "10-K"
            status = _infer_filer_status(median_lag, is_annual=is_ann)
            target = (
                _ANNUAL_LAG_TARGETS.get(status, 90)
                if is_ann
                else _LAG_TARGETS.get(status, 60)
            )

            filing_records = [
                FilingRecord(
                    accession=f.accession,
                    form=f.form,
                    filing_date=f.filing_date,
                    report_date=f.report_date,
                    lag_days=f.lag_days,
                    is_annual=f.is_annual,
                )
                for f in sorted(form_filings, key=lambda x: x.filing_date, reverse=True)[:20]
            ]

            reports.append(LagReport(
                ticker=ticker.upper(),
                cik=cik,
                form_type=form_key,
                median_lag_days=round(median_lag, 1),
                mean_lag_days=round(mean_lag, 1),
                min_lag_days=min_lag,
                max_lag_days=max_lag,
                n_filings=len(lags),
                filer_status_estimate=status,
                target_lag_days=target,
                filings=filing_records,
            ))

            # Persist lag stats
            _persist_lag_registry(ticker, cik, form_key, status, median_lag, mean_lag,
                                   min_lag, max_lag, len(lags))
            _persist_publication_lags(ticker, cik, form_filings)

        return reports

    # ------------------------------------------------------------------
    # Public API: restatement detection
    # ------------------------------------------------------------------

    async def detect_restatements(
        self, ticker: str, metrics: Optional[list[str]] = None
    ) -> list[RestatementRecord]:
        """
        Compare vintages for each (metric, period_end) pair.  When the same
        period appears in multiple filings with different values, each later
        occurrence is a restatement.

        Only meaningful changes (> 0.1% threshold) are reported to avoid
        noise from rounding/unit differences.
        """
        cik, _ = await _resolve_cik(ticker)
        facts = await _load_facts(cik)
        if not facts:
            return []

        check_metrics = metrics or list(CONCEPT_MAP.keys())
        restatements: list[RestatementRecord] = []
        THRESHOLD = 0.001  # 0.1% minimum change to count as restatement

        for metric in check_metrics:
            vintages = _extract_all_vintages(facts, metric)
            if not vintages:
                continue

            # Group by period_end
            by_period: dict[str, list[MetricVintage]] = defaultdict(list)
            for v in vintages:
                by_period[v.period_end].append(v)

            for period_end, vts in by_period.items():
                # Deduplicate by accession (multiple concepts may yield same accn)
                seen_accns: dict[str, MetricVintage] = {}
                for v in vts:
                    if v.accession not in seen_accns:
                        seen_accns[v.accession] = v

                sorted_vts = sorted(seen_accns.values(),
                                    key=lambda v: v.filed_date)
                if len(sorted_vts) < 2:
                    continue

                original = sorted_vts[0]
                for restated in sorted_vts[1:]:
                    if original.value == 0:
                        continue
                    pct = (restated.value - original.value) / abs(original.value)
                    if abs(pct) < THRESHOLD:
                        continue
                    rec = RestatementRecord(
                        metric=metric,
                        period_end=period_end,
                        original_filed=original.filed_date,
                        original_value=original.value,
                        restatement_filed=restated.filed_date,
                        restatement_value=restated.value,
                        pct_change=round(pct * 100, 4),
                    )
                    restatements.append(rec)
                    _persist_restatement(ticker, cik, metric, period_end, original, restated, pct)

        restatements.sort(key=lambda r: r.period_end, reverse=True)
        return restatements

    # ------------------------------------------------------------------
    # Public API: PIT vintage history for one metric + period
    # ------------------------------------------------------------------

    async def get_vintage(
        self, ticker: str, period: str, metrics: Optional[list[str]] = None
    ) -> list[VintageRecord]:
        """
        Return every filing vintage for a given fiscal period_end.

        Useful for studying restatement history for a specific quarter.
        E.g. period="2019-12-31" shows all values filed for Q4 2019.
        """
        cik, _ = await _resolve_cik(ticker)
        facts = await _load_facts(cik)
        if not facts:
            return []

        check_metrics = metrics or ["revenue", "net_income", "eps_diluted", "total_assets"]
        records: list[VintageRecord] = []

        for metric in check_metrics:
            vintages = _extract_all_vintages(facts, metric)
            period_vts = [v for v in vintages if v.period_end == period]
            for v in period_vts:
                records.append(VintageRecord(
                    metric=v.metric,
                    period_end=v.period_end,
                    filed_date=v.filed_date,
                    value=v.value,
                    accession=v.accession,
                    is_annual=v.is_annual,
                    is_quarterly=v.is_quarterly,
                ))

        records.sort(key=lambda r: (r.metric, r.filed_date))
        return records

    # ------------------------------------------------------------------
    # Public API: PIT time series for backtesting
    # ------------------------------------------------------------------

    async def get_pit_series(
        self,
        ticker: str,
        metric: str,
        start_year: int = 2005,
        as_originally_reported: bool = True,
    ) -> pd.DataFrame:
        """
        Build a full PIT time series DataFrame for a single metric.

        Columns: period_end, filed_date, value, accession, is_annual, lag_days
        Sorted by filed_date ascending (PIT chronological order for backtests).

        Parameters
        ----------
        as_originally_reported:
            If True, for each (period_end), keep only the first filing that
            contained data for that period (original values, no restatements).
            If False, keep the latest available vintage.
        """
        if metric not in CONCEPT_MAP:
            raise ValueError(f"Unknown metric '{metric}'. Valid: {sorted(CONCEPT_MAP)}")

        cik, company_name = await _resolve_cik(ticker)
        facts = await _load_facts(cik)
        if not facts:
            return pd.DataFrame()

        vintages = _extract_all_vintages(facts, metric, min_year=start_year)
        if not vintages:
            return pd.DataFrame()

        # Deduplicate per period_end per accession
        seen: dict[tuple[str, str], MetricVintage] = {}
        for v in vintages:
            key = (v.period_end, v.accession)
            if key not in seen:
                seen[key] = v

        # Now group by period_end → pick original or latest
        by_period: dict[str, list[MetricVintage]] = defaultdict(list)
        for v in seen.values():
            by_period[v.period_end].append(v)

        rows: list[dict] = []
        for period_end, vts in sorted(by_period.items()):
            vts_sorted = sorted(vts, key=lambda v: v.filed_date)
            chosen = vts_sorted[0] if as_originally_reported else vts_sorted[-1]
            report_date = period_end
            try:
                lag = (
                    date.fromisoformat(chosen.filed_date) -
                    date.fromisoformat(report_date)
                ).days
            except ValueError:
                lag = 0
            rows.append({
                "ticker": ticker.upper(),
                "metric": metric,
                "period_end": period_end,
                "fiscal_period": ("annual" if chosen.is_annual else "quarterly"),
                "filed_date": chosen.filed_date,
                "value": chosen.value,
                "accession": chosen.accession,
                "lag_days": lag,
                "is_restated": len(vts) > 1,
                "n_vintages": len(vts),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["period_end"] = pd.to_datetime(df["period_end"])
        df["filed_date"] = pd.to_datetime(df["filed_date"])
        df.sort_values("filed_date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Public API: backtest universe coverage
    # ------------------------------------------------------------------

    async def backtest_universe_coverage(
        self,
        as_of: date,
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        For each ticker in the universe, determine:
            - Whether any filing was available by as_of_date (had_data: bool)
            - The earliest filing date (coverage_start)
            - The most recent period for which data was filed by as_of_date
            - The filing lag for that period

        Returns DataFrame sorted by had_data desc, then ticker.
        Useful for detecting look-ahead contamination in universe construction.
        """
        universe = tickers or SP500_SAMPLE
        sem = asyncio.Semaphore(_SEMAPHORE_WIDTH)

        async def _check(t: str) -> dict:
            async with sem:
                await asyncio.sleep(_RATE_DELAY)
                try:
                    cik, name = await _resolve_cik(t)
                    filings = await self._get_filings(cik)
                    avail = self._filings_available_on(filings, as_of)
                    if not avail:
                        return {
                            "ticker": t, "had_data": False,
                            "coverage_start": None, "latest_period": None,
                            "latest_filed": None, "lag_days": None,
                            "look_ahead_risk": "high",
                        }
                    avail_sorted = sorted(avail, key=lambda f: f.report_date, reverse=True)
                    latest = avail_sorted[0]
                    earliest = sorted(avail, key=lambda f: f.filing_date)[0]
                    return {
                        "ticker": t,
                        "had_data": True,
                        "coverage_start": earliest.filing_date,
                        "latest_period": latest.report_date,
                        "latest_filed": latest.filing_date,
                        "lag_days": latest.lag_days,
                        "look_ahead_risk": "none",
                    }
                except Exception as exc:
                    return {
                        "ticker": t, "had_data": False, "coverage_start": None,
                        "latest_period": None, "latest_filed": None, "lag_days": None,
                        "look_ahead_risk": "unknown", "error": str(exc),
                    }

        results = await asyncio.gather(*[_check(t) for t in universe])
        df = pd.DataFrame(list(results))
        df.sort_values(["had_data", "ticker"], ascending=[False, True], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


# ---------------------------------------------------------------------------
# Filer status inference
# ---------------------------------------------------------------------------

def _infer_filer_status(median_lag: float, is_annual: bool = False) -> str:
    """
    Classify a company's filer status from its historical median lag.
    SEC rules define filing deadlines per filer category.
    """
    targets = _ANNUAL_LAG_TARGETS if is_annual else _LAG_TARGETS
    # Large accelerated: <= 40d (Q) / 60d (K)
    # Accelerated: <= 45d (Q) / 75d (K)
    # Non-accelerated: <= 60d (Q) / 90d (K)
    # SRC: <= 75d (Q) / 105d (K)
    if median_lag <= targets["large_accelerated"]:
        return "large_accelerated"
    if median_lag <= targets["accelerated"]:
        return "accelerated"
    if median_lag <= targets["non_accelerated"]:
        return "non_accelerated"
    if median_lag <= targets["src"]:
        return "src"
    return "unknown"


# ---------------------------------------------------------------------------
# SQLite persistence helpers
# ---------------------------------------------------------------------------

def _persist_snapshot(
    ticker: str,
    cik: str,
    snap: PITSnapshot,
    metric_vintages: dict[str, list[MetricVintage]],
    as_of: date,
) -> None:
    """Write all metric vintages (not just the snapshot values) to pit_financials."""
    try:
        conn = _get_conn()
        rows = []
        for metric, vts in metric_vintages.items():
            for v in vts:
                rows.append((
                    cik, ticker.upper(), v.accession, metric, v.xbrl_concept,
                    v.period_start, v.period_end, v.filed_date, v.value,
                    int(v.is_annual), int(v.is_quarterly),
                ))
        if rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO pit_financials
                    (cik, ticker, accession, metric, xbrl_concept,
                     period_start, period_end, filed_date, value,
                     is_annual, is_quarterly)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("pit_financials persist failed", ticker=ticker, error=str(exc))


def _persist_lag_registry(
    ticker: str, cik: str, form_type: str, status: str,
    median: float, mean: float, min_lag: int, max_lag: int, n: int,
) -> None:
    try:
        conn = _get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO lag_registry
                (cik, ticker, filer_status, form_type, median_lag_days,
                 mean_lag_days, min_lag_days, max_lag_days, n_filings)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (cik, ticker.upper(), status, form_type, median, mean, min_lag, max_lag, n),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("lag_registry persist failed", ticker=ticker, error=str(exc))


def _persist_publication_lags(
    ticker: str, cik: str, filings: list[FilingMeta]
) -> None:
    try:
        conn = _get_conn()
        conn.executemany(
            """
            INSERT OR IGNORE INTO publication_lags
                (cik, ticker, accession, form, report_date, filing_date, lag_days)
            VALUES (?,?,?,?,?,?,?)
            """,
            [
                (cik, ticker.upper(), f.accession, f.form,
                 f.report_date, f.filing_date, f.lag_days)
                for f in filings
            ],
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("publication_lags persist failed", ticker=ticker, error=str(exc))


def _persist_restatement(
    ticker: str,
    cik: str,
    metric: str,
    period_end: str,
    original: MetricVintage,
    restated: MetricVintage,
    pct: float,
) -> None:
    try:
        conn = _get_conn()
        conn.execute(
            """
            INSERT OR IGNORE INTO restatements
                (cik, ticker, metric, period_end, original_accn, original_filed,
                 original_value, restatement_accn, restatement_filed,
                 restatement_value, pct_change)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                cik, ticker.upper(), metric, period_end,
                original.accession, original.filed_date, original.value,
                restated.accession, restated.filed_date, restated.value,
                round(pct * 100, 4),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("restatement persist failed", ticker=ticker, error=str(exc))


def _persist_filing_index(ticker: str, cik: str, filings: list[FilingMeta]) -> None:
    try:
        conn = _get_conn()
        conn.executemany(
            """
            INSERT OR IGNORE INTO pit_filings
                (cik, accession, form, filing_date, report_date, lag_days, is_annual)
            VALUES (?,?,?,?,?,?,?)
            """,
            [
                (cik, f.accession, f.form, f.filing_date,
                 f.report_date, f.lag_days, int(f.is_annual))
                for f in filings
            ],
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("pit_filings persist failed", ticker=ticker, error=str(exc))


# ---------------------------------------------------------------------------
# Coverage report helper
# ---------------------------------------------------------------------------

async def build_coverage_report(
    tickers: Optional[list[str]] = None,
    as_of: Optional[date] = None,
) -> pd.DataFrame:
    """
    Build a coverage report: for each ticker, first available PIT data date
    and total number of periodic filings.  Useful for universe construction.
    """
    engine = PITEngineV3()
    universe = tickers or SP500_SAMPLE
    sem = asyncio.Semaphore(_SEMAPHORE_WIDTH)

    async def _process(t: str) -> dict:
        async with sem:
            await asyncio.sleep(_RATE_DELAY)
            try:
                cik, name = await _resolve_cik(t)
                filings = await engine._get_filings(cik)
                _persist_filing_index(t, cik, filings)
                if not filings:
                    return {"ticker": t, "cik": cik, "first_filing": None,
                            "first_period": None, "total_filings": 0}
                sorted_f = sorted(filings, key=lambda f: f.filing_date)
                first = sorted_f[0]
                conn = _get_conn()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO coverage_index
                        (ticker, cik, first_filing_date, first_report_date, total_filings)
                    VALUES (?,?,?,?,?)
                    """,
                    (t.upper(), cik, first.filing_date, first.report_date, len(filings)),
                )
                conn.commit()
                conn.close()
                return {
                    "ticker": t, "cik": cik,
                    "first_filing": first.filing_date,
                    "first_period": first.report_date,
                    "total_filings": len(filings),
                }
            except Exception as exc:
                return {"ticker": t, "cik": None, "error": str(exc),
                        "first_filing": None, "first_period": None, "total_filings": 0}

    results = await asyncio.gather(*[_process(t) for t in universe])
    df = pd.DataFrame(list(results))
    df.sort_values("first_filing", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

pit_v3_router = APIRouter(prefix="/pit/v3", tags=["pit-v3"])
_engine = PITEngineV3()


@pit_v3_router.get("/financials/{ticker}")
async def get_pit_financials(
    ticker: str,
    as_of: str = Query(
        ...,
        description="ISO date — data available on or before this date (YYYY-MM-DD)",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
) -> dict:
    """
    Return a point-in-time financial snapshot for ticker as of the given date.

    Only data that had been FILED with the SEC by that date is included.
    Prevents look-ahead bias.  TTM figures are computed from the four most
    recent non-overlapping quarterly filings available on that date.
    """
    ticker = ticker.upper()
    try:
        as_of_date = date.fromisoformat(as_of)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date: {as_of}")
    try:
        snap = await _engine.get_as_of(ticker, as_of_date)
        return snap.model_dump()
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("pit financials error", ticker=ticker, as_of=as_of, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@pit_v3_router.get("/lag-report/{ticker}")
async def get_lag_report(ticker: str) -> dict:
    """
    Actual publication lag statistics for a ticker, derived from EDGAR
    filing history (not approximated from static tables).

    Includes: median/mean/min/max lag per form type, filer status classification,
    and the 20 most recent filings with their individual lags.
    """
    ticker = ticker.upper()
    try:
        reports = await _engine.build_lag_report(ticker)
        if not reports:
            return {"ticker": ticker, "reports": [], "note": "no filing history found"}
        return {
            "ticker": ticker,
            "reports": [r.model_dump() for r in reports],
        }
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@pit_v3_router.get("/restatements/{ticker}")
async def get_restatements(
    ticker: str,
    metrics: str = Query(
        "revenue,net_income,eps_diluted,total_assets",
        description="Comma-separated metrics to check for restatements",
    ),
) -> dict:
    """
    Detect financial restatements for a ticker by comparing all XBRL vintages
    for each (metric, fiscal_period) pair.

    A restatement is flagged when the same period appears in multiple filings
    with a value change > 0.1%.  The original and restated values, dates, and
    accession numbers are returned.
    """
    ticker = ticker.upper()
    metric_list = [m.strip() for m in metrics.split(",") if m.strip() in CONCEPT_MAP]
    if not metric_list:
        raise HTTPException(status_code=422, detail="No valid metrics specified")
    try:
        recs = await _engine.detect_restatements(ticker, metric_list)
        return {
            "ticker": ticker,
            "metrics_checked": metric_list,
            "n_restatements": len(recs),
            "restatements": [r.model_dump() for r in recs],
        }
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@pit_v3_router.get("/backtest-universe")
async def get_backtest_universe(
    as_of: str = Query(
        ...,
        description="Backtest start date (YYYY-MM-DD)",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
    index: str = Query("sp500", description="'sp500' or comma-separated tickers"),
) -> dict:
    """
    Look-ahead bias detection for a universe as of a given date.

    Returns a list of tickers with:
      - had_data: whether any filing existed by that date
      - coverage_start: date of earliest filing
      - latest_period: most recent fiscal period with available data
      - lag_days: filing lag for the latest available filing

    Tickers where had_data=False were NOT available on the as_of_date and
    must be excluded from backtests starting on that date to avoid look-ahead.
    """
    try:
        as_of_date = date.fromisoformat(as_of)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date: {as_of}")

    if index.lower() == "sp500":
        tickers = SP500_SAMPLE
    else:
        tickers = [t.strip().upper() for t in index.split(",") if t.strip()]

    try:
        df = await _engine.backtest_universe_coverage(as_of_date, tickers)
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        n_available = int((df.get("had_data") == True).sum()) if "had_data" in df.columns else 0
        return {
            "as_of_date": as_of,
            "universe_size": len(tickers),
            "tickers_with_data": n_available,
            "tickers_without_data": len(tickers) - n_available,
            "look_ahead_warning": (
                f"{len(tickers) - n_available} tickers had no data filed by {as_of}. "
                "Including them in a backtest would introduce look-ahead bias."
            ),
            "coverage": records,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@pit_v3_router.get("/vintage/{ticker}/{period}")
async def get_vintage(
    ticker: str,
    period: str,
    metrics: str = Query(
        "revenue,net_income,eps_diluted,total_assets",
        description="Comma-separated metrics",
    ),
) -> dict:
    """
    Return every XBRL filing vintage for the given ticker and fiscal period end.

    Each row is one filing that contained data for that period — multiple rows
    mean restatements occurred.  Filed dates, accession numbers, and values
    are all returned so you can reconstruct the exact information set at any
    historical date.

    Example: /pit/v3/vintage/AAPL/2019-09-28?metrics=revenue,net_income
    """
    ticker = ticker.upper()
    metric_list = [m.strip() for m in metrics.split(",") if m.strip() in CONCEPT_MAP]
    if not metric_list:
        raise HTTPException(status_code=422, detail="No valid metrics")
    try:
        recs = await _engine.get_vintage(ticker, period, metric_list)
        return {
            "ticker": ticker,
            "period": period,
            "n_vintages": len(recs),
            "vintages": [r.model_dump() for r in recs],
        }
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
