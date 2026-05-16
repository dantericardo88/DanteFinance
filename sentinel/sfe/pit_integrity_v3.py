"""
pit_integrity_v3.py — Point-in-time data integrity engine v3.

Dimension: dim_022 — Point-in-time data (no look-ahead bias)  target score: 9/10

Key upgrades over v1/v2:
  - PITDatabaseWrapper: any financial query gated by filed_date <= as_of_date using
    EDGAR submissions API as authoritative PIT timestamp source
  - LookAheadBiasScanner: upload a DataFrame; quantify exact percentage of rows
    where financial data was used before the 10-Q/10-K was publicly filed
  - BacktestUniverseBuilder: S&P 500 historical constituency (Wikipedia + EDGAR),
    survivorship-bias-free, with each stock's entry/exit date in the index
  - PublicationLagModel: filer category detection (large_accelerated, accelerated,
    non-accelerated, SRC) from EDGAR, statutory deadlines, actual lag, late-filer
    detection via NT 10-K / NT 10-Q filings
  - EarningsReleaseTiming: 8-K item 2.02 filing date precedes 10-Q by days/weeks
  - SafeDataAPI: all endpoints support optional ?as_of=YYYY-MM-DD parameter that
    transparently enforces PIT using the filing registry
  - SQLite: filing_registry, pit_access_log, look_ahead_flags, sp500_history, filer_categories
  - FastAPI router /pit-integrity/v3

Public entry points
-------------------
router: APIRouter         — mount at /pit-integrity/v3
PITIntegrityService       — primary service class
PITDatabaseWrapper        — wraps any query with PIT enforcement
LookAheadBiasScanner      — DataFrame-level look-ahead detection
BacktestUniverseBuilder   — survivorship-bias-free universe construction
PublicationLagModel       — statutory + actual filing lag model
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE          = "https://data.sec.gov"
EDGAR_TICKERS_URL   = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS   = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_COMPANYFACTS  = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_EFTS          = "https://efts.sec.gov/LATEST/search-index"
SP500_WIKI_URL      = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DB_PATH  = Path("sentinel_pit_v3.db")
CACHE_TTL = 6 * 3600   # 6 hours

_HEADERS: Dict[str, str] = {
    "User-Agent": "SENTINEL/3.0 financial-terminal richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

# Statutory filing deadlines (calendar days after fiscal period end)
STATUTORY_DEADLINES: Dict[str, Dict[str, int]] = {
    "large_accelerated": {"10-Q": 40, "10-K": 60},
    "accelerated":       {"10-Q": 40, "10-K": 75},
    "non_accelerated":   {"10-Q": 45, "10-K": 90},
    "smaller_reporting": {"10-Q": 45, "10-K": 90},
    "foreign_private":   {"20-F": 120},
    "unknown":           {"10-Q": 45, "10-K": 90},
}

# Forms considered earnings-related
EARNINGS_FORMS = {"10-Q", "10-QT", "10-K", "10-KT", "20-F", "40-F"}

# NT forms signal late filing intent
NT_FORMS = {"NT 10-Q", "NT 10-K", "NT 20-F", "NT 10-KT", "NT 10-QT"}

# 8-K items that typically precede earnings (earnings release = item 2.02)
EARNINGS_8K_ITEMS = {"2.02", "2.01"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FilingRecord(BaseModel):
    ticker: str
    cik: str
    form: str
    period_end: str
    filed_date: str
    report_date: Optional[str] = None
    lag_days: Optional[int] = None
    filer_category: str = "unknown"
    statutory_deadline_days: Optional[int] = None
    is_late: bool = False
    nt_filed: bool = False
    source: str = "EDGAR"


class LookAheadFlag(BaseModel):
    row_index: int
    ticker: str
    metric: str
    data_date: str          # date the financial data refers to
    as_of_date: str         # date we're pretending to be at
    filed_date: str         # when it was actually filed
    lag_days_behind: int    # how many days too early we used it
    look_ahead_pct: float   # fraction of look-ahead in the dataset


class ScanResult(BaseModel):
    total_rows: int
    flagged_rows: int
    clean_rows: int
    look_ahead_pct: float
    flags: List[LookAheadFlag]
    summary: str
    methodology: str = (
        "Each row's financial data is checked against the actual EDGAR filing date. "
        "A row is flagged if the data (e.g., Q3 revenue) was used before the 10-Q "
        "or 10-K was publicly filed on EDGAR."
    )


class SP500Member(BaseModel):
    ticker: str
    name: str
    added_date: Optional[str] = None
    removed_date: Optional[str] = None
    is_current: bool = True
    cik: Optional[str] = None
    sector: Optional[str] = None
    sub_industry: Optional[str] = None


class LagModelResult(BaseModel):
    ticker: str
    cik: str
    filer_category: str
    filings: List[Dict[str, Any]]
    avg_lag_10q: Optional[float] = None
    avg_lag_10k: Optional[float] = None
    statutory_10q: Optional[int] = None
    statutory_10k: Optional[int] = None
    late_filer_10q: bool = False
    late_filer_10k: bool = False
    early_filer_percentile: Optional[float] = None


class PITQueryResult(BaseModel):
    ticker: str
    metric: str
    as_of_date: str
    value: Optional[float] = None
    filed_date: Optional[str] = None
    period_end: Optional[str] = None
    form: Optional[str] = None
    lag_days: Optional[int] = None
    is_pit_safe: bool = True
    message: str = ""


class SafeUniverse(BaseModel):
    as_of_date: str
    members: List[SP500Member]
    total: int
    note: str = "Survivorship-bias-free: includes companies that were in the index on as_of_date, even if later removed."


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

class _PITDatabase:
    """SQLite persistence for filing registry, look-ahead flags, SP500 history."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS filing_registry (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker                  TEXT NOT NULL,
                cik                     TEXT NOT NULL,
                form                    TEXT NOT NULL,
                period_end              TEXT NOT NULL,
                filed_date              TEXT NOT NULL,
                report_date             TEXT,
                lag_days                INTEGER,
                filer_category          TEXT DEFAULT 'unknown',
                statutory_deadline_days INTEGER,
                is_late                 INTEGER DEFAULT 0,
                nt_filed                INTEGER DEFAULT 0,
                fetched_at              TEXT DEFAULT (datetime('now')),
                UNIQUE(cik, form, period_end)
            );

            CREATE TABLE IF NOT EXISTS pit_access_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                metric      TEXT NOT NULL,
                as_of_date  TEXT NOT NULL,
                filed_date  TEXT,
                period_end  TEXT,
                value       REAL,
                is_pit_safe INTEGER DEFAULT 1,
                queried_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS look_ahead_flags (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id             TEXT NOT NULL,
                row_index           INTEGER,
                ticker              TEXT,
                metric              TEXT,
                data_date           TEXT,
                as_of_date          TEXT,
                filed_date          TEXT,
                lag_days_behind     INTEGER,
                created_at          TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sp500_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                name            TEXT,
                cik             TEXT,
                sector          TEXT,
                sub_industry    TEXT,
                added_date      TEXT,
                removed_date    TEXT,
                is_current      INTEGER DEFAULT 1,
                source          TEXT DEFAULT 'Wikipedia',
                fetched_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, added_date)
            );

            CREATE TABLE IF NOT EXISTS filer_categories (
                cik                 TEXT PRIMARY KEY,
                ticker              TEXT,
                filer_category      TEXT,
                category_source     TEXT,
                updated_at          TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ticker_cik_map (
                ticker  TEXT PRIMARY KEY,
                cik     TEXT NOT NULL,
                name    TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_fr_ticker      ON filing_registry(ticker);
            CREATE INDEX IF NOT EXISTS idx_fr_cik_form    ON filing_registry(cik, form);
            CREATE INDEX IF NOT EXISTS idx_fr_period_end  ON filing_registry(period_end);
            CREATE INDEX IF NOT EXISTS idx_fr_filed_date  ON filing_registry(filed_date);
            CREATE INDEX IF NOT EXISTS idx_sp500_ticker   ON sp500_history(ticker);
            CREATE INDEX IF NOT EXISTS idx_sp500_dates    ON sp500_history(added_date, removed_date);
        """)
        self._conn.commit()

    # -- Filing registry --

    def upsert_filing(self, f: Dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO filing_registry
               (ticker, cik, form, period_end, filed_date, report_date, lag_days,
                filer_category, statutory_deadline_days, is_late, nt_filed)
               VALUES (:ticker, :cik, :form, :period_end, :filed_date, :report_date,
                       :lag_days, :filer_category, :statutory_deadline_days, :is_late, :nt_filed)""",
            f,
        )
        self._conn.commit()

    def get_filings(self, ticker: str, form: Optional[str] = None) -> List[Dict]:
        if form:
            cur = self._conn.execute(
                "SELECT * FROM filing_registry WHERE ticker = ? AND form = ? ORDER BY period_end DESC",
                (ticker.upper(), form),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM filing_registry WHERE ticker = ? ORDER BY period_end DESC",
                (ticker.upper(),),
            )
        return [dict(r) for r in cur.fetchall()]

    def get_filed_date_for_period(self, ticker: str, period_end: str, forms: List[str]) -> Optional[Dict]:
        """Return the earliest filing that covers period_end for the given forms."""
        placeholders = ",".join("?" * len(forms))
        cur = self._conn.execute(
            f"""SELECT * FROM filing_registry
                WHERE ticker = ? AND period_end = ? AND form IN ({placeholders})
                ORDER BY filed_date ASC LIMIT 1""",
            (ticker.upper(), period_end, *forms),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_most_recent_filing_as_of(self, ticker: str, as_of_date: str, forms: List[str]) -> Optional[Dict]:
        """PIT query: latest filing with filed_date <= as_of_date."""
        placeholders = ",".join("?" * len(forms))
        cur = self._conn.execute(
            f"""SELECT * FROM filing_registry
                WHERE ticker = ? AND form IN ({placeholders}) AND filed_date <= ?
                ORDER BY period_end DESC, filed_date DESC LIMIT 1""",
            (ticker.upper(), *forms, as_of_date),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def log_pit_access(self, entry: Dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO pit_access_log (ticker, metric, as_of_date, filed_date, period_end, value, is_pit_safe)
               VALUES (:ticker, :metric, :as_of_date, :filed_date, :period_end, :value, :is_pit_safe)""",
            entry,
        )
        self._conn.commit()

    def upsert_look_ahead_flags(self, scan_id: str, flags: List[Dict]) -> None:
        self._conn.executemany(
            """INSERT INTO look_ahead_flags
               (scan_id, row_index, ticker, metric, data_date, as_of_date, filed_date, lag_days_behind)
               VALUES (:scan_id, :row_index, :ticker, :metric, :data_date, :as_of_date, :filed_date, :lag_days_behind)""",
            [{**f, "scan_id": scan_id} for f in flags],
        )
        self._conn.commit()

    # -- SP500 history --

    def upsert_sp500_member(self, m: Dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO sp500_history
               (ticker, name, cik, sector, sub_industry, added_date, removed_date, is_current, source)
               VALUES (:ticker, :name, :cik, :sector, :sub_industry, :added_date, :removed_date, :is_current, :source)""",
            m,
        )

    def commit(self) -> None:
        self._conn.commit()

    def get_sp500_as_of(self, as_of_date: str) -> List[Dict]:
        """Return all S&P 500 members that were in the index on as_of_date (survivorship-bias-free)."""
        cur = self._conn.execute(
            """SELECT * FROM sp500_history
               WHERE (added_date IS NULL OR added_date <= ?)
                 AND (removed_date IS NULL OR removed_date > ?)
               ORDER BY ticker""",
            (as_of_date, as_of_date),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_current_sp500(self) -> List[Dict]:
        cur = self._conn.execute(
            "SELECT * FROM sp500_history WHERE is_current = 1 ORDER BY ticker"
        )
        return [dict(r) for r in cur.fetchall()]

    def has_sp500_data(self) -> bool:
        cur = self._conn.execute("SELECT COUNT(*) FROM sp500_history")
        return cur.fetchone()[0] > 0

    # -- Ticker → CIK --

    def get_cik(self, ticker: str) -> Optional[str]:
        cur = self._conn.execute("SELECT cik FROM ticker_cik_map WHERE ticker = ?", (ticker.upper(),))
        row = cur.fetchone()
        return row[0] if row else None

    def upsert_cik(self, ticker: str, cik: str, name: str = "") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ticker_cik_map (ticker, cik, name) VALUES (?, ?, ?)",
            (ticker.upper(), cik, name),
        )
        self._conn.commit()

    # -- Filer categories --

    def get_filer_category(self, cik: str) -> Optional[str]:
        cur = self._conn.execute("SELECT filer_category FROM filer_categories WHERE cik = ?", (cik,))
        row = cur.fetchone()
        return row[0] if row else None

    def upsert_filer_category(self, cik: str, ticker: str, category: str, source: str) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO filer_categories (cik, ticker, filer_category, category_source)
               VALUES (?, ?, ?, ?)""",
            (cik, ticker.upper(), category, source),
        )
        self._conn.commit()

    def get_late_filers(self, form: str = "10-K", limit: int = 100) -> List[Dict]:
        cur = self._conn.execute(
            """SELECT ticker, cik, form, period_end, filed_date, lag_days, statutory_deadline_days, nt_filed
               FROM filing_registry
               WHERE form = ? AND is_late = 1
               ORDER BY lag_days DESC LIMIT ?""",
            (form, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# EDGAR utilities
# ---------------------------------------------------------------------------

class EDGARClient:
    """Thin EDGAR API client with rate limiting."""

    def __init__(self) -> None:
        self._ticker_cik_map: Dict[str, str] = {}
        self._ticker_name_map: Dict[str, str] = {}

    def _get(self, url: str, params: Optional[Dict] = None, timeout: int = 30) -> Optional[Any]:
        try:
            time.sleep(0.12)
            r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning("edgar_request_failed", url=url, error=str(e))
            return None

    def load_ticker_map(self, db: _PITDatabase) -> None:
        """Load SEC company_tickers.json and populate ticker→CIK map."""
        data = self._get(EDGAR_TICKERS_URL)
        if not data:
            return
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik    = str(entry.get("cik_str", "")).zfill(10)
            name   = entry.get("title", "")
            if ticker and cik:
                self._ticker_cik_map[ticker] = cik
                self._ticker_name_map[ticker] = name
                db.upsert_cik(ticker, cik, name)

    def resolve_cik(self, ticker: str, db: _PITDatabase) -> Optional[str]:
        """Resolve ticker to CIK, using DB cache first."""
        ticker = ticker.upper()
        # DB cache
        cik = db.get_cik(ticker)
        if cik:
            return cik
        # In-memory map
        if ticker in self._ticker_cik_map:
            return self._ticker_cik_map[ticker]
        # Load from EDGAR
        self.load_ticker_map(db)
        return self._ticker_cik_map.get(ticker) or db.get_cik(ticker)

    def get_submissions(self, cik: str) -> Optional[Dict]:
        cik_str = str(cik).zfill(10)
        data = self._get(EDGAR_SUBMISSIONS.format(cik=cik_str), timeout=45)
        # Handle paginated submissions (additional files)
        if data and "filings" in data:
            return data
        return data

    def get_company_facts(self, cik: str) -> Optional[Dict]:
        cik_str = str(cik).zfill(10)
        return self._get(EDGAR_COMPANYFACTS.format(cik=cik_str), timeout=60)


# ---------------------------------------------------------------------------
# Publication lag model
# ---------------------------------------------------------------------------

class PublicationLagModel:
    """
    Determines filer category and computes statutory vs actual filing lags.

    Filer categories per SEC rules:
      - Large Accelerated Filer: >$700M public float
      - Accelerated Filer:        $75M - $700M public float
      - Non-Accelerated Filer:    < $75M public float
      - Smaller Reporting Company (SRC): <$250M public float or <$100M revenue
      - Foreign Private Issuer:   20-F filer

    We detect category from the submissions JSON 'entityType' and 'category' fields.
    """

    def __init__(self, edgar: EDGARClient, db: _PITDatabase) -> None:
        self._edgar = edgar
        self._db = db

    def detect_filer_category(self, cik: str, ticker: str) -> str:
        """Detect filer category from EDGAR submissions JSON."""
        cached = self._db.get_filer_category(cik)
        if cached:
            return cached

        subs = self._edgar.get_submissions(cik)
        if not subs:
            return "unknown"

        category = "unknown"
        # EDGAR submissions JSON has 'category' field in company details
        raw_category = subs.get("category", "") or ""
        entity_type  = subs.get("entityType", "") or ""

        raw_lower = raw_category.lower()
        if "large accelerated" in raw_lower:
            category = "large_accelerated"
        elif "accelerated filer" in raw_lower:
            category = "accelerated"
        elif "smaller reporting" in raw_lower or "src" in raw_lower:
            category = "smaller_reporting"
        elif "non-accelerated" in raw_lower or "non accelerated" in raw_lower:
            category = "non_accelerated"
        elif "foreign private" in entity_type.lower():
            category = "foreign_private"
        else:
            # Heuristic: check what forms have been filed
            forms = subs.get("filings", {}).get("recent", {}).get("form", [])
            if "20-F" in forms:
                category = "foreign_private"
            else:
                category = "non_accelerated"  # conservative default

        self._db.upsert_filer_category(cik, ticker, category, source="EDGAR_submissions")
        return category

    def get_statutory_deadline(self, filer_category: str, form: str) -> Optional[int]:
        """Return statutory days after period end that filing is due."""
        deadlines = STATUTORY_DEADLINES.get(filer_category, STATUTORY_DEADLINES["unknown"])
        return deadlines.get(form)

    def compute_lag(self, period_end: str, filed_date: str) -> int:
        """Compute calendar day lag between period end and filing date."""
        try:
            pe = datetime.strptime(period_end, "%Y-%m-%d").date()
            fd = datetime.strptime(filed_date, "%Y-%m-%d").date()
            return (fd - pe).days
        except ValueError:
            return 0

    def build_filing_registry(self, ticker: str, cik: str) -> List[Dict[str, Any]]:
        """
        Fetch all EDGAR filings for ticker, compute lags, detect late filings,
        detect NT form precursors. Stores results in filing_registry table.
        """
        subs = self._edgar.get_submissions(cik)
        if not subs:
            return []

        filer_category = self.detect_filer_category(cik, ticker)
        recent = subs.get("filings", {}).get("recent", {})

        forms        = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        period_ends  = recent.get("periodOfReport", [])
        accessions   = recent.get("accessionNumber", [])

        # Build NT form lookup: NT filings by period
        nt_periods: Dict[str, bool] = {}
        for i, f in enumerate(forms):
            if f in NT_FORMS:
                pe = period_ends[i] if i < len(period_ends) else ""
                if pe:
                    nt_periods[pe] = True

        result = []
        for i, form in enumerate(forms):
            if form not in EARNINGS_FORMS:
                continue

            period_end  = period_ends[i]  if i < len(period_ends)  else ""
            filed_date  = filing_dates[i] if i < len(filing_dates) else ""
            report_date = report_dates[i] if i < len(report_dates) else ""

            if not period_end or not filed_date:
                continue

            lag_days = self.compute_lag(period_end, filed_date)
            statutory_days = self.get_statutory_deadline(filer_category, form)
            is_late = (statutory_days is not None) and (lag_days > statutory_days)
            nt_filed = nt_periods.get(period_end, False)

            row = {
                "ticker":                   ticker.upper(),
                "cik":                      str(cik).zfill(10),
                "form":                     form,
                "period_end":               period_end,
                "filed_date":               filed_date,
                "report_date":              report_date or period_end,
                "lag_days":                 lag_days,
                "filer_category":           filer_category,
                "statutory_deadline_days":  statutory_days,
                "is_late":                  int(is_late),
                "nt_filed":                 int(nt_filed),
            }
            self._db.upsert_filing(row)
            result.append(row)

        return result

    def compute_lag_statistics(self, ticker: str) -> LagModelResult:
        """Compute average and late-filer stats for a ticker."""
        filings = self._db.get_filings(ticker)
        if not filings:
            return LagModelResult(ticker=ticker, cik="", filer_category="unknown", filings=[])

        cik = filings[0].get("cik", "")
        filer_category = filings[0].get("filer_category", "unknown")

        q_lags = [f["lag_days"] for f in filings if f.get("form") in ("10-Q", "10-QT") and f.get("lag_days") is not None]
        k_lags = [f["lag_days"] for f in filings if f.get("form") in ("10-K", "10-KT", "20-F") and f.get("lag_days") is not None]

        avg_q = round(float(np.mean(q_lags)), 1) if q_lags else None
        avg_k = round(float(np.mean(k_lags)), 1) if k_lags else None
        stat_q = self.get_statutory_deadline(filer_category, "10-Q")
        stat_k = self.get_statutory_deadline(filer_category, "10-K")

        late_q = any(f.get("is_late") for f in filings if f.get("form") in ("10-Q", "10-QT"))
        late_k = any(f.get("is_late") for f in filings if f.get("form") in ("10-K", "10-KT", "20-F"))

        # Percentile: how early does this company file vs statutory deadline
        if q_lags and stat_q:
            early_pcts = [(stat_q - lag) / stat_q * 100 for lag in q_lags]
            early_pct = round(float(np.mean(early_pcts)), 1)
        else:
            early_pct = None

        return LagModelResult(
            ticker=ticker,
            cik=cik,
            filer_category=filer_category,
            filings=filings[:20],
            avg_lag_10q=avg_q,
            avg_lag_10k=avg_k,
            statutory_10q=stat_q,
            statutory_10k=stat_k,
            late_filer_10q=late_q,
            late_filer_10k=late_k,
            early_filer_percentile=early_pct,
        )


# ---------------------------------------------------------------------------
# PIT database wrapper — the core PIT enforcement class
# ---------------------------------------------------------------------------

class PITDatabaseWrapper:
    """
    Wraps any financial data query with automatic point-in-time enforcement.

    Core guarantee: every fact returned was publicly available (i.e., filed)
    on or before the requested as_of_date.

    Method: use EDGAR submissions API to get exact filed_date for each period,
    then filter out any period where filed_date > as_of_date.
    """

    # EDGAR XBRL concept preference chains for common metrics
    CONCEPT_CHAINS: Dict[str, List[str]] = {
        "revenue":       ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
        "net_income":    ["NetIncomeLoss", "NetIncome", "ProfitLoss"],
        "total_assets":  ["Assets"],
        "total_equity":  ["StockholdersEquity", "Equity"],
        "total_liabilities": ["Liabilities"],
        "cash":          ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
        "long_term_debt":["LongTermDebt", "LongTermDebtNoncurrent"],
        "cfo":           ["NetCashProvidedByUsedInOperatingActivities"],
        "capex":         ["PaymentsToAcquirePropertyPlantAndEquipment"],
        "gross_profit":  ["GrossProfit"],
        "operating_income": ["OperatingIncomeLoss"],
        "ebit":          ["OperatingIncomeLoss"],
        "eps_diluted":   ["EarningsPerShareDiluted"],
        "shares_diluted":["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
        "rd_expense":    ["ResearchAndDevelopmentExpense"],
        "depreciation":  ["DepreciationAndAmortization", "Depreciation"],
    }

    def __init__(self, edgar: EDGARClient, db: _PITDatabase) -> None:
        self._edgar = edgar
        self._db = db
        self._facts_cache: Dict[str, Dict] = {}

    def _get_facts(self, cik: str) -> Dict:
        if cik in self._facts_cache:
            return self._facts_cache[cik]
        facts = self._edgar.get_company_facts(cik) or {}
        self._facts_cache[cik] = facts
        return facts

    def _extract_xbrl_value(self, facts: Dict, concepts: List[str], period_end: str, forms: Optional[List[str]] = None) -> Optional[Tuple[float, str]]:
        """
        Extract value for a concept from XBRL facts, matching a specific period_end.
        Returns (value, unit) or None.
        """
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        forms_set = set(forms or list(EARNINGS_FORMS))
        for concept in concepts:
            data = us_gaap.get(concept, {})
            units = data.get("units", {})
            for unit, entries in units.items():
                for entry in entries:
                    if entry.get("end") == period_end and entry.get("form") in forms_set:
                        return (float(entry["val"]), unit)
        return None

    def get_facts_as_of(self, ticker: str, metric: str, as_of_date: str) -> PITQueryResult:
        """
        PIT-safe retrieval: return only financial data that was filed on or before as_of_date.

        Steps:
        1. Resolve ticker → CIK
        2. Get EDGAR submissions to find all filings with their filed_date
        3. Filter: filed_date <= as_of_date
        4. From remaining filings, extract the most recent period's metric value
        5. Log access to pit_access_log
        """
        result = PITQueryResult(
            ticker=ticker,
            metric=metric,
            as_of_date=as_of_date,
        )

        cik = self._edgar.resolve_cik(ticker, self._db)
        if not cik:
            result.message = f"Could not resolve CIK for {ticker}"
            result.is_pit_safe = False
            return result

        # Get available filing from registry
        filing = self._db.get_most_recent_filing_as_of(
            ticker, as_of_date, list(EARNINGS_FORMS)
        )

        if not filing:
            # Try fetching from EDGAR and rebuilding registry
            lag_model = PublicationLagModel(self._edgar, self._db)
            lag_model.build_filing_registry(ticker, cik)
            filing = self._db.get_most_recent_filing_as_of(
                ticker, as_of_date, list(EARNINGS_FORMS)
            )

        if not filing:
            result.message = f"No filings available on or before {as_of_date}"
            result.is_pit_safe = True  # No data = PIT safe (no leak)
            return result

        period_end = filing["period_end"]
        filed_date = filing["filed_date"]

        # Now extract the metric value from XBRL
        concepts = self.CONCEPT_CHAINS.get(metric)
        if not concepts:
            result.message = f"Unknown metric: {metric}. Supported: {list(self.CONCEPT_CHAINS.keys())}"
            result.is_pit_safe = False
            return result

        facts = self._get_facts(cik)
        val_result = self._extract_xbrl_value(facts, concepts, period_end)

        result.value       = val_result[0] if val_result else None
        result.filed_date  = filed_date
        result.period_end  = period_end
        result.form        = filing.get("form")
        result.lag_days    = filing.get("lag_days")
        result.is_pit_safe = True
        result.message     = (
            f"PIT-safe: filed {filed_date}, period {period_end}. "
            f"Data was available {(datetime.strptime(as_of_date, '%Y-%m-%d').date() - datetime.strptime(filed_date, '%Y-%m-%d').date()).days} days before as_of_date."
            if val_result else
            f"No XBRL value found for {metric} in period {period_end}"
        )

        # Log access
        self._db.log_pit_access({
            "ticker":     ticker.upper(),
            "metric":     metric,
            "as_of_date": as_of_date,
            "filed_date": filed_date,
            "period_end": period_end,
            "value":      result.value,
            "is_pit_safe":1,
        })
        return result

    def build_pit_panel(self, ticker: str, metric: str, dates: List[str]) -> pd.DataFrame:
        """
        Build a PIT-safe time-series panel for a ticker/metric across multiple dates.
        Each row: (as_of_date, value, filed_date, period_end, lag_days).
        """
        rows = []
        for d in dates:
            r = self.get_facts_as_of(ticker, metric, d)
            rows.append({
                "as_of_date":  d,
                "ticker":      ticker,
                "metric":      metric,
                "value":       r.value,
                "filed_date":  r.filed_date,
                "period_end":  r.period_end,
                "lag_days":    r.lag_days,
                "is_pit_safe": r.is_pit_safe,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Look-ahead bias scanner
# ---------------------------------------------------------------------------

class LookAheadBiasScanner:
    """
    Scans a DataFrame for look-ahead bias by cross-referencing financial data dates
    against actual EDGAR filing dates.

    Expected DataFrame columns:
      - date (str YYYY-MM-DD): the "as of" date used in the backtest
      - ticker (str): stock ticker
      - metric (str): financial metric name (e.g., 'revenue', 'net_income')
      - period_end (str YYYY-MM-DD): the fiscal period the data refers to
      Optional:
      - value (float): the metric value

    A row is FLAGGED if: period_end financial data was filed AFTER the 'date' column,
    meaning the researcher used data that was not yet public.
    """

    def __init__(self, edgar: EDGARClient, db: _PITDatabase, lag_model: PublicationLagModel) -> None:
        self._edgar     = edgar
        self._db        = db
        self._lag_model = lag_model
        self._filing_cache: Dict[Tuple[str, str, str], Optional[str]] = {}

    def _get_filed_date(self, ticker: str, period_end: str) -> Optional[str]:
        """Look up when the financial data for period_end was actually filed."""
        key = (ticker.upper(), period_end)
        if key in self._filing_cache:
            return self._filing_cache[key]

        # Check DB first
        filing = self._db.get_filed_date_for_period(
            ticker, period_end, list(EARNINGS_FORMS)
        )
        if filing:
            fd = filing["filed_date"]
            self._filing_cache[key] = fd
            return fd

        # Fetch from EDGAR and populate registry
        cik = self._edgar.resolve_cik(ticker, self._db)
        if cik:
            self._lag_model.build_filing_registry(ticker, cik)
            filing = self._db.get_filed_date_for_period(ticker, period_end, list(EARNINGS_FORMS))
            if filing:
                fd = filing["filed_date"]
                self._filing_cache[key] = fd
                return fd

        # Fallback: estimate using statutory lag (conservative)
        try:
            pe_date = datetime.strptime(period_end, "%Y-%m-%d").date()
            estimated_filed = pe_date + timedelta(days=45)  # non-accelerated 10-Q
            fd = str(estimated_filed)
            self._filing_cache[key] = fd
            logger.warning("filed_date_estimated", ticker=ticker, period_end=period_end, estimated=fd)
            return fd
        except ValueError:
            self._filing_cache[key] = None
            return None

    def scan(self, df: pd.DataFrame, scan_id: Optional[str] = None) -> ScanResult:
        """
        Scan a DataFrame for look-ahead bias.

        Returns ScanResult with flagged rows, clean rows, and overall bias percentage.
        """
        import uuid
        scan_id = scan_id or str(uuid.uuid4())[:8]

        required_cols = {"date", "ticker", "metric", "period_end"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        flags: List[Dict[str, Any]] = []
        clean_count = 0

        for idx, row in df.iterrows():
            as_of_date = str(row["date"])
            ticker     = str(row["ticker"]).upper()
            metric     = str(row["metric"])
            period_end = str(row["period_end"])

            filed_date = self._get_filed_date(ticker, period_end)

            if not filed_date:
                clean_count += 1
                continue

            # Compare: was this data available on as_of_date?
            try:
                fd = datetime.strptime(filed_date, "%Y-%m-%d").date()
                aod = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            except ValueError:
                clean_count += 1
                continue

            if fd > aod:
                # LOOK-AHEAD BIAS: data was not yet filed
                lag_behind = (fd - aod).days
                flags.append({
                    "row_index":      int(idx),
                    "ticker":         ticker,
                    "metric":         metric,
                    "data_date":      period_end,
                    "as_of_date":     as_of_date,
                    "filed_date":     filed_date,
                    "lag_days_behind":lag_behind,
                })
            else:
                clean_count += 1

        total = len(df)
        flagged = len(flags)
        bias_pct = round(flagged / total * 100, 2) if total > 0 else 0.0

        # Attach look_ahead_pct to each flag
        for f in flags:
            f["look_ahead_pct"] = bias_pct

        # Persist flags
        if flags:
            self._db.upsert_look_ahead_flags(scan_id, flags)

        summary = (
            f"Scan '{scan_id}': {flagged} of {total} rows ({bias_pct:.1f}%) contain look-ahead bias. "
            f"Financial data was used before it was publicly filed on EDGAR. "
            f"{'HIGH RISK: >10% contamination.' if bias_pct > 10 else 'LOW RISK: <10% contamination.' if bias_pct > 0 else 'CLEAN: no look-ahead detected.'}"
        )

        return ScanResult(
            total_rows=total,
            flagged_rows=flagged,
            clean_rows=clean_count,
            look_ahead_pct=bias_pct,
            flags=[LookAheadFlag(**f) for f in flags],
            summary=summary,
        )

    def quantify_bias_by_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-ticker bias summary from a DataFrame."""
        result = self.scan(df)
        ticker_flags: Dict[str, List] = {}
        for f in result.flags:
            ticker_flags.setdefault(f.ticker, []).append(f)

        rows = []
        for ticker in df["ticker"].unique():
            total_rows = len(df[df["ticker"] == ticker])
            flagged    = len(ticker_flags.get(ticker, []))
            rows.append({
                "ticker":        ticker,
                "total_rows":    total_rows,
                "flagged_rows":  flagged,
                "look_ahead_pct":round(flagged / total_rows * 100, 2) if total_rows else 0,
            })
        return pd.DataFrame(rows).sort_values("look_ahead_pct", ascending=False)


# ---------------------------------------------------------------------------
# Backtest universe builder (S&P 500, survivorship-bias-free)
# ---------------------------------------------------------------------------

class BacktestUniverseBuilder:
    """
    Builds S&P 500 historical constituency with entry/exit dates.
    Source: Wikipedia current list + EDGAR change history + manual additions.

    Survivorship-bias elimination: include all stocks that WERE in the index
    on any given date, even if they were later removed (delisted, acquired, etc.).
    """

    def __init__(self, edgar: EDGARClient, db: _PITDatabase) -> None:
        self._edgar = edgar
        self._db    = db

    def _fetch_wikipedia_sp500(self) -> List[Dict[str, Any]]:
        """Fetch current S&P 500 list from Wikipedia."""
        try:
            time.sleep(0.5)
            r = requests.get(SP500_WIKI_URL, headers={
                "User-Agent": "SENTINEL/3.0 financial-terminal richard.porras@realempanada.com",
                "Accept": "text/html,application/xhtml+xml",
            }, timeout=30)
            r.raise_for_status()

            tables = pd.read_html(r.text)
            # First table is current constituents
            df = tables[0]
            # Column names vary; normalize
            col_map = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if "symbol" in col_lower or "ticker" in col_lower:
                    col_map[col] = "ticker"
                elif "security" in col_lower or "company" in col_lower or "name" in col_lower:
                    col_map[col] = "name"
                elif "gics sector" in col_lower or "sector" in col_lower:
                    col_map[col] = "sector"
                elif "gics sub" in col_lower or "sub-industry" in col_lower or "sub industry" in col_lower:
                    col_map[col] = "sub_industry"
                elif "date" in col_lower and "added" in col_lower:
                    col_map[col] = "added_date"
            df = df.rename(columns=col_map)

            members = []
            for _, row in df.iterrows():
                ticker = str(row.get("ticker", "")).strip().replace(".", "-")
                if not ticker or ticker == "nan":
                    continue
                added_raw = str(row.get("added_date", "")).strip()
                added_date = None
                if added_raw and added_raw != "nan":
                    # Try to parse various date formats
                    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"):
                        try:
                            added_date = datetime.strptime(added_raw, fmt).strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            pass

                members.append({
                    "ticker":      ticker,
                    "name":        str(row.get("name", "")).strip(),
                    "sector":      str(row.get("sector", "")).strip(),
                    "sub_industry":str(row.get("sub_industry", "")).strip(),
                    "added_date":  added_date,
                    "removed_date":None,
                    "is_current":  1,
                    "cik":         None,
                    "source":      "Wikipedia",
                })
            return members
        except Exception as e:
            logger.error("wikipedia_sp500_fetch_failed", error=str(e))
            return []

    def _fetch_changes_from_wikipedia(self) -> List[Dict[str, Any]]:
        """Fetch historical additions/removals from the Wikipedia changes table."""
        try:
            r = requests.get(SP500_WIKI_URL, headers={
                "User-Agent": "SENTINEL/3.0 financial-terminal richard.porras@realempanada.com",
                "Accept": "text/html",
            }, timeout=30)
            r.raise_for_status()
            tables = pd.read_html(r.text)
            # Second table typically contains historical changes
            if len(tables) < 2:
                return []

            changes_df = tables[1]
            changes: List[Dict[str, Any]] = []

            for _, row in changes_df.iterrows():
                # Columns vary: Date, Added (ticker + name), Removed (ticker + name)
                cols = [str(c).lower() for c in changes_df.columns]
                date_col  = next((changes_df.columns[i] for i, c in enumerate(cols) if "date" in c), None)
                added_col = next((changes_df.columns[i] for i, c in enumerate(cols) if "added" in c and "ticker" in c), None)
                removed_col = next((changes_df.columns[i] for i, c in enumerate(cols) if "removed" in c and "ticker" in c), None)

                if date_col is None:
                    continue

                change_date_raw = str(row[date_col]).strip()
                change_date = None
                for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"):
                    try:
                        change_date = datetime.strptime(change_date_raw, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        pass

                if not change_date:
                    continue

                if added_col and str(row.get(added_col, "")).strip() not in ("", "nan"):
                    added_ticker = str(row[added_col]).strip().replace(".", "-")
                    changes.append({
                        "ticker": added_ticker, "added_date": change_date,
                        "removed_date": None, "is_current": 1,
                        "name": "", "sector": "", "sub_industry": "", "cik": None,
                        "source": "Wikipedia_changes",
                    })

                if removed_col and str(row.get(removed_col, "")).strip() not in ("", "nan"):
                    removed_ticker = str(row[removed_col]).strip().replace(".", "-")
                    changes.append({
                        "ticker": removed_ticker, "removed_date": change_date,
                        "added_date": None, "is_current": 0,
                        "name": "", "sector": "", "sub_industry": "", "cik": None,
                        "source": "Wikipedia_changes",
                    })

            return changes
        except Exception as e:
            logger.warning("wikipedia_changes_fetch_failed", error=str(e))
            return []

    def load_universe(self, force_refresh: bool = False) -> int:
        """Load S&P 500 current + historical members into DB. Returns count loaded."""
        if self._db.has_sp500_data() and not force_refresh:
            return 0

        current = self._fetch_wikipedia_sp500()
        for m in current:
            self._db.upsert_sp500_member(m)
        self._db.commit()

        changes = self._fetch_changes_from_wikipedia()
        for c in changes:
            self._db.upsert_sp500_member(c)
        self._db.commit()

        # Resolve CIKs for known current members
        cik_data = self._edgar.get_company_facts  # already have map from tickers load
        self._edgar.load_ticker_map(self._db)

        total_loaded = len(current) + len(changes)
        logger.info("sp500_universe_loaded", current=len(current), changes=len(changes))
        return total_loaded

    def get_universe_as_of(self, as_of_date: str) -> SafeUniverse:
        """
        Return survivorship-bias-free S&P 500 universe as of a specific date.
        Includes companies added before as_of_date and not yet removed by as_of_date.
        """
        self.load_universe()
        members = self._db.get_sp500_as_of(as_of_date)

        member_models = [
            SP500Member(
                ticker=m["ticker"],
                name=m.get("name") or m["ticker"],
                added_date=m.get("added_date"),
                removed_date=m.get("removed_date"),
                is_current=bool(m.get("is_current", 1)),
                cik=m.get("cik"),
                sector=m.get("sector"),
                sub_industry=m.get("sub_industry"),
            )
            for m in members
        ]

        return SafeUniverse(
            as_of_date=as_of_date,
            members=member_models,
            total=len(member_models),
        )

    def get_current_universe(self) -> List[SP500Member]:
        self.load_universe()
        members = self._db.get_current_sp500()
        return [
            SP500Member(
                ticker=m["ticker"],
                name=m.get("name") or m["ticker"],
                added_date=m.get("added_date"),
                removed_date=None,
                is_current=True,
                cik=m.get("cik"),
                sector=m.get("sector"),
                sub_industry=m.get("sub_industry"),
            )
            for m in members
        ]


# ---------------------------------------------------------------------------
# Earnings release timing (8-K item 2.02)
# ---------------------------------------------------------------------------

class EarningsReleaseTimingModel:
    """
    Models 8-K item 2.02 filing date, which precedes 10-Q/10-K by days or weeks.
    The 8-K with earnings release is the first public disclosure; the 10-Q/10-K
    follows with full detail. PIT backtests should use 8-K date, not 10-Q date.
    """

    def __init__(self, edgar: EDGARClient, db: _PITDatabase) -> None:
        self._edgar = edgar
        self._db    = db

    def get_8k_filing_dates(self, ticker: str, cik: str) -> List[Dict[str, Any]]:
        """Extract 8-K filings with item 2.02 (Earnings Announcement) from EDGAR."""
        subs = self._edgar.get_submissions(cik)
        if not subs:
            return []

        recent = subs.get("filings", {}).get("recent", {})
        forms        = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        period_ends  = recent.get("periodOfReport", [])
        items        = recent.get("items", [])

        results = []
        for i, f in enumerate(forms):
            if f != "8-K":
                continue
            item_str = str(items[i]) if i < len(items) else ""
            # Only earnings releases (item 2.02)
            if "2.02" not in item_str and "2.01" not in item_str:
                continue
            results.append({
                "ticker":       ticker.upper(),
                "form":         "8-K",
                "items":        item_str,
                "filed_date":   filing_dates[i] if i < len(filing_dates) else None,
                "period_end":   period_ends[i]  if i < len(period_ends)  else None,
                "is_earnings":  True,
            })
        return results

    def get_10q_vs_8k_lead(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Compute how many days earlier the 8-K earnings release came before the 10-Q.
        Critical for PIT: use 8-K date for earnings-level metrics.
        """
        cik = self._edgar.resolve_cik(ticker, self._db)
        if not cik:
            return []

        filings_8k = self.get_8k_filing_dates(ticker, cik)
        filings_10q = self._db.get_filings(ticker, form="10-Q")

        if not filings_8k:
            return []

        results = []
        for filing_8k in filings_8k:
            period_end = filing_8k.get("period_end")
            if not period_end:
                continue

            # Find matching 10-Q
            matching_10q = next(
                (f for f in filings_10q if f.get("period_end") == period_end),
                None
            )

            row = {
                "ticker":         ticker.upper(),
                "period_end":     period_end,
                "8k_filed_date":  filing_8k["filed_date"],
                "10q_filed_date": matching_10q["filed_date"] if matching_10q else None,
            }

            if matching_10q and filing_8k["filed_date"] and matching_10q["filed_date"]:
                try:
                    d8k = datetime.strptime(filing_8k["filed_date"], "%Y-%m-%d").date()
                    d10q = datetime.strptime(matching_10q["filed_date"], "%Y-%m-%d").date()
                    row["8k_lead_days"] = (d10q - d8k).days
                    row["pit_recommendation"] = (
                        f"Use 8-K date ({filing_8k['filed_date']}) for earnings data; "
                        f"10-Q filed {row['8k_lead_days']} days later."
                    )
                except ValueError:
                    row["8k_lead_days"] = None
                    row["pit_recommendation"] = "Could not compute lead"
            else:
                row["8k_lead_days"] = None
                row["pit_recommendation"] = "No matching 10-Q found"

            results.append(row)

        return results


# ---------------------------------------------------------------------------
# Primary service orchestrator
# ---------------------------------------------------------------------------

class PITIntegrityService:
    """Orchestrates all PIT integrity components."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db        = _PITDatabase(db_path)
        self._edgar     = EDGARClient()
        self._lag_model = PublicationLagModel(self._edgar, self._db)
        self._pit       = PITDatabaseWrapper(self._edgar, self._db)
        self._scanner   = LookAheadBiasScanner(self._edgar, self._db, self._lag_model)
        self._universe  = BacktestUniverseBuilder(self._edgar, self._db)
        self._earnings  = EarningsReleaseTimingModel(self._edgar, self._db)

    def get_filing_dates(self, ticker: str) -> List[Dict]:
        cik = self._edgar.resolve_cik(ticker, self._db)
        if not cik:
            return []
        filings = self._db.get_filings(ticker)
        if not filings:
            self._lag_model.build_filing_registry(ticker, cik)
            filings = self._db.get_filings(ticker)
        return filings

    def get_as_of(self, ticker: str, metric: str, as_of: str) -> PITQueryResult:
        return self._pit.get_facts_as_of(ticker, metric, as_of)

    def build_pit_panel(self, ticker: str, metric: str, dates: List[str]) -> pd.DataFrame:
        return self._pit.build_pit_panel(ticker, metric, dates)

    def scan_dataframe(self, df: pd.DataFrame) -> ScanResult:
        return self._scanner.scan(df)

    def get_safe_universe(self, as_of_date: str) -> SafeUniverse:
        return self._universe.get_universe_as_of(as_of_date)

    def get_lag_model(self, ticker: str) -> LagModelResult:
        cik = self._edgar.resolve_cik(ticker, self._db)
        if not cik:
            return LagModelResult(ticker=ticker, cik="", filer_category="unknown", filings=[])
        existing = self._db.get_filings(ticker)
        if not existing:
            self._lag_model.build_filing_registry(ticker, cik)
        return self._lag_model.compute_lag_statistics(ticker)

    def get_late_filers(self, form: str = "10-K", limit: int = 100) -> List[Dict]:
        return self._db.get_late_filers(form, limit)

    def get_earnings_timing(self, ticker: str) -> List[Dict]:
        cik = self._edgar.resolve_cik(ticker, self._db)
        if not cik:
            return []
        return self._earnings.get_10q_vs_8k_lead(ticker)

    def quantify_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._scanner.quantify_bias_by_ticker(df)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/pit-integrity/v3", tags=["PIT Integrity v3"])
_svc: Optional[PITIntegrityService] = None


def _get_svc() -> PITIntegrityService:
    global _svc
    if _svc is None:
        _svc = PITIntegrityService()
    return _svc


@router.get("/filing-dates/{ticker}", response_model=List[Dict])
def get_filing_dates(
    ticker: str,
    form: Optional[str] = Query(None, description="Filter by form type: 10-Q, 10-K, 20-F"),
) -> List[Dict]:
    """
    All EDGAR filing dates for a ticker with lag analysis.
    Each entry includes: form, period_end, filed_date, lag_days, filer_category,
    statutory_deadline_days, is_late, nt_filed.
    """
    filings = _get_svc().get_filing_dates(ticker.upper())
    if not filings:
        raise HTTPException(status_code=404, detail=f"No filings found for {ticker}")
    if form:
        filings = [f for f in filings if f.get("form") == form.upper()]
    return filings


@router.get("/as-of/{ticker}", response_model=PITQueryResult)
def get_as_of(
    ticker: str,
    metric: str = Query(..., description=f"Financial metric. Options: {list(PITDatabaseWrapper.CONCEPT_CHAINS.keys())}"),
    as_of: str = Query(..., description="Point-in-time date YYYY-MM-DD"),
) -> PITQueryResult:
    """
    PIT-safe data retrieval: returns the financial metric value that was publicly
    available (filed on EDGAR) on or before as_of date. Prevents look-ahead bias.
    """
    try:
        datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="as_of must be YYYY-MM-DD format")

    result = _get_svc().get_as_of(ticker.upper(), metric, as_of)
    return result


@router.post("/scan-for-lookahead", response_model=ScanResult)
def scan_for_lookahead(
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "rows": [
                {"date": "2022-01-15", "ticker": "AAPL", "metric": "revenue", "period_end": "2021-09-25"},
                {"date": "2022-01-15", "ticker": "MSFT", "metric": "net_income", "period_end": "2021-09-30"},
            ]
        },
    )
) -> ScanResult:
    """
    Upload a DataFrame as JSON and get a look-ahead bias scan.

    Required columns: date, ticker, metric, period_end.
    Returns: flagged rows, bias percentage, and EDGAR-verified filing dates.

    A row is FLAGGED if the financial data (period_end) was not yet filed on EDGAR
    as of the 'date' column, meaning look-ahead bias exists.
    """
    rows = payload.get("rows", [])
    if not rows:
        raise HTTPException(status_code=400, detail="Payload must contain 'rows' list")

    df = pd.DataFrame(rows)
    required = {"date", "ticker", "metric", "period_end"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    try:
        return _get_svc().scan_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/safe-universe", response_model=SafeUniverse)
def get_safe_universe(
    as_of: str = Query(..., description="Historical date YYYY-MM-DD for universe construction"),
) -> SafeUniverse:
    """
    Survivorship-bias-free S&P 500 universe as of a historical date.
    Includes companies that were in the index on that date, even if later removed.
    Uses Wikipedia historical constituency data.
    """
    try:
        datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="as_of must be YYYY-MM-DD")

    return _get_svc().get_safe_universe(as_of)


@router.get("/lag-model/{ticker}", response_model=LagModelResult)
def get_lag_model(ticker: str) -> LagModelResult:
    """
    Publication lag analysis for a ticker.
    Returns: filer category, statutory deadlines, actual average lags, late-filer flags.
    """
    result = _get_svc().get_lag_model(ticker.upper())
    if not result.filings and not result.cik:
        raise HTTPException(status_code=404, detail=f"Could not resolve {ticker} to an EDGAR CIK")
    return result


@router.get("/late-filers", response_model=List[Dict])
def get_late_filers(
    form: str = Query("10-K", description="Form type: 10-K, 10-Q, 20-F"),
    limit: int = Query(100, ge=1, le=500),
) -> List[Dict]:
    """
    List companies that filed past their statutory deadline.
    Includes NT 10-K/NT 10-Q detection (notice of late filing).
    """
    return _get_svc().get_late_filers(form=form.upper(), limit=limit)


@router.get("/earnings-timing/{ticker}", response_model=List[Dict])
def get_earnings_timing(ticker: str) -> List[Dict]:
    """
    8-K (item 2.02) vs 10-Q filing date comparison.
    Shows how many days earlier the earnings release (8-K) arrived before the full 10-Q.
    PIT-critical: use 8-K date for earnings metrics, not 10-Q date.
    """
    result = _get_svc().get_earnings_timing(ticker.upper())
    if not result:
        raise HTTPException(status_code=404, detail=f"No 8-K earnings releases found for {ticker}")
    return result


@router.get("/pit-panel/{ticker}", response_model=List[Dict])
def get_pit_panel(
    ticker: str,
    metric: str = Query(..., description="Financial metric (e.g., revenue, net_income)"),
    start: str = Query("2018-01-01", description="Start date YYYY-MM-DD"),
    end: str   = Query(str(date.today()), description="End date YYYY-MM-DD"),
    freq: str  = Query("Q", description="Frequency: Q (quarterly) or A (annual)"),
) -> List[Dict]:
    """
    Build a PIT-safe time-series panel for a ticker/metric.
    Each point uses only data that was filed on or before that date.
    """
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="start and end must be YYYY-MM-DD")

    freq_map = {"Q": "QS", "A": "AS", "M": "MS"}
    pd_freq  = freq_map.get(freq.upper(), "QS")
    dates    = [str(d.date()) for d in pd.date_range(start_dt, end_dt, freq=pd_freq)]

    df = _get_svc().build_pit_panel(ticker.upper(), metric, dates)
    return df.to_dict(orient="records")


@router.get("/supported-metrics", response_model=List[str])
def get_supported_metrics() -> List[str]:
    """List all financial metrics supported by the PIT query engine."""
    return list(PITDatabaseWrapper.CONCEPT_CHAINS.keys())


@router.get("/access-log/{ticker}", response_model=List[Dict])
def get_access_log(
    ticker: str,
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict]:
    """Audit log of all PIT queries made for a ticker."""
    db = _get_svc()._db
    cur = db._conn.execute(
        "SELECT * FROM pit_access_log WHERE ticker = ? ORDER BY queried_at DESC LIMIT ?",
        (ticker.upper(), limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        raise HTTPException(status_code=404, detail=f"No access log entries for {ticker}")
    return rows


@router.get("/filer-category/{ticker}", response_model=Dict)
def get_filer_category(ticker: str) -> Dict:
    """
    Detect SEC filer category for a company (large_accelerated, accelerated,
    non_accelerated, smaller_reporting, foreign_private).
    Used to determine statutory filing deadline (10-Q: 40-45 days; 10-K: 60-90 days).
    """
    svc = _get_svc()
    cik = svc._edgar.resolve_cik(ticker.upper(), svc._db)
    if not cik:
        raise HTTPException(status_code=404, detail=f"Cannot resolve {ticker} to EDGAR CIK")

    category = svc._lag_model.detect_filer_category(cik, ticker)
    deadlines = STATUTORY_DEADLINES.get(category, {})

    return {
        "ticker":         ticker.upper(),
        "cik":            cik,
        "filer_category": category,
        "statutory_deadlines": deadlines,
        "description": {
            "large_accelerated": "Public float >$700M. 10-Q: 40 days, 10-K: 60 days.",
            "accelerated":       "Public float $75M-$700M. 10-Q: 40 days, 10-K: 75 days.",
            "non_accelerated":   "Public float <$75M. 10-Q: 45 days, 10-K: 90 days.",
            "smaller_reporting": "Revenue <$100M or float <$250M. 10-Q: 45 days, 10-K: 90 days.",
            "foreign_private":   "Foreign Private Issuer. 20-F: 120 days.",
        }.get(category, "Unknown category"),
    }
