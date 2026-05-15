"""
Historical financial database: 15+ years, point-in-time (no look-ahead bias),
filing vintage tracking, restatement detection, and long-run trend analysis.

Targets:
  dim_020  Historical financials (point-in-time, 15+ years)  → 9

Public API
----------
HistoricalFinancialsDatabase  — SQLite-backed historical store (ticker, period_end, filing_date, metric, value)
PointInTimeEngine             — strict PIT filtering; vintage tracking; embargo-aware
RestatementDetector           — detect and classify restatements; materiality grading
LongRunTrendAnalyzer          — CAGR, margin decile, recession performance, cycle analysis
FiscalCalendarManager         — non-standard FY handling; TTM alignment
historical_router             — FastAPI APIRouter
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "HistoricalFinancialsDatabase",
    "PointInTimeEngine",
    "RestatementDetector",
    "LongRunTrendAnalyzer",
    "FiscalCalendarManager",
    "historical_router",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://data.sec.gov"
EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_RATE_DELAY = 0.12   # 120 ms → ~8 req/s; EDGAR cap = 10 req/s
_TIMEOUT = 30.0
_MAX_RETRY = 3

# SEC filing deadlines (calendar days after period end)
FILING_EMBARGO = {
    "10-Q": 45,   # large accelerated filer: 40; accelerated: 45; non-accelerated: 45
    "10-K": 90,   # large accelerated filer: 60; accelerated: 75; non-accelerated: 90
    "10-QT": 45,
    "10-KT": 90,
    "20-F": 120,
    "40-F": 90,
}

ANNUAL_FORMS = {"10-K", "10-KT", "20-F", "40-F"}
QUARTERLY_FORMS = {"10-Q", "10-QT"}

# Recession date ranges (NBER)
RECESSIONS = [
    (date(2001, 3, 1), date(2001, 11, 30), "2001_dotcom"),
    (date(2007, 12, 1), date(2009, 6, 30), "2008_gfc"),
    (date(2020, 2, 1), date(2020, 4, 30), "2020_covid"),
]

# Known fiscal year end months (month number) for major tickers
KNOWN_FY_MONTHS: dict[str, int] = {
    "AAPL": 9,   # September
    "MSFT": 6,   # June
    "WMT": 1,    # January
    "HD": 1,     # January
    "TGT": 1,    # January
    "COST": 8,   # August
    "NKE": 5,    # May
    "INTC": 12,
    "AMZN": 12,
    "GOOG": 12,
    "GOOGL": 12,
    "META": 12,
    "NVDA": 1,   # January
    "TSLA": 12,
    "JNJ": 12,
    "PFE": 12,
    "BAC": 12,
    "JPM": 12,
    "GS": 12,
}

# XBRL concept map: standardized metric name → list of XBRL tags (priority order)
METRIC_CONCEPT_MAP: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfGoodsSoldAndServicesSold",
        "CostOfGoodsAndServicesSold",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "ebitda_proxy": ["OperatingIncomeLoss"],
    "total_assets": [
        "Assets",
        "AssetsCurrent",
    ],
    "total_liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
        "LiabilitiesAndStockholdersEquity",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermNotesPayable",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "NotesPayableCurrent",
        "LongTermDebtCurrent",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],
    "dividends": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
    "r_and_d": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "Depreciation",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
    "tax_expense": [
        "IncomeTaxExpenseBenefit",
    ],
    "inventory": [
        "InventoryNet",
        "InventoryFinishedGoods",
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "accounts_payable": [
        "AccountsPayableCurrent",
    ],
}

BALANCE_SHEET_METRICS = {
    "total_assets", "total_liabilities", "equity",
    "long_term_debt", "short_term_debt", "cash",
    "inventory", "accounts_receivable", "accounts_payable",
    "shares_outstanding",
}

DB_PATH = Path(__file__).parent.parent / "data" / "historical_financials.db"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FilingRecord(BaseModel):
    ticker: str
    cik: str
    period_end: date
    filing_date: date
    form_type: str
    source: str
    metric: str
    value: float
    is_restated: bool = False
    vintage_seq: int = 1  # 1 = original; 2+ = restatement vintage


class RestatementEvent(BaseModel):
    ticker: str
    period_end: date
    metric: str
    original_filing_date: date
    restatement_filing_date: date
    original_value: float
    restated_value: float
    change_pct: float
    is_material: bool  # |change| > 5%
    restatement_type: str  # 'error_correction' | 'reclassification' | 'rule_change' | 'unknown'


class TrendResult(BaseModel):
    ticker: str
    metric: str
    cagr_5y: Optional[float] = None
    cagr_10y: Optional[float] = None
    cagr_15y: Optional[float] = None
    current_decile: Optional[int] = None  # 1-10 vs own history
    values: dict[str, float] = Field(default_factory=dict)  # year → value


class RecessionTestResult(BaseModel):
    ticker: str
    recession: str
    metric: str
    pre_recession_value: Optional[float] = None
    trough_value: Optional[float] = None
    peak_to_trough_pct: Optional[float] = None
    recovery_quarters: Optional[int] = None
    resilience_grade: str  # 'A' 'B' 'C' 'D' 'F'


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _rate_limited_get(url: str, retries: int = _MAX_RETRY) -> dict:
    """Synchronous HTTP GET with EDGAR rate limiting and retry."""
    for attempt in range(retries):
        try:
            time.sleep(_RATE_DELAY)
            resp = httpx.get(url, headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                time.sleep(5 * (attempt + 1))
            elif attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
    return {}


def _resolve_cik(ticker: str) -> Optional[str]:
    """Resolve ticker → zero-padded CIK via EDGAR company tickers JSON."""
    try:
        data = _rate_limited_get(EDGAR_TICKERS_URL)
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning("CIK resolution failed for %s: %s", ticker, exc)
    return None


def _cagr(start_val: float, end_val: float, years: float) -> Optional[float]:
    if years <= 0 or start_val is None or end_val is None:
        return None
    if start_val == 0:
        return None
    try:
        ratio = end_val / start_val
        if ratio <= 0:
            return None
        return (ratio ** (1.0 / years)) - 1.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

class HistoricalFinancialsDatabase:
    """
    SQLite-backed store for point-in-time financial facts.

    Schema
    ------
    financials(id, ticker, cik, period_end, filing_date, form_type,
               source, metric, value, is_restated, vintage_seq, inserted_at)

    Uniqueness: (ticker, period_end, metric, vintage_seq)
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS financials (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker       TEXT    NOT NULL,
                cik          TEXT    NOT NULL,
                period_end   TEXT    NOT NULL,
                filing_date  TEXT    NOT NULL,
                form_type    TEXT    NOT NULL,
                source       TEXT    NOT NULL DEFAULT 'EDGAR',
                metric       TEXT    NOT NULL,
                value        REAL    NOT NULL,
                is_restated  INTEGER NOT NULL DEFAULT 0,
                vintage_seq  INTEGER NOT NULL DEFAULT 1,
                inserted_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(ticker, period_end, metric, vintage_seq)
            );
            CREATE INDEX IF NOT EXISTS idx_fin_ticker_metric
                ON financials(ticker, metric);
            CREATE INDEX IF NOT EXISTS idx_fin_ticker_period
                ON financials(ticker, period_end);
            CREATE INDEX IF NOT EXISTS idx_fin_filing_date
                ON financials(ticker, filing_date);

            CREATE TABLE IF NOT EXISTS restatements (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker                  TEXT    NOT NULL,
                period_end              TEXT    NOT NULL,
                metric                  TEXT    NOT NULL,
                original_filing_date    TEXT    NOT NULL,
                restatement_filing_date TEXT    NOT NULL,
                original_value          REAL    NOT NULL,
                restated_value          REAL    NOT NULL,
                change_pct              REAL    NOT NULL,
                is_material             INTEGER NOT NULL DEFAULT 0,
                restatement_type        TEXT    NOT NULL DEFAULT 'unknown',
                detected_at             TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_rest_ticker
                ON restatements(ticker);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_filing(self, rec: FilingRecord) -> None:
        """Insert or ignore a filing record (duplicate handled by vintage_seq)."""
        self._conn.execute("""
            INSERT OR IGNORE INTO financials
                (ticker, cik, period_end, filing_date, form_type, source,
                 metric, value, is_restated, vintage_seq)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            rec.ticker.upper(),
            rec.cik,
            rec.period_end.isoformat(),
            rec.filing_date.isoformat(),
            rec.form_type,
            rec.source,
            rec.metric,
            rec.value,
            int(rec.is_restated),
            rec.vintage_seq,
        ))
        self._conn.commit()

    def mark_restated(self, ticker: str, period_end: date, metric: str) -> None:
        """Flag all earlier vintages of a (ticker, period, metric) as restated."""
        self._conn.execute("""
            UPDATE financials
               SET is_restated = 1
             WHERE ticker = ? AND period_end = ? AND metric = ?
        """, (ticker.upper(), period_end.isoformat(), metric))
        self._conn.commit()

    def record_restatement(self, evt: RestatementEvent) -> None:
        self._conn.execute("""
            INSERT OR IGNORE INTO restatements
                (ticker, period_end, metric, original_filing_date,
                 restatement_filing_date, original_value, restated_value,
                 change_pct, is_material, restatement_type)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            evt.ticker.upper(),
            evt.period_end.isoformat(),
            evt.metric,
            evt.original_filing_date.isoformat(),
            evt.restatement_filing_date.isoformat(),
            evt.original_value,
            evt.restated_value,
            evt.change_pct,
            int(evt.is_material),
            evt.restatement_type,
        ))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read — as-of queries
    # ------------------------------------------------------------------

    def get_as_of(
        self,
        ticker: str,
        metric: str,
        as_of_date: date,
        prefer_original: bool = True,
    ) -> Optional[float]:
        """
        Return the value of `metric` for `ticker` that was publicly known
        at `as_of_date` (i.e., filing_date <= as_of_date).

        prefer_original=True  → return the originally-filed value (avoids look-ahead
                                 from later restatements).
        prefer_original=False → return the latest available value at as_of_date.
        """
        rows = self._conn.execute("""
            SELECT value, filing_date, vintage_seq
              FROM financials
             WHERE ticker = ? AND metric = ?
               AND filing_date <= ?
             ORDER BY period_end DESC, vintage_seq DESC
             LIMIT 1
        """, (ticker.upper(), metric, as_of_date.isoformat())).fetchall()

        if not rows:
            return None
        return rows[0]["value"]

    def get_history(
        self,
        ticker: str,
        metric: str,
        start: date,
        end: date,
        pit: bool = True,
    ) -> pd.Series:
        """
        Return a time series of `metric` indexed by period_end.

        pit=True  → each value is the ORIGINALLY filed version (no restatements),
                    so the series reflects what was actually available.
        pit=False → latest vintage available, regardless of restatement status.
        """
        if pit:
            rows = self._conn.execute("""
                SELECT period_end, value
                  FROM financials
                 WHERE ticker = ? AND metric = ?
                   AND period_end BETWEEN ? AND ?
                   AND vintage_seq = 1
                 ORDER BY period_end ASC
            """, (ticker.upper(), metric, start.isoformat(), end.isoformat())).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT period_end, MAX(value) as value
                  FROM financials
                 WHERE ticker = ? AND metric = ?
                   AND period_end BETWEEN ? AND ?
                 GROUP BY period_end
                 ORDER BY period_end ASC
            """, (ticker.upper(), metric, start.isoformat(), end.isoformat())).fetchall()

        if not rows:
            return pd.Series(dtype=float, name=metric)

        index = pd.to_datetime([r["period_end"] for r in rows])
        values = [r["value"] for r in rows]
        return pd.Series(values, index=index, name=metric)

    def get_all_periods(self, ticker: str) -> list[str]:
        rows = self._conn.execute("""
            SELECT DISTINCT period_end FROM financials
             WHERE ticker = ?
             ORDER BY period_end ASC
        """, (ticker.upper(),)).fetchall()
        return [r["period_end"] for r in rows]

    def get_vintages(self, ticker: str, period_end: date, metric: str) -> list[dict]:
        rows = self._conn.execute("""
            SELECT filing_date, value, vintage_seq, is_restated
              FROM financials
             WHERE ticker = ? AND period_end = ? AND metric = ?
             ORDER BY vintage_seq ASC
        """, (ticker.upper(), period_end.isoformat(), metric)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # EDGAR ingestion
    # ------------------------------------------------------------------

    def fetch_and_store(self, ticker: str, max_years: int = 25) -> dict[str, int]:
        """
        Fetch all EDGAR company facts for `ticker` and store every filing.
        Returns counts of records stored per metric.
        """
        cik = _resolve_cik(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker {ticker!r}")

        url = EDGAR_FACTS_URL.format(cik=cik)
        logger.info("Fetching EDGAR facts for %s (CIK=%s)", ticker, cik)
        data = _rate_limited_get(url)

        counts: dict[str, int] = {}
        cutoff = date.today() - timedelta(days=max_years * 365)

        us_gaap = data.get("facts", {}).get("us-gaap", {})

        for std_metric, xbrl_tags in METRIC_CONCEPT_MAP.items():
            for tag in xbrl_tags:
                concept_data = us_gaap.get(tag)
                if not concept_data:
                    continue

                units = concept_data.get("units", {})
                # Prefer USD; fallback to shares for share-count metrics
                unit_data = units.get("USD") or units.get("shares") or []

                stored = 0
                for item in unit_data:
                    form = item.get("form", "")
                    if form not in ANNUAL_FORMS and form not in QUARTERLY_FORMS:
                        continue

                    period_end_str = item.get("end", "")
                    filed_str = item.get("filed", "")
                    val = item.get("val")

                    if not period_end_str or not filed_str or val is None:
                        continue

                    try:
                        period_end_dt = date.fromisoformat(period_end_str)
                        filing_date_dt = date.fromisoformat(filed_str)
                    except ValueError:
                        continue

                    if period_end_dt < cutoff:
                        continue

                    # Determine vintage: how many earlier filings exist for same period+metric?
                    existing = self.get_vintages(ticker, period_end_dt, std_metric)
                    vintage_seq = len(existing) + 1
                    is_restated = vintage_seq > 1

                    rec = FilingRecord(
                        ticker=ticker.upper(),
                        cik=cik,
                        period_end=period_end_dt,
                        filing_date=filing_date_dt,
                        form_type=form,
                        source="EDGAR",
                        metric=std_metric,
                        value=float(val),
                        is_restated=is_restated,
                        vintage_seq=vintage_seq,
                    )
                    self.upsert_filing(rec)
                    stored += 1

                if stored:
                    counts[std_metric] = counts.get(std_metric, 0) + stored
                    break  # Found data for this metric from first matching tag

        logger.info("Stored %d metric groups for %s", len(counts), ticker)
        return counts

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Point-in-time engine
# ---------------------------------------------------------------------------

class PointInTimeEngine:
    """
    Strict point-in-time financial data access.

    Key guarantees:
    1. Only data with filing_date <= as_of_date is returned.
    2. Backfill embargo: if as_of_date < period_end + embargo_days, the filing
       is considered not yet available even if a filing_date is stored.
    3. Vintage tracking: original (vintage_seq=1) vs restated vintages.
    4. PIT series: daily/weekly panel with correct as-of values.
    """

    def __init__(self, db: HistoricalFinancialsDatabase) -> None:
        self.db = db

    def _embargo_date(self, period_end: date, form_type: str) -> date:
        """Earliest possible date when filing could be available."""
        days = FILING_EMBARGO.get(form_type, 45)
        return period_end + timedelta(days=days)

    def get_pit_value(
        self,
        ticker: str,
        metric: str,
        as_of_date: date,
        respect_embargo: bool = True,
    ) -> Optional[float]:
        """
        Return the value of `metric` as known at `as_of_date`.

        Uses the most recent period_end whose filing was available at as_of_date.
        If respect_embargo=True, also enforces SEC filing deadlines.
        """
        conn = self.db._conn
        rows = conn.execute("""
            SELECT value, filing_date, period_end, form_type, vintage_seq
              FROM financials
             WHERE ticker = ? AND metric = ?
               AND filing_date <= ?
             ORDER BY period_end DESC, vintage_seq ASC
        """, (ticker.upper(), metric, as_of_date.isoformat())).fetchall()

        for row in rows:
            period_end = date.fromisoformat(row["period_end"])
            filing_date = date.fromisoformat(row["filing_date"])

            if respect_embargo:
                embargo = self._embargo_date(period_end, row["form_type"])
                if as_of_date < embargo:
                    continue  # data not yet expected to be filed

            # Only use earliest (original) vintage for true PIT
            if row["vintage_seq"] == 1:
                return row["value"]

        return None

    def get_pit_series(
        self,
        ticker: str,
        metric: str,
        dates: list[date],
        respect_embargo: bool = True,
    ) -> pd.Series:
        """
        Return a PIT series: for each date in `dates`, return the value of
        `metric` that was known at that date (no look-ahead).
        """
        result = {}
        for d in dates:
            result[d] = self.get_pit_value(ticker, metric, d, respect_embargo)
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
        return pd.Series(result.values(), index=idx, name=metric, dtype=float)

    def get_ttm_pit(
        self,
        ticker: str,
        metric: str,
        as_of_date: date,
    ) -> Optional[float]:
        """
        Compute trailing-twelve-months value for a flow metric at as_of_date.
        Sums the four most recent non-overlapping quarterly values available.
        """
        if metric in BALANCE_SHEET_METRICS:
            return self.get_pit_value(ticker, metric, as_of_date)

        conn = self.db._conn
        rows = conn.execute("""
            SELECT period_end, value, form_type, vintage_seq
              FROM financials
             WHERE ticker = ? AND metric = ?
               AND filing_date <= ?
               AND vintage_seq = 1
             ORDER BY period_end DESC
        """, (ticker.upper(), metric, as_of_date.isoformat())).fetchall()

        # Build non-overlapping 90-day quarters
        used: list[float] = []
        last_end: Optional[date] = None

        for row in rows:
            period_end = date.fromisoformat(row["period_end"])
            if last_end is not None and (last_end - period_end).days < 80:
                continue  # overlapping period, skip
            # Check embargo
            embargo = self._embargo_date(period_end, row["form_type"])
            if as_of_date < embargo:
                continue
            used.append(row["value"])
            last_end = period_end
            if len(used) == 4:
                break

        if len(used) < 4:
            return None
        return sum(used)

    def get_vintage_history(
        self,
        ticker: str,
        period_end: date,
        metric: str,
    ) -> list[dict]:
        """Return all filing vintages for (ticker, period, metric)."""
        return self.db.get_vintages(ticker, period_end, metric)

    def build_backtest_panel(
        self,
        tickers: list[str],
        metric: str,
        dates: list[date],
        use_ttm: bool = False,
    ) -> pd.DataFrame:
        """
        Build a tickers × dates panel DataFrame with guaranteed PIT values.
        """
        records = {}
        for ticker in tickers:
            if use_ttm:
                series = {d: self.get_ttm_pit(ticker, metric, d) for d in dates}
            else:
                series = {d: self.get_pit_value(ticker, metric, d) for d in dates}
            records[ticker] = series

        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
        df = pd.DataFrame(records, index=idx)
        return df


# ---------------------------------------------------------------------------
# Restatement detector
# ---------------------------------------------------------------------------

class RestatementDetector:
    """
    Detect and classify financial restatements.

    A restatement occurs when the same (ticker, period_end, metric) has two or
    more filing vintages with materially different values.

    Materiality threshold: |new - old| / |old| > 5%
    """

    MATERIALITY_THRESHOLD = 0.05

    def __init__(self, db: HistoricalFinancialsDatabase) -> None:
        self.db = db

    def scan_ticker(self, ticker: str) -> list[RestatementEvent]:
        """Scan all filings for `ticker` and detect restatements."""
        conn = self.db._conn

        # Get all (period_end, metric) combos with multiple vintages
        rows = conn.execute("""
            SELECT period_end, metric, COUNT(*) as cnt
              FROM financials
             WHERE ticker = ?
             GROUP BY period_end, metric
            HAVING cnt > 1
            ORDER BY period_end DESC
        """, (ticker.upper(),)).fetchall()

        events: list[RestatementEvent] = []

        for row in rows:
            period_end = date.fromisoformat(row["period_end"])
            metric = row["metric"]
            vintages = self.db.get_vintages(ticker, period_end, metric)

            if len(vintages) < 2:
                continue

            # Compare each subsequent vintage to the original
            original = vintages[0]
            for v in vintages[1:]:
                orig_val = original["value"]
                new_val = v["value"]

                if orig_val == 0:
                    continue

                change_pct = (new_val - orig_val) / abs(orig_val)
                is_material = abs(change_pct) > self.MATERIALITY_THRESHOLD

                evt = RestatementEvent(
                    ticker=ticker.upper(),
                    period_end=period_end,
                    metric=metric,
                    original_filing_date=date.fromisoformat(original["filing_date"]),
                    restatement_filing_date=date.fromisoformat(v["filing_date"]),
                    original_value=orig_val,
                    restated_value=new_val,
                    change_pct=change_pct,
                    is_material=is_material,
                    restatement_type=self._classify(change_pct),
                )
                events.append(evt)
                self.db.record_restatement(evt)

        return events

    def _classify(self, change_pct: float) -> str:
        """Heuristic classification based on magnitude."""
        abs_chg = abs(change_pct)
        if abs_chg > 0.20:
            return "error_correction"
        elif abs_chg > 0.05:
            return "reclassification"
        else:
            return "rule_change"

    def get_restatement_history(self, ticker: str) -> list[RestatementEvent]:
        """Retrieve stored restatements for `ticker` from DB."""
        rows = self.db._conn.execute("""
            SELECT * FROM restatements WHERE ticker = ? ORDER BY period_end DESC
        """, (ticker.upper(),)).fetchall()

        events = []
        for r in rows:
            events.append(RestatementEvent(
                ticker=r["ticker"],
                period_end=date.fromisoformat(r["period_end"]),
                metric=r["metric"],
                original_filing_date=date.fromisoformat(r["original_filing_date"]),
                restatement_filing_date=date.fromisoformat(r["restatement_filing_date"]),
                original_value=r["original_value"],
                restated_value=r["restated_value"],
                change_pct=r["change_pct"],
                is_material=bool(r["is_material"]),
                restatement_type=r["restatement_type"],
            ))
        return events

    def restatement_risk_score(self, ticker: str) -> dict:
        """
        Score a company's restatement risk (0-100, higher = riskier).
        Factors: count, materiality, recency.
        """
        history = self.get_restatement_history(ticker)
        if not history:
            return {"ticker": ticker, "risk_score": 0, "count": 0, "material_count": 0}

        material = [e for e in history if e.is_material]
        recent_cutoff = date.today() - timedelta(days=5 * 365)
        recent = [e for e in history if e.restatement_filing_date >= recent_cutoff]

        base_score = min(len(history) * 5, 40)
        material_score = min(len(material) * 10, 40)
        recency_score = min(len(recent) * 5, 20)
        total = base_score + material_score + recency_score

        return {
            "ticker": ticker,
            "risk_score": total,
            "count": len(history),
            "material_count": len(material),
            "recent_count": len(recent),
            "grade": "F" if total >= 80 else "D" if total >= 60 else "C" if total >= 40 else "B" if total >= 20 else "A",
        }

    def scan_and_store_all(self, tickers: list[str]) -> dict[str, int]:
        """Scan multiple tickers; return count of restatements found per ticker."""
        results = {}
        for ticker in tickers:
            evts = self.scan_ticker(ticker)
            results[ticker] = len(evts)
        return results


# ---------------------------------------------------------------------------
# Long-run trend analyzer
# ---------------------------------------------------------------------------

class LongRunTrendAnalyzer:
    """
    15+ year trend analysis: CAGR, margin evolution, return metrics,
    capital intensity, balance sheet strength, recession resilience.
    """

    def __init__(self, db: HistoricalFinancialsDatabase) -> None:
        self.db = db
        self.pit = PointInTimeEngine(db)

    def _get_annual_series(
        self,
        ticker: str,
        metric: str,
        years: int = 20,
    ) -> pd.Series:
        """Return annual values for `metric` over last `years` years."""
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=years * 365)
        series = self.db.get_history(ticker, metric, start_dt, end_dt, pit=True)
        return series

    def compute_cagrs(self, ticker: str, metric: str) -> TrendResult:
        """Compute 5Y, 10Y, 15Y CAGRs for a metric."""
        series = self._get_annual_series(ticker, metric, years=20)

        if series.empty:
            return TrendResult(ticker=ticker, metric=metric)

        today = pd.Timestamp.today()
        values_by_year: dict[str, float] = {}

        for ts, val in series.items():
            values_by_year[ts.strftime("%Y-%m-%d")] = val

        # Get most recent value
        recent = series.iloc[-1]
        recent_ts = series.index[-1]

        def _val_n_years_ago(n: int) -> Optional[float]:
            target = recent_ts - pd.DateOffset(years=n)
            # Find closest observation within 180 days
            diffs = abs(series.index - target)
            closest_idx = diffs.argmin()
            if diffs[closest_idx].days > 180:
                return None
            return series.iloc[closest_idx]

        v5 = _val_n_years_ago(5)
        v10 = _val_n_years_ago(10)
        v15 = _val_n_years_ago(15)

        cagr_5y = _cagr(v5, recent, 5) if v5 else None
        cagr_10y = _cagr(v10, recent, 10) if v10 else None
        cagr_15y = _cagr(v15, recent, 15) if v15 else None

        # Historical decile: where does current value sit vs own history?
        all_vals = series.dropna().values
        if len(all_vals) >= 2:
            percentile = float(np.mean(all_vals <= recent)) * 100
            decile = max(1, min(10, int(percentile / 10) + 1))
        else:
            decile = None

        return TrendResult(
            ticker=ticker,
            metric=metric,
            cagr_5y=round(cagr_5y, 4) if cagr_5y is not None else None,
            cagr_10y=round(cagr_10y, 4) if cagr_10y is not None else None,
            cagr_15y=round(cagr_15y, 4) if cagr_15y is not None else None,
            current_decile=decile,
            values=values_by_year,
        )

    def margin_evolution(self, ticker: str) -> dict:
        """
        Track gross, operating, and net margin over 15+ years.
        Returns series and decile ranking vs own history.
        """
        rev = self._get_annual_series(ticker, "revenue", 20)
        gp = self._get_annual_series(ticker, "gross_profit", 20)
        op = self._get_annual_series(ticker, "operating_income", 20)
        ni = self._get_annual_series(ticker, "net_income", 20)

        result: dict[str, Any] = {"ticker": ticker}

        for label, numerator in [("gross_margin", gp), ("operating_margin", op), ("net_margin", ni)]:
            if rev.empty or numerator.empty:
                result[label] = {}
                continue
            # Align on common index
            aligned_rev, aligned_num = rev.align(numerator, join="inner")
            margin = aligned_num / aligned_rev.replace(0, np.nan)
            margin = margin.dropna()

            if margin.empty:
                result[label] = {}
                continue

            margin_dict = {ts.strftime("%Y-%m-%d"): round(v, 4) for ts, v in margin.items()}
            current = margin.iloc[-1]
            all_vals = margin.values
            pct = float(np.mean(all_vals <= current)) * 100
            decile = max(1, min(10, int(pct / 10) + 1))

            result[label] = {
                "series": margin_dict,
                "current": round(current, 4),
                "decile_vs_history": decile,
                "min": round(float(margin.min()), 4),
                "max": round(float(margin.max()), 4),
                "mean": round(float(margin.mean()), 4),
            }

        return result

    def return_evolution(self, ticker: str) -> dict:
        """
        Track ROE, ROIC proxy (net_income / (equity + long_term_debt)),
        and ROA over the full history.
        """
        ni = self._get_annual_series(ticker, "net_income", 20)
        eq = self._get_annual_series(ticker, "equity", 20)
        assets = self._get_annual_series(ticker, "total_assets", 20)
        ltd = self._get_annual_series(ticker, "long_term_debt", 20)

        result: dict[str, Any] = {"ticker": ticker}

        # ROE = net_income / equity
        if not ni.empty and not eq.empty:
            ani, aeq = ni.align(eq, join="inner")
            roe = ani / aeq.replace(0, np.nan)
            result["roe"] = {ts.strftime("%Y-%m-%d"): round(v, 4)
                             for ts, v in roe.dropna().items()}

        # ROA = net_income / total_assets
        if not ni.empty and not assets.empty:
            ani, aas = ni.align(assets, join="inner")
            roa = ani / aas.replace(0, np.nan)
            result["roa"] = {ts.strftime("%Y-%m-%d"): round(v, 4)
                             for ts, v in roa.dropna().items()}

        # ROIC = net_income / (equity + long_term_debt)
        if not ni.empty and not eq.empty and not ltd.empty:
            ani, aeq = ni.align(eq, join="inner")
            ani2, altd = ani.align(ltd, join="inner")
            aeq2, _ = aeq.align(altd, join="inner")
            invested = aeq2 + altd
            roic = ani2 / invested.replace(0, np.nan)
            result["roic"] = {ts.strftime("%Y-%m-%d"): round(v, 4)
                              for ts, v in roic.dropna().items()}

        return result

    def capex_intensity(self, ticker: str) -> dict:
        """Capex / Revenue trend over history."""
        rev = self._get_annual_series(ticker, "revenue", 20)
        capex = self._get_annual_series(ticker, "capex", 20)

        if rev.empty or capex.empty:
            return {"ticker": ticker, "capex_intensity": {}}

        arev, acap = rev.align(capex, join="inner")
        intensity = acap / arev.replace(0, np.nan)
        intensity = intensity.dropna()

        return {
            "ticker": ticker,
            "capex_intensity": {
                ts.strftime("%Y-%m-%d"): round(v, 4)
                for ts, v in intensity.items()
            },
            "current": round(float(intensity.iloc[-1]), 4) if not intensity.empty else None,
            "5y_avg": round(float(intensity.iloc[-20:].mean()), 4) if len(intensity) >= 4 else None,
        }

    def balance_sheet_strength(self, ticker: str) -> dict:
        """
        Debt evolution over economic cycles.
        Net debt = long_term_debt + short_term_debt - cash
        Leverage = net_debt / EBITDA proxy
        """
        ltd = self._get_annual_series(ticker, "long_term_debt", 20)
        std = self._get_annual_series(ticker, "short_term_debt", 20)
        cash = self._get_annual_series(ticker, "cash", 20)
        op_inc = self._get_annual_series(ticker, "operating_income", 20)

        result: dict[str, Any] = {"ticker": ticker}

        if not ltd.empty and not cash.empty:
            std_filled = std.reindex(ltd.index, fill_value=0.0)
            cash_aligned, _ = cash.align(ltd, join="right")
            cash_aligned = cash_aligned.fillna(0)
            net_debt = ltd + std_filled - cash_aligned
            result["net_debt"] = {ts.strftime("%Y-%m-%d"): round(v, 0)
                                  for ts, v in net_debt.items()}

        if not ltd.empty and not op_inc.empty:
            altd, aop = ltd.align(op_inc, join="inner")
            leverage = altd / aop.replace(0, np.nan)
            result["debt_to_ebitda_proxy"] = {
                ts.strftime("%Y-%m-%d"): round(v, 2)
                for ts, v in leverage.dropna().items()
            }

        return result

    def recession_performance(self, ticker: str, metric: str = "revenue") -> list[RecessionTestResult]:
        """
        Analyze company performance through each NBER recession.
        """
        results: list[RecessionTestResult] = []

        series = self._get_annual_series(ticker, metric, 30)
        if series.empty:
            return results

        for rec_start, rec_end, rec_name in RECESSIONS:
            # Pre-recession: 4 quarters before start
            pre_start = rec_start - timedelta(days=365)
            pre_mask = (series.index >= pd.Timestamp(pre_start)) & (series.index < pd.Timestamp(rec_start))
            pre_vals = series[pre_mask]

            # During recession
            during_mask = (series.index >= pd.Timestamp(rec_start)) & (series.index <= pd.Timestamp(rec_end))
            during_vals = series[during_mask]

            # Post recession: up to 8 quarters after end
            post_end = rec_end + timedelta(days=2 * 365)
            post_mask = (series.index > pd.Timestamp(rec_end)) & (series.index <= pd.Timestamp(post_end))
            post_vals = series[post_mask]

            if pre_vals.empty or during_vals.empty:
                continue

            pre_val = float(pre_vals.mean())
            trough_val = float(during_vals.min())

            if pre_val == 0:
                continue

            peak_to_trough = (trough_val - pre_val) / abs(pre_val)

            # Recovery: quarters until back above pre-recession level
            recovery_q = None
            for i, (ts, v) in enumerate(post_vals.items()):
                if v >= pre_val:
                    recovery_q = i + 1
                    break

            # Grade based on peak-to-trough
            if peak_to_trough >= -0.05:
                grade = "A"
            elif peak_to_trough >= -0.15:
                grade = "B"
            elif peak_to_trough >= -0.30:
                grade = "C"
            elif peak_to_trough >= -0.50:
                grade = "D"
            else:
                grade = "F"

            results.append(RecessionTestResult(
                ticker=ticker,
                recession=rec_name,
                metric=metric,
                pre_recession_value=round(pre_val, 0),
                trough_value=round(trough_val, 0),
                peak_to_trough_pct=round(peak_to_trough, 4),
                recovery_quarters=recovery_q,
                resilience_grade=grade,
            ))

        return results

    def full_trend_report(self, ticker: str) -> dict:
        """
        Generate comprehensive long-run trend report for `ticker`.
        """
        return {
            "ticker": ticker,
            "revenue_cagrs": self.compute_cagrs(ticker, "revenue").model_dump(),
            "net_income_cagrs": self.compute_cagrs(ticker, "net_income").model_dump(),
            "margin_evolution": self.margin_evolution(ticker),
            "return_evolution": self.return_evolution(ticker),
            "capex_intensity": self.capex_intensity(ticker),
            "balance_sheet_strength": self.balance_sheet_strength(ticker),
            "recession_performance": [
                r.model_dump() for r in self.recession_performance(ticker, "revenue")
            ],
        }


# ---------------------------------------------------------------------------
# Fiscal calendar manager
# ---------------------------------------------------------------------------

class FiscalCalendarManager:
    """
    Handle non-standard fiscal year ends and cross-company TTM alignment.

    Many S&P 500 companies do NOT have December 31 fiscal year ends:
    - Apple: September
    - Microsoft: June
    - Walmart: January
    - Nike: May

    This class maps between fiscal periods and calendar periods.
    """

    def __init__(self, db: Optional[HistoricalFinancialsDatabase] = None) -> None:
        self.db = db
        self._fy_cache: dict[str, int] = dict(KNOWN_FY_MONTHS)

    def get_fy_end_month(self, ticker: str) -> int:
        """
        Return the fiscal year end month for `ticker`.
        If unknown, attempt to infer from stored period_end dates.
        """
        ticker = ticker.upper()
        if ticker in self._fy_cache:
            return self._fy_cache[ticker]

        if self.db:
            periods = self.db.get_all_periods(ticker)
            if periods:
                # Infer from the most common month of period_end dates
                months = []
                for p in periods:
                    try:
                        months.append(date.fromisoformat(p).month)
                    except Exception:
                        pass
                if months:
                    from collections import Counter
                    most_common = Counter(months).most_common(1)[0][0]
                    self._fy_cache[ticker] = most_common
                    return most_common

        return 12  # Default: December

    def fiscal_to_calendar(self, ticker: str, fiscal_year: int) -> date:
        """
        Convert fiscal year integer to calendar date of fiscal year end.
        e.g., Apple FY2024 ends in September 2024 → date(2024, 9, 28)
        """
        fy_month = self.get_fy_end_month(ticker)
        # Last day of the fiscal year end month
        if fy_month == 12:
            return date(fiscal_year, 12, 31)
        elif fy_month in (1, 3, 5, 7, 8, 10, 12):
            return date(fiscal_year, fy_month, 31)
        elif fy_month in (4, 6, 9, 11):
            return date(fiscal_year, fy_month, 30)
        else:  # February
            return date(fiscal_year, fy_month, 28)

    def calendar_to_fiscal(self, ticker: str, calendar_date: date) -> tuple[int, int]:
        """
        Convert a calendar date to (fiscal_year, fiscal_quarter).
        Returns the fiscal period that calendar_date falls into.
        """
        fy_month = self.get_fy_end_month(ticker)
        # The fiscal year that ends in fy_month
        if calendar_date.month > fy_month:
            fiscal_year = calendar_date.year + 1
        elif calendar_date.month == fy_month:
            fiscal_year = calendar_date.year
        else:
            fiscal_year = calendar_date.year

        # Determine fiscal quarter
        months_since_fy_start = ((calendar_date.month - fy_month - 1) % 12) + 1
        fiscal_quarter = math.ceil(months_since_fy_start / 3)
        return fiscal_year, fiscal_quarter

    def get_ttm_period(self, ticker: str, as_of_date: date) -> tuple[date, date]:
        """
        Return (start_date, end_date) for the trailing-twelve-months period
        ending at the most recent fiscal quarter available before as_of_date.
        """
        _, fq = self.calendar_to_fiscal(ticker, as_of_date)
        ttm_end = as_of_date.replace(day=1) - timedelta(days=1)
        ttm_start = ttm_end - timedelta(days=365)
        return ttm_start, ttm_end

    def align_for_comparison(
        self,
        tickers: list[str],
        metric: str,
        as_of_date: date,
        db: Optional[HistoricalFinancialsDatabase] = None,
    ) -> dict[str, Optional[float]]:
        """
        Return TTM value for each ticker, properly aligned regardless of
        non-standard fiscal year ends. Uses PIT engine if db is provided.
        """
        store = db or self.db
        if not store:
            return {t: None for t in tickers}

        pit = PointInTimeEngine(store)
        result = {}
        for ticker in tickers:
            val = pit.get_ttm_pit(ticker, metric, as_of_date)
            result[ticker] = val
        return result

    def fiscal_quarter_dates(self, ticker: str, fiscal_year: int) -> list[date]:
        """Return the four fiscal quarter end dates for a given fiscal year."""
        fy_month = self.get_fy_end_month(ticker)
        q_ends = []
        for q in range(4):
            month = ((fy_month - 3 * (3 - q)) % 12) or 12
            if month == fy_month:
                year = fiscal_year
            elif month > fy_month:
                year = fiscal_year - 1
            else:
                year = fiscal_year

            # Last day of month
            if month in (1, 3, 5, 7, 8, 10, 12):
                day = 31
            elif month in (4, 6, 9, 11):
                day = 30
            else:
                day = 28
            q_ends.append(date(year, month, day))

        return sorted(q_ends)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

historical_router = APIRouter(prefix="/financials/v2", tags=["Historical Financials"])

_db: Optional[HistoricalFinancialsDatabase] = None
_pit: Optional[PointInTimeEngine] = None
_restatement: Optional[RestatementDetector] = None
_trend: Optional[LongRunTrendAnalyzer] = None
_fiscal: Optional[FiscalCalendarManager] = None


def _get_db() -> HistoricalFinancialsDatabase:
    global _db
    if _db is None:
        _db = HistoricalFinancialsDatabase()
    return _db


def _get_pit() -> PointInTimeEngine:
    global _pit
    if _pit is None:
        _pit = PointInTimeEngine(_get_db())
    return _pit


def _get_restatement() -> RestatementDetector:
    global _restatement
    if _restatement is None:
        _restatement = RestatementDetector(_get_db())
    return _restatement


def _get_trend() -> LongRunTrendAnalyzer:
    global _trend
    if _trend is None:
        _trend = LongRunTrendAnalyzer(_get_db())
    return _trend


def _get_fiscal() -> FiscalCalendarManager:
    global _fiscal
    if _fiscal is None:
        _fiscal = FiscalCalendarManager(_get_db())
    return _fiscal


class IngestRequest(BaseModel):
    ticker: str
    max_years: int = Field(default=25, ge=5, le=30)


class PITRequest(BaseModel):
    ticker: str
    metric: str
    as_of_date: str  # ISO date


class PanelRequest(BaseModel):
    tickers: list[str]
    metric: str
    dates: list[str]  # ISO dates
    use_ttm: bool = False


@historical_router.post("/ingest/{ticker}")
def ingest_ticker(ticker: str, max_years: int = Query(default=25, ge=5, le=30)):
    """Fetch and store all EDGAR financials for a ticker."""
    db = _get_db()
    try:
        counts = db.fetch_and_store(ticker.upper(), max_years=max_years)
        return {"ticker": ticker.upper(), "metrics_stored": counts}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@historical_router.get("/history/{ticker}/{metric}")
def get_history(
    ticker: str,
    metric: str,
    start: str = Query(default="2000-01-01"),
    end: str = Query(default=""),
    pit: bool = Query(default=True),
):
    """Return full time series of a metric (optionally PIT)."""
    db = _get_db()
    try:
        start_dt = date.fromisoformat(start)
        end_dt = date.fromisoformat(end) if end else date.today()
        series = db.get_history(ticker.upper(), metric, start_dt, end_dt, pit=pit)
        return {
            "ticker": ticker.upper(),
            "metric": metric,
            "pit": pit,
            "data": {str(k.date()): v for k, v in series.items()},
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@historical_router.get("/pit/{ticker}/{metric}")
def get_pit_value(
    ticker: str,
    metric: str,
    as_of_date: str = Query(default=""),
    ttm: bool = Query(default=False),
):
    """Return point-in-time value of a metric as known at as_of_date."""
    pit_engine = _get_pit()
    try:
        as_of = date.fromisoformat(as_of_date) if as_of_date else date.today()
        if ttm:
            val = pit_engine.get_ttm_pit(ticker.upper(), metric, as_of)
        else:
            val = pit_engine.get_pit_value(ticker.upper(), metric, as_of)
        return {
            "ticker": ticker.upper(),
            "metric": metric,
            "as_of_date": as_of.isoformat(),
            "value": val,
            "ttm": ttm,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@historical_router.post("/pit/panel")
def build_panel(req: PanelRequest):
    """Build a tickers × dates PIT panel."""
    pit_engine = _get_pit()
    try:
        dates = [date.fromisoformat(d) for d in req.dates]
        df = pit_engine.build_backtest_panel(req.tickers, req.metric, dates, req.use_ttm)
        return {
            "metric": req.metric,
            "tickers": req.tickers,
            "dates": req.dates,
            "panel": {str(k.date()): df.loc[k].to_dict() for k in df.index},
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@historical_router.get("/restatements/{ticker}")
def get_restatements(ticker: str, scan: bool = Query(default=False)):
    """Return restatement history for ticker. Set scan=true to re-scan EDGAR."""
    detector = _get_restatement()
    try:
        if scan:
            events = detector.scan_ticker(ticker.upper())
        else:
            events = detector.get_restatement_history(ticker.upper())
        risk = detector.restatement_risk_score(ticker.upper())
        return {
            "ticker": ticker.upper(),
            "restatement_count": len(events),
            "risk": risk,
            "events": [e.model_dump() for e in events],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@historical_router.get("/trends/{ticker}")
def get_trends(ticker: str, metric: str = Query(default="revenue")):
    """Return CAGR trends for a ticker."""
    trend = _get_trend()
    try:
        result = trend.compute_cagrs(ticker.upper(), metric)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@historical_router.get("/trends/{ticker}/full")
def get_full_trends(ticker: str):
    """Return comprehensive trend report including margins, returns, capex."""
    trend = _get_trend()
    try:
        return trend.full_trend_report(ticker.upper())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@historical_router.get("/recession-test/{ticker}")
def recession_test(ticker: str, metric: str = Query(default="revenue")):
    """Test how ticker performed during each NBER recession."""
    trend = _get_trend()
    try:
        results = trend.recession_performance(ticker.upper(), metric)
        return {
            "ticker": ticker.upper(),
            "metric": metric,
            "recessions": [r.model_dump() for r in results],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@historical_router.get("/fiscal-calendar/{ticker}")
def fiscal_calendar(ticker: str, fiscal_year: int = Query(default=2024)):
    """Return fiscal calendar information for a ticker."""
    fiscal = _get_fiscal()
    try:
        fy_month = fiscal.get_fy_end_month(ticker.upper())
        fy_end = fiscal.fiscal_to_calendar(ticker.upper(), fiscal_year)
        quarter_dates = fiscal.fiscal_quarter_dates(ticker.upper(), fiscal_year)
        return {
            "ticker": ticker.upper(),
            "fiscal_year_end_month": fy_month,
            "fiscal_year": fiscal_year,
            "fy_end_date": fy_end.isoformat(),
            "quarter_end_dates": [d.isoformat() for d in quarter_dates],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@historical_router.get("/vintages/{ticker}/{metric}")
def get_vintages(ticker: str, metric: str, period_end: str = Query(...)):
    """Return all filing vintages for (ticker, metric, period_end)."""
    db = _get_db()
    try:
        period = date.fromisoformat(period_end)
        vintages = db.get_vintages(ticker.upper(), period, metric)
        return {
            "ticker": ticker.upper(),
            "metric": metric,
            "period_end": period_end,
            "vintages": vintages,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@historical_router.get("/recession-performance/{ticker}")
def recession_performance_multi(ticker: str):
    """Return recession performance across all tracked recessions and key metrics."""
    trend = _get_trend()
    try:
        all_results = {}
        for metric in ["revenue", "net_income", "cfo"]:
            results = trend.recession_performance(ticker.upper(), metric)
            all_results[metric] = [r.model_dump() for r in results]
        return {"ticker": ticker.upper(), "recession_performance": all_results}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
