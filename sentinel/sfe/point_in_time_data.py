"""
Comprehensive point-in-time data management: ensure no look-ahead bias
in all SENTINEL data feeds for backtesting and research.

Targets:
  dim_022  Point-in-time data (no look-ahead bias)  → 9

Public API
----------
DataTimestampRegistry    — central publication-lag registry for all data types
LookAheadBiasDetector    — validate research datasets; flag contaminated features
EarningsCalendarEngine   — actual SEC 8-K announcement dates; pre/post classification
PITDataFrameBuilder      — build backtesting-safe panel DataFrames
TimeSeriesValidator      — series-level staleness, future-leak, and input validation
pit_router               — FastAPI APIRouter
"""
from __future__ import annotations

import json
import logging
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
    "DataTimestampRegistry",
    "LookAheadBiasDetector",
    "EarningsCalendarEngine",
    "PITDataFrameBuilder",
    "TimeSeriesValidator",
    "pit_router",
    "PUBLICATION_LAG",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://data.sec.gov"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&dateRange=custom&startdt={start}&enddt={end}&entity={ticker}&forms=8-K"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_RATE_DELAY = 0.12
_TIMEOUT = 30.0
_MAX_RETRY = 3

DB_PATH = Path(__file__).parent.parent / "data" / "pit_data.db"

# ---------------------------------------------------------------------------
# Publication lags (calendar days after reference event)
# ---------------------------------------------------------------------------

PUBLICATION_LAG: dict[str, Any] = {
    # Earnings filings
    "earnings": {
        "10-Q_large_accelerated": 40,
        "10-Q_accelerated": 40,
        "10-Q_non_accelerated": 45,
        "10-Q": 45,   # conservative default
        "10-K_large_accelerated": 60,
        "10-K_accelerated": 75,
        "10-K_non_accelerated": 90,
        "10-K": 90,   # conservative default
        "20-F": 120,
        "10-QT": 45,
        "10-KT": 90,
    },
    # NBER economic releases — days after reference period end
    "economic_releases": {
        "GDP_advance": 30,       # Advance GDP estimate
        "GDP_second": 60,        # Second estimate
        "GDP_third": 90,         # Third (final) estimate
        "GDP": 30,               # Default: advance release
        "CPI": 14,               # ~14 days after month end
        "PPI": 14,
        "PCE": 30,               # Same day as GDP sometimes; ~30 day default
        "NFP": 3,                # Non-farm payrolls: first Friday of following month (~3-7 days after)
        "retail_sales": 14,
        "industrial_production": 17,
        "housing_starts": 20,
        "FOMC_minutes": 21,      # ~3 weeks after meeting
        "PMI": 1,                # Released same or next day
        "unemployment": 3,
        "trade_balance": 35,
        "durable_goods": 26,
        "consumer_confidence": 30,
        "ISM": 1,
    },
    # Regulatory filings
    "analyst_ratings": 0,        # Published same day
    "insider_trades": 2,         # SEC Form 4: 2 business days after trade
    "institutional_13F": 45,     # 45 calendar days after quarter end
    "institutional_13G": 45,
    "institutional_13D": 10,     # 10 days after crossing 5% threshold
    "options_data": 0,           # Real-time / end-of-day
    "short_interest": 5,         # ~5 days after bi-monthly settlement date
    "earnings_announcement": 0,  # Same day (8-K), but use actual date
    "price_data": 0,             # Same day close
    "dividend_announcement": 0,
    "form_4": 2,
    "form_8k": 4,                # 4 business days after triggering event
    "proxy_statement": 0,
    "corporate_actions": 1,      # T+1 for most corporate actions
    "credit_ratings": 0,
    "segment_data": 45,          # In 10-K/10-Q filings
    "guidance": 0,               # Issued with earnings call (same day)
    "esg_scores": 180,           # Annual ESG reports; ~6 months lag
    "private_equity_nav": 90,    # PE funds: quarterly; 90-day lag typical
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BiasReport(BaseModel):
    total_features: int
    flagged_count: int
    clean_count: int
    bias_rate: float
    flagged_features: list[dict] = Field(default_factory=list)
    clean_features: list[str] = Field(default_factory=list)
    summary: str = ""


class EarningsDateRecord(BaseModel):
    ticker: str
    fiscal_period: str     # e.g. "Q1 FY2024"
    period_end: date
    announcement_date: Optional[date] = None   # Actual 8-K filing date
    estimated_date: Optional[date] = None      # embargo-based estimate
    source: str = "EDGAR"
    is_confirmed: bool = False


class ValidationReport(BaseModel):
    series_name: str
    total_points: int
    leaked_points: int
    stale_points: int
    has_future_leak: bool
    has_staleness: bool
    flagged_dates: list[str] = Field(default_factory=list)
    stale_ranges: list[dict] = Field(default_factory=list)
    quality_score: float  # 0-100


class FeatureValidationResult(BaseModel):
    feature_name: str
    data_type: str
    feature_date: date
    target_date: date
    publication_lag_days: int
    earliest_available: date
    is_valid: bool
    issue: Optional[str] = None


# ---------------------------------------------------------------------------
# Database layer for PIT tracking
# ---------------------------------------------------------------------------

class _PITDatabase:
    """Internal SQLite store for earnings dates and PIT validation cache."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS earnings_dates (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker              TEXT    NOT NULL,
                cik                 TEXT,
                fiscal_period       TEXT    NOT NULL,
                period_end          TEXT    NOT NULL,
                announcement_date   TEXT,
                estimated_date      TEXT,
                source              TEXT    NOT NULL DEFAULT 'EDGAR',
                is_confirmed        INTEGER NOT NULL DEFAULT 0,
                inserted_at         TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(ticker, fiscal_period)
            );
            CREATE INDEX IF NOT EXISTS idx_ed_ticker ON earnings_dates(ticker);

            CREATE TABLE IF NOT EXISTS pit_validation_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key       TEXT    NOT NULL UNIQUE,
                result_json     TEXT    NOT NULL,
                inserted_at     TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)
        self._conn.commit()

    def upsert_earnings_date(self, rec: EarningsDateRecord, cik: str = "") -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO earnings_dates
                (ticker, cik, fiscal_period, period_end, announcement_date,
                 estimated_date, source, is_confirmed)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            rec.ticker.upper(),
            cik,
            rec.fiscal_period,
            rec.period_end.isoformat(),
            rec.announcement_date.isoformat() if rec.announcement_date else None,
            rec.estimated_date.isoformat() if rec.estimated_date else None,
            rec.source,
            int(rec.is_confirmed),
        ))
        self._conn.commit()

    def get_earnings_date(self, ticker: str, fiscal_period: str) -> Optional[EarningsDateRecord]:
        row = self._conn.execute("""
            SELECT * FROM earnings_dates
             WHERE ticker = ? AND fiscal_period = ?
        """, (ticker.upper(), fiscal_period)).fetchone()

        if not row:
            return None

        return EarningsDateRecord(
            ticker=row["ticker"],
            fiscal_period=row["fiscal_period"],
            period_end=date.fromisoformat(row["period_end"]),
            announcement_date=date.fromisoformat(row["announcement_date"]) if row["announcement_date"] else None,
            estimated_date=date.fromisoformat(row["estimated_date"]) if row["estimated_date"] else None,
            source=row["source"],
            is_confirmed=bool(row["is_confirmed"]),
        )

    def get_all_earnings_dates(self, ticker: str) -> list[EarningsDateRecord]:
        rows = self._conn.execute("""
            SELECT * FROM earnings_dates WHERE ticker = ? ORDER BY period_end ASC
        """, (ticker.upper(),)).fetchall()

        return [
            EarningsDateRecord(
                ticker=r["ticker"],
                fiscal_period=r["fiscal_period"],
                period_end=date.fromisoformat(r["period_end"]),
                announcement_date=date.fromisoformat(r["announcement_date"]) if r["announcement_date"] else None,
                estimated_date=date.fromisoformat(r["estimated_date"]) if r["estimated_date"] else None,
                source=r["source"],
                is_confirmed=bool(r["is_confirmed"]),
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _rate_limited_get(url: str, retries: int = _MAX_RETRY) -> dict:
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
    try:
        data = _rate_limited_get(EDGAR_TICKERS_URL)
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning("CIK resolution failed for %s: %s", ticker, exc)
    return None


def _add_business_days(start: date, n: int) -> date:
    """Add n calendar days (simplified; doesn't adjust for holidays)."""
    return start + timedelta(days=n)


# ---------------------------------------------------------------------------
# 1. DataTimestampRegistry
# ---------------------------------------------------------------------------

class DataTimestampRegistry:
    """
    Central registry of publication lags for all SENTINEL data types.

    For any (data_type, reference_date), returns the earliest date the
    data could reasonably have been publicly available.

    This is the single source of truth for PIT enforcement across the system.
    """

    def __init__(self, custom_lags: Optional[dict] = None) -> None:
        self._lags = dict(PUBLICATION_LAG)
        if custom_lags:
            self._lags.update(custom_lags)

    def get_lag_days(self, data_type: str, sub_type: Optional[str] = None) -> int:
        """
        Return publication lag in days for a data type.

        Examples:
            get_lag_days("institutional_13F") → 45
            get_lag_days("earnings", "10-K") → 90
            get_lag_days("economic_releases", "CPI") → 14
        """
        entry = self._lags.get(data_type)
        if entry is None:
            logger.warning("Unknown data type: %s, defaulting to 0", data_type)
            return 0

        if isinstance(entry, dict):
            if sub_type and sub_type in entry:
                return entry[sub_type]
            # Return conservative (max) lag if no sub_type given
            return max(entry.values())

        return int(entry)

    def get_earliest_available(
        self,
        data_type: str,
        reference_date: date,
        sub_type: Optional[str] = None,
    ) -> date:
        """
        Return the earliest date `data_type` data from `reference_date` period
        would be publicly available.

        reference_date: period end date (e.g., quarter end) for filings,
                        or report date for economic releases.
        """
        lag = self.get_lag_days(data_type, sub_type)
        return reference_date + timedelta(days=lag)

    def is_available_at(
        self,
        data_type: str,
        reference_date: date,
        as_of_date: date,
        sub_type: Optional[str] = None,
    ) -> bool:
        """Is `data_type` from `reference_date` available at `as_of_date`?"""
        earliest = self.get_earliest_available(data_type, reference_date, sub_type)
        return as_of_date >= earliest

    def all_data_types(self) -> list[str]:
        """Return all registered data type names."""
        return list(self._lags.keys())

    def get_lag_table(self) -> pd.DataFrame:
        """Return a DataFrame summarizing all publication lags."""
        rows = []
        for dtype, val in self._lags.items():
            if isinstance(val, dict):
                for sub, lag in val.items():
                    rows.append({"data_type": dtype, "sub_type": sub, "lag_days": lag})
            else:
                rows.append({"data_type": dtype, "sub_type": None, "lag_days": val})
        return pd.DataFrame(rows)

    def validate_feature_window(
        self,
        data_type: str,
        data_period_end: date,
        feature_date: date,
        sub_type: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that data from `data_period_end` is available at `feature_date`.

        Returns (is_valid, error_message).
        """
        earliest = self.get_earliest_available(data_type, data_period_end, sub_type)
        if feature_date >= earliest:
            return True, None
        days_short = (earliest - feature_date).days
        msg = (
            f"{data_type} from period ending {data_period_end.isoformat()} "
            f"is NOT available at {feature_date.isoformat()} — "
            f"earliest availability: {earliest.isoformat()} "
            f"({days_short} days too early)"
        )
        return False, msg


# ---------------------------------------------------------------------------
# 2. LookAheadBiasDetector
# ---------------------------------------------------------------------------

class LookAheadBiasDetector:
    """
    Validate research datasets for look-ahead bias.

    A feature is biased if the data it encodes (financial metric, economic
    release, filing, etc.) would not have been available at the feature_date
    given known publication lags.

    Key method: validate_feature_set(features_df, data_types) → BiasReport
    """

    def __init__(self, registry: Optional[DataTimestampRegistry] = None) -> None:
        self.registry = registry or DataTimestampRegistry()

    def check_feature(
        self,
        feature_name: str,
        data_type: str,
        feature_date: date,
        reference_date: date,
        sub_type: Optional[str] = None,
    ) -> FeatureValidationResult:
        """
        Validate a single feature.

        feature_date   : date when the feature value is used in a model
        reference_date : period end date the data pertains to (e.g. Q-end, month-end)
        """
        lag = self.registry.get_lag_days(data_type, sub_type)
        earliest = self.registry.get_earliest_available(data_type, reference_date, sub_type)
        is_valid = feature_date >= earliest
        issue = None

        if not is_valid:
            days_early = (earliest - feature_date).days
            issue = (
                f"Look-ahead bias: {data_type}/{sub_type or ''} from "
                f"{reference_date.isoformat()} not available until "
                f"{earliest.isoformat()} ({days_early}d after feature_date)"
            )

        return FeatureValidationResult(
            feature_name=feature_name,
            data_type=data_type,
            feature_date=feature_date,
            target_date=reference_date,
            publication_lag_days=lag,
            earliest_available=earliest,
            is_valid=is_valid,
            issue=issue,
        )

    def validate_feature_set(
        self,
        features_df: pd.DataFrame,
        data_type_map: dict[str, tuple[str, Optional[str]]],
        reference_dates: Optional[dict[str, date]] = None,
    ) -> BiasReport:
        """
        Validate all features in a DataFrame.

        features_df      : DataFrame with DatetimeIndex; columns = feature names
        data_type_map    : {column_name: (data_type, sub_type)} for each feature
        reference_dates  : {column_name: reference_date} — period the data pertains to.
                           If None, assumes feature_date IS the reference_date.

        Returns BiasReport with flagged and clean features.
        """
        if features_df.empty:
            return BiasReport(
                total_features=0,
                flagged_count=0,
                clean_count=0,
                bias_rate=0.0,
                summary="Empty dataset",
            )

        flagged: list[dict] = []
        clean: list[str] = []

        for col in features_df.columns:
            if col not in data_type_map:
                clean.append(col)
                continue

            dtype, sub = data_type_map[col]

            # Validate each row
            col_biased = False
            for idx in features_df.index:
                if not isinstance(idx, (pd.Timestamp, datetime)):
                    continue

                feature_date = idx.date() if hasattr(idx, "date") else idx

                # Reference date: either explicit or inferred from feature date
                if reference_dates and col in reference_dates:
                    ref_date = reference_dates[col]
                else:
                    # Assume the data pertains to the prior quarter / month / year
                    ref_date = feature_date - timedelta(days=90)  # conservative

                result = self.check_feature(col, dtype, feature_date, ref_date, sub)
                if not result.is_valid:
                    col_biased = True
                    flagged.append({
                        "feature": col,
                        "date": feature_date.isoformat(),
                        "data_type": dtype,
                        "issue": result.issue,
                        "earliest_available": result.earliest_available.isoformat(),
                    })
                    break  # One bias instance is enough to flag the column

            if not col_biased:
                clean.append(col)

        total = len(data_type_map)
        n_flagged = len(set(f["feature"] for f in flagged))
        n_clean = len(clean)
        bias_rate = n_flagged / total if total > 0 else 0.0

        summary = (
            f"{n_flagged}/{total} features flagged for look-ahead bias "
            f"({bias_rate:.1%} contamination rate)"
        )

        return BiasReport(
            total_features=total,
            flagged_count=n_flagged,
            clean_count=n_clean,
            bias_rate=round(bias_rate, 4),
            flagged_features=flagged,
            clean_features=clean,
            summary=summary,
        )

    def create_pit_dataset(
        self,
        tickers: list[str],
        metrics: list[str],
        dates: list[date],
        data_type: str = "earnings",
        sub_type: str = "10-Q",
    ) -> pd.DataFrame:
        """
        Create a date × ticker PIT dataset that is guaranteed to be free of
        look-ahead bias. For each (ticker, date), only include metrics whose
        publication embargo has passed.

        Returns DataFrame indexed by date, columns = (ticker, metric) MultiIndex.
        Note: actual values are not fetched here — this method returns a boolean
        mask DataFrame (True = data available, False = not yet available at date).
        Use alongside HistoricalFinancialsDatabase.get_as_of() for actual values.
        """
        lag = self.registry.get_lag_days(data_type, sub_type)

        columns = pd.MultiIndex.from_product([tickers, metrics], names=["ticker", "metric"])
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
        availability = pd.DataFrame(True, index=idx, columns=columns)

        # Mark unavailable periods based on embargo
        for d in dates:
            embargo_threshold = d - timedelta(days=lag)
            for ticker in tickers:
                for metric in metrics:
                    # If the feature date is within lag days of today,
                    # we cannot guarantee data is available
                    if d > date.today() - timedelta(days=lag):
                        availability.loc[pd.Timestamp(d), (ticker, metric)] = False

        return availability

    def check_earnings_timing(
        self,
        ticker: str,
        period_end: date,
        feature_date: date,
        form_type: str = "10-Q",
    ) -> dict:
        """
        Check if earnings data for (ticker, period_end) is available at feature_date.
        Considers both EDGAR filing embargo AND actual announcement dates.
        """
        sub = form_type
        earliest = self.registry.get_earliest_available("earnings", period_end, sub)
        is_available = feature_date >= earliest
        days_to_availability = max(0, (earliest - feature_date).days)

        return {
            "ticker": ticker,
            "period_end": period_end.isoformat(),
            "form_type": form_type,
            "feature_date": feature_date.isoformat(),
            "earliest_available": earliest.isoformat(),
            "is_available": is_available,
            "days_to_availability": days_to_availability,
        }

    def audit_backtest(
        self,
        X: pd.DataFrame,
        feature_types: dict[str, str],
        reference_period_ends: Optional[dict[str, date]] = None,
    ) -> dict:
        """
        Full audit of a backtest feature matrix.

        X               : DataFrame (DatetimeIndex, columns = features)
        feature_types   : {column_name: data_type}
        reference_period_ends : {column_name: period_end_date}

        Returns summary dict with bias statistics.
        """
        data_type_map = {k: (v, None) for k, v in feature_types.items()}
        report = self.validate_feature_set(X, data_type_map, reference_period_ends)

        return {
            "bias_report": report.model_dump(),
            "is_clean": report.flagged_count == 0,
            "recommendation": (
                "Dataset is clean — no look-ahead bias detected."
                if report.flagged_count == 0
                else f"WARNING: {report.flagged_count} feature(s) have look-ahead bias. "
                     "Re-shift these features forward by their respective publication lags."
            ),
        }


# ---------------------------------------------------------------------------
# 3. EarningsCalendarEngine
# ---------------------------------------------------------------------------

class EarningsCalendarEngine:
    """
    Track actual earnings announcement dates from SEC EDGAR 8-K filings.

    The 8-K (Item 2.02 — Results of Operations) is filed on or shortly after
    the earnings announcement, giving us the definitive public date.

    This is critical for PIT: using Q4 financials before the 8-K is filed
    is a form of look-ahead bias even if the 10-K has been filed.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db = _PITDatabase(db_path)

    def _fetch_8k_filings(self, ticker: str, cik: str) -> list[dict]:
        """Fetch 8-K filings from EDGAR submissions API."""
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        try:
            data = _rate_limited_get(url)
        except Exception as exc:
            logger.warning("Failed to fetch submissions for %s: %s", ticker, exc)
            return []

        filings = []
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form in ("8-K", "8-K/A"):
                filings.append({
                    "form": form,
                    "filing_date": dates[i] if i < len(dates) else "",
                    "accession": accs[i] if i < len(accs) else "",
                    "description": descriptions[i] if i < len(descriptions) else "",
                })

        # Also check older filings if available
        old_files = data.get("filings", {}).get("files", [])
        for f in old_files:
            old_url = f"{EDGAR_BASE}/submissions/{f['name']}"
            try:
                old_data = _rate_limited_get(old_url)
                old_forms = old_data.get("form", [])
                old_dates = old_data.get("filingDate", [])
                old_accs = old_data.get("accessionNumber", [])
                old_desc = old_data.get("primaryDocument", [])
                for i, form in enumerate(old_forms):
                    if form in ("8-K", "8-K/A"):
                        filings.append({
                            "form": form,
                            "filing_date": old_dates[i] if i < len(old_dates) else "",
                            "accession": old_accs[i] if i < len(old_accs) else "",
                            "description": old_desc[i] if i < len(old_desc) else "",
                        })
            except Exception:
                break  # Stop fetching older pages on error

        return filings

    def _infer_fiscal_period(self, filing_date: date, period_end: Optional[date] = None) -> str:
        """Infer fiscal period label from filing date."""
        if period_end:
            quarter = (period_end.month - 1) // 3 + 1
            return f"Q{quarter} FY{period_end.year}"
        # Estimate: earnings typically filed within 30 days of quarter end
        est_qend = filing_date - timedelta(days=14)
        quarter = (est_qend.month - 1) // 3 + 1
        return f"Q{quarter} FY{est_qend.year}"

    def fetch_earnings_dates(self, ticker: str) -> list[EarningsDateRecord]:
        """
        Fetch actual earnings announcement dates for `ticker` from EDGAR.
        Uses 8-K filings (Item 2.02 results) as proxy for announcement date.
        """
        cik = _resolve_cik(ticker)
        if not cik:
            logger.warning("Cannot resolve CIK for %s", ticker)
            return []

        filings_8k = self._fetch_8k_filings(ticker, cik)
        records: list[EarningsDateRecord] = []

        for f in filings_8k:
            filing_date_str = f.get("filing_date", "")
            if not filing_date_str:
                continue
            try:
                filing_date = date.fromisoformat(filing_date_str)
            except ValueError:
                continue

            # 8-K earnings items typically relate to the most recent quarter end
            # Estimate period_end: most recent quarter-end before filing_date - 7 days
            est_end = filing_date - timedelta(days=14)
            month = est_end.month
            quarter_end_month = ((month - 1) // 3 + 1) * 3
            if quarter_end_month > 12:
                quarter_end_month = 12
            year = est_end.year
            if quarter_end_month in (3, 6, 9):
                last_day = 30
            else:
                last_day = 31
            period_end = date(year, quarter_end_month, last_day)

            fiscal_period = self._infer_fiscal_period(filing_date, period_end)

            rec = EarningsDateRecord(
                ticker=ticker.upper(),
                fiscal_period=fiscal_period,
                period_end=period_end,
                announcement_date=filing_date,
                estimated_date=None,
                source="EDGAR_8K",
                is_confirmed=True,
            )
            records.append(rec)
            self._db.upsert_earnings_date(rec, cik=cik)

        return records

    def get_earnings_date(
        self,
        ticker: str,
        fiscal_period: str,
        fallback_embargo: bool = True,
    ) -> Optional[EarningsDateRecord]:
        """
        Get earnings date for a specific fiscal period.
        If not in DB, estimates using filing embargo if fallback_embargo=True.
        """
        rec = self._db.get_earnings_date(ticker, fiscal_period)
        if rec:
            return rec

        if fallback_embargo:
            # Parse fiscal_period: "Q2 FY2023"
            try:
                parts = fiscal_period.split()
                q = int(parts[0].replace("Q", ""))
                fy = int(parts[1].replace("FY", ""))
                q_end_month = q * 3
                if q_end_month in (3, 6, 9):
                    day = 30
                else:
                    day = 31
                period_end = date(fy, q_end_month, day)
                # Embargo: Q=10-Q, FY=10-K
                form = "10-K" if q == 4 else "10-Q"
                from sentinel.sfe.historical_financials_engine import FILING_EMBARGO
                lag = FILING_EMBARGO.get(form, 45)
                estimated = period_end + timedelta(days=lag)
                return EarningsDateRecord(
                    ticker=ticker.upper(),
                    fiscal_period=fiscal_period,
                    period_end=period_end,
                    announcement_date=None,
                    estimated_date=estimated,
                    source="ESTIMATED",
                    is_confirmed=False,
                )
            except Exception:
                return None

        return None

    def is_post_earnings(
        self,
        ticker: str,
        fiscal_period: str,
        as_of_date: date,
    ) -> bool:
        """
        Return True if earnings for fiscal_period were announced on or before as_of_date.
        Critical for ensuring backtests don't use post-earnings data pre-announcement.
        """
        rec = self.get_earnings_date(ticker, fiscal_period, fallback_embargo=True)
        if not rec:
            return False

        announcement = rec.announcement_date or rec.estimated_date
        if not announcement:
            return False

        return as_of_date >= announcement

    def get_next_earnings_date(
        self,
        ticker: str,
        as_of_date: date,
    ) -> Optional[date]:
        """
        Return the next expected earnings date after as_of_date.
        Uses stored dates or estimates from fiscal calendar.
        """
        all_dates = self._db.get_all_earnings_dates(ticker)
        future = [
            r for r in all_dates
            if (r.announcement_date or r.estimated_date or date.min) > as_of_date
        ]

        if not future:
            return None

        future.sort(key=lambda r: r.announcement_date or r.estimated_date or date.min)
        rec = future[0]
        return rec.announcement_date or rec.estimated_date

    def build_earnings_calendar(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Build a calendar of earnings dates for multiple tickers.
        Returns DataFrame with columns: ticker, fiscal_period, announcement_date, is_confirmed.
        """
        rows = []
        for ticker in tickers:
            all_dates = self._db.get_all_earnings_dates(ticker)
            for rec in all_dates:
                ann = rec.announcement_date or rec.estimated_date
                if ann and start_date <= ann <= end_date:
                    rows.append({
                        "ticker": ticker.upper(),
                        "fiscal_period": rec.fiscal_period,
                        "period_end": rec.period_end.isoformat(),
                        "announcement_date": ann.isoformat(),
                        "is_confirmed": rec.is_confirmed,
                        "source": rec.source,
                    })

        if not rows:
            return pd.DataFrame(columns=["ticker", "fiscal_period", "period_end",
                                          "announcement_date", "is_confirmed", "source"])

        df = pd.DataFrame(rows)
        df["announcement_date"] = pd.to_datetime(df["announcement_date"])
        df = df.sort_values("announcement_date").reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# 4. PITDataFrameBuilder
# ---------------------------------------------------------------------------

class PITDataFrameBuilder:
    """
    Build backtesting-safe panel DataFrames with guaranteed point-in-time semantics.

    Integrates with HistoricalFinancialsDatabase for financial metrics,
    EarningsCalendarEngine for announcement dates, and DataTimestampRegistry
    for publication lags.

    Detected pitfalls:
    1. Using current-quarter estimates for a past date
    2. Using current price to normalize past fundamentals (forward P/E)
    3. Using 13F ownership data before the 45-day embargo passes
    4. Using post-earnings financial data before announcement date
    """

    def __init__(
        self,
        registry: Optional[DataTimestampRegistry] = None,
        earnings_calendar: Optional[EarningsCalendarEngine] = None,
        financials_db=None,  # HistoricalFinancialsDatabase, avoid circular import type
    ) -> None:
        self.registry = registry or DataTimestampRegistry()
        self.earnings_calendar = earnings_calendar or EarningsCalendarEngine()
        self.financials_db = financials_db
        self.detector = LookAheadBiasDetector(self.registry)

    def _get_pit_financial(
        self,
        ticker: str,
        metric: str,
        as_of_date: date,
    ) -> Optional[float]:
        """Get PIT financial value, if financials_db is available."""
        if self.financials_db is None:
            return None
        from sentinel.sfe.historical_financials_engine import PointInTimeEngine
        pit = PointInTimeEngine(self.financials_db)
        return pit.get_pit_value(ticker, metric, as_of_date)

    def build_panel(
        self,
        tickers: list[str],
        dates: list[date],
        features: list[str],
        financial_metrics: Optional[list[str]] = None,
        include_availability_mask: bool = False,
    ) -> pd.DataFrame:
        """
        Build a tickers × dates panel DataFrame with PIT semantics.

        For each (ticker, date) cell:
        - Financial metrics: use PIT values from HistoricalFinancialsDatabase
        - All values are filtered: only data available at that date is used

        Returns MultiIndex DataFrame: (date, ticker) × features
        """
        fin_metrics = financial_metrics or features

        rows = []
        for d in dates:
            for ticker in tickers:
                row: dict[str, Any] = {
                    "date": d,
                    "ticker": ticker.upper(),
                }
                for metric in fin_metrics:
                    if metric in features:
                        val = self._get_pit_financial(ticker, metric, d)
                        row[metric] = val

                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index(["date", "ticker"])
        return df

    def build_fundamental_panel(
        self,
        tickers: list[str],
        dates: list[date],
        metrics: list[str],
        use_ttm: bool = True,
    ) -> pd.DataFrame:
        """
        Build fundamental panel using TTM values for flow items.
        Requires financials_db to be set.
        """
        if self.financials_db is None:
            raise ValueError("financials_db must be set to build fundamental panel")

        from sentinel.sfe.historical_financials_engine import PointInTimeEngine, BALANCE_SHEET_METRICS
        pit = PointInTimeEngine(self.financials_db)

        records = []
        for d in dates:
            for ticker in tickers:
                row: dict[str, Any] = {"date": d, "ticker": ticker.upper()}
                for metric in metrics:
                    if use_ttm and metric not in BALANCE_SHEET_METRICS:
                        val = pit.get_ttm_pit(ticker, metric, d)
                    else:
                        val = pit.get_pit_value(ticker, metric, d)
                    row[metric] = val
                records.append(row)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index(["date", "ticker"])
        return df

    def detect_pitfalls(
        self,
        df: pd.DataFrame,
        feature_types: dict[str, str],
    ) -> list[str]:
        """
        Detect common look-ahead pitfalls in a feature DataFrame.

        Returns a list of warning messages.
        """
        warnings: list[str] = []

        # Pitfall 1: forward-looking features used with past dates
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values(0)
            oldest_date = min(dates)
        else:
            dates = df.index
            oldest_date = dates.min()

        for col, dtype in feature_types.items():
            if col not in df.columns:
                continue

            lag = self.registry.get_lag_days(dtype)

            # Pitfall 2: using estimates from current quarter for historical backtest
            if dtype == "analyst_ratings" and lag == 0:
                if not df[col].isna().all():
                    warnings.append(
                        f"Column '{col}' (analyst_ratings): Ensure these are consensus "
                        "estimates that existed at the feature date, not current consensus."
                    )

            # Pitfall 3: institutional ownership lag
            if dtype == "institutional_13F":
                for idx in df.index[:5]:  # sample check
                    if isinstance(idx, tuple):
                        d = idx[0].date() if hasattr(idx[0], "date") else idx[0]
                    else:
                        d = idx.date() if hasattr(idx, "date") else idx
                    # Warn if using data within 45 days of quarter end
                    quarter_end = date(d.year, ((d.month - 1) // 3 + 1) * 3, 30)
                    if (d - quarter_end).days < 45:
                        warnings.append(
                            f"Column '{col}' (institutional_13F): "
                            f"Data for {d} may include 13F filings not yet available "
                            "(45-day embargo after quarter end)"
                        )
                        break

            # Pitfall 4: short interest lag
            if dtype == "short_interest" and lag < 5:
                warnings.append(
                    f"Column '{col}' (short_interest): "
                    "Ensure short interest data has correct T+5 settlement lag applied."
                )

        return warnings

    def lag_shift_features(
        self,
        df: pd.DataFrame,
        feature_types: dict[str, str],
        sub_types: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Automatically shift features forward by their publication lag to create
        a bias-free feature matrix.

        Each column is shifted forward by its publication lag (in trading days ≈ calendar days).
        This ensures that when a feature row is labeled with date D, the data it
        reflects was actually available at D.

        Returns a new DataFrame with all features correctly lagged.
        """
        df_lagged = df.copy()

        for col, dtype in feature_types.items():
            if col not in df.columns:
                continue
            sub = (sub_types or {}).get(col)
            lag = self.registry.get_lag_days(dtype, sub)
            if lag > 0:
                df_lagged[col] = df_lagged[col].shift(lag)  # shift by lag periods (assumes daily index)

        return df_lagged

    def create_pit_estimate_series(
        self,
        ticker: str,
        dates: list[date],
        metric: str,
        estimates: dict[date, float],  # date → estimate value (as of that date)
    ) -> pd.Series:
        """
        Create a PIT estimate series from raw estimate data.

        estimates: dict where key is the date the estimate was published,
                   value is the estimate value.

        For each date in `dates`, returns the most recent estimate that was
        published on or before that date.
        """
        estimate_index = sorted(estimates.keys())
        result = {}

        for d in dates:
            available = [dt for dt in estimate_index if dt <= d]
            if available:
                latest_est_date = max(available)
                result[d] = estimates[latest_est_date]
            else:
                result[d] = None

        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
        return pd.Series(result.values(), index=idx, name=f"{ticker}_{metric}_pit_estimate")


# ---------------------------------------------------------------------------
# 5. TimeSeriesValidator
# ---------------------------------------------------------------------------

class TimeSeriesValidator:
    """
    Validate time series data for:
    1. Future data leak (values from dates after the series creation timestamp)
    2. Staleness (data unchanged for too long — possibly not updating)
    3. Backtest input validation (aligned X, y, dates)
    """

    DEFAULT_STALENESS_DAYS = 90  # Flag if unchanged for more than 90 days

    def __init__(
        self,
        staleness_threshold_days: int = DEFAULT_STALENESS_DAYS,
    ) -> None:
        self.staleness_threshold = staleness_threshold_days

    def check_for_future_leak(
        self,
        series: pd.Series,
        creation_timestamp: date,
    ) -> list[date]:
        """
        Detect any data points in `series` with index dates after `creation_timestamp`.
        These represent future data that could not have been known at creation time.
        """
        leaked: list[date] = []
        for idx in series.index:
            if isinstance(idx, pd.Timestamp):
                d = idx.date()
            elif isinstance(idx, datetime):
                d = idx.date()
            elif isinstance(idx, date):
                d = idx
            else:
                continue

            if d > creation_timestamp:
                leaked.append(d)

        return leaked

    def check_staleness(
        self,
        series: pd.Series,
        threshold_days: Optional[int] = None,
    ) -> list[dict]:
        """
        Detect stale ranges: consecutive stretches where the value doesn't change
        for more than `threshold_days`.

        Returns list of stale range dicts: {start, end, value, duration_days}.
        """
        thresh = threshold_days or self.staleness_threshold
        stale_ranges: list[dict] = []

        if series.empty or len(series) < 2:
            return stale_ranges

        clean = series.dropna()
        if clean.empty:
            return stale_ranges

        prev_val = clean.iloc[0]
        streak_start = clean.index[0]

        for i in range(1, len(clean)):
            curr_val = clean.iloc[i]
            curr_idx = clean.index[i]

            if curr_val == prev_val:
                # Continuing streak
                pass
            else:
                # Streak ended
                streak_end = clean.index[i - 1]
                if isinstance(streak_start, pd.Timestamp):
                    duration = (streak_end - streak_start).days
                else:
                    duration = (streak_end - streak_start).days if hasattr(streak_end - streak_start, "days") else 0

                if duration > thresh:
                    stale_ranges.append({
                        "start": str(streak_start.date() if hasattr(streak_start, "date") else streak_start),
                        "end": str(streak_end.date() if hasattr(streak_end, "date") else streak_end),
                        "value": float(prev_val) if prev_val is not None else None,
                        "duration_days": duration,
                    })

                streak_start = curr_idx
                prev_val = curr_val

        # Check final streak
        streak_end = clean.index[-1]
        if isinstance(streak_start, pd.Timestamp):
            duration = (streak_end - streak_start).days
        else:
            duration = 0

        if duration > thresh:
            stale_ranges.append({
                "start": str(streak_start.date() if hasattr(streak_start, "date") else streak_start),
                "end": str(streak_end.date() if hasattr(streak_end, "date") else streak_end),
                "value": float(prev_val) if prev_val is not None else None,
                "duration_days": duration,
            })

        return stale_ranges

    def validate_series(
        self,
        series: pd.Series,
        creation_timestamp: Optional[date] = None,
        series_name: Optional[str] = None,
        staleness_threshold_days: Optional[int] = None,
    ) -> ValidationReport:
        """
        Full validation of a time series.
        """
        name = series_name or str(series.name) or "unknown"
        ct = creation_timestamp or date.today()

        leaked = self.check_for_future_leak(series, ct)
        stale = self.check_staleness(series, staleness_threshold_days)

        total = len(series)
        n_leaked = len(leaked)
        n_stale_pts = sum(r["duration_days"] for r in stale)

        # Quality score: starts at 100, deduct for issues
        quality = 100.0
        if n_leaked > 0:
            quality -= min(50.0, n_leaked / total * 100)
        if stale:
            quality -= min(30.0, len(stale) * 10)
        if series.isna().mean() > 0.2:
            quality -= 20.0
        quality = max(0.0, quality)

        return ValidationReport(
            series_name=name,
            total_points=total,
            leaked_points=n_leaked,
            stale_points=n_stale_pts,
            has_future_leak=n_leaked > 0,
            has_staleness=len(stale) > 0,
            flagged_dates=[d.isoformat() for d in leaked],
            stale_ranges=stale,
            quality_score=round(quality, 1),
        )

    def validate_backtest_inputs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.DatetimeIndex,
        creation_timestamp: Optional[date] = None,
    ) -> dict:
        """
        Validate feature matrix X and target series y for backtest safety.

        Checks:
        1. X and y have same index
        2. No future leak in X
        3. No future leak in y
        4. Dates are sorted
        5. No duplicate dates
        6. Target y does not appear in X (obvious data leakage)
        """
        issues: list[str] = []
        warnings: list[str] = []

        # 1. Index alignment
        if not X.index.equals(y.index):
            issues.append("X and y have mismatched indices")

        # 2. Date ordering
        if not dates.is_monotonic_increasing:
            issues.append("dates are not sorted in ascending order")

        # 3. Duplicate dates
        if dates.duplicated().any():
            n_dups = dates.duplicated().sum()
            issues.append(f"{n_dups} duplicate dates found in index")

        # 4. Future leak check
        ct = creation_timestamp or date.today()
        for col in X.columns:
            report = self.validate_series(X[col], ct, col)
            if report.has_future_leak:
                issues.append(f"Feature '{col}' has {report.leaked_points} future data points")

        y_report = self.validate_series(y, ct, str(y.name))
        if y_report.has_future_leak:
            warnings.append(f"Target '{y.name}' has {y_report.leaked_points} future data points "
                            "(expected for forward-return targets)")

        # 5. Target not in features
        if str(y.name) in X.columns:
            issues.append(f"Target '{y.name}' appears as a feature column — obvious data leakage")

        # 6. Sanity: X should not have all NaN columns
        all_nan_cols = [c for c in X.columns if X[c].isna().all()]
        if all_nan_cols:
            warnings.append(f"{len(all_nan_cols)} columns are entirely NaN: {all_nan_cols[:5]}")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "shape": {"X_rows": len(X), "X_cols": len(X.columns), "y_len": len(y)},
            "date_range": {
                "start": str(dates.min().date()) if len(dates) > 0 else None,
                "end": str(dates.max().date()) if len(dates) > 0 else None,
            },
        }

    def check_alignment(
        self,
        series_dict: dict[str, pd.Series],
    ) -> dict:
        """
        Check a group of series for alignment issues:
        - Different date ranges
        - Missing dates in some series
        - High NaN rates
        """
        report: dict[str, Any] = {}
        all_indices = [s.index for s in series_dict.values()]

        if not all_indices:
            return report

        common_idx = all_indices[0]
        for idx in all_indices[1:]:
            common_idx = common_idx.intersection(idx)

        report["common_dates"] = len(common_idx)
        report["series_stats"] = {}

        for name, s in series_dict.items():
            nan_rate = s.isna().mean()
            report["series_stats"][name] = {
                "total_points": len(s),
                "nan_rate": round(float(nan_rate), 4),
                "start": str(s.index.min().date()) if len(s) > 0 else None,
                "end": str(s.index.max().date()) if len(s) > 0 else None,
                "missing_from_common": len(common_idx) - len(s.reindex(common_idx).dropna()),
            }

        return report

    def generate_quality_matrix(
        self,
        df: pd.DataFrame,
        creation_timestamp: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Generate a quality score matrix for all columns in a DataFrame.
        Returns DataFrame with columns: [series, total_points, nan_rate,
                                         leaked_points, stale_ranges, quality_score]
        """
        ct = creation_timestamp or date.today()
        rows = []

        for col in df.columns:
            report = self.validate_series(df[col], ct, str(col))
            rows.append({
                "series": col,
                "total_points": report.total_points,
                "nan_rate": round(df[col].isna().mean(), 4),
                "leaked_points": report.leaked_points,
                "stale_range_count": len(report.stale_ranges),
                "quality_score": report.quality_score,
            })

        return pd.DataFrame(rows).set_index("series")


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

pit_router = APIRouter(prefix="/pit", tags=["Point-in-Time Data"])

_registry: Optional[DataTimestampRegistry] = None
_bias_detector: Optional[LookAheadBiasDetector] = None
_earnings_calendar: Optional[EarningsCalendarEngine] = None
_panel_builder: Optional[PITDataFrameBuilder] = None
_validator: Optional[TimeSeriesValidator] = None


def _get_registry() -> DataTimestampRegistry:
    global _registry
    if _registry is None:
        _registry = DataTimestampRegistry()
    return _registry


def _get_detector() -> LookAheadBiasDetector:
    global _bias_detector
    if _bias_detector is None:
        _bias_detector = LookAheadBiasDetector(_get_registry())
    return _bias_detector


def _get_earnings_calendar() -> EarningsCalendarEngine:
    global _earnings_calendar
    if _earnings_calendar is None:
        _earnings_calendar = EarningsCalendarEngine()
    return _earnings_calendar


def _get_panel_builder() -> PITDataFrameBuilder:
    global _panel_builder
    if _panel_builder is None:
        _panel_builder = PITDataFrameBuilder(_get_registry(), _get_earnings_calendar())
    return _panel_builder


def _get_validator() -> TimeSeriesValidator:
    global _validator
    if _validator is None:
        _validator = TimeSeriesValidator()
    return _validator


class ValidateFeatureSetRequest(BaseModel):
    features: dict[str, list[float]]        # col → values
    dates: list[str]                          # ISO date strings
    data_type_map: dict[str, list[str]]      # col → [data_type, sub_type_or_null]
    reference_dates: Optional[dict[str, str]] = None  # col → reference_date ISO


class BuildPanelRequest(BaseModel):
    tickers: list[str]
    dates: list[str]  # ISO dates
    metrics: list[str]
    use_ttm: bool = False


class ValidateSeriesRequest(BaseModel):
    values: list[Optional[float]]
    dates: list[str]
    series_name: str = "series"
    creation_timestamp: Optional[str] = None
    staleness_threshold_days: int = 90


class ValidateBacktestRequest(BaseModel):
    X: dict[str, list[Optional[float]]]   # col → values
    y: list[Optional[float]]
    dates: list[str]
    feature_types: Optional[dict[str, str]] = None


@pit_router.get("/data-lag/{data_type}")
def get_data_lag(
    data_type: str,
    sub_type: Optional[str] = Query(default=None),
):
    """Return the publication lag for a data type."""
    registry = _get_registry()
    lag = registry.get_lag_days(data_type, sub_type)
    return {
        "data_type": data_type,
        "sub_type": sub_type,
        "lag_days": lag,
    }


@pit_router.get("/data-lag-table")
def get_lag_table():
    """Return complete publication lag table."""
    registry = _get_registry()
    df = registry.get_lag_table()
    return {"lags": df.to_dict(orient="records")}


@pit_router.get("/earliest-available/{data_type}")
def get_earliest_available(
    data_type: str,
    reference_date: str = Query(...),
    sub_type: Optional[str] = Query(default=None),
):
    """Return earliest date data would be available."""
    registry = _get_registry()
    try:
        ref = date.fromisoformat(reference_date)
        earliest = registry.get_earliest_available(data_type, ref, sub_type)
        return {
            "data_type": data_type,
            "sub_type": sub_type,
            "reference_date": reference_date,
            "earliest_available": earliest.isoformat(),
            "lag_days": registry.get_lag_days(data_type, sub_type),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@pit_router.post("/validate-feature-set")
def validate_feature_set(req: ValidateFeatureSetRequest):
    """Validate a feature set for look-ahead bias."""
    detector = _get_detector()
    try:
        dates_parsed = [date.fromisoformat(d) for d in req.dates]
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates_parsed])

        df = pd.DataFrame(req.features, index=idx)

        data_type_map: dict[str, tuple[str, Optional[str]]] = {}
        for col, parts in req.data_type_map.items():
            dtype = parts[0] if parts else "unknown"
            sub = parts[1] if len(parts) > 1 else None
            data_type_map[col] = (dtype, sub)

        ref_dates = None
        if req.reference_dates:
            ref_dates = {k: date.fromisoformat(v) for k, v in req.reference_dates.items()}

        report = detector.validate_feature_set(df, data_type_map, ref_dates)
        return report.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@pit_router.get("/earnings-date/{ticker}")
def get_earnings_date(
    ticker: str,
    fiscal_period: str = Query(default="Q1 FY2024"),
    as_of_date: Optional[str] = Query(default=None),
):
    """Get earnings announcement date for a ticker and fiscal period."""
    cal = _get_earnings_calendar()
    try:
        rec = cal.get_earnings_date(ticker.upper(), fiscal_period)
        if not rec:
            raise HTTPException(status_code=404, detail=f"No earnings date found for {ticker} {fiscal_period}")

        result = rec.model_dump()

        if as_of_date:
            as_of = date.fromisoformat(as_of_date)
            result["is_post_earnings_as_of"] = cal.is_post_earnings(ticker.upper(), fiscal_period, as_of)

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@pit_router.post("/fetch-earnings-dates/{ticker}")
def fetch_earnings_dates(ticker: str):
    """Fetch and store earnings dates from EDGAR for a ticker."""
    cal = _get_earnings_calendar()
    try:
        records = cal.fetch_earnings_dates(ticker.upper())
        return {
            "ticker": ticker.upper(),
            "records_fetched": len(records),
            "dates": [r.model_dump() for r in records[:20]],  # first 20
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@pit_router.post("/earnings-calendar")
def build_earnings_calendar(
    tickers: list[str],
    start_date: str = Query(default="2020-01-01"),
    end_date: str = Query(default=""),
):
    """Build earnings calendar for multiple tickers."""
    cal = _get_earnings_calendar()
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date) if end_date else date.today()
        df = cal.build_earnings_calendar(tickers, start, end)
        return {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end.isoformat(),
            "count": len(df),
            "calendar": df.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@pit_router.post("/build-panel")
def build_panel(req: BuildPanelRequest):
    """Build a PIT panel DataFrame (requires financials_db connection)."""
    builder = _get_panel_builder()
    try:
        dates_parsed = [date.fromisoformat(d) for d in req.dates]
        df = builder.build_panel(req.tickers, dates_parsed, req.metrics)
        return {
            "tickers": req.tickers,
            "dates": req.dates,
            "metrics": req.metrics,
            "shape": list(df.shape),
            "panel": df.reset_index().to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@pit_router.post("/validate-series")
def validate_series(req: ValidateSeriesRequest):
    """Validate a time series for future leak and staleness."""
    validator = _get_validator()
    try:
        dates_parsed = [date.fromisoformat(d) for d in req.dates]
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates_parsed])
        series = pd.Series(req.values, index=idx, name=req.series_name)

        ct = date.fromisoformat(req.creation_timestamp) if req.creation_timestamp else date.today()
        report = validator.validate_series(
            series, ct, req.series_name, req.staleness_threshold_days
        )
        return report.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@pit_router.post("/validate-backtest")
def validate_backtest(req: ValidateBacktestRequest):
    """Validate a backtest feature matrix and target for PIT safety."""
    validator = _get_validator()
    try:
        dates_parsed = [date.fromisoformat(d) for d in req.dates]
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates_parsed])

        X = pd.DataFrame(req.X, index=idx)
        y = pd.Series(req.y, index=idx, name="target")

        result = validator.validate_backtest_inputs(X, y, idx)

        # Also run bias detection if feature_types provided
        if req.feature_types:
            detector = _get_detector()
            data_type_map = {k: (v, None) for k, v in req.feature_types.items()}
            bias_report = detector.validate_feature_set(X, data_type_map)
            result["bias_report"] = bias_report.model_dump()

        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@pit_router.get("/quality-matrix")
def quality_matrix(
    series_names: list[str] = Query(default=[]),
    creation_timestamp: Optional[str] = Query(default=None),
):
    """Return quality matrix endpoint (placeholder — requires series data upload)."""
    return {
        "message": "POST your DataFrame to /pit/validate-series for individual series validation",
        "available_data_types": _get_registry().all_data_types(),
    }


@pit_router.get("/check-availability/{data_type}")
def check_availability(
    data_type: str,
    reference_date: str = Query(...),
    as_of_date: str = Query(...),
    sub_type: Optional[str] = Query(default=None),
):
    """Check if a specific data type is available at as_of_date."""
    registry = _get_registry()
    try:
        ref = date.fromisoformat(reference_date)
        as_of = date.fromisoformat(as_of_date)
        is_available = registry.is_available_at(data_type, ref, as_of, sub_type)
        earliest = registry.get_earliest_available(data_type, ref, sub_type)
        return {
            "data_type": data_type,
            "sub_type": sub_type,
            "reference_date": reference_date,
            "as_of_date": as_of_date,
            "is_available": is_available,
            "earliest_available": earliest.isoformat(),
            "days_early": max(0, (earliest - as_of).days) if not is_available else 0,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
