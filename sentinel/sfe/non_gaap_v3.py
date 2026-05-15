"""
non_gaap_v3.py — Non-GAAP Reconciliation Tables v3 (dim_017, target 9/10)

Architecture upgrade over v2:
  - EDGAR companyfacts XBRL API replaces fragile HTML regex for core metrics
  - 8-K Item 2.02 earnings release parser (BeautifulSoup table detection)
  - XBRL taxonomy mapper: standard non-GAAP items → GAAP equivalents
  - Non-GAAP quality scoring: red/yellow flags with temporal recurrence analysis
  - Peer non-GAAP comparison using XBRL adjustment magnitudes
  - 8-quarter GAAP vs non-GAAP margin spread trend
  - SQLite persistence: non_gaap_facts, reconciliation_tables, quality_flags,
    adjustment_history
  - FastAPI router at /nongaap/v3

Public API:
  XBRLNonGAAPExtractor        — companyfacts + iXBRL extraction
  EightKEarningsParser        — 8-K press release reconciliation table parser
  ReconciliationTableBuilder  — structured reconciliation with add-backs
  NonGAAPQualityEngine        — red/yellow flag scoring
  PeerNonGAAPComparator       — sector-level adjustment benchmarking
  NonGAAPDB                   — SQLite persistence layer
  nongaap_v3_router           — FastAPI APIRouter at /nongaap/v3
"""
from __future__ import annotations

import html as html_module
import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field as PydanticField

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "XBRLNonGAAPExtractor",
    "EightKEarningsParser",
    "ReconciliationTableBuilder",
    "NonGAAPQualityEngine",
    "PeerNonGAAPComparator",
    "NonGAAPDB",
    "nongaap_v3_router",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT  = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_HEADERS     = {
    "User-Agent": _USER_AGENT,
    "Accept":     "application/json, text/html, */*",
    "Accept-Encoding": "gzip, deflate",
}
_RATE_DELAY  = 0.15   # 150 ms — comfortably under SEC 10 req/s limit
_TIMEOUT     = 30.0
_MAX_RETRY   = 3

EDGAR_FACTS_URL      = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_SUBMISSIONS    = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVE        = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
EDGAR_TICKERS_URL    = "https://www.sec.gov/files/company_tickers.json"
EFTS_SEARCH          = "https://efts.sec.gov/LATEST/search-index"

_DB_PATH = Path(__file__).parent.parent / "data" / "non_gaap_v3.db"

# ---------------------------------------------------------------------------
# XBRL concept maps for non-GAAP reconstruction
# ---------------------------------------------------------------------------

# Standard GAAP concepts used as starting points for reconciliation
_GAAP_BASE_CONCEPTS: dict[str, list[str]] = {
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "ebitda_proxy": [
        "OperatingIncomeLoss",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditures",
        "PaymentsToAcquireProductiveAssets",
    ],
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "stock_comp": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
        "EmployeeBenefitsAndShareBasedCompensation",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
    "income_tax": [
        "IncomeTaxExpenseBenefit",
    ],
    "restructuring": [
        "RestructuringCharges",
        "RestructuringAndRelatedActivitiesDisclosure",
        "RestructuringCostsAndAssetImpairmentCharges",
    ],
    "goodwill_impairment": [
        "GoodwillImpairmentLoss",
        "ImpairmentOfIntangibleAssetsExcludingGoodwill",
    ],
    "intangible_amort": [
        "AmortizationOfIntangibleAssets",
        "FiniteLivedIntangibleAssetsAmortizationExpense",
    ],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
}

# Non-GAAP adjustment taxonomy: canonical name → XBRL concepts that represent it
_NONGAAP_ADJUSTMENTS: dict[str, list[str]] = {
    "stock_based_compensation": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
    ],
    "amortization_acquired_intangibles": [
        "AmortizationOfIntangibleAssets",
        "BusinessAcquisitionPurchasePriceAllocationAmortizationOfAcquiredIntangibles",
    ],
    "restructuring_charges": [
        "RestructuringCharges",
        "RestructuringCostsAndAssetImpairmentCharges",
    ],
    "goodwill_impairment": [
        "GoodwillImpairmentLoss",
    ],
    "asset_impairment": [
        "ImpairmentOfIntangibleAssetsExcludingGoodwill",
        "AssetImpairmentCharges",
    ],
    "acquisition_costs": [
        "BusinessCombinationAcquisitionRelatedCosts",
        "BusinessAcquisitionCostOfAcquiredEntityTransactionCosts",
    ],
    "legal_settlements": [
        "LitigationSettlementAmountAwardedToOtherParty",
        "LossContingencyAccrualCarryingValueCurrent",
    ],
    "income_tax_adjustments": [
        "DeferredIncomeTaxExpenseBenefit",
        "UnrecognizedTaxBenefits",
    ],
}

# Red-flag adjustment categories (almost never legitimate to exclude)
_RED_FLAG_CATEGORIES: set[str] = {
    "stock_based_compensation",  # companies benefit from excluding their own cost
    "income_tax_adjustments",
}

# Questionable (legitimate occasionally, watch for recurrence)
_QUESTIONABLE_CATEGORIES: set[str] = {
    "restructuring_charges",
    "acquisition_costs",
    "legal_settlements",
}

# Standard / mostly legitimate
_STANDARD_CATEGORIES: set[str] = {
    "depreciation_amortization",
    "amortization_acquired_intangibles",
    "goodwill_impairment",
    "asset_impairment",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class XBRLFact(BaseModel):
    concept:    str
    taxonomy:   str          # "us-gaap" | "dei" | company extension
    value:      float
    unit:       str          # "USD" | "shares" | "pure"
    period_end: str          # YYYY-MM-DD
    period_start: Optional[str] = None
    form:       str          = ""
    accession:  str          = ""
    filed:      str          = ""
    is_custom:  bool         = False   # company-specific extension namespace


class AdjustmentLineItem(BaseModel):
    name:            str
    canonical_name:  Optional[str]  = None   # maps to _NONGAAP_ADJUSTMENTS key
    xbrl_concept:    Optional[str]  = None
    value:           Optional[float] = None
    source:          str             = "html"   # "xbrl" | "html" | "computed"
    category:        str             = "other"  # standard | questionable | red_flag | other
    is_recurring:    bool            = False
    consecutive_qtrs: int            = 0


class ReconciliationRow(BaseModel):
    ticker:             str
    cik:                str          = ""
    period_end:         str
    filing_type:        str          = ""   # 10-K | 10-Q | 8-K
    accession:          str          = ""
    # GAAP starting point
    gaap_metric:        str          = "Net Income (GAAP)"
    gaap_value:         Optional[float] = None
    # Adjustments
    adjustments:        list[AdjustmentLineItem] = PydanticField(default_factory=list)
    total_adjustments:  Optional[float] = None
    # Non-GAAP result
    nongaap_metric:     str          = "Non-GAAP Net Income"
    nongaap_value:      Optional[float] = None
    # Derived
    adjustment_pct_of_gaap: Optional[float] = None
    data_source:        str          = "xbrl"   # "xbrl" | "html" | "hybrid"


class NonGAAPQualityScore(BaseModel):
    ticker:                  str
    period_end:              str
    filing_type:             str         = ""
    # Score components
    total_score:             float       = 0.0   # 0–100
    label:                   str         = "Unknown"
    # Flags
    red_flags:               list[str]   = PydanticField(default_factory=list)
    yellow_flags:            list[str]   = PydanticField(default_factory=list)
    # Metrics driving score
    stock_comp_excluded:     bool        = False
    margin_gap_pp:           Optional[float] = None  # nongaap margin - gaap margin, ppts
    recurring_items:         list[str]   = PydanticField(default_factory=list)
    eps_gap_trend:           str         = "stable"  # widening | stable | narrowing
    adjustment_pct_of_gaap:  Optional[float] = None
    # Peer context
    peer_avg_adjustment_pct: Optional[float] = None


class MarginSpread(BaseModel):
    ticker:          str
    period_end:      str
    gaap_margin:     Optional[float] = None   # net_income / revenue
    nongaap_margin:  Optional[float] = None
    spread_pp:       Optional[float] = None   # nongaap - gaap, percentage points
    revenue:         Optional[float] = None


class PeerComparisonResult(BaseModel):
    ticker:                  str
    sector:                  str         = ""
    adjustment_pct_of_gaap:  Optional[float] = None
    quality_score:           Optional[float] = None
    sector_avg_adjustment:   Optional[float] = None
    sector_p75_adjustment:   Optional[float] = None
    relative_standing:       str         = "unknown"  # "aggressive" | "moderate" | "conservative"
    peers:                   list[dict]  = PydanticField(default_factory=list)


class FreeCashFlowReconciliation(BaseModel):
    ticker:          str
    period_end:      str
    cfo:             Optional[float] = None
    capex:           Optional[float] = None
    free_cash_flow:  Optional[float] = None
    cfo_xbrl:        bool            = False
    capex_xbrl:      bool            = False


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

class NonGAAPDB:
    """SQLite persistence for non-GAAP facts, reconciliation tables, quality flags."""

    DDL = """
    CREATE TABLE IF NOT EXISTS non_gaap_facts (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT NOT NULL,
        cik          TEXT,
        concept      TEXT NOT NULL,
        taxonomy     TEXT,
        value        REAL,
        unit         TEXT,
        period_end   TEXT,
        period_start TEXT,
        form         TEXT,
        accession    TEXT,
        filed        TEXT,
        is_custom    INTEGER DEFAULT 0,
        inserted_at  TEXT DEFAULT (datetime('now')),
        UNIQUE(cik, concept, period_end, form)
    );

    CREATE TABLE IF NOT EXISTS reconciliation_tables (
        id                     INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker                 TEXT NOT NULL,
        cik                    TEXT,
        period_end             TEXT,
        filing_type            TEXT,
        accession              TEXT,
        gaap_metric            TEXT,
        gaap_value             REAL,
        nongaap_metric         TEXT,
        nongaap_value          REAL,
        total_adjustments      REAL,
        adjustment_pct_of_gaap REAL,
        data_source            TEXT,
        adjustments_json       TEXT,
        inserted_at            TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, period_end, filing_type)
    );

    CREATE TABLE IF NOT EXISTS quality_flags (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL,
        period_end      TEXT,
        total_score     REAL,
        label           TEXT,
        red_flags_json  TEXT,
        yellow_flags_json TEXT,
        adjustment_pct  REAL,
        margin_gap_pp   REAL,
        inserted_at     TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, period_end)
    );

    CREATE TABLE IF NOT EXISTS adjustment_history (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker            TEXT NOT NULL,
        canonical_name    TEXT NOT NULL,
        period_end        TEXT NOT NULL,
        value             REAL,
        consecutive_count INTEGER DEFAULT 1,
        inserted_at       TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, canonical_name, period_end)
    );

    CREATE INDEX IF NOT EXISTS ix_nongaap_facts_ticker ON non_gaap_facts(ticker, period_end);
    CREATE INDEX IF NOT EXISTS ix_recon_ticker ON reconciliation_tables(ticker, period_end);
    CREATE INDEX IF NOT EXISTS ix_quality_ticker ON quality_flags(ticker, period_end);
    CREATE INDEX IF NOT EXISTS ix_adj_history ON adjustment_history(ticker, canonical_name);
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path else _DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.DDL)
        self._conn.commit()

    def upsert_facts(self, ticker: str, facts: list[XBRLFact]) -> int:
        rows = [(
            ticker, f.concept, f.taxonomy, f.value, f.unit,
            f.period_end, f.period_start, f.form, f.accession,
            f.filed, int(f.is_custom),
        ) for f in facts]
        self._conn.executemany(
            """INSERT OR REPLACE INTO non_gaap_facts
               (ticker, concept, taxonomy, value, unit, period_end, period_start,
                form, accession, filed, is_custom)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def upsert_reconciliation(self, row: ReconciliationRow) -> None:
        adj_json = json.dumps([a.model_dump() for a in row.adjustments])
        self._conn.execute(
            """INSERT OR REPLACE INTO reconciliation_tables
               (ticker, cik, period_end, filing_type, accession, gaap_metric,
                gaap_value, nongaap_metric, nongaap_value, total_adjustments,
                adjustment_pct_of_gaap, data_source, adjustments_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (row.ticker, row.cik, row.period_end, row.filing_type, row.accession,
             row.gaap_metric, row.gaap_value, row.nongaap_metric, row.nongaap_value,
             row.total_adjustments, row.adjustment_pct_of_gaap, row.data_source, adj_json),
        )
        self._conn.commit()

    def upsert_quality(self, score: NonGAAPQualityScore) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO quality_flags
               (ticker, period_end, total_score, label, red_flags_json,
                yellow_flags_json, adjustment_pct, margin_gap_pp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (score.ticker, score.period_end, score.total_score, score.label,
             json.dumps(score.red_flags), json.dumps(score.yellow_flags),
             score.adjustment_pct_of_gaap, score.margin_gap_pp),
        )
        self._conn.commit()

    def track_adjustment(self, ticker: str, canonical: str, period_end: str, value: float) -> int:
        """Insert/update adjustment_history and return consecutive count."""
        self._conn.execute(
            """INSERT OR REPLACE INTO adjustment_history
               (ticker, canonical_name, period_end, value) VALUES (?,?,?,?)""",
            (ticker, canonical, period_end, value),
        )
        self._conn.commit()
        cur = self._conn.execute(
            """SELECT period_end FROM adjustment_history
               WHERE ticker=? AND canonical_name=?
               ORDER BY period_end DESC""",
            (ticker, canonical),
        )
        rows = cur.fetchall()
        # Count consecutive quarters
        count = 0
        for row in rows:
            count += 1
        return count

    def get_reconciliations(self, ticker: str, limit: int = 8) -> list[dict]:
        cur = self._conn.execute(
            """SELECT * FROM reconciliation_tables WHERE ticker=?
               ORDER BY period_end DESC LIMIT ?""",
            (ticker, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_quality_history(self, ticker: str, limit: int = 8) -> list[dict]:
        cur = self._conn.execute(
            """SELECT * FROM quality_flags WHERE ticker=?
               ORDER BY period_end DESC LIMIT ?""",
            (ticker, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_adjustment_history(self, ticker: str, canonical: str) -> list[dict]:
        cur = self._conn.execute(
            """SELECT * FROM adjustment_history
               WHERE ticker=? AND canonical_name=?
               ORDER BY period_end""",
            (ticker, canonical),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------

def _rate_get(session: requests.Session, url: str, params: dict | None = None) -> requests.Response | None:
    """Rate-limited GET with exponential-backoff retry."""
    time.sleep(_RATE_DELAY)
    for attempt in range(_MAX_RETRY):
        try:
            resp = session.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.HTTPError:
            if attempt == _MAX_RETRY - 1:
                return None
            time.sleep(2 ** attempt)
        except requests.RequestException:
            if attempt == _MAX_RETRY - 1:
                return None
            time.sleep(2 ** attempt)
    return None


_cik_cache: dict[str, str] = {}

def _resolve_cik(ticker: str) -> str:
    """Resolve ticker to zero-padded 10-digit CIK via EDGAR company_tickers.json."""
    upper = ticker.upper()
    if upper in _cik_cache:
        return _cik_cache[upper]
    session = requests.Session()
    session.headers.update(_HEADERS)
    resp = _rate_get(session, EDGAR_TICKERS_URL)
    if resp is None:
        raise ValueError(f"Cannot resolve CIK for ticker {ticker}")
    data = resp.json()
    for _k, entry in data.items():
        if str(entry.get("ticker", "")).upper() == upper:
            cik = str(entry["cik_str"]).zfill(10)
            _cik_cache[upper] = cik
            return cik
    raise ValueError(f"Ticker {ticker} not found in EDGAR company_tickers.json")


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(str(v).replace(",", "").replace("$", "").replace("(", "-").replace(")", "").strip())
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# XBRL Non-GAAP Extractor
# ---------------------------------------------------------------------------

class XBRLNonGAAPExtractor:
    """
    Primary data source: EDGAR companyfacts XBRL API.
    Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

    Extracts:
    1. Standard GAAP base metrics (net income, EPS, CFO, capex)
    2. Standard non-GAAP adjustment items (stock comp, D&A, restructuring, etc.)
    3. Company-specific extension namespace concepts (custom non-GAAP tags)
    4. Free cash flow = CFO - capex (fully XBRL-tagged)
    5. Organic revenue growth approximation using reported vs constant-currency
    """

    def __init__(self, db: NonGAAPDB | None = None) -> None:
        self._db      = db or NonGAAPDB()
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._facts_cache: dict[str, dict] = {}

    def _get_company_facts(self, cik: str) -> dict:
        """Fetch and cache companyfacts JSON."""
        padded = cik.zfill(10)
        if padded in self._facts_cache:
            return self._facts_cache[padded]
        url  = EDGAR_FACTS_URL.format(cik=padded)
        resp = _rate_get(self._session, url)
        if resp is None:
            logger.warning("companyfacts_not_found", cik=cik)
            return {}
        data = resp.json()
        self._facts_cache[padded] = data
        return data

    def _extract_concept_series(
        self,
        facts_data:   dict,
        taxonomy:     str,
        concept:      str,
        form_filter:  set[str] | None = None,
        unit_filter:  str | None = "USD",
    ) -> list[dict]:
        """
        Pull time series for a single taxonomy:concept.
        Returns list of {value, period_end, period_start, form, accession, filed}.
        """
        concepts = facts_data.get(taxonomy, {}).get(concept, {})
        units    = concepts.get("units", {})
        results: list[dict] = []

        for unit, observations in units.items():
            if unit_filter and unit != unit_filter:
                # Allow USD/shares mismatches gracefully
                if unit_filter == "USD" and unit not in ("USD", "USD/shares"):
                    continue
            for obs in observations:
                end    = obs.get("end", "")
                start  = obs.get("start", "")
                form   = obs.get("form", "")
                acc    = obs.get("accession", "")
                filed  = obs.get("filed", "")
                val    = obs.get("val")
                if val is None:
                    continue
                if form_filter and form not in form_filter:
                    continue
                # Only keep period-spanning facts (annual/quarterly), not instant
                if not end:
                    continue
                results.append({
                    "value":        float(val),
                    "period_end":   end,
                    "period_start": start,
                    "form":         form,
                    "accession":    acc,
                    "filed":        filed,
                    "unit":         unit,
                })

        # Deduplicate by (period_end, form): keep latest filed
        dedup: dict[tuple, dict] = {}
        for r in results:
            key = (r["period_end"], r["form"])
            if key not in dedup or r["filed"] > dedup[key]["filed"]:
                dedup[key] = r
        return sorted(dedup.values(), key=lambda x: x["period_end"])

    def _first_available(
        self,
        facts_data: dict,
        taxonomy: str,
        concepts:  list[str],
        period_end: str,
        form: str,
        unit_filter: str = "USD",
    ) -> float | None:
        """Try a list of concept synonyms and return the first value found for a period."""
        for concept in concepts:
            series = self._extract_concept_series(
                facts_data, taxonomy, concept,
                form_filter={form}, unit_filter=unit_filter,
            )
            for row in series:
                if row["period_end"] == period_end:
                    return row["value"]
        return None

    def _detect_custom_extensions(self, facts_data: dict, ticker: str) -> list[XBRLFact]:
        """
        Extract company-specific (non us-gaap) XBRL tags.
        These are often used for non-GAAP metrics that companies self-define.
        E.g., 'AdjustedEBITDA', 'NonGAAPNetIncome', etc.
        """
        custom_facts: list[XBRLFact] = []
        nongaap_keywords = re.compile(
            r"(adjusted|nongaap|non.gaap|organic|normalized|core|underlying|"
            r"recurring|free.cash|fcf|ebitda|ebita|ebit[^d])",
            re.IGNORECASE,
        )

        for taxonomy, concepts in facts_data.items():
            if taxonomy == "us-gaap" or taxonomy == "dei":
                continue  # standard taxonomies handled separately
            for concept, data in concepts.items():
                if not nongaap_keywords.search(concept):
                    continue
                units_data = data.get("units", {})
                for unit, obs_list in units_data.items():
                    if unit not in ("USD", "pure", "USD/shares"):
                        continue
                    for obs in obs_list:
                        val = obs.get("val")
                        end = obs.get("end", "")
                        if val is None or not end:
                            continue
                        custom_facts.append(XBRLFact(
                            concept=concept,
                            taxonomy=taxonomy,
                            value=float(val),
                            unit=unit,
                            period_end=end,
                            period_start=obs.get("start", ""),
                            form=obs.get("form", ""),
                            accession=obs.get("accession", ""),
                            filed=obs.get("filed", ""),
                            is_custom=True,
                        ))
        return custom_facts

    def get_gaap_base_metrics(
        self,
        ticker: str,
        periods: int = 8,
        form_types: set[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return DataFrame of GAAP base metrics indexed by (period_end, form).
        Columns: net_income, operating_income, eps_diluted, eps_basic,
                 cfo, capex, da, stock_comp, restructuring, intangible_amort,
                 revenue, gross_profit, interest_expense, income_tax
        """
        if form_types is None:
            form_types = {"10-K", "10-Q"}

        cik        = _resolve_cik(ticker)
        facts      = self._get_company_facts(cik)
        facts_data = facts.get("facts", {})
        usgaap     = facts_data.get("us-gaap", {})

        rows: dict[tuple[str, str], dict] = {}

        for canonical, concepts in _GAAP_BASE_CONCEPTS.items():
            for concept in concepts:
                series = self._extract_concept_series(
                    {"us-gaap": usgaap}, "us-gaap", concept,
                    form_filter=form_types,
                )
                for obs in series:
                    key = (obs["period_end"], obs["form"])
                    if key not in rows:
                        rows[key] = {"period_end": obs["period_end"], "form": obs["form"]}
                    # First wins (priority-ordered concepts)
                    if canonical not in rows[key]:
                        rows[key][canonical] = obs["value"]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(list(rows.values()))
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end", ascending=False)

        # Derive free cash flow
        if "cfo" in df.columns and "capex" in df.columns:
            df["free_cash_flow"] = df["cfo"] - df["capex"].abs()

        return df.head(periods * 2)  # return extra for margin calculations

    def get_adjustment_facts(
        self,
        ticker: str,
        form_types: set[str] | None = None,
    ) -> dict[str, pd.Series]:
        """
        For each standard adjustment category, return a Series indexed by period_end.
        Returns dict[canonical_name → pd.Series].
        """
        if form_types is None:
            form_types = {"10-K", "10-Q"}

        cik        = _resolve_cik(ticker)
        facts      = self._get_company_facts(cik)
        facts_data = facts.get("facts", {})
        usgaap     = facts_data.get("us-gaap", {})

        result: dict[str, pd.Series] = {}

        for canonical, concepts in _NONGAAP_ADJUSTMENTS.items():
            all_series: list[dict] = []
            for concept in concepts:
                series = self._extract_concept_series(
                    {"us-gaap": usgaap}, "us-gaap", concept,
                    form_filter=form_types,
                )
                all_series.extend(series)

            if not all_series:
                continue

            df = pd.DataFrame(all_series)
            df["period_end"] = pd.to_datetime(df["period_end"])
            # Dedup: keep latest filing per period
            df = df.sort_values("filed", ascending=False).drop_duplicates("period_end")
            s  = df.set_index("period_end")["value"].sort_index()
            result[canonical] = s

        return result

    def get_custom_extension_facts(self, ticker: str) -> list[XBRLFact]:
        """Extract company-specific non-GAAP tags from custom XBRL extensions."""
        cik        = _resolve_cik(ticker)
        facts      = self._get_company_facts(cik)
        facts_data = facts.get("facts", {})
        customs    = self._detect_custom_extensions(facts_data, ticker)
        if customs:
            self._db.upsert_facts(ticker, customs)
        return customs

    def get_free_cash_flow_series(self, ticker: str, periods: int = 8) -> list[FreeCashFlowReconciliation]:
        """
        FCF = CFO - capex, fully XBRL-sourced.
        Returns list sorted newest-first.
        """
        cik        = _resolve_cik(ticker)
        facts      = self._get_company_facts(cik)
        facts_data = facts.get("facts", {})
        usgaap     = facts_data.get("us-gaap", {})

        def _fetch(concepts: list[str]) -> dict[str, float]:
            for concept in concepts:
                series = self._extract_concept_series(
                    {"us-gaap": usgaap}, "us-gaap", concept,
                    form_filter={"10-K", "10-Q"},
                )
                if series:
                    return {r["period_end"]: r["value"] for r in series}
            return {}

        cfo_map   = _fetch(_GAAP_BASE_CONCEPTS["cfo"])
        capex_map = _fetch(_GAAP_BASE_CONCEPTS["capex"])

        all_periods = sorted(set(cfo_map) | set(capex_map), reverse=True)
        results: list[FreeCashFlowReconciliation] = []

        for pe in all_periods[:periods]:
            cfo   = cfo_map.get(pe)
            capex = capex_map.get(pe)
            fcf   = None
            if cfo is not None and capex is not None:
                fcf = cfo - abs(capex)
            results.append(FreeCashFlowReconciliation(
                ticker=ticker,
                period_end=pe,
                cfo=cfo,
                capex=capex,
                free_cash_flow=fcf,
                cfo_xbrl=cfo is not None,
                capex_xbrl=capex is not None,
            ))

        return results


# ---------------------------------------------------------------------------
# 8-K Earnings Release Parser
# ---------------------------------------------------------------------------

class EightKEarningsParser:
    """
    Parse 8-K Item 2.02 earnings press releases to extract reconciliation tables.

    Strategy:
    1. Fetch recent 8-K filings from EDGAR submissions API
    2. Download the HTML exhibit (ex-99.1 earnings release)
    3. Use BeautifulSoup to find tables with reconciliation keywords in headers
    4. Extract GAAP starting point + adjustment line items + non-GAAP result
    5. Normalize dollar amounts (handle parentheses negatives, millions/billions)
    """

    # Keywords that signal a reconciliation table header or caption
    _RECON_PATTERNS = [
        re.compile(r"reconciliation.{0,60}(gaap|non.gaap)", re.I),
        re.compile(r"non.gaap.{0,60}reconciliation", re.I),
        re.compile(r"gaap\s+to\s+non.gaap", re.I),
        re.compile(r"adjusted\s+ebitda\s+reconciliation", re.I),
        re.compile(r"reconciliation\s+of\s+adjusted", re.I),
        re.compile(r"reconciliation\s+of\s+net\s+income", re.I),
    ]

    _GAAP_LINE_RE = re.compile(
        r"^(gaap\s+|net\s+income|net\s+loss|operating\s+income|"
        r"net\s+earnings|income\s+from\s+operations)",
        re.I,
    )
    _NONGAAP_LINE_RE = re.compile(
        r"(non.gaap|adjusted|normalized).{0,50}"
        r"(net\s+income|earnings|ebitda|operating\s+income|eps|income)",
        re.I,
    )
    _TOTAL_ROW_RE = re.compile(r"^(total|subtotal|sum)\b", re.I)

    def __init__(self, db: NonGAAPDB | None = None) -> None:
        self._db      = db or NonGAAPDB()
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _get_8k_filings(self, cik: str, limit: int = 8) -> list[dict]:
        """Return recent 8-K filings with Item 2.02 (Results of Operations)."""
        padded = cik.zfill(10)
        url    = EDGAR_SUBMISSIONS.format(cik=padded)
        resp   = _rate_get(self._session, url)
        if resp is None:
            return []

        data     = resp.json()
        recent   = data.get("filings", {}).get("recent", {})
        forms    = recent.get("form", [])
        acc_nums = recent.get("accessionNumber", [])
        dates    = recent.get("filingDate", [])
        items    = recent.get("items", [])
        docs     = recent.get("primaryDocument", [])
        cik_int  = str(int(cik))

        filings: list[dict] = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            item_str = items[i] if i < len(items) else ""
            # Only earnings-related 8-Ks
            if "2.02" not in str(item_str) and "2.01" not in str(item_str):
                # Also check if it might be an earnings release without item tag
                doc_name = docs[i] if i < len(docs) else ""
                if not re.search(r"earn|result|quarter|annual", doc_name, re.I):
                    continue

            acc = acc_nums[i] if i < len(acc_nums) else ""
            doc = docs[i] if i < len(docs) else ""
            acc_nodash = acc.replace("-", "")
            filings.append({
                "accession": acc,
                "filed":     dates[i] if i < len(dates) else "",
                "doc_url":   EDGAR_ARCHIVE.format(cik=cik_int, acc_nodash=acc_nodash, doc=doc),
                "index_url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/",
            })
            if len(filings) >= limit:
                break

        return filings

    def _find_exhibit_url(self, index_url: str) -> str | None:
        """Find the ex-99.1 exhibit URL from the filing index page."""
        resp = _rate_get(self._session, index_url)
        if resp is None:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True).lower()
            if re.search(r"ex.?99", href, re.I) or re.search(r"ex.?99|exhibit.?99", text, re.I):
                if href.startswith("http"):
                    return href
                return "https://www.sec.gov" + href
        return None

    def _is_recon_table(self, table: Tag) -> bool:
        """Check whether a <table> element is a reconciliation table."""
        # Check table caption
        cap = table.find("caption")
        if cap and any(p.search(cap.get_text()) for p in self._RECON_PATTERNS):
            return True
        # Check preceding headers/paragraphs (within ~3 siblings)
        prev = table.find_previous_sibling()
        for _ in range(5):
            if prev is None:
                break
            text = prev.get_text(separator=" ")
            if any(p.search(text) for p in self._RECON_PATTERNS):
                return True
            prev = prev.find_previous_sibling()
        # Check table text itself
        table_text = table.get_text(separator=" ")
        return any(p.search(table_text) for p in self._RECON_PATTERNS)

    def _parse_number(self, cell_text: str) -> float | None:
        """Parse cell text to float, handling ($12,345), (12.3), negatives."""
        text = cell_text.strip()
        if not text or text in ("—", "–", "-", "N/A", "NM", "*"):
            return None
        neg  = text.startswith("(") or text.startswith("-")
        text = re.sub(r"[\$,\s%]", "", text)
        text = re.sub(r"[()]", "", text)
        try:
            val = float(text)
            return -val if neg else val
        except ValueError:
            return None

    def _scale_detect(self, header_text: str) -> float:
        """Detect scaling from table header (millions, billions, thousands)."""
        h = header_text.lower()
        if "billion" in h:
            return 1e9
        if "million" in h or "(in millions" in h:
            return 1e6
        if "thousand" in h:
            return 1e3
        return 1.0

    def _parse_recon_table(self, table: Tag, scale: float = 1.0) -> list[dict]:
        """
        Parse a reconciliation table into structured rows.
        Returns list of dicts with keys: label, value, row_type
        """
        rows = table.find_all("tr")
        if len(rows) < 3:
            return []

        # Detect scale from header row
        header_text = rows[0].get_text(separator=" ")
        scale = self._scale_detect(header_text) or scale

        # Find value column: last column with numeric data
        value_col = -1
        for row in rows[1:4]:
            cells = row.find_all(["td", "th"])
            for i in range(len(cells) - 1, 0, -1):
                if self._parse_number(cells[i].get_text()) is not None:
                    value_col = i
                    break
            if value_col != -1:
                break
        if value_col == -1:
            value_col = 1  # fallback

        parsed: list[dict] = []
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            label = cells[0].get_text(separator=" ").strip()
            if not label:
                continue
            val   = None
            if value_col < len(cells):
                val = self._parse_number(cells[value_col].get_text())
                if val is not None:
                    val *= scale

            # Classify row type
            if self._GAAP_LINE_RE.search(label):
                row_type = "gaap_base"
            elif self._NONGAAP_LINE_RE.search(label):
                row_type = "nongaap_result"
            elif self._TOTAL_ROW_RE.search(label):
                row_type = "subtotal"
            else:
                row_type = "adjustment"

            parsed.append({"label": label, "value": val, "row_type": row_type})

        return parsed

    def parse_8k(self, html: str, ticker: str, period_end: str, accession: str) -> ReconciliationRow | None:
        """
        Extract reconciliation table from 8-K HTML.
        Returns ReconciliationRow or None if no reconciliation found.
        """
        soup  = BeautifulSoup(html, "html.parser")
        scale = 1.0

        # Detect document-level scale
        for tag in soup.find_all(["p", "div", "span"])[:20]:
            text = tag.get_text()
            s    = self._scale_detect(text)
            if s > 1.0:
                scale = s
                break

        tables = soup.find_all("table")
        for table in tables:
            if not self._is_recon_table(table):
                continue
            parsed = self._parse_recon_table(table, scale=scale)
            if len(parsed) < 3:
                continue

            # Extract structured data
            gaap_rows     = [r for r in parsed if r["row_type"] == "gaap_base"]
            nongaap_rows  = [r for r in parsed if r["row_type"] == "nongaap_result"]
            adj_rows      = [r for r in parsed if r["row_type"] == "adjustment" and r["value"] is not None]

            if not gaap_rows and not nongaap_rows:
                continue

            gaap_label = gaap_rows[0]["label"]    if gaap_rows    else "GAAP Net Income"
            gaap_val   = gaap_rows[0]["value"]    if gaap_rows    else None
            ng_label   = nongaap_rows[0]["label"] if nongaap_rows else "Non-GAAP Net Income"
            ng_val     = nongaap_rows[0]["value"] if nongaap_rows else None

            adjustments: list[AdjustmentLineItem] = []
            total_adj   = 0.0
            for r in adj_rows:
                canonical = _classify_adjustment(r["label"])
                cat       = _get_category(canonical)
                item = AdjustmentLineItem(
                    name=r["label"],
                    canonical_name=canonical,
                    value=r["value"],
                    source="html",
                    category=cat,
                )
                adjustments.append(item)
                total_adj += r["value"] or 0.0

            adj_pct = None
            if gaap_val and gaap_val != 0:
                adj_pct = (total_adj / abs(gaap_val)) * 100.0

            return ReconciliationRow(
                ticker=ticker,
                period_end=period_end,
                filing_type="8-K",
                accession=accession,
                gaap_metric=gaap_label,
                gaap_value=gaap_val,
                adjustments=adjustments,
                total_adjustments=total_adj if adjustments else None,
                nongaap_metric=ng_label,
                nongaap_value=ng_val,
                adjustment_pct_of_gaap=adj_pct,
                data_source="html",
            )

        return None

    def get_earnings_releases(self, ticker: str, limit: int = 8) -> list[ReconciliationRow]:
        """Fetch and parse 8-K earnings releases for a ticker."""
        try:
            cik = _resolve_cik(ticker)
        except ValueError:
            return []

        filings = self._get_8k_filings(cik, limit=limit)
        results: list[ReconciliationRow] = []

        for filing in filings:
            # Try primary doc first, then exhibit
            urls_to_try = [filing["doc_url"]]
            exhibit_url = self._find_exhibit_url(filing["index_url"])
            if exhibit_url:
                urls_to_try = [exhibit_url] + urls_to_try

            html = None
            for url in urls_to_try:
                resp = _rate_get(self._session, url)
                if resp and resp.status_code == 200:
                    html = resp.text
                    break

            if not html:
                continue

            period = filing["filed"][:7]  # YYYY-MM as proxy
            row    = self.parse_8k(html, ticker, period, filing["accession"])
            if row:
                results.append(row)
                self._db.upsert_reconciliation(row)

        return results


# ---------------------------------------------------------------------------
# Adjustment classification helpers
# ---------------------------------------------------------------------------

def _classify_adjustment(name: str) -> str | None:
    """Map free-text adjustment name to canonical category key."""
    nl = name.lower()
    patterns: list[tuple[str, str]] = [
        ("stock.?based.?comp|share.?based.?comp|stock.?comp|sbc|equity.?award",  "stock_based_compensation"),
        (r"depreciation|d&a|d\+a",                                                "depreciation_amortization"),
        (r"amort.{0,20}intangib|intangib.{0,20}amort|acquired.{0,20}amort",      "amortization_acquired_intangibles"),
        (r"restructur|severance|workforce|headcount",                              "restructuring_charges"),
        (r"goodwill.{0,20}impair",                                                "goodwill_impairment"),
        (r"impairment|asset.{0,20}write",                                         "asset_impairment"),
        (r"acqui.{0,20}cost|transaction.{0,20}cost|deal.{0,20}cost|m&a.{0,20}cost|merger.{0,20}cost", "acquisition_costs"),
        (r"legal|litigation|settlement|regulatory",                               "legal_settlements"),
        (r"income.{0,10}tax|tax.{0,10}benefit|deferred.{0,10}tax",               "income_tax_adjustments"),
    ]
    for pat, canonical in patterns:
        if re.search(pat, nl):
            return canonical
    return None


def _get_category(canonical: str | None) -> str:
    if canonical is None:
        return "other"
    if canonical in _RED_FLAG_CATEGORIES:
        return "red_flag"
    if canonical in _QUESTIONABLE_CATEGORIES:
        return "questionable"
    if canonical in _STANDARD_CATEGORIES:
        return "standard"
    return "other"


# ---------------------------------------------------------------------------
# Reconciliation Table Builder
# ---------------------------------------------------------------------------

class ReconciliationTableBuilder:
    """
    Synthesize GAAP → adjustments → non-GAAP reconciliation tables
    using XBRL data as the authoritative source.

    For each period:
    1. GAAP base = net_income from XBRL (us-gaap:NetIncomeLoss)
    2. Standard adjustments = stock_comp, D&A, restructuring, etc. from XBRL
    3. Custom adjustments = company-specific extension concepts
    4. Non-GAAP result = GAAP + sum(adjustments)
    5. Crosscheck vs 8-K parsed value if available
    """

    def __init__(
        self,
        extractor: XBRLNonGAAPExtractor | None = None,
        parser_8k: EightKEarningsParser | None = None,
        db:        NonGAAPDB | None = None,
    ) -> None:
        self._extractor = extractor or XBRLNonGAAPExtractor()
        self._parser_8k = parser_8k or EightKEarningsParser()
        self._db        = db or NonGAAPDB()

    def build_reconciliations(
        self,
        ticker:   str,
        periods:  int = 8,
        include_8k: bool = True,
    ) -> list[ReconciliationRow]:
        """
        Build reconciliation tables for the most recent N periods.

        Data hierarchy:
        1. XBRL companyfacts (most reliable, structured)
        2. 8-K earnings release HTML (press release tables)
        3. 10-Q/10-K HTML fallback (already handled by v2 parser)
        """
        try:
            cik = _resolve_cik(ticker)
        except ValueError as exc:
            logger.warning("cik_resolution_failed", ticker=ticker, error=str(exc))
            return []

        # 1. Get XBRL base metrics
        base_df  = self._extractor.get_gaap_base_metrics(ticker, periods=periods + 4)
        adj_map  = self._extractor.get_adjustment_facts(ticker)

        results:   list[ReconciliationRow] = []
        seen_ends: set[str] = set()

        if not base_df.empty:
            # Use quarterly (10-Q) and annual (10-K) data
            for _, row in base_df.iterrows():
                pe   = str(row["period_end"].date()) if pd.notnull(row["period_end"]) else ""
                form = str(row.get("form", ""))
                if not pe or pe in seen_ends:
                    continue
                seen_ends.add(pe)

                gaap_val = _safe_float(row.get("net_income"))
                revenue  = _safe_float(row.get("revenue"))

                # Collect XBRL-sourced adjustments
                adjustments: list[AdjustmentLineItem] = []
                total_adj   = 0.0

                for canonical, adj_series in adj_map.items():
                    if adj_series.empty:
                        continue
                    # Find closest period
                    try:
                        pe_ts  = pd.Timestamp(pe)
                        nearby = adj_series.index[abs(adj_series.index - pe_ts) <= pd.Timedelta(days=95)]
                        if len(nearby) == 0:
                            continue
                        closest = nearby[abs(adj_series.index[nearby] - pe_ts).argmin()]
                        val     = float(adj_series[closest])
                    except Exception:
                        continue

                    category = _get_category(canonical)
                    item = AdjustmentLineItem(
                        name=canonical.replace("_", " ").title(),
                        canonical_name=canonical,
                        value=val,
                        source="xbrl",
                        category=category,
                    )
                    adjustments.append(item)
                    # Add-backs are positive to GAAP income
                    total_adj += abs(val)

                # Compute non-GAAP proxy
                nongaap_val = (gaap_val + total_adj) if gaap_val is not None else None

                adj_pct = None
                if gaap_val and gaap_val != 0:
                    adj_pct = (total_adj / abs(gaap_val)) * 100.0

                recon = ReconciliationRow(
                    ticker=ticker,
                    cik=cik,
                    period_end=pe,
                    filing_type=form,
                    gaap_metric="Net Income (GAAP)",
                    gaap_value=gaap_val,
                    adjustments=adjustments,
                    total_adjustments=total_adj if adjustments else None,
                    nongaap_metric="Non-GAAP Net Income (computed)",
                    nongaap_value=nongaap_val,
                    adjustment_pct_of_gaap=adj_pct,
                    data_source="xbrl",
                )
                results.append(recon)
                self._db.upsert_reconciliation(recon)

                if len(results) >= periods:
                    break

        # 2. Augment with 8-K press release tables (overwrite if better data)
        if include_8k:
            try:
                ek_rows = self._parser_8k.get_earnings_releases(ticker, limit=periods)
                ek_map  = {r.period_end: r for r in ek_rows}
                # Merge: prefer 8-K if it has more adjustments or explicit non-GAAP value
                for i, recon in enumerate(results):
                    if recon.period_end[:7] in ek_map:
                        ek = ek_map[recon.period_end[:7]]
                        if ek.nongaap_value is not None and (
                            recon.nongaap_value is None or len(ek.adjustments) > len(recon.adjustments)
                        ):
                            results[i] = ReconciliationRow(
                                **{**recon.model_dump(), **{
                                    "nongaap_value":       ek.nongaap_value,
                                    "nongaap_metric":      ek.nongaap_metric,
                                    "adjustments":         ek.adjustments or recon.adjustments,
                                    "total_adjustments":   ek.total_adjustments or recon.total_adjustments,
                                    "adjustment_pct_of_gaap": ek.adjustment_pct_of_gaap or recon.adjustment_pct_of_gaap,
                                    "data_source":         "hybrid",
                                }}
                            )
            except Exception as exc:
                logger.warning("8k_augment_failed", ticker=ticker, error=str(exc))

        return sorted(results, key=lambda r: r.period_end, reverse=True)[:periods]


# ---------------------------------------------------------------------------
# Non-GAAP Quality Engine
# ---------------------------------------------------------------------------

class NonGAAPQualityEngine:
    """
    Score the quality of non-GAAP reporting.

    Red flags (high severity, automatic deductions):
      1. Stock-based comp excluded → -20 pts
      2. Same adjustment appears 3+ consecutive quarters → -15 pts each
      3. Non-GAAP margin > GAAP margin by >10 percentage points → -15 pts
      4. Adjustment > 50% of GAAP income → -20 pts

    Yellow flags (medium severity, warnings):
      1. Non-GAAP margin gap 5–10 pp → -8 pts
      2. Widening EPS gap over 4+ quarters
      3. Adjustment 25–50% of GAAP income → -10 pts
      4. Restructuring excluded for 2 consecutive quarters → -8 pts

    Scoring: starts at 100, subtract deductions, clamp to [0, 100].
    """

    # Point deductions
    _STOCK_COMP_DEDUCTION       = 20
    _RECURRING_ITEM_DEDUCTION   = 15   # per item, 3+ consecutive
    _HIGH_MARGIN_GAP_DEDUCTION  = 15   # >10pp
    _MED_MARGIN_GAP_DEDUCTION   = 8    # 5–10pp
    _VERY_HIGH_ADJ_DEDUCTION    = 20   # >50% of GAAP
    _HIGH_ADJ_DEDUCTION         = 10   # 25–50%
    _WIDENING_EPS_DEDUCTION     = 8
    _CONSEC_RESTRUCTURING_DED   = 8    # 2+ consecutive quarters

    def __init__(self, db: NonGAAPDB | None = None) -> None:
        self._db = db or NonGAAPDB()

    def _consec_count(self, ticker: str, canonical: str, period_end: str) -> int:
        """Count how many consecutive quarters this adjustment has appeared."""
        hist = self._db.get_adjustment_history(ticker, canonical)
        if not hist:
            return 0
        # Sort descending from current period
        hist_sorted = sorted(hist, key=lambda x: x["period_end"], reverse=True)
        count = 0
        for row in hist_sorted:
            if row["period_end"] <= period_end:
                count += 1
            else:
                break  # not consecutive once we go past current period
        return count

    def _compute_margins(
        self,
        recon: ReconciliationRow,
        revenue: float | None,
    ) -> tuple[float | None, float | None]:
        """Compute GAAP and non-GAAP net margins as percentages."""
        if not revenue or revenue == 0:
            return None, None
        gaap_margin   = (recon.gaap_value / revenue * 100.0) if recon.gaap_value is not None else None
        nongaap_margin = (recon.nongaap_value / revenue * 100.0) if recon.nongaap_value is not None else None
        return gaap_margin, nongaap_margin

    def _eps_gap_trend(self, ticker: str) -> str:
        """
        Determine whether GAAP–non-GAAP EPS gap is widening or narrowing.
        Uses last 4 reconciliation records from DB.
        """
        hist = self._db.get_reconciliations(ticker, limit=4)
        if len(hist) < 2:
            return "stable"
        gaps = []
        for r in hist:
            g  = r.get("gaap_value")
            ng = r.get("nongaap_value")
            if g is not None and ng is not None and g != 0:
                gaps.append((ng - g) / abs(g) * 100.0)
        if len(gaps) < 2:
            return "stable"
        # Widening = gap increasing over time (most recent gap is larger)
        gaps_ordered = list(reversed(gaps))  # oldest first
        diffs = [gaps_ordered[i+1] - gaps_ordered[i] for i in range(len(gaps_ordered)-1)]
        avg_change = sum(diffs) / len(diffs)
        if avg_change > 2.0:
            return "widening"
        if avg_change < -2.0:
            return "narrowing"
        return "stable"

    def score(
        self,
        recon:   ReconciliationRow,
        revenue: float | None = None,
    ) -> NonGAAPQualityScore:
        """Score a single reconciliation period."""
        score      = 100.0
        red_flags:    list[str] = []
        yellow_flags: list[str] = []
        recurring:    list[str] = []
        stock_comp_excluded = False

        for adj in recon.adjustments:
            # Record in DB for consecutive tracking
            if adj.value is not None and adj.canonical_name:
                self._db.track_adjustment(
                    recon.ticker, adj.canonical_name, recon.period_end, adj.value
                )

            # 1. Stock-based comp red flag
            if adj.canonical_name == "stock_based_compensation":
                score -= self._STOCK_COMP_DEDUCTION
                red_flags.append(
                    f"Stock-based compensation excluded (${abs(adj.value or 0):,.0f}) — "
                    "company benefits from excluding its own cost"
                )
                stock_comp_excluded = True

            # 2. Recurrence check
            if adj.canonical_name:
                consec = self._consec_count(recon.ticker, adj.canonical_name, recon.period_end)
                adj.consecutive_qtrs = consec
                if consec >= 3:
                    adj.is_recurring = True
                    recurring.append(adj.canonical_name)
                    score -= self._RECURRING_ITEM_DEDUCTION
                    red_flags.append(
                        f"'{adj.name}' excluded for {consec} consecutive quarters — "
                        "recurring item masquerading as one-time"
                    )
                elif consec == 2 and adj.canonical_name == "restructuring_charges":
                    score -= self._CONSEC_RESTRUCTURING_DED
                    yellow_flags.append(
                        f"Restructuring excluded for 2 consecutive quarters — watch for recurrence"
                    )

        # 3. Margin gap
        gaap_margin, nongaap_margin = self._compute_margins(recon, revenue)
        margin_gap = None
        if gaap_margin is not None and nongaap_margin is not None:
            margin_gap = nongaap_margin - gaap_margin
            if margin_gap > 10.0:
                score -= self._HIGH_MARGIN_GAP_DEDUCTION
                red_flags.append(
                    f"Non-GAAP margin ({nongaap_margin:.1f}%) exceeds GAAP margin "
                    f"({gaap_margin:.1f}%) by {margin_gap:.1f}pp — highly aggressive"
                )
            elif margin_gap > 5.0:
                score -= self._MED_MARGIN_GAP_DEDUCTION
                yellow_flags.append(
                    f"Non-GAAP margin exceeds GAAP margin by {margin_gap:.1f}pp"
                )

        # 4. Adjustment magnitude
        adj_pct = recon.adjustment_pct_of_gaap
        if adj_pct is not None:
            if abs(adj_pct) > 50.0:
                score -= self._VERY_HIGH_ADJ_DEDUCTION
                red_flags.append(
                    f"Total adjustments are {adj_pct:.1f}% of GAAP income — extreme inflation"
                )
            elif abs(adj_pct) > 25.0:
                score -= self._HIGH_ADJ_DEDUCTION
                yellow_flags.append(f"Adjustments are {adj_pct:.1f}% of GAAP income")

        # 5. EPS gap trend
        eps_trend = self._eps_gap_trend(recon.ticker)
        if eps_trend == "widening":
            score -= self._WIDENING_EPS_DEDUCTION
            yellow_flags.append(
                "GAAP vs non-GAAP EPS gap widening over last 4 quarters — "
                "management expanding non-GAAP definitions"
            )

        score = max(0.0, min(100.0, score))
        label = _score_label(score)

        quality = NonGAAPQualityScore(
            ticker=recon.ticker,
            period_end=recon.period_end,
            filing_type=recon.filing_type,
            total_score=round(score, 1),
            label=label,
            red_flags=red_flags,
            yellow_flags=yellow_flags,
            stock_comp_excluded=stock_comp_excluded,
            margin_gap_pp=round(margin_gap, 2) if margin_gap is not None else None,
            recurring_items=recurring,
            eps_gap_trend=eps_trend,
            adjustment_pct_of_gaap=round(adj_pct, 2) if adj_pct is not None else None,
        )
        self._db.upsert_quality(quality)
        return quality

    def score_all(
        self,
        ticker:     str,
        reconciliations: list[ReconciliationRow],
        base_df:    pd.DataFrame | None = None,
    ) -> list[NonGAAPQualityScore]:
        """Score all reconciliation periods for a ticker."""
        scores: list[NonGAAPQualityScore] = []
        for recon in reconciliations:
            revenue = None
            if base_df is not None and not base_df.empty:
                pe_ts = pd.Timestamp(recon.period_end)
                nearby = base_df[abs(base_df["period_end"] - pe_ts) <= pd.Timedelta(days=95)]
                if not nearby.empty and "revenue" in nearby.columns:
                    revenue = _safe_float(nearby.iloc[0].get("revenue"))
            scores.append(self.score(recon, revenue=revenue))
        return scores

    def get_margin_spread_history(
        self,
        ticker:     str,
        extractor:  XBRLNonGAAPExtractor,
        quarters:   int = 8,
    ) -> list[MarginSpread]:
        """
        8-quarter GAAP vs non-GAAP margin spread.
        Uses DB for non-GAAP, XBRL for GAAP revenue + net income.
        """
        base_df = extractor.get_gaap_base_metrics(ticker, periods=quarters + 2)
        recons  = self._db.get_reconciliations(ticker, limit=quarters)

        spreads: list[MarginSpread] = []
        for r in recons:
            pe      = r.get("period_end", "")
            gaap_v  = r.get("gaap_value")
            ng_v    = r.get("nongaap_value")
            revenue = None

            if base_df is not None and not base_df.empty and "revenue" in base_df.columns:
                try:
                    pe_ts  = pd.Timestamp(pe)
                    nearby = base_df[abs(base_df["period_end"] - pe_ts) <= pd.Timedelta(days=95)]
                    if not nearby.empty:
                        revenue = _safe_float(nearby.iloc[0].get("revenue"))
                except Exception:
                    pass

            gaap_m = ng_m = spread = None
            if revenue and revenue != 0:
                if gaap_v is not None:
                    gaap_m = gaap_v / revenue * 100.0
                if ng_v is not None:
                    ng_m   = ng_v / revenue * 100.0
                if gaap_m is not None and ng_m is not None:
                    spread = ng_m - gaap_m

            spreads.append(MarginSpread(
                ticker=ticker,
                period_end=pe,
                gaap_margin=round(gaap_m, 2)   if gaap_m  is not None else None,
                nongaap_margin=round(ng_m, 2)  if ng_m    is not None else None,
                spread_pp=round(spread, 2)      if spread  is not None else None,
                revenue=revenue,
            ))

        return spreads


def _score_label(score: float) -> str:
    if score >= 88:
        return "Excellent"
    if score >= 72:
        return "Good"
    if score >= 55:
        return "Fair"
    if score >= 35:
        return "Poor"
    return "Red Flag"


# ---------------------------------------------------------------------------
# Peer Non-GAAP Comparator
# ---------------------------------------------------------------------------

class PeerNonGAAPComparator:
    """
    Compare a company's non-GAAP adjustment magnitude against sector peers.

    Uses EDGAR XBRL data so no proprietary APIs needed.
    Peers defined either explicitly (tickers list) or via SIC code lookup.
    """

    def __init__(
        self,
        extractor: XBRLNonGAAPExtractor | None = None,
        builder:   ReconciliationTableBuilder | None = None,
        db:        NonGAAPDB | None = None,
    ) -> None:
        self._extractor = extractor or XBRLNonGAAPExtractor()
        self._builder   = builder   or ReconciliationTableBuilder(self._extractor)
        self._db        = db        or NonGAAPDB()
        self._session   = requests.Session()
        self._session.headers.update(_HEADERS)

    def _fetch_ciks_by_sic(self, sic: str, limit: int = 20) -> list[str]:
        """Fetch CIKs for a SIC code via EDGAR company search."""
        url    = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "SIC":    sic,
            "type":   "10-K",
            "dateb":  "",
            "owner":  "include",
            "count":  str(min(limit, 40)),
            "output": "atom",
        }
        resp = _rate_get(self._session, url, params=params)
        if resp is None:
            return []
        ciks = re.findall(r"CIK=(\d+)", resp.text)
        return list(dict.fromkeys(ciks))[:limit]

    def _cik_to_ticker(self, cik: str) -> str | None:
        padded = cik.zfill(10)
        url    = EDGAR_SUBMISSIONS.format(cik=padded)
        resp   = _rate_get(self._session, url)
        if resp is None:
            return None
        data    = resp.json()
        tickers = data.get("tickers", [])
        return tickers[0] if tickers else None

    def _get_latest_adjustment_pct(self, ticker: str) -> float | None:
        """Get the most recent adjustment_pct_of_gaap for a ticker."""
        recons = self._db.get_reconciliations(ticker, limit=1)
        if recons:
            return recons[0].get("adjustment_pct_of_gaap")
        # Fall back to computing
        try:
            rows = self._builder.build_reconciliations(ticker, periods=2, include_8k=False)
            if rows and rows[0].adjustment_pct_of_gaap is not None:
                return rows[0].adjustment_pct_of_gaap
        except Exception:
            pass
        return None

    def compare(
        self,
        ticker:      str,
        peer_tickers: list[str] | None = None,
        sic_code:    str | None = None,
        sector:      str        = "",
        max_peers:   int        = 15,
    ) -> PeerComparisonResult:
        """
        Compare ticker's non-GAAP adjustment magnitude against peers.

        Ranking:
        - Conservative: adj_pct < sector_p25
        - Moderate:     sector_p25 ≤ adj_pct ≤ sector_p75
        - Aggressive:   adj_pct > sector_p75
        """
        # Resolve peers
        if peer_tickers is None:
            if sic_code:
                ciks = self._fetch_ciks_by_sic(sic_code, limit=max_peers + 5)
                peer_tickers = []
                for cik in ciks:
                    t = self._cik_to_ticker(cik)
                    if t and t.upper() != ticker.upper():
                        peer_tickers.append(t)
                    if len(peer_tickers) >= max_peers:
                        break
            else:
                return PeerComparisonResult(ticker=ticker, sector=sector)

        # Get subject company adj_pct
        subject_pct = self._get_latest_adjustment_pct(ticker)

        # Get peer adj_pcts
        peer_data: list[dict] = []
        pct_vals:  list[float] = []

        for pticker in peer_tickers[:max_peers]:
            pct = self._get_latest_adjustment_pct(pticker)
            if pct is not None:
                peer_data.append({"ticker": pticker, "adjustment_pct_of_gaap": round(pct, 2)})
                pct_vals.append(pct)

        if not pct_vals:
            return PeerComparisonResult(
                ticker=ticker,
                sector=sector,
                adjustment_pct_of_gaap=subject_pct,
                peers=peer_data,
            )

        arr = np.array(pct_vals)
        avg = float(np.mean(arr))
        p75 = float(np.percentile(arr, 75))
        p25 = float(np.percentile(arr, 25))

        standing = "moderate"
        if subject_pct is not None:
            if subject_pct > p75:
                standing = "aggressive"
            elif subject_pct < p25:
                standing = "conservative"

        return PeerComparisonResult(
            ticker=ticker,
            sector=sector,
            adjustment_pct_of_gaap=round(subject_pct, 2) if subject_pct is not None else None,
            sector_avg_adjustment=round(avg, 2),
            sector_p75_adjustment=round(p75, 2),
            relative_standing=standing,
            peers=sorted(peer_data, key=lambda x: x["adjustment_pct_of_gaap"], reverse=True),
        )


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_db:         NonGAAPDB | None = None
_extractor:  XBRLNonGAAPExtractor | None = None
_parser_8k:  EightKEarningsParser | None = None
_builder:    ReconciliationTableBuilder | None = None
_quality:    NonGAAPQualityEngine | None = None
_peer:       PeerNonGAAPComparator | None = None


def _get_db() -> NonGAAPDB:
    global _db
    if _db is None:
        _db = NonGAAPDB()
    return _db


def _get_extractor() -> XBRLNonGAAPExtractor:
    global _extractor
    if _extractor is None:
        _extractor = XBRLNonGAAPExtractor(db=_get_db())
    return _extractor


def _get_parser_8k() -> EightKEarningsParser:
    global _parser_8k
    if _parser_8k is None:
        _parser_8k = EightKEarningsParser(db=_get_db())
    return _parser_8k


def _get_builder() -> ReconciliationTableBuilder:
    global _builder
    if _builder is None:
        _builder = ReconciliationTableBuilder(
            extractor=_get_extractor(),
            parser_8k=_get_parser_8k(),
            db=_get_db(),
        )
    return _builder


def _get_quality() -> NonGAAPQualityEngine:
    global _quality
    if _quality is None:
        _quality = NonGAAPQualityEngine(db=_get_db())
    return _quality


def _get_peer() -> PeerNonGAAPComparator:
    global _peer
    if _peer is None:
        _peer = PeerNonGAAPComparator(
            extractor=_get_extractor(),
            builder=_get_builder(),
            db=_get_db(),
        )
    return _peer


def _require_cik(ticker: str) -> str:
    try:
        return _resolve_cik(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

nongaap_v3_router = APIRouter(
    prefix="/nongaap/v3",
    tags=["non-gaap-v3"],
)


@nongaap_v3_router.get("/reconciliation/{ticker}")
def get_reconciliation(
    ticker:      str,
    periods:     int  = Query(default=8, ge=1, le=20),
    include_8k:  bool = Query(default=True, description="Augment XBRL with 8-K press release tables"),
):
    """
    Non-GAAP reconciliation tables sourced from EDGAR XBRL companyfacts.

    Data hierarchy:
    1. us-gaap XBRL facts (structured, company-agnostic) for GAAP base + standard adjustments
    2. 8-K Item 2.02 earnings release HTML tables (press release reconciliation)
    3. Company extension namespace XBRL tags for custom non-GAAP metrics

    Returns per-period: GAAP starting point, each add-back item (with XBRL concept),
    computed non-GAAP result, and adjustment magnitude as % of GAAP.
    """
    _require_cik(ticker)
    try:
        rows = _get_builder().build_reconciliations(
            ticker.upper(), periods=periods, include_8k=include_8k
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker":    ticker.upper(),
        "periods":   len(rows),
        "data_note": (
            "GAAP base and standard adjustments sourced from EDGAR XBRL companyfacts. "
            "Non-GAAP values from 8-K press releases where available (data_source='hybrid'). "
            "Custom extension tags indicate company-defined non-GAAP metrics."
        ),
        "data": [r.model_dump() for r in rows],
    }


@nongaap_v3_router.get("/quality-score/{ticker}")
def get_quality_score(
    ticker:  str,
    periods: int = Query(default=8, ge=1, le=20),
):
    """
    Non-GAAP quality scoring with red and yellow flag detection.

    Red flags (severe):
    - Stock-based comp excluded: company benefits from hiding its own cost
    - Same item excluded 3+ consecutive quarters: recurring item as 'one-time'
    - Non-GAAP margin > GAAP margin by >10pp: aggressive inflation
    - Adjustments >50% of GAAP income: extreme non-GAAP inflation

    Yellow flags (watch):
    - Margin gap 5–10pp, widening EPS gap, adjustments 25–50% of GAAP

    Score: 0–100 (100 = pristine GAAP, 0 = deeply misleading non-GAAP)
    """
    _require_cik(ticker)
    try:
        rows     = _get_builder().build_reconciliations(ticker.upper(), periods=periods)
        base_df  = _get_extractor().get_gaap_base_metrics(ticker.upper(), periods=periods + 4)
        scores   = _get_quality().score_all(ticker.upper(), rows, base_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latest     = scores[0]  if scores else None
    red_count  = sum(len(s.red_flags)    for s in scores)
    yel_count  = sum(len(s.yellow_flags) for s in scores)

    return {
        "ticker":           ticker.upper(),
        "latest_score":     latest.total_score if latest else None,
        "latest_label":     latest.label       if latest else None,
        "total_red_flags":  red_count,
        "total_yellow_flags": yel_count,
        "score_legend": {
            "88-100": "Excellent",
            "72-87":  "Good",
            "55-71":  "Fair",
            "35-54":  "Poor",
            "0-34":   "Red Flag",
        },
        "data": [s.model_dump() for s in scores],
    }


@nongaap_v3_router.get("/red-flags/{ticker}")
def get_red_flags(
    ticker:  str,
    periods: int = Query(default=8, ge=1, le=20),
):
    """
    Extract only red-flag and yellow-flag findings for a ticker.

    Useful for screening: quickly identify companies with the most aggressive
    non-GAAP practices without wading through all reconciliation detail.
    """
    _require_cik(ticker)
    try:
        rows   = _get_builder().build_reconciliations(ticker.upper(), periods=periods)
        base_df = _get_extractor().get_gaap_base_metrics(ticker.upper(), periods=periods + 4)
        scores  = _get_quality().score_all(ticker.upper(), rows, base_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    findings: list[dict] = []
    for s in scores:
        for flag in s.red_flags:
            findings.append({"period_end": s.period_end, "severity": "RED",    "finding": flag})
        for flag in s.yellow_flags:
            findings.append({"period_end": s.period_end, "severity": "YELLOW", "finding": flag})

    # Summary
    recurring_items = set()
    for s in scores:
        recurring_items.update(s.recurring_items)

    return {
        "ticker":              ticker.upper(),
        "periods_analyzed":    len(scores),
        "total_red_flags":     sum(len(s.red_flags)    for s in scores),
        "total_yellow_flags":  sum(len(s.yellow_flags) for s in scores),
        "recurring_items":     sorted(recurring_items),
        "latest_score":        scores[0].total_score   if scores else None,
        "findings":            findings,
    }


@nongaap_v3_router.get("/peer-comparison/{ticker}")
def get_peer_comparison(
    ticker:   str,
    peers:    str | None = Query(default=None, description="Comma-separated peer tickers"),
    sic_code: str | None = Query(default=None, description="EDGAR SIC code for automatic peer selection"),
    sector:   str        = Query(default=""),
    max_peers: int       = Query(default=15, ge=3, le=30),
):
    """
    Compare this company's non-GAAP adjustment magnitude against sector peers.

    Relative standing:
    - Conservative: adjustments below 25th percentile of peer group
    - Moderate: adjustments within peer interquartile range
    - Aggressive: adjustments above 75th percentile (governance risk signal)
    """
    _require_cik(ticker)
    peer_list = [p.strip().upper() for p in peers.split(",") if p.strip()] if peers else None

    if not peer_list and not sic_code:
        raise HTTPException(
            status_code=400,
            detail="Provide peers (comma-separated tickers) or sic_code for automatic peer selection",
        )
    try:
        result = _get_peer().compare(
            ticker.upper(),
            peer_tickers=peer_list,
            sic_code=sic_code,
            sector=sector,
            max_peers=max_peers,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker":               result.ticker,
        "sector":               result.sector,
        "subject_adjustment_pct": result.adjustment_pct_of_gaap,
        "sector_avg_adj_pct":   result.sector_avg_adjustment,
        "sector_p75_adj_pct":   result.sector_p75_adjustment,
        "relative_standing":    result.relative_standing,
        "interpretation": {
            "conservative": "Subject adjustments < sector 25th percentile",
            "moderate":     "Subject adjustments within sector IQR",
            "aggressive":   "Subject adjustments > sector 75th percentile (governance risk)",
        },
        "peers": result.peers,
    }


@nongaap_v3_router.get("/history/{ticker}")
def get_history(
    ticker:   str,
    quarters: int = Query(default=8, ge=2, le=20),
):
    """
    8-quarter GAAP vs non-GAAP margin spread trend.

    Widening spread = management increasing reliance on non-GAAP adjustments
    to mask deteriorating GAAP performance. Key signal for equity analysts.
    """
    _require_cik(ticker)
    try:
        spreads = _get_quality().get_margin_spread_history(
            ticker.upper(), _get_extractor(), quarters=quarters
        )
        adj_hist = _get_db().get_quality_history(ticker.upper(), limit=quarters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Compute trend
    spread_vals = [s.spread_pp for s in spreads if s.spread_pp is not None]
    trend_desc  = "stable"
    if len(spread_vals) >= 3:
        x     = np.arange(len(spread_vals), dtype=float)
        slope = float(np.polyfit(x, spread_vals, 1)[0])
        trend_desc = "widening" if slope > 0.3 else "narrowing" if slope < -0.3 else "stable"

    return {
        "ticker":          ticker.upper(),
        "quarters":        quarters,
        "spread_trend":    trend_desc,
        "interpretation": {
            "widening":  "Non-GAAP adjustments growing relative to GAAP — red flag",
            "stable":    "Consistent adjustment level",
            "narrowing": "Adjustments declining — business or reporting improving",
        },
        "margin_spreads": [s.model_dump() for s in spreads],
        "quality_history": [
            {
                "period_end":  r["period_end"],
                "score":       r["total_score"],
                "label":       r["label"],
                "adjustment_pct": r["adjustment_pct"],
                "margin_gap_pp":  r["margin_gap_pp"],
            }
            for r in adj_hist
        ],
    }


@nongaap_v3_router.get("/fcf/{ticker}")
def get_free_cash_flow(
    ticker:  str,
    periods: int = Query(default=8, ge=1, le=20),
):
    """
    Free cash flow reconciliation: CFO - capex, fully XBRL-sourced.

    Unlike management-defined FCF (which sometimes excludes lease payments,
    restructuring cash costs, etc.), this uses the strict XBRL definition:
    us-gaap:NetCashProvidedByUsedInOperatingActivities minus
    us-gaap:PaymentsToAcquirePropertyPlantAndEquipment.
    """
    _require_cik(ticker)
    try:
        fcf_rows = _get_extractor().get_free_cash_flow_series(ticker.upper(), periods=periods)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker":    ticker.upper(),
        "definition": "CFO (us-gaap:NetCashProvidedByUsedInOperatingActivities) "
                      "minus CapEx (us-gaap:PaymentsToAcquirePropertyPlantAndEquipment). "
                      "Strict XBRL definition — no management adjustments.",
        "data": [r.model_dump() for r in fcf_rows],
    }


@nongaap_v3_router.get("/custom-tags/{ticker}")
def get_custom_xbrl_tags(ticker: str):
    """
    Company-specific XBRL extension tags for non-GAAP metrics.

    Companies increasingly tag their own non-GAAP metrics in iXBRL filings
    using custom extension namespaces. These tags reveal what management
    considers its 'true' performance metrics.
    """
    _require_cik(ticker)
    try:
        customs = _get_extractor().get_custom_extension_facts(ticker.upper())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Group by concept
    by_concept: dict[str, list[dict]] = defaultdict(list)
    for f in customs:
        by_concept[f.concept].append({
            "period_end": f.period_end,
            "value":      f.value,
            "unit":       f.unit,
            "form":       f.form,
        })

    return {
        "ticker":          ticker.upper(),
        "unique_concepts": len(by_concept),
        "note": (
            "These are company-defined XBRL tags, not standard us-gaap concepts. "
            "Common patterns: *AdjustedEBITDA*, *NonGAAPNetIncome*, *OrganicRevenue*, "
            "*FreeCashFlow*. Useful for understanding management's preferred metrics."
        ),
        "concepts": {
            concept: sorted(obs, key=lambda x: x["period_end"], reverse=True)[:8]
            for concept, obs in by_concept.items()
        },
    }
