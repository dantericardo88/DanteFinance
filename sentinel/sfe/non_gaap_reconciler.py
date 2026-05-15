"""
Non-GAAP reconciliation: parse management's adjusted metrics,
detect optimistic adjustments, compute normalized earnings.
GAAP vs non-GAAP comparison with quality scoring.

Dimension: dim_017 — Non-GAAP reconciliation tables (target: 9, from 8)

Components:
  NonGAAPParser              — HTML table extraction from EDGAR 10-Q/10-K filings
  AdjustmentQualityScorer    — Score adjustment quality with red-flag detection
  NormalizedEarningsEngine   — Through-the-cycle normalized earnings
  CrossCompanyNonGAAPBenchmark — Sector comparison of GAAP vs non-GAAP gaps
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Tag
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field as PydanticField

from sentinel.core.logging import get_logger
from sentinel.sfe.standardized_financials import (
    EDGAR_BASE,
    EDGAR_HEADERS,
    FinancialsCache,
    _RATE_DELAY,
    _TIMEOUT,
    _MAX_RETRY,
    _safe_div,
    resolve_cik,
)

logger = get_logger(__name__)

__all__ = [
    "NonGAAPParser",
    "AdjustmentQualityScorer",
    "NormalizedEarningsEngine",
    "CrossCompanyNonGAAPBenchmark",
    "ReconciliationTable",
    "AdjustmentItem",
    "NonGAAPQualityReport",
    "NormalizedEPS",
    "nongaap_router",
]

# ---------------------------------------------------------------------------
# EDGAR filing constants
# ---------------------------------------------------------------------------

EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{}.json"
EDGAR_FILING_DOC  = "https://www.sec.gov/Archives/edgar/data/{}/{}"
SEC_FULL_TEXT     = "https://efts.sec.gov/LATEST/search-index?q={}&dateRange=custom&startdt={}&enddt={}&forms=10-K,10-Q"

# ---------------------------------------------------------------------------
# Adjustment classification taxonomy
# ---------------------------------------------------------------------------

# Adjustments that are sometimes legitimate (still need frequency check)
RECURRING_ADJUSTMENTS: set[str] = {
    "stock-based compensation",
    "stock based compensation",
    "share-based compensation",
    "share based compensation",
    "amortization of acquired intangibles",
    "amortization of intangible assets",
    "acquired intangibles amortization",
    "depreciation and amortization",
    "d&a",
}

# Questionable — legitimate occasionally but often abused
QUESTIONABLE_ADJUSTMENTS: set[str] = {
    "restructuring",
    "restructuring charges",
    "restructuring and related charges",
    "restructuring costs",
    "severance",
    "workforce reduction",
    "customer acquisition costs",
    "acquisition costs",
    "transaction costs",
    "integration costs",
    "merger-related costs",
    "deal costs",
    "advisory fees",
    "impairment",
    "asset impairment",
    "goodwill impairment",
    "intangible impairment",
}

# Red flags — these should almost never be excluded
RED_FLAG_ADJUSTMENTS: set[str] = {
    "cost of revenue",
    "cost of goods sold",
    "cost of sales",
    "normalized revenue",
    "pro forma revenue",
    "normalized" ,
    "cash rent",
    "adjusted rent",
    "customer success",
    "sales and marketing",
    "research and development",
    "r&d expense",
    "income tax benefit",
    "tax benefit",
    "interest expense",
}

# Litigation / legal (moderate concern)
LITIGATION_ADJUSTMENTS: set[str] = {
    "litigation",
    "legal settlements",
    "legal charges",
    "regulatory settlements",
    "legal and regulatory",
    "class action",
    "settlement",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AdjustmentItem(BaseModel):
    name: str
    value: Optional[float] = None
    category: str = "unknown"  # recurring | questionable | red_flag | litigation | other
    is_recurring_flag: bool = False
    consecutive_years: int = 0
    description: str = ""


class ReconciliationTable(BaseModel):
    ticker: str
    cik: Optional[str] = None
    period: str = ""
    filing_type: str = ""
    accession: str = ""
    gaap_metric: str = ""
    gaap_value: Optional[float] = None
    non_gaap_metric: str = ""
    non_gaap_value: Optional[float] = None
    adjustments: list[AdjustmentItem] = PydanticField(default_factory=list)
    total_adjustments: Optional[float] = None
    adjustment_magnitude_pct: Optional[float] = None  # adj / gaap_value


class NonGAAPQualityReport(BaseModel):
    ticker: str
    period: str
    filing_type: str = ""
    quality_score: float = 0.0  # 0-100
    quality_label: str = "Unknown"
    adjustment_magnitude_pct: Optional[float] = None
    recurring_adjustment_count: int = 0
    questionable_adjustment_count: int = 0
    red_flag_count: int = 0
    consecutive_restructuring_years: int = 0
    adjustments: list[AdjustmentItem] = PydanticField(default_factory=list)
    flags: list[str] = PydanticField(default_factory=list)
    gaap_eps: Optional[float] = None
    reported_nongaap_eps: Optional[float] = None
    eps_gap_pct: Optional[float] = None


class NormalizedEPS(BaseModel):
    ticker: str
    period: str
    gaap_eps: Optional[float] = None
    reported_nongaap_eps: Optional[float] = None
    sentinel_normalized_eps: Optional[float] = None
    normalization_method: str = ""
    five_year_avg_margin: Optional[float] = None
    current_revenue: Optional[float] = None
    through_cycle_net_income: Optional[float] = None
    shares_outstanding: Optional[float] = None
    adjustment_from_gaap: Optional[float] = None
    adjustment_from_nongaap: Optional[float] = None


class SectorNonGAAPComparison(BaseModel):
    sector: str
    companies: list[dict] = PydanticField(default_factory=list)
    avg_adjustment_pct: Optional[float] = None
    max_adjustment_pct: Optional[float] = None
    industry_standard_adjustments: list[str] = PydanticField(default_factory=list)
    highest_quality_companies: list[str] = PydanticField(default_factory=list)
    lowest_quality_companies: list[str] = PydanticField(default_factory=list)


# ---------------------------------------------------------------------------
# Non-GAAP Parser
# ---------------------------------------------------------------------------


class NonGAAPParser:
    """
    Extract non-GAAP reconciliation tables from EDGAR 10-K/10-Q filings.

    Strategy:
    1. Fetch filing index from EDGAR submissions API
    2. Retrieve the primary HTML filing document
    3. Parse reconciliation tables using BeautifulSoup
    4. Classify adjustments using keyword matching
    5. Extract GAAP vs non-GAAP EPS/earnings figures

    Handles the most common formats:
    - "Reconciliation of GAAP to Non-GAAP" tables
    - "Adjusted EBITDA reconciliation" tables
    - "Non-GAAP earnings per share" tables
    """

    # Table header patterns that signal reconciliation tables
    RECONCILIATION_HEADERS = [
        r"reconciliation.{0,50}gaap.{0,50}non.gaap",
        r"reconciliation.{0,50}non.gaap.{0,50}gaap",
        r"non.gaap.{0,50}reconciliation",
        r"adjusted.{0,30}reconciliation",
        r"gaap.{0,30}to.{0,30}non.gaap",
        r"gaap.{0,30}non.gaap.{0,30}measure",
        r"adjusted ebitda",
        r"adjusted earnings",
        r"adjusted eps",
        r"non.gaap.{0,30}earnings",
        r"non.gaap.{0,30}income",
        r"non.gaap.{0,30}operating",
        r"reconciling item",
    ]

    # Patterns to identify GAAP line in reconciliation
    GAAP_LINE_PATTERNS = [
        r"^gaap\s",
        r"gaap net income",
        r"gaap earnings",
        r"gaap operating income",
        r"net income \(loss\)",
        r"net income",
        r"net earnings",
        r"operating income",
    ]

    # Patterns to identify the non-GAAP result line
    NONGAAP_LINE_PATTERNS = [
        r"non.gaap.{0,40}income",
        r"non.gaap.{0,40}earnings",
        r"adjusted.{0,30}income",
        r"adjusted.{0,30}earnings",
        r"adjusted ebitda",
        r"adjusted operating income",
        r"non.gaap net income",
    ]

    def __init__(self, cache_path: str | None = None) -> None:
        self._cache = FinancialsCache(db_path=cache_path)
        self._http = httpx.Client(
            headers=EDGAR_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        )

    def _get_filings_list(
        self,
        cik: str,
        form_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Fetch list of filings from EDGAR submissions API.
        Returns list of {accession, form, filingDate, primaryDocument}.
        """
        if form_types is None:
            form_types = ["10-K", "10-Q"]

        url = EDGAR_SUBMISSIONS.format(cik.zfill(10))
        try:
            time.sleep(_RATE_DELAY)
            resp = self._http.get(url)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("submissions_fetch_failed", cik=cik, error=str(exc))
            return []

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])
        primary_docs = filings.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form in form_types:
                if len(results) >= limit:
                    break
                results.append({
                    "form": form,
                    "accession": accessions[i] if i < len(accessions) else "",
                    "filingDate": dates[i] if i < len(dates) else "",
                    "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
                })

        return results

    def _fetch_filing_html(self, cik: str, accession: str, primary_doc: str) -> str | None:
        """Fetch the HTML content of a filing document."""
        # Normalize accession number format
        acc_clean = accession.replace("-", "")
        acc_formatted = accession if "-" in accession else (
            f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
        )

        # Try multiple URL formats
        urls_to_try = [
            EDGAR_FILING_DOC.format(cik.lstrip("0"), f"{acc_clean}/{primary_doc}"),
            EDGAR_FILING_DOC.format(cik.lstrip("0"), f"{acc_formatted.replace('-', '')}/{primary_doc}"),
        ]

        for url in urls_to_try:
            try:
                time.sleep(_RATE_DELAY)
                resp = self._http.get(url)
                if resp.status_code == 200:
                    return resp.text
            except httpx.HTTPError as exc:
                logger.debug("filing_fetch_failed", url=url, error=str(exc))

        return None

    def _extract_number(self, text: str) -> float | None:
        """Extract numeric value from table cell text."""
        if not text:
            return None
        # Clean: remove $, commas, spaces, parentheses (neg), %
        cleaned = text.strip()
        is_negative = cleaned.startswith("(") or cleaned.startswith("-")
        cleaned = re.sub(r"[$,%\s]", "", cleaned)
        cleaned = re.sub(r"[()]", "", cleaned)
        cleaned = re.sub(r"[,\s]", "", cleaned)

        try:
            value = float(cleaned)
            return -value if is_negative else value
        except ValueError:
            return None

    def _classify_adjustment(self, name: str) -> str:
        """Classify an adjustment item by name."""
        name_lower = name.lower().strip()

        for pattern in RED_FLAG_ADJUSTMENTS:
            if pattern in name_lower:
                return "red_flag"

        for pattern in QUESTIONABLE_ADJUSTMENTS:
            if pattern in name_lower:
                return "questionable"

        for pattern in LITIGATION_ADJUSTMENTS:
            if pattern in name_lower:
                return "litigation"

        for pattern in RECURRING_ADJUSTMENTS:
            if pattern in name_lower:
                return "recurring"

        return "other"

    def _is_reconciliation_table(self, table: Tag) -> bool:
        """Check if an HTML table contains non-GAAP reconciliation data."""
        table_text = table.get_text(separator=" ").lower()
        for pattern in self.RECONCILIATION_HEADERS:
            if re.search(pattern, table_text, re.IGNORECASE):
                return True

        # Also check preceding sibling text (table caption or nearby header)
        prev = table.find_previous(["h1", "h2", "h3", "h4", "p", "div"])
        if prev:
            prev_text = prev.get_text(separator=" ").lower()
            for pattern in self.RECONCILIATION_HEADERS:
                if re.search(pattern, prev_text, re.IGNORECASE):
                    return True

        return False

    def extract_reconciliation_table(self, filing_html: str) -> list[dict]:
        """
        Parse HTML filing and extract all non-GAAP reconciliation tables.

        Returns list of raw reconciliation dicts:
        {gaap_line, nongaap_line, adjustments: [(name, value)], period}
        """
        soup = BeautifulSoup(filing_html, "html.parser")
        tables = soup.find_all("table")
        results = []

        for table in tables:
            if not self._is_reconciliation_table(table):
                continue

            rows = table.find_all("tr")
            if len(rows) < 3:
                continue

            # Parse headers to find period columns
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(["th", "td"]):
                headers.append(th.get_text(separator=" ").strip())

            # Find the column with numeric data (usually last 1-2 columns)
            # Try to detect period headers (Q1 2024, FY 2023, etc.)
            period_cols = []
            for i, h in enumerate(headers):
                if re.search(r"\d{4}", h) or re.search(r"q[1-4]|fy|ytd|ttm|annual|quarter", h, re.I):
                    period_cols.append(i)

            if not period_cols:
                # Default: use last column as primary numeric column
                period_cols = [len(headers) - 1] if headers else []

            # Parse data rows
            gaap_value = None
            gaap_line_name = ""
            nongaap_value = None
            nongaap_line_name = ""
            adjustments: list[tuple[str, float]] = []
            in_adjustments = False

            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue

                label = cells[0].get_text(separator=" ").strip()
                label_lower = label.lower()

                # Get value from the primary period column
                value = None
                for col_idx in period_cols:
                    if col_idx < len(cells):
                        raw = cells[col_idx].get_text(separator=" ").strip()
                        value = self._extract_number(raw)
                        if value is not None:
                            break

                # Identify GAAP line
                is_gaap = any(
                    re.search(p, label_lower)
                    for p in self.GAAP_LINE_PATTERNS
                )

                # Identify non-GAAP result line
                is_nongaap = any(
                    re.search(p, label_lower)
                    for p in self.NONGAAP_LINE_PATTERNS
                )

                if is_gaap and gaap_value is None:
                    gaap_value = value
                    gaap_line_name = label
                    in_adjustments = True
                elif is_nongaap:
                    nongaap_value = value
                    nongaap_line_name = label
                    in_adjustments = False
                elif in_adjustments and label and value is not None:
                    # Skip subtotals and separator rows
                    if not re.search(r"^total|^sub.total|^net total", label_lower):
                        adjustments.append((label, value))

            if gaap_value is not None or nongaap_value is not None:
                # Detect period from headers
                period = ""
                for h in headers[1:]:
                    if re.search(r"\d{4}", h):
                        period = h.strip()
                        break

                results.append({
                    "gaap_line": gaap_line_name,
                    "gaap_value": gaap_value,
                    "nongaap_line": nongaap_line_name,
                    "nongaap_value": nongaap_value,
                    "adjustments": adjustments,
                    "period": period,
                    "headers": headers,
                })

        return results

    def get_reconciliation_tables(
        self,
        ticker: str,
        num_filings: int = 8,
        form_types: list[str] | None = None,
    ) -> list[ReconciliationTable]:
        """
        Fetch and parse reconciliation tables for a ticker across recent filings.

        Parameters
        ----------
        ticker      : stock ticker symbol
        num_filings : number of recent filings to parse
        form_types  : list of form types to search (default: ["10-K", "10-Q"])
        """
        if form_types is None:
            form_types = ["10-K", "10-Q"]

        cik = resolve_cik(ticker)
        cik_padded = cik.zfill(10)
        filings = self._get_filings_list(cik_padded, form_types=form_types, limit=num_filings)

        results: list[ReconciliationTable] = []

        for filing in filings:
            accession = filing.get("accession", "")
            primary_doc = filing.get("primaryDocument", "")
            form = filing.get("form", "")
            filing_date = filing.get("filingDate", "")

            if not accession or not primary_doc:
                continue

            html = self._fetch_filing_html(cik_padded, accession, primary_doc)
            if not html:
                logger.debug("no_html", accession=accession)
                continue

            try:
                raw_tables = self.extract_reconciliation_table(html)
            except Exception as exc:
                logger.warning("parse_failed", accession=accession, error=str(exc))
                continue

            for raw in raw_tables:
                adjustment_items = []
                total_adj = 0.0

                for adj_name, adj_value in raw.get("adjustments", []):
                    category = self._classify_adjustment(adj_name)
                    adjustment_items.append(AdjustmentItem(
                        name=adj_name,
                        value=adj_value,
                        category=category,
                        description=f"From {form} filed {filing_date}",
                    ))
                    total_adj += adj_value or 0.0

                gaap_val = raw.get("gaap_value")
                adj_magnitude = None
                if gaap_val and gaap_val != 0:
                    adj_magnitude = (total_adj / abs(gaap_val)) * 100.0

                results.append(ReconciliationTable(
                    ticker=ticker,
                    cik=cik_padded,
                    period=raw.get("period", filing_date[:7] if filing_date else ""),
                    filing_type=form,
                    accession=accession,
                    gaap_metric=raw.get("gaap_line", "GAAP Net Income"),
                    gaap_value=gaap_val,
                    non_gaap_metric=raw.get("nongaap_line", "Non-GAAP Net Income"),
                    non_gaap_value=raw.get("nongaap_value"),
                    adjustments=adjustment_items,
                    total_adjustments=total_adj if adjustment_items else None,
                    adjustment_magnitude_pct=adj_magnitude,
                ))

        return results


# ---------------------------------------------------------------------------
# Adjustment Quality Scorer
# ---------------------------------------------------------------------------


class AdjustmentQualityScorer:
    """
    Score the quality of non-GAAP adjustments on a 0-100 scale.

    Scoring methodology:
    - Starts at 100 (pristine: no adjustments)
    - Deductions for each type of adjustment found
    - Extra deductions for recurring "one-time" items
    - Extra deductions for high magnitude adjustments
    - Hard cap at 0 for egregious cases

    Score interpretation:
    100   — No adjustments (purely GAAP)
    85-99 — Minor, clearly legitimate (D&A on acquired intangibles only)
    70-84 — Moderate, largely standard but watch trends
    50-69 — Material adjustments, some questionable items
    30-49 — Aggressive, multiple questionable items or high magnitude
    0-29  — Highly aggressive, red flags present
    """

    # Deduction points per adjustment category
    DEDUCTIONS = {
        "recurring":     5,    # small: sometimes legitimate
        "questionable": 15,    # moderate: needs justification
        "litigation":   10,    # moderate: one-off but watch
        "red_flag":     30,    # severe: almost never justified
        "other":         8,    # unknown: default moderate deduction
    }

    # Extra deductions
    RECURRING_EXTRA_DEDUCTION = 10   # per item that recurs 3+ consecutive years
    HIGH_MAGNITUDE_THRESHOLD  = 0.30  # > 30% = aggressive
    HIGH_MAGNITUDE_DEDUCTION  = 20
    VERY_HIGH_MAGNITUDE_DEDUCTION = 35  # > 50%

    def __init__(self, parser: NonGAAPParser | None = None) -> None:
        self._parser = parser or NonGAAPParser()

    def _check_recurring_items(
        self,
        tables_history: list[ReconciliationTable],
        item_name: str,
    ) -> int:
        """Count how many consecutive periods a named adjustment appears."""
        count = 0
        item_lower = item_name.lower()
        for table in sorted(tables_history, key=lambda t: t.period, reverse=True):
            found = any(
                item_lower in adj.name.lower()
                for adj in table.adjustments
            )
            if found:
                count += 1
            else:
                break  # must be consecutive
        return count

    def score_reconciliation(
        self,
        table: ReconciliationTable,
        history: list[ReconciliationTable] | None = None,
    ) -> NonGAAPQualityReport:
        """
        Score a single reconciliation table.

        Parameters
        ----------
        table   : current period reconciliation table
        history : historical tables for this ticker (for recurrence checks)
        """
        if history is None:
            history = []

        score = 100.0
        flags = []
        recurring_count = 0
        questionable_count = 0
        red_flag_count = 0
        max_consecutive_restructuring = 0

        scored_adjustments = []

        for adj in table.adjustments:
            category = adj.category
            deduction = self.DEDUCTIONS.get(category, self.DEDUCTIONS["other"])

            # Check recurrence in history
            consecutive = self._check_recurring_items(history + [table], adj.name)
            adj.consecutive_years = consecutive

            if consecutive >= 3 and category in ("questionable", "litigation", "other"):
                adj.is_recurring_flag = True
                deduction += self.RECURRING_EXTRA_DEDUCTION
                flags.append(
                    f"'{adj.name}' appears for {consecutive} consecutive periods — "
                    f"not truly 'one-time'"
                )

            # Category-specific flags
            if category == "red_flag":
                flags.append(
                    f"Red flag: '{adj.name}' is an operating cost excluded from non-GAAP"
                )
                red_flag_count += 1
            elif category == "questionable":
                questionable_count += 1
            elif category == "recurring":
                recurring_count += 1

            # Restructuring streak
            if "restructur" in adj.name.lower():
                max_consecutive_restructuring = max(
                    max_consecutive_restructuring, consecutive
                )
                if consecutive >= 3:
                    flags.append(
                        f"Restructuring charges for {consecutive} consecutive years "
                        f"— this is an operating cost"
                    )

            score -= deduction
            scored_adjustments.append(adj)

        # Magnitude check
        adj_magnitude = table.adjustment_magnitude_pct
        if adj_magnitude is not None:
            if abs(adj_magnitude) > 50:
                score -= self.VERY_HIGH_MAGNITUDE_DEDUCTION
                flags.append(
                    f"Very high adjustment magnitude: {adj_magnitude:.1f}% of GAAP earnings — "
                    f"highly aggressive"
                )
            elif abs(adj_magnitude) > 30:
                score -= self.HIGH_MAGNITUDE_DEDUCTION
                flags.append(
                    f"High adjustment magnitude: {adj_magnitude:.1f}% of GAAP earnings — "
                    f"aggressive"
                )

        # No adjustments is perfect
        if not table.adjustments:
            score = 100.0

        # Clamp to [0, 100]
        score = max(0.0, min(100.0, score))

        # Label
        if score >= 90:
            label = "Excellent"
        elif score >= 75:
            label = "Good"
        elif score >= 55:
            label = "Fair"
        elif score >= 35:
            label = "Poor"
        else:
            label = "Red Flag"

        # Compute EPS gap
        gaap_eps = None
        nongaap_eps = None
        eps_gap = None
        if table.gaap_value and table.non_gaap_value:
            if table.gaap_value != 0:
                eps_gap = ((table.non_gaap_value - table.gaap_value) / abs(table.gaap_value)) * 100

        return NonGAAPQualityReport(
            ticker=table.ticker,
            period=table.period,
            filing_type=table.filing_type,
            quality_score=round(score, 1),
            quality_label=label,
            adjustment_magnitude_pct=adj_magnitude,
            recurring_adjustment_count=recurring_count,
            questionable_adjustment_count=questionable_count,
            red_flag_count=red_flag_count,
            consecutive_restructuring_years=max_consecutive_restructuring,
            adjustments=scored_adjustments,
            flags=flags,
            gaap_eps=table.gaap_value,
            reported_nongaap_eps=table.non_gaap_value,
            eps_gap_pct=eps_gap,
        )

    def score_ticker(
        self,
        ticker: str,
        num_filings: int = 8,
    ) -> list[NonGAAPQualityReport]:
        """
        Score all reconciliation tables for a ticker.
        Uses filing history for recurrence detection.
        """
        tables = self._parser.get_reconciliation_tables(ticker, num_filings=num_filings)

        if not tables:
            return []

        # Sort by period ascending
        tables_sorted = sorted(tables, key=lambda t: t.period)
        reports = []

        for i, table in enumerate(tables_sorted):
            history = tables_sorted[:i]  # all prior periods
            report = self.score_reconciliation(table, history=history)
            reports.append(report)

        return reports

    def get_trending_quality(
        self,
        reports: list[NonGAAPQualityReport],
    ) -> dict[str, Any]:
        """
        Analyze quality trend over time.
        Deteriorating quality (decreasing score) = warning signal.
        """
        if not reports:
            return {"trend": "insufficient_data", "data": []}

        scores = [r.quality_score for r in reports]
        periods = [r.period for r in reports]

        # Simple linear trend
        if len(scores) >= 2:
            x = np.arange(len(scores), dtype=float)
            slope = float(np.polyfit(x, scores, 1)[0])
            trend = "improving" if slope > 1 else "deteriorating" if slope < -1 else "stable"
        else:
            slope = 0.0
            trend = "insufficient_data"

        return {
            "trend": trend,
            "slope_per_period": round(slope, 2),
            "latest_score": scores[-1] if scores else None,
            "average_score": round(float(np.mean(scores)), 1) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "data": [{"period": p, "score": s} for p, s in zip(periods, scores)],
        }


# ---------------------------------------------------------------------------
# Normalized Earnings Engine
# ---------------------------------------------------------------------------


class NormalizedEarningsEngine:
    """
    Compute normalized (through-the-cycle) earnings.

    Method:
    1. Pull 5-year historical net income margins (GAAP)
    2. Apply 5-year average margin to current revenue → normalized net income
    3. Divide by current shares outstanding → normalized EPS
    4. Compare to: GAAP EPS, management's reported non-GAAP EPS

    This strips out:
    - Peak/trough cyclicality
    - One-time charges that are NOT included in our normalization
    - D&A acceleration from acquisitions (normalized D&A applied)
    """

    _ASSUMED_TAX = 0.21

    def __init__(
        self,
        parser: NonGAAPParser | None = None,
        cache_path: str | None = None,
    ) -> None:
        self._parser = parser or NonGAAPParser()
        self._cache = FinancialsCache(db_path=cache_path)
        self._http = httpx.Client(headers=EDGAR_HEADERS, timeout=_TIMEOUT)

    def _get_income_data(self, cik: str, periods: int = 8) -> pd.DataFrame:
        """Pull historical income statement data from EDGAR."""
        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        std = FinancialStatementStandardizer(cache_path=None)
        return std.get_income_statement(cik, periods=periods, period_type="annual")

    def _get_shares_outstanding(self, facts: dict) -> float | None:
        """Get most recent shares outstanding from EDGAR facts."""
        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        std = FinancialStatementStandardizer()
        concepts = [
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "CommonStockSharesOutstanding",
        ]
        series = std.extract_metric(facts, concepts, "annual", units="shares")
        if not series.empty:
            return float(series.iloc[-1]["value"])
        return None

    def compute_normalized_eps(
        self,
        ticker: str,
        years: int = 5,
    ) -> NormalizedEPS:
        """
        Compute SENTINEL-normalized EPS alongside GAAP and reported non-GAAP.

        Parameters
        ----------
        ticker : stock ticker
        years  : number of years to use for through-cycle margin average

        Returns
        -------
        NormalizedEPS with all three EPS variants
        """
        cik = resolve_cik(ticker)

        # Fetch income history
        is_df = self._get_income_data(cik, periods=years + 2)

        if is_df.empty:
            return NormalizedEPS(
                ticker=ticker,
                period="N/A",
                normalization_method="insufficient_data",
            )

        # Fetch company facts for shares
        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        std = FinancialStatementStandardizer()
        facts = std.get_company_facts(cik)

        shares = self._get_shares_outstanding(facts)

        # 5-year average net margin
        if "net_income" not in is_df.columns or "revenue" not in is_df.columns:
            return NormalizedEPS(
                ticker=ticker,
                period=str(is_df.index[-1].date()) if not is_df.empty else "N/A",
                normalization_method="missing_margin_data",
            )

        hist = is_df.tail(years)
        margins = hist["net_income"] / hist["revenue"].replace(0, float("nan"))
        avg_margin = float(margins.dropna().mean()) if not margins.dropna().empty else None

        # Current period data
        latest = is_df.iloc[-1]
        current_revenue = float(latest.get("revenue", 0) or 0)
        current_ni = float(latest.get("net_income", 0) or 0)
        current_period = str(is_df.index[-1].date())

        # GAAP EPS
        gaap_eps = None
        if "eps_diluted" in is_df.columns:
            gaap_eps = float(latest.get("eps_diluted") or 0) or None
        elif shares and shares > 0:
            gaap_eps = current_ni / shares if current_ni else None

        # Normalized net income = avg_margin × current_revenue
        normalized_ni = None
        normalized_eps = None
        if avg_margin is not None and current_revenue > 0:
            normalized_ni = avg_margin * current_revenue
            if shares and shares > 0:
                normalized_eps = normalized_ni / shares

        # Try to get reported non-GAAP from reconciliation tables
        reported_nongaap_eps = None
        try:
            tables = self._parser.get_reconciliation_tables(ticker, num_filings=2)
            if tables:
                latest_table = max(tables, key=lambda t: t.period)
                # If gaap_value looks like per-share (< 100), treat as EPS
                if latest_table.non_gaap_value is not None:
                    if abs(latest_table.non_gaap_value) < 200:
                        reported_nongaap_eps = latest_table.non_gaap_value
                    elif shares and shares > 0:
                        reported_nongaap_eps = latest_table.non_gaap_value / shares
        except Exception as exc:
            logger.debug("nongaap_fetch_skipped", ticker=ticker, error=str(exc))

        # Adjustment vs GAAP and non-GAAP
        adj_from_gaap = None
        adj_from_nongaap = None
        if gaap_eps and normalized_eps:
            adj_from_gaap = normalized_eps - gaap_eps
        if reported_nongaap_eps and normalized_eps:
            adj_from_nongaap = normalized_eps - reported_nongaap_eps

        return NormalizedEPS(
            ticker=ticker,
            period=current_period,
            gaap_eps=gaap_eps,
            reported_nongaap_eps=reported_nongaap_eps,
            sentinel_normalized_eps=normalized_eps,
            normalization_method=f"{years}-year avg margin ({avg_margin:.1%}) × current revenue",
            five_year_avg_margin=avg_margin,
            current_revenue=current_revenue,
            through_cycle_net_income=normalized_ni,
            shares_outstanding=shares,
            adjustment_from_gaap=adj_from_gaap,
            adjustment_from_nongaap=adj_from_nongaap,
        )

    def compute_normalized_series(
        self,
        ticker: str,
        periods: int = 5,
        lookback_years: int = 5,
    ) -> pd.DataFrame:
        """
        Compute normalized EPS for each period in the last N years.
        For each period, uses rolling lookback_years to compute avg margin.

        Returns DataFrame indexed by period_end.
        """
        cik = resolve_cik(ticker)
        is_df = self._get_income_data(cik, periods=periods + lookback_years)

        if is_df.empty or "net_income" not in is_df.columns or "revenue" not in is_df.columns:
            return pd.DataFrame()

        from sentinel.sfe.standardized_financials import FinancialStatementStandardizer
        std = FinancialStatementStandardizer()
        facts = std.get_company_facts(cik)
        shares = self._get_shares_outstanding(facts)

        # Rolling margin computation
        margins = (is_df["net_income"] / is_df["revenue"].replace(0, float("nan"))).rename("net_margin")
        rolling_avg_margin = margins.rolling(window=lookback_years, min_periods=2).mean()

        result = pd.DataFrame(index=is_df.tail(periods).index)
        result["gaap_net_income"] = is_df["net_income"].tail(periods)
        result["revenue"] = is_df["revenue"].tail(periods)
        result["gaap_margin"] = margins.tail(periods)
        result["rolling_avg_margin"] = rolling_avg_margin.tail(periods)

        # Normalized NI = rolling_avg_margin × current revenue
        result["normalized_net_income"] = (
            result["rolling_avg_margin"] * result["revenue"]
        )

        if shares and shares > 0:
            result["gaap_eps"] = result["gaap_net_income"] / shares
            result["normalized_eps"] = result["normalized_net_income"] / shares
            if "eps_diluted" in is_df.columns:
                result["gaap_eps"] = is_df["eps_diluted"].tail(periods)
        else:
            if "eps_diluted" in is_df.columns:
                result["gaap_eps"] = is_df["eps_diluted"].tail(periods)
            else:
                result["gaap_eps"] = None
            result["normalized_eps"] = None

        result.index.name = "period_end"
        return result

    def compute_normalized_d_and_a(
        self,
        ticker: str,
        periods: int = 5,
    ) -> pd.Series:
        """
        Compute normalized D&A: straight-line estimate.
        Uses average of historical D&A as a percentage of PP&E.
        """
        cik = resolve_cik(ticker)
        from sentinel.sfe.standardized_financials import (
            FinancialStatementStandardizer,
            INCOME_STATEMENT_MAP,
            BALANCE_SHEET_MAP,
        )
        std = FinancialStatementStandardizer()
        facts = std.get_company_facts(cik)

        da_concepts = INCOME_STATEMENT_MAP.get("depreciation", [])
        ppe_concepts = BALANCE_SHEET_MAP.get("ppe_net", [])

        if not da_concepts or not ppe_concepts:
            return pd.Series(dtype=float)

        da_series = std.extract_metric(facts, da_concepts, "annual")
        ppe_series = std.extract_metric(facts, ppe_concepts, "annual")

        if da_series.empty or ppe_series.empty:
            return pd.Series(dtype=float)

        da_vals = da_series.set_index("period_end")["value"].astype(float)
        ppe_vals = ppe_series.set_index("period_end")["value"].astype(float)

        # Align
        aligned = pd.DataFrame({"da": da_vals, "ppe": ppe_vals}).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)

        # Normalized D&A rate = avg(D&A / PPE)
        rates = aligned["da"] / aligned["ppe"].replace(0, float("nan"))
        avg_rate = rates.mean()

        # Apply to current PPE
        normalized = ppe_vals.tail(periods) * avg_rate
        return normalized.rename("normalized_da")


# ---------------------------------------------------------------------------
# Cross-Company Non-GAAP Benchmark
# ---------------------------------------------------------------------------


class CrossCompanyNonGAAPBenchmark:
    """
    Sector-wide comparison of GAAP vs non-GAAP adjustment practices.

    Identifies:
    - Which companies have the largest adjustments (highest non-GAAP inflation)
    - Industry-standard adjustments (what's normal vs aggressive for sector)
    - Historical adjustment trend: increasing = worsening quality
    - Outlier detection: adjustments > 2 std devs above sector mean
    """

    def __init__(
        self,
        parser: NonGAAPParser | None = None,
        scorer: AdjustmentQualityScorer | None = None,
        cache_path: str | None = None,
    ) -> None:
        self._parser = parser or NonGAAPParser()
        self._scorer = scorer or AdjustmentQualityScorer(self._parser)
        self._cache = FinancialsCache(db_path=cache_path)
        self._http = httpx.Client(headers=EDGAR_HEADERS, timeout=_TIMEOUT)

    def _get_ciks_for_sic(self, sic_code: str, limit: int = 50) -> list[str]:
        """Get CIKs for a SIC code from EDGAR."""
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "SIC": sic_code,
            "type": "10-K",
            "dateb": "",
            "owner": "include",
            "count": min(limit, 40),
            "output": "atom",
        }
        try:
            time.sleep(_RATE_DELAY)
            r = self._http.get(url, params=params)
            r.raise_for_status()
            ciks = re.findall(r"CIK=(\d+)", r.text)
            return list(dict.fromkeys(ciks))[:limit]
        except httpx.HTTPError as exc:
            logger.warning("sic_search_failed", sic=sic_code, error=str(exc))
            return []

    def _resolve_ticker_from_cik(self, cik: str) -> str | None:
        """Attempt reverse lookup: CIK → ticker via EDGAR company facts."""
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            time.sleep(_RATE_DELAY)
            resp = self._http.get(url)
            resp.raise_for_status()
            data = resp.json()
            tickers = data.get("tickers", [])
            return tickers[0] if tickers else None
        except Exception:
            return None

    def compare_sector(
        self,
        tickers: list[str] | None = None,
        sic_code: str | None = None,
        sector_name: str = "Unknown",
        limit: int = 20,
    ) -> SectorNonGAAPComparison:
        """
        Compare non-GAAP quality across companies in a sector.

        Parameters
        ----------
        tickers     : explicit list of tickers to compare
        sic_code    : EDGAR SIC code (alternative to tickers)
        sector_name : human-readable sector name
        limit       : max companies to include
        """
        if tickers is None and sic_code:
            ciks = self._get_ciks_for_sic(sic_code, limit=limit)
            tickers = []
            for cik in ciks:
                t = self._resolve_ticker_from_cik(cik)
                if t:
                    tickers.append(t)
                if len(tickers) >= limit:
                    break
        elif tickers is None:
            return SectorNonGAAPComparison(sector=sector_name)

        company_data = []
        all_adjustments: list[str] = []
        adjustment_pcts: list[float] = []
        quality_scores: dict[str, float] = {}

        for ticker in tickers[:limit]:
            try:
                reports = self._scorer.score_ticker(ticker, num_filings=4)
                if not reports:
                    continue

                latest = reports[-1]
                adj_pct = latest.adjustment_magnitude_pct
                score = latest.quality_score

                # Collect adjustment names for industry-standard analysis
                for adj in latest.adjustments:
                    all_adjustments.append(adj.name.lower())

                if adj_pct is not None:
                    adjustment_pcts.append(adj_pct)

                quality_scores[ticker] = score

                # Trend
                trend_data = self._scorer.get_trending_quality(reports)

                company_data.append({
                    "ticker": ticker,
                    "quality_score": score,
                    "quality_label": latest.quality_label,
                    "adjustment_magnitude_pct": adj_pct,
                    "red_flag_count": latest.red_flag_count,
                    "questionable_count": latest.questionable_adjustment_count,
                    "recurring_count": latest.recurring_adjustment_count,
                    "trend": trend_data.get("trend", "unknown"),
                    "flags": latest.flags[:3],  # top 3 flags
                })

            except Exception as exc:
                logger.warning("sector_compare_skip", ticker=ticker, error=str(exc))

        if not company_data:
            return SectorNonGAAPComparison(sector=sector_name)

        # Industry-standard adjustments: appear in >50% of companies
        from collections import Counter
        adj_counts = Counter(all_adjustments)
        total_companies = len(company_data)
        industry_standard = [
            adj for adj, count in adj_counts.most_common(10)
            if count / max(total_companies, 1) > 0.50
        ]

        # Average adjustment magnitude
        avg_adj = float(np.mean(adjustment_pcts)) if adjustment_pcts else None
        max_adj = float(max(adjustment_pcts)) if adjustment_pcts else None

        # Rank by quality
        sorted_by_quality = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        highest_quality = [t for t, _ in sorted_by_quality[:3]]
        lowest_quality = [t for t, _ in sorted_by_quality[-3:]]

        return SectorNonGAAPComparison(
            sector=sector_name,
            companies=sorted(company_data, key=lambda x: x["quality_score"], reverse=True),
            avg_adjustment_pct=round(avg_adj, 2) if avg_adj else None,
            max_adjustment_pct=round(max_adj, 2) if max_adj else None,
            industry_standard_adjustments=industry_standard,
            highest_quality_companies=highest_quality,
            lowest_quality_companies=lowest_quality,
        )

    def get_adjustment_trend(
        self,
        ticker: str,
        num_filings: int = 12,
    ) -> dict[str, Any]:
        """
        Track adjustment magnitude trend for a ticker over time.
        Increasing trend = worsening quality signal.
        """
        tables = self._parser.get_reconciliation_tables(ticker, num_filings=num_filings)

        if not tables:
            return {"ticker": ticker, "trend": "no_data", "data": []}

        tables_sorted = sorted(tables, key=lambda t: t.period)
        data = []
        magnitudes = []

        for table in tables_sorted:
            mag = table.adjustment_magnitude_pct
            if mag is not None:
                magnitudes.append(mag)
                data.append({
                    "period": table.period,
                    "filing_type": table.filing_type,
                    "adjustment_magnitude_pct": mag,
                    "total_adjustments": table.total_adjustments,
                    "gaap_value": table.gaap_value,
                    "non_gaap_value": table.non_gaap_value,
                    "num_adjustments": len(table.adjustments),
                })

        if len(magnitudes) >= 2:
            x = np.arange(len(magnitudes), dtype=float)
            slope = float(np.polyfit(x, magnitudes, 1)[0])
            trend = "worsening" if slope > 1 else "improving" if slope < -1 else "stable"
        else:
            slope = 0.0
            trend = "insufficient_data"

        return {
            "ticker": ticker,
            "trend": trend,
            "slope_per_period": round(slope, 2),
            "avg_magnitude": round(float(np.mean(magnitudes)), 2) if magnitudes else None,
            "latest_magnitude": magnitudes[-1] if magnitudes else None,
            "data": data,
        }

    def find_outliers(
        self,
        comparison: SectorNonGAAPComparison,
        z_score_threshold: float = 2.0,
    ) -> list[dict]:
        """
        Identify companies with adjustment magnitudes > z_score_threshold std devs
        above the sector mean. These are outlier adjusters.
        """
        magnitudes = [
            (c["ticker"], c["adjustment_magnitude_pct"])
            for c in comparison.companies
            if c.get("adjustment_magnitude_pct") is not None
        ]

        if len(magnitudes) < 3:
            return []

        values = [m for _, m in magnitudes]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))

        if std_val == 0:
            return []

        outliers = []
        for ticker, mag in magnitudes:
            z = (mag - mean_val) / std_val
            if z > z_score_threshold:
                outliers.append({
                    "ticker": ticker,
                    "adjustment_magnitude_pct": mag,
                    "z_score": round(z, 2),
                    "sector_mean": round(mean_val, 2),
                    "sector_std": round(std_val, 2),
                })

        return sorted(outliers, key=lambda x: x["z_score"], reverse=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _df_to_json_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    reset = df.reset_index()
    reset.columns = [str(c) for c in reset.columns]
    for col in reset.columns:
        if pd.api.types.is_datetime64_any_dtype(reset[col]):
            reset[col] = reset[col].dt.strftime("%Y-%m-%d")
    return reset.where(pd.notnull(reset), other=None).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_ng_parser: NonGAAPParser | None = None
_ng_scorer: AdjustmentQualityScorer | None = None
_normalized_engine: NormalizedEarningsEngine | None = None
_sector_benchmark: CrossCompanyNonGAAPBenchmark | None = None


def _get_parser() -> NonGAAPParser:
    global _ng_parser
    if _ng_parser is None:
        _ng_parser = NonGAAPParser()
    return _ng_parser


def _get_scorer() -> AdjustmentQualityScorer:
    global _ng_scorer
    if _ng_scorer is None:
        _ng_scorer = AdjustmentQualityScorer(_get_parser())
    return _ng_scorer


def _get_normalized_engine() -> NormalizedEarningsEngine:
    global _normalized_engine
    if _normalized_engine is None:
        _normalized_engine = NormalizedEarningsEngine(_get_parser())
    return _normalized_engine


def _get_sector_benchmark() -> CrossCompanyNonGAAPBenchmark:
    global _sector_benchmark
    if _sector_benchmark is None:
        _sector_benchmark = CrossCompanyNonGAAPBenchmark(_get_parser(), _get_scorer())
    return _sector_benchmark


def _ticker_404(ticker: str) -> str:
    try:
        return resolve_cik(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

nongaap_router = APIRouter(
    prefix="/api/financials/v2/nongaap",
    tags=["non-gaap-reconciliation"],
)


@nongaap_router.get("/{ticker}")
def get_reconciliation_tables(
    ticker: str,
    num_filings: int = Query(default=8, ge=1, le=20),
    form_types: str = Query(default="10-K,10-Q", description="Comma-separated form types"),
):
    """
    Extract non-GAAP reconciliation tables from recent EDGAR filings.

    Parses HTML from 10-K and 10-Q filings to extract:
    - GAAP starting value
    - All adjustment line items with classifications
    - Non-GAAP result
    - Period identification
    """
    _ticker_404(ticker)
    forms = [f.strip() for f in form_types.split(",")]

    try:
        tables = _get_parser().get_reconciliation_tables(
            ticker, num_filings=num_filings, form_types=forms
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker,
        "num_filings_searched": num_filings,
        "reconciliation_tables_found": len(tables),
        "adjustment_categories": {
            "recurring":    "Sometimes legitimate (stock comp, D&A on intangibles)",
            "questionable": "Needs justification (restructuring, deal costs)",
            "litigation":   "Moderate concern (watch for recurrence)",
            "red_flag":     "Almost never justified (operating cost exclusions)",
            "other":        "Unclassified",
        },
        "data": [t.model_dump() for t in tables],
    }


@nongaap_router.get("/{ticker}/quality")
def get_quality_analysis(
    ticker: str,
    num_filings: int = Query(default=8, ge=1, le=20),
    include_trend: bool = Query(default=True),
):
    """
    Score the quality of non-GAAP adjustments across recent filings.

    Scoring methodology:
    - Starts at 100 (no adjustments = pristine GAAP)
    - Deductions by category: recurring (-5), questionable (-15),
      litigation (-10), red flag (-30)
    - Extra -10 per item recurring for 3+ consecutive years
    - Extra -20/-35 for high/very high adjustment magnitude (>30%/>50%)
    - Final score: 0-100, labeled Excellent/Good/Fair/Poor/Red Flag
    """
    _ticker_404(ticker)
    try:
        reports = _get_scorer().score_ticker(ticker, num_filings=num_filings)
        trend = {}
        if include_trend and reports:
            trend = _get_scorer().get_trending_quality(reports)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Summary stats
    latest_score = reports[-1].quality_score if reports else None
    latest_label = reports[-1].quality_label if reports else "Unknown"
    total_red_flags = sum(r.red_flag_count for r in reports)

    return {
        "ticker": ticker,
        "latest_score": latest_score,
        "latest_label": latest_label,
        "total_red_flags_across_periods": total_red_flags,
        "score_legend": {
            "90-100": "Excellent",
            "75-89": "Good",
            "55-74": "Fair",
            "35-54": "Poor",
            "0-34": "Red Flag",
        },
        "trend": trend,
        "data": [r.model_dump() for r in reports],
    }


@nongaap_router.get("/{ticker}/normalized")
def get_normalized_earnings(
    ticker: str,
    years: int = Query(default=5, ge=2, le=10),
    include_series: bool = Query(default=True),
):
    """
    SENTINEL-normalized EPS: through-the-cycle earnings estimate.

    Returns three EPS variants:
    - GAAP EPS: reported per EDGAR
    - Reported Non-GAAP EPS: management's adjusted figure
    - SENTINEL Normalized EPS: 5-year avg margin × current revenue / shares

    This normalization strips cyclicality and acquisition D&A distortions.
    Use for valuation when either GAAP or management non-GAAP is distorted.
    """
    _ticker_404(ticker)
    try:
        normalized = _get_normalized_engine().compute_normalized_eps(ticker, years=years)
        series_data = []
        if include_series:
            series_df = _get_normalized_engine().compute_normalized_series(
                ticker, periods=years
            )
            series_data = _df_to_json_records(series_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker,
        "normalization_years": years,
        "method": (
            f"{years}-year average net margin applied to current revenue. "
            "Strips cyclicality and one-time items without management's adjustments."
        ),
        "current": normalized.model_dump(),
        "historical_series": series_data,
    }


@nongaap_router.get("/sector-comparison")
def get_sector_comparison(
    tickers: str = Query(description="Comma-separated list of tickers"),
    sector: str = Query(default="Unknown Sector"),
    sic_code: str | None = Query(default=None),
    include_outliers: bool = Query(default=True),
):
    """
    Cross-company non-GAAP quality comparison for a sector.

    Identifies:
    - Companies with highest vs lowest adjustment quality
    - Industry-standard adjustments (appear in >50% of companies)
    - Outlier adjusters (z-score > 2 vs sector mean)
    - Average and max adjustment magnitude for the sector
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    if not ticker_list and not sic_code:
        raise HTTPException(
            status_code=400,
            detail="Provide tickers (comma-separated) or sic_code."
        )

    try:
        comparison = _get_sector_benchmark().compare_sector(
            tickers=ticker_list or None,
            sic_code=sic_code,
            sector_name=sector,
            limit=30,
        )
        outliers = []
        if include_outliers:
            outliers = _get_sector_benchmark().find_outliers(comparison)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "sector": sector,
        "companies_analyzed": len(comparison.companies),
        "sector_avg_adjustment_pct": comparison.avg_adjustment_pct,
        "sector_max_adjustment_pct": comparison.max_adjustment_pct,
        "industry_standard_adjustments": comparison.industry_standard_adjustments,
        "highest_quality": comparison.highest_quality_companies,
        "lowest_quality": comparison.lowest_quality_companies,
        "outlier_adjusters": outliers,
        "data": comparison.companies,
    }


@nongaap_router.get("/{ticker}/trend")
def get_adjustment_trend(
    ticker: str,
    num_filings: int = Query(default=12, ge=2, le=24),
):
    """
    Track adjustment magnitude trend over time for a ticker.

    An increasing trend (worsening) indicates management is:
    - Expanding the definition of 'one-time' items
    - Adding new adjustment categories over time
    - Using non-GAAP to obscure deteriorating GAAP earnings

    A decreasing trend (improving) indicates:
    - Business becoming more capital-efficient
    - Fewer restructuring cycles needed
    - More alignment between GAAP and economic reality
    """
    _ticker_404(ticker)
    try:
        trend = _get_sector_benchmark().get_adjustment_trend(
            ticker, num_filings=num_filings
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ticker": ticker,
        "interpretation": {
            "worsening": "Adjustment magnitude increasing — management expanding non-GAAP definitions",
            "stable":    "Consistent adjustment level — monitor for absolute magnitude",
            "improving": "Declining adjustments — business becoming more cash-generative",
        },
        **trend,
    }


@nongaap_router.get("/{ticker}/full")
def get_full_nongaap_analysis(
    ticker: str,
    num_filings: int = Query(default=8, ge=2, le=20),
    normalization_years: int = Query(default=5, ge=2, le=10),
):
    """
    Full non-GAAP analysis in a single call.

    Combines: reconciliation tables, quality scoring, normalized EPS,
    and adjustment trend into one comprehensive response.
    """
    _ticker_404(ticker)
    try:
        tables = _get_parser().get_reconciliation_tables(ticker, num_filings=num_filings)
        quality_reports = _get_scorer().score_ticker(ticker, num_filings=num_filings)
        normalized = _get_normalized_engine().compute_normalized_eps(
            ticker, years=normalization_years
        )
        trend = _get_sector_benchmark().get_adjustment_trend(ticker, num_filings=num_filings)
        quality_trend = _get_scorer().get_trending_quality(quality_reports)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latest_score = quality_reports[-1].quality_score if quality_reports else None
    latest_label = quality_reports[-1].quality_label if quality_reports else "Unknown"

    return {
        "ticker": ticker,
        "summary": {
            "latest_quality_score": latest_score,
            "latest_quality_label": latest_label,
            "quality_trend": quality_trend.get("trend"),
            "adjustment_trend": trend.get("trend"),
            "gaap_eps": normalized.gaap_eps,
            "reported_nongaap_eps": normalized.reported_nongaap_eps,
            "sentinel_normalized_eps": normalized.sentinel_normalized_eps,
            "eps_gap_gaap_vs_nongaap": (
                round(
                    ((normalized.reported_nongaap_eps - normalized.gaap_eps) /
                     abs(normalized.gaap_eps) * 100), 2
                )
                if normalized.gaap_eps and normalized.reported_nongaap_eps
                and normalized.gaap_eps != 0
                else None
            ),
        },
        "reconciliation_tables": [t.model_dump() for t in tables],
        "quality_reports": [r.model_dump() for r in quality_reports],
        "normalized_eps": normalized.model_dump(),
        "magnitude_trend": trend,
        "quality_score_trend": quality_trend,
    }
