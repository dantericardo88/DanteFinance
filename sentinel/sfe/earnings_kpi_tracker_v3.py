"""
earnings_kpi_tracker_v3.py — KPI / Earnings Surprise Tracking (dim_019, target 9/10)

Provides production-grade earnings intelligence sourced entirely from free APIs:
  - EDGAR 8-K Item 2.02 (Results of Operations) full-text fetch + EPS/revenue extraction
  - EDGAR XBRL companyfacts API for YoY/sequential growth and accruals metrics
  - Finviz HTML scrape for analyst EPS estimates (enables real surprise calculation)
  - Earnings date calendar from SEC filing deadlines
  - EPS quality metrics: accruals ratio, cash EPS, one-time item frequency
  - Revenue quality: recognition-risk heuristic via deferred-revenue comparison
  - Beat/miss streak tracking
  - Guidance language detection from 8-K text
  - SQLite persistence layer
  - FastAPI router at /earnings/v3

No paid APIs required. Uses: requests, sqlite3, pandas, numpy, fastapi, re, bs4.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "SENTINEL financial-terminal/3.0 richard.porras@realempanada.com"
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_HTML_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

EFTS_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%228-K%22+%22results+of+operations%22"
    "&forms=8-K"
    "&dateRange=custom"
    "&startdt={startdt}"
    "&enddt={enddt}"
    "&entity={entity}"
)
EFTS_GENERIC_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22results+of+operations%22"
    "&forms=8-K"
    "&dateRange=custom"
    "&startdt={startdt}"
    "&enddt={enddt}"
)
EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
EDGAR_FACTS_BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_SUBMISSIONS_BASE = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FINVIZ_QUOTE_URL = "https://finviz.com/quote.ashx?t={ticker}"

_REQUEST_DELAY = 0.15  # seconds between EDGAR requests
_TIMEOUT = 30
_MAX_8K_TEXT_BYTES = 500_000  # cap on 8-K text fetch (500 KB)

# Surprise thresholds
STRONG_BEAT_THRESHOLD = 0.10
BEAT_THRESHOLD = 0.02
MISS_THRESHOLD = -0.02
STRONG_MISS_THRESHOLD = -0.10

# Filer-type deadlines (days after quarter-end for 10-Q)
FILER_DEADLINE_DAYS = {
    "large_accelerated": 40,
    "accelerated": 40,
    "non_accelerated": 45,
}

DB_PATH = Path(__file__).parent.parent / "data" / "earnings_kpi_v3.db"

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS quarterly_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    cik             TEXT,
    period_label    TEXT NOT NULL,       -- e.g. "2025-Q1"
    period_end      TEXT,               -- ISO date
    actual_eps      REAL,
    revenue         REAL,               -- millions
    net_income      REAL,               -- millions
    cfo             REAL,               -- millions
    diluted_shares  REAL,               -- millions
    source          TEXT DEFAULT 'edgar',
    filed_date      TEXT,
    accession_no    TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, period_label)
);

CREATE TABLE IF NOT EXISTS surprise_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    period_label    TEXT NOT NULL,
    actual_eps      REAL,
    estimate_eps    REAL,               -- from finviz or prior-year same period
    surprise_pct    REAL,
    surprise_cat    TEXT,               -- STRONG_BEAT / BEAT / IN_LINE / MISS / STRONG_MISS
    estimate_source TEXT,               -- 'finviz' | 'yoy_history'
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, period_label)
);

CREATE TABLE IF NOT EXISTS guidance_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    period_label    TEXT NOT NULL,
    guidance_action TEXT,               -- raises / lowers / reaffirms / initiates / withdraws
    raw_excerpt     TEXT,
    filed_date      TEXT,
    accession_no    TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS quality_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    fiscal_year     TEXT NOT NULL,
    accruals_ratio  REAL,
    sloan_ratio     REAL,
    cash_eps        REAL,
    gaap_eps        REAL,
    cash_eps_gap    REAL,
    one_time_count  INTEGER DEFAULT 0,
    deferred_rev_risk INTEGER DEFAULT 0,
    quality_flag    TEXT,               -- HIGH / MODERATE / LOW / VERY_LOW
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, fiscal_year)
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    cik             TEXT,
    quarter_end     TEXT NOT NULL,      -- ISO date
    filing_deadline TEXT NOT NULL,      -- ISO date (40 or 45 days after quarter end)
    filer_type      TEXT DEFAULT 'non_accelerated',
    confirmed_date  TEXT,               -- actual filed date if known
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, quarter_end)
);
"""

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QuarterlyResult(BaseModel):
    ticker: str
    period_label: str
    period_end: Optional[str] = None
    actual_eps: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    cfo: Optional[float] = None
    diluted_shares: Optional[float] = None
    filed_date: Optional[str] = None
    accession_no: Optional[str] = None


class SurpriseResult(BaseModel):
    ticker: str
    period_label: str
    actual_eps: Optional[float] = None
    estimate_eps: Optional[float] = None
    surprise_pct: Optional[float] = None
    surprise_cat: str = "UNKNOWN"
    estimate_source: str = "none"
    beat_streak: int = 0
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class GuidanceEntry(BaseModel):
    ticker: str
    period_label: str
    guidance_action: str
    raw_excerpt: str
    filed_date: Optional[str] = None


class QualityMetrics(BaseModel):
    ticker: str
    fiscal_year: str
    accruals_ratio: Optional[float] = None
    sloan_ratio: Optional[float] = None
    cash_eps: Optional[float] = None
    gaap_eps: Optional[float] = None
    cash_eps_gap: Optional[float] = None
    one_time_count: int = 0
    deferred_rev_risk: int = 0
    quality_flag: str = "MODERATE"


class CalendarEntry(BaseModel):
    ticker: str
    cik: Optional[str] = None
    quarter_end: str
    filing_deadline: str
    filer_type: str = "non_accelerated"
    confirmed_date: Optional[str] = None
    days_until: Optional[int] = None


class BatchSurpriseItem(BaseModel):
    ticker: str
    latest_period: Optional[str] = None
    surprise_pct: Optional[float] = None
    surprise_cat: str = "UNKNOWN"
    beat_streak: int = 0


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _get_db_path() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def _db_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    for stmt in _DDL.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s)
    conn.commit()


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------

_CIK_CACHE: dict[str, str] = {}


def resolve_cik(ticker: str) -> Optional[str]:
    """Map ticker to zero-padded 10-digit CIK via SEC company_tickers.json."""
    ticker_upper = ticker.upper()
    if ticker_upper in _CIK_CACHE:
        return _CIK_CACHE[ticker_upper]
    try:
        resp = requests.get(EDGAR_TICKERS_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                _CIK_CACHE[ticker_upper] = cik
                return cik
    except Exception as exc:
        logger.warning("CIK resolution failed for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# EDGAR 8-K Item 2.02 fetcher
# ---------------------------------------------------------------------------

class EightKFiling(BaseModel):
    accession_no: str
    filed_date: str
    entity_name: str
    cik: str
    period_of_report: Optional[str] = None
    full_text_url: Optional[str] = None


def _accession_to_url(cik: str, accession_no: str) -> str:
    """Build EDGAR filing index URL from CIK and accession number."""
    acc_clean = accession_no.replace("-", "")
    return f"{EDGAR_ARCHIVES_BASE}/{int(cik)}/{acc_clean}/{accession_no}-index.htm"


def fetch_8k_filings(
    ticker: str,
    start_date: str,
    end_date: str,
    cik: Optional[str] = None,
) -> list[EightKFiling]:
    """
    Fetch list of 8-K filings tagged as 'Results of Operations' (Item 2.02)
    for a given ticker/CIK in the date range.

    Uses EFTS search-index endpoint — no API key needed.
    """
    if cik is None:
        cik = resolve_cik(ticker)
    entity_param = ticker.upper() if not cik else ""

    # Try entity-name search first if we lack CIK
    url = EFTS_SEARCH_URL.format(startdt=start_date, enddt=end_date, entity=entity_param)
    if cik:
        # Use CIK-scoped search for precision
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=%22results+of+operations%22"
            f"&forms=8-K"
            f"&dateRange=custom"
            f"&startdt={start_date}"
            f"&enddt={end_date}"
            f"&entity={ticker}"
        )

    time.sleep(_REQUEST_DELAY)
    filings: list[EightKFiling] = []
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("EFTS returned %s for %s", resp.status_code, ticker)
            return filings
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            filing_cik = src.get("entity_id", cik or "")
            if not filing_cik:
                continue
            # Filter by CIK if we resolved one
            if cik and str(int(filing_cik)).zfill(10) != cik:
                continue
            acc = src.get("accession_no", "")
            if not acc:
                continue
            filing_cik_padded = str(int(filing_cik)).zfill(10)
            filings.append(
                EightKFiling(
                    accession_no=acc,
                    filed_date=src.get("file_date", ""),
                    entity_name=src.get("entity_name", ticker),
                    cik=filing_cik_padded,
                    period_of_report=src.get("period_of_report"),
                )
            )
    except Exception as exc:
        logger.error("8-K fetch failed for %s: %s", ticker, exc)
    return filings


def fetch_8k_full_text(filing: EightKFiling) -> str:
    """
    Download and return full text of an 8-K filing from EDGAR archives.
    Prefers .htm document; falls back to index scan.
    """
    cik_int = int(filing.cik)
    acc_clean = filing.accession_no.replace("-", "")
    index_url = f"{EDGAR_ARCHIVES_BASE}/{cik_int}/{acc_clean}/{filing.accession_no}-index.htm"
    time.sleep(_REQUEST_DELAY)
    try:
        resp = requests.get(index_url, headers={**_HEADERS, "Accept": "text/html"}, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        # Find the primary document link (8-K body)
        doc_url: Optional[str] = None
        for a in soup.find_all("a", href=True):
            href: str = a["href"]
            if href.lower().endswith(".htm") and "index" not in href.lower():
                doc_url = f"https://www.sec.gov{href}" if href.startswith("/") else href
                break
        if not doc_url:
            return ""
        time.sleep(_REQUEST_DELAY)
        doc_resp = requests.get(
            doc_url,
            headers={**_HEADERS, "Accept": "text/html"},
            timeout=_TIMEOUT,
            stream=True,
        )
        if doc_resp.status_code != 200:
            return ""
        chunks = []
        total = 0
        for chunk in doc_resp.iter_content(chunk_size=32_768):
            chunks.append(chunk)
            total += len(chunk)
            if total >= _MAX_8K_TEXT_BYTES:
                break
        raw_bytes = b"".join(chunks)
        html_text = raw_bytes.decode("utf-8", errors="replace")
        # Strip HTML tags for regex extraction
        text_soup = BeautifulSoup(html_text, "html.parser")
        return text_soup.get_text(separator=" ", strip=True)
    except Exception as exc:
        logger.warning("Failed to fetch 8-K text for %s/%s: %s", filing.cik, filing.accession_no, exc)
        return ""


# ---------------------------------------------------------------------------
# EPS extractor
# ---------------------------------------------------------------------------

# All common EPS presentation patterns in 8-K texts
_EPS_PATTERNS: list[re.Pattern] = [
    # "diluted earnings per share of $1.23"
    re.compile(
        r"diluted\s+(?:net\s+)?(?:earnings|income|loss)\s+per\s+(?:common\s+)?share[^$\d]{0,30}\$\s*([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # "EPS of $X.XX" or "EPS: $X.XX"
    re.compile(r"\bEPS\b[^$\d]{0,20}\$\s*([\d,]+\.?\d*)", re.IGNORECASE),
    # "earnings per share" then dollar amount
    re.compile(
        r"earnings\s+per\s+(?:diluted\s+)?share[^$\d]{0,40}\$\s*([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # "per diluted share" — often appears after amount: "$1.23 per diluted share"
    re.compile(r"\$\s*([\d,]+\.?\d*)\s+per\s+diluted\s+share", re.IGNORECASE),
    # "(loss) per diluted share" with negative: "$(0.45) per diluted share"
    re.compile(r"\$\s*\(\s*([\d,]+\.?\d*)\s*\)\s+per\s+diluted\s+share", re.IGNORECASE),
    # "diluted EPS of X.XX" (no dollar sign)
    re.compile(r"diluted\s+EPS\s+of\s+([\d,]+\.?\d*)", re.IGNORECASE),
]

_NEGATIVE_EPS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\$\s*\(\s*([\d,]+\.?\d*)\s*\)\s+per\s+diluted\s+share", re.IGNORECASE),
    re.compile(r"loss[^$\d]{0,20}\$\s*([\d,]+\.?\d*)\s+per\s+(?:diluted\s+)?share", re.IGNORECASE),
]


def extract_eps_from_text(text: str) -> Optional[float]:
    """
    Extract diluted EPS from 8-K text. Returns float or None.
    Handles both positive and negative (parenthetical) formats.
    """
    # Check negative patterns first
    for pat in _NEGATIVE_EPS_PATTERNS:
        m = pat.search(text)
        if m:
            val_str = m.group(1).replace(",", "")
            try:
                return -float(val_str)
            except ValueError:
                continue

    # Positive EPS patterns — take first plausible match
    for pat in _EPS_PATTERNS:
        m = pat.search(text)
        if m:
            val_str = m.group(1).replace(",", "")
            try:
                val = float(val_str)
                # Sanity: EPS shouldn't exceed $1,000 or be zero exactly from regex
                if 0 < val < 1000:
                    return val
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Revenue extractor
# ---------------------------------------------------------------------------

_REVENUE_PATTERNS: list[tuple[re.Pattern, float]] = [
    # "$X.X billion" / "$X billion"
    (
        re.compile(
            r"(?:revenue|net\s+sales|total\s+revenues?)\s+(?:of\s+|were\s+|was\s+)?\$\s*([\d,]+\.?\d*)\s+billion",
            re.IGNORECASE,
        ),
        1000.0,  # scale to millions
    ),
    # "$X.X million"
    (
        re.compile(
            r"(?:revenue|net\s+sales|total\s+revenues?)\s+(?:of\s+|were\s+|was\s+)?\$\s*([\d,]+\.?\d*)\s+million",
            re.IGNORECASE,
        ),
        1.0,
    ),
    # Reverse: "$X.X billion in revenue/sales"
    (
        re.compile(r"\$\s*([\d,]+\.?\d*)\s+billion\s+in\s+(?:revenue|net\s+sales|total\s+revenues?)", re.IGNORECASE),
        1000.0,
    ),
    (
        re.compile(r"\$\s*([\d,]+\.?\d*)\s+million\s+in\s+(?:revenue|net\s+sales|total\s+revenues?)", re.IGNORECASE),
        1.0,
    ),
]


def extract_revenue_from_text(text: str) -> Optional[float]:
    """
    Extract revenue from 8-K text. Returns millions as float or None.
    """
    for pat, scale in _REVENUE_PATTERNS:
        m = pat.search(text)
        if m:
            val_str = m.group(1).replace(",", "")
            try:
                val = float(val_str) * scale
                if val > 0:
                    return val
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# One-time item counter
# ---------------------------------------------------------------------------

_ONE_TIME_KEYWORDS = [
    r"restructuring\s+charge",
    r"impairment\s+charge",
    r"goodwill\s+impairment",
    r"asset\s+write(?:-|\s)?(?:down|off)",
    r"severance\s+(?:cost|charge)",
    r"litigation\s+settlement",
    r"legal\s+settlement",
    r"gain\s+on\s+(?:sale|disposal)",
    r"loss\s+on\s+(?:sale|disposal|extinguishment)",
    r"debt\s+extinguishment",
    r"acquisition[- ]related\s+cost",
    r"merger[- ]related\s+cost",
    r"integration\s+(?:cost|charge)",
]

_ONE_TIME_RE = re.compile("|".join(_ONE_TIME_KEYWORDS), re.IGNORECASE)


def count_one_time_items(text: str) -> int:
    """Count distinct one-time / non-recurring item mentions in 8-K text."""
    return len(_ONE_TIME_RE.findall(text))


# ---------------------------------------------------------------------------
# Guidance language detector
# ---------------------------------------------------------------------------

_GUIDANCE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("raises", re.compile(r"\b(?:raise[sd]?|increase[sd]?|raise[sd]?\s+(?:its\s+)?(?:full[- ]year\s+)?(?:guidance|outlook|forecast))\b", re.IGNORECASE)),
    ("lowers", re.compile(r"\b(?:lower[sd]?|reduce[sd]?|decrease[sd]?|lower[sd]?\s+(?:its\s+)?(?:full[- ]year\s+)?(?:guidance|outlook|forecast))\b", re.IGNORECASE)),
    ("reaffirms", re.compile(r"\b(?:reaffirm[sd]?|reiterate[sd]?|maintain[sd]?|confirm[sd]?)\s+(?:its\s+)?(?:full[- ]year\s+)?(?:guidance|outlook|forecast)\b", re.IGNORECASE)),
    ("initiates", re.compile(r"\b(?:initiate[sd]?|provide[sd]?|establish[esd]?)\s+(?:initial\s+)?(?:full[- ]year\s+)?(?:guidance|outlook|forecast)\b", re.IGNORECASE)),
    ("withdraws", re.compile(r"\b(?:withdraw[sn]?|suspend[sd]?|withdraw[sn]?\s+(?:its\s+)?(?:full[- ]year\s+)?(?:guidance|outlook|forecast))\b", re.IGNORECASE)),
]

# Context window around guidance match for excerpt
_GUIDANCE_CONTEXT_RE = re.compile(
    r"(?:guidance|outlook|forecast|expect|anticipate|project)[^\.\n]{0,200}",
    re.IGNORECASE,
)


def extract_guidance_action(text: str) -> tuple[str, str]:
    """
    Detect guidance language in 8-K text.
    Returns (action, excerpt) where action is one of:
      raises / lowers / reaffirms / initiates / withdraws / none_found
    """
    for action, pat in _GUIDANCE_PATTERNS:
        if pat.search(text):
            # Extract first context snippet
            ctx_match = _GUIDANCE_CONTEXT_RE.search(text)
            excerpt = ctx_match.group(0).strip()[:250] if ctx_match else ""
            return action, excerpt
    return "none_found", ""


# ---------------------------------------------------------------------------
# Finviz EPS estimate scraper
# ---------------------------------------------------------------------------


def fetch_finviz_eps_estimate(ticker: str) -> Optional[float]:
    """
    Scrape Finviz quote page for 'EPS next Q' estimate.
    Returns float or None. Free, no API key needed.
    """
    url = FINVIZ_QUOTE_URL.format(ticker=ticker.upper())
    try:
        time.sleep(0.5)  # gentle rate limit for finviz
        resp = requests.get(url, headers=_HTML_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("Finviz returned %s for %s", resp.status_code, ticker)
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # Finviz table: look for "EPS next Q" label and sibling value
        cells = soup.find_all("td")
        for i, cell in enumerate(cells):
            if cell.get_text(strip=True) == "EPS next Q":
                if i + 1 < len(cells):
                    val_text = cells[i + 1].get_text(strip=True)
                    val_text = val_text.replace(",", "")
                    try:
                        return float(val_text)
                    except ValueError:
                        return None
    except Exception as exc:
        logger.warning("Finviz scrape failed for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# EDGAR XBRL companyfacts helpers
# ---------------------------------------------------------------------------


def fetch_xbrl_facts(cik: str) -> dict:
    """Fetch EDGAR companyfacts XBRL JSON for a CIK. Returns raw dict or {}."""
    url = EDGAR_FACTS_BASE.format(cik=cik)
    time.sleep(_REQUEST_DELAY)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return {}
        return resp.json()
    except Exception as exc:
        logger.warning("XBRL facts fetch failed for CIK %s: %s", cik, exc)
        return {}


def _extract_concept(facts: dict, namespace: str, concept: str) -> list[dict]:
    """
    Pull all period-value entries for a given US-GAAP concept from XBRL facts.
    Returns list of dicts with keys: end, val, form, accn.
    """
    try:
        units_block = facts["facts"][namespace][concept]["units"]
        # Most numeric facts use USD; shares use "shares"
        for unit_key, entries in units_block.items():
            return [e for e in entries if e.get("form") in ("10-Q", "10-K", "10-K405")]
    except (KeyError, TypeError):
        return []


def _quarterly_series(entries: list[dict]) -> pd.DataFrame:
    """
    Convert XBRL fact entries to quarterly DataFrame.
    Keeps only point-in-time quarterly observations (end - start ~ 90 days).
    """
    rows = []
    for e in entries:
        start = e.get("start")
        end = e.get("end")
        val = e.get("val")
        if not end or val is None:
            continue
        if start:
            try:
                d_start = datetime.strptime(start, "%Y-%m-%d")
                d_end = datetime.strptime(end, "%Y-%m-%d")
                days = (d_end - d_start).days
                if not (70 <= days <= 110):
                    continue
            except ValueError:
                continue
        rows.append({"end": end, "val": float(val), "accn": e.get("accn", "")})
    if not rows:
        return pd.DataFrame(columns=["end", "val", "accn"])
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values("end").drop_duplicates("end", keep="last").reset_index(drop=True)
    return df


def _annual_series(entries: list[dict]) -> pd.DataFrame:
    """
    Convert XBRL fact entries to annual DataFrame (10-K filings only).
    Keeps entries where end - start ~ 365 days.
    """
    rows = []
    for e in entries:
        if e.get("form") not in ("10-K", "10-K405"):
            continue
        start = e.get("start")
        end = e.get("end")
        val = e.get("val")
        if not end or val is None:
            continue
        if start:
            try:
                d_start = datetime.strptime(start, "%Y-%m-%d")
                d_end = datetime.strptime(end, "%Y-%m-%d")
                days = (d_end - d_start).days
                if not (340 <= days <= 390):
                    continue
            except ValueError:
                continue
        rows.append({"end": end, "val": float(val)})
    if not rows:
        return pd.DataFrame(columns=["end", "val"])
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values("end").drop_duplicates("end", keep="last").reset_index(drop=True)
    return df


def build_xbrl_quarterly_profile(cik: str) -> dict[str, pd.DataFrame]:
    """
    Fetch XBRL facts and return a dict of concept -> quarterly DataFrame.
    Covers concepts needed for accruals, revenue growth, and EPS quality.
    """
    facts = fetch_xbrl_facts(cik)
    if not facts:
        return {}

    concepts = {
        "net_income": ("us-gaap", "NetIncomeLoss"),
        "cfo": ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        "total_assets": ("us-gaap", "Assets"),
        "revenue": ("us-gaap", "Revenues"),
        "revenue_alt": ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        "deferred_revenue": ("us-gaap", "DeferredRevenueCurrent"),
        "diluted_shares": ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        "eps_diluted": ("us-gaap", "EarningsPerShareDiluted"),
    }

    result: dict[str, pd.DataFrame] = {}
    for key, (ns, concept) in concepts.items():
        entries = _extract_concept(facts, ns, concept)
        result[key] = _quarterly_series(entries)

    # For total_assets use annual entries (balance sheet)
    assets_all = _extract_concept(facts, "us-gaap", "Assets")
    result["total_assets_annual"] = _annual_series(assets_all)

    # CFO from annual for accruals ratio
    cfo_all = _extract_concept(facts, "us-gaap", "NetCashProvidedByUsedInOperatingActivities")
    result["cfo_annual"] = _annual_series(cfo_all)

    ni_all = _extract_concept(facts, "us-gaap", "NetIncomeLoss")
    result["net_income_annual"] = _annual_series(ni_all)

    return result


# ---------------------------------------------------------------------------
# Accruals / EPS quality computation
# ---------------------------------------------------------------------------


def compute_accruals_metrics(profile: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Compute annual accruals ratio and Sloan ratio from XBRL data.
    Accruals ratio = (Net Income - CFO) / avg(Total Assets)
    Sloan ratio    = (Net Income - CFO) / avg(Total Assets)  [same; Sloan uses BS accruals variant]
    Cash EPS gap   = (CFO / diluted_shares) - EPS_GAAP
    Returns list of annual records.
    """
    ni_df = profile.get("net_income_annual", pd.DataFrame())
    cfo_df = profile.get("cfo_annual", pd.DataFrame())
    assets_df = profile.get("total_assets_annual", pd.DataFrame())

    if ni_df.empty or cfo_df.empty or assets_df.empty:
        return []

    # Merge on year (nearest end date)
    ni_df = ni_df.rename(columns={"val": "net_income"})
    cfo_df = cfo_df.rename(columns={"val": "cfo"})
    assets_df = assets_df.rename(columns={"val": "assets"})

    merged = pd.merge_asof(
        ni_df.sort_values("end"),
        cfo_df.sort_values("end"),
        on="end",
        tolerance=pd.Timedelta("45d"),
        direction="nearest",
    )
    merged = pd.merge_asof(
        merged.sort_values("end"),
        assets_df.sort_values("end"),
        on="end",
        tolerance=pd.Timedelta("45d"),
        direction="nearest",
    )
    merged = merged.dropna(subset=["net_income", "cfo", "assets"])
    if merged.empty:
        return []

    records = []
    for i, row in merged.iterrows():
        avg_assets = row["assets"]
        if i > 0:
            prev_assets = merged.iloc[i - 1]["assets"]
            avg_assets = (row["assets"] + prev_assets) / 2

        if avg_assets == 0:
            continue

        net_income = row["net_income"] / 1e6  # to millions
        cfo = row["cfo"] / 1e6
        assets = row["assets"] / 1e6
        avg_assets_m = avg_assets / 1e6

        accruals = net_income - cfo
        accruals_ratio = accruals / avg_assets_m if avg_assets_m else None
        sloan_ratio = accruals_ratio  # simplified; full Sloan uses ΔNOA

        quality_flag = "HIGH"
        if accruals_ratio is not None:
            if abs(accruals_ratio) > 0.10:
                quality_flag = "VERY_LOW"
            elif abs(accruals_ratio) > 0.05:
                quality_flag = "LOW"
            elif abs(accruals_ratio) > 0.02:
                quality_flag = "MODERATE"

        records.append({
            "fiscal_year": row["end"].strftime("%Y"),
            "net_income": net_income,
            "cfo": cfo,
            "assets": assets,
            "accruals_ratio": accruals_ratio,
            "sloan_ratio": sloan_ratio,
            "quality_flag": quality_flag,
        })
    return records


def compute_cash_eps(profile: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Compute Cash EPS vs GAAP EPS for each quarter.
    Cash EPS = (CFO / diluted_shares)  in dollars
    Gap      = Cash EPS - GAAP EPS  (positive = cash earnings > reported)
    """
    cfo_df = profile.get("cfo", pd.DataFrame())
    shares_df = profile.get("diluted_shares", pd.DataFrame())
    eps_df = profile.get("eps_diluted", pd.DataFrame())

    if cfo_df.empty or shares_df.empty:
        return []

    cfo_df = cfo_df.rename(columns={"val": "cfo"})
    shares_df = shares_df.rename(columns={"val": "shares"})
    eps_df = eps_df.rename(columns={"val": "gaap_eps"}) if not eps_df.empty else pd.DataFrame()

    merged = pd.merge_asof(
        cfo_df.sort_values("end"),
        shares_df.sort_values("end"),
        on="end",
        tolerance=pd.Timedelta("15d"),
        direction="nearest",
    )
    if not eps_df.empty:
        merged = pd.merge_asof(
            merged.sort_values("end"),
            eps_df.sort_values("end"),
            on="end",
            tolerance=pd.Timedelta("15d"),
            direction="nearest",
        )
    merged = merged.dropna(subset=["cfo", "shares"])

    records = []
    for _, row in merged.iterrows():
        shares = row["shares"]
        if shares <= 0:
            continue
        cash_eps = (row["cfo"] / shares)
        gaap_eps = row.get("gaap_eps", None)
        gap = (cash_eps - gaap_eps) if gaap_eps is not None else None
        records.append({
            "end": row["end"].strftime("%Y-%m-%d"),
            "cash_eps": round(cash_eps, 4),
            "gaap_eps": gaap_eps,
            "gap": round(gap, 4) if gap is not None else None,
        })
    return records


def check_revenue_recognition_risk(profile: dict[str, pd.DataFrame]) -> bool:
    """
    Returns True if deferred revenue is growing faster than revenue
    (potential recognition-timing risk — revenue booked before cash).
    Uses last 4 quarters of data.
    """
    rev_df = profile.get("revenue", pd.DataFrame())
    if rev_df.empty:
        rev_df = profile.get("revenue_alt", pd.DataFrame())
    def_df = profile.get("deferred_revenue", pd.DataFrame())

    if rev_df.empty or def_df.empty or len(rev_df) < 2 or len(def_df) < 2:
        return False

    rev_growth = (rev_df.iloc[-1]["val"] - rev_df.iloc[-2]["val"]) / max(abs(rev_df.iloc[-2]["val"]), 1)
    def_growth = (def_df.iloc[-1]["val"] - def_df.iloc[-2]["val"]) / max(abs(def_df.iloc[-2]["val"]), 1)

    return bool(def_growth > rev_growth and def_growth > 0.05)


# ---------------------------------------------------------------------------
# YoY growth from XBRL
# ---------------------------------------------------------------------------


def compute_yoy_growth(profile: dict[str, pd.DataFrame], concept: str = "revenue") -> list[dict]:
    """
    Compute YoY quarterly growth for a given concept.
    Returns list of {period, yoy_pct} records (most recent last).
    """
    df = profile.get(concept, pd.DataFrame())
    if df.empty or len(df) < 5:
        return []

    records = []
    for i in range(4, len(df)):
        current = df.iloc[i]
        prior_year = df.iloc[i - 4]
        if prior_year["val"] == 0:
            continue
        yoy = (current["val"] - prior_year["val"]) / abs(prior_year["val"])
        records.append({
            "period": current["end"].strftime("%Y-%m-%d"),
            "value": current["val"],
            "prior_year_value": prior_year["val"],
            "yoy_pct": round(yoy * 100, 2),
        })
    return records


# ---------------------------------------------------------------------------
# Surprise categorization
# ---------------------------------------------------------------------------


def categorize_surprise(surprise_pct: float) -> str:
    if surprise_pct > STRONG_BEAT_THRESHOLD:
        return "STRONG_BEAT"
    elif surprise_pct > BEAT_THRESHOLD:
        return "BEAT"
    elif surprise_pct >= MISS_THRESHOLD:
        return "IN_LINE"
    elif surprise_pct >= STRONG_MISS_THRESHOLD:
        return "MISS"
    else:
        return "STRONG_MISS"


# ---------------------------------------------------------------------------
# Beat/miss streak computation
# ---------------------------------------------------------------------------


def compute_beat_streak(ticker: str) -> int:
    """
    Compute current consecutive beat streak (most recent quarters).
    Positive = beats, negative = misses.
    """
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT surprise_pct, surprise_cat
            FROM surprise_history
            WHERE ticker = ?
            ORDER BY period_label DESC
            LIMIT 8
            """,
            (ticker.upper(),),
        ).fetchall()

    if not rows:
        return 0

    streak = 0
    direction: Optional[bool] = None

    for row in rows:
        cat = row["surprise_cat"]
        is_beat = cat in ("BEAT", "STRONG_BEAT")
        is_miss = cat in ("MISS", "STRONG_MISS")

        if direction is None:
            if is_beat:
                direction = True
                streak = 1
            elif is_miss:
                direction = False
                streak = -1
            else:
                break  # IN_LINE breaks streak
        else:
            if direction and is_beat:
                streak += 1
            elif not direction and is_miss:
                streak -= 1
            else:
                break

    return streak


# ---------------------------------------------------------------------------
# Earnings calendar
# ---------------------------------------------------------------------------


def _quarter_end_dates(year: int) -> list[date]:
    """Return standard calendar quarter-end dates for a year."""
    return [
        date(year, 3, 31),
        date(year, 6, 30),
        date(year, 9, 30),
        date(year, 12, 31),
    ]


def build_earnings_calendar(
    ticker: str,
    cik: Optional[str] = None,
    filer_type: str = "non_accelerated",
    lookahead_quarters: int = 4,
) -> list[CalendarEntry]:
    """
    Compute prospective earnings calendar entries based on 10-Q filing deadlines.
    Uses SEC mandated days: large_accelerated / accelerated = 40, non_accelerated = 45.
    """
    if cik is None:
        cik = resolve_cik(ticker)

    deadline_days = FILER_DEADLINE_DAYS.get(filer_type, 45)
    today = date.today()
    entries: list[CalendarEntry] = []

    # Determine last 4 quarter-ends starting from current quarter
    current_year = today.year
    all_quarter_ends: list[date] = []
    for yr in range(current_year - 1, current_year + 2):
        all_quarter_ends.extend(_quarter_end_dates(yr))

    # Keep quarter-ends that are past (filed) or upcoming
    relevant = [qe for qe in all_quarter_ends if qe <= today + timedelta(days=90)]
    relevant = relevant[-lookahead_quarters * 2:]  # last N * 2 to ensure coverage

    for qe in relevant[-lookahead_quarters:]:
        deadline = qe + timedelta(days=deadline_days)
        days_until = (deadline - today).days
        entry = CalendarEntry(
            ticker=ticker.upper(),
            cik=cik,
            quarter_end=qe.isoformat(),
            filing_deadline=deadline.isoformat(),
            filer_type=filer_type,
            days_until=days_until if days_until >= 0 else None,
        )
        entries.append(entry)

        # Persist to DB
        with _db_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO earnings_calendar
                    (ticker, cik, quarter_end, filing_deadline, filer_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ticker.upper(), cik, qe.isoformat(), deadline.isoformat(), filer_type),
            )
    return entries


# ---------------------------------------------------------------------------
# Main orchestrator: full earnings surprise for a ticker
# ---------------------------------------------------------------------------


class EarningsKPITrackerV3:
    """
    Main entry point for earnings KPI tracking.
    Coordinates 8-K fetching, EPS/revenue extraction, XBRL data, Finviz estimates,
    surprise calculation, and SQLite persistence.
    """

    def __init__(self) -> None:
        _get_db_path()  # ensure DB directory exists

    def run_ticker(self, ticker: str, lookback_days: int = 365) -> SurpriseResult:
        """
        Full pipeline for a single ticker:
          1. Resolve CIK
          2. Fetch recent 8-K Item 2.02 filings
          3. Extract EPS + revenue from most recent filing text
          4. Fetch Finviz estimate for surprise calculation
          5. Fall back to YoY comparison if no Finviz estimate
          6. Persist results
          7. Return SurpriseResult
        """
        ticker = ticker.upper()
        cik = resolve_cik(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker: {ticker}")

        end_dt = date.today()
        start_dt = end_dt - timedelta(days=lookback_days)

        filings = fetch_8k_filings(
            ticker,
            start_date=start_dt.isoformat(),
            end_date=end_dt.isoformat(),
            cik=cik,
        )

        if not filings:
            logger.info("No 8-K Item 2.02 filings found for %s in last %d days", ticker, lookback_days)
            return SurpriseResult(ticker=ticker, period_label="N/A", surprise_cat="UNKNOWN")

        # Use most recent filing
        latest = sorted(filings, key=lambda f: f.filed_date, reverse=True)[0]
        period_label = self._derive_period_label(latest)

        # Extract text
        text = fetch_8k_full_text(latest)
        actual_eps = extract_eps_from_text(text)
        revenue = extract_revenue_from_text(text)
        one_time_count = count_one_time_items(text)
        guidance_action, guidance_excerpt = extract_guidance_action(text)

        # Persist guidance
        if guidance_action != "none_found":
            self._save_guidance(ticker, period_label, guidance_action, guidance_excerpt, latest)

        # Get analyst estimate from Finviz
        estimate_eps = fetch_finviz_eps_estimate(ticker)
        estimate_source = "finviz" if estimate_eps is not None else "none"

        # Fall back: YoY comparison from XBRL
        if estimate_eps is None and actual_eps is not None:
            estimate_eps = self._get_yoy_eps_estimate(cik, period_label)
            if estimate_eps is not None:
                estimate_source = "yoy_history"

        # Compute surprise
        surprise_pct: Optional[float] = None
        surprise_cat = "UNKNOWN"
        if actual_eps is not None and estimate_eps is not None and estimate_eps != 0:
            surprise_pct = (actual_eps - estimate_eps) / abs(estimate_eps)
            surprise_cat = categorize_surprise(surprise_pct)

        # Persist quarterly result
        self._save_quarterly_result(ticker, cik, period_label, latest, actual_eps, revenue)

        # Persist surprise
        self._save_surprise(ticker, period_label, actual_eps, estimate_eps, surprise_pct, surprise_cat, estimate_source)

        # Compute beat streak
        streak = compute_beat_streak(ticker)

        # XBRL quality metrics (run in background, best-effort)
        try:
            self._run_quality_metrics(ticker, cik, one_time_count, revenue)
        except Exception as exc:
            logger.warning("Quality metrics failed for %s: %s", ticker, exc)

        return SurpriseResult(
            ticker=ticker,
            period_label=period_label,
            actual_eps=actual_eps,
            estimate_eps=estimate_eps,
            surprise_pct=round(surprise_pct * 100, 2) if surprise_pct is not None else None,
            surprise_cat=surprise_cat,
            estimate_source=estimate_source,
            beat_streak=streak,
        )

    def _derive_period_label(self, filing: EightKFiling) -> str:
        """Derive 'YYYY-Qn' label from filing period_of_report or filed_date."""
        date_str = filing.period_of_report or filing.filed_date
        if not date_str:
            return "UNKNOWN"
        try:
            d = datetime.strptime(date_str[:10], "%Y-%m-%d")
            month = d.month
            quarter = (month - 1) // 3 + 1
            return f"{d.year}-Q{quarter}"
        except ValueError:
            return date_str[:7]

    def _get_yoy_eps_estimate(self, cik: str, period_label: str) -> Optional[float]:
        """
        Use same quarter prior year EPS from XBRL as the 'estimate'.
        period_label format: "2025-Q1"
        """
        try:
            parts = period_label.split("-")
            year = int(parts[0])
            quarter = int(parts[1][1])
            prior_label = f"{year - 1}-Q{quarter}"

            with _db_conn() as conn:
                row = conn.execute(
                    "SELECT actual_eps FROM quarterly_results WHERE ticker = (SELECT ticker FROM quarterly_results WHERE cik = ? LIMIT 1) AND period_label = ?",
                    (cik, prior_label),
                ).fetchone()
                if row and row["actual_eps"] is not None:
                    return row["actual_eps"]
        except Exception:
            pass

        # Try from XBRL EPS directly
        profile = build_xbrl_quarterly_profile(cik)
        eps_df = profile.get("eps_diluted", pd.DataFrame())
        if eps_df.empty or len(eps_df) < 5:
            return None
        # Most recent - 4 quarters back
        return float(eps_df.iloc[-5]["val"]) if len(eps_df) >= 5 else None

    def _save_quarterly_result(
        self,
        ticker: str,
        cik: str,
        period_label: str,
        filing: EightKFiling,
        actual_eps: Optional[float],
        revenue: Optional[float],
    ) -> None:
        with _db_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO quarterly_results
                    (ticker, cik, period_label, period_end, actual_eps, revenue,
                     source, filed_date, accession_no)
                VALUES (?, ?, ?, ?, ?, ?, 'edgar', ?, ?)
                """,
                (
                    ticker,
                    cik,
                    period_label,
                    filing.period_of_report,
                    actual_eps,
                    revenue,
                    filing.filed_date,
                    filing.accession_no,
                ),
            )

    def _save_surprise(
        self,
        ticker: str,
        period_label: str,
        actual_eps: Optional[float],
        estimate_eps: Optional[float],
        surprise_pct: Optional[float],
        surprise_cat: str,
        estimate_source: str,
    ) -> None:
        with _db_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO surprise_history
                    (ticker, period_label, actual_eps, estimate_eps,
                     surprise_pct, surprise_cat, estimate_source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, period_label, actual_eps, estimate_eps, surprise_pct, surprise_cat, estimate_source),
            )

    def _save_guidance(
        self,
        ticker: str,
        period_label: str,
        action: str,
        excerpt: str,
        filing: EightKFiling,
    ) -> None:
        with _db_conn() as conn:
            conn.execute(
                """
                INSERT INTO guidance_log
                    (ticker, period_label, guidance_action, raw_excerpt, filed_date, accession_no)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, period_label, action, excerpt, filing.filed_date, filing.accession_no),
            )

    def _run_quality_metrics(
        self,
        ticker: str,
        cik: str,
        one_time_count: int,
        extracted_revenue: Optional[float],
    ) -> None:
        """Compute and persist EPS quality metrics from XBRL."""
        profile = build_xbrl_quarterly_profile(cik)
        if not profile:
            return

        accruals_records = compute_accruals_metrics(profile)
        cash_eps_records = compute_cash_eps(profile)
        rev_risk = check_revenue_recognition_risk(profile)

        # Match most recent annual accruals with most recent quarterly cash EPS
        if not accruals_records:
            return

        latest_acc = accruals_records[-1]
        latest_cash = cash_eps_records[-1] if cash_eps_records else {}

        fiscal_year = latest_acc["fiscal_year"]
        accruals_ratio = latest_acc.get("accruals_ratio")
        sloan_ratio = latest_acc.get("sloan_ratio")
        quality_flag = latest_acc.get("quality_flag", "MODERATE")
        cash_eps = latest_cash.get("cash_eps")
        gaap_eps = latest_cash.get("gaap_eps")
        gap = latest_cash.get("gap")

        with _db_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO quality_metrics
                    (ticker, fiscal_year, accruals_ratio, sloan_ratio,
                     cash_eps, gaap_eps, cash_eps_gap, one_time_count,
                     deferred_rev_risk, quality_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    fiscal_year,
                    accruals_ratio,
                    sloan_ratio,
                    cash_eps,
                    gaap_eps,
                    gap,
                    one_time_count,
                    int(rev_risk),
                    quality_flag,
                ),
            )


# ---------------------------------------------------------------------------
# Batch surprise runner
# ---------------------------------------------------------------------------


def batch_surprise(tickers: list[str]) -> list[BatchSurpriseItem]:
    """
    Run surprise calculation for multiple tickers sequentially (rate-limit safe).
    Returns list of BatchSurpriseItem with latest period + cat + streak.
    """
    tracker = EarningsKPITrackerV3()
    results: list[BatchSurpriseItem] = []
    for ticker in tickers:
        try:
            res = tracker.run_ticker(ticker, lookback_days=120)
            results.append(
                BatchSurpriseItem(
                    ticker=ticker.upper(),
                    latest_period=res.period_label,
                    surprise_pct=res.surprise_pct,
                    surprise_cat=res.surprise_cat,
                    beat_streak=res.beat_streak,
                )
            )
        except Exception as exc:
            logger.warning("Batch surprise failed for %s: %s", ticker, exc)
            results.append(BatchSurpriseItem(ticker=ticker.upper()))
        time.sleep(0.2)
    return results


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/earnings/v3", tags=["earnings-kpi-v3"])

_tracker = EarningsKPITrackerV3()


@router.get("/surprise/{ticker}", response_model=SurpriseResult, summary="Get latest earnings surprise for a ticker")
def get_surprise(ticker: str, lookback_days: int = Query(default=365, ge=30, le=730)) -> SurpriseResult:
    """
    Fetch 8-K filings from EDGAR, extract EPS, compare against Finviz analyst
    estimate (or same-quarter prior-year as fallback), and return surprise data.
    """
    try:
        return _tracker.run_ticker(ticker.upper(), lookback_days=lookback_days)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Surprise endpoint error for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail="Internal error computing surprise")


@router.get("/calendar", response_model=list[CalendarEntry], summary="Earnings filing deadlines calendar")
def get_calendar(
    ticker: str = Query(..., description="Ticker symbol"),
    days: int = Query(default=30, ge=1, le=365),
    filer_type: str = Query(default="non_accelerated"),
) -> list[CalendarEntry]:
    """
    Return upcoming 10-Q filing deadlines for a ticker (based on SEC mandated
    days after quarter-end: 40 days for large/accelerated, 45 for non-accelerated).
    Optionally filter to within `days` days from today.
    """
    try:
        entries = build_earnings_calendar(ticker.upper(), filer_type=filer_type, lookahead_quarters=4)
        today = date.today()
        cutoff = today + timedelta(days=days)
        return [e for e in entries if e.days_until is not None and e.days_until <= days]
    except Exception as exc:
        logger.error("Calendar endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail="Error building earnings calendar")


@router.get("/quality/{ticker}", response_model=list[QualityMetrics], summary="EPS quality metrics (accruals, cash EPS)")
def get_quality(ticker: str) -> list[QualityMetrics]:
    """
    Return EPS quality metrics for a ticker from SQLite cache.
    Includes accruals ratio (Sloan), cash EPS vs GAAP EPS gap, one-time item count.
    Run /surprise/{ticker} first to populate.
    """
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT fiscal_year, accruals_ratio, sloan_ratio, cash_eps, gaap_eps,
                   cash_eps_gap, one_time_count, deferred_rev_risk, quality_flag
            FROM quality_metrics
            WHERE ticker = ?
            ORDER BY fiscal_year DESC
            LIMIT 10
            """,
            (ticker.upper(),),
        ).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No quality metrics found for {ticker}. Run /surprise/{ticker} first.")

    return [
        QualityMetrics(
            ticker=ticker.upper(),
            fiscal_year=row["fiscal_year"],
            accruals_ratio=row["accruals_ratio"],
            sloan_ratio=row["sloan_ratio"],
            cash_eps=row["cash_eps"],
            gaap_eps=row["gaap_eps"],
            cash_eps_gap=row["cash_eps_gap"],
            one_time_count=row["one_time_count"] or 0,
            deferred_rev_risk=row["deferred_rev_risk"] or 0,
            quality_flag=row["quality_flag"] or "MODERATE",
        )
        for row in rows
    ]


@router.get("/guidance/{ticker}", response_model=list[GuidanceEntry], summary="Guidance language history from 8-Ks")
def get_guidance(ticker: str, limit: int = Query(default=8, ge=1, le=40)) -> list[GuidanceEntry]:
    """
    Return guidance language history (raises/lowers/reaffirms/etc.) extracted
    from 8-K text for a ticker.
    """
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT period_label, guidance_action, raw_excerpt, filed_date
            FROM guidance_log
            WHERE ticker = ?
            ORDER BY filed_date DESC
            LIMIT ?
            """,
            (ticker.upper(), limit),
        ).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No guidance entries for {ticker}")

    return [
        GuidanceEntry(
            ticker=ticker.upper(),
            period_label=row["period_label"],
            guidance_action=row["guidance_action"],
            raw_excerpt=row["raw_excerpt"] or "",
            filed_date=row["filed_date"],
        )
        for row in rows
    ]


@router.get("/beat-streak/{ticker}", summary="Consecutive beat/miss streak")
def get_beat_streak(ticker: str) -> dict:
    """
    Return current consecutive beat or miss streak for a ticker.
    Positive = consecutive beats, negative = consecutive misses.
    """
    streak = compute_beat_streak(ticker.upper())
    direction = "beats" if streak > 0 else "misses" if streak < 0 else "none"
    return {
        "ticker": ticker.upper(),
        "streak": streak,
        "direction": direction,
        "as_of": datetime.utcnow().isoformat(),
    }


@router.get("/accruals/{ticker}", summary="Accruals ratio time series from XBRL")
def get_accruals(ticker: str) -> dict:
    """
    Fetch XBRL companyfacts and compute annual accruals ratio series.
    Accruals ratio = (Net Income - CFO) / avg(Total Assets).
    Higher absolute value = lower earnings quality.
    """
    cik = resolve_cik(ticker.upper())
    if not cik:
        raise HTTPException(status_code=404, detail=f"Cannot resolve CIK for {ticker}")
    try:
        profile = build_xbrl_quarterly_profile(cik)
        records = compute_accruals_metrics(profile)
        cash_eps_records = compute_cash_eps(profile)
        yoy_rev = compute_yoy_growth(profile, "revenue")
        rev_risk = check_revenue_recognition_risk(profile)
    except Exception as exc:
        logger.error("Accruals computation failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail="Error computing accruals")

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "accruals_series": records,
        "cash_eps_series": cash_eps_records[-8:] if cash_eps_records else [],
        "yoy_revenue_growth": yoy_rev[-8:] if yoy_rev else [],
        "deferred_revenue_risk": rev_risk,
        "as_of": datetime.utcnow().isoformat(),
    }


@router.get("/batch-surprise", response_model=list[BatchSurpriseItem], summary="Batch surprise for multiple tickers")
def get_batch_surprise(
    tickers: str = Query(..., description="Comma-separated ticker list, e.g. AAPL,MSFT,GOOG"),
    max_tickers: int = Query(default=10, ge=1, le=25),
) -> list[BatchSurpriseItem]:
    """
    Run earnings surprise calculation for up to 25 tickers.
    Fetches live data from EDGAR and Finviz — allow ~10-30 seconds per ticker.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:max_tickers]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No valid tickers provided")
    try:
        return batch_surprise(ticker_list)
    except Exception as exc:
        logger.error("Batch surprise error: %s", exc)
        raise HTTPException(status_code=500, detail="Batch surprise failed")


# ---------------------------------------------------------------------------
# dim_019 additions: earnings quality score, guidance walk-up, beat rate
# ---------------------------------------------------------------------------


def compute_earnings_quality_score(
    net_income: float,
    cfo: float,
    avg_assets: float,
) -> dict:
    """
    Compute earnings quality score from the Sloan accruals ratio.

    Accruals ratio = (NI - CFO) / avg_assets

    A **negative** accruals ratio means CFO > NI — cash earnings exceed
    reported earnings — which signals **higher quality**.  Sloan (1996)
    showed that high-accruals firms subsequently underperform.

    Parameters
    ----------
    net_income : float
        Annual net income (same currency units as avg_assets).
    cfo : float
        Annual cash flow from operations.
    avg_assets : float
        Average of beginning-of-year and end-of-year total assets.

    Returns
    -------
    dict with keys:
        accruals_ratio : float        (NI - CFO) / avg_assets
        quality_flag   : str          HIGH | MODERATE | LOW | VERY_LOW
        interpretation : str          plain-English note
    """
    if avg_assets == 0:
        raise ValueError("avg_assets must be non-zero")

    accruals_ratio = (net_income - cfo) / avg_assets

    # Quality thresholds (Sloan-convention, absolute value)
    if abs(accruals_ratio) <= 0.02:
        quality_flag = "HIGH"
        interpretation = "Low accruals: cash earnings closely match reported earnings."
    elif abs(accruals_ratio) <= 0.05:
        quality_flag = "MODERATE"
        interpretation = "Moderate accruals: some divergence between cash and reported earnings."
    elif abs(accruals_ratio) <= 0.10:
        quality_flag = "LOW"
        interpretation = "High accruals: reported earnings substantially diverge from cash flow."
    else:
        quality_flag = "VERY_LOW"
        interpretation = "Very high accruals: aggressive accounting or earnings management risk."

    return {
        "accruals_ratio": round(accruals_ratio, 6),
        "quality_flag": quality_flag,
        "interpretation": interpretation,
        "net_income": net_income,
        "cfo": cfo,
        "avg_assets": avg_assets,
    }


def detect_guidance_walk_up(guidance_history: list[dict]) -> dict:
    """
    Detect whether EPS guidance has been raised for ≥2 consecutive quarters.

    Parameters
    ----------
    guidance_history : list[dict]
        Ordered (oldest first) list of guidance entries.  Each dict must
        contain at least {"period_label": str, "guidance_action": str}
        where guidance_action is one of: raises | lowers | reaffirms |
        initiates | withdraws | none_found.

    Returns
    -------
    dict with keys:
        walk_up_detected : bool    True if ≥2 consecutive "raises"
        consecutive_raises : int   Length of current raise streak
        periods : list[str]        Period labels in the raise streak
    """
    if not guidance_history:
        return {"walk_up_detected": False, "consecutive_raises": 0, "periods": []}

    # Work backwards from most recent to find the current raise streak
    streak: list[str] = []
    for entry in reversed(guidance_history):
        if entry.get("guidance_action") == "raises":
            streak.append(entry.get("period_label", ""))
        else:
            break  # streak broken

    streak.reverse()  # oldest first
    walk_up = len(streak) >= 2

    return {
        "walk_up_detected": walk_up,
        "consecutive_raises": len(streak),
        "periods": streak,
    }


def compute_beat_rate(surprise_history: list[dict]) -> dict:
    """
    Compute the fraction of the last 8 quarters where actual EPS > consensus.

    Parameters
    ----------
    surprise_history : list[dict]
        Ordered (oldest first) list of surprise records.  Each dict must
        contain at least {"surprise_cat": str} where surprise_cat is one
        of: STRONG_BEAT | BEAT | IN_LINE | MISS | STRONG_MISS.

    Returns
    -------
    dict with keys:
        beat_rate      : float   Fraction in [0.0, 1.0]
        beats          : int     Number of beat quarters
        quarters_used  : int     Number of quarters examined (≤8)
        assessment     : str     HIGH / MODERATE / LOW
    """
    BEAT_CATS = {"BEAT", "STRONG_BEAT"}
    window = surprise_history[-8:]  # last 8 quarters
    total = len(window)

    if total == 0:
        return {"beat_rate": 0.0, "beats": 0, "quarters_used": 0, "assessment": "INSUFFICIENT_DATA"}

    beats = sum(1 for q in window if q.get("surprise_cat") in BEAT_CATS)
    beat_rate = beats / total

    if beat_rate >= 0.75:
        assessment = "HIGH"
    elif beat_rate >= 0.50:
        assessment = "MODERATE"
    else:
        assessment = "LOW"

    return {
        "beat_rate": round(beat_rate, 4),
        "beats": beats,
        "quarters_used": total,
        "assessment": assessment,
    }


# ---------------------------------------------------------------------------
# Standalone utilities (callable outside FastAPI)
# ---------------------------------------------------------------------------


def get_surprise_from_db(ticker: str) -> list[dict]:
    """Return cached surprise history for a ticker from SQLite."""
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT period_label, actual_eps, estimate_eps, surprise_pct,
                   surprise_cat, estimate_source, created_at
            FROM surprise_history
            WHERE ticker = ?
            ORDER BY period_label DESC
            LIMIT 12
            """,
            (ticker.upper(),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_quarterly_results_from_db(ticker: str) -> list[dict]:
    """Return cached quarterly results for a ticker."""
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT period_label, period_end, actual_eps, revenue, filed_date, source
            FROM quarterly_results
            WHERE ticker = ?
            ORDER BY period_label DESC
            LIMIT 12
            """,
            (ticker.upper(),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_upcoming_calendar(days: int = 30) -> list[dict]:
    """Return all tracked tickers with earnings deadlines within `days` days."""
    cutoff = (date.today() + timedelta(days=days)).isoformat()
    today = date.today().isoformat()
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT ticker, quarter_end, filing_deadline, filer_type, confirmed_date
            FROM earnings_calendar
            WHERE filing_deadline BETWEEN ? AND ?
            ORDER BY filing_deadline ASC
            """,
            (today, cutoff),
        ).fetchall()
    return [dict(row) for row in rows]


def analyze_eps_quality_for_ticker(ticker: str) -> dict:
    """
    Full EPS quality analysis: fetches XBRL, computes all metrics, persists.
    Returns a comprehensive quality summary dict.
    """
    cik = resolve_cik(ticker)
    if not cik:
        return {"error": f"Cannot resolve CIK for {ticker}"}

    profile = build_xbrl_quarterly_profile(cik)
    if not profile:
        return {"error": "No XBRL data available"}

    accruals = compute_accruals_metrics(profile)
    cash_eps = compute_cash_eps(profile)
    yoy_rev = compute_yoy_growth(profile, "revenue")
    yoy_ni = compute_yoy_growth(profile, "net_income")
    rev_risk = check_revenue_recognition_risk(profile)

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "accruals_series": accruals,
        "cash_eps_series": cash_eps[-8:],
        "yoy_revenue_growth": yoy_rev[-8:],
        "yoy_net_income_growth": yoy_ni[-8:],
        "deferred_revenue_risk": rev_risk,
        "summary": {
            "latest_accruals_ratio": accruals[-1]["accruals_ratio"] if accruals else None,
            "latest_quality_flag": accruals[-1]["quality_flag"] if accruals else "UNKNOWN",
            "cash_eps_gap": cash_eps[-1]["gap"] if cash_eps else None,
        },
        "as_of": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pprint

    logging.basicConfig(level=logging.INFO)
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"\n=== SENTINEL Earnings KPI Tracker v3 — {ticker} ===\n")

    tracker = EarningsKPITrackerV3()

    print("1. Running full surprise pipeline...")
    try:
        result = tracker.run_ticker(ticker, lookback_days=200)
        pprint.pprint(result.model_dump())
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n2. Accruals / EPS quality...")
    quality = analyze_eps_quality_for_ticker(ticker)
    pprint.pprint(quality)

    print("\n3. Earnings calendar (next 90 days)...")
    cal = build_earnings_calendar(ticker, lookahead_quarters=4)
    for entry in cal:
        print(f"   {entry.quarter_end} → deadline {entry.filing_deadline} (in {entry.days_until} days)")

    print("\n4. Cached surprise history...")
    hist = get_surprise_from_db(ticker)
    for h in hist:
        print(f"   {h['period_label']}: actual={h['actual_eps']}, est={h['estimate_eps']}, cat={h['surprise_cat']}")

    print("\nDone.")
