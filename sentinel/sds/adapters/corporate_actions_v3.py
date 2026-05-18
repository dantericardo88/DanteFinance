"""
corporate_actions_v3.py — Multi-source corporate action aggregator (dim_008).

Sources (all free, no paid APIs):
  1. SEC EDGAR EFTS full-text search — 8-K items 2.01, 2.04, 3.02, 5.02, 5.03, 8.01
  2. SEC EDGAR Form 8937 (Organizational Actions Affecting Basis of Securities)
       — official split/spin-off adjustment factors
       https://efts.sec.gov/LATEST/search-index?forms=8937
  3. SEC EDGAR DEF 14C — reverse splits / authorized share changes without shareholder vote
  4. SEC Form 10-12B/G — new entity registration signals spin-off
  5. SEC Form SC TO-T — tender offer / M&A
  6. yfinance dividends + splits — baseline (clearly labelled as approximate)

Corporate action categories covered:
  - Regular dividends, special dividends, stock dividends
  - Dividend cuts / suspensions (>20% reduction flagged)
  - Forward splits, reverse splits (with 8937 factor verification)
  - Spin-offs / carve-outs (10-12B, 8-K language, distribution ratio)
  - M&A completions (8-K item 2.01, SC TO-T)
  - Rights offerings (S-1 + 8-K language)
  - Going-private / deregistration (Form 15)

Storage: SQLite with six tables — corporate_actions, dividend_history, split_history,
         spinoff_history, ma_actions, adjustment_factors

FastAPI router at /corp-actions/v3:
  GET /history/{ticker}
  GET /dividends/{ticker}
  GET /splits/{ticker}
  GET /spinoffs/{ticker}
  GET /adjustment-factor/{ticker}/{date}
  GET /upcoming
  GET /calendar

Dependencies: requests, sqlite3, pandas, numpy, fastapi, yfinance
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EDGAR_EFTS       = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_DATA       = "https://data.sec.gov"
_SEC_TICKERS_URL  = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept":     "application/json",
}

_DB_PATH = Path(os.getenv("SENTINEL_CORP_ACTIONS_DB", ".sentinel/corporate_actions_v3.db"))

_SEC_RATE_DELAY   = 0.12    # 120 ms between SEC calls (10 req/sec limit)
_HTTP_TIMEOUT     = 30
_RETRY_MAX        = 3
_RETRY_BACKOFF    = 1.5
_DIVIDEND_CUT_THRESHOLD = 0.20   # flag reduction >= 20% as cut/suspension

# 8-K item codes for different corporate action types
_8K_ITEMS = {
    "merger_completion":    "2.01",
    "triggering_events":    "2.04",
    "unregistered_sec":     "3.02",
    "director_changes":     "5.02",
    "articles_amendment":   "5.03",
    "extraordinary":        "8.01",
}

# Text patterns for EDGAR full-text search (case-insensitive in practice)
_SPECIAL_DIV_PATTERNS    = ["special cash dividend", "extraordinary dividend", "special dividend"]
_SPINOFF_PATTERNS        = ["spin-off", "spinoff", "separation", "distribution ratio", "pro rata distribution"]
_RIGHTS_OFFERING_PATTERNS = ["rights offering", "subscription rights", "oversubscription privilege"]
_REVERSE_SPLIT_PATTERNS  = ["reverse stock split", "reverse split", "consolidation of shares"]
_FORWARD_SPLIT_PATTERNS  = ["stock split", "forward split", "stock dividend"]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CorporateActionRecord(BaseModel):
    ticker:       str
    action_type:  str      # dividend | split | reverse_split | spinoff | merger | rights | deregistration
    ex_date:      Optional[date] = None
    record_date:  Optional[date] = None
    pay_date:     Optional[date] = None
    amount:       Optional[float] = None   # cash per share (dividends)
    ratio:        Optional[str] = None     # "4:1" for splits
    factor:       Optional[float] = None   # backward-adj factor (e.g. 0.25 for 4:1 split)
    source:       str = "unknown"
    notes:        Optional[str] = None
    filing_date:  Optional[date] = None
    accession:    Optional[str] = None

class DividendRecord(BaseModel):
    ticker:       str
    ex_date:      date
    record_date:  Optional[date] = None
    pay_date:     Optional[date] = None
    amount:       float
    div_type:     str = "regular"   # regular | special | stock | cut | suspended
    source:       str = "yfinance"
    prev_amount:  Optional[float] = None
    pct_change:   Optional[float] = None
    is_cut:       bool = False

class SplitRecord(BaseModel):
    ticker:       str
    ex_date:      date
    ratio_new:    int
    ratio_old:    int
    factor:       float
    split_type:   str = "forward"    # forward | reverse
    source:       str = "yfinance"
    verified_8937: bool = False
    notes:        Optional[str] = None

class SpinoffRecord(BaseModel):
    ticker:        str
    ex_date:       Optional[date] = None
    spinoff_ticker: Optional[str] = None
    distribution_ratio: Optional[str] = None
    source:        str = "edgar"
    filing_date:   Optional[date] = None
    accession:     Optional[str] = None
    description:   Optional[str] = None

class MAActionRecord(BaseModel):
    ticker:          str
    action_type:     str    # tender_offer | merger_completion | going_private
    announcement_date: Optional[date] = None
    expiry_date:     Optional[date] = None
    consideration:   Optional[str] = None
    acquirer:        Optional[str] = None
    source:          str = "edgar"
    accession:       Optional[str] = None

class AdjustmentFactor(BaseModel):
    ticker:     str
    as_of_date: date
    factor:     float
    source:     str
    action_type: str

class UpcomingAction(BaseModel):
    ticker:      str
    action_type: str
    ex_date:     Optional[date] = None
    pay_date:    Optional[date] = None
    amount:      Optional[float] = None
    description: Optional[str] = None

# ---------------------------------------------------------------------------
# SQLite persistence layer
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()


def _get_db() -> sqlite3.Connection:
    """Open (or create) the SQLite database with WAL mode."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create all tables if they do not exist."""
    with _db_lock:
        conn = _get_db()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker       TEXT NOT NULL,
                    action_type  TEXT NOT NULL,
                    ex_date      TEXT,
                    record_date  TEXT,
                    pay_date     TEXT,
                    amount       REAL,
                    ratio        TEXT,
                    factor       REAL,
                    source       TEXT,
                    notes        TEXT,
                    filing_date  TEXT,
                    accession    TEXT,
                    inserted_at  TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, action_type, ex_date, source)
                );

                CREATE INDEX IF NOT EXISTS idx_ca_ticker
                    ON corporate_actions (ticker, ex_date DESC);

                CREATE TABLE IF NOT EXISTS dividend_history (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker       TEXT NOT NULL,
                    ex_date      TEXT NOT NULL,
                    record_date  TEXT,
                    pay_date     TEXT,
                    amount       REAL NOT NULL,
                    div_type     TEXT DEFAULT 'regular',
                    source       TEXT DEFAULT 'yfinance',
                    prev_amount  REAL,
                    pct_change   REAL,
                    is_cut       INTEGER DEFAULT 0,
                    inserted_at  TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, ex_date, source)
                );

                CREATE INDEX IF NOT EXISTS idx_dh_ticker
                    ON dividend_history (ticker, ex_date DESC);

                CREATE TABLE IF NOT EXISTS split_history (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker        TEXT NOT NULL,
                    ex_date       TEXT NOT NULL,
                    ratio_new     INTEGER,
                    ratio_old     INTEGER,
                    factor        REAL,
                    split_type    TEXT DEFAULT 'forward',
                    source        TEXT DEFAULT 'yfinance',
                    verified_8937 INTEGER DEFAULT 0,
                    notes         TEXT,
                    inserted_at   TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, ex_date, source)
                );

                CREATE INDEX IF NOT EXISTS idx_sh_ticker
                    ON split_history (ticker, ex_date DESC);

                CREATE TABLE IF NOT EXISTS spinoff_history (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker              TEXT NOT NULL,
                    ex_date             TEXT,
                    spinoff_ticker      TEXT,
                    distribution_ratio  TEXT,
                    source              TEXT DEFAULT 'edgar',
                    filing_date         TEXT,
                    accession           TEXT,
                    description         TEXT,
                    inserted_at         TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, accession)
                );

                CREATE INDEX IF NOT EXISTS idx_soh_ticker
                    ON spinoff_history (ticker);

                CREATE TABLE IF NOT EXISTS ma_actions (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker              TEXT NOT NULL,
                    action_type         TEXT NOT NULL,
                    announcement_date   TEXT,
                    expiry_date         TEXT,
                    consideration       TEXT,
                    acquirer            TEXT,
                    source              TEXT DEFAULT 'edgar',
                    accession           TEXT,
                    inserted_at         TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, action_type, accession)
                );

                CREATE INDEX IF NOT EXISTS idx_ma_ticker
                    ON ma_actions (ticker, announcement_date DESC);

                CREATE TABLE IF NOT EXISTS adjustment_factors (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker       TEXT NOT NULL,
                    as_of_date   TEXT NOT NULL,
                    factor       REAL NOT NULL,
                    source       TEXT,
                    action_type  TEXT,
                    inserted_at  TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, as_of_date, action_type)
                );

                CREATE INDEX IF NOT EXISTS idx_af_ticker_date
                    ON adjustment_factors (ticker, as_of_date DESC);
            """)
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, params: Optional[Dict] = None, timeout: int = _HTTP_TIMEOUT) -> requests.Response:
    """GET with retry and SEC-compliant rate limiting."""
    for attempt in range(_RETRY_MAX):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            if resp.status_code == 429:
                wait = _RETRY_BACKOFF ** (attempt + 2)
                logger.warning("SEC rate limit hit, backing off", wait=wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == _RETRY_MAX - 1:
                raise
            time.sleep(_RETRY_BACKOFF ** (attempt + 1))
    raise RuntimeError(f"Failed to GET {url} after {_RETRY_MAX} attempts")


def _sec_search(
    query: str,
    forms: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    hits: int = 40,
) -> List[Dict]:
    """EDGAR EFTS full-text search returning list of hit dicts."""
    params: Dict[str, Any] = {
        "q":     f'"{query}"',
        "forms": forms,
        "hits.hits.total.value": hits,
        "_source": "file_date,period_of_report,entity_name,form_type,accession_no,file_num",
    }
    if date_from:
        params["dateRange"] = "custom"
        params["startdt"]   = date_from
    if date_to:
        params["dateRange"] = "custom"
        params["enddt"]     = date_to

    try:
        time.sleep(_SEC_RATE_DELAY)
        resp = _http_get(_EDGAR_EFTS, params=params)
        data = resp.json()
        return data.get("hits", {}).get("hits", [])[:hits]
    except Exception as exc:
        logger.error("EDGAR EFTS search failed", query=query, forms=forms, error=str(exc))
        return []


def _load_cik_map() -> Dict[str, str]:
    """Load ticker -> CIK mapping from SEC bulk file (cached in module)."""
    global _CIK_MAP_CACHE
    if _CIK_MAP_CACHE:
        return _CIK_MAP_CACHE
    try:
        time.sleep(_SEC_RATE_DELAY)
        resp = _http_get(_SEC_TICKERS_URL)
        data = resp.json()
        result: Dict[str, str] = {}
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik    = str(entry.get("cik_str", "")).zfill(10)
            if ticker and cik:
                result[ticker] = cik
        _CIK_MAP_CACHE = result
        logger.info("SEC CIK map loaded", count=len(result))
        return result
    except Exception as exc:
        logger.error("Failed to load SEC CIK map", error=str(exc))
        return {}


_CIK_MAP_CACHE: Dict[str, str] = {}


def _ticker_to_cik(ticker: str) -> Optional[str]:
    return _load_cik_map().get(ticker.upper())


# ---------------------------------------------------------------------------
# yfinance baseline fetchers
# ---------------------------------------------------------------------------

def _fetch_yf_dividends(ticker: str) -> List[DividendRecord]:
    """Fetch dividend history from yfinance, detect cuts / special dividends."""
    if not _HAS_YF:
        return []
    try:
        info = yf.Ticker(ticker)
        divs = info.dividends
    except Exception as exc:
        logger.warning("yfinance dividends failed", ticker=ticker, error=str(exc))
        return []

    if divs is None or divs.empty:
        return []

    records: List[DividendRecord] = []
    amounts = divs.values.tolist()
    dates   = [d.date() if hasattr(d, "date") else d for d in divs.index]

    for i, (ex_dt, amount) in enumerate(zip(dates, amounts)):
        if amount <= 0:
            continue
        prev_amount = amounts[i - 1] if i > 0 else None
        pct_change  = None
        is_cut      = False
        div_type    = "regular"

        if prev_amount and prev_amount > 0:
            pct_change = (amount - prev_amount) / prev_amount
            if pct_change <= -_DIVIDEND_CUT_THRESHOLD:
                is_cut   = True
                div_type = "cut" if amount > 0 else "suspended"

        records.append(DividendRecord(
            ticker      = ticker.upper(),
            ex_date     = ex_dt,
            amount      = round(float(amount), 8),
            div_type    = div_type,
            source      = "yfinance",
            prev_amount = round(float(prev_amount), 8) if prev_amount else None,
            pct_change  = round(pct_change, 6) if pct_change is not None else None,
            is_cut      = is_cut,
        ))
    logger.info("yfinance dividends fetched", ticker=ticker, count=len(records))
    return records


def _fetch_yf_splits(ticker: str) -> List[SplitRecord]:
    """Fetch split history from yfinance and convert to SplitRecord list."""
    if not _HAS_YF:
        return []
    try:
        splits = yf.Ticker(ticker).splits
    except Exception as exc:
        logger.warning("yfinance splits failed", ticker=ticker, error=str(exc))
        return []

    if splits is None or splits.empty:
        return []

    records: List[SplitRecord] = []
    for dt, ratio in splits.items():
        if ratio <= 0:
            continue
        ex_dt     = dt.date() if hasattr(dt, "date") else dt
        ratio_new = int(round(ratio)) if ratio >= 1 else 1
        ratio_old = 1 if ratio >= 1 else int(round(1 / ratio))
        factor    = round(1.0 / float(ratio), 10) if ratio != 0 else 1.0
        split_type = "forward" if ratio > 1 else "reverse"

        records.append(SplitRecord(
            ticker     = ticker.upper(),
            ex_date    = ex_dt,
            ratio_new  = ratio_new,
            ratio_old  = ratio_old,
            factor     = factor,
            split_type = split_type,
            source     = "yfinance",
        ))
    logger.info("yfinance splits fetched", ticker=ticker, count=len(records))
    return records


# ---------------------------------------------------------------------------
# EDGAR 8937 — official adjustment factor source
# ---------------------------------------------------------------------------

def _fetch_8937_filings(ticker: str, cik: str) -> List[Dict]:
    """Fetch Form 8937 filings for a company.

    8937 reports organizational actions affecting the basis of securities:
    splits, reverse splits, spin-offs, stock dividends.
    Returns list of raw filing dicts from EDGAR submissions.
    """
    try:
        time.sleep(_SEC_RATE_DELAY)
        url  = f"{_EDGAR_DATA}/submissions/CIK{cik}.json"
        resp = _http_get(url)
        subs = resp.json()
    except Exception as exc:
        logger.error("EDGAR submissions error", cik=cik, error=str(exc))
        return []

    filings = subs.get("filings", {}).get("recent", {})
    if not filings:
        return []

    forms   = filings.get("form",          [])
    dates   = filings.get("filingDate",    [])
    accs    = filings.get("accessionNumber", [])
    results = []

    for i, form in enumerate(forms):
        if form == "8937":
            results.append({
                "form_type":   form,
                "filing_date": dates[i] if i < len(dates) else None,
                "accession":   accs[i]  if i < len(accs)  else None,
                "cik":         cik,
                "ticker":      ticker,
            })
    logger.info("8937 filings found", ticker=ticker, count=len(results))
    return results


def _parse_8937_factors(filings: List[Dict]) -> List[SplitRecord]:
    """Download and parse each 8937 document for split/spin-off factors.

    8937 documents are HTML/XML — we extract ratio text with regex since
    the SEC does not publish a structured 8937 XBRL endpoint.
    """
    records: List[SplitRecord] = []

    # Regex patterns for common 8937 language
    ratio_pattern   = re.compile(
        r"(\d+)\s*(?:for|:)\s*(\d+)\s*(?:stock\s+)?(?:split|reverse\s+split|share)",
        re.IGNORECASE,
    )
    factor_pattern  = re.compile(
        r"adjustment\s+factor\s*[:\-=]\s*([0-9]+\.?[0-9]*)",
        re.IGNORECASE,
    )
    date_pattern    = re.compile(
        r"(?:ex[- ]?date|effective\s+date|distribution\s+date)\s*[:\-]?\s*"
        r"(\w+ \d{1,2},? \d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    )

    for filing in filings[:20]:   # cap at 20 per ticker to stay within rate limits
        acc = (filing.get("accession") or "").replace("-", "")
        cik = filing.get("cik", "")
        if not acc or not cik:
            continue

        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{acc}/{acc}-index.htm"
        )
        try:
            time.sleep(_SEC_RATE_DELAY)
            resp = _http_get(doc_url)
            text = resp.text

            # Extract ratio
            ratio_match = ratio_pattern.search(text)
            if not ratio_match:
                continue

            new_shares = int(ratio_match.group(1))
            old_shares = int(ratio_match.group(2))
            if old_shares == 0:
                continue

            raw_ratio   = new_shares / old_shares
            factor      = round(1.0 / raw_ratio, 10)
            split_type  = "forward" if raw_ratio > 1 else "reverse"

            # Try to find effective date
            ex_dt: Optional[date] = None
            date_match = date_pattern.search(text)
            if date_match:
                for fmt in ("%B %d, %Y", "%B %d %Y", "%m/%d/%Y", "%Y-%m-%d"):
                    try:
                        ex_dt = datetime.strptime(date_match.group(1).strip().rstrip(","), fmt).date()
                        break
                    except ValueError:
                        continue
            if ex_dt is None:
                raw_fd = filing.get("filing_date")
                if raw_fd:
                    try:
                        ex_dt = date.fromisoformat(raw_fd)
                    except ValueError:
                        pass

            if ex_dt is None:
                continue

            records.append(SplitRecord(
                ticker         = filing["ticker"].upper(),
                ex_date        = ex_dt,
                ratio_new      = new_shares,
                ratio_old      = old_shares,
                factor         = factor,
                split_type     = split_type,
                source         = "edgar_8937",
                verified_8937  = True,
                notes          = f"8937 filing {filing.get('accession')}",
            ))
        except Exception as exc:
            logger.warning("8937 parse error", accession=filing.get("accession"), error=str(exc))
            continue

    logger.info("8937 split records parsed", ticker=filings[0].get("ticker") if filings else "?", count=len(records))
    return records


# ---------------------------------------------------------------------------
# EDGAR 8-K special dividend / spin-off / M&A detectors
# ---------------------------------------------------------------------------

def _search_8k_special_dividends(ticker: str, cik: str, lookback_years: int = 5) -> List[CorporateActionRecord]:
    """Search 8-K filings for special cash dividend language."""
    date_from = (datetime.utcnow() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    results: List[CorporateActionRecord] = []

    for pattern in _SPECIAL_DIV_PATTERNS:
        hits = _sec_search(pattern, "8-K", date_from=date_from)
        for hit in hits:
            src   = hit.get("_source", {})
            name  = (src.get("entity_name") or "").upper()
            # Try to match by entity name containing the ticker
            # (SEC doesn't directly map hits to tickers in EFTS by default)
            acc   = src.get("accession_no", "")
            fd    = src.get("file_date") or src.get("period_of_report")
            ex_dt = None
            if fd:
                try:
                    ex_dt = date.fromisoformat(fd[:10])
                except ValueError:
                    pass

            results.append(CorporateActionRecord(
                ticker      = ticker.upper(),
                action_type = "special_dividend",
                ex_date     = ex_dt,
                source      = "edgar_8k",
                notes       = f"Matched pattern '{pattern}' — entity: {name}",
                filing_date = ex_dt,
                accession   = acc,
            ))
        if results:
            break   # found matches on first working pattern

    return results


def _search_spinoffs_edgar(ticker: str, cik: str, lookback_years: int = 10) -> List[SpinoffRecord]:
    """Search EDGAR for spin-off signals: 10-12B/G, 8-K spin-off language, Form 15."""
    date_from = (datetime.utcnow() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    records: List[SpinoffRecord] = []

    # 1. Search 10-12B registrations by the parent company — signal of a new entity
    try:
        time.sleep(_SEC_RATE_DELAY)
        resp = _http_get(
            f"{_EDGAR_DATA}/submissions/CIK{cik}.json",
        )
        subs    = resp.json()
        filings = subs.get("filings", {}).get("recent", {})
        forms   = filings.get("form",          [])
        dates   = filings.get("filingDate",    [])
        accs    = filings.get("accessionNumber", [])

        for i, form in enumerate(forms):
            if form in ("10-12B", "10-12G", "S-11"):
                fd  = dates[i] if i < len(dates) else None
                acc = accs[i]  if i < len(accs)  else None
                fd_date = date.fromisoformat(fd) if fd else None
                if fd_date and fd_date < date.fromisoformat(date_from):
                    continue
                records.append(SpinoffRecord(
                    ticker      = ticker.upper(),
                    filing_date = fd_date,
                    accession   = acc,
                    source      = "edgar_10_12b",
                    description = f"New entity registration ({form}) — potential spin-off",
                ))
    except Exception as exc:
        logger.warning("Spinoff 10-12B search failed", ticker=ticker, error=str(exc))

    # 2. EFTS search for spin-off language in 8-K filings
    for pattern in _SPINOFF_PATTERNS[:2]:   # limit EFTS calls
        hits = _sec_search(pattern, "8-K", date_from=date_from, hits=20)
        for hit in hits:
            src = hit.get("_source", {})
            acc = src.get("accession_no", "")
            fd  = src.get("file_date") or src.get("period_of_report")
            fd_date = date.fromisoformat(fd[:10]) if fd else None

            # Extract distribution ratio from text if available
            text_snip = hit.get("_source", {}).get("file_date", "")
            dist_ratio = None
            dr_match = re.search(r"(\d+)\s*(?:shares?|units?)\s*(?:of|for)\s*(?:every|each)\s*(\d+)", text_snip, re.IGNORECASE)
            if dr_match:
                dist_ratio = f"{dr_match.group(1)}:{dr_match.group(2)}"

            records.append(SpinoffRecord(
                ticker             = ticker.upper(),
                filing_date        = fd_date,
                distribution_ratio = dist_ratio,
                source             = "edgar_8k_efts",
                accession          = acc,
                description        = f"8-K spin-off keyword match: '{pattern}'",
            ))
        if records:
            break

    logger.info("Spinoff records found", ticker=ticker, count=len(records))
    return records


def _search_ma_actions_edgar(ticker: str, cik: str, lookback_years: int = 5) -> List[MAActionRecord]:
    """Search EDGAR for M&A events: SC TO-T (tender offer), 8-K item 2.01 (merger completion)."""
    date_from = (datetime.utcnow() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    records: List[MAActionRecord] = []

    try:
        time.sleep(_SEC_RATE_DELAY)
        url  = f"{_EDGAR_DATA}/submissions/CIK{cik}.json"
        resp = _http_get(url)
        subs    = resp.json()
        filings = subs.get("filings", {}).get("recent", {})
        forms   = filings.get("form",          [])
        dates   = filings.get("filingDate",    [])
        accs    = filings.get("accessionNumber", [])

        for i, form in enumerate(forms):
            fd  = dates[i] if i < len(dates) else None
            acc = accs[i]  if i < len(accs)  else None
            fd_date = date.fromisoformat(fd) if fd else None
            if fd_date and fd_date < date.fromisoformat(date_from):
                continue

            if form in ("SC TO-T", "SC TO-T/A"):
                records.append(MAActionRecord(
                    ticker            = ticker.upper(),
                    action_type       = "tender_offer",
                    announcement_date = fd_date,
                    source            = "edgar_sc_to_t",
                    accession         = acc,
                ))
            elif form in ("15", "15F"):
                records.append(MAActionRecord(
                    ticker            = ticker.upper(),
                    action_type       = "going_private",
                    announcement_date = fd_date,
                    source            = "edgar_form_15",
                    accession         = acc,
                ))

    except Exception as exc:
        logger.warning("M&A EDGAR search failed", ticker=ticker, error=str(exc))

    # EFTS search for merger completion language in 8-K
    hits = _sec_search("merger completion", "8-K", date_from=date_from, hits=10)
    for hit in hits:
        src = hit.get("_source", {})
        acc = src.get("accession_no", "")
        fd  = src.get("file_date") or src.get("period_of_report")
        fd_date = date.fromisoformat(fd[:10]) if fd else None
        records.append(MAActionRecord(
            ticker            = ticker.upper(),
            action_type       = "merger_completion",
            announcement_date = fd_date,
            source            = "edgar_8k_efts",
            accession         = acc,
        ))

    logger.info("M&A records found", ticker=ticker, count=len(records))
    return records


def _search_reverse_splits_def14c(ticker: str, cik: str, lookback_years: int = 10) -> List[SplitRecord]:
    """Search DEF 14C filings for reverse split authorizations without shareholder vote."""
    date_from = (datetime.utcnow() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    records: List[SplitRecord] = []

    try:
        time.sleep(_SEC_RATE_DELAY)
        url  = f"{_EDGAR_DATA}/submissions/CIK{cik}.json"
        resp = _http_get(url)
        subs    = resp.json()
        filings = subs.get("filings", {}).get("recent", {})
        forms   = filings.get("form",          [])
        dates   = filings.get("filingDate",    [])
        accs    = filings.get("accessionNumber", [])

        for i, form in enumerate(forms):
            if form in ("DEF 14C", "PRE 14C"):
                fd  = dates[i] if i < len(dates) else None
                acc = accs[i]  if i < len(accs)  else None
                fd_date = date.fromisoformat(fd) if fd else None
                if fd_date and fd_date < date.fromisoformat(date_from):
                    continue

                # Mark as potential reverse split pending ratio extraction
                records.append(SplitRecord(
                    ticker     = ticker.upper(),
                    ex_date    = fd_date or date.today(),
                    ratio_new  = 1,
                    ratio_old  = 1,
                    factor     = 1.0,
                    split_type = "reverse",
                    source     = "edgar_def14c",
                    notes      = f"DEF 14C filing — potential reverse split authorization. Accession: {acc}",
                ))
    except Exception as exc:
        logger.warning("DEF 14C search failed", ticker=ticker, error=str(exc))

    return records


def _search_rights_offerings(ticker: str, cik: str, lookback_years: int = 5) -> List[CorporateActionRecord]:
    """Search for rights offerings via S-1 filings and 8-K language."""
    date_from = (datetime.utcnow() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    records: List[CorporateActionRecord] = []

    try:
        time.sleep(_SEC_RATE_DELAY)
        url  = f"{_EDGAR_DATA}/submissions/CIK{cik}.json"
        resp = _http_get(url)
        subs    = resp.json()
        filings = subs.get("filings", {}).get("recent", {})
        forms   = filings.get("form",          [])
        dates   = filings.get("filingDate",    [])
        accs    = filings.get("accessionNumber", [])

        for i, form in enumerate(forms):
            # S-1 with rights offering language, or direct S-3D (shelf for dividend reinvestment)
            if form in ("S-1", "S-1/A", "F-1", "F-1/A"):
                fd  = dates[i] if i < len(dates) else None
                acc = accs[i]  if i < len(accs)  else None
                fd_date = date.fromisoformat(fd) if fd else None
                if fd_date and fd_date < date.fromisoformat(date_from):
                    continue
                records.append(CorporateActionRecord(
                    ticker      = ticker.upper(),
                    action_type = "rights_offering",
                    ex_date     = fd_date,
                    source      = "edgar_s1",
                    notes       = f"S-1 registration — potential rights offering. Accession: {acc}",
                    filing_date = fd_date,
                    accession   = acc,
                ))
    except Exception as exc:
        logger.warning("Rights offering search failed", ticker=ticker, error=str(exc))

    return records


# ---------------------------------------------------------------------------
# Adjustment factor computation
# ---------------------------------------------------------------------------

def _compute_backward_adj_factors(
    ticker: str,
    splits: List[SplitRecord],
    dividends: List[DividendRecord],
    as_of: date,
) -> List[AdjustmentFactor]:
    """Compute cumulative backward-adjustment factors for each corporate action.

    Convention: factor multiplied into raw price equals adjusted (current-comparable) price.
    - Split 4:1 on 2020-08-31 → bars before that date get factor = 0.25
    - Dividend factor = (price_before_ex - div) / price_before_ex (price-based, skipped here)

    Returns one AdjustmentFactor per action, sorted oldest→newest.
    """
    factors: List[AdjustmentFactor] = []

    for sp in sorted(splits, key=lambda s: s.ex_date):
        if sp.ex_date <= as_of:
            factors.append(AdjustmentFactor(
                ticker      = ticker.upper(),
                as_of_date  = sp.ex_date,
                factor      = sp.factor,
                source      = sp.source,
                action_type = sp.split_type,
            ))

    # Cumulative product over time — each factor compounds with all subsequent
    cumulative = 1.0
    result: List[AdjustmentFactor] = []
    for af in reversed(sorted(factors, key=lambda f: f.as_of_date)):
        cumulative *= af.factor
        result.append(AdjustmentFactor(
            ticker      = af.ticker,
            as_of_date  = af.as_of_date,
            factor      = round(cumulative, 10),
            source      = af.source,
            action_type = af.action_type,
        ))

    return sorted(result, key=lambda f: f.as_of_date)


def _get_cumulative_factor_for_date(
    ticker: str,
    query_date: date,
    splits: List[SplitRecord],
) -> float:
    """Return the single backward-adjustment factor for a bar on query_date.

    Multiply all split factors whose ex_date > query_date (events after the bar date).
    """
    factor = 1.0
    for sp in splits:
        if sp.ex_date > query_date:
            factor *= sp.factor
    return round(factor, 10)


# ---------------------------------------------------------------------------
# Database persistence helpers
# ---------------------------------------------------------------------------

def _upsert_dividends(records: List[DividendRecord]) -> int:
    """Persist dividend records to SQLite, returns inserted count."""
    if not records:
        return 0
    rows = [
        (
            r.ticker, str(r.ex_date),
            str(r.record_date) if r.record_date else None,
            str(r.pay_date)    if r.pay_date    else None,
            r.amount, r.div_type, r.source,
            r.prev_amount, r.pct_change, int(r.is_cut),
        )
        for r in records
    ]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO dividend_history
                   (ticker, ex_date, record_date, pay_date, amount, div_type, source,
                    prev_amount, pct_change, is_cut)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


def _upsert_splits(records: List[SplitRecord]) -> int:
    """Persist split records to SQLite, returns inserted count."""
    if not records:
        return 0
    rows = [
        (
            r.ticker, str(r.ex_date), r.ratio_new, r.ratio_old,
            r.factor, r.split_type, r.source, int(r.verified_8937),
            r.notes,
        )
        for r in records
    ]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO split_history
                   (ticker, ex_date, ratio_new, ratio_old, factor, split_type,
                    source, verified_8937, notes)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


def _upsert_spinoffs(records: List[SpinoffRecord]) -> int:
    if not records:
        return 0
    rows = [
        (
            r.ticker,
            str(r.ex_date)      if r.ex_date      else None,
            r.spinoff_ticker,
            r.distribution_ratio,
            r.source,
            str(r.filing_date)  if r.filing_date  else None,
            r.accession,
            r.description,
        )
        for r in records
    ]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR IGNORE INTO spinoff_history
                   (ticker, ex_date, spinoff_ticker, distribution_ratio,
                    source, filing_date, accession, description)
                   VALUES (?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


def _upsert_ma_actions(records: List[MAActionRecord]) -> int:
    if not records:
        return 0
    rows = [
        (
            r.ticker, r.action_type,
            str(r.announcement_date) if r.announcement_date else None,
            str(r.expiry_date)       if r.expiry_date       else None,
            r.consideration, r.acquirer, r.source, r.accession,
        )
        for r in records
    ]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR IGNORE INTO ma_actions
                   (ticker, action_type, announcement_date, expiry_date,
                    consideration, acquirer, source, accession)
                   VALUES (?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


def _upsert_adjustment_factors(records: List[AdjustmentFactor]) -> int:
    if not records:
        return 0
    rows = [
        (r.ticker, str(r.as_of_date), r.factor, r.source, r.action_type)
        for r in records
    ]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO adjustment_factors
                   (ticker, as_of_date, factor, source, action_type)
                   VALUES (?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


def _upsert_corporate_actions(records: List[CorporateActionRecord]) -> int:
    if not records:
        return 0
    rows = [
        (
            r.ticker, r.action_type,
            str(r.ex_date)      if r.ex_date      else None,
            str(r.record_date)  if r.record_date  else None,
            str(r.pay_date)     if r.pay_date     else None,
            r.amount, r.ratio, r.factor,
            r.source, r.notes,
            str(r.filing_date)  if r.filing_date  else None,
            r.accession,
        )
        for r in records
    ]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR IGNORE INTO corporate_actions
                   (ticker, action_type, ex_date, record_date, pay_date,
                    amount, ratio, factor, source, notes, filing_date, accession)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Core aggregator: full pipeline for one ticker
# ---------------------------------------------------------------------------

def aggregate_corporate_actions(ticker: str) -> Dict[str, Any]:
    """Run the full multi-source aggregation pipeline for one ticker.

    Returns a summary dict with counts of each category.
    All results are persisted to SQLite.
    """
    ticker = ticker.upper()
    logger.info("Corporate actions aggregation start", ticker=ticker)
    cik    = _ticker_to_cik(ticker)
    summary: Dict[str, Any] = {
        "ticker":    ticker,
        "cik":       cik,
        "dividends": 0,
        "splits":    0,
        "spinoffs":  0,
        "ma_actions": 0,
        "rights":    0,
        "adj_factors": 0,
        "errors":    [],
    }

    # ── 1. yfinance baseline ─────────────────────────────────────────────────
    try:
        div_records  = _fetch_yf_dividends(ticker)
        summary["dividends"] += _upsert_dividends(div_records)
    except Exception as exc:
        summary["errors"].append(f"yf_dividends: {exc}")

    try:
        split_records = _fetch_yf_splits(ticker)
        summary["splits"] += _upsert_splits(split_records)
    except Exception as exc:
        summary["errors"].append(f"yf_splits: {exc}")
        split_records = []

    # ── 2. EDGAR 8937 official factors (if CIK available) ───────────────────
    verified_splits: List[SplitRecord] = []
    if cik:
        try:
            filings_8937  = _fetch_8937_filings(ticker, cik)
            verified_splits = _parse_8937_factors(filings_8937)
            summary["splits"] += _upsert_splits(verified_splits)
        except Exception as exc:
            summary["errors"].append(f"edgar_8937: {exc}")

        # ── 3. DEF 14C reverse splits ────────────────────────────────────────
        try:
            def14c_splits = _search_reverse_splits_def14c(ticker, cik)
            summary["splits"] += _upsert_splits(def14c_splits)
        except Exception as exc:
            summary["errors"].append(f"def14c: {exc}")

        # ── 4. Spin-offs ─────────────────────────────────────────────────────
        try:
            spinoff_records = _search_spinoffs_edgar(ticker, cik)
            summary["spinoffs"] += _upsert_spinoffs(spinoff_records)
        except Exception as exc:
            summary["errors"].append(f"spinoffs: {exc}")

        # ── 5. M&A ───────────────────────────────────────────────────────────
        try:
            ma_records = _search_ma_actions_edgar(ticker, cik)
            summary["ma_actions"] += _upsert_ma_actions(ma_records)
        except Exception as exc:
            summary["errors"].append(f"ma_actions: {exc}")

        # ── 6. Rights offerings ──────────────────────────────────────────────
        try:
            rights_records = _search_rights_offerings(ticker, cik)
            summary["rights"] += _upsert_corporate_actions(rights_records)
        except Exception as exc:
            summary["errors"].append(f"rights: {exc}")

    # ── 7. Adjustment factor database ────────────────────────────────────────
    all_splits = split_records + verified_splits
    try:
        adj_factors = _compute_backward_adj_factors(ticker, all_splits, div_records if 'div_records' in dir() else [], date.today())
        summary["adj_factors"] += _upsert_adjustment_factors(adj_factors)
    except Exception as exc:
        summary["errors"].append(f"adj_factors: {exc}")

    logger.info("Corporate actions aggregation complete", ticker=ticker, summary=summary)
    return summary


# ---------------------------------------------------------------------------
# Database query helpers for API layer
# ---------------------------------------------------------------------------

def _query_dividends(ticker: str, limit: int = 200) -> List[DividendRecord]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, ex_date, record_date, pay_date, amount, div_type,
                      source, prev_amount, pct_change, is_cut
               FROM dividend_history WHERE ticker = ?
               ORDER BY ex_date DESC LIMIT ?""",
            (ticker.upper(), limit),
        ).fetchall()
        return [
            DividendRecord(
                ticker      = r["ticker"],
                ex_date     = date.fromisoformat(r["ex_date"]),
                record_date = date.fromisoformat(r["record_date"]) if r["record_date"] else None,
                pay_date    = date.fromisoformat(r["pay_date"])    if r["pay_date"]    else None,
                amount      = r["amount"],
                div_type    = r["div_type"] or "regular",
                source      = r["source"] or "unknown",
                prev_amount = r["prev_amount"],
                pct_change  = r["pct_change"],
                is_cut      = bool(r["is_cut"]),
            )
            for r in rows
        ]
    finally:
        conn.close()


def _query_splits(ticker: str) -> List[SplitRecord]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, ex_date, ratio_new, ratio_old, factor, split_type,
                      source, verified_8937, notes
               FROM split_history WHERE ticker = ?
               ORDER BY ex_date DESC""",
            (ticker.upper(),),
        ).fetchall()
        return [
            SplitRecord(
                ticker        = r["ticker"],
                ex_date       = date.fromisoformat(r["ex_date"]),
                ratio_new     = r["ratio_new"] or 1,
                ratio_old     = r["ratio_old"] or 1,
                factor        = r["factor"] or 1.0,
                split_type    = r["split_type"] or "forward",
                source        = r["source"] or "unknown",
                verified_8937 = bool(r["verified_8937"]),
                notes         = r["notes"],
            )
            for r in rows
        ]
    finally:
        conn.close()


def _query_spinoffs(ticker: str) -> List[SpinoffRecord]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, ex_date, spinoff_ticker, distribution_ratio,
                      source, filing_date, accession, description
               FROM spinoff_history WHERE ticker = ?
               ORDER BY filing_date DESC""",
            (ticker.upper(),),
        ).fetchall()
        return [
            SpinoffRecord(
                ticker             = r["ticker"],
                ex_date            = date.fromisoformat(r["ex_date"]) if r["ex_date"] else None,
                spinoff_ticker     = r["spinoff_ticker"],
                distribution_ratio = r["distribution_ratio"],
                source             = r["source"] or "unknown",
                filing_date        = date.fromisoformat(r["filing_date"]) if r["filing_date"] else None,
                accession          = r["accession"],
                description        = r["description"],
            )
            for r in rows
        ]
    finally:
        conn.close()


def _query_all_history(ticker: str) -> List[CorporateActionRecord]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, action_type, ex_date, record_date, pay_date,
                      amount, ratio, factor, source, notes, filing_date, accession
               FROM corporate_actions WHERE ticker = ?
               ORDER BY ex_date DESC LIMIT 500""",
            (ticker.upper(),),
        ).fetchall()
        results = [
            CorporateActionRecord(
                ticker      = r["ticker"],
                action_type = r["action_type"],
                ex_date     = date.fromisoformat(r["ex_date"]) if r["ex_date"] else None,
                record_date = date.fromisoformat(r["record_date"]) if r["record_date"] else None,
                pay_date    = date.fromisoformat(r["pay_date"])    if r["pay_date"]    else None,
                amount      = r["amount"],
                ratio       = r["ratio"],
                factor      = r["factor"],
                source      = r["source"] or "unknown",
                notes       = r["notes"],
                filing_date = date.fromisoformat(r["filing_date"]) if r["filing_date"] else None,
                accession   = r["accession"],
            )
            for r in rows
        ]
        return results
    finally:
        conn.close()


def _query_adjustment_factor(ticker: str, query_date: date) -> Optional[AdjustmentFactor]:
    """Return the most recent adjustment factor record for a ticker as of query_date."""
    conn = _get_db()
    try:
        row = conn.execute(
            """SELECT ticker, as_of_date, factor, source, action_type
               FROM adjustment_factors
               WHERE ticker = ? AND as_of_date <= ?
               ORDER BY as_of_date DESC LIMIT 1""",
            (ticker.upper(), str(query_date)),
        ).fetchone()
        if not row:
            return None
        return AdjustmentFactor(
            ticker      = row["ticker"],
            as_of_date  = date.fromisoformat(row["as_of_date"]),
            factor      = row["factor"],
            source      = row["source"] or "unknown",
            action_type = row["action_type"] or "split",
        )
    finally:
        conn.close()


def _query_upcoming(days_ahead: int = 30) -> List[UpcomingAction]:
    """Return corporate actions with ex_date in the next N calendar days."""
    today  = date.today()
    cutoff = today + timedelta(days=days_ahead)
    conn   = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, action_type, ex_date, pay_date, amount, notes
               FROM corporate_actions
               WHERE ex_date >= ? AND ex_date <= ?
               ORDER BY ex_date ASC LIMIT 200""",
            (str(today), str(cutoff)),
        ).fetchall()

        div_rows = conn.execute(
            """SELECT ticker, 'dividend' AS action_type, ex_date, pay_date, amount, div_type AS notes
               FROM dividend_history
               WHERE ex_date >= ? AND ex_date <= ?
               ORDER BY ex_date ASC LIMIT 200""",
            (str(today), str(cutoff)),
        ).fetchall()

        results: List[UpcomingAction] = []
        for r in list(rows) + list(div_rows):
            results.append(UpcomingAction(
                ticker      = r["ticker"],
                action_type = r["action_type"],
                ex_date     = date.fromisoformat(r["ex_date"]) if r["ex_date"] else None,
                pay_date    = date.fromisoformat(r["pay_date"]) if r["pay_date"] else None,
                amount      = r["amount"],
                description = r["notes"],
            ))
        results.sort(key=lambda u: u.ex_date or date.max)
        return results
    finally:
        conn.close()


def _query_calendar(month: int, year: int) -> List[UpcomingAction]:
    """Return all corporate actions with ex_date in a given calendar month."""
    from calendar import monthrange
    last_day = monthrange(year, month)[1]
    first    = date(year, month, 1)
    last     = date(year, month, last_day)
    conn     = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, action_type, ex_date, pay_date, amount, notes
               FROM corporate_actions
               WHERE ex_date >= ? AND ex_date <= ?
               ORDER BY ex_date ASC""",
            (str(first), str(last)),
        ).fetchall()
        return [
            UpcomingAction(
                ticker      = r["ticker"],
                action_type = r["action_type"],
                ex_date     = date.fromisoformat(r["ex_date"]) if r["ex_date"] else None,
                pay_date    = date.fromisoformat(r["pay_date"]) if r["pay_date"] else None,
                amount      = r["amount"],
                description = r["notes"],
            )
            for r in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Initialise DB at import time
# ---------------------------------------------------------------------------

_init_db()

# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/corp-actions/v3", tags=["Corporate Actions v3"])


@router.get(
    "/history/{ticker}",
    response_model=List[CorporateActionRecord],
    summary="Full corporate action history for a ticker",
)
def get_history(
    ticker: str,
    refresh: bool = Query(False, description="Re-aggregate from all sources before returning"),
) -> List[CorporateActionRecord]:
    """Return every corporate action on record for the ticker.

    Set refresh=true to trigger a fresh multi-source aggregation (slow).
    """
    if refresh:
        try:
            aggregate_corporate_actions(ticker)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Aggregation failed: {exc}")
    records = _query_all_history(ticker)
    # Supplement with typed records from specialised tables
    div_recs = _query_dividends(ticker)
    for d in div_recs:
        records.append(CorporateActionRecord(
            ticker      = d.ticker,
            action_type = f"dividend_{d.div_type}",
            ex_date     = d.ex_date,
            record_date = d.record_date,
            pay_date    = d.pay_date,
            amount      = d.amount,
            source      = d.source,
        ))
    split_recs = _query_splits(ticker)
    for s in split_recs:
        records.append(CorporateActionRecord(
            ticker      = s.ticker,
            action_type = s.split_type,
            ex_date     = s.ex_date,
            ratio       = f"{s.ratio_new}:{s.ratio_old}",
            factor      = s.factor,
            source      = s.source,
            notes       = s.notes,
        ))
    records.sort(key=lambda r: r.ex_date or date.min, reverse=True)
    return records


@router.get(
    "/dividends/{ticker}",
    response_model=List[DividendRecord],
    summary="Dividend history with cut/suspension detection",
)
def get_dividends(
    ticker: str,
    limit:   int  = Query(200, ge=1, le=2000),
    refresh: bool = Query(False),
) -> List[DividendRecord]:
    if refresh:
        try:
            div_recs = _fetch_yf_dividends(ticker)
            _upsert_dividends(div_recs)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    return _query_dividends(ticker, limit=limit)


@router.get(
    "/splits/{ticker}",
    response_model=List[SplitRecord],
    summary="Split and reverse-split history with 8937 verification",
)
def get_splits(
    ticker:  str,
    refresh: bool = Query(False),
) -> List[SplitRecord]:
    if refresh:
        try:
            yf_splits = _fetch_yf_splits(ticker)
            _upsert_splits(yf_splits)
            cik = _ticker_to_cik(ticker)
            if cik:
                filings_8937   = _fetch_8937_filings(ticker, cik)
                verified_splits = _parse_8937_factors(filings_8937)
                _upsert_splits(verified_splits)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    return _query_splits(ticker)


@router.get(
    "/spinoffs/{ticker}",
    response_model=List[SpinoffRecord],
    summary="Spin-off and carve-out history from EDGAR",
)
def get_spinoffs(
    ticker:  str,
    refresh: bool = Query(False),
) -> List[SpinoffRecord]:
    if refresh:
        cik = _ticker_to_cik(ticker)
        if not cik:
            raise HTTPException(status_code=404, detail=f"CIK not found for {ticker}")
        try:
            spinoff_recs = _search_spinoffs_edgar(ticker, cik)
            _upsert_spinoffs(spinoff_recs)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    return _query_spinoffs(ticker)


@router.get(
    "/adjustment-factor/{ticker}/{as_of_date}",
    response_model=AdjustmentFactor,
    summary="Backward-adjustment factor for continuous-return calculation",
)
def get_adjustment_factor(
    ticker:     str,
    as_of_date: str,
) -> AdjustmentFactor:
    """Return the cumulative backward-adjustment factor for ticker as of the given date.

    Multiply this factor by the raw closing price to obtain a price comparable
    to today's price level (backward-adjusted, current-comparable convention).

    Date format: YYYY-MM-DD
    """
    try:
        query_date = date.fromisoformat(as_of_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="Date must be YYYY-MM-DD")

    # Fast path: query DB
    af = _query_adjustment_factor(ticker, query_date)
    if af:
        return af

    # Fallback: compute on the fly from splits in DB
    splits = _query_splits(ticker)
    if not splits:
        # Trigger aggregation and retry once
        try:
            aggregate_corporate_actions(ticker)
        except Exception:
            pass
        splits = _query_splits(ticker)

    factor = _get_cumulative_factor_for_date(ticker, query_date, splits)
    return AdjustmentFactor(
        ticker      = ticker.upper(),
        as_of_date  = query_date,
        factor      = factor,
        source      = "computed",
        action_type = "cumulative",
    )


@router.get(
    "/upcoming",
    response_model=List[UpcomingAction],
    summary="Upcoming corporate actions in the next N days",
)
def get_upcoming(
    days: int = Query(30, ge=1, le=365),
) -> List[UpcomingAction]:
    return _query_upcoming(days_ahead=days)


@router.get(
    "/calendar",
    response_model=List[UpcomingAction],
    summary="Corporate action calendar for a given month/year",
)
def get_calendar(
    year:  int = Query(..., ge=2000, le=2100),
    month: int = Query(..., ge=1, le=12),
) -> List[UpcomingAction]:
    return _query_calendar(month=month, year=year)


@router.post(
    "/aggregate/{ticker}",
    summary="Trigger full multi-source aggregation for a ticker",
)
def trigger_aggregation(ticker: str) -> Dict[str, Any]:
    """Run the complete aggregation pipeline synchronously and return a summary."""
    try:
        return aggregate_corporate_actions(ticker)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# M&A Analytics — deal quality, arb spread, accretion/dilution, outcome tracker
# ---------------------------------------------------------------------------

def score_deal_quality(
    current_price: float,
    acquisition_price: float,
    week_52_high: float,
    consideration_type: str,         # "cash" | "stock" | "mixed"
    is_cross_border: bool = False,
    acquirer_market_share_pct: float = 0.0,
    target_revenue: float = 0.0,
    deal_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute a multi-factor M&A deal quality score (0–100) and component sub-scores.

    Components:
    - premium_to_52w_high: how much the offer exceeds the 52-week high (>0 = rich premium)
    - synergy_multiple: deal_value / target_revenue (EV/Revenue proxy for synergy pricing)
    - deal_certainty: higher for all-cash (no financing / share price risk vs stock)
    - regulatory_risk: penalised for cross-border deals and high combined market share

    Returns a dict with each component score (0–100) and a composite score.
    """
    if current_price <= 0 or acquisition_price <= 0:
        raise ValueError("Prices must be positive")

    # Premium to 52-week high: negative means bid is below prior high (stale price)
    if week_52_high > 0:
        premium_to_52w_high = (acquisition_price / week_52_high - 1.0) * 100
    else:
        premium_to_52w_high = 0.0

    # Premium to 52w high score: 0 = at/below 52w high, 100 = 25%+ above
    p52_score = min(100.0, max(0.0, (premium_to_52w_high / 25.0) * 100.0))

    # Synergy multiple: EV/Revenue. Score 100 = multiple >= 5x (rich synergies implied)
    if target_revenue > 0 and deal_value > 0:
        synergy_multiple = deal_value / target_revenue
        synergy_score = min(100.0, (synergy_multiple / 5.0) * 100.0)
    else:
        synergy_multiple = None
        synergy_score = 50.0   # neutral when not provided

    # Deal certainty score: cash = 100, mixed = 70, stock = 40
    # Cash deals have no financing risk or exchange ratio risk
    consideration_lower = consideration_type.lower()
    if "cash" in consideration_lower and "stock" not in consideration_lower:
        deal_certainty_score = 100.0
    elif "mixed" in consideration_lower or (
        "cash" in consideration_lower and "stock" in consideration_lower
    ):
        deal_certainty_score = 70.0
    else:  # stock-for-stock
        deal_certainty_score = 40.0

    # Regulatory risk: penalise cross-border and high market share
    regulatory_risk_score = 100.0  # start at no-risk
    if is_cross_border:
        regulatory_risk_score -= 30.0   # CFIUS / foreign regulatory review
    if acquirer_market_share_pct >= 30:
        regulatory_risk_score -= 40.0   # near-monopoly concerns
    elif acquirer_market_share_pct >= 15:
        regulatory_risk_score -= 20.0   # moderate antitrust scrutiny
    regulatory_risk_score = max(0.0, regulatory_risk_score)

    # Composite: weighted average of four components
    # Certainty and regulatory_risk dominate (deal completion)
    composite = (
        0.20 * p52_score
        + 0.20 * synergy_score
        + 0.35 * deal_certainty_score
        + 0.25 * regulatory_risk_score
    )

    return {
        "composite_score": round(composite, 2),
        "premium_to_52w_high_pct": round(premium_to_52w_high, 4),
        "premium_to_52w_high_score": round(p52_score, 2),
        "synergy_multiple": round(synergy_multiple, 4) if synergy_multiple is not None else None,
        "synergy_score": round(synergy_score, 2),
        "deal_certainty_score": round(deal_certainty_score, 2),
        "regulatory_risk_score": round(regulatory_risk_score, 2),
        "is_cross_border": is_cross_border,
        "consideration_type": consideration_type,
    }


def compute_arb_spread(
    current_price: float,
    acquisition_price: float,
    annualise: bool = False,
    days_to_close: Optional[int] = None,
) -> Dict[str, float]:
    """
    M&A arbitrage spread: return available to risk-arb investors.

    arb_spread_pct = (acquisition_price / current_price - 1) * 100

    For cash deals: current_price should be the market price of the target.
    A positive spread means the market still prices in deal risk.
    A negative spread (deal premium eroded) signals anticipated failure.

    Optionally annualises the spread given expected days to close.
    """
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if acquisition_price <= 0:
        raise ValueError("acquisition_price must be positive")

    arb_spread_pct = (acquisition_price / current_price - 1.0) * 100.0
    result: Dict[str, float] = {
        "current_price": current_price,
        "acquisition_price": acquisition_price,
        "arb_spread_pct": round(arb_spread_pct, 6),
    }

    if annualise and days_to_close and days_to_close > 0:
        # Annualised: (1 + spread)^(365/days) - 1
        annualised_pct = ((1.0 + arb_spread_pct / 100.0) ** (365.0 / days_to_close) - 1.0) * 100.0
        result["days_to_close"] = float(days_to_close)
        result["annualised_arb_spread_pct"] = round(annualised_pct, 4)

    return result


def compute_eps_accretion_dilution(
    acquirer_eps: float,
    target_earnings: float,
    share_exchange_ratio: float,
    new_shares_issued: float = 0.0,
    acquirer_shares_outstanding: float = 1.0,
    financing_cost_after_tax: float = 0.0,
) -> Dict[str, float]:
    """
    Accretion/dilution model for stock-for-stock M&A deals.

    EPS accretion = (target_earnings / share_exchange_ratio) / acquirer_shares_outstanding
    Net: adds target_earnings, dilutes with new shares, subtracts financing cost.

    Parameters
    ----------
    acquirer_eps          : Acquirer's current earnings per share (diluted)
    target_earnings       : Target's total net income (same currency as acquirer EPS × shares)
    share_exchange_ratio  : Shares of acquirer per share of target offered
    new_shares_issued     : New acquirer shares issued to fund the deal (stock consideration)
    acquirer_shares_outstanding : Pre-deal diluted shares outstanding
    financing_cost_after_tax    : After-tax cost of any cash/debt financing used

    Returns
    -------
    pro_forma_eps, eps_change, eps_change_pct, is_accretive
    """
    if share_exchange_ratio <= 0:
        raise ValueError("share_exchange_ratio must be positive")
    if acquirer_shares_outstanding <= 0:
        raise ValueError("acquirer_shares_outstanding must be positive")

    # Current total acquirer earnings
    acquirer_total_earnings = acquirer_eps * acquirer_shares_outstanding

    # Pro-forma combined earnings
    combined_earnings = acquirer_total_earnings + target_earnings - financing_cost_after_tax

    # Pro-forma shares (including newly issued)
    pro_forma_shares = acquirer_shares_outstanding + new_shares_issued

    pro_forma_eps = combined_earnings / pro_forma_shares if pro_forma_shares > 0 else 0.0
    eps_change = pro_forma_eps - acquirer_eps
    eps_change_pct = (eps_change / acquirer_eps * 100.0) if acquirer_eps != 0 else 0.0

    # Simplified: target_earnings per acquirer share issued = target_earnings / share_exchange_ratio
    # This is the "EPS contribution" metric referenced in the spec
    eps_contribution_per_acquirer_share = (
        target_earnings / share_exchange_ratio if share_exchange_ratio > 0 else 0.0
    )

    return {
        "acquirer_eps": round(acquirer_eps, 6),
        "target_earnings": round(target_earnings, 6),
        "share_exchange_ratio": round(share_exchange_ratio, 6),
        "eps_contribution_per_acquirer_share": round(eps_contribution_per_acquirer_share, 6),
        "pro_forma_eps": round(pro_forma_eps, 6),
        "eps_change": round(eps_change, 6),
        "eps_change_pct": round(eps_change_pct, 4),
        "is_accretive": eps_change > 0,
    }


class DealOutcomeTracker:
    """
    Tracks M&A deal outcomes (closed/failed) and computes completion rates by deal type.

    Stores deal records in-memory (and optionally persists to SQLite via ma_actions table).
    """

    def __init__(self) -> None:
        self._deals: List[Dict[str, Any]] = []

    def record_deal(
        self,
        ticker: str,
        deal_type: str,          # "tender_offer" | "merger_completion" | "going_private"
        consideration_type: str, # "cash" | "stock" | "mixed"
        outcome: str,            # "closed" | "failed" | "pending"
        announced_date: Optional[date] = None,
        closed_date: Optional[date] = None,
    ) -> None:
        """Record a deal outcome."""
        self._deals.append({
            "ticker": ticker.upper(),
            "deal_type": deal_type,
            "consideration_type": consideration_type,
            "outcome": outcome,
            "announced_date": announced_date,
            "closed_date": closed_date,
        })

    def get_completion_rate(
        self,
        deal_type: Optional[str] = None,
        consideration_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute completion rate (closed / (closed + failed)) for historical deals.

        Filters by deal_type and/or consideration_type if provided.
        Pending deals are excluded from the denominator.
        """
        deals = self._deals
        if deal_type:
            deals = [d for d in deals if d["deal_type"] == deal_type]
        if consideration_type:
            deals = [d for d in deals if d["consideration_type"] == consideration_type]

        closed = sum(1 for d in deals if d["outcome"] == "closed")
        failed = sum(1 for d in deals if d["outcome"] == "failed")
        pending = sum(1 for d in deals if d["outcome"] == "pending")
        total_resolved = closed + failed

        completion_rate = (closed / total_resolved) if total_resolved > 0 else None

        return {
            "total_deals_tracked": len(deals),
            "closed": closed,
            "failed": failed,
            "pending": pending,
            "total_resolved": total_resolved,
            "completion_rate": round(completion_rate, 4) if completion_rate is not None else None,
            "completion_rate_pct": round(completion_rate * 100, 2) if completion_rate is not None else None,
        }

    def get_all_deals(self) -> List[Dict[str, Any]]:
        return list(self._deals)
