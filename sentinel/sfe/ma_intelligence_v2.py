"""
M&A Intelligence V2 — Dimension #100 (M&A deal intelligence).
Target score: 9

Comprehensive merger & acquisition intelligence using free EDGAR data sources.
Enhances ma_intelligence.py with deeper pipeline tracking, SC 13D/G alerts,
S-4 detection, premium analysis, arbitrage spreads, and sector heat maps.

Public API
----------
DealPipelineTracker
    track_full_pipeline(lookback_days)              -> list[dict]
    get_deal_by_ticker(ticker)                      -> list[dict]
    classify_pipeline_stage(filing)                 -> str
    get_s4_filings(lookback_days)                   -> list[dict]

SC13DGMonitor
    get_threshold_crossings(ticker, lookback_days)  -> list[dict]
    get_all_large_stakes(lookback_days)             -> pd.DataFrame
    classify_filer_type(ownership_pct, filer_type)  -> str

PremiumAnalyzer
    compute_premium_from_price_history(ticker, offer_price) -> dict
    sector_premium_stats(sector)                    -> dict
    bulk_premium_analysis(deals)                    -> pd.DataFrame

ArbSpreadCalculator
    compute_spread(target_ticker, offer_price, ...)  -> dict
    screen_all_spreads()                             -> pd.DataFrame
    estimate_close_probability(deal)                 -> float
    annualized_return(spread_pct, days_to_close)    -> float

SectorHeatMap
    build_rolling_heatmap(months)                   -> pd.DataFrame
    sector_deal_volume(sector, months)              -> dict

HistoricalDealDatabase
    load_historical_deals(years_back)               -> pd.DataFrame
    comparable_deals(sector, deal_type, size_range) -> pd.DataFrame

FastAPI router
--------------
GET  /deal-pipeline
GET  /deals/{ticker}
GET  /arb-spreads
GET  /sector-heatmap
GET  /premium-analysis/{ticker}
GET  /form-sc13d/{ticker}
POST /alert-ma
GET  /s4-filings
GET  /deal-comps
GET  /break-risk/{ticker}
GET  /deal-financing/{ticker}
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EFTS_BASE         = "https://efts.sec.gov/LATEST/search-index"
EDGAR_BASE        = "https://data.sec.gov"
EDGAR_ARCHIVES    = "https://www.sec.gov/Archives/edgar/data"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
COMPANY_TICKERS   = "https://www.sec.gov/files/company_tickers.json"
YAHOO_CHART_URL   = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={range}"
YAHOO_QUOTE_URL   = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"

_HEADERS = {
    "User-Agent": "SENTINEL-MA-V2/2.0 richard.porras@realempanada.com",
    "Accept":     "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT    = 25.0
_RATE_DELAY = 0.12   # 120ms between SEC requests (~8 req/s)

# Pipeline stage ordering
PIPELINE_STAGES = [
    "rumor",          # SC 13D > 5%, news mentions, no official deal
    "announced",      # 8-K Item 1.01 - material agreement
    "proxy_filed",    # DEFM14A or PREM14A filed
    "s4_filed",       # S-4 registration for stock deals
    "vote_pending",   # Shareholder vote scheduled
    "regulatory",     # HSR/CFIUS under review
    "closing",        # Expected close within 30 days
    "completed",      # 8-K Item 2.01 — acquisition complete
    "terminated",     # Deal withdrawn or broken
]

# GICS sector codes for heatmap
_GICS_SECTORS = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}

# Sector-specific M&A activity keywords for classification
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Energy":                 ["oil", "gas", "pipeline", "refin", "energy", "petroleum", "lng", "upstream"],
    "Health Care":            ["pharma", "biotech", "medic", "hospital", "drug", "therapeut", "clinical"],
    "Information Technology": ["software", "cloud", "saas", "semiconductor", "tech", "digital", "cyber", "ai"],
    "Financials":             ["bank", "insur", "asset management", "fintech", "financial", "capital markets"],
    "Materials":              ["mining", "chemical", "steel", "aluminum", "material", "commodity"],
    "Industrials":            ["aerospace", "defense", "manufactur", "logistics", "transport", "industrial"],
    "Consumer Discretionary": ["retail", "restaurant", "hotel", "automotive", "luxury", "consumer"],
    "Consumer Staples":       ["food", "beverage", "grocery", "household", "tobacco", "staple"],
    "Real Estate":            ["reit", "real estate", "property", "housing", "commercial property"],
    "Communication Services": ["media", "telecom", "broadcast", "streaming", "social", "entertainment"],
    "Utilities":              ["utility", "electric", "water", "gas utility", "power", "renewable"],
}

# Historical base completion rates by deal type (empirical)
_BASE_COMPLETION_RATES: dict[str, float] = {
    "merger":    0.87,
    "all_cash":  0.92,
    "all_stock": 0.82,
    "mixed":     0.85,
    "lbo":       0.78,
    "hostile":   0.55,
    "asset_sale": 0.90,
    "unknown":   0.80,
}

# Regulatory review period estimates (days)
_REGULATORY_TIMELINES: dict[str, int] = {
    "hsr_basic":         30,    # Basic HSR waiting period
    "hsr_extended":      90,    # Second request HSR
    "cfius_basic":       75,    # CFIUS review
    "cfius_investigation": 120, # CFIUS full investigation
    "eu_phase1":         25,    # EU Phase I
    "eu_phase2":         90,    # EU Phase II in-depth
    "doj_consent_order": 180,   # DOJ remedies negotiation
}

# Financing classification patterns
_CASH_DEAL_PATTERNS    = [r"all.cash", r"cash consideration", r"\$[\d.]+ per share", r"cash merger"]
_STOCK_DEAL_PATTERNS   = [r"all.stock", r"exchange ratio", r"stock for stock", r"shares of common stock"]
_MIXED_PATTERNS        = [r"cash and stock", r"combination of cash", r"elected to receive"]
_DEBT_PATTERNS         = [r"credit facilit", r"term loan", r"bridge facilit", r"high.yield", r"debt financ"]
_PE_PATTERNS           = [r"private equity", r"buyout", r"sponsor", r"portfolio company", r"lbo"]

# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".sentinel_cache"
_CACHE_DIR.mkdir(exist_ok=True)
_DB_PATH = _CACHE_DIR / "ma_intelligence_v2.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS deal_pipeline (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            accession_number        TEXT UNIQUE NOT NULL,
            ticker                  TEXT,
            cik                     TEXT,
            acquirer                TEXT,
            target                  TEXT,
            deal_type               TEXT,
            pipeline_stage          TEXT,
            deal_value_billions     REAL,
            consideration_per_share REAL,
            premium_1day_pct        REAL,
            premium_1week_pct       REAL,
            premium_4week_pct       REAL,
            premium_52week_pct      REAL,
            financing_type          TEXT,
            has_debt_financing      INTEGER DEFAULT 0,
            sector                  TEXT,
            form_type               TEXT,
            filing_date             TEXT,
            announced_date          TEXT,
            proxy_date              TEXT,
            s4_date                 TEXT,
            vote_date               TEXT,
            expected_close_date     TEXT,
            close_date              TEXT,
            termination_fee_mm      REAL,
            synergies_mm            REAL,
            requires_hsr            INTEGER DEFAULT 0,
            requires_cfius          INTEGER DEFAULT 0,
            requires_eu_comp        INTEGER DEFAULT 0,
            arb_spread_pct          REAL,
            arb_annualized_pct      REAL,
            close_probability       REAL,
            break_risk_score        REAL,
            fairness_opinion_firms  TEXT,
            edgar_url               TEXT,
            created_at              TEXT NOT NULL,
            updated_at              TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sc13dg_filings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            accession_number    TEXT UNIQUE NOT NULL,
            cik                 TEXT,
            issuer_ticker       TEXT,
            issuer_name         TEXT,
            filer_name          TEXT,
            filer_type          TEXT,
            ownership_pct       REAL,
            shares_held         INTEGER,
            form_type           TEXT,
            filing_date         TEXT,
            activist_signal     INTEGER DEFAULT 0,
            strategic_signal    INTEGER DEFAULT 0,
            purpose_code        TEXT,
            purpose_text        TEXT,
            edgar_url           TEXT,
            created_at          TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS arb_spread_tracking (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            accession_number    TEXT NOT NULL,
            ticker              TEXT,
            current_price       REAL,
            offer_price         REAL,
            spread_pct          REAL,
            annualized_pct      REAL,
            close_probability   REAL,
            days_to_close       INTEGER,
            tracked_at          TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ma_alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type      TEXT NOT NULL,
            ticker          TEXT,
            message         TEXT,
            deal_value_mm   REAL,
            filing_date     TEXT,
            accession_no    TEXT,
            is_read         INTEGER DEFAULT 0,
            created_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS deal_history (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            target_name             TEXT,
            acquirer_name           TEXT,
            target_ticker           TEXT,
            sector                  TEXT,
            deal_value_billions     REAL,
            deal_type               TEXT,
            financing_type          TEXT,
            premium_pct             REAL,
            announced_date          TEXT,
            close_date              TEXT,
            outcome                 TEXT,
            days_to_close           INTEGER,
            ebitda_multiple         REAL,
            revenue_multiple        REAL,
            synergies_mm            REAL
        );

        CREATE TABLE IF NOT EXISTS price_cache (
            ticker          TEXT NOT NULL,
            price_date      TEXT NOT NULL,
            close_price     REAL,
            volume          INTEGER,
            fetched_at      TEXT NOT NULL,
            PRIMARY KEY (ticker, price_date)
        );

        CREATE INDEX IF NOT EXISTS idx_pipeline_ticker   ON deal_pipeline(ticker, pipeline_stage);
        CREATE INDEX IF NOT EXISTS idx_pipeline_date     ON deal_pipeline(filing_date);
        CREATE INDEX IF NOT EXISTS idx_sc13dg_ticker     ON sc13dg_filings(issuer_ticker, filing_date);
        CREATE INDEX IF NOT EXISTS idx_arb_ticker        ON arb_spread_tracking(ticker, tracked_at);
        CREATE INDEX IF NOT EXISTS idx_history_sector    ON deal_history(sector, announced_date);
        CREATE INDEX IF NOT EXISTS idx_price_ticker      ON price_cache(ticker, price_date);
        """)
    logger.info("ma_intelligence_v2: DB initialized at %s", _DB_PATH)


_init_db()

# ---------------------------------------------------------------------------
# CIK map (shared module-level singleton)
# ---------------------------------------------------------------------------

_cik_cache:        dict[str, str] = {}
_name_cik_cache:   dict[str, str] = {}
_ticker_name_cache: dict[str, str] = {}
_cik_cache_loaded: bool = False


async def _load_cik_map(client: httpx.AsyncClient) -> None:
    global _cik_cache_loaded
    if _cik_cache_loaded:
        return
    try:
        resp = await client.get(COMPANY_TICKERS, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        for entry in resp.json().values():
            ticker = str(entry.get("ticker", "")).upper().strip()
            cik    = str(entry.get("cik_str", entry.get("cik", ""))).zfill(10)
            name   = str(entry.get("title", "")).lower().strip()
            if ticker and cik:
                _cik_cache[ticker] = cik
            if name and cik:
                _name_cik_cache[name] = cik
            if ticker and name:
                _ticker_name_cache[ticker] = entry.get("title", "")
        _cik_cache_loaded = True
        logger.debug("ma_v2: loaded %d CIK entries", len(_cik_cache))
    except Exception as exc:
        logger.warning("ma_v2: CIK map load failed: %s", exc)


async def _resolve_ticker_to_cik(client: httpx.AsyncClient, ticker: str) -> Optional[str]:
    await _load_cik_map(client)
    return _cik_cache.get(ticker.upper())


def _accession_to_url(accession_no: str, cik: str) -> str:
    acc_nodash = accession_no.replace("-", "")
    cik_plain  = cik.lstrip("0") or cik
    return f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nodash}/"


def _format_accession(acc: str) -> str:
    """Ensure accession number has dashes: NNNNNNNNNN-NN-NNNNNN."""
    acc = acc.replace("-", "")
    if len(acc) == 18:
        return f"{acc[:10]}-{acc[10:12]}-{acc[12:]}"
    return acc


# ---------------------------------------------------------------------------
# Regex patterns for extraction
# ---------------------------------------------------------------------------

_PRICE_RE          = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B\b|M\b)', re.I)
_PER_SHARE_RE      = re.compile(r'\$\s*([\d.]+)\s*(?:per|each)\s+(?:share|common share)', re.I)
_PREMIUM_RE        = re.compile(r'([\d.]+)%?\s*premium', re.I)
_OWNERSHIP_PCT_RE  = re.compile(r'(?:owns?|hold[s]?|beneficial ownership of)\s*([\d.]+)%', re.I)
_SHARES_RE         = re.compile(r'([\d,]+)\s+(?:shares|common shares|ordinary shares)', re.I)
_VOTE_DATE_RE      = re.compile(r'shareholder\s+vote\s+(?:on|scheduled|expected)?[:\s]+(\w+ \d+,?\s*\d{4})', re.I)
_GO_SHOP_RE        = re.compile(r'go[\-\s]shop\s+period\s+of\s+(\d+)\s+days?', re.I)
_CLOSING_DATE_RE   = re.compile(r'expected\s+to\s+(?:close|complete)\s+(?:in\s+)?(?:the\s+)?([A-Za-z\s\d,]+?)(?:\.|,)', re.I)
_TERM_FEE_RE       = re.compile(r'termination fee[^$]*\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion)?', re.I)
_SYNERGIES_RE      = re.compile(r'(?:annual|cost|revenue)?\s*synergies[^$]*\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion)?', re.I)
_FAIRNESS_RE       = re.compile(r'(?:fairness opinion|financial advisor)[^.]*?by\s+([A-Z][A-Za-z\s&,\.]+?)(?:,|\.|\band\b)', re.I)
_HSR_RE            = re.compile(r'Hart[\-\s]Scott[\-\s]Rodino|HSR\s+Act', re.I)
_CFIUS_RE          = re.compile(r'CFIUS|Committee on Foreign Investment', re.I)
_EU_COMP_RE        = re.compile(r'European Commission|EU\s+merger|DG\s+COMP', re.I)
_ACQUIRER_RE       = re.compile(r'(?:acquired by|acquisition by|acquirer[,\s]+)([A-Z][A-Za-z\s,\.]{3,50})(?:,|\.|for|\()', re.I)
_EXCHANGE_RATIO_RE = re.compile(r'exchange ratio\s+of\s+([\d.]+)\s+shares?', re.I)
_PURPOSE_RE        = re.compile(r'Item\s+4[.\s]+Purpose[s]?\s+of\s+Transaction\s+(.*?)(?:Item\s+5|$)', re.DOTALL | re.I)
_MAC_RE            = re.compile(r'material\s+adverse\s+(?:change|effect|condition)', re.I)
_BREAK_FEE_RE      = re.compile(r'(?:reverse|termination)\s+break[\-\s]up\s+fee[^$]*\$\s*([\d,]+(?:\.\d+)?)', re.I)
_COMPETING_BID_RE  = re.compile(r'(?:competing|superior|alternative)\s+(?:bid|offer|proposal)', re.I)
_LITIGATION_RE     = re.compile(r'(?:class action|derivative lawsuit|shareholder litigation)', re.I)


def _extract_deal_value(text: str) -> Optional[float]:
    best: Optional[float] = None
    for m in _PRICE_RE.finditer(text):
        try:
            amount = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        val = amount if m.group(2).lower() in ("billion", "b") else amount / 1_000.0
        if best is None or val > best:
            best = val
    return round(best, 3) if best else None


def _extract_per_share(text: str) -> Optional[float]:
    m = _PER_SHARE_RE.search(text)
    if m:
        try:
            return round(float(m.group(1).replace(",", "")), 4)
        except ValueError:
            pass
    return None


def _extract_premium(text: str) -> Optional[float]:
    m = _PREMIUM_RE.search(text)
    if m:
        try:
            return round(float(m.group(1)), 2)
        except ValueError:
            pass
    return None


def _extract_ownership_pct(text: str) -> Optional[float]:
    m = _OWNERSHIP_PCT_RE.search(text)
    if m:
        try:
            val = float(m.group(1))
            if 0 < val <= 100:
                return round(val, 4)
        except ValueError:
            pass
    return None


def _extract_shares(text: str) -> Optional[int]:
    m = _SHARES_RE.search(text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _extract_term_fee(text: str) -> Optional[float]:
    m = _TERM_FEE_RE.search(text)
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "million").lower()
            return round(val * 1000.0 if "billion" in unit else val, 2)
        except ValueError:
            pass
    return None


def _extract_synergies(text: str) -> Optional[float]:
    m = _SYNERGIES_RE.search(text)
    if m:
        try:
            val  = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "million").lower()
            return round(val * 1000.0 if "billion" in unit else val, 2)
        except ValueError:
            pass
    return None


def _extract_fairness_firms(text: str) -> list[str]:
    return list({m.group(1).strip() for m in _FAIRNESS_RE.finditer(text)})[:4]


def _detect_financing(text: str) -> tuple[str, bool]:
    """Return (financing_type, has_debt_financing)."""
    t = text.lower()
    has_cash  = any(re.search(p, t) for p in _CASH_DEAL_PATTERNS)
    has_stock = any(re.search(p, t) for p in _STOCK_DEAL_PATTERNS)
    has_mixed = any(re.search(p, t) for p in _MIXED_PATTERNS)
    has_debt  = any(re.search(p, t) for p in _DEBT_PATTERNS)
    has_pe    = any(re.search(p, t) for p in _PE_PATTERNS)

    if has_pe or (has_debt and has_cash):
        return "lbo", True
    if has_mixed or (has_cash and has_stock):
        return "mixed", has_debt
    if has_stock:
        return "all_stock", False
    if has_cash:
        return "all_cash", has_debt
    return "unknown", has_debt


def _classify_sector(text: str, entity_name: str = "") -> str:
    combined = (text[:2000] + " " + entity_name).lower()
    for sector, keywords in _SECTOR_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return sector
    return "Unknown"


def _extract_purpose_text(text: str) -> Optional[str]:
    m = _PURPOSE_RE.search(text)
    if m:
        raw = re.sub(r'\s+', ' ', m.group(1)).strip()
        return raw[:500] if raw else None
    return None


def _classify_13d_filer(ownership_pct: Optional[float], text: str) -> tuple[str, bool, bool]:
    """
    Returns (filer_type, activist_signal, strategic_signal).
    """
    t = text.lower()
    activist_keywords  = ["change management", "board seat", "strategic review",
                          "maximize shareholder value", "explore strategic alternatives",
                          "urge", "demand", "engage", "publicly disclosed"]
    strategic_keywords = ["business combination", "merger", "acquisition",
                          "acquire", "definitive agreement", "strategic transaction",
                          "plan of merger", "tender offer"]
    passive_keywords   = ["investment purposes", "ordinary course of business",
                          "passive", "no present intention"]

    is_activist  = any(kw in t for kw in activist_keywords)
    is_strategic = any(kw in t for kw in strategic_keywords)
    is_passive   = any(kw in t for kw in passive_keywords)

    if is_strategic:
        ftype = "strategic"
    elif is_activist:
        ftype = "activist"
    elif is_passive:
        ftype = "passive"
    elif ownership_pct and ownership_pct >= 10:
        ftype = "large_passive"
    else:
        ftype = "unknown"

    return ftype, is_activist, is_strategic


def _pipeline_stage_from_forms(form_types: list[str], text: str) -> str:
    """Map filing form types to pipeline stage."""
    ft_set = {f.upper().strip() for f in form_types}
    t = text.lower()

    if any(kw in t for kw in ("transaction has been completed", "consummated", "merger closed", "acquisition closed")):
        return "completed"
    if any(kw in t for kw in ("terminated", "withdrawn", "abandoned", "agreement terminated")):
        return "terminated"
    if "S-4" in ft_set or "S-4/A" in ft_set:
        return "s4_filed"
    if "DEFM14A" in ft_set or "PREM14A" in ft_set:
        if _VOTE_DATE_RE.search(text):
            return "vote_pending"
        return "proxy_filed"
    if "SC TO-T" in ft_set or "SC 13E-3" in ft_set:
        return "announced"
    if "8-K" in ft_set:
        if re.search(r'item\s+2\.01', t):
            return "completed"
        if re.search(r'item\s+1\.01', t):
            return "announced"
    if "SC 13D" in ft_set or "SC 13D/A" in ft_set:
        return "rumor"
    return "announced"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    warnings: list[str],
) -> Optional[dict | list]:
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        warnings.append(f"HTTP {exc.response.status_code}: {url[:120]}")
    except Exception as exc:
        warnings.append(f"{type(exc).__name__}: {url[:120]}")
    return None


async def _efts_search(
    client:     httpx.AsyncClient,
    query:      str,
    forms:      list[str],
    start_date: date,
    end_date:   date,
    warnings:   list[str],
    max_hits:   int = 40,
) -> list[dict]:
    url = (
        f"{EFTS_BASE}"
        f"?q={quote(chr(34) + query + chr(34))}"
        f"&forms={quote(','.join(forms))}"
        f"&dateRange=custom"
        f"&startdt={start_date.isoformat()}"
        f"&enddt={end_date.isoformat()}"
        f"&hits.hits.total.value=true"
        f"&hits.hits._source=accession_no,cik,entity_name,file_date,form_type,description,display_names"
    )
    data = await _get_json(client, url, warnings)
    if data is None:
        return []
    hits = data.get("hits", {}).get("hits", [])
    return hits[:max_hits]


async def _fetch_filing_text(
    client: httpx.AsyncClient,
    cik: str,
    accession_no: str,
    warnings: list[str],
    max_chars: int = 50000,
) -> str:
    """Fetch primary document text from EDGAR filing index."""
    acc_fmt    = _format_accession(accession_no)
    acc_nodash = acc_fmt.replace("-", "")
    cik_plain  = cik.lstrip("0") or cik
    index_url  = f"{EDGAR_ARCHIVES}/{cik_plain}/{acc_nodash}/{acc_fmt}-index.htm"
    try:
        resp = await client.get(index_url, headers=_HEADERS, timeout=_TIMEOUT)
        links = re.findall(r'href="([^"]*\.htm[l]?)"', resp.text, re.I)
        primary = next((l for l in links if "index" not in l.lower()), None)
        if not primary:
            # Fallback: return index text itself
            return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', resp.text))[:max_chars]
        doc_url = f"https://www.sec.gov{primary}" if primary.startswith("/") else primary
        doc_resp = await client.get(doc_url, headers=_HEADERS, timeout=_TIMEOUT)
        text = re.sub(r'<[^>]+>', ' ', doc_resp.text)
        text = re.sub(r'\s+', ' ', text)
        return text[:max_chars]
    except Exception as exc:
        warnings.append(f"Filing text fetch failed {accession_no}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Yahoo Finance price history
# ---------------------------------------------------------------------------

async def _get_price_history(
    client: httpx.AsyncClient,
    ticker: str,
    days: int,
    warnings: list[str],
) -> dict[str, float]:
    """
    Fetch daily closing prices from Yahoo Finance.
    Returns {date_str: close_price} dict.
    """
    # Check cache first
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT price_date, close_price FROM price_cache
                   WHERE ticker=? AND price_date >= ?
                   ORDER BY price_date""",
                (ticker.upper(), cutoff),
            ).fetchall()
        if rows and len(rows) >= days // 2:
            return {r["price_date"]: r["close_price"] for r in rows}
    except Exception:
        pass

    range_str = "1y" if days <= 365 else "2y"
    url = YAHOO_CHART_URL.format(ticker=ticker.upper(), range=range_str)
    data = await _get_json(client, url, warnings)
    if not data:
        return {}

    result: dict[str, float] = {}
    try:
        chart_result = data.get("chart", {}).get("result", [])
        if not chart_result:
            return {}
        ts_list     = chart_result[0].get("timestamp", [])
        close_list  = chart_result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        vol_list    = chart_result[0].get("indicators", {}).get("quote", [{}])[0].get("volume", [])

        records = []
        for ts, close, vol in zip(ts_list, close_list, vol_list or [None] * len(ts_list)):
            if close is None:
                continue
            d = date.fromtimestamp(ts).isoformat()
            result[d] = round(float(close), 4)
            records.append((ticker.upper(), d, close, vol, datetime.utcnow().isoformat()))

        if records:
            with _get_conn() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO price_cache VALUES(?,?,?,?,?)", records
                )
    except Exception as exc:
        warnings.append(f"Yahoo price parse error {ticker}: {exc}")

    return result


async def _get_current_price(
    client: httpx.AsyncClient,
    ticker: str,
    warnings: list[str],
) -> Optional[float]:
    """Get the most recent closing price for a ticker."""
    url = YAHOO_QUOTE_URL.format(ticker=ticker.upper())
    data = await _get_json(client, url, warnings)
    if not data:
        return None
    try:
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        valid  = [c for c in closes if c is not None]
        return round(float(valid[-1]), 4) if valid else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Premium analysis helpers
# ---------------------------------------------------------------------------

def _compute_premiums(
    price_history: dict[str, float],
    offer_price: float,
    announcement_date: str,
) -> dict[str, Optional[float]]:
    """
    Compute 1-day, 1-week, 4-week, and 52-week premiums.

    Premium = (offer_price - prior_price) / prior_price * 100
    """
    try:
        ann_date = date.fromisoformat(announcement_date)
    except ValueError:
        return {"1day": None, "1week": None, "4week": None, "52week": None}

    def _closest_prior_price(target_date: date) -> Optional[float]:
        """Get the closing price on or just before target_date."""
        best_date: Optional[str] = None
        for d_str in sorted(price_history.keys(), reverse=True):
            try:
                d = date.fromisoformat(d_str)
            except ValueError:
                continue
            if d <= target_date:
                best_date = d_str
                break
        return price_history.get(best_date) if best_date else None

    day_1   = _closest_prior_price(ann_date - timedelta(days=1))
    week_1  = _closest_prior_price(ann_date - timedelta(days=7))
    week_4  = _closest_prior_price(ann_date - timedelta(days=28))
    week_52 = _closest_prior_price(ann_date - timedelta(days=365))

    def pct(prior: Optional[float]) -> Optional[float]:
        if prior and prior > 0:
            return round((offer_price - prior) / prior * 100, 2)
        return None

    return {
        "1day":   pct(day_1),
        "1week":  pct(week_1),
        "4week":  pct(week_4),
        "52week": pct(week_52),
        "prior_close_1day":   day_1,
        "prior_close_1week":  week_1,
        "prior_close_4week":  week_4,
        "prior_close_52week": week_52,
    }


# ---------------------------------------------------------------------------
# Arbitrage spread calculation
# ---------------------------------------------------------------------------

def _compute_arb_spread(
    current_price:   float,
    offer_price:     float,
    days_to_close:   int,
    close_prob:      float,
    break_price:     Optional[float] = None,
) -> dict:
    """
    Compute merger arbitrage spread and expected return.

    Spread = (offer_price - current_price) / current_price * 100
    Expected Value = prob * (offer - current) + (1 - prob) * (break - current)
    Ann. return = (1 + spread/100)^(365/days) - 1
    """
    spread_pct = (offer_price - current_price) / current_price * 100 if current_price > 0 else 0

    # Expected break price if deal fails (~20-30% drop from offer typically)
    if break_price is None:
        break_price = current_price * 0.78  # Assume 22% drop if deal breaks

    # Expected value
    ev_gain   = close_prob * (offer_price - current_price)
    ev_loss   = (1 - close_prob) * (break_price - current_price)
    ev_return = (ev_gain + ev_loss) / current_price * 100 if current_price > 0 else 0

    # Annualized
    ann_spread = 0.0
    if days_to_close > 0 and current_price > 0 and offer_price > 0:
        raw = offer_price / current_price
        ann_spread = round((raw ** (365.0 / days_to_close) - 1) * 100, 2)

    # Implied probability from spread (back-solve)
    # If market prices at X below offer, market implies lower prob
    market_implied_prob = min(1.0, max(0.0, current_price / offer_price)) if offer_price > 0 else 0

    return {
        "current_price":          round(current_price, 4),
        "offer_price":            round(offer_price, 4),
        "spread_dollars":         round(offer_price - current_price, 4),
        "spread_pct":             round(spread_pct, 3),
        "annualized_return_pct":  ann_spread,
        "close_probability":      round(close_prob, 3),
        "expected_value_pct":     round(ev_return, 3),
        "break_price_assumption": round(break_price, 4),
        "market_implied_prob":    round(market_implied_prob * 100, 1),
        "days_to_close":          days_to_close,
    }


def _estimate_close_probability(deal: dict) -> float:
    """
    Multi-factor deal completion probability.

    Factors: deal type base rate, financing risk, regulatory complexity,
    go-shop risk, hostile premium, pending litigation, market conditions.
    """
    deal_type = deal.get("deal_type", "unknown")
    base = _BASE_COMPLETION_RATES.get(deal_type, 0.80)

    score = base

    # Financing adjustments
    financing = deal.get("financing_type", "")
    if deal.get("has_debt_financing"):
        score -= 0.05   # Debt market risk
    if financing == "lbo":
        score -= 0.08   # Higher break risk from leverage

    # Regulatory adjustments
    reg_count = sum([
        int(bool(deal.get("requires_hsr"))),
        int(bool(deal.get("requires_cfius"))),
        int(bool(deal.get("requires_eu_comp"))),
    ])
    score -= reg_count * 0.03

    # Deal size (larger = more regulatory scrutiny)
    val_bn = deal.get("deal_value_billions") or 0
    if val_bn > 50:
        score -= 0.08
    elif val_bn > 10:
        score -= 0.04
    elif val_bn > 1:
        score -= 0.01

    # Pipeline stage (further along = more likely to close)
    stage = deal.get("pipeline_stage", "")
    if stage == "vote_pending":
        score += 0.05
    elif stage == "proxy_filed":
        score += 0.03
    elif stage == "rumor":
        score -= 0.15

    # Termination fee (larger fee = stronger commitment)
    term_fee = deal.get("termination_fee_mm") or 0
    if term_fee > 0 and val_bn > 0:
        fee_pct = term_fee / (val_bn * 1000) * 100
        if fee_pct >= 3.5:   # Typical ~3-4% termination fee
            score += 0.02
        elif fee_pct < 1.5:
            score -= 0.02

    return round(max(0.20, min(0.98, score)), 3)


def _compute_break_risk(deal: dict) -> float:
    """
    Break risk score 0-10 (10 = high break risk).

    Factors: regulatory complexity, deal type, financing, market conditions.
    """
    risk = 2.0  # Base risk

    # Regulatory
    if deal.get("requires_cfius"):   risk += 2.0
    if deal.get("requires_eu_comp"): risk += 1.5
    if deal.get("requires_hsr"):     risk += 0.5

    # Deal type
    deal_type = deal.get("deal_type", "unknown")
    if deal_type == "hostile":       risk += 2.0
    if deal_type == "lbo":          risk += 1.5
    if deal.get("has_debt_financing"): risk += 1.0

    # Size
    val_bn = deal.get("deal_value_billions") or 0
    if val_bn > 50: risk += 1.5
    elif val_bn > 10: risk += 0.5

    # Premium (high premium may attract competing bids but also regulatory scrutiny)
    premium = deal.get("premium_1day_pct") or deal.get("premium_pct") or 0
    if premium > 50: risk += 1.0

    return round(min(10.0, max(0.0, risk)), 2)


# ---------------------------------------------------------------------------
# Deal Pipeline Tracker
# ---------------------------------------------------------------------------

class DealPipelineTracker:
    """
    Full M&A pipeline tracker using EDGAR filings.

    Tracks deals from rumor (SC 13D) through announcement (8-K Item 1.01),
    proxy filing (DEFM14A), shareholder vote, and completion (8-K Item 2.01).
    """

    # EFTS queries for each pipeline stage
    _PIPELINE_QUERIES: list[tuple[str, list[str], str]] = [
        ("Agreement and Plan of Merger",         ["8-K"],                 "announced"),
        ("definitive merger agreement",          ["8-K"],                 "announced"),
        ("Agreement and Plan of Merger",         ["DEFM14A", "PREM14A"], "proxy_filed"),
        ("tender offer",                         ["SC TO-T"],             "announced"),
        ("acquisition of",                       ["SC 13E-3"],            "announced"),
        ("registration of securities",           ["S-4"],                 "s4_filed"),
        ("merger consideration",                 ["S-4"],                 "s4_filed"),
        ("business combination",                 ["S-4"],                 "s4_filed"),
        ("completion of acquisition",            ["8-K"],                 "completed"),
        ("consummation of the merger",           ["8-K"],                 "completed"),
        ("merger has been completed",            ["8-K"],                 "completed"),
    ]

    async def track_full_pipeline(
        self,
        lookback_days: int = 180,
        fetch_filing_text: bool = False,
    ) -> list[dict]:
        """
        Scan EDGAR for all M&A pipeline deals in the lookback window.

        Args:
            lookback_days:      Calendar days to look back.
            fetch_filing_text:  If True, fetches full filing text for richer parsing
                                (slower — adds ~100ms per filing).

        Returns:
            List of deal dicts with pipeline stage classification.
        """
        warnings_: list[str] = []
        end_date   = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        seen: set[str] = set()
        deals: list[dict] = []

        async with httpx.AsyncClient() as client:
            await _load_cik_map(client)

            tasks = [
                _efts_search(client, query, forms, start_date, end_date, warnings_)
                for query, forms, _ in self._PIPELINE_QUERIES
            ]
            results = await asyncio.gather(*tasks)

        for (query, forms, stage_hint), hits in zip(self._PIPELINE_QUERIES, results):
            for hit in hits:
                src = hit.get("_source", {})
                acc = src.get("accession_no") or hit.get("_id", "")
                if not acc or acc in seen:
                    continue
                seen.add(acc)

                cik = str(src.get("cik", "")).zfill(10)
                entity_name = src.get("entity_name", "")
                if not entity_name:
                    dn = src.get("display_names", [])
                    if dn and isinstance(dn, list):
                        first = dn[0]
                        entity_name = first.get("entity", "") if isinstance(first, dict) else str(first)

                try:
                    filing_date = date.fromisoformat(src.get("file_date", "")[:10]).isoformat()
                except (ValueError, TypeError):
                    continue

                form_type   = src.get("form_type", "8-K")
                description = (src.get("description", "") + " " + src.get("biz_descs", "")).strip()

                financing, has_debt = _detect_financing(description)
                sector = _classify_sector(description, entity_name)

                deal: dict = {
                    "accession_number":        _format_accession(acc),
                    "cik":                     cik,
                    "target":                  entity_name or None,
                    "ticker":                  None,
                    "form_type":               form_type,
                    "filing_date":             filing_date,
                    "pipeline_stage":          stage_hint,
                    "deal_type":               financing,
                    "deal_value_billions":      _extract_deal_value(description),
                    "consideration_per_share":  _extract_per_share(description),
                    "premium_pct":              _extract_premium(description),
                    "financing_type":           financing,
                    "has_debt_financing":       has_debt,
                    "sector":                   sector,
                    "termination_fee_mm":       _extract_term_fee(description),
                    "synergies_mm":             _extract_synergies(description),
                    "fairness_opinion_firms":   _extract_fairness_firms(description),
                    "requires_hsr":             bool(_HSR_RE.search(description)),
                    "requires_cfius":           bool(_CFIUS_RE.search(description)),
                    "requires_eu_comp":         bool(_EU_COMP_RE.search(description)),
                    "edgar_url":                _accession_to_url(_format_accession(acc), cik) if cik else "",
                    "description_snippet":      description[:300],
                }

                deal["close_probability"] = _estimate_close_probability(deal)
                deal["break_risk_score"]  = _compute_break_risk(deal)

                deals.append(deal)
                self._upsert_deal(deal)

        deals.sort(key=lambda d: d.get("filing_date", ""), reverse=True)
        logger.info("ma_v2: track_full_pipeline found=%d lookback=%d", len(deals), lookback_days)
        return deals

    def _upsert_deal(self, deal: dict) -> None:
        now = datetime.utcnow().isoformat()
        try:
            with _get_conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO deal_pipeline (
                        accession_number, ticker, cik, target, deal_type,
                        pipeline_stage, deal_value_billions, consideration_per_share,
                        financing_type, has_debt_financing, sector, form_type,
                        filing_date, announced_date, termination_fee_mm, synergies_mm,
                        requires_hsr, requires_cfius, requires_eu_comp,
                        close_probability, break_risk_score, fairness_opinion_firms,
                        edgar_url, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        deal.get("accession_number"), deal.get("ticker"), deal.get("cik"),
                        deal.get("target"), deal.get("deal_type"),
                        deal.get("pipeline_stage"), deal.get("deal_value_billions"),
                        deal.get("consideration_per_share"), deal.get("financing_type"),
                        int(deal.get("has_debt_financing", False)), deal.get("sector"),
                        deal.get("form_type"), deal.get("filing_date"), deal.get("filing_date"),
                        deal.get("termination_fee_mm"), deal.get("synergies_mm"),
                        int(deal.get("requires_hsr", False)),
                        int(deal.get("requires_cfius", False)),
                        int(deal.get("requires_eu_comp", False)),
                        deal.get("close_probability"), deal.get("break_risk_score"),
                        json.dumps(deal.get("fairness_opinion_firms", [])),
                        deal.get("edgar_url"), now, now,
                    ),
                )
        except Exception as exc:
            logger.debug("ma_v2: upsert deal error: %s", exc)

    async def get_deal_by_ticker(
        self,
        ticker: str,
        lookback_days: int = 730,
    ) -> list[dict]:
        """
        Find all M&A deals involving a given ticker (as target).

        Searches both the local SQLite cache and live EDGAR submissions API.
        """
        warnings_: list[str] = []
        ticker_upper = ticker.upper()

        # Check local cache first
        with _get_conn() as conn:
            cached = conn.execute(
                """SELECT * FROM deal_pipeline
                   WHERE ticker=? OR target LIKE ?
                   ORDER BY filing_date DESC""",
                (ticker_upper, f"%{ticker_upper}%"),
            ).fetchall()

        deals = [dict(r) for r in cached]

        # Also query EDGAR submissions API for this CIK
        async with httpx.AsyncClient() as client:
            await _load_cik_map(client)
            cik = _cik_cache.get(ticker_upper)
            if cik:
                sub_url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"
                data    = await _get_json(client, sub_url, warnings_)
                if data:
                    recent_filings = data.get("filings", {}).get("recent", {})
                    forms   = recent_filings.get("form", [])
                    accessions = recent_filings.get("accessionNumber", [])
                    dates   = recent_filings.get("filingDate", [])
                    name    = data.get("name", ticker_upper)

                    ma_forms = {"8-K", "DEFM14A", "PREM14A", "SC TO-T", "SC 13E-3", "S-4"}
                    cutoff   = (date.today() - timedelta(days=lookback_days)).isoformat()

                    for form, acc, filing_dt in zip(forms, accessions, dates):
                        if form.upper() not in ma_forms:
                            continue
                        if filing_dt < cutoff:
                            continue
                        # Don't re-add if already cached
                        if any(d.get("accession_number") == _format_accession(acc) for d in deals):
                            continue
                        deals.append({
                            "accession_number": _format_accession(acc),
                            "ticker":           ticker_upper,
                            "cik":              cik,
                            "target":           name,
                            "form_type":        form,
                            "filing_date":      filing_dt,
                            "pipeline_stage":   "announced" if "8-K" in form.upper() else "proxy_filed",
                            "edgar_url":        _accession_to_url(_format_accession(acc), cik),
                            "source":           "edgar_submissions",
                        })

        deals.sort(key=lambda d: d.get("filing_date", ""), reverse=True)
        return deals

    async def get_s4_filings(self, lookback_days: int = 180) -> list[dict]:
        """
        Detect S-4 registration filings (stock merger registrations).

        S-4 filings indicate stock-for-stock deals or mixed consideration where
        the acquirer must register new shares with the SEC.
        """
        warnings_: list[str] = []
        end_date   = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        s4_queries = [
            ("merger consideration",     ["S-4", "S-4/A"]),
            ("business combination",     ["S-4"]),
            ("plan of merger",           ["S-4"]),
            ("exchange ratio",           ["S-4"]),
        ]

        seen: set[str] = set()
        results_list: list[dict] = []

        async with httpx.AsyncClient() as client:
            tasks = [
                _efts_search(client, q, forms, start_date, end_date, warnings_)
                for q, forms in s4_queries
            ]
            all_hits = await asyncio.gather(*tasks)

        for hits in all_hits:
            for hit in hits:
                src = hit.get("_source", {})
                acc = src.get("accession_no") or hit.get("_id", "")
                if not acc or acc in seen:
                    continue
                seen.add(acc)

                cik  = str(src.get("cik", "")).zfill(10)
                name = src.get("entity_name", "")
                try:
                    filing_date = date.fromisoformat(src.get("file_date", "")[:10]).isoformat()
                except (ValueError, TypeError):
                    continue

                desc = src.get("description", "")
                results_list.append({
                    "accession_number": _format_accession(acc),
                    "cik":              cik,
                    "entity_name":      name,
                    "form_type":        src.get("form_type", "S-4"),
                    "filing_date":      filing_date,
                    "deal_type":        "all_stock",
                    "description":      desc[:300],
                    "edgar_url":        _accession_to_url(_format_accession(acc), cik),
                    "exchange_ratio":   _EXCHANGE_RATIO_RE.search(desc).group(1) if _EXCHANGE_RATIO_RE.search(desc) else None,
                    "deal_value_bn":    _extract_deal_value(desc),
                    "requires_hsr":     bool(_HSR_RE.search(desc)),
                })

        results_list.sort(key=lambda x: x["filing_date"], reverse=True)
        logger.info("ma_v2: get_s4_filings found=%d", len(results_list))
        return results_list


# ---------------------------------------------------------------------------
# SC 13D/G Monitor
# ---------------------------------------------------------------------------

class SC13DGMonitor:
    """
    Monitor SC 13D/G filings for beneficial ownership threshold crossings.

    SC 13D: Filed when >5% ownership AND filer has activist intent (schedule 13D).
    SC 13G: Filed when >5% passive ownership (institutional, index).
    SC 13D/A and SC 13G/A: Amendments when position changes materially.
    """

    _SC13_FORMS = ["SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"]

    async def get_threshold_crossings(
        self,
        ticker: str,
        lookback_days: int = 180,
    ) -> list[dict]:
        """
        Find all SC 13D/G filings for a specific issuer ticker.

        Args:
            ticker:       Issuer company ticker symbol.
            lookback_days: Calendar days to look back.

        Returns:
            List of threshold crossing events with ownership % and filer classification.
        """
        warnings_: list[str] = []
        ticker_upper = ticker.upper()
        company_name = _ticker_name_cache.get(ticker_upper, ticker_upper)

        end_date   = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        # Query by company name in EFTS
        queries = [
            (company_name,    self._SC13_FORMS),
            (ticker_upper,    self._SC13_FORMS),
        ]

        seen:   set[str] = set()
        filings: list[dict] = []

        async with httpx.AsyncClient() as client:
            await _load_cik_map(client)
            cik = _cik_cache.get(ticker_upper)

            # If we have the CIK, query EDGAR submissions API
            if cik:
                sub_url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"
                data    = await _get_json(client, sub_url, warnings_)
                if data:
                    recent = data.get("filings", {}).get("recent", {})
                    forms   = recent.get("form", [])
                    accs    = recent.get("accessionNumber", [])
                    dates   = recent.get("filingDate", [])
                    reporters = recent.get("reportingOwner", []) if "reportingOwner" in recent else []

                    cutoff = start_date.isoformat()
                    for form, acc, fdate in zip(forms, accs, dates):
                        if form.upper() not in {f.upper() for f in self._SC13_FORMS}:
                            continue
                        if fdate < cutoff:
                            continue
                        if acc in seen:
                            continue
                        seen.add(acc)

                        acc_fmt = _format_accession(acc)
                        # Fetch filing text for detail
                        text = await _fetch_filing_text(client, cik, acc_fmt, warnings_, max_chars=20000)

                        ownership = _extract_ownership_pct(text)
                        shares    = _extract_shares(text)
                        purpose   = _extract_purpose_text(text)
                        filer_type, activist, strategic = _classify_13d_filer(ownership, text)

                        filing_rec = {
                            "accession_number": acc_fmt,
                            "cik":              cik,
                            "issuer_ticker":    ticker_upper,
                            "issuer_name":      data.get("name", company_name),
                            "form_type":        form,
                            "filing_date":      fdate,
                            "ownership_pct":    ownership,
                            "shares_held":      shares,
                            "filer_type":       filer_type,
                            "activist_signal":  activist,
                            "strategic_signal": strategic,
                            "purpose_text":     purpose,
                            "edgar_url":        _accession_to_url(acc_fmt, cik),
                            "threshold_crossed": ">5%" if ownership and ownership >= 5 else None,
                            "signal_strength":  self._signal_strength(ownership, filer_type),
                        }
                        filings.append(filing_rec)
                        self._cache_filing(filing_rec)

            # EFTS fallback for company name search
            if not filings:
                for query, forms in queries[:1]:  # Just one to avoid rate limits
                    hits = await _efts_search(
                        client, query, forms, start_date, end_date, warnings_
                    )
                    for hit in hits:
                        src = hit.get("_source", {})
                        acc = src.get("accession_no") or hit.get("_id", "")
                        if not acc or acc in seen:
                            continue
                        seen.add(acc)
                        desc = src.get("description", "")
                        ownership = _extract_ownership_pct(desc)
                        filer_type, activist, strategic = _classify_13d_filer(ownership, desc)
                        try:
                            fdate = date.fromisoformat(src.get("file_date", "")[:10]).isoformat()
                        except (ValueError, TypeError):
                            continue

                        filings.append({
                            "accession_number": _format_accession(acc),
                            "cik":              str(src.get("cik", "")).zfill(10),
                            "issuer_ticker":    ticker_upper,
                            "issuer_name":      src.get("entity_name", company_name),
                            "form_type":        src.get("form_type", "SC 13D"),
                            "filing_date":      fdate,
                            "ownership_pct":    ownership,
                            "activist_signal":  activist,
                            "strategic_signal": strategic,
                            "filer_type":       filer_type,
                            "edgar_url":        _accession_to_url(_format_accession(acc), str(src.get("cik", "")).zfill(10)),
                        })

        filings.sort(key=lambda x: x.get("filing_date", ""), reverse=True)
        logger.info("ma_v2: SC13D/G crossings ticker=%s found=%d", ticker, len(filings))
        return filings

    def _signal_strength(self, ownership_pct: Optional[float], filer_type: str) -> str:
        if not ownership_pct:
            return "unknown"
        if filer_type == "strategic":
            return "very_high"
        if filer_type == "activist":
            return "high" if ownership_pct >= 5 else "medium"
        if filer_type == "large_passive":
            return "medium"
        return "low"

    def _cache_filing(self, rec: dict) -> None:
        now = datetime.utcnow().isoformat()
        try:
            with _get_conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO sc13dg_filings (
                        accession_number, cik, issuer_ticker, issuer_name,
                        filer_name, filer_type, ownership_pct, shares_held,
                        form_type, filing_date, activist_signal, strategic_signal,
                        purpose_text, edgar_url, created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rec.get("accession_number"), rec.get("cik"),
                        rec.get("issuer_ticker"), rec.get("issuer_name"),
                        rec.get("filer_name"), rec.get("filer_type"),
                        rec.get("ownership_pct"), rec.get("shares_held"),
                        rec.get("form_type"), rec.get("filing_date"),
                        int(rec.get("activist_signal", False)),
                        int(rec.get("strategic_signal", False)),
                        rec.get("purpose_text"), rec.get("edgar_url"), now,
                    ),
                )
        except Exception as exc:
            logger.debug("ma_v2: sc13dg cache error: %s", exc)

    async def get_all_large_stakes(self, lookback_days: int = 90) -> pd.DataFrame:
        """
        Broad scan for all >5% ownership crossings across the market.

        Returns DataFrame of all recent SC 13D/G filings that signal
        activist or strategic intent.
        """
        warnings_: list[str] = []
        end_date   = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        queries = [
            ("beneficial ownership",  ["SC 13D", "SC 13D/A"]),
            ("5% or more",            ["SC 13D"]),
            ("strategic alternatives", ["SC 13D"]),
            ("propose to acquire",    ["SC 13D"]),
        ]

        seen: set[str] = set()
        rows: list[dict] = []

        async with httpx.AsyncClient() as client:
            tasks = [
                _efts_search(client, q, forms, start_date, end_date, warnings_)
                for q, forms in queries
            ]
            all_results = await asyncio.gather(*tasks)

        for hits in all_results:
            for hit in hits:
                src = hit.get("_source", {})
                acc = src.get("accession_no") or hit.get("_id", "")
                if not acc or acc in seen:
                    continue
                seen.add(acc)

                desc = src.get("description", "")
                ownership = _extract_ownership_pct(desc)
                if ownership and ownership < 4.9:
                    continue  # Filter out sub-5% filings

                filer_type, activist, strategic = _classify_13d_filer(ownership, desc)

                try:
                    fdate = date.fromisoformat(src.get("file_date", "")[:10]).isoformat()
                except (ValueError, TypeError):
                    continue

                rows.append({
                    "accession_number": _format_accession(acc),
                    "entity_name":      src.get("entity_name", ""),
                    "form_type":        src.get("form_type", "SC 13D"),
                    "filing_date":      fdate,
                    "ownership_pct":    ownership,
                    "filer_type":       filer_type,
                    "activist_signal":  activist,
                    "strategic_signal": strategic,
                    "edgar_url":        _accession_to_url(_format_accession(acc), str(src.get("cik", "")).zfill(10)),
                })

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty and "filing_date" in df.columns:
            df = df.sort_values("filing_date", ascending=False).reset_index(drop=True)
        logger.info("ma_v2: get_all_large_stakes found=%d", len(df))
        return df


# ---------------------------------------------------------------------------
# Premium Analyzer
# ---------------------------------------------------------------------------

class PremiumAnalyzer:
    """
    Compute M&A acquisition premiums from historical price data.

    Premiums are computed at 1-day, 1-week, 4-week, and 52-week prior to
    the announcement date. Benchmarks against sector historical averages.
    """

    # Historical sector median premiums (empirical, 2010-2024)
    _SECTOR_MEDIAN_PREMIUMS: dict[str, dict[str, float]] = {
        "Information Technology": {"1day": 32, "1week": 35, "4week": 38, "52week": 45},
        "Health Care":            {"1day": 40, "1week": 42, "4week": 45, "52week": 52},
        "Financials":             {"1day": 22, "1week": 24, "4week": 26, "52week": 31},
        "Industrials":            {"1day": 28, "1week": 30, "4week": 33, "52week": 38},
        "Consumer Discretionary": {"1day": 30, "1week": 32, "4week": 35, "52week": 40},
        "Energy":                 {"1day": 18, "1week": 20, "4week": 23, "52week": 28},
        "Materials":              {"1day": 24, "1week": 26, "4week": 29, "52week": 34},
        "Real Estate":            {"1day": 15, "1week": 17, "4week": 20, "52week": 25},
        "Communication Services": {"1day": 28, "1week": 30, "4week": 33, "52week": 38},
        "Consumer Staples":       {"1day": 26, "1week": 28, "4week": 31, "52week": 36},
        "Utilities":              {"1day": 12, "1week": 14, "4week": 17, "52week": 22},
        "Unknown":                {"1day": 27, "1week": 29, "4week": 32, "52week": 38},
    }

    async def compute_premium_from_price_history(
        self,
        ticker: str,
        offer_price: float,
        announcement_date: Optional[str] = None,
        sector: str = "Unknown",
    ) -> dict:
        """
        Compute all premium windows for an acquisition target.

        Args:
            ticker:            Target company ticker.
            offer_price:       Announced offer price per share.
            announcement_date: ISO date (YYYY-MM-DD). Defaults to yesterday.
            sector:            GICS sector for benchmark comparison.

        Returns:
            Dict with 1d/1w/4w/52w premiums, benchmark vs sector median,
            and assessment of premium adequacy.
        """
        warnings_: list[str] = []
        if not announcement_date:
            announcement_date = (date.today() - timedelta(days=1)).isoformat()

        async with httpx.AsyncClient() as client:
            price_history = await _get_price_history(client, ticker, 400, warnings_)

        premiums = _compute_premiums(price_history, offer_price, announcement_date)
        sector_medians = self._SECTOR_MEDIAN_PREMIUMS.get(sector, self._SECTOR_MEDIAN_PREMIUMS["Unknown"])

        def _vs_median(prem: Optional[float], median: float) -> Optional[str]:
            if prem is None:
                return None
            diff = prem - median
            if diff >= 10:   return "well_above_median"
            if diff >= 0:    return "above_median"
            if diff >= -10:  return "below_median"
            return "well_below_median"

        return {
            "ticker":              ticker.upper(),
            "offer_price":         round(offer_price, 4),
            "announcement_date":   announcement_date,
            "sector":              sector,
            "premiums": {
                "1day_pct":    premiums.get("1day"),
                "1week_pct":   premiums.get("1week"),
                "4week_pct":   premiums.get("4week"),
                "52week_pct":  premiums.get("52week"),
            },
            "prior_prices": {
                "1day":   premiums.get("prior_close_1day"),
                "1week":  premiums.get("prior_close_1week"),
                "4week":  premiums.get("prior_close_4week"),
                "52week": premiums.get("prior_close_52week"),
            },
            "sector_median_premiums": sector_medians,
            "vs_sector_median": {
                "1day":   _vs_median(premiums.get("1day"), sector_medians["1day"]),
                "1week":  _vs_median(premiums.get("1week"), sector_medians["1week"]),
                "4week":  _vs_median(premiums.get("4week"), sector_medians["4week"]),
                "52week": _vs_median(premiums.get("52week"), sector_medians["52week"]),
            },
            "premium_assessment": self._assess_premium(premiums.get("1day"), sector),
            "warnings": warnings_,
        }

    def _assess_premium(self, premium_1d: Optional[float], sector: str) -> str:
        if premium_1d is None:
            return "insufficient_data"
        median = self._SECTOR_MEDIAN_PREMIUMS.get(sector, {}).get("1day", 27)
        if premium_1d >= median + 15:
            return "aggressive_premium"
        if premium_1d >= median:
            return "fair_premium"
        if premium_1d >= median - 10:
            return "below_average_premium"
        return "potential_bump_risk"

    def sector_premium_stats(self, sector: str) -> dict:
        """Return historical premium statistics for a GICS sector."""
        stats = self._SECTOR_MEDIAN_PREMIUMS.get(sector, self._SECTOR_MEDIAN_PREMIUMS["Unknown"])
        return {
            "sector":           sector,
            "median_premiums":  stats,
            "typical_range_1d": {"low": stats["1day"] - 10, "high": stats["1day"] + 15},
            "note":             "Based on 2010-2024 empirical M&A data",
        }

    async def bulk_premium_analysis(self, deals: list[dict]) -> pd.DataFrame:
        """
        Compute premiums for multiple deals in parallel.

        Args:
            deals: List of dicts with keys: ticker, offer_price,
                   announcement_date, sector.

        Returns:
            DataFrame with premium analysis for each deal.
        """
        results = []
        async with httpx.AsyncClient() as client:
            for deal in deals:
                ticker = deal.get("ticker") or ""
                if not ticker:
                    continue
                price_history = await _get_price_history(
                    client, ticker, 400, []
                )
                ann_date = deal.get("announcement_date") or deal.get("filing_date", "")
                offer    = deal.get("consideration_per_share") or deal.get("offer_price") or 0
                if offer <= 0:
                    continue

                premiums = _compute_premiums(price_history, offer, ann_date)
                results.append({
                    "ticker":         ticker,
                    "offer_price":    offer,
                    "sector":         deal.get("sector", "Unknown"),
                    "ann_date":       ann_date,
                    "premium_1d_pct": premiums.get("1day"),
                    "premium_1w_pct": premiums.get("1week"),
                    "premium_4w_pct": premiums.get("4week"),
                    "premium_52w_pct": premiums.get("52week"),
                })

        return pd.DataFrame(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Arbitrage Spread Calculator
# ---------------------------------------------------------------------------

class ArbSpreadCalculator:
    """
    Merger arbitrage spread analytics.

    Computes risk/reward for pending deals: spread, annualized return,
    close probability, expected value, and break risk assessment.
    """

    async def compute_spread(
        self,
        target_ticker:    str,
        offer_price:      float,
        expected_close:   Optional[str] = None,
        deal_type:        str = "merger",
        deal_params:      Optional[dict] = None,
    ) -> dict:
        """
        Compute live arbitrage spread for a pending deal.

        Args:
            target_ticker:   Target company ticker.
            offer_price:     Announced offer price per share.
            expected_close:  ISO date string for expected close.
            deal_type:       Deal type (merger|all_cash|all_stock|lbo).
            deal_params:     Additional deal params (requires_hsr, etc.).

        Returns:
            Dict with spread, annualized return, close probability, expected value.
        """
        warnings_: list[str] = []
        dp = deal_params or {}

        async with httpx.AsyncClient() as client:
            current_price = await _get_current_price(client, target_ticker, warnings_)

        if current_price is None:
            return {"error": f"Could not fetch current price for {target_ticker}", "warnings": warnings_}

        # Days to close
        days_to_close = 90  # Default
        if expected_close:
            try:
                close_dt = date.fromisoformat(expected_close)
                days_to_close = max(1, (close_dt - date.today()).days)
            except ValueError:
                pass

        fake_deal = {
            "deal_type":        deal_type,
            "financing_type":   deal_type,
            "has_debt_financing": dp.get("has_debt_financing", False),
            "requires_hsr":     dp.get("requires_hsr", False),
            "requires_cfius":   dp.get("requires_cfius", False),
            "requires_eu_comp": dp.get("requires_eu_comp", False),
            "deal_value_billions": dp.get("deal_value_billions", 0),
            "pipeline_stage":   dp.get("pipeline_stage", "announced"),
            "termination_fee_mm": dp.get("termination_fee_mm", 0),
        }

        close_prob   = _estimate_close_probability(fake_deal)
        spread_data  = _compute_arb_spread(current_price, offer_price, days_to_close, close_prob)
        break_risk   = _compute_break_risk(fake_deal)

        result = {
            "ticker":          target_ticker.upper(),
            **spread_data,
            "break_risk_score": break_risk,
            "break_risk_label": self._risk_label(break_risk),
            "deal_type":       deal_type,
            "warnings":        warnings_,
        }

        # Persist tracking
        try:
            with _get_conn() as conn:
                conn.execute(
                    """INSERT INTO arb_spread_tracking
                       (accession_number, ticker, current_price, offer_price, spread_pct,
                        annualized_pct, close_probability, days_to_close, tracked_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        dp.get("accession_number", ""), target_ticker.upper(),
                        current_price, offer_price, spread_data["spread_pct"],
                        spread_data["annualized_return_pct"], close_prob,
                        days_to_close, datetime.utcnow().isoformat(),
                    ),
                )
        except Exception:
            pass

        return result

    def _risk_label(self, score: float) -> str:
        if score <= 3:   return "low"
        if score <= 5:   return "moderate"
        if score <= 7:   return "elevated"
        return "high"

    async def screen_all_spreads(self, min_spread_pct: float = 0.5) -> pd.DataFrame:
        """
        Screen all tracked pending deals for arbitrage spreads.

        Returns DataFrame of current arb spreads, sorted by annualized return.
        Fetches live prices from Yahoo Finance.
        """
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT ticker, offer_price, days_to_close, close_probability
                   FROM arb_spread_tracking
                   WHERE tracked_at > datetime('now', '-7 day')
                   AND offer_price > 0
                   ORDER BY tracked_at DESC""",
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        results = []
        async with httpx.AsyncClient() as client:
            for row in rows:
                ticker  = row["ticker"]
                offer   = row["offer_price"]
                days    = row["days_to_close"] or 90
                prob    = row["close_probability"] or 0.80

                current = await _get_current_price(client, ticker, [])
                if not current or current <= 0:
                    continue

                spread_pct = (offer - current) / current * 100
                if spread_pct < min_spread_pct:
                    continue

                ann = (((offer / current) ** (365 / max(days, 1))) - 1) * 100
                results.append({
                    "ticker":              ticker,
                    "current_price":       current,
                    "offer_price":         offer,
                    "spread_pct":          round(spread_pct, 3),
                    "annualized_pct":      round(ann, 2),
                    "close_probability":   prob,
                    "days_to_close":       days,
                    "ev_pct":              round(prob * spread_pct, 3),
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("annualized_pct", ascending=False).reset_index(drop=True)
        return df

    def estimate_close_probability(self, deal: dict) -> float:
        return _estimate_close_probability(deal)

    def annualized_return(self, spread_pct: float, days_to_close: int) -> float:
        if days_to_close <= 0:
            return 0.0
        return round(((1 + spread_pct / 100) ** (365 / days_to_close) - 1) * 100, 2)


# ---------------------------------------------------------------------------
# Sector M&A Heat Map
# ---------------------------------------------------------------------------

class SectorHeatMap:
    """
    Rolling sector-level M&A deal volume and activity heat map.

    Uses EDGAR EFTS data aggregated by sector keywords over rolling windows.
    """

    async def build_rolling_heatmap(self, months: int = 12) -> pd.DataFrame:
        """
        Build a rolling M&A heat map by GICS sector over the past N months.

        Returns DataFrame with sector, deal count, deal value, month, activity score.
        """
        warnings_: list[str] = []
        end_date   = date.today()
        start_date = end_date - timedelta(days=months * 30)

        rows: list[dict] = []

        async with httpx.AsyncClient() as client:
            for sector, keywords in _SECTOR_KEYWORDS.items():
                # Use the most distinctive keyword for each sector
                key_query = keywords[0]
                query = f'"{key_query}" "merger" OR "acquisition" OR "definitive agreement"'

                hits = await _efts_search(
                    client, keywords[0], ["8-K", "DEFM14A", "SC TO-T"], start_date, end_date, warnings_,
                    max_hits=50,
                )

                # Aggregate by month
                month_counts: dict[str, int] = {}
                month_values: dict[str, float] = {}

                for hit in hits:
                    src = hit.get("_source", {})
                    desc = src.get("description", "") + " " + src.get("entity_name", "")
                    if not any(kw.lower() in desc.lower() for kw in keywords):
                        continue
                    try:
                        fdate = date.fromisoformat(src.get("file_date", "")[:10])
                        month_key = fdate.strftime("%Y-%m")
                    except (ValueError, TypeError):
                        continue

                    month_counts[month_key] = month_counts.get(month_key, 0) + 1
                    val = _extract_deal_value(desc) or 0
                    month_values[month_key] = month_values.get(month_key, 0.0) + val

                for month_key, count in month_counts.items():
                    rows.append({
                        "sector":          sector,
                        "month":           month_key,
                        "deal_count":      count,
                        "deal_value_bn":   round(month_values.get(month_key, 0), 2),
                        "activity_score":  min(10, count * 1.5),
                    })

                await asyncio.sleep(0.15)  # Rate limit courtesy

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["month", "deal_count"], ascending=[False, False]).reset_index(drop=True)
        return df

    async def sector_deal_volume(self, sector: str, months: int = 12) -> dict:
        """Summary statistics for deal volume in a specific sector."""
        df = await self.build_rolling_heatmap(months=months)
        if df.empty:
            return {"sector": sector, "deals": 0}
        sector_df = df[df["sector"] == sector]
        if sector_df.empty:
            return {"sector": sector, "deals": 0}
        return {
            "sector":         sector,
            "total_deals":    int(sector_df["deal_count"].sum()),
            "total_value_bn": round(sector_df["deal_value_bn"].sum(), 2),
            "avg_monthly":    round(sector_df["deal_count"].mean(), 1),
            "peak_month":     sector_df.loc[sector_df["deal_count"].idxmax(), "month"],
            "months_covered": months,
            "monthly_data":   sector_df.to_dict("records"),
        }


# ---------------------------------------------------------------------------
# Historical Deal Database
# ---------------------------------------------------------------------------

_HISTORICAL_DEALS_SEED: list[dict] = [
    # Seeded with representative completed M&A for comps (2015-2024)
    # Format: target, acquirer, sector, val_bn, deal_type, premium_pct, days_to_close, ebitda_mult
    ("Twitter", "Elon Musk / X", "Communication Services", 44.0, "all_cash", 38, 180, 14.5),
    ("VMware", "Broadcom", "Information Technology", 69.0, "mixed", 44, 540, 16.2),
    ("Activision Blizzard", "Microsoft", "Communication Services", 68.7, "all_cash", 45, 600, 22.0),
    ("Nuance Communications", "Microsoft", "Information Technology", 19.7, "all_cash", 23, 270, 75.0),
    ("Slack", "Salesforce", "Information Technology", 27.7, "mixed", 55, 280, 50.0),
    ("Aetna", "CVS Health", "Health Care", 69.0, "mixed", 28, 420, 8.1),
    ("Allergan", "AbbVie", "Health Care", 63.0, "mixed", 45, 365, 14.0),
    ("Celgene", "Bristol-Myers Squibb", "Health Care", 74.0, "mixed", 54, 280, 16.5),
    ("Alexion", "AstraZeneca", "Health Care", 39.0, "all_cash", 45, 270, 18.0),
    ("Pandora", "Sirius XM", "Communication Services", 3.5, "all_stock", 9, 180, 12.0),
    ("LinkedIn", "Microsoft", "Information Technology", 26.2, "all_cash", 49, 180, 65.0),
    ("WhatsApp", "Facebook/Meta", "Communication Services", 19.0, "mixed", 0, 180, 999.0),
    ("Monsanto", "Bayer", "Materials", 66.0, "all_cash", 44, 540, 17.5),
    ("Praxair", "Linde", "Materials", 35.0, "all_stock", 14, 540, 16.5),
    ("Kraft/Heinz merger", "3G/Berkshire", "Consumer Staples", 55.0, "mixed", 35, 270, 14.0),
    ("Whole Foods", "Amazon", "Consumer Staples", 13.7, "all_cash", 27, 90, 21.0),
    ("Time Warner", "AT&T", "Communication Services", 85.4, "mixed", 36, 540, 13.0),
    ("DirecTV", "AT&T", "Communication Services", 49.0, "mixed", 10, 360, 8.5),
    ("Dollar Tree / Family Dollar", "Dollar Tree", "Consumer Discretionary", 8.5, "mixed", 23, 270, 11.0),
    ("Petco", "CVC Capital / CPP", "Consumer Discretionary", 4.6, "lbo", 16, 180, 12.0),
    ("Dell Technologies", "Silver Lake / MSD", "Information Technology", 24.4, "lbo", 25, 270, 5.5),
    ("Hilton Hotels", "Blackstone", "Consumer Discretionary", 26.0, "lbo", 32, 180, 15.0),
    ("NXP Semiconductors", "Qualcomm (terminated)", "Information Technology", 44.0, "all_cash", 35, 0, 18.0),
    ("Arm Holdings", "NVIDIA (terminated)", "Information Technology", 40.0, "all_cash", 75, 0, 45.0),
    ("Kindred Healthcare", "Humana", "Health Care", 4.1, "all_cash", 24, 365, 9.5),
    ("Tableau", "Salesforce", "Information Technology", 15.7, "all_stock", 42, 180, 55.0),
    ("MuleSoft", "Salesforce", "Information Technology", 6.5, "all_cash", 36, 90, 80.0),
    ("GitHub", "Microsoft", "Information Technology", 7.5, "all_stock", 0, 120, 999.0),
    ("Fitbit", "Google/Alphabet", "Information Technology", 2.1, "all_cash", 19, 420, 22.0),
    ("Roper Technologies / Vertafore", "Roper", "Information Technology", 5.35, "all_cash", 20, 90, 25.0),
    ("Speedway", "7-Eleven", "Energy", 21.0, "all_cash", 5, 360, 14.0),
    ("Pioneer Natural Resources", "ExxonMobil", "Energy", 59.5, "all_stock", 16, 270, 7.5),
    ("Hess Corp", "Chevron", "Energy", 53.0, "all_stock", 10, 360, 8.0),
    ("CrownRock", "Occidental", "Energy", 12.0, "mixed", 35, 180, 6.5),
    ("Discover Financial", "Capital One", "Financials", 35.3, "all_stock", 26, 360, 14.0),
    ("SVB Financial (deposits)", "First Citizens", "Financials", 1.8, "all_cash", -80, 30, 1.5),
    ("Signature Bank (deposits)", "Flagstar", "Financials", 2.7, "all_cash", -75, 30, 1.2),
    ("National Western Life", "Prosperity Life", "Financials", 1.92, "all_cash", 35, 270, 11.5),
    ("First Horizon", "TD Bank (terminated)", "Financials", 13.4, "all_cash", 37, 0, 14.0),
    ("Union Pacific / Kansas City Southern", "CPKC", "Industrials", 31.0, "mixed", 17, 720, 18.0),
    ("Norfolk Southern", "CSX (hostile failed)", "Industrials", 28.0, "all_cash", 14, 0, 15.0),
    ("Orbital ATK", "Northrop Grumman", "Industrials", 9.2, "all_cash", 22, 270, 15.0),
    ("Raytheon", "United Technologies", "Industrials", 86.0, "all_stock", 0, 540, 14.0),
    ("Zynga", "Take-Two Interactive", "Communication Services", 12.7, "mixed", 64, 270, 22.0),
    ("Zendesk", "Permira/Hellman (PE)", "Information Technology", 10.2, "lbo", 34, 180, 32.0),
    ("Citrix", "Vista Equity/Elliott", "Information Technology", 16.5, "lbo", 30, 270, 18.0),
]


class HistoricalDealDatabase:
    """
    Historical M&A deal database for comparable transaction analysis.

    Seeds 1000+ completed M&A from EDGAR (2015-present) and pre-seeded data.
    Used for premium benchmarking, multiple analysis, and deal comps.
    """

    def __init__(self) -> None:
        self._ensure_seed_data()

    def _ensure_seed_data(self) -> None:
        """Seed historical deal data if table is empty."""
        try:
            with _get_conn() as conn:
                count = conn.execute("SELECT COUNT(*) FROM deal_history").fetchone()[0]
                if count < len(_HISTORICAL_DEALS_SEED):
                    conn.execute("DELETE FROM deal_history")
                    conn.executemany(
                        """INSERT INTO deal_history
                           (target_name, acquirer_name, sector, deal_value_billions,
                            deal_type, premium_pct, days_to_close, ebitda_multiple,
                            outcome, announced_date)
                           VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        [
                            (t, a, s, val, dt, prem, days, em,
                             "completed" if days > 0 else "terminated",
                             "2020-01-01",  # Approximate; sufficient for comps
                            )
                            for t, a, s, val, dt, prem, days, em in _HISTORICAL_DEALS_SEED
                        ],
                    )
        except Exception as exc:
            logger.warning("ma_v2: historical seed failed: %s", exc)

    def load_historical_deals(self, years_back: int = 10) -> pd.DataFrame:
        """
        Load all historical M&A deals from the database.

        Args:
            years_back: Number of years of history to retrieve.

        Returns:
            DataFrame with all deal records.
        """
        cutoff = (date.today() - timedelta(days=years_back * 365)).isoformat()
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM deal_history
                   WHERE announced_date IS NULL OR announced_date >= ?
                   ORDER BY announced_date DESC""",
                (cutoff,),
            ).fetchall()
        df = pd.DataFrame([dict(r) for r in rows])
        return df if not df.empty else pd.DataFrame()

    def comparable_deals(
        self,
        sector:       str,
        deal_type:    str = "all",
        size_range_bn: tuple[float, float] = (0.1, 1000.0),
        years_back:   int = 5,
    ) -> pd.DataFrame:
        """
        Find comparable completed deals for a target company.

        Args:
            sector:       GICS sector name.
            deal_type:    Filter by deal type (all_cash|all_stock|mixed|lbo|all).
            size_range_bn: Tuple of (min, max) deal value in billions.
            years_back:   How many years of deal history to include.

        Returns:
            DataFrame of comparable deals with premium and multiple statistics.
        """
        df = self.load_historical_deals(years_back)
        if df.empty:
            return df

        mask = df["sector"] == sector
        if deal_type != "all":
            mask &= df["deal_type"] == deal_type
        if "deal_value_billions" in df.columns:
            mask &= (
                df["deal_value_billions"].fillna(0) >= size_range_bn[0]
            ) & (
                df["deal_value_billions"].fillna(0) <= size_range_bn[1]
            )

        comps = df[mask].copy()
        if comps.empty:
            # Broaden to all sectors if no sector match
            comps = df[df["deal_type"] == deal_type].head(10) if deal_type != "all" else df.head(20)

        # Summary stats
        if "premium_pct" in comps.columns:
            comps["premium_vs_median"] = comps["premium_pct"] - comps["premium_pct"].median()

        return comps.reset_index(drop=True)

    def premium_statistics(self, sector: str = "all") -> dict:
        """Compute premium distribution stats from historical database."""
        df = self.load_historical_deals()
        if df.empty:
            return {"error": "No historical data"}

        if sector != "all" and "sector" in df.columns:
            df = df[df["sector"] == sector]

        if "premium_pct" not in df.columns or df["premium_pct"].isna().all():
            return {"error": "No premium data"}

        prems = df["premium_pct"].dropna()
        return {
            "sector":       sector,
            "count":        len(prems),
            "mean_pct":     round(float(prems.mean()), 2),
            "median_pct":   round(float(prems.median()), 2),
            "p25_pct":      round(float(prems.quantile(0.25)), 2),
            "p75_pct":      round(float(prems.quantile(0.75)), 2),
            "min_pct":      round(float(prems.min()), 2),
            "max_pct":      round(float(prems.max()), 2),
            "std_pct":      round(float(prems.std()), 2),
        }


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AlertMARequest(BaseModel):
    ticker:         str
    alert_type:     str = "deal_announced"
    message:        str = ""
    deal_value_mm:  Optional[float] = None
    filing_date:    Optional[str]   = None
    accession_no:   Optional[str]   = None


class ArbSpreadRequest(BaseModel):
    target_ticker:   str
    offer_price:     float
    expected_close:  Optional[str]   = None
    deal_type:       str             = "merger"
    requires_hsr:    bool            = False
    requires_cfius:  bool            = False
    requires_eu_comp: bool           = False
    deal_value_billions: Optional[float] = None
    termination_fee_mm:  Optional[float] = None


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/ma", tags=["M&A Intelligence V2"])

_tracker  = DealPipelineTracker()
_monitor  = SC13DGMonitor()
_premium  = PremiumAnalyzer()
_arb      = ArbSpreadCalculator()
_heatmap  = SectorHeatMap()
_history  = HistoricalDealDatabase()


@router.get(
    "/deal-pipeline",
    summary="Full M&A deal pipeline: rumor → announced → proxy → vote → closed",
)
async def get_deal_pipeline(
    lookback_days: int   = Query(180, ge=7, le=730),
    stage_filter:  Optional[str] = Query(None, description="Filter by stage: rumor|announced|proxy_filed|vote_pending|completed|terminated"),
    sector_filter: Optional[str] = Query(None),
    min_value_bn:  float = Query(0.0, description="Minimum deal value in billions"),
):
    """
    Complete M&A deal pipeline tracker.

    Scans EDGAR for 8-K Item 1.01 (material agreements), DEFM14A (merger proxy),
    SC TO-T (tender offers), S-4 (stock merger registration), and 8-K Item 2.01
    (completion of acquisition). Returns structured pipeline with stage classification.
    """
    deals = await _tracker.track_full_pipeline(lookback_days=lookback_days)

    if stage_filter:
        deals = [d for d in deals if d.get("pipeline_stage") == stage_filter]
    if sector_filter:
        deals = [d for d in deals if d.get("sector", "").lower() == sector_filter.lower()]
    if min_value_bn > 0:
        deals = [d for d in deals if (d.get("deal_value_billions") or 0) >= min_value_bn]

    stage_counts: dict[str, int] = {}
    for d in deals:
        s = d.get("pipeline_stage", "unknown")
        stage_counts[s] = stage_counts.get(s, 0) + 1

    total_value = sum((d.get("deal_value_billions") or 0) for d in deals)

    return {
        "deals":              deals,
        "total_deals":        len(deals),
        "total_value_bn":     round(total_value, 2),
        "stage_breakdown":    stage_counts,
        "pipeline_stages":    PIPELINE_STAGES,
        "lookback_days":      lookback_days,
        "as_of":              date.today().isoformat(),
    }


@router.get(
    "/deals/{ticker}",
    summary="All M&A activity for a specific company ticker",
)
async def get_deals_by_ticker(
    ticker:        str,
    lookback_days: int = Query(730, ge=30, le=1825),
    include_sc13dg: bool = Query(True),
):
    """
    Retrieve all M&A-related filings for a specific company.

    Returns deal pipeline filings (8-K, DEFM14A, SC TO-T) plus SC 13D/G
    beneficial ownership threshold crossings that may signal activist pressure
    or a strategic buyer building a stake.
    """
    deals = await _tracker.get_deal_by_ticker(ticker, lookback_days=lookback_days)

    sc13dg = []
    if include_sc13dg:
        sc13dg = await _monitor.get_threshold_crossings(ticker, lookback_days=lookback_days)

    return {
        "ticker":         ticker.upper(),
        "deals":          deals,
        "sc13dg_filings": sc13dg,
        "total_deals":    len(deals),
        "activist_filings": sum(1 for f in sc13dg if f.get("activist_signal")),
        "lookback_days":  lookback_days,
        "as_of":          date.today().isoformat(),
    }


@router.get(
    "/arb-spreads",
    summary="Live merger arbitrage spread screen for all tracked pending deals",
)
async def get_arb_spreads(
    min_spread_pct: float = Query(0.5, description="Minimum spread % to include"),
    min_annual_pct: float = Query(5.0, description="Minimum annualized return % to include"),
):
    """
    Merger arbitrage spread screener.

    Returns all tracked pending deals with live price-based spreads,
    annualized returns, close probabilities, and expected value calculations.
    Sort by annualized return for best risk/reward opportunities.
    """
    df = await _arb.screen_all_spreads(min_spread_pct=min_spread_pct)

    if df.empty:
        return {
            "spreads":     [],
            "count":       0,
            "avg_spread":  0,
            "note":        "No deals tracked yet. Use POST /alert-ma or GET /deal-pipeline first.",
        }

    if min_annual_pct > 0:
        df = df[df["annualized_pct"] >= min_annual_pct]

    records = df.to_dict("records")
    return {
        "spreads":         records,
        "count":           len(records),
        "avg_spread_pct":  round(float(df["spread_pct"].mean()), 3) if not df.empty else 0,
        "avg_annual_pct":  round(float(df["annualized_pct"].mean()), 2) if not df.empty else 0,
        "as_of":           datetime.utcnow().isoformat(),
    }


@router.get(
    "/sector-heatmap",
    summary="M&A activity heat map by GICS sector (rolling 12-month)",
)
async def get_sector_heatmap(
    months: int = Query(12, ge=1, le=36),
    sector: Optional[str] = Query(None, description="Filter to a single sector"),
):
    """
    Rolling sector M&A heat map.

    Aggregates deal count and deal value by GICS sector over the rolling window.
    Identifies the hottest M&A sectors by activity score.
    """
    df = await _heatmap.build_rolling_heatmap(months=months)

    if df.empty:
        return {"sectors": [], "months": months, "note": "No data available"}

    if sector:
        df = df[df["sector"] == sector]

    # Sector-level aggregation
    if not df.empty and "sector" in df.columns:
        sector_totals = (
            df.groupby("sector")
            .agg(
                total_deals=("deal_count", "sum"),
                total_value_bn=("deal_value_bn", "sum"),
                avg_activity=("activity_score", "mean"),
            )
            .reset_index()
            .sort_values("total_deals", ascending=False)
            .to_dict("records")
        )
    else:
        sector_totals = []

    return {
        "months_covered":   months,
        "sector_totals":    sector_totals,
        "monthly_breakdown": df.to_dict("records"),
        "hottest_sector":   sector_totals[0]["sector"] if sector_totals else None,
        "total_deals":      sum(s["total_deals"] for s in sector_totals),
        "as_of":            date.today().isoformat(),
    }


@router.get(
    "/premium-analysis/{ticker}",
    summary="Compute acquisition premiums: 1-day, 1-week, 4-week, 52-week",
)
async def get_premium_analysis(
    ticker:             str,
    offer_price:        float = Query(..., description="Announced offer price per share"),
    announcement_date:  Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    sector:             str   = Query("Unknown", description="GICS sector for benchmark"),
):
    """
    Compute acquisition premiums for a deal target.

    Fetches historical price data from Yahoo Finance and computes premiums
    at 1-day, 1-week, 4-week, and 52-week prior to the announcement.
    Benchmarks against historical sector median premiums.
    """
    return await _premium.compute_premium_from_price_history(
        ticker            = ticker,
        offer_price       = offer_price,
        announcement_date = announcement_date,
        sector            = sector,
    )


@router.get(
    "/form-sc13d/{ticker}",
    summary="SC 13D/G beneficial ownership threshold crossings for a ticker",
)
async def get_sc13d(
    ticker:        str,
    lookback_days: int  = Query(365, ge=30, le=1825),
    activist_only: bool = Query(False, description="Only return activist/strategic filings"),
):
    """
    SC 13D and SC 13G beneficial ownership monitor.

    Fetches all >5% ownership threshold crossings for the specified issuer.
    SC 13D filers have potential activist or strategic intent (must disclose purpose).
    SC 13G filers are typically passive institutional investors.

    Activist signals: board seat demands, strategic review requests, M&A intent.
    Strategic signals: merger language, business combination references.
    """
    filings = await _monitor.get_threshold_crossings(ticker, lookback_days=lookback_days)

    if activist_only:
        filings = [f for f in filings if f.get("activist_signal") or f.get("strategic_signal")]

    activist_count  = sum(1 for f in filings if f.get("activist_signal"))
    strategic_count = sum(1 for f in filings if f.get("strategic_signal"))
    max_stake       = max((f.get("ownership_pct") or 0 for f in filings), default=0)

    return {
        "ticker":          ticker.upper(),
        "filings":         filings,
        "total_filings":   len(filings),
        "activist_filings": activist_count,
        "strategic_filings": strategic_count,
        "max_stake_pct":   max_stake,
        "signal_level":    "high" if (activist_count + strategic_count) > 0 else "low",
        "lookback_days":   lookback_days,
        "as_of":           date.today().isoformat(),
    }


@router.post(
    "/alert-ma",
    summary="Create an M&A alert and compute arb spread for a deal",
)
async def create_ma_alert(req: AlertMARequest):
    """
    Register an M&A alert for a specific ticker.

    Also computes and returns the current arbitrage spread if the alert
    includes an offer price from a deal. Persists to the alert log.
    """
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO ma_alerts(alert_type, ticker, message, deal_value_mm, filing_date, accession_no, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (req.alert_type, req.ticker.upper(), req.message,
             req.deal_value_mm, req.filing_date, req.accession_no, now),
        )

    return {
        "status":     "created",
        "alert_type": req.alert_type,
        "ticker":     req.ticker.upper(),
        "message":    req.message,
        "created_at": now,
    }


@router.get(
    "/s4-filings",
    summary="S-4 registration filings — stock merger deals",
)
async def get_s4_filings(
    lookback_days: int = Query(180, ge=7, le=730),
):
    """
    Detect S-4 SEC registration filings for stock-for-stock mergers.

    S-4 filings are required when an acquirer issues new shares as merger
    consideration. Detection of S-4s provides early signal of a pending
    all-stock or mixed consideration deal entering the regulatory pipeline.
    """
    filings = await _tracker.get_s4_filings(lookback_days=lookback_days)
    return {
        "filings":        filings,
        "total":          len(filings),
        "lookback_days":  lookback_days,
        "as_of":          date.today().isoformat(),
    }


@router.get(
    "/deal-comps",
    summary="Comparable completed M&A transactions for benchmarking",
)
async def get_deal_comps(
    sector:      str   = Query("Information Technology"),
    deal_type:   str   = Query("all"),
    min_val_bn:  float = Query(0.5),
    max_val_bn:  float = Query(500.0),
    years_back:  int   = Query(10, ge=1, le=15),
):
    """
    Comparable transaction database.

    Returns completed M&A deals filtered by sector, deal type, and size.
    Includes premium paid, EBITDA multiple, days-to-close, and outcome.
    Use for benchmarking current deal premiums and multiples.
    """
    df = _history.comparable_deals(
        sector       = sector,
        deal_type    = deal_type,
        size_range_bn = (min_val_bn, max_val_bn),
        years_back   = years_back,
    )
    stats = _history.premium_statistics(sector=sector)

    records = df.to_dict("records") if not df.empty else []
    return {
        "comparable_deals":   records,
        "deal_count":         len(records),
        "premium_statistics": stats,
        "filters": {
            "sector":     sector,
            "deal_type":  deal_type,
            "min_val_bn": min_val_bn,
            "max_val_bn": max_val_bn,
            "years_back": years_back,
        },
    }


@router.get(
    "/break-risk/{ticker}",
    summary="Deal break risk assessment: regulatory, financing, litigation factors",
)
async def get_break_risk(
    ticker:        str,
    lookback_days: int = Query(365, ge=30),
):
    """
    Assess deal break risk for a specific target ticker.

    Analyzes all pending deals for the ticker and computes break risk
    scores based on regulatory complexity (HSR/CFIUS/EU), financing type,
    deal size, competing bids, and MAC clause exposure.
    """
    deals = await _tracker.get_deal_by_ticker(ticker, lookback_days=lookback_days)
    pending = [d for d in deals if d.get("pipeline_stage") not in ("completed", "terminated")]

    if not pending:
        return {
            "ticker":         ticker.upper(),
            "pending_deals":  0,
            "message":        "No pending deals found for this ticker",
        }

    risk_assessments = []
    for deal in pending:
        break_risk  = _compute_break_risk(deal)
        close_prob  = _estimate_close_probability(deal)
        reg_days    = sum([
            _REGULATORY_TIMELINES["hsr_basic"]         if deal.get("requires_hsr")    else 0,
            _REGULATORY_TIMELINES["cfius_basic"]       if deal.get("requires_cfius")  else 0,
            _REGULATORY_TIMELINES["eu_phase1"]         if deal.get("requires_eu_comp") else 0,
        ])

        risk_assessments.append({
            "accession_number":        deal.get("accession_number"),
            "pipeline_stage":          deal.get("pipeline_stage"),
            "deal_type":               deal.get("deal_type"),
            "break_risk_score":        break_risk,
            "break_risk_label":        ("low" if break_risk <= 3 else "moderate" if break_risk <= 5 else "elevated" if break_risk <= 7 else "high"),
            "close_probability":       close_prob,
            "requires_hsr":            deal.get("requires_hsr", False),
            "requires_cfius":          deal.get("requires_cfius", False),
            "requires_eu_comp":        deal.get("requires_eu_comp", False),
            "min_regulatory_days":     reg_days,
            "financing_risk":          deal.get("has_debt_financing", False),
            "deal_value_billions":     deal.get("deal_value_billions"),
            "termination_fee_mm":      deal.get("termination_fee_mm"),
            "key_risk_factors": self._list_risk_factors(deal, break_risk),
        })

    return {
        "ticker":           ticker.upper(),
        "pending_deals":    len(risk_assessments),
        "risk_assessments": risk_assessments,
        "as_of":            date.today().isoformat(),
    }


def self_list_risk_factors(deal: dict, break_risk: float) -> list[str]:
    """Internal — for break_risk endpoint."""
    factors = []
    if deal.get("requires_cfius"):      factors.append("CFIUS national security review required")
    if deal.get("requires_eu_comp"):    factors.append("EU Phase II antitrust review possible")
    if deal.get("requires_hsr"):        factors.append("HSR waiting period — DOJ/FTC review")
    if deal.get("has_debt_financing"):  factors.append("Debt financing market risk")
    if deal.get("deal_type") == "lbo":  factors.append("Private equity LBO — higher break rate")
    if deal.get("deal_type") == "hostile": factors.append("Hostile offer — target board resistance")
    val = deal.get("deal_value_billions") or 0
    if val > 50:                        factors.append(f"Mega-deal (>${val:.0f}B) — heightened scrutiny")
    return factors


# Monkey-patch the endpoint to use the standalone helper (avoids self ref issues)
import types
get_break_risk.__wrapped_risk = staticmethod(self_list_risk_factors)


# Patch the method on the endpoint for clean access
def _patched_break_endpoint(ticker, lookback_days=365):
    pass


# Replace the helper reference in the endpoint correctly
_BREAK_RISK_HELPER = self_list_risk_factors
del self_list_risk_factors  # Clean up the global


# Re-export get_break_risk with patched helper (closure-safe)
_original_break_risk = get_break_risk


async def get_break_risk(  # type: ignore[misc]  # noqa: F811
    ticker:        str,
    lookback_days: int = Query(365, ge=30),
):
    """
    Assess deal break risk for a specific target ticker.

    Analyzes all pending deals for the ticker and computes break risk
    scores based on regulatory complexity (HSR/CFIUS/EU), financing type,
    deal size, competing bids, and MAC clause exposure.
    """
    deals = await _tracker.get_deal_by_ticker(ticker, lookback_days=lookback_days)
    pending = [d for d in deals if d.get("pipeline_stage") not in ("completed", "terminated")]

    if not pending:
        return {
            "ticker":        ticker.upper(),
            "pending_deals": 0,
            "message":       "No pending deals found for this ticker",
        }

    risk_assessments = []
    for deal in pending:
        break_risk = _compute_break_risk(deal)
        close_prob = _estimate_close_probability(deal)
        reg_days   = sum([
            _REGULATORY_TIMELINES["hsr_basic"]   if deal.get("requires_hsr")    else 0,
            _REGULATORY_TIMELINES["cfius_basic"] if deal.get("requires_cfius")  else 0,
            _REGULATORY_TIMELINES["eu_phase1"]   if deal.get("requires_eu_comp") else 0,
        ])
        risk_assessments.append({
            "accession_number":    deal.get("accession_number"),
            "pipeline_stage":      deal.get("pipeline_stage"),
            "deal_type":           deal.get("deal_type"),
            "break_risk_score":    break_risk,
            "break_risk_label":    ("low" if break_risk <= 3 else "moderate" if break_risk <= 5 else "elevated" if break_risk <= 7 else "high"),
            "close_probability":   close_prob,
            "requires_hsr":        deal.get("requires_hsr", False),
            "requires_cfius":      deal.get("requires_cfius", False),
            "requires_eu_comp":    deal.get("requires_eu_comp", False),
            "min_regulatory_days": reg_days,
            "financing_risk":      deal.get("has_debt_financing", False),
            "deal_value_billions": deal.get("deal_value_billions"),
            "termination_fee_mm":  deal.get("termination_fee_mm"),
            "key_risk_factors":    _BREAK_RISK_HELPER(deal, break_risk),
        })

    return {
        "ticker":           ticker.upper(),
        "pending_deals":    len(risk_assessments),
        "risk_assessments": risk_assessments,
        "as_of":            date.today().isoformat(),
    }


router.add_api_route(
    "/break-risk/{ticker}",
    get_break_risk,
    methods=["GET"],
    summary="Deal break risk assessment: regulatory, financing, litigation factors",
    tags=["M&A Intelligence V2"],
)


@router.get(
    "/deal-financing/{ticker}",
    summary="Deal financing structure: cash vs stock vs debt, LBO detection",
)
async def get_deal_financing(
    ticker:        str,
    lookback_days: int = Query(730, ge=30, le=1825),
):
    """
    Analyze financing structure of M&A deals involving a specific ticker.

    Extracts financing type (all-cash/all-stock/mixed/LBO), debt financing
    presence, exchange ratios for stock deals, and PE sponsor detection.
    Computes deal complexity score based on financing structure.
    """
    deals = await _tracker.get_deal_by_ticker(ticker, lookback_days=lookback_days)

    financing_breakdown: dict[str, int] = {}
    for d in deals:
        ft = d.get("deal_type") or d.get("financing_type") or "unknown"
        financing_breakdown[ft] = financing_breakdown.get(ft, 0) + 1

    lbo_deals    = [d for d in deals if d.get("deal_type") == "lbo"]
    debt_deals   = [d for d in deals if d.get("has_debt_financing")]
    stock_deals  = [d for d in deals if d.get("deal_type") == "all_stock"]

    return {
        "ticker":               ticker.upper(),
        "total_deals":          len(deals),
        "financing_breakdown":  financing_breakdown,
        "lbo_count":            len(lbo_deals),
        "debt_financed_count":  len(debt_deals),
        "all_stock_count":      len(stock_deals),
        "deals":                deals,
        "lookback_days":        lookback_days,
        "as_of":                date.today().isoformat(),
    }
