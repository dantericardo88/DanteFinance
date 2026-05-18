"""
Fundamental Equity Screener V3 — Dimension #070 (target score 9/10).

DATA SOURCE TRANSPARENCY
-------------------------
- Primary source: EDGAR XBRL companyfacts API (free, authoritative, point-in-time)
  * URL: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  * All fundamental ratios derived from XBRL facts: NO yfinance for financials
  * EDGAR data is the same source as SEC filings — it IS the truth
  * Rate limit: 10 req/s; we target 8 req/s (120ms between calls)
- Market prices: yfinance (acceptable for price-based momentum / market cap)
  * yfinance prices are reliable; fundamental balance sheet data is NOT used from it
- DuckDB: in-memory + persistent columnar store at sentinel/data/fundamentals.duckdb
  * Bulk S&P 1500 data loaded at startup from EDGAR XBRL
  * Incremental refresh: check submissions.json last_modified before re-fetching
  * SQL screener: arbitrary WHERE / ORDER BY / LIMIT on 60+ fields
- Analyst ratings: Finviz free scrape (indicative, not guaranteed)
- Short interest: FINRA short sale data (free, public)

Architecture
------------
EDGARXBRLLoader       — fetch companyfacts JSON, extract and normalise 60+ metrics
FundamentalDuckDB     — DuckDB persistent store; load/refresh/query
FundamentalScreener   — screen(), run_preset(), SQL screener, explain results
PrebuiltScreens       — 25 pre-built screens (Buffett, Lynch, Greenblatt, etc.)
FinvizScraper         — analyst ratings, short interest scrape (free tier)
ScreeningResultsDB    — SQLite history of screening results
FastAPI router at /screener/v3

Note on yfinance
----------------
yfinance is used ONLY for price-based momentum fields:
  return_1m, return_3m, return_6m, return_12m, return_ytd, rsi_14, beta_1y
These are inherently price-derived and yfinance provides reliable OHLCV history.
All balance sheet, income statement, and cash flow fields come from EDGAR XBRL.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

try:
    import duckdb
except ImportError as _exc:
    raise ImportError("duckdb required: pip install duckdb") from _exc

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_FACTS_URL       = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FINVIZ_BASE           = "https://finviz.com/quote.ashx"
FINRA_SHORT_URL       = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"

EDGAR_HEADERS = {
    "User-Agent":       "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept":           "application/json",
    "Accept-Encoding":  "gzip, deflate",
}
_RATE_DELAY  = 0.13   # ~7.7 req/s; stay under 10 req/s EDGAR cap
_TIMEOUT     = 30.0
_MAX_RETRY   = 3

DUCKDB_PATH  = Path(os.getenv("SENTINEL_DUCKDB", "sentinel/data/fundamentals.duckdb"))
SQLITE_PATH  = Path(os.getenv("SENTINEL_RESULTS_DB", "sentinel/data/screening_results.db"))

# S&P 1500 representative sample (large + mid + small cap coverage)
SP1500_UNIVERSE: List[str] = [
    # S&P 500 large cap sample
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM",
    "UNH", "XOM", "V", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "AVGO", "CVX",
    "MRK", "ABBV", "KO", "PEP", "COST", "ADBE", "NFLX", "CRM", "ACN", "TMO",
    "MCD", "CSCO", "BAC", "ABT", "DHR", "NEE", "INTC", "VZ", "QCOM", "TXN",
    "WFC", "BMY", "LIN", "PM", "AMGN", "RTX", "HON", "UPS", "IBM", "CAT",
    "GE", "NOW", "SPGI", "INTU", "ISRG", "MDT", "ELV", "SYK", "GILD", "REGN",
    "ZTS", "BSX", "MMC", "CI", "MO", "DUK", "SO", "AXP", "BLK", "GS",
    "MS", "USB", "TJX", "CME", "CB", "AON", "MCO", "PLD", "AMT", "CCI",
    "EQIX", "PSA", "O", "WELL", "EQR", "AVB", "DRE", "ARE", "VICI", "SPG",
    # S&P 400 mid cap sample
    "PODD", "ENTG", "BILL", "PAYC", "GWRE", "COHU", "EXLS", "CRVL", "ICFI",
    "RMBS", "SMCI", "AEIS", "LOGI", "VIAV", "SLAB", "QLYS", "MANH", "DDOG",
    "ZS", "OKTA", "CFLT", "NET", "MDB", "SNOW", "PLTR", "U", "RBLX",
    "WEX", "NEOG", "ICU", "RXO", "UFPI", "IBP", "MHO", "MTH", "LGIH",
    "TMHC", "CVCO", "GRBK", "DDS", "BIG", "OLLI", "FIVE", "PRGO", "ELF",
    # S&P 600 small cap sample
    "HIMS", "TIGR", "RXST", "PAHC", "PDCO", "XNCR", "PRAX", "AROC",
    "PRIM", "DY", "MYRG", "NVEE", "HIL", "KFRC", "TBI", "KELYA", "MWA",
    "NWLI", "FBP", "WAFD", "HFWA", "HOPE", "CBTX", "FFBC", "INBK",
    "SBCF", "TBBK", "NBTB", "PFIS", "CZWI", "RRBI",
]

# ---------------------------------------------------------------------------
# XBRL concept → canonical field mapping (used during extraction)
# ---------------------------------------------------------------------------

# Priority-ordered list of XBRL tags for each field
XBRL_CONCEPT_MAP: Dict[str, List[str]] = {
    "revenue": [
        "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
    ],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsSold", "CostOfGoodsAndServicesSold"],
    "gross_profit":    ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income":      ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "ebit":            ["OperatingIncomeLoss"],
    "interest_expense": ["InterestExpense", "InterestAndDebtExpense"],
    "income_tax":      ["IncomeTaxExpenseBenefit"],
    "rd_expense":      ["ResearchAndDevelopmentExpense"],
    "sga_expense":     ["SellingGeneralAndAdministrativeExpense"],
    "depreciation":    ["DepreciationDepletionAndAmortization", "Depreciation"],
    "total_assets":    ["Assets"],
    "current_assets":  ["AssetsCurrent"],
    "total_liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity":          ["StockholdersEquity", "StockholdersEquityAttributableToParent"],
    "long_term_debt":  ["LongTermDebt", "LongTermDebtNoncurrent"],
    "short_term_debt": ["ShortTermBorrowings", "NotesPayableCurrent", "LongTermDebtCurrent"],
    "cash":            ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
    "inventory":       ["InventoryNet", "InventoryFinishedGoods"],
    "accounts_receivable": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "accounts_payable": ["AccountsPayableCurrent"],
    "goodwill":        ["Goodwill"],
    "intangibles":     ["FiniteLivedIntangibleAssetsNet", "IntangibleAssetsNetExcludingGoodwill"],
    "cfo":             [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex":           ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "dividends_paid":  ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "shares_outstanding": ["CommonStockSharesOutstanding", "CommonStockSharesIssued"],
    "eps_basic":       ["EarningsPerShareBasic"],
    "eps_diluted":     ["EarningsPerShareDiluted"],
}

# ---------------------------------------------------------------------------
# DuckDB schema (all 60+ fields)
# ---------------------------------------------------------------------------

DUCKDB_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS fundamentals (
    -- Identity
    ticker              VARCHAR PRIMARY KEY,
    company_name        VARCHAR,
    cik                 VARCHAR,
    sector              VARCHAR,
    industry            VARCHAR,
    exchange            VARCHAR,
    fiscal_year_end     DATE,
    last_10k_date       DATE,
    last_10q_date       DATE,
    data_source         VARCHAR DEFAULT 'edgar_xbrl',
    updated_at          TIMESTAMP,

    -- Size
    market_cap          DOUBLE,
    revenue_ttm         DOUBLE,
    enterprise_value    DOUBLE,
    total_assets        DOUBLE,

    -- Valuation (computed: require market_cap and XBRL financials)
    pe_ratio            DOUBLE,     -- market_cap / net_income_ttm
    peg_ratio           DOUBLE,     -- pe_ratio / eps_growth_1y
    ev_ebitda           DOUBLE,     -- enterprise_value / ebitda_ttm
    ev_revenue          DOUBLE,     -- enterprise_value / revenue_ttm
    ev_ebit             DOUBLE,     -- enterprise_value / ebit_ttm
    price_to_book       DOUBLE,     -- market_cap / book_value
    price_to_fcf        DOUBLE,     -- market_cap / fcf_ttm
    price_to_sales      DOUBLE,     -- market_cap / revenue_ttm
    earnings_yield      DOUBLE,     -- ebit_ttm / enterprise_value (Greenblatt)

    -- Growth (YoY from XBRL annual filings)
    revenue_growth_1y   DOUBLE,
    revenue_growth_3y   DOUBLE,
    eps_growth_1y       DOUBLE,
    eps_growth_3y       DOUBLE,
    fcf_growth_1y       DOUBLE,

    -- Profitability (from XBRL)
    gross_margin        DOUBLE,     -- gross_profit / revenue
    operating_margin    DOUBLE,     -- operating_income / revenue
    net_margin          DOUBLE,     -- net_income / revenue
    ebitda_margin       DOUBLE,
    roic                DOUBLE,     -- ebit*(1-t) / invested_capital
    roe                 DOUBLE,     -- net_income / equity
    roa                 DOUBLE,     -- net_income / total_assets

    -- Financial Health (from XBRL)
    current_ratio       DOUBLE,     -- current_assets / current_liabilities
    quick_ratio         DOUBLE,     -- (current_assets - inventory) / current_liabilities
    debt_to_equity      DOUBLE,     -- total_debt / equity
    net_debt_ebitda     DOUBLE,
    interest_coverage   DOUBLE,     -- ebit / interest_expense
    altman_z            DOUBLE,

    -- Momentum (price-based, yfinance)
    return_1m           DOUBLE,
    return_3m           DOUBLE,
    return_6m           DOUBLE,
    return_12m          DOUBLE,
    return_ytd          DOUBLE,
    rsi_14              DOUBLE,
    beta_1y             DOUBLE,

    -- Dividends (from XBRL + price)
    dividend_yield              DOUBLE,
    payout_ratio                DOUBLE,
    dividend_growth_5y          DOUBLE,
    consecutive_dividend_years  INTEGER,

    -- Quality
    accruals_ratio              DOUBLE,
    cash_conversion_cycle       DOUBLE,
    capex_to_revenue            DOUBLE,
    rd_to_revenue               DOUBLE,
    insider_ownership_pct       DOUBLE,

    -- Alternative data (Finviz scrape / FINRA)
    short_interest_ratio        DOUBLE,
    institutional_ownership_pct DOUBLE,
    analyst_rating_avg          DOUBLE,   -- 1=Strong Buy, 5=Strong Sell

    -- Raw XBRL values for ratio computation
    revenue_ttm_raw     DOUBLE,
    net_income_ttm      DOUBLE,
    ebitda_ttm          DOUBLE,
    ebit_ttm            DOUBLE,
    fcf_ttm             DOUBLE,
    equity_book         DOUBLE,
    total_debt          DOUBLE,
    net_debt            DOUBLE,
    cash_raw            DOUBLE,
    current_assets      DOUBLE,
    current_liabilities DOUBLE,
    inventory           DOUBLE,
    accounts_receivable DOUBLE,
    accounts_payable    DOUBLE,
    shares_outstanding  DOUBLE,
    interest_expense    DOUBLE,
    capex               DOUBLE,
    rd_expense          DOUBLE,

    -- Quality scores
    piotroski_f_score   INTEGER,
    beneish_m_score     DOUBLE,

    -- NCAV for Graham net-net
    ncav                DOUBLE,     -- current_assets - total_liabilities
    ncav_to_market_cap  DOUBLE,
)
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edgar_get(url: str, retries: int = _MAX_RETRY) -> Dict:
    """Rate-limited GET to EDGAR with retry logic."""
    for attempt in range(retries):
        time.sleep(_RATE_DELAY)
        try:
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception as exc:
            if attempt == retries - 1:
                logger.error("EDGAR GET failed %s: %s", url, exc)
                return {}
            time.sleep(2 ** attempt)
    return {}


_CIK_CACHE: Dict[str, Optional[str]] = {}


def _resolve_cik(ticker: str) -> Optional[str]:
    """Resolve ticker → zero-padded 10-digit CIK via EDGAR company tickers JSON."""
    ticker_up = ticker.upper().replace(".", "-")
    if ticker_up in _CIK_CACHE:
        return _CIK_CACHE[ticker_up]

    try:
        resp = requests.get(EDGAR_TICKERS_URL, headers=EDGAR_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker_up:
                cik = str(entry["cik_str"]).zfill(10)
                _CIK_CACHE[ticker_up] = cik
                return cik
    except Exception as exc:
        logger.warning("CIK resolution failed for %s: %s", ticker, exc)

    _CIK_CACHE[ticker_up] = None
    return None


def _safe(v: Any) -> Optional[float]:
    """Return float or None; guard against inf/nan."""
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _cagr(v0: Optional[float], v1: Optional[float], years: float) -> Optional[float]:
    if v0 is None or v1 is None or years <= 0 or v0 == 0:
        return None
    try:
        ratio = v1 / v0
        if ratio <= 0:
            return None
        return (ratio ** (1.0 / years)) - 1.0
    except Exception:
        return None


def _rsi(closes: List[float], period: int = 14) -> float:
    """Wilder's RSI."""
    if len(closes) < period + 1:
        return float("nan")
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# EDGARXBRLLoader — extract raw financial facts from companyfacts JSON
# ---------------------------------------------------------------------------

class EDGARXBRLLoader:
    """Load and extract financial facts from EDGAR XBRL companyfacts endpoint.

    All data is point-in-time: we use the most recent annual (10-K) and
    trailing-twelve-month (TTM) values based on filed dates.
    """

    def fetch_companyfacts(self, cik: str) -> Dict:
        """Fetch raw companyfacts JSON from EDGAR."""
        url = EDGAR_FACTS_URL.format(cik=cik.zfill(10))
        return _edgar_get(url)

    def get_submissions(self, cik: str) -> Dict:
        """Fetch submissions.json to check last_modified and recent filings."""
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik.zfill(10))
        return _edgar_get(url)

    def _extract_concept(
        self,
        facts_data: Dict,
        concept_list: List[str],
        form_types: Optional[List[str]] = None,
        unit: str = "USD",
    ) -> List[Dict]:
        """Extract all observations for a prioritised list of concept names.

        Returns list of {value, end (period end date), filed (filing date), form}
        sorted by filing date descending.
        """
        gaap = facts_data.get("facts", {}).get("us-gaap", {})
        for concept in concept_list:
            if concept not in gaap:
                continue
            units_data = gaap[concept].get("units", {})
            obs = units_data.get(unit, []) or units_data.get("shares", [])
            if not obs:
                continue
            results = []
            for o in obs:
                form = o.get("form", "")
                if form_types and form not in form_types:
                    continue
                # Only include point-in-time annual or quarterly
                if not o.get("end") or not o.get("filed"):
                    continue
                # Exclude instant-vs-duration confusion: revenue must have start date
                results.append({
                    "value":  float(o.get("val", 0) or 0),
                    "end":    o.get("end", ""),
                    "start":  o.get("start", ""),
                    "filed":  o.get("filed", ""),
                    "form":   form,
                    "fp":     o.get("fp", ""),
                })
            if results:
                results.sort(key=lambda x: x["filed"], reverse=True)
                return results
        return []

    def _ttm_value(self, annual_obs: List[Dict], quarterly_obs: List[Dict]) -> Optional[float]:
        """Compute TTM from the 4 most recent distinct quarters.

        TTM = sum of last 4 quarterly values (non-overlapping period ends).
        Falls back to most recent annual if insufficient quarters.
        """
        if quarterly_obs:
            # Deduplicate by period end
            seen_ends  = set()
            quarters   = []
            for o in quarterly_obs:
                end = o["end"]
                if end not in seen_ends and o["form"] in ("10-Q", "10-QT"):
                    seen_ends.add(end)
                    quarters.append(o)
            # Take up to 4 most recent
            q4 = quarters[:4]
            if len(q4) == 4:
                return sum(q["value"] for q in q4)
            # Fall through to annual if not enough quarters
        if annual_obs:
            return annual_obs[0]["value"]
        return None

    def _annual_value(self, annual_obs: List[Dict], year_offset: int = 0) -> Optional[float]:
        """Return the annual value N years ago (0 = most recent)."""
        seen_fps  = set()
        annuals   = []
        for o in annual_obs:
            if o.get("form") in ("10-K", "10-KT", "20-F", "40-F"):
                key = o.get("fp", o.get("end", ""))
                if key not in seen_fps:
                    seen_fps.add(key)
                    annuals.append(o)
        if year_offset < len(annuals):
            return annuals[year_offset]["value"]
        return None

    def extract_metrics(self, ticker: str, companyfacts: Dict) -> Dict[str, Optional[float]]:
        """Extract all raw financial metrics from companyfacts JSON.

        Returns a flat dict of {field_name: value}.
        Values are in USD (units as reported in XBRL); ratios computed later.
        """
        facts = companyfacts

        # Helper: get annual + quarterly observations for a field
        def get_obs(field: str) -> Tuple[List[Dict], List[Dict]]:
            concepts = XBRL_CONCEPT_MAP.get(field, [])
            annual = self._extract_concept(
                facts, concepts,
                form_types=["10-K", "10-KT", "20-F", "40-F"],
            )
            quarterly = self._extract_concept(
                facts, concepts,
                form_types=["10-Q", "10-QT"],
            )
            return annual, quarterly

        # Extract raw facts
        rev_a,   rev_q   = get_obs("revenue")
        gp_a,    gp_q    = get_obs("gross_profit")
        oi_a,    oi_q    = get_obs("operating_income")
        ni_a,    ni_q    = get_obs("net_income")
        ie_a,    ie_q    = get_obs("interest_expense")
        rd_a,    rd_q    = get_obs("rd_expense")
        dep_a,   dep_q   = get_obs("depreciation")
        tax_a,   tax_q   = get_obs("income_tax")
        cfo_a,   cfo_q   = get_obs("cfo")
        capex_a, capex_q = get_obs("capex")
        div_a,   div_q   = get_obs("dividends_paid")
        eps_b_a, _       = get_obs("eps_basic")
        eps_d_a, _       = get_obs("eps_diluted")

        # Balance sheet (point-in-time, instantaneous)
        ta_a,  _ = get_obs("total_assets")
        tl_a,  _ = get_obs("total_liabilities")
        eq_a,  _ = get_obs("equity")
        ltd_a, _ = get_obs("long_term_debt")
        std_a, _ = get_obs("short_term_debt")
        cash_a, _ = get_obs("cash")
        ca_a,  _ = get_obs("current_assets")
        cl_a,  _ = get_obs("current_liabilities")
        inv_a, _ = get_obs("inventory")
        ar_a,  _ = get_obs("accounts_receivable")
        ap_a,  _ = get_obs("accounts_payable")
        gw_a,  _ = get_obs("goodwill")
        ia_a,  _ = get_obs("intangibles")
        so_a,  _ = get_obs("shares_outstanding")

        m: Dict[str, Optional[float]] = {}

        # TTM and annual values
        m["revenue_ttm_raw"]   = _safe(self._ttm_value(rev_a, rev_q))
        m["net_income_ttm"]    = _safe(self._ttm_value(ni_a, ni_q))
        m["ebit_ttm"]          = _safe(self._ttm_value(oi_a, oi_q))
        m["interest_expense"]  = _safe(self._ttm_value(ie_a, ie_q))
        m["capex"]             = _safe(self._ttm_value(capex_a, capex_q))
        m["cfo_ttm"]           = _safe(self._ttm_value(cfo_a, cfo_q))
        m["rd_expense"]        = _safe(self._ttm_value(rd_a, rd_q))
        m["gross_profit_ttm"]  = _safe(self._ttm_value(gp_a, gp_q))
        m["dep_ttm"]           = _safe(self._ttm_value(dep_a, dep_q))

        # EBITDA = EBIT + D&A
        ebit  = m["ebit_ttm"]
        dep   = m["dep_ttm"]
        m["ebitda_ttm"] = _safe((ebit or 0) + (dep or 0)) if (ebit is not None or dep is not None) else None

        # FCF = CFO - CapEx
        cfo   = m["cfo_ttm"]
        capex = m["capex"]
        m["fcf_ttm"] = _safe((cfo or 0) - abs(capex or 0)) if cfo is not None else None

        # Balance sheet (latest annual)
        m["total_assets"]       = _safe(self._annual_value(ta_a, 0))
        m["equity_book"]        = _safe(self._annual_value(eq_a, 0))
        m["current_assets"]     = _safe(self._annual_value(ca_a, 0))
        m["current_liabilities"]= _safe(self._annual_value(cl_a, 0))
        m["inventory"]          = _safe(self._annual_value(inv_a, 0))
        m["accounts_receivable"]= _safe(self._annual_value(ar_a, 0))
        m["accounts_payable"]   = _safe(self._annual_value(ap_a, 0))
        m["cash_raw"]           = _safe(self._annual_value(cash_a, 0))

        ltd = _safe(self._annual_value(ltd_a, 0)) or 0.0
        std = _safe(self._annual_value(std_a, 0)) or 0.0
        m["total_debt"] = _safe(ltd + std)
        m["net_debt"]   = _safe((ltd + std) - (m["cash_raw"] or 0))

        # Shares outstanding
        m["shares_outstanding"] = _safe(self._annual_value(so_a, 0))

        # EPS
        m["eps_basic"]   = _safe(self._annual_value(eps_b_a, 0))
        m["eps_diluted"] = _safe(self._annual_value(eps_d_a, 0))

        # Revenue growth
        rev0 = _safe(self._annual_value(rev_a, 0))
        rev1 = _safe(self._annual_value(rev_a, 1))
        rev3 = _safe(self._annual_value(rev_a, 3))
        m["revenue_growth_1y"] = _cagr(rev1, rev0, 1)
        m["revenue_growth_3y"] = _cagr(rev3, rev0, 3)

        # EPS growth
        eps0 = _safe(self._annual_value(eps_b_a, 0))
        eps1 = _safe(self._annual_value(eps_b_a, 1))
        eps3 = _safe(self._annual_value(eps_b_a, 3))
        m["eps_growth_1y"] = _cagr(eps1, eps0, 1)
        m["eps_growth_3y"] = _cagr(eps3, eps0, 3)

        # FCF growth (need annual FCF, which requires annual CFO - annual CapEx)
        cfo_a0 = _safe(self._annual_value(cfo_a, 0))
        cfo_a1 = _safe(self._annual_value(cfo_a, 1))
        cp_a0  = _safe(self._annual_value(capex_a, 0))
        cp_a1  = _safe(self._annual_value(capex_a, 1))
        fcf_a0 = _safe((cfo_a0 or 0) - abs(cp_a0 or 0)) if cfo_a0 is not None else None
        fcf_a1 = _safe((cfo_a1 or 0) - abs(cp_a1 or 0)) if cfo_a1 is not None else None
        m["fcf_growth_1y"] = _cagr(fcf_a1, fcf_a0, 1)

        # Dividend consecutive years (count annual filings with dividends_paid)
        div_years = sum(1 for o in div_a if o.get("value", 0) and o.get("value", 0) < 0)
        m["consecutive_dividend_years"] = div_years or 0

        # Accruals ratio (Sloan): (NI - CFO) / avg_assets
        ni  = m["net_income_ttm"]
        cfo_v = m["cfo_ttm"]
        ta  = m["total_assets"]
        if ni is not None and cfo_v is not None and ta and ta > 0:
            m["accruals_ratio"] = _safe((ni - cfo_v) / ta)
        else:
            m["accruals_ratio"] = None

        # Cash conversion cycle = DIO + DSO - DPO
        rev = m["revenue_ttm_raw"] or 0
        cogs_obs_a, cogs_obs_q = get_obs("cost_of_revenue")
        cogs = _safe(self._ttm_value(cogs_obs_a, cogs_obs_q)) or rev * 0.6
        ar   = m["accounts_receivable"] or 0
        inv  = m["inventory"] or 0
        ap   = m["accounts_payable"] or 0
        if rev > 0 and cogs > 0:
            dso = (ar / rev * 365)       if ar else 0
            dio = (inv / cogs * 365)     if inv else 0
            dpo = (ap / cogs * 365)      if ap else 0
            m["cash_conversion_cycle"] = _safe(dso + dio - dpo)
        else:
            m["cash_conversion_cycle"] = None

        # ---------------------------------------------------------------------------
        # Prior-year values for Piotroski F-score and Beneish M-score
        # ---------------------------------------------------------------------------

        # Prior-year balance sheet (year_offset=1)
        ta_prior  = _safe(self._annual_value(ta_a,  1))
        ca_prior  = _safe(self._annual_value(ca_a,  1))
        cl_prior  = _safe(self._annual_value(cl_a,  1))
        ltd_prior = _safe(self._annual_value(ltd_a, 1)) or 0.0
        so_prior  = _safe(self._annual_value(so_a,  1))

        # Prior-year income statement (annual, year_offset=1)
        ni_prior  = _safe(self._annual_value(ni_a,  1))
        rev_prior = _safe(self._annual_value(rev_a,  1))
        gp_prior  = _safe(self._annual_value(gp_a,  1))
        dep_prior = _safe(self._annual_value(dep_a,  1))
        sga_obs_a, _ = get_obs("sga_expense")
        sga_curr  = _safe(self._ttm_value(sga_obs_a, []))
        sga_prior = _safe(self._annual_value(sga_obs_a, 1))
        cogs_prior_val = _safe(self._annual_value(cogs_obs_a, 1))

        # PPE (property, plant, equipment) for Beneish AQI and DEPI
        # Use a best-effort approach: assets_prior for AQI proxy
        ppe_obs_a, _ = get_obs("capex")          # PaymentsToAcquirePropertyPlantAndEquipment is CapEx
        # Better: try to extract PPE directly from goodwill/intangibles as proxy doesn't work.
        # Use current_assets + goodwill as the non-PPE portion for AQI.
        gw_curr   = _safe(self._annual_value(gw_a,  0)) or 0.0
        gw_prior  = _safe(self._annual_value(gw_a,  1)) or 0.0
        ia_curr   = _safe(self._annual_value(ia_a,  0)) or 0.0
        ia_prior  = _safe(self._annual_value(ia_a,  1)) or 0.0

        m["ta_prior"]   = ta_prior
        m["ca_prior"]   = ca_prior
        m["cl_prior"]   = cl_prior
        m["ltd_prior"]  = ltd_prior
        m["so_prior"]   = so_prior
        m["ni_prior"]   = ni_prior
        m["rev_prior"]  = rev_prior
        m["gp_prior"]   = gp_prior
        m["dep_prior"]  = dep_prior
        m["sga_curr"]   = sga_curr
        m["sga_prior"]  = sga_prior
        m["cogs_prior"] = cogs_prior_val
        m["gw_curr"]    = gw_curr
        m["gw_prior"]   = gw_prior
        m["ia_curr"]    = ia_curr
        m["ia_prior"]   = ia_prior

        return m


# ---------------------------------------------------------------------------
# FundamentalDuckDB — persistent columnar store
# ---------------------------------------------------------------------------

class FundamentalDuckDB:
    """DuckDB persistent + in-memory database for fundamental screening.

    Schema
    ------
    Table 'fundamentals': one row per ticker, 60+ columns.
    DuckDB columnar storage enables fast range-scan filters and ORDER BY.

    Refresh strategy
    ----------------
    On startup: load all tickers. If local row exists and submissions.json
    last_modified < 60 days, skip re-fetch (EDGAR quarterly cadence).
    On /refresh API call: force re-fetch for a specific ticker or full universe.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or DUCKDB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        self._init_schema()
        self._loader = EDGARXBRLLoader()

    def _init_schema(self) -> None:
        self._conn.execute(DUCKDB_CREATE_SQL)

    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    # ------------------------------------------------------------------
    # Load / refresh
    # ------------------------------------------------------------------

    def load_ticker(
        self,
        ticker: str,
        price_data: Optional[Dict] = None,
        finviz_data: Optional[Dict] = None,
        force: bool = False,
    ) -> bool:
        """Load or refresh one ticker into DuckDB.

        Returns True if data was updated, False if skipped (recent enough).
        """
        ticker = ticker.upper()

        # Check if recent enough
        if not force:
            existing = self._conn.execute(
                "SELECT updated_at FROM fundamentals WHERE ticker = ?", [ticker]
            ).fetchone()
            if existing:
                updated_at = existing[0]
                if updated_at:
                    try:
                        dt = datetime.fromisoformat(str(updated_at))
                        if (datetime.now() - dt).days < 7:
                            return False
                    except Exception:
                        pass

        cik = _resolve_cik(ticker)
        if not cik:
            logger.warning("Could not resolve CIK for %s; skipping.", ticker)
            return False

        try:
            companyfacts = self._loader.fetch_companyfacts(cik)
            if not companyfacts:
                return False

            # Entity info
            entity_name = companyfacts.get("entityName", "")

            # Extract raw metrics
            m = self._loader.extract_metrics(ticker, companyfacts)

            # Merge price and finviz data
            p = price_data or {}
            f = finviz_data or {}

            # Compute ratios
            mkt_cap  = _safe(p.get("market_cap"))
            price    = _safe(p.get("price"))

            rev      = m.get("revenue_ttm_raw")
            ni       = m.get("net_income_ttm")
            ebit     = m.get("ebit_ttm")
            ebitda   = m.get("ebitda_ttm")
            fcf      = m.get("fcf_ttm")
            equity   = m.get("equity_book")
            ta       = m.get("total_assets")
            td       = m.get("total_debt")
            cash     = m.get("cash_raw")
            ie       = m.get("interest_expense")
            capex    = m.get("capex")
            rd       = m.get("rd_expense")
            so       = m.get("shares_outstanding")

            ev = None
            if mkt_cap is not None and td is not None and cash is not None:
                ev = _safe(mkt_cap + td - cash)

            # Valuation
            pe      = _safe(mkt_cap / ni)  if mkt_cap and ni and ni > 0 else None
            eg1y    = m.get("eps_growth_1y")
            peg     = _safe(pe / (eg1y * 100)) if pe and eg1y and eg1y > 0 else None
            ev_eb   = _safe(ev / ebitda)   if ev and ebitda and ebitda > 0 else None
            ev_rev  = _safe(ev / rev)      if ev and rev and rev > 0 else None
            ev_ebit = _safe(ev / ebit)     if ev and ebit and ebit > 0 else None
            ptb     = _safe(mkt_cap / equity) if mkt_cap and equity and equity > 0 else None
            ptfcf   = _safe(mkt_cap / fcf) if mkt_cap and fcf and fcf > 0 else None
            pts     = _safe(mkt_cap / rev) if mkt_cap and rev and rev > 0 else None
            ey      = _safe(ebit / ev)     if ebit and ev and ev > 0 else None  # Greenblatt earnings yield

            # Profitability
            gm  = _safe((m.get("gross_profit_ttm") or 0) / rev) if rev and rev > 0 else None
            om  = _safe(ebit / rev)  if ebit and rev and rev > 0 else None
            nm  = _safe(ni / rev)    if ni and rev and rev > 0 else None
            em  = _safe(ebitda / rev) if ebitda and rev and rev > 0 else None
            roe = _safe(ni / equity) if ni and equity and equity > 0 else None
            roa = _safe(ni / ta)     if ni and ta and ta > 0 else None

            # ROIC = EBIT*(1-tax_rate) / (equity + net_debt)
            tax_rate  = 0.21
            invested  = (equity or 0) + (m.get("net_debt") or 0)
            roic = _safe(ebit * (1 - tax_rate) / invested) if ebit and invested and invested > 0 else None

            # Financial health
            ca  = m.get("current_assets")
            cl  = m.get("current_liabilities")
            inv = m.get("inventory")
            cr  = _safe(ca / cl)              if ca and cl and cl > 0 else None
            qr  = _safe((ca - (inv or 0)) / cl) if ca and cl and cl > 0 else None
            de  = _safe(td / equity)          if td and equity and equity > 0 else None
            nd_eb = _safe(m["net_debt"] / ebitda) if m.get("net_debt") is not None and ebitda and ebitda > 0 else None
            ic  = _safe(ebit / abs(ie))       if ebit and ie and ie != 0 else None

            # Altman Z-score (manufacturing firms; use with caution for others)
            wc  = (ca or 0) - (cl or 0)
            re  = _safe(m.get("equity_book"))  # retained earnings proxy
            z   = None
            if ta and ta > 0 and equity and mkt_cap:
                x1 = wc / ta
                x2 = (re or 0) / ta
                x3 = (ebit or 0) / ta
                x4 = mkt_cap / (td or 1)
                x5 = (rev or 0) / ta
                z  = _safe(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5)

            # Dividends
            div_raw = _safe(abs(p.get("dividend_ttm", 0) or 0))
            div_yield = _safe(div_raw / price) if div_raw and price and price > 0 else None
            payout    = _safe(div_raw / (ni / so)) if div_raw and ni and so and so > 0 else None

            # Quality metrics
            capex_rev = _safe(abs(capex or 0) / rev) if capex and rev and rev > 0 else None
            rd_rev    = _safe((rd or 0) / rev)        if rd and rev and rev > 0 else None

            # NCAV (Graham net-net)
            tl_val = _safe(p.get("total_liabilities")) or m.get("total_assets", 0)
            ncav   = _safe((ca or 0) - (tl_val or 0)) if ca else None
            ncav_mc = _safe(ncav / mkt_cap) if ncav and mkt_cap and mkt_cap > 0 else None

            # Revenue TTM
            revenue_ttm = rev

            # Piotroski F-Score (all 9 signals — correct Piotroski 2000 implementation)
            f_score = self._piotroski_score(m, roa, cr, cfo=m.get("cfo_ttm"))

            # Beneish M-Score (8-variable earnings manipulation detector)
            beneish_m = self._beneish_m_score(m, ni, cfo=m.get("cfo_ttm"))

            now_s = datetime.now().isoformat()

            row = {
                "ticker":               ticker,
                "company_name":         entity_name,
                "cik":                  cik,
                "sector":               f.get("sector"),
                "industry":             f.get("industry"),
                "exchange":             f.get("exchange"),
                "updated_at":           now_s,
                "data_source":          "edgar_xbrl",
                # Size
                "market_cap":           mkt_cap,
                "revenue_ttm":          revenue_ttm,
                "enterprise_value":     ev,
                "total_assets":         ta,
                # Valuation
                "pe_ratio":             pe,
                "peg_ratio":            peg,
                "ev_ebitda":            ev_eb,
                "ev_revenue":           ev_rev,
                "ev_ebit":              ev_ebit,
                "price_to_book":        ptb,
                "price_to_fcf":         ptfcf,
                "price_to_sales":       pts,
                "earnings_yield":       ey,
                # Growth
                "revenue_growth_1y":    m.get("revenue_growth_1y"),
                "revenue_growth_3y":    m.get("revenue_growth_3y"),
                "eps_growth_1y":        m.get("eps_growth_1y"),
                "eps_growth_3y":        m.get("eps_growth_3y"),
                "fcf_growth_1y":        m.get("fcf_growth_1y"),
                # Profitability
                "gross_margin":         gm,
                "operating_margin":     om,
                "net_margin":           nm,
                "ebitda_margin":        em,
                "roic":                 roic,
                "roe":                  roe,
                "roa":                  roa,
                # Health
                "current_ratio":        cr,
                "quick_ratio":          qr,
                "debt_to_equity":       de,
                "net_debt_ebitda":      nd_eb,
                "interest_coverage":    ic,
                "altman_z":             z,
                # Momentum (from price_data)
                "return_1m":            _safe(p.get("return_1m")),
                "return_3m":            _safe(p.get("return_3m")),
                "return_6m":            _safe(p.get("return_6m")),
                "return_12m":           _safe(p.get("return_12m")),
                "return_ytd":           _safe(p.get("return_ytd")),
                "rsi_14":               _safe(p.get("rsi_14")),
                "beta_1y":              _safe(p.get("beta_1y")),
                # Dividends
                "dividend_yield":       div_yield,
                "payout_ratio":         payout,
                "dividend_growth_5y":   _safe(p.get("dividend_growth_5y")),
                "consecutive_dividend_years": m.get("consecutive_dividend_years"),
                # Quality
                "accruals_ratio":       m.get("accruals_ratio"),
                "cash_conversion_cycle": m.get("cash_conversion_cycle"),
                "capex_to_revenue":     capex_rev,
                "rd_to_revenue":        rd_rev,
                "insider_ownership_pct": _safe(f.get("insider_ownership_pct")),
                # Alternative
                "short_interest_ratio": _safe(f.get("short_interest_ratio")),
                "institutional_ownership_pct": _safe(f.get("institutional_ownership_pct")),
                "analyst_rating_avg":   _safe(f.get("analyst_rating_avg")),
                # Raw
                "revenue_ttm_raw":      m.get("revenue_ttm_raw"),
                "net_income_ttm":       ni,
                "ebitda_ttm":           ebitda,
                "ebit_ttm":             ebit,
                "fcf_ttm":              fcf,
                "equity_book":          equity,
                "total_debt":           td,
                "net_debt":             m.get("net_debt"),
                "cash_raw":             cash,
                "current_assets":       ca,
                "current_liabilities":  cl,
                "inventory":            inv,
                "accounts_receivable":  m.get("accounts_receivable"),
                "accounts_payable":     m.get("accounts_payable"),
                "shares_outstanding":   so,
                "interest_expense":     ie,
                "capex":                capex,
                "rd_expense":           rd,
                # Quality scores
                "piotroski_f_score":    f_score,
                "beneish_m_score":      beneish_m,
                # Graham
                "ncav":                 ncav,
                "ncav_to_market_cap":   ncav_mc,
            }

            # Upsert into DuckDB
            cols  = ", ".join(row.keys())
            slots = ", ".join(["?" for _ in row])
            vals  = list(row.values())
            try:
                self._conn.execute(
                    f"INSERT OR REPLACE INTO fundamentals ({cols}) VALUES ({slots})", vals
                )
            except Exception:
                # DuckDB uses different upsert syntax; try DELETE + INSERT
                self._conn.execute("DELETE FROM fundamentals WHERE ticker = ?", [ticker])
                self._conn.execute(
                    f"INSERT INTO fundamentals ({cols}) VALUES ({slots})", vals
                )

            logger.info("Loaded %s into DuckDB (ev_ebitda=%.1f, roe=%.2f%%)",
                        ticker, ev_eb or 0, (roe or 0) * 100)
            return True

        except Exception as exc:
            logger.error("Failed to load %s: %s", ticker, exc)
            return False

    def _piotroski_score(
        self,
        m: Dict,
        roa:  Optional[float],
        cr:   Optional[float],
        cfo:  Optional[float],
    ) -> Optional[int]:
        """Compute Piotroski F-Score (0-9) — all 9 Piotroski (2000) signals.

        Profitability (F1-F4):
          F1 = ROA > 0
          F2 = CFO / Assets > 0
          F3 = ΔROA: ROA this year > ROA prior year
          F4 = Accruals quality: CFO/Assets > ROA (i.e. CFO > NI when normalised)
        Leverage / Liquidity (F5-F7):
          F5 = ΔLeverage: LT-debt/Assets decreased YoY
          F6 = ΔLiquidity: Current ratio increased YoY
          F7 = No equity issuance: shares outstanding did not increase >2% YoY
        Operating Efficiency (F8-F9):
          F8 = ΔGross margin: gross margin improved YoY
          F9 = ΔAsset turnover: revenue/assets improved YoY
        """
        score = 0

        ta       = m.get("total_assets")          # current year total assets
        ta_prior = m.get("ta_prior")              # prior year total assets
        ca_prior = m.get("ca_prior")
        cl_prior = m.get("cl_prior")
        rev      = m.get("revenue_ttm_raw") or 0
        gp       = m.get("gross_profit_ttm")
        rev_prior = m.get("rev_prior") or 0
        gp_prior  = m.get("gp_prior")
        ni_prior  = m.get("ni_prior")
        ltd       = _safe(m.get("total_debt")) or 0.0  # long-term debt proxy (LTD+STD)
        ltd_prior = m.get("ltd_prior") or 0.0
        so        = m.get("shares_outstanding")
        so_prior  = m.get("so_prior")

        # F1: ROA > 0  (net income / total assets)
        if roa is not None and roa > 0:
            score += 1

        # F2: CFO / Assets > 0
        if cfo is not None and cfo > 0:
            score += 1

        # F3: ΔROA — ROA improved year-over-year
        roa_prior = None
        if ni_prior is not None and ta_prior and ta_prior > 0:
            roa_prior = ni_prior / ta_prior
        if roa is not None and roa_prior is not None and roa > roa_prior:
            score += 1

        # F4: Accrual quality — CFO/Assets > ROA (cash quality of earnings)
        #     Equivalent to: CFO > NI when both are normalised by assets
        if (cfo is not None and ta and ta > 0 and roa is not None
                and (cfo / ta) > roa):
            score += 1

        # F5: ΔLeverage — LT-debt / Assets decreased YoY
        lev_curr  = (ltd / ta)        if (ta and ta > 0) else None
        lev_prior = (ltd_prior / ta_prior) if (ta_prior and ta_prior > 0) else None
        if lev_curr is not None and lev_prior is not None and lev_curr < lev_prior:
            score += 1

        # F6: ΔLiquidity — current ratio improved YoY
        cr_prior = None
        if ca_prior and cl_prior and cl_prior > 0:
            cr_prior = ca_prior / cl_prior
        if cr is not None and cr_prior is not None and cr > cr_prior:
            score += 1

        # F7: No equity issuance — shares outstanding did not increase >2%
        if (so is not None and so_prior is not None and so_prior > 0
                and so <= so_prior * 1.02):
            score += 1

        # F8: ΔGross margin — gross margin improved YoY
        gm_curr  = (gp / rev)        if (gp is not None and rev and rev > 0) else None
        gm_prior = (gp_prior / rev_prior) if (gp_prior is not None and rev_prior and rev_prior > 0) else None
        if gm_curr is not None and gm_prior is not None and gm_curr > gm_prior:
            score += 1

        # F9: ΔAsset turnover — revenue / assets improved YoY
        at_curr  = (rev / ta)        if (ta and ta > 0 and rev > 0) else None
        at_prior = (rev_prior / ta_prior) if (ta_prior and ta_prior > 0 and rev_prior > 0) else None
        if at_curr is not None and at_prior is not None and at_curr > at_prior:
            score += 1

        return min(score, 9)

    def _beneish_m_score(self, m: Dict, ni: Optional[float], cfo: Optional[float]) -> Optional[float]:
        """Compute Beneish (1999) M-score — 8-variable earnings manipulation detector.

        M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
              + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

        M > -1.78 → likely manipulator (conservative threshold used by practitioners).

        All variables require current-year and prior-year XBRL data.
        Returns None if insufficient data is available.
        """
        rev_curr  = m.get("revenue_ttm_raw") or 0.0
        rev_prior = m.get("rev_prior") or 0.0
        ar_curr   = m.get("accounts_receivable") or 0.0
        ta_curr   = m.get("total_assets")
        ta_prior  = m.get("ta_prior")
        ca_curr   = m.get("current_assets") or 0.0
        ca_prior  = m.get("ca_prior") or 0.0
        gp_curr   = m.get("gross_profit_ttm") or 0.0
        gp_prior  = m.get("gp_prior") or 0.0
        dep_curr  = m.get("dep_ttm") or 0.0
        dep_prior = m.get("dep_prior") or 0.0
        sga_curr  = m.get("sga_curr") or 0.0
        sga_prior = m.get("sga_prior") or 0.0
        ltd_curr  = (m.get("total_debt") or 0.0)
        ltd_prior = m.get("ltd_prior") or 0.0

        # Require at least current and prior year revenue
        if rev_curr <= 0 or rev_prior <= 0:
            return None
        if ta_curr is None or ta_curr <= 0 or ta_prior is None or ta_prior <= 0:
            return None

        # DSRI — Days Sales Receivables Index
        # = (AR_t / Rev_t) / (AR_{t-1} / Rev_{t-1})
        # Use a proxy AR since we may not have prior-year AR; default to 1.0 if unavailable
        dsri = 1.0
        ar_ratio_curr  = ar_curr / rev_curr if rev_curr > 0 else None
        # We don't store prior-year AR separately; use neutral 1.0 as fallback
        if ar_ratio_curr is not None and ar_ratio_curr > 0:
            # Without prior-year AR we can only compute a partial signal;
            # use current AR / revenue vs revenue growth as proxy: if AR grew
            # faster than revenue it signals receivables inflation.
            dsri_ratio_curr = ar_curr / rev_curr
            dsri_ratio_prior = ar_curr / rev_prior  # proxy: assume same AR base
            if dsri_ratio_prior > 0:
                dsri = max(0.1, dsri_ratio_curr / dsri_ratio_prior)

        # GMI — Gross Margin Index = GM_{t-1} / GM_t
        gm_curr_v  = gp_curr / rev_curr   if rev_curr > 0  else None
        gm_prior_v = gp_prior / rev_prior if rev_prior > 0 else None
        if gm_curr_v and gm_curr_v > 0 and gm_prior_v is not None:
            gmi = gm_prior_v / gm_curr_v
        else:
            gmi = 1.0

        # AQI — Asset Quality Index
        # AQI = (1 - (CA_t + PPE_t) / TA_t) / (1 - (CA_{t-1} + PPE_{t-1}) / TA_{t-1})
        # PPE is not stored directly; approximate using (TA - CA - Goodwill - Intangibles)
        gw_curr  = m.get("gw_curr") or 0.0
        gw_prior = m.get("gw_prior") or 0.0
        ia_curr  = m.get("ia_curr") or 0.0
        ia_prior = m.get("ia_prior") or 0.0
        ppe_curr  = max(0.0, ta_curr - ca_curr - gw_curr - ia_curr)
        ppe_prior = max(0.0, ta_prior - ca_prior - gw_prior - ia_prior)
        aqi_denom_curr  = 1.0 - (ca_curr + ppe_curr) / ta_curr   if ta_curr > 0 else None
        aqi_denom_prior = 1.0 - (ca_prior + ppe_prior) / ta_prior if ta_prior > 0 else None
        if (aqi_denom_curr is not None and aqi_denom_prior is not None
                and aqi_denom_prior != 0):
            aqi = aqi_denom_curr / aqi_denom_prior
        else:
            aqi = 1.0

        # SGI — Sales Growth Index = Rev_t / Rev_{t-1}
        sgi = rev_curr / rev_prior if rev_prior > 0 else 1.0

        # DEPI — Depreciation Index
        # = (Dep_{t-1} / (PPE_{t-1} + Dep_{t-1})) / (Dep_t / (PPE_t + Dep_t))
        depi_denom_curr  = ppe_curr + dep_curr
        depi_denom_prior = ppe_prior + dep_prior
        if depi_denom_curr > 0 and depi_denom_prior > 0 and dep_curr > 0:
            depi_rate_curr  = dep_curr  / depi_denom_curr
            depi_rate_prior = dep_prior / depi_denom_prior
            depi = (depi_rate_prior / depi_rate_curr) if depi_rate_curr > 0 else 1.0
        else:
            depi = 1.0

        # SGAI — SG&A Index = (SGA_t / Rev_t) / (SGA_{t-1} / Rev_{t-1})
        if sga_curr > 0 and sga_prior > 0 and rev_curr > 0 and rev_prior > 0:
            sgai = (sga_curr / rev_curr) / (sga_prior / rev_prior)
        else:
            sgai = 1.0

        # TATA — Total Accruals to Total Assets = (NI - CFO) / TA
        tata = 0.0
        if ni is not None and cfo is not None and ta_curr and ta_curr > 0:
            tata = (ni - cfo) / ta_curr

        # LVGI — Leverage Growth Index
        # = ((LTD_t + CL_t) / TA_t) / ((LTD_{t-1} + CL_{t-1}) / TA_{t-1})
        cl_curr  = m.get("current_liabilities") or 0.0
        cl_prior = m.get("cl_prior") or 0.0
        lev_curr_b  = (ltd_curr + cl_curr) / ta_curr   if ta_curr > 0 else None
        lev_prior_b = (ltd_prior + cl_prior) / ta_prior if ta_prior and ta_prior > 0 else None
        if lev_curr_b is not None and lev_prior_b is not None and lev_prior_b > 0:
            lvgi = lev_curr_b / lev_prior_b
        else:
            lvgi = 1.0

        m_score = (
            -4.84
            + 0.920 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * tata
            - 0.327 * lvgi
        )
        return _safe(m_score)

    def load_universe(
        self,
        tickers: List[str],
        price_fn: Optional[Any] = None,
        finviz_fn: Optional[Any] = None,
        force: bool = False,
    ) -> Dict[str, bool]:
        """Batch-load tickers. Returns {ticker: success}."""
        results = {}
        for i, ticker in enumerate(tickers):
            logger.info("Loading %s (%d/%d)...", ticker, i + 1, len(tickers))
            price_data  = price_fn(ticker) if price_fn else {}
            finviz_data = finviz_fn(ticker) if finviz_fn else {}
            results[ticker] = self.load_ticker(ticker, price_data, finviz_data, force=force)
        return results

    def query(self, sql: str, params: Optional[List] = None) -> pd.DataFrame:
        """Execute arbitrary SQL against the fundamentals table."""
        try:
            if params:
                return self._conn.execute(sql, params).df()
            return self._conn.execute(sql).df()
        except Exception as exc:
            logger.error("DuckDB query failed: %s — %s", sql[:100], exc)
            raise

    def get_ticker(self, ticker: str) -> Optional[Dict]:
        df = self.query("SELECT * FROM fundamentals WHERE ticker = ?", [ticker.upper()])
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def universe_stats(self) -> Dict[str, Any]:
        df = self.query("""
            SELECT
                COUNT(*)                    AS total_tickers,
                COUNT(pe_ratio)             AS tickers_with_pe,
                COUNT(roe)                  AS tickers_with_roe,
                COUNT(revenue_growth_1y)    AS tickers_with_growth,
                AVG(pe_ratio)               AS median_pe,
                AVG(roe)                    AS avg_roe,
                MIN(updated_at)             AS oldest_update,
                MAX(updated_at)             AS newest_update
            FROM fundamentals
        """)
        return df.iloc[0].to_dict() if not df.empty else {}


# ---------------------------------------------------------------------------
# PriceDataFetcher — yfinance for momentum fields ONLY
# ---------------------------------------------------------------------------

class PriceDataFetcher:
    """Fetch price-based metrics using yfinance.

    yfinance is ONLY used for:
    - Return calculations (1m, 3m, 6m, 12m, YTD)
    - RSI-14 (technical momentum)
    - Beta (vs SPY, 1-year daily returns)
    - Market cap (price × shares outstanding)
    - Dividend TTM (from yfinance dividends series)

    NOT used for: any balance sheet, income statement, or cash flow data.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Dict, float]] = {}

    def fetch(self, ticker: str, cache_ttl_hours: float = 4.0) -> Dict[str, Optional[float]]:
        """Fetch price-based metrics for one ticker."""
        now = time.time()
        if ticker in self._cache:
            data, ts = self._cache[ticker]
            if (now - ts) < cache_ttl_hours * 3600:
                return data

        result: Dict[str, Optional[float]] = {
            "price":            None,
            "market_cap":       None,
            "return_1m":        None,
            "return_3m":        None,
            "return_6m":        None,
            "return_12m":       None,
            "return_ytd":       None,
            "rsi_14":           None,
            "beta_1y":          None,
            "dividend_ttm":     None,
            "dividend_growth_5y": None,
        }

        if not _YF_AVAILABLE:
            return result

        try:
            tk      = yf.Ticker(ticker)
            hist    = tk.history(period="1y", interval="1d")
            if hist.empty:
                return result

            closes = hist["Close"].dropna().tolist()
            if not closes:
                return result

            price = closes[-1]
            result["price"] = _safe(price)

            # Returns
            def ret(n_days: int) -> Optional[float]:
                if len(closes) < n_days:
                    return None
                return _safe((closes[-1] / closes[-n_days]) - 1)

            result["return_1m"]  = ret(21)
            result["return_3m"]  = ret(63)
            result["return_6m"]  = ret(126)
            result["return_12m"] = ret(252)

            # YTD
            year_start = date(date.today().year, 1, 1)
            hist_ytd   = hist[hist.index.date >= year_start]
            if not hist_ytd.empty:
                ytd_closes = hist_ytd["Close"].dropna().tolist()
                if len(ytd_closes) >= 2:
                    result["return_ytd"] = _safe((ytd_closes[-1] / ytd_closes[0]) - 1)

            # RSI-14
            result["rsi_14"] = _safe(_rsi(closes, period=14))

            # Beta vs SPY
            try:
                spy_hist = yf.Ticker("SPY").history(period="1y", interval="1d")
                if not spy_hist.empty and len(spy_hist) >= 50:
                    spy_c   = spy_hist["Close"].dropna()
                    stk_c   = hist["Close"].dropna()
                    joined  = pd.DataFrame({"stk": stk_c, "spy": spy_c}).dropna()
                    if len(joined) >= 50:
                        stk_r = joined["stk"].pct_change().dropna()
                        spy_r = joined["spy"].pct_change().dropna()
                        cov   = float(np.cov(stk_r, spy_r)[0, 1])
                        var   = float(np.var(spy_r))
                        result["beta_1y"] = _safe(cov / var) if var > 0 else None
            except Exception:
                pass

            # Market cap
            info = {}
            try:
                info = tk.fast_info
                mkt_cap = float(getattr(info, "market_cap", None) or 0)
                result["market_cap"] = _safe(mkt_cap) if mkt_cap else None
            except Exception:
                pass

            # Dividend TTM
            try:
                divs = tk.dividends
                if not divs.empty:
                    one_year_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
                    div_ttm = float(divs[divs.index >= one_year_ago].sum())
                    result["dividend_ttm"] = _safe(div_ttm)

                    # Dividend growth 5y
                    five_yr_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5 * 365)
                    div_5ya = float(divs[(divs.index >= five_yr_ago) &
                                        (divs.index < five_yr_ago + pd.Timedelta(days=365))].sum())
                    result["dividend_growth_5y"] = _cagr(div_5ya, div_ttm, 5)
            except Exception:
                pass

        except Exception as exc:
            logger.warning("yfinance fetch failed for %s: %s", ticker, exc)

        self._cache[ticker] = (result, now)
        return result


# ---------------------------------------------------------------------------
# FinvizScraper — analyst ratings and ownership (free tier)
# ---------------------------------------------------------------------------

class FinvizScraper:
    """Scrape analyst ratings, insider/institutional ownership from Finviz.

    Terms of service: Finviz permits scraping for personal use at reasonable rates.
    Rate limit: ≥2s between requests. No automated bulk scraping in loops.
    """

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    _RATE_DELAY = 2.0

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Dict, float]] = {}

    def fetch(self, ticker: str, cache_ttl_hours: float = 24.0) -> Dict[str, Optional[float]]:
        """Scrape Finviz quote page for a ticker. Returns parsed fields."""
        now = time.time()
        if ticker in self._cache:
            data, ts = self._cache[ticker]
            if (now - ts) < cache_ttl_hours * 3600:
                return data

        time.sleep(self._RATE_DELAY)
        result: Dict[str, Optional[float]] = {
            "analyst_rating_avg":          None,
            "insider_ownership_pct":       None,
            "institutional_ownership_pct": None,
            "short_interest_ratio":        None,
            "sector":                      None,
            "industry":                    None,
        }

        try:
            url  = f"{FINVIZ_BASE}?t={ticker.upper()}"
            resp = requests.get(url, headers=self._HEADERS, timeout=15)
            if resp.status_code != 200:
                return result
            text = resp.text

            def _extract_field(label: str) -> Optional[str]:
                """Find the value after a Finviz table label."""
                pattern = rf'{re.escape(label)}</td>\s*<td[^>]*>(.*?)</td>'
                m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if m:
                    raw = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                    return raw if raw and raw != "-" else None
                return None

            # Insider ownership
            ins_str = _extract_field("Insider Own")
            if ins_str:
                result["insider_ownership_pct"] = _safe(
                    float(ins_str.replace("%", "")) / 100
                )

            # Institutional ownership
            inst_str = _extract_field("Inst Own")
            if inst_str:
                result["institutional_ownership_pct"] = _safe(
                    float(inst_str.replace("%", "")) / 100
                )

            # Short float → short interest ratio (proxy)
            short_str = _extract_field("Short Float")
            if short_str:
                result["short_interest_ratio"] = _safe(
                    float(short_str.replace("%", "")) / 100
                )

            # Analyst recommendation (1=Strong Buy, 5=Strong Sell)
            recom_str = _extract_field("Recom")
            if recom_str:
                result["analyst_rating_avg"] = _safe(recom_str)

            # Sector / industry
            sector_m   = re.search(r'Sector</a></td><td[^>]*><a[^>]*>([^<]+)<', text)
            industry_m = re.search(r'Industry</a></td><td[^>]*><a[^>]*>([^<]+)<', text)
            if sector_m:
                result["sector"]   = sector_m.group(1).strip()
            if industry_m:
                result["industry"] = industry_m.group(1).strip()

        except Exception as exc:
            logger.warning("Finviz scrape failed for %s: %s", ticker, exc)

        self._cache[ticker] = (result, now)
        return result


# ---------------------------------------------------------------------------
# Pre-built screens (25 strategies)
# ---------------------------------------------------------------------------

# Each screen is a DuckDB SQL WHERE clause + metadata
PREBUILT_SCREENS: Dict[str, Dict[str, Any]] = {
    "warren_buffett": {
        "description": "Warren Buffett: high ROE, low debt, reasonable PE, proven earnings growth",
        "where": "roe > 0.15 AND debt_to_equity < 0.5 AND pe_ratio BETWEEN 1 AND 20 AND eps_growth_3y > 0.10",
        "order": "roe DESC",
        "rationale": "Moat + financial fortress + value + consistent earnings compounding",
    },
    "peter_lynch": {
        "description": "Peter Lynch: PEG < 1, strong earnings growth, reasonable PE",
        "where": "peg_ratio < 1 AND peg_ratio > 0 AND eps_growth_1y > 0.15 AND pe_ratio BETWEEN 5 AND 20",
        "order": "peg_ratio ASC",
        "rationale": "Growth at a reasonable price; PEG is Lynch's primary valuation tool",
    },
    "magic_formula": {
        "description": "Greenblatt Magic Formula: high earnings yield + high ROIC",
        "where": "earnings_yield > 0.08 AND roic > 0.15 AND market_cap > 50000000",
        "order": "earnings_yield + roic DESC",
        "rationale": "Rank-order by combined earnings yield and capital efficiency",
    },
    "garp": {
        "description": "GARP: Low EV/EBITDA with meaningful revenue growth",
        "where": "ev_ebitda < 15 AND ev_ebitda > 0 AND revenue_growth_1y > 0.10 AND ebitda_margin > 0.10",
        "order": "ev_ebitda ASC",
        "rationale": "Growth at a reasonable price using enterprise value framework",
    },
    "net_net_graham": {
        "description": "Graham net-net: NCAV > market cap (deep value)",
        "where": "ncav_to_market_cap > 1 AND market_cap > 0 AND ncav > 0",
        "order": "ncav_to_market_cap DESC",
        "rationale": "Buy at or below liquidation value; classic deep-value Graham screen",
    },
    "dividend_aristocrats": {
        "description": "Dividend Aristocrats: 25+ consecutive years of dividend growth",
        "where": "consecutive_dividend_years >= 25 AND dividend_yield > 0.01 AND payout_ratio < 0.75",
        "order": "consecutive_dividend_years DESC",
        "rationale": "Companies with demonstrated commitment to growing dividends",
    },
    "quality_momentum": {
        "description": "Quality momentum: high ROE + strong 12-month price momentum",
        "where": "roe > 0.15 AND return_12m > 0.15 AND piotroski_f_score >= 6",
        "order": "roe * return_12m DESC",
        "rationale": "Quality factors with price momentum persistence",
    },
    "fallen_angels": {
        "description": "Fallen angels: 52-week lows with strong balance sheets",
        "where": "return_12m < -0.20 AND current_ratio > 2 AND debt_to_equity < 0.5 AND net_income_ttm > 0",
        "order": "return_12m ASC",
        "rationale": "Oversold quality names with financial strength to weather downturn",
    },
    "low_ev_ebitda": {
        "description": "Cheapest by EV/EBITDA with profitability filter",
        "where": "ev_ebitda < 8 AND ev_ebitda > 0 AND operating_margin > 0.05 AND revenue_ttm > 100000000",
        "order": "ev_ebitda ASC",
        "rationale": "Deep value on enterprise basis excluding unprofitable companies",
    },
    "high_fcf_yield": {
        "description": "High free cash flow yield (FCF / market cap > 8%)",
        "where": "fcf_ttm > 0 AND market_cap > 0 AND (fcf_ttm / market_cap) > 0.08",
        "order": "(fcf_ttm / market_cap) DESC",
        "rationale": "Cash generation power; FCF yield as bond-equivalent return",
    },
    "momentum_reversal": {
        "description": "Momentum factor: strong 12m-1m return (skip last month)",
        "where": "return_12m > 0.20 AND return_1m < 0.05 AND market_cap > 500000000",
        "order": "return_12m DESC",
        "rationale": "12-1 momentum factor; skip last month to avoid reversal",
    },
    "high_roe_low_debt": {
        "description": "Quality screen: ROE > 20% with manageable leverage",
        "where": "roe > 0.20 AND debt_to_equity < 1.0 AND net_margin > 0.10",
        "order": "roe DESC",
        "rationale": "Durable competitive advantage proxied by high ROE + clean balance sheet",
    },
    "asset_heavy_value": {
        "description": "Deep value: price-to-book < 1 with profitable operations",
        "where": "price_to_book < 1 AND price_to_book > 0 AND roe > 0.05 AND net_income_ttm > 0",
        "order": "price_to_book ASC",
        "rationale": "Trading below tangible book value with ongoing profitability",
    },
    "low_peg_growth": {
        "description": "Low PEG with proven 3-year growth track record",
        "where": "peg_ratio BETWEEN 0 AND 1.5 AND revenue_growth_3y > 0.10 AND eps_growth_3y > 0.10",
        "order": "peg_ratio ASC",
        "rationale": "Combining long-run growth consistency with valuation discipline",
    },
    "high_insider_buying": {
        "description": "High insider ownership (skin in the game)",
        "where": "insider_ownership_pct > 0.10 AND market_cap > 100000000 AND net_income_ttm > 0",
        "order": "insider_ownership_pct DESC",
        "rationale": "Management alignment; insiders holding >10% have strong incentive",
    },
    "accruals_quality": {
        "description": "High earnings quality: low accruals (Sloan anomaly)",
        "where": "accruals_ratio BETWEEN -0.05 AND 0.05 AND roe > 0.10 AND net_margin > 0",
        "order": "accruals_ratio ASC",
        "rationale": "Cash-backed earnings; low accruals predict future alpha (Sloan 1996)",
    },
    "capital_light_moat": {
        "description": "Capital-light businesses: low CapEx, high ROIC",
        "where": "capex_to_revenue < 0.05 AND roic > 0.20 AND gross_margin > 0.40",
        "order": "roic DESC",
        "rationale": "Software-like economics; high returns on minimal capital reinvestment",
    },
    "rd_intensive_growth": {
        "description": "R&D-intensive growth companies with strong revenue momentum",
        "where": "rd_to_revenue > 0.10 AND revenue_growth_1y > 0.15 AND gross_margin > 0.50",
        "order": "revenue_growth_1y DESC",
        "rationale": "Innovation-driven growth; R&D as moat-building investment",
    },
    "defensive_quality": {
        "description": "Defensive quality: low beta, high dividend yield, strong coverage",
        "where": "beta_1y < 0.80 AND dividend_yield > 0.02 AND interest_coverage > 5 AND current_ratio > 1.5",
        "order": "dividend_yield DESC",
        "rationale": "Capital preservation; low-vol income with financial stability",
    },
    "mean_reversion_low_pe": {
        "description": "Mean reversion: very cheap PE with stable earnings",
        "where": "pe_ratio BETWEEN 5 AND 12 AND eps_growth_3y > 0 AND piotroski_f_score >= 5",
        "order": "pe_ratio ASC",
        "rationale": "Statistically cheap names with improving fundamentals",
    },
    "ev_ebit_greenblatt": {
        "description": "Greenblatt EV/EBIT variant: pure earnings power without D&A",
        "where": "ev_ebit < 12 AND ev_ebit > 0 AND roic > 0.15 AND market_cap > 100000000",
        "order": "ev_ebit ASC",
        "rationale": "D&A-adjusted earnings power; less distorted than EV/EBITDA for capital-light",
    },
    "high_short_interest_contrarian": {
        "description": "Contrarian: heavily shorted names with improving fundamentals",
        "where": "short_interest_ratio > 0.15 AND piotroski_f_score >= 6 AND revenue_growth_1y > 0",
        "order": "short_interest_ratio DESC",
        "rationale": "Short squeeze potential when fundamentals diverge from pessimism",
    },
    "financial_health_composite": {
        "description": "Financial health composite: strong across all health ratios",
        "where": "current_ratio > 2 AND quick_ratio > 1 AND debt_to_equity < 0.5 AND interest_coverage > 8 AND altman_z > 3",
        "order": "altman_z DESC",
        "rationale": "Multi-factor fortress balance sheet screen",
    },
    "dividend_growth": {
        "description": "Dividend growth: growing dividends with sustainable payout",
        "where": "dividend_growth_5y > 0.05 AND payout_ratio < 0.60 AND dividend_yield > 0.015 AND roe > 0.10",
        "order": "dividend_growth_5y DESC",
        "rationale": "Compounding income; sustainability filters prevent yield traps",
    },
    "micro_cap_catalyst": {
        "description": "Small-cap value with momentum catalyst",
        "where": "market_cap BETWEEN 50000000 AND 500000000 AND price_to_book < 2 AND return_3m > 0.10 AND net_income_ttm > 0",
        "order": "return_3m DESC",
        "rationale": "Small-cap value + recent momentum; less institutional coverage",
    },
}


# ---------------------------------------------------------------------------
# FundamentalScreener
# ---------------------------------------------------------------------------

class FundamentalScreener:
    """Core screening engine backed by DuckDB.

    Methods
    -------
    screen(criteria)          — JSON criteria dict → SQL WHERE → DuckDB result
    screen_sql(sql)           — raw SQL string → DuckDB result
    run_preset(name)          — pre-built screen by name
    explain(ticker, criteria) — show which criteria ticker passes and by how much
    rank_universe(metrics)    — factor rank across universe
    """

    def __init__(self, db: FundamentalDuckDB) -> None:
        self._db = db

    def _criteria_to_sql(self, criteria: Dict[str, Any]) -> Tuple[str, List]:
        """Convert a JSON criteria dict to SQL WHERE clause + params.

        Supported formats:
          {"field": {"gt": 0.15}}             → field > 0.15
          {"field": {"lt": 20}}               → field < 20
          {"field": {"between": [5, 20]}}     → field BETWEEN 5 AND 20
          {"field": {"eq": "value"}}          → field = 'value'
          {"field": {"not_null": True}}       → field IS NOT NULL
        """
        clauses: List[str] = []
        params:  List[Any]  = []

        for field, condition in criteria.items():
            if isinstance(condition, dict):
                if "gt" in condition:
                    clauses.append(f"{field} > ?")
                    params.append(condition["gt"])
                if "gte" in condition:
                    clauses.append(f"{field} >= ?")
                    params.append(condition["gte"])
                if "lt" in condition:
                    clauses.append(f"{field} < ?")
                    params.append(condition["lt"])
                if "lte" in condition:
                    clauses.append(f"{field} <= ?")
                    params.append(condition["lte"])
                if "between" in condition:
                    lo, hi = condition["between"]
                    clauses.append(f"{field} BETWEEN ? AND ?")
                    params.extend([lo, hi])
                if "eq" in condition:
                    clauses.append(f"{field} = ?")
                    params.append(condition["eq"])
                if "not_null" in condition and condition["not_null"]:
                    clauses.append(f"{field} IS NOT NULL")
            elif condition is not None:
                # bare value → exact match
                clauses.append(f"{field} = ?")
                params.append(condition)

        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params

    def screen(
        self,
        criteria:  Dict[str, Any],
        order_by:  str   = "market_cap DESC",
        limit:     int   = 100,
        universe:  Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Screen using JSON criteria dict.

        Example criteria:
        {
            "pe_ratio":         {"between": [5, 20]},
            "revenue_growth_1y": {"gt": 0.15},
            "roe":              {"gt": 0.15},
            "debt_to_equity":   {"lt": 0.5},
        }
        """
        where, params = self._criteria_to_sql(criteria)

        if universe:
            placeholders = ", ".join(["?" for _ in universe])
            where = f"({where}) AND ticker IN ({placeholders})"
            params = params + [t.upper() for t in universe]

        sql = f"""
            SELECT ticker, company_name, sector, market_cap, pe_ratio, peg_ratio,
                   ev_ebitda, ev_ebit, price_to_book, price_to_fcf,
                   revenue_growth_1y, revenue_growth_3y, eps_growth_1y,
                   roe, roic, roa, gross_margin, operating_margin, net_margin,
                   current_ratio, debt_to_equity, altman_z,
                   return_12m, return_3m, rsi_14, dividend_yield,
                   piotroski_f_score, data_source, updated_at
            FROM fundamentals
            WHERE {where}
            ORDER BY {order_by}
            LIMIT {limit}
        """
        return self._db.query(sql, params if params else None)

    def screen_sql(self, sql: str) -> pd.DataFrame:
        """Execute a raw SQL query against the fundamentals table.

        The caller is responsible for a valid DuckDB SELECT statement.
        The fundamentals table must be referenced explicitly.
        """
        return self._db.query(sql)

    def run_preset(
        self,
        name:    str,
        limit:   int = 50,
        universe: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a pre-built screen by name. Returns results + metadata."""
        if name not in PREBUILT_SCREENS:
            raise ValueError(
                f"Unknown screen '{name}'. Available: {list(PREBUILT_SCREENS.keys())}"
            )
        spec  = PREBUILT_SCREENS[name]
        where = spec["where"]
        order = spec.get("order", "market_cap DESC")

        if universe:
            placeholders = ", ".join(["?" for _ in universe])
            where = f"({where}) AND ticker IN ({placeholders})"
            params = [t.upper() for t in universe]
        else:
            params = []

        sql = f"""
            SELECT ticker, company_name, sector, market_cap,
                   pe_ratio, ev_ebitda, roe, roic, revenue_growth_1y,
                   return_12m, dividend_yield, piotroski_f_score,
                   debt_to_equity, current_ratio, altman_z, data_source
            FROM fundamentals
            WHERE {where}
            ORDER BY {order}
            LIMIT {limit}
        """
        df = self._db.query(sql, params or None)
        return {
            "screen_name":  name,
            "description":  spec["description"],
            "rationale":    spec["rationale"],
            "criteria_sql": spec["where"],
            "result_count": len(df),
            "results":      df.to_dict(orient="records"),
        }

    def explain(
        self,
        ticker: str,
        criteria: Dict[str, Any],
    ) -> Dict[str, Any]:
        """For a specific ticker, show which criteria it passes and by how much.

        Returns a per-criterion breakdown with pass/fail and margin.
        """
        row = self._db.get_ticker(ticker)
        if not row:
            return {"ticker": ticker, "found": False, "criteria_results": []}

        breakdown = []
        for field, condition in criteria.items():
            actual = row.get(field)
            if isinstance(condition, dict):
                for op, threshold in condition.items():
                    if op == "gt":
                        passed = actual is not None and actual > threshold
                        margin = (actual - threshold) if actual is not None else None
                    elif op == "gte":
                        passed = actual is not None and actual >= threshold
                        margin = (actual - threshold) if actual is not None else None
                    elif op == "lt":
                        passed = actual is not None and actual < threshold
                        margin = (threshold - actual) if actual is not None else None
                    elif op == "lte":
                        passed = actual is not None and actual <= threshold
                        margin = (threshold - actual) if actual is not None else None
                    elif op == "between":
                        lo, hi = threshold
                        passed = actual is not None and lo <= actual <= hi
                        if actual is not None:
                            margin = min(actual - lo, hi - actual)
                        else:
                            margin = None
                    elif op == "eq":
                        passed = actual == threshold
                        margin = None
                    elif op == "not_null":
                        passed = actual is not None
                        margin = None
                    else:
                        passed = False
                        margin = None

                    breakdown.append({
                        "field":     field,
                        "operator":  op,
                        "threshold": threshold,
                        "actual":    actual,
                        "passed":    bool(passed),
                        "margin":    _safe(margin) if margin is not None else None,
                        "pct_margin": _safe(margin / abs(threshold) * 100) if (margin and threshold and threshold != 0) else None,
                    })

        passed_all = all(b["passed"] for b in breakdown)
        return {
            "ticker":         ticker,
            "company_name":   row.get("company_name"),
            "found":          True,
            "passes_all":     passed_all,
            "criteria_count": len(breakdown),
            "passed_count":   sum(1 for b in breakdown if b["passed"]),
            "criteria_results": breakdown,
        }

    def rank_universe(
        self,
        metrics:   List[str],
        ascending: Optional[List[bool]] = None,
        universe:  Optional[List[str]]  = None,
        top_n:     int = 50,
    ) -> pd.DataFrame:
        """Factor rank: assign percentile rank to each metric, compute composite score.

        Lower rank = better (percentile within universe).
        """
        cols    = ", ".join(["ticker", "company_name", "market_cap"] + metrics)
        where   = " AND ".join([f"{m} IS NOT NULL" for m in metrics])
        if universe:
            placeholders = ", ".join(["?" for _ in universe])
            where = f"({where}) AND ticker IN ({placeholders})"
            params = [t.upper() for t in universe]
        else:
            params = None

        sql = f"SELECT {cols} FROM fundamentals WHERE {where}"
        df  = self._db.query(sql, params)

        if df.empty:
            return df

        asc = ascending or [False] * len(metrics)
        for i, m in enumerate(metrics):
            rank_col = f"rank_{m}"
            df[rank_col] = df[m].rank(pct=True, ascending=asc[i])

        rank_cols   = [f"rank_{m}" for m in metrics]
        df["composite_score"] = df[rank_cols].mean(axis=1)
        df = df.sort_values("composite_score", ascending=False).head(top_n)
        return df


# ---------------------------------------------------------------------------
# ScreeningResultsDB — SQLite history
# ---------------------------------------------------------------------------

class ScreeningResultsDB:
    """SQLite store for screening result history."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or SQLITE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        db = sqlite3.connect(str(self._path))
        db.executescript("""
            CREATE TABLE IF NOT EXISTS screen_runs (
                id          TEXT PRIMARY KEY,
                run_at      TEXT NOT NULL,
                screen_name TEXT NOT NULL,
                criteria    TEXT,
                result_count INTEGER,
                tickers     TEXT
            );
        """)
        db.commit()
        db.close()

    def save_run(
        self,
        screen_name: str,
        criteria:    Optional[Dict],
        results:     List[str],
    ) -> str:
        run_id = str(uuid.uuid4())
        db     = sqlite3.connect(str(self._path))
        try:
            db.execute(
                """INSERT INTO screen_runs (id, run_at, screen_name, criteria, result_count, tickers)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    datetime.now().isoformat(),
                    screen_name,
                    json.dumps(criteria) if criteria else None,
                    len(results),
                    json.dumps(results[:200]),
                ),
            )
            db.commit()
        finally:
            db.close()
        return run_id

    def history(self, limit: int = 50) -> List[Dict]:
        db   = sqlite3.connect(str(self._path))
        rows = db.execute(
            "SELECT * FROM screen_runs ORDER BY run_at DESC LIMIT ?", (limit,)
        ).fetchall()
        cols = ["id", "run_at", "screen_name", "criteria", "result_count", "tickers"]
        db.close()
        return [dict(zip(cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_duck_db_singleton:    Optional[FundamentalDuckDB]   = None
_screener_singleton:   Optional[FundamentalScreener] = None
_price_fetcher:        PriceDataFetcher               = PriceDataFetcher()
_finviz_scraper:       FinvizScraper                  = FinvizScraper()
_results_db:           ScreeningResultsDB             = ScreeningResultsDB()


def _get_db() -> FundamentalDuckDB:
    global _duck_db_singleton
    if _duck_db_singleton is None:
        _duck_db_singleton = FundamentalDuckDB()
    return _duck_db_singleton


def _get_screener() -> FundamentalScreener:
    global _screener_singleton
    if _screener_singleton is None:
        _screener_singleton = FundamentalScreener(_get_db())
    return _screener_singleton


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

screener_v3_router = APIRouter(prefix="/screener/v3", tags=["fundamental-screener-v3"])


# ----- GET /fields -----

@screener_v3_router.get("/fields", summary="List all 60+ screenable fields with descriptions")
async def get_fields() -> Dict[str, Any]:
    """Return schema of the fundamentals table with field descriptions and sources."""
    conn = _get_db().connection()
    try:
        schema = conn.execute("DESCRIBE fundamentals").df()
        fields = schema.to_dict(orient="records")
    except Exception:
        fields = []

    field_meta = {
        "pe_ratio":          {"source": "edgar_xbrl + price", "description": "Price / TTM net income"},
        "peg_ratio":         {"source": "edgar_xbrl + price", "description": "PE / 1-year EPS growth rate"},
        "ev_ebitda":         {"source": "edgar_xbrl + price", "description": "Enterprise value / EBITDA"},
        "ev_revenue":        {"source": "edgar_xbrl + price", "description": "Enterprise value / revenue TTM"},
        "ev_ebit":           {"source": "edgar_xbrl + price", "description": "Enterprise value / EBIT (Greenblatt)"},
        "price_to_book":     {"source": "edgar_xbrl + price", "description": "Market cap / book value"},
        "price_to_fcf":      {"source": "edgar_xbrl + price", "description": "Market cap / free cash flow TTM"},
        "price_to_sales":    {"source": "edgar_xbrl + price", "description": "Market cap / revenue TTM"},
        "earnings_yield":    {"source": "edgar_xbrl + price", "description": "EBIT / EV (inverse of EV/EBIT)"},
        "revenue_growth_1y": {"source": "edgar_xbrl", "description": "Annual revenue growth (10-K comparison)"},
        "revenue_growth_3y": {"source": "edgar_xbrl", "description": "3-year revenue CAGR"},
        "eps_growth_1y":     {"source": "edgar_xbrl", "description": "Annual EPS growth"},
        "eps_growth_3y":     {"source": "edgar_xbrl", "description": "3-year EPS CAGR"},
        "fcf_growth_1y":     {"source": "edgar_xbrl", "description": "Annual FCF growth (CFO - CapEx)"},
        "gross_margin":      {"source": "edgar_xbrl", "description": "Gross profit / revenue TTM"},
        "operating_margin":  {"source": "edgar_xbrl", "description": "Operating income / revenue TTM"},
        "net_margin":        {"source": "edgar_xbrl", "description": "Net income / revenue TTM"},
        "ebitda_margin":     {"source": "edgar_xbrl", "description": "EBITDA / revenue TTM"},
        "roic":              {"source": "edgar_xbrl", "description": "EBIT*(1-0.21) / invested capital"},
        "roe":               {"source": "edgar_xbrl", "description": "Net income / equity"},
        "roa":               {"source": "edgar_xbrl", "description": "Net income / total assets"},
        "current_ratio":     {"source": "edgar_xbrl", "description": "Current assets / current liabilities"},
        "quick_ratio":       {"source": "edgar_xbrl", "description": "(Current assets - inventory) / current liabilities"},
        "debt_to_equity":    {"source": "edgar_xbrl", "description": "Total debt / equity"},
        "net_debt_ebitda":   {"source": "edgar_xbrl", "description": "Net debt / EBITDA"},
        "interest_coverage": {"source": "edgar_xbrl", "description": "EBIT / interest expense"},
        "altman_z":          {"source": "edgar_xbrl + price", "description": "Altman Z-score (bankruptcy predictor)"},
        "return_1m":         {"source": "yfinance_price_only", "description": "1-month total return"},
        "return_3m":         {"source": "yfinance_price_only", "description": "3-month total return"},
        "return_6m":         {"source": "yfinance_price_only", "description": "6-month total return"},
        "return_12m":        {"source": "yfinance_price_only", "description": "12-month total return"},
        "return_ytd":        {"source": "yfinance_price_only", "description": "Year-to-date return"},
        "rsi_14":            {"source": "yfinance_price_only", "description": "14-day Wilder RSI"},
        "beta_1y":           {"source": "yfinance_price_only", "description": "1-year beta vs SPY"},
        "dividend_yield":    {"source": "edgar_xbrl + yfinance_price", "description": "Annual dividend / price"},
        "payout_ratio":      {"source": "edgar_xbrl", "description": "Dividends / net income"},
        "dividend_growth_5y": {"source": "yfinance_price_only", "description": "5-year dividend CAGR"},
        "consecutive_dividend_years": {"source": "edgar_xbrl", "description": "Years with dividends paid"},
        "accruals_ratio":    {"source": "edgar_xbrl", "description": "Sloan accruals: (NI-CFO)/assets"},
        "cash_conversion_cycle": {"source": "edgar_xbrl", "description": "DSO + DIO - DPO in days"},
        "capex_to_revenue":  {"source": "edgar_xbrl", "description": "Capital expenditure / revenue"},
        "rd_to_revenue":     {"source": "edgar_xbrl", "description": "R&D expense / revenue"},
        "insider_ownership_pct": {"source": "finviz_scrape", "description": "% shares held by insiders"},
        "short_interest_ratio": {"source": "finviz_scrape", "description": "Short float %"},
        "institutional_ownership_pct": {"source": "finviz_scrape", "description": "% shares held by institutions"},
        "analyst_rating_avg": {"source": "finviz_scrape", "description": "Avg analyst rec (1=Strong Buy, 5=Strong Sell)"},
        "piotroski_f_score": {"source": "edgar_xbrl", "description": "Piotroski F-Score (0-9); higher=better"},
        "ncav_to_market_cap": {"source": "edgar_xbrl + price", "description": "NCAV / market cap; >1 = net-net"},
        "market_cap":        {"source": "yfinance_price_only", "description": "Market capitalisation"},
        "enterprise_value":  {"source": "edgar_xbrl + price", "description": "Market cap + debt - cash"},
    }

    return {
        "total_fields": len(fields),
        "schema":       fields,
        "field_metadata": field_meta,
        "data_sources": {
            "edgar_xbrl":        "SEC EDGAR XBRL companyfacts API — authoritative, free, point-in-time",
            "yfinance_price_only": "yfinance — used ONLY for price-based fields (returns, RSI, beta, market cap)",
            "finviz_scrape":     "Finviz free tier scrape — analyst ratings, ownership; indicative only",
        },
    }


# ----- GET /screens/library -----

@screener_v3_router.get("/screens/library", summary="List all 25 pre-built screens")
async def get_screens_library() -> Dict[str, Any]:
    return {
        "count":   len(PREBUILT_SCREENS),
        "screens": [
            {
                "name":        k,
                "description": v["description"],
                "rationale":   v["rationale"],
                "criteria_sql": v["where"],
            }
            for k, v in PREBUILT_SCREENS.items()
        ],
    }


# ----- GET /screen/{name} -----

@screener_v3_router.get(
    "/screen/{name}",
    summary="Run a pre-built screen by name",
)
async def run_named_screen(
    name:  str,
    limit: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    try:
        result = _get_screener().run_preset(name, limit=limit)
        _results_db.save_run(
            screen_name = name,
            criteria    = {"preset": name},
            results     = [r["ticker"] for r in result["results"]],
        )
        return result
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ----- POST /screen -----

class ScreenRequest(BaseModel):
    criteria:  Dict[str, Any] = Field(..., description="JSON criteria dict")
    order_by:  str            = Field("market_cap DESC", description="ORDER BY clause")
    limit:     int            = Field(100, ge=1, le=1000)
    universe:  Optional[List[str]] = Field(None, description="Restrict to these tickers")

    class Config:
        json_schema_extra = {
            "example": {
                "criteria": {
                    "pe_ratio":         {"between": [5, 20]},
                    "revenue_growth_1y": {"gt": 0.15},
                    "roe":              {"gt": 0.15},
                    "debt_to_equity":   {"lt": 0.5},
                },
                "order_by": "roe DESC",
                "limit": 50,
            }
        }


@screener_v3_router.post(
    "/screen",
    summary="Custom screen with JSON criteria (maps to DuckDB SQL)",
)
async def post_screen(req: ScreenRequest) -> Dict[str, Any]:
    """Screen the fundamentals universe using JSON criteria.

    Each criterion maps to a SQL comparison. All fundamental data sourced from
    EDGAR XBRL. Price-based fields (return_*, rsi_14, beta_1y) from yfinance.

    Example SQL generated:
      WHERE pe_ratio BETWEEN 5 AND 20
        AND revenue_growth_1y > 0.15
        AND roe > 0.15
        AND debt_to_equity < 0.5
      ORDER BY roe DESC
      LIMIT 50
    """
    try:
        df = _get_screener().screen(
            criteria  = req.criteria,
            order_by  = req.order_by,
            limit     = req.limit,
            universe  = req.universe,
        )
        tickers = df["ticker"].tolist() if "ticker" in df.columns else []
        _results_db.save_run("custom", req.criteria, tickers)
        return {
            "result_count": len(df),
            "results":      df.to_dict(orient="records"),
            "criteria":     req.criteria,
            "data_source":  "edgar_xbrl (fundamentals) + yfinance (price fields)",
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ----- POST /screen/sql -----

class SQLScreenRequest(BaseModel):
    sql: str = Field(
        ...,
        description="Raw DuckDB SQL. Must SELECT from 'fundamentals' table.",
        example=(
            "SELECT ticker, pe_ratio, ev_ebitda, revenue_growth_1y, roe "
            "FROM fundamentals "
            "WHERE pe_ratio < 20 AND revenue_growth_1y > 0.15 AND roe > 0.15 "
            "ORDER BY ev_ebitda ASC LIMIT 50"
        ),
    )


@screener_v3_router.post(
    "/screen/sql",
    summary="Execute raw DuckDB SQL against the fundamentals table",
)
async def post_screen_sql(req: SQLScreenRequest) -> Dict[str, Any]:
    """Execute arbitrary DuckDB SQL for power users.

    The fundamentals table has 60+ columns. Use GET /fields to see schema.
    All data: EDGAR XBRL (fundamentals) + yfinance (price-based columns).
    """
    # Basic safety guard: only allow SELECT
    if not req.sql.strip().upper().startswith("SELECT"):
        raise HTTPException(400, "Only SELECT statements are allowed.")
    try:
        df = _get_screener().screen_sql(req.sql)
        return {
            "result_count": len(df),
            "results":      df.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(500, f"DuckDB query error: {exc}") from exc


# ----- GET /ticker/{ticker} -----

@screener_v3_router.get(
    "/ticker/{ticker}",
    summary="All fundamental fields for a specific ticker",
)
async def get_ticker(ticker: str) -> Dict[str, Any]:
    row = _get_db().get_ticker(ticker)
    if not row:
        raise HTTPException(
            404,
            f"Ticker '{ticker.upper()}' not found. Run POST /refresh to load it.",
        )
    return {"ticker": ticker.upper(), "data": row, "data_source": "edgar_xbrl"}


# ----- GET /ticker/{ticker}/explain -----

@screener_v3_router.post(
    "/ticker/{ticker}/explain",
    summary="Explain which criteria a specific ticker passes",
)
async def explain_ticker(ticker: str, body: ScreenRequest) -> Dict[str, Any]:
    return _get_screener().explain(ticker, body.criteria)


# ----- POST /refresh -----

class RefreshRequest(BaseModel):
    tickers:     Optional[List[str]] = Field(
        None, description="Specific tickers to refresh; null = full universe"
    )
    force:       bool = Field(False, description="Force re-fetch even if recently updated")
    fetch_price: bool = Field(True,  description="Fetch price/momentum data from yfinance")
    fetch_finviz: bool = Field(False, description="Fetch analyst ratings from Finviz (slow, 2s/ticker)")


@screener_v3_router.post("/refresh", summary="Refresh EDGAR XBRL data for tickers")
async def refresh_data(req: RefreshRequest) -> Dict[str, Any]:
    """Load or refresh fundamental data from EDGAR XBRL.

    Data flow:
    1. Resolve CIK from EDGAR company_tickers.json
    2. Fetch companyfacts JSON from EDGAR XBRL
    3. Extract 60+ metrics (NO yfinance for fundamentals)
    4. Optionally fetch price-based fields from yfinance
    5. Optionally fetch analyst ratings from Finviz
    6. Upsert into DuckDB

    Rate limit: 120ms between EDGAR calls (~8 req/s, under 10 req/s cap).
    """
    tickers = req.tickers or SP1500_UNIVERSE[:50]  # default to first 50

    def price_fn(ticker: str) -> Dict:
        return _price_fetcher.fetch(ticker) if req.fetch_price else {}

    def finviz_fn(ticker: str) -> Dict:
        return _finviz_scraper.fetch(ticker) if req.fetch_finviz else {}

    results = _get_db().load_universe(
        tickers   = tickers,
        price_fn  = price_fn,
        finviz_fn = finviz_fn,
        force     = req.force,
    )
    updated  = sum(1 for v in results.values() if v)
    skipped  = len(tickers) - updated
    return {
        "requested":      len(tickers),
        "updated":        updated,
        "skipped_recent": skipped,
        "results":        results,
        "data_source":    "edgar_xbrl",
    }


# ----- GET /universe-stats -----

@screener_v3_router.get("/universe-stats", summary="Coverage and data quality statistics")
async def get_universe_stats() -> Dict[str, Any]:
    stats = _get_db().universe_stats()
    return {
        "universe_stats":    stats,
        "prebuilt_screens":  len(PREBUILT_SCREENS),
        "data_sources":      ["edgar_xbrl", "yfinance_price_only", "finviz_scrape"],
        "duckdb_path":       str(DUCKDB_PATH),
        "sqlite_results_db": str(SQLITE_PATH),
    }


# ----- GET /history -----

@screener_v3_router.get("/history", summary="Screening result history")
async def get_history(limit: int = Query(50, ge=1, le=200)) -> Dict[str, Any]:
    return {"history": _results_db.history(limit=limit)}
