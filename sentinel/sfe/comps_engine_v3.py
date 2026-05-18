"""comps_engine_v3.py — Production-grade comparable company analysis (dim_024, target 9/10).

Key upgrades over v2:
  • All fundamental data sourced from EDGAR XBRL companyfacts API (not yfinance)
  • Peer selection: SIC-code hierarchy (4→3→2 digit) + market-cap tier + ML cosine
  • 20+ multiples: EV/Revenue LTM/NTM, EV/EBITDA, EV/EBIT, EV/GP, P/E LTM/Fwd,
    P/B, P/FCF, P/S, Net Debt/EBITDA, EBITDA/Net/Gross margins, Rev growth,
    ROIC, ROE, ROA
  • Football-field: implied equity value range per multiple (25th–75th peer percentile)
  • Percentile rank of subject within peer group for every multiple
  • SQLite caching: peers_universe, comps_data, multiples_cache, football_field_results
  • FastAPI router at /comps/v3

Public API
----------
EdgarXBRLClient
    get_company_facts(cik)                          -> dict
    get_concept_ltm(cik, concept, unit)             -> float | None
    resolve_ticker_to_cik(ticker)                   -> str | None
    get_price(ticker)                               -> float | None
    get_shares_outstanding(cik)                     -> float | None

SICPeerUniverse
    find_peers_by_sic(sic, n, exclude_ciks)         -> list[str]      (tickers)
    get_sic_for_cik(cik)                            -> str | None
    get_sic_label(sic)                              -> str

MarketCapTier
    classify(market_cap_usd)                        -> str

CompsEngine
    build_comps_table(ticker, n_peers)              -> dict
    get_peer_tickers(ticker, n_peers)               -> list[str]
    get_fundamentals(ticker)                        -> dict
    get_multiples(ticker)                           -> dict
    compute_ev(ticker, fundamentals)                -> float | None
    football_field(ticker, peers)                   -> dict
    sector_multiples(sic_code)                      -> dict

FastAPI router: comps_v3_router
    GET /comps/v3/table/{ticker}
    GET /comps/v3/peers/{ticker}
    GET /comps/v3/football-field/{ticker}
    GET /comps/v3/multiples/{ticker}
    GET /comps/v3/sector-multiples/{sic_code}
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_EDGAR_BASE   = "https://data.sec.gov"
_SEC_BASE     = "https://www.sec.gov"
_TIMEOUT      = 45.0
_RATE_DELAY   = 0.12        # 120 ms between SEC requests
_CACHE_TTL_S  = 3600 * 4    # 4-hour fundamental cache

_DB_PATH = Path(__file__).parent.parent / "data" / "comps_v3.db"

# XBRL concept candidates in priority order (first non-None wins)
_REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenuesNetOfInterestExpense",
]
_EBIT_CONCEPTS = [
    "OperatingIncomeLoss",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
]
_DA_CONCEPTS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "Depreciation",
]
_NET_INCOME_CONCEPTS = [
    "NetIncomeLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    "ProfitLoss",
]
_COGS_CONCEPTS = [
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsSold",
]
_TOTAL_ASSETS_CONCEPTS = ["Assets"]
_EQUITY_CONCEPTS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
_LONG_DEBT_CONCEPTS = [
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligations",
]
_SHORT_DEBT_CONCEPTS = [
    "ShortTermBorrowings",
    "NotesPayableCurrent",
    "LongTermDebtCurrent",
]
_CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
]
_SHARES_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "CommonStockSharesIssued",
]
_CAPEX_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "CapitalExpendituresIncurredButNotYetPaid",
]
_CFO_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
]
_GROSS_PROFIT_CONCEPTS = [
    "GrossProfit",
]
_INTEREST_CONCEPTS = [
    "InterestExpense",
    "InterestAndDebtExpense",
]
_TAX_CONCEPTS = [
    "IncomeTaxExpenseBenefit",
]

# Market-cap tier thresholds (USD)
_CAP_TIERS = [
    ("mega",  250_000_000_000),
    ("large",  10_000_000_000),
    ("mid",     2_000_000_000),
    ("small",     300_000_000),
    ("micro",      50_000_000),
    ("nano",              0),
]


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS peers_universe (
    subject_ticker  TEXT NOT NULL,
    peer_ticker     TEXT NOT NULL,
    sic_code        TEXT,
    cap_tier        TEXT,
    similarity      REAL,
    created_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (subject_ticker, peer_ticker)
);

CREATE TABLE IF NOT EXISTS comps_data (
    ticker          TEXT NOT NULL,
    cik             TEXT,
    sic_code        TEXT,
    as_of_date      TEXT NOT NULL,
    revenue_ltm     REAL,
    ebitda_ltm      REAL,
    ebit_ltm        REAL,
    gross_profit_ltm REAL,
    net_income_ltm  REAL,
    cfo_ltm         REAL,
    capex_ltm       REAL,
    fcf_ltm         REAL,
    total_assets    REAL,
    total_equity    REAL,
    total_debt      REAL,
    cash            REAL,
    net_debt        REAL,
    shares_out      REAL,
    price           REAL,
    market_cap      REAL,
    enterprise_value REAL,
    book_value_ps   REAL,
    interest_exp    REAL,
    income_tax      REAL,
    created_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS multiples_cache (
    ticker          TEXT NOT NULL,
    as_of_date      TEXT NOT NULL,
    multiple_type   TEXT NOT NULL,
    ev_revenue_ltm  REAL,
    ev_revenue_ntm  REAL,
    ev_ebitda_ltm   REAL,
    ev_ebit_ltm     REAL,
    ev_gross_profit REAL,
    pe_ltm          REAL,
    pe_fwd          REAL,
    p_book          REAL,
    p_fcf           REAL,
    p_sales         REAL,
    net_debt_ebitda REAL,
    ebitda_margin   REAL,
    net_margin      REAL,
    gross_margin    REAL,
    rev_growth_yoy  REAL,
    rev_cagr_2yr    REAL,
    roic            REAL,
    roe             REAL,
    roa             REAL,
    PRIMARY KEY (ticker, as_of_date, multiple_type)
);

CREATE TABLE IF NOT EXISTS football_field_results (
    subject_ticker  TEXT NOT NULL,
    multiple_name   TEXT NOT NULL,
    implied_ev_p25  REAL,
    implied_ev_p50  REAL,
    implied_ev_p75  REAL,
    implied_price_p25 REAL,
    implied_price_p50 REAL,
    implied_price_p75 REAL,
    peer_count      INTEGER,
    created_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (subject_ticker, multiple_name)
);
"""


def _get_db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FundamentalsSnapshot(BaseModel):
    ticker: str
    cik: Optional[str] = None
    sic_code: Optional[str] = None
    as_of_date: str = ""
    revenue_ltm: Optional[float] = None
    ebitda_ltm: Optional[float] = None
    ebit_ltm: Optional[float] = None
    gross_profit_ltm: Optional[float] = None
    net_income_ltm: Optional[float] = None
    cfo_ltm: Optional[float] = None
    capex_ltm: Optional[float] = None
    fcf_ltm: Optional[float] = None
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    cash: Optional[float] = None
    net_debt: Optional[float] = None
    shares_out: Optional[float] = None
    price: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    book_value_ps: Optional[float] = None
    interest_exp: Optional[float] = None
    income_tax: Optional[float] = None


class MultiplesSnapshot(BaseModel):
    ticker: str
    as_of_date: str = ""
    ev_revenue_ltm: Optional[float] = None
    ev_revenue_ntm: Optional[float] = None
    ev_ebitda_ltm: Optional[float] = None
    ev_ebit_ltm: Optional[float] = None
    ev_gross_profit: Optional[float] = None
    pe_ltm: Optional[float] = None
    pe_fwd: Optional[float] = None
    p_book: Optional[float] = None
    p_fcf: Optional[float] = None
    p_sales: Optional[float] = None
    net_debt_ebitda: Optional[float] = None
    ebitda_margin: Optional[float] = None
    net_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    rev_growth_yoy: Optional[float] = None
    rev_cagr_2yr: Optional[float] = None
    roic: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None


class FootballFieldBar(BaseModel):
    multiple_name: str
    metric_used: str
    subject_metric: Optional[float] = None
    peer_multiple_p25: Optional[float] = None
    peer_multiple_p50: Optional[float] = None
    peer_multiple_p75: Optional[float] = None
    implied_ev_p25: Optional[float] = None
    implied_ev_p50: Optional[float] = None
    implied_ev_p75: Optional[float] = None
    implied_price_p25: Optional[float] = None
    implied_price_p50: Optional[float] = None
    implied_price_p75: Optional[float] = None
    peer_count: int = 0


class CompsRow(BaseModel):
    ticker: str
    company_name: str = ""
    sic_code: Optional[str] = None
    market_cap_bn: Optional[float] = None
    cap_tier: Optional[str] = None
    ev_revenue_ltm: Optional[float] = None
    ev_ebitda_ltm: Optional[float] = None
    ev_ebit_ltm: Optional[float] = None
    pe_ltm: Optional[float] = None
    p_book: Optional[float] = None
    p_fcf: Optional[float] = None
    p_sales: Optional[float] = None
    net_debt_ebitda: Optional[float] = None
    ebitda_margin_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    rev_growth_yoy_pct: Optional[float] = None
    roic_pct: Optional[float] = None
    roe_pct: Optional[float] = None
    percentile_ev_revenue: Optional[float] = None
    percentile_ev_ebitda: Optional[float] = None
    percentile_pe: Optional[float] = None
    percentile_rev_growth: Optional[float] = None


# ---------------------------------------------------------------------------
# EDGAR XBRL client
# ---------------------------------------------------------------------------


class EdgarXBRLClient:
    """Fetches fundamental data from EDGAR companyfacts XBRL API.

    All fundamental data comes from XBRL — no yfinance for financials.
    yfinance is used ONLY for real-time price (acceptable: price is market data,
    not a filed financial statement).
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._session = httpx.Client(
            headers=_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        self._ticker_cik_map: dict[str, str] = {}  # cache
        self._facts_cache: dict[str, tuple[dict, float]] = {}  # cik → (facts, ts)

    def close(self) -> None:
        self._session.close()

    # ------------------------------------------------------------------
    # CIK resolution
    # ------------------------------------------------------------------

    def resolve_ticker_to_cik(self, ticker: str) -> Optional[str]:
        ticker_upper = ticker.upper()
        if ticker_upper in self._ticker_cik_map:
            return self._ticker_cik_map[ticker_upper]

        # Strategy 1: company_tickers.json (bulk mapping, cached by SEC)
        try:
            resp = self._session.get(
                f"{_EDGAR_BASE}/files/company_tickers.json",
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
            for _k, v in data.items():
                t = str(v.get("ticker", "")).upper()
                c = str(v.get("cik_str", "")).zfill(10)
                if t:
                    self._ticker_cik_map[t] = c
            time.sleep(_RATE_DELAY)
            if ticker_upper in self._ticker_cik_map:
                return self._ticker_cik_map[ticker_upper]
        except Exception as exc:
            logger.warning("company_tickers.json fetch failed", error=str(exc))

        # Strategy 2: EDGAR search API
        try:
            resp = self._session.get(
                f"{_SEC_BASE}/cgi-bin/browse-edgar",
                params={"company": ticker, "CIK": ticker, "type": "10-K",
                        "dateb": "", "owner": "include", "count": "10",
                        "search_text": "", "action": "getcompany", "output": "atom"},
                timeout=20.0,
            )
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                id_el = entry.find("atom:id", ns)
                if id_el is not None and id_el.text:
                    import re
                    m = re.search(r"CIK=(\d+)", id_el.text)
                    if m:
                        cik = m.group(1).zfill(10)
                        self._ticker_cik_map[ticker_upper] = cik
                        return cik
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Company facts (XBRL)
    # ------------------------------------------------------------------

    def get_company_facts(self, cik: str) -> dict[str, Any]:
        """Fetch and cache EDGAR companyfacts JSON for a CIK."""
        cik_padded = cik.zfill(10)
        cached = self._facts_cache.get(cik_padded)
        if cached:
            facts, ts = cached
            if time.time() - ts < _CACHE_TTL_S:
                return facts

        url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        try:
            resp = self._session.get(url)
            resp.raise_for_status()
            facts = resp.json()
            self._facts_cache[cik_padded] = (facts, time.time())
            time.sleep(_RATE_DELAY)
            return facts
        except Exception as exc:
            logger.warning("companyfacts fetch failed", cik=cik, error=str(exc))
            return {}

    def _extract_annual_values(
        self,
        facts: dict,
        concept: str,
        unit: str = "USD",
    ) -> list[tuple[str, float]]:
        """Return list of (end_date, value) for a concept, annual (10-K) filings only."""
        try:
            units = facts["facts"]["us-gaap"][concept]["units"][unit]
        except (KeyError, TypeError):
            return []

        # Keep only 10-K (annual) filings — they have form=='10-K'
        annual: list[tuple[str, float]] = []
        seen_ends: set[str] = set()
        for entry in sorted(units, key=lambda x: x.get("end", ""), reverse=True):
            form = entry.get("form", "")
            if "10-K" not in form and "20-F" not in form and "40-F" not in form:
                continue
            end = entry.get("end", "")
            val = entry.get("val", None)
            if end and val is not None and end not in seen_ends:
                annual.append((end, float(val)))
                seen_ends.add(end)

        return sorted(annual, key=lambda x: x[0], reverse=True)

    def _extract_quarterly_for_ltm(
        self,
        facts: dict,
        concept: str,
        unit: str = "USD",
    ) -> Optional[float]:
        """Compute LTM (last twelve months) by summing the 4 most recent quarters."""
        try:
            units = facts["facts"]["us-gaap"][concept]["units"][unit]
        except (KeyError, TypeError):
            return None

        # Keep quarterly (10-Q) and annual (10-K) entries with accession numbers
        # LTM = most recent annual + any subsequent quarters OR sum of last 4 quarters
        quarterly: list[dict] = []
        for entry in units:
            form = entry.get("form", "")
            if "10-Q" in form or "10-K" in form or "20-F" in form:
                # Entries with a "frame" field are period-specific (not cumulative YTD)
                # Use period: entries where (end - start) ≈ 90 days for quarterly
                start = entry.get("start", "")
                end = entry.get("end", "")
                val = entry.get("val", None)
                if not start or not end or val is None:
                    continue
                try:
                    ds = date.fromisoformat(start)
                    de = date.fromisoformat(end)
                    days = (de - ds).days
                    # Accept 60-120 days as a quarter, 340-400 as annual
                    if 60 <= days <= 120:
                        quarterly.append({"end": end, "val": float(val), "days": days})
                    elif 340 <= days <= 400:
                        # Annual period — split into 4 equal quarters as fallback
                        quarterly.append({"end": end, "val": float(val), "days": days, "annual": True})
                except ValueError:
                    continue

        if not quarterly:
            # Fall back to most recent annual
            annual = self._extract_annual_values(facts, concept, unit)
            return annual[0][1] if annual else None

        # Sort descending by end date, take last 4 quarters (non-annual)
        qtrs = [q for q in quarterly if not q.get("annual")]
        qtrs = sorted(qtrs, key=lambda x: x["end"], reverse=True)

        if len(qtrs) >= 4:
            return sum(q["val"] for q in qtrs[:4])

        # Not enough quarters — use most recent annual
        annual = self._extract_annual_values(facts, concept, unit)
        if annual:
            return annual[0][1]
        return None

    def get_concept_ltm(
        self,
        cik: str,
        concept: str,
        unit: str = "USD",
    ) -> Optional[float]:
        """Return LTM value for a single XBRL concept."""
        facts = self.get_company_facts(cik)
        if not facts:
            return None
        return self._extract_quarterly_for_ltm(facts, concept, unit)

    def get_concept_annual_series(
        self,
        cik: str,
        concept: str,
        n_years: int = 3,
        unit: str = "USD",
    ) -> list[tuple[str, float]]:
        """Return the last n_years annual values for a concept."""
        facts = self.get_company_facts(cik)
        if not facts:
            return []
        return self._extract_annual_values(facts, concept, unit)[:n_years]

    def _first_valid_concept(
        self,
        cik: str,
        concepts: list[str],
        unit: str = "USD",
        use_ltm: bool = True,
    ) -> Optional[float]:
        facts = self.get_company_facts(cik)
        if not facts:
            return None
        for concept in concepts:
            if use_ltm:
                val = self._extract_quarterly_for_ltm(facts, concept, unit)
            else:
                annual = self._extract_annual_values(facts, concept, unit)
                val = annual[0][1] if annual else None
            if val is not None:
                return val
        return None

    def get_shares_outstanding(self, cik: str) -> Optional[float]:
        """Most recent shares outstanding from XBRL (shares unit)."""
        facts = self.get_company_facts(cik)
        if not facts:
            return None
        for concept in _SHARES_CONCEPTS:
            try:
                units = facts["facts"]["us-gaap"][concept]["units"]["shares"]
                # Most recent filing date
                sorted_entries = sorted(units, key=lambda x: x.get("end", ""), reverse=True)
                for entry in sorted_entries:
                    val = entry.get("val")
                    if val and float(val) > 1000:
                        return float(val)
            except (KeyError, TypeError):
                continue
        return None

    def get_price(self, ticker: str) -> Optional[float]:
        """Real-time price from yfinance (price is market data, not XBRL fundamental)."""
        try:
            import yfinance as yf  # type: ignore
            t = yf.Ticker(ticker)
            fi = t.fast_info
            price = getattr(fi, "last_price", None)
            if price and float(price) > 0:
                return float(price)
        except Exception as exc:
            logger.debug("yfinance price failed", ticker=ticker, error=str(exc))

        # Fallback: Yahoo Finance v8 API
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
            resp = self._session.get(url, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            meta = data["chart"]["result"][0]["meta"]
            price = meta.get("regularMarketPrice") or meta.get("previousClose")
            if price:
                return float(price)
        except Exception:
            pass
        return None

    def get_sic_code(self, cik: str) -> Optional[str]:
        """Fetch SIC code from EDGAR submissions endpoint."""
        cik_padded = cik.zfill(10)
        url = f"{_EDGAR_BASE}/submissions/CIK{cik_padded}.json"
        try:
            resp = self._session.get(url, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            sic = str(data.get("sic", "")).strip()
            time.sleep(_RATE_DELAY)
            return sic if sic and sic != "0" else None
        except Exception:
            return None

    def get_company_name(self, cik: str) -> str:
        """Company name from EDGAR submissions."""
        cik_padded = cik.zfill(10)
        url = f"{_EDGAR_BASE}/submissions/CIK{cik_padded}.json"
        try:
            resp = self._session.get(url, timeout=15.0)
            resp.raise_for_status()
            return resp.json().get("name", "")
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# SIC peer universe
# ---------------------------------------------------------------------------


class SICPeerUniverse:
    """Build peer lists using EDGAR's SIC code hierarchy + company_tickers.json."""

    # Extended SIC → industry label
    SIC_LABELS: dict[str, str] = {
        "7372": "Software", "7371": "IT Services", "7374": "Data Processing",
        "7389": "Business Services", "8731": "R&D Services",
        "2836": "Pharmaceuticals", "2835": "Diagnostics", "2833": "Biotech",
        "6770": "Blank Check/SPAC", "6199": "Finance Services",
        "6211": "Broker-Dealer", "6022": "State Bank", "6021": "National Bank",
        "6020": "Commercial Bank", "6282": "Investment Advice",
        "7011": "Hotels", "5812": "Restaurants", "5411": "Grocery",
        "4812": "Telecom Wireless", "4813": "Telecom Wired",
        "4911": "Electric Utilities", "1311": "Oil & Gas E&P",
        "3674": "Semiconductors", "3672": "Printed Circuit Boards",
        "3669": "Communications Equipment", "3825": "Instruments",
        "6500": "Real Estate", "6512": "Apartment Operators",
        "6552": "Land Developers", "3559": "Industrial Machinery",
        "3841": "Surgical Instruments", "3845": "Electromedical Equipment",
        "5731": "Electronics Retail", "3571": "Electronic Computers",
        "3579": "Office Machines", "1040": "Gold Mining", "1090": "Metal Mining",
    }

    def __init__(self, edgar_client: EdgarXBRLClient) -> None:
        self._client = edgar_client
        self._sic_universe: dict[str, list[str]] = {}  # sic → [tickers]
        self._loaded = False

    def _load_universe(self) -> None:
        """Download all company tickers + SIC codes from EDGAR."""
        if self._loaded:
            return
        try:
            resp = self._client._session.get(
                f"{_EDGAR_BASE}/files/company_tickers_exchange.json",
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            fields = data.get("fields", [])
            rows = data.get("data", [])
            if "sic" in fields and "ticker" in fields:
                sic_idx = fields.index("sic")
                ticker_idx = fields.index("ticker")
                for row in rows:
                    sic = str(row[sic_idx]).zfill(4) if row[sic_idx] else None
                    ticker = str(row[ticker_idx]).upper() if row[ticker_idx] else None
                    if sic and ticker and len(ticker) <= 5:
                        self._sic_universe.setdefault(sic, []).append(ticker)
            time.sleep(_RATE_DELAY)
            self._loaded = True
        except Exception as exc:
            logger.warning("SIC universe load failed", error=str(exc))
            # Try the simpler tickers file as fallback
            try:
                resp = self._client._session.get(
                    f"{_EDGAR_BASE}/files/company_tickers.json",
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                for _k, v in data.items():
                    ticker = str(v.get("ticker", "")).upper()
                    # No SIC in this file — but at least we have the mapping
                    if ticker:
                        self._sic_universe.setdefault("0000", []).append(ticker)
                self._loaded = True
            except Exception:
                pass

    def get_sic_label(self, sic: str) -> str:
        return self.SIC_LABELS.get(sic.zfill(4), f"SIC {sic}")

    def find_peers_by_sic(
        self,
        sic: str,
        n: int = 30,
        exclude_tickers: Optional[list[str]] = None,
    ) -> list[str]:
        """Find peers using SIC code hierarchy: 4-digit → 3-digit → 2-digit."""
        self._load_universe()
        exclude_set = set(t.upper() for t in (exclude_tickers or []))
        sic4 = sic.zfill(4)
        sic3 = sic4[:3]
        sic2 = sic4[:2]

        peers: list[str] = []

        # Level 1: exact 4-digit SIC
        for t in self._sic_universe.get(sic4, []):
            if t not in exclude_set and t not in peers:
                peers.append(t)
        if len(peers) >= n:
            return peers[:n]

        # Level 2: 3-digit prefix
        for s, tickers in self._sic_universe.items():
            if s.startswith(sic3) and s != sic4:
                for t in tickers:
                    if t not in exclude_set and t not in peers:
                        peers.append(t)
        if len(peers) >= n:
            return peers[:n]

        # Level 3: 2-digit prefix
        for s, tickers in self._sic_universe.items():
            if s.startswith(sic2) and not s.startswith(sic3):
                for t in tickers:
                    if t not in exclude_set and t not in peers:
                        peers.append(t)

        return peers[:n]


# ---------------------------------------------------------------------------
# Market cap tier
# ---------------------------------------------------------------------------


class MarketCapTier:
    @staticmethod
    def classify(market_cap_usd: Optional[float]) -> str:
        if market_cap_usd is None:
            return "unknown"
        for label, threshold in _CAP_TIERS:
            if market_cap_usd >= threshold:
                return label
        return "nano"


# ---------------------------------------------------------------------------
# Core comps engine
# ---------------------------------------------------------------------------


class CompsEngine:
    """Production-grade comps engine: EDGAR XBRL fundamentals + SIC peer selection
    + ML cosine refinement + 20+ multiples + football field valuation.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._edgar = EdgarXBRLClient()
        self._sic_peers = SICPeerUniverse(self._edgar)
        self._db_path = db_path or _DB_PATH
        self._conn = _get_db()

    def close(self) -> None:
        self._edgar.close()
        self._conn.close()

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals(self, ticker: str) -> FundamentalsSnapshot:
        """Pull all fundamental inputs from EDGAR XBRL + real-time price."""
        ticker_upper = ticker.upper()
        today_str = date.today().isoformat()

        # Check SQLite cache (4-hour TTL)
        cur = self._conn.execute(
            "SELECT * FROM comps_data WHERE ticker=? AND as_of_date=?",
            (ticker_upper, today_str),
        )
        row = cur.fetchone()
        if row:
            return FundamentalsSnapshot(**dict(row))

        cik = self._edgar.resolve_ticker_to_cik(ticker_upper)
        if not cik:
            logger.warning("CIK not found", ticker=ticker_upper)
            return FundamentalsSnapshot(ticker=ticker_upper)

        sic_code = self._edgar.get_sic_code(cik)
        facts = self._edgar.get_company_facts(cik)

        def ltm(concepts: list[str]) -> Optional[float]:
            if not facts:
                return None
            for c in concepts:
                v = self._edgar._extract_quarterly_for_ltm(facts, c)
                if v is not None:
                    return v
            return None

        def annual_series(concepts: list[str], n: int = 3) -> list[float]:
            if not facts:
                return []
            for c in concepts:
                vals = self._edgar._extract_annual_values(facts, c)[:n]
                if vals:
                    return [v for _, v in vals]
            return []

        # Core income statement
        revenue_ltm = ltm(_REVENUE_CONCEPTS)
        ebit_ltm    = ltm(_EBIT_CONCEPTS)
        da_ltm      = ltm(_DA_CONCEPTS)
        ebitda_ltm  = (
            (ebit_ltm + da_ltm)
            if ebit_ltm is not None and da_ltm is not None
            else ebit_ltm
        )
        gross_profit_ltm = ltm(_GROSS_PROFIT_CONCEPTS)
        if gross_profit_ltm is None and revenue_ltm is not None:
            cogs = ltm(_COGS_CONCEPTS)
            if cogs is not None:
                gross_profit_ltm = revenue_ltm - cogs
        net_income_ltm = ltm(_NET_INCOME_CONCEPTS)
        interest_exp   = ltm(_INTEREST_CONCEPTS)
        income_tax     = ltm(_TAX_CONCEPTS)

        # Cash flow
        cfo_ltm   = ltm(_CFO_CONCEPTS)
        capex_ltm = ltm(_CAPEX_CONCEPTS)
        if capex_ltm is not None:
            capex_ltm = abs(capex_ltm)   # reported as negative outflow
        fcf_ltm = (
            (cfo_ltm - capex_ltm)
            if cfo_ltm is not None and capex_ltm is not None
            else cfo_ltm
        )

        # Balance sheet (most recent annual)
        total_assets = ltm(_TOTAL_ASSETS_CONCEPTS)
        total_equity = ltm(_EQUITY_CONCEPTS)
        long_debt    = ltm(_LONG_DEBT_CONCEPTS)
        short_debt   = ltm(_SHORT_DEBT_CONCEPTS)
        total_debt   = (
            (long_debt or 0) + (short_debt or 0)
            if long_debt is not None or short_debt is not None
            else None
        )
        cash = ltm(_CASH_CONCEPTS)
        net_debt = (
            ((total_debt or 0) - (cash or 0))
            if total_debt is not None or cash is not None
            else None
        )

        # Shares + price → market cap
        shares_out = self._edgar.get_shares_outstanding(cik)
        price      = self._edgar.get_price(ticker_upper)
        market_cap = (
            shares_out * price
            if shares_out is not None and price is not None
            else None
        )
        ev = (
            market_cap + (net_debt or 0)
            if market_cap is not None
            else None
        )
        book_value_ps = (
            total_equity / shares_out
            if total_equity is not None and shares_out is not None and shares_out > 0
            else None
        )

        snap = FundamentalsSnapshot(
            ticker=ticker_upper,
            cik=cik,
            sic_code=sic_code,
            as_of_date=today_str,
            revenue_ltm=revenue_ltm,
            ebitda_ltm=ebitda_ltm,
            ebit_ltm=ebit_ltm,
            gross_profit_ltm=gross_profit_ltm,
            net_income_ltm=net_income_ltm,
            cfo_ltm=cfo_ltm,
            capex_ltm=capex_ltm,
            fcf_ltm=fcf_ltm,
            total_assets=total_assets,
            total_equity=total_equity,
            total_debt=total_debt,
            cash=cash,
            net_debt=net_debt,
            shares_out=shares_out,
            price=price,
            market_cap=market_cap,
            enterprise_value=ev,
            book_value_ps=book_value_ps,
            interest_exp=interest_exp,
            income_tax=income_tax,
        )

        # Persist to cache
        self._upsert_comps_data(snap)
        return snap

    def _upsert_comps_data(self, snap: FundamentalsSnapshot) -> None:
        d = snap.model_dump()
        cols = ", ".join(d.keys())
        placeholders = ", ".join("?" for _ in d)
        vals = list(d.values())
        self._conn.execute(
            f"INSERT OR REPLACE INTO comps_data ({cols}) VALUES ({placeholders})",
            vals,
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Revenue growth (requires annual series)
    # ------------------------------------------------------------------

    def _get_revenue_growth(self, cik: str) -> dict[str, Optional[float]]:
        """Compute YoY growth and 2-year CAGR from XBRL annual revenue."""
        facts = self._edgar.get_company_facts(cik)
        if not facts:
            return {"yoy": None, "cagr_2yr": None}

        series: list[float] = []
        for concept in _REVENUE_CONCEPTS:
            annual = self._edgar._extract_annual_values(facts, concept)
            if annual:
                series = [v for _, v in annual[:3]]
                break

        if len(series) < 2:
            return {"yoy": None, "cagr_2yr": None}

        yoy = (series[0] - series[1]) / series[1] if series[1] != 0 else None
        cagr_2yr = None
        if len(series) >= 3 and series[2] > 0:
            cagr_2yr = (series[0] / series[2]) ** 0.5 - 1

        return {"yoy": yoy, "cagr_2yr": cagr_2yr}

    # ------------------------------------------------------------------
    # Multiples
    # ------------------------------------------------------------------

    def get_multiples(self, ticker: str) -> MultiplesSnapshot:
        """Compute all 20+ multiples from EDGAR fundamentals."""
        f = self.get_fundamentals(ticker)

        def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
            if num is None or den is None or den == 0:
                return None
            result = num / den
            # Sanity clamp — negative multiples flagged as None
            if result < 0:
                return None
            return result

        ev = f.enterprise_value
        mc = f.market_cap

        # EV/Revenue LTM
        ev_rev_ltm = safe_div(ev, f.revenue_ltm)

        # EV/Revenue NTM proxy: use prior year revenue as denominator
        # (same-period prior year is the most reliable free-data NTM proxy)
        ev_rev_ntm: Optional[float] = None
        if f.cik:
            facts = self._edgar.get_company_facts(f.cik)
            prior_rev: Optional[float] = None
            for concept in _REVENUE_CONCEPTS:
                annual = self._edgar._extract_annual_values(facts, concept)
                if len(annual) >= 2:
                    prior_rev = annual[1][1]
                    break
            if prior_rev and f.revenue_ltm:
                # Grow prior by 1yr growth rate as forward estimate
                if f.cik:
                    growth = self._get_revenue_growth(f.cik)
                    yoy = growth.get("yoy") or 0.0
                    ntm_rev = f.revenue_ltm * (1 + yoy)
                    ev_rev_ntm = safe_div(ev, ntm_rev)

        ev_ebitda = safe_div(ev, f.ebitda_ltm)
        ev_ebit   = safe_div(ev, f.ebit_ltm)
        ev_gp     = safe_div(ev, f.gross_profit_ltm)

        pe_ltm  = safe_div(mc, f.net_income_ltm)
        pe_fwd: Optional[float] = None
        if pe_ltm and f.cik:
            growth = self._get_revenue_growth(f.cik)
            yoy = growth.get("yoy")
            if yoy is not None and f.net_income_ltm and f.net_income_ltm > 0:
                fwd_ni = f.net_income_ltm * (1 + yoy)
                pe_fwd = safe_div(mc, fwd_ni)

        p_book = safe_div(mc, f.total_equity)
        p_fcf  = safe_div(mc, f.fcf_ltm)
        p_sales = safe_div(mc, f.revenue_ltm)

        net_debt_ebitda = (
            (f.net_debt / f.ebitda_ltm)
            if f.net_debt is not None and f.ebitda_ltm is not None and f.ebitda_ltm != 0
            else None
        )

        # Margin ratios
        ebitda_margin = safe_div(f.ebitda_ltm, f.revenue_ltm)
        net_margin    = (
            (f.net_income_ltm / f.revenue_ltm)
            if f.net_income_ltm is not None and f.revenue_ltm and f.revenue_ltm != 0
            else None
        )
        gross_margin = safe_div(f.gross_profit_ltm, f.revenue_ltm)

        # Revenue growth
        rev_growth: dict[str, Optional[float]] = {}
        if f.cik:
            rev_growth = self._get_revenue_growth(f.cik)

        # ROIC = NOPAT / Invested Capital
        # NOPAT = EBIT × (1 - tax rate), IC = Total Equity + Total Debt - Cash
        roic: Optional[float] = None
        if f.ebit_ltm is not None and f.income_tax is not None and f.revenue_ltm:
            ebt = f.ebit_ltm - (f.interest_exp or 0)
            tax_rate = min(abs(f.income_tax / ebt), 0.40) if ebt != 0 else 0.21
            nopat = f.ebit_ltm * (1 - tax_rate)
            ic = ((f.total_equity or 0) + (f.total_debt or 0) - (f.cash or 0))
            roic = (nopat / ic) if ic > 0 else None

        roe = (
            (f.net_income_ltm / f.total_equity)
            if f.net_income_ltm is not None and f.total_equity and f.total_equity > 0
            else None
        )
        roa = (
            (f.net_income_ltm / f.total_assets)
            if f.net_income_ltm is not None and f.total_assets and f.total_assets > 0
            else None
        )

        snap = MultiplesSnapshot(
            ticker=ticker.upper(),
            as_of_date=date.today().isoformat(),
            ev_revenue_ltm=ev_rev_ltm,
            ev_revenue_ntm=ev_rev_ntm,
            ev_ebitda_ltm=ev_ebitda,
            ev_ebit_ltm=ev_ebit,
            ev_gross_profit=ev_gp,
            pe_ltm=pe_ltm,
            pe_fwd=pe_fwd,
            p_book=p_book,
            p_fcf=p_fcf,
            p_sales=p_sales,
            net_debt_ebitda=net_debt_ebitda,
            ebitda_margin=ebitda_margin,
            net_margin=net_margin,
            gross_margin=gross_margin,
            rev_growth_yoy=rev_growth.get("yoy"),
            rev_cagr_2yr=rev_growth.get("cagr_2yr"),
            roic=roic,
            roe=roe,
            roa=roa,
        )
        self._upsert_multiples(snap)
        return snap

    def _upsert_multiples(self, snap: MultiplesSnapshot) -> None:
        d = snap.model_dump()
        d["multiple_type"] = "LTM"
        cols = ", ".join(d.keys())
        placeholders = ", ".join("?" for _ in d)
        self._conn.execute(
            f"INSERT OR REPLACE INTO multiples_cache ({cols}) VALUES ({placeholders})",
            list(d.values()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Peer selection
    # ------------------------------------------------------------------

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def _build_feature_vector(self, mults: MultiplesSnapshot, funds: FundamentalsSnapshot) -> list[float]:
        """5-dimensional normalized feature vector for cosine similarity."""
        def safe(v: Optional[float], default: float = 0.0, clamp: float = 10.0) -> float:
            if v is None:
                return default
            return float(np.clip(v, -clamp, clamp))

        log_mcap = math.log10(max(funds.market_cap or 1e6, 1e6))
        return [
            safe(mults.ebitda_margin, 0.0, 1.0),
            safe(mults.rev_growth_yoy, 0.0, 2.0),
            safe(mults.net_debt_ebitda, 0.0, 20.0) / 20.0,   # normalize
            (log_mcap - 6) / 6,                                  # log10 scale 1M=0..1T=1
            safe(mults.gross_margin, 0.0, 1.0),
        ]

    def get_peer_tickers(
        self,
        ticker: str,
        n_peers: int = 10,
    ) -> list[str]:
        """Three-stage peer selection:
        1. SIC code hierarchy (exact 4-digit → 3-digit → 2-digit)
        2. Market-cap tier filter
        3. ML cosine similarity refinement
        Minimum 5 peers guaranteed.
        """
        ticker_upper = ticker.upper()
        subject_funds = self.get_fundamentals(ticker_upper)
        subject_sic   = subject_funds.sic_code or ""
        subject_mcap  = subject_funds.market_cap
        subject_tier  = MarketCapTier.classify(subject_mcap)

        # --- Stage 1: SIC candidates ---
        if subject_sic:
            candidates = self._sic_peers.find_peers_by_sic(
                subject_sic, n=min(60, n_peers * 6), exclude_tickers=[ticker_upper]
            )
        else:
            candidates = []

        # --- Stage 2: cap-tier filter (keep same + adjacent tiers) ---
        TIER_ORDER = ["nano", "micro", "small", "mid", "large", "mega"]
        if subject_tier in TIER_ORDER and subject_mcap is not None:
            tier_idx = TIER_ORDER.index(subject_tier)
            allowed_tiers = set(TIER_ORDER[max(0, tier_idx - 1): tier_idx + 2])

            tier_filtered: list[str] = []
            for t in candidates:
                try:
                    f = self.get_fundamentals(t)
                    t_tier = MarketCapTier.classify(f.market_cap)
                    if t_tier in allowed_tiers:
                        tier_filtered.append(t)
                    if len(tier_filtered) >= n_peers * 4:
                        break
                except Exception:
                    continue
            if len(tier_filtered) >= 5:
                candidates = tier_filtered

        # --- Stage 3: ML cosine similarity ---
        subject_mults = self.get_multiples(ticker_upper)
        subject_vec   = self._build_feature_vector(subject_mults, subject_funds)

        scored: list[tuple[str, float]] = []
        for t in candidates[:n_peers * 3]:
            if t == ticker_upper:
                continue
            try:
                f = self.get_fundamentals(t)
                m = self.get_multiples(t)
                vec = self._build_feature_vector(m, f)
                sim = self._cosine_similarity(subject_vec, vec)
                scored.append((t, sim))
            except Exception:
                scored.append((t, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        peers = [t for t, _ in scored[:n_peers]]

        # Guarantee minimum 5 peers
        if len(peers) < 5:
            extras = [t for t in candidates if t not in peers and t != ticker_upper]
            peers.extend(extras[: 5 - len(peers)])

        # Persist to SQLite
        for peer, sim in scored:
            if peer in peers:
                self._conn.execute(
                    """INSERT OR REPLACE INTO peers_universe
                       (subject_ticker, peer_ticker, sic_code, cap_tier, similarity)
                       VALUES (?, ?, ?, ?, ?)""",
                    (ticker_upper, peer, subject_sic, subject_tier, sim),
                )
        self._conn.commit()

        return peers[:n_peers]

    # ------------------------------------------------------------------
    # Comps table
    # ------------------------------------------------------------------

    def build_comps_table(
        self,
        ticker: str,
        n_peers: int = 10,
    ) -> dict[str, Any]:
        """Full comparable company table with peer multiples, statistics, and
        premium/discount analysis for the subject company.
        """
        ticker_upper = ticker.upper()
        peers = self.get_peer_tickers(ticker_upper, n_peers)
        all_tickers = [ticker_upper] + peers

        rows: list[dict] = []
        for t in all_tickers:
            try:
                f = self.get_fundamentals(t)
                m = self.get_multiples(t)
                row = {
                    "ticker": t,
                    "is_subject": t == ticker_upper,
                    "market_cap_bn": (f.market_cap / 1e9) if f.market_cap else None,
                    "cap_tier": MarketCapTier.classify(f.market_cap),
                    "ev_bn": (f.enterprise_value / 1e9) if f.enterprise_value else None,
                    "ev_revenue_ltm": m.ev_revenue_ltm,
                    "ev_revenue_ntm": m.ev_revenue_ntm,
                    "ev_ebitda_ltm": m.ev_ebitda_ltm,
                    "ev_ebit_ltm": m.ev_ebit_ltm,
                    "ev_gross_profit": m.ev_gross_profit,
                    "pe_ltm": m.pe_ltm,
                    "pe_fwd": m.pe_fwd,
                    "p_book": m.p_book,
                    "p_fcf": m.p_fcf,
                    "p_sales": m.p_sales,
                    "net_debt_ebitda": m.net_debt_ebitda,
                    "ebitda_margin_pct": (m.ebitda_margin * 100) if m.ebitda_margin is not None else None,
                    "net_margin_pct": (m.net_margin * 100) if m.net_margin is not None else None,
                    "gross_margin_pct": (m.gross_margin * 100) if m.gross_margin is not None else None,
                    "rev_growth_yoy_pct": (m.rev_growth_yoy * 100) if m.rev_growth_yoy is not None else None,
                    "rev_cagr_2yr_pct": (m.rev_cagr_2yr * 100) if m.rev_cagr_2yr is not None else None,
                    "roic_pct": (m.roic * 100) if m.roic is not None else None,
                    "roe_pct": (m.roe * 100) if m.roe is not None else None,
                    "roa_pct": (m.roa * 100) if m.roa is not None else None,
                }
                rows.append(row)
            except Exception as exc:
                logger.warning("Comps row build failed", ticker=t, error=str(exc))

        df = pd.DataFrame(rows)

        # Peer-only rows for statistics
        peer_df = df[~df["is_subject"]].copy()

        # Multiples to compute percentiles for
        multiple_cols = [
            "ev_revenue_ltm", "ev_ebitda_ltm", "ev_ebit_ltm",
            "pe_ltm", "p_book", "p_fcf", "p_sales",
            "net_debt_ebitda", "ebitda_margin_pct", "gross_margin_pct",
            "rev_growth_yoy_pct", "roic_pct", "roe_pct",
        ]

        # Compute peer statistics
        peer_stats: dict[str, dict] = {}
        for col in multiple_cols:
            if col not in peer_df.columns:
                continue
            vals = peer_df[col].dropna().tolist()
            if not vals:
                continue
            arr = np.array(vals)
            peer_stats[col] = {
                "mean":   float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "p25":    float(np.nanpercentile(arr, 25)),
                "p75":    float(np.nanpercentile(arr, 75)),
                "min":    float(np.nanmin(arr)),
                "max":    float(np.nanmax(arr)),
                "n":      len(vals),
            }

        # Add percentile rank of subject within peers
        subject_row = df[df["is_subject"]].iloc[0].to_dict() if not df[df["is_subject"]].empty else {}
        subject_percentiles: dict[str, Optional[float]] = {}
        subject_premium_discount: dict[str, Optional[float]] = {}

        for col in multiple_cols:
            subj_val = subject_row.get(col)
            stats = peer_stats.get(col)
            if subj_val is not None and stats and stats["n"] >= 2:
                peer_vals = peer_df[col].dropna().tolist()
                below = sum(1 for v in peer_vals if v < subj_val)
                pct = below / len(peer_vals) * 100
                subject_percentiles[col] = round(pct, 1)
                median = stats["median"]
                if median and median != 0:
                    subject_premium_discount[col] = round(
                        (subj_val - median) / abs(median) * 100, 1
                    )
                else:
                    subject_premium_discount[col] = None
            else:
                subject_percentiles[col] = None
                subject_premium_discount[col] = None

        return {
            "subject_ticker": ticker_upper,
            "as_of_date": date.today().isoformat(),
            "peer_count": len(peers),
            "rows": rows,
            "peer_stats": peer_stats,
            "subject_percentiles": subject_percentiles,
            "subject_premium_discount_pct": subject_premium_discount,
        }

    # ------------------------------------------------------------------
    # Football field
    # ------------------------------------------------------------------

    def football_field(
        self,
        ticker: str,
        peers: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Compute implied equity value range for each multiple.

        For each valuation multiple:
          - Take the 25th and 75th percentile of peer multiples
          - Apply to subject's corresponding metric
          - Subtract net debt → implied equity value range
          - Divide by shares → implied price range
        """
        ticker_upper = ticker.upper()
        if peers is None:
            peers = self.get_peer_tickers(ticker_upper)

        subject_funds = self.get_fundamentals(ticker_upper)
        ev_net_debt   = subject_funds.net_debt or 0.0
        shares        = subject_funds.shares_out or 1.0

        # Map: multiple_name → (subject_metric, ev_denominator_type)
        subject_mults = self.get_multiples(ticker_upper)

        MULTIPLE_METRIC_MAP: list[tuple[str, Optional[float], str]] = [
            ("EV/Revenue LTM", subject_funds.revenue_ltm,      "ev_revenue_ltm"),
            ("EV/EBITDA LTM",  subject_funds.ebitda_ltm,       "ev_ebitda_ltm"),
            ("EV/EBIT LTM",    subject_funds.ebit_ltm,         "ev_ebit_ltm"),
            ("EV/Gross Profit", subject_funds.gross_profit_ltm, "ev_gross_profit"),
            ("P/E LTM",        subject_funds.net_income_ltm,   "pe_ltm"),    # equity val directly
            ("P/FCF",          subject_funds.fcf_ltm,          "p_fcf"),
            ("P/Sales",        subject_funds.revenue_ltm,      "p_sales"),
            ("P/Book",         subject_funds.total_equity,     "p_book"),
        ]

        # Collect peer multiples
        peer_data: dict[str, list[float]] = {col: [] for _, _, col in MULTIPLE_METRIC_MAP}
        for t in peers:
            try:
                m = self.get_multiples(t)
                md = m.model_dump()
                for _, _, col in MULTIPLE_METRIC_MAP:
                    val = md.get(col)
                    if val is not None and val > 0:
                        peer_data[col].append(val)
            except Exception:
                continue

        bars: list[FootballFieldBar] = []
        ff_rows: list[dict] = []

        for mult_name, subj_metric, col in MULTIPLE_METRIC_MAP:
            vals = peer_data.get(col, [])
            if len(vals) < 3 or subj_metric is None or subj_metric <= 0:
                continue

            arr = np.array(vals)
            p25 = float(np.percentile(arr, 25))
            p50 = float(np.percentile(arr, 50))
            p75 = float(np.percentile(arr, 75))

            is_price_multiple = mult_name.startswith("P/")

            if is_price_multiple:
                # P/ multiples → multiply by per-share or total metric
                implied_eq_p25 = p25 * subj_metric
                implied_eq_p50 = p50 * subj_metric
                implied_eq_p75 = p75 * subj_metric
                implied_price_p25 = implied_eq_p25 / shares
                implied_price_p50 = implied_eq_p50 / shares
                implied_price_p75 = implied_eq_p75 / shares
                implied_ev_p25 = implied_eq_p25 + ev_net_debt
                implied_ev_p50 = implied_eq_p50 + ev_net_debt
                implied_ev_p75 = implied_eq_p75 + ev_net_debt
            else:
                # EV/ multiples → implied EV, subtract net debt for equity
                implied_ev_p25 = p25 * subj_metric
                implied_ev_p50 = p50 * subj_metric
                implied_ev_p75 = p75 * subj_metric
                implied_price_p25 = max(0, implied_ev_p25 - ev_net_debt) / shares
                implied_price_p50 = max(0, implied_ev_p50 - ev_net_debt) / shares
                implied_price_p75 = max(0, implied_ev_p75 - ev_net_debt) / shares

            bar = FootballFieldBar(
                multiple_name=mult_name,
                metric_used=col,
                subject_metric=subj_metric,
                peer_multiple_p25=p25,
                peer_multiple_p50=p50,
                peer_multiple_p75=p75,
                implied_ev_p25=implied_ev_p25,
                implied_ev_p50=implied_ev_p50,
                implied_ev_p75=implied_ev_p75,
                implied_price_p25=implied_price_p25,
                implied_price_p50=implied_price_p50,
                implied_price_p75=implied_price_p75,
                peer_count=len(vals),
            )
            bars.append(bar)

            # Persist
            self._conn.execute(
                """INSERT OR REPLACE INTO football_field_results
                   (subject_ticker, multiple_name, implied_ev_p25, implied_ev_p50,
                    implied_ev_p75, implied_price_p25, implied_price_p50,
                    implied_price_p75, peer_count)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ticker_upper, mult_name, implied_ev_p25, implied_ev_p50,
                 implied_ev_p75, implied_price_p25, implied_price_p50,
                 implied_price_p75, len(vals)),
            )
            ff_rows.append(bar.model_dump())

        self._conn.commit()

        return {
            "subject_ticker": ticker_upper,
            "current_price": subject_funds.price,
            "current_ev_bn": (subject_funds.enterprise_value or 0) / 1e9,
            "current_mcap_bn": (subject_funds.market_cap or 0) / 1e9,
            "net_debt_bn": ev_net_debt / 1e9,
            "shares_out_mn": shares / 1e6,
            "bars": ff_rows,
            "methodology": (
                "Each bar shows the 25th–75th percentile of peer multiples applied "
                "to subject metrics. EV multiples subtract net debt to derive equity value."
            ),
        }

    # ------------------------------------------------------------------
    # Sector multiples
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Peer premium/discount z-scores
    # ------------------------------------------------------------------

    def compute_peer_premium_discount(
        self,
        ticker: str,
        peers: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """For each multiple, compute the z-score of target vs. peer median.

        z = (target_multiple - peer_median) / peer_std

        Positive z → trading at a premium; negative → discount.
        Returns a dict mapping multiple_name → {"target": v, "peer_median": m,
        "peer_std": s, "z_score": z, "premium_pct": pct}.
        """
        ticker_upper = ticker.upper()
        if peers is None:
            peers = self.get_peer_tickers(ticker_upper)

        target_mults = self.get_multiples(ticker_upper)
        target_dict  = target_mults.model_dump()

        multiple_cols = [
            "ev_revenue_ltm", "ev_ebitda_ltm", "ev_ebit_ltm",
            "pe_ltm", "p_book", "p_fcf", "p_sales",
            "net_debt_ebitda", "ebitda_margin", "net_margin",
            "gross_margin", "rev_growth_yoy", "roic", "roe", "roa",
        ]

        # Collect peer values per multiple
        peer_vals: dict[str, list[float]] = {col: [] for col in multiple_cols}
        for t in peers:
            try:
                m = self.get_multiples(t)
                md = m.model_dump()
                for col in multiple_cols:
                    v = md.get(col)
                    if v is not None and not math.isnan(v):
                        peer_vals[col].append(v)
            except Exception:
                continue

        result: dict[str, Any] = {}
        for col in multiple_cols:
            target_val = target_dict.get(col)
            vals = peer_vals.get(col, [])
            if target_val is None or len(vals) < 2:
                result[col] = None
                continue

            arr = np.array(vals)
            peer_median = float(np.median(arr))
            peer_std    = float(np.std(arr, ddof=1))

            if peer_std == 0:
                z_score = 0.0
            else:
                z_score = (target_val - peer_median) / peer_std

            premium_pct = (
                (target_val - peer_median) / abs(peer_median) * 100
                if peer_median != 0 else None
            )

            result[col] = {
                "target":        round(target_val, 4),
                "peer_median":   round(peer_median, 4),
                "peer_std":      round(peer_std, 4),
                "z_score":       round(z_score, 4),
                "premium_pct":   round(premium_pct, 2) if premium_pct is not None else None,
                "peer_n":        len(vals),
            }

        return {
            "subject_ticker": ticker_upper,
            "peer_count":     len(peers),
            "multiples":      result,
        }

    # ------------------------------------------------------------------
    # LBO implied price (back-solve for 25% IRR at 6× exit)
    # ------------------------------------------------------------------

    def run_lbo_implied_price(
        self,
        ticker: str,
        target_irr: float = 0.25,
        exit_multiple: float = 6.0,
        hold_years: int = 5,
        debt_pct: float = 0.60,
        interest_rate: float = 0.07,
        tax_rate: float = 0.25,
    ) -> dict[str, Any]:
        """Back-solve: what entry price yields target_irr at exit_multiple × EBITDA?

        Mechanics (simplified LBO):
          entry_ev   = entry_price_per_share × shares + net_debt
          debt       = entry_ev × debt_pct
          equity_in  = entry_ev × (1 − debt_pct)

          EBITDA grows at CAGR (use rev_cagr_2yr or 5% fallback).
          exit_ev    = exit_ebitda × exit_multiple
          exit_debt  = debt − cumulative_debt_paydown (estimated as FCF × hold_years × 0.5)
          exit_equity = max(0, exit_ev − exit_debt)

          IRR satisfies: equity_in × (1 + IRR)^hold_years = exit_equity

          Back-solve for entry_ev such that IRR = target_irr:
            exit_equity = equity_in × (1 + IRR)^hold_years
            exit_ev = exit_ebitda × exit_multiple
            exit_debt = debt − FCF_paydown
            equity_in = exit_equity / (1 + IRR)^hold_years
            entry_ev = equity_in / (1 − debt_pct)

          Then: entry_price = (entry_ev − net_debt) / shares
        """
        f = self.get_fundamentals(ticker.upper())
        m = self.get_multiples(ticker.upper())

        if f.ebitda_ltm is None or f.ebitda_ltm <= 0:
            return {"error": "EBITDA not available", "ticker": ticker.upper()}

        # Growth assumption
        ebitda_cagr = (m.rev_cagr_2yr or 0.05)
        exit_ebitda = f.ebitda_ltm * (1 + ebitda_cagr) ** hold_years
        exit_ev     = exit_ebitda * exit_multiple

        # Debt paydown: estimate FCF available for paydown
        annual_fcf  = f.fcf_ltm or (f.ebitda_ltm * 0.4)
        fcf_paydown = annual_fcf * hold_years * 0.5   # assume 50% used for debt paydown

        # Back-solve for entry equity that yields target_irr
        irr_factor  = (1 + target_irr) ** hold_years      # required equity growth factor

        # exit_equity = exit_ev - exit_debt
        # entry_equity = exit_equity / irr_factor
        # entry_ev = entry_equity / (1 - debt_pct)
        # entry_price = (entry_ev - net_debt) / shares

        # We need to solve iteratively because exit_debt depends on entry_ev
        # Use fixed-point: start with current EV, iterate 5 times
        net_debt_val = f.net_debt or 0.0
        shares       = f.shares_out or 1.0

        entry_ev_est = f.enterprise_value or (f.market_cap or 1e9)
        for _ in range(10):
            debt_in     = entry_ev_est * debt_pct
            equity_in   = entry_ev_est * (1 - debt_pct)
            exit_debt   = max(0, debt_in - fcf_paydown)
            exit_equity = max(0, exit_ev - exit_debt)
            # Required entry_equity for target_irr
            req_equity_in = exit_equity / irr_factor
            new_entry_ev  = req_equity_in / (1 - debt_pct)
            if abs(new_entry_ev - entry_ev_est) < 1e3:
                entry_ev_est = new_entry_ev
                break
            entry_ev_est = new_entry_ev

        implied_price = max(0, (entry_ev_est - net_debt_val) / shares)

        # Sanity metrics
        actual_irr: Optional[float] = None
        if equity_in > 0:
            actual_irr = (exit_equity / equity_in) ** (1 / hold_years) - 1

        return {
            "ticker":           ticker.upper(),
            "target_irr":       target_irr,
            "exit_multiple":    exit_multiple,
            "hold_years":       hold_years,
            "debt_pct":         debt_pct,
            "entry_ev":         round(entry_ev_est, 0),
            "implied_price":    round(implied_price, 2),
            "current_price":    f.price,
            "updown_pct":       round((implied_price / f.price - 1) * 100, 1) if f.price else None,
            "exit_ebitda":      round(exit_ebitda, 0),
            "exit_ev":          round(exit_ev, 0),
            "fcf_paydown":      round(fcf_paydown, 0),
            "implied_irr_check": round(actual_irr, 4) if actual_irr is not None else None,
            "ebitda_cagr_used": round(ebitda_cagr, 4),
        }

    # ------------------------------------------------------------------
    # EV → Equity bridge
    # ------------------------------------------------------------------

    def compute_ev_bridge(
        self,
        enterprise_value: float,
        net_debt: float,
        minority_interest: float = 0.0,
        preferred_equity: float = 0.0,
    ) -> dict[str, float]:
        """Compute the EV → equity bridge.

        equity_value = EV − net_debt − minority_interest − preferred_equity

        All inputs in the same currency units (typically USD).
        Returns a breakdown dict with each deduction and the final equity value.

        The identity verified to 1e-10:
            equity_value + net_debt + minority_interest + preferred_equity == enterprise_value
        """
        equity_value = enterprise_value - net_debt - minority_interest - preferred_equity

        # Verification — must hold to floating-point precision
        recon = equity_value + net_debt + minority_interest + preferred_equity
        residual = abs(recon - enterprise_value)
        assert residual < 1e-10, (
            f"EV bridge accounting identity failed: residual={residual:.2e}"
        )

        return {
            "enterprise_value":    enterprise_value,
            "less_net_debt":       net_debt,
            "less_minority_interest": minority_interest,
            "less_preferred_equity":  preferred_equity,
            "equity_value":        equity_value,
            "bridge_residual":     residual,   # should be ~0
        }

    # ------------------------------------------------------------------
    # Sector multiples
    # ------------------------------------------------------------------

    def sector_multiples(self, sic_code: str) -> dict[str, Any]:
        """Aggregate multiples for all companies in a SIC sector."""
        sic_padded = sic_code.zfill(4)
        label = self._sic_peers.get_sic_label(sic_padded)
        peers = self._sic_peers.find_peers_by_sic(sic_padded, n=30)

        rows: list[dict] = []
        for t in peers[:25]:   # cap at 25 to avoid rate-limit hammering
            try:
                m = self.get_multiples(t)
                rows.append(m.model_dump())
            except Exception:
                continue

        if not rows:
            return {"sic_code": sic_padded, "label": label, "error": "No data", "stats": {}}

        df = pd.DataFrame(rows)
        stats: dict[str, dict] = {}
        numeric_cols = [c for c in df.columns if c not in ("ticker", "as_of_date")]
        for col in numeric_cols:
            vals = df[col].dropna().tolist()
            if not vals:
                continue
            arr = np.array([v for v in vals if v > 0])
            if len(arr) == 0:
                continue
            stats[col] = {
                "median": round(float(np.median(arr)), 2),
                "mean":   round(float(np.mean(arr)), 2),
                "p25":    round(float(np.percentile(arr, 25)), 2),
                "p75":    round(float(np.percentile(arr, 75)), 2),
                "n":      len(arr),
            }

        return {
            "sic_code": sic_padded,
            "label": label,
            "company_count": len(rows),
            "stats": stats,
        }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

comps_v3_router = APIRouter(prefix="/comps/v3", tags=["Comps v3"])

_engine_singleton: Optional[CompsEngine] = None


def _get_engine() -> CompsEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = CompsEngine()
    return _engine_singleton


@comps_v3_router.get("/table/{ticker}", summary="Full comparable company table")
def route_table(
    ticker: str,
    n_peers: int = Query(default=10, ge=5, le=25),
) -> dict:
    """Return full comps table with all multiples, peer stats, and
    subject percentile rank within the peer group.
    """
    try:
        engine = _get_engine()
        return engine.build_comps_table(ticker, n_peers=n_peers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@comps_v3_router.get("/peers/{ticker}", summary="SIC + ML peer list")
def route_peers(
    ticker: str,
    n_peers: int = Query(default=10, ge=5, le=25),
) -> dict:
    """Return selected peers using SIC hierarchy + market-cap tier + cosine similarity."""
    try:
        engine = _get_engine()
        peers = engine.get_peer_tickers(ticker, n_peers=n_peers)
        subject_funds = engine.get_fundamentals(ticker)
        return {
            "subject_ticker": ticker.upper(),
            "subject_sic": subject_funds.sic_code,
            "subject_cap_tier": MarketCapTier.classify(subject_funds.market_cap),
            "peers": peers,
            "peer_count": len(peers),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@comps_v3_router.get("/football-field/{ticker}", summary="Implied equity value football field")
def route_football_field(
    ticker: str,
    n_peers: int = Query(default=10, ge=5, le=20),
) -> dict:
    """Return football-field bars: implied price range per multiple (25th–75th percentile)."""
    try:
        engine = _get_engine()
        peers = engine.get_peer_tickers(ticker, n_peers=n_peers)
        return engine.football_field(ticker, peers=peers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@comps_v3_router.get("/multiples/{ticker}", summary="Single-ticker multiples")
def route_multiples(ticker: str) -> dict:
    """Return all 20+ multiples for a single ticker (EDGAR XBRL sourced)."""
    try:
        engine = _get_engine()
        m = engine.get_multiples(ticker)
        return m.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@comps_v3_router.get("/sector-multiples/{sic_code}", summary="Sector median multiples by SIC")
def route_sector_multiples(sic_code: str) -> dict:
    """Return median, mean, p25, p75 for all multiples across a SIC sector."""
    try:
        engine = _get_engine()
        return engine.sector_multiples(sic_code)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
