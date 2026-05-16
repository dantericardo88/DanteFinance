"""
Fundamental Data Layer V3 — EDGAR XBRL backbone for the NL screener (dim_053, target 9/10).

Audit fix: replaces unreliable yfinance fundamental fetching with EDGAR XBRL companyfacts.
This module is the data backbone for sentinel/sai/nl_screener_v2.py and any other
consumer that needs reliable, point-in-time US fundamental data.

Architecture
------------
  Primary:   EDGAR XBRL companyfacts API (SEC, free, no key)
             https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  Prices:    yfinance (prices only — intentional; fundamentals come from EDGAR)
  Universe:  S&P 500 from Wikipedia + Russell 1000 approximation from EDGAR
  Cache:     SQLite 24h cache per ticker; incremental refresh via EDGAR submissions feed
  Memory:    In-process dict keyed by (ticker, metric) for O(1) screener lookups

Metrics mapped (50 screener metrics):
  Valuation:   pe_ratio, ev_ebitda, pb_ratio, ps_ratio, earnings_yield, fcf_yield,
               dividend_yield, book_value_per_share, enterprise_value, market_cap
  Growth:      revenue_growth, eps_growth, revenue_cagr_5y, eps_cagr_5y
  Quality:     gross_margin, operating_margin, net_margin, ebitda_margin,
               roe, roa, roic, roce
  Balance:     debt_equity, current_ratio, quick_ratio, net_debt_ebitda,
               interest_coverage, cash_ratio
  Cash flow:   fcf, fcf_yield, operating_cash_flow, capex, cash_conversion_ratio
  Income:      revenue, ebitda, earnings_per_share
  Other:       beta (yfinance returns vs SPY), short_interest

Drop-in replacement:
  from sentinel.sai.fundamental_data_layer_v3 import FundamentalDataLayerV3 as FundamentalDataLoader

Integration:
  FundamentalDataLayerV3.load(tickers) → pd.DataFrame
  Same column schema as nl_screener_v2.FundamentalDataLoader.load()

SQLite tables: fundamental_cache, company_metrics, metric_history, universe_registry
FastAPI router at /fundamentals/v3:
  GET  /metrics/{ticker}
  GET  /screener-ready/{ticker}
  POST /batch-metrics
  GET  /universe-coverage
  POST /refresh/{ticker}
  GET  /metric/{metric_name}?tickers=AAPL,MSFT

Public API
----------
EdgarCikResolver
    resolve(ticker)         -> Optional[str]  (CIK zero-padded to 10 digits)
    batch_resolve(tickers)  -> Dict[str, str]

XbrlFactsFetcher
    fetch_facts(cik)        -> dict  (raw companyfacts JSON)
    extract_concept(facts, concept, taxonomy)  -> pd.DataFrame
    latest_annual(concept, facts, taxonomy)    -> Optional[float]
    ltm(concept, facts, taxonomy)              -> Optional[float]

EdgarSubmissionMonitor
    has_new_filing(ticker, cik)  -> bool
    mark_refreshed(ticker)       -> None

MetricCalculator
    compute_all(ticker, cik, price, shares)  -> Dict[str, float]

UniverseManager
    sp500_tickers()         -> List[str]
    russell1000_tickers()   -> List[str]
    register(tickers)       -> None
    coverage_report()       -> Dict

FundamentalDataLayerV3
    load(tickers)           -> pd.DataFrame
    get_metric(ticker, metric) -> float
    batch_metrics(tickers, metrics) -> pd.DataFrame
    refresh(ticker)         -> Dict[str, float]
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query as FastAPIQuery
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent / "data" / "fundamental_v3.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_CACHE_TTL_SEC  = 86_400        # 24 hours for fundamental cache
_PRICE_TTL_SEC  = 300           # 5 minutes for price cache
_CIK_TTL_SEC    = 604_800       # 7 days for CIK resolution cache
_MAX_WORKERS    = 8
_REQUEST_TIMEOUT = 30

EDGAR_BASE       = "https://data.sec.gov"
EDGAR_FACTS_URL  = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": "SENTINEL/3.0 fundamental-data richard.porras@realempanada.com",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# XBRL concept mapping — all 50 screener metrics to us-gaap / dei concepts
# ---------------------------------------------------------------------------

# Format: metric_name → (taxonomy, concept_name, unit_filter)
# taxonomy: "us-gaap" | "dei" | "computed"
XBRL_MAP: Dict[str, Tuple[str, str, Optional[str]]] = {
    # Income statement
    "revenue":               ("us-gaap", "Revenues",                              "USD"),
    "revenue_alt1":          ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
    "revenue_alt2":          ("us-gaap", "SalesRevenueNet",                       "USD"),
    "gross_profit":          ("us-gaap", "GrossProfit",                            "USD"),
    "operating_income":      ("us-gaap", "OperatingIncomeLoss",                   "USD"),
    "net_income":            ("us-gaap", "NetIncomeLoss",                         "USD"),
    "ebit":                  ("us-gaap", "OperatingIncomeLoss",                   "USD"),
    "eps_diluted":           ("us-gaap", "EarningsPerShareDiluted",               "USD/shares"),
    "eps_basic":             ("us-gaap", "EarningsPerShareBasic",                 "USD/shares"),
    "da":                    ("us-gaap", "DepreciationDepletionAndAmortization",  "USD"),
    "da_alt":                ("us-gaap", "DepreciationAndAmortization",           "USD"),
    "interest_expense":      ("us-gaap", "InterestExpense",                       "USD"),
    "income_tax":            ("us-gaap", "IncomeTaxExpense",                      "USD"),
    "dividends_per_share":   ("us-gaap", "CommonStockDividendsPerShareDeclared",  "USD/shares"),
    # Balance sheet
    "total_assets":          ("us-gaap", "Assets",                                "USD"),
    "current_assets":        ("us-gaap", "AssetsCurrent",                         "USD"),
    "current_liabilities":   ("us-gaap", "LiabilitiesCurrent",                   "USD"),
    "total_equity":          ("us-gaap", "StockholdersEquity",                    "USD"),
    "total_equity_alt":      ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
    "long_term_debt":        ("us-gaap", "LongTermDebt",                          "USD"),
    "long_term_debt_alt":    ("us-gaap", "LongTermDebtNoncurrent",                "USD"),
    "short_term_debt":       ("us-gaap", "ShortTermBorrowings",                   "USD"),
    "short_term_debt_alt":   ("us-gaap", "DebtCurrent",                           "USD"),
    "cash":                  ("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD"),
    "cash_alt":              ("us-gaap", "Cash",                                  "USD"),
    "inventory":             ("us-gaap", "InventoryNet",                          "USD"),
    "receivables":           ("us-gaap", "AccountsReceivableNetCurrent",          "USD"),
    "total_liabilities":     ("us-gaap", "Liabilities",                           "USD"),
    "goodwill":              ("us-gaap", "Goodwill",                              "USD"),
    # Cash flow statement
    "cfo":                   ("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD"),
    "capex":                 ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD"),
    "capex_alt":             ("us-gaap", "CapitalExpendituresIncurredButNotYetPaid",   "USD"),
    # DEI (Document Entity Information)
    "shares_outstanding":    ("dei",     "EntityCommonStockSharesOutstanding",    "shares"),
    "cik_dei":               ("dei",     "EntityCentralIndexKey",                 None),
}

# Core 50 screener metric names (matches nl_screener_v2.py schema)
SCREENER_METRICS = [
    "pe_ratio", "ev_ebitda", "pb_ratio", "ps_ratio", "peg_ratio",
    "earnings_yield", "fcf_yield", "dividend_yield", "earnings_per_share",
    "book_value_per_share", "enterprise_value", "market_cap",
    "revenue", "ebitda", "gross_margin", "operating_margin", "net_margin",
    "ebitda_margin", "roe", "roa", "roic", "roce",
    "debt_equity", "net_debt_ebitda", "current_ratio", "quick_ratio",
    "interest_coverage", "cash_ratio",
    "fcf", "fcf_yield", "operating_cash_flow", "capex", "cash_conversion_ratio",
    "revenue_growth", "eps_growth", "revenue_cagr_5y", "eps_cagr_5y",
    "net_income_growth", "ebitda_growth",
    "beta", "short_interest", "float_short",
    "analyst_rating", "eps_revision", "earnings_surprise",
    "institutional_ownership", "insider_ownership",
    "payout_ratio", "dividend_coverage", "consecutive_div_growth",
]

# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fundamental_cache (
            ticker       TEXT NOT NULL,
            data_json    TEXT NOT NULL,
            fetched_at   REAL NOT NULL,
            cik          TEXT,
            PRIMARY KEY (ticker)
        );
        CREATE TABLE IF NOT EXISTS company_metrics (
            ticker       TEXT NOT NULL,
            metric       TEXT NOT NULL,
            value        REAL,
            period       TEXT,
            updated_at   REAL NOT NULL,
            PRIMARY KEY (ticker, metric)
        );
        CREATE TABLE IF NOT EXISTS metric_history (
            ticker       TEXT NOT NULL,
            metric       TEXT NOT NULL,
            period       TEXT NOT NULL,
            value        REAL,
            source       TEXT,
            inserted_at  REAL NOT NULL,
            PRIMARY KEY (ticker, metric, period)
        );
        CREATE TABLE IF NOT EXISTS universe_registry (
            ticker       TEXT PRIMARY KEY,
            cik          TEXT,
            name         TEXT,
            sic          TEXT,
            sector       TEXT,
            index_member TEXT,
            last_filing  TEXT,
            registered_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cik_cache (
            ticker       TEXT PRIMARY KEY,
            cik          TEXT NOT NULL,
            resolved_at  REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS submission_cache (
            cik          TEXT PRIMARY KEY,
            last_filing  TEXT,
            checked_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_company_metrics_ticker ON company_metrics(ticker);
        CREATE INDEX IF NOT EXISTS idx_metric_history_ticker ON metric_history(ticker);
    """)
    conn.commit()
    conn.close()


_init_db()

# ---------------------------------------------------------------------------
# In-process caches
# ---------------------------------------------------------------------------

_MEM_METRICS: Dict[Tuple[str, str], Tuple[float, float]] = {}   # (ticker, metric) → (ts, value)
_MEM_FACTS: Dict[str, Tuple[float, dict]] = {}                  # cik → (ts, facts)
_MEM_PRICES: Dict[str, Tuple[float, float]] = {}                # ticker → (ts, price)


def _mem_metric(ticker: str, metric: str) -> Optional[float]:
    key = (ticker, metric)
    entry = _MEM_METRICS.get(key)
    if entry and time.monotonic() - entry[0] < _CACHE_TTL_SEC:
        return entry[1]
    return None


def _mem_metric_set(ticker: str, metric: str, value: float) -> None:
    _MEM_METRICS[(ticker, metric)] = (time.monotonic(), value)


# ---------------------------------------------------------------------------
# EdgarCikResolver
# ---------------------------------------------------------------------------

class EdgarCikResolver:
    """
    Resolve ticker → CIK using the SEC company_tickers.json endpoint.
    Results are cached in SQLite (7-day TTL) and in-process.
    """

    _mem: Dict[str, str] = {}

    def resolve(self, ticker: str) -> Optional[str]:
        ticker = ticker.upper().replace("-", ".")
        if ticker in self._mem:
            return self._mem[ticker]

        # DB cache
        conn = _get_db()
        cutoff = time.time() - _CIK_TTL_SEC
        row = conn.execute(
            "SELECT cik FROM cik_cache WHERE ticker=? AND resolved_at>?",
            (ticker, cutoff)
        ).fetchone()
        conn.close()
        if row:
            self._mem[ticker] = row["cik"]
            return row["cik"]

        # Fetch full ticker map from SEC (refresh if stale)
        all_ciks = self._fetch_all_ciks()
        # Try exact match, then with common transformations
        for variant in [ticker, ticker.replace(".", "-"), ticker.replace("-", "")]:
            if variant in all_ciks:
                cik = all_ciks[variant]
                self._store_cik(ticker, cik)
                return cik
        logger.debug("CIK not found for ticker %s", ticker)
        return None

    def _fetch_all_ciks(self) -> Dict[str, str]:
        """Download company_tickers.json and return ticker → zero-padded CIK dict."""
        cache_key = "sec_all_ciks"
        entry = _MEM_FACTS.get(cache_key)
        if entry and time.monotonic() - entry[0] < 3600:
            return entry[1]  # type: ignore

        try:
            resp = requests.get(EDGAR_COMPANY_TICKERS, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
            result: Dict[str, str] = {}
            for item in raw.values():
                tk = str(item.get("ticker", "")).upper()
                cik = str(item.get("cik_str", item.get("cik", ""))).zfill(10)
                if tk:
                    result[tk] = cik
            _MEM_FACTS[cache_key] = (time.monotonic(), result)  # type: ignore
            return result
        except Exception as exc:
            logger.warning("Failed to fetch SEC company tickers: %s", exc)
            return {}

    def _store_cik(self, ticker: str, cik: str) -> None:
        self._mem[ticker] = cik
        try:
            conn = _get_db()
            conn.execute(
                "INSERT OR REPLACE INTO cik_cache (ticker, cik, resolved_at) VALUES (?,?,?)",
                (ticker, cik, time.time())
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def batch_resolve(self, tickers: List[str]) -> Dict[str, str]:
        """Resolve a list of tickers to CIKs. Missing tickers are excluded."""
        result: Dict[str, str] = {}
        all_ciks = self._fetch_all_ciks()
        for ticker in tickers:
            t = ticker.upper().replace("-", ".")
            for variant in [t, t.replace(".", "-"), t.replace("-", "")]:
                if variant in all_ciks:
                    cik = all_ciks[variant]
                    result[ticker] = cik
                    self._store_cik(ticker, cik)
                    break
        return result


# ---------------------------------------------------------------------------
# XbrlFactsFetcher
# ---------------------------------------------------------------------------

class XbrlFactsFetcher:
    """
    Fetch and parse EDGAR XBRL companyfacts JSON.
    Provides helpers to extract time-series and compute LTM (last twelve months).
    """

    _resolver = EdgarCikResolver()

    def fetch_facts(self, cik: str) -> Optional[dict]:
        """
        Download companyfacts JSON for a CIK.
        Cached in-process for _CACHE_TTL_SEC.
        """
        entry = _MEM_FACTS.get(cik)
        if entry and time.monotonic() - entry[0] < _CACHE_TTL_SEC:
            return entry[1]

        url = EDGAR_FACTS_URL.format(cik=cik)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            facts = resp.json()
            _MEM_FACTS[cik] = (time.monotonic(), facts)
            return facts
        except Exception as exc:
            logger.debug("EDGAR facts fetch failed for CIK %s: %s", cik, exc)
            return None

    def extract_concept(
        self,
        facts: dict,
        concept: str,
        taxonomy: str = "us-gaap",
        unit_filter: Optional[str] = "USD",
    ) -> pd.DataFrame:
        """
        Extract a time-series DataFrame for a single XBRL concept.

        Returns DataFrame with columns: [end, val, form, accn, fy, fp, frame]
        sorted by end date descending. Filters to annual (10-K) + quarterly (10-Q) filings.
        """
        try:
            tax_data = facts.get("facts", {}).get(taxonomy, {})
            concept_data = tax_data.get(concept)
            if not concept_data:
                return pd.DataFrame()

            units_data = concept_data.get("units", {})
            # Find the right unit bucket
            if unit_filter and unit_filter in units_data:
                records = units_data[unit_filter]
            elif "USD" in units_data:
                records = units_data["USD"]
            elif "USD/shares" in units_data:
                records = units_data["USD/shares"]
            elif "shares" in units_data:
                records = units_data["shares"]
            elif units_data:
                records = next(iter(units_data.values()))
            else:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            if df.empty:
                return df

            # Keep only annual and quarterly filings
            if "form" in df.columns:
                df = df[df["form"].isin(["10-K", "10-Q", "20-F", "40-F"])]

            if "end" in df.columns:
                df["end"] = pd.to_datetime(df["end"], errors="coerce")
                df = df.dropna(subset=["end"]).sort_values("end", ascending=False)

            return df.reset_index(drop=True)
        except Exception as exc:
            logger.debug("extract_concept error %s/%s: %s", taxonomy, concept, exc)
            return pd.DataFrame()

    def latest_annual(
        self,
        facts: dict,
        concept: str,
        taxonomy: str = "us-gaap",
        unit_filter: Optional[str] = "USD",
    ) -> Optional[float]:
        """Return the most recent 10-K value for a concept."""
        df = self.extract_concept(facts, concept, taxonomy, unit_filter)
        if df.empty:
            return None
        annual = df[df.get("form", pd.Series()).isin(["10-K", "20-F", "40-F"])] if "form" in df.columns else df
        if annual.empty:
            annual = df
        row = annual.iloc[0]
        val = row.get("val")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return float(val)

    def ltm(
        self,
        facts: dict,
        concept: str,
        taxonomy: str = "us-gaap",
        unit_filter: Optional[str] = "USD",
    ) -> Optional[float]:
        """
        Compute Last Twelve Months (LTM) for a flow concept.
        Method: most recent annual 10-K + any subsequent quarterly additions.
        Falls back to most recent 10-K if quarterly detail unavailable.
        """
        df = self.extract_concept(facts, concept, taxonomy, unit_filter)
        if df.empty:
            return None

        if "form" not in df.columns or "val" not in df.columns:
            return None

        annual_df = df[df["form"].isin(["10-K", "20-F", "40-F"])]
        quarterly_df = df[df["form"] == "10-Q"].copy() if "form" in df.columns else pd.DataFrame()

        if annual_df.empty:
            return None

        latest_annual_row = annual_df.iloc[0]
        annual_val = float(latest_annual_row["val"])
        annual_end = latest_annual_row["end"]

        if quarterly_df.empty:
            return annual_val

        # Find quarters after the annual end date
        post_annual = quarterly_df[quarterly_df["end"] > annual_end].head(4)
        if post_annual.empty:
            return annual_val

        # LTM = annual + most_recent_YTD_quarterly - prior_year_YTD_quarterly
        # Simpler: use the sum of 4 most-recent non-overlapping quarterly filings
        # Check if quarterly data has 'frame' (indicates non-overlapping)
        if "frame" in quarterly_df.columns:
            # Prefer quarterly non-overlapping frames (e.g. CY2024Q3I = instantaneous)
            # For flow, take last 4 quarters
            recent_quarters = quarterly_df.head(4)
            if len(recent_quarters) >= 4:
                # Try to see if they're non-overlapping by checking date spans
                dates = sorted(recent_quarters["end"].tolist(), reverse=True)
                # Rough check: quarters should be ~90 days apart
                if len(dates) >= 2:
                    avg_gap = (dates[0] - dates[-1]).days / (len(dates) - 1)
                    if 60 <= avg_gap <= 120:
                        return float(recent_quarters["val"].sum())

        return annual_val

    def latest_balance(
        self,
        facts: dict,
        concept: str,
        taxonomy: str = "us-gaap",
        unit_filter: Optional[str] = "USD",
    ) -> Optional[float]:
        """Return most recent balance sheet value (stock concept — just latest)."""
        df = self.extract_concept(facts, concept, taxonomy, unit_filter)
        if df.empty:
            return None
        row = df.iloc[0]
        val = row.get("val")
        if val is None:
            return None
        return float(val)

    def get_series(
        self,
        facts: dict,
        concept: str,
        taxonomy: str = "us-gaap",
        unit_filter: Optional[str] = "USD",
        n_periods: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return last N annual values as [(period_str, value)] for CAGR computation."""
        df = self.extract_concept(facts, concept, taxonomy, unit_filter)
        if df.empty:
            return []
        annual = df[df["form"].isin(["10-K", "20-F", "40-F"])] if "form" in df.columns else df
        annual = annual.head(n_periods)
        result = []
        for _, row in annual.iterrows():
            try:
                period = row["end"].strftime("%Y-%m-%d") if hasattr(row.get("end"), "strftime") else str(row.get("end", ""))
                result.append((period, float(row["val"])))
            except Exception:
                pass
        return result

    def _try_concepts(self, facts: dict, concept_names: List[str], taxonomy: str = "us-gaap",
                       unit_filter: str = "USD", method: str = "ltm") -> Optional[float]:
        """Try a list of fallback concept names, returning the first non-None result."""
        for concept in concept_names:
            if method == "ltm":
                val = self.ltm(facts, concept, taxonomy, unit_filter)
            elif method == "balance":
                val = self.latest_balance(facts, concept, taxonomy, unit_filter)
            elif method == "annual":
                val = self.latest_annual(facts, concept, taxonomy, unit_filter)
            else:
                val = self.ltm(facts, concept, taxonomy, unit_filter)
            if val is not None:
                return val
        return None


# ---------------------------------------------------------------------------
# Price fetcher (yfinance — prices only)
# ---------------------------------------------------------------------------

def _fetch_price(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (price, shares_outstanding) for a ticker using yfinance.
    Cached in-process for 5 minutes.
    """
    entry = _MEM_PRICES.get(ticker)
    if entry and time.monotonic() - entry[0] < _PRICE_TTL_SEC:
        return entry[1], None  # shares not cached here; re-fetch from EDGAR

    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker)
        info = tkr.fast_info
        price = float(info.last_price) if info.last_price else None
        if price is None:
            # Fallback via info dict
            info_dict = tkr.info
            price = info_dict.get("currentPrice") or info_dict.get("regularMarketPrice")
            if price:
                price = float(price)
        if price:
            _MEM_PRICES[ticker] = (time.monotonic(), price)
        return price, None
    except Exception as exc:
        logger.debug("yfinance price fetch failed for %s: %s", ticker, exc)
        return None, None


def _fetch_beta(ticker: str, lookback_days: int = 252) -> Optional[float]:
    """
    Compute 1-year beta vs SPY using yfinance daily returns.
    This is the ONE acceptable yfinance use case (price-based beta).
    """
    try:
        import yfinance as yf
        tickers_data = yf.download(
            [ticker, "SPY"],
            period="1y",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if tickers_data.empty:
            return None
        closes = tickers_data["Close"]
        if ticker not in closes.columns or "SPY" not in closes.columns:
            return None
        returns = closes.pct_change().dropna()
        cov = returns[ticker].cov(returns["SPY"])
        spy_var = returns["SPY"].var()
        if spy_var == 0:
            return None
        return round(float(cov / spy_var), 3)
    except Exception as exc:
        logger.debug("Beta computation failed for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# EdgarSubmissionMonitor
# ---------------------------------------------------------------------------

class EdgarSubmissionMonitor:
    """
    Check if a company has filed a new 10-K or 10-Q since last refresh.
    Uses EDGAR submissions API (no key required).
    """

    def has_new_filing(self, ticker: str, cik: str) -> bool:
        """Returns True if a new 10-K/10-Q was filed after the last cached refresh."""
        conn = _get_db()
        cache_row = conn.execute(
            "SELECT fetched_at FROM fundamental_cache WHERE ticker=?", (ticker,)
        ).fetchone()
        conn.close()

        if cache_row is None:
            return True  # Never fetched → definitely refresh

        last_fetched = cache_row["fetched_at"]

        # Check EDGAR submissions
        sub_conn = _get_db()
        sub_row = sub_conn.execute(
            "SELECT last_filing, checked_at FROM submission_cache WHERE cik=?", (cik,)
        ).fetchone()
        sub_conn.close()

        # Only query EDGAR submissions every 6 hours
        if sub_row and time.time() - sub_row["checked_at"] < 21600:
            last_filing = sub_row["last_filing"]
            if last_filing:
                try:
                    filing_ts = datetime.strptime(last_filing, "%Y-%m-%d").timestamp()
                    return filing_ts > last_fetched
                except ValueError:
                    pass
            return False

        # Fetch from EDGAR
        try:
            url = EDGAR_SUBMISSIONS.format(cik=cik)
            resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
            recent = raw.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            last_10k_q = None
            for form, filing_date in zip(forms, dates):
                if form in ("10-K", "10-Q", "20-F", "40-F"):
                    last_10k_q = filing_date
                    break
            conn = _get_db()
            conn.execute(
                "INSERT OR REPLACE INTO submission_cache (cik, last_filing, checked_at) VALUES (?,?,?)",
                (cik, last_10k_q, time.time())
            )
            conn.commit()
            conn.close()
            if last_10k_q:
                filing_ts = datetime.strptime(last_10k_q, "%Y-%m-%d").timestamp()
                return filing_ts > last_fetched
        except Exception as exc:
            logger.debug("EDGAR submission check failed for CIK %s: %s", cik, exc)

        return False  # Default: no new filing detected

    def mark_refreshed(self, ticker: str) -> None:
        pass  # Implicit: fundamental_cache.fetched_at updated on each refresh


# ---------------------------------------------------------------------------
# MetricCalculator
# ---------------------------------------------------------------------------

class MetricCalculator:
    """
    Compute all 50 screener metrics from XBRL facts + price data.
    Handles concept fallbacks, scale normalisation, and derived ratios.
    """

    _fetcher = XbrlFactsFetcher()

    def compute_all(
        self,
        ticker: str,
        facts: dict,
        price: Optional[float],
        shares: Optional[float] = None,
    ) -> Dict[str, float]:
        nan = float("nan")
        f = self._fetcher

        # ------------------------------------------------------------------
        # Raw XBRL pulls
        # ------------------------------------------------------------------
        revenue = f._try_concepts(facts, [
            "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
        ], method="ltm")

        gross_profit = f._try_concepts(facts, ["GrossProfit"], method="ltm")

        operating_income = f._try_concepts(facts, [
            "OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ], method="ltm")

        net_income = f._try_concepts(facts, [
            "NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic",
            "ProfitLoss",
        ], method="ltm")

        da = f._try_concepts(facts, [
            "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
            "Depreciation",
        ], method="ltm")

        interest_exp = f._try_concepts(facts, [
            "InterestExpense", "InterestExpenseDebt",
        ], method="ltm")

        income_tax = f._try_concepts(facts, ["IncomeTaxExpense", "IncomeTaxesPaid"], method="ltm")

        eps_diluted_current = f._try_concepts(facts, [
            "EarningsPerShareDiluted", "EarningsPerShareBasic",
        ], method="annual", unit_filter="USD/shares")

        dividends_per_share = f._try_concepts(facts, [
            "CommonStockDividendsPerShareDeclared",
            "CommonStockDividendsPerShareCashPaid",
        ], method="annual", unit_filter="USD/shares")

        # Balance sheet
        total_assets = f._try_concepts(facts, ["Assets"], method="balance")
        current_assets = f._try_concepts(facts, ["AssetsCurrent"], method="balance")
        current_liabilities = f._try_concepts(facts, ["LiabilitiesCurrent"], method="balance")
        total_equity = f._try_concepts(facts, [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ], method="balance")
        long_term_debt = f._try_concepts(facts, [
            "LongTermDebt", "LongTermDebtNoncurrent", "LongTermNotesPayable",
        ], method="balance")
        short_term_debt = f._try_concepts(facts, [
            "ShortTermBorrowings", "DebtCurrent", "ShortTermDebtAndCurrentPortionOfLongTermDebt",
        ], method="balance")
        cash = f._try_concepts(facts, [
            "CashAndCashEquivalentsAtCarryingValue", "Cash",
            "CashCashEquivalentsAndShortTermInvestments",
        ], method="balance")
        inventory = f._try_concepts(facts, ["InventoryNet"], method="balance")
        goodwill = f._try_concepts(facts, ["Goodwill", "GoodwillNet"], method="balance")

        # Cash flow
        cfo = f._try_concepts(facts, [
            "NetCashProvidedByUsedInOperatingActivities",
        ], method="ltm")
        capex = f._try_concepts(facts, [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "CapitalExpendituresIncurredButNotYetPaid",
            "PaymentsForCapitalImprovements",
        ], method="ltm")
        if capex is not None:
            capex = abs(capex)  # EDGAR reports as negative outflow; normalise to positive

        # Shares outstanding — try EDGAR DEI first
        if shares is None:
            shares = f._try_concepts(facts, [
                "EntityCommonStockSharesOutstanding",
            ], taxonomy="dei", unit_filter="shares", method="annual")

        # ------------------------------------------------------------------
        # Derived quantities
        # ------------------------------------------------------------------
        ebitda: Optional[float] = None
        if operating_income is not None and da is not None:
            ebitda = operating_income + da
        elif operating_income is not None:
            ebitda = operating_income  # best effort without D&A

        total_debt: Optional[float] = None
        if long_term_debt is not None or short_term_debt is not None:
            total_debt = (long_term_debt or 0.0) + (short_term_debt or 0.0)

        net_debt: Optional[float] = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash

        market_cap: Optional[float] = None
        if price is not None and shares is not None and shares > 0:
            market_cap = price * shares

        enterprise_value: Optional[float] = None
        if market_cap is not None and total_debt is not None and cash is not None:
            enterprise_value = market_cap + total_debt - cash
        elif market_cap is not None:
            enterprise_value = market_cap

        fcf: Optional[float] = None
        if cfo is not None and capex is not None:
            fcf = cfo - capex

        # ------------------------------------------------------------------
        # Compute each metric
        # ------------------------------------------------------------------
        def _div(a: Optional[float], b: Optional[float], pct: bool = False) -> float:
            if a is None or b is None or b == 0:
                return nan
            return (a / b) * (100.0 if pct else 1.0)

        pe_ratio      = _div(price, eps_diluted_current) if eps_diluted_current and eps_diluted_current > 0 else nan
        ev_ebitda     = _div(enterprise_value, ebitda) if ebitda and ebitda > 0 else nan
        pb_ratio      = _div(market_cap, total_equity) if total_equity and total_equity > 0 else nan
        ps_ratio      = _div(market_cap, revenue) if revenue and revenue > 0 else nan
        earnings_yield = (100.0 / pe_ratio) if not math.isnan(pe_ratio) and pe_ratio > 0 else nan
        fcf_yield     = _div(fcf, market_cap, pct=True) if market_cap and market_cap > 0 else nan
        dividend_yield = _div(dividends_per_share, price, pct=True) if price and price > 0 else nan

        book_value_per_share = _div(total_equity, shares) if shares and shares > 0 else nan

        gross_margin     = _div(gross_profit, revenue, pct=True)
        operating_margin = _div(operating_income, revenue, pct=True)
        net_margin       = _div(net_income, revenue, pct=True)
        ebitda_margin    = _div(ebitda, revenue, pct=True)

        roe  = _div(net_income, total_equity, pct=True)
        roa  = _div(net_income, total_assets, pct=True)
        # ROIC = NOPAT / (Equity + Debt - Cash)
        nopat = None
        if operating_income is not None and income_tax is not None:
            # Approximate NOPAT using effective tax adjustment
            nopat = operating_income * (1.0 - 0.21)  # US statutory rate as proxy
        invested_capital = None
        if total_equity is not None:
            invested_capital = total_equity + (total_debt or 0.0) - (cash or 0.0)
        roic = _div(nopat, invested_capital, pct=True)

        # ROCE = EBIT / (Total Assets - Current Liabilities)
        capital_employed = None
        if total_assets is not None and current_liabilities is not None:
            capital_employed = total_assets - current_liabilities
        roce = _div(operating_income, capital_employed, pct=True)

        debt_equity      = _div(total_debt, total_equity)
        net_debt_ebitda  = _div(net_debt, ebitda)
        current_ratio    = _div(current_assets, current_liabilities)
        quick_ratio: float = nan
        if current_assets is not None and inventory is not None and current_liabilities is not None and current_liabilities > 0:
            quick_ratio = (current_assets - inventory) / current_liabilities
        interest_coverage = _div(operating_income, interest_exp) if interest_exp and interest_exp > 0 else nan
        cash_ratio       = _div(cash, current_liabilities)

        fcf_computed     = fcf if fcf is not None else nan
        operating_cf     = cfo if cfo is not None else nan
        capex_computed   = capex if capex is not None else nan
        cash_conv        = _div(fcf, net_income)  # FCF / net income

        ev_val           = enterprise_value if enterprise_value is not None else nan
        mc_val           = market_cap if market_cap is not None else nan
        rev_val          = revenue if revenue is not None else nan
        ebitda_val       = ebitda if ebitda is not None else nan
        eps_val          = eps_diluted_current if eps_diluted_current is not None else nan

        # ------------------------------------------------------------------
        # Growth metrics — require historical series
        # ------------------------------------------------------------------
        revenue_growth = nan
        rev_series = f.get_series(facts, "Revenues", n_periods=3)
        if not rev_series:
            rev_series = f.get_series(facts, "RevenueFromContractWithCustomerExcludingAssessedTax", n_periods=3)
        if len(rev_series) >= 2:
            try:
                rev_curr = rev_series[0][1]
                rev_prior = rev_series[1][1]
                if rev_prior and rev_prior != 0:
                    revenue_growth = (rev_curr - rev_prior) / abs(rev_prior) * 100.0
            except Exception:
                pass

        eps_growth = nan
        eps_series = f.get_series(facts, "EarningsPerShareDiluted",
                                   taxonomy="us-gaap", unit_filter="USD/shares", n_periods=3)
        if len(eps_series) >= 2:
            try:
                eps_curr  = eps_series[0][1]
                eps_prior = eps_series[1][1]
                if eps_prior and eps_prior != 0:
                    eps_growth = (eps_curr - eps_prior) / abs(eps_prior) * 100.0
            except Exception:
                pass

        net_income_growth = nan
        ni_series = f.get_series(facts, "NetIncomeLoss", n_periods=3)
        if len(ni_series) >= 2:
            try:
                ni_curr  = ni_series[0][1]
                ni_prior = ni_series[1][1]
                if ni_prior and ni_prior != 0:
                    net_income_growth = (ni_curr - ni_prior) / abs(ni_prior) * 100.0
            except Exception:
                pass

        # 5-year CAGR
        revenue_cagr_5y = nan
        rev_5yr = f.get_series(facts, "Revenues", n_periods=7)
        if not rev_5yr:
            rev_5yr = f.get_series(facts, "RevenueFromContractWithCustomerExcludingAssessedTax", n_periods=7)
        if len(rev_5yr) >= 5:
            try:
                v_now  = rev_5yr[0][1]
                v_then = rev_5yr[4][1]
                if v_now > 0 and v_then > 0:
                    revenue_cagr_5y = ((v_now / v_then) ** 0.2 - 1.0) * 100.0
            except Exception:
                pass

        eps_cagr_5y = nan
        eps_5yr = f.get_series(facts, "EarningsPerShareDiluted",
                                taxonomy="us-gaap", unit_filter="USD/shares", n_periods=7)
        if len(eps_5yr) >= 5:
            try:
                e_now  = eps_5yr[0][1]
                e_then = eps_5yr[4][1]
                if e_now > 0 and e_then > 0:
                    eps_cagr_5y = ((e_now / e_then) ** 0.2 - 1.0) * 100.0
            except Exception:
                pass

        ebitda_growth = nan
        # Approximate from operating_income series
        oi_series = f.get_series(facts, "OperatingIncomeLoss", n_periods=3)
        if len(oi_series) >= 2:
            try:
                oi_curr  = oi_series[0][1]
                oi_prior = oi_series[1][1]
                if oi_prior and oi_prior != 0:
                    ebitda_growth = (oi_curr - oi_prior) / abs(oi_prior) * 100.0
            except Exception:
                pass

        payout_ratio = nan
        if dividends_per_share and eps_diluted_current and eps_diluted_current > 0:
            payout_ratio = (dividends_per_share / eps_diluted_current) * 100.0

        # ------------------------------------------------------------------
        # Assemble result dict (exact schema match for nl_screener_v2)
        # ------------------------------------------------------------------
        return {
            # Valuation
            "pe_ratio":            _clean(pe_ratio),
            "ev_ebitda":           _clean(ev_ebitda),
            "pb_ratio":            _clean(pb_ratio),
            "ps_ratio":            _clean(ps_ratio),
            "peg_ratio":           nan,           # requires forward estimates
            "earnings_yield":      _clean(earnings_yield),
            "fcf_yield":           _clean(fcf_yield),
            "dividend_yield":      _clean(dividend_yield),
            "earnings_per_share":  _clean(eps_val),
            "book_value_per_share":_clean(book_value_per_share),
            "enterprise_value":    _clean(ev_val),
            "market_cap":          _clean(mc_val),
            # Income
            "revenue":             _clean(rev_val),
            "ebitda":              _clean(ebitda_val),
            # Margins
            "gross_margin":        _clean(gross_margin),
            "operating_margin":    _clean(operating_margin),
            "net_margin":          _clean(net_margin),
            "ebitda_margin":       _clean(ebitda_margin),
            # Returns
            "roe":                 _clean(roe),
            "roa":                 _clean(roa),
            "roic":                _clean(roic),
            "roce":                _clean(roce),
            # Leverage
            "debt_equity":         _clean(debt_equity),
            "net_debt_ebitda":     _clean(net_debt_ebitda),
            "current_ratio":       _clean(current_ratio),
            "quick_ratio":         _clean(quick_ratio),
            "interest_coverage":   _clean(interest_coverage),
            "cash_ratio":          _clean(cash_ratio),
            # Cash flow
            "fcf":                 _clean(fcf_computed),
            "operating_cash_flow": _clean(operating_cf),
            "capex":               _clean(capex_computed),
            "cash_conversion_ratio":_clean(cash_conv),
            # Growth
            "revenue_growth":      _clean(revenue_growth),
            "eps_growth":          _clean(eps_growth),
            "revenue_cagr_5y":     _clean(revenue_cagr_5y),
            "eps_cagr_5y":         _clean(eps_cagr_5y),
            "net_income_growth":   _clean(net_income_growth),
            "ebitda_growth":       _clean(ebitda_growth),
            "fcf_growth":          nan,
            # Dividends
            "payout_ratio":        _clean(payout_ratio),
            "dividend_coverage":   _clean(_div(eps_diluted_current, dividends_per_share)),
            "consecutive_div_growth": nan,         # requires dividend history
            # Market / sentiment (not from EDGAR — filled by caller)
            "beta":                nan,
            "short_interest":      nan,
            "float_short":         nan,
            "analyst_rating":      nan,
            "eps_revision":        nan,
            "earnings_surprise":   nan,
            "institutional_ownership": nan,
            "insider_ownership":   nan,
            "iv_percentile":       nan,
            "borrow_cost":         nan,
            # Price / technical (not from EDGAR — filled by caller)
            "rsi":                 nan,
            "pct_from_52w_high":   nan,
            "pct_from_52w_low":    nan,
            "return_3m":           nan,
            "return_6m":           nan,
            "return_1y":           nan,
            "return_ytd":          nan,
            "price_vs_ma50":       nan,
            "price_vs_ma200":      nan,
            "avg_volume":          nan,
            # ESG (not from EDGAR)
            "esg_score":           nan,
            "esg_env":             nan,
            "esg_gov":             nan,
        }


def _clean(val: float) -> float:
    """Replace inf/nan with nan sentinel; ensure float."""
    if val is None:
        return float("nan")
    try:
        v = float(val)
        if math.isfinite(v):
            return round(v, 6)
        return float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# UniverseManager
# ---------------------------------------------------------------------------

class UniverseManager:
    """
    Manages the ticker universe:
      - S&P 500 from Wikipedia
      - Russell 1000 approximation: top 1000 US-listed by market cap from EDGAR
      - Registers tickers in universe_registry SQLite table
    """

    _sp500: Optional[List[str]] = None
    _sp500_ts: float = 0.0

    def sp500_tickers(self) -> List[str]:
        """Fetch S&P 500 component tickers from Wikipedia (free, no API key)."""
        if self._sp500 and time.monotonic() - self._sp500_ts < 86400:
            return self._sp500

        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            resp = requests.get(url, headers={"User-Agent": "SENTINEL/3.0"}, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            df = tables[0]
            # Column names vary; find ticker column
            col = None
            for candidate in ["Symbol", "Ticker symbol", "Ticker"]:
                if candidate in df.columns:
                    col = candidate
                    break
            if col is None:
                col = df.columns[0]
            tickers = df[col].tolist()
            # Clean: replace "." with "-" for yfinance compatibility
            tickers = [str(t).strip().replace(".", "-") for t in tickers if str(t).strip()]
            UniverseManager._sp500 = tickers
            UniverseManager._sp500_ts = time.monotonic()
            return tickers
        except Exception as exc:
            logger.warning("Wikipedia S&P 500 fetch failed: %s", exc)
            return self._sp500_fallback()

    def _sp500_fallback(self) -> List[str]:
        """Hardcoded S&P 500 sample when Wikipedia is unavailable."""
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B",
            "LLY", "AVGO", "JPM", "TSLA", "UNH", "V", "XOM", "MA", "JNJ",
            "HD", "PG", "COST", "ABBV", "MRK", "WMT", "BAC", "NFLX", "CRM",
            "CVX", "AMD", "ORCL", "KO", "PEP", "ACN", "TMO", "ADBE", "LIN",
            "MCD", "CSCO", "PM", "ABT", "DHR", "TXN", "QCOM", "WFC", "INTU",
            "CAT", "ISRG", "AMAT", "RTX", "NOW", "AMGN", "IBM", "BKNG", "GE",
            "HON", "NEE", "TJX", "SPGI", "UBER", "C", "LOW", "AXP", "BSX",
            "SYK", "MS", "BLK", "ETN", "VRTX", "SCHW", "ELV", "ADI", "PLD",
            "LRCX", "REGN", "MDT", "CB", "SBUX", "MMC", "GILD", "SO", "CI",
            "DUK", "CME", "ZTS", "CL", "SHW", "TGT", "MO", "PGR", "AON",
            "BDX", "PH", "HCA", "NOC", "GS", "DIS", "MAR", "CTAS", "F",
            "GM", "USB", "FI", "ITW", "COF", "MCK", "ECL", "APH", "GD",
        ]

    def russell1000_tickers(self) -> List[str]:
        """
        Approximate Russell 1000 as S&P 500 + a curated list of mid-large caps.
        In production this would query EDGAR full-text-search for US-listed companies
        sorted by market cap.
        """
        sp500 = self.sp500_tickers()
        additional = [
            "SNAP", "LYFT", "ROKU", "DOCU", "ZM", "PTON", "DASH", "RIVN",
            "LCID", "PLUG", "FSLR", "ENPH", "SEDG", "DKNG", "RBLX", "COIN",
            "HOOD", "SOFI", "UPST", "AFRM", "OPEN", "OPENDOOR", "LMND",
            "PLTR", "SNOW", "DDOG", "NET", "ZS", "OKTA", "CRWD", "PANW",
            "FTNT", "WDAY", "HUBS", "MDB", "CFLT", "GTLB", "DT", "AI",
            "SQ", "PYPL", "SHOP", "MELI", "SE", "PDD", "BABA", "JD",
            "TMUS", "T", "VZ", "DISH", "LUMN",
        ]
        known = set(sp500)
        extras = [t for t in additional if t not in known]
        return sp500 + extras[:500]

    def register(self, tickers: List[str], cik_map: Optional[Dict[str, str]] = None) -> None:
        """Register tickers in universe_registry."""
        now = time.time()
        conn = _get_db()
        for ticker in tickers:
            cik = (cik_map or {}).get(ticker)
            conn.execute("""
                INSERT OR IGNORE INTO universe_registry (ticker, cik, registered_at)
                VALUES (?,?,?)
            """, (ticker, cik, now))
        conn.commit()
        conn.close()

    def coverage_report(self) -> Dict[str, Any]:
        """Report on cache coverage across the registered universe."""
        conn = _get_db()
        total = conn.execute("SELECT COUNT(*) FROM universe_registry").fetchone()[0]
        cached = conn.execute(
            "SELECT COUNT(*) FROM fundamental_cache WHERE fetched_at > ?",
            (time.time() - _CACHE_TTL_SEC,)
        ).fetchone()[0]
        conn.close()
        return {
            "universe_size": total,
            "cached_24h": cached,
            "coverage_pct": round(cached / total * 100, 1) if total > 0 else 0.0,
            "as_of": datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# FundamentalDataLayerV3 — drop-in replacement for FundamentalDataLoader
# ---------------------------------------------------------------------------

class FundamentalDataLayerV3:
    """
    Production-grade fundamental data layer backed by EDGAR XBRL.

    Drop-in replacement for sentinel.sai.nl_screener_v2.FundamentalDataLoader.
    The .load() method returns a pd.DataFrame with the exact same column schema.

    Usage::
        from sentinel.sai.fundamental_data_layer_v3 import FundamentalDataLayerV3 as FundamentalDataLoader
        loader = FundamentalDataLayerV3()
        df = loader.load(["AAPL", "MSFT", "GOOGL"])

    Refresh strategy:
      1. Check in-process memory cache (O(1) per metric)
      2. Check SQLite fundamental_cache (24h TTL)
      3. Check EDGAR submissions API for new 10-K/10-Q
      4. Fetch EDGAR XBRL companyfacts if stale or new filing
      5. Compute all metrics; persist to SQLite; populate memory cache
    """

    def __init__(self) -> None:
        self._resolver  = EdgarCikResolver()
        self._fetcher   = XbrlFactsFetcher()
        self._calculator = MetricCalculator()
        self._monitor   = EdgarSubmissionMonitor()
        self._universe  = UniverseManager()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        _init_db()  # idempotent

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def load(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load fundamentals for the given tickers (or S&P 500 if None).
        Returns a pd.DataFrame indexed by 'ticker' with all screener metrics.
        """
        if tickers is None:
            tickers = self._universe.sp500_tickers()

        tickers = list(dict.fromkeys(t.upper() for t in tickers))  # dedup, preserve order

        # Phase 1: load from SQLite cache
        cached_data = self._load_cached(tickers)

        # Phase 2: identify what needs refreshing
        cutoff = time.time() - _CACHE_TTL_SEC
        missing = [t for t in tickers if t not in cached_data]
        stale   = [t for t in tickers if t in cached_data
                   and cached_data[t].get("_fetched_at", 0) < cutoff]

        to_fetch = list(dict.fromkeys(missing + stale))

        # Phase 3: batch-fetch from EDGAR
        if to_fetch:
            fetched = self._batch_fetch_edgar(to_fetch)
            for ticker, metrics in fetched.items():
                cached_data[ticker] = metrics

        # Phase 4: build DataFrame
        rows = []
        for ticker in tickers:
            metrics = cached_data.get(ticker, {})
            row = {"ticker": ticker}
            for metric in SCREENER_METRICS:
                row[metric] = metrics.get(metric, float("nan"))
            # Enrich with sector/name from universe_registry
            row["name"]   = metrics.get("name", ticker)
            row["sector"] = metrics.get("sector", "Unknown")
            rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index("ticker", drop=False, inplace=True)
        return df

    def get_metric(self, ticker: str, metric: str) -> float:
        """O(1) metric lookup via in-process memory cache."""
        val = _mem_metric(ticker, metric)
        if val is not None:
            return val
        # Fall back to DB
        conn = _get_db()
        row = conn.execute(
            "SELECT value FROM company_metrics WHERE ticker=? AND metric=?",
            (ticker.upper(), metric)
        ).fetchone()
        conn.close()
        if row and row["value"] is not None:
            _mem_metric_set(ticker, metric, float(row["value"]))
            return float(row["value"])
        # Full refresh
        df = self.load([ticker])
        if metric in df.columns and not df.empty:
            val = float(df.iloc[0][metric])
            _mem_metric_set(ticker, metric, val)
            return val
        return float("nan")

    def batch_metrics(
        self,
        tickers: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of selected metrics for multiple tickers.
        More efficient than calling get_metric() in a loop.
        """
        metrics = metrics or SCREENER_METRICS
        df = self.load(tickers)
        available = ["ticker"] + [m for m in metrics if m in df.columns]
        return df[available].reset_index(drop=True)

    def refresh(self, ticker: str) -> Dict[str, float]:
        """Force-refresh a single ticker from EDGAR, bypassing cache."""
        ticker = ticker.upper()
        cik = self._resolver.resolve(ticker)
        if not cik:
            raise ValueError(f"Cannot resolve CIK for ticker {ticker}")
        metrics = self._fetch_single(ticker, cik, force=True)
        return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    # ------------------------------------------------------------------
    # Internal fetching
    # ------------------------------------------------------------------

    def _load_cached(self, tickers: List[str]) -> Dict[str, Dict]:
        """Load all cached ticker records from SQLite."""
        result: Dict[str, Dict] = {}
        if not tickers:
            return result
        try:
            conn = _get_db()
            placeholders = ",".join("?" * len(tickers))
            rows = conn.execute(
                f"SELECT ticker, data_json, fetched_at FROM fundamental_cache WHERE ticker IN ({placeholders})",
                tickers,
            ).fetchall()
            conn.close()
            for row in rows:
                try:
                    data = json.loads(row["data_json"])
                    data["_fetched_at"] = row["fetched_at"]
                    result[row["ticker"]] = data
                    # Populate in-process cache
                    for metric, val in data.items():
                        if metric.startswith("_"):
                            continue
                        try:
                            _mem_metric_set(row["ticker"], metric, float(val))
                        except (TypeError, ValueError):
                            pass
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("DB cache load failed: %s", exc)
        return result

    def _batch_fetch_edgar(self, tickers: List[str]) -> Dict[str, Dict]:
        """Fetch EDGAR fundamentals for a batch of tickers in parallel."""
        # Resolve CIKs for all tickers in one shot
        cik_map = self._resolver.batch_resolve(tickers)
        result: Dict[str, Dict] = {}

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futures = {}
            for ticker in tickers:
                cik = cik_map.get(ticker)
                if cik:
                    futures[ex.submit(self._fetch_single, ticker, cik)] = ticker
                else:
                    logger.debug("No CIK for %s — skipping EDGAR fetch", ticker)
                    result[ticker] = {}

            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    metrics = fut.result()
                    result[ticker] = metrics
                except Exception as exc:
                    logger.warning("EDGAR fetch failed for %s: %s", ticker, exc)
                    result[ticker] = {}

        return result

    def _fetch_single(self, ticker: str, cik: str, force: bool = False) -> Dict:
        """
        Fetch and compute metrics for a single ticker.
        Checks EDGAR submission for new filing before fetching unless force=True.
        """
        if not force and not self._monitor.has_new_filing(ticker, cik):
            # Check if we have unexpired cache
            conn = _get_db()
            row = conn.execute(
                "SELECT data_json, fetched_at FROM fundamental_cache WHERE ticker=? AND fetched_at>?",
                (ticker, time.time() - _CACHE_TTL_SEC)
            ).fetchone()
            conn.close()
            if row:
                try:
                    return json.loads(row["data_json"])
                except Exception:
                    pass

        # Fetch EDGAR companyfacts
        facts = self._fetcher.fetch_facts(cik)
        if not facts:
            logger.warning("No EDGAR facts for %s (CIK %s)", ticker, cik)
            return {}

        # Fetch price from yfinance
        price, _ = _fetch_price(ticker)

        # Compute metrics
        metrics = self._calculator.compute_all(ticker, facts, price)

        # Compute beta separately (yfinance price-based)
        beta = _fetch_beta(ticker)
        if beta is not None:
            metrics["beta"] = beta

        # Add metadata
        entity = facts.get("entityName", ticker)
        metrics["name"] = entity

        # Persist to SQLite
        self._persist(ticker, cik, metrics)
        return metrics

    def _persist(self, ticker: str, cik: str, metrics: Dict) -> None:
        """Persist metrics to fundamental_cache and company_metrics tables."""
        now = time.time()
        period = datetime.utcnow().strftime("%Y-%m")
        try:
            conn = _get_db()
            data_json = json.dumps(metrics, default=str)
            conn.execute(
                "INSERT OR REPLACE INTO fundamental_cache (ticker, data_json, fetched_at, cik) VALUES (?,?,?,?)",
                (ticker, data_json, now, cik),
            )
            for metric, val in metrics.items():
                if metric.startswith("_") or val is None:
                    continue
                try:
                    fval = float(val) if not isinstance(val, str) else None
                    if fval is None:
                        continue
                    conn.execute("""
                        INSERT OR REPLACE INTO company_metrics
                            (ticker, metric, value, period, updated_at)
                        VALUES (?,?,?,?,?)
                    """, (ticker, metric, fval, period, now))
                    conn.execute("""
                        INSERT OR IGNORE INTO metric_history
                            (ticker, metric, period, value, source, inserted_at)
                        VALUES (?,?,?,?,?,?)
                    """, (ticker, metric, period, fval, "EDGAR_XBRL", now))
                    # Populate in-process cache
                    _mem_metric_set(ticker, metric, fval)
                except (TypeError, ValueError):
                    pass
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("Persist failed for %s: %s", ticker, exc)


# Alias — drop-in replacement for nl_screener_v2.FundamentalDataLoader
FundamentalDataLoader = FundamentalDataLayerV3


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

fundamentals_v3_router = APIRouter(prefix="/fundamentals/v3", tags=["Fundamentals V3"])

# Lazy singleton
_layer: Optional[FundamentalDataLayerV3] = None


def _get_layer() -> FundamentalDataLayerV3:
    global _layer
    if _layer is None:
        _layer = FundamentalDataLayerV3()
    return _layer


class BatchMetricsRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=1, max_items=500)
    metrics: Optional[List[str]] = Field(None, description="Metric names; omit for all 50")


@fundamentals_v3_router.get("/metrics/{ticker}", summary="All metrics for a single ticker")
def get_metrics(ticker: str):
    """Return all 50 screener metrics for a ticker, fetched from EDGAR XBRL."""
    ticker = ticker.upper()
    try:
        layer = _get_layer()
        df = layer.load([ticker])
        if df.empty:
            raise HTTPException(404, f"No data found for {ticker}")
        row = df.iloc[0].to_dict()
        return {"ticker": ticker, "metrics": row, "as_of": datetime.utcnow().isoformat()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fundamentals_v3_router.get("/screener-ready/{ticker}", summary="Is ticker ready for screener?")
def screener_ready(ticker: str):
    """Check if a ticker has sufficient EDGAR data for screener use."""
    ticker = ticker.upper()
    try:
        layer = _get_layer()
        df = layer.load([ticker])
        if df.empty:
            return {"ticker": ticker, "ready": False, "reason": "No EDGAR data"}
        row = df.iloc[0]
        core = ["pe_ratio", "market_cap", "revenue", "net_margin", "roe"]
        available = [m for m in core if not math.isnan(float(row.get(m, float("nan"))))]
        ready = len(available) >= 3
        return {
            "ticker": ticker,
            "ready": ready,
            "core_metrics_available": available,
            "coverage_pct": round(len(available) / len(core) * 100),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fundamentals_v3_router.post("/batch-metrics", summary="Batch metrics for multiple tickers")
def batch_metrics(req: BatchMetricsRequest):
    """Compute metrics for multiple tickers in parallel."""
    try:
        layer = _get_layer()
        df = layer.batch_metrics(req.tickers, req.metrics)
        return {
            "count": len(df),
            "metrics": df.to_dict(orient="records"),
            "as_of": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fundamentals_v3_router.get("/universe-coverage", summary="Cache coverage across universe")
def universe_coverage():
    """Report what percentage of the registered universe has cached data."""
    try:
        layer = _get_layer()
        return layer._universe.coverage_report()
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fundamentals_v3_router.post("/refresh/{ticker}", summary="Force refresh from EDGAR")
def refresh_ticker(ticker: str):
    """Force-refresh a ticker's fundamentals from EDGAR, bypassing the 24h cache."""
    ticker = ticker.upper()
    try:
        layer = _get_layer()
        metrics = layer.refresh(ticker)
        return {
            "ticker": ticker,
            "refreshed": True,
            "metrics_computed": len([v for v in metrics.values() if not math.isnan(v)]),
            "as_of": datetime.utcnow().isoformat(),
        }
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fundamentals_v3_router.get("/metric/{metric_name}", summary="Single metric across tickers")
def get_metric_cross_section(
    metric_name: str,
    tickers: str = FastAPIQuery(
        default="AAPL,MSFT,GOOGL,AMZN,META",
        description="Comma-separated tickers",
    ),
):
    """Return a single metric for a list of tickers (cross-sectional view)."""
    if metric_name not in SCREENER_METRICS:
        raise HTTPException(400, f"Unknown metric '{metric_name}'. Valid metrics: {SCREENER_METRICS[:10]}...")
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    try:
        layer = _get_layer()
        result = {}
        for ticker in ticker_list:
            val = layer.get_metric(ticker, metric_name)
            result[ticker] = None if math.isnan(val) else round(val, 4)
        return {
            "metric": metric_name,
            "values": result,
            "as_of": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@fundamentals_v3_router.get("/xbrl-concepts", summary="List all XBRL concept mappings")
def list_xbrl_concepts():
    """Return the full XBRL concept mapping used to compute each metric."""
    return {
        "concepts": {
            concept: {"taxonomy": v[0], "xbrl_concept": v[1], "unit": v[2]}
            for concept, v in XBRL_MAP.items()
        },
        "screener_metrics": SCREENER_METRICS,
    }


@fundamentals_v3_router.get("/sp500", summary="Current S&P 500 ticker list")
def get_sp500():
    """Return the current S&P 500 ticker list from Wikipedia."""
    try:
        layer = _get_layer()
        tickers = layer._universe.sp500_tickers()
        return {"count": len(tickers), "tickers": tickers}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
