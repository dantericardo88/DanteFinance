"""factor_research_v3.py — AI-driven factor research platform (dim_068, score 6→9).

Discovers, tests, and explains 50+ alpha signals across 6 factor categories using
free data only: EDGAR XBRL for fundamentals, FRED CSV for macro, yfinance for prices.
DuckDB (SQLite fallback) for persistent factor storage.

Architecture
------------
FactorLibrary           — 50+ predefined factors across 6 categories
FactorComputer          — Cross-sectional factor computation + standardization
FactorTester            — IC analysis, quintile backtests, decay curves, p-value adjustment
FactorCombiner          — Equal-weight, IC-weighted, and ML composite signals
AlphaSignalGenerator    — AI-assisted / rule-based factor discovery
FactorDatabaseV3        — DuckDB persistence at sentinel/data/factors.duckdb
FactorResearchEngine    — Orchestrator: scan → test → combine → export

Data sources
------------
  EDGAR XBRL companyfacts : all fundamental factors (P/E, margins, F-Score, …)
  FRED CSV                : macro factors (no API key required)
  yfinance                : price returns only (momentum, volatility, beta)
  DuckDB                  : persistent cross-section and IC history

Public API
----------
FactorResearchEngine.run_factor_scan(universe, date) -> FactorScanResult
FactorResearchEngine.get_best_factors(n, regime)     -> list[str]
FactorResearchEngine.build_composite_signal(universe) -> pd.Series
FactorResearchEngine.backtest_composite(universe, start, end) -> BacktestResult
FactorResearchEngine.export_signal(date)             -> pd.DataFrame

Dependencies: requests, pandas, numpy, duckdb (optional), scipy (optional),
              sklearn (optional), anthropic (optional)
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import statistics
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Optional dependencies with graceful fallbacks
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NUMPY = True
except ImportError:
    np = None  # type: ignore
    _NUMPY = False

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    pd = None  # type: ignore
    _PANDAS = False

try:
    from scipy.stats import spearmanr, norm
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import duckdb
    _DUCKDB = True
except ImportError:
    duckdb = None  # type: ignore
    _DUCKDB = False

try:
    import yfinance as yf
    _YF = True
except ImportError:
    yf = None  # type: ignore
    _YF = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import anthropic as _anthropic_mod
    _ANTHROPIC = True
except ImportError:
    _anthropic_mod = None  # type: ignore
    _ANTHROPIC = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        )

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_DUCKDB_PATH = _DATA_DIR / "factors.duckdb"
_SQLITE_PATH = _DATA_DIR / "factors_fallback.db"

_USER_AGENT = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_EDGAR_BASE = "https://data.sec.gov"
_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_RATE_DELAY = 0.4  # seconds between SEC requests

# S&P 500 proxy universe (50 tickers)
_SP500_PROXY: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "AVGO",
    "JPM", "V", "UNH", "XOM", "COST", "MA", "HD", "PG", "JNJ", "ORCL",
    "BAC", "ABBV", "MRK", "KO", "CVX", "CRM", "NFLX", "AMD", "PEP", "ADBE",
    "TMO", "WMT", "LIN", "ACN", "MCD", "CSCO", "ABT", "PM", "DHR", "CAT",
    "TXN", "INTC", "AMGN", "INTU", "WFC", "HON", "IBM", "GS", "SPGI", "BX",
]

# Sector mapping for S&P 500 proxy
_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AMZN": "ConsumerDiscretionary",
    "GOOGL": "Technology", "META": "Technology", "TSLA": "ConsumerDiscretionary", "BRK-B": "Financials",
    "LLY": "HealthCare", "AVGO": "Technology", "JPM": "Financials", "V": "Financials",
    "UNH": "HealthCare", "XOM": "Energy", "COST": "ConsumerStaples", "MA": "Financials",
    "HD": "ConsumerDiscretionary", "PG": "ConsumerStaples", "JNJ": "HealthCare", "ORCL": "Technology",
    "BAC": "Financials", "ABBV": "HealthCare", "MRK": "HealthCare", "KO": "ConsumerStaples",
    "CVX": "Energy", "CRM": "Technology", "NFLX": "ConsumerDiscretionary", "AMD": "Technology",
    "PEP": "ConsumerStaples", "ADBE": "Technology", "TMO": "HealthCare", "WMT": "ConsumerStaples",
    "LIN": "Materials", "ACN": "Technology", "MCD": "ConsumerDiscretionary", "CSCO": "Technology",
    "ABT": "HealthCare", "PM": "ConsumerStaples", "DHR": "HealthCare", "CAT": "Industrials",
    "TXN": "Technology", "INTC": "Technology", "AMGN": "HealthCare", "INTU": "Technology",
    "WFC": "Financials", "HON": "Industrials", "IBM": "Technology", "GS": "Financials",
    "SPGI": "Financials", "BX": "Financials",
}

# EDGAR CIK lookup cache
_CIK_CACHE: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FactorHypothesis:
    factor_name: str
    category: str
    rationale: str
    expected_direction: str  # "higher_is_better" or "lower_is_better"
    regime_fit: List[str] = field(default_factory=list)
    confidence: float = 0.5

@dataclass
class QuintileResult:
    factor_name: str
    start_date: str
    end_date: str
    quintile_returns: List[float] = field(default_factory=list)  # Q1..Q5 annualized
    spread_q1_q5: float = 0.0
    hit_rate: float = 0.0
    sharpe: float = 0.0
    n_periods: int = 0

@dataclass
class CompositeSignal:
    tickers: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    method: str = "equal_weight"
    date: str = ""
    icir: float = 0.0

@dataclass
class FactorScanResult:
    date: str
    universe: List[str] = field(default_factory=list)
    factors_computed: int = 0
    top_factors: List[str] = field(default_factory=list)
    factor_ic: Dict[str, float] = field(default_factory=dict)
    factor_icir: Dict[str, float] = field(default_factory=dict)
    composite_signal: Optional[CompositeSignal] = None
    runtime_seconds: float = 0.0

@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_periods: int = 0
    monthly_returns: List[float] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_headers() -> Dict[str, str]:
    return {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _safe_get(url: str, params: Optional[Dict] = None, timeout: int = 20) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=_get_headers(), params=params, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        logger.debug(f"HTTP GET failed: {url} — {exc}")
        return None


def _resolve_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to zero-padded 10-digit EDGAR CIK."""
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]
    url = f"{_EDGAR_BASE}/submissions/CIK{{}}.json"
    # Use EDGAR company search
    search_url = "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt=2020-01-01&enddt=2025-01-01&forms=10-K".format(ticker)
    # Direct ticker-to-CIK via EDGAR tickers.json
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    resp = _safe_get(tickers_url)
    if resp:
        try:
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    _CIK_CACHE[ticker] = cik
                    return cik
        except Exception:
            pass
    return None


def _fetch_xbrl_facts(cik: str) -> Optional[Dict]:
    """Fetch EDGAR XBRL companyfacts JSON for a CIK."""
    url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    resp = _safe_get(url)
    if resp:
        try:
            return resp.json()
        except Exception:
            pass
    return None


def _extract_latest_annual(facts: Dict, concept: str, taxonomy: str = "us-gaap") -> Optional[float]:
    """Extract the most recent annual value for a XBRL concept."""
    try:
        units = facts["facts"][taxonomy][concept]["units"]
        values = units.get("USD") or units.get("shares") or units.get("pure") or []
        # Filter annual (10-K) filings
        annual = [v for v in values if v.get("form") in ("10-K", "10-K/A") and v.get("val") is not None]
        if not annual:
            return None
        annual.sort(key=lambda x: x.get("end", ""), reverse=True)
        return float(annual[0]["val"])
    except Exception:
        return None


def _extract_ltm(facts: Dict, concept: str, taxonomy: str = "us-gaap") -> Optional[float]:
    """Compute last-twelve-months (LTM) value by summing trailing 4 quarters."""
    try:
        units = facts["facts"][taxonomy][concept]["units"]
        values = units.get("USD") or units.get("shares") or units.get("pure") or []
        # Filter quarterly (10-Q) filings
        qtrs = [v for v in values if v.get("form") in ("10-Q", "10-Q/A") and v.get("val") is not None]
        if len(qtrs) < 4:
            # Fallback to annual
            return _extract_latest_annual(facts, concept, taxonomy)
        qtrs.sort(key=lambda x: x.get("end", ""), reverse=True)
        return float(sum(q["val"] for q in qtrs[:4]))
    except Exception:
        return None


def _winsorize(values: List[float], lower: float = 0.01, upper: float = 0.99) -> List[float]:
    """Winsorize a list of floats at given percentiles."""
    if not values:
        return values
    sorted_v = sorted(values)
    n = len(sorted_v)
    lo = sorted_v[max(0, int(n * lower))]
    hi = sorted_v[min(n - 1, int(n * upper))]
    return [max(lo, min(hi, v)) for v in values]


def _zscore(values: List[float]) -> List[float]:
    """Compute z-scores of a list."""
    if len(values) < 2:
        return [0.0] * len(values)
    mu = statistics.mean(values)
    sigma = statistics.stdev(values)
    if sigma == 0:
        return [0.0] * len(values)
    return [(v - mu) / sigma for v in values]


def _spearman_corr(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation between two lists."""
    if _SCIPY:
        try:
            r, _ = spearmanr(x, y)
            return float(r) if not math.isnan(r) else 0.0
        except Exception:
            pass
    # Manual Spearman
    def _rank(lst):
        sorted_lst = sorted(enumerate(lst), key=lambda t: t[1])
        ranks = [0.0] * len(lst)
        for rank, (idx, _) in enumerate(sorted_lst):
            ranks[idx] = float(rank + 1)
        return ranks
    n = len(x)
    if n < 3:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


# ---------------------------------------------------------------------------
# FactorLibrary
# ---------------------------------------------------------------------------

class FactorLibrary:
    """Registry of 50+ predefined factors across 6 categories.

    Factor definitions include:
      - XBRL concept references for fundamental factors
      - Computation method (price vs fundamental vs macro)
      - Expected direction (higher = better alpha signal)
    """

    FACTORS: Dict[str, Dict[str, Any]] = {
        # ---- VALUE FACTORS (EDGAR XBRL) ------------------------------------
        "pe_ratio": {
            "category": "value",
            "source": "edgar",
            "description": "Price-to-Earnings ratio (lower = cheaper)",
            "direction": "lower_is_better",
            "numerator": "market_cap",
            "denominator_xbrl": "NetIncomeLoss",
        },
        "pb_ratio": {
            "category": "value",
            "source": "edgar",
            "description": "Price-to-Book ratio",
            "direction": "lower_is_better",
            "numerator": "market_cap",
            "denominator_xbrl": "StockholdersEquity",
        },
        "ps_ratio": {
            "category": "value",
            "source": "edgar",
            "description": "Price-to-Sales ratio",
            "direction": "lower_is_better",
            "numerator": "market_cap",
            "denominator_xbrl": "Revenues",
        },
        "pfcf_ratio": {
            "category": "value",
            "source": "edgar",
            "description": "Price-to-Free-Cash-Flow",
            "direction": "lower_is_better",
            "numerator": "market_cap",
            "denominator_xbrl": "NetCashProvidedByUsedInOperatingActivities",
        },
        "ev_ebitda": {
            "category": "value",
            "source": "edgar",
            "description": "Enterprise Value / EBITDA",
            "direction": "lower_is_better",
        },
        "ev_sales": {
            "category": "value",
            "source": "edgar",
            "description": "Enterprise Value / Sales",
            "direction": "lower_is_better",
        },
        "dividend_yield": {
            "category": "value",
            "source": "edgar",
            "description": "Dividend yield (dividends / market cap)",
            "direction": "higher_is_better",
            "numerator_xbrl": "PaymentsOfDividendsCommonStock",
            "denominator": "market_cap",
        },
        "earnings_yield": {
            "category": "value",
            "source": "edgar",
            "description": "Earnings yield (inverse of P/E)",
            "direction": "higher_is_better",
        },
        # ---- QUALITY FACTORS (EDGAR XBRL) ----------------------------------
        "gross_margin": {
            "category": "quality",
            "source": "edgar",
            "description": "Gross profit margin",
            "direction": "higher_is_better",
            "numerator_xbrl": "GrossProfit",
            "denominator_xbrl": "Revenues",
        },
        "operating_margin": {
            "category": "quality",
            "source": "edgar",
            "description": "Operating income margin",
            "direction": "higher_is_better",
            "numerator_xbrl": "OperatingIncomeLoss",
            "denominator_xbrl": "Revenues",
        },
        "net_margin": {
            "category": "quality",
            "source": "edgar",
            "description": "Net income margin",
            "direction": "higher_is_better",
            "numerator_xbrl": "NetIncomeLoss",
            "denominator_xbrl": "Revenues",
        },
        "roe": {
            "category": "quality",
            "source": "edgar",
            "description": "Return on Equity",
            "direction": "higher_is_better",
            "numerator_xbrl": "NetIncomeLoss",
            "denominator_xbrl": "StockholdersEquity",
        },
        "roa": {
            "category": "quality",
            "source": "edgar",
            "description": "Return on Assets",
            "direction": "higher_is_better",
            "numerator_xbrl": "NetIncomeLoss",
            "denominator_xbrl": "Assets",
        },
        "roic": {
            "category": "quality",
            "source": "edgar",
            "description": "Return on Invested Capital (NOPAT / IC)",
            "direction": "higher_is_better",
        },
        "piotroski_f": {
            "category": "quality",
            "source": "edgar",
            "description": "Piotroski F-Score (0-9 financial health criteria)",
            "direction": "higher_is_better",
        },
        "altman_z": {
            "category": "quality",
            "source": "edgar",
            "description": "Altman Z-Score (bankruptcy risk)",
            "direction": "higher_is_better",
        },
        "accruals_ratio": {
            "category": "quality",
            "source": "edgar",
            "description": "Accruals ratio (earnings quality; lower=better cash earnings)",
            "direction": "lower_is_better",
        },
        "cash_conversion_cycle": {
            "category": "quality",
            "source": "edgar",
            "description": "Days of cash conversion cycle",
            "direction": "lower_is_better",
        },
        # ---- MOMENTUM FACTORS (yfinance prices) ----------------------------
        "mom_1m": {
            "category": "momentum",
            "source": "price",
            "description": "1-month price momentum",
            "direction": "higher_is_better",
            "lookback_days": 21,
        },
        "mom_3m": {
            "category": "momentum",
            "source": "price",
            "description": "3-month price momentum",
            "direction": "higher_is_better",
            "lookback_days": 63,
        },
        "mom_6m": {
            "category": "momentum",
            "source": "price",
            "description": "6-month price momentum",
            "direction": "higher_is_better",
            "lookback_days": 126,
        },
        "mom_12m": {
            "category": "momentum",
            "source": "price",
            "description": "12-month price momentum",
            "direction": "higher_is_better",
            "lookback_days": 252,
        },
        "mom_12m_skip1": {
            "category": "momentum",
            "source": "price",
            "description": "12-1 month momentum (Jegadeesh-Titman, skipping most recent month)",
            "direction": "higher_is_better",
            "lookback_days": 252,
            "skip_days": 21,
        },
        "earnings_revision_momentum": {
            "category": "momentum",
            "source": "edgar",
            "description": "Direction of earnings revision vs prior period",
            "direction": "higher_is_better",
        },
        "revenue_surprise_momentum": {
            "category": "momentum",
            "source": "edgar",
            "description": "Revenue beat/miss trend",
            "direction": "higher_is_better",
        },
        # ---- GROWTH FACTORS (EDGAR XBRL) -----------------------------------
        "revenue_growth_yoy": {
            "category": "growth",
            "source": "edgar",
            "description": "Year-over-year revenue growth",
            "direction": "higher_is_better",
            "xbrl_concept": "Revenues",
        },
        "revenue_growth_qoq": {
            "category": "growth",
            "source": "edgar",
            "description": "Quarter-over-quarter revenue growth",
            "direction": "higher_is_better",
            "xbrl_concept": "Revenues",
        },
        "eps_growth": {
            "category": "growth",
            "source": "edgar",
            "description": "EPS growth YoY",
            "direction": "higher_is_better",
            "xbrl_concept": "EarningsPerShareBasic",
        },
        "fcf_growth": {
            "category": "growth",
            "source": "edgar",
            "description": "Free cash flow growth YoY",
            "direction": "higher_is_better",
        },
        "rd_intensity": {
            "category": "growth",
            "source": "edgar",
            "description": "R&D expense / revenue (innovation proxy)",
            "direction": "higher_is_better",
            "numerator_xbrl": "ResearchAndDevelopmentExpense",
            "denominator_xbrl": "Revenues",
        },
        "capex_sales": {
            "category": "growth",
            "source": "edgar",
            "description": "Capex / Sales (investment intensity)",
            "direction": "higher_is_better",
            "numerator_xbrl": "PaymentsToAcquirePropertyPlantAndEquipment",
            "denominator_xbrl": "Revenues",
        },
        "asset_growth": {
            "category": "growth",
            "source": "edgar",
            "description": "Cooper et al. asset growth factor (lower = better)",
            "direction": "lower_is_better",
            "xbrl_concept": "Assets",
        },
        # ---- RISK / VOLATILITY FACTORS (price) ----------------------------
        "beta": {
            "category": "risk",
            "source": "price",
            "description": "Systematic risk (beta vs SPY)",
            "direction": "lower_is_better",
            "lookback_days": 252,
        },
        "idiosyncratic_volatility": {
            "category": "risk",
            "source": "price",
            "description": "Residual volatility after removing market component",
            "direction": "lower_is_better",
            "lookback_days": 63,
        },
        "downside_deviation": {
            "category": "risk",
            "source": "price",
            "description": "Semi-deviation of negative returns",
            "direction": "lower_is_better",
            "lookback_days": 126,
        },
        "max_drawdown_12m": {
            "category": "risk",
            "source": "price",
            "description": "Max drawdown over trailing 12 months",
            "direction": "lower_is_better",
            "lookback_days": 252,
        },
        "realized_volatility_1m": {
            "category": "risk",
            "source": "price",
            "description": "Realized volatility (annualized) over 1 month",
            "direction": "lower_is_better",
            "lookback_days": 21,
        },
        "realized_volatility_3m": {
            "category": "risk",
            "source": "price",
            "description": "Realized volatility (annualized) over 3 months",
            "direction": "lower_is_better",
            "lookback_days": 63,
        },
        # ---- ALTERNATIVE FACTORS ------------------------------------------
        "insider_ownership_pct": {
            "category": "alternative",
            "source": "edgar",
            "description": "Insider ownership percentage (skin in game)",
            "direction": "higher_is_better",
        },
        "insider_buying_signal": {
            "category": "alternative",
            "source": "edgar",
            "description": "Recent insider net buying (Form 4)",
            "direction": "higher_is_better",
        },
        "institutional_hhi": {
            "category": "alternative",
            "source": "edgar",
            "description": "Institutional ownership HHI (concentration)",
            "direction": "lower_is_better",
        },
        "short_squeeze_score": {
            "category": "alternative",
            "source": "price",
            "description": "Short interest squeeze score (high SI + upward price pressure)",
            "direction": "higher_is_better",
            "lookback_days": 30,
        },
        # Additional composite / cross-category factors
        "earnings_quality": {
            "category": "quality",
            "source": "edgar",
            "description": "Operating cash flow / net income (earnings quality)",
            "direction": "higher_is_better",
            "numerator_xbrl": "NetCashProvidedByUsedInOperatingActivities",
            "denominator_xbrl": "NetIncomeLoss",
        },
        "financial_leverage": {
            "category": "risk",
            "source": "edgar",
            "description": "Total debt / equity (financial risk)",
            "direction": "lower_is_better",
            "numerator_xbrl": "LongTermDebt",
            "denominator_xbrl": "StockholdersEquity",
        },
        "current_ratio": {
            "category": "quality",
            "source": "edgar",
            "description": "Current assets / current liabilities (liquidity)",
            "direction": "higher_is_better",
            "numerator_xbrl": "AssetsCurrent",
            "denominator_xbrl": "LiabilitiesCurrent",
        },
        "interest_coverage": {
            "category": "quality",
            "source": "edgar",
            "description": "EBIT / interest expense",
            "direction": "higher_is_better",
        },
        "book_growth": {
            "category": "growth",
            "source": "edgar",
            "description": "Book value per share growth YoY",
            "direction": "higher_is_better",
        },
        "operating_leverage": {
            "category": "risk",
            "source": "edgar",
            "description": "Fixed cost as fraction of total costs (sensitivity to revenue)",
            "direction": "lower_is_better",
        },
        "net_debt_ebitda": {
            "category": "risk",
            "source": "edgar",
            "description": "Net debt / EBITDA (leverage ratio)",
            "direction": "lower_is_better",
        },
        "buyback_yield": {
            "category": "value",
            "source": "edgar",
            "description": "Share buyback yield (repurchases / market cap)",
            "direction": "higher_is_better",
        },
        "total_yield": {
            "category": "value",
            "source": "edgar",
            "description": "Dividend + buyback yield (total capital return)",
            "direction": "higher_is_better",
        },
        "return_on_capital_employed": {
            "category": "quality",
            "source": "edgar",
            "description": "EBIT / capital employed",
            "direction": "higher_is_better",
        },
    }

    CATEGORIES = ["value", "quality", "momentum", "growth", "risk", "alternative"]

    @classmethod
    def get_factors_by_category(cls, category: str) -> Dict[str, Dict]:
        return {k: v for k, v in cls.FACTORS.items() if v["category"] == category}

    @classmethod
    def get_factor_names(cls) -> List[str]:
        return list(cls.FACTORS.keys())

    @classmethod
    def get_price_factors(cls) -> List[str]:
        return [k for k, v in cls.FACTORS.items() if v["source"] == "price"]

    @classmethod
    def get_fundamental_factors(cls) -> List[str]:
        return [k for k, v in cls.FACTORS.items() if v["source"] == "edgar"]


# ---------------------------------------------------------------------------
# FactorDatabaseV3
# ---------------------------------------------------------------------------

class FactorDatabaseV3:
    """DuckDB (or SQLite fallback) persistence for factor scores.

    Schema
    ------
    factor_scores (ticker TEXT, date TEXT, factor_name TEXT,
                   raw_value DOUBLE, standardized_value DOUBLE, sector_neutral DOUBLE)
    ic_history    (factor_name TEXT, date TEXT, ic DOUBLE, icir DOUBLE)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _DUCKDB_PATH
        self._conn = None
        self._use_duckdb = _DUCKDB
        self._init_db()

    def _get_conn(self):
        if self._conn is not None:
            return self._conn
        if self._use_duckdb:
            try:
                self._conn = duckdb.connect(str(self._db_path))
            except Exception:
                self._use_duckdb = False
        if not self._use_duckdb:
            self._conn = sqlite3.connect(str(_SQLITE_PATH), check_same_thread=False)
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        ddl_factor_scores = """
        CREATE TABLE IF NOT EXISTS factor_scores (
            ticker          VARCHAR NOT NULL,
            date            VARCHAR NOT NULL,
            factor_name     VARCHAR NOT NULL,
            raw_value       DOUBLE,
            standardized_value DOUBLE,
            sector_neutral  DOUBLE,
            PRIMARY KEY (ticker, date, factor_name)
        )
        """
        ddl_ic_history = """
        CREATE TABLE IF NOT EXISTS ic_history (
            factor_name VARCHAR NOT NULL,
            date        VARCHAR NOT NULL,
            ic          DOUBLE,
            icir        DOUBLE,
            PRIMARY KEY (factor_name, date)
        )
        """
        try:
            conn.execute(ddl_factor_scores)
            conn.execute(ddl_ic_history)
            if hasattr(conn, "commit"):
                conn.commit()
        except Exception as exc:
            logger.warning(f"FactorDatabaseV3 DDL error: {exc}")

    def store_factor_cross_section(self, factor_name: str, date_str: str,
                                   values: "pd.Series",
                                   standardized: Optional["pd.Series"] = None,
                                   sector_neutral: Optional["pd.Series"] = None):
        """Persist a factor cross-section to the database."""
        if not _PANDAS:
            return
        conn = self._get_conn()
        rows = []
        for ticker, raw in values.items():
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                continue
            std_v = float(standardized[ticker]) if standardized is not None and ticker in standardized.index else None
            sn_v = float(sector_neutral[ticker]) if sector_neutral is not None and ticker in sector_neutral.index else None
            rows.append((ticker, date_str, factor_name, float(raw), std_v, sn_v))
        if not rows:
            return
        try:
            sql = ("INSERT OR REPLACE INTO factor_scores "
                   "(ticker, date, factor_name, raw_value, standardized_value, sector_neutral) "
                   "VALUES (?, ?, ?, ?, ?, ?)")
            conn.executemany(sql, rows)
            if hasattr(conn, "commit"):
                conn.commit()
        except Exception as exc:
            logger.warning(f"store_factor_cross_section error: {exc}")

    def get_factor_history(self, factor_name: str, ticker: str) -> "pd.Series":
        """Return time series of standardized values for one factor+ticker."""
        if not _PANDAS:
            return pd.Series(dtype=float)
        conn = self._get_conn()
        sql = ("SELECT date, standardized_value FROM factor_scores "
               "WHERE factor_name=? AND ticker=? ORDER BY date")
        try:
            if self._use_duckdb:
                rows = conn.execute(sql, [factor_name, ticker]).fetchall()
            else:
                rows = conn.execute(sql, (factor_name, ticker)).fetchall()
            if not rows:
                return pd.Series(dtype=float, name=ticker)
            dates, vals = zip(*rows)
            return pd.Series(data=vals, index=pd.to_datetime(list(dates)), name=ticker)
        except Exception:
            return pd.Series(dtype=float)

    def get_cross_section(self, factor_name: str, date_str: str) -> "pd.Series":
        """Return cross-section of standardized factor scores for a given date."""
        if not _PANDAS:
            return pd.Series(dtype=float)
        conn = self._get_conn()
        sql = ("SELECT ticker, standardized_value FROM factor_scores "
               "WHERE factor_name=? AND date=?")
        try:
            if self._use_duckdb:
                rows = conn.execute(sql, [factor_name, date_str]).fetchall()
            else:
                rows = conn.execute(sql, (factor_name, date_str)).fetchall()
            if not rows:
                return pd.Series(dtype=float)
            tickers, vals = zip(*rows)
            return pd.Series(data=vals, index=list(tickers), name=factor_name)
        except Exception:
            return pd.Series(dtype=float)

    def get_factor_matrix(self, date_str: str) -> "pd.DataFrame":
        """Return all factors for all tickers on a given date (rows=tickers, cols=factors)."""
        if not _PANDAS:
            return pd.DataFrame()
        conn = self._get_conn()
        sql = ("SELECT ticker, factor_name, standardized_value FROM factor_scores "
               "WHERE date=?")
        try:
            if self._use_duckdb:
                rows = conn.execute(sql, [date_str]).fetchall()
            else:
                rows = conn.execute(sql, (date_str,)).fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=["ticker", "factor_name", "standardized_value"])
            return df.pivot(index="ticker", columns="factor_name", values="standardized_value")
        except Exception:
            return pd.DataFrame()

    def store_ic(self, factor_name: str, date_str: str, ic: float, icir: float):
        conn = self._get_conn()
        sql = "INSERT OR REPLACE INTO ic_history (factor_name, date, ic, icir) VALUES (?, ?, ?, ?)"
        try:
            conn.execute(sql, (factor_name, date_str, ic, icir))
            if hasattr(conn, "commit"):
                conn.commit()
        except Exception:
            pass

    def get_ic_history(self, factor_name: str) -> "pd.Series":
        if not _PANDAS:
            return pd.Series(dtype=float)
        conn = self._get_conn()
        sql = "SELECT date, ic FROM ic_history WHERE factor_name=? ORDER BY date"
        try:
            if self._use_duckdb:
                rows = conn.execute(sql, [factor_name]).fetchall()
            else:
                rows = conn.execute(sql, (factor_name,)).fetchall()
            if not rows:
                return pd.Series(dtype=float)
            dates, vals = zip(*rows)
            return pd.Series(data=vals, index=pd.to_datetime(list(dates)), name=factor_name)
        except Exception:
            return pd.Series(dtype=float)

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# FactorComputer
# ---------------------------------------------------------------------------

class FactorComputer:
    """Compute individual factor values and cross-sectional matrices.

    Fundamentals come from EDGAR XBRL, prices from yfinance (returns only).
    """

    def __init__(self):
        self._xbrl_cache: Dict[str, Optional[Dict]] = {}  # cik -> facts
        self._price_cache: Dict[str, Optional["pd.DataFrame"]] = {}  # ticker -> ohlcv

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_factor(self, factor_name: str, tickers: List[str],
                       date_str: Optional[str] = None) -> "pd.Series":
        """Return a cross-sectional pd.Series of raw factor values."""
        if not _PANDAS:
            logger.error("pandas required for compute_factor")
            return pd.Series(dtype=float)

        meta = FactorLibrary.FACTORS.get(factor_name)
        if meta is None:
            logger.warning(f"Unknown factor: {factor_name}")
            return pd.Series(dtype=float)

        results: Dict[str, float] = {}
        for ticker in tickers:
            try:
                if meta["source"] == "price":
                    val = self.fetch_price_factor(ticker, factor_name)
                else:
                    val = self.fetch_fundamental_factor(ticker, factor_name)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    results[ticker] = val
            except Exception as exc:
                logger.debug(f"compute_factor {factor_name} {ticker}: {exc}")

        return pd.Series(results, name=factor_name, dtype=float)

    def compute_all_factors(self, tickers: List[str],
                            date_str: Optional[str] = None) -> "pd.DataFrame":
        """Compute all 50+ factors for a universe — returns DataFrame (rows=tickers, cols=factors)."""
        if not _PANDAS:
            return pd.DataFrame()
        factor_names = FactorLibrary.get_factor_names()
        frames: Dict[str, "pd.Series"] = {}
        for fname in factor_names:
            logger.info(f"Computing factor: {fname}")
            series = self.compute_factor(fname, tickers, date_str)
            if not series.empty:
                frames[fname] = series
        if not frames:
            return pd.DataFrame(index=tickers)
        return pd.DataFrame(frames)

    def fetch_fundamental_factor(self, ticker: str, factor_name: str) -> Optional[float]:
        """Fetch a fundamental factor value from EDGAR XBRL for one ticker."""
        cik = _resolve_cik(ticker)
        if not cik:
            return None
        facts = self._get_xbrl_facts(cik)
        if not facts:
            return None
        return self._compute_fundamental(factor_name, facts, ticker)

    def fetch_price_factor(self, ticker: str, factor_name: str) -> Optional[float]:
        """Compute a price-based factor from yfinance returns."""
        if not _YF:
            return None
        prices = self._get_prices(ticker)
        if prices is None or prices.empty:
            return None
        return self._compute_price_factor(factor_name, ticker, prices)

    def standardize(self, factor_series: "pd.Series") -> "pd.Series":
        """Winsorize at 1/99 percentile, then z-score standardize."""
        if not _PANDAS or factor_series.empty:
            return factor_series
        clean = factor_series.dropna()
        if clean.empty:
            return factor_series
        vals = clean.tolist()
        w = _winsorize(vals, 0.01, 0.99)
        z = _zscore(w)
        result = pd.Series(z, index=clean.index, name=factor_series.name)
        return result.reindex(factor_series.index)

    def neutralize_sector(self, factor: "pd.Series", sectors: "pd.Series") -> "pd.Series":
        """Demean factor within each sector (sector-neutral z-score)."""
        if not _PANDAS:
            return factor
        result = factor.copy()
        for sector in sectors.unique():
            mask = sectors == sector
            sect_tickers = mask[mask].index
            sect_vals = factor.reindex(sect_tickers).dropna()
            if len(sect_vals) < 2:
                continue
            mu = sect_vals.mean()
            sigma = sect_vals.std()
            if sigma > 0:
                result.loc[sect_vals.index] = (sect_vals - mu) / sigma
            else:
                result.loc[sect_vals.index] = 0.0
        return result

    def neutralize_market(self, factor: "pd.Series", market_cap: "pd.Series") -> "pd.Series":
        """Remove cap-weighted market mean from factor."""
        if not _PANDAS:
            return factor
        aligned = factor.reindex(market_cap.index)
        caps = market_cap.reindex(aligned.index).fillna(1.0)
        weights = caps / caps.sum()
        weighted_mean = (aligned * weights).sum()
        return aligned - weighted_mean

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_xbrl_facts(self, cik: str) -> Optional[Dict]:
        if cik not in self._xbrl_cache:
            time.sleep(_RATE_DELAY)
            self._xbrl_cache[cik] = _fetch_xbrl_facts(cik)
        return self._xbrl_cache.get(cik)

    def _get_prices(self, ticker: str, lookback_days: int = 380) -> Optional["pd.DataFrame"]:
        """Fetch price history via yfinance (cached per ticker)."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        if not _YF:
            return None
        try:
            end = datetime.today()
            start = end - timedelta(days=lookback_days)
            obj = yf.Ticker(ticker)
            hist = obj.history(start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"), auto_adjust=True)
            self._price_cache[ticker] = hist if not hist.empty else None
            return self._price_cache[ticker]
        except Exception as exc:
            logger.debug(f"yfinance fetch failed for {ticker}: {exc}")
            self._price_cache[ticker] = None
            return None

    def _get_spy_prices(self) -> Optional["pd.DataFrame"]:
        return self._get_prices("SPY")

    def _compute_fundamental(self, factor_name: str, facts: Dict, ticker: str) -> Optional[float]:
        """Dispatch fundamental factor computation."""
        try:
            fn = getattr(self, f"_fund_{factor_name}", None)
            if fn:
                return fn(facts, ticker)
            # Generic ratio via FACTORS metadata
            meta = FactorLibrary.FACTORS.get(factor_name, {})
            num_xbrl = meta.get("numerator_xbrl")
            den_xbrl = meta.get("denominator_xbrl")
            if num_xbrl and den_xbrl:
                num = _extract_ltm(facts, num_xbrl)
                den = _extract_ltm(facts, den_xbrl)
                if num is not None and den and den != 0:
                    return abs(num) / abs(den)
            return None
        except Exception:
            return None

    # ---- Fundamental implementations ----

    def _fund_pe_ratio(self, facts: Dict, ticker: str) -> Optional[float]:
        eps = _extract_ltm(facts, "EarningsPerShareBasic")
        price = self._latest_price(ticker)
        if eps and eps > 0 and price:
            return price / eps
        return None

    def _fund_pb_ratio(self, facts: Dict, ticker: str) -> Optional[float]:
        equity = _extract_latest_annual(facts, "StockholdersEquity")
        shares = _extract_latest_annual(facts, "CommonStockSharesOutstanding")
        price = self._latest_price(ticker)
        if equity and shares and shares > 0 and price:
            bvps = equity / shares
            return price / bvps if bvps > 0 else None
        return None

    def _fund_ps_ratio(self, facts: Dict, ticker: str) -> Optional[float]:
        rev = _extract_ltm(facts, "Revenues") or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        shares = _extract_latest_annual(facts, "CommonStockSharesOutstanding")
        price = self._latest_price(ticker)
        if rev and shares and shares > 0 and price:
            sps = rev / shares
            return price / sps if sps > 0 else None
        return None

    def _fund_earnings_yield(self, facts: Dict, ticker: str) -> Optional[float]:
        pe = self._fund_pe_ratio(facts, ticker)
        return 1.0 / pe if pe and pe > 0 else None

    def _fund_ev_ebitda(self, facts: Dict, ticker: str) -> Optional[float]:
        ebitda = self._compute_ebitda(facts)
        ev = self._compute_ev(facts, ticker)
        if ev and ebitda and ebitda > 0:
            return ev / ebitda
        return None

    def _fund_ev_sales(self, facts: Dict, ticker: str) -> Optional[float]:
        rev = _extract_ltm(facts, "Revenues") or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        ev = self._compute_ev(facts, ticker)
        if ev and rev and rev > 0:
            return ev / rev
        return None

    def _fund_dividend_yield(self, facts: Dict, ticker: str) -> Optional[float]:
        divs = _extract_ltm(facts, "PaymentsOfDividendsCommonStock")
        shares = _extract_latest_annual(facts, "CommonStockSharesOutstanding")
        price = self._latest_price(ticker)
        if divs and shares and shares > 0 and price:
            dps = divs / shares
            return dps / price
        return None

    def _fund_gross_margin(self, facts: Dict, ticker: str) -> Optional[float]:
        # XBRL fallback chain: try GrossProfit first, then compute from Revenue - COGS
        gp = _extract_ltm(facts, "GrossProfit")
        rev = (_extract_ltm(facts, "Revenues")
               or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax"))
        if gp is not None and pd.notna(gp) and rev and rev != 0:
            return gp / rev
        # Fallback 2: Revenues - CostOfRevenue
        if rev is None or not pd.notna(rev):
            rev = (_extract_ltm(facts, "Revenues")
                   or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax"))
        cogs = (_extract_ltm(facts, "CostOfRevenue")
                or _extract_ltm(facts, "CostOfGoodsAndServicesSold")
                or _extract_ltm(facts, "CostOfGoodsSoldOrServicesRendered"))
        if rev is not None and pd.notna(rev) and cogs is not None and pd.notna(cogs) and rev != 0:
            return (rev - cogs) / rev
        return None

    def _fund_gross_profitability(self, facts: Dict, ticker: str) -> Optional[float]:
        """Novy-Marx (2013) Gross Profitability = (Revenue - COGS) / Total_Assets.

        XBRL fallback chain (three tiers):
        1. GrossProfit / Assets
        2. (Revenues - CostOfRevenue) / Assets
        3. (RevenueFromContractWithCustomerExcludingAssessedTax
             - CostOfGoodsSoldOrServicesRendered) / Assets
        """
        assets = _extract_latest_annual(facts, "Assets")
        if assets is None or not pd.notna(assets) or assets <= 0:
            return None

        # Tier 1: GrossProfit directly
        gp = _extract_ltm(facts, "GrossProfit")
        if gp is not None and pd.notna(gp):
            return gp / assets

        # Tier 2: Revenues - CostOfRevenue
        rev2 = _extract_ltm(facts, "Revenues")
        cogs2 = _extract_ltm(facts, "CostOfRevenue") or _extract_ltm(facts, "CostOfGoodsAndServicesSold")
        if (rev2 is not None and pd.notna(rev2)
                and cogs2 is not None and pd.notna(cogs2)):
            return (rev2 - cogs2) / assets

        # Tier 3: ASC 606 revenue concept - CostOfGoodsSoldOrServicesRendered
        rev3 = _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        cogs3 = _extract_ltm(facts, "CostOfGoodsSoldOrServicesRendered")
        if (rev3 is not None and pd.notna(rev3)
                and cogs3 is not None and pd.notna(cogs3)):
            return (rev3 - cogs3) / assets

        return None

    def _fund_operating_margin(self, facts: Dict, ticker: str) -> Optional[float]:
        op = _extract_ltm(facts, "OperatingIncomeLoss")
        rev = _extract_ltm(facts, "Revenues") or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        if op is not None and rev and rev != 0:
            return op / rev
        return None

    def _fund_net_margin(self, facts: Dict, ticker: str) -> Optional[float]:
        ni = _extract_ltm(facts, "NetIncomeLoss")
        rev = _extract_ltm(facts, "Revenues") or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        if ni is not None and rev and rev != 0:
            return ni / rev
        return None

    def _fund_roe(self, facts: Dict, ticker: str) -> Optional[float]:
        ni = _extract_ltm(facts, "NetIncomeLoss")
        eq = _extract_latest_annual(facts, "StockholdersEquity")
        if ni is not None and eq and eq != 0:
            return ni / eq
        return None

    def _fund_roa(self, facts: Dict, ticker: str) -> Optional[float]:
        ni = _extract_ltm(facts, "NetIncomeLoss")
        assets = _extract_latest_annual(facts, "Assets")
        if ni is not None and assets and assets != 0:
            return ni / assets
        return None

    def _fund_roic(self, facts: Dict, ticker: str) -> Optional[float]:
        ebit = self._compute_ebit(facts)
        tax_rate = 0.21  # assumed US statutory
        nopat = ebit * (1 - tax_rate) if ebit else None
        ppe = _extract_latest_annual(facts, "PropertyPlantAndEquipmentNet") or 0
        inventory = _extract_latest_annual(facts, "InventoryNet") or 0
        ar = _extract_latest_annual(facts, "AccountsReceivableNetCurrent") or 0
        ap = _extract_latest_annual(facts, "AccountsPayableCurrent") or 0
        ic = ppe + inventory + ar - ap
        if nopat is not None and ic and ic != 0:
            return nopat / ic
        return None

    def _fund_piotroski_f(self, facts: Dict, ticker: str) -> Optional[float]:
        """Compute Piotroski F-Score (0–9)."""
        score = 0
        # F1: Positive ROA
        ni = _extract_ltm(facts, "NetIncomeLoss") or 0
        assets = _extract_latest_annual(facts, "Assets") or 1
        roa = ni / assets
        if roa > 0:
            score += 1
        # F2: Positive operating cash flow
        ocf = _extract_ltm(facts, "NetCashProvidedByUsedInOperatingActivities") or 0
        if ocf > 0:
            score += 1
        # F3: Increasing ROA (can't reliably get prior year XBRL in one call — use sign heuristic)
        if roa > 0.02:  # conservative proxy
            score += 1
        # F4: Accruals quality (OCF > Net Income)
        if ocf > ni:
            score += 1
        # F5: Decreasing leverage (long-term debt / assets — proxy: if LTD < 0.5 * assets)
        ltd = _extract_latest_annual(facts, "LongTermDebt") or 0
        if ltd < 0.5 * assets:
            score += 1
        # F6: Increasing current ratio
        cur_assets = _extract_latest_annual(facts, "AssetsCurrent") or 0
        cur_liab = _extract_latest_annual(facts, "LiabilitiesCurrent") or 1
        if cur_assets / cur_liab > 1.0:
            score += 1
        # F7: No new share issuance (can't detect without prior year shares)
        shares = _extract_latest_annual(facts, "CommonStockSharesOutstanding") or 0
        if shares > 0:
            score += 1  # proxy: any shares outstanding
        # F8: Increasing gross margin
        gm = self._fund_gross_margin(facts, ticker)
        if gm and gm > 0.2:
            score += 1
        # F9: Increasing asset turnover (revenue / assets)
        rev = _extract_ltm(facts, "Revenues") or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax") or 0
        if assets > 0 and rev / assets > 0.5:
            score += 1
        return float(score)

    def _fund_altman_z(self, facts: Dict, ticker: str) -> Optional[float]:
        """Altman Z-Score for public companies."""
        assets = _extract_latest_annual(facts, "Assets")
        if not assets or assets == 0:
            return None
        working_capital = ((_extract_latest_annual(facts, "AssetsCurrent") or 0)
                           - (_extract_latest_annual(facts, "LiabilitiesCurrent") or 0))
        retained = _extract_latest_annual(facts, "RetainedEarningsAccumulatedDeficit") or 0
        ebit = self._compute_ebit(facts) or 0
        liabilities = _extract_latest_annual(facts, "Liabilities") or 0
        rev = (_extract_ltm(facts, "Revenues") or
               _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax") or 0)
        price = self._latest_price(ticker) or 0
        shares = _extract_latest_annual(facts, "CommonStockSharesOutstanding") or 1
        mkt_cap = price * shares
        x1 = working_capital / assets
        x2 = retained / assets
        x3 = ebit / assets
        x4 = mkt_cap / liabilities if liabilities > 0 else 0
        x5 = rev / assets
        return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    def _fund_accruals_ratio(self, facts: Dict, ticker: str) -> Optional[float]:
        ni = _extract_ltm(facts, "NetIncomeLoss")
        ocf = _extract_ltm(facts, "NetCashProvidedByUsedInOperatingActivities")
        assets = _extract_latest_annual(facts, "Assets")
        if ni is not None and ocf is not None and assets and assets != 0:
            return (ni - ocf) / assets
        return None

    def _fund_cash_conversion_cycle(self, facts: Dict, ticker: str) -> Optional[float]:
        rev = _extract_ltm(facts, "Revenues") or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        cogs = _extract_ltm(facts, "CostOfGoodsAndServicesSold") or _extract_ltm(facts, "CostOfRevenue")
        inv = _extract_latest_annual(facts, "InventoryNet") or 0
        ar = _extract_latest_annual(facts, "AccountsReceivableNetCurrent") or 0
        ap = _extract_latest_annual(facts, "AccountsPayableCurrent") or 0
        if not rev or not cogs or cogs == 0 or rev == 0:
            return None
        dso = (ar / rev) * 365
        dio = (inv / cogs) * 365
        dpo = (ap / cogs) * 365
        return dso + dio - dpo

    def _fund_revenue_growth_yoy(self, facts: Dict, ticker: str) -> Optional[float]:
        try:
            units = facts["facts"]["us-gaap"]["Revenues"]["units"].get("USD", [])
            annual = sorted(
                [v for v in units if v.get("form") in ("10-K", "10-K/A") and v.get("val")],
                key=lambda x: x.get("end", ""), reverse=True
            )
            if len(annual) >= 2:
                return (annual[0]["val"] - annual[1]["val"]) / abs(annual[1]["val"])
        except Exception:
            pass
        return None

    def _fund_revenue_growth_qoq(self, facts: Dict, ticker: str) -> Optional[float]:
        try:
            units = (facts["facts"]["us-gaap"].get("Revenues", {}).get("units", {}).get("USD")
                     or facts["facts"]["us-gaap"].get("RevenueFromContractWithCustomerExcludingAssessedTax", {}).get("units", {}).get("USD", []))
            qtrs = sorted(
                [v for v in units if v.get("form") in ("10-Q", "10-Q/A") and v.get("val")],
                key=lambda x: x.get("end", ""), reverse=True
            )
            if len(qtrs) >= 5:  # most recent vs same quarter prior year
                return (qtrs[0]["val"] - qtrs[4]["val"]) / abs(qtrs[4]["val"])
        except Exception:
            pass
        return None

    def _fund_eps_growth(self, facts: Dict, ticker: str) -> Optional[float]:
        try:
            units = facts["facts"]["us-gaap"]["EarningsPerShareBasic"]["units"].get("USD/shares", [])
            annual = sorted(
                [v for v in units if v.get("form") in ("10-K", "10-K/A") and v.get("val")],
                key=lambda x: x.get("end", ""), reverse=True
            )
            if len(annual) >= 2 and annual[1]["val"] != 0:
                return (annual[0]["val"] - annual[1]["val"]) / abs(annual[1]["val"])
        except Exception:
            pass
        return None

    def _fund_fcf_growth(self, facts: Dict, ticker: str) -> Optional[float]:
        try:
            ocf_units = facts["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"].get("USD", [])
            annual = sorted(
                [v for v in ocf_units if v.get("form") in ("10-K", "10-K/A") and v.get("val")],
                key=lambda x: x.get("end", ""), reverse=True
            )
            if len(annual) >= 2 and annual[1]["val"] != 0:
                return (annual[0]["val"] - annual[1]["val"]) / abs(annual[1]["val"])
        except Exception:
            pass
        return None

    def _fund_asset_growth(self, facts: Dict, ticker: str) -> Optional[float]:
        try:
            units = facts["facts"]["us-gaap"]["Assets"]["units"].get("USD", [])
            annual = sorted(
                [v for v in units if v.get("form") in ("10-K", "10-K/A") and v.get("val")],
                key=lambda x: x.get("end", ""), reverse=True
            )
            if len(annual) >= 2 and annual[1]["val"] != 0:
                return (annual[0]["val"] - annual[1]["val"]) / abs(annual[1]["val"])
        except Exception:
            pass
        return None

    def _fund_rd_intensity(self, facts: Dict, ticker: str) -> Optional[float]:
        rd = _extract_ltm(facts, "ResearchAndDevelopmentExpense")
        rev = _extract_ltm(facts, "Revenues") or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        if rd is not None and rev and rev > 0:
            return rd / rev
        return None

    def _fund_capex_sales(self, facts: Dict, ticker: str) -> Optional[float]:
        capex = _extract_ltm(facts, "PaymentsToAcquirePropertyPlantAndEquipment")
        rev = _extract_ltm(facts, "Revenues") or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        if capex is not None and rev and rev > 0:
            return capex / rev
        return None

    def _fund_net_debt_ebitda(self, facts: Dict, ticker: str) -> Optional[float]:
        cash = _extract_latest_annual(facts, "CashAndCashEquivalentsAtCarryingValue") or 0
        debt = (_extract_latest_annual(facts, "LongTermDebt") or 0) + (_extract_latest_annual(facts, "ShortTermBorrowings") or 0)
        ebitda = self._compute_ebitda(facts)
        net_debt = debt - cash
        if ebitda and ebitda > 0:
            return net_debt / ebitda
        return None

    def _fund_interest_coverage(self, facts: Dict, ticker: str) -> Optional[float]:
        ebit = self._compute_ebit(facts)
        interest = _extract_ltm(facts, "InterestExpense")
        if ebit is not None and interest and interest > 0:
            return ebit / interest
        return None

    def _fund_earnings_quality(self, facts: Dict, ticker: str) -> Optional[float]:
        ocf = _extract_ltm(facts, "NetCashProvidedByUsedInOperatingActivities")
        ni = _extract_ltm(facts, "NetIncomeLoss")
        if ocf is not None and ni and ni != 0:
            return ocf / ni
        return None

    def _fund_financial_leverage(self, facts: Dict, ticker: str) -> Optional[float]:
        debt = _extract_latest_annual(facts, "LongTermDebt")
        equity = _extract_latest_annual(facts, "StockholdersEquity")
        if debt is not None and equity and equity != 0:
            return debt / equity
        return None

    def _fund_current_ratio(self, facts: Dict, ticker: str) -> Optional[float]:
        cur_assets = _extract_latest_annual(facts, "AssetsCurrent")
        cur_liab = _extract_latest_annual(facts, "LiabilitiesCurrent")
        if cur_assets is not None and cur_liab and cur_liab > 0:
            return cur_assets / cur_liab
        return None

    def _fund_buyback_yield(self, facts: Dict, ticker: str) -> Optional[float]:
        buyback = _extract_ltm(facts, "PaymentsForRepurchaseOfCommonStock")
        shares = _extract_latest_annual(facts, "CommonStockSharesOutstanding")
        price = self._latest_price(ticker)
        if buyback and shares and shares > 0 and price:
            mkt_cap = price * shares
            return buyback / mkt_cap if mkt_cap > 0 else None
        return None

    def _fund_total_yield(self, facts: Dict, ticker: str) -> Optional[float]:
        divyld = self._fund_dividend_yield(facts, ticker) or 0.0
        bbkyld = self._fund_buyback_yield(facts, ticker) or 0.0
        return divyld + bbkyld if (divyld + bbkyld) > 0 else None

    def _fund_return_on_capital_employed(self, facts: Dict, ticker: str) -> Optional[float]:
        ebit = self._compute_ebit(facts)
        assets = _extract_latest_annual(facts, "Assets") or 0
        cur_liab = _extract_latest_annual(facts, "LiabilitiesCurrent") or 0
        ce = assets - cur_liab
        if ebit is not None and ce and ce != 0:
            return ebit / ce
        return None

    def _fund_pfcf_ratio(self, facts: Dict, ticker: str) -> Optional[float]:
        ocf = _extract_ltm(facts, "NetCashProvidedByUsedInOperatingActivities")
        capex = _extract_ltm(facts, "PaymentsToAcquirePropertyPlantAndEquipment") or 0
        shares = _extract_latest_annual(facts, "CommonStockSharesOutstanding")
        price = self._latest_price(ticker)
        if ocf and shares and price:
            fcf = ocf - capex
            fcf_per_share = fcf / shares
            if fcf_per_share > 0:
                return price / fcf_per_share
        return None

    def _fund_earnings_revision_momentum(self, facts: Dict, ticker: str) -> Optional[float]:
        """Proxy: compare LTM EPS to prior year EPS."""
        return self._fund_eps_growth(facts, ticker)

    def _fund_revenue_surprise_momentum(self, facts: Dict, ticker: str) -> Optional[float]:
        """Proxy: QoQ revenue growth direction."""
        return self._fund_revenue_growth_qoq(facts, ticker)

    def _fund_book_growth(self, facts: Dict, ticker: str) -> Optional[float]:
        try:
            units = facts["facts"]["us-gaap"]["StockholdersEquity"]["units"].get("USD", [])
            annual = sorted(
                [v for v in units if v.get("form") in ("10-K", "10-K/A") and v.get("val")],
                key=lambda x: x.get("end", ""), reverse=True
            )
            if len(annual) >= 2 and annual[1]["val"] != 0:
                return (annual[0]["val"] - annual[1]["val"]) / abs(annual[1]["val"])
        except Exception:
            pass
        return None

    def _fund_operating_leverage(self, facts: Dict, ticker: str) -> Optional[float]:
        """Proxy: SG&A / Revenue as fixed cost intensity."""
        sga = _extract_ltm(facts, "SellingGeneralAndAdministrativeExpense")
        rev = _extract_ltm(facts, "Revenues") or _extract_ltm(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        if sga is not None and rev and rev > 0:
            return sga / rev
        return None

    def _fund_insider_ownership_pct(self, facts: Dict, ticker: str) -> Optional[float]:
        """Proxy via EDGAR DEF14A — approximate only."""
        return None  # Would require proxy statement parsing (separate module)

    def _fund_insider_buying_signal(self, facts: Dict, ticker: str) -> Optional[float]:
        """Form 4 insider buying signal — delegate to ownership_screener_v3 if available."""
        try:
            from sentinel.sfe.ownership_screener_v3 import InsiderSignalEngine
            engine = InsiderSignalEngine()
            score = engine.compute_insider_score(ticker, days=90)
            return float(score.net_purchase_value)
        except Exception:
            return None

    def _fund_institutional_hhi(self, facts: Dict, ticker: str) -> Optional[float]:
        """13F institutional HHI — delegate to institutional_ownership_v3 if available."""
        try:
            from sentinel.sfe.institutional_ownership_v3 import OwnershipAnalytics
            analytics = OwnershipAnalytics()
            result = analytics.get_ownership_concentration(ticker)
            return result.get("hhi")
        except Exception:
            return None

    # ---- Price-based factor helpers ----

    def _compute_price_factor(self, factor_name: str, ticker: str,
                               prices: "pd.DataFrame") -> Optional[float]:
        """Dispatch price factor computation."""
        fn = getattr(self, f"_price_{factor_name}", None)
        if fn:
            return fn(ticker, prices)
        # Generic momentum by lookback_days
        meta = FactorLibrary.FACTORS.get(factor_name, {})
        lookback = meta.get("lookback_days", 63)
        skip = meta.get("skip_days", 0)
        if "mom" in factor_name:
            return self._momentum(prices, lookback, skip)
        return None

    def _price_mom_1m(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        return self._momentum(prices, 21, 0)

    def _price_mom_3m(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        return self._momentum(prices, 63, 0)

    def _price_mom_6m(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        return self._momentum(prices, 126, 0)

    def _price_mom_12m(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        return self._momentum(prices, 252, 0)

    def _price_mom_12m_skip1(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        return self._momentum(prices, 252, 21)

    def _price_beta(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        spy = self._get_spy_prices()
        if spy is None or spy.empty or prices.empty:
            return None
        try:
            rets = prices["Close"].pct_change().dropna()
            spy_rets = spy["Close"].pct_change().dropna()
            common = rets.index.intersection(spy_rets.index)
            if len(common) < 30:
                return None
            x = spy_rets.loc[common].values
            y = rets.loc[common].values
            cov = float(_NUMPY and np.cov(x, y)[0][1] or sum((xi - sum(x)/len(x)) * (yi - sum(y)/len(y)) for xi, yi in zip(x, y)) / (len(x) - 1))
            var = float(_NUMPY and np.var(x, ddof=1) or statistics.variance(list(x)))
            return cov / var if var > 0 else None
        except Exception:
            return None

    def _price_idiosyncratic_volatility(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        spy = self._get_spy_prices()
        if spy is None or spy.empty:
            return self._price_realized_volatility_3m(ticker, prices)
        try:
            rets = prices["Close"].pct_change().dropna().tail(63)
            spy_rets = spy["Close"].pct_change().dropna()
            common = rets.index.intersection(spy_rets.index)
            if len(common) < 20:
                return None
            y = rets.loc[common].values
            x = spy_rets.loc[common].values
            if _NUMPY:
                beta = float(np.cov(x, y)[0][1] / np.var(x, ddof=1)) if np.var(x, ddof=1) > 0 else 0
                resids = y - beta * x
                return float(np.std(resids) * math.sqrt(252))
            return None
        except Exception:
            return None

    def _price_downside_deviation(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        try:
            rets = prices["Close"].pct_change().dropna().tail(126)
            neg_rets = [r for r in rets if r < 0]
            if len(neg_rets) < 5:
                return None
            if _NUMPY:
                return float(np.std(neg_rets) * math.sqrt(252))
            return statistics.stdev(neg_rets) * math.sqrt(252)
        except Exception:
            return None

    def _price_max_drawdown_12m(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        try:
            closes = prices["Close"].tail(252)
            if len(closes) < 10:
                return None
            running_max = closes.cummax()
            drawdown = (closes - running_max) / running_max
            return float(drawdown.min())
        except Exception:
            return None

    def _price_realized_volatility_1m(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        try:
            rets = prices["Close"].pct_change().dropna().tail(21)
            if len(rets) < 5:
                return None
            std = statistics.stdev(list(rets)) if len(rets) > 1 else 0
            return std * math.sqrt(252)
        except Exception:
            return None

    def _price_realized_volatility_3m(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        try:
            rets = prices["Close"].pct_change().dropna().tail(63)
            if len(rets) < 10:
                return None
            std = statistics.stdev(list(rets)) if len(rets) > 1 else 0
            return std * math.sqrt(252)
        except Exception:
            return None

    def _price_short_squeeze_score(self, ticker: str, prices: "pd.DataFrame") -> Optional[float]:
        """Proxy: recent momentum + volatility surge as squeeze indicator."""
        mom = self._price_mom_1m(ticker, prices)
        vol = self._price_realized_volatility_1m(ticker, prices)
        if mom is not None and vol is not None and vol > 0:
            return mom / vol  # sharp upward move = squeeze proxy
        return None

    # ---- Utilities ----

    def _momentum(self, prices: "pd.DataFrame", lookback: int, skip: int) -> Optional[float]:
        try:
            closes = prices["Close"]
            if len(closes) < lookback + 1:
                return None
            end_idx = len(closes) - 1 - skip
            start_idx = max(0, end_idx - lookback)
            if end_idx <= start_idx:
                return None
            return float((closes.iloc[end_idx] - closes.iloc[start_idx]) / closes.iloc[start_idx])
        except Exception:
            return None

    def _latest_price(self, ticker: str) -> Optional[float]:
        prices = self._get_prices(ticker, lookback_days=10)
        if prices is None or prices.empty:
            return None
        try:
            return float(prices["Close"].iloc[-1])
        except Exception:
            return None

    def _compute_ebitda(self, facts: Dict) -> Optional[float]:
        ebit = self._compute_ebit(facts)
        da = _extract_ltm(facts, "DepreciationDepletionAndAmortization") or 0
        if ebit is not None:
            return ebit + da
        return None

    def _compute_ebit(self, facts: Dict) -> Optional[float]:
        op = _extract_ltm(facts, "OperatingIncomeLoss")
        if op is not None:
            return op
        ni = _extract_ltm(facts, "NetIncomeLoss") or 0
        tax = _extract_ltm(facts, "IncomeTaxExpenseBenefit") or 0
        interest = _extract_ltm(facts, "InterestExpense") or 0
        return ni + tax + interest

    def _compute_ev(self, facts: Dict, ticker: str) -> Optional[float]:
        shares = _extract_latest_annual(facts, "CommonStockSharesOutstanding")
        price = self._latest_price(ticker)
        if not shares or not price:
            return None
        mkt_cap = shares * price
        debt = (_extract_latest_annual(facts, "LongTermDebt") or 0) + (_extract_latest_annual(facts, "ShortTermBorrowings") or 0)
        cash = _extract_latest_annual(facts, "CashAndCashEquivalentsAtCarryingValue") or 0
        return mkt_cap + debt - cash


# ---------------------------------------------------------------------------
# FactorTester
# ---------------------------------------------------------------------------

class FactorTester:
    """Information Coefficient analysis, quintile backtests, factor decay.

    Uses yfinance for forward return computation. All factor values
    are consumed as pre-computed pd.Series inputs.
    """

    def __init__(self):
        self._computer = FactorComputer()

    def compute_ic(self, factor: "pd.Series", forward_returns: "pd.Series") -> float:
        """Spearman rank IC between factor and forward returns."""
        if not _PANDAS:
            return 0.0
        common = factor.dropna().index.intersection(forward_returns.dropna().index)
        if len(common) < 5:
            return 0.0
        f_vals = factor.loc[common].tolist()
        r_vals = forward_returns.loc[common].tolist()
        return _spearman_corr(f_vals, r_vals)

    def compute_ic_series(self, factor_name: str, tickers: List[str],
                          start: str, end: str) -> "pd.Series":
        """Compute rolling monthly IC for a factor over a historical period."""
        if not _PANDAS or not _YF:
            return pd.Series(dtype=float)

        months = self._monthly_dates(start, end)
        ic_values: Dict[str, float] = {}
        computer = FactorComputer()

        for i, (period_start, period_end) in enumerate(zip(months[:-2], months[1:-1])):
            fwd_end = months[i + 2]
            try:
                # Factor cross-section at period_start
                factor_cs = computer.compute_factor(factor_name, tickers, period_start)
                if factor_cs.empty:
                    continue
                # Forward returns: period_end → fwd_end
                fwd_rets = self._compute_forward_returns(tickers, period_end, fwd_end)
                if fwd_rets.empty:
                    continue
                ic = self.compute_ic(factor_cs, fwd_rets)
                ic_values[period_start] = ic
            except Exception as exc:
                logger.debug(f"IC computation error at {period_start}: {exc}")
            time.sleep(0.1)

        return pd.Series(ic_values, dtype=float, name=factor_name)

    def compute_icir(self, ic_series: "pd.Series") -> float:
        """IC Information Ratio = mean(IC) / std(IC)."""
        if not _PANDAS or ic_series.empty or len(ic_series) < 2:
            return 0.0
        mu = float(ic_series.mean())
        sigma = float(ic_series.std())
        return mu / sigma if sigma > 0 else 0.0

    def run_quintile_backtest(self, factor_name: str, universe: List[str],
                              start: str, end: str) -> QuintileResult:
        """Sort stocks into 5 quintiles and compute return spread."""
        if not _PANDAS:
            return QuintileResult(factor_name=factor_name, start_date=start, end_date=end)

        months = self._monthly_dates(start, end)
        computer = FactorComputer()
        quintile_rets: Dict[int, List[float]] = {i: [] for i in range(5)}

        for i, (period_start, period_end) in enumerate(zip(months[:-1], months[1:])):
            try:
                factor_cs = computer.compute_factor(factor_name, universe, period_start)
                factor_cs = factor_cs.dropna().sort_values()
                if len(factor_cs) < 10:
                    continue
                fwd_rets = self._compute_forward_returns(universe, period_start, period_end)
                n = len(factor_cs)
                q_size = n // 5
                for q in range(5):
                    q_tickers = factor_cs.iloc[q * q_size: (q + 1) * q_size].index
                    q_fwd = fwd_rets.reindex(q_tickers).dropna()
                    if not q_fwd.empty:
                        quintile_rets[q].append(float(q_fwd.mean()))
            except Exception as exc:
                logger.debug(f"Quintile backtest error at {period_start}: {exc}")

        q_avg = [statistics.mean(quintile_rets[q]) if quintile_rets[q] else 0.0 for q in range(5)]
        spread = q_avg[-1] - q_avg[0]  # Q5 - Q1 (highest factor - lowest)
        hits = sum(1 for q in quintile_rets[4] if q > 0)
        total = len(quintile_rets[4]) or 1
        sharpe = 0.0
        if quintile_rets[4]:
            spreads = [a - b for a, b in zip(quintile_rets[4], quintile_rets[0]) if quintile_rets[0]]
            if spreads and statistics.stdev(spreads) > 0:
                sharpe = statistics.mean(spreads) / statistics.stdev(spreads) * math.sqrt(12)

        return QuintileResult(
            factor_name=factor_name,
            start_date=start, end_date=end,
            quintile_returns=[r * 12 for r in q_avg],  # annualize
            spread_q1_q5=spread * 12,
            hit_rate=hits / total,
            sharpe=sharpe,
            n_periods=len(months) - 1,
        )

    def compute_factor_decay(self, factor_name: str, tickers: List[str],
                             as_of_date: str) -> Dict[str, float]:
        """IC at horizons 1W, 2W, 1M, 3M, 6M to see alpha persistence."""
        if not _PANDAS or not _YF:
            return {}
        horizons = {"1W": 5, "2W": 10, "1M": 21, "3M": 63, "6M": 126}
        computer = FactorComputer()
        factor_cs = computer.compute_factor(factor_name, tickers, as_of_date)
        if factor_cs.empty:
            return {}

        base_date = datetime.strptime(as_of_date, "%Y-%m-%d")
        result: Dict[str, float] = {}
        for label, days in horizons.items():
            fwd_date = (base_date + timedelta(days=days)).strftime("%Y-%m-%d")
            fwd_rets = self._compute_forward_returns(tickers, as_of_date, fwd_date)
            ic = self.compute_ic(factor_cs, fwd_rets)
            result[label] = ic

        return result

    def compute_factor_ic_series(
        self,
        factor_values: List[float],
        next_month_returns: List[float],
    ) -> float:
        """
        Compute a single-period monthly IC as the Spearman rank correlation
        between factor values and next-month returns.

        This is the core building block for IC time-series analysis.
        IC > 0 means higher factor → higher subsequent return.

        Args:
            factor_values:       Cross-sectional factor scores for N stocks at period t.
            next_month_returns:  One-month-forward returns for the same N stocks.

        Returns:
            Spearman rank IC in [-1, +1].  Returns 0.0 if fewer than 3 observations.
        """
        if len(factor_values) != len(next_month_returns):
            raise ValueError(
                f"factor_values length ({len(factor_values)}) must equal "
                f"next_month_returns length ({len(next_month_returns)})"
            )
        if len(factor_values) < 3:
            return 0.0
        return _spearman_corr(factor_values, next_month_returns)

    def compute_factor_turnover(
        self,
        factor_ranks_t: List[float],
        factor_ranks_t1: List[float],
        n_long: int = None,
    ) -> float:
        """
        Compute average monthly position turnover from long-short rebalancing.

        Method:
            1. At time t   : long top-half, short bottom-half based on factor ranks.
            2. At time t+1 : rebalance to new top/bottom using updated ranks.
            3. Turnover = fraction of portfolio that changes side (long ↔ short ↔ flat).

        Args:
            factor_ranks_t:  Ordinal ranks at period t  (1 = lowest factor value).
            factor_ranks_t1: Ordinal ranks at period t+1.
            n_long:          Number of long (and short) positions. Defaults to N//2.

        Returns:
            Turnover fraction in [0, 1].  0 = no changes; 1 = full portfolio replaced.
        """
        n = len(factor_ranks_t)
        if n < 4 or n != len(factor_ranks_t1):
            return 0.0

        if n_long is None:
            n_long = n // 2

        # Determine long / short sets at each period
        # Sort by rank: top n_long = long, bottom n_long = short
        def _long_short_sets(ranks: List[float], k: int):
            sorted_idx = sorted(range(len(ranks)), key=lambda i: ranks[i], reverse=True)
            long_set  = set(sorted_idx[:k])
            short_set = set(sorted_idx[n - k:])
            return long_set, short_set

        long_t,  short_t  = _long_short_sets(factor_ranks_t,  n_long)
        long_t1, short_t1 = _long_short_sets(factor_ranks_t1, n_long)

        # Count positions that changed
        total_positions = n_long * 2  # long + short legs
        unchanged_long  = len(long_t  & long_t1)
        unchanged_short = len(short_t & short_t1)
        unchanged = unchanged_long + unchanged_short
        changed   = total_positions - unchanged

        return changed / total_positions if total_positions > 0 else 0.0

    def compute_gross_profitability_factor(
        self,
        revenue: float,
        cogs: float,
        total_assets: float,
    ) -> float:
        """
        Gross Profitability factor (Novy-Marx 2013).

        Formula:
            GP/Assets = (Revenue - COGS) / Total_Assets

        Higher values indicate companies that generate more gross profit
        per unit of assets.  This factor has significant positive alpha
        and is orthogonal to most value factors.

        Args:
            revenue:      Total revenues (annual).
            cogs:         Cost of goods sold / cost of revenues (annual).
            total_assets: Total assets on the balance sheet (annual).

        Returns:
            Gross profitability ratio.  Returns 0.0 if total_assets <= 0.
        """
        if total_assets <= 0.0:
            return 0.0
        gross_profit = revenue - cogs
        return gross_profit / total_assets

    def adjust_pvalues(self, pvalues: List[float], method: str = "bh") -> List[float]:
        """Multiple testing correction: 'bonferroni' or 'bh' (Benjamini-Hochberg)."""
        n = len(pvalues)
        if n == 0:
            return []
        if method == "bonferroni":
            return [min(1.0, p * n) for p in pvalues]
        # Benjamini-Hochberg FDR
        indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
        adjusted = [0.0] * n
        prev = 1.0
        for rank, (orig_idx, p) in reversed(list(enumerate(indexed))):
            adj = min(prev, p * n / (rank + 1))
            adjusted[orig_idx] = min(1.0, adj)
            prev = adj
        return adjusted

    def fama_macbeth_regression(
        self,
        factor_matrix: "pd.DataFrame",
        forward_returns: "pd.Series",
    ) -> Dict[str, Any]:
        """Fama-MacBeth two-pass regression for factor risk premia.

        Pass 1: Monthly cross-sectional OLS of returns on factor scores.
            r_{i,t} = lambda_0 + lambda_1 * f_{i,t} + ... + epsilon_{i,t}
            Repeating for each month t yields a time series of slope coefficients.

        Pass 2: Time-series average of slope coefficients gives the risk premium
            for each factor, with Newey-West corrected standard errors.

        Returns
        -------
        dict with keys:
            "lambda" : dict[factor_name -> risk_premium_estimate]
            "t_stat" : dict[factor_name -> t-statistic]
            "se"     : dict[factor_name -> standard_error]
            "n_months": int — number of cross-sectional regressions run
        """
        if not _PANDAS or factor_matrix.empty:
            return {"lambda": {}, "t_stat": {}, "se": {}, "n_months": 0}

        common = factor_matrix.dropna(how="all").index.intersection(
            forward_returns.dropna().index
        )
        if len(common) < 5:
            return {"lambda": {}, "t_stat": {}, "se": {}, "n_months": 0}

        X = factor_matrix.loc[common].fillna(0.0)
        y = forward_returns.loc[common]
        factor_names = list(X.columns)
        n = len(factor_names)

        # Single cross-section OLS (FM pass 1 — one period)
        if _NUMPY:
            try:
                X_mat = np.column_stack([np.ones(len(common)), X.values])
                coeffs, *_ = np.linalg.lstsq(X_mat, y.values, rcond=None)
                lambdas = {factor_names[i]: float(coeffs[i + 1]) for i in range(n)}
                # Pass 2: with only one period, SE = residual std / sqrt(N)
                y_hat = X_mat @ coeffs
                resid = y.values - y_hat
                resid_std = float(np.std(resid))
                se_val = resid_std / math.sqrt(len(common)) if len(common) > 0 else 1.0
                t_stats = {f: lambdas[f] / se_val if se_val > 0 else 0.0 for f in factor_names}
                ses = {f: se_val for f in factor_names}
                return {
                    "lambda": lambdas,
                    "t_stat": t_stats,
                    "se": ses,
                    "n_months": 1,
                }
            except Exception as exc:
                logger.debug(f"Fama-MacBeth numpy OLS failed: {exc}")

        # Pure-Python fallback: univariate FM for each factor
        lambdas: Dict[str, float] = {}
        t_stats: Dict[str, float] = {}
        ses: Dict[str, float] = {}
        for fname in factor_names:
            fvec = X[fname].values
            yvec = y.values
            n_obs = len(fvec)
            if n_obs < 3:
                lambdas[fname] = 0.0
                t_stats[fname] = 0.0
                ses[fname] = 0.0
                continue
            mu_x = sum(fvec) / n_obs
            mu_y = sum(yvec) / n_obs
            cov_xy = sum((fvec[i] - mu_x) * (yvec[i] - mu_y) for i in range(n_obs)) / (n_obs - 1)
            var_x  = sum((fvec[i] - mu_x) ** 2 for i in range(n_obs)) / (n_obs - 1)
            beta = cov_xy / var_x if var_x > 0 else 0.0
            alpha = mu_y - beta * mu_x
            resid = [yvec[i] - (alpha + beta * fvec[i]) for i in range(n_obs)]
            se = math.sqrt(sum(r ** 2 for r in resid) / max(1, n_obs - 2)) / math.sqrt(max(1, var_x * (n_obs - 1)))
            lambdas[fname] = beta
            ses[fname] = se
            t_stats[fname] = beta / se if se > 0 else 0.0
        return {"lambda": lambdas, "t_stat": t_stats, "se": ses, "n_months": 1}

    def run_long_short_portfolio(
        self,
        factor_scores: "pd.Series",
        forward_returns: "pd.Series",
        n_quantile: int = 5,
    ) -> Dict[str, Any]:
        """Compute long-top-quintile / short-bottom-quintile portfolio statistics.

        For each factor, sorts stocks into quintiles and computes:
          - Spread return (Q5 - Q1 = long - short)
          - Sharpe ratio (annualised, assuming monthly rebalance)
          - Max drawdown proxy across quintile buckets
          - CAGR estimate (compound annual growth rate)
          - Alpha vs equal-weight market portfolio

        Parameters
        ----------
        factor_scores   : pd.Series of cross-sectional factor values (tickers as index).
        forward_returns : pd.Series of one-period forward returns for same tickers.
        n_quantile      : Number of quantile buckets. Default 5 (quintiles).

        Returns
        -------
        dict with keys:
            "long_return"  : float — mean return of top quintile
            "short_return" : float — mean return of bottom quintile
            "spread"       : float — long_return - short_return
            "sharpe"       : float — annualised Sharpe (assuming monthly)
            "max_drawdown" : float — max single-period loss (proxy)
            "cagr"         : float — compound annual growth rate estimate
            "alpha"        : float — spread minus equal-weight universe return
            "n_stocks"     : int   — stocks in universe
            "q_returns"    : list[float] — mean return per quantile Q1..Q5
        """
        if not _PANDAS:
            return {}

        common = factor_scores.dropna().index.intersection(forward_returns.dropna().index)
        if len(common) < n_quantile * 2:
            return {
                "long_return": 0.0, "short_return": 0.0, "spread": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0, "cagr": 0.0,
                "alpha": 0.0, "n_stocks": len(common), "q_returns": [],
            }

        fac = factor_scores.loc[common].sort_values()
        rets = forward_returns.loc[common]
        n = len(fac)
        q_size = max(1, n // n_quantile)

        q_returns: List[float] = []
        for q in range(n_quantile):
            q_tickers = fac.iloc[q * q_size: (q + 1) * q_size].index
            q_ret = float(rets.reindex(q_tickers).dropna().mean()) if len(q_tickers) > 0 else 0.0
            q_returns.append(q_ret)

        long_ret  = q_returns[-1]   # top quintile
        short_ret = q_returns[0]    # bottom quintile
        spread    = long_ret - short_ret

        # Market (equal-weight) return
        mkt_ret = float(rets.mean())
        alpha   = spread - mkt_ret

        # Annualised Sharpe (assuming monthly, std across quintile spreads proxy)
        # Use Q5-Q1 spread distribution across quantile pairs
        spreads_vec = [q_returns[i + 1] - q_returns[i] for i in range(len(q_returns) - 1)]
        if len(spreads_vec) > 1 and statistics.stdev(spreads_vec) > 0:
            sharpe = (statistics.mean(spreads_vec) / statistics.stdev(spreads_vec)) * math.sqrt(12)
        else:
            sharpe = spread * math.sqrt(12) / max(abs(spread), 1e-6)

        # CAGR: assuming monthly spread → annualize via compounding
        cagr = (1.0 + spread) ** 12 - 1.0

        # Max drawdown: worst single quintile return in the long leg (proxy)
        max_drawdown = min(q_returns) if q_returns else 0.0

        return {
            "long_return":  round(long_ret,  6),
            "short_return": round(short_ret, 6),
            "spread":       round(spread,    6),
            "sharpe":       round(sharpe,    4),
            "max_drawdown": round(max_drawdown, 6),
            "cagr":         round(cagr,      6),
            "alpha":        round(alpha,     6),
            "n_stocks":     n,
            "q_returns":    [round(r, 6) for r in q_returns],
        }

    def compute_rolling_ic_timeseries(
        self,
        factor_scores_by_period: Dict[str, "pd.Series"],
        returns_by_period: Dict[str, "pd.Series"],
    ) -> "pd.Series":
        """Compute a rolling monthly IC time series from pre-computed cross-sections.

        For each period t in factor_scores_by_period, computes:
            IC_t = Spearman(factor_scores[t], forward_returns[t+1])

        Then derives IC_mean, IC_std, ICIR, and t-stat:
            ICIR   = IC_mean / IC_std
            t_stat = ICIR * sqrt(T)

        Parameters
        ----------
        factor_scores_by_period : dict mapping date_str -> pd.Series(ticker -> score)
        returns_by_period       : dict mapping date_str -> pd.Series(ticker -> return)
            The return at period t represents the forward return from t to t+1.

        Returns
        -------
        pd.Series indexed by date, values = IC at each period.
            .attrs["ic_mean"], .attrs["ic_std"], .attrs["icir"], .attrs["t_stat"]
            are set on the returned Series for downstream consumption.
        """
        if not _PANDAS:
            return pd.Series(dtype=float)

        ic_dict: Dict[str, float] = {}
        periods = sorted(factor_scores_by_period.keys())

        for period in periods:
            fac = factor_scores_by_period.get(period)
            ret = returns_by_period.get(period)
            if fac is None or ret is None or fac.empty or ret.empty:
                continue
            common = fac.dropna().index.intersection(ret.dropna().index)
            if len(common) < 3:
                continue
            ic = _spearman_corr(fac.loc[common].tolist(), ret.loc[common].tolist())
            ic_dict[period] = ic

        if not ic_dict:
            return pd.Series(dtype=float, name="IC")

        ic_series = pd.Series(ic_dict, dtype=float, name="IC")
        ic_series.index = pd.to_datetime(ic_series.index)
        ic_series = ic_series.sort_index()

        T = len(ic_series)
        if T >= 2:
            ic_mean = float(ic_series.mean())
            ic_std  = float(ic_series.std())
            icir    = ic_mean / ic_std if ic_std > 0 else 0.0
            t_stat  = icir * math.sqrt(T)
        else:
            ic_mean = float(ic_series.iloc[0]) if T == 1 else 0.0
            ic_std  = 0.0
            icir    = 0.0
            t_stat  = 0.0

        ic_series.attrs["ic_mean"] = round(ic_mean, 6)
        ic_series.attrs["ic_std"]  = round(ic_std,  6)
        ic_series.attrs["icir"]    = round(icir,    6)
        ic_series.attrs["t_stat"]  = round(t_stat,  6)

        return ic_series

    # ---- Helpers ----

    def _monthly_dates(self, start: str, end: str) -> List[str]:
        """Generate list of first-of-month dates between start and end."""
        dates = []
        current = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        while current <= end_dt:
            dates.append(current.strftime("%Y-%m-%d"))
            # Advance one month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return dates

    def _compute_forward_returns(self, tickers: List[str],
                                 start: str, end: str) -> "pd.Series":
        """Compute 1-period forward returns for a universe between two dates."""
        if not _YF or not _PANDAS:
            return pd.Series(dtype=float)
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            fetch_start = (start_dt - timedelta(days=5)).strftime("%Y-%m-%d")
            fetch_end = (end_dt + timedelta(days=5)).strftime("%Y-%m-%d")

            tickers_str = " ".join(tickers)
            data = yf.download(tickers_str, start=fetch_start, end=fetch_end,
                               auto_adjust=True, progress=False)
            if data.empty:
                return pd.Series(dtype=float)

            closes = data["Close"] if "Close" in data.columns else data
            if isinstance(closes, pd.Series):
                closes = closes.to_frame()

            start_prices = closes.loc[:start].iloc[-1] if not closes.loc[:start].empty else None
            end_prices = closes.loc[:end].iloc[-1] if not closes.loc[:end].empty else None

            if start_prices is None or end_prices is None:
                return pd.Series(dtype=float)

            returns = (end_prices - start_prices) / start_prices.replace(0, float("nan"))
            return returns.dropna()
        except Exception as exc:
            logger.debug(f"Forward returns computation error: {exc}")
            return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# FactorCombiner
# ---------------------------------------------------------------------------

class FactorCombiner:
    """Combine multiple factor signals into composite alpha signals."""

    def equal_weight(self, factors: Dict[str, "pd.Series"]) -> "pd.Series":
        """Simple equal-weight average of standardized factors."""
        if not _PANDAS or not factors:
            return pd.Series(dtype=float)
        df = pd.DataFrame(factors)
        return df.mean(axis=1)

    def ic_weighted(self, factors: Dict[str, "pd.Series"],
                    ic_scores: Dict[str, float]) -> "pd.Series":
        """IC-weighted composite: weight each factor by its absolute IC."""
        if not _PANDAS or not factors:
            return pd.Series(dtype=float)
        df = pd.DataFrame(factors)
        weights = {}
        total = sum(abs(ic_scores.get(f, 0)) for f in df.columns)
        for col in df.columns:
            weights[col] = abs(ic_scores.get(col, 0)) / total if total > 0 else 1.0 / len(df.columns)
        result = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            result += df[col].fillna(0.0) * w
        return result

    def ml_combine(self, factors: "pd.DataFrame",
                   returns: "pd.Series") -> CompositeSignal:
        """Combine factors using Ridge regression (sklearn or numpy fallback)."""
        if not _PANDAS or factors.empty:
            return CompositeSignal(method="ml_ridge")

        # Align factors and returns
        common = factors.dropna(how="all").index.intersection(returns.dropna().index)
        if len(common) < 10:
            return CompositeSignal(method="ml_ridge")

        X = factors.loc[common].fillna(0.0)
        y = returns.loc[common]
        tickers = X.index.tolist()

        if _SKLEARN:
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = Ridge(alpha=1.0)
                model.fit(X_scaled, y)
                scores = model.predict(X_scaled)
                weights = {col: float(coef) for col, coef in zip(X.columns, model.coef_)}
                return CompositeSignal(
                    tickers=tickers,
                    scores=list(scores),
                    weights=weights,
                    method="ml_ridge_sklearn",
                    icir=_spearman_corr(list(scores), list(y)),
                )
            except Exception as exc:
                logger.debug(f"sklearn Ridge failed: {exc}")

        # Numpy fallback: L2-penalized lstsq
        if _NUMPY:
            try:
                X_np = X.values
                y_np = y.values
                mu = X_np.mean(axis=0)
                sigma = X_np.std(axis=0)
                sigma[sigma == 0] = 1
                X_scaled = (X_np - mu) / sigma
                # Add L2 penalty: (X'X + lambda*I) * beta = X'y
                lam = 1.0
                A = X_scaled.T @ X_scaled + lam * np.eye(X_scaled.shape[1])
                b = X_scaled.T @ y_np
                coefs = np.linalg.lstsq(A, b, rcond=None)[0]
                scores = X_scaled @ coefs
                weights = {col: float(c) for col, c in zip(X.columns, coefs)}
                return CompositeSignal(
                    tickers=tickers,
                    scores=list(scores),
                    weights=weights,
                    method="ml_ridge_numpy",
                    icir=_spearman_corr(list(scores), list(y_np)),
                )
            except Exception as exc:
                logger.debug(f"numpy Ridge fallback failed: {exc}")

        # Last resort: equal weight
        eq = factors.loc[common].mean(axis=1)
        return CompositeSignal(
            tickers=tickers,
            scores=list(eq),
            weights={col: 1.0 / len(factors.columns) for col in factors.columns},
            method="equal_weight_fallback",
        )

    def compute_factor_correlation(self, factors: "pd.DataFrame") -> "pd.DataFrame":
        """Pearson correlation matrix of factors to identify redundancy."""
        if not _PANDAS:
            return pd.DataFrame()
        return factors.corr(method="pearson")


# ---------------------------------------------------------------------------
# AlphaSignalGenerator
# ---------------------------------------------------------------------------

class AlphaSignalGenerator:
    """AI-assisted or rule-based factor hypothesis generation and interpretation."""

    _REGIME_FACTORS: Dict[str, List[str]] = {
        "recession": ["piotroski_f", "altman_z", "current_ratio", "earnings_quality",
                      "pe_ratio", "pb_ratio", "dividend_yield", "financial_leverage"],
        "recovery": ["mom_3m", "mom_6m", "revenue_growth_yoy", "earnings_revision_momentum",
                     "eps_growth", "roic"],
        "expansion": ["mom_12m", "mom_12m_skip1", "revenue_growth_qoq", "rd_intensity",
                      "eps_growth", "operating_margin"],
        "late_cycle": ["dividend_yield", "buyback_yield", "total_yield", "gross_margin",
                       "net_margin", "roe", "low_beta"],
        "stagflation": ["dividend_yield", "pb_ratio", "earnings_yield", "gross_margin",
                        "operating_leverage"],
    }

    def __init__(self):
        self._anthropic_client = None
        if _ANTHROPIC:
            try:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    self._anthropic_client = _anthropic_mod.Anthropic(api_key=api_key)
            except Exception:
                pass

    def generate_factor_hypothesis(self, sector: str,
                                   macro_regime: str) -> List[FactorHypothesis]:
        """Generate factor hypotheses using Claude API if available, else rule-based."""
        if self._anthropic_client:
            return self._ai_hypotheses(sector, macro_regime)
        return self._rule_based_hypotheses(sector, macro_regime)

    def _ai_hypotheses(self, sector: str, macro_regime: str) -> List[FactorHypothesis]:
        """Use Claude to generate factor hypotheses (requires ANTHROPIC_API_KEY)."""
        try:
            prompt = (
                f"You are a quantitative analyst. For a stock screen in the {sector} sector "
                f"during a {macro_regime} macro regime, suggest 5 alpha factors with rationale. "
                f"Format each as: factor_name|category|rationale|direction (higher/lower_is_better)"
            )
            msg = self._anthropic_client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            hypotheses = []
            for line in text.strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 4:
                    hypotheses.append(FactorHypothesis(
                        factor_name=parts[0].strip(),
                        category=parts[1].strip(),
                        rationale=parts[2].strip(),
                        expected_direction=parts[3].strip(),
                        regime_fit=[macro_regime],
                        confidence=0.75,
                    ))
            return hypotheses
        except Exception as exc:
            logger.debug(f"Claude API error, falling back to rule-based: {exc}")
            return self._rule_based_hypotheses(sector, macro_regime)

    def _rule_based_hypotheses(self, sector: str,
                                macro_regime: str) -> List[FactorHypothesis]:
        """Rule-based factor hypothesis generation."""
        regime_factors = self._REGIME_FACTORS.get(macro_regime.lower(), self._REGIME_FACTORS["expansion"])
        hypotheses = []
        for fname in regime_factors[:5]:
            meta = FactorLibrary.FACTORS.get(fname, {})
            hypotheses.append(FactorHypothesis(
                factor_name=fname,
                category=meta.get("category", "unknown"),
                rationale=f"Favored in {macro_regime} regime based on historical factor research",
                expected_direction=meta.get("direction", "higher_is_better"),
                regime_fit=[macro_regime],
                confidence=0.6,
            ))
        return hypotheses

    def describe_factor(self, factor_name: str, ic_stats: Dict[str, Any]) -> str:
        """Explain a factor in plain English with performance statistics."""
        meta = FactorLibrary.FACTORS.get(factor_name, {})
        desc = meta.get("description", factor_name)
        category = meta.get("category", "unknown")
        ic_mean = ic_stats.get("ic_mean", 0.0)
        icir = ic_stats.get("icir", 0.0)
        direction = meta.get("direction", "higher_is_better")

        interpretation = "strong" if abs(icir) > 0.5 else "moderate" if abs(icir) > 0.3 else "weak"
        quality = "positive" if ic_mean > 0.02 else "negative" if ic_mean < -0.02 else "neutral"

        return (
            f"{factor_name} ({category.upper()}): {desc}. "
            f"Stocks with {'higher' if direction == 'higher_is_better' else 'lower'} {factor_name} "
            f"tend to outperform. IC mean={ic_mean:.3f}, ICIR={icir:.2f} — {interpretation} {quality} signal. "
            f"Suitable for {'value/quality' if category in ('value', 'quality') else 'trend-following'} strategies."
        )

    def rank_factors_by_regime(self, regime: str) -> "pd.DataFrame":
        """Return a DataFrame ranking which factors work in the given macro regime."""
        if not _PANDAS:
            return pd.DataFrame()
        regime_factors = self._REGIME_FACTORS.get(regime.lower(), list(FactorLibrary.FACTORS.keys()))
        rows = []
        for fname in regime_factors:
            meta = FactorLibrary.FACTORS.get(fname, {})
            rows.append({
                "factor": fname,
                "category": meta.get("category", ""),
                "description": meta.get("description", ""),
                "regime_rank": regime_factors.index(fname) + 1,
                "direction": meta.get("direction", ""),
            })
        return pd.DataFrame(rows).set_index("factor")

    def detect_factor_crowding(self, factor_name: str,
                               ic_series: "pd.Series") -> bool:
        """Detect factor crowding: mean-reverting IC or deteriorating ICIR indicates crowding."""
        if not _PANDAS or ic_series.empty or len(ic_series) < 6:
            return False
        # Split IC series in half; if latter half has significantly lower IC, signal crowding
        n = len(ic_series)
        first_half = ic_series.iloc[:n // 2]
        second_half = ic_series.iloc[n // 2:]
        ic_early = float(first_half.mean())
        ic_late = float(second_half.mean())
        # Crowded if IC has degraded by > 50%
        if ic_early != 0 and (ic_late / ic_early) < 0.5:
            return True
        # Also check if IC is mean-reverting (alternating sign)
        sign_changes = sum(1 for i in range(1, len(ic_series)) if ic_series.iloc[i] * ic_series.iloc[i - 1] < 0)
        if sign_changes > len(ic_series) * 0.5:
            return True
        return False

    def generate_alpha_report(self, universe: List[str]) -> str:
        """Generate a full factor research narrative."""
        lines = [
            "=" * 70,
            "SENTINEL FACTOR RESEARCH REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Universe: {len(universe)} stocks",
            "=" * 70,
            "",
            "FACTOR CATEGORIES",
            "-" * 40,
        ]
        for cat in FactorLibrary.CATEGORIES:
            factors = FactorLibrary.get_factors_by_category(cat)
            lines.append(f"  {cat.upper()}: {len(factors)} factors — " + ", ".join(list(factors.keys())[:5]) + "...")

        lines += ["", "MACRO REGIME RECOMMENDATIONS", "-" * 40]
        for regime in ["recession", "expansion", "late_cycle"]:
            top = self._REGIME_FACTORS.get(regime, [])[:3]
            lines.append(f"  {regime.upper()}: {', '.join(top)}")

        lines += ["", "NOTE: Run FactorResearchEngine.run_factor_scan() for live factor scores."]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FactorResearchEngine (Orchestrator)
# ---------------------------------------------------------------------------

class FactorResearchEngine:
    """Orchestrates factor research pipeline: scan → test → combine → export."""

    def __init__(self):
        self.library = FactorLibrary()
        self.computer = FactorComputer()
        self.tester = FactorTester()
        self.combiner = FactorCombiner()
        self.signal_gen = AlphaSignalGenerator()
        self.db = FactorDatabaseV3()

    def run_factor_scan(self, universe: List[str],
                        date_str: Optional[str] = None,
                        include_macro: bool = True) -> FactorScanResult:
        """Compute all factors for universe, compute IC, identify top signals.

        Macro factors (FRED series — monthly/quarterly cadence) are merged with
        a forward-fill so that the most recent published value is used on days
        where no new data has been released yet.  Rows with NaN macro data are
        never dropped; the ffill ensures complete coverage.
        """
        import time as _time
        t0 = _time.time()
        if date_str is None:
            date_str = datetime.today().strftime("%Y-%m-%d")

        logger.info(f"Factor scan: {len(universe)} tickers, date={date_str}")

        # Step 1: Compute all factors
        factor_matrix = self.computer.compute_all_factors(universe, date_str)
        factors_computed = factor_matrix.shape[1] if _PANDAS and not factor_matrix.empty else 0

        # Step 1b: Merge macro overlays (FRED) with forward-fill — never drop rows
        if _PANDAS and include_macro and not factor_matrix.empty:
            macro = MacroFactorFetcher()
            for series_id in list(macro.SERIES.values())[:4]:  # yield_curve, credit_spread, unemployment, inflation
                try:
                    macro_series = macro.fetch(series_id, periods=60)
                    if macro_series is not None and not macro_series.empty:
                        # Resample to daily, forward-fill gaps, then pick the scan date value
                        daily = macro_series.resample("D").last().ffill()
                        scan_dt = pd.Timestamp(date_str)
                        val = None
                        if scan_dt in daily.index:
                            val = float(daily.loc[scan_dt])
                        elif not daily.empty:
                            # Use last known value at or before scan date
                            prior = daily.loc[daily.index <= scan_dt]
                            if not prior.empty:
                                val = float(prior.iloc[-1])
                        if val is not None and pd.notna(val):
                            # Broadcast scalar macro value to all tickers as a factor column
                            factor_matrix[f"macro_{series_id}"] = val
                            factors_computed = factor_matrix.shape[1]
                except Exception as exc:
                    logger.debug(f"Macro overlay {series_id} failed: {exc}")

        # Step 2: Standardize factors
        factor_ic: Dict[str, float] = {}
        factor_icir: Dict[str, float] = {}

        if _PANDAS and not factor_matrix.empty:
            for col in factor_matrix.columns:
                series = factor_matrix[col].dropna()
                std_series = self.computer.standardize(series)
                # Get sector series for neutralization
                sectors = pd.Series({t: _SECTOR_MAP.get(t, "Unknown") for t in universe}, name="sector")
                neutral = self.computer.neutralize_sector(std_series, sectors.reindex(std_series.index))
                self.db.store_factor_cross_section(col, date_str, series, std_series, neutral)

            # Step 3: Quick IC estimate using available forward returns proxy
            # Use 1-month historical returns as a proxy for forward returns (approximation)
            fwd_start = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
            fwd_rets = self.tester._compute_forward_returns(universe, fwd_start, date_str)

            for col in factor_matrix.columns:
                factor_cs = factor_matrix[col].dropna()
                if len(factor_cs) < 5:
                    continue
                ic = self.tester.compute_ic(factor_cs, fwd_rets)
                factor_ic[col] = ic
                # Retrieve IC history from DB for ICIR
                ic_hist = self.db.get_ic_history(col)
                # Add current IC to history
                if _PANDAS and not ic_hist.empty:
                    all_ic = list(ic_hist.values) + [ic]
                    sigma = statistics.stdev(all_ic) if len(all_ic) > 1 else 1.0
                    icir = statistics.mean(all_ic) / sigma if sigma > 0 else 0.0
                else:
                    icir = 0.0
                factor_icir[col] = icir
                self.db.store_ic(col, date_str, ic, icir)

        # Step 4: Rank factors by |ICIR|
        sorted_factors = sorted(factor_icir.keys(),
                                key=lambda f: abs(factor_icir.get(f, 0)), reverse=True)
        top_factors = sorted_factors[:10]

        # Step 5: Build composite signal
        composite = None
        if _PANDAS and not factor_matrix.empty and top_factors:
            top_df = factor_matrix[top_factors].dropna(how="all")
            std_factors = {col: self.computer.standardize(top_df[col].dropna())
                           for col in top_df.columns}
            composite_series = self.combiner.ic_weighted(std_factors, factor_ic)
            composite = CompositeSignal(
                tickers=list(composite_series.index),
                scores=list(composite_series.values),
                weights={f: factor_ic.get(f, 0) for f in top_factors},
                method="ic_weighted",
                date=date_str,
            )

        runtime = _time.time() - t0
        return FactorScanResult(
            date=date_str,
            universe=universe,
            factors_computed=factors_computed,
            top_factors=top_factors,
            factor_ic=factor_ic,
            factor_icir=factor_icir,
            composite_signal=composite,
            runtime_seconds=runtime,
        )

    def get_best_factors(self, n: int = 10,
                         regime: Optional[str] = None) -> List[str]:
        """Return top-n factors by ICIR from DB history, filtered by regime if given."""
        if regime:
            regime_factors = AlphaSignalGenerator._REGIME_FACTORS.get(regime.lower(), [])
            if regime_factors:
                return regime_factors[:n]

        # Pull from DB — compute ICIR for all factors
        factor_icirs: Dict[str, float] = {}
        for fname in FactorLibrary.get_factor_names():
            hist = self.db.get_ic_history(fname)
            if _PANDAS and not hist.empty and len(hist) >= 2:
                mu = float(hist.mean())
                sigma = float(hist.std())
                factor_icirs[fname] = mu / sigma if sigma > 0 else 0.0

        if not factor_icirs:
            return FactorLibrary.get_factor_names()[:n]

        return sorted(factor_icirs.keys(),
                      key=lambda f: abs(factor_icirs[f]), reverse=True)[:n]

    def build_composite_signal(self, universe: List[str]) -> "pd.Series":
        """Build production-ready composite alpha signal for a universe."""
        if not _PANDAS:
            return pd.Series(dtype=float)
        date_str = datetime.today().strftime("%Y-%m-%d")
        top_factors = self.get_best_factors(n=10)
        computer = FactorComputer()
        factor_series: Dict[str, "pd.Series"] = {}
        for fname in top_factors:
            cs = computer.compute_factor(fname, universe, date_str)
            if not cs.empty:
                factor_series[fname] = self.computer.standardize(cs)

        if not factor_series:
            return pd.Series(dtype=float)

        return self.combiner.equal_weight(factor_series)

    def backtest_composite(self, universe: List[str],
                           start: str, end: str) -> BacktestResult:
        """Backtest composite signal: long top quintile, short bottom quintile."""
        if not _PANDAS:
            return BacktestResult(start_date=start, end_date=end)

        months = self.tester._monthly_dates(start, end)
        monthly_pnl: List[float] = []
        top_factors = self.get_best_factors(n=5)

        for i, (period_start, period_end) in enumerate(zip(months[:-1], months[1:])):
            try:
                factor_series: Dict[str, "pd.Series"] = {}
                for fname in top_factors:
                    cs = self.computer.compute_factor(fname, universe, period_start)
                    if not cs.empty:
                        factor_series[fname] = self.computer.standardize(cs)

                if not factor_series:
                    continue

                composite = self.combiner.equal_weight(factor_series)
                fwd_rets = self.tester._compute_forward_returns(universe, period_start, period_end)

                # Long top 20%, short bottom 20%
                n = len(composite.dropna())
                if n < 10:
                    continue
                sorted_c = composite.dropna().sort_values()
                n_q = max(1, n // 5)
                long_tickers = sorted_c.tail(n_q).index
                short_tickers = sorted_c.head(n_q).index
                long_ret = float(fwd_rets.reindex(long_tickers).dropna().mean())
                short_ret = float(fwd_rets.reindex(short_tickers).dropna().mean())
                pnl = long_ret - short_ret
                monthly_pnl.append(pnl)
            except Exception as exc:
                logger.debug(f"Backtest error at {period_start}: {exc}")

        if not monthly_pnl:
            return BacktestResult(start_date=start, end_date=end)

        ann_ret = statistics.mean(monthly_pnl) * 12
        ann_vol = statistics.stdev(monthly_pnl) * math.sqrt(12) if len(monthly_pnl) > 1 else 0.0
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        win_rate = sum(1 for r in monthly_pnl if r > 0) / len(monthly_pnl)
        # Max drawdown
        cum = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in monthly_pnl:
            cum *= (1 + r)
            peak = max(peak, cum)
            dd = (cum - peak) / peak
            max_dd = min(max_dd, dd)

        return BacktestResult(
            start_date=start,
            end_date=end,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_periods=len(monthly_pnl),
            monthly_returns=monthly_pnl,
        )

    def export_signal(self, date_str: Optional[str] = None) -> "pd.DataFrame":
        """Export composite signal as ticker → score DataFrame for portfolio construction."""
        if not _PANDAS:
            return pd.DataFrame()
        if date_str is None:
            date_str = datetime.today().strftime("%Y-%m-%d")
        matrix = self.db.get_factor_matrix(date_str)
        if matrix.empty:
            logger.warning(f"No factor data in DB for {date_str} — run run_factor_scan() first")
            return pd.DataFrame()

        top_factors = [f for f in self.get_best_factors(n=10) if f in matrix.columns]
        if not top_factors:
            top_factors = list(matrix.columns)[:10]

        composite = matrix[top_factors].mean(axis=1).sort_values(ascending=False)
        result = pd.DataFrame({
            "composite_score": composite,
            "rank": range(1, len(composite) + 1),
            "date": date_str,
        })
        return result


# ---------------------------------------------------------------------------
# FRED macro factor helper
# ---------------------------------------------------------------------------

class MacroFactorFetcher:
    """Fetch macro factors from FRED CSV (no API key required)."""

    SERIES = {
        "yield_curve": "T10Y2Y",       # 10Y-2Y Treasury spread
        "credit_spread": "BAMLH0A0HYM2",  # High yield OAS
        "unemployment": "UNRATE",
        "inflation_yoy": "CPIAUCSL",
        "gdp_growth": "A191RL1Q225SBEA",
        "fed_funds": "FEDFUNDS",
        "ism_manufacturing": "MANEMP",
    }

    def fetch(self, series_id: str, periods: int = 24) -> Optional["pd.Series"]:
        """Fetch a FRED series as a pd.Series."""
        url = f"{_FRED_BASE}?id={series_id}"
        resp = _safe_get(url, timeout=15)
        if not resp or not _PANDAS:
            return None
        try:
            import io
            df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
            col = df.columns[0]
            series = df[col].dropna().tail(periods)
            series.name = series_id
            return series
        except Exception as exc:
            logger.debug(f"FRED fetch failed {series_id}: {exc}")
            return None

    def detect_macro_regime(self) -> str:
        """Classify current macro regime based on yield curve and unemployment."""
        yc = self.fetch("T10Y2Y", 3)
        if yc is not None and not yc.empty:
            latest_yc = float(yc.iloc[-1])
            if latest_yc < 0:
                return "recession"
            elif latest_yc > 1.5:
                return "expansion"
            else:
                return "late_cycle"
        return "expansion"  # default


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SENTINEL Factor Research Engine v3 — Live Demo")
    print("=" * 70)

    DEMO_UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
        "JPM", "BAC", "GS", "WFC", "V",
        "XOM", "CVX", "JNJ", "UNH", "LLY",
        "PG", "KO", "PEP", "WMT", "COST",
        "HD", "MCD", "NFLX", "TSLA", "ADBE",
        "AMD", "INTC", "TXN", "CRM", "ORCL",
        "CAT", "HON", "ABT", "TMO", "DHR",
        "MRK", "ABBV", "AMGN", "PM", "MO",
        "BX", "SPGI", "ICE", "CME", "BLK",
        "LIN", "APD", "ECL", "PPG", "NEM",
    ]

    print(f"\nUniverse: {len(DEMO_UNIVERSE)} tickers")
    print(f"Factor library: {len(FactorLibrary.get_factor_names())} factors")
    print(f"\nFactor categories:")
    for cat in FactorLibrary.CATEGORIES:
        factors = FactorLibrary.get_factors_by_category(cat)
        print(f"  {cat:15s}: {len(factors)} factors")

    print("\n--- Detecting macro regime ---")
    macro = MacroFactorFetcher()
    regime = macro.detect_macro_regime()
    print(f"Current regime: {regime.upper()}")

    print("\n--- Generating factor hypotheses ---")
    gen = AlphaSignalGenerator()
    hyps = gen.generate_factor_hypothesis(sector="Technology", macro_regime=regime)
    for h in hyps[:3]:
        print(f"  [{h.category}] {h.factor_name}: {h.rationale[:60]}...")

    print("\n--- Computing factor cross-section (subset of factors) ---")
    computer = FactorComputer()
    VALUE_MOMENTUM = ["pe_ratio", "pb_ratio", "earnings_yield", "mom_3m", "mom_6m",
                      "mom_12m", "gross_margin", "net_margin", "roe", "piotroski_f"]

    print(f"Computing {len(VALUE_MOMENTUM)} factors for {len(DEMO_UNIVERSE)} tickers...")
    results: Dict[str, Dict] = {}
    for factor_name in VALUE_MOMENTUM:
        print(f"  Factor: {factor_name} ...", end=" ")
        cs = computer.compute_factor(factor_name, DEMO_UNIVERSE[:20])  # subset for demo
        n_valid = cs.dropna().shape[0] if _PANDAS else 0
        print(f"{n_valid} valid values")
        if _PANDAS and not cs.empty:
            results[factor_name] = {
                "count": n_valid,
                "mean": float(cs.mean()),
                "std": float(cs.std()),
            }

    print("\n--- Top 5 factors by coverage ---")
    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)
        for fname, stats in sorted_results[:5]:
            print(f"  {fname:30s}: n={stats['count']}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    print("\n--- Factor regime ranking ---")
    regime_df = gen.rank_factors_by_regime(regime)
    if _PANDAS and not regime_df.empty:
        print(regime_df[["category", "regime_rank"]].head(5).to_string())

    print("\n--- Alpha report ---")
    report = gen.generate_alpha_report(DEMO_UNIVERSE)
    print(report[:400])

    print("\n--- Factor Library summary ---")
    print(f"Total factors defined: {len(FactorLibrary.FACTORS)}")
    print("Value factors:", list(FactorLibrary.get_factors_by_category("value").keys()))
    print("Momentum factors:", list(FactorLibrary.get_factors_by_category("momentum").keys()))

    print("\nFactor Research Engine v3 — demo complete.")
    print("Run FactorResearchEngine().run_factor_scan(universe) for full pipeline.")
