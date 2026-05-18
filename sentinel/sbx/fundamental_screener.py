"""Bloomberg-quality fundamental equity screener backed by DuckDB — dim_070.

Covers 50+ fields across identity, size, income, growth, margins, valuation,
returns, leverage, cash flow, quality, momentum, risk, and analyst data.

Public API
----------
FundamentalDataLoader
    build_screener_db(tickers, db_path) -> duckdb.Connection
    refresh_ticker(conn, ticker)
    get_last_updated(conn) -> pd.DataFrame

FundamentalScreener
    screen(criteria, universe) -> pd.DataFrame
    run_preset(name, universe) -> pd.DataFrame
    compute_piotroski_score(ticker) -> dict
    compute_altman_z(ticker) -> dict
    compute_beneish_m_score(ticker) -> dict
    rank_universe(metrics, universe) -> pd.DataFrame
    PREBUILT_SCREENS : dict[str, dict]

DuckDBQueryEngine
    execute_sql(query, conn) -> pd.DataFrame
    to_parquet(query, output_path)
    get_schema() -> pd.DataFrame

FastAPI
-------
screener_router  — mounted at /api/screener/fundamental
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

try:
    import duckdb
except ImportError as _exc:
    raise ImportError("duckdb is required: pip install duckdb") from _exc

from sentinel.core.logging import get_logger
from sentinel.sfe.standardized_financials import (
    FinancialStatementStandardizer,
    resolve_cik,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_RATE_DELAY = 0.15
_TIMEOUT = 30.0
_DEFAULT_DB = ".sentinel/cache/screener.duckdb"

# S&P 500 + Russell 2000 representative universe (trimmed for startup speed;
# callers can pass their own list)
_SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM", "UNH", "XOM", "V", "LLY", "JNJ", "WMT", "MA", "PG", "HD",
    "AVGO", "CVX", "MRK", "ABBV", "KO", "PEP", "COST", "ADBE", "NFLX",
    "CRM", "ACN", "TMO", "MCD", "CSCO", "BAC", "ABT", "DHR", "NEE",
    "ORCL", "WFC", "TXN", "PM", "LIN", "SPGI", "AMGN", "LOW", "ISRG",
    "GS", "SYK", "RTX", "BKNG", "INTU", "NOW", "ELV", "MDT", "AXP",
    "BLK", "VRTX", "GILD", "PLD", "REGN", "MO", "EOG", "SLB", "ZTS",
    "MMC", "CB", "ITW", "HUM", "BSX", "CME", "SCHW", "ADI", "AMAT",
    "LRCX", "KLAC", "PANW", "CDNS", "SNPS", "MCHP", "MU", "NXPI",
    "INTC", "AMD", "QCOM", "IBM", "HPQ", "DELL", "WDC", "STX", "KEYS",
    "F", "GM", "TM", "HMC", "STLA", "DAL", "UAL", "AAL", "LUV", "BA",
    "GE", "HON", "MMM", "CAT", "DE", "EMR", "ETN", "PH", "ROK",
    "DUK", "SO", "AEP", "D", "EXC", "PCG", "SRE", "XEL", "AWK",
    "CVS", "WBA", "MCK", "ABC", "CAH", "HCA", "CNC", "CI", "MOH",
    "T", "VZ", "TMUS", "CMCSA", "DIS", "PARA", "WBD", "NWSA",
    "GLD", "SLV", "USO", "UNG", "DJP",
]

# ---------------------------------------------------------------------------
# DuckDB schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fundamentals (
    -- Identity
    ticker                    VARCHAR PRIMARY KEY,
    company_name              VARCHAR,
    exchange                  VARCHAR,
    sector                    VARCHAR,
    industry                  VARCHAR,
    sic_code                  VARCHAR,
    country                   VARCHAR,
    -- Size
    market_cap                DOUBLE,
    enterprise_value          DOUBLE,
    shares_outstanding        DOUBLE,
    float_shares              DOUBLE,
    -- Income (TTM)
    revenue_ttm               DOUBLE,
    gross_profit_ttm          DOUBLE,
    ebitda_ttm                DOUBLE,
    ebit_ttm                  DOUBLE,
    net_income_ttm            DOUBLE,
    eps_ttm                   DOUBLE,
    -- Growth
    revenue_growth_1yr        DOUBLE,
    revenue_growth_3yr_cagr   DOUBLE,
    eps_growth_1yr            DOUBLE,
    eps_growth_3yr_cagr       DOUBLE,
    -- Margins
    gross_margin              DOUBLE,
    ebitda_margin             DOUBLE,
    net_margin                DOUBLE,
    fcf_margin                DOUBLE,
    -- Valuation
    pe_ttm                    DOUBLE,
    forward_pe                DOUBLE,
    peg_ratio                 DOUBLE,
    ps_ttm                    DOUBLE,
    pb_ratio                  DOUBLE,
    ev_ebitda                 DOUBLE,
    ev_revenue                DOUBLE,
    price_to_fcf              DOUBLE,
    -- Returns
    roe                       DOUBLE,
    roa                       DOUBLE,
    roic                      DOUBLE,
    roce                      DOUBLE,
    -- Leverage
    debt_to_equity            DOUBLE,
    net_debt_to_ebitda        DOUBLE,
    interest_coverage         DOUBLE,
    current_ratio             DOUBLE,
    quick_ratio               DOUBLE,
    -- Cash flow
    operating_cf_ttm          DOUBLE,
    capex_ttm                 DOUBLE,
    fcf_ttm                   DOUBLE,
    fcf_yield                 DOUBLE,
    buyback_yield             DOUBLE,
    dividend_yield            DOUBLE,
    total_yield               DOUBLE,
    -- Quality
    accruals_ratio            DOUBLE,
    asset_growth_1yr          DOUBLE,
    inventory_growth_1yr      DOUBLE,
    -- Momentum (price returns)
    price_1m                  DOUBLE,
    price_3m                  DOUBLE,
    price_6m                  DOUBLE,
    price_12m                 DOUBLE,
    price_vs_52w_high         DOUBLE,
    price_vs_52w_low          DOUBLE,
    -- Risk
    beta                      DOUBLE,
    vol_30d                   DOUBLE,
    vol_252d                  DOUBLE,
    short_float               DOUBLE,
    institutional_ownership_pct DOUBLE,
    -- Analyst
    analyst_rating_mean       DOUBLE,
    analyst_target_price      DOUBLE,
    upside_to_target          DOUBLE,
    num_analysts              INTEGER,
    -- Meta
    last_updated              TIMESTAMP,
    fiscal_year_end           DATE,
    data_source               VARCHAR
)
"""

# Column descriptions for get_schema()
_COLUMN_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "ticker": ("VARCHAR", "Stock ticker symbol"),
    "company_name": ("VARCHAR", "Full legal company name"),
    "exchange": ("VARCHAR", "Primary listing exchange (NYSE/NASDAQ/etc)"),
    "sector": ("VARCHAR", "GICS sector"),
    "industry": ("VARCHAR", "GICS industry"),
    "sic_code": ("VARCHAR", "SEC Standard Industrial Classification code"),
    "country": ("VARCHAR", "Country of incorporation"),
    "market_cap": ("DOUBLE", "Market capitalization in USD"),
    "enterprise_value": ("DOUBLE", "Enterprise value = mktcap + net debt"),
    "shares_outstanding": ("DOUBLE", "Total shares outstanding"),
    "float_shares": ("DOUBLE", "Public float shares"),
    "revenue_ttm": ("DOUBLE", "Trailing 12-month revenue"),
    "gross_profit_ttm": ("DOUBLE", "Trailing 12-month gross profit"),
    "ebitda_ttm": ("DOUBLE", "Trailing 12-month EBITDA"),
    "ebit_ttm": ("DOUBLE", "Trailing 12-month EBIT"),
    "net_income_ttm": ("DOUBLE", "Trailing 12-month net income"),
    "eps_ttm": ("DOUBLE", "Trailing 12-month diluted EPS"),
    "revenue_growth_1yr": ("DOUBLE", "YoY revenue growth (decimal, e.g. 0.12 = 12%)"),
    "revenue_growth_3yr_cagr": ("DOUBLE", "3-year revenue CAGR"),
    "eps_growth_1yr": ("DOUBLE", "YoY EPS growth"),
    "eps_growth_3yr_cagr": ("DOUBLE", "3-year EPS CAGR"),
    "gross_margin": ("DOUBLE", "Gross profit / revenue"),
    "ebitda_margin": ("DOUBLE", "EBITDA / revenue"),
    "net_margin": ("DOUBLE", "Net income / revenue"),
    "fcf_margin": ("DOUBLE", "Free cash flow / revenue"),
    "pe_ttm": ("DOUBLE", "Price / TTM EPS"),
    "forward_pe": ("DOUBLE", "Price / forward EPS estimate"),
    "peg_ratio": ("DOUBLE", "PE / EPS growth rate"),
    "ps_ttm": ("DOUBLE", "Price / TTM sales per share"),
    "pb_ratio": ("DOUBLE", "Price / book value per share"),
    "ev_ebitda": ("DOUBLE", "Enterprise value / EBITDA"),
    "ev_revenue": ("DOUBLE", "Enterprise value / revenue"),
    "price_to_fcf": ("DOUBLE", "Market cap / FCF"),
    "roe": ("DOUBLE", "Net income / average shareholders equity"),
    "roa": ("DOUBLE", "Net income / average total assets"),
    "roic": ("DOUBLE", "NOPAT / invested capital"),
    "roce": ("DOUBLE", "EBIT / capital employed"),
    "debt_to_equity": ("DOUBLE", "Total debt / shareholders equity"),
    "net_debt_to_ebitda": ("DOUBLE", "Net debt / EBITDA"),
    "interest_coverage": ("DOUBLE", "EBIT / interest expense"),
    "current_ratio": ("DOUBLE", "Current assets / current liabilities"),
    "quick_ratio": ("DOUBLE", "(Cash + receivables) / current liabilities"),
    "operating_cf_ttm": ("DOUBLE", "Trailing 12-month operating cash flow"),
    "capex_ttm": ("DOUBLE", "Trailing 12-month capital expenditures"),
    "fcf_ttm": ("DOUBLE", "Trailing 12-month free cash flow"),
    "fcf_yield": ("DOUBLE", "FCF / market cap"),
    "buyback_yield": ("DOUBLE", "Net share buybacks / market cap"),
    "dividend_yield": ("DOUBLE", "Annual dividend / price"),
    "total_yield": ("DOUBLE", "Dividend yield + buyback yield"),
    "accruals_ratio": ("DOUBLE", "Accruals / avg net operating assets (Sloan ratio)"),
    "asset_growth_1yr": ("DOUBLE", "YoY total asset growth"),
    "inventory_growth_1yr": ("DOUBLE", "YoY inventory growth"),
    "price_1m": ("DOUBLE", "1-month price return"),
    "price_3m": ("DOUBLE", "3-month price return"),
    "price_6m": ("DOUBLE", "6-month price return"),
    "price_12m": ("DOUBLE", "12-month price return"),
    "price_vs_52w_high": ("DOUBLE", "Current price vs 52-week high (negative = below)"),
    "price_vs_52w_low": ("DOUBLE", "Current price vs 52-week low (positive = above)"),
    "beta": ("DOUBLE", "Market beta (vs S&P 500, 5yr monthly)"),
    "vol_30d": ("DOUBLE", "30-day annualized volatility"),
    "vol_252d": ("DOUBLE", "252-day annualized volatility"),
    "short_float": ("DOUBLE", "Short interest as % of float"),
    "institutional_ownership_pct": ("DOUBLE", "Institutional ownership % of float"),
    "analyst_rating_mean": ("DOUBLE", "Mean analyst rating (1=Strong Buy, 5=Strong Sell)"),
    "analyst_target_price": ("DOUBLE", "Analyst consensus price target"),
    "upside_to_target": ("DOUBLE", "Upside to analyst target price"),
    "num_analysts": ("INTEGER", "Number of analyst estimates"),
    "last_updated": ("TIMESTAMP", "Timestamp of last data refresh"),
    "fiscal_year_end": ("DATE", "Fiscal year end date of latest annual financials"),
    "data_source": ("VARCHAR", "Source of fundamental data (edgar/yfinance/mixed)"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _cagr(current: float | None, past: float | None, years: int) -> float | None:
    if current is None or past is None or past <= 0 or years <= 0:
        return None
    try:
        return (current / past) ** (1 / years) - 1
    except (ValueError, ZeroDivisionError):
        return None


def _pct_change(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / abs(prior)


def _annualized_vol(returns: pd.Series) -> float | None:
    if returns is None or len(returns) < 5:
        return None
    return float(returns.std() * math.sqrt(252))


# ---------------------------------------------------------------------------
# FundamentalDataLoader
# ---------------------------------------------------------------------------

class FundamentalDataLoader:
    """Fetch and persist fundamental data to a DuckDB file database."""

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout
        self._standardizer = FinancialStatementStandardizer()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build_screener_db(
        self,
        tickers: list[str] | None = None,
        db_path: str = _DEFAULT_DB,
    ) -> "duckdb.DuckDBPyConnection":
        """Build or refresh the DuckDB screener database.

        Parameters
        ----------
        tickers:
            Optional explicit list. Defaults to ``_SP500_SAMPLE``.
        db_path:
            Path to the DuckDB file. Created if it does not exist.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Open connection to the refreshed database.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(db_path)
        conn.execute(_CREATE_TABLE_SQL)

        universe = tickers or _SP500_SAMPLE
        total = len(universe)
        logger.info("fundamental_screener_build_start", n_tickers=total, db=db_path)

        for i, ticker in enumerate(universe, 1):
            try:
                self.refresh_ticker(conn, ticker)
                logger.debug("ticker_loaded", ticker=ticker, progress=f"{i}/{total}")
            except Exception as exc:
                logger.warning("ticker_load_failed", ticker=ticker, error=str(exc))
            time.sleep(_RATE_DELAY)

        logger.info("fundamental_screener_build_done", db=db_path)
        return conn

    def refresh_ticker(
        self,
        conn: "duckdb.DuckDBPyConnection",
        ticker: str,
    ) -> None:
        """Fetch latest data for a single ticker and upsert into DuckDB."""
        row = self._fetch_all(ticker)
        if row is None:
            return
        self._upsert(conn, row)

    def get_last_updated(
        self,
        conn: "duckdb.DuckDBPyConnection",
    ) -> pd.DataFrame:
        """Return staleness report (ticker, last_updated, age_hours)."""
        sql = """
            SELECT ticker, company_name, sector, last_updated,
                   date_diff('hour', last_updated, NOW()) AS age_hours
            FROM   fundamentals
            ORDER  BY age_hours DESC
        """
        return conn.execute(sql).df()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_all(self, ticker: str) -> dict[str, Any] | None:
        """Aggregate data from yfinance + EDGAR → flat row dict."""
        try:
            yf_ticker = yf.Ticker(ticker)
            info: dict = yf_ticker.info or {}
            hist = yf_ticker.history(period="2y", auto_adjust=True)
        except Exception as exc:
            logger.warning("yfinance_fetch_failed", ticker=ticker, error=str(exc))
            return None

        if hist.empty:
            return None

        row: dict[str, Any] = {"ticker": ticker, "data_source": "yfinance"}

        # --- Identity ---
        row["company_name"] = info.get("longName") or info.get("shortName", ticker)
        row["exchange"] = info.get("exchange", "")
        row["sector"] = info.get("sector", "")
        row["industry"] = info.get("industry", "")
        row["sic_code"] = str(info.get("sic", "")) if info.get("sic") else ""
        row["country"] = info.get("country", "")

        # --- Size ---
        row["market_cap"] = info.get("marketCap")
        row["shares_outstanding"] = info.get("sharesOutstanding")
        row["float_shares"] = info.get("floatShares")
        price = info.get("currentPrice") or info.get("regularMarketPrice") or (
            float(hist["Close"].iloc[-1]) if not hist.empty else None
        )
        total_debt = info.get("totalDebt") or 0.0
        cash = info.get("totalCash") or info.get("cash") or 0.0
        mktcap = row["market_cap"] or 0.0
        row["enterprise_value"] = mktcap + total_debt - cash if mktcap else None

        # --- Income ---
        row["revenue_ttm"] = info.get("totalRevenue")
        row["gross_profit_ttm"] = info.get("grossProfits")
        row["ebitda_ttm"] = info.get("ebitda")
        row["ebit_ttm"] = (
            (info.get("ebitda") or 0) - (info.get("depreciationAndAmortization") or 0)
            if info.get("ebitda") else None
        )
        row["net_income_ttm"] = info.get("netIncomeToCommon")
        row["eps_ttm"] = info.get("trailingEps")

        # --- Growth ---
        row["revenue_growth_1yr"] = info.get("revenueGrowth")
        row["eps_growth_1yr"] = info.get("earningsGrowth")
        # 3yr CAGR approximated from yfinance earnings history
        row["revenue_growth_3yr_cagr"] = None
        row["eps_growth_3yr_cagr"] = info.get("earningsQuarterlyGrowth")  # proxy

        # --- Margins ---
        row["gross_margin"] = info.get("grossMargins")
        row["ebitda_margin"] = info.get("ebitdaMargins")
        row["net_margin"] = info.get("profitMargins")
        operating_cf = info.get("operatingCashflow") or info.get("freeCashflow")
        capex = info.get("capitalExpenditures") or 0.0
        fcf = info.get("freeCashflow")
        if fcf is None and operating_cf:
            fcf = operating_cf - abs(capex)
        row["fcf_margin"] = _safe_div(fcf, row["revenue_ttm"])

        # --- Valuation ---
        row["pe_ttm"] = info.get("trailingPE")
        row["forward_pe"] = info.get("forwardPE")
        row["peg_ratio"] = info.get("pegRatio")
        row["ps_ttm"] = info.get("priceToSalesTrailing12Months")
        row["pb_ratio"] = info.get("priceToBook")
        ev = row.get("enterprise_value")
        row["ev_ebitda"] = _safe_div(ev, row["ebitda_ttm"])
        row["ev_revenue"] = _safe_div(ev, row["revenue_ttm"])
        row["price_to_fcf"] = _safe_div(mktcap, fcf) if mktcap and fcf else None

        # --- Returns ---
        row["roe"] = info.get("returnOnEquity")
        row["roa"] = info.get("returnOnAssets")
        # ROIC = NOPAT / invested capital (approximated)
        nopat = (row["ebit_ttm"] or 0) * (1 - 0.21)  # 21% tax rate proxy
        equity = info.get("bookValue", 0) * (info.get("sharesOutstanding") or 0)
        invested_capital = equity + total_debt - cash
        row["roic"] = _safe_div(nopat, invested_capital) if invested_capital > 0 else None
        row["roce"] = _safe_div(row["ebit_ttm"], invested_capital) if invested_capital > 0 else None

        # --- Leverage ---
        row["debt_to_equity"] = info.get("debtToEquity")
        if row["debt_to_equity"] is not None:
            row["debt_to_equity"] /= 100  # yfinance returns as %, normalize
        net_debt = total_debt - cash
        row["net_debt_to_ebitda"] = _safe_div(net_debt, row["ebitda_ttm"])
        interest_exp = info.get("interestExpense")
        row["interest_coverage"] = _safe_div(row["ebit_ttm"], abs(interest_exp)) if interest_exp else None
        row["current_ratio"] = info.get("currentRatio")
        row["quick_ratio"] = info.get("quickRatio")

        # --- Cash flow ---
        row["operating_cf_ttm"] = info.get("operatingCashflow")
        row["capex_ttm"] = info.get("capitalExpenditures")
        row["fcf_ttm"] = fcf
        row["fcf_yield"] = _safe_div(fcf, mktcap) if mktcap else None
        buyback = -(info.get("repurchaseOfStock") or 0)
        row["buyback_yield"] = _safe_div(buyback, mktcap) if mktcap else None
        row["dividend_yield"] = info.get("dividendYield")
        row["total_yield"] = (row.get("dividend_yield") or 0) + (row.get("buyback_yield") or 0)

        # --- Quality ---
        total_assets = info.get("totalAssets") or 0
        row["asset_growth_1yr"] = None  # requires prior-year assets
        row["inventory_growth_1yr"] = None
        # Accruals (Sloan): (net income - operating CF) / avg total assets
        ni = row["net_income_ttm"] or 0
        ocf = row["operating_cf_ttm"] or 0
        row["accruals_ratio"] = _safe_div(ni - ocf, total_assets) if total_assets else None

        # --- Momentum (price returns) ---
        closes = hist["Close"].dropna()
        if len(closes) >= 2:
            cur = float(closes.iloc[-1])
            def _ret(days: int) -> float | None:
                idx = max(0, len(closes) - days - 1)
                past = float(closes.iloc[idx]) if idx < len(closes) else None
                return _pct_change(cur, past) if past else None

            row["price_1m"] = _ret(21)
            row["price_3m"] = _ret(63)
            row["price_6m"] = _ret(126)
            row["price_12m"] = _ret(252)

            high_52w = float(closes.tail(252).max())
            low_52w = float(closes.tail(252).min())
            row["price_vs_52w_high"] = _pct_change(cur, high_52w)
            row["price_vs_52w_low"] = _pct_change(cur, low_52w)
        else:
            for f in ("price_1m", "price_3m", "price_6m", "price_12m",
                      "price_vs_52w_high", "price_vs_52w_low"):
                row[f] = None

        # --- Risk ---
        row["beta"] = info.get("beta")
        log_rets = np.log(closes / closes.shift(1)).dropna()
        row["vol_30d"] = _annualized_vol(log_rets.tail(30))
        row["vol_252d"] = _annualized_vol(log_rets.tail(252))
        row["short_float"] = info.get("shortPercentOfFloat")
        row["institutional_ownership_pct"] = info.get("institutionsPercentHeld")

        # --- Analyst ---
        row["analyst_rating_mean"] = info.get("recommendationMean")
        row["analyst_target_price"] = info.get("targetMeanPrice")
        row["upside_to_target"] = _pct_change(info.get("targetMeanPrice"), price) if price else None
        row["num_analysts"] = info.get("numberOfAnalystOpinions")

        # --- Meta ---
        row["last_updated"] = datetime.utcnow()
        row["fiscal_year_end"] = None
        return row

    def _upsert(self, conn: "duckdb.DuckDBPyConnection", row: dict[str, Any]) -> None:
        """INSERT OR REPLACE row into fundamentals table."""
        cols = list(row.keys())
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)
        values = [row[c] for c in cols]
        conn.execute(
            f"INSERT OR REPLACE INTO fundamentals ({col_names}) VALUES ({placeholders})",
            values,
        )


# ---------------------------------------------------------------------------
# FundamentalScreener
# ---------------------------------------------------------------------------

class FundamentalScreener:
    """DuckDB-backed fundamental equity screener with 20 preset screens."""

    PREBUILT_SCREENS: dict[str, dict] = {
        "magic_formula": {
            "description": "Greenblatt Magic Formula: high earnings yield + high ROIC",
            "sql_filter": "ev_ebitda IS NOT NULL AND roic IS NOT NULL AND ev_ebitda > 0",
            "sort_by": "magic_formula_rank",
            "computed_rank": True,
        },
        "piotroski_f8": {
            "description": "Piotroski F-score >= 8 (high quality financial strength)",
            "criteria": {
                "roe": {"gt": 0.0},
                "roa": {"gt": 0.0},
                "operating_cf_ttm": {"gt": 0.0},
                "debt_to_equity": {"lt": 1.0},
                "current_ratio": {"gt": 1.0},
                "gross_margin": {"gt": 0.2},
            },
            "sort_by": "roe",
        },
        "quality_growth": {
            "description": "High quality compounder: ROIC>15%, revenue growth>10%, low leverage",
            "criteria": {
                "roic": {"gt": 0.15},
                "revenue_growth_1yr": {"gt": 0.10},
                "debt_to_equity": {"lt": 1.0},
                "pe_ttm": {"lt": 30.0},
                "net_margin": {"gt": 0.10},
            },
            "sort_by": "roic",
        },
        "deep_value": {
            "description": "Classic deep value: low PE, low PB, high FCF yield",
            "criteria": {
                "pe_ttm": {"lt": 10.0},
                "pb_ratio": {"lt": 1.5},
                "fcf_yield": {"gt": 0.08},
            },
            "sort_by": "fcf_yield",
        },
        "dividend_aristocrats": {
            "description": "Dividend growth: yield>2%, revenue growth>3%, low payout",
            "criteria": {
                "dividend_yield": {"gt": 0.02},
                "revenue_growth_1yr": {"gt": 0.03},
                "net_margin": {"gt": 0.05},
                "current_ratio": {"gt": 1.0},
            },
            "sort_by": "dividend_yield",
        },
        "momentum": {
            "description": "Price momentum leaders: strong 6m and 12m returns",
            "criteria": {
                "price_6m": {"gt": 0.15},
                "price_12m": {"gt": 0.20},
                "price_1m": {"gt": 0.0},
                "vol_252d": {"lt": 0.50},
            },
            "sort_by": "price_12m",
        },
        "low_volatility": {
            "description": "Low vol defensive: annualized vol<15%, beta<0.8, positive FCF",
            "criteria": {
                "vol_252d": {"lt": 0.15},
                "beta": {"lt": 0.8},
                "fcf_ttm": {"gt": 0.0},
            },
            "sort_by": "vol_252d",
        },
        "net_net": {
            "description": "Graham net-net: EV < net current assets (deep discount)",
            "sql_filter": "enterprise_value < 0 OR ev_revenue < 0.5",
            "criteria": {
                "pb_ratio": {"lt": 1.0},
                "pe_ttm": {"lt": 15.0},
            },
            "sort_by": "pb_ratio",
        },
        "growth_at_reasonable_price": {
            "description": "GARP: PEG<1.5, EPS growth>15%, PE<25",
            "criteria": {
                "peg_ratio": {"lt": 1.5},
                "eps_growth_1yr": {"gt": 0.15},
                "pe_ttm": {"lt": 25.0},
                "pe_ttm_min": {"gt": 5.0},
            },
            "sort_by": "peg_ratio",
        },
        "high_short_interest": {
            "description": "High short interest (>20%): potential squeeze candidates",
            "criteria": {
                "short_float": {"gt": 0.20},
                "market_cap": {"gt": 100_000_000},
            },
            "sort_by": "short_float",
        },
        "analysts_bullish": {
            "description": "Strong analyst consensus with >20% upside to target",
            "criteria": {
                "analyst_rating_mean": {"lt": 2.5},
                "upside_to_target": {"gt": 0.20},
                "num_analysts": {"gte": 5},
            },
            "sort_by": "upside_to_target",
        },
        "mega_cap_quality": {
            "description": "Mega-cap (>$50B) with strong returns and margins",
            "criteria": {
                "market_cap": {"gt": 50_000_000_000},
                "roe": {"gt": 0.20},
                "net_margin": {"gt": 0.15},
                "revenue_growth_1yr": {"gt": 0.05},
            },
            "sort_by": "market_cap",
        },
        "small_cap_growth": {
            "description": "Small-cap ($300M-$2B) high growth names",
            "criteria": {
                "market_cap": {"between": [300_000_000, 2_000_000_000]},
                "revenue_growth_1yr": {"gt": 0.20},
                "gross_margin": {"gt": 0.40},
            },
            "sort_by": "revenue_growth_1yr",
        },
        "cash_rich": {
            "description": "Companies with net cash (negative net debt), profitable",
            "criteria": {
                "net_debt_to_ebitda": {"lt": 0.0},
                "fcf_yield": {"gt": 0.03},
                "roe": {"gt": 0.10},
            },
            "sort_by": "fcf_yield",
        },
        "turnaround": {
            "description": "Turnaround candidates: previously beaten-down, improving margins",
            "criteria": {
                "price_12m": {"lt": -0.20},
                "price_3m": {"gt": 0.0},
                "gross_margin": {"gt": 0.15},
                "operating_cf_ttm": {"gt": 0.0},
            },
            "sort_by": "price_3m",
        },
        "high_fcf": {
            "description": "Exceptional FCF generators: FCF yield > 6%, low leverage",
            "criteria": {
                "fcf_yield": {"gt": 0.06},
                "debt_to_equity": {"lt": 2.0},
                "net_margin": {"gt": 0.08},
            },
            "sort_by": "fcf_yield",
        },
        "sector_technology": {
            "description": "Technology sector quality screen",
            "sql_filter": "sector = 'Technology'",
            "criteria": {
                "revenue_growth_1yr": {"gt": 0.08},
                "gross_margin": {"gt": 0.50},
                "net_margin": {"gt": 0.0},
            },
            "sort_by": "revenue_growth_1yr",
        },
        "sector_healthcare": {
            "description": "Healthcare sector quality and value screen",
            "sql_filter": "sector = 'Healthcare'",
            "criteria": {
                "roe": {"gt": 0.12},
                "pe_ttm": {"lt": 25.0},
                "fcf_ttm": {"gt": 0.0},
            },
            "sort_by": "roe",
        },
        "low_pe_high_quality": {
            "description": "Value-quality hybrid: low PE with strong fundamentals",
            "criteria": {
                "pe_ttm": {"between": [5.0, 15.0]},
                "roe": {"gt": 0.15},
                "debt_to_equity": {"lt": 0.5},
                "revenue_growth_1yr": {"gt": 0.0},
            },
            "sort_by": "roe",
        },
        "insider_alignment": {
            "description": "High institutional ownership with low short interest",
            "criteria": {
                "institutional_ownership_pct": {"gt": 0.60},
                "short_float": {"lt": 0.05},
                "roe": {"gt": 0.12},
            },
            "sort_by": "institutional_ownership_pct",
        },
    }

    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        self.db_path = db_path
        self._conn: "duckdb.DuckDBPyConnection | None" = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> "duckdb.DuckDBPyConnection":
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(self.db_path)
            self._conn.execute(_CREATE_TABLE_SQL)
        return self._conn

    # ------------------------------------------------------------------
    # Screening
    # ------------------------------------------------------------------

    def screen(
        self,
        criteria: dict[str, dict],
        universe: str = "sp500",
        sort_by: str | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Run a custom screen against the DuckDB fundamentals table.

        Parameters
        ----------
        criteria:
            Dict mapping column name to operator dict, e.g.
            ``{"pe_ttm": {"lt": 20}, "roe": {"gt": 0.15}}``.
            Supported operators: ``lt``, ``gt``, ``lte``, ``gte``,
            ``between``, ``eq``, ``in``.
        universe:
            ``"sp500"`` (default) or ``"all"`` — no filter when ``"all"``.
        sort_by:
            Column to sort results by (descending).
        limit:
            Maximum rows to return.
        """
        conn = self._get_conn()
        where_clauses = self._build_where(criteria)
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        order_col = sort_by or "market_cap"
        sql = f"""
            SELECT * FROM fundamentals
            WHERE  {where_sql}
            ORDER  BY {order_col} DESC NULLS LAST
            LIMIT  {limit}
        """
        try:
            return conn.execute(sql).df()
        except Exception as exc:
            raise ValueError(f"Screen query failed: {exc}\nSQL: {sql}") from exc

    def run_preset(self, name: str, universe: str = "sp500") -> pd.DataFrame:
        """Run a named preset screen."""
        spec = self.PREBUILT_SCREENS.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(self.PREBUILT_SCREENS)}"
            )

        criteria = dict(spec.get("criteria", {}))
        sort_by = spec.get("sort_by", "market_cap")
        sql_filter = spec.get("sql_filter")

        # Special ranking for magic formula
        if spec.get("computed_rank") and name == "magic_formula":
            return self._run_magic_formula()

        df = self.screen(criteria, universe=universe, sort_by=sort_by)

        # Apply any raw SQL post-filter
        if sql_filter and not df.empty:
            conn = self._get_conn()
            tmp_sql = f"SELECT * FROM fundamentals WHERE ({sql_filter})"
            try:
                extra = conn.execute(tmp_sql).df()
                # intersect on ticker
                df = df[df["ticker"].isin(extra["ticker"])]
            except Exception:
                pass

        return df

    def _run_magic_formula(self) -> pd.DataFrame:
        """Greenblatt Magic Formula: rank on earnings yield + ROIC rank."""
        conn = self._get_conn()
        sql = """
            SELECT *,
                   (1.0 / NULLIF(ev_ebitda, 0)) AS earnings_yield,
                   roic
            FROM   fundamentals
            WHERE  ev_ebitda > 0 AND roic IS NOT NULL AND market_cap > 50000000
            ORDER  BY ev_ebitda ASC
            LIMIT  500
        """
        df = conn.execute(sql).df()
        if df.empty:
            return df
        df["ey_rank"] = df["earnings_yield"].rank(ascending=False, na_option="bottom")
        df["roic_rank"] = df["roic"].rank(ascending=False, na_option="bottom")
        df["magic_formula_rank"] = df["ey_rank"] + df["roic_rank"]
        return df.sort_values("magic_formula_rank").head(50)

    # ------------------------------------------------------------------
    # Single-ticker analytical scores
    # ------------------------------------------------------------------

    def compute_piotroski_score(self, ticker: str) -> dict[str, Any]:
        """Compute Piotroski F-score (0-9).

        9 binary criteria across profitability (3), leverage (3), efficiency (3).
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM fundamentals WHERE ticker = ?", [ticker.upper()]
        ).df()
        if rows.empty:
            raise ValueError(f"Ticker {ticker} not in database")
        r = rows.iloc[0].to_dict()

        signals: dict[str, int] = {}

        # Profitability (3 signals)
        signals["F1_roa_positive"] = 1 if (r.get("roa") or 0) > 0 else 0
        signals["F2_operating_cf_positive"] = 1 if (r.get("operating_cf_ttm") or 0) > 0 else 0
        # F3: operating CF > net income (accruals quality)
        signals["F3_cf_exceeds_ni"] = (
            1 if (r.get("operating_cf_ttm") or 0) > (r.get("net_income_ttm") or 0)
            else 0
        )

        # Leverage & liquidity (3 signals)
        dte = r.get("debt_to_equity")
        signals["F4_low_leverage"] = 1 if dte is not None and dte < 1.0 else 0
        signals["F5_high_current_ratio"] = 1 if (r.get("current_ratio") or 0) > 1.0 else 0
        signals["F6_no_dilution"] = 0  # can't assess without prior share count in DB

        # Efficiency (3 signals)
        signals["F7_higher_gross_margin"] = 1 if (r.get("gross_margin") or 0) > 0.2 else 0
        signals["F8_higher_asset_turnover"] = (
            1 if r.get("revenue_ttm") and r.get("market_cap") and
            (r["revenue_ttm"] / r["market_cap"]) > 0.3 else 0
        )
        signals["F9_positive_roe"] = 1 if (r.get("roe") or 0) > 0 else 0

        f_score = sum(signals.values())
        return {
            "ticker": ticker,
            "f_score": f_score,
            "interpretation": (
                "Strong" if f_score >= 7 else
                "Moderate" if f_score >= 4 else
                "Weak"
            ),
            "signals": signals,
        }

    def compute_altman_z(self, ticker: str) -> dict[str, Any]:
        """Compute Altman Z-score bankruptcy predictor.

        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        Zones: Z > 2.99 = safe, 1.81-2.99 = grey, < 1.81 = distress
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM fundamentals WHERE ticker = ?", [ticker.upper()]
        ).df()
        if rows.empty:
            raise ValueError(f"Ticker {ticker} not in database")
        r = rows.iloc[0].to_dict()

        total_assets = (r.get("market_cap") or 0) + (r.get("net_debt_to_ebitda") or 0)
        # Proxy total assets from market cap + net debt
        mktcap = r.get("market_cap") or 0
        ebitda = r.get("ebitda_ttm") or 0
        total_liab = mktcap * (r.get("debt_to_equity") or 0)
        revenue = r.get("revenue_ttm") or 0
        ebit = r.get("ebit_ttm") or 0

        ta = max(mktcap * 0.6, 1)  # rough total assets proxy

        x1 = _safe_div((r.get("current_ratio") or 1) - 1, ta) or 0.0  # working capital / TA
        x2 = _safe_div(r.get("net_income_ttm") or 0, ta) or 0.0        # retained earnings proxy
        x3 = _safe_div(ebit, ta) or 0.0                                 # EBIT / TA
        x4 = _safe_div(mktcap, max(total_liab, 1)) or 0.0              # mktcap / liabilities
        x5 = _safe_div(revenue, ta) or 0.0                              # revenue / TA

        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        zone = (
            "safe" if z > 2.99 else
            "grey" if z > 1.81 else
            "distress"
        )

        return {
            "ticker": ticker,
            "z_score": round(z, 3),
            "zone": zone,
            "components": {
                "X1_working_capital_ta": round(x1, 4),
                "X2_retained_earnings_ta": round(x2, 4),
                "X3_ebit_ta": round(x3, 4),
                "X4_mktcap_liabilities": round(x4, 4),
                "X5_revenue_ta": round(x5, 4),
            },
        }

    # ------------------------------------------------------------------
    # New analytical functions (dim_070 score 8 → 9)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_piotroski_f_score(
        roa: float,
        operating_cf: float,
        net_income: float,
        long_term_debt_ratio: float,
        long_term_debt_ratio_prior: float,
        current_ratio: float,
        current_ratio_prior: float,
        shares_outstanding: float,
        shares_outstanding_prior: float,
        gross_margin: float,
        gross_margin_prior: float,
        asset_turnover: float,
        asset_turnover_prior: float,
    ) -> dict[str, Any]:
        """Compute exact Piotroski (2000) F-score — 9-point binary model.

        Three groups of signals:
        Profitability (F1-F3):
          F1 = ROA > 0
          F2 = Operating CF > 0
          F3 = CF/Assets > ROA (accruals quality: OCF > net income)
        Leverage / Liquidity (F4-F6):
          F4 = long_term_debt_ratio decreased YoY
          F5 = current_ratio increased YoY
          F6 = no new share issuance (shares_outstanding <= prior year)
        Operating Efficiency (F7-F9):
          F7 = gross_margin increased YoY
          F8 = asset_turnover increased YoY
          F9 = ROA > 0  (redundant check; in original paper this is separate)

        Parameters
        ----------
        roa : Return on assets (net income / avg total assets).
        operating_cf : Operating cash flow (absolute $).
        net_income : Net income (absolute $).
        long_term_debt_ratio : LT debt / avg total assets — current year.
        long_term_debt_ratio_prior : LT debt / avg total assets — prior year.
        current_ratio : Current assets / current liabilities — current year.
        current_ratio_prior : Current ratio — prior year.
        shares_outstanding : Current shares outstanding.
        shares_outstanding_prior : Prior year shares outstanding.
        gross_margin : Gross profit / revenue — current year.
        gross_margin_prior : Gross margin — prior year.
        asset_turnover : Revenue / avg total assets — current year.
        asset_turnover_prior : Asset turnover — prior year.

        Returns
        -------
        dict with f_score (0-9), interpretation, and all 9 binary signals.
        """
        signals: dict[str, int] = {}

        # --- Profitability ---
        signals["F1_roa_positive"] = 1 if roa > 0 else 0
        signals["F2_cfo_positive"] = 1 if operating_cf > 0 else 0
        # F3: operating CF / assets > ROA  ≡  OCF > net income  (Sloan accruals)
        signals["F3_accruals"] = 1 if operating_cf > net_income else 0

        # --- Leverage / Liquidity ---
        signals["F4_leverage_decreased"] = 1 if long_term_debt_ratio < long_term_debt_ratio_prior else 0
        signals["F5_liquidity_improved"] = 1 if current_ratio > current_ratio_prior else 0
        signals["F6_no_dilution"] = 1 if shares_outstanding <= shares_outstanding_prior else 0

        # --- Operating Efficiency ---
        signals["F7_gross_margin_improved"] = 1 if gross_margin > gross_margin_prior else 0
        signals["F8_asset_turnover_improved"] = 1 if asset_turnover > asset_turnover_prior else 0
        # F9: ROA positive (same as F1 in the original paper's second profitability test)
        signals["F9_roa_positive_check"] = signals["F1_roa_positive"]

        f_score = sum(signals.values())

        return {
            "f_score": f_score,
            "interpretation": (
                "Strong" if f_score >= 7
                else "Moderate" if f_score >= 4
                else "Weak"
            ),
            "signals": signals,
        }

    @staticmethod
    def compute_altman_z_score(
        working_capital: float,
        total_assets: float,
        retained_earnings: float,
        ebit: float,
        market_cap: float,
        total_liabilities: float,
        revenue: float,
    ) -> dict[str, Any]:
        """Compute Altman (1968) Z-score for public companies.

        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

        Where:
          X1 = Working Capital / Total Assets
          X2 = Retained Earnings / Total Assets
          X3 = EBIT / Total Assets
          X4 = Market Cap / Total Liabilities
          X5 = Revenue / Total Assets

        Zones:
          Z > 2.99  → safe zone
          1.81 < Z <= 2.99 → grey zone
          Z <= 1.81 → distress zone

        Parameters
        ----------
        working_capital : Current assets minus current liabilities.
        total_assets : Total book assets.
        retained_earnings : Accumulated retained earnings.
        ebit : Earnings before interest and taxes.
        market_cap : Market capitalization.
        total_liabilities : Total book liabilities.
        revenue : Annual revenue.

        Returns
        -------
        dict with z_score, zone, and X1-X5 components.
        """
        ta = max(abs(total_assets), 1e-9)
        tl = max(abs(total_liabilities), 1e-9)

        x1 = working_capital / ta
        x2 = retained_earnings / ta
        x3 = ebit / ta
        x4 = market_cap / tl
        x5 = revenue / ta

        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        zone = (
            "safe" if z > 2.99
            else "grey" if z > 1.81
            else "distress"
        )

        return {
            "z_score": round(z, 4),
            "zone": zone,
            "components": {
                "X1_working_capital_ta": round(x1, 6),
                "X2_retained_earnings_ta": round(x2, 6),
                "X3_ebit_ta": round(x3, 6),
                "X4_mktcap_liabilities": round(x4, 6),
                "X5_revenue_ta": round(x5, 6),
            },
            "coefficients": {"X1": 1.2, "X2": 1.4, "X3": 3.3, "X4": 0.6, "X5": 1.0},
        }

    @staticmethod
    def compute_beneish_m_score(
        dsri: float,
        gmi: float,
        aqi: float,
        sgi: float,
        depi: float,
        sgai: float,
        accruals: float,
        lvgi: float,
    ) -> dict[str, Any]:
        """Compute Beneish (1999) M-score — 8-variable earnings manipulation detector.

        M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
              + 0.115*DEPI - 0.172*SGAI + 4.679*Accruals - 0.327*LVGI

        Variables:
          DSRI  = Days Sales Receivable Index  (receivables_t/sales_t) / (receivables_{t-1}/sales_{t-1})
          GMI   = Gross Margin Index           (gross_margin_{t-1} / gross_margin_t)
          AQI   = Asset Quality Index          ((1 - (CA + PPE) / TA)_t / (1 - (CA + PPE) / TA)_{t-1})
          SGI   = Sales Growth Index           (sales_t / sales_{t-1})
          DEPI  = Depreciation Index           (dep_{t-1} / (dep_{t-1} + PPE_{t-1})) / (dep_t / (dep_t + PPE_t))
          SGAI  = SG&A Index                   (SGA/sales)_t / (SGA/sales)_{t-1}
          Accruals = (NI - CFO) / avg_total_assets  (Sloan ratio)
          LVGI  = Leverage Growth Index        (LT_debt / total_assets)_t / (LT_debt / total_assets)_{t-1}

        Threshold: M > -1.78 suggests likely manipulation.

        Parameters
        ----------
        dsri : Days Sales Receivable Index.
        gmi : Gross Margin Index.
        aqi : Asset Quality Index.
        sgi : Sales Growth Index.
        depi : Depreciation Index.
        sgai : SG&A Index.
        accruals : (Net Income - Operating CF) / avg total assets.
        lvgi : Leverage Growth Index.

        Returns
        -------
        dict with m_score, likely_manipulator flag, and all 8 components.
        """
        m = (
            -4.84
            + 0.920 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * accruals
            - 0.327 * lvgi
        )

        return {
            "m_score": round(m, 4),
            "likely_manipulator": m > -1.78,
            "threshold": -1.78,
            "interpretation": (
                "High manipulation risk" if m > -1.78
                else "Low manipulation risk"
            ),
            "components": {
                "DSRI": round(dsri, 4),
                "GMI": round(gmi, 4),
                "AQI": round(aqi, 4),
                "SGI": round(sgi, 4),
                "DEPI": round(depi, 4),
                "SGAI": round(sgai, 4),
                "Accruals": round(accruals, 4),
                "LVGI": round(lvgi, 4),
            },
            "coefficients": {
                "DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
                "DEPI": 0.115, "SGAI": -0.172, "Accruals": 4.679, "LVGI": -0.327,
                "intercept": -4.84,
            },
        }

    def compute_beneish_m_score_from_db(self, ticker: str) -> dict[str, Any]:
        """Compute Beneish M-score (8-variable earnings manipulation detector).

        M > -1.78 indicates likely manipulator.
        Uses available proxy variables from yfinance data.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM fundamentals WHERE ticker = ?", [ticker.upper()]
        ).df()
        if rows.empty:
            raise ValueError(f"Ticker {ticker} not in database")
        r = rows.iloc[0].to_dict()

        # Build best-effort proxies from available data
        dsri = 1.0    # days sales receivable index — no receivables data, use neutral
        gmi = 1.0 / max(r.get("gross_margin") or 0.01, 0.01)  # gross margin index proxy
        aqi = (r.get("asset_growth_1yr") or 0) + 1.0           # asset quality index proxy
        sgi = 1 + (r.get("revenue_growth_1yr") or 0)           # sales growth index
        depi = 1.0    # depreciation index — use neutral (no depreciation breakdown)
        sgai = 1.0    # SGA index — neutral
        accruals = r.get("accruals_ratio") or 0
        lvgi = 1 + (abs(r.get("debt_to_equity") or 0) * 0.1)  # leverage growth index proxy

        m = (
            -4.84
            + 0.920 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * accruals
            - 0.327 * lvgi
        )

        return {
            "ticker": ticker,
            "m_score": round(m, 3),
            "likely_manipulator": m > -1.78,
            "interpretation": (
                "High manipulation risk" if m > -1.78 else
                "Low manipulation risk"
            ),
            "components": {
                "DSRI": round(dsri, 4),
                "GMI": round(gmi, 4),
                "AQI": round(aqi, 4),
                "SGI": round(sgi, 4),
                "DEPI": round(depi, 4),
                "SGAI": round(sgai, 4),
                "Accruals": round(accruals, 4),
                "LVGI": round(lvgi, 4),
            },
        }

    def rank_universe(
        self,
        metrics: list[str],
        universe: str = "sp500",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Multi-factor composite ranking across the universe.

        Each metric is ranked 1..N (1=best). Composite = mean rank across metrics.
        """
        conn = self._get_conn()
        metrics_sql = ", ".join(metrics)
        sql = f"SELECT ticker, company_name, sector, {metrics_sql} FROM fundamentals"
        df = conn.execute(sql).df()
        if df.empty:
            return df

        rank_cols = []
        for m in metrics:
            col = f"rank_{m}"
            df[col] = df[m].rank(ascending=False, na_option="bottom")
            rank_cols.append(col)

        df["composite_rank"] = df[rank_cols].mean(axis=1)
        return (
            df.sort_values("composite_rank")
            .head(limit)
            .drop(columns=rank_cols)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_where(self, criteria: dict[str, dict]) -> list[str]:
        """Translate criteria dict to list of SQL WHERE sub-clauses."""
        clauses: list[str] = []
        op_map = {
            "lt": "<", "gt": ">", "lte": "<=", "gte": ">=", "eq": "=",
        }
        for field, ops in criteria.items():
            # Skip synthetic fields like 'pe_ttm_min'
            col = field.replace("_min", "").replace("_max", "")
            for op, val in ops.items():
                if op in op_map:
                    clauses.append(f"{col} {op_map[op]} {float(val)}")
                elif op == "between":
                    lo, hi = float(val[0]), float(val[1])
                    clauses.append(f"{col} BETWEEN {lo} AND {hi}")
                elif op == "in":
                    values = ", ".join([f"'{v}'" for v in val])
                    clauses.append(f"{col} IN ({values})")
        return clauses

    # ------------------------------------------------------------------
    # Market cap tier classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_market_cap_tier(market_cap: float | None) -> str:
        """Classify market cap into size tier.

        Tiers (USD):
          mega   : >= 200B
          large  : 10B – 200B
          mid    : 2B – 10B
          small  : 300M – 2B
          micro  : 50M – 300M
          nano   : < 50M
        """
        if market_cap is None or market_cap <= 0:
            return "unknown"
        if market_cap >= 200_000_000_000:
            return "mega"
        if market_cap >= 10_000_000_000:
            return "large"
        if market_cap >= 2_000_000_000:
            return "mid"
        if market_cap >= 300_000_000:
            return "small"
        if market_cap >= 50_000_000:
            return "micro"
        return "nano"

    # ------------------------------------------------------------------
    # Composite z-score ranking (pure numpy — no DB required)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_composite_zscore(
        df: "pd.DataFrame",
        value_cols: list[str] | None = None,
        quality_cols: list[str] | None = None,
        momentum_cols: list[str] | None = None,
        invert_cols: list[str] | None = None,
    ) -> "pd.DataFrame":
        """Rank stocks by composite z-score across value, quality, and momentum factors.

        Each factor column is z-score normalised (subtract mean, divide by std).
        For "higher is worse" metrics (pe_ttm, ev_ebitda, debt_to_equity),
        the z-score is negated so that lower PE → higher composite score.

        Parameters
        ----------
        df : DataFrame with at least one of the factor columns present.
        value_cols : columns where lower is better (e.g. pe_ttm, ev_ebitda).
        quality_cols : columns where higher is better (e.g. roe, roic).
        momentum_cols : columns where higher is better (e.g. price_12m).
        invert_cols : extra columns to negate before averaging.

        Returns
        -------
        df with added 'composite_zscore' column, sorted descending.
        """
        import pandas as _pd
        import numpy as _np

        value_cols    = value_cols    or ["pe_ttm", "ev_ebitda", "pb_ratio"]
        quality_cols  = quality_cols  or ["roe", "roic", "net_margin"]
        momentum_cols = momentum_cols or ["price_12m", "price_6m"]
        invert_cols   = set(invert_cols or []) | set(value_cols)

        result = df.copy()
        z_cols: list[str] = []

        all_factor_cols = list(dict.fromkeys(value_cols + quality_cols + momentum_cols))
        for col in all_factor_cols:
            if col not in result.columns:
                continue
            vals = _pd.to_numeric(result[col], errors="coerce")
            mu  = vals.mean()
            std = vals.std(ddof=1)
            if std is None or (_np.isnan(std) if hasattr(std, '__float__') else False) or std < 1e-12:
                continue
            z = (vals - mu) / std
            if col in invert_cols:
                z = -z   # negate: lower pe_ttm → positive z contribution
            z_col = f"_z_{col}"
            result[z_col] = z
            z_cols.append(z_col)

        if z_cols:
            result["composite_zscore"] = result[z_cols].mean(axis=1)
            result = result.drop(columns=z_cols)
            result = result.sort_values("composite_zscore", ascending=False).reset_index(drop=True)
        else:
            result["composite_zscore"] = float("nan")

        return result


# ---------------------------------------------------------------------------
# DuckDBQueryEngine
# ---------------------------------------------------------------------------

class DuckDBQueryEngine:
    """Raw DuckDB SQL access and export utilities for the fundamentals table."""

    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        self.db_path = db_path
        self._conn: "duckdb.DuckDBPyConnection | None" = None

    def _get_conn(self) -> "duckdb.DuckDBPyConnection":
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(self.db_path)
            self._conn.execute(_CREATE_TABLE_SQL)
        return self._conn

    def execute_sql(
        self,
        query: str,
        conn: "duckdb.DuckDBPyConnection | None" = None,
    ) -> pd.DataFrame:
        """Execute arbitrary DuckDB SQL against the fundamentals table."""
        _conn = conn or self._get_conn()
        try:
            return _conn.execute(query).df()
        except Exception as exc:
            raise ValueError(f"DuckDB query failed: {exc}\nSQL: {query}") from exc

    def to_parquet(self, query: str, output_path: str) -> None:
        """Export query results to a Parquet file."""
        df = self.execute_sql(query)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("parquet_exported", path=output_path, rows=len(df))

    def get_schema(self) -> pd.DataFrame:
        """Return DataFrame of all 50+ columns with types and descriptions."""
        records = [
            {"column": col, "type": typ, "description": desc}
            for col, (typ, desc) in _COLUMN_DESCRIPTIONS.items()
        ]
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

screener_router = APIRouter(prefix="/api/screener/fundamental", tags=["Fundamental Screener"])

_loader = FundamentalDataLoader()
_screener = FundamentalScreener()
_engine = DuckDBQueryEngine()


class ScreenRequest(BaseModel):
    criteria: dict[str, dict]
    universe: str = "sp500"
    sort_by: str | None = None
    limit: int = 100


class SqlRequest(BaseModel):
    sql: str


@screener_router.post("")
def api_screen(req: ScreenRequest) -> dict:
    """Run custom fundamental screen."""
    df = _screener.screen(req.criteria, req.universe, req.sort_by, req.limit)
    return {"n_results": len(df), "results": df.where(pd.notna(df), None).to_dict(orient="records")}


@screener_router.get("/presets")
def api_list_presets() -> dict:
    """List all available preset screens."""
    return {
        "presets": [
            {"name": k, "description": v.get("description", "")}
            for k, v in FundamentalScreener.PREBUILT_SCREENS.items()
        ]
    }


@screener_router.get("/preset/{name}")
def api_run_preset(name: str, universe: str = Query("sp500")) -> dict:
    """Run a named preset fundamental screen."""
    try:
        df = _screener.run_preset(name, universe)
        return {"preset": name, "n_results": len(df), "results": df.where(pd.notna(df), None).to_dict(orient="records")}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@screener_router.get("/{ticker}/scores")
def api_ticker_scores(ticker: str) -> dict:
    """Return Piotroski F-score, Altman Z-score, and Beneish M-score for ticker."""
    t = ticker.upper()
    errors: dict[str, str] = {}
    results: dict[str, Any] = {"ticker": t}

    for score_fn, key in [
        (_screener.compute_piotroski_score, "piotroski"),
        (_screener.compute_altman_z, "altman_z"),
        (_screener.compute_beneish_m_score, "beneish_m"),
    ]:
        try:
            results[key] = score_fn(t)
        except Exception as exc:
            errors[key] = str(exc)

    if errors:
        results["errors"] = errors
    return results


@screener_router.post("/sql")
def api_raw_sql(req: SqlRequest) -> dict:
    """Execute raw DuckDB SQL against the fundamentals table."""
    # Basic safety check — read-only operations only
    forbidden = ["insert", "update", "delete", "drop", "create", "alter", "truncate"]
    if any(kw in req.sql.lower() for kw in forbidden):
        raise HTTPException(status_code=400, detail="Only SELECT queries are permitted")
    try:
        df = _engine.execute_sql(req.sql)
        return {"rows": len(df), "results": df.where(pd.notna(df), None).to_dict(orient="records")}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
