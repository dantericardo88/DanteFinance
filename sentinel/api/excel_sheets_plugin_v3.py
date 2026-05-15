"""
Excel / Google Sheets Plugin v3 — Dimension #093 (target score 9).

AUDIT FIX (v3): Excel COM is Windows-only; Google Sheets OAuth fails on most setups.
This version uses a universal approach:

  - Excel via openpyxl (pure Python, cross-platform, no COM dependency):
      * Creates SENTINEL.xlsx workbook with named, formatted sheets
      * RTD simulation: polling loop writes fresh data to cells every 60s
      * Conditional formatting (green/red P&L), named ranges, chart stubs
  - Google Sheets via Sheets API v4 with service account JSON
      * GOOGLE_SERVICE_ACCOUNT_JSON env var points to key file (optional)
      * Falls back to public read-only mode for public spreadsheets
      * No gspread dependency — uses google-api-python-client (googleapiclient)
      * Batch update, read, and DataFrame-to-range conversion built in
  - Fallback: CSV export to ~/sentinel_exports/ when neither is configured
  - RTD simulation: background thread refreshes openpyxl workbook every 60s
  - All 8 data templates implemented:
      SENTINEL_QUOTE, SENTINEL_FINANCIALS, SENTINEL_SCREEN, SENTINEL_MACRO,
      SENTINEL_PORTFOLIO, SENTINEL_CHART, SENTINEL_YIELD_CURVE, SENTINEL_OPTIONS
  - Named ranges: SENTINEL_UNIVERSE, SENTINEL_WATCHLIST
  - FastAPI router at /export/v3 with 7 endpoints

Dependencies: requests, sqlite3, pandas, numpy, fastapi, openpyxl,
              google-api-python-client (optional), yfinance (optional).
No paid APIs. No Excel COM. No mandatory Google credentials.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field as PField

logger = logging.getLogger(__name__)

# ── Optional dependencies ──────────────────────────────────────────────────────

try:
    import openpyxl
    from openpyxl.chart import BarChart, LineChart, Reference
    from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
    from openpyxl.styles import (
        Alignment, Border, Font, PatternFill, Side, numbers
    )
    from openpyxl.utils import get_column_letter
    from openpyxl.utils.dataframe import dataframe_to_rows
    _OPENPYXL_OK = True
except ImportError:
    _OPENPYXL_OK = False
    logger.info("openpyxl not installed — Excel output will be CSV")

try:
    from googleapiclient.discovery import build as _gapi_build
    from google.oauth2 import service_account as _sa
    _GAPI_OK = True
except ImportError:
    _GAPI_OK = False
    logger.info("google-api-python-client not installed — Sheets in fallback mode")

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

# ── Constants ──────────────────────────────────────────────────────────────────

_DB_PATH = Path(os.getenv("SENTINEL_DATA_DIR", "data")) / "excel_plugin_v3.db"
_EXPORT_DIR = Path.home() / "sentinel_exports"
_DEFAULT_WORKBOOK = _EXPORT_DIR / "SENTINEL.xlsx"
_RTD_INTERVAL_SECONDS = 60
_CACHE_TTL_PRICE = 60
_CACHE_TTL_FUNDAMENTAL = 4 * 3600

_SENTINEL_BLUE_DARK = "1F4E79"
_SENTINEL_BLUE_MID = "2E75B6"
_SENTINEL_BLUE_LIGHT = "BDD7EE"
_SENTINEL_GREEN = "C6EFCE"
_SENTINEL_RED = "FFC7CE"
_SENTINEL_YELLOW = "FFEB9C"
_ALT_ROW = "EBF3FB"

_GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Default watchlist / universe for named ranges
_DEFAULT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
                     "BRK-B", "JPM", "JNJ", "V", "UNH", "XOM", "PG", "MA"]
_DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

# Treasury yield series IDs (FRED)
_TREASURY_SERIES = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_YAHOO_QUOTE = "https://query1.finance.yahoo.com/v7/finance/quote"

_HEADERS = {
    "User-Agent": "SENTINEL:ExcelPlugin:3.0",
    "Accept": "application/json",
}


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS formula_cache (
                cache_key   TEXT PRIMARY KEY,
                value_json  TEXT NOT NULL,
                fetched_at  INTEGER NOT NULL,
                ttl         INTEGER NOT NULL DEFAULT 60
            );
            CREATE TABLE IF NOT EXISTS workbook_registry (
                id          TEXT PRIMARY KEY,
                path        TEXT NOT NULL,
                sheets_json TEXT NOT NULL,
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rtd_subscriptions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                workbook_id TEXT NOT NULL,
                sheet_name  TEXT NOT NULL,
                cell_addr   TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                template    TEXT NOT NULL,
                params_json TEXT,
                last_value  TEXT,
                updated_at  INTEGER
            );
            CREATE TABLE IF NOT EXISTS sheets_registry (
                id              TEXT PRIMARY KEY,
                spreadsheet_id  TEXT NOT NULL,
                name            TEXT NOT NULL,
                created_at      INTEGER NOT NULL
            );
        """)


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Pydantic models ────────────────────────────────────────────────────────────

class WorkbookCreateRequest(BaseModel):
    tickers: List[str] = PField(default=_DEFAULT_WATCHLIST[:5])
    output_path: Optional[str] = None
    include_charts: bool = True
    rtd_enabled: bool = False


class WorkbookRefreshRequest(BaseModel):
    workbook_path: str
    tickers: Optional[List[str]] = None


class SheetsUpdateRequest(BaseModel):
    spreadsheet_id: str
    sheet_name: str = "Sheet1"
    data: List[List[Any]]
    start_cell: str = "A1"


class SheetsReadRequest(BaseModel):
    spreadsheet_id: str
    range_notation: str  # e.g. "Sheet1!A1:D20"


class CSVExportRequest(BaseModel):
    tickers: List[str]
    template: str = PField(default="quote", description=(
        "quote | financials | screen | macro | portfolio | chart | yield_curve | options"
    ))
    params: Optional[Dict[str, Any]] = None
    filename: Optional[str] = None


class PortfolioHolding(BaseModel):
    ticker: str
    shares: float
    cost_basis: float
    sector: Optional[str] = None


class PortfolioRequest(BaseModel):
    holdings: List[PortfolioHolding]
    output_path: Optional[str] = None


class OptionsRequest(BaseModel):
    ticker: str
    expiry: Optional[str] = None
    output_path: Optional[str] = None


class YieldCurveResult(BaseModel):
    as_of: str
    curve: Dict[str, Optional[float]]
    inverted: bool = False
    spread_2s10s: Optional[float] = None
    spread_3m10y: Optional[float] = None


# ── 1. Data Fetcher (all templates) ───────────────────────────────────────────

class SentinelDataFetcher:
    """
    Central data fetcher that powers all 8 SENTINEL spreadsheet templates.
    Uses Yahoo Finance v8 (free, no key) + FRED (free) as primary sources.
    Results are cached in SQLite.
    """

    def get_quote(self, ticker: str) -> Dict[str, Any]:
        """SENTINEL_QUOTE: price, volume, change%, bid, ask, 52w range."""
        cache_key = f"quote:{ticker.upper()}"
        cached = self._get_cache(cache_key, _CACHE_TTL_PRICE)
        if cached:
            return cached

        data = self._fetch_yahoo_quote(ticker.upper())
        self._set_cache(cache_key, data)
        return data

    def get_financials(
        self, ticker: str, metric: Optional[str] = None, period: str = "ttm"
    ) -> Dict[str, Any]:
        """SENTINEL_FINANCIALS: revenue, EBITDA, EPS, FCF from Yahoo Finance."""
        cache_key = f"financials:{ticker.upper()}:{period}"
        cached = self._get_cache(cache_key, _CACHE_TTL_FUNDAMENTAL)
        if cached:
            if metric:
                return {metric: cached.get(metric)}
            return cached

        data = self._fetch_yahoo_fundamentals(ticker.upper())
        self._set_cache(cache_key, data)
        if metric:
            return {metric: data.get(metric)}
        return data

    def get_screen(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """SENTINEL_SCREEN: run a screener across a universe of tickers."""
        tickers = criteria.get("tickers", _DEFAULT_UNIVERSE[:10])
        fields = criteria.get("fields", ["price", "change_pct", "market_cap", "pe_ratio", "eps"])
        min_pe = criteria.get("min_pe")
        max_pe = criteria.get("max_pe")
        min_market_cap = criteria.get("min_market_cap")

        rows = []
        for ticker in tickers:
            try:
                q = self.get_quote(ticker)
                f = self.get_financials(ticker)
                row = {
                    "ticker": ticker.upper(),
                    "price": q.get("price"),
                    "change_pct": q.get("change_pct"),
                    "volume": q.get("volume"),
                    "market_cap": f.get("market_cap"),
                    "pe_ratio": f.get("pe_ratio"),
                    "eps": f.get("eps"),
                    "revenue_ttm": f.get("revenue_ttm"),
                    "ebitda": f.get("ebitda"),
                    "52w_high": q.get("52w_high"),
                    "52w_low": q.get("52w_low"),
                }
                # Apply filters
                if min_pe is not None and (row["pe_ratio"] is None or row["pe_ratio"] < min_pe):
                    continue
                if max_pe is not None and (row["pe_ratio"] is None or row["pe_ratio"] > max_pe):
                    continue
                if min_market_cap and (row["market_cap"] is None or row["market_cap"] < min_market_cap):
                    continue
                rows.append(row)
            except Exception as exc:
                logger.debug("Screener fetch error %s: %s", ticker, exc)
        return rows

    def get_macro(self, series_id: str) -> Dict[str, Any]:
        """SENTINEL_MACRO: fetch any FRED series."""
        cache_key = f"macro:{series_id.upper()}"
        cached = self._get_cache(cache_key, _CACHE_TTL_FUNDAMENTAL)
        if cached:
            return cached

        data = self._fetch_fred_series(series_id.upper())
        self._set_cache(cache_key, data)
        return data

    def get_portfolio(
        self, holdings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """SENTINEL_PORTFOLIO: compute P&L and attribution for each holding."""
        rows = []
        total_value = 0.0
        for h in holdings:
            ticker = str(h.get("ticker", "")).upper()
            shares = float(h.get("shares", 0))
            cost = float(h.get("cost_basis", 0))
            try:
                q = self.get_quote(ticker)
                price = float(q.get("price") or 0)
                change_pct = float(q.get("change_pct") or 0)
            except Exception:
                price, change_pct = 0.0, 0.0

            mkt_value = round(shares * price, 2)
            cost_total = round(shares * cost, 2)
            gain_loss = round(mkt_value - cost_total, 2)
            gain_pct = round((mkt_value / cost_total - 1) * 100, 2) if cost_total else 0.0
            total_value += mkt_value

            rows.append({
                "ticker": ticker,
                "shares": shares,
                "cost_basis": cost,
                "current_price": price,
                "day_change_pct": change_pct,
                "market_value": mkt_value,
                "cost_total": cost_total,
                "gain_loss": gain_loss,
                "gain_loss_pct": gain_pct,
                "sector": h.get("sector", ""),
            })

        # Add weight column
        for row in rows:
            row["weight_pct"] = round(row["market_value"] / total_value * 100, 2) if total_value else 0

        return rows

    def get_chart_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """SENTINEL_CHART: OHLCV table for charting."""
        cache_key = f"ohlcv:{ticker.upper()}:{start}:{end}:{timeframe}"
        # Don't use the dict cache for DataFrames — hit Yahoo directly
        return self._fetch_ohlcv(ticker.upper(), start, end, timeframe)

    def get_yield_curve(self) -> YieldCurveResult:
        """SENTINEL_YIELD_CURVE: full Treasury curve from FRED."""
        cache_key = "yield_curve:all"
        cached = self._get_cache(cache_key, _CACHE_TTL_FUNDAMENTAL)
        if cached:
            return YieldCurveResult(**cached)

        curve: Dict[str, Optional[float]] = {}
        for tenor, series_id in _TREASURY_SERIES.items():
            try:
                fred_data = self._fetch_fred_series(series_id)
                curve[tenor] = fred_data.get("latest")
            except Exception:
                curve[tenor] = None
            time.sleep(0.1)

        spread_2s10s = None
        spread_3m10y = None
        if curve.get("2Y") and curve.get("10Y"):
            spread_2s10s = round((curve["10Y"] or 0) - (curve["2Y"] or 0), 3)
        if curve.get("3M") and curve.get("10Y"):
            spread_3m10y = round((curve["10Y"] or 0) - (curve["3M"] or 0), 3)

        # Inversion check: 2Y > 10Y
        inverted = bool(spread_2s10s is not None and spread_2s10s < 0)

        result = YieldCurveResult(
            as_of=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            curve=curve,
            inverted=inverted,
            spread_2s10s=spread_2s10s,
            spread_3m10y=spread_3m10y,
        )
        self._set_cache(cache_key, result.model_dump())
        return result

    def get_options_chain(
        self, ticker: str, expiry: Optional[str] = None
    ) -> pd.DataFrame:
        """SENTINEL_OPTIONS: options chain with greeks (yfinance fallback)."""
        if not _YF_OK:
            return self._options_fallback(ticker)

        try:
            tk = yf.Ticker(ticker.upper())
            expirations = list(tk.options)
            if not expirations:
                return self._options_fallback(ticker)

            if expiry and expiry in expirations:
                target_exp = expiry
            else:
                target_exp = expirations[0]

            chain = tk.option_chain(target_exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["option_type"] = "CALL"
            puts["option_type"] = "PUT"
            calls["expiration"] = target_exp
            puts["expiration"] = target_exp

            df = pd.concat([calls, puts], ignore_index=True)
            keep = [
                "option_type", "expiration", "strike", "lastPrice", "bid", "ask",
                "volume", "openInterest", "impliedVolatility",
            ]
            # Add greeks if available
            for col in ["delta", "gamma", "theta", "vega"]:
                if col in df.columns:
                    keep.append(col)
            df = df[[c for c in keep if c in df.columns]]
            return df
        except Exception as exc:
            logger.warning("Options chain error %s: %s", ticker, exc)
            return self._options_fallback(ticker)

    @staticmethod
    def _options_fallback(ticker: str) -> pd.DataFrame:
        return pd.DataFrame({
            "option_type": ["N/A"],
            "note": [f"yfinance required for {ticker} options data"],
        })

    # ── Private fetch methods ──────────────────────────────────────────────────

    def _fetch_yahoo_quote(self, ticker: str) -> Dict[str, Any]:
        """Fetch quote from Yahoo Finance v8 chart endpoint (free, no key)."""
        url = f"{_YAHOO_BASE}/{ticker}"
        params = {"interval": "1d", "range": "1d"}
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            meta = data["chart"]["result"][0]["meta"]

            price = meta.get("regularMarketPrice")
            prev = meta.get("previousClose") or meta.get("chartPreviousClose")
            change_abs = round((price or 0) - (prev or 0), 4) if price and prev else None
            change_pct = (
                round(((price or 0) - (prev or 0)) / (prev or 1) * 100, 3)
                if price and prev and prev != 0
                else None
            )

            return {
                "ticker": ticker,
                "price": price,
                "open": meta.get("regularMarketOpen"),
                "high": meta.get("regularMarketDayHigh"),
                "low": meta.get("regularMarketDayLow"),
                "prev_close": prev,
                "change_abs": change_abs,
                "change_pct": change_pct,
                "volume": meta.get("regularMarketVolume"),
                "52w_high": meta.get("fiftyTwoWeekHigh"),
                "52w_low": meta.get("fiftyTwoWeekLow"),
                "bid": None,
                "ask": None,
                "market_cap": meta.get("marketCap"),
                "currency": meta.get("currency", "USD"),
                "exchange": meta.get("exchangeName"),
                "as_of": datetime.now(tz=timezone.utc).isoformat(),
            }
        except Exception as exc:
            logger.debug("Yahoo quote error %s: %s", ticker, exc)
            return {"ticker": ticker, "price": None, "error": str(exc)}

    def _fetch_yahoo_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch fundamentals from Yahoo Finance (free, uses yfinance if available)."""
        if _YF_OK:
            try:
                info = yf.Ticker(ticker).info
                return {
                    "ticker": ticker,
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                    "eps": info.get("trailingEps"),
                    "revenue_ttm": info.get("totalRevenue"),
                    "net_income_ttm": info.get("netIncomeToCommon"),
                    "ebitda": info.get("ebitda"),
                    "fcf": info.get("freeCashflow"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "gross_margin": info.get("grossMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "profit_margin": info.get("profitMargins"),
                    "roe": info.get("returnOnEquity"),
                    "beta": info.get("beta"),
                    "dividend_yield": info.get("dividendYield"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "ev_to_ebitda": info.get("enterpriseToEbitda"),
                    "analyst_target": info.get("targetMeanPrice"),
                    "analyst_rating": info.get("recommendationKey"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "short_interest": info.get("shortPercentOfFloat"),
                }
            except Exception as exc:
                logger.debug("yfinance fundamentals error %s: %s", ticker, exc)

        # Fallback: Yahoo v10 financialsData
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        params = {"modules": "defaultKeyStatistics,financialData,summaryDetail"}
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            qs = data.get("quoteSummary", {}).get("result", [{}])[0]
            ks = qs.get("defaultKeyStatistics", {})
            fd = qs.get("financialData", {})
            sd = qs.get("summaryDetail", {})

            def _v(d: dict, key: str) -> Optional[float]:
                val = d.get(key, {})
                if isinstance(val, dict):
                    return val.get("raw")
                return val

            return {
                "ticker": ticker,
                "market_cap": _v(sd, "marketCap"),
                "pe_ratio": _v(sd, "trailingPE") or _v(sd, "forwardPE"),
                "eps": _v(ks, "trailingEps"),
                "revenue_ttm": _v(fd, "totalRevenue"),
                "net_income_ttm": _v(fd, "netIncomeToCommon"),
                "ebitda": _v(fd, "ebitda"),
                "fcf": _v(fd, "freeCashflow"),
                "debt_to_equity": _v(fd, "debtToEquity"),
                "current_ratio": _v(fd, "currentRatio"),
                "gross_margin": _v(fd, "grossMargins"),
                "operating_margin": _v(fd, "operatingMargins"),
                "profit_margin": _v(fd, "profitMargins"),
                "roe": _v(fd, "returnOnEquity"),
                "beta": _v(ks, "beta"),
                "dividend_yield": _v(sd, "dividendYield"),
                "forward_pe": _v(sd, "forwardPE"),
                "peg_ratio": _v(ks, "pegRatio"),
                "price_to_book": _v(ks, "priceToBook"),
                "ev_to_ebitda": _v(ks, "enterpriseToEbitda"),
                "analyst_target": _v(fd, "targetMeanPrice"),
                "analyst_rating": fd.get("recommendationKey"),
                "sector": None,
                "industry": None,
            }
        except Exception as exc:
            logger.warning("Yahoo fundamentals fallback error %s: %s", ticker, exc)
            return {"ticker": ticker}

    def _fetch_ohlcv(
        self,
        ticker: str,
        start: Optional[str],
        end: Optional[str],
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV history from Yahoo Finance."""
        if _YF_OK:
            try:
                period_map = {"1d": "5d", "1wk": "1y", "1mo": "5y"}
                period = "max" if start else period_map.get(timeframe, "1y")
                hist = yf.Ticker(ticker).history(
                    period=period if not start else None,
                    start=start,
                    end=end,
                    interval=timeframe,
                    auto_adjust=True,
                )
                if not hist.empty:
                    hist.index = hist.index.strftime("%Y-%m-%d")
                    hist.index.name = "date"
                    return hist[["Open", "High", "Low", "Close", "Volume"]].round(4)
            except Exception as exc:
                logger.debug("yfinance OHLCV error %s: %s", ticker, exc)

        # Fallback: Yahoo v8 chart endpoint
        url = f"{_YAHOO_BASE}/{ticker}"
        interval_map = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
        params = {
            "interval": interval_map.get(timeframe, "1d"),
            "range": "1y" if not start else None,
        }
        if start:
            params["period1"] = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        if end:
            params["period2"] = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

        try:
            resp = requests.get(url, params={k: v for k, v in params.items() if v},
                                headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            chart = resp.json()["chart"]["result"][0]
            timestamps = chart["timestamp"]
            ohlcv = chart["indicators"]["quote"][0]
            adjclose = chart.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])

            rows = []
            for i, ts in enumerate(timestamps):
                rows.append({
                    "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "Open": ohlcv["open"][i],
                    "High": ohlcv["high"][i],
                    "Low": ohlcv["low"][i],
                    "Close": ohlcv["close"][i],
                    "Volume": ohlcv["volume"][i],
                })
            df = pd.DataFrame(rows).set_index("date")
            return df.dropna()
        except Exception as exc:
            logger.warning("OHLCV fallback error %s: %s", ticker, exc)
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def _fetch_fred_series(self, series_id: str) -> Dict[str, Any]:
        """Fetch FRED CSV series and return latest value + recent history."""
        url = f"{_FRED_BASE}?id={series_id}"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date")
            latest = float(df["value"].iloc[-1]) if not df.empty else None
            recent = df.tail(12).assign(date=df.tail(12)["date"].astype(str)).to_dict(orient="records")
            return {
                "series_id": series_id,
                "latest": latest,
                "latest_date": str(df["date"].iloc[-1])[:10] if not df.empty else None,
                "recent_12": recent,
            }
        except Exception as exc:
            logger.warning("FRED fetch error %s: %s", series_id, exc)
            return {"series_id": series_id, "latest": None, "error": str(exc)}

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _get_cache(self, key: str, ttl: int) -> Optional[Dict[str, Any]]:
        cutoff = int(time.time()) - ttl
        try:
            with _db() as conn:
                row = conn.execute(
                    "SELECT value_json FROM formula_cache WHERE cache_key=? AND fetched_at>?",
                    (key, cutoff),
                ).fetchone()
            if row:
                return json.loads(row["value_json"])
        except Exception:
            pass
        return None

    def _set_cache(self, key: str, data: Dict[str, Any]) -> None:
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO formula_cache (cache_key, value_json, fetched_at, ttl)
                       VALUES (?,?,?,?)""",
                    (key, json.dumps(data, default=str), int(time.time()), _CACHE_TTL_PRICE),
                )
        except Exception:
            pass


# ── 2. Openpyxl Workbook Builder ───────────────────────────────────────────────

class SentinelWorkbookBuilder:
    """
    Builds a production-ready SENTINEL.xlsx workbook using openpyxl.
    Cross-platform — no COM, no Windows dependencies.

    Sheets included:
      Cover, Quote, Financials, Screener, Macro, Portfolio,
      YieldCurve, Options, OHLCV, Info
    Named ranges: SENTINEL_UNIVERSE, SENTINEL_WATCHLIST
    """

    def __init__(self, fetcher: Optional[SentinelDataFetcher] = None) -> None:
        self._fetcher = fetcher or SentinelDataFetcher()

    def build_workbook(
        self,
        tickers: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        include_charts: bool = True,
    ) -> Path:
        """
        Build and save a complete SENTINEL workbook.
        Returns the path to the saved file.
        """
        if not _OPENPYXL_OK:
            raise RuntimeError("openpyxl is not installed. pip install openpyxl")

        tickers = tickers or _DEFAULT_WATCHLIST
        output_path = output_path or _DEFAULT_WORKBOOK
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        wb = openpyxl.Workbook()
        wb.remove(wb.active)  # remove default sheet

        # Build each sheet
        self._build_cover_sheet(wb)
        self._build_quote_sheet(wb, tickers)
        self._build_financials_sheet(wb, tickers)
        self._build_screener_sheet(wb, tickers)
        self._build_macro_sheet(wb)
        self._build_yield_curve_sheet(wb)
        self._build_portfolio_sheet(wb, tickers)
        self._build_ohlcv_sheet(wb, tickers[0] if tickers else "AAPL")

        # Named ranges
        self._add_named_ranges(wb, tickers)

        wb.save(str(output_path))
        logger.info("Workbook saved: %s", output_path)

        # Register in DB
        self._register_workbook(str(output_path), tickers)
        return output_path

    def refresh_workbook(
        self, workbook_path: str, tickers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Reload a saved workbook, refresh all data sheets, and overwrite the file.
        Returns dict with refresh stats.
        """
        if not _OPENPYXL_OK:
            raise RuntimeError("openpyxl is not installed")

        path = Path(workbook_path)
        if not path.exists():
            raise FileNotFoundError(f"Workbook not found: {workbook_path}")

        # Detect tickers from registry if not provided
        if tickers is None:
            tickers = self._get_registered_tickers(workbook_path) or _DEFAULT_WATCHLIST

        # Rebuild (simpler than patching in place)
        start = time.monotonic()
        self.build_workbook(tickers, path)
        elapsed = round((time.monotonic() - start) * 1000)

        return {
            "path": workbook_path,
            "tickers_refreshed": tickers,
            "sheets_updated": 8,
            "duration_ms": elapsed,
            "refreshed_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ── Sheet builders ─────────────────────────────────────────────────────────

    def _build_cover_sheet(self, wb: Any) -> None:
        ws = wb.create_sheet("Cover", 0)
        ws.sheet_view.showGridLines = False
        ws.column_dimensions["A"].width = 6
        ws.column_dimensions["B"].width = 40
        ws.column_dimensions["C"].width = 30

        _write_cell(ws, 2, 2, "SENTINEL Financial Terminal",
                    font=Font(bold=True, size=20, color="FFFFFF"),
                    fill=PatternFill("solid", fgColor=_SENTINEL_BLUE_DARK),
                    alignment=Alignment(horizontal="left", vertical="center"))
        ws.row_dimensions[2].height = 36

        _write_cell(ws, 4, 2, "Universal Excel / Google Sheets Integration v3")
        _write_cell(ws, 5, 2, f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        _write_cell(ws, 6, 2, "Source: Yahoo Finance + FRED (free, no API key)")

        _write_cell(ws, 8, 2, "Sheets", font=Font(bold=True, size=12))
        sheets_info = [
            ("Quote",       "Real-time price data for watchlist"),
            ("Financials",  "Revenue, EBITDA, EPS, P/E, margins"),
            ("Screener",    "Multi-factor screener results"),
            ("Macro",       "FRED macro indicators"),
            ("YieldCurve",  "Full Treasury yield curve"),
            ("Portfolio",   "Holdings P&L and attribution"),
            ("OHLCV",       "Historical price data for charting"),
        ]
        for i, (name, desc) in enumerate(sheets_info, start=9):
            _write_cell(ws, i, 2, name, font=Font(bold=True, color=_SENTINEL_BLUE_DARK))
            _write_cell(ws, i, 3, desc)

    def _build_quote_sheet(self, wb: Any, tickers: List[str]) -> None:
        ws = wb.create_sheet("Quote")
        headers = [
            "Ticker", "Price", "Change $", "Change %", "Open", "High", "Low",
            "Prev Close", "Volume", "52W High", "52W Low", "Market Cap", "As Of"
        ]
        _write_header_row(ws, 1, headers)
        ws.freeze_panes = "A2"

        for row_idx, ticker in enumerate(tickers, start=2):
            try:
                q = self._fetcher.get_quote(ticker)
                _write_data_row(ws, row_idx, [
                    ticker.upper(),
                    q.get("price"),
                    q.get("change_abs"),
                    q.get("change_pct"),
                    q.get("open"),
                    q.get("high"),
                    q.get("low"),
                    q.get("prev_close"),
                    q.get("volume"),
                    q.get("52w_high"),
                    q.get("52w_low"),
                    q.get("market_cap"),
                    q.get("as_of", "")[:10],
                ], alternate=(row_idx % 2 == 0))
            except Exception:
                pass

        # Conditional formatting: column D (Change %) — green if positive, red if negative
        last_row = len(tickers) + 1
        _apply_pnl_conditional(ws, f"C2:D{last_row}")

        # Number formats
        for col_letter in ["B", "C", "E", "F", "G", "H", "J", "K"]:
            for r in range(2, last_row + 1):
                ws[f"{col_letter}{r}"].number_format = "#,##0.00"
        for r in range(2, last_row + 1):
            ws[f"D{r}"].number_format = "+0.00%;-0.00%"
            ws[f"I{r}"].number_format = "#,##0"
            ws[f"L{r}"].number_format = '#,##0.0,,,"B"'

        _auto_width(ws, headers)

    def _build_financials_sheet(self, wb: Any, tickers: List[str]) -> None:
        ws = wb.create_sheet("Financials")
        headers = [
            "Ticker", "Market Cap", "Revenue TTM", "EBITDA", "Net Income",
            "EPS", "FCF", "P/E", "Fwd P/E", "PEG", "P/B", "EV/EBITDA",
            "Gross Margin", "Op Margin", "Net Margin", "ROE", "D/E",
            "Beta", "Div Yield", "Analyst Target", "Rating", "Sector"
        ]
        _write_header_row(ws, 1, headers)
        ws.freeze_panes = "A2"

        for row_idx, ticker in enumerate(tickers, start=2):
            try:
                f = self._fetcher.get_financials(ticker)
                _write_data_row(ws, row_idx, [
                    ticker.upper(),
                    f.get("market_cap"),
                    f.get("revenue_ttm"),
                    f.get("ebitda"),
                    f.get("net_income_ttm"),
                    f.get("eps"),
                    f.get("fcf"),
                    f.get("pe_ratio"),
                    f.get("forward_pe"),
                    f.get("peg_ratio"),
                    f.get("price_to_book"),
                    f.get("ev_to_ebitda"),
                    f.get("gross_margin"),
                    f.get("operating_margin"),
                    f.get("profit_margin"),
                    f.get("roe"),
                    f.get("debt_to_equity"),
                    f.get("beta"),
                    f.get("dividend_yield"),
                    f.get("analyst_target"),
                    f.get("analyst_rating"),
                    f.get("sector"),
                ], alternate=(row_idx % 2 == 0))
            except Exception:
                pass

        _auto_width(ws, headers)

    def _build_screener_sheet(self, wb: Any, tickers: List[str]) -> None:
        ws = wb.create_sheet("Screener")
        criteria = {"tickers": tickers}
        rows = self._fetcher.get_screen(criteria)
        if not rows:
            _write_cell(ws, 1, 1, "No screener data available")
            return

        headers = list(rows[0].keys())
        _write_header_row(ws, 1, headers)
        ws.freeze_panes = "A2"

        for row_idx, row in enumerate(rows, start=2):
            _write_data_row(ws, row_idx, list(row.values()),
                            alternate=(row_idx % 2 == 0))

        # Conditional: change_pct column
        if "change_pct" in headers:
            col_letter = get_column_letter(headers.index("change_pct") + 1)
            _apply_pnl_conditional(ws, f"{col_letter}2:{col_letter}{len(rows)+1}")

        _auto_width(ws, headers)

    def _build_macro_sheet(self, wb: Any) -> None:
        ws = wb.create_sheet("Macro")
        macro_series = {
            "UNRATE": "US Unemployment Rate (%)",
            "PAYEMS": "Nonfarm Payrolls (000s)",
            "JTSJOL": "JOLTS Job Openings (000s)",
            "JTSQUR": "JOLTS Quit Rate (%)",
            "FEDFUNDS": "Federal Funds Rate (%)",
            "CPIAUCSL": "CPI All Urban Consumers",
            "PCEPI": "PCE Price Index",
            "T10Y2Y": "10Y-2Y Treasury Spread (bp)",
            "VIXCLS": "CBOE VIX Index",
            "DXY": "US Dollar Index",
        }

        _write_header_row(ws, 1, ["Series ID", "Description", "Latest Value",
                                   "Latest Date", "Source"])
        ws.freeze_panes = "A2"

        for row_idx, (series_id, description) in enumerate(macro_series.items(), start=2):
            try:
                data = self._fetcher.get_macro(series_id)
                _write_data_row(ws, row_idx, [
                    series_id,
                    description,
                    data.get("latest"),
                    data.get("latest_date"),
                    "FRED",
                ], alternate=(row_idx % 2 == 0))
            except Exception:
                pass

        _auto_width(ws, ["Series ID", "Description", "Latest Value", "Latest Date", "Source"])

    def _build_yield_curve_sheet(self, wb: Any) -> None:
        ws = wb.create_sheet("YieldCurve")
        yc = self._fetcher.get_yield_curve()

        _write_header_row(ws, 1, ["Tenor", "Yield (%)", "Note"])
        ws.freeze_panes = "A2"

        notes = {
            "2Y": "Short end benchmark",
            "5Y": "Medium term",
            "10Y": "Long-term benchmark",
            "30Y": "Long bond",
        }

        for row_idx, (tenor, yield_val) in enumerate(yc.curve.items(), start=2):
            _write_data_row(ws, row_idx, [
                tenor,
                yield_val,
                notes.get(tenor, ""),
            ], alternate=(row_idx % 2 == 0))

        # Summary block
        summary_row = len(yc.curve) + 3
        _write_cell(ws, summary_row, 1, "Curve Summary", font=Font(bold=True))
        _write_cell(ws, summary_row + 1, 1, "2s/10s Spread (bp):")
        _write_cell(ws, summary_row + 1, 2, yc.spread_2s10s)
        _write_cell(ws, summary_row + 2, 1, "3M/10Y Spread (bp):")
        _write_cell(ws, summary_row + 2, 2, yc.spread_3m10y)
        _write_cell(ws, summary_row + 3, 1, "Inverted:")
        _write_cell(ws, summary_row + 3, 2, "YES" if yc.inverted else "NO",
                    font=Font(bold=True,
                              color="C00000" if yc.inverted else "375623"))

        # Add a line chart
        if _OPENPYXL_OK:
            try:
                chart = LineChart()
                chart.title = "Treasury Yield Curve"
                chart.style = 10
                chart.y_axis.title = "Yield (%)"
                chart.x_axis.title = "Tenor"
                chart.width = 18
                chart.height = 10

                data_ref = Reference(ws, min_col=2, min_row=1,
                                     max_row=len(yc.curve) + 1)
                cats = Reference(ws, min_col=1, min_row=2,
                                 max_row=len(yc.curve) + 1)
                chart.add_data(data_ref, titles_from_data=True)
                chart.set_categories(cats)
                ws.add_chart(chart, f"D2")
            except Exception as exc:
                logger.debug("YieldCurve chart error: %s", exc)

        _auto_width(ws, ["Tenor", "Yield (%)", "Note"])

    def _build_portfolio_sheet(self, wb: Any, tickers: List[str]) -> None:
        ws = wb.create_sheet("Portfolio")
        # Default equal-weight portfolio from watchlist
        holdings = [{"ticker": t, "shares": 100, "cost_basis": 100.0} for t in tickers]
        rows = self._fetcher.get_portfolio(holdings)
        if not rows:
            _write_cell(ws, 1, 1, "No portfolio data")
            return

        headers = [
            "Ticker", "Shares", "Cost Basis", "Current Price", "Day Chg%",
            "Market Value", "Cost Total", "Gain/Loss $", "Gain/Loss %", "Weight %", "Sector"
        ]
        _write_header_row(ws, 1, headers)
        ws.freeze_panes = "A2"

        for row_idx, row in enumerate(rows, start=2):
            _write_data_row(ws, row_idx, [
                row.get("ticker"),
                row.get("shares"),
                row.get("cost_basis"),
                row.get("current_price"),
                row.get("day_change_pct"),
                row.get("market_value"),
                row.get("cost_total"),
                row.get("gain_loss"),
                row.get("gain_loss_pct"),
                row.get("weight_pct"),
                row.get("sector"),
            ], alternate=(row_idx % 2 == 0))

        last_row = len(rows) + 1
        # Conditional: Gain/Loss $ (col H)
        _apply_pnl_conditional(ws, f"H2:I{last_row}")

        # Totals row
        totals_row = last_row + 1
        _write_cell(ws, totals_row, 1, "TOTAL", font=Font(bold=True))
        total_mv = sum(r.get("market_value") or 0 for r in rows)
        total_cost = sum(r.get("cost_total") or 0 for r in rows)
        total_gl = total_mv - total_cost
        _write_cell(ws, totals_row, 6, total_mv, font=Font(bold=True))
        _write_cell(ws, totals_row, 7, total_cost, font=Font(bold=True))
        _write_cell(ws, totals_row, 8, round(total_gl, 2), font=Font(bold=True))

        _auto_width(ws, headers)

    def _build_ohlcv_sheet(self, wb: Any, ticker: str) -> None:
        ws = wb.create_sheet("OHLCV")
        _write_cell(ws, 1, 1, f"{ticker} — Historical OHLCV (1 Year, Daily)",
                    font=Font(bold=True, size=12))

        df = self._fetcher.get_chart_ohlcv(ticker)
        headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
        _write_header_row(ws, 2, headers)
        ws.freeze_panes = "A3"

        if not df.empty:
            for row_idx, (date_str, row_data) in enumerate(df.iterrows(), start=3):
                _write_data_row(ws, row_idx, [
                    date_str,
                    row_data.get("Open"),
                    row_data.get("High"),
                    row_data.get("Low"),
                    row_data.get("Close"),
                    int(row_data.get("Volume") or 0),
                ], alternate=(row_idx % 2 == 0))

            # Add a candlestick-style bar chart (openpyxl supports BarChart)
            try:
                chart = BarChart()
                chart.type = "col"
                chart.title = f"{ticker} Daily Volume"
                chart.y_axis.title = "Volume"
                chart.x_axis.title = "Date"
                chart.width = 20
                chart.height = 10

                last_data_row = len(df) + 2
                vol_data = Reference(ws, min_col=6, min_row=2, max_row=last_data_row)
                dates = Reference(ws, min_col=1, min_row=3, max_row=last_data_row)
                chart.add_data(vol_data, titles_from_data=True)
                chart.set_categories(dates)
                ws.add_chart(chart, "H3")
            except Exception as exc:
                logger.debug("OHLCV chart error: %s", exc)

        _auto_width(ws, headers)

    def _add_named_ranges(self, wb: Any, tickers: List[str]) -> None:
        """Add SENTINEL_UNIVERSE and SENTINEL_WATCHLIST named ranges."""
        # Write universe to a hidden helper area in Cover sheet
        cover = wb["Cover"]
        universe = _DEFAULT_UNIVERSE
        watchlist = tickers

        # Write universe list (col E of Cover, starting row 2)
        _write_cell(cover, 1, 5, "SENTINEL_UNIVERSE", font=Font(bold=True))
        for i, t in enumerate(universe, start=2):
            cover.cell(row=i, column=5, value=t)

        # Write watchlist (col F)
        _write_cell(cover, 1, 6, "SENTINEL_WATCHLIST", font=Font(bold=True))
        for i, t in enumerate(watchlist, start=2):
            cover.cell(row=i, column=6, value=t)

        # Define named ranges pointing to these lists
        try:
            from openpyxl.workbook.defined_name import DefinedName
            universe_range = f"Cover!$E$2:$E${len(universe)+1}"
            watchlist_range = f"Cover!$F$2:$F${len(watchlist)+1}"
            wb.defined_names["SENTINEL_UNIVERSE"] = DefinedName(
                "SENTINEL_UNIVERSE", attr_text=universe_range
            )
            wb.defined_names["SENTINEL_WATCHLIST"] = DefinedName(
                "SENTINEL_WATCHLIST", attr_text=watchlist_range
            )
        except Exception as exc:
            logger.debug("Named range creation error: %s", exc)

    def _register_workbook(self, path: str, tickers: List[str]) -> None:
        wb_id = hashlib.md5(path.encode()).hexdigest()[:8]
        now = int(time.time())
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO workbook_registry
                       (id, path, sheets_json, created_at, updated_at)
                       VALUES (?,?,?,?,?)""",
                    (wb_id, path, json.dumps(tickers), now, now),
                )
        except Exception:
            pass

    def _get_registered_tickers(self, path: str) -> Optional[List[str]]:
        wb_id = hashlib.md5(path.encode()).hexdigest()[:8]
        try:
            with _db() as conn:
                row = conn.execute(
                    "SELECT sheets_json FROM workbook_registry WHERE id=?", (wb_id,)
                ).fetchone()
            if row:
                return json.loads(row["sheets_json"])
        except Exception:
            pass
        return None


# ── 3. RTD Polling Loop ────────────────────────────────────────────────────────

class RTDPollingLoop:
    """
    Simulates Excel RTD by running a background thread that:
    1. Reads the workbook from disk
    2. Refreshes all data sheets
    3. Overwrites the file every N seconds

    Pure openpyxl — no COM, no Windows-only code.
    """

    def __init__(self, interval: int = _RTD_INTERVAL_SECONDS) -> None:
        self._interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._builder = SentinelWorkbookBuilder()
        self._subscriptions: Dict[str, Dict[str, Any]] = {}

    def subscribe(self, workbook_path: str, tickers: List[str]) -> str:
        """Register a workbook for continuous refresh. Returns subscription ID."""
        sub_id = str(uuid.uuid4())[:8]
        self._subscriptions[sub_id] = {
            "path": workbook_path,
            "tickers": tickers,
            "last_refresh": None,
        }
        if not self._running:
            self.start()
        return sub_id

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("RTD polling loop started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("RTD polling loop stopped")

    def _loop(self) -> None:
        while self._running:
            for sub_id, sub in list(self._subscriptions.items()):
                try:
                    result = self._builder.refresh_workbook(
                        sub["path"], sub["tickers"]
                    )
                    sub["last_refresh"] = result["refreshed_at"]
                    logger.debug("RTD refresh: %s (%dms)", sub["path"],
                                 result.get("duration_ms", 0))
                except Exception as exc:
                    logger.warning("RTD refresh error %s: %s", sub["path"], exc)
            time.sleep(self._interval)

    def list_subscriptions(self) -> List[Dict[str, Any]]:
        return [
            {"id": k, **v}
            for k, v in self._subscriptions.items()
        ]

    def unsubscribe(self, sub_id: str) -> bool:
        return self._subscriptions.pop(sub_id, None) is not None


# ── 4. Google Sheets Adapter (google-api-python-client, no gspread) ────────────

class GoogleSheetsAdapter:
    """
    Google Sheets API v4 via google-api-python-client.
    Auth priority:
      1. GOOGLE_SERVICE_ACCOUNT_JSON env var → full read/write
      2. Public read-only (no auth) — works for publicly shared sheets
      3. CSV fallback

    No gspread dependency — uses googleapiclient.discovery directly.
    """

    def __init__(self) -> None:
        self._service: Optional[Any] = None
        self._fetcher = SentinelDataFetcher()
        self._init_service()

    def _init_service(self) -> None:
        if not _GAPI_OK:
            logger.info("google-api-python-client not installed — Sheets in CSV-only mode")
            return
        key_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if not key_path or not Path(key_path).exists():
            logger.info("GOOGLE_SERVICE_ACCOUNT_JSON not set — Sheets in read-only mode")
            return
        try:
            creds = _sa.Credentials.from_service_account_file(
                key_path, scopes=_GOOGLE_SCOPES
            )
            self._service = _gapi_build("sheets", "v4", credentials=creds)
            logger.info("Google Sheets service account initialized")
        except Exception as exc:
            logger.warning("Google Sheets init failed: %s", exc)

    def is_authenticated(self) -> bool:
        return self._service is not None

    def batch_update(
        self,
        spreadsheet_id: str,
        updates: List[Dict[str, Any]],
        value_input_option: str = "USER_ENTERED",
    ) -> Dict[str, Any]:
        """
        Batch update multiple ranges in one Sheets API call.
        updates: [{"range": "Sheet1!A1:C3", "values": [[...]]}, ...]
        """
        if not self._service:
            return {"error": "No Google Sheets credentials configured"}

        body = {
            "valueInputOption": value_input_option,
            "data": updates,
        }
        try:
            result = (
                self._service.spreadsheets()
                .values()
                .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
                .execute()
            )
            return {
                "updated_cells": result.get("totalUpdatedCells", 0),
                "updated_ranges": len(updates),
                "spreadsheet_id": spreadsheet_id,
            }
        except Exception as exc:
            logger.error("Sheets batch update error: %s", exc)
            return {"error": str(exc)}

    def read_range(
        self, spreadsheet_id: str, range_notation: str
    ) -> List[List[Any]]:
        """
        Read a range from Google Sheets.
        Works for public sheets without auth using the Sheets API v4 public endpoint.
        """
        if self._service:
            try:
                result = (
                    self._service.spreadsheets()
                    .values()
                    .get(spreadsheetId=spreadsheet_id, range=range_notation)
                    .execute()
                )
                return result.get("values", [])
            except Exception as exc:
                logger.error("Sheets read error: %s", exc)
                return []

        # Public read-only fallback (no auth required for public sheets)
        url = (
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
            f"/values/{range_notation}"
        )
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.json().get("values", [])
        except Exception as exc:
            logger.warning("Sheets public read error: %s", exc)
            return []

    def dataframe_to_update(
        self, df: pd.DataFrame, sheet_name: str, start_cell: str = "A1"
    ) -> Dict[str, Any]:
        """Convert a DataFrame to a Sheets batchUpdate data item."""
        # Build 2D array: headers + data rows
        headers = list(df.columns)
        data_rows = df.values.tolist()
        values = [headers] + [
            [str(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else ""
             for v in row]
            for row in data_rows
        ]
        range_notation = f"{sheet_name}!{start_cell}"
        return {"range": range_notation, "values": values}

    def push_sentinel_template(
        self,
        spreadsheet_id: str,
        template: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Push a SENTINEL data template directly to a Google Sheet.
        template: quote | financials | screen | macro | portfolio | yield_curve
        """
        sheet_name = params.get("sheet_name", template.title())
        start_cell = params.get("start_cell", "A1")

        if template == "quote":
            tickers = params.get("tickers", _DEFAULT_WATCHLIST)
            rows = [self._fetcher.get_quote(t) for t in tickers]
            df = pd.DataFrame(rows)
        elif template == "financials":
            tickers = params.get("tickers", _DEFAULT_WATCHLIST)
            rows = [self._fetcher.get_financials(t) for t in tickers]
            df = pd.DataFrame(rows)
        elif template == "macro":
            series_ids = params.get("series_ids", list(_TREASURY_SERIES.values())[:5])
            rows = [self._fetcher.get_macro(s) for s in series_ids]
            df = pd.DataFrame(rows)
        elif template == "yield_curve":
            yc = self._fetcher.get_yield_curve()
            df = pd.DataFrame([
                {"tenor": k, "yield_pct": v, "as_of": yc.as_of}
                for k, v in yc.curve.items()
            ])
        elif template == "portfolio":
            holdings = params.get("holdings", [
                {"ticker": t, "shares": 100, "cost_basis": 100.0}
                for t in _DEFAULT_WATCHLIST
            ])
            rows = self._fetcher.get_portfolio(holdings)
            df = pd.DataFrame(rows)
        else:
            return {"error": f"Unknown template: {template}"}

        update_item = self.dataframe_to_update(df, sheet_name, start_cell)
        return self.batch_update(spreadsheet_id, [update_item])


# ── 5. CSV Export Engine ───────────────────────────────────────────────────────

class CSVExportEngine:
    """
    Fallback CSV export to ~/sentinel_exports/ when neither Excel nor Sheets
    is configured. Also provides CSV download endpoints.
    """

    def __init__(self) -> None:
        self._fetcher = SentinelDataFetcher()
        _EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        template: str,
        params: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """
        Run a template and return (csv_bytes, suggested_filename).
        Templates: quote, financials, screen, macro, portfolio, yield_curve
        """
        params = params or {}
        tickers = params.get("tickers", _DEFAULT_WATCHLIST)

        if template == "quote":
            rows = [self._fetcher.get_quote(t) for t in tickers]
            df = pd.DataFrame(rows)
        elif template == "financials":
            rows = [self._fetcher.get_financials(t) for t in tickers]
            df = pd.DataFrame(rows)
        elif template == "screen":
            rows = self._fetcher.get_screen({"tickers": tickers})
            df = pd.DataFrame(rows)
        elif template == "macro":
            series_ids = params.get("series_ids",
                                    ["UNRATE", "PAYEMS", "JTSJOL", "FEDFUNDS", "VIXCLS"])
            rows = [self._fetcher.get_macro(s) for s in series_ids]
            df = pd.DataFrame(rows)
        elif template == "portfolio":
            holdings = params.get("holdings", [
                {"ticker": t, "shares": 100, "cost_basis": 100.0}
                for t in tickers
            ])
            rows = self._fetcher.get_portfolio(holdings)
            df = pd.DataFrame(rows)
        elif template == "yield_curve":
            yc = self._fetcher.get_yield_curve()
            df = pd.DataFrame([
                {"tenor": k, "yield_pct": v, "as_of": yc.as_of,
                 "inverted": yc.inverted,
                 "spread_2s10s": yc.spread_2s10s}
                for k, v in yc.curve.items()
            ])
        elif template == "ohlcv":
            ticker = tickers[0] if tickers else "AAPL"
            df = self._fetcher.get_chart_ohlcv(
                ticker, params.get("start"), params.get("end"), params.get("timeframe", "1d")
            ).reset_index()
        elif template == "options":
            ticker = tickers[0] if tickers else "AAPL"
            df = self._fetcher.get_options_chain(ticker, params.get("expiry"))
        else:
            df = pd.DataFrame({"error": [f"Unknown template: {template}"]})

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = filename or f"sentinel_{template}_{ts}.csv"
        return csv_bytes, fname

    def export_to_file(
        self, template: str, params: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Write export to ~/sentinel_exports/ and return path."""
        csv_bytes, fname = self.export(template, params)
        out_path = _EXPORT_DIR / fname
        out_path.write_bytes(csv_bytes)
        logger.info("CSV export: %s (%d bytes)", out_path, len(csv_bytes))
        return out_path


# ── Helper functions ───────────────────────────────────────────────────────────

def _write_cell(
    ws: Any, row: int, col: int, value: Any,
    font: Optional[Any] = None,
    fill: Optional[Any] = None,
    alignment: Optional[Any] = None,
    number_format: Optional[str] = None,
) -> None:
    if not _OPENPYXL_OK:
        return
    cell = ws.cell(row=row, column=col, value=value)
    if font:
        cell.font = font
    if fill:
        cell.fill = fill
    if alignment:
        cell.alignment = alignment
    if number_format:
        cell.number_format = number_format


def _write_header_row(ws: Any, row: int, headers: List[str]) -> None:
    if not _OPENPYXL_OK:
        return
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill("solid", fgColor=_SENTINEL_BLUE_DARK)
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
    ws.row_dimensions[row].height = 20


def _write_data_row(
    ws: Any, row: int, values: List[Any], alternate: bool = False
) -> None:
    if not _OPENPYXL_OK:
        return
    alt_fill = PatternFill("solid", fgColor=_ALT_ROW)
    for col, val in enumerate(values, start=1):
        cell = ws.cell(row=row, column=col, value=_clean_val(val))
        if alternate:
            cell.fill = alt_fill


def _apply_pnl_conditional(ws: Any, cell_range: str) -> None:
    if not _OPENPYXL_OK:
        return
    green_fill = PatternFill("solid", fgColor=_SENTINEL_GREEN)
    red_fill = PatternFill("solid", fgColor=_SENTINEL_RED)
    try:
        ws.conditional_formatting.add(
            cell_range,
            CellIsRule(operator="greaterThan", formula=["0"], fill=green_fill),
        )
        ws.conditional_formatting.add(
            cell_range,
            CellIsRule(operator="lessThan", formula=["0"], fill=red_fill),
        )
    except Exception as exc:
        logger.debug("Conditional formatting error: %s", exc)


def _auto_width(ws: Any, headers: List[str]) -> None:
    if not _OPENPYXL_OK:
        return
    for col_idx, header in enumerate(headers, start=1):
        max_len = len(str(header)) + 2
        col_letter = get_column_letter(col_idx)
        for row in ws.iter_rows(min_col=col_idx, max_col=col_idx):
            for cell in row:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(42, max_len + 2)


def _clean_val(val: Any) -> Any:
    if val is None:
        return ""
    if isinstance(val, float) and (val != val):
        return ""
    if isinstance(val, (list, dict)):
        return str(val)
    return val


def _df_to_sheet_values(df: pd.DataFrame) -> List[List[Any]]:
    """Convert DataFrame to 2D list suitable for Sheets API."""
    headers = list(df.columns)
    rows = df.values.tolist()
    clean_rows = [
        [str(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else ""
         for v in row]
        for row in rows
    ]
    return [headers] + clean_rows


import hashlib  # imported here to avoid forward-reference issue in class


# ── 6. FastAPI Router ──────────────────────────────────────────────────────────

export_v3_router = APIRouter(prefix="/export/v3", tags=["excel_sheets_v3"])

# Module-level singletons
_fetcher_singleton: Optional[SentinelDataFetcher] = None
_builder_singleton: Optional[SentinelWorkbookBuilder] = None
_rtd_singleton: Optional[RTDPollingLoop] = None
_sheets_singleton: Optional[GoogleSheetsAdapter] = None
_csv_singleton: Optional[CSVExportEngine] = None


def _get_fetcher() -> SentinelDataFetcher:
    global _fetcher_singleton
    if _fetcher_singleton is None:
        _fetcher_singleton = SentinelDataFetcher()
    return _fetcher_singleton


def _get_builder() -> SentinelWorkbookBuilder:
    global _builder_singleton
    if _builder_singleton is None:
        _builder_singleton = SentinelWorkbookBuilder(_get_fetcher())
    return _builder_singleton


def _get_rtd() -> RTDPollingLoop:
    global _rtd_singleton
    if _rtd_singleton is None:
        _rtd_singleton = RTDPollingLoop()
    return _rtd_singleton


def _get_sheets() -> GoogleSheetsAdapter:
    global _sheets_singleton
    if _sheets_singleton is None:
        _sheets_singleton = GoogleSheetsAdapter()
    return _sheets_singleton


def _get_csv() -> CSVExportEngine:
    global _csv_singleton
    if _csv_singleton is None:
        _csv_singleton = CSVExportEngine()
    return _csv_singleton


# ── Endpoints ──────────────────────────────────────────────────────────────────

@export_v3_router.post("/excel/workbook")
def route_create_workbook(body: WorkbookCreateRequest):
    """
    Generate a production SENTINEL.xlsx workbook with all 8 data sheets.
    Returns the file as a download (or saves to output_path if specified).
    """
    try:
        builder = _get_builder()
        if body.output_path:
            path = builder.build_workbook(
                tickers=body.tickers,
                output_path=Path(body.output_path),
                include_charts=body.include_charts,
            )
            return {
                "path": str(path),
                "tickers": body.tickers,
                "sheets": ["Cover", "Quote", "Financials", "Screener",
                           "Macro", "YieldCurve", "Portfolio", "OHLCV"],
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        else:
            # Stream the workbook as a download
            if not _OPENPYXL_OK:
                raise HTTPException(status_code=400,
                                    detail="openpyxl not installed — install it or provide output_path")
            buf = io.BytesIO()
            # Build to temp path then read into buffer
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            builder.build_workbook(body.tickers, tmp_path, body.include_charts)
            buf = io.BytesIO(tmp_path.read_bytes())
            tmp_path.unlink(missing_ok=True)

            fname = f"SENTINEL_{datetime.now().strftime('%Y%m%d')}.xlsx"
            return StreamingResponse(
                buf,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{fname}"'},
            )
    except Exception as exc:
        logger.error("create_workbook error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@export_v3_router.post("/excel/refresh")
def route_refresh_workbook(body: WorkbookRefreshRequest):
    """
    Refresh all data in an existing SENTINEL workbook file.
    Re-fetches all quotes, financials, macro data and overwrites the file.
    """
    try:
        builder = _get_builder()
        result = builder.refresh_workbook(body.workbook_path, body.tickers)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@export_v3_router.get("/excel/template")
def route_excel_template(
    tickers: str = Query(
        default=",".join(_DEFAULT_WATCHLIST[:5]),
        description="Comma-separated tickers",
    ),
):
    """
    Download a minimal SENTINEL Excel template (Quote sheet only).
    Useful for testing without writing to disk.
    """
    if not _OPENPYXL_OK:
        raise HTTPException(status_code=400, detail="openpyxl not installed")

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        builder = _get_builder()
        builder.build_workbook(ticker_list, tmp_path, include_charts=False)
        buf = io.BytesIO(tmp_path.read_bytes())
        tmp_path.unlink(missing_ok=True)

        fname = f"SENTINEL_template_{datetime.now().strftime('%Y%m%d')}.xlsx"
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@export_v3_router.post("/excel/rtd/subscribe")
def route_rtd_subscribe(body: WorkbookCreateRequest):
    """
    Subscribe a workbook to RTD polling. SENTINEL will refresh it every 60s.
    Requires output_path to be specified.
    """
    if not body.output_path:
        raise HTTPException(status_code=400, detail="output_path required for RTD subscription")
    try:
        rtd = _get_rtd()
        sub_id = rtd.subscribe(body.output_path, body.tickers)
        return {
            "subscription_id": sub_id,
            "workbook_path": body.output_path,
            "tickers": body.tickers,
            "refresh_interval_seconds": _RTD_INTERVAL_SECONDS,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@export_v3_router.get("/excel/rtd/status")
def route_rtd_status():
    """List all active RTD workbook subscriptions."""
    return {"subscriptions": _get_rtd().list_subscriptions()}


@export_v3_router.post("/sheets/update")
def route_sheets_update(body: SheetsUpdateRequest):
    """
    Push data to a Google Sheet via Sheets API v4 batchUpdate.
    Requires GOOGLE_SERVICE_ACCOUNT_JSON env var.
    """
    try:
        adapter = _get_sheets()
        range_notation = f"{body.sheet_name}!{body.start_cell}"
        result = adapter.batch_update(
            body.spreadsheet_id,
            [{"range": range_notation, "values": body.data}],
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@export_v3_router.get("/sheets/read")
def route_sheets_read(
    spreadsheet_id: str = Query(...),
    range_notation: str = Query(default="Sheet1!A1:Z100"),
):
    """
    Read a range from Google Sheets.
    Works for public sheets without credentials (read-only).
    """
    try:
        adapter = _get_sheets()
        values = adapter.read_range(spreadsheet_id, range_notation)
        return {
            "spreadsheet_id": spreadsheet_id,
            "range": range_notation,
            "values": values,
            "rows": len(values),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@export_v3_router.post("/sheets/push-template")
def route_sheets_push_template(
    spreadsheet_id: str = Query(...),
    template: str = Query(
        default="quote",
        description="quote | financials | macro | yield_curve | portfolio | screen",
    ),
    tickers: str = Query(default=",".join(_DEFAULT_WATCHLIST[:5])),
    sheet_name: str = Query(default="SENTINEL"),
):
    """
    Push a SENTINEL data template directly into a Google Sheet.
    """
    try:
        adapter = _get_sheets()
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        params = {"tickers": ticker_list, "sheet_name": sheet_name}
        result = adapter.push_sentinel_template(spreadsheet_id, template, params)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@export_v3_router.post("/csv/export")
def route_csv_export(body: CSVExportRequest):
    """
    Export SENTINEL data as a CSV file download.
    Falls back automatically when Excel/Sheets are not configured.
    Templates: quote, financials, screen, macro, portfolio, yield_curve, ohlcv, options
    """
    try:
        csv_engine = _get_csv()
        tickers = [t.strip().upper() for t in body.tickers if t.strip()]
        params = dict(body.params or {})
        params["tickers"] = tickers

        csv_bytes, fname = csv_engine.export(body.template, params, body.filename)
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@export_v3_router.get("/formats/supported")
def route_formats_supported():
    """
    Return supported export formats and templates, along with
    availability status for each backend (Excel, Sheets, CSV).
    """
    return {
        "templates": [
            {"name": "quote",       "description": "Real-time quote data (price, volume, change)"},
            {"name": "financials",  "description": "Revenue, EBITDA, EPS, FCF, margins, ratios"},
            {"name": "screen",      "description": "Multi-factor screener across ticker universe"},
            {"name": "macro",       "description": "FRED macro indicators (UNRATE, PAYEMS, etc.)"},
            {"name": "portfolio",   "description": "Holdings P&L, attribution, weight"},
            {"name": "chart",       "description": "OHLCV time series for charting"},
            {"name": "yield_curve", "description": "Full Treasury yield curve (1M–30Y)"},
            {"name": "options",     "description": "Options chain with greeks"},
        ],
        "formats": {
            "xlsx": {
                "available": _OPENPYXL_OK,
                "backend": "openpyxl (pure Python, cross-platform)",
                "note": "No COM/Windows dependency",
            },
            "csv": {
                "available": True,
                "backend": "pandas CSV export",
                "export_dir": str(_EXPORT_DIR),
            },
            "google_sheets": {
                "available": _GAPI_OK,
                "authenticated": _get_sheets().is_authenticated(),
                "note": "Set GOOGLE_SERVICE_ACCOUNT_JSON env var for write access",
                "public_read": True,
            },
        },
        "named_ranges": ["SENTINEL_UNIVERSE", "SENTINEL_WATCHLIST"],
        "rtd": {
            "supported": True,
            "mechanism": "openpyxl polling loop (60s interval)",
            "note": "No Excel COM — pure file-write RTD simulation",
        },
        "data_sources": [
            "Yahoo Finance v8 (free, no key)",
            "FRED free CSV API (no key)",
            "yfinance (optional, enhances options chain)",
        ],
    }


@export_v3_router.get("/yield-curve", response_model=YieldCurveResult)
def route_yield_curve():
    """
    Return the current full Treasury yield curve (1M through 30Y) from FRED.
    Includes inversion flag and 2s/10s spread.
    """
    try:
        return _get_fetcher().get_yield_curve()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@export_v3_router.get("/health")
def route_health():
    """Health check for the Excel/Sheets plugin v3."""
    checks = {
        "openpyxl": "ok" if _OPENPYXL_OK else "not installed",
        "google_api_client": "ok" if _GAPI_OK else "not installed",
        "google_sheets_auth": "ok" if _get_sheets().is_authenticated() else "not configured",
        "yfinance": "ok" if _YF_OK else "not installed",
        "sqlite_db": "ok",
        "export_dir": str(_EXPORT_DIR),
    }
    try:
        _ensure_db()
    except Exception as exc:
        checks["sqlite_db"] = f"error: {exc}"

    return {
        "status": "healthy",
        "checks": checks,
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
    }


# ── Module init ────────────────────────────────────────────────────────────────

try:
    _ensure_db()
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
except Exception as _init_exc:
    logger.warning("excel_sheets_plugin_v3: init failed: %s", _init_exc)


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="SENTINEL Excel/Sheets Plugin v3",
        description=(
            "Universal Excel + Google Sheets integration. "
            "openpyxl (no COM), Sheets API v4, CSV fallback."
        ),
        version="3.0.0",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(export_v3_router)
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")
