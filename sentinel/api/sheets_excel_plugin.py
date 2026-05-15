"""
Excel RTD server + Google Sheets integration for SENTINEL.
Enables =SENTINEL("AAPL", "price") formulas in spreadsheets.

Covers: real-time quotes, fundamentals, options, financials, screener results,
CSV/Excel export, webhooks, and Office Add-in manifest generation.

Dimensions:
  dim_093 — Excel / Google Sheets plugin  target: 9

Runs as a FastAPI router mounted into the main SENTINEL API, or standalone
on port 8082 via run_server().
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import pandas as pd  # type: ignore[import]
    _PANDAS_OK = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_OK = False

try:
    import openpyxl  # type: ignore[import]
    from openpyxl.styles import (  # type: ignore[import]
        Font, PatternFill, Alignment, Border, Side
    )
    from openpyxl.utils import get_column_letter  # type: ignore[import]
    from openpyxl.chart import BarChart, Reference  # type: ignore[import]
    from openpyxl.formatting.rule import ColorScaleRule  # type: ignore[import]
    _OPENPYXL_OK = True
except ImportError:
    openpyxl = None  # type: ignore[assignment]
    _OPENPYXL_OK = False

try:
    import gspread  # type: ignore[import]
    from google.oauth2.service_account import Credentials as _GCredentials  # type: ignore[import]
    _GSPREAD_OK = True
except ImportError:
    gspread = None  # type: ignore[assignment]
    _GCredentials = None  # type: ignore[assignment]
    _GSPREAD_OK = False

try:
    import win32com.client as _win32  # type: ignore[import]
    import pythoncom  # type: ignore[import]
    _WIN32_OK = True
except ImportError:
    _win32 = None  # type: ignore[assignment]
    _WIN32_OK = False

try:
    import yfinance as yf  # type: ignore[import]
    _YF_OK = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YF_OK = False

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DB_PATH = Path(os.getenv("SENTINEL_DATA_DIR", "data")) / "sheets_plugin.db"
_RTD_PROGID = "SENTINEL.RTD"
_MANIFEST_TEMPLATE_GUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"

# Cache TTLs in seconds
_PRICE_CACHE_TTL = 60
_FUNDAMENTAL_CACHE_TTL = 4 * 3600  # 4 hours

# Webhook retry settings
_WEBHOOK_MAX_RETRIES = 3
_WEBHOOK_BACKOFF_BASE = 2.0  # seconds

# Supported formula fields
SUPPORTED_FIELDS = [
    "price", "change_pct", "volume", "market_cap", "pe_ratio", "eps",
    "dividend_yield", "revenue_ttm", "net_income_ttm", "ebitda",
    "debt_to_equity", "current_ratio", "beta", "52w_high", "52w_low",
    "rsi_14", "sma_50", "sma_200", "short_interest", "iv_30d",
    "put_call_ratio", "earnings_date", "analyst_target", "analyst_rating",
    "open", "high", "low", "close", "prev_close", "change_abs",
    "shares_outstanding", "free_float", "sector", "industry",
]

_HEADERS = {
    "User-Agent": "SENTINEL:SheetsPlugin:2.0",
    "Accept": "application/json",
}


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS formula_cache (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                key     TEXT UNIQUE NOT NULL,
                value   TEXT NOT NULL,
                ts      INTEGER NOT NULL,
                ttl     INTEGER NOT NULL DEFAULT 60
            );
            CREATE TABLE IF NOT EXISTS webhooks (
                id          TEXT PRIMARY KEY,
                url         TEXT NOT NULL,
                tickers     TEXT NOT NULL,
                fields      TEXT NOT NULL,
                frequency   INTEGER NOT NULL DEFAULT 60,
                created_at  INTEGER NOT NULL,
                active      INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS webhook_deliveries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                webhook_id  TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                field       TEXT NOT NULL,
                value       TEXT,
                ts          INTEGER NOT NULL,
                status      INTEGER NOT NULL DEFAULT 200
            );
            CREATE TABLE IF NOT EXISTS rtd_topics (
                topic_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                progid      TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                field       TEXT NOT NULL,
                last_value  TEXT,
                last_ts     INTEGER
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

class FormulaRequest(BaseModel):
    ticker: str
    field: str
    params: Optional[Dict[str, Any]] = None


class FormulaResult(BaseModel):
    ticker: str
    field: str
    value: Any
    cached: bool = False
    as_of: Optional[datetime] = None
    error: Optional[str] = None


class BatchFormulaRequest(BaseModel):
    requests: List[FormulaRequest]


class BatchFormulaResult(BaseModel):
    results: List[FormulaResult]
    duration_ms: float


class RTDTopic(BaseModel):
    topic_id: int
    ticker: str
    field: str
    last_value: Optional[str] = None
    last_ts: Optional[datetime] = None


class WebhookRegistration(BaseModel):
    url: str
    tickers: List[str]
    fields: List[str]
    frequency: int = Field(default=60, ge=10, le=3600, description="Seconds between pushes")


class WebhookInfo(BaseModel):
    webhook_id: str
    url: str
    tickers: List[str]
    fields: List[str]
    frequency: int
    active: bool
    created_at: datetime


class ExportRequest(BaseModel):
    ticker: str
    filename: Optional[str] = None
    format: str = Field(default="xlsx", pattern="^(xlsx|csv)$")


class ScreenerExportRequest(BaseModel):
    criteria: Dict[str, Any]
    filename: Optional[str] = None
    format: str = Field(default="xlsx", pattern="^(xlsx|csv)$")


# ── 1. SentinelFormulaEngine ───────────────────────────────────────────────────

class SentinelFormulaEngine:
    """
    Core formula evaluation engine.
    Evaluates =SENTINEL(ticker, field) expressions with a two-tier cache:
      - price fields: 60s TTL
      - fundamental fields: 4h TTL
    """

    _FUNDAMENTAL_FIELDS = frozenset([
        "market_cap", "pe_ratio", "eps", "dividend_yield", "revenue_ttm",
        "net_income_ttm", "ebitda", "debt_to_equity", "current_ratio",
        "beta", "short_interest", "iv_30d", "put_call_ratio",
        "earnings_date", "analyst_target", "analyst_rating",
        "shares_outstanding", "free_float", "sector", "industry",
    ])

    _TA_FIELDS = frozenset(["rsi_14", "sma_50", "sma_200"])

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key → (value, expiry_ts)
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, ticker: str, field: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Evaluate a single formula cell.
        Returns scalar (float, str, int) or None on error.
        """
        ticker_up = ticker.upper().strip()
        field_low = field.lower().strip()

        if field_low not in SUPPORTED_FIELDS:
            return f"#UNKNOWN_FIELD:{field}"

        cache_key = f"{ticker_up}:{field_low}"
        cached_val = self._get_cache(cache_key)
        if cached_val is not None:
            return cached_val

        try:
            value = self._fetch(ticker_up, field_low, params)
        except Exception as exc:
            logger.warning("Formula fetch error %s/%s: %s", ticker_up, field_low, exc)
            return f"#ERROR:{exc}"

        ttl = _FUNDAMENTAL_CACHE_TTL if field_low in self._FUNDAMENTAL_FIELDS else _PRICE_CACHE_TTL
        self._set_cache(cache_key, value, ttl)
        return value

    def evaluate_batch(
        self, requests: List[FormulaRequest]
    ) -> List[FormulaResult]:
        """Parallel batch evaluation using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[FormulaResult] = [None] * len(requests)  # type: ignore[list-item]

        def _eval_one(idx: int, req: FormulaRequest) -> Tuple[int, FormulaResult]:
            t0 = time.monotonic()
            cache_key = f"{req.ticker.upper()}:{req.field.lower()}"
            cached = self._get_cache(cache_key)
            if cached is not None:
                return idx, FormulaResult(
                    ticker=req.ticker,
                    field=req.field,
                    value=cached,
                    cached=True,
                    as_of=datetime.now(tz=timezone.utc),
                )
            try:
                val = self._fetch(req.ticker.upper(), req.field.lower(), req.params)
                ttl = (
                    _FUNDAMENTAL_CACHE_TTL
                    if req.field.lower() in self._FUNDAMENTAL_FIELDS
                    else _PRICE_CACHE_TTL
                )
                self._set_cache(cache_key, val, ttl)
                return idx, FormulaResult(
                    ticker=req.ticker,
                    field=req.field,
                    value=val,
                    cached=False,
                    as_of=datetime.now(tz=timezone.utc),
                )
            except Exception as exc:
                return idx, FormulaResult(
                    ticker=req.ticker,
                    field=req.field,
                    value=None,
                    error=str(exc),
                )

        with ThreadPoolExecutor(max_workers=min(12, len(requests))) as pool:
            futures = {pool.submit(_eval_one, i, r): i for i, r in enumerate(requests)}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    # ── Fetch implementations ──────────────────────────────────────────────────

    def _fetch(self, ticker: str, field: str, params: Optional[Dict[str, Any]]) -> Any:
        """Route field to the appropriate data fetcher."""
        if field in ("price", "open", "high", "low", "close", "prev_close",
                     "change_abs", "change_pct", "volume", "52w_high", "52w_low"):
            return self._fetch_quote(ticker, field)
        if field in self._FUNDAMENTAL_FIELDS:
            return self._fetch_fundamental(ticker, field)
        if field in self._TA_FIELDS:
            return self._fetch_ta(ticker, field)
        return None

    def _fetch_quote(self, ticker: str, field: str) -> Any:
        """Fetch real-time quote field via yfinance."""
        if not _YF_OK:
            return self._fetch_yahoo_json(ticker, field)
        try:
            info = yf.Ticker(ticker).fast_info
            mapping = {
                "price": getattr(info, "last_price", None),
                "open": getattr(info, "open", None),
                "high": getattr(info, "day_high", None),
                "low": getattr(info, "day_low", None),
                "close": getattr(info, "last_price", None),
                "prev_close": getattr(info, "previous_close", None),
                "change_abs": (
                    (getattr(info, "last_price", 0) or 0)
                    - (getattr(info, "previous_close", 0) or 0)
                ),
                "change_pct": None,
                "volume": getattr(info, "three_month_average_volume", None),
                "52w_high": getattr(info, "fifty_two_week_high", None),
                "52w_low": getattr(info, "fifty_two_week_low", None),
            }
            val = mapping.get(field)
            if field == "change_pct" and mapping.get("prev_close"):
                prev = mapping["prev_close"]
                curr = mapping["price"] or 0
                val = round((curr - prev) / prev * 100, 3) if prev else None
            if isinstance(val, float):
                val = round(val, 4)
            return val
        except Exception:
            return self._fetch_yahoo_json(ticker, field)

    def _fetch_yahoo_json(self, ticker: str, field: str) -> Any:
        """Fallback: scrape Yahoo Finance v8 quote endpoint."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1d", "range": "1d"}
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            meta = data["chart"]["result"][0]["meta"]
            price = meta.get("regularMarketPrice")
            prev = meta.get("previousClose") or meta.get("chartPreviousClose")
            mapping = {
                "price": price,
                "close": price,
                "prev_close": prev,
                "change_abs": round((price or 0) - (prev or 0), 4) if price and prev else None,
                "change_pct": round(((price or 0) - (prev or 0)) / (prev or 1) * 100, 3) if price and prev else None,
                "volume": meta.get("regularMarketVolume"),
                "52w_high": meta.get("fiftyTwoWeekHigh"),
                "52w_low": meta.get("fiftyTwoWeekLow"),
                "open": meta.get("regularMarketOpen"),
                "high": meta.get("regularMarketDayHigh"),
                "low": meta.get("regularMarketDayLow"),
            }
            return mapping.get(field)
        except Exception as exc:
            logger.debug("Yahoo JSON fetch error %s: %s", ticker, exc)
            return None

    def _fetch_fundamental(self, ticker: str, field: str) -> Any:
        """Fetch fundamental data via yfinance info dict."""
        if not _YF_OK:
            return None
        try:
            info = yf.Ticker(ticker).info
            mapping = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "dividend_yield": info.get("dividendYield"),
                "revenue_ttm": info.get("totalRevenue"),
                "net_income_ttm": info.get("netIncomeToCommon"),
                "ebitda": info.get("ebitda"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "beta": info.get("beta"),
                "short_interest": info.get("shortPercentOfFloat"),
                "iv_30d": info.get("impliedVolatility"),
                "put_call_ratio": None,  # not in yfinance
                "earnings_date": _format_earnings_date(info.get("earningsTimestamp")),
                "analyst_target": info.get("targetMeanPrice"),
                "analyst_rating": info.get("recommendationKey"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "free_float": info.get("floatShares"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
            val = mapping.get(field)
            if isinstance(val, float):
                val = round(val, 6)
            return val
        except Exception as exc:
            logger.debug("Fundamental fetch error %s/%s: %s", ticker, field, exc)
            return None

    def _fetch_ta(self, ticker: str, field: str) -> Any:
        """Compute technical indicators from yfinance OHLCV history."""
        if not _YF_OK or not _PANDAS_OK:
            return None
        try:
            hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
            if hist.empty or "Close" not in hist.columns:
                return None
            close = hist["Close"].dropna()
            if field == "sma_50":
                return round(float(close.rolling(50).mean().iloc[-1]), 4)
            if field == "sma_200":
                return round(float(close.rolling(200).mean().iloc[-1]), 4)
            if field == "rsi_14":
                return round(_compute_rsi(close, 14), 4)
        except Exception as exc:
            logger.debug("TA fetch error %s/%s: %s", ticker, field, exc)
        return None

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _get_cache(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._cache[key]
                return None
            return value

    def _set_cache(self, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            self._cache[key] = (value, time.monotonic() + ttl)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()


# ── Helper: RSI computation ────────────────────────────────────────────────────

def _compute_rsi(series: Any, period: int = 14) -> float:
    """Wilder RSI from a pandas Series of close prices."""
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _format_earnings_date(ts: Optional[int]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


# ── 2. GoogleSheetsIntegration ─────────────────────────────────────────────────

class GoogleSheetsIntegration:
    """
    Google Sheets API v4 integration.
    Requires GOOGLE_SERVICE_ACCOUNT_JSON env var (path to JSON key file)
    or falls back to CSV-mode (no credentials needed).
    """

    _SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._formula_engine = SentinelFormulaEngine()
        self._init_client()

    def _init_client(self) -> None:
        if not _GSPREAD_OK:
            logger.info("gspread not installed — Google Sheets in CSV-only mode")
            return
        key_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if not key_path or not Path(key_path).exists():
            logger.info("No service account JSON — Google Sheets in CSV-only mode")
            return
        try:
            creds = _GCredentials.from_service_account_file(  # type: ignore[union-attr]
                key_path, scopes=self._SCOPES
            )
            self._client = gspread.authorize(creds)  # type: ignore[union-attr]
            logger.info("Google Sheets client initialized")
        except Exception as exc:
            logger.warning("Google Sheets init failed: %s", exc)

    def is_available(self) -> bool:
        return self._client is not None

    def write_to_sheet(self, spreadsheet_id: str, range_name: str, data: List[List[Any]]) -> bool:
        """Write data to a Google Sheet range. Returns True on success."""
        if not self._client:
            logger.warning("Google Sheets client not available")
            return False
        try:
            sheet = self._client.open_by_key(spreadsheet_id)
            worksheet = sheet.worksheet(range_name.split("!")[0]) if "!" in range_name else sheet.sheet1
            cell_range = range_name.split("!")[-1] if "!" in range_name else range_name
            worksheet.update(cell_range, data)
            return True
        except Exception as exc:
            logger.error("Sheets write error: %s", exc)
            return False

    def read_from_sheet(self, spreadsheet_id: str, range_name: str) -> List[List[Any]]:
        """Read values from a Google Sheet range."""
        if not self._client:
            return []
        try:
            sheet = self._client.open_by_key(spreadsheet_id)
            worksheet = sheet.worksheet(range_name.split("!")[0]) if "!" in range_name else sheet.sheet1
            cell_range = range_name.split("!")[-1] if "!" in range_name else range_name
            return worksheet.get(cell_range)
        except Exception as exc:
            logger.error("Sheets read error: %s", exc)
            return []

    def batch_update(
        self, spreadsheet_id: str, updates: List[Dict[str, Any]]
    ) -> bool:
        """
        Batch update multiple ranges in one API call.
        updates: [{"range": "Sheet1!A1", "values": [[...], ...]}, ...]
        """
        if not self._client:
            return False
        try:
            sheet = self._client.open_by_key(spreadsheet_id)
            sheet.values_batch_update({"valueInputOption": "USER_ENTERED", "data": updates})
            return True
        except Exception as exc:
            logger.error("Sheets batch update error: %s", exc)
            return False

    def auto_refresh(
        self,
        spreadsheet_id: str,
        cell_map: Dict[str, Tuple[str, str]],
        interval_seconds: int = 60,
        iterations: int = 10,
    ) -> None:
        """
        Auto-refresh SENTINEL formula cells in a Google Sheet.
        cell_map: {"Sheet1!A1": ("AAPL", "price"), ...}
        Runs `iterations` refresh cycles then stops.
        """
        for _ in range(iterations):
            updates: List[Dict[str, Any]] = []
            for cell_range, (ticker, field) in cell_map.items():
                val = self._formula_engine.evaluate(ticker, field)
                updates.append({"range": cell_range, "values": [[val]]})
            if updates:
                self.batch_update(spreadsheet_id, updates)
            time.sleep(interval_seconds)

    def export_to_csv(self, data: List[List[Any]]) -> str:
        """Fallback: convert 2D list to CSV string."""
        lines: List[str] = []
        for row in data:
            lines.append(",".join(str(v) if v is not None else "" for v in row))
        return "\n".join(lines)


# ── 3. ExcelRTDServer ──────────────────────────────────────────────────────────

class ExcelRTDServer:
    """
    Excel Real-Time Data (RTD) server.

    On Windows with pywin32: implements the IRTDServer COM interface.
    On all platforms: provides an HTTP polling endpoint that Excel can call
    via a VBA macro or HTTP RTD shim.

    Topic format: "{progid}|{ticker}|{field}"
    """

    def __init__(self, update_frequency: int = 5) -> None:
        self._update_frequency = update_frequency
        self._topics: Dict[int, Tuple[str, str]] = {}  # topic_id → (ticker, field)
        self._values: Dict[int, Any] = {}
        self._formula_engine = SentinelFormulaEngine()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._heartbeat_func: Optional[Any] = None  # COM callback

    # ── COM interface methods (pywin32 path) ───────────────────────────────────

    def ServerStart(self, callback_object: Any) -> int:  # noqa: N802
        """Called by Excel when the RTD server starts."""
        self._heartbeat_func = callback_object
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("RTD server started")
        return 1

    def ServerTerminate(self) -> None:  # noqa: N802
        """Called by Excel when RTD server shuts down."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("RTD server terminated")

    def ConnectData(  # noqa: N802
        self, topic_id: int, strings: List[str], get_new_values: bool
    ) -> Tuple[Any, bool]:
        """Called by Excel to connect to a data topic."""
        if len(strings) < 2:
            return "#INVALID_TOPIC", True
        ticker = strings[0].upper().strip()
        field = strings[1].lower().strip()
        with self._lock:
            self._topics[topic_id] = (ticker, field)
        value = self._formula_engine.evaluate(ticker, field)
        self._values[topic_id] = value
        return value, True

    def DisconnectData(self, topic_id: int) -> None:  # noqa: N802
        """Called by Excel when a cell no longer uses this topic."""
        with self._lock:
            self._topics.pop(topic_id, None)
            self._values.pop(topic_id, None)

    def RefreshData(self, topic_count: int) -> Tuple[List[List[Any]], int]:  # noqa: N802
        """Called by Excel to get updated values."""
        updates: List[List[Any]] = [[], []]
        with self._lock:
            for tid, val in list(self._values.items()):
                updates[0].append(tid)
                updates[1].append(val if val is not None else "N/A")
        return updates, len(updates[0])

    def Heartbeat(self) -> int:  # noqa: N802
        return 1

    # ── Background update loop ─────────────────────────────────────────────────

    def _update_loop(self) -> None:
        """Continuously refresh all connected topics."""
        while self._running:
            with self._lock:
                topics_snapshot = dict(self._topics)
            for topic_id, (ticker, field) in topics_snapshot.items():
                try:
                    new_val = self._formula_engine.evaluate(ticker, field)
                    old_val = self._values.get(topic_id)
                    if new_val != old_val:
                        with self._lock:
                            self._values[topic_id] = new_val
                        # Persist to DB for HTTP polling
                        self._persist_topic(topic_id, ticker, field, new_val)
                except Exception as exc:
                    logger.debug("RTD update error topic=%d: %s", topic_id, exc)

            # Notify Excel of updates (COM path)
            if self._heartbeat_func:
                try:
                    self._heartbeat_func.UpdateNotify()
                except Exception:
                    pass
            time.sleep(self._update_frequency)

    def _persist_topic(
        self, topic_id: int, ticker: str, field: str, value: Any
    ) -> None:
        """Store latest RTD value in DB for HTTP polling clients."""
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT INTO rtd_topics(topic_id, progid, ticker, field, last_value, last_ts)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(topic_id) DO UPDATE SET
                         last_value=excluded.last_value, last_ts=excluded.last_ts""",
                    (topic_id, _RTD_PROGID, ticker, field, str(value), int(time.time())),
                )
        except Exception:
            pass

    # ── HTTP polling endpoint data ─────────────────────────────────────────────

    def get_poll_data(self, topics: List[str]) -> Dict[str, Any]:
        """
        Return current values for requested topics (HTTP polling mode).
        topics: ["AAPL|price", "TSLA|change_pct", ...]
        """
        result: Dict[str, Any] = {}
        engine = self._formula_engine
        for topic in topics:
            parts = topic.split("|")
            if len(parts) < 2:
                result[topic] = "#INVALID"
                continue
            ticker, field = parts[0].upper(), parts[1].lower()
            result[topic] = engine.evaluate(ticker, field)
        return result

    def register_topic_http(self, ticker: str, field: str) -> int:
        """Register a new RTD topic and return its ID (HTTP mode)."""
        with self._lock:
            new_id = max(self._topics.keys(), default=0) + 1
            self._topics[new_id] = (ticker.upper(), field.lower())
            self._values[new_id] = None
        return new_id


# ── 4. CSVExportEngine ────────────────────────────────────────────────────────

class CSVExportEngine:
    """
    Export SENTINEL data to CSV and Excel files with formatting.
    Uses openpyxl for rich Excel output; falls back to CSV if not installed.
    """

    def __init__(self) -> None:
        self._engine = SentinelFormulaEngine()

    # ── Screener export ────────────────────────────────────────────────────────

    def export_screener(
        self, criteria: Dict[str, Any], filename: Optional[str] = None
    ) -> bytes:
        """
        Export screener results to Excel/CSV.
        criteria: {"tickers": [...], "fields": [...], "min_pe": ..., "max_pe": ..., ...}
        Returns bytes of the file.
        """
        tickers = criteria.get("tickers", [])
        fields = criteria.get("fields", SUPPORTED_FIELDS[:10])

        if not tickers:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]

        # Batch fetch
        reqs = [FormulaRequest(ticker=t, field=f) for t in tickers for f in fields]
        results_flat = self._engine.evaluate_batch(reqs)

        # Reshape into {ticker: {field: value}}
        data: Dict[str, Dict[str, Any]] = {}
        idx = 0
        for t in tickers:
            data[t] = {}
            for f in fields:
                if idx < len(results_flat):
                    data[t][f] = results_flat[idx].value
                idx += 1

        # Apply optional filters
        min_pe = criteria.get("min_pe")
        max_pe = criteria.get("max_pe")
        filtered_tickers = []
        for t in tickers:
            pe = data[t].get("pe_ratio")
            if min_pe is not None and (pe is None or pe < min_pe):
                continue
            if max_pe is not None and (pe is None or pe > max_pe):
                continue
            filtered_tickers.append(t)

        return self._write_excel(
            headers=["Ticker"] + fields,
            rows=[[t] + [data[t].get(f) for f in fields] for t in filtered_tickers],
            sheet_name="Screener",
            filename=filename,
        )

    # ── Portfolio export ───────────────────────────────────────────────────────

    def export_portfolio(
        self, holdings: List[Dict[str, Any]], filename: Optional[str] = None
    ) -> bytes:
        """
        Export portfolio to Excel.
        holdings: [{"ticker": "AAPL", "shares": 100, "cost_basis": 150.0}, ...]
        """
        if not _PANDAS_OK:
            return self._export_portfolio_csv(holdings)

        rows: List[List[Any]] = []
        for h in holdings:
            ticker = h.get("ticker", "").upper()
            shares = float(h.get("shares", 0))
            cost = float(h.get("cost_basis", 0))
            price = self._engine.evaluate(ticker, "price") or 0
            change_pct = self._engine.evaluate(ticker, "change_pct") or 0
            mkt_value = round(shares * (price or 0), 2)
            gain_loss = round(mkt_value - shares * cost, 2)
            gain_loss_pct = round((mkt_value / (shares * cost) - 1) * 100, 2) if cost else 0

            rows.append([
                ticker,
                shares,
                round(cost, 4),
                round(price or 0, 4),
                f"{change_pct:+.2f}%" if change_pct else "N/A",
                mkt_value,
                gain_loss,
                f"{gain_loss_pct:+.2f}%",
            ])

        headers = [
            "Ticker", "Shares", "Cost Basis", "Current Price",
            "Day Chg%", "Mkt Value", "Gain/Loss $", "Gain/Loss %",
        ]

        return self._write_excel(
            headers=headers,
            rows=rows,
            sheet_name="Portfolio",
            filename=filename,
            apply_conditional=True,
        )

    def _export_portfolio_csv(self, holdings: List[Dict[str, Any]]) -> bytes:
        lines = ["Ticker,Shares,Cost Basis,Current Price,Mkt Value,Gain/Loss $"]
        for h in holdings:
            t = h.get("ticker", "")
            s = h.get("shares", 0)
            c = h.get("cost_basis", 0)
            p = self._engine.evaluate(t, "price") or 0
            mv = round(s * p, 2)
            gl = round(mv - s * c, 2)
            lines.append(f"{t},{s},{c},{p},{mv},{gl}")
        return "\n".join(lines).encode()

    # ── Financials export ──────────────────────────────────────────────────────

    def export_financials(
        self, ticker: str, filename: Optional[str] = None
    ) -> bytes:
        """Export 3-statement financial model template to Excel."""
        ticker_up = ticker.upper()
        fields_to_fetch = [
            "revenue_ttm", "net_income_ttm", "ebitda", "eps",
            "pe_ratio", "market_cap", "debt_to_equity", "current_ratio",
            "beta", "dividend_yield", "price", "52w_high", "52w_low",
            "analyst_target", "analyst_rating",
        ]
        reqs = [FormulaRequest(ticker=ticker_up, field=f) for f in fields_to_fetch]
        results = self._engine.evaluate_batch(reqs)
        values = {r.field: r.value for r in results}

        rows: List[List[Any]] = [
            ["SENTINEL Financial Snapshot", "", ""],
            ["Ticker", ticker_up, ""],
            ["As of", datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), ""],
            ["", "", ""],
            ["=== INCOME STATEMENT ===", "", ""],
            ["Revenue (TTM)", values.get("revenue_ttm"), "USD"],
            ["Net Income (TTM)", values.get("net_income_ttm"), "USD"],
            ["EBITDA", values.get("ebitda"), "USD"],
            ["EPS (TTM)", values.get("eps"), "USD/share"],
            ["", "", ""],
            ["=== VALUATION ===", "", ""],
            ["Market Cap", values.get("market_cap"), "USD"],
            ["P/E Ratio", values.get("pe_ratio"), "x"],
            ["Analyst Target", values.get("analyst_target"), "USD"],
            ["Analyst Rating", values.get("analyst_rating"), ""],
            ["", "", ""],
            ["=== BALANCE SHEET RATIOS ===", "", ""],
            ["Debt/Equity", values.get("debt_to_equity"), "x"],
            ["Current Ratio", values.get("current_ratio"), "x"],
            ["", "", ""],
            ["=== PRICE DATA ===", "", ""],
            ["Current Price", values.get("price"), "USD"],
            ["52-Week High", values.get("52w_high"), "USD"],
            ["52-Week Low", values.get("52w_low"), "USD"],
            ["Beta", values.get("beta"), ""],
            ["Dividend Yield", values.get("dividend_yield"), "%"],
        ]

        return self._write_excel(
            headers=["Metric", "Value", "Unit"],
            rows=rows,
            sheet_name=f"{ticker_up}_Financials",
            filename=filename,
        )

    # ── Options chain export ───────────────────────────────────────────────────

    def export_options_chain(
        self, ticker: str, filename: Optional[str] = None
    ) -> bytes:
        """Export options chain with greeks (via yfinance)."""
        ticker_up = ticker.upper()
        rows: List[List[Any]] = []
        headers: List[str] = []

        if _YF_OK and _PANDAS_OK:
            try:
                tk = yf.Ticker(ticker_up)
                expirations = tk.options[:3] if tk.options else []
                for exp_date in expirations:
                    chain = tk.option_chain(exp_date)
                    for opt_type, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
                        if df is None or df.empty:
                            continue
                        df = df.copy()
                        df["type"] = opt_type
                        df["expiration"] = exp_date
                        for _, row_s in df.iterrows():
                            rows.append([
                                opt_type,
                                exp_date,
                                row_s.get("strike"),
                                row_s.get("lastPrice"),
                                row_s.get("bid"),
                                row_s.get("ask"),
                                row_s.get("volume"),
                                row_s.get("openInterest"),
                                row_s.get("impliedVolatility"),
                                row_s.get("delta"),
                                row_s.get("gamma"),
                                row_s.get("theta"),
                                row_s.get("vega"),
                            ])
                headers = [
                    "Type", "Expiration", "Strike", "Last", "Bid", "Ask",
                    "Volume", "Open Int", "IV", "Delta", "Gamma", "Theta", "Vega",
                ]
            except Exception as exc:
                logger.warning("Options chain fetch error %s: %s", ticker_up, exc)

        if not rows:
            rows = [["Options data unavailable — yfinance required"]]
            headers = ["Status"]

        return self._write_excel(
            headers=headers,
            rows=rows,
            sheet_name=f"{ticker_up}_Options",
            filename=filename,
        )

    # ── Core Excel writer ──────────────────────────────────────────────────────

    def _write_excel(
        self,
        headers: List[str],
        rows: List[List[Any]],
        sheet_name: str = "Data",
        filename: Optional[str] = None,
        apply_conditional: bool = False,
    ) -> bytes:
        """Write headers + rows to an xlsx workbook and return bytes."""
        if not _OPENPYXL_OK:
            return self._write_csv(headers, rows)

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet_name[:31]

        # Header style
        header_font = Font(bold=True, color="FFFFFF", size=11)
        header_fill = PatternFill(fill_type="solid", fgColor="1F4E79")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        ws.append(headers)
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
        ws.row_dimensions[1].height = 22

        # Data rows
        alt_fill = PatternFill(fill_type="solid", fgColor="EBF3FB")
        for row_idx, row in enumerate(rows, start=2):
            ws.append([self._fmt_val(v) for v in row])
            if row_idx % 2 == 0:
                for cell in ws[row_idx]:
                    if not cell.fill or cell.fill.fgColor.rgb == "00000000":
                        cell.fill = alt_fill

        # Auto-width columns
        for col_idx, header in enumerate(headers, start=1):
            col_letter = get_column_letter(col_idx)
            max_len = len(str(header))
            for row in rows:
                if col_idx - 1 < len(row):
                    val_len = len(str(row[col_idx - 1] or ""))
                    max_len = max(max_len, val_len)
            ws.column_dimensions[col_letter].width = min(40, max_len + 3)

        # Conditional formatting on gain/loss columns (if requested)
        if apply_conditional and _OPENPYXL_OK and len(rows) > 0:
            try:
                from openpyxl.formatting.rule import CellIsRule
                last_col = get_column_letter(len(headers))
                last_row = len(rows) + 1
                range_str = f"G2:G{last_row}"
                green_fill = PatternFill(fill_type="solid", fgColor="C6EFCE")
                red_fill = PatternFill(fill_type="solid", fgColor="FFC7CE")
                ws.conditional_formatting.add(
                    range_str,
                    CellIsRule(operator="greaterThan", formula=["0"], fill=green_fill),
                )
                ws.conditional_formatting.add(
                    range_str,
                    CellIsRule(operator="lessThan", formula=["0"], fill=red_fill),
                )
            except Exception:
                pass

        # Freeze top row
        ws.freeze_panes = "A2"

        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    @staticmethod
    def _write_csv(headers: List[str], rows: List[List[Any]]) -> bytes:
        lines = [",".join(str(h) for h in headers)]
        for row in rows:
            lines.append(",".join(str(v) if v is not None else "" for v in row))
        return "\n".join(lines).encode()

    @staticmethod
    def _fmt_val(v: Any) -> Any:
        """Clean up values for Excel cells."""
        if v is None:
            return ""
        if isinstance(v, float) and (v != v):  # NaN check
            return ""
        return v


# ── 5. WebhookIntegration ──────────────────────────────────────────────────────

class WebhookIntegration:
    """
    Push SENTINEL data updates to external systems via HTTP webhooks.
    Supports registration, delivery history, and retry logic.
    """

    def __init__(self) -> None:
        self._engine = SentinelFormulaEngine()
        self._active_workers: Dict[str, threading.Thread] = {}

    def register(
        self, url: str, tickers: List[str], fields: List[str], frequency: int = 60
    ) -> str:
        """Register a new webhook. Returns webhook_id."""
        webhook_id = str(uuid.uuid4())
        with _db() as conn:
            conn.execute(
                """INSERT INTO webhooks(id, url, tickers, fields, frequency, created_at, active)
                   VALUES(?,?,?,?,?,?,1)""",
                (webhook_id, url, json.dumps(tickers), json.dumps(fields),
                 frequency, int(time.time())),
            )
        # Start background worker
        self._start_worker(webhook_id, url, tickers, fields, frequency)
        logger.info("Webhook registered: %s -> %s", webhook_id, url)
        return webhook_id

    def list_webhooks(self) -> List[WebhookInfo]:
        try:
            with _db() as conn:
                rows = conn.execute(
                    "SELECT id, url, tickers, fields, frequency, created_at, active FROM webhooks"
                ).fetchall()
            return [
                WebhookInfo(
                    webhook_id=row["id"],
                    url=row["url"],
                    tickers=json.loads(row["tickers"]),
                    fields=json.loads(row["fields"]),
                    frequency=row["frequency"],
                    active=bool(row["active"]),
                    created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
                )
                for row in rows
            ]
        except Exception:
            return []

    def deactivate(self, webhook_id: str) -> bool:
        try:
            with _db() as conn:
                conn.execute(
                    "UPDATE webhooks SET active=0 WHERE id=?", (webhook_id,)
                )
            worker = self._active_workers.pop(webhook_id, None)
            # Worker will stop on next iteration when it checks active flag
            return True
        except Exception:
            return False

    def _start_worker(
        self,
        webhook_id: str,
        url: str,
        tickers: List[str],
        fields: List[str],
        frequency: int,
    ) -> None:
        t = threading.Thread(
            target=self._worker_loop,
            args=(webhook_id, url, tickers, fields, frequency),
            daemon=True,
        )
        self._active_workers[webhook_id] = t
        t.start()

    def _worker_loop(
        self,
        webhook_id: str,
        url: str,
        tickers: List[str],
        fields: List[str],
        frequency: int,
    ) -> None:
        while True:
            # Check if still active
            try:
                with _db() as conn:
                    row = conn.execute(
                        "SELECT active FROM webhooks WHERE id=?", (webhook_id,)
                    ).fetchone()
                if not row or not row["active"]:
                    break
            except Exception:
                pass

            # Collect values and push
            for ticker in tickers:
                for field in fields:
                    val = self._engine.evaluate(ticker, field)
                    payload = {
                        "ticker": ticker,
                        "field": field,
                        "value": val,
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        "webhook_id": webhook_id,
                    }
                    status = self._deliver(url, payload)
                    self._record_delivery(webhook_id, ticker, field, val, status)

            time.sleep(frequency)

    def _deliver(self, url: str, payload: dict) -> int:
        """POST payload to webhook URL with retry. Returns HTTP status code."""
        for attempt in range(_WEBHOOK_MAX_RETRIES):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                return resp.status_code
            except Exception as exc:
                wait = _WEBHOOK_BACKOFF_BASE ** attempt
                logger.debug("Webhook delivery attempt %d failed: %s; retrying in %.1fs",
                             attempt + 1, exc, wait)
                time.sleep(wait)
        return 0  # All retries failed

    def _record_delivery(
        self, webhook_id: str, ticker: str, field: str, value: Any, status: int
    ) -> None:
        try:
            with _db() as conn:
                # Keep last 100 per webhook
                conn.execute(
                    """DELETE FROM webhook_deliveries WHERE webhook_id=?
                       AND id NOT IN (
                           SELECT id FROM webhook_deliveries
                           WHERE webhook_id=? ORDER BY ts DESC LIMIT 99
                       )""",
                    (webhook_id, webhook_id),
                )
                conn.execute(
                    """INSERT INTO webhook_deliveries(webhook_id, ticker, field, value, ts, status)
                       VALUES(?,?,?,?,?,?)""",
                    (webhook_id, ticker, field, str(value), int(time.time()), status),
                )
        except Exception as exc:
            logger.debug("Webhook delivery record error: %s", exc)


# ── 6. SentinelAddInManifest ───────────────────────────────────────────────────

class SentinelAddInManifest:
    """
    Generates Office Add-in manifest.xml and custom function schemas
    for the SENTINEL Excel Add-in.
    """

    _ADD_IN_VERSION = "1.0.0.0"
    _MIN_OFFICE_VERSION = "16.0.0.0"

    def generate_manifest(
        self,
        api_base_url: str = "http://localhost:8082",
        provider_name: str = "SENTINEL Finance",
        add_in_name: str = "SENTINEL Financial Terminal",
    ) -> str:
        """Generate manifest.xml content as a string."""
        guid = _MANIFEST_TEMPLATE_GUID
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<OfficeApp
  xmlns="http://schemas.microsoft.com/office/appforoffice/1.1"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:bt="http://schemas.microsoft.com/office/officeappbasictypes/1.0"
  xmlns:ov="http://schemas.microsoft.com/office/taskpaneappversionoverrides"
  xsi:type="TaskPaneApp">

  <Id>{guid}</Id>
  <Version>{self._ADD_IN_VERSION}</Version>
  <ProviderName>{provider_name}</ProviderName>
  <DefaultLocale>en-US</DefaultLocale>
  <DisplayName DefaultValue="{add_in_name}" />
  <Description DefaultValue="Institutional-grade financial data in Excel. =SENTINEL(ticker, field) for prices, fundamentals, options, and more." />

  <SupportUrl DefaultValue="{api_base_url}/docs" />
  <AppDomains>
    <AppDomain>{api_base_url}</AppDomain>
  </AppDomains>

  <Hosts>
    <Host Name="Workbook" />
  </Hosts>

  <Requirements>
    <Sets DefaultMinVersion="1.1">
      <Set Name="ExcelApi" MinVersion="1.1" />
    </Sets>
  </Requirements>

  <DefaultSettings>
    <SourceLocation DefaultValue="{api_base_url}/addin/taskpane.html" />
  </DefaultSettings>

  <Permissions>ReadWriteDocument</Permissions>

  <VersionOverrides xmlns="http://schemas.microsoft.com/office/taskpaneappversionoverrides"
                    xsi:type="VersionOverridesV1_0">
    <Hosts>
      <Host xsi:type="Workbook">
        <AllFormFactors>
          <ExtensionPoint xsi:type="CustomFunctions">
            <Script>
              <SourceLocation resid="Functions.Script.Url" />
            </Script>
            <Page>
              <SourceLocation resid="Taskpane.Url" />
            </Page>
            <Metadata>
              <SourceLocation resid="Functions.Metadata.Url" />
            </Metadata>
            <Namespace resid="Functions.Namespace" />
          </ExtensionPoint>
        </AllFormFactors>
        <DesktopFormFactor>
          <GetStarted>
            <Title resid="GetStarted.Title" />
            <Description resid="GetStarted.Description" />
            <LearnMoreUrl resid="GetStarted.LearnMoreUrl" />
          </GetStarted>
          <FunctionFile resid="Functions.Script.Url" />
          <ExtensionPoint xsi:type="PrimaryCommandSurface">
            <OfficeTab id="TabHome">
              <Group id="CommandsGroup">
                <Label resid="CommandsGroup.Label" />
                <Icon>
                  <bt:Image size="16" resid="Icon.16x16" />
                  <bt:Image size="32" resid="Icon.32x32" />
                </Icon>
                <Control xsi:type="Button" id="OpenTaskpane">
                  <Label resid="OpenTaskpane.Label" />
                  <Supertip>
                    <Title resid="OpenTaskpane.Title" />
                    <Description resid="OpenTaskpane.Tooltip" />
                  </Supertip>
                  <Icon>
                    <bt:Image size="16" resid="Icon.16x16" />
                    <bt:Image size="32" resid="Icon.32x32" />
                  </Icon>
                  <Action xsi:type="ShowTaskpane">
                    <TaskpaneId>ButtonId1</TaskpaneId>
                    <SourceLocation resid="Taskpane.Url" />
                  </Action>
                </Control>
              </Group>
            </OfficeTab>
          </ExtensionPoint>
        </DesktopFormFactor>
      </Host>
    </Hosts>

    <Resources>
      <bt:Urls>
        <bt:Url id="Functions.Script.Url" DefaultValue="{api_base_url}/addin/functions.js" />
        <bt:Url id="Functions.Metadata.Url" DefaultValue="{api_base_url}/addin/functions.json" />
        <bt:Url id="Taskpane.Url" DefaultValue="{api_base_url}/addin/taskpane.html" />
        <bt:Url id="GetStarted.LearnMoreUrl" DefaultValue="{api_base_url}/docs" />
        <bt:Url id="Icon.16x16" DefaultValue="{api_base_url}/addin/assets/icon-16.png" />
        <bt:Url id="Icon.32x32" DefaultValue="{api_base_url}/addin/assets/icon-32.png" />
      </bt:Urls>
      <bt:ShortStrings>
        <bt:String id="Functions.Namespace" DefaultValue="SENTINEL" />
        <bt:String id="GetStarted.Title" DefaultValue="SENTINEL is ready!" />
        <bt:String id="CommandsGroup.Label" DefaultValue="SENTINEL" />
        <bt:String id="OpenTaskpane.Label" DefaultValue="Open SENTINEL" />
        <bt:String id="OpenTaskpane.Title" DefaultValue="SENTINEL Financial Terminal" />
      </bt:ShortStrings>
      <bt:LongStrings>
        <bt:String id="GetStarted.Description"
          DefaultValue="Use =SENTINEL(ticker, field) for live financial data." />
        <bt:String id="OpenTaskpane.Tooltip"
          DefaultValue="Open the SENTINEL data panel." />
      </bt:LongStrings>
    </Resources>
  </VersionOverrides>
</OfficeApp>
"""

    def generate_functions_json(self, api_base_url: str = "http://localhost:8082") -> dict:
        """Generate the custom functions metadata JSON schema."""
        functions = []
        for field in SUPPORTED_FIELDS:
            desc = _FIELD_DESCRIPTIONS.get(field, f"Fetch {field} for a ticker")
            functions.append({
                "id": f"GET_{field.upper()}",
                "name": f"GET_{field.upper()}",
                "description": desc,
                "parameters": [
                    {
                        "name": "ticker",
                        "description": "Stock ticker symbol (e.g., AAPL, TSLA)",
                        "type": "string",
                    },
                ],
                "result": {
                    "type": "number" if field not in ("sector", "industry", "analyst_rating",
                                                       "earnings_date") else "string",
                    "dimensionality": "scalar",
                },
            })

        # Primary SENTINEL function
        functions.insert(0, {
            "id": "SENTINEL",
            "name": "SENTINEL",
            "description": "Fetch any financial data field. =SENTINEL(\"AAPL\", \"price\")",
            "parameters": [
                {
                    "name": "ticker",
                    "description": "Stock ticker symbol",
                    "type": "string",
                },
                {
                    "name": "field",
                    "description": (
                        "Data field: " + ", ".join(SUPPORTED_FIELDS[:12]) + ", ..."
                    ),
                    "type": "string",
                },
            ],
            "result": {
                "type": "any",
                "dimensionality": "scalar",
            },
        })

        return {
            "functions": functions,
            "allowCustomDataForDataTypeAny": True,
        }

    def generate_taskpane_html(self, api_base_url: str = "http://localhost:8082") -> str:
        """Generate a minimal Office Add-in task pane HTML."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SENTINEL Financial Terminal</title>
  <script src="https://appsforoffice.microsoft.com/lib/1/hosted/office.js"></script>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 12px;
            background: #0d1117; color: #e6edf3; }}
    h1 {{ font-size: 16px; color: #58a6ff; margin-bottom: 8px; }}
    input, select {{ width: 100%; padding: 6px; margin: 4px 0; box-sizing: border-box;
                     background: #161b22; color: #e6edf3; border: 1px solid #30363d;
                     border-radius: 4px; }}
    button {{ width: 100%; padding: 8px; margin-top: 8px; background: #1f6feb;
              color: white; border: none; border-radius: 4px; cursor: pointer;
              font-size: 14px; }}
    button:hover {{ background: #388bfd; }}
    #result {{ margin-top: 12px; padding: 8px; background: #161b22;
               border-radius: 4px; font-family: monospace; min-height: 40px; }}
    .supported-fields {{ font-size: 11px; color: #8b949e; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>SENTINEL</h1>
  <label>Ticker:</label>
  <input id="ticker" type="text" value="AAPL" />
  <label>Field:</label>
  <select id="field">
    {"".join(f'<option value="{f}">{f}</option>' for f in SUPPORTED_FIELDS)}
  </select>
  <button onclick="fetchData()">Fetch Data</button>
  <div id="result">Ready.</div>
  <div class="supported-fields">
    Use =SENTINEL(ticker, field) in any cell. Supported fields: {", ".join(SUPPORTED_FIELDS[:8])}, ...
  </div>

  <script>
    Office.onReady(function() {{ console.log("SENTINEL Add-in ready"); }});

    async function fetchData() {{
      const ticker = document.getElementById('ticker').value.toUpperCase();
      const field = document.getElementById('field').value;
      document.getElementById('result').innerText = 'Loading...';
      try {{
        const resp = await fetch('{api_base_url}/sheets/formula?ticker=' + ticker + '&field=' + field);
        const data = await resp.json();
        document.getElementById('result').innerText = ticker + ' ' + field + ' = ' + data.value;
      }} catch (e) {{
        document.getElementById('result').innerText = 'Error: ' + e.message;
      }}
    }}
  </script>
</body>
</html>
"""


_FIELD_DESCRIPTIONS: Dict[str, str] = {
    "price": "Current market price",
    "change_pct": "Day change in percent",
    "change_abs": "Day change in dollars",
    "volume": "Trading volume (3-month avg)",
    "market_cap": "Market capitalization in USD",
    "pe_ratio": "Price-to-earnings ratio (trailing)",
    "eps": "Earnings per share (trailing 12m)",
    "dividend_yield": "Annual dividend yield",
    "revenue_ttm": "Revenue trailing twelve months",
    "net_income_ttm": "Net income trailing twelve months",
    "ebitda": "EBITDA",
    "debt_to_equity": "Debt to equity ratio",
    "current_ratio": "Current ratio (liquidity)",
    "beta": "Beta (market sensitivity)",
    "52w_high": "52-week high price",
    "52w_low": "52-week low price",
    "rsi_14": "14-day Relative Strength Index",
    "sma_50": "50-day simple moving average",
    "sma_200": "200-day simple moving average",
    "short_interest": "Short interest as % of float",
    "iv_30d": "30-day implied volatility",
    "put_call_ratio": "Put/call open interest ratio",
    "earnings_date": "Next earnings date (YYYY-MM-DD)",
    "analyst_target": "Mean analyst price target",
    "analyst_rating": "Consensus analyst recommendation",
    "open": "Today's opening price",
    "high": "Today's intraday high",
    "low": "Today's intraday low",
    "close": "Most recent closing price",
    "prev_close": "Previous session close",
    "shares_outstanding": "Total shares outstanding",
    "free_float": "Freely tradeable shares (float)",
    "sector": "GICS sector",
    "industry": "Industry classification",
}


# ── 7. FastAPI Router ──────────────────────────────────────────────────────────

sheets_router = APIRouter(prefix="/sheets", tags=["sheets_excel"])

# Module-level singletons
_formula_engine: Optional[SentinelFormulaEngine] = None
_sheets_integration: Optional[GoogleSheetsIntegration] = None
_rtd_server: Optional[ExcelRTDServer] = None
_export_engine: Optional[CSVExportEngine] = None
_webhook_integration: Optional[WebhookIntegration] = None
_manifest_gen: Optional[SentinelAddInManifest] = None


def _get_engine() -> SentinelFormulaEngine:
    global _formula_engine
    if _formula_engine is None:
        _formula_engine = SentinelFormulaEngine()
    return _formula_engine


def _get_sheets() -> GoogleSheetsIntegration:
    global _sheets_integration
    if _sheets_integration is None:
        _sheets_integration = GoogleSheetsIntegration()
    return _sheets_integration


def _get_rtd() -> ExcelRTDServer:
    global _rtd_server
    if _rtd_server is None:
        _rtd_server = ExcelRTDServer()
    return _rtd_server


def _get_export() -> CSVExportEngine:
    global _export_engine
    if _export_engine is None:
        _export_engine = CSVExportEngine()
    return _export_engine


def _get_webhooks() -> WebhookIntegration:
    global _webhook_integration
    if _webhook_integration is None:
        _webhook_integration = WebhookIntegration()
    return _webhook_integration


def _get_manifest() -> SentinelAddInManifest:
    global _manifest_gen
    if _manifest_gen is None:
        _manifest_gen = SentinelAddInManifest()
    return _manifest_gen


# ── Formula endpoints ──────────────────────────────────────────────────────────

@sheets_router.get("/formula", response_model=FormulaResult)
def route_formula(
    ticker: str = Query(..., description="Ticker symbol"),
    field: str = Query(..., description="Field name"),
) -> FormulaResult:
    """Evaluate a single SENTINEL formula cell."""
    engine = _get_engine()
    if field.lower() not in SUPPORTED_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown field '{field}'. Supported: {SUPPORTED_FIELDS}",
        )
    value = engine.evaluate(ticker, field)
    cached = engine._get_cache(f"{ticker.upper()}:{field.lower()}") is not None
    return FormulaResult(
        ticker=ticker.upper(),
        field=field.lower(),
        value=value,
        cached=cached,
        as_of=datetime.now(tz=timezone.utc),
    )


@sheets_router.post("/batch", response_model=BatchFormulaResult)
def route_batch(body: BatchFormulaRequest) -> BatchFormulaResult:
    """Evaluate multiple SENTINEL formula cells in parallel."""
    t0 = time.monotonic()
    engine = _get_engine()
    results = engine.evaluate_batch(body.requests)
    duration_ms = round((time.monotonic() - t0) * 1000, 1)
    return BatchFormulaResult(results=results, duration_ms=duration_ms)


# ── RTD polling endpoint ───────────────────────────────────────────────────────

@sheets_router.get("/rtd/poll")
def route_rtd_poll(
    topics: str = Query(..., description="Pipe-separated: AAPL|price,TSLA|change_pct"),
) -> dict:
    """
    HTTP RTD polling endpoint for Excel VBA shims and Google Sheets =IMPORTDATA().
    topics format: "TICKER|field,TICKER|field,..."
    """
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    rtd = _get_rtd()
    data = rtd.get_poll_data(topic_list)
    return {
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
        "values": data,
    }


@sheets_router.post("/rtd/register")
def route_rtd_register(
    ticker: str = Query(...),
    field: str = Query(...),
) -> dict:
    """Register a new RTD topic and return its topic_id."""
    rtd = _get_rtd()
    topic_id = rtd.register_topic_http(ticker, field)
    return {"topic_id": topic_id, "ticker": ticker.upper(), "field": field.lower()}


# ── Export endpoints ───────────────────────────────────────────────────────────

@sheets_router.post("/export/screener")
def route_export_screener(body: ScreenerExportRequest) -> StreamingResponse:
    """Export screener results to Excel/CSV. Returns file download."""
    export = _get_export()
    file_bytes = export.export_screener(body.criteria, filename=body.filename)
    ext = body.format
    fname = body.filename or f"sentinel_screener_{int(time.time())}.{ext}"
    media_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if ext == "xlsx" else "text/csv"
    )
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@sheets_router.post("/export/portfolio")
def route_export_portfolio(
    holdings: List[Dict[str, Any]],
    fmt: str = Query(default="xlsx", pattern="^(xlsx|csv)$"),
) -> StreamingResponse:
    """Export portfolio snapshot to Excel/CSV."""
    export = _get_export()
    file_bytes = export.export_portfolio(holdings)
    fname = f"sentinel_portfolio_{int(time.time())}.{fmt}"
    media_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if fmt == "xlsx" else "text/csv"
    )
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@sheets_router.get("/export/financials/{ticker}")
def route_export_financials(
    ticker: str,
    fmt: str = Query(default="xlsx", pattern="^(xlsx|csv)$"),
) -> StreamingResponse:
    """Export 3-statement financial model for a ticker."""
    export = _get_export()
    file_bytes = export.export_financials(ticker.upper())
    fname = f"sentinel_{ticker.upper()}_financials_{int(time.time())}.{fmt}"
    media_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if fmt == "xlsx" else "text/csv"
    )
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@sheets_router.get("/export/options/{ticker}")
def route_export_options(
    ticker: str,
    fmt: str = Query(default="xlsx", pattern="^(xlsx|csv)$"),
) -> StreamingResponse:
    """Export full options chain with greeks."""
    export = _get_export()
    file_bytes = export.export_options_chain(ticker.upper())
    fname = f"sentinel_{ticker.upper()}_options_{int(time.time())}.{fmt}"
    media_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if fmt == "xlsx" else "text/csv"
    )
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ── Webhook endpoints ──────────────────────────────────────────────────────────

@sheets_router.post("/webhook/register")
def route_webhook_register(body: WebhookRegistration) -> dict:
    """Register a new push webhook for real-time SENTINEL data."""
    wh = _get_webhooks()
    webhook_id = wh.register(
        url=body.url,
        tickers=body.tickers,
        fields=body.fields,
        frequency=body.frequency,
    )
    return {"webhook_id": webhook_id, "status": "registered", "url": body.url}


@sheets_router.get("/webhook/list", response_model=List[WebhookInfo])
def route_webhook_list() -> List[WebhookInfo]:
    """List all registered webhooks."""
    return _get_webhooks().list_webhooks()


@sheets_router.delete("/webhook/{webhook_id}")
def route_webhook_delete(webhook_id: str) -> dict:
    """Deactivate a webhook."""
    ok = _get_webhooks().deactivate(webhook_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"webhook_id": webhook_id, "status": "deactivated"}


# ── Manifest endpoints ─────────────────────────────────────────────────────────

@sheets_router.get("/manifest")
def route_manifest(
    api_base_url: str = Query(default="http://localhost:8082"),
) -> PlainTextResponse:
    """Return Office Add-in manifest.xml."""
    manifest = _get_manifest().generate_manifest(api_base_url=api_base_url)
    return PlainTextResponse(content=manifest, media_type="application/xml")


@sheets_router.get("/manifest/functions.json")
def route_functions_json(
    api_base_url: str = Query(default="http://localhost:8082"),
) -> dict:
    """Return custom Excel functions metadata JSON."""
    return _get_manifest().generate_functions_json(api_base_url=api_base_url)


@sheets_router.get("/manifest/taskpane.html")
def route_taskpane_html(
    api_base_url: str = Query(default="http://localhost:8082"),
) -> PlainTextResponse:
    """Return task pane HTML for the Office Add-in."""
    html_content = _get_manifest().generate_taskpane_html(api_base_url=api_base_url)
    return PlainTextResponse(content=html_content, media_type="text/html")


# ── Supported fields discovery ─────────────────────────────────────────────────

@sheets_router.get("/fields")
def route_fields() -> dict:
    """Return all supported SENTINEL formula fields with descriptions."""
    return {
        "fields": [
            {"name": f, "description": _FIELD_DESCRIPTIONS.get(f, f)}
            for f in SUPPORTED_FIELDS
        ],
        "count": len(SUPPORTED_FIELDS),
    }


# ── Standalone server ──────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8082) -> None:
    """Run the sheets/Excel plugin as a standalone FastAPI server."""
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(
            title="SENTINEL Sheets & Excel Plugin",
            description="=SENTINEL(ticker, field) formula backend + RTD + webhooks",
            version="2.0.0",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.include_router(sheets_router)
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError as exc:
        logger.error("uvicorn not installed, cannot run standalone server: %s", exc)


# ── Init DB on module load ─────────────────────────────────────────────────────
try:
    _ensure_db()
except Exception as _db_exc:
    logger.warning("sheets_excel_plugin: DB init failed: %s", _db_exc)


if __name__ == "__main__":
    run_server()
