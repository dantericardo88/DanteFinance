"""
historical_ohlcv_v3.py — Production-grade 1-minute intraday OHLCV engine (dim_003).

Primary source: Alpaca Historical Bars API (free with account)
  GET https://data.alpaca.markets/v2/stocks/{symbol}/bars
  - Timeframe: 1Min, session=extended (pre/post-market included)
  - Paginated via next_page_token until exhausted
  - 10+ years of 1-minute bars with free API key

Fallback (no Alpaca keys): yfinance 7-day 1-min window, clearly labelled.

Storage: DuckDB at sentinel/data/ohlcv_intraday.duckdb
  - Columnar, compressed, fast range queries
  - Schema: symbol, ts, open, high, low, close, volume, vwap, trade_count, extended_hours

Multi-timeframe aggregation from 1-min base via DuckDB time_bucket:
  5min, 15min, 30min, 1h, 4h

Data quality:
  - Gap detection: missing bars 09:30–16:00 ET on trading days
  - Outlier detection: >5x volume spike, >10% price move in a single bar
  - Stale data flag: no new bars within expected latency window

FastAPI router at /ohlcv/v3:
  GET  /bars/{ticker}
  GET  /aggregate/{ticker}
  POST /backfill/{ticker}
  GET  /quality/{ticker}
  GET  /available-range/{ticker}

Dependencies: requests, duckdb, pandas, numpy, fastapi, yfinance
"""
from __future__ import annotations

import os
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

_ALPACA_DATA_BASE = "https://data.alpaca.markets/v2/stocks"
_ALPACA_BARS_MAX_PER_PAGE = 10_000
_ALPACA_TIMEOUT = 30          # seconds per HTTP call
_ALPACA_RETRY_MAX = 3
_ALPACA_RETRY_BACKOFF = 1.5   # exponential base

# DuckDB path — sentinel/data/ohlcv_intraday.duckdb
_SENTINEL_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _SENTINEL_ROOT / "sentinel" / "data"
_DB_PATH = _DATA_DIR / "ohlcv_intraday.duckdb"

# Chunk size for backfill batches (days per HTTP window)
_BACKFILL_CHUNK_DAYS = 30

# Market session ET times
_MARKET_OPEN_H, _MARKET_OPEN_M = 9, 30
_MARKET_CLOSE_H, _MARKET_CLOSE_M = 16, 0

# Outlier thresholds
_VOLUME_SPIKE_FACTOR = 5.0
_PRICE_MOVE_THRESHOLD = 0.10   # 10% single-bar move

# Known US equity holidays 2015-2026 (abbreviated; extend as needed)
_US_HOLIDAYS: set[date] = {
    # 2015
    date(2015, 1, 1), date(2015, 1, 19), date(2015, 2, 16), date(2015, 4, 3),
    date(2015, 5, 25), date(2015, 7, 3), date(2015, 9, 7), date(2015, 11, 26),
    date(2015, 12, 25),
    # 2016
    date(2016, 1, 1), date(2016, 1, 18), date(2016, 2, 15), date(2016, 3, 25),
    date(2016, 5, 30), date(2016, 7, 4), date(2016, 9, 5), date(2016, 11, 24),
    date(2016, 12, 26),
    # 2017
    date(2017, 1, 2), date(2017, 1, 16), date(2017, 2, 20), date(2017, 4, 14),
    date(2017, 5, 29), date(2017, 7, 4), date(2017, 9, 4), date(2017, 11, 23),
    date(2017, 12, 25),
    # 2018
    date(2018, 1, 1), date(2018, 1, 15), date(2018, 2, 19), date(2018, 3, 30),
    date(2018, 5, 28), date(2018, 7, 4), date(2018, 9, 3), date(2018, 11, 22),
    date(2018, 12, 5), date(2018, 12, 25),
    # 2019
    date(2019, 1, 1), date(2019, 1, 21), date(2019, 2, 18), date(2019, 4, 19),
    date(2019, 5, 27), date(2019, 7, 4), date(2019, 9, 2), date(2019, 11, 28),
    date(2019, 12, 25),
    # 2020
    date(2020, 1, 1), date(2020, 1, 20), date(2020, 2, 17), date(2020, 4, 10),
    date(2020, 5, 25), date(2020, 7, 3), date(2020, 9, 7), date(2020, 11, 26),
    date(2020, 12, 25),
    # 2021
    date(2021, 1, 1), date(2021, 1, 18), date(2021, 2, 15), date(2021, 4, 2),
    date(2021, 5, 31), date(2021, 7, 5), date(2021, 9, 6), date(2021, 11, 25),
    date(2021, 12, 24),
    # 2022
    date(2022, 1, 17), date(2022, 2, 21), date(2022, 4, 15),
    date(2022, 5, 30), date(2022, 6, 20), date(2022, 7, 4), date(2022, 9, 5),
    date(2022, 11, 24), date(2022, 12, 26),
    # 2023
    date(2023, 1, 2), date(2023, 1, 16), date(2023, 2, 20), date(2023, 4, 7),
    date(2023, 5, 29), date(2023, 6, 19), date(2023, 7, 4), date(2023, 9, 4),
    date(2023, 11, 23), date(2023, 12, 25),
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}

# Multi-timeframe specs: label -> minutes
_TIMEFRAME_MINUTES: Dict[str, int] = {
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "4h": 240,
}


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class OHLCVBar(BaseModel):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    extended_hours: bool = False


class AggregatedBar(BaseModel):
    symbol: str
    ts: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    bar_count: int = 0


class QualityReport(BaseModel):
    symbol: str
    checked_at: datetime
    total_bars: int
    regular_session_bars: int
    extended_hours_bars: int
    missing_bar_count: int
    missing_bar_pct: float
    outlier_bars: int
    volume_spikes: int
    price_spikes: int
    stale: bool
    stale_reason: Optional[str] = None
    data_start: Optional[datetime] = None
    data_end: Optional[datetime] = None
    source: str


class AvailableRange(BaseModel):
    symbol: str
    earliest_ts: Optional[datetime]
    latest_ts: Optional[datetime]
    total_bars: int
    trading_days_covered: int
    source: str
    alpaca_available: bool


class BackfillStatus(BaseModel):
    symbol: str
    status: str
    bars_inserted: int
    chunks_fetched: int
    start_date: str
    end_date: str
    source: str
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DuckDB schema & connection management
# ---------------------------------------------------------------------------

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv_intraday (
    symbol       TEXT        NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,
    open         DOUBLE      NOT NULL,
    high         DOUBLE      NOT NULL,
    low          DOUBLE      NOT NULL,
    close        DOUBLE      NOT NULL,
    volume       BIGINT      NOT NULL,
    vwap         DOUBLE,
    trade_count  INTEGER,
    extended_hours BOOLEAN   NOT NULL DEFAULT FALSE,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS backfill_metadata (
    symbol          TEXT        NOT NULL,
    last_backfill   TIMESTAMPTZ,
    earliest_bar    TIMESTAMPTZ,
    latest_bar      TIMESTAMPTZ,
    total_bars      BIGINT      DEFAULT 0,
    source          TEXT        NOT NULL DEFAULT 'alpaca',
    PRIMARY KEY (symbol)
);
"""

_INSERT_SQL = """
INSERT OR IGNORE INTO ohlcv_intraday
    (symbol, ts, open, high, low, close, volume, vwap, trade_count, extended_hours)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _ensure_db() -> duckdb.DuckDBPyConnection:
    """Open (or create) the DuckDB database, run DDL if needed, return connection."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(_DB_PATH))
    conn.execute(_SCHEMA_DDL)
    return conn


def _get_conn() -> duckdb.DuckDBPyConnection:
    """Thread-local connection for request-scoped use. Caller must close."""
    return _ensure_db()


# ---------------------------------------------------------------------------
# Alpaca HTTP client helpers
# ---------------------------------------------------------------------------

def _alpaca_headers() -> Dict[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    headers: Dict[str, str] = {"Accept": "application/json"}
    if key and secret:
        headers["APCA-API-KEY-ID"] = key
        headers["APCA-API-SECRET-KEY"] = secret
    return headers


def _alpaca_available() -> bool:
    """True when both Alpaca env vars are present."""
    return bool(
        os.environ.get("ALPACA_API_KEY", "").strip()
        and os.environ.get("ALPACA_SECRET_KEY", "").strip()
    )


def _alpaca_request(
    url: str,
    params: Dict[str, Any],
    timeout: int = _ALPACA_TIMEOUT,
) -> Dict[str, Any]:
    """GET with exponential-backoff retry on 429 / 5xx."""
    headers = _alpaca_headers()
    for attempt in range(1, _ALPACA_RETRY_MAX + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = _ALPACA_RETRY_BACKOFF ** attempt
                logger.warning(
                    "Alpaca rate-limited",
                    attempt=attempt,
                    wait_s=wait,
                    url=url,
                )
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = _ALPACA_RETRY_BACKOFF ** attempt
                logger.warning(
                    "Alpaca 5xx error",
                    status=resp.status_code,
                    attempt=attempt,
                    wait_s=wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warning("Alpaca timeout", attempt=attempt, url=url)
            if attempt == _ALPACA_RETRY_MAX:
                raise
            time.sleep(_ALPACA_RETRY_BACKOFF ** attempt)
        except requests.exceptions.RequestException as exc:
            if attempt == _ALPACA_RETRY_MAX:
                raise
            logger.warning("Alpaca request error", error=str(exc), attempt=attempt)
            time.sleep(_ALPACA_RETRY_BACKOFF ** attempt)
    return {}


# ---------------------------------------------------------------------------
# Alpaca paginated bar fetcher
# ---------------------------------------------------------------------------

def _fetch_alpaca_bars_window(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1Min",
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield raw bar dicts from Alpaca for [start, end).
    Handles pagination via next_page_token until None.

    Each bar dict keys: t, o, h, l, c, v, vw, n
    """
    url = f"{_ALPACA_DATA_BASE}/{symbol}/bars"
    params: Dict[str, Any] = {
        "timeframe": timeframe,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": _ALPACA_BARS_MAX_PER_PAGE,
        "adjustment": "all",      # split + dividend adjusted
        "feed": "iex",            # free IEX feed
        "session": "extended",    # include pre/post-market
        "sort": "asc",
    }

    page = 0
    while True:
        data = _alpaca_request(url, params)
        bars = data.get("bars") or []
        for bar in bars:
            yield bar
        next_token = data.get("next_page_token")
        if not next_token:
            break
        params["page_token"] = next_token
        page += 1
        # Small courtesy sleep between pages to avoid sustained hammering
        if page % 5 == 0:
            time.sleep(0.2)


def _parse_alpaca_bar(symbol: str, bar: Dict[str, Any]) -> Optional[Tuple]:
    """Convert raw Alpaca bar dict to a tuple matching INSERT_SQL parameter order."""
    try:
        ts_str = bar.get("t", "")
        if not ts_str:
            return None
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

        o = float(bar.get("o", 0))
        h = float(bar.get("h", 0))
        lo = float(bar.get("l", 0))
        c = float(bar.get("c", 0))
        v = int(bar.get("v", 0))
        vw = bar.get("vw")
        vwap = float(vw) if vw is not None else None
        n = bar.get("n")
        trade_count = int(n) if n is not None else None

        # Classify extended-hours: before 09:30 ET or after 16:00 ET
        ts_et = ts.astimezone(ET)
        market_open = ts_et.replace(hour=_MARKET_OPEN_H, minute=_MARKET_OPEN_M, second=0, microsecond=0)
        market_close = ts_et.replace(hour=_MARKET_CLOSE_H, minute=_MARKET_CLOSE_M, second=0, microsecond=0)
        extended = ts_et < market_open or ts_et >= market_close

        # Sanity: skip bars with obviously bad prices
        if o <= 0 or h <= 0 or lo <= 0 or c <= 0:
            return None
        if lo > h:
            return None

        return (symbol, ts, o, h, lo, c, v, vwap, trade_count, extended)
    except Exception as exc:
        logger.warning("Failed to parse Alpaca bar", error=str(exc), bar=bar)
        return None


# ---------------------------------------------------------------------------
# yfinance fallback
# ---------------------------------------------------------------------------

def _fetch_yfinance_bars(
    symbol: str,
    period: str = "7d",
    interval: str = "1m",
) -> pd.DataFrame:
    """
    Fallback: use yfinance for up to 7 days of 1-min bars.
    Returns a DataFrame with columns matching our schema (or empty DF).
    Logs a prominent warning so the caller knows data is limited.
    """
    if not _HAS_YFINANCE:
        logger.error("yfinance not installed — cannot fall back")
        return pd.DataFrame()

    warnings.warn(
        f"[SENTINEL OHLCV-V3] ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
        f"Falling back to yfinance for {symbol}. "
        f"yfinance provides at most 7 days of 1-min bars. "
        f"Set Alpaca credentials to unlock 10+ years of history.",
        UserWarning,
        stacklevel=4,
    )
    logger.warning(
        "Alpaca credentials missing — yfinance fallback active",
        symbol=symbol,
        history_limit="7 days",
        remedy="Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables",
    )

    try:
        tkr = yf.Ticker(symbol)
        df = tkr.history(period=period, interval=interval, prepost=True)
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        # yfinance columns: Datetime, Open, High, Low, Close, Volume, Dividends, Stock Splits
        df = df.rename(columns={
            "Datetime": "ts",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df["symbol"] = symbol
        df["vwap"] = None
        df["trade_count"] = None

        # Classify extended hours
        def _is_extended(ts_val: Any) -> bool:
            if hasattr(ts_val, "tzinfo") and ts_val.tzinfo is not None:
                ts_et = ts_val.astimezone(ET)
            else:
                ts_et = ts_val.replace(tzinfo=ET)
            market_open = ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = ts_et.replace(hour=16, minute=0, second=0, microsecond=0)
            return ts_et < market_open or ts_et >= market_close

        df["extended_hours"] = df["ts"].apply(_is_extended)

        # Ensure UTC timestamps
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("UTC")
        else:
            df["ts"] = df["ts"].dt.tz_convert("UTC")

        keep_cols = ["symbol", "ts", "open", "high", "low", "close", "volume",
                     "vwap", "trade_count", "extended_hours"]
        return df[[c for c in keep_cols if c in df.columns]]

    except Exception as exc:
        logger.error("yfinance fallback failed", symbol=symbol, error=str(exc))
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# DuckDB insert helpers
# ---------------------------------------------------------------------------

def _insert_bars_df(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """
    Bulk-insert a DataFrame of bars into ohlcv_intraday using INSERT OR IGNORE.
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    required = ["symbol", "ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing columns in bars DataFrame", missing=missing)
        return 0

    # Fill optional columns
    if "vwap" not in df.columns:
        df["vwap"] = None
    if "trade_count" not in df.columns:
        df["trade_count"] = None
    if "extended_hours" not in df.columns:
        df["extended_hours"] = False

    # Cast types
    df["volume"] = df["volume"].fillna(0).astype(int)
    df["extended_hours"] = df["extended_hours"].fillna(False).astype(bool)

    before = conn.execute("SELECT COUNT(*) FROM ohlcv_intraday").fetchone()[0]

    # Register DataFrame as a temporary view and upsert
    conn.register("_bars_tmp", df)
    conn.execute("""
        INSERT OR IGNORE INTO ohlcv_intraday
            (symbol, ts, open, high, low, close, volume, vwap, trade_count, extended_hours)
        SELECT symbol, ts, open, high, low, close, volume, vwap, trade_count, extended_hours
        FROM _bars_tmp
    """)
    conn.unregister("_bars_tmp")

    after = conn.execute("SELECT COUNT(*) FROM ohlcv_intraday").fetchone()[0]
    return after - before


def _insert_bars_tuples(
    conn: duckdb.DuckDBPyConnection,
    tuples: List[Tuple],
) -> int:
    """Insert a list of tuples (from _parse_alpaca_bar) — INSERT OR IGNORE."""
    if not tuples:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM ohlcv_intraday").fetchone()[0]
    conn.executemany(_INSERT_SQL, tuples)
    after = conn.execute("SELECT COUNT(*) FROM ohlcv_intraday").fetchone()[0]
    return after - before


def _update_backfill_metadata(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    source: str,
) -> None:
    """Refresh backfill_metadata stats for a symbol after insert."""
    conn.execute("""
        INSERT INTO backfill_metadata (symbol, last_backfill, earliest_bar, latest_bar, total_bars, source)
        SELECT
            ? AS symbol,
            NOW() AS last_backfill,
            MIN(ts) AS earliest_bar,
            MAX(ts) AS latest_bar,
            COUNT(*) AS total_bars,
            ? AS source
        FROM ohlcv_intraday
        WHERE symbol = ?
        ON CONFLICT (symbol) DO UPDATE SET
            last_backfill = excluded.last_backfill,
            earliest_bar  = excluded.earliest_bar,
            latest_bar    = excluded.latest_bar,
            total_bars    = excluded.total_bars,
            source        = excluded.source
    """, [symbol, source, symbol])


# ---------------------------------------------------------------------------
# Backfill engine
# ---------------------------------------------------------------------------

def _trading_day_chunks(
    start: date,
    end: date,
    chunk_days: int = _BACKFILL_CHUNK_DAYS,
) -> Generator[Tuple[datetime, datetime], None, None]:
    """Yield (chunk_start, chunk_end) UTC datetime pairs for backfill pagination."""
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        # Convert to UTC datetimes spanning full day
        yield (
            datetime(cursor.year, cursor.month, cursor.day, 0, 0, 0, tzinfo=UTC),
            datetime(chunk_end.year, chunk_end.month, chunk_end.day, 23, 59, 59, tzinfo=UTC),
        )
        cursor = chunk_end + timedelta(days=1)


def backfill_symbol(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> BackfillStatus:
    """
    Fetch all available 1-min history for *symbol* and store in DuckDB.

    Uses Alpaca when credentials are present, falls back to yfinance (7 days).
    Processes in 30-day chunks to stay well within Alpaca page limits.
    """
    symbol = symbol.upper().strip()
    warnings_list: List[str] = []

    if start_date is None:
        start_date = date(2015, 1, 2)   # Alpaca 1-min history from ~2015
    if end_date is None:
        end_date = date.today()

    conn = _get_conn()
    total_inserted = 0
    chunks_fetched = 0
    source = "alpaca"

    if not _alpaca_available():
        # yfinance fallback: only last 7 days
        warnings_list.append(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
            "Falling back to yfinance — maximum 7 days of 1-min history available. "
            "Set Alpaca credentials to unlock 10+ years of backfill."
        )
        logger.warning(
            "Backfill using yfinance fallback",
            symbol=symbol,
            max_history="7 days",
        )
        source = "yfinance"
        df = _fetch_yfinance_bars(symbol, period="7d", interval="1m")
        inserted = _insert_bars_df(conn, df)
        total_inserted += inserted
        chunks_fetched = 1
    else:
        logger.info(
            "Starting Alpaca backfill",
            symbol=symbol,
            start=str(start_date),
            end=str(end_date),
        )
        for chunk_start, chunk_end in _trading_day_chunks(start_date, end_date):
            chunk_tuples: List[Tuple] = []
            try:
                for raw_bar in _fetch_alpaca_bars_window(symbol, chunk_start, chunk_end):
                    parsed = _parse_alpaca_bar(symbol, raw_bar)
                    if parsed is not None:
                        chunk_tuples.append(parsed)
            except Exception as exc:
                msg = f"Chunk {chunk_start.date()} – {chunk_end.date()} failed: {exc}"
                logger.warning("Backfill chunk error", symbol=symbol, error=str(exc))
                warnings_list.append(msg)
                continue

            if chunk_tuples:
                inserted = _insert_bars_tuples(conn, chunk_tuples)
                total_inserted += inserted
            chunks_fetched += 1

            logger.info(
                "Backfill chunk complete",
                symbol=symbol,
                chunk_start=str(chunk_start.date()),
                chunk_end=str(chunk_end.date()),
                bars_in_chunk=len(chunk_tuples),
                inserted=total_inserted,
            )

    _update_backfill_metadata(conn, symbol, source)
    conn.close()

    logger.info(
        "Backfill finished",
        symbol=symbol,
        total_inserted=total_inserted,
        chunks=chunks_fetched,
        source=source,
    )

    return BackfillStatus(
        symbol=symbol,
        status="complete",
        bars_inserted=total_inserted,
        chunks_fetched=chunks_fetched,
        start_date=str(start_date),
        end_date=str(end_date),
        source=source,
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# Multi-timeframe aggregation
# ---------------------------------------------------------------------------

_AGG_SQL_TEMPLATE = """
SELECT
    symbol,
    time_bucket(INTERVAL '{bucket}', ts) AS ts,
    FIRST(open  ORDER BY ts) AS open,
    MAX(high)                AS high,
    MIN(low)                 AS low,
    LAST(close  ORDER BY ts) AS close,
    SUM(volume)              AS volume,
    SUM(vwap * volume) / NULLIF(SUM(volume), 0) AS vwap,
    COUNT(*)                 AS bar_count
FROM ohlcv_intraday
WHERE symbol = ?
  AND ts >= ?
  AND ts <= ?
  {session_filter}
GROUP BY symbol, time_bucket(INTERVAL '{bucket}', ts)
ORDER BY ts ASC
"""

_TIMEFRAME_TO_BUCKET: Dict[str, str] = {
    "1Min":  "1 minute",
    "1min":  "1 minute",
    "5min":  "5 minutes",
    "15min": "15 minutes",
    "30min": "30 minutes",
    "1h":    "1 hour",
    "4h":    "4 hours",
    "1d":    "1 day",
}


def aggregate_bars(
    symbol: str,
    timeframe: str = "1h",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    extended_hours: bool = True,
) -> List[AggregatedBar]:
    """
    Aggregate stored 1-min bars to the requested timeframe using DuckDB time_bucket.
    Supported timeframes: 1Min, 5min, 15min, 30min, 1h, 4h, 1d.
    """
    symbol = symbol.upper().strip()
    bucket = _TIMEFRAME_TO_BUCKET.get(timeframe)
    if bucket is None:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Valid: {list(_TIMEFRAME_TO_BUCKET.keys())}"
        )

    if start is None:
        start = datetime(2015, 1, 1, tzinfo=UTC)
    if end is None:
        end = datetime.now(UTC)

    session_filter = "" if extended_hours else "AND extended_hours = FALSE"
    sql = _AGG_SQL_TEMPLATE.format(bucket=bucket, session_filter=session_filter)

    conn = _get_conn()
    try:
        rows = conn.execute(sql, [symbol, start, end]).fetchall()
    finally:
        conn.close()

    results: List[AggregatedBar] = []
    for row in rows:
        sym, ts, o, h, lo, c, v, vw, bar_count = row
        results.append(AggregatedBar(
            symbol=sym,
            ts=ts,
            timeframe=timeframe,
            open=float(o),
            high=float(h),
            low=float(lo),
            close=float(c),
            volume=int(v),
            vwap=float(vw) if vw is not None else None,
            bar_count=int(bar_count),
        ))
    return results


# ---------------------------------------------------------------------------
# Raw bar query
# ---------------------------------------------------------------------------

def query_bars(
    symbol: str,
    timeframe: str = "1Min",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    extended_hours: bool = True,
    limit: int = 10_000,
) -> List[OHLCVBar]:
    """
    Return stored bars for *symbol* in [start, end].
    For timeframes other than 1Min, delegates to aggregate_bars.
    """
    symbol = symbol.upper().strip()

    if timeframe != "1Min" and timeframe != "1min":
        agg = aggregate_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            extended_hours=extended_hours,
        )
        # Coerce to OHLCVBar
        return [
            OHLCVBar(
                symbol=a.symbol,
                ts=a.ts,
                open=a.open,
                high=a.high,
                low=a.low,
                close=a.close,
                volume=a.volume,
                vwap=a.vwap,
                trade_count=None,
                extended_hours=extended_hours,
            )
            for a in agg
        ]

    if start is None:
        start = datetime(2015, 1, 1, tzinfo=UTC)
    if end is None:
        end = datetime.now(UTC)

    session_filter = "" if extended_hours else "AND extended_hours = FALSE"
    sql = f"""
        SELECT symbol, ts, open, high, low, close, volume, vwap, trade_count, extended_hours
        FROM ohlcv_intraday
        WHERE symbol = ?
          AND ts >= ?
          AND ts <= ?
          {session_filter}
        ORDER BY ts ASC
        LIMIT ?
    """
    conn = _get_conn()
    try:
        rows = conn.execute(sql, [symbol, start, end, limit]).fetchall()
    finally:
        conn.close()

    results: List[OHLCVBar] = []
    for row in rows:
        sym, ts, o, h, lo, c, v, vw, tc, ext = row
        results.append(OHLCVBar(
            symbol=sym,
            ts=ts,
            open=float(o),
            high=float(h),
            low=float(lo),
            close=float(c),
            volume=int(v),
            vwap=float(vw) if vw is not None else None,
            trade_count=int(tc) if tc is not None else None,
            extended_hours=bool(ext),
        ))
    return results


# ---------------------------------------------------------------------------
# Data quality engine
# ---------------------------------------------------------------------------

def _expected_regular_bars(trading_day: date) -> int:
    """Number of 1-min bars expected in a regular 09:30–16:00 ET session."""
    # 09:30 to 16:00 = 390 minutes = 390 bars
    return 390


def _is_trading_day(d: date) -> bool:
    """True if d is a weekday and not a US equity holiday."""
    return d.weekday() < 5 and d not in _US_HOLIDAYS


def _get_trading_days(start: date, end: date) -> List[date]:
    """Return list of trading days in [start, end] inclusive."""
    result = []
    cursor = start
    while cursor <= end:
        if _is_trading_day(cursor):
            result.append(cursor)
        cursor += timedelta(days=1)
    return result


def _detect_gaps(
    df: pd.DataFrame,
    trading_days: List[date],
) -> int:
    """
    Count missing 1-min bars during regular session (09:30–16:00 ET).
    A bar is 'missing' if a trading day has fewer bars than expected.
    """
    if df.empty:
        return sum(_expected_regular_bars(d) for d in trading_days)

    # Filter to regular session
    df_et = df.copy()
    df_et["ts_et"] = pd.to_datetime(df_et["ts"]).dt.tz_convert(ET)
    df_et["date_et"] = df_et["ts_et"].dt.date
    df_et["hour"] = df_et["ts_et"].dt.hour
    df_et["minute"] = df_et["ts_et"].dt.minute

    regular = df_et[
        ((df_et["hour"] == 9) & (df_et["minute"] >= 30)) |
        ((df_et["hour"] > 9) & (df_et["hour"] < 16))
    ]

    bars_by_day = regular.groupby("date_et").size().to_dict()

    missing = 0
    for td in trading_days:
        actual = bars_by_day.get(td, 0)
        expected = _expected_regular_bars(td)
        if actual < expected:
            missing += expected - actual
    return missing


def _detect_outliers(df: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Returns (total_outliers, volume_spikes, price_spikes).
    Volume spike: volume > 5x rolling 20-bar median.
    Price spike: |close - open| / open > 10% in a single bar.
    """
    if df.empty or len(df) < 2:
        return 0, 0, 0

    volume = df["volume"].astype(float)
    roll_med = volume.rolling(window=20, min_periods=1).median()
    vol_spikes = int((volume > _VOLUME_SPIKE_FACTOR * roll_med).sum())

    price_move = (df["close"].astype(float) - df["open"].astype(float)).abs() / df["open"].astype(float).replace(0, np.nan)
    price_spikes = int((price_move > _PRICE_MOVE_THRESHOLD).sum())

    return vol_spikes + price_spikes, vol_spikes, price_spikes


def run_quality_check(symbol: str) -> QualityReport:
    """
    Run data quality checks for *symbol* against stored bars.
    Returns a QualityReport with gap counts, outliers, stale flag.
    """
    symbol = symbol.upper().strip()
    now = datetime.now(UTC)

    conn = _get_conn()
    try:
        # Metadata
        meta = conn.execute(
            "SELECT source, earliest_bar, latest_bar, total_bars "
            "FROM backfill_metadata WHERE symbol = ?",
            [symbol],
        ).fetchone()

        total_bars = conn.execute(
            "SELECT COUNT(*) FROM ohlcv_intraday WHERE symbol = ?",
            [symbol],
        ).fetchone()[0]

        regular_bars = conn.execute(
            "SELECT COUNT(*) FROM ohlcv_intraday WHERE symbol = ? AND extended_hours = FALSE",
            [symbol],
        ).fetchone()[0]

        extended_bars = total_bars - regular_bars

        # Pull recent 30 trading days for quality assessment (avoid loading years of data)
        thirty_days_ago = now - timedelta(days=45)
        rows = conn.execute("""
            SELECT symbol, ts, open, high, low, close, volume, vwap, trade_count, extended_hours
            FROM ohlcv_intraday
            WHERE symbol = ?
              AND ts >= ?
            ORDER BY ts ASC
        """, [symbol, thirty_days_ago]).fetchall()

        df = pd.DataFrame(rows, columns=[
            "symbol", "ts", "open", "high", "low", "close",
            "volume", "vwap", "trade_count", "extended_hours"
        ])

        # Trading days in window
        window_start = thirty_days_ago.date()
        window_end = now.date()
        trading_days = _get_trading_days(window_start, window_end)

        missing_bars = _detect_gaps(df, trading_days)
        expected_total = sum(_expected_regular_bars(d) for d in trading_days)
        missing_pct = (missing_bars / expected_total * 100) if expected_total > 0 else 0.0

        total_outliers, vol_spikes, price_spikes = _detect_outliers(df)

        # Stale flag
        stale = False
        stale_reason = None
        if meta and meta[2]:
            latest_bar_ts = meta[2]
            if hasattr(latest_bar_ts, "tzinfo") and latest_bar_ts.tzinfo is None:
                latest_bar_ts = latest_bar_ts.replace(tzinfo=UTC)
            # Stale if latest bar is more than 3 trading days old
            age_days = (now - latest_bar_ts).days
            if age_days > 3:
                stale = True
                stale_reason = f"Latest bar is {age_days} days old (>{3} threshold)"
        elif total_bars == 0:
            stale = True
            stale_reason = "No bars found in database — backfill not run"

        source = meta[0] if meta else "unknown"
        data_start = meta[1] if meta else None
        data_end = meta[2] if meta else None

    finally:
        conn.close()

    return QualityReport(
        symbol=symbol,
        checked_at=now,
        total_bars=total_bars,
        regular_session_bars=regular_bars,
        extended_hours_bars=extended_bars,
        missing_bar_count=missing_bars,
        missing_bar_pct=round(missing_pct, 2),
        outlier_bars=total_outliers,
        volume_spikes=vol_spikes,
        price_spikes=price_spikes,
        stale=stale,
        stale_reason=stale_reason,
        data_start=data_start,
        data_end=data_end,
        source=source,
    )


# ---------------------------------------------------------------------------
# Available-range query
# ---------------------------------------------------------------------------

def get_available_range(symbol: str) -> AvailableRange:
    """Return earliest/latest timestamps and coverage stats for *symbol*."""
    symbol = symbol.upper().strip()
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT MIN(ts), MAX(ts), COUNT(*)
            FROM ohlcv_intraday
            WHERE symbol = ?
        """, [symbol]).fetchone()

        earliest_ts, latest_ts, total_bars = row if row else (None, None, 0)

        trading_days = 0
        if earliest_ts and latest_ts:
            d_start = earliest_ts.date() if hasattr(earliest_ts, "date") else earliest_ts
            d_end = latest_ts.date() if hasattr(latest_ts, "date") else latest_ts
            trading_days = len(_get_trading_days(d_start, d_end))

        meta = conn.execute(
            "SELECT source FROM backfill_metadata WHERE symbol = ?",
            [symbol],
        ).fetchone()
        source = meta[0] if meta else "unknown"

    finally:
        conn.close()

    return AvailableRange(
        symbol=symbol,
        earliest_ts=earliest_ts,
        latest_ts=latest_ts,
        total_bars=int(total_bars or 0),
        trading_days_covered=trading_days,
        source=source,
        alpaca_available=_alpaca_available(),
    )


# ---------------------------------------------------------------------------
# Live incremental fetch (non-backfill: top up to now from last bar)
# ---------------------------------------------------------------------------

def incremental_update(symbol: str) -> int:
    """
    Fetch bars from the last stored bar to now and insert.
    Returns number of new bars inserted.
    Designed for use as a periodic refresh (e.g. scheduled job).
    """
    symbol = symbol.upper().strip()

    conn = _get_conn()
    row = conn.execute(
        "SELECT MAX(ts) FROM ohlcv_intraday WHERE symbol = ?",
        [symbol],
    ).fetchone()
    conn.close()

    last_ts = row[0] if row and row[0] else None
    if last_ts is None:
        # No data yet — do a full backfill
        result = backfill_symbol(symbol)
        return result.bars_inserted

    if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=UTC)

    start = last_ts + timedelta(minutes=1)
    end = datetime.now(UTC)

    if not _alpaca_available():
        logger.warning(
            "Incremental update using yfinance fallback",
            symbol=symbol,
            note="Alpaca keys not set; only last 7d available",
        )
        df = _fetch_yfinance_bars(symbol, period="7d", interval="1m")
        conn = _get_conn()
        inserted = _insert_bars_df(conn, df)
        _update_backfill_metadata(conn, symbol, "yfinance")
        conn.close()
        return inserted

    chunk_tuples: List[Tuple] = []
    for raw_bar in _fetch_alpaca_bars_window(symbol, start, end):
        parsed = _parse_alpaca_bar(symbol, raw_bar)
        if parsed is not None:
            chunk_tuples.append(parsed)

    conn = _get_conn()
    inserted = _insert_bars_tuples(conn, chunk_tuples)
    if inserted > 0:
        _update_backfill_metadata(conn, symbol, "alpaca")
    conn.close()

    logger.info("Incremental update done", symbol=symbol, new_bars=inserted)
    return inserted


# ---------------------------------------------------------------------------
# FastAPI router — /ohlcv/v3
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/ohlcv/v3", tags=["OHLCV-v3"])


@router.get(
    "/bars/{ticker}",
    response_model=List[OHLCVBar],
    summary="Get raw 1-min OHLCV bars (or aggregate for other timeframes)",
)
def get_bars(
    ticker: str,
    timeframe: str = Query(default="1Min", description="1Min | 5min | 15min | 30min | 1h | 4h | 1d"),
    start: Optional[str] = Query(default=None, description="ISO date or datetime (e.g. 2020-01-01)"),
    end: Optional[str] = Query(default=None, description="ISO date or datetime"),
    extended_hours: bool = Query(default=True, description="Include pre/post-market bars"),
    limit: int = Query(default=10_000, le=50_000, description="Max bars returned"),
) -> List[OHLCVBar]:
    """
    Return stored OHLCV bars for *ticker*.

    For timeframe=1Min, returns raw 1-minute bars from DuckDB.
    For other timeframes (5min, 15min, 30min, 1h, 4h, 1d), aggregates on the fly.

    If no data is stored, triggers an automatic incremental update first.
    """
    try:
        start_dt = _parse_dt(start) if start else datetime(2020, 1, 1, tzinfo=UTC)
        end_dt = _parse_dt(end) if end else datetime.now(UTC)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    bars = query_bars(
        symbol=ticker,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        extended_hours=extended_hours,
        limit=limit,
    )

    if not bars:
        # Attempt incremental update, then re-query
        try:
            incremental_update(ticker)
            bars = query_bars(
                symbol=ticker,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
                extended_hours=extended_hours,
                limit=limit,
            )
        except Exception as exc:
            logger.warning("Auto-update failed", ticker=ticker, error=str(exc))

    if not bars:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No bars found for {ticker} in requested range. "
                "Run POST /ohlcv/v3/backfill/{ticker} to populate historical data."
            ),
        )
    return bars


@router.get(
    "/aggregate/{ticker}",
    response_model=List[AggregatedBar],
    summary="Multi-timeframe aggregation of stored 1-min bars",
)
def get_aggregate(
    ticker: str,
    timeframe: str = Query(default="1h", description="5min | 15min | 30min | 1h | 4h | 1d"),
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
    extended_hours: bool = Query(default=True),
) -> List[AggregatedBar]:
    """
    Aggregate 1-min base bars to the requested multi-timeframe using DuckDB time_bucket.

    Aggregation is computed on-the-fly from the stored 1-min data — no separate
    aggregated table needed, keeping storage lean.
    """
    try:
        start_dt = _parse_dt(start) if start else None
        end_dt = _parse_dt(end) if end else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        bars = aggregate_bars(
            symbol=ticker,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
            extended_hours=extended_hours,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not bars:
        raise HTTPException(
            status_code=404,
            detail=f"No aggregated bars found for {ticker}. Run backfill first.",
        )
    return bars


@router.post(
    "/backfill/{ticker}",
    response_model=BackfillStatus,
    summary="Trigger full historical backfill for a ticker",
)
def trigger_backfill(
    ticker: str,
    background_tasks: BackgroundTasks,
    start_date: Optional[str] = Query(
        default=None,
        description="Start date ISO format (default: 2015-01-02)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="End date ISO format (default: today)",
    ),
    async_mode: bool = Query(
        default=False,
        description="If true, run backfill in background and return immediately",
    ),
) -> BackfillStatus:
    """
    Fetch all available 1-min history for *ticker* and store in DuckDB.

    With Alpaca credentials: fetches 10+ years in 30-day chunks.
    Without credentials: falls back to yfinance (7 days, with clear warning).

    Set async_mode=true to run in background without blocking the request.
    """
    try:
        sd = date.fromisoformat(start_date) if start_date else None
        ed = date.fromisoformat(end_date) if end_date else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date: {exc}")

    if async_mode:
        # Return immediately, run in background
        background_tasks.add_task(backfill_symbol, ticker.upper(), sd, ed)
        return BackfillStatus(
            symbol=ticker.upper(),
            status="queued",
            bars_inserted=0,
            chunks_fetched=0,
            start_date=str(sd or date(2015, 1, 2)),
            end_date=str(ed or date.today()),
            source="alpaca" if _alpaca_available() else "yfinance",
            warnings=["Backfill is running in background — check /available-range for progress"],
        )

    return backfill_symbol(ticker.upper(), sd, ed)


@router.get(
    "/quality/{ticker}",
    response_model=QualityReport,
    summary="Data quality report: gaps, outliers, stale detection",
)
def get_quality(ticker: str) -> QualityReport:
    """
    Run quality checks against stored 1-min bars for *ticker*:

    - **Gap detection**: counts missing bars during regular session (09:30–16:00 ET)
      across the last 30 trading days
    - **Volume spikes**: bars where volume > 5x rolling 20-bar median
    - **Price spikes**: single bars with |close−open|/open > 10%
    - **Stale flag**: latest bar more than 3 trading days old, or no data at all
    """
    return run_quality_check(ticker)


@router.get(
    "/available-range/{ticker}",
    response_model=AvailableRange,
    summary="Show earliest/latest stored timestamps and coverage stats",
)
def get_available_range_endpoint(ticker: str) -> AvailableRange:
    """
    Returns the time range of stored 1-min bars for *ticker*, plus:
    - Total bar count
    - Number of trading days covered
    - Data source (alpaca | yfinance | unknown)
    - Whether Alpaca API credentials are configured in the environment
    """
    return get_available_range(ticker)


# ---------------------------------------------------------------------------
# Utility parsers
# ---------------------------------------------------------------------------

def _parse_dt(s: str) -> datetime:
    """Parse an ISO date or datetime string, returning UTC datetime."""
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue
    # Try fromisoformat last (Python 3.11+ handles 'Z')
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        raise ValueError(f"Cannot parse datetime '{s}'. Expected ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ")


# ---------------------------------------------------------------------------
# Module-level DB initialisation (runs on import)
# ---------------------------------------------------------------------------

def _init_db_on_import() -> None:
    """Ensure DB and schema exist when module is imported."""
    try:
        conn = _ensure_db()
        conn.close()
        logger.info("ohlcv_intraday DuckDB ready", path=str(_DB_PATH))
    except Exception as exc:
        logger.warning(
            "Could not initialise DuckDB on import",
            error=str(exc),
            path=str(_DB_PATH),
        )


_init_db_on_import()


# ---------------------------------------------------------------------------
# CLI entry point: python -m sentinel.sds.adapters.historical_ohlcv_v3 AAPL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    _parser = argparse.ArgumentParser(
        description="SENTINEL historical_ohlcv_v3 — CLI interface",
    )
    _parser.add_argument("symbol", help="Ticker symbol (e.g. AAPL)")
    _parser.add_argument(
        "--action",
        choices=["backfill", "quality", "range", "bars", "aggregate"],
        default="backfill",
        help="Action to perform",
    )
    _parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    _parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    _parser.add_argument("--timeframe", default="1Min", help="Timeframe for bars/aggregate")
    _parser.add_argument("--limit", type=int, default=100, help="Max bars for --action bars")
    _args = _parser.parse_args()

    sym = _args.symbol.upper()

    if _args.action == "backfill":
        sd = date.fromisoformat(_args.start) if _args.start else None
        ed = date.fromisoformat(_args.end) if _args.end else None
        result = backfill_symbol(sym, sd, ed)
        print(json.dumps(result.model_dump(), default=str, indent=2))

    elif _args.action == "quality":
        report = run_quality_check(sym)
        print(json.dumps(report.model_dump(), default=str, indent=2))

    elif _args.action == "range":
        rng = get_available_range(sym)
        print(json.dumps(rng.model_dump(), default=str, indent=2))

    elif _args.action == "bars":
        start_dt = _parse_dt(_args.start) if _args.start else datetime(2020, 1, 1, tzinfo=UTC)
        end_dt = _parse_dt(_args.end) if _args.end else datetime.now(UTC)
        bars = query_bars(sym, _args.timeframe, start_dt, end_dt, limit=_args.limit)
        print(json.dumps([b.model_dump() for b in bars], default=str, indent=2))

    elif _args.action == "aggregate":
        start_dt = _parse_dt(_args.start) if _args.start else None
        end_dt = _parse_dt(_args.end) if _args.end else None
        agg = aggregate_bars(sym, _args.timeframe, start_dt, end_dt)
        print(json.dumps([b.model_dump() for b in agg], default=str, indent=2))
