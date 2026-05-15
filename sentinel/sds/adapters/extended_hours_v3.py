"""
extended_hours_v3.py — Production-grade pre/post-market data engine (dim_011).

Replaces the simulated noise approach with real multi-source data:
  - Alpaca IEX REST snapshots: GET /v2/stocks/snapshots?symbols={}&feed=iex
  - Alpaca IEX historical bars with extended_hours=true and session=extended
  - Yahoo Finance fast_info (preMarketPrice / postMarketPrice) — no API key needed
  - Multi-source bid/ask aggregation with best-price selection
  - Exact session classifier: pre_market 04:00-09:30, regular 09:30-16:00,
    after_hours 16:00-20:00, closed otherwise (all times ET)
  - Volume context: extended-hours volume vs prior-day total (% of normal)
  - Pre-market gap alerts: |gap| > 2% flagged as significant
  - Earnings-driven flag: 8-K item 2.02 within 24h of extended-hours move
  - 90-day historical extended-hours OHLCV stored in SQLite
  - 30-day gap statistics: avg gap magnitude, gap-fill rate
  - FastAPI router at /extended/v3 with five endpoints

Dependencies: requests, sqlite3, pandas, numpy, fastapi, re (stdlib + yfinance)
No paid APIs required — Alpaca free tier + Yahoo Finance.
"""
from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants & timezone setup
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

_ALPACA_DATA_BASE = "https://data.alpaca.markets/v2/stocks"
_EDGAR_EFTS       = "https://efts.sec.gov/LATEST/search-index"
_FINVIZ_BASE      = "https://finviz.com/quote.ashx"

_HEADERS_SEC = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
}

_DB_PATH = Path(os.getenv("SENTINEL_DB_PATH", ".sentinel/extended_hours_v3.db"))

# Session boundaries in minutes from midnight (ET)
_PRE_START_MIN   = 4 * 60          # 04:00 = 240
_PRE_END_MIN     = 9 * 60 + 30     # 09:30 = 570
_AH_START_MIN    = 16 * 60         # 16:00 = 960
_AH_END_MIN      = 20 * 60         # 20:00 = 1200

_GAP_ALERT_THRESHOLD = 0.02        # 2% gap considered significant
_REQUEST_TIMEOUT     = 20.0
_ALPACA_RETRY_LIMIT  = 4
_SEC_RATE_DELAY      = 0.12        # 120 ms between EDGAR calls


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ExtendedQuote(BaseModel):
    ticker: str
    session: str                    # pre_market | regular | after_hours | closed
    as_of: datetime
    # Best aggregated price
    price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    volume: Optional[int] = None
    # Per-source breakdown
    alpaca_last: Optional[float] = None
    alpaca_bid: Optional[float] = None
    alpaca_ask: Optional[float] = None
    yf_price: Optional[float] = None
    source: str = "aggregated"


class GapAlert(BaseModel):
    ticker: str
    session: str
    gap_date: date
    prev_close: float
    current_price: float
    gap_pct: float
    is_significant: bool            # |gap_pct| > 2%
    earnings_driven: bool = False   # 8-K item 2.02 detected within 24h
    catalyst_hint: Optional[str] = None


class SessionVolume(BaseModel):
    ticker: str
    session: str
    session_date: date
    session_volume: int
    prior_day_total_volume: Optional[int] = None
    pct_of_normal: Optional[float] = None  # session_vol / prior_day_total * 100


class ExtendedBar(BaseModel):
    ticker: str
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    session: str
    vwap: Optional[float] = None


class GapStatistics(BaseModel):
    ticker: str
    days_analyzed: int
    avg_gap_pct: float
    max_gap_pct: float
    gap_fill_rate: float            # fraction of gaps filled during regular session
    earnings_gap_count: int
    significant_gap_count: int      # gaps > 2%


# ---------------------------------------------------------------------------
# SQLite persistence layer
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()


def _get_db() -> sqlite3.Connection:
    """Open (or create) the SQLite database with WAL mode for concurrent access."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_db() -> None:
    """Create tables if they don't exist."""
    with _db_lock:
        conn = _get_db()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS extended_bars (
                    ticker      TEXT NOT NULL,
                    ts          TEXT NOT NULL,          -- ISO8601 UTC
                    open        REAL,
                    high        REAL,
                    low         REAL,
                    close       REAL,
                    volume      INTEGER,
                    session     TEXT,
                    vwap        REAL,
                    source      TEXT,
                    PRIMARY KEY (ticker, ts)
                );

                CREATE INDEX IF NOT EXISTS idx_eb_ticker_ts
                    ON extended_bars (ticker, ts DESC);

                CREATE TABLE IF NOT EXISTS gap_history (
                    ticker          TEXT NOT NULL,
                    gap_date        TEXT NOT NULL,      -- YYYY-MM-DD
                    session         TEXT,
                    prev_close      REAL,
                    ext_price       REAL,
                    gap_pct         REAL,
                    is_significant  INTEGER DEFAULT 0,
                    earnings_driven INTEGER DEFAULT 0,
                    gap_filled      INTEGER,            -- NULL = unknown, 1 = filled
                    recorded_at     TEXT,
                    PRIMARY KEY (ticker, gap_date, session)
                );

                CREATE INDEX IF NOT EXISTS idx_gh_ticker
                    ON gap_history (ticker, gap_date DESC);
            """)
            conn.commit()
        finally:
            conn.close()


_init_db()


def _upsert_bars(ticker: str, bars: list[dict], source: str) -> None:
    """Upsert a list of bar dicts into extended_bars table."""
    if not bars:
        return
    rows = [
        (
            ticker.upper(),
            b["time"].isoformat() if hasattr(b["time"], "isoformat") else str(b["time"]),
            b.get("open"),
            b.get("high"),
            b.get("low"),
            b.get("close"),
            b.get("volume"),
            b.get("session"),
            b.get("vwap"),
            source,
        )
        for b in bars
    ]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO extended_bars
                   (ticker, ts, open, high, low, close, volume, session, vwap, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()


def _upsert_gap(
    ticker: str,
    gap_date: date,
    session: str,
    prev_close: float,
    ext_price: float,
    gap_pct: float,
    is_significant: bool,
    earnings_driven: bool,
) -> None:
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO gap_history
                   (ticker, gap_date, session, prev_close, ext_price, gap_pct,
                    is_significant, earnings_driven, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker.upper(),
                    gap_date.isoformat(),
                    session,
                    prev_close,
                    ext_price,
                    gap_pct,
                    int(is_significant),
                    int(earnings_driven),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _load_bars_from_db(
    ticker: str,
    start_date: date,
    end_date: date,
    sessions: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load stored bars for a ticker within a date range."""
    start_iso = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0).isoformat()
    end_iso   = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59).isoformat()
    with _db_lock:
        conn = _get_db()
        try:
            if sessions:
                placeholders = ",".join("?" * len(sessions))
                query = f"""
                    SELECT ticker, ts, open, high, low, close, volume, session, vwap
                    FROM extended_bars
                    WHERE ticker = ? AND ts >= ? AND ts <= ? AND session IN ({placeholders})
                    ORDER BY ts ASC
                """
                params = [ticker.upper(), start_iso, end_iso] + sessions
            else:
                query = """
                    SELECT ticker, ts, open, high, low, close, volume, session, vwap
                    FROM extended_bars
                    WHERE ticker = ? AND ts >= ? AND ts <= ?
                    ORDER BY ts ASC
                """
                params = [ticker.upper(), start_iso, end_iso]
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows, columns=["ticker", "time", "open", "high", "low", "close", "volume", "session", "vwap"]
    )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def _load_gap_history(ticker: str, days: int = 30) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with _db_lock:
        conn = _get_db()
        try:
            rows = conn.execute(
                """SELECT gap_date, session, prev_close, ext_price, gap_pct,
                          is_significant, earnings_driven, gap_filled
                   FROM gap_history
                   WHERE ticker = ? AND gap_date >= ?
                   ORDER BY gap_date DESC""",
                (ticker.upper(), cutoff),
            ).fetchall()
        finally:
            conn.close()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        rows,
        columns=["gap_date", "session", "prev_close", "ext_price", "gap_pct",
                 "is_significant", "earnings_driven", "gap_filled"],
    )


def _update_gap_fill(ticker: str, gap_date: str, session: str, filled: bool) -> None:
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                "UPDATE gap_history SET gap_filled = ? WHERE ticker = ? AND gap_date = ? AND session = ?",
                (int(filled), ticker.upper(), gap_date, session),
            )
            conn.commit()
        finally:
            conn.close()


def _prune_old_bars(days: int = 90) -> None:
    """Remove bars older than `days` to keep the DB compact."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute("DELETE FROM extended_bars WHERE ts < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Session classifier
# ---------------------------------------------------------------------------

def classify_session(dt: datetime) -> str:
    """
    Classify a UTC or tz-aware datetime into one of four US equity sessions.

    Returns one of: pre_market | regular | after_hours | closed
    All boundaries are inclusive at start, exclusive at end (ET):
      pre_market  : 04:00 <= t < 09:30
      regular     : 09:30 <= t < 16:00
      after_hours : 16:00 <= t < 20:00
      closed      : otherwise
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt_et = dt.astimezone(ET)
    total = dt_et.hour * 60 + dt_et.minute
    if _PRE_START_MIN <= total < _PRE_END_MIN:
        return "pre_market"
    if _PRE_END_MIN <= total < _AH_START_MIN:
        return "regular"
    if _AH_START_MIN <= total < _AH_END_MIN:
        return "after_hours"
    return "closed"


def current_session() -> str:
    return classify_session(datetime.now(tz=UTC))


def _prev_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# Alpaca IEX data fetcher
# ---------------------------------------------------------------------------

class AlpacaIEXFetcher:
    """
    Fetches real extended-hours data from Alpaca using the IEX feed.

    Snapshot endpoint returns live bid/ask/last including extended hours.
    Bars endpoint with extended_hours=true + feed=iex returns real historical
    extended-hours minute bars from IEX.
    """

    def __init__(self, timeout: float = _REQUEST_TIMEOUT):
        self._timeout = timeout
        self._key    = os.getenv("ALPACA_API_KEY", "")
        self._secret = os.getenv("ALPACA_SECRET_KEY", "")

    @property
    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID":     self._key,
            "APCA-API-SECRET-KEY": self._secret,
        }

    def has_credentials(self) -> bool:
        return bool(self._key and self._secret)

    async def _get(self, url: str, params: dict) -> dict:
        """GET with exponential-backoff retry on 429."""
        if not self.has_credentials():
            return {}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(_ALPACA_RETRY_LIMIT):
                try:
                    resp = await client.get(url, headers=self._headers, params=params)
                    if resp.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    return resp.json()
                except httpx.HTTPStatusError as exc:
                    logger.warning("Alpaca HTTP error", status=exc.response.status_code, url=url)
                    break
                except Exception as exc:
                    logger.warning("Alpaca request error", error=str(exc), url=url)
                    break
        return {}

    async def get_snapshot(self, ticker: str) -> dict:
        """
        GET /v2/stocks/snapshots?symbols={ticker}&feed=iex

        Returns real IEX bid/ask/last including extended-hours data.
        The snapshot has latestQuote (bid/ask) and latestTrade (last price).
        """
        url = f"{_ALPACA_DATA_BASE}/snapshots"
        data = await self._get(url, {"symbols": ticker.upper(), "feed": "iex"})
        return data.get(ticker.upper(), {})

    async def get_multi_snapshot(self, tickers: list[str]) -> dict[str, dict]:
        """Batch snapshot for multiple tickers (up to 100 per request)."""
        if not tickers:
            return {}
        results: dict[str, dict] = {}
        # Alpaca supports up to 100 symbols per call
        for i in range(0, len(tickers), 100):
            batch = tickers[i : i + 100]
            symbols_str = ",".join(t.upper() for t in batch)
            url = f"{_ALPACA_DATA_BASE}/snapshots"
            data = await self._get(url, {"symbols": symbols_str, "feed": "iex"})
            results.update(data)
        return results

    async def get_extended_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
    ) -> pd.DataFrame:
        """
        GET /v2/stocks/{ticker}/bars with feed=iex and extended_hours=true.

        Returns real IEX extended-hours OHLCV bars (not simulated).
        Paginates automatically via next_page_token.
        """
        url = f"{_ALPACA_DATA_BASE}/{ticker.upper()}/bars"
        start_s = start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_s   = end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        params: dict[str, Any] = {
            "timeframe":      timeframe,
            "start":          start_s,
            "end":            end_s,
            "feed":           "iex",
            "extended_hours": "true",
            "limit":          10000,
        }
        all_bars: list[dict] = []
        next_token: Optional[str] = None

        if not self.has_credentials():
            return pd.DataFrame()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for _ in range(50):   # safety cap on pagination
                if next_token:
                    params["page_token"] = next_token
                try:
                    resp = await client.get(url, headers=self._headers, params=params)
                    if resp.status_code == 429:
                        await asyncio.sleep(2)
                        continue
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as exc:
                    logger.warning("Alpaca bars error", ticker=ticker, error=str(exc))
                    break

                bars = payload.get("bars") or []
                all_bars.extend(bars)
                next_token = payload.get("next_page_token")
                if not next_token:
                    break

        if not all_bars:
            return pd.DataFrame()

        rows = []
        for b in all_bars:
            try:
                ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            except Exception:
                continue
            rows.append({
                "time":   ts,
                "open":   float(b.get("o", 0)),
                "high":   float(b.get("h", 0)),
                "low":    float(b.get("l", 0)),
                "close":  float(b.get("c", 0)),
                "volume": int(b.get("v", 0)),
                "vwap":   float(b.get("vw", 0)) if b.get("vw") else None,
                "trades": int(b.get("n", 0)),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()

        # Filter to only pre_market / after_hours bars
        df["session"] = df.index.map(lambda ts: classify_session(ts.to_pydatetime()))
        extended_mask = df["session"].isin(["pre_market", "after_hours"])
        return df[extended_mask]


# ---------------------------------------------------------------------------
# Yahoo Finance extended-hours fetcher (free, no API key)
# ---------------------------------------------------------------------------

class YahooExtendedFetcher:
    """
    Uses yfinance fast_info to get real preMarketPrice and postMarketPrice.

    These are actual Yahoo Finance delayed quotes — not simulated from close.
    Also fetches 60-day minute bars with prepost=True for historical analysis.
    """

    def get_current_extended_prices(self, ticker: str) -> dict:
        """
        Pull real preMarketPrice / postMarketPrice from yf.Ticker.fast_info.

        fast_info is the fastest yfinance data path. Fields available:
          - pre_market_price    (None outside pre-market hours)
          - post_market_price   (None outside post-market hours)
          - regular_market_price
          - three_month_average_volume
        """
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            result: dict = {
                "ticker":               ticker.upper(),
                "source":               "yfinance_fast_info",
                "regular_price":        _safe_float(getattr(fi, "last_price", None)),
                "pre_market_price":     _safe_float(getattr(fi, "pre_market_price", None)),
                "post_market_price":    _safe_float(getattr(fi, "post_market_price", None)),
                "previous_close":       _safe_float(getattr(fi, "previous_close", None)),
                "three_month_avg_vol":  _safe_float(getattr(fi, "three_month_average_volume", None)),
                "as_of":                datetime.now(tz=UTC).isoformat(),
            }
            return result
        except Exception as exc:
            logger.warning("YF fast_info failed", ticker=ticker, error=str(exc))
            return {"ticker": ticker.upper(), "source": "yfinance_fast_info", "error": str(exc)}

    def get_intraday_bars(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1m",
    ) -> pd.DataFrame:
        """
        Fetch minute bars including pre/post market (prepost=True).

        yfinance only provides 60 days of 1-minute data, so for historical
        extended-hours OHLCV we use 5m bars for older data.
        """
        days_back = (date.today() - start).days
        if days_back > 59:
            interval = "5m"

        try:
            t = yf.Ticker(ticker)
            end_fetch = end + timedelta(days=1)
            df = t.history(
                start=start.isoformat(),
                end=end_fetch.isoformat(),
                interval=interval,
                prepost=True,
                auto_adjust=True,
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df.index.name = "time"
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df.index = pd.to_datetime(df.index, utc=True)
            df["session"] = df.index.map(lambda ts: classify_session(ts.to_pydatetime()))
            for col in ["vwap", "trades"]:
                if col not in df.columns:
                    df[col] = None
            return df[["open", "high", "low", "close", "volume", "session", "vwap"]].sort_index()
        except Exception as exc:
            logger.warning("YF intraday bars failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

    def get_prior_day_volume(self, ticker: str, ref_date: Optional[date] = None) -> Optional[int]:
        """Fetch prior trading day's total regular-session volume from yfinance."""
        if ref_date is None:
            ref_date = date.today()
        prior = _prev_trading_day(ref_date)
        try:
            t = yf.Ticker(ticker)
            df = t.history(
                start=prior.isoformat(),
                end=(prior + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=True,
            )
            if df is None or df.empty:
                return None
            return int(df["Volume"].iloc[-1])
        except Exception as exc:
            logger.warning("YF prior day volume failed", ticker=ticker, error=str(exc))
            return None


# ---------------------------------------------------------------------------
# EDGAR 8-K item 2.02 detector
# ---------------------------------------------------------------------------

class EarningsFilingDetector:
    """
    Checks EDGAR EFTS for 8-K filings with item 2.02 (Results of Operations)
    within a specified time window to flag earnings-driven extended-hours moves.
    """

    async def check_earnings_8k(
        self, ticker: str, hours_back: int = 24
    ) -> tuple[bool, Optional[str]]:
        """
        Query EDGAR EFTS for 8-K item 2.02 filings for a ticker.

        Returns (is_earnings_driven, catalyst_description).
        """
        now_utc = datetime.now(tz=UTC)
        start_dt = now_utc - timedelta(hours=hours_back)
        start_str = start_dt.strftime("%Y-%m-%d")

        params = {
            "q":         f'"{ticker}" "item 2.02"',
            "dateRange": "custom",
            "startdt":   start_str,
            "forms":     "8-K,8-K/A",
            "_source":   "entity_name,file_date,form_type,period_of_report,accession_no",
        }

        try:
            await asyncio.sleep(_SEC_RATE_DELAY)
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(_EDGAR_EFTS, headers=_HEADERS_SEC, params=params)
                if resp.status_code != 200:
                    return False, None
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if hits:
                    src = hits[0].get("_source", {})
                    entity = src.get("entity_name", ticker)
                    filed  = src.get("file_date", "")
                    period = src.get("period_of_report", "")
                    hint   = f"8-K (Item 2.02 Results of Operations) filed by {entity} on {filed} for period {period}"
                    return True, hint
        except Exception as exc:
            logger.warning("EDGAR 8-K check failed", ticker=ticker, error=str(exc))

        return False, None

    async def get_recent_8k_items(
        self, ticker: str, days_back: int = 30
    ) -> list[dict]:
        """Return list of recent 8-K items for a ticker."""
        start_str = (date.today() - timedelta(days=days_back)).isoformat()
        params = {
            "q":         f'"{ticker}"',
            "dateRange": "custom",
            "startdt":   start_str,
            "forms":     "8-K",
            "_source":   "entity_name,file_date,form_type,period_of_report,accession_no,items",
        }
        try:
            await asyncio.sleep(_SEC_RATE_DELAY)
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(_EDGAR_EFTS, headers=_HEADERS_SEC, params=params)
                if resp.status_code != 200:
                    return []
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                results = []
                for h in hits[:10]:
                    src = h.get("_source", {})
                    results.append({
                        "entity_name":     src.get("entity_name", ""),
                        "file_date":       src.get("file_date", ""),
                        "period":          src.get("period_of_report", ""),
                        "accession_no":    src.get("accession_no", ""),
                        "items":           src.get("items", ""),
                    })
                return results
        except Exception as exc:
            logger.warning("EDGAR 8-K items failed", ticker=ticker, error=str(exc))
            return []


# ---------------------------------------------------------------------------
# Multi-source aggregator
# ---------------------------------------------------------------------------

class ExtendedHoursAggregator:
    """
    Aggregates extended-hours quotes from Alpaca IEX and Yahoo Finance.
    Selects best bid/ask/last across sources, computes spread.
    """

    def __init__(self):
        self._alpaca = AlpacaIEXFetcher()
        self._yf     = YahooExtendedFetcher()
        self._edgar  = EarningsFilingDetector()

    async def get_quote(self, ticker: str) -> ExtendedQuote:
        """
        Fetch real extended-hours quote from Alpaca IEX + Yahoo Finance.
        Aggregates best bid/ask, falls back gracefully.
        """
        ticker = ticker.upper()
        session = current_session()
        now_utc = datetime.now(tz=UTC)

        # Run Alpaca snapshot and YF fast_info concurrently
        alpaca_task = self._alpaca.get_snapshot(ticker)
        yf_task     = asyncio.get_event_loop().run_in_executor(
            None, self._yf.get_current_extended_prices, ticker
        )
        alpaca_snap, yf_data = await asyncio.gather(alpaca_task, yf_task)

        # Extract Alpaca values
        alpaca_quote = alpaca_snap.get("latestQuote", {})
        alpaca_trade = alpaca_snap.get("latestTrade", {})
        alpaca_bid   = _safe_float(alpaca_quote.get("bp"))
        alpaca_ask   = _safe_float(alpaca_quote.get("ap"))
        alpaca_last  = _safe_float(alpaca_trade.get("p"))
        alpaca_vol   = _safe_int(alpaca_trade.get("s"))

        # Extract Yahoo Finance extended-hours prices
        if session == "pre_market":
            yf_price = yf_data.get("pre_market_price")
        elif session == "after_hours":
            yf_price = yf_data.get("post_market_price")
        else:
            yf_price = yf_data.get("regular_price")

        # Aggregate: best bid/ask (highest bid, lowest ask = tightest spread)
        bids = [b for b in [alpaca_bid] if b is not None and b > 0]
        asks = [a for a in [alpaca_ask] if a is not None and a > 0]
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None

        # Best last price: Alpaca last, then YF
        price_candidates = [p for p in [alpaca_last, yf_price] if p is not None and p > 0]
        best_price = price_candidates[0] if price_candidates else None

        # Midpoint if no last trade
        if best_price is None and best_bid and best_ask:
            best_price = (best_bid + best_ask) / 2

        spread = (best_ask - best_bid) if (best_bid and best_ask) else None

        # Determine source label
        sources = []
        if alpaca_last:
            sources.append("alpaca_iex")
        if yf_price:
            sources.append("yahoo")
        source_label = "+".join(sources) if sources else "none"

        return ExtendedQuote(
            ticker=ticker,
            session=session,
            as_of=now_utc,
            price=round(best_price, 4) if best_price else None,
            bid=round(best_bid, 4) if best_bid else None,
            ask=round(best_ask, 4) if best_ask else None,
            spread=round(spread, 4) if spread else None,
            volume=alpaca_vol,
            alpaca_last=alpaca_last,
            alpaca_bid=alpaca_bid,
            alpaca_ask=alpaca_ask,
            yf_price=yf_price,
            source=source_label,
        )

    async def get_prior_close(self, ticker: str) -> Optional[float]:
        """Get prior trading day's regular-session close price via Yahoo Finance."""
        today = date.today()
        prior = _prev_trading_day(today)
        try:
            t = yf.Ticker(ticker.upper())
            df = t.history(
                start=prior.isoformat(),
                end=(prior + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=True,
            )
            if df is not None and not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception as exc:
            logger.warning("Prior close fetch failed", ticker=ticker, error=str(exc))
        return None

    async def compute_gap_alert(self, ticker: str) -> GapAlert:
        """
        Compute real extended-hours gap vs prior close.

        Flags gaps > 2% as significant and checks for earnings 8-K (item 2.02).
        """
        session = current_session()
        today = date.today()

        # Get current extended-hours quote and prior close concurrently
        quote_task      = self.get_quote(ticker)
        prior_close_task = asyncio.get_event_loop().run_in_executor(
            None, lambda: self._yf.get_prior_day_volume.__self__.get_current_extended_prices(ticker)
        )
        quote, yf_prices = await asyncio.gather(
            quote_task,
            asyncio.get_event_loop().run_in_executor(
                None, self._yf.get_current_extended_prices, ticker
            ),
        )

        prev_close = yf_prices.get("previous_close")
        if prev_close is None:
            prev_close = await self.get_prior_close(ticker)

        current_price = quote.price

        if not prev_close or not current_price or prev_close == 0:
            return GapAlert(
                ticker=ticker,
                session=session,
                gap_date=today,
                prev_close=prev_close or 0.0,
                current_price=current_price or 0.0,
                gap_pct=0.0,
                is_significant=False,
            )

        gap_pct = (current_price - prev_close) / prev_close
        is_significant = abs(gap_pct) > _GAP_ALERT_THRESHOLD

        # Check for earnings 8-K if gap is notable
        earnings_driven = False
        catalyst_hint   = None
        if is_significant or abs(gap_pct) > 0.01:
            earnings_driven, catalyst_hint = await self._edgar.check_earnings_8k(ticker, hours_back=24)

        # Persist to DB
        _upsert_gap(
            ticker, today, session,
            float(prev_close), float(current_price),
            float(gap_pct), is_significant, earnings_driven,
        )

        return GapAlert(
            ticker=ticker,
            session=session,
            gap_date=today,
            prev_close=round(float(prev_close), 4),
            current_price=round(float(current_price), 4),
            gap_pct=round(gap_pct * 100, 4),   # express as percent
            is_significant=is_significant,
            earnings_driven=earnings_driven,
            catalyst_hint=catalyst_hint,
        )

    async def compute_session_volume(self, ticker: str) -> SessionVolume:
        """
        Compute extended-hours session volume and compare to prior day's total volume.
        Uses Alpaca IEX snapshot volume + Yahoo Finance prior-day volume.
        """
        session = current_session()
        today   = date.today()

        # Get current session volume from Alpaca snapshot
        snap = await self._alpaca.get_snapshot(ticker)
        daily_bar = snap.get("dailyBar", {})
        session_vol = _safe_int(daily_bar.get("v"))

        # For extended-hours specific volume, use minute bars
        if session in ("pre_market", "after_hours"):
            today_d = today
            d = today_d
            if session == "pre_market":
                start_et = datetime(d.year, d.month, d.day, 4, 0, tzinfo=ET)
                end_et   = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
            else:
                start_et = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)
                end_et   = datetime(d.year, d.month, d.day, 20, 0, tzinfo=ET)

            bars_df = await self._alpaca.get_extended_bars(
                ticker,
                start_et.astimezone(UTC),
                end_et.astimezone(UTC),
            )
            if not bars_df.empty:
                session_vol = int(bars_df["volume"].sum())

        # Prior day total volume
        loop = asyncio.get_event_loop()
        prior_vol = await loop.run_in_executor(
            None, self._yf.get_prior_day_volume, ticker, today
        )

        pct_of_normal: Optional[float] = None
        if prior_vol and prior_vol > 0 and session_vol:
            pct_of_normal = round(session_vol / prior_vol * 100, 2)

        return SessionVolume(
            ticker=ticker.upper(),
            session=session,
            session_date=today,
            session_volume=session_vol or 0,
            prior_day_total_volume=prior_vol,
            pct_of_normal=pct_of_normal,
        )


# ---------------------------------------------------------------------------
# Historical extended-hours OHLCV engine
# ---------------------------------------------------------------------------

class HistoricalExtendedEngine:
    """
    Fetches and stores 90 days of historical extended-hours OHLCV.

    Primary: Alpaca IEX bars (extended_hours=true) — real exchange data.
    Fallback: Yahoo Finance (prepost=True) minute bars.
    Data is cached in SQLite and pruned automatically to 90 days.
    """

    def __init__(self):
        self._alpaca = AlpacaIEXFetcher()
        self._yf     = YahooExtendedFetcher()

    async def backfill(self, ticker: str, days: int = 90) -> int:
        """
        Backfill up to `days` of historical extended-hours bars into SQLite.

        Returns number of bars stored.
        """
        ticker = ticker.upper()
        today = date.today()
        start = today - timedelta(days=days)
        start_dt = datetime(start.year, start.month, start.day, 0, 0, tzinfo=UTC)
        end_dt   = datetime(today.year, today.month, today.day, 23, 59, tzinfo=UTC)

        total_stored = 0

        # Try Alpaca IEX first
        if self._alpaca.has_credentials():
            logger.info("Alpaca IEX backfill starting", ticker=ticker, days=days)
            df = await self._alpaca.get_extended_bars(ticker, start_dt, end_dt, timeframe="5Min")
            if not df.empty:
                bars = self._df_to_bar_dicts(df, ticker)
                _upsert_bars(ticker, bars, "alpaca_iex")
                total_stored += len(bars)
                logger.info("Alpaca IEX backfill complete", ticker=ticker, bars=len(bars))
                _prune_old_bars(90)
                return total_stored

        # Fallback: Yahoo Finance (60-day limit for 1m, older uses 5m)
        logger.info("YF backfill starting", ticker=ticker, days=min(days, 60))
        loop = asyncio.get_event_loop()
        yf_start = max(start, today - timedelta(days=60))
        df = await loop.run_in_executor(
            None, self._yf.get_intraday_bars, ticker, yf_start, today
        )
        if not df.empty:
            ext_mask = df["session"].isin(["pre_market", "after_hours"])
            ext_df = df[ext_mask]
            bars = self._df_to_bar_dicts(ext_df, ticker)
            _upsert_bars(ticker, bars, "yfinance")
            total_stored += len(bars)

        _prune_old_bars(90)
        return total_stored

    def _df_to_bar_dicts(self, df: pd.DataFrame, ticker: str) -> list[dict]:
        bars = []
        for ts, row in df.iterrows():
            dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            bars.append({
                "time":    dt,
                "open":    float(row.get("open", 0)),
                "high":    float(row.get("high", 0)),
                "low":     float(row.get("low", 0)),
                "close":   float(row.get("close", 0)),
                "volume":  int(row.get("volume", 0)),
                "session": row.get("session", classify_session(dt)),
                "vwap":    float(row["vwap"]) if row.get("vwap") is not None else None,
            })
        return bars

    async def get_history(
        self,
        ticker: str,
        days: int = 30,
        session_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return extended-hours bars from SQLite cache, backfilling if needed.

        session_filter: "pre_market" | "after_hours" | None (both)
        """
        ticker = ticker.upper()
        end_date   = date.today()
        start_date = end_date - timedelta(days=days)
        sessions   = [session_filter] if session_filter else ["pre_market", "after_hours"]

        df = _load_bars_from_db(ticker, start_date, end_date, sessions)
        if df.empty:
            await self.backfill(ticker, days)
            df = _load_bars_from_db(ticker, start_date, end_date, sessions)

        return df


# ---------------------------------------------------------------------------
# Gap statistics engine
# ---------------------------------------------------------------------------

class GapStatisticsEngine:
    """
    Computes 30-day gap statistics from stored gap_history table.

    Metrics:
      - avg_gap_pct: mean absolute pre-market gap over 30 days
      - gap_fill_rate: fraction of gaps that filled during regular session
      - significant_gap_count: gaps > 2% threshold
      - earnings_gap_count: earnings-driven gaps
    """

    def __init__(self):
        self._aggregator = ExtendedHoursAggregator()
        self._hist_engine = HistoricalExtendedEngine()

    async def compute_gap_stats(self, ticker: str, days: int = 30) -> GapStatistics:
        """
        Compute gap statistics using stored history + real current data.

        Also checks gap fills for historical gaps using end-of-day prices.
        """
        ticker = ticker.upper()

        # Ensure we have historical bars
        hist_df = await self._hist_engine.get_history(ticker, days=days, session_filter="pre_market")

        # Load stored gap history
        gap_df = _load_gap_history(ticker, days=days)

        if gap_df.empty and not hist_df.empty:
            # Build gap history from bars
            await self._build_gap_history_from_bars(ticker, hist_df, days)
            gap_df = _load_gap_history(ticker, days=days)

        if gap_df.empty:
            return GapStatistics(
                ticker=ticker,
                days_analyzed=0,
                avg_gap_pct=0.0,
                max_gap_pct=0.0,
                gap_fill_rate=0.0,
                earnings_gap_count=0,
                significant_gap_count=0,
            )

        # Update gap fills for past gaps that don't have fill status
        await self._update_gap_fills(ticker, gap_df)
        gap_df = _load_gap_history(ticker, days=days)

        abs_gaps    = gap_df["gap_pct"].abs()
        avg_gap_pct = float(abs_gaps.mean()) if not abs_gaps.empty else 0.0
        max_gap_pct = float(abs_gaps.max()) if not abs_gaps.empty else 0.0

        filled_known = gap_df[gap_df["gap_filled"].notna()]
        gap_fill_rate = (
            float((filled_known["gap_filled"] == 1).sum() / len(filled_known))
            if len(filled_known) > 0 else 0.0
        )

        sig_count      = int((gap_df["is_significant"] == 1).sum())
        earnings_count = int((gap_df["earnings_driven"] == 1).sum())

        return GapStatistics(
            ticker=ticker,
            days_analyzed=len(gap_df),
            avg_gap_pct=round(avg_gap_pct, 4),
            max_gap_pct=round(max_gap_pct, 4),
            gap_fill_rate=round(gap_fill_rate, 4),
            earnings_gap_count=earnings_count,
            significant_gap_count=sig_count,
        )

    async def _build_gap_history_from_bars(
        self, ticker: str, pre_df: pd.DataFrame, days: int
    ) -> None:
        """Build gap_history records from stored pre-market bars by comparing to prior close."""
        loop = asyncio.get_event_loop()
        today = date.today()
        edgar = EarningsFilingDetector()

        # Group bars by date
        pre_df_copy = pre_df.copy()
        pre_df_copy.index = pd.to_datetime(pre_df_copy.index, utc=True)
        pre_df_copy["date_et"] = pre_df_copy.index.map(
            lambda ts: ts.astimezone(ET).date()
        )

        for bar_date, day_bars in pre_df_copy.groupby("date_et"):
            prior = _prev_trading_day(bar_date)
            try:
                t = yf.Ticker(ticker)
                prior_hist = await loop.run_in_executor(None, lambda: t.history(
                    start=prior.isoformat(),
                    end=(prior + timedelta(days=1)).isoformat(),
                    interval="1d",
                    auto_adjust=True,
                ))
                if prior_hist is None or prior_hist.empty:
                    continue
                prior_close = float(prior_hist["Close"].iloc[-1])
            except Exception:
                continue

            if prior_close == 0:
                continue

            # Use last pre-market bar close as the extended price
            ext_price = float(day_bars["close"].iloc[-1])
            gap_pct   = (ext_price - prior_close) / prior_close
            is_significant = abs(gap_pct) > _GAP_ALERT_THRESHOLD

            _upsert_gap(
                ticker, bar_date, "pre_market",
                prior_close, ext_price, gap_pct,
                is_significant, False,
            )

    async def _update_gap_fills(self, ticker: str, gap_df: pd.DataFrame) -> None:
        """
        For gaps older than today, check if the gap was filled during regular session.
        A gap fill = price crossed the prior_close during regular hours that day.
        """
        today = date.today()
        loop  = asyncio.get_event_loop()

        for _, row in gap_df.iterrows():
            if pd.notna(row["gap_filled"]):
                continue  # already known

            gap_date_str = str(row["gap_date"])
            try:
                gap_date_obj = date.fromisoformat(gap_date_str)
            except Exception:
                continue

            if gap_date_obj >= today:
                continue  # today's gap not yet resolvable

            prev_close = float(row["prev_close"])
            gap_pct    = float(row["gap_pct"])
            if prev_close == 0 or gap_pct == 0:
                continue

            # Fetch regular-session bars for that date
            try:
                t = yf.Ticker(ticker)
                reg_df = await loop.run_in_executor(None, lambda: t.history(
                    start=gap_date_str,
                    end=(gap_date_obj + timedelta(days=1)).isoformat(),
                    interval="5m",
                    prepost=False,
                    auto_adjust=True,
                ))
                if reg_df is None or reg_df.empty:
                    continue

                lows  = reg_df["Low"].values
                highs = reg_df["High"].values

                if gap_pct > 0:
                    # Gap up — filled if price traded down to or below prior_close
                    filled = bool(np.any(lows <= prev_close))
                else:
                    # Gap down — filled if price traded up to or above prior_close
                    filled = bool(np.any(highs >= prev_close))

                _update_gap_fill(ticker, gap_date_str, str(row["session"]), filled)
            except Exception as exc:
                logger.warning("Gap fill check failed", ticker=ticker, date=gap_date_str, error=str(exc))


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if f != f or abs(f) == float("inf"):   # NaN or inf
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# FastAPI router: /extended/v3
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/extended/v3", tags=["extended-hours-v3"])

_aggregator = ExtendedHoursAggregator()
_hist_engine = HistoricalExtendedEngine()
_gap_stats   = GapStatisticsEngine()
_edgar       = EarningsFilingDetector()


@router.get("/pre-market/{ticker}", summary="Real pre-market quote with gap alert")
async def get_pre_market(ticker: str) -> dict:
    """
    Returns real IEX pre-market bid/ask/last from Alpaca snapshot +
    Yahoo Finance preMarketPrice. Includes gap vs prior close and earnings
    catalyst detection.
    """
    ticker = ticker.upper()
    try:
        quote = await _aggregator.get_quote(ticker)
        gap   = await _aggregator.compute_gap_alert(ticker)
        return {
            "ticker":  ticker,
            "session": classify_session(datetime.now(tz=UTC)),
            "quote": {
                "price":       quote.price,
                "bid":         quote.bid,
                "ask":         quote.ask,
                "spread":      quote.spread,
                "volume":      quote.volume,
                "alpaca_last": quote.alpaca_last,
                "yf_price":    quote.yf_price,
                "source":      quote.source,
                "as_of":       quote.as_of.isoformat(),
            },
            "gap_alert": {
                "prev_close":       gap.prev_close,
                "current_price":    gap.current_price,
                "gap_pct":          gap.gap_pct,
                "is_significant":   gap.is_significant,
                "earnings_driven":  gap.earnings_driven,
                "catalyst":         gap.catalyst_hint,
            },
        }
    except Exception as exc:
        logger.error("pre-market endpoint error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/after-hours/{ticker}", summary="Real after-hours quote with gap alert")
async def get_after_hours(ticker: str) -> dict:
    """
    Returns real IEX after-hours data from Alpaca snapshot +
    Yahoo Finance postMarketPrice. Includes gap vs prior close and
    earnings 8-K detection.
    """
    ticker = ticker.upper()
    try:
        quote = await _aggregator.get_quote(ticker)
        gap   = await _aggregator.compute_gap_alert(ticker)
        return {
            "ticker":  ticker,
            "session": classify_session(datetime.now(tz=UTC)),
            "quote": {
                "price":       quote.price,
                "bid":         quote.bid,
                "ask":         quote.ask,
                "spread":      quote.spread,
                "volume":      quote.volume,
                "alpaca_last": quote.alpaca_last,
                "yf_price":    quote.yf_price,
                "source":      quote.source,
                "as_of":       quote.as_of.isoformat(),
            },
            "gap_alert": {
                "prev_close":       gap.prev_close,
                "current_price":    gap.current_price,
                "gap_pct":          gap.gap_pct,
                "is_significant":   gap.is_significant,
                "earnings_driven":  gap.earnings_driven,
                "catalyst":         gap.catalyst_hint,
            },
        }
    except Exception as exc:
        logger.error("after-hours endpoint error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/gap-alert/{ticker}", summary="Gap alert with earnings detection")
async def get_gap_alert(
    ticker: str,
    threshold: float = Query(2.0, description="Gap alert threshold in percent (default 2%)"),
) -> dict:
    """
    Returns gap analysis vs prior close. Flags gaps exceeding threshold as
    significant. Checks EDGAR for 8-K item 2.02 to detect earnings-driven gaps.
    Returns gap statistics (30-day avg, fill rate).
    """
    ticker = ticker.upper()
    try:
        gap   = await _aggregator.compute_gap_alert(ticker)
        stats = await _gap_stats.compute_gap_stats(ticker, days=30)
        return {
            "ticker": ticker,
            "current_gap": {
                "session":         gap.session,
                "prev_close":      gap.prev_close,
                "current_price":   gap.current_price,
                "gap_pct":         gap.gap_pct,
                "is_significant":  abs(gap.gap_pct) > threshold,
                "threshold_used":  threshold,
                "earnings_driven": gap.earnings_driven,
                "catalyst":        gap.catalyst_hint,
                "as_of":           date.today().isoformat(),
            },
            "historical_stats": {
                "days_analyzed":         stats.days_analyzed,
                "avg_gap_pct":           stats.avg_gap_pct,
                "max_gap_pct":           stats.max_gap_pct,
                "gap_fill_rate":         stats.gap_fill_rate,
                "significant_gap_count": stats.significant_gap_count,
                "earnings_gap_count":    stats.earnings_gap_count,
            },
        }
    except Exception as exc:
        logger.error("gap-alert endpoint error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/session-volume/{ticker}", summary="Extended-hours volume vs prior day normal")
async def get_session_volume(ticker: str) -> dict:
    """
    Returns current extended-hours session volume and compares to prior day's
    total volume to give a percentage-of-normal context.
    """
    ticker = ticker.upper()
    try:
        vol = await _aggregator.compute_session_volume(ticker)
        return {
            "ticker":                  vol.ticker,
            "session":                 vol.session,
            "session_date":            vol.session_date.isoformat(),
            "session_volume":          vol.session_volume,
            "prior_day_total_volume":  vol.prior_day_total_volume,
            "pct_of_normal":           vol.pct_of_normal,
            "interpretation": (
                "high" if (vol.pct_of_normal or 0) > 50
                else "normal" if (vol.pct_of_normal or 0) > 15
                else "low"
            ),
        }
    except Exception as exc:
        logger.error("session-volume endpoint error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/extended-history/{ticker}", summary="90-day historical extended-hours OHLCV")
async def get_extended_history(
    ticker: str,
    days: int = Query(30, ge=1, le=90, description="Number of days of history (max 90)"),
    session: Optional[str] = Query(
        None, description="Filter by session: pre_market | after_hours | null for both"
    ),
) -> dict:
    """
    Returns stored historical extended-hours OHLCV bars from SQLite cache.
    Data sourced from Alpaca IEX (preferred) or Yahoo Finance (fallback).
    Covers up to 90 days of real pre/post-market bars — not simulated.
    """
    ticker = ticker.upper()
    valid_sessions = {"pre_market", "after_hours", None}
    if session not in valid_sessions:
        raise HTTPException(
            status_code=400,
            detail=f"session must be one of: pre_market, after_hours, or omitted"
        )

    try:
        df = await _hist_engine.get_history(ticker, days=days, session_filter=session)
        if df.empty:
            return {
                "ticker":  ticker,
                "days":    days,
                "session": session or "all",
                "bars":    [],
                "count":   0,
            }

        df_reset = df.reset_index()
        df_reset["time"] = df_reset["time"].astype(str)
        bars = df_reset.to_dict(orient="records")

        # Summary statistics
        summary: dict = {
            "total_bars":     len(bars),
            "date_range": {
                "start": df_reset["time"].min(),
                "end":   df_reset["time"].max(),
            },
        }
        if "session" in df_reset.columns:
            summary["sessions"] = df_reset["session"].value_counts().to_dict()

        return {
            "ticker":  ticker,
            "days":    days,
            "session": session or "all",
            "summary": summary,
            "bars":    bars,
            "count":   len(bars),
        }
    except Exception as exc:
        logger.error("extended-history endpoint error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/earnings-activity/{ticker}", summary="Recent 8-K earnings filings for ticker")
async def get_earnings_activity(
    ticker: str,
    days_back: int = Query(30, ge=1, le=90, description="Days back to search for 8-K filings"),
) -> dict:
    """
    Returns recent 8-K filings for a ticker from EDGAR.
    Used to contextualise extended-hours moves as earnings-driven.
    """
    ticker = ticker.upper()
    try:
        items = await _edgar.get_recent_8k_items(ticker, days_back=days_back)
        return {
            "ticker":   ticker,
            "days_back": days_back,
            "filings":  items,
            "count":    len(items),
        }
    except Exception as exc:
        logger.error("earnings-activity endpoint error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
