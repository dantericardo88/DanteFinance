"""Intraday deep adapter — historical 1-min OHLCV, multi-source, volume profile analytics.

Targets dim_003 "Historical OHLCV intraday (1-min, 20+ years)".

True 20-year 1-min data requires paid sources (Polygon $29/mo, FirstRate Data $100/yr).
This module maximises what is freely achievable while being fully transparent about limits:

  Source          | Min interval | Max free history
  --------------- | ------------ | ----------------
  Alpaca IEX      | 1 min        | ~5 years (free, no auth needed for delayed)
  yfinance        | 1 min        | 7 days only
  yfinance        | 2-15 min     | 60 days
  yfinance        | 1 h          | 730 days
  Stooq           | 5 min        | ~365 days
  Polygon (free)  | 1 min        | 730 days (requires free API key)
  FirstRate Data  | 1 min        | 25 years (paid, ~$100/yr)

Cache layer: DuckDB at .sentinel/cache/intraday.duckdb
"""
from __future__ import annotations

import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# IntradayDepthMap — canonical reference for data availability per source
# ---------------------------------------------------------------------------

DEPTH_BY_SOURCE: dict[str, dict[str, Any]] = {
    "alpaca_iex": {
        "min_interval": "1min",
        "max_history_days": 1825,
        "max_history_years": 5,
        "note": "Free IEX feed; requires Alpaca account (free tier OK). "
                "SIP feed (real-time) requires funded account.",
        "auth_required": True,
        "cost": "free",
    },
    "yfinance": {
        "min_interval": "1min",
        "max_history_days": 7,
        "max_history_years": 0,
        "note": "Yahoo Finance hard limit: 7 days for 1-min bars.",
        "auth_required": False,
        "cost": "free",
    },
    "yfinance_2m": {
        "min_interval": "2min",
        "max_history_days": 60,
        "note": "Yahoo Finance 2-min bars, rolling 60-day window.",
        "auth_required": False,
        "cost": "free",
    },
    "yfinance_5m": {
        "min_interval": "5min",
        "max_history_days": 60,
        "note": "Yahoo Finance 5-min bars, rolling 60-day window.",
        "auth_required": False,
        "cost": "free",
    },
    "yfinance_15m": {
        "min_interval": "15min",
        "max_history_days": 60,
        "note": "Yahoo Finance 15-min bars, rolling 60-day window.",
        "auth_required": False,
        "cost": "free",
    },
    "yfinance_1h": {
        "min_interval": "1h",
        "max_history_days": 730,
        "note": "Yahoo Finance 1-hour bars, ~2 years rolling.",
        "auth_required": False,
        "cost": "free",
    },
    "stooq_5m": {
        "min_interval": "5min",
        "max_history_days": 365,
        "note": "Stooq provides 5-min for US tickers (append .us). "
                "~1 year coverage. No API key needed.",
        "auth_required": False,
        "cost": "free",
    },
    "polygon_free": {
        "min_interval": "1min",
        "max_history_days": 730,
        "note": "Polygon free tier: 2-year history, 5 API calls/min rate limit. "
                "Requires free API key at polygon.io.",
        "auth_required": True,
        "cost": "free",
    },
    "firstrate_data": {
        "min_interval": "1min",
        "max_history_years": 25,
        "note": "FirstRate Data: paid service, ~$100/yr, 25 years of 1-min data. "
                "Best option for true 20-year intraday back-tests.",
        "auth_required": True,
        "cost": "paid ~$100/yr",
    },
}

# DuckDB cache path (respects XDG base or falls back to project root)
_CACHE_DIR = Path(os.environ.get("SENTINEL_CACHE_DIR", Path.cwd() / ".sentinel" / "cache"))
_INTRADAY_DB = _CACHE_DIR / "intraday.duckdb"

# ---------------------------------------------------------------------------
# Alpaca intraday adapter
# ---------------------------------------------------------------------------

class AlpacaIntradayAdapter:
    """Historical intraday bars from Alpaca free IEX feed (~5 years, 1-min).

    Free tier: no credentials needed for delayed IEX data.
    For real-time SIP feed: pass api_key + secret_key from funded account.
    """

    ALPACA_TF_MAP = {
        "1Min": "Minute",
        "1min": "Minute",
        "5Min": "Minute",
        "5min": "Minute",
        "15Min": "Minute",
        "15min": "Minute",
        "1Hour": "Hour",
        "1hour": "Hour",
        "1h": "Hour",
        "1Day": "Day",
        "1day": "Day",
        "1d": "Day",
    }

    def __init__(self, api_key: str = "", secret_key: str = "") -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._available = False
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed
            self._StockHistoricalDataClient = StockHistoricalDataClient
            self._StockBarsRequest = StockBarsRequest
            self._TimeFrame = TimeFrame
            self._TimeFrameUnit = TimeFrameUnit
            self._DataFeed = DataFeed
            self._client = StockHistoricalDataClient(
                self._api_key or None, self._secret_key or None
            )
            self._available = True
        except ImportError:
            logger.warning("alpaca-py not installed. Run: pip install alpaca-py")

    # ------------------------------------------------------------------
    def get_bars(
        self,
        ticker: str,
        start: str,
        end: str,
        timeframe: str = "1Min",
    ) -> pd.DataFrame:
        """Fetch historical bars for a single ticker.

        Parameters
        ----------
        ticker:    uppercase symbol, e.g. "AAPL"
        start:     ISO date string "YYYY-MM-DD" or datetime
        end:       ISO date string or datetime
        timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"

        Returns
        -------
        DataFrame with columns: time, open, high, low, close, volume, vwap, trade_count
        """
        if not self._available:
            return pd.DataFrame()
        tf_unit = self._TimeFrameUnit.Minute
        tf_amount = 1
        tfl = timeframe.lower()
        if "hour" in tfl or tfl == "1h":
            tf_unit = self._TimeFrameUnit.Hour
            tf_amount = int("".join(c for c in timeframe if c.isdigit()) or "1")
        elif "day" in tfl or tfl == "1d":
            tf_unit = self._TimeFrameUnit.Day
            tf_amount = 1
        else:
            tf_amount = int("".join(c for c in timeframe if c.isdigit()) or "1")

        request = self._StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=self._TimeFrame(tf_amount, tf_unit),
            start=start,
            end=end,
            feed=self._DataFeed.IEX,
        )
        try:
            bars = self._client.get_stock_bars(request)
            df = bars.df
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            # Alpaca multi-index: (symbol, timestamp) -> flatten
            if "symbol" in df.columns:
                df = df.drop(columns=["symbol"], errors="ignore")
            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "time"})
            col_map = {
                "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume",
                "vwap": "vwap", "trade_count": "trade_count",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            df["ticker"] = ticker
            return df.sort_values("time").reset_index(drop=True)
        except Exception as exc:
            logger.warning("AlpacaIntradayAdapter.get_bars failed for %s: %s", ticker, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def get_bulk_bars(
        self,
        tickers: list[str],
        start: str,
        end: str,
        timeframe: str = "1Min",
    ) -> dict[str, pd.DataFrame]:
        """Multi-ticker request — Alpaca supports up to 100 symbols per call."""
        if not self._available or not tickers:
            return {}
        # Alpaca caps at 100 symbols per request
        results: dict[str, pd.DataFrame] = {}
        for chunk_start in range(0, len(tickers), 100):
            chunk = tickers[chunk_start: chunk_start + 100]
            request = self._StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=self._TimeFrame(1, self._TimeFrameUnit.Minute),
                start=start,
                end=end,
                feed=self._DataFeed.IEX,
            )
            try:
                bars = self._client.get_stock_bars(request)
                df_all = bars.df.reset_index()
                if df_all.empty:
                    continue
                for sym in chunk:
                    if "symbol" in df_all.columns:
                        sub = df_all[df_all["symbol"] == sym].copy()
                        sub = sub.drop(columns=["symbol"], errors="ignore")
                    else:
                        sub = df_all.copy()
                    if "timestamp" in sub.columns:
                        sub = sub.rename(columns={"timestamp": "time"})
                    sub["ticker"] = sym
                    results[sym] = sub.sort_values("time").reset_index(drop=True)
            except Exception as exc:
                logger.warning("AlpacaIntradayAdapter.get_bulk_bars chunk failed: %s", exc)
        return results

    # ------------------------------------------------------------------
    def get_bars_chunked(
        self,
        ticker: str,
        start: str,
        end: str,
        chunk_days: int = 90,
    ) -> pd.DataFrame:
        """Break long date ranges into chunks to avoid Alpaca API timeouts.

        Recommended chunk_days=90 for 1-min data to stay within rate limits.
        """
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        chunks: list[pd.DataFrame] = []
        cursor = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
            df = self.get_bars(ticker, cursor.isoformat(), chunk_end.isoformat())
            if not df.empty:
                chunks.append(df)
            cursor = chunk_end + timedelta(seconds=1)
            time.sleep(0.3)  # polite pause between chunks
        if not chunks:
            return pd.DataFrame()
        combined = pd.concat(chunks, ignore_index=True)
        return combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Polygon intraday adapter (free tier)
# ---------------------------------------------------------------------------

class PolygonIntradayAdapter:
    """Polygon.io free tier — 2-year intraday history, 5 calls/min.

    Requires free API key from polygon.io. Set POLYGON_API_KEY env var.
    """

    BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

    def __init__(self, api_key: str = None) -> None:
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self._api_key:
            logger.warning(
                "PolygonIntradayAdapter: no POLYGON_API_KEY found. "
                "Free key available at https://polygon.io/dashboard/signup"
            )
        self._client = httpx.Client(timeout=30)
        self._last_call = 0.0
        self._min_interval_s = 12.1  # 5 calls/min free tier

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)
        self._last_call = time.monotonic()

    def _get(self, url: str, params: dict) -> dict:
        self._throttle()
        params["apiKey"] = self._api_key
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def get_bars(
        self,
        ticker: str,
        start: str,
        end: str,
        interval_minutes: int = 1,
    ) -> pd.DataFrame:
        """Fetch intraday bars. Free tier: 2-year history, 5 calls/min.

        Returns DataFrame with columns: time, open, high, low, close, volume, vwap, trade_count
        """
        if not self._api_key:
            logger.error("Polygon API key required. Set POLYGON_API_KEY.")
            return pd.DataFrame()
        url = self.BASE_URL.format(
            ticker=ticker.upper(),
            multiplier=interval_minutes,
            timespan="minute",
            from_date=start,
            to_date=end,
        )
        try:
            data = self._get(url, {"adjusted": "true", "sort": "asc", "limit": 50000})
            results = data.get("results", [])
            if not results:
                return pd.DataFrame()
            df = pd.DataFrame(results)
            df = df.rename(columns={
                "t": "time", "o": "open", "h": "high", "l": "low",
                "c": "close", "v": "volume", "vw": "vwap", "n": "trade_count",
            })
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df["ticker"] = ticker.upper()
            # Handle pagination
            next_url = data.get("next_url")
            if next_url:
                extra = self.get_bars_paginated(ticker, start, end, interval_minutes, _initial_data=data)
                if not extra.empty:
                    df = pd.concat([df, extra], ignore_index=True)
            return df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        except Exception as exc:
            logger.warning("PolygonIntradayAdapter.get_bars failed for %s: %s", ticker, exc)
            return pd.DataFrame()

    def get_bars_paginated(
        self,
        ticker: str,
        start: str,
        end: str,
        interval_minutes: int = 1,
        _initial_data: dict = None,
    ) -> pd.DataFrame:
        """Handle Polygon pagination via next_url field in response."""
        if not self._api_key:
            return pd.DataFrame()
        all_results: list[dict] = []
        if _initial_data:
            all_results.extend(_initial_data.get("results", []))
            next_url = _initial_data.get("next_url")
        else:
            base_url = self.BASE_URL.format(
                ticker=ticker.upper(),
                multiplier=interval_minutes,
                timespan="minute",
                from_date=start,
                to_date=end,
            )
            try:
                data = self._get(base_url, {"adjusted": "true", "sort": "asc", "limit": 50000})
                all_results.extend(data.get("results", []))
                next_url = data.get("next_url")
            except Exception as exc:
                logger.warning("Polygon paginated initial fetch failed: %s", exc)
                return pd.DataFrame()

        page_count = 1
        while next_url and page_count < 20:
            try:
                self._throttle()
                resp = self._client.get(next_url, params={"apiKey": self._api_key})
                resp.raise_for_status()
                data = resp.json()
                all_results.extend(data.get("results", []))
                next_url = data.get("next_url")
                page_count += 1
            except Exception as exc:
                logger.warning("Polygon pagination page %d failed: %s", page_count, exc)
                break

        if not all_results:
            return pd.DataFrame()
        df = pd.DataFrame(all_results)
        df = df.rename(columns={
            "t": "time", "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "vw": "vwap", "n": "trade_count",
        })
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df["ticker"] = ticker.upper()
        return df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stooq intraday adapter (free, 5-min, ~1 year)
# ---------------------------------------------------------------------------

class StooqIntradayAdapter:
    """Stooq.com free 5-min bars — ~1 year history, no API key needed.

    US tickers must use the .us suffix (e.g. aapl.us, msft.us).
    Data availability varies; some tickers have gaps.
    """

    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SENTINEL/1.0)"},
        )

    def _stooq_ticker(self, ticker: str) -> str:
        """Convert standard ticker to Stooq format (append .us for US equities)."""
        t = ticker.lower()
        if "." not in t:
            return f"{t}.us"
        return t

    def get_5min_bars(self, ticker: str, start: str = None) -> pd.DataFrame:
        """Fetch ~1 year of 5-min bars from Stooq.

        Parameters
        ----------
        ticker: e.g. "AAPL" (converted to aapl.us internally)
        start:  optional ISO date string; Stooq returns all available if omitted

        Returns
        -------
        DataFrame: time, open, high, low, close, volume
        """
        params: dict[str, Any] = {
            "s": self._stooq_ticker(ticker),
            "i": "5",  # 5-minute interval code
        }
        if start:
            params["d1"] = start.replace("-", "")  # Stooq uses YYYYMMDD
        try:
            resp = self._client.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            if df.empty or "Date" not in df.columns:
                logger.warning("Stooq returned no data for %s", ticker)
                return pd.DataFrame()
            # Combine Date + Time columns
            if "Time" in df.columns:
                df["time"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
            else:
                df["time"] = pd.to_datetime(df["Date"])
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            keep = ["time", "open", "high", "low", "close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            df["ticker"] = ticker.upper()
            df["data_source"] = "stooq_5m"
            return df.sort_values("time").reset_index(drop=True)
        except Exception as exc:
            logger.warning("StooqIntradayAdapter.get_5min_bars failed for %s: %s", ticker, exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# IntradayDataManager — unified router
# ---------------------------------------------------------------------------

class IntradayDataManager:
    """Unified interface for intraday data, routing to best available source.

    Source selection logic:
      1min + recent (<= 5yr)  → Alpaca IEX (free), yfinance fallback (7d only)
      1min + older            → Polygon (if API key set), otherwise limitation note
      5min + <= 1yr           → Stooq (free, no key)
      1h   + <= 2yr           → yfinance
      Any                     → graceful degradation with coverage_note

    All results include metadata: data_source, coverage_note, max_lookback_days
    """

    def __init__(
        self,
        alpaca_key: str = "",
        alpaca_secret: str = "",
        polygon_key: str = "",
    ) -> None:
        self.alpaca = AlpacaIntradayAdapter(alpaca_key, alpaca_secret)
        self.polygon = PolygonIntradayAdapter(polygon_key)
        self.stooq = StooqIntradayAdapter()
        self._meta: dict[str, Any] = {}

    def get_intraday(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1min",
    ) -> pd.DataFrame:
        """Fetch intraday OHLCV with automatic source routing.

        Returns DataFrame with extra columns:
          data_source     — which provider was used
          coverage_note   — human-readable limitation string
        """
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        lookback_days = (end_dt - start_dt).days

        df = pd.DataFrame()
        source = "none"
        note = ""

        norm = interval.lower().replace(" ", "")

        # --- 1-minute routing ---
        if norm in ("1min", "1m", "1minute"):
            if lookback_days <= 1825:  # <= 5 years
                df = self.alpaca.get_bars_chunked(ticker, start, end, chunk_days=90)
                if not df.empty:
                    source = "alpaca_iex"
                    note = DEPTH_BY_SOURCE["alpaca_iex"]["note"]
                elif lookback_days <= 7:
                    df = self._yfinance_fetch(ticker, start, end, "1m")
                    source = "yfinance"
                    note = DEPTH_BY_SOURCE["yfinance"]["note"]
                else:
                    note = (
                        "Alpaca unavailable and yfinance limited to 7 days for 1-min. "
                        "Set ALPACA_API_KEY or use Polygon/FirstRate for older data."
                    )
            elif self.polygon._api_key:
                df = self.polygon.get_bars_paginated(ticker, start, end, 1)
                source = "polygon_free"
                note = DEPTH_BY_SOURCE["polygon_free"]["note"]
            else:
                note = (
                    f"1-min data older than 5 years requires Polygon (POLYGON_API_KEY) "
                    f"or FirstRate Data (paid). {DEPTH_BY_SOURCE['firstrate_data']['note']}"
                )

        # --- 5-minute routing ---
        elif norm in ("5min", "5m", "5minute"):
            if lookback_days <= 365:
                df = self.stooq.get_5min_bars(ticker, start)
                source = "stooq_5m"
                note = DEPTH_BY_SOURCE["stooq_5m"]["note"]
            elif lookback_days <= 60:
                df = self._yfinance_fetch(ticker, start, end, "5m")
                source = "yfinance_5m"
                note = DEPTH_BY_SOURCE["yfinance_5m"]["note"]
            elif self.polygon._api_key:
                df = self.polygon.get_bars_paginated(ticker, start, end, 5)
                source = "polygon_free"
                note = DEPTH_BY_SOURCE["polygon_free"]["note"]

        # --- 15-minute routing ---
        elif norm in ("15min", "15m", "15minute"):
            if lookback_days <= 60:
                df = self._yfinance_fetch(ticker, start, end, "15m")
                source = "yfinance_15m"
                note = DEPTH_BY_SOURCE["yfinance_15m"]["note"]
            elif self.polygon._api_key:
                df = self.polygon.get_bars_paginated(ticker, start, end, 15)
                source = "polygon_free"
                note = DEPTH_BY_SOURCE["polygon_free"]["note"]

        # --- 1-hour routing ---
        elif norm in ("1h", "1hour", "60min", "60m"):
            if lookback_days <= 730:
                df = self._yfinance_fetch(ticker, start, end, "1h")
                source = "yfinance_1h"
                note = DEPTH_BY_SOURCE["yfinance_1h"]["note"]
            else:
                note = "yfinance 1h limited to ~2 years. Use Alpaca or Polygon for longer history."

        else:
            note = f"Interval '{interval}' not recognised. Supported: 1min, 5min, 15min, 1h."

        if not df.empty:
            df["data_source"] = source
            df["coverage_note"] = note
        self._meta = {"data_source": source, "coverage_note": note, "rows": len(df)}
        return df

    def _yfinance_fetch(
        self, ticker: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, interval=interval, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            time_col = "datetime" if "datetime" in df.columns else "date"
            df = df.rename(columns={time_col: "time"})
            df["ticker"] = ticker.upper()
            return df[["time", "open", "high", "low", "close", "volume", "ticker"]]
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s [%s]: %s", ticker, interval, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def build_intraday_cache(
        self,
        ticker: str,
        lookback_days: int = 90,
        interval: str = "1min",
    ) -> None:
        """Download intraday data and persist to DuckDB cache.

        Table: intraday_{interval} (time, ticker, open, high, low, close, volume, vwap)
        Database: .sentinel/cache/intraday.duckdb
        """
        try:
            import duckdb
        except ImportError:
            logger.error("duckdb not installed. Run: pip install duckdb")
            return
        end = datetime.utcnow().date().isoformat()
        start = (datetime.utcnow().date() - timedelta(days=lookback_days)).isoformat()
        df = self.get_intraday(ticker, start, end, interval)
        if df.empty:
            logger.warning("build_intraday_cache: no data for %s [%s]", ticker, interval)
            return
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        table = "intraday_" + interval.replace("-", "_").replace("/", "_")
        con = duckdb.connect(str(_INTRADAY_DB))
        try:
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    time        TIMESTAMPTZ,
                    ticker      VARCHAR,
                    open        DOUBLE,
                    high        DOUBLE,
                    low         DOUBLE,
                    close       DOUBLE,
                    volume      DOUBLE,
                    vwap        DOUBLE,
                    data_source VARCHAR,
                    PRIMARY KEY (time, ticker)
                )
            """)
            # Upsert: delete existing range then insert
            con.execute(
                f"DELETE FROM {table} WHERE ticker = ? AND time >= ? AND time <= ?",
                [ticker, start, end],
            )
            cols = ["time", "ticker", "open", "high", "low", "close", "volume"]
            optional = ["vwap", "data_source"]
            for oc in optional:
                if oc not in df.columns:
                    df[oc] = None
            insert_df = df[[c for c in cols + optional if c in df.columns]]
            con.register("_intraday_insert", insert_df)
            con.execute(f"INSERT INTO {table} SELECT * FROM _intraday_insert")
            logger.info(
                "build_intraday_cache: wrote %d rows for %s [%s] to %s",
                len(insert_df), ticker, interval, _INTRADAY_DB,
            )
        finally:
            con.close()

    def get_cached(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame | None:
        """Read intraday data from DuckDB cache. Returns None if cache miss."""
        try:
            import duckdb
        except ImportError:
            return None
        if not _INTRADAY_DB.exists():
            return None
        table = "intraday_" + interval.replace("-", "_").replace("/", "_")
        con = duckdb.connect(str(_INTRADAY_DB), read_only=True)
        try:
            df = con.execute(
                f"SELECT * FROM {table} WHERE ticker = ? AND time >= ? AND time <= ? ORDER BY time",
                [ticker, start, end],
            ).df()
            return df if not df.empty else None
        except Exception:
            return None
        finally:
            con.close()

    def compute_intraday_patterns(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute intraday trading statistics from a bars DataFrame.

        Returns dict with keys:
          avg_open_to_close, avg_first_hour_move, avg_last_hour_move,
          overnight_gap_avg, am_pm_volume_ratio, vwap_relative_vol_by_hour
        """
        if df.empty or "close" not in df.columns:
            return {}
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time")
        # Daily grouping
        df["date"] = df["time"].dt.date
        result: dict[str, Any] = {}
        # Open-to-close move per day
        daily = df.groupby("date").agg(
            day_open=("open", "first"),
            day_close=("close", "last"),
            day_high=("high", "max"),
            day_low=("low", "min"),
        )
        daily["oc_move"] = (daily["day_close"] - daily["day_open"]) / daily["day_open"]
        result["avg_open_to_close"] = float(daily["oc_move"].mean())
        result["std_open_to_close"] = float(daily["oc_move"].std())
        # Overnight gap: today's open vs yesterday's close
        daily_sorted = daily.sort_index()
        gaps = (daily_sorted["day_open"] - daily_sorted["day_close"].shift(1)) / daily_sorted["day_close"].shift(1)
        result["overnight_gap_avg"] = float(gaps.mean())
        result["overnight_gap_std"] = float(gaps.std())
        # First-hour and last-hour moves (assume market 09:30–16:00 ET)
        df["hour"] = df["time"].dt.hour
        first_hour = df[df["hour"] == 9]  # 09:xx bars (ET)
        last_hour = df[df["hour"] == 15]  # 15:xx bars (ET)
        if not first_hour.empty:
            fh_by_day = first_hour.groupby("date").agg(fh_open=("open", "first"), fh_close=("close", "last"))
            fh_move = (fh_by_day["fh_close"] - fh_by_day["fh_open"]) / fh_by_day["fh_open"]
            result["avg_first_hour_move"] = float(fh_move.mean())
        if not last_hour.empty:
            lh_by_day = last_hour.groupby("date").agg(lh_open=("open", "first"), lh_close=("close", "last"))
            lh_move = (lh_by_day["lh_close"] - lh_by_day["lh_open"]) / lh_by_day["lh_open"]
            result["avg_last_hour_move"] = float(lh_move.mean())
        # Volume by hour
        if "volume" in df.columns:
            vol_by_hour = df.groupby("hour")["volume"].mean()
            result["vwap_relative_vol_by_hour"] = vol_by_hour.to_dict()
            am_vol = df[df["hour"] < 12]["volume"].sum()
            pm_vol = df[df["hour"] >= 12]["volume"].sum()
            result["am_pm_volume_ratio"] = float(am_vol / pm_vol) if pm_vol > 0 else None
        return result


# ---------------------------------------------------------------------------
# VolumeProfileAnalyzer
# ---------------------------------------------------------------------------

class VolumeProfileAnalyzer:
    """Volume Profile, VWAP, and Market Profile analytics for intraday data."""

    def compute_vwap(self, df: pd.DataFrame, period: str = "day") -> pd.Series:
        """Compute VWAP with daily reset.

        VWAP = cumsum(typical_price * volume) / cumsum(volume)
        typical_price = (high + low + close) / 3
        """
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time")
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["tp_vol"] = df["typical_price"] * df["volume"]

        if period == "day":
            df["_group"] = df["time"].dt.date
        elif period == "week":
            df["_group"] = df["time"].dt.to_period("W")
        else:
            df["_group"] = 0  # cumulative over all data

        df["cum_tp_vol"] = df.groupby("_group")["tp_vol"].cumsum()
        df["cum_vol"] = df.groupby("_group")["volume"].cumsum()
        vwap = df["cum_tp_vol"] / df["cum_vol"]
        vwap.name = "vwap"
        return vwap

    def compute_volume_profile(self, df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
        """Volume at Price — compute VAH, VAL, POC.

        Returns DataFrame indexed by price bins with:
          volume, pct_volume, poc (bool), vah (bool), val (bool)

        VAH = Value Area High (top of 70% volume value area)
        VAL = Value Area Low  (bottom of 70% volume value area)
        POC = Point of Control (price level with most volume)
        """
        if df.empty:
            return pd.DataFrame()
        price_min = df["low"].min()
        price_max = df["high"].max()
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_by_bin = np.zeros(bins)
        for _, row in df.iterrows():
            # Distribute bar volume evenly across price range it spans
            bar_low, bar_high = row["low"], row["high"]
            bar_vol = row.get("volume", 0)
            bar_bins = np.where(
                (bin_centers >= bar_low) & (bin_centers <= bar_high)
            )[0]
            if len(bar_bins) > 0:
                vol_by_bin[bar_bins] += bar_vol / len(bar_bins)
        profile = pd.DataFrame({
            "price": bin_centers,
            "volume": vol_by_bin,
        })
        profile["pct_volume"] = profile["volume"] / profile["volume"].sum()
        poc_idx = profile["volume"].idxmax()
        profile["poc"] = False
        profile.loc[poc_idx, "poc"] = True
        # Value area: sort by volume desc, accumulate to 70%
        sorted_idx = profile["volume"].sort_values(ascending=False).index
        cum_pct = 0.0
        va_idx: set[int] = set()
        for idx in sorted_idx:
            va_idx.add(idx)
            cum_pct += profile.loc[idx, "pct_volume"]
            if cum_pct >= 0.70:
                break
        profile["in_value_area"] = profile.index.isin(va_idx)
        va_prices = profile.loc[profile["in_value_area"], "price"]
        profile["vah"] = profile["price"] == va_prices.max()
        profile["val"] = profile["price"] == va_prices.min()
        return profile.reset_index(drop=True)

    def compute_market_profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """TPO-based Market Profile (Time Price Opportunity).

        Each 30-minute period is one TPO letter. Returns:
          tpo_counts: {price_level: count_of_30min_periods}
          poc_price: price with most TPOs
          initial_balance: high/low of first hour (9:30–10:30 ET)
          range_extension: number of TPOs outside initial balance
        """
        if df.empty:
            return {}
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        # Convert to ET (UTC-4 or UTC-5; approximate with -4)
        df["time_et"] = df["time"] - pd.Timedelta(hours=4)
        df["tpo_period"] = (df["time_et"].dt.hour * 60 + df["time_et"].dt.minute) // 30
        price_min = df["low"].min()
        price_max = df["high"].max()
        tick = (price_max - price_min) / 100  # 100 price buckets
        tpo_counts: dict[float, int] = {}
        for _, row in df.iterrows():
            period = row["tpo_period"]
            bar_low, bar_high = row["low"], row["high"]
            level = price_min
            while level <= bar_high:
                if level >= bar_low:
                    bucket = round(level / tick) * tick
                    tpo_counts[bucket] = tpo_counts.get(bucket, 0) + 1
                level += tick
        poc_price = max(tpo_counts, key=tpo_counts.get) if tpo_counts else None
        # Initial balance: 9:30–10:30 ET = tpo_periods 19-21 (9.5hr = period 19)
        ib_periods = [19, 20, 21]
        ib_df = df[df["tpo_period"].isin(ib_periods)]
        ib_high = float(ib_df["high"].max()) if not ib_df.empty else None
        ib_low = float(ib_df["low"].min()) if not ib_df.empty else None
        range_extension = 0
        if ib_high and ib_low:
            for bucket, count in tpo_counts.items():
                if bucket > ib_high or bucket < ib_low:
                    range_extension += count
        return {
            "tpo_counts": tpo_counts,
            "poc_price": poc_price,
            "initial_balance_high": ib_high,
            "initial_balance_low": ib_low,
            "range_extension_tpos": range_extension,
            "total_tpos": sum(tpo_counts.values()),
        }

    def vwap_bands(
        self,
        df: pd.DataFrame,
        std_multipliers: list[float] = [1.0, 2.0, 3.0],
    ) -> pd.DataFrame:
        """Compute VWAP with standard deviation bands.

        Returns DataFrame with columns:
          vwap, vwap_upper_1, vwap_lower_1, vwap_upper_2, ... etc.
        based on rolling standard deviation of price from VWAP.
        """
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time")
        vwap_series = self.compute_vwap(df, period="day")
        df["vwap"] = vwap_series.values
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["_group"] = df["time"].dt.date
        # Rolling std of typical price deviation from VWAP within each day
        df["dev_sq"] = (df["typical_price"] - df["vwap"]) ** 2
        df["cum_dev_sq"] = df.groupby("_group")["dev_sq"].cumsum()
        df["bar_count"] = df.groupby("_group").cumcount() + 1
        df["vwap_std"] = np.sqrt(df["cum_dev_sq"] / df["bar_count"])
        result = df[["time", "open", "high", "low", "close", "volume", "vwap"]].copy()
        for mult in std_multipliers:
            label = str(int(mult)) if mult == int(mult) else str(mult)
            result[f"vwap_upper_{label}"] = df["vwap"] + mult * df["vwap_std"]
            result[f"vwap_lower_{label}"] = df["vwap"] - mult * df["vwap_std"]
        return result.reset_index(drop=True)
