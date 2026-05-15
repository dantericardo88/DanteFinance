"""SENTINEL TradingView Enhanced UDF Integration — Dimension #092 (target 9+).

Full implementation of the TradingView Universal Data Feed (UDF) specification,
making SENTINEL a proper TradingView data provider.

UDF Spec: https://www.tradingview.com/charting-library-docs/latest/connecting_data/UDF/

Endpoints
---------
GET /tv/config              — Data feed configuration
GET /tv/symbols             — Symbol metadata (symbol resolution)
GET /tv/search              — Symbol search
GET /tv/history             — OHLCV bars
GET /tv/marks               — Earnings/dividend/split event marks
GET /tv/timescale_marks     — News event timescale marks
GET /tv/time                — Server UTC timestamp

Data source routing
-------------------
Intraday equities  → Alpaca IEX (via existing QuoteAggregator)
Daily equities     → yfinance (Stooq fallback)
Daily crypto       → CoinGecko OHLC API
Intraday crypto    → Binance public klines API
Futures            → yfinance futures tickers

Usage
-----
    from sentinel.api.tradingview_enhanced import tradingview_router
    app.include_router(tradingview_router, prefix="")
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# In-process bar cache  (resolution → TTL seconds)
# ---------------------------------------------------------------------------

_BAR_CACHE: dict[str, tuple[float, dict]] = {}
_BAR_CACHE_TTL: dict[str, float] = {
    "intraday": 60.0,   # 1-minute TTL for intraday bars
    "daily":    3600.0, # 1-hour TTL for daily/weekly/monthly
}


def _bar_cache_key(ticker: str, resolution: str, from_ts: int, to_ts: int) -> str:
    return f"{ticker}:{resolution}:{from_ts}:{to_ts}"


def _bar_cache_get(key: str, is_intraday: bool) -> Optional[dict]:
    entry = _BAR_CACHE.get(key)
    if entry is None:
        return None
    ts, data = entry
    ttl = _BAR_CACHE_TTL["intraday"] if is_intraday else _BAR_CACHE_TTL["daily"]
    if time.monotonic() - ts > ttl:
        _BAR_CACHE.pop(key, None)
        return None
    return data


def _bar_cache_set(key: str, data: dict) -> None:
    _BAR_CACHE[key] = (time.monotonic(), data)


# ---------------------------------------------------------------------------
# UDF Configuration constants
# ---------------------------------------------------------------------------

UDF_CONFIG = {
    "supported_resolutions": ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"],
    "supports_group_request": False,
    "supports_marks": True,
    "supports_search": True,
    "supports_timescale_marks": True,
    "supports_time": True,
    "exchanges": [
        {"value": "SENTINEL", "name": "SENTINEL Terminal", "desc": "SENTINEL unified feed"},
        {"value": "NASDAQ",   "name": "NASDAQ",            "desc": ""},
        {"value": "NYSE",     "name": "NYSE",              "desc": ""},
        {"value": "CRYPTO",   "name": "Crypto",            "desc": ""},
    ],
    "symbols_types": [
        {"name": "Stock",  "value": "stock"},
        {"name": "Crypto", "value": "crypto"},
        {"name": "ETF",    "value": "etf"},
        {"name": "Forex",  "value": "forex"},
        {"name": "Index",  "value": "index"},
    ],
}

# ---------------------------------------------------------------------------
# SymbolResolver
# ---------------------------------------------------------------------------

class SymbolResolver:
    """Parse and resolve symbol strings to structured metadata.

    Handles formats: AAPL, SENTINEL:AAPL, AAPL:NASDAQ, BTC-USD (crypto).
    """

    TIMEZONE_MAP: dict[str, str] = {
        "NMS":   "America/New_York",
        "NGM":   "America/New_York",
        "NCM":   "America/New_York",
        "NYQ":   "America/New_York",
        "NYSE":  "America/New_York",
        "NASDAQ":"America/New_York",
        "AMEX":  "America/New_York",
        "BATS":  "America/New_York",
        "LSE":   "Europe/London",
        "XLON":  "Europe/London",
        "FRA":   "Europe/Berlin",
        "XETR":  "Europe/Berlin",
        "STO":   "Europe/Stockholm",
        "TSX":   "America/Toronto",
        "ASX":   "Australia/Sydney",
        "HKG":   "Asia/Hong_Kong",
        "TYO":   "Asia/Tokyo",
        "SNP":   "America/New_York",  # S&P 500 index
        "CCC":   "UTC",               # Crypto
        "CCY":   "UTC",               # Currency
    }

    PRICESCALE_MAP: dict[str, int] = {
        "USD": 100,
        "EUR": 100,
        "GBP": 100,
        "JPY": 1,
        "CAD": 100,
        "AUD": 100,
        "CHF": 100,
        "HKD": 100,
        "BTC": 100000,
        "ETH": 10000,
        "USDT": 10000,
    }

    SESSION_MAP: dict[str, str] = {
        "America/New_York": "0930-1600",
        "Europe/London":    "0800-1630",
        "Europe/Berlin":    "0900-1730",
        "Asia/Tokyo":       "0900-1530",
        "Asia/Hong_Kong":   "0930-1600",
        "Australia/Sydney": "1000-1600",
        "UTC":              "24x7",
    }

    # Small curated universe for fast fuzzy search (augmented by yfinance Search)
    _UNIVERSE_CACHE: list[dict] = []
    _UNIVERSE_LOADED = False

    def resolve(self, symbol: str) -> dict:
        """Parse symbol string into {ticker, exchange} dict.

        Examples:
            "AAPL"           → {"ticker": "AAPL",    "exchange": ""}
            "SENTINEL:AAPL"  → {"ticker": "AAPL",    "exchange": "SENTINEL"}
            "AAPL:NASDAQ"    → {"ticker": "AAPL",    "exchange": "NASDAQ"}
            "BTC-USD"        → {"ticker": "BTC-USD", "exchange": "CRYPTO"}
        """
        s = symbol.strip().upper()
        # Strip SENTINEL: prefix
        if s.startswith("SENTINEL:"):
            s = s[len("SENTINEL:"):]

        # Handle TICKER:EXCHANGE format
        if ":" in s:
            parts = s.split(":", 1)
            return {"ticker": parts[0], "exchange": parts[1]}

        # Detect crypto by common patterns
        if any(s.endswith(suffix) for suffix in ("-USD", "-USDT", "-BTC", "-ETH")):
            return {"ticker": s, "exchange": "CRYPTO"}

        return {"ticker": s, "exchange": ""}

    def get_symbol_info(self, ticker: str) -> dict:
        """Return full TradingView SymbolInfo from yfinance.

        Called synchronously — wrap in executor from async contexts.
        """
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
        except Exception:
            info = {}

        exchange_raw = info.get("exchange", "")
        tz = self.TIMEZONE_MAP.get(exchange_raw, "America/New_York")
        session = self.SESSION_MAP.get(tz, "0930-1600")

        # Determine asset class
        quote_type = (info.get("quoteType") or "EQUITY").upper()
        if quote_type in ("CRYPTOCURRENCY", "DIGITAL CURRENCY"):
            asset_type = "crypto"
            session = "24x7"
            tz = "UTC"
        elif quote_type in ("ETF", "MUTUALFUND"):
            asset_type = "etf"
        elif quote_type in ("FUTURE",):
            asset_type = "futures"
        elif quote_type in ("FOREX", "CURRENCY"):
            asset_type = "forex"
            session = "24x7"
            tz = "UTC"
        elif quote_type in ("INDEX",):
            asset_type = "index"
        else:
            asset_type = "stock"

        currency = (info.get("currency") or "USD").upper()
        pricescale = self.PRICESCALE_MAP.get(currency, 100)

        # Intraday availability: yfinance supports 1m for equities (last 7 days)
        has_intraday = asset_type in ("stock", "etf", "crypto", "forex")

        return {
            "name":                    ticker.upper(),
            "ticker":                  ticker.upper(),
            "description":             info.get("shortName") or info.get("longName") or ticker.upper(),
            "type":                    asset_type,
            "session":                 session,
            "timezone":                tz,
            "exchange":                exchange_raw,
            "listed_exchange":         info.get("exchange", ""),
            "minmov":                  1,
            "pricescale":              pricescale,
            "has_intraday":            has_intraday,
            "intraday_multipliers":    ["1", "3", "5", "15", "30", "60", "120", "240"],
            "has_daily":               True,
            "has_weekly_and_monthly":  True,
            "has_empty_bars":          False,
            "volume_precision":        0,
            "data_status":             "streaming",
            "currency_code":           currency,
            "supported_resolutions":   ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"],
            "sector":                  info.get("sector"),
            "industry":                info.get("industry"),
        }

    def search_symbols(self, query: str, limit: int = 30, type_filter: str = "") -> list[dict]:
        """Fuzzy symbol search using yfinance Search API.

        Returns list of TradingView search result dicts.
        """
        results = []
        try:
            search = yf.Search(query, max_results=limit * 2)
            quotes = search.quotes or []
            for r in quotes:
                sym = (r.get("symbol") or "").upper()
                if not sym:
                    continue
                qt = (r.get("quoteType") or "EQUITY").upper()
                asset_type = {
                    "EQUITY": "stock",
                    "ETF": "etf",
                    "MUTUALFUND": "etf",
                    "CRYPTOCURRENCY": "crypto",
                    "FUTURE": "futures",
                    "FOREX": "forex",
                    "INDEX": "index",
                }.get(qt, "stock")

                if type_filter and type_filter.lower() not in (asset_type, ""):
                    continue

                results.append({
                    "symbol":      sym,
                    "full_name":   f"SENTINEL:{sym}",
                    "description": r.get("shortname") or r.get("longname") or sym,
                    "exchange":    r.get("exchange") or "NASDAQ",
                    "type":        asset_type,
                })
                if len(results) >= limit:
                    break
        except Exception as exc:
            logger.debug("symbol_search_error: %s", exc)

        return results


# ---------------------------------------------------------------------------
# BarDataRouter
# ---------------------------------------------------------------------------

class BarDataRouter:
    """Route bar data requests to the appropriate data source.

    Priority:
        Intraday equities/ETFs → Alpaca IEX (falls back to yfinance)
        Daily equities          → yfinance (Stooq fallback)
        Crypto (any resolution) → CoinGecko daily / Binance klines intraday
        Futures                 → yfinance futures tickers
    """

    _ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"
    _COINGECKO_BASE   = "https://api.coingecko.com/api/v3"
    _BINANCE_BASE     = "https://api.binance.com/api/v3"

    # Map yfinance crypto ticker → CoinGecko id
    _COINGECKO_ID_MAP: dict[str, str] = {
        "BTC-USD": "bitcoin",
        "ETH-USD": "ethereum",
        "SOL-USD": "solana",
        "BNB-USD": "binancecoin",
        "XRP-USD": "ripple",
        "ADA-USD": "cardano",
        "AVAX-USD": "avalanche-2",
        "DOGE-USD": "dogecoin",
        "MATIC-USD": "matic-network",
        "DOT-USD": "polkadot",
        "LINK-USD": "chainlink",
        "LTC-USD": "litecoin",
    }

    # Binance trading pair from yfinance ticker
    _BINANCE_SYMBOL_MAP: dict[str, str] = {
        "BTC-USD":  "BTCUSDT",
        "ETH-USD":  "ETHUSDT",
        "SOL-USD":  "SOLUSDT",
        "BNB-USD":  "BNBUSDT",
        "XRP-USD":  "XRPUSDT",
        "ADA-USD":  "ADAUSDT",
        "AVAX-USD": "AVAXUSDT",
        "DOGE-USD": "DOGEUSDT",
        "MATIC-USD":"MATICUSDT",
        "DOT-USD":  "DOTUSDT",
        "LINK-USD": "LINKUSDT",
        "LTC-USD":  "LTCUSDT",
    }

    def __init__(self) -> None:
        self._alpaca_key    = os.environ.get("ALPACA_API_KEY", "")
        self._alpaca_secret = os.environ.get("ALPACA_SECRET_KEY", "")

    # ------------------------------------------------------------------
    def _resolution_to_interval(self, resolution: str) -> str:
        """Map TradingView resolution to yfinance interval string."""
        _MAP = {
            "1":   "1m",
            "3":   "5m",   # yfinance has no 3m; use 5m
            "5":   "5m",
            "15":  "15m",
            "30":  "30m",
            "60":  "1h",
            "120": "1h",
            "240": "1h",
            "D":   "1d",
            "W":   "1wk",
            "M":   "1mo",
        }
        return _MAP.get(resolution, "1d")

    def _resolution_to_alpaca_timeframe(self, resolution: str) -> str:
        """Map TradingView resolution to Alpaca timeframe string."""
        _MAP = {
            "1":   "1Min",
            "3":   "3Min",
            "5":   "5Min",
            "15":  "15Min",
            "30":  "30Min",
            "60":  "1Hour",
            "120": "2Hour",
            "240": "4Hour",
            "D":   "1Day",
            "W":   "1Week",
            "M":   "1Month",
        }
        return _MAP.get(resolution, "1Day")

    def _resolution_to_binance_interval(self, resolution: str) -> str:
        """Map TradingView resolution to Binance klines interval."""
        _MAP = {
            "1":   "1m",
            "3":   "3m",
            "5":   "5m",
            "15":  "15m",
            "30":  "30m",
            "60":  "1h",
            "120": "2h",
            "240": "4h",
            "D":   "1d",
            "W":   "1w",
            "M":   "1M",
        }
        return _MAP.get(resolution, "1d")

    def _is_intraday(self, resolution: str) -> bool:
        return resolution not in ("D", "W", "M")

    def _is_crypto(self, ticker: str) -> bool:
        return "-USD" in ticker.upper() or "-USDT" in ticker.upper() or "-BTC" in ticker.upper()

    # ------------------------------------------------------------------
    async def get_bars(
        self,
        ticker: str,
        resolution: str,
        from_ts: int,
        to_ts: int,
    ) -> dict:
        """Main entry: route bar request and return UDF-format dict."""
        ticker = ticker.upper()
        cache_key = _bar_cache_key(ticker, resolution, from_ts, to_ts)
        is_intraday = self._is_intraday(resolution)
        cached = _bar_cache_get(cache_key, is_intraday)
        if cached is not None:
            return cached

        try:
            if self._is_crypto(ticker):
                if is_intraday:
                    result = await self._get_binance_bars(ticker, resolution, from_ts, to_ts)
                else:
                    result = await self._get_coingecko_bars(ticker, from_ts, to_ts)
            elif is_intraday:
                result = await self._get_alpaca_bars(ticker, resolution, from_ts, to_ts)
            else:
                result = await self._get_yfinance_bars(ticker, resolution, from_ts, to_ts)
        except Exception as exc:
            logger.error("bar_router_error: ticker=%s resolution=%s error=%s", ticker, resolution, exc)
            result = {"s": "error", "errmsg": str(exc)}

        if result.get("s") == "ok":
            _bar_cache_set(cache_key, result)

        return result

    # ------------------------------------------------------------------
    async def _get_alpaca_bars(
        self, ticker: str, resolution: str, from_ts: int, to_ts: int
    ) -> dict:
        """Fetch intraday bars from Alpaca IEX."""
        timeframe = self._resolution_to_alpaca_timeframe(resolution)
        from_dt = datetime.fromtimestamp(from_ts, tz=_UTC).isoformat()
        to_dt   = datetime.fromtimestamp(to_ts,   tz=_UTC).isoformat()

        url = f"{self._ALPACA_DATA_BASE}/stocks/{ticker}/bars"
        params: dict = {
            "timeframe": timeframe,
            "start":     from_dt,
            "end":       to_dt,
            "limit":     10000,
            "feed":      "iex",
            "sort":      "asc",
        }
        headers: dict = {"Accept": "application/json"}
        if self._alpaca_key:
            headers["APCA-API-KEY-ID"]     = self._alpaca_key
            headers["APCA-API-SECRET-KEY"] = self._alpaca_secret

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, params=params, headers=headers)
            if resp.status_code in (403, 404, 422):
                # Fall back to yfinance
                return await self._get_yfinance_bars(ticker, resolution, from_ts, to_ts)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return await self._get_yfinance_bars(ticker, resolution, from_ts, to_ts)

        bars = data.get("bars") or []
        if not bars:
            return {"s": "no_data"}

        t_list, o_list, h_list, l_list, c_list, v_list = [], [], [], [], [], []
        for bar in bars:
            dt = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
            t_list.append(int(dt.timestamp()))
            o_list.append(bar.get("o"))
            h_list.append(bar.get("h"))
            l_list.append(bar.get("l"))
            c_list.append(bar.get("c"))
            v_list.append(int(bar.get("v", 0)))

        return {"s": "ok", "t": t_list, "o": o_list, "h": h_list,
                "l": l_list, "c": c_list, "v": v_list}

    # ------------------------------------------------------------------
    async def _get_yfinance_bars(
        self, ticker: str, resolution: str, from_ts: int, to_ts: int
    ) -> dict:
        """Fetch daily/weekly/monthly bars from yfinance with Stooq fallback."""
        interval = self._resolution_to_interval(resolution)
        from_dt  = datetime.fromtimestamp(from_ts, tz=_UTC)
        to_dt    = datetime.fromtimestamp(to_ts,   tz=_UTC)

        loop = asyncio.get_event_loop()

        def _fetch() -> pd.DataFrame:
            df = yf.download(
                ticker,
                start=from_dt.strftime("%Y-%m-%d"),
                end=(to_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df

        try:
            df = await loop.run_in_executor(None, _fetch)
        except Exception as exc:
            return {"s": "error", "errmsg": f"yfinance: {exc}"}

        if df is None or df.empty:
            # Stooq fallback for daily bars
            if resolution in ("D", "W", "M"):
                return await self._get_stooq_bars(ticker, from_ts, to_ts)
            return {"s": "no_data"}

        return self._df_to_udf(df)

    async def _get_stooq_bars(self, ticker: str, from_ts: int, to_ts: int) -> dict:
        """Stooq.com daily OHLCV as fallback."""
        from_dt = datetime.fromtimestamp(from_ts, tz=_UTC)
        to_dt   = datetime.fromtimestamp(to_ts,   tz=_UTC)
        stooq_ticker = ticker.replace("-", ".").lower()
        url = (
            f"https://stooq.com/q/d/l/?s={stooq_ticker}"
            f"&d1={from_dt.strftime('%Y%m%d')}&d2={to_dt.strftime('%Y%m%d')}&i=d"
        )
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url)
            if resp.status_code != 200 or "Date" not in resp.text:
                return {"s": "no_data"}
            df = pd.read_csv(pd.io.common.StringIO(resp.text), parse_dates=["Date"])
            df = df.rename(columns={"Date": "date", "Open": "Open", "High": "High",
                                     "Low": "Low", "Close": "Close", "Volume": "Volume"})
            df = df.set_index("date").sort_index()
            return self._df_to_udf(df)
        except Exception as exc:
            return {"s": "error", "errmsg": f"stooq: {exc}"}

    # ------------------------------------------------------------------
    async def _get_coingecko_bars(self, ticker: str, from_ts: int, to_ts: int) -> dict:
        """Fetch daily OHLC from CoinGecko."""
        coin_id = self._COINGECKO_ID_MAP.get(ticker)
        if not coin_id:
            # Fallback to yfinance
            return await self._get_yfinance_bars(ticker, "D", from_ts, to_ts)

        days = max(1, (to_ts - from_ts) // 86400 + 1)
        url = f"{self._COINGECKO_BASE}/coins/{coin_id}/ohlc"
        params = {"vs_currency": "usd", "days": min(days, 365)}

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return {"s": "error", "errmsg": f"coingecko: {exc}"}

        if not data:
            return {"s": "no_data"}

        t_list, o_list, h_list, l_list, c_list, v_list = [], [], [], [], [], []
        for row in data:
            ts_ms, o, h, l, c = row
            ts_s = ts_ms // 1000
            if ts_s < from_ts or ts_s > to_ts:
                continue
            t_list.append(ts_s)
            o_list.append(o)
            h_list.append(h)
            l_list.append(l)
            c_list.append(c)
            v_list.append(0)

        if not t_list:
            return {"s": "no_data"}

        return {"s": "ok", "t": t_list, "o": o_list, "h": h_list,
                "l": l_list, "c": c_list, "v": v_list}

    # ------------------------------------------------------------------
    async def _get_binance_bars(
        self, ticker: str, resolution: str, from_ts: int, to_ts: int
    ) -> dict:
        """Fetch intraday klines from Binance public API."""
        binance_sym = self._BINANCE_SYMBOL_MAP.get(ticker)
        if not binance_sym:
            return await self._get_yfinance_bars(ticker, resolution, from_ts, to_ts)

        interval  = self._resolution_to_binance_interval(resolution)
        url       = f"{self._BINANCE_BASE}/klines"
        params    = {
            "symbol":    binance_sym,
            "interval":  interval,
            "startTime": from_ts * 1000,
            "endTime":   to_ts * 1000,
            "limit":     1000,
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return {"s": "error", "errmsg": f"binance: {exc}"}

        if not data:
            return {"s": "no_data"}

        t_list, o_list, h_list, l_list, c_list, v_list = [], [], [], [], [], []
        for kline in data:
            t_list.append(kline[0] // 1000)
            o_list.append(float(kline[1]))
            h_list.append(float(kline[2]))
            l_list.append(float(kline[3]))
            c_list.append(float(kline[4]))
            v_list.append(float(kline[5]))

        return {"s": "ok", "t": t_list, "o": o_list, "h": h_list,
                "l": l_list, "c": c_list, "v": v_list}

    # ------------------------------------------------------------------
    @staticmethod
    def _df_to_udf(df: pd.DataFrame) -> dict:
        """Convert OHLCV DataFrame to TradingView UDF bars response."""
        if df is None or df.empty:
            return {"s": "no_data"}

        df = df.sort_index()

        def _ts(idx_val) -> int:
            if hasattr(idx_val, "timestamp"):
                return int(idx_val.timestamp())
            return int(pd.Timestamp(idx_val).timestamp())

        def _col(name: str) -> list:
            cap = name.capitalize()
            col = cap if cap in df.columns else name
            if col not in df.columns:
                return [None] * len(df)
            return [round(float(v), 6) if pd.notna(v) else None for v in df[col]]

        return {
            "s": "ok",
            "t": [_ts(i) for i in df.index],
            "o": _col("Open"),
            "h": _col("High"),
            "l": _col("Low"),
            "c": _col("Close"),
            "v": [int(v) if pd.notna(v) else 0 for v in df.get("Volume", [0] * len(df))],
        }


# ---------------------------------------------------------------------------
# EventMarkProvider
# ---------------------------------------------------------------------------

class EventMarkProvider:
    """Produce TradingView event marks for earnings, dividends, splits, and news.

    Mark colors follow TradingView conventions:
        red    — earnings miss / dividends
        green  — earnings beat
        blue   — splits / general events
        orange — analyst ratings
    """

    def get_earnings_marks(self, ticker: str, from_ts: int, to_ts: int) -> list[dict]:
        """Return earnings dates as TradingView marks."""
        marks = []
        try:
            tk = yf.Ticker(ticker)
            earnings = tk.earnings_dates
            if earnings is None or earnings.empty:
                return []

            for idx, row in earnings.iterrows():
                ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else None
                if ts is None or ts < from_ts or ts > to_ts:
                    continue

                eps_est   = row.get("EPS Estimate")
                eps_act   = row.get("Reported EPS")
                surprise  = row.get("Surprise(%)")

                color = "green" if (surprise and surprise > 0) else "red"
                label = "E"
                text  = f"Earnings: {ticker}"
                if eps_act is not None:
                    text += f" | EPS: {eps_act:.2f}"
                if surprise is not None:
                    text += f" | Surprise: {surprise:.1f}%"

                marks.append({
                    "id":              f"earn_{ticker}_{ts}",
                    "time":            ts,
                    "color":           color,
                    "text":            text,
                    "label":           label,
                    "labelFontColor":  "white",
                    "minSize":         14,
                })
        except Exception as exc:
            logger.debug("earnings_marks_error: %s %s", ticker, exc)

        return marks

    def get_dividend_marks(self, ticker: str, from_ts: int, to_ts: int) -> list[dict]:
        """Return dividend ex-dates as TradingView marks."""
        marks = []
        try:
            tk = yf.Ticker(ticker)
            dividends = tk.dividends
            if dividends is None or dividends.empty:
                return []

            for idx, amount in dividends.items():
                ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else None
                if ts is None or ts < from_ts or ts > to_ts:
                    continue

                marks.append({
                    "id":             f"div_{ticker}_{ts}",
                    "time":           ts,
                    "color":          "blue",
                    "text":           f"Dividend: ${amount:.4f}",
                    "label":          "D",
                    "labelFontColor": "white",
                    "minSize":        10,
                })
        except Exception as exc:
            logger.debug("dividend_marks_error: %s %s", ticker, exc)

        return marks

    def get_split_marks(self, ticker: str, from_ts: int, to_ts: int) -> list[dict]:
        """Return stock split dates as TradingView marks."""
        marks = []
        try:
            tk = yf.Ticker(ticker)
            splits = tk.splits
            if splits is None or splits.empty:
                return []

            for idx, ratio in splits.items():
                ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else None
                if ts is None or ts < from_ts or ts > to_ts:
                    continue

                marks.append({
                    "id":             f"split_{ticker}_{ts}",
                    "time":           ts,
                    "color":          "orange",
                    "text":           f"Stock Split: {ratio:.0f}:1",
                    "label":          "S",
                    "labelFontColor": "white",
                    "minSize":        12,
                })
        except Exception as exc:
            logger.debug("split_marks_error: %s %s", ticker, exc)

        return marks

    def get_news_marks(self, ticker: str, from_ts: int, to_ts: int) -> list[dict]:
        """Return major news events as TradingView timescale marks."""
        marks = []
        try:
            tk = yf.Ticker(ticker)
            news_items = getattr(tk, "news", None) or []
            for item in news_items[:20]:
                pub_ts = item.get("providerPublishTime") or item.get("publishTime")
                if pub_ts is None:
                    continue
                pub_ts = int(pub_ts)
                if pub_ts < from_ts or pub_ts > to_ts:
                    continue

                title   = (item.get("title") or "")[:80]
                pub_str = datetime.fromtimestamp(pub_ts, tz=_UTC).strftime("%Y-%m-%d")

                marks.append({
                    "id":       f"news_{ticker}_{pub_ts}",
                    "time":     pub_ts,
                    "color":    "red",
                    "label":    "N",
                    "tooltip":  f"[{pub_str}] {title}",
                })
        except Exception as exc:
            logger.debug("news_marks_error: %s %s", ticker, exc)

        return marks


# ---------------------------------------------------------------------------
# TradingViewUDFServer — composes the three classes above
# ---------------------------------------------------------------------------

class TradingViewUDFServer:
    """Facade that composes resolver, bar router, and event mark provider.

    Instantiated once as a module-level singleton; endpoint functions
    delegate to this instance.
    """

    def __init__(self) -> None:
        self.resolver  = SymbolResolver()
        self.bar_router = BarDataRouter()
        self.marks      = EventMarkProvider()

    async def resolve_symbol(self, symbol: str) -> dict:
        """Async wrapper for symbol resolution + info fetch."""
        parsed = self.resolver.resolve(symbol)
        ticker = parsed["ticker"]
        loop   = asyncio.get_event_loop()
        info   = await loop.run_in_executor(None, lambda: self.resolver.get_symbol_info(ticker))
        return info

    async def search(self, query: str, type_filter: str = "", limit: int = 30) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.resolver.search_symbols(query, limit=limit, type_filter=type_filter),
        )

    async def history(
        self, symbol: str, resolution: str, from_ts: int, to_ts: int
    ) -> dict:
        parsed = self.resolver.resolve(symbol)
        return await self.bar_router.get_bars(
            parsed["ticker"], resolution, from_ts, to_ts
        )

    async def get_marks(
        self, symbol: str, from_ts: int, to_ts: int
    ) -> list[dict]:
        parsed = self.resolver.resolve(symbol)
        ticker = parsed["ticker"]
        loop   = asyncio.get_event_loop()

        def _fetch():
            result = []
            result.extend(self.marks.get_earnings_marks(ticker, from_ts, to_ts))
            result.extend(self.marks.get_dividend_marks(ticker, from_ts, to_ts))
            result.extend(self.marks.get_split_marks(ticker, from_ts, to_ts))
            return result

        return await loop.run_in_executor(None, _fetch)

    async def get_timescale_marks(
        self, symbol: str, from_ts: int, to_ts: int
    ) -> list[dict]:
        parsed = self.resolver.resolve(symbol)
        ticker = parsed["ticker"]
        loop   = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.marks.get_news_marks(ticker, from_ts, to_ts),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_udf_server = TradingViewUDFServer()

# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

tradingview_router = APIRouter(tags=["TradingView UDF"])


@tradingview_router.get(
    "/tv/config",
    summary="TradingView UDF configuration",
    response_class=JSONResponse,
)
async def tv_config():
    """TradingView UDF /config endpoint.

    Returns the data feed capabilities: supported resolutions, exchange list,
    symbol types, and feature flags.  TradingView calls this once on init.
    """
    return UDF_CONFIG


@tradingview_router.get(
    "/tv/symbols",
    summary="TradingView symbol metadata (symbol resolution)",
    response_class=JSONResponse,
)
async def tv_symbols(
    symbol: str = Query(..., description="Symbol to resolve, e.g. AAPL or SENTINEL:AAPL"),
):
    """TradingView UDF /symbols endpoint.

    Returns full SymbolInfo for the requested symbol.  Called once per
    symbol before bar data is requested.

    Supports formats: AAPL, SENTINEL:AAPL, AAPL:NASDAQ, BTC-USD.
    """
    try:
        info = await _udf_server.resolve_symbol(symbol)
        return info
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Symbol resolution failed: {exc}")


@tradingview_router.get(
    "/tv/search",
    summary="TradingView symbol search",
    response_class=JSONResponse,
)
async def tv_search(
    query: str = Query(..., description="Search string"),
    type: str = Query("", description="Symbol type filter: stock|crypto|etf|forex|index"),
    exchange: str = Query("", description="Exchange filter (not enforced)"),
    limit: int = Query(30, ge=1, le=50, description="Maximum results"),
):
    """TradingView UDF /search endpoint.

    Returns matching symbols for the query string.  Called as the user
    types in the TradingView symbol search box.
    """
    results = await _udf_server.search(query, type_filter=type, limit=limit)
    return results


@tradingview_router.get(
    "/tv/history",
    summary="TradingView OHLCV bar history",
    response_class=JSONResponse,
)
async def tv_history(
    symbol: str = Query(..., description="Symbol, e.g. AAPL"),
    resolution: str = Query(
        "D",
        description="Bar resolution: 1|3|5|15|30|60|120|240|D|W|M",
    ),
    from_ts: int = Query(..., alias="from", description="Start Unix timestamp (seconds)"),
    to_ts: int = Query(..., alias="to", description="End Unix timestamp (seconds)"),
    countback: Optional[int] = Query(
        None, description="Number of bars to return (overrides from)"
    ),
):
    """TradingView UDF /history endpoint.

    Returns OHLCV bars in UDF format::

        {"s": "ok", "t": [...], "o": [...], "h": [...], "l": [...], "c": [...], "v": [...]}

    Data source routing:
        - Intraday equities → Alpaca IEX (yfinance fallback)
        - Daily equities    → yfinance (Stooq fallback)
        - Crypto intraday   → Binance klines
        - Crypto daily      → CoinGecko OHLC
    """
    if countback is not None and countback > 0:
        # Estimate from_ts from countback
        resolution_minutes = {
            "1": 1, "3": 3, "5": 5, "15": 15, "30": 30,
            "60": 60, "120": 120, "240": 240,
            "D": 1440, "W": 10080, "M": 43200,
        }.get(resolution, 1440)
        from_ts = to_ts - (countback * resolution_minutes * 60) - 86400  # +1 day buffer

    try:
        result = await _udf_server.history(symbol, resolution, from_ts, to_ts)
        return result
    except Exception as exc:
        return {"s": "error", "errmsg": str(exc)}


@tradingview_router.get(
    "/tv/marks",
    summary="TradingView event marks (earnings, dividends, splits)",
    response_class=JSONResponse,
)
async def tv_marks(
    symbol: str = Query(..., description="Symbol"),
    from_ts: int = Query(..., alias="from", description="Start Unix timestamp"),
    to_ts: int = Query(..., alias="to", description="End Unix timestamp"),
    resolution: str = Query("D", description="Current chart resolution"),
):
    """TradingView UDF /marks endpoint.

    Returns earnings, dividend, and split dates as TradingView event marks.
    Marks appear as coloured symbols on the price chart.
    """
    try:
        marks = await _udf_server.get_marks(symbol, from_ts, to_ts)
        return marks
    except Exception as exc:
        logger.error("tv_marks_error: %s", exc)
        return []


@tradingview_router.get(
    "/tv/timescale_marks",
    summary="TradingView timescale marks (news events)",
    response_class=JSONResponse,
)
async def tv_timescale_marks(
    symbol: str = Query(..., description="Symbol"),
    from_ts: int = Query(..., alias="from", description="Start Unix timestamp"),
    to_ts: int = Query(..., alias="to", description="End Unix timestamp"),
    resolution: str = Query("D", description="Current chart resolution"),
):
    """TradingView UDF /timescale_marks endpoint.

    Returns major news events as timescale marks (shown below the time axis).
    """
    try:
        marks = await _udf_server.get_timescale_marks(symbol, from_ts, to_ts)
        return marks
    except Exception as exc:
        logger.error("tv_timescale_marks_error: %s", exc)
        return []


@tradingview_router.get(
    "/tv/time",
    summary="TradingView server time",
    response_class=JSONResponse,
)
async def tv_time():
    """TradingView UDF /time endpoint.

    Returns current server UTC timestamp as an integer (seconds).
    TradingView uses this to detect time-zone mismatches.
    """
    return int(datetime.now(_UTC).timestamp())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "tradingview_router",
    "TradingViewUDFServer",
    "SymbolResolver",
    "BarDataRouter",
    "EventMarkProvider",
    "UDF_CONFIG",
]
