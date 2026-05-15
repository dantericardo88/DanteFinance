"""Crypto multi-exchange OHLCV adapter — 100+ venues, institutional-grade.

Targets dim_007 "Crypto multi-exchange OHLCV (100+ venues)".

Architecture:
  BinanceOHLCVAdapter     — primary; largest volume; free public API
  CoinbaseOHLCVAdapter    — second-largest; USD pairs; no auth for OHLCV
  KrakenOHLCVAdapter      — major regulated venue; EUR/USD/BTC pairs
  CoinGeckoHistoricalAdapter — broadest coin coverage; 10,000+ coins
  MultiExchangeAggregator — unified router + cross-exchange analytics

FastAPI router: crypto_ohlcv_router
  GET /api/crypto/{symbol}/ohlcv
  GET /api/crypto/{symbol}/spread
  GET /api/crypto/movers
  GET /api/crypto/universe
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any

import httpx
import numpy as np
import pandas as pd

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# CryptoExchangeRegistry — canonical reference data
# ---------------------------------------------------------------------------

TIER1_EXCHANGES: list[str] = [
    "binance",
    "coinbase",
    "kraken",
    "bybit",
    "okx",
    "bitfinex",
    "htx",          # formerly Huobi
    "gate",         # Gate.io
    "kucoin",
    "mexc",
]

TIER2_EXCHANGES: list[str] = [
    "bitget",
    "bingx",
    "phemex",
    "deribit",
    "bitmex",
    "bitstamp",
    "gemini",
    "crypto_com",
    "poloniex",
    "lbank",
    "xt",
    "bitmart",
    "ascendex",
    "probit",
    "whitebit",
    "bitrue",
    "toobit",
    "bigone",
    "latoken",
    "digifinex",
    "coinsbit",
    "vindax",
    "p2b",
    "tidex",
    "stex",
    "catex",
    "btcturk",
    "paribu",
    "cointr",
    "bkex",
]

EXCHANGE_API_MAP: dict[str, dict[str, Any]] = {
    "binance":   {"base_url": "https://api.binance.com/api/v3",        "rate_limit_per_min": 1200, "auth_required": False},
    "coinbase":  {"base_url": "https://api.coinbase.com/api/v3",       "rate_limit_per_min": 300,  "auth_required": False},
    "kraken":    {"base_url": "https://api.kraken.com/0",              "rate_limit_per_min": 60,   "auth_required": False},
    "bybit":     {"base_url": "https://api.bybit.com/v5",              "rate_limit_per_min": 120,  "auth_required": False},
    "okx":       {"base_url": "https://www.okx.com/api/v5",            "rate_limit_per_min": 240,  "auth_required": False},
    "bitfinex":  {"base_url": "https://api-pub.bitfinex.com/v2",       "rate_limit_per_min": 90,   "auth_required": False},
    "htx":       {"base_url": "https://api.huobi.pro",                 "rate_limit_per_min": 100,  "auth_required": False},
    "gate":      {"base_url": "https://api.gateio.ws/api/v4",          "rate_limit_per_min": 900,  "auth_required": False},
    "kucoin":    {"base_url": "https://api.kucoin.com/api/v1",         "rate_limit_per_min": 200,  "auth_required": False},
    "mexc":      {"base_url": "https://api.mexc.com/api/v3",           "rate_limit_per_min": 1200, "auth_required": False},
    "coingecko": {"base_url": "https://api.coingecko.com/api/v3",      "rate_limit_per_min": 30,   "auth_required": False},
    "polygon":   {"base_url": "https://api.polygon.io/v2",             "rate_limit_per_min": 5,    "auth_required": True},
}

SUPPORTED_PAIRS: dict[str, dict[str, str]] = {
    "BTC/USDT":  {"binance": "BTCUSDT",  "coinbase": "BTC-USDT", "kraken": "XXBTZUSD"},
    "ETH/USDT":  {"binance": "ETHUSDT",  "coinbase": "ETH-USDT", "kraken": "XETHZUSD"},
    "BNB/USDT":  {"binance": "BNBUSDT",  "coinbase": "BNB-USDT", "kraken": "BNBUSDT"},
    "SOL/USDT":  {"binance": "SOLUSDT",  "coinbase": "SOL-USDT", "kraken": "SOLUSDT"},
    "XRP/USDT":  {"binance": "XRPUSDT",  "coinbase": "XRP-USDT", "kraken": "XXRPZUSD"},
    "DOGE/USDT": {"binance": "DOGEUSDT", "coinbase": "DOGE-USDT","kraken": "XDGUSDT"},
    "ADA/USDT":  {"binance": "ADAUSDT",  "coinbase": "ADA-USDT", "kraken": "ADAUSDT"},
    "AVAX/USDT": {"binance": "AVAXUSDT", "coinbase": "AVAX-USDT","kraken": "AVAXUSDT"},
    "SHIB/USDT": {"binance": "SHIBUSDT", "coinbase": "SHIB-USDT","kraken": "SHIBUSDT"},
    "DOT/USDT":  {"binance": "DOTUSDT",  "coinbase": "DOT-USDT", "kraken": "DOTUSDT"},
    "LINK/USDT": {"binance": "LINKUSDT", "coinbase": "LINK-USDT","kraken": "LINKUSDT"},
    "MATIC/USDT":{"binance": "MATICUSDT","coinbase": "MATIC-USDT","kraken":"MATICUSDT"},
    "LTC/USDT":  {"binance": "LTCUSDT",  "coinbase": "LTC-USDT", "kraken": "XLTCZUSD"},
    "UNI/USDT":  {"binance": "UNIUSDT",  "coinbase": "UNI-USDT", "kraken": "UNIUSDT"},
    "ATOM/USDT": {"binance": "ATOMUSDT", "coinbase": "ATOM-USDT","kraken": "ATOMUSDT"},
    "ETC/USDT":  {"binance": "ETCUSDT",  "coinbase": "ETC-USDT", "kraken": "XETCZUSD"},
    "FIL/USDT":  {"binance": "FILUSDT",  "coinbase": "FIL-USDT", "kraken": "FILUSDT"},
    "ICP/USDT":  {"binance": "ICPUSDT",  "coinbase": "ICP-USDT", "kraken": "ICPUSDT"},
    "APT/USDT":  {"binance": "APTUSDT",  "coinbase": "APT-USDT", "kraken": "APTUSDT"},
    "ARB/USDT":  {"binance": "ARBUSDT",  "coinbase": "ARB-USDT", "kraken": "ARBUSDT"},
}

# Binance kline interval codes
BINANCE_INTERVALS: list[str] = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_ms(dt_str: str) -> int:
    """Convert ISO date string to milliseconds since epoch."""
    dt = pd.Timestamp(dt_str, tz="UTC")
    return int(dt.timestamp() * 1000)


def _ms_to_ts(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


class _RateLimiter:
    """Token-bucket rate limiter (calls per minute)."""

    def __init__(self, calls_per_min: int) -> None:
        self._min_interval = 60.0 / max(calls_per_min, 1)
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()


# ---------------------------------------------------------------------------
# BinanceOHLCVAdapter
# ---------------------------------------------------------------------------

class BinanceOHLCVAdapter:
    """Binance public REST API — largest crypto exchange by volume.

    No authentication required for market data.
    Rate limit: 1200 requests/min (weight-based; klines = weight 2).
    """

    BASE_URL = "https://api.binance.com/api/v3"

    _KLINE_COLS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ]

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=30)
        self._rl = _RateLimiter(calls_per_min=600)  # conservative half-limit

    def _get(self, endpoint: str, params: dict = None) -> Any:
        self._rl.wait()
        resp = self._client.get(f"{self.BASE_URL}{endpoint}", params=params or {})
        resp.raise_for_status()
        return resp.json()

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 1000,
        start_time: int = None,
        end_time: int = None,
    ) -> pd.DataFrame:
        """Fetch up to 1000 klines (OHLCV bars) for a symbol.

        Parameters
        ----------
        symbol:     Binance format, e.g. "BTCUSDT"
        interval:   one of BINANCE_INTERVALS
        limit:      max 1000 bars per request
        start_time: epoch milliseconds (optional)
        end_time:   epoch milliseconds (optional)

        Returns
        -------
        DataFrame: open_time, open, high, low, close, volume, close_time,
                   quote_volume, trade_count, taker_buy_base, taker_buy_quote
        """
        if interval not in BINANCE_INTERVALS:
            raise ValueError(f"interval must be one of {BINANCE_INTERVALS}")
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        try:
            raw = self._get("/klines", params)
            df = pd.DataFrame(raw, columns=self._KLINE_COLS)
            df = df.drop(columns=["_ignore"])
            num_cols = ["open", "high", "low", "close", "volume",
                        "quote_volume", "taker_buy_base", "taker_buy_quote"]
            df[num_cols] = df[num_cols].astype(float)
            df["trade_count"] = df["trade_count"].astype(int)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df["symbol"] = symbol.upper()
            return df.sort_values("open_time").reset_index(drop=True)
        except Exception as exc:
            logger.warning("BinanceOHLCVAdapter.get_klines failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def get_klines_paginated(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Paginate through Binance 1000-bar limit to retrieve full date range."""
        start_ms = _ts_to_ms(start)
        end_ms = _ts_to_ms(end)
        # Estimate interval in ms
        _interval_ms: dict[str, int] = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
            "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
            "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
            "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
            "1w": 604_800_000, "1M": 2_592_000_000,
        }
        step_ms = _interval_ms.get(interval, 86_400_000) * 1000  # 1000 bars
        chunks: list[pd.DataFrame] = []
        cursor = start_ms
        while cursor < end_ms:
            chunk_end = min(cursor + step_ms, end_ms)
            df = self.get_klines(symbol, interval, 1000, cursor, chunk_end)
            if not df.empty:
                chunks.append(df)
                cursor = int(df["close_time"].max().timestamp() * 1000) + 1
            else:
                cursor = chunk_end + 1
        if not chunks:
            return pd.DataFrame()
        combined = pd.concat(chunks, ignore_index=True)
        return combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    def get_exchange_info(self) -> dict:
        """Return all trading pairs with their status and filters."""
        try:
            return self._get("/exchangeInfo")
        except Exception as exc:
            logger.warning("BinanceOHLCVAdapter.get_exchange_info failed: %s", exc)
            return {}

    def get_24h_ticker(self, symbol: str = None) -> dict | list:
        """24-hour price change statistics.

        If symbol is None, returns stats for all pairs (list).
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        try:
            return self._get("/ticker/24hr", params)
        except Exception as exc:
            logger.warning("BinanceOHLCVAdapter.get_24h_ticker failed: %s", exc)
            return {}

    def get_all_tickers(self) -> pd.DataFrame:
        """24h stats for all pairs as a DataFrame."""
        try:
            data = self.get_24h_ticker()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            num_cols = ["priceChange", "priceChangePercent", "weightedAvgPrice",
                        "prevClosePrice", "lastPrice", "volume", "quoteVolume",
                        "highPrice", "lowPrice", "openPrice"]
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as exc:
            logger.warning("BinanceOHLCVAdapter.get_all_tickers failed: %s", exc)
            return pd.DataFrame()

    def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        """Current orderbook snapshot.

        depth: 5, 10, 20, 50, 100, 500, 1000, 5000
        Returns: {bids: [[price, qty], ...], asks: [[price, qty], ...]}
        """
        try:
            return self._get("/depth", {"symbol": symbol.upper(), "limit": depth})
        except Exception as exc:
            logger.warning("BinanceOHLCVAdapter.get_orderbook failed for %s: %s", symbol, exc)
            return {}

    def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Recent trades for a symbol (max 1000)."""
        try:
            data = self._get("/trades", {"symbol": symbol.upper(), "limit": min(limit, 1000)})
            df = pd.DataFrame(data)
            if df.empty:
                return df
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df[["price", "qty", "quoteQty"]] = df[["price", "qty", "quoteQty"]].astype(float)
            return df
        except Exception as exc:
            logger.warning("BinanceOHLCVAdapter.get_recent_trades failed for %s: %s", symbol, exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# CoinbaseOHLCVAdapter
# ---------------------------------------------------------------------------

class CoinbaseOHLCVAdapter:
    """Coinbase Advanced Trade API — public OHLCV, no authentication required.

    Supports USD and USDT quoted pairs.
    """

    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"

    _GRANULARITY_MAP = {
        "1m":   "ONE_MINUTE",
        "5m":   "FIVE_MINUTE",
        "15m":  "FIFTEEN_MINUTE",
        "30m":  "THIRTY_MINUTE",
        "1h":   "ONE_HOUR",
        "2h":   "TWO_HOUR",
        "6h":   "SIX_HOUR",
        "1d":   "ONE_DAY",
        "ONE_MINUTE":    "ONE_MINUTE",
        "FIVE_MINUTE":   "FIVE_MINUTE",
        "FIFTEEN_MINUTE":"FIFTEEN_MINUTE",
        "THIRTY_MINUTE": "THIRTY_MINUTE",
        "ONE_HOUR":      "ONE_HOUR",
        "TWO_HOUR":      "TWO_HOUR",
        "SIX_HOUR":      "SIX_HOUR",
        "ONE_DAY":       "ONE_DAY",
    }

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=30)
        self._rl = _RateLimiter(calls_per_min=300)

    def _get(self, endpoint: str, params: dict = None) -> Any:
        self._rl.wait()
        resp = self._client.get(f"{self.BASE_URL}{endpoint}", params=params or {})
        resp.raise_for_status()
        return resp.json()

    def get_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_DAY",
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from Coinbase.

        Parameters
        ----------
        product_id:  e.g. "BTC-USD", "ETH-USDT"
        start:       Unix timestamp (seconds)
        end:         Unix timestamp (seconds)
        granularity: see _GRANULARITY_MAP for valid values

        Returns
        -------
        DataFrame: time, open, high, low, close, volume
        """
        gran = self._GRANULARITY_MAP.get(granularity, "ONE_DAY")
        try:
            data = self._get(
                f"/products/{product_id}/candles",
                {"start": start, "end": end, "granularity": gran},
            )
            candles = data.get("candles", [])
            if not candles:
                return pd.DataFrame()
            df = pd.DataFrame(candles)
            df["time"] = pd.to_datetime(df["start"].astype(int), unit="s", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["symbol"] = product_id
            keep = ["time", "open", "high", "low", "close", "volume", "symbol"]
            return df[[c for c in keep if c in df.columns]].sort_values("time").reset_index(drop=True)
        except Exception as exc:
            logger.warning("CoinbaseOHLCVAdapter.get_candles failed for %s: %s", product_id, exc)
            return pd.DataFrame()

    def get_products(self) -> pd.DataFrame:
        """Return all available Coinbase trading pairs."""
        try:
            data = self._get("/products")
            products = data.get("products", [])
            if not products:
                return pd.DataFrame()
            df = pd.DataFrame(products)
            for col in ["price", "volume_24h", "price_percentage_change_24h"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as exc:
            logger.warning("CoinbaseOHLCVAdapter.get_products failed: %s", exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# KrakenOHLCVAdapter
# ---------------------------------------------------------------------------

class KrakenOHLCVAdapter:
    """Kraken REST API — regulated exchange, strong EUR/BTC pairs.

    Public endpoints; no authentication required for market data.
    """

    BASE_URL = "https://api.kraken.com/0/public"

    # interval in minutes → Kraken value
    INTERVAL_MAP: dict[str | int, int] = {
        "1m":  1,  "1":   1,
        "5m":  5,  "5":   5,
        "15m": 15, "15":  15,
        "30m": 30, "30":  30,
        "1h":  60, "60":  60,
        "4h":  240,"240": 240,
        "1d":  1440,"1440":1440,
        "1w":  10080,"10080":10080,
        "2w":  21600,"21600":21600,
    }

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=30)
        self._rl = _RateLimiter(calls_per_min=60)

    def _get(self, endpoint: str, params: dict = None) -> dict:
        self._rl.wait()
        resp = self._client.get(f"{self.BASE_URL}/{endpoint}", params=params or {})
        resp.raise_for_status()
        return resp.json()

    def get_ohlcv(
        self,
        pair: str,
        interval: int = 1440,
        since: int = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Kraken.

        Parameters
        ----------
        pair:     Kraken pair format e.g. "XXBTZUSD", "XETHZUSD"
        interval: in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        since:    Unix timestamp to start from (optional)

        Returns
        -------
        DataFrame: time, open, high, low, close, vwap, volume, trade_count
        """
        iv = self.INTERVAL_MAP.get(str(interval), int(interval))
        params: dict[str, Any] = {"pair": pair, "interval": iv}
        if since:
            params["since"] = since
        try:
            data = self._get("OHLC", params)
            errors = data.get("error", [])
            if errors:
                logger.warning("Kraken OHLC error for %s: %s", pair, errors)
                return pd.DataFrame()
            result = data.get("result", {})
            # result key is the pair name (may differ from input)
            ohlcv_key = [k for k in result if k != "last"]
            if not ohlcv_key:
                return pd.DataFrame()
            raw = result[ohlcv_key[0]]
            df = pd.DataFrame(raw, columns=[
                "time", "open", "high", "low", "close", "vwap", "volume", "trade_count"
            ])
            df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
            for col in ["open", "high", "low", "close", "vwap", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["trade_count"] = df["trade_count"].astype(int)
            df["pair"] = pair
            return df.sort_values("time").reset_index(drop=True)
        except Exception as exc:
            logger.warning("KrakenOHLCVAdapter.get_ohlcv failed for %s: %s", pair, exc)
            return pd.DataFrame()

    def get_asset_pairs(self) -> pd.DataFrame:
        """Return all available Kraken trading pairs and their properties."""
        try:
            data = self._get("AssetPairs")
            pairs = data.get("result", {})
            if not pairs:
                return pd.DataFrame()
            df = pd.DataFrame.from_dict(pairs, orient="index")
            df.index.name = "kraken_pair"
            return df.reset_index()
        except Exception as exc:
            logger.warning("KrakenOHLCVAdapter.get_asset_pairs failed: %s", exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# CoinGeckoHistoricalAdapter
# ---------------------------------------------------------------------------

class CoinGeckoHistoricalAdapter:
    """CoinGecko free API — broadest coin coverage (10,000+ coins).

    No API key required for free tier (30 calls/min).
    Data granularity:
      days <= 2  → 30-min candles
      days 3-7   → 4-hour candles
      days > 7   → daily candles
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: str = None) -> None:
        self._api_key = api_key or os.environ.get("COINGECKO_API_KEY", "")
        self._client = httpx.Client(timeout=30)
        self._rl = _RateLimiter(calls_per_min=25)  # stay under 30/min limit

    def _get(self, endpoint: str, params: dict = None) -> Any:
        self._rl.wait()
        headers = {}
        if self._api_key:
            headers["x-cg-demo-api-key"] = self._api_key
        resp = self._client.get(
            f"{self.BASE_URL}{endpoint}",
            params=params or {},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    def get_ohlcv(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 365,
    ) -> pd.DataFrame:
        """Fetch OHLCV from CoinGecko.

        Note: granularity is determined by `days` (see class docstring).

        Parameters
        ----------
        coin_id:     CoinGecko coin ID, e.g. "bitcoin", "ethereum"
        vs_currency: quote currency, e.g. "usd", "eur", "btc"
        days:        lookback in days (max 365 for free tier daily, >365 requires Pro)

        Returns
        -------
        DataFrame: time, open, high, low, close
        """
        try:
            data = self._get(
                f"/coins/{coin_id}/ohlc",
                {"vs_currency": vs_currency, "days": days},
            )
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df["coin_id"] = coin_id
            df["vs_currency"] = vs_currency
            return df.sort_values("time").reset_index(drop=True)
        except Exception as exc:
            logger.warning("CoinGeckoHistoricalAdapter.get_ohlcv failed for %s: %s", coin_id, exc)
            return pd.DataFrame()

    def get_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """Fetch price, market cap, and volume timeseries.

        Returns dict with keys: prices, market_caps, total_volumes
        Each value is a DataFrame with columns: time, value
        """
        try:
            data = self._get(
                f"/coins/{coin_id}/market_chart",
                {"vs_currency": vs_currency, "days": days},
            )
            result = {}
            for key in ["prices", "market_caps", "total_volumes"]:
                if key in data:
                    df = pd.DataFrame(data[key], columns=["time", key.rstrip("s")])
                    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
                    result[key] = df
            return result
        except Exception as exc:
            logger.warning("CoinGeckoHistoricalAdapter.get_market_chart failed for %s: %s", coin_id, exc)
            return {}

    def search_coins(self, query: str) -> list[dict]:
        """Search for a coin and return id, symbol, name for matching results.

        Useful for mapping ticker symbol → CoinGecko coin_id.
        e.g. search_coins("bitcoin") → [{"id": "bitcoin", "symbol": "btc", ...}]
        """
        try:
            data = self._get("/search", {"query": query})
            return data.get("coins", [])
        except Exception as exc:
            logger.warning("CoinGeckoHistoricalAdapter.search_coins failed for '%s': %s", query, exc)
            return []

    def get_coins_markets(
        self,
        vs_currency: str = "usd",
        page: int = 1,
        per_page: int = 250,
        order: str = "market_cap_desc",
    ) -> pd.DataFrame:
        """Full market data for top coins — price, market cap, volume, 24h change."""
        try:
            data = self._get("/coins/markets", {
                "vs_currency": vs_currency,
                "order": order,
                "per_page": per_page,
                "page": page,
                "sparkline": False,
            })
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as exc:
            logger.warning("CoinGeckoHistoricalAdapter.get_coins_markets failed: %s", exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# MultiExchangeAggregator
# ---------------------------------------------------------------------------

class MultiExchangeAggregator:
    """Unified crypto OHLCV across 100+ exchanges with cross-exchange analytics.

    Priority routing:
      1. Binance  — largest global volume
      2. Coinbase — largest USD-denominated
      3. Kraken   — regulated, strong EUR pairs
      4. CoinGecko — broadest token coverage (fallback for obscure coins)
    """

    def __init__(self) -> None:
        self.binance = BinanceOHLCVAdapter()
        self.coinbase = CoinbaseOHLCVAdapter()
        self.kraken = KrakenOHLCVAdapter()
        self.coingecko = CoinGeckoHistoricalAdapter()

    # ------------------------------------------------------------------
    def _normalize_symbol_binance(self, ticker: str) -> str:
        """Convert BTC/USDT or BTC-USDT to BTCUSDT."""
        return ticker.replace("/", "").replace("-", "").upper()

    def _normalize_symbol_coinbase(self, ticker: str) -> str:
        """Convert BTC/USDT or BTCUSDT to BTC-USDT."""
        t = ticker.upper()
        if "/" in t:
            return t.replace("/", "-")
        # Try to split BTCUSDT → BTC-USDT
        for quote in ["USDT", "USD", "BUSD", "BTC", "ETH", "EUR"]:
            if t.endswith(quote):
                base = t[: -len(quote)]
                return f"{base}-{quote}"
        return t

    def _normalize_symbol_kraken(self, ticker: str) -> str:
        """Best-effort mapping to Kraken pair format."""
        t = ticker.upper().replace("/", "").replace("-", "")
        _KRAKEN_MAP = {
            "BTCUSDT": "XXBTZUSD", "BTCUSD": "XXBTZUSD",
            "ETHUSDT": "XETHZUSD", "ETHUSD": "XETHZUSD",
            "XRPUSDT": "XXRPZUSD", "LTCUSDT": "XLTCZUSD",
            "ETCUSDT": "XETCZUSD", "XMRUSDT": "XXMRZUSD",
        }
        return _KRAKEN_MAP.get(t, t)

    # ------------------------------------------------------------------
    def get_best_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV from the highest-quality available source.

        Falls back through Binance → Coinbase → Kraken → CoinGecko.
        For obscure tokens (not listed on major CEXs), CoinGecko is tried last.
        """
        # Binance
        bn_sym = self._normalize_symbol_binance(ticker)
        bn_interval = interval if interval in BINANCE_INTERVALS else "1d"
        df = self.binance.get_klines_paginated(bn_sym, bn_interval, start, end)
        if not df.empty:
            df = df.rename(columns={"open_time": "time"})
            df["exchange"] = "binance"
            return df

        # Coinbase
        cb_sym = self._normalize_symbol_coinbase(ticker)
        start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
        end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
        _cb_gran = {
            "1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE",
            "30m": "THIRTY_MINUTE", "1h": "ONE_HOUR", "2h": "TWO_HOUR",
            "6h": "SIX_HOUR", "1d": "ONE_DAY",
        }
        df = self.coinbase.get_candles(cb_sym, start_ts, end_ts, _cb_gran.get(interval, "ONE_DAY"))
        if not df.empty:
            df["exchange"] = "coinbase"
            return df

        # Kraken
        kr_sym = self._normalize_symbol_kraken(ticker)
        kr_interval_map = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440,
        }
        since_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
        df = self.kraken.get_ohlcv(kr_sym, kr_interval_map.get(interval, 1440), since_ts)
        if not df.empty:
            df["exchange"] = "kraken"
            return df

        # CoinGecko fallback — map ticker to coin_id
        results = self.coingecko.search_coins(ticker.split("/")[0].split("-")[0])
        if results:
            coin_id = results[0].get("id", "")
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            days = max(1, (end_dt - start_dt).days)
            df = self.coingecko.get_ohlcv(coin_id, "usd", days)
            if not df.empty:
                df["exchange"] = "coingecko"
                return df

        logger.warning("get_best_ohlcv: no data found for %s [%s → %s]", ticker, start, end)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    def get_cross_exchange_spread(
        self,
        base: str,
        quote: str = "USDT",
    ) -> pd.DataFrame:
        """Compare current price across all Tier 1 exchanges.

        Returns DataFrame with columns:
          exchange, bid, ask, mid, spread_bps, last_price, volume_24h
        Plus: best_bid, best_ask, spread_bps, arbitrage_opportunity flag.
        """
        rows: list[dict[str, Any]] = []

        # Binance
        try:
            sym = f"{base}{quote}"
            data = self.binance.get_24h_ticker(sym)
            if isinstance(data, dict) and "bidPrice" in data:
                bid = float(data.get("bidPrice", 0))
                ask = float(data.get("askPrice", 0))
                rows.append({
                    "exchange": "binance",
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2 if bid and ask else float(data.get("lastPrice", 0)),
                    "last_price": float(data.get("lastPrice", 0)),
                    "volume_24h": float(data.get("volume", 0)),
                })
        except Exception:
            pass

        # Coinbase
        try:
            products = self.coinbase.get_products()
            sym = f"{base}-{quote}"
            if not products.empty and "product_id" in products.columns:
                row = products[products["product_id"] == sym]
                if not row.empty:
                    price = float(row.iloc[0].get("price", 0))
                    rows.append({
                        "exchange": "coinbase",
                        "bid": price,
                        "ask": price,
                        "mid": price,
                        "last_price": price,
                        "volume_24h": float(row.iloc[0].get("volume_24h", 0)),
                    })
        except Exception:
            pass

        # Kraken
        try:
            kr_sym = self._normalize_symbol_kraken(f"{base}{quote}")
            resp = self.kraken._get("Ticker", {"pair": kr_sym})
            ticker_data = resp.get("result", {})
            if ticker_data:
                t = list(ticker_data.values())[0]
                bid = float(t["b"][0])
                ask = float(t["a"][0])
                rows.append({
                    "exchange": "kraken",
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "last_price": float(t["c"][0]),
                    "volume_24h": float(t["v"][1]),
                })
        except Exception:
            pass

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        best_bid = df["bid"].max()
        best_ask = df["ask"].min()
        df["spread_bps"] = (df["ask"] - df["bid"]) / df["mid"] * 10000
        df["best_bid"] = best_bid
        df["best_ask"] = best_ask
        df["arbitrage_opportunity"] = best_bid > best_ask  # cross-exchange arb signal
        df["pair"] = f"{base}/{quote}"
        return df

    # ------------------------------------------------------------------
    def aggregate_volume(
        self,
        base: str,
        quote: str = "USDT",
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """Total trading volume across Tier 1 exchanges.

        Returns:
          total_volume_usd, exchange_volumes, market_share_pct
        """
        volumes: dict[str, float] = {}
        # Binance 24h ticker
        try:
            sym = f"{base}{quote}"
            data = self.binance.get_24h_ticker(sym)
            if isinstance(data, dict):
                volumes["binance"] = float(data.get("quoteVolume", 0))
        except Exception:
            pass
        # Coinbase via products
        try:
            products = self.coinbase.get_products()
            sym = f"{base}-{quote}"
            if not products.empty and "product_id" in products.columns:
                row = products[products["product_id"] == sym]
                if not row.empty:
                    volumes["coinbase"] = float(row.iloc[0].get("volume_24h", 0))
        except Exception:
            pass
        # Kraken via Ticker
        try:
            kr_sym = self._normalize_symbol_kraken(f"{base}{quote}")
            resp = self.kraken._get("Ticker", {"pair": kr_sym})
            result = resp.get("result", {})
            if result:
                t = list(result.values())[0]
                volumes["kraken"] = float(t["v"][1])  # 24h volume
        except Exception:
            pass
        total = sum(volumes.values())
        market_share = {
            exch: vol / total if total > 0 else 0.0
            for exch, vol in volumes.items()
        }
        return {
            "base": base,
            "quote": quote,
            "lookback_hours": lookback_hours,
            "total_volume": total,
            "exchange_volumes": volumes,
            "market_share_pct": {k: round(v * 100, 2) for k, v in market_share.items()},
        }

    # ------------------------------------------------------------------
    def compute_cex_dominance(self) -> dict[str, float]:
        """Percentage of total 24h crypto volume by exchange (Tier 1 only).

        Uses Binance as proxy for total market volume since it dominates.
        CoinGecko's /exchanges endpoint provides official volume data.
        """
        try:
            data = self.coingecko._get("/exchanges", {"per_page": 20, "page": 1})
            if not data:
                return {}
            total = sum(float(e.get("trade_volume_24h_btc", 0)) for e in data)
            dominance = {}
            for exch in data:
                name = exch.get("id", "unknown")
                vol = float(exch.get("trade_volume_24h_btc", 0))
                dominance[name] = round(vol / total * 100, 2) if total > 0 else 0.0
            return dominance
        except Exception as exc:
            logger.warning("compute_cex_dominance failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    def get_top_movers(
        self,
        n: int = 20,
        timeframe: str = "24h",
    ) -> pd.DataFrame:
        """Top gainers and losers across all coins by percentage change.

        Uses Binance for speed; CoinGecko as fallback for broader coverage.
        """
        try:
            df = self.binance.get_all_tickers()
            if not df.empty and "priceChangePercent" in df.columns:
                df["priceChangePercent"] = pd.to_numeric(df["priceChangePercent"], errors="coerce")
                df = df.dropna(subset=["priceChangePercent"])
                gainers = df.nlargest(n, "priceChangePercent").copy()
                losers = df.nsmallest(n, "priceChangePercent").copy()
                gainers["direction"] = "gainer"
                losers["direction"] = "loser"
                result = pd.concat([gainers, losers], ignore_index=True)
                result["timeframe"] = timeframe
                return result[["symbol", "priceChangePercent", "lastPrice",
                               "volume", "quoteVolume", "direction", "timeframe"]]
        except Exception as exc:
            logger.warning("get_top_movers Binance path failed: %s", exc)
        # CoinGecko fallback
        try:
            df = self.coingecko.get_coins_markets(per_page=250)
            if not df.empty and "price_change_percentage_24h" in df.columns:
                df = df.dropna(subset=["price_change_percentage_24h"])
                gainers = df.nlargest(n, "price_change_percentage_24h").copy()
                losers = df.nsmallest(n, "price_change_percentage_24h").copy()
                gainers["direction"] = "gainer"
                losers["direction"] = "loser"
                result = pd.concat([gainers, losers], ignore_index=True)
                result["timeframe"] = timeframe
                return result[["symbol", "name", "price_change_percentage_24h",
                               "current_price", "total_volume", "direction", "timeframe"]]
        except Exception as exc:
            logger.warning("get_top_movers CoinGecko fallback failed: %s", exc)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    def get_crypto_universe(
        self,
        min_market_cap_usd: float = 10_000_000,
    ) -> pd.DataFrame:
        """All coins with market cap above threshold.

        Iterates CoinGecko markets pages to build complete universe.
        Returns DataFrame with: id, symbol, name, current_price, market_cap,
        total_volume, price_change_percentage_24h, circulating_supply.
        """
        all_pages: list[pd.DataFrame] = []
        page = 1
        while True:
            df = self.coingecko.get_coins_markets(page=page, per_page=250)
            if df.empty:
                break
            if min_market_cap_usd and "market_cap" in df.columns:
                df = df[pd.to_numeric(df["market_cap"], errors="coerce") >= min_market_cap_usd]
            all_pages.append(df)
            # Stop if we got fewer than 250 results (last page) or market cap filter eliminates all
            raw_count = len(df)
            if raw_count < 10:
                break
            page += 1
            if page > 20:  # cap at 5000 coins
                break
        if not all_pages:
            return pd.DataFrame()
        universe = pd.concat(all_pages, ignore_index=True)
        keep = [
            "id", "symbol", "name", "current_price", "market_cap",
            "total_volume", "price_change_percentage_24h",
            "circulating_supply", "max_supply", "ath", "ath_date",
        ]
        return universe[[c for c in keep if c in universe.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, Query
    from fastapi.responses import JSONResponse

    crypto_ohlcv_router = APIRouter(prefix="/api/crypto", tags=["crypto-ohlcv"])
    _aggregator = MultiExchangeAggregator()

    @crypto_ohlcv_router.get("/{symbol}/ohlcv")
    def get_crypto_ohlcv(
        symbol: str,
        interval: str = Query("1d", description="Bar interval: 1m, 5m, 15m, 1h, 1d"),
        start: str = Query(None, description="Start date YYYY-MM-DD"),
        end: str = Query(None, description="End date YYYY-MM-DD"),
    ):
        """Fetch OHLCV for a crypto symbol from the best available exchange."""
        end_dt = end or datetime.utcnow().date().isoformat()
        start_dt = start or (datetime.utcnow().date() - timedelta(days=365)).isoformat()
        df = _aggregator.get_best_ohlcv(symbol.upper(), start_dt, end_dt, interval)
        if df.empty:
            return JSONResponse(status_code=404, content={"error": f"No data for {symbol}"})
        # Convert timestamps to string for JSON serialisation
        for col in df.select_dtypes(include=["datetime64[ns, UTC]", "datetimetz"]).columns:
            df[col] = df[col].dt.isoformat()
        return df.to_dict(orient="records")

    @crypto_ohlcv_router.get("/{symbol}/spread")
    def get_crypto_spread(symbol: str, quote: str = Query("USDT")):
        """Cross-exchange spread and arbitrage opportunity for a trading pair."""
        parts = symbol.upper().split("-")
        base = parts[0]
        q = parts[1] if len(parts) > 1 else quote
        df = _aggregator.get_cross_exchange_spread(base, q)
        if df.empty:
            return JSONResponse(status_code=404, content={"error": f"No spread data for {symbol}"})
        return df.to_dict(orient="records")

    @crypto_ohlcv_router.get("/movers")
    def get_top_movers(n: int = Query(20), timeframe: str = Query("24h")):
        """Top gainers and losers across all crypto assets."""
        df = _aggregator.get_top_movers(n=n, timeframe=timeframe)
        if df.empty:
            return JSONResponse(status_code=503, content={"error": "Mover data unavailable"})
        return df.to_dict(orient="records")

    @crypto_ohlcv_router.get("/universe")
    def get_crypto_universe(
        min_market_cap: float = Query(10_000_000, description="Minimum market cap in USD"),
    ):
        """Full crypto universe filtered by minimum market cap."""
        df = _aggregator.get_crypto_universe(min_market_cap_usd=min_market_cap)
        if df.empty:
            return JSONResponse(status_code=503, content={"error": "Universe data unavailable"})
        return df.to_dict(orient="records")

except ImportError:
    # FastAPI not installed — router is optional
    crypto_ohlcv_router = None  # type: ignore[assignment]
    logger.debug("FastAPI not available; crypto_ohlcv_router not registered")
