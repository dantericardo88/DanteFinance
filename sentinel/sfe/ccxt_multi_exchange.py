"""
Multi-exchange crypto data and execution via CCXT library.
Covers: 20+ exchanges, OHLCV, order books, trade history, unified execution API.
Also includes free REST fallbacks when CCXT is not installed.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)

try:
    import ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    ccxt = None  # type: ignore[assignment]
    _CCXT_AVAILABLE = False

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants / free REST fallback endpoints
# ---------------------------------------------------------------------------

_BINANCE_REST = "https://api.binance.com/api/v3"
_COINBASE_REST = "https://api.exchange.coinbase.com"
_KRAKEN_REST = "https://api.kraken.com/0/public"
_BITFINEX_REST = "https://api-pub.bitfinex.com/v2"
_BYBIT_REST = "https://api.bybit.com/v5"

_HEADERS = {"User-Agent": "SENTINEL-financial-terminal/2.0", "Accept": "application/json"}
_TIMEOUT = 20.0
_CACHE_TTL = 30  # 30-second cache for live order-book data

_TF_MAP_BINANCE = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
}

_DB_PATH = Path(__file__).parent.parent / "data" / "ccxt_exchange.db"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def _ensure_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                exchange    TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                timeframe   TEXT NOT NULL,
                ts          INTEGER NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                PRIMARY KEY (exchange, symbol, timeframe, ts)
            );

            CREATE TABLE IF NOT EXISTS paper_orders (
                order_id    TEXT PRIMARY KEY,
                exchange    TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                side        TEXT NOT NULL,
                order_type  TEXT NOT NULL,
                amount      REAL NOT NULL,
                price       REAL,
                status      TEXT NOT NULL DEFAULT 'open',
                filled      REAL DEFAULT 0.0,
                avg_fill    REAL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS arb_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT NOT NULL,
                buy_exchange    TEXT NOT NULL,
                sell_exchange   TEXT NOT NULL,
                profit_pct      REAL NOT NULL,
                buy_price       REAL NOT NULL,
                sell_price      REAL NOT NULL,
                detected_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exchange_health (
                exchange        TEXT NOT NULL,
                checked_at      TEXT NOT NULL,
                latency_ms      REAL,
                status          TEXT,
                PRIMARY KEY (exchange, checked_at)
            );
        """)


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# TTL in-memory cache
# ---------------------------------------------------------------------------

_mem_cache: Dict[str, Tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    entry = _mem_cache.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.monotonic() - ts > _CACHE_TTL:
        del _mem_cache[key]
        return None
    return val


def _cache_set(key: str, val: Any) -> None:
    _mem_cache[key] = (time.monotonic(), val)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ExchangeInfo(BaseModel):
    exchange_id: str
    has_spot: bool = True
    has_futures: bool = False
    has_options: bool = False
    has_margin: bool = False
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    min_order_size_usd: float = 1.0
    supported_fiats: List[str] = Field(default_factory=list)
    countries_blocked: List[str] = Field(default_factory=list)
    rest_url: str = ""


class OHLCVBar(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str = ""


class OrderBookLevel(BaseModel):
    price: float
    size: float


class ConsolidatedOrderBook(BaseModel):
    symbol: str
    timestamp: str
    best_bid: float
    best_ask: float
    spread: float
    spread_pct: float
    imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    bids: List[OrderBookLevel] = Field(default_factory=list)
    asks: List[OrderBookLevel] = Field(default_factory=list)
    exchange_spreads: Dict[str, float] = Field(default_factory=dict)


class ArbitrageOpportunity(BaseModel):
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    gross_pct: float
    fee_pct: float
    net_pct: float
    transfer_time_min: int
    actionable: bool
    detected_at: str


class OrderResult(BaseModel):
    order_id: str
    exchange: str
    symbol: str
    side: str
    order_type: str
    amount: float
    price: Optional[float]
    status: str
    filled: float = 0.0
    avg_fill: Optional[float] = None
    paper: bool = False
    created_at: str


class PositionInfo(BaseModel):
    exchange: str
    symbol: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float = 1.0


class ExchangeHealthStatus(BaseModel):
    exchange: str
    latency_ms: Optional[float]
    status: str  # "ok" | "slow" | "down" | "maintenance"
    checked_at: str
    rate_limit_remaining: Optional[int] = None


# ---------------------------------------------------------------------------
# 1. CCXTExchangeRegistry
# ---------------------------------------------------------------------------


class CCXTExchangeRegistry:
    """Registry of supported exchanges and their capabilities."""

    SUPPORTED_EXCHANGES: Dict[str, Dict[str, Any]] = {
        "binance": {
            "has_spot": True, "has_futures": True, "has_options": False, "has_margin": True,
            "maker_fee": 0.001, "taker_fee": 0.001, "min_order_size_usd": 10.0,
            "supported_fiats": ["USD", "EUR", "GBP", "BRL", "AUD"],
            "countries_blocked": ["US"],
            "rest_url": _BINANCE_REST,
        },
        "coinbase": {
            "has_spot": True, "has_futures": False, "has_options": False, "has_margin": False,
            "maker_fee": 0.004, "taker_fee": 0.006, "min_order_size_usd": 1.0,
            "supported_fiats": ["USD", "EUR", "GBP"],
            "countries_blocked": [],
            "rest_url": _COINBASE_REST,
        },
        "kraken": {
            "has_spot": True, "has_futures": True, "has_options": False, "has_margin": True,
            "maker_fee": 0.0016, "taker_fee": 0.0026, "min_order_size_usd": 1.0,
            "supported_fiats": ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"],
            "countries_blocked": [],
            "rest_url": _KRAKEN_REST,
        },
        "bitfinex": {
            "has_spot": True, "has_futures": False, "has_options": False, "has_margin": True,
            "maker_fee": 0.001, "taker_fee": 0.002, "min_order_size_usd": 5.0,
            "supported_fiats": ["USD", "EUR", "GBP"],
            "countries_blocked": ["US"],
            "rest_url": _BITFINEX_REST,
        },
        "bybit": {
            "has_spot": True, "has_futures": True, "has_options": True, "has_margin": True,
            "maker_fee": 0.0001, "taker_fee": 0.0006, "min_order_size_usd": 1.0,
            "supported_fiats": ["USD", "USDT"],
            "countries_blocked": ["US", "UK"],
            "rest_url": _BYBIT_REST,
        },
        "okx": {
            "has_spot": True, "has_futures": True, "has_options": True, "has_margin": True,
            "maker_fee": 0.0008, "taker_fee": 0.001, "min_order_size_usd": 1.0,
            "supported_fiats": ["USD", "EUR"],
            "countries_blocked": ["US"],
            "rest_url": "https://www.okx.com/api/v5",
        },
        "huobi": {
            "has_spot": True, "has_futures": True, "has_options": False, "has_margin": True,
            "maker_fee": 0.002, "taker_fee": 0.002, "min_order_size_usd": 5.0,
            "supported_fiats": ["USD", "CNY"],
            "countries_blocked": ["US"],
            "rest_url": "https://api.huobi.pro",
        },
        "kucoin": {
            "has_spot": True, "has_futures": True, "has_options": False, "has_margin": True,
            "maker_fee": 0.001, "taker_fee": 0.001, "min_order_size_usd": 0.1,
            "supported_fiats": ["USD", "EUR"],
            "countries_blocked": [],
            "rest_url": "https://api.kucoin.com",
        },
        "gate": {
            "has_spot": True, "has_futures": True, "has_options": False, "has_margin": True,
            "maker_fee": 0.002, "taker_fee": 0.002, "min_order_size_usd": 1.0,
            "supported_fiats": ["USD", "USDT"],
            "countries_blocked": ["US"],
            "rest_url": "https://api.gateio.ws/api/v4",
        },
        "mexc": {
            "has_spot": True, "has_futures": True, "has_options": False, "has_margin": False,
            "maker_fee": 0.0, "taker_fee": 0.0002, "min_order_size_usd": 1.0,
            "supported_fiats": ["USDT"],
            "countries_blocked": ["US"],
            "rest_url": "https://api.mexc.com/api/v3",
        },
        "bitget": {
            "has_spot": True, "has_futures": True, "has_options": False, "has_margin": True,
            "maker_fee": 0.001, "taker_fee": 0.001, "min_order_size_usd": 5.0,
            "supported_fiats": ["USDT", "USD"],
            "countries_blocked": ["US"],
            "rest_url": "https://api.bitget.com",
        },
        "bitmex": {
            "has_spot": False, "has_futures": True, "has_options": False, "has_margin": True,
            "maker_fee": -0.00025, "taker_fee": 0.00075, "min_order_size_usd": 1.0,
            "supported_fiats": ["USD"],
            "countries_blocked": ["US"],
            "rest_url": "https://www.bitmex.com/api/v1",
        },
        "deribit": {
            "has_spot": False, "has_futures": True, "has_options": True, "has_margin": True,
            "maker_fee": 0.0, "taker_fee": 0.0003, "min_order_size_usd": 1.0,
            "supported_fiats": ["USD", "EUR"],
            "countries_blocked": ["US"],
            "rest_url": "https://www.deribit.com/api/v2",
        },
        "ftx_us": {
            "has_spot": True, "has_futures": False, "has_options": False, "has_margin": False,
            "maker_fee": 0.002, "taker_fee": 0.002, "min_order_size_usd": 1.0,
            "supported_fiats": ["USD"],
            "countries_blocked": [],
            "rest_url": "https://ftx.us/api",
        },
        "gemini": {
            "has_spot": True, "has_futures": False, "has_options": False, "has_margin": False,
            "maker_fee": 0.0, "taker_fee": 0.004, "min_order_size_usd": 0.01,
            "supported_fiats": ["USD", "GBP", "EUR", "SGD"],
            "countries_blocked": [],
            "rest_url": "https://api.gemini.com/v1",
        },
        "bitstamp": {
            "has_spot": True, "has_futures": False, "has_options": False, "has_margin": False,
            "maker_fee": 0.003, "taker_fee": 0.004, "min_order_size_usd": 10.0,
            "supported_fiats": ["USD", "EUR", "GBP"],
            "countries_blocked": [],
            "rest_url": "https://www.bitstamp.net/api/v2",
        },
        "bittrex": {
            "has_spot": True, "has_futures": False, "has_options": False, "has_margin": False,
            "maker_fee": 0.002, "taker_fee": 0.002, "min_order_size_usd": 0.1,
            "supported_fiats": ["USD"],
            "countries_blocked": [],
            "rest_url": "https://api.bittrex.com/v3",
        },
        "poloniex": {
            "has_spot": True, "has_futures": False, "has_options": False, "has_margin": False,
            "maker_fee": 0.001, "taker_fee": 0.002, "min_order_size_usd": 0.01,
            "supported_fiats": ["USD", "USDT"],
            "countries_blocked": ["US"],
            "rest_url": "https://api.poloniex.com",
        },
        "cryptocom": {
            "has_spot": True, "has_futures": True, "has_options": False, "has_margin": False,
            "maker_fee": 0.004, "taker_fee": 0.004, "min_order_size_usd": 1.0,
            "supported_fiats": ["USD", "EUR", "GBP"],
            "countries_blocked": [],
            "rest_url": "https://api.crypto.com/v2",
        },
        "upbit": {
            "has_spot": True, "has_futures": False, "has_options": False, "has_margin": False,
            "maker_fee": 0.0005, "taker_fee": 0.0005, "min_order_size_usd": 1.0,
            "supported_fiats": ["KRW", "BTC", "USDT"],
            "countries_blocked": [],
            "rest_url": "https://api.upbit.com/v1",
        },
    }

    def list_exchanges(self) -> List[str]:
        return list(self.SUPPORTED_EXCHANGES.keys())

    def get_info(self, exchange_id: str) -> ExchangeInfo:
        data = self.SUPPORTED_EXCHANGES.get(exchange_id)
        if data is None:
            raise ValueError(f"Unknown exchange: {exchange_id}")
        return ExchangeInfo(exchange_id=exchange_id, **data)

    def get_exchange(self, exchange_id: str) -> Any:
        """Return a ccxt exchange instance or a REST-fallback stub."""
        data = self.SUPPORTED_EXCHANGES.get(exchange_id)
        if data is None:
            raise ValueError(f"Unknown exchange: {exchange_id}")
        if _CCXT_AVAILABLE:
            try:
                cls = getattr(ccxt, exchange_id, None)
                if cls is None:
                    # Try aliases
                    _aliases = {"cryptocom": "cryptocom", "gate": "gateio"}
                    cls = getattr(ccxt, _aliases.get(exchange_id, exchange_id), None)
                if cls is not None:
                    return cls({"enableRateLimit": True})
            except Exception as exc:  # noqa: BLE001
                logger.warning("CCXT init failed for %s: %s", exchange_id, exc)
        # Return a simple REST-fallback stub
        return _RESTFallbackExchange(exchange_id, data.get("rest_url", ""))

    def filter_by_capability(
        self,
        has_futures: bool = False,
        has_options: bool = False,
        fiat: Optional[str] = None,
    ) -> List[str]:
        result = []
        for eid, data in self.SUPPORTED_EXCHANGES.items():
            if has_futures and not data.get("has_futures"):
                continue
            if has_options and not not data.get("has_options"):
                continue
            if fiat and fiat not in data.get("supported_fiats", []):
                continue
            result.append(eid)
        return result


# ---------------------------------------------------------------------------
# REST fallback exchange stub
# ---------------------------------------------------------------------------


class _RESTFallbackExchange:
    """Minimal REST-based shim used when ccxt is not installed."""

    def __init__(self, exchange_id: str, rest_url: str) -> None:
        self.id = exchange_id
        self._rest_url = rest_url

    def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 500,
    ) -> List[List[float]]:
        """Fetch OHLCV via Binance public REST (works for most BTC/ETH pairs)."""
        # Normalise symbol: BTC/USDT → BTCUSDT
        b_symbol = symbol.replace("/", "")
        tf = _TF_MAP_BINANCE.get(timeframe, "1h")
        params: Dict[str, Any] = {"symbol": b_symbol, "interval": tf, "limit": min(limit, 1000)}
        if since is not None:
            params["startTime"] = since
        try:
            data = self._get(f"{_BINANCE_REST}/klines", params)
            # Binance returns [ts, open, high, low, close, vol, ...]
            return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                    for r in data]
        except Exception as exc:  # noqa: BLE001
            logger.error("REST fallback OHLCV failed for %s/%s: %s", self.id, symbol, exc)
            return []

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        b_symbol = symbol.replace("/", "")
        params = {"symbol": b_symbol, "limit": min(limit, 100)}
        try:
            data = self._get(f"{_BINANCE_REST}/depth", params)
            return {
                "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
                "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
                "timestamp": int(time.time() * 1000),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("REST fallback order book failed: %s", exc)
            return {"bids": [], "asks": [], "timestamp": int(time.time() * 1000)}

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        b_symbol = symbol.replace("/", "")
        try:
            data = self._get(f"{_BINANCE_REST}/ticker/bookTicker", {"symbol": b_symbol})
            return {
                "bid": float(data.get("bidPrice", 0)),
                "ask": float(data.get("askPrice", 0)),
                "bidVolume": float(data.get("bidQty", 0)),
                "askVolume": float(data.get("askQty", 0)),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("REST fallback ticker failed: %s", exc)
            return {"bid": 0.0, "ask": 0.0, "bidVolume": 0.0, "askVolume": 0.0}

    def fetch_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        b_symbol = symbol.replace("/", "")
        try:
            data = self._get(f"{_BINANCE_REST}/trades", {"symbol": b_symbol, "limit": limit})
            return [
                {
                    "id": str(t["id"]),
                    "timestamp": t["time"],
                    "price": float(t["price"]),
                    "amount": float(t["qty"]),
                    "side": "sell" if t["isBuyerMaker"] else "buy",
                }
                for t in data
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("REST fallback trades failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# 2. UnifiedOHLCVCollector
# ---------------------------------------------------------------------------


class UnifiedOHLCVCollector:
    """Multi-exchange OHLCV fetching with pagination and VWAP aggregation."""

    VALID_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}

    # Bars per page per exchange (conservative)
    _PAGE_SIZE = 500

    def __init__(self) -> None:
        self._registry = CCXTExchangeRegistry()

    def fetch_ohlcv(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV from a single exchange with auto-pagination."""
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe {timeframe!r}. Valid: {self.VALID_TIMEFRAMES}")

        cache_key = f"ohlcv:{exchange}:{symbol}:{timeframe}:{since}:{limit}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        exch = self._registry.get_exchange(exchange)
        all_bars: List[List[float]] = []
        remaining = limit
        current_since = since

        while remaining > 0:
            batch_size = min(remaining, self._PAGE_SIZE)
            try:
                bars = exch.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=current_since, limit=batch_size
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("OHLCV fetch error (%s/%s): %s", exchange, symbol, exc)
                break

            if not bars:
                break
            all_bars.extend(bars)
            remaining -= len(bars)

            # Advance since pointer for next page
            last_ts = bars[-1][0]
            if current_since is not None and last_ts <= current_since:
                break  # no progress → stop
            current_since = last_ts + 1

            if len(bars) < batch_size:
                break  # exchange returned fewer bars than requested → done

        df = self._bars_to_df(all_bars, exchange)
        self._persist_ohlcv(df, exchange, symbol, timeframe)
        _cache_set(cache_key, df)
        return df

    def fetch_multi_exchange(
        self,
        symbol: str,
        exchanges: List[str],
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV from multiple exchanges and compute cross-exchange VWAP."""
        frames: List[pd.DataFrame] = []
        for exch in exchanges:
            try:
                df = self.fetch_ohlcv(symbol, exch, timeframe, since, limit)
                df["exchange"] = exch
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s for multi-exchange OHLCV: %s", exch, exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        return self._compute_vwap_aggregation(combined)

    def _compute_vwap_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute VWAP-weighted best price across exchanges for each timestamp."""
        if df.empty:
            return df

        agg_rows = []
        for ts, group in df.groupby("ts"):
            total_vol = group["volume"].sum()
            if total_vol == 0:
                vwap_close = group["close"].mean()
            else:
                vwap_close = (group["close"] * group["volume"]).sum() / total_vol

            agg_rows.append({
                "ts": ts,
                "open": group["open"].iloc[0],
                "high": group["high"].max(),
                "low": group["low"].min(),
                "close": vwap_close,
                "volume": total_vol,
                "exchange": "vwap_aggregate",
                "n_exchanges": len(group),
            })

        result = pd.DataFrame(agg_rows)
        result.sort_values("ts", inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    @staticmethod
    def _bars_to_df(bars: List[List[float]], exchange: str) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "exchange"])
        df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        df["exchange"] = exchange
        df.sort_values("ts", inplace=True)
        df.drop_duplicates("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _persist_ohlcv(
        self, df: pd.DataFrame, exchange: str, symbol: str, timeframe: str
    ) -> None:
        if df.empty:
            return
        rows = [
            (exchange, symbol, timeframe, int(r["ts"]),
             r["open"], r["high"], r["low"], r["close"], r["volume"])
            for _, r in df.iterrows()
        ]
        with _db() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO ohlcv_cache
                   (exchange, symbol, timeframe, ts, open, high, low, close, volume)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def load_cached_ohlcv(
        self, exchange: str, symbol: str, timeframe: str, limit: int = 500
    ) -> pd.DataFrame:
        with _db() as conn:
            rows = conn.execute(
                """SELECT ts, open, high, low, close, volume FROM ohlcv_cache
                   WHERE exchange=? AND symbol=? AND timeframe=?
                   ORDER BY ts DESC LIMIT ?""",
                (exchange, symbol, timeframe, limit),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df.sort_values("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_latest_price(self, symbol: str, exchange: str) -> Optional[float]:
        df = self.fetch_ohlcv(symbol, exchange, "1m", limit=1)
        if df.empty:
            return None
        return float(df["close"].iloc[-1])


# ---------------------------------------------------------------------------
# 3. OrderBookAggregator
# ---------------------------------------------------------------------------


class OrderBookAggregator:
    """Fetch and aggregate L2 order books across multiple exchanges."""

    def __init__(self, depth: int = 20) -> None:
        self._registry = CCXTExchangeRegistry()
        self.depth = depth

    def fetch_order_book(self, exchange: str, symbol: str) -> Dict[str, Any]:
        cache_key = f"ob:{exchange}:{symbol}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        exch = self._registry.get_exchange(exchange)
        try:
            ob = exch.fetch_order_book(symbol, self.depth)
        except Exception as exc:  # noqa: BLE001
            logger.error("Order book fetch error (%s/%s): %s", exchange, symbol, exc)
            return {"bids": [], "asks": [], "timestamp": int(time.time() * 1000), "exchange": exchange}

        ob["exchange"] = exchange
        _cache_set(cache_key, ob)
        return ob

    def aggregate(self, symbol: str, exchanges: List[str]) -> ConsolidatedOrderBook:
        """Aggregate order books from multiple exchanges into a consolidated book."""
        all_bids: List[Tuple[float, float]] = []
        all_asks: List[Tuple[float, float]] = []
        exchange_spreads: Dict[str, float] = {}

        for exch_id in exchanges:
            ob = self.fetch_order_book(exch_id, symbol)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if bids:
                all_bids.extend([(float(b[0]), float(b[1])) for b in bids if len(b) >= 2])
            if asks:
                all_asks.extend([(float(a[0]), float(a[1])) for a in asks if len(a) >= 2])

            if bids and asks:
                best_bid = max(b[0] for b in bids)
                best_ask = min(a[0] for a in asks)
                exchange_spreads[exch_id] = best_ask - best_bid

        # Sort bids descending, asks ascending
        all_bids.sort(key=lambda x: x[0], reverse=True)
        all_asks.sort(key=lambda x: x[0])

        # Deduplicate by aggregating size at same price level
        agg_bids = self._aggregate_levels(all_bids)
        agg_asks = self._aggregate_levels(all_asks)

        best_bid = agg_bids[0][0] if agg_bids else 0.0
        best_ask = agg_asks[0][0] if agg_asks else 0.0
        spread = best_ask - best_bid
        spread_pct = (spread / best_ask * 100) if best_ask > 0 else 0.0

        total_bid_vol = sum(p for _, p in agg_bids[:self.depth])
        total_ask_vol = sum(p for _, p in agg_asks[:self.depth])
        total_vol = total_bid_vol + total_ask_vol
        imbalance = ((total_bid_vol - total_ask_vol) / total_vol) if total_vol > 0 else 0.0

        return ConsolidatedOrderBook(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_pct=spread_pct,
            imbalance=imbalance,
            bids=[OrderBookLevel(price=p, size=s) for p, s in agg_bids[:self.depth]],
            asks=[OrderBookLevel(price=p, size=s) for p, s in agg_asks[:self.depth]],
            exchange_spreads=exchange_spreads,
        )

    @staticmethod
    def _aggregate_levels(
        levels: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        aggregated: Dict[float, float] = {}
        for price, size in levels:
            aggregated[price] = aggregated.get(price, 0.0) + size
        return sorted(aggregated.items(), key=lambda x: x[0], reverse=True)

    def get_depth_profile(
        self, symbol: str, exchange: str, price_range_pct: float = 2.0
    ) -> Dict[str, Any]:
        """Cumulative bid/ask volume within price_range_pct of mid-price."""
        ob = self.fetch_order_book(exchange, symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        if not bids or not asks:
            return {"error": "no data"}

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2
        floor_price = mid * (1 - price_range_pct / 100)
        ceiling_price = mid * (1 + price_range_pct / 100)

        cum_bid_vol = sum(float(b[1]) for b in bids if float(b[0]) >= floor_price)
        cum_ask_vol = sum(float(a[1]) for a in asks if float(a[0]) <= ceiling_price)

        return {
            "symbol": symbol,
            "exchange": exchange,
            "mid_price": mid,
            "price_range_pct": price_range_pct,
            "cum_bid_vol": cum_bid_vol,
            "cum_ask_vol": cum_ask_vol,
            "bid_ask_ratio": (cum_bid_vol / cum_ask_vol) if cum_ask_vol > 0 else None,
        }


# ---------------------------------------------------------------------------
# 4. CrossExchangeArbitrageDetector
# ---------------------------------------------------------------------------


# Approximate transfer times (minutes) per asset
_TRANSFER_TIMES: Dict[str, int] = {
    "BTC": 30, "ETH": 5, "USDC": 1, "USDT": 1, "SOL": 1,
    "BNB": 1, "XRP": 1, "ADA": 5, "DOGE": 5, "LTC": 3,
}
_DEFAULT_TRANSFER_TIME = 10


class CrossExchangeArbitrageDetector:
    """Detect cross-exchange arbitrage opportunities in real time."""

    MIN_PROFIT_PCT = 0.10  # 10 basis points

    def __init__(self) -> None:
        self._registry = CCXTExchangeRegistry()
        self._ob_agg = OrderBookAggregator(depth=5)

    def _get_ticker(self, exchange: str, symbol: str) -> Dict[str, float]:
        exch = self._registry.get_exchange(exchange)
        try:
            if hasattr(exch, "fetch_ticker"):
                t = exch.fetch_ticker(symbol)
                return {
                    "bid": float(t.get("bid") or 0),
                    "ask": float(t.get("ask") or 0),
                    "bid_vol": float(t.get("bidVolume") or 0),
                    "ask_vol": float(t.get("askVolume") or 0),
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("ticker fetch failed %s/%s: %s", exchange, symbol, exc)
        return {"bid": 0.0, "ask": 0.0, "bid_vol": 0.0, "ask_vol": 0.0}

    def scan_pair(
        self, symbol: str, exchanges: List[str]
    ) -> List[ArbitrageOpportunity]:
        """Scan all exchange pairs for the given symbol for arb opportunities."""
        tickers: Dict[str, Dict[str, float]] = {}
        for exch in exchanges:
            cache_key = f"ticker:{exch}:{symbol}"
            cached = _cache_get(cache_key)
            if cached is not None:
                tickers[exch] = cached  # type: ignore[assignment]
            else:
                t = self._get_ticker(exch, symbol)
                tickers[exch] = t
                _cache_set(cache_key, t)

        opportunities: List[ArbitrageOpportunity] = []
        exchange_info = {e: self._registry.get_info(e) for e in exchanges}
        base_asset = symbol.split("/")[0] if "/" in symbol else symbol[:3]
        transfer_time = _TRANSFER_TIMES.get(base_asset.upper(), _DEFAULT_TRANSFER_TIME)

        for buy_exch in exchanges:
            for sell_exch in exchanges:
                if buy_exch == sell_exch:
                    continue
                buy_ticker = tickers.get(buy_exch, {})
                sell_ticker = tickers.get(sell_exch, {})
                buy_ask = buy_ticker.get("ask", 0.0)
                sell_bid = sell_ticker.get("bid", 0.0)

                if buy_ask <= 0 or sell_bid <= 0:
                    continue

                gross_pct = (sell_bid - buy_ask) / buy_ask * 100
                buy_fee_pct = exchange_info[buy_exch].taker_fee * 100
                sell_fee_pct = exchange_info[sell_exch].taker_fee * 100
                fee_pct = buy_fee_pct + sell_fee_pct
                net_pct = gross_pct - fee_pct

                if net_pct > self.MIN_PROFIT_PCT:
                    opp = ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=buy_exch,
                        sell_exchange=sell_exch,
                        buy_price=buy_ask,
                        sell_price=sell_bid,
                        gross_pct=round(gross_pct, 4),
                        fee_pct=round(fee_pct, 4),
                        net_pct=round(net_pct, 4),
                        transfer_time_min=transfer_time,
                        actionable=net_pct > self.MIN_PROFIT_PCT,
                        detected_at=datetime.now(timezone.utc).isoformat(),
                    )
                    opportunities.append(opp)
                    self._log_arb(opp)

        return sorted(opportunities, key=lambda x: x.net_pct, reverse=True)

    def scan_arbitrage(
        self,
        symbols: List[str],
        exchanges: Optional[List[str]] = None,
    ) -> List[ArbitrageOpportunity]:
        """Scan multiple symbols across exchanges for arbitrage."""
        if exchanges is None:
            exchanges = ["binance", "coinbase", "kraken", "gemini", "bitstamp"]
        all_opps: List[ArbitrageOpportunity] = []
        for sym in symbols:
            opps = self.scan_pair(sym, exchanges)
            all_opps.extend(opps)
        return sorted(all_opps, key=lambda x: x.net_pct, reverse=True)

    def _log_arb(self, opp: ArbitrageOpportunity) -> None:
        with _db() as conn:
            conn.execute(
                """INSERT INTO arb_log
                   (symbol, buy_exchange, sell_exchange, profit_pct, buy_price, sell_price, detected_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (opp.symbol, opp.buy_exchange, opp.sell_exchange,
                 opp.net_pct, opp.buy_price, opp.sell_price, opp.detected_at),
            )

    def get_arb_history(self, symbol: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        with _db() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM arb_log WHERE symbol=? ORDER BY detected_at DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM arb_log ORDER BY detected_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def historical_arb_frequency(self, top_n: int = 10) -> pd.DataFrame:
        """Which symbol/pair combination arbitrages most often."""
        with _db() as conn:
            rows = conn.execute(
                """SELECT symbol, buy_exchange, sell_exchange,
                          COUNT(*) as freq, AVG(profit_pct) as avg_profit
                   FROM arb_log
                   GROUP BY symbol, buy_exchange, sell_exchange
                   ORDER BY freq DESC
                   LIMIT ?""",
                (top_n,),
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. UnifiedExecutionAdapter
# ---------------------------------------------------------------------------


_PAPER_POSITIONS: Dict[str, PositionInfo] = {}


class UnifiedExecutionAdapter:
    """Place and manage orders across exchanges via a unified interface."""

    MAX_ORDER_SIZE_USD = 100_000.0
    PRICE_DEVIATION_PCT = 5.0  # ±5% from market price

    def __init__(self, paper_trading: bool = True) -> None:
        self._registry = CCXTExchangeRegistry()
        self.paper_trading = paper_trading
        self._ohlcv = UnifiedOHLCVCollector()

    def place_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        trail_pct: Optional[float] = None,
        iceberg_show: Optional[float] = None,
    ) -> OrderResult:
        """Place an order on exchange. In paper mode: simulated."""
        side = side.lower()
        order_type = order_type.lower()

        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side!r}")
        if order_type not in ("market", "limit", "stop_limit", "trailing_stop", "iceberg"):
            raise ValueError(f"Invalid order_type: {order_type!r}")

        market_price = self._ohlcv.get_latest_price(symbol, exchange) or 0.0

        # Risk checks
        if market_price > 0:
            order_value_usd = amount * market_price
            if order_value_usd > self.MAX_ORDER_SIZE_USD:
                raise ValueError(
                    f"Order size ${order_value_usd:,.0f} exceeds max ${self.MAX_ORDER_SIZE_USD:,.0f}"
                )
            if price is not None:
                deviation_pct = abs(price - market_price) / market_price * 100
                if deviation_pct > self.PRICE_DEVIATION_PCT:
                    raise ValueError(
                        f"Limit price {price} deviates {deviation_pct:.1f}% from market {market_price}"
                    )

        if self.paper_trading:
            return self._paper_place(exchange, symbol, side, order_type, amount, price, market_price)

        return self._live_place(exchange, symbol, side, order_type, amount, price)

    def _paper_place(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float],
        market_price: float,
    ) -> OrderResult:
        """Simulate order execution."""
        order_id = str(uuid.uuid4())
        fill_price: Optional[float] = None
        status = "open"
        filled = 0.0

        if order_type == "market":
            # Simulate immediate fill at market price with 0.05% slippage
            slippage = 0.0005
            fill_price = market_price * (1 + slippage) if side == "buy" else market_price * (1 - slippage)
            filled = amount
            status = "closed"
        elif order_type == "limit" and price is not None:
            # Check if immediately fillable
            if side == "buy" and price >= market_price:
                fill_price = price
                filled = amount
                status = "closed"
            elif side == "sell" and price <= market_price:
                fill_price = price
                filled = amount
                status = "closed"

        now = datetime.now(timezone.utc).isoformat()
        result = OrderResult(
            order_id=order_id,
            exchange=exchange,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            status=status,
            filled=filled,
            avg_fill=fill_price,
            paper=True,
            created_at=now,
        )

        with _db() as conn:
            conn.execute(
                """INSERT INTO paper_orders
                   (order_id, exchange, symbol, side, order_type, amount, price,
                    status, filled, avg_fill, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (order_id, exchange, symbol, side, order_type, amount, price,
                 status, filled, fill_price, now, now),
            )

        # Update paper position
        if status == "closed" and fill_price is not None:
            self._update_paper_position(exchange, symbol, side, amount, fill_price)

        return result

    def _update_paper_position(
        self, exchange: str, symbol: str, side: str, amount: float, fill_price: float
    ) -> None:
        key = f"{exchange}:{symbol}"
        existing = _PAPER_POSITIONS.get(key)
        if existing is None:
            pos_side = "long" if side == "buy" else "short"
            _PAPER_POSITIONS[key] = PositionInfo(
                exchange=exchange, symbol=symbol, side=pos_side,
                size=amount, entry_price=fill_price, unrealized_pnl=0.0,
            )
        else:
            if (existing.side == "long" and side == "buy") or (existing.side == "short" and side == "sell"):
                new_size = existing.size + amount
                new_entry = (existing.entry_price * existing.size + fill_price * amount) / new_size
                _PAPER_POSITIONS[key] = PositionInfo(
                    exchange=exchange, symbol=symbol, side=existing.side,
                    size=new_size, entry_price=new_entry, unrealized_pnl=0.0,
                )
            else:
                new_size = existing.size - amount
                if new_size <= 0:
                    del _PAPER_POSITIONS[key]
                else:
                    _PAPER_POSITIONS[key] = PositionInfo(
                        exchange=exchange, symbol=symbol, side=existing.side,
                        size=new_size, entry_price=existing.entry_price, unrealized_pnl=0.0,
                    )

    def _live_place(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float],
    ) -> OrderResult:
        """Attempt to place an order via CCXT. Requires API keys in env."""
        if not _CCXT_AVAILABLE:
            raise RuntimeError("CCXT not installed; cannot place live orders")
        exch = self._registry.get_exchange(exchange)
        api_key = os.environ.get(f"{exchange.upper()}_API_KEY")
        api_secret = os.environ.get(f"{exchange.upper()}_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(f"API keys not set for {exchange}")
        exch.apiKey = api_key
        exch.secret = api_secret

        try:
            raw = exch.create_order(symbol, order_type, side, amount, price)
        except Exception as exc:
            raise RuntimeError(f"Order placement failed: {exc}") from exc

        now = datetime.now(timezone.utc).isoformat()
        return OrderResult(
            order_id=str(raw.get("id", "")),
            exchange=exchange,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            status=raw.get("status", "open"),
            filled=float(raw.get("filled", 0)),
            avg_fill=raw.get("average"),
            paper=False,
            created_at=raw.get("datetime", now),
        )

    def get_open_orders(self, exchange: Optional[str] = None) -> pd.DataFrame:
        with _db() as conn:
            if exchange:
                rows = conn.execute(
                    "SELECT * FROM paper_orders WHERE exchange=? AND status='open' ORDER BY created_at DESC",
                    (exchange,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM paper_orders WHERE status='open' ORDER BY created_at DESC"
                ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_paper_positions(self) -> List[PositionInfo]:
        return list(_PAPER_POSITIONS.values())

    def cancel_order(self, order_id: str) -> bool:
        with _db() as conn:
            cur = conn.execute(
                "UPDATE paper_orders SET status='cancelled', updated_at=? WHERE order_id=? AND status='open'",
                (datetime.now(timezone.utc).isoformat(), order_id),
            )
        return cur.rowcount > 0

    def get_fills(self, exchange: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        with _db() as conn:
            if exchange:
                rows = conn.execute(
                    "SELECT * FROM paper_orders WHERE exchange=? AND status='closed' ORDER BY updated_at DESC LIMIT ?",
                    (exchange, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM paper_orders WHERE status='closed' ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def pnl_summary(self) -> Dict[str, Any]:
        fills = self.get_fills(limit=10000)
        if fills.empty:
            return {"realized_pnl": 0.0, "n_trades": 0, "win_rate": 0.0}

        fills["side_mult"] = fills["side"].map({"buy": -1, "sell": 1})
        fills["cash_flow"] = fills["side_mult"] * fills["filled"] * fills["avg_fill"].fillna(0)
        realized = fills["cash_flow"].sum()
        buys = fills[fills["side"] == "buy"]
        sells = fills[fills["side"] == "sell"]
        winning = sells[sells["avg_fill"] > fills.loc[buys.index, "avg_fill"] if not buys.empty else 0].shape[0]
        return {
            "realized_pnl": round(realized, 2),
            "n_trades": len(fills),
            "win_rate": round(winning / len(sells) * 100, 1) if len(sells) > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# 6. ExchangeHealthMonitor
# ---------------------------------------------------------------------------


class ExchangeHealthMonitor:
    """Ping exchanges and track latency, rate limits, and maintenance status."""

    SLOW_THRESHOLD_MS = 500.0

    def __init__(self) -> None:
        self._registry = CCXTExchangeRegistry()
        self._rate_limits: Dict[str, int] = {}

    def ping(self, exchange: str) -> ExchangeHealthStatus:
        """Measure latency to an exchange's public API."""
        info = self._registry.get_info(exchange)
        url = info.rest_url
        if not url:
            return ExchangeHealthStatus(
                exchange=exchange,
                latency_ms=None,
                status="unknown",
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

        start = time.monotonic()
        try:
            resp = requests.get(
                url, headers=_HEADERS, timeout=5.0, allow_redirects=True
            )
            latency_ms = (time.monotonic() - start) * 1000
            status = "ok" if resp.status_code < 500 else "down"
            if latency_ms > self.SLOW_THRESHOLD_MS:
                status = "slow"
        except requests.exceptions.Timeout:
            latency_ms = None
            status = "down"
        except Exception:  # noqa: BLE001
            latency_ms = None
            status = "down"

        now = datetime.now(timezone.utc).isoformat()
        hs = ExchangeHealthStatus(
            exchange=exchange,
            latency_ms=round(latency_ms, 1) if latency_ms else None,
            status=status,
            checked_at=now,
            rate_limit_remaining=self._rate_limits.get(exchange),
        )
        with _db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO exchange_health (exchange, checked_at, latency_ms, status) VALUES (?,?,?,?)",
                (exchange, now, latency_ms, status),
            )
        return hs

    def ping_all(self, exchanges: Optional[List[str]] = None) -> List[ExchangeHealthStatus]:
        if exchanges is None:
            exchanges = list(self._registry.SUPPORTED_EXCHANGES.keys())
        results = []
        for exch in exchanges:
            try:
                results.append(self.ping(exch))
            except Exception as exc:  # noqa: BLE001
                logger.error("Health ping failed for %s: %s", exch, exc)
        return sorted(results, key=lambda x: (x.status != "ok", x.latency_ms or 9999))

    def get_best_exchange(
        self, exchanges: List[str], max_latency_ms: float = 300.0
    ) -> Optional[str]:
        """Return the fastest healthy exchange from the list."""
        results = [self.ping(e) for e in exchanges]
        healthy = [r for r in results if r.status == "ok" and (r.latency_ms or 9999) <= max_latency_ms]
        if not healthy:
            return None
        return min(healthy, key=lambda x: x.latency_ms or 9999).exchange

    def track_rate_limit(self, exchange: str, remaining: int) -> None:
        self._rate_limits[exchange] = remaining

    def health_history(self, exchange: str, hours: int = 24) -> pd.DataFrame:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM exchange_health WHERE exchange=? AND checked_at>=? ORDER BY checked_at",
                (exchange, since),
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

ccxt_router = APIRouter(prefix="/ccxt", tags=["ccxt-multi-exchange"])

_registry = CCXTExchangeRegistry()
_ohlcv_collector = UnifiedOHLCVCollector()
_ob_aggregator = OrderBookAggregator()
_arb_detector = CrossExchangeArbitrageDetector()
_execution = UnifiedExecutionAdapter(paper_trading=True)
_health_monitor = ExchangeHealthMonitor()


@ccxt_router.get("/exchanges")
def list_exchanges(
    has_futures: bool = Query(False),
    has_options: bool = Query(False),
    fiat: Optional[str] = Query(None),
):
    """List supported exchanges, optionally filtered by capability."""
    if has_futures or has_options or fiat:
        exchanges = _registry.filter_by_capability(
            has_futures=has_futures, has_options=has_options, fiat=fiat
        )
    else:
        exchanges = _registry.list_exchanges()
    return {
        "exchanges": [_registry.get_info(e).model_dump() for e in exchanges],
        "total": len(exchanges),
    }


@ccxt_router.get("/ohlcv/{exchange}/{symbol:path}")
def get_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str = Query("1h"),
    limit: int = Query(200),
    since: Optional[int] = Query(None),
    multi_exchange: bool = Query(False),
):
    """Fetch OHLCV data. Use ?multi_exchange=true for VWAP aggregation."""
    symbol = symbol.replace("-", "/").upper()
    if exchange not in _registry.SUPPORTED_EXCHANGES:
        raise HTTPException(404, f"Unknown exchange: {exchange}")
    if multi_exchange:
        exchanges = [exchange, "binance", "coinbase", "kraken"]
        df = _ohlcv_collector.fetch_multi_exchange(symbol, exchanges, timeframe, since, limit)
    else:
        df = _ohlcv_collector.fetch_ohlcv(symbol, exchange, timeframe, since, limit)

    if df.empty:
        raise HTTPException(404, f"No OHLCV data for {symbol} on {exchange}")
    return {"symbol": symbol, "exchange": exchange, "timeframe": timeframe, "bars": df.to_dict("records")}


@ccxt_router.get("/orderbook/{exchange}/{symbol:path}")
def get_orderbook(
    exchange: str,
    symbol: str,
    depth: int = Query(20),
    aggregate: bool = Query(False),
):
    """Fetch L2 order book. Use ?aggregate=true for multi-exchange consolidated book."""
    symbol = symbol.replace("-", "/").upper()
    if aggregate:
        exchanges = ["binance", "coinbase", "kraken", "bitstamp"]
        ob = _ob_aggregator.aggregate(symbol, exchanges)
        return ob.model_dump()
    ob_data = _ob_aggregator.fetch_order_book(exchange, symbol)
    return {"symbol": symbol, "exchange": exchange, **ob_data}


@ccxt_router.get("/arbitrage/scan")
def scan_arbitrage(
    symbols: str = Query("BTC/USDT,ETH/USDT"),
    exchanges: str = Query("binance,coinbase,kraken,gemini,bitstamp"),
):
    """Scan for cross-exchange arbitrage opportunities."""
    sym_list = [s.strip().upper() for s in symbols.split(",")]
    exch_list = [e.strip() for e in exchanges.split(",")]
    opportunities = _arb_detector.scan_arbitrage(sym_list, exch_list)
    return {
        "opportunities": [o.model_dump() for o in opportunities],
        "total": len(opportunities),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


class PlaceOrderRequest(BaseModel):
    exchange: str
    symbol: str
    side: str
    order_type: str = "market"
    amount: float
    price: Optional[float] = None
    paper: bool = True


@ccxt_router.post("/order/place")
def place_order(req: PlaceOrderRequest):
    """Place an order (paper mode by default)."""
    adapter = UnifiedExecutionAdapter(paper_trading=req.paper)
    try:
        result = adapter.place_order(
            exchange=req.exchange,
            symbol=req.symbol.upper(),
            side=req.side,
            order_type=req.order_type,
            amount=req.amount,
            price=req.price,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))
    return result.model_dump()


@ccxt_router.get("/positions")
def get_positions():
    """Get open paper trading positions."""
    return {
        "positions": [p.model_dump() for p in _execution.get_paper_positions()],
        "open_orders": _execution.get_open_orders().to_dict("records"),
        "pnl": _execution.pnl_summary(),
    }


@ccxt_router.get("/health")
def get_exchange_health(
    exchanges: str = Query("binance,coinbase,kraken,bybit,okx"),
):
    """Ping exchanges and return latency + health status."""
    exch_list = [e.strip() for e in exchanges.split(",")]
    results = _health_monitor.ping_all(exch_list)
    return {
        "results": [r.model_dump() for r in results],
        "best_exchange": _health_monitor.get_best_exchange(exch_list),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@ccxt_router.get("/arb/history")
def arb_history(symbol: Optional[str] = Query(None), limit: int = Query(50)):
    df = _arb_detector.get_arb_history(symbol, limit)
    return {"history": df.to_dict("records")}


@ccxt_router.get("/arb/frequency")
def arb_frequency(top_n: int = Query(10)):
    df = _arb_detector.historical_arb_frequency(top_n)
    return {"frequency": df.to_dict("records")}
