"""Alpaca WebSocket market-data streaming client.

Connects to the Alpaca free paper-account market data stream and buffers
real-time trades into completed 1-minute OHLCV bars.  Completed bars are
published to a Redis pub/sub channel so any downstream consumer
(backtest replay, live dashboard, signal engine) can subscribe without
coupling to the WebSocket lifecycle.

Architecture
------------

    AlpacaWSClient
        │  receives trade/quote ticks from wss://stream.data.alpaca.markets
        │
        ▼
    BarBuffer
        │  accumulates ticks; emits OHLCVBar on minute boundary
        │
        ▼
    Redis pub/sub  →  channel  "sentinel:bars:1m:{ticker}"
        │
        ▼
    downstream consumers (dashboard, signal engine, …)

Usage
-----
    client = AlpacaWSClient(
        api_key="...", secret_key="...",
        tickers=["AAPL", "TSLA", "SPY"],
        redis_url="redis://localhost:6379/0",
    )
    await client.run()   # blocks; use asyncio.create_task() for concurrent use
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Alpaca Streaming endpoints
# ---------------------------------------------------------------------------

_WS_URL_IEX = "wss://stream.data.alpaca.markets/v2/iex"   # free delayed
_WS_URL_SIP = "wss://stream.data.alpaca.markets/v2/sip"   # paid
_WS_URL_PAPER = "wss://paper-api.alpaca.markets/stream"    # paper trading events

# Default to IEX feed (free, 15-min delayed)
DEFAULT_WS_URL = _WS_URL_IEX

# Redis channel template: sentinel:bars:1m:<ticker>
REDIS_CHANNEL_TPL = "sentinel:bars:1m:{ticker}"


# ---------------------------------------------------------------------------
# Bar buffer — accumulates ticks into 1-min bars
# ---------------------------------------------------------------------------

@dataclass
class _MinuteBar:
    """Mutable accumulator for a single 1-minute bar."""
    ticker: str
    minute_ts: datetime          # floor of the minute (UTC, tz-aware)
    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: int = 0
    vwap_num: float = 0.0        # sum(price * size) — numerator
    vwap_den: float = 0.0        # sum(size)          — denominator
    trade_count: int = 0

    def update(self, price: float, size: int) -> None:
        if self.trade_count == 0:
            self.open = price
            self.high = price
            self.low = price
        else:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.vwap_num += price * size
        self.vwap_den += size
        self.trade_count += 1

    @property
    def vwap(self) -> Optional[float]:
        return round(self.vwap_num / self.vwap_den, 6) if self.vwap_den else None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "time": self.minute_ts.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low if self.low != float("inf") else self.close,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
            "source": "alpaca_ws",
        }


class BarBuffer:
    """Aggregate per-second trade ticks into completed 1-minute bars.

    When a tick arrives in a *new* minute the current bar is sealed and
    emitted via the ``on_bar`` callback before a fresh bar is started.
    """

    def __init__(self, on_bar) -> None:
        self._on_bar = on_bar  # async callable(bar: dict)
        self._bars: dict[str, _MinuteBar] = {}

    async def add_tick(self, ticker: str, price: float, size: int, ts: datetime) -> None:
        """Process a single trade tick."""
        minute_ts = ts.replace(second=0, microsecond=0)
        current = self._bars.get(ticker)

        if current is not None and current.minute_ts != minute_ts:
            # Minute rolled — emit the completed bar
            await self._emit(current)
            current = None

        if current is None:
            current = _MinuteBar(ticker=ticker, minute_ts=minute_ts)
            self._bars[ticker] = current

        current.update(price, size)

    async def flush_all(self) -> None:
        """Emit all open bars (called on shutdown or end-of-session)."""
        for bar in list(self._bars.values()):
            await self._emit(bar)
        self._bars.clear()

    async def _emit(self, bar: _MinuteBar) -> None:
        try:
            await self._on_bar(bar.to_dict())
        except Exception as exc:
            logger.error("BarBuffer emit error", ticker=bar.ticker, error=str(exc))


# ---------------------------------------------------------------------------
# Redis publisher (optional — gracefully skipped when redis is absent)
# ---------------------------------------------------------------------------

class _RedisPublisher:
    """Thin wrapper around aioredis / redis-py async that publishes bar dicts."""

    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._client = None

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self._url, decode_responses=True)
            await self._client.ping()
            logger.info("Redis connected", url=self._url)
        except ImportError:
            logger.warning("redis package not installed — pub/sub disabled")
            self._client = None
        except Exception as exc:
            logger.error("Redis connection failed", error=str(exc))
            self._client = None

    async def publish(self, ticker: str, bar: dict) -> None:
        if self._client is None:
            return
        channel = REDIS_CHANNEL_TPL.format(ticker=ticker)
        try:
            await self._client.publish(channel, json.dumps(bar))
        except Exception as exc:
            logger.error("Redis publish error", ticker=ticker, error=str(exc))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Alpaca WebSocket client
# ---------------------------------------------------------------------------

class AlpacaWSClient:
    """WebSocket client for Alpaca market-data streams.

    Subscribes to trade ticks for *tickers*, buffers them into 1-minute
    OHLCV bars and publishes completed bars to Redis pub/sub.

    Parameters
    ----------
    api_key / secret_key:
        Alpaca credentials (paper account works fine for free IEX data).
    tickers:
        List of equity symbols to subscribe to.
    redis_url:
        Redis connection string.  Pass ``None`` to disable pub/sub.
    ws_url:
        WebSocket endpoint.  Defaults to the free IEX feed.
    reconnect_delay:
        Base delay in seconds before reconnecting after an error.
    max_reconnects:
        Maximum reconnect attempts before giving up.  ``0`` = unlimited.
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        tickers: list[str] | None = None,
        redis_url: Optional[str] = "redis://localhost:6379/0",
        ws_url: str = DEFAULT_WS_URL,
        reconnect_delay: float = 5.0,
        max_reconnects: int = 0,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._tickers = [t.upper() for t in (tickers or [])]
        self._ws_url = ws_url
        self._reconnect_delay = reconnect_delay
        self._max_reconnects = max_reconnects
        self._redis: Optional[_RedisPublisher] = (
            _RedisPublisher(redis_url) if redis_url else None
        )
        self._buffer = BarBuffer(on_bar=self._on_bar)
        self._running = False
        self._reconnect_count = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect, authenticate, subscribe and stream indefinitely."""
        if self._redis:
            await self._redis.connect()

        self._running = True
        while self._running:
            try:
                await self._stream_loop()
            except Exception as exc:
                self._reconnect_count += 1
                if self._max_reconnects and self._reconnect_count > self._max_reconnects:
                    logger.error(
                        "AlpacaWSClient: max reconnects reached — stopping",
                        reconnects=self._reconnect_count,
                    )
                    break
                logger.warning(
                    "AlpacaWSClient: stream error, reconnecting",
                    error=str(exc),
                    delay=self._reconnect_delay,
                    attempt=self._reconnect_count,
                )
                await asyncio.sleep(self._reconnect_delay)

        await self._buffer.flush_all()
        if self._redis:
            await self._redis.close()

    async def stop(self) -> None:
        """Signal the streaming loop to stop after the current reconnect cycle."""
        self._running = False
        await self._buffer.flush_all()

    # ------------------------------------------------------------------
    # Internal streaming logic
    # ------------------------------------------------------------------

    async def _stream_loop(self) -> None:
        """Single WebSocket session — auth, subscribe, receive loop."""
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "websockets package is required for AlpacaWSClient: pip install websockets"
            ) from exc

        logger.info("AlpacaWSClient: connecting", url=self._ws_url)
        async with websockets.connect(self._ws_url, ping_interval=20, ping_timeout=20) as ws:
            # 1. Wait for connected message
            msg = json.loads(await ws.recv())
            if not self._expect(msg, "connected"):
                raise RuntimeError(f"Unexpected connect message: {msg}")

            # 2. Authenticate
            await ws.send(json.dumps({
                "action": "auth",
                "key": self._api_key,
                "secret": self._secret_key,
            }))
            auth_resp = json.loads(await ws.recv())
            if not self._expect(auth_resp, "authenticated"):
                raise RuntimeError(f"Auth failed: {auth_resp}")
            logger.info("AlpacaWSClient: authenticated")

            # 3. Subscribe to trades
            if self._tickers:
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "trades": self._tickers,
                }))
                sub_resp = json.loads(await ws.recv())
                logger.info("AlpacaWSClient: subscribed", tickers=self._tickers, resp=sub_resp)

            # 4. Receive loop
            async for raw in ws:
                if not self._running:
                    break
                await self._handle_message(raw)

    @staticmethod
    def _expect(msg: list | dict, event_type: str) -> bool:
        """Return True if the first message in *msg* has the expected T field."""
        if isinstance(msg, list):
            return bool(msg) and msg[0].get("T") == event_type
        return msg.get("T") == event_type

    async def _handle_message(self, raw: str | bytes) -> None:
        """Parse a WebSocket frame and route to the bar buffer."""
        try:
            messages = json.loads(raw)
            if not isinstance(messages, list):
                messages = [messages]
            for msg in messages:
                msg_type = msg.get("T")
                if msg_type == "t":  # trade
                    await self._handle_trade(msg)
                elif msg_type == "b":  # minute bar (pre-computed by Alpaca)
                    await self._handle_alpaca_bar(msg)
                elif msg_type in ("error", "subscription"):
                    logger.debug("AlpacaWS message", type=msg_type, msg=msg)
        except json.JSONDecodeError as exc:
            logger.warning("AlpacaWS JSON decode error", error=str(exc))
        except Exception as exc:
            logger.error("AlpacaWS message handling error", error=str(exc))

    async def _handle_trade(self, msg: dict) -> None:
        """Process a trade tick — feed into the bar buffer."""
        ticker = msg.get("S", "")
        price = float(msg.get("p", 0))
        size = int(msg.get("s", 0))
        ts_str = msg.get("t", "")
        if not ticker or price <= 0:
            return
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            ts = datetime.now(tz=timezone.utc)
        await self._buffer.add_tick(ticker, price, size, ts)

    async def _handle_alpaca_bar(self, msg: dict) -> None:
        """Alpaca pre-built minute bars (``T=b``) — publish directly."""
        ticker = msg.get("S", "")
        if not ticker:
            return
        bar = {
            "ticker": ticker,
            "time": msg.get("t", ""),
            "open": msg.get("o"),
            "high": msg.get("h"),
            "low": msg.get("l"),
            "close": msg.get("c"),
            "volume": msg.get("v"),
            "vwap": msg.get("vw"),
            "trade_count": msg.get("n"),
            "source": "alpaca_ws_bar",
        }
        await self._on_bar(bar)

    async def _on_bar(self, bar: dict) -> None:
        """Called when a 1-minute bar is completed — publish to Redis."""
        ticker = bar.get("ticker", "")
        logger.debug(
            "AlpacaWS: 1m bar complete",
            ticker=ticker,
            time=bar.get("time"),
            close=bar.get("close"),
            volume=bar.get("volume"),
        )
        if self._redis:
            await self._redis.publish(ticker, bar)
