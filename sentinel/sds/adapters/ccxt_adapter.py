"""CCXT adapter — unified interface for 100+ crypto exchanges."""
from __future__ import annotations
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional
try:
    import ccxt.async_support as ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    ccxt = None  # type: ignore[assignment]
    _CCXT_AVAILABLE = False
from sentinel.core.types import OHLCVBar, DataHealthEvent
from sentinel.sds.base_adapter import BaseAdapter
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXCHANGES = [
    "binance", "coinbase", "kraken", "okx", "bybit",
    "bitfinex", "hyperliquid", "kucoin", "gate",
]

CCXT_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1wk": "1w",
}


class CCXTAdapter(BaseAdapter):
    name = "ccxt"
    rate_limit_per_min = 120
    supports_crypto = True

    def __init__(self, exchange_id: str = "binance", api_key: str = "", secret: str = "") -> None:
        super().__init__()
        self._exchange_id = exchange_id
        exchange_class = getattr(ccxt, exchange_id, None)
        if not exchange_class:
            raise ValueError(f"Unknown exchange: {exchange_id}")
        kwargs: dict = {"enableRateLimit": True}
        if api_key:
            kwargs["apiKey"] = api_key
        if secret:
            kwargs["secret"] = secret
        self._exchange: ccxt.Exchange = exchange_class(kwargs)

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        figi: Optional[str] = None,
    ) -> list[OHLCVBar]:
        symbol = ticker.replace("-", "/").upper()
        tf = CCXT_INTERVAL_MAP.get(interval, "1d")
        since_ms = int(start.timestamp() * 1000)
        limit = min(1000, int((end.timestamp() - start.timestamp()) / _tf_to_seconds(tf)))

        try:
            data = await self._exchange.fetch_ohlcv(symbol, tf, since=since_ms, limit=limit)
        except Exception as exc:
            logger.error("CCXT OHLCV error", exchange=self._exchange_id, symbol=symbol, error=str(exc))
            return []

        bars = []
        for row in data:
            ts_ms, o, h, l, c, v = row
            bar_time = datetime.utcfromtimestamp(ts_ms / 1000)
            if bar_time > end:
                break
            bars.append(OHLCVBar(
                time=bar_time,
                figi=figi or symbol,
                open=Decimal(str(o)),
                high=Decimal(str(h)),
                low=Decimal(str(l)),
                close=Decimal(str(c)),
                volume=Decimal(str(v)),
                source=f"ccxt:{self._exchange_id}",
            ))
        return bars

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """Fetch current order book depth."""
        try:
            return await self._exchange.fetch_order_book(symbol.upper(), limit)
        except Exception as exc:
            logger.error("CCXT orderbook error", symbol=symbol, error=str(exc))
            return {}

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker (bid/ask/last/volume)."""
        try:
            return await self._exchange.fetch_ticker(symbol.upper())
        except Exception as exc:
            logger.error("CCXT ticker error", symbol=symbol, error=str(exc))
            return {}

    async def fetch_funding_rate(self, symbol: str) -> dict:
        """Fetch perpetual futures funding rate."""
        try:
            return await self._exchange.fetch_funding_rate(symbol.upper())
        except Exception:
            return {}

    async def fetch_markets(self) -> list[dict]:
        """Fetch all available markets for this exchange."""
        try:
            return await self._exchange.fetch_markets()
        except Exception as exc:
            logger.error("CCXT markets error", exchange=self._exchange_id, error=str(exc))
            return []

    async def close(self) -> None:
        await self._exchange.close()

    async def health_check(self) -> DataHealthEvent:
        try:
            ticker = await self.fetch_ticker("BTC/USDT")
            if ticker.get("last"):
                return self._ok_event()
            return self._error_event("BTC/USDT ticker empty")
        except Exception as exc:
            return self._error_event(str(exc))


def _tf_to_seconds(tf: str) -> int:
    mapping = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
               "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
    return mapping.get(tf, 86400)
