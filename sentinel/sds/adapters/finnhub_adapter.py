"""Finnhub adapter — real-time WebSocket quotes, news, estimates, earnings calendar."""
from __future__ import annotations
import asyncio
import json
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Awaitable
try:
    import finnhub
    import websockets
    _FINNHUB_AVAILABLE = True
except ImportError:
    finnhub = None  # type: ignore[assignment]
    websockets = None  # type: ignore[assignment]
    _FINNHUB_AVAILABLE = False
from sentinel.core.types import OHLCVBar, Quote, DataHealthEvent, DataSeverity
from sentinel.sds.base_adapter import BaseAdapter
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

FINNHUB_WS_URL = "wss://ws.finnhub.io"


class FinnhubAdapter(BaseAdapter):
    name = "finnhub"
    rate_limit_per_min = 60
    supports_realtime = True

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self._api_key = api_key
        self._client = finnhub.Client(api_key=api_key) if api_key else None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_subscriptions: set[str] = set()
        self._quote_handler: Optional[Callable[[Quote], Awaitable[None]]] = None

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        figi: Optional[str] = None,
    ) -> list[OHLCVBar]:
        if not self._client:
            return []
        await self._throttle()

        resolution_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30",
                          "1h": "60", "1d": "D", "1wk": "W", "1mo": "M"}
        resolution = resolution_map.get(interval, "D")
        loop = asyncio.get_event_loop()

        try:
            data = await loop.run_in_executor(
                None,
                lambda: self._client.stock_candles(
                    ticker, resolution,
                    int(start.timestamp()), int(end.timestamp())
                )
            )
        except Exception as exc:
            logger.error("Finnhub OHLCV error", ticker=ticker, error=str(exc))
            return []

        if data.get("s") != "ok" or not data.get("t"):
            return []

        bars = []
        for i, ts in enumerate(data["t"]):
            bars.append(OHLCVBar(
                time=datetime.utcfromtimestamp(ts),
                figi=figi or ticker,
                open=Decimal(str(data["o"][i])),
                high=Decimal(str(data["h"][i])),
                low=Decimal(str(data["l"][i])),
                close=Decimal(str(data["c"][i])),
                volume=int(data["v"][i]),
                source="finnhub",
            ))
        return bars

    async def fetch_news(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        """Company-specific news with sentiment."""
        if not self._client:
            return []
        await self._throttle()
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: self._client.company_news(ticker, _from=from_date, to=to_date)
            ) or []
        except Exception as exc:
            logger.error("Finnhub news error", ticker=ticker, error=str(exc))
            return []

    async def fetch_earnings_calendar(
        self, from_date: str, to_date: str
    ) -> list[dict]:
        if not self._client:
            return []
        await self._throttle()
        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(
                None, lambda: self._client.earnings_calendar(
                    _from=from_date, to=to_date, symbol="", international=False
                )
            )
            return data.get("earningsCalendar", [])
        except Exception as exc:
            logger.error("Finnhub earnings calendar error", error=str(exc))
            return []

    async def fetch_basic_financials(self, ticker: str) -> dict:
        if not self._client:
            return {}
        await self._throttle()
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: self._client.company_basic_financials(ticker, "all")
            ) or {}
        except Exception:
            return {}

    async def subscribe_quotes(
        self,
        symbols: list[str],
        handler: Callable[[Quote], Awaitable[None]],
    ) -> None:
        """Subscribe to real-time WebSocket quotes for a list of symbols."""
        self._quote_handler = handler
        self._ws_subscriptions.update(symbols)
        asyncio.create_task(self._ws_loop(symbols))

    async def _ws_loop(self, symbols: list[str]) -> None:
        url = f"{FINNHUB_WS_URL}?token={self._api_key}"
        while True:
            try:
                async with websockets.connect(url) as ws:
                    self._ws = ws
                    for sym in symbols:
                        await ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
                    logger.info("Finnhub WS connected", symbols=len(symbols))
                    async for raw in ws:
                        await self._handle_ws_message(raw)
            except Exception as exc:
                logger.warning("Finnhub WS disconnected, retrying", error=str(exc))
                await asyncio.sleep(5)

    async def _handle_ws_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            if msg.get("type") != "trade" or not self._quote_handler:
                return
            for trade in msg.get("data", []):
                quote = Quote(
                    time=datetime.utcfromtimestamp(trade["t"] / 1000),
                    figi=trade["s"],
                    last=Decimal(str(trade["p"])),
                    volume=int(trade.get("v", 0)),
                    source="finnhub",
                )
                await self._quote_handler(quote)
        except Exception as exc:
            logger.debug("WS parse error", error=str(exc))

    async def health_check(self) -> DataHealthEvent:
        if not self._client:
            return DataHealthEvent(
                time=datetime.utcnow(), adapter=self.name,
                event_type="no_key", severity=DataSeverity.WARNING,
                details={"message": "No Finnhub API key configured"}
            )
        try:
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(
                None, lambda: self._client.quote("AAPL")
            )
            if quote and quote.get("c"):
                return self._ok_event()
            return self._error_event("AAPL quote returned empty")
        except Exception as exc:
            return self._error_event(str(exc))
