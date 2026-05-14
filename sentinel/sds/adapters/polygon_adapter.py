"""Polygon.io adapter — institutional-grade real-time + historical US equities/options/crypto."""
from __future__ import annotations
import asyncio
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Callable, Awaitable
import httpx
from sentinel.core.types import OHLCVBar, Quote, DataHealthEvent
from sentinel.sds.base_adapter import BaseAdapter
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

POLYGON_BASE = "https://api.polygon.io"

POLYGON_INTERVAL_MAP = {
    "1m": ("1", "minute"), "5m": ("5", "minute"), "15m": ("15", "minute"),
    "30m": ("30", "minute"), "1h": ("1", "hour"), "4h": ("4", "hour"),
    "1d": ("1", "day"), "1wk": ("1", "week"), "1mo": ("1", "month"),
}


class PolygonAdapter(BaseAdapter):
    name = "polygon"
    rate_limit_per_min = 300  # Starter plan: 5/min; Pro: unlimited — set conservatively
    supports_realtime = True
    supports_crypto = True
    supports_options = True

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self._api_key = api_key
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        figi: Optional[str] = None,
    ) -> list[OHLCVBar]:
        if not self._api_key:
            return []
        await self._throttle()
        mult, span = POLYGON_INTERVAL_MAP.get(interval, ("1", "day"))
        from_str = start.date().isoformat()
        to_str = end.date().isoformat()
        url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker.upper()}/range/{mult}/{span}/{from_str}/{to_str}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self._api_key}

        try:
            resp = await self._get(url, params=params)
            data = resp.json()
        except Exception as exc:
            logger.error("Polygon OHLCV error", ticker=ticker, error=str(exc))
            return []

        results = data.get("results") or []
        bars = []
        for r in results:
            ts = r.get("t", 0)
            bar_time = datetime.utcfromtimestamp(ts / 1000)
            if bar_time > end:
                break
            bars.append(OHLCVBar(
                time=bar_time,
                figi=figi or ticker,
                open=Decimal(str(r.get("o", 0))),
                high=Decimal(str(r.get("h", 0))),
                low=Decimal(str(r.get("l", 0))),
                close=Decimal(str(r.get("c", 0))),
                volume=int(r.get("v", 0)),
                vwap=Decimal(str(r["vw"])) if r.get("vw") else None,
                source="polygon",
            ))
        logger.info("Polygon fetched", ticker=ticker, bars=len(bars))
        return bars

    async def fetch_snapshot(self, ticker: str) -> dict:
        """Intraday snapshot: last quote, last trade, minute/day/prev bars."""
        if not self._api_key:
            return {}
        await self._throttle()
        url = f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"
        try:
            resp = await self._get(url, params={"apiKey": self._api_key})
            return resp.json().get("ticker", {})
        except Exception as exc:
            logger.error("Polygon snapshot error", ticker=ticker, error=str(exc))
            return {}

    async def fetch_trades(
        self, ticker: str, trade_date: date, limit: int = 50000
    ) -> list[dict]:
        """Fetch all trades for a ticker on a given date (tick data)."""
        if not self._api_key:
            return []
        await self._throttle()
        url = f"{POLYGON_BASE}/v3/trades/{ticker.upper()}"
        params = {
            "timestamp.gte": f"{trade_date.isoformat()}T00:00:00Z",
            "timestamp.lte": f"{trade_date.isoformat()}T23:59:59Z",
            "limit": limit,
            "apiKey": self._api_key,
        }
        try:
            resp = await self._get(url, params=params)
            return resp.json().get("results", [])
        except Exception as exc:
            logger.error("Polygon trades error", ticker=ticker, error=str(exc))
            return []

    async def fetch_quotes_nbbo(
        self, ticker: str, trade_date: date, limit: int = 50000
    ) -> list[dict]:
        """NBBO quote stream for a date — L1 bid/ask history."""
        if not self._api_key:
            return []
        await self._throttle()
        url = f"{POLYGON_BASE}/v3/quotes/{ticker.upper()}"
        params = {
            "timestamp.gte": f"{trade_date.isoformat()}T00:00:00Z",
            "timestamp.lte": f"{trade_date.isoformat()}T23:59:59Z",
            "limit": limit,
            "apiKey": self._api_key,
        }
        try:
            resp = await self._get(url, params=params)
            return resp.json().get("results", [])
        except Exception as exc:
            logger.error("Polygon NBBO error", ticker=ticker, error=str(exc))
            return []

    async def fetch_options_chain(
        self,
        underlying: str,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        contract_type: Optional[str] = None,
        limit: int = 250,
    ) -> list[dict]:
        """Fetch options contracts with greeks from Polygon."""
        if not self._api_key:
            return []
        await self._throttle()
        url = f"{POLYGON_BASE}/v3/reference/options/contracts"
        params: dict = {
            "underlying_ticker": underlying.upper(),
            "limit": limit,
            "apiKey": self._api_key,
        }
        if expiration_date_gte:
            params["expiration_date.gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date.lte"] = expiration_date_lte
        if contract_type:
            params["contract_type"] = contract_type
        try:
            resp = await self._get(url, params=params)
            return resp.json().get("results", [])
        except Exception as exc:
            logger.error("Polygon options error", underlying=underlying, error=str(exc))
            return []

    async def fetch_crypto_snapshot(self, pair: str) -> dict:
        """Real-time crypto snapshot (BTC-USD style pair)."""
        if not self._api_key:
            return {}
        await self._throttle()
        url = f"{POLYGON_BASE}/v2/snapshot/locale/global/markets/crypto/tickers/X:{pair.replace('-', '')}"
        try:
            resp = await self._get(url, params={"apiKey": self._api_key})
            return resp.json().get("ticker", {})
        except Exception as exc:
            logger.error("Polygon crypto snapshot error", pair=pair, error=str(exc))
            return {}

    async def search_tickers(
        self, query: str, asset_class: str = "stocks", limit: int = 20
    ) -> list[dict]:
        """Search Polygon reference data for matching tickers."""
        if not self._api_key:
            return []
        await self._throttle()
        url = f"{POLYGON_BASE}/v3/reference/tickers"
        params = {
            "search": query,
            "market": asset_class,
            "active": "true",
            "limit": limit,
            "apiKey": self._api_key,
        }
        try:
            resp = await self._get(url, params=params)
            return resp.json().get("results", [])
        except Exception as exc:
            logger.error("Polygon ticker search error", query=query, error=str(exc))
            return []

    async def fetch_ticker_details(self, ticker: str) -> dict:
        """Fetch fundamental reference data for a ticker (name, SIC, exchange, etc.)."""
        if not self._api_key:
            return {}
        await self._throttle()
        url = f"{POLYGON_BASE}/v3/reference/tickers/{ticker.upper()}"
        try:
            resp = await self._get(url, params={"apiKey": self._api_key})
            return resp.json().get("results", {})
        except Exception as exc:
            logger.error("Polygon ticker details error", ticker=ticker, error=str(exc))
            return {}

    async def health_check(self) -> DataHealthEvent:
        if not self._api_key:
            from sentinel.core.types import DataSeverity
            from datetime import datetime as _dt
            return DataHealthEvent(
                time=_dt.utcnow(), adapter=self.name,
                event_type="no_key", severity=DataSeverity.WARNING,
                details={"message": "No Polygon API key configured"},
            )
        try:
            snap = await self.fetch_snapshot("SPY")
            if snap:
                return self._ok_event()
            return self._error_event("SPY snapshot empty")
        except Exception as exc:
            return self._error_event(str(exc))
