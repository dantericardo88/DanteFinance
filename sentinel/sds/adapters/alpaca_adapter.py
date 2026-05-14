"""Alpaca adapter — free real-time + historical US equities, paper/live trading data."""
from __future__ import annotations
import asyncio
from datetime import datetime, date
from decimal import Decimal
from typing import Optional
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, StockLatestQuoteRequest, StockSnapshotRequest,
    CryptoBarsRequest, CryptoLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from sentinel.core.types import OHLCVBar, Quote, DataHealthEvent
from sentinel.sds.base_adapter import BaseAdapter
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

ALPACA_TF_MAP = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "30m": TimeFrame(30, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
    "1wk": TimeFrame(1, TimeFrameUnit.Week),
    "1mo": TimeFrame(1, TimeFrameUnit.Month),
}


class AlpacaAdapter(BaseAdapter):
    name = "alpaca"
    rate_limit_per_min = 200
    supports_realtime = True
    supports_crypto = True

    def __init__(self, api_key: str = "", secret_key: str = "") -> None:
        super().__init__()
        self._api_key = api_key
        self._secret_key = secret_key
        # Both clients work without keys in delayed-data mode
        self._stock_client = StockHistoricalDataClient(
            api_key or None, secret_key or None
        )
        self._crypto_client = CryptoHistoricalDataClient(
            api_key or None, secret_key or None
        )

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        figi: Optional[str] = None,
    ) -> list[OHLCVBar]:
        await self._throttle()
        tf = ALPACA_TF_MAP.get(interval, TimeFrame(1, TimeFrameUnit.Day))
        loop = asyncio.get_event_loop()

        is_crypto = "/" in ticker or ticker.endswith("USD") and len(ticker) >= 6

        try:
            if is_crypto:
                req = CryptoBarsRequest(symbol_or_symbols=ticker, timeframe=tf,
                                        start=start, end=end)
                bars_df = await loop.run_in_executor(
                    None, lambda: self._crypto_client.get_crypto_bars(req).df
                )
            else:
                req = StockBarsRequest(symbol_or_symbols=ticker, timeframe=tf,
                                       start=start, end=end, adjustment="all")
                bars_df = await loop.run_in_executor(
                    None, lambda: self._stock_client.get_stock_bars(req).df
                )
        except Exception as exc:
            logger.error("Alpaca OHLCV error", ticker=ticker, error=str(exc))
            return []

        if bars_df is None or bars_df.empty:
            return []

        # Drop symbol level from multi-index if present
        if hasattr(bars_df.index, "levels"):
            bars_df = bars_df.xs(ticker, level="symbol") if ticker in bars_df.index.get_level_values("symbol") else bars_df

        bars = []
        for ts, row in bars_df.iterrows():
            import pandas as pd
            bar_time = pd.Timestamp(ts).to_pydatetime()
            bars.append(OHLCVBar(
                time=bar_time,
                figi=figi or ticker,
                open=Decimal(str(round(float(row["open"]), 6))),
                high=Decimal(str(round(float(row["high"]), 6))),
                low=Decimal(str(round(float(row["low"]), 6))),
                close=Decimal(str(round(float(row["close"]), 6))),
                volume=int(row.get("volume", 0)),
                vwap=Decimal(str(round(float(row["vwap"]), 6))) if row.get("vwap") else None,
                source="alpaca",
            ))
        logger.info("Alpaca fetched", ticker=ticker, bars=len(bars))
        return bars

    async def fetch_latest_quote(self, ticker: str) -> dict:
        """Return latest NBBO quote for a US equity."""
        await self._throttle()
        loop = asyncio.get_event_loop()
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            result = await loop.run_in_executor(
                None, lambda: self._stock_client.get_stock_latest_quote(req)
            )
            quote = result.get(ticker)
            if not quote:
                return {}
            return {
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "bid_size": quote.bid_size,
                "ask_size": quote.ask_size,
                "timestamp": quote.timestamp.isoformat(),
            }
        except Exception as exc:
            logger.error("Alpaca quote error", ticker=ticker, error=str(exc))
            return {}

    async def fetch_snapshot(self, ticker: str) -> dict:
        """Full snapshot: latest bar, trade, quote for a ticker."""
        await self._throttle()
        loop = asyncio.get_event_loop()
        try:
            req = StockSnapshotRequest(symbol_or_symbols=ticker)
            result = await loop.run_in_executor(
                None, lambda: self._stock_client.get_stock_snapshot(req)
            )
            snap = result.get(ticker)
            if not snap:
                return {}
            return {
                "ticker": ticker,
                "latest_trade_price": float(snap.latest_trade.price) if snap.latest_trade else None,
                "latest_quote_bid": float(snap.latest_quote.bid_price) if snap.latest_quote else None,
                "latest_quote_ask": float(snap.latest_quote.ask_price) if snap.latest_quote else None,
                "minute_bar_close": float(snap.minute_bar.close) if snap.minute_bar else None,
                "daily_bar_close": float(snap.daily_bar.close) if snap.daily_bar else None,
            }
        except Exception as exc:
            logger.error("Alpaca snapshot error", ticker=ticker, error=str(exc))
            return {}

    async def fetch_crypto_quote(self, pair: str) -> dict:
        """Latest quote for a crypto pair (e.g. BTC/USD)."""
        await self._throttle()
        loop = asyncio.get_event_loop()
        try:
            req = CryptoLatestQuoteRequest(symbol_or_symbols=pair)
            result = await loop.run_in_executor(
                None, lambda: self._crypto_client.get_crypto_latest_quote(req)
            )
            q = result.get(pair)
            if not q:
                return {}
            return {
                "bid": float(q.bid_price),
                "ask": float(q.ask_price),
                "bid_size": float(q.bid_size),
                "ask_size": float(q.ask_size),
            }
        except Exception as exc:
            logger.error("Alpaca crypto quote error", pair=pair, error=str(exc))
            return {}

    async def health_check(self) -> DataHealthEvent:
        try:
            snap = await self.fetch_snapshot("SPY")
            if snap:
                return self._ok_event()
            return self._error_event("SPY snapshot empty from Alpaca")
        except Exception as exc:
            return self._error_event(str(exc))
