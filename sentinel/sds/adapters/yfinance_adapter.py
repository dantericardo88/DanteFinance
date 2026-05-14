"""yfinance adapter — primary free equity/ETF/FX/crypto OHLCV source."""
from __future__ import annotations
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional
try:
    import pandas as pd
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    yf = None  # type: ignore[assignment]
    _YF_AVAILABLE = False
from sentinel.core.types import OHLCVBar, DataHealthEvent
from sentinel.sds.base_adapter import BaseAdapter
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "1d": "1d", "1wk": "1wk", "1mo": "1mo",
}


class YFinanceAdapter(BaseAdapter):
    name = "yfinance"
    rate_limit_per_min = 60
    supports_crypto = True

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        figi: Optional[str] = None,
    ) -> list[OHLCVBar]:
        await self._throttle()
        yf_interval = INTERVAL_MAP.get(interval, "1d")
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: yf.download(
                ticker,
                start=start.date(),
                end=end.date(),
                interval=yf_interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            ),
        )
        if df.empty:
            logger.warning("yfinance returned empty data", ticker=ticker)
            return []

        bars = []
        for ts, row in df.iterrows():
            # Handle MultiIndex columns from yfinance >= 0.2
            def col(name: str) -> float:
                val = row[name] if name in row.index else row.get((name, ticker), 0)
                return float(val) if pd.notna(val) else 0.0

            bars.append(
                OHLCVBar(
                    time=pd.Timestamp(ts).to_pydatetime(),
                    figi=figi or ticker,
                    open=Decimal(str(round(col("Open"), 6))),
                    high=Decimal(str(round(col("High"), 6))),
                    low=Decimal(str(round(col("Low"), 6))),
                    close=Decimal(str(round(col("Close"), 6))),
                    volume=int(col("Volume")),
                    source="yfinance",
                )
            )
        logger.info("yfinance fetched", ticker=ticker, bars=len(bars))
        return bars

    async def fetch_info(self, ticker: str) -> dict:
        """Return Ticker.info dict — fundamentals, sector, market cap, etc."""
        await self._throttle()
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).info)
        return info or {}

    async def fetch_options_chain(self, ticker: str, expiry: Optional[str] = None) -> dict:
        """Return options chain for nearest expiry or specified date."""
        await self._throttle()
        loop = asyncio.get_event_loop()
        t = yf.Ticker(ticker)
        expiries = await loop.run_in_executor(None, lambda: t.options)
        if not expiries:
            return {}
        target = expiry or expiries[0]
        chain = await loop.run_in_executor(None, lambda: t.option_chain(target))
        return {
            "expiry": target,
            "calls": chain.calls.to_dict("records"),
            "puts": chain.puts.to_dict("records"),
        }

    async def fetch_dividends(self, ticker: str) -> pd.Series:
        """Return historical dividends."""
        await self._throttle()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: yf.Ticker(ticker).dividends
        )

    async def health_check(self) -> DataHealthEvent:
        try:
            bars = await self.fetch_ohlcv(
                "SPY",
                datetime(2024, 1, 2),
                datetime(2024, 1, 5),
            )
            if bars:
                return self._ok_event()
            return self._error_event("SPY returned 0 bars")
        except Exception as exc:
            return self._error_event(str(exc))
