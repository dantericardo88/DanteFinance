"""FRED adapter — 765K+ macroeconomic time series from St. Louis Fed."""
from __future__ import annotations
import asyncio
from datetime import datetime, date
from decimal import Decimal
from typing import Optional
import pandas as pd
from fredapi import Fred
from sentinel.core.types import MacroDataPoint, DataHealthEvent
from sentinel.sds.base_adapter import BaseAdapter
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# Key series always ingested
CORE_FRED_SERIES = {
    # Yield curve
    "DGS1MO": "Treasury 1-Month",
    "DGS3MO": "Treasury 3-Month",
    "DGS6MO": "Treasury 6-Month",
    "DGS1": "Treasury 1-Year",
    "DGS2": "Treasury 2-Year",
    "DGS5": "Treasury 5-Year",
    "DGS10": "Treasury 10-Year",
    "DGS20": "Treasury 20-Year",
    "DGS30": "Treasury 30-Year",
    # Spreads
    "T10Y2Y": "10Y-2Y Spread",
    "T10Y3M": "10Y-3M Spread",
    "T5YIFR": "5Y5Y Forward Inflation",
    # Inflation
    "T5YIE": "5-Year Breakeven Inflation",
    "T10YIE": "10-Year Breakeven Inflation",
    "CPIAUCSL": "CPI All Items",
    "CPILFESL": "Core CPI",
    "PCEPI": "PCE Price Index",
    "PCEPILFE": "Core PCE",
    # Employment
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    "ICSA": "Initial Jobless Claims",
    # Growth
    "GDP": "Nominal GDP",
    "GDPC1": "Real GDP",
    # Volatility / Risk
    "VIXCLS": "VIX",
    "BAMLC0A0CM": "IG Credit Spread",
    "BAMLH0A0HYM2": "HY Credit Spread",
    # Money / Credit
    "M2SL": "M2 Money Supply",
    "FEDFUNDS": "Fed Funds Rate",
    "SOFR": "SOFR",
    "DXY": "USD Index (via FRED proxy)",
}


class FREDAdapter(BaseAdapter):
    name = "fred"
    rate_limit_per_min = 120  # FRED allows 120/min with API key

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self._fred = Fred(api_key=api_key) if api_key else Fred()
        self._cache: dict[str, pd.Series] = {}

    async def fetch_series(
        self,
        series_id: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        vintage_date: Optional[date] = None,
    ) -> list[MacroDataPoint]:
        """Fetch a FRED time series. vintage_date enables ALFRED point-in-time."""
        await self._throttle()
        loop = asyncio.get_event_loop()

        def _fetch():
            kwargs: dict = {}
            if start:
                kwargs["observation_start"] = start.isoformat()
            if end:
                kwargs["observation_end"] = end.isoformat()
            if vintage_date:
                kwargs["vintage_dates"] = vintage_date.isoformat()
            return self._fred.get_series(series_id, **kwargs)

        try:
            series = await loop.run_in_executor(None, _fetch)
        except Exception as exc:
            logger.error("FRED fetch failed", series_id=series_id, error=str(exc))
            return []

        self._cache[series_id] = series
        points = []
        for ts, val in series.items():
            if pd.isna(val):
                continue
            points.append(
                MacroDataPoint(
                    time=pd.Timestamp(ts).to_pydatetime(),
                    series_id=series_id,
                    value=Decimal(str(round(float(val), 8))),
                    vintage=datetime.combine(vintage_date, datetime.min.time()) if vintage_date else None,
                )
            )
        logger.info("FRED series loaded", series_id=series_id, points=len(points))
        return points

    async def fetch_ohlcv(self, ticker, start, end, interval="1d", figi=None):
        # FRED doesn't have OHLCV — map series_id to macro data
        points = await self.fetch_series(ticker, start.date(), end.date())
        return []  # Not applicable; use fetch_series directly

    async def fetch_core_series(self) -> dict[str, list[MacroDataPoint]]:
        """Fetch all core FRED series in parallel."""
        tasks = {sid: self.fetch_series(sid) for sid in CORE_FRED_SERIES}
        results = {}
        for sid, coro in tasks.items():
            try:
                results[sid] = await coro
            except Exception as exc:
                logger.warning("Core FRED series failed", series_id=sid, error=str(exc))
                results[sid] = []
        return results

    async def get_series_info(self, series_id: str) -> dict:
        loop = asyncio.get_event_loop()
        try:
            info = await loop.run_in_executor(
                None, lambda: self._fred.get_series_info(series_id)
            )
            return info.to_dict()
        except Exception:
            return {}

    async def health_check(self) -> DataHealthEvent:
        try:
            pts = await self.fetch_series("DGS10", start=date(2024, 1, 1))
            if pts:
                return self._ok_event()
            return self._error_event("DGS10 returned 0 points")
        except Exception as exc:
            return self._error_event(str(exc))
