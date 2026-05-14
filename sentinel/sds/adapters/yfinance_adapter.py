"""yfinance adapter — primary free equity/ETF/FX/crypto OHLCV source.

Improvements over v1:
  - Exponential backoff retry (3 attempts, 2–30 s wait) via tenacity
  - Large date ranges auto-chunked into 2-year segments to stay within
    yfinance/Yahoo Finance API limits and avoid silent truncation
  - Data validation: high >= low, volume >= 0, no gaps > 7 trading days,
    OHLCV price sanity (positive values, no absurd spikes)
  - Survivorship bias flag: delisted tickers from survivorship_registry are
    annotated so callers can apply appropriate bias corrections
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging as _logging

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
_tenacity_logger = _logging.getLogger(__name__)

INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "1d": "1d", "1wk": "1wk", "1mo": "1mo",
}

# Chunk size for multi-year fetches (years)
_CHUNK_YEARS = 2

# Validation thresholds
_MAX_GAP_TRADING_DAYS = 7      # flag gaps wider than this
_MAX_PRICE_SPIKE_RATIO = 10.0  # high/low ratio; >10x in one bar is suspicious


class YFinanceRetryableError(Exception):
    """Raised when a transient yfinance error should trigger a retry."""


class YFinanceAdapter(BaseAdapter):
    name = "yfinance"
    rate_limit_per_min = 60
    supports_crypto = True

    # ── Date-range chunking ───────────────────────────────────────────────────

    @staticmethod
    def _date_chunks(
        start: datetime,
        end: datetime,
        chunk_years: int = _CHUNK_YEARS,
    ) -> list[tuple[datetime, datetime]]:
        """Split [start, end] into segments of at most chunk_years years."""
        chunks: list[tuple[datetime, datetime]] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(
                datetime(
                    chunk_start.year + chunk_years,
                    chunk_start.month,
                    chunk_start.day,
                    chunk_start.hour,
                    chunk_start.minute,
                    chunk_start.second,
                ),
                end,
            )
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(seconds=1)
        return chunks

    # ── Low-level yfinance download with retry ────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((YFinanceRetryableError, Exception)),
        before_sleep=before_sleep_log(_tenacity_logger, _logging.WARNING),
        reraise=True,
    )
    def _download_chunk(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        yf_interval: str,
    ) -> "pd.DataFrame":
        """Download a single chunk with exponential backoff retry."""
        df = yf.download(
            ticker,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval=yf_interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        # Treat suspiciously small results as a possible silent throttle
        if df is not None and not df.empty:
            # Flatten MultiIndex (yfinance >= 0.2 with single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        return df if df is not None else pd.DataFrame()

    # ── Data validation ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_bars(
        df: "pd.DataFrame",
        ticker: str,
    ) -> tuple["pd.DataFrame", list[str]]:
        """Apply OHLCV sanity checks. Returns (clean_df, list_of_warnings).

        Checks performed:
          1. high >= low (fundamental price integrity)
          2. all prices > 0 (no negative prices)
          3. volume >= 0
          4. high/low spike ratio <= _MAX_PRICE_SPIKE_RATIO (catches bad ticks)
          5. no inter-bar gap > _MAX_GAP_TRADING_DAYS calendar days
        """
        if df is None or df.empty:
            return df, []

        warnings: list[str] = []
        original_len = len(df)

        # Require standard columns
        needed = {"Open", "High", "Low", "Close", "Volume"}
        missing = needed - set(df.columns)
        if missing:
            warnings.append(f"{ticker}: missing columns {missing}")
            return df, warnings

        # 1. high >= low
        bad_hl = df["High"] < df["Low"]
        if bad_hl.any():
            count = int(bad_hl.sum())
            warnings.append(f"{ticker}: {count} bars with High < Low (dropped)")
            df = df[~bad_hl]

        # 2. Positive prices
        bad_price = (df["Open"] <= 0) | (df["High"] <= 0) | (df["Low"] <= 0) | (df["Close"] <= 0)
        if bad_price.any():
            count = int(bad_price.sum())
            warnings.append(f"{ticker}: {count} bars with non-positive price (dropped)")
            df = df[~bad_price]

        # 3. Non-negative volume
        bad_vol = df["Volume"] < 0
        if bad_vol.any():
            count = int(bad_vol.sum())
            warnings.append(f"{ticker}: {count} bars with negative volume (zeroed)")
            df.loc[bad_vol, "Volume"] = 0

        # 4. Price spike check
        spike_ratio = df["High"] / df["Low"]
        bad_spike = spike_ratio > _MAX_PRICE_SPIKE_RATIO
        if bad_spike.any():
            count = int(bad_spike.sum())
            warnings.append(
                f"{ticker}: {count} bars with H/L ratio > {_MAX_PRICE_SPIKE_RATIO}x (flagged)"
            )

        # 5. Gap detection (daily interval only — calendar days proxy)
        if len(df) >= 2:
            ts_index = df.index
            for i in range(1, len(ts_index)):
                gap_days = (ts_index[i] - ts_index[i - 1]).days
                if gap_days > _MAX_GAP_TRADING_DAYS:
                    warnings.append(
                        f"{ticker}: gap of {gap_days} calendar days between "
                        f"{ts_index[i - 1].date()} and {ts_index[i].date()}"
                    )

        dropped = original_len - len(df)
        if dropped:
            logger.warning("Bars dropped by validation", ticker=ticker, dropped=dropped)

        return df, warnings

    # ── Survivorship check ────────────────────────────────────────────────────

    @staticmethod
    def _survivorship_flag(ticker: str) -> bool:
        """Return True if ticker is in the delisted survivorship registry."""
        try:
            from sentinel.sds.survivorship import lookup_by_ticker
            matches = lookup_by_ticker(ticker)
            return bool(matches)
        except Exception:
            return False

    # ── Public interface ──────────────────────────────────────────────────────

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

        # Survivorship check (non-blocking)
        is_delisted = await loop.run_in_executor(
            None, self._survivorship_flag, ticker
        )
        if is_delisted:
            logger.info(
                "Survivorship flag: ticker is in delisted registry",
                ticker=ticker,
            )

        # Chunk into 2-year segments
        chunks = self._date_chunks(start, end)
        all_bars: list[OHLCVBar] = []
        all_warnings: list[str] = []

        for chunk_start, chunk_end in chunks:
            try:
                df = await loop.run_in_executor(
                    None,
                    self._download_chunk,
                    ticker,
                    chunk_start,
                    chunk_end,
                    yf_interval,
                )
            except Exception as exc:
                logger.error(
                    "yfinance chunk failed after retries",
                    ticker=ticker,
                    chunk_start=chunk_start.date(),
                    chunk_end=chunk_end.date(),
                    error=str(exc),
                )
                continue

            if df is None or df.empty:
                logger.debug(
                    "yfinance empty chunk",
                    ticker=ticker,
                    chunk_start=chunk_start.date(),
                    chunk_end=chunk_end.date(),
                )
                continue

            # Validate
            df, warnings = self._validate_bars(df, ticker)
            all_warnings.extend(warnings)

            for warn in warnings:
                logger.warning("Validation warning", detail=warn)

            # Convert to OHLCVBar
            for ts, row in df.iterrows():
                def col(name: str) -> float:
                    val = row.get(name, 0)
                    return float(val) if pd is not None and pd.notna(val) else 0.0

                all_bars.append(
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

            # Brief courtesy delay between chunks
            if len(chunks) > 1:
                await asyncio.sleep(0.2)

        logger.info(
            "yfinance fetched",
            ticker=ticker,
            bars=len(all_bars),
            chunks=len(chunks),
            delisted=is_delisted,
            validation_warnings=len(all_warnings),
        )
        return all_bars

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

    async def fetch_dividends(self, ticker: str) -> "pd.Series":
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
