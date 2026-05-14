"""Abstract base adapter — all data providers implement this interface."""
from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)
import httpx
import logging
from sentinel.core.types import OHLCVBar, DataHealthEvent, DataSeverity

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    pass


class AdapterError(Exception):
    pass


class BaseAdapter(ABC):
    name: str = "base"
    rate_limit_per_min: int = 60
    supports_realtime: bool = False
    supports_crypto: bool = False

    def __init__(self) -> None:
        self._call_times: list[float] = []

    @abstractmethod
    async def fetch_ohlcv(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        figi: Optional[str] = None,
    ) -> list[OHLCVBar]:
        ...

    @abstractmethod
    async def health_check(self) -> DataHealthEvent:
        ...

    async def _throttle(self) -> None:
        """Simple sliding window rate limiter — waits if over limit."""
        now = asyncio.get_event_loop().time()
        window = 60.0
        self._call_times = [t for t in self._call_times if now - t < window]
        if len(self._call_times) >= self.rate_limit_per_min:
            sleep_for = window - (now - self._call_times[0]) + 0.1
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        self._call_times.append(asyncio.get_event_loop().time())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _get(self, url: str, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, **kwargs)
            if resp.status_code == 429:
                raise RateLimitError(f"Rate limit hit on {self.name}")
            resp.raise_for_status()
            return resp

    def _ok_event(self) -> DataHealthEvent:
        return DataHealthEvent(
            time=datetime.utcnow(),
            adapter=self.name,
            event_type="ok",
            severity=DataSeverity.INFO,
            details={},
        )

    def _error_event(self, error: str) -> DataHealthEvent:
        return DataHealthEvent(
            time=datetime.utcnow(),
            adapter=self.name,
            event_type="error",
            severity=DataSeverity.CRITICAL,
            details={"error": error},
        )
