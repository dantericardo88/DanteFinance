"""Data health monitor — every adapter emits DataHealthEvents; this module tracks them."""
from __future__ import annotations
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from sentinel.core.types import DataHealthEvent, DataSeverity
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# Maximum age before a data series is considered stale (per asset class)
STALENESS_THRESHOLDS: dict[str, timedelta] = {
    "equity_ohlcv_daily": timedelta(days=2),       # weekends add 1 day
    "equity_ohlcv_intraday": timedelta(hours=1),
    "quote": timedelta(minutes=5),
    "edgar": timedelta(days=1),
    "fred": timedelta(days=7),
    "cot": timedelta(days=8),                       # published weekly
    "crypto": timedelta(minutes=10),
    "news": timedelta(hours=2),
}


class DataHealthMonitor:
    """
    Tracks the last-seen timestamp for each (adapter, series) pair.
    Fires DataHealthEvent when staleness/gap/drift is detected.
    """

    def __init__(self) -> None:
        self._last_seen: dict[str, datetime] = {}
        self._event_counts: dict[str, int] = defaultdict(int)
        self._running = False

    def record(self, adapter: str, series_key: str = "default") -> None:
        key = f"{adapter}:{series_key}"
        self._last_seen[key] = datetime.utcnow()

    def get_last_seen(self, adapter: str, series_key: str = "default") -> Optional[datetime]:
        return self._last_seen.get(f"{adapter}:{series_key}")

    def check_staleness(
        self,
        adapter: str,
        series_key: str = "default",
        data_type: str = "equity_ohlcv_daily",
    ) -> Optional[DataHealthEvent]:
        last = self.get_last_seen(adapter, series_key)
        if last is None:
            return None

        threshold = STALENESS_THRESHOLDS.get(data_type, timedelta(days=1))
        age = datetime.utcnow() - last

        if age > threshold * 2:
            severity = DataSeverity.CRITICAL
        elif age > threshold:
            severity = DataSeverity.WARNING
        else:
            return None

        return DataHealthEvent(
            time=datetime.utcnow(),
            adapter=adapter,
            event_type="staleness",
            severity=severity,
            details={
                "series_key": series_key,
                "last_seen": last.isoformat(),
                "age_hours": round(age.total_seconds() / 3600, 2),
                "threshold_hours": round(threshold.total_seconds() / 3600, 2),
            },
        )

    async def run_periodic_checks(self, interval_seconds: int = 300) -> None:
        """Background task: check all tracked adapters every interval."""
        self._running = True
        while self._running:
            await asyncio.sleep(interval_seconds)
            events = []
            for key, last in self._last_seen.items():
                adapter, series = key.split(":", 1)
                # Infer data_type from series key
                data_type = _infer_data_type(series)
                event = self.check_staleness(adapter, series, data_type)
                if event:
                    events.append(event)
                    logger.warning(
                        "data_health_alert",
                        adapter=event.adapter,
                        severity=event.severity,
                        event_type=event.event_type,
                        **event.details,
                    )

    def stop(self) -> None:
        self._running = False

    def summary(self) -> dict:
        now = datetime.utcnow()
        return {
            key: {
                "last_seen": last.isoformat(),
                "age_minutes": round((now - last).total_seconds() / 60, 1),
            }
            for key, last in self._last_seen.items()
        }


def _infer_data_type(series_key: str) -> str:
    key = series_key.lower()
    if "intraday" in key or "1min" in key:
        return "equity_ohlcv_intraday"
    if "quote" in key:
        return "quote"
    if "edgar" in key or "filing" in key:
        return "edgar"
    if "fred" in key or "macro" in key:
        return "fred"
    if "cot" in key:
        return "cot"
    if "crypto" in key or "btc" in key or "eth" in key:
        return "crypto"
    if "news" in key:
        return "news"
    return "equity_ohlcv_daily"


_monitor: DataHealthMonitor | None = None


def get_monitor() -> DataHealthMonitor:
    global _monitor
    if _monitor is None:
        _monitor = DataHealthMonitor()
    return _monitor
