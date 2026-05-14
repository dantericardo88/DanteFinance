"""SDS normalizer — raw adapter payloads → canonical SENTINEL types with fallback chain."""
from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Optional
from sentinel.core.types import OHLCVBar, Quote, DataHealthEvent
from sentinel.core.logging import get_logger
from sentinel.sds import get_adapter, OHLCV_CHAIN

logger = get_logger(__name__)


async def fetch_ohlcv_with_fallback(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
    figi: Optional[str] = None,
    chain: Optional[list[str]] = None,
) -> list[OHLCVBar]:
    """Try each adapter in the chain; return first non-empty result."""
    chain = chain or OHLCV_CHAIN
    for name in chain:
        adapter = get_adapter(name)
        if adapter is None:
            continue
        try:
            bars = await adapter.fetch_ohlcv(ticker, start, end, interval, figi)
            if bars:
                logger.info("OHLCV sourced", adapter=name, ticker=ticker, bars=len(bars))
                return bars
        except Exception as exc:
            logger.warning("OHLCV adapter failed", adapter=name, ticker=ticker, error=str(exc))
    logger.error("All OHLCV adapters exhausted", ticker=ticker)
    return []


async def run_all_health_checks() -> list[DataHealthEvent]:
    """Run health_check() on every registered adapter concurrently."""
    from sentinel.sds import get_all_adapters
    adapters = get_all_adapters()
    tasks = {name: asyncio.create_task(a.health_check()) for name, a in adapters.items()}
    results = []
    for name, task in tasks.items():
        try:
            event = await task
            results.append(event)
        except Exception as exc:
            logger.error("Health check crashed", adapter=name, error=str(exc))
    return results


def deduplicate_bars(bars: list[OHLCVBar]) -> list[OHLCVBar]:
    """Remove duplicate timestamps, keeping first occurrence (source priority preserved)."""
    seen: set[datetime] = set()
    out = []
    for bar in bars:
        if bar.time not in seen:
            seen.add(bar.time)
            out.append(bar)
    return out


def align_to_calendar(
    bars: list[OHLCVBar],
    expected_dates: list[datetime],
) -> list[OHLCVBar]:
    """Forward-fill gaps so result has exactly one bar per expected date."""
    bar_map = {b.time.date(): b for b in bars}
    result = []
    last: Optional[OHLCVBar] = None
    for dt in expected_dates:
        d = dt.date()
        if d in bar_map:
            last = bar_map[d]
            result.append(last)
        elif last is not None:
            # Forward-fill with same close, zero volume
            from decimal import Decimal
            result.append(OHLCVBar(
                time=dt,
                figi=last.figi,
                open=last.close,
                high=last.close,
                low=last.close,
                close=last.close,
                volume=Decimal("0"),
                source=f"{last.source}:ffill",
            ))
    return result
