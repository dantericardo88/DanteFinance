"""Trading day gap detector.

Identifies missing sessions in an OHLCV bar series by comparing against
a market calendar. Also detects silent API throttling — when an adapter
returns an empty DataFrame instead of raising an error under load.

Silent throttle signature: actual_count == 0, expected_count > 0.
This is a known yfinance behavior under rate pressure.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from sentinel.core.types import OHLCVBar
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_INTRADAY_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h"}


@dataclass
class GapReport:
    ticker: str
    exchange: str
    interval: str
    missing_dates: list[date] = field(default_factory=list)
    expected_count: int = 0
    actual_count: int = 0

    @property
    def gap_count(self) -> int:
        return len(self.missing_dates)

    @property
    def completeness(self) -> float:
        return self.actual_count / self.expected_count if self.expected_count else 0.0

    @property
    def is_silent_throttle(self) -> bool:
        """True when adapter returned 0 bars for a period that should have data."""
        return self.actual_count == 0 and self.expected_count > 0

    @property
    def is_clean(self) -> bool:
        return self.gap_count == 0 and not self.is_silent_throttle


def _get_expected_trading_days(exchange: str, start: date, end: date) -> list[date]:
    """Return expected trading days via pandas_market_calendars; falls back to weekdays."""
    try:
        import pandas_market_calendars as mcal
        import pandas as pd
        cal = mcal.get_calendar(exchange)
        schedule = cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())
        return [pd.Timestamp(d).date() for d in schedule.index]
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Calendar lookup failed, falling back to weekdays", exchange=exchange, error=str(exc))

    import pandas as pd
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


def detect_gaps(
    bars: list[OHLCVBar],
    ticker: str,
    start: date,
    end: date,
    exchange: str = "NYSE",
    interval: str = "1d",
) -> GapReport:
    """Detect missing trading sessions in a bar series.

    For intraday intervals, only checks for silent throttle (0 bars).
    For daily/weekly/monthly, compares against market calendar.
    """
    if interval in _INTRADAY_INTERVALS:
        # Calendar-level gap detection is not reliable for intraday without session data
        report = GapReport(
            ticker=ticker, exchange=exchange, interval=interval,
            expected_count=max(len(bars), 1),
            actual_count=len(bars),
        )
        if report.is_silent_throttle:
            logger.error(
                "Silent throttle — intraday adapter returned 0 bars",
                ticker=ticker, interval=interval, start=start, end=end,
            )
        return report

    expected = _get_expected_trading_days(exchange, start, end)
    bar_dates = {b.time.date() for b in bars}
    missing = [d for d in expected if d not in bar_dates]

    report = GapReport(
        ticker=ticker,
        exchange=exchange,
        interval=interval,
        missing_dates=missing,
        expected_count=len(expected),
        actual_count=len(bars),
    )

    if report.is_silent_throttle:
        logger.error(
            "Silent throttle — daily adapter returned 0 bars",
            ticker=ticker, exchange=exchange, start=start, end=end,
        )
    elif report.gap_count > 0:
        logger.warning(
            "Trading day gaps detected",
            ticker=ticker, gaps=report.gap_count,
            completeness=f"{report.completeness:.1%}",
            first_missing=missing[0].isoformat() if missing else None,
        )

    return report
