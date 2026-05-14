"""Cross-source OHLCV validator.

Two validation layers:
1. Single-source: hard-rejects impossible values (Close > High, negative prices).
2. Cross-source: flags bars where two sources diverge by more than threshold (default 1%).

A bar that fails single-source validation is dropped. A bar that fails cross-source
validation is flagged in the report but still returned for manual review.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sentinel.core.types import OHLCVBar
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

CROSS_SOURCE_THRESHOLD = Decimal("0.01")   # 1% tolerance


@dataclass
class ValidationError:
    bar_time: datetime
    ticker: str
    source: str
    rule: str
    details: str


@dataclass
class ValidationReport:
    ticker: str
    total: int
    passed: int
    rejected: int
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def is_clean(self) -> bool:
        return self.rejected == 0


def _pct_delta(a: Decimal, b: Decimal) -> Decimal:
    if b == Decimal("0"):
        return Decimal("0")
    return abs(a - b) / b


def validate_single_source(
    bars: list[OHLCVBar],
    ticker: str,
) -> tuple[list[OHLCVBar], ValidationReport]:
    """Validate internal OHLCV consistency.

    Hard-rejects bars with:
    - Close > High
    - Close < Low
    - High < Low
    - Negative open/high/low/close
    - Negative volume
    """
    passed: list[OHLCVBar] = []
    errors: list[ValidationError] = []

    for bar in bars:
        bar_errors: list[ValidationError] = []

        if bar.close > bar.high:
            bar_errors.append(ValidationError(
                bar_time=bar.time, ticker=ticker, source=bar.source,
                rule="close_above_high",
                details=f"close={bar.close} > high={bar.high}",
            ))
        if bar.close < bar.low:
            bar_errors.append(ValidationError(
                bar_time=bar.time, ticker=ticker, source=bar.source,
                rule="close_below_low",
                details=f"close={bar.close} < low={bar.low}",
            ))
        if bar.high < bar.low:
            bar_errors.append(ValidationError(
                bar_time=bar.time, ticker=ticker, source=bar.source,
                rule="high_below_low",
                details=f"high={bar.high} < low={bar.low}",
            ))
        if bar.open < Decimal("0") or bar.high < Decimal("0") or bar.low < Decimal("0") or bar.close < Decimal("0"):
            bar_errors.append(ValidationError(
                bar_time=bar.time, ticker=ticker, source=bar.source,
                rule="negative_price",
                details=f"open={bar.open} high={bar.high} low={bar.low} close={bar.close}",
            ))
        if bar.volume < 0:
            bar_errors.append(ValidationError(
                bar_time=bar.time, ticker=ticker, source=bar.source,
                rule="negative_volume",
                details=f"volume={bar.volume}",
            ))

        if bar_errors:
            errors.extend(bar_errors)
        else:
            passed.append(bar)

    report = ValidationReport(
        ticker=ticker,
        total=len(bars),
        passed=len(passed),
        rejected=len(bars) - len(passed),
        errors=errors,
    )

    if report.rejected:
        logger.warning(
            "OHLCV single-source validation failures",
            ticker=ticker, rejected=report.rejected, total=report.total,
        )

    return passed, report


def cross_validate(
    primary: list[OHLCVBar],
    secondary: list[OHLCVBar],
    ticker: str,
    threshold: Decimal = CROSS_SOURCE_THRESHOLD,
) -> tuple[list[OHLCVBar], ValidationReport]:
    """Compare close prices across two sources; flag bars where delta exceeds threshold.

    Returns primary bars that are within tolerance.
    Bars with no matching secondary timestamp pass through (no comparison available).
    """
    secondary_map: dict[datetime, OHLCVBar] = {b.time: b for b in secondary}
    passed: list[OHLCVBar] = []
    errors: list[ValidationError] = []

    for bar in primary:
        sec = secondary_map.get(bar.time)
        if sec is None:
            passed.append(bar)
            continue

        delta = _pct_delta(bar.close, sec.close)
        if delta > threshold:
            errors.append(ValidationError(
                bar_time=bar.time, ticker=ticker, source=bar.source,
                rule="cross_source_delta",
                details=(
                    f"{bar.source}={bar.close} vs {sec.source}={sec.close} "
                    f"delta={float(delta):.4%}"
                ),
            ))
        else:
            passed.append(bar)

    report = ValidationReport(
        ticker=ticker,
        total=len(primary),
        passed=len(passed),
        rejected=len(primary) - len(passed),
        errors=errors,
    )

    if errors:
        logger.warning(
            "Cross-source validation flags",
            ticker=ticker, flagged=len(errors),
            primary_source=primary[0].source if primary else "?",
            secondary_source=secondary[0].source if secondary else "?",
        )

    return passed, report
