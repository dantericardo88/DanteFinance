"""Corporate actions engine — splits, dividends, reverse splits.

Raw OHLCV prices are NEVER modified after ingestion.
adj_factor is computed here and applied on read.

Convention: backward-adjustment (current-comparable).
  - 4-for-1 split on 2020-08-31: bars before that date get factor = 0.25
  - adjusted_close = raw_close × 0.25 → matches post-split price level

Dividend factors:
  - factor = (price_before_ex - dividend) / price_before_ex
  - Requires the closing price on the day before ex_date.
  - resolve_dividend_factors() fetches that price and updates the factor in-place.
"""
from __future__ import annotations
import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YF_AVAILABLE = False

from sentinel.core.types import CorporateAction, CorporateActionType, OHLCVBar
from sentinel.core.logging import get_logger

logger = get_logger(__name__)


def fetch_splits_yfinance(ticker: str) -> list[CorporateAction]:
    """Fetch full split history from yfinance and convert to CorporateAction list."""
    try:
        splits = yf.Ticker(ticker).splits
    except Exception as exc:
        logger.error("yfinance splits fetch failed", ticker=ticker, error=str(exc))
        return []

    if splits is None or splits.empty:
        return []

    actions: list[CorporateAction] = []
    for dt, ratio in splits.items():
        if ratio <= 0:
            continue
        ex_dt: date = dt.date() if hasattr(dt, "date") else dt
        action_type = (
            CorporateActionType.SPLIT if ratio > 1 else CorporateActionType.REVERSE_SPLIT
        )
        # factor = 1/ratio so that adjusted_close = raw_close × factor
        # e.g. 4-for-1 split: ratio=4, factor=0.25 → $400 × 0.25 = $100 (post-split comparable)
        actions.append(
            CorporateAction(
                figi=ticker,
                ticker=ticker,
                action_type=action_type,
                ex_date=ex_dt,
                ratio_new=Decimal(str(ratio)),
                ratio_old=Decimal("1"),
                factor=Decimal("1") / Decimal(str(ratio)),
                source="yfinance",
            )
        )

    logger.info("Splits fetched", ticker=ticker, count=len(actions))
    return actions


def fetch_dividends_yfinance(ticker: str) -> list[CorporateAction]:
    """Fetch dividend history from yfinance.

    factor=1.0 is stored as a placeholder — exact dividend adjustment requires the
    closing price on the day before ex_date, which is resolved separately.
    The ratio_new field stores the raw cash dividend per share.
    """
    try:
        divs = yf.Ticker(ticker).dividends
    except Exception as exc:
        logger.error("yfinance dividends fetch failed", ticker=ticker, error=str(exc))
        return []

    if divs is None or divs.empty:
        return []

    actions: list[CorporateAction] = []
    for dt, amount in divs.items():
        if amount <= 0:
            continue
        ex_dt: date = dt.date() if hasattr(dt, "date") else dt
        actions.append(
            CorporateAction(
                figi=ticker,
                ticker=ticker,
                action_type=CorporateActionType.DIVIDEND,
                ex_date=ex_dt,
                ratio_new=Decimal(str(round(float(amount), 8))),
                ratio_old=Decimal("0"),   # 0 = price not yet resolved
                factor=Decimal("1.0"),    # conservative until price-based factor computed
                source="yfinance",
            )
        )

    logger.info("Dividends fetched", ticker=ticker, count=len(actions))
    return actions


def compute_cumulative_factor(
    bar_date: date,
    actions: list[CorporateAction],
) -> Decimal:
    """Backward-adjustment cumulative factor for a single bar date.

    Multiplies all action.factor values whose ex_date falls AFTER bar_date.
    A bar dated 2020-01-01 will include the 2020-08-31 AAPL split factor.
    """
    factor = Decimal("1.0")
    for action in sorted(actions, key=lambda a: a.ex_date):
        if action.ex_date > bar_date:
            factor *= action.factor
    return factor


def apply_adjustments(
    bars: list[OHLCVBar],
    actions: list[CorporateAction],
) -> list[OHLCVBar]:
    """Return new OHLCVBar list with adj_factor populated.

    Raw close is preserved (frozen model). adjusted_close = close × adj_factor.
    """
    if not actions:
        return bars

    return [
        bar.model_copy(
            update={"adj_factor": compute_cumulative_factor(bar.time.date(), actions)}
        )
        for bar in bars
    ]


async def fetch_and_apply(
    ticker: str,
    bars: list[OHLCVBar],
) -> list[OHLCVBar]:
    """Fetch split history from yfinance and apply backward adjustments to bars."""
    loop = asyncio.get_event_loop()
    splits = await loop.run_in_executor(None, lambda: fetch_splits_yfinance(ticker))
    if not splits:
        return bars
    return apply_adjustments(bars, splits)


async def resolve_dividend_factors(
    actions: list[CorporateAction],
    ticker: str,
) -> list[CorporateAction]:
    """Replace placeholder dividend factors (1.0) with price-based factors.

    Dividend adjustment: factor = (price_before_ex - dividend_per_share) / price_before_ex

    Requires fetching the closing price on the day before each ex_date.
    Actions that already have a computed factor (ratio_old != 0) are left unchanged.
    """
    from sentinel.sds.normalizer import fetch_ohlcv_with_fallback

    resolved: list[CorporateAction] = []
    for action in actions:
        if action.action_type != CorporateActionType.DIVIDEND or action.ratio_old != Decimal("0"):
            resolved.append(action)
            continue

        day_before = action.ex_date - timedelta(days=1)
        start = datetime.combine(day_before, datetime.min.time())
        end = datetime.combine(day_before, datetime.max.time())

        try:
            bars = await fetch_ohlcv_with_fallback(ticker, start, end, "1d")
            if not bars:
                resolved.append(action)
                continue

            price_before = bars[-1].close
            if price_before <= Decimal("0"):
                resolved.append(action)
                continue

            div_amount = action.ratio_new
            raw_factor = (price_before - div_amount) / price_before
            # Floor at 0.5 — a single dividend cannot halve the price; signals bad data
            factor = max(Decimal("0.5"), raw_factor)

            resolved.append(action.model_copy(update={
                "factor": factor,
                "ratio_old": price_before,
            }))
            logger.info(
                "Dividend factor resolved",
                ticker=ticker, ex_date=action.ex_date,
                div=float(div_amount), price_before=float(price_before),
                factor=float(factor),
            )
        except Exception as exc:
            logger.warning("Dividend factor resolution failed", ticker=ticker,
                           ex_date=action.ex_date, error=str(exc))
            resolved.append(action)

    return resolved
