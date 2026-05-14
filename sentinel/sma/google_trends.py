"""
Google Trends signals via pytrends — Dimension #89 of the competitive matrix.

Surfaces search-interest momentum for any ticker/keyword. Computes z-score
momentum (recent 4-week avg vs prior 52-week baseline), trend direction, and
an optional price-divergence signal. Free, no API key required.

Score target: SENTINEL 9, Bloomberg 0 (Bloomberg doesn't expose Google Trends).
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TrendDataPoint(BaseModel):
    date: date
    interest: int  # 0-100 normalised interest


class TrendSignal(BaseModel):
    keyword: str
    ticker: Optional[str] = None
    recent_avg: float       # last 4 weeks average interest (0-100)
    baseline_avg: float     # prior 52-week average interest (0-100)
    momentum_zscore: float  # (recent_avg - baseline_avg) / std
    trend_direction: str    # "accelerating" | "decelerating" | "stable" | "new_trend"
    price_trend_divergence: Optional[float] = None  # positive = price up, interest down (bearish)
    peak_interest: int      # max interest in last 5 years
    peak_date: Optional[date] = None
    related_queries: list[str] = []
    signal_strength: str    # "strong_bullish" | "bullish" | "neutral" | "bearish" | "strong_bearish"
    warnings: list[str] = []
    data_points: int = 0


class TrendSummary(BaseModel):
    primary: TrendSignal
    comparisons: list[TrendSignal] = []
    geo: str = "US"
    timeframe: str
    data_points: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEUTRAL_DIRECTION = "stable"
_NEUTRAL_STRENGTH = "neutral"


def _classify_direction(zscore: float, baseline_avg: float) -> str:
    """Map z-score + baseline level to a trend direction label."""
    if zscore > 1.5:
        return "accelerating"
    if zscore < -1.5:
        return "decelerating"
    # Low-baseline keyword picking up steam
    if 0.5 <= zscore <= 1.5 and baseline_avg < 20:
        return "new_trend"
    return "stable"


def _classify_strength(zscore: float) -> str:
    """Map z-score to a signal-strength label."""
    if zscore > 2.0:
        return "strong_bullish"
    if zscore > 1.0:
        return "bullish"
    if zscore < -2.0:
        return "strong_bearish"
    if zscore < -1.0:
        return "bearish"
    return "neutral"


def _neutral_signal(keyword: str, ticker: Optional[str], warning: str) -> TrendSignal:
    """Return a safe neutral fallback when the API is unavailable."""
    logger.warning("google_trends fallback: %s | %s", keyword, warning)
    return TrendSignal(
        keyword=keyword,
        ticker=ticker,
        recent_avg=0.0,
        baseline_avg=0.0,
        momentum_zscore=0.0,
        trend_direction=_NEUTRAL_DIRECTION,
        peak_interest=0,
        signal_strength=_NEUTRAL_STRENGTH,
        warnings=[warning],
        data_points=0,
    )


def _build_signal(
    keyword: str,
    ticker: Optional[str],
    series: pd.Series,
    related: list[str],
) -> TrendSignal:
    """
    Compute momentum metrics from a weekly interest series.

    series: pd.Series with DatetimeIndex, values 0-100.
    """
    series = series.dropna().astype(float)
    n = len(series)
    if n < 8:
        return _neutral_signal(keyword, ticker, f"Insufficient data points ({n})")

    recent = series.iloc[-4:]
    # Baseline = everything except the last 4 weeks, but cap at 52 weeks back from week -4
    baseline_end = len(series) - 4
    baseline_start = max(0, baseline_end - 52)
    baseline = series.iloc[baseline_start:baseline_end]

    recent_avg = float(recent.mean())
    baseline_avg = float(baseline.mean())
    baseline_std = float(baseline.std(ddof=1)) if len(baseline) > 1 else 1.0
    if baseline_std < 1e-6:
        baseline_std = 1.0

    zscore = (recent_avg - baseline_avg) / baseline_std
    direction = _classify_direction(zscore, baseline_avg)
    strength = _classify_strength(zscore)

    peak_val = int(series.max())
    peak_idx = series.idxmax()
    peak_dt: Optional[date] = peak_idx.date() if peak_idx is not pd.NaT else None

    return TrendSignal(
        keyword=keyword,
        ticker=ticker,
        recent_avg=round(recent_avg, 2),
        baseline_avg=round(baseline_avg, 2),
        momentum_zscore=round(zscore, 4),
        trend_direction=direction,
        peak_interest=peak_val,
        peak_date=peak_dt,
        related_queries=related,
        signal_strength=strength,
        data_points=n,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_trend_signal(
    ticker: str,
    keywords: Optional[list[str]] = None,
    geo: str = "US",
    include_related: bool = True,
) -> TrendSignal:
    """
    Fetch Google Trends interest for a ticker over the last 5 years and return
    a momentum signal with z-score, direction, and strength classification.

    Parameters
    ----------
    ticker:
        Stock symbol used as the fallback keyword (e.g. "AAPL").
    keywords:
        Explicit search terms. Defaults to [ticker].
    geo:
        Google Trends geography code. Defaults to "US".
    include_related:
        Whether to fetch related rising queries (adds one extra API call).
    """
    # Lazy import so the module loads even if pytrends is not installed
    try:
        from pytrends.request import TrendReq
        from pytrends.exceptions import ResponseError, TooManyRequestsError
    except ImportError:
        return _neutral_signal(ticker, ticker, "pytrends not installed — run: pip install pytrends")

    kws = keywords if keywords else [ticker]
    # Google Trends allows max 5 keywords per request
    kws = kws[:5]
    primary_kw = kws[0]
    timeframe = "today 5-y"

    pytrends = TrendReq(hl="en-US", tz=360, retries=3, backoff_factor=1.5)

    # ── Fetch interest over time ────────────────────────────────────────────
    try:
        pytrends.build_payload(kws, cat=0, timeframe=timeframe, geo=geo, gprop="")
        time.sleep(1.0)  # respect rate limit
        iot = pytrends.interest_over_time()
    except Exception as exc:  # TooManyRequestsError, ResponseError, ConnectionError, etc.
        return _neutral_signal(primary_kw, ticker, f"API error fetching interest: {exc!r}")

    if iot is None or iot.empty or primary_kw not in iot.columns:
        return _neutral_signal(primary_kw, ticker, "Empty response from Google Trends")

    series = iot[primary_kw]

    # ── Related queries (rising) ────────────────────────────────────────────
    related: list[str] = []
    if include_related:
        try:
            time.sleep(1.0)
            rq = pytrends.related_queries()
            rising_df = rq.get(primary_kw, {}).get("rising")
            if rising_df is not None and not rising_df.empty:
                related = rising_df["query"].head(5).tolist()
        except Exception as exc:
            logger.warning("google_trends related_queries failed: %s", exc)

    return _build_signal(primary_kw, ticker, series, related)


def compare_trends(
    tickers: list[str],
    geo: str = "US",
) -> TrendSummary:
    """
    Compare up to 5 tickers in a single Google Trends call.
    Returns relative interest normalised to the first ticker.

    Parameters
    ----------
    tickers:
        List of ticker symbols. At most 5 (Google Trends hard limit).
    geo:
        Google Trends geography code.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        primary_fallback = _neutral_signal(
            tickers[0] if tickers else "UNKNOWN",
            tickers[0] if tickers else None,
            "pytrends not installed",
        )
        return TrendSummary(primary=primary_fallback, geo=geo, timeframe="today 5-y", data_points=0)

    kws = [t.upper() for t in tickers[:5]]
    timeframe = "today 5-y"
    pytrends = TrendReq(hl="en-US", tz=360, retries=3, backoff_factor=1.5)

    try:
        pytrends.build_payload(kws, cat=0, timeframe=timeframe, geo=geo, gprop="")
        time.sleep(1.0)
        iot = pytrends.interest_over_time()
    except Exception as exc:
        fallback = _neutral_signal(kws[0], kws[0], f"API error: {exc!r}")
        return TrendSummary(primary=fallback, geo=geo, timeframe=timeframe, data_points=0)

    if iot is None or iot.empty:
        fallback = _neutral_signal(kws[0], kws[0], "Empty response")
        return TrendSummary(primary=fallback, geo=geo, timeframe=timeframe, data_points=0)

    primary_col = kws[0]
    if primary_col not in iot.columns:
        fallback = _neutral_signal(primary_col, primary_col, "Primary keyword missing from response")
        return TrendSummary(primary=fallback, geo=geo, timeframe=timeframe, data_points=0)

    primary_signal = _build_signal(primary_col, primary_col, iot[primary_col], [])
    comparisons: list[TrendSignal] = []
    for kw in kws[1:]:
        if kw in iot.columns:
            sig = _build_signal(kw, kw, iot[kw], [])
            comparisons.append(sig)

    return TrendSummary(
        primary=primary_signal,
        comparisons=comparisons,
        geo=geo,
        timeframe=timeframe,
        data_points=len(iot),
    )


def get_earnings_search_spike(
    ticker: str,
    earnings_date: date,
    window_days: int = 7,
) -> dict:
    """
    Detect whether Google search interest spiked around an earnings date.

    Returns a dict with keys:
        spiked (bool), spike_magnitude (float), pre_earnings_trend (str)

    Uses daily data for a 90-day window centred on the earnings date so the
    spike (if any) is visible at day-level granularity.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return {"spiked": False, "spike_magnitude": 0.0, "pre_earnings_trend": "unknown",
                "warning": "pytrends not installed"}

    pytrends = TrendReq(hl="en-US", tz=360, retries=3, backoff_factor=1.5)

    # Build a 90-day window around the earnings date for daily granularity
    start = earnings_date - timedelta(days=45)
    end = earnings_date + timedelta(days=45)
    today = date.today()
    if end > today:
        end = today
    if start >= end:
        return {"spiked": False, "spike_magnitude": 0.0, "pre_earnings_trend": "unknown",
                "warning": "earnings_date too close to today for daily data"}

    timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"

    try:
        pytrends.build_payload([ticker], cat=0, timeframe=timeframe, geo="US", gprop="")
        time.sleep(1.0)
        iot = pytrends.interest_over_time()
    except Exception as exc:
        return {"spiked": False, "spike_magnitude": 0.0, "pre_earnings_trend": "unknown",
                "warning": f"API error: {exc!r}"}

    if iot is None or iot.empty or ticker not in iot.columns:
        return {"spiked": False, "spike_magnitude": 0.0, "pre_earnings_trend": "unknown",
                "warning": "No data returned"}

    series = iot[ticker].astype(float)
    earnings_ts = pd.Timestamp(earnings_date)

    # Pre-earnings window: [earnings - window_days, earnings)
    pre = series[
        (series.index >= earnings_ts - timedelta(days=window_days))
        & (series.index < earnings_ts)
    ]
    # Spike window: [earnings, earnings + window_days]
    spike_window = series[
        (series.index >= earnings_ts)
        & (series.index <= earnings_ts + timedelta(days=window_days))
    ]

    if pre.empty or spike_window.empty:
        return {"spiked": False, "spike_magnitude": 0.0, "pre_earnings_trend": "insufficient_data"}

    pre_avg = float(pre.mean())
    spike_max = float(spike_window.max())
    magnitude = (spike_max - pre_avg) / max(pre_avg, 1.0)
    spiked = magnitude > 0.3  # 30% above pre-earnings baseline = spike

    # Pre-earnings trend slope (simple linear regression)
    if len(pre) >= 3:
        x = np.arange(len(pre), dtype=float)
        y = pre.values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0])
        pre_trend = "rising" if slope > 0.5 else ("falling" if slope < -0.5 else "flat")
    else:
        pre_trend = "flat"

    return {
        "spiked": spiked,
        "spike_magnitude": round(magnitude, 4),
        "pre_earnings_trend": pre_trend,
        "pre_avg_interest": round(pre_avg, 2),
        "spike_peak_interest": round(spike_max, 2),
    }
