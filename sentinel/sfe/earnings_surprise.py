"""earnings_surprise.py — Earnings surprise tracker via yfinance (Dim 19)."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QuarterlySurprise(BaseModel):
    model_config = ConfigDict(frozen=True)

    period: str                   # "YYYY-Qn" or raw date string
    actual_eps: Optional[float] = None
    estimate_eps: Optional[float] = None
    surprise_pct: Optional[float] = None   # None if no estimate
    beat: Optional[bool] = None            # True/False/None
    magnitude_label: str          # large beat/beat/inline/miss/large miss/no estimate


class EarningsSurpriseHistory(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    quarters_analyzed: int
    surprises: list[QuarterlySurprise]     # most recent first
    beat_rate: Optional[float] = None      # fraction 0-1
    avg_surprise_pct: Optional[float] = None
    consistency_score: float               # 0-10
    trend: str                             # improving/stable/deteriorating/insufficient data
    next_earnings_date: Optional[str] = None
    estimated_next_eps: Optional[float] = None
    consecutive_beats: int                 # current streak from most recent
    as_of: str
    warnings: list[str]


class EarningsBeatSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    beat_rate: float
    avg_surprise_pct: Optional[float] = None
    consistency_score: float
    quarters_analyzed: int
    last_quarter_beat: Optional[bool] = None


class EarningsBeatScreen(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    min_beat_rate_filter: float
    results: list[EarningsBeatSummary]     # sorted by avg_surprise_pct desc
    most_consistent: Optional[str] = None  # highest beat_rate
    biggest_avg_beat: Optional[str] = None # highest avg_surprise_pct
    recent_misses: list[str]               # last quarter was a miss
    avg_beat_rate: Optional[float] = None
    as_of: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> Optional[float]:
    """Return None on NaN/None/inf; otherwise return float."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _date_to_quarter_label(dt) -> str:
    """Convert a date/datetime to 'YYYY-Qn' format."""
    try:
        import pandas as pd  # noqa: PLC0415 — lazy import by design
        ts = pd.Timestamp(dt)
        q = (ts.month - 1) // 3 + 1
        return f"{ts.year}-Q{q}"
    except Exception:
        return str(dt)


def _magnitude_label(surprise_pct: Optional[float]) -> str:
    """Classify surprise magnitude as a human-readable label."""
    if surprise_pct is None:
        return "no estimate"
    if surprise_pct > 10.0:
        return "large beat"
    if surprise_pct > 1.0:
        return "beat"
    if surprise_pct >= -1.0:
        return "inline"
    if surprise_pct >= -10.0:
        return "miss"
    return "large miss"


def _compute_consistency_score(beat_rate: float, surprises: list[QuarterlySurprise]) -> float:
    """0-10 score: base = beat_rate * 10, penalise large misses by 0.5 each."""
    score = beat_rate * 10.0
    for s in surprises:
        if s.magnitude_label == "large miss":
            score -= 0.5
    return round(max(0.0, min(10.0, score)), 4)


def _compute_trend(surprises: list[QuarterlySurprise]) -> str:
    """Compare avg surprise_pct of recent half vs older half."""
    valued = [s for s in surprises if s.surprise_pct is not None]
    if len(valued) < 4:
        return "insufficient data"
    half = len(valued) // 2
    recent_half = valued[:half]   # most recent first
    older_half = valued[half:]
    recent_avg = float(np.mean([s.surprise_pct for s in recent_half]))  # type: ignore[arg-type]
    older_avg = float(np.mean([s.surprise_pct for s in older_half]))    # type: ignore[arg-type]
    diff = recent_avg - older_avg
    if diff > 2.0:
        return "improving"
    if diff < -2.0:
        return "deteriorating"
    return "stable"


def _count_consecutive_beats(surprises: list[QuarterlySurprise]) -> int:
    """Count consecutive beats from the most recent quarter backward."""
    count = 0
    for s in surprises:
        if s.beat is True:
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Sync yfinance fetch — runs inside asyncio.to_thread
# ---------------------------------------------------------------------------


def _fetch_surprise_data_sync(ticker: str) -> dict:  # noqa: C901
    """Sync yfinance fetch — called via asyncio.to_thread. Lazy-imports yfinance/pandas."""
    import yfinance as yf   # noqa: PLC0415 — lazy import by design
    import pandas as pd     # noqa: PLC0415 — lazy import by design

    result: dict = {
        "earnings_history": None,
        "quarterly_earnings": None,
        "earnings_dates": None,
        "errors": [],
    }

    try:
        yf_ticker = yf.Ticker(ticker)
    except Exception as exc:
        result["errors"].append(f"Ticker init failed: {exc}")
        return result

    # --- earnings_history (Reported EPS, EPS Estimate, Surprise%) -----------
    try:
        hist = yf_ticker.earnings_history
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            result["earnings_history"] = hist
    except Exception as exc:
        result["errors"].append(f"earnings_history fetch failed: {exc}")

    # --- quarterly_earnings (Revenue, Earnings) ------------------------------
    try:
        qe = yf_ticker.quarterly_earnings
        if isinstance(qe, pd.DataFrame) and not qe.empty:
            result["quarterly_earnings"] = qe
    except Exception as exc:
        result["errors"].append(f"quarterly_earnings fetch failed: {exc}")

    # --- earnings_dates (upcoming/recent with EPS Estimate) -----------------
    try:
        ed = yf_ticker.earnings_dates
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            result["earnings_dates"] = ed
    except Exception as exc:
        result["errors"].append(f"earnings_dates fetch failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Build QuarterlySurprise list from raw fetch
# ---------------------------------------------------------------------------


def _build_surprises(raw: dict, quarters: int, warnings: list[str]) -> list[QuarterlySurprise]:  # noqa: C901
    """Parse earnings_history DataFrame into sorted QuarterlySurprise list."""
    import pandas as pd  # noqa: PLC0415 — lazy import by design

    hist_df: Optional[object] = raw.get("earnings_history")
    surprises: list[QuarterlySurprise] = []

    if hist_df is None or not isinstance(hist_df, pd.DataFrame) or hist_df.empty:
        warnings.append("No earnings_history data available")
        return surprises

    # Normalise column names — yfinance may vary casing
    col_map = {c.lower().replace(" ", "").replace("(", "").replace(")", ""): c
               for c in hist_df.columns}
    reported_col = col_map.get("reportedeps") or col_map.get("epsactual") or col_map.get("reported")
    estimate_col = (col_map.get("epsestimate") or col_map.get("estimate")
                    or col_map.get("estimatedeps"))
    surprise_col = col_map.get("surprise%") or col_map.get("surprise") or col_map.get("surprisepct")

    # Attempt to sort by index descending (most recent first)
    try:
        df = hist_df.sort_index(ascending=False)
    except Exception:
        df = hist_df

    df = df.head(quarters)

    for idx, row in df.iterrows():
        period = _date_to_quarter_label(idx)

        actual_eps = _safe_float(row.get(reported_col) if reported_col else None)
        estimate_eps = _safe_float(row.get(estimate_col) if estimate_col else None)

        # Prefer yfinance's pre-computed surprise % if available
        raw_surprise = _safe_float(row.get(surprise_col) if surprise_col else None)

        surprise_pct: Optional[float] = None
        if raw_surprise is not None:
            # yfinance may return as fraction (e.g. 0.05) or percent (5.0) — normalise
            surprise_pct = raw_surprise if abs(raw_surprise) > 1.5 else raw_surprise * 100.0
        elif actual_eps is not None and estimate_eps is not None and estimate_eps != 0:
            surprise_pct = round((actual_eps - estimate_eps) / abs(estimate_eps) * 100, 4)

        beat: Optional[bool] = None
        if surprise_pct is not None:
            beat = surprise_pct > 0

        surprises.append(
            QuarterlySurprise(
                period=period,
                actual_eps=actual_eps,
                estimate_eps=estimate_eps,
                surprise_pct=surprise_pct,
                beat=beat,
                magnitude_label=_magnitude_label(surprise_pct),
            )
        )

    return surprises


# ---------------------------------------------------------------------------
# Parse next earnings date and estimated EPS
# ---------------------------------------------------------------------------


def _parse_next_earnings(raw: dict, warnings: list[str]) -> tuple[Optional[str], Optional[float]]:
    """Return (next_date_str, estimated_eps) from earnings_dates DataFrame."""
    import pandas as pd  # noqa: PLC0415 — lazy import by design

    ed: Optional[object] = raw.get("earnings_dates")
    if ed is None or not isinstance(ed, pd.DataFrame) or ed.empty:
        return None, None

    now = pd.Timestamp.utcnow()
    try:
        # Index is DatetimeTZDtype; look for the nearest future date
        future = ed[ed.index > now]
        if future.empty:
            # Fall back to most recent past date
            future = ed.sort_index(ascending=False).head(1)

        row = future.sort_index().iloc[0]
        date_str = row.name.strftime("%Y-%m-%d") if hasattr(row.name, "strftime") else str(row.name)

        # Find EPS estimate column
        col_map = {c.lower().replace(" ", "").replace("(", "").replace(")", ""): c
                   for c in ed.columns}
        eps_col = col_map.get("epsestimate") or col_map.get("estimate")
        estimated_eps = _safe_float(row.get(eps_col) if eps_col else None)
        return date_str, estimated_eps

    except Exception as exc:
        warnings.append(f"earnings_dates parse error: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Entry point: get_earnings_surprise
# ---------------------------------------------------------------------------


async def get_earnings_surprise(ticker: str, quarters: int = 8) -> EarningsSurpriseHistory:
    """Fetch and compute earnings surprise history for a single ticker via yfinance."""
    ticker_upper = ticker.upper()
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    logger.info("Fetching earnings surprise", ticker=ticker_upper, quarters=quarters)

    try:
        raw = await asyncio.to_thread(_fetch_surprise_data_sync, ticker_upper)
    except Exception as exc:
        logger.error("Earnings surprise fetch failed", ticker=ticker_upper, error=str(exc))
        warnings.append(f"Data fetch failed: {exc}")
        raw = {"earnings_history": None, "quarterly_earnings": None,
               "earnings_dates": None, "errors": [str(exc)]}

    warnings.extend(raw.get("errors", []))

    surprises = _build_surprises(raw, quarters, warnings)

    # --- Aggregate stats -------------------------------------------------------
    with_estimates = [s for s in surprises if s.surprise_pct is not None]
    beats = [s for s in with_estimates if s.beat is True]

    beat_rate: Optional[float] = None
    avg_surprise_pct: Optional[float] = None
    consistency_score = 0.0
    trend = "insufficient data"

    if with_estimates:
        beat_rate = round(len(beats) / len(with_estimates), 4)
        avg_surprise_pct = round(
            float(np.mean([s.surprise_pct for s in with_estimates])), 4  # type: ignore[arg-type]
        )
        consistency_score = _compute_consistency_score(beat_rate, surprises)
        trend = _compute_trend(surprises)
    else:
        warnings.append("No quarters with EPS estimates found — beat_rate unavailable")

    next_earnings_date, estimated_next_eps = _parse_next_earnings(raw, warnings)
    consecutive_beats = _count_consecutive_beats(surprises)

    history = EarningsSurpriseHistory(
        ticker=ticker_upper,
        quarters_analyzed=len(surprises),
        surprises=surprises,
        beat_rate=beat_rate,
        avg_surprise_pct=avg_surprise_pct,
        consistency_score=consistency_score,
        trend=trend,
        next_earnings_date=next_earnings_date,
        estimated_next_eps=estimated_next_eps,
        consecutive_beats=consecutive_beats,
        as_of=as_of,
        warnings=warnings,
    )

    logger.info(
        "Earnings surprise complete",
        ticker=ticker_upper,
        quarters_analyzed=len(surprises),
        beat_rate=beat_rate,
        avg_surprise_pct=avg_surprise_pct,
        consistency_score=consistency_score,
        trend=trend,
        consecutive_beats=consecutive_beats,
    )

    return history


# ---------------------------------------------------------------------------
# Entry point: screen_earnings_beats
# ---------------------------------------------------------------------------


async def screen_earnings_beats(
    tickers: list[str],
    min_beat_rate: float = 0.60,
) -> EarningsBeatScreen:
    """Parallel-fetch tickers, filter by min_beat_rate, return ranked screen."""
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    logger.info(
        "Screening earnings beats",
        tickers=len(tickers),
        min_beat_rate=min_beat_rate,
    )

    # --- Parallel fetch -------------------------------------------------------
    fetch_tasks = [get_earnings_surprise(t) for t in tickers]
    raw_results: list[EarningsSurpriseHistory | BaseException] = await asyncio.gather(
        *fetch_tasks, return_exceptions=True
    )

    summaries: list[EarningsBeatSummary] = []
    recent_misses: list[str] = []

    for ticker, res in zip(tickers, raw_results):
        if isinstance(res, BaseException):
            warnings.append(f"Fetch error for {ticker}: {res}")
            continue
        warnings.extend(res.warnings)

        # Require at least 4 quarters and meet beat_rate threshold
        if res.beat_rate is None or res.quarters_analyzed < 4:
            continue
        if res.beat_rate < min_beat_rate:
            continue

        last_quarter_beat: Optional[bool] = (
            res.surprises[0].beat if res.surprises else None
        )

        # Collect recent misses (last quarter was a miss) — regardless of filter
        if last_quarter_beat is False:
            recent_misses.append(res.ticker)

        summaries.append(
            EarningsBeatSummary(
                ticker=res.ticker,
                beat_rate=res.beat_rate,
                avg_surprise_pct=res.avg_surprise_pct,
                consistency_score=res.consistency_score,
                quarters_analyzed=res.quarters_analyzed,
                last_quarter_beat=last_quarter_beat,
            )
        )

    # Also capture recent_misses for tickers that didn't pass filter
    for ticker, res in zip(tickers, raw_results):
        if isinstance(res, BaseException):
            continue
        if res.ticker in [s.ticker for s in summaries]:
            continue  # already handled above
        if res.surprises and res.surprises[0].beat is False:
            recent_misses.append(res.ticker)

    # --- Sort by avg_surprise_pct descending ----------------------------------
    summaries.sort(key=lambda s: (s.avg_surprise_pct is None, -(s.avg_surprise_pct or 0)))

    # --- Screen-level aggregates ----------------------------------------------
    most_consistent: Optional[str] = None
    biggest_avg_beat: Optional[str] = None
    avg_beat_rate: Optional[float] = None

    if summaries:
        most_consistent = max(summaries, key=lambda s: s.beat_rate).ticker
        with_avg = [s for s in summaries if s.avg_surprise_pct is not None]
        if with_avg:
            biggest_avg_beat = max(with_avg, key=lambda s: s.avg_surprise_pct or 0.0).ticker  # type: ignore[arg-type]
        all_beat_rates = [s.beat_rate for s in summaries]
        avg_beat_rate = round(float(np.mean(all_beat_rates)), 4)

    # Deduplicate recent_misses preserving order
    seen: set[str] = set()
    deduped_misses: list[str] = []
    for t in recent_misses:
        if t not in seen:
            seen.add(t)
            deduped_misses.append(t)

    screen = EarningsBeatScreen(
        tickers_screened=len(tickers),
        min_beat_rate_filter=min_beat_rate,
        results=summaries,
        most_consistent=most_consistent,
        biggest_avg_beat=biggest_avg_beat,
        recent_misses=deduped_misses,
        avg_beat_rate=avg_beat_rate,
        as_of=as_of,
        warnings=warnings,
    )

    logger.info(
        "Earnings beat screen complete",
        tickers_screened=screen.tickers_screened,
        results_returned=len(summaries),
        most_consistent=most_consistent,
        biggest_avg_beat=biggest_avg_beat,
        recent_misses=len(deduped_misses),
        avg_beat_rate=avg_beat_rate,
    )

    return screen
