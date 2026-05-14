"""Sector rotation analytics using SPDR ETFs — Dimension 62/69 enhancement."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTOR_NAMES: dict[str, str] = {
    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}

_SECTOR_TICKERS: list[str] = list(SECTOR_NAMES.keys())
_SPY = "SPY"

# Approximate trading-day windows per period label
_WINDOWS: dict[str, int] = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}

# Momentum composite weights (must sum to 1.0)
_W_1M: float = 0.4
_W_3M: float = 0.3
_W_6M: float = 0.2
_W_12M: float = 0.1

# Clamp boundaries and score mapping for momentum score
_RS_MIN: float = -0.20
_RS_MAX: float = +0.20
_SCORE_MIN: float = 0.0
_SCORE_MAX: float = 10.0

# Market-regime momentum thresholds
_RISK_ON_THRESHOLD: float = 6.0
_RISK_OFF_THRESHOLD: float = 4.0


# ---------------------------------------------------------------------------
# Pydantic models — all frozen
# ---------------------------------------------------------------------------

class SectorMetrics(BaseModel):
    """Per-sector relative-strength and momentum summary."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str
    rs_1m: Optional[float] = None
    rs_3m: Optional[float] = None
    rs_6m: Optional[float] = None
    rs_12m: Optional[float] = None
    momentum_score: float = Field(default=5.0, ge=0.0, le=10.0)
    vol_30d: Optional[float] = None
    avg_volume_ratio: Optional[float] = None
    trend: str = "neutral"
    rank_1m: Optional[int] = None
    rank_3m: Optional[int] = None
    rank_composite: Optional[int] = None
    as_of: str = ""


class SectorHeatmap(BaseModel):
    """Full cross-sector snapshot with regime classification."""

    model_config = ConfigDict(frozen=True)

    as_of: str
    sectors: list[SectorMetrics] = Field(default_factory=list)
    market_regime: str = "neutral"
    top_sectors: list[str] = Field(default_factory=list)
    bottom_sectors: list[str] = Field(default_factory=list)
    rotation_signal: str = ""
    warnings: list[str] = Field(default_factory=list)


class SectorStrengthSummary(BaseModel):
    """Lightweight sector record used in screener output."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str
    momentum_score: float
    rank_composite: Optional[int] = None
    trend: str


class SectorRotationScreen(BaseModel):
    """Top-N sector screener result."""

    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    top_n: int
    results: list[SectorStrengthSummary] = Field(default_factory=list)
    avg_momentum: Optional[float] = None
    market_breadth: Optional[float] = None
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _relative_strength(
    sector_closes: np.ndarray,
    spy_closes: np.ndarray,
    window: int,
) -> Optional[float]:
    """RS = (sector_return - spy_return) over the last *window* trading days.

    Returns None when either series contains fewer than window + 1 bars.
    """
    if len(sector_closes) < window + 1 or len(spy_closes) < window + 1:
        return None
    sector_return = sector_closes[-1] / sector_closes[-window] - 1.0
    spy_return = spy_closes[-1] / spy_closes[-window] - 1.0
    return float(sector_return - spy_return)


def _momentum_score(
    rs_1m: Optional[float],
    rs_3m: Optional[float],
    rs_6m: Optional[float],
    rs_12m: Optional[float],
) -> float:
    """Weighted composite RS mapped to 0-10 scale.

    Weights: 1M×0.4 + 3M×0.3 + 6M×0.2 + 12M×0.1.  Raw composite is clamped
    to [−0.20, +0.20] then mapped linearly to [0, 10].  Returns 5.0 when all
    inputs are None (neutral / insufficient data).
    """
    if rs_1m is None and rs_3m is None and rs_6m is None and rs_12m is None:
        return 5.0
    raw = (
        _W_1M * (rs_1m or 0.0)
        + _W_3M * (rs_3m or 0.0)
        + _W_6M * (rs_6m or 0.0)
        + _W_12M * (rs_12m or 0.0)
    )
    clamped = max(_RS_MIN, min(_RS_MAX, raw))
    score = (clamped - _RS_MIN) / (_RS_MAX - _RS_MIN) * (_SCORE_MAX - _SCORE_MIN) + _SCORE_MIN
    return round(float(score), 4)


def _vol_30d(closes: np.ndarray) -> Optional[float]:
    """Annualized 30-day close-to-close volatility.  Requires ≥ 31 bars."""
    if len(closes) < 31:
        return None
    tail = closes[-31:]
    log_rets = np.log(tail[1:] / tail[:-1])
    return float(np.std(log_rets) * np.sqrt(252))


def _trend(rs_1m: Optional[float], rs_3m: Optional[float]) -> str:
    """Classify momentum trend from 1M and 3M relative-strength readings.

    Rules (in priority order):
      rs_1m > rs_3m > 0  → "accelerating"
      rs_1m > 0, rs_3m > 0 → "positive"
      rs_1m < 0, rs_3m < 0 → "negative"
      rs_1m > rs_3m       → "improving"
      else                → "neutral"
    """
    if rs_1m is None or rs_3m is None:
        return "neutral"
    if rs_1m > rs_3m > 0.0:
        return "accelerating"
    if rs_1m > 0.0 and rs_3m > 0.0:
        return "positive"
    if rs_1m < 0.0 and rs_3m < 0.0:
        return "negative"
    if rs_1m > rs_3m:
        return "improving"
    return "neutral"


def _rank_series(values: list[Optional[float]], ascending: bool = False) -> list[Optional[int]]:
    """Dense-rank a list of optional floats; None inputs receive None rank.

    Default: highest value → rank 1 (descending).
    """
    indexed = [(v, i) for i, v in enumerate(values) if v is not None]
    if not indexed:
        return [None] * len(values)
    indexed.sort(key=lambda x: x[0], reverse=not ascending)
    rank_map: dict[int, int] = {}
    current_rank = 1
    prev_val: Optional[float] = None
    for pos, (val, orig_idx) in enumerate(indexed):
        if prev_val is None or val != prev_val:
            current_rank = pos + 1
        rank_map[orig_idx] = current_rank
        prev_val = val
    return [rank_map.get(i) for i in range(len(values))]


def _market_regime(sector_metrics: list[SectorMetrics]) -> str:
    """Classify the broad market regime from cross-sector momentum.

    Primary rule:
      avg_momentum > 6.0 → "risk_on"
      avg_momentum < 4.0 → "risk_off"
      else               → "neutral"

    Sector-composition overlays (override primary):
      XLE and XLF both in top-3 composite rank → "late_cycle"
      XLU and XLP both in top-3 composite rank → "defensive"
    """
    if not sector_metrics:
        return "neutral"
    avg_momentum = float(np.mean([m.momentum_score for m in sector_metrics]))
    if avg_momentum > _RISK_ON_THRESHOLD:
        regime = "risk_on"
    elif avg_momentum < _RISK_OFF_THRESHOLD:
        regime = "risk_off"
    else:
        regime = "neutral"

    # Top-3 overlay
    ranked = sorted(
        [m for m in sector_metrics if m.rank_composite is not None],
        key=lambda m: m.rank_composite,  # type: ignore[arg-type]
    )
    top3 = {m.ticker for m in ranked[:3]}
    if "XLE" in top3 and "XLF" in top3:
        return "late_cycle"
    if "XLU" in top3 and "XLP" in top3:
        return "defensive"
    return regime


# ---------------------------------------------------------------------------
# yfinance sync fetcher — runs inside asyncio.to_thread
# ---------------------------------------------------------------------------

def _fetch_sector_data_sync(tickers: list[str], history_days: int) -> dict[str, np.ndarray]:
    """Batch-download *tickers* for history_days+5 days via yfinance.

    Returns dict ticker → numpy array of adjusted closes (oldest first).
    Tickers with no data are silently omitted; callers must check presence.
    """
    import yfinance as yf  # noqa: PLC0415

    period = f"{history_days + 5}d"
    result: dict[str, np.ndarray] = {}
    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as exc:
        logger.warning("yfinance batch download failed", error=str(exc))
        return result

    # MultiIndex DataFrame when multiple tickers requested
    if hasattr(raw.columns, "levels"):
        for ticker in tickers:
            try:
                lvl1 = raw.columns.get_level_values(1)
                lvl0 = raw.columns.get_level_values(0)
                if ticker in lvl1:
                    series = raw["Close"][ticker].dropna()
                elif ticker in lvl0:
                    series = raw[ticker]["Close"].dropna()
                else:
                    logger.warning("Ticker absent from download", ticker=ticker)
                    continue
                if len(series) > 0:
                    result[ticker] = series.values.astype(float)
            except Exception as exc:
                logger.warning("Ticker slice error", ticker=ticker, error=str(exc))
    else:
        # Single-ticker fallback
        if len(tickers) == 1 and "Close" in raw.columns:
            series = raw["Close"].dropna()
            if len(series) > 0:
                result[tickers[0]] = series.values.astype(float)
    return result


async def _fetch_sector_data(
    tickers: list[str],
    history_days: int,
    client=None,  # reserved for future httpx / broker-client injection
) -> dict[str, np.ndarray]:
    """Async wrapper: dispatch blocking yfinance download to a thread pool."""
    return await asyncio.to_thread(_fetch_sector_data_sync, tickers, history_days)


# ---------------------------------------------------------------------------
# Public async entry points
# ---------------------------------------------------------------------------

async def get_sector_rotation(
    history_days: int = 252,
    top_n: int = 3,
) -> SectorHeatmap:
    """Compute the full sector-rotation heatmap for the 11 SPDR sector ETFs.

    Steps:
      1. Batch-download SPY + 11 ETFs via asyncio.to_thread / yfinance.
      2. Compute RS at 1M (21d), 3M (63d), 6M (126d), 12M (252d) vs SPY.
      3. Rank sectors by rs_1m, rs_3m, and composite momentum_score.
      4. Classify market regime; select top_n and bottom_n sectors.
      5. Emit a concise rotation_signal string.

    Args:
        history_days: Calendar days of price history to request (default 252).
        top_n: Number of sectors labelled top / bottom (default 3).

    Returns:
        SectorHeatmap with full per-sector metrics, regime, and warnings.
    """
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    all_tickers = [_SPY] + _SECTOR_TICKERS
    logger.info("Fetching sector data", tickers=all_tickers, history_days=history_days)

    closes_map = await _fetch_sector_data(all_tickers, history_days)

    spy_closes = closes_map.get(_SPY)
    if spy_closes is None or len(spy_closes) == 0:
        warnings.append("SPY data unavailable — relative-strength calculations skipped")
        logger.error("SPY data missing; cannot compute relative strength")
        spy_closes = None

    # Build per-sector metrics (pre-ranking)
    pre_rank: list[SectorMetrics] = []
    for ticker in _SECTOR_TICKERS:
        name = SECTOR_NAMES.get(ticker, ticker)
        sec = closes_map.get(ticker)

        if sec is None or len(sec) == 0:
            warnings.append(f"{ticker}: no price data — excluded from ranking")
            logger.warning("Missing sector data", ticker=ticker)
            pre_rank.append(
                SectorMetrics(ticker=ticker, name=name, momentum_score=5.0, as_of=as_of)
            )
            continue

        rs_vals: dict[str, Optional[float]] = {}
        for label, window in _WINDOWS.items():
            rs = _relative_strength(sec, spy_closes, window) if spy_closes is not None else None
            if rs is None:
                warnings.append(
                    f"{ticker}: insufficient history for rs_{label} "
                    f"(need {window + 1} bars, have {len(sec)})"
                )
            rs_vals[label] = rs

        pre_rank.append(
            SectorMetrics(
                ticker=ticker,
                name=name,
                rs_1m=rs_vals["1m"],
                rs_3m=rs_vals["3m"],
                rs_6m=rs_vals["6m"],
                rs_12m=rs_vals["12m"],
                momentum_score=_momentum_score(
                    rs_vals["1m"], rs_vals["3m"], rs_vals["6m"], rs_vals["12m"]
                ),
                vol_30d=_vol_30d(sec),
                avg_volume_ratio=None,
                trend=_trend(rs_vals["1m"], rs_vals["3m"]),
                as_of=as_of,
            )
        )

    # Compute ranks across all sectors
    ranks_1m = _rank_series([m.rs_1m for m in pre_rank], ascending=False)
    ranks_3m = _rank_series([m.rs_3m for m in pre_rank], ascending=False)
    ranks_composite = _rank_series([m.momentum_score for m in pre_rank], ascending=False)

    ranked_metrics: list[SectorMetrics] = []
    for i, m in enumerate(pre_rank):
        ranked_metrics.append(
            SectorMetrics(
                ticker=m.ticker,
                name=m.name,
                rs_1m=m.rs_1m,
                rs_3m=m.rs_3m,
                rs_6m=m.rs_6m,
                rs_12m=m.rs_12m,
                momentum_score=m.momentum_score,
                vol_30d=m.vol_30d,
                avg_volume_ratio=m.avg_volume_ratio,
                trend=m.trend,
                rank_1m=ranks_1m[i],
                rank_3m=ranks_3m[i],
                rank_composite=ranks_composite[i],
                as_of=as_of,
            )
        )

    regime = _market_regime(ranked_metrics)

    sorted_composite = sorted(
        [m for m in ranked_metrics if m.rank_composite is not None],
        key=lambda m: m.rank_composite,  # type: ignore[arg-type]
    )
    top_sectors = [m.ticker for m in sorted_composite[:top_n]]
    bottom_sectors = [m.ticker for m in sorted_composite[-top_n:]]

    if len(top_sectors) >= 2 and len(bottom_sectors) >= 2:
        rotation_signal = (
            f"Overweight {'/'.join(top_sectors[:2])}; "
            f"Underweight {'/'.join(bottom_sectors[:2])}"
        )
    elif top_sectors and bottom_sectors:
        rotation_signal = f"Overweight {top_sectors[0]}; Underweight {bottom_sectors[0]}"
    else:
        rotation_signal = "Insufficient data for rotation signal"

    logger.info(
        "Sector rotation computed",
        regime=regime,
        top=top_sectors,
        bottom=bottom_sectors,
        warnings=len(warnings),
    )

    return SectorHeatmap(
        as_of=as_of,
        sectors=ranked_metrics,
        market_regime=regime,
        top_sectors=top_sectors,
        bottom_sectors=bottom_sectors,
        rotation_signal=rotation_signal,
        warnings=warnings,
    )


async def screen_sector_strength(
    top_n: int = 5,
    history_days: int = 63,
) -> SectorRotationScreen:
    """Screener: return the top-N SPDR sectors ranked by composite momentum score.

    Internally calls get_sector_rotation() and filters to the requested top_n.
    market_breadth = fraction of sectors with positive rs_1m (0–1 scale).

    Args:
        top_n: Number of top sectors to return (default 5).
        history_days: Look-back window in calendar days (default 63 ≈ 3M).

    Returns:
        SectorRotationScreen with top-N summaries, breadth, and avg momentum.
    """
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    heatmap = await get_sector_rotation(history_days=history_days, top_n=top_n)
    warnings.extend(heatmap.warnings)

    all_sectors = heatmap.sectors
    sorted_sectors = sorted(all_sectors, key=lambda m: m.momentum_score, reverse=True)

    results: list[SectorStrengthSummary] = [
        SectorStrengthSummary(
            ticker=m.ticker,
            name=m.name,
            momentum_score=m.momentum_score,
            rank_composite=m.rank_composite,
            trend=m.trend,
        )
        for m in sorted_sectors[:top_n]
    ]

    # Market breadth: fraction of sectors with positive 1M RS
    sectors_with_rs1m = [m for m in all_sectors if m.rs_1m is not None]
    if sectors_with_rs1m:
        positive_count = sum(1 for m in sectors_with_rs1m if (m.rs_1m or 0.0) > 0.0)
        market_breadth: Optional[float] = round(positive_count / len(sectors_with_rs1m), 4)
    else:
        market_breadth = None
        warnings.append("No rs_1m data available — market breadth cannot be computed")

    avg_momentum: Optional[float] = (
        round(float(np.mean([m.momentum_score for m in all_sectors])), 4)
        if all_sectors
        else None
    )

    logger.info(
        "Sector screen complete",
        top_n=top_n,
        market_breadth=market_breadth,
        avg_momentum=avg_momentum,
    )

    return SectorRotationScreen(
        tickers_screened=len(all_sectors),
        top_n=top_n,
        results=results,
        avg_momentum=avg_momentum,
        market_breadth=market_breadth,
        as_of=as_of,
        warnings=warnings,
    )
