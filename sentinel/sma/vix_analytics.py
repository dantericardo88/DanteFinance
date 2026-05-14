"""
VIX Term Structure & Volatility Risk Premium Analytics — Dimensions #48/#50.

Provides VIX term structure (spot/3M/6M/1Y), contango/backwardation slope,
volatility risk premium (implied minus realized), VIX regime classification,
VVIX tail-risk signal, and per-ticker realized vol summary.
All data sourced from yfinance (free, no API key required).
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_VIX_TICKERS = ["^VIX", "^VIX3M", "^VIX6M", "^VIX1Y", "^VVIX", "^SKEW"]
_SPX_TICKER = "^GSPC"
_DEFAULT_TICKERS: list[str] = ["SPY", "QQQ", "IWM"]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class VIXTermPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenor: str
    level: Optional[float] = None
    percentile: Optional[float] = None  # vs past 252d history


class VIXHistory(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: str
    vix: Optional[float] = None
    vix3m: Optional[float] = None
    term_slope: Optional[float] = None


class VIXAnalytics(BaseModel):
    model_config = ConfigDict(frozen=True)

    term_structure: list[VIXTermPoint]
    spot_vix: Optional[float] = None
    vix3m: Optional[float] = None
    vix6m: Optional[float] = None
    vix1y: Optional[float] = None
    vvix: Optional[float] = None
    skew_index: Optional[float] = None
    term_slope_pct: Optional[float] = None        # (VIX3M - VIX) / VIX * 100
    contango: Optional[bool] = None               # True = normal (VIX3M > VIX)
    vix_percentile: Optional[float] = None        # 0-100
    vrp: Optional[float] = None                   # VIX - SPY realized vol 30d
    realized_vol_30d_spy: Optional[float] = None
    vol_regime: str = "unknown"
    mean_reversion_signal: str = "neutral"
    vvix_regime: str = "normal"
    history_60d: list[VIXHistory] = Field(default_factory=list)
    as_of: str
    warnings: list[str] = Field(default_factory=list)


class TickerVolSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    realized_vol_20d: Optional[float] = None   # annualized
    realized_vol_60d: Optional[float] = None
    iv_rv_premium: Optional[float] = None      # VIX - realized_vol_20d


class VolRegime(BaseModel):
    model_config = ConfigDict(frozen=True)

    market_regime: str
    spot_vix: Optional[float] = None
    vix_percentile: Optional[float] = None
    vrp: Optional[float] = None
    ticker_vols: list[TickerVolSummary] = Field(default_factory=list)
    avg_realized_vol: Optional[float] = None
    fear_greed_proxy: Optional[float] = None   # 100 - vix_percentile
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Realized vol helper
# ---------------------------------------------------------------------------

def _realized_vol(closes: np.ndarray, window: int) -> Optional[float]:
    """Close-to-close annualized realized vol over last *window* days."""
    if len(closes) < window + 1:
        return None
    tail = closes[-(window + 1):]
    log_rets = np.log(tail[1:] / tail[:-1])
    return float(np.std(log_rets) * np.sqrt(252))


# ---------------------------------------------------------------------------
# yfinance sync fetchers (called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _fetch_vix_data_sync(history_days: int) -> dict:
    """
    Batch-download VIX term structure tickers + SPX via yfinance.
    Returns raw dict of DataFrames keyed by ticker.
    """
    import yfinance as yf  # noqa: PLC0415

    period = f"{min(history_days + 60, 730)}d"  # pad for 60d history window
    tickers_to_fetch = _VIX_TICKERS + [_SPX_TICKER]

    try:
        raw = yf.download(
            tickers_to_fetch,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as exc:
        logger.warning("yf batch download failed", error=str(exc))
        return {}

    result: dict = {}
    for ticker in tickers_to_fetch:
        try:
            if len(tickers_to_fetch) > 1:
                df = raw[ticker] if ticker in raw.columns.get_level_values(0) else None
            else:
                df = raw
            if df is not None and not df.empty and "Close" in df.columns:
                result[ticker] = df["Close"].dropna()
        except Exception as exc:
            logger.warning("Ticker slice failed", ticker=ticker, error=str(exc))

    return result


def _fetch_ticker_vols_sync(tickers: list[str], history_days: int) -> dict[str, np.ndarray]:
    """Fetch close prices for user-supplied tickers for realized vol computation."""
    import yfinance as yf  # noqa: PLC0415

    period = f"{history_days + 30}d"
    result: dict[str, np.ndarray] = {}

    try:
        if len(tickers) == 1:
            raw = yf.download(tickers[0], period=period, auto_adjust=True, progress=False)
            if not raw.empty and "Close" in raw.columns:
                closes = raw["Close"].dropna().values
                if len(closes) > 0:
                    result[tickers[0]] = closes.astype(float)
        else:
            raw = yf.download(
                tickers, period=period, auto_adjust=True,
                progress=False, group_by="ticker", threads=True,
            )
            for t in tickers:
                try:
                    series = raw[t]["Close"].dropna() if t in raw.columns.get_level_values(0) else None
                    if series is not None and len(series) > 0:
                        result[t] = series.values.astype(float)
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("Ticker vol fetch failed", tickers=tickers, error=str(exc))

    return result


# ---------------------------------------------------------------------------
# Analytics builders
# ---------------------------------------------------------------------------

def _classify_vol_regime(vix: Optional[float]) -> str:
    if vix is None:
        return "unknown"
    if vix < 15:
        return "low vol"
    if vix < 20:
        return "normal"
    if vix < 25:
        return "elevated"
    if vix < 35:
        return "high stress"
    return "crisis"


def _classify_vvix_regime(vvix: Optional[float]) -> str:
    if vvix is None:
        return "normal"
    if vvix >= 120:
        return "extreme"
    if vvix >= 100:
        return "elevated"
    return "normal"


def _mean_reversion_signal(pct: Optional[float]) -> str:
    if pct is None:
        return "neutral"
    if pct > 80:
        return "revert lower"
    if pct < 20:
        return "revert higher"
    return "neutral"


def _safe_last(series) -> Optional[float]:  # type: ignore[type-arg]
    """Return last value of a pandas Series, or None if empty/NaN."""
    try:
        val = float(series.iloc[-1])
        return None if np.isnan(val) else round(val, 4)
    except Exception:
        return None


def _percentile_rank(series_values: np.ndarray, current: float) -> float:
    """Percentile of *current* vs *series_values* (0-100)."""
    return float(np.sum(series_values <= current) / len(series_values) * 100)


def _series_percentile(series, window: int = 252) -> Optional[float]:
    """Compute percentile of most recent value vs prior *window* observations."""
    try:
        vals = series.dropna().values.astype(float)
        if len(vals) < 2:
            return None
        tail = vals[-min(window, len(vals)):]
        current = tail[-1]
        hist = tail[:-1]
        if len(hist) == 0:
            return None
        return round(_percentile_rank(hist, current), 2)
    except Exception:
        return None


def _build_history_60d(vix_series, vix3m_series) -> list[VIXHistory]:
    """Align VIX and VIX3M by date for last 60 trading days."""
    history: list[VIXHistory] = []
    try:
        vix_vals = vix_series.tail(60) if vix_series is not None else None
        vix3m_vals = vix3m_series.tail(60) if vix3m_series is not None else None

        if vix_vals is None or len(vix_vals) == 0:
            return history

        for ts in vix_vals.index:
            date_str = ts.strftime("%Y-%m-%d")
            vix_pt = float(vix_vals.get(ts, float("nan")))
            vix_pt = None if np.isnan(vix_pt) else round(vix_pt, 4)

            vix3m_pt: Optional[float] = None
            if vix3m_vals is not None and ts in vix3m_vals.index:
                raw = float(vix3m_vals[ts])
                vix3m_pt = None if np.isnan(raw) else round(raw, 4)

            slope: Optional[float] = None
            if vix_pt is not None and vix3m_pt is not None and vix_pt != 0:
                slope = round((vix3m_pt - vix_pt) / vix_pt * 100, 4)

            history.append(VIXHistory(date=date_str, vix=vix_pt, vix3m=vix3m_pt, term_slope=slope))

    except Exception as exc:
        logger.warning("history_60d build failed", error=str(exc))

    return history


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def get_vix_analytics(history_days: int = 252) -> VIXAnalytics:
    """
    Fetch and compute VIX term structure, VRP, regime, and mean-reversion signal.
    All data sourced from yfinance (free). Safe to call concurrently.
    """
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    data = await asyncio.to_thread(_fetch_vix_data_sync, history_days)

    if not data:
        warnings.append("All VIX data fetch failed — yfinance unavailable")
        return VIXAnalytics(
            term_structure=[], as_of=as_of, warnings=warnings,
            vol_regime="unknown", mean_reversion_signal="neutral", vvix_regime="normal",
        )

    vix_s   = data.get("^VIX")
    vix3m_s = data.get("^VIX3M")
    vix6m_s = data.get("^VIX6M")
    vix1y_s = data.get("^VIX1Y")
    vvix_s  = data.get("^VVIX")
    skew_s  = data.get("^SKEW")
    spx_s   = data.get(_SPX_TICKER)

    if vix_s is None or len(vix_s) == 0:
        warnings.append("^VIX data unavailable — all downstream signals degraded")
    if vix1y_s is None or len(vix1y_s) == 0:
        warnings.append("^VIX1Y unavailable (often not published) — skipped")
    if vvix_s is None or len(vvix_s) == 0:
        warnings.append("^VVIX unavailable — VVIX regime set to normal")

    spot_vix   = _safe_last(vix_s) if vix_s is not None else None
    vix3m_val  = _safe_last(vix3m_s) if vix3m_s is not None else None
    vix6m_val  = _safe_last(vix6m_s) if vix6m_s is not None else None
    vix1y_val  = _safe_last(vix1y_s) if vix1y_s is not None else None
    vvix_val   = _safe_last(vvix_s) if vvix_s is not None else None
    skew_val   = _safe_last(skew_s) if skew_s is not None else None

    # Term slope: (VIX3M - VIX) / VIX * 100
    term_slope: Optional[float] = None
    contango:   Optional[bool]  = None
    if spot_vix is not None and vix3m_val is not None and spot_vix != 0:
        term_slope = round((vix3m_val - spot_vix) / spot_vix * 100, 4)
        contango   = vix3m_val > spot_vix

    # VIX percentile vs history
    vix_pct: Optional[float] = None
    if vix_s is not None and spot_vix is not None:
        vix_pct = _series_percentile(vix_s, window=history_days)

    # Realized vol on SPX/GSPC (30d close-to-close annualized)
    rv30: Optional[float] = None
    if spx_s is not None and len(spx_s) >= 32:
        rv30_raw = _realized_vol(spx_s.values.astype(float), window=30)
        if rv30_raw is not None:
            rv30 = round(rv30_raw * 100, 4)  # express in VIX-equivalent % units

    vrp: Optional[float] = None
    if spot_vix is not None and rv30 is not None:
        vrp = round(spot_vix - rv30, 4)

    # Term structure points with percentiles
    term_structure: list[VIXTermPoint] = [
        VIXTermPoint(
            tenor="spot",
            level=spot_vix,
            percentile=vix_pct,
        ),
        VIXTermPoint(
            tenor="3M",
            level=vix3m_val,
            percentile=_series_percentile(vix3m_s, history_days) if vix3m_s is not None else None,
        ),
        VIXTermPoint(
            tenor="6M",
            level=vix6m_val,
            percentile=_series_percentile(vix6m_s, history_days) if vix6m_s is not None else None,
        ),
        VIXTermPoint(
            tenor="1Y",
            level=vix1y_val,
            percentile=_series_percentile(vix1y_s, history_days) if vix1y_s is not None else None,
        ),
    ]

    history_60d = _build_history_60d(vix_s, vix3m_s)

    return VIXAnalytics(
        term_structure=term_structure,
        spot_vix=spot_vix,
        vix3m=vix3m_val,
        vix6m=vix6m_val,
        vix1y=vix1y_val,
        vvix=vvix_val,
        skew_index=skew_val,
        term_slope_pct=term_slope,
        contango=contango,
        vix_percentile=vix_pct,
        vrp=vrp,
        realized_vol_30d_spy=rv30,
        vol_regime=_classify_vol_regime(spot_vix),
        mean_reversion_signal=_mean_reversion_signal(vix_pct),
        vvix_regime=_classify_vvix_regime(vvix_val),
        history_60d=history_60d,
        as_of=as_of,
        warnings=warnings,
    )


async def get_vol_regime(
    tickers: list[str] | None = None,
    history_days: int = 252,
) -> VolRegime:
    """
    Return market vol regime (from VIX) plus per-ticker realized vol summary.
    Fetches VIX analytics and ticker price history concurrently.
    """
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved_tickers = tickers if tickers else _DEFAULT_TICKERS
    warnings: list[str] = []

    vix_task     = get_vix_analytics(history_days)
    ticker_task  = asyncio.to_thread(_fetch_ticker_vols_sync, resolved_tickers, history_days)

    vix_analytics, ticker_closes = await asyncio.gather(vix_task, ticker_task)

    warnings.extend(vix_analytics.warnings)

    spot_vix = vix_analytics.spot_vix
    ticker_summaries: list[TickerVolSummary] = []

    for t in resolved_tickers:
        closes = ticker_closes.get(t)
        if closes is None or len(closes) < 22:
            warnings.append(f"{t}: insufficient price history for realized vol")
            ticker_summaries.append(TickerVolSummary(ticker=t))
            continue

        rv20_raw = _realized_vol(closes, window=20)
        rv60_raw = _realized_vol(closes, window=60)

        rv20 = round(rv20_raw * 100, 4) if rv20_raw is not None else None
        rv60 = round(rv60_raw * 100, 4) if rv60_raw is not None else None

        iv_rv: Optional[float] = None
        if spot_vix is not None and rv20 is not None:
            iv_rv = round(spot_vix - rv20, 4)

        ticker_summaries.append(TickerVolSummary(
            ticker=t,
            realized_vol_20d=rv20,
            realized_vol_60d=rv60,
            iv_rv_premium=iv_rv,
        ))

    # Average realized vol across tickers (20d)
    rv20_vals = [s.realized_vol_20d for s in ticker_summaries if s.realized_vol_20d is not None]
    avg_rv: Optional[float] = round(float(np.mean(rv20_vals)), 4) if rv20_vals else None

    fear_greed: Optional[float] = (
        round(100 - vix_analytics.vix_percentile, 2)
        if vix_analytics.vix_percentile is not None else None
    )

    return VolRegime(
        market_regime=vix_analytics.vol_regime,
        spot_vix=spot_vix,
        vix_percentile=vix_analytics.vix_percentile,
        vrp=vix_analytics.vrp,
        ticker_vols=ticker_summaries,
        avg_realized_vol=avg_rv,
        fear_greed_proxy=fear_greed,
        as_of=as_of,
        warnings=warnings,
    )
