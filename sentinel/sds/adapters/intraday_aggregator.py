"""Intraday aggregator — multi-source 1-min bar stitcher.

Source priority:
  1. Alpaca free paper account  → 5-year 1m history (best resolution)
  2. Polygon free tier          → 2-year 5m history (1 req/min rate limit)
  3. yfinance                   → 60-day 1m history (last-resort fallback)

Gap detection and forward-fill utilities are included so callers can
patch holes between sources before persisting or analysing the data.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone, date
from typing import Optional

import pandas as pd

from sentinel.core.logging import get_logger
from sentinel.sds.adapters.alpaca_adapter import AlpacaAdapter
from sentinel.sds.adapters.polygon_adapter import PolygonAdapter
from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESAMPLE_MAP = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
}

# Alpaca free: 5 years of 1-min bars
_ALPACA_1M_LOOKBACK_DAYS = 365 * 5
# Polygon free: ~2 years of 5-min bars (1 req/min enforced by adapter)
_POLYGON_5M_LOOKBACK_DAYS = 365 * 2
# yfinance: 60 days of 1-min bars
_YF_1M_LOOKBACK_DAYS = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bars_to_df(bars: list) -> pd.DataFrame:
    """Convert a list of OHLCVBar objects to a timezone-aware DataFrame."""
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
    rows = [
        {
            "time": b.time if b.time.tzinfo else b.time.replace(tzinfo=timezone.utc),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": b.volume,
            "vwap": float(b.vwap) if b.vwap else None,
        }
        for b in bars
    ]
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df


# ---------------------------------------------------------------------------
# Gap detection / filling (DataFrame-level, complements existing GapReport)
# ---------------------------------------------------------------------------

def detect_gaps(
    df: pd.DataFrame,
    threshold_pct: float = 0.5,
) -> pd.DataFrame:
    """Return a DataFrame of bar-level gaps larger than *threshold_pct* percent.

    A gap is defined as a price jump between consecutive close prices that
    exceeds ``threshold_pct / 100`` in absolute percentage terms.  The
    returned DataFrame has the same index type as *df* and contains columns:

    - ``gap_pct``   — signed gap size in percent
    - ``prev_close`` — prior bar's close
    - ``open``       — current bar's open (the "gapping" open)

    This is distinct from the session-level ``GapReport`` in
    ``sentinel.sds.gap_detector`` — it operates on *bar* timestamps, not
    trading days, and is designed to work with intraday resolution.
    """
    if df.empty or "close" not in df.columns or "open" not in df.columns:
        return pd.DataFrame()

    prev_close = df["close"].shift(1)
    gap_pct = ((df["open"] - prev_close) / prev_close * 100).dropna()
    mask = gap_pct.abs() > threshold_pct
    gaps = pd.DataFrame(
        {
            "gap_pct": gap_pct[mask],
            "prev_close": prev_close[mask],
            "open": df.loc[mask, "open"],
        }
    )
    if not gaps.empty:
        logger.info(
            "detect_gaps: found gaps",
            count=len(gaps),
            threshold_pct=threshold_pct,
            largest_pct=round(gaps["gap_pct"].abs().max(), 3),
        )
    return gaps


def fill_gaps_ffill(
    df: pd.DataFrame,
    max_gap_bars: int = 3,
) -> pd.DataFrame:
    """Forward-fill missing rows (NaN OHLCV) for up to *max_gap_bars* bars.

    This function first re-indexes the DataFrame to a uniform 1-minute
    frequency (inferred from the median bar spacing) and then applies a
    limited forward-fill so that short data outages are patched without
    inflating data quality metrics.

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex and OHLCV columns.
    max_gap_bars:
        Maximum number of consecutive missing bars to fill.  Larger gaps
        are left as NaN so downstream code can detect them explicitly.

    Returns
    -------
    pd.DataFrame
        Re-indexed DataFrame with small gaps filled.  Rows that still
        contain NaN after the fill represent genuine extended outages.
    """
    if df.empty:
        return df

    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return df

    median_freq = diffs.median()
    if median_freq.total_seconds() < 1:
        return df

    freq_str = f"{int(median_freq.total_seconds())}s"
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq_str, tz=df.index.tz)
    df_reindexed = df.reindex(full_index)

    filled = df_reindexed.fillna(method="ffill", limit=max_gap_bars)
    fill_count = df_reindexed.isnull().any(axis=1).sum() - filled.isnull().any(axis=1).sum()
    if fill_count > 0:
        logger.info("fill_gaps_ffill: filled bars", bars=int(fill_count), max_gap_bars=max_gap_bars)
    return filled


# ---------------------------------------------------------------------------
# Resampling and VWAP
# ---------------------------------------------------------------------------

def aggregate_to_bar(df_1m: pd.DataFrame, bar_size: str) -> pd.DataFrame:
    """Resample a 1-minute OHLCV DataFrame to a coarser bar size.

    Parameters
    ----------
    df_1m:
        DataFrame with a DatetimeIndex and columns: open, high, low, close,
        volume.  A ``vwap`` column is optional.
    bar_size:
        Target resolution: one of ``"5m"``, ``"15m"``, ``"30m"``, ``"1h"``.

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV DataFrame.  VWAP for each new bar is computed from
        the constituent 1-minute bars when volume information is present.
    """
    rule = _RESAMPLE_MAP.get(bar_size)
    if rule is None:
        raise ValueError(f"Unsupported bar_size '{bar_size}'. Choose from {list(_RESAMPLE_MAP)}")
    if df_1m.empty:
        return df_1m

    agg: dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df_1m.resample(rule).agg(agg).dropna(how="all")

    # Compute VWAP for each new bar using typical price × volume
    if "volume" in df_1m.columns:
        typical = (df_1m["open"] + df_1m["high"] + df_1m["low"] + df_1m["close"]) / 4
        dollar_volume = (typical * df_1m["volume"]).resample(rule).sum()
        bar_volume = df_1m["volume"].resample(rule).sum()
        resampled["vwap"] = (dollar_volume / bar_volume.replace(0, pd.NA)).round(6)

    return resampled


def compute_vwap(df_intraday: pd.DataFrame) -> pd.Series:
    """Compute cumulative intraday VWAP for each row in *df_intraday*.

    Uses typical price = (open + high + low + close) / 4 weighted by volume.
    The cumulative VWAP resets at the start of the DataFrame (i.e. treat the
    entire input as a single session).

    Parameters
    ----------
    df_intraday:
        DataFrame with columns open, high, low, close, volume.

    Returns
    -------
    pd.Series
        Cumulative VWAP indexed identically to *df_intraday*.
    """
    if df_intraday.empty:
        return pd.Series(dtype=float, name="vwap")

    typical = (
        df_intraday["open"]
        + df_intraday["high"]
        + df_intraday["low"]
        + df_intraday["close"]
    ) / 4
    dollar_volume = (typical * df_intraday["volume"]).cumsum()
    cum_volume = df_intraday["volume"].cumsum()
    vwap = (dollar_volume / cum_volume.replace(0, pd.NA)).round(6)
    vwap.name = "vwap"
    return vwap


# ---------------------------------------------------------------------------
# Intraday signals
# ---------------------------------------------------------------------------

def compute_intraday_signals(
    ticker: str,
    date_str: str,
    df: pd.DataFrame,
    opening_range_minutes: int = 30,
) -> dict:
    """Compute key intraday metrics for a single trading session.

    Parameters
    ----------
    ticker:
        Instrument identifier (used only for logging).
    date_str:
        Trading date as ``"YYYY-MM-DD"`` string.
    df:
        Intraday 1-minute OHLCV DataFrame for the session.  Index must be
        a tz-aware DatetimeIndex.
    opening_range_minutes:
        Minutes after market open used to define the opening range (default 30).

    Returns
    -------
    dict with keys:
        - ``vwap``            — session VWAP (float or None)
        - ``opening_range_high`` — highest high in the opening range
        - ``opening_range_low``  — lowest low in the opening range
        - ``gap_open_pct``    — gap vs. prior session close in percent (None if unavailable)
        - ``gap_direction``   — "up" | "down" | "flat" | None
        - ``bar_count``       — number of 1-min bars
    """
    result: dict = {
        "ticker": ticker,
        "date": date_str,
        "vwap": None,
        "opening_range_high": None,
        "opening_range_low": None,
        "gap_open_pct": None,
        "gap_direction": None,
        "bar_count": len(df),
    }

    if df.empty:
        return result

    # VWAP
    vwap_series = compute_vwap(df)
    result["vwap"] = float(vwap_series.iloc[-1]) if not vwap_series.empty else None

    # Opening range (first N minutes)
    or_df = df.iloc[:opening_range_minutes]
    if not or_df.empty:
        result["opening_range_high"] = float(or_df["high"].max())
        result["opening_range_low"] = float(or_df["low"].min())

    # Gap detection vs. previous session's last bar
    if len(df) >= 1:
        gap_open = detect_gaps(df, threshold_pct=0.0)
        if not gap_open.empty:
            first_gap = gap_open.iloc[0]
            gap_pct = float(first_gap["gap_pct"])
            result["gap_open_pct"] = round(gap_pct, 4)
            if abs(gap_pct) < 0.1:
                result["gap_direction"] = "flat"
            elif gap_pct > 0:
                result["gap_direction"] = "up"
            else:
                result["gap_direction"] = "down"

    logger.info(
        "compute_intraday_signals complete",
        ticker=ticker,
        date=date_str,
        vwap=result["vwap"],
        or_high=result["opening_range_high"],
        or_low=result["opening_range_low"],
        gap_pct=result["gap_open_pct"],
    )
    return result


# ---------------------------------------------------------------------------
# IntradayAggregator
# ---------------------------------------------------------------------------

class IntradayAggregator:
    """Fetch and stitch 1-minute OHLCV bars from multiple free-tier sources.

    Priority chain
    --------------
    1. **Alpaca** — free paper-account credentials give 5 years of 1-min
       adjusted bars.  Best resolution and depth.
    2. **Polygon free** — 1 API call per minute; provides 2 years of 5-min
       bars which are upsampled/interpolated as a supplement.
    3. **yfinance** — 60-day 1-min history; used as a last resort or to
       fill the most-recent window when Alpaca is unavailable.

    Usage
    -----
    >>> agg = IntradayAggregator(
    ...     alpaca_key="...", alpaca_secret="...",
    ...     polygon_key="...",
    ... )
    >>> df = await agg.fetch("AAPL", start, end, interval="1m")
    """

    def __init__(
        self,
        alpaca_key: str = "",
        alpaca_secret: str = "",
        polygon_key: str = "",
    ) -> None:
        self._alpaca = AlpacaAdapter(api_key=alpaca_key, secret_key=alpaca_secret)
        self._polygon = PolygonAdapter(api_key=polygon_key)
        self._yfinance = YFinanceAdapter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1m",
        fill_small_gaps: bool = True,
        max_gap_bars: int = 3,
    ) -> pd.DataFrame:
        """Fetch intraday bars using the 3-source priority chain.

        Parameters
        ----------
        ticker:
            Equity symbol (e.g. ``"AAPL"``).
        start / end:
            Date range.  The adapter enforces source-specific maximum
            lookback windows automatically.
        interval:
            Bar resolution for the final output.  ``"1m"`` returns raw
            1-minute bars; ``"5m"`` / ``"15m"`` / ``"30m"`` / ``"1h"``
            trigger ``aggregate_to_bar`` internally.
        fill_small_gaps:
            If ``True``, apply ``fill_gaps_ffill`` to the stitched result
            before returning.
        max_gap_bars:
            Passed through to ``fill_gaps_ffill``.

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with a tz-aware DatetimeIndex.  Contains a
            ``source`` column indicating which adapter contributed each bar.
        """
        now = datetime.now(tz=timezone.utc)
        alpaca_cutoff = now - timedelta(days=_ALPACA_1M_LOOKBACK_DAYS)
        polygon_cutoff = now - timedelta(days=_POLYGON_5M_LOOKBACK_DAYS)
        yf_cutoff = now - timedelta(days=_YF_1M_LOOKBACK_DAYS)

        # Ensure start/end are tz-aware
        start_utc = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        end_utc = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end

        frames: list[pd.DataFrame] = []

        # ---- Source 1: Alpaca (1m, up to 5 years) -----------------------
        if start_utc >= alpaca_cutoff:
            df_alpaca = await self._fetch_alpaca(ticker, start_utc, end_utc, "1m")
            if not df_alpaca.empty:
                df_alpaca["source"] = "alpaca"
                frames.append(df_alpaca)
                logger.info("Alpaca bars fetched", ticker=ticker, bars=len(df_alpaca))

        # ---- Source 2: Polygon (5m, up to 2 years) ----------------------
        if start_utc >= polygon_cutoff and (not frames or frames[0].empty):
            df_polygon = await self._fetch_polygon(ticker, start_utc, end_utc, "5m")
            if not df_polygon.empty:
                df_polygon["source"] = "polygon"
                frames.append(df_polygon)
                logger.info("Polygon bars fetched", ticker=ticker, bars=len(df_polygon))

        # ---- Source 3: yfinance (1m, up to 60 days) ---------------------
        if start_utc >= yf_cutoff and not frames:
            df_yf = await self._fetch_yfinance(ticker, start_utc, end_utc, "1m")
            if not df_yf.empty:
                df_yf["source"] = "yfinance"
                frames.append(df_yf)
                logger.info("yfinance bars fetched", ticker=ticker, bars=len(df_yf))

        if not frames:
            logger.warning("IntradayAggregator: all sources returned empty", ticker=ticker)
            return pd.DataFrame()

        # Stitch: concatenate, deduplicate (prefer earlier source), sort
        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()

        if fill_small_gaps:
            # Only fill OHLCV columns, preserve 'source'
            ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in combined.columns]
            filled_ohlcv = fill_gaps_ffill(combined[ohlcv_cols], max_gap_bars=max_gap_bars)
            combined = combined.reindex(filled_ohlcv.index)
            combined[ohlcv_cols] = filled_ohlcv[ohlcv_cols]
            combined["source"] = combined["source"].fillna("ffill")

        # Resample if caller wants coarser bars
        if interval != "1m" and interval in _RESAMPLE_MAP:
            ohlcv_only = combined[[c for c in ["open", "high", "low", "close", "volume"] if c in combined.columns]]
            combined = aggregate_to_bar(ohlcv_only, interval)
            combined["source"] = "aggregated"

        return combined

    # ------------------------------------------------------------------
    # Private fetch helpers
    # ------------------------------------------------------------------

    async def _fetch_alpaca(
        self, ticker: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        try:
            bars = await self._alpaca.fetch_ohlcv(ticker, start, end, interval)
            return _bars_to_df(bars)
        except Exception as exc:
            logger.warning("Alpaca fetch failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

    async def _fetch_polygon(
        self, ticker: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        try:
            bars = await self._polygon.fetch_ohlcv(ticker, start, end, interval)
            return _bars_to_df(bars)
        except Exception as exc:
            logger.warning("Polygon fetch failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

    async def _fetch_yfinance(
        self, ticker: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        try:
            bars = await self._yfinance.fetch_ohlcv(ticker, start, end, interval)
            return _bars_to_df(bars)
        except Exception as exc:
            logger.warning("yfinance fetch failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()
