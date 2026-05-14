"""Comprehensive technical screener — 30+ indicators including trend, momentum,
volume, volatility, support/resistance, and candlestick pattern detection."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, Literal

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ── Score weight configuration ─────────────────────────────────────────────────
# Weights must sum to 1.0
_SCORE_WEIGHTS = {
    "rsi":       0.15,
    "macd":      0.15,
    "trend":     0.25,
    "volume":    0.15,
    "bollinger": 0.10,
    "patterns":  0.20,
}

_RATING_THRESHOLDS = [
    (80.0, "Strong Buy"),
    (60.0, "Buy"),
    (40.0, "Neutral"),
    (20.0, "Sell"),
    (0.0,  "Strong Sell"),
]


# ── Pydantic models ────────────────────────────────────────────────────────────

class TechnicalSignal(BaseModel):
    name: str
    value: Optional[float] = None
    signal: Literal["bullish", "bearish", "neutral"]
    strength: float   # 0-1
    description: str


class TechnicalProfile(BaseModel):
    ticker: str
    as_of: date

    # Price action
    price: float
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

    # Trend
    trend_short: str   # "uptrend", "downtrend", "sideways"
    trend_medium: str
    trend_long: str
    above_200ma: bool
    golden_cross: bool   # 50 MA crossed above 200 MA recently (within 20 sessions)
    death_cross: bool

    # Momentum
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci_20: Optional[float] = None

    # Volume
    volume_ratio: Optional[float] = None   # today vs 20-day average
    obv_signal: Optional[str] = None       # "accumulation", "distribution", "neutral"

    # Volatility
    atr_14: Optional[float] = None
    bollinger_width: Optional[float] = None       # (upper - lower) / middle
    bollinger_position: Optional[float] = None    # (price - lower) / (upper - lower)

    # Support / Resistance
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    distance_to_52w_high_pct: Optional[float] = None
    distance_to_52w_low_pct: Optional[float] = None
    at_52w_high: bool = False
    at_52w_low: bool = False

    # Pattern recognition
    patterns_detected: list[str] = Field(default_factory=list)

    # Overall score
    technical_score: float    # 0-100
    technical_rating: str     # "Strong Buy" … "Strong Sell"
    signals: list[TechnicalSignal]


class ScreenCriteria(BaseModel):
    # Trend
    above_sma_50: Optional[bool] = None
    above_sma_200: Optional[bool] = None
    golden_cross_recent: Optional[bool] = None

    # Momentum
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    macd_bullish: Optional[bool] = None

    # Volume
    volume_ratio_min: Optional[float] = None

    # Volatility
    bollinger_squeeze: Optional[bool] = None   # width < 10% of middle

    # Score filter
    min_technical_score: Optional[float] = None
    max_technical_score: Optional[float] = None


class ScreenResult(BaseModel):
    criteria: ScreenCriteria
    n_screened: int
    n_passed: int
    results: list[TechnicalProfile]


# ── Core screener ──────────────────────────────────────────────────────────────

class TechnicalScreener:
    """Full-featured technical screener using yfinance OHLCV data."""

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout

    # ── Public API ─────────────────────────────────────────────────────────────

    async def analyze(self, ticker: str, days_history: int = 252) -> TechnicalProfile:
        """Download OHLCV history and compute all technical indicators."""
        end = date.today()
        start = end - timedelta(days=days_history + 60)  # buffer for indicator warmup

        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                ),
            )
        except Exception as exc:
            logger.warning("yfinance_download_failed", ticker=ticker, error=str(exc))
            raise ValueError(f"Failed to download data for {ticker}: {exc}") from exc

        if raw.empty or len(raw) < 30:
            raise ValueError(f"Insufficient data for {ticker}: {len(raw)} bars")

        # Flatten MultiIndex if present (single ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        closes = raw["Close"].dropna()
        highs = raw["High"].dropna()
        lows = raw["Low"].dropna()
        opens = raw["Open"].dropna()
        volumes = raw["Volume"].dropna()

        # Align all series to common index
        common_idx = closes.index.intersection(highs.index).intersection(lows.index)
        closes = closes.loc[common_idx]
        highs = highs.loc[common_idx]
        lows = lows.loc[common_idx]
        opens = opens.loc[common_idx]
        volumes = volumes.reindex(common_idx).fillna(0)

        price = float(closes.iloc[-1])
        as_of = closes.index[-1].date() if hasattr(closes.index[-1], "date") else end

        # ── Moving averages ──
        sma_20 = self.compute_sma(closes, 20) if len(closes) >= 20 else None
        sma_50 = self.compute_sma(closes, 50) if len(closes) >= 50 else None
        sma_200 = self.compute_sma(closes, 200) if len(closes) >= 200 else None
        ema_12 = self.compute_ema(closes, 12) if len(closes) >= 12 else None
        ema_26 = self.compute_ema(closes, 26) if len(closes) >= 26 else None

        # ── Trend classification ──
        trend_short, trend_medium, trend_long = self._classify_trend(
            price,
            sma_20 or price,
            sma_50 or price,
            sma_200 or price,
        )

        above_200ma = price > (sma_200 or 0)
        golden_cross, death_cross = self._detect_cross(closes)

        # ── Momentum ──
        rsi_14 = self.compute_rsi(closes, 14) if len(closes) >= 15 else None
        macd_val, macd_sig, macd_hist = (
            self.compute_macd(closes)
            if len(closes) >= 35 else (None, None, None)
        )
        stoch_k, stoch_d = (
            self.compute_stochastic(highs, lows, closes)
            if len(closes) >= 17 else (None, None)
        )
        williams_r = (
            self._compute_williams_r(highs, lows, closes, 14)
            if len(closes) >= 14 else None
        )
        cci_20 = (
            self._compute_cci(highs, lows, closes, 20)
            if len(closes) >= 20 else None
        )

        # ── Volume ──
        volume_ratio = self._compute_volume_ratio(volumes)
        obv_signal = self.compute_obv(closes, volumes)

        # ── Volatility ──
        atr_14 = (
            self.compute_atr(highs, lows, closes, 14)
            if len(closes) >= 15 else None
        )
        boll_upper, boll_middle, boll_lower = (
            self.compute_bollinger_bands(closes)
            if len(closes) >= 20 else (None, None, None)
        )
        bollinger_width = (
            round((boll_upper - boll_lower) / boll_middle, 4)
            if boll_upper and boll_middle and boll_middle > 0 else None
        )
        bollinger_position = (
            round((price - boll_lower) / (boll_upper - boll_lower), 4)
            if boll_upper and boll_lower and (boll_upper - boll_lower) > 0 else None
        )

        # ── Support / Resistance ──
        support, resistance = self.compute_support_resistance(highs, lows, closes)

        # 52-week metrics
        year_ago = closes.index[-1] - pd.DateOffset(days=252)
        ytd_closes = closes[closes.index >= year_ago]
        high_52w = float(ytd_closes.max()) if len(ytd_closes) > 0 else price
        low_52w = float(ytd_closes.min()) if len(ytd_closes) > 0 else price
        dist_high = round((price - high_52w) / high_52w * 100, 2) if high_52w > 0 else 0.0
        dist_low = round((price - low_52w) / low_52w * 100, 2) if low_52w > 0 else 0.0
        at_52w_high = abs(dist_high) < 2.0
        at_52w_low = abs(dist_low) < 2.0

        # ── Candlestick patterns ──
        patterns = self.detect_candlestick_patterns(opens, highs, lows, closes)

        # ── Build signal list ──
        signals = self._build_signals(
            price=price,
            rsi=rsi_14,
            macd=macd_val,
            macd_signal=macd_sig,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            williams_r=williams_r,
            cci=cci_20,
            volume_ratio=volume_ratio,
            obv_signal=obv_signal,
            bollinger_position=bollinger_position,
            bollinger_width=bollinger_width,
            trend_short=trend_short,
            trend_medium=trend_medium,
            trend_long=trend_long,
            golden_cross=golden_cross,
            death_cross=death_cross,
            at_52w_high=at_52w_high,
            at_52w_low=at_52w_low,
            patterns=patterns,
        )

        # ── Assemble partial profile (score computed next) ──
        profile = TechnicalProfile(
            ticker=ticker.upper(),
            as_of=as_of,
            price=round(price, 4),
            sma_20=round(sma_20, 4) if sma_20 else None,
            sma_50=round(sma_50, 4) if sma_50 else None,
            sma_200=round(sma_200, 4) if sma_200 else None,
            ema_12=round(ema_12, 4) if ema_12 else None,
            ema_26=round(ema_26, 4) if ema_26 else None,
            trend_short=trend_short,
            trend_medium=trend_medium,
            trend_long=trend_long,
            above_200ma=above_200ma,
            golden_cross=golden_cross,
            death_cross=death_cross,
            rsi_14=round(rsi_14, 2) if rsi_14 is not None else None,
            macd=round(macd_val, 4) if macd_val is not None else None,
            macd_signal=round(macd_sig, 4) if macd_sig is not None else None,
            macd_histogram=round(macd_hist, 4) if macd_hist is not None else None,
            stoch_k=round(stoch_k, 2) if stoch_k is not None else None,
            stoch_d=round(stoch_d, 2) if stoch_d is not None else None,
            williams_r=round(williams_r, 2) if williams_r is not None else None,
            cci_20=round(cci_20, 2) if cci_20 is not None else None,
            volume_ratio=round(volume_ratio, 4) if volume_ratio is not None else None,
            obv_signal=obv_signal,
            atr_14=round(atr_14, 4) if atr_14 is not None else None,
            bollinger_width=bollinger_width,
            bollinger_position=bollinger_position,
            support_level=round(support, 4) if support else None,
            resistance_level=round(resistance, 4) if resistance else None,
            distance_to_52w_high_pct=dist_high,
            distance_to_52w_low_pct=dist_low,
            at_52w_high=at_52w_high,
            at_52w_low=at_52w_low,
            patterns_detected=patterns,
            technical_score=0.0,    # computed below
            technical_rating="Neutral",
            signals=signals,
        )

        score, rating = self.compute_technical_score(profile)
        profile.technical_score = round(score, 2)
        profile.technical_rating = rating

        logger.info(
            "technical_screener.analyzed",
            ticker=ticker,
            score=profile.technical_score,
            rating=rating,
        )
        return profile

    async def screen(
        self,
        tickers: list[str],
        criteria: ScreenCriteria,
        max_concurrent: int = 10,
    ) -> ScreenResult:
        """Analyze all tickers concurrently and filter by criteria."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _safe_analyze(ticker: str) -> Optional[TechnicalProfile]:
            async with semaphore:
                try:
                    return await self.analyze(ticker)
                except Exception as exc:
                    logger.warning(
                        "technical_screener.skip",
                        ticker=ticker,
                        reason=str(exc),
                    )
                    return None

        tasks = [_safe_analyze(t) for t in tickers]
        results = await asyncio.gather(*tasks)
        all_profiles = [p for p in results if p is not None]

        passed = [p for p in all_profiles if self._matches_criteria(p, criteria)]
        passed.sort(key=lambda p: p.technical_score, reverse=True)

        logger.info(
            "technical_screener.screen_complete",
            n_screened=len(tickers),
            n_analyzed=len(all_profiles),
            n_passed=len(passed),
        )

        return ScreenResult(
            criteria=criteria,
            n_screened=len(tickers),
            n_passed=len(passed),
            results=passed,
        )

    # ── Indicator computation ──────────────────────────────────────────────────

    def compute_sma(self, closes: pd.Series, window: int) -> float:
        """Simple Moving Average — arithmetic mean of last `window` closes."""
        if len(closes) < window:
            return float(closes.mean())
        return float(closes.iloc[-window:].mean())

    def compute_ema(self, closes: pd.Series, window: int) -> float:
        """Exponential Moving Average using pandas ewm (span=window)."""
        if len(closes) < 2:
            return float(closes.iloc[-1])
        ema_series = closes.ewm(span=window, adjust=False).mean()
        return float(ema_series.iloc[-1])

    def compute_rsi(self, closes: pd.Series, window: int = 14) -> float:
        """Wilder RSI.

        RS = average_gain / average_loss over `window` periods.
        RSI = 100 − 100 / (1 + RS).
        Uses Wilder smoothing (ewm with com=window-1).
        """
        delta = closes.diff().dropna()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def compute_macd(
        self,
        closes: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, float, float]:
        """MACD line, signal line, and histogram.

        MACD line   = EMA(fast) − EMA(slow)
        Signal line = EMA(MACD, signal_period)
        Histogram   = MACD − Signal
        """
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            float(histogram.iloc[-1]),
        )

    def compute_bollinger_bands(
        self,
        closes: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> tuple[float, float, float]:
        """Bollinger Bands: (upper, middle, lower).

        middle = SMA(window)
        upper  = middle + num_std × rolling_std
        lower  = middle − num_std × rolling_std
        """
        middle = closes.rolling(window).mean()
        std = closes.rolling(window).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return (
            float(upper.iloc[-1]),
            float(middle.iloc[-1]),
            float(lower.iloc[-1]),
        )

    def compute_atr(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        window: int = 14,
    ) -> float:
        """Average True Range (Wilder smoothing).

        TR = max(H−L, |H−prev_C|, |L−prev_C|).
        ATR = Wilder EMA of TR (com = window−1).
        """
        prev_close = closes.shift(1)
        tr = pd.concat([
            highs - lows,
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(com=window - 1, adjust=False).mean()
        return float(atr.iloc[-1])

    def compute_stochastic(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        k_window: int = 14,
        d_window: int = 3,
    ) -> tuple[float, float]:
        """Stochastic Oscillator (%K, %D).

        %K = (C − lowest_low(k)) / (highest_high(k) − lowest_low(k)) × 100
        %D = SMA(%K, d_window)
        """
        lowest_low = lows.rolling(k_window).min()
        highest_high = highs.rolling(k_window).max()
        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)
        pct_k = (closes - lowest_low) / denom * 100
        pct_d = pct_k.rolling(d_window).mean()
        return float(pct_k.iloc[-1]), float(pct_d.iloc[-1])

    def compute_obv(self, closes: pd.Series, volumes: pd.Series) -> str:
        """On-Balance Volume signal classification.

        OBV cumulates volume positively on up days, negatively on down days.
        Signal is determined by comparing 10-bar OBV trend to 10-bar price trend.
        Returns "accumulation", "distribution", or "neutral".
        """
        direction = np.sign(closes.diff().fillna(0))
        obv = (direction * volumes).cumsum()
        if len(obv) < 10:
            return "neutral"

        obv_slope = float(np.polyfit(range(10), obv.iloc[-10:].values, 1)[0])
        price_slope = float(np.polyfit(range(10), closes.iloc[-10:].values, 1)[0])

        if obv_slope > 0 and price_slope > 0:
            return "accumulation"
        elif obv_slope < 0 and price_slope < 0:
            return "distribution"
        elif obv_slope > 0 and price_slope <= 0:
            return "accumulation"   # bullish divergence
        elif obv_slope < 0 and price_slope >= 0:
            return "distribution"   # bearish divergence
        return "neutral"

    def detect_candlestick_patterns(
        self,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
    ) -> list[str]:
        """Detect common candlestick patterns from the last 5 bars.

        Detected patterns:
          - hammer, shooting_star, doji, spinning_top
          - bullish_engulfing, bearish_engulfing
          - morning_star, evening_star
          - three_white_soldiers, three_black_crows
          - bullish_harami, bearish_harami
          - dragonfly_doji, gravestone_doji
        """
        if len(closes) < 3:
            return []

        patterns: list[str] = []

        # Use last 5 bars for pattern detection
        o = opens.iloc[-5:].values.astype(float)
        h = highs.iloc[-5:].values.astype(float)
        l = lows.iloc[-5:].values.astype(float)
        c = closes.iloc[-5:].values.astype(float)

        def _body(i: int) -> float:
            return abs(c[i] - o[i])

        def _upper_shadow(i: int) -> float:
            return h[i] - max(o[i], c[i])

        def _lower_shadow(i: int) -> float:
            return min(o[i], c[i]) - l[i]

        def _range(i: int) -> float:
            return h[i] - l[i]

        def _is_bullish(i: int) -> bool:
            return c[i] > o[i]

        def _is_bearish(i: int) -> bool:
            return c[i] < o[i]

        last = len(c) - 1   # index of most recent bar

        # ── Single-bar patterns (applied to last bar) ──
        body = _body(last)
        rng = _range(last)
        upper = _upper_shadow(last)
        lower = _lower_shadow(last)

        if rng > 0:
            body_ratio = body / rng

            # Doji: body < 10% of range
            if body_ratio < 0.1:
                if lower > 2 * upper and lower > 0.6 * rng:
                    patterns.append("dragonfly_doji")
                elif upper > 2 * lower and upper > 0.6 * rng:
                    patterns.append("gravestone_doji")
                else:
                    patterns.append("doji")

            # Spinning top: small body, significant shadows on both sides
            elif body_ratio < 0.3 and upper > body * 0.5 and lower > body * 0.5:
                patterns.append("spinning_top")

            # Hammer: small body at top, long lower shadow, short upper shadow
            elif (lower > 2 * body and upper < body and
                  min(o[last], c[last]) > (l[last] + rng * 0.6)):
                patterns.append("hammer")

            # Shooting star: small body at bottom, long upper shadow, short lower shadow
            elif (upper > 2 * body and lower < body and
                  max(o[last], c[last]) < (l[last] + rng * 0.4)):
                patterns.append("shooting_star")

        # ── Two-bar patterns (last two bars) ──
        if last >= 1:
            prev = last - 1

            # Bullish engulfing
            if (_is_bearish(prev) and _is_bullish(last) and
                    o[last] < c[prev] and c[last] > o[prev]):
                patterns.append("bullish_engulfing")

            # Bearish engulfing
            elif (_is_bullish(prev) and _is_bearish(last) and
                  o[last] > c[prev] and c[last] < o[prev]):
                patterns.append("bearish_engulfing")

            # Bullish harami
            elif (_is_bearish(prev) and _is_bullish(last) and
                  o[last] > c[prev] and c[last] < o[prev]):
                patterns.append("bullish_harami")

            # Bearish harami
            elif (_is_bullish(prev) and _is_bearish(last) and
                  o[last] < c[prev] and c[last] > o[prev]):
                patterns.append("bearish_harami")

        # ── Three-bar patterns ──
        if last >= 2:
            p2, p1, p0 = last - 2, last - 1, last

            # Morning star (bullish reversal): large bearish, small body, large bullish
            if (_is_bearish(p2) and
                    _body(p1) < _body(p2) * 0.3 and
                    _is_bullish(p0) and
                    c[p0] > (o[p2] + c[p2]) / 2):
                patterns.append("morning_star")

            # Evening star (bearish reversal): large bullish, small body, large bearish
            elif (_is_bullish(p2) and
                  _body(p1) < _body(p2) * 0.3 and
                  _is_bearish(p0) and
                  c[p0] < (o[p2] + c[p2]) / 2):
                patterns.append("evening_star")

            # Three white soldiers: three consecutive bullish bars, each higher
            if (all(_is_bullish(i) for i in [p2, p1, p0]) and
                    c[p1] > c[p2] and c[p0] > c[p1] and
                    o[p1] > o[p2] and o[p0] > o[p1]):
                patterns.append("three_white_soldiers")

            # Three black crows: three consecutive bearish bars, each lower
            elif (all(_is_bearish(i) for i in [p2, p1, p0]) and
                  c[p1] < c[p2] and c[p0] < c[p1] and
                  o[p1] < o[p2] and o[p0] < o[p1]):
                patterns.append("three_black_crows")

        return patterns

    def compute_support_resistance(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        lookback: int = 50,
        tolerance_pct: float = 0.015,
    ) -> tuple[float, float]:
        """Pivot-based support and resistance levels.

        Algorithm:
          1. Find local minima in `lows` (support candidates) using a 5-bar window.
          2. Find local maxima in `highs` (resistance candidates) using a 5-bar window.
          3. Cluster nearby levels (within `tolerance_pct` of each other).
          4. Return the strongest support below current price and resistance above it.
        """
        if len(closes) < 10:
            price = float(closes.iloc[-1])
            return price * 0.97, price * 1.03

        price = float(closes.iloc[-1])
        recent_lows = lows.iloc[-lookback:]
        recent_highs = highs.iloc[-lookback:]

        # Local minima: lower than 2 bars on each side
        support_candidates: list[float] = []
        for i in range(2, len(recent_lows) - 2):
            v = float(recent_lows.iloc[i])
            if (v <= float(recent_lows.iloc[i - 1]) and
                    v <= float(recent_lows.iloc[i - 2]) and
                    v <= float(recent_lows.iloc[i + 1]) and
                    v <= float(recent_lows.iloc[i + 2])):
                support_candidates.append(v)

        # Local maxima
        resistance_candidates: list[float] = []
        for i in range(2, len(recent_highs) - 2):
            v = float(recent_highs.iloc[i])
            if (v >= float(recent_highs.iloc[i - 1]) and
                    v >= float(recent_highs.iloc[i - 2]) and
                    v >= float(recent_highs.iloc[i + 1]) and
                    v >= float(recent_highs.iloc[i + 2])):
                resistance_candidates.append(v)

        def _cluster_best(candidates: list[float], below_price: bool) -> float:
            if not candidates:
                return price * (0.95 if below_price else 1.05)
            # Filter to correct side of price
            filtered = [v for v in candidates if (v < price if below_price else v > price)]
            if not filtered:
                return price * (0.97 if below_price else 1.03)
            # Cluster by tolerance
            clusters: list[list[float]] = []
            for v in sorted(filtered):
                placed = False
                for cluster in clusters:
                    if abs(v - np.mean(cluster)) / np.mean(cluster) < tolerance_pct:
                        cluster.append(v)
                        placed = True
                        break
                if not placed:
                    clusters.append([v])
            # Strongest cluster = most members; pick level closest to price
            strongest = max(clusters, key=len)
            return float(np.mean(strongest))

        support = _cluster_best(support_candidates, below_price=True)
        resistance = _cluster_best(resistance_candidates, below_price=False)
        return support, resistance

    def compute_technical_score(
        self, profile: TechnicalProfile
    ) -> tuple[float, str]:
        """Weight individual signals into a 0-100 composite technical score.

        Weight breakdown (must match _SCORE_WEIGHTS):
          RSI:        15%
          MACD:       15%
          Trend:      25%
          Volume:     15%
          Bollinger:  10%
          Patterns:   20%
        """
        components: dict[str, float] = {}

        # ── RSI (15%) ──
        if profile.rsi_14 is not None:
            r = profile.rsi_14
            if r < 30:
                rsi_score = 90.0                          # strongly oversold → contrarian bullish
            elif r < 40:
                rsi_score = 70.0                          # mildly oversold
            elif r < 60:
                rsi_score = 50.0                          # neutral
            elif r < 70:
                rsi_score = 65.0                          # bullish momentum
            else:
                rsi_score = 30.0                          # overbought
            components["rsi"] = rsi_score
        else:
            components["rsi"] = 50.0

        # ── MACD (15%) ──
        if profile.macd is not None and profile.macd_signal is not None:
            if profile.macd > profile.macd_signal:
                if profile.macd > 0:
                    components["macd"] = 80.0             # above zero, bullish
                else:
                    components["macd"] = 65.0             # crossing up from negative
            else:
                if profile.macd < 0:
                    components["macd"] = 25.0             # below zero, bearish
                else:
                    components["macd"] = 40.0             # rolling over from positive
        else:
            components["macd"] = 50.0

        # ── Trend (25%) ──
        trend_scores = {"uptrend": 85.0, "sideways": 50.0, "downtrend": 20.0}
        short_s = trend_scores.get(profile.trend_short, 50.0)
        med_s = trend_scores.get(profile.trend_medium, 50.0)
        long_s = trend_scores.get(profile.trend_long, 50.0)
        trend_base = (short_s * 0.4 + med_s * 0.35 + long_s * 0.25)
        if profile.golden_cross:
            trend_base = min(trend_base + 10.0, 100.0)
        if profile.death_cross:
            trend_base = max(trend_base - 10.0, 0.0)
        components["trend"] = trend_base

        # ── Volume (15%) ──
        vr = profile.volume_ratio
        if vr is not None:
            if vr > 2.0:
                vol_score = 85.0                          # very high volume
            elif vr > 1.5:
                vol_score = 70.0
            elif vr > 1.0:
                vol_score = 55.0
            elif vr > 0.5:
                vol_score = 40.0
            else:
                vol_score = 25.0                          # very low volume
            # OBV adjustment
            if profile.obv_signal == "accumulation":
                vol_score = min(vol_score + 10.0, 100.0)
            elif profile.obv_signal == "distribution":
                vol_score = max(vol_score - 10.0, 0.0)
            components["volume"] = vol_score
        else:
            components["volume"] = 50.0

        # ── Bollinger (10%) ──
        bp = profile.bollinger_position
        bw = profile.bollinger_width
        if bp is not None:
            if bp < 0.2:
                boll_score = 80.0                         # near lower band → oversold
            elif bp < 0.4:
                boll_score = 65.0
            elif bp < 0.6:
                boll_score = 50.0
            elif bp < 0.8:
                boll_score = 40.0
            else:
                boll_score = 25.0                         # near upper band → overbought
            # Squeeze boost: tight bands → breakout pending (neutral signal itself)
            if bw is not None and bw < 0.1:
                boll_score = 55.0
            components["bollinger"] = boll_score
        else:
            components["bollinger"] = 50.0

        # ── Patterns (20%) ──
        bullish_patterns = {
            "hammer", "morning_star", "bullish_engulfing",
            "three_white_soldiers", "bullish_harami", "dragonfly_doji",
        }
        bearish_patterns = {
            "shooting_star", "evening_star", "bearish_engulfing",
            "three_black_crows", "bearish_harami", "gravestone_doji",
        }
        neutral_patterns = {"doji", "spinning_top"}

        detected = set(profile.patterns_detected)
        bull_count = len(detected & bullish_patterns)
        bear_count = len(detected & bearish_patterns)

        if bull_count > bear_count:
            pattern_score = min(60.0 + bull_count * 10.0, 95.0)
        elif bear_count > bull_count:
            pattern_score = max(40.0 - bear_count * 10.0, 10.0)
        else:
            pattern_score = 50.0
        components["patterns"] = pattern_score

        # ── Composite score ──
        score = sum(
            components[k] * _SCORE_WEIGHTS[k]
            for k in _SCORE_WEIGHTS
        )
        score = max(0.0, min(100.0, score))

        rating = "Neutral"
        for threshold, label in _RATING_THRESHOLDS:
            if score >= threshold:
                rating = label
                break

        return score, rating

    # ── Private helpers ────────────────────────────────────────────────────────

    def _classify_trend(
        self,
        price: float,
        sma20: float,
        sma50: float,
        sma200: float,
    ) -> tuple[str, str, str]:
        """Classify short / medium / long trend.

        Short  = price vs SMA20  (±1% tolerance for "sideways")
        Medium = price vs SMA50
        Long   = price vs SMA200
        """
        def _direction(p: float, ma: float) -> str:
            pct = (p - ma) / ma if ma > 0 else 0.0
            if pct > 0.01:
                return "uptrend"
            elif pct < -0.01:
                return "downtrend"
            return "sideways"

        return (
            _direction(price, sma20),
            _direction(price, sma50),
            _direction(price, sma200),
        )

    def _detect_cross(
        self, closes: pd.Series, lookback: int = 20
    ) -> tuple[bool, bool]:
        """Detect golden cross (50 SMA crosses above 200 SMA) or death cross within lookback bars."""
        if len(closes) < 200 + lookback:
            return False, False

        sma50 = closes.rolling(50).mean()
        sma200 = closes.rolling(200).mean()
        diff = sma50 - sma200
        recent_diff = diff.iloc[-lookback:]

        # Golden cross: diff went from negative to positive
        golden = bool(
            (recent_diff.iloc[-1] > 0) and
            (recent_diff.min() < 0)
        )
        # Death cross: diff went from positive to negative
        death = bool(
            (recent_diff.iloc[-1] < 0) and
            (recent_diff.max() > 0)
        )
        return golden, death

    def _compute_williams_r(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        window: int = 14,
    ) -> float:
        """Williams %R = (highest_high - close) / (highest_high - lowest_low) × -100.

        Range: -100 (oversold) to 0 (overbought).
        """
        hh = highs.rolling(window).max()
        ll = lows.rolling(window).min()
        denom = (hh - ll).replace(0, np.nan)
        wr = (hh - closes) / denom * -100
        return float(wr.iloc[-1])

    def _compute_cci(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        window: int = 20,
    ) -> float:
        """Commodity Channel Index.

        CCI = (Typical Price − SMA(TP)) / (0.015 × Mean Absolute Deviation).
        TP = (H + L + C) / 3.
        """
        tp = (highs + lows + closes) / 3
        sma_tp = tp.rolling(window).mean()
        mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
        return float(cci.iloc[-1])

    def _compute_volume_ratio(self, volumes: pd.Series, window: int = 20) -> Optional[float]:
        """Current volume vs 20-day average volume."""
        if len(volumes) < 2:
            return None
        avg = float(volumes.iloc[-window:].mean()) if len(volumes) >= window else float(volumes.mean())
        if avg < 1:
            return None
        return float(volumes.iloc[-1]) / avg

    def _build_signals(
        self,
        price: float,
        rsi: Optional[float],
        macd: Optional[float],
        macd_signal: Optional[float],
        stoch_k: Optional[float],
        stoch_d: Optional[float],
        williams_r: Optional[float],
        cci: Optional[float],
        volume_ratio: Optional[float],
        obv_signal: Optional[str],
        bollinger_position: Optional[float],
        bollinger_width: Optional[float],
        trend_short: str,
        trend_medium: str,
        trend_long: str,
        golden_cross: bool,
        death_cross: bool,
        at_52w_high: bool,
        at_52w_low: bool,
        patterns: list[str],
    ) -> list[TechnicalSignal]:
        """Assemble the full TechnicalSignal list."""
        signals: list[TechnicalSignal] = []

        # RSI
        if rsi is not None:
            if rsi < 30:
                sig, strength, desc = "bullish", 0.85, f"RSI {rsi:.1f} — oversold (<30)"
            elif rsi > 70:
                sig, strength, desc = "bearish", 0.85, f"RSI {rsi:.1f} — overbought (>70)"
            elif rsi < 45:
                sig, strength, desc = "bearish", 0.4, f"RSI {rsi:.1f} — mildly bearish"
            elif rsi > 55:
                sig, strength, desc = "bullish", 0.4, f"RSI {rsi:.1f} — mildly bullish"
            else:
                sig, strength, desc = "neutral", 0.2, f"RSI {rsi:.1f} — neutral"
            signals.append(TechnicalSignal(
                name="RSI(14)", value=round(rsi, 2),
                signal=sig, strength=strength, description=desc,
            ))

        # MACD
        if macd is not None and macd_signal is not None:
            hist = macd - macd_signal
            if macd > macd_signal and macd > 0:
                sig, strength, desc = "bullish", 0.8, "MACD above signal and zero line"
            elif macd > macd_signal:
                sig, strength, desc = "bullish", 0.55, "MACD crossed above signal"
            elif macd < macd_signal and macd < 0:
                sig, strength, desc = "bearish", 0.8, "MACD below signal and zero line"
            else:
                sig, strength, desc = "bearish", 0.55, "MACD crossed below signal"
            signals.append(TechnicalSignal(
                name="MACD", value=round(macd, 4),
                signal=sig, strength=strength, description=desc,
            ))

        # Stochastic
        if stoch_k is not None and stoch_d is not None:
            if stoch_k < 20:
                sig, strength = "bullish", 0.75
                desc = f"Stoch %K {stoch_k:.1f} — oversold"
            elif stoch_k > 80:
                sig, strength = "bearish", 0.75
                desc = f"Stoch %K {stoch_k:.1f} — overbought"
            else:
                sig, strength = "neutral", 0.3
                desc = f"Stoch %K {stoch_k:.1f}"
            signals.append(TechnicalSignal(
                name="Stochastic(%K)", value=round(stoch_k, 2),
                signal=sig, strength=strength, description=desc,
            ))

        # Williams %R
        if williams_r is not None:
            if williams_r < -80:
                sig, strength = "bullish", 0.7
                desc = f"Williams %R {williams_r:.1f} — oversold"
            elif williams_r > -20:
                sig, strength = "bearish", 0.7
                desc = f"Williams %R {williams_r:.1f} — overbought"
            else:
                sig, strength = "neutral", 0.25
                desc = f"Williams %R {williams_r:.1f}"
            signals.append(TechnicalSignal(
                name="Williams%R", value=round(williams_r, 2),
                signal=sig, strength=strength, description=desc,
            ))

        # CCI
        if cci is not None:
            if cci < -100:
                sig, strength = "bullish", 0.65
                desc = f"CCI {cci:.1f} — oversold territory"
            elif cci > 100:
                sig, strength = "bearish", 0.65
                desc = f"CCI {cci:.1f} — overbought territory"
            else:
                sig, strength = "neutral", 0.2
                desc = f"CCI {cci:.1f}"
            signals.append(TechnicalSignal(
                name="CCI(20)", value=round(cci, 2),
                signal=sig, strength=strength, description=desc,
            ))

        # Volume
        if volume_ratio is not None:
            if volume_ratio > 2.0:
                sig, strength = "bullish", 0.7
                desc = f"Volume {volume_ratio:.1f}x avg — strong interest"
            elif volume_ratio > 1.3:
                sig, strength = "bullish", 0.4
                desc = f"Volume {volume_ratio:.1f}x avg — above average"
            elif volume_ratio < 0.5:
                sig, strength = "bearish", 0.4
                desc = f"Volume {volume_ratio:.1f}x avg — very light"
            else:
                sig, strength = "neutral", 0.2
                desc = f"Volume {volume_ratio:.1f}x avg"
            signals.append(TechnicalSignal(
                name="Volume Ratio", value=round(volume_ratio, 2),
                signal=sig, strength=strength, description=desc,
            ))

        # OBV
        if obv_signal:
            sig_map = {
                "accumulation": ("bullish", 0.6, "OBV showing accumulation"),
                "distribution": ("bearish", 0.6, "OBV showing distribution"),
                "neutral":      ("neutral", 0.2, "OBV neutral"),
            }
            s, st, d = sig_map.get(obv_signal, ("neutral", 0.2, "OBV unknown"))
            signals.append(TechnicalSignal(
                name="OBV", value=None, signal=s, strength=st, description=d,
            ))

        # Bollinger bands
        if bollinger_position is not None:
            if bollinger_position < 0:
                sig, strength = "bullish", 0.8
                desc = f"Price below lower Bollinger band (position {bollinger_position:.2f})"
            elif bollinger_position > 1:
                sig, strength = "bearish", 0.8
                desc = f"Price above upper Bollinger band (position {bollinger_position:.2f})"
            elif bollinger_position < 0.2:
                sig, strength = "bullish", 0.5
                desc = f"Price near lower Bollinger band"
            elif bollinger_position > 0.8:
                sig, strength = "bearish", 0.5
                desc = f"Price near upper Bollinger band"
            else:
                sig, strength = "neutral", 0.2
                desc = f"Bollinger position {bollinger_position:.2f}"
            signals.append(TechnicalSignal(
                name="Bollinger Position", value=bollinger_position,
                signal=sig, strength=strength, description=desc,
            ))

        if bollinger_width is not None and bollinger_width < 0.1:
            signals.append(TechnicalSignal(
                name="Bollinger Squeeze", value=bollinger_width,
                signal="neutral", strength=0.5,
                description="Bollinger squeeze — volatility breakout pending",
            ))

        # Trend signals
        trend_map = {
            "uptrend": ("bullish", 0.7),
            "downtrend": ("bearish", 0.7),
            "sideways": ("neutral", 0.2),
        }
        for label, trend in [("Short", trend_short), ("Medium", trend_medium), ("Long", trend_long)]:
            s, st = trend_map.get(trend, ("neutral", 0.2))
            signals.append(TechnicalSignal(
                name=f"{label}-Term Trend", value=None,
                signal=s, strength=st, description=f"{label}-term trend: {trend}",
            ))

        # Cross signals
        if golden_cross:
            signals.append(TechnicalSignal(
                name="Golden Cross", value=None, signal="bullish", strength=0.9,
                description="50 SMA crossed above 200 SMA — strong bullish signal",
            ))
        if death_cross:
            signals.append(TechnicalSignal(
                name="Death Cross", value=None, signal="bearish", strength=0.9,
                description="50 SMA crossed below 200 SMA — strong bearish signal",
            ))

        # 52-week extremes
        if at_52w_high:
            signals.append(TechnicalSignal(
                name="52-Week High", value=None, signal="bullish", strength=0.75,
                description="Price within 2% of 52-week high — strong momentum",
            ))
        if at_52w_low:
            signals.append(TechnicalSignal(
                name="52-Week Low", value=None, signal="bearish", strength=0.75,
                description="Price within 2% of 52-week low — significant weakness",
            ))

        # Pattern signals
        bullish_pats = {
            "hammer", "morning_star", "bullish_engulfing",
            "three_white_soldiers", "bullish_harami", "dragonfly_doji",
        }
        bearish_pats = {
            "shooting_star", "evening_star", "bearish_engulfing",
            "three_black_crows", "bearish_harami", "gravestone_doji",
        }
        for p in patterns:
            if p in bullish_pats:
                signals.append(TechnicalSignal(
                    name=f"Pattern: {p.replace('_', ' ').title()}",
                    value=None, signal="bullish", strength=0.65,
                    description=f"Bullish candlestick pattern detected: {p}",
                ))
            elif p in bearish_pats:
                signals.append(TechnicalSignal(
                    name=f"Pattern: {p.replace('_', ' ').title()}",
                    value=None, signal="bearish", strength=0.65,
                    description=f"Bearish candlestick pattern detected: {p}",
                ))
            else:
                signals.append(TechnicalSignal(
                    name=f"Pattern: {p.replace('_', ' ').title()}",
                    value=None, signal="neutral", strength=0.3,
                    description=f"Neutral candlestick pattern detected: {p}",
                ))

        return signals

    def _matches_criteria(
        self, profile: TechnicalProfile, criteria: ScreenCriteria
    ) -> bool:
        """Return True if the profile satisfies all non-None criteria."""
        if criteria.above_sma_50 is not None:
            sma50 = profile.sma_50
            above = sma50 is not None and profile.price > sma50
            if above != criteria.above_sma_50:
                return False

        if criteria.above_sma_200 is not None:
            if profile.above_200ma != criteria.above_sma_200:
                return False

        if criteria.golden_cross_recent is not None:
            if profile.golden_cross != criteria.golden_cross_recent:
                return False

        if criteria.rsi_min is not None:
            if profile.rsi_14 is None or profile.rsi_14 < criteria.rsi_min:
                return False

        if criteria.rsi_max is not None:
            if profile.rsi_14 is None or profile.rsi_14 > criteria.rsi_max:
                return False

        if criteria.macd_bullish is not None:
            macd_bull = (
                profile.macd is not None and
                profile.macd_signal is not None and
                profile.macd > profile.macd_signal
            )
            if macd_bull != criteria.macd_bullish:
                return False

        if criteria.volume_ratio_min is not None:
            if profile.volume_ratio is None or profile.volume_ratio < criteria.volume_ratio_min:
                return False

        if criteria.bollinger_squeeze is not None:
            is_squeeze = (
                profile.bollinger_width is not None and
                profile.bollinger_width < 0.1
            )
            if is_squeeze != criteria.bollinger_squeeze:
                return False

        if criteria.min_technical_score is not None:
            if profile.technical_score < criteria.min_technical_score:
                return False

        if criteria.max_technical_score is not None:
            if profile.technical_score > criteria.max_technical_score:
                return False

        return True


# ── Module-level convenience helpers ──────────────────────────────────────────

async def technical_profile(ticker: str, days_history: int = 252) -> TechnicalProfile:
    """Analyze a single ticker and return its TechnicalProfile."""
    screener = TechnicalScreener()
    return await screener.analyze(ticker, days_history=days_history)


async def technical_screen(
    tickers: list[str],
    criteria: dict,
    max_concurrent: int = 10,
) -> ScreenResult:
    """Screen a list of tickers against the given criteria dict."""
    screener = TechnicalScreener()
    screen_criteria = ScreenCriteria(**criteria)
    return await screener.screen(tickers, screen_criteria, max_concurrent=max_concurrent)
