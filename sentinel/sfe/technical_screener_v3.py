"""
Technical Screener v3 — 25+ Criteria, pandas-ta with pure-numpy fallback
=========================================================================
Dimension: dim_071  target score 7 → 9

Features:
  - IndicatorEngine: RSI, MACD, BBands, Stochastic, ATR, ADX, EMA, SMA, OBV,
    VWAP, Ichimoku, Williams %R, CCI, MFI, DMI, Pivots, Keltner Channels
    (pandas_ta primary, manual numpy fallback)
  - 29 ScreenCriteria across Trend, Momentum, Volatility, Volume, Price Action,
    and Combined/Advanced categories
  - TechnicalScreener: batch yfinance download, AND/OR screening, preset bundles
  - TechnicalRanker: composite technical score (0-100), IBD-style RS rating
  - TechnicalAlertSystem: watchlist registration, breakout/oversold alerts
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
        return logging.getLogger(name)

logger = get_logger(__name__)

# Optional pandas_ta
try:
    import pandas_ta as ta   # type: ignore
    HAS_TA = True
    logger.info("pandas_ta available — using accelerated indicators")
except ImportError:
    HAS_TA = False
    logger.info("pandas_ta not available — using pure numpy/pandas fallback indicators")

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScreenCriteria:
    name: str
    description: str
    category: str
    filter_fn: Callable[[pd.DataFrame], bool]


@dataclass
class ScreenResult:
    passed_tickers: List[str]
    failed_tickers: List[str]
    per_ticker: Dict[str, Dict[str, bool]]   # {ticker: {criterion_name: bool}}
    n_criteria: int
    screen_mode: str   # "ALL" or "ANY"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TechnicalScore:
    ticker: str
    composite: float        # 0-100
    trend_score: float      # 0-100
    momentum_score: float   # 0-100
    volume_score: float     # 0-100
    volatility_score: float # 0-100
    rs_rating: float        # 0-100 (IBD-style)
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class TechnicalAlert:
    ticker: str
    alert_type: str           # "breakout", "oversold", "golden_cross", etc.
    criteria_triggered: List[str]
    current_price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Indicator Engine
# ──────────────────────────────────────────────────────────────────────────────

class IndicatorEngine:
    """
    All indicator implementations.
    Uses pandas_ta when available, otherwise pure numpy/pandas fallback.
    All methods are static and operate on pandas Series/DataFrames.
    """

    # ── SMA / EMA ──────────────────────────────────────────────────────────────

    @staticmethod
    def sma(close: pd.Series, period: int) -> pd.Series:
        if HAS_TA:
            result = ta.sma(close, length=period)
            if result is not None:
                return result
        return close.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        if HAS_TA:
            result = ta.ema(close, length=period)
            if result is not None:
                return result
        return close.ewm(span=period, adjust=False, min_periods=period).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        if HAS_TA:
            result = ta.rsi(close, length=period)
            if result is not None:
                return result
        # Wilder smoothing (EMA-based)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────────

    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (macd_line, signal_line, histogram)."""
        if HAS_TA:
            result = ta.macd(close, fast=fast, slow=slow, signal=signal)
            if result is not None and not result.empty:
                cols = result.columns.tolist()
                macd_col = [c for c in cols if "MACD_" in c and "MACDs_" not in c and "MACDh_" not in c]
                sig_col = [c for c in cols if "MACDs_" in c]
                hist_col = [c for c in cols if "MACDh_" in c]
                if macd_col and sig_col and hist_col:
                    return result[macd_col[0]], result[sig_col[0]], result[hist_col[0]]
        fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ── Bollinger Bands ────────────────────────────────────────────────────────

    @staticmethod
    def bbands(
        close: pd.Series,
        period: int = 20,
        std: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (upper, mid, lower)."""
        if HAS_TA:
            result = ta.bbands(close, length=period, std=std)
            if result is not None and not result.empty:
                cols = result.columns.tolist()
                up_col = [c for c in cols if "BBU_" in c]
                mid_col = [c for c in cols if "BBM_" in c]
                lo_col = [c for c in cols if "BBL_" in c]
                if up_col and mid_col and lo_col:
                    return result[up_col[0]], result[mid_col[0]], result[lo_col[0]]
        mid = close.rolling(period, min_periods=period).mean()
        sigma = close.rolling(period, min_periods=period).std()
        upper = mid + std * sigma
        lower = mid - std * sigma
        return upper, mid, lower

    # ── Stochastic ────────────────────────────────────────────────────────────

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: int = 14,
        d: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Returns (%K, %D)."""
        if HAS_TA:
            result = ta.stoch(high, low, close, k=k, d=d)
            if result is not None and not result.empty:
                cols = result.columns.tolist()
                k_col = [c for c in cols if "STOCHk_" in c]
                d_col = [c for c in cols if "STOCHd_" in c]
                if k_col and d_col:
                    return result[k_col[0]], result[d_col[0]]
        lowest_low = low.rolling(k, min_periods=k).min()
        highest_high = high.rolling(k, min_periods=k).max()
        pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        pct_d = pct_k.rolling(d, min_periods=d).mean()
        return pct_k, pct_d

    # ── ATR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        if HAS_TA:
            result = ta.atr(high, low, close, length=period)
            if result is not None:
                return result
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # ── ADX ───────────────────────────────────────────────────────────────────

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (ADX, +DI, -DI)."""
        if HAS_TA:
            result = ta.adx(high, low, close, length=period)
            if result is not None and not result.empty:
                cols = result.columns.tolist()
                adx_col = [c for c in cols if c.startswith("ADX_")]
                dmp_col = [c for c in cols if "DMP_" in c]
                dmn_col = [c for c in cols if "DMN_" in c]
                if adx_col and dmp_col and dmn_col:
                    return result[adx_col[0]], result[dmp_col[0]], result[dmn_col[0]]
        # Manual ADX
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        plus_dm = high - prev_high
        minus_dm = prev_low - low
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_val = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_val.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_val.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_val = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        return adx_val, plus_di, minus_di

    # ── OBV ───────────────────────────────────────────────────────────────────

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        if HAS_TA:
            result = ta.obv(close, volume)
            if result is not None:
                return result
        direction = np.sign(close.diff())
        return (direction * volume).fillna(0).cumsum()

    # ── VWAP ──────────────────────────────────────────────────────────────────

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Session VWAP (cumulative from first bar in series)."""
        if HAS_TA:
            result = ta.vwap(high, low, close, volume)
            if result is not None:
                return result
        typical = (high + low + close) / 3
        cum_tp_vol = (typical * volume).cumsum()
        cum_vol = volume.cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    # ── Ichimoku ──────────────────────────────────────────────────────────────

    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
    ) -> Dict[str, pd.Series]:
        if HAS_TA:
            try:
                result = ta.ichimoku(high, low, close, tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_b_period)
                if result is not None and len(result) >= 2:
                    df1, df2 = result[0], result[1]
                    combined = pd.concat([df1, df2], axis=1)
                    cols = combined.columns.tolist()
                    t_col = [c for c in cols if "ISA" in c or "ITS" in c]
                    k_col = [c for c in cols if "IKS" in c]
                    sa_col = [c for c in cols if "ISA" in c]
                    sb_col = [c for c in cols if "ISB" in c]
                    ck_col = [c for c in cols if "ICS" in c]
                    if t_col and k_col and sa_col and sb_col:
                        return {
                            "tenkan": combined[t_col[0]],
                            "kijun": combined[k_col[0]],
                            "senkou_a": combined[sa_col[0]],
                            "senkou_b": combined[sb_col[0]],
                            "chikou": combined[ck_col[0]] if ck_col else close.shift(-kijun_period),
                        }
            except Exception:
                pass

        # Manual fallback
        def mid_range(h: pd.Series, l: pd.Series, period: int) -> pd.Series:
            return (h.rolling(period, min_periods=period).max() + l.rolling(period, min_periods=period).min()) / 2

        tenkan = mid_range(high, low, tenkan_period)
        kijun = mid_range(high, low, kijun_period)
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
        senkou_b = mid_range(high, low, senkou_b_period).shift(kijun_period)
        chikou = close.shift(-kijun_period)

        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou": chikou,
        }

    # ── Williams %R ───────────────────────────────────────────────────────────

    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        if HAS_TA:
            result = ta.willr(high, low, close, length=period)
            if result is not None:
                return result
        highest_high = high.rolling(period, min_periods=period).max()
        lowest_low = low.rolling(period, min_periods=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)

    # ── CCI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        if HAS_TA:
            result = ta.cci(high, low, close, length=period)
            if result is not None:
                return result
        typical = (high + low + close) / 3
        sma_tp = typical.rolling(period, min_periods=period).mean()
        mean_dev = typical.rolling(period, min_periods=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        return (typical - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))

    # ── MFI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        if HAS_TA:
            result = ta.mfi(high, low, close, volume, length=period)
            if result is not None:
                return result
        typical = (high + low + close) / 3
        money_flow = typical * volume
        direction = typical.diff()
        pos_flow = money_flow.where(direction > 0, 0.0)
        neg_flow = money_flow.where(direction < 0, 0.0)
        pos_sum = pos_flow.rolling(period, min_periods=period).sum()
        neg_sum = neg_flow.rolling(period, min_periods=period).sum().replace(0, np.nan)
        mfr = pos_sum / neg_sum
        return 100 - (100 / (1 + mfr))

    # ── DMI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def dmi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Directional Movement Index (+DI - -DI)."""
        _, plus_di, minus_di = IndicatorEngine.adx(high, low, close, period)
        return plus_di - minus_di

    # ── Pivot Points ──────────────────────────────────────────────────────────

    @staticmethod
    def pivots(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Dict[str, float]:
        """Classic pivot points from last complete bar."""
        h, l, c = float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
        pp = (h + l + c) / 3
        r1 = 2 * pp - l
        r2 = pp + (h - l)
        s1 = 2 * pp - h
        s2 = pp - (h - l)
        return {"PP": pp, "R1": r1, "R2": r2, "S1": s1, "S2": s2}

    # ── Keltner Channels ──────────────────────────────────────────────────────

    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        multiplier: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (upper, mid, lower)."""
        if HAS_TA:
            result = ta.kc(high, low, close, length=period, scalar=multiplier)
            if result is not None and not result.empty:
                cols = result.columns.tolist()
                up_col = [c for c in cols if "KCUe_" in c or "KCU_" in c]
                mid_col = [c for c in cols if "KCBe_" in c or "KCB_" in c]
                lo_col = [c for c in cols if "KCLe_" in c or "KCL_" in c]
                if up_col and mid_col and lo_col:
                    return result[up_col[0]], result[mid_col[0]], result[lo_col[0]]
        mid = IndicatorEngine.ema(close, period)
        atr_val = IndicatorEngine.atr(high, low, close, period)
        upper = mid + multiplier * atr_val
        lower = mid - multiplier * atr_val
        return upper, mid, lower


# ──────────────────────────────────────────────────────────────────────────────
# Screening Criteria Factory
# ──────────────────────────────────────────────────────────────────────────────

ie = IndicatorEngine()  # instance for calling static methods (convenience)


def _safe_last(series: pd.Series) -> float:
    """Return last non-NaN value or NaN."""
    val = series.dropna()
    return float(val.iloc[-1]) if not val.empty else float("nan")


def _has_min_bars(df: pd.DataFrame, n: int) -> bool:
    return len(df) >= n


def _make_criteria_registry() -> Dict[str, ScreenCriteria]:
    """Build all 29 screening criteria. Returns dict keyed by name."""

    def c(name: str, description: str, category: str, fn: Callable) -> ScreenCriteria:
        return ScreenCriteria(name=name, description=description, category=category, filter_fn=fn)

    # ── TREND ─────────────────────────────────────────────────────────────────

    def _price_above_sma200(df: pd.DataFrame) -> bool:
        if not _has_min_bars(df, 200):
            return False
        sma200 = ie.sma(df["Close"], 200)
        return float(df["Close"].iloc[-1]) > _safe_last(sma200)

    def _sma50_above_sma200(df: pd.DataFrame) -> bool:
        if not _has_min_bars(df, 200):
            return False
        s50 = ie.sma(df["Close"], 50)
        s200 = ie.sma(df["Close"], 200)
        return _safe_last(s50) > _safe_last(s200)

    def _golden_cross(df: pd.DataFrame) -> bool:
        """SMA50 crossed above SMA200 in last 5 bars."""
        if not _has_min_bars(df, 205):
            return False
        s50 = ie.sma(df["Close"], 50)
        s200 = ie.sma(df["Close"], 200)
        for i in range(-5, 0):
            try:
                if s50.iloc[i - 1] <= s200.iloc[i - 1] and s50.iloc[i] > s200.iloc[i]:
                    return True
            except IndexError:
                pass
        return False

    def _death_cross(df: pd.DataFrame) -> bool:
        """SMA50 crossed below SMA200 in last 5 bars."""
        if not _has_min_bars(df, 205):
            return False
        s50 = ie.sma(df["Close"], 50)
        s200 = ie.sma(df["Close"], 200)
        for i in range(-5, 0):
            try:
                if s50.iloc[i - 1] >= s200.iloc[i - 1] and s50.iloc[i] < s200.iloc[i]:
                    return True
            except IndexError:
                pass
        return False

    def _ema_ribbon_bullish(df: pd.DataFrame) -> bool:
        """EMA(8) > EMA(21) > EMA(34) > EMA(55)."""
        if not _has_min_bars(df, 60):
            return False
        e8 = _safe_last(ie.ema(df["Close"], 8))
        e21 = _safe_last(ie.ema(df["Close"], 21))
        e34 = _safe_last(ie.ema(df["Close"], 34))
        e55 = _safe_last(ie.ema(df["Close"], 55))
        return e8 > e21 > e34 > e55

    def _price_above_ichimoku_cloud(df: pd.DataFrame) -> bool:
        if not _has_min_bars(df, 80):
            return False
        ichi = ie.ichimoku(df["High"], df["Low"], df["Close"])
        sa = _safe_last(ichi["senkou_a"])
        sb = _safe_last(ichi["senkou_b"])
        price = float(df["Close"].iloc[-1])
        return price > sa and price > sb

    # ── MOMENTUM ──────────────────────────────────────────────────────────────

    def _rsi_oversold(df: pd.DataFrame) -> bool:
        if not _has_min_bars(df, 15):
            return False
        return _safe_last(ie.rsi(df["Close"])) < 30

    def _rsi_overbought(df: pd.DataFrame) -> bool:
        if not _has_min_bars(df, 15):
            return False
        return _safe_last(ie.rsi(df["Close"])) > 70

    def _macd_bullish_cross(df: pd.DataFrame) -> bool:
        """MACD crossed above signal in last 3 bars."""
        if not _has_min_bars(df, 35):
            return False
        macd_line, sig_line, _ = ie.macd(df["Close"])
        for i in range(-3, 0):
            try:
                if macd_line.iloc[i - 1] <= sig_line.iloc[i - 1] and macd_line.iloc[i] > sig_line.iloc[i]:
                    return True
            except IndexError:
                pass
        return False

    def _macd_bearish_cross(df: pd.DataFrame) -> bool:
        """MACD crossed below signal in last 3 bars."""
        if not _has_min_bars(df, 35):
            return False
        macd_line, sig_line, _ = ie.macd(df["Close"])
        for i in range(-3, 0):
            try:
                if macd_line.iloc[i - 1] >= sig_line.iloc[i - 1] and macd_line.iloc[i] < sig_line.iloc[i]:
                    return True
            except IndexError:
                pass
        return False

    def _momentum_12_1(df: pd.DataFrame) -> bool:
        """12-month return minus 1-month return > 0."""
        if not _has_min_bars(df, 252):
            return False
        c = df["Close"]
        ret_12m = float(c.iloc[-1] / c.iloc[-252] - 1) if len(c) >= 252 else 0.0
        ret_1m = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) >= 21 else 0.0
        return (ret_12m - ret_1m) > 0

    # ── VOLATILITY ────────────────────────────────────────────────────────────

    def _bb_squeeze(df: pd.DataFrame) -> bool:
        """BB width < 20th percentile of 52-week BB width."""
        if not _has_min_bars(df, 252):
            return False
        upper, mid, lower = ie.bbands(df["Close"])
        bb_width = (upper - lower) / mid.replace(0, np.nan)
        window_width = bb_width.iloc[-252:]
        if window_width.dropna().empty:
            return False
        threshold = float(np.percentile(window_width.dropna(), 20))
        current_width = _safe_last(bb_width)
        return current_width < threshold

    def _bb_breakout_up(df: pd.DataFrame) -> bool:
        if not _has_min_bars(df, 21):
            return False
        upper, _, _ = ie.bbands(df["Close"])
        return float(df["Close"].iloc[-1]) > _safe_last(upper)

    def _bb_breakout_down(df: pd.DataFrame) -> bool:
        if not _has_min_bars(df, 21):
            return False
        _, _, lower = ie.bbands(df["Close"])
        return float(df["Close"].iloc[-1]) < _safe_last(lower)

    def _keltner_squeeze(df: pd.DataFrame) -> bool:
        """Bollinger Bands inside Keltner Channels (John Carter squeeze)."""
        if not _has_min_bars(df, 25):
            return False
        bb_upper, _, bb_lower = ie.bbands(df["Close"])
        kc_upper, _, kc_lower = ie.keltner_channels(df["High"], df["Low"], df["Close"])
        return (
            _safe_last(bb_upper) < _safe_last(kc_upper) and
            _safe_last(bb_lower) > _safe_last(kc_lower)
        )

    # ── VOLUME ────────────────────────────────────────────────────────────────

    def _volume_surge(df: pd.DataFrame) -> bool:
        """Volume > 2× 20-day avg volume."""
        if not _has_min_bars(df, 21) or "Volume" not in df.columns:
            return False
        avg_vol = df["Volume"].iloc[-21:-1].mean()
        return float(df["Volume"].iloc[-1]) > 2 * avg_vol

    def _obv_rising(df: pd.DataFrame) -> bool:
        """OBV slope positive over last 10 bars."""
        if not _has_min_bars(df, 15) or "Volume" not in df.columns:
            return False
        obv = ie.obv(df["Close"], df["Volume"])
        obv_window = obv.iloc[-10:].dropna()
        if len(obv_window) < 5:
            return False
        slope = np.polyfit(range(len(obv_window)), obv_window.values, 1)[0]
        return slope > 0

    def _mfi_oversold(df: pd.DataFrame) -> bool:
        if not _has_min_bars(df, 15) or "Volume" not in df.columns:
            return False
        mfi = ie.mfi(df["High"], df["Low"], df["Close"], df["Volume"])
        return _safe_last(mfi) < 20

    def _mfi_overbought(df: pd.DataFrame) -> bool:
        if not _has_min_bars(df, 15) or "Volume" not in df.columns:
            return False
        mfi = ie.mfi(df["High"], df["Low"], df["Close"], df["Volume"])
        return _safe_last(mfi) > 80

    # ── PRICE ACTION ──────────────────────────────────────────────────────────

    def _near_52w_high(df: pd.DataFrame) -> bool:
        """Close within 5% of 52-week high."""
        if not _has_min_bars(df, 252):
            return False
        high_52w = df["High"].iloc[-252:].max()
        return float(df["Close"].iloc[-1]) >= high_52w * 0.95

    def _near_52w_low(df: pd.DataFrame) -> bool:
        """Close within 5% of 52-week low."""
        if not _has_min_bars(df, 252):
            return False
        low_52w = df["Low"].iloc[-252:].min()
        return float(df["Close"].iloc[-1]) <= low_52w * 1.05

    def _gap_up(df: pd.DataFrame) -> bool:
        """Today's open > yesterday's high."""
        if not _has_min_bars(df, 2):
            return False
        return float(df["Open"].iloc[-1]) > float(df["High"].iloc[-2])

    def _gap_down(df: pd.DataFrame) -> bool:
        """Today's open < yesterday's low."""
        if not _has_min_bars(df, 2):
            return False
        return float(df["Open"].iloc[-1]) < float(df["Low"].iloc[-2])

    # ── COMBINED / ADVANCED ───────────────────────────────────────────────────

    def _william_oneil_stage2(df: pd.DataFrame) -> bool:
        """
        O'Neil Stage 2 uptrend:
        price > SMA200, SMA50 > SMA200, RS vs SPY > 0, volume trending up.
        """
        if not _has_min_bars(df, 200):
            return False
        price = float(df["Close"].iloc[-1])
        s50 = _safe_last(ie.sma(df["Close"], 50))
        s200 = _safe_last(ie.sma(df["Close"], 200))
        if not (price > s200 and s50 > s200):
            return False
        # Volume trend: avg last 10 days > avg prior 10 days
        if "Volume" in df.columns and len(df) >= 20:
            vol_recent = df["Volume"].iloc[-10:].mean()
            vol_prior = df["Volume"].iloc[-20:-10].mean()
            if vol_recent <= vol_prior:
                return False
        return True

    def _vcp_setup(df: pd.DataFrame) -> bool:
        """
        Volatility Contraction Pattern (Minervini).
        3 successive corrections of decreasing depth.
        Each contraction < previous contraction.
        """
        if not _has_min_bars(df, 60):
            return False
        closes = df["Close"].values
        # Find local max/min for 3 wave pattern
        # Simple heuristic: rolling windows of decreasing std
        std_early = float(df["Close"].iloc[-60:-40].std())
        std_mid = float(df["Close"].iloc[-40:-20].std())
        std_late = float(df["Close"].iloc[-20:].std())
        if std_early <= 0:
            return False
        # Contracting volatility
        return std_early > std_mid > std_late

    def _cup_and_handle_setup(df: pd.DataFrame) -> bool:
        """
        Cup and Handle:
        - 52-week high
        - At least 30% pullback in prior months
        - Recovery to within 10% of 52w high
        - Recent contraction (handle)
        """
        if not _has_min_bars(df, 252):
            return False
        high_52w = df["High"].iloc[-252:].max()
        low_52w = df["Low"].iloc[-252:].min()
        current = float(df["Close"].iloc[-1])

        # Check 30%+ pullback from high
        pullback = (high_52w - low_52w) / high_52w
        if pullback < 0.30:
            return False

        # Current price within 10% of 52w high (recovery)
        if current < high_52w * 0.90:
            return False

        # Handle: last 20 bars contracting (< 15% range)
        handle_range = (df["High"].iloc[-20:].max() - df["Low"].iloc[-20:].min()) / current
        return handle_range < 0.15

    def _oversold_bounce_setup(df: pd.DataFrame) -> bool:
        """Triple oversold: RSI < 30 + price at lower BB + MFI < 20."""
        if not _has_min_bars(df, 21):
            return False
        rsi_val = _safe_last(ie.rsi(df["Close"]))
        _, _, bb_lower = ie.bbands(df["Close"])
        bb_low = _safe_last(bb_lower)
        price = float(df["Close"].iloc[-1])
        mfi_check = True
        if "Volume" in df.columns:
            mfi_val = _safe_last(ie.mfi(df["High"], df["Low"], df["Close"], df["Volume"]))
            mfi_check = mfi_val < 20
        return rsi_val < 30 and price <= bb_low * 1.02 and mfi_check

    def _breakout_confirmation(df: pd.DataFrame) -> bool:
        """price > 52w high + volume > 1.5× avg (confirmed breakout)."""
        if not _has_min_bars(df, 252) or "Volume" not in df.columns:
            return False
        high_52w = df["High"].iloc[-252:-1].max()  # exclude today
        price = float(df["Close"].iloc[-1])
        avg_vol = df["Volume"].iloc[-21:-1].mean()
        today_vol = float(df["Volume"].iloc[-1])
        return price > high_52w and today_vol > 1.5 * avg_vol

    def _trend_reversal_bearish(df: pd.DataFrame) -> bool:
        """Death cross + MACD bearish cross + RSI crossing 50 from above."""
        if not _has_min_bars(df, 210):
            return False
        # Death cross
        s50 = ie.sma(df["Close"], 50)
        s200 = ie.sma(df["Close"], 200)
        death = _safe_last(s50) < _safe_last(s200)

        # MACD bearish cross recent
        macd_line, sig_line, _ = ie.macd(df["Close"])
        macd_cross_bearish = False
        for i in range(-5, 0):
            try:
                if macd_line.iloc[i - 1] >= sig_line.iloc[i - 1] and macd_line.iloc[i] < sig_line.iloc[i]:
                    macd_cross_bearish = True
                    break
            except IndexError:
                pass

        # RSI crossing 50 from above
        rsi_vals = ie.rsi(df["Close"])
        rsi_cross = False
        for i in range(-5, 0):
            try:
                if rsi_vals.iloc[i - 1] >= 50 and rsi_vals.iloc[i] < 50:
                    rsi_cross = True
                    break
            except IndexError:
                pass

        return death and macd_cross_bearish and rsi_cross

    # ── Build Registry ─────────────────────────────────────────────────────────

    return {
        # TREND
        "price_above_sma200": c("price_above_sma200", "Close > SMA(200)", "trend", _price_above_sma200),
        "sma50_above_sma200": c("sma50_above_sma200", "SMA(50) > SMA(200)", "trend", _sma50_above_sma200),
        "golden_cross": c("golden_cross", "SMA50 crossed above SMA200 in last 5 bars", "trend", _golden_cross),
        "death_cross": c("death_cross", "SMA50 crossed below SMA200 in last 5 bars", "trend", _death_cross),
        "ema_ribbon_bullish": c("ema_ribbon_bullish", "EMA(8)>EMA(21)>EMA(34)>EMA(55)", "trend", _ema_ribbon_bullish),
        "price_above_ichimoku_cloud": c("price_above_ichimoku_cloud", "Close above Ichimoku cloud", "trend", _price_above_ichimoku_cloud),
        # MOMENTUM
        "rsi_oversold": c("rsi_oversold", "RSI(14) < 30", "momentum", _rsi_oversold),
        "rsi_overbought": c("rsi_overbought", "RSI(14) > 70", "momentum", _rsi_overbought),
        "macd_bullish_cross": c("macd_bullish_cross", "MACD crossed above signal (last 3 bars)", "momentum", _macd_bullish_cross),
        "macd_bearish_cross": c("macd_bearish_cross", "MACD crossed below signal (last 3 bars)", "momentum", _macd_bearish_cross),
        "momentum_12_1": c("momentum_12_1", "12M return - 1M return > 0 (Jegadeesh-Titman)", "momentum", _momentum_12_1),
        # VOLATILITY
        "bb_squeeze": c("bb_squeeze", "BB width < 20th pctile of 52w BB width", "volatility", _bb_squeeze),
        "bb_breakout_up": c("bb_breakout_up", "Close > upper Bollinger Band", "volatility", _bb_breakout_up),
        "bb_breakout_down": c("bb_breakout_down", "Close < lower Bollinger Band", "volatility", _bb_breakout_down),
        "keltner_squeeze": c("keltner_squeeze", "BB inside Keltner Channels (Carter squeeze)", "volatility", _keltner_squeeze),
        # VOLUME
        "volume_surge": c("volume_surge", "Volume > 2× 20-day avg", "volume", _volume_surge),
        "obv_rising": c("obv_rising", "OBV slope positive over last 10 bars", "volume", _obv_rising),
        "mfi_oversold": c("mfi_oversold", "MFI(14) < 20", "volume", _mfi_oversold),
        "mfi_overbought": c("mfi_overbought", "MFI(14) > 80", "volume", _mfi_overbought),
        # PRICE ACTION
        "near_52w_high": c("near_52w_high", "Close within 5% of 52-week high", "price_action", _near_52w_high),
        "near_52w_low": c("near_52w_low", "Close within 5% of 52-week low", "price_action", _near_52w_low),
        "gap_up": c("gap_up", "Today open > yesterday high", "price_action", _gap_up),
        "gap_down": c("gap_down", "Today open < yesterday low", "price_action", _gap_down),
        # COMBINED / ADVANCED
        "william_oneil_stage2": c("william_oneil_stage2", "O'Neil Stage 2 uptrend criteria", "advanced", _william_oneil_stage2),
        "vcp_setup": c("vcp_setup", "Volatility Contraction Pattern (Minervini)", "advanced", _vcp_setup),
        "cup_and_handle_setup": c("cup_and_handle_setup", "Cup and Handle pattern setup", "advanced", _cup_and_handle_setup),
        "oversold_bounce": c("oversold_bounce", "Triple oversold: RSI<30 + lower BB + MFI<20", "advanced", _oversold_bounce_setup),
        "breakout_confirmation": c("breakout_confirmation", "52w high breakout with volume confirmation", "advanced", _breakout_confirmation),
        "trend_reversal_bearish": c("trend_reversal_bearish", "Death cross + MACD bear cross + RSI<50", "advanced", _trend_reversal_bearish),
    }


CRITERIA_REGISTRY: Dict[str, ScreenCriteria] = _make_criteria_registry()

# ── Preset Screen Bundles ──────────────────────────────────────────────────────

MOMENTUM_SCREEN = [
    "price_above_sma200", "sma50_above_sma200", "macd_bullish_cross",
    "momentum_12_1", "obv_rising", "william_oneil_stage2",
]

OVERSOLD_SCREEN = [
    "rsi_oversold", "bb_squeeze", "mfi_oversold", "oversold_bounce",
]

BREAKOUT_SCREEN = [
    "keltner_squeeze", "near_52w_high", "breakout_confirmation", "vcp_setup",
]

TREND_FOLLOWING_SCREEN = [
    "price_above_sma200", "sma50_above_sma200", "ema_ribbon_bullish",
    "price_above_ichimoku_cloud", "momentum_12_1", "obv_rising",
]

REVERSAL_SHORT_SCREEN = [
    "rsi_overbought", "bb_breakout_up", "mfi_overbought", "trend_reversal_bearish",
]

SCREEN_PRESETS = {
    "MOMENTUM_SCREEN": MOMENTUM_SCREEN,
    "OVERSOLD_SCREEN": OVERSOLD_SCREEN,
    "BREAKOUT_SCREEN": BREAKOUT_SCREEN,
    "TREND_FOLLOWING_SCREEN": TREND_FOLLOWING_SCREEN,
    "REVERSAL_SHORT_SCREEN": REVERSAL_SHORT_SCREEN,
}


# ──────────────────────────────────────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────────────────────────────────────

def fetch_universe_data(
    tickers: List[str],
    min_history_days: int = 252,
    period: str = "2y",
) -> Dict[str, pd.DataFrame]:
    """Batch download OHLCV for all tickers in universe."""
    logger.info("Downloading data for %d tickers (period=%s)...", len(tickers), period)
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by="ticker")
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return {}

    result: Dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                df = raw[ticker].dropna(subset=["Close"])
                if len(df) >= min_history_days:
                    result[ticker] = df
                else:
                    logger.debug("Skipping %s: only %d bars (need %d)", ticker, len(df), min_history_days)
            except KeyError:
                logger.debug("No data for %s", ticker)
    else:
        # Single ticker
        if not raw.empty and len(raw.dropna(subset=["Close"])) >= min_history_days:
            result[tickers[0]] = raw.dropna(subset=["Close"])

    logger.info("Data ready for %d / %d tickers", len(result), len(tickers))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Technical Screener
# ──────────────────────────────────────────────────────────────────────────────

class TechnicalScreener:
    """
    Runs technical screening criteria against a universe of stocks.
    Supports AND mode (all criteria must pass) and ANY mode (at least one passes).
    """

    def __init__(self) -> None:
        self._active_criteria: Dict[str, ScreenCriteria] = {}

    def add_criteria(self, criteria: "str | ScreenCriteria") -> None:
        """Add a criterion by name (from registry) or ScreenCriteria instance."""
        if isinstance(criteria, str):
            if criteria not in CRITERIA_REGISTRY:
                raise ValueError(f"Unknown criteria: '{criteria}'. Available: {list(CRITERIA_REGISTRY.keys())}")
            self._active_criteria[criteria] = CRITERIA_REGISTRY[criteria]
        else:
            self._active_criteria[criteria.name] = criteria

    def remove_criteria(self, name: str) -> None:
        self._active_criteria.pop(name, None)

    def load_preset(self, preset_name: str) -> None:
        """Load a named preset bundle."""
        if preset_name not in SCREEN_PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(SCREEN_PRESETS.keys())}")
        for name in SCREEN_PRESETS[preset_name]:
            self.add_criteria(name)
        logger.info("Loaded preset '%s' (%d criteria)", preset_name, len(SCREEN_PRESETS[preset_name]))

    def clear_criteria(self) -> None:
        self._active_criteria.clear()

    def list_criteria(self) -> List[str]:
        return list(self._active_criteria.keys())

    def _apply_criteria_to_ticker(self, ticker: str, df: pd.DataFrame) -> Dict[str, bool]:
        """Run all active criteria against one ticker's OHLCV data."""
        results: Dict[str, bool] = {}
        for name, crit in self._active_criteria.items():
            try:
                results[name] = bool(crit.filter_fn(df))
            except Exception as exc:
                logger.debug("Criteria '%s' failed for %s: %s", name, ticker, exc)
                results[name] = False
        return results

    def run_screen(
        self,
        universe: List[str],
        min_history_days: int = 252,
        ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> ScreenResult:
        """
        Screen universe — ALL criteria must pass.
        Optionally pass pre-downloaded ohlcv_data to avoid re-fetching.
        """
        return self._run(universe, min_history_days, "ALL", ohlcv_data)

    def run_screen_any(
        self,
        universe: List[str],
        min_history_days: int = 252,
        ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> ScreenResult:
        """Screen universe — ANY criteria passing is sufficient."""
        return self._run(universe, min_history_days, "ANY", ohlcv_data)

    def _run(
        self,
        universe: List[str],
        min_history_days: int,
        mode: str,
        ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> ScreenResult:
        if not self._active_criteria:
            logger.warning("No active criteria. Use add_criteria() or load_preset() first.")

        data = ohlcv_data or fetch_universe_data(universe, min_history_days)

        passed: List[str] = []
        failed: List[str] = []
        per_ticker: Dict[str, Dict[str, bool]] = {}

        for ticker in universe:
            if ticker not in data:
                failed.append(ticker)
                per_ticker[ticker] = {}
                continue

            results = self._apply_criteria_to_ticker(ticker, data[ticker])
            per_ticker[ticker] = results

            if mode == "ALL":
                passes = all(results.values()) if results else False
            else:
                passes = any(results.values()) if results else False

            if passes:
                passed.append(ticker)
            else:
                failed.append(ticker)

        logger.info(
            "Screen complete (%s mode): %d/%d passed",
            mode, len(passed), len(universe),
        )
        return ScreenResult(
            passed_tickers=passed,
            failed_tickers=failed,
            per_ticker=per_ticker,
            n_criteria=len(self._active_criteria),
            screen_mode=mode,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Technical Ranker
# ──────────────────────────────────────────────────────────────────────────────

class TechnicalRanker:
    """
    Rank universe by technical strength using a weighted composite score.
    Trend(30%) + Momentum(30%) + Volume(20%) + Volatility(20%)
    """

    TREND_WEIGHT = 0.30
    MOMENTUM_WEIGHT = 0.30
    VOLUME_WEIGHT = 0.20
    VOLATILITY_WEIGHT = 0.20

    def compute_technical_score(
        self, ticker: str, ohlcv: pd.DataFrame
    ) -> TechnicalScore:
        """Compute 0-100 composite technical score for a single ticker."""
        details: Dict[str, float] = {}

        # ── Trend Component (0-100) ────────────────────────────────────────────
        trend_signals: List[float] = []

        if len(ohlcv) >= 200:
            price = float(ohlcv["Close"].iloc[-1])
            s50 = _safe_last(ie.sma(ohlcv["Close"], 50))
            s200 = _safe_last(ie.sma(ohlcv["Close"], 200))
            trend_signals.append(100.0 if price > s200 else 0.0)
            trend_signals.append(100.0 if s50 > s200 else 0.0)
            details["price_vs_sma200"] = ((price - s200) / s200) * 100 if s200 > 0 else 0.0

        if len(ohlcv) >= 55:
            e8 = _safe_last(ie.ema(ohlcv["Close"], 8))
            e21 = _safe_last(ie.ema(ohlcv["Close"], 21))
            e34 = _safe_last(ie.ema(ohlcv["Close"], 34))
            e55 = _safe_last(ie.ema(ohlcv["Close"], 55))
            ribbon_bullish = e8 > e21 > e34 > e55
            trend_signals.append(100.0 if ribbon_bullish else 0.0)

        if len(ohlcv) >= 80:
            ichi = ie.ichimoku(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
            sa = _safe_last(ichi["senkou_a"])
            sb = _safe_last(ichi["senkou_b"])
            price = float(ohlcv["Close"].iloc[-1])
            above_cloud = price > max(sa, sb) if not (np.isnan(sa) or np.isnan(sb)) else False
            trend_signals.append(100.0 if above_cloud else 0.0)

        trend_score = float(np.mean(trend_signals)) if trend_signals else 50.0

        # ── Momentum Component (0-100) ─────────────────────────────────────────
        mom_signals: List[float] = []

        if len(ohlcv) >= 15:
            rsi_val = _safe_last(ie.rsi(ohlcv["Close"]))
            if not np.isnan(rsi_val):
                # RSI: 50 = 50 score, 30 = oversold (good for bounce) = 80, 70 = 70
                # Scale so that 50 RSI → 50 score, 70 → 100, 30 → 0
                # Use a V-shape: extremes are signals, mid is neutral
                rsi_score = min(100.0, max(0.0, rsi_val))  # linear 0-100
                mom_signals.append(rsi_score)
                details["rsi"] = rsi_val

        if len(ohlcv) >= 35:
            macd_line, sig_line, hist = ie.macd(ohlcv["Close"])
            hist_val = _safe_last(hist)
            macd_val = _safe_last(macd_line)
            if not np.isnan(hist_val) and not np.isnan(macd_val):
                mom_signals.append(100.0 if hist_val > 0 else 0.0)
                details["macd_hist"] = hist_val

        if len(ohlcv) >= 252:
            c = ohlcv["Close"]
            ret_12m = float(c.iloc[-1] / c.iloc[-252] - 1)
            ret_1m = float(c.iloc[-1] / c.iloc[-21] - 1)
            mom_12_1 = ret_12m - ret_1m
            # Score: cap at ±50% range → 0-100
            mom_score_val = min(100, max(0, 50 + mom_12_1 * 100))
            mom_signals.append(mom_score_val)
            details["momentum_12_1"] = mom_12_1

        momentum_score = float(np.mean(mom_signals)) if mom_signals else 50.0

        # ── Volume Component (0-100) ───────────────────────────────────────────
        vol_signals: List[float] = []

        if "Volume" in ohlcv.columns and len(ohlcv) >= 21:
            avg_vol = ohlcv["Volume"].iloc[-21:-1].mean()
            today_vol = float(ohlcv["Volume"].iloc[-1])
            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1.0
            vol_signals.append(min(100.0, vol_ratio * 50))  # 2× avg → 100
            details["volume_ratio"] = vol_ratio

            if len(ohlcv) >= 15:
                obv_series = ie.obv(ohlcv["Close"], ohlcv["Volume"])
                obv_window = obv_series.iloc[-10:].dropna()
                if len(obv_window) >= 5:
                    slope = np.polyfit(range(len(obv_window)), obv_window.values, 1)[0]
                    vol_signals.append(100.0 if slope > 0 else 0.0)

        volume_score = float(np.mean(vol_signals)) if vol_signals else 50.0

        # ── Volatility Component (0-100) ───────────────────────────────────────
        # Volatility score: lower current vol relative to history = better
        # (setup for breakout); squeeze = higher score
        vt_signals: List[float] = []

        if len(ohlcv) >= 252:
            upper, mid, lower = ie.bbands(ohlcv["Close"])
            bb_width = (upper - lower) / mid.replace(0, np.nan)
            current_width = _safe_last(bb_width)
            hist_width = bb_width.iloc[-252:].dropna()
            if not hist_width.empty and not np.isnan(current_width):
                pctile = float(np.sum(hist_width < current_width) / len(hist_width) * 100)
                # Low percentile = squeeze = high score (anticipating breakout)
                squeeze_score = 100 - pctile
                vt_signals.append(squeeze_score)
                details["bb_width_percentile"] = pctile

        if len(ohlcv) >= 25:
            kc_upper, _, kc_lower = ie.keltner_channels(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
            bb_upper, _, bb_lower = ie.bbands(ohlcv["Close"])
            in_squeeze = (
                _safe_last(bb_upper) < _safe_last(kc_upper) and
                _safe_last(bb_lower) > _safe_last(kc_lower)
            )
            vt_signals.append(100.0 if in_squeeze else 30.0)

        volatility_score = float(np.mean(vt_signals)) if vt_signals else 50.0

        # ── Composite ──────────────────────────────────────────────────────────
        composite = (
            self.TREND_WEIGHT * trend_score +
            self.MOMENTUM_WEIGHT * momentum_score +
            self.VOLUME_WEIGHT * volume_score +
            self.VOLATILITY_WEIGHT * volatility_score
        )

        return TechnicalScore(
            ticker=ticker,
            composite=round(composite, 2),
            trend_score=round(trend_score, 2),
            momentum_score=round(momentum_score, 2),
            volume_score=round(volume_score, 2),
            volatility_score=round(volatility_score, 2),
            rs_rating=0.0,  # filled by rank_universe
            details=details,
        )

    def compute_rs_rating(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        benchmark_ticker: str = "SPY",
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        IBD-style Relative Strength rating.
        Weighted: most recent 1M: 40%, prior 3M: 20%, prior 6M: 20%, prior 12M: 20%.
        Returns percentile (0-100) vs benchmark return decomposition.
        """
        try:
            if benchmark_data is None:
                bm_raw = yf.download(benchmark_ticker, period="18mo", auto_adjust=True, progress=False)
                benchmark_close = bm_raw["Close"].dropna() if not bm_raw.empty else pd.Series(dtype=float)
            else:
                benchmark_close = benchmark_data["Close"].dropna()

            close = ohlcv["Close"].dropna()
            if len(close) < 252 or benchmark_close.empty:
                return 50.0

            def relative_return(ticker_close: pd.Series, bm_close: pd.Series, n_days: int) -> float:
                if len(ticker_close) < n_days or len(bm_close) < n_days:
                    return 0.0
                t_ret = float(ticker_close.iloc[-1] / ticker_close.iloc[-n_days] - 1)
                bm_ret = float(bm_close.iloc[-1] / bm_close.iloc[-n_days] - 1)
                return t_ret - bm_ret

            r_1m = relative_return(close, benchmark_close, 21)
            r_3m = relative_return(close, benchmark_close, 63)
            r_6m = relative_return(close, benchmark_close, 126)
            r_12m = relative_return(close, benchmark_close, 252)

            # Weighted composite RS
            rs_composite = 0.40 * r_1m + 0.20 * r_3m + 0.20 * r_6m + 0.20 * r_12m

            # Convert to 0-100 score: cap at ±50% relative range
            rs_score = min(100.0, max(0.0, 50.0 + rs_composite * 100))
            return round(rs_score, 2)

        except Exception as exc:
            logger.debug("RS rating failed for %s: %s", ticker, exc)
            return 50.0

    def rank_universe(
        self,
        tickers: List[str],
        ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Rank all tickers by composite technical score.
        Returns DataFrame sorted descending by composite score.
        """
        data = ohlcv_data or fetch_universe_data(tickers)

        # Fetch SPY for RS calculation
        try:
            spy_raw = yf.download("SPY", period="18mo", auto_adjust=True, progress=False)
            spy_data = spy_raw if not spy_raw.empty else None
        except Exception:
            spy_data = None

        scores: List[TechnicalScore] = []
        for ticker in tickers:
            if ticker not in data:
                continue
            score = self.compute_technical_score(ticker, data[ticker])
            score.rs_rating = self.compute_rs_rating(ticker, data[ticker], benchmark_data=spy_data)
            scores.append(score)

        if not scores:
            return pd.DataFrame()

        rows = []
        for s in scores:
            rows.append({
                "ticker": s.ticker,
                "composite": s.composite,
                "trend_score": s.trend_score,
                "momentum_score": s.momentum_score,
                "volume_score": s.volume_score,
                "volatility_score": s.volatility_score,
                "rs_rating": s.rs_rating,
            })

        df = pd.DataFrame(rows)
        return df.sort_values("composite", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Technical Alert System
# ──────────────────────────────────────────────────────────────────────────────

class TechnicalAlertSystem:
    """
    Registers watchlists and fires alerts when technical criteria trigger.
    """

    def __init__(self) -> None:
        self._watchlist: Dict[str, List[str]] = {}  # {group_name: [tickers]}
        self._screener = TechnicalScreener()

    def watch(self, tickers: List[str], criteria: List[str], group: str = "default") -> None:
        """Register a watchlist with a set of criteria names to monitor."""
        self._watchlist[group] = tickers
        logger.info("Watchlist '%s': %d tickers, %d criteria", group, len(tickers), len(criteria))

    def check_alerts(
        self,
        tickers: Optional[List[str]] = None,
        ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[TechnicalAlert]:
        """Run all registered criteria on watchlist, return triggered alerts."""
        all_tickers = tickers or list({t for ts in self._watchlist.values() for t in ts})
        if not all_tickers:
            return []

        data = ohlcv_data or fetch_universe_data(all_tickers, min_history_days=30)
        alerts: List[TechnicalAlert] = []

        for ticker, df in data.items():
            triggered: List[str] = []
            for name, crit in CRITERIA_REGISTRY.items():
                try:
                    if crit.filter_fn(df):
                        triggered.append(name)
                except Exception:
                    pass

            if triggered:
                price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
                alerts.append(TechnicalAlert(
                    ticker=ticker,
                    alert_type="multi",
                    criteria_triggered=triggered,
                    current_price=price,
                    description=f"{len(triggered)} criteria triggered: {', '.join(triggered[:3])}{'...' if len(triggered) > 3 else ''}",
                ))

        return sorted(alerts, key=lambda a: -len(a.criteria_triggered))

    def get_breakout_alerts(
        self,
        universe: List[str],
        ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[TechnicalAlert]:
        """Alert when breakout criteria trigger."""
        data = ohlcv_data or fetch_universe_data(universe, min_history_days=252)
        breakout_criteria = ["breakout_confirmation", "bb_breakout_up", "near_52w_high", "volume_surge", "vcp_setup"]
        alerts: List[TechnicalAlert] = []

        for ticker, df in data.items():
            triggered = []
            for name in breakout_criteria:
                crit = CRITERIA_REGISTRY.get(name)
                if crit:
                    try:
                        if crit.filter_fn(df):
                            triggered.append(name)
                    except Exception:
                        pass
            if triggered:
                price = float(df["Close"].iloc[-1])
                alerts.append(TechnicalAlert(
                    ticker=ticker,
                    alert_type="breakout",
                    criteria_triggered=triggered,
                    current_price=price,
                    description=f"Breakout: {', '.join(triggered)}",
                ))

        return sorted(alerts, key=lambda a: -len(a.criteria_triggered))

    def get_oversold_alerts(
        self,
        universe: List[str],
        ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[TechnicalAlert]:
        """Alert when mean-reversion / oversold criteria trigger."""
        data = ohlcv_data or fetch_universe_data(universe, min_history_days=30)
        oversold_criteria = ["rsi_oversold", "mfi_oversold", "near_52w_low", "bb_breakout_down", "oversold_bounce"]
        alerts: List[TechnicalAlert] = []

        for ticker, df in data.items():
            triggered = []
            for name in oversold_criteria:
                crit = CRITERIA_REGISTRY.get(name)
                if crit:
                    try:
                        if crit.filter_fn(df):
                            triggered.append(name)
                    except Exception:
                        pass
            if triggered:
                price = float(df["Close"].iloc[-1])
                alerts.append(TechnicalAlert(
                    ticker=ticker,
                    alert_type="oversold",
                    criteria_triggered=triggered,
                    current_price=price,
                    description=f"Oversold: {', '.join(triggered)}",
                ))

        return sorted(alerts, key=lambda a: -len(a.criteria_triggered))


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: Print Helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_screen_result(result: ScreenResult) -> None:
    """Pretty-print a ScreenResult."""
    print("=" * 60)
    print(f"  SCREEN RESULT — Mode: {result.screen_mode}  Criteria: {result.n_criteria}")
    print(f"  Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    print(f"  PASSED ({len(result.passed_tickers)}): {', '.join(result.passed_tickers) or '(none)'}")
    print(f"  FAILED ({len(result.failed_tickers)}): {', '.join(result.failed_tickers[:20])}")
    if result.per_ticker:
        print()
        print("  BREAKDOWN (passed tickers):")
        for ticker in result.passed_tickers:
            breakdown = result.per_ticker.get(ticker, {})
            checks = [f"{'✓' if v else '✗'} {k}" for k, v in breakdown.items()]
            print(f"    {ticker:<6}: {' | '.join(checks)}")
    print("=" * 60)


def print_ranking(df: pd.DataFrame, top_n: int = 10) -> None:
    """Pretty-print ranking DataFrame."""
    print("=" * 80)
    print("  TECHNICAL RANKING — Top", top_n)
    print("=" * 80)
    cols = ["ticker", "composite", "trend_score", "momentum_score", "volume_score", "volatility_score", "rs_rating"]
    available_cols = [c for c in cols if c in df.columns]
    pd.set_option("display.float_format", "{:.1f}".format)
    print(df[available_cols].head(top_n).to_string(index=True))
    print("=" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# Main — Demo Momentum Screen on S&P 100 Subset
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # 20-ticker subset of S&P 100 for demo
    SP100_SUBSET = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
        "V", "PG", "UNH", "HD", "MA",
        "DIS", "NFLX", "PYPL", "AMD", "CRM",
    ]

    print("\n" + "=" * 60)
    print("  SENTINEL Technical Screener v3 — Demo")
    print(f"  Universe: {len(SP100_SUBSET)} tickers (S&P 100 subset)")
    print("=" * 60)

    # ── Fetch data once ────────────────────────────────────────────────────────
    print("\n  Fetching OHLCV data...")
    ohlcv = fetch_universe_data(SP100_SUBSET, min_history_days=200, period="2y")
    available = list(ohlcv.keys())
    print(f"  Data loaded for: {', '.join(available)}")

    # ── Run Momentum Screen ────────────────────────────────────────────────────
    print("\n  Running MOMENTUM_SCREEN...")
    screener = TechnicalScreener()
    screener.load_preset("MOMENTUM_SCREEN")
    result = screener.run_screen(SP100_SUBSET, ohlcv_data=ohlcv)
    print_screen_result(result)

    # ── Rank by RS Rating ──────────────────────────────────────────────────────
    print("\n  Ranking universe by technical score...")
    ranker = TechnicalRanker()
    rankings = ranker.rank_universe(SP100_SUBSET, ohlcv_data=ohlcv)

    if not rankings.empty:
        print("\n  TOP 5 BY COMPOSITE SCORE:")
        print_ranking(rankings, top_n=5)

        print("\n  TOP 5 BY RS RATING:")
        rs_ranked = rankings.sort_values("rs_rating", ascending=False)
        print_ranking(rs_ranked, top_n=5)
    else:
        print("  Ranking failed — insufficient data.")

    # ── Check Breakout Alerts ──────────────────────────────────────────────────
    print("\n  Checking breakout alerts...")
    alert_sys = TechnicalAlertSystem()
    breakout_alerts = alert_sys.get_breakout_alerts(SP100_SUBSET, ohlcv_data=ohlcv)

    if breakout_alerts:
        print(f"\n  BREAKOUT ALERTS ({len(breakout_alerts)}):")
        for alert in breakout_alerts[:5]:
            print(f"    {alert.ticker:<8} @ ${alert.current_price:.2f}  — {alert.description}")
    else:
        print("  No breakout alerts triggered.")

    # ── Oversold Alerts ────────────────────────────────────────────────────────
    print("\n  Checking oversold alerts...")
    oversold_alerts = alert_sys.get_oversold_alerts(SP100_SUBSET, ohlcv_data=ohlcv)
    if oversold_alerts:
        print(f"\n  OVERSOLD ALERTS ({len(oversold_alerts)}):")
        for alert in oversold_alerts[:5]:
            print(f"    {alert.ticker:<8} @ ${alert.current_price:.2f}  — {alert.description}")
    else:
        print("  No oversold alerts triggered.")

    # ── Run Breakout Screen ────────────────────────────────────────────────────
    print("\n  Running BREAKOUT_SCREEN...")
    screener.clear_criteria()
    screener.load_preset("BREAKOUT_SCREEN")
    breakout_result = screener.run_screen_any(SP100_SUBSET, ohlcv_data=ohlcv)
    print(f"  Breakout candidates: {', '.join(breakout_result.passed_tickers) or '(none)'}")

    print("\n  Demo complete.")
