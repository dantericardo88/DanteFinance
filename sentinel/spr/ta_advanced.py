"""
Advanced technical analysis signals — Dimensions 73-76 of the SENTINEL competitive matrix.

Ichimoku Cloud, Fibonacci Retracement, ADX/DI, Parabolic SAR, and a composite score.
All computation is offloaded via asyncio.to_thread; yfinance and pandas are sync.
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

try:
    import yfinance as yf  # type: ignore
    _YF = True
except ImportError:
    _YF = False

_MIN_BARS = 60


# ── Models ─────────────────────────────────────────────────────────────────────

class IchimokuCloud(BaseModel):
    tenkan_sen: float         # conversion line (9-period)
    kijun_sen: float          # base line (26-period)
    senkou_span_a: float      # leading span A
    senkou_span_b: float      # leading span B (52-period)
    chikou_span: float        # lagging span (price 26 periods back)
    signal: str               # "bullish" | "bearish" | "neutral"
    price_vs_cloud: str       # "above" | "below" | "inside"


class FibonacciLevels(BaseModel):
    swing_high: float
    swing_low: float
    levels: dict[str, float]  # {"0%": x, "23.6%": y, ...}
    nearest_support: float
    nearest_resistance: float


class ADXSignal(BaseModel):
    adx: float                # 0-100, trend strength (>25 = trending)
    plus_di: float            # +DI
    minus_di: float           # -DI
    signal: str               # "strong_uptrend" | "strong_downtrend" | "weak_trend" | "ranging"


class ParabolicSAR(BaseModel):
    sar: float                # current SAR value
    trend: str                # "bullish" | "bearish"
    acceleration: float       # current AF
    reversal_price: float     # price at which SAR would reverse


class AdvancedTAResult(BaseModel):
    ticker: str
    as_of: str
    price: float
    ichimoku: IchimokuCloud
    fibonacci: FibonacciLevels
    adx: ADXSignal
    parabolic_sar: ParabolicSAR
    composite_score: float     # -1.0 to 1.0
    composite_label: str       # "Strong Buy" | "Buy" | "Neutral" | "Sell" | "Strong Sell"
    warnings: list[str]


# ── Data fetch ──────────────────────────────────────────────────────────────────

def _fetch_ohlcv_sync(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not _YF:
        logger.warning("yfinance_not_installed", ticker=ticker)
        return pd.DataFrame()
    logger.info("fetching_ohlcv_advanced", ticker=ticker, period=period, interval=interval)
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False, threads=False)
    except Exception as exc:
        logger.error("yfinance_download_failed", ticker=ticker, error=str(exc))
        return pd.DataFrame()
    if raw.empty:
        logger.warning("yfinance_returned_empty", ticker=ticker)
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).strip().title() for c in raw.columns]
    required = {"High", "Low", "Close"}
    if not required.issubset(raw.columns):
        logger.warning("ohlcv_columns_missing", ticker=ticker, columns=list(raw.columns))
        return pd.DataFrame()
    df = raw[list(required)].copy().sort_index().dropna(subset=["Close"])
    logger.info("ohlcv_fetched_advanced", rows=len(df), ticker=ticker)
    return df


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _last(s: pd.Series) -> float:  return float(s.iloc[-1])
def _nan(v: float)       -> bool:  return math.isnan(v)
def _r(v: float, n: int = 4) -> float: return round(float(v), n)


# ── Ichimoku Cloud ──────────────────────────────────────────────────────────────

def _ichimoku(hi: pd.Series, lo: pd.Series, cl: pd.Series) -> IchimokuCloud:
    def mid(n: int) -> pd.Series:
        return (hi.rolling(n, min_periods=n).max() + lo.rolling(n, min_periods=n).min()) / 2.0

    tenkan = mid(9)
    kijun  = mid(26)
    span_a = (tenkan + kijun) / 2.0
    span_b = mid(52)

    tenkan_val = _last(tenkan)
    kijun_val  = _last(kijun)

    # Senkou spans project 26 bars forward on a live chart; to find what overlaps
    # the current bar we read 26 bars back in the unshifted series.
    if len(span_a) > 26:
        span_a_val = float(span_a.iloc[-27])
        span_b_val = float(span_b.iloc[-27])
    else:
        span_a_val = float(span_a.dropna().iloc[0]) if not span_a.dropna().empty else float("nan")
        span_b_val = float(span_b.dropna().iloc[0]) if not span_b.dropna().empty else float("nan")

    chikou_val = float(cl.iloc[-27]) if len(cl) > 26 else float(cl.iloc[0])
    price      = _last(cl)

    cloud_top    = max(span_a_val, span_b_val)
    cloud_bottom = min(span_a_val, span_b_val)

    if price > cloud_top:       price_vs_cloud = "above"
    elif price < cloud_bottom:  price_vs_cloud = "below"
    else:                       price_vs_cloud = "inside"

    valid = not _nan(span_a_val) and not _nan(span_b_val)
    bullish = valid and price > cloud_top    and span_a_val > span_b_val and price > tenkan_val and price > kijun_val
    bearish = valid and price < cloud_bottom and span_b_val > span_a_val and price < tenkan_val and price < kijun_val
    signal  = "bullish" if bullish else ("bearish" if bearish else "neutral")

    return IchimokuCloud(
        tenkan_sen=_r(tenkan_val), kijun_sen=_r(kijun_val),
        senkou_span_a=_r(span_a_val), senkou_span_b=_r(span_b_val),
        chikou_span=_r(chikou_val), signal=signal, price_vs_cloud=price_vs_cloud,
    )


# ── Fibonacci Retracement ───────────────────────────────────────────────────────

_FIB_RATIOS: list[tuple[str, float]] = [
    ("0%", 0.0), ("23.6%", 0.236), ("38.2%", 0.382), ("50%", 0.5),
    ("61.8%", 0.618), ("78.6%", 0.786), ("100%", 1.0),
]

def _fibonacci(cl: pd.Series) -> FibonacciLevels:
    window     = cl.iloc[-90:] if len(cl) >= 90 else cl
    swing_high = float(window.max())
    swing_low  = float(window.min())
    rng        = swing_high - swing_low
    levels     = {lbl: _r(swing_high - rng * ratio) for lbl, ratio in _FIB_RATIOS}
    price      = _last(cl)
    vals       = sorted(levels.values())
    support    = max((v for v in vals if v < price), default=swing_low)
    resistance = min((v for v in vals if v > price), default=swing_high)
    return FibonacciLevels(
        swing_high=_r(swing_high), swing_low=_r(swing_low), levels=levels,
        nearest_support=_r(support), nearest_resistance=_r(resistance),
    )


# ── ADX / Directional Movement ──────────────────────────────────────────────────

def _adx(hi: pd.Series, lo: pd.Series, cl: pd.Series, period: int = 14) -> ADXSignal:
    alpha  = 1.0 / period
    hi_a   = hi.values.astype(float)
    lo_a   = lo.values.astype(float)
    cl_a   = cl.values.astype(float)
    n      = len(hi_a)
    p_dm   = np.zeros(n)
    m_dm   = np.zeros(n)
    tr_arr = np.zeros(n)

    for i in range(1, n):
        up, dn    = hi_a[i] - hi_a[i-1], lo_a[i-1] - lo_a[i]
        p_dm[i]   = up if (up > dn  and up > 0)  else 0.0
        m_dm[i]   = dn if (dn > up  and dn > 0)  else 0.0
        tr_arr[i] = max(hi_a[i] - lo_a[i], abs(hi_a[i] - cl_a[i-1]), abs(lo_a[i] - cl_a[i-1]))

    def ws(arr: np.ndarray) -> pd.Series:
        return pd.Series(arr).ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    s_tr   = ws(tr_arr).replace(0.0, np.nan)
    plus_di  = 100.0 * ws(p_dm)  / s_tr
    minus_di = 100.0 * ws(m_dm)  / s_tr
    dx       = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_s    = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    adx_v   = _r(_last(adx_s))
    pdi_v   = _r(_last(plus_di))
    mdi_v   = _r(_last(minus_di))

    if   adx_v > 25 and pdi_v > mdi_v: sig = "strong_uptrend"
    elif adx_v > 25 and mdi_v > pdi_v: sig = "strong_downtrend"
    elif adx_v < 20:                    sig = "ranging"
    else:                               sig = "weak_trend"

    return ADXSignal(adx=adx_v, plus_di=pdi_v, minus_di=mdi_v, signal=sig)


# ── Parabolic SAR ───────────────────────────────────────────────────────────────

def _parabolic_sar(
    hi: pd.Series, lo: pd.Series, cl: pd.Series,
    af_step: float = 0.02, af_max: float = 0.20,
) -> ParabolicSAR:
    hi_a = hi.values.astype(float)
    lo_a = lo.values.astype(float)
    cl_a = cl.values.astype(float)
    n    = len(hi_a)

    if n < 2:
        p = float(cl_a[-1])
        return ParabolicSAR(sar=p, trend="bullish", acceleration=af_step, reversal_price=p)

    if cl_a[1] >= cl_a[0]:
        trend, ep, sar = 1,  hi_a[1], lo_a[0]
    else:
        trend, ep, sar = -1, lo_a[1], hi_a[0]
    af = af_step

    for i in range(2, n):
        sar_new = sar + af * (ep - sar)
        if trend == 1:
            sar_new = min(sar_new, lo_a[i-1], lo_a[i-2] if i >= 2 else lo_a[i-1])
            if lo_a[i] < sar_new:
                trend, sar_new, ep, af = -1, ep, lo_a[i], af_step
            elif hi_a[i] > ep:
                ep = hi_a[i]; af = min(af + af_step, af_max)
        else:
            sar_new = max(sar_new, hi_a[i-1], hi_a[i-2] if i >= 2 else hi_a[i-1])
            if hi_a[i] > sar_new:
                trend, sar_new, ep, af = 1, ep, hi_a[i], af_step
            elif lo_a[i] < ep:
                ep = lo_a[i]; af = min(af + af_step, af_max)
        sar = sar_new

    return ParabolicSAR(
        sar=_r(sar), trend="bullish" if trend == 1 else "bearish",
        acceleration=_r(af), reversal_price=_r(sar),
    )


# ── Composite score ─────────────────────────────────────────────────────────────

def _composite(ichi: IchimokuCloud, adx: ADXSignal, psar: ParabolicSAR) -> tuple[float, str]:
    ichi_score = {"bullish": 0.40, "bearish": -0.40, "neutral": 0.0}.get(ichi.signal, 0.0)

    if   adx.signal == "strong_uptrend":   adx_score =  0.30
    elif adx.signal == "strong_downtrend": adx_score = -0.30
    elif adx.signal == "weak_trend":       adx_score =  0.10 if adx.plus_di > adx.minus_di else -0.10
    else:                                  adx_score =  0.0

    psar_score = {"bullish": 0.30, "bearish": -0.30}.get(psar.trend, 0.0)

    score = max(-1.0, min(1.0, ichi_score + adx_score + psar_score))
    if   score >= 0.6:  label = "Strong Buy"
    elif score >= 0.2:  label = "Buy"
    elif score > -0.2:  label = "Neutral"
    elif score > -0.6:  label = "Sell"
    else:               label = "Strong Sell"
    return _r(score, 3), label


# ── Neutral fallbacks ───────────────────────────────────────────────────────────

def _neutral_ichimoku(p: float) -> IchimokuCloud:
    return IchimokuCloud(tenkan_sen=p, kijun_sen=p, senkou_span_a=p, senkou_span_b=p,
                         chikou_span=p, signal="neutral", price_vs_cloud="inside")

def _neutral_fibonacci(p: float) -> FibonacciLevels:
    levels = {lbl: _r(p - p * r) for lbl, r in _FIB_RATIOS}
    return FibonacciLevels(swing_high=p, swing_low=p, levels=levels,
                           nearest_support=p, nearest_resistance=p)

def _neutral_adx() -> ADXSignal:
    return ADXSignal(adx=0.0, plus_di=0.0, minus_di=0.0, signal="ranging")

def _neutral_psar(p: float) -> ParabolicSAR:
    return ParabolicSAR(sar=p, trend="bullish", acceleration=0.02, reversal_price=p)


# ── Core sync computation ───────────────────────────────────────────────────────

def _compute_advanced_ta_sync(ticker: str, period: str, interval: str) -> AdvancedTAResult:
    warnings: list[str] = []
    df    = _fetch_ohlcv_sync(ticker, period, interval)
    as_of = pd.Timestamp.now().strftime("%Y-%m-%d")

    if df.empty:
        warnings.append("no_data_returned_from_yfinance")
        logger.warning("advanced_ta_no_data", ticker=ticker)
        price = 0.0
        ichi, fib, adx_s, psar = (_neutral_ichimoku(price), _neutral_fibonacci(price),
                                   _neutral_adx(),           _neutral_psar(price))
        score, label = _composite(ichi, adx_s, psar)
        return AdvancedTAResult(ticker=ticker, as_of=as_of, price=price,
                                ichimoku=ichi, fibonacci=fib, adx=adx_s,
                                parabolic_sar=psar, composite_score=score,
                                composite_label=label, warnings=warnings)

    if len(df) < _MIN_BARS:
        warnings.append(f"insufficient_bars: got {len(df)}, need {_MIN_BARS}; signals may be unreliable")
        logger.warning("advanced_ta_insufficient_bars", ticker=ticker, bars=len(df))

    hi, lo, cl = df["High"], df["Low"], df["Close"]
    price = _last(cl)
    if hasattr(cl.index[-1], "strftime"):
        as_of = cl.index[-1].strftime("%Y-%m-%d")

    def _try(fn, fallback):
        try:
            return fn()
        except Exception as exc:
            warnings.append(f"{fn.__name__}_failed: {exc}")
            logger.error(f"{fn.__name__}_failed", ticker=ticker, error=str(exc))
            return fallback

    ichi  = _try(lambda: _ichimoku(hi, lo, cl),      _neutral_ichimoku(price))
    fib   = _try(lambda: _fibonacci(cl),              _neutral_fibonacci(price))
    adx_s = _try(lambda: _adx(hi, lo, cl),            _neutral_adx())
    psar  = _try(lambda: _parabolic_sar(hi, lo, cl),  _neutral_psar(price))

    score, label = _composite(ichi, adx_s, psar)
    logger.info("advanced_ta_computed", ticker=ticker, composite_score=score,
                composite_label=label, ichimoku_signal=ichi.signal,
                adx_signal=adx_s.signal, psar_trend=psar.trend)

    return AdvancedTAResult(
        ticker=ticker, as_of=as_of, price=_r(price),
        ichimoku=ichi, fibonacci=fib, adx=adx_s, parabolic_sar=psar,
        composite_score=score, composite_label=label, warnings=warnings,
    )


# ── Public async interface ──────────────────────────────────────────────────────

async def get_advanced_ta(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> AdvancedTAResult:
    """
    Compute advanced TA signals for *ticker* asynchronously.

    All CPU/IO-bound work runs in a thread pool via asyncio.to_thread.
    Returns an AdvancedTAResult containing Ichimoku Cloud, Fibonacci Retracement,
    ADX/DI, Parabolic SAR, and a [-1, 1] composite score with label.

    Raises no exceptions — all failures are captured in AdvancedTAResult.warnings.
    """
    return await asyncio.to_thread(_compute_advanced_ta_sync, ticker, period, interval)
