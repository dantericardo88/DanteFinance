"""
Technical analysis engine — Dimensions 69-76 of the SENTINEL competitive matrix.

Computes 14+ TA signals (trend, momentum, volatility, volume, oscillators) for any
ticker via yfinance and returns a composite score, trend label, and support/resistance.
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


# ── Models ─────────────────────────────────────────────────────────────────────

class TASignal(BaseModel):
    name: str
    value: float
    signal: str        # "bullish" | "bearish" | "neutral"
    description: str


class TASignalResult(BaseModel):
    ticker: str
    as_of: str
    price: float
    signals: list[TASignal]
    composite_score: float        # -1.0 to 1.0
    composite_label: str          # "Strong Buy" | "Buy" | "Neutral" | "Sell" | "Strong Sell"
    trend: str                    # "uptrend" | "downtrend" | "sideways"
    support_levels: list[float]
    resistance_levels: list[float]
    warnings: list[str]


# ── Data fetch ──────────────────────────────────────────────────────────────────

def _fetch_ohlcv_sync(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not _YF:
        logger.warning("yfinance_not_installed", ticker=ticker)
        return pd.DataFrame()
    logger.info("fetching_ohlcv", ticker=ticker, period=period, interval=interval)
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
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
        logger.warning("ohlcv_columns_missing", ticker=ticker, columns=list(raw.columns))
        return pd.DataFrame()
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy().sort_index().dropna(subset=["Close"])
    logger.info("ohlcv_fetched", rows=len(df), ticker=ticker)
    return df


# ── Indicator math (pandas/numpy only) ─────────────────────────────────────────

def _sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p, min_periods=p).mean()

def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def _rsi(close: pd.Series, p: int = 14) -> pd.Series:
    d = close.diff()
    ag = d.clip(lower=0.0).ewm(com=p - 1, min_periods=p).mean()
    al = (-d).clip(lower=0.0).ewm(com=p - 1, min_periods=p).mean()
    return 100.0 - 100.0 / (1.0 + ag / al.replace(0.0, np.nan))

def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ml = _ema(close, 12) - _ema(close, 26)
    sig = _ema(ml, 9)
    return ml, sig, ml - sig

def _bollinger(close: pd.Series, p: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(close, p)
    std = close.rolling(p, min_periods=p).std(ddof=1)
    return mid + k * std, mid, mid - k * std

def _atr(hi: pd.Series, lo: pd.Series, cl: pd.Series, p: int = 14) -> pd.Series:
    pc = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p - 1, min_periods=p).mean()

def _obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    return (np.sign(close.diff().fillna(0.0)) * vol).cumsum()

def _vwap_rolling(hi: pd.Series, lo: pd.Series, cl: pd.Series, vol: pd.Series, p: int = 20) -> pd.Series:
    tp = (hi + lo + cl) / 3.0
    return (tp * vol).rolling(p, min_periods=1).sum() / vol.rolling(p, min_periods=1).sum()

def _stochastic(hi: pd.Series, lo: pd.Series, cl: pd.Series, kp: int = 14, dp: int = 3) -> tuple[pd.Series, pd.Series]:
    lo_k = lo.rolling(kp, min_periods=kp).min()
    hi_k = hi.rolling(kp, min_periods=kp).max()
    pct_k = 100.0 * (cl - lo_k) / (hi_k - lo_k).replace(0.0, np.nan)
    return pct_k, pct_k.rolling(dp, min_periods=dp).mean()

def _williams_r(hi: pd.Series, lo: pd.Series, cl: pd.Series, p: int = 14) -> pd.Series:
    hh = hi.rolling(p, min_periods=p).max()
    ll = lo.rolling(p, min_periods=p).min()
    return -100.0 * (hh - cl) / (hh - ll).replace(0.0, np.nan)


# ── Signal helpers ──────────────────────────────────────────────────────────────

def _sw(bullish: bool, bearish: bool) -> str:
    return "bullish" if bullish else ("bearish" if bearish else "neutral")

def _sig(name: str, value: float, signal: str, description: str) -> TASignal:
    return TASignal(name=name, value=round(float(value), 4), signal=signal, description=description)

def _last(s: pd.Series) -> float:
    return float(s.iloc[-1])

def _nan(v: float) -> bool:
    return math.isnan(v)


# ── Signal computation ──────────────────────────────────────────────────────────

def _compute_signals(df: pd.DataFrame, warnings: list[str]) -> list[TASignal]:  # noqa: C901
    out: list[TASignal] = []
    cl, hi, lo, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    n, price = len(df), _last(df["Close"])

    # Trend — SMA 20/50/200
    for p in (20, 50, 200):
        if n < p:
            warnings.append(f"Need {p} bars for SMA{p}, got {n}")
            continue
        v = _last(_sma(cl, p))
        if not _nan(v):
            bull = price > v
            out.append(_sig(f"SMA{p}", v, _sw(bull, not bull),
                            f"Price {'above' if bull else 'below'} SMA{p} ({v:.2f})"))

    # Trend — EMA 12/26 cross
    e12, e26 = _last(_ema(cl, 12)), _last(_ema(cl, 26))
    if not (_nan(e12) or _nan(e26)):
        bull = e12 > e26
        out.append(_sig("EMA_Cross_12_26", e12 - e26, _sw(bull, not bull),
                        f"EMA12 ({e12:.2f}) {'above' if bull else 'below'} EMA26 ({e26:.2f})"))

    # Momentum — RSI(14)
    if n >= 15:
        rv = _last(_rsi(cl, 14))
        if not _nan(rv):
            sig = "bearish" if rv >= 70 else ("bullish" if rv <= 30 else "neutral")
            out.append(_sig("RSI_14", rv, sig,
                            f"RSI {rv:.1f} — {'overbought' if rv >= 70 else 'oversold' if rv <= 30 else 'neutral'}"))
    else:
        warnings.append(f"Need 15 bars for RSI, got {n}")

    # Momentum — MACD(12,26,9)
    if n >= 35:
        ml, ms, mh = _macd(cl)
        mlv, msv, mhv = _last(ml), _last(ms), _last(mh)
        if not any(_nan(v) for v in (mlv, msv, mhv)):
            bull = mlv > msv
            out.append(_sig("MACD_Line", mlv, _sw(bull, not bull),
                            f"MACD ({mlv:.4f}) {'above' if bull else 'below'} signal ({msv:.4f})"))
            prev_h = _last(mh.iloc[:-1]) if n > 35 else 0.0
            rising = mhv > prev_h
            out.append(_sig("MACD_Histogram", mhv,
                            _sw(mhv > 0 and rising, mhv < 0 and not rising),
                            f"Histogram {mhv:.4f} ({'expanding' if rising else 'contracting'})"))
    else:
        warnings.append(f"Need 35 bars for MACD, got {n}")

    # Volatility — Bollinger Bands %B + Bandwidth
    if n >= 20:
        bu, bm, bl = _bollinger(cl)
        buv, bmv, blv = _last(bu), _last(bm), _last(bl)
        if not any(_nan(v) for v in (buv, bmv, blv)):
            bw = buv - blv
            pct_b = (price - blv) / bw if bw > 0 else 0.5
            bw_pct = bw / bmv if bmv != 0 else 0.0
            bb_sig = "bearish" if pct_b >= 1.0 else ("bullish" if pct_b <= 0.0 else "neutral")
            out.append(_sig("BB_PctB", pct_b, bb_sig,
                            f"%B {pct_b:.2f} — {'above upper' if pct_b >= 1 else 'below lower' if pct_b <= 0 else 'within'} band"))
            out.append(_sig("BB_Bandwidth", bw_pct, "neutral",
                            f"Bandwidth {bw_pct:.3f} ({'high' if bw_pct > 0.1 else 'low'} volatility)"))

    # Volatility — ATR(14)
    if n >= 15:
        av = _last(_atr(hi, lo, cl, 14))
        if not _nan(av):
            out.append(_sig("ATR_14", av, "neutral",
                            f"ATR {av:.2f} ({av / price:.1%} of price)"))

    # Volume — OBV trend (10-bar)
    if n >= 10:
        obv = _obv(cl, vol)
        rising = _last(obv) > float(obv.iloc[-10])
        out.append(_sig("OBV", _last(obv), _sw(rising, not rising),
                        f"OBV {'rising' if rising else 'falling'} — volume {'confirms' if rising else 'diverges'}"))

    # Volume — VWAP deviation
    if n >= 5:
        vv = _last(_vwap_rolling(hi, lo, cl, vol, min(20, n)))
        if not _nan(vv):
            above = price > vv
            out.append(_sig("VWAP_Deviation", vv, _sw(above, not above),
                            f"Price {'above' if above else 'below'} VWAP ({vv:.2f})"))

    # Volume — Volume vs SMA ratio
    vp = min(20, n)
    vol_avg = _last(_sma(vol, vp))
    vol_now = _last(vol)
    if vol_avg > 0 and not _nan(vol_avg):
        vr = vol_now / vol_avg
        prev_cl = float(cl.iloc[-2]) if n >= 2 else price
        out.append(_sig("Volume_Ratio", vr,
                        _sw(vr >= 1.5 and price > prev_cl, vr >= 1.5 and price < prev_cl),
                        f"Volume {vr:.2f}x avg ({'high' if vr >= 1.5 else 'low' if vr < 0.5 else 'normal'})"))

    # Oscillator — Stochastic(14,3)
    if n >= 17:
        kv, dv = _stochastic(hi, lo, cl)
        kv, dv = _last(kv), _last(dv)
        if not any(_nan(v) for v in (kv, dv)):
            sig = "bearish" if kv >= 80 else ("bullish" if kv <= 20 else "neutral")
            out.append(_sig("Stochastic_K", kv, sig,
                            f"Stoch %K {kv:.1f} — {'overbought' if kv >= 80 else 'oversold' if kv <= 20 else 'neutral'}"))

    # Oscillator — Williams %R(14)
    if n >= 15:
        wrv = _last(_williams_r(hi, lo, cl, 14))
        if not _nan(wrv):
            sig = "bearish" if wrv >= -20 else ("bullish" if wrv <= -80 else "neutral")
            out.append(_sig("Williams_R_14", wrv, sig,
                            f"W%R {wrv:.1f} — {'overbought' if wrv >= -20 else 'oversold' if wrv <= -80 else 'neutral'}"))

    return out


# ── Composite scoring ────────────────────────────────────────────────────────────

_SIGNAL_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
_SKIP = {"ATR_14", "BB_Bandwidth"}
_TREND_PFX = ("SMA", "EMA")
_MOMENTUM_PFX = ("RSI", "MACD", "Stochastic", "Williams")
_CAT_WEIGHTS = {"trend": 0.40, "momentum": 0.35, "volume": 0.25}


def _category(name: str) -> str:
    if any(name.startswith(p) for p in _TREND_PFX):
        return "trend"
    if any(name.startswith(p) for p in _MOMENTUM_PFX):
        return "momentum"
    return "volume"


def _composite_score(signals: list[TASignal]) -> float:
    buckets: dict[str, list[float]] = {"trend": [], "momentum": [], "volume": []}
    for s in signals:
        if s.name not in _SKIP:
            buckets[_category(s.name)].append(_SIGNAL_MAP.get(s.signal, 0.0))
    score = total_w = 0.0
    for cat, w in _CAT_WEIGHTS.items():
        vals = buckets[cat]
        if vals:
            score += w * sum(vals) / len(vals)
            total_w += w
    return round(score / total_w, 4) if total_w > 0 else 0.0


def _composite_label(score: float) -> str:
    if score >= 0.6: return "Strong Buy"
    if score >= 0.2: return "Buy"
    if score <= -0.6: return "Strong Sell"
    if score <= -0.2: return "Sell"
    return "Neutral"


def _trend_label(df: pd.DataFrame) -> str:
    cl, n = df["Close"], len(df["Close"])
    if n < 5:
        return "sideways"
    sma = _sma(cl, min(20, n))
    slope = _last(sma) - float(sma.iloc[-5])
    price = _last(cl)
    if n >= 50:
        sma50 = _last(_sma(cl, 50))
        if slope > 0 and price > sma50: return "uptrend"
        if slope < 0 and price < sma50: return "downtrend"
        return "sideways"
    if slope > 0: return "uptrend"
    if slope < 0: return "downtrend"
    return "sideways"


# ── Support / Resistance ─────────────────────────────────────────────────────────

def _pivot_points(h: float, l: float, c: float) -> tuple[float, list[float], list[float]]:
    pp = (h + l + c) / 3.0
    return pp, [2*pp - l, pp + (h - l), h + 2*(pp - l)], [2*pp - h, pp - (h - l), l - 2*(h - pp)]


def _dedup(levels: list[float], tol: float = 0.005) -> list[float]:
    out: list[float] = []
    for lv in sorted(levels):
        if not out or abs(lv - out[-1]) / max(abs(out[-1]), 1e-8) > tol:
            out.append(lv)
    return out


def _build_sr_levels(df: pd.DataFrame, n: int = 3) -> tuple[list[float], list[float]]:
    cl, hi, lo = df["Close"], df["High"], df["Low"]
    price = _last(cl)
    idx = -2 if len(df) >= 2 else -1
    _, pp_r, pp_s = _pivot_points(float(hi.iloc[idx]), float(lo.iloc[idx]), float(cl.iloc[idx]))
    look = hi.iloc[-20:].values
    recents_hi = [float(v) for v in look if v > price]
    recents_lo = [float(v) for v in lo.iloc[-20:].values if v < price]
    support = _dedup(sorted([s for s in pp_s + recents_lo if s < price], reverse=True))[:n]
    resist = _dedup(sorted([r for r in pp_r + recents_hi if r > price]))[:n]
    return [round(v, 2) for v in support], [round(v, 2) for v in resist]


# ── Main sync worker (called via asyncio.to_thread) ──────────────────────────────

def _compute_ta_sync(ticker: str, period: str, interval: str) -> TASignalResult:
    warnings: list[str] = []
    df = _fetch_ohlcv_sync(ticker, period, interval)
    if df.empty:
        raise RuntimeError(
            f"No OHLCV data for {ticker!r} (period={period!r}, interval={interval!r}). "
            "Check the ticker symbol and yfinance installation."
        )
    price = float(df["Close"].iloc[-1])
    as_of = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
    signals = _compute_signals(df, warnings)
    if not signals:
        warnings.append("No signals computed — possibly too few bars.")
    score = _composite_score(signals)
    label = _composite_label(score)
    trend = _trend_label(df)
    support, resistance = _build_sr_levels(df)
    logger.info("ta_signals_computed", ticker=ticker, price=price, n_signals=len(signals),
                composite_score=score, composite_label=label, trend=trend)
    return TASignalResult(
        ticker=ticker.upper(), as_of=as_of, price=round(price, 2),
        signals=signals, composite_score=score, composite_label=label,
        trend=trend, support_levels=support, resistance_levels=resistance, warnings=warnings,
    )


# ── Public async entry point ─────────────────────────────────────────────────────

async def get_ta_signals(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> TASignalResult:
    """Compute a full TA signal suite for *ticker*.

    Args:
        ticker:   Yahoo Finance ticker (e.g. "AAPL", "BTC-USD").
        period:   yfinance period string ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max").
        interval: yfinance interval string ("1d", "1wk", "1h", "15m", etc.).

    Returns:
        TASignalResult with 14+ signals, composite score/label, trend, and S/R levels.
    """
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    ticker = ticker.strip().upper()
    logger.info("get_ta_signals_start", ticker=ticker, period=period, interval=interval)
    return await asyncio.to_thread(_compute_ta_sync, ticker, period, interval)
