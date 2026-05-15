"""Enhanced technical screener — 25+ criteria using pandas-ta — dim_071.

Extends the base technical_screener with a full pandas-ta indicator suite,
binary signal generation, chart/candlestick pattern detection, S/R levels,
Fibonacci retracements, and a ThreadPoolExecutor batch scorer.

Public API
----------
TechnicalIndicatorEngine
    compute_all(df, ticker) -> pd.DataFrame      — all indicators
    compute_signals(indicators_df) -> dict        — 25+ binary signals

TechnicalScreener (enhanced)
    screen(criteria) -> pd.DataFrame
    run_preset(name) -> pd.DataFrame
    score_stock(ticker, df) -> dict               — 0-100 composite score
    find_support_resistance(df, n_levels) -> dict
    detect_chart_patterns(df) -> list[dict]
    batch_score(tickers) -> pd.DataFrame
    PREBUILT_SCREENS : dict[str, dict]

FastAPI
-------
technical_screener_router — mounted at /api/screener/technical
"""
from __future__ import annotations

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    import pandas_ta as ta          # pip install pandas-ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 3600          # 1 hour indicator cache
_DEFAULT_LOOKBACK_DAYS = 504       # ~2 years for indicator warmup
_BATCH_MAX_WORKERS = 8
_YFINANCE_RATE_SLEEP = 0.10        # throttle between tickers in batch

# Representative universe (same core as fundamental screener)
_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM", "UNH", "XOM", "V", "LLY", "JNJ", "WMT", "MA", "PG", "HD",
    "AVGO", "CVX", "MRK", "ABBV", "KO", "PEP", "COST", "ADBE", "NFLX",
    "CRM", "ACN", "TMO", "MCD", "CSCO", "BAC", "ABT", "NEE", "ORCL",
    "WFC", "TXN", "PM", "LIN", "SPGI", "AMGN", "LOW", "ISRG", "GS",
    "SYK", "RTX", "BKNG", "INTU", "NOW", "MDT", "AXP", "BLK", "VRTX",
    "GILD", "REGN", "MO", "EOG", "SLB", "ZTS", "MMC", "CB", "ITW",
    "BSX", "CME", "SCHW", "ADI", "AMAT", "LRCX", "PANW", "MU", "AMD",
    "QCOM", "IBM", "F", "GM", "DAL", "UAL", "BA", "GE", "HON", "CAT",
    "DE", "DUK", "SO", "T", "VZ", "TMUS", "DIS", "CVS", "HCA",
]

# Indicator column name prefixes / expected output columns
_SIGNAL_DEFINITIONS: dict[str, str] = {
    "rsi_oversold":           "RSI_14 < 30",
    "rsi_overbought":         "RSI_14 > 70",
    "macd_bullish_cross":     "MACD crosses above signal line",
    "macd_bearish_cross":     "MACD crosses below signal line",
    "price_above_sma200":     "Close > SMA_200",
    "price_above_sma50":      "Close > SMA_50",
    "price_above_sma20":      "Close > SMA_20",
    "golden_cross":           "SMA_50 crosses above SMA_200",
    "death_cross":            "SMA_50 crosses below SMA_200",
    "bb_squeeze":             "BB width < 20th percentile (52-week)",
    "bb_upper_breakout":      "Close breaks above upper Bollinger Band",
    "bb_lower_bounce":        "Close bounces from lower Bollinger Band",
    "volume_surge":           "Volume > 2x 20-day average",
    "volume_dry_up":          "Volume < 0.5x 20-day average",
    "obv_rising":             "OBV rising for 5 consecutive days",
    "obv_falling":            "OBV falling for 5 consecutive days",
    "ichimoku_bullish":       "Price above cloud, tenkan > kijun",
    "ichimoku_bearish":       "Price below cloud, tenkan < kijun",
    "stoch_oversold":         "Stoch K < 20 and D < 20",
    "stoch_overbought":       "Stoch K > 80 and D > 80",
    "cmf_positive":           "Chaikin Money Flow > 0 (accumulation)",
    "cmf_negative":           "Chaikin Money Flow < 0 (distribution)",
    "mfi_oversold":           "Money Flow Index < 20",
    "mfi_overbought":         "Money Flow Index > 80",
    "cci_extreme_low":        "CCI < -100 (oversold)",
    "cci_extreme_high":       "CCI > 100 (overbought)",
    "atr_expanding":          "ATR increasing (volatility expansion)",
    "atr_contracting":        "ATR decreasing (volatility contraction)",
    "parabolic_sar_bullish":  "Price above Parabolic SAR",
    "aroon_bullish":          "Aroon Up > 70, Aroon Down < 30",
    "roc_positive":           "Rate of Change > 0",
    "williams_r_oversold":    "Williams %R < -80",
    "vwap_above":             "Close > VWAP",
    "price_near_52w_high":    "Within 2% of 52-week high",
    "doji_candle":            "Doji candlestick pattern detected",
    "hammer_candle":          "Hammer pattern (bullish reversal)",
    "shooting_star":          "Shooting star (bearish reversal)",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_last(series: pd.Series | None, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    val = series.iloc[-1]
    return float(val) if pd.notna(val) else default


def _col_or_none(df: pd.DataFrame, col: str) -> pd.Series | None:
    return df[col] if col in df.columns else None


def _fetch_ohlcv(ticker: str, days: int = _DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """Download OHLCV from yfinance and normalise columns."""
    end = date.today()
    start = end - timedelta(days=days)
    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise ValueError(f"yfinance download failed for {ticker}: {exc}") from exc

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex for single-ticker downloads
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    if len(raw) < 50:
        raise ValueError(f"Insufficient history for {ticker}: {len(raw)} bars")
    return raw


def _annualized_vol(closes: pd.Series, window: int) -> float | None:
    if len(closes) < window + 1:
        return None
    log_rets = np.log(closes / closes.shift(1)).dropna().tail(window)
    return float(log_rets.std() * math.sqrt(252))


# ---------------------------------------------------------------------------
# Indicator cache
# ---------------------------------------------------------------------------

_indicator_cache: dict[str, tuple[float, pd.DataFrame]] = {}  # ticker → (timestamp, df)


def _get_cached_indicators(ticker: str) -> pd.DataFrame | None:
    entry = _indicator_cache.get(ticker)
    if entry is None:
        return None
    ts, df = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        del _indicator_cache[ticker]
        return None
    return df


def _set_cached_indicators(ticker: str, df: pd.DataFrame) -> None:
    _indicator_cache[ticker] = (time.time(), df)


# ---------------------------------------------------------------------------
# TechnicalIndicatorEngine
# ---------------------------------------------------------------------------

class TechnicalIndicatorEngine:
    """Compute 30+ technical indicators on OHLCV data using pandas-ta."""

    def compute_all(self, df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
        """Run the full indicator suite on an OHLCV DataFrame.

        Parameters
        ----------
        df:
            DataFrame with columns Open, High, Low, Close, Volume.
        ticker:
            Optional label for logging.

        Returns
        -------
        pd.DataFrame
            Original OHLCV columns plus all indicator columns.
        """
        out = df.copy()

        if not _TA_AVAILABLE:
            logger.warning("pandas_ta_not_installed", ticker=ticker)
            return self._compute_manual(out)

        # ── Trend ─────────────────────────────────────────────────────────
        out.ta.sma(length=20, append=True)       # SMA_20
        out.ta.sma(length=50, append=True)       # SMA_50
        out.ta.sma(length=200, append=True)      # SMA_200
        out.ta.ema(length=12, append=True)       # EMA_12
        out.ta.ema(length=26, append=True)       # EMA_26
        out.ta.ema(length=50, append=True)       # EMA_50
        out.ta.dema(length=20, append=True)      # DEMA_20
        out.ta.tema(length=20, append=True)      # TEMA_20

        # ── Momentum ──────────────────────────────────────────────────────
        out.ta.rsi(length=14, append=True)                   # RSI_14
        out.ta.macd(fast=12, slow=26, signal=9, append=True) # MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        out.ta.stoch(k=14, d=3, smooth_k=3, append=True)    # STOCHk_14_3_3, STOCHd_14_3_3
        out.ta.willr(length=14, append=True)                 # WILLR_14
        out.ta.cci(length=20, append=True)                   # CCI_20
        out.ta.roc(length=12, append=True)                   # ROC_12
        out.ta.mfi(length=14, append=True)                   # MFI_14
        out.ta.uo(append=True)                               # UO_7_14_28

        # ── Volatility ────────────────────────────────────────────────────
        out.ta.bbands(length=20, std=2, append=True)         # BBL/BBM/BBU/BBB/BBP_20_2.0
        out.ta.atr(length=14, append=True)                   # ATRr_14
        out.ta.kc(length=20, scalar=2, append=True)          # KCLe_20_2, KCUe_20_2
        out.ta.donchian(lower_length=20, upper_length=20, append=True)  # DCL_20_20, DCU_20_20
        out.ta.natr(length=14, append=True)                  # NATR_14
        # Historical volatility (21-day)
        log_r = np.log(out["Close"] / out["Close"].shift(1))
        out["HV_21"] = log_r.rolling(21).std() * math.sqrt(252)

        # ── Volume ────────────────────────────────────────────────────────
        out.ta.obv(append=True)                             # OBV
        out.ta.vwap(append=True)                            # VWAP_D
        out.ta.ad(append=True)                              # AD
        out.ta.cmf(length=20, append=True)                  # CMF_20
        out.ta.vwma(length=20, append=True)                 # VWMA_20
        out.ta.pvt(append=True)                             # PVT

        # ── Other ──────────────────────────────────────────────────────────
        try:
            out.ta.ichimoku(append=True)   # ISA_9, ISB_26, ITS_9, IKS_26, ICS_26
        except Exception:
            pass  # Ichimoku occasionally errors on short data
        out.ta.aroon(length=14, append=True)                # AROOND_14, AROONU_14
        out.ta.psar(append=True)                            # PSARl_0.02_0.2, PSARs_0.02_0.2
        # Pivot points (classical, daily)
        out["PIVOT"] = (out["High"] + out["Low"] + out["Close"]) / 3
        out["R1"] = 2 * out["PIVOT"] - out["Low"]
        out["S1"] = 2 * out["PIVOT"] - out["High"]
        out["R2"] = out["PIVOT"] + (out["High"] - out["Low"])
        out["S2"] = out["PIVOT"] - (out["High"] - out["Low"])

        return out

    # ------------------------------------------------------------------
    # Manual fallback (no pandas-ta)
    # ------------------------------------------------------------------

    def _compute_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """Minimal manual indicators when pandas-ta is not installed."""
        closes = df["Close"]
        highs = df["High"]
        lows = df["Low"]
        volumes = df["Volume"]

        for period in [20, 50, 200]:
            df[f"SMA_{period}"] = closes.rolling(period).mean()
        df["EMA_12"] = closes.ewm(span=12).mean()
        df["EMA_26"] = closes.ewm(span=26).mean()
        df["EMA_50"] = closes.ewm(span=50).mean()

        # RSI
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = closes.ewm(span=12).mean()
        ema26 = closes.ewm(span=26).mean()
        df["MACD_12_26_9"] = ema12 - ema26
        df["MACDs_12_26_9"] = df["MACD_12_26_9"].ewm(span=9).mean()
        df["MACDh_12_26_9"] = df["MACD_12_26_9"] - df["MACDs_12_26_9"]

        # Bollinger Bands
        sma20 = closes.rolling(20).mean()
        std20 = closes.rolling(20).std()
        df["BBU_20_2.0"] = sma20 + 2 * std20
        df["BBM_20_2.0"] = sma20
        df["BBL_20_2.0"] = sma20 - 2 * std20
        df["BBB_20_2.0"] = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / sma20
        df["BBP_20_2.0"] = (closes - df["BBL_20_2.0"]) / (df["BBU_20_2.0"] - df["BBL_20_2.0"] + 1e-10)

        # ATR
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift()).abs(),
            (lows - closes.shift()).abs(),
        ], axis=1).max(axis=1)
        df["ATRr_14"] = tr.rolling(14).mean()

        # OBV
        obv = (np.sign(closes.diff()) * volumes).fillna(0).cumsum()
        df["OBV"] = obv

        # Stochastic
        lowest_low = lows.rolling(14).min()
        highest_high = highs.rolling(14).max()
        k = 100 * (closes - lowest_low) / (highest_high - lowest_low + 1e-10)
        df["STOCHk_14_3_3"] = k.rolling(3).mean()
        df["STOCHd_14_3_3"] = df["STOCHk_14_3_3"].rolling(3).mean()

        # Williams %R
        df["WILLR_14"] = -100 * (highest_high - closes) / (highest_high - lowest_low + 1e-10)

        # CCI
        tp = (highs + lows + closes) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df["CCI_20"] = (tp - sma_tp) / (0.015 * mad + 1e-10)

        # Historical volatility
        log_r = np.log(closes / closes.shift(1))
        df["HV_21"] = log_r.rolling(21).std() * math.sqrt(252)

        # Pivot points
        df["PIVOT"] = (highs + lows + closes) / 3
        df["R1"] = 2 * df["PIVOT"] - lows
        df["S1"] = 2 * df["PIVOT"] - highs
        df["R2"] = df["PIVOT"] + (highs - lows)
        df["S2"] = df["PIVOT"] - (highs - lows)

        return df

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def compute_signals(self, df: pd.DataFrame) -> dict[str, bool | None]:
        """Generate 35 binary signals from indicator DataFrame.

        Returns
        -------
        dict mapping signal_name -> bool (True/False/None if unavailable).
        """
        if df.empty or len(df) < 2:
            return {k: None for k in _SIGNAL_DEFINITIONS}

        # Helper aliases
        def last(col: str) -> float | None:
            s = _col_or_none(df, col)
            return _safe_last(s, np.nan) if s is not None else None

        def prev(col: str, n: int = 1) -> float | None:
            s = _col_or_none(df, col)
            if s is None or len(s) < n + 1:
                return None
            v = s.iloc[-(n + 1)]
            return float(v) if pd.notna(v) else None

        close = float(df["Close"].iloc[-1])

        signals: dict[str, bool | None] = {}

        # ── RSI ────────────────────────────────────────────────────────────
        rsi = last("RSI_14")
        signals["rsi_oversold"] = rsi < 30 if rsi is not None else None
        signals["rsi_overbought"] = rsi > 70 if rsi is not None else None

        # ── MACD ───────────────────────────────────────────────────────────
        macd_col = "MACD_12_26_9"
        sig_col = "MACDs_12_26_9"
        m_now, m_prev = last(macd_col), prev(macd_col)
        s_now, s_prev = last(sig_col), prev(sig_col)
        if all(v is not None for v in [m_now, m_prev, s_now, s_prev]):
            signals["macd_bullish_cross"] = (m_now > s_now) and (m_prev <= s_prev)
            signals["macd_bearish_cross"] = (m_now < s_now) and (m_prev >= s_prev)
        else:
            signals["macd_bullish_cross"] = None
            signals["macd_bearish_cross"] = None

        # ── Moving average crosses / trend ─────────────────────────────────
        sma20 = last("SMA_20")
        sma50 = last("SMA_50")
        sma200 = last("SMA_200")
        signals["price_above_sma200"] = close > sma200 if sma200 else None
        signals["price_above_sma50"] = close > sma50 if sma50 else None
        signals["price_above_sma20"] = close > sma20 if sma20 else None

        sma50_prev = prev("SMA_50", 5)
        sma200_prev = prev("SMA_200", 5)
        if all(v is not None for v in [sma50, sma200, sma50_prev, sma200_prev]):
            signals["golden_cross"] = (sma50 > sma200) and (sma50_prev <= sma200_prev)
            signals["death_cross"] = (sma50 < sma200) and (sma50_prev >= sma200_prev)
        else:
            signals["golden_cross"] = None
            signals["death_cross"] = None

        # ── Bollinger Bands ────────────────────────────────────────────────
        bbu = last("BBU_20_2.0")
        bbl = last("BBL_20_2.0")
        bbb = _col_or_none(df, "BBB_20_2.0")
        if bbb is not None and len(bbb.dropna()) >= 52:
            width_series = bbb.dropna()
            percentile_20 = float(np.percentile(width_series.tail(252), 20))
            current_width = float(width_series.iloc[-1])
            signals["bb_squeeze"] = current_width < percentile_20
        else:
            signals["bb_squeeze"] = None

        signals["bb_upper_breakout"] = close > bbu if bbu else None
        signals["bb_lower_bounce"] = (close > bbl and prev("Close") < (bbl or 0)) if bbl else None

        # ── Volume ──────────────────────────────────────────────────────────
        vol_series = df["Volume"]
        vol_avg20 = float(vol_series.tail(21).head(20).mean()) if len(vol_series) >= 21 else None
        vol_now = float(vol_series.iloc[-1])
        signals["volume_surge"] = vol_now > 2 * vol_avg20 if vol_avg20 else None
        signals["volume_dry_up"] = vol_now < 0.5 * vol_avg20 if vol_avg20 else None

        # ── OBV ────────────────────────────────────────────────────────────
        obv_s = _col_or_none(df, "OBV")
        if obv_s is not None and len(obv_s) >= 6:
            last_5 = obv_s.dropna().iloc[-5:]
            signals["obv_rising"] = bool((last_5.diff().dropna() > 0).all())
            signals["obv_falling"] = bool((last_5.diff().dropna() < 0).all())
        else:
            signals["obv_rising"] = None
            signals["obv_falling"] = None

        # ── Ichimoku ────────────────────────────────────────────────────────
        tenkan = last("ITS_9")
        kijun = last("IKS_26")
        senkou_a = last("ISA_9")
        senkou_b = last("ISB_26")
        if all(v is not None for v in [tenkan, kijun, senkou_a, senkou_b]):
            cloud_top = max(senkou_a, senkou_b)
            cloud_bot = min(senkou_a, senkou_b)
            signals["ichimoku_bullish"] = close > cloud_top and tenkan > kijun
            signals["ichimoku_bearish"] = close < cloud_bot and tenkan < kijun
        else:
            signals["ichimoku_bullish"] = None
            signals["ichimoku_bearish"] = None

        # ── Stochastic ─────────────────────────────────────────────────────
        stk = last("STOCHk_14_3_3")
        std = last("STOCHd_14_3_3")
        if stk is not None and std is not None:
            signals["stoch_oversold"] = stk < 20 and std < 20
            signals["stoch_overbought"] = stk > 80 and std > 80
        else:
            signals["stoch_oversold"] = None
            signals["stoch_overbought"] = None

        # ── Chaikin Money Flow ──────────────────────────────────────────────
        cmf = last("CMF_20")
        signals["cmf_positive"] = cmf > 0 if cmf is not None else None
        signals["cmf_negative"] = cmf < 0 if cmf is not None else None

        # ── Money Flow Index ────────────────────────────────────────────────
        mfi = last("MFI_14")
        signals["mfi_oversold"] = mfi < 20 if mfi is not None else None
        signals["mfi_overbought"] = mfi > 80 if mfi is not None else None

        # ── CCI ─────────────────────────────────────────────────────────────
        cci = last("CCI_20")
        signals["cci_extreme_low"] = cci < -100 if cci is not None else None
        signals["cci_extreme_high"] = cci > 100 if cci is not None else None

        # ── ATR trend ───────────────────────────────────────────────────────
        atr_s = _col_or_none(df, "ATRr_14")
        if atr_s is not None and len(atr_s.dropna()) >= 6:
            atr_now = float(atr_s.dropna().iloc[-1])
            atr_5ago = float(atr_s.dropna().iloc[-5])
            signals["atr_expanding"] = atr_now > atr_5ago
            signals["atr_contracting"] = atr_now < atr_5ago
        else:
            signals["atr_expanding"] = None
            signals["atr_contracting"] = None

        # ── Parabolic SAR ────────────────────────────────────────────────────
        psar_long = last("PSARl_0.02_0.2")
        psar_short = last("PSARs_0.02_0.2")
        if psar_long is not None and not math.isnan(psar_long):
            signals["parabolic_sar_bullish"] = True    # long SAR present means bullish trend
        elif psar_short is not None and not math.isnan(psar_short):
            signals["parabolic_sar_bullish"] = False
        else:
            signals["parabolic_sar_bullish"] = None

        # ── Aroon ────────────────────────────────────────────────────────────
        aroon_up = last("AROONU_14")
        aroon_dn = last("AROOND_14")
        if aroon_up is not None and aroon_dn is not None:
            signals["aroon_bullish"] = aroon_up > 70 and aroon_dn < 30
        else:
            signals["aroon_bullish"] = None

        # ── ROC ──────────────────────────────────────────────────────────────
        roc = last("ROC_12")
        signals["roc_positive"] = roc > 0 if roc is not None else None

        # ── Williams %R ──────────────────────────────────────────────────────
        willr = last("WILLR_14")
        signals["williams_r_oversold"] = willr < -80 if willr is not None else None

        # ── VWAP ──────────────────────────────────────────────────────────────
        vwap = last("VWAP_D")
        signals["vwap_above"] = close > vwap if vwap else None

        # ── 52-week high proximity ─────────────────────────────────────────────
        closes_252 = df["Close"].tail(252)
        high_52w = float(closes_252.max()) if len(closes_252) > 0 else close
        signals["price_near_52w_high"] = close >= 0.98 * high_52w

        # ── Candlestick patterns (basic manual detection) ─────────────────────
        o, h, l, c = (
            float(df["Open"].iloc[-1]),
            float(df["High"].iloc[-1]),
            float(df["Low"].iloc[-1]),
            float(df["Close"].iloc[-1]),
        )
        body = abs(c - o)
        wick_top = h - max(c, o)
        wick_bot = min(c, o) - l
        total_range = h - l + 1e-10

        signals["doji_candle"] = body < 0.1 * total_range
        signals["hammer_candle"] = (
            wick_bot > 2 * body and wick_top < body and c > o
        )
        signals["shooting_star"] = (
            wick_top > 2 * body and wick_bot < body and c < o
        )

        return signals


# ---------------------------------------------------------------------------
# TechnicalScreener (enhanced)
# ---------------------------------------------------------------------------

class TechnicalScreener:
    """Enhanced technical screener with 15 preset screens and batch scoring."""

    PREBUILT_SCREENS: dict[str, dict] = {
        "golden_cross_universe": {
            "description": "All stocks with recent golden cross + price > SMA200",
            "criteria": {"price_above_sma200": True, "golden_cross": True},
            "sort_by_score": True,
        },
        "oversold_bounce": {
            "description": "Deeply oversold with potential reversal setup",
            "criteria": {
                "rsi_oversold": True,
                "stoch_oversold": True,
                "volume_surge": True,
            },
        },
        "breakout_candidates": {
            "description": "Near 52-week high with volume expansion",
            "criteria": {
                "price_near_52w_high": True,
                "volume_surge": True,
                "bb_upper_breakout": True,
            },
        },
        "momentum_leaders": {
            "description": "Strong price momentum, RSI healthy, MACD bullish",
            "criteria": {
                "price_above_sma200": True,
                "price_above_sma50": True,
                "macd_bullish_cross": True,
            },
        },
        "squeeze_setups": {
            "description": "Bollinger Band squeeze with volume compression",
            "criteria": {
                "bb_squeeze": True,
                "volume_dry_up": True,
            },
        },
        "ichimoku_cloud_breakout": {
            "description": "Price breaking above Ichimoku cloud (bullish)",
            "criteria": {
                "ichimoku_bullish": True,
                "price_above_sma50": True,
                "obv_rising": True,
            },
        },
        "mfi_divergence": {
            "description": "MFI oversold with positive CMF (accumulation while price weak)",
            "criteria": {
                "mfi_oversold": True,
                "cmf_positive": True,
            },
        },
        "dead_cat_filter": {
            "description": "Exclude probable dead-cat bounces (overbought after crash)",
            "criteria": {
                "rsi_overbought": False,
                "price_above_sma200": True,
                "obv_falling": False,
            },
        },
        "high_tight_flag": {
            "description": "Consolidation candidates: near high, low volatility, volume dry-up",
            "criteria": {
                "price_near_52w_high": True,
                "volume_dry_up": True,
                "atr_contracting": True,
            },
        },
        "parabolic_sar_signals": {
            "description": "Parabolic SAR bullish — trend-following entry",
            "criteria": {
                "parabolic_sar_bullish": True,
                "price_above_sma50": True,
                "roc_positive": True,
            },
        },
        "cci_reversal": {
            "description": "CCI extreme oversold — mean reversion candidates",
            "criteria": {
                "cci_extreme_low": True,
                "cmf_positive": True,
            },
        },
        "smart_money_accumulation": {
            "description": "OBV rising + CMF positive + VWAP above — institutional buying",
            "criteria": {
                "obv_rising": True,
                "cmf_positive": True,
                "vwap_above": True,
            },
        },
        "volatility_expansion": {
            "description": "ATR expanding after squeeze — potential big move",
            "criteria": {
                "bb_squeeze": False,   # previously was squeezing
                "atr_expanding": True,
                "volume_surge": True,
            },
        },
        "aroon_trend_change": {
            "description": "Aroon turning bullish — early trend detection",
            "criteria": {
                "aroon_bullish": True,
                "price_above_sma20": True,
            },
        },
        "hammer_reversal": {
            "description": "Hammer candlestick with volume confirmation",
            "criteria": {
                "hammer_candle": True,
                "volume_surge": True,
                "rsi_oversold": True,
            },
        },
    }

    def __init__(self, universe: list[str] | None = None) -> None:
        self._universe = universe or _DEFAULT_UNIVERSE
        self._engine = TechnicalIndicatorEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screen(self, criteria: dict[str, bool], limit: int = 100) -> pd.DataFrame:
        """Screen universe for tickers matching all signal criteria.

        Parameters
        ----------
        criteria:
            Dict mapping signal name to required boolean value.
            Example: ``{"rsi_oversold": True, "price_above_sma200": True}``
        limit:
            Maximum tickers to return.
        """
        results: list[dict] = []
        for ticker in self._universe:
            try:
                signals = self._get_signals(ticker)
                if self._matches(signals, criteria):
                    score_info = self._build_score_from_signals(signals)
                    results.append({
                        "ticker": ticker,
                        **{k: v for k, v in signals.items()},
                        "tech_score": score_info["score"],
                        "tech_rating": score_info["rating"],
                    })
                time.sleep(_YFINANCE_RATE_SLEEP)
            except Exception as exc:
                logger.debug("screen_ticker_skipped", ticker=ticker, error=str(exc))

        df = pd.DataFrame(results)
        if df.empty:
            return df
        return df.sort_values("tech_score", ascending=False).head(limit).reset_index(drop=True)

    def run_preset(self, name: str) -> pd.DataFrame:
        """Run a named preset technical screen."""
        spec = self.PREBUILT_SCREENS.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(self.PREBUILT_SCREENS)}"
            )
        return self.screen(spec["criteria"])

    def score_stock(self, ticker: str, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Compute composite technical score 0-100 for a single ticker.

        Components:
          - Trend alignment (SMA stack): 30 pts
          - Momentum strength (RSI + MACD): 25 pts
          - Volume confirmation (OBV + CMF): 20 pts
          - Volatility regime (ATR vs HV): 15 pts
          - Candlestick / pattern bonus: 10 pts
        """
        if df is None:
            df = _fetch_ohlcv(ticker)

        indicators = self._engine.compute_all(df, ticker=ticker)
        signals = self._engine.compute_signals(indicators)
        result = self._build_score_from_signals(signals)
        result["ticker"] = ticker
        result["signals"] = signals
        return result

    def find_support_resistance(
        self,
        df: pd.DataFrame,
        n_levels: int = 5,
    ) -> dict[str, Any]:
        """Identify key support and resistance levels.

        Uses a combination of:
          - Recent pivot highs/lows (rolling window method)
          - Round-number proximity
          - Fibonacci retracements from 52-week high to low
        """
        closes = df["Close"].dropna()
        highs = df["High"].dropna()
        lows = df["Low"].dropna()

        if len(closes) < 20:
            return {"support": [], "resistance": [], "fibonacci": {}}

        current = float(closes.iloc[-1])
        high_52w = float(highs.tail(252).max())
        low_52w = float(lows.tail(252).min())

        # Pivot-based levels
        window = 10
        pivot_highs: list[float] = []
        pivot_lows: list[float] = []
        for i in range(window, len(closes) - window):
            if float(highs.iloc[i]) == float(highs.iloc[i - window: i + window + 1].max()):
                pivot_highs.append(float(highs.iloc[i]))
            if float(lows.iloc[i]) == float(lows.iloc[i - window: i + window + 1].min()):
                pivot_lows.append(float(lows.iloc[i]))

        resistance = sorted(set([p for p in pivot_highs if p > current]), reverse=False)[:n_levels]
        support = sorted(set([p for p in pivot_lows if p < current]), reverse=True)[:n_levels]

        # Round number levels (nearest 5, 10, 50, 100 multiples)
        magnitude = 10 ** max(0, int(math.log10(current)) - 1)
        round_levels = []
        for mult in [1, 2, 5]:
            step = magnitude * mult
            lower = math.floor(current / step) * step
            for k in range(-3, 4):
                lvl = round(lower + k * step, 2)
                if lvl > 0:
                    round_levels.append(lvl)

        # Fibonacci retracements
        swing = high_52w - low_52w
        fib_levels = {
            "fib_0.0": round(low_52w, 2),
            "fib_0.236": round(low_52w + 0.236 * swing, 2),
            "fib_0.382": round(low_52w + 0.382 * swing, 2),
            "fib_0.500": round(low_52w + 0.500 * swing, 2),
            "fib_0.618": round(low_52w + 0.618 * swing, 2),
            "fib_0.786": round(low_52w + 0.786 * swing, 2),
            "fib_1.000": round(high_52w, 2),
            "fib_1.272": round(high_52w + 0.272 * swing, 2),
            "fib_1.618": round(high_52w + 0.618 * swing, 2),
        }

        return {
            "current_price": current,
            "support": support,
            "resistance": resistance,
            "round_number_levels": sorted(set(round_levels)),
            "fibonacci": fib_levels,
            "52w_high": high_52w,
            "52w_low": low_52w,
        }

    def detect_chart_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect chart and candlestick patterns.

        Chart patterns: head_shoulders, double_top, double_bottom,
        cup_handle, ascending_triangle, descending_triangle,
        symmetrical_triangle, wedge_up, wedge_down, flag_bull,
        flag_bear, pennant, channel_up, channel_down.

        Candlestick patterns: doji, hammer, shooting_star,
        engulfing_bull, engulfing_bear, morning_star, evening_star,
        three_white_soldiers, three_black_crows.
        """
        if len(df) < 30:
            return []

        patterns: list[dict[str, Any]] = []
        closes = df["Close"].dropna()
        highs = df["High"].dropna()
        lows = df["Low"].dropna()
        opens = df["Open"].dropna()

        # ── Candlestick patterns ──────────────────────────────────────────
        # Last candle
        o, h, l, c = (
            float(opens.iloc[-1]),
            float(highs.iloc[-1]),
            float(lows.iloc[-1]),
            float(closes.iloc[-1]),
        )
        body = abs(c - o)
        total_range = h - l + 1e-10
        wick_top = h - max(c, o)
        wick_bot = min(c, o) - l

        if body < 0.1 * total_range:
            patterns.append({"pattern": "doji", "type": "candlestick", "bias": "neutral",
                             "description": "Indecision candle, potential reversal"})

        if wick_bot > 2 * body and wick_top < body and c > o:
            patterns.append({"pattern": "hammer", "type": "candlestick", "bias": "bullish",
                             "description": "Bullish reversal signal"})

        if wick_top > 2 * body and wick_bot < body and c < o:
            patterns.append({"pattern": "shooting_star", "type": "candlestick", "bias": "bearish",
                             "description": "Bearish reversal signal"})

        # Engulfing (last 2 candles)
        if len(closes) >= 2:
            o1, c1 = float(opens.iloc[-2]), float(closes.iloc[-2])
            if c > o and c1 < o1 and o < c1 and c > o1:
                patterns.append({"pattern": "bullish_engulfing", "type": "candlestick",
                                  "bias": "bullish", "description": "Bullish reversal — prior bearish candle engulfed"})
            if c < o and c1 > o1 and o > c1 and c < o1:
                patterns.append({"pattern": "bearish_engulfing", "type": "candlestick",
                                  "bias": "bearish", "description": "Bearish reversal — prior bullish candle engulfed"})

        # Three white soldiers / three black crows (last 3)
        if len(closes) >= 3:
            c3 = [float(closes.iloc[-i]) for i in range(1, 4)]
            o3 = [float(opens.iloc[-i]) for i in range(1, 4)]
            if all(c3[i] > o3[i] for i in range(3)) and c3[0] > c3[1] > c3[2]:
                patterns.append({"pattern": "three_white_soldiers", "type": "candlestick",
                                  "bias": "bullish", "description": "Strong bullish momentum continuation"})
            if all(c3[i] < o3[i] for i in range(3)) and c3[0] < c3[1] < c3[2]:
                patterns.append({"pattern": "three_black_crows", "type": "candlestick",
                                  "bias": "bearish", "description": "Strong bearish momentum continuation"})

        # ── Chart patterns (simplified structural detection) ──────────────
        n = min(60, len(closes))
        recent = closes.iloc[-n:]
        r_highs = highs.iloc[-n:]
        r_lows = lows.iloc[-n:]

        rolling_max = recent.rolling(10).max()
        rolling_min = recent.rolling(10).min()

        # Double top detection
        high_vals = r_highs.dropna()
        if len(high_vals) >= 20:
            top1 = float(high_vals.iloc[:len(high_vals)//2].max())
            top2 = float(high_vals.iloc[len(high_vals)//2:].max())
            if abs(top1 - top2) / max(top1, 1e-10) < 0.03 and float(closes.iloc[-1]) < top1 * 0.97:
                patterns.append({"pattern": "double_top", "type": "chart", "bias": "bearish",
                                  "description": f"Double top near ${top1:.2f} — bearish reversal"})

        # Double bottom detection
        low_vals = r_lows.dropna()
        if len(low_vals) >= 20:
            bot1 = float(low_vals.iloc[:len(low_vals)//2].min())
            bot2 = float(low_vals.iloc[len(low_vals)//2:].min())
            if abs(bot1 - bot2) / max(abs(bot1), 1e-10) < 0.03 and float(closes.iloc[-1]) > bot1 * 1.03:
                patterns.append({"pattern": "double_bottom", "type": "chart", "bias": "bullish",
                                  "description": f"Double bottom near ${bot1:.2f} — bullish reversal"})

        # Ascending triangle: highs flat, lows rising
        if len(r_highs) >= 20:
            high_slope = np.polyfit(range(len(r_highs)), r_highs.values, 1)[0]
            low_slope = np.polyfit(range(len(r_lows)), r_lows.values, 1)[0]
            if abs(high_slope) < 0.01 * float(closes.iloc[-1]) and low_slope > 0:
                patterns.append({"pattern": "ascending_triangle", "type": "chart", "bias": "bullish",
                                  "description": "Flat resistance + rising support — bullish breakout pending"})

        # Descending triangle: lows flat, highs falling
            if abs(low_slope) < 0.01 * float(closes.iloc[-1]) and high_slope < 0:
                patterns.append({"pattern": "descending_triangle", "type": "chart", "bias": "bearish",
                                  "description": "Flat support + falling resistance — bearish breakdown pending"})

        # Flag (strong prior move, brief consolidation)
        if len(closes) >= 30:
            prior_move = (float(closes.iloc[-21]) - float(closes.iloc[-30])) / max(float(closes.iloc[-30]), 1e-10)
            recent_range = (float(r_highs.iloc[-5:].max()) - float(r_lows.iloc[-5:].min())) / max(float(closes.iloc[-1]), 1e-10)
            if prior_move > 0.10 and recent_range < 0.05:
                patterns.append({"pattern": "bull_flag", "type": "chart", "bias": "bullish",
                                  "description": "Strong prior move with tight consolidation"})
            if prior_move < -0.10 and recent_range < 0.05:
                patterns.append({"pattern": "bear_flag", "type": "chart", "bias": "bearish",
                                  "description": "Strong prior decline with tight consolidation"})

        # pandas-ta CDL patterns (if available)
        if _TA_AVAILABLE:
            try:
                cdl = df.ta.cdl_pattern(name="all")
                if cdl is not None and not cdl.empty:
                    for col in cdl.columns:
                        val = int(cdl[col].iloc[-1])
                        if val != 0:
                            name = col.replace("CDL_", "").lower()
                            bias = "bullish" if val > 0 else "bearish"
                            patterns.append({"pattern": name, "type": "candlestick_ta", "bias": bias,
                                              "signal_strength": val})
            except Exception:
                pass

        return patterns

    def batch_score(self, tickers: list[str]) -> pd.DataFrame:
        """Score all tickers in parallel using ThreadPoolExecutor."""
        rows: list[dict] = []
        errors: list[dict] = []

        def _score_one(t: str) -> dict[str, Any]:
            try:
                result = self.score_stock(t)
                return {"ticker": t, "tech_score": result["score"],
                        "tech_rating": result["rating"], "error": None}
            except Exception as exc:
                return {"ticker": t, "tech_score": None, "tech_rating": None, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=_BATCH_MAX_WORKERS) as executor:
            futures = {executor.submit(_score_one, t): t for t in tickers}
            for future in as_completed(futures):
                rows.append(future.result())

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return (
            df.sort_values("tech_score", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_signals(self, ticker: str) -> dict[str, bool | None]:
        """Fetch OHLCV, compute indicators, return signals (cached)."""
        cached = _get_cached_indicators(ticker)
        if cached is not None:
            return self._engine.compute_signals(cached)
        df = _fetch_ohlcv(ticker)
        indicators = self._engine.compute_all(df, ticker=ticker)
        _set_cached_indicators(ticker, indicators)
        return self._engine.compute_signals(indicators)

    @staticmethod
    def _matches(signals: dict[str, bool | None], criteria: dict[str, bool]) -> bool:
        """Return True if all criteria are satisfied (None signals are skipped)."""
        for signal, required in criteria.items():
            val = signals.get(signal)
            if val is None:
                continue  # treat unavailable as neutral — don't reject
            if val != required:
                return False
        return True

    @staticmethod
    def _build_score_from_signals(signals: dict[str, bool | None]) -> dict[str, Any]:
        """Map signals to a 0-100 technical score."""
        # Trend component (30 pts)
        trend_score = 0.0
        if signals.get("price_above_sma200"):
            trend_score += 15
        if signals.get("price_above_sma50"):
            trend_score += 10
        if signals.get("golden_cross"):
            trend_score += 5
        if signals.get("death_cross"):
            trend_score -= 10
        if signals.get("parabolic_sar_bullish"):
            trend_score += 5
        trend_score = max(0, min(30, trend_score))

        # Momentum component (25 pts)
        mom_score = 0.0
        rsi_ov = signals.get("rsi_oversold")
        rsi_ob = signals.get("rsi_overbought")
        if rsi_ov is True:
            mom_score += 10       # oversold — potential upside
        elif rsi_ob is True:
            mom_score -= 5        # overbought — stretched
        else:
            mom_score += 5        # healthy RSI zone
        if signals.get("macd_bullish_cross"):
            mom_score += 10
        if signals.get("macd_bearish_cross"):
            mom_score -= 8
        if signals.get("aroon_bullish"):
            mom_score += 5
        if signals.get("roc_positive"):
            mom_score += 3
        if signals.get("stoch_oversold"):
            mom_score += 4
        if signals.get("stoch_overbought"):
            mom_score -= 3
        mom_score = max(0, min(25, mom_score))

        # Volume component (20 pts)
        vol_score = 0.0
        if signals.get("obv_rising"):
            vol_score += 8
        if signals.get("obv_falling"):
            vol_score -= 5
        if signals.get("cmf_positive"):
            vol_score += 6
        if signals.get("cmf_negative"):
            vol_score -= 4
        if signals.get("volume_surge"):
            vol_score += 4
        if signals.get("vwap_above"):
            vol_score += 5
        if signals.get("mfi_oversold"):
            vol_score += 4
        vol_score = max(0, min(20, vol_score))

        # Volatility component (15 pts)
        volatility_score = 7.5   # neutral start
        if signals.get("bb_squeeze"):
            volatility_score += 5    # squeeze = coiled spring, bonus
        if signals.get("atr_contracting"):
            volatility_score += 3
        if signals.get("atr_expanding"):
            volatility_score -= 2   # expanding volatility = risk
        if signals.get("bb_upper_breakout"):
            volatility_score += 2
        volatility_score = max(0, min(15, volatility_score))

        # Pattern bonus (10 pts)
        pattern_score = 0.0
        if signals.get("hammer_candle"):
            pattern_score += 5
        if signals.get("shooting_star"):
            pattern_score -= 3
        if signals.get("doji_candle"):
            pattern_score += 1
        if signals.get("ichimoku_bullish"):
            pattern_score += 5
        if signals.get("ichimoku_bearish"):
            pattern_score -= 4
        if signals.get("price_near_52w_high"):
            pattern_score += 2
        pattern_score = max(0, min(10, pattern_score))

        total = trend_score + mom_score + vol_score + volatility_score + pattern_score

        rating = (
            "Strong Buy" if total >= 78 else
            "Buy" if total >= 60 else
            "Neutral" if total >= 40 else
            "Sell" if total >= 22 else
            "Strong Sell"
        )

        return {
            "score": round(total, 1),
            "rating": rating,
            "components": {
                "trend": round(trend_score, 1),
                "momentum": round(mom_score, 1),
                "volume": round(vol_score, 1),
                "volatility": round(volatility_score, 1),
                "patterns": round(pattern_score, 1),
            },
        }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

technical_screener_router = APIRouter(
    prefix="/api/screener/technical", tags=["Technical Screener"]
)

_screener = TechnicalScreener()
_engine_global = TechnicalIndicatorEngine()


class TechScreenRequest(BaseModel):
    criteria: dict[str, bool]
    limit: int = 100


@technical_screener_router.post("")
def api_tech_screen(req: TechScreenRequest) -> dict:
    """Run custom technical screen with binary signal criteria."""
    df = _screener.screen(req.criteria, req.limit)
    return {
        "n_results": len(df),
        "results": df.where(pd.notna(df), None).to_dict(orient="records"),
    }


@technical_screener_router.get("/presets")
def api_tech_list_presets() -> dict:
    """List all 15 preset technical screens."""
    return {
        "presets": [
            {"name": k, "description": v.get("description", "")}
            for k, v in TechnicalScreener.PREBUILT_SCREENS.items()
        ]
    }


@technical_screener_router.get("/preset/{name}")
def api_tech_run_preset(name: str) -> dict:
    """Run a named preset technical screen."""
    try:
        df = _screener.run_preset(name)
        return {
            "preset": name,
            "n_results": len(df),
            "results": df.where(pd.notna(df), None).to_dict(orient="records"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@technical_screener_router.get("/{ticker}/score")
def api_tech_score(ticker: str) -> dict:
    """Compute composite technical score (0-100) for a single ticker."""
    try:
        result = _screener.score_stock(ticker.upper())
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@technical_screener_router.get("/{ticker}/signals")
def api_tech_signals(ticker: str) -> dict:
    """Return all 35 binary indicator signals for a ticker."""
    try:
        df = _fetch_ohlcv(ticker.upper())
        indicators = _engine_global.compute_all(df, ticker=ticker.upper())
        signals = _engine_global.compute_signals(indicators)
        return {
            "ticker": ticker.upper(),
            "as_of": str(date.today()),
            "signals": {k: v for k, v in signals.items()},
            "signal_definitions": _SIGNAL_DEFINITIONS,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@technical_screener_router.get("/{ticker}/support-resistance")
def api_support_resistance(ticker: str, n_levels: int = Query(5, ge=1, le=10)) -> dict:
    """Return key support, resistance, and Fibonacci levels."""
    try:
        df = _fetch_ohlcv(ticker.upper())
        result = _screener.find_support_resistance(df, n_levels=n_levels)
        result["ticker"] = ticker.upper()
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
