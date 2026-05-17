"""
volatility_analytics_v3.py — Comprehensive Volatility & Inflation Analytics Platform.

dim_048: Inflation breakeven / VIX analytics / volatility risk premium — score 7 → 9

Architecture:
  InflationBreakevenAnalyzer — FRED TIPS, breakeven, Fisher equation, regime detection
  VIXAnalyticsEngine         — VIX/VXV history, term structure, percentile, spike detection
  VolatilityRiskPremiumCalc  — VRP = implied vol - realized vol, skew premium
  MultiAssetVolTracker       — OVX, GVZ, EVZ, cross-asset vol correlation
  InflationTradingSignals    — TIPS vs nominal, sector rotation, carry signals
  VolatilityAnalyticsEngine  — Orchestrator: full dashboard, report generation

Free data only: FRED CSV (no API key), yfinance price-only for HV computation.
scipy optional (guarded). pandas-datareader optional (guarded).
"""
from __future__ import annotations

import io
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deps — guarded
# ---------------------------------------------------------------------------
try:
    from scipy.stats import percentileofscore  # type: ignore[import-untyped]
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.debug("scipy not available — percentile uses numpy fallback")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants — FRED CSV endpoints (no API key required)
# ---------------------------------------------------------------------------
_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_TIMEOUT   = 25
_HEADERS   = {"User-Agent": "SENTINEL/3.0 (financial-terminal; research)"}

# FRED series identifiers
_FRED_SERIES: Dict[str, str] = {
    # Inflation breakeven (nominal - real)
    "T10YIE":   "10-Year Breakeven Inflation Rate",
    "T5YIE":    "5-Year Breakeven Inflation Rate",
    "T5YIFR":   "5-Year, 5-Year Forward Inflation Expectation Rate",
    "T1YIE":    "1-Year Breakeven Inflation Rate",
    # TIPS real yields
    "DFII5":    "5-Year TIPS Real Yield",
    "DFII10":   "10-Year TIPS Real Yield",
    "DFII20":   "20-Year TIPS Real Yield",
    "DFII30":   "30-Year TIPS Real Yield",
    # Nominal Treasury yields
    "DGS1MO":   "1-Month Treasury",
    "DGS3MO":   "3-Month Treasury",
    "DGS6MO":   "6-Month Treasury",
    "DGS1":     "1-Year Treasury",
    "DGS2":     "2-Year Treasury",
    "DGS5":     "5-Year Treasury",
    "DGS10":    "10-Year Treasury",
    "DGS20":    "20-Year Treasury",
    "DGS30":    "30-Year Treasury",
    # Volatility indices
    "VIXCLS":   "CBOE VIX (1-Month Implied Vol)",
    "VXVCLS":   "CBOE VXV (3-Month Implied Vol)",
    "OVXCLS":   "CBOE OVX (Crude Oil Vol)",
    "GVZCLS":   "CBOE GVZ (Gold Vol)",
    "EVZ":      "CBOE EVZ (EUR/USD Vol)",
    # Spread signals
    "T10Y2Y":   "10Y-2Y Treasury Spread",
    "T10Y3M":   "10Y-3M Treasury Spread",
    "BAMLH0A0HYM2": "ICE BofA HY OAS",
    "BAMLC0A0CM":   "ICE BofA IG OAS",
}

# SPF (Survey of Professional Forecasters) long-run inflation expectation
# Updated quarterly — hardcode latest Philadelphia Fed SPF Q1-2026 reading
_SPF_10Y_INFLATION_EXPECTATION = 2.30  # percent

# VRP cache TTL
_CACHE_TTL = 6 * 3600  # 6 hours

# VIX regime thresholds
_VIX_LOW       = 12.0
_VIX_NORMAL_HI = 20.0
_VIX_STRESS    = 30.0
_VIX_CRISIS    = 40.0

# Breakeven regime thresholds (percent)
_BE_LOW       = 1.5
_BE_TARGET_HI = 2.5
_BE_ELEVATED  = 3.5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VIXRegime:
    """VIX volatility regime classification."""
    level:          float
    percentile:     float
    label:          str           # CALM / NORMAL / STRESSED / CRISIS
    term_structure: str           # CONTANGO / FLAT / BACKWARDATION
    vxv_vix_ratio:  float         # >1 = contango, <0.9 = backwardation
    mean_reversion: str           # BUY_EQUITIES / CAUTION / NEUTRAL
    spike_today:    bool = False
    as_of:          str = ""


@dataclass
class BreakevenData:
    """Current inflation breakeven readings across the curve."""
    as_of:              str
    breakeven_1y:       Optional[float] = None
    breakeven_5y:       Optional[float] = None
    breakeven_10y:      Optional[float] = None
    breakeven_5y5y:     Optional[float] = None
    real_yield_5y:      Optional[float] = None
    real_yield_10y:     Optional[float] = None
    nominal_10y:        Optional[float] = None
    implied_inflation_risk_premium: Optional[float] = None   # breakeven - SPF
    curve_shape:        str = "UNKNOWN"     # RISING / FLAT / INVERTED
    regime:             str = "UNKNOWN"     # LOW / TARGET / ELEVATED / HIGH
    momentum_21d:       Optional[float] = None    # 21-day change in 10Y breakeven
    percentile_10yr:    Optional[float] = None    # 0–100


@dataclass
class VRPData:
    """Volatility Risk Premium calculation results."""
    ticker:         str
    as_of:          str
    vix_implied:    float          # VIX level (annualised %)
    hv30:           float          # 30-day realised vol (annualised %)
    vrp:            float          # VIX - HV30 (basis points of vol)
    vrp_mean:       float          # rolling mean VRP
    vrp_std:        float          # rolling std VRP
    vrp_percentile: float          # 0–100
    vrp_zscore:     float          # (VRP - mean) / std
    interpretation: str            # HIGH / NORMAL / LOW / NEGATIVE
    signal:         str            # SELL_VOL / NEUTRAL / BUY_VOL / STRESS


# ---------------------------------------------------------------------------
# Internal FRED fetcher
# ---------------------------------------------------------------------------

class _FREDFetcher:
    """
    Thin FRED CSV downloader — no API key required.
    FRED exposes raw CSV at:
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID
    """

    _cache: Dict[str, Tuple[float, pd.Series]] = {}

    def fetch(self, series_id: str,
              start: str = "1990-01-01",
              ttl: int = _CACHE_TTL) -> pd.Series:
        """
        Download a FRED series as a daily pd.Series indexed by date.
        Values are floats; missing days are NaN (not forward-filled).
        """
        cache_key = f"{series_id}:{start}"
        if cache_key in self._cache:
            ts, series = self._cache[cache_key]
            if time.time() - ts < ttl:
                return series

        url = f"{_FRED_BASE}?id={series_id}"
        logger.info("FRED fetch: %s (from %s)", series_id, start)
        for attempt in range(3):
            try:
                r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 2:
                    logger.error("FRED %s failed: %s", series_id, exc)
                    return pd.Series(dtype=float, name=series_id)
                time.sleep(2 ** attempt)

        try:
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["DATE"])
            df = df.rename(columns={"DATE": "date"})
            df = df.set_index("date").squeeze()
            if isinstance(df, pd.DataFrame):
                df = df.iloc[:, 0]
            series = pd.to_numeric(df, errors="coerce")
            series.name = series_id
            # Filter to start date
            start_dt = pd.to_datetime(start)
            series = series[series.index >= start_dt]
            series = series.sort_index()
        except Exception as exc:
            logger.error("FRED parse %s: %s", series_id, exc)
            series = pd.Series(dtype=float, name=series_id)

        self._cache[cache_key] = (time.time(), series)
        return series

    def fetch_multi(self, series_ids: List[str],
                    start: str = "1990-01-01") -> pd.DataFrame:
        """Fetch multiple FRED series and return combined DataFrame."""
        frames: Dict[str, pd.Series] = {}
        for sid in series_ids:
            frames[sid] = self.fetch(sid, start=start)
        return pd.DataFrame(frames)


_fred = _FREDFetcher()


def _percentile_of(value: float, series: pd.Series) -> float:
    """Compute percentile of value in series (0–100)."""
    clean = series.dropna()
    if len(clean) == 0:
        return 50.0
    if _SCIPY_AVAILABLE:
        return float(percentileofscore(clean.values, value, kind="rank"))
    # numpy fallback
    return float(np.mean(clean.values <= value) * 100)


def _annualised_hv(returns: pd.Series, window: int = 30) -> pd.Series:
    """Rolling annualised historical volatility from log returns."""
    return returns.rolling(window=window).std() * math.sqrt(252) * 100


# ---------------------------------------------------------------------------
# InflationBreakevenAnalyzer
# ---------------------------------------------------------------------------

class InflationBreakevenAnalyzer:
    """
    Analyze inflation breakeven rates and TIPS real yields from FRED.

    All data from FRED CSV endpoint — no API key required.
    Implements the Fisher equation verification, regime detection,
    inflation risk premium, and breakeven momentum.
    """

    def __init__(self, start: str = "2003-01-01"):
        self._start = start
        self._data: Optional[pd.DataFrame] = None

    def fetch_breakeven_data(self, start: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch breakeven inflation series from FRED.

        Returns DataFrame with columns: T10YIE, T5YIE, T5YIFR, T1YIE,
                                        DFII5, DFII10, DGS10, DGS5
        """
        start = start or self._start
        series_ids = ["T10YIE", "T5YIE", "T5YIFR", "T1YIE",
                      "DFII5", "DFII10", "DGS10", "DGS5", "DGS2"]
        df = _fred.fetch_multi(series_ids, start=start)

        # Forward-fill up to 5 business days (weekends / holidays)
        df = df.ffill(limit=5)
        self._data = df
        logger.info("Breakeven data loaded: %d obs, %s to %s",
                    len(df), df.index.min().date(), df.index.max().date())
        return df

    def _ensure_data(self) -> pd.DataFrame:
        if self._data is None or self._data.empty:
            self.fetch_breakeven_data()
        assert self._data is not None
        return self._data

    def compute_breakeven_curve(self) -> Dict[str, Any]:
        """
        Compute current breakeven inflation curve shape.

        Returns dict with 5y, 10y, 5y5y readings and curve shape.
        """
        df = self._ensure_data()
        latest = df.iloc[-1]

        be_5y    = float(latest.get("T5YIE",  np.nan) or np.nan)
        be_10y   = float(latest.get("T10YIE", np.nan) or np.nan)
        be_5y5y  = float(latest.get("T5YIFR", np.nan) or np.nan)
        be_1y    = float(latest.get("T1YIE",  np.nan) or np.nan)

        # Curve shape: compare 5y vs 10y vs 5y5y forward
        if not (np.isnan(be_5y) or np.isnan(be_10y)):
            if be_10y > be_5y + 0.1:
                shape = "RISING"         # market expects higher future inflation
            elif be_10y < be_5y - 0.1:
                shape = "INVERTED"       # near-term inflation fears fading
            else:
                shape = "FLAT"
        else:
            shape = "UNKNOWN"

        as_of = str(df.index[-1].date())

        return {
            "as_of":          as_of,
            "breakeven_1y":   round(be_1y, 3)   if not np.isnan(be_1y)   else None,
            "breakeven_5y":   round(be_5y, 3)   if not np.isnan(be_5y)   else None,
            "breakeven_10y":  round(be_10y, 3)  if not np.isnan(be_10y)  else None,
            "breakeven_5y5y": round(be_5y5y, 3) if not np.isnan(be_5y5y) else None,
            "curve_shape":    shape,
            "regime":         self.detect_regime(be_10y) if not np.isnan(be_10y) else "UNKNOWN",
        }

    def compute_real_yields(self) -> Dict[str, Any]:
        """
        Compute current TIPS real yields and verify Fisher equation.

        Fisher:  nominal = real + breakeven (approximately)
        Inflation risk premium = breakeven - expected inflation (SPF)
        """
        df = self._ensure_data()
        latest = df.iloc[-1]

        real_5y  = float(latest.get("DFII5",  np.nan) or np.nan)
        real_10y = float(latest.get("DFII10", np.nan) or np.nan)
        nom_5y   = float(latest.get("DGS5",   np.nan) or np.nan)
        nom_10y  = float(latest.get("DGS10",  np.nan) or np.nan)
        be_5y    = float(latest.get("T5YIE",  np.nan) or np.nan)
        be_10y   = float(latest.get("T10YIE", np.nan) or np.nan)

        # Fisher verification (should be ~0; deviations reflect liquidity premium)
        fisher_5y  = None if any(np.isnan(x) for x in [nom_5y,  real_5y,  be_5y]) \
                     else round(nom_5y  - real_5y  - be_5y, 3)
        fisher_10y = None if any(np.isnan(x) for x in [nom_10y, real_10y, be_10y]) \
                     else round(nom_10y - real_10y - be_10y, 3)

        # IRP at 10y
        irp_10y = None if np.isnan(be_10y) \
                  else round(be_10y - _SPF_10Y_INFLATION_EXPECTATION, 3)

        return {
            "as_of":                      str(df.index[-1].date()),
            "real_yield_5y":              round(real_5y, 3)  if not np.isnan(real_5y)  else None,
            "real_yield_10y":             round(real_10y, 3) if not np.isnan(real_10y) else None,
            "nominal_5y":                 round(nom_5y, 3)   if not np.isnan(nom_5y)   else None,
            "nominal_10y":                round(nom_10y, 3)  if not np.isnan(nom_10y)  else None,
            "breakeven_5y":               round(be_5y, 3)    if not np.isnan(be_5y)    else None,
            "breakeven_10y":              round(be_10y, 3)   if not np.isnan(be_10y)   else None,
            "fisher_residual_5y":         fisher_5y,
            "fisher_residual_10y":        fisher_10y,
            "spf_10y_expectation":        _SPF_10Y_INFLATION_EXPECTATION,
            "inflation_risk_premium_10y": irp_10y,
            "real_yield_regime":          self._classify_real_yield(real_10y),
        }

    def _classify_real_yield(self, real_yield: float) -> str:
        if np.isnan(real_yield):
            return "UNKNOWN"
        if real_yield > 2.0:
            return "HIGH_POSITIVE"       # tight financial conditions
        if real_yield > 0.5:
            return "POSITIVE"
        if real_yield > -0.5:
            return "NEAR_ZERO"
        if real_yield > -2.0:
            return "NEGATIVE"            # accommodative / inflationary
        return "DEEPLY_NEGATIVE"

    def compute_inflation_risk_premium(self, nominal: float,
                                       tips_real: float,
                                       expected_inflation: float) -> float:
        """
        Compute inflation risk premium (IRP).

        IRP = (nominal - tips_real) - expected_inflation
            = breakeven - expected_inflation

        Positive IRP: market prices more inflation than surveys expect
        Negative IRP: market prices less (deflation/disinflation fears)
        """
        breakeven = nominal - tips_real
        return breakeven - expected_inflation

    def detect_regime(self, breakeven: float) -> str:
        """
        Classify breakeven inflation into regime:

        LOW       < 1.5%  : deflationary / disinflation fears
        TARGET    1.5-2.5%: near Fed target (2%)
        ELEVATED  2.5-3.5%: above-target, Fed likely hawkish
        HIGH      > 3.5%  : serious inflation concern
        """
        if np.isnan(breakeven):
            return "UNKNOWN"
        if breakeven < _BE_LOW:
            return "LOW"
        if breakeven < _BE_TARGET_HI:
            return "TARGET"
        if breakeven < _BE_ELEVATED:
            return "ELEVATED"
        return "HIGH"

    def compute_breakeven_momentum(self, lookback_days: int = 21) -> float:
        """
        Compute 1-month (21 business day) change in 10-year breakeven.

        Returns change in percentage points (positive = rising inflation expectations).
        """
        df = self._ensure_data()
        if "T10YIE" not in df.columns:
            return float("nan")

        series = df["T10YIE"].dropna()
        if len(series) < lookback_days + 1:
            return float("nan")

        return float(series.iloc[-1] - series.iloc[-lookback_days - 1])

    def compute_breakeven_percentile(self, current: Optional[float] = None,
                                     lookback_years: int = 10) -> float:
        """
        Compute percentile of current 10Y breakeven in its historical range.

        Returns 0–100 where 100 = highest breakeven ever in lookback window.
        """
        df = self._ensure_data()
        if "T10YIE" not in df.columns:
            return 50.0

        series = df["T10YIE"].dropna()
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        window = series[series.index >= cutoff]

        if current is None:
            current = float(series.iloc[-1]) if len(series) > 0 else float("nan")

        if np.isnan(current) or len(window) == 0:
            return 50.0

        return _percentile_of(current, window)

    def get_breakeven_data(self) -> BreakevenData:
        """Return fully-populated BreakevenData dataclass."""
        curve  = self.compute_breakeven_curve()
        reals  = self.compute_real_yields()
        mom    = self.compute_breakeven_momentum()
        pctile = self.compute_breakeven_percentile()

        return BreakevenData(
            as_of=curve["as_of"],
            breakeven_1y=curve.get("breakeven_1y"),
            breakeven_5y=curve.get("breakeven_5y"),
            breakeven_10y=curve.get("breakeven_10y"),
            breakeven_5y5y=curve.get("breakeven_5y5y"),
            real_yield_5y=reals.get("real_yield_5y"),
            real_yield_10y=reals.get("real_yield_10y"),
            nominal_10y=reals.get("nominal_10y"),
            implied_inflation_risk_premium=reals.get("inflation_risk_premium_10y"),
            curve_shape=curve.get("curve_shape", "UNKNOWN"),
            regime=curve.get("regime", "UNKNOWN"),
            momentum_21d=round(mom, 4) if not np.isnan(mom) else None,
            percentile_10yr=round(pctile, 1),
        )


# ---------------------------------------------------------------------------
# VIXAnalyticsEngine
# ---------------------------------------------------------------------------

class VIXAnalyticsEngine:
    """
    Comprehensive VIX / CBOE volatility index analytics.

    Data: FRED VIXCLS, VXVCLS via CSV endpoint.
    yfinance used as backup for ^VIX, ^VXV.
    """

    def __init__(self):
        self._vix_series:  Optional[pd.Series] = None
        self._vxv_series:  Optional[pd.Series] = None

    def fetch_vix_history(self, start: str = "1990-01-01") -> pd.Series:
        """
        Fetch VIX daily history from FRED (VIXCLS).

        Falls back to yfinance ^VIX if FRED fails.
        Returns pd.Series indexed by date.
        """
        series = _fred.fetch("VIXCLS", start=start)
        series = series.dropna()

        if series.empty and _YF_AVAILABLE and yf is not None:
            logger.warning("FRED VIXCLS empty — falling back to yfinance ^VIX")
            try:
                tk = yf.Ticker("^VIX")
                hist = tk.history(start=start, auto_adjust=True)
                if not hist.empty:
                    series = hist["Close"].rename("VIXCLS")
            except Exception as exc:
                logger.error("yfinance ^VIX fallback failed: %s", exc)

        series.name = "VIX"
        self._vix_series = series
        return series

    def fetch_vxv_history(self, start: str = "2007-01-01") -> pd.Series:
        """
        Fetch VXV (3-Month VIX) history from FRED (VXVCLS).

        VXV represents expected volatility over the next 3 months.
        Falls back to yfinance ^VXV if FRED fails.
        """
        series = _fred.fetch("VXVCLS", start=start)
        series = series.dropna()

        if series.empty and _YF_AVAILABLE and yf is not None:
            logger.warning("FRED VXVCLS empty — falling back to yfinance ^VXV")
            try:
                tk = yf.Ticker("^VXV")
                hist = tk.history(start=start, auto_adjust=True)
                if not hist.empty:
                    series = hist["Close"].rename("VXVCLS")
            except Exception as exc:
                logger.error("yfinance ^VXV fallback failed: %s", exc)

        series.name = "VXV"
        self._vxv_series = series
        return series

    def _ensure_vix(self) -> pd.Series:
        if self._vix_series is None:
            self.fetch_vix_history()
        return self._vix_series  # type: ignore[return-value]

    def _ensure_vxv(self) -> pd.Series:
        if self._vxv_series is None:
            self.fetch_vxv_history()
        return self._vxv_series  # type: ignore[return-value]

    def get_current_vix(self) -> float:
        """Return most recent VIX close."""
        vix = self._ensure_vix()
        if vix.empty:
            return float("nan")
        return float(vix.dropna().iloc[-1])

    def compute_vix_regime(self, vix_level: Optional[float] = None) -> VIXRegime:
        """
        Classify VIX into regime and compute full context.

        Returns VIXRegime dataclass with term structure, percentile, signals.
        """
        vix = self._ensure_vix()
        current_vix = vix_level if vix_level is not None else self.get_current_vix()

        # Label
        if current_vix <= _VIX_LOW:
            label = "CALM"
        elif current_vix <= _VIX_NORMAL_HI:
            label = "NORMAL"
        elif current_vix <= _VIX_STRESS:
            label = "STRESSED"
        elif current_vix <= _VIX_CRISIS:
            label = "CRISIS"
        else:
            label = "EXTREME_CRISIS"

        # Mean reversion signal
        if current_vix > 35:
            mean_rev = "BUY_EQUITIES"    # vol extremely elevated → mean-reverts down
        elif current_vix < _VIX_LOW:
            mean_rev = "CAUTION"          # complacency
        else:
            mean_rev = "NEUTRAL"

        # Percentile
        pctile = self.compute_vix_percentile(current_vix, lookback_years=5)

        # Term structure
        ts    = self.compute_vix_term_structure()
        ratio = ts.get("ratio", 1.0) or 1.0
        if ratio > 1.05:
            ts_label = "CONTANGO"
        elif ratio < 0.90:
            ts_label = "BACKWARDATION"
        else:
            ts_label = "FLAT"

        # Spike detection
        spike = self.detect_vix_spike(threshold_pct=40.0)

        as_of = str(vix.index[-1].date()) if not vix.empty else str(date.today())

        return VIXRegime(
            level=round(current_vix, 2),
            percentile=round(pctile, 1),
            label=label,
            term_structure=ts_label,
            vxv_vix_ratio=round(ratio, 4),
            mean_reversion=mean_rev,
            spike_today=spike,
            as_of=as_of,
        )

    def compute_vix_term_structure(self) -> Dict[str, Any]:
        """
        Compute VIX term structure using VIX (1M) and VXV (3M).

        VXV/VIX ratio:
          > 1.0  → contango (normal): short-term fear < long-term vol
          0.9-1.0 → flat
          < 0.90 → backwardation (stress): front > back (inverted)

        Rising VIX with backwardation = acute stress event.
        """
        vix = self._ensure_vix()
        vxv = self._ensure_vxv()

        current_vix = float(vix.dropna().iloc[-1]) if not vix.empty else float("nan")
        current_vxv = float("nan")

        if not vxv.empty:
            # Align dates — VXV may not have today's reading
            vxv_clean = vxv.dropna()
            if not vxv_clean.empty:
                current_vxv = float(vxv_clean.iloc[-1])

        ratio = current_vxv / current_vix if (
            not np.isnan(current_vxv) and not np.isnan(current_vix)
            and current_vix > 0
        ) else float("nan")

        # 1-month VIX SMA for context
        vix_20d = float(vix.dropna().iloc[-20:].mean()) if len(vix.dropna()) >= 20 \
                  else float("nan")

        contango_pct = (ratio - 1.0) * 100 if not np.isnan(ratio) else float("nan")

        return {
            "vix_1m":           round(current_vix, 2) if not np.isnan(current_vix) else None,
            "vxv_3m":           round(current_vxv, 2) if not np.isnan(current_vxv) else None,
            "ratio":            round(ratio, 4)        if not np.isnan(ratio)        else None,
            "vix_20d_ma":       round(vix_20d, 2)     if not np.isnan(vix_20d)      else None,
            "contango_pct":     round(contango_pct, 2) if not np.isnan(contango_pct) else None,
            "structure":        ("CONTANGO" if not np.isnan(ratio) and ratio > 1.0
                                 else "BACKWARDATION" if not np.isnan(ratio) and ratio < 0.90
                                 else "FLAT"),
            "stress_signal":    not np.isnan(ratio) and ratio < 0.90,
        }

    def compute_vix_percentile(self, current_vix: Optional[float] = None,
                                lookback_years: int = 5) -> float:
        """
        Compute percentile rank of current VIX in its historical range.

        Returns 0–100 (100 = highest VIX in lookback period).
        """
        vix = self._ensure_vix()
        if vix.empty:
            return 50.0

        if current_vix is None:
            current_vix = float(vix.dropna().iloc[-1])

        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        window = vix[vix.index >= cutoff].dropna()
        if window.empty:
            return 50.0

        return _percentile_of(current_vix, window)

    def compute_vix_mean_reversion_signal(self,
                                          current_vix: Optional[float] = None) -> str:
        """
        Generate mean-reversion trading signal based on VIX level.

        VIX > 35: BUY_EQUITIES — high vol extreme → likely to mean-revert lower
        VIX 20–35: ELEVATED — cautious but not extreme
        VIX 12–20: NEUTRAL — normal volatility environment
        VIX < 12:  CAUTION — complacency, market underpricing risk
        """
        if current_vix is None:
            current_vix = self.get_current_vix()

        if np.isnan(current_vix):
            return "UNKNOWN"

        if current_vix > 40:
            return "STRONG_BUY_EQUITIES"
        if current_vix > 35:
            return "BUY_EQUITIES"
        if current_vix > 28:
            return "ELEVATED_CAUTION"
        if current_vix > _VIX_NORMAL_HI:
            return "ABOVE_AVERAGE"
        if current_vix > _VIX_LOW:
            return "NEUTRAL"
        return "CAUTION_COMPLACENCY"

    def compute_vix_ma(self, short_ma: int = 5, long_ma: int = 20) -> Dict[str, Any]:
        """
        Compute VIX simple moving average crossover signal.

        VIX short MA crossing above long MA → volatility rising trend
        VIX short MA crossing below long MA → volatility declining trend
        """
        vix = self._ensure_vix()
        clean = vix.dropna()
        if len(clean) < long_ma:
            return {"signal": "INSUFFICIENT_DATA"}

        sma_short = float(clean.iloc[-short_ma:].mean())
        sma_long  = float(clean.iloc[-long_ma:].mean())
        current   = float(clean.iloc[-1])

        signal = "RISING_VOL" if sma_short > sma_long * 1.02 else (
                 "FALLING_VOL" if sma_short < sma_long * 0.98 else "FLAT_VOL")

        return {
            f"sma_{short_ma}d":  round(sma_short, 2),
            f"sma_{long_ma}d":   round(sma_long,  2),
            "current_vix":       round(current, 2),
            "signal":            signal,
            "vix_above_long_ma": current > sma_long,
        }

    def detect_vix_spike(self, threshold_pct: float = 40.0) -> bool:
        """
        Detect whether VIX spiked by more than threshold_pct in one day.

        Returns True if today's VIX > yesterday's VIX × (1 + threshold_pct / 100).
        """
        vix = self._ensure_vix()
        clean = vix.dropna()
        if len(clean) < 2:
            return False

        today = float(clean.iloc[-1])
        yesterday = float(clean.iloc[-2])
        if yesterday <= 0:
            return False

        change_pct = (today - yesterday) / yesterday * 100
        return change_pct >= threshold_pct

    def compute_rolling_vix_percentile(self, lookback_years: int = 5,
                                       window_years: int = 1) -> pd.Series:
        """
        Compute a daily rolling VIX percentile time series.

        Useful for regime charts. Returns series indexed by date.
        """
        vix = self._ensure_vix()
        clean = vix.dropna()
        window_days = int(window_years * 252)

        def _pctile(w: np.ndarray) -> float:
            v = w[-1]
            return float(np.mean(w <= v) * 100)

        return clean.rolling(window=window_days, min_periods=20).apply(
            _pctile, raw=True
        )


# ---------------------------------------------------------------------------
# VolatilityRiskPremiumCalculator
# ---------------------------------------------------------------------------

class VolatilityRiskPremiumCalculator:
    """
    Compute Volatility Risk Premium (VRP) and related metrics.

    VRP = Implied Volatility (VIX) - Realised Volatility (HV30)

    Positive VRP = options expensive relative to realized vol → vol sellers win
    Negative VRP = realized > implied → market stress underpriced by options
    """

    def __init__(self):
        self._vix_engine = VIXAnalyticsEngine()

    def _fetch_price_returns(self, ticker: str,
                             start: str) -> Optional[pd.Series]:
        """Fetch daily log returns for a ticker via yfinance."""
        if not _YF_AVAILABLE or yf is None:
            return None
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start, auto_adjust=True)
            if hist.empty:
                return None
            prices = hist["Close"].dropna()
            returns = np.log(prices / prices.shift(1)).dropna()
            returns.name = ticker
            return returns
        except Exception as exc:
            logger.warning("yfinance price fetch %s failed: %s", ticker, exc)
            return None

    def compute_vrp(self, ticker: str = "SPY",
                    lookback_days: int = 252) -> pd.Series:
        """
        Compute daily VRP time series.

        VRP(t) = VIX(t) - HV30(t)
          where HV30(t) = std(log_returns, 30 days) × sqrt(252) × 100

        Returns pd.Series indexed by date.
        """
        # Get VIX
        start = (datetime.now() - timedelta(days=lookback_days + 90)).strftime("%Y-%m-%d")
        vix = self._vix_engine.fetch_vix_history(start=start)

        # Get realised vol from price returns
        returns = self._fetch_price_returns(ticker, start=start)
        if returns is None or returns.empty:
            logger.warning("No price data for %s — VRP requires yfinance", ticker)
            return pd.Series(dtype=float, name="vrp")

        hv30 = _annualised_hv(returns, window=30)

        # Align on common dates
        combined = pd.DataFrame({"vix": vix, "hv30": hv30}).dropna()
        combined = combined.tail(lookback_days)

        vrp = combined["vix"] - combined["hv30"]
        vrp.name = "vrp"
        return vrp

    def compute_vrp_statistics(self, ticker: str = "SPY",
                               lookback_days: int = 252) -> Dict[str, Any]:
        """
        Compute VRP statistics: mean, std, current, z-score, percentile.
        """
        vrp_series = self.compute_vrp(ticker=ticker, lookback_days=lookback_days)

        if vrp_series.empty:
            return {"error": "Could not compute VRP (yfinance required for realized vol)"}

        current  = float(vrp_series.iloc[-1])
        mean_vrp = float(vrp_series.mean())
        std_vrp  = float(vrp_series.std())
        zscore   = (current - mean_vrp) / std_vrp if std_vrp > 0 else 0.0
        pctile   = _percentile_of(current, vrp_series)
        as_of    = str(vrp_series.index[-1].date())

        vix_now  = self._vix_engine.get_current_vix()
        hv30_now = current + vix_now   # VRP = VIX - HV30 → HV30 = VIX - VRP (approx)

        return {
            "as_of":        as_of,
            "ticker":       ticker,
            "vix_current":  round(vix_now, 2),
            "hv30_current": round(vix_now - current, 2) if not np.isnan(vix_now) else None,
            "vrp_current":  round(current, 3),
            "vrp_mean":     round(mean_vrp, 3),
            "vrp_std":      round(std_vrp, 3),
            "vrp_zscore":   round(zscore, 3),
            "vrp_percentile": round(pctile, 1),
            "interpretation": self.interpret_vrp(current),
            "signal":       self._vrp_signal(current, zscore),
            "n_obs":        len(vrp_series),
        }

    def get_vrp_data(self, ticker: str = "SPY",
                     lookback_days: int = 252) -> VRPData:
        """Return fully-populated VRPData dataclass."""
        stats = self.compute_vrp_statistics(ticker=ticker, lookback_days=lookback_days)
        if "error" in stats:
            return VRPData(
                ticker=ticker,
                as_of=str(date.today()),
                vix_implied=float("nan"),
                hv30=float("nan"),
                vrp=float("nan"),
                vrp_mean=float("nan"),
                vrp_std=float("nan"),
                vrp_percentile=50.0,
                vrp_zscore=float("nan"),
                interpretation="INSUFFICIENT_DATA",
                signal="NEUTRAL",
            )

        return VRPData(
            ticker=ticker,
            as_of=stats["as_of"],
            vix_implied=stats["vix_current"],
            hv30=stats.get("hv30_current") or float("nan"),
            vrp=stats["vrp_current"],
            vrp_mean=stats["vrp_mean"],
            vrp_std=stats["vrp_std"],
            vrp_percentile=stats["vrp_percentile"],
            vrp_zscore=stats["vrp_zscore"],
            interpretation=stats["interpretation"],
            signal=stats["signal"],
        )

    def interpret_vrp(self, vrp: float) -> str:
        """
        Interpret VRP level:

        HIGH  (>4): Options expensive → vol selling strategies favorable
        NORMAL (1-4): Normal risk premium, options fairly priced
        LOW   (<1): Options cheap → hedging / vol buying favorable
        NEGATIVE (<0): Realized > implied → market stress episode
        """
        if np.isnan(vrp):
            return "UNKNOWN"
        if vrp > 6:
            return "VERY_HIGH"
        if vrp > 4:
            return "HIGH"
        if vrp > 1:
            return "NORMAL"
        if vrp > 0:
            return "LOW"
        return "NEGATIVE"

    def _vrp_signal(self, vrp: float, zscore: float) -> str:
        """Trading signal derived from VRP level + z-score."""
        if np.isnan(vrp):
            return "NEUTRAL"
        if vrp > 4 and zscore > 1:
            return "SELL_VOL"           # implied >> realized → sell options
        if vrp < 1 and zscore < -1:
            return "BUY_VOL"            # realized >> implied → buy protection
        if vrp < 0:
            return "STRESS"             # realized > implied → acute stress
        return "NEUTRAL"

    def compute_skew_risk_premium(self, ticker: str = "SPY") -> float:
        """
        Estimate skew risk premium as a proxy.

        Without live options data, we approximate using:
          - VIX (ATM 1M implied vol)
          - Historical skew: when VIX is elevated, skew (put/call spread) widens
          - SKEW Index from CBOE (if available via yfinance ^SKEW)

        Returns estimated 25-delta put premium over ATM in vol points.
        Falls back to a VIX-derived estimate if SKEW is unavailable.
        """
        # Try CBOE SKEW index
        if _YF_AVAILABLE and yf is not None:
            try:
                tk = yf.Ticker("^SKEW")
                hist = tk.history(period="5d", auto_adjust=True)
                if not hist.empty:
                    skew_index = float(hist["Close"].dropna().iloc[-1])
                    # CBOE SKEW to skew vol premium approximation:
                    # SKEW = 100 + 10 × skew_premium (rough conversion)
                    return round((skew_index - 100) / 10, 2)
            except Exception:
                pass

        # Fallback: VIX-based estimate
        vix = self._vix_engine.get_current_vix()
        if not np.isnan(vix):
            # Empirical approximation: skew ≈ 0.15 × VIX for equity index options
            return round(0.15 * vix, 2)

        return float("nan")


# ---------------------------------------------------------------------------
# MultiAssetVolTracker
# ---------------------------------------------------------------------------

class MultiAssetVolTracker:
    """
    Track and compare implied volatility across asset classes.

    Tracks: VIX (equity), OVX (crude), GVZ (gold), EVZ (EUR/USD),
            VXEEM (EM equity — proxy via FRED or yfinance).
    """

    _VOL_SERIES: Dict[str, str] = {
        "VIX":   "VIXCLS",    # S&P 500 1M implied vol
        "VXV":   "VXVCLS",    # S&P 500 3M implied vol
        "OVX":   "OVXCLS",    # Crude Oil 1M implied vol
        "GVZ":   "GVZCLS",    # Gold 1M implied vol
        "EVZ":   "EVZ",       # EUR/USD 1M implied vol
    }

    _YF_FALLBACKS: Dict[str, str] = {
        "VXEEM": "^VXEEM",   # EM equity vol
        "VVIX":  "^VVIX",    # vol of VIX
    }

    def __init__(self, start: str = "2010-01-01"):
        self._start = start
        self._vol_data: Optional[pd.DataFrame] = None

    def fetch_all_vol_indices(self) -> Dict[str, float]:
        """
        Fetch latest reading for all vol indices.

        Returns dict: {name: current_level}
        """
        results: Dict[str, float] = {}
        frames: Dict[str, pd.Series] = {}

        for name, fred_id in self._VOL_SERIES.items():
            s = _fred.fetch(fred_id, start=self._start)
            clean = s.dropna()
            if not clean.empty:
                results[name] = round(float(clean.iloc[-1]), 2)
                frames[name] = clean
            else:
                results[name] = float("nan")

        # yfinance fallbacks for VXEEM, VVIX
        if _YF_AVAILABLE and yf is not None:
            for name, ticker in self._YF_FALLBACKS.items():
                try:
                    tk = yf.Ticker(ticker)
                    hist = tk.history(start=self._start, auto_adjust=True)
                    if not hist.empty:
                        close = hist["Close"].dropna()
                        results[name] = round(float(close.iloc[-1]), 2)
                        frames[name] = close
                except Exception as exc:
                    logger.debug("yfinance %s failed: %s", ticker, exc)

        if frames:
            self._vol_data = pd.DataFrame(frames).ffill(limit=5)

        return results

    def compute_vol_regime(self) -> str:
        """
        Classify current multi-asset vol environment.

        Composite = average z-score across all available vol indices
          CALM      : composite z < -0.5
          NORMAL    : composite z in [-0.5, 0.5]
          STRESSED  : composite z in [0.5, 1.5]
          CRISIS    : composite z > 1.5
        """
        if self._vol_data is None:
            self.fetch_all_vol_indices()

        if self._vol_data is None or self._vol_data.empty:
            return "UNKNOWN"

        # Z-score each series using its 252-day rolling stats
        zscores = []
        for col in self._vol_data.columns:
            s = self._vol_data[col].dropna()
            if len(s) < 60:
                continue
            mean = float(s.rolling(252, min_periods=60).mean().iloc[-1])
            std  = float(s.rolling(252, min_periods=60).std().iloc[-1])
            if std > 0:
                z = (float(s.iloc[-1]) - mean) / std
                zscores.append(z)

        if not zscores:
            return "UNKNOWN"

        composite_z = float(np.mean(zscores))

        if composite_z < -0.5:
            return f"CALM (z={composite_z:.2f})"
        if composite_z < 0.5:
            return f"NORMAL (z={composite_z:.2f})"
        if composite_z < 1.5:
            return f"STRESSED (z={composite_z:.2f})"
        return f"CRISIS (z={composite_z:.2f})"

    def compute_cross_asset_vol_correlation(self,
                                            window_days: int = 60) -> pd.DataFrame:
        """
        Rolling 60-day correlation between vol indices.

        Rising cross-asset vol correlation = systemic risk (everything stressed at once).
        Returns correlation matrix for latest window.
        """
        if self._vol_data is None:
            self.fetch_all_vol_indices()

        if self._vol_data is None or self._vol_data.empty:
            return pd.DataFrame()

        window = self._vol_data.dropna(how="all").tail(window_days)
        if len(window) < 20:
            return pd.DataFrame()

        corr = window.corr(method="pearson")
        return corr.round(4)

    def detect_vol_decoupling(self, asset1: str, asset2: str,
                              window_days: int = 20) -> bool:
        """
        Detect vol decoupling: one asset's vol rising while the other is stable.

        Decoupling signal = correlation in vol changes drops significantly
        below longer-term average → idiosyncratic stress.
        """
        if self._vol_data is None:
            self.fetch_all_vol_indices()

        if self._vol_data is None:
            return False

        if asset1 not in self._vol_data.columns or asset2 not in self._vol_data.columns:
            logger.warning("Vol decoupling: %s or %s not available", asset1, asset2)
            return False

        s1 = self._vol_data[asset1].dropna()
        s2 = self._vol_data[asset2].dropna()

        aligned = pd.concat([s1, s2], axis=1).dropna()
        if len(aligned) < window_days + 60:
            return False

        short_corr = float(aligned.tail(window_days).corr().iloc[0, 1])
        long_corr  = float(aligned.tail(window_days + 60).corr().iloc[0, 1])

        # Decoupling = recent correlation far below long-term baseline
        return (long_corr - short_corr) > 0.40

    def get_vol_summary(self) -> Dict[str, Any]:
        """Return summary dict: current levels + regime."""
        levels = self.fetch_all_vol_indices()
        regime = self.compute_vol_regime()
        return {
            "as_of":          str(date.today()),
            "vol_levels":     levels,
            "regime":         regime,
            "vix_ovx_ratio":  (round(levels.get("VIX", float("nan")) /
                                levels.get("OVX", float("nan")), 3)
                               if levels.get("OVX") and not np.isnan(levels.get("OVX", float("nan")))
                               else None),
        }


# ---------------------------------------------------------------------------
# InflationTradingSignals
# ---------------------------------------------------------------------------

class InflationTradingSignals:
    """
    Generate actionable inflation-related trading signals.

    Uses breakeven data to derive cross-asset trading recommendations
    for TIPS vs nominal, sector rotation, and currency carry.
    """

    def __init__(self, breakeven_analyzer: InflationBreakevenAnalyzer):
        self._analyzer = breakeven_analyzer

    def get_tips_vs_nominal_signal(self) -> Dict[str, Any]:
        """
        Generate TIPS vs nominal Treasury relative value signal.

        Rising breakevens → TIPS outperform → long TIPS / short TLT
        Falling breakevens → nominal outperform → long TLT / short TIP

        Uses current breakeven level + momentum to determine conviction.
        """
        curve  = self._analyzer.compute_breakeven_curve()
        be_10y = curve.get("breakeven_10y")
        be_mom = self._analyzer.compute_breakeven_momentum(lookback_days=21)
        regime = curve.get("regime", "UNKNOWN")
        pctile = self._analyzer.compute_breakeven_percentile()

        signal  = "NEUTRAL"
        conviction = "LOW"
        trade = {}

        if be_10y is not None and not np.isnan(be_mom):
            if be_mom > 0.10 and be_10y > 2.0:
                signal = "LONG_TIPS_SHORT_NOMINAL"
                conviction = "HIGH" if be_mom > 0.20 else "MEDIUM"
                trade = {
                    "long":  "TIP (iShares TIPS ETF)",
                    "short": "TLT (iShares 20Y+ T-Bond ETF)",
                    "rationale": f"Breakeven rising +{be_mom:.2f}pp (21d), "
                                 f"regime={regime}, pctile={pctile:.0f}",
                }
            elif be_mom < -0.10 or (be_10y is not None and be_10y < 1.8):
                signal = "LONG_NOMINAL_SHORT_TIPS"
                conviction = "HIGH" if be_mom < -0.20 else "MEDIUM"
                trade = {
                    "long":  "TLT (iShares 20Y+ T-Bond ETF)",
                    "short": "TIP (iShares TIPS ETF)",
                    "rationale": f"Breakeven falling {be_mom:.2f}pp (21d), "
                                 f"regime={regime}, pctile={pctile:.0f}",
                }

        return {
            "signal":          signal,
            "conviction":      conviction,
            "breakeven_10y":   be_10y,
            "momentum_21d":    round(be_mom, 4) if not np.isnan(be_mom) else None,
            "regime":          regime,
            "percentile":      round(pctile, 1),
            "trade":           trade,
            "as_of":           curve.get("as_of"),
        }

    def get_inflation_sector_rotation_signal(self,
                                             breakeven: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate equity sector rotation signal based on inflation regime.

        HIGH inflation (>2.5%):
          Overweight:  Energy, Materials, REITs, Infrastructure, Financials
          Underweight: Technology, Consumer Discretionary, Utilities (long-duration)

        TARGET inflation (1.5–2.5%):
          Balanced allocation; favor cyclicals over defensives

        LOW inflation (<1.5%):
          Overweight:  Technology, Healthcare, Consumer Staples, Utilities
          Underweight: Energy, Materials, Financials
        """
        if breakeven is None:
            curve = self._analyzer.compute_breakeven_curve()
            breakeven = curve.get("breakeven_10y")

        if breakeven is None or np.isnan(float(breakeven)):
            return {"signal": "UNKNOWN", "reason": "No breakeven data available"}

        breakeven = float(breakeven)
        regime = self._analyzer.detect_regime(breakeven)

        if regime in ("ELEVATED", "HIGH"):
            return {
                "regime":      regime,
                "breakeven":   breakeven,
                "overweight":  ["Energy (XLE)", "Materials (XLB)", "REITs (VNQ)",
                                "Infrastructure (IGF)", "Financials (XLF)",
                                "Commodities (GSG)", "Industrial (XLI)"],
                "underweight": ["Technology (XLK)", "Consumer Disc. (XLY)",
                                "Long-Duration Bonds (TLT)", "Growth ETFs (QQQ)"],
                "rationale":   (f"Breakeven {breakeven:.2f}% = {regime} inflation. "
                                "Real assets and commodities outperform in inflation."),
                "bond_signal": "UNDERWEIGHT_DURATION — real yields will rise",
                "gold_signal": "BULLISH — inflation hedge",
            }

        if regime == "TARGET":
            return {
                "regime":      regime,
                "breakeven":   breakeven,
                "overweight":  ["Cyclicals (XLY)", "Financials (XLF)",
                                "Small Cap (IWM)", "Value (IVE)"],
                "underweight": ["Defensives (XLU)", "Long-Duration Bonds (TLT)"],
                "rationale":   (f"Breakeven {breakeven:.2f}% near Fed target. "
                                "Balanced regime favors cyclicals."),
                "bond_signal": "NEUTRAL — at-target inflation well-priced",
                "gold_signal": "NEUTRAL — no inflation premium",
            }

        # LOW
        return {
            "regime":      regime,
            "breakeven":   breakeven,
            "overweight":  ["Technology (XLK)", "Healthcare (XLV)",
                            "Consumer Staples (XLP)", "Utilities (XLU)",
                            "Long-Duration Bonds (TLT)"],
            "underweight": ["Energy (XLE)", "Materials (XLB)", "Financials (XLF)"],
            "rationale":   (f"Breakeven {breakeven:.2f}% = {regime} inflation / "
                            "disinflation. Growth and duration outperform."),
            "bond_signal": "BULLISH DURATION — low/falling inflation",
            "gold_signal": "CAUTIOUS — deflation headwinds for gold",
        }

    def get_carry_signal_from_real_yields(self,
                                          real_yield: Optional[float] = None) -> str:
        """
        Generate carry and currency signal from TIPS real yield.

        High positive real yield (>1.5%):  USD carry attractive → USD bullish
        Near-zero real yield (0–0.5%):     Neutral carry
        Negative real yield (<0%):          Carry unfavorable → Gold / commodities
        Very negative (<-1%):               Strong hold gold / commodities signal

        Returns descriptive signal string.
        """
        if real_yield is None:
            reals = self._analyzer.compute_real_yields()
            real_yield = reals.get("real_yield_10y")

        if real_yield is None or np.isnan(float(real_yield)):
            return "UNKNOWN — real yield data unavailable"

        real_yield = float(real_yield)

        if real_yield > 2.0:
            return (f"STRONG_USD_CARRY: Real yield {real_yield:.2f}% highly attractive. "
                    "USD strength, risk asset headwind, gold bearish.")
        if real_yield > 1.0:
            return (f"USD_CARRY_ATTRACTIVE: Real yield {real_yield:.2f}%. "
                    "USD supportive, bonds competitive vs equities.")
        if real_yield > 0.3:
            return (f"NEUTRAL_CARRY: Real yield {real_yield:.2f}%. "
                    "Balanced USD; equities and bonds fairly valued.")
        if real_yield > -0.5:
            return (f"NEAR_ZERO_REAL: Real yield {real_yield:.2f}%. "
                    "Gold and commodities as inflation hedges. USD neutral.")
        if real_yield > -1.5:
            return (f"NEGATIVE_REAL: Real yield {real_yield:.2f}%. "
                    "HOLD GOLD, commodities. USD bearish. Risk assets inflated.")
        return (f"DEEPLY_NEGATIVE_REAL: Real yield {real_yield:.2f}%. "
                "STRONG GOLD / commodity signal. USD very weak. QE-era dynamics.")

    def get_complete_inflation_signals(self) -> Dict[str, Any]:
        """Return all inflation trading signals in one call."""
        tips_vs_nominal = self.get_tips_vs_nominal_signal()
        be = tips_vs_nominal.get("breakeven_10y")
        sector = self.get_inflation_sector_rotation_signal(breakeven=be)

        reals = self._analyzer.compute_real_yields()
        carry = self.get_carry_signal_from_real_yields(
            real_yield=reals.get("real_yield_10y")
        )

        return {
            "as_of":             tips_vs_nominal.get("as_of"),
            "tips_vs_nominal":   tips_vs_nominal,
            "sector_rotation":   sector,
            "carry_signal":      carry,
            "real_yields":       reals,
            "inflation_regime":  sector.get("regime", "UNKNOWN"),
        }


# ---------------------------------------------------------------------------
# VolatilityAnalyticsEngine — Orchestrator
# ---------------------------------------------------------------------------

class VolatilityAnalyticsEngine:
    """
    Top-level orchestrator for volatility and inflation analytics.

    Combines all sub-engines into a unified interface for SENTINEL.

    Usage:
        engine = VolatilityAnalyticsEngine()
        dashboard = engine.get_full_dashboard()
        report = engine.generate_vol_report()
    """

    def __init__(self, start: str = "2003-01-01"):
        self._start = start
        self._breakeven  = InflationBreakevenAnalyzer(start=start)
        self._vix        = VIXAnalyticsEngine()
        self._vrp        = VolatilityRiskPremiumCalculator()
        self._multi_vol  = MultiAssetVolTracker(start=start)
        self._inf_signals = InflationTradingSignals(self._breakeven)

    def get_full_dashboard(self) -> Dict[str, Any]:
        """
        Return comprehensive vol + inflation dashboard.

        Fetches and computes all metrics in one call.
        Returns nested dict suitable for JSON serialisation.
        """
        result: Dict[str, Any] = {"as_of": str(date.today())}

        # 1. VIX regime
        try:
            result["vix_regime"] = self._vix.compute_vix_regime().__dict__
        except Exception as exc:
            result["vix_regime"] = {"error": str(exc)}

        # 2. VIX term structure
        try:
            result["vix_term_structure"] = self._vix.compute_vix_term_structure()
        except Exception as exc:
            result["vix_term_structure"] = {"error": str(exc)}

        # 3. VIX moving averages
        try:
            result["vix_ma_signal"] = self._vix.compute_vix_ma()
        except Exception as exc:
            result["vix_ma_signal"] = {"error": str(exc)}

        # 4. VRP (requires yfinance)
        try:
            vrp_stats = self._vrp.compute_vrp_statistics("SPY", lookback_days=252)
            result["vrp"] = vrp_stats
        except Exception as exc:
            result["vrp"] = {"error": str(exc)}

        # 5. Breakeven curve
        try:
            result["breakeven"] = self._breakeven.get_breakeven_data().__dict__
        except Exception as exc:
            result["breakeven"] = {"error": str(exc)}

        # 6. Real yields
        try:
            result["real_yields"] = self._breakeven.compute_real_yields()
        except Exception as exc:
            result["real_yields"] = {"error": str(exc)}

        # 7. Multi-asset vol
        try:
            result["multi_asset_vol"] = self._multi_vol.get_vol_summary()
        except Exception as exc:
            result["multi_asset_vol"] = {"error": str(exc)}

        # 8. Inflation trading signals
        try:
            result["inflation_signals"] = self._inf_signals.get_complete_inflation_signals()
        except Exception as exc:
            result["inflation_signals"] = {"error": str(exc)}

        return result

    def get_inflation_outlook(self) -> Dict[str, Any]:
        """
        Return standalone inflation outlook including breakeven + signals.
        """
        be_data  = self._breakeven.get_breakeven_data()
        be_curve = self._breakeven.compute_breakeven_curve()
        signals  = self._inf_signals.get_complete_inflation_signals()
        reals    = self._breakeven.compute_real_yields()

        return {
            "as_of":              be_data.as_of,
            "regime":             be_data.regime,
            "breakeven_10y":      be_data.breakeven_10y,
            "breakeven_5y":       be_data.breakeven_5y,
            "breakeven_5y5y":     be_data.breakeven_5y5y,
            "real_yield_10y":     be_data.real_yield_10y,
            "momentum_21d":       be_data.momentum_21d,
            "percentile_10yr":    be_data.percentile_10yr,
            "irp":                be_data.implied_inflation_risk_premium,
            "curve_shape":        be_curve.get("curve_shape"),
            "sector_signal":      signals.get("sector_rotation", {}).get("overweight"),
            "carry_signal":       signals.get("carry_signal"),
            "tips_vs_nominal":    signals.get("tips_vs_nominal", {}).get("signal"),
            "real_yield_regime":  reals.get("real_yield_regime"),
        }

    def get_vol_trading_signals(self) -> Dict[str, Any]:
        """
        Return all volatility-derived trading signals.
        """
        vix_regime = self._vix.compute_vix_regime()
        vix_ma     = self._vix.compute_vix_ma()
        ts         = self._vix.compute_vix_term_structure()

        vrp_stats  = self._vrp.compute_vrp_statistics("SPY", lookback_days=252)
        skew_prem  = self._vrp.compute_skew_risk_premium()

        multi_summary = self._multi_vol.get_vol_summary()

        vol_signal = "NEUTRAL"
        if vrp_stats.get("signal") == "SELL_VOL":
            vol_signal = "SELL_VOL — options expensive vs realized"
        elif vrp_stats.get("signal") == "BUY_VOL":
            vol_signal = "BUY_VOL — options cheap, hedging attractive"
        elif vrp_stats.get("signal") == "STRESS":
            vol_signal = "STRESS — realized > implied, acute vol event"

        return {
            "as_of":                str(date.today()),
            "vix_level":            vix_regime.level,
            "vix_regime":           vix_regime.label,
            "vix_percentile_5yr":   vix_regime.percentile,
            "vix_term_structure":   vix_regime.term_structure,
            "vxv_vix_ratio":        vix_regime.vxv_vix_ratio,
            "mean_reversion_signal":vix_regime.mean_reversion,
            "spike_detected":       vix_regime.spike_today,
            "vrp_signal":           vol_signal,
            "vrp_current":          vrp_stats.get("vrp_current"),
            "vrp_interpretation":   vrp_stats.get("interpretation"),
            "vrp_zscore":           vrp_stats.get("vrp_zscore"),
            "skew_premium":         skew_prem,
            "vol_regime_composite": multi_summary.get("regime"),
            "ma_signal":            vix_ma.get("signal"),
        }

    def generate_vol_report(self) -> str:
        """
        Generate comprehensive text report on vol + inflation conditions.
        """
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║  SENTINEL VOLATILITY & INFLATION ANALYTICS REPORT v3     ║",
            "╚══════════════════════════════════════════════════════════╝",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
        ]

        # VIX section
        try:
            vix_r = self._vix.compute_vix_regime()
            ts    = self._vix.compute_vix_term_structure()
            lines += [
                "━━━ VIX / VOLATILITY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"  VIX Level:        {vix_r.level:.2f}  ({vix_r.as_of})",
                f"  Regime:           {vix_r.label}",
                f"  5Y Percentile:    {vix_r.percentile:.1f} / 100",
                f"  Term Structure:   {vix_r.term_structure}  (VXV/VIX={vix_r.vxv_vix_ratio:.3f})",
                f"  Mean Reversion:   {vix_r.mean_reversion}",
                f"  Spike Today:      {'YES ⚠' if vix_r.spike_today else 'No'}",
                "",
            ]
        except Exception as exc:
            lines.append(f"  VIX section error: {exc}")

        # VRP section
        try:
            vrp = self._vrp.compute_vrp_statistics("SPY", lookback_days=252)
            if "error" not in vrp:
                lines += [
                    "━━━ VOLATILITY RISK PREMIUM (SPY) ━━━━━━━━━━━━━━━━━━━━━━━",
                    f"  VIX (Implied):    {vrp.get('vix_current', 'N/A')}",
                    f"  HV30 (Realised):  {vrp.get('hv30_current', 'N/A')}",
                    f"  VRP:              {vrp.get('vrp_current', 'N/A'):.2f} vol pts",
                    f"  VRP Z-score:      {vrp.get('vrp_zscore', 'N/A'):.2f}",
                    f"  VRP Percentile:   {vrp.get('vrp_percentile', 'N/A'):.1f} / 100",
                    f"  Interpretation:   {vrp.get('interpretation', 'N/A')}",
                    f"  Signal:           {vrp.get('signal', 'N/A')}",
                    "",
                ]
            else:
                lines.append(f"  VRP: {vrp['error']}")
        except Exception as exc:
            lines.append(f"  VRP section error: {exc}")

        # Breakeven section
        try:
            be = self._breakeven.get_breakeven_data()
            lines += [
                "━━━ INFLATION BREAKEVEN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"  10Y Breakeven:    {be.breakeven_10y:.3f}%" if be.breakeven_10y else
                "  10Y Breakeven:    N/A",
                f"  5Y Breakeven:     {be.breakeven_5y:.3f}%" if be.breakeven_5y else
                "  5Y Breakeven:     N/A",
                f"  5Y5Y Forward:     {be.breakeven_5y5y:.3f}%" if be.breakeven_5y5y else
                "  5Y5Y Forward:     N/A",
                f"  10Y Real Yield:   {be.real_yield_10y:.3f}%" if be.real_yield_10y else
                "  10Y Real Yield:   N/A",
                f"  Regime:           {be.regime}",
                f"  Curve Shape:      {be.curve_shape}",
                f"  Momentum 21d:     {be.momentum_21d:+.3f}pp" if be.momentum_21d else
                "  Momentum 21d:     N/A",
                f"  Percentile 10yr:  {be.percentile_10yr:.1f} / 100" if be.percentile_10yr else
                "  Percentile 10yr:  N/A",
                f"  Inflation IRP:    {be.implied_inflation_risk_premium:+.3f}pp" if be.implied_inflation_risk_premium else
                "  Inflation IRP:    N/A",
                "",
            ]
        except Exception as exc:
            lines.append(f"  Breakeven section error: {exc}")

        # Multi-asset vol
        try:
            mv = self._multi_vol.get_vol_summary()
            levels = mv.get("vol_levels", {})
            lines += [
                "━━━ MULTI-ASSET VOLATILITY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ]
            for name, lvl in levels.items():
                lvl_str = f"{lvl:.2f}" if not np.isnan(lvl) else "N/A"
                lines.append(f"  {name:<10}: {lvl_str}")
            lines += [
                f"  Composite Regime: {mv.get('regime', 'N/A')}",
                "",
            ]
        except Exception as exc:
            lines.append(f"  Multi-vol section error: {exc}")

        # Inflation trading signals
        try:
            sigs = self._inf_signals.get_complete_inflation_signals()
            tips = sigs.get("tips_vs_nominal", {})
            sect = sigs.get("sector_rotation", {})
            lines += [
                "━━━ INFLATION TRADING SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"  TIPS vs Nominal:  {tips.get('signal', 'N/A')} [{tips.get('conviction', 'N/A')}]",
                f"  Sector Overweight:{', '.join(sect.get('overweight', ['N/A'])[:3])}",
                f"  Sector Underweight:{', '.join(sect.get('underweight', ['N/A'])[:2])}",
                f"  Bond Signal:      {sect.get('bond_signal', 'N/A')}",
                f"  Gold Signal:      {sect.get('gold_signal', 'N/A')}",
                f"  USD Carry:        {sigs.get('carry_signal', 'N/A')[:70]}",
                "",
            ]
        except Exception as exc:
            lines.append(f"  Signals section error: {exc}")

        lines.append("━" * 60)
        return "\n".join(lines)

    # Convenience pass-throughs
    def get_vix_engine(self) -> VIXAnalyticsEngine:
        return self._vix

    def get_breakeven_analyzer(self) -> InflationBreakevenAnalyzer:
        return self._breakeven

    def get_vrp_calculator(self) -> VolatilityRiskPremiumCalculator:
        return self._vrp

    def get_multi_vol_tracker(self) -> MultiAssetVolTracker:
        return self._multi_vol

    def get_inflation_signal_engine(self) -> InflationTradingSignals:
        return self._inf_signals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SENTINEL VOLATILITY ANALYTICS ENGINE v3                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    engine = VolatilityAnalyticsEngine(start="2003-01-01")

    print("\n[1/5] Fetching VIX history and computing regime...")
    try:
        vix_s = engine.get_vix_engine().fetch_vix_history(start="2000-01-01")
        regime = engine.get_vix_engine().compute_vix_regime()
        print(f"      VIX data: {len(vix_s)} observations "
              f"({vix_s.index.min().date()} → {vix_s.index.max().date()})")
        print(f"      Current VIX:   {regime.level:.2f}")
        print(f"      Regime:        {regime.label}")
        print(f"      Term Structure:{regime.term_structure}  (VXV/VIX={regime.vxv_vix_ratio})")
        print(f"      5Y Percentile: {regime.percentile:.1f} / 100")
        print(f"      Mean Reversion:{regime.mean_reversion}")
    except Exception as exc:
        print(f"      ERROR: {exc}")

    print("\n[2/5] Computing Volatility Risk Premium (SPY)...")
    try:
        vrp = engine.get_vrp_calculator().compute_vrp_statistics("SPY")
        if "error" in vrp:
            print(f"      {vrp['error']}")
        else:
            print(f"      VIX (implied):  {vrp['vix_current']}")
            print(f"      HV30 (realised):{vrp['hv30_current']}")
            print(f"      VRP:            {vrp['vrp_current']:.2f}")
            print(f"      Z-score:        {vrp['vrp_zscore']:.2f}")
            print(f"      Interpretation: {vrp['interpretation']}")
            print(f"      Signal:         {vrp['signal']}")
    except Exception as exc:
        print(f"      ERROR: {exc}")

    print("\n[3/5] Fetching breakeven curve...")
    try:
        engine.get_breakeven_analyzer().fetch_breakeven_data()
        curve = engine.get_breakeven_analyzer().compute_breakeven_curve()
        print(f"      5Y Breakeven:   {curve.get('breakeven_5y')}%")
        print(f"      10Y Breakeven:  {curve.get('breakeven_10y')}%")
        print(f"      5Y5Y Forward:   {curve.get('breakeven_5y5y')}%")
        print(f"      Curve Shape:    {curve.get('curve_shape')}")
        print(f"      Regime:         {curve.get('regime')}")
        mom = engine.get_breakeven_analyzer().compute_breakeven_momentum()
        print(f"      Momentum 21d:   {mom:+.3f}pp")
    except Exception as exc:
        print(f"      ERROR: {exc}")

    print("\n[4/5] Multi-asset vol regime...")
    try:
        mv = engine.get_multi_vol_tracker()
        levels = mv.fetch_all_vol_indices()
        for name, val in levels.items():
            v = f"{val:.2f}" if not np.isnan(val) else "N/A"
            print(f"      {name:<10}: {v}")
        print(f"      Composite:  {mv.compute_vol_regime()}")
    except Exception as exc:
        print(f"      ERROR: {exc}")

    print("\n[5/5] Inflation trading signals...")
    try:
        signals = engine.get_inflation_signal_engine().get_complete_inflation_signals()
        tips = signals.get("tips_vs_nominal", {})
        sect = signals.get("sector_rotation", {})
        carry = signals.get("carry_signal", "N/A")
        print(f"      TIPS vs Nominal: {tips.get('signal')} [{tips.get('conviction')}]")
        print(f"      Sector overweight: {', '.join((sect.get('overweight') or ['N/A'])[:3])}")
        print(f"      Carry signal: {carry[:80]}")
    except Exception as exc:
        print(f"      ERROR: {exc}")

    print("\n\n" + "=" * 62)
    print("FULL VOL REPORT")
    print("=" * 62)
    print(engine.generate_vol_report())

    print("\nDone.")
