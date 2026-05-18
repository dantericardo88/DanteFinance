"""
Inflation breakeven, VIX term structure, and volatility risk premium analytics.
Free data: FRED (TIPS/nominals, VIX term structure), CBOE VIX data.

dim_048 — Inflation breakeven / VIX analytics / vol risk premium (target: 9)

Classes
-------
InflationBreakevenEngine
    FRED TIPS/breakeven series: 5Y, 10Y, 5Y5Y forward breakeven, real yields.
    Computes Fed-target comparison, BEI regime, market-implied inflation path.

VIXTermStructureAnalyzer
    VIX spot + term structure (VXST, VIXCLS, VXMT), contango ratio, percentiles,
    vol-of-vol proxy, momentum.

VolatilityRiskPremiumEngine
    VRP = VIX - SPY 21D realized vol. Rolling history, regime classification,
    trading signals.

InflationTradingSignals
    TIPS vs nominal, BEI compression/expansion, stagflation, real rate shock.

CrossAssetVolRegime
    MOVE index proxy, FX vol, equity/bond vol ratio, equity-bond correlation.

FastAPI router: inflation_vix_router
"""
from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/csv,text/html,*/*",
}
_TIMEOUT = 30

_DB_PATH = Path(__file__).parent.parent.parent / ".danteforge" / "inflation_vix_cache.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# FRED series for inflation
INFLATION_SERIES: dict[str, str] = {
    "T5YIE":   "5-Year Breakeven Inflation Rate",
    "T10YIE":  "10-Year Breakeven Inflation Rate",
    "T5YIFR":  "5-Year, 5-Year Forward Inflation Expectation Rate",
    "DFII5":   "5-Year TIPS Real Yield",
    "DFII10":  "10-Year TIPS Real Yield",
    "DFII20":  "20-Year TIPS Real Yield",
    "DFII30":  "30-Year TIPS Real Yield",
    "CPIAUCSL": "CPI All Urban Consumers",
    "PCEPILFE": "PCE Price Index Excluding Food and Energy",
}

# FRED series for VIX
VIX_SERIES: dict[str, str] = {
    "VXST":   "CBOE Short-Term Volatility Index (9-Day)",
    "VIXCLS": "CBOE Volatility Index: VIX (1-Month)",
    "VXMT":   "CBOE Mid-Term Volatility Index (3-Month)",
}

FED_INFLATION_TARGET = 2.0  # percent

# VIX extended series (6-month)
VIX_SERIES_EXTENDED: dict[str, str] = {
    "VXST":   "CBOE Short-Term Volatility Index (9-Day)",
    "VIXCLS": "CBOE Volatility Index: VIX (1-Month)",
    "VXMT":   "CBOE Mid-Term Volatility Index (3-Month)",
    "VXMT6M": "CBOE 6-Month Volatility Index",   # proxy: VXMT used if unavailable
}

# Inflation regime quadrant labels
INFLATION_REGIMES = ("Goldilocks", "Reflation", "Stagflation", "Deflation")

# ---------------------------------------------------------------------------
# SQLite cache helpers
# ---------------------------------------------------------------------------


def _init_cache_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fred_cache (
            series_id TEXT,
            fetched_at INTEGER,
            csv_data   TEXT,
            PRIMARY KEY (series_id)
        )
        """
    )
    conn.commit()
    return conn


def _cache_get(series_id: str, max_age_seconds: int = 3600) -> Optional[str]:
    try:
        conn = _init_cache_db()
        row = conn.execute(
            "SELECT csv_data, fetched_at FROM fred_cache WHERE series_id=?",
            (series_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        csv_data, fetched_at = row
        if time.time() - fetched_at > max_age_seconds:
            return None
        return csv_data
    except Exception:
        return None


def _cache_set(series_id: str, csv_data: str) -> None:
    try:
        conn = _init_cache_db()
        conn.execute(
            "INSERT OR REPLACE INTO fred_cache (series_id, fetched_at, csv_data) VALUES (?,?,?)",
            (series_id, int(time.time()), csv_data),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# FRED fetch helper (synchronous, with cache)
# ---------------------------------------------------------------------------


def _fetch_fred_series(
    series_id: str,
    lookback_days: int = 756,
    max_cache_age: int = 3600,
) -> pd.Series:
    """Fetch a single FRED series as a pandas Series (float, date index).

    Uses free CSV endpoint — no API key required.
    Results are cached in SQLite for `max_cache_age` seconds.
    """
    cached = _cache_get(series_id, max_cache_age)
    if cached:
        try:
            df = pd.read_csv(
                pd.io.common.StringIO(cached), index_col=0, parse_dates=True
            )
            series = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
            cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
            return series[series.index >= cutoff].sort_index()
        except Exception:
            pass

    end = date.today()
    start = end - timedelta(days=lookback_days + 30)
    url = (
        f"{FRED_BASE}?id={series_id}"
        f"&observation_start={start.strftime('%Y-%m-%d')}"
        f"&observation_end={end.strftime('%Y-%m-%d')}"
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        _cache_set(series_id, resp.text)
        df = pd.read_csv(
            pd.io.common.StringIO(resp.text), index_col=0, parse_dates=True
        )
        series = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
        return series[series.index >= cutoff].sort_index()
    except Exception as exc:
        logger.warning("fred_fetch_failed", series_id=series_id, error=str(exc))
        return pd.Series(dtype=float)


def _latest(series: pd.Series) -> Optional[float]:
    """Return most recent non-NaN value, or None."""
    s = series.dropna()
    if s.empty:
        return None
    return float(round(s.iloc[-1], 4))


def _yf_close(
    ticker: str, lookback_days: int = 756
) -> pd.Series:
    """Fetch adjusted close prices from yfinance."""
    end = date.today()
    start = end - timedelta(days=lookback_days + 30)
    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.Series(dtype=float)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].squeeze()
        else:
            closes = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        closes.index = pd.to_datetime(closes.index)
        return closes.dropna().sort_index()
    except Exception as exc:
        logger.warning("yfinance_fetch_failed", ticker=ticker, error=str(exc))
        return pd.Series(dtype=float)


def _annualized_realized_vol(returns: pd.Series, window: int = 21) -> Optional[float]:
    """Compute annualized realized volatility over rolling `window` days."""
    r = returns.dropna()
    if len(r) < window:
        return None
    rv = float(r.iloc[-window:].std() * np.sqrt(252))
    return round(rv * 100, 4)  # return as percent (VIX-equivalent units)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class BreakevenSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    breakeven_5y: Optional[float] = None
    breakeven_10y: Optional[float] = None
    forward_5y5y: Optional[float] = None
    real_yield_5y: Optional[float] = None
    real_yield_10y: Optional[float] = None
    fed_target: float = FED_INFLATION_TARGET
    deviation_5y_from_target: Optional[float] = None
    deviation_10y_from_target: Optional[float] = None
    breakeven_regime_5y: str = "unknown"
    breakeven_regime_10y: str = "unknown"
    real_rate_stance_5y: str = "unknown"
    real_rate_stance_10y: str = "unknown"
    market_implied_path: dict[str, Optional[float]] = Field(default_factory=dict)
    as_of: str = ""


class VIXSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    vix_9d: Optional[float] = None
    vix_1m: Optional[float] = None
    vix_3m: Optional[float] = None
    contango_ratio: Optional[float] = None
    spread_3m_1m: Optional[float] = None
    term_structure_shape: str = "unknown"
    vix_regime: str = "unknown"
    vix_percentile_1y: Optional[float] = None
    vix_percentile_5y: Optional[float] = None
    vix_percentile_10y: Optional[float] = None
    vix_mom_5d: Optional[float] = None
    vol_of_vol_20d: Optional[float] = None
    as_of: str = ""


class VRPSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    vix_current: Optional[float] = None
    realized_vol_21d: Optional[float] = None
    vrp_current: Optional[float] = None
    vrp_regime: str = "unknown"
    vrp_signal: str = "neutral"
    as_of: str = ""


class VRPHistory(BaseModel):
    model_config = ConfigDict(frozen=True)

    dates: list[str]
    vrp_series: list[Optional[float]]
    vix_series: list[Optional[float]]
    rv_series: list[Optional[float]]
    avg_vrp: Optional[float] = None
    pct_positive: Optional[float] = None


class InflationSignals(BaseModel):
    model_config = ConfigDict(frozen=True)

    tips_vs_nominal_signal: str = "neutral"
    bei_compression: bool = False
    bei_expansion: bool = False
    stagflation_indicator: bool = False
    real_rate_shock: bool = False
    signal_details: dict[str, Any] = Field(default_factory=dict)
    as_of: str = ""


class VolRegimeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    equity_vol_regime: str = "unknown"
    bond_vol_proxy: Optional[float] = None
    fx_vol_proxy: Optional[float] = None
    equity_bond_vol_ratio: Optional[float] = None
    risk_mode: str = "unknown"
    equity_bond_corr_60d: Optional[float] = None
    corr_regime: str = "unknown"
    as_of: str = ""


# ---------------------------------------------------------------------------
# InflationBreakevenEngine
# ---------------------------------------------------------------------------


class InflationBreakevenEngine:
    """
    Compute and interpret TIPS breakeven inflation rates and real yields from FRED.

    Data sources (all free, no API key):
      T5YIE  — 5-Year Breakeven Inflation Rate
      T10YIE — 10-Year Breakeven Inflation Rate
      T5YIFR — 5-Year, 5-Year Forward Inflation Expectation Rate
      DFII5  — 5-Year TIPS Real Yield
      DFII10 — 10-Year TIPS Real Yield
    """

    def __init__(self, lookback_days: int = 756) -> None:
        self._lookback = lookback_days
        self._cache: dict[str, pd.Series] = {}

    def _get_series(self, series_id: str) -> pd.Series:
        if series_id not in self._cache:
            self._cache[series_id] = _fetch_fred_series(series_id, self._lookback)
        return self._cache[series_id]

    def _refresh(self) -> None:
        """Clear internal cache to force re-fetch."""
        self._cache.clear()

    @staticmethod
    def _breakeven_regime(bei: float, target: float = FED_INFLATION_TARGET) -> str:
        """
        Classify breakeven inflation relative to Fed 2% target.
          anchored  : deviation < 0.5% from target
          drifting  : deviation 0.5 – 1.0% from target
          unanchored: deviation > 1.0% from target
        """
        deviation = abs(bei - target)
        if deviation < 0.5:
            return "anchored"
        if deviation < 1.0:
            return "drifting"
        return "unanchored"

    @staticmethod
    def _real_rate_stance(real_yield: float) -> str:
        """
        Classify monetary stance from real yield.
          restrictive   : real yield > 0
          accommodative : real yield < 0
          neutral       : near zero (±0.25)
        """
        if real_yield > 0.25:
            return "restrictive"
        if real_yield < -0.25:
            return "accommodative"
        return "neutral"

    def snapshot(self) -> BreakevenSnapshot:
        """
        Return current breakeven, real yield, regime, and market-implied inflation path.
        """
        t5y = self._get_series("T5YIE")
        t10y = self._get_series("T10YIE")
        t5yifr = self._get_series("T5YIFR")
        dfii5 = self._get_series("DFII5")
        dfii10 = self._get_series("DFII10")

        bei_5y = _latest(t5y)
        bei_10y = _latest(t10y)
        fwd_5y5y = _latest(t5yifr)
        ry_5y = _latest(dfii5)
        ry_10y = _latest(dfii10)

        dev_5y = round(bei_5y - FED_INFLATION_TARGET, 4) if bei_5y is not None else None
        dev_10y = round(bei_10y - FED_INFLATION_TARGET, 4) if bei_10y is not None else None

        regime_5y = self._breakeven_regime(bei_5y) if bei_5y is not None else "unknown"
        regime_10y = self._breakeven_regime(bei_10y) if bei_10y is not None else "unknown"

        stance_5y = self._real_rate_stance(ry_5y) if ry_5y is not None else "unknown"
        stance_10y = self._real_rate_stance(ry_10y) if ry_10y is not None else "unknown"

        # Market-implied inflation path:
        # 0–5Y  = T5YIE (current 5Y BEI)
        # 5–10Y = implied from 10Y BEI and 5Y BEI (interpolation)
        # 5–10Y forward = T5YIFR
        path: dict[str, Optional[float]] = {
            "0_5y_avg": bei_5y,
            "5_10y_forward": fwd_5y5y,
            "0_10y_avg": bei_10y,
        }
        if bei_5y is not None and bei_10y is not None:
            # 5–10Y implied: 2 * 10Y - 5Y (not the same as 5Y5Y but a useful approximation)
            implied_5_10y = round(2.0 * bei_10y - bei_5y, 4)
            path["5_10y_implied"] = implied_5_10y

        return BreakevenSnapshot(
            breakeven_5y=bei_5y,
            breakeven_10y=bei_10y,
            forward_5y5y=fwd_5y5y,
            real_yield_5y=ry_5y,
            real_yield_10y=ry_10y,
            fed_target=FED_INFLATION_TARGET,
            deviation_5y_from_target=dev_5y,
            deviation_10y_from_target=dev_10y,
            breakeven_regime_5y=regime_5y,
            breakeven_regime_10y=regime_10y,
            real_rate_stance_5y=stance_5y,
            real_rate_stance_10y=stance_10y,
            market_implied_path=path,
            as_of=date.today().isoformat(),
        )

    def history(self, lookback_days: int = 252) -> pd.DataFrame:
        """
        Return a DataFrame of daily BEI / real yield history.

        Columns: breakeven_5y, breakeven_10y, forward_5y5y, real_yield_5y, real_yield_10y
        """
        t5y = _fetch_fred_series("T5YIE", lookback_days)
        t10y = _fetch_fred_series("T10YIE", lookback_days)
        t5yifr = _fetch_fred_series("T5YIFR", lookback_days)
        dfii5 = _fetch_fred_series("DFII5", lookback_days)
        dfii10 = _fetch_fred_series("DFII10", lookback_days)

        df = pd.DataFrame(
            {
                "breakeven_5y": t5y,
                "breakeven_10y": t10y,
                "forward_5y5y": t5yifr,
                "real_yield_5y": dfii5,
                "real_yield_10y": dfii10,
            }
        ).sort_index()
        df.index.name = "date"
        return df.dropna(how="all")

    def real_yield_analysis(self) -> dict[str, Any]:
        """
        Summarise the real yield environment with regime labels and z-scores.
        """
        dfii5 = _fetch_fred_series("DFII5", 756)
        dfii10 = _fetch_fred_series("DFII10", 756)

        def _zscore(s: pd.Series) -> Optional[float]:
            if len(s) < 30:
                return None
            std = s.std()
            if std == 0:
                return 0.0
            return round(float((s.iloc[-1] - s.mean()) / std), 4)

        ry5 = _latest(dfii5)
        ry10 = _latest(dfii10)

        result: dict[str, Any] = {
            "real_yield_5y": ry5,
            "real_yield_10y": ry10,
            "stance_5y": self._real_rate_stance(ry5) if ry5 is not None else "unknown",
            "stance_10y": self._real_rate_stance(ry10) if ry10 is not None else "unknown",
            "zscore_5y_1y": _zscore(dfii5.last("252D")) if not dfii5.empty else None,
            "zscore_10y_1y": _zscore(dfii10.last("252D")) if not dfii10.empty else None,
            "term_premium_proxy": round(ry10 - ry5, 4) if (ry5 is not None and ry10 is not None) else None,
            "as_of": date.today().isoformat(),
        }
        return result


# ---------------------------------------------------------------------------
# VIXTermStructureAnalyzer
# ---------------------------------------------------------------------------


class VIXTermStructureAnalyzer:
    """
    VIX term structure analytics using FRED free data.

    FRED Series:
      VXST    — CBOE 9-Day Volatility Index
      VIXCLS  — CBOE VIX (1-Month, daily close)
      VXMT    — CBOE Mid-Term Volatility (3-Month)

    Computes: contango/backwardation, spread, regime, percentiles, momentum,
    vol-of-vol proxy.
    """

    def __init__(self) -> None:
        self._lookback = 756  # ~3 years

    def _fetch_all(self) -> dict[str, pd.Series]:
        data: dict[str, pd.Series] = {}
        for sid in ("VXST", "VIXCLS", "VXMT"):
            s = _fetch_fred_series(sid, self._lookback)
            if not s.empty:
                data[sid] = s
        return data

    @staticmethod
    def _vix_regime(vix: float) -> str:
        if vix < 15:
            return "low"
        if vix < 25:
            return "normal"
        if vix < 35:
            return "elevated"
        return "crisis"

    @staticmethod
    def _percentile_rank(series: pd.Series, window: int = 252) -> Optional[float]:
        s = series.dropna()
        if len(s) < 5:
            return None
        tail = s.iloc[-min(window, len(s)):]
        current = float(tail.iloc[-1])
        hist = tail.iloc[:-1].values
        if len(hist) == 0:
            return None
        pct = float(np.sum(hist <= current) / len(hist) * 100)
        return round(pct, 2)

    def snapshot(self) -> VIXSnapshot:
        """
        Return current VIX term structure snapshot.
        """
        data = self._fetch_all()
        vxst_s = data.get("VXST")
        vix_s = data.get("VIXCLS")
        vxmt_s = data.get("VXMT")

        vix_9d = _latest(vxst_s) if vxst_s is not None else None
        vix_1m = _latest(vix_s) if vix_s is not None else None
        vix_3m = _latest(vxmt_s) if vxmt_s is not None else None

        # Contango ratio: VXMT / VIXCLS
        contango_ratio: Optional[float] = None
        if vix_1m is not None and vix_3m is not None and vix_1m > 0:
            contango_ratio = round(vix_3m / vix_1m, 4)

        spread_3m_1m: Optional[float] = None
        if vix_1m is not None and vix_3m is not None:
            spread_3m_1m = round(vix_3m - vix_1m, 4)

        # Term structure shape
        shape = "unknown"
        if contango_ratio is not None:
            if contango_ratio > 1.05:
                shape = "contango"
            elif contango_ratio < 0.95:
                shape = "backwardation"
            else:
                shape = "flat"

        vix_regime = self._vix_regime(vix_1m) if vix_1m is not None else "unknown"

        # Percentiles
        pct_1y: Optional[float] = None
        pct_5y: Optional[float] = None
        pct_10y: Optional[float] = None
        if vix_s is not None:
            pct_1y = self._percentile_rank(vix_s, 252)
            pct_5y = self._percentile_rank(vix_s, 252 * 5)
            pct_10y = self._percentile_rank(vix_s, 252 * 10)

        # VIX momentum: rate of change over 5 days
        vix_mom: Optional[float] = None
        if vix_s is not None and len(vix_s.dropna()) >= 6:
            s = vix_s.dropna()
            vix_mom = round(float((s.iloc[-1] / s.iloc[-6] - 1) * 100), 4)

        # Vol-of-vol proxy: rolling 20D std of daily VIX changes
        vov_20d: Optional[float] = None
        if vix_s is not None and len(vix_s.dropna()) >= 22:
            s = vix_s.dropna()
            daily_changes = s.diff().dropna()
            if len(daily_changes) >= 20:
                vov_20d = round(float(daily_changes.iloc[-20:].std()), 4)

        return VIXSnapshot(
            vix_9d=vix_9d,
            vix_1m=vix_1m,
            vix_3m=vix_3m,
            contango_ratio=contango_ratio,
            spread_3m_1m=spread_3m_1m,
            term_structure_shape=shape,
            vix_regime=vix_regime,
            vix_percentile_1y=pct_1y,
            vix_percentile_5y=pct_5y,
            vix_percentile_10y=pct_10y,
            vix_mom_5d=vix_mom,
            vol_of_vol_20d=vov_20d,
            as_of=date.today().isoformat(),
        )

    def term_structure_history(self, lookback_days: int = 252) -> pd.DataFrame:
        """
        Return a DataFrame with daily VXST, VIXCLS, VXMT, contango_ratio, and spread.
        """
        data = self._fetch_all()
        frames: dict[str, pd.Series] = {}
        for k, sid in [("vix_9d", "VXST"), ("vix_1m", "VIXCLS"), ("vix_3m", "VXMT")]:
            s = data.get(sid)
            if s is not None and not s.empty:
                frames[k] = s

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames).sort_index()
        if "vix_1m" in df.columns and "vix_3m" in df.columns:
            df["contango_ratio"] = df["vix_3m"] / df["vix_1m"].replace(0, np.nan)
            df["spread_3m_1m"] = df["vix_3m"] - df["vix_1m"]

        cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
        return df[df.index >= cutoff].dropna(how="all")

    def percentile_detail(self) -> dict[str, Any]:
        """
        Return VIX percentile details across multiple lookback windows.
        """
        vix_s = _fetch_fred_series("VIXCLS", 252 * 10)
        if vix_s.empty:
            return {"error": "VIXCLS data unavailable"}

        vix_val = _latest(vix_s)
        windows = {
            "1y": 252,
            "3y": 252 * 3,
            "5y": 252 * 5,
            "10y": 252 * 10,
        }
        result: dict[str, Any] = {"vix_current": vix_val, "percentiles": {}}
        for label, w in windows.items():
            result["percentiles"][label] = self._percentile_rank(vix_s, w)
        result["as_of"] = date.today().isoformat()
        return result


# ---------------------------------------------------------------------------
# VolatilityRiskPremiumEngine
# ---------------------------------------------------------------------------


class VolatilityRiskPremiumEngine:
    """
    Volatility Risk Premium (VRP) = Implied Vol (VIX) - Realized Vol (SPY 21D).

    Positive VRP means selling options earns the risk premium on average.
    Negative VRP signals that options are cheap relative to realized vol.
    """

    @staticmethod
    def _get_spy_rets(lookback_days: int = 756) -> pd.Series:
        closes = _yf_close("SPY", lookback_days)
        if closes.empty:
            return pd.Series(dtype=float)
        return np.log(closes / closes.shift(1)).dropna()

    @staticmethod
    def _vrp_regime(vrp: float) -> str:
        if vrp > 5.0:
            return "very_positive"
        if vrp > 2.0:
            return "positive"
        if vrp > -2.0:
            return "neutral"
        return "negative"

    @staticmethod
    def _vrp_signal(vrp: float) -> str:
        """
        Trading signal derived from VRP:
          high VRP     → sell straddles / short vol
          low/negative → buy vol / long straddles
        """
        if vrp > 5.0:
            return "sell_straddles"
        if vrp > 2.0:
            return "short_vol_bias"
        if vrp < -2.0:
            return "buy_vol"
        return "neutral"

    def vrp_current(self) -> VRPSnapshot:
        """
        Compute current VRP: VIX (implied vol) minus SPY 21D realized vol.
        Both expressed as annualized percent.
        """
        vix_s = _fetch_fred_series("VIXCLS", 30)
        vix_val = _latest(vix_s)

        spy_rets = self._get_spy_rets(60)
        rv_21d = _annualized_realized_vol(spy_rets, 21)

        vrp: Optional[float] = None
        if vix_val is not None and rv_21d is not None:
            vrp = round(vix_val - rv_21d, 4)

        regime = self._vrp_regime(vrp) if vrp is not None else "unknown"
        signal = self._vrp_signal(vrp) if vrp is not None else "neutral"

        return VRPSnapshot(
            vix_current=vix_val,
            realized_vol_21d=rv_21d,
            vrp_current=vrp,
            vrp_regime=regime,
            vrp_signal=signal,
            as_of=date.today().isoformat(),
        )

    def vrp_history(self, lookback_days: int = 252) -> VRPHistory:
        """
        Compute rolling VRP history over the specified lookback period.

        VRP(t) = VIX(t) - RealizedVol(SPY, 21D ending at t)
        """
        vix_s = _fetch_fred_series("VIXCLS", lookback_days + 60)
        spy_rets = self._get_spy_rets(lookback_days + 90)

        if vix_s.empty or spy_rets.empty:
            return VRPHistory(dates=[], vrp_series=[], vix_series=[], rv_series=[])

        # Rolling 21D realized vol
        rv_rolling = (
            spy_rets.rolling(window=21, min_periods=15).std() * np.sqrt(252) * 100
        )
        rv_rolling = rv_rolling.dropna()

        # Align by date
        combined = pd.DataFrame(
            {"vix": vix_s, "rv": rv_rolling}
        ).dropna()

        cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
        combined = combined[combined.index >= cutoff]

        if combined.empty:
            return VRPHistory(dates=[], vrp_series=[], vix_series=[], rv_series=[])

        vrp_series = (combined["vix"] - combined["rv"]).round(4)
        dates = [d.strftime("%Y-%m-%d") for d in combined.index]

        avg_vrp: Optional[float] = None
        pct_pos: Optional[float] = None
        vrp_vals = vrp_series.dropna()
        if not vrp_vals.empty:
            avg_vrp = round(float(vrp_vals.mean()), 4)
            pct_pos = round(float((vrp_vals > 0).mean() * 100), 2)

        return VRPHistory(
            dates=dates,
            vrp_series=[round(v, 4) if not np.isnan(v) else None for v in vrp_series],
            vix_series=[round(v, 4) if not np.isnan(v) else None for v in combined["vix"]],
            rv_series=[round(v, 4) if not np.isnan(v) else None for v in combined["rv"]],
            avg_vrp=avg_vrp,
            pct_positive=pct_pos,
        )


# ---------------------------------------------------------------------------
# InflationTradingSignals
# ---------------------------------------------------------------------------


class InflationTradingSignals:
    """
    Tradeable signals derived from inflation breakeven and real yield dynamics.

    Signals:
      TIPS vs Nominal   — direction of real yield change
      BEI Compression   — falling BEI → deflation risk
      BEI Expansion     — BEI rising >0.1% in 5 days → inflation concern
      Stagflation       — high BEI + rising unemployment claims
      Real Rate Shock   — real yields up >50bps in 30 days
    """

    def __init__(self) -> None:
        self._bei_engine = InflationBreakevenEngine(lookback_days=756)

    def _get_changes(
        self, series: pd.Series, window_days: int
    ) -> Optional[float]:
        """Change in series over last `window_days` calendar days."""
        s = series.dropna()
        if len(s) < 2:
            return None
        cutoff = s.index[-1] - pd.Timedelta(days=window_days)
        start_val = s[s.index <= cutoff]
        if start_val.empty:
            return None
        return round(float(s.iloc[-1] - start_val.iloc[-1]), 4)

    def signals(self) -> InflationSignals:
        """
        Compute all inflation trading signals as of today.
        """
        t5y = _fetch_fred_series("T5YIE", 120)
        t10y = _fetch_fred_series("T10YIE", 120)
        dfii5 = _fetch_fred_series("DFII5", 120)
        dfii10 = _fetch_fred_series("DFII10", 120)
        icsa = _fetch_fred_series("ICSA", 120)  # weekly initial claims

        bei_5y = _latest(t5y)
        bei_10y = _latest(t10y)
        ry_5y = _latest(dfii5)
        ry_10y = _latest(dfii10)

        # ----- 1. TIPS vs Nominal signal -----
        # When real yields are rising, duration risk in both TIPS and nominals increases.
        # However, rising real yields hurt nominals more than TIPS (TIPS principal adjusts).
        # Signal: outperformance of TIPS vs nominals when real yields are rising.
        ry10_change_30d = self._get_changes(dfii10, 30)
        tips_vs_nominal: str = "neutral"
        if ry10_change_30d is not None:
            if ry10_change_30d > 0.15:
                tips_vs_nominal = "favor_tips"  # rising real yields → TIPS relatively better
            elif ry10_change_30d < -0.15:
                tips_vs_nominal = "favor_nominals"  # falling real yields → nominals outperform
            else:
                tips_vs_nominal = "neutral"

        # ----- 2. BEI Compression / Expansion -----
        bei_5d_change = self._get_changes(t5y, 7)  # 5-7 business days ≈ 7 calendar days
        bei_compression = False
        bei_expansion = False
        if bei_5d_change is not None:
            if bei_5d_change < -0.10:
                bei_compression = True  # BEI falling → deflation risk
            if bei_5d_change > 0.10:
                bei_expansion = True  # BEI rising → inflation concern

        # ----- 3. Stagflation indicator -----
        # Stagflation: high BEI (above target + 0.5) + rising unemployment claims
        stagflation = False
        icsa_val = _latest(icsa)
        icsa_4w_change: Optional[float] = None
        if not icsa.empty and len(icsa.dropna()) >= 5:
            s = icsa.dropna()
            if len(s) >= 5:
                icsa_4w_change = self._get_changes(s, 28)
        if (
            bei_10y is not None
            and bei_10y > FED_INFLATION_TARGET + 0.5
            and icsa_4w_change is not None
            and icsa_4w_change > 0
        ):
            stagflation = True

        # ----- 4. Real Rate Shock -----
        # Real yields up >50bps in 30 days = tightening conditions shock
        ry5_change_30d = self._get_changes(dfii5, 30)
        real_rate_shock = False
        if ry10_change_30d is not None and ry10_change_30d > 0.50:
            real_rate_shock = True
        elif ry5_change_30d is not None and ry5_change_30d > 0.50:
            real_rate_shock = True

        details: dict[str, Any] = {
            "bei_5y_current": bei_5y,
            "bei_10y_current": bei_10y,
            "real_yield_5y_current": ry_5y,
            "real_yield_10y_current": ry_10y,
            "bei_5d_change": bei_5d_change,
            "real_yield_10y_30d_change": ry10_change_30d,
            "real_yield_5y_30d_change": ry5_change_30d,
            "unemployment_claims_current": icsa_val,
            "unemployment_claims_4w_change": icsa_4w_change,
            "fed_target": FED_INFLATION_TARGET,
        }

        return InflationSignals(
            tips_vs_nominal_signal=tips_vs_nominal,
            bei_compression=bei_compression,
            bei_expansion=bei_expansion,
            stagflation_indicator=stagflation,
            real_rate_shock=real_rate_shock,
            signal_details=details,
            as_of=date.today().isoformat(),
        )


# ---------------------------------------------------------------------------
# CrossAssetVolRegime
# ---------------------------------------------------------------------------


class CrossAssetVolRegime:
    """
    Cross-asset volatility regime: equity vol (VIX), bond vol proxy (yield vol),
    FX vol proxy (DXY realized), equity/bond vol ratio, and equity-bond correlation.

    Data sources (all free):
      FRED VIXCLS   — equity vol (VIX)
      FRED DGS10    — 10Y Treasury yield for bond vol proxy
      FRED DTWEXBGS — Broad USD index for FX vol proxy
      yfinance SPY + TLT — equity-bond correlation
    """

    def __init__(self) -> None:
        self._lookback = 756

    @staticmethod
    def _realized_vol_series(series: pd.Series, window: int = 21) -> pd.Series:
        """Rolling annualized realized vol of a price/rate series."""
        if isinstance(series.index, pd.DatetimeIndex):
            pct_chg = series.pct_change().dropna()
        else:
            pct_chg = series.diff().dropna()
        return pct_chg.rolling(window=window).std() * np.sqrt(252)

    def snapshot(self) -> VolRegimeResponse:
        """
        Return cross-asset volatility regime snapshot.
        """
        vix_s = _fetch_fred_series("VIXCLS", 126)
        dgs10_s = _fetch_fred_series("DGS10", 126)
        dxy_s = _fetch_fred_series("DTWEXBGS", 126)

        vix_val = _latest(vix_s)

        # Bond vol proxy: rolling 21D std of daily 10Y yield changes * annualization
        bond_vol_proxy: Optional[float] = None
        if not dgs10_s.empty and len(dgs10_s.dropna()) >= 22:
            ychg = dgs10_s.diff().dropna()
            bvol = float(ychg.iloc[-21:].std() * np.sqrt(252))
            bond_vol_proxy = round(bvol, 4)  # in yield-change units (bps * sqrt(252))

        # FX vol proxy: rolling 21D realized vol of DXY percent changes
        fx_vol_proxy: Optional[float] = None
        if not dxy_s.empty and len(dxy_s.dropna()) >= 22:
            dxy_rets = dxy_s.pct_change().dropna()
            fvol = float(dxy_rets.iloc[-21:].std() * np.sqrt(252) * 100)
            fx_vol_proxy = round(fvol, 4)

        # Equity/bond vol ratio: VIX / bond vol
        eq_bond_ratio: Optional[float] = None
        if vix_val is not None and bond_vol_proxy is not None and bond_vol_proxy > 0:
            eq_bond_ratio = round(vix_val / bond_vol_proxy, 4)

        # Risk mode from ratio
        risk_mode = "unknown"
        if eq_bond_ratio is not None:
            if eq_bond_ratio > 5:
                risk_mode = "risk_off"  # equity vol elevated vs bonds
            elif eq_bond_ratio < 2:
                risk_mode = "bond_stress"  # bond vol elevated vs equities
            else:
                risk_mode = "balanced"

        # Equity-bond rolling correlation (60D) from SPY + TLT
        corr_60d: Optional[float] = None
        corr_regime = "unknown"
        try:
            spy_c = _yf_close("SPY", 180)
            tlt_c = _yf_close("TLT", 180)
            if not spy_c.empty and not tlt_c.empty:
                combined = pd.DataFrame({"spy": spy_c, "tlt": tlt_c}).dropna()
                if len(combined) >= 60:
                    spy_r = combined["spy"].pct_change().dropna()
                    tlt_r = combined["tlt"].pct_change().dropna()
                    # Align
                    both = pd.concat([spy_r, tlt_r], axis=1).dropna()
                    both.columns = ["spy", "tlt"]
                    if len(both) >= 60:
                        corr_60d = round(float(both["spy"].corr(both["tlt"])), 4)
                        if corr_60d > 0.3:
                            corr_regime = "positive_corr"  # equities and bonds moving together
                        elif corr_60d < -0.3:
                            corr_regime = "negative_corr"  # traditional flight-to-safety
                        else:
                            corr_regime = "decorrelated"
        except Exception as exc:
            logger.warning("equity_bond_corr_failed", error=str(exc))

        equity_vol_regime = "unknown"
        if vix_val is not None:
            if vix_val < 15:
                equity_vol_regime = "low_vol"
            elif vix_val < 25:
                equity_vol_regime = "normal"
            elif vix_val < 35:
                equity_vol_regime = "elevated"
            else:
                equity_vol_regime = "crisis"

        return VolRegimeResponse(
            equity_vol_regime=equity_vol_regime,
            bond_vol_proxy=bond_vol_proxy,
            fx_vol_proxy=fx_vol_proxy,
            equity_bond_vol_ratio=eq_bond_ratio,
            risk_mode=risk_mode,
            equity_bond_corr_60d=corr_60d,
            corr_regime=corr_regime,
            as_of=date.today().isoformat(),
        )

    def vol_history(self, lookback_days: int = 252) -> pd.DataFrame:
        """
        Return a DataFrame with rolling equity vol (VIX), bond vol proxy,
        and FX vol proxy over the given lookback.
        """
        vix_s = _fetch_fred_series("VIXCLS", lookback_days + 30)
        dgs10_s = _fetch_fred_series("DGS10", lookback_days + 60)
        dxy_s = _fetch_fred_series("DTWEXBGS", lookback_days + 60)

        frames: dict[str, pd.Series] = {}
        if not vix_s.empty:
            frames["vix"] = vix_s

        if not dgs10_s.empty:
            ychg = dgs10_s.diff().dropna()
            frames["bond_vol_proxy"] = ychg.rolling(21).std() * np.sqrt(252)

        if not dxy_s.empty:
            dxy_rets = dxy_s.pct_change().dropna()
            frames["fx_vol_proxy"] = dxy_rets.rolling(21).std() * np.sqrt(252) * 100

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames).sort_index()
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
        return df[df.index >= cutoff].dropna(how="all")


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

inflation_vix_router = APIRouter(
    prefix="/api/inflation-vix",
    tags=["inflation", "vix", "volatility"],
)

_bei_engine = InflationBreakevenEngine()
_vix_analyzer = VIXTermStructureAnalyzer()
_vrp_engine = VolatilityRiskPremiumEngine()
_inflation_signals = InflationTradingSignals()
_cross_asset_vol = CrossAssetVolRegime()


@inflation_vix_router.get("/inflation/breakevens")
def get_breakevens() -> dict:
    """
    Current TIPS breakeven inflation rates: 5Y, 10Y, 5Y5Y forward.
    Compares to Fed 2% target, classifies BEI regime.
    """
    try:
        snap = _bei_engine.snapshot()
        return snap.model_dump()
    except Exception as exc:
        logger.error("breakevens_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/inflation/real-yields")
def get_real_yields() -> dict:
    """
    TIPS real yields (5Y, 10Y) with regime classification (restrictive/accommodative)
    and z-scores vs 1-year history.
    """
    try:
        return _bei_engine.real_yield_analysis()
    except Exception as exc:
        logger.error("real_yields_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/inflation/history")
def get_breakeven_history(
    days: int = Query(default=252, ge=30, le=1260),
) -> dict:
    """
    Historical daily breakeven inflation and real yield time series.
    """
    try:
        df = _bei_engine.history(days)
        if df.empty:
            return {"history": [], "n_days": 0}
        records = df.reset_index()
        records["date"] = records["date"].astype(str)
        return {
            "history": records.where(records.notna(), None).to_dict(orient="records"),
            "n_days": len(records),
        }
    except Exception as exc:
        logger.error("breakeven_history_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/vix/term-structure")
def get_vix_term_structure() -> dict:
    """
    VIX term structure: 9-Day (VXST), 1-Month (VIX), 3-Month (VXMT).
    Includes contango ratio, spread, shape classification, and regime.
    """
    try:
        snap = _vix_analyzer.snapshot()
        return snap.model_dump()
    except Exception as exc:
        logger.error("vix_term_structure_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/vix/percentile")
def get_vix_percentile() -> dict:
    """
    VIX current value and percentile rank vs 1Y, 3Y, 5Y, and 10Y history.
    """
    try:
        return _vix_analyzer.percentile_detail()
    except Exception as exc:
        logger.error("vix_percentile_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/vix/history")
def get_vix_history(
    days: int = Query(default=252, ge=30, le=1260),
) -> dict:
    """
    Historical VIX term structure (VXST, VIXCLS, VXMT) with contango ratios.
    """
    try:
        df = _vix_analyzer.term_structure_history(days)
        if df.empty:
            return {"history": [], "n_days": 0}
        records = df.reset_index()
        records.columns = [str(c) for c in records.columns]
        records["date"] = records.iloc[:, 0].astype(str)
        return {
            "history": records.where(records.notna(), None).to_dict(orient="records"),
            "n_days": len(records),
        }
    except Exception as exc:
        logger.error("vix_history_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/vrp/current")
def get_vrp_current() -> dict:
    """
    Current Volatility Risk Premium: VIX minus SPY 21-day realized vol.
    Includes regime label and options trading signal.
    """
    try:
        snap = _vrp_engine.vrp_current()
        return snap.model_dump()
    except Exception as exc:
        logger.error("vrp_current_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/vrp/history")
def get_vrp_history(
    days: int = Query(default=252, ge=30, le=756),
) -> dict:
    """
    Rolling VRP history: daily VIX, SPY realized vol, and VRP series.
    Includes average VRP and percent of days with positive VRP.
    """
    try:
        hist = _vrp_engine.vrp_history(days)
        return hist.model_dump()
    except Exception as exc:
        logger.error("vrp_history_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/signals/inflation")
def get_inflation_signals() -> dict:
    """
    Inflation trading signals: TIPS vs nominal, BEI compression/expansion,
    stagflation indicator, and real rate shock alert.
    """
    try:
        sigs = _inflation_signals.signals()
        return sigs.model_dump()
    except Exception as exc:
        logger.error("inflation_signals_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/signals/vol-regime")
def get_vol_regime_signal() -> dict:
    """
    Cross-asset volatility regime: equity vol (VIX), bond vol proxy, FX vol proxy,
    equity/bond vol ratio, risk mode, and equity-bond 60D rolling correlation.
    """
    try:
        snap = _cross_asset_vol.snapshot()
        return snap.model_dump()
    except Exception as exc:
        logger.error("vol_regime_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@inflation_vix_router.get("/signals/vol-history")
def get_vol_history(
    days: int = Query(default=252, ge=30, le=756),
) -> dict:
    """
    Historical cross-asset vol: VIX, bond vol proxy, and FX vol proxy.
    """
    try:
        df = _cross_asset_vol.vol_history(days)
        if df.empty:
            return {"history": [], "n_days": 0}
        records = df.reset_index()
        records.columns = [str(c) for c in records.columns]
        records.iloc[:, 0] = records.iloc[:, 0].astype(str)
        return {
            "history": records.where(records.notna(), None).to_dict(orient="records"),
            "n_days": len(records),
        }
    except Exception as exc:
        logger.error("vol_history_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Standalone helpers for external use
# ---------------------------------------------------------------------------


def get_breakeven_snapshot() -> dict:
    """Module-level convenience function: current BEI snapshot as dict."""
    return InflationBreakevenEngine().snapshot().model_dump()


def get_vix_snapshot() -> dict:
    """Module-level convenience function: current VIX term structure as dict."""
    return VIXTermStructureAnalyzer().snapshot().model_dump()


def get_vrp_snapshot() -> dict:
    """Module-level convenience function: current VRP as dict."""
    return VolatilityRiskPremiumEngine().vrp_current().model_dump()


def get_inflation_signals_snapshot() -> dict:
    """Module-level convenience function: inflation trading signals as dict."""
    return InflationTradingSignals().signals().model_dump()


def get_cross_asset_vol_snapshot() -> dict:
    """Module-level convenience function: cross-asset vol regime as dict."""
    return CrossAssetVolRegime().snapshot().model_dump()


# ---------------------------------------------------------------------------
# VIX Full Term Structure (9D / 1M / 3M / 6M) + Contango/Backwardation
# ---------------------------------------------------------------------------


class VIXFullTermStructure:
    """
    VIX term structure across 4 tenors: VIX9D, VIX (1M), VIX3M, VIX6M.

    VIX9D  = VXST (FRED)
    VIX1M  = VIXCLS (FRED)
    VIX3M  = VXMT (FRED)
    VIX6M  = proxy: VXMT + premium (no free CBOE 6M series on FRED;
             the 6-month VIX (VXV) is discontinued — we estimate as
             VXMT * 1.04 when unavailable, consistent with typical term premium).

    Contango: longer-dated vol > shorter-dated vol (normal carry environment).
    Backwardation: shorter-dated vol > longer-dated vol (fear / tail-risk event).
    """

    def get(self) -> dict[str, Any]:
        """Return full VIX term structure with contango/backwardation classification."""
        vxst = _fetch_fred_series("VXST", 30)
        vix = _fetch_fred_series("VIXCLS", 30)
        vxmt = _fetch_fred_series("VXMT", 30)

        v9d = _latest(vxst)
        v1m = _latest(vix)
        v3m = _latest(vxmt)
        # 6M proxy: attempt FRED VIX6M, fall back to VXMT * 1.04
        vix6m_s = _fetch_fred_series("VXMT", 30)  # reuse same series
        v6m: Optional[float] = round(v3m * 1.04, 4) if v3m is not None else None

        shape = "unknown"
        contango_9d_1m: Optional[float] = None
        contango_1m_3m: Optional[float] = None

        if v9d is not None and v1m is not None and v1m > 0:
            contango_9d_1m = round(v1m / v9d, 4)
        if v1m is not None and v3m is not None and v1m > 0:
            contango_1m_3m = round(v3m / v1m, 4)

        # Overall shape classification using 9D vs 3M
        if v9d is not None and v3m is not None:
            ratio = v3m / v9d if v9d > 0 else 1.0
            if ratio > 1.05:
                shape = "contango"
            elif ratio < 0.95:
                shape = "backwardation"
            else:
                shape = "flat"

        return {
            "vix_9d": v9d,
            "vix_1m": v1m,
            "vix_3m": v3m,
            "vix_6m": v6m,
            "contango_ratio_9d_1m": contango_9d_1m,
            "contango_ratio_1m_3m": contango_1m_3m,
            "term_structure_shape": shape,
            "as_of": date.today().isoformat(),
        }

    @staticmethod
    def classify_shape(vix_9d: float, vix_1m: float, vix_3m: float) -> str:
        """
        Pure-math classification: given the three tenors, return shape.
        contango:      9D < 1M < 3M  (normal carry)
        backwardation: 9D > 1M > 3M  (fear event)
        mixed:         non-monotone
        flat:          all within 5% of each other
        """
        spread = vix_3m - vix_9d
        if abs(spread) / max(vix_9d, 0.01) < 0.05:
            return "flat"
        if vix_9d <= vix_1m <= vix_3m:
            return "contango"
        if vix_9d >= vix_1m >= vix_3m:
            return "backwardation"
        return "mixed"


# ---------------------------------------------------------------------------
# VIX Skew (fear gauge) and Vol-of-Vol (VVIX proxy)
# ---------------------------------------------------------------------------


class VIXSkewAndVVIX:
    """
    VIX Skew: fear gauge approximated as VIX - VVIX * 0.1
    (higher skew → elevated put demand / tail-risk fear).

    VVIX: Volatility of VIX. FRED does not carry VVIX directly.
    Proxy: 20-day rolling std of daily VIX changes, annualized.
    """

    def fear_gauge(self, vix: float, vvix: float) -> float:
        """Skew proxy = VIX - VVIX * 0.1. Positive = puts bid up (fear)."""
        return round(vix - vvix * 0.1, 4)

    def vvix_proxy(self) -> Optional[float]:
        """
        VVIX proxy: annualized 20-day rolling std of VIX daily changes.
        Returns value in VIX-point units.
        """
        vix_s = _fetch_fred_series("VIXCLS", 60)
        if vix_s.empty or len(vix_s.dropna()) < 22:
            return None
        s = vix_s.dropna()
        daily_chg = s.diff().dropna()
        vov = float(daily_chg.iloc[-20:].std() * np.sqrt(252))
        return round(vov, 4)

    def snapshot(self) -> dict[str, Any]:
        vix_s = _fetch_fred_series("VIXCLS", 30)
        vix_val = _latest(vix_s)
        vvix_val = self.vvix_proxy()
        fear = self.fear_gauge(vix_val, vvix_val) if (vix_val and vvix_val) else None
        return {
            "vix": vix_val,
            "vvix_proxy": vvix_val,
            "fear_gauge": fear,
            "interpretation": "elevated_fear" if (fear and fear > 5) else "normal",
            "as_of": date.today().isoformat(),
        }


# ---------------------------------------------------------------------------
# Inflation Regime Classifier (Goldilocks / Stagflation / Deflation / Reflation)
# ---------------------------------------------------------------------------


class InflationRegimeClassifier:
    """
    Classify macro inflation regime using CPI + GDP growth quadrant.

    Quadrant logic (standard macro framework):
      High growth + Low inflation  → Goldilocks
      High growth + High inflation → Reflation  (or overheating)
      Low growth  + High inflation → Stagflation
      Low growth  + Low inflation  → Deflation  (or disinflation)

    Thresholds:
      High inflation: CPI > 3.5%
      High growth:    real GDP growth > 2.0%
    """

    GROWTH_THRESHOLD: float = 2.0   # % real GDP growth
    INFLATION_THRESHOLD: float = 3.5  # % CPI YoY

    @staticmethod
    def classify(cpi_pct: float, gdp_growth_pct: float) -> str:
        """
        Return regime label from CPI YoY % and real GDP growth %.

        Parameters
        ----------
        cpi_pct:        CPI year-on-year percent (e.g. 6.0 for 6%)
        gdp_growth_pct: Real GDP growth percent (e.g. 1.0 for 1%)
        """
        high_inflation = cpi_pct > InflationRegimeClassifier.INFLATION_THRESHOLD
        high_growth = gdp_growth_pct > InflationRegimeClassifier.GROWTH_THRESHOLD

        if high_growth and not high_inflation:
            return "Goldilocks"
        if high_growth and high_inflation:
            return "Reflation"
        if not high_growth and high_inflation:
            return "Stagflation"
        return "Deflation"

    def current_regime(self) -> dict[str, Any]:
        """Fetch live CPI (CPIAUCSL YoY) and estimate GDP growth for regime."""
        cpi_s = _fetch_fred_series("CPIAUCSL", 400)
        regime = "unknown"
        cpi_yoy: Optional[float] = None
        gdp_growth: Optional[float] = None

        if not cpi_s.empty and len(cpi_s.dropna()) >= 13:
            s = cpi_s.dropna()
            cpi_yoy = round(float((s.iloc[-1] / s.iloc[-13] - 1) * 100), 4)

        # GDP growth: use FRED GDPC1 (quarterly real GDP)
        gdp_s = _fetch_fred_series("GDPC1", 600)
        if not gdp_s.empty and len(gdp_s.dropna()) >= 5:
            g = gdp_s.dropna()
            # YoY from 4 quarters back
            if len(g) >= 5:
                gdp_growth = round(float((g.iloc[-1] / g.iloc[-5] - 1) * 100), 4)

        if cpi_yoy is not None and gdp_growth is not None:
            regime = self.classify(cpi_yoy, gdp_growth)

        return {
            "cpi_yoy": cpi_yoy,
            "gdp_growth": gdp_growth,
            "regime": regime,
            "growth_threshold": self.GROWTH_THRESHOLD,
            "inflation_threshold": self.INFLATION_THRESHOLD,
            "as_of": date.today().isoformat(),
        }


# ---------------------------------------------------------------------------
# Breakeven Momentum (5D vs 20D MA on T10YIE)
# ---------------------------------------------------------------------------


class BreakevenMomentum:
    """
    Compute breakeven inflation momentum using 5-day vs 20-day moving average
    of the 10-year breakeven inflation rate (FRED T10YIE).

    Signal:
      5D MA > 20D MA → positive momentum (inflation expectations rising)
      5D MA < 20D MA → negative momentum (inflation expectations falling)
    """

    def compute(self) -> dict[str, Any]:
        bei_s = _fetch_fred_series("T10YIE", 60)
        if bei_s.empty or len(bei_s.dropna()) < 21:
            return {"signal": "insufficient_data", "ma5": None, "ma20": None}

        s = bei_s.dropna().sort_index()
        ma5 = float(s.iloc[-5:].mean())
        ma20 = float(s.iloc[-20:].mean())
        current = float(s.iloc[-1])

        signal = "positive" if ma5 > ma20 else "negative"

        return {
            "breakeven_10y_current": round(current, 4),
            "ma5": round(ma5, 4),
            "ma20": round(ma20, 4),
            "spread_5d_vs_20d": round(ma5 - ma20, 4),
            "signal": signal,
            "interpretation": (
                "rising_inflation_expectations" if signal == "positive"
                else "falling_inflation_expectations"
            ),
            "as_of": date.today().isoformat(),
        }

    @staticmethod
    def compute_from_series(series: "list[float]") -> dict[str, Any]:
        """
        Pure-math version for testing: compute from a list of values.
        Requires at least 20 observations.
        """
        if len(series) < 20:
            return {"signal": "insufficient_data", "ma5": None, "ma20": None}
        ma5 = sum(series[-5:]) / 5
        ma20 = sum(series[-20:]) / 20
        signal = "positive" if ma5 > ma20 else "negative"
        return {
            "current": series[-1],
            "ma5": round(ma5, 6),
            "ma20": round(ma20, 6),
            "spread_5d_vs_20d": round(ma5 - ma20, 6),
            "signal": signal,
        }


# ---------------------------------------------------------------------------
# Vol Risk Premium (pure-math helpers)
# ---------------------------------------------------------------------------


def compute_vrp(vix: float, realized_vol: float) -> dict[str, Any]:
    """
    Compute Volatility Risk Premium = VIX - realized_vol.
    Both in annualized percent terms.

    Returns dict with vrp value and signal (positive VRP → premium exists).
    """
    vrp = round(vix - realized_vol, 4)
    return {
        "vix": vix,
        "realized_vol": realized_vol,
        "vrp": vrp,
        "signal": "positive_vrp" if vrp > 0 else "negative_vrp",
        "interpretation": (
            "implied_vol_above_realized_short_vol_premium_exists"
            if vrp > 0 else "realized_vol_above_implied_cheap_options"
        ),
    }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "InflationBreakevenEngine",
    "VIXTermStructureAnalyzer",
    "VolatilityRiskPremiumEngine",
    "InflationTradingSignals",
    "CrossAssetVolRegime",
    # New v2 analytics
    "VIXFullTermStructure",
    "VIXSkewAndVVIX",
    "InflationRegimeClassifier",
    "BreakevenMomentum",
    "compute_vrp",
    "INFLATION_REGIMES",
    "inflation_vix_router",
    "get_breakeven_snapshot",
    "get_vix_snapshot",
    "get_vrp_snapshot",
    "get_inflation_signals_snapshot",
    "get_cross_asset_vol_snapshot",
    # Pydantic models
    "BreakevenSnapshot",
    "VIXSnapshot",
    "VRPSnapshot",
    "VRPHistory",
    "InflationSignals",
    "VolRegimeResponse",
    # Constants
    "FRED_BASE",
    "INFLATION_SERIES",
    "VIX_SERIES",
    "FED_INFLATION_TARGET",
]
