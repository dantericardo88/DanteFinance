"""
Credit spread analysis and Merton structural credit model.

Dimension #039 — Credit spread analysis / Merton model (target: 9).

Free data sources:
  - FRED CSV: ICE BofA OAS indices — BAMLC0A0CM (IG), BAMLH0A0HYM2 (HY),
    BAMLC0A4CBBB (BBB), BAMLH0A3HYC (CCC), BAMLH0A1HYBB (BB)
  - FRED CSV: Treasury rates for risk-free inputs
  - EDGAR XBRL financials for per-company default risk inputs

Covers:
  IG/HY spreads, spread regime classification, spread curve, Merton structural
  PD, KMV distance-to-default, Altman Z-Score, Ohlson O-Score, composite
  default risk score, credit signal engine, sector stress screening.
"""
from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Literal, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import fsolve
from scipy.stats import norm

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_BASE = "https://api.stlouisfed.org/fred"
EDGAR_BASE = "https://data.sec.gov"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/html,application/json,*/*",
}

# FRED series for credit spreads (all free, no API key needed via CSV endpoint)
CREDIT_SPREAD_SERIES: Dict[str, Dict] = {
    "IG_OAS": {
        "series_id": "BAMLC0A0CM",
        "name": "ICE BofA US Investment Grade OAS",
        "rating": "IG",
        "description": "Option-Adjusted Spread, all maturities, investment grade",
    },
    "HY_OAS": {
        "series_id": "BAMLH0A0HYM2",
        "name": "ICE BofA US High Yield OAS",
        "rating": "HY",
        "description": "Option-Adjusted Spread, all maturities, high yield",
    },
    "BBB_OAS": {
        "series_id": "BAMLC0A4CBBB",
        "name": "ICE BofA BBB US Corporate OAS",
        "rating": "BBB",
        "description": "BBB-rated IG corporates OAS",
    },
    "BB_OAS": {
        "series_id": "BAMLH0A1HYBB",
        "name": "ICE BofA BB US High Yield OAS",
        "rating": "BB",
        "description": "BB-rated (crossover) OAS — fallen angel watch",
    },
    "CCC_OAS": {
        "series_id": "BAMLH0A3HYC",
        "name": "ICE BofA CCC & Lower US HY OAS",
        "rating": "CCC",
        "description": "CCC-rated distressed/near-default OAS",
    },
    "AAA_OAS": {
        "series_id": "BAMLC0A1CAAA",
        "name": "ICE BofA AAA US Corporate OAS",
        "rating": "AAA",
        "description": "Highest quality IG OAS",
    },
    "IG_1_3Y": {
        "series_id": "BAMLC1A0C13Y",
        "name": "ICE BofA 1-3Y IG OAS",
        "rating": "IG",
        "description": "Short-end IG spread (1-3 year maturity)",
    },
    "IG_7_10Y": {
        "series_id": "BAMLC4A0C710Y",
        "name": "ICE BofA 7-10Y IG OAS",
        "rating": "IG",
        "description": "Long-end IG spread (7-10 year maturity)",
    },
    "RF_1Y": {
        "series_id": "DGS1",
        "name": "1-Year Treasury CMT",
        "rating": "RF",
        "description": "Risk-free rate 1Y",
    },
    "RF_5Y": {
        "series_id": "DGS5",
        "name": "5-Year Treasury CMT",
        "rating": "RF",
        "description": "Risk-free rate 5Y",
    },
}

# Spread regime thresholds (approximate historical percentiles)
REGIME_THRESHOLDS = {
    "IG_OAS": {"tight": 80, "normal_high": 130, "wide": 180, "crisis": 280},
    "HY_OAS": {"tight": 300, "normal_high": 500, "wide": 700, "crisis": 1000},
    "BBB_OAS": {"tight": 100, "normal_high": 160, "wide": 220, "crisis": 350},
}

SpreadRegime = Literal["tight", "normal", "wide", "crisis"]
SignalDirection = Literal["buy", "sell", "hold", "watch"]
DefaultRiskBand = Literal["minimal", "low", "moderate", "elevated", "high", "distressed"]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class SpreadSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    series_key: str
    series_id: str
    name: str
    rating: str
    as_of: date
    current_bps: float
    # Changes
    dod_bps: Optional[float] = None       # day-over-day change
    wow_bps: Optional[float] = None       # week-over-week change
    mom_bps: Optional[float] = None       # month-over-month change
    ytd_bps: Optional[float] = None       # year-to-date change
    # Percentiles
    pct_1y: Optional[float] = None        # current vs 1Y history
    pct_3y: Optional[float] = None
    pct_5y: Optional[float] = None
    pct_10y: Optional[float] = None
    # Regime
    regime: Optional[SpreadRegime] = None
    regime_description: str = ""
    # Extremes
    min_1y: Optional[float] = None
    max_1y: Optional[float] = None
    avg_1y: Optional[float] = None


class CreditRegimeReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    ig_spread: SpreadSnapshot
    hy_spread: SpreadSnapshot
    hy_ig_ratio: float                    # risk appetite ratio (HY/IG)
    hy_ig_ratio_avg_2y: Optional[float] = None
    hy_ig_ratio_regime: str = ""          # risk_on / risk_off / neutral
    spread_curve_slope: Optional[float] = None   # long - short spread
    spread_curve_regime: str = ""
    overall_credit_regime: SpreadRegime
    regime_confidence: float              # 0–1
    narrative: str = ""


class MertonResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    # Inputs
    equity_value: float          # E (market cap, $M)
    equity_vol: float            # σE (annualised)
    total_debt: float            # D (face, $M)
    risk_free_rate: float        # r
    time_horizon: float          # T years
    # Solved outputs
    asset_value: float           # V ($M)
    asset_vol: float             # σV (annualised)
    distance_to_default: float   # DD
    prob_default_rn: float       # N(-DD) risk-neutral
    prob_default_physical: float # physical PD (Moody's adjustment: DD - Sharpe)
    implied_spread_bps: float    # credit spread implied by PD and LGD
    lgd_assumed: float           # loss given default (default 0.60)
    # Convergence
    converged: bool
    iterations: int
    error_norm: float


class KMVResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    default_point: float         # current liabilities + 0.5 × long-term debt
    asset_value: float
    asset_vol: float
    dd_1y: float                 # distance to default 1Y
    dd_5y: float                 # distance to default 5Y
    edf_1y: float                # expected default frequency 1Y (%)
    edf_5y: float                # expected default frequency 5Y (%)
    edf_band: str


class AltmanZResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    z_score: float
    x1: float    # working capital / total assets
    x2: float    # retained earnings / total assets
    x3: float    # EBIT / total assets
    x4: float    # market cap / total liabilities
    x5: float    # revenue / total assets
    zone: str    # "safe", "grey", "distress"
    bankruptcy_probability_pct: float


class OhlsonOResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    o_score: float               # log-odds
    probability_bankruptcy: float  # 0–1
    # Component scores
    size: float                  # log(total assets / GNP deflator)
    leverage: float              # total liabilities / total assets
    liquidity: float             # working capital / total assets
    performance: float           # net income / total assets
    cash_flow_coverage: float    # funds from operations / total liabilities
    recent_loss: float           # 1 if net income < 0 for 2 consecutive years
    current_ratio_flag: float    # 1 if current liabilities > current assets


class CompanyDefaultRisk(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    # Financials
    market_cap_m: Optional[float] = None
    total_debt_m: Optional[float] = None
    net_debt_m: Optional[float] = None
    total_assets_m: Optional[float] = None
    total_liabilities_m: Optional[float] = None
    current_assets_m: Optional[float] = None
    current_liabilities_m: Optional[float] = None
    retained_earnings_m: Optional[float] = None
    ebit_m: Optional[float] = None
    revenue_m: Optional[float] = None
    net_income_m: Optional[float] = None
    interest_expense_m: Optional[float] = None
    # Leverage ratios
    interest_coverage: Optional[float] = None   # EBIT / interest
    debt_to_ebitda: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    # Model scores
    altman_z: Optional[AltmanZResult] = None
    ohlson_o: Optional[OhlsonOResult] = None
    merton: Optional[MertonResult] = None
    kmv: Optional[KMVResult] = None
    # Composite
    default_risk_score: float = 0.0      # 0=minimal risk, 100=imminent default
    default_risk_band: DefaultRiskBand = "moderate"
    composite_pd_pct: float = 0.0        # weighted average PD (%)
    narrative: str = ""


class CreditSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_id: str
    signal_type: str             # "compression", "widening_alert", "fallen_angel", "sector_stress", "basis"
    direction: SignalDirection
    description: str
    magnitude_bps: Optional[float] = None
    confidence: float            # 0–1
    as_of: date
    affected_tickers: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class SectorStress(BaseModel):
    model_config = ConfigDict(frozen=True)

    sector: str
    as_of: date
    avg_spread_bps: Optional[float] = None
    spread_change_wow_bps: Optional[float] = None
    stress_level: str            # "normal", "elevated", "stressed", "distressed"
    fallen_angel_candidates: List[str] = Field(default_factory=list)
    comment: str = ""


# ---------------------------------------------------------------------------
# FRED Credit Spread Adapter
# ---------------------------------------------------------------------------

class FREDCreditSpreadAdapter:
    """
    Fetch ICE BofA OAS credit spreads from FRED public CSV endpoint.
    No API key required.
    """

    def __init__(self, timeout: float = 25.0):
        self._timeout = timeout
        self._cache: Dict[str, pd.Series] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_ttl = 3600  # 1-hour TTL

    async def _fetch_series(self, series_id: str, years_back: int = 15) -> pd.Series:
        """Fetch a single FRED series via CSV, with in-memory TTL cache."""
        cache_key = f"{series_id}_{years_back}"
        now = time.monotonic()
        if cache_key in self._cache and (now - self._cache_ts.get(cache_key, 0)) < self._cache_ttl:
            return self._cache[cache_key]

        url = f"{FRED_CSV_BASE}?id={series_id}"
        try:
            async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
                resp = await client.get(url, timeout=self._timeout)
                resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
            series = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().sort_index()
            cutoff = pd.Timestamp(date.today() - timedelta(days=365 * years_back))
            series = series[series.index >= cutoff]
            self._cache[cache_key] = series
            self._cache_ts[cache_key] = now
            return series
        except Exception as exc:
            logger.warning("FREDCreditSpreadAdapter._fetch_series failed",
                           series_id=series_id, error=str(exc))
            return pd.Series(dtype=float)

    async def fetch_spread(self, series_key: str, years_back: int = 10) -> pd.Series:
        """Fetch a named credit spread series (key from CREDIT_SPREAD_SERIES)."""
        meta = CREDIT_SPREAD_SERIES.get(series_key)
        if meta is None:
            raise ValueError(f"Unknown series_key: {series_key!r}")
        return await self._fetch_series(meta["series_id"], years_back=years_back)

    async def fetch_all_spreads(self, years_back: int = 10) -> Dict[str, pd.Series]:
        """Fetch all defined credit spread series concurrently."""
        keys = list(CREDIT_SPREAD_SERIES.keys())
        results = await asyncio.gather(
            *[self.fetch_spread(k, years_back=years_back) for k in keys],
            return_exceptions=True,
        )
        out: Dict[str, pd.Series] = {}
        for k, r in zip(keys, results):
            if isinstance(r, Exception):
                logger.warning("fetch_all_spreads: failed", key=k, error=str(r))
                out[k] = pd.Series(dtype=float)
            else:
                out[k] = r
        return out

    async def get_snapshot(self, series_key: str) -> SpreadSnapshot:
        """Build a SpreadSnapshot for a given series key."""
        series = await self.fetch_spread(series_key, years_back=10)
        meta = CREDIT_SPREAD_SERIES[series_key]

        if series.empty:
            raise ValueError(f"No data available for {series_key}")

        today = date.today()
        current_bps = float(series.iloc[-1])
        as_of = series.index[-1].date() if hasattr(series.index[-1], "date") else today

        def _change(days: int) -> Optional[float]:
            cutoff = series.index[-1] - pd.Timedelta(days=days)
            before = series[series.index <= cutoff]
            if before.empty:
                return None
            return round(current_bps - float(before.iloc[-1]), 2)

        def _pct_rank(lookback_days: int) -> Optional[float]:
            cutoff = series.index[-1] - pd.Timedelta(days=lookback_days)
            window = series[series.index >= cutoff]
            if len(window) < 10:
                return None
            rank = float((window < current_bps).mean()) * 100.0
            return round(rank, 1)

        # Spread regime
        thresholds = REGIME_THRESHOLDS.get(series_key)
        regime: Optional[SpreadRegime] = None
        regime_desc = ""
        if thresholds:
            if current_bps <= thresholds["tight"]:
                regime = "tight"
                regime_desc = f"Spread below {thresholds['tight']}bp — historically tight, strong credit demand"
            elif current_bps <= thresholds["normal_high"]:
                regime = "normal"
                regime_desc = f"Spread in normal range ({thresholds['tight']}–{thresholds['normal_high']}bp)"
            elif current_bps <= thresholds["crisis"]:
                regime = "wide"
                regime_desc = f"Spread above {thresholds['normal_high']}bp — elevated risk aversion"
            else:
                regime = "crisis"
                regime_desc = f"Spread above {thresholds['crisis']}bp — crisis-level credit stress"

        w1y = series[series.index >= (series.index[-1] - pd.Timedelta(days=365))]
        return SpreadSnapshot(
            series_key=series_key,
            series_id=meta["series_id"],
            name=meta["name"],
            rating=meta["rating"],
            as_of=as_of,
            current_bps=round(current_bps, 2),
            dod_bps=_change(1),
            wow_bps=_change(7),
            mom_bps=_change(30),
            ytd_bps=_change((series.index[-1] - pd.Timestamp(f"{as_of.year}-01-01")).days),
            pct_1y=_pct_rank(365),
            pct_3y=_pct_rank(365 * 3),
            pct_5y=_pct_rank(365 * 5),
            pct_10y=_pct_rank(365 * 10),
            regime=regime,
            regime_description=regime_desc,
            min_1y=round(float(w1y.min()), 2) if not w1y.empty else None,
            max_1y=round(float(w1y.max()), 2) if not w1y.empty else None,
            avg_1y=round(float(w1y.mean()), 2) if not w1y.empty else None,
        )


# ---------------------------------------------------------------------------
# Credit Spread Analyzer
# ---------------------------------------------------------------------------

class CreditSpreadAnalyzer:
    """
    Analyze credit spread dynamics, regimes, and carry/roll-down estimates.
    """

    def __init__(self, adapter: Optional[FREDCreditSpreadAdapter] = None):
        self._adapter = adapter or FREDCreditSpreadAdapter()

    async def get_regime_report(self) -> CreditRegimeReport:
        """Build a comprehensive credit regime report from live FRED data."""
        ig_task = self._adapter.get_snapshot("IG_OAS")
        hy_task = self._adapter.get_snapshot("HY_OAS")
        ig_short_task = self._adapter.fetch_spread("IG_1_3Y", years_back=5)
        ig_long_task = self._adapter.fetch_spread("IG_7_10Y", years_back=5)

        ig_snap, hy_snap, ig_short, ig_long = await asyncio.gather(
            ig_task, hy_task, ig_short_task, ig_long_task, return_exceptions=True
        )

        today = date.today()

        # Defaults on failure
        if isinstance(ig_snap, Exception):
            logger.error("CreditSpreadAnalyzer: IG fetch failed", error=str(ig_snap))
            raise RuntimeError("Cannot build regime report: IG spread unavailable")
        if isinstance(hy_snap, Exception):
            logger.error("CreditSpreadAnalyzer: HY fetch failed", error=str(hy_snap))
            raise RuntimeError("Cannot build regime report: HY spread unavailable")

        ig_bps = ig_snap.current_bps
        hy_bps = hy_snap.current_bps

        # HY/IG ratio
        hy_ig_ratio = round(hy_bps / ig_bps, 2) if ig_bps > 0 else 0.0

        # 2Y average ratio
        hy_ig_ratio_avg_2y: Optional[float] = None
        try:
            hy_series = await self._adapter.fetch_spread("HY_OAS", years_back=3)
            ig_series = await self._adapter.fetch_spread("IG_OAS", years_back=3)
            if not hy_series.empty and not ig_series.empty:
                combined = pd.DataFrame({"hy": hy_series, "ig": ig_series}).dropna()
                cutoff_2y = combined.index[-1] - pd.Timedelta(days=730)
                w2y = combined[combined.index >= cutoff_2y]
                if not w2y.empty:
                    hy_ig_ratio_avg_2y = round(float((w2y["hy"] / w2y["ig"]).mean()), 2)
        except Exception as exc:
            logger.warning("Could not compute 2Y HY/IG ratio", error=str(exc))

        # Ratio regime
        ratio_regime = "neutral"
        if hy_ig_ratio_avg_2y:
            if hy_ig_ratio < hy_ig_ratio_avg_2y * 0.92:
                ratio_regime = "risk_on_compression"  # HY expensive vs IG
            elif hy_ig_ratio > hy_ig_ratio_avg_2y * 1.08:
                ratio_regime = "risk_off_widening"
            else:
                ratio_regime = "neutral"

        # Spread curve slope (long - short IG)
        curve_slope: Optional[float] = None
        curve_regime = ""
        if not isinstance(ig_short, Exception) and not isinstance(ig_long, Exception):
            if not ig_short.empty and not ig_long.empty:
                short_val = float(ig_short.iloc[-1])
                long_val = float(ig_long.iloc[-1])
                curve_slope = round(long_val - short_val, 2)
                if curve_slope > 30:
                    curve_regime = "steep — long-end credit premium elevated"
                elif curve_slope > 10:
                    curve_regime = "normal"
                elif curve_slope > -5:
                    curve_regime = "flat — compressed term premium"
                else:
                    curve_regime = "inverted — unusual, watch for technical factors"

        # Overall regime (weighted combination)
        pct_ig = ig_snap.pct_5y or 50.0
        pct_hy = hy_snap.pct_5y or 50.0
        composite_pct = 0.4 * pct_ig + 0.6 * pct_hy

        if composite_pct <= 25:
            overall: SpreadRegime = "tight"
        elif composite_pct <= 60:
            overall = "normal"
        elif composite_pct <= 85:
            overall = "wide"
        else:
            overall = "crisis"

        confidence = min(1.0, 0.5 + abs(composite_pct - 50) / 100)

        # Narrative
        narrative = (
            f"IG OAS: {ig_bps:.0f}bp ({overall.upper()} regime, {pct_ig:.0f}th pct vs 5Y). "
            f"HY OAS: {hy_bps:.0f}bp ({pct_hy:.0f}th pct vs 5Y). "
            f"HY/IG ratio: {hy_ig_ratio:.1f}x ({ratio_regime}). "
        )
        if curve_slope is not None:
            narrative += f"Spread curve slope: {curve_slope:+.0f}bp ({curve_regime}). "
        if overall == "tight":
            narrative += "Credit is priced for perfection — carry attractive but convexity risk high."
        elif overall == "crisis":
            narrative += "Crisis-level spreads — significant credit stress, idiosyncratic risk elevated."

        return CreditRegimeReport(
            as_of=today,
            ig_spread=ig_snap,
            hy_spread=hy_snap,
            hy_ig_ratio=hy_ig_ratio,
            hy_ig_ratio_avg_2y=hy_ig_ratio_avg_2y,
            hy_ig_ratio_regime=ratio_regime,
            spread_curve_slope=curve_slope,
            spread_curve_regime=curve_regime,
            overall_credit_regime=overall,
            regime_confidence=round(confidence, 2),
            narrative=narrative,
        )

    async def spread_percentile_table(self) -> pd.DataFrame:
        """
        Return a DataFrame with current spread, percentile vs 1Y/3Y/5Y/10Y,
        and regime for all rated buckets.
        """
        keys = ["AAA_OAS", "IG_OAS", "BBB_OAS", "BB_OAS", "HY_OAS", "CCC_OAS"]
        snaps = await asyncio.gather(
            *[self._adapter.get_snapshot(k) for k in keys],
            return_exceptions=True,
        )
        rows = []
        for k, snap in zip(keys, snaps):
            if isinstance(snap, Exception):
                continue
            rows.append({
                "key": snap.series_key,
                "name": snap.name,
                "rating": snap.rating,
                "current_bps": snap.current_bps,
                "pct_1y": snap.pct_1y,
                "pct_3y": snap.pct_3y,
                "pct_5y": snap.pct_5y,
                "pct_10y": snap.pct_10y,
                "regime": snap.regime,
                "dod_bps": snap.dod_bps,
                "wow_bps": snap.wow_bps,
                "mom_bps": snap.mom_bps,
                "ytd_bps": snap.ytd_bps,
            })
        return pd.DataFrame(rows)

    async def carry_roll_estimate(
        self, series_key: str = "IG_OAS", duration_years: float = 7.0
    ) -> Dict:
        """
        Estimate carry + roll-down return for a credit position.
        carry = current spread (annualised)
        roll_down = approximated by slope of spread curve per year of seasoning
        excess_return = carry + roll_down (simple linear approximation)
        """
        series = await self._adapter.fetch_spread(series_key, years_back=5)
        if series.empty:
            return {"error": "No data"}

        current_spread = float(series.iloc[-1]) / 10000  # convert bp to decimal

        # Roll-down: use 1-year change in spread scaled by duration
        try:
            short_series = await self._adapter.fetch_spread("IG_1_3Y", years_back=5)
            long_series = await self._adapter.fetch_spread("IG_7_10Y", years_back=5)
            if not short_series.empty and not long_series.empty:
                short_bps = float(short_series.iloc[-1])
                long_bps = float(long_series.iloc[-1])
                slope_per_year = (long_bps - short_bps) / 7.0  # 7Y spread
                roll_down_bps = slope_per_year  # 1Y roll-down approx
            else:
                roll_down_bps = 0.0
        except Exception:
            roll_down_bps = 0.0

        carry_bps = float(series.iloc[-1])
        total_excess_bps = carry_bps + roll_down_bps

        # Historical vol of excess return
        daily_chg = series.diff().dropna()
        ann_vol_bps = float(daily_chg.std() * math.sqrt(252)) if not daily_chg.empty else 0.0

        carry_sharpe = (total_excess_bps / 100) / (ann_vol_bps / 100) if ann_vol_bps > 0 else 0.0

        return {
            "series_key": series_key,
            "carry_bps": round(carry_bps, 2),
            "roll_down_bps_per_year": round(roll_down_bps, 2),
            "total_excess_return_bps": round(total_excess_bps, 2),
            "annualized_spread_vol_bps": round(ann_vol_bps, 2),
            "carry_adjusted_sharpe": round(carry_sharpe, 2),
            "duration_years": duration_years,
        }


# ---------------------------------------------------------------------------
# Merton Model
# ---------------------------------------------------------------------------

class MertonModel:
    """
    Merton (1974) structural credit model.

    Treats equity as a call option on firm assets:
      E = V·N(d1) - D·e^(-rT)·N(d2)
      σE·E = V·σV·N(d1)

    Solves the 2-equation nonlinear system iteratively via scipy.fsolve.
    """

    DEFAULT_LGD = 0.60   # Loss Given Default

    @staticmethod
    def _d1(V: float, D: float, r: float, sigma_v: float, T: float) -> float:
        if V <= 0 or D <= 0 or sigma_v <= 0 or T <= 0:
            return -999.0
        return (math.log(V / D) + (r + 0.5 * sigma_v ** 2) * T) / (sigma_v * math.sqrt(T))

    @staticmethod
    def _d2(d1: float, sigma_v: float, T: float) -> float:
        return d1 - sigma_v * math.sqrt(T)

    @classmethod
    def _equity_call(cls, V: float, D: float, r: float, sigma_v: float, T: float) -> float:
        d1 = cls._d1(V, D, r, sigma_v, T)
        d2 = cls._d2(d1, sigma_v, T)
        return V * norm.cdf(d1) - D * math.exp(-r * T) * norm.cdf(d2)

    @classmethod
    def _system(
        cls,
        x: Tuple[float, float],
        E: float,
        sigma_e: float,
        D: float,
        r: float,
        T: float,
    ) -> Tuple[float, float]:
        """
        System of two equations:
          eq1: E = V·N(d1) - D·e^(-rT)·N(d2)   ← equity = call on assets
          eq2: σE·E = V·σV·N(d1)                ← equity vol linkage
        """
        V, sigma_v = x
        if V <= 0 or sigma_v <= 0:
            return (1e10, 1e10)
        d1 = cls._d1(V, D, r, sigma_v, T)
        d2 = cls._d2(d1, sigma_v, T)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        eq1 = V * nd1 - D * math.exp(-r * T) * nd2 - E
        eq2 = sigma_v * V * nd1 - sigma_e * E
        return (eq1, eq2)

    @classmethod
    def solve(
        cls,
        ticker: str,
        equity_value: float,
        equity_vol: float,
        total_debt: float,
        risk_free_rate: float,
        T: float = 1.0,
        lgd: float = DEFAULT_LGD,
        market_sharpe: float = 0.35,  # physical-measure drift adjustment
    ) -> MertonResult:
        """
        Solve for asset value V and asset volatility σV given equity observables.

        Parameters
        ----------
        equity_value   : market capitalisation ($M)
        equity_vol     : annualised equity return volatility (decimal, e.g. 0.30)
        total_debt     : total book debt ($M)
        risk_free_rate : annualised risk-free rate (decimal, e.g. 0.045)
        T              : time horizon in years
        lgd            : loss given default (fraction, default 0.60)
        market_sharpe  : Sharpe ratio for physical-measure PD adjustment
        """
        E = equity_value
        D = total_debt
        r = risk_free_rate
        sigma_e = equity_vol

        # Initial guess: V ≈ E + D, σV ≈ σE * E / (E + D)
        V0 = E + D
        sigma_v0 = sigma_e * E / V0 if V0 > 0 else sigma_e * 0.5

        converged = False
        iterations = 0
        error_norm = float("inf")
        V_sol = V0
        sigma_v_sol = sigma_v0

        try:
            sol, info, ier, msg = fsolve(
                cls._system,
                x0=[V0, sigma_v0],
                args=(E, sigma_e, D, r, T),
                full_output=True,
                maxfev=500,
            )
            V_sol, sigma_v_sol = float(sol[0]), float(sol[1])
            iterations = info["nfev"]
            error_norm = float(np.linalg.norm(info["fvec"]))
            converged = ier == 1 and V_sol > 0 and sigma_v_sol > 0

            # Guard: asset value must be positive and exceed equity floor
            if V_sol <= 0:
                V_sol = V0
                sigma_v_sol = sigma_v0
                converged = False
            if sigma_v_sol <= 0:
                sigma_v_sol = abs(sigma_v_sol) + 1e-6

        except Exception as exc:
            logger.warning("MertonModel.solve: fsolve failed", ticker=ticker, error=str(exc))

        # Distance to default
        dd = cls.distance_to_default(V_sol, sigma_v_sol, D, r, T)

        # Risk-neutral PD = N(-DD)
        pd_rn = float(norm.cdf(-dd))

        # Physical PD: adjust for equity risk premium (subtract Sharpe × sqrt(T))
        dd_physical = dd - market_sharpe * math.sqrt(T)
        pd_physical = float(norm.cdf(-dd_physical))

        # Implied credit spread from Merton: -ln(1 - PD × LGD) / T
        implied_spread_decimal = cls.credit_spread_implied(pd_rn, lgd, T)
        implied_spread_bps = round(implied_spread_decimal * 10000, 2)

        return MertonResult(
            ticker=ticker,
            as_of=date.today(),
            equity_value=round(E, 2),
            equity_vol=round(sigma_e, 4),
            total_debt=round(D, 2),
            risk_free_rate=round(r, 4),
            time_horizon=T,
            asset_value=round(V_sol, 2),
            asset_vol=round(sigma_v_sol, 4),
            distance_to_default=round(dd, 4),
            prob_default_rn=round(pd_rn, 6),
            prob_default_physical=round(pd_physical, 6),
            implied_spread_bps=implied_spread_bps,
            lgd_assumed=lgd,
            converged=converged,
            iterations=int(iterations),
            error_norm=round(error_norm, 8),
        )

    @staticmethod
    def distance_to_default(
        V: float, sigma_v: float, D: float, r: float, T: float
    ) -> float:
        """
        DD = [ln(V/D) + (r - 0.5·σV²)·T] / (σV·√T)
        Positive DD = asset value far above debt → low default risk.
        """
        if V <= 0 or D <= 0 or sigma_v <= 0 or T <= 0:
            return 0.0
        numerator = math.log(V / D) + (r - 0.5 * sigma_v ** 2) * T
        denominator = sigma_v * math.sqrt(T)
        return numerator / denominator if denominator > 1e-10 else 0.0

    @staticmethod
    def credit_spread_implied(pd: float, lgd: float, T: float) -> float:
        """
        Implied credit spread = -ln(1 - PD × LGD) / T
        Returns spread as decimal (multiply by 10000 for bps).
        """
        expected_loss = pd * lgd
        if expected_loss >= 1.0:
            return 0.50  # cap at 5000bp
        if T <= 0:
            return 0.0
        return -math.log(1.0 - expected_loss) / T

    @staticmethod
    def probability_of_default(dd: float) -> float:
        """Risk-neutral PD = N(-DD)."""
        return float(norm.cdf(-dd))


# ---------------------------------------------------------------------------
# KMV Adapter
# ---------------------------------------------------------------------------

class KMVAdapter:
    """
    KMV-style Expected Default Frequency (EDF) model.

    Default point (KMV convention):
      DP = current liabilities + 0.5 × long-term debt

    Uses Merton asset value and vol as inputs, then computes DD at 1Y and 5Y
    and maps to EDF using N(-DD).
    """

    EDF_BANDS = [
        (0.0010, "AAA/AA grade — exceptional credit quality"),
        (0.0025, "A grade — strong credit quality"),
        (0.0100, "BBB grade — adequate credit quality"),
        (0.0300, "BB grade — speculative"),
        (0.0700, "B grade — highly speculative"),
        (0.1500, "CCC grade — substantial credit risk"),
        (0.3000, "CC/C grade — near default"),
        (1.0000, "D grade — default or imminent default"),
    ]

    @classmethod
    def band_label(cls, edf: float) -> str:
        for threshold, label in cls.EDF_BANDS:
            if edf <= threshold:
                return label
        return "D grade — default or imminent default"

    @classmethod
    def compute(
        cls,
        ticker: str,
        current_liabilities: float,
        long_term_debt: float,
        asset_value: float,
        asset_vol: float,
        risk_free_rate: float = 0.045,
    ) -> KMVResult:
        """
        Compute KMV EDF at 1Y and 5Y horizons.

        Parameters (all in $M):
          current_liabilities : short-term obligations
          long_term_debt      : long-term debt
          asset_value         : V from Merton solve (or market cap + total debt)
          asset_vol           : σV from Merton solve
          risk_free_rate      : annual risk-free rate
        """
        dp = current_liabilities + 0.5 * long_term_debt

        dd_1y = MertonModel.distance_to_default(asset_value, asset_vol, dp, risk_free_rate, 1.0)
        dd_5y = MertonModel.distance_to_default(asset_value, asset_vol, dp, risk_free_rate, 5.0)

        edf_1y = float(norm.cdf(-dd_1y))
        edf_5y = float(norm.cdf(-dd_5y))

        return KMVResult(
            ticker=ticker,
            as_of=date.today(),
            default_point=round(dp, 2),
            asset_value=round(asset_value, 2),
            asset_vol=round(asset_vol, 4),
            dd_1y=round(dd_1y, 4),
            dd_5y=round(dd_5y, 4),
            edf_1y=round(edf_1y * 100, 4),  # as %
            edf_5y=round(edf_5y * 100, 4),  # as %
            edf_band=cls.band_label(edf_1y),
        )


# ---------------------------------------------------------------------------
# Altman Z-Score
# ---------------------------------------------------------------------------

def compute_altman_z(
    ticker: str,
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    revenue: float,
) -> AltmanZResult:
    """
    Altman Z-Score (manufacturing model, 1968):
      Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5

    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap / Total Liabilities
    X5 = Revenue / Total Assets

    Zones:
      Z > 2.99   → safe zone
      1.81–2.99  → grey zone
      Z < 1.81   → distress zone
    """
    if total_assets <= 0:
        return AltmanZResult(
            ticker=ticker, z_score=0.0, x1=0, x2=0, x3=0, x4=0, x5=0,
            zone="unknown", bankruptcy_probability_pct=50.0
        )

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities if total_liabilities > 0 else 0.0
    x5 = revenue / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > 2.99:
        zone = "safe"
        bp_pct = max(0.5, 10.0 - (z - 2.99) * 3)
    elif z > 1.81:
        zone = "grey"
        bp_pct = 15.0 + (2.99 - z) / (2.99 - 1.81) * 30.0
    else:
        zone = "distress"
        bp_pct = min(95.0, 45.0 + (1.81 - z) * 25.0)

    return AltmanZResult(
        ticker=ticker,
        z_score=round(z, 4),
        x1=round(x1, 4),
        x2=round(x2, 4),
        x3=round(x3, 4),
        x4=round(x4, 4),
        x5=round(x5, 4),
        zone=zone,
        bankruptcy_probability_pct=round(bp_pct, 2),
    )


# ---------------------------------------------------------------------------
# Ohlson O-Score
# ---------------------------------------------------------------------------

def compute_ohlson_o(
    ticker: str,
    total_assets: float,
    total_liabilities: float,
    current_assets: float,
    current_liabilities: float,
    net_income: float,
    funds_from_operations: float,   # net income + D&A
    retained_earnings: float,
    ebit: float,
    net_income_prior: float = 0.0,  # prior year net income for loss flag
    gnp_deflator: float = 100.0,    # price-level index; use 100 as approximation
) -> OhlsonOResult:
    """
    Ohlson (1980) O-Score bankruptcy prediction model.

    O = -1.32 - 0.407·SIZE + 6.03·TLTA - 1.43·WCTA + 0.076·CLCA
         - 1.72·OENEG - 2.37·NITA - 1.83·FUTL + 0.285·INTWO - 0.521·CHIN

    SIZE   = log(Total Assets / GNP Price Deflator)
    TLTA   = Total Liabilities / Total Assets
    WCTA   = Working Capital / Total Assets
    CLCA   = Current Liabilities / Current Assets
    OENEG  = 1 if Total Liabilities > Total Assets, else 0
    NITA   = Net Income / Total Assets
    FUTL   = Funds From Operations / Total Liabilities
    INTWO  = 1 if Net Income < 0 for last 2 years, else 0
    CHIN   = (NI_t - NI_{t-1}) / (|NI_t| + |NI_{t-1}|)
    """
    size = math.log(max(total_assets / gnp_deflator, 1e-6))
    tlta = total_liabilities / total_assets if total_assets > 0 else 0.0
    wc = current_assets - current_liabilities
    wcta = wc / total_assets if total_assets > 0 else 0.0
    clca = current_liabilities / current_assets if current_assets > 0 else 0.0
    oeneg = 1.0 if total_liabilities > total_assets else 0.0
    nita = net_income / total_assets if total_assets > 0 else 0.0
    futl = funds_from_operations / total_liabilities if total_liabilities > 0 else 0.0
    intwo = 1.0 if (net_income < 0 and net_income_prior < 0) else 0.0
    denom_chin = abs(net_income) + abs(net_income_prior)
    chin = (net_income - net_income_prior) / denom_chin if denom_chin > 1e-9 else 0.0

    o_score = (
        -1.32
        - 0.407 * size
        + 6.03 * tlta
        - 1.43 * wcta
        + 0.076 * clca
        - 1.72 * oeneg
        - 2.37 * nita
        - 1.83 * futl
        + 0.285 * intwo
        - 0.521 * chin
    )

    # Convert log-odds to probability
    prob = 1.0 / (1.0 + math.exp(-o_score))

    return OhlsonOResult(
        ticker=ticker,
        o_score=round(o_score, 4),
        probability_bankruptcy=round(prob, 6),
        size=round(size, 4),
        leverage=round(tlta, 4),
        liquidity=round(wcta, 4),
        performance=round(nita, 4),
        cash_flow_coverage=round(futl, 4),
        recent_loss=intwo,
        current_ratio_flag=1.0 if current_liabilities > current_assets else 0.0,
    )


# ---------------------------------------------------------------------------
# Company Default Risk Engine
# ---------------------------------------------------------------------------

class CompanyDefaultRiskEngine:
    """
    Per-company default risk combining Merton + Altman Z + Ohlson O.
    Fetches financials from SEC EDGAR XBRL API (free).
    """

    EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
    EDGAR_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    EDGAR_TICKER = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        self._ticker_map: Dict[str, str] = {}  # ticker → CIK

    async def _load_ticker_cik_map(self) -> None:
        if self._ticker_map:
            return
        try:
            async with httpx.AsyncClient(headers=_HEADERS) as client:
                resp = await client.get(self.EDGAR_TICKER, timeout=self._timeout)
                resp.raise_for_status()
                data = resp.json()
            for entry in data.values():
                t = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                if t and cik:
                    self._ticker_map[t] = cik
        except Exception as exc:
            logger.warning("CompanyDefaultRiskEngine: ticker→CIK load failed", error=str(exc))

    async def _get_cik(self, ticker: str) -> Optional[str]:
        await self._load_ticker_cik_map()
        return self._ticker_map.get(ticker.upper())

    async def _fetch_company_facts(self, cik: str) -> Optional[dict]:
        url = self.EDGAR_FACTS.format(cik=cik)
        try:
            async with httpx.AsyncClient(headers=_HEADERS) as client:
                resp = await client.get(url, timeout=self._timeout)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("_fetch_company_facts failed", cik=cik, error=str(exc))
            return None

    @staticmethod
    def _latest_xbrl_value(facts: dict, concept: str, namespace: str = "us-gaap") -> Optional[float]:
        """Extract the most recent annual value of an XBRL concept."""
        try:
            units = facts["facts"][namespace][concept]["units"]
            # Prefer USD units
            unit_data = units.get("USD") or units.get("shares") or next(iter(units.values()), [])
            # Filter to 10-K (annual) filings
            annual = [
                r for r in unit_data
                if r.get("form") in ("10-K", "20-F", "40-F")
                   and r.get("val") is not None
            ]
            if not annual:
                # Fall back to any data
                annual = [r for r in unit_data if r.get("val") is not None]
            if not annual:
                return None
            # Sort by end date, take the most recent
            annual.sort(key=lambda r: r.get("end", ""), reverse=True)
            return float(annual[0]["val"])
        except (KeyError, IndexError, TypeError):
            return None

    async def get_financials(self, ticker: str) -> Optional[Dict]:
        """
        Fetch key financial metrics from EDGAR XBRL API.
        Returns dict with: total_assets, total_liabilities, current_assets,
        current_liabilities, retained_earnings, ebit, revenue, net_income,
        long_term_debt, interest_expense.
        """
        cik = await self._get_cik(ticker)
        if not cik:
            return None

        facts = await self._fetch_company_facts(cik)
        if not facts:
            return None

        lv = self._latest_xbrl_value

        # Core balance sheet
        total_assets = lv(facts, "Assets")
        total_liabilities = lv(facts, "Liabilities")
        current_assets = lv(facts, "AssetsCurrent")
        current_liabilities = lv(facts, "LiabilitiesCurrent")
        retained_earnings = lv(facts, "RetainedEarningsAccumulatedDeficit")

        # Income statement
        revenue = (
            lv(facts, "Revenues")
            or lv(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
            or lv(facts, "SalesRevenueNet")
        )
        net_income = (
            lv(facts, "NetIncomeLoss")
            or lv(facts, "NetIncome")
        )
        operating_income = lv(facts, "OperatingIncomeLoss")
        interest_expense = lv(facts, "InterestExpense")
        da = (
            lv(facts, "DepreciationDepletionAndAmortization")
            or lv(facts, "DepreciationAndAmortization")
        )

        # Debt
        long_term_debt = (
            lv(facts, "LongTermDebtNoncurrent")
            or lv(facts, "LongTermDebt")
        )
        current_lt_debt = lv(facts, "LongTermDebtCurrent")
        total_debt = (
            lv(facts, "LongTermDebtAndCapitalLeaseObligations")
            or (
                (long_term_debt or 0) + (current_lt_debt or 0)
            )
        )

        # EBIT ≈ operating income or net income + interest + taxes
        ebit = operating_income
        if ebit is None and net_income is not None and interest_expense is not None:
            ebit = net_income + interest_expense  # rough approximation

        # Convert from $ to $M (EDGAR reports in USD)
        def to_m(v: Optional[float]) -> Optional[float]:
            return round(v / 1e6, 2) if v is not None else None

        return {
            "total_assets": to_m(total_assets),
            "total_liabilities": to_m(total_liabilities),
            "current_assets": to_m(current_assets),
            "current_liabilities": to_m(current_liabilities),
            "retained_earnings": to_m(retained_earnings),
            "ebit": to_m(ebit),
            "revenue": to_m(revenue),
            "net_income": to_m(net_income),
            "long_term_debt": to_m(long_term_debt),
            "interest_expense": to_m(interest_expense),
            "total_debt": to_m(total_debt),
            "depreciation_amortization": to_m(da),
        }

    async def compute_default_risk(
        self,
        ticker: str,
        equity_vol: float = 0.30,
        risk_free_rate: float = 0.045,
        equity_market_cap_m: Optional[float] = None,
    ) -> CompanyDefaultRisk:
        """
        Compute composite default risk for a company.

        equity_vol: annualised equity return vol (if not passed, defaults to 0.30)
        equity_market_cap_m: optional override; if None, attempts EDGAR lookup
        """
        fin = await self.get_financials(ticker)
        today = date.today()

        if not fin:
            return CompanyDefaultRisk(
                ticker=ticker,
                as_of=today,
                default_risk_score=50.0,
                default_risk_band="moderate",
                composite_pd_pct=5.0,
                narrative="EDGAR financials unavailable — default to moderate risk",
            )

        total_assets = fin.get("total_assets")
        total_liabilities = fin.get("total_liabilities")
        current_assets = fin.get("current_assets")
        current_liabilities = fin.get("current_liabilities")
        retained_earnings = fin.get("retained_earnings") or 0.0
        ebit = fin.get("ebit")
        revenue = fin.get("revenue")
        net_income = fin.get("net_income")
        total_debt = fin.get("total_debt")
        long_term_debt = fin.get("long_term_debt") or 0.0
        interest_expense = fin.get("interest_expense")
        da = fin.get("depreciation_amortization") or 0.0

        # Use provided market cap or fall back to assets - liabilities (rough BV equity)
        if equity_market_cap_m is not None:
            mkt_cap = equity_market_cap_m
        elif total_assets and total_liabilities:
            mkt_cap = max(total_assets - total_liabilities, 1.0)
        else:
            mkt_cap = 1000.0  # placeholder

        # Derived metrics
        net_debt: Optional[float] = (
            (total_debt or 0) - 0  # simplified; cash not always available
        )

        interest_coverage: Optional[float] = None
        if ebit is not None and interest_expense and interest_expense > 0:
            interest_coverage = round(ebit / interest_expense, 2)

        ebitda = (ebit or 0) + da
        debt_to_ebitda: Optional[float] = None
        net_debt_to_ebitda: Optional[float] = None
        if ebitda > 0:
            if total_debt:
                debt_to_ebitda = round(total_debt / ebitda, 2)
            if net_debt is not None:
                net_debt_to_ebitda = round(net_debt / ebitda, 2)

        # ---------- Altman Z-Score ----------
        altman = None
        if all(v is not None for v in [total_assets, total_liabilities, ebit, revenue]):
            wc = (current_assets or 0) - (current_liabilities or 0)
            altman = compute_altman_z(
                ticker=ticker,
                working_capital=wc,
                total_assets=total_assets,
                retained_earnings=retained_earnings,
                ebit=ebit,
                market_cap=mkt_cap,
                total_liabilities=total_liabilities,
                revenue=revenue,
            )

        # ---------- Ohlson O-Score ----------
        ohlson = None
        if all(v is not None for v in [total_assets, total_liabilities, current_assets,
                                        current_liabilities, net_income]):
            ffo = (net_income or 0) + da  # funds from operations proxy
            ohlson = compute_ohlson_o(
                ticker=ticker,
                total_assets=total_assets,
                total_liabilities=total_liabilities,
                current_assets=current_assets,
                current_liabilities=current_liabilities,
                net_income=net_income,
                funds_from_operations=ffo,
                retained_earnings=retained_earnings,
                ebit=ebit or 0,
            )

        # ---------- Merton Model ----------
        merton = None
        kmv = None
        if total_debt and total_debt > 0:
            merton = MertonModel.solve(
                ticker=ticker,
                equity_value=mkt_cap,
                equity_vol=equity_vol,
                total_debt=total_debt,
                risk_free_rate=risk_free_rate,
                T=1.0,
            )

            kmv = KMVAdapter.compute(
                ticker=ticker,
                current_liabilities=current_liabilities or (total_liabilities * 0.3 if total_liabilities else 0),
                long_term_debt=long_term_debt,
                asset_value=merton.asset_value,
                asset_vol=merton.asset_vol,
                risk_free_rate=risk_free_rate,
            )

        # ---------- Composite Score ----------
        # Weights: Merton 40%, Altman 30%, Ohlson 30%
        pds: List[float] = []

        if merton:
            pds.append(merton.prob_default_rn * 100 * 0.40)
        else:
            pds.append(5.0 * 0.40)

        if altman:
            pds.append(altman.bankruptcy_probability_pct * 0.30)
        else:
            pds.append(5.0 * 0.30)

        if ohlson:
            pds.append(ohlson.probability_bankruptcy * 100 * 0.30)
        else:
            pds.append(5.0 * 0.30)

        composite_pd = sum(pds)

        # Map PD → 0-100 risk score (log scale)
        # PD 0.1% → score ~5, 1% → ~20, 5% → ~45, 20% → ~70, 50%+ → ~90+
        risk_score = min(100.0, max(0.0, 20.0 * math.log1p(composite_pd * 2)))

        # Risk band
        if risk_score < 15:
            band: DefaultRiskBand = "minimal"
        elif risk_score < 30:
            band = "low"
        elif risk_score < 50:
            band = "moderate"
        elif risk_score < 65:
            band = "elevated"
        elif risk_score < 80:
            band = "high"
        else:
            band = "distressed"

        # Narrative
        parts = [f"{ticker}: composite PD ~{composite_pd:.1f}%, risk score {risk_score:.0f}/100 ({band})."]
        if altman:
            parts.append(f"Altman Z={altman.z_score:.2f} ({altman.zone}).")
        if merton:
            parts.append(f"Merton DD={merton.distance_to_default:.2f}, implied spread={merton.implied_spread_bps:.0f}bp.")
        if interest_coverage:
            cov_comment = "adequate" if interest_coverage > 3 else "thin" if interest_coverage > 1.5 else "concerning"
            parts.append(f"Interest coverage {interest_coverage:.1f}x ({cov_comment}).")
        if debt_to_ebitda:
            lev_comment = "manageable" if debt_to_ebitda < 3 else "elevated" if debt_to_ebitda < 5 else "stressed"
            parts.append(f"Debt/EBITDA {debt_to_ebitda:.1f}x ({lev_comment}).")

        return CompanyDefaultRisk(
            ticker=ticker,
            as_of=today,
            market_cap_m=round(mkt_cap, 2),
            total_debt_m=total_debt,
            net_debt_m=net_debt,
            total_assets_m=total_assets,
            total_liabilities_m=total_liabilities,
            current_assets_m=current_assets,
            current_liabilities_m=current_liabilities,
            retained_earnings_m=retained_earnings,
            ebit_m=ebit,
            revenue_m=revenue,
            net_income_m=net_income,
            interest_expense_m=interest_expense,
            interest_coverage=interest_coverage,
            debt_to_ebitda=debt_to_ebitda,
            net_debt_to_ebitda=net_debt_to_ebitda,
            altman_z=altman,
            ohlson_o=ohlson,
            merton=merton,
            kmv=kmv,
            default_risk_score=round(risk_score, 2),
            default_risk_band=band,
            composite_pd_pct=round(composite_pd, 4),
            narrative=" ".join(parts),
        )


# ---------------------------------------------------------------------------
# Credit Signal Engine
# ---------------------------------------------------------------------------

class CreditSignalEngine:
    """
    Generate tradeable credit signals from spread dynamics and sector analysis.

    Signals:
      1. Compression: HY-IG ratio below 2Y average → rotate HY→IG
      2. Widening alert: >30bp WoW change in any rating bucket
      3. Fallen angel watch: BB OAS approaching BBB levels
      4. CDS-bond basis: when implied Merton spread >> actual OAS
      5. Regime shift: spread crosses from normal to wide (or vice versa)
    """

    WIDENING_ALERT_THRESHOLD_BPS = 30.0

    def __init__(self, adapter: Optional[FREDCreditSpreadAdapter] = None):
        self._adapter = adapter or FREDCreditSpreadAdapter()
        self._spread_analyzer = CreditSpreadAnalyzer(self._adapter)

    async def generate_all_signals(self) -> List[CreditSignal]:
        """Run all signal generators and return combined list."""
        tasks = [
            self._compression_signal(),
            self._widening_alerts(),
            self._fallen_angel_watch(),
            self._regime_shift_signal(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        signals: List[CreditSignal] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("CreditSignalEngine: signal error", error=str(r))
            elif isinstance(r, list):
                signals.extend(r)
            elif r is not None:
                signals.append(r)
        return signals

    async def _compression_signal(self) -> Optional[CreditSignal]:
        """
        Compression: If HY/IG ratio < 2Y average by >8%, HY is expensive vs IG.
        Recommendation: rotate from HY into IG.
        """
        try:
            regime = await self._spread_analyzer.get_regime_report()
            today = date.today()

            if regime.hy_ig_ratio_avg_2y and regime.hy_ig_ratio > 0:
                ratio_delta_pct = (regime.hy_ig_ratio / regime.hy_ig_ratio_avg_2y - 1) * 100
                if ratio_delta_pct < -8:
                    return CreditSignal(
                        signal_id="compression_rotate_hy_ig",
                        signal_type="compression",
                        direction="sell",
                        description=(
                            f"HY/IG spread ratio ({regime.hy_ig_ratio:.1f}x) is "
                            f"{abs(ratio_delta_pct):.1f}% below 2Y average "
                            f"({regime.hy_ig_ratio_avg_2y:.1f}x). "
                            "HY priced rich vs IG — consider rotating into higher-quality credit."
                        ),
                        magnitude_bps=regime.hy_spread.current_bps - regime.ig_spread.current_bps,
                        confidence=min(0.85, 0.5 + abs(ratio_delta_pct) / 50),
                        as_of=today,
                        metadata={
                            "hy_oas_bps": regime.hy_spread.current_bps,
                            "ig_oas_bps": regime.ig_spread.current_bps,
                            "ratio_current": regime.hy_ig_ratio,
                            "ratio_2y_avg": regime.hy_ig_ratio_avg_2y,
                        },
                    )
                elif ratio_delta_pct > 8:
                    return CreditSignal(
                        signal_id="decompression_overweight_hy",
                        signal_type="compression",
                        direction="buy",
                        description=(
                            f"HY/IG ratio ({regime.hy_ig_ratio:.1f}x) elevated vs 2Y avg "
                            f"({regime.hy_ig_ratio_avg_2y:.1f}x) by {ratio_delta_pct:.1f}%. "
                            "HY cheap relative to IG — potential overweight opportunity if fundamentals stable."
                        ),
                        magnitude_bps=regime.hy_spread.current_bps - regime.ig_spread.current_bps,
                        confidence=min(0.75, 0.4 + ratio_delta_pct / 60),
                        as_of=today,
                        metadata={
                            "ratio_current": regime.hy_ig_ratio,
                            "ratio_2y_avg": regime.hy_ig_ratio_avg_2y,
                        },
                    )
        except Exception as exc:
            logger.warning("_compression_signal failed", error=str(exc))
        return None

    async def _widening_alerts(self) -> List[CreditSignal]:
        """Alert on >30bp week-over-week widening in any bucket."""
        keys = ["IG_OAS", "HY_OAS", "BBB_OAS", "BB_OAS", "CCC_OAS"]
        snaps = await asyncio.gather(
            *[self._adapter.get_snapshot(k) for k in keys],
            return_exceptions=True,
        )
        alerts = []
        today = date.today()
        for k, snap in zip(keys, snaps):
            if isinstance(snap, Exception):
                continue
            wow = snap.wow_bps
            if wow and wow >= self.WIDENING_ALERT_THRESHOLD_BPS:
                confidence = min(0.95, 0.5 + wow / 200)
                alerts.append(CreditSignal(
                    signal_id=f"widening_alert_{k}",
                    signal_type="widening_alert",
                    direction="sell",
                    description=(
                        f"{snap.name} widened {wow:+.0f}bp WoW to {snap.current_bps:.0f}bp. "
                        f"Exceeds {self.WIDENING_ALERT_THRESHOLD_BPS:.0f}bp alert threshold. "
                        "Reduce duration/credit risk exposure."
                    ),
                    magnitude_bps=wow,
                    confidence=confidence,
                    as_of=today,
                    metadata={
                        "series_key": k,
                        "current_bps": snap.current_bps,
                        "wow_bps": wow,
                        "regime": snap.regime,
                        "pct_5y": snap.pct_5y,
                    },
                ))
        return alerts

    async def _fallen_angel_watch(self) -> Optional[CreditSignal]:
        """
        Fallen angel watch: BB OAS within 50bp of IG BBB OAS suggests
        crossover issuers at risk of downgrade to HY.
        """
        try:
            bb_task = self._adapter.get_snapshot("BB_OAS")
            bbb_task = self._adapter.get_snapshot("BBB_OAS")
            bb_snap, bbb_snap = await asyncio.gather(bb_task, bbb_task, return_exceptions=True)

            if isinstance(bb_snap, Exception) or isinstance(bbb_snap, Exception):
                return None

            gap = bb_snap.current_bps - bbb_snap.current_bps
            today = date.today()

            if gap < 50:
                return CreditSignal(
                    signal_id="fallen_angel_watch",
                    signal_type="fallen_angel",
                    direction="watch",
                    description=(
                        f"BB OAS ({bb_snap.current_bps:.0f}bp) is only {gap:.0f}bp above "
                        f"BBB OAS ({bbb_snap.current_bps:.0f}bp). "
                        "Crossover compression: BBB- issuers with negative outlook face "
                        "elevated fallen-angel risk. Screen BBB- rated issuers."
                    ),
                    magnitude_bps=gap,
                    confidence=min(0.80, 0.5 + (50 - gap) / 100),
                    as_of=today,
                    metadata={
                        "bb_oas_bps": bb_snap.current_bps,
                        "bbb_oas_bps": bbb_snap.current_bps,
                        "gap_bps": gap,
                    },
                )
        except Exception as exc:
            logger.warning("_fallen_angel_watch failed", error=str(exc))
        return None

    async def _regime_shift_signal(self) -> Optional[CreditSignal]:
        """Detect transition between credit regimes (e.g., normal → wide)."""
        try:
            ig_series = await self._adapter.fetch_spread("IG_OAS", years_back=3)
            if ig_series.empty:
                return None

            current = float(ig_series.iloc[-1])
            month_ago_series = ig_series[ig_series.index <= ig_series.index[-1] - pd.Timedelta(days=30)]
            if month_ago_series.empty:
                return None

            prior = float(month_ago_series.iloc[-1])
            thresholds = REGIME_THRESHOLDS.get("IG_OAS", {})
            today = date.today()

            def _classify(val: float) -> str:
                if val <= thresholds.get("tight", 80):
                    return "tight"
                elif val <= thresholds.get("normal_high", 130):
                    return "normal"
                elif val <= thresholds.get("crisis", 280):
                    return "wide"
                else:
                    return "crisis"

            current_regime = _classify(current)
            prior_regime = _classify(prior)

            if current_regime != prior_regime:
                worsening = {"tight": 0, "normal": 1, "wide": 2, "crisis": 3}
                direction: SignalDirection = (
                    "sell" if worsening.get(current_regime, 1) > worsening.get(prior_regime, 1)
                    else "buy"
                )
                return CreditSignal(
                    signal_id=f"regime_shift_{prior_regime}_to_{current_regime}",
                    signal_type="regime_shift",
                    direction=direction,
                    description=(
                        f"IG credit regime shifted from {prior_regime.upper()} to "
                        f"{current_regime.upper()} over the past 30 days. "
                        f"Current IG OAS: {current:.0f}bp (was {prior:.0f}bp)."
                    ),
                    magnitude_bps=round(current - prior, 2),
                    confidence=0.70,
                    as_of=today,
                    metadata={
                        "prior_regime": prior_regime,
                        "current_regime": current_regime,
                        "current_bps": current,
                        "prior_bps": prior,
                    },
                )
        except Exception as exc:
            logger.warning("_regime_shift_signal failed", error=str(exc))
        return None

    async def cds_bond_basis_signal(
        self,
        ticker: str,
        merton_result: MertonResult,
        observed_spread_bps: float,
    ) -> CreditSignal:
        """
        CDS-bond basis: when Merton implied spread >> observed bond spread,
        the bond is rich (cheap to buy CDS protection).
        Negative basis = bond cheap (buy bond, sell CDS protection).
        """
        implied = merton_result.implied_spread_bps
        basis = implied - observed_spread_bps
        today = date.today()

        if abs(basis) < 20:
            direction: SignalDirection = "hold"
            desc = f"{ticker}: Merton implied spread ({implied:.0f}bp) ~ observed ({observed_spread_bps:.0f}bp). Fairly priced."
            confidence = 0.3
        elif basis > 50:
            direction = "sell"
            desc = (
                f"{ticker}: Merton implied spread ({implied:.0f}bp) >> observed ({observed_spread_bps:.0f}bp). "
                f"Basis: +{basis:.0f}bp. Bond appears RICH vs structural model — "
                "consider selling protection or reducing exposure."
            )
            confidence = min(0.85, 0.4 + basis / 300)
        else:
            direction = "buy"
            desc = (
                f"{ticker}: Observed spread ({observed_spread_bps:.0f}bp) >> Merton implied ({implied:.0f}bp). "
                f"Basis: {basis:.0f}bp. Bond appears CHEAP vs structural model — potential value opportunity."
            )
            confidence = min(0.80, 0.4 + abs(basis) / 300)

        return CreditSignal(
            signal_id=f"cds_bond_basis_{ticker}",
            signal_type="basis",
            direction=direction,
            description=desc,
            magnitude_bps=round(basis, 2),
            confidence=confidence,
            as_of=today,
            affected_tickers=[ticker],
            metadata={
                "ticker": ticker,
                "implied_spread_bps": implied,
                "observed_spread_bps": observed_spread_bps,
                "basis_bps": basis,
                "merton_pd_pct": round(merton_result.prob_default_rn * 100, 3),
            },
        )

    def sector_stress_from_spreads(
        self,
        sector: str,
        spreads_bps: List[float],
        tickers: Optional[List[str]] = None,
        prior_week_avg: Optional[float] = None,
    ) -> SectorStress:
        """
        Aggregate company-level spreads into sector stress indicator.
        spreads_bps: list of individual bond/CDS spreads in bp
        """
        today = date.today()
        if not spreads_bps:
            return SectorStress(sector=sector, as_of=today, stress_level="normal")

        avg = float(np.mean(spreads_bps))
        wow_chg = round(avg - prior_week_avg, 2) if prior_week_avg else None

        # Classify stress
        if avg < 150:
            stress = "normal"
        elif avg < 300:
            stress = "elevated"
        elif avg < 600:
            stress = "stressed"
        else:
            stress = "distressed"

        # Fallen angel candidates: those >400bp
        fallen_angel_candidates = [
            t for t, s in zip(tickers or [], spreads_bps) if s > 400
        ]

        comment = (
            f"{sector}: avg spread {avg:.0f}bp, "
            f"WoW {wow_chg:+.0f}bp, "
            f"stress level: {stress}"
        ) if wow_chg else f"{sector}: avg spread {avg:.0f}bp, stress level: {stress}"

        return SectorStress(
            sector=sector,
            as_of=today,
            avg_spread_bps=round(avg, 2),
            spread_change_wow_bps=wow_chg,
            stress_level=stress,
            fallen_angel_candidates=fallen_angel_candidates,
            comment=comment,
        )


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel as _PydanticBase

    credit_router = APIRouter(prefix="/credit", tags=["credit"])

    # Singletons
    _adapter = FREDCreditSpreadAdapter()
    _analyzer = CreditSpreadAnalyzer(_adapter)
    _signal_engine = CreditSignalEngine(_adapter)
    _default_risk_engine = CompanyDefaultRiskEngine()

    class MertonRequest(_PydanticBase):
        equity_value_m: float = Field(..., description="Market cap in $M")
        equity_vol: float = Field(0.30, description="Annualised equity vol (decimal)")
        total_debt_m: float = Field(..., description="Total debt in $M")
        risk_free_rate: float = Field(0.045, description="Annual risk-free rate (decimal)")
        time_horizon_years: float = Field(1.0, description="Forecast horizon in years")
        lgd: float = Field(0.60, description="Loss given default")

    @credit_router.get("/spreads", summary="Current credit spread snapshots for all rating buckets")
    async def get_spreads(years_back: int = Query(5, ge=1, le=20)) -> dict:
        try:
            df = await _analyzer.spread_percentile_table()
            return {
                "as_of": date.today().isoformat(),
                "spreads": df.to_dict(orient="records"),
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @credit_router.get("/regime", summary="Overall credit market regime report")
    async def get_regime() -> dict:
        try:
            report = await _analyzer.get_regime_report()
            return report.model_dump()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @credit_router.post("/merton/{ticker}", summary="Merton structural credit model for a company")
    async def merton_endpoint(ticker: str, req: MertonRequest) -> dict:
        try:
            result = MertonModel.solve(
                ticker=ticker.upper(),
                equity_value=req.equity_value_m,
                equity_vol=req.equity_vol,
                total_debt=req.total_debt_m,
                risk_free_rate=req.risk_free_rate,
                T=req.time_horizon_years,
                lgd=req.lgd,
            )
            return result.model_dump()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @credit_router.get("/default-risk/{ticker}", summary="Composite default risk for a company")
    async def default_risk(
        ticker: str,
        equity_vol: float = Query(0.30, ge=0.01, le=5.0),
        risk_free_rate: float = Query(0.045, ge=0.0, le=0.2),
        market_cap_m: Optional[float] = Query(None),
    ) -> dict:
        try:
            result = await _default_risk_engine.compute_default_risk(
                ticker=ticker.upper(),
                equity_vol=equity_vol,
                risk_free_rate=risk_free_rate,
                equity_market_cap_m=market_cap_m,
            )
            return result.model_dump()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @credit_router.get("/signals", summary="Credit market signals and alerts")
    async def get_signals() -> dict:
        try:
            signals = await _signal_engine.generate_all_signals()
            return {
                "as_of": date.today().isoformat(),
                "count": len(signals),
                "signals": [s.model_dump() for s in signals],
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @credit_router.get("/sector-stress", summary="Sector-level credit stress from aggregate spreads")
    async def sector_stress_endpoint(
        sector: str = Query("technology"),
        spreads_csv: str = Query("150,180,200,250,120", description="Comma-separated spread values in bp"),
    ) -> dict:
        try:
            spreads = [float(x.strip()) for x in spreads_csv.split(",") if x.strip()]
            result = _signal_engine.sector_stress_from_spreads(sector=sector, spreads_bps=spreads)
            return result.model_dump()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @credit_router.get("/carry-roll/{series_key}", summary="Carry and roll-down estimate for a spread index")
    async def carry_roll(series_key: str = "IG_OAS") -> dict:
        try:
            result = await _analyzer.carry_roll_estimate(series_key=series_key)
            return result
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

except ImportError:
    credit_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available; credit_router not registered")


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

async def credit_spread_snapshot(series_key: str = "IG_OAS") -> SpreadSnapshot:
    """Fetch a credit spread snapshot for a given series key."""
    adapter = FREDCreditSpreadAdapter()
    return await adapter.get_snapshot(series_key)


async def merton_default_risk(
    ticker: str,
    equity_value_m: float,
    equity_vol: float,
    total_debt_m: float,
    risk_free_rate: float = 0.045,
    T: float = 1.0,
) -> MertonResult:
    """Convenience: compute Merton structural credit model."""
    return MertonModel.solve(
        ticker=ticker,
        equity_value=equity_value_m,
        equity_vol=equity_vol,
        total_debt=total_debt_m,
        risk_free_rate=risk_free_rate,
        T=T,
    )


async def company_default_risk(
    ticker: str,
    equity_vol: float = 0.30,
    risk_free_rate: float = 0.045,
    market_cap_m: Optional[float] = None,
) -> CompanyDefaultRisk:
    """Convenience: compute composite default risk from EDGAR financials."""
    engine = CompanyDefaultRiskEngine()
    return await engine.compute_default_risk(
        ticker=ticker,
        equity_vol=equity_vol,
        risk_free_rate=risk_free_rate,
        equity_market_cap_m=market_cap_m,
    )


async def credit_regime() -> CreditRegimeReport:
    """Convenience: get current credit market regime report."""
    adapter = FREDCreditSpreadAdapter()
    analyzer = CreditSpreadAnalyzer(adapter)
    return await analyzer.get_regime_report()


async def credit_signals() -> List[CreditSignal]:
    """Convenience: generate all active credit signals."""
    engine = CreditSignalEngine()
    return await engine.generate_all_signals()


def altman_z_score(
    ticker: str,
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    revenue: float,
) -> AltmanZResult:
    """Convenience: compute Altman Z-Score directly from financial inputs."""
    return compute_altman_z(
        ticker=ticker,
        working_capital=working_capital,
        total_assets=total_assets,
        retained_earnings=retained_earnings,
        ebit=ebit,
        market_cap=market_cap,
        total_liabilities=total_liabilities,
        revenue=revenue,
    )
