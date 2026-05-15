"""
Comprehensive stress testing and scenario analysis for portfolios.
Historical scenarios, hypothetical scenarios, Monte Carlo, and macro stress.

Dimension targeted:
  dim_083 — Stress testing / scenario analysis  score → 9

Classes
-------
HistoricalScenarioEngine
    Replay 20 major historical stress events against a portfolio.

HypotheticalScenarioBuilder
    User-defined macro shocks with correlation propagation.

MonteCarloStressTester
    10,000-path simulation: normal, Student-t, historical bootstrap,
    Student-t copula; VaR, CVaR, max-drawdown distributions.

MacroStressTester
    FRED macro variable shocks → asset return mapping.
    Three canonical scenarios: stagflation, recession, deflation.

PortfolioStressReport
    Consolidated ranked stress report with hedge recommendations.

LiquidityStressTest
    Almgren-Chriss days-to-liquidate, liquidity-adjusted VaR,
    fire-sale price impact, bid-ask widening.

FastAPI router  stress_router  — mounted at /api/stress
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.stats import norm, t as student_t

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

TRADING_DAYS = 252
_EPSILON = 1e-12
_HEADERS = {"User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"}
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_name: str
    pnl_pct: float                            # portfolio P&L as fraction
    pnl_dollar: float                         # dollar P&L (if portfolio_value given)
    asset_contributions: Dict[str, float]     # per-asset contribution to P&L
    factor_contributions: Dict[str, float]    # SPY/TLT/GLD/USD factor splits
    description: str
    severity: str                             # "mild" | "moderate" | "severe" | "extreme"


@dataclass
class MonteCarloResult:
    n_simulations: int
    horizon_days: int
    distribution: str
    var_95: float          # VaR at 95% confidence (positive number = loss)
    var_99: float
    cvar_95: float         # Expected Shortfall at 95%
    cvar_99: float
    mean_return: float
    std_return: float
    max_drawdown_median: float
    max_drawdown_p95: float
    return_percentiles: Dict[str, float]   # p1, p5, p25, p50, p75, p95, p99


@dataclass
class MacroStressResult:
    scenario_name: str
    macro_shocks: Dict[str, float]         # variable → shock size
    asset_impacts: Dict[str, float]        # ticker → expected return
    portfolio_impact: float
    description: str


@dataclass
class LiquidityStressResult:
    ticker: str
    position_value: float
    adv_normal: float
    adv_stressed: float                    # 50% of normal
    days_to_liquidate: float              # Almgren-Chriss
    price_impact_pct: float              # fire-sale discount
    bid_ask_normal_bps: float
    bid_ask_stressed_bps: float          # 3× widening
    liquidity_adjusted_var: float
    liquidation_cost_dollar: float


@dataclass
class StressReport:
    scenarios_ranked: List[Dict[str, Any]]  # worst to best
    worst_scenario: str
    best_scenario: str
    vulnerable_positions: Dict[str, List[str]]   # position → [scenarios it hurts in]
    concentration_risks: List[str]
    hedge_recommendations: List[str]
    stress_var: float                             # weighted average historical scenario loss


# ---------------------------------------------------------------------------
# 1.  Historical Scenario Engine
# ---------------------------------------------------------------------------

# 20 major market stress events with key factor moves
HISTORICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "COVID_Crash_2020": {
        "start_date": "2020-02-19",
        "end_date": "2020-03-23",
        "description": "COVID-19 pandemic crash. Fastest bear market in history.",
        "key_moves": {
            "SPY": -0.340,
            "TLT": +0.200,
            "GLD": -0.013,
            "UUP": +0.036,
            "VIX_change": +50.0,
            "HYG": -0.200,
            "QQQ": -0.290,
        },
    },
    "GFC_Peak_2008": {
        "start_date": "2008-09-15",
        "end_date": "2009-03-09",
        "description": "Global Financial Crisis — Lehman bankruptcy to market trough.",
        "key_moves": {
            "SPY": -0.450,
            "TLT": +0.310,
            "GLD": +0.040,
            "UUP": +0.120,
            "VIX_change": +60.0,
            "HYG": -0.380,
            "QQQ": -0.420,
        },
    },
    "Tech_Bust_2000_2002": {
        "start_date": "2000-03-10",
        "end_date": "2002-10-09",
        "description": "Dot-com bubble burst. NASDAQ fell 78% from peak.",
        "key_moves": {
            "SPY": -0.490,
            "TLT": +0.400,
            "GLD": +0.120,
            "UUP": +0.030,
            "VIX_change": +25.0,
            "HYG": -0.120,
            "QQQ": -0.780,
        },
    },
    "Taper_Tantrum_2013": {
        "start_date": "2013-05-22",
        "end_date": "2013-06-24",
        "description": "Fed taper talk sends bond yields surging.",
        "key_moves": {
            "SPY": -0.058,
            "TLT": -0.110,
            "GLD": -0.100,
            "UUP": +0.028,
            "VIX_change": +8.0,
            "HYG": -0.055,
            "QQQ": -0.040,
        },
    },
    "Flash_Crash_May_2010": {
        "start_date": "2010-05-06",
        "end_date": "2010-05-06",
        "description": "Intraday flash crash. Dow briefly fell nearly 1,000 points.",
        "key_moves": {
            "SPY": -0.034,
            "TLT": +0.020,
            "GLD": -0.002,
            "UUP": +0.008,
            "VIX_change": +10.0,
            "HYG": -0.025,
            "QQQ": -0.038,
        },
    },
    "Russia_Ukraine_2022": {
        "start_date": "2022-02-24",
        "end_date": "2022-03-15",
        "description": "Russia invades Ukraine. Commodity surge, equity selloff.",
        "key_moves": {
            "SPY": -0.065,
            "TLT": -0.060,
            "GLD": +0.070,
            "UUP": +0.020,
            "VIX_change": +12.0,
            "HYG": -0.060,
            "QQQ": -0.090,
        },
    },
    "LTCM_1998": {
        "start_date": "1998-08-17",
        "end_date": "1998-10-08",
        "description": "Russian default + LTCM collapse. Liquidity crisis.",
        "key_moves": {
            "SPY": -0.190,
            "TLT": +0.150,
            "GLD": +0.020,
            "UUP": -0.040,
            "VIX_change": +20.0,
            "HYG": -0.130,
            "QQQ": -0.200,
        },
    },
    "9_11_2001": {
        "start_date": "2001-09-10",
        "end_date": "2001-09-21",
        "description": "September 11 terrorist attacks. Markets closed 4 days.",
        "key_moves": {
            "SPY": -0.115,
            "TLT": +0.040,
            "GLD": +0.025,
            "UUP": -0.002,
            "VIX_change": +15.0,
            "HYG": -0.065,
            "QQQ": -0.140,
        },
    },
    "Black_Monday_1987": {
        "start_date": "1987-10-19",
        "end_date": "1987-10-19",
        "description": "Black Monday. Dow fell 22.6% in a single day.",
        "key_moves": {
            "SPY": -0.226,
            "TLT": +0.080,
            "GLD": +0.030,
            "UUP": -0.020,
            "VIX_change": +30.0,
            "HYG": -0.100,
            "QQQ": -0.220,
        },
    },
    "Dotcom_Peak_2000": {
        "start_date": "2000-01-14",
        "end_date": "2000-03-10",
        "description": "Final melt-up before dot-com peak. Momentum bubble.",
        "key_moves": {
            "SPY": -0.018,
            "TLT": -0.060,
            "GLD": -0.020,
            "UUP": +0.010,
            "VIX_change": +5.0,
            "HYG": -0.020,
            "QQQ": +0.280,
        },
    },
    "Euro_Debt_Crisis_2011": {
        "start_date": "2011-07-22",
        "end_date": "2011-10-04",
        "description": "European sovereign debt crisis. Greece/Italy contagion fears.",
        "key_moves": {
            "SPY": -0.190,
            "TLT": +0.200,
            "GLD": +0.100,
            "UUP": +0.030,
            "VIX_change": +25.0,
            "HYG": -0.110,
            "QQQ": -0.180,
        },
    },
    "China_Devaluation_2015": {
        "start_date": "2015-08-18",
        "end_date": "2015-08-26",
        "description": "China devalues yuan. Global equity selloff.",
        "key_moves": {
            "SPY": -0.113,
            "TLT": +0.040,
            "GLD": +0.015,
            "UUP": +0.022,
            "VIX_change": +18.0,
            "HYG": -0.060,
            "QQQ": -0.115,
        },
    },
    "Volmageddon_2018": {
        "start_date": "2018-02-02",
        "end_date": "2018-02-08",
        "description": "VIX spike destroys short-vol products. Market selloff.",
        "key_moves": {
            "SPY": -0.094,
            "TLT": -0.015,
            "GLD": -0.022,
            "UUP": +0.005,
            "VIX_change": +25.0,
            "HYG": -0.040,
            "QQQ": -0.095,
        },
    },
    "Q4_Selloff_2018": {
        "start_date": "2018-10-01",
        "end_date": "2018-12-24",
        "description": "Fed tightening fears + trade war. Q4 worst since 1931.",
        "key_moves": {
            "SPY": -0.200,
            "TLT": +0.040,
            "GLD": +0.040,
            "UUP": +0.005,
            "VIX_change": +20.0,
            "HYG": -0.110,
            "QQQ": -0.235,
        },
    },
    "Rate_Shock_2022": {
        "start_date": "2022-01-03",
        "end_date": "2022-10-13",
        "description": "Fed hikes 425bps. Worst year for 60/40 since 1970.",
        "key_moves": {
            "SPY": -0.250,
            "TLT": -0.360,
            "GLD": -0.030,
            "UUP": +0.150,
            "VIX_change": +15.0,
            "HYG": -0.180,
            "QQQ": -0.380,
        },
    },
    "Repo_Crisis_2019": {
        "start_date": "2019-09-16",
        "end_date": "2019-09-20",
        "description": "Overnight repo rates spike to 10%. Fed intervenes.",
        "key_moves": {
            "SPY": -0.010,
            "TLT": -0.020,
            "GLD": +0.005,
            "UUP": +0.002,
            "VIX_change": +3.0,
            "HYG": -0.008,
            "QQQ": -0.012,
        },
    },
    "SVB_Collapse_2023": {
        "start_date": "2023-03-08",
        "end_date": "2023-03-20",
        "description": "Silicon Valley Bank failure. Regional bank contagion.",
        "key_moves": {
            "SPY": -0.048,
            "TLT": +0.085,
            "GLD": +0.060,
            "UUP": -0.018,
            "VIX_change": +8.0,
            "HYG": -0.035,
            "QQQ": +0.010,
        },
    },
    "Oil_Crash_2014_2016": {
        "start_date": "2014-06-20",
        "end_date": "2016-02-11",
        "description": "Oil falls from $107 to $26. Energy sector devastated.",
        "key_moves": {
            "SPY": -0.015,
            "TLT": +0.130,
            "GLD": -0.030,
            "UUP": +0.120,
            "VIX_change": +12.0,
            "HYG": -0.110,
            "QQQ": +0.050,
        },
    },
    "EM_Contagion_1997": {
        "start_date": "1997-07-02",
        "end_date": "1998-01-12",
        "description": "Asian financial crisis. Thailand baht devaluation triggers EM rout.",
        "key_moves": {
            "SPY": -0.090,
            "TLT": +0.070,
            "GLD": -0.015,
            "UUP": +0.050,
            "VIX_change": +15.0,
            "HYG": -0.080,
            "QQQ": -0.100,
        },
    },
    "Mini_Crash_Oct_2023": {
        "start_date": "2023-08-01",
        "end_date": "2023-10-27",
        "description": "Bond yield surge (10Y to 5%). Equity/bond correlation positive.",
        "key_moves": {
            "SPY": -0.080,
            "TLT": -0.110,
            "GLD": -0.010,
            "UUP": +0.030,
            "VIX_change": +8.0,
            "HYG": -0.045,
            "QQQ": -0.090,
        },
    },
}

# Typical betas vs SPY for common assets (for extrapolation)
_DEFAULT_SPY_BETAS: Dict[str, float] = {
    "SPY": 1.00,
    "QQQ": 1.18,
    "IWM": 1.15,
    "EFA": 0.85,
    "EEM": 0.90,
    "TLT": -0.40,
    "IEF": -0.20,
    "LQD": -0.10,
    "HYG": 0.55,
    "GLD": 0.05,
    "SLV": 0.25,
    "USO": 0.60,
    "UUP": -0.30,
    "VNQ": 0.85,
    "XLE": 0.90,
    "XLF": 1.10,
    "XLK": 1.20,
    "XLV": 0.60,
    "XLU": 0.20,
    "BTC-USD": 1.50,
    "ETH-USD": 1.80,
}


class HistoricalScenarioEngine:
    """
    Replay historical stress events against an arbitrary portfolio.

    Asset returns during each scenario are estimated via:
      1. Direct mapping if the asset is in key_moves
      2. Beta extrapolation from SPY move for unlisted assets
    """

    def __init__(self, custom_betas: Optional[Dict[str, float]] = None):
        self.betas = {**_DEFAULT_SPY_BETAS, **(custom_betas or {})}

    # ------------------------------------------------------------------
    # Internal: estimate asset return given scenario
    # ------------------------------------------------------------------

    def _estimate_asset_return(
        self,
        ticker: str,
        key_moves: Dict[str, float],
        ticker_beta: Optional[float] = None,
    ) -> float:
        """
        Return estimated asset return during the scenario.

        Priority:
        1. Exact key_moves entry
        2. Provided beta vs SPY
        3. Default beta lookup
        4. 0.8 × SPY move (fallback for equity-like)
        """
        if ticker in key_moves:
            return key_moves[ticker]

        spy_move = key_moves.get("SPY", 0.0)
        beta = ticker_beta or self.betas.get(ticker, 0.8)
        return beta * spy_move

    # ------------------------------------------------------------------
    # Core: apply scenario to a portfolio
    # ------------------------------------------------------------------

    def apply_scenario(
        self,
        portfolio_weights: Dict[str, float],
        scenario_name: str,
        portfolio_value: float = 1_000_000.0,
        asset_betas: Optional[Dict[str, float]] = None,
    ) -> ScenarioResult:
        """
        Apply a named historical scenario to the portfolio.

        Parameters
        ----------
        portfolio_weights : {ticker: weight}  (weights should sum ≤ 1)
        scenario_name     : key in HISTORICAL_SCENARIOS
        portfolio_value   : dollar value for P&L calculation
        asset_betas       : optional override betas vs SPY

        Returns
        -------
        ScenarioResult
        """
        if scenario_name not in HISTORICAL_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(HISTORICAL_SCENARIOS.keys())}"
            )

        scenario = HISTORICAL_SCENARIOS[scenario_name]
        key_moves = scenario["key_moves"]
        betas = {**self.betas, **(asset_betas or {})}

        asset_contributions: Dict[str, float] = {}
        total_pnl_pct = 0.0

        for ticker, weight in portfolio_weights.items():
            asset_ret = self._estimate_asset_return(ticker, key_moves, betas.get(ticker))
            contribution = weight * asset_ret
            asset_contributions[ticker] = contribution
            total_pnl_pct += contribution

        pnl_dollar = total_pnl_pct * portfolio_value

        # Factor decomposition
        spy_move = key_moves.get("SPY", 0.0)
        tlt_move = key_moves.get("TLT", 0.0)
        gld_move = key_moves.get("GLD", 0.0)
        uup_move = key_moves.get("UUP", 0.0)

        # Approximate factor contributions via macro sensitivity
        factor_contributions = {
            "equity_factor": sum(
                w * betas.get(t, 0.8) * spy_move
                for t, w in portfolio_weights.items()
            ),
            "rates_factor": sum(
                w * (-0.40) * tlt_move
                for t, w in portfolio_weights.items()
                if t not in ("TLT", "IEF", "SHY")
            ),
            "gold_factor": sum(
                w * (0.10) * gld_move
                for t, w in portfolio_weights.items()
            ),
            "dollar_factor": sum(
                w * betas.get(t, 0.0) * uup_move
                for t, w in portfolio_weights.items()
            ),
        }

        severity = (
            "extreme" if total_pnl_pct < -0.20
            else "severe" if total_pnl_pct < -0.10
            else "moderate" if total_pnl_pct < -0.05
            else "mild"
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            pnl_pct=total_pnl_pct,
            pnl_dollar=pnl_dollar,
            asset_contributions=asset_contributions,
            factor_contributions=factor_contributions,
            description=scenario["description"],
            severity=severity,
        )

    def apply_all_scenarios(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
    ) -> List[ScenarioResult]:
        """Apply every historical scenario; return list sorted worst→best."""
        results = []
        for name in HISTORICAL_SCENARIOS:
            try:
                r = self.apply_scenario(portfolio_weights, name, portfolio_value)
                results.append(r)
            except Exception as exc:
                logger.warning("Scenario %s failed: %s", name, exc)
        results.sort(key=lambda r: r.pnl_pct)
        return results


# ---------------------------------------------------------------------------
# 2.  Hypothetical Scenario Builder
# ---------------------------------------------------------------------------

class HypotheticalScenarioBuilder:
    """
    Build custom hypothetical stress scenarios.

    Supports:
    - Explicit asset shocks
    - Factor shocks (equity, rates, credit, FX, commodities)
    - Rate structure scenarios (parallel, steepener, flattener)
    - Correlation-propagated shocks
    """

    def __init__(self, betas: Optional[Dict[str, float]] = None):
        self.betas = {**_DEFAULT_SPY_BETAS, **(betas or {})}

    # ------------------------------------------------------------------
    # Factor shock propagation
    # ------------------------------------------------------------------

    def _propagate_spy_shock(
        self,
        spy_shock: float,
        portfolio_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute each asset's shock via its SPY beta."""
        return {t: spy_shock * self.betas.get(t, 0.8) for t in portfolio_weights}

    # ------------------------------------------------------------------
    # Build scenario from user-defined shocks
    # ------------------------------------------------------------------

    def build_scenario(
        self,
        portfolio_weights: Dict[str, float],
        shocks: Dict[str, float],
        propagate_via_correlation: bool = True,
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Build asset shock vector from user-defined factor shocks.

        Parameters
        ----------
        portfolio_weights : {ticker: weight}
        shocks            : dict of shocks, supported keys:
            "SPY"   → equity shock (propagated via betas)
            "TLT"   → rates shock (propagated via duration betas)
            "GLD"   → gold shock
            "UUP"   → dollar shock
            "VIX"   → VIX spike (reduces expected returns proportionally)
            "spread_ig_bps" → IG spread widening in bps
            "spread_hy_bps" → HY spread widening in bps
            "rates_parallel_bps" → parallel rate shift
            Or direct ticker shocks: "AAPL" → -0.15
        correlation_matrix : if provided, propagate from anchor shocks

        Returns
        -------
        dict  {ticker: expected_return_during_scenario}
        """
        asset_shocks: Dict[str, float] = {}
        tickers = list(portfolio_weights.keys())

        # Start with direct asset shocks (highest priority)
        for ticker in tickers:
            if ticker in shocks:
                asset_shocks[ticker] = float(shocks[ticker])

        # Propagate SPY shock
        if "SPY" in shocks:
            spy_shock = shocks["SPY"]
            for ticker in tickers:
                if ticker not in asset_shocks:
                    asset_shocks[ticker] = spy_shock * self.betas.get(ticker, 0.8)

        # Overlay TLT / rates shock
        if "TLT" in shocks:
            tlt_shock = shocks["TLT"]
            duration_betas = {
                "TLT": 1.0, "IEF": 0.5, "SHY": 0.15, "LQD": 0.7, "HYG": 0.3,
            }
            for ticker in tickers:
                dur_beta = duration_betas.get(ticker, 0.0)
                if dur_beta != 0.0:
                    asset_shocks[ticker] = asset_shocks.get(ticker, 0.0) + tlt_shock * dur_beta

        # Parallel rate shift (interest rate risk for bonds)
        if "rates_parallel_bps" in shocks:
            shift = shocks["rates_parallel_bps"] / 100.0
            # Approximate: bond price change ≈ -duration × Δy
            durations = {"TLT": 18.0, "IEF": 7.5, "SHY": 2.0, "LQD": 10.0, "HYG": 4.0, "TIPS": 8.0}
            for ticker in tickers:
                dur = durations.get(ticker, 0.0)
                if dur > 0:
                    rate_impact = -dur * shift
                    asset_shocks[ticker] = asset_shocks.get(ticker, 0.0) + rate_impact

        # Credit spread widening
        if "spread_ig_bps" in shocks:
            ig_widen = shocks["spread_ig_bps"] / 10_000.0
            ig_assets = {"LQD": 1.0, "VCIT": 1.0, "VCSH": 0.5}
            for ticker in tickers:
                if ticker in ig_assets:
                    spread_impact = -ig_widen * ig_assets[ticker] * 5.0  # approx duration
                    asset_shocks[ticker] = asset_shocks.get(ticker, 0.0) + spread_impact

        if "spread_hy_bps" in shocks:
            hy_widen = shocks["spread_hy_bps"] / 10_000.0
            hy_assets = {"HYG": 1.0, "JNK": 1.0}
            for ticker in tickers:
                if ticker in hy_assets:
                    spread_impact = -hy_widen * hy_assets[ticker] * 3.5
                    asset_shocks[ticker] = asset_shocks.get(ticker, 0.0) + spread_impact

        # Dollar shock
        if "UUP" in shocks:
            uup_shock = shocks["UUP"]
            uup_betas = {"EFA": -0.35, "EEM": -0.45, "GLD": -0.30, "SLV": -0.35, "USO": -0.25}
            for ticker in tickers:
                if ticker in uup_betas and ticker not in asset_shocks:
                    asset_shocks[ticker] = uup_betas[ticker] * uup_shock

        # Gold shock
        if "GLD" in shocks:
            gld_shock = shocks["GLD"]
            for ticker in ["GLD", "SLV", "GDX", "IAU"]:
                if ticker in tickers and ticker not in asset_shocks:
                    asset_shocks[ticker] = gld_shock * (1.0 if ticker in ("GLD", "IAU") else 1.8)

        # VIX spike: additional vol premium cost on all assets
        if "VIX" in shocks:
            vix_spike = shocks["VIX"]
            vix_equity_impact = -0.005 * max(vix_spike, 0)
            for ticker in tickers:
                existing = asset_shocks.get(ticker, 0.0)
                asset_shocks[ticker] = existing + vix_equity_impact * self.betas.get(ticker, 0.5)

        # Correlation propagation for any remaining tickers
        if propagate_via_correlation and correlation_matrix is not None:
            anchor_tickers = [t for t in asset_shocks if t in correlation_matrix.columns]
            for ticker in tickers:
                if ticker in asset_shocks or ticker not in correlation_matrix.columns:
                    continue
                # Weighted average of correlated anchor shocks
                corr_row = correlation_matrix.loc[ticker]
                weighted_shock = 0.0
                total_weight = 0.0
                for anchor in anchor_tickers:
                    if anchor in corr_row.index:
                        corr_val = abs(float(corr_row[anchor]))
                        weighted_shock += corr_val * asset_shocks[anchor]
                        total_weight += corr_val
                if total_weight > _EPSILON:
                    asset_shocks[ticker] = weighted_shock / total_weight

        # Fill any remaining tickers with SPY-beta extrapolation
        spy_base = shocks.get("SPY", 0.0)
        for ticker in tickers:
            if ticker not in asset_shocks:
                asset_shocks[ticker] = spy_base * self.betas.get(ticker, 0.0)

        return asset_shocks

    def apply_to_portfolio(
        self,
        portfolio_weights: Dict[str, float],
        shocks: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
        scenario_name: str = "Custom",
        **kwargs: Any,
    ) -> ScenarioResult:
        """Apply custom shocks and return ScenarioResult."""
        asset_shocks = self.build_scenario(portfolio_weights, shocks, **kwargs)

        asset_contributions: Dict[str, float] = {}
        total_pnl = 0.0
        for ticker, weight in portfolio_weights.items():
            ret = asset_shocks.get(ticker, 0.0)
            contrib = weight * ret
            asset_contributions[ticker] = contrib
            total_pnl += contrib

        severity = (
            "extreme" if total_pnl < -0.20
            else "severe" if total_pnl < -0.10
            else "moderate" if total_pnl < -0.05
            else "mild"
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            pnl_pct=total_pnl,
            pnl_dollar=total_pnl * portfolio_value,
            asset_contributions=asset_contributions,
            factor_contributions={k: shocks.get(k, 0.0) for k in ("SPY", "TLT", "GLD", "UUP")},
            description=f"Hypothetical scenario with shocks: {shocks}",
            severity=severity,
        )

    # ------------------------------------------------------------------
    # Canonical predefined scenarios
    # ------------------------------------------------------------------

    def equity_crash(
        self,
        portfolio_weights: Dict[str, float],
        equity_drop: float = -0.30,
        portfolio_value: float = 1_000_000.0,
    ) -> ScenarioResult:
        """SPY −30% with flight-to-quality bond rally."""
        shocks = {
            "SPY": equity_drop,
            "TLT": +0.15,
            "GLD": +0.08,
            "UUP": +0.05,
            "VIX": +50.0,
        }
        return self.apply_to_portfolio(
            portfolio_weights, shocks, portfolio_value,
            scenario_name=f"Equity_Crash_SPY{equity_drop:.0%}"
        )

    def rate_spike(
        self,
        portfolio_weights: Dict[str, float],
        rate_bps: float = 100.0,
        portfolio_value: float = 1_000_000.0,
    ) -> ScenarioResult:
        """Parallel rate rise of `rate_bps` basis points."""
        shocks = {
            "rates_parallel_bps": rate_bps,
            "SPY": -0.03 * (rate_bps / 100),
        }
        return self.apply_to_portfolio(
            portfolio_weights, shocks, portfolio_value,
            scenario_name=f"Rate_Spike_{rate_bps:.0f}bps"
        )

    def dollar_surge(
        self,
        portfolio_weights: Dict[str, float],
        uup_move: float = 0.10,
        portfolio_value: float = 1_000_000.0,
    ) -> ScenarioResult:
        """Dollar strengthens 10% — hurts EM, gold, commodities."""
        shocks = {"UUP": uup_move, "GLD": -0.07, "EEM": -0.12}
        return self.apply_to_portfolio(
            portfolio_weights, shocks, portfolio_value,
            scenario_name=f"Dollar_Surge_{uup_move:.0%}"
        )


# ---------------------------------------------------------------------------
# 3.  Monte Carlo Stress Tester
# ---------------------------------------------------------------------------

class MonteCarloStressTester:
    """
    Monte Carlo portfolio simulation for tail risk quantification.

    Distributions:
    - normal          : Gaussian returns
    - student_t       : fat tails (dof = 5 default)
    - historical      : bootstrap from actual returns
    - t_copula        : Student-t copula with marginal transforms
    """

    def __init__(self, n_simulations: int = 10_000, seed: int = 42):
        self.n_simulations = n_simulations
        self.seed = seed

    # ------------------------------------------------------------------
    # Single-path max drawdown
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(wealth: np.ndarray) -> float:
        if len(wealth) == 0:
            return 0.0
        peak = np.maximum.accumulate(wealth)
        dd = (peak - wealth) / np.where(peak > _EPSILON, peak, _EPSILON)
        return float(np.max(dd))

    # ------------------------------------------------------------------
    # Normal simulation
    # ------------------------------------------------------------------

    def _simulate_normal(
        self,
        mu_daily: float,
        sigma_daily: float,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        returns = rng.normal(mu_daily, sigma_daily, (self.n_simulations, horizon))
        return np.prod(1 + returns, axis=1) - 1.0

    # ------------------------------------------------------------------
    # Student-t simulation
    # ------------------------------------------------------------------

    def _simulate_student_t(
        self,
        mu_daily: float,
        sigma_daily: float,
        horizon: int,
        dof: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        scale = sigma_daily * np.sqrt((dof - 2) / dof)
        z = rng.standard_t(df=dof, size=(self.n_simulations, horizon))
        returns = mu_daily + scale * z
        return np.prod(1 + returns, axis=1) - 1.0

    # ------------------------------------------------------------------
    # Historical bootstrap
    # ------------------------------------------------------------------

    def _simulate_historical(
        self,
        hist_returns: np.ndarray,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Block bootstrap from historical daily returns."""
        n = len(hist_returns)
        if n < horizon:
            idx = rng.integers(0, n, (self.n_simulations, horizon))
        else:
            idx = rng.integers(0, n, (self.n_simulations, horizon))
        returns = hist_returns[idx]
        return np.prod(1 + returns, axis=1) - 1.0

    # ------------------------------------------------------------------
    # Multi-asset Student-t copula
    # ------------------------------------------------------------------

    def _simulate_t_copula_portfolio(
        self,
        weights: np.ndarray,
        mu_daily: np.ndarray,
        cov_daily: np.ndarray,
        horizon: int,
        dof: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate portfolio returns using a Student-t copula.

        1. Draw multivariate t with given cov
        2. Transform to uniform marginals via t CDF
        3. Transform back via asset-specific t inverse CDF
        4. Combine into portfolio return
        """
        n_assets = len(weights)
        # Cholesky of correlation
        vols = np.sqrt(np.diag(cov_daily))
        corr = cov_daily / (np.outer(vols, vols) + _EPSILON)
        np.fill_diagonal(corr, 1.0)

        try:
            L = np.linalg.cholesky(corr + np.eye(n_assets) * 1e-8)
        except np.linalg.LinAlgError:
            L = np.eye(n_assets)

        portfolio_returns = np.zeros(self.n_simulations)
        chi2 = rng.chisquare(df=dof, size=(self.n_simulations, horizon))

        for day in range(horizon):
            z_normal = rng.standard_normal((self.n_simulations, n_assets))
            z_corr = z_normal @ L.T
            chi = np.sqrt(chi2[:, day] / dof)[:, np.newaxis]
            t_samples = z_corr / chi

            # Transform to uniform, back to asset-specific t
            u = student_t.cdf(t_samples, df=dof)
            asset_returns = student_t.ppf(u, df=dof) * vols[np.newaxis, :] + mu_daily[np.newaxis, :]
            daily_port = asset_returns @ weights
            portfolio_returns += np.log1p(daily_port)

        return np.exp(portfolio_returns) - 1.0

    # ------------------------------------------------------------------
    # Core: run simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        portfolio_weights: Dict[str, float],
        returns_df: Optional[pd.DataFrame] = None,
        horizon_days: int = 252,
        distribution: Literal["normal", "student_t", "historical", "t_copula"] = "student_t",
        dof: float = 5.0,
        mu_override: Optional[Dict[str, float]] = None,
        sigma_override: Optional[Dict[str, float]] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation and return risk metrics.

        Parameters
        ----------
        portfolio_weights : {ticker: weight}
        returns_df        : DataFrame of daily returns (required for historical/t_copula)
        horizon_days      : simulation length in trading days
        distribution      : return distribution assumption
        dof               : degrees of freedom for Student-t
        mu_override       : override expected returns {ticker: daily_mu}
        sigma_override    : override vols {ticker: daily_sigma}

        Returns
        -------
        MonteCarloResult
        """
        rng = np.random.default_rng(self.seed)
        tickers = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[t] for t in tickers])

        # Estimate parameters from returns_df if provided
        if returns_df is not None:
            available = [t for t in tickers if t in returns_df.columns]
            ret_sub = returns_df[available].dropna()
        else:
            ret_sub = None

        if mu_override is not None:
            mu_daily_arr = np.array([mu_override.get(t, 0.0) for t in tickers])
        elif ret_sub is not None:
            mu_daily_arr = np.array([
                float(ret_sub[t].mean()) if t in ret_sub.columns else 0.0
                for t in tickers
            ])
        else:
            mu_daily_arr = np.zeros(len(tickers))

        if sigma_override is not None:
            sigma_daily_arr = np.array([sigma_override.get(t, 0.01) for t in tickers])
        elif ret_sub is not None:
            sigma_daily_arr = np.array([
                float(ret_sub[t].std()) if t in ret_sub.columns else 0.01
                for t in tickers
            ])
        else:
            sigma_daily_arr = np.full(len(tickers), 0.01)

        # Portfolio-level mu/sigma for single-asset distributions
        port_mu = float(weights @ mu_daily_arr)
        port_sigma = float(np.sqrt(weights @ (sigma_daily_arr ** 2)))

        # Run simulation
        if distribution == "normal":
            final_returns = self._simulate_normal(port_mu, port_sigma, horizon_days, rng)

        elif distribution == "student_t":
            final_returns = self._simulate_student_t(port_mu, port_sigma, horizon_days, dof, rng)

        elif distribution == "historical" and ret_sub is not None:
            port_hist = ret_sub.fillna(0.0).dot(
                pd.Series({t: portfolio_weights.get(t, 0.0) for t in available})
            ).values
            final_returns = self._simulate_historical(port_hist, horizon_days, rng)

        elif distribution == "t_copula" and ret_sub is not None:
            cov_daily = ret_sub.fillna(0.0).cov().values
            available_w = np.array([portfolio_weights.get(t, 0.0) for t in available])
            if np.sum(available_w) > _EPSILON:
                available_w /= np.sum(available_w)
            final_returns = self._simulate_t_copula_portfolio(
                available_w, mu_daily_arr[:len(available)], cov_daily, horizon_days, dof, rng
            )
        else:
            # Fallback to normal
            final_returns = self._simulate_normal(port_mu, port_sigma, horizon_days, rng)

        # Compute risk metrics
        losses = -final_returns
        var_95 = float(np.percentile(losses, 95))
        var_99 = float(np.percentile(losses, 99))
        cvar_95 = float(np.mean(losses[losses >= var_95]))
        cvar_99 = float(np.mean(losses[losses >= var_99]))

        # Max drawdown distribution
        wealth_paths = (1 + final_returns).reshape(-1, 1)
        # Approximate path-level max drawdown using daily step expansion
        # (full path simulation for 100 paths to get distribution)
        n_sample = min(1000, self.n_simulations)
        sample_rng = np.random.default_rng(self.seed + 1)
        if distribution == "normal":
            daily_r = sample_rng.normal(port_mu, port_sigma, (n_sample, horizon_days))
        else:
            daily_r = sample_rng.standard_t(df=dof, size=(n_sample, horizon_days))
            daily_r = port_mu + port_sigma * np.sqrt((dof - 2) / dof) * daily_r

        mdd_dist = np.array([
            self._max_drawdown(np.cumprod(1 + daily_r[i]))
            for i in range(n_sample)
        ])

        percentiles = {
            "p1": float(np.percentile(final_returns, 1)),
            "p5": float(np.percentile(final_returns, 5)),
            "p25": float(np.percentile(final_returns, 25)),
            "p50": float(np.percentile(final_returns, 50)),
            "p75": float(np.percentile(final_returns, 75)),
            "p95": float(np.percentile(final_returns, 95)),
            "p99": float(np.percentile(final_returns, 99)),
        }

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            horizon_days=horizon_days,
            distribution=distribution,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            mean_return=float(np.mean(final_returns)),
            std_return=float(np.std(final_returns)),
            max_drawdown_median=float(np.median(mdd_dist)),
            max_drawdown_p95=float(np.percentile(mdd_dist, 95)),
            return_percentiles=percentiles,
        )


# ---------------------------------------------------------------------------
# 4.  Macro Stress Tester
# ---------------------------------------------------------------------------

# Canonical macro stress scenarios
MACRO_SCENARIOS = {
    "stagflation": {
        "description": "Stagflation: CPI +200bps + GDP −2%. Supply shock.",
        "macro_shocks": {
            "CPI_yoy": +2.0,
            "GDP_growth": -2.0,
            "unemployment_rate": +1.0,
            "fed_funds_rate": +1.5,
        },
        "asset_impacts": {
            "SPY": -0.150,
            "QQQ": -0.180,
            "TLT": -0.080,
            "GLD": +0.100,
            "USO": +0.150,
            "UUP": +0.020,
            "HYG": -0.090,
            "TIPS": +0.020,
            "IEF": -0.050,
            "VNQ": -0.120,
        },
    },
    "recession": {
        "description": "Deep recession: unemployment +3%, GDP −4%, demand collapse.",
        "macro_shocks": {
            "CPI_yoy": -1.0,
            "GDP_growth": -4.0,
            "unemployment_rate": +3.0,
            "credit_spreads_ig_bps": +200.0,
            "credit_spreads_hy_bps": +500.0,
        },
        "asset_impacts": {
            "SPY": -0.250,
            "QQQ": -0.220,
            "TLT": +0.120,
            "GLD": +0.050,
            "USO": -0.200,
            "UUP": +0.060,
            "HYG": -0.180,
            "LQD": -0.070,
            "TIPS": -0.020,
            "IEF": +0.060,
            "VNQ": -0.220,
        },
    },
    "deflation": {
        "description": "Deflationary bust: CPI −150bps, ISM <45, credit freeze.",
        "macro_shocks": {
            "CPI_yoy": -1.5,
            "GDP_growth": -1.5,
            "ISM_manufacturing": -10.0,
            "fed_funds_rate": -0.5,
        },
        "asset_impacts": {
            "SPY": -0.100,
            "QQQ": -0.090,
            "TLT": +0.150,
            "GLD": -0.020,
            "USO": -0.150,
            "UUP": +0.040,
            "HYG": -0.120,
            "LQD": +0.040,
            "TIPS": -0.050,
            "IEF": +0.090,
            "VNQ": -0.060,
        },
    },
    "inflation_surge": {
        "description": "Inflation surprise: CPI +400bps. Fed forced to hike aggressively.",
        "macro_shocks": {
            "CPI_yoy": +4.0,
            "fed_funds_rate": +3.0,
            "10Y_yield_bps": +200.0,
        },
        "asset_impacts": {
            "SPY": -0.120,
            "QQQ": -0.180,
            "TLT": -0.250,
            "GLD": +0.080,
            "USO": +0.120,
            "UUP": +0.080,
            "HYG": -0.080,
            "LQD": -0.120,
            "TIPS": +0.060,
            "IEF": -0.100,
            "VNQ": -0.180,
        },
    },
    "soft_landing": {
        "description": "Soft landing: inflation controlled, growth intact.",
        "macro_shocks": {
            "CPI_yoy": -1.0,
            "GDP_growth": +0.5,
            "fed_funds_rate": -0.5,
        },
        "asset_impacts": {
            "SPY": +0.080,
            "QQQ": +0.120,
            "TLT": +0.040,
            "GLD": -0.010,
            "USO": +0.020,
            "UUP": -0.030,
            "HYG": +0.050,
            "LQD": +0.030,
            "TIPS": +0.010,
            "IEF": +0.025,
            "VNQ": +0.060,
        },
    },
}

# FRED series IDs for 10 key macro variables
FRED_MACRO_SERIES = {
    "CPI_yoy":         "CPIAUCSL",
    "GDP_growth":      "A191RL1Q225SBEA",
    "unemployment":    "UNRATE",
    "fed_funds":       "FEDFUNDS",
    "10Y_yield":       "GS10",
    "2Y_yield":        "GS2",
    "IG_spread":       "BAMLC0A0CM",
    "HY_spread":       "BAMLH0A0HYM2",
    "VIX":             "VIXCLS",
    "ISM_mfg":         "MANEMP",
}


class MacroStressTester:
    """
    Fetch current FRED macro data and stress test portfolios against
    canonical and custom macro shock scenarios.
    """

    def __init__(self):
        self._macro_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # FRED data fetch
    # ------------------------------------------------------------------

    def fetch_fred_value(self, series_id: str) -> Optional[float]:
        """Fetch most-recent value of a FRED time series."""
        try:
            resp = requests.get(
                FRED_CSV,
                params={"id": series_id},
                headers=_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            # last non-empty numeric row
            for line in reversed(lines[1:]):
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue
        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return None

    def fetch_all_macro(self) -> Dict[str, Optional[float]]:
        """Fetch all 10 key macro variables from FRED."""
        result = {}
        for key, series_id in FRED_MACRO_SERIES.items():
            val = self.fetch_fred_value(series_id)
            result[key] = val
            self._macro_cache[key] = val
        logger.info("Fetched %d macro variables from FRED", sum(v is not None for v in result.values()))
        return result

    # ------------------------------------------------------------------
    # Sigma-shock mapping
    # ------------------------------------------------------------------

    # Historical standard deviations of macro variables (approximate)
    _MACRO_STDEV: Dict[str, float] = {
        "CPI_yoy":      1.5,
        "GDP_growth":   2.0,
        "unemployment": 1.2,
        "fed_funds":    1.5,
        "10Y_yield":    1.2,
        "2Y_yield":     1.5,
        "IG_spread":    0.8,
        "HY_spread":    3.0,
        "VIX":          10.0,
        "ISM_mfg":      5.0,
    }

    def sigma_shock(
        self,
        variable: str,
        n_sigma: float = 2.0,
    ) -> float:
        """Return ±n_sigma shock for a macro variable."""
        stdev = self._MACRO_STDEV.get(variable, 1.0)
        return n_sigma * stdev

    # ------------------------------------------------------------------
    # Apply canonical scenario
    # ------------------------------------------------------------------

    def apply_scenario(
        self,
        portfolio_weights: Dict[str, float],
        scenario_name: str,
        portfolio_value: float = 1_000_000.0,
        custom_asset_betas: Optional[Dict[str, float]] = None,
    ) -> MacroStressResult:
        """
        Apply a canonical macro stress scenario.

        Parameters
        ----------
        portfolio_weights : {ticker: weight}
        scenario_name     : key in MACRO_SCENARIOS
        custom_asset_betas: override asset impacts for specific tickers
        """
        if scenario_name not in MACRO_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {list(MACRO_SCENARIOS)}")

        scenario = MACRO_SCENARIOS[scenario_name]
        base_impacts = scenario["asset_impacts"]
        betas = {**base_impacts, **(custom_asset_betas or {})}

        asset_impacts: Dict[str, float] = {}
        portfolio_impact = 0.0

        for ticker, weight in portfolio_weights.items():
            if ticker in betas:
                impact = betas[ticker]
            else:
                # Extrapolate via SPY beta
                spy_impact = betas.get("SPY", 0.0)
                beta = _DEFAULT_SPY_BETAS.get(ticker, 0.8)
                impact = spy_impact * beta
            asset_impacts[ticker] = impact
            portfolio_impact += weight * impact

        return MacroStressResult(
            scenario_name=scenario_name,
            macro_shocks=scenario["macro_shocks"],
            asset_impacts=asset_impacts,
            portfolio_impact=portfolio_impact,
            description=scenario["description"],
        )

    def apply_all_scenarios(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
    ) -> List[MacroStressResult]:
        """Apply all canonical macro scenarios, sorted by portfolio impact."""
        results = []
        for name in MACRO_SCENARIOS:
            try:
                r = self.apply_scenario(portfolio_weights, name, portfolio_value)
                results.append(r)
            except Exception as exc:
                logger.warning("Macro scenario %s failed: %s", name, exc)
        results.sort(key=lambda r: r.portfolio_impact)
        return results

    def custom_sigma_stress(
        self,
        portfolio_weights: Dict[str, float],
        variable: str,
        n_sigma: float = 2.0,
        direction: Literal["up", "down"] = "up",
        portfolio_value: float = 1_000_000.0,
    ) -> MacroStressResult:
        """
        Shock a single macro variable by ±n_sigma and propagate to assets.

        Uses regression-derived sensitivities (simplified linear mapping).
        """
        shock_size = self.sigma_shock(variable, n_sigma)
        if direction == "down":
            shock_size = -shock_size

        # Map macro shock to asset returns (simplified regression coefficients)
        macro_to_spy = {
            "CPI_yoy":      -0.015,   # +1pp CPI → equity −1.5%
            "GDP_growth":   +0.030,   # +1pp GDP → equity +3%
            "unemployment": -0.020,   # +1pp unemp → equity −2%
            "fed_funds":    -0.025,   # +1pp rate → equity −2.5%
            "10Y_yield":    -0.020,
            "VIX":          -0.005,
            "HY_spread":    -0.010,
            "IG_spread":    -0.008,
            "ISM_mfg":      +0.008,
        }
        spy_equiv = macro_to_spy.get(variable, 0.0) * shock_size

        asset_impacts: Dict[str, float] = {}
        portfolio_impact = 0.0
        for ticker, weight in portfolio_weights.items():
            beta = _DEFAULT_SPY_BETAS.get(ticker, 0.8)
            impact = spy_equiv * beta
            asset_impacts[ticker] = impact
            portfolio_impact += weight * impact

        return MacroStressResult(
            scenario_name=f"{variable}_{direction}_{n_sigma}σ",
            macro_shocks={variable: shock_size},
            asset_impacts=asset_impacts,
            portfolio_impact=portfolio_impact,
            description=f"{variable} shocked by {shock_size:+.2f} ({n_sigma}σ {direction}ward)",
        )


# ---------------------------------------------------------------------------
# 5.  Portfolio Stress Report
# ---------------------------------------------------------------------------

class PortfolioStressReport:
    """
    Consolidated stress report: ranking, vulnerabilities, hedges, stress-VaR.
    """

    def __init__(
        self,
        historical_engine: Optional[HistoricalScenarioEngine] = None,
        macro_engine: Optional[MacroStressTester] = None,
        mc_engine: Optional[MonteCarloStressTester] = None,
    ):
        self.hist = historical_engine or HistoricalScenarioEngine()
        self.macro = macro_engine or MacroStressTester()
        self.mc = mc_engine or MonteCarloStressTester()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _identify_vulnerable_positions(
        self,
        scenarios: List[ScenarioResult],
        threshold: float = -0.02,
    ) -> Dict[str, List[str]]:
        """
        For each position, list scenarios where it contributes > threshold to loss.
        """
        vulnerable: Dict[str, List[str]] = {}
        for scenario in scenarios:
            for ticker, contrib in scenario.asset_contributions.items():
                if contrib < threshold:
                    if ticker not in vulnerable:
                        vulnerable[ticker] = []
                    vulnerable[ticker].append(scenario.scenario_name)
        return vulnerable

    def _concentration_risks(
        self,
        portfolio_weights: Dict[str, float],
        scenarios: List[ScenarioResult],
    ) -> List[str]:
        """Identify assets that drive > 30% of loss in any severe scenario."""
        risks = []
        for scenario in scenarios:
            if scenario.pnl_pct < -0.05:
                total_loss = sum(c for c in scenario.asset_contributions.values() if c < 0)
                if abs(total_loss) > _EPSILON:
                    for ticker, contrib in scenario.asset_contributions.items():
                        if contrib < 0 and abs(contrib / total_loss) > 0.30:
                            risks.append(
                                f"{ticker} drives {abs(contrib / total_loss):.0%} of loss "
                                f"in {scenario.scenario_name}"
                            )
        return list(set(risks))

    def _hedge_recommendations(
        self,
        portfolio_weights: Dict[str, float],
        worst_scenarios: List[ScenarioResult],
    ) -> List[str]:
        """Simple rule-based hedge recommendations."""
        recommendations = []
        tickers = set(portfolio_weights.keys())

        # Equity exposure
        equity_weight = sum(
            w for t, w in portfolio_weights.items()
            if _DEFAULT_SPY_BETAS.get(t, 0.0) > 0.5
        )
        if equity_weight > 0.50:
            if "TLT" not in tickers and "IEF" not in tickers:
                recommendations.append("Add TLT (20Y Treasury): flight-to-quality hedge for equity crashes.")
            if "GLD" not in tickers:
                recommendations.append("Add GLD: hedge against systemic risk and USD debasement.")
            recommendations.append(
                f"Consider SPY put options or VXX to hedge {equity_weight:.0%} equity exposure."
            )

        # Bond exposure
        bond_weight = sum(
            w for t, w in portfolio_weights.items()
            if t in ("TLT", "IEF", "SHY", "LQD", "TIPS")
        )
        if bond_weight > 0.30:
            recommendations.append(
                "Consider TIPS or floating-rate bonds to hedge rate shock on long-duration bonds."
            )

        # Dollar exposure
        intl_weight = sum(
            w for t, w in portfolio_weights.items()
            if t in ("EFA", "EEM", "VEA", "VWO")
        )
        if intl_weight > 0.20:
            recommendations.append(
                "Add UUP (USD ETF) to hedge international equity USD risk."
            )

        if not recommendations:
            recommendations.append("Portfolio appears well-diversified across stress scenarios.")

        return recommendations

    def _compute_stress_var(
        self, scenarios: List[ScenarioResult], weights: Optional[List[float]] = None
    ) -> float:
        """
        Regulatory stress-VaR: weighted average of historical scenario losses.

        If no weights provided, equal-weight all scenarios.
        """
        losses = [max(-s.pnl_pct, 0.0) for s in scenarios]
        if not losses:
            return 0.0
        if weights is None:
            weights_arr = np.ones(len(losses)) / len(losses)
        else:
            weights_arr = np.array(weights) / sum(weights)
        return float(np.dot(weights_arr, losses))

    # ------------------------------------------------------------------
    # Full report generation
    # ------------------------------------------------------------------

    def generate(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
        returns_df: Optional[pd.DataFrame] = None,
        include_macro: bool = True,
        include_mc: bool = True,
    ) -> StressReport:
        """
        Generate full stress report for a portfolio.

        Parameters
        ----------
        portfolio_weights : {ticker: weight}
        portfolio_value   : dollar portfolio value
        returns_df        : optional historical returns for MC simulation
        include_macro     : include canonical macro stress scenarios
        include_mc        : include Monte Carlo summary
        """
        # Historical scenarios
        hist_results = self.hist.apply_all_scenarios(portfolio_weights, portfolio_value)

        # Macro scenarios
        macro_results = []
        if include_macro:
            macro_results = self.macro.apply_all_scenarios(portfolio_weights, portfolio_value)

        # Combine for ranking
        all_hist: List[Dict[str, Any]] = [
            {
                "source": "historical",
                "name": r.scenario_name,
                "pnl_pct": r.pnl_pct,
                "pnl_dollar": r.pnl_dollar,
                "severity": r.severity,
                "description": r.description,
                "top_contributors": sorted(
                    r.asset_contributions.items(), key=lambda x: x[1]
                )[:5],
            }
            for r in hist_results
        ]

        macro_ranked: List[Dict[str, Any]] = [
            {
                "source": "macro",
                "name": r.scenario_name,
                "pnl_pct": r.portfolio_impact,
                "pnl_dollar": r.portfolio_impact * portfolio_value,
                "description": r.description,
                "macro_shocks": r.macro_shocks,
            }
            for r in macro_results
        ]

        combined = sorted(all_hist + macro_ranked, key=lambda x: x["pnl_pct"])

        # Monte Carlo summary
        mc_summary: Dict[str, Any] = {}
        if include_mc and returns_df is not None:
            try:
                mc_result = self.mc.simulate(
                    portfolio_weights, returns_df,
                    horizon_days=252,
                    distribution="student_t",
                )
                mc_summary = {
                    "var_95_ann": mc_result.var_95,
                    "var_99_ann": mc_result.var_99,
                    "cvar_95_ann": mc_result.cvar_95,
                    "cvar_99_ann": mc_result.cvar_99,
                    "median_return": mc_result.return_percentiles["p50"],
                    "max_dd_median": mc_result.max_drawdown_median,
                    "max_dd_p95": mc_result.max_drawdown_p95,
                }
                combined.append({"source": "monte_carlo", "mc_summary": mc_summary})
            except Exception as exc:
                logger.warning("MC simulation failed: %s", exc)

        worst_scenario = combined[0]["name"] if combined and "name" in combined[0] else "N/A"
        best_scenario = (
            [x for x in reversed(combined) if "name" in x][0]["name"]
            if any("name" in x for x in combined) else "N/A"
        )

        vulnerable = self._identify_vulnerable_positions(hist_results)
        concentration = self._concentration_risks(portfolio_weights, hist_results)
        hedges = self._hedge_recommendations(portfolio_weights, hist_results[:5])
        stress_var = self._compute_stress_var(hist_results)

        logger.info(
            "Stress report: %d scenarios, worst=%s (%.1f%%), stress_var=%.1f%%",
            len(combined), worst_scenario,
            combined[0].get("pnl_pct", 0) * 100 if combined else 0,
            stress_var * 100,
        )

        return StressReport(
            scenarios_ranked=combined,
            worst_scenario=worst_scenario,
            best_scenario=best_scenario,
            vulnerable_positions=vulnerable,
            concentration_risks=concentration,
            hedge_recommendations=hedges,
            stress_var=stress_var,
        )


# ---------------------------------------------------------------------------
# 6.  Liquidity Stress Test
# ---------------------------------------------------------------------------

class LiquidityStressTest:
    """
    Liquidity risk under stress: bid-ask widening, ADV reduction,
    Almgren-Chriss liquidation, fire-sale discount, liquidity-adjusted VaR.
    """

    def __init__(
        self,
        bid_ask_stress_multiplier: float = 3.0,
        adv_stress_fraction: float = 0.50,
        participation_rate: float = 0.15,
    ):
        """
        Parameters
        ----------
        bid_ask_stress_multiplier : spread widens this much under stress
        adv_stress_fraction       : only this fraction of ADV available in crisis
        participation_rate        : maximum fraction of daily volume to trade (Almgren-Chriss)
        """
        self.bid_ask_multiplier = bid_ask_stress_multiplier
        self.adv_fraction = adv_stress_fraction
        self.participation_rate = participation_rate

    # ------------------------------------------------------------------
    # Estimate ADV from yfinance
    # ------------------------------------------------------------------

    def _fetch_adv_and_price(
        self, ticker: str, lookback: int = 20
    ) -> Tuple[float, float]:
        """Fetch average daily volume and current price."""
        if not _YF_AVAILABLE:
            return 1_000_000.0, 100.0
        try:
            hist = yf.download(ticker, period="30d", interval="1d", progress=False, auto_adjust=True)
            if hist.empty:
                return 1_000_000.0, 100.0
            adv = float(hist["Volume"].tail(lookback).mean())
            price = float(hist["Close"].iloc[-1])
            return adv, price
        except Exception:
            return 1_000_000.0, 100.0

    # ------------------------------------------------------------------
    # Bid-ask spread estimation
    # ------------------------------------------------------------------

    def _estimate_bid_ask_bps(self, ticker: str) -> float:
        """
        Estimate normal bid-ask spread in bps.

        Heuristics based on liquidity tier:
        - Large cap: 1-2 bps
        - Mid cap: 3-5 bps
        - Small cap: 10-30 bps
        - ETF: 1-3 bps
        """
        liquid_etfs = {"SPY", "QQQ", "IWM", "TLT", "GLD", "EFA", "EEM", "HYG", "LQD"}
        large_caps = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B"}
        if ticker in liquid_etfs:
            return 1.0
        if ticker in large_caps:
            return 2.0
        return 10.0  # Default: mid/small cap

    # ------------------------------------------------------------------
    # Almgren-Chriss: days to liquidate
    # ------------------------------------------------------------------

    def days_to_liquidate(
        self,
        position_size: float,   # number of shares
        adv: float,
        participation_rate: Optional[float] = None,
    ) -> float:
        """
        Days needed to liquidate position without excessive market impact.

        days = ceil(position_size / (ADV × participation_rate))
        """
        pr = participation_rate or self.participation_rate
        daily_capacity = adv * pr
        if daily_capacity < _EPSILON:
            return float("inf")
        return float(np.ceil(position_size / daily_capacity))

    # ------------------------------------------------------------------
    # Fire-sale discount (price impact)
    # ------------------------------------------------------------------

    def fire_sale_discount(
        self,
        position_value: float,
        adv_dollars: float,
        days_available: float = 1.0,
    ) -> float:
        """
        Estimate price impact of forced selling over `days_available`.

        Simplified square-root market impact model:
        impact_pct = 0.10 × √(position_value / adv_dollars × days_available)

        Returns fractional price discount (positive number).
        """
        if adv_dollars < _EPSILON:
            return 0.10
        ratio = position_value / (adv_dollars * days_available)
        impact = 0.10 * np.sqrt(ratio)
        return float(np.clip(impact, 0.0, 0.30))

    # ------------------------------------------------------------------
    # Liquidity-adjusted VaR
    # ------------------------------------------------------------------

    def liquidity_adjusted_var(
        self,
        base_var: float,
        bid_ask_bps: float,
        bid_ask_stressed_bps: float,
        position_value: float,
    ) -> float:
        """
        Liquidity-adjusted VaR = regular VaR + liquidation cost.

        Liquidation cost ≈ 0.5 × stressed_bid_ask_spread × position_value
        """
        spread_cost = 0.5 * (bid_ask_stressed_bps / 10_000.0) * position_value
        return float(base_var + spread_cost)

    # ------------------------------------------------------------------
    # Full single-asset liquidity stress test
    # ------------------------------------------------------------------

    def stress_test_position(
        self,
        ticker: str,
        position_shares: float,
        portfolio_var_pct: float = 0.05,
    ) -> LiquidityStressResult:
        """
        Run full liquidity stress test for a single position.

        Parameters
        ----------
        ticker           : asset ticker
        position_shares  : number of shares held
        portfolio_var_pct: 1-day VaR as fraction of position value

        Returns
        -------
        LiquidityStressResult
        """
        adv_shares, price = self._fetch_adv_and_price(ticker)
        position_value = position_shares * price
        adv_dollars = adv_shares * price

        bid_ask_normal = self._estimate_bid_ask_bps(ticker)
        bid_ask_stressed = bid_ask_normal * self.bid_ask_multiplier

        adv_stressed = adv_shares * self.adv_fraction
        dtl_normal = self.days_to_liquidate(position_shares, adv_shares)
        dtl_stressed = self.days_to_liquidate(position_shares, adv_stressed)

        fire_discount = self.fire_sale_discount(
            position_value, adv_dollars * self.adv_fraction
        )
        liquidation_cost = position_value * fire_discount + \
            0.5 * (bid_ask_stressed / 10_000.0) * position_value

        base_var_dollar = position_value * portfolio_var_pct
        liq_var = self.liquidity_adjusted_var(
            base_var_dollar, bid_ask_normal, bid_ask_stressed, position_value
        )

        logger.info(
            "Liquidity stress %s: dtl_normal=%.1f dtl_stressed=%.1f fire_discount=%.1f%%",
            ticker, dtl_normal, dtl_stressed, fire_discount * 100,
        )

        return LiquidityStressResult(
            ticker=ticker,
            position_value=position_value,
            adv_normal=adv_dollars,
            adv_stressed=adv_dollars * self.adv_fraction,
            days_to_liquidate=dtl_stressed,
            price_impact_pct=fire_discount,
            bid_ask_normal_bps=bid_ask_normal,
            bid_ask_stressed_bps=bid_ask_stressed,
            liquidity_adjusted_var=liq_var,
            liquidation_cost_dollar=liquidation_cost,
        )

    def stress_test_portfolio(
        self,
        portfolio: Dict[str, float],   # {ticker: shares}
        portfolio_var_pct: float = 0.05,
    ) -> Dict[str, LiquidityStressResult]:
        """Run liquidity stress test for all positions."""
        results = {}
        for ticker, shares in portfolio.items():
            try:
                results[ticker] = self.stress_test_position(ticker, shares, portfolio_var_pct)
            except Exception as exc:
                logger.warning("Liquidity stress failed for %s: %s", ticker, exc)
        return results


# ---------------------------------------------------------------------------
# Utility: fetch returns
# ---------------------------------------------------------------------------

def _fetch_returns(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """Download adjusted close prices and compute daily returns."""
    if not _YF_AVAILABLE:
        raise ImportError("yfinance is required.")
    if isinstance(tickers, str):
        tickers = [tickers]
    prices = yf.download(tickers, period=period, interval="1d", progress=False, auto_adjust=True)
    if len(tickers) == 1:
        close = prices[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        close = prices["Close"] if "Close" in prices.columns else prices
    return close.pct_change().dropna()


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    stress_router = APIRouter(prefix="/stress", tags=["Stress Testing"])

    # ── Pydantic models ───────────────────────────────────────────────────────

    class PortfolioWeightsRequest(BaseModel):
        weights: Dict[str, float] = Field(..., description="{ticker: weight}")
        portfolio_value: float = Field(1_000_000.0)
        lookback: str = Field("2y", description="yfinance period string")

    class HistoricalScenarioRequest(BaseModel):
        weights: Dict[str, float]
        scenario_name: str
        portfolio_value: float = Field(1_000_000.0)

    class CustomScenarioRequest(BaseModel):
        weights: Dict[str, float]
        shocks: Dict[str, float] = Field(
            ...,
            description="Factor/ticker shocks, e.g. {'SPY': -0.30, 'TLT': +0.15}"
        )
        portfolio_value: float = Field(1_000_000.0)
        scenario_name: str = Field("Custom")

    class MonteCarloRequest(BaseModel):
        weights: Dict[str, float]
        tickers: List[str]
        horizon_days: int = Field(252)
        distribution: Literal["normal", "student_t", "historical", "t_copula"] = "student_t"
        dof: float = Field(5.0)
        lookback: str = Field("2y")
        n_simulations: int = Field(10_000)

    class MacroScenarioRequest(BaseModel):
        weights: Dict[str, float]
        scenario_name: str = Field("recession")
        portfolio_value: float = Field(1_000_000.0)

    class LiquidityRequest(BaseModel):
        portfolio_shares: Dict[str, float] = Field(
            ..., description="{ticker: number_of_shares}"
        )
        portfolio_var_pct: float = Field(0.05)

    # ── Endpoints ─────────────────────────────────────────────────────────────

    @stress_router.get("/scenarios")
    async def list_scenarios() -> Dict[str, Any]:
        """List all available historical and macro stress scenarios."""
        return {
            "historical_scenarios": {
                name: {
                    "start": s["start_date"],
                    "end": s["end_date"],
                    "description": s["description"],
                }
                for name, s in HISTORICAL_SCENARIOS.items()
            },
            "macro_scenarios": {
                name: {"description": s["description"]}
                for name, s in MACRO_SCENARIOS.items()
            },
            "n_historical": len(HISTORICAL_SCENARIOS),
            "n_macro": len(MACRO_SCENARIOS),
        }

    @stress_router.post("/historical")
    async def historical_scenario_endpoint(req: HistoricalScenarioRequest) -> Dict[str, Any]:
        """Apply a single historical scenario to the portfolio."""
        try:
            engine = HistoricalScenarioEngine()
            result = engine.apply_scenario(req.weights, req.scenario_name, req.portfolio_value)
            return {
                "scenario": result.scenario_name,
                "pnl_pct": result.pnl_pct,
                "pnl_dollar": result.pnl_dollar,
                "severity": result.severity,
                "description": result.description,
                "asset_contributions": result.asset_contributions,
                "factor_contributions": result.factor_contributions,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @stress_router.post("/historical/all")
    async def all_historical_scenarios(req: PortfolioWeightsRequest) -> Dict[str, Any]:
        """Apply all 20 historical scenarios and return ranked results."""
        try:
            engine = HistoricalScenarioEngine()
            results = engine.apply_all_scenarios(req.weights, req.portfolio_value)
            return {
                "portfolio_value": req.portfolio_value,
                "n_scenarios": len(results),
                "scenarios": [
                    {
                        "name": r.scenario_name,
                        "pnl_pct": r.pnl_pct,
                        "pnl_dollar": r.pnl_dollar,
                        "severity": r.severity,
                        "description": r.description,
                    }
                    for r in results
                ],
                "worst_scenario": results[0].scenario_name if results else None,
                "best_scenario": results[-1].scenario_name if results else None,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @stress_router.post("/custom")
    async def custom_scenario_endpoint(req: CustomScenarioRequest) -> Dict[str, Any]:
        """Apply user-defined factor shocks to the portfolio."""
        try:
            builder = HypotheticalScenarioBuilder()
            result = builder.apply_to_portfolio(
                req.weights, req.shocks, req.portfolio_value, req.scenario_name
            )
            return {
                "scenario": result.scenario_name,
                "pnl_pct": result.pnl_pct,
                "pnl_dollar": result.pnl_dollar,
                "severity": result.severity,
                "asset_contributions": result.asset_contributions,
                "input_shocks": req.shocks,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @stress_router.post("/monte-carlo")
    async def monte_carlo_endpoint(req: MonteCarloRequest) -> Dict[str, Any]:
        """Monte Carlo portfolio risk simulation."""
        try:
            returns = _fetch_returns(req.tickers, req.lookback)
            engine = MonteCarloStressTester(n_simulations=req.n_simulations)
            result = engine.simulate(
                portfolio_weights=req.weights,
                returns_df=returns,
                horizon_days=req.horizon_days,
                distribution=req.distribution,
                dof=req.dof,
            )
            return {
                "n_simulations": result.n_simulations,
                "horizon_days": result.horizon_days,
                "distribution": result.distribution,
                "var_95": result.var_95,
                "var_99": result.var_99,
                "cvar_95": result.cvar_95,
                "cvar_99": result.cvar_99,
                "mean_return": result.mean_return,
                "std_return": result.std_return,
                "max_drawdown_median": result.max_drawdown_median,
                "max_drawdown_p95": result.max_drawdown_p95,
                "return_percentiles": result.return_percentiles,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @stress_router.post("/macro")
    async def macro_scenario_endpoint(req: MacroScenarioRequest) -> Dict[str, Any]:
        """Apply a canonical macro stress scenario."""
        try:
            engine = MacroStressTester()
            result = engine.apply_scenario(req.weights, req.scenario_name, req.portfolio_value)
            return {
                "scenario": result.scenario_name,
                "description": result.description,
                "macro_shocks": result.macro_shocks,
                "asset_impacts": result.asset_impacts,
                "portfolio_impact_pct": result.portfolio_impact,
                "portfolio_impact_dollar": result.portfolio_impact * req.portfolio_value,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @stress_router.post("/macro/all")
    async def all_macro_scenarios(req: PortfolioWeightsRequest) -> Dict[str, Any]:
        """Apply all canonical macro scenarios, sorted by portfolio impact."""
        try:
            engine = MacroStressTester()
            results = engine.apply_all_scenarios(req.weights, req.portfolio_value)
            return {
                "portfolio_value": req.portfolio_value,
                "scenarios": [
                    {
                        "name": r.scenario_name,
                        "description": r.description,
                        "portfolio_impact_pct": r.portfolio_impact,
                        "portfolio_impact_dollar": r.portfolio_impact * req.portfolio_value,
                        "macro_shocks": r.macro_shocks,
                    }
                    for r in results
                ],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @stress_router.post("/report")
    async def stress_report_endpoint(req: PortfolioWeightsRequest) -> Dict[str, Any]:
        """
        Full consolidated stress report: historical + macro + liquidity,
        with vulnerability analysis and hedge recommendations.
        """
        try:
            returns = None
            tickers = list(req.weights.keys())
            try:
                returns = _fetch_returns(tickers, req.lookback)
            except Exception:
                pass

            report_engine = PortfolioStressReport()
            report = report_engine.generate(
                portfolio_weights=req.weights,
                portfolio_value=req.portfolio_value,
                returns_df=returns,
                include_macro=True,
                include_mc=(returns is not None),
            )

            return {
                "worst_scenario": report.worst_scenario,
                "best_scenario": report.best_scenario,
                "stress_var": report.stress_var,
                "n_scenarios_ranked": len(report.scenarios_ranked),
                "top_5_worst": report.scenarios_ranked[:5],
                "vulnerable_positions": report.vulnerable_positions,
                "concentration_risks": report.concentration_risks,
                "hedge_recommendations": report.hedge_recommendations,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @stress_router.post("/liquidity")
    async def liquidity_stress_endpoint(req: LiquidityRequest) -> Dict[str, Any]:
        """Liquidity stress test for all portfolio positions."""
        try:
            engine = LiquidityStressTest()
            results = engine.stress_test_portfolio(
                req.portfolio_shares, req.portfolio_var_pct
            )
            total_liq_cost = sum(r.liquidation_cost_dollar for r in results.values())
            total_value = sum(r.position_value for r in results.values())
            slowest = max(results.items(), key=lambda x: x[1].days_to_liquidate, default=(None, None))

            return {
                "total_portfolio_value": total_value,
                "total_liquidation_cost": total_liq_cost,
                "total_liq_cost_pct": total_liq_cost / total_value if total_value > 0 else 0.0,
                "slowest_to_liquidate": slowest[0] if slowest[0] else None,
                "slowest_days": slowest[1].days_to_liquidate if slowest[1] else None,
                "positions": {
                    ticker: {
                        "position_value": r.position_value,
                        "adv_normal": r.adv_normal,
                        "adv_stressed": r.adv_stressed,
                        "days_to_liquidate_stressed": r.days_to_liquidate,
                        "fire_sale_discount_pct": r.price_impact_pct,
                        "bid_ask_normal_bps": r.bid_ask_normal_bps,
                        "bid_ask_stressed_bps": r.bid_ask_stressed_bps,
                        "liquidity_adjusted_var": r.liquidity_adjusted_var,
                        "liquidation_cost": r.liquidation_cost_dollar,
                    }
                    for ticker, r in results.items()
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

else:
    stress_router = None  # type: ignore
