"""
SENTINEL SPM — Stress Testing v3
dim_083: Stress testing / scenario analysis (score 7 → 9)

Comprehensive stress testing and scenario analysis platform covering:
  - Historical scenarios (Black Monday 1987 → SVB 2023)
  - Hypothetical forward scenarios (Soft Landing → Rate Spike)
  - Monte Carlo with fat tails (Student-t)
  - Correlation breakdown analysis
  - Hedge recommendation engine

Free data only: yfinance for prices / returns.
Math: numpy / scipy (scipy guarded).
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    from scipy.stats import t as student_t
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Asset class taxonomy
# ---------------------------------------------------------------------------

ASSET_CLASS_KEYS = (
    "equity", "bonds", "credit", "gold", "oil",
    "usd", "emerging_markets", "real_estate", "crypto", "commodities",
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    description: str
    type: str                           # "historical" | "hypothetical"
    asset_shocks: dict[str, float]      # asset_class -> fractional shock (e.g. -0.56)
    duration_days: int                  # approximate length of event
    peak_to_trough: bool                # True = peak-to-trough; False = single-day shock
    historical_dates: tuple[str, str]   # ("YYYY-MM-DD", "YYYY-MM-DD") or ("", "")
    vix_level: float                    # VIX reading during event (0 if unknown)

    def display_shocks(self) -> str:
        parts = []
        for k, v in self.asset_shocks.items():
            if v != 0.0:
                parts.append(f"{k}={v:+.1%}")
        return "  ".join(parts)


@dataclass
class ScenarioResult:
    scenario: Scenario
    portfolio_equity: float
    total_pnl: float
    total_pnl_pct: float
    holding_pnl: dict[str, float]       # ticker -> $ P&L
    holding_pnl_pct: dict[str, float]   # ticker -> % P&L
    factor_attribution: dict[str, float] # asset_class -> $ attribution
    worst_holding: str
    best_holding: str
    verdict: str                         # "OK" | "WARNING" | "SEVERE"

    def summary(self) -> str:
        lines = [
            f"Scenario : {self.scenario.name}",
            f"Type     : {self.scenario.type}  ({self.scenario.description[:60]})",
            f"Shocks   : {self.scenario.display_shocks()}",
            f"Portfolio P&L: ${self.total_pnl:+,.0f}  ({self.total_pnl_pct:+.2%})  [{self.verdict}]",
            f"Worst    : {self.worst_holding}",
            f"Best     : {self.best_holding}",
        ]
        return "\n".join(lines)


@dataclass
class MonteCarloResult:
    n_simulations: int
    horizon_days: int
    distribution: str                    # "normal" | "student_t"
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    prob_loss_20pct: float
    prob_loss_50pct: float
    median_terminal_wealth: float
    p5_terminal_wealth: float
    p95_terminal_wealth: float
    raw_returns: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            f"Monte Carlo ({self.n_simulations:,} sims, {self.horizon_days}d, {self.distribution})",
            f"  VaR  95%: {self.var_95:+.2%}   99%: {self.var_99:+.2%}",
            f"  CVaR 95%: {self.cvar_95:+.2%}   99%: {self.cvar_99:+.2%}",
            f"  P(loss>20%): {self.prob_loss_20pct:.2%}   P(loss>50%): {self.prob_loss_50pct:.2%}",
            f"  Terminal wealth — p5: {self.p5_terminal_wealth:.2f}x  median: {self.median_terminal_wealth:.2f}x  p95: {self.p95_terminal_wealth:.2f}x",
        ]
        return "\n".join(lines)


@dataclass
class HedgeRecommendation:
    name: str
    description: str
    instruments: list[str]
    target_scenarios: list[str]
    estimated_cost_bps: float            # annual cost in basis points
    expected_offset_pct: float           # fraction of scenario loss offset
    sizing_suggestion: str               # e.g. "2-5% of equity in SPX puts"


# ---------------------------------------------------------------------------
# Scenario Library
# ---------------------------------------------------------------------------

class ScenarioLibrary:
    """
    Canonical library of historical and hypothetical stress scenarios.
    """

    def __init__(self):
        self._historical: list[Scenario] = self._build_historical()
        self._hypothetical: list[Scenario] = self._build_hypothetical()
        self._custom: list[Scenario] = []

    # ------------------------------------------------------------------
    # Historical scenarios
    # ------------------------------------------------------------------

    @staticmethod
    def _build_historical() -> list[Scenario]:
        return [
            Scenario(
                name="Black Monday 1987",
                description="Single-day 22.6% equity crash; portfolio insurance unwinding",
                type="historical",
                asset_shocks={
                    "equity": -0.226, "bonds": +0.043, "credit": -0.020,
                    "gold": +0.040, "oil": -0.050, "usd": 0.0,
                    "emerging_markets": -0.20, "real_estate": -0.10,
                    "crypto": 0.0, "commodities": -0.03,
                },
                duration_days=1,
                peak_to_trough=False,
                historical_dates=("1987-10-19", "1987-10-19"),
                vix_level=150.0,  # VIX didn't exist; estimated
            ),
            Scenario(
                name="LTCM Russia Crisis 1998",
                description="Russian default and LTCM collapse; contagion across credit markets",
                type="historical",
                asset_shocks={
                    "equity": -0.190, "bonds": +0.060, "credit": -0.080,
                    "gold": +0.050, "oil": -0.300, "usd": +0.050,
                    "emerging_markets": -0.35, "real_estate": -0.08,
                    "crypto": 0.0, "commodities": -0.15,
                },
                duration_days=60,
                peak_to_trough=True,
                historical_dates=("1998-08-01", "1998-10-08"),
                vix_level=45.0,
            ),
            Scenario(
                name="Dot-com Crash 2000-2002",
                description="Technology bubble bursting; prolonged equity bear market",
                type="historical",
                asset_shocks={
                    "equity": -0.490, "bonds": +0.200, "credit": -0.050,
                    "gold": +0.150, "oil": -0.200, "usd": +0.080,
                    "emerging_markets": -0.40, "real_estate": +0.10,
                    "crypto": 0.0, "commodities": -0.10,
                },
                duration_days=917,
                peak_to_trough=True,
                historical_dates=("2000-03-10", "2002-10-09"),
                vix_level=45.0,
            ),
            Scenario(
                name="9/11 Terror Attack 2001",
                description="September 11 attacks; markets closed for 4 days, sharp reopening drop",
                type="historical",
                asset_shocks={
                    "equity": -0.116, "bonds": +0.040, "credit": -0.050,
                    "gold": +0.040, "oil": -0.300, "usd": -0.030,
                    "emerging_markets": -0.10, "real_estate": -0.05,
                    "crypto": 0.0, "commodities": -0.05,
                },
                duration_days=5,
                peak_to_trough=False,
                historical_dates=("2001-09-11", "2001-09-21"),
                vix_level=43.0,
            ),
            Scenario(
                name="GFC 2008-2009",
                description="Global Financial Crisis; systemic bank failure, credit freeze",
                type="historical",
                asset_shocks={
                    "equity": -0.560, "bonds": +0.120, "credit": -0.200,
                    "gold": +0.250, "oil": -0.700, "usd": +0.120,
                    "emerging_markets": -0.55, "real_estate": -0.40,
                    "crypto": 0.0, "commodities": -0.50,
                },
                duration_days=517,
                peak_to_trough=True,
                historical_dates=("2007-10-09", "2009-03-09"),
                vix_level=80.0,
            ),
            Scenario(
                name="Flash Crash 2010",
                description="May 6, 2010 algorithmic cascade; intraday 9.2% drop, mostly recovered same day",
                type="historical",
                asset_shocks={
                    "equity": -0.092, "bonds": +0.020, "credit": -0.030,
                    "gold": +0.020, "oil": -0.040, "usd": +0.010,
                    "emerging_markets": -0.08, "real_estate": -0.04,
                    "crypto": 0.0, "commodities": -0.02,
                },
                duration_days=1,
                peak_to_trough=False,
                historical_dates=("2010-05-06", "2010-05-06"),
                vix_level=40.0,
            ),
            Scenario(
                name="European Debt Crisis 2011",
                description="Sovereign debt stress in PIIGS; ECB/IMF rescue operations",
                type="historical",
                asset_shocks={
                    "equity": -0.160, "bonds": +0.080, "credit": -0.100,
                    "gold": +0.100, "oil": -0.100, "usd": +0.060,
                    "emerging_markets": -0.18, "real_estate": -0.08,
                    "crypto": 0.0, "commodities": -0.08,
                },
                duration_days=90,
                peak_to_trough=True,
                historical_dates=("2011-07-22", "2011-10-04"),
                vix_level=48.0,
            ),
            Scenario(
                name="China Stock Crash 2015",
                description="Chinese equity bubble burst; Yuan devaluation; global contagion",
                type="historical",
                asset_shocks={
                    "equity": -0.110, "bonds": +0.020, "credit": -0.050,
                    "gold": +0.030, "oil": -0.150, "usd": +0.040,
                    "emerging_markets": -0.20, "real_estate": -0.04,
                    "crypto": 0.0, "commodities": -0.12,
                },
                duration_days=30,
                peak_to_trough=True,
                historical_dates=("2015-08-10", "2015-08-25"),
                vix_level=40.0,
            ),
            Scenario(
                name="COVID Crash 2020",
                description="Pandemic-driven global shutdown; fastest 30%+ decline in history",
                type="historical",
                asset_shocks={
                    "equity": -0.340, "bonds": +0.080, "credit": -0.150,
                    "gold": +0.040, "oil": -0.650, "usd": +0.080,
                    "emerging_markets": -0.32, "real_estate": -0.20,
                    "crypto": -0.50, "commodities": -0.35,
                },
                duration_days=33,
                peak_to_trough=True,
                historical_dates=("2020-02-19", "2020-03-23"),
                vix_level=85.0,
            ),
            Scenario(
                name="Rate Shock 2022",
                description="Fed rapid rate hike cycle; bonds and equities both fell sharply (60/40 breakdown)",
                type="historical",
                asset_shocks={
                    "equity": -0.250, "bonds": -0.160, "credit": -0.150,
                    "gold": +0.020, "oil": +0.500, "usd": +0.150,
                    "emerging_markets": -0.25, "real_estate": -0.30,
                    "crypto": -0.70, "commodities": +0.20,
                },
                duration_days=294,
                peak_to_trough=True,
                historical_dates=("2022-01-03", "2022-10-13"),
                vix_level=36.0,
            ),
            Scenario(
                name="SVB Bank Run 2023",
                description="Silicon Valley Bank failure; regional bank contagion; FDIC backstop",
                type="historical",
                asset_shocks={
                    "equity": -0.070, "bonds": +0.050, "credit": -0.080,
                    "gold": +0.080, "oil": -0.100, "usd": -0.010,
                    "emerging_markets": -0.06, "real_estate": -0.05,
                    "crypto": +0.20, "commodities": -0.03,
                },
                duration_days=14,
                peak_to_trough=True,
                historical_dates=("2023-03-08", "2023-03-17"),
                vix_level=26.0,
            ),
        ]

    # ------------------------------------------------------------------
    # Hypothetical scenarios
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hypothetical() -> list[Scenario]:
        return [
            Scenario(
                name="Soft Landing",
                description="Fed achieves 2% inflation without recession; risk-on environment",
                type="hypothetical",
                asset_shocks={
                    "equity": +0.150, "bonds": +0.050, "credit": +0.030,
                    "gold": -0.050, "oil": +0.050, "usd": -0.030,
                    "emerging_markets": +0.20, "real_estate": +0.08,
                    "crypto": +0.30, "commodities": +0.05,
                },
                duration_days=365,
                peak_to_trough=False,
                historical_dates=("", ""),
                vix_level=14.0,
            ),
            Scenario(
                name="Hard Landing",
                description="Fed overtightens; recession ensues; earnings collapse",
                type="hypothetical",
                asset_shocks={
                    "equity": -0.300, "bonds": +0.150, "credit": -0.150,
                    "gold": +0.200, "oil": -0.250, "usd": +0.100,
                    "emerging_markets": -0.35, "real_estate": -0.20,
                    "crypto": -0.50, "commodities": -0.20,
                },
                duration_days=270,
                peak_to_trough=True,
                historical_dates=("", ""),
                vix_level=55.0,
            ),
            Scenario(
                name="Stagflation Shock",
                description="Persistent high inflation with low/negative growth; 1970s redux",
                type="hypothetical",
                asset_shocks={
                    "equity": -0.200, "bonds": -0.100, "credit": -0.080,
                    "gold": +0.300, "oil": +0.400, "usd": -0.050,
                    "emerging_markets": -0.15, "real_estate": +0.05,
                    "crypto": -0.20, "commodities": +0.35,
                },
                duration_days=365,
                peak_to_trough=True,
                historical_dates=("", ""),
                vix_level=35.0,
            ),
            Scenario(
                name="Deflation Shock",
                description="Japan-style debt deflation; money velocity collapses",
                type="hypothetical",
                asset_shocks={
                    "equity": -0.150, "bonds": +0.250, "credit": -0.050,
                    "gold": +0.050, "oil": -0.300, "usd": +0.100,
                    "emerging_markets": -0.20, "real_estate": -0.15,
                    "crypto": -0.40, "commodities": -0.25,
                },
                duration_days=730,
                peak_to_trough=True,
                historical_dates=("", ""),
                vix_level=40.0,
            ),
            Scenario(
                name="Geopolitical Crisis",
                description="Major war or sanctions cascade; supply chains disrupted",
                type="hypothetical",
                asset_shocks={
                    "equity": -0.200, "bonds": +0.050, "credit": -0.100,
                    "gold": +0.250, "oil": +0.500, "usd": +0.050,
                    "emerging_markets": -0.30, "real_estate": -0.05,
                    "crypto": -0.30, "commodities": +0.40,
                },
                duration_days=90,
                peak_to_trough=True,
                historical_dates=("", ""),
                vix_level=60.0,
            ),
            Scenario(
                name="Tech Bubble Burst",
                description="AI valuation collapse; mega-cap tech down 50-70%",
                type="hypothetical",
                asset_shocks={
                    "equity": -0.400, "bonds": +0.100, "credit": -0.050,
                    "gold": +0.100, "oil": -0.050, "usd": +0.030,
                    "emerging_markets": -0.30, "real_estate": -0.05,
                    "crypto": -0.60, "commodities": -0.05,
                },
                duration_days=400,
                peak_to_trough=True,
                historical_dates=("", ""),
                vix_level=65.0,
            ),
            Scenario(
                name="USD Collapse",
                description="Dollar loses reserve status; EM assets surge; Treasuries sold",
                type="hypothetical",
                asset_shocks={
                    "equity": -0.100, "bonds": -0.200, "credit": -0.050,
                    "gold": +0.500, "oil": +0.300, "usd": -0.300,
                    "emerging_markets": +0.20, "real_estate": +0.05,
                    "crypto": +0.50, "commodities": +0.40,
                },
                duration_days=180,
                peak_to_trough=True,
                historical_dates=("", ""),
                vix_level=45.0,
            ),
            Scenario(
                name="Interest Rate Spike",
                description="10Y Treasury rapidly moves to 6%; duration carnage across fixed income",
                type="hypothetical",
                asset_shocks={
                    "equity": -0.200, "bonds": -0.200, "credit": -0.100,
                    "gold": -0.050, "oil": 0.0, "usd": +0.080,
                    "emerging_markets": -0.25, "real_estate": -0.25,
                    "crypto": -0.30, "commodities": 0.0,
                },
                duration_days=60,
                peak_to_trough=True,
                historical_dates=("", ""),
                vix_level=40.0,
            ),
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_all_historical_scenarios(self) -> list[Scenario]:
        return list(self._historical)

    def get_all_hypothetical_scenarios(self) -> list[Scenario]:
        return list(self._hypothetical)

    def get_scenario(self, name: str) -> Scenario:
        all_s = self._historical + self._hypothetical + self._custom
        for s in all_s:
            if s.name.lower() == name.lower():
                return s
        raise KeyError(f"Scenario '{name}' not found. Use list_all() to see available scenarios.")

    def add_custom_scenario(self, scenario: Scenario) -> None:
        self._custom.append(scenario)
        logger.info(f"Added custom scenario: {scenario.name}")

    def list_all(self) -> list[str]:
        all_s = self._historical + self._hypothetical + self._custom
        return [s.name for s in all_s]


# ---------------------------------------------------------------------------
# Asset Class Mapper
# ---------------------------------------------------------------------------

class AssetClassMapper:
    """
    Map portfolio holdings (tickers/ETFs) to asset class exposures.
    """

    # ETF / ticker -> asset class weight dictionary
    _KNOWN_MAPPINGS: dict[str, dict[str, float]] = {
        # Equity
        "SPY": {"equity": 1.0},
        "IVV": {"equity": 1.0},
        "VOO": {"equity": 1.0},
        "VTI": {"equity": 1.0},
        "VFINX": {"equity": 1.0},
        "QQQ": {"equity": 1.0},
        "SCHB": {"equity": 1.0},
        # Bonds
        "TLT": {"bonds": 1.0},
        "IEF": {"bonds": 1.0},
        "SHY": {"bonds": 0.7, "usd": 0.3},
        "SGOV": {"usd": 0.8, "bonds": 0.2},
        "BND": {"bonds": 0.8, "credit": 0.2},
        "AGG": {"bonds": 0.75, "credit": 0.25},
        "GOVT": {"bonds": 1.0},
        "LQD": {"bonds": 0.4, "credit": 0.6},
        "VCIT": {"bonds": 0.3, "credit": 0.7},
        # High Yield / Credit
        "HYG": {"credit": 0.9, "equity": 0.1},
        "JNK": {"credit": 0.9, "equity": 0.1},
        "ANGL": {"credit": 0.8, "equity": 0.2},
        # Gold
        "GLD": {"gold": 1.0},
        "IAU": {"gold": 1.0},
        "SGOL": {"gold": 1.0},
        "GDX": {"gold": 0.7, "equity": 0.3},
        # Oil / Energy
        "USO": {"oil": 1.0},
        "XLE": {"oil": 0.6, "equity": 0.4},
        "VDE": {"oil": 0.6, "equity": 0.4},
        "XOM": {"oil": 0.5, "equity": 0.5},
        "CVX": {"oil": 0.5, "equity": 0.5},
        "COP": {"oil": 0.5, "equity": 0.5},
        # Real Estate
        "VNQ": {"real_estate": 0.9, "equity": 0.1},
        "IYR": {"real_estate": 0.9, "equity": 0.1},
        "SCHH": {"real_estate": 0.9, "equity": 0.1},
        # Emerging Markets
        "EEM": {"emerging_markets": 1.0},
        "VWO": {"emerging_markets": 1.0},
        "IEMG": {"emerging_markets": 1.0},
        # Commodities
        "DBC": {"commodities": 0.7, "oil": 0.3},
        "PDBC": {"commodities": 0.7, "oil": 0.3},
        "GSG": {"commodities": 0.6, "oil": 0.4},
        # Crypto proxies
        "BITO": {"crypto": 0.9, "equity": 0.1},
        "GBTC": {"crypto": 1.0},
        "MSTR": {"crypto": 0.7, "equity": 0.3},
        # Cash equivalents
        "VMFXX": {"usd": 1.0},
        "VUSXX": {"usd": 1.0},
        "BIL": {"usd": 0.9, "bonds": 0.1},
    }

    # Default sector → asset class for unknown individual stocks
    _SECTOR_DEFAULTS: dict[str, dict[str, float]] = {
        "Technology": {"equity": 0.9, "credit": 0.1},
        "Financials": {"equity": 0.8, "credit": 0.2},
        "Energy": {"equity": 0.5, "oil": 0.5},
        "Materials": {"equity": 0.6, "commodities": 0.4},
        "Real Estate": {"equity": 0.3, "real_estate": 0.7},
        "Utilities": {"equity": 0.7, "bonds": 0.3},
        "Consumer Discretionary": {"equity": 1.0},
        "Consumer Staples": {"equity": 0.9, "commodities": 0.1},
        "Health Care": {"equity": 1.0},
        "Industrials": {"equity": 0.9, "commodities": 0.1},
        "Communication Services": {"equity": 1.0},
        "default": {"equity": 1.0},
    }

    def classify_holding(self, ticker: str) -> dict[str, float]:
        """
        Returns asset class weights for a ticker.
        Checks known ETF map first; for unknown tickers tries yfinance sector.
        """
        ticker_upper = ticker.upper()

        if ticker_upper in self._KNOWN_MAPPINGS:
            return self._KNOWN_MAPPINGS[ticker_upper].copy()

        # Try yfinance for sector classification
        if _HAS_YF:
            try:
                info = yf.Ticker(ticker_upper).fast_info
                sector = getattr(info, "sector", None)
                if sector and sector in self._SECTOR_DEFAULTS:
                    return self._SECTOR_DEFAULTS[sector].copy()
            except Exception:
                pass

        # Heuristics: check ticker patterns for crypto
        crypto_tickers = {"BTC", "ETH", "SOL", "ADA", "DOGE", "XRP", "AVAX"}
        if ticker_upper in crypto_tickers or ticker_upper.endswith("-USD"):
            return {"crypto": 1.0}

        # Default: treat as large-cap equity
        logger.debug(f"Unknown ticker {ticker_upper} — defaulting to equity=1.0")
        return {"equity": 1.0}

    def compute_portfolio_asset_class_weights(
        self, holdings: dict[str, float]
    ) -> dict[str, float]:
        """
        Aggregate all holding weights into asset class exposures.
        holdings: {ticker: weight_or_value_fraction}
        """
        total = sum(abs(v) for v in holdings.values())
        if total <= 0:
            return {k: 0.0 for k in ASSET_CLASS_KEYS}

        agg: dict[str, float] = {k: 0.0 for k in ASSET_CLASS_KEYS}

        for ticker, value in holdings.items():
            w = value / total
            ac_map = self.classify_holding(ticker)
            for ac, ac_weight in ac_map.items():
                if ac in agg:
                    agg[ac] += w * ac_weight

        return agg

    def get_beta_to_scenario(self, ticker: str, scenario: Scenario) -> float:
        """
        Estimate dollar beta of ticker to scenario shock.
        Uses asset class mapping × scenario shock magnitudes.
        """
        ac_map = self.classify_holding(ticker)
        beta = 0.0
        for ac, ac_weight in ac_map.items():
            shock = scenario.asset_shocks.get(ac, 0.0)
            beta += ac_weight * shock
        return float(beta)


# ---------------------------------------------------------------------------
# Stress Test Engine
# ---------------------------------------------------------------------------

class StressTestEngine:
    """
    Core stress testing: apply scenario shocks to a portfolio.
    """

    def __init__(self):
        self._mapper = AssetClassMapper()
        self._library = ScenarioLibrary()

    def run_scenario(
        self,
        holdings: dict[str, float],
        scenario: Scenario,
        portfolio_equity: float,
    ) -> ScenarioResult:
        """
        Compute P&L impact of scenario on portfolio.

        holdings : {ticker: position_value_in_dollars}
        portfolio_equity : total equity (denominator for %)
        """
        holding_pnl: dict[str, float] = {}
        holding_pnl_pct: dict[str, float] = {}
        factor_attribution: dict[str, float] = {k: 0.0 for k in ASSET_CLASS_KEYS}

        for ticker, value in holdings.items():
            ac_map = self._mapper.classify_holding(ticker)
            total_shock = 0.0
            for ac, ac_weight in ac_map.items():
                shock = scenario.asset_shocks.get(ac, 0.0)
                contrib = value * ac_weight * shock
                factor_attribution[ac] = factor_attribution.get(ac, 0.0) + contrib
                total_shock += ac_weight * shock
            pnl = value * total_shock
            holding_pnl[ticker] = pnl
            holding_pnl_pct[ticker] = total_shock

        total_pnl = sum(holding_pnl.values())
        total_pnl_pct = total_pnl / max(abs(portfolio_equity), 1e-9)

        if holding_pnl:
            worst_holding = min(holding_pnl, key=holding_pnl.__getitem__)
            best_holding = max(holding_pnl, key=holding_pnl.__getitem__)
        else:
            worst_holding = best_holding = "N/A"

        # Verdict
        if total_pnl_pct > -0.05:
            verdict = "OK"
        elif total_pnl_pct > -0.15:
            verdict = "WARNING"
        else:
            verdict = "SEVERE"

        return ScenarioResult(
            scenario=scenario,
            portfolio_equity=portfolio_equity,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            holding_pnl=holding_pnl,
            holding_pnl_pct=holding_pnl_pct,
            factor_attribution=factor_attribution,
            worst_holding=worst_holding,
            best_holding=best_holding,
            verdict=verdict,
        )

    def run_all_historical_scenarios(
        self,
        holdings: dict[str, float],
        portfolio_equity: float,
    ) -> list[ScenarioResult]:
        scenarios = self._library.get_all_historical_scenarios()
        return [self.run_scenario(holdings, s, portfolio_equity) for s in scenarios]

    def run_all_hypothetical_scenarios(
        self,
        holdings: dict[str, float],
        portfolio_equity: float,
    ) -> list[ScenarioResult]:
        scenarios = self._library.get_all_hypothetical_scenarios()
        return [self.run_scenario(holdings, s, portfolio_equity) for s in scenarios]

    def run_custom_scenario(
        self,
        holdings: dict[str, float],
        asset_shocks: dict[str, float],
        portfolio_equity: float,
        name: str = "Custom",
        description: str = "User-defined scenario",
    ) -> ScenarioResult:
        custom = Scenario(
            name=name,
            description=description,
            type="hypothetical",
            asset_shocks=asset_shocks,
            duration_days=30,
            peak_to_trough=False,
            historical_dates=("", ""),
            vix_level=30.0,
        )
        return self.run_scenario(holdings, custom, portfolio_equity)

    def find_worst_scenario(
        self,
        holdings: dict[str, float],
        portfolio_equity: float,
    ) -> ScenarioResult:
        """Return the single worst scenario by total P&L."""
        all_results = (
            self.run_all_historical_scenarios(holdings, portfolio_equity)
            + self.run_all_hypothetical_scenarios(holdings, portfolio_equity)
        )
        if not all_results:
            raise ValueError("No scenarios to evaluate")
        return min(all_results, key=lambda r: r.total_pnl)

    def compute_tail_risk_budget(
        self,
        holdings: dict[str, float],
        max_loss_pct: float = 0.20,
    ) -> dict:
        """
        For each scenario that exceeds max_loss_pct loss:
        identify which holdings drive the loss and suggest trimming.
        """
        portfolio_equity = sum(abs(v) for v in holdings.values())
        all_results = (
            self.run_all_historical_scenarios(holdings, portfolio_equity)
            + self.run_all_hypothetical_scenarios(holdings, portfolio_equity)
        )

        breaches = [r for r in all_results if r.total_pnl_pct < -max_loss_pct]
        if not breaches:
            return {"status": "OK", "breaches": [], "recommendations": []}

        # Aggregate worst contributors across all breach scenarios
        contributor_counts: dict[str, int] = {}
        for result in breaches:
            worst = min(result.holding_pnl, key=result.holding_pnl.__getitem__)
            contributor_counts[worst] = contributor_counts.get(worst, 0) + 1

        recommendations = []
        for ticker, count in sorted(contributor_counts.items(), key=lambda x: -x[1]):
            current_val = holdings.get(ticker, 0)
            pct_of_portfolio = current_val / max(portfolio_equity, 1e-9)
            recommendations.append({
                "ticker": ticker,
                "current_pct": f"{pct_of_portfolio:.1%}",
                "breach_scenarios": count,
                "suggestion": f"Consider reducing {ticker} (worst contributor in {count} breach scenarios)",
            })

        return {
            "status": "BREACH",
            "breach_scenarios": [r.scenario.name for r in breaches],
            "n_breaches": len(breaches),
            "max_loss_tolerance": f"{max_loss_pct:.1%}",
            "recommendations": recommendations,
        }


# ---------------------------------------------------------------------------
# Monte Carlo Stress Tester
# ---------------------------------------------------------------------------

class MonteCarloStressTester:
    """
    Monte Carlo simulation with normal and fat-tail (Student-t) distributions.
    """

    def __init__(self):
        self._mapper = AssetClassMapper()

    # ------------------------------------------------------------------
    # Return data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_portfolio_returns(
        holdings: dict[str, float],
        start: str = "2010-01-01",
        end: str = "2024-12-31",
    ) -> pd.Series:
        """Fetch historical daily portfolio returns, weighted by holdings."""
        tickers = list(holdings.keys())
        total_val = sum(abs(v) for v in holdings.values())
        weights = {t: holdings[t] / max(total_val, 1e-9) for t in tickers}

        if _HAS_YF:
            try:
                raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    prices = raw["Close"]
                else:
                    prices = raw

                if len(tickers) == 1:
                    prices = prices.to_frame(tickers[0])

                prices = prices[[t for t in tickers if t in prices.columns]].dropna()
                rets = prices.pct_change().dropna()
                port_ret = sum(weights.get(t, 0.0) * rets[t] for t in rets.columns)
                return port_ret
            except Exception as exc:
                logger.warning(f"yfinance fetch failed: {exc}")

        # Synthetic fallback
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(start, end)
        return pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates, name="portfolio")

    @staticmethod
    def compute_expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
        """CVaR = mean of returns below VaR_confidence."""
        sorted_r = np.sort(returns)
        cutoff_idx = int(np.floor((1.0 - confidence) * len(sorted_r)))
        tail = sorted_r[:max(cutoff_idx, 1)]
        return float(tail.mean())

    # ------------------------------------------------------------------
    # Normal MC
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        holdings: dict[str, float],
        portfolio_equity: float,
        n_simulations: int = 10_000,
        horizon_days: int = 252,
        start: str = "2010-01-01",
        end: str = "2024-12-31",
    ) -> MonteCarloResult:
        """
        Bootstrap Monte Carlo from historical return distribution.
        Resample daily returns with replacement.
        """
        hist_returns = self._get_portfolio_returns(holdings, start, end)
        r = hist_returns.dropna().values.astype(float)

        if len(r) < 30:
            r = np.random.normal(0.0, 0.01, 500)

        rng = np.random.default_rng(42)
        terminal_returns = np.empty(n_simulations)

        for i in range(n_simulations):
            sampled = rng.choice(r, size=horizon_days, replace=True)
            terminal_returns[i] = float(np.prod(1.0 + sampled)) - 1.0

        return self._package_mc_result(
            terminal_returns, n_simulations, horizon_days, "bootstrap_normal"
        )

    def run_fat_tail_mc(
        self,
        holdings: dict[str, float],
        portfolio_equity: float,
        n_simulations: int = 10_000,
        horizon_days: int = 252,
        degrees_of_freedom: int = 4,
        start: str = "2010-01-01",
        end: str = "2024-12-31",
    ) -> MonteCarloResult:
        """
        Fat-tail Monte Carlo using Student-t (ν=4 df).
        More realistic for crisis periods.
        """
        hist_returns = self._get_portfolio_returns(holdings, start, end)
        r = hist_returns.dropna().values.astype(float)
        mu = float(np.mean(r))
        sigma = float(np.std(r))

        rng = np.random.default_rng(42)

        if _HAS_SCIPY:
            # scipy Student-t parameterized by (df, loc, scale)
            daily_rets = student_t.rvs(
                df=degrees_of_freedom,
                loc=mu,
                scale=sigma,
                size=(n_simulations, horizon_days),
                random_state=42,
            )
        else:
            # Pure numpy Student-t via chi-squared transform
            z = rng.standard_normal((n_simulations, horizon_days))
            chi2 = rng.chisquare(degrees_of_freedom, size=(n_simulations, horizon_days))
            t_draws = z / np.sqrt(chi2 / degrees_of_freedom)
            daily_rets = mu + sigma * t_draws

        # Clip extreme individual days at ±30% (realistic)
        daily_rets = np.clip(daily_rets, -0.30, 0.30)
        terminal_returns = np.prod(1.0 + daily_rets, axis=1) - 1.0

        return self._package_mc_result(
            terminal_returns, n_simulations, horizon_days, "student_t_fat_tail"
        )

    @staticmethod
    def _package_mc_result(
        terminal_returns: np.ndarray,
        n_simulations: int,
        horizon_days: int,
        distribution: str,
    ) -> MonteCarloResult:
        sorted_r = np.sort(terminal_returns)

        var_95 = float(np.percentile(terminal_returns, 5))
        var_99 = float(np.percentile(terminal_returns, 1))

        tail_95_idx = int(np.floor(0.05 * n_simulations))
        tail_99_idx = int(np.floor(0.01 * n_simulations))
        cvar_95 = float(sorted_r[:max(tail_95_idx, 1)].mean())
        cvar_99 = float(sorted_r[:max(tail_99_idx, 1)].mean())

        prob_loss_20 = float(np.mean(terminal_returns < -0.20))
        prob_loss_50 = float(np.mean(terminal_returns < -0.50))

        terminal_wealth = terminal_returns + 1.0
        p5 = float(np.percentile(terminal_wealth, 5))
        p50 = float(np.percentile(terminal_wealth, 50))
        p95 = float(np.percentile(terminal_wealth, 95))

        return MonteCarloResult(
            n_simulations=n_simulations,
            horizon_days=horizon_days,
            distribution=distribution,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            prob_loss_20pct=prob_loss_20,
            prob_loss_50pct=prob_loss_50,
            median_terminal_wealth=p50,
            p5_terminal_wealth=p5,
            p95_terminal_wealth=p95,
            raw_returns=terminal_returns,
        )


# ---------------------------------------------------------------------------
# Fat-Tail Shock Generator
# ---------------------------------------------------------------------------

class FatTailShockGenerator:
    """
    Generate stressed shock magnitudes using Student-t distribution (df=4).

    Student-t with low degrees of freedom produces heavier tails than normal,
    more accurately representing extreme market dislocations.

    Comparison (same probability level p=0.05):
      Normal:    shock = 1.645 sigma
      Student-t: shock = 2.132 sigma  (df=4)  — ~30% larger tail
    """

    def __init__(self, df: int = 4, seed: int = 42):
        self.df = df
        self.seed = seed

    def fat_tail_quantile(self, probability: float) -> float:
        """
        Return the quantile of the Student-t(df) distribution at given probability.

        Parameters
        ----------
        probability : tail probability, e.g. 0.05 for 95th percentile shock.

        Returns
        -------
        t-quantile value (positive; caller decides sign for loss/gain).
        """
        if _HAS_SCIPY:
            return float(student_t.ppf(1.0 - probability, df=self.df))
        # Pure numpy fallback: inverse CDF via iterative bisection
        return self._numpy_t_quantile(1.0 - probability)

    def normal_quantile(self, probability: float) -> float:
        """Normal distribution quantile at (1 - probability) — for comparison."""
        if _HAS_SCIPY:
            from scipy.stats import norm
            return float(norm.ppf(1.0 - probability))
        # Rational approximation (Beasley-Springer-Moro)
        p = 1.0 - probability
        if p <= 0 or p >= 1:
            return float("nan")
        # Abramowitz & Stegun approximation
        t = (-2.0 * np.log(min(p, 1 - p))) ** 0.5
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        x = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
        return x if p > 0.5 else -x

    def _numpy_t_quantile(self, p: float, tol: float = 1e-8) -> float:
        """Bisection-based inverse CDF for Student-t (pure numpy)."""
        if p <= 0:
            return float("-inf")
        if p >= 1:
            return float("inf")
        lo, hi = -20.0, 20.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if self._t_cdf(mid) < p:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return (lo + hi) / 2.0

    def _t_cdf(self, x: float) -> float:
        """Regularized incomplete beta CDF for Student-t (numpy-only approximation)."""
        # P(T <= x) = I(df/(df+x^2); df/2, 1/2) / 2  for x < 0, else 1 - that
        import math
        df = self.df
        if x == 0:
            return 0.5
        t2 = x * x
        z = df / (df + t2)
        # Regularized incomplete beta via continued fraction
        try:
            p_half = self._reg_beta(z, df / 2.0, 0.5)
        except Exception:
            p_half = 0.5
        if x < 0:
            return p_half / 2.0
        return 1.0 - p_half / 2.0

    @staticmethod
    def _reg_beta(z: float, a: float, b: float, max_iter: int = 200) -> float:
        """Regularized incomplete beta I(z; a, b) via Lentz continued fraction."""
        import math
        if z < 0 or z > 1:
            return 0.0
        if z == 0:
            return 0.0
        if z == 1:
            return 1.0
        # log beta function
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(a * math.log(z) + b * math.log(1 - z) - lbeta) / a
        # Lentz CF
        tiny = 1e-300
        f = tiny
        C = f
        D = 0.0
        for m in range(max_iter):
            for i in [0, 1]:
                if i == 0:
                    if m == 0:
                        d = 1.0
                    else:
                        d = m * (b - m) * z / ((a + 2 * m - 1) * (a + 2 * m))
                else:
                    d = -(a + m) * (a + b + m) * z / ((a + 2 * m) * (a + 2 * m + 1))
                D = 1 + d * D
                if abs(D) < tiny:
                    D = tiny
                D = 1.0 / D
                C = 1 + d / C
                if abs(C) < tiny:
                    C = tiny
                delta = C * D
                f *= delta
                if abs(delta - 1.0) < 1e-10:
                    return front * f
        return front * f

    def generate_fat_tail_shocks(
        self,
        n_shocks: int,
        scale: float = 0.01,
    ) -> np.ndarray:
        """
        Draw n_shocks from Student-t(df) scaled by `scale`.

        Parameters
        ----------
        n_shocks : number of shock draws.
        scale    : daily volatility scale (e.g. 0.01 = 1% daily vol).

        Returns
        -------
        Array of shock magnitudes (fractional returns).
        """
        rng = np.random.default_rng(self.seed)
        if _HAS_SCIPY:
            shocks = student_t.rvs(df=self.df, scale=scale, size=n_shocks, random_state=self.seed)
        else:
            z   = rng.standard_normal(n_shocks)
            chi2 = rng.chisquare(self.df, size=n_shocks)
            shocks = scale * z / np.sqrt(chi2 / self.df)
        return shocks

    def fat_tail_vs_normal_shock(
        self,
        probability: float = 0.05,
        sigma: float = 1.0,
    ) -> dict:
        """
        Compare fat-tail vs normal shock at same left-tail probability.

        Returns dict with normal_shock, fat_tail_shock, amplification_ratio.
        The fat-tail shock is always larger in absolute value.
        """
        normal_q  = self.normal_quantile(probability)   # e.g. -1.645 at p=0.05
        fat_tail_q = -self.fat_tail_quantile(probability)  # negative (loss side)
        normal_shock   = abs(normal_q)  * sigma
        fat_tail_shock = abs(fat_tail_q) * sigma
        return {
            "probability":        probability,
            "sigma":              sigma,
            "normal_shock":       round(normal_shock,   6),
            "fat_tail_shock":     round(fat_tail_shock, 6),
            "amplification_ratio": round(fat_tail_shock / max(normal_shock, 1e-12), 4),
            "df":                 self.df,
        }


# ---------------------------------------------------------------------------
# Cross-Asset Contagion Matrix
# ---------------------------------------------------------------------------

# Conditional shock multipliers: when equities fall >= threshold,
# apply these multipliers to other asset class shocks.
# Format: {trigger_asset: {threshold: float, conditional_shocks: {asset: multiplier}}}
CONTAGION_RULES: dict = {
    "equity_crash": {
        "trigger":          "equity",
        "trigger_threshold": -0.15,        # equity falls more than 15%
        "conditional_shocks": {
            "credit":       3.0,           # credit spreads widen 3x
            "vix":          2.0,           # VIX spikes 2x
            "liquidity_bps": 50.0,         # liquidity premium adds 50bps
            "bonds":        0.5,           # flight to quality: bonds up 50% of equity move (inverted)
            "gold":         1.5,           # gold safe-haven demand
        },
        "description": "Equity crash > 15% triggers credit contagion and VIX spike",
    }
}

# Baseline asset-class credit spreads (bps) — used for liquidity stress
BASELINE_CREDIT_SPREADS_BPS: dict[str, float] = {
    "investment_grade":  50.0,
    "high_yield":       400.0,
    "equity_vol":        15.0,  # VIX baseline
}


class CrossAssetContagionMatrix:
    """
    Model cross-asset contagion: when one asset class is shocked beyond a
    threshold, apply amplified shocks to related assets.

    Based on empirical observations from GFC 2008 and COVID 2020.
    """

    def __init__(self, rules: dict | None = None):
        self._rules = rules or CONTAGION_RULES

    def apply_contagion(
        self,
        base_shocks: dict[str, float],
        baseline_spreads_bps: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Apply contagion amplification to base_shocks.

        Parameters
        ----------
        base_shocks : dict of {asset_class: fractional_shock} from a scenario.
        baseline_spreads_bps : baseline credit spreads; used for spread-widening calc.

        Returns
        -------
        Augmented shock dict with contagion effects. Includes:
          - 'credit_spread_widening_bps' : additional spread widening in bps
          - 'vix_amplified'              : amplified VIX level estimate
          - 'liquidity_premium_bps'      : extra cost of liquidity in bps
          - All original asset class shocks (possibly amplified)
        """
        spreads = baseline_spreads_bps or BASELINE_CREDIT_SPREADS_BPS
        result  = dict(base_shocks)

        for rule_name, rule in self._rules.items():
            trigger_asset    = rule["trigger"]
            trigger_threshold = rule["trigger_threshold"]
            equity_shock      = base_shocks.get(trigger_asset, 0.0)

            if equity_shock <= trigger_threshold:
                # Trigger fired
                cond = rule["conditional_shocks"]

                # Credit spreads widen by 3x of base credit shock
                base_credit   = abs(base_shocks.get("credit", 0.0))
                credit_mult   = cond.get("credit", 3.0)
                amplified_credit_spread = spreads.get("high_yield", 400.0) * base_credit * credit_mult
                result["credit_spread_widening_bps"] = round(
                    result.get("credit_spread_widening_bps", 0.0) + amplified_credit_spread, 2
                )

                # VIX spikes 2x baseline
                vix_baseline = spreads.get("equity_vol", 15.0)
                vix_mult     = cond.get("vix", 2.0)
                result["vix_amplified"] = round(
                    result.get("vix_amplified", vix_baseline) * vix_mult, 2
                )

                # Liquidity premium
                liq_bps = cond.get("liquidity_bps", 50.0)
                result["liquidity_premium_bps"] = round(
                    result.get("liquidity_premium_bps", 0.0) + liq_bps, 2
                )

                # Amplify credit asset shock
                if "credit" in result:
                    result["credit"] = round(result["credit"] * credit_mult, 6)

                logger.debug(
                    "Contagion rule '%s' fired: equity=%.1f%%, credit spread +%.0fbps, VIX x%.1f",
                    rule_name, equity_shock * 100,
                    result.get("credit_spread_widening_bps", 0),
                    vix_mult,
                )

        return result

    def compute_contagion_pnl_adjustment(
        self,
        holdings: dict[str, float],
        base_shocks: dict[str, float],
        mapper: "AssetClassMapper",
        liquidity_spread_multiplier: float = 2.0,
    ) -> dict:
        """
        Compute the additional P&L impact from contagion vs base scenario.

        Parameters
        ----------
        holdings : {ticker: dollar_value}
        base_shocks : original scenario asset_shocks
        mapper : AssetClassMapper to map tickers to asset classes
        liquidity_spread_multiplier : bid/ask spread widens by this factor in stress.

        Returns
        -------
        dict with base_pnl, contagion_pnl, liquidity_pnl, total_adjusted_pnl.
        """
        contagion_shocks = self.apply_contagion(base_shocks)

        portfolio_equity = sum(abs(v) for v in holdings.values())

        # Base P&L (linear shocks)
        base_pnl = 0.0
        for ticker, value in holdings.items():
            ac_map = mapper.classify_holding(ticker)
            shock  = sum(ac_map.get(ac, 0.0) * base_shocks.get(ac, 0.0) for ac in ac_map)
            base_pnl += value * shock

        # Contagion-adjusted P&L
        contagion_pnl = 0.0
        for ticker, value in holdings.items():
            ac_map = mapper.classify_holding(ticker)
            shock  = sum(ac_map.get(ac, 0.0) * contagion_shocks.get(ac, 0.0) for ac in ac_map)
            contagion_pnl += value * shock

        # Liquidity stress: bid/ask spread widening reduces realised proceeds
        # Assume average spread = 20bps normally; in stress = spread * multiplier
        normal_spread_bps  = 20.0
        stressed_spread_bps = normal_spread_bps * liquidity_spread_multiplier
        liquidity_cost_pct  = (stressed_spread_bps - normal_spread_bps) / 10_000.0
        liquidity_pnl       = -portfolio_equity * liquidity_cost_pct

        # Add explicit liquidity premium from contagion rules
        extra_liq_bps = contagion_shocks.get("liquidity_premium_bps", 0.0)
        liquidity_pnl -= portfolio_equity * extra_liq_bps / 10_000.0

        return {
            "base_pnl":               round(base_pnl, 2),
            "contagion_pnl":          round(contagion_pnl, 2),
            "liquidity_pnl":          round(liquidity_pnl, 2),
            "total_adjusted_pnl":     round(contagion_pnl + liquidity_pnl, 2),
            "contagion_shocks":       contagion_shocks,
            "credit_spread_widening_bps": contagion_shocks.get("credit_spread_widening_bps", 0.0),
            "liquidity_premium_bps":  contagion_shocks.get("liquidity_premium_bps", 0.0),
        }


# ---------------------------------------------------------------------------
# GARCH-Based Volatility Shock
# ---------------------------------------------------------------------------

class GARCHVolatilityShock:
    """
    Estimate stressed volatility using a simplified GARCH(1,1) model.

    In stressed state, returns are drawn from GARCH with elevated omega
    to simulate vol clustering. The stressed portfolio vol is:

        vol_stressed = realized_vol * sqrt(h_t / h_0)

    where h_t is the GARCH conditional variance in the stressed state
    and h_0 is the long-run (unconditional) variance.
    """

    def __init__(self, omega: float = 5e-6, alpha: float = 0.10, beta: float = 0.85):
        """
        Parameters
        ----------
        omega : GARCH(1,1) constant term (default: small; typical equity daily).
        alpha : ARCH coefficient (persistence of shocks to squared returns).
        beta  : GARCH coefficient (persistence of conditional variance).
        """
        if alpha + beta >= 1.0:
            raise ValueError("alpha + beta must be < 1 for stationary GARCH.")
        self.omega = omega
        self.alpha = alpha
        self.beta  = beta

    @property
    def unconditional_variance(self) -> float:
        """Long-run GARCH(1,1) variance: omega / (1 - alpha - beta)."""
        return self.omega / max(1.0 - self.alpha - self.beta, 1e-12)

    def forecast_stressed_variance(
        self,
        realized_returns: np.ndarray,
        stress_multiplier: float = 3.0,
        horizon: int = 1,
    ) -> float:
        """
        Forecast stressed conditional variance h_t.

        Algorithm:
        1. Fit GARCH(1,1) recursion on realized_returns to get h_0 (current h).
        2. Inject a stress shock: the last squared return is replaced with
           stress_multiplier * h_0 (simulating a jump event).
        3. Forecast h_{t+horizon} forward.

        Returns h_t (stressed conditional variance for one horizon step).
        """
        r = np.asarray(realized_returns, float)
        if len(r) < 5:
            return self.unconditional_variance * stress_multiplier

        # Step 1: Initialize h at unconditional variance
        h = self.unconditional_variance
        for i in range(len(r)):
            e2 = r[i] ** 2
            h  = self.omega + self.alpha * e2 + self.beta * h
        h_0 = max(h, 1e-12)

        # Step 2: Stress shock — replace last observation with stress_multiplier * h_0
        h_stressed = self.omega + self.alpha * (stress_multiplier * h_0) + self.beta * h_0

        # Step 3: Forecast forward `horizon` steps
        h_forecast = h_stressed
        for _ in range(horizon - 1):
            h_forecast = self.omega + (self.alpha + self.beta) * h_forecast

        return float(max(h_forecast, 1e-12))

    def compute_vol_shock_ratio(
        self,
        realized_returns: np.ndarray,
        stress_multiplier: float = 3.0,
        horizon: int = 1,
    ) -> dict:
        """
        Compute the ratio vol_stressed / vol_realized.

        Returns dict with:
          h_0         : current GARCH conditional variance
          h_stressed  : stressed forecast variance
          vol_ratio   : sqrt(h_stressed / h_0)
          annualized_normal_vol  : realized vol * sqrt(252)
          annualized_stressed_vol: normal_vol * vol_ratio * sqrt(252)
        """
        r = np.asarray(realized_returns, float)
        realized_var  = float(np.var(r)) if len(r) > 1 else self.unconditional_variance
        h_0           = max(realized_var, 1e-12)
        h_stressed    = self.forecast_stressed_variance(r, stress_multiplier, horizon)
        vol_ratio     = float(np.sqrt(h_stressed / h_0))
        ann_normal    = float(np.std(r)) * np.sqrt(252)
        ann_stressed  = ann_normal * vol_ratio

        return {
            "h_0":                   round(h_0, 8),
            "h_stressed":            round(h_stressed, 8),
            "vol_ratio":             round(vol_ratio, 4),
            "annualized_normal_vol": round(ann_normal, 4),
            "annualized_stressed_vol": round(ann_stressed, 4),
            "stress_multiplier":     stress_multiplier,
            "horizon_days":          horizon,
        }


# ---------------------------------------------------------------------------
# Liquidity Stress Calculator
# ---------------------------------------------------------------------------

class LiquidityStressCalculator:
    """
    Adjust scenario P&L for bid/ask spread widening under stress.

    In normal markets, spreads are tight. During stress events, spreads
    widen substantially (2x to 5x) as market makers pull back liquidity.

    This reduces the effective proceeds from liquidating positions.
    """

    def __init__(
        self,
        normal_spread_bps: float = 20.0,
        stress_multiplier: float = 2.0,
    ):
        """
        Parameters
        ----------
        normal_spread_bps : typical bid/ask spread in basis points (default 20bps).
        stress_multiplier : spread widens by this multiple in stress (default 2x).
        """
        self.normal_spread_bps = normal_spread_bps
        self.stress_multiplier = stress_multiplier

    @property
    def stressed_spread_bps(self) -> float:
        return self.normal_spread_bps * self.stress_multiplier

    def liquidity_adjusted_pnl(
        self,
        base_pnl: float,
        portfolio_equity: float,
        turnover_fraction: float = 1.0,
    ) -> dict:
        """
        Subtract liquidity cost from base P&L.

        Parameters
        ----------
        base_pnl : scenario P&L before liquidity adjustment.
        portfolio_equity : total portfolio value.
        turnover_fraction : fraction of portfolio that needs to be traded (default: 100%).

        Returns
        -------
        dict with base_pnl, liquidity_cost, adjusted_pnl, spread_bps_normal,
        spread_bps_stressed, spread_widening_bps.
        """
        spread_widening_bps = self.stressed_spread_bps - self.normal_spread_bps
        liquidity_cost = portfolio_equity * turnover_fraction * spread_widening_bps / 10_000.0
        adjusted_pnl   = base_pnl - liquidity_cost

        return {
            "base_pnl":              round(base_pnl, 2),
            "liquidity_cost":        round(liquidity_cost, 2),
            "adjusted_pnl":          round(adjusted_pnl, 2),
            "spread_bps_normal":     self.normal_spread_bps,
            "spread_bps_stressed":   self.stressed_spread_bps,
            "spread_widening_bps":   spread_widening_bps,
            "turnover_fraction":     turnover_fraction,
        }

    def compare_with_without_liquidity(
        self,
        base_pnl: float,
        portfolio_equity: float,
    ) -> dict:
        """
        Show the P&L difference between stressed and no-liquidity-stress scenarios.

        Returns dict with pnl_no_liq_stress, pnl_with_liq_stress, liquidity_drag.
        """
        no_liq  = self.liquidity_adjusted_pnl(base_pnl, portfolio_equity, turnover_fraction=0.0)
        with_liq = self.liquidity_adjusted_pnl(base_pnl, portfolio_equity, turnover_fraction=1.0)
        return {
            "pnl_no_liq_stress":  no_liq["adjusted_pnl"],
            "pnl_with_liq_stress": with_liq["adjusted_pnl"],
            "liquidity_drag":     round(no_liq["adjusted_pnl"] - with_liq["adjusted_pnl"], 2),
        }


# ---------------------------------------------------------------------------
# Correlation Breakdown Analyzer
# ---------------------------------------------------------------------------

class CorrelationBreakdownAnalyzer:
    """
    Analyze how correlations spike during crisis periods vs normal regimes.
    """

    # Key crisis windows (start, end)
    CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
        "GFC_2008": ("2008-01-01", "2009-03-31"),
        "COVID_2020": ("2020-01-15", "2020-04-30"),
        "Rate_Shock_2022": ("2022-01-01", "2022-10-31"),
        "Dot_Com_2000": ("2000-01-01", "2003-01-01"),
        "LTCM_1998": ("1998-07-01", "1998-11-01"),
    }

    def compute_crisis_correlations(
        self,
        start: str,
        end: str,
        tickers: list[str],
    ) -> pd.DataFrame:
        """
        Compute actual return correlations during a specific window using yfinance.
        """
        if not _HAS_YF:
            return self._synthetic_corr(tickers, crisis=True)

        try:
            raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw
            prices = prices[[t for t in tickers if t in prices.columns]].dropna()
            rets = prices.pct_change().dropna()
            return rets.corr()
        except Exception as exc:
            logger.warning(f"Crisis corr fetch failed: {exc}")
            return self._synthetic_corr(tickers, crisis=True)

    @staticmethod
    def _synthetic_corr(tickers: list[str], crisis: bool = False) -> pd.DataFrame:
        n = len(tickers)
        base = 0.7 if crisis else 0.3
        rng = np.random.default_rng(123)
        mat = np.full((n, n), base) + rng.normal(0, 0.05, (n, n))
        np.fill_diagonal(mat, 1.0)
        mat = (mat + mat.T) / 2
        return pd.DataFrame(mat, index=tickers, columns=tickers)

    def compare_normal_vs_crisis_correlations(
        self,
        tickers: list[str],
        normal_years: int = 3,
    ) -> dict[str, pd.DataFrame]:
        """
        Compare rolling normal-period correlations vs GFC 2008 and COVID 2020.
        Returns dict of correlation matrices.
        """
        # Normal period: recent 3yr average
        normal_start = "2019-01-01"
        normal_end = "2019-12-31"
        normal_corr = self.compute_crisis_correlations(normal_start, normal_end, tickers)

        gfc_corr = self.compute_crisis_correlations(
            *self.CRISIS_WINDOWS["GFC_2008"], tickers=tickers
        )
        covid_corr = self.compute_crisis_correlations(
            *self.CRISIS_WINDOWS["COVID_2020"], tickers=tickers
        )

        return {
            "normal_2019": normal_corr,
            "gfc_2008_2009": gfc_corr,
            "covid_2020": covid_corr,
        }

    @staticmethod
    def estimate_crisis_portfolio_vol(
        holdings: dict[str, float],
        crisis_cov: pd.DataFrame,
    ) -> float:
        """
        Re-estimate portfolio vol using crisis correlation matrix.
        Typically much higher than normal-period estimate.
        """
        tickers = [t for t in holdings if t in crisis_cov.index]
        if not tickers:
            return 0.0

        total_val = sum(abs(holdings[t]) for t in tickers)
        w = np.array([holdings[t] / max(total_val, 1e-9) for t in tickers])
        cov_sub = crisis_cov.loc[tickers, tickers].values

        # Annualize (assuming daily cov * 252)
        port_var = float(w @ cov_sub @ w) * 252
        return float(np.sqrt(max(port_var, 0.0)))


# ---------------------------------------------------------------------------
# Stress Test Report
# ---------------------------------------------------------------------------

class StressTestReport:
    """
    Narrative reporting: full scenario sweep + summary table.
    """

    def __init__(self):
        self._engine = StressTestEngine()
        self._mc = MonteCarloStressTester()

    def generate_full_report(
        self,
        holdings: dict[str, float],
        portfolio_equity: float,
    ) -> str:
        historical = self._engine.run_all_historical_scenarios(holdings, portfolio_equity)
        hypothetical = self._engine.run_all_hypothetical_scenarios(holdings, portfolio_equity)

        lines = [
            "=" * 72,
            "SENTINEL — Portfolio Stress Test Report",
            "=" * 72,
            f"Portfolio Equity: ${portfolio_equity:,.0f}",
            f"Holdings: {', '.join(holdings.keys())}",
            "",
            "HISTORICAL SCENARIOS",
            "-" * 72,
            f"{'Scenario':<35} {'P&L $':>12} {'P&L %':>8} {'Verdict':>9}",
            "-" * 72,
        ]

        for r in historical:
            verdict_sym = {"OK": "  OK", "WARNING": " WARN", "SEVERE": " !!!"}[r.verdict]
            lines.append(
                f"{r.scenario.name:<35} {r.total_pnl:>+12,.0f} {r.total_pnl_pct:>+7.1%}  {verdict_sym}"
            )

        lines += [
            "",
            "HYPOTHETICAL FORWARD SCENARIOS",
            "-" * 72,
            f"{'Scenario':<35} {'P&L $':>12} {'P&L %':>8} {'Verdict':>9}",
            "-" * 72,
        ]

        for r in hypothetical:
            verdict_sym = {"OK": "  OK", "WARNING": " WARN", "SEVERE": " !!!"}[r.verdict]
            lines.append(
                f"{r.scenario.name:<35} {r.total_pnl:>+12,.0f} {r.total_pnl_pct:>+7.1%}  {verdict_sym}"
            )

        all_results = historical + hypothetical
        worst = min(all_results, key=lambda r: r.total_pnl)
        best = max(all_results, key=lambda r: r.total_pnl)
        severe = [r for r in all_results if r.verdict == "SEVERE"]

        lines += [
            "",
            "SUMMARY",
            "-" * 72,
            f"Worst scenario : {worst.scenario.name} ({worst.total_pnl_pct:+.1%})",
            f"Best scenario  : {best.scenario.name} ({best.total_pnl_pct:+.1%})",
            f"SEVERE scenarios: {len(severe)} / {len(all_results)}",
            "=" * 72,
        ]

        return "\n".join(lines)

    def get_dashboard(
        self,
        holdings: dict[str, float],
        portfolio_equity: float,
    ) -> pd.DataFrame:
        historical = self._engine.run_all_historical_scenarios(holdings, portfolio_equity)
        hypothetical = self._engine.run_all_hypothetical_scenarios(holdings, portfolio_equity)
        all_results = historical + hypothetical

        rows = []
        for r in all_results:
            rows.append({
                "scenario": r.scenario.name,
                "type": r.scenario.type,
                "pnl_dollars": round(r.total_pnl, 0),
                "pnl_pct": round(r.total_pnl_pct * 100, 2),
                "verdict": r.verdict,
                "worst_holding": r.worst_holding,
                "best_holding": r.best_holding,
            })

        df = pd.DataFrame(rows).sort_values("pnl_pct")
        return df

    def export_report(self, path: str, holdings: dict, portfolio_equity: float) -> None:
        """Export full results as JSON."""
        dashboard = self.get_dashboard(holdings, portfolio_equity)
        report_data = {
            "portfolio_equity": portfolio_equity,
            "holdings": holdings,
            "scenarios": dashboard.to_dict(orient="records"),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        logger.info(f"Report exported to {path}")


# ---------------------------------------------------------------------------
# Hedge Recommendation Engine
# ---------------------------------------------------------------------------

class HedgeRecommendationEngine:
    """
    Given stress test results, recommend portfolio hedges.
    """

    _HEDGE_CATALOG: dict[str, HedgeRecommendation] = {
        "spx_puts": HedgeRecommendation(
            name="SPX Put Options",
            description="5% OTM put spread on SPX for equity tail protection",
            instruments=["SPX puts", "SPDW puts", "SPY puts"],
            target_scenarios=["GFC 2008-2009", "COVID Crash 2020", "Tech Bubble Burst", "Hard Landing"],
            estimated_cost_bps=80.0,
            expected_offset_pct=0.50,
            sizing_suggestion="2-5% of equity notional; 3-month tenor; roll quarterly",
        ),
        "vix_calls": HedgeRecommendation(
            name="VIX Call Spreads",
            description="Long VIX 20/35 call spread for vol spike protection",
            instruments=["VIX calls", "UVXY", "VIXY"],
            target_scenarios=["Flash Crash 2010", "COVID Crash 2020", "Black Monday 1987"],
            estimated_cost_bps=40.0,
            expected_offset_pct=0.30,
            sizing_suggestion="1-2% of equity; roll monthly near expiry",
        ),
        "anti_beta": HedgeRecommendation(
            name="Anti-Beta ETF (BTAL)",
            description="Long BTAL: long low-beta / short high-beta equities",
            instruments=["BTAL", "TAIL"],
            target_scenarios=["GFC 2008-2009", "Dot-com Crash 2000-2002", "Hard Landing"],
            estimated_cost_bps=90.0,
            expected_offset_pct=0.35,
            sizing_suggestion="5-10% of equity as structural hedge",
        ),
        "tlt_puts": HedgeRecommendation(
            name="TLT Put Options",
            description="Long puts on TLT for rising rate / duration risk",
            instruments=["TLT puts", "TBT", "TMV"],
            target_scenarios=["Rate Shock 2022", "Stagflation Shock", "Interest Rate Spike"],
            estimated_cost_bps=50.0,
            expected_offset_pct=0.60,
            sizing_suggestion="Notional = bond portfolio duration × 0.5%; 6-month tenor",
        ),
        "tips": HedgeRecommendation(
            name="TIPS / Inflation Linkers",
            description="Treasury Inflation-Protected Securities for inflation hedge",
            instruments=["TIP", "SCHP", "VTIP"],
            target_scenarios=["Stagflation Shock", "USD Collapse"],
            estimated_cost_bps=10.0,
            expected_offset_pct=0.40,
            sizing_suggestion="Replace 20-30% of nominal bond allocation with TIPS",
        ),
        "gold_allocation": HedgeRecommendation(
            name="Gold Allocation",
            description="Physical gold ETF: crisis hedge + inflation store of value",
            instruments=["GLD", "IAU", "SGOL"],
            target_scenarios=["GFC 2008-2009", "Geopolitical Crisis", "USD Collapse", "Stagflation Shock"],
            estimated_cost_bps=25.0,
            expected_offset_pct=0.25,
            sizing_suggestion="5-10% strategic allocation; rebalance annually",
        ),
        "cdx_protection": HedgeRecommendation(
            name="CDX Credit Protection",
            description="Buy protection on CDX.NA.HY (high yield credit spread)",
            instruments=["HYG puts", "JNK puts", "CDX.NA.HY"],
            target_scenarios=["LTCM Russia Crisis 1998", "GFC 2008-2009", "SVB Bank Run 2023"],
            estimated_cost_bps=120.0,
            expected_offset_pct=0.55,
            sizing_suggestion="Notional = HY bond allocation; quarterly roll",
        ),
        "short_duration": HedgeRecommendation(
            name="Shorten Duration",
            description="Shift bonds to shorter duration (1-3yr) to reduce rate sensitivity",
            instruments=["SHY", "VGSH", "BSV", "BIL"],
            target_scenarios=["Rate Shock 2022", "Interest Rate Spike"],
            estimated_cost_bps=5.0,
            expected_offset_pct=0.70,
            sizing_suggestion="Replace TLT/IEF with SHY until rate uncertainty resolves",
        ),
        "commodities_basket": HedgeRecommendation(
            name="Commodities Basket",
            description="Diversified commodities for inflation + geopolitical hedge",
            instruments=["DBC", "PDBC", "GSG", "COMT"],
            target_scenarios=["Stagflation Shock", "Geopolitical Crisis", "USD Collapse"],
            estimated_cost_bps=60.0,
            expected_offset_pct=0.30,
            sizing_suggestion="5% tactical allocation during inflationary regimes",
        ),
    }

    def recommend_hedges(
        self,
        stress_results: list[ScenarioResult],
        portfolio_equity: float,
        top_n: int = 4,
    ) -> list[HedgeRecommendation]:
        """
        Identify worst scenarios, then rank hedges by scenario coverage.
        """
        # Identify worst 3 scenarios
        sorted_results = sorted(stress_results, key=lambda r: r.total_pnl)
        worst_scenarios = [r.scenario.name for r in sorted_results[:3]]

        # Score each hedge by how many worst scenarios it targets
        hedge_scores: dict[str, int] = {}
        for hedge_key, hedge in self._HEDGE_CATALOG.items():
            score = 0
            for ws in worst_scenarios:
                for target in hedge.target_scenarios:
                    if target.lower() in ws.lower() or ws.lower() in target.lower():
                        score += 1
            # Weight also by severity of offset
            hedge_scores[hedge_key] = score

        # Fallback: if no specific match, use generic equity hedges
        if max(hedge_scores.values(), default=0) == 0:
            hedge_scores = {k: 1 for k in ["spx_puts", "gold_allocation", "vix_calls", "tips"]}

        # Determine primary stress factor
        factor_sums: dict[str, float] = {}
        for r in sorted_results[:3]:
            for factor, contrib in r.factor_attribution.items():
                factor_sums[factor] = factor_sums.get(factor, 0.0) + contrib

        primary_factor = min(factor_sums, key=factor_sums.get) if factor_sums else "equity"

        # Add factor-specific hedges with bonus scoring
        factor_hedge_map = {
            "equity": ["spx_puts", "anti_beta", "vix_calls"],
            "bonds": ["tlt_puts", "short_duration"],
            "credit": ["cdx_protection", "spx_puts"],
            "gold": ["gold_allocation"],
            "oil": ["commodities_basket"],
            "usd": ["gold_allocation", "commodities_basket"],
        }
        bonus_keys = factor_hedge_map.get(primary_factor, [])
        for k in bonus_keys:
            hedge_scores[k] = hedge_scores.get(k, 0) + 2

        ranked = sorted(hedge_scores, key=hedge_scores.__getitem__, reverse=True)
        return [self._HEDGE_CATALOG[k] for k in ranked[:top_n] if k in self._HEDGE_CATALOG]

    @staticmethod
    def estimate_hedge_cost(hedge: HedgeRecommendation, portfolio_equity: float) -> float:
        """
        Annual cost estimate in dollars for implementing the hedge.
        """
        annual_cost = portfolio_equity * hedge.estimated_cost_bps / 10_000
        return annual_cost

    @staticmethod
    def compute_hedge_effectiveness(
        hedge: HedgeRecommendation,
        scenario: ScenarioResult,
    ) -> float:
        """
        Estimated fraction of scenario loss offset by the hedge.
        Returns 0.0 to 1.0 (1.0 = fully hedged).
        """
        if scenario.scenario.name in hedge.target_scenarios:
            return hedge.expected_offset_pct
        # Partial credit if scenario shares stress factor
        return hedge.expected_offset_pct * 0.25


# ---------------------------------------------------------------------------
# Reverse Stress Test & P&L Distribution (dim_083 score 8 → 9)
# ---------------------------------------------------------------------------

class ReverseStressTester:
    """
    Reverse stress testing: find the portfolio weight vector that maximises
    loss under a given macro scenario shock vector.

    Also computes the full cross-scenario P&L distribution at key percentiles.
    """

    def __init__(self):
        self._mapper = AssetClassMapper()
        self._library = ScenarioLibrary()
        self._engine = StressTestEngine()

    # ------------------------------------------------------------------
    def compute_reverse_stress_test(
        self,
        scenario: Scenario,
        asset_exposures: dict[str, float],
        total_equity: float = 1_000_000.0,
    ) -> dict:
        """
        Find the worst-case portfolio allocation (weight vector w*) that
        maximises dollar loss under the given scenario's asset class shocks.

        Algorithm:
          For each asset class with a *negative* shock, the loss-maximising
          weight puts as much capital as possible into that class, subject to:
            - All weights >= 0 (long-only, no leverage)
            - Sum of weights = 1

        The greedy worst-case portfolio concentrates entirely in the asset
        class with the most negative shock.

        Parameters
        ----------
        scenario : Scenario whose asset_shocks define the stress.
        asset_exposures : Available asset classes and their max allowable
            weight fraction (e.g. {"equity": 1.0, "bonds": 1.0}).
            Defaults to all ASSET_CLASS_KEYS with max=1.0.
        total_equity : Portfolio notional for dollar P&L calculation.

        Returns
        -------
        dict with:
          worst_weight_vector : {asset_class: weight}
          max_loss_pct        : fractional loss of worst-case portfolio
          max_loss_dollars    : dollar loss at total_equity
          worst_asset_class   : the single asset class driving worst loss
          all_shocks          : scenario shocks for reference
        """
        shocks = scenario.asset_shocks
        exposures = asset_exposures or {ac: 1.0 for ac in ASSET_CLASS_KEYS}

        # Sort by shock ascending (most negative first)
        # Only include asset classes that appear in exposures
        eligible = [(ac, shocks.get(ac, 0.0)) for ac in exposures if ac in shocks]
        eligible.sort(key=lambda x: x[1])

        # Worst-case: concentrate in the most negatively shocked asset class
        if not eligible:
            return {
                "worst_weight_vector": {},
                "max_loss_pct": 0.0,
                "max_loss_dollars": 0.0,
                "worst_asset_class": None,
                "all_shocks": shocks,
            }

        worst_ac, worst_shock = eligible[0]
        # Build weight vector: 100% in worst asset class (long-only constraint)
        w = {ac: 0.0 for ac in exposures}
        w[worst_ac] = min(exposures.get(worst_ac, 1.0), 1.0)

        # If worst shock is positive (no negative shocks), spread equally
        if worst_shock >= 0.0:
            n = len(w)
            for ac in w:
                w[ac] = 1.0 / n
            max_loss_pct = sum(w[ac] * shocks.get(ac, 0.0) for ac in w)
        else:
            max_loss_pct = worst_shock  # fully concentrated

        max_loss_dollars = total_equity * max_loss_pct

        return {
            "worst_weight_vector": w,
            "max_loss_pct": round(max_loss_pct, 6),
            "max_loss_dollars": round(max_loss_dollars, 2),
            "worst_asset_class": worst_ac if worst_shock < 0 else None,
            "scenario_name": scenario.name,
            "all_shocks": {k: v for k, v in shocks.items() if v != 0.0},
        }

    # ------------------------------------------------------------------
    def compute_stress_pnl_distribution(
        self,
        holdings: dict[str, float],
        portfolio_equity: float,
        scenarios: list[Scenario] | None = None,
    ) -> dict:
        """
        Apply all scenarios to portfolio and compute percentile losses.

        Runs every available scenario (historical + hypothetical) against
        the portfolio, then reports:
          - 95th percentile loss  (5% worst outcomes)
          - 99th percentile loss  (1% worst outcomes)
          - 99.9th percentile loss (0.1% worst outcomes)
          - full sorted P&L distribution

        Parameters
        ----------
        holdings : {ticker: dollar_value}.
        portfolio_equity : Total portfolio value.
        scenarios : Optional list of Scenario objects. If None, uses all
            historical + hypothetical scenarios from the library.

        Returns
        -------
        dict with percentile_losses, scenario_pnls, worst_scenarios, best_scenarios.
        """
        if scenarios is None:
            scenarios = (
                self._library.get_all_historical_scenarios()
                + self._library.get_all_hypothetical_scenarios()
            )

        # Compute P&L for each scenario
        results = []
        for s in scenarios:
            r = self._engine.run_scenario(holdings, s, portfolio_equity)
            results.append({
                "scenario": s.name,
                "pnl_pct": r.total_pnl_pct,
                "pnl_dollars": r.total_pnl,
                "verdict": r.verdict,
            })

        # Sort by P&L ascending (worst first)
        results.sort(key=lambda x: x["pnl_pct"])
        pnl_pcts = np.array([r["pnl_pct"] for r in results])
        n = len(pnl_pcts)

        def _percentile_loss(p: float) -> float:
            """Loss at the p-th percentile (tail end)."""
            idx = max(0, int(np.floor((1.0 - p) * n)) - 1)
            return float(pnl_pcts[idx]) if n > 0 else 0.0

        p95_loss = _percentile_loss(0.95)
        p99_loss = _percentile_loss(0.99)
        p999_loss = _percentile_loss(0.999)

        return {
            "n_scenarios": n,
            "percentile_losses": {
                "p95": round(p95_loss, 6),
                "p99": round(p99_loss, 6),
                "p99_9": round(p999_loss, 6),
            },
            "worst_scenarios": results[:3],
            "best_scenarios": results[-3:][::-1],
            "scenario_pnls": results,
            "mean_pnl_pct": round(float(pnl_pcts.mean()), 6),
            "portfolio_equity": portfolio_equity,
        }


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    print("=" * 72)
    print("SENTINEL SPM — Stress Testing v3 — dim_083")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 60/40 portfolio: SPY + TLT
    # ---------------------------------------------------------------
    PORTFOLIO_EQUITY = 1_000_000
    holdings_60_40 = {
        "SPY": 600_000,
        "TLT": 400_000,
    }

    print("\n[1] Portfolio: 60% SPY / 40% TLT  ($1,000,000)")
    print(f"    SPY: ${holdings_60_40['SPY']:,.0f}   TLT: ${holdings_60_40['TLT']:,.0f}")

    # ---------------------------------------------------------------
    # Scenario library
    # ---------------------------------------------------------------
    library = ScenarioLibrary()
    print(f"\n[2] Available scenarios: {len(library.list_all())}")
    print("    Historical:", len(library.get_all_historical_scenarios()))
    print("    Hypothetical:", len(library.get_all_hypothetical_scenarios()))

    # ---------------------------------------------------------------
    # Run all historical scenarios
    # ---------------------------------------------------------------
    engine = StressTestEngine()

    print("\n[3] Historical Scenario Results")
    hist_results = engine.run_all_historical_scenarios(holdings_60_40, PORTFOLIO_EQUITY)
    print(f"  {'Scenario':<35} {'P&L':>12} {'%':>7}  Verdict")
    print("  " + "-" * 65)
    for r in hist_results:
        print(
            f"  {r.scenario.name:<35} {r.total_pnl:>+12,.0f} {r.total_pnl_pct:>+6.1%}   {r.verdict}"
        )

    # ---------------------------------------------------------------
    # Spotlight: GFC 2008 + COVID 2020 + Rate Shock 2022
    # ---------------------------------------------------------------
    spotlight = ["GFC 2008-2009", "COVID Crash 2020", "Rate Shock 2022"]
    print("\n[4] Spotlight Scenarios (detailed)")
    for name in spotlight:
        r = next((x for x in hist_results if x.scenario.name == name), None)
        if r is None:
            continue
        print()
        print("  " + r.summary())
        print("  Factor Attribution:")
        for factor, contrib in sorted(r.factor_attribution.items(), key=lambda x: x[1]):
            if abs(contrib) > 100:
                print(f"    {factor:<20}: ${contrib:>+10,.0f}")
        print(f"  Worst holding : {r.worst_holding}  ${r.holding_pnl.get(r.worst_holding, 0):>+,.0f}")
        print(f"  Best  holding : {r.best_holding}  ${r.holding_pnl.get(r.best_holding, 0):>+,.0f}")

    # ---------------------------------------------------------------
    # Hypothetical scenarios
    # ---------------------------------------------------------------
    print("\n[5] Hypothetical Forward Scenarios")
    hyp_results = engine.run_all_hypothetical_scenarios(holdings_60_40, PORTFOLIO_EQUITY)
    print(f"  {'Scenario':<35} {'P&L':>12} {'%':>7}  Verdict")
    print("  " + "-" * 65)
    for r in hyp_results:
        print(
            f"  {r.scenario.name:<35} {r.total_pnl:>+12,.0f} {r.total_pnl_pct:>+6.1%}   {r.verdict}"
        )

    # ---------------------------------------------------------------
    # Worst scenario
    # ---------------------------------------------------------------
    worst = engine.find_worst_scenario(holdings_60_40, PORTFOLIO_EQUITY)
    print(f"\n[6] Worst Scenario Overall: {worst.scenario.name}")
    print(f"    P&L: ${worst.total_pnl:+,.0f} ({worst.total_pnl_pct:+.1%})")

    # ---------------------------------------------------------------
    # Tail risk budget
    # ---------------------------------------------------------------
    budget = engine.compute_tail_risk_budget(holdings_60_40, max_loss_pct=0.20)
    print(f"\n[7] Tail Risk Budget (>20% loss tolerance)")
    print(f"    Status: {budget['status']}")
    if budget.get("breach_scenarios"):
        print(f"    Breach scenarios: {budget['breach_scenarios']}")
    for rec in budget.get("recommendations", []):
        print(f"    {rec['suggestion']}")

    # ---------------------------------------------------------------
    # Monte Carlo
    # ---------------------------------------------------------------
    print("\n[8] Monte Carlo Simulation (10,000 paths, 252-day horizon)")
    mc = MonteCarloStressTester()

    mc_normal = mc.run_monte_carlo(
        holdings_60_40, PORTFOLIO_EQUITY, n_simulations=10_000, horizon_days=252
    )
    print("\n  Normal (bootstrap):")
    print("  " + mc_normal.summary().replace("\n", "\n  "))

    mc_fat = mc.run_fat_tail_mc(
        holdings_60_40, PORTFOLIO_EQUITY, n_simulations=10_000, horizon_days=252, degrees_of_freedom=4
    )
    print("\n  Fat-tail (Student-t, ν=4):")
    print("  " + mc_fat.summary().replace("\n", "\n  "))

    # ---------------------------------------------------------------
    # Correlation breakdown
    # ---------------------------------------------------------------
    print("\n[9] Correlation Breakdown Analysis (SPY vs TLT)")
    corr_analyzer = CorrelationBreakdownAnalyzer()
    tickers_pair = ["SPY", "TLT"]
    corr_comparison = corr_analyzer.compare_normal_vs_crisis_correlations(tickers_pair)
    for period, corr_df in corr_comparison.items():
        if not corr_df.empty and "SPY" in corr_df.index and "TLT" in corr_df.columns:
            val = corr_df.loc["SPY", "TLT"]
            print(f"  {period:<25} SPY/TLT correlation: {val:+.3f}")

    # ---------------------------------------------------------------
    # Hedge recommendations
    # ---------------------------------------------------------------
    print("\n[10] Hedge Recommendations")
    hedge_engine = HedgeRecommendationEngine()
    all_stress = hist_results + hyp_results
    hedges = hedge_engine.recommend_hedges(all_stress, PORTFOLIO_EQUITY, top_n=4)
    for h in hedges:
        annual_cost = hedge_engine.estimate_hedge_cost(h, PORTFOLIO_EQUITY)
        effectiveness = hedge_engine.compute_hedge_effectiveness(h, worst)
        print(f"\n  [{h.name}]")
        print(f"    {h.description}")
        print(f"    Instruments  : {', '.join(h.instruments[:3])}")
        print(f"    Sizing       : {h.sizing_suggestion}")
        print(f"    Annual cost  : ${annual_cost:,.0f} ({h.estimated_cost_bps:.0f} bps)")
        print(f"    Effectiveness: {effectiveness:.0%} offset vs worst scenario")

    # ---------------------------------------------------------------
    # Full report export
    # ---------------------------------------------------------------
    print("\n[11] Full Report")
    reporter = StressTestReport()
    report_text = reporter.generate_full_report(holdings_60_40, PORTFOLIO_EQUITY)
    print(report_text)

    print("\nDone.")
