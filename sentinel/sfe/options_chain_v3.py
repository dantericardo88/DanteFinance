"""
Options Chain V3 — dim_004: Options chain (all strikes/expiries, live Greeks).

Upgrades dim_073 (options_flow_v3.py) from flow screener to a full chain analytics
platform: multi-expiry chains, live Greeks, vol surface, strategy builder, earnings
event analytics.

What is different from dim_073
-------------------------------
  dim_073  — unusual flow, block trades, gamma exposure, flow screening
  dim_004  — complete multi-expiry chain, all-strike Greeks, vol surface,
              SABR calibration, earnings analytics, strategy payoffs

Architecture
------------
  CompleteOptionsChain      — fetch all expiries × all strikes via yfinance
  GreeksEngine              — BS + CRR Binomial, IV Newton-Raphson, portfolio Greeks
  VolatilitySurface         — surface construction, smile, term structure, SABR
  OptionsEventAnalytics     — earnings implied move, IV crush, historical moves
  OptionsStrategyBuilder    — multi-leg payoff, breakevens, POP, strategy Greeks
  OptionsChainEngine        — orchestrator / export

Free Data Sources
-----------------
  yfinance           — options chains (all expiries), stock prices, earnings calendar
  FRED               — FEDFUNDS for risk-free rate (public CSV endpoint, no key)
  EDGAR EFTS         — earnings event dates (EDGAR full-text search)

Public API
----------
  engine = OptionsChainEngine()
  chain  = engine.get_chain("AAPL")
  surf   = engine.get_surface("AAPL")
  earn   = engine.get_earnings_analytics("AAPL")
  strat  = engine.build_and_analyze_strategy("AAPL", "iron_condor", {...})
  engine.export_chain("AAPL", "/tmp/aapl_chain.csv")
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from scipy.stats import norm as _scipy_norm
    from scipy.optimize import brentq as _scipy_brentq
    from scipy.interpolate import RectBivariateSpline as _RBS
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RF = 0.053          # fallback risk-free rate
_DAYS_PER_YEAR = 365.25
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
_MAX_WORKERS = 6
_RETRY_DELAY = 1.5
_MONEYNESS_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
_TENOR_GRID_YEARS = [
    7 / 365, 14 / 365, 30 / 365, 60 / 365, 90 / 365, 180 / 365, 365 / 365
]


# ---------------------------------------------------------------------------
# Math helpers — no scipy dependency
# ---------------------------------------------------------------------------

def _ncdf(x: float) -> float:
    """Standard normal CDF using math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _npdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _ncdf_v(x: np.ndarray) -> np.ndarray:
    """Vectorized standard normal CDF."""
    if HAS_SCIPY:
        return _scipy_norm.cdf(x)
    return np.vectorize(_ncdf)(x)


def _npdf_v(x: np.ndarray) -> np.ndarray:
    """Vectorized standard normal PDF."""
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GreeksResult:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0     # daily $ theta
    vega: float = 0.0      # per 1 vol point
    rho: float = 0.0
    charm: float = 0.0     # delta decay per day
    vanna: float = 0.0     # dDelta/dSigma
    volga: float = 0.0     # dVega/dSigma
    iv: float = 0.0
    option_price: float = 0.0
    option_type: str = ""

    def __add__(self, other: "GreeksResult") -> "GreeksResult":
        return GreeksResult(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            charm=self.charm + other.charm,
            vanna=self.vanna + other.vanna,
            volga=self.volga + other.volga,
        )

    def scale(self, qty: float) -> "GreeksResult":
        return GreeksResult(
            delta=self.delta * qty,
            gamma=self.gamma * qty,
            theta=self.theta * qty,
            vega=self.vega * qty,
            rho=self.rho * qty,
            charm=self.charm * qty,
            vanna=self.vanna * qty,
            volga=self.volga * qty,
        )


@dataclass
class OptionsSnapshot:
    ticker: str
    expiry: str
    spot: float
    dte: int
    risk_free_rate: float
    calls: pd.DataFrame = field(default_factory=pd.DataFrame)
    puts: pd.DataFrame = field(default_factory=pd.DataFrame)
    fetch_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def all_strikes(self) -> List[float]:
        strikes = set()
        if not self.calls.empty and "strike" in self.calls.columns:
            strikes.update(self.calls["strike"].dropna().tolist())
        if not self.puts.empty and "strike" in self.puts.columns:
            strikes.update(self.puts["strike"].dropna().tolist())
        return sorted(strikes)

    @property
    def atm_strike(self) -> float:
        strikes = self.all_strikes
        if not strikes:
            return self.spot
        return min(strikes, key=lambda k: abs(k - self.spot))


@dataclass
class CompleteChain:
    ticker: str
    spot: float
    risk_free_rate: float
    fetch_time: datetime
    snapshots: Dict[str, OptionsSnapshot] = field(default_factory=dict)

    @property
    def expiries(self) -> List[str]:
        return sorted(self.snapshots.keys())

    @property
    def all_options(self) -> pd.DataFrame:
        """Unified DataFrame of all calls and puts across all expiries."""
        frames = []
        for exp, snap in self.snapshots.items():
            for opt_type, df in [("call", snap.calls), ("put", snap.puts)]:
                if df.empty:
                    continue
                df2 = df.copy()
                df2["expiry"] = exp
                df2["option_type"] = opt_type
                df2["dte"] = snap.dte
                frames.append(df2)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return combined

    @property
    def total_options_count(self) -> int:
        n = 0
        for snap in self.snapshots.values():
            n += len(snap.calls) + len(snap.puts)
        return n


@dataclass
class VolSurface:
    ticker: str
    spot: float
    build_time: datetime
    moneyness_grid: List[float]      # K/S values
    tenor_grid_years: List[float]    # T values in years
    iv_matrix: np.ndarray            # shape: (len(tenor_grid), len(moneyness_grid))
    raw_points: pd.DataFrame = field(default_factory=pd.DataFrame)  # raw IV data
    is_valid: bool = False

    def get_iv(self, moneyness: float, T_years: float) -> float:
        """Lookup IV at given moneyness and tenor using bilinear interpolation."""
        return _bilinear_interpolate(
            self.moneyness_grid, self.tenor_grid_years, self.iv_matrix,
            moneyness, T_years
        )


@dataclass
class SABRParams:
    alpha: float   # initial vol level
    beta: float    # elasticity (fixed at 0.5 or calibrated)
    rho: float     # vol-spot correlation
    nu: float      # vol of vol
    T: float       # tenor
    F: float       # forward price


@dataclass
class EarningsPlayScore:
    ticker: str
    earnings_date: Optional[str]
    expected_move_pct: float       # ±% implied by ATM straddle
    historical_avg_move_pct: float
    iv_crush_estimate_pct: float
    pre_earnings_iv_rank: float
    strategy_recommendation: str  # "long_straddle", "iron_condor", "calendar", "skip"
    confidence: str               # "high", "medium", "low"
    rationale: str
    historical_moves: List[float] = field(default_factory=list)


@dataclass
class OptionsLeg:
    option_type: str    # "call" or "put"
    strike: float
    expiry: str
    quantity: int       # positive = long, negative = short
    premium: float      # per-share premium (midpoint)
    multiplier: int = 100


@dataclass
class OptionsStrategy:
    name: str
    ticker: str
    legs: List[OptionsLeg]
    spot_at_entry: float
    strategy_type: str  # "debit", "credit", "defined_risk"
    net_premium: float = 0.0  # positive = credit received, negative = debit paid

    def __post_init__(self):
        self.net_premium = sum(
            -leg.quantity * leg.premium * leg.multiplier for leg in self.legs
        )


@dataclass
class StrategyAnalysis:
    strategy: OptionsStrategy
    payoff_at_expiry: pd.DataFrame       # columns: spot_price, pnl
    breakevens: List[float]
    max_profit: float
    max_loss: float
    probability_of_profit: float
    greeks: GreeksResult
    notes: str = ""


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

_cached_rf: Optional[float] = None
_rf_fetched_at: float = 0.0
_RF_TTL = 3600 * 6   # refresh every 6 hours


def _get_risk_free_rate() -> float:
    """Fetch latest FEDFUNDS rate from FRED public CSV (no key required)."""
    global _cached_rf, _rf_fetched_at
    if _cached_rf is not None and time.time() - _rf_fetched_at < _RF_TTL:
        return _cached_rf
    try:
        resp = requests.get(_FRED_CSV, timeout=15)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        # Header line + data. Last non-empty line = most recent observation.
        for line in reversed(lines):
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    val = float(parts[1]) / 100.0
                    _cached_rf = val
                    _rf_fetched_at = time.time()
                    return val
                except ValueError:
                    continue
    except Exception as exc:
        logger.debug("FRED fetch failed: %s", exc)
    _cached_rf = _DEFAULT_RF
    _rf_fetched_at = time.time()
    return _DEFAULT_RF


def _dte_from_expiry(expiry_str: str) -> int:
    """Days to expiry (inclusive of today)."""
    try:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return max(0, (exp - date.today()).days)
    except Exception:
        return 0


def _T_from_expiry(expiry_str: str) -> float:
    """Time to expiry in years."""
    dte = _dte_from_expiry(expiry_str)
    return max(dte / _DAYS_PER_YEAR, 1e-6)


# ---------------------------------------------------------------------------
# GreeksEngine
# ---------------------------------------------------------------------------

class GreeksEngine:
    """
    Comprehensive Greeks computation using Black-Scholes and CRR Binomial tree.
    Supports European (BS) and American (Binomial) options.
    """

    # ------------------------------------------------------------------
    # Black-Scholes core
    # ------------------------------------------------------------------

    @staticmethod
    def _bs_d1d2(S: float, K: float, T: float, r: float,
                 sigma: float) -> Tuple[float, float]:
        """Compute d1 and d2 for Black-Scholes formula."""
        if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
            return (0.0, 0.0)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    @classmethod
    def bs_price(cls, S: float, K: float, T: float, r: float,
                 sigma: float, option_type: str) -> float:
        """Black-Scholes option price."""
        if T <= 0:
            return max(0.0, (S - K) if option_type == "call" else (K - S))
        d1, d2 = cls._bs_d1d2(S, K, T, r, sigma)
        df = math.exp(-r * T)
        if option_type == "call":
            return S * _ncdf(d1) - K * df * _ncdf(d2)
        else:
            return K * df * _ncdf(-d2) - S * _ncdf(-d1)

    @classmethod
    def compute_bs_greeks(cls, S: float, K: float, T: float, r: float,
                           sigma: float, option_type: str) -> GreeksResult:
        """
        Full set of Black-Scholes Greeks including second-order (charm, vanna, volga).
        Returns GreeksResult with all fields populated.
        """
        result = GreeksResult(option_type=option_type)
        if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
            result.option_price = cls.bs_price(S, K, T, r, sigma, option_type)
            return result

        d1, d2 = cls._bs_d1d2(S, K, T, r, sigma)
        sqrt_T = math.sqrt(T)
        nd1 = _npdf(d1)
        df = math.exp(-r * T)
        Nd1 = _ncdf(d1)
        Nd2 = _ncdf(d2)
        Nm_d1 = _ncdf(-d1)
        Nm_d2 = _ncdf(-d2)

        price = cls.bs_price(S, K, T, r, sigma, option_type)
        result.option_price = price

        # Delta
        if option_type == "call":
            result.delta = Nd1
        else:
            result.delta = Nd1 - 1.0

        # Gamma (same for call and put)
        result.gamma = nd1 / (S * sigma * sqrt_T)

        # Theta (per calendar day, in dollars for one share)
        if option_type == "call":
            theta_annual = (
                -(S * nd1 * sigma) / (2 * sqrt_T)
                - r * K * df * Nd2
            )
        else:
            theta_annual = (
                -(S * nd1 * sigma) / (2 * sqrt_T)
                + r * K * df * Nm_d2
            )
        result.theta = theta_annual / _DAYS_PER_YEAR

        # Vega (per 1% move in vol, for one share)
        vega_per_unit = S * nd1 * sqrt_T
        result.vega = vega_per_unit / 100.0  # per 1 vol point

        # Rho (per 1% move in rate)
        if option_type == "call":
            result.rho = K * T * df * Nd2 / 100.0
        else:
            result.rho = -K * T * df * Nm_d2 / 100.0

        # Charm: d(Delta)/d(t) per calendar day
        # charm = -npdf(d1) × [2rT - d2σ√T] / (2Tσ√T)
        if T > 1e-6:
            if option_type == "call":
                charm = -nd1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
            else:
                charm = -nd1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
            result.charm = charm / _DAYS_PER_YEAR
        else:
            result.charm = 0.0

        # Vanna: d(Delta)/d(sigma) = d(Vega)/d(S)
        result.vanna = (vega_per_unit / S) * (1 - d1 / (sigma * sqrt_T)) if sigma * sqrt_T > 0 else 0.0

        # Volga: d(Vega)/d(sigma) — convexity of vega
        result.volga = vega_per_unit * d1 * d2 / sigma

        return result

    # ------------------------------------------------------------------
    # Binomial (CRR) tree — American options
    # ------------------------------------------------------------------

    @classmethod
    def compute_binomial_greeks(cls, S: float, K: float, T: float, r: float,
                                 sigma: float, option_type: str,
                                 n_steps: int = 100) -> GreeksResult:
        """
        Cox-Ross-Rubinstein binomial tree.
        Supports early exercise (American options).
        Greeks via central differences around the tree.
        """
        result = GreeksResult(option_type=option_type)
        if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
            result.option_price = cls.bs_price(S, K, T, r, sigma, option_type)
            return result

        dt = T / n_steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp(r * dt) - d) / (u - d)
        p = max(0.0, min(1.0, p))
        q = 1.0 - p
        df_step = math.exp(-r * dt)

        def _tree_price(S0: float) -> float:
            # Terminal values
            ST = np.array([S0 * (u ** (n_steps - 2 * i)) for i in range(n_steps + 1)])
            if option_type == "call":
                V = np.maximum(ST - K, 0.0)
            else:
                V = np.maximum(K - ST, 0.0)
            # Backward induction
            for step in range(n_steps - 1, -1, -1):
                S_node = np.array([S0 * (u ** (step - 2 * i)) for i in range(step + 1)])
                V = df_step * (p * V[:-1] + q * V[1:])
                if option_type == "call":
                    intrinsic = np.maximum(S_node - K, 0.0)
                else:
                    intrinsic = np.maximum(K - S_node, 0.0)
                V = np.maximum(V, intrinsic)  # American early exercise
            return float(V[0])

        price = _tree_price(S)
        result.option_price = price

        # Delta and Gamma via central differences
        h = S * 0.005
        V_up = _tree_price(S + h)
        V_dn = _tree_price(S - h)
        result.delta = (V_up - V_dn) / (2 * h)
        result.gamma = (V_up - 2 * price + V_dn) / (h * h)

        # Theta via one-step backward
        if n_steps > 2:
            # Re-build tree at T - dt
            dt2 = (T - 2 * dt) / max(n_steps - 2, 1)
            u2 = math.exp(sigma * math.sqrt(dt2)) if dt2 > 0 else 1.0
            d2 = 1.0 / u2 if u2 > 0 else 1.0
            # Approximate: price(T) - price(T - 2dt) / (2dt)
            # Use BS for Theta as good approximation
            bs_g = cls.compute_bs_greeks(S, K, T, r, sigma, option_type)
            result.theta = bs_g.theta
        else:
            result.theta = 0.0

        # Vega via central diff on sigma
        h_sig = 0.01
        V_vs_up = _tree_price(S) if sigma + h_sig > 2.0 else None
        # Use separate trees for vega
        def _tree_sigma(sig: float) -> float:
            dt_ = T / n_steps
            u_ = math.exp(sig * math.sqrt(dt_))
            d_ = 1.0 / u_
            p_ = max(0, min(1, (math.exp(r * dt_) - d_) / (u_ - d_)))
            q_ = 1.0 - p_
            df_ = math.exp(-r * dt_)
            ST_ = np.array([S * (u_ ** (n_steps - 2 * i)) for i in range(n_steps + 1)])
            V_ = np.maximum(ST_ - K, 0) if option_type == "call" else np.maximum(K - ST_, 0)
            for step in range(n_steps - 1, -1, -1):
                S_ = np.array([S * (u_ ** (step - 2 * i)) for i in range(step + 1)])
                V_ = df_ * (p_ * V_[:-1] + q_ * V_[1:])
                intr_ = np.maximum(S_ - K, 0) if option_type == "call" else np.maximum(K - S_, 0)
                V_ = np.maximum(V_, intr_)
            return float(V_[0])

        sig_up = min(sigma + h_sig, 1.99)
        sig_dn = max(sigma - h_sig, 0.001)
        V_sig_up = _tree_sigma(sig_up)
        V_sig_dn = _tree_sigma(sig_dn)
        result.vega = (V_sig_up - V_sig_dn) / (2 * h_sig * 100)  # per vol point

        # Rho from BS (rate sensitivity less affected by tree structure)
        bs_res = cls.compute_bs_greeks(S, K, T, r, sigma, option_type)
        result.rho = bs_res.rho
        result.charm = bs_res.charm
        result.vanna = bs_res.vanna
        result.volga = bs_res.volga
        result.iv = sigma

        return result

    # ------------------------------------------------------------------
    # Implied volatility
    # ------------------------------------------------------------------

    @classmethod
    def compute_iv(cls, market_price: float, S: float, K: float, T: float,
                    r: float, option_type: str) -> float:
        """
        Compute implied volatility via Newton-Raphson with Brent fallback.
        Handles deep ITM/OTM edge cases.
        Returns float IV or nan on failure.
        """
        if T <= 0 or market_price <= 0:
            return float("nan")

        # Intrinsic check
        intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
        if market_price < intrinsic:
            return float("nan")

        # Newton-Raphson
        sigma = 0.30  # starting guess
        for _ in range(100):
            try:
                price = cls.bs_price(S, K, T, r, sigma, option_type)
                vega = S * _npdf(cls._bs_d1d2(S, K, T, r, sigma)[0]) * math.sqrt(T)
                if vega < 1e-8:
                    break
                diff = price - market_price
                if abs(diff) < 1e-8:
                    return round(sigma, 6)
                sigma -= diff / vega
                if sigma <= 0:
                    sigma = 1e-6
                if sigma > 10:
                    sigma = 10.0
            except Exception:
                break

        # Brent fallback
        def _objective(sig: float) -> float:
            return cls.bs_price(S, K, T, r, sig, option_type) - market_price

        if HAS_SCIPY:
            try:
                iv = _scipy_brentq(_objective, 1e-4, 10.0, xtol=1e-6, maxiter=200)
                return round(float(iv), 6)
            except Exception:
                pass
        else:
            # Manual bisection
            lo, hi = 1e-4, 10.0
            for _ in range(100):
                mid = (lo + hi) / 2.0
                if _objective(mid) > 0:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < 1e-6:
                    return round(mid, 6)

        # Final Newton attempt with broader range
        for guess in [0.10, 0.50, 1.00, 2.00]:
            sigma = guess
            for _ in range(50):
                try:
                    price = cls.bs_price(S, K, T, r, sigma, option_type)
                    vega = S * _npdf(cls._bs_d1d2(S, K, T, r, sigma)[0]) * math.sqrt(T)
                    if vega < 1e-10:
                        break
                    sigma -= (price - market_price) / vega
                    sigma = max(1e-4, min(10.0, sigma))
                    if abs(cls.bs_price(S, K, T, r, sigma, option_type) - market_price) < 1e-6:
                        return round(sigma, 6)
                except Exception:
                    break

        return float("nan")

    @classmethod
    def compute_greeks_for_chain(cls, chain: pd.DataFrame, spot: float,
                                  r: float = None) -> pd.DataFrame:
        """
        Vectorized Greek computation for an entire chain DataFrame.
        Expected columns: strike, expiry, option_type, impliedVolatility, bid, ask.
        Returns chain with appended Greek columns.
        """
        if r is None:
            r = _get_risk_free_rate()
        if chain.empty:
            return chain

        chain = chain.copy()

        # Compute mid price
        if "bid" in chain.columns and "ask" in chain.columns:
            chain["mid_price"] = (chain["bid"].fillna(0) + chain["ask"].fillna(0)) / 2
        elif "lastPrice" in chain.columns:
            chain["mid_price"] = chain["lastPrice"].fillna(0)
        else:
            chain["mid_price"] = 0.0

        greek_rows = []
        for _, row in chain.iterrows():
            K = float(row.get("strike", 0) or 0)
            expiry_str = str(row.get("expiry", ""))
            T = _T_from_expiry(expiry_str) if expiry_str else 1 / 365
            opt_type = str(row.get("option_type", "call")).lower()
            iv = float(row.get("impliedVolatility", 0) or 0)
            mid = float(row.get("mid_price", 0) or 0)

            # Compute or verify IV
            if iv <= 0.001 and mid > 0 and K > 0:
                iv = cls.compute_iv(mid, spot, K, T, r, opt_type)
            if iv <= 0 or math.isnan(iv):
                iv = 0.30  # fallback

            g = cls.compute_bs_greeks(spot, K, T, r, iv, opt_type)
            greek_rows.append({
                "iv_computed": iv,
                "delta": g.delta,
                "gamma": g.gamma,
                "theta": g.theta,
                "vega": g.vega,
                "rho": g.rho,
                "charm": g.charm,
                "vanna": g.vanna,
                "volga": g.volga,
                "bs_price": g.option_price,
            })

        greek_df = pd.DataFrame(greek_rows, index=chain.index)
        return pd.concat([chain, greek_df], axis=1)

    @classmethod
    def compute_portfolio_greeks(cls, positions: List[dict]) -> GreeksResult:
        """
        Aggregate Greeks for a portfolio of options positions.
        Each position: {S, K, T, r, sigma, option_type, quantity}
        quantity = number of contracts × multiplier (e.g., +100 = long 1 call)
        """
        total = GreeksResult()
        for pos in positions:
            S = float(pos.get("S", 100))
            K = float(pos.get("K", 100))
            T = float(pos.get("T", 30 / 365))
            r = float(pos.get("r", _DEFAULT_RF))
            sigma = float(pos.get("sigma", 0.30))
            opt_type = str(pos.get("option_type", "call")).lower()
            qty = float(pos.get("quantity", 1))
            g = cls.compute_bs_greeks(S, K, T, r, sigma, opt_type)
            total = total + g.scale(qty)
        return total


# ---------------------------------------------------------------------------
# CompleteOptionsChain
# ---------------------------------------------------------------------------

class CompleteOptionsChain:
    """
    Fetch complete options chain for all expiries and all strikes via yfinance.
    """

    def __init__(self):
        self._greeks = GreeksEngine()
        self._rf = _get_risk_free_rate()

    def get_all_expiries(self, ticker: str) -> List[str]:
        """Return all available option expiry dates for a ticker."""
        if not HAS_YF:
            return []
        try:
            t = yf.Ticker(ticker)
            return list(t.options)
        except Exception as exc:
            logger.warning("Cannot fetch expiries for %s: %s", ticker, exc)
            return []

    def _fetch_one_expiry(self, ticker_obj: Any, expiry: str, spot: float) -> OptionsSnapshot:
        """Fetch chain for one expiry, compute missing IVs."""
        try:
            chain = ticker_obj.option_chain(expiry)
        except Exception as exc:
            logger.warning("Failed to fetch chain for expiry %s: %s", expiry, exc)
            return OptionsSnapshot(
                ticker="", expiry=expiry, spot=spot,
                dte=_dte_from_expiry(expiry), risk_free_rate=self._rf
            )

        T = _T_from_expiry(expiry)
        dte = _dte_from_expiry(expiry)

        def _clean(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            df["option_type"] = opt_type
            df["expiry"] = expiry
            df["dte"] = dte
            for col in ["bid", "ask", "lastPrice", "impliedVolatility",
                         "volume", "openInterest"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Fill missing IV via Black-Scholes
            needs_iv = df["impliedVolatility"].le(0.001)
            if needs_iv.any():
                mid = (df["bid"] + df["ask"]) / 2
                for idx in df[needs_iv].index:
                    K = float(df.at[idx, "strike"])
                    mid_p = float(mid.at[idx])
                    if mid_p > 0 and K > 0:
                        iv = self._greeks.compute_iv(mid_p, spot, K, T, self._rf, opt_type)
                        if not math.isnan(iv):
                            df.at[idx, "impliedVolatility"] = iv
            return df

        calls = _clean(chain.calls, "call")
        puts = _clean(chain.puts, "put")

        return OptionsSnapshot(
            ticker="",
            expiry=expiry,
            spot=spot,
            dte=dte,
            risk_free_rate=self._rf,
            calls=calls,
            puts=puts,
        )

    def fetch_complete_chain(self, ticker: str) -> CompleteChain:
        """
        Fetch complete options chain for all expiries in parallel.
        Returns CompleteChain with all snapshots.
        """
        if not HAS_YF:
            return CompleteChain(ticker=ticker, spot=0.0, risk_free_rate=self._rf,
                                  fetch_time=datetime.utcnow())

        t = yf.Ticker(ticker)
        try:
            hist = t.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        except Exception:
            spot = 0.0

        if spot <= 0:
            try:
                info = t.info
                spot = float(info.get("regularMarketPrice") or info.get("previousClose") or 0)
            except Exception:
                spot = 100.0

        expiries = self.get_all_expiries(ticker)
        if not expiries:
            return CompleteChain(ticker=ticker, spot=spot, risk_free_rate=self._rf,
                                  fetch_time=datetime.utcnow())

        snapshots: Dict[str, OptionsSnapshot] = {}

        # Parallel fetch with thread pool
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(self._fetch_one_expiry, t, exp, spot): exp
                for exp in expiries
            }
            for fut in as_completed(futures):
                exp = futures[fut]
                try:
                    snap = fut.result()
                    snap.ticker = ticker
                    snapshots[exp] = snap
                except Exception as exc:
                    logger.warning("Expiry %s fetch error: %s", exp, exc)

        chain = CompleteChain(
            ticker=ticker,
            spot=spot,
            risk_free_rate=self._rf,
            fetch_time=datetime.utcnow(),
            snapshots=snapshots,
        )
        return chain

    def fetch_chain_snapshot(self, ticker: str, expiry: str) -> OptionsSnapshot:
        """Full chain for one expiry with computed Greeks for every contract."""
        if not HAS_YF:
            return OptionsSnapshot(ticker=ticker, expiry=expiry, spot=0.0,
                                    dte=_dte_from_expiry(expiry), risk_free_rate=self._rf)
        t = yf.Ticker(ticker)
        try:
            hist = t.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        except Exception:
            spot = 0.0

        snap = self._fetch_one_expiry(t, expiry, spot)
        snap.ticker = ticker

        # Compute Greeks for all contracts
        greeks_eng = GreeksEngine()
        for opt_type, attr in [("call", "calls"), ("put", "puts")]:
            df = getattr(snap, attr)
            if not df.empty:
                df_with_greeks = greeks_eng.compute_greeks_for_chain(df, spot, self._rf)
                setattr(snap, attr, df_with_greeks)

        return snap

    def get_atm_options(self, ticker: str, expiry: str, n_strikes: int = 10) -> pd.DataFrame:
        """
        Return n_strikes options around ATM (half below, half above).
        """
        snap = self.fetch_chain_snapshot(ticker, expiry)
        spot = snap.spot
        all_strikes = snap.all_strikes

        if not all_strikes:
            return pd.DataFrame()

        # Find ATM index
        idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - spot))
        half = n_strikes // 2
        lo = max(0, idx - half)
        hi = min(len(all_strikes), idx + half + 1)
        selected_strikes = set(all_strikes[lo:hi])

        frames = []
        for opt_type, df in [("call", snap.calls), ("put", snap.puts)]:
            if df.empty:
                continue
            mask = df["strike"].isin(selected_strikes) if "strike" in df.columns else pd.Series(False, index=df.index)
            subset = df[mask].copy()
            if not subset.empty:
                frames.append(subset)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            ["strike", "option_type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bilinear interpolation helper
# ---------------------------------------------------------------------------

def _bilinear_interpolate(x_grid: List[float], y_grid: List[float],
                            z_matrix: np.ndarray,
                            x: float, y: float) -> float:
    """
    Bilinear interpolation on a 2D grid.
    x_grid = moneyness axis, y_grid = tenor axis.
    z_matrix shape: (len(y_grid), len(x_grid)).
    """
    if HAS_SCIPY and z_matrix.shape[0] >= 3 and z_matrix.shape[1] >= 3:
        try:
            spline = _RBS(y_grid, x_grid, z_matrix, kx=min(3, len(y_grid) - 1),
                           ky=min(3, len(x_grid) - 1))
            val = float(spline(y, x))
            return max(0.001, val)
        except Exception:
            pass

    # Manual bilinear
    x_arr = np.array(x_grid)
    y_arr = np.array(y_grid)

    xi = np.searchsorted(x_arr, x) - 1
    xi = max(0, min(xi, len(x_arr) - 2))
    yi = np.searchsorted(y_arr, y) - 1
    yi = max(0, min(yi, len(y_arr) - 2))

    x0, x1 = x_arr[xi], x_arr[xi + 1]
    y0, y1 = y_arr[yi], y_arr[yi + 1]

    dx = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
    dy = (y - y0) / (y1 - y0) if y1 > y0 else 0.0

    z00 = z_matrix[yi, xi]
    z10 = z_matrix[yi + 1, xi]
    z01 = z_matrix[yi, xi + 1]
    z11 = z_matrix[yi + 1, xi + 1]

    z = (z00 * (1 - dx) * (1 - dy) +
         z01 * dx * (1 - dy) +
         z10 * (1 - dx) * dy +
         z11 * dx * dy)
    return max(0.001, float(z))


# ---------------------------------------------------------------------------
# VolatilitySurface
# ---------------------------------------------------------------------------

class VolatilitySurface:
    """
    Build and analyze the implied volatility surface.
    """

    def __init__(self):
        self._greeks = GreeksEngine()

    def build_surface(self, complete_chain: CompleteChain, spot: float) -> VolSurface:
        """
        Construct IV surface from complete chain data.
        Grid: moneyness [0.70..1.30] × tenor [1W..1Y].
        """
        if spot <= 0:
            spot = complete_chain.spot

        # Collect all valid IV points
        records = []
        for expiry, snap in complete_chain.snapshots.items():
            T = _T_from_expiry(expiry)
            if T < 1 / 365:
                continue
            for opt_type, df in [("call", snap.calls), ("put", snap.puts)]:
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    K = float(row.get("strike", 0) or 0)
                    if K <= 0 or spot <= 0:
                        continue
                    moneyness = K / spot
                    iv = float(row.get("impliedVolatility", 0) or 0)
                    if iv < 0.01 or iv > 5.0:
                        continue
                    if moneyness < 0.50 or moneyness > 2.00:
                        continue
                    bid = float(row.get("bid", 0) or 0)
                    if bid <= 0:
                        continue  # Filter illiquid strikes
                    records.append({
                        "expiry": expiry,
                        "T": T,
                        "K": K,
                        "moneyness": moneyness,
                        "iv": iv,
                        "option_type": opt_type,
                    })

        if not records:
            return VolSurface(
                ticker=complete_chain.ticker, spot=spot,
                build_time=datetime.utcnow(),
                moneyness_grid=_MONEYNESS_GRID,
                tenor_grid_years=_TENOR_GRID_YEARS,
                iv_matrix=np.full((len(_TENOR_GRID_YEARS), len(_MONEYNESS_GRID)), 0.30),
                is_valid=False,
            )

        raw_df = pd.DataFrame(records)

        # Build grid via interpolation
        iv_matrix = np.full((len(_TENOR_GRID_YEARS), len(_MONEYNESS_GRID)), np.nan)

        for j, T_target in enumerate(_TENOR_GRID_YEARS):
            # Use nearby tenors (within ±14 days)
            tol = max(7 / 365, T_target * 0.5)
            nearby = raw_df[(raw_df["T"] >= T_target - tol) &
                             (raw_df["T"] <= T_target + tol)]
            if nearby.empty:
                continue

            for i, m_target in enumerate(_MONEYNESS_GRID):
                # Find nearby moneyness
                m_tol = 0.05
                pts = nearby[(nearby["moneyness"] >= m_target - m_tol) &
                              (nearby["moneyness"] <= m_target + m_tol)]
                if pts.empty:
                    continue
                # Weight by proximity to target moneyness
                weights = 1.0 / (abs(pts["moneyness"] - m_target) + 1e-4)
                iv_matrix[j, i] = float(np.average(pts["iv"], weights=weights))

        # Fill NaN via forward-fill in both dimensions
        iv_df = pd.DataFrame(iv_matrix)
        iv_df = iv_df.interpolate(method="linear", axis=0).interpolate(method="linear", axis=1)
        iv_df = iv_df.fillna(0.30)
        iv_matrix = iv_df.values

        return VolSurface(
            ticker=complete_chain.ticker,
            spot=spot,
            build_time=datetime.utcnow(),
            moneyness_grid=_MONEYNESS_GRID,
            tenor_grid_years=_TENOR_GRID_YEARS,
            iv_matrix=iv_matrix,
            raw_points=raw_df,
            is_valid=True,
        )

    def compute_term_structure(self, complete_chain: CompleteChain = None,
                                surface: VolSurface = None) -> pd.Series:
        """
        ATM IV at each expiry — time structure of volatility.
        If surface provided, reads from grid; otherwise uses chain directly.
        """
        if surface is not None and surface.is_valid:
            records = {}
            for T in _TENOR_GRID_YEARS:
                iv = surface.get_iv(1.0, T)  # moneyness=1 = ATM
                exp_label = f"{round(T * 365)}d"
                records[exp_label] = iv
            return pd.Series(records, name="atm_iv")

        if complete_chain is None:
            return pd.Series(dtype=float)

        records = {}
        for expiry, snap in sorted(complete_chain.snapshots.items()):
            T = _T_from_expiry(expiry)
            if T <= 0:
                continue
            atm = snap.atm_strike
            # Find ATM call IV
            for df, opt_type in [(snap.calls, "call"), (snap.puts, "put")]:
                if df.empty:
                    continue
                if "strike" not in df.columns:
                    continue
                row = df.iloc[(df["strike"] - atm).abs().argsort()[:1]]
                if row.empty:
                    continue
                iv = float(row["impliedVolatility"].iloc[0] or 0)
                if iv > 0.01:
                    records[expiry] = iv
                    break

        return pd.Series(records, name="atm_iv").sort_index()

    def compute_smile(self, expiry: str, complete_chain: CompleteChain) -> pd.Series:
        """IV across strikes for one expiry — the volatility smile."""
        snap = complete_chain.snapshots.get(expiry)
        if snap is None:
            return pd.Series(dtype=float)

        smile_data = {}
        for df, opt_type in [(snap.calls, "call"), (snap.puts, "put")]:
            if df.empty:
                continue
            if "strike" not in df.columns or "impliedVolatility" not in df.columns:
                continue
            for _, row in df.iterrows():
                K = float(row["strike"])
                iv = float(row.get("impliedVolatility", 0) or 0)
                if iv > 0.01:
                    smile_data[K] = iv

        return pd.Series(dict(sorted(smile_data.items())), name=f"iv_smile_{expiry}")

    def detect_term_structure_inversion(self, surface: VolSurface) -> bool:
        """
        Check if front-month IV > back-month IV = near-term event risk (backwardation).
        """
        if not surface.is_valid:
            return False
        ts = self.compute_term_structure(surface=surface)
        if len(ts) < 2:
            return False
        # Compare first two tenors
        ivs = ts.values
        return bool(ivs[0] > ivs[1])

    def interpolate(self, surface: VolSurface, moneyness: float, T: float) -> float:
        """Get IV at arbitrary moneyness and tenor via surface interpolation."""
        return surface.get_iv(moneyness, T)

    @staticmethod
    def sabr_vol(K: float, F: float, T: float, alpha: float,
                  beta: float, rho: float, nu: float) -> float:
        """
        SABR volatility formula (Hagan et al. 2002).
        Returns implied vol for given strike K, forward F, expiry T.
        """
        if abs(F - K) < 1e-8:  # ATM case
            FK = F
            sqrt_FK = math.sqrt(FK)
            if T <= 0 or alpha <= 0:
                return alpha
            term1 = alpha / (FK ** (1 - beta))
            term2 = 1 + (
                ((1 - beta) ** 2 * alpha ** 2) / (24 * FK ** (2 - 2 * beta))
                + rho * beta * nu * alpha / (4 * FK ** (1 - beta))
                + (2 - 3 * rho ** 2) * nu ** 2 / 24
            ) * T
            return term1 * term2

        FK_mid = math.sqrt(F * K)
        log_FK = math.log(F / K)

        z = (nu / alpha) * (FK_mid ** (1 - beta)) * log_FK
        if abs(z) < 1e-8:
            x_z = 1.0
        else:
            x_z = math.log((math.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) /
                             (1 - rho)) / z

        A = alpha / (
            (FK_mid ** (1 - beta)) *
            (1 + (1 - beta) ** 2 / 24 * log_FK ** 2
             + (1 - beta) ** 4 / 1920 * log_FK ** 4)
        )
        B = 1 + (
            ((1 - beta) ** 2 * alpha ** 2) / (24 * FK_mid ** (2 - 2 * beta))
            + rho * beta * nu * alpha / (4 * FK_mid ** (1 - beta))
            + (2 - 3 * rho ** 2) * nu ** 2 / 24
        ) * T

        return A * (z / x_z) * B

    def calibrate_sabr(self, strikes: np.ndarray, ivs: np.ndarray,
                        T: float, F: float, beta: float = 0.5) -> Optional[SABRParams]:
        """
        Calibrate SABR parameters (alpha, rho, nu) by minimizing squared IV error.
        beta is typically fixed at 0.5 for equity options.
        """
        if not HAS_SCIPY:
            return None
        try:
            from scipy.optimize import minimize

            def _objective(params: np.ndarray) -> float:
                alpha, rho, nu = params
                if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                    return 1e10
                errors = []
                for K, mkt_iv in zip(strikes, ivs):
                    try:
                        model_iv = self.sabr_vol(K, F, T, alpha, beta, rho, nu)
                        errors.append((model_iv - mkt_iv) ** 2)
                    except Exception:
                        errors.append(1.0)
                return sum(errors)

            x0 = np.array([0.30, -0.30, 0.50])
            bounds = [(0.001, 2.0), (-0.999, 0.999), (0.001, 3.0)]
            res = minimize(_objective, x0, method="L-BFGS-B", bounds=bounds)

            if res.success:
                alpha, rho, nu = res.x
                return SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu, T=T, F=F)
        except Exception as exc:
            logger.warning("SABR calibration failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# OptionsEventAnalytics
# ---------------------------------------------------------------------------

class OptionsEventAnalytics:
    """
    Analyze options behavior around corporate earnings and events.
    """

    def __init__(self):
        self._chain = CompleteOptionsChain()
        self._greeks = GreeksEngine()

    def detect_earnings_date(self, ticker: str) -> Optional[str]:
        """
        Attempt to find next earnings date from yfinance calendar.
        Falls back to EDGAR EFTS search for 8-K 2.02 filings.
        """
        if not HAS_YF:
            return None
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
                if isinstance(cal, pd.DataFrame):
                    if "Earnings Date" in cal.index:
                        val = cal.loc["Earnings Date"].iloc[0]
                        return str(val.date()) if hasattr(val, "date") else str(val)
                    if "Earnings Date" in cal.columns:
                        val = cal["Earnings Date"].iloc[0]
                        return str(val.date()) if hasattr(val, "date") else str(val)
                elif isinstance(cal, dict):
                    earnings_dates = cal.get("Earnings Date", [])
                    if earnings_dates:
                        d = earnings_dates[0]
                        return str(d.date()) if hasattr(d, "date") else str(d)
        except Exception:
            pass

        # EDGAR EFTS fallback: find most recent 8-K (earnings release)
        try:
            url = (
                f"https://efts.sec.gov/LATEST/search-index?q=%22earnings%22&dateRange=custom"
                f"&startdt={(datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')}"
                f"&enddt={datetime.utcnow().strftime('%Y-%m-%d')}"
                f"&forms=8-K&entity={ticker}"
            )
            resp = requests.get(url, timeout=15,
                                 headers={"User-Agent": "SENTINEL/3.0 richard.porras@realempanada.com"})
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if hits:
                    filed = hits[0].get("_source", {}).get("period_of_report", "")
                    if filed:
                        return filed
        except Exception:
            pass
        return None

    def compute_pre_earnings_iv_run(self, ticker: str) -> float:
        """
        Estimate how much ATM IV typically rises into earnings.
        Looks at current vs 30-day-ago IV using historical prices.
        Returns percentage increase (e.g. 0.15 = +15% increase).
        """
        if not HAS_YF:
            return float("nan")
        try:
            t = yf.Ticker(ticker)
            expiries = list(t.options)
            if not expiries:
                return float("nan")
            # Use near-term expiry for IV measurement
            expiry = expiries[0]
            chain = t.option_chain(expiry)
            hist = t.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
            if spot <= 0:
                return float("nan")

            # Current ATM IV
            calls = chain.calls
            if calls.empty:
                return float("nan")
            atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
            current_iv = float(atm_call["impliedVolatility"].iloc[0] or 0)

            # Historical IV proxy: use 30-day HV as baseline
            hist_30 = t.history(period="3mo")
            if len(hist_30) < 20:
                return float("nan")
            returns = hist_30["Close"].pct_change().dropna()
            hv_30 = float(returns.std() * math.sqrt(252))

            if hv_30 <= 0:
                return float("nan")

            iv_run = (current_iv - hv_30) / hv_30
            return round(iv_run, 4)
        except Exception as exc:
            logger.debug("IV run computation failed for %s: %s", ticker, exc)
            return float("nan")

    def compute_earnings_expected_move(self, ticker: str) -> float:
        """
        Compute expected earnings move using ATM straddle price for
        the nearest post-earnings expiry.
        Returns expected move as fraction of spot (e.g. 0.08 = ±8%).
        """
        if not HAS_YF:
            return float("nan")
        try:
            t = yf.Ticker(ticker)
            expiries = list(t.options)
            if not expiries:
                return float("nan")

            hist = t.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
            if spot <= 0:
                return float("nan")

            # Use first available expiry (nearest)
            expiry = expiries[0]
            try:
                chain = t.option_chain(expiry)
            except Exception:
                return float("nan")

            calls = chain.calls
            puts = chain.puts
            if calls.empty or puts.empty:
                return float("nan")

            # Find ATM strike
            atm_strike = float(calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]["strike"].iloc[0])

            # Get ATM call and put prices
            call_row = calls[calls["strike"] == atm_strike]
            put_row = puts[puts["strike"] == atm_strike]

            call_mid = 0.0
            put_mid = 0.0
            if not call_row.empty:
                call_mid = float((call_row["bid"].iloc[0] + call_row["ask"].iloc[0]) / 2)
            if not put_row.empty:
                put_mid = float((put_row["bid"].iloc[0] + put_row["ask"].iloc[0]) / 2)

            straddle_price = call_mid + put_mid
            expected_move = straddle_price / spot if spot > 0 else float("nan")
            return round(expected_move, 4)
        except Exception as exc:
            logger.debug("Expected move computation failed for %s: %s", ticker, exc)
            return float("nan")

    def compute_historical_earnings_moves(self, ticker: str,
                                           n_events: int = 8) -> List[float]:
        """
        Estimate historical earnings day moves from price history.
        Uses quarterly patterns (every ~91 days) as earnings date proxy.
        Returns list of absolute percentage moves.
        """
        if not HAS_YF:
            return []
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="3y")
            if hist.empty:
                return []
            returns = hist["Close"].pct_change().dropna()
            # Find top N return days as proxy for earnings surprises
            abs_returns = returns.abs().sort_values(ascending=False)
            top_n = abs_returns.head(n_events * 2)
            # Filter for days > 3% move (earnings-like)
            earnings_like = top_n[top_n > 0.03].head(n_events)
            return [round(float(r), 4) for r in earnings_like.values]
        except Exception:
            return []

    def compute_iv_crush(self, ticker: str) -> float:
        """
        Estimate post-earnings IV crush magnitude.
        Proxy: difference between near-term and second-expiry ATM IV.
        Positive value = expected crush (near-term IV > back).
        """
        if not HAS_YF:
            return float("nan")
        try:
            t = yf.Ticker(ticker)
            expiries = list(t.options)
            if len(expiries) < 2:
                return float("nan")

            hist = t.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

            ivs = []
            for exp in expiries[:3]:
                try:
                    chain = t.option_chain(exp)
                    calls = chain.calls
                    if calls.empty:
                        continue
                    atm = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
                    iv = float(atm["impliedVolatility"].iloc[0] or 0)
                    if iv > 0.01:
                        ivs.append(iv)
                except Exception:
                    continue

            if len(ivs) < 2:
                return float("nan")
            # Crush = how much front IV exceeds back
            crush = (ivs[0] - ivs[1]) / ivs[0] if ivs[0] > 0 else 0.0
            return round(crush, 4)
        except Exception:
            return float("nan")

    def score_earnings_play(self, ticker: str) -> EarningsPlayScore:
        """
        Full earnings options strategy scoring.
        Recommends: long_straddle, iron_condor, calendar, or skip.
        """
        earnings_date = self.detect_earnings_date(ticker)
        expected_move = self.compute_earnings_expected_move(ticker)
        hist_moves = self.compute_historical_earnings_moves(ticker)
        iv_crush = self.compute_iv_crush(ticker)
        iv_run = self.compute_pre_earnings_iv_run(ticker)

        hist_avg = float(np.mean(hist_moves)) if hist_moves else float("nan")

        # Scoring logic
        recommendation = "skip"
        confidence = "low"
        rationale_parts = []

        if math.isnan(expected_move) or expected_move <= 0:
            rationale_parts.append("Cannot compute expected move — insufficient option data.")
        else:
            rationale_parts.append(f"Market implies ±{expected_move*100:.1f}% earnings move.")

            if not math.isnan(hist_avg):
                rationale_parts.append(f"Historical avg move: {hist_avg*100:.1f}%.")
                if expected_move > hist_avg * 1.2:
                    # Market is pricing more than history suggests — sell vol
                    recommendation = "iron_condor"
                    confidence = "medium"
                    rationale_parts.append("IV appears rich vs history — favor selling premium (iron condor).")
                elif expected_move < hist_avg * 0.8:
                    # Market is underpricing — buy vol
                    recommendation = "long_straddle"
                    confidence = "medium"
                    rationale_parts.append("IV appears cheap vs history — favor buying vol (straddle).")
                else:
                    recommendation = "calendar"
                    confidence = "low"
                    rationale_parts.append("Implied move in line with history — calendar spread may exploit IV crush.")
            else:
                recommendation = "long_straddle" if expected_move > 0.05 else "iron_condor"
                confidence = "low"

            if not math.isnan(iv_crush) and iv_crush > 0.30:
                rationale_parts.append(f"Significant IV crush expected ({iv_crush*100:.0f}%) — post-earnings vol may collapse.")

        # IV run: if already inflated, selling makes more sense
        pre_earnings_iv_rank = iv_run if not math.isnan(iv_run) else 0.0

        return EarningsPlayScore(
            ticker=ticker,
            earnings_date=earnings_date,
            expected_move_pct=expected_move if not math.isnan(expected_move) else 0.0,
            historical_avg_move_pct=hist_avg if not math.isnan(hist_avg) else 0.0,
            iv_crush_estimate_pct=iv_crush if not math.isnan(iv_crush) else 0.0,
            pre_earnings_iv_rank=pre_earnings_iv_rank,
            strategy_recommendation=recommendation,
            confidence=confidence,
            rationale=" ".join(rationale_parts),
            historical_moves=hist_moves,
        )


# ---------------------------------------------------------------------------
# OptionsStrategyBuilder
# ---------------------------------------------------------------------------

class OptionsStrategyBuilder:
    """
    Build and analyze multi-leg options strategies.
    Supports: covered_call, csp, straddle, strangle, spreads,
              iron_condor, iron_butterfly, calendar, diagonal, pmcc.
    """

    def __init__(self):
        self._chain = CompleteOptionsChain()
        self._greeks = GreeksEngine()

    def _get_snap(self, ticker: str, expiry: str) -> OptionsSnapshot:
        return self._chain.fetch_chain_snapshot(ticker, expiry)

    def _mid_price(self, row: pd.Series) -> float:
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return float(row.get("lastPrice", 0) or 0)

    def _find_option(self, snap: OptionsSnapshot, strike: float,
                      opt_type: str) -> Optional[pd.Series]:
        df = snap.calls if opt_type == "call" else snap.puts
        if df.empty or "strike" not in df.columns:
            return None
        mask = df["strike"] == strike
        if not mask.any():
            # Nearest strike
            idx = (df["strike"] - strike).abs().idxmin()
            return df.loc[idx]
        return df[mask].iloc[0]

    def build_strategy(self, ticker: str, strategy_name: str,
                        params: dict) -> OptionsStrategy:
        """
        Build a multi-leg strategy.

        Required params vary by strategy:
          covered_call:      expiry, strike (short call)
          csp:               expiry, strike (short put)
          long_straddle:     expiry, strike (ATM)
          short_straddle:    expiry, strike (ATM)
          long_strangle:     expiry, call_strike, put_strike
          short_strangle:    expiry, call_strike, put_strike
          bull_call_spread:  expiry, long_strike, short_strike
          bear_put_spread:   expiry, long_strike, short_strike
          iron_condor:       expiry, put_buy, put_sell, call_sell, call_buy
          iron_butterfly:    expiry, put_buy, atm_strike, call_buy
          calendar:          near_expiry, far_expiry, strike, option_type
          diagonal:          near_expiry, far_expiry, near_strike, far_strike, option_type
          pmcc:              near_expiry, far_expiry, short_strike, long_strike
        """
        s = strategy_name.lower().replace(" ", "_").replace("-", "_")
        expiry = params.get("expiry", "")

        if not HAS_YF:
            return OptionsStrategy(name=s, ticker=ticker, legs=[],
                                    spot_at_entry=0.0, strategy_type="unknown")

        t = yf.Ticker(ticker)
        try:
            hist = t.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        except Exception:
            spot = 100.0

        rf = _get_risk_free_rate()
        legs: List[OptionsLeg] = []

        def _leg(opt_type, strike, qty, exp):
            snap = self._get_snap(ticker, exp)
            row = self._find_option(snap, strike, opt_type)
            prem = self._mid_price(row) if row is not None else 0.0
            return OptionsLeg(option_type=opt_type, strike=strike, expiry=exp,
                               quantity=qty, premium=prem)

        if s == "covered_call":
            strike = params.get("strike", spot * 1.05)
            legs = [_leg("call", strike, -1, expiry)]
            strategy_type = "credit"

        elif s == "csp":  # cash-secured put
            strike = params.get("strike", spot * 0.95)
            legs = [_leg("put", strike, -1, expiry)]
            strategy_type = "credit"

        elif s in ("long_straddle", "straddle"):
            strike = params.get("strike", spot)
            legs = [_leg("call", strike, 1, expiry), _leg("put", strike, 1, expiry)]
            strategy_type = "debit"

        elif s == "short_straddle":
            strike = params.get("strike", spot)
            legs = [_leg("call", strike, -1, expiry), _leg("put", strike, -1, expiry)]
            strategy_type = "credit"

        elif s == "long_strangle":
            call_k = params.get("call_strike", spot * 1.05)
            put_k = params.get("put_strike", spot * 0.95)
            legs = [_leg("call", call_k, 1, expiry), _leg("put", put_k, 1, expiry)]
            strategy_type = "debit"

        elif s == "short_strangle":
            call_k = params.get("call_strike", spot * 1.05)
            put_k = params.get("put_strike", spot * 0.95)
            legs = [_leg("call", call_k, -1, expiry), _leg("put", put_k, -1, expiry)]
            strategy_type = "credit"

        elif s == "bull_call_spread":
            long_k = params.get("long_strike", spot)
            short_k = params.get("short_strike", spot * 1.05)
            legs = [_leg("call", long_k, 1, expiry), _leg("call", short_k, -1, expiry)]
            strategy_type = "debit"

        elif s == "bear_put_spread":
            long_k = params.get("long_strike", spot)
            short_k = params.get("short_strike", spot * 0.95)
            legs = [_leg("put", long_k, 1, expiry), _leg("put", short_k, -1, expiry)]
            strategy_type = "debit"

        elif s == "bull_put_spread":
            sell_k = params.get("sell_strike", spot * 0.95)
            buy_k = params.get("buy_strike", spot * 0.90)
            legs = [_leg("put", sell_k, -1, expiry), _leg("put", buy_k, 1, expiry)]
            strategy_type = "credit"

        elif s == "bear_call_spread":
            sell_k = params.get("sell_strike", spot * 1.05)
            buy_k = params.get("buy_strike", spot * 1.10)
            legs = [_leg("call", sell_k, -1, expiry), _leg("call", buy_k, 1, expiry)]
            strategy_type = "credit"

        elif s == "iron_condor":
            pb = params.get("put_buy", spot * 0.85)
            ps = params.get("put_sell", spot * 0.92)
            cs = params.get("call_sell", spot * 1.08)
            cb = params.get("call_buy", spot * 1.15)
            legs = [
                _leg("put", pb, 1, expiry),
                _leg("put", ps, -1, expiry),
                _leg("call", cs, -1, expiry),
                _leg("call", cb, 1, expiry),
            ]
            strategy_type = "credit"

        elif s == "iron_butterfly":
            atm = params.get("atm_strike", spot)
            pb = params.get("put_buy", spot * 0.90)
            cb = params.get("call_buy", spot * 1.10)
            legs = [
                _leg("put", pb, 1, expiry),
                _leg("put", atm, -1, expiry),
                _leg("call", atm, -1, expiry),
                _leg("call", cb, 1, expiry),
            ]
            strategy_type = "credit"

        elif s == "calendar":
            near_exp = params.get("near_expiry", expiry)
            far_exp = params.get("far_expiry", expiry)
            strike = params.get("strike", spot)
            opt_type = params.get("option_type", "call")
            legs = [
                _leg(opt_type, strike, -1, near_exp),
                _leg(opt_type, strike, 1, far_exp),
            ]
            strategy_type = "debit"

        elif s == "diagonal":
            near_exp = params.get("near_expiry", expiry)
            far_exp = params.get("far_expiry", expiry)
            near_k = params.get("near_strike", spot * 1.05)
            far_k = params.get("far_strike", spot * 1.00)
            opt_type = params.get("option_type", "call")
            legs = [
                _leg(opt_type, near_k, -1, near_exp),
                _leg(opt_type, far_k, 1, far_exp),
            ]
            strategy_type = "debit"

        elif s in ("pmcc", "poor_mans_covered_call"):
            near_exp = params.get("near_expiry", expiry)
            far_exp = params.get("far_expiry", expiry)
            short_k = params.get("short_strike", spot * 1.05)
            long_k = params.get("long_strike", spot * 0.80)  # Deep ITM LEAPS
            legs = [
                _leg("call", short_k, -1, near_exp),
                _leg("call", long_k, 1, far_exp),
            ]
            strategy_type = "debit"

        else:
            legs = []
            strategy_type = "unknown"

        strat = OptionsStrategy(
            name=s, ticker=ticker, legs=legs,
            spot_at_entry=spot, strategy_type=strategy_type
        )
        return strat

    def compute_payoff(self, strategy: OptionsStrategy,
                        spot_range: np.ndarray) -> np.ndarray:
        """
        Compute strategy PnL at expiry across a range of spot prices.
        Returns array of PnL values (per contract × multiplier).
        """
        pnl = np.zeros(len(spot_range))
        for leg in strategy.legs:
            K = leg.strike
            qty = leg.quantity
            mult = leg.multiplier
            premium = leg.premium

            if leg.option_type == "call":
                intrinsic = np.maximum(spot_range - K, 0.0)
            else:
                intrinsic = np.maximum(K - spot_range, 0.0)

            # PnL = qty × (intrinsic - premium) × multiplier
            leg_pnl = qty * (intrinsic - premium) * mult
            pnl += leg_pnl

        return pnl

    def compute_breakevens(self, strategy: OptionsStrategy) -> List[float]:
        """
        Find breakeven prices by zero-crossing of payoff curve.
        """
        spot = strategy.spot_at_entry
        lo = spot * 0.50
        hi = spot * 2.00
        spot_range = np.linspace(lo, hi, 1000)
        pnl = self.compute_payoff(strategy, spot_range)

        breakevens = []
        for i in range(len(pnl) - 1):
            if pnl[i] * pnl[i + 1] <= 0:
                # Linear interpolation for zero crossing
                if pnl[i + 1] - pnl[i] != 0:
                    be = spot_range[i] - pnl[i] * (spot_range[i + 1] - spot_range[i]) / (pnl[i + 1] - pnl[i])
                    breakevens.append(round(float(be), 2))
        return sorted(set(breakevens))

    def compute_max_profit_loss(self, strategy: OptionsStrategy) -> Tuple[float, float]:
        """
        Returns (max_profit, max_loss) using payoff scan.
        max_loss is negative (cost/loss).
        """
        spot = strategy.spot_at_entry
        lo = spot * 0.10
        hi = spot * 3.00
        spot_range = np.linspace(lo, hi, 2000)
        pnl = self.compute_payoff(strategy, spot_range)

        max_profit = float(np.max(pnl))
        max_loss = float(np.min(pnl))

        # Clamp "infinite" profit/loss for certain strategy types
        if strategy.strategy_type == "debit":
            max_loss = min(max_loss, -abs(strategy.net_premium))
        return max_profit, max_loss

    def compute_probability_of_profit(self, strategy: OptionsStrategy,
                                       vol: float) -> float:
        """
        Estimate probability of profit using log-normal stock price distribution.
        Integrates over the regions where payoff > 0.
        """
        spot = strategy.spot_at_entry
        breakevens = self.compute_breakevens(strategy)
        if not breakevens:
            return 0.50  # No information

        T = min(_T_from_expiry(leg.expiry) for leg in strategy.legs) if strategy.legs else 30 / 365
        r = _get_risk_free_rate()

        # Log-normal parameters
        mu = math.log(spot) + (r - 0.5 * vol ** 2) * T
        sigma_t = vol * math.sqrt(T)

        if sigma_t <= 0:
            return 0.50

        # Sample payoff at many spot points
        spot_range = np.linspace(spot * 0.30, spot * 3.0, 2000)
        pnl = self.compute_payoff(strategy, spot_range)

        # Compute log-normal probability weights
        log_spots = np.log(np.maximum(spot_range, 1e-8))
        probs = _npdf_v((log_spots - mu) / sigma_t) / (spot_range * sigma_t)
        probs = probs / probs.sum()

        pop = float(np.sum(probs[pnl > 0]))
        return round(min(1.0, max(0.0, pop)), 4)

    def compute_strategy_greeks(self, strategy: OptionsStrategy,
                                 spot: float) -> GreeksResult:
        """Aggregate Greeks for all legs of the strategy."""
        positions = []
        r = _get_risk_free_rate()
        for leg in strategy.legs:
            T = _T_from_expiry(leg.expiry)
            snap = self._chain._fetch_one_expiry(yf.Ticker(strategy.ticker), leg.expiry, spot) if HAS_YF else None
            row = self._find_option(snap, leg.strike, leg.option_type) if snap else None
            iv = float(row.get("impliedVolatility", 0.30) or 0.30) if row is not None else 0.30
            positions.append({
                "S": spot, "K": leg.strike, "T": T, "r": r,
                "sigma": iv, "option_type": leg.option_type,
                "quantity": leg.quantity * leg.multiplier,
            })
        return GreeksEngine.compute_portfolio_greeks(positions)


# ---------------------------------------------------------------------------
# OptionsChainEngine (Orchestrator)
# ---------------------------------------------------------------------------

class OptionsChainEngine:
    """
    Orchestrator for complete options chain analytics.
    """

    def __init__(self):
        self._chain = CompleteOptionsChain()
        self._greeks = GreeksEngine()
        self._surface = VolatilitySurface()
        self._events = OptionsEventAnalytics()
        self._strategy = OptionsStrategyBuilder()

    def get_chain(self, ticker: str) -> CompleteChain:
        """Fetch and return complete options chain with all expiries."""
        return self._chain.fetch_complete_chain(ticker)

    def get_surface(self, ticker: str, chain: Optional[CompleteChain] = None) -> VolSurface:
        """Build and return volatility surface for a ticker."""
        if chain is None:
            chain = self.get_chain(ticker)
        return self._surface.build_surface(chain, chain.spot)

    def get_earnings_analytics(self, ticker: str) -> EarningsPlayScore:
        """Return earnings options analytics and strategy recommendation."""
        return self._events.score_earnings_play(ticker)

    def build_and_analyze_strategy(self, ticker: str, strategy: str,
                                    params: dict) -> StrategyAnalysis:
        """
        Build strategy and return complete analysis:
        payoff diagram, breakevens, max P&L, POP, Greeks.
        """
        strat = self._strategy.build_strategy(ticker, strategy, params)
        spot = strat.spot_at_entry

        # Payoff
        lo = spot * 0.50
        hi = spot * 2.00
        spot_range = np.linspace(lo, hi, 500)
        pnl = self._strategy.compute_payoff(strat, spot_range)
        payoff_df = pd.DataFrame({"spot_price": spot_range, "pnl": pnl})

        breakevens = self._strategy.compute_breakevens(strat)
        max_profit, max_loss = self._strategy.compute_max_profit_loss(strat)

        # POP — use mean of option IVs
        avg_iv = 0.30
        if strat.legs:
            ivs = []
            for leg in strat.legs:
                if HAS_YF:
                    try:
                        snap = self._chain.fetch_chain_snapshot(ticker, leg.expiry)
                        row = self._strategy._find_option(snap, leg.strike, leg.option_type)
                        if row is not None:
                            iv = float(row.get("impliedVolatility", 0) or 0)
                            if iv > 0:
                                ivs.append(iv)
                    except Exception:
                        pass
            if ivs:
                avg_iv = float(np.mean(ivs))

        pop = self._strategy.compute_probability_of_profit(strat, avg_iv)

        # Greeks
        greeks = self._strategy.compute_strategy_greeks(strat, spot)

        notes = (
            f"Net premium: ${strat.net_premium:.2f}  "
            f"Type: {strat.strategy_type}  "
            f"Legs: {len(strat.legs)}"
        )

        return StrategyAnalysis(
            strategy=strat,
            payoff_at_expiry=payoff_df,
            breakevens=breakevens,
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            probability_of_profit=pop,
            greeks=greeks,
            notes=notes,
        )

    def export_chain(self, ticker: str, path: str,
                      chain: Optional[CompleteChain] = None) -> None:
        """
        Export complete chain with Greeks to CSV.
        """
        if chain is None:
            chain = self.get_chain(ticker)

        spot = chain.spot
        rf = _get_risk_free_rate()

        # Compute Greeks for entire chain
        all_frames = []
        for expiry, snap in chain.snapshots.items():
            for opt_type, df in [("call", snap.calls), ("put", snap.puts)]:
                if df.empty:
                    continue
                df2 = df.copy()
                df2["expiry"] = expiry
                df2["option_type"] = opt_type
                df2["dte"] = snap.dte
                df2 = self._greeks.compute_greeks_for_chain(df2, spot, rf)
                all_frames.append(df2)

        if not all_frames:
            logger.warning("No chain data to export for %s", ticker)
            return

        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(path, index=False)
        logger.info("Exported %d option contracts for %s to %s",
                    len(combined), ticker, path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    log = logging.getLogger("options_chain_v3.main")

    TICKER = "AAPL"
    engine = OptionsChainEngine()

    print("\n" + "=" * 70)
    print(f"SENTINEL — Options Chain V3 (dim_004) — {TICKER}")
    print("=" * 70)

    # 1. Fetch complete chain
    print(f"\n[1] Fetching complete {TICKER} options chain (all expiries)...")
    chain = engine.get_chain(TICKER)
    print(f"    Spot:        ${chain.spot:.2f}")
    print(f"    Expiries:    {len(chain.expiries)}")
    print(f"    Total opts:  {chain.total_options_count}")
    if chain.expiries:
        print(f"    Range:       {chain.expiries[0]} → {chain.expiries[-1]}")

    # 2. Build volatility surface
    print(f"\n[2] Building volatility surface...")
    if chain.total_options_count > 0:
        surf = engine.get_surface(TICKER, chain)
        print(f"    Surface valid: {surf.is_valid}")
        if surf.is_valid:
            atm_1m = surf.get_iv(1.0, 30 / 365)
            atm_3m = surf.get_iv(1.0, 90 / 365)
            otm_1m = surf.get_iv(0.90, 30 / 365)
            print(f"    ATM 1M IV:  {atm_1m*100:.1f}%")
            print(f"    ATM 3M IV:  {atm_3m*100:.1f}%")
            print(f"    90% 1M IV:  {otm_1m*100:.1f}% (skew proxy)")
            ts = engine._surface.compute_term_structure(chain)
            if not ts.empty:
                print(f"    Term structure:")
                for exp, iv in ts.head(6).items():
                    print(f"      {exp}: {iv*100:.1f}%")
            inverted = engine._surface.detect_term_structure_inversion(surf)
            print(f"    Term structure inverted: {inverted}")
    else:
        print("    Insufficient chain data for surface construction.")

    # 3. Earnings expected move
    print(f"\n[3] Earnings Analytics — {TICKER}")
    earn = engine.get_earnings_analytics(TICKER)
    print(f"    Next earnings date:   {earn.earnings_date or 'Unknown'}")
    print(f"    Expected move (±):    {earn.expected_move_pct*100:.1f}%")
    print(f"    Historical avg move:  {earn.historical_avg_move_pct*100:.1f}%")
    print(f"    IV crush estimate:    {earn.iv_crush_estimate_pct*100:.1f}%")
    print(f"    Recommendation:       {earn.strategy_recommendation.upper()}")
    print(f"    Confidence:           {earn.confidence}")
    print(f"    Rationale:            {earn.rationale}")

    # 4. Iron condor strategy analysis
    print(f"\n[4] Iron Condor Strategy Analysis — {TICKER}")
    expiries = chain.expiries
    if expiries:
        # Pick 30-45 DTE expiry
        target_exp = next(
            (e for e in expiries if 25 <= _dte_from_expiry(e) <= 50),
            expiries[min(2, len(expiries) - 1)]
        )
        spot = chain.spot
        ic_params = {
            "expiry": target_exp,
            "put_buy":  round(spot * 0.85, 0),
            "put_sell": round(spot * 0.92, 0),
            "call_sell": round(spot * 1.08, 0),
            "call_buy": round(spot * 1.15, 0),
        }
        try:
            analysis = engine.build_and_analyze_strategy(TICKER, "iron_condor", ic_params)
            strat = analysis.strategy
            print(f"    Expiry:        {target_exp} ({_dte_from_expiry(target_exp)} DTE)")
            print(f"    Strikes:       {ic_params['put_buy']:.0f} / {ic_params['put_sell']:.0f} / "
                  f"{ic_params['call_sell']:.0f} / {ic_params['call_buy']:.0f}")
            print(f"    Net premium:   ${strat.net_premium:.2f}")
            print(f"    Max profit:    ${analysis.max_profit:.2f}")
            print(f"    Max loss:      ${analysis.max_loss:.2f}")
            print(f"    Breakevens:    {analysis.breakevens}")
            print(f"    Prob of profit:{analysis.probability_of_profit*100:.1f}%")
            g = analysis.greeks
            print(f"    Net delta:     {g.delta:.4f}")
            print(f"    Net gamma:     {g.gamma:.6f}")
            print(f"    Net theta:     ${g.theta:.4f}/day")
            print(f"    Net vega:      ${g.vega:.4f}/vol-pt")
        except Exception as exc:
            print(f"    Strategy analysis error: {exc}")
    else:
        print("    No expiries available for strategy analysis.")

    # 5. Top 20 contracts by open interest with Greeks
    print(f"\n[5] Top 20 Contracts by Open Interest — {TICKER}")
    all_opts = chain.all_options
    if not all_opts.empty and "openInterest" in all_opts.columns:
        top_oi = all_opts.nlargest(20, "openInterest")
        top_oi_g = GreeksEngine.compute_greeks_for_chain(top_oi, chain.spot)
        print(f"    {'Expiry':12s} {'Type':6s} {'Strike':8s} {'OI':10s} "
              f"{'IV':7s} {'Delta':7s} {'Gamma':8s} {'Theta':8s}")
        print("    " + "-" * 68)
        for _, row in top_oi_g.head(20).iterrows():
            print(f"    {str(row.get('expiry','?')):12s} "
                  f"{str(row.get('option_type','?')):6s} "
                  f"{row.get('strike',0):8.1f} "
                  f"{int(row.get('openInterest',0)):10,d} "
                  f"{row.get('iv_computed',0)*100:6.1f}% "
                  f"{row.get('delta',0):7.4f} "
                  f"{row.get('gamma',0):8.6f} "
                  f"${row.get('theta',0):7.4f}")
    else:
        print("    No open interest data available.")

    # 6. Greeks example — standalone computation
    print(f"\n[6] Greeks Test — AAPL-like call (S=200, K=210, T=30d, vol=30%)")
    g = GreeksEngine.compute_bs_greeks(S=200, K=210, T=30/365, r=0.053,
                                        sigma=0.30, option_type="call")
    print(f"    Price:  ${g.option_price:.4f}")
    print(f"    Delta:  {g.delta:.4f}")
    print(f"    Gamma:  {g.gamma:.6f}")
    print(f"    Theta:  ${g.theta:.4f}/day")
    print(f"    Vega:   ${g.vega:.4f}/vol-pt")
    print(f"    Rho:    ${g.rho:.4f}/rate-pt")
    print(f"    Charm:  {g.charm:.6f}/day")
    print(f"    Vanna:  {g.vanna:.6f}")
    print(f"    Volga:  {g.volga:.6f}")

    print("\n" + "=" * 70)
    print("Options Chain V3 complete.")
    sys.exit(0)
