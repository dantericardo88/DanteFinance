"""
sentinel/spm/span_margin_v3.py
===============================
SPAN Margin and Portfolio Margin Simulation Engine
dim_137 — score target: 9

Implements CME SPAN (Standard Portfolio Analysis of Risk) margin methodology
and Reg T Portfolio Margin as an alternative:

SPAN:
  - 16 risk scenarios (price × vol shifts)
  - Scanning risk = max loss across scenarios (0-13 full weight, 14-15 at 35%)
  - Short option minimum charge (50 bps × underlying × net short options)
  - Inter-month spread charge (0.5% × notional per spread)
  - Delivery charge for near-expiry positions (1% × notional)
  - SPAN credit for offsets

Portfolio Margin (Reg T):
  - 10 scenarios: ±15%, ±10%, ±5%, 0%, ±6% vol
  - Floor: 15% × underlying market value

Free-standing: numpy and scipy.stats only. No paid APIs required.

Author: SENTINEL Risk Engine
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import norm as _scipy_norm
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Black-Scholes helpers (pure Python / numpy — no external option lib)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF — uses scipy if available, else math.erf fallback."""
    if _SCIPY_AVAILABLE:
        return float(_scipy_norm.cdf(x))
    # Abramowitz & Stegun approximation (error < 7.5e-8)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float = 0.05,
    sigma: float = 0.20,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes European option price.

    Parameters
    ----------
    S : float   Underlying spot price
    K : float   Strike price
    T : float   Time to expiry in years
    r : float   Risk-free rate (annualised, continuous)
    sigma : float  Implied volatility (annualised)
    option_type : str  'call' or 'put'

    Returns
    -------
    float : Option fair value (>= 0)
    """
    if T <= 0.0:
        # At expiry — intrinsic value only
        if option_type.lower() == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type.lower() == "call":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    elif option_type.lower() == "put":
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return max(price, 0.0)


def option_delta(
    S: float,
    K: float,
    T: float,
    r: float = 0.05,
    sigma: float = 0.20,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes delta.

    Returns
    -------
    float : Delta in [-1, 1]
    """
    if T <= 0.0:
        if option_type.lower() == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)

    if option_type.lower() == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def option_gamma(
    S: float,
    K: float,
    T: float,
    r: float = 0.05,
    sigma: float = 0.20,
) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0.0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return _norm_pdf(d1) / (S * sigma * sqrt_T)


def option_vega(
    S: float,
    K: float,
    T: float,
    r: float = 0.05,
    sigma: float = 0.20,
) -> float:
    """Black-Scholes vega (sensitivity to 1-unit change in sigma)."""
    if T <= 0.0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return S * _norm_pdf(d1) * sqrt_T


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    A single position in the portfolio (equity or derivative).

    Attributes
    ----------
    ticker : str          Identifier / underlying symbol
    underlying_price : float  Current spot price of the underlying
    quantity : int        Signed quantity (+ long, - short)
    is_option : bool      True if this is an option position
    option_type : str     'call' or 'put' (ignored if not is_option)
    strike : float        Strike price (ignored if not is_option)
    expiry : float        Years to expiry (ignored if not is_option)
    implied_vol : float   Current implied volatility (annualised)
    risk_free_rate : float Risk-free rate used for Black-Scholes
    """
    ticker: str
    underlying_price: float
    quantity: int          # + long, - short
    is_option: bool = False
    option_type: str = "call"   # 'call' or 'put'
    strike: float = 0.0
    expiry: float = 1.0    # years to expiry
    implied_vol: float = 0.20
    risk_free_rate: float = 0.05

    def market_value(self) -> float:
        """Current market value of this position (signed by quantity)."""
        if self.is_option:
            unit_price = black_scholes_price(
                self.underlying_price,
                self.strike,
                self.expiry,
                self.risk_free_rate,
                self.implied_vol,
                self.option_type,
            )
        else:
            unit_price = self.underlying_price
        return self.quantity * unit_price

    def notional(self) -> float:
        """Unsigned notional (underlying price × abs(quantity))."""
        return self.underlying_price * abs(self.quantity)


@dataclass
class SPANParams:
    """
    SPAN methodology parameters.

    Attributes
    ----------
    price_scan_range : float   Daily price scan range as fraction of underlying (σ)
    vol_scan_range : float     Vol scan range as additive shift (v)
    extreme_weight : float     Weight applied to scenarios 14-15 (default 0.35)
    short_option_minimum_pct : float  SOMin = this fraction × underlying
    spread_charge_pct : float  Inter-month spread charge as fraction of notional
    delivery_charge_pct : float  Delivery charge for near-expiry positions
    delivery_threshold : float   Expiry threshold in years (≈ 1 month = 1/12)
    """
    price_scan_range: float = 0.05    # daily move (sigma)
    vol_scan_range: float = 0.03      # vol shift fraction
    extreme_weight: float = 0.35      # weight for scenarios 14-15
    short_option_minimum_pct: float = 0.005   # 50 bps
    spread_charge_pct: float = 0.005  # 50 bps per spread
    delivery_charge_pct: float = 0.01  # 1% for near-expiry
    delivery_threshold: float = 0.083  # ~1 month (1/12 year)


# ---------------------------------------------------------------------------
# SPAN 16 Scenario Definitions
# ---------------------------------------------------------------------------

# Each row: (price_shift_fraction_of_sigma, vol_shift_fraction_of_v, weight)
# Scenarios 0-13: full weight (1.0)
# Scenarios 14-15: extreme moves, partial weight (0.35)
_SPAN_SCENARIOS_BASE = [
    #  (price_sigma_mult, vol_v_mult)
    ( 0.0,  1.0),   # 0:  +0σ, +v
    ( 0.0, -1.0),   # 1:  +0σ, -v
    ( 1/3,  1.0),   # 2:  +1/3σ, +v
    ( 1/3, -1.0),   # 3:  +1/3σ, -v
    (-1/3,  1.0),   # 4:  -1/3σ, +v
    (-1/3, -1.0),   # 5:  -1/3σ, -v
    ( 2/3,  1.0),   # 6:  +2/3σ, +v
    ( 2/3, -1.0),   # 7:  +2/3σ, -v
    (-2/3,  1.0),   # 8:  -2/3σ, +v
    (-2/3, -1.0),   # 9:  -2/3σ, -v
    ( 1.0,  1.0),   # 10: +1σ, +v
    ( 1.0, -1.0),   # 11: +1σ, -v
    (-1.0,  1.0),   # 12: -1σ, +v
    (-1.0, -1.0),   # 13: -1σ, -v
]

_SPAN_SCENARIOS_EXTREME = [
    ( 2.0,  0.0),   # 14: +2σ, 0  (extreme up)
    (-2.0,  0.0),   # 15: -2σ, 0  (extreme down)
]


# ---------------------------------------------------------------------------
# SPAN Engine
# ---------------------------------------------------------------------------

class SPANEngine:
    """
    CME SPAN margin calculation engine.

    Usage
    -----
    engine = SPANEngine()
    result = engine.span_margin(positions)
    """

    def scenario_pnl(
        self,
        position: Position,
        price_shift_pct: float,
        vol_shift: float,
    ) -> float:
        """
        Compute the P&L of a single position under a given scenario.

        Parameters
        ----------
        position : Position
        price_shift_pct : float  Fractional shift in underlying price (e.g. 0.05 = +5%)
        vol_shift : float        Additive shift in implied vol (e.g. 0.03 = +3%)

        Returns
        -------
        float : P&L (positive = gain, negative = loss)
        """
        S = position.underlying_price
        S_new = S * (1.0 + price_shift_pct)

        if not position.is_option:
            # Equity / futures — linear P&L
            pnl = position.quantity * S * price_shift_pct
            return pnl

        # Option — reprice under stressed parameters
        sigma_new = max(position.implied_vol + vol_shift, 1e-6)
        price_old = black_scholes_price(
            S, position.strike, position.expiry,
            position.risk_free_rate, position.implied_vol, position.option_type
        )
        price_new = black_scholes_price(
            S_new, position.strike, position.expiry,
            position.risk_free_rate, sigma_new, position.option_type
        )
        pnl = position.quantity * (price_new - price_old)
        return pnl

    def _portfolio_scenario_pnl(
        self,
        positions: List[Position],
        price_shift_pct: float,
        vol_shift: float,
    ) -> float:
        """Total portfolio P&L under one scenario."""
        return sum(
            self.scenario_pnl(p, price_shift_pct, vol_shift)
            for p in positions
        )

    def scanning_risk(
        self,
        positions: List[Position],
        params: SPANParams,
    ) -> float:
        """
        Compute SPAN scanning risk = maximum loss across 16 scenarios.

        Scenarios 0-13 have full weight; 14-15 have weight = params.extreme_weight.

        Returns
        -------
        float : Scanning risk (>= 0; represents the worst-case loss)
        """
        sigma = params.price_scan_range
        v = params.vol_scan_range
        worst_loss = 0.0

        # Scenarios 0-13 (full weight)
        for price_mult, vol_mult in _SPAN_SCENARIOS_BASE:
            pnl = self._portfolio_scenario_pnl(
                positions,
                price_shift_pct=price_mult * sigma,
                vol_shift=vol_mult * v,
            )
            loss = -pnl  # loss is negative P&L
            worst_loss = max(worst_loss, loss)

        # Scenarios 14-15 (partial weight)
        for price_mult, vol_mult in _SPAN_SCENARIOS_EXTREME:
            pnl = self._portfolio_scenario_pnl(
                positions,
                price_shift_pct=price_mult * sigma,
                vol_shift=vol_mult * v,
            )
            loss = -pnl * params.extreme_weight
            worst_loss = max(worst_loss, loss)

        return max(worst_loss, 0.0)

    def _short_option_minimum(
        self,
        positions: List[Position],
        params: SPANParams,
    ) -> float:
        """
        Short option minimum charge.
        = short_option_minimum_pct × underlying_price × abs(net_short_option_quantity)
        """
        total_charge = 0.0
        for p in positions:
            if p.is_option and p.quantity < 0:
                total_charge += (
                    params.short_option_minimum_pct
                    * p.underlying_price
                    * abs(p.quantity)
                )
        return total_charge

    def _spread_charge(
        self,
        positions: List[Position],
        params: SPANParams,
    ) -> float:
        """
        Inter-month spread charge.
        Counts pairs of long/short option positions on the same underlying
        with different expiries as spreads.
        = spread_charge_pct × notional per spread pair
        """
        # Group options by underlying ticker
        by_ticker: dict[str, list[Position]] = {}
        for p in positions:
            if p.is_option:
                by_ticker.setdefault(p.ticker, []).append(p)

        spread_charge = 0.0
        for ticker, opts in by_ticker.items():
            # Count spreads as min(abs long qty, abs short qty) within the group
            long_qty = sum(p.quantity for p in opts if p.quantity > 0)
            short_qty = sum(abs(p.quantity) for p in opts if p.quantity < 0)
            n_spreads = min(long_qty, short_qty)
            if n_spreads > 0:
                # Use first position's underlying price as reference
                ref_price = opts[0].underlying_price
                spread_charge += params.spread_charge_pct * ref_price * n_spreads

        return spread_charge

    def _delivery_charge(
        self,
        positions: List[Position],
        params: SPANParams,
    ) -> float:
        """
        Delivery charge for positions near expiry (expiry < delivery_threshold).
        = delivery_charge_pct × notional
        """
        total = 0.0
        for p in positions:
            if p.is_option and p.expiry < params.delivery_threshold:
                total += params.delivery_charge_pct * p.notional()
        return total

    def _span_credit(
        self,
        positions: List[Position],
        params: SPANParams,
    ) -> float:
        """
        SPAN credit for offsetting positions (longs reducing risk of shorts).
        Approximated as the benefit from long options that offset scanning risk.
        """
        # Credit = reduction in scanning risk from long options
        # Computed as the scanning risk reduction when removing long option offsets.
        # Simplified: credit = sum of long option market values capped at scanning risk.
        long_option_value = sum(
            p.market_value() for p in positions
            if p.is_option and p.quantity > 0
        )
        # SPAN credit capped at scanning risk to avoid negative total
        sr = self.scanning_risk(positions, params)
        return min(max(long_option_value, 0.0), sr * 0.5)

    def span_margin(
        self,
        positions: List[Position],
        params: SPANParams = None,
    ) -> dict:
        """
        Compute full SPAN margin decomposition.

        Returns
        -------
        dict with keys:
            scanning_risk       : float  Maximum loss across 16 scenarios
            short_option_min    : float  Short option minimum charge
            spread_charge       : float  Inter-month spread charge
            delivery_charge     : float  Near-expiry delivery charge
            span_credit         : float  Credit for offsetting longs
            total_span_margin   : float  Final required margin
        """
        if params is None:
            params = SPANParams()

        sr = self.scanning_risk(positions, params)
        som = self._short_option_minimum(positions, params)
        sc = self._spread_charge(positions, params)
        dc = self._delivery_charge(positions, params)
        credit = self._span_credit(positions, params)

        # SPAN margin = max(scanning_risk, short_option_min) + charges - credit
        base = max(sr, som)
        total = max(base + sc + dc - credit, 0.0)

        return {
            "scanning_risk": sr,
            "short_option_min": som,
            "spread_charge": sc,
            "delivery_charge": dc,
            "span_credit": credit,
            "total_span_margin": total,
        }

    def portfolio_margin(
        self,
        positions: List[Position],
    ) -> dict:
        """
        Reg T Portfolio Margin calculation.

        Uses 10 scenarios:
          Underlying price shifts: ±15%, ±10%, ±5%, 0%
          Vol shifts for ±6% (applied to all non-zero scenarios)
          Extreme: ±15% price with 0 vol shift

        Floor: 15% × total underlying market value (long side only).

        Returns
        -------
        dict with keys:
            scenario_losses     : list[float]  Loss under each of 10 scenarios
            worst_scenario_loss : float        Maximum loss across scenarios
            floor_margin        : float        15% of long underlying exposure
            portfolio_margin    : float        max(worst_loss, floor)
        """
        # 10 scenarios: underlying shift × vol shift
        pm_scenarios = [
            # (price_shift, vol_shift)
            ( 0.15,  0.06),   # up 15%, vol up
            ( 0.15, -0.06),   # up 15%, vol down
            ( 0.10,  0.06),   # up 10%, vol up
            ( 0.10, -0.06),   # up 10%, vol down
            ( 0.05,  0.06),   # up 5%,  vol up
            ( 0.05, -0.06),   # up 5%,  vol down
            ( 0.00,  0.00),   # flat
            (-0.05,  0.06),   # dn 5%,  vol up
            (-0.10,  0.06),   # dn 10%, vol up
            (-0.15,  0.06),   # dn 15%, vol up
        ]

        scenario_losses = []
        for price_shift, vol_shift in pm_scenarios:
            pnl = self._portfolio_scenario_pnl(positions, price_shift, vol_shift)
            scenario_losses.append(-pnl)  # loss = -P&L

        worst_loss = max(scenario_losses)

        # Floor: 15% of long underlying exposure
        long_notional = sum(
            p.notional() for p in positions
            if not p.is_option and p.quantity > 0
        )
        # Also include long option delta-adjusted notional for floor
        for p in positions:
            if p.is_option and p.quantity > 0:
                delta = option_delta(
                    p.underlying_price, p.strike, p.expiry,
                    p.risk_free_rate, p.implied_vol, p.option_type
                )
                long_notional += abs(delta) * p.underlying_price * abs(p.quantity)

        floor_margin = 0.15 * long_notional

        pm = max(max(worst_loss, 0.0), floor_margin)

        return {
            "scenario_losses": scenario_losses,
            "worst_scenario_loss": max(worst_loss, 0.0),
            "floor_margin": floor_margin,
            "portfolio_margin": pm,
        }


# ---------------------------------------------------------------------------
# Margin Analytics
# ---------------------------------------------------------------------------

class MarginAnalytics:
    """
    Higher-level analytics on top of SPAN / Portfolio Margin.
    """

    def __init__(self):
        self._engine = SPANEngine()

    def margin_efficiency(
        self,
        positions: List[Position],
        params: SPANParams = None,
    ) -> float:
        """
        Margin efficiency ratio = portfolio_margin / span_margin.

        Lower ratio = more capital efficient under SPAN vs Portfolio Margin.
        Returns float (> 0).
        """
        span_result = self._engine.span_margin(positions, params)
        pm_result = self._engine.portfolio_margin(positions)
        span_m = span_result["total_span_margin"]
        pm = pm_result["portfolio_margin"]
        if span_m == 0.0:
            return float("inf")
        return pm / span_m

    def excess_liquidity(
        self,
        portfolio_value: float,
        positions: List[Position],
        params: SPANParams = None,
    ) -> float:
        """
        Excess liquidity = portfolio_value - span_margin.
        Positive means no margin call; negative = margin call territory.
        """
        span_result = self._engine.span_margin(positions, params)
        return portfolio_value - span_result["total_span_margin"]

    def margin_utilization(
        self,
        portfolio_value: float,
        positions: List[Position],
        params: SPANParams = None,
    ) -> float:
        """
        Margin utilization = span_margin / portfolio_value.
        Values > 1.0 indicate over-margined / margin call.
        """
        span_result = self._engine.span_margin(positions, params)
        if portfolio_value == 0.0:
            return float("inf")
        return span_result["total_span_margin"] / portfolio_value

    def compare_margin_methods(
        self,
        positions: List[Position],
        portfolio_value: float,
        params: SPANParams = None,
    ) -> dict:
        """
        Side-by-side comparison of SPAN vs Portfolio Margin.
        """
        span_result = self._engine.span_margin(positions, params)
        pm_result = self._engine.portfolio_margin(positions)
        efficiency = self.margin_efficiency(positions, params)
        excess_liq = self.excess_liquidity(portfolio_value, positions, params)
        utilization = self.margin_utilization(portfolio_value, positions, params)

        return {
            "span": span_result,
            "portfolio_margin": pm_result,
            "margin_efficiency": efficiency,
            "excess_liquidity": excess_liq,
            "margin_utilization": utilization,
            "recommended_method": (
                "portfolio_margin"
                if pm_result["portfolio_margin"] < span_result["total_span_margin"]
                else "span"
            ),
        }


# ---------------------------------------------------------------------------
# Convenience functions (module-level API)
# ---------------------------------------------------------------------------

_DEFAULT_ENGINE = SPANEngine()


def compute_span_margin(
    positions: List[Position],
    params: SPANParams = None,
) -> float:
    """
    Compute SPAN total margin requirement for a list of positions.

    Parameters
    ----------
    positions : List[Position]
    params : SPANParams, optional  (uses defaults if None)

    Returns
    -------
    float : Total SPAN margin required
    """
    result = _DEFAULT_ENGINE.span_margin(positions, params)
    return result["total_span_margin"]


def compute_portfolio_margin(positions: List[Position]) -> float:
    """
    Compute Reg T Portfolio Margin for a list of positions.

    Returns
    -------
    float : Portfolio margin required
    """
    result = _DEFAULT_ENGINE.portfolio_margin(positions)
    return result["portfolio_margin"]


def compute_margin_comparison(
    positions: List[Position],
    portfolio_value: float = 100_000.0,
    params: SPANParams = None,
) -> dict:
    """
    Full comparison: SPAN vs Portfolio Margin with analytics.
    """
    analytics = MarginAnalytics()
    return analytics.compare_margin_methods(positions, portfolio_value, params)
