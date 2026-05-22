"""
SENTINEL SBX — Tail Risk Hedging Analytics v3
dim_145: Tail risk hedging (VIX options / put spreads / hedge effectiveness)

Comprehensive tail risk hedging analytics covering:
  - VIX options-based tail hedges (crisis payoff, optimal allocation)
  - Put spread construction (protective put, bull/bear put spread, ratio spread)
  - Hedge effectiveness measurement (R², VaR/CVaR reduction, cost efficiency)
  - Conditional Value at Risk (CVaR / Expected Shortfall)
  - Crisis scenario simulation (2008 GFC, 2020 COVID crash)
  - Optimal hedge ratio via minimum variance

Math: numpy / scipy only.  No external network calls.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Public re-exports (satisfy capability-test imports)
# ---------------------------------------------------------------------------

__all__ = [
    # dataclasses
    "HedgeInstrument",
    "TailRiskMetrics",
    # classes
    "PutSpreadStrategy",
    "VIXHedge",
    "HedgeEffectiveness",
    # module-level helpers
    "compute_var",
    "compute_cvar",
    "bs_put_price",
    "optimal_hedge_ratio",
    # legacy aliases expected by some tests
    "TailRiskHedge",
    "VIXOptionHedge",
    "PutSpreadHedge",
    "compute_hedge_cost",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
VIX_MULTIPLIER = 1000  # VIX options: cash-settled, $1 000 per point


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price.

    Args:
        S: Underlying spot price
        K: Strike price
        T: Time to expiry in years (must be > 0)
        r: Continuously compounded risk-free rate
        sigma: Annualised implied volatility

    Returns:
        Put option fair value.  Returns intrinsic value when T is tiny.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    put = (K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return max(put, 0.0)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return max(call, 0.0)


# ---------------------------------------------------------------------------
# Module-level VaR / CVaR helpers
# ---------------------------------------------------------------------------

def compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical Value at Risk (positive number = loss magnitude).

    VaR_alpha = -quantile(returns, 1 - confidence)

    Args:
        returns: Array of P&L returns (daily, e.g. -0.02 = -2%)
        confidence: Confidence level (0.95 → 95% VaR)

    Returns:
        VaR expressed as a positive loss magnitude.
    """
    returns = np.asarray(returns, dtype=float)
    q = np.percentile(returns, (1.0 - confidence) * 100.0)
    return float(-q)


def compute_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional Value at Risk / Expected Shortfall.

    CVaR_alpha = -E[R | R <= -VaR_alpha]

    Args:
        returns: Array of P&L returns
        confidence: Confidence level (0.95 → 95% CVaR)

    Returns:
        CVaR expressed as a positive loss magnitude (always >= VaR).
    """
    returns = np.asarray(returns, dtype=float)
    var = compute_var(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


def optimal_hedge_ratio(
    portfolio_returns: np.ndarray,
    hedge_returns: np.ndarray,
) -> float:
    """Minimum-variance optimal hedge ratio (OLS beta convention).

    h* = Cov(portfolio, hedge) / Var(hedge)

    This is the OLS regression slope of portfolio returns on hedge returns.
    A negative result indicates an inverse relationship — the hedge instrument
    moves in the opposite direction to the portfolio (e.g. puts, VIX calls).
    Negative h* means: "hold the hedge instrument short" in the futures sense,
    or equivalently the hedge instrument naturally offsets portfolio losses.

    Returns:
        Hedge ratio h* (negative when hedge inversely correlated with portfolio).
    """
    p = np.asarray(portfolio_returns, dtype=float)
    h = np.asarray(hedge_returns, dtype=float)
    cov_ph = float(np.cov(p, h)[0, 1])
    var_h = float(np.var(h, ddof=1))
    if var_h == 0.0:
        return 0.0
    return cov_ph / var_h


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HedgeInstrument:
    """Specification of a tail hedge position.

    Attributes:
        instrument_type: One of 'put', 'put_spread', 'vix_call', 'variance_swap'
        cost: Total premium paid (positive = debit)
        notional: Notional portfolio value being hedged
        payoff_params: Instrument-specific parameters (strike, expiry, etc.)
    """
    instrument_type: str          # 'put' | 'put_spread' | 'vix_call' | 'variance_swap'
    cost: float                   # total premium paid
    notional: float
    payoff_params: dict = field(default_factory=dict)

    def payoff_at_expiry(self, spot: float) -> float:
        """Calculate instrument payoff at expiry given underlying spot/level."""
        p = self.payoff_params
        if self.instrument_type == "put":
            K = p.get("strike", 0.0)
            return max(K - spot, 0.0)
        elif self.instrument_type == "put_spread":
            K_long = p.get("K_long", 0.0)
            K_short = p.get("K_short", 0.0)
            return max(K_long - spot, 0.0) - max(K_short - spot, 0.0)
        elif self.instrument_type == "vix_call":
            K_vix = p.get("strike", 0.0)
            mult = p.get("multiplier", VIX_MULTIPLIER)
            return max(spot - K_vix, 0.0) * mult
        elif self.instrument_type == "variance_swap":
            realized_var = p.get("realized_variance", 0.0)
            strike_var = p.get("strike_variance", 0.0)
            notional_vega = p.get("vega_notional", 1.0)
            return (realized_var - strike_var) * notional_vega
        return 0.0


@dataclass
class TailRiskMetrics:
    """Consolidated tail risk statistics for a return series.

    Attributes:
        var_95:  95% VaR (positive = loss)
        var_99:  99% VaR
        cvar_95: 95% CVaR / Expected Shortfall
        cvar_99: 99% CVaR
        max_drawdown: Maximum peak-to-trough drawdown
        skewness: Third standardised moment (negative = left-skewed)
        excess_kurtosis: Fourth standardised moment minus 3 (fat-tailed > 0)
    """
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    skewness: float
    excess_kurtosis: float


# ---------------------------------------------------------------------------
# Put Spread Strategy
# ---------------------------------------------------------------------------

class PutSpreadStrategy:
    """Black-Scholes put option and spread construction.

    Args:
        S: Current underlying price (spot)
        r: Risk-free rate (annualised, continuously compounded); default 0.05
    """

    def __init__(self, S: float, r: float = 0.05) -> None:
        self.S = float(S)
        self.r = float(r)

    # ------------------------------------------------------------------
    # Protective Put
    # ------------------------------------------------------------------

    def protective_put(self, K: float, T: float, sigma: float) -> dict:
        """Long underlying + long put (floor strategy).

        Args:
            K: Put strike price
            T: Time to expiry (years)
            sigma: Implied volatility

        Returns:
            dict with keys:
                premium         — put option cost (per share)
                breakeven       — price at which position breaks even at expiry
                max_loss        — maximum loss (always = S - K + premium, capped)
                unlimited_upside — True (underlying upside remains)
                payoff_at_K     — payoff when spot = K at expiry (net of premium)
        """
        premium = bs_put_price(self.S, K, T, self.r, sigma)
        breakeven = self.S + premium          # must rise by premium cost to profit
        max_loss = (self.S - K) + premium     # if stock falls to K or below
        return {
            "premium": premium,
            "breakeven": breakeven,
            "max_loss": max(max_loss, 0.0),
            "unlimited_upside": True,
            "payoff_at_K": -premium,          # net at K: put pays 0, equity flat vs S
        }

    # ------------------------------------------------------------------
    # Bull Put Spread (net credit)
    # ------------------------------------------------------------------

    def bull_put_spread(self, K_long: float, K_short: float,
                        T: float, sigma: float) -> dict:
        """Sell higher-strike put, buy lower-strike put (bullish).

        K_long < K_short — collect net credit if market stays above K_short.

        Args:
            K_long:  Lower strike (long put — protection)
            K_short: Higher strike (short put — premium collected)
            T:       Time to expiry (years)
            sigma:   Implied volatility

        Returns:
            dict with keys:
                premium_long, premium_short, net_credit,
                max_profit, max_loss, breakeven
        """
        if K_long >= K_short:
            raise ValueError("bull_put_spread: K_long must be < K_short")
        p_long = bs_put_price(self.S, K_long, T, self.r, sigma)
        p_short = bs_put_price(self.S, K_short, T, self.r, sigma)
        net_credit = p_short - p_long         # received upfront
        max_profit = net_credit               # both puts expire OTM
        max_loss = (K_short - K_long) - net_credit
        breakeven = K_short - net_credit      # spot where spread P&L = 0
        return {
            "premium_long": p_long,
            "premium_short": p_short,
            "net_credit": net_credit,
            "max_profit": max_profit,
            "max_loss": max(max_loss, 0.0),
            "breakeven": breakeven,
        }

    # ------------------------------------------------------------------
    # Bear Put Spread (net debit)
    # ------------------------------------------------------------------

    def bear_put_spread(self, K_long: float, K_short: float,
                        T: float, sigma: float) -> dict:
        """Buy higher-strike put, sell lower-strike put (bearish / protective).

        K_short < K_long — cheaper than outright protective put.

        Args:
            K_long:  Higher strike (long put — main protection)
            K_short: Lower strike (short put — partially funded)
            T:       Time to expiry (years)
            sigma:   Implied volatility

        Returns:
            dict with keys:
                premium_long, premium_short, net_debit,
                max_profit, max_loss, breakeven
        """
        if K_short >= K_long:
            raise ValueError("bear_put_spread: K_short must be < K_long")
        p_long = bs_put_price(self.S, K_long, T, self.r, sigma)
        p_short = bs_put_price(self.S, K_short, T, self.r, sigma)
        net_debit = p_long - p_short          # cost paid upfront
        max_profit = (K_long - K_short) - net_debit
        max_loss = net_debit
        breakeven = K_long - net_debit
        return {
            "premium_long": p_long,
            "premium_short": p_short,
            "net_debit": net_debit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": breakeven,
        }

    # ------------------------------------------------------------------
    # Ratio Put Spread
    # ------------------------------------------------------------------

    def ratio_put_spread(self, K_long: float, K_short: float,
                         ratio: int, T: float, sigma: float) -> dict:
        """Buy 1 put at K_long, sell `ratio` puts at K_short.

        K_short < K_long (typically OTM vs further OTM).

        Args:
            K_long:  Higher strike (buy 1 put)
            K_short: Lower strike (sell `ratio` puts)
            ratio:   Number of puts sold per long put (typically 2)
            T:       Time to expiry (years)
            sigma:   Implied volatility

        Returns:
            dict with keys:
                premium_long, premium_short_each, net_cost,
                max_profit_zone, max_loss_below_K_short, breakeven_down,
                ratio
        """
        if K_short >= K_long:
            raise ValueError("ratio_put_spread: K_short must be < K_long")
        p_long = bs_put_price(self.S, K_long, T, self.r, sigma)
        p_short = bs_put_price(self.S, K_short, T, self.r, sigma)
        net_cost = p_long - ratio * p_short   # negative = net credit
        # Max profit: at K_short all puts settle ITM
        max_profit = (K_long - K_short) + (ratio - 1) * p_short - p_long
        # Below K_short: additional short puts cause unlimited (well, very large) loss
        # Loss = net_cost + (ratio-1)*(K_short - spot) as spot -> 0
        # Simplified: max_loss_below expressed at spot=0
        max_loss_below_K_short = net_cost + (ratio - 1) * K_short
        # Lower breakeven: K_short - max_profit / (ratio - 1)  [for ratio > 1]
        if ratio > 1:
            breakeven_down = K_short - max_profit / (ratio - 1)
        else:
            breakeven_down = None
        return {
            "premium_long": p_long,
            "premium_short_each": p_short,
            "net_cost": net_cost,
            "max_profit_zone": (K_short, K_long),
            "max_loss_below_K_short": max_loss_below_K_short,
            "breakeven_down": breakeven_down,
            "ratio": ratio,
        }


# ---------------------------------------------------------------------------
# VIX Hedge
# ---------------------------------------------------------------------------

class VIXHedge:
    """VIX options-based tail hedge analytics.

    VIX options are cash-settled with a multiplier of $1 000 per VIX point.
    Long VIX calls profit when volatility spikes during a crisis.
    """

    def __init__(self, multiplier: int = VIX_MULTIPLIER) -> None:
        self.multiplier = int(multiplier)

    def vix_call_payoff(self, vix_at_expiry: float, strike: float,
                        multiplier: Optional[int] = None) -> float:
        """Cash payoff of a long VIX call option at expiry.

        Args:
            vix_at_expiry: VIX settlement level
            strike:        Call strike (VIX points)
            multiplier:    Contract multiplier (default $1 000)

        Returns:
            Dollar payoff per contract (positive or zero).
        """
        mult = multiplier if multiplier is not None else self.multiplier
        return max(vix_at_expiry - strike, 0.0) * mult

    def crisis_pnl(
        self,
        portfolio_loss_pct: float,
        vix_spike: float,
        vix_strike: float,
        vix_premium: float,
    ) -> dict:
        """Net P&L during a crisis event (per unit notional).

        Args:
            portfolio_loss_pct: Portfolio loss (e.g. -0.30 for -30%)
            vix_spike:          VIX level at crisis peak
            vix_strike:         VIX call strike purchased
            vix_premium:        Premium paid for VIX calls (dollar amount)

        Returns:
            dict:
                portfolio_pnl   — dollar loss on portfolio
                hedge_payoff    — gross VIX call payoff
                hedge_pnl       — net hedge P&L (payoff - premium)
                net_pnl         — portfolio + hedge combined
                protection_ratio — hedge_pnl / |portfolio_pnl| (fraction offset)
        """
        payoff = self.vix_call_payoff(vix_spike, vix_strike)
        hedge_pnl = payoff - vix_premium
        net_pnl = portfolio_loss_pct + hedge_pnl   # portfolio_loss is negative
        protection_ratio = (
            hedge_pnl / abs(portfolio_loss_pct) if portfolio_loss_pct != 0 else 0.0
        )
        return {
            "portfolio_pnl": portfolio_loss_pct,
            "hedge_payoff": payoff,
            "hedge_pnl": hedge_pnl,
            "net_pnl": net_pnl,
            "protection_ratio": protection_ratio,
        }

    def optimal_vix_allocation(
        self,
        portfolio_vol: float,
        vix_vol: float,
        correlation: float,
        portfolio_size: float,
    ) -> float:
        """Minimum-variance optimal dollar allocation to VIX calls.

        h* = -Cov(portfolio, hedge) / Var(hedge)
           = -(rho * sigma_p * sigma_h) / sigma_h^2
           = -(rho * sigma_p) / sigma_h

        Dollar hedge = h* * portfolio_size

        Args:
            portfolio_vol:  Annualised portfolio volatility (e.g. 0.15)
            vix_vol:        Annualised VIX return volatility (e.g. 1.0)
            correlation:    Correlation between portfolio and VIX (usually negative)
            portfolio_size: Dollar value of portfolio

        Returns:
            Optimal dollar notional to allocate to VIX calls (positive = long).
        """
        if vix_vol == 0:
            return 0.0
        h_star = -(correlation * portfolio_vol) / vix_vol
        return h_star * portfolio_size


# ---------------------------------------------------------------------------
# Hedge Effectiveness
# ---------------------------------------------------------------------------

class HedgeEffectiveness:
    """Measure and quantify the quality of a tail risk hedge."""

    # ------------------------------------------------------------------
    # Risk measures
    # ------------------------------------------------------------------

    def compute_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Historical VaR (positive = loss magnitude)."""
        return compute_var(returns, confidence)

    def compute_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """CVaR / Expected Shortfall (positive = loss magnitude)."""
        return compute_cvar(returns, confidence)

    def tail_risk_metrics(self, returns: np.ndarray) -> TailRiskMetrics:
        """Comprehensive tail risk statistics.

        Args:
            returns: Daily return series

        Returns:
            TailRiskMetrics dataclass with VaR, CVaR, drawdown, moments.
        """
        r = np.asarray(returns, dtype=float)
        var_95 = compute_var(r, 0.95)
        var_99 = compute_var(r, 0.99)
        cvar_95 = compute_cvar(r, 0.95)
        cvar_99 = compute_cvar(r, 0.99)

        # Maximum drawdown (peak-to-trough on cumulative returns)
        cum = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_drawdown = float(-dd.min())

        # Moments
        mean = float(r.mean())
        std = float(r.std(ddof=1))
        if std == 0:
            skewness = 0.0
            excess_kurtosis = 0.0
        else:
            skewness = float(np.mean(((r - mean) / std) ** 3))
            excess_kurtosis = float(np.mean(((r - mean) / std) ** 4)) - 3.0

        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
        )

    # ------------------------------------------------------------------
    # Hedge ratio
    # ------------------------------------------------------------------

    def hedge_ratio(
        self,
        portfolio_returns: np.ndarray,
        hedge_returns: np.ndarray,
    ) -> float:
        """Minimum-variance optimal hedge ratio h*.

        h* = -Cov(portfolio, hedge) / Var(hedge)

        Negative result → hedge instrument moves inversely to portfolio.
        """
        return optimal_hedge_ratio(portfolio_returns, hedge_returns)

    # ------------------------------------------------------------------
    # Hedge effectiveness metrics
    # ------------------------------------------------------------------

    def hedge_effectiveness(
        self,
        portfolio_returns: np.ndarray,
        hedge_returns: np.ndarray,
        hedge_cost: float,
    ) -> dict:
        """Quantify how well the hedge reduces tail risk.

        Args:
            portfolio_returns: Unhedged portfolio daily returns
            hedge_returns:     Hedge instrument daily returns
            hedge_cost:        Total cost of the hedge (positive = debit paid)

        Returns:
            dict:
                r_squared              — proportion of portfolio variance explained
                hedge_ratio            — optimal h*
                hedged_variance        — variance of hedged portfolio
                unhedged_variance      — variance of unhedged portfolio
                var_reduction_95       — VaR reduction at 95%
                var_reduction_99       — VaR reduction at 99%
                cvar_reduction_95      — CVaR reduction at 95%
                cvar_reduction_99      — CVaR reduction at 99%
                cost_per_unit_protection — hedge_cost / cvar_reduction_95
        """
        p = np.asarray(portfolio_returns, dtype=float)
        h = np.asarray(hedge_returns, dtype=float)

        # h_star = Cov/Var (OLS beta convention — negative for inverse hedges)
        h_star = optimal_hedge_ratio(p, h)
        # Minimum-variance hedged portfolio: subtract the beta-weighted hedge
        # (h_star is negative for inverse hedges, so subtracting amplifies protection)
        hedged = p - h_star * h

        var_p = float(np.var(p, ddof=1))
        var_hed = float(np.var(hedged, ddof=1))
        r_squared = max(0.0, 1.0 - var_hed / var_p) if var_p > 0 else 0.0

        # VaR / CVaR comparison
        var95_unhd = compute_var(p, 0.95)
        var99_unhd = compute_var(p, 0.99)
        cvar95_unhd = compute_cvar(p, 0.95)
        cvar99_unhd = compute_cvar(p, 0.99)

        var95_hed = compute_var(hedged, 0.95)
        var99_hed = compute_var(hedged, 0.99)
        cvar95_hed = compute_cvar(hedged, 0.95)
        cvar99_hed = compute_cvar(hedged, 0.99)

        cvar95_reduction = cvar95_unhd - cvar95_hed
        cost_per_unit = (
            hedge_cost / cvar95_reduction if cvar95_reduction > 0 else float("inf")
        )

        return {
            "r_squared": r_squared,
            "hedge_ratio": h_star,
            "hedged_variance": var_hed,
            "unhedged_variance": var_p,
            "var_reduction_95": var95_unhd - var95_hed,
            "var_reduction_99": var99_unhd - var99_hed,
            "cvar_reduction_95": cvar95_reduction,
            "cvar_reduction_99": cvar99_unhd - cvar99_hed,
            "cost_per_unit_protection": cost_per_unit,
        }

    # ------------------------------------------------------------------
    # Crisis scenario
    # ------------------------------------------------------------------

    def crisis_scenario(
        self,
        portfolio_size: float,
        crash_pct: float,
        hedge: HedgeInstrument,
    ) -> dict:
        """Evaluate portfolio + hedge outcome under a crash scenario.

        Args:
            portfolio_size: Dollar value of portfolio
            crash_pct:      Fractional decline (e.g. -0.30 for -30%)
            hedge:          HedgeInstrument instance

        Returns:
            dict:
                portfolio_loss      — dollar loss (negative)
                hedge_payoff        — dollar payoff from hedge
                hedge_pnl           — net hedge P&L (payoff - cost)
                net_loss            — combined net loss
                protection_pct      — fraction of loss recovered by hedge
                cost_of_carry       — hedge cost as % of portfolio
        """
        portfolio_loss = portfolio_size * crash_pct  # e.g. -300_000

        # Determine terminal spot / VIX level for payoff
        p = hedge.payoff_params
        if hedge.instrument_type == "vix_call":
            spot_at_expiry = p.get("vix_at_crisis", p.get("strike", 25.0) * 2.0)
        else:
            spot_at_expiry = portfolio_size * (1.0 + crash_pct) / portfolio_size * p.get("strike", 100.0)
            # Simpler: use crash_pct to move initial spot S
            S0 = p.get("S", portfolio_size)
            spot_at_expiry = S0 * (1.0 + crash_pct)

        hedge_payoff = hedge.payoff_at_expiry(spot_at_expiry)
        hedge_pnl = hedge_payoff - hedge.cost
        net_loss = portfolio_loss + hedge_pnl
        protection_pct = (
            hedge_pnl / abs(portfolio_loss) if portfolio_loss != 0 else 0.0
        )
        cost_of_carry = hedge.cost / portfolio_size

        return {
            "portfolio_loss": portfolio_loss,
            "hedge_payoff": hedge_payoff,
            "hedge_pnl": hedge_pnl,
            "net_loss": net_loss,
            "protection_pct": protection_pct,
            "cost_of_carry": cost_of_carry,
        }


# ---------------------------------------------------------------------------
# Pre-built Crisis Scenarios
# ---------------------------------------------------------------------------

CRISIS_SCENARIOS = {
    "2008_GFC": {
        "name": "2008 Global Financial Crisis",
        "equity_crash": -0.565,   # S&P 500 peak-to-trough
        "vix_peak": 89.5,
        "duration_days": 128,
        "credit_spread_widening_bps": 500,
    },
    "2020_COVID": {
        "name": "2020 COVID-19 Market Crash",
        "equity_crash": -0.338,   # S&P 500 Feb-Mar 2020
        "vix_peak": 82.7,
        "duration_days": 23,
        "credit_spread_widening_bps": 350,
    },
    "2000_DOTCOM": {
        "name": "2000-2002 Dot-com Bust",
        "equity_crash": -0.491,
        "vix_peak": 45.1,
        "duration_days": 919,
        "credit_spread_widening_bps": 200,
    },
}


def run_crisis_analysis(
    portfolio_size: float,
    hedge: HedgeInstrument,
    scenario_key: str = "2008_GFC",
) -> dict:
    """Run a named historical crisis scenario through the hedge model.

    Args:
        portfolio_size: Dollar value of portfolio
        hedge:          HedgeInstrument to evaluate
        scenario_key:   Key from CRISIS_SCENARIOS

    Returns:
        Combined scenario + effectiveness dict.
    """
    if scenario_key not in CRISIS_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_key}. "
                         f"Available: {list(CRISIS_SCENARIOS.keys())}")
    scenario = CRISIS_SCENARIOS[scenario_key]
    engine = HedgeEffectiveness()
    result = engine.crisis_scenario(
        portfolio_size=portfolio_size,
        crash_pct=scenario["equity_crash"],
        hedge=hedge,
    )
    result.update({
        "scenario_name": scenario["name"],
        "vix_peak": scenario["vix_peak"],
    })
    return result


# ---------------------------------------------------------------------------
# Legacy compatibility aliases (used by existing dim_145 test imports)
# ---------------------------------------------------------------------------

class TailRiskHedge:
    """Legacy alias — wraps HedgeEffectiveness + crisis scenario utilities."""

    def __init__(self) -> None:
        self._eff = HedgeEffectiveness()
        self._vix = VIXHedge()

    def tail_risk_metrics(self, returns: np.ndarray) -> TailRiskMetrics:
        return self._eff.tail_risk_metrics(returns)

    def hedge_effectiveness(self, portfolio_returns, hedge_returns, hedge_cost=0.0):
        return self._eff.hedge_effectiveness(portfolio_returns, hedge_returns, hedge_cost)

    def crisis_scenario(self, portfolio_size, crash_pct, hedge):
        return self._eff.crisis_scenario(portfolio_size, crash_pct, hedge)

    def var(self, returns, confidence=0.95):
        return compute_var(returns, confidence)

    def cvar(self, returns, confidence=0.95):
        return compute_cvar(returns, confidence)


class VIXOptionHedge:
    """Legacy alias — wraps VIXHedge with direct method access."""

    def __init__(self, multiplier: int = VIX_MULTIPLIER) -> None:
        self._hedge = VIXHedge(multiplier)

    def vix_call_payoff(self, vix_at_expiry, strike, multiplier=None):
        return self._hedge.vix_call_payoff(vix_at_expiry, strike, multiplier)

    def crisis_pnl(self, portfolio_loss_pct, vix_spike, vix_strike, vix_premium):
        return self._hedge.crisis_pnl(
            portfolio_loss_pct, vix_spike, vix_strike, vix_premium
        )

    def optimal_allocation(self, portfolio_vol, vix_vol, correlation, portfolio_size):
        return self._hedge.optimal_vix_allocation(
            portfolio_vol, vix_vol, correlation, portfolio_size
        )


class PutSpreadHedge:
    """Legacy alias — wraps PutSpreadStrategy."""

    def __init__(self, S: float, r: float = 0.05) -> None:
        self._strat = PutSpreadStrategy(S, r)

    def protective_put(self, K, T, sigma):
        return self._strat.protective_put(K, T, sigma)

    def bear_put_spread(self, K_long, K_short, T, sigma):
        return self._strat.bear_put_spread(K_long, K_short, T, sigma)

    def bull_put_spread(self, K_long, K_short, T, sigma):
        return self._strat.bull_put_spread(K_long, K_short, T, sigma)

    def ratio_put_spread(self, K_long, K_short, ratio, T, sigma):
        return self._strat.ratio_put_spread(K_long, K_short, ratio, T, sigma)


def compute_hedge_cost(
    hedge: HedgeInstrument,
    portfolio_size: float,
    annualised: bool = False,
    T: float = 0.25,
) -> dict:
    """Summarise hedge cost metrics.

    Args:
        hedge:          HedgeInstrument
        portfolio_size: Portfolio dollar value
        annualised:     Whether to annualise the cost
        T:              Time horizon in years (for annualisation)

    Returns:
        dict: cost, cost_as_pct_portfolio, annualised_cost_pct
    """
    cost_pct = hedge.cost / portfolio_size if portfolio_size > 0 else 0.0
    ann_cost_pct = cost_pct / T if (annualised and T > 0) else cost_pct
    return {
        "cost": hedge.cost,
        "cost_as_pct_portfolio": cost_pct,
        "annualised_cost_pct": ann_cost_pct,
        "instrument_type": hedge.instrument_type,
        "notional": hedge.notional,
    }
