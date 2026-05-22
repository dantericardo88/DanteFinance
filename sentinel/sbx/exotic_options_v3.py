"""Exotic options pricing engine — barrier, Asian, lookback, digital, chooser.

Implementations:
- Barrier options: Rubinstein-Reiner (1991) closed-form for standard 4 types
- Asian geometric: closed-form via adjusted BS parameters
- Asian arithmetic: Monte Carlo simulation
- Lookback floating: Goldman-Sosin-Gatto (1979) closed-form
- Lookback fixed-strike: Monte Carlo
- Digital (binary): cash-or-nothing and asset-or-nothing closed-form
- Chooser options: closed-form decomposition
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy.stats import norm

__all__ = [
    # Dataclass pricers
    "BarrierOption",
    "AsianOption",
    "LookbackOption",
    # Convenience class
    "ExoticPricer",
    # Module-level functions
    "price_barrier",
    "price_asian_geometric",
    "price_asian_mc",
    "price_lookback_mc",
    "price_digital",
    "price_exotic",
    # BS vanilla (used internally + exposed)
    "bs_call",
    "bs_put",
]

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(d1_val: float, sigma: float, T: float) -> float:
    return d1_val - sigma * math.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European call price with continuous dividend yield q."""
    if T <= 0:
        return max(S * math.exp(-q * T) - K, 0.0)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(d1, sigma, T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European put price with continuous dividend yield q."""
    if T <= 0:
        return max(K - S * math.exp(-q * T), 0.0)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(d1, sigma, T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Barrier option pricing — Rubinstein & Reiner (1991)
# ---------------------------------------------------------------------------

def _barrier_mu_lambda(r: float, q: float, sigma: float) -> tuple[float, float]:
    """Compute mu and lambda for barrier option formulas."""
    mu = (r - q - 0.5 * sigma ** 2) / (sigma ** 2)
    lam = math.sqrt(mu ** 2 + 2.0 * r / sigma ** 2)
    return mu, lam


def _barrier_A(S, K, H, T, r, q, sigma, phi, eta) -> float:
    """Term A for Rubinstein-Reiner barrier formula."""
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(d1, sigma, T)
    return (phi * S * math.exp(-q * T) * norm.cdf(phi * d1)
            - phi * K * math.exp(-r * T) * norm.cdf(phi * d2))


def _barrier_B(S, K, H, T, r, q, sigma, phi, eta) -> float:
    """Term B — same as A but with H^2/S replacing S."""
    x1 = (math.log(H ** 2 / (S * K)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    x2 = x1 - sigma * math.sqrt(T)
    mu = (r - q - 0.5 * sigma ** 2) / sigma ** 2
    return (phi * (H / S) ** (2 * (mu + 1)) * S * math.exp(-q * T) * norm.cdf(phi * x1)
            - phi * (H / S) ** (2 * mu) * K * math.exp(-r * T) * norm.cdf(phi * x2))


def _barrier_C(S, K, H, T, r, q, sigma, phi, eta) -> float:
    """Term C — rebate term."""
    mu, lam = _barrier_mu_lambda(r, q, sigma)
    y = (math.log(H / S) / (sigma * math.sqrt(T))) + lam * sigma * math.sqrt(T)
    return ((H / S) ** (mu + lam) * norm.cdf(eta * y)
            + (H / S) ** (mu - lam) * norm.cdf(eta * (y - 2 * lam * sigma * math.sqrt(T))))


def _barrier_D(S, H, T, r, q, sigma, phi, eta) -> float:
    """Term D — asset-touches-barrier term."""
    mu, lam = _barrier_mu_lambda(r, q, sigma)
    y1 = (math.log(H / S) / (sigma * math.sqrt(T))) + lam * sigma * math.sqrt(T)
    return ((H / S) ** (mu + lam) * norm.cdf(eta * y1)
            + (H / S) ** (mu - lam) * norm.cdf(eta * (y1 - 2 * lam * sigma * math.sqrt(T))))


def _price_down_out_call(S, K, H, T, r, sigma, q) -> float:
    """Down-and-out call (H < S, knocked out when S hits H from above)."""
    if S <= H:
        return 0.0
    phi, eta = 1.0, 1.0
    if K >= H:
        return _barrier_A(S, K, H, T, r, q, sigma, phi, eta) - _barrier_B(S, K, H, T, r, q, sigma, phi, eta)
    else:
        # K < H: option is always knocked out before any payoff
        return 0.0


def _price_down_in_call(S, K, H, T, r, sigma, q) -> float:
    """Down-and-in call (activated when S falls to H)."""
    vanilla = bs_call(S, K, T, r, sigma, q)
    out_price = _price_down_out_call(S, K, H, T, r, sigma, q)
    return max(vanilla - out_price, 0.0)


def _price_up_out_put(S, K, H, T, r, sigma, q) -> float:
    """Up-and-out put (H > S, knocked out when S rises to H)."""
    if S >= H:
        return 0.0
    phi, eta = -1.0, -1.0
    if K <= H:
        return _barrier_A(S, K, H, T, r, q, sigma, phi, eta) - _barrier_B(S, K, H, T, r, q, sigma, phi, eta)
    else:
        return 0.0


def _price_up_in_put(S, K, H, T, r, sigma, q) -> float:
    """Up-and-in put (activated when S rises to H)."""
    vanilla = bs_put(S, K, T, r, sigma, q)
    out_price = _price_up_out_put(S, K, H, T, r, sigma, q)
    return max(vanilla - out_price, 0.0)


def _price_up_out_call(S, K, H, T, r, sigma, q) -> float:
    """Up-and-out call (knocked out when S rises to H, H > S)."""
    if S >= H:
        return 0.0
    vanilla = bs_call(S, K, T, r, sigma, q)
    # Up-and-in call = vanilla - up-and-out call; derive from put-call-barrier symmetry
    # Use direct MC for up-out call (barrier above S for call)
    phi, eta = 1.0, -1.0
    if K <= H:
        val = (_barrier_A(S, K, H, T, r, q, sigma, phi, eta)
               - _barrier_B(S, K, H, T, r, q, sigma, phi, eta))
        return max(val, 0.0)
    return vanilla  # barrier above payoff region: never knocked


def _price_up_in_call(S, K, H, T, r, sigma, q) -> float:
    """Up-and-in call (activated when S rises to H)."""
    vanilla = bs_call(S, K, T, r, sigma, q)
    out_price = _price_up_out_call(S, K, H, T, r, sigma, q)
    return max(vanilla - out_price, 0.0)


def _price_down_out_put(S, K, H, T, r, sigma, q) -> float:
    """Down-and-out put (knocked out when S falls to H)."""
    if S <= H:
        return 0.0
    vanilla = bs_put(S, K, T, r, sigma, q)
    phi, eta = -1.0, 1.0
    if K >= H:
        val = (_barrier_A(S, K, H, T, r, q, sigma, phi, eta)
               - _barrier_B(S, K, H, T, r, q, sigma, phi, eta))
        return max(val, 0.0)
    return vanilla


def _price_down_in_put(S, K, H, T, r, sigma, q) -> float:
    """Down-and-in put (activated when S falls to H)."""
    vanilla = bs_put(S, K, T, r, sigma, q)
    out_price = _price_down_out_put(S, K, H, T, r, sigma, q)
    return max(vanilla - out_price, 0.0)


_BARRIER_DISPATCH = {
    ("down-and-out", "call"): _price_down_out_call,
    ("down-and-in",  "call"): _price_down_in_call,
    ("up-and-out",   "call"): _price_up_out_call,
    ("up-and-in",    "call"): _price_up_in_call,
    ("down-and-out", "put"):  _price_down_out_put,
    ("down-and-in",  "put"):  _price_down_in_put,
    ("up-and-out",   "put"):  _price_up_out_put,
    ("up-and-in",    "put"):  _price_up_in_put,
}


# ---------------------------------------------------------------------------
# Asian option pricing
# ---------------------------------------------------------------------------

def _asian_geometric_params(r: float, q: float, sigma: float, T: float) -> tuple[float, float]:
    """Adjusted drift and vol for continuous geometric averaging."""
    sigma_g = sigma / math.sqrt(3.0)
    # Adjusted interest rate (b_g) for geometric average:
    # b_g = 0.5 * ((r - q) + sigma_g^2)  per standard textbook
    b = r - q
    b_g = 0.5 * (b - sigma ** 2 / 6.0)
    return sigma_g, b_g


def _price_asian_geometric_call(S: float, K: float, T: float, r: float,
                                 sigma: float, q: float = 0.0) -> float:
    """Closed-form geometric Asian call using Kemna-Vorst approximation."""
    sigma_g, b_g = _asian_geometric_params(r, q, sigma, T)
    if sigma_g <= 0 or T <= 0:
        return max(S * math.exp(b_g * T) - K, 0.0) * math.exp(-r * T)
    d1 = (math.log(S / K) + (b_g + 0.5 * sigma_g ** 2) * T) / (sigma_g * math.sqrt(T))
    d2 = d1 - sigma_g * math.sqrt(T)
    return math.exp(-r * T) * (S * math.exp(b_g * T) * norm.cdf(d1) - K * norm.cdf(d2))


def _price_asian_geometric_put(S: float, K: float, T: float, r: float,
                                sigma: float, q: float = 0.0) -> float:
    """Closed-form geometric Asian put."""
    sigma_g, b_g = _asian_geometric_params(r, q, sigma, T)
    if sigma_g <= 0 or T <= 0:
        return max(K - S * math.exp(b_g * T), 0.0) * math.exp(-r * T)
    d1 = (math.log(S / K) + (b_g + 0.5 * sigma_g ** 2) * T) / (sigma_g * math.sqrt(T))
    d2 = d1 - sigma_g * math.sqrt(T)
    return math.exp(-r * T) * (K * norm.cdf(-d2) - S * math.exp(b_g * T) * norm.cdf(-d1))


def _price_asian_arithmetic_mc(S: float, K: float, T: float, r: float, sigma: float,
                                q: float = 0.0, n_sims: int = 10_000,
                                n_steps: int = 252, seed: int = 42,
                                option_type: str = "call") -> float:
    """Monte Carlo arithmetic Asian option pricing."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * math.sqrt(dt)
    Z = rng.standard_normal((n_sims, n_steps))
    log_returns = drift + vol_dt * Z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S * np.exp(log_paths)  # shape (n_sims, n_steps)
    avg = paths.mean(axis=1)  # arithmetic average
    if option_type == "call":
        payoffs = np.maximum(avg - K, 0.0)
    else:
        payoffs = np.maximum(K - avg, 0.0)
    return math.exp(-r * T) * payoffs.mean()


# ---------------------------------------------------------------------------
# Lookback option pricing
# ---------------------------------------------------------------------------

def _price_lookback_floating_call_closed(S: float, T: float, r: float,
                                          sigma: float, q: float = 0.0,
                                          S_min: Optional[float] = None) -> float:
    """Goldman-Sosin-Gatto (1979) closed-form for floating-strike lookback call.

    Payoff = S_T - S_min  (where S_min is the minimum observed price over [0,T])
    Assumes S_min = S (path starts now, so current price IS the minimum initially).

    Formula from Haug "Complete Guide to Option Pricing Formulas" (floating lookback call):
      C = S*e^{(b-r)T}*N(a1) - S_min*e^{-rT}*N(a2)
          - S*e^{(b-r)T}*(sigma^2/(2b))*N(-a1)
          + S_min*e^{-rT}*(S/S_min)^{-2b/sigma^2}*(sigma^2/(2b))*N(-a3)
    where b = r - q (cost of carry).
    """
    if S_min is None:
        S_min = S  # at inception, current price is the running minimum
    b = r - q
    if sigma <= 0 or T <= 0:
        return max(S - S_min, 0.0) * math.exp(-r * T)
    sqrtT = math.sqrt(T)
    a1 = (math.log(S / S_min) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    a2 = a1 - sigma * sqrtT
    a3 = (math.log(S / S_min) + (-b + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    ebr_T = math.exp((b - r) * T)
    disc = math.exp(-r * T)
    if abs(b) < 1e-10:
        # Limiting case b -> 0: sigma^2/(2b) * [N(-a1) - N(-a3)] -> use L'Hopital
        # For b=0: price = S*N(a1) - S_min*e^{-rT}*N(a2)
        #                 + S*e^{-rT}*sigma*sqrtT*norm.pdf(a1)  (limiting form)
        price = (S * norm.cdf(a1) - S_min * disc * norm.cdf(a2)
                 + S * disc * sigma * sqrtT * norm.pdf(a1))
    else:
        coeff = sigma ** 2 / (2.0 * b)
        term1 = S * ebr_T * norm.cdf(a1)
        term2 = S_min * disc * norm.cdf(a2)
        term3 = S * ebr_T * coeff * norm.cdf(-a1)
        # Power term: (S/S_min)^{-2b/sigma^2}
        power = (S / S_min) ** (-2.0 * b / sigma ** 2)
        term4 = S_min * disc * power * coeff * norm.cdf(-a3)
        price = term1 - term2 - term3 + term4
    return max(price, 0.0)


def _price_lookback_mc(S: float, K: float, T: float, r: float, sigma: float,
                        q: float = 0.0, strike_type: str = "floating",
                        option_type: str = "call",
                        n_sims: int = 10_000, n_steps: int = 252,
                        seed: int = 42) -> float:
    """Monte Carlo lookback option pricing (floating or fixed strike)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * math.sqrt(dt)
    Z = rng.standard_normal((n_sims, n_steps))
    log_returns = drift + vol_dt * Z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S * np.exp(log_paths)  # (n_sims, n_steps)
    S_T = paths[:, -1]
    S_max = paths.max(axis=1)
    S_min = paths.min(axis=1)
    if strike_type == "floating" and option_type == "call":
        # payoff = S_T - S_min
        payoffs = np.maximum(S_T - S_min, 0.0)
    elif strike_type == "floating" and option_type == "put":
        # payoff = S_max - S_T
        payoffs = np.maximum(S_max - S_T, 0.0)
    elif strike_type == "fixed" and option_type == "call":
        # payoff = max(S_max - K, 0)
        payoffs = np.maximum(S_max - K, 0.0)
    else:
        # fixed put: max(K - S_min, 0)
        payoffs = np.maximum(K - S_min, 0.0)
    return math.exp(-r * T) * payoffs.mean()


# ---------------------------------------------------------------------------
# Digital / Binary options
# ---------------------------------------------------------------------------

def _price_digital_cash_or_nothing(S: float, K: float, T: float, r: float,
                                    sigma: float, q: float = 0.0,
                                    option_type: str = "call") -> float:
    """Cash-or-nothing digital: pays $1 if option expires in the money.

    Call: e^{-rT} * N(d2)
    Put:  e^{-rT} * N(-d2)
    """
    if T <= 0:
        if option_type == "call":
            return math.exp(-r * T) if S > K else 0.0
        return math.exp(-r * T) if S < K else 0.0
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(d1, sigma, T)
    discount = math.exp(-r * T)
    if option_type == "call":
        return discount * norm.cdf(d2)
    return discount * norm.cdf(-d2)


def _price_digital_asset_or_nothing(S: float, K: float, T: float, r: float,
                                     sigma: float, q: float = 0.0,
                                     option_type: str = "call") -> float:
    """Asset-or-nothing digital: pays S_T if option expires in the money.

    Call: S * e^{-qT} * N(d1)
    Put:  S * e^{-qT} * N(-d1)
    """
    if T <= 0:
        if option_type == "call":
            return S * math.exp(-q * T) if S > K else 0.0
        return S * math.exp(-q * T) if S < K else 0.0
    d1 = _d1(S, K, T, r, sigma, q)
    asset_pv = S * math.exp(-q * T)
    if option_type == "call":
        return asset_pv * norm.cdf(d1)
    return asset_pv * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Chooser option
# ---------------------------------------------------------------------------

def _price_chooser(S: float, K: float, T: float, t_c: float,
                   r: float, sigma: float, q: float = 0.0) -> float:
    """Simple chooser option (holder picks call or put at time t_c).

    Decomposition:
      Chooser = Call(S, K, T) + Put(S, K*, t_c)
    where K* = K * e^{-(r-q)*(T-t_c)} adjusted strike for the embedded put.

    This follows from the identity:
      max(C, P) = C + max(P - C, 0)
    and put-call parity to convert to: Call(S,K,T) + Put(S,K*,t_c).
    """
    K_star = K * math.exp(-(r - q) * (T - t_c))
    call_price = bs_call(S, K, T, r, sigma, q)
    # The embedded put has maturity t_c (when choice is made) and strike K*
    put_price = bs_put(S, K_star, t_c, r, sigma, q)
    return call_price + put_price


# ---------------------------------------------------------------------------
# Dataclass pricers
# ---------------------------------------------------------------------------

@dataclass
class BarrierOption:
    """Barrier option pricer (Rubinstein-Reiner 1991 closed-form).

    barrier_type: 'down-and-out' | 'down-and-in' | 'up-and-out' | 'up-and-in'
    option_type:  'call' | 'put'
    """
    S: float
    K: float
    H: float  # barrier level
    T: float
    r: float
    sigma: float
    q: float = 0.0
    barrier_type: str = "down-and-out"
    option_type: str = "call"

    def price(self) -> float:
        """Return the barrier option price."""
        key = (self.barrier_type, self.option_type)
        fn = _BARRIER_DISPATCH.get(key)
        if fn is None:
            raise ValueError(f"Unknown barrier type combo: {key}")
        return fn(self.S, self.K, self.H, self.T, self.r, self.sigma, self.q)

    def rebate_price(self, rebate: float = 0.0) -> float:
        """Price including a rebate paid at knock-in/knock-out event."""
        base = self.price()
        if rebate == 0.0:
            return base
        # Rebate component: PV of rebate * probability of hitting barrier
        # Simplified: for knock-out, rebate paid if barrier is hit
        mu, lam = _barrier_mu_lambda(self.r, self.q, self.sigma)
        if "down" in self.barrier_type:
            eta = 1.0
        else:
            eta = -1.0
        y = (math.log(self.H / self.S) / (self.sigma * math.sqrt(self.T))
             + lam * self.sigma * math.sqrt(self.T))
        rebate_term = rebate * math.exp(-self.r * self.T) * _barrier_D(
            self.S, self.H, self.T, self.r, self.q, self.sigma, 1.0, eta
        )
        if "out" in self.barrier_type:
            return base + rebate_term
        return base

    def vanilla_price(self) -> float:
        """Black-Scholes vanilla price for comparison."""
        if self.option_type == "call":
            return bs_call(self.S, self.K, self.T, self.r, self.sigma, self.q)
        return bs_put(self.S, self.K, self.T, self.r, self.sigma, self.q)


@dataclass
class AsianOption:
    """Asian (average-rate) option pricer.

    averaging: 'geometric' (closed-form) | 'arithmetic' (Monte Carlo)
    option_type: 'call' | 'put'
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    averaging: str = "geometric"
    option_type: str = "call"
    n_sims: int = 10_000
    n_steps: int = 252
    seed: int = 42

    def price(self) -> float:
        """Return the Asian option price."""
        if self.averaging == "geometric":
            if self.option_type == "call":
                return _price_asian_geometric_call(
                    self.S, self.K, self.T, self.r, self.sigma, self.q
                )
            return _price_asian_geometric_put(
                self.S, self.K, self.T, self.r, self.sigma, self.q
            )
        # arithmetic MC
        return _price_asian_arithmetic_mc(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            self.n_sims, self.n_steps, self.seed, self.option_type
        )

    def delta(self, bump: float = 0.01) -> float:
        """Numerical delta via finite-difference bump (1% of spot)."""
        dS = self.S * bump
        up = AsianOption(
            self.S + dS, self.K, self.T, self.r, self.sigma,
            self.q, self.averaging, self.option_type,
            self.n_sims, self.n_steps, self.seed
        ).price()
        dn = AsianOption(
            self.S - dS, self.K, self.T, self.r, self.sigma,
            self.q, self.averaging, self.option_type,
            self.n_sims, self.n_steps, self.seed
        ).price()
        return (up - dn) / (2.0 * dS)


@dataclass
class LookbackOption:
    """Lookback option pricer.

    strike_type: 'floating' (closed-form available) | 'fixed' (Monte Carlo)
    option_type: 'call' | 'put'
    """
    S: float
    K: float  # used only for fixed-strike; set to 0.0 for floating
    T: float
    r: float
    sigma: float
    q: float = 0.0
    strike_type: str = "floating"
    option_type: str = "call"
    n_sims: int = 10_000
    n_steps: int = 252
    seed: int = 42

    def price(self) -> float:
        """Return the lookback option price."""
        if self.strike_type == "floating" and self.option_type == "call":
            # Use closed-form GSG formula
            return _price_lookback_floating_call_closed(
                self.S, self.T, self.r, self.sigma, self.q
            )
        # Fall back to Monte Carlo for all other combinations
        return _price_lookback_mc(
            self.S, self.K, self.T, self.r, self.sigma,
            self.q, self.strike_type, self.option_type,
            self.n_sims, self.n_steps, self.seed
        )


# ---------------------------------------------------------------------------
# ExoticPricer convenience class
# ---------------------------------------------------------------------------

class ExoticPricer:
    """Unified interface for all exotic option types."""

    def barrier_option(self, S: float, K: float, H: float, T: float,
                       r: float, sigma: float, q: float = 0.0,
                       barrier_type: str = "down-and-out",
                       option_type: str = "call") -> float:
        """Price a barrier option using Rubinstein-Reiner closed-form."""
        return BarrierOption(
            S=S, K=K, H=H, T=T, r=r, sigma=sigma, q=q,
            barrier_type=barrier_type, option_type=option_type
        ).price()

    def asian_geometric(self, S: float, K: float, T: float, r: float,
                        sigma: float, q: float = 0.0,
                        option_type: str = "call") -> float:
        """Price a geometric Asian option (closed-form Kemna-Vorst)."""
        if option_type == "call":
            return _price_asian_geometric_call(S, K, T, r, sigma, q)
        return _price_asian_geometric_put(S, K, T, r, sigma, q)

    def asian_arithmetic_mc(self, S: float, K: float, T: float, r: float,
                             sigma: float, q: float = 0.0,
                             n_sims: int = 10_000, n_steps: int = 252,
                             seed: int = 42,
                             option_type: str = "call") -> float:
        """Price an arithmetic Asian option via Monte Carlo."""
        return _price_asian_arithmetic_mc(
            S, K, T, r, sigma, q, n_sims, n_steps, seed, option_type
        )

    def lookback_floating(self, S: float, T: float, r: float, sigma: float,
                          q: float = 0.0, n_sims: int = 10_000,
                          seed: int = 42) -> float:
        """Price a floating-strike lookback call (GSG closed-form)."""
        return _price_lookback_floating_call_closed(S, T, r, sigma, q)

    def digital_cash_or_nothing(self, S: float, K: float, T: float, r: float,
                                 sigma: float, q: float = 0.0,
                                 option_type: str = "call") -> float:
        """Price a cash-or-nothing digital option."""
        return _price_digital_cash_or_nothing(S, K, T, r, sigma, q, option_type)

    def digital_asset_or_nothing(self, S: float, K: float, T: float, r: float,
                                  sigma: float, q: float = 0.0,
                                  option_type: str = "call") -> float:
        """Price an asset-or-nothing digital option."""
        return _price_digital_asset_or_nothing(S, K, T, r, sigma, q, option_type)

    def chooser(self, S: float, K: float, T: float, t_c: float,
                r: float, sigma: float, q: float = 0.0) -> float:
        """Price a simple chooser option (choice at t_c, expiry at T)."""
        return _price_chooser(S, K, T, t_c, r, sigma, q)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def price_barrier(S: float, K: float, H: float, T: float, r: float, sigma: float,
                  barrier_type: str = "down-and-out", option_type: str = "call",
                  q: float = 0.0) -> float:
    """Price a barrier option. Dispatch to Rubinstein-Reiner closed-form."""
    return BarrierOption(S=S, K=K, H=H, T=T, r=r, sigma=sigma, q=q,
                         barrier_type=barrier_type, option_type=option_type).price()


def price_asian_geometric(S: float, K: float, T: float, r: float, sigma: float,
                           q: float = 0.0, option_type: str = "call") -> float:
    """Closed-form geometric Asian option price."""
    if option_type == "call":
        return _price_asian_geometric_call(S, K, T, r, sigma, q)
    return _price_asian_geometric_put(S, K, T, r, sigma, q)


def price_asian_mc(S: float, K: float, T: float, r: float, sigma: float,
                   q: float = 0.0, n_sims: int = 10_000, n_steps: int = 252,
                   seed: int = 42, option_type: str = "call") -> float:
    """Arithmetic Asian option price via Monte Carlo."""
    return _price_asian_arithmetic_mc(S, K, T, r, sigma, q, n_sims, n_steps, seed, option_type)


def price_lookback_mc(S: float, T: float, r: float, sigma: float,
                      q: float = 0.0, n_sims: int = 10_000,
                      n_steps: int = 252, seed: int = 42) -> float:
    """Floating-strike lookback call price (closed-form GSG)."""
    return _price_lookback_floating_call_closed(S, T, r, sigma, q)


def price_digital(S: float, K: float, T: float, r: float, sigma: float,
                  q: float = 0.0, digital_type: str = "cash",
                  option_type: str = "call") -> float:
    """Price a digital (binary) option.

    digital_type: 'cash' (cash-or-nothing) | 'asset' (asset-or-nothing)
    option_type:  'call' | 'put'
    """
    if digital_type == "cash":
        return _price_digital_cash_or_nothing(S, K, T, r, sigma, q, option_type)
    return _price_digital_asset_or_nothing(S, K, T, r, sigma, q, option_type)


def price_exotic(exotic_type: str, **kwargs) -> float:
    """Generic entry point for any exotic option type.

    exotic_type: 'barrier' | 'asian_geometric' | 'asian_mc' |
                 'lookback' | 'digital' | 'chooser'
    """
    dispatch = {
        "barrier": lambda kw: price_barrier(**kw),
        "asian_geometric": lambda kw: price_asian_geometric(**kw),
        "asian_mc": lambda kw: price_asian_mc(**kw),
        "lookback": lambda kw: price_lookback_mc(**kw),
        "digital": lambda kw: price_digital(**kw),
        "chooser": lambda kw: _price_chooser(**kw),
    }
    fn = dispatch.get(exotic_type)
    if fn is None:
        raise ValueError(f"Unknown exotic option type: {exotic_type!r}. "
                         f"Choose from: {list(dispatch)}")
    return fn(kwargs)
