"""Options Greeks surface engine — Black-Scholes delta/gamma/vega/theta with vectorised surface support."""
from __future__ import annotations

import math
from typing import Literal, Union

import numpy as np
from scipy.stats import norm

__all__ = [
    "GreeksSurface",
    "compute_delta",
    "compute_gamma",
    "compute_vega",
    "compute_theta",
]

# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes d1 parameter."""
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(d1_val: float, sigma: float, T: float) -> float:
    """Black-Scholes d2 parameter."""
    return d1_val - sigma * math.sqrt(T)


# ---------------------------------------------------------------------------
# Public scalar functions
# ---------------------------------------------------------------------------

def compute_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Black-Scholes delta for a European call or put.

    delta_call = N(d1)
    delta_put  = N(d1) - 1
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1_val = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1_val)
    return norm.cdf(d1_val) - 1.0


def compute_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",  # noqa: ARG001  same for both
) -> float:
    """Black-Scholes gamma — identical for calls and puts.

    gamma = N'(d1) / (S * sigma * sqrt(T))
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1_val = _d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * math.sqrt(T))


def compute_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",  # noqa: ARG001  same for both
) -> float:
    """Black-Scholes vega per 1% move in implied volatility.

    vega = S * N'(d1) * sqrt(T) / 100
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1_val = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1_val) * math.sqrt(T) / 100.0


def compute_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Black-Scholes theta per calendar day.

    theta_call = (-S*N'(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2))  / 365
    theta_put  = (-S*N'(d1)*sigma/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)) / 365
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    sqrtT = math.sqrt(T)
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(d1_val, sigma, T)
    disc = math.exp(-r * T)
    nd1 = norm.pdf(d1_val)
    common = -S * nd1 * sigma / (2.0 * sqrtT)
    if option_type == "call":
        return (common - r * K * disc * norm.cdf(d2_val)) / 365.0
    return (common + r * K * disc * norm.cdf(-d2_val)) / 365.0


# ---------------------------------------------------------------------------
# GreeksSurface class
# ---------------------------------------------------------------------------

class GreeksSurface:
    """Compute the full Black-Scholes Greeks surface for a European option.

    Supports both scalar inputs and NumPy array inputs for vectorised
    surface computation (e.g. a grid of strikes × expiries).

    Parameters
    ----------
    S : float | np.ndarray
        Underlying spot price.
    K : float | np.ndarray
        Strike price(s).
    T : float | np.ndarray
        Time to expiration in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float | np.ndarray
        Annualised implied volatility.
    option_type : "call" | "put"
        European option type.
    """

    def __init__(
        self,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: float,
        sigma: Union[float, np.ndarray],
        option_type: Literal["call", "put"] = "call",
    ) -> None:
        self.S = np.asarray(S, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.r = float(r)
        self.sigma = np.asarray(sigma, dtype=float)
        self.option_type = option_type

        # Pre-compute common terms (broadcast-safe)
        self._sqrtT = np.sqrt(self.T)
        self._d1 = (
            np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * self._sqrtT)
        self._d2 = self._d1 - self.sigma * self._sqrtT
        self._disc = np.exp(-self.r * self.T)
        self._nd1 = norm.pdf(self._d1)          # N'(d1)
        self._Nd1 = norm.cdf(self._d1)           # N(d1)
        self._Nd2 = norm.cdf(self._d2)           # N(d2)
        self._Nnd2 = norm.cdf(-self._d2)         # N(-d2)

    # ------------------------------------------------------------------ #
    # Core Greeks
    # ------------------------------------------------------------------ #

    def delta(self) -> Union[float, np.ndarray]:
        """delta_call = N(d1),  delta_put = N(d1) - 1."""
        if self.option_type == "call":
            return self._Nd1
        return self._Nd1 - 1.0

    def gamma(self) -> Union[float, np.ndarray]:
        """gamma = N'(d1) / (S * sigma * sqrt(T))  — same for calls and puts."""
        return self._nd1 / (self.S * self.sigma * self._sqrtT)

    def vega(self) -> Union[float, np.ndarray]:
        """vega = S * N'(d1) * sqrt(T) / 100  (per 1% move in vol)."""
        return self.S * self._nd1 * self._sqrtT / 100.0

    def theta(self) -> Union[float, np.ndarray]:
        """theta per calendar day.

        theta_call = (-S*N'(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2))  / 365
        theta_put  = (-S*N'(d1)*sigma/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)) / 365
        """
        common = -self.S * self._nd1 * self.sigma / (2.0 * self._sqrtT)
        if self.option_type == "call":
            return (common - self.r * self.K * self._disc * self._Nd2) / 365.0
        return (common + self.r * self.K * self._disc * self._Nnd2) / 365.0

    def rho(self) -> Union[float, np.ndarray]:
        """rho per 1% change in r.

        rho_call = K * T * exp(-r*T) * N(d2)  / 100
        rho_put  = -K * T * exp(-r*T) * N(-d2) / 100
        """
        if self.option_type == "call":
            return self.K * self.T * self._disc * self._Nd2 / 100.0
        return -self.K * self.T * self._disc * self._Nnd2 / 100.0

    # ------------------------------------------------------------------ #
    # Higher-order Greeks
    # ------------------------------------------------------------------ #

    def vanna(self) -> Union[float, np.ndarray]:
        """vanna = d²V/(dS dσ) = -N'(d1) * d2 / sigma."""
        return -self._nd1 * self._d2 / self.sigma

    def vomma(self) -> Union[float, np.ndarray]:
        """vomma = d²V/dσ² = vega * d1 * d2 / sigma."""
        return self.vega() * self._d1 * self._d2 / self.sigma

    def charm(self) -> Union[float, np.ndarray]:
        """charm = dDelta/dt per calendar day."""
        numerator = 2.0 * self.r * self.T - self._d2 * self.sigma * self._sqrtT
        charm_annual = -self._nd1 * numerator / (2.0 * self.T * self.sigma * self._sqrtT)
        return charm_annual / 365.0

    # ------------------------------------------------------------------ #
    # Convenience: return all greeks as a dict
    # ------------------------------------------------------------------ #

    def all_greeks(self) -> dict[str, Union[float, np.ndarray]]:
        """Return a dict of all computed Greeks."""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
            "rho": self.rho(),
            "vanna": self.vanna(),
            "vomma": self.vomma(),
            "charm": self.charm(),
        }

    # ------------------------------------------------------------------ #
    # Class-method: build a surface grid
    # ------------------------------------------------------------------ #

    @classmethod
    def surface(
        cls,
        S: float,
        strikes: Union[list[float], np.ndarray],
        expiries: Union[list[float], np.ndarray],
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
    ) -> dict[str, np.ndarray]:
        """Compute Greeks over a (len(strikes) × len(expiries)) grid.

        Returns a dict where each value is a 2-D array of shape
        (len(strikes), len(expiries)).
        """
        K_arr = np.asarray(strikes, dtype=float)
        T_arr = np.asarray(expiries, dtype=float)
        # Build meshgrid: rows = strikes, cols = expiries
        K_grid, T_grid = np.meshgrid(K_arr, T_arr, indexing="ij")
        inst = cls(S, K_grid, T_grid, r, sigma, option_type)
        return inst.all_greeks()

    # ------------------------------------------------------------------ #
    # String representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        scalar = self.S.ndim == 0
        return (
            f"GreeksSurface("
            f"S={float(self.S):.2f}, "
            f"K={float(self.K):.2f}, "
            f"T={float(self.T):.4f}, "
            f"r={self.r:.4f}, "
            f"sigma={float(self.sigma):.4f}, "
            f"type={self.option_type}"
            f")"
            if scalar
            else f"GreeksSurface(surface grid, type={self.option_type})"
        )
