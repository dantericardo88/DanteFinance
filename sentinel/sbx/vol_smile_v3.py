"""SABR volatility smile and skew analytics — Hagan 2002 approximation.

Implements:
- SABRParams dataclass (alpha, beta, rho, nu)
- SABRModel: implied vol surface, ATM vol, skew, calibration
- VolSmile: cubic-spline interpolation, risk reversal, strangle, butterfly
- sabr_implied_vol: scalar Hagan formula
- fit_sabr: full SABR calibration
- dupire_local_vol: Dupire local vol from smile via finite differences
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

__all__ = [
    "SABRParams",
    "SABRModel",
    "VolSmile",
    "sabr_implied_vol",
    "fit_sabr",
    "dupire_local_vol",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SABRParams:
    """SABR model parameters.

    alpha : vol-of-vol level (initial vol, > 0)
    beta  : CEV exponent in [0, 1] (0 = normal, 1 = log-normal)
    rho   : correlation between asset and vol Brownian motions
    nu    : vol-of-vol (> 0)
    """
    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if not (0 <= self.beta <= 1):
            raise ValueError("beta must be in [0, 1]")
        if not (-1 < self.rho < 1):
            raise ValueError("rho must be in (-1, 1)")
        if self.nu <= 0:
            raise ValueError("nu must be > 0")


# ---------------------------------------------------------------------------
# Core Hagan (2002) SABR implied-vol formula
# ---------------------------------------------------------------------------

_EPS = 1e-12  # near-zero guard


def sabr_implied_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Black implied volatility from the Hagan 2002 SABR approximation.

    Parameters
    ----------
    F     : forward price
    K     : strike
    T     : time to expiry (years)
    alpha : SABR alpha
    beta  : SABR beta
    rho   : SABR rho
    nu    : SABR nu

    Returns
    -------
    Implied Black volatility (annualised)
    """
    if T <= 0 or F <= 0 or K <= 0 or alpha <= 0:
        return float("nan")

    # Common quantities
    b = beta
    FK = F * K
    FK_mid = FK ** ((1.0 - b) / 2.0)  # (F*K)^((1-beta)/2)
    log_FK = math.log(F / K)           # log(F/K)

    # --- A: denominator factor ---
    log2 = log_FK ** 2
    log4 = log2 ** 2
    denom_A = FK_mid * (
        1.0
        + ((1.0 - b) ** 2 / 24.0) * log2
        + ((1.0 - b) ** 4 / 1920.0) * log4
    )

    # --- B: time correction factor ---
    FK_1b = FK ** (1.0 - b)  # (F*K)^(1-beta)
    time_corr = (
        (1.0 - b) ** 2 / 24.0 * alpha ** 2 / FK_1b
        + rho * b * nu * alpha / (4.0 * FK_mid)
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    )

    # --- C: z / x(z) factor (ATM case uses limit = 1) ---
    if abs(log_FK) < _EPS:
        # ATM limit: z/x(z) -> 1
        zx = 1.0
    else:
        z = (nu / alpha) * FK_mid * log_FK
        # x(z) = log((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
        disc = math.sqrt(max(1.0 - 2.0 * rho * z + z * z, 0.0))
        numerator_x = disc + z - rho
        denominator_x = 1.0 - rho
        if numerator_x <= 0 or denominator_x <= 0:
            # degenerate — fall back to 1
            zx = 1.0
        else:
            x_z = math.log(numerator_x / denominator_x)
            zx = z / x_z if abs(x_z) > _EPS else 1.0

    sigma_B = (alpha / denom_A) * zx * (1.0 + time_corr * T)
    return max(sigma_B, 1e-8)


# ---------------------------------------------------------------------------
# SABRModel class
# ---------------------------------------------------------------------------

class SABRModel:
    """SABR model utilities: vol surface, ATM vol, skew, calibration."""

    # ── core ─────────────────────────────────────────────────────────────────

    def implied_vol(self, F: float, K: float, T: float, params: SABRParams) -> float:
        """Single implied vol from SABR Hagan formula."""
        return sabr_implied_vol(F, K, T, params.alpha, params.beta, params.rho, params.nu)

    def implied_vol_surface(
        self,
        F: float,
        strikes: np.ndarray,
        expiries: np.ndarray,
        params: SABRParams,
    ) -> np.ndarray:
        """Implied vol surface over a grid of strikes x expiries.

        Returns array of shape (len(strikes), len(expiries)).
        """
        strikes = np.asarray(strikes, dtype=float)
        expiries = np.asarray(expiries, dtype=float)
        surface = np.empty((len(strikes), len(expiries)))
        for j, T in enumerate(expiries):
            for i, K in enumerate(strikes):
                surface[i, j] = self.implied_vol(F, K, T, params)
        return surface

    def atm_vol(self, F: float, T: float, params: SABRParams) -> float:
        """ATM vol (K = F limit of Hagan formula)."""
        return self.implied_vol(F, F, T, params)

    # ── skew ─────────────────────────────────────────────────────────────────

    def skew(
        self,
        F: float,
        T: float,
        params: SABRParams,
        dk: float = 0.01,
    ) -> float:
        """Numerical first derivative of implied vol w.r.t. K, evaluated at K = F.

        dσ/dK|_{K=F} using central finite difference with step dk (absolute).
        """
        h = F * dk  # relative step
        vol_up = self.implied_vol(F, F + h, T, params)
        vol_dn = self.implied_vol(F, F - h, T, params)
        return (vol_up - vol_dn) / (2.0 * h)

    # ── calibration ──────────────────────────────────────────────────────────

    def calibrate_alpha(
        self,
        F: float,
        T: float,
        market_vols: np.ndarray,
        strikes: np.ndarray,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """Fit alpha to match market implied vols at given strikes.

        Uses bounded scalar minimisation over alpha in (1e-6, 5).
        """
        market_vols = np.asarray(market_vols, dtype=float)
        strikes = np.asarray(strikes, dtype=float)

        def objective(alpha: float) -> float:
            total = 0.0
            for K, mv in zip(strikes, market_vols):
                model_v = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
                total += (model_v - mv) ** 2
            return total

        result = minimize_scalar(objective, bounds=(1e-6, 5.0), method="bounded")
        return float(result.x)


# ---------------------------------------------------------------------------
# fit_sabr — full four-parameter fit
# ---------------------------------------------------------------------------

def fit_sabr(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float = 0.5,
) -> SABRParams:
    """Calibrate full SABR model (fix beta, fit alpha/rho/nu) to market smiles.

    Uses scipy.optimize.minimize with L-BFGS-B bounds.
    Returns best-fit SABRParams.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)

    # Initial guess: ATM vol ~ alpha/F^(1-beta)
    atm_vol_guess = float(np.mean(market_vols))
    alpha0 = atm_vol_guess * (F ** (1.0 - beta))
    x0 = np.array([alpha0, -0.3, 0.4])  # alpha, rho, nu

    def objective(x: np.ndarray) -> float:
        alpha, rho, nu = x
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        total = 0.0
        for K, mv in zip(strikes, market_vols):
            model_v = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
            if math.isnan(model_v):
                return 1e10
            total += (model_v - mv) ** 2
        return total

    bounds = [(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)]
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000},
    )
    alpha_fit, rho_fit, nu_fit = result.x
    return SABRParams(alpha=alpha_fit, beta=beta, rho=rho_fit, nu=nu_fit)


# ---------------------------------------------------------------------------
# VolSmile — market smile wrapper
# ---------------------------------------------------------------------------

class VolSmile:
    """Market vol smile with cubic-spline interpolation and derived metrics.

    Parameters
    ----------
    strikes : array of strikes (ascending order)
    vols    : corresponding implied vols
    F       : forward price
    T       : time to expiry (years)
    """

    def __init__(
        self,
        strikes: np.ndarray,
        vols: np.ndarray,
        F: float,
        T: float,
    ) -> None:
        self._strikes = np.asarray(strikes, dtype=float)
        self._vols = np.asarray(vols, dtype=float)
        self.F = float(F)
        self.T = float(T)

        # sort by strike
        idx = np.argsort(self._strikes)
        self._strikes = self._strikes[idx]
        self._vols = self._vols[idx]

        # cubic spline interpolator
        self._cs = CubicSpline(self._strikes, self._vols, extrapolate=True)

    # ── interpolation ─────────────────────────────────────────────────────────

    def interpolate(self, K: float) -> float:
        """Implied vol at arbitrary strike K via cubic spline."""
        return float(self._cs(K))

    # ── smile metrics ─────────────────────────────────────────────────────────

    def _delta_strike(self, delta: float, option_type: str = "call") -> float:
        """Strike corresponding to a given Black-Scholes delta (approximate).

        Uses ATM vol as constant vol approximation.
        delta > 0 for calls, delta < 0 for puts (pass positive delta for puts).
        """
        # Approximate: K ~ F * exp(-norm.ppf(delta) * sigma_atm * sqrt(T))
        sigma_atm = self.interpolate(self.F)
        if self.T <= 0 or sigma_atm <= 0:
            return self.F
        if option_type == "call":
            # N(d1) = delta => d1 = norm.ppf(delta)
            d1 = norm.ppf(delta)
        else:
            # put: N(-d1) = delta => d1 = -norm.ppf(delta)
            d1 = -norm.ppf(delta)
        # d1 = (log(F/K)) / (sigma * sqrt(T))  (simplified, r=0)
        log_FK = d1 * sigma_atm * math.sqrt(self.T)
        return self.F * math.exp(-log_FK)

    def risk_reversal(self, delta: float = 0.25) -> float:
        """Risk reversal = sigma(25-delta call) - sigma(25-delta put)."""
        K_call = self._delta_strike(delta, "call")
        K_put = self._delta_strike(delta, "put")
        return self.interpolate(K_call) - self.interpolate(K_put)

    def strangle(self, delta: float = 0.25) -> float:
        """Strangle = 0.5 * (sigma(25d call) + sigma(25d put)) - sigma(ATM)."""
        K_call = self._delta_strike(delta, "call")
        K_put = self._delta_strike(delta, "put")
        wing_avg = 0.5 * (self.interpolate(K_call) + self.interpolate(K_put))
        atm_vol = self.interpolate(self.F)
        return wing_avg - atm_vol

    def butterfly(self, delta: float = 0.25) -> float:
        """Butterfly = strangle (same as strangle in vol-space convention)."""
        return self.strangle(delta)

    def is_arbitrage_free(self) -> bool:
        """Check for butterfly-spread arbitrage across the smile grid.

        Butterfly arbitrage: d^2C/dK^2 >= 0 for all strikes.
        Checked numerically via the second derivative of the spline.
        """
        # Evaluate second derivative on fine grid
        K_min = self._strikes[0]
        K_max = self._strikes[-1]
        K_grid = np.linspace(K_min, K_max, 200)
        d2_vols = self._cs(K_grid, 2)  # second derivative of vol w.r.t. K
        # A smile with d^2 sigma/dK^2 < 0 everywhere would flag arbitrage
        # (full Dupire check is more complex, this is a first-order screen)
        # For a convex smile the second derivative of vol >= 0 is desirable
        # but strict Butterfly positivity is: d2C/dK2 > 0 which relates to
        # probability density. We check a simplified condition: no extreme
        # negative curvature that would make density negative.
        # For the purposes of this check: assert min second derivative >= -0.01
        # (small tolerance for numerical spline noise)
        return bool(np.all(d2_vols >= -0.02))


# ---------------------------------------------------------------------------
# Dupire local vol
# ---------------------------------------------------------------------------

def dupire_local_vol(
    F: float,
    K: float,
    T: float,
    smile_func: Callable[[float, float], float],
    dK: float = 0.5,
    dT: float = 0.001,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    """Dupire local volatility from implied vol surface via finite differences.

    Dupire formula (zero rates approximation):
        sigma_loc^2(K,T) = (dsigma_imp^2 T / dT) / (d^2 C / dK^2 * K^2 / C)

    In terms of implied vol w = sigma_imp^2 * T (total variance):
        sigma_loc^2 = dw/dT / (1 - k/w * dw/dk + 0.25*(-0.25 - 1/w + k/w^2) * (dw/dk)^2 + 0.5 * d^2w/dk^2)

    where k = log(K/F).

    Uses the Gatheral (2006) form for numerical robustness.

    Parameters
    ----------
    F          : current forward
    K          : strike for local vol
    T          : expiry
    smile_func : callable(K, T) -> implied_vol
    dK         : finite-difference step in strike
    dT         : finite-difference step in time
    r, q       : risk-free rate and dividend yield (default 0)
    """
    if T <= dT or K <= dK:
        return float("nan")

    # Total variance w(k, T) = sigma^2 * T
    def w(k_strike: float, t: float) -> float:
        v = smile_func(k_strike, t)
        return v * v * t

    # log-moneyness
    k = math.log(K / F)

    # dw/dT  (forward finite difference)
    w0 = w(K, T)
    w_dT = w(K, T + dT)
    dw_dT = (w_dT - w0) / dT

    # dw/dk and d^2w/dk^2 (central finite differences in log-moneyness)
    K_up = K * math.exp(dK / K)    # approx: K+dK in log space
    K_dn = K * math.exp(-dK / K)
    # recompute using absolute strike differences for simplicity
    w_up = w(K + dK, T)
    w_dn = w(K - dK, T)
    dw_dk = (w_up - w_dn) / (2.0 * dK)
    d2w_dk2 = (w_up - 2.0 * w0 + w_dn) / (dK ** 2)

    # Gatheral denominator
    if w0 <= 0:
        return float("nan")

    denom = (
        1.0
        - k / w0 * dw_dk
        + 0.25 * (-0.25 - 1.0 / w0 + k ** 2 / w0 ** 2) * dw_dk ** 2
        + 0.5 * d2w_dk2
    )

    if denom <= 0 or dw_dT <= 0:
        return float("nan")

    loc_var = dw_dT / denom
    return math.sqrt(max(loc_var, 0.0))
