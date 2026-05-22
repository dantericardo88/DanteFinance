"""Implied Volatility Term Structure and Surface Fitting — Gatheral SVI, Nelson-Siegel, RBS interpolation.

Implements:
- SVIParams: Gatheral (2004) stochastic-volatility-inspired parametrization
- NelsonSiegelParams: Nelson-Siegel term-structure model for ATM vol vs expiry
- SVICalibrator: per-slice and full-surface SVI calibration
- IVTermStructure: ATM vol term structure with NS fitting and forward vol
- IVSurface: 2-D vol surface interpolation with arbitrage checks
- Module-level helper functions
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm

__all__ = [
    "SVIParams",
    "NelsonSiegelParams",
    "SVICalibrator",
    "IVTermStructure",
    "IVSurface",
    "fit_svi",
    "fit_nelson_siegel",
    "forward_vol",
    "iv_surface_from_svi",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-8


# ---------------------------------------------------------------------------
# SVIParams dataclass
# ---------------------------------------------------------------------------

@dataclass
class SVIParams:
    """Gatheral (2004) SVI parametrization for total variance.

    w(k) = a + b*(rho*(k - m) + sqrt((k - m)^2 + sigma^2))

    Parameters
    ----------
    a     : level (total-variance intercept at m, minus b*sigma)
    b     : angle (controls slope / wing steepness, >= 0)
    rho   : rotation / skew in (-1, 1)
    m     : translation (moneyness at minimum variance)
    sigma : smoothness (curvature of the smile, > 0)
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: float) -> float:
        """Compute SVI total variance w(k) = sigma_imp^2 * T."""
        diff = k - self.m
        return self.a + self.b * (self.rho * diff + math.sqrt(diff ** 2 + self.sigma ** 2))

    def implied_vol(self, k: float, T: float) -> float:
        """Return implied vol from SVI total variance for log-moneyness k and expiry T."""
        if T <= 0:
            raise ValueError("T must be positive")
        w = self.total_variance(k)
        if w < 0:
            w = 0.0
        return math.sqrt(w / T)

    @staticmethod
    def _from_array(x: np.ndarray) -> "SVIParams":
        return SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])

    def _to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])


# ---------------------------------------------------------------------------
# NelsonSiegelParams dataclass
# ---------------------------------------------------------------------------

@dataclass
class NelsonSiegelParams:
    """Nelson-Siegel parametrization for ATM vol as a function of expiry T.

    sigma(T) = beta0
               + beta1 * (1 - exp(-T/tau)) / (T/tau)
               + beta2 * ((1 - exp(-T/tau)) / (T/tau) - exp(-T/tau))

    beta0 : long-term level
    beta1 : short-term slope
    beta2 : curvature (hump)
    tau   : decay speed (> 0)
    """
    beta0: float
    beta1: float
    beta2: float
    tau: float

    def vol(self, T: float) -> float:
        """Evaluate the Nelson-Siegel ATM vol at expiry T."""
        if T <= 0:
            raise ValueError("T must be positive")
        tau = max(self.tau, _EPS)
        ratio = T / tau
        loading1 = (1.0 - math.exp(-ratio)) / ratio
        loading2 = loading1 - math.exp(-ratio)
        return self.beta0 + self.beta1 * loading1 + self.beta2 * loading2


# ---------------------------------------------------------------------------
# SVI Calibrator
# ---------------------------------------------------------------------------

class SVICalibrator:
    """Calibrate SVI parameters to market implied vols."""

    def fit(
        self,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        F: float,
        T: float,
    ) -> SVIParams:
        """Fit SVI to a single expiry smile.

        Parameters
        ----------
        strikes     : array of strike prices
        market_vols : array of implied vols (same length as strikes)
        F           : forward price
        T           : time to expiry (years)
        """
        strikes = np.asarray(strikes, dtype=float)
        market_vols = np.asarray(market_vols, dtype=float)
        if len(strikes) != len(market_vols):
            raise ValueError("strikes and market_vols must have equal length")

        k = np.log(strikes / F)
        w_market = market_vols ** 2 * T  # total variance targets

        def objective(x: np.ndarray) -> float:
            a, b, rho, m, sigma = x
            if b < 0 or sigma < _EPS or abs(rho) >= 1:
                return 1e9
            w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
            if np.any(w_model < 0):
                return 1e9
            return float(np.sum((w_model - w_market) ** 2))

        # Initial guess: flat smile with ATM level
        atm_var = float(np.median(market_vols) ** 2 * T)
        x0 = np.array([atm_var * 0.8, 0.1, -0.2, 0.0, 0.2])

        bounds = [
            (-0.5, 2.0),   # a
            (1e-4, 2.0),   # b
            (-0.999, 0.999),  # rho
            (-1.0, 1.0),   # m
            (1e-4, 2.0),   # sigma
        ]

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-12})
        if not res.success:
            # Try Nelder-Mead as fallback
            res2 = minimize(objective, x0, method="Nelder-Mead",
                            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
            if res2.fun < res.fun:
                res = res2

        return SVIParams._from_array(res.x)

    def fit_surface(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        market_vol_surface: np.ndarray,
        F: float,
    ) -> List[SVIParams]:
        """Fit SVI slice-by-slice to a vol surface.

        Parameters
        ----------
        strikes           : 1-D array of strike prices (n_strikes,)
        expiries          : 1-D array of expiries in years (n_expiries,)
        market_vol_surface: 2-D array (n_expiries, n_strikes) of implied vols
        F                 : forward price (assumed constant here for simplicity)

        Returns
        -------
        List of SVIParams, one per expiry slice.
        """
        strikes = np.asarray(strikes, dtype=float)
        expiries = np.asarray(expiries, dtype=float)
        market_vol_surface = np.asarray(market_vol_surface, dtype=float)

        n_exp, n_str = market_vol_surface.shape
        if len(expiries) != n_exp or len(strikes) != n_str:
            raise ValueError("market_vol_surface shape must be (n_expiries, n_strikes)")

        result: List[SVIParams] = []
        for i, T in enumerate(expiries):
            params = self.fit(strikes, market_vol_surface[i], F, T)
            result.append(params)
        return result


# ---------------------------------------------------------------------------
# IV Term Structure
# ---------------------------------------------------------------------------

class IVTermStructure:
    """ATM implied-volatility term structure.

    Stores observed (expiry, atm_vol) pairs and provides:
    - Nelson-Siegel fitting
    - Interpolation via cubic spline
    - Forward vol calculation
    - Term-structure diagnostics
    """

    def __init__(self, expiries: List[float], atm_vols: List[float]) -> None:
        self.expiries = np.asarray(expiries, dtype=float)
        self.atm_vols = np.asarray(atm_vols, dtype=float)
        if len(self.expiries) != len(self.atm_vols):
            raise ValueError("expiries and atm_vols must have equal length")
        if np.any(self.expiries <= 0):
            raise ValueError("All expiries must be positive")
        # sort by expiry
        idx = np.argsort(self.expiries)
        self.expiries = self.expiries[idx]
        self.atm_vols = self.atm_vols[idx]
        # build interpolator
        self._interp = interp1d(
            self.expiries, self.atm_vols,
            kind="cubic", bounds_error=False,
            fill_value=(self.atm_vols[0], self.atm_vols[-1]),
        )
        self._ns_params: Optional[NelsonSiegelParams] = None

    def fit_nelson_siegel(self) -> NelsonSiegelParams:
        """Fit Nelson-Siegel model to observed ATM vols.

        Returns
        -------
        Fitted NelsonSiegelParams.
        """
        expiries = self.expiries
        vols = self.atm_vols

        def ns_model(T: np.ndarray, beta0: float, beta1: float, beta2: float, tau: float) -> np.ndarray:
            tau = max(tau, _EPS)
            ratio = T / tau
            loading1 = (1.0 - np.exp(-ratio)) / ratio
            loading2 = loading1 - np.exp(-ratio)
            return beta0 + beta1 * loading1 + beta2 * loading2

        p0 = [vols[-1], vols[0] - vols[-1], 0.0, 1.0]
        bounds_low = [0.0, -2.0, -2.0, 0.01]
        bounds_high = [2.0, 2.0, 2.0, 20.0]

        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    ns_model, expiries, vols,
                    p0=p0, bounds=(bounds_low, bounds_high),
                    maxfev=10000,
                )
            ns = NelsonSiegelParams(beta0=popt[0], beta1=popt[1], beta2=popt[2], tau=popt[3])
        except Exception:
            # Fallback: least-squares minimization
            def obj(x: np.ndarray) -> float:
                try:
                    v = ns_model(expiries, x[0], x[1], x[2], max(x[3], _EPS))
                    return float(np.sum((v - vols) ** 2))
                except Exception:
                    return 1e9

            res = minimize(obj, p0, method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-10})
            x = res.x
            ns = NelsonSiegelParams(beta0=x[0], beta1=x[1], beta2=x[2], tau=max(x[3], _EPS))

        self._ns_params = ns
        return ns

    def interpolate(self, T: float) -> float:
        """Interpolate ATM vol at expiry T (cubic spline on observed points)."""
        return float(self._interp(T))

    def forward_vol(self, T1: float, T2: float) -> float:
        """Compute the forward implied vol between T1 and T2.

        sigma_fwd = sqrt((T2*sigma(T2)^2 - T1*sigma(T1)^2) / (T2 - T1))
        """
        if T2 <= T1:
            raise ValueError("T2 must be greater than T1")
        s1 = self.interpolate(T1)
        s2 = self.interpolate(T2)
        num = T2 * s2 ** 2 - T1 * s1 ** 2
        denom = T2 - T1
        if num < 0:
            raise ValueError(
                f"Calendar arbitrage: T2*sigma2^2 < T1*sigma1^2 at T1={T1}, T2={T2}"
            )
        return math.sqrt(num / denom)

    def term_structure_slope(self) -> float:
        """Linear slope of ATM vol vs expiry (via OLS)."""
        if len(self.expiries) < 2:
            return 0.0
        x = self.expiries
        y = self.atm_vols
        x_mean = x.mean()
        y_mean = y.mean()
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))
        return slope

    def vol_of_vol(self) -> float:
        """Standard deviation of first differences in ATM vols."""
        if len(self.atm_vols) < 2:
            return 0.0
        diffs = np.diff(self.atm_vols)
        return float(np.std(diffs, ddof=0))


# ---------------------------------------------------------------------------
# IV Surface
# ---------------------------------------------------------------------------

class IVSurface:
    """2-D implied volatility surface over (strikes, expiries).

    Uses scipy.interpolate.RectBivariateSpline for smooth interpolation.
    """

    def __init__(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        vol_surface: np.ndarray,
    ) -> None:
        self.strikes = np.asarray(strikes, dtype=float)
        self.expiries = np.asarray(expiries, dtype=float)
        self.vol_surface = np.asarray(vol_surface, dtype=float)

        # Sort axes
        kidx = np.argsort(self.strikes)
        tidx = np.argsort(self.expiries)
        self.strikes = self.strikes[kidx]
        self.expiries = self.expiries[tidx]
        self.vol_surface = self.vol_surface[np.ix_(tidx, kidx)]

        n_exp, n_str = self.vol_surface.shape
        if len(self.expiries) != n_exp or len(self.strikes) != n_str:
            raise ValueError("vol_surface shape must be (n_expiries, n_strikes)")

        # Spline degree: use kx=ky=3 but fall back to 1 for small grids
        kx = min(3, n_exp - 1)
        ky = min(3, n_str - 1)
        self._spline = RectBivariateSpline(
            self.expiries, self.strikes, self.vol_surface, kx=kx, ky=ky
        )

    def interpolate(self, K: float, T: float) -> float:
        """Bivariate-spline interpolation at (K, T)."""
        return float(self._spline(T, K))

    def smile_at_expiry(self, T: float) -> np.ndarray:
        """Return vol smile (array over self.strikes) at expiry T."""
        return self._spline(T, self.strikes).flatten()

    def term_structure_at_strike(self, K: float) -> np.ndarray:
        """Return vol term structure (array over self.expiries) at strike K."""
        return self._spline(self.expiries, K).flatten()

    def is_arbitrage_free(self) -> bool:
        """Check for calendar-spread and butterfly arbitrage.

        Calendar spread: total variance must be non-decreasing in T for each K.
        Butterfly: vol smile must be convex in K at each expiry (second derivative >= 0
                   for implied vol, approximately).
        """
        # 1. Calendar spread: total variance non-decreasing in T
        for j in range(len(self.strikes)):
            for i in range(len(self.expiries) - 1):
                T1, T2 = self.expiries[i], self.expiries[i + 1]
                v1 = self.vol_surface[i, j] ** 2 * T1
                v2 = self.vol_surface[i + 1, j] ** 2 * T2
                if v2 < v1 - 1e-6:
                    return False

        # 2. Butterfly (convexity in K): for each expiry, check that there is no
        #    concavity in total-variance w(k) using finite differences on fine grid
        K_fine = np.linspace(self.strikes[0], self.strikes[-1], 50)
        for i, T in enumerate(self.expiries):
            smile = self._spline(T, K_fine).flatten()
            w = smile ** 2 * T
            # second finite difference must be >= 0 (total variance convex in K)
            d2w = np.diff(w, 2)
            if np.any(d2w < -1e-4):
                return False

        return True


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def fit_svi(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    F: float,
    T: float,
) -> SVIParams:
    """Convenience wrapper: calibrate SVI to a single expiry smile."""
    return SVICalibrator().fit(strikes, market_vols, F, T)


def fit_nelson_siegel(
    expiries: np.ndarray,
    atm_vols: np.ndarray,
) -> NelsonSiegelParams:
    """Convenience wrapper: fit Nelson-Siegel to ATM vol term structure."""
    ts = IVTermStructure(list(expiries), list(atm_vols))
    return ts.fit_nelson_siegel()


def forward_vol(T1: float, T2: float, sigma1: float, sigma2: float) -> float:
    """Forward implied vol between T1 and T2.

    sigma_fwd = sqrt((T2*sigma2^2 - T1*sigma1^2) / (T2 - T1))
    """
    if T2 <= T1:
        raise ValueError("T2 must be greater than T1")
    num = T2 * sigma2 ** 2 - T1 * sigma1 ** 2
    if num < 0:
        raise ValueError(f"Calendar arbitrage detected: num={num:.6f}")
    return math.sqrt(num / (T2 - T1))


def iv_surface_from_svi(
    strikes: np.ndarray,
    expiries: np.ndarray,
    svi_params_list: List[SVIParams],
    F: float,
) -> np.ndarray:
    """Build an implied-vol surface (n_expiries x n_strikes) from per-slice SVI params.

    Parameters
    ----------
    strikes         : 1-D array of strike prices
    expiries        : 1-D array of expiries (must match len(svi_params_list))
    svi_params_list : list of SVIParams, one per expiry
    F               : forward price

    Returns
    -------
    vol_surface : ndarray of shape (n_expiries, n_strikes)
    """
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    if len(expiries) != len(svi_params_list):
        raise ValueError("len(expiries) must equal len(svi_params_list)")

    n_exp = len(expiries)
    n_str = len(strikes)
    surface = np.empty((n_exp, n_str), dtype=float)

    for i, (T, params) in enumerate(zip(expiries, svi_params_list)):
        for j, K in enumerate(strikes):
            k = math.log(K / F)
            surface[i, j] = params.implied_vol(k, T)

    return surface
