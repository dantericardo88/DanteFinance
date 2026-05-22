"""
GARCH(1,1) and EWMA volatility forecasting — pure numpy/scipy, zero external deps.

dim_114 — GARCH / EWMA volatility forecasting (target: 9)

Classes
-------
GARCHParams
    Fitted GARCH(1,1) parameter container with derived properties.

GARCH11
    GARCH(1,1) via MLE (scipy.optimize.minimize, L-BFGS-B).
    .fit()       → GARCHParams
    .filter()    → conditional volatility array
    .forecast()  → h-step ahead variance array
    .simulate()  → synthetic return series

EGARCH11
    EGARCH(1,1) via MLE — asymmetric / leverage effect.

EWMA
    RiskMetrics exponentially weighted moving average.
    .fit()      → conditional volatility array
    .forecast() → h-step ahead variance array

VolForecastEnsemble
    Weighted combination of GARCH + EWMA.

Convenience functions
---------------------
fit_garch        Fit GARCH(1,1) and return GARCHParams.
ewma_vol         EWMA volatility series.
garch_forecast   h-step ahead variance forecast.
vol_cone         Rolling percentile vol surface.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GARCHParams:
    """Fitted GARCH(1,1) parameters."""
    omega: float
    alpha: float
    beta: float
    log_likelihood: float = 0.0
    converged: bool = True

    @property
    def persistence(self) -> float:
        """alpha + beta (< 1 required for covariance-stationarity)."""
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        """Unconditional (long-run) volatility = sqrt(omega / (1 - alpha - beta))."""
        denom = 1.0 - self.alpha - self.beta
        if denom <= 0:
            return float("nan")
        return float(np.sqrt(self.omega / denom))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GARCHParams(omega={self.omega:.6e}, alpha={self.alpha:.4f}, "
            f"beta={self.beta:.4f}, persistence={self.persistence:.4f}, "
            f"long_run_vol={self.long_run_vol:.6f}, ll={self.log_likelihood:.2f}, "
            f"converged={self.converged})"
        )


@dataclass
class EGARCHParams:
    """Fitted EGARCH(1,1) parameters."""
    omega: float
    alpha: float  # magnitude effect
    gamma: float  # asymmetry / leverage
    beta: float
    log_likelihood: float = 0.0
    converged: bool = True

    @property
    def persistence(self) -> float:
        return abs(self.beta)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EGARCHParams(omega={self.omega:.4f}, alpha={self.alpha:.4f}, "
            f"gamma={self.gamma:.4f}, beta={self.beta:.4f}, "
            f"ll={self.log_likelihood:.2f}, converged={self.converged})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# GARCH(1,1) — MLE via scipy
# ──────────────────────────────────────────────────────────────────────────────

class GARCH11:
    """
    GARCH(1,1) model fitted via maximum likelihood estimation.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Log-likelihood (Gaussian):
        L = -0.5 * sum( log(sigma_t^2) + r_t^2 / sigma_t^2 )
    """

    def __init__(self) -> None:
        self.params_: Optional[GARCHParams] = None

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _variance_filter(
        returns: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """Compute conditional variance series given parameters."""
        n = len(returns)
        var = np.empty(n)
        # Initialise with sample variance
        var[0] = float(np.var(returns))
        for t in range(1, n):
            var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
        return var

    @staticmethod
    def _neg_log_likelihood(
        params: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """Negative Gaussian log-likelihood (to minimise)."""
        omega, alpha, beta = params
        # Hard penalty for invalid parameters
        if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1.0:
            return 1e10
        n = len(returns)
        var = np.empty(n)
        var[0] = float(np.var(returns)) if np.var(returns) > 0 else 1e-6
        for t in range(1, n):
            v = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
            var[t] = v if v > 1e-12 else 1e-12
        # Guard against non-positive variance
        if np.any(var <= 0):
            return 1e10
        ll = -0.5 * float(np.sum(np.log(var) + returns ** 2 / var))
        return -ll  # negative because we minimise

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, returns: np.ndarray) -> GARCHParams:
        """
        Fit GARCH(1,1) parameters via MLE.

        Parameters
        ----------
        returns : 1-D array of log-returns (demeaned internally).

        Returns
        -------
        GARCHParams
        """
        returns = np.asarray(returns, dtype=float)
        # Demean
        returns = returns - returns.mean()

        sample_var = float(np.var(returns))
        if sample_var <= 0:
            sample_var = 1e-6

        # Starting values: omega = 5% of variance, alpha=0.09, beta=0.90
        x0 = np.array([0.05 * sample_var, 0.09, 0.90])

        bounds = [(1e-8, None), (1e-6, 0.9999), (1e-6, 0.9999)]

        # Try multiple starting points for robustness
        best_result = None
        best_nll = np.inf

        start_configs = [
            np.array([0.05 * sample_var, 0.09, 0.90]),
            np.array([0.10 * sample_var, 0.05, 0.93]),
            np.array([0.20 * sample_var, 0.15, 0.80]),
            np.array([sample_var * (1 - 0.1 - 0.85), 0.10, 0.85]),
        ]

        for x_start in start_configs:
            # Clip starting values to feasible range
            x_start[0] = max(x_start[0], 1e-8)
            x_start[1] = float(np.clip(x_start[1], 1e-6, 0.9999))
            x_start[2] = float(np.clip(x_start[2], 1e-6, 0.9999))
            if x_start[1] + x_start[2] >= 1.0:
                x_start[2] = 0.99 - x_start[1]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = minimize(
                        self._neg_log_likelihood,
                        x_start,
                        args=(returns,),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8},
                    )
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_result = res
            except Exception:
                continue

        if best_result is None or not np.isfinite(best_result.fun):
            # Fallback: use method-of-moments estimates
            alpha_mm = 0.10
            beta_mm = 0.85
            omega_mm = sample_var * (1 - alpha_mm - beta_mm)
            self.params_ = GARCHParams(
                omega=omega_mm,
                alpha=alpha_mm,
                beta=beta_mm,
                log_likelihood=-best_nll if (best_nll != np.inf) else 0.0,
                converged=False,
            )
            return self.params_

        omega, alpha, beta = best_result.x
        # Enforce stationarity — if optimizer violated it slightly, project
        if alpha + beta >= 1.0:
            scale = 0.999 / (alpha + beta)
            alpha *= scale
            beta *= scale

        self.params_ = GARCHParams(
            omega=float(omega),
            alpha=float(alpha),
            beta=float(beta),
            log_likelihood=float(-best_result.fun),
            converged=bool(best_result.success),
        )
        return self.params_

    def filter(
        self,
        returns: np.ndarray,
        params: Optional[GARCHParams] = None,
    ) -> np.ndarray:
        """
        Return conditional volatility series (annualised if desired — raw here).

        Parameters
        ----------
        returns : 1-D return array.
        params  : GARCHParams (uses fitted params if None).

        Returns
        -------
        np.ndarray of conditional volatilities (same length as returns).
        """
        p = params or self.params_
        if p is None:
            raise RuntimeError("Call fit() first or supply params.")
        returns = np.asarray(returns, dtype=float)
        var = self._variance_filter(returns - returns.mean(), p.omega, p.alpha, p.beta)
        return np.sqrt(np.maximum(var, 0.0))

    def forecast(
        self,
        params: Optional[GARCHParams] = None,
        current_var: Optional[float] = None,
        h: int = 10,
    ) -> np.ndarray:
        """
        Multi-step ahead variance forecast (mean-reverting GARCH formula).

        sigma_{t+k}^2 = sigma_LR^2 + (alpha+beta)^k * (sigma_t^2 - sigma_LR^2)

        Parameters
        ----------
        params      : GARCHParams.
        current_var : Current conditional variance (sigma_t^2).
        h           : Forecast horizon (steps).

        Returns
        -------
        np.ndarray of shape (h,) — variance forecasts for t+1 … t+h.
        """
        p = params or self.params_
        if p is None:
            raise RuntimeError("Call fit() first or supply params.")
        if current_var is None:
            current_var = p.long_run_vol ** 2

        lr_var = p.long_run_vol ** 2
        persistence = p.persistence
        horizons = np.arange(1, h + 1)
        forecasts = lr_var + (persistence ** horizons) * (current_var - lr_var)
        return np.maximum(forecasts, 0.0)

    def simulate(
        self,
        params: Optional[GARCHParams] = None,
        n: int = 252,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Simulate a GARCH(1,1) return series.

        Returns
        -------
        np.ndarray of shape (n,) — simulated returns.
        """
        p = params or self.params_
        if p is None:
            raise RuntimeError("Call fit() first or supply params.")
        rng = np.random.default_rng(seed)
        returns = np.empty(n)
        var = p.long_run_vol ** 2  # initial variance
        for t in range(n):
            eps = rng.standard_normal()
            returns[t] = np.sqrt(max(var, 0.0)) * eps
            var = p.omega + p.alpha * returns[t] ** 2 + p.beta * var
        return returns


# ──────────────────────────────────────────────────────────────────────────────
# EGARCH(1,1)
# ──────────────────────────────────────────────────────────────────────────────

_E_ABS_Z = float(np.sqrt(2.0 / np.pi))  # E[|z|] for standard normal


class EGARCH11:
    """
    EGARCH(1,1) — Nelson (1991).

    log(sigma_t^2) = omega + alpha*(|z_{t-1}| - E[|z|]) + gamma*z_{t-1}
                            + beta*log(sigma_{t-1}^2)

    where z_t = r_t / sigma_t and E[|z|] = sqrt(2/pi).

    Captures asymmetric (leverage) effects.  beta is unconstrained in theory
    but we bound |beta| < 1 for stationarity.
    """

    def __init__(self) -> None:
        self.params_: Optional[EGARCHParams] = None

    @staticmethod
    def _log_var_filter(
        returns: np.ndarray,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
    ) -> np.ndarray:
        n = len(returns)
        log_var = np.empty(n)
        # Initialise at sample log-variance
        sv = max(float(np.var(returns)), 1e-10)
        log_var[0] = np.log(sv)

        for t in range(1, n):
            sigma_prev = np.exp(0.5 * log_var[t - 1])
            z_prev = returns[t - 1] / (sigma_prev + 1e-12)
            log_var[t] = (
                omega
                + alpha * (abs(z_prev) - _E_ABS_Z)
                + gamma * z_prev
                + beta * log_var[t - 1]
            )
        return log_var

    @staticmethod
    def _neg_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, gamma, beta = params
        if abs(beta) >= 1.0:
            return 1e10
        n = len(returns)
        log_var = np.empty(n)
        sv = max(float(np.var(returns)), 1e-10)
        log_var[0] = np.log(sv)
        for t in range(1, n):
            sigma_prev = np.exp(0.5 * log_var[t - 1])
            z_prev = returns[t - 1] / (sigma_prev + 1e-12)
            log_var[t] = (
                omega
                + alpha * (abs(z_prev) - _E_ABS_Z)
                + gamma * z_prev
                + beta * log_var[t - 1]
            )
        var = np.exp(log_var)
        if not np.all(np.isfinite(var)) or np.any(var <= 0):
            return 1e10
        ll = -0.5 * float(np.sum(np.log(var) + returns ** 2 / var))
        return -ll

    def fit(self, returns: np.ndarray) -> EGARCHParams:
        returns = np.asarray(returns, dtype=float)
        returns = returns - returns.mean()
        sv = float(np.var(returns))
        x0 = np.array([np.log(sv) * (1 - 0.9), 0.1, -0.05, 0.9])
        bounds = [(-10, 10), (-2, 2), (-2, 2), (-0.999, 0.999)]

        best_res = None
        best_nll = np.inf
        for b_init in [0.9, 0.8, 0.95]:
            x_try = np.array([np.log(sv) * (1 - b_init), 0.1, -0.05, b_init])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = minimize(
                        self._neg_log_likelihood,
                        x_try,
                        args=(returns,),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 2000, "ftol": 1e-10},
                    )
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_res = res
            except Exception:
                continue

        if best_res is None or not np.isfinite(best_res.fun):
            self.params_ = EGARCHParams(
                omega=float(x0[0]),
                alpha=float(x0[1]),
                gamma=float(x0[2]),
                beta=float(x0[3]),
                log_likelihood=0.0,
                converged=False,
            )
            return self.params_

        omega, alpha, gamma, beta = best_res.x
        self.params_ = EGARCHParams(
            omega=float(omega),
            alpha=float(alpha),
            gamma=float(gamma),
            beta=float(beta),
            log_likelihood=float(-best_res.fun),
            converged=bool(best_res.success),
        )
        return self.params_

    def filter(
        self,
        returns: np.ndarray,
        params: Optional[EGARCHParams] = None,
    ) -> np.ndarray:
        p = params or self.params_
        if p is None:
            raise RuntimeError("Call fit() first or supply params.")
        returns = np.asarray(returns, dtype=float)
        log_var = self._log_var_filter(
            returns - returns.mean(), p.omega, p.alpha, p.gamma, p.beta
        )
        return np.sqrt(np.exp(log_var))


# ──────────────────────────────────────────────────────────────────────────────
# EWMA (RiskMetrics)
# ──────────────────────────────────────────────────────────────────────────────

class EWMA:
    """
    Exponentially Weighted Moving Average (RiskMetrics) volatility.

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2

    Typical: lambda=0.94 (daily), lambda=0.97 (monthly).
    """

    def __init__(self, lam: float = 0.94) -> None:
        if not (0 < lam < 1):
            raise ValueError(f"lambda must be in (0, 1), got {lam}")
        self.lam = lam
        self._last_var: Optional[float] = None

    def fit(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute EWMA conditional volatility series.

        Parameters
        ----------
        returns : 1-D return array.

        Returns
        -------
        np.ndarray of conditional volatilities (same length as returns).
        """
        returns = np.asarray(returns, dtype=float)
        n = len(returns)
        var = np.empty(n)
        # Initialise with sample variance
        var[0] = float(np.var(returns))
        for t in range(1, n):
            var[t] = self.lam * var[t - 1] + (1.0 - self.lam) * returns[t - 1] ** 2
        self._last_var = float(var[-1])
        return np.sqrt(np.maximum(var, 0.0))

    def forecast(
        self,
        current_var: Optional[float] = None,
        h: int = 10,
    ) -> np.ndarray:
        """
        h-step ahead EWMA variance forecast.

        Under EWMA (I-GARCH), the forecast is flat: sigma_{t+h}^2 = sigma_t^2.

        Returns
        -------
        np.ndarray of shape (h,) — variance forecasts.
        """
        cv = current_var if current_var is not None else self._last_var
        if cv is None:
            raise RuntimeError("Call fit() first or supply current_var.")
        return np.full(h, cv)


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble
# ──────────────────────────────────────────────────────────────────────────────

class VolForecastEnsemble:
    """
    Weighted combination of GARCH(1,1) + EWMA volatility forecasts.

    Default weights: GARCH 60%, EWMA 40%.
    """

    def __init__(
        self,
        garch_weight: float = 0.60,
        ewma_lam: float = 0.94,
    ) -> None:
        if not (0.0 <= garch_weight <= 1.0):
            raise ValueError("garch_weight must be in [0, 1]")
        self.garch_weight = garch_weight
        self.ewma_weight = 1.0 - garch_weight
        self._garch = GARCH11()
        self._ewma = EWMA(lam=ewma_lam)
        self._garch_params: Optional[GARCHParams] = None
        self._last_var: Optional[float] = None
        self._ewma_last_var: Optional[float] = None

    def fit(self, returns: np.ndarray) -> "VolForecastEnsemble":
        """Fit both GARCH and EWMA on the same return series."""
        returns = np.asarray(returns, dtype=float)
        self._garch_params = self._garch.fit(returns)
        ewma_vols = self._ewma.fit(returns)
        # Store last conditional variance from each model
        garch_vols = self._garch.filter(returns, self._garch_params)
        self._last_var = float(garch_vols[-1] ** 2)
        self._ewma_last_var = float(ewma_vols[-1] ** 2)
        return self

    def forecast(self, h: int = 10) -> Dict[str, np.ndarray]:
        """
        Return h-step ahead vol forecasts for each model and ensemble.

        Returns
        -------
        dict with keys 'garch', 'ewma', 'ensemble' — each a numpy array of
        volatilities (not variances) of length h.
        """
        if self._garch_params is None:
            raise RuntimeError("Call fit() before forecast().")

        garch_var_fc = self._garch.forecast(
            params=self._garch_params, current_var=self._last_var, h=h
        )
        ewma_var_fc = self._ewma.forecast(current_var=self._ewma_last_var, h=h)

        garch_vol_fc = np.sqrt(np.maximum(garch_var_fc, 0.0))
        ewma_vol_fc = np.sqrt(np.maximum(ewma_var_fc, 0.0))
        ensemble_vol_fc = self.garch_weight * garch_vol_fc + self.ewma_weight * ewma_vol_fc

        return {
            "garch": garch_vol_fc,
            "ewma": ewma_vol_fc,
            "ensemble": ensemble_vol_fc,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────────────────────────

def fit_garch(returns: np.ndarray) -> GARCHParams:
    """Fit GARCH(1,1) on *returns* and return the fitted GARCHParams."""
    model = GARCH11()
    return model.fit(returns)


def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    Compute EWMA conditional volatility series.

    Returns
    -------
    np.ndarray of conditional volatilities.
    """
    return EWMA(lam=lam).fit(returns)


def garch_forecast(
    params: GARCHParams,
    current_var: float,
    h: int = 10,
) -> np.ndarray:
    """
    h-step ahead GARCH variance forecast using mean-reversion formula.

    Returns
    -------
    np.ndarray of shape (h,) — variance forecasts.
    """
    return GARCH11().forecast(params=params, current_var=current_var, h=h)


def vol_cone(
    returns: np.ndarray,
    windows: List[int] = [5, 10, 21, 63, 252],
) -> Dict[str, Dict[str, float]]:
    """
    Compute rolling realised-volatility percentile cone.

    For each window w in *windows*, computes the annualised realised vol
    over each w-day sub-period, then returns summary statistics.

    Returns
    -------
    dict keyed by '{w}d', each containing:
        mean, median, p25, p75, p90, current
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    result: Dict[str, Dict[str, float]] = {}

    for w in windows:
        if w > n:
            continue
        key = f"{w}d"
        # Rolling window realised vol
        rolling_vols: List[float] = []
        for i in range(w, n + 1):
            window_returns = returns[i - w : i]
            rv = float(np.std(window_returns, ddof=1)) * np.sqrt(252)
            rolling_vols.append(rv)

        if not rolling_vols:
            continue

        arr = np.array(rolling_vols)
        result[key] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "current": float(arr[-1]),
        }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI router (optional — only if fastapi is installed)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from fastapi import APIRouter
    from pydantic import BaseModel

    garch_vol_router = APIRouter(prefix="/garch-vol", tags=["garch-vol"])

    class _FitRequest(BaseModel):
        returns: List[float]
        lam: float = 0.94
        h: int = 10

    class _FitResponse(BaseModel):
        omega: float
        alpha: float
        beta: float
        persistence: float
        long_run_vol: float
        log_likelihood: float
        converged: bool
        garch_forecast: List[float]
        ewma_vol_last: float
        ensemble_forecast: List[float]

    @garch_vol_router.post("/fit", response_model=_FitResponse)
    def _fit_endpoint(req: _FitRequest) -> _FitResponse:
        r = np.array(req.returns)
        params = fit_garch(r)
        ewma = EWMA(lam=req.lam)
        ewma_vols = ewma.fit(r)
        g_fc = garch_forecast(params, float(ewma_vols[-1] ** 2), h=req.h)
        ensemble = VolForecastEnsemble(ewma_lam=req.lam)
        ensemble.fit(r)
        ens_fc = ensemble.forecast(h=req.h)
        return _FitResponse(
            omega=params.omega,
            alpha=params.alpha,
            beta=params.beta,
            persistence=params.persistence,
            long_run_vol=params.long_run_vol,
            log_likelihood=params.log_likelihood,
            converged=params.converged,
            garch_forecast=list(g_fc),
            ewma_vol_last=float(ewma_vols[-1]),
            ensemble_forecast=list(ens_fc["ensemble"]),
        )

except ImportError:
    pass  # FastAPI not available — module still fully functional
