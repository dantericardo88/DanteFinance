"""
sentinel/spm/portfolio_optimizer_v3.py
=======================================
Comprehensive portfolio optimization engine -- dim_081 (score 8 -> 9).

Methods implemented:
  - Mean-Variance (Markowitz) with robust covariance estimation
  - Black-Litterman with proper tau calibration and posterior computation
  - Hierarchical Risk Parity (Lopez de Prado 2016)
  - Minimum CVaR / ES (Rockafellar-Uryasev 2000) -- robust to fat tails
  - Ledoit-Wolf shrinkage (OAS estimator + Marchenko-Pastur denoising)
  - Maximum Diversification (Choueifnaour & Coignard 2008)
  - Risk Budgeting with arbitrary target risk contributions (ERC)
  - Transaction cost-aware optimization (turnover constraints)
  - Resampled Efficient Frontier (Michaud 1989)

Free data sources only:
  - yfinance (guarded) for price history
  - FRED CSV for risk-free rate (TB3MS)
  - numpy / scipy (scipy guarded)

Author: SENTINEL Portfolio Engine
"""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional / guarded imports
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import linprog, minimize
    from scipy.stats import norm

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    _SCIPY_CLUSTER = True
except ImportError:
    _SCIPY_CLUSTER = False

try:
    import yfinance as yf

    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    import requests

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from sklearn.covariance import LedoitWolf as SklearnLW  # type: ignore

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class View:
    """A Black-Litterman investor view."""

    assets: List[str]
    weights: List[float]  # must sum to 0 for relative; 1 for absolute
    expected_return: float  # annualized expected return for this view
    confidence: float = 0.5  # 0 = no confidence, 1 = certainty
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.assets) != len(self.weights):
            raise ValueError("assets and weights must have the same length")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class OptResult:
    """Optimization result container."""

    weights: np.ndarray
    method: str
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_ratio: float = 1.0
    risk_contributions: Optional[np.ndarray] = None
    asset_names: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        idx = self.asset_names or list(range(len(self.weights)))
        return pd.Series(self.weights, index=idx, name=self.method)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "weights": dict(zip(self.asset_names or [], self.weights.tolist())),
            "expected_return": round(self.expected_return, 6),
            "expected_volatility": round(self.expected_volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
        }


@dataclass
class PortfolioMetrics:
    """Portfolio performance/risk metrics."""

    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    diversification_ratio: float
    effective_n: float  # 1/HHI of weights
    risk_contributions: np.ndarray
    beta: float = 0.0


# ---------------------------------------------------------------------------
# Covariance Estimation
# ---------------------------------------------------------------------------


class CovarianceEstimator:
    """
    Robust covariance matrix estimation methods.
    All methods return an (N x N) numpy array.
    """

    @staticmethod
    def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
        """Standard sample covariance matrix (biased MLE estimator)."""
        return returns.cov().values

    @staticmethod
    def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
        """
        Oracle Approximating Shrinkage (OAS) estimator.
        Target: constant-correlation / diagonal structure.

        Σ_LW = (1-α) x Σ_sample + α x Σ_target

        Uses sklearn if available, else analytical Ledoit-Wolf (2004).
        """
        X = returns.values
        T, N = X.shape

        if _SKLEARN_AVAILABLE:
            lw = SklearnLW(assume_centered=False)
            lw.fit(X)
            return lw.covariance_

        # Analytical Ledoit-Wolf shrinkage (Ledoit & Wolf 2004, JMVA)
        X_c = X - X.mean(axis=0)
        S = (X_c.T @ X_c) / T  # biased sample cov

        # Target: scaled identity (variance-weighted)
        mu_target = np.trace(S) / N  # mean eigenvalue
        F = mu_target * np.eye(N)  # target matrix

        # Compute shrinkage intensity analytically
        # rho = sum_i sum_j Var(s_ij) / ||S - F||_F^2
        # Simplified formula (Ledoit-Wolf oracle approximation):
        delta2 = 0.0
        for i in range(T):
            xi = X_c[i]
            outer = np.outer(xi, xi)
            delta2 += np.sum((outer - S) ** 2)
        delta2 /= T**2

        norm_sq = np.sum((S - F) ** 2)
        if norm_sq < 1e-12:
            alpha = 0.0
        else:
            alpha = min(1.0, max(0.0, delta2 / norm_sq))

        cov_lw = (1.0 - alpha) * S + alpha * F
        return cov_lw

    @staticmethod
    def constant_correlation_shrinkage(returns: pd.DataFrame) -> np.ndarray:
        """
        Ledoit-Wolf (2003) constant-correlation target.
        Target: all off-diagonal correlations equal to mean correlation.
        """
        X = returns.values
        T, N = X.shape
        X_c = X - X.mean(axis=0)
        S = (X_c.T @ X_c) / T

        # Standard deviations
        std = np.sqrt(np.diag(S))
        std_outer = np.outer(std, std)
        corr = S / (std_outer + 1e-12)

        # Mean correlation (off-diagonal)
        mask = ~np.eye(N, dtype=bool)
        rho_bar = corr[mask].mean()

        # Constant correlation target
        F = rho_bar * std_outer
        np.fill_diagonal(F, np.diag(S))

        # Shrinkage intensity (simplified Ledoit-Wolf constant-correlation formula)
        # Numerator: sum of variances of s_ij estimates
        alpha_sum = 0.0
        for i in range(T):
            xi = X_c[i]
            outer = np.outer(xi, xi)
            alpha_sum += np.sum((outer - S) ** 2)
        alpha_sum /= T**2

        denom = np.sum((S - F) ** 2)
        if denom < 1e-12:
            alpha = 0.0
        else:
            alpha = min(1.0, max(0.0, alpha_sum / denom))

        return (1.0 - alpha) * S + alpha * F

    @staticmethod
    def exponential_weighted_covariance(
        returns: pd.DataFrame, lambda_: float = 0.94
    ) -> np.ndarray:
        """
        RiskMetrics EWMA covariance: Σ_t = λΣ_{t-1} + (1-λ) r_t r_t'
        lambda_ = 0.94 is the RiskMetrics daily decay factor.
        """
        X = returns.values
        T, N = X.shape

        # Initialize with sample covariance of first 21 observations
        init_obs = min(21, T // 5)
        Sigma = np.cov(X[:init_obs].T, bias=False) if init_obs > 1 else np.eye(N) * 1e-4

        for t in range(init_obs, T):
            r = X[t].reshape(-1, 1)
            Sigma = lambda_ * Sigma + (1.0 - lambda_) * (r @ r.T)

        return Sigma

    @staticmethod
    def denoised_covariance(returns: pd.DataFrame) -> np.ndarray:
        """
        Marchenko-Pastur denoising of the covariance matrix.

        Remove eigenvalues within the Marchenko-Pastur distribution (noise).
        λ_max = σ² x (1 + sqrt(N/T))²

        Steps:
          1. Compute correlation matrix
          2. Eigendecompose
          3. Identify noise eigenvalues (< λ_max_MP)
          4. Replace noise eigenvalues with their mean
          5. Reconstruct denoised covariance
        """
        X = returns.values
        T, N = X.shape
        S = np.cov(X.T, bias=False)

        # Correlation matrix
        std = np.sqrt(np.diag(S))
        std_outer = np.outer(std, std)
        corr = S / (std_outer + 1e-12)
        np.fill_diagonal(corr, 1.0)

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(corr)

        # Marchenko-Pastur upper bound (assuming variance σ² ≈ 1 for correlation matrix)
        q = T / N  # ratio T/N
        sigma_sq = 1.0  # correlation matrix trace/N = 1
        lambda_max_mp = sigma_sq * (1.0 + np.sqrt(1.0 / q)) ** 2

        # Identify noise vs. signal eigenvalues
        noise_mask = eigvals <= lambda_max_mp

        if noise_mask.sum() == N:
            # All noise -- return sample cov
            return S

        # Mean of noise eigenvalues
        noise_mean = eigvals[noise_mask].mean() if noise_mask.any() else 1e-4

        # Replace noise eigenvalues with their mean
        eigvals_denoised = eigvals.copy()
        eigvals_denoised[noise_mask] = noise_mean

        # Reconstruct denoised correlation matrix
        corr_denoised = eigvecs @ np.diag(eigvals_denoised) @ eigvecs.T
        np.fill_diagonal(corr_denoised, 1.0)

        # Rescale back to covariance
        cov_denoised = corr_denoised * std_outer
        return cov_denoised


# ---------------------------------------------------------------------------
# Mean Estimation
# ---------------------------------------------------------------------------


class MeanEstimator:
    """Robust expected return estimation methods."""

    @staticmethod
    def sample_mean(returns: pd.DataFrame) -> np.ndarray:
        """Simple historical mean (annualized assuming daily returns)."""
        mu = returns.mean().values * 252
        return mu

    @staticmethod
    def ewma_mean(returns: pd.DataFrame, lambda_: float = 0.94) -> np.ndarray:
        """Exponentially weighted mean return (annualized)."""
        X = returns.values
        T, N = X.shape
        weights = np.array([(1 - lambda_) * lambda_**i for i in range(T - 1, -1, -1)])
        weights /= weights.sum()
        mu = (weights[:, None] * X).sum(axis=0) * 252
        return mu

    @staticmethod
    def james_stein_shrinkage(returns: pd.DataFrame) -> np.ndarray:
        """
        James-Stein shrinkage estimator for expected returns.
        Shrinks each asset's mean toward the grand mean (constant).

        μ_JS = (1 - b) x μ_sample + b x μ_grand
        b = (N - 2) / (N - 2 + T x ||μ - μ_grand||²_Σ^{-1})
        """
        X = returns.values
        T, N = X.shape
        mu = X.mean(axis=0)
        grand_mean = mu.mean()
        mu_grand = np.full(N, grand_mean)

        if N <= 2:
            return mu * 252

        # Estimate covariance for Mahalanobis distance
        S = np.cov(X.T, bias=False) + np.eye(N) * 1e-8

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        diff = mu - mu_grand
        maha_sq = diff @ S_inv @ diff

        b = (N - 2) / (N - 2 + T * maha_sq + 1e-10)
        b = max(0.0, min(1.0, b))

        mu_js = (1.0 - b) * mu + b * mu_grand
        return mu_js * 252

    @staticmethod
    def black_litterman_posterior(
        pi: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
        tau: float,
        Sigma: np.ndarray,
    ) -> np.ndarray:
        """
        Black-Litterman posterior mean.

        μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} x [(τΣ)^{-1}π + P'Ω^{-1}Q]

        Args:
            pi: equilibrium returns (N,)
            P: pick matrix (K x N)
            Q: view returns (K,)
            Omega: view uncertainty matrix (K x K)
            tau: scalar scaling parameter
            Sigma: covariance matrix (N x N)
        """
        tau_sigma_inv = np.linalg.inv(tau * Sigma + np.eye(Sigma.shape[0]) * 1e-10)
        try:
            omega_inv = np.linalg.inv(Omega + np.eye(Omega.shape[0]) * 1e-10)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.pinv(Omega)

        M1 = tau_sigma_inv + P.T @ omega_inv @ P
        try:
            M1_inv = np.linalg.inv(M1)
        except np.linalg.LinAlgError:
            M1_inv = np.linalg.pinv(M1)

        M2 = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        return M1_inv @ M2


# ---------------------------------------------------------------------------
# Mean-Variance Optimizer
# ---------------------------------------------------------------------------


class MeanVarianceOptimizer:
    """
    Classic Markowitz mean-variance optimization with robust extensions.
    All weights are long-only (w >= 0) and fully invested (sum = 1).
    """

    def __init__(self, allow_short: bool = False):
        self.allow_short = allow_short

    def _validate_inputs(self, N: int) -> None:
        if N < 2:
            raise ValueError("Need at least 2 assets")

    def _build_constraints(
        self, N: int, constraints: Optional[dict]
    ) -> Tuple[list, list]:
        """Build scipy constraints and bounds from constraint dict."""
        bounds = [(0.0, 1.0)] * N if not self.allow_short else [(-1.0, 1.0)] * N
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if constraints:
            if "min_weight" in constraints:
                lb = constraints["min_weight"]
                bounds = [(lb, 1.0)] * N if not self.allow_short else [(lb, 1.0)] * N
            if "max_weight" in constraints:
                ub = constraints["max_weight"]
                bounds = [(b[0], ub) for b in bounds]
            if "group_constraints" in constraints:
                for grp in constraints["group_constraints"]:
                    idx = grp["indices"]
                    lo, hi = grp.get("min", 0.0), grp.get("max", 1.0)
                    cons.append({
                        "type": "ineq",
                        "fun": lambda w, i=idx, h=hi: h - np.sum(w[i]),
                    })
                    cons.append({
                        "type": "ineq",
                        "fun": lambda w, i=idx, l=lo: np.sum(w[i]) - l,
                    })

        return bounds, cons

    def max_sharpe(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        rf: float = 0.0,
        constraints: Optional[dict] = None,
        asset_names: Optional[List[str]] = None,
    ) -> OptResult:
        """
        Maximize Sharpe ratio: (μ - rf)' w / sqrt(w' Σ w)
        Via scipy SLSQP; parametric numpy fallback if scipy unavailable.
        """
        N = len(mu)
        self._validate_inputs(N)

        excess_mu = mu - rf
        w0 = np.ones(N) / N

        if _SCIPY_AVAILABLE:
            bounds, cons = self._build_constraints(N, constraints)

            def neg_sharpe(w: np.ndarray) -> float:
                port_ret = w @ excess_mu
                port_vol = np.sqrt(w @ Sigma @ w + 1e-12)
                return -port_ret / port_vol

            def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
                port_ret = w @ excess_mu
                port_var = w @ Sigma @ w + 1e-12
                port_vol = np.sqrt(port_var)
                d_ret = excess_mu
                d_vol = (Sigma @ w) / port_vol
                sharpe = port_ret / port_vol
                return -(d_ret * port_vol - port_ret * d_vol) / port_var

            result = minimize(
                neg_sharpe,
                w0,
                jac=neg_sharpe_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 1000, "ftol": 1e-10},
            )
            weights = np.maximum(result.x, 0.0)
            weights /= weights.sum() + 1e-12

        else:
            # Parametric approach: solve for tangency portfolio analytically
            # w* ∝ Σ^{-1} (μ - rf)
            try:
                Sigma_inv = np.linalg.inv(Sigma + np.eye(N) * 1e-8)
                weights = Sigma_inv @ excess_mu
                weights = np.maximum(weights, 0.0)
                if weights.sum() < 1e-10:
                    weights = np.ones(N) / N
                else:
                    weights /= weights.sum()
            except np.linalg.LinAlgError:
                weights = np.ones(N) / N

        weights = np.clip(weights, 0.0, 1.0)
        weights /= weights.sum()

        port_ret = weights @ mu
        port_vol = np.sqrt(weights @ Sigma @ weights)
        sharpe = (port_ret - rf) / (port_vol + 1e-10)
        dr = self._diversification_ratio(weights, Sigma)

        return OptResult(
            weights=weights,
            method="max_sharpe",
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            diversification_ratio=dr,
            risk_contributions=self._risk_contributions(weights, Sigma),
            asset_names=asset_names,
        )

    def min_variance(
        self,
        Sigma: np.ndarray,
        constraints: Optional[dict] = None,
        asset_names: Optional[List[str]] = None,
    ) -> OptResult:
        """
        Minimize portfolio variance: w'Σw.
        Closed form for unconstrained; SLSQP for constrained.
        """
        N = Sigma.shape[0]
        self._validate_inputs(N)

        if constraints is None and not self.allow_short:
            # Closed form: w* ∝ Σ^{-1} 1
            try:
                Sigma_inv = np.linalg.inv(Sigma + np.eye(N) * 1e-8)
                ones = np.ones(N)
                weights = Sigma_inv @ ones
                weights = np.maximum(weights, 0.0)
                if weights.sum() < 1e-10:
                    weights = np.ones(N) / N
                else:
                    weights /= weights.sum()
            except np.linalg.LinAlgError:
                weights = np.ones(N) / N
        elif _SCIPY_AVAILABLE:
            bounds, cons = self._build_constraints(N, constraints)
            w0 = np.ones(N) / N

            def port_var(w: np.ndarray) -> float:
                return w @ Sigma @ w

            def port_var_grad(w: np.ndarray) -> np.ndarray:
                return 2.0 * Sigma @ w

            result = minimize(
                port_var,
                w0,
                jac=port_var_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            weights = np.maximum(result.x, 0.0)
            weights /= weights.sum() + 1e-12
        else:
            weights = np.ones(N) / N

        port_vol = np.sqrt(weights @ Sigma @ weights)
        dr = self._diversification_ratio(weights, Sigma)

        return OptResult(
            weights=weights,
            method="min_variance",
            expected_return=0.0,
            expected_volatility=port_vol,
            sharpe_ratio=0.0,
            diversification_ratio=dr,
            risk_contributions=self._risk_contributions(weights, Sigma),
            asset_names=asset_names,
        )

    def min_cvar(
        self,
        returns: pd.DataFrame,
        alpha: float = 0.05,
        constraints: Optional[dict] = None,
        asset_names: Optional[List[str]] = None,
    ) -> OptResult:
        """
        Minimum CVaR (Expected Shortfall) portfolio.
        Rockafellar & Uryasev (2000): linear programming formulation.

        Variables: [w (N), VaR (1), u (T)] where u_t ≥ 0
        min   VaR + 1/(Txα) x Σ u_t
        s.t.  u_t ≥ -r_t' w - VaR  ∀t
              u_t ≥ 0
              1'w = 1, w ≥ 0 (long only)
        """
        X = -returns.values  # losses (positive = bad)
        T, N = X.shape

        if not _SCIPY_AVAILABLE:
            # Fallback: equal weight
            weights = np.ones(N) / N
            port_rets = returns.values @ weights
            port_var = np.percentile(port_rets, int(alpha * 100))
            tail = port_rets[port_rets <= port_var]
            cvar = -tail.mean() if len(tail) > 0 else 0.0
            return OptResult(
                weights=weights,
                method="min_cvar",
                expected_return=0.0,
                expected_volatility=returns.std().mean(),
                sharpe_ratio=0.0,
                asset_names=asset_names,
                metadata={"cvar_95": cvar, "alpha": alpha},
            )

        # LP: min c'x  s.t. A_ub x <= b_ub, A_eq x = b_eq, bounds
        # Variables: [w_1..w_N, gamma, u_1..u_T]
        # Total variables: N + 1 + T

        n_vars = N + 1 + T
        c = np.zeros(n_vars)
        c[N] = 1.0  # VaR coefficient
        c[N + 1 :] = 1.0 / (T * alpha)  # u_t coefficients

        # Constraints: u_t >= -X[t]'w - gamma  ↔  -u_t - X[t]'w - gamma <= 0
        # ↔  [-X[t], -1, 0..,-1_t,..,0] x <= 0
        A_ub = np.zeros((T, n_vars))
        b_ub = np.zeros(T)
        for t in range(T):
            A_ub[t, :N] = -X[t]  # -loss_t' w
            A_ub[t, N] = -1.0  # -gamma
            A_ub[t, N + 1 + t] = -1.0  # -u_t

        # Equality: sum(w) = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        # Bounds
        w_bounds = [(0.0, 1.0)] * N  # long-only
        gamma_bounds = [(-10.0, 10.0)]  # VaR scalar
        u_bounds = [(0.0, None)] * T  # u_t >= 0
        bounds = w_bounds + gamma_bounds + u_bounds

        try:
            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={"disp": False},
            )
            if res.success:
                weights = np.maximum(res.x[:N], 0.0)
                weights /= weights.sum() + 1e-12
            else:
                logger.warning("CVaR LP did not converge: %s", res.message)
                weights = np.ones(N) / N
        except Exception as exc:
            logger.warning("CVaR LP failed: %s", exc)
            weights = np.ones(N) / N

        port_rets = returns.values @ weights
        port_vol = port_rets.std() * np.sqrt(252)
        port_var_val = np.percentile(port_rets, int(alpha * 100))
        tail = port_rets[port_rets <= port_var_val]
        cvar = float(-tail.mean()) if len(tail) > 0 else 0.0

        Sigma = CovarianceEstimator.sample_covariance(returns)
        dr = self._diversification_ratio(weights, Sigma)

        return OptResult(
            weights=weights,
            method="min_cvar",
            expected_return=float(port_rets.mean() * 252),
            expected_volatility=port_vol,
            sharpe_ratio=0.0,
            diversification_ratio=dr,
            risk_contributions=self._risk_contributions(weights, Sigma),
            asset_names=asset_names,
            metadata={"cvar_95": cvar, "alpha": alpha},
        )

    def compute_efficient_frontier(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        n_points: int = 50,
        asset_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute the parametric efficient frontier.
        Vary target return from min-var to max-return in n_points steps.
        Returns DataFrame with columns: [return, volatility, sharpe, weights...]
        """
        N = len(mu)

        # Bounds of the frontier
        min_var_result = self.min_variance(Sigma, asset_names=asset_names)
        ret_min = min_var_result.weights @ mu
        ret_max = mu.max()

        targets = np.linspace(ret_min, ret_max, n_points)
        records = []

        for target in targets:
            if not _SCIPY_AVAILABLE:
                # Parametric approximation -- no optimizer available
                idx = np.argmax(mu >= target) if (mu >= target).any() else N - 1
                w = np.zeros(N)
                w[idx] = 1.0
            else:
                bounds = [(0.0, 1.0)] * N
                cons = [
                    {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                    {"type": "ineq", "fun": lambda w, t=target: w @ mu - t},
                ]
                w0 = np.ones(N) / N
                result = minimize(
                    lambda w: w @ Sigma @ w,
                    w0,
                    jac=lambda w: 2 * Sigma @ w,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-10},
                )
                if result.success:
                    w = np.maximum(result.x, 0.0)
                    w /= w.sum() + 1e-12
                else:
                    continue

            port_ret = w @ mu
            port_vol = np.sqrt(w @ Sigma @ w)
            record = {
                "target_return": target,
                "expected_return": port_ret,
                "expected_volatility": port_vol,
                "sharpe_ratio": port_ret / (port_vol + 1e-10),
            }
            if asset_names:
                for name, wi in zip(asset_names, w):
                    record[f"w_{name}"] = wi
            records.append(record)

        return pd.DataFrame(records)

    def resampled_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_simulations: int = 500,
        n_points: int = 20,
        rf: float = 0.0,
        asset_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Michaud (1989) Resampled Efficient Frontier.
        Bootstrap return distributions -> optimize each -> average weights.

        This reduces the sensitivity of MV optimization to estimation error.
        """
        N = returns.shape[1]
        T = returns.shape[0]

        # Target returns on the original frontier
        mu_orig = self.sample_mean(returns)
        Sigma_orig = CovarianceEstimator.ledoit_wolf_shrinkage(returns)

        min_ret = np.min(mu_orig)
        max_ret = np.max(mu_orig)
        target_returns = np.linspace(min_ret, max_ret, n_points)

        # Accumulate weights across simulations
        accumulated_weights = np.zeros((n_points, N))
        successful_sims = 0

        rng = np.random.default_rng(42)

        for sim in range(n_simulations):
            # Bootstrap resample T observations with replacement
            idx = rng.integers(0, T, size=T)
            sim_returns = returns.iloc[idx]

            mu_sim = MeanEstimator.sample_mean(sim_returns)
            Sigma_sim = CovarianceEstimator.ledoit_wolf_shrinkage(sim_returns)

            # Ensure PSD
            Sigma_sim = _ensure_psd(Sigma_sim)

            try:
                frontier = self.compute_efficient_frontier(
                    mu_sim, Sigma_sim, n_points=n_points, asset_names=asset_names
                )
                if len(frontier) == 0:
                    continue

                # Interpolate to target returns on the original scale
                for i, target in enumerate(target_returns):
                    idx_closest = (
                        (frontier["expected_return"] - target).abs().idxmin()
                    )
                    row = frontier.iloc[idx_closest]
                    if asset_names:
                        w = np.array([row.get(f"w_{name}", 0.0) for name in asset_names])
                    else:
                        cols = [c for c in frontier.columns if c.startswith("w_")]
                        w = row[cols].values if cols else np.ones(N) / N
                    accumulated_weights[i] += w

                successful_sims += 1
            except Exception:
                continue

        if successful_sims == 0:
            return pd.DataFrame()

        # Average weights across simulations
        avg_weights = accumulated_weights / successful_sims

        records = []
        for i, target in enumerate(target_returns):
            w = avg_weights[i]
            if w.sum() > 1e-10:
                w /= w.sum()
            port_ret = w @ mu_orig
            port_vol = np.sqrt(w @ Sigma_orig @ w)
            record = {
                "target_return": target,
                "expected_return": port_ret,
                "expected_volatility": port_vol,
                "sharpe_ratio": (port_ret - rf) / (port_vol + 1e-10),
            }
            if asset_names:
                for name, wi in zip(asset_names, w):
                    record[f"w_{name}"] = wi
            records.append(record)

        return pd.DataFrame(records)

    @staticmethod
    def _risk_contributions(weights: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Marginal risk contributions: w_i x (Σw)_i / sqrt(w'Σw)"""
        port_var = weights @ Sigma @ weights
        if port_var < 1e-12:
            return np.ones(len(weights)) / len(weights)
        marginal = Sigma @ weights
        rc = weights * marginal / np.sqrt(port_var)
        return rc / rc.sum()

    @staticmethod
    def _diversification_ratio(weights: np.ndarray, Sigma: np.ndarray) -> float:
        """DR = w'σ / sqrt(w'Σw)"""
        vols = np.sqrt(np.diag(Sigma))
        port_vol = np.sqrt(weights @ Sigma @ weights)
        if port_vol < 1e-12:
            return 1.0
        return float((weights @ vols) / port_vol)

    # Convenience access
    sample_mean = staticmethod(MeanEstimator.sample_mean)


# ---------------------------------------------------------------------------
# Black-Litterman Optimizer
# ---------------------------------------------------------------------------


class BlackLittermanOptimizer:
    """
    Full Black-Litterman model implementation.
    He & Litterman (1999) / Idzorek (2005) methodology.
    """

    def compute_implied_equilibrium_returns(
        self,
        market_caps: np.ndarray,
        Sigma: np.ndarray,
        delta: float = 2.5,
        rf: float = 0.0,
    ) -> np.ndarray:
        """
        Implied equilibrium returns via reverse optimization.

        π = δ x Σ x w_market
        δ: market risk aversion (typically 2.5 for long-term investors)
        w_market: market-cap weighted portfolio
        """
        w_market = market_caps / (market_caps.sum() + 1e-12)
        pi = delta * Sigma @ w_market + rf
        return pi

    @staticmethod
    def calibrate_tau(T: int, n_assets: int, method: str = "satchell") -> float:
        """
        Calibrate tau (scaling parameter for uncertainty in equilibrium).

        "satchell": τ = 1/T  (Satchell & Scowcroft 2000)
        "he_litterman": τ = 1.0  (He & Litterman 1999)
        "meucci": τ = 1/N  (alternative, asset-count based)
        """
        if method == "he_litterman":
            return 1.0
        elif method == "meucci":
            return 1.0 / n_assets
        else:  # satchell (default)
            return 1.0 / T

    def compute_omega_proportional(
        self, P: np.ndarray, Sigma: np.ndarray, tau: float
    ) -> np.ndarray:
        """
        Proportional uncertainty matrix (Idzorek 2005).
        Ω = diag(τ x P Σ P')

        Each view's uncertainty is proportional to the variance of the
        view portfolio, scaled by tau.
        """
        view_var = tau * P @ Sigma @ P.T
        return np.diag(np.diag(view_var))

    def compute_omega_from_confidence(
        self,
        P: np.ndarray,
        Sigma: np.ndarray,
        tau: float,
        confidences: np.ndarray,
    ) -> np.ndarray:
        """
        Idzorek (2005) confidence-based Omega.
        confidence=1 -> Omega->0 (certainty); confidence=0 -> Omega->∞

        Ω_i = (1 - c_i) / c_i x τ x (P Σ P')_ii
        """
        view_var = tau * np.diag(P @ Sigma @ P.T)
        omega_diag = np.where(
            confidences > 1e-4,
            (1.0 - confidences) / confidences * view_var,
            view_var * 1e6,
        )
        return np.diag(omega_diag)

    def compute_bl_posterior(
        self,
        pi: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
        tau: float,
        Sigma: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full BL posterior: mean and covariance.

        Posterior mean:
          μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} x [(τΣ)^{-1}π + P'Ω^{-1}Q]

        Posterior covariance (uncertainty in mean estimate):
          Σ_BL = Σ + M^{-1}
          where M = (τΣ)^{-1} + P'Ω^{-1}P
        """
        N = Sigma.shape[0]
        reg = np.eye(N) * 1e-8

        tau_sigma_inv = np.linalg.inv(tau * Sigma + reg)
        try:
            omega_inv = np.linalg.inv(Omega + np.eye(Omega.shape[0]) * 1e-8)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.pinv(Omega)

        M = tau_sigma_inv + P.T @ omega_inv @ P
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        mu_bl = M_inv @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        Sigma_bl = Sigma + M_inv  # posterior predictive covariance

        return mu_bl, Sigma_bl

    def optimize_with_views(
        self,
        market_caps: pd.Series,
        returns_history: pd.DataFrame,
        views: List[View],
        tau: float = None,
        delta: float = 2.5,
        rf: float = 0.0,
        asset_names: Optional[List[str]] = None,
    ) -> OptResult:
        """
        Full BL optimization pipeline with investor views.

        Args:
            market_caps: market capitalizations (Series indexed by ticker)
            returns_history: historical returns DataFrame
            views: list of View objects
            tau: uncertainty scaling (auto-calibrated if None)
            delta: risk aversion
            rf: risk-free rate (annualized)
        """
        tickers = list(returns_history.columns)
        N = len(tickers)
        T = len(returns_history)

        if tau is None:
            tau = self.calibrate_tau(T, N, method="satchell")

        # Align market caps to returns universe
        caps = np.array([market_caps.get(t, 1.0) for t in tickers], dtype=float)

        # Estimate covariance
        Sigma = CovarianceEstimator.ledoit_wolf_shrinkage(returns_history)
        Sigma = _ensure_psd(Sigma)

        # Implied equilibrium returns
        pi = self.compute_implied_equilibrium_returns(caps, Sigma, delta=delta, rf=rf)

        # Build pick matrix P and view vector Q
        K = len(views)
        if K == 0:
            # No views: just optimize on equilibrium returns
            mv = MeanVarianceOptimizer()
            return mv.max_sharpe(pi, Sigma, rf=rf, asset_names=tickers)

        P = np.zeros((K, N))
        Q = np.zeros(K)
        confidences = np.zeros(K)

        ticker_idx = {t: i for i, t in enumerate(tickers)}

        for k, view in enumerate(views):
            for asset, wt in zip(view.assets, view.weights):
                if asset in ticker_idx:
                    P[k, ticker_idx[asset]] = wt
            Q[k] = view.expected_return
            confidences[k] = view.confidence

        # Omega: confidence-based
        Omega = self.compute_omega_from_confidence(P, Sigma, tau, confidences)

        # BL posterior
        mu_bl, Sigma_bl = self.compute_bl_posterior(pi, P, Q, Omega, tau, Sigma)

        # Optimize on posterior
        mv = MeanVarianceOptimizer()
        result = mv.max_sharpe(mu_bl, Sigma_bl, rf=rf, asset_names=tickers)
        result.method = "black_litterman"
        result.metadata = {
            "tau": tau,
            "delta": delta,
            "n_views": K,
            "equilibrium_returns": pi.tolist(),
            "bl_posterior_returns": mu_bl.tolist(),
        }
        return result


# ---------------------------------------------------------------------------
# Maximum Diversification Optimizer
# ---------------------------------------------------------------------------


class MaxDiversificationOptimizer:
    """
    Choueifnaour & Coignard (2008) Maximum Diversification Portfolio.
    Maximizes the Diversification Ratio: DR = w'σ / sqrt(w'Σw)
    """

    @staticmethod
    def compute_diversification_ratio(
        weights: np.ndarray, vols: np.ndarray, Sigma: np.ndarray
    ) -> float:
        """DR = weighted-average volatility / portfolio volatility"""
        port_vol = np.sqrt(weights @ Sigma @ weights + 1e-12)
        weighted_vols = weights @ vols
        return float(weighted_vols / port_vol)

    def optimize(
        self,
        returns: pd.DataFrame,
        asset_names: Optional[List[str]] = None,
    ) -> OptResult:
        """
        Maximize DR = w'σ / sqrt(w'Σw).

        Equivalent to maximizing a proxy Sharpe with μ = σ (individual vols
        as expected returns). This yields the maximum diversification weights.
        """
        Sigma = CovarianceEstimator.ledoit_wolf_shrinkage(returns)
        Sigma = _ensure_psd(Sigma)
        vols = np.sqrt(np.diag(Sigma))
        N = len(vols)

        if not _SCIPY_AVAILABLE:
            # Inverse volatility fallback
            weights = (1.0 / (vols + 1e-10))
            weights /= weights.sum()
            dr = self.compute_diversification_ratio(weights, vols, Sigma)
            return OptResult(
                weights=weights,
                method="max_diversification",
                expected_volatility=float(np.sqrt(weights @ Sigma @ weights)),
                diversification_ratio=dr,
                asset_names=asset_names,
            )

        bounds = [(0.0, 1.0)] * N
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def neg_dr(w: np.ndarray) -> float:
            port_vol = np.sqrt(w @ Sigma @ w + 1e-12)
            return -(w @ vols) / port_vol

        def neg_dr_grad(w: np.ndarray) -> np.ndarray:
            port_var = w @ Sigma @ w + 1e-12
            port_vol = np.sqrt(port_var)
            wv = w @ vols
            d_port_vol = (Sigma @ w) / port_vol
            # d/dw [(w'σ) / ||w||_Σ] = σ/||w||_Σ - (w'σ)/(||w||_Σ)³ x Σw
            grad = vols / port_vol - wv / port_var * (Sigma @ w)
            return -grad

        w0 = vols / vols.sum()  # start from inv-vol weights
        result = minimize(
            neg_dr,
            w0,
            jac=neg_dr_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = np.maximum(result.x, 0.0) if result.success else w0
        weights /= weights.sum() + 1e-12
        weights = np.clip(weights, 0.0, 1.0)
        weights /= weights.sum()

        dr = self.compute_diversification_ratio(weights, vols, Sigma)
        port_vol = float(np.sqrt(weights @ Sigma @ weights))
        mu_ann = MeanEstimator.sample_mean(returns)
        port_ret = float(weights @ mu_ann)
        rc = MeanVarianceOptimizer._risk_contributions(weights, Sigma)

        return OptResult(
            weights=weights,
            method="max_diversification",
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=port_ret / (port_vol + 1e-10),
            diversification_ratio=dr,
            risk_contributions=rc,
            asset_names=asset_names,
        )


# ---------------------------------------------------------------------------
# Risk Budgeting / Equal Risk Contribution
# ---------------------------------------------------------------------------


class RiskBudgetingOptimizer:
    """
    Risk Budgeting with arbitrary target risk contributions.
    Special case: Equal Risk Contribution (ERC) / Risk Parity.

    Reference: Maillard, Roncalli & Teiletche (2010).
    """

    def optimize(
        self,
        returns: pd.DataFrame,
        target_risk_budgets: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None,
    ) -> OptResult:
        """
        Find weights such that RC_i = budget_i x total_portfolio_risk.

        If target_risk_budgets is None -> ERC (all budgets equal to 1/N).

        Minimize: Σ_i Σ_j (RC_i/b_i - RC_j/b_j)²
        where RC_i = w_i x (Σw)_i / sqrt(w'Σw) (risk contribution of asset i)
        """
        Sigma = CovarianceEstimator.ledoit_wolf_shrinkage(returns)
        Sigma = _ensure_psd(Sigma)
        N = Sigma.shape[0]

        if target_risk_budgets is None:
            budgets = np.ones(N) / N
        else:
            budgets = np.array(target_risk_budgets, dtype=float)
            budgets /= budgets.sum()

        if not _SCIPY_AVAILABLE:
            # Inverse volatility approximation for ERC
            vols = np.sqrt(np.diag(Sigma))
            weights = (1.0 / (vols + 1e-10))
            weights /= weights.sum()
        else:
            bounds = [(1e-6, 1.0)] * N
            cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            w0 = budgets.copy()

            def risk_budget_objective(w: np.ndarray) -> float:
                port_var = w @ Sigma @ w + 1e-12
                port_vol = np.sqrt(port_var)
                rc = w * (Sigma @ w) / port_vol
                total_rc = rc.sum()
                if total_rc < 1e-12:
                    return 1e10
                rc_norm = rc / total_rc
                diff = rc_norm / budgets - 1.0
                return float(np.sum(diff**2))

            # Try multiple starting points for robustness
            best_result = None
            best_val = np.inf
            rng = np.random.default_rng(0)
            for start_attempt in range(5):
                if start_attempt == 0:
                    w_start = budgets.copy()
                else:
                    w_start = rng.dirichlet(np.ones(N))

                res = minimize(
                    risk_budget_objective,
                    w_start,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=cons,
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res

            weights = np.maximum(best_result.x, 0.0) if best_result else budgets
            weights /= weights.sum() + 1e-12

        port_vol = float(np.sqrt(weights @ Sigma @ weights))
        mu_ann = MeanEstimator.sample_mean(returns)
        port_ret = float(weights @ mu_ann)
        rc = MeanVarianceOptimizer._risk_contributions(weights, Sigma)

        method = (
            "erc" if target_risk_budgets is None else "risk_parity"
        )

        return OptResult(
            weights=weights,
            method=method,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=port_ret / (port_vol + 1e-10),
            diversification_ratio=MeanVarianceOptimizer._diversification_ratio(
                weights, Sigma
            ),
            risk_contributions=rc,
            asset_names=asset_names,
            metadata={"target_budgets": budgets.tolist(), "achieved_rc": rc.tolist()},
        )


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------


class HierarchicalRiskParity:
    """
    Lopez de Prado (2016) Hierarchical Risk Parity.
    Cluster-based allocation that avoids inversion of the covariance matrix.
    """

    @staticmethod
    def compute_correlation_matrix(returns: pd.DataFrame) -> np.ndarray:
        corr = returns.corr().values
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    @staticmethod
    def compute_distance_matrix(corr: np.ndarray) -> np.ndarray:
        """sqrt(0.5 x (1 - corr)) -- ensures triangle inequality."""
        return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))

    @staticmethod
    def hierarchical_clustering(distances: np.ndarray) -> np.ndarray:
        """
        Single-linkage hierarchical clustering.
        Returns scipy linkage matrix if available, else numpy implementation.
        """
        if _SCIPY_CLUSTER:
            condensed = squareform(distances, checks=False)
            return linkage(condensed, method="single")

        # Numpy fallback: simple greedy single-linkage
        N = distances.shape[0]
        link = []
        clusters = {i: [i] for i in range(N)}
        dist = distances.copy()
        np.fill_diagonal(dist, np.inf)

        for step in range(N - 1):
            # Find minimum distance pair
            ij = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = int(ij[0]), int(ij[1])
            if i > j:
                i, j = j, i

            new_id = N + step
            link.append([i, j, dist[i, j], len(clusters[i]) + len(clusters[j])])
            clusters[new_id] = clusters[i] + clusters[j]

            # Update distances (single-linkage: min)
            new_dist = np.minimum(dist[i], dist[j])
            new_dist = np.minimum(new_dist, new_dist)
            dist = np.vstack([dist, new_dist])
            dist = np.hstack([dist, np.append(new_dist, [np.inf]).reshape(-1, 1)])
            dist[i, :] = np.inf
            dist[:, i] = np.inf
            dist[j, :] = np.inf
            dist[:, j] = np.inf

        return np.array(link)

    @staticmethod
    def quasi_diagonalize(link: np.ndarray) -> List[int]:
        """
        Reorder asset indices so correlated assets are adjacent.
        Produces a quasi-diagonal covariance matrix.
        """
        N = int(link.shape[0] + 1)
        # Build cluster membership from linkage
        clusters = {i: [i] for i in range(N)}
        for k, row in enumerate(link):
            i, j = int(row[0]), int(row[1])
            new_id = N + k
            clusters[new_id] = clusters.pop(i, [i]) + clusters.pop(j, [j])

        # The last entry in clusters is the full sorted list
        if clusters:
            sorted_idx = list(clusters.values())[-1]
        else:
            sorted_idx = list(range(N))

        return sorted_idx

    @staticmethod
    def recursive_bisection(cov: np.ndarray, sorted_indices: List[int]) -> np.ndarray:
        """
        Recursive bisection: allocate weights by inverse cluster variance.

        At each step, bisect the sorted index list.
        Weight each sub-cluster proportionally to its inverse variance.
        """
        N = cov.shape[0]
        weights = np.ones(N)

        def _bisect(items: List[int]) -> None:
            if len(items) <= 1:
                return

            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]

            # Compute cluster variances (equal-weight within cluster)
            def cluster_var(idx: List[int]) -> float:
                sub = cov[np.ix_(idx, idx)]
                w = np.ones(len(idx)) / len(idx)
                return float(w @ sub @ w)

            var_left = cluster_var(left)
            var_right = cluster_var(right)

            # Inverse variance weighting
            total_inv = (1.0 / (var_left + 1e-12)) + (1.0 / (var_right + 1e-12))
            alpha = (1.0 / (var_left + 1e-12)) / total_inv

            for idx in left:
                weights[idx] *= alpha
            for idx in right:
                weights[idx] *= 1.0 - alpha

            _bisect(left)
            _bisect(right)

        _bisect(sorted_indices)
        return weights / weights.sum()

    def optimize(
        self,
        returns: pd.DataFrame,
        asset_names: Optional[List[str]] = None,
    ) -> OptResult:
        """Full HRP pipeline."""
        tickers = asset_names or list(returns.columns)
        N = returns.shape[1]

        Sigma = CovarianceEstimator.ledoit_wolf_shrinkage(returns)
        Sigma = _ensure_psd(Sigma)

        corr = self.compute_correlation_matrix(returns)
        dist = self.compute_distance_matrix(corr)
        link = self.hierarchical_clustering(dist)
        sorted_idx = self.quasi_diagonalize(link)

        # Validate sorted_idx
        if len(sorted_idx) != N or set(sorted_idx) != set(range(N)):
            sorted_idx = list(range(N))

        weights = self.recursive_bisection(Sigma, sorted_idx)

        mu_ann = MeanEstimator.sample_mean(returns)
        port_ret = float(weights @ mu_ann)
        port_vol = float(np.sqrt(weights @ Sigma @ weights))
        rc = MeanVarianceOptimizer._risk_contributions(weights, Sigma)
        dr = MeanVarianceOptimizer._diversification_ratio(weights, Sigma)

        return OptResult(
            weights=weights,
            method="hrp",
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=port_ret / (port_vol + 1e-10),
            diversification_ratio=dr,
            risk_contributions=rc,
            asset_names=tickers,
            metadata={"sorted_indices": sorted_idx},
        )

    def get_dendrogram_data(self, returns: pd.DataFrame) -> dict:
        """
        Return linkage data suitable for plotting the dendrogram.
        Includes: linkage matrix, sorted labels, distance matrix.
        """
        tickers = list(returns.columns)
        corr = self.compute_correlation_matrix(returns)
        dist = self.compute_distance_matrix(corr)
        link = self.hierarchical_clustering(dist)
        sorted_idx = self.quasi_diagonalize(link)

        dend_data = {
            "linkage_matrix": link.tolist() if isinstance(link, np.ndarray) else link,
            "labels": [tickers[i] for i in sorted_idx],
            "original_labels": tickers,
            "sorted_indices": sorted_idx,
            "distance_matrix": dist.tolist(),
            "correlation_matrix": corr.tolist(),
        }

        if _SCIPY_CLUSTER:
            try:
                import io as _io

                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(
                    link, labels=tickers, ax=ax, orientation="top"
                )
                dend_data["figure_available"] = True
                plt.close(fig)
            except Exception:
                dend_data["figure_available"] = False

        return dend_data


# ---------------------------------------------------------------------------
# Transaction Cost-Aware Optimizer
# ---------------------------------------------------------------------------


class TransactionCostAwareOptimizer:
    """
    Portfolio optimization with explicit turnover constraints and
    transaction cost penalization.
    """

    def optimize_with_turnover_constraint(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        current_weights: np.ndarray,
        max_turnover: float = 0.20,
        tc_bps: float = 10.0,
        rf: float = 0.0,
        asset_names: Optional[List[str]] = None,
    ) -> OptResult:
        """
        Maximize Sharpe subject to:
          Σ|w_new - w_old| ≤ max_turnover
          Transaction cost: tc_bps / 10000 x Σ|w_new - w_old|

        Linearize absolute value by introducing slack vars:
          u_i = |w_i - w_i_old|  ->  u_i ≥ w_i - w_i_old
                                     u_i ≥ w_i_old - w_i
        """
        N = len(mu)
        tc_rate = tc_bps / 10_000.0
        w_old = np.clip(current_weights, 0.0, 1.0)

        if not _SCIPY_AVAILABLE:
            logger.warning("scipy not available; ignoring turnover constraints")
            mv = MeanVarianceOptimizer()
            return mv.max_sharpe(mu, Sigma, rf=rf, asset_names=asset_names)

        # Variables: [w (N), u (N)] -- total 2N
        # Sharpe maximization via negative Sharpe, with TC subtracted from return
        excess_mu = mu - rf

        def neg_sharpe_with_tc(x: np.ndarray) -> float:
            w = x[:N]
            u = x[N:]
            port_ret = w @ excess_mu - tc_rate * u.sum()
            port_vol = np.sqrt(w @ Sigma @ w + 1e-12)
            return -port_ret / (port_vol + 1e-10)

        # Constraints
        cons = [
            # Sum of weights = 1
            {
                "type": "eq",
                "fun": lambda x: np.sum(x[:N]) - 1.0,
            },
            # Turnover bound: sum(u) <= max_turnover
            {
                "type": "ineq",
                "fun": lambda x: max_turnover - np.sum(x[N:]),
            },
        ]
        # u_i >= w_i - w_old_i
        for i in range(N):
            i_ = i
            cons.append({
                "type": "ineq",
                "fun": lambda x, i=i_: x[N + i] - (x[i] - w_old[i]),
            })
            # u_i >= w_old_i - w_i
            cons.append({
                "type": "ineq",
                "fun": lambda x, i=i_: x[N + i] - (w_old[i] - x[i]),
            })

        bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
        x0 = np.concatenate([w_old, np.abs(w_old - w_old) + 1e-4])
        x0[:N] /= x0[:N].sum() + 1e-10

        result = minimize(
            neg_sharpe_with_tc,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if result.success:
            weights = np.maximum(result.x[:N], 0.0)
            u = np.maximum(result.x[N:], 0.0)
        else:
            weights = w_old.copy()
            u = np.zeros(N)

        weights /= weights.sum() + 1e-12
        turnover = float(np.sum(np.abs(weights - w_old)))
        tc_drag = tc_rate * turnover

        port_ret = float(weights @ mu)
        port_vol = float(np.sqrt(weights @ Sigma @ weights))
        net_ret = port_ret - tc_drag
        dr = MeanVarianceOptimizer._diversification_ratio(weights, Sigma)
        rc = MeanVarianceOptimizer._risk_contributions(weights, Sigma)

        return OptResult(
            weights=weights,
            method="tc_aware_max_sharpe",
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=(net_ret - rf) / (port_vol + 1e-10),
            diversification_ratio=dr,
            risk_contributions=rc,
            asset_names=asset_names,
            metadata={
                "turnover": turnover,
                "tc_drag_bps": tc_drag * 10_000,
                "max_turnover_constraint": max_turnover,
                "tc_bps": tc_bps,
            },
        )

    @staticmethod
    def compute_break_even_alpha(
        current_weights: np.ndarray,
        optimal_weights: np.ndarray,
        tc_bps: float = 10.0,
        annualized_sharpe: float = 1.0,
    ) -> float:
        """
        Break-even alpha: the minimum expected Sharpe improvement required
        to justify the transaction costs of rebalancing.

        Break-even alpha = TC / (portfolio volatility x time horizon)
        = (tc_bps/10000 x turnover)

        Returns the annualized return needed (in bps) to recover TC.
        """
        tc_rate = tc_bps / 10_000.0
        turnover = float(np.sum(np.abs(optimal_weights - current_weights)))
        tc_cost = tc_rate * turnover  # as fraction of portfolio
        return tc_cost * 10_000  # return in bps


# ---------------------------------------------------------------------------
# Portfolio Optimizer Engine (Orchestrator)
# ---------------------------------------------------------------------------


class PortfolioOptimizerEngine:
    """
    Central orchestrator for all portfolio optimization methods.
    """

    SUPPORTED_METHODS = [
        "max_sharpe",
        "min_variance",
        "min_cvar",
        "max_diversification",
        "hrp",
        "bl",
        "erc",
        "risk_parity",
    ]

    def __init__(self, rf: float = 0.0, cov_method: str = "ledoit_wolf"):
        self.rf = rf
        self.cov_method = cov_method
        self._mv = MeanVarianceOptimizer()
        self._hrp = HierarchicalRiskParity()
        self._bl = BlackLittermanOptimizer()
        self._md = MaxDiversificationOptimizer()
        self._rb = RiskBudgetingOptimizer()
        self._tc = TransactionCostAwareOptimizer()
        self._cov_est = CovarianceEstimator()

    def _estimate_cov(self, returns: pd.DataFrame) -> np.ndarray:
        methods = {
            "ledoit_wolf": self._cov_est.ledoit_wolf_shrinkage,
            "sample": self._cov_est.sample_covariance,
            "ewma": self._cov_est.exponential_weighted_covariance,
            "denoised": self._cov_est.denoised_covariance,
            "constant_corr": self._cov_est.constant_correlation_shrinkage,
        }
        fn = methods.get(self.cov_method, self._cov_est.ledoit_wolf_shrinkage)
        Sigma = fn(returns)
        return _ensure_psd(Sigma)

    def _estimate_mu(self, returns: pd.DataFrame) -> np.ndarray:
        return MeanEstimator.james_stein_shrinkage(returns)

    def optimize(
        self,
        method: str,
        returns: pd.DataFrame,
        **kwargs,
    ) -> OptResult:
        """
        Dispatch to appropriate optimizer.

        Args:
            method: one of SUPPORTED_METHODS
            returns: daily returns DataFrame
            **kwargs: method-specific arguments
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.SUPPORTED_METHODS}")

        asset_names = list(returns.columns)
        N = len(asset_names)

        if method == "max_sharpe":
            mu = self._estimate_mu(returns)
            Sigma = self._estimate_cov(returns)
            return self._mv.max_sharpe(
                mu, Sigma, rf=self.rf, asset_names=asset_names, **kwargs
            )

        elif method == "min_variance":
            Sigma = self._estimate_cov(returns)
            return self._mv.min_variance(Sigma, asset_names=asset_names, **kwargs)

        elif method == "min_cvar":
            return self._mv.min_cvar(returns, asset_names=asset_names, **kwargs)

        elif method == "max_diversification":
            return self._md.optimize(returns, asset_names=asset_names)

        elif method == "hrp":
            return self._hrp.optimize(returns, asset_names=asset_names)

        elif method == "bl":
            # Requires market_caps and views in kwargs
            market_caps = kwargs.get("market_caps", pd.Series(np.ones(N), index=asset_names))
            views = kwargs.get("views", [])
            return self._bl.optimize_with_views(
                market_caps=market_caps,
                returns_history=returns,
                views=views,
                rf=self.rf,
                asset_names=asset_names,
            )

        elif method in ("erc", "risk_parity"):
            budgets = kwargs.get("target_risk_budgets", None)
            return self._rb.optimize(returns, target_risk_budgets=budgets, asset_names=asset_names)

        else:
            raise ValueError(f"Method '{method}' routing not implemented")

    def run_comparison(
        self,
        returns: pd.DataFrame,
        methods: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Run all requested methods and return a comparison DataFrame.
        """
        if methods is None:
            methods = [m for m in self.SUPPORTED_METHODS if m != "bl"]

        asset_names = list(returns.columns)
        records = []

        for method in methods:
            try:
                result = self.optimize(method, returns, **kwargs)
                row = {
                    "method": method,
                    "expected_return": round(result.expected_return * 100, 2),
                    "expected_volatility": round(result.expected_volatility * 100, 2),
                    "sharpe_ratio": round(result.sharpe_ratio, 3),
                    "diversification_ratio": round(result.diversification_ratio, 3),
                }
                for name, w in zip(asset_names, result.weights):
                    row[f"w_{name}"] = round(w, 4)
                records.append(row)
                logger.info("Optimized [%s]: Sharpe=%.3f  Vol=%.2f%%", method,
                            result.sharpe_ratio, result.expected_volatility * 100)
            except Exception as exc:
                logger.warning("Method '%s' failed: %s", method, exc)
                records.append({"method": method, "error": str(exc)})

        return pd.DataFrame(records)

    def compute_metrics(
        self, weights: np.ndarray, returns: pd.DataFrame, rf: float = None
    ) -> PortfolioMetrics:
        """Compute full risk/return metrics for a given weight vector."""
        rf = self.rf if rf is None else rf
        port_rets = returns.values @ weights

        # Annualized return & vol
        ann_ret = float(port_rets.mean() * 252)
        ann_vol = float(port_rets.std() * np.sqrt(252))

        # Sharpe
        sharpe = (ann_ret - rf) / (ann_vol + 1e-10)

        # Sortino (downside deviation)
        downside_rets = port_rets[port_rets < 0]
        downside_vol = float(downside_rets.std() * np.sqrt(252)) if len(downside_rets) > 0 else ann_vol
        sortino = (ann_ret - rf) / (downside_vol + 1e-10)

        # Max drawdown
        cum_ret = (1 + port_rets).cumprod()
        rolling_max = np.maximum.accumulate(cum_ret)
        drawdowns = (cum_ret - rolling_max) / (rolling_max + 1e-12)
        max_dd = float(drawdowns.min())

        calmar = ann_ret / (abs(max_dd) + 1e-10)

        # VaR / CVaR
        var_95 = float(np.percentile(port_rets, 5))
        tail = port_rets[port_rets <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

        # Risk contributions
        Sigma = self._estimate_cov(returns)
        rc = MeanVarianceOptimizer._risk_contributions(weights, Sigma)

        # Diversification ratio
        dr = MeanVarianceOptimizer._diversification_ratio(weights, Sigma)

        # Effective N (inverse HHI)
        hhi = float(np.sum(weights**2))
        effective_n = 1.0 / (hhi + 1e-10)

        return PortfolioMetrics(
            expected_return=ann_ret,
            expected_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            var_95=var_95,
            cvar_95=cvar_95,
            diversification_ratio=dr,
            effective_n=effective_n,
            risk_contributions=rc,
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_psd(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Ensure matrix is positive semi-definite by clipping negative eigenvalues."""
    Sigma = (Sigma + Sigma.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def fetch_returns(
    tickers: List[str],
    years: int = 5,
    fallback_n_obs: int = 1260,
) -> pd.DataFrame:
    """
    Fetch daily log-returns for a list of tickers using yfinance.
    Falls back to synthetic correlated returns if yfinance unavailable.
    """
    if _YF_AVAILABLE:
        end = datetime.today()
        start = end - timedelta(days=years * 365)
        try:
            raw = yf.download(
                tickers,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]] if len(tickers) == 1 else raw

            prices = prices.dropna(how="all")
            returns = np.log(prices / prices.shift(1)).dropna()
            return returns
        except Exception as exc:
            logger.warning("yfinance download failed: %s. Using synthetic data.", exc)

    # Synthetic correlated returns
    logger.warning("yfinance unavailable. Generating synthetic returns.")
    N = len(tickers)
    T = fallback_n_obs
    rng = np.random.default_rng(42)

    # Random covariance structure
    A = rng.standard_normal((N, N)) * 0.3
    cov_true = A @ A.T / N + np.eye(N) * 0.01
    mu_true = rng.uniform(0.05, 0.15, N) / 252

    raw_returns = rng.multivariate_normal(mu_true, cov_true / 252, size=T)
    idx = pd.date_range(end=datetime.today(), periods=T, freq="B")
    return pd.DataFrame(raw_returns, index=idx, columns=tickers)


def fetch_risk_free_rate() -> float:
    """Fetch current 3-month T-bill rate from FRED as risk-free rate."""
    if not _REQUESTS_AVAILABLE:
        return 0.04  # default 4%

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=TB3MS&vintage_date=&output_type=file"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        last_line = [l for l in lines if l and not l.startswith("DATE")][-1]
        rate_str = last_line.split(",")[-1].strip()
        if rate_str and rate_str != ".":
            return float(rate_str) / 100.0
    except Exception:
        pass
    return 0.04


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    )

    TICKERS = ["SPY", "QQQ", "GLD", "TLT", "AGG", "VNQ", "EEM"]
    print(f"\n{'='*70}")
    print("SENTINEL Portfolio Optimizer V3 -- dim_081")
    print(f"{'='*70}")
    print(f"Fetching 5-year returns for: {TICKERS}")

    returns = fetch_returns(TICKERS, years=5)
    rf = fetch_risk_free_rate()
    print(f"Risk-free rate (TB3MS): {rf:.2%}")
    print(f"Returns shape: {returns.shape}  ({returns.index[0].date()} -> {returns.index[-1].date()})")

    engine = PortfolioOptimizerEngine(rf=rf, cov_method="ledoit_wolf")

    # ---- Run all methods ------------------------------------------------
    methods_to_run = [
        "max_sharpe",
        "min_variance",
        "min_cvar",
        "max_diversification",
        "hrp",
        "erc",
    ]

    print(f"\n{'─'*70}")
    print("Running 6 optimization methods (excluding BL -- needs market caps):")
    comparison = engine.run_comparison(returns, methods=methods_to_run)
    weight_cols = [c for c in comparison.columns if c.startswith("w_")]
    print(comparison[["method", "expected_return", "expected_volatility",
                        "sharpe_ratio", "diversification_ratio"] + weight_cols].to_string(index=False))

    # ---- Black-Litterman with sample views --------------------------------
    print(f"\n{'─'*70}")
    print("Black-Litterman with 3 investor views:")

    # Approximate market caps (relative)
    market_caps = pd.Series({
        "SPY": 500e9, "QQQ": 200e9, "GLD": 60e9,
        "TLT": 40e9, "AGG": 90e9, "VNQ": 50e9, "EEM": 70e9,
    })

    views = [
        View(
            assets=["QQQ"],
            weights=[1.0],
            expected_return=0.15,
            confidence=0.7,
            description="QQQ outperforms: strong tech cycle",
        ),
        View(
            assets=["SPY", "QQQ"],
            weights=[1.0, -1.0],
            expected_return=0.03,
            confidence=0.5,
            description="Value (SPY) outperforms Growth (QQQ) by 3%",
        ),
        View(
            assets=["EEM"],
            weights=[1.0],
            expected_return=0.04,
            confidence=0.4,
            description="EM underperforms: dollar strength headwind",
        ),
    ]

    bl_result = engine.optimize("bl", returns, market_caps=market_caps, views=views)
    print(f"BL Posterior Expected Return: {bl_result.expected_return:.2%}")
    print(f"BL Optimal Weights:")
    for ticker, w in zip(TICKERS, bl_result.weights):
        print(f"  {ticker:6s}: {w:.4f}  ({w*100:.1f}%)")
    print(f"Sharpe: {bl_result.sharpe_ratio:.3f}")
    print(f"Diversification Ratio: {bl_result.diversification_ratio:.3f}")
    eq_rets = bl_result.metadata.get("equilibrium_returns", [])
    bl_rets = bl_result.metadata.get("bl_posterior_returns", [])
    if eq_rets and bl_rets:
        print("\n  Equilibrium -> BL Posterior returns (annualized):")
        for ticker, eq, bl in zip(TICKERS, eq_rets, bl_rets):
            print(f"  {ticker:6s}:  π={eq:.3%}  ->  μ_BL={bl:.3%}")

    # ---- Efficient Frontier -----------------------------------------------
    print(f"\n{'─'*70}")
    print("Computing Efficient Frontier (20 points):")
    mu = MeanEstimator.james_stein_shrinkage(returns)
    Sigma = CovarianceEstimator.ledoit_wolf_shrinkage(returns)
    Sigma = _ensure_psd(Sigma)
    frontier = engine._mv.compute_efficient_frontier(mu, Sigma, n_points=20, asset_names=TICKERS)
    print(frontier[["expected_return", "expected_volatility", "sharpe_ratio"]].to_string(index=False))

    # ---- Resampled Efficient Frontier ------------------------------------
    print(f"\n{'─'*70}")
    print("Resampled Efficient Frontier (100 simulations, 10 points):")
    resampled = engine._mv.resampled_efficient_frontier(
        returns, n_simulations=100, n_points=10, rf=rf, asset_names=TICKERS
    )
    print(resampled[["expected_return", "expected_volatility", "sharpe_ratio"]].to_string(index=False))

    # ---- HRP Dendrogram Data ---------------------------------------------
    print(f"\n{'─'*70}")
    print("HRP Dendrogram cluster order:")
    dend = engine._hrp.get_dendrogram_data(returns)
    print(f"  Sorted labels: {' -> '.join(dend['labels'])}")

    # ---- Turnover-constrained optimization --------------------------------
    print(f"\n{'─'*70}")
    print("Transaction Cost-Aware Optimization (max 20% turnover, 10bps TC):")
    current_weights = np.array([1/7] * 7)
    mu = MeanEstimator.james_stein_shrinkage(returns)
    Sigma_lw = CovarianceEstimator.ledoit_wolf_shrinkage(returns)
    Sigma_lw = _ensure_psd(Sigma_lw)
    tc_result = engine._tc.optimize_with_turnover_constraint(
        mu=mu,
        Sigma=Sigma_lw,
        current_weights=current_weights,
        max_turnover=0.20,
        tc_bps=10.0,
        rf=rf,
        asset_names=TICKERS,
    )
    print(f"TC-Adjusted Sharpe: {tc_result.sharpe_ratio:.3f}")
    print(f"Turnover: {tc_result.metadata['turnover']:.2%}")
    print(f"TC Drag: {tc_result.metadata['tc_drag_bps']:.1f} bps")
    bea = engine._tc.compute_break_even_alpha(current_weights, tc_result.weights, tc_bps=10.0)
    print(f"Break-Even Alpha: {bea:.1f} bps")

    # ---- Full metrics for best method ------------------------------------
    print(f"\n{'─'*70}")
    max_sharpe_result = engine.optimize("max_sharpe", returns)
    metrics = engine.compute_metrics(max_sharpe_result.weights, returns)
    print(f"Full Metrics -- Max Sharpe Portfolio:")
    print(f"  Ann. Return:      {metrics.expected_return:.2%}")
    print(f"  Ann. Volatility:  {metrics.expected_volatility:.2%}")
    print(f"  Sharpe Ratio:     {metrics.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio:    {metrics.sortino_ratio:.3f}")
    print(f"  Max Drawdown:     {metrics.max_drawdown:.2%}")
    print(f"  Calmar Ratio:     {metrics.calmar_ratio:.3f}")
    print(f"  VaR 95%:          {metrics.var_95:.4f} (daily)")
    print(f"  CVaR 95%:         {metrics.cvar_95:.4f} (daily)")
    print(f"  Diversification:  {metrics.diversification_ratio:.3f}")
    print(f"  Effective N:      {metrics.effective_n:.2f}")
    print(f"  Risk Contributions:")
    for ticker, rc in zip(TICKERS, metrics.risk_contributions):
        print(f"    {ticker:6s}: {rc:.2%}")

    print(f"\n{'='*70}")
    print("Portfolio Optimizer V3 -- complete.")
