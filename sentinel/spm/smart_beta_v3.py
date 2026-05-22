"""
sentinel/spm/smart_beta_v3.py
==============================
Smart Beta / Alternative Weighting Strategies
dim_147 — score target: 9

Implements:
  - Equal Weight (EW)
  - Minimum Variance (MinVar)
  - Equal Risk Contribution / Risk Parity (ERC)
  - Momentum Tilt (risk-adjusted, softmax)
  - Maximum Diversification (MaxDiv)
  - Factor-Tilted Portfolio

All formulas are pure numpy/scipy. No network calls.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def portfolio_vol(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Annualised portfolio volatility = sqrt(w' Sigma w)."""
    w = np.asarray(weights, dtype=float)
    vol_sq = w @ cov_matrix @ w
    return float(np.sqrt(max(vol_sq, 0.0)))


def diversification_ratio(
    weights: np.ndarray,
    vols: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Diversification Ratio = sum(w_i * sigma_i) / sqrt(w' Sigma w)
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(vols, dtype=float)
    port_vol = portfolio_vol(w, cov_matrix)
    if port_vol <= 0.0:
        return 1.0
    weighted_vol = float(w @ v)
    return weighted_vol / port_vol


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature scaling."""
    z = np.asarray(x, dtype=float) / temperature
    z -= z.max()  # numerical stability
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class PortfolioWeights:
    """Container for portfolio weights and metadata."""
    weights: np.ndarray
    strategy: str
    expected_vol: float
    diversification_ratio: float
    effective_n: float   # inverse of Herfindahl-Hirschman Index = 1/sum(w^2)

    def concentration(self) -> float:
        """Herfindahl-Hirschman Index (HHI): sum of squared weights."""
        return float(np.sum(self.weights ** 2))


# ---------------------------------------------------------------------------
# Equal Risk Contribution (Risk Parity)
# ---------------------------------------------------------------------------

class EqualRiskContribution:
    """
    Solves for weights such that every asset contributes equally to total risk.

    Risk contribution: RC_i = w_i * (Sigma w)_i / sqrt(w' Sigma w)
    Objective: minimize sum_i sum_j (RC_i - RC_j)^2
    """

    def fit(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Return ERC weights for the given covariance matrix."""
        n = cov_matrix.shape[0]
        w0 = np.ones(n) / n  # start from equal weight

        def objective(w: np.ndarray) -> float:
            w = np.maximum(w, 1e-10)
            sigma_w = cov_matrix @ w
            port_vol = float(np.sqrt(w @ sigma_w))
            if port_vol <= 0.0:
                return 1e10
            rc = w * sigma_w / port_vol
            # Pairwise squared differences
            diff = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    diff += (rc[i] - rc[j]) ** 2
            return diff

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(1e-6, 1.0)] * n

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        w = result.x
        w = np.maximum(w, 0.0)
        w /= w.sum()
        return w

    def risk_contributions(
        self, weights: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Marginal risk contributions: RC_i = w_i * (Sigma w)_i / vol."""
        w = np.asarray(weights, dtype=float)
        sigma_w = cov_matrix @ w
        vol = portfolio_vol(w, cov_matrix)
        if vol <= 0.0:
            return np.zeros(len(w))
        return w * sigma_w / vol

    def is_erc(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        tol: float = 1e-4,
    ) -> bool:
        """Return True if weights are ERC (all contributions equal within tol)."""
        rc = self.risk_contributions(weights, cov_matrix)
        if rc.sum() <= 0.0:
            return False
        rc_norm = rc / rc.sum()
        n = len(rc_norm)
        target = 1.0 / n
        return bool(np.all(np.abs(rc_norm - target) < tol))


# ---------------------------------------------------------------------------
# Minimum Variance
# ---------------------------------------------------------------------------

class MinimumVariance:
    """Global minimum variance portfolio (long-only constrained)."""

    def fit(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Return minimum variance weights via quadratic optimisation."""
        n = cov_matrix.shape[0]
        w0 = np.ones(n) / n

        def objective(w: np.ndarray) -> float:
            return float(w @ cov_matrix @ w)

        def objective_grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * cov_matrix @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n

        result = minimize(
            objective,
            w0,
            jac=objective_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        w = result.x
        w = np.maximum(w, 0.0)
        w /= w.sum()
        return w

    def analytical_solution(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Analytical MinVar: w = Sigma^-1 * 1 / (1' * Sigma^-1 * 1)
        (unconstrained; may produce negative weights).
        """
        n = cov_matrix.shape[0]
        ones = np.ones(n)
        try:
            cov_inv = np.linalg.inv(cov_matrix)
            raw = cov_inv @ ones
            denom = ones @ raw
            w = raw / denom
            return w
        except np.linalg.LinAlgError:
            return np.ones(n) / n


# ---------------------------------------------------------------------------
# Momentum Tilt
# ---------------------------------------------------------------------------

class MomentumTilt:
    """
    Risk-adjusted momentum weighting via softmax.

    momentum_score_i = mean_return_i / std_return_i  (Sharpe-like)
    weights = softmax(scores * temperature)
    """

    def __init__(self, lookback: int = 252, temperature: float = 1.0) -> None:
        self.lookback = lookback
        self.temperature = temperature

    def fit(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute momentum weights from a (T, N) returns array.
        Uses the last `lookback` rows.
        """
        r = np.asarray(returns, dtype=float)
        if r.ndim == 1:
            r = r.reshape(-1, 1)
        T, n = r.shape
        window = min(self.lookback, T)
        r_window = r[-window:]
        mu = r_window.mean(axis=0)
        sigma = r_window.std(axis=0)
        # Risk-adjusted score; handle zero vol
        scores = np.where(sigma > 0, mu / sigma, 0.0)
        weights = _softmax(scores, temperature=self.temperature)
        return weights


# ---------------------------------------------------------------------------
# Maximum Diversification
# ---------------------------------------------------------------------------

class MaximumDiversification:
    """
    Maximises the Diversification Ratio:
      DR = sum(w_i * sigma_i) / sqrt(w' Sigma w)
    subject to sum(w) = 1, w >= 0.
    """

    def fit(self, vols: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Return maximum diversification weights."""
        n = len(vols)
        v = np.asarray(vols, dtype=float)
        w0 = np.ones(n) / n

        def neg_dr(w: np.ndarray) -> float:
            return -diversification_ratio(w, v, cov_matrix)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n

        result = minimize(
            neg_dr,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        w = result.x
        w = np.maximum(w, 0.0)
        w /= w.sum()
        return w

    def diversification_ratio(
        self,
        weights: np.ndarray,
        vols: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Compute DR for given weights."""
        return diversification_ratio(weights, vols, cov_matrix)


# ---------------------------------------------------------------------------
# Smart Beta Portfolio (facade)
# ---------------------------------------------------------------------------

class SmartBetaPortfolio:
    """
    Unified interface for all smart beta strategies.

    Parameters
    ----------
    returns     : (T, N) daily returns array
    asset_names : optional list of ticker names
    """

    def __init__(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> None:
        self.returns = np.asarray(returns, dtype=float)
        if self.returns.ndim == 1:
            self.returns = self.returns.reshape(-1, 1)
        T, n = self.returns.shape
        self.n = n
        self.asset_names = asset_names or [f"Asset{i}" for i in range(n)]
        # Sample covariance (annualise if daily: * 252)
        self.cov_matrix = np.cov(self.returns.T) * 252 if T > 1 else np.eye(n) * 0.01
        if self.cov_matrix.ndim == 0:
            # Single asset
            self.cov_matrix = self.cov_matrix.reshape(1, 1)
        self.vols = np.sqrt(np.diag(self.cov_matrix))

    def _make_pw(self, weights: np.ndarray, strategy: str) -> PortfolioWeights:
        w = np.asarray(weights, dtype=float)
        ev = portfolio_vol(w, self.cov_matrix)
        dr = diversification_ratio(w, self.vols, self.cov_matrix)
        eff_n = 1.0 / float(np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
        return PortfolioWeights(
            weights=w,
            strategy=strategy,
            expected_vol=ev,
            diversification_ratio=dr,
            effective_n=eff_n,
        )

    def equal_weight(self) -> PortfolioWeights:
        """Equal weight: w_i = 1/n for all i."""
        w = np.ones(self.n) / self.n
        return self._make_pw(w, "equal_weight")

    def min_variance(self) -> PortfolioWeights:
        """Minimum variance portfolio."""
        w = MinimumVariance().fit(self.cov_matrix)
        return self._make_pw(w, "min_variance")

    def risk_parity(self) -> PortfolioWeights:
        """Equal Risk Contribution (Risk Parity) portfolio."""
        w = EqualRiskContribution().fit(self.cov_matrix)
        return self._make_pw(w, "risk_parity")

    def momentum_tilt(self, lookback: int = 63) -> PortfolioWeights:
        """Risk-adjusted momentum tilt portfolio."""
        w = MomentumTilt(lookback=lookback).fit(self.returns)
        return self._make_pw(w, "momentum_tilt")

    def max_diversification(self) -> PortfolioWeights:
        """Maximum diversification portfolio."""
        w = MaximumDiversification().fit(self.vols, self.cov_matrix)
        return self._make_pw(w, "max_diversification")

    def factor_tilt(
        self,
        factor_scores: np.ndarray,
        alpha: float = 0.5,
    ) -> PortfolioWeights:
        """
        Factor-tilted portfolio.
        w_tilt = w_base + alpha * (f_norm - w_base)
        where f_norm = factor_scores / sum(factor_scores).
        alpha=0 → equal weight; alpha=1 → full factor exposure.
        """
        w_base = np.ones(self.n) / self.n
        fs = np.asarray(factor_scores, dtype=float)
        # Normalise factor scores to sum to 1
        fs_sum = fs.sum()
        if fs_sum <= 0.0:
            f_norm = w_base.copy()
        else:
            f_norm = fs / fs_sum
        w = w_base + alpha * (f_norm - w_base)
        w = np.maximum(w, 0.0)
        w /= w.sum()
        return self._make_pw(w, "factor_tilt")

    def compare_strategies(self) -> Dict[str, PortfolioWeights]:
        """Run all strategies and return comparison dict."""
        return {
            "equal_weight": self.equal_weight(),
            "min_variance": self.min_variance(),
            "risk_parity": self.risk_parity(),
            "momentum_tilt": self.momentum_tilt(),
            "max_diversification": self.max_diversification(),
        }


# ---------------------------------------------------------------------------
# Standalone convenience functions
# ---------------------------------------------------------------------------

def equal_risk_contribution(cov_matrix: np.ndarray) -> np.ndarray:
    """Return ERC weights for given covariance matrix."""
    return EqualRiskContribution().fit(cov_matrix)


def minimum_variance(cov_matrix: np.ndarray) -> np.ndarray:
    """Return minimum variance weights for given covariance matrix."""
    return MinimumVariance().fit(cov_matrix)


# ---------------------------------------------------------------------------
# Legacy aliases (for backward compat with existing test stubs)
# ---------------------------------------------------------------------------

class RiskParityEngine(EqualRiskContribution):
    """Alias for EqualRiskContribution."""
    pass


class MomentumTiltPortfolio(MomentumTilt):
    """Alias for MomentumTilt."""
    pass


def compute_factor_tilt(
    factor_scores: np.ndarray,
    alpha: float = 0.5,
    n: Optional[int] = None,
) -> np.ndarray:
    """
    Standalone factor tilt: produce tilted weights from factor scores.
    """
    fs = np.asarray(factor_scores, dtype=float)
    if n is None:
        n = len(fs)
    w_base = np.ones(n) / n
    fs_sum = fs.sum()
    f_norm = fs / fs_sum if fs_sum > 0 else w_base.copy()
    w = w_base + alpha * (f_norm - w_base)
    w = np.maximum(w, 0.0)
    w /= w.sum()
    return w
