"""
Portfolio optimization: Mean-Variance (Markowitz), Black-Litterman,
Equal Risk Contribution, Hierarchical Risk Parity, and more.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

try:
    from sklearn.covariance import LedoitWolf as SklearnLW
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field, validator
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
_EPSILON = 1e-10  # numerical stability floor


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result from a portfolio optimization."""
    method: str
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    success: bool
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EfficientFrontierPoint:
    """A single point on the efficient frontier."""
    target_return: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]


@dataclass
class BLResult:
    """Black-Litterman posterior and optimal weights."""
    assets: List[str]
    prior_returns: Dict[str, float]       # equilibrium returns π
    posterior_returns: Dict[str, float]   # BL posterior μ_BL
    posterior_covariance: np.ndarray
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    views_applied: int
    tau: float


@dataclass
class BacktestResult:
    """Portfolio backtest result."""
    method: str
    returns: pd.Series
    cumulative_returns: pd.Series
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    n_rebalances: int
    weights_history: List[Dict[str, float]]


# ---------------------------------------------------------------------------
# 1. ReturnCovariance — Estimation
# ---------------------------------------------------------------------------

class ReturnCovariance:
    """
    Return and covariance matrix estimation with multiple methods.

    Supports: sample, Ledoit-Wolf shrinkage, factor model, EWMA,
    Pearson/Spearman/distance correlation matrices.
    """

    def get_returns(
        self,
        tickers: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily returns for a list of tickers via yfinance."""
        if not _YF_AVAILABLE:
            return pd.DataFrame()
        end_str = end or datetime.today().strftime("%Y-%m-%d")
        try:
            data = yf.download(
                tickers,
                start=start,
                end=end_str,
                progress=False,
                auto_adjust=True,
                threads=True,
            )
            if data.empty:
                return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"]
            elif "Close" in data.columns:
                closes = data["Close"]
            else:
                closes = data
            if isinstance(closes, pd.Series):
                closes = closes.to_frame(name=tickers[0])
            return closes.pct_change().dropna(how="all")
        except Exception as exc:
            logger.warning("yfinance fetch failed: %s", exc)
            return pd.DataFrame()

    def sample_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Standard sample covariance matrix."""
        return returns.dropna().cov()

    def ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf analytical shrinkage estimator.

        Falls back to manual Oracle Approximating Shrinkage if sklearn unavailable.
        """
        r = returns.dropna()
        tickers = r.columns.tolist()
        X = r.values

        if _SKLEARN_AVAILABLE:
            lw = SklearnLW()
            lw.fit(X)
            cov_arr = lw.covariance_
            return pd.DataFrame(cov_arr, index=tickers, columns=tickers)
        else:
            # Manual Ledoit-Wolf (Oracle Approximating Shrinkage)
            return self._manual_ledoit_wolf(X, tickers)

    def _manual_ledoit_wolf(self, X: np.ndarray, tickers: List[str]) -> pd.DataFrame:
        """
        Manual Ledoit-Wolf shrinkage toward scaled identity matrix.
        Shrinks sample covariance toward: mu * I where mu = trace(S)/p
        Shrinkage intensity alpha minimizes MSE analytically (Ledoit-Wolf 2004).
        """
        n, p = X.shape
        mu = X.mean(axis=0)
        Xc = X - mu
        S = (Xc.T @ Xc) / n  # sample cov (MLE)

        # Target: mu_hat * I
        mu_hat = float(np.trace(S) / p)
        F = mu_hat * np.eye(p)  # shrinkage target

        # Estimate optimal alpha (Ledoit-Wolf constant correlation estimate)
        # delta_sq: sum of squared entries in (S - F)
        delta = S - F
        delta_sq = float(np.sum(delta ** 2))

        # pi_hat (sum of asymptotic variances of sample cov entries)
        pi_hat = 0.0
        for t in range(n):
            xt = Xc[t, :].reshape(-1, 1)
            St = xt @ xt.T
            diff = St - S
            pi_hat += float(np.sum(diff ** 2))
        pi_hat /= n

        # rho_hat
        gamma = float(np.sum(delta ** 2))  # Frobenius norm squared of (S - F)
        alpha = float(np.clip((pi_hat / n) / gamma, 0.0, 1.0)) if gamma > 1e-12 else 0.0

        S_lw = (1.0 - alpha) * S + alpha * F
        return pd.DataFrame(S_lw, index=tickers, columns=tickers)

    def ewma_covariance(
        self,
        returns: pd.DataFrame,
        lam: float = 0.94,
    ) -> pd.DataFrame:
        """
        Exponentially Weighted Moving Average covariance (RiskMetrics, λ=0.94).

        Each observation weighted by (1-λ)λ^(T-t) and normalized.
        """
        r = returns.dropna()
        tickers = r.columns.tolist()
        X = r.values
        n, p = X.shape

        # Decay weights: most recent = index n-1 gets highest weight
        weights = np.array([(1.0 - lam) * lam ** i for i in range(n - 1, -1, -1)])
        weights /= weights.sum()

        mu_ew = (X.T @ weights)
        Xc = X - mu_ew
        cov_arr = (Xc * weights[:, None]).T @ Xc
        return pd.DataFrame(cov_arr, index=tickers, columns=tickers)

    def factor_model_covariance(
        self,
        returns: pd.DataFrame,
        n_factors: int = 5,
    ) -> pd.DataFrame:
        """
        Factor model covariance via PCA factor extraction.

        Σ = B × F_cov × B' + D  (where D = diagonal specific variances)
        """
        r = returns.dropna()
        tickers = r.columns.tolist()
        X = r.values
        n, p = X.shape

        # Standardize
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1)
        sigma = np.where(sigma < _EPSILON, 1.0, sigma)
        Xs = (X - mu) / sigma

        # PCA via SVD
        U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
        n_factors = min(n_factors, min(n, p) - 1)
        B = Vt[:n_factors, :].T  # p x n_factors, loadings

        # Factor returns
        factor_returns = (Xs @ B)  # n x n_factors
        F_cov = np.cov(factor_returns.T)  # n_factors x n_factors

        # Specific variances (residuals)
        resid = Xs - factor_returns @ B.T
        D = np.diag(np.var(resid, axis=0, ddof=1))

        # Reconstruct cov in original scale
        sigma_diag = np.diag(sigma)
        Sigma = sigma_diag @ (B @ F_cov @ B.T + D) @ sigma_diag
        return pd.DataFrame(Sigma, index=tickers, columns=tickers)

    def correlation_matrix(
        self,
        returns: pd.DataFrame,
        method: Literal["pearson", "spearman", "distance"] = "pearson",
    ) -> pd.DataFrame:
        """
        Compute correlation matrix using Pearson, Spearman, or distance correlation.

        Distance correlation measures both linear and nonlinear dependence.
        """
        r = returns.dropna()
        tickers = r.columns.tolist()

        if method == "pearson":
            return r.corr()
        elif method == "spearman":
            return r.corr(method="spearman")
        elif method == "distance":
            # Székely distance correlation (simplified via centering)
            X = r.values
            n, p = X.shape
            dcor_mat = np.eye(p)
            for i in range(p):
                for j in range(i + 1, p):
                    d = self._distance_correlation(X[:, i], X[:, j])
                    dcor_mat[i, j] = d
                    dcor_mat[j, i] = d
            return pd.DataFrame(dcor_mat, index=tickers, columns=tickers)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

    @staticmethod
    def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Simplified distance correlation (Székely & Rizzo)."""
        n = len(x)
        if n < 4:
            return 0.0
        # Pairwise distance matrices
        ax = np.abs(x[:, None] - x[None, :])
        ay = np.abs(y[:, None] - y[None, :])
        # Double-center
        ax_c = ax - ax.mean(axis=0) - ax.mean(axis=1)[:, None] + ax.mean()
        ay_c = ay - ay.mean(axis=0) - ay.mean(axis=1)[:, None] + ay.mean()
        # dCov²
        dcov2_xy = float(np.mean(ax_c * ay_c))
        dcov2_xx = float(np.mean(ax_c ** 2))
        dcov2_yy = float(np.mean(ay_c ** 2))
        denom = float(np.sqrt(dcov2_xx * dcov2_yy))
        return float(np.sqrt(max(dcov2_xy, 0.0)) / denom) if denom > 0 else 0.0

    def expected_returns(
        self,
        returns: pd.DataFrame,
        method: Literal["historical", "momentum", "shrinkage"] = "historical",
        lookback: int = 252,
    ) -> pd.Series:
        """
        Estimate expected (annualized) returns.

        historical  — simple mean of past returns, annualized
        momentum    — trailing 6-month cumulative return, annualized
        shrinkage   — James-Stein shrinkage toward grand mean
        """
        r = returns.dropna().tail(lookback)

        if method == "historical":
            return r.mean() * TRADING_DAYS_PER_YEAR

        elif method == "momentum":
            # 6-month momentum
            window = min(126, len(r))
            recent = r.tail(window)
            cumulative = (1 + recent).prod() - 1
            annualized = (1 + cumulative) ** (TRADING_DAYS_PER_YEAR / window) - 1
            return annualized

        elif method == "shrinkage":
            # James-Stein shrinkage: shrink toward grand mean
            mu_hat = r.mean() * TRADING_DAYS_PER_YEAR
            grand_mean = float(mu_hat.mean())
            n, p = r.shape
            # Shrinkage intensity
            excess = mu_hat - grand_mean
            c = max(0.0, 1.0 - (p - 3) / max(n * (excess ** 2).sum(), _EPSILON))
            return pd.Series(grand_mean + c * excess.values, index=mu_hat.index)

        else:
            raise ValueError(f"Unknown returns method: {method}")

    def regularize_covariance(
        self,
        cov: pd.DataFrame,
        epsilon: float = 1e-5,
    ) -> pd.DataFrame:
        """
        Add epsilon * I to diagonal for numerical stability.
        Ensures positive-definiteness for optimization.
        """
        vals = cov.values + np.eye(len(cov)) * epsilon
        return pd.DataFrame(vals, index=cov.index, columns=cov.columns)


# ---------------------------------------------------------------------------
# 2. MeanVarianceOptimizer — Markowitz MVO
# ---------------------------------------------------------------------------

class MeanVarianceOptimizer:
    """
    Markowitz mean-variance portfolio optimization via scipy.optimize.

    Supports: max Sharpe, min variance, efficient frontier, constrained optimization.
    """

    def __init__(self, rf: float = 0.05):
        self.rf = rf
        self._rc = ReturnCovariance()

    def _validate_inputs(
        self,
        mu: pd.Series,
        sigma: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Align mu and sigma, return numpy arrays and asset list."""
        assets = [a for a in mu.index if a in sigma.index]
        mu_arr = mu[assets].values.astype(float)
        sig_arr = sigma.loc[assets, assets].values.astype(float)
        return mu_arr, sig_arr, assets

    def _portfolio_stats(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Return (expected_return, expected_vol, sharpe_ratio)."""
        port_return = float(weights @ mu)
        port_var = float(weights @ sigma @ weights)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        daily_rf = (1 + self.rf) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        # mu is already annualized; rf is annual
        sharpe = (port_return - self.rf) / port_vol if port_vol > _EPSILON else 0.0
        return port_return, port_vol, sharpe

    def _build_constraints(
        self,
        n: int,
        constraints: Optional["PortfolioConstraints"] = None,
        target_return: Optional[float] = None,
        target_vol: Optional[float] = None,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict], List[Tuple]]:
        """Build scipy constraint dicts and bounds for SLSQP."""
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n  # default: long-only

        if constraints is not None:
            # Long/short
            if constraints.allow_short:
                bounds = [(-constraints.max_short, 1.0)] * n

            # Max weight per asset
            if constraints.max_weight is not None:
                bounds = [(b[0], min(b[1], constraints.max_weight)) for b in bounds]

            # Min weight per asset
            if constraints.min_weight is not None:
                bounds = [(max(b[0], constraints.min_weight), b[1]) for b in bounds]

            # Market neutral
            if constraints.market_neutral:
                cons.append({"type": "eq", "fun": lambda w: np.sum(w)})

            # Target return
            if target_return is not None and mu is not None:
                cons.append({
                    "type": "eq",
                    "fun": lambda w, mu=mu, tr=target_return: float(w @ mu) - tr
                })

            # Target vol
            if target_vol is not None and sigma is not None:
                cons.append({
                    "type": "ineq",
                    "fun": lambda w, sigma=sigma, tv=target_vol: tv - float(np.sqrt(w @ sigma @ w))
                })

            # Max turnover
            if constraints.max_turnover is not None and constraints.current_weights is not None:
                w_curr = np.array(constraints.current_weights)
                cons.append({
                    "type": "ineq",
                    "fun": lambda w, wc=w_curr, mt=constraints.max_turnover:
                        mt - float(np.sum(np.abs(w - wc))) / 2.0
                })

        return cons, bounds

    def max_sharpe(
        self,
        mu: pd.Series,
        sigma: pd.DataFrame,
        rf: Optional[float] = None,
        constraints: Optional["PortfolioConstraints"] = None,
    ) -> OptimizationResult:
        """
        Find the maximum Sharpe ratio (tangency) portfolio.

        Uses SLSQP minimization of -Sharpe. Falls back to analytical
        solution for unconstrained long-only via σ^(-1)(μ-rf) normalization.
        """
        rf = rf if rf is not None else self.rf
        mu_arr, sig_arr, assets = self._validate_inputs(mu, sigma)
        sig_arr = self._rc.regularize_covariance(
            pd.DataFrame(sig_arr, index=assets, columns=assets)
        ).values

        n = len(assets)
        w0 = np.ones(n) / n

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ mu_arr)
            port_vol = float(np.sqrt(max(w @ sig_arr @ w, 0.0)))
            return -(port_ret - rf) / port_vol if port_vol > _EPSILON else 0.0

        cons, bounds = self._build_constraints(n, constraints)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_sharpe,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
            )

        w_opt = result.x if result.success else w0
        w_opt = np.clip(w_opt, 0.0, None)
        w_opt /= w_opt.sum() if w_opt.sum() > _EPSILON else 1.0

        exp_ret, exp_vol, sharpe = self._portfolio_stats(w_opt, mu_arr, sig_arr)

        return OptimizationResult(
            method="max_sharpe",
            weights=dict(zip(assets, w_opt.tolist())),
            expected_return=round(exp_ret, 4),
            expected_volatility=round(exp_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            success=result.success,
            message=result.message if hasattr(result, "message") else "",
        )

    def min_variance(
        self,
        sigma: pd.DataFrame,
        constraints: Optional["PortfolioConstraints"] = None,
    ) -> OptimizationResult:
        """
        Minimum variance portfolio: minimize w'Σw subject to constraints.
        """
        assets = sigma.index.tolist()
        sig_arr = self._rc.regularize_covariance(sigma).values
        n = len(assets)
        w0 = np.ones(n) / n

        def port_variance(w: np.ndarray) -> float:
            return float(w @ sig_arr @ w)

        # Dummy mu for constraint building
        mu_dummy = np.zeros(n)
        cons, bounds = self._build_constraints(n, constraints, mu=mu_dummy, sigma=sig_arr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                port_variance,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
            )

        w_opt = result.x if result.success else w0
        w_opt = np.clip(w_opt, 0.0, None)
        w_opt /= w_opt.sum() if w_opt.sum() > _EPSILON else 1.0

        port_var = float(w_opt @ sig_arr @ w_opt)
        port_vol = float(np.sqrt(max(port_var, 0.0)))

        return OptimizationResult(
            method="min_variance",
            weights=dict(zip(assets, w_opt.tolist())),
            expected_return=0.0,
            expected_volatility=round(port_vol, 4),
            sharpe_ratio=0.0,
            success=result.success,
            message=result.message if hasattr(result, "message") else "",
        )

    def efficient_frontier(
        self,
        mu: pd.Series,
        sigma: pd.DataFrame,
        n_points: int = 50,
        constraints: Optional["PortfolioConstraints"] = None,
    ) -> List[EfficientFrontierPoint]:
        """
        Trace the efficient frontier by optimizing min-variance at each target return.

        Returns n_points (return, vol, weights) tuples from min-var to max-return.
        """
        mu_arr, sig_arr, assets = self._validate_inputs(mu, sigma)
        sig_arr = self._rc.regularize_covariance(
            pd.DataFrame(sig_arr, index=assets, columns=assets)
        ).values
        n = len(assets)

        # Bounds for feasible returns
        min_ret_result = self.min_variance(pd.DataFrame(sig_arr, index=assets, columns=assets), constraints)
        max_ret = float(mu_arr.max())
        min_ret = float(min_ret_result.expected_return) if min_ret_result.success else float(mu_arr.min())
        # Ensure we have a positive spread
        if max_ret <= min_ret:
            max_ret = min_ret + 0.01

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_points: List[EfficientFrontierPoint] = []
        w0 = np.ones(n) / n

        for target_ret in target_returns:
            cons_base, bounds = self._build_constraints(n, constraints, target_return=target_ret, mu=mu_arr, sigma=sig_arr)
            # Add return constraint
            ret_con = {"type": "eq", "fun": lambda w, tr=target_ret: float(w @ mu_arr) - tr}
            cons_with_ret = cons_base + [ret_con] if not any(
                c.get("type") == "eq" and "mu_arr" in str(c.get("fun", "")) for c in cons_base
            ) else cons_base

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    lambda w: float(w @ sig_arr @ w),
                    w0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=cons_with_ret,
                    options={"ftol": 1e-9, "maxiter": 500, "disp": False},
                )

            if result.success:
                w = np.clip(result.x, 0.0, None)
                w /= w.sum() if w.sum() > _EPSILON else 1.0
                port_vol = float(np.sqrt(max(w @ sig_arr @ w, 0.0)))
                port_ret = float(w @ mu_arr)
                sharpe = (port_ret - self.rf) / port_vol if port_vol > _EPSILON else 0.0
                frontier_points.append(EfficientFrontierPoint(
                    target_return=round(target_ret, 4),
                    expected_return=round(port_ret, 4),
                    expected_volatility=round(port_vol, 4),
                    sharpe_ratio=round(sharpe, 4),
                    weights=dict(zip(assets, w.tolist())),
                ))
                w0 = result.x  # warm start

        return frontier_points

    def tangency_portfolio(
        self,
        mu: pd.Series,
        sigma: pd.DataFrame,
        rf: Optional[float] = None,
        constraints: Optional["PortfolioConstraints"] = None,
    ) -> OptimizationResult:
        """Alias for max_sharpe — the tangency portfolio maximizes Sharpe."""
        return self.max_sharpe(mu, sigma, rf=rf, constraints=constraints)

    def constrained_optimize(
        self,
        mu: pd.Series,
        sigma: pd.DataFrame,
        objective: Literal["sharpe", "min_var", "max_return"] = "sharpe",
        constraints: Optional["PortfolioConstraints"] = None,
        target_return: Optional[float] = None,
        target_vol: Optional[float] = None,
    ) -> OptimizationResult:
        """
        General constrained optimization with rich constraint set.
        """
        mu_arr, sig_arr, assets = self._validate_inputs(mu, sigma)
        sig_arr = self._rc.regularize_covariance(
            pd.DataFrame(sig_arr, index=assets, columns=assets)
        ).values
        n = len(assets)
        w0 = np.ones(n) / n

        if objective == "sharpe":
            def obj(w):
                pr = float(w @ mu_arr)
                pv = float(np.sqrt(max(w @ sig_arr @ w, 0.0)))
                return -(pr - self.rf) / pv if pv > _EPSILON else 0.0
        elif objective == "min_var":
            def obj(w):
                return float(w @ sig_arr @ w)
        elif objective == "max_return":
            def obj(w):
                return -float(w @ mu_arr)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        cons, bounds = self._build_constraints(
            n, constraints,
            target_return=target_return,
            target_vol=target_vol,
            mu=mu_arr,
            sigma=sig_arr,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
            )

        w_opt = result.x if result.success else w0
        w_opt = np.clip(w_opt, 0.0, None)
        if w_opt.sum() > _EPSILON:
            w_opt /= w_opt.sum()

        exp_ret, exp_vol, sharpe = self._portfolio_stats(w_opt, mu_arr, sig_arr)

        return OptimizationResult(
            method=f"constrained_{objective}",
            weights=dict(zip(assets, w_opt.tolist())),
            expected_return=round(exp_ret, 4),
            expected_volatility=round(exp_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            success=result.success,
            message=result.message if hasattr(result, "message") else "",
        )


# ---------------------------------------------------------------------------
# 3. BlackLittermanOptimizer
# ---------------------------------------------------------------------------

class BlackLittermanOptimizer:
    """
    Black-Litterman model for portfolio optimization.

    Blends market equilibrium returns (implied from market caps) with
    investor views to produce a posterior return estimate, then runs MVO.
    """

    def __init__(self, rf: float = 0.05):
        self.rf = rf
        self._rc = ReturnCovariance()
        self._mvo = MeanVarianceOptimizer(rf=rf)

    def implied_equilibrium_returns(
        self,
        market_caps: Dict[str, float],
        sigma: pd.DataFrame,
        delta: float = 2.5,
    ) -> pd.Series:
        """
        Compute implied equilibrium returns: π = δ × Σ × w_mkt

        δ = risk aversion coefficient (typically 2.5 for global equity)
        w_mkt = market-cap weights of assets
        """
        assets = [a for a in market_caps if a in sigma.index]
        total_cap = sum(market_caps[a] for a in assets)
        if total_cap < _EPSILON:
            return pd.Series(dtype=float)

        w_mkt = np.array([market_caps[a] / total_cap for a in assets])
        sig_arr = sigma.loc[assets, assets].values

        pi = delta * (sig_arr @ w_mkt)
        return pd.Series(pi, index=assets, name="equilibrium_returns")

    def _build_P_Q_Omega(
        self,
        assets: List[str],
        views: List[Dict[str, Any]],
        sigma: np.ndarray,
        tau: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct BL view matrices from views list.

        Each view dict:
        {
          "type": "absolute" | "relative",
          "assets": ["AAPL"] or ["AAPL", "MSFT"],   # relative: long first, short second
          "return": 0.10,    # expected return (annual)
          "confidence": 0.8  # 0-1 scale; higher → smaller Ω_ii
        }
        """
        k = len(views)
        n = len(assets)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))

        for i, view in enumerate(views):
            Q[i] = float(view["return"])
            confidence = float(view.get("confidence", 0.5))

            if view["type"] == "absolute":
                asset_name = view["assets"][0] if isinstance(view["assets"], list) else view["assets"]
                if asset_name in assets:
                    idx = assets.index(asset_name)
                    P[i, idx] = 1.0
            elif view["type"] == "relative":
                long_asset = view["assets"][0]
                short_asset = view["assets"][1] if len(view["assets"]) > 1 else None
                if long_asset in assets:
                    P[i, assets.index(long_asset)] = 1.0
                if short_asset and short_asset in assets:
                    P[i, assets.index(short_asset)] = -1.0

            # Omega_ii = (1 - confidence) * P_i' * (tau * Sigma) * P_i
            # High confidence → small Omega (view dominates prior)
            p_i = P[i, :]
            view_var = float(p_i @ (tau * sigma) @ p_i)
            Omega[i, i] = max((1.0 - confidence) * view_var, _EPSILON)

        return P, Q, Omega

    def bl_posterior(
        self,
        assets: List[str],
        pi: np.ndarray,
        sigma: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
        tau: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Black-Litterman posterior mean and covariance.

        μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)π + P'Ω^(-1)Q]
        Σ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1)
        """
        tau_sigma = tau * sigma
        try:
            tau_sigma_inv = np.linalg.inv(tau_sigma + np.eye(len(assets)) * 1e-8)
        except np.linalg.LinAlgError:
            tau_sigma_inv = np.linalg.pinv(tau_sigma)

        try:
            omega_inv = np.linalg.inv(Omega + np.eye(len(Omega)) * 1e-12)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.pinv(Omega)

        # BL precision matrix
        M = tau_sigma_inv + P.T @ omega_inv @ P

        try:
            M_inv = np.linalg.inv(M + np.eye(len(assets)) * 1e-8)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        # BL posterior mean
        mu_bl = M_inv @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

        # BL posterior covariance = original sigma + estimation uncertainty
        sigma_bl = sigma + M_inv

        return mu_bl, sigma_bl

    def optimize(
        self,
        assets: List[str],
        market_caps: Dict[str, float],
        sigma: pd.DataFrame,
        views: List[Dict[str, Any]],
        tau: float = 0.05,
        delta: float = 2.5,
        constraints: Optional["PortfolioConstraints"] = None,
    ) -> BLResult:
        """
        Full Black-Litterman optimization pipeline.

        1. Compute implied equilibrium returns π
        2. Build P, Q, Ω from views
        3. Compute BL posterior (μ_BL, Σ_BL)
        4. Run MVO on posterior returns

        Parameters
        ----------
        assets : List of asset tickers to include
        market_caps : Dict[ticker -> market_cap] for all assets
        sigma : Covariance matrix DataFrame
        views : List of view dicts (see _build_P_Q_Omega)
        tau : Scaling factor for uncertainty in prior (typically 0.01 - 0.10)
        delta : Risk aversion coefficient (typically 2.5)
        """
        avail = [a for a in assets if a in sigma.index and a in market_caps]
        if not avail:
            raise ValueError("No assets available in both sigma and market_caps")

        pi_series = self.implied_equilibrium_returns(market_caps, sigma.loc[avail, avail], delta=delta)
        pi = pi_series.values
        sig_arr = sigma.loc[avail, avail].values
        sig_arr = self._rc.regularize_covariance(
            pd.DataFrame(sig_arr, index=avail, columns=avail)
        ).values

        P, Q, Omega = self._build_P_Q_Omega(avail, views, sig_arr, tau)
        mu_bl, sigma_bl = self.bl_posterior(avail, pi, sig_arr, P, Q, Omega, tau)

        # Run MVO on BL posterior
        mu_bl_series = pd.Series(mu_bl, index=avail)
        sigma_bl_df = pd.DataFrame(sigma_bl, index=avail, columns=avail)
        mvo_result = self._mvo.max_sharpe(mu_bl_series, sigma_bl_df, constraints=constraints)

        prior_dict = {a: round(float(v), 4) for a, v in zip(avail, pi)}
        posterior_dict = {a: round(float(v), 4) for a, v in zip(avail, mu_bl)}

        return BLResult(
            assets=avail,
            prior_returns=prior_dict,
            posterior_returns=posterior_dict,
            posterior_covariance=sigma_bl,
            optimal_weights=mvo_result.weights,
            expected_return=mvo_result.expected_return,
            expected_volatility=mvo_result.expected_volatility,
            sharpe_ratio=mvo_result.sharpe_ratio,
            views_applied=len(views),
            tau=tau,
        )


# ---------------------------------------------------------------------------
# 4. EqualRiskContribution — Risk Parity
# ---------------------------------------------------------------------------

class EqualRiskContribution:
    """
    Equal Risk Contribution (ERC) / Risk Parity portfolio optimizer.

    Each asset contributes equal risk to the portfolio:
    RC_i = w_i × (Σw)_i / σ_p = σ_p / n  for all i

    Solved via cyclical coordinate descent (Bai, Scheinberg, Tutuncu 2016).
    """

    def __init__(self, target_vol: float = 0.10):
        self.target_vol = target_vol
        self._rc = ReturnCovariance()

    def _risk_contributions(
        self,
        w: np.ndarray,
        sigma: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute portfolio vol and risk contributions."""
        sigma_w = sigma @ w
        port_var = float(w @ sigma_w)
        port_vol = float(np.sqrt(max(port_var, _EPSILON)))
        rc = w * sigma_w / port_vol
        return port_vol, rc

    def _erc_objective(self, w: np.ndarray, sigma: np.ndarray) -> float:
        """
        Convex surrogate for ERC: minimize sum of squared log-weight differences.
        From Maillard, Roncalli, Teïletche (2010).
        """
        port_vol, rc = self._risk_contributions(w, sigma)
        n = len(w)
        # RC_i = w_i * (Σw)_i; we want all equal → minimize variance of RC
        rc_mean = rc.mean()
        return float(np.sum((rc - rc_mean) ** 2))

    def optimize(
        self,
        sigma: pd.DataFrame,
        target_vol: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Find ERC weights via cyclical coordinate descent.

        Scales weights to target_vol after finding the ERC solution.
        """
        tv = target_vol if target_vol is not None else self.target_vol
        assets = sigma.index.tolist()
        n = len(assets)
        sig_arr = self._rc.regularize_covariance(sigma).values

        # Cyclical coordinate descent
        w = self._cyclical_descent(sig_arr, n)

        # Scale to target volatility
        port_vol, rc = self._risk_contributions(w, sig_arr)
        if tv is not None and port_vol > _EPSILON:
            w = w * tv / port_vol

        # Recompute after scaling
        port_vol, rc = self._risk_contributions(w / w.sum(), sig_arr)

        return OptimizationResult(
            method="erc",
            weights=dict(zip(assets, (w / w.sum()).tolist())),
            expected_return=0.0,
            expected_volatility=round(port_vol, 4),
            sharpe_ratio=0.0,
            success=True,
            message="ERC solution via cyclical coordinate descent",
            metadata={
                "risk_contributions": dict(zip(assets, rc.tolist())),
                "rc_equal_pct": round(float(rc.std() / rc.mean()) * 100, 2) if rc.mean() > 0 else None,
            },
        )

    def _cyclical_descent(
        self,
        sigma: np.ndarray,
        n: int,
        tol: float = 1e-10,
        max_iter: int = 5000,
    ) -> np.ndarray:
        """
        Cyclical coordinate descent for ERC.

        Update rule: w_i ← √[(b_i) / σ_ii] where b_i handles cross-terms.
        Based on Bai, Scheinberg, Tutuncu (2016) Newton-type method.
        """
        w = np.ones(n) / n  # equal-weight start

        for iteration in range(max_iter):
            w_old = w.copy()

            for i in range(n):
                # Sigma_w excluding asset i
                sigma_w = sigma @ w
                # Coefficient for w_i in the quadratic equation
                a = sigma[i, i]
                b = sigma_w[i] - sigma[i, i] * w[i]
                # Target RC: 1/n (sum of all RC = port_vol)
                # Quadratic: sigma[i,i]*w_i^2 + b*w_i - (port_vol/n) = 0
                # Use gradient step instead of quadratic solve for stability
                port_var = max(float(w @ sigma @ w), _EPSILON)
                port_vol = float(np.sqrt(port_var))
                rc_i = float(w[i] * sigma_w[i]) / port_vol
                target_rc = port_vol / n
                # Newton step on the ERC condition
                grad = rc_i - target_rc
                hess = float(sigma[i, i]) / port_vol  # simplified Hessian
                step = grad / max(abs(hess), _EPSILON)
                w[i] = max(w[i] - 0.5 * step, 1e-8)

            w /= w.sum()

            # Convergence
            if np.max(np.abs(w - w_old)) < tol:
                logger.debug("ERC converged at iteration %d", iteration)
                break

        return w

    def leverage_to_target(
        self,
        weights: np.ndarray,
        sigma: np.ndarray,
        target_vol: float,
    ) -> np.ndarray:
        """Scale portfolio weights to achieve exact target volatility."""
        port_var = float(weights @ sigma @ weights)
        port_vol = float(np.sqrt(max(port_var, _EPSILON)))
        scale = target_vol / port_vol if port_vol > _EPSILON else 1.0
        return weights * scale

    def risk_parity_backtest(
        self,
        returns: pd.DataFrame,
        rebalance_freq: str = "ME",
        target_vol: float = 0.10,
    ) -> BacktestResult:
        """
        Backtest risk parity portfolio with monthly rebalancing.

        Returns comprehensive performance metrics and weights history.
        """
        returns_clean = returns.dropna(how="all").fillna(0.0)
        assets = returns_clean.columns.tolist()

        # Group into rebalance periods
        rebalance_dates = returns_clean.resample(rebalance_freq).last().index

        port_rets: List[float] = []
        port_dates: List[Any] = []
        weights_history: List[Dict[str, float]] = []
        current_weights = np.ones(len(assets)) / len(assets)

        for i, reb_date in enumerate(rebalance_dates[:-1]):
            next_reb = rebalance_dates[i + 1]
            # Estimation window: all data up to rebalance date
            hist = returns_clean.loc[:reb_date]
            if len(hist) < 30:
                continue

            sig_arr = self._rc.ledoit_wolf_shrinkage(hist).values
            w_erc = self._cyclical_descent(sig_arr, len(assets))
            # Scale to target vol
            w_erc = self.leverage_to_target(w_erc, sig_arr, target_vol)
            # Normalize for long-only
            w_erc = np.clip(w_erc, 0.0, None)
            w_erc /= w_erc.sum() if w_erc.sum() > _EPSILON else 1.0
            current_weights = w_erc
            weights_history.append(dict(zip(assets, w_erc.tolist())))

            # Apply weights to next period returns
            period_rets = returns_clean.loc[reb_date:next_reb]
            if period_rets.empty:
                continue
            for dt, row in period_rets.iterrows():
                port_rets.append(float(current_weights @ row.values))
                port_dates.append(dt)

        if not port_rets:
            return BacktestResult(
                method="risk_parity", returns=pd.Series(dtype=float),
                cumulative_returns=pd.Series(dtype=float),
                annualized_return=0.0, annualized_volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
                n_rebalances=0, weights_history=[],
            )

        ret_series = pd.Series(port_rets, index=pd.DatetimeIndex(port_dates), name="risk_parity")
        cum_rets = (1 + ret_series).cumprod()
        ann_ret = float(ret_series.mean() * TRADING_DAYS_PER_YEAR)
        ann_vol = float(ret_series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = (ann_ret - 0.05) / ann_vol if ann_vol > _EPSILON else 0.0
        drawdown = (cum_rets - cum_rets.cummax()) / cum_rets.cummax()
        max_dd = float(drawdown.min())
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

        return BacktestResult(
            method="risk_parity",
            returns=ret_series,
            cumulative_returns=cum_rets,
            annualized_return=round(ann_ret, 4),
            annualized_volatility=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            calmar_ratio=round(calmar, 4),
            n_rebalances=len(weights_history),
            weights_history=weights_history,
        )


# ---------------------------------------------------------------------------
# 5. HierarchicalRiskParity — Lopez de Prado HRP
# ---------------------------------------------------------------------------

class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (Lopez de Prado, 2016).

    Steps:
    1. Compute correlation distance matrix: d = √(0.5 × (1 - ρ))
    2. Hierarchical clustering (Ward linkage via scipy)
    3. Quasi-diagonalization: reorder by cluster similarity
    4. Recursive bisection allocation

    More robust than MVO out-of-sample due to avoidance of matrix inversion.
    """

    def _correlation_distance(self, corr: pd.DataFrame) -> np.ndarray:
        """Distance matrix from correlation: d_ij = √(0.5 × (1 - ρ_ij))."""
        return np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0.0, 1.0))

    def _get_cluster_var(
        self,
        cov: pd.DataFrame,
        cluster_items: List[int],
    ) -> float:
        """Minimum-variance weight cluster variance for inverse-vol weighting."""
        cov_slice = cov.iloc[cluster_items, cluster_items]
        # Inverse vol weights
        vols = np.sqrt(np.diag(cov_slice.values))
        inv_vol = 1.0 / np.where(vols > _EPSILON, vols, 1.0)
        w = inv_vol / inv_vol.sum()
        return float(w @ cov_slice.values @ w)

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        sorted_items: List[int],
    ) -> np.ndarray:
        """
        Recursive bisection allocation.

        Split portfolio into two halves based on cluster dendrogram,
        allocate inversely proportional to cluster variance.
        """
        weights = pd.Series(1.0, index=sorted_items)
        cluster_items = [sorted_items]

        while len(cluster_items) > 0:
            cluster_items = [
                item[j:k]
                for item in cluster_items
                for j, k in ((0, len(item) // 2), (len(item) // 2, len(item)))
                if len(item) > 1
            ]

            for i in range(0, len(cluster_items), 2):
                if i + 1 >= len(cluster_items):
                    break
                left = cluster_items[i]
                right = cluster_items[i + 1]
                var_left = self._get_cluster_var(cov, left)
                var_right = self._get_cluster_var(cov, right)
                total_var = var_left + var_right
                alloc_left = 1.0 - var_left / total_var if total_var > _EPSILON else 0.5
                weights[left] *= alloc_left
                weights[right] *= (1.0 - alloc_left)

        return weights.values

    def _quasi_diagonalize(
        self,
        linkage_matrix: np.ndarray,
        n: int,
    ) -> List[int]:
        """
        Reorder assets by dendrogram structure (quasi-diagonalization).
        Returns sorted list of asset indices.
        """
        # Build cluster ordering from linkage matrix
        sorted_idx = list(range(n))
        linkage = linkage_matrix.astype(int)

        # Map cluster IDs to leaves
        cluster_map: Dict[int, List[int]] = {i: [i] for i in range(n)}

        for row in linkage_matrix:
            left_id = int(row[0])
            right_id = int(row[1])
            new_id = n + list(linkage_matrix[:, 0:2].flatten().tolist()).index(row[0]) // 2 \
                if left_id < n else left_id
            # Use scipy's dendrogram ordering instead
            pass

        # Use scipy's leaves_list for correct ordering
        return list(hierarchy.leaves_list(linkage_matrix))

    def hrp_weights(
        self,
        returns: pd.DataFrame,
        cov_method: Literal["sample", "ledoit_wolf", "ewma"] = "ledoit_wolf",
    ) -> OptimizationResult:
        """
        Compute HRP portfolio weights from historical returns.

        Parameters
        ----------
        returns : DataFrame of asset daily returns
        cov_method : Covariance estimation method

        Returns
        -------
        OptimizationResult with HRP weights
        """
        rc = ReturnCovariance()
        r = returns.dropna(how="all")
        assets = r.columns.tolist()

        # Covariance and correlation
        if cov_method == "sample":
            cov = rc.sample_covariance(r)
        elif cov_method == "ledoit_wolf":
            cov = rc.ledoit_wolf_shrinkage(r)
        elif cov_method == "ewma":
            cov = rc.ewma_covariance(r)
        else:
            cov = rc.sample_covariance(r)

        corr = r.corr()

        # Step 1: Distance matrix
        dist = self._correlation_distance(corr)

        # Step 2: Hierarchical clustering (Ward linkage)
        condensed = squareform(dist, checks=False)
        condensed = np.clip(condensed, 0.0, None)
        linkage_mat = hierarchy.linkage(condensed, method="ward")

        # Step 3: Quasi-diagonalization
        sorted_idx = hierarchy.leaves_list(linkage_mat).tolist()

        # Step 4: Recursive bisection
        w_arr = self._recursive_bisection(
            cov.iloc[sorted_idx, sorted_idx],
            list(range(len(sorted_idx))),
        )

        # Map back to original asset order
        weights_sorted = {assets[sorted_idx[i]]: float(w_arr[i]) for i in range(len(sorted_idx))}
        w_final = np.array([weights_sorted.get(a, 0.0) for a in assets])
        w_final = np.clip(w_final, 0.0, None)
        w_final /= w_final.sum() if w_final.sum() > _EPSILON else 1.0

        # Portfolio stats
        sig_arr = cov.values
        port_var = float(w_final @ sig_arr @ w_final)
        port_vol = float(np.sqrt(max(port_var, 0.0)))

        return OptimizationResult(
            method="hrp",
            weights=dict(zip(assets, w_final.tolist())),
            expected_return=0.0,
            expected_volatility=round(port_vol, 4),
            sharpe_ratio=0.0,
            success=True,
            message="HRP via hierarchical clustering + recursive bisection",
            metadata={
                "cluster_order": [assets[i] for i in sorted_idx],
                "n_clusters": len(linkage_mat) + 1,
            },
        )

    def hrp_backtest(
        self,
        returns: pd.DataFrame,
        rebalance_freq: str = "ME",
    ) -> BacktestResult:
        """Backtest HRP with monthly rebalancing."""
        returns_clean = returns.dropna(how="all").fillna(0.0)
        assets = returns_clean.columns.tolist()
        rebalance_dates = returns_clean.resample(rebalance_freq).last().index

        port_rets: List[float] = []
        port_dates: List[Any] = []
        weights_history: List[Dict[str, float]] = []
        current_weights = np.ones(len(assets)) / len(assets)

        for i, reb_date in enumerate(rebalance_dates[:-1]):
            next_reb = rebalance_dates[i + 1]
            hist = returns_clean.loc[:reb_date]
            if len(hist) < 30:
                continue

            try:
                res = self.hrp_weights(hist)
                current_weights = np.array([res.weights.get(a, 0.0) for a in assets])
                weights_history.append(res.weights)
            except Exception as exc:
                logger.warning("HRP failed at %s: %s", reb_date, exc)

            period_rets = returns_clean.loc[reb_date:next_reb]
            for dt, row in period_rets.iterrows():
                port_rets.append(float(current_weights @ row.values))
                port_dates.append(dt)

        if not port_rets:
            return BacktestResult(
                method="hrp", returns=pd.Series(dtype=float),
                cumulative_returns=pd.Series(dtype=float),
                annualized_return=0.0, annualized_volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
                n_rebalances=0, weights_history=[],
            )

        ret_series = pd.Series(port_rets, index=pd.DatetimeIndex(port_dates), name="hrp")
        cum_rets = (1 + ret_series).cumprod()
        ann_ret = float(ret_series.mean() * TRADING_DAYS_PER_YEAR)
        ann_vol = float(ret_series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = (ann_ret - 0.05) / ann_vol if ann_vol > _EPSILON else 0.0
        dd = (cum_rets - cum_rets.cummax()) / cum_rets.cummax()
        max_dd = float(dd.min())
        calmar = ann_ret / abs(max_dd) if max_dd < -_EPSILON else 0.0

        return BacktestResult(
            method="hrp",
            returns=ret_series,
            cumulative_returns=cum_rets,
            annualized_return=round(ann_ret, 4),
            annualized_volatility=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            calmar_ratio=round(calmar, 4),
            n_rebalances=len(weights_history),
            weights_history=weights_history,
        )


# ---------------------------------------------------------------------------
# 6. PortfolioConstraints
# ---------------------------------------------------------------------------

class PortfolioConstraints:
    """
    Rich constraint set for portfolio optimization.

    Supports: long-only, long/short, market neutral, max/min weight,
    sector constraints, turnover constraint, tracking error constraint.
    """

    def __init__(
        self,
        allow_short: bool = False,
        max_short: float = 0.3,
        max_weight: Optional[float] = 0.40,
        min_weight: Optional[float] = None,
        market_neutral: bool = False,
        sector_limits: Optional[Dict[str, float]] = None,
        max_turnover: Optional[float] = 0.20,
        current_weights: Optional[List[float]] = None,
        max_tracking_error: Optional[float] = 0.05,
        benchmark_weights: Optional[List[float]] = None,
        max_factor_tilt: Optional[Dict[str, float]] = None,
    ):
        self.allow_short = allow_short
        self.max_short = max_short
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.market_neutral = market_neutral
        self.sector_limits = sector_limits or {}
        self.max_turnover = max_turnover
        self.current_weights = current_weights
        self.max_tracking_error = max_tracking_error
        self.benchmark_weights = benchmark_weights
        self.max_factor_tilt = max_factor_tilt or {}

    @classmethod
    def long_only(cls, max_weight: float = 0.40) -> "PortfolioConstraints":
        """Long-only with max 40% per asset."""
        return cls(allow_short=False, max_weight=max_weight)

    @classmethod
    def long_short_market_neutral(
        cls, max_short: float = 0.30
    ) -> "PortfolioConstraints":
        """Market neutral long/short."""
        return cls(allow_short=True, max_short=max_short, market_neutral=True, max_weight=1.0)

    @classmethod
    def low_turnover(
        cls, current_weights: List[float], max_turnover: float = 0.20
    ) -> "PortfolioConstraints":
        """Low turnover constraint for live portfolios."""
        return cls(max_turnover=max_turnover, current_weights=current_weights)

    def add_sector_constraint(
        self,
        sector: str,
        asset_indices: List[int],
        max_pct: float = 0.30,
    ) -> Dict[str, Any]:
        """
        Build a sector-level constraint dict for scipy.

        Returns a constraint dict: sum of weights in sector <= max_pct.
        """
        return {
            "type": "ineq",
            "fun": lambda w, idx=asset_indices, mp=max_pct:
                mp - sum(w[i] for i in idx)
        }

    def build_all_constraints(
        self,
        n: int,
        asset_to_sector: Optional[Dict[int, str]] = None,
        sigma: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the full set of scipy constraint dicts.

        Parameters
        ----------
        n : Number of assets
        asset_to_sector : Dict[asset_index -> sector_name]
        sigma : Covariance matrix (for tracking error constraint)
        """
        cons = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]

        if self.market_neutral:
            cons.append({"type": "eq", "fun": lambda w: float(np.sum(w))})

        # Sector constraints
        if asset_to_sector and self.sector_limits:
            sector_to_indices: Dict[str, List[int]] = {}
            for idx, sector in asset_to_sector.items():
                sector_to_indices.setdefault(sector, []).append(idx)
            for sector, max_pct in self.sector_limits.items():
                if sector in sector_to_indices:
                    indices = sector_to_indices[sector]
                    cons.append(self.add_sector_constraint(sector, indices, max_pct))

        # Turnover constraint
        if self.max_turnover is not None and self.current_weights is not None:
            w_curr = np.array(self.current_weights[:n])
            cons.append({
                "type": "ineq",
                "fun": lambda w, wc=w_curr, mt=self.max_turnover:
                    mt * 2 - float(np.sum(np.abs(w - wc)))
            })

        # Tracking error constraint
        if self.max_tracking_error is not None and self.benchmark_weights is not None and sigma is not None:
            w_bench = np.array(self.benchmark_weights[:n])
            max_te = self.max_tracking_error
            cons.append({
                "type": "ineq",
                "fun": lambda w, wb=w_bench, sig=sigma, te=max_te:
                    te - float(np.sqrt(max((w - wb) @ sig @ (w - wb), 0.0)))
            })

        return cons


# ---------------------------------------------------------------------------
# Optimizer Facade
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Unified facade combining all optimization methods.

    Provides compare_methods() for side-by-side evaluation and
    portfolio_backtest() for walk-forward validation.
    """

    def __init__(self, rf: float = 0.05):
        self.rf = rf
        self._rc = ReturnCovariance()
        self._mvo = MeanVarianceOptimizer(rf=rf)
        self._bl = BlackLittermanOptimizer(rf=rf)
        self._erc = EqualRiskContribution()
        self._hrp = HierarchicalRiskParity()

    def optimize(
        self,
        tickers: List[str],
        method: Literal["mvo_sharpe", "mvo_minvar", "erc", "hrp", "bl"] = "mvo_sharpe",
        start: str = "2015-01-01",
        end: Optional[str] = None,
        cov_method: Literal["sample", "ledoit_wolf", "ewma"] = "ledoit_wolf",
        mu_method: Literal["historical", "momentum", "shrinkage"] = "historical",
        constraints: Optional[PortfolioConstraints] = None,
        views: Optional[List[Dict[str, Any]]] = None,
        market_caps: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Unified optimization entry point.

        Downloads returns, estimates covariance and expected returns,
        then calls the appropriate optimizer.
        """
        returns = self._rc.get_returns(tickers, start=start, end=end)
        if returns.empty:
            return OptimizationResult(
                method=method, weights={t: 1/len(tickers) for t in tickers},
                expected_return=0.0, expected_volatility=0.0,
                sharpe_ratio=0.0, success=False, message="No return data"
            )

        # Covariance
        if cov_method == "sample":
            cov = self._rc.sample_covariance(returns)
        elif cov_method == "ledoit_wolf":
            cov = self._rc.ledoit_wolf_shrinkage(returns)
        elif cov_method == "ewma":
            cov = self._rc.ewma_covariance(returns)
        else:
            cov = self._rc.ledoit_wolf_shrinkage(returns)

        if method == "mvo_sharpe":
            mu = self._rc.expected_returns(returns, method=mu_method)
            return self._mvo.max_sharpe(mu, cov, constraints=constraints)

        elif method == "mvo_minvar":
            return self._mvo.min_variance(cov, constraints=constraints)

        elif method == "erc":
            return self._erc.optimize(cov)

        elif method == "hrp":
            return self._hrp.hrp_weights(returns, cov_method=cov_method)

        elif method == "bl":
            if views is None or market_caps is None:
                return OptimizationResult(
                    method="bl", weights={t: 1/len(tickers) for t in tickers},
                    expected_return=0.0, expected_volatility=0.0,
                    sharpe_ratio=0.0, success=False,
                    message="BL requires views and market_caps"
                )
            bl_result = self._bl.optimize(
                assets=tickers, market_caps=market_caps,
                sigma=cov, views=views, constraints=constraints
            )
            return OptimizationResult(
                method="bl",
                weights=bl_result.optimal_weights,
                expected_return=bl_result.expected_return,
                expected_volatility=bl_result.expected_volatility,
                sharpe_ratio=bl_result.sharpe_ratio,
                success=True,
                metadata={"views_applied": bl_result.views_applied, "tau": bl_result.tau},
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def compare_methods(
        self,
        tickers: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        cov_method: str = "ledoit_wolf",
    ) -> Dict[str, OptimizationResult]:
        """
        Run MVO (Sharpe), Min-Var, ERC, and HRP on the same dataset.
        Returns a dict of method_name -> OptimizationResult for comparison.
        """
        returns = self._rc.get_returns(tickers, start=start, end=end)
        if returns.empty:
            return {}

        cov = self._rc.ledoit_wolf_shrinkage(returns)
        mu_hist = self._rc.expected_returns(returns, method="historical")
        mu_mom = self._rc.expected_returns(returns, method="momentum")

        results: Dict[str, OptimizationResult] = {}

        # MVO Max Sharpe (historical mu)
        results["mvo_sharpe_hist"] = self._mvo.max_sharpe(mu_hist, cov)
        # MVO Max Sharpe (momentum mu)
        results["mvo_sharpe_mom"] = self._mvo.max_sharpe(mu_mom, cov)
        # Min Variance
        results["min_variance"] = self._mvo.min_variance(cov)
        # Equal Weight (naive benchmark)
        n = len(tickers)
        ew = np.ones(n) / n
        mu_arr = mu_hist[tickers].values
        sig_arr = cov.loc[tickers, tickers].values
        ew_ret = float(ew @ mu_arr)
        ew_vol = float(np.sqrt(max(ew @ sig_arr @ ew, 0.0)))
        ew_sharpe = (ew_ret - self.rf) / ew_vol if ew_vol > _EPSILON else 0.0
        results["equal_weight"] = OptimizationResult(
            method="equal_weight",
            weights={t: 1/n for t in tickers},
            expected_return=round(ew_ret, 4),
            expected_volatility=round(ew_vol, 4),
            sharpe_ratio=round(ew_sharpe, 4),
            success=True,
        )
        # ERC
        results["erc"] = self._erc.optimize(cov)
        # HRP
        results["hrp"] = self._hrp.hrp_weights(returns, cov_method="ledoit_wolf")

        return results

    def portfolio_backtest(
        self,
        tickers: List[str],
        method: Literal["mvo_sharpe", "min_variance", "erc", "hrp", "equal_weight"] = "hrp",
        start: str = "2010-01-01",
        end: Optional[str] = None,
        rebalance_freq: str = "ME",
        estimation_window: int = 252,
        target_vol: Optional[float] = None,
    ) -> BacktestResult:
        """
        Walk-forward portfolio backtest with periodic rebalancing.

        Estimation window: trailing estimation_window days before each rebalance date.
        No lookahead: covariance and expected returns only use past data.
        """
        returns = self._rc.get_returns(tickers, start=start, end=end)
        if returns.empty:
            return BacktestResult(
                method=method, returns=pd.Series(dtype=float),
                cumulative_returns=pd.Series(dtype=float),
                annualized_return=0.0, annualized_volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
                n_rebalances=0, weights_history=[],
            )

        if method == "hrp":
            return self._hrp.hrp_backtest(returns, rebalance_freq=rebalance_freq)
        elif method == "erc":
            return self._erc.risk_parity_backtest(
                returns, rebalance_freq=rebalance_freq, target_vol=target_vol or 0.10
            )

        # Walk-forward for MVO methods
        returns_clean = returns.dropna(how="all").fillna(0.0)
        assets = returns_clean.columns.tolist()
        rebalance_dates = returns_clean.resample(rebalance_freq).last().index

        port_rets: List[float] = []
        port_dates: List[Any] = []
        weights_history: List[Dict[str, float]] = []
        current_weights = np.ones(len(assets)) / len(assets)

        for i, reb_date in enumerate(rebalance_dates[:-1]):
            next_reb = rebalance_dates[i + 1]
            # Estimation window
            hist_start = reb_date - timedelta(days=estimation_window)
            hist = returns_clean.loc[hist_start:reb_date]
            if len(hist) < 60:
                continue

            cov = self._rc.ledoit_wolf_shrinkage(hist)
            mu = self._rc.expected_returns(hist, method="historical")

            try:
                if method == "mvo_sharpe":
                    res = self._mvo.max_sharpe(mu, cov)
                elif method == "min_variance":
                    res = self._mvo.min_variance(cov)
                elif method == "equal_weight":
                    n = len(assets)
                    res = OptimizationResult(
                        method="equal_weight",
                        weights={t: 1/n for t in assets},
                        expected_return=0.0, expected_volatility=0.0,
                        sharpe_ratio=0.0, success=True,
                    )
                else:
                    continue

                current_weights = np.array([res.weights.get(a, 0.0) for a in assets])
                weights_history.append(res.weights)
            except Exception as exc:
                logger.warning("Optimization failed at %s: %s", reb_date, exc)

            period_rets = returns_clean.loc[reb_date:next_reb]
            for dt, row in period_rets.iterrows():
                port_rets.append(float(current_weights @ row.values))
                port_dates.append(dt)

        if not port_rets:
            return BacktestResult(
                method=method, returns=pd.Series(dtype=float),
                cumulative_returns=pd.Series(dtype=float),
                annualized_return=0.0, annualized_volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
                n_rebalances=0, weights_history=[],
            )

        ret_series = pd.Series(port_rets, index=pd.DatetimeIndex(port_dates), name=method)
        cum_rets = (1 + ret_series).cumprod()
        ann_ret = float(ret_series.mean() * TRADING_DAYS_PER_YEAR)
        ann_vol = float(ret_series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = (ann_ret - self.rf) / ann_vol if ann_vol > _EPSILON else 0.0
        dd = (cum_rets - cum_rets.cummax()) / cum_rets.cummax()
        max_dd = float(dd.min())
        calmar = ann_ret / abs(max_dd) if max_dd < -_EPSILON else 0.0

        return BacktestResult(
            method=method,
            returns=ret_series,
            cumulative_returns=cum_rets,
            annualized_return=round(ann_ret, 4),
            annualized_volatility=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            calmar_ratio=round(calmar, 4),
            n_rebalances=len(weights_history),
            weights_history=weights_history,
        )

    def efficient_frontier_data(
        self,
        tickers: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        n_points: int = 50,
    ) -> List[EfficientFrontierPoint]:
        """Compute and return efficient frontier points."""
        returns = self._rc.get_returns(tickers, start=start, end=end)
        if returns.empty:
            return []
        cov = self._rc.ledoit_wolf_shrinkage(returns)
        mu = self._rc.expected_returns(returns, method="historical")
        return self._mvo.efficient_frontier(mu, cov, n_points=n_points)


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    optimizer_router = APIRouter(prefix="/optimize", tags=["Portfolio Optimizer"])

    class MVORequest(BaseModel):
        tickers: List[str]
        start: str = "2015-01-01"
        end: Optional[str] = None
        cov_method: str = "ledoit_wolf"
        mu_method: str = "historical"
        rf: float = 0.05
        max_weight: Optional[float] = 0.40
        allow_short: bool = False

    class BLRequest(BaseModel):
        tickers: List[str]
        market_caps: Dict[str, float]
        views: List[Dict[str, Any]]
        start: str = "2015-01-01"
        end: Optional[str] = None
        tau: float = 0.05
        delta: float = 2.5

    class ERCRequest(BaseModel):
        tickers: List[str]
        start: str = "2015-01-01"
        end: Optional[str] = None
        target_vol: float = 0.10

    class HRPRequest(BaseModel):
        tickers: List[str]
        start: str = "2015-01-01"
        end: Optional[str] = None
        cov_method: str = "ledoit_wolf"

    class CompareRequest(BaseModel):
        tickers: List[str]
        start: str = "2015-01-01"
        end: Optional[str] = None

    class FrontierRequest(BaseModel):
        tickers: List[str]
        start: str = "2015-01-01"
        end: Optional[str] = None
        n_points: int = 50

    class BacktestRequest(BaseModel):
        tickers: List[str]
        method: str = "hrp"
        start: str = "2010-01-01"
        end: Optional[str] = None
        rebalance_freq: str = "ME"
        estimation_window: int = 252

    def _result_to_dict(r: OptimizationResult) -> Dict[str, Any]:
        return {
            "method": r.method,
            "weights": r.weights,
            "expected_return": r.expected_return,
            "expected_volatility": r.expected_volatility,
            "sharpe_ratio": r.sharpe_ratio,
            "success": r.success,
            "message": r.message,
            "metadata": r.metadata,
        }

    def _backtest_to_dict(r: BacktestResult) -> Dict[str, Any]:
        return {
            "method": r.method,
            "annualized_return": r.annualized_return,
            "annualized_volatility": r.annualized_volatility,
            "sharpe_ratio": r.sharpe_ratio,
            "max_drawdown": r.max_drawdown,
            "calmar_ratio": r.calmar_ratio,
            "n_rebalances": r.n_rebalances,
            "final_weights": r.weights_history[-1] if r.weights_history else {},
            "cumulative_return": float(r.cumulative_returns.iloc[-1] - 1) if not r.cumulative_returns.empty else None,
        }

    _optimizer_singleton = PortfolioOptimizer()

    @optimizer_router.post("/mvo")
    async def optimize_mvo(req: MVORequest) -> Dict[str, Any]:
        """Mean-Variance (Max Sharpe) portfolio optimization."""
        if not req.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        constraints = PortfolioConstraints(
            allow_short=req.allow_short,
            max_weight=req.max_weight,
        )
        result = _optimizer_singleton.optimize(
            tickers=req.tickers,
            method="mvo_sharpe",
            start=req.start,
            end=req.end,
            cov_method=req.cov_method,
            mu_method=req.mu_method,
            constraints=constraints,
        )
        return _result_to_dict(result)

    @optimizer_router.post("/bl")
    async def optimize_bl(req: BLRequest) -> Dict[str, Any]:
        """Black-Litterman portfolio optimization."""
        if not req.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        rc = ReturnCovariance()
        returns = rc.get_returns(req.tickers, start=req.start, end=req.end)
        if returns.empty:
            raise HTTPException(status_code=400, detail="No return data available")
        cov = rc.ledoit_wolf_shrinkage(returns)
        bl = BlackLittermanOptimizer(rf=0.05)
        bl_result = bl.optimize(
            assets=req.tickers,
            market_caps=req.market_caps,
            sigma=cov,
            views=req.views,
            tau=req.tau,
            delta=req.delta,
        )
        return {
            "assets": bl_result.assets,
            "prior_returns": bl_result.prior_returns,
            "posterior_returns": bl_result.posterior_returns,
            "optimal_weights": bl_result.optimal_weights,
            "expected_return": bl_result.expected_return,
            "expected_volatility": bl_result.expected_volatility,
            "sharpe_ratio": bl_result.sharpe_ratio,
            "views_applied": bl_result.views_applied,
            "tau": bl_result.tau,
        }

    @optimizer_router.post("/erc")
    async def optimize_erc(req: ERCRequest) -> Dict[str, Any]:
        """Equal Risk Contribution (Risk Parity) optimization."""
        if not req.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        result = _optimizer_singleton.optimize(
            tickers=req.tickers,
            method="erc",
            start=req.start,
            end=req.end,
        )
        return _result_to_dict(result)

    @optimizer_router.post("/hrp")
    async def optimize_hrp(req: HRPRequest) -> Dict[str, Any]:
        """Hierarchical Risk Parity optimization."""
        if not req.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        rc = ReturnCovariance()
        returns = rc.get_returns(req.tickers, start=req.start, end=req.end)
        if returns.empty:
            raise HTTPException(status_code=400, detail="No return data available")
        hrp = HierarchicalRiskParity()
        result = hrp.hrp_weights(returns, cov_method=req.cov_method)
        return _result_to_dict(result)

    @optimizer_router.post("/efficient-frontier")
    async def efficient_frontier(req: FrontierRequest) -> Dict[str, Any]:
        """Compute the efficient frontier (return-risk tradeoff)."""
        if not req.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        points = _optimizer_singleton.efficient_frontier_data(
            tickers=req.tickers,
            start=req.start,
            end=req.end,
            n_points=req.n_points,
        )
        return {
            "n_points": len(points),
            "frontier": [
                {
                    "target_return": p.target_return,
                    "expected_return": p.expected_return,
                    "expected_volatility": p.expected_volatility,
                    "sharpe_ratio": p.sharpe_ratio,
                    "weights": p.weights,
                }
                for p in points
            ],
        }

    @optimizer_router.post("/compare-methods")
    async def compare_methods(req: CompareRequest) -> Dict[str, Any]:
        """Compare MVO, Min-Var, ERC, HRP, and Equal-Weight side by side."""
        if not req.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        results = _optimizer_singleton.compare_methods(
            tickers=req.tickers,
            start=req.start,
            end=req.end,
        )
        return {
            method: _result_to_dict(res)
            for method, res in results.items()
        }

    @optimizer_router.post("/backtest")
    async def backtest_portfolio(req: BacktestRequest) -> Dict[str, Any]:
        """Walk-forward portfolio backtest with periodic rebalancing."""
        if not req.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        result = _optimizer_singleton.portfolio_backtest(
            tickers=req.tickers,
            method=req.method,
            start=req.start,
            end=req.end,
            rebalance_freq=req.rebalance_freq,
            estimation_window=req.estimation_window,
        )
        return _backtest_to_dict(result)

else:
    optimizer_router = None  # type: ignore


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def optimize_portfolio(
    tickers: List[str],
    method: str = "hrp",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    rf: float = 0.05,
    max_weight: float = 0.40,
) -> OptimizationResult:
    """Convenience: run a single optimization and return weights."""
    optimizer = PortfolioOptimizer(rf=rf)
    constraints = PortfolioConstraints(max_weight=max_weight)
    return optimizer.optimize(
        tickers=tickers,
        method=method,  # type: ignore
        start=start,
        end=end,
        constraints=constraints,
    )


def compare_optimizers(
    tickers: List[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> Dict[str, OptimizationResult]:
    """Convenience: compare all methods for a ticker list."""
    optimizer = PortfolioOptimizer()
    return optimizer.compare_methods(tickers=tickers, start=start, end=end)


def black_litterman(
    tickers: List[str],
    market_caps: Dict[str, float],
    views: List[Dict[str, Any]],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    tau: float = 0.05,
) -> BLResult:
    """
    Convenience: run Black-Litterman optimization.

    Example view:
    {
        "type": "absolute",
        "assets": ["AAPL"],
        "return": 0.15,   # 15% expected annual return
        "confidence": 0.7
    }
    """
    rc = ReturnCovariance()
    returns = rc.get_returns(tickers, start=start, end=end)
    cov = rc.ledoit_wolf_shrinkage(returns)
    bl = BlackLittermanOptimizer()
    return bl.optimize(
        assets=tickers,
        market_caps=market_caps,
        sigma=cov,
        views=views,
        tau=tau,
    )
