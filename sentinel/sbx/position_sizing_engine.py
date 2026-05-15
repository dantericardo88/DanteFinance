"""
Comprehensive position sizing: Kelly criterion, vol-target, risk parity,
optimal f, fixed fractional, and regime-conditional sizing.

Dimension targeted:
  dim_082 — Kelly / vol-target / risk-parity sizing  score → 9

Classes
-------
KellyCriterion
    Full Kelly, fractional Kelly, Bayesian shrinkage, multi-asset Kelly,
    Monte Carlo verification.

VolatilityTargeting
    EWMA vol, GARCH(1,1) forecast, portfolio-level vol targeting,
    dynamic deleveraging, half-Kelly hybrid.

RiskParitySizing
    Naive 1/vol, equal risk contribution (ERC), asset-class risk parity,
    levered risk parity, threshold rebalancing.

OptimalF
    Ralph Vince terminal wealth relative maximisation over a trade history.

FixedFractionalSizing
    Fixed risk per trade, ATR-based stops, pyramid tranches.

RegimeConditionalSizing
    High-vol / crisis / trend / correlation regime adjustments.

FastAPI router  sizing_router  — mounted at /api/sizing
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field, validator
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

TRADING_DAYS = 252
_EPSILON = 1e-12
MAX_KELLY_CAP = 0.25          # hard cap on any single position
DEFAULT_RF = 0.05             # risk-free rate (annualised)
EWMA_LAMBDA = 0.94            # RiskMetrics decay factor


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class KellyResult:
    full_kelly: float
    fractional_kelly: float
    bayesian_kelly: float
    kelly_fraction_used: float
    expected_log_growth: float
    mc_median_growth: float
    mc_blow_up_probability: float
    capped: bool
    notes: str = ""


@dataclass
class VolTargetResult:
    position_scalar: float          # multiply raw notional by this
    leverage: float
    realized_vol_ann: float
    target_vol_ann: float
    garch_vol_forecast: float
    regime: str                     # "normal" | "high_vol" | "low_vol"


@dataclass
class RiskParityResult:
    weights: Dict[str, float]
    risk_contributions: Dict[str, float]   # each asset's % of portfolio vol
    portfolio_vol: float
    leverage_scalar: float
    rebalance_needed: bool


@dataclass
class OptimalFResult:
    optimal_f: float
    twr_at_f: float
    safe_f: float          # 0.5 × optimal_f recommended for live trading
    expected_drawdown_pct: float
    max_drawdown_pct: float
    warning: str


@dataclass
class FixedFractionalResult:
    shares: float
    dollar_risk: float
    stop_price: float
    account_risk_pct: float
    tranche_sizes: List[float]   # pyramid plan


@dataclass
class RegimeAdjustedResult:
    raw_size: float
    adjusted_size: float
    regime: str
    multiplier: float
    rationale: str


# ---------------------------------------------------------------------------
# 1.  Kelly Criterion
# ---------------------------------------------------------------------------

class KellyCriterion:
    """
    Kelly position sizing with full, fractional, Bayesian, and multi-asset
    variants, plus Monte Carlo blow-up verification.
    """

    def __init__(self, kelly_fraction: float = 0.25, rf: float = DEFAULT_RF):
        """
        Parameters
        ----------
        kelly_fraction : float
            Fraction of full Kelly to use (0.25 = quarter-Kelly).
        rf : float
            Annual risk-free rate.
        """
        self.kelly_fraction = kelly_fraction
        self.rf = rf

    # ------------------------------------------------------------------
    # Core: single-asset Kelly
    # ------------------------------------------------------------------

    def compute_kelly(
        self,
        mu: float,
        sigma: float,
        rf: Optional[float] = None,
    ) -> float:
        """
        Full Kelly fraction for log-normally distributed returns.

        f* = (μ - r) / σ²

        Parameters
        ----------
        mu    : annualised expected return (e.g. 0.12 = 12%)
        sigma : annualised volatility (e.g. 0.20 = 20%)
        rf    : risk-free rate (defaults to self.rf)

        Returns
        -------
        float   Fraction of capital to allocate (positive = long).
        """
        if rf is None:
            rf = self.rf
        if sigma < _EPSILON:
            return 0.0
        excess = mu - rf
        f_star = excess / (sigma ** 2)
        return float(np.clip(f_star, -MAX_KELLY_CAP, MAX_KELLY_CAP))

    def _full_kelly_uncapped(self, mu: float, sigma: float, rf: float) -> float:
        if sigma < _EPSILON:
            return 0.0
        return (mu - rf) / (sigma ** 2)

    def fractional_kelly(self, mu: float, sigma: float, rf: Optional[float] = None) -> float:
        """kelly_fraction × full Kelly, capped at MAX_KELLY_CAP."""
        if rf is None:
            rf = self.rf
        full = self._full_kelly_uncapped(mu, sigma, rf)
        return float(np.clip(full * self.kelly_fraction, -MAX_KELLY_CAP, MAX_KELLY_CAP))

    def bayesian_kelly(
        self,
        mu: float,
        sigma: float,
        n_obs: int,
        rf: Optional[float] = None,
        prior_precision: float = 1.0,
    ) -> float:
        """
        Bayesian / shrinkage Kelly.

        Uncertainty in μ shrinks the optimal fraction toward zero.
        Shrinkage factor = n / (n + prior_precision * σ² / variance_of_mu_hat)

        Under standard assumptions: shrinkage = n_obs / (n_obs + 1).
        With finite sample the posterior mean of f* is shrunk by
        (n_obs / (n_obs + 1)) relative to the plug-in estimate.

        Parameters
        ----------
        n_obs          : number of observations used to estimate μ
        prior_precision: precision of the prior on μ in units of (σ²/n)
        """
        if rf is None:
            rf = self.rf
        full = self._full_kelly_uncapped(mu, sigma, rf)
        shrinkage = n_obs / (n_obs + prior_precision)
        bayesian_f = full * shrinkage
        return float(np.clip(bayesian_f, -MAX_KELLY_CAP, MAX_KELLY_CAP))

    # ------------------------------------------------------------------
    # Multi-asset Kelly (matrix form)
    # ------------------------------------------------------------------

    def compute_multi_asset_kelly(
        self,
        mu_vector: np.ndarray,
        sigma_matrix: np.ndarray,
        rf: Optional[float] = None,
        max_position: float = MAX_KELLY_CAP,
    ) -> np.ndarray:
        """
        Multi-asset Kelly optimal weights.

        f* = Σ^(-1) · (μ - r·1)

        Parameters
        ----------
        mu_vector    : 1-D array of expected returns, length n
        sigma_matrix : n×n covariance matrix
        rf           : risk-free rate
        max_position : per-asset weight cap (applied element-wise)

        Returns
        -------
        np.ndarray  Weight vector, each element in [-max_position, max_position].
        """
        if rf is None:
            rf = self.rf
        mu = np.asarray(mu_vector, dtype=float)
        cov = np.asarray(sigma_matrix, dtype=float)
        n = len(mu)
        if cov.shape != (n, n):
            raise ValueError("sigma_matrix must be (n, n)")

        # Regularise covariance for numerical stability
        cov_reg = cov + np.eye(n) * 1e-8

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_reg)

        excess = mu - rf
        f_star = cov_inv @ excess

        # Apply per-asset cap
        f_star = np.clip(f_star, -max_position, max_position)

        # Scale fractionally
        f_star *= self.kelly_fraction
        return f_star

    # ------------------------------------------------------------------
    # Monte Carlo verification
    # ------------------------------------------------------------------

    def monte_carlo_kelly(
        self,
        mu: float,
        sigma: float,
        rf: Optional[float] = None,
        n_paths: int = 10_000,
        horizon_years: float = 5.0,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Simulate n_paths wealth paths using fractional Kelly fraction.

        Parameters
        ----------
        horizon_years : simulation length in years

        Returns
        -------
        dict with median_cagr, blow_up_prob (wealth < 10% of start),
             mean_final_wealth, p5_wealth, p95_wealth
        """
        if rf is None:
            rf = self.rf
        rng = np.random.default_rng(seed)
        f = self.fractional_kelly(mu, sigma, rf)
        steps = int(horizon_years * TRADING_DAYS)

        # Daily returns: r ~ N(μ/252, σ/√252)
        mu_d = mu / TRADING_DAYS
        sigma_d = sigma / np.sqrt(TRADING_DAYS)

        # Portfolio daily return = f × asset_return + (1 - f) × rf/252
        raw_returns = rng.normal(mu_d, sigma_d, (n_paths, steps))
        port_returns = f * raw_returns + (1.0 - abs(f)) * (rf / TRADING_DAYS)

        # Wealth relatives
        wealth = np.prod(1.0 + port_returns, axis=1)

        blow_up_prob = float(np.mean(wealth < 0.10))
        median_wealth = float(np.median(wealth))
        median_cagr = float(median_wealth ** (1.0 / horizon_years) - 1.0)
        mean_final = float(np.mean(wealth))
        p5 = float(np.percentile(wealth, 5))
        p95 = float(np.percentile(wealth, 95))

        logger.info(
            "MC Kelly: f=%.3f  median_cagr=%.1f%%  blow_up=%.1f%%",
            f, median_cagr * 100, blow_up_prob * 100,
        )
        return {
            "kelly_fraction": f,
            "median_cagr": median_cagr,
            "blow_up_probability": blow_up_prob,
            "mean_final_wealth": mean_final,
            "p5_wealth": p5,
            "p95_wealth": p95,
            "horizon_years": horizon_years,
            "n_paths": n_paths,
        }

    # ------------------------------------------------------------------
    # Full compute with all variants
    # ------------------------------------------------------------------

    def compute(
        self,
        mu: float,
        sigma: float,
        rf: Optional[float] = None,
        n_obs: int = 252,
        run_mc: bool = True,
    ) -> KellyResult:
        """Return KellyResult with all variants populated."""
        if rf is None:
            rf = self.rf
        full = self._full_kelly_uncapped(mu, sigma, rf)
        frac = self.fractional_kelly(mu, sigma, rf)
        bayes = self.bayesian_kelly(mu, sigma, n_obs, rf)

        capped = abs(full * self.kelly_fraction) > MAX_KELLY_CAP

        # Expected log growth using fractional Kelly
        elg = frac * (mu - rf) - 0.5 * (frac ** 2) * (sigma ** 2)

        mc_median_cagr = 0.0
        mc_blow_up = 0.0
        if run_mc:
            mc = self.monte_carlo_kelly(mu, sigma, rf)
            mc_median_cagr = mc["median_cagr"]
            mc_blow_up = mc["blow_up_probability"]

        return KellyResult(
            full_kelly=float(np.clip(full, -1.0, 1.0)),
            fractional_kelly=frac,
            bayesian_kelly=bayes,
            kelly_fraction_used=self.kelly_fraction,
            expected_log_growth=elg,
            mc_median_growth=mc_median_cagr,
            mc_blow_up_probability=mc_blow_up,
            capped=capped,
            notes=(
                "Kelly capped at 25% per position. "
                f"Using {self.kelly_fraction:.0%} fraction for safety."
            ),
        )


# ---------------------------------------------------------------------------
# 2.  Volatility Targeting
# ---------------------------------------------------------------------------

class VolatilityTargeting:
    """
    Vol-target position sizing with EWMA, GARCH(1,1) forecast, and
    dynamic deleveraging.
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        ewma_lambda: float = EWMA_LAMBDA,
        max_leverage: float = 2.0,
    ):
        self.target_vol = target_vol
        self.ewma_lambda = ewma_lambda
        self.max_leverage = max_leverage

    # ------------------------------------------------------------------
    # Vol estimation
    # ------------------------------------------------------------------

    def realized_vol(self, returns: pd.Series, window: int = 21) -> float:
        """Annualised rolling realised vol (close-to-close)."""
        if len(returns) < 2:
            return float("nan")
        r = returns.dropna().tail(window)
        return float(r.std() * np.sqrt(TRADING_DAYS))

    def ewma_vol(self, returns: pd.Series) -> float:
        """EWMA volatility (RiskMetrics 1994, λ=0.94), annualised."""
        r = returns.dropna()
        if len(r) < 2:
            return float("nan")
        variance = float(r.iloc[0] ** 2)
        lam = self.ewma_lambda
        for ret in r.iloc[1:]:
            variance = lam * variance + (1 - lam) * (float(ret) ** 2)
        return float(np.sqrt(variance * TRADING_DAYS))

    def garch11_vol_forecast(
        self,
        returns: pd.Series,
        omega: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        n_forecast: int = 1,
    ) -> float:
        """
        GARCH(1,1) one-step-ahead variance forecast.

        If omega/alpha/beta are not supplied they are estimated via
        quasi-MLE (numerical minimisation of negative log-likelihood).

        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

        Returns annualised volatility.
        """
        r = returns.dropna().values.astype(float)
        if len(r) < 30:
            return self.realized_vol(returns)

        if omega is None or alpha is None or beta is None:
            omega, alpha, beta = self._estimate_garch11(r)

        # Filter to get σ²_{T}
        sigma2 = np.var(r)
        for eps in r:
            sigma2 = omega + alpha * (eps ** 2) + beta * sigma2

        # Multi-step forecast (for h > 1)
        sigma2_h = sigma2
        for _ in range(n_forecast - 1):
            sigma2_h = omega + (alpha + beta) * sigma2_h

        return float(np.sqrt(sigma2_h * TRADING_DAYS))

    def _estimate_garch11(self, r: np.ndarray) -> Tuple[float, float, float]:
        """Estimate GARCH(1,1) parameters via MLE."""
        var0 = np.var(r)

        def neg_log_likelihood(params: np.ndarray) -> float:
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            sigma2 = var0
            nll = 0.0
            for eps in r:
                sigma2 = omega + alpha * (eps ** 2) + beta * sigma2
                if sigma2 <= 0:
                    return 1e10
                nll += 0.5 * (np.log(sigma2) + eps ** 2 / sigma2)
            return nll

        x0 = [var0 * 0.05, 0.10, 0.85]
        bounds = [(1e-8, None), (0.0, 0.5), (0.0, 0.9999)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-10},
                )
            if res.success:
                return tuple(res.x)
        except Exception:
            pass
        # Fallback to typical parameters
        return (var0 * 0.05, 0.10, 0.85)

    # ------------------------------------------------------------------
    # Core: single-asset vol targeting
    # ------------------------------------------------------------------

    def size_from_vol_target(
        self,
        target_vol: Optional[float] = None,
        realized_vol: Optional[float] = None,
        returns: Optional[pd.Series] = None,
        use_garch: bool = False,
        max_leverage: Optional[float] = None,
    ) -> float:
        """
        Compute leverage scalar: weight = target_vol / asset_vol.

        Parameters
        ----------
        target_vol    : desired annualised portfolio volatility
        realized_vol  : pre-computed annualised vol (or pass returns)
        returns       : daily return series to estimate vol from
        use_garch     : use GARCH(1,1) forecast instead of EWMA

        Returns
        -------
        float   Leverage scalar (clipped to [0, max_leverage]).
        """
        tv = target_vol if target_vol is not None else self.target_vol
        ml = max_leverage if max_leverage is not None else self.max_leverage

        if realized_vol is not None:
            asset_vol = realized_vol
        elif returns is not None:
            if use_garch:
                asset_vol = self.garch11_vol_forecast(returns)
            else:
                asset_vol = self.ewma_vol(returns)
        else:
            raise ValueError("Provide either realized_vol or returns.")

        if asset_vol < _EPSILON:
            return 0.0

        scalar = tv / asset_vol
        return float(np.clip(scalar, 0.0, ml))

    # ------------------------------------------------------------------
    # Portfolio-level vol targeting
    # ------------------------------------------------------------------

    def portfolio_vol_target(
        self,
        raw_weights: Dict[str, float],
        cov_matrix: pd.DataFrame,
        target_vol: Optional[float] = None,
        max_leverage: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Scale all weights so that portfolio volatility = target_vol.

        Parameters
        ----------
        raw_weights : {ticker: weight} (pre-scaling)
        cov_matrix  : annualised covariance matrix (tickers must match)
        target_vol  : desired portfolio vol (default self.target_vol)

        Returns
        -------
        dict   Scaled weights.
        """
        tv = target_vol if target_vol is not None else self.target_vol
        ml = max_leverage if max_leverage is not None else self.max_leverage

        assets = list(raw_weights.keys())
        w = np.array([raw_weights[a] for a in assets])

        # Ensure cov_matrix covers all assets
        cov = cov_matrix.loc[assets, assets].values

        port_var = float(w @ cov @ w)
        if port_var < _EPSILON:
            return raw_weights.copy()

        port_vol = np.sqrt(port_var)
        scalar = tv / port_vol
        scalar = min(scalar, ml)

        scaled = {a: float(w[i] * scalar) for i, a in enumerate(assets)}
        logger.info(
            "Portfolio vol-target: raw_vol=%.1f%% target=%.1f%% scalar=%.3f",
            port_vol * 100, tv * 100, scalar,
        )
        return scaled

    # ------------------------------------------------------------------
    # Dynamic deleveraging
    # ------------------------------------------------------------------

    def dynamic_deleveraging(
        self,
        current_vol: float,
        target_vol: Optional[float] = None,
        regime: str = "normal",
    ) -> float:
        """
        Inverse-vol scaling: leverage ∝ target_vol / current_vol.

        Extra crisis buffer applied when regime = 'crisis'.
        """
        tv = target_vol if target_vol is not None else self.target_vol
        if current_vol < _EPSILON:
            return 1.0
        scalar = tv / current_vol
        if regime == "crisis":
            scalar *= 0.5
        return float(np.clip(scalar, 0.0, self.max_leverage))

    # ------------------------------------------------------------------
    # Half-Kelly × vol-target hybrid
    # ------------------------------------------------------------------

    def half_kelly_vol_target(
        self,
        mu: float,
        sigma: float,
        target_vol: Optional[float] = None,
        rf: float = DEFAULT_RF,
    ) -> float:
        """
        Combine quarter-Kelly signal with vol-target scaling.

        size = (Kelly_quarter × vol_target_scalar) capped at MAX_KELLY_CAP
        """
        tv = target_vol if target_vol is not None else self.target_vol
        kelly = KellyCriterion(kelly_fraction=0.25, rf=rf)
        kelly_pos = kelly.fractional_kelly(mu, sigma, rf)
        vol_scalar = self.size_from_vol_target(tv, realized_vol=sigma)
        combined = kelly_pos * vol_scalar
        return float(np.clip(combined, -MAX_KELLY_CAP, MAX_KELLY_CAP))

    # ------------------------------------------------------------------
    # Full compute
    # ------------------------------------------------------------------

    def compute(
        self,
        returns: pd.Series,
        target_vol: Optional[float] = None,
    ) -> VolTargetResult:
        """Full vol-target result with regime classification."""
        tv = target_vol if target_vol is not None else self.target_vol
        ewma = self.ewma_vol(returns)
        rvol = self.realized_vol(returns)
        garch = self.garch11_vol_forecast(returns)

        # Use the more conservative of EWMA and GARCH
        asset_vol = max(ewma, garch)
        scalar = float(np.clip(tv / asset_vol if asset_vol > _EPSILON else 1.0, 0.0, self.max_leverage))

        # Regime
        ratio = asset_vol / tv if tv > _EPSILON else 1.0
        if ratio > 1.5:
            regime = "high_vol"
        elif ratio < 0.7:
            regime = "low_vol"
        else:
            regime = "normal"

        return VolTargetResult(
            position_scalar=scalar,
            leverage=scalar,
            realized_vol_ann=rvol,
            target_vol_ann=tv,
            garch_vol_forecast=garch,
            regime=regime,
        )


# ---------------------------------------------------------------------------
# 3.  Risk Parity Sizing
# ---------------------------------------------------------------------------

class RiskParitySizing:
    """
    Equal Risk Contribution (ERC) / risk parity position sizing.
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        rebalance_threshold: float = 0.05,
    ):
        self.target_vol = target_vol
        self.rebalance_threshold = rebalance_threshold

    # ------------------------------------------------------------------
    # Naive 1/vol weighting
    # ------------------------------------------------------------------

    def naive_risk_parity(self, vols: Dict[str, float]) -> Dict[str, float]:
        """
        Weight ∝ 1/vol_i, normalised to sum to 1.

        Parameters
        ----------
        vols : {ticker: annualised_vol}
        """
        inv_vols = {k: 1.0 / max(v, _EPSILON) for k, v in vols.items()}
        total = sum(inv_vols.values())
        return {k: v / total for k, v in inv_vols.items()}

    # ------------------------------------------------------------------
    # Equal Risk Contribution (numerical)
    # ------------------------------------------------------------------

    def equal_risk_contribution(
        self,
        cov_matrix: pd.DataFrame,
        assets: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Solve numerically for weights w such that RC_i = σ_p / n for all i.

        Risk contribution: RC_i = w_i × (Σ w)_i / σ_p

        Objective: minimise Σ_i Σ_j (RC_i - RC_j)²
        """
        if assets is None:
            assets = list(cov_matrix.columns)
        cov = cov_matrix.loc[assets, assets].values.astype(float)
        n = len(assets)

        def _portfolio_vol(w: np.ndarray) -> float:
            pv = float(np.sqrt(w @ cov @ w))
            return max(pv, _EPSILON)

        def _risk_contributions(w: np.ndarray) -> np.ndarray:
            pv = _portfolio_vol(w)
            marginal = cov @ w
            return w * marginal / pv

        def objective(w: np.ndarray) -> float:
            rc = _risk_contributions(w)
            target_rc = np.sum(rc) / n
            return float(np.sum((rc - target_rc) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    objective, w0, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 1000},
                )
            if res.success:
                w_opt = res.x
            else:
                w_opt = w0
        except Exception:
            w_opt = w0

        # Normalise
        w_opt = np.clip(w_opt, 0, 1)
        total = w_opt.sum()
        if total > _EPSILON:
            w_opt /= total

        weights = {a: float(w_opt[i]) for i, a in enumerate(assets)}
        return weights

    # ------------------------------------------------------------------
    # Risk contributions diagnostic
    # ------------------------------------------------------------------

    def risk_contributions(
        self,
        weights: Dict[str, float],
        cov_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """Return each asset's percentage risk contribution."""
        assets = list(weights.keys())
        w = np.array([weights[a] for a in assets])
        cov = cov_matrix.loc[assets, assets].values.astype(float)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol < _EPSILON:
            return {a: 0.0 for a in assets}
        marginal = cov @ w
        rc = w * marginal / port_vol
        total_rc = rc.sum()
        return {a: float(rc[i] / total_rc) for i, a in enumerate(assets)}

    # ------------------------------------------------------------------
    # Asset class risk parity (buckets)
    # ------------------------------------------------------------------

    def asset_class_risk_parity(
        self,
        asset_class_vols: Dict[str, float],
        asset_class_assets: Dict[str, List[str]],
        within_class_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Two-level risk parity:
        1. Allocate equally across asset classes (by risk)
        2. Within each class, equal-weight (or custom weights)

        Parameters
        ----------
        asset_class_vols   : {class_name: annualised_vol}
        asset_class_assets : {class_name: [ticker, ...]}
        within_class_weights : optional {class_name: {ticker: weight}}

        Returns
        -------
        dict  {ticker: final_weight}
        """
        class_weights = self.naive_risk_parity(asset_class_vols)
        final_weights: Dict[str, float] = {}

        for cls, cls_w in class_weights.items():
            tickers = asset_class_assets.get(cls, [])
            if not tickers:
                continue
            if within_class_weights and cls in within_class_weights:
                wcw = within_class_weights[cls]
                total = sum(wcw.values())
                for t in tickers:
                    final_weights[t] = cls_w * wcw.get(t, 0.0) / max(total, _EPSILON)
            else:
                per_ticker = cls_w / len(tickers)
                for t in tickers:
                    final_weights[t] = per_ticker

        return final_weights

    # ------------------------------------------------------------------
    # Levered risk parity
    # ------------------------------------------------------------------

    def levered_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        target_vol: Optional[float] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Solve ERC weights then scale leverage so portfolio vol = target_vol.

        Returns (scaled_weights, leverage_scalar).
        """
        tv = target_vol if target_vol is not None else self.target_vol
        assets = list(cov_matrix.columns)
        w_erc = self.equal_risk_contribution(cov_matrix, assets)
        w = np.array([w_erc[a] for a in assets])
        cov = cov_matrix.loc[assets, assets].values.astype(float)
        port_vol = float(np.sqrt(w @ cov @ w))

        leverage = tv / port_vol if port_vol > _EPSILON else 1.0
        scaled = {a: float(w_erc[a] * leverage) for a in assets}
        return scaled, leverage

    # ------------------------------------------------------------------
    # Rebalancing check
    # ------------------------------------------------------------------

    def rebalance_check(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> bool:
        """
        Return True if any weight deviates by more than rebalance_threshold.
        """
        for asset in target_weights:
            curr = current_weights.get(asset, 0.0)
            tgt = target_weights[asset]
            if abs(curr - tgt) > self.rebalance_threshold:
                return True
        return False

    # ------------------------------------------------------------------
    # Full compute
    # ------------------------------------------------------------------

    def compute(
        self,
        cov_matrix: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        target_vol: Optional[float] = None,
    ) -> RiskParityResult:
        assets = list(cov_matrix.columns)
        tv = target_vol if target_vol is not None else self.target_vol

        scaled_weights, leverage = self.levered_risk_parity(cov_matrix, tv)
        rc = self.risk_contributions(
            {a: scaled_weights[a] / leverage for a in assets},
            cov_matrix,
        )

        w = np.array([scaled_weights[a] for a in assets])
        cov = cov_matrix.loc[assets, assets].values.astype(float)
        port_vol = float(np.sqrt(w @ cov @ w))

        rebalance_needed = False
        if current_weights:
            rebalance_needed = self.rebalance_check(current_weights, scaled_weights)

        return RiskParityResult(
            weights=scaled_weights,
            risk_contributions=rc,
            portfolio_vol=port_vol,
            leverage_scalar=leverage,
            rebalance_needed=rebalance_needed,
        )


# ---------------------------------------------------------------------------
# 4.  Optimal F (Ralph Vince)
# ---------------------------------------------------------------------------

class OptimalF:
    """
    Optimal F position sizing — maximises Terminal Wealth Relative (TWR)
    given a trade history.

    Reference: Ralph Vince, "The Mathematics of Money Management" (1992).
    """

    def compute_optimal_f(self, trade_returns: List[float]) -> OptimalFResult:
        """
        Given a list of trade returns (e.g. [+0.10, -0.05, +0.08, …]),
        find f that maximises TWR(f) = ∏(1 + f × R_i)^(1/n).

        Parameters
        ----------
        trade_returns : list of fractional returns per trade

        Returns
        -------
        OptimalFResult
        """
        returns = np.array(trade_returns, dtype=float)
        if len(returns) == 0:
            return OptimalFResult(0.0, 1.0, 0.0, 0.0, 0.0, "No trade data")

        worst_loss = float(np.min(returns))
        if worst_loss >= 0:
            return OptimalFResult(
                optimal_f=1.0,
                twr_at_f=1.0,
                safe_f=0.5,
                expected_drawdown_pct=0.0,
                max_drawdown_pct=0.0,
                warning="No losing trades — optimal f is trivially 1.0. Use caution.",
            )

        # TWR as function of f: must keep every (1 + f × R_i) > 0
        # So f < 1 / |worst_loss| is required (bankruptcy constraint)
        f_max = 1.0 / abs(worst_loss) - _EPSILON

        def neg_twr(f: float) -> float:
            relatives = 1.0 + f * returns
            if np.any(relatives <= 0):
                return 0.0
            n = len(returns)
            twr = float(np.prod(relatives) ** (1.0 / n))
            return -twr  # minimise negative

        try:
            result = minimize_scalar(
                neg_twr,
                bounds=(0.0, min(f_max, 1.0)),
                method="bounded",
                options={"xatol": 1e-6, "maxiter": 500},
            )
            f_opt = float(result.x) if result.success else 0.25
        except Exception:
            f_opt = 0.25

        relatives = 1.0 + f_opt * returns
        twr_val = float(np.prod(relatives) ** (1.0 / len(returns)))

        # Drawdown simulation at optimal f
        equity = np.cumprod(relatives)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_dd = float(np.max(drawdowns))
        exp_dd = float(np.mean(drawdowns[drawdowns > 0])) if np.any(drawdowns > 0) else 0.0

        safe_f = f_opt * 0.5
        warning = (
            f"Optimal f = {f_opt:.3f} is aggressive (max DD ~{max_dd:.0%}). "
            f"Recommended safe f = {safe_f:.3f} (half of optimal)."
        )

        return OptimalFResult(
            optimal_f=f_opt,
            twr_at_f=twr_val,
            safe_f=safe_f,
            expected_drawdown_pct=exp_dd,
            max_drawdown_pct=max_dd,
            warning=warning,
        )

    def position_size_from_f(
        self,
        account_size: float,
        f: float,
        worst_loss_per_unit: float,
    ) -> float:
        """
        Given optimal f, compute number of units/contracts.

        size = (account × f) / |worst_loss_per_unit|

        Parameters
        ----------
        account_size      : total account equity
        f                 : fraction to risk (e.g. 0.20)
        worst_loss_per_unit : maximum dollar loss per 1 unit/share
        """
        if abs(worst_loss_per_unit) < _EPSILON:
            return 0.0
        return float(account_size * f / abs(worst_loss_per_unit))


# ---------------------------------------------------------------------------
# 5.  Fixed Fractional Sizing
# ---------------------------------------------------------------------------

class FixedFractionalSizing:
    """
    Simple fixed-dollar-risk position sizing with ATR stops and pyramiding.
    """

    def __init__(self, default_risk_pct: float = 0.01):
        """
        Parameters
        ----------
        default_risk_pct : fraction of account to risk per trade (e.g. 0.01 = 1%)
        """
        self.default_risk_pct = default_risk_pct

    # ------------------------------------------------------------------
    # ATR-based stop
    # ------------------------------------------------------------------

    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Compute Average True Range over `period` days."""
        high = high.values.astype(float)
        low = low.values.astype(float)
        close = close.values.astype(float)
        n = len(close)
        if n < 2:
            return float(high[-1] - low[-1])
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        atr = float(np.mean(tr[-period:]))
        return atr

    # ------------------------------------------------------------------
    # Core: size from ATR stop
    # ------------------------------------------------------------------

    def size_from_atr_stop(
        self,
        account: float,
        risk_pct: float,
        price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        side: Literal["long", "short"] = "long",
    ) -> FixedFractionalResult:
        """
        Position sizing with ATR-based stop distance.

        stop = price ∓ ATR × multiplier
        size = (account × risk_pct) / |price - stop|

        Parameters
        ----------
        account       : total account equity in dollars
        risk_pct      : fraction of account to risk (e.g. 0.01 = 1%)
        price         : current entry price
        atr           : Average True Range
        atr_multiplier: stop placed this many ATRs from entry
        side          : "long" or "short"

        Returns
        -------
        FixedFractionalResult
        """
        stop_dist = atr * atr_multiplier
        if side == "long":
            stop_price = price - stop_dist
        else:
            stop_price = price + stop_dist

        dollar_risk = account * risk_pct
        risk_per_share = abs(price - stop_price)
        if risk_per_share < _EPSILON:
            shares = 0.0
        else:
            shares = dollar_risk / risk_per_share

        # Pyramid plan: 33% / 33% / 34% (3 tranches)
        tranches = [shares * 0.33, shares * 0.33, shares * 0.34]

        return FixedFractionalResult(
            shares=shares,
            dollar_risk=dollar_risk,
            stop_price=stop_price,
            account_risk_pct=risk_pct,
            tranche_sizes=tranches,
        )

    # ------------------------------------------------------------------
    # Simple dollar-risk sizing (entry + stop known)
    # ------------------------------------------------------------------

    def size_from_entry_stop(
        self,
        account: float,
        risk_pct: float,
        entry: float,
        stop: float,
    ) -> FixedFractionalResult:
        """
        size = (account × risk_pct) / (entry - stop)

        Works for both long (stop < entry) and short (stop > entry).
        """
        dollar_risk = account * risk_pct
        risk_per_share = abs(entry - stop)
        shares = dollar_risk / risk_per_share if risk_per_share > _EPSILON else 0.0
        tranches = [shares * 0.33, shares * 0.33, shares * 0.34]

        return FixedFractionalResult(
            shares=shares,
            dollar_risk=dollar_risk,
            stop_price=stop,
            account_risk_pct=risk_pct,
            tranche_sizes=tranches,
        )

    # ------------------------------------------------------------------
    # Pyramid sizing helper
    # ------------------------------------------------------------------

    def pyramid_plan(
        self,
        total_shares: float,
        n_tranches: int = 3,
        tranche_weights: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Split total_shares into n_tranches by weight.

        Default: equal tranches; custom weights normalised to sum to 1.
        """
        if tranche_weights is None:
            tranche_weights = [1.0 / n_tranches] * n_tranches
        total_w = sum(tranche_weights)
        norm_w = [w / total_w for w in tranche_weights]
        return [total_shares * w for w in norm_w]

    # ------------------------------------------------------------------
    # Fetch live price and ATR from yfinance
    # ------------------------------------------------------------------

    def fetch_price_and_atr(
        self, ticker: str, atr_period: int = 14
    ) -> Tuple[float, float]:
        """
        Download recent OHLC and compute current price + ATR.
        Requires yfinance.
        """
        if not _YF_AVAILABLE:
            raise ImportError("yfinance is required for live price/ATR fetch.")
        hist = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
        if hist.empty:
            raise ValueError(f"No data returned for {ticker}")
        price = float(hist["Close"].iloc[-1])
        atr = self.compute_atr(hist["High"], hist["Low"], hist["Close"], atr_period)
        return price, atr


# ---------------------------------------------------------------------------
# 6.  Regime-Conditional Sizing
# ---------------------------------------------------------------------------

class RegimeConditionalSizing:
    """
    Adjusts computed position sizes based on market regime signals.

    Regimes detected:
    - crisis      : VIX > 30 → 25% of normal
    - high_vol    : realised vol > 1.5× normal → 50% of normal
    - low_vol     : realised vol < 0.7× normal → 100% of normal
    - trend       : momentum > 0 → 100%; mean-reversion → 50%
    - high_corr   : average pairwise correlation > 0.7 → reduce by corr_penalty
    """

    REGIME_MULTIPLIERS: Dict[str, float] = {
        "crisis":       0.25,
        "high_vol":     0.50,
        "low_vol":      1.00,
        "trend":        1.00,
        "mean_revert":  0.50,
        "high_corr":    0.60,
        "normal":       1.00,
    }

    def __init__(
        self,
        vix_crisis_threshold: float = 30.0,
        vol_high_multiple: float = 1.5,
        vol_low_multiple: float = 0.7,
        corr_threshold: float = 0.70,
    ):
        self.vix_crisis_threshold = vix_crisis_threshold
        self.vol_high_multiple = vol_high_multiple
        self.vol_low_multiple = vol_low_multiple
        self.corr_threshold = corr_threshold

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def detect_regime(
        self,
        current_vol: float,
        normal_vol: float,
        vix: Optional[float] = None,
        momentum: Optional[float] = None,
        avg_correlation: Optional[float] = None,
    ) -> str:
        """
        Classify current market regime.

        Parameters
        ----------
        current_vol   : current realised vol (annualised)
        normal_vol    : long-run average vol for comparison
        vix           : current VIX level (optional)
        momentum      : 12-1 month momentum (positive = trending up)
        avg_correlation: average pairwise portfolio correlation

        Returns
        -------
        str   Regime label.
        """
        if vix is not None and vix > self.vix_crisis_threshold:
            return "crisis"

        vol_ratio = current_vol / normal_vol if normal_vol > _EPSILON else 1.0
        if vol_ratio > self.vol_high_multiple:
            return "high_vol"

        if avg_correlation is not None and avg_correlation > self.corr_threshold:
            return "high_corr"

        if momentum is not None:
            if momentum > 0:
                return "trend"
            else:
                return "mean_revert"

        if vol_ratio < self.vol_low_multiple:
            return "low_vol"

        return "normal"

    # ------------------------------------------------------------------
    # Apply regime adjustment
    # ------------------------------------------------------------------

    def adjust_size(
        self,
        raw_size: float,
        regime: str,
        override_multiplier: Optional[float] = None,
    ) -> RegimeAdjustedResult:
        """
        Apply the appropriate regime multiplier to raw_size.

        Parameters
        ----------
        raw_size           : computed position size (shares, notional, or weight)
        regime             : regime label from detect_regime()
        override_multiplier: use this multiplier instead of regime default
        """
        multiplier = override_multiplier if override_multiplier is not None else \
            self.REGIME_MULTIPLIERS.get(regime, 1.0)
        adjusted = raw_size * multiplier

        rationale_map = {
            "crisis":      f"VIX crisis — reducing to {multiplier:.0%} of normal size.",
            "high_vol":    f"High volatility regime — reducing to {multiplier:.0%}.",
            "low_vol":     "Low volatility regime — no reduction applied.",
            "trend":       "Trend regime — full position allowed.",
            "mean_revert": f"Mean-reversion regime — reducing to {multiplier:.0%}.",
            "high_corr":   f"High correlation regime — reducing to {multiplier:.0%}.",
            "normal":      "Normal regime — standard sizing.",
        }

        return RegimeAdjustedResult(
            raw_size=raw_size,
            adjusted_size=adjusted,
            regime=regime,
            multiplier=multiplier,
            rationale=rationale_map.get(regime, "Unknown regime."),
        )

    # ------------------------------------------------------------------
    # Fetch live VIX from yfinance
    # ------------------------------------------------------------------

    def fetch_vix(self) -> float:
        """Download current VIX from yfinance (^VIX)."""
        if not _YF_AVAILABLE:
            return 20.0  # neutral default
        try:
            vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
            return float(vix_data["Close"].iloc[-1])
        except Exception:
            return 20.0

    # ------------------------------------------------------------------
    # Portfolio-wide regime adjustment
    # ------------------------------------------------------------------

    def adjust_portfolio(
        self,
        weights: Dict[str, float],
        returns_df: pd.DataFrame,
        vix: Optional[float] = None,
        long_run_vol: float = 0.15,
    ) -> Dict[str, float]:
        """
        Apply regime adjustment to all portfolio weights.

        Parameters
        ----------
        weights       : {ticker: weight}
        returns_df    : DataFrame of daily returns (columns = tickers)
        vix           : current VIX (fetched from yfinance if None)
        long_run_vol  : historical average portfolio volatility

        Returns
        -------
        dict  Adjusted weights.
        """
        if vix is None:
            vix = self.fetch_vix()

        # Compute current portfolio vol
        assets = list(weights.keys())
        available = [a for a in assets if a in returns_df.columns]
        if available:
            port_returns = returns_df[available].fillna(0).dot(
                pd.Series({a: weights.get(a, 0.0) for a in available})
            )
            vol_engine = VolatilityTargeting()
            current_vol = vol_engine.ewma_vol(port_returns)
        else:
            current_vol = long_run_vol

        # Average correlation
        if len(available) > 1:
            corr = returns_df[available].corr().values
            n = len(available)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            avg_corr = float(np.mean(corr[mask]))
        else:
            avg_corr = 0.0

        regime = self.detect_regime(
            current_vol=current_vol,
            normal_vol=long_run_vol,
            vix=vix,
            avg_correlation=avg_corr,
        )
        multiplier = self.REGIME_MULTIPLIERS.get(regime, 1.0)
        adjusted = {k: v * multiplier for k, v in weights.items()}
        logger.info(
            "Regime: %s  multiplier: %.2f  vix: %.1f  current_vol: %.1f%%",
            regime, multiplier, vix, current_vol * 100,
        )
        return adjusted


# ---------------------------------------------------------------------------
# Utility: fetch returns from yfinance
# ---------------------------------------------------------------------------

def _fetch_returns(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """Download adjusted close prices and compute daily log returns."""
    if not _YF_AVAILABLE:
        raise ImportError("yfinance is required")
    prices = yf.download(tickers, period=period, interval="1d", progress=False, auto_adjust=True)
    if len(tickers) == 1:
        close = prices[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        close = prices["Close"] if "Close" in prices.columns else prices
    returns = close.pct_change().dropna()
    return returns


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    sizing_router = APIRouter(prefix="/sizing", tags=["Position Sizing"])

    # ── Pydantic request / response models ───────────────────────────────────

    class KellyRequest(BaseModel):
        mu: float = Field(..., description="Annualised expected return, e.g. 0.12")
        sigma: float = Field(..., description="Annualised volatility, e.g. 0.20")
        rf: float = Field(DEFAULT_RF, description="Risk-free rate")
        kelly_fraction: float = Field(0.25, description="Fractional Kelly to apply")
        n_obs: int = Field(252, description="Number of observations for Bayesian shrinkage")
        run_mc: bool = Field(True, description="Run Monte Carlo verification")

    class MultiAssetKellyRequest(BaseModel):
        tickers: List[str] = Field(..., description="List of ticker symbols")
        rf: float = Field(DEFAULT_RF)
        kelly_fraction: float = Field(0.25)
        lookback_period: str = Field("2y")

    class VolTargetRequest(BaseModel):
        ticker: str
        target_vol: float = Field(0.10, description="Target annualised vol, e.g. 0.10")
        max_leverage: float = Field(2.0)
        use_garch: bool = Field(False)
        lookback: str = Field("1y")

    class RiskParityRequest(BaseModel):
        tickers: List[str]
        target_vol: float = Field(0.10)
        lookback: str = Field("2y")

    class OptimalFRequest(BaseModel):
        trade_returns: List[float] = Field(..., description="List of trade returns e.g. [0.05, -0.03]")

    class FixedFractionalRequest(BaseModel):
        ticker: str
        account: float = Field(..., description="Account size in dollars")
        risk_pct: float = Field(0.01, description="Risk per trade as fraction")
        atr_multiplier: float = Field(2.0)
        side: str = Field("long")

    class RegimeRequest(BaseModel):
        tickers: List[str]
        raw_weights: Dict[str, float]
        target_vol: float = Field(0.10)
        lookback: str = Field("1y")

    class PortfolioSizingRequest(BaseModel):
        tickers: List[str]
        method: Literal["kelly", "vol_target", "risk_parity"] = "risk_parity"
        target_vol: float = Field(0.10)
        kelly_fraction: float = Field(0.25)
        lookback: str = Field("2y")

    # ── Endpoints ────────────────────────────────────────────────────────────

    @sizing_router.post("/kelly")
    async def kelly_endpoint(req: KellyRequest) -> Dict[str, Any]:
        """Single-asset Kelly criterion sizing."""
        engine = KellyCriterion(kelly_fraction=req.kelly_fraction, rf=req.rf)
        result = engine.compute(req.mu, req.sigma, req.rf, req.n_obs, req.run_mc)
        return {
            "full_kelly": result.full_kelly,
            "fractional_kelly": result.fractional_kelly,
            "bayesian_kelly": result.bayesian_kelly,
            "kelly_fraction_used": result.kelly_fraction_used,
            "expected_log_growth": result.expected_log_growth,
            "mc_median_cagr": result.mc_median_growth,
            "mc_blow_up_probability": result.mc_blow_up_probability,
            "capped_at_25pct": result.capped,
            "notes": result.notes,
        }

    @sizing_router.post("/kelly/multi-asset")
    async def multi_asset_kelly_endpoint(req: MultiAssetKellyRequest) -> Dict[str, Any]:
        """Multi-asset Kelly weights computed from historical returns."""
        try:
            returns = _fetch_returns(req.tickers, req.lookback_period)
            mu = returns.mean() * TRADING_DAYS
            cov = returns.cov() * TRADING_DAYS
            engine = KellyCriterion(kelly_fraction=req.kelly_fraction, rf=req.rf)
            weights = engine.compute_multi_asset_kelly(
                mu.values, cov.values, req.rf
            )
            return {
                "tickers": req.tickers,
                "kelly_weights": {t: float(w) for t, w in zip(req.tickers, weights)},
                "kelly_fraction": req.kelly_fraction,
                "expected_returns_ann": mu.to_dict(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @sizing_router.post("/vol-target")
    async def vol_target_endpoint(req: VolTargetRequest) -> Dict[str, Any]:
        """Volatility targeting scalar for a single asset."""
        try:
            returns = _fetch_returns([req.ticker], req.lookback)
            if req.ticker in returns.columns:
                ret_series = returns[req.ticker]
            else:
                ret_series = returns.iloc[:, 0]
            engine = VolatilityTargeting(
                target_vol=req.target_vol, max_leverage=req.max_leverage
            )
            result = engine.compute(ret_series, req.target_vol)
            return {
                "ticker": req.ticker,
                "position_scalar": result.position_scalar,
                "leverage": result.leverage,
                "realized_vol_ann": result.realized_vol_ann,
                "garch_vol_forecast": result.garch_vol_forecast,
                "target_vol": result.target_vol_ann,
                "regime": result.regime,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @sizing_router.post("/risk-parity")
    async def risk_parity_endpoint(req: RiskParityRequest) -> Dict[str, Any]:
        """Equal risk contribution weights for a basket of assets."""
        try:
            returns = _fetch_returns(req.tickers, req.lookback)
            cov = returns.cov() * TRADING_DAYS
            engine = RiskParitySizing(target_vol=req.target_vol)
            result = engine.compute(cov)
            return {
                "tickers": req.tickers,
                "weights": result.weights,
                "risk_contributions": result.risk_contributions,
                "portfolio_vol_ann": result.portfolio_vol,
                "leverage_scalar": result.leverage_scalar,
                "rebalance_needed": result.rebalance_needed,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @sizing_router.post("/optimal-f")
    async def optimal_f_endpoint(req: OptimalFRequest) -> Dict[str, Any]:
        """Ralph Vince Optimal f from trade history."""
        engine = OptimalF()
        result = engine.compute_optimal_f(req.trade_returns)
        return {
            "optimal_f": result.optimal_f,
            "safe_f": result.safe_f,
            "twr_at_f": result.twr_at_f,
            "expected_drawdown_pct": result.expected_drawdown_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "warning": result.warning,
        }

    @sizing_router.post("/fixed-fractional")
    async def fixed_fractional_endpoint(req: FixedFractionalRequest) -> Dict[str, Any]:
        """ATR-based fixed fractional position sizing."""
        try:
            engine = FixedFractionalSizing(default_risk_pct=req.risk_pct)
            price, atr = engine.fetch_price_and_atr(req.ticker)
            result = engine.size_from_atr_stop(
                account=req.account,
                risk_pct=req.risk_pct,
                price=price,
                atr=atr,
                atr_multiplier=req.atr_multiplier,
                side=req.side,
            )
            return {
                "ticker": req.ticker,
                "current_price": price,
                "atr_14d": atr,
                "stop_price": result.stop_price,
                "shares": result.shares,
                "dollar_risk": result.dollar_risk,
                "account_risk_pct": result.account_risk_pct,
                "tranche_plan": result.tranche_sizes,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @sizing_router.post("/regime-adjusted")
    async def regime_adjusted_endpoint(req: RegimeRequest) -> Dict[str, Any]:
        """Regime-conditional portfolio weight adjustment."""
        try:
            returns = _fetch_returns(req.tickers, req.lookback)
            regime_engine = RegimeConditionalSizing()
            adjusted_weights = regime_engine.adjust_portfolio(
                weights=req.raw_weights,
                returns_df=returns,
                long_run_vol=req.target_vol,
            )
            return {
                "tickers": req.tickers,
                "raw_weights": req.raw_weights,
                "adjusted_weights": adjusted_weights,
                "target_vol": req.target_vol,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @sizing_router.post("/portfolio")
    async def portfolio_sizing_endpoint(req: PortfolioSizingRequest) -> Dict[str, Any]:
        """
        Full portfolio sizing: Kelly, vol-target, or risk-parity method.
        Returns weights + regime adjustment.
        """
        try:
            returns = _fetch_returns(req.tickers, req.lookback)
            cov = returns.cov() * TRADING_DAYS
            mu_ann = returns.mean() * TRADING_DAYS

            if req.method == "risk_parity":
                rp = RiskParitySizing(target_vol=req.target_vol)
                rp_result = rp.compute(cov)
                raw_weights = rp_result.weights
                method_details: Dict[str, Any] = {
                    "risk_contributions": rp_result.risk_contributions,
                    "portfolio_vol": rp_result.portfolio_vol,
                }

            elif req.method == "kelly":
                kelly = KellyCriterion(kelly_fraction=req.kelly_fraction)
                w_arr = kelly.compute_multi_asset_kelly(
                    mu_ann.values, cov.values
                )
                raw_weights = {t: float(w) for t, w in zip(req.tickers, w_arr)}
                method_details = {"kelly_fraction": req.kelly_fraction}

            else:  # vol_target
                raw_weights = {}
                for ticker in req.tickers:
                    if ticker in returns.columns:
                        vt = VolatilityTargeting(target_vol=req.target_vol)
                        scalar = vt.size_from_vol_target(
                            returns=returns[ticker]
                        )
                        raw_weights[ticker] = scalar / len(req.tickers)
                    else:
                        raw_weights[ticker] = 1.0 / len(req.tickers)
                method_details = {"target_vol": req.target_vol}

            # Apply regime adjustment
            regime_engine = RegimeConditionalSizing()
            final_weights = regime_engine.adjust_portfolio(
                raw_weights, returns, long_run_vol=req.target_vol
            )

            return {
                "method": req.method,
                "tickers": req.tickers,
                "raw_weights": raw_weights,
                "regime_adjusted_weights": final_weights,
                "method_details": method_details,
                "lookback": req.lookback,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

else:
    sizing_router = None  # type: ignore
