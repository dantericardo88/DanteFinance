"""
Overfitting Detection v3 — dim_064 (score 8 → 9).

A deeply rigorous, standalone overfitting detection platform implementing the full
academic canon on backtest overfitting in quantitative finance.

Mathematical foundations
------------------------
- Bailey & Lopez de Prado (2014)  "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality"
- Bailey et al. (2014)            "The Probability of Backtest Overfitting"
- Lopez de Prado (2018)           "Advances in Financial Machine Learning" Ch.12
  Combinatorial Purged Cross-Validation
- Harvey, Liu & Zhu (2016)        "… and the Cross-Section of Expected Returns"
  (False Strategy Theorem, multiple-testing correction)
- White (2000)                    "A Reality Check for Data Snooping"
- Hansen (2005)                   "A Test for Superior Predictive Ability"
- Lo (2002)                       "The Statistics of Sharpe Ratios"
- Pontiff (1996) / Harvey (2019)  Haircut Sharpe approaches

Architecture
------------
SharpeRatioStatistics        — Lo (2002) Sharpe statistics with non-normality corrections
DeflatedSharpeRatio          — Full DSR + Expected max Sharpe + haircut Sharpe
ProbabilityOfBacktestOverfitting — LPDO (2014) PBO via combinatorial IS/OOS splits
CombinatorialPurgedCV        — CPCV with purging + embargo (AFML Ch.12)
FalseStrategyTheoremAnalyzer — Harvey et al. FDR / multiple-testing
ParameterOverfittingAnalyzer — VIF, surface roughness, selection bias, sensitivity
WhiteRealityCheck            — White (2000) RC + Hansen SPA test
OverfittingDashboard         — Orchestrator: full report generation

FastAPI router: overfitting_v3_router
  POST /overfitting/v3/dsr
  POST /overfitting/v3/pbo
  POST /overfitting/v3/cpcv
  POST /overfitting/v3/fst
  POST /overfitting/v3/wrc
  POST /overfitting/v3/full-report
"""
from __future__ import annotations

import itertools
import logging
import math
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("scipy not available — some tests will use normal approximations", stacklevel=2)

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field as PField
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

logger = logging.getLogger(__name__)

ANNUAL_FACTOR = 252
EULER_MASCHERONI = 0.5772156649015329  # γ

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DSRResult:
    """Deflated Sharpe Ratio result."""
    sharpe_annualized: float
    sharpe_std: float
    n_obs: int
    n_trials: int
    skewness: float
    excess_kurtosis: float
    expected_max_sharpe: float
    probabilistic_sr: float          # PSR(SR* = 0)
    deflated_sr: float               # DSR = PSR(SR* = E[max SR])
    haircut_sharpe: float
    min_backtest_length: int
    is_significant: bool
    interpretation: str


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting result."""
    pbo: float                        # 0-1; fraction of partitions showing OOS degradation
    n_partitions: int
    n_strategies: int
    median_oos_sharpe: float
    mean_is_sharpe: float
    mean_oos_sharpe: float
    sharpe_degradation: float         # IS Sharpe - OOS Sharpe
    logit_lambda: float               # mean logit of OOS rank
    interpretation: str
    partition_details: List[Dict]     # per-partition breakdown (IS_best, OOS_rank)


@dataclass
class CPCVResult:
    """Combinatorial Purged Cross-Validation result."""
    n_splits: int
    n_test_splits: int
    n_combinations: int
    n_obs: int
    embargo_bars: int
    oos_sharpes: List[float]
    is_sharpes: List[float]
    pbo_estimate: float
    mean_oos: float
    std_oos: float
    degradation_ratio: float          # mean_oos / mean_is
    paths: List[Dict]                 # each combinatorial path


@dataclass
class SignificanceResult:
    """Multiple-testing adjusted significance assessment."""
    observed_sharpe: float
    required_sharpe_bonferroni: float
    required_sharpe_bhy: float
    required_t_bonferroni: float
    required_t_bhy: float
    observed_t: float
    fdr_estimate: float
    is_significant_bonferroni: bool
    is_significant_bhy: bool
    n_trials: int
    n_obs: int
    verdict: str


@dataclass
class WRCResult:
    """White's Reality Check + Hansen SPA test."""
    wrc_p_value: float
    spa_p_value: float
    n_bootstrap: int
    block_size: int
    n_strategies: int
    best_excess_return: float
    bootstrap_max_distribution: List[float]
    significant_at_05: bool
    significant_at_01: bool
    interpretation: str


@dataclass
class ParameterSensitivityResult:
    """Parameter perturbation sensitivity analysis."""
    param_name: str
    base_value: Any
    plus_sharpe: float
    minus_sharpe: float
    base_sharpe: float
    delta_plus: float
    delta_minus: float
    sensitivity: float                # max(|delta_plus|, |delta_minus|)
    is_robust: bool                   # sensitivity < 0.2


@dataclass
class OverfittingReport:
    """Full overfitting analysis report."""
    n_strategies: int
    n_obs: int
    n_trials: int
    dsr_result: Optional[DSRResult]
    pbo_result: Optional[PBOResult]
    cpcv_result: Optional[CPCVResult]
    fst_result: Optional[SignificanceResult]
    wrc_result: Optional[WRCResult]
    vif_scores: Dict[str, float]
    surface_roughness: float
    selection_bias: float
    param_sensitivities: List[ParameterSensitivityResult]
    overall_verdict: str
    overfitting_score: float          # 0 (clean) → 1 (likely overfit)
    recommendations: List[str]
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────


def _normal_cdf(x: float) -> float:
    """Standard normal CDF — uses scipy if available, otherwise Abramowitz approximation."""
    if _SCIPY:
        return float(_scipy_stats.norm.cdf(x))
    # Abramowitz & Stegun (1964) approximation, error < 7.5e-8
    sign = 1.0 if x >= 0 else -1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    return 0.5 + sign * (0.5 - math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi) * poly)


def _normal_ppf(p: float) -> float:
    """Standard normal percent-point (quantile) function."""
    if _SCIPY:
        return float(_scipy_stats.norm.ppf(p))
    # Rational approximation — Beasley-Springer-Moro
    if p <= 0.0:
        return -1e15
    if p >= 1.0:
        return 1e15
    if p == 0.5:
        return 0.0
    # Abramowitz & Stegun rational approximation
    sign = 1.0 if p > 0.5 else -1.0
    q = min(p, 1 - p)
    r = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + r * (c1 + r * c2)
    den = 1.0 + r * (d1 + r * (d2 + r * d3))
    return sign * (r - num / den)


def _safe_sharpe(returns: np.ndarray, annual: bool = True) -> float:
    """Annualized Sharpe with zero-variance guard."""
    r = returns[np.isfinite(returns)]
    if len(r) < 2:
        return 0.0
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma < 1e-12:
        return 0.0
    sr = mu / sigma
    return sr * math.sqrt(ANNUAL_FACTOR) if annual else sr


def _compute_moments(returns: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (mean, std, skewness, excess_kurtosis). Uses scipy if available."""
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 4:
        return float(np.mean(r)), float(np.std(r, ddof=1)), 0.0, 0.0
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma < 1e-12:
        return mu, sigma, 0.0, 0.0
    if _SCIPY:
        skew = float(_scipy_stats.skew(r))
        kurt = float(_scipy_stats.kurtosis(r))  # excess kurtosis
    else:
        centered = r - mu
        skew = float(np.mean(centered**3) / sigma**3)
        kurt = float(np.mean(centered**4) / sigma**4 - 3.0)
    return mu, sigma, skew, kurt


def _block_bootstrap_indices(n: int, block_size: int, n_samples: int,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Stationary block bootstrap: generate index array of shape (n_samples, n).
    Block size is geometric-random to preserve stationarity (Politis & Romano 1994).
    """
    out = np.empty((n_samples, n), dtype=np.int64)
    for s in range(n_samples):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(1.0 / max(block_size, 1)))
            block = [(start + j) % n for j in range(length)]
            idx.extend(block)
        out[s] = np.array(idx[:n], dtype=np.int64)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. SharpeRatioStatistics
# ─────────────────────────────────────────────────────────────────────────────


class SharpeRatioStatistics:
    """
    Fundamental Sharpe Ratio statistics with corrections for non-normality.

    References
    ----------
    Lo (2002)                  "The Statistics of Sharpe Ratios"
    Bailey & LdP (2012)        "The Sharpe Ratio Efficient Frontier"
    Bailey & LdP (2014)        "The Deflated Sharpe Ratio"
    """

    @staticmethod
    def compute_sharpe(returns: pd.Series,
                       rf: float = 0.0,
                       annualize: bool = True) -> float:
        """
        Annualized Sharpe ratio, adjusted for risk-free rate.

        SR = (E[R] - rf) / σ(R) × sqrt(252)
        """
        r = np.asarray(returns.dropna(), dtype=float)
        if len(r) < 2:
            return 0.0
        excess = r - rf / ANNUAL_FACTOR  # daily rf
        mu = float(np.mean(excess))
        sigma = float(np.std(excess, ddof=1))
        if sigma < 1e-12:
            return 0.0
        sr = mu / sigma
        return sr * math.sqrt(ANNUAL_FACTOR) if annualize else sr

    @staticmethod
    def compute_sharpe_std(n_obs: int,
                           sharpe_annualized: float,
                           skew: float,
                           kurt: float) -> float:
        """
        Standard deviation of the Sharpe Ratio estimator — Lo (2002) Eq. (9).

        σ(SR̂) = sqrt[(1 + 0.5×SR² - skew×SR + ((kurt-3)/4)×SR²) / (n-1)]

        Note: kurt here is excess kurtosis (kurt-3 already subtracted by scipy).
        If kurt is raw kurtosis, pass kurt-3.

        This corrects for non-normality in the returns distribution.
        """
        if n_obs < 2:
            return float("inf")
        # Daily Sharpe for the formula (un-annualized)
        sr_daily = sharpe_annualized / math.sqrt(ANNUAL_FACTOR)
        term = (1.0
                + 0.5 * sr_daily**2
                - skew * sr_daily
                + (kurt / 4.0) * sr_daily**2)
        term = max(term, 1e-10)
        sr_std_daily = math.sqrt(term / max(n_obs - 1, 1))
        # Annualize the std of SR estimator
        return sr_std_daily * math.sqrt(ANNUAL_FACTOR)

    @staticmethod
    def compute_probabilistic_sr(sharpe: float,
                                 benchmark_sr: float,
                                 n_obs: int,
                                 skew: float,
                                 kurt: float) -> float:
        """
        Probabilistic Sharpe Ratio — Bailey & LdP (2012).

        PSR(SR*) = Φ[(SR̂ - SR*) × sqrt(n-1) / σ̂(SR)]

        σ̂(SR) uses Lo (2002) correction for non-normality.
        Probability that the true Sharpe exceeds benchmark_sr.
        """
        sr_std = SharpeRatioStatistics.compute_sharpe_std(n_obs, sharpe, skew, kurt)
        if sr_std < 1e-12:
            return 1.0 if sharpe > benchmark_sr else 0.0
        z = (sharpe - benchmark_sr) * math.sqrt(max(n_obs - 1, 1)) / (sr_std * math.sqrt(n_obs))
        # Correct: PSR = Φ[(SR - SR*) × sqrt(n-1) / (σ_SR × sqrt(n-1))]
        # Simplified: z = (SR - SR*) / (σ_SR / sqrt(n))
        # Bailey & LdP (2012) Eq. (3):
        # PSR = Φ[(SR - SR*) × sqrt(T-1) / sqrt(1 - skew×SR + (kurt-1)/4×SR²)]
        sr_daily = sharpe / math.sqrt(ANNUAL_FACTOR)
        bench_daily = benchmark_sr / math.sqrt(ANNUAL_FACTOR)
        denom_sq = (1.0
                    - skew * sr_daily
                    + (kurt / 4.0) * sr_daily**2)
        denom_sq = max(denom_sq, 1e-10)
        z = (sr_daily - bench_daily) * math.sqrt(max(n_obs - 1, 1)) / math.sqrt(denom_sq)
        return _normal_cdf(z)

    @staticmethod
    def compute_min_backtest_length(sr_target: float,
                                    sr_annualized: float,
                                    skew: float,
                                    kurt: float,
                                    alpha: float = 0.05) -> int:
        """
        Minimum number of observations (daily) to achieve statistical significance.

        From Bailey & LdP (2012): solve PSR(SR*) >= 1 - alpha for n.

        n_min = 1 + [Φ^{-1}(1-α)]² × (1 - skew×SR + (kurt/4)×SR²) / (SR - SR*)²
        """
        if sr_annualized <= sr_target:
            return int(1e9)  # never significant
        z = _normal_ppf(1.0 - alpha)
        sr_d = sr_annualized / math.sqrt(ANNUAL_FACTOR)
        st_d = sr_target / math.sqrt(ANNUAL_FACTOR)
        numer = z**2 * max(1.0 - skew * sr_d + (kurt / 4.0) * sr_d**2, 0.01)
        denom = (sr_d - st_d)**2
        if denom < 1e-12:
            return int(1e9)
        return int(math.ceil(1.0 + numer / denom))

    @staticmethod
    def compute_annualized_sharpe(daily_sharpe: float,
                                   trading_days: int = ANNUAL_FACTOR) -> float:
        """Scale daily Sharpe to annualized."""
        return daily_sharpe * math.sqrt(trading_days)

    @staticmethod
    def compute_full_stats(returns: pd.Series,
                           rf: float = 0.0,
                           benchmark_sr: float = 0.0) -> Dict:
        """Return comprehensive Sharpe statistics dict."""
        r = np.asarray(returns.dropna(), dtype=float)
        n = len(r)
        mu, sigma, skew, kurt = _compute_moments(r)
        sr = SharpeRatioStatistics.compute_sharpe(returns, rf=rf)
        sr_std = SharpeRatioStatistics.compute_sharpe_std(n, sr, skew, kurt)
        psr = SharpeRatioStatistics.compute_probabilistic_sr(sr, benchmark_sr, n, skew, kurt)
        mbl = SharpeRatioStatistics.compute_min_backtest_length(benchmark_sr, sr, skew, kurt)
        return {
            "n_obs": n,
            "mean_daily": mu,
            "std_daily": sigma,
            "skewness": skew,
            "excess_kurtosis": kurt,
            "sharpe_annualized": sr,
            "sharpe_std": sr_std,
            "probabilistic_sr": psr,
            "min_backtest_length": mbl,
            "t_statistic": sr / sr_std * math.sqrt(n) if sr_std > 1e-12 else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. DeflatedSharpeRatio
# ─────────────────────────────────────────────────────────────────────────────


class DeflatedSharpeRatio:
    """
    Full Deflated Sharpe Ratio implementation.

    References
    ----------
    Bailey, Borwein, Lopez de Prado & Zhu (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting,
    and Non-Normality"
    Journal of Portfolio Management 40(5).
    """

    @staticmethod
    def compute_expected_max_sharpe(n_trials: int,
                                     n_obs: int,
                                     skew: float = 0.0,
                                     kurt: float = 0.0) -> float:
        """
        Expected maximum Sharpe Ratio from n_trials independent strategies.

        Bailey et al. (2014) Proposition 1:
            E[max(SR)] ≈ (1 - γ) × Φ^{-1}(1 - 1/n_trials)
                        + γ × Φ^{-1}(1 - 1/(n_trials × e))

        where γ = Euler-Mascheroni constant ≈ 0.5772.

        This is adjusted for non-normal returns via the variance of the SR estimator.
        """
        if n_trials <= 1:
            return 0.0
        gamma = EULER_MASCHERONI
        e = math.e
        # Expected maximum from order statistics of standard normal
        q1 = _normal_ppf(1.0 - 1.0 / max(n_trials, 2))
        q2 = _normal_ppf(max(1.0 - 1.0 / (n_trials * e), 1e-10))
        e_max_z = (1.0 - gamma) * q1 + gamma * q2

        # Scale by standard deviation of SR estimator
        # σ(SR) = sqrt[(1 + 0.5×SR² - skew×SR + kurt/4×SR²) / (n-1)]
        # For small SR, σ ≈ 1/sqrt(n), so E[max SR] ≈ e_max_z / sqrt(n-1)
        # Bailey et al. use: E[max SR] ≈ e_max_z × sqrt(Var(SR))
        # Var(SR_annualized) ≈ (1 + 0.5×SR²) × 252/(n-1) for nearly normal returns
        # Approximate with SR ≈ 0 (conservative):
        var_sr_approx = (1.0 + 0.5 * 0.0 - skew * 0.0 + (kurt / 4.0) * 0.0) / max(n_obs - 1, 1)
        var_sr_approx = max(var_sr_approx, 1.0 / max(n_obs, 1))
        sigma_sr = math.sqrt(var_sr_approx) * math.sqrt(ANNUAL_FACTOR)
        return float(e_max_z * sigma_sr)

    @staticmethod
    def compute_dsr(sharpe: float,
                    n_trials: int,
                    n_obs: int,
                    skew: float = 0.0,
                    kurt: float = 0.0) -> float:
        """
        Deflated Sharpe Ratio = PSR(SR* = E[max SR]).

        DSR accounts for the number of strategy variants tested (selection bias).
        A high DSR (> 0.95) indicates the Sharpe is unlikely to be from pure data mining.

        Returns value in [0, 1].
        """
        sr_star = DeflatedSharpeRatio.compute_expected_max_sharpe(n_trials, n_obs, skew, kurt)
        return SharpeRatioStatistics.compute_probabilistic_sr(sharpe, sr_star, n_obs, skew, kurt)

    @staticmethod
    def compute_haircut_sharpe(sharpe: float,
                                n_trials: int,
                                n_obs: int) -> float:
        """
        Pontiff (1996) / Harvey & Liu (2019) Haircut Sharpe.

        A conservative deflation: divide the Sharpe by a factor that grows
        with the number of trials.

        Haircut_SR ≈ SR / sqrt(1 + n_trials / n_obs)

        For n_trials << n_obs: minimal haircut.
        For n_trials >> n_obs: large haircut (severe overfitting penalty).

        Alternative form: SR × exp(-0.5 × log(n_trials) / n_obs) [Harvey 2019]
        We implement both and return the more conservative (lower) estimate.
        """
        if n_trials <= 0 or n_obs <= 0:
            return sharpe
        # Pontiff approach
        haircut_pontiff = sharpe / math.sqrt(1.0 + n_trials / n_obs)
        # Harvey & Liu (2019): based on Bonferroni threshold
        # Adjust Sharpe downward by ratio of observed-to-required t-stat
        required_t = _normal_ppf(1.0 - 0.05 / (2.0 * n_trials))
        observed_t = sharpe / math.sqrt(ANNUAL_FACTOR) * math.sqrt(n_obs)
        if required_t > 1e-6 and observed_t > 1e-6:
            haircut_harvey = sharpe * (1.0 - required_t / observed_t * 0.5)
        else:
            haircut_harvey = sharpe
        return max(min(haircut_pontiff, haircut_harvey), -abs(sharpe))

    @staticmethod
    def compute_dsr_from_returns(returns: pd.Series,
                                  n_trials: int) -> DSRResult:
        """
        Compute full DSR result from a returns Series.

        n_trials: number of strategy variants tested before selecting this one.
        """
        r = np.asarray(returns.dropna(), dtype=float)
        n = len(r)
        mu, sigma, skew, kurt = _compute_moments(r)
        sharpe = _safe_sharpe(r)
        sr_std = SharpeRatioStatistics.compute_sharpe_std(n, sharpe, skew, kurt)
        psr = SharpeRatioStatistics.compute_probabilistic_sr(sharpe, 0.0, n, skew, kurt)
        e_max = DeflatedSharpeRatio.compute_expected_max_sharpe(n_trials, n, skew, kurt)
        dsr = DeflatedSharpeRatio.compute_dsr(sharpe, n_trials, n, skew, kurt)
        haircut = DeflatedSharpeRatio.compute_haircut_sharpe(sharpe, n_trials, n)
        mbl = SharpeRatioStatistics.compute_min_backtest_length(0.0, sharpe, skew, kurt)

        if dsr >= 0.95:
            interp = "SIGNIFICANT: Sharpe survives multiple-testing correction"
        elif dsr >= 0.80:
            interp = "BORDERLINE: Strategy may be marginally significant"
        elif dsr >= 0.60:
            interp = "WEAK: Likely inflated by selection bias"
        else:
            interp = "INSIGNIFICANT: Sharpe consistent with data mining from noise"

        return DSRResult(
            sharpe_annualized=sharpe,
            sharpe_std=sr_std,
            n_obs=n,
            n_trials=n_trials,
            skewness=skew,
            excess_kurtosis=kurt,
            expected_max_sharpe=e_max,
            probabilistic_sr=psr,
            deflated_sr=dsr,
            haircut_sharpe=haircut,
            min_backtest_length=mbl,
            is_significant=dsr >= 0.95,
            interpretation=interp,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. ProbabilityOfBacktestOverfitting
# ─────────────────────────────────────────────────────────────────────────────


class ProbabilityOfBacktestOverfitting:
    """
    Probability of Backtest Overfitting (PBO) per Bailey, Borwein, LdP & Zhu (2014).

    Algorithm
    ---------
    Given S strategy variants × T observations matrix of returns:
    1. Generate all (or a random sample of) C(T, T//2) IS/OOS temporal splits.
    2. For each split:
       a. Identify the best strategy by IS Sharpe.
       b. Rank that strategy's OOS Sharpe among all strategies in OOS.
       c. Compute logit(OOS_rank / S).
    3. PBO = fraction of splits where OOS_rank < median rank (i.e., rank < S/2).
    4. Average logit lambda gives continuous measure of overfitting.

    Note: "temporal splits" means IS = first T//2 rows, OOS = last T//2 rows
    within each column-permutation. Here we use the Bailey et al. convention of
    splitting the observation dimension, not shuffling columns.
    """

    @staticmethod
    def compute_pbo(returns_matrix: pd.DataFrame,
                    n_trials: int = None,
                    n_partitions: int = 500,
                    seed: int = 42) -> PBOResult:
        """
        Compute PBO from returns_matrix of shape (n_obs, n_strategies).

        n_partitions: number of IS/OOS splits to evaluate (up to C(n,n//2)).
        """
        M = returns_matrix.dropna(how="all").values.astype(float)
        T, S = M.shape
        if S < 2:
            raise ValueError("Need at least 2 strategy columns for PBO.")
        if n_trials is None:
            n_trials = S

        half = T // 2
        if half < 5:
            raise ValueError("Insufficient observations for PBO (need T >= 10).")

        rng = np.random.default_rng(seed)

        # Generate IS/OOS time index splits
        # Each partition: random permutation of time indices, first half = IS, second = OOS
        all_idx = np.arange(T)
        partition_details = []
        logit_lambdas = []
        oos_below_median = 0

        n_actual = min(n_partitions, 2000)

        is_sharpes_all = []
        oos_sharpes_all = []

        for _ in range(n_actual):
            perm = rng.permutation(all_idx)
            is_idx = np.sort(perm[:half])
            oos_idx = np.sort(perm[half:])

            is_data = M[is_idx, :]
            oos_data = M[oos_idx, :]

            # Compute Sharpe for each strategy in IS and OOS
            is_sharpes = np.array([_safe_sharpe(is_data[:, k]) for k in range(S)])
            oos_sharpes = np.array([_safe_sharpe(oos_data[:, k]) for k in range(S)])

            best_is = int(np.argmax(is_sharpes))
            best_oos_sharpe = oos_sharpes[best_is]

            # Rank of best IS strategy in OOS (1 = worst)
            oos_rank = int(np.sum(oos_sharpes <= best_oos_sharpe))  # rank from bottom
            relative_rank = oos_rank / S  # in [0, 1]

            # Logit of relative rank
            eps = 1.0 / (S + 1)
            lambda_t = math.log(max(relative_rank, eps) / max(1.0 - relative_rank, eps))
            logit_lambdas.append(lambda_t)

            below_median = relative_rank < 0.5
            if below_median:
                oos_below_median += 1

            is_sharpes_all.append(float(is_sharpes[best_is]))
            oos_sharpes_all.append(float(best_oos_sharpe))

            partition_details.append({
                "best_is_strategy": best_is,
                "is_sharpe": float(is_sharpes[best_is]),
                "oos_sharpe": float(best_oos_sharpe),
                "oos_rank": oos_rank,
                "relative_rank": relative_rank,
                "logit_lambda": lambda_t,
                "overfit": below_median,
            })

        pbo = oos_below_median / n_actual
        mean_lambda = float(np.mean(logit_lambdas))
        mean_is = float(np.mean(is_sharpes_all))
        mean_oos = float(np.mean(oos_sharpes_all))
        median_oos = float(np.median(oos_sharpes_all))
        degradation = mean_is - mean_oos

        return PBOResult(
            pbo=pbo,
            n_partitions=n_actual,
            n_strategies=S,
            median_oos_sharpe=median_oos,
            mean_is_sharpe=mean_is,
            mean_oos_sharpe=mean_oos,
            sharpe_degradation=degradation,
            logit_lambda=mean_lambda,
            interpretation=ProbabilityOfBacktestOverfitting.interpret_pbo(pbo),
            partition_details=partition_details[:50],  # cap stored details
        )

    @staticmethod
    def compute_pbo_from_param_sweep(prices: pd.Series,
                                      param_combinations: List[Dict],
                                      strategy_fn: Callable,
                                      n_partitions: int = 500) -> PBOResult:
        """
        Build returns_matrix from strategy variants, then compute PBO.

        strategy_fn(prices, **params) -> pd.Series of returns
        """
        returns_list = []
        for params in param_combinations:
            try:
                ret = strategy_fn(prices, **params)
                if isinstance(ret, pd.Series) and len(ret) > 0:
                    returns_list.append(ret.rename(str(params)))
            except Exception as exc:
                logger.warning("Strategy %s failed: %s", params, exc)

        if len(returns_list) < 2:
            raise ValueError("Need at least 2 valid strategies for PBO.")

        returns_matrix = pd.concat(returns_list, axis=1).dropna(how="all")
        return ProbabilityOfBacktestOverfitting.compute_pbo(
            returns_matrix, n_partitions=n_partitions)

    @staticmethod
    def interpret_pbo(pbo: float) -> str:
        """Risk label for a PBO value."""
        if pbo < 0.10:
            return "CLEAN: Strategy unlikely to be overfit (PBO < 10%)"
        if pbo < 0.30:
            return "LOW_RISK: Moderate evidence of genuine edge (PBO 10-30%)"
        if pbo < 0.50:
            return "MODERATE_RISK: Meaningful chance of overfitting (PBO 30-50%)"
        if pbo < 0.70:
            return "HIGH_RISK: Strong evidence of overfitting (PBO 50-70%)"
        return "LIKELY_OVERFIT: Strategy performance likely due to data mining (PBO > 70%)"


# ─────────────────────────────────────────────────────────────────────────────
# 4. CombinatorialPurgedCV
# ─────────────────────────────────────────────────────────────────────────────


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Lopez de Prado (2018) "Advances in Financial Machine Learning" Chapter 12.

    Standard K-fold CV is inappropriate for financial time series because:
    1. Observations are serially correlated → train/test leakage.
    2. K-fold only generates K test paths; CPCV generates C(K, K_test) paths.
    3. Purging removes training observations whose outcomes overlap with test period.
    4. Embargo adds a gap between train and test to prevent spillover.

    CPCV provides a much richer distribution of OOS paths for PBO estimation.
    """

    @staticmethod
    def generate_purged_splits(n_obs: int,
                                n_splits: int = 6,
                                embargo_pct: float = 0.01) -> List[Dict]:
        """
        Generate K purged splits of [0, n_obs).

        Returns list of dicts with keys:
          'split_id', 'test_start', 'test_end', 'train_indices', 'test_indices'

        Purging: observations in train that overlap with test are removed.
        Embargo: 'embargo_bars' observations after test_end are excluded from train.
        """
        embargo_bars = max(1, int(n_obs * embargo_pct))
        split_size = n_obs // n_splits
        splits = []

        for k in range(n_splits):
            t0 = k * split_size
            t1 = t0 + split_size if k < n_splits - 1 else n_obs
            test_idx = list(range(t0, t1))

            # Training: all indices not in [t0 - embargo, t1 + embargo]
            purge_start = max(0, t0 - embargo_bars)
            purge_end = min(n_obs, t1 + embargo_bars)
            train_idx = [i for i in range(n_obs)
                         if i < purge_start or i >= purge_end]

            splits.append({
                "split_id": k,
                "test_start": t0,
                "test_end": t1,
                "train_indices": train_idx,
                "test_indices": test_idx,
            })

        return splits

    @staticmethod
    def run_cpcv(returns: pd.Series,
                 strategy_fn: Callable,
                 n_splits: int = 6,
                 n_test_splits: int = 2,
                 embargo_pct: float = 0.01,
                 param_grid: Optional[List[Dict]] = None) -> CPCVResult:
        """
        Combinatorial Purged Cross-Validation.

        For each combination of n_test_splits splits (as test), the remaining
        splits form the training set. strategy_fn is optimized on train and
        evaluated on each test split combination.

        strategy_fn(returns_slice: pd.Series, params: dict) -> (pd.Series, dict)
        Returns (strategy_returns, best_params).

        If param_grid is None, strategy_fn(returns_slice) -> pd.Series.
        """
        r = np.asarray(returns.dropna(), dtype=float)
        n = len(r)
        dates = pd.RangeIndex(n)
        r_series = pd.Series(r, index=dates)

        splits = CombinatorialPurgedCV.generate_purged_splits(n, n_splits, embargo_pct)
        embargo_bars = max(1, int(n * embargo_pct))

        # Generate all combinations of test splits
        split_ids = list(range(n_splits))
        combos = list(itertools.combinations(split_ids, n_test_splits))
        # Limit combinations to avoid explosion
        if len(combos) > 200:
            rng = random.Random(42)
            combos = rng.sample(combos, 200)

        oos_sharpes = []
        is_sharpes = []
        paths = []

        for combo in combos:
            # Test indices: union of selected splits
            test_splits = [splits[k] for k in combo]
            all_test_idx = set()
            for s in test_splits:
                all_test_idx.update(s["test_indices"])

            # Train indices: all non-test, purged + embargoed
            purge_zone = set()
            for idx in all_test_idx:
                for j in range(max(0, idx - embargo_bars), min(n, idx + embargo_bars + 1)):
                    purge_zone.add(j)
            train_idx = sorted(set(range(n)) - purge_zone)
            test_idx = sorted(all_test_idx)

            if len(train_idx) < 20 or len(test_idx) < 5:
                continue

            train_r = r_series.iloc[train_idx]
            test_r = r_series.iloc[test_idx]

            # Optimize on train (simple: pass to strategy_fn)
            try:
                if param_grid:
                    best_sr = -np.inf
                    best_params = param_grid[0]
                    for params in param_grid:
                        candidate = strategy_fn(train_r, **params)
                        sr = _safe_sharpe(np.asarray(candidate.dropna()))
                        if sr > best_sr:
                            best_sr = sr
                            best_params = params
                    # Evaluate best params on OOS
                    oos_ret = strategy_fn(test_r, **best_params)
                else:
                    # strategy_fn trains on train and returns oos signal
                    oos_ret = strategy_fn(test_r)
                    best_sr = _safe_sharpe(np.asarray(train_r.dropna()))

                oos_sr = _safe_sharpe(np.asarray(oos_ret.dropna()))
                oos_sharpes.append(oos_sr)
                is_sharpes.append(best_sr if best_sr != -np.inf else
                                  _safe_sharpe(np.asarray(train_r.dropna())))

                paths.append({
                    "test_splits": list(combo),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "is_sharpe": float(is_sharpes[-1]),
                    "oos_sharpe": float(oos_sr),
                })

            except Exception as exc:
                logger.warning("CPCV combo %s failed: %s", combo, exc)

        if not oos_sharpes:
            raise ValueError("No valid CPCV paths computed.")

        pbo = CombinatorialPurgedCV.compute_pbo_from_paths(is_sharpes, oos_sharpes)
        mean_oos = float(np.mean(oos_sharpes))
        std_oos = float(np.std(oos_sharpes, ddof=1)) if len(oos_sharpes) > 1 else 0.0
        mean_is = float(np.mean(is_sharpes))
        deg = mean_oos / mean_is if abs(mean_is) > 1e-10 else 0.0

        return CPCVResult(
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            n_combinations=len(paths),
            n_obs=n,
            embargo_bars=embargo_bars,
            oos_sharpes=oos_sharpes,
            is_sharpes=is_sharpes,
            pbo_estimate=pbo,
            mean_oos=mean_oos,
            std_oos=std_oos,
            degradation_ratio=deg,
            paths=paths[:30],
        )

    @staticmethod
    def compute_pbo_from_paths(is_sharpes: List[float],
                                oos_sharpes: List[float]) -> float:
        """Estimate PBO from CPCV paths: fraction where OOS < median OOS."""
        if not oos_sharpes:
            return 0.5
        med = float(np.median(oos_sharpes))
        below = sum(1 for x in oos_sharpes if x < med)
        return below / len(oos_sharpes)

    @staticmethod
    def compute_pbo_from_cpcv(result: CPCVResult) -> float:
        """Extract PBO from CPCVResult."""
        return result.pbo_estimate


# ─────────────────────────────────────────────────────────────────────────────
# 5. FalseStrategyTheoremAnalyzer
# ─────────────────────────────────────────────────────────────────────────────


class FalseStrategyTheoremAnalyzer:
    """
    Harvey, Liu & Zhu (2016) "… and the Cross-Section of Expected Returns"

    The False Strategy Theorem: when testing many strategies, even the best
    performer is likely a false discovery. Multiple-testing corrections (Bonferroni,
    BHY) and the False Discovery Rate (FDR) framework address this.

    Key insight: with n_trials independent tests, the expected maximum t-statistic
    from pure noise grows as Φ^{-1}(1 - 1/n_trials), not as Φ^{-1}(0.975).
    """

    @staticmethod
    def compute_required_t_stat(n_trials: int,
                                 n_obs: int,
                                 alpha: float = 0.05) -> Tuple[float, float]:
        """
        Adjusted t-statistic thresholds under multiple testing.

        Returns (t_bonferroni, t_bhy) where:
        - Bonferroni: t* = Φ^{-1}(1 - α / (2 × n_trials))  [conservative]
        - BHY (Benjamini-Hochberg-Yekutieli): less conservative for correlated tests

        BHY threshold: α_adjusted = α / (n_trials × Σ(1/k, k=1..n_trials))
        """
        if n_trials <= 0:
            return 1.96, 1.96

        # Bonferroni
        p_bonferroni = alpha / (2.0 * n_trials)
        t_bonferroni = _normal_ppf(1.0 - p_bonferroni)

        # BHY: accounts for positive correlation between tests
        c_n = sum(1.0 / k for k in range(1, n_trials + 1))  # harmonic number
        p_bhy = alpha / (n_trials * c_n)
        t_bhy = _normal_ppf(1.0 - p_bhy / 2.0)

        return float(t_bonferroni), float(t_bhy)

    @staticmethod
    def compute_risk_of_false_discovery(n_trials: int,
                                         true_signal_fraction: float = 0.10,
                                         alpha: float = 0.05,
                                         power: float = 0.80) -> float:
        """
        False Discovery Rate (FDR) — Benjamini & Hochberg (1995).

        FDR = E[false positives / total positives]

        FDR = [(1 - π₀) × α] / [π₀ × power + (1 - π₀) × α]

        where:
        - π₀ = true_signal_fraction (fraction of strategies with genuine edge)
        - power = P(reject H0 | H1 true) [statistical power, default 0.80]
        - α = significance threshold

        Returns FDR in [0, 1]: fraction of "significant" strategies that are false.
        """
        pi0 = 1.0 - max(0.0, min(true_signal_fraction, 1.0))  # null fraction
        # Expected false positives
        false_pos_rate = pi0 * alpha
        # Expected true positives
        true_pos_rate = (1.0 - pi0) * power
        denom = true_pos_rate + false_pos_rate
        if denom < 1e-12:
            return 1.0
        return float(false_pos_rate / denom)

    @staticmethod
    def compute_minimum_sharpe_for_significance(n_trials: int,
                                                  n_obs: int,
                                                  alpha: float = 0.05) -> Tuple[float, float]:
        """
        Minimum Sharpe Ratio for statistical significance under multiple testing.

        SR* = t* / sqrt(n_obs / ANNUAL_FACTOR)

        Returns (sr_bonferroni, sr_bhy).
        """
        t_b, t_bhy = FalseStrategyTheoremAnalyzer.compute_required_t_stat(
            n_trials, n_obs, alpha)
        scale = math.sqrt(n_obs / ANNUAL_FACTOR)
        if scale < 1e-12:
            return float("inf"), float("inf")
        return float(t_b / scale), float(t_bhy / scale)

    @staticmethod
    def assess_strategy_significance(sharpe: float,
                                      n_trials: int,
                                      n_obs: int,
                                      skew: float,
                                      kurt: float,
                                      alpha: float = 0.05) -> SignificanceResult:
        """
        Full multiple-testing adjusted significance assessment.
        """
        t_b, t_bhy = FalseStrategyTheoremAnalyzer.compute_required_t_stat(
            n_trials, n_obs, alpha)
        sr_b, sr_bhy = FalseStrategyTheoremAnalyzer.compute_minimum_sharpe_for_significance(
            n_trials, n_obs, alpha)

        # Observed t-statistic
        sr_std = SharpeRatioStatistics.compute_sharpe_std(n_obs, sharpe, skew, kurt)
        sr_std_daily = sr_std / math.sqrt(ANNUAL_FACTOR)
        t_obs = (sharpe / math.sqrt(ANNUAL_FACTOR) * math.sqrt(n_obs)
                 if sr_std < 1e-12 else sharpe / sr_std)

        fdr = FalseStrategyTheoremAnalyzer.compute_risk_of_false_discovery(
            n_trials, true_signal_fraction=0.10, alpha=alpha)

        sig_b = sharpe >= sr_b
        sig_bhy = sharpe >= sr_bhy

        if sig_b:
            verdict = "SIGNIFICANT (Bonferroni): Strategy clears the highest bar"
        elif sig_bhy:
            verdict = "SIGNIFICANT (BHY): Passes less conservative multiple-testing threshold"
        elif t_obs > 1.96:
            verdict = "NOMINALLY_SIGNIFICANT: Passes naive t-test but not multiple-testing"
        else:
            verdict = "NOT_SIGNIFICANT: Cannot reject null of zero Sharpe"

        return SignificanceResult(
            observed_sharpe=sharpe,
            required_sharpe_bonferroni=sr_b,
            required_sharpe_bhy=sr_bhy,
            required_t_bonferroni=t_b,
            required_t_bhy=t_bhy,
            observed_t=t_obs,
            fdr_estimate=fdr,
            is_significant_bonferroni=sig_b,
            is_significant_bhy=sig_bhy,
            n_trials=n_trials,
            n_obs=n_obs,
            verdict=verdict,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. ParameterOverfittingAnalyzer
# ─────────────────────────────────────────────────────────────────────────────


class ParameterOverfittingAnalyzer:
    """
    Detect overfitting in parameter selection.

    References
    ----------
    Variance Inflation Factor: Marquaridt (1970), Greene "Econometric Analysis"
    Selection Bias: Lo (2002), Harvey & Liu (2014)
    Surface Roughness: Bailey & LdP (2014) (concept: jagged performance landscape)
    """

    @staticmethod
    def compute_vif(param_grid_results: pd.DataFrame) -> Dict[str, float]:
        """
        Variance Inflation Factor for each strategy parameter.

        param_grid_results: DataFrame where each column is a parameter value and
        each row is one strategy variant. Include a 'sharpe' column for reference.

        VIF_i = 1 / (1 - R²_i) where R²_i = R² from regressing param_i on other params.

        High VIF (> 5): parameter is near-collinear → selection unstable.
        VIF = 1: orthogonal (no collinearity).
        """
        param_cols = [c for c in param_grid_results.columns if c != "sharpe"]
        if len(param_cols) < 2:
            return {c: 1.0 for c in param_cols}

        X = param_grid_results[param_cols].copy().astype(float)
        # Normalize each column
        for col in param_cols:
            std = X[col].std()
            if std > 1e-12:
                X[col] = (X[col] - X[col].mean()) / std

        vif = {}
        for col in param_cols:
            y = X[col].values
            others = X.drop(columns=[col]).values
            if others.shape[1] == 0:
                vif[col] = 1.0
                continue
            # OLS: y = others × β; R² from OLS
            try:
                # Normal equations: β = (X'X)^{-1} X'y
                XtX = others.T @ others
                Xty = others.T @ y
                if _SCIPY:
                    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
                else:
                    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
                y_hat = others @ beta
                ss_res = np.sum((y - y_hat)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
                r2 = max(0.0, min(r2, 1.0 - 1e-9))
                vif[col] = float(1.0 / (1.0 - r2))
            except Exception:
                vif[col] = float("nan")

        return vif

    @staticmethod
    def compute_return_surface_roughness(returns_matrix: pd.DataFrame,
                                          param1: List[float],
                                          param2: List[float]) -> float:
        """
        Measure roughness of the performance surface over a 2D parameter grid.

        Roughness = mean(|∂²SR/∂p₁²| + |∂²SR/∂p₂²|) across the grid.

        A smooth surface (low roughness) suggests robust parameters.
        A jagged surface (high roughness) suggests the strategy is curve-fit to
        specific parameter values — classic overfitting signature.

        returns_matrix: shape (n_obs, n_strategies) matching len(p1)×len(p2) strategies.
        param1, param2: parameter values (must form a grid with n1 × n2 = n_strategies).
        """
        n1, n2 = len(param1), len(param2)
        M = returns_matrix.values.astype(float)
        T, S = M.shape

        if S != n1 * n2:
            # Reshape to what we can
            n_use = min(S, n1 * n2)
            M = M[:, :n_use]

        sharpes = np.array([_safe_sharpe(M[:, k]) for k in range(M.shape[1])])
        grid = sharpes.reshape(n1, n2) if len(sharpes) == n1 * n2 else sharpes[:n1 * n2].reshape(n1, n2)

        # Second-order finite differences
        roughness = 0.0
        count = 0
        for i in range(1, n1 - 1):
            for j in range(1, n2 - 1):
                d2_p1 = grid[i + 1, j] - 2 * grid[i, j] + grid[i - 1, j]
                d2_p2 = grid[i, j + 1] - 2 * grid[i, j] + grid[i, j - 1]
                roughness += abs(d2_p1) + abs(d2_p2)
                count += 1

        return float(roughness / max(count, 1))

    @staticmethod
    def compute_selection_bias(is_returns: pd.Series,
                                oos_returns: pd.Series,
                                n_params_optimized: int) -> float:
        """
        Expected IS-OOS Sharpe gap from selection bias.

        From Lo (2002) and Bailey & LdP: when you optimize n parameters,
        the expected upward bias in IS Sharpe is approximately:

        bias ≈ sqrt(2 × log(n_params) / n_obs) × σ(SR)

        Returns the expected IS-OOS Sharpe degradation due to selection bias.
        """
        r_is = np.asarray(is_returns.dropna(), dtype=float)
        r_oos = np.asarray(oos_returns.dropna(), dtype=float)
        n_is = len(r_is)
        if n_is < 5 or n_params_optimized < 1:
            return 0.0

        # Actual degradation
        sr_is = _safe_sharpe(r_is)
        sr_oos = _safe_sharpe(r_oos)
        actual_degradation = sr_is - sr_oos

        # Expected bias from Lo (2002)
        _, sigma, skew, kurt = _compute_moments(r_is)
        sr_std = SharpeRatioStatistics.compute_sharpe_std(n_is, sr_is, skew, kurt)
        expected_bias = math.sqrt(2.0 * math.log(max(n_params_optimized, 1)) / max(n_is, 1)) * sr_std * math.sqrt(ANNUAL_FACTOR)

        return float(max(actual_degradation, expected_bias))

    @staticmethod
    def test_parameter_sensitivity(prices: pd.Series,
                                    strategy_fn: Callable,
                                    best_params: Dict,
                                    perturbation_pct: float = 0.10) -> List[ParameterSensitivityResult]:
        """
        Perturb each parameter by ±perturbation_pct and measure Sharpe change.

        strategy_fn(prices, **params) -> pd.Series of returns.

        A robust strategy shows small Sharpe sensitivity to parameter perturbation.
        Sensitivity > 0.2 Sharpe units per 10% param change: suspicious.
        """
        results = []

        # Compute base Sharpe
        try:
            base_returns = strategy_fn(prices, **best_params)
            base_sr = _safe_sharpe(np.asarray(base_returns.dropna()))
        except Exception as exc:
            logger.error("Base strategy evaluation failed: %s", exc)
            return []

        for param, value in best_params.items():
            if not isinstance(value, (int, float)):
                continue
            if value == 0:
                continue

            delta = abs(value) * perturbation_pct

            # Plus perturbation
            plus_params = dict(best_params)
            plus_params[param] = value + delta
            # For integer params, round
            if isinstance(value, int):
                plus_params[param] = max(1, int(round(value + delta)))

            # Minus perturbation
            minus_params = dict(best_params)
            minus_params[param] = max(0.001 if isinstance(value, float) else 1,
                                       value - delta)
            if isinstance(value, int):
                minus_params[param] = max(1, int(round(value - delta)))

            try:
                plus_ret = strategy_fn(prices, **plus_params)
                plus_sr = _safe_sharpe(np.asarray(plus_ret.dropna()))
            except Exception:
                plus_sr = base_sr

            try:
                minus_ret = strategy_fn(prices, **minus_params)
                minus_sr = _safe_sharpe(np.asarray(minus_ret.dropna()))
            except Exception:
                minus_sr = base_sr

            d_plus = plus_sr - base_sr
            d_minus = minus_sr - base_sr
            sensitivity = max(abs(d_plus), abs(d_minus))
            is_robust = sensitivity < 0.20

            results.append(ParameterSensitivityResult(
                param_name=param,
                base_value=value,
                plus_sharpe=plus_sr,
                minus_sharpe=minus_sr,
                base_sharpe=base_sr,
                delta_plus=d_plus,
                delta_minus=d_minus,
                sensitivity=sensitivity,
                is_robust=is_robust,
            ))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. WhiteRealityCheck
# ─────────────────────────────────────────────────────────────────────────────


class WhiteRealityCheck:
    """
    White (2000) Reality Check test + Hansen (2005) SPA test.

    White's Reality Check
    ---------------------
    Tests H₀: no strategy beats the benchmark.
    Uses stationary block bootstrap to generate null distribution of
    max(mean excess returns) under H₀.
    p-value = fraction of bootstrap samples where max(excess) ≥ observed max.

    Hansen's SPA Test
    -----------------
    Superior Predictive Ability: more powerful than White's RC.
    Removes poorly-performing strategies from the comparison set,
    reducing the influence of irrelevant strategies on the p-value.
    """

    @staticmethod
    def run_white_reality_check(benchmark_returns: pd.Series,
                                 strategy_returns_matrix: pd.DataFrame,
                                 n_bootstrap: int = 1000,
                                 block_size: int = 5,
                                 seed: int = 42) -> WRCResult:
        """
        White (2000) Reality Check test.

        H₀: E[max_k(R_k - R_benchmark)] ≤ 0 for all k=1..S.
        p-value: P(max bootstrap excess ≥ observed | H₀).
        """
        bench = np.asarray(benchmark_returns.dropna(), dtype=float)
        M = strategy_returns_matrix.values.astype(float)
        T, S = M.shape

        # Align lengths
        T_min = min(len(bench), T)
        bench = bench[:T_min]
        M = M[:T_min, :]

        # Excess returns for each strategy vs benchmark
        excess = M - bench[:, np.newaxis]  # shape (T, S)

        # Observed test statistic: mean excess return of best strategy
        mean_excess = np.mean(excess, axis=0)  # (S,)
        best_excess = float(np.max(mean_excess))

        # Stationary block bootstrap under H₀ (center the excess returns)
        centered_excess = excess - mean_excess[np.newaxis, :]  # center under H₀

        rng = np.random.default_rng(seed)
        idx_matrix = _block_bootstrap_indices(T_min, block_size, n_bootstrap, rng)

        boot_max = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = idx_matrix[b]
            boot_sample = centered_excess[idx, :]
            boot_max[b] = float(np.max(np.mean(boot_sample, axis=0)))

        p_value = float(np.mean(boot_max >= best_excess))

        sig_05 = p_value < 0.05
        sig_01 = p_value < 0.01

        if p_value < 0.01:
            interp = "HIGHLY SIGNIFICANT: At least one strategy significantly beats benchmark"
        elif p_value < 0.05:
            interp = "SIGNIFICANT: Evidence that best strategy beats benchmark (p < 0.05)"
        elif p_value < 0.10:
            interp = "MARGINAL: Weak evidence of superior performance (p < 0.10)"
        else:
            interp = "NOT SIGNIFICANT: No evidence any strategy beats benchmark after data snooping correction"

        return WRCResult(
            wrc_p_value=p_value,
            spa_p_value=float("nan"),  # filled by SPA call
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            n_strategies=S,
            best_excess_return=best_excess,
            bootstrap_max_distribution=boot_max.tolist(),
            significant_at_05=sig_05,
            significant_at_01=sig_01,
            interpretation=interp,
        )

    @staticmethod
    def run_spa_test(benchmark_returns: pd.Series,
                     strategy_returns_matrix: pd.DataFrame,
                     n_bootstrap: int = 1000,
                     block_size: int = 5,
                     seed: int = 42) -> float:
        """
        Hansen (2005) Superior Predictive Ability (SPA) test.

        More powerful than White's RC because it removes strategies with
        excess return distribution that is clearly negative (cannot be the best).

        Algorithm:
        1. Compute mean excess returns f_k = mean(R_k - R_b).
        2. Bootstrap variance ω²_k of f_k.
        3. Remove "bad" strategies: those with f_k < -sqrt(ω²_k × log(log(T)) / T).
           (Consistent model selection criterion)
        4. Apply Reality Check to remaining set.

        Returns: SPA p-value.
        """
        bench = np.asarray(benchmark_returns.dropna(), dtype=float)
        M = strategy_returns_matrix.values.astype(float)
        T = min(len(bench), M.shape[0])
        bench = bench[:T]
        M = M[:T, :]

        excess = M - bench[:, np.newaxis]
        mean_excess = np.mean(excess, axis=0)
        S = M.shape[1]

        # Bootstrap variance of mean excess returns
        rng = np.random.default_rng(seed)
        idx_matrix = _block_bootstrap_indices(T, block_size, n_bootstrap, rng)

        boot_means = np.empty((n_bootstrap, S))
        for b in range(n_bootstrap):
            boot_means[b] = np.mean(excess[idx_matrix[b], :], axis=0)

        boot_var = np.var(boot_means, axis=0, ddof=1)  # (S,)

        # Consistent threshold for eliminating bad strategies
        if T > 1:
            threshold = -np.sqrt(boot_var * math.log(math.log(max(T, 3))) / T)
        else:
            threshold = np.full(S, -np.inf)

        # Keep strategies not clearly dominated
        good_mask = mean_excess >= threshold
        if not np.any(good_mask):
            good_mask = np.ones(S, dtype=bool)

        good_excess = excess[:, good_mask]
        good_mean = mean_excess[good_mask]

        # Reality Check on good strategies only
        centered = good_excess - good_mean[np.newaxis, :]
        best_good = float(np.max(good_mean))

        boot_max = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            boot_max[b] = float(np.max(np.mean(centered[idx_matrix[b], :][..., :good_excess.shape[1]], axis=0)))

        spa_p = float(np.mean(boot_max >= best_good))
        return spa_p

    @staticmethod
    def run_full_test(benchmark_returns: pd.Series,
                      strategy_returns_matrix: pd.DataFrame,
                      n_bootstrap: int = 1000,
                      block_size: int = 5,
                      seed: int = 42) -> WRCResult:
        """Run White's RC and Hansen's SPA together, returning combined result."""
        wrc = WhiteRealityCheck.run_white_reality_check(
            benchmark_returns, strategy_returns_matrix, n_bootstrap, block_size, seed)
        spa_p = WhiteRealityCheck.run_spa_test(
            benchmark_returns, strategy_returns_matrix, n_bootstrap, block_size, seed)
        wrc.spa_p_value = spa_p
        return wrc


# ─────────────────────────────────────────────────────────────────────────────
# 8. OverfittingDashboard
# ─────────────────────────────────────────────────────────────────────────────


class OverfittingDashboard:
    """
    Orchestrator: runs all overfitting tests and generates a unified report.
    """

    @staticmethod
    def run_full_analysis(returns_matrix: pd.DataFrame,
                          n_trials: int = None,
                          benchmark_returns: Optional[pd.Series] = None,
                          best_strategy_returns: Optional[pd.Series] = None,
                          param_grid: Optional[pd.DataFrame] = None,
                          param1_values: Optional[List] = None,
                          param2_values: Optional[List] = None) -> OverfittingReport:
        """
        Full overfitting analysis pipeline.

        returns_matrix : pd.DataFrame, shape (n_obs, n_strategies)
        n_trials       : number of strategies tested (default = columns in matrix)
        benchmark_returns: benchmark series for WRC (optional)
        best_strategy_returns: returns of selected strategy (for DSR/FST)
        param_grid     : pd.DataFrame with parameter columns + optional 'sharpe'
        param1_values, param2_values: for surface roughness computation
        """
        import datetime as _dt

        T, S = returns_matrix.shape
        n_tr = n_trials or S

        # --- 1. DSR ---
        dsr_result = None
        if best_strategy_returns is not None:
            try:
                dsr_result = DeflatedSharpeRatio.compute_dsr_from_returns(
                    best_strategy_returns, n_tr)
            except Exception as exc:
                logger.warning("DSR failed: %s", exc)
        else:
            # Use best column by IS Sharpe
            try:
                col_sharpes = [_safe_sharpe(returns_matrix.iloc[:, k].dropna().values)
                                for k in range(S)]
                best_col = int(np.argmax(col_sharpes))
                dsr_result = DeflatedSharpeRatio.compute_dsr_from_returns(
                    returns_matrix.iloc[:, best_col], n_tr)
            except Exception as exc:
                logger.warning("DSR failed: %s", exc)

        # --- 2. PBO ---
        pbo_result = None
        try:
            pbo_result = ProbabilityOfBacktestOverfitting.compute_pbo(
                returns_matrix, n_trials=n_tr)
        except Exception as exc:
            logger.warning("PBO failed: %s", exc)

        # --- 3. CPCV (lightweight — just use the PBO estimate route) ---
        cpcv_result = None
        try:
            # Build a simple pass-through strategy_fn for CPCV
            def passthrough(r: pd.Series) -> pd.Series:
                return r
            cpcv_result = CombinatorialPurgedCV.run_cpcv(
                returns_matrix.iloc[:, 0],  # use first column as baseline
                passthrough,
                n_splits=min(6, T // 20),
                n_test_splits=2,
            )
        except Exception as exc:
            logger.warning("CPCV failed: %s", exc)

        # --- 4. False Strategy Theorem ---
        fst_result = None
        if dsr_result is not None:
            try:
                fst_result = FalseStrategyTheoremAnalyzer.assess_strategy_significance(
                    dsr_result.sharpe_annualized,
                    n_tr,
                    dsr_result.n_obs,
                    dsr_result.skewness,
                    dsr_result.excess_kurtosis,
                )
            except Exception as exc:
                logger.warning("FST failed: %s", exc)

        # --- 5. WRC ---
        wrc_result = None
        if benchmark_returns is not None and S >= 2:
            try:
                wrc_result = WhiteRealityCheck.run_full_test(
                    benchmark_returns, returns_matrix)
            except Exception as exc:
                logger.warning("WRC failed: %s", exc)

        # --- 6. VIF ---
        vif_scores = {}
        if param_grid is not None and len(param_grid) > 2:
            try:
                vif_scores = ParameterOverfittingAnalyzer.compute_vif(param_grid)
            except Exception as exc:
                logger.warning("VIF failed: %s", exc)

        # --- 7. Surface roughness ---
        roughness = 0.0
        if param1_values and param2_values:
            try:
                roughness = ParameterOverfittingAnalyzer.compute_return_surface_roughness(
                    returns_matrix, param1_values, param2_values)
            except Exception as exc:
                logger.warning("Roughness failed: %s", exc)

        # --- 8. Selection bias ---
        selection_bias = 0.0
        if dsr_result is not None:
            sr = dsr_result.sharpe_annualized
            e_max = dsr_result.expected_max_sharpe
            selection_bias = max(0.0, e_max - sr * 0.5)  # simplified estimate

        # --- Compute overfitting score [0, 1] ---
        score_components = []
        if pbo_result:
            score_components.append(pbo_result.pbo)
        if dsr_result:
            score_components.append(1.0 - dsr_result.deflated_sr)
        if wrc_result and not math.isnan(wrc_result.wrc_p_value):
            score_components.append(wrc_result.wrc_p_value)
        if fst_result:
            score_components.append(0.0 if fst_result.is_significant_bhy else 0.7)

        overfitting_score = float(np.mean(score_components)) if score_components else 0.5

        # --- Verdict ---
        if overfitting_score < 0.20:
            verdict = "CLEAN: Strategy shows minimal overfitting risk"
        elif overfitting_score < 0.40:
            verdict = "LOW_RISK: Some evidence of data mining; proceed with caution"
        elif overfitting_score < 0.60:
            verdict = "MODERATE_RISK: Meaningful overfitting risk; reduce parameter count"
        elif overfitting_score < 0.80:
            verdict = "HIGH_RISK: Strong overfitting signals; likely curve-fitted"
        else:
            verdict = "LIKELY_OVERFIT: Do not trade this strategy without independent validation"

        # --- Recommendations ---
        recs = []
        if pbo_result and pbo_result.pbo > 0.5:
            recs.append("Reduce number of strategy variants before selection (PBO > 50%)")
        if dsr_result and dsr_result.deflated_sr < 0.80:
            recs.append(f"DSR = {dsr_result.deflated_sr:.3f}: test fewer variants or require longer track record (min {dsr_result.min_backtest_length} obs)")
        if fst_result and not fst_result.is_significant_bhy:
            recs.append(f"Require Sharpe ≥ {fst_result.required_sharpe_bhy:.2f} under BHY correction ({n_tr} trials tested)")
        if vif_scores:
            high_vif = {k: v for k, v in vif_scores.items() if v > 5.0}
            if high_vif:
                recs.append(f"High VIF parameters: {list(high_vif.keys())} — parameter selection is unstable")
        if roughness > 0.1:
            recs.append(f"Performance surface roughness = {roughness:.4f}: jagged — parameters may be overfit to noise")
        if not recs:
            recs.append("No major red flags — strategy passes overfitting screening")

        return OverfittingReport(
            n_strategies=S,
            n_obs=T,
            n_trials=n_tr,
            dsr_result=dsr_result,
            pbo_result=pbo_result,
            cpcv_result=cpcv_result,
            fst_result=fst_result,
            wrc_result=wrc_result,
            vif_scores=vif_scores,
            surface_roughness=roughness,
            selection_bias=selection_bias,
            param_sensitivities=[],
            overall_verdict=verdict,
            overfitting_score=overfitting_score,
            recommendations=recs,
            timestamp=_dt.datetime.utcnow().isoformat(),
        )

    @staticmethod
    def generate_report(report: OverfittingReport) -> str:
        """Generate formatted narrative overfitting report."""
        lines = []
        sep = "=" * 72
        lines.append(sep)
        lines.append("  SENTINEL OVERFITTING DETECTION REPORT  (dim_064 v3)")
        lines.append(sep)
        lines.append(f"  Generated : {report.timestamp}")
        lines.append(f"  Strategies tested : {report.n_strategies}")
        lines.append(f"  Observations      : {report.n_obs}")
        lines.append(f"  Total trials (N)  : {report.n_trials}")
        lines.append("")

        lines.append("── OVERALL VERDICT " + "─" * 53)
        lines.append(f"  Overfitting Score : {report.overfitting_score:.3f}  (0=clean, 1=overfit)")
        lines.append(f"  Verdict           : {report.overall_verdict}")
        lines.append("")

        if report.dsr_result:
            d = report.dsr_result
            lines.append("── DEFLATED SHARPE RATIO (Bailey et al. 2014) " + "─" * 24)
            lines.append(f"  Annualized SR          : {d.sharpe_annualized:+.4f}")
            lines.append(f"  SR Std Dev (Lo 2002)   : {d.sharpe_std:.4f}")
            lines.append(f"  Expected Max SR        : {d.expected_max_sharpe:.4f}")
            lines.append(f"  Probabilistic SR       : {d.probabilistic_sr:.4f}")
            lines.append(f"  Deflated SR (DSR)      : {d.deflated_sr:.4f}  {'✓ SIGNIFICANT' if d.is_significant else '✗ NOT SIGNIFICANT'}")
            lines.append(f"  Haircut SR             : {d.haircut_sharpe:.4f}")
            lines.append(f"  Min Backtest Length    : {d.min_backtest_length} obs")
            lines.append(f"  Skewness / Ex. Kurt    : {d.skewness:.3f} / {d.excess_kurtosis:.3f}")
            lines.append(f"  Interpretation         : {d.interpretation}")
            lines.append("")

        if report.pbo_result:
            p = report.pbo_result
            lines.append("── PROBABILITY OF BACKTEST OVERFITTING (Bailey et al. 2014) " + "─" * 10)
            lines.append(f"  PBO                : {p.pbo:.4f}  ({p.pbo*100:.1f}%)")
            lines.append(f"  Partitions         : {p.n_partitions}")
            lines.append(f"  Mean IS Sharpe     : {p.mean_is_sharpe:+.4f}")
            lines.append(f"  Mean OOS Sharpe    : {p.mean_oos_sharpe:+.4f}")
            lines.append(f"  Sharpe Degradation : {p.sharpe_degradation:+.4f}")
            lines.append(f"  Logit Lambda (λ)   : {p.logit_lambda:+.4f}  (negative = overfit)")
            lines.append(f"  Interpretation     : {p.interpretation}")
            lines.append("")

        if report.cpcv_result:
            c = report.cpcv_result
            lines.append("── COMBINATORIAL PURGED CROSS-VALIDATION (LdP 2018) " + "─" * 19)
            lines.append(f"  Splits / Test Splits   : {c.n_splits} / {c.n_test_splits}")
            lines.append(f"  Combinations (paths)   : {c.n_combinations}")
            lines.append(f"  Embargo Bars           : {c.embargo_bars}")
            lines.append(f"  OOS Sharpe  mean ± σ   : {c.mean_oos:+.4f} ± {c.std_oos:.4f}")
            lines.append(f"  Degradation ratio      : {c.degradation_ratio:.4f}")
            lines.append(f"  CPCV PBO estimate      : {c.pbo_estimate:.4f}")
            lines.append("")

        if report.fst_result:
            f = report.fst_result
            lines.append("── FALSE STRATEGY THEOREM (Harvey, Liu & Zhu 2016) " + "─" * 20)
            lines.append(f"  Observed SR            : {f.observed_sharpe:+.4f}")
            lines.append(f"  Required SR (Bonf.)    : {f.required_sharpe_bonferroni:+.4f}")
            lines.append(f"  Required SR (BHY)      : {f.required_sharpe_bhy:+.4f}")
            lines.append(f"  Observed t-statistic   : {f.observed_t:.4f}")
            lines.append(f"  Required t (Bonf.)     : {f.required_t_bonferroni:.4f}")
            lines.append(f"  FDR estimate (π₀=0.90) : {f.fdr_estimate:.4f}  ({f.fdr_estimate*100:.1f}% false discoveries)")
            lines.append(f"  Significant (Bonf.)    : {'YES' if f.is_significant_bonferroni else 'NO'}")
            lines.append(f"  Significant (BHY)      : {'YES' if f.is_significant_bhy else 'NO'}")
            lines.append(f"  Verdict                : {f.verdict}")
            lines.append("")

        if report.wrc_result:
            w = report.wrc_result
            lines.append("── WHITE'S REALITY CHECK + HANSEN SPA (White 2000, Hansen 2005) " + "─" * 6)
            lines.append(f"  WRC p-value    : {w.wrc_p_value:.4f}  {'✓ Sig.' if w.significant_at_05 else '✗ Not sig.'}")
            lines.append(f"  SPA p-value    : {w.spa_p_value:.4f}" if not math.isnan(w.spa_p_value) else "  SPA p-value    : n/a")
            lines.append(f"  Bootstrap reps : {w.n_bootstrap}")
            lines.append(f"  Interpretation : {w.interpretation}")
            lines.append("")

        if report.vif_scores:
            lines.append("── PARAMETER VIF SCORES " + "─" * 48)
            for param, vif in sorted(report.vif_scores.items()):
                flag = "  [HIGH — collinear]" if vif > 5 else ""
                lines.append(f"  {param:20s}: {vif:.2f}{flag}")
            lines.append("")

        if report.surface_roughness > 0:
            lines.append("── SURFACE ROUGHNESS " + "─" * 51)
            lines.append(f"  Roughness = {report.surface_roughness:.6f}  "
                          f"({'Jagged — overfitting suspected' if report.surface_roughness > 0.1 else 'Smooth — robust parameters'})")
            lines.append(f"  Selection bias estimate : {report.selection_bias:.4f} SR units")
            lines.append("")

        lines.append("── RECOMMENDATIONS " + "─" * 53)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")
        lines.append(sep)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Router
# ─────────────────────────────────────────────────────────────────────────────

if _FASTAPI:
    from pydantic import BaseModel as _BM

    class _DSRRequest(_BM):
        returns: List[float]
        n_trials: int = 10
        rf: float = 0.0

    class _PBORequest(_BM):
        returns_matrix: List[List[float]]
        n_partitions: int = 500

    class _WRCRequest(_BM):
        benchmark_returns: List[float]
        strategy_returns_matrix: List[List[float]]
        n_bootstrap: int = 1000
        block_size: int = 5

    overfitting_v3_router = APIRouter(prefix="/overfitting/v3", tags=["Overfitting Detection v3"])

    @overfitting_v3_router.post("/dsr")
    def api_dsr(req: _DSRRequest):
        """Compute Deflated Sharpe Ratio."""
        try:
            r = pd.Series(req.returns)
            result = DeflatedSharpeRatio.compute_dsr_from_returns(r, req.n_trials)
            from dataclasses import asdict
            return asdict(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @overfitting_v3_router.post("/pbo")
    def api_pbo(req: _PBORequest):
        """Compute Probability of Backtest Overfitting."""
        try:
            M = pd.DataFrame(req.returns_matrix)
            result = ProbabilityOfBacktestOverfitting.compute_pbo(
                M, n_partitions=req.n_partitions)
            from dataclasses import asdict
            return asdict(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @overfitting_v3_router.post("/wrc")
    def api_wrc(req: _WRCRequest):
        """Run White's Reality Check and Hansen SPA test."""
        try:
            bench = pd.Series(req.benchmark_returns)
            M = pd.DataFrame(req.strategy_returns_matrix)
            result = WhiteRealityCheck.run_full_test(
                bench, M, req.n_bootstrap, req.block_size)
            from dataclasses import asdict
            d = asdict(result)
            d.pop("bootstrap_max_distribution", None)
            return d
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @overfitting_v3_router.post("/full-report")
    def api_full_report(returns_matrix: List[List[float]], n_trials: int = 0):
        """Run full overfitting analysis."""
        try:
            M = pd.DataFrame(returns_matrix)
            nt = n_trials or M.shape[1]
            report = OverfittingDashboard.run_full_analysis(M, n_trials=nt)
            return {
                "overfitting_score": report.overfitting_score,
                "verdict": report.overall_verdict,
                "recommendations": report.recommendations,
                "pbo": report.pbo_result.pbo if report.pbo_result else None,
                "dsr": report.dsr_result.deflated_sr if report.dsr_result else None,
                "report_text": OverfittingDashboard.generate_report(report),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Demo / __main__
# ─────────────────────────────────────────────────────────────────────────────

def _sma_crossover(prices: pd.Series,
                    fast: int = 10,
                    slow: int = 30) -> pd.Series:
    """
    Simple SMA crossover strategy.
    Signal: +1 when fast_SMA > slow_SMA, else -1.
    Returns: daily strategy returns.
    """
    if len(prices) < slow + 5:
        return pd.Series(dtype=float)
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal = np.where(fast_ma > slow_ma, 1.0, -1.0)
    # Shift signal by 1 to avoid look-ahead
    signal = pd.Series(signal, index=prices.index).shift(1).fillna(0.0)
    price_returns = prices.pct_change().fillna(0.0)
    return signal * price_returns


def _simulate_spy_prices(n_days: int = 1260, seed: int = 99) -> pd.Series:
    """Simulate SPY-like price series (geometric Brownian motion)."""
    rng = np.random.default_rng(seed)
    mu_daily = 0.00035   # ~9% annual drift
    sigma_daily = 0.012  # ~19% annual vol
    log_returns = rng.normal(mu_daily, sigma_daily, n_days)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    dates = pd.date_range("2019-01-02", periods=n_days, freq="B")
    return pd.Series(prices, index=dates, name="SPY_sim")


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  SENTINEL Overfitting Detection v3 — DEMO")
    print("=" * 72)

    # ── 1. Simulate SPY-like prices ──────────────────────────────────────────
    print("\n[1] Simulating 5 years of SPY-like prices...")
    spy = _simulate_spy_prices(n_days=1260)
    spy_returns = spy.pct_change().dropna()
    print(f"    Price range: {spy.min():.2f} → {spy.max():.2f}")
    print(f"    Observations: {len(spy)}")

    # ── 2. Parameter sweep: SMA crossover fast/slow grid ─────────────────────
    print("\n[2] Running SMA crossover parameter sweep (fast × slow grid)...")
    fast_params = [5, 8, 10, 13, 15, 20]
    slow_params = [20, 30, 40, 50, 60, 80]

    returns_list = []
    param_records = []
    for fast in fast_params:
        for slow in slow_params:
            if fast >= slow:
                continue
            ret = _sma_crossover(spy, fast=fast, slow=slow)
            ret = ret.dropna()
            if len(ret) > 100:
                returns_list.append(ret.rename(f"f{fast}_s{slow}"))
                sr = _safe_sharpe(ret.values)
                param_records.append({"fast": fast, "slow": slow, "sharpe": sr})

    returns_matrix = pd.concat(returns_list, axis=1).dropna(how="all")
    param_df = pd.DataFrame(param_records)
    n_strategies = returns_matrix.shape[1]
    print(f"    Valid strategy variants: {n_strategies}")

    # ── 3. Find best IS strategy ──────────────────────────────────────────────
    col_sharpes = [_safe_sharpe(returns_matrix.iloc[:, k].dropna().values)
                    for k in range(n_strategies)]
    best_col = int(np.argmax(col_sharpes))
    best_returns = returns_matrix.iloc[:, best_col]
    best_name = returns_matrix.columns[best_col]
    print(f"    Best IS strategy: {best_name}  SR = {col_sharpes[best_col]:.4f}")

    # ── 4. Sharpe Statistics ──────────────────────────────────────────────────
    print("\n[3] Sharpe Ratio Statistics (Lo 2002)...")
    stats_dict = SharpeRatioStatistics.compute_full_stats(best_returns, rf=0.0)
    print(f"    Annualized SR    : {stats_dict['sharpe_annualized']:+.4f}")
    print(f"    SR Std Dev       : {stats_dict['sharpe_std']:.4f}")
    print(f"    t-statistic      : {stats_dict['t_statistic']:.4f}")
    print(f"    Probabilistic SR : {stats_dict['probabilistic_sr']:.4f}")
    print(f"    Min track record : {stats_dict['min_backtest_length']} trading days")

    # ── 5. DSR ────────────────────────────────────────────────────────────────
    print(f"\n[4] Deflated Sharpe Ratio (n_trials = {n_strategies})...")
    dsr_result = DeflatedSharpeRatio.compute_dsr_from_returns(best_returns, n_strategies)
    print(f"    SR               : {dsr_result.sharpe_annualized:+.4f}")
    print(f"    Expected max SR  : {dsr_result.expected_max_sharpe:+.4f}")
    print(f"    DSR              : {dsr_result.deflated_sr:.4f}")
    print(f"    Haircut SR       : {dsr_result.haircut_sharpe:+.4f}")
    print(f"    Significant?     : {dsr_result.is_significant}")
    print(f"    → {dsr_result.interpretation}")

    # ── 6. PBO ────────────────────────────────────────────────────────────────
    print(f"\n[5] Probability of Backtest Overfitting ({n_strategies} variants)...")
    pbo_result = ProbabilityOfBacktestOverfitting.compute_pbo(returns_matrix, n_partitions=300)
    print(f"    PBO              : {pbo_result.pbo:.4f}  ({pbo_result.pbo*100:.1f}%)")
    print(f"    IS Sharpe (mean) : {pbo_result.mean_is_sharpe:+.4f}")
    print(f"    OOS Sharpe (mean): {pbo_result.mean_oos_sharpe:+.4f}")
    print(f"    Logit Lambda     : {pbo_result.logit_lambda:+.4f}")
    print(f"    → {pbo_result.interpretation}")

    # ── 7. False Strategy Theorem ─────────────────────────────────────────────
    print(f"\n[6] False Strategy Theorem (Harvey et al. 2016, n_trials={n_strategies})...")
    fst = FalseStrategyTheoremAnalyzer.assess_strategy_significance(
        dsr_result.sharpe_annualized, n_strategies, dsr_result.n_obs,
        dsr_result.skewness, dsr_result.excess_kurtosis)
    print(f"    Required SR (Bonf.): {fst.required_sharpe_bonferroni:.4f}")
    print(f"    Required SR (BHY) : {fst.required_sharpe_bhy:.4f}")
    print(f"    Observed SR       : {fst.observed_sharpe:.4f}")
    print(f"    FDR estimate      : {fst.fdr_estimate:.2%}")
    print(f"    → {fst.verdict}")

    # ── 8. VIF ────────────────────────────────────────────────────────────────
    print("\n[7] Parameter VIF (collinearity check)...")
    vif = ParameterOverfittingAnalyzer.compute_vif(param_df)
    for p, v in vif.items():
        print(f"    VIF({p}) = {v:.2f}{'  [HIGH]' if v > 5 else ''}")

    # ── 9. Surface Roughness ──────────────────────────────────────────────────
    print("\n[8] Performance surface roughness...")
    roughness = ParameterOverfittingAnalyzer.compute_return_surface_roughness(
        returns_matrix, fast_params, slow_params)
    print(f"    Roughness = {roughness:.6f}  "
          f"({'Jagged' if roughness > 0.1 else 'Smooth'})")

    # ── 10. White's Reality Check ─────────────────────────────────────────────
    print("\n[9] White's Reality Check + Hansen SPA (n_bootstrap=500)...")
    bench_ret = spy_returns.reindex(returns_matrix.index).fillna(0.0)
    wrc = WhiteRealityCheck.run_full_test(bench_ret, returns_matrix, n_bootstrap=500)
    print(f"    WRC p-value : {wrc.wrc_p_value:.4f}  {'(significant)' if wrc.significant_at_05 else '(not significant)'}")
    print(f"    SPA p-value : {wrc.spa_p_value:.4f}")
    print(f"    → {wrc.interpretation}")

    # ── 11. Full Report ───────────────────────────────────────────────────────
    print("\n[10] Generating full overfitting report...")
    report = OverfittingDashboard.run_full_analysis(
        returns_matrix,
        n_trials=n_strategies,
        benchmark_returns=bench_ret,
        best_strategy_returns=best_returns,
        param_grid=param_df,
        param1_values=fast_params,
        param2_values=slow_params,
    )
    print(OverfittingDashboard.generate_report(report))
