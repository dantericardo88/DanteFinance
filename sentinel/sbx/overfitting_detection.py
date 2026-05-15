"""
Overfitting detection for trading strategies.
Implements: Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO),
Combinatorial Purged Cross-Validation (CPCV), and Bailey-LdP minimum track record length.

dim_064 — Overfitting detection (DSR + PBO) (target: 9)

Mathematical references:
- DSR: Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
- PBO: Bailey et al (2014) "The Probability of Backtest Overfitting"
- CPCV: Lopez de Prado (2018) "Advances in Financial Machine Learning" Chapter 12
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb as scipy_comb
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ANNUAL_FACTOR = 252  # trading days per year

# ---------------------------------------------------------------------------
# Low-level statistical utilities
# ---------------------------------------------------------------------------


def _annualized_sharpe(returns: np.ndarray, freq: int = ANNUAL_FACTOR) -> float:
    """Standard annualized Sharpe ratio (rf = 0)."""
    mu = np.nanmean(returns)
    sigma = np.nanstd(returns, ddof=1)
    if sigma == 0 or not np.isfinite(sigma):
        return 0.0
    return float(mu / sigma * np.sqrt(freq))


def _higher_moment_sharpe(returns: np.ndarray, freq: int = ANNUAL_FACTOR) -> float:
    """
    Annualized Sharpe ratio with higher-moment correction (Bailey & LdP 2014).

    SR_hat = SR × (1 - gamma3/6 × SR + (gamma4 - 3)/24 × SR²)

    where gamma3 = skewness, gamma4 = excess kurtosis + 3.

    This corrects for the upward bias in Sharpe when returns are non-normal.
    """
    r = returns[np.isfinite(returns)]
    if len(r) < 4:
        return _annualized_sharpe(r, freq)

    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    if sigma == 0:
        return 0.0

    sr_daily = mu / sigma
    gamma3 = float(stats.skew(r))           # skewness
    gamma4 = float(stats.kurtosis(r)) + 3   # kurtosis (Fisher → Pearson)

    # Higher-moment correction factor
    hm_correction = 1 - (gamma3 / 6) * sr_daily + ((gamma4 - 3) / 24) * sr_daily ** 2
    sr_hat_daily = sr_daily * hm_correction
    return float(sr_hat_daily * np.sqrt(freq))


def _sharpe_distribution_variance(returns: np.ndarray, freq: int = ANNUAL_FACTOR) -> float:
    """
    Variance of the Sharpe ratio estimator (Lo 2002).
    Var(SR_hat) ≈ (1 + SR²/2 * (kurtosis - 1) - skew * SR) / T
    Annualized equivalent.
    """
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 4:
        return 1.0

    sr = _annualized_sharpe(r, freq)
    gamma3 = float(stats.skew(r))
    gamma4 = float(stats.kurtosis(r)) + 3  # Pearson kurtosis

    var_sr = (1 + 0.5 * sr ** 2 * (gamma4 - 1) - gamma3 * sr) / n
    return max(float(var_sr), 1e-12)


# ---------------------------------------------------------------------------
# DeflatedSharpeRatio
# ---------------------------------------------------------------------------


class DeflatedSharpeRatio:
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    The DSR adjusts the observed Sharpe ratio for:
    1. Selection bias (testing N strategies, picking the best)
    2. Non-normal returns (skewness, kurtosis)
    3. Finite sample size

    DSR > 0 → strategy likely has genuine alpha.
    DSR < 0 → Sharpe ratio likely an artifact of selection bias.
    """

    def compute_sr_hat(
        self, returns: np.ndarray, freq: int = ANNUAL_FACTOR
    ) -> float:
        """Annualized Sharpe with higher-moment correction."""
        return _higher_moment_sharpe(returns, freq)

    def compute_expected_max_sr(
        self, n_trials: int, sr_mean: float = 0.0, sr_std: float = 1.0
    ) -> float:
        """
        Expected maximum Sharpe ratio over n_trials independent trials
        (Bailey & LdP eq. 2).

        E[max(SR₁..SRₙ)] ≈ (1 - γ) × Φ⁻¹(1 - 1/n) + γ × Φ⁻¹(1 - 1/(n×e))

        where γ is the Euler-Mascheroni constant, Φ⁻¹ is the normal quantile.
        """
        if n_trials <= 0:
            return 0.0
        gamma_em = 0.5772156649  # Euler-Mascheroni constant
        z1 = stats.norm.ppf(1 - 1.0 / max(n_trials, 1))
        z2 = stats.norm.ppf(1 - 1.0 / (max(n_trials, 1) * math.e))
        e_max_sr = ((1 - gamma_em) * z1 + gamma_em * z2) * sr_std + sr_mean
        return float(e_max_sr)

    def compute_dsr(
        self,
        returns: np.ndarray,
        n_trials: int,
        benchmark_returns: Optional[np.ndarray] = None,
        freq: int = ANNUAL_FACTOR,
    ) -> Dict[str, float]:
        """
        Compute the Deflated Sharpe Ratio.

        Steps:
        1. Compute SR* (corrected Sharpe of tested strategy)
        2. Compute E[max SR] across n_trials
        3. DSR = Φ(SR* − E[max SR]) / √Var(SR*)

        Returns dict with DSR, SR*, E[max SR], p-value, is_significant.
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 10:
            return {"dsr": 0.0, "sr_hat": 0.0, "e_max_sr": 0.0, "p_value": 1.0, "is_significant": False}

        # Strategy returns adjusted for benchmark
        if benchmark_returns is not None:
            b = np.asarray(benchmark_returns, dtype=float)
            b = b[np.isfinite(b)]
            min_len = min(len(r), len(b))
            r = r[:min_len] - b[:min_len]

        sr_hat = self.compute_sr_hat(r, freq)
        sr_std_est = math.sqrt(_sharpe_distribution_variance(r, freq))

        # Expected max SR across n_trials
        e_max_sr = self.compute_expected_max_sr(n_trials, sr_mean=0.0, sr_std=sr_std_est)

        # DSR: z-score of (SR* - E[max SR]) normalized by SR std
        if sr_std_est == 0:
            dsr = 0.0
            p_value = 1.0
        else:
            z = (sr_hat - e_max_sr) / sr_std_est
            dsr = float(z)
            p_value = float(1 - stats.norm.cdf(z))

        return {
            "dsr": dsr,
            "sr_hat": sr_hat,
            "e_max_sr": e_max_sr,
            "sr_std": sr_std_est,
            "p_value": p_value,
            "is_significant": dsr > 0 and p_value < 0.05,
            "n_trials": n_trials,
            "n_obs": len(r),
        }

    def min_track_record_length(
        self,
        returns: np.ndarray,
        target_sr: float = 0.0,
        confidence: float = 0.95,
        freq: int = ANNUAL_FACTOR,
    ) -> Dict[str, float]:
        """
        Minimum Track Record Length (minTRL) — Bailey & LdP (2014).

        minTRL = 1 + (1 - skew*SR + kurt/4 * SR²) × (Zα / SR)²

        This is the minimum number of observations needed to conclude the
        strategy's SR is statistically above target_sr at given confidence.
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 4:
            return {"min_trl_days": np.inf, "min_trl_years": np.inf}

        sr_daily = np.mean(r) / np.std(r, ddof=1) if np.std(r, ddof=1) > 0 else 0.0
        gamma3 = float(stats.skew(r))
        gamma4 = float(stats.kurtosis(r))  # excess kurtosis (Fisher)

        z_alpha = float(stats.norm.ppf(confidence))

        if sr_daily == 0:
            return {"min_trl_days": np.inf, "min_trl_years": np.inf}

        # Bailey-LdP formula (daily SR units)
        higher_moment_term = 1 - gamma3 * sr_daily + (gamma4 / 4) * sr_daily ** 2
        min_trl = 1 + higher_moment_term * (z_alpha / sr_daily) ** 2

        min_trl_days = max(float(min_trl), 1.0)
        min_trl_years = min_trl_days / freq

        current_sr_ann = sr_daily * math.sqrt(freq)

        return {
            "min_trl_days": min_trl_days,
            "min_trl_years": min_trl_years,
            "current_sr_annualized": current_sr_ann,
            "current_n_obs": len(r),
            "sufficient_track_record": len(r) >= min_trl_days,
        }


# ---------------------------------------------------------------------------
# ProbabilityBacktestOverfitting
# ---------------------------------------------------------------------------


class ProbabilityBacktestOverfitting:
    """
    Bailey et al (2014) — Probability of Backtest Overfitting (PBO).

    Algorithm:
    1. Divide T time periods into S equal sub-periods.
    2. For each combination of S/2 sub-periods as IS, the other S/2 as OOS:
       a. Rank all N strategies by IS performance.
       b. Select the IS winner (rank = N).
       c. Compute the OOS rank of the IS winner (as fraction 0..1).
       d. If OOS rank < 0.5 → the IS winner performed below median OOS → overfit.
    3. PBO = fraction of combinations where IS winner was below OOS median.
    """

    def _split_into_subperiods(
        self, returns_matrix: np.ndarray, n_splits: int
    ) -> List[np.ndarray]:
        """
        Split returns_matrix (N_strategies × T) into n_splits roughly equal columns.
        Returns list of n_splits arrays, each shape (N_strategies, T_sub).
        """
        n_strats, T = returns_matrix.shape
        chunk_size = T // n_splits
        subperiods = []
        for i in range(n_splits):
            start = i * chunk_size
            end = start + chunk_size if i < n_splits - 1 else T
            subperiods.append(returns_matrix[:, start:end])
        return subperiods

    def _compute_sr_per_subperiod(self, subperiod: np.ndarray) -> np.ndarray:
        """Compute annualized Sharpe for each strategy in a subperiod."""
        mu = np.nanmean(subperiod, axis=1)
        sigma = np.nanstd(subperiod, axis=1, ddof=1)
        sigma[sigma == 0] = np.nan
        sr = mu / sigma * np.sqrt(ANNUAL_FACTOR)
        return np.nan_to_num(sr, nan=0.0)

    def compute_pbo(
        self,
        returns_matrix: np.ndarray,
        n_splits: int = 16,
    ) -> Dict[str, Any]:
        """
        Compute PBO.

        returns_matrix: shape (N_strategies, T_periods)
        n_splits: number of sub-periods to divide T into (must be even)

        Returns:
            pbo: float in [0, 1], probability of backtest overfitting
            is_sharpes: list of IS Sharpe for winner per combination
            oos_sharpes: list of OOS Sharpe for IS winner per combination
            oos_ranks: list of OOS rank (0..1) for IS winner
        """
        if n_splits % 2 != 0:
            n_splits += 1  # ensure even

        returns_matrix = np.asarray(returns_matrix, dtype=float)
        if returns_matrix.ndim == 1:
            returns_matrix = returns_matrix.reshape(1, -1)

        n_strats, T = returns_matrix.shape

        if n_strats < 2:
            return {
                "pbo": 0.0,
                "error": "Need at least 2 strategies for PBO computation",
                "n_combinations": 0,
            }

        if T < n_splits:
            n_splits = max(2, T // 2 * 2)

        subperiods = self._split_into_subperiods(returns_matrix, n_splits)

        half = n_splits // 2
        all_indices = list(range(n_splits))
        combo_count = 0
        overfit_count = 0

        is_sharpes_list = []
        oos_sharpes_list = []
        oos_ranks_list = []
        degradation_list = []  # IS Sharpe / OOS Sharpe ratio

        for is_indices in combinations(all_indices, half):
            oos_indices = [i for i in all_indices if i not in is_indices]

            # Concatenate IS and OOS returns
            is_data = np.concatenate([subperiods[i] for i in is_indices], axis=1)
            oos_data = np.concatenate([subperiods[i] for i in oos_indices], axis=1)

            is_sr = self._compute_sr_per_subperiod(is_data)
            oos_sr = self._compute_sr_per_subperiod(oos_data)

            # IS winner: highest IS Sharpe
            is_winner = int(np.argmax(is_sr))
            winner_is_sr = float(is_sr[is_winner])
            winner_oos_sr = float(oos_sr[is_winner])

            # OOS rank of IS winner (rank among all N strategies in OOS)
            oos_rank = float(np.sum(oos_sr < winner_oos_sr) / (n_strats - 1))

            is_sharpes_list.append(winner_is_sr)
            oos_sharpes_list.append(winner_oos_sr)
            oos_ranks_list.append(oos_rank)

            if winner_is_sr != 0:
                degradation_list.append(winner_oos_sr / winner_is_sr)

            if oos_rank < 0.5:
                overfit_count += 1
            combo_count += 1

        pbo = float(overfit_count / combo_count) if combo_count > 0 else 0.0
        is_arr = np.array(is_sharpes_list)
        oos_arr = np.array(oos_sharpes_list)
        oos_rank_arr = np.array(oos_ranks_list)
        deg_arr = np.array(degradation_list) if degradation_list else np.array([1.0])

        return {
            "pbo": pbo,
            "n_combinations": combo_count,
            "n_overfit": overfit_count,
            "is_sharpe_mean": float(np.mean(is_arr)),
            "is_sharpe_std": float(np.std(is_arr)),
            "oos_sharpe_mean": float(np.mean(oos_arr)),
            "oos_sharpe_p5": float(np.percentile(oos_arr, 5)),
            "oos_sharpe_p25": float(np.percentile(oos_arr, 25)),
            "oos_sharpe_median": float(np.median(oos_arr)),
            "oos_rank_mean": float(np.mean(oos_rank_arr)),
            "oos_rank_std": float(np.std(oos_rank_arr)),
            "is_oos_degradation_median": float(np.median(deg_arr)),
            "is_sharpes": is_sharpes_list,
            "oos_sharpes": oos_sharpes_list,
            "oos_ranks": oos_ranks_list,
            "interpretation": _interpret_pbo(pbo),
        }

    def performance_degradation(
        self, returns_matrix: np.ndarray, n_splits: int = 8
    ) -> Dict[str, float]:
        """
        Summarize IS vs OOS Sharpe degradation across strategies.
        Uses simple even split (first half = IS, second half = OOS).
        """
        returns_matrix = np.asarray(returns_matrix, dtype=float)
        n_strats, T = returns_matrix.shape
        mid = T // 2

        is_data = returns_matrix[:, :mid]
        oos_data = returns_matrix[:, mid:]

        is_sr = np.array([_annualized_sharpe(r) for r in is_data])
        oos_sr = np.array([_annualized_sharpe(r) for r in oos_data])

        degradation = oos_sr - is_sr
        ratio = np.where(is_sr != 0, oos_sr / is_sr, 0.0)

        return {
            "is_sr_mean": float(np.mean(is_sr)),
            "oos_sr_mean": float(np.mean(oos_sr)),
            "mean_degradation": float(np.mean(degradation)),
            "pct_positive_oos": float(np.mean(oos_sr > 0)),
            "is_oos_ratio_mean": float(np.mean(ratio)),
            "is_oos_ratio_median": float(np.median(ratio)),
        }


def _interpret_pbo(pbo: float) -> str:
    if pbo < 0.1:
        return "LOW overfitting risk (PBO < 10%)"
    elif pbo < 0.25:
        return "MODERATE overfitting risk (PBO 10-25%)"
    elif pbo < 0.5:
        return "HIGH overfitting risk (PBO 25-50%)"
    else:
        return "SEVERE overfitting risk (PBO >= 50%)"


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV
# ---------------------------------------------------------------------------


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV) — Lopez de Prado (2018).

    Unlike k-fold CV, CPCV creates C(k, k/2) unique test paths,
    giving a better OOS distribution estimate.
    Purging prevents look-ahead bias from overlapping label horizons.
    Embargo prevents leakage from autocorrelation.
    """

    def create_splits(
        self,
        n_obs: int,
        n_splits: int = 6,
        purge_days: int = 1,
        embargo_pct: float = 0.01,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create CPCV splits: C(n_splits, n_splits//2) (train, test) index pairs.

        purge_days: observations adjacent to test boundaries removed from train.
        embargo_pct: fraction of observations embargoed after training ends.

        Returns list of (train_indices, test_indices).
        """
        embargo_days = max(1, int(n_obs * embargo_pct))

        # Divide observations into n_splits groups
        groups = np.array_split(np.arange(n_obs), n_splits)
        group_sizes = [len(g) for g in groups]

        splits = []
        test_k = max(1, n_splits // 2)

        for test_group_indices in combinations(range(n_splits), test_k):
            test_set = set(test_group_indices)
            train_group_indices = [i for i in range(n_splits) if i not in test_set]

            # Gather raw test and train indices
            test_idx = np.concatenate([groups[i] for i in test_group_indices])
            train_idx_raw = np.concatenate([groups[i] for i in train_group_indices])

            # Purge: remove train obs within purge_days of test boundaries
            test_start = int(test_idx.min())
            test_end = int(test_idx.max())

            purge_mask = (
                (train_idx_raw >= test_start - purge_days) &
                (train_idx_raw <= test_end + purge_days)
            )
            train_idx = train_idx_raw[~purge_mask]

            # Embargo: remove train obs immediately after the max train observation
            if len(train_idx) > 0:
                max_train = int(train_idx.max())
                embargo_mask = (
                    (train_idx > max_train - embargo_days) &
                    (train_idx >= test_start)
                )
                train_idx = train_idx[~embargo_mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((np.sort(train_idx), np.sort(test_idx)))

        return splits

    def cpcv_backtest(
        self,
        strategy_func: Callable[[np.ndarray, np.ndarray], float],
        data: np.ndarray,
        n_splits: int = 6,
        purge_days: int = 1,
        embargo_pct: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Run CPCV backtest.

        strategy_func(train_data, test_data) → OOS Sharpe ratio (float)
        data: 1D array of returns (single strategy)

        Returns distribution of OOS Sharpe ratios across all CPCV paths.
        """
        data = np.asarray(data, dtype=float)
        n_obs = len(data)

        splits = self.create_splits(n_obs, n_splits, purge_days, embargo_pct)

        oos_sharpes = []
        for train_idx, test_idx in splits:
            try:
                train_data = data[train_idx]
                test_data = data[test_idx]
                sr = strategy_func(train_data, test_data)
                oos_sharpes.append(float(sr))
            except Exception as exc:
                logger.warning("CPCV fold failed: %s", exc)

        if not oos_sharpes:
            return {"error": "All CPCV folds failed"}

        arr = np.array(oos_sharpes)
        return {
            "n_paths": len(oos_sharpes),
            "oos_sharpe_mean": float(np.mean(arr)),
            "oos_sharpe_median": float(np.median(arr)),
            "oos_sharpe_std": float(np.std(arr)),
            "oos_sharpe_p5": float(np.percentile(arr, 5)),
            "oos_sharpe_p25": float(np.percentile(arr, 25)),
            "oos_sharpe_p75": float(np.percentile(arr, 75)),
            "pct_positive": float(np.mean(arr > 0)),
            "oos_sharpes": arr.tolist(),
        }

    def n_unique_paths(self, n_splits: int) -> int:
        """Number of unique CPCV test paths: C(n_splits, n_splits//2)."""
        return int(scipy_comb(n_splits, n_splits // 2, exact=True))


# ---------------------------------------------------------------------------
# OverfittingMetrics & scorecard
# ---------------------------------------------------------------------------


@dataclass
class OverfittingMetrics:
    """Comprehensive overfitting scorecard for a single strategy."""
    dsr: float = 0.0
    dsr_pvalue: float = 1.0
    sr_hat: float = 0.0
    e_max_sr: float = 0.0

    pbo: float = 0.0
    oos_sharpe_median: float = 0.0
    oos_sharpe_p5: float = 0.0
    oos_sharpe_p25: float = 0.0
    is_oos_sr_ratio: float = 0.0

    min_trl_days: float = np.inf
    sufficient_track_record: bool = False

    parameter_sensitivity: float = 0.0   # std of Sharpe across param perturbations
    deflation_factor: float = 1.0        # SR_OOS / SR_IS

    n_trials: int = 1
    n_obs: int = 0

    traffic_light: str = "RED"
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dsr": self.dsr,
            "dsr_pvalue": self.dsr_pvalue,
            "sr_hat": self.sr_hat,
            "e_max_sr": self.e_max_sr,
            "pbo": self.pbo,
            "oos_sharpe_median": self.oos_sharpe_median,
            "oos_sharpe_p5": self.oos_sharpe_p5,
            "oos_sharpe_p25": self.oos_sharpe_p25,
            "is_oos_sr_ratio": self.is_oos_sr_ratio,
            "min_trl_days": float(self.min_trl_days) if np.isfinite(self.min_trl_days) else -1,
            "sufficient_track_record": self.sufficient_track_record,
            "parameter_sensitivity": self.parameter_sensitivity,
            "deflation_factor": self.deflation_factor,
            "n_trials": self.n_trials,
            "n_obs": self.n_obs,
            "traffic_light": self.traffic_light,
            "explanation": self.explanation,
        }


def _traffic_light(dsr: float, pbo: float, dsr_pvalue: float) -> Tuple[str, str]:
    """
    Classify strategy overfitting risk:
    GREEN: DSR > 0.5, PBO < 0.10, p-value < 0.05
    YELLOW: DSR > 0 or PBO < 0.25 (caution)
    RED: DSR <= 0 or PBO >= 0.25
    """
    if dsr > 0.5 and pbo < 0.10 and dsr_pvalue < 0.05:
        color = "GREEN"
        msg = (
            f"Strategy appears genuine. DSR={dsr:.2f} (>0.5), "
            f"PBO={pbo:.1%} (<10%), p={dsr_pvalue:.3f}."
        )
    elif dsr > 0 and pbo < 0.25:
        color = "YELLOW"
        msg = (
            f"Caution. DSR={dsr:.2f} (positive but moderate), "
            f"PBO={pbo:.1%} (<25%). Gather more data or reduce strategy count."
        )
    else:
        color = "RED"
        msg = (
            f"Likely overfit. DSR={dsr:.2f} (<=0 or low), "
            f"PBO={pbo:.1%} (>=25%). Strategy may not survive OOS."
        )
    return color, msg


# ---------------------------------------------------------------------------
# StrategyValidator
# ---------------------------------------------------------------------------


class StrategyValidator:
    """
    Full overfitting validation pipeline combining DSR, PBO, CPCV,
    and parameter stability checks.
    """

    def __init__(self):
        self._dsr_engine = DeflatedSharpeRatio()
        self._pbo_engine = ProbabilityBacktestOverfitting()
        self._cpcv_engine = CombinatorialPurgedCV()

    def _parameter_sensitivity(
        self,
        returns: np.ndarray,
        params_dict: Dict[str, Any],
        perturb_pct: float = 0.20,
        n_perturb: int = 20,
    ) -> float:
        """
        Estimate parameter sensitivity: std of Sharpe ratio under
        ±perturb_pct random perturbations of parameters.

        Since we don't re-run the strategy here (no strategy function passed),
        we approximate by bootstrap subsampling: draw n_perturb subsamples of
        size (1 ± perturb_pct) × T and compute SR distribution.
        """
        r = returns[np.isfinite(returns)]
        n = len(r)
        if n < 20:
            return 0.0

        sharpes = []
        for _ in range(n_perturb):
            # perturb window size
            scale = 1 + np.random.uniform(-perturb_pct, perturb_pct)
            subset_size = max(10, int(n * scale))
            subset_size = min(subset_size, n)
            start = np.random.randint(0, n - subset_size + 1)
            subset = r[start: start + subset_size]
            sharpes.append(_annualized_sharpe(subset))

        return float(np.std(sharpes))

    def validate(
        self,
        strategy_returns: np.ndarray,
        n_strategies_tried: int = 1,
        params_dict: Optional[Dict[str, Any]] = None,
        all_strategy_returns: Optional[np.ndarray] = None,
        freq: int = ANNUAL_FACTOR,
    ) -> OverfittingMetrics:
        """
        Run full validation pipeline.

        strategy_returns: 1D array of daily returns for the selected strategy
        n_strategies_tried: how many strategies/params were tested before selecting this one
        params_dict: parameter dictionary (used for sensitivity labels)
        all_strategy_returns: (N_strategies × T) matrix for PBO computation
        """
        returns = np.asarray(strategy_returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        n_obs = len(returns)

        if n_obs < 10:
            return OverfittingMetrics(
                traffic_light="RED",
                explanation="Insufficient data (< 10 observations).",
                n_obs=n_obs,
            )

        # --- DSR ---
        dsr_result = self._dsr_engine.compute_dsr(returns, n_strategies_tried, freq=freq)
        trl_result = self._dsr_engine.min_track_record_length(returns, freq=freq)

        # --- PBO ---
        if all_strategy_returns is not None:
            matrix = np.asarray(all_strategy_returns, dtype=float)
            # Ensure the selected strategy is in the matrix
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            pbo_result = self._pbo_engine.compute_pbo(matrix)
        elif n_strategies_tried > 1:
            # Simulate: create random null strategies (returns = white noise)
            T = len(returns)
            null_strats = np.random.normal(
                np.mean(returns), np.std(returns), (n_strategies_tried - 1, T)
            )
            matrix = np.vstack([returns.reshape(1, -1), null_strats])
            pbo_result = self._pbo_engine.compute_pbo(matrix)
        else:
            # Single strategy: PBO not applicable
            pbo_result = {
                "pbo": 0.0,
                "oos_sharpe_median": float(_annualized_sharpe(returns[len(returns) // 2:])),
                "oos_sharpe_p5": 0.0,
                "oos_sharpe_p25": 0.0,
            }

        # IS Sharpe vs OOS Sharpe
        mid = len(returns) // 2
        is_sr = _annualized_sharpe(returns[:mid]) if mid > 0 else 0.0
        oos_sr = _annualized_sharpe(returns[mid:]) if mid < len(returns) else 0.0
        deflation_factor = float(oos_sr / is_sr) if is_sr != 0 else 0.0
        is_oos_ratio = float(oos_sr / is_sr) if is_sr > 0 else (1.0 if is_sr == 0 else -1.0)

        # Parameter sensitivity
        param_sensitivity = self._parameter_sensitivity(
            returns, params_dict or {}, n_perturb=30
        )

        # Traffic light
        dsr = float(dsr_result.get("dsr", 0.0))
        dsr_pvalue = float(dsr_result.get("p_value", 1.0))
        pbo = float(pbo_result.get("pbo", 0.0))
        tl, explanation = _traffic_light(dsr, pbo, dsr_pvalue)

        metrics = OverfittingMetrics(
            dsr=dsr,
            dsr_pvalue=dsr_pvalue,
            sr_hat=float(dsr_result.get("sr_hat", 0.0)),
            e_max_sr=float(dsr_result.get("e_max_sr", 0.0)),
            pbo=pbo,
            oos_sharpe_median=float(pbo_result.get("oos_sharpe_median", oos_sr)),
            oos_sharpe_p5=float(pbo_result.get("oos_sharpe_p5", 0.0)),
            oos_sharpe_p25=float(pbo_result.get("oos_sharpe_p25", 0.0)),
            is_oos_sr_ratio=is_oos_ratio,
            min_trl_days=float(trl_result.get("min_trl_days", np.inf)),
            sufficient_track_record=bool(trl_result.get("sufficient_track_record", False)),
            parameter_sensitivity=param_sensitivity,
            deflation_factor=deflation_factor,
            n_trials=n_strategies_tried,
            n_obs=n_obs,
            traffic_light=tl,
            explanation=explanation,
        )
        return metrics


# ---------------------------------------------------------------------------
# MultipleTestingCorrection
# ---------------------------------------------------------------------------


class MultipleTestingCorrection:
    """
    Adjust p-values for multiple strategy testing.
    Bonferroni: conservative, controls FWER.
    Benjamini-Hochberg-Yekutieli (BHY): controls FDR, less conservative.
    """

    @staticmethod
    def bonferroni(pvalues: np.ndarray) -> np.ndarray:
        """
        Bonferroni correction: p_adj = min(p * N, 1.0)
        """
        pvalues = np.asarray(pvalues, dtype=float)
        n = len(pvalues)
        return np.minimum(pvalues * n, 1.0)

    @staticmethod
    def benjamini_hochberg(pvalues: np.ndarray, fdr: float = 0.05) -> np.ndarray:
        """
        Benjamini-Hochberg correction for independent tests.
        Reject H0 for all tests where p_adj ≤ fdr.
        Returns adjusted p-values.
        """
        pvalues = np.asarray(pvalues, dtype=float)
        n = len(pvalues)
        if n == 0:
            return pvalues

        sorted_idx = np.argsort(pvalues)
        sorted_p = pvalues[sorted_idx]
        ranks = np.arange(1, n + 1)

        # BH adjusted p-values: p_adj[i] = min over j≥i of (n/j) * p[j]
        adj = np.minimum.accumulate((n / ranks * sorted_p)[::-1])[::-1]
        adj = np.minimum(adj, 1.0)

        result = np.empty_like(pvalues)
        result[sorted_idx] = adj
        return result

    @staticmethod
    def benjamini_hochberg_yekutieli(
        pvalues: np.ndarray, fdr: float = 0.05
    ) -> np.ndarray:
        """
        BHY correction — handles arbitrary dependence structure
        (more conservative than BH but less than Bonferroni).
        Multiplies BH critical values by c_n = sum(1/k, k=1..n).
        """
        pvalues = np.asarray(pvalues, dtype=float)
        n = len(pvalues)
        if n == 0:
            return pvalues

        c_n = float(np.sum(1.0 / np.arange(1, n + 1)))
        sorted_idx = np.argsort(pvalues)
        sorted_p = pvalues[sorted_idx]
        ranks = np.arange(1, n + 1)

        adj = np.minimum.accumulate(
            (n * c_n / ranks * sorted_p)[::-1]
        )[::-1]
        adj = np.minimum(adj, 1.0)

        result = np.empty_like(pvalues)
        result[sorted_idx] = adj
        return result

    def adjust_pvalues(
        self, pvalues: List[float], method: str = "bhy"
    ) -> Dict[str, Any]:
        """
        Adjust p-values for multiple testing.
        method: 'bonferroni', 'bh', 'bhy'
        Returns adjusted p-values and which strategies pass FDR 5%.
        """
        arr = np.asarray(pvalues, dtype=float)
        if method == "bonferroni":
            adj = self.bonferroni(arr)
        elif method == "bh":
            adj = self.benjamini_hochberg(arr)
        elif method == "bhy":
            adj = self.benjamini_hochberg_yekutieli(arr)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bonferroni', 'bh', or 'bhy'")

        significant = (adj < 0.05).tolist()

        return {
            "method": method,
            "original_pvalues": arr.tolist(),
            "adjusted_pvalues": adj.tolist(),
            "significant_at_5pct": significant,
            "n_significant": int(np.sum(significant)),
            "n_tested": len(arr),
            "fdr_bound": 0.05,
        }

    def minimum_backtest_sr(
        self,
        n_trials: int,
        target_sr: float = 0.0,
        confidence: float = 0.95,
    ) -> float:
        """
        Minimum Sharpe ratio that a strategy must achieve to be considered
        non-random after testing n_trials strategies.
        Uses the expected maximum of a standard normal (Bailey & LdP).
        """
        dsr = DeflatedSharpeRatio()
        return dsr.compute_expected_max_sr(n_trials) + target_sr


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Full overfitting validation report."""
    strategy_name: str
    metrics: OverfittingMetrics
    dsr_detail: Dict[str, Any] = field(default_factory=dict)
    pbo_detail: Dict[str, Any] = field(default_factory=dict)
    cpcv_detail: Dict[str, Any] = field(default_factory=dict)
    mtc_detail: Dict[str, Any] = field(default_factory=dict)
    trl_detail: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "metrics": self.metrics.to_dict(),
            "dsr_detail": self.dsr_detail,
            "pbo_detail": self.pbo_detail,
            "cpcv_detail": self.cpcv_detail,
            "mtc_detail": self.mtc_detail,
            "trl_detail": self.trl_detail,
            "recommendations": self.recommendations,
        }


def _generate_recommendations(metrics: OverfittingMetrics) -> List[str]:
    recs = []
    if metrics.dsr < 0:
        recs.append(
            f"DSR is negative ({metrics.dsr:.2f}): Sharpe ratio likely inflated by selection "
            f"bias from testing {metrics.n_trials} strategies. Reduce number of trials or "
            f"collect more independent data."
        )
    if metrics.pbo > 0.25:
        recs.append(
            f"PBO is {metrics.pbo:.1%}: High probability the IS winner is overfit. "
            "Consider reducing the strategy parameter space."
        )
    if not metrics.sufficient_track_record:
        trl = metrics.min_trl_days
        recs.append(
            f"Insufficient track record: need {trl:.0f} days, have {metrics.n_obs}. "
            "Do not deploy until more data is accumulated."
        )
    if metrics.is_oos_sr_ratio < 0.5:
        recs.append(
            f"IS/OOS SR degradation is severe (ratio={metrics.is_oos_sr_ratio:.2f}). "
            "Strategy may be over-parameterized."
        )
    if metrics.parameter_sensitivity > 0.5:
        recs.append(
            f"Parameter sensitivity is high (σ={metrics.parameter_sensitivity:.2f}). "
            "Strategy performance is fragile to parameter choices."
        )
    if not recs:
        recs.append("No major overfitting concerns detected. Continue monitoring OOS performance.")
    return recs


class FullValidator:
    """End-to-end validator with DSR, PBO, CPCV, and MTC."""

    def __init__(self):
        self._dsr = DeflatedSharpeRatio()
        self._pbo = ProbabilityBacktestOverfitting()
        self._cpcv = CombinatorialPurgedCV()
        self._mtc = MultipleTestingCorrection()
        self._validator = StrategyValidator()

    def full_validate(
        self,
        strategy_returns: np.ndarray,
        strategy_name: str = "strategy",
        n_strategies_tried: int = 1,
        all_returns_matrix: Optional[np.ndarray] = None,
        params_dict: Optional[Dict[str, Any]] = None,
        pvalues: Optional[List[float]] = None,
    ) -> ValidationReport:
        """
        Run complete validation suite and produce a ValidationReport.
        """
        r = np.asarray(strategy_returns, dtype=float)
        r = r[np.isfinite(r)]

        # DSR
        dsr_detail = self._dsr.compute_dsr(r, n_strategies_tried)
        trl_detail = self._dsr.min_track_record_length(r)

        # PBO
        if all_returns_matrix is not None:
            mat = np.asarray(all_returns_matrix, dtype=float)
        elif n_strategies_tried > 1:
            T = len(r)
            null = np.random.normal(np.mean(r), np.std(r), (n_strategies_tried - 1, T))
            mat = np.vstack([r.reshape(1, -1), null])
        else:
            mat = r.reshape(1, -1)

        pbo_detail = self._pbo.compute_pbo(mat)

        # CPCV (simple Sharpe as strategy function)
        def _sr_func(train: np.ndarray, test: np.ndarray) -> float:
            return _annualized_sharpe(test)

        cpcv_detail = self._cpcv.cpcv_backtest(_sr_func, r, n_splits=6)

        # Multiple testing correction
        if pvalues is not None:
            mtc_detail = self._mtc.adjust_pvalues(pvalues, method="bhy")
        else:
            p = float(dsr_detail.get("p_value", 1.0))
            mtc_detail = self._mtc.adjust_pvalues([p] * n_strategies_tried, method="bhy")

        # Core metrics
        metrics = self._validator.validate(
            r, n_strategies_tried, params_dict, all_returns_matrix
        )

        recommendations = _generate_recommendations(metrics)

        return ValidationReport(
            strategy_name=strategy_name,
            metrics=metrics,
            dsr_detail=dsr_detail,
            pbo_detail=pbo_detail,
            cpcv_detail=cpcv_detail,
            mtc_detail=mtc_detail,
            trl_detail=trl_detail,
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

overfitting_router = APIRouter(prefix="/overfitting", tags=["overfitting"])

_dsr_engine = DeflatedSharpeRatio()
_pbo_engine = ProbabilityBacktestOverfitting()
_cpcv_engine = CombinatorialPurgedCV()
_mtc_engine = MultipleTestingCorrection()
_full_validator = FullValidator()


class DSRRequest(BaseModel):
    returns: List[float] = Field(..., description="Daily returns of the strategy")
    n_trials: int = Field(1, ge=1, description="Number of strategies tested")
    benchmark_returns: Optional[List[float]] = Field(None)
    freq: int = Field(252, ge=1, le=252)


class PBORequest(BaseModel):
    returns_matrix: List[List[float]] = Field(
        ..., description="N_strategies × T matrix of daily returns"
    )
    n_splits: int = Field(16, ge=4, le=64)


class ValidateRequest(BaseModel):
    returns: List[float]
    n_strategies_tried: int = Field(1, ge=1)
    strategy_name: str = "strategy"
    all_returns_matrix: Optional[List[List[float]]] = None
    params_dict: Optional[Dict[str, Any]] = None


class CPCVRequest(BaseModel):
    returns: List[float]
    n_splits: int = Field(6, ge=4, le=20)
    purge_days: int = Field(1, ge=0, le=10)
    embargo_pct: float = Field(0.01, ge=0.0, le=0.1)


class MultipleTesting(BaseModel):
    pvalues: List[float]
    method: str = Field("bhy", description="bonferroni|bh|bhy")


@overfitting_router.post("/dsr")
async def deflated_sharpe_ratio(req: DSRRequest) -> Dict[str, Any]:
    """Compute Deflated Sharpe Ratio for a strategy."""
    try:
        r = np.array(req.returns)
        bench = np.array(req.benchmark_returns) if req.benchmark_returns else None
        dsr = _dsr_engine.compute_dsr(r, req.n_trials, bench, req.freq)
        trl = _dsr_engine.min_track_record_length(r, freq=req.freq)
        sr_hat = _dsr_engine.compute_sr_hat(r, req.freq)
        return {
            "dsr": dsr,
            "min_track_record_length": trl,
            "sr_hat_annualized": sr_hat,
        }
    except Exception as exc:
        logger.exception("DSR error")
        raise HTTPException(status_code=500, detail=str(exc))


@overfitting_router.post("/pbo")
async def probability_backtest_overfitting(req: PBORequest) -> Dict[str, Any]:
    """Compute Probability of Backtest Overfitting."""
    try:
        matrix = np.array(req.returns_matrix)
        result = _pbo_engine.compute_pbo(matrix, req.n_splits)
        degradation = _pbo_engine.performance_degradation(matrix)
        return {"pbo": result, "performance_degradation": degradation}
    except Exception as exc:
        logger.exception("PBO error")
        raise HTTPException(status_code=500, detail=str(exc))


@overfitting_router.post("/validate")
async def validate_strategy(req: ValidateRequest) -> Dict[str, Any]:
    """Full overfitting validation: DSR + PBO + CPCV + recommendations."""
    try:
        r = np.array(req.returns)
        mat = np.array(req.all_returns_matrix) if req.all_returns_matrix else None
        report = _full_validator.full_validate(
            r,
            strategy_name=req.strategy_name,
            n_strategies_tried=req.n_strategies_tried,
            all_returns_matrix=mat,
            params_dict=req.params_dict,
        )
        return report.to_dict()
    except Exception as exc:
        logger.exception("Validate error")
        raise HTTPException(status_code=500, detail=str(exc))


@overfitting_router.post("/cpcv")
async def cpcv_analysis(req: CPCVRequest) -> Dict[str, Any]:
    """
    Combinatorial Purged Cross-Validation.
    Returns OOS Sharpe distribution across all CPCV paths.
    """
    try:
        r = np.array(req.returns)

        def _sr_func(train: np.ndarray, test: np.ndarray) -> float:
            return _annualized_sharpe(test)

        result = _cpcv_engine.cpcv_backtest(
            _sr_func, r, req.n_splits, req.purge_days, req.embargo_pct
        )
        splits = _cpcv_engine.create_splits(
            len(r), req.n_splits, req.purge_days, req.embargo_pct
        )
        result["n_unique_paths"] = _cpcv_engine.n_unique_paths(req.n_splits)
        result["n_actual_splits"] = len(splits)
        return result
    except Exception as exc:
        logger.exception("CPCV error")
        raise HTTPException(status_code=500, detail=str(exc))


@overfitting_router.post("/multiple-testing")
async def multiple_testing_correction(req: MultipleTesting) -> Dict[str, Any]:
    """Adjust p-values for multiple strategy testing (Bonferroni / BH / BHY)."""
    try:
        return _mtc_engine.adjust_pvalues(req.pvalues, req.method)
    except Exception as exc:
        logger.exception("MTC error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------


def haircut_sharpe(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    freq: int = ANNUAL_FACTOR,
) -> Dict[str, float]:
    """
    Bailey & LdP (2014) Haircut Sharpe: expected discount due to selection bias.

    Returns the expected max SR under H0 (all strategies are noise),
    and the haircut = max_sr / observed_sr.
    """
    dsr = DeflatedSharpeRatio()
    sr_var = (1 + 0.5 * observed_sr ** 2 * (kurtosis - 1) - skewness * observed_sr) / n_obs
    sr_std = math.sqrt(max(sr_var, 1e-12))
    e_max_sr = dsr.compute_expected_max_sr(n_trials, sr_mean=0.0, sr_std=sr_std)
    haircut = float(e_max_sr / observed_sr) if observed_sr != 0 else 0.0
    adjusted_sr = float(observed_sr - e_max_sr)

    return {
        "observed_sr": observed_sr,
        "expected_max_sr_under_h0": e_max_sr,
        "adjusted_sr": adjusted_sr,
        "haircut_fraction": haircut,
        "n_trials": n_trials,
    }


def sharpe_ratio_test(
    returns: np.ndarray,
    target_sr: float = 0.0,
    freq: int = ANNUAL_FACTOR,
    use_higher_moments: bool = True,
) -> Dict[str, float]:
    """
    One-sided t-test: H0: SR ≤ target_sr vs H1: SR > target_sr.
    Uses Jobson-Korkie statistic with Memmel (2003) correction.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4:
        return {"sr": 0.0, "t_stat": 0.0, "p_value": 1.0}

    mu = np.mean(r)
    sigma = np.std(r, ddof=1)

    if use_higher_moments:
        sr = _higher_moment_sharpe(r, freq)
    else:
        sr = mu / sigma * math.sqrt(freq) if sigma > 0 else 0.0

    gamma3 = float(stats.skew(r))
    gamma4 = float(stats.kurtosis(r)) + 3  # Pearson kurtosis

    # Asymptotic variance of annualized SR (Jobson-Korkie / Memmel)
    var_sr = (1 + 0.5 * (mu / sigma) ** 2 * (gamma4 - 1) - gamma3 * (mu / sigma)) * freq / n
    se_sr = math.sqrt(max(var_sr, 1e-12))

    t_stat = float((sr - target_sr) / se_sr)
    p_value = float(1 - stats.norm.cdf(t_stat))

    return {
        "sr_annualized": sr,
        "target_sr": target_sr,
        "t_stat": t_stat,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "n_obs": n,
        "se_sr": se_sr,
    }


def expected_max_sr_table(
    max_trials: int = 100,
    freq: int = ANNUAL_FACTOR,
) -> pd.DataFrame:
    """
    Build a reference table: for N trials, what Sharpe must a strategy achieve
    to be statistically credible (after selection bias)?

    Returns DataFrame with columns: n_trials, min_required_sr, e_max_sr.
    """
    dsr = DeflatedSharpeRatio()
    rows = []
    for n in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        if n > max_trials:
            break
        e_max = dsr.compute_expected_max_sr(n)
        rows.append(
            {"n_trials": n, "e_max_sr": round(e_max, 3), "min_required_sr": round(e_max + 0.5, 3)}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-Forward Overfitting Score
# ---------------------------------------------------------------------------


class WalkForwardOverfittingScore:
    """
    Compute an overfitting score from walk-forward backtest results.
    IS / OOS ratio across multiple windows = overfitting indicator.
    """

    def score(
        self,
        is_sharpes: List[float],
        oos_sharpes: List[float],
    ) -> Dict[str, float]:
        """
        Given per-fold IS and OOS Sharpe ratios, compute:
        - Mean IS/OOS ratio
        - Fraction of folds where OOS > 0
        - Fraction of folds where OOS > IS (lucky OOS)
        """
        if len(is_sharpes) != len(oos_sharpes) or len(is_sharpes) == 0:
            return {"error": "is_sharpes and oos_sharpes must have same non-zero length"}

        is_arr = np.array(is_sharpes)
        oos_arr = np.array(oos_sharpes)

        ratio = np.where(is_arr != 0, oos_arr / is_arr, np.nan)

        return {
            "mean_is_oos_ratio": float(np.nanmean(ratio)),
            "median_is_oos_ratio": float(np.nanmedian(ratio)),
            "pct_oos_positive": float(np.mean(oos_arr > 0)),
            "pct_oos_beats_is": float(np.mean(oos_arr > is_arr)),
            "is_sr_mean": float(np.mean(is_arr)),
            "oos_sr_mean": float(np.mean(oos_arr)),
            "is_sr_std": float(np.std(is_arr)),
            "oos_sr_std": float(np.std(oos_arr)),
            "n_folds": len(is_arr),
            "overfitting_risk": "HIGH" if np.nanmean(ratio) < 0.5 else (
                "MODERATE" if np.nanmean(ratio) < 0.75 else "LOW"
            ),
        }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


__all__ = [
    "DeflatedSharpeRatio",
    "ProbabilityBacktestOverfitting",
    "CombinatorialPurgedCV",
    "OverfittingMetrics",
    "StrategyValidator",
    "MultipleTestingCorrection",
    "FullValidator",
    "ValidationReport",
    "WalkForwardOverfittingScore",
    "overfitting_router",
    "haircut_sharpe",
    "sharpe_ratio_test",
    "expected_max_sr_table",
    "_annualized_sharpe",
    "_higher_moment_sharpe",
]
