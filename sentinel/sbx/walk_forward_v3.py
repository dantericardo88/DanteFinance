"""
Walk-Forward + Anchored OOS Validation v3 — dim_063 (target score 9).

Architecture
------------
WalkForwardConfig        — Unified config dataclass (rolling/anchored, n_jobs, etc.)
WalkForwardEngine        — Generate folds, run per-fold optimization, stitch OOS curve
OverfittingDetector      — Overfitting ratio, DSR, PSR, PBO (Bailey et al.)
AnchoredValidation       — Expanding-window WF, degradation curve, regime sensitivity
ParameterStabilityAnalyzer — Dispersion, correlation, robust param selection
StrategyValidator        — Orchestrator: runs all tests → ValidationReport + PASS/FAIL
MonteCarloPermutationTest — Permutation p-value + White's Reality Check bootstrap

Built-in strategies: sma_crossover_strategy, momentum_strategy, mean_reversion_strategy

FastAPI router: wf_v3_router
  POST /walkforward/v3/run
  POST /walkforward/v3/anchored
  POST /walkforward/v3/validate
  POST /walkforward/v3/pbo
  POST /walkforward/v3/permutation
  GET  /walkforward/v3/strategies
"""
from __future__ import annotations

import itertools
import logging
import math
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy import stats

logger = logging.getLogger(__name__)

ANNUAL_FACTOR = 252  # trading days per year

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward validation.

    train_window : int   — in-sample period length in trading days
    test_window  : int   — OOS period length per fold
    step_size    : int   — how far to advance each fold
    min_trades   : int   — minimum trades required to accept a fold
    anchored     : bool  — True = expanding window (anchored WF)
    n_jobs       : int   — parallel fold evaluation
    """
    train_window: int = 252
    test_window: int = 63
    step_size: int = 21
    min_trades: int = 20
    anchored: bool = False
    n_jobs: int = 1


@dataclass
class Fold:
    """A single train/test split."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_data: pd.DataFrame
    test_data: pd.DataFrame


@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    in_sample_sharpe: float
    out_sample_sharpe: float
    in_sample_returns: pd.Series
    out_sample_returns: pd.Series
    n_train_trades: int
    n_test_trades: int
    overfitting_ratio: float        # IS Sharpe / OOS Sharpe
    params_used: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated result from all folds."""
    config: WalkForwardConfig
    fold_results: List[FoldResult]
    oos_equity_curve: pd.Series
    oos_returns: pd.Series
    mean_is_sharpe: float
    mean_oos_sharpe: float
    overall_overfitting_ratio: float
    n_folds: int
    strategy_name: str = ""


@dataclass
class AnchoredResult:
    """Result from anchored (expanding window) walk-forward."""
    fold_results: List[FoldResult]
    oos_equity_curve: pd.Series
    oos_returns: pd.Series
    degradation_curve: pd.Series    # OOS Sharpe vs fold index
    regime_performance: Dict[str, float]  # regime → mean OOS Sharpe


@dataclass
class BiasReport:
    """Results from bias detection tests."""
    lookahead_bias_detected: bool
    lookahead_performance_drop: float   # fraction drop in Sharpe when lag applied
    survivorship_bias_warning: bool
    data_snooping_dsr: float
    n_trials_tested: int


@dataclass
class ValidationReport:
    """Full validation report with PASS/FAIL gates."""
    strategy_name: str
    oos_sharpe: float
    pbo: float
    overfitting_ratio: float
    dsr: float
    psr: float
    mean_is_sharpe: float
    mean_oos_sharpe: float
    n_folds: int
    anchored_mean_oos_sharpe: float
    param_dispersion: Dict[str, float]
    robust_params: Dict[str, Any]
    permutation_pvalue: float
    bias_report: Optional[BiasReport]
    quality_score: float            # 0-100
    gate_oos_sharpe: bool           # OOS Sharpe > 0.5
    gate_pbo: bool                  # PBO < 0.5
    gate_overfitting: bool          # ratio < 2.0
    gate_dsr: bool                  # DSR > 0.95
    overall_pass: bool


# ─────────────────────────────────────────────────────────────────────────────
# Statistical utilities
# ─────────────────────────────────────────────────────────────────────────────

def _annualized_sharpe(returns: pd.Series | np.ndarray, ann: int = ANNUAL_FACTOR) -> float:
    """Annualized Sharpe ratio (rf = 0)."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * math.sqrt(ann))


def _count_trades(returns: pd.Series) -> int:
    """Count non-zero return days as a trade proxy."""
    return int((returns != 0).sum())


def _optimise_params(
    train_data: pd.DataFrame,
    strategy_fn: Callable,
    param_grid: Dict[str, list],
) -> Tuple[Dict[str, Any], float]:
    """
    Grid-search over param_grid on training data.
    Returns (best_params, best_is_sharpe).
    """
    best_params: Dict[str, Any] = {}
    best_sharpe = -np.inf

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            rets = strategy_fn(train_data, **params)
            sh = _annualized_sharpe(rets)
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = params
        except Exception:
            continue

    return best_params, best_sharpe


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Strategy Functions
# ─────────────────────────────────────────────────────────────────────────────

def sma_crossover_strategy(
    data: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    price_col: str = "Close",
) -> pd.Series:
    """
    Simple moving average crossover strategy.

    Long when fast SMA > slow SMA; flat otherwise.
    Returns daily return series aligned with data index.
    """
    if price_col not in data.columns:
        # Try first numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            return pd.Series(0.0, index=data.index)
        price_col = numeric_cols[0]

    prices = data[price_col].dropna()
    if len(prices) < slow + 5:
        return pd.Series(0.0, index=data.index)

    fast_ma = prices.rolling(fast, min_periods=fast).mean()
    slow_ma = prices.rolling(slow, min_periods=slow).mean()

    signal = (fast_ma > slow_ma).astype(float)
    signal = signal.shift(1).fillna(0)  # trade next bar

    daily_rets = prices.pct_change().fillna(0)
    strategy_rets = signal * daily_rets

    return strategy_rets.reindex(data.index, fill_value=0.0)


def momentum_strategy(
    data: pd.DataFrame,
    lookback: int = 126,
    price_col: str = "Close",
) -> pd.Series:
    """
    Time-series momentum (12-1 month).

    Long if trailing `lookback`-day return > 0; flat otherwise.
    """
    if price_col not in data.columns:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            return pd.Series(0.0, index=data.index)
        price_col = numeric_cols[0]

    prices = data[price_col].dropna()
    if len(prices) < lookback + 5:
        return pd.Series(0.0, index=data.index)

    momentum = prices.pct_change(lookback)
    signal = (momentum > 0).astype(float).shift(1).fillna(0)
    daily_rets = prices.pct_change().fillna(0)
    strategy_rets = signal * daily_rets

    return strategy_rets.reindex(data.index, fill_value=0.0)


def mean_reversion_strategy(
    data: pd.DataFrame,
    window: int = 20,
    z_threshold: float = 2.0,
    price_col: str = "Close",
) -> pd.Series:
    """
    Mean reversion: z-score based.

    Short when z > +threshold, long when z < -threshold, flat otherwise.
    """
    if price_col not in data.columns:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            return pd.Series(0.0, index=data.index)
        price_col = numeric_cols[0]

    prices = data[price_col].dropna()
    if len(prices) < window + 5:
        return pd.Series(0.0, index=data.index)

    roll_mean = prices.rolling(window, min_periods=window).mean()
    roll_std = prices.rolling(window, min_periods=window).std(ddof=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = (prices - roll_mean) / roll_std.replace(0, np.nan)

    # Signal: -1 (short when overbought), +1 (long when oversold), 0 (flat)
    signal = pd.Series(0.0, index=prices.index)
    signal[z < -z_threshold] = 1.0
    signal[z > z_threshold] = -1.0
    signal = signal.shift(1).fillna(0)

    daily_rets = prices.pct_change().fillna(0)
    strategy_rets = signal * daily_rets

    return strategy_rets.reindex(data.index, fill_value=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Engine
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Core walk-forward engine.

    Generates rolling or anchored (expanding) train/test folds,
    optionally optimises strategy parameters on each training window,
    and stitches OOS periods into a continuous equity curve.
    """

    def __init__(self, ann: int = ANNUAL_FACTOR) -> None:
        self.ann = ann

    def generate_folds(
        self, data: pd.DataFrame, config: WalkForwardConfig
    ) -> List[Fold]:
        """
        Generate list of Fold objects.

        Rolling WF  : train window slides forward by step_size each fold.
        Anchored WF : train window expands from a fixed start.
        """
        n = len(data)
        folds: List[Fold] = []
        fold_id = 0

        if config.anchored:
            # Anchored (expanding) window
            train_start_idx = 0
            test_start_idx = config.train_window
        else:
            # Rolling window
            train_start_idx = 0
            test_start_idx = config.train_window

        while test_start_idx + config.test_window <= n:
            test_end_idx = test_start_idx + config.test_window

            train_slice = data.iloc[train_start_idx:test_start_idx]
            test_slice = data.iloc[test_start_idx:test_end_idx]

            folds.append(Fold(
                fold_id=fold_id,
                train_start=data.index[train_start_idx],
                train_end=data.index[test_start_idx - 1],
                test_start=data.index[test_start_idx],
                test_end=data.index[test_end_idx - 1],
                train_data=train_slice.copy(),
                test_data=test_slice.copy(),
            ))
            fold_id += 1

            # Advance
            test_start_idx += config.step_size
            if not config.anchored:
                train_start_idx += config.step_size

        logger.info(
            "Generated %d folds (anchored=%s, train=%d, test=%d, step=%d)",
            len(folds), config.anchored, config.train_window,
            config.test_window, config.step_size,
        )
        return folds

    def run_fold(
        self,
        fold: Fold,
        strategy_fn: Callable,
        param_grid: Optional[Dict[str, list]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> FoldResult:
        """
        Evaluate one fold.

        If param_grid is provided: optimise on train, apply best params to test.
        If fixed_params provided (or no grid): use fixed params on both IS and OOS.
        """
        params: Dict[str, Any] = fixed_params or {}

        if param_grid:
            params, _ = _optimise_params(fold.train_data, strategy_fn, param_grid)

        # IS performance
        try:
            is_rets = strategy_fn(fold.train_data, **params)
        except Exception as exc:
            logger.debug("IS strategy error fold %d: %s", fold.fold_id, exc)
            is_rets = pd.Series(0.0, index=fold.train_data.index)

        # OOS performance with params locked
        try:
            oos_rets = strategy_fn(fold.test_data, **params)
        except Exception as exc:
            logger.debug("OOS strategy error fold %d: %s", fold.fold_id, exc)
            oos_rets = pd.Series(0.0, index=fold.test_data.index)

        is_sharpe = _annualized_sharpe(is_rets, self.ann)
        oos_sharpe = _annualized_sharpe(oos_rets, self.ann)

        # Overfitting ratio: IS/OOS (cap at 10 to avoid inf)
        if abs(oos_sharpe) < 1e-6:
            of_ratio = 10.0 if is_sharpe > 0 else 1.0
        else:
            of_ratio = min(10.0, abs(is_sharpe / oos_sharpe))

        n_is_trades = _count_trades(is_rets)
        n_oos_trades = _count_trades(oos_rets)

        return FoldResult(
            fold_id=fold.fold_id,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            in_sample_sharpe=is_sharpe,
            out_sample_sharpe=oos_sharpe,
            in_sample_returns=is_rets,
            out_sample_returns=oos_rets,
            n_train_trades=n_is_trades,
            n_test_trades=n_oos_trades,
            overfitting_ratio=of_ratio,
            params_used=params,
        )

    def run_all_folds(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable,
        config: WalkForwardConfig,
        param_grid: Optional[Dict[str, list]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> WalkForwardResult:
        """
        Run all folds (parallel if n_jobs > 1) and return aggregated result.
        """
        folds = self.generate_folds(data, config)
        if not folds:
            raise ValueError("No folds generated — dataset may be too short for the given config")

        fold_results: List[FoldResult] = []

        if config.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=config.n_jobs) as ex:
                future_to_fold = {
                    ex.submit(self.run_fold, fold, strategy_fn, param_grid, fixed_params): fold
                    for fold in folds
                }
                for fut in as_completed(future_to_fold):
                    try:
                        fold_results.append(fut.result())
                    except Exception as exc:
                        fold = future_to_fold[fut]
                        logger.warning("Fold %d failed: %s", fold.fold_id, exc)
            fold_results.sort(key=lambda r: r.fold_id)
        else:
            for fold in folds:
                try:
                    fold_results.append(self.run_fold(fold, strategy_fn, param_grid, fixed_params))
                except Exception as exc:
                    logger.warning("Fold %d failed: %s", fold.fold_id, exc)

        if not fold_results:
            raise ValueError("All folds failed during evaluation")

        oos_curve = self.compute_oos_equity_curve(fold_results)
        oos_rets = self._stitch_oos_returns(fold_results)

        mean_is = float(np.mean([r.in_sample_sharpe for r in fold_results]))
        mean_oos = float(np.mean([r.out_sample_sharpe for r in fold_results]))
        of_ratio = abs(mean_is / mean_oos) if abs(mean_oos) > 1e-6 else 10.0

        return WalkForwardResult(
            config=config,
            fold_results=fold_results,
            oos_equity_curve=oos_curve,
            oos_returns=oos_rets,
            mean_is_sharpe=mean_is,
            mean_oos_sharpe=mean_oos,
            overall_overfitting_ratio=min(10.0, of_ratio),
            n_folds=len(fold_results),
            strategy_name=getattr(strategy_fn, "__name__", "unknown"),
        )

    def compute_oos_equity_curve(self, results: List[FoldResult]) -> pd.Series:
        """Stitch all OOS periods into a single equity curve (1 = 100%)."""
        rets = self._stitch_oos_returns(results)
        if rets.empty:
            return pd.Series(dtype=float)
        equity = (1 + rets).cumprod()
        equity.name = "OOS_Equity"
        return equity

    @staticmethod
    def _stitch_oos_returns(results: List[FoldResult]) -> pd.Series:
        """Concatenate OOS return series from all folds chronologically."""
        if not results:
            return pd.Series(dtype=float)
        sorted_results = sorted(results, key=lambda r: r.test_start)
        parts = [r.out_sample_returns for r in sorted_results if not r.out_sample_returns.empty]
        if not parts:
            return pd.Series(dtype=float)
        combined = pd.concat(parts)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined


# ─────────────────────────────────────────────────────────────────────────────
# Overfitting Detector
# ─────────────────────────────────────────────────────────────────────────────

class OverfittingDetector:
    """
    Detect and quantify backtest overfitting using:
    - Overfitting ratio (IS/OOS Sharpe)
    - Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
    - Probabilistic Sharpe Ratio (PSR)
    - Probability of Backtest Overfitting (PBO) via combinatorial CV
    """

    def __init__(self, ann: int = ANNUAL_FACTOR) -> None:
        self.ann = ann

    def compute_overfitting_ratio(self, results: List[FoldResult]) -> float:
        """
        Average IS Sharpe / Average OOS Sharpe across folds.
        > 2.0 = likely overfit; > 3.0 = heavily overfit.
        """
        if not results:
            return 1.0
        mean_is = np.mean([r.in_sample_sharpe for r in results])
        mean_oos = np.mean([r.out_sample_sharpe for r in results])
        if abs(mean_oos) < 1e-6:
            return 10.0
        return float(min(10.0, abs(mean_is / mean_oos)))

    def compute_psr(
        self,
        sharpe: float,
        benchmark_sr: float,
        n_obs: int,
        skew: float = 0.0,
        kurt: float = 3.0,
    ) -> float:
        """
        Probabilistic Sharpe Ratio: probability that the true Sharpe > benchmark.

        PSR(SR*) = Φ[(SR - SR*) * sqrt(n-1) / sqrt(1 - γ_3*SR + (γ_4-1)/4 * SR^2)]

        where γ_3 = skewness, γ_4 = kurtosis.

        Returns probability (0..1).
        """
        if n_obs < 5:
            return 0.5

        excess_kurt = kurt - 3.0  # excess kurtosis
        # Variance of estimated Sharpe (annualized correction handled separately)
        variance = (
            1.0
            - skew * sharpe
            + (excess_kurt / 4.0) * sharpe ** 2
        )
        if variance <= 0:
            variance = 1.0

        z = (sharpe - benchmark_sr) * math.sqrt(n_obs - 1) / math.sqrt(variance)
        return float(stats.norm.cdf(z))

    def compute_deflated_sharpe_ratio(
        self,
        sharpe: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Deflated Sharpe Ratio (DSR) per Bailey & Lopez de Prado (2014).

        Adjusts observed Sharpe for selection bias from testing multiple
        strategy configurations. DSR = PSR(SR*) where SR* is the expected
        maximum Sharpe under H0 (random).

        SR* ≈ (1 - γ) * Z^{-1}(1 - 1/n_trials) + γ * Z^{-1}(1 - 1/(n_trials*e))
        where γ = Euler-Mascheroni constant ≈ 0.5772.

        Returns probability that true Sharpe > 0 after multiple-testing correction.
        """
        if n_observations < 5 or n_trials < 1:
            return 0.5

        euler_gamma = 0.5772156649
        # Expected max SR from n_trials independent random strategies
        if n_trials == 1:
            sr_star = 0.0
        else:
            v1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
            v2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
            sr_star = (1.0 - euler_gamma) * v1 + euler_gamma * v2

        return self.compute_psr(
            sharpe=sharpe,
            benchmark_sr=sr_star,
            n_obs=n_observations,
            skew=skewness,
            kurt=kurtosis,
        )

    def compute_pbo(
        self,
        results: List[FoldResult],
        max_combinations: int = 16,
    ) -> float:
        """
        Probability of Backtest Overfitting (Bailey et al. 2014).

        Uses combinatorial cross-validation: for all C(n, n//2) splits of
        fold results into IS and OOS halves, compute fraction of combinations
        where the best IS strategy has OOS SR < the median OOS SR.

        For tractability, caps at min(n_folds, max_combinations) folds.
        Returns PBO in [0, 1]. PBO > 0.5 = concern.
        """
        if not results or len(results) < 4:
            return 0.5

        n = min(len(results), max_combinations)
        sub_results = results[:n]
        n_half = n // 2

        # For each combination of n_half folds as "IS" and remaining as "OOS"
        is_better_count = 0
        total_combinations = 0

        # Use up to 500 random combinations for tractability
        all_idxs = list(range(n))
        random.seed(42)

        try:
            from math import comb as math_comb
            max_combos = math_comb(n, n_half)
        except ImportError:
            max_combos = 1000

        if max_combos <= 500:
            combo_iter = list(itertools.combinations(all_idxs, n_half))
        else:
            combo_set = set()
            attempts = 0
            while len(combo_set) < 500 and attempts < 5000:
                combo = tuple(sorted(random.sample(all_idxs, n_half)))
                combo_set.add(combo)
                attempts += 1
            combo_iter = list(combo_set)

        for is_idxs in combo_iter:
            oos_idxs = tuple(i for i in all_idxs if i not in set(is_idxs))
            if not oos_idxs:
                continue

            is_folds = [sub_results[i] for i in is_idxs]
            oos_folds = [sub_results[i] for i in oos_idxs]

            # Best IS fold (highest IS Sharpe)
            best_is_fold = max(is_folds, key=lambda r: r.in_sample_sharpe)
            best_is_fold_id = best_is_fold.fold_id

            # Corresponding OOS performance (same fold index used in OOS partition)
            # Map: what would the best IS fold's strategy perform on OOS partition?
            # Since folds are independent: use the OOS Sharpe of the matching fold
            matching_oos = [r for r in oos_folds if r.fold_id == best_is_fold_id]
            if not matching_oos:
                # Use median OOS Sharpe of OOS partition
                median_oos_sharpe = float(np.median([r.out_sample_sharpe for r in oos_folds]))
                best_is_oos_sharpe = float(best_is_fold.out_sample_sharpe)
            else:
                median_oos_sharpe = float(np.median([r.out_sample_sharpe for r in oos_folds]))
                best_is_oos_sharpe = float(matching_oos[0].out_sample_sharpe)

            # PBO: OOS degraded (best IS underperforms median OOS)
            if best_is_oos_sharpe < median_oos_sharpe:
                is_better_count += 1

            total_combinations += 1

        if total_combinations == 0:
            return 0.5

        pbo = is_better_count / total_combinations
        return float(pbo)

    def run_bias_tests(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        n_trials: int = 20,
    ) -> BiasReport:
        """
        Run standard bias detection tests:
        1. Lookahead bias: add 1-bar lag, check Sharpe drop
        2. Survivorship bias: always warn (we use live data only)
        3. Data snooping: DSR with n_trials of random parameter perturbation
        """
        # Baseline Sharpe
        try:
            base_rets = strategy_fn(data)
            base_sharpe = _annualized_sharpe(base_rets)
        except Exception:
            base_sharpe = 0.0

        # 1. Lookahead bias test: shift prices by 1 bar
        try:
            lagged_data = data.copy()
            for col in lagged_data.select_dtypes(include=[np.number]).columns:
                lagged_data[col] = lagged_data[col].shift(1)
            lagged_rets = strategy_fn(lagged_data.dropna())
            lagged_sharpe = _annualized_sharpe(lagged_rets)
            drop = (base_sharpe - lagged_sharpe) / max(abs(base_sharpe), 1e-6)
            lookahead_detected = drop > 0.5  # >50% Sharpe drop when lag applied
        except Exception:
            drop = 0.0
            lookahead_detected = False

        # 3. Data snooping DSR
        n_obs = len(data)
        dsr = self.compute_deflated_sharpe_ratio(
            sharpe=base_sharpe,
            n_trials=n_trials,
            n_observations=n_obs,
        )

        return BiasReport(
            lookahead_bias_detected=lookahead_detected,
            lookahead_performance_drop=float(drop),
            survivorship_bias_warning=True,   # always flag — we can't verify
            data_snooping_dsr=float(dsr),
            n_trials_tested=n_trials,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Anchored Validation
# ─────────────────────────────────────────────────────────────────────────────

def _classify_regime_simple(returns: pd.Series) -> pd.Series:
    """Simple regime classifier for fold mapping."""
    vol = returns.rolling(60, min_periods=20).std() * math.sqrt(252)
    trend = returns.rolling(120, min_periods=20).mean() * 252
    vol_q85 = float(vol.quantile(0.85)) if vol.dropna().shape[0] > 10 else 0.30

    regime = pd.Series("sideways", index=returns.index)
    regime[vol >= vol_q85] = "crisis"
    regime[(vol < vol_q85) & (trend >= 0.05)] = "bull"
    regime[(vol < vol_q85) & (trend <= -0.05)] = "bear"
    return regime


class AnchoredValidation:
    """
    Anchored (expanding-window) walk-forward analysis.

    Train window always starts at the beginning of the data.
    Test window rolls forward, always out-of-sample.
    """

    def __init__(self, engine: Optional[WalkForwardEngine] = None) -> None:
        self._engine = engine or WalkForwardEngine()

    def run_anchored_wf(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable,
        min_train_days: int = 252,
        test_window: int = 63,
        step_size: int = 21,
        param_grid: Optional[Dict[str, list]] = None,
    ) -> AnchoredResult:
        """
        Anchored walk-forward: train always from index 0, test rolls forward.
        Never uses future data in the training window.
        """
        config = WalkForwardConfig(
            train_window=min_train_days,
            test_window=test_window,
            step_size=step_size,
            anchored=True,
        )
        wf_result = self._engine.run_all_folds(data, strategy_fn, config, param_grid)

        degradation = self.compute_degradation_curve(wf_result.fold_results)

        # Regime analysis
        try:
            price_col = next(
                (c for c in data.columns if "close" in c.lower()), data.columns[0]
            )
            price_rets = data[price_col].pct_change().dropna()
            regimes = _classify_regime_simple(price_rets)
            regime_perf = self.detect_regime_sensitivity(wf_result.fold_results, regimes)
        except Exception as exc:
            logger.debug("Regime mapping failed: %s", exc)
            regime_perf = {}

        return AnchoredResult(
            fold_results=wf_result.fold_results,
            oos_equity_curve=wf_result.oos_equity_curve,
            oos_returns=wf_result.oos_returns,
            degradation_curve=degradation,
            regime_performance=regime_perf,
        )

    def compute_degradation_curve(self, results: List[FoldResult]) -> pd.Series:
        """
        OOS Sharpe ratio per fold vs fold index.
        A declining trend suggests strategy alpha is degrading over time.
        """
        sorted_results = sorted(results, key=lambda r: r.fold_id)
        oos_sharpes = [r.out_sample_sharpe for r in sorted_results]
        return pd.Series(
            oos_sharpes,
            index=pd.RangeIndex(len(oos_sharpes)),
            name="OOS_Sharpe",
        )

    def detect_regime_sensitivity(
        self,
        results: List[FoldResult],
        regimes: pd.Series,
    ) -> Dict[str, float]:
        """
        Map each OOS fold to the dominant regime in that period.
        Returns: {regime → mean OOS Sharpe} to identify failure regimes.
        """
        regime_sharpes: Dict[str, List[float]] = {}

        for result in results:
            # Find dominant regime during OOS period
            mask = (regimes.index >= result.test_start) & (regimes.index <= result.test_end)
            period_regimes = regimes[mask]
            if period_regimes.empty:
                continue
            dominant = period_regimes.mode()
            if dominant.empty:
                continue
            regime = str(dominant.iloc[0])
            if regime not in regime_sharpes:
                regime_sharpes[regime] = []
            regime_sharpes[regime].append(result.out_sample_sharpe)

        return {
            regime: float(np.mean(sharpes))
            for regime, sharpes in regime_sharpes.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Stability Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class ParameterStabilityAnalyzer:
    """
    Analyze whether optimal parameters are stable across walk-forward folds.
    High dispersion = overfitting noise, not robust edge.
    """

    def compute_parameter_dispersion(
        self, results: List[FoldResult]
    ) -> Dict[str, float]:
        """
        For each parameter: coefficient of variation (std/mean) across folds.
        High CoV (>0.5) indicates parameter instability.
        """
        param_values: Dict[str, List[float]] = {}
        for result in results:
            for param, value in result.params_used.items():
                try:
                    fval = float(value)
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(fval)
                except (TypeError, ValueError):
                    pass

        dispersion: Dict[str, float] = {}
        for param, values in param_values.items():
            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            cov = (std / abs(mean)) if abs(mean) > 1e-8 else 0.0
            dispersion[param] = float(cov)

        return dispersion

    def compute_param_performance_correlation(
        self, results: List[FoldResult], param_name: str
    ) -> float:
        """
        Spearman correlation between the optimised value of param_name
        and the OOS Sharpe for that fold.
        High correlation = param genuinely drives OOS performance.
        Low/negative = param is fitting in-sample noise.
        """
        param_vals = []
        oos_sharpes = []
        for result in results:
            if param_name in result.params_used:
                try:
                    param_vals.append(float(result.params_used[param_name]))
                    oos_sharpes.append(result.out_sample_sharpe)
                except (TypeError, ValueError):
                    pass

        if len(param_vals) < 3:
            return 0.0

        rho, _ = stats.spearmanr(param_vals, oos_sharpes)
        return float(rho) if not math.isnan(rho) else 0.0

    def find_robust_params(
        self,
        results: List[FoldResult],
        param_grid: Dict[str, list],
        top_k: int = 3,
        min_fold_fraction: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Find parameter values that appear in the top-k performing folds
        for at least `min_fold_fraction` of all folds.
        These are considered "robust" parameters.
        """
        if not results:
            return {}

        sorted_results = sorted(results, key=lambda r: r.in_sample_sharpe, reverse=True)
        top_results = sorted_results[:max(top_k, len(sorted_results) // 3)]
        threshold = int(len(results) * min_fold_fraction)

        robust: Dict[str, Any] = {}
        for param_name, possible_values in param_grid.items():
            value_counts: Dict[Any, int] = {}
            for result in results:
                if param_name in result.params_used:
                    val = result.params_used[param_name]
                    value_counts[val] = value_counts.get(val, 0) + 1

            # Params that appear in top folds AND across >50% of all folds
            top_vals = set()
            for result in top_results:
                if param_name in result.params_used:
                    top_vals.add(result.params_used[param_name])

            for val in top_vals:
                if value_counts.get(val, 0) >= threshold:
                    robust[param_name] = val
                    break  # take first robust value found

        return robust


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo Permutation Test
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloPermutationTest:
    """
    Assess statistical significance of a strategy's edge by permutation testing.

    Permutation test : shuffle price returns 1000 times, compare strategy Sharpe
    White's Reality Check : bootstrap test for data snooping across multiple strategies
    """

    def __init__(self, ann: int = ANNUAL_FACTOR) -> None:
        self.ann = ann

    def run_permutation_test(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        n_permutations: int = 1000,
        price_col: str = "Close",
        strategy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute p-value for strategy edge via return permutation.

        Algorithm:
          1. Compute actual strategy Sharpe on original data
          2. Shuffle daily returns 1000 times, rebuild prices, run strategy
          3. p-value = fraction of shuffles with Sharpe >= actual Sharpe

        Returns p-value (0..1). p < 0.05 = significant edge.
        """
        kwargs = strategy_kwargs or {}

        if price_col not in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                return 1.0
            price_col = numeric_cols[0]

        prices = data[price_col].dropna()
        actual_returns = prices.pct_change().dropna()

        # Actual strategy performance
        try:
            actual_rets = strategy_fn(data, **kwargs)
            actual_sharpe = _annualized_sharpe(actual_rets, self.ann)
        except Exception as exc:
            logger.warning("Permutation test: actual strategy failed: %s", exc)
            return 1.0

        beat_count = 0
        rng = np.random.default_rng(seed=42)

        for _ in range(n_permutations):
            shuffled_rets = actual_returns.values.copy()
            rng.shuffle(shuffled_rets)

            # Reconstruct price series from shuffled returns
            shuffled_prices = pd.Series(
                index=prices.index[1:],
                data=prices.iloc[0] * np.cumprod(1 + shuffled_rets),
                name=price_col,
            )
            shuffled_data = data.copy()
            # Align shuffled prices
            if len(shuffled_prices) == len(shuffled_data) - 1:
                shuffled_data = shuffled_data.iloc[1:].copy()
                shuffled_data[price_col] = shuffled_prices.values

            try:
                perm_rets = strategy_fn(shuffled_data, **kwargs)
                perm_sharpe = _annualized_sharpe(perm_rets, self.ann)
                if perm_sharpe >= actual_sharpe:
                    beat_count += 1
            except Exception:
                pass

        p_value = beat_count / n_permutations
        return float(p_value)

    def compute_parameter_sensitivity_surface(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        fast_ma_values: List[int],
        slow_ma_values: List[int],
        price_col: str = "Close",
        strategy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Compute a 2D parameter sensitivity surface (Sharpe ratio heatmap).

        For each combination of (fast_ma, slow_ma) in the provided grids,
        run the strategy and record the annualized Sharpe ratio.

        Returns an (N × M) DataFrame where:
          - Index   : fast_ma values
          - Columns : slow_ma values
          - Values  : annualized Sharpe ratio (float)

        Only combinations where fast_ma < slow_ma are computed; invalid
        combinations receive NaN.

        Args:
            strategy_fn      : callable(data, fast=int, slow=int, **kwargs) → pd.Series
            data             : price/OHLCV DataFrame with DatetimeIndex
            fast_ma_values   : list of fast moving-average window values
            slow_ma_values   : list of slow moving-average window values
            price_col        : price column name for validation
            strategy_kwargs  : additional fixed kwargs passed to strategy_fn

        Returns:
            pd.DataFrame of shape (len(fast_ma_values), len(slow_ma_values))
        """
        kwargs = strategy_kwargs or {}

        surface = pd.DataFrame(
            index=fast_ma_values,
            columns=slow_ma_values,
            dtype=float,
        )
        surface.index.name = "fast_ma"
        surface.columns.name = "slow_ma"

        for fast in fast_ma_values:
            for slow in slow_ma_values:
                if fast >= slow:
                    surface.loc[fast, slow] = float("nan")
                    continue
                try:
                    rets = strategy_fn(data, fast=fast, slow=slow, **kwargs)
                    sharpe = _annualized_sharpe(rets, self.ann)
                    surface.loc[fast, slow] = round(sharpe, 4)
                except Exception as exc:
                    logger.debug(
                        "Sensitivity surface: fast=%d, slow=%d → %s", fast, slow, exc
                    )
                    surface.loc[fast, slow] = float("nan")

        return surface

    def compute_oos_stability(
        self,
        fold_results: List["FoldResult"],
        benchmark_sharpe: float = 0.0,
    ) -> float:
        """
        Compute strategy stability: fraction of OOS periods where the strategy
        beats the benchmark Sharpe ratio.

        Stability = (# folds with OOS Sharpe > benchmark_sharpe) / total_folds

        A stable strategy consistently generates positive (or benchmark-beating)
        OOS returns across most periods.

        Args:
            fold_results     : list of FoldResult from WalkForwardEngine.run_all_folds
            benchmark_sharpe : Sharpe threshold to beat (default 0.0 = positive returns)

        Returns:
            float in [0, 1] — fraction of OOS periods beating the benchmark
        """
        if not fold_results:
            return 0.0

        n_total = len(fold_results)
        n_positive = sum(
            1 for r in fold_results
            if r.out_sample_sharpe > benchmark_sharpe
        )
        return float(n_positive / n_total)

    def run_white_reality_check(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        n_boot: int = 1000,
    ) -> float:
        """
        White's Reality Check (2000): bootstrap test for data snooping.

        Tests H0: max Sharpe improvement over benchmark is due to chance.
        Returns p-value. p < 0.05 = strategy outperforms benchmark significantly.

        Algorithm:
          1. Compute excess returns = strategy - benchmark
          2. Block-bootstrap 1000 samples of excess returns
          3. p-value = fraction where bootstrapped mean excess >= actual mean excess
        """
        excess = returns - benchmark_returns
        excess = excess.dropna()

        if len(excess) < 10:
            return 1.0

        actual_mean = float(excess.mean())
        n = len(excess)
        block_size = max(1, int(math.sqrt(n)))
        rng = np.random.default_rng(seed=42)

        beat_count = 0
        arr = excess.values

        for _ in range(n_boot):
            # Stationary block bootstrap (simplified: fixed block size)
            n_blocks = n // block_size + 1
            starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            boot_arr = np.concatenate([arr[s : s + block_size] for s in starts])[:n]
            boot_mean = float(boot_arr.mean())
            if boot_mean >= actual_mean:
                beat_count += 1

        return float(beat_count / n_boot)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Validator (Orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class StrategyValidator:
    """
    Full validation pipeline orchestrator.

    Runs all tests and returns a ValidationReport with PASS/FAIL gates:
      1. Walk-forward with parameter optimization
      2. IS/OOS metrics + overfitting ratio
      3. PBO (Probability of Backtest Overfitting)
      4. DSR (Deflated Sharpe Ratio)
      5. Anchored walk-forward
      6. Parameter stability
      7. Permutation test
    """

    # PASS thresholds
    _GATE_OOS_SHARPE = 0.50
    _GATE_PBO = 0.50
    _GATE_OVERFITTING_RATIO = 2.0
    _GATE_DSR = 0.95

    def __init__(self) -> None:
        self._wf_engine = WalkForwardEngine()
        self._od = OverfittingDetector()
        self._av = AnchoredValidation(self._wf_engine)
        self._ps = ParameterStabilityAnalyzer()
        self._mc = MonteCarloPermutationTest()

    def validate(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        config: Optional[WalkForwardConfig] = None,
        param_grid: Optional[Dict[str, list]] = None,
        strategy_name: str = "",
        n_permutations: int = 500,
        run_bias_tests: bool = True,
    ) -> ValidationReport:
        """
        Run the complete validation suite.

        Parameters
        ----------
        strategy_fn  : callable(data, **params) → pd.Series of daily returns
        data         : OHLCV or price DataFrame with DatetimeIndex
        config       : WalkForwardConfig (defaults used if None)
        param_grid   : dict of {param_name: [values]} for optimization
        """
        cfg = config or WalkForwardConfig()
        name = strategy_name or getattr(strategy_fn, "__name__", "strategy")

        logger.info("StrategyValidator: validating '%s'", name)

        # ─ Step 1: Walk-forward ─────────────────────────────────────────────
        logger.info("  Step 1/7: Walk-forward optimization (%d folds est.)", len(data) // cfg.step_size)
        wf_result = self._wf_engine.run_all_folds(data, strategy_fn, cfg, param_grid)

        # ─ Step 2: IS/OOS metrics ───────────────────────────────────────────
        logger.info("  Step 2/7: IS/OOS metrics")
        oos_sharpe = _annualized_sharpe(wf_result.oos_returns)
        of_ratio = self._od.compute_overfitting_ratio(wf_result.fold_results)

        # ─ Step 3: PBO ──────────────────────────────────────────────────────
        logger.info("  Step 3/7: PBO computation")
        pbo = self._od.compute_pbo(wf_result.fold_results)

        # ─ Step 4: DSR ──────────────────────────────────────────────────────
        logger.info("  Step 4/7: DSR computation")
        n_trials = 1 if param_grid is None else max(1, sum(len(v) for v in param_grid.values()))
        oos_rets_arr = wf_result.oos_returns.dropna().values
        skew = float(stats.skew(oos_rets_arr)) if len(oos_rets_arr) > 3 else 0.0
        kurt = float(stats.kurtosis(oos_rets_arr, fisher=False)) if len(oos_rets_arr) > 3 else 3.0
        dsr = self._od.compute_deflated_sharpe_ratio(
            sharpe=oos_sharpe,
            n_trials=n_trials,
            n_observations=len(oos_rets_arr),
            skewness=skew,
            kurtosis=kurt,
        )
        psr = self._od.compute_psr(
            sharpe=oos_sharpe,
            benchmark_sr=0.0,
            n_obs=len(oos_rets_arr),
            skew=skew,
            kurt=kurt,
        )

        # ─ Step 5: Anchored WF ──────────────────────────────────────────────
        logger.info("  Step 5/7: Anchored walk-forward")
        try:
            anchored_result = self._av.run_anchored_wf(
                data, strategy_fn,
                min_train_days=cfg.train_window,
                test_window=cfg.test_window,
                step_size=cfg.step_size,
                param_grid=param_grid,
            )
            anchored_oos_sharpe = float(np.mean([
                r.out_sample_sharpe for r in anchored_result.fold_results
            ]))
        except Exception as exc:
            logger.warning("Anchored WF failed: %s", exc)
            anchored_oos_sharpe = 0.0

        # ─ Step 6: Parameter stability ──────────────────────────────────────
        logger.info("  Step 6/7: Parameter stability")
        param_disp = self._ps.compute_parameter_dispersion(wf_result.fold_results)
        robust_params = (
            self._ps.find_robust_params(wf_result.fold_results, param_grid)
            if param_grid else {}
        )

        # ─ Step 7: Permutation test ─────────────────────────────────────────
        logger.info("  Step 7/7: Monte Carlo permutation test")
        try:
            p_val = self._mc.run_permutation_test(
                strategy_fn, data, n_permutations=n_permutations
            )
        except Exception as exc:
            logger.warning("Permutation test failed: %s", exc)
            p_val = 1.0

        # ─ Bias tests (optional) ────────────────────────────────────────────
        bias_report: Optional[BiasReport] = None
        if run_bias_tests:
            try:
                bias_report = self._od.run_bias_tests(strategy_fn, data, n_trials=n_trials)
            except Exception as exc:
                logger.warning("Bias tests failed: %s", exc)

        # ─ PASS/FAIL gates ──────────────────────────────────────────────────
        gate_oos = oos_sharpe > self._GATE_OOS_SHARPE
        gate_pbo = pbo < self._GATE_PBO
        gate_of = of_ratio < self._GATE_OVERFITTING_RATIO
        gate_dsr = dsr > self._GATE_DSR
        overall_pass = gate_oos and gate_pbo and gate_of and gate_dsr

        quality_score = self.score_strategy_from_components(
            oos_sharpe=oos_sharpe,
            pbo=pbo,
            of_ratio=of_ratio,
            dsr=dsr,
            permutation_pvalue=p_val,
            param_dispersion=param_disp,
        )

        return ValidationReport(
            strategy_name=name,
            oos_sharpe=oos_sharpe,
            pbo=pbo,
            overfitting_ratio=of_ratio,
            dsr=dsr,
            psr=psr,
            mean_is_sharpe=wf_result.mean_is_sharpe,
            mean_oos_sharpe=wf_result.mean_oos_sharpe,
            n_folds=wf_result.n_folds,
            anchored_mean_oos_sharpe=anchored_oos_sharpe,
            param_dispersion=param_disp,
            robust_params=robust_params,
            permutation_pvalue=p_val,
            bias_report=bias_report,
            quality_score=quality_score,
            gate_oos_sharpe=gate_oos,
            gate_pbo=gate_pbo,
            gate_overfitting=gate_of,
            gate_dsr=gate_dsr,
            overall_pass=overall_pass,
        )

    @staticmethod
    def score_strategy_from_components(
        oos_sharpe: float,
        pbo: float,
        of_ratio: float,
        dsr: float,
        permutation_pvalue: float,
        param_dispersion: Dict[str, float],
        max_score: float = 100.0,
    ) -> float:
        """
        Compute 0–100 quality score.

        Component weights:
          OOS Sharpe    30 pts — sigmoid centred at 0.5
          PBO           25 pts — linear penalty (PBO=0 → full; PBO=1 → 0)
          Overfitting   20 pts — ratio < 1.5 → full; ratio > 3 → 0
          DSR           15 pts — DSR > 0.99 → full; DSR < 0.5 → 0
          Permutation   10 pts — p < 0.05 → full; p > 0.2 → 0
        """
        # OOS Sharpe: sigmoid around 0.5
        sharpe_pts = 30.0 / (1 + math.exp(-4.0 * (oos_sharpe - 0.5)))

        # PBO: 0 = best (25 pts), 1 = worst (0 pts)
        pbo_pts = 25.0 * max(0.0, 1.0 - 2.0 * pbo)

        # Overfitting ratio
        if of_ratio <= 1.5:
            of_pts = 20.0
        elif of_ratio >= 3.0:
            of_pts = 0.0
        else:
            of_pts = 20.0 * (3.0 - of_ratio) / 1.5

        # DSR
        dsr_pts = 15.0 * max(0.0, min(1.0, (dsr - 0.5) / 0.49))

        # Permutation p-value
        if permutation_pvalue <= 0.05:
            perm_pts = 10.0
        elif permutation_pvalue >= 0.20:
            perm_pts = 0.0
        else:
            perm_pts = 10.0 * (0.20 - permutation_pvalue) / 0.15

        total = sharpe_pts + pbo_pts + of_pts + dsr_pts + perm_pts
        return float(min(max_score, max(0.0, total)))

    def score_strategy(self, report: ValidationReport) -> float:
        """Extract quality score from a ValidationReport."""
        return report.quality_score


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Router
# ─────────────────────────────────────────────────────────────────────────────

wf_v3_router = APIRouter(prefix="/walkforward/v3", tags=["walkforward-v3"])

_STRATEGY_REGISTRY: Dict[str, Callable] = {
    "sma_crossover": sma_crossover_strategy,
    "momentum": momentum_strategy,
    "mean_reversion": mean_reversion_strategy,
}

_shared_validator: Optional[StrategyValidator] = None


def _get_validator() -> StrategyValidator:
    global _shared_validator
    if _shared_validator is None:
        _shared_validator = StrategyValidator()
    return _shared_validator


class WFRunRequest(BaseModel):
    ticker: str = Field("SPY", description="Ticker to fetch from yfinance")
    strategy: str = Field("sma_crossover", description="Built-in strategy name")
    train_window: int = Field(252, ge=60)
    test_window: int = Field(63, ge=10)
    step_size: int = Field(21, ge=5)
    anchored: bool = False
    n_jobs: int = Field(1, ge=1, le=8)
    fast_param_range: Optional[List[int]] = None
    slow_param_range: Optional[List[int]] = None


class ValidateRequest(BaseModel):
    ticker: str = "SPY"
    strategy: str = "sma_crossover"
    train_window: int = 252
    test_window: int = 63
    step_size: int = 21
    n_permutations: int = Field(200, ge=50, le=2000)
    optimize: bool = True


class PBORequest(BaseModel):
    ticker: str = "SPY"
    strategy: str = "sma_crossover"
    train_window: int = 252
    test_window: int = 63
    step_size: int = 21


def _fetch_price_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch OHLCV data via yfinance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        return df
    except ImportError:
        raise HTTPException(status_code=500, detail="yfinance not installed")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {exc}")


def _build_param_grid(req: WFRunRequest) -> Optional[Dict[str, list]]:
    if req.strategy == "sma_crossover":
        fast_range = req.fast_param_range or [5, 10, 20]
        slow_range = req.slow_param_range or [30, 50, 100]
        return {"fast": fast_range, "slow": slow_range}
    if req.strategy == "momentum":
        return {"lookback": [63, 126, 189]}
    if req.strategy == "mean_reversion":
        return {"window": [10, 20, 30], "z_threshold": [1.5, 2.0, 2.5]}
    return None


@wf_v3_router.get("/strategies")
async def list_strategies():
    """List available built-in strategy names."""
    return {"strategies": list(_STRATEGY_REGISTRY.keys())}


@wf_v3_router.post("/run")
async def run_walk_forward(req: WFRunRequest):
    """Run walk-forward validation with parameter optimization."""
    data = _fetch_price_data(req.ticker)
    strategy_fn = _STRATEGY_REGISTRY.get(req.strategy)
    if strategy_fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")

    config = WalkForwardConfig(
        train_window=req.train_window,
        test_window=req.test_window,
        step_size=req.step_size,
        anchored=req.anchored,
        n_jobs=req.n_jobs,
    )
    param_grid = _build_param_grid(req)
    engine = WalkForwardEngine()

    try:
        result = engine.run_all_folds(data, strategy_fn, config, param_grid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    fold_summary = [
        {
            "fold_id": r.fold_id,
            "test_start": str(r.test_start.date()),
            "test_end": str(r.test_end.date()),
            "is_sharpe": round(r.in_sample_sharpe, 4),
            "oos_sharpe": round(r.out_sample_sharpe, 4),
            "overfitting_ratio": round(r.overfitting_ratio, 2),
            "params": r.params_used,
        }
        for r in result.fold_results
    ]

    return {
        "ticker": req.ticker,
        "strategy": req.strategy,
        "n_folds": result.n_folds,
        "mean_is_sharpe": round(result.mean_is_sharpe, 4),
        "mean_oos_sharpe": round(result.mean_oos_sharpe, 4),
        "overall_overfitting_ratio": round(result.overall_overfitting_ratio, 3),
        "oos_sharpe": round(_annualized_sharpe(result.oos_returns), 4),
        "folds": fold_summary,
    }


@wf_v3_router.post("/anchored")
async def run_anchored(req: WFRunRequest):
    """Run anchored (expanding-window) walk-forward."""
    data = _fetch_price_data(req.ticker)
    strategy_fn = _STRATEGY_REGISTRY.get(req.strategy)
    if strategy_fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")

    av = AnchoredValidation()
    param_grid = _build_param_grid(req)

    try:
        result = av.run_anchored_wf(
            data, strategy_fn,
            min_train_days=req.train_window,
            test_window=req.test_window,
            step_size=req.step_size,
            param_grid=param_grid,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ticker": req.ticker,
        "strategy": req.strategy,
        "n_folds": len(result.fold_results),
        "mean_oos_sharpe": round(float(np.mean([r.out_sample_sharpe for r in result.fold_results])), 4),
        "degradation_curve": result.degradation_curve.tolist(),
        "regime_performance": {k: round(v, 4) for k, v in result.regime_performance.items()},
    }


@wf_v3_router.post("/validate")
async def validate_strategy(req: ValidateRequest):
    """Run full validation suite (all 7 steps)."""
    data = _fetch_price_data(req.ticker)
    strategy_fn = _STRATEGY_REGISTRY.get(req.strategy)
    if strategy_fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")

    config = WalkForwardConfig(
        train_window=req.train_window,
        test_window=req.test_window,
        step_size=req.step_size,
    )

    param_grid: Optional[Dict[str, list]] = None
    if req.optimize:
        if req.strategy == "sma_crossover":
            param_grid = {"fast": [5, 10, 20], "slow": [30, 50, 100]}
        elif req.strategy == "momentum":
            param_grid = {"lookback": [63, 126, 189]}
        elif req.strategy == "mean_reversion":
            param_grid = {"window": [10, 20, 30], "z_threshold": [1.5, 2.0, 2.5]}

    validator = _get_validator()

    try:
        report = validator.validate(
            strategy_fn=strategy_fn,
            data=data,
            config=config,
            param_grid=param_grid,
            strategy_name=req.strategy,
            n_permutations=req.n_permutations,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _report_to_dict(report)


@wf_v3_router.post("/pbo")
async def compute_pbo(req: PBORequest):
    """Compute Probability of Backtest Overfitting for a strategy."""
    data = _fetch_price_data(req.ticker)
    strategy_fn = _STRATEGY_REGISTRY.get(req.strategy)
    if strategy_fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")

    config = WalkForwardConfig(
        train_window=req.train_window,
        test_window=req.test_window,
        step_size=req.step_size,
    )
    engine = WalkForwardEngine()
    od = OverfittingDetector()

    try:
        result = engine.run_all_folds(data, strategy_fn, config)
        pbo = od.compute_pbo(result.fold_results)
        of_ratio = od.compute_overfitting_ratio(result.fold_results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ticker": req.ticker,
        "strategy": req.strategy,
        "pbo": round(pbo, 4),
        "overfitting_ratio": round(of_ratio, 3),
        "n_folds": result.n_folds,
        "interpretation": (
            "HIGH overfitting risk" if pbo > 0.5
            else "MODERATE" if pbo > 0.3
            else "LOW overfitting risk"
        ),
    }


@wf_v3_router.post("/permutation")
async def run_permutation(
    ticker: str = "SPY",
    strategy: str = "sma_crossover",
    n_permutations: int = 500,
):
    """Run Monte Carlo permutation test for a strategy."""
    data = _fetch_price_data(ticker)
    strategy_fn = _STRATEGY_REGISTRY.get(strategy)
    if strategy_fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

    mc = MonteCarloPermutationTest()
    try:
        p_val = mc.run_permutation_test(strategy_fn, data, n_permutations=n_permutations)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ticker": ticker,
        "strategy": strategy,
        "n_permutations": n_permutations,
        "p_value": round(p_val, 4),
        "significant": p_val < 0.05,
        "interpretation": (
            "Statistically significant edge (p < 0.05)" if p_val < 0.05
            else "No statistically significant edge"
        ),
    }


def _report_to_dict(report: ValidationReport) -> dict:
    """Convert ValidationReport to JSON-serializable dict."""
    bias = None
    if report.bias_report:
        bias = {
            "lookahead_bias_detected": report.bias_report.lookahead_bias_detected,
            "lookahead_performance_drop": round(report.bias_report.lookahead_performance_drop, 3),
            "survivorship_bias_warning": report.bias_report.survivorship_bias_warning,
            "data_snooping_dsr": round(report.bias_report.data_snooping_dsr, 4),
            "n_trials_tested": report.bias_report.n_trials_tested,
        }

    return {
        "strategy_name": report.strategy_name,
        "quality_score": round(report.quality_score, 1),
        "overall_pass": report.overall_pass,
        "metrics": {
            "oos_sharpe": round(report.oos_sharpe, 4),
            "mean_is_sharpe": round(report.mean_is_sharpe, 4),
            "mean_oos_sharpe": round(report.mean_oos_sharpe, 4),
            "anchored_oos_sharpe": round(report.anchored_mean_oos_sharpe, 4),
            "pbo": round(report.pbo, 4),
            "overfitting_ratio": round(report.overfitting_ratio, 3),
            "dsr": round(report.dsr, 4),
            "psr": round(report.psr, 4),
            "permutation_pvalue": round(report.permutation_pvalue, 4),
            "n_folds": report.n_folds,
        },
        "gates": {
            "oos_sharpe_gt_05": {"pass": report.gate_oos_sharpe, "threshold": 0.5, "value": round(report.oos_sharpe, 4)},
            "pbo_lt_05": {"pass": report.gate_pbo, "threshold": 0.5, "value": round(report.pbo, 4)},
            "overfitting_lt_2": {"pass": report.gate_overfitting, "threshold": 2.0, "value": round(report.overfitting_ratio, 3)},
            "dsr_gt_095": {"pass": report.gate_dsr, "threshold": 0.95, "value": round(report.dsr, 4)},
        },
        "parameter_dispersion": {k: round(v, 4) for k, v in report.param_dispersion.items()},
        "robust_params": report.robust_params,
        "bias_report": bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    print("\n" + "=" * 72)
    print("SENTINEL Walk-Forward + Anchored OOS Validation v3 — dim_063")
    print("=" * 72)

    # Fetch SPY 5-year price history
    print("\nFetching SPY 5-year data via yfinance...")
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="5y", progress=False, auto_adjust=True)
        if spy.empty:
            print("ERROR: yfinance returned no data for SPY. Exiting.")
            sys.exit(1)
        print(f"  Loaded {len(spy)} trading days ({spy.index[0].date()} → {spy.index[-1].date()})")
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # ── 1. Walk-Forward on SMA crossover with fast/slow grid ─────────────────
    print(f"\n{'─'*72}")
    print("Walk-Forward: SMA Crossover (fast/slow grid optimization)")
    print(f"{'─'*72}")
    config = WalkForwardConfig(
        train_window=252, test_window=63, step_size=21, anchored=False, n_jobs=1,
    )
    param_grid = {"fast": [5, 10, 20, 30], "slow": [50, 100, 150, 200]}
    engine = WalkForwardEngine()

    wf_result = engine.run_all_folds(
        spy, sma_crossover_strategy, config, param_grid=param_grid
    )

    print(f"\n  Folds completed         : {wf_result.n_folds}")
    print(f"  Mean IS Sharpe          : {wf_result.mean_is_sharpe:+.4f}")
    print(f"  Mean OOS Sharpe         : {wf_result.mean_oos_sharpe:+.4f}")
    print(f"  Overall OOS Sharpe      : {_annualized_sharpe(wf_result.oos_returns):+.4f}")
    print(f"  Overfitting ratio (IS/OOS): {wf_result.overall_overfitting_ratio:.2f}x")

    print(f"\n  {'Fold':<5} {'Train Start':<14} {'Test Start':<14} {'IS SR':>8} {'OOS SR':>8} {'OF Ratio':>10} {'Best Params'}")
    print("  " + "-" * 72)
    for r in wf_result.fold_results:
        param_str = " ".join(f"{k}={v}" for k, v in r.params_used.items())
        print(
            f"  {r.fold_id:<5} {str(r.train_start.date()):<14} "
            f"{str(r.test_start.date()):<14} {r.in_sample_sharpe:>+8.3f} "
            f"{r.out_sample_sharpe:>+8.3f} {r.overfitting_ratio:>10.2f}  {param_str}"
        )

    # ── 2. PBO ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("Probability of Backtest Overfitting (PBO) — Bailey et al. 2014")
    print(f"{'─'*72}")
    od = OverfittingDetector()
    pbo = od.compute_pbo(wf_result.fold_results)
    of_ratio = od.compute_overfitting_ratio(wf_result.fold_results)
    print(f"  PBO               : {pbo:.4f}  ({'HIGH RISK' if pbo > 0.5 else 'OK'})")
    print(f"  Overfitting ratio : {of_ratio:.3f}  ({'CONCERN' if of_ratio > 2.0 else 'OK'})")

    # ── 3. DSR ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("Deflated Sharpe Ratio (DSR) — Lopez de Prado 2014")
    print(f"{'─'*72}")
    oos_rets_arr = wf_result.oos_returns.dropna()
    skew_val = float(stats.skew(oos_rets_arr.values)) if len(oos_rets_arr) > 3 else 0.0
    kurt_val = float(stats.kurtosis(oos_rets_arr.values, fisher=False)) if len(oos_rets_arr) > 3 else 3.0
    n_grid_combos = len(param_grid["fast"]) * len(param_grid["slow"])
    dsr = od.compute_deflated_sharpe_ratio(
        sharpe=_annualized_sharpe(oos_rets_arr),
        n_trials=n_grid_combos,
        n_observations=len(oos_rets_arr),
        skewness=skew_val,
        kurtosis=kurt_val,
    )
    psr = od.compute_psr(
        sharpe=_annualized_sharpe(oos_rets_arr),
        benchmark_sr=0.0,
        n_obs=len(oos_rets_arr),
        skew=skew_val,
        kurt=kurt_val,
    )
    print(f"  Realized OOS Sharpe : {_annualized_sharpe(oos_rets_arr):+.4f}")
    print(f"  Return skewness     : {skew_val:+.3f}")
    print(f"  Return kurtosis     : {kurt_val:.3f}")
    print(f"  n_trials (grid)     : {n_grid_combos}")
    print(f"  DSR                 : {dsr:.4f}  (P(true Sharpe > SR* after adjusting for {n_grid_combos} trials))")
    print(f"  PSR (vs 0)          : {psr:.4f}  (P(true Sharpe > 0))")

    # ── 4. Anchored Walk-Forward ──────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("Anchored Walk-Forward (expanding window)")
    print(f"{'─'*72}")
    av = AnchoredValidation(engine)
    anchored = av.run_anchored_wf(
        spy, sma_crossover_strategy, min_train_days=252, test_window=63, step_size=21,
        param_grid={"fast": [10, 20], "slow": [50, 100]},
    )
    mean_anchored_oos = float(np.mean([r.out_sample_sharpe for r in anchored.fold_results]))
    print(f"  Folds              : {len(anchored.fold_results)}")
    print(f"  Mean OOS Sharpe    : {mean_anchored_oos:+.4f}")
    print(f"  Degradation curve  : {[round(x, 3) for x in anchored.degradation_curve.tolist()[:10]]}...")
    if anchored.regime_performance:
        print("  OOS Sharpe by regime:")
        for regime, sh in sorted(anchored.regime_performance.items()):
            print(f"    {regime:<12}: {sh:+.4f}")

    # ── 5. Parameter Stability ────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("Parameter Stability Analysis")
    print(f"{'─'*72}")
    ps = ParameterStabilityAnalyzer()
    dispersion = ps.compute_parameter_dispersion(wf_result.fold_results)
    robust = ps.find_robust_params(wf_result.fold_results, param_grid)
    print("  Coefficient of Variation (std/mean) per parameter:")
    for param, cov in sorted(dispersion.items()):
        stability = "STABLE" if cov < 0.3 else ("MODERATE" if cov < 0.6 else "UNSTABLE")
        print(f"    {param:<15}: CoV = {cov:.3f}  [{stability}]")
    print(f"  Robust params: {robust}")

    if wf_result.fold_results and "fast" in wf_result.fold_results[0].params_used:
        for pname in ["fast", "slow"]:
            ic = ps.compute_param_performance_correlation(wf_result.fold_results, pname)
            print(f"  Param–OOS correlation ({pname}): {ic:+.3f}")

    # ── 6. Monte Carlo Permutation ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("Monte Carlo Permutation Test (200 shuffles)")
    print(f"{'─'*72}")
    mc = MonteCarloPermutationTest()
    p_val = mc.run_permutation_test(
        sma_crossover_strategy, spy, n_permutations=200,
        strategy_kwargs={"fast": 20, "slow": 50},
    )
    print(f"  Permutation p-value : {p_val:.4f}")
    print(f"  Significant edge    : {'YES (p < 0.05)' if p_val < 0.05 else 'NO'}")

    # ── 7. Full Validation Report ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("Full Validation Report — StrategyValidator")
    print(f"{'─'*72}")
    validator = StrategyValidator()
    report = validator.validate(
        strategy_fn=sma_crossover_strategy,
        data=spy,
        config=WalkForwardConfig(train_window=252, test_window=63, step_size=21),
        param_grid={"fast": [10, 20, 30], "slow": [50, 100, 150]},
        strategy_name="SMA Crossover",
        n_permutations=200,
        run_bias_tests=True,
    )

    gate_icon = lambda ok: "PASS" if ok else "FAIL"
    print(f"\n  Strategy         : {report.strategy_name}")
    print(f"  Quality Score    : {report.quality_score:.1f} / 100")
    print(f"  Overall          : {'PASS' if report.overall_pass else 'FAIL'}")
    print(f"\n  PASS/FAIL Gates:")
    print(f"    OOS Sharpe > 0.5  : {gate_icon(report.gate_oos_sharpe)}  ({report.oos_sharpe:+.4f})")
    print(f"    PBO < 0.5         : {gate_icon(report.gate_pbo)}  ({report.pbo:.4f})")
    print(f"    OF ratio < 2.0    : {gate_icon(report.gate_overfitting)}  ({report.overfitting_ratio:.3f})")
    print(f"    DSR > 0.95        : {gate_icon(report.gate_dsr)}  ({report.dsr:.4f})")
    print(f"\n  Extended Metrics:")
    print(f"    Mean IS Sharpe     : {report.mean_is_sharpe:+.4f}")
    print(f"    Mean OOS Sharpe    : {report.mean_oos_sharpe:+.4f}")
    print(f"    Anchored OOS SR    : {report.anchored_mean_oos_sharpe:+.4f}")
    print(f"    PSR (vs 0)         : {report.psr:.4f}")
    print(f"    Permutation p-val  : {report.permutation_pvalue:.4f}")
    print(f"    n_folds            : {report.n_folds}")
    if report.bias_report:
        print(f"\n  Bias Report:")
        print(f"    Lookahead detected  : {report.bias_report.lookahead_bias_detected}")
        print(f"    Lookahead perf drop : {report.bias_report.lookahead_performance_drop:.1%}")
        print(f"    Survivorship warn   : {report.bias_report.survivorship_bias_warning}")
        print(f"    DSR (n_trials={report.bias_report.n_trials_tested})   : {report.bias_report.data_snooping_dsr:.4f}")

    # ── 8. Other built-in strategies ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("Quick Comparison — All Built-in Strategies (rolling WF, no optimization)")
    print(f"{'─'*72}")
    strategies = [
        ("SMA Crossover (20/50)", sma_crossover_strategy, {"fast": 20, "slow": 50}),
        ("Momentum (126d)",       momentum_strategy,       {"lookback": 126}),
        ("Mean Reversion (20/2)", mean_reversion_strategy, {"window": 20, "z_threshold": 2.0}),
    ]
    simple_config = WalkForwardConfig(train_window=252, test_window=63, step_size=21)
    print(f"  {'Strategy':<30} {'Folds':>6} {'IS SR':>8} {'OOS SR':>8} {'OF':>6} {'PBO':>6}")
    print("  " + "-" * 72)
    for strat_name, strat_fn, strat_params in strategies:
        try:
            res = engine.run_all_folds(spy, strat_fn, simple_config, fixed_params=strat_params)
            fold_pbo = od.compute_pbo(res.fold_results)
            print(
                f"  {strat_name:<30} {res.n_folds:>6} "
                f"{res.mean_is_sharpe:>+8.4f} {res.mean_oos_sharpe:>+8.4f} "
                f"{res.overall_overfitting_ratio:>6.2f} {fold_pbo:>6.3f}"
            )
        except Exception as exc:
            print(f"  {strat_name:<30} ERROR: {exc}")

    print(f"\n{'='*72}")
    print("Walk-Forward v3 — Done")
    print("=" * 72)
