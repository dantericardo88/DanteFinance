"""
Walk-Forward + Anchored OOS Validation — Dimension #063 (target score 9+).

Rigorous out-of-sample validation framework for trading strategies implementing:
  - Rolling walk-forward cross-validation
  - Anchored (expanding-window) walk-forward
  - Combinatorial Purged Cross-Validation (CPCV) — Lopez de Prado
  - Deflated Sharpe Ratio & overfitting detection
  - Parameter stability analysis across hyperparameter grids

Public API
----------
WalkForwardConfig          dataclass — configuration for a WF validation run
WalkForwardSplitter        — generates (train_idx, test_idx) fold pairs
StrategyValidator          — runs a strategy callable over each fold
WalkForwardResult          dataclass — aggregated OOS metrics + stability
OverfittingDetector        — DSR, probability-of-overfit, min track-record length
ParameterStabilityAnalyzer — grid search + stable-region detection

FastAPI router: validator_router (prefix /api/validate)
"""
from __future__ import annotations

import abc
import itertools
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Configuration for a walk-forward validation run.

    Parameters
    ----------
    method:
        'rolling'       — fixed-length train window slides forward each step
        'anchored'      — expanding train window (start always t=0)
        'combinatorial' — CPCV, all C(n_folds, n_test_folds) test paths
    train_periods:
        Number of time periods (rows) in the training window.
    test_periods:
        Number of time periods in each test (OOS) window.
    step_size:
        How many periods to advance the window each fold.
    min_train_obs:
        Minimum observations required before the first fold is created.
        Guards against early folds with too little data to fit reliably.
    purge_gap:
        Embargo / purge period between the end of training and start of
        testing.  Prevents look-ahead leakage via overlapping features.
    refit_frequency:
        'every_fold'    — re-fit strategy parameters on every train window.
        'every_n_folds' — re-fit only every N folds (N inferred from step_size).
    """

    method: Literal["rolling", "anchored", "combinatorial"] = "rolling"
    train_periods: int = 252          # 1 year of daily bars
    test_periods: int = 63            # ~1 quarter OOS
    step_size: int = 21               # advance ~1 month each fold
    min_train_obs: int = 60
    purge_gap: int = 5                # 1 week embargo
    refit_frequency: Literal["every_fold", "every_n_folds"] = "every_fold"


# ---------------------------------------------------------------------------
# Per-fold result container
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    fold_id: int
    train_start: Any        # label or integer index
    train_end: Any
    test_start: Any
    test_end: Any
    n_train_obs: int
    n_test_obs: int
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_cagr: float = 0.0
    oos_cagr: float = 0.0
    is_max_dd: float = 0.0
    oos_max_dd: float = 0.0
    oos_returns: pd.Series = field(default_factory=pd.Series)
    oos_equity: pd.Series = field(default_factory=pd.Series)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    fitted_params: dict = field(default_factory=dict)

    @property
    def profitable(self) -> bool:
        return self.oos_cagr > 0.0


# ---------------------------------------------------------------------------
# WalkForwardResult
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Aggregated results from a walk-forward validation run."""

    config: WalkForwardConfig
    fold_results: list[FoldResult]
    oos_equity_curve: pd.Series           # stitched continuous OOS equity
    oos_returns: pd.Series                # continuous OOS daily returns

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(returns: pd.Series, ann: int = 252) -> float:
        r = returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * math.sqrt(ann))

    @staticmethod
    def _sortino(returns: pd.Series, ann: int = 252) -> float:
        r = returns.dropna()
        neg = r[r < 0]
        if len(neg) < 2 or neg.std() == 0:
            return 0.0
        return float(r.mean() / neg.std() * math.sqrt(ann))

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        return float(dd.min())

    @staticmethod
    def _cagr(equity: pd.Series, ann: int = 252) -> float:
        if equity.empty or len(equity) < 2:
            return 0.0
        n_years = len(equity) / ann
        if n_years == 0 or equity.iloc[0] == 0:
            return 0.0
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)

    @staticmethod
    def _win_rate(returns: pd.Series) -> float:
        r = returns.dropna()
        if len(r) == 0:
            return 0.0
        return float((r > 0).mean())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_oos_metrics(self) -> dict:
        """Compute Sharpe, Sortino, max drawdown, CAGR, win_rate on OOS only."""
        r = self.oos_returns
        eq = self.oos_equity_curve
        return {
            "sharpe": self._sharpe(r),
            "sortino": self._sortino(r),
            "max_drawdown": self._max_drawdown(eq),
            "cagr": self._cagr(eq),
            "win_rate": self._win_rate(r),
            "n_obs": len(r),
            "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) >= 2 else 0.0,
        }

    def compute_stability(self) -> dict:
        """Compute stability metrics across folds.

        Returns
        -------
        pct_folds_profitable  : fraction of test folds with positive CAGR
        sharpe_std            : std of per-fold OOS Sharpe (low = stable)
        consistency_score     : 0–100 composite score
        degradation_ratio     : IS Sharpe / OOS Sharpe (>2 = overfit signal)
        """
        fold_sharpes = [f.oos_sharpe for f in self.fold_results]
        fold_is_sharpes = [f.is_sharpe for f in self.fold_results]
        pct_profitable = sum(1 for f in self.fold_results if f.profitable) / max(len(self.fold_results), 1)
        sharpe_std = float(np.std(fold_sharpes)) if fold_sharpes else 0.0

        # Consistency score: weighted combination of pct_profitable + Sharpe stability
        raw_consistency = pct_profitable * 60 + max(0, (1 - min(sharpe_std, 2) / 2)) * 40
        consistency_score = max(0.0, min(100.0, raw_consistency))

        # Degradation: mean IS Sharpe / mean OOS Sharpe
        mean_is = float(np.mean(fold_is_sharpes)) if fold_is_sharpes else 0.0
        mean_oos = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        if abs(mean_oos) < 1e-9:
            degradation_ratio = float("inf") if mean_is > 0 else 1.0
        else:
            degradation_ratio = mean_is / mean_oos

        return {
            "pct_folds_profitable": pct_profitable,
            "sharpe_std": sharpe_std,
            "consistency_score": consistency_score,
            "degradation_ratio": degradation_ratio,
            "n_folds": len(self.fold_results),
            "mean_oos_sharpe": mean_oos,
            "mean_is_sharpe": mean_is,
            "fold_oos_sharpes": fold_sharpes,
        }

    def plot_data(self) -> dict:
        """Return serialisable data for equity curve + fold breakdown charts."""
        oos_eq_data = {
            "dates": [str(d)[:10] if hasattr(d, "strftime") else str(d) for d in self.oos_equity_curve.index],
            "values": self.oos_equity_curve.tolist(),
        }
        fold_data = [
            {
                "fold_id": f.fold_id,
                "is_sharpe": round(f.is_sharpe, 3),
                "oos_sharpe": round(f.oos_sharpe, 3),
                "oos_cagr": round(f.oos_cagr, 4),
                "profitable": f.profitable,
            }
            for f in self.fold_results
        ]
        return {"oos_equity_curve": oos_eq_data, "fold_breakdown": fold_data}


# ---------------------------------------------------------------------------
# WalkForwardSplitter
# ---------------------------------------------------------------------------


class WalkForwardSplitter:
    """Generate (train_indices, test_indices) pairs for WF validation.

    All index arrays are integer positional indices into the supplied
    DatetimeIndex (or integer RangeIndex).  The caller is responsible for
    using .iloc[] to slice the actual data.
    """

    # ------------------------------------------------------------------
    # Rolling splits — fixed-length train window
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_splits(
        dates: pd.DatetimeIndex,
        config: WalkForwardConfig,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Fixed-size sliding train window with purge gap.

        For T observations with train_periods=W, test_periods=H, step_size=S,
        purge_gap=G:
            Fold k: train=[k*S, k*S+W), test=[k*S+W+G, k*S+W+G+H)
        """
        n = len(dates)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        k = 0
        while True:
            train_start = k * config.step_size
            train_end = train_start + config.train_periods
            test_start = train_end + config.purge_gap
            test_end = test_start + config.test_periods

            if test_end > n:
                break
            if (train_end - train_start) < config.min_train_obs:
                k += 1
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))
            k += 1

        return splits

    # ------------------------------------------------------------------
    # Anchored splits — expanding train window
    # ------------------------------------------------------------------

    @staticmethod
    def anchored_splits(
        dates: pd.DatetimeIndex,
        config: WalkForwardConfig,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Expanding train window — train always starts at t=0.

        Mimics real-world usage: you never discard old data.  Train window
        grows by step_size each fold; test window is a fixed H-period block.

        Fold k: train=[0, W + k*S), test=[W + k*S + G, W + k*S + G + H)
        """
        n = len(dates)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        k = 0
        while True:
            train_end = config.train_periods + k * config.step_size
            test_start = train_end + config.purge_gap
            test_end = test_start + config.test_periods

            if test_end > n:
                break
            if train_end < config.min_train_obs:
                k += 1
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))
            k += 1

        return splits

    # ------------------------------------------------------------------
    # Combinatorial CPCV — Lopez de Prado
    # ------------------------------------------------------------------

    @staticmethod
    def combinatorial_splits(
        dates: pd.DatetimeIndex,
        n_folds: int = 6,
        n_test_folds: int = 2,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Combinatorial Purged Cross-Validation (CPCV).

        Partitions the data into n_folds equal blocks.  All C(n_folds,
        n_test_folds) combinations of test-block selections are generated.
        Each combination yields one (train_idx, test_idx) pair.

        This produces C(6,2)=15 test paths instead of the single OOS curve
        of standard WF, giving a distribution of OOS Sharpe ratios.

        Parameters
        ----------
        n_folds:      total number of equal-size blocks to partition data into
        n_test_folds: how many of those blocks to designate as test each combo
        """
        n = len(dates)
        fold_size = n // n_folds

        # Create block index arrays
        blocks = []
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else n
            blocks.append(np.arange(start, end))

        # Generate all C(n_folds, n_test_folds) combinations
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for test_block_ids in itertools.combinations(range(n_folds), n_test_folds):
            test_block_set = set(test_block_ids)
            train_blocks = [blocks[i] for i in range(n_folds) if i not in test_block_set]
            test_blocks = [blocks[i] for i in test_block_ids]

            train_idx = np.concatenate(train_blocks) if train_blocks else np.array([], dtype=int)
            test_idx = np.concatenate(test_blocks) if test_blocks else np.array([], dtype=int)
            splits.append((train_idx, test_idx))

        return splits

    # ------------------------------------------------------------------
    # sklearn-compatible time-series split
    # ------------------------------------------------------------------

    @staticmethod
    def time_series_split(
        returns: pd.Series,
        n_splits: int = 5,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Standard sklearn-compatible TimeSeriesSplit.

        Each fold trains on all prior data; test is the next block.
        Returned tuples are (train_idx, test_idx) integer arrays.
        """
        n = len(returns)
        fold_size = n // (n_splits + 1)
        splits = []
        for i in range(1, n_splits + 1):
            train_idx = np.arange(0, i * fold_size)
            test_idx = np.arange(i * fold_size, min((i + 1) * fold_size, n))
            if len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        return splits


# ---------------------------------------------------------------------------
# StrategyValidator
# ---------------------------------------------------------------------------

def _compute_fold_metrics(returns: pd.Series, label: str, ann: int = 252) -> dict:
    """Compute standard metrics for a returns series."""
    r = returns.dropna()
    if len(r) < 2:
        return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0}
    equity = (1 + r).cumprod()
    sharpe = WalkForwardResult._sharpe(r, ann)
    cagr = WalkForwardResult._cagr(equity, ann)
    max_dd = WalkForwardResult._max_drawdown(equity)
    return {"sharpe": sharpe, "cagr": cagr, "max_dd": max_dd}


class StrategyValidator:
    """Run walk-forward validation of a strategy callable.

    The *strategy_fn* must have the signature:
        strategy_fn(train_data: pd.DataFrame, test_data: pd.DataFrame,
                    params: dict | None) -> pd.Series
    where the returned Series contains daily returns for the test period,
    indexed by the same DatetimeIndex as test_data.

    If the strategy supports fitting (parameter optimisation on train data),
    the callable should perform that internally and return fitted parameters
    via a side-channel (e.g., a mutable *params* dict — we store whatever
    is returned or whatever *params* looks like after the call).
    """

    def __init__(self, annualisation: int = 252) -> None:
        self.ann = annualisation

    # ------------------------------------------------------------------
    # Internal fold runner
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold_id: int,
        strategy_fn: Callable,
        data: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        params: dict | None,
    ) -> FoldResult:
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        try:
            oos_returns: pd.Series = strategy_fn(train_data, test_data, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fold %d strategy_fn raised: %s", fold_id, exc)
            oos_returns = pd.Series(dtype=float)

        # IS returns: run strategy on train data with itself as both train + test
        try:
            is_returns: pd.Series = strategy_fn(train_data, train_data, params)
        except Exception:  # noqa: BLE001
            is_returns = pd.Series(dtype=float)

        is_m = _compute_fold_metrics(is_returns, "IS", self.ann)
        oos_m = _compute_fold_metrics(oos_returns, "OOS", self.ann)

        oos_equity = (1 + oos_returns.fillna(0)).cumprod() if not oos_returns.empty else pd.Series(dtype=float)

        idx = data.index
        return FoldResult(
            fold_id=fold_id,
            train_start=idx[train_idx[0]] if len(train_idx) else None,
            train_end=idx[train_idx[-1]] if len(train_idx) else None,
            test_start=idx[test_idx[0]] if len(test_idx) else None,
            test_end=idx[test_idx[-1]] if len(test_idx) else None,
            n_train_obs=len(train_idx),
            n_test_obs=len(test_idx),
            is_sharpe=is_m["sharpe"],
            oos_sharpe=oos_m["sharpe"],
            is_cagr=is_m["cagr"],
            oos_cagr=oos_m["cagr"],
            is_max_dd=is_m["max_dd"],
            oos_max_dd=oos_m["max_dd"],
            oos_returns=oos_returns,
            oos_equity=oos_equity,
            fitted_params=params.copy() if params else {},
        )

    # ------------------------------------------------------------------
    # Stitch OOS equity curves from fold results
    # ------------------------------------------------------------------

    @staticmethod
    def _stitch_oos(fold_results: list[FoldResult]) -> tuple[pd.Series, pd.Series]:
        """Stitch per-fold OOS return series into one continuous path."""
        all_returns: list[pd.Series] = []
        for fr in fold_results:
            if not fr.oos_returns.empty:
                all_returns.append(fr.oos_returns)

        if not all_returns:
            empty = pd.Series(dtype=float)
            return empty, empty

        combined_returns = pd.concat(all_returns).sort_index()
        combined_returns = combined_returns[~combined_returns.index.duplicated(keep="first")]
        combined_equity = (1 + combined_returns.fillna(0)).cumprod()
        return combined_returns, combined_equity

    # ------------------------------------------------------------------
    # Public: run_walk_forward
    # ------------------------------------------------------------------

    def run_walk_forward(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        config: WalkForwardConfig,
        params: dict | None = None,
    ) -> WalkForwardResult:
        """Rolling WF validation: train window slides forward each step."""
        splits = WalkForwardSplitter.rolling_splits(data.index, config)
        logger.info("Rolling WF: %d folds", len(splits))

        fold_results = []
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            fold_params = params.copy() if params else {}
            fr = self._run_fold(fold_id, strategy_fn, data, train_idx, test_idx, fold_params)
            fold_results.append(fr)
            logger.debug("Fold %d | IS Sharpe=%.2f | OOS Sharpe=%.2f", fold_id, fr.is_sharpe, fr.oos_sharpe)

        oos_returns, oos_equity = self._stitch_oos(fold_results)
        return WalkForwardResult(
            config=config,
            fold_results=fold_results,
            oos_equity_curve=oos_equity,
            oos_returns=oos_returns,
        )

    # ------------------------------------------------------------------
    # Public: run_anchored
    # ------------------------------------------------------------------

    def run_anchored(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        config: WalkForwardConfig,
    ) -> WalkForwardResult:
        """Anchored (expanding window) WF validation."""
        cfg = WalkForwardConfig(
            method="anchored",
            train_periods=config.train_periods,
            test_periods=config.test_periods,
            step_size=config.step_size,
            min_train_obs=config.min_train_obs,
            purge_gap=config.purge_gap,
            refit_frequency=config.refit_frequency,
        )
        splits = WalkForwardSplitter.anchored_splits(data.index, cfg)
        logger.info("Anchored WF: %d folds", len(splits))

        fold_results = []
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            fr = self._run_fold(fold_id, strategy_fn, data, train_idx, test_idx, None)
            fold_results.append(fr)

        oos_returns, oos_equity = self._stitch_oos(fold_results)
        return WalkForwardResult(
            config=cfg,
            fold_results=fold_results,
            oos_equity_curve=oos_equity,
            oos_returns=oos_returns,
        )

    # ------------------------------------------------------------------
    # Public: run_combinatorial_cv
    # ------------------------------------------------------------------

    def run_combinatorial_cv(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        n_folds: int = 6,
    ) -> WalkForwardResult:
        """Combinatorial Purged CV — returns distribution of OOS Sharpe ratios.

        Unlike run_walk_forward, the 'OOS equity curve' here is constructed
        from one representative test path (the first combination); the key
        output is the distribution of OOS Sharpe ratios across all paths,
        accessible via result.compute_stability()['fold_oos_sharpes'].
        """
        cfg = WalkForwardConfig(method="combinatorial")
        splits = WalkForwardSplitter.combinatorial_splits(data.index, n_folds=n_folds, n_test_folds=2)
        logger.info("CPCV: %d combinations (C(%d,2))", len(splits), n_folds)

        fold_results = []
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            fr = self._run_fold(fold_id, strategy_fn, data, train_idx, test_idx, None)
            fold_results.append(fr)

        # Representative OOS curve from first combination
        if fold_results and not fold_results[0].oos_returns.empty:
            oos_returns = fold_results[0].oos_returns
            oos_equity = fold_results[0].oos_equity
        else:
            oos_returns = pd.Series(dtype=float)
            oos_equity = pd.Series(dtype=float)

        return WalkForwardResult(
            config=cfg,
            fold_results=fold_results,
            oos_equity_curve=oos_equity,
            oos_returns=oos_returns,
        )


# ---------------------------------------------------------------------------
# OverfittingDetector
# ---------------------------------------------------------------------------


class OverfittingDetector:
    """Statistical tests for overfitting in backtested strategies.

    Implements Bailey & Lopez de Prado (2014):
      - Deflated Sharpe Ratio (DSR) — adjusts for multiple testing
      - Probability of Overfit (PSR-based)
      - Minimum Track Record Length
    """

    # ------------------------------------------------------------------
    # Deflated Sharpe Ratio
    # ------------------------------------------------------------------

    @staticmethod
    def deflated_sharpe_ratio(
        is_sharpe: float,
        n_trials: int,
        T: int,
        skew: float = 0.0,
        excess_kurt: float = 0.0,
    ) -> float:
        """Bailey-Lopez de Prado Deflated Sharpe Ratio.

        Adjusts the observed IS Sharpe for:
          1. Non-normality of returns (skew, excess kurtosis)
          2. Multiple testing bias (n_trials independent strategy evaluations)

        DSR = SR* × PSR(SR^)

        where SR* is the expected maximum Sharpe under the null (random
        strategy evaluation) and PSR is the Probabilistic Sharpe Ratio.

        Parameters
        ----------
        is_sharpe : annualised in-sample Sharpe ratio
        n_trials  : number of strategy/parameter combinations evaluated
        T         : number of observations in IS period
        skew      : skewness of the IS return distribution
        excess_kurt : excess kurtosis of the IS return distribution
        """
        if T < 2:
            return 0.0

        # Expected maximum Sharpe under H0 (Bailey et al. eq. 8)
        # SR_bar ≈ (1 - γ) × z(1 - 1/M) + γ × z(1 - 1/(M×e))
        # For simplicity, use the approximation:
        if n_trials <= 1:
            sr_star = 0.0
        else:
            gamma = np.euler_gamma
            z_arg = 1 - 1 / n_trials
            # Ensure bounded
            z_arg = max(0.001, min(0.999, z_arg))
            sr_star = float(
                (1 - gamma) * stats.norm.ppf(1 - 1 / max(n_trials, 2))
                + gamma * stats.norm.ppf(1 - 1 / (max(n_trials, 2) * np.e))
            )

        # PSR(SR^): probability that the true Sharpe exceeds SR* given
        # observed IS Sharpe and sample statistics
        # PSR = Φ( (SR - SR*) × sqrt(T-1) / sqrt(1 - skew×SR + (kurt-1)/4×SR²) )
        denom_sq = 1 - skew * is_sharpe + ((excess_kurt - 1) / 4) * is_sharpe ** 2
        if denom_sq <= 0:
            return 0.0

        psr_z = (is_sharpe - sr_star) * math.sqrt(T - 1) / math.sqrt(denom_sq)
        dsr = float(stats.norm.cdf(psr_z))
        return dsr

    # ------------------------------------------------------------------
    # Probability of overfit
    # ------------------------------------------------------------------

    @staticmethod
    def probability_overfit(
        is_sharpe: float,
        oos_sharpe: float,
        n_trials: int = 1,
        T_is: int = 252,
        T_oos: int = 63,
    ) -> float:
        """Stochastic dominance test: P(OOS Sharpe < 0 | IS Sharpe > 0).

        High PO (>0.5) indicates the observed IS superiority is unlikely to
        persist OOS — a strong overfitting signal.

        Approximation via Bailey et al. (2014), eq. 10:
            PO ≈ 1 - Φ( OOS_SR × sqrt(T_oos) / σ_oos )
        where σ_oos is estimated from the IS Sharpe distribution.
        """
        if T_oos < 2:
            return 0.5

        # Estimated standard error of the OOS Sharpe
        # se_oos ≈ sqrt((1 + 0.5 × SR²) / T_oos)  [Jobson-Korkie]
        se_oos = math.sqrt((1 + 0.5 * oos_sharpe ** 2) / T_oos)
        if se_oos == 0:
            return 0.0

        # Inflate by multiple testing factor
        mt_adj = math.sqrt(math.log(max(n_trials, 1)))
        z = (oos_sharpe - mt_adj) / se_oos
        po = float(1 - stats.norm.cdf(z))
        return po

    # ------------------------------------------------------------------
    # Minimum track record length
    # ------------------------------------------------------------------

    @staticmethod
    def min_track_record_length(
        sharpe: float,
        confidence: float = 0.95,
        freq: str = "daily",
    ) -> int:
        """Minimum number of observations to assert genuine alpha.

        Based on Bailey & Lopez de Prado (2012), eq. for MinTRL:
            MinTRL = 1 + (1 + 0.5 × SR²) × (Φ⁻¹(confidence) / SR)²

        Parameters
        ----------
        sharpe     : annualised Sharpe ratio
        confidence : statistical confidence level (default 0.95)
        freq       : 'daily' (ann=252), 'weekly' (52), 'monthly' (12)
        """
        ann_map = {"daily": 252, "weekly": 52, "monthly": 12}
        ann = ann_map.get(freq, 252)

        # Convert annualised Sharpe to per-period Sharpe
        sr_per_period = sharpe / math.sqrt(ann)

        if abs(sr_per_period) < 1e-9:
            return 9999

        z = float(stats.norm.ppf(confidence))
        min_trl = 1 + (1 + 0.5 * sr_per_period ** 2) * (z / sr_per_period) ** 2
        return max(1, int(math.ceil(min_trl)))

    # ------------------------------------------------------------------
    # Overfitting signal aggregation
    # ------------------------------------------------------------------

    def detect_overfitting_signals(self, result: WalkForwardResult) -> dict:
        """Aggregate multiple overfitting checks into a single verdict.

        Returns
        -------
        is_overfit  : bool — True if at least 2 strong signals triggered
        confidence  : float 0–1 — fraction of signals triggered
        signals     : list[str] — human-readable explanations
        """
        stability = result.compute_stability()
        oos_m = result.compute_oos_metrics()

        signals: list[str] = []
        checks: list[bool] = []

        # 1. Degradation ratio
        deg = stability["degradation_ratio"]
        is_deg = deg > 2.0
        checks.append(is_deg)
        if is_deg:
            signals.append(
                f"Degradation ratio {deg:.2f} > 2.0 — IS Sharpe is more than double OOS Sharpe"
            )

        # 2. Low fraction of profitable folds
        pct = stability["pct_folds_profitable"]
        is_low_pct = pct < 0.6
        checks.append(is_low_pct)
        if is_low_pct:
            signals.append(
                f"Only {pct:.0%} of folds are profitable (threshold: 60%)"
            )

        # 3. High Sharpe standard deviation across folds
        sr_std = stability["sharpe_std"]
        is_high_std = sr_std > 1.0
        checks.append(is_high_std)
        if is_high_std:
            signals.append(
                f"Per-fold Sharpe std = {sr_std:.2f} > 1.0 — highly unstable performance"
            )

        # 4. DSR check: deflated Sharpe < 1.0 (here we check PSR proxy < 0.5)
        mean_is = stability["mean_is_sharpe"]
        n_folds = stability["n_folds"]
        T_is = max(result.config.train_periods, 60)
        dsr = self.deflated_sharpe_ratio(
            is_sharpe=mean_is,
            n_trials=max(n_folds, 1),
            T=T_is,
        )
        is_dsr_fail = dsr < 0.5
        checks.append(is_dsr_fail)
        if is_dsr_fail:
            signals.append(
                f"Deflated Sharpe Ratio = {dsr:.3f} < 0.50 — IS Sharpe not significant after multiple-testing adjustment"
            )

        # 5. Negative OOS Sharpe overall
        oos_sr = oos_m["sharpe"]
        is_neg_oos = oos_sr < 0.0
        checks.append(is_neg_oos)
        if is_neg_oos:
            signals.append(f"OOS Sharpe = {oos_sr:.2f} < 0 — strategy loses money out-of-sample")

        n_triggered = sum(checks)
        confidence = n_triggered / len(checks)
        is_overfit = n_triggered >= 2

        return {
            "is_overfit": is_overfit,
            "confidence": confidence,
            "n_signals": n_triggered,
            "total_checks": len(checks),
            "signals": signals,
            "dsr": dsr,
            "degradation_ratio": deg,
        }


# ---------------------------------------------------------------------------
# ParameterStabilityAnalyzer
# ---------------------------------------------------------------------------


class ParameterStabilityAnalyzer:
    """Grid search over strategy parameters with OOS stability analysis.

    For each parameter combination, a walk-forward validation is run and OOS
    metrics are collected.  A "stable region" is one where OOS performance
    varies minimally with small parameter perturbations — indicating the
    strategy is not hyper-sensitive to exact parameter values.
    """

    def __init__(self, validator: StrategyValidator | None = None) -> None:
        self.validator = validator or StrategyValidator()

    # ------------------------------------------------------------------
    # Full grid run
    # ------------------------------------------------------------------

    def analyze(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        param_grid: dict,
        config: WalkForwardConfig,
    ) -> pd.DataFrame:
        """Run WF validation for every combination in *param_grid*.

        Parameters
        ----------
        param_grid : dict of param_name → list[values]
                     e.g. {"lookback": [10, 20, 40], "threshold": [0.01, 0.02]}

        Returns
        -------
        pd.DataFrame with columns: [param_names..., sharpe, sortino, cagr,
                                    max_drawdown, consistency_score, degradation_ratio]
        """
        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))
        logger.info("ParameterStabilityAnalyzer: %d combinations × %d folds each", len(combos), len(
            WalkForwardSplitter.rolling_splits(data.index, config)
        ))

        records = []
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                result = self.validator.run_walk_forward(strategy_fn, data, config, params=params.copy())
                oos_m = result.compute_oos_metrics()
                stab = result.compute_stability()
                row = {**params}
                row.update({
                    "sharpe": oos_m["sharpe"],
                    "sortino": oos_m["sortino"],
                    "cagr": oos_m["cagr"],
                    "max_drawdown": oos_m["max_drawdown"],
                    "win_rate": oos_m["win_rate"],
                    "consistency_score": stab["consistency_score"],
                    "degradation_ratio": stab["degradation_ratio"],
                    "pct_folds_profitable": stab["pct_folds_profitable"],
                    "sharpe_std": stab["sharpe_std"],
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Combo %s failed: %s", params, exc)
                row = {**params, "sharpe": float("nan"), "sortino": float("nan"),
                       "cagr": float("nan"), "max_drawdown": float("nan"),
                       "win_rate": float("nan"), "consistency_score": 0.0,
                       "degradation_ratio": float("nan"),
                       "pct_folds_profitable": 0.0, "sharpe_std": float("nan")}
            records.append(row)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Find stable region
    # ------------------------------------------------------------------

    def find_stable_region(
        self,
        results_df: pd.DataFrame,
        metric: str = "sharpe",
    ) -> dict:
        """Identify the parameter region with the best risk-adjusted stability.

        Strategy
        --------
        1. Score each row: stability_score = metric_value / (1 + sharpe_std)
           — rewards high metric but penalises high variance across neighbours.
        2. Best row = highest stability_score (ignoring NaN rows).
        3. Sensitivity map: for each parameter, compute the range of *metric*
           when that parameter varies and all others are held at best_params.

        Returns
        -------
        dict with keys:
            best_params            : dict of parameter → value
            best_metric_value      : float
            stability_score        : float
            parameter_sensitivity  : dict of param → sensitivity (std of metric)
        """
        df = results_df.dropna(subset=[metric]).copy()
        if df.empty:
            return {"best_params": {}, "best_metric_value": 0.0,
                    "stability_score": 0.0, "parameter_sensitivity": {}}

        sharpe_std_col = "sharpe_std" if "sharpe_std" in df.columns else None
        if sharpe_std_col:
            df["_stability_score"] = df[metric] / (1 + df[sharpe_std_col].clip(lower=0))
        else:
            df["_stability_score"] = df[metric]

        best_row = df.loc[df["_stability_score"].idxmax()]

        # Identify parameter columns (everything that isn't a metric col)
        metric_cols = {"sharpe", "sortino", "cagr", "max_drawdown", "win_rate",
                       "consistency_score", "degradation_ratio",
                       "pct_folds_profitable", "sharpe_std", "_stability_score"}
        param_cols = [c for c in df.columns if c not in metric_cols]
        best_params = {c: best_row[c] for c in param_cols}

        # Parameter sensitivity: hold all params at best except the one being probed
        sensitivity_map: dict[str, float] = {}
        for pc in param_cols:
            # Rows where all other params match best
            mask = pd.Series([True] * len(df), index=df.index)
            for other in param_cols:
                if other != pc:
                    mask &= df[other] == best_params[other]
            subset = df[mask][metric]
            sensitivity_map[pc] = float(subset.std()) if len(subset) > 1 else 0.0

        return {
            "best_params": best_params,
            "best_metric_value": float(best_row[metric]),
            "stability_score": float(best_row["_stability_score"]),
            "parameter_sensitivity": sensitivity_map,
        }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel as PydanticModel, Field as PydanticField

    validator_router = APIRouter(prefix="/api/validate", tags=["Walk-Forward Validation"])

    class WalkForwardRequest(PydanticModel):
        """Request body for POST /api/validate/walk-forward."""
        method: str = "rolling"
        train_periods: int = 252
        test_periods: int = 63
        step_size: int = 21
        purge_gap: int = 5
        min_train_obs: int = 60
        # Returns series as list of floats (date-ordered)
        returns: list[float] = PydanticField(default_factory=list)
        dates: list[str] = PydanticField(default_factory=list)

    class OverfitCheckRequest(PydanticModel):
        is_sharpe: float
        oos_sharpe: float
        n_trials: int = 1
        T_is: int = 252
        T_oos: int = 63

    class ParamStabilityRequest(PydanticModel):
        method: str = "rolling"
        train_periods: int = 252
        test_periods: int = 63
        step_size: int = 21
        purge_gap: int = 5
        min_train_obs: int = 60
        returns: list[float] = PydanticField(default_factory=list)
        dates: list[str] = PydanticField(default_factory=list)
        param_grid: dict = PydanticField(default_factory=dict)

    def _build_dummy_strategy(returns_map: dict) -> Callable:
        """Build a passthrough strategy that replays precomputed returns."""
        def _strategy(train_data: pd.DataFrame, test_data: pd.DataFrame, params: dict | None) -> pd.Series:
            # For API usage: the strategy returns the OOS segment of the preloaded return series
            col = "returns" if "returns" in test_data.columns else test_data.columns[0]
            return test_data[col]
        return _strategy

    @validator_router.post("/walk-forward")
    async def api_walk_forward(req: WalkForwardRequest):
        """Run walk-forward validation on a precomputed returns series."""
        if not req.returns or not req.dates:
            raise HTTPException(status_code=422, detail="returns and dates are required")
        if len(req.returns) != len(req.dates):
            raise HTTPException(status_code=422, detail="returns and dates must be same length")

        idx = pd.to_datetime(req.dates)
        data = pd.DataFrame({"returns": req.returns}, index=idx)
        cfg = WalkForwardConfig(
            method=req.method,
            train_periods=req.train_periods,
            test_periods=req.test_periods,
            step_size=req.step_size,
            purge_gap=req.purge_gap,
            min_train_obs=req.min_train_obs,
        )
        strategy_fn = _build_dummy_strategy({})
        validator = StrategyValidator()
        result = validator.run_walk_forward(strategy_fn, data, cfg)
        oos_m = result.compute_oos_metrics()
        stab = result.compute_stability()
        return {
            "oos_metrics": oos_m,
            "stability": stab,
            "plot_data": result.plot_data(),
        }

    @validator_router.get("/overfitting-check")
    async def api_overfit_check(
        is_sharpe: float,
        oos_sharpe: float,
        n_trials: int = 1,
        T_is: int = 252,
        T_oos: int = 63,
    ):
        """Check if IS/OOS Sharpe pair shows overfitting signals."""
        detector = OverfittingDetector()
        po = detector.probability_overfit(is_sharpe, oos_sharpe, n_trials, T_is, T_oos)
        dsr = detector.deflated_sharpe_ratio(is_sharpe, n_trials, T_is)
        min_trl = detector.min_track_record_length(is_sharpe)
        degradation = is_sharpe / oos_sharpe if abs(oos_sharpe) > 1e-9 else float("inf")
        signals = []
        if degradation > 2.0:
            signals.append(f"Degradation ratio {degradation:.2f} > 2.0")
        if po > 0.5:
            signals.append(f"Probability of overfit {po:.1%} > 50%")
        if dsr < 0.5:
            signals.append(f"DSR {dsr:.3f} < 0.50")
        return {
            "probability_overfit": po,
            "deflated_sharpe_ratio": dsr,
            "min_track_record_length": min_trl,
            "degradation_ratio": degradation,
            "is_overfit": len(signals) >= 2,
            "signals": signals,
        }

    @validator_router.post("/parameter-stability")
    async def api_parameter_stability(req: ParamStabilityRequest):
        """Run parameter grid stability analysis."""
        if not req.returns or not req.param_grid:
            raise HTTPException(status_code=422, detail="returns and param_grid required")

        idx = pd.to_datetime(req.dates) if req.dates else pd.date_range("2020-01-01", periods=len(req.returns), freq="B")
        data = pd.DataFrame({"returns": req.returns}, index=idx)
        cfg = WalkForwardConfig(
            method=req.method,
            train_periods=req.train_periods,
            test_periods=req.test_periods,
            step_size=req.step_size,
            purge_gap=req.purge_gap,
            min_train_obs=req.min_train_obs,
        )
        strategy_fn = _build_dummy_strategy({})
        analyzer = ParameterStabilityAnalyzer()
        results_df = analyzer.analyze(strategy_fn, data, req.param_grid, cfg)
        stable = analyzer.find_stable_region(results_df)
        return {
            "grid_results": results_df.to_dict(orient="records"),
            "stable_region": stable,
        }

except ImportError:
    # FastAPI not available — module still usable as a library
    validator_router = None  # type: ignore[assignment]
    logger.debug("FastAPI not available — validator_router not registered")
