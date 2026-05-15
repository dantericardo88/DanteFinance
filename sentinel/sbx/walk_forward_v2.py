"""
Walk-Forward + Anchored OOS Validation v2 — Dimension #063 (target score 9).

Builds on walk_forward_validator.py with:
  - AnchoredWalkForwardEngine: anchored/rolling/hybrid + optimal window detection
  - RegimeConditionalValidation: per-regime performance validation
  - ParameterStabilityTestV2: 2D heatmaps, sensitivity ranking, adaptive WF
  - StatisticalSignificanceTester: White's Reality Check, Romano-Wolf, Welch t-test,
    bootstrap confidence intervals

FastAPI router: wf_v2_router
  POST /walkforward/v2/run
  POST /walkforward/v2/regime-test
  POST /walkforward/v2/param-stability
  POST /walkforward/v2/significance
"""
from __future__ import annotations

import abc
import itertools
import math
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sentinel.core.logging import get_logger

# Re-export from base module for convenience
from sentinel.sbx.walk_forward_validator import (
    FoldResult,
    OverfittingDetector,
    ParameterStabilityAnalyzer,
    StrategyValidator,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSplitter,
    _compute_fold_metrics,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

def classify_regime(
    returns: pd.Series,
    volatility_window: int = 60,
    trend_window: int = 120,
    crisis_vol_quantile: float = 0.85,
) -> pd.Series:
    """
    Classify each bar into a market regime.

    Regimes:
      'bull'      — rising market, moderate volatility
      'bear'      — falling market, moderate volatility
      'sideways'  — flat market, low volatility
      'crisis'    — extreme volatility (top quantile)

    Parameters
    ----------
    returns            : daily returns series (DatetimeIndex)
    volatility_window  : lookback for rolling volatility
    trend_window       : lookback for rolling mean return (trend)
    crisis_vol_quantile: percentile threshold for crisis regime
    """
    r = returns.dropna()
    roll_ret = r.rolling(trend_window, min_periods=max(20, trend_window // 3)).mean() * 252
    roll_vol = r.rolling(volatility_window, min_periods=max(20, volatility_window // 3)).std() * math.sqrt(252)

    vol_threshold = float(roll_vol.quantile(crisis_vol_quantile)) if roll_vol.dropna().shape[0] > 10 else 0.30

    regime = pd.Series("sideways", index=r.index, dtype=str)

    crisis_mask = roll_vol >= vol_threshold
    bull_mask = (~crisis_mask) & (roll_ret >= 0.05)
    bear_mask = (~crisis_mask) & (roll_ret <= -0.05)

    regime[bull_mask] = "bull"
    regime[bear_mask] = "bear"
    regime[crisis_mask] = "crisis"
    regime[~bull_mask & ~bear_mask & ~crisis_mask] = "sideways"

    return regime


# ---------------------------------------------------------------------------
# AnchoredWalkForwardEngine
# ---------------------------------------------------------------------------

@dataclass
class AnchoredWFConfig:
    """Configuration for anchored walk-forward with hybrid support."""
    wf_type: Literal["anchored", "rolling", "hybrid"] = "anchored"
    min_train_periods: int = 252        # minimum training window (anchored and hybrid)
    max_train_periods: int = 1260       # max training periods (rolling cap for hybrid)
    test_periods: int = 63             # OOS test window size
    step_size: int = 21                # advance per fold
    purge_gap: int = 5                 # embargo gap
    detect_optimal_window: bool = True  # run IC vs window curve
    ic_lookback_range: list[int] = field(
        default_factory=lambda: [63, 126, 189, 252, 378, 504]
    )


@dataclass
class AnchoredWFResult:
    """Extended result from AnchoredWalkForwardEngine."""
    config: AnchoredWFConfig
    fold_results: list[FoldResult]
    oos_equity_curve: pd.Series
    oos_returns: pd.Series
    optimal_window: int | None          # detected optimal training window
    ic_vs_window: pd.DataFrame | None   # IC vs window length curve
    window_type_used: str               # which window type was actually used


class AnchoredWalkForwardEngine:
    """
    Enhanced anchored walk-forward with window optimisation.

    Supports three window modes:
      anchored  — expanding window (train always from start)
      rolling   — fixed-size window sliding forward
      hybrid    — anchored but capped at max_train_periods (rolls after cap)

    Also detects the optimal training window by computing IC vs window length.
    """

    def __init__(self, ann: int = 252) -> None:
        self.ann = ann

    # ------------------------------------------------------------------
    # Split generators
    # ------------------------------------------------------------------

    def _anchored_splits(self, n: int, cfg: AnchoredWFConfig) -> list[tuple[np.ndarray, np.ndarray]]:
        splits = []
        k = 0
        while True:
            train_end = cfg.min_train_periods + k * cfg.step_size
            test_start = train_end + cfg.purge_gap
            test_end = test_start + cfg.test_periods
            if test_end > n:
                break
            if train_end < cfg.min_train_periods:
                k += 1
                continue
            splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
            k += 1
        return splits

    def _rolling_splits(self, n: int, cfg: AnchoredWFConfig) -> list[tuple[np.ndarray, np.ndarray]]:
        splits = []
        k = 0
        train_size = cfg.min_train_periods
        while True:
            train_start = k * cfg.step_size
            train_end = train_start + train_size
            test_start = train_end + cfg.purge_gap
            test_end = test_start + cfg.test_periods
            if test_end > n:
                break
            if train_end - train_start < cfg.min_train_periods:
                k += 1
                continue
            splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
            k += 1
        return splits

    def _hybrid_splits(self, n: int, cfg: AnchoredWFConfig) -> list[tuple[np.ndarray, np.ndarray]]:
        """Anchored up to max_train_periods, then rolls forward."""
        splits = []
        k = 0
        while True:
            anchor_end = cfg.min_train_periods + k * cfg.step_size
            # Cap the training window
            if anchor_end <= cfg.max_train_periods:
                train_start = 0
                train_end = anchor_end
            else:
                # Roll: start moves forward to maintain max_train_periods
                excess = anchor_end - cfg.max_train_periods
                train_start = excess
                train_end = anchor_end

            test_start = anchor_end + cfg.purge_gap
            test_end = test_start + cfg.test_periods
            if test_end > n:
                break
            if train_end - train_start < cfg.min_train_periods:
                k += 1
                continue
            splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
            k += 1
        return splits

    def _compute_ic_vs_window(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        lookback_range: list[int],
        test_periods: int,
        purge_gap: int,
    ) -> pd.DataFrame:
        """
        Compute Information Coefficient (IC) vs training window length.

        For each training window size, runs a small set of rolling folds and
        records the rank correlation between IS and OOS Sharpe (IC proxy).

        Returns DataFrame: columns [window, mean_oos_sharpe, ic, n_folds].
        """
        n = len(data)
        records = []

        for lb in lookback_range:
            if lb + purge_gap + test_periods > n:
                continue

            # Run 3 test folds for this window size
            fold_is_sharpes = []
            fold_oos_sharpes = []
            k = 0
            folds_done = 0
            while folds_done < 3:
                train_start = k * (lb // 4)
                train_end = train_start + lb
                test_start = train_end + purge_gap
                test_end = test_start + test_periods
                if test_end > n:
                    break
                try:
                    train_d = data.iloc[train_start:train_end]
                    test_d = data.iloc[test_start:test_end]
                    oos_rets = strategy_fn(train_d, test_d, None)
                    is_rets = strategy_fn(train_d, train_d, None)

                    is_m = _compute_fold_metrics(is_rets, "IS", self.ann)
                    oos_m = _compute_fold_metrics(oos_rets, "OOS", self.ann)
                    fold_is_sharpes.append(is_m["sharpe"])
                    fold_oos_sharpes.append(oos_m["sharpe"])
                    folds_done += 1
                except Exception:
                    pass
                k += 1

            if len(fold_oos_sharpes) >= 2:
                mean_oos = float(np.mean(fold_oos_sharpes))
                if len(fold_is_sharpes) >= 2 and np.std(fold_is_sharpes) > 0 and np.std(fold_oos_sharpes) > 0:
                    ic = float(np.corrcoef(fold_is_sharpes, fold_oos_sharpes)[0, 1])
                else:
                    ic = 0.0
                records.append({
                    "window": lb,
                    "mean_oos_sharpe": round(mean_oos, 4),
                    "ic": round(ic, 4),
                    "n_folds": len(fold_oos_sharpes),
                })

        return pd.DataFrame(records)

    def detect_optimal_window(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        cfg: AnchoredWFConfig,
    ) -> int:
        """
        Find the training window with the best OOS performance.

        Uses IC vs window curve: picks the window with highest
        (mean_oos_sharpe × (1 + ic)) — rewards both performance and IS/OOS consistency.
        """
        ic_curve = self._compute_ic_vs_window(
            strategy_fn, data, cfg.ic_lookback_range,
            cfg.test_periods, cfg.purge_gap
        )
        if ic_curve.empty:
            return cfg.min_train_periods

        ic_curve["score"] = ic_curve["mean_oos_sharpe"] * (1 + ic_curve["ic"].clip(lower=0))
        best_row = ic_curve.loc[ic_curve["score"].idxmax()]
        return int(best_row["window"])

    def run(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        cfg: AnchoredWFConfig,
        params: dict | None = None,
    ) -> AnchoredWFResult:
        """
        Run anchored/rolling/hybrid walk-forward validation.

        strategy_fn signature: (train_data, test_data, params) -> pd.Series (returns)
        """
        n = len(data)
        optimal_window = None
        ic_curve_df = None

        # Detect optimal window if requested
        if cfg.detect_optimal_window and cfg.wf_type in ("anchored", "rolling", "hybrid"):
            try:
                optimal_window = self.detect_optimal_window(strategy_fn, data, cfg)
                logger.info("Optimal training window detected: %d bars", optimal_window)
                ic_curve_df = self._compute_ic_vs_window(
                    strategy_fn, data, cfg.ic_lookback_range, cfg.test_periods, cfg.purge_gap
                )
            except Exception as exc:
                logger.warning("Optimal window detection failed: %s", exc)
                optimal_window = cfg.min_train_periods

        # Adjust config for optimal window
        effective_cfg = AnchoredWFConfig(
            wf_type=cfg.wf_type,
            min_train_periods=optimal_window or cfg.min_train_periods,
            max_train_periods=cfg.max_train_periods,
            test_periods=cfg.test_periods,
            step_size=cfg.step_size,
            purge_gap=cfg.purge_gap,
            detect_optimal_window=False,
            ic_lookback_range=cfg.ic_lookback_range,
        )

        # Generate splits
        if cfg.wf_type == "anchored":
            splits = self._anchored_splits(n, effective_cfg)
            window_type_used = "anchored"
        elif cfg.wf_type == "rolling":
            splits = self._rolling_splits(n, effective_cfg)
            window_type_used = "rolling"
        else:  # hybrid
            splits = self._hybrid_splits(n, effective_cfg)
            window_type_used = "hybrid"

        logger.info("AnchoredWF[%s]: %d folds, window=%d", window_type_used, len(splits), effective_cfg.min_train_periods)

        fold_results = []
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            fold_params = params.copy() if params else {}

            try:
                oos_rets = strategy_fn(train_data, test_data, fold_params)
            except Exception as exc:
                logger.warning("Fold %d OOS failed: %s", fold_id, exc)
                oos_rets = pd.Series(dtype=float)

            try:
                is_rets = strategy_fn(train_data, train_data, fold_params)
            except Exception:
                is_rets = pd.Series(dtype=float)

            is_m = _compute_fold_metrics(is_rets, "IS", self.ann)
            oos_m = _compute_fold_metrics(oos_rets, "OOS", self.ann)
            oos_eq = (1 + oos_rets.fillna(0)).cumprod() if not oos_rets.empty else pd.Series(dtype=float)

            idx = data.index
            fr = FoldResult(
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
                oos_returns=oos_rets,
                oos_equity=oos_eq,
                fitted_params=fold_params,
            )
            fold_results.append(fr)

        # Stitch OOS
        all_rets = [fr.oos_returns for fr in fold_results if not fr.oos_returns.empty]
        if all_rets:
            oos_returns = pd.concat(all_rets).sort_index()
            oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
            oos_equity = (1 + oos_returns.fillna(0)).cumprod()
        else:
            oos_returns = pd.Series(dtype=float)
            oos_equity = pd.Series(dtype=float)

        return AnchoredWFResult(
            config=cfg,
            fold_results=fold_results,
            oos_equity_curve=oos_equity,
            oos_returns=oos_returns,
            optimal_window=optimal_window,
            ic_vs_window=ic_curve_df,
            window_type_used=window_type_used,
        )


# ---------------------------------------------------------------------------
# RegimeConditionalValidation
# ---------------------------------------------------------------------------

@dataclass
class RegimeReport:
    """Per-regime performance breakdown."""
    regime: str
    n_observations: int
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    win_rate: float
    total_return: float
    is_significant: bool        # t-test: mean return > 0 at 95% confidence

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "n_observations": self.n_observations,
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "cagr": round(self.cagr, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "total_return": round(self.total_return, 4),
            "is_significant": self.is_significant,
        }


@dataclass
class RegimeConditionalResult:
    """Full result from regime-conditional validation."""
    regime_reports: dict[str, RegimeReport]
    overall_sharpe: float
    regime_dependence_score: float   # 0–100: how regime-specific is the strategy
    dominant_regime: str             # regime with best Sharpe
    regime_coverage: dict[str, float]  # fraction of time in each regime
    pairwise_sharpe_diff: dict[str, float]  # regime pairs → Sharpe difference
    is_regime_specific: bool         # True if strategy only works in 1-2 regimes


class RegimeConditionalValidation:
    """
    Validate strategy performance separately in each market regime.

    Splits the full backtest period into regime sub-periods and evaluates
    strategy returns within each. Identifies regime-specific strategies.

    Usage:
        rcv = RegimeConditionalValidation()
        result = rcv.validate_by_regime(strategy_returns, market_returns)
    """

    def __init__(self, ann: int = 252) -> None:
        self.ann = ann

    @staticmethod
    def _sharpe(r: pd.Series, ann: int = 252) -> float:
        r = r.dropna()
        if len(r) < 5 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * math.sqrt(ann))

    @staticmethod
    def _sortino(r: pd.Series, ann: int = 252) -> float:
        r = r.dropna()
        neg = r[r < 0]
        if len(neg) < 3 or neg.std() == 0:
            return 0.0
        return float(r.mean() / neg.std() * math.sqrt(ann))

    @staticmethod
    def _cagr(r: pd.Series, ann: int = 252) -> float:
        r = r.dropna()
        if len(r) < 5:
            return 0.0
        equity = (1 + r).cumprod()
        n_years = len(r) / ann
        if n_years <= 0 or equity.iloc[0] <= 0:
            return 0.0
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)

    @staticmethod
    def _max_drawdown(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 2:
            return 0.0
        equity = (1 + r).cumprod()
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        return float(dd.min())

    @staticmethod
    def _is_significant(r: pd.Series, confidence: float = 0.95) -> bool:
        """One-sample t-test: is mean return > 0 at given confidence?"""
        r = r.dropna()
        if len(r) < 10:
            return False
        t_stat, p_val = stats.ttest_1samp(r, 0.0)
        return bool(t_stat > 0 and p_val / 2 < (1 - confidence))

    def validate_by_regime(
        self,
        strategy_returns: pd.Series,
        regime_classifier: pd.Series | None = None,
        market_returns: pd.Series | None = None,
        volatility_window: int = 60,
        trend_window: int = 120,
    ) -> RegimeConditionalResult:
        """
        Evaluate strategy returns split by market regime.

        Parameters
        ----------
        strategy_returns   : daily strategy returns (DatetimeIndex)
        regime_classifier  : pre-computed regime series (same index); if None,
                             derived from market_returns
        market_returns     : benchmark returns for regime classification (e.g. SPY)
                             required if regime_classifier is None
        """
        # Build regime series
        if regime_classifier is not None:
            regime = regime_classifier
        elif market_returns is not None:
            regime = classify_regime(market_returns, volatility_window, trend_window)
        else:
            # Fall back to classifying the strategy returns themselves
            regime = classify_regime(strategy_returns, volatility_window, trend_window)

        # Align indices
        common_idx = strategy_returns.index.intersection(regime.index)
        strategy_returns = strategy_returns.loc[common_idx]
        regime = regime.loc[common_idx]

        if strategy_returns.empty:
            return RegimeConditionalResult(
                regime_reports={},
                overall_sharpe=0.0,
                regime_dependence_score=0.0,
                dominant_regime="none",
                regime_coverage={},
                pairwise_sharpe_diff={},
                is_regime_specific=False,
            )

        all_regimes = ["bull", "bear", "sideways", "crisis"]
        regime_reports: dict[str, RegimeReport] = {}
        regime_coverage: dict[str, float] = {}
        n_total = len(strategy_returns)

        for reg in all_regimes:
            mask = regime == reg
            reg_rets = strategy_returns[mask]
            n_obs = len(reg_rets)
            coverage = n_obs / n_total if n_total > 0 else 0.0
            regime_coverage[reg] = round(coverage, 4)

            if n_obs < 5:
                regime_reports[reg] = RegimeReport(
                    regime=reg,
                    n_observations=n_obs,
                    sharpe=0.0, sortino=0.0, cagr=0.0, max_drawdown=0.0,
                    win_rate=0.0, total_return=0.0, is_significant=False,
                )
                continue

            equity_sub = (1 + reg_rets.fillna(0)).cumprod()
            total_ret = float(equity_sub.iloc[-1] - 1) if len(equity_sub) > 0 else 0.0
            win_rate = float((reg_rets > 0).mean())

            regime_reports[reg] = RegimeReport(
                regime=reg,
                n_observations=n_obs,
                sharpe=self._sharpe(reg_rets, self.ann),
                sortino=self._sortino(reg_rets, self.ann),
                cagr=self._cagr(reg_rets, self.ann),
                max_drawdown=self._max_drawdown(reg_rets),
                win_rate=win_rate,
                total_return=total_ret,
                is_significant=self._is_significant(reg_rets),
            )

        # Overall Sharpe
        overall_sharpe = self._sharpe(strategy_returns, self.ann)

        # Dominant regime
        sharpe_by_regime = {reg: rr.sharpe for reg, rr in regime_reports.items() if rr.n_observations >= 5}
        dominant_regime = max(sharpe_by_regime, key=sharpe_by_regime.get) if sharpe_by_regime else "none"

        # Pairwise Sharpe differences
        pairwise: dict[str, float] = {}
        regime_list = list(sharpe_by_regime.keys())
        for i in range(len(regime_list)):
            for j in range(i + 1, len(regime_list)):
                r1, r2 = regime_list[i], regime_list[j]
                diff = sharpe_by_regime[r1] - sharpe_by_regime[r2]
                pairwise[f"{r1}_vs_{r2}"] = round(diff, 3)

        # Regime dependence score
        sharpe_vals = [v for v in sharpe_by_regime.values() if not math.isnan(v)]
        if len(sharpe_vals) >= 2:
            sharpe_std = float(np.std(sharpe_vals))
            sharpe_range = max(sharpe_vals) - min(sharpe_vals)
            # High std/range → regime specific
            dependence_score = min(100.0, sharpe_std * 30 + sharpe_range * 10)
        else:
            dependence_score = 0.0

        # Strategy is regime-specific if it significantly outperforms in ≤2 regimes
        positive_regimes = sum(1 for v in sharpe_by_regime.values() if v > 0.5)
        is_regime_specific = (positive_regimes <= 2 and len(sharpe_by_regime) >= 3)

        return RegimeConditionalResult(
            regime_reports=regime_reports,
            overall_sharpe=round(overall_sharpe, 3),
            regime_dependence_score=round(dependence_score, 1),
            dominant_regime=dominant_regime,
            regime_coverage=regime_coverage,
            pairwise_sharpe_diff=pairwise,
            is_regime_specific=is_regime_specific,
        )


# ---------------------------------------------------------------------------
# ParameterStabilityTestV2
# ---------------------------------------------------------------------------

@dataclass
class ParamStabilityResult:
    """Result from ParameterStabilityTestV2."""
    grid_results: pd.DataFrame
    heatmap_data: dict              # 2D Sharpe heatmap (param1 × param2)
    sensitivity_ranking: list[dict] # params ranked by impact on performance
    robustness_scores: dict[str, float]  # parameter → robustness (0–100)
    cliff_edge_params: list[str]    # parameters with cliff-edge degradation
    best_params: dict
    best_sharpe: float
    adaptive_params: dict | None    # AdaWF time-varying best params


class ParameterStabilityTestV2:
    """
    Enhanced parameter stability with 2D heatmaps, sensitivity ranking,
    robustness scoring, and adaptive walk-forward (AdaWF).

    AdaWF: re-optimises parameters at each fold using the preceding
    training window → time-varying "adaptive" parameter set.
    """

    def __init__(self, validator: StrategyValidator | None = None, ann: int = 252) -> None:
        self.validator = validator or StrategyValidator(annualisation=ann)
        self.ann = ann

    # ------------------------------------------------------------------
    # Full grid search
    # ------------------------------------------------------------------

    def run_grid(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        param_grid: dict,
        config: WalkForwardConfig,
    ) -> pd.DataFrame:
        """
        Run walk-forward validation for every parameter combination.

        Returns DataFrame with columns: [params..., sharpe, sortino, cagr,
        max_drawdown, win_rate, consistency_score, degradation_ratio, sharpe_std].
        """
        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))
        logger.info("ParamStabilityV2 grid: %d combinations", len(combos))

        records = []
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                result = self.validator.run_walk_forward(strategy_fn, data, config, params=params.copy())
                oos_m = result.compute_oos_metrics()
                stab = result.compute_stability()
                row = dict(params)
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
                    "n_folds": stab["n_folds"],
                })
            except Exception as exc:
                logger.warning("Combo %s failed: %s", params, exc)
                row = dict(params)
                row.update({
                    "sharpe": float("nan"), "sortino": float("nan"),
                    "cagr": float("nan"), "max_drawdown": float("nan"),
                    "win_rate": float("nan"), "consistency_score": 0.0,
                    "degradation_ratio": float("nan"), "pct_folds_profitable": 0.0,
                    "sharpe_std": float("nan"), "n_folds": 0,
                })
            records.append(row)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 2D Heatmap
    # ------------------------------------------------------------------

    def build_heatmap(
        self,
        grid_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = "sharpe",
    ) -> dict:
        """
        Build 2D heatmap data for param1 × param2.

        Returns dict with:
          - x_values: sorted unique values of param2
          - y_values: sorted unique values of param1
          - matrix: list of lists (param1 index, param2 index → metric value)
        """
        if param1 not in grid_df.columns or param2 not in grid_df.columns:
            return {"error": f"Params {param1}, {param2} not in grid results"}

        # Average over any other parameters
        pivot = grid_df.groupby([param1, param2])[metric].mean().unstack(param2)
        y_vals = list(pivot.index)
        x_vals = list(pivot.columns)
        matrix = pivot.values.tolist()

        return {
            "x_param": param2,
            "y_param": param1,
            "x_values": [str(v) for v in x_vals],
            "y_values": [str(v) for v in y_vals],
            "matrix": [[round(v, 4) if not math.isnan(v) else None for v in row] for row in matrix],
            "metric": metric,
        }

    # ------------------------------------------------------------------
    # Sensitivity ranking
    # ------------------------------------------------------------------

    def rank_sensitivity(
        self,
        grid_df: pd.DataFrame,
        metric: str = "sharpe",
    ) -> list[dict]:
        """
        Rank parameters by their impact on OOS performance.

        Method: For each parameter p, compute the variance of *metric* as p varies
        (holding other params fixed at their modal values).
        Higher variance → more sensitive → ranks higher.

        Returns list of dicts sorted by sensitivity (descending).
        """
        metric_cols = {
            "sharpe", "sortino", "cagr", "max_drawdown", "win_rate",
            "consistency_score", "degradation_ratio", "pct_folds_profitable",
            "sharpe_std", "n_folds",
        }
        param_cols = [c for c in grid_df.columns if c not in metric_cols]
        clean_df = grid_df.dropna(subset=[metric])

        ranking = []
        for pc in param_cols:
            vals = clean_df[pc].unique()
            if len(vals) <= 1:
                ranking.append({"param": pc, "sensitivity": 0.0, "range": 0.0, "n_values": 1})
                continue

            # Compute metric variance as this param changes
            group_means = clean_df.groupby(pc)[metric].mean()
            sensitivity = float(group_means.std()) if len(group_means) > 1 else 0.0
            metric_range = float(group_means.max() - group_means.min()) if len(group_means) > 1 else 0.0
            ranking.append({
                "param": pc,
                "sensitivity": round(sensitivity, 4),
                "range": round(metric_range, 4),
                "n_values": len(vals),
                "best_value": group_means.idxmax() if not group_means.empty else None,
                "worst_value": group_means.idxmin() if not group_means.empty else None,
            })

        ranking.sort(key=lambda x: x["sensitivity"], reverse=True)
        return ranking

    # ------------------------------------------------------------------
    # Robustness scoring
    # ------------------------------------------------------------------

    def compute_robustness(
        self,
        grid_df: pd.DataFrame,
        best_params: dict,
        metric: str = "sharpe",
        n_neighbors: int = 2,
    ) -> dict[str, float]:
        """
        Compute robustness score per parameter.

        For each parameter, measure how much performance degrades as we move
        n_neighbors steps away from the best value. High robustness means
        graceful degradation (not a cliff-edge).

        Returns dict: param → robustness_score (0–100, higher = more robust).
        """
        metric_cols = {
            "sharpe", "sortino", "cagr", "max_drawdown", "win_rate",
            "consistency_score", "degradation_ratio", "pct_folds_profitable",
            "sharpe_std", "n_folds",
        }
        param_cols = [c for c in best_params.keys() if c not in metric_cols]
        clean_df = grid_df.dropna(subset=[metric])
        robustness: dict[str, float] = {}

        for pc in param_cols:
            sorted_vals = sorted(grid_df[pc].unique())
            if len(sorted_vals) <= 2:
                robustness[pc] = 100.0  # only 1-2 values, can't test
                continue

            best_val = best_params.get(pc)
            if best_val not in sorted_vals:
                robustness[pc] = 50.0
                continue

            best_idx = sorted_vals.index(best_val)
            # Get metric at best and at ±n_neighbors positions
            neighbor_metrics = []
            for delta in range(1, n_neighbors + 1):
                for direction in [-1, 1]:
                    neighbor_idx = best_idx + direction * delta
                    if 0 <= neighbor_idx < len(sorted_vals):
                        neighbor_val = sorted_vals[neighbor_idx]
                        # Filter: best_params but vary this param
                        mask = pd.Series([True] * len(clean_df), index=clean_df.index)
                        for other in param_cols:
                            if other != pc and other in best_params:
                                mask &= (clean_df[other] == best_params[other])
                        mask &= clean_df[pc] == neighbor_val
                        sub = clean_df[mask][metric]
                        if not sub.empty:
                            neighbor_metrics.append(float(sub.mean()))

            if not neighbor_metrics:
                robustness[pc] = 50.0
                continue

            # Get best metric value
            best_mask = pd.Series([True] * len(clean_df), index=clean_df.index)
            for other in param_cols:
                if other in best_params:
                    best_mask &= (clean_df[other] == best_params[other])
            best_metric = float(clean_df[best_mask][metric].mean()) if not clean_df[best_mask].empty else 0.0

            if abs(best_metric) < 1e-9:
                robustness[pc] = 50.0
                continue

            avg_neighbor = float(np.mean(neighbor_metrics))
            # Robustness: how well neighbors perform relative to best
            score = max(0.0, min(100.0, (avg_neighbor / best_metric) * 100))
            robustness[pc] = round(score, 1)

        return robustness

    # ------------------------------------------------------------------
    # Cliff-edge detection
    # ------------------------------------------------------------------

    def detect_cliff_edges(
        self,
        grid_df: pd.DataFrame,
        metric: str = "sharpe",
        cliff_threshold: float = 0.5,
    ) -> list[str]:
        """
        Detect parameters where performance has cliff-edge behaviour.

        A cliff-edge occurs when moving one step in a parameter causes
        performance to drop by more than cliff_threshold × best_performance.

        Returns list of parameter names with cliff-edge behaviour.
        """
        metric_cols = {
            "sharpe", "sortino", "cagr", "max_drawdown", "win_rate",
            "consistency_score", "degradation_ratio", "pct_folds_profitable",
            "sharpe_std", "n_folds",
        }
        param_cols = [c for c in grid_df.columns if c not in metric_cols]
        clean_df = grid_df.dropna(subset=[metric])
        cliff_params = []

        for pc in param_cols:
            group_means = clean_df.groupby(pc)[metric].mean().sort_index()
            if len(group_means) < 3:
                continue
            vals = group_means.values
            diffs = np.abs(np.diff(vals))
            best_val = float(group_means.max())
            if best_val == 0:
                continue
            # Cliff-edge: any step where drop > cliff_threshold × best
            if any(d > cliff_threshold * abs(best_val) for d in diffs):
                cliff_params.append(pc)

        return cliff_params

    # ------------------------------------------------------------------
    # Adaptive Walk-Forward (AdaWF)
    # ------------------------------------------------------------------

    def adaptive_walk_forward(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        param_grid: dict,
        config: WalkForwardConfig,
        inner_train_pct: float = 0.7,
    ) -> dict:
        """
        Adaptive Walk-Forward (AdaWF): re-optimise parameters at each fold.

        For each OOS test fold:
          1. Split the training window into inner-train (70%) and inner-val (30%)
          2. Grid search over param_grid using inner-train/val
          3. Use best inner-val params to trade the OOS test fold

        Returns dict with:
          - oos_returns: stitched OOS returns with adaptive params
          - param_history: list of {fold, params, inner_val_sharpe, oos_sharpe}
          - oos_sharpe: overall OOS Sharpe
          - stability: how often the same params were chosen
        """
        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))

        # Get outer folds (train+test windows)
        outer_cfg = WalkForwardConfig(
            method=config.method,
            train_periods=config.train_periods,
            test_periods=config.test_periods,
            step_size=config.step_size,
            min_train_obs=config.min_train_obs,
            purge_gap=config.purge_gap,
        )
        splits = WalkForwardSplitter.rolling_splits(data.index, outer_cfg)
        logger.info("AdaWF: %d outer folds × %d param combos", len(splits), len(combos))

        oos_rets_list = []
        param_history = []
        param_counts: dict[str, int] = {}

        for fold_id, (train_idx, test_idx) in enumerate(splits):
            train_data = data.iloc[train_idx]

            # Inner split
            n_inner = len(train_idx)
            inner_train_end = int(n_inner * inner_train_pct)
            inner_val_start = inner_train_end + config.purge_gap
            inner_train = train_data.iloc[:inner_train_end]
            inner_val = train_data.iloc[inner_val_start:] if inner_val_start < n_inner else train_data.iloc[inner_train_end:]

            # Inner grid search
            best_inner_sharpe = -np.inf
            best_combo = dict(zip(keys, combos[0]))
            for combo in combos:
                p = dict(zip(keys, combo))
                try:
                    val_rets = strategy_fn(inner_train, inner_val, p.copy())
                    if val_rets.empty or val_rets.std() == 0:
                        sh = 0.0
                    else:
                        sh = float(val_rets.mean() / val_rets.std() * math.sqrt(self.ann))
                except Exception:
                    sh = -np.inf

                if sh > best_inner_sharpe:
                    best_inner_sharpe = sh
                    best_combo = p

            # Trade OOS with best inner params
            test_data = data.iloc[test_idx]
            try:
                oos_rets = strategy_fn(train_data, test_data, best_combo.copy())
                oos_m = _compute_fold_metrics(oos_rets, "OOS", self.ann)
                oos_sh = oos_m["sharpe"]
                if not oos_rets.empty:
                    oos_rets_list.append(oos_rets)
            except Exception as exc:
                logger.warning("AdaWF fold %d OOS failed: %s", fold_id, exc)
                oos_sh = 0.0

            combo_key = str(sorted(best_combo.items()))
            param_counts[combo_key] = param_counts.get(combo_key, 0) + 1

            param_history.append({
                "fold": fold_id,
                "params": best_combo,
                "inner_val_sharpe": round(best_inner_sharpe, 3),
                "oos_sharpe": round(oos_sh, 3),
            })

        # Stitch OOS
        if oos_rets_list:
            combined_oos = pd.concat(oos_rets_list).sort_index()
            combined_oos = combined_oos[~combined_oos.index.duplicated(keep="first")]
            oos_sharpe = float(combined_oos.mean() / combined_oos.std() * math.sqrt(self.ann)) if combined_oos.std() > 0 else 0.0
        else:
            combined_oos = pd.Series(dtype=float)
            oos_sharpe = 0.0

        # Stability: how often was the same combo chosen?
        n_folds_done = len(param_history)
        most_chosen = max(param_counts.values()) if param_counts else 0
        stability_pct = most_chosen / n_folds_done if n_folds_done > 0 else 0.0

        return {
            "oos_returns": combined_oos,
            "param_history": param_history,
            "oos_sharpe": round(oos_sharpe, 3),
            "n_folds": n_folds_done,
            "param_stability_pct": round(stability_pct, 3),
            "most_common_params": max(param_counts, key=param_counts.get) if param_counts else None,
        }

    # ------------------------------------------------------------------
    # Combined stability analysis
    # ------------------------------------------------------------------

    def full_stability_analysis(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        param_grid: dict,
        config: WalkForwardConfig,
        param1: str | None = None,
        param2: str | None = None,
        metric: str = "sharpe",
    ) -> ParamStabilityResult:
        """Run full parameter stability suite and return consolidated result."""
        grid_df = self.run_grid(strategy_fn, data, param_grid, config)

        # Best params
        clean = grid_df.dropna(subset=[metric])
        if clean.empty:
            best_params = {}
            best_sharpe = 0.0
        else:
            best_row = clean.loc[clean[metric].idxmax()]
            metric_cols = {
                "sharpe", "sortino", "cagr", "max_drawdown", "win_rate",
                "consistency_score", "degradation_ratio", "pct_folds_profitable",
                "sharpe_std", "n_folds",
            }
            param_cols = [c for c in grid_df.columns if c not in metric_cols]
            best_params = {c: best_row[c] for c in param_cols}
            best_sharpe = float(best_row[metric])

        # 2D Heatmap (use first two params if not specified)
        keys = list(param_grid.keys())
        p1 = param1 or (keys[0] if keys else None)
        p2 = param2 or (keys[1] if len(keys) > 1 else None)
        heatmap = self.build_heatmap(grid_df, p1, p2, metric) if p1 and p2 else {}

        sensitivity = self.rank_sensitivity(grid_df, metric)
        robustness = self.compute_robustness(grid_df, best_params, metric)
        cliff_edges = self.detect_cliff_edges(grid_df, metric)

        return ParamStabilityResult(
            grid_results=grid_df,
            heatmap_data=heatmap,
            sensitivity_ranking=sensitivity,
            robustness_scores=robustness,
            cliff_edge_params=cliff_edges,
            best_params=best_params,
            best_sharpe=best_sharpe,
            adaptive_params=None,
        )


# ---------------------------------------------------------------------------
# StatisticalSignificanceTester
# ---------------------------------------------------------------------------

@dataclass
class SignificanceResult:
    """Results from formal significance testing."""
    # White's Reality Check
    white_rc_p_value: float
    white_rc_significant: bool     # p < 0.05

    # Romano-Wolf stepdown
    romano_wolf_rejected: list[int]   # indices of strategies that rejected H0
    romano_wolf_p_values: list[float]

    # Welch t-test
    welch_t_stat: float
    welch_p_value: float
    welch_significant: bool        # mean OOS return > 0 at 95%

    # Bootstrap CIs
    sharpe_ci: tuple[float, float, float]    # (p5, median, p95)
    cagr_ci: tuple[float, float, float]
    max_dd_ci: tuple[float, float, float]
    sortino_ci: tuple[float, float, float]

    # Summary
    overall_significant: bool
    n_bootstrap: int

    def to_dict(self) -> dict:
        return {
            "white_reality_check": {
                "p_value": round(self.white_rc_p_value, 4),
                "significant_at_5pct": self.white_rc_significant,
            },
            "romano_wolf": {
                "rejected_strategy_indices": self.romano_wolf_rejected,
                "p_values": [round(p, 4) for p in self.romano_wolf_p_values],
            },
            "welch_t_test": {
                "t_statistic": round(self.welch_t_stat, 4),
                "p_value": round(self.welch_p_value, 4),
                "significant_at_5pct": self.welch_significant,
            },
            "bootstrap_confidence_intervals": {
                "sharpe_ratio": {
                    "p5": self.sharpe_ci[0], "median": self.sharpe_ci[1], "p95": self.sharpe_ci[2]
                },
                "cagr": {
                    "p5": self.cagr_ci[0], "median": self.cagr_ci[1], "p95": self.cagr_ci[2]
                },
                "max_drawdown": {
                    "p5": self.max_dd_ci[0], "median": self.max_dd_ci[1], "p95": self.max_dd_ci[2]
                },
                "sortino_ratio": {
                    "p5": self.sortino_ci[0], "median": self.sortino_ci[1], "p95": self.sortino_ci[2]
                },
            },
            "overall_significant": self.overall_significant,
            "n_bootstrap": self.n_bootstrap,
        }


class StatisticalSignificanceTester:
    """
    Formal statistical significance testing for strategy performance.

    Implements:
    1. White's Reality Check — bootstrap multiple-testing correction
    2. Romano-Wolf stepdown — controls FWER across strategy universe
    3. Welch's t-test — OOS Sharpe > 0 test
    4. Bootstrap confidence intervals — all key metrics

    References:
    - White (2000): "A Reality Check for Data Snooping"
    - Romano & Wolf (2005): "Stepwise Multiple Testing as Formalized Data Snooping"
    - Jobson & Korkie (1981): Sharpe ratio variance formula
    """

    def __init__(self, n_bootstrap: int = 1000, seed: int | None = 42, ann: int = 252) -> None:
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.ann = ann
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # White's Reality Check
    # ------------------------------------------------------------------

    def whites_reality_check(
        self,
        benchmark_returns: pd.Series,
        strategy_returns_list: list[pd.Series],
        n_bootstrap: int | None = None,
    ) -> float:
        """
        White's Reality Check p-value.

        H0: No strategy among the tested set has positive expected return
            beyond the benchmark.

        Implementation:
          1. Compute excess returns f_k = strategy_k - benchmark for each strategy k
          2. Compute test statistic V = max_k E[f_k]
          3. Bootstrap: resample f_k, compute V* each time
          4. p-value = P(V* > V)

        Parameters
        ----------
        benchmark_returns     : daily benchmark returns (or zeros for absolute)
        strategy_returns_list : list of strategy daily returns (each pd.Series)
        n_bootstrap           : number of bootstrap iterations

        Returns
        -------
        p_value : float (0–1). Small p-value → reject H0 → at least one strategy is genuine.
        """
        n_bs = n_bootstrap or self.n_bootstrap

        if not strategy_returns_list:
            return 1.0

        # Align all series to common index
        all_series = [benchmark_returns] + strategy_returns_list
        common_idx = all_series[0].index
        for s in all_series[1:]:
            common_idx = common_idx.intersection(s.index)

        bench = benchmark_returns.loc[common_idx].fillna(0).values
        strats = [s.loc[common_idx].fillna(0).values for s in strategy_returns_list]

        # Excess returns
        excess = np.column_stack([s - bench for s in strats])  # T × K
        T, K = excess.shape

        # Observed test statistic: max mean excess return across strategies
        f_bar = excess.mean(axis=0)
        V_obs = float(np.max(f_bar))

        # Block bootstrap preserving serial correlation (block size ≈ sqrt(T))
        block_size = max(1, int(math.sqrt(T)))
        n_blocks = math.ceil(T / block_size)

        boot_V = np.zeros(n_bs)
        for b in range(n_bs):
            # Draw block starts
            block_starts = self.rng.integers(0, T - block_size + 1, size=n_blocks)
            boot_indices = np.concatenate([np.arange(s, min(s + block_size, T)) for s in block_starts])[:T]
            boot_excess = excess[boot_indices, :]
            f_boot = boot_excess.mean(axis=0)
            # Centred bootstrap statistic
            V_boot = float(np.max(f_boot - f_bar))
            boot_V[b] = V_boot

        p_value = float(np.mean(boot_V > V_obs))
        return p_value

    # ------------------------------------------------------------------
    # Romano-Wolf stepdown
    # ------------------------------------------------------------------

    def romano_wolf_stepdown(
        self,
        benchmark_returns: pd.Series,
        strategy_returns_list: list[pd.Series],
        alpha: float = 0.05,
        n_bootstrap: int | None = None,
    ) -> tuple[list[int], list[float]]:
        """
        Romano-Wolf stepdown procedure for strong FWER control.

        Steps:
          1. Sort strategies by mean excess return (best first)
          2. Test most promising strategy against block-bootstrap null
          3. Remove if rejected, adjust critical value, continue

        Returns
        -------
        rejected_indices : list of strategy indices (0-based) that reject H0
        adjusted_p_values: list of adjusted p-values per strategy (same order as input)
        """
        n_bs = n_bootstrap or self.n_bootstrap

        if not strategy_returns_list:
            return [], []

        # Align
        common_idx = benchmark_returns.index
        for s in strategy_returns_list:
            common_idx = common_idx.intersection(s.index)

        bench = benchmark_returns.loc[common_idx].fillna(0).values
        strats = [s.loc[common_idx].fillna(0).values for s in strategy_returns_list]
        excess = np.column_stack([s - bench for s in strats])
        T, K = excess.shape
        f_bar = excess.mean(axis=0)

        # Sort by mean excess return (descending)
        sort_order = np.argsort(-f_bar)

        # Block bootstrap
        block_size = max(1, int(math.sqrt(T)))
        n_blocks = math.ceil(T / block_size)

        # Pre-compute bootstrap distributions for all strategies
        boot_max = np.zeros((n_bs, K))
        for b in range(n_bs):
            block_starts = self.rng.integers(0, T - block_size + 1, size=n_blocks)
            boot_idx = np.concatenate([np.arange(s, min(s + block_size, T)) for s in block_starts])[:T]
            f_boot = excess[boot_idx, :].mean(axis=0) - f_bar
            # Stepdown: use max across remaining strategies
            for j in range(K):
                remaining = sort_order[j:]
                boot_max[b, j] = float(np.max(f_boot[remaining])) if len(remaining) > 0 else 0.0

        # Stepdown testing
        rejected_indices = []
        adjusted_p_values = [1.0] * K

        for step in range(K):
            idx = sort_order[step]
            # p-value = fraction of bootstrap stats exceeding observed
            p = float(np.mean(boot_max[:, step] > f_bar[idx]))
            adjusted_p_values[idx] = p
            if p <= alpha:
                rejected_indices.append(int(idx))
            else:
                break  # stepdown: stop at first non-rejection

        return rejected_indices, adjusted_p_values

    # ------------------------------------------------------------------
    # Welch's t-test for OOS Sharpe > 0
    # ------------------------------------------------------------------

    def welch_t_test(self, oos_returns: pd.Series) -> tuple[float, float, bool]:
        """
        Welch's t-test: H0: E[OOS return] = 0, H1: E[OOS return] > 0.

        More robust than standard t-test for unequal variances.
        Also accounts for autocorrelation using Newey-West se.

        Returns (t_stat, p_value, significant_at_5pct).
        """
        r = oos_returns.dropna()
        if len(r) < 10:
            return 0.0, 1.0, False

        # Newey-West standard error (corrects for autocorrelation)
        n = len(r)
        lags = int(math.sqrt(n))
        var_r = float(r.var())

        # Compute Newey-West variance
        nw_var = var_r
        for lag in range(1, lags + 1):
            weight = 1 - lag / (lags + 1)
            autocov = float(pd.Series(r.values[lag:] * r.values[:-lag]).mean())
            nw_var += 2 * weight * autocov

        if nw_var <= 0:
            nw_var = var_r

        se = math.sqrt(max(nw_var, 1e-12) / n)
        t_stat = float(r.mean()) / se if se > 0 else 0.0
        # One-sided p-value
        p_value = float(stats.t.sf(t_stat, df=n - 1))
        significant = p_value < 0.05

        return round(t_stat, 4), round(p_value, 4), significant

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        oos_returns: pd.Series,
        metric_fn: Callable,
        lo: float = 5.0,
        hi: float = 95.0,
    ) -> tuple[float, float, float]:
        """
        Bootstrap confidence interval for a scalar metric.

        Parameters
        ----------
        metric_fn : function(returns: np.ndarray) → float

        Returns (p_lo, median, p_hi).
        """
        r = oos_returns.dropna().values
        if len(r) < 10:
            return 0.0, 0.0, 0.0

        boot_metrics = np.zeros(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            sample = self.rng.choice(r, size=len(r), replace=True)
            try:
                boot_metrics[b] = metric_fn(sample)
            except Exception:
                boot_metrics[b] = 0.0

        return (
            round(float(np.percentile(boot_metrics, lo)), 4),
            round(float(np.median(boot_metrics)), 4),
            round(float(np.percentile(boot_metrics, hi)), 4),
        )

    def _sharpe_fn(self, r: np.ndarray) -> float:
        if r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * math.sqrt(self.ann))

    def _sortino_fn(self, r: np.ndarray) -> float:
        neg = r[r < 0]
        if len(neg) < 2 or neg.std() == 0:
            return 0.0
        return float(r.mean() / neg.std() * math.sqrt(self.ann))

    def _cagr_fn(self, r: np.ndarray) -> float:
        if len(r) == 0:
            return 0.0
        eq = np.cumprod(1 + r)
        n_years = len(r) / self.ann
        if n_years <= 0 or eq[0] <= 0:
            return 0.0
        return float((eq[-1] / eq[0]) ** (1 / n_years) - 1)

    def _max_dd_fn(self, r: np.ndarray) -> float:
        if len(r) == 0:
            return 0.0
        eq = np.cumprod(1 + r)
        roll_max = np.maximum.accumulate(eq)
        dd = (eq - roll_max) / roll_max
        return float(dd.min())

    # ------------------------------------------------------------------
    # Full test suite
    # ------------------------------------------------------------------

    def run_full_significance_test(
        self,
        oos_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        additional_strategies: list[pd.Series] | None = None,
    ) -> SignificanceResult:
        """
        Run the complete significance testing suite.

        Parameters
        ----------
        oos_returns           : primary strategy's OOS returns
        benchmark_returns     : benchmark (e.g. SPY returns); zeros if None
        additional_strategies : other strategies to include in White's RC / Romano-Wolf
        """
        r = oos_returns.dropna()

        # Build benchmark
        if benchmark_returns is not None:
            bench = benchmark_returns.reindex(r.index).fillna(0)
        else:
            bench = pd.Series(0.0, index=r.index)

        # Strategy list for White's RC and Romano-Wolf
        all_strats = [r]
        if additional_strategies:
            for s in additional_strategies:
                aligned = s.reindex(r.index).fillna(0)
                all_strats.append(aligned)

        # White's Reality Check
        try:
            white_p = self.whites_reality_check(bench, all_strats)
        except Exception as exc:
            logger.warning("White's RC failed: %s", exc)
            white_p = 1.0

        # Romano-Wolf
        try:
            rw_rejected, rw_p_vals = self.romano_wolf_stepdown(bench, all_strats)
        except Exception as exc:
            logger.warning("Romano-Wolf failed: %s", exc)
            rw_rejected, rw_p_vals = [], [1.0] * len(all_strats)

        # Welch t-test
        t_stat, p_val, welch_sig = self.welch_t_test(r)

        # Bootstrap CIs
        sharpe_ci = self.bootstrap_ci(r, self._sharpe_fn)
        cagr_ci = self.bootstrap_ci(r, self._cagr_fn)
        max_dd_ci = self.bootstrap_ci(r, self._max_dd_fn)
        sortino_ci = self.bootstrap_ci(r, self._sortino_fn)

        # Overall significance: at least 2 of 3 tests agree
        white_sig = white_p < 0.05
        rw_sig = 0 in rw_rejected  # primary strategy (index 0) rejected H0
        n_sig = sum([white_sig, rw_sig, welch_sig])
        overall_sig = n_sig >= 2

        return SignificanceResult(
            white_rc_p_value=round(white_p, 4),
            white_rc_significant=white_sig,
            romano_wolf_rejected=rw_rejected,
            romano_wolf_p_values=[round(p, 4) for p in rw_p_vals],
            welch_t_stat=t_stat,
            welch_p_value=p_val,
            welch_significant=welch_sig,
            sharpe_ci=sharpe_ci,
            cagr_ci=cagr_ci,
            max_dd_ci=max_dd_ci,
            sortino_ci=sortino_ci,
            overall_significant=overall_sig,
            n_bootstrap=self.n_bootstrap,
        )

    def quick_significance(self, oos_returns: pd.Series) -> dict:
        """
        Lightweight significance summary (Welch + bootstrap Sharpe CI only).

        Returns dict suitable for API responses.
        """
        r = oos_returns.dropna()
        t_stat, p_val, welch_sig = self.welch_t_test(r)
        sharpe_ci = self.bootstrap_ci(r, self._sharpe_fn)
        return {
            "welch_t_stat": t_stat,
            "welch_p_value": p_val,
            "welch_significant": welch_sig,
            "sharpe_ci_p5": sharpe_ci[0],
            "sharpe_ci_median": sharpe_ci[1],
            "sharpe_ci_p95": sharpe_ci[2],
            "n_observations": len(r),
        }


# ---------------------------------------------------------------------------
# WalkForwardV2Engine — orchestrates all components
# ---------------------------------------------------------------------------

class WalkForwardV2Engine:
    """
    Top-level engine combining all v2 components.

    Convenience class: run anchored WF + regime validation + significance test
    in a single call.
    """

    def __init__(
        self,
        n_bootstrap: int = 500,
        seed: int = 42,
        ann: int = 252,
    ) -> None:
        self.anchored_engine = AnchoredWalkForwardEngine(ann=ann)
        self.regime_validator = RegimeConditionalValidation(ann=ann)
        self.significance_tester = StatisticalSignificanceTester(n_bootstrap=n_bootstrap, seed=seed, ann=ann)
        self.param_stability = ParameterStabilityTestV2(ann=ann)
        self.ann = ann

    def run_complete_validation(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        anchored_cfg: AnchoredWFConfig | None = None,
        market_returns: pd.Series | None = None,
        param_grid: dict | None = None,
        wf_config: WalkForwardConfig | None = None,
    ) -> dict:
        """
        Run the complete v2 validation suite:
          1. Anchored walk-forward
          2. Regime-conditional performance
          3. Statistical significance
          (4. Parameter stability — if param_grid provided)

        Returns consolidated dict.
        """
        cfg = anchored_cfg or AnchoredWFConfig()
        wf_cfg = wf_config or WalkForwardConfig()

        # 1. Anchored WF
        logger.info("Running anchored walk-forward...")
        awf_result = self.anchored_engine.run(strategy_fn, data, cfg)

        # 2. Regime-conditional
        logger.info("Running regime-conditional validation...")
        regime_result = self.regime_validator.validate_by_regime(
            awf_result.oos_returns,
            market_returns=market_returns,
        )

        # 3. Statistical significance
        logger.info("Running significance tests...")
        bench = market_returns if market_returns is not None else pd.Series(0.0, index=awf_result.oos_returns.index)
        sig_result = self.significance_tester.run_full_significance_test(
            awf_result.oos_returns, bench
        )

        output = {
            "anchored_wf": {
                "window_type": awf_result.window_type_used,
                "optimal_window": awf_result.optimal_window,
                "n_folds": len(awf_result.fold_results),
                "oos_sharpe": self.significance_tester._sharpe_fn(awf_result.oos_returns.fillna(0).values),
                "fold_sharpes": [round(fr.oos_sharpe, 3) for fr in awf_result.fold_results],
                "pct_folds_profitable": sum(1 for fr in awf_result.fold_results if fr.profitable) / max(len(awf_result.fold_results), 1),
                "ic_vs_window": awf_result.ic_vs_window.to_dict(orient="records") if awf_result.ic_vs_window is not None else [],
                "oos_equity_curve": {
                    "dates": [str(d)[:10] for d in awf_result.oos_equity_curve.index],
                    "values": awf_result.oos_equity_curve.round(4).tolist(),
                } if not awf_result.oos_equity_curve.empty else {},
            },
            "regime_conditional": {
                "overall_sharpe": regime_result.overall_sharpe,
                "is_regime_specific": regime_result.is_regime_specific,
                "dominant_regime": regime_result.dominant_regime,
                "regime_dependence_score": regime_result.regime_dependence_score,
                "regime_coverage": regime_result.regime_coverage,
                "regime_reports": {
                    reg: rr.to_dict()
                    for reg, rr in regime_result.regime_reports.items()
                },
                "pairwise_sharpe_diff": regime_result.pairwise_sharpe_diff,
            },
            "significance": sig_result.to_dict(),
        }

        # 4. Parameter stability (optional)
        if param_grid and wf_cfg:
            logger.info("Running parameter stability analysis...")
            try:
                stability = self.param_stability.full_stability_analysis(
                    strategy_fn, data, param_grid, wf_cfg
                )
                output["parameter_stability"] = {
                    "best_params": stability.best_params,
                    "best_sharpe": stability.best_sharpe,
                    "sensitivity_ranking": stability.sensitivity_ranking,
                    "robustness_scores": stability.robustness_scores,
                    "cliff_edge_params": stability.cliff_edge_params,
                    "heatmap_data": stability.heatmap_data,
                }
            except Exception as exc:
                logger.warning("Parameter stability failed: %s", exc)
                output["parameter_stability"] = {"error": str(exc)}

        return output


# ---------------------------------------------------------------------------
# In-memory result store
# ---------------------------------------------------------------------------

_WF_RESULT_STORE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel as PydanticModel, Field as PydanticField

    wf_v2_router = APIRouter(prefix="/walkforward/v2", tags=["Walk-Forward V2"])

    class WFV2Request(PydanticModel):
        returns: list[float] = PydanticField(default_factory=list)
        dates: list[str] = PydanticField(default_factory=list)
        wf_type: str = "anchored"           # anchored | rolling | hybrid
        min_train_periods: int = 252
        max_train_periods: int = 1260
        test_periods: int = 63
        step_size: int = 21
        purge_gap: int = 5
        detect_optimal_window: bool = True

    class RegimeTestRequest(PydanticModel):
        strategy_returns: list[float] = PydanticField(default_factory=list)
        market_returns: list[float] | None = None
        dates: list[str] = PydanticField(default_factory=list)
        volatility_window: int = 60
        trend_window: int = 120

    class ParamStabilityV2Request(PydanticModel):
        returns: list[float] = PydanticField(default_factory=list)
        dates: list[str] = PydanticField(default_factory=list)
        param_grid: dict = PydanticField(default_factory=dict)
        train_periods: int = 252
        test_periods: int = 63
        step_size: int = 21
        purge_gap: int = 5
        run_adaptive: bool = False

    class SignificanceRequest(PydanticModel):
        strategy_returns: list[float] = PydanticField(default_factory=list)
        benchmark_returns: list[float] | None = None
        dates: list[str] = PydanticField(default_factory=list)
        n_bootstrap: int = 1000

    def _build_passthrough_fn(returns_series: pd.Series) -> Callable:
        """Build strategy function that replays precomputed returns."""
        def _fn(train_data: pd.DataFrame, test_data: pd.DataFrame, params: dict | None) -> pd.Series:
            col = "returns" if "returns" in test_data.columns else test_data.columns[0]
            return test_data[col].dropna()
        return _fn

    @wf_v2_router.post("/run")
    async def api_wf_v2_run(req: WFV2Request):
        """Run anchored/rolling/hybrid walk-forward with optional optimal window detection."""
        if not req.returns:
            raise HTTPException(status_code=422, detail="returns required")

        idx = pd.to_datetime(req.dates) if req.dates and len(req.dates) == len(req.returns) \
            else pd.date_range("2015-01-01", periods=len(req.returns), freq="B")

        data = pd.DataFrame({"returns": req.returns}, index=idx)
        strategy_fn = _build_passthrough_fn(data["returns"])

        cfg = AnchoredWFConfig(
            wf_type=req.wf_type,
            min_train_periods=req.min_train_periods,
            max_train_periods=req.max_train_periods,
            test_periods=req.test_periods,
            step_size=req.step_size,
            purge_gap=req.purge_gap,
            detect_optimal_window=req.detect_optimal_window,
        )

        engine = AnchoredWalkForwardEngine()
        try:
            result = engine.run(strategy_fn, data, cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        run_id = str(uuid.uuid4())[:12]
        oos_rets = result.oos_returns.dropna()
        oos_metrics = {
            "sharpe": round(float(oos_rets.mean() / oos_rets.std() * math.sqrt(252)) if oos_rets.std() > 0 else 0.0, 3),
            "n_obs": len(oos_rets),
            "total_return": round(float((1 + oos_rets.fillna(0)).prod() - 1), 4),
        }
        out = {
            "run_id": run_id,
            "window_type": result.window_type_used,
            "optimal_window": result.optimal_window,
            "n_folds": len(result.fold_results),
            "oos_metrics": oos_metrics,
            "fold_sharpes": [round(fr.oos_sharpe, 3) for fr in result.fold_results],
            "pct_folds_profitable": round(sum(1 for fr in result.fold_results if fr.profitable) / max(len(result.fold_results), 1), 3),
            "ic_vs_window": result.ic_vs_window.to_dict(orient="records") if result.ic_vs_window is not None else [],
            "oos_returns": oos_rets.round(6).tolist(),
            "oos_dates": [str(d)[:10] for d in oos_rets.index],
        }
        _WF_RESULT_STORE[run_id] = out
        return out

    @wf_v2_router.post("/regime-test")
    async def api_regime_test(req: RegimeTestRequest):
        """Validate strategy performance split by market regime."""
        if not req.strategy_returns:
            raise HTTPException(status_code=422, detail="strategy_returns required")

        idx = pd.to_datetime(req.dates) if req.dates and len(req.dates) == len(req.strategy_returns) \
            else pd.date_range("2015-01-01", periods=len(req.strategy_returns), freq="B")

        strat_rets = pd.Series(req.strategy_returns, index=idx)
        mkt_rets = None
        if req.market_returns and len(req.market_returns) == len(req.strategy_returns):
            mkt_rets = pd.Series(req.market_returns, index=idx)

        validator = RegimeConditionalValidation()
        try:
            result = validator.validate_by_regime(
                strat_rets,
                market_returns=mkt_rets,
                volatility_window=req.volatility_window,
                trend_window=req.trend_window,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        run_id = str(uuid.uuid4())[:12]
        out = {
            "run_id": run_id,
            "overall_sharpe": result.overall_sharpe,
            "is_regime_specific": result.is_regime_specific,
            "dominant_regime": result.dominant_regime,
            "regime_dependence_score": result.regime_dependence_score,
            "regime_coverage": result.regime_coverage,
            "pairwise_sharpe_diff": result.pairwise_sharpe_diff,
            "regime_reports": {reg: rr.to_dict() for reg, rr in result.regime_reports.items()},
        }
        _WF_RESULT_STORE[run_id] = out
        return out

    @wf_v2_router.post("/param-stability")
    async def api_param_stability_v2(req: ParamStabilityV2Request):
        """Enhanced parameter stability analysis with heatmaps and sensitivity ranking."""
        if not req.returns or not req.param_grid:
            raise HTTPException(status_code=422, detail="returns and param_grid required")

        idx = pd.to_datetime(req.dates) if req.dates and len(req.dates) == len(req.returns) \
            else pd.date_range("2015-01-01", periods=len(req.returns), freq="B")

        data = pd.DataFrame({"returns": req.returns}, index=idx)
        strategy_fn = _build_passthrough_fn(data["returns"])

        wf_cfg = WalkForwardConfig(
            train_periods=req.train_periods,
            test_periods=req.test_periods,
            step_size=req.step_size,
            purge_gap=req.purge_gap,
        )

        tester = ParameterStabilityTestV2()
        try:
            stability = tester.full_stability_analysis(strategy_fn, data, req.param_grid, wf_cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        out: dict = {
            "best_params": stability.best_params,
            "best_sharpe": stability.best_sharpe,
            "sensitivity_ranking": stability.sensitivity_ranking,
            "robustness_scores": stability.robustness_scores,
            "cliff_edge_params": stability.cliff_edge_params,
            "heatmap_data": stability.heatmap_data,
            "grid_results": stability.grid_results.fillna(0).to_dict(orient="records"),
        }

        # Adaptive WF (optional, expensive)
        if req.run_adaptive and len(req.param_grid) >= 1:
            try:
                ada = tester.adaptive_walk_forward(strategy_fn, data, req.param_grid, wf_cfg)
                out["adaptive_walk_forward"] = {
                    "oos_sharpe": ada["oos_sharpe"],
                    "n_folds": ada["n_folds"],
                    "param_stability_pct": ada["param_stability_pct"],
                    "param_history": ada["param_history"],
                }
            except Exception as exc:
                out["adaptive_walk_forward"] = {"error": str(exc)}

        run_id = str(uuid.uuid4())[:12]
        _WF_RESULT_STORE[run_id] = out
        return {"run_id": run_id, **out}

    @wf_v2_router.post("/significance")
    async def api_significance(req: SignificanceRequest):
        """Run formal statistical significance tests (White's RC, Romano-Wolf, Welch t-test, bootstrap CIs)."""
        if not req.strategy_returns:
            raise HTTPException(status_code=422, detail="strategy_returns required")

        idx = pd.to_datetime(req.dates) if req.dates and len(req.dates) == len(req.strategy_returns) \
            else pd.date_range("2015-01-01", periods=len(req.strategy_returns), freq="B")

        strat_rets = pd.Series(req.strategy_returns, index=idx)
        bench_rets = None
        if req.benchmark_returns and len(req.benchmark_returns) == len(req.strategy_returns):
            bench_rets = pd.Series(req.benchmark_returns, index=idx)

        tester = StatisticalSignificanceTester(n_bootstrap=req.n_bootstrap)
        try:
            result = tester.run_full_significance_test(strat_rets, bench_rets)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        run_id = str(uuid.uuid4())[:12]
        out = {"run_id": run_id, **result.to_dict()}
        _WF_RESULT_STORE[run_id] = out
        return out

except ImportError:
    wf_v2_router = None  # type: ignore[assignment]
    logger.debug("FastAPI not available — wf_v2_router not registered")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def build_passthrough_strategy(
    returns: pd.Series,
) -> Callable:
    """
    Build a simple passthrough strategy function for API/test usage.

    The returned function replays the provided returns for any test period.
    """
    def _fn(train_data: pd.DataFrame, test_data: pd.DataFrame, params: dict | None) -> pd.Series:
        # Find any column with numeric returns
        for col in test_data.columns:
            try:
                r = test_data[col].dropna()
                if len(r) > 0:
                    return r
            except Exception:
                continue
        return pd.Series(dtype=float)
    return _fn


def quick_anchored_validation(
    strategy_fn: Callable,
    returns_data: pd.DataFrame,
    min_train: int = 252,
    test_periods: int = 63,
    wf_type: str = "anchored",
) -> dict:
    """
    One-liner anchored walk-forward validation.

    Returns dict with key OOS metrics and fold breakdown.
    """
    cfg = AnchoredWFConfig(
        wf_type=wf_type,
        min_train_periods=min_train,
        test_periods=test_periods,
        detect_optimal_window=True,
    )
    engine = AnchoredWalkForwardEngine()
    result = engine.run(strategy_fn, returns_data, cfg)

    oos = result.oos_returns.dropna()
    sharpe = float(oos.mean() / oos.std() * math.sqrt(252)) if oos.std() > 0 else 0.0
    total_ret = float((1 + oos.fillna(0)).prod() - 1)
    pct_profitable = sum(1 for fr in result.fold_results if fr.profitable) / max(len(result.fold_results), 1)

    return {
        "wf_type": result.window_type_used,
        "optimal_window": result.optimal_window,
        "n_folds": len(result.fold_results),
        "oos_sharpe": round(sharpe, 3),
        "oos_total_return": round(total_ret, 4),
        "pct_folds_profitable": round(pct_profitable, 3),
        "fold_sharpes": [round(fr.oos_sharpe, 3) for fr in result.fold_results],
    }


def regime_significance_summary(
    strategy_returns: pd.Series,
    market_returns: pd.Series | None = None,
    n_bootstrap: int = 500,
) -> dict:
    """
    Run regime validation + significance tests and return a combined summary dict.
    """
    rcv = RegimeConditionalValidation()
    regime_result = rcv.validate_by_regime(strategy_returns, market_returns=market_returns)

    tester = StatisticalSignificanceTester(n_bootstrap=n_bootstrap)
    sig_result = tester.run_full_significance_test(
        strategy_returns,
        market_returns,
    )

    return {
        "regime_validation": {
            "is_regime_specific": regime_result.is_regime_specific,
            "dominant_regime": regime_result.dominant_regime,
            "dependence_score": regime_result.regime_dependence_score,
            "regime_sharpes": {
                reg: round(rr.sharpe, 3)
                for reg, rr in regime_result.regime_reports.items()
                if rr.n_observations >= 5
            },
        },
        "significance": {
            "overall_significant": sig_result.overall_significant,
            "welch_p_value": sig_result.welch_p_value,
            "white_rc_p_value": sig_result.white_rc_p_value,
            "sharpe_ci": {
                "p5": sig_result.sharpe_ci[0],
                "median": sig_result.sharpe_ci[1],
                "p95": sig_result.sharpe_ci[2],
            },
        },
    }
