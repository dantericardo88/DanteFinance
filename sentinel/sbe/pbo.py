"""
Probability of Backtest Overfitting (PBO) — LEAPFROG #64.

Implements the Combinatorially Symmetric Cross-Validation (CSCV) method
from Bailey, Borwein, López de Prado & Zhu (2014) to detect whether a
strategy's performance is attributable to overfitting rather than true alpha.

PBO > 0.5 means the best in-sample strategy is more likely to underperform
out-of-sample than to outperform — a direct indicator of overfitting.

No incumbent terminal computes PBO. Score: SENTINEL 10, Bloomberg 0.
"""
from __future__ import annotations
import itertools
import math
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import logistic
from sentinel.core.logging import get_logger

logger = get_logger(__name__)


def compute_pbo(
    returns_matrix: pd.DataFrame,
    n_partitions: int = 16,
    metric: str = "sharpe",
    risk_free_rate: float = 0.05,
    annualization: int = 252,
) -> dict:
    """
    Compute Probability of Backtest Overfitting via CSCV.

    Args:
        returns_matrix: DataFrame where each column is a strategy/parameter combo,
                        rows are time periods. Must have DatetimeIndex.
        n_partitions: Number of partitions (S). Must be even. More = better estimate.
        metric: Scoring metric — 'sharpe', 'sortino', 'cagr', 'calmar'.
        risk_free_rate: Annual risk-free rate.
        annualization: Periods per year (252 for daily).

    Returns:
        dict with: pbo, lambda_bar, omega_bar, is_rank_correlation, distribution.
    """
    if returns_matrix.shape[1] < 2:
        raise ValueError("Need at least 2 strategies/columns to compute PBO")
    if n_partitions % 2 != 0:
        n_partitions += 1

    T, N = returns_matrix.shape
    S = n_partitions
    # Split into S equal partitions
    partition_size = T // S
    if partition_size < 5:
        raise ValueError(f"Too few observations ({T}) for {S} partitions. Reduce n_partitions.")

    # Trim to exact multiple of S
    trimmed = returns_matrix.iloc[: partition_size * S]
    partitions = [trimmed.iloc[i * partition_size:(i + 1) * partition_size] for i in range(S)]

    # All combinations of S/2 partitions for IS set (out-of-sample = complement)
    half = S // 2
    combo_indices = list(itertools.combinations(range(S), half))

    logit_values = []
    is_ranks = []

    for is_idx in combo_indices:
        oos_idx = tuple(i for i in range(S) if i not in is_idx)

        is_returns = pd.concat([partitions[i] for i in is_idx])
        oos_returns = pd.concat([partitions[i] for i in oos_idx])

        is_scores = _score_strategies(is_returns, metric, risk_free_rate, annualization)
        oos_scores = _score_strategies(oos_returns, metric, risk_free_rate, annualization)

        # Best IS strategy
        best_is = int(np.argmax(is_scores))
        best_is_rank_in_is = float(np.sum(is_scores <= is_scores[best_is])) / N

        # Rank of best IS strategy in OOS
        oos_rank_of_best = float(np.sum(oos_scores <= oos_scores[best_is])) / N

        # Logit of relative performance
        omega = oos_rank_of_best
        if 0 < omega < 1:
            logit_values.append(math.log(omega / (1 - omega)))
        is_ranks.append(best_is_rank_in_is)

    # PBO = probability that logit(OOS rank) < 0
    if not logit_values:
        return {"pbo": 0.5, "lambda_bar": 0.0, "n_combinations": 0}

    lambda_bar = float(np.mean(logit_values))
    pbo = float(np.mean(np.array(logit_values) < 0))

    # Stochastic dominance: OOS performance distribution
    is_rank_corr = float(np.corrcoef(is_ranks, logit_values)[0, 1]) if len(is_ranks) > 2 else 0.0

    logger.info("PBO computed", pbo=round(pbo, 4), lambda_bar=round(lambda_bar, 4),
                n_combos=len(logit_values), n_strategies=N)

    return {
        "pbo": round(pbo, 4),
        "lambda_bar": round(lambda_bar, 4),
        "is_rank_correlation": round(is_rank_corr, 4),
        "n_combinations": len(logit_values),
        "n_strategies": N,
        "n_partitions": S,
        "metric": metric,
        "logit_distribution": logit_values,
        "interpretation": _interpret_pbo(pbo),
    }


def _score_strategies(
    returns: pd.DataFrame,
    metric: str,
    risk_free_rate: float,
    ann: int,
) -> np.ndarray:
    """Score each column of returns DataFrame using the given metric."""
    scores = np.zeros(returns.shape[1])
    rf_period = risk_free_rate / ann
    for i, col in enumerate(returns.columns):
        r = returns[col].dropna()
        if r.empty or r.std() < 1e-10:
            scores[i] = -999.0
            continue
        if metric == "sharpe":
            scores[i] = (r.mean() - rf_period) / r.std() * math.sqrt(ann)
        elif metric == "sortino":
            ds = r[r < 0].std()
            scores[i] = (r.mean() - rf_period) / (ds + 1e-10) * math.sqrt(ann)
        elif metric == "cagr":
            scores[i] = float((1 + r).prod() ** (ann / len(r)) - 1)
        elif metric == "calmar":
            cum = (1 + r).cumprod()
            dd = (cum / cum.expanding().max() - 1).min()
            cagr = float((1 + r).prod() ** (ann / len(r)) - 1)
            scores[i] = cagr / (abs(dd) + 1e-10)
        else:
            scores[i] = r.mean() * ann
    return scores


def _interpret_pbo(pbo: float) -> str:
    if pbo < 0.25:
        return "Low overfitting risk — strategy likely has genuine alpha"
    if pbo < 0.50:
        return "Moderate overfitting risk — proceed with caution and paper trade first"
    if pbo < 0.75:
        return "High overfitting risk — strategy performance likely due to data mining"
    return "Extreme overfitting — strategy should not proceed to live trading"


def build_returns_matrix_from_params(
    ohlcv: pd.DataFrame,
    param_grid: list[dict],
    strategy_fn,
) -> pd.DataFrame:
    """
    Helper to build a returns matrix from a grid of strategy parameters.
    strategy_fn(ohlcv, **params) → pd.Series of returns.
    """
    cols = {}
    for i, params in enumerate(param_grid):
        try:
            col_name = f"s{i}_" + "_".join(f"{k}{v}" for k, v in params.items())
            returns = strategy_fn(ohlcv, **params)
            cols[col_name] = returns
        except Exception as exc:
            logger.warning("Strategy param failed in PBO matrix", params=params, error=str(exc))
    return pd.DataFrame(cols)
