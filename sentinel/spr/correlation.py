"""Rolling correlation matrix and regime-change correlation alerts."""
from __future__ import annotations

from datetime import date
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class CorrelationAlert(BaseModel):
    ticker_pair: tuple[str, str]
    current_corr: float
    prior_corr: float
    change: float
    alert_type: str   # "correlation_spike" | "decorrelation" | "regime_break"
    severity: str     # "low" | "medium" | "high"


class CorrelationReport(BaseModel):
    as_of: date
    window_days: int
    matrix: dict[str, dict[str, float]]
    alerts: list[CorrelationAlert]
    avg_pairwise_corr: float
    regime_stress_indicator: float  # 0-1, high = correlations spiking (crisis mode)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_rolling_correlation(
    returns_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling pairwise Pearson correlation over `window` trading days.

    Returns a DataFrame whose columns are a MultiIndex of (ticker_a, ticker_b)
    and whose index matches returns_df.index. Only the upper triangle of unique
    pairs is computed to avoid duplication; the diagonal (self-correlation) is
    excluded.
    """
    tickers = list(returns_df.columns)
    if len(tickers) < 2:
        logger.warning("compute_rolling_correlation_needs_at_least_2_tickers")
        return pd.DataFrame(index=returns_df.index)

    logger.debug(
        "compute_rolling_correlation",
        tickers=tickers,
        window=window,
        rows=len(returns_df),
    )

    pairs = list(combinations(tickers, 2))
    result_cols: dict[tuple[str, str], pd.Series] = {}

    for t1, t2 in pairs:
        rolling_corr = (
            returns_df[t1]
            .rolling(window=window, min_periods=max(window // 2, 2))
            .corr(returns_df[t2])
        )
        result_cols[(t1, t2)] = rolling_corr

    result = pd.DataFrame(result_cols, index=returns_df.index)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


def _snapshot_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a static Pearson correlation matrix from a returns DataFrame."""
    return returns_df.corr(method="pearson")


def detect_regime_break(
    current_matrix: pd.DataFrame,
    prior_matrix: pd.DataFrame,
    threshold: float = 0.25,
) -> list[CorrelationAlert]:
    """Detect pairs where correlation changed by more than `threshold`.

    Alert types:
      - correlation_spike : change > 0 (correlations increased)
      - decorrelation     : change < 0 (correlations decreased)
      - regime_break      : |change| > 0.4 (large structural shift)

    Severity levels:
      - low    : threshold <= |change| < 0.35
      - medium : 0.35 <= |change| < 0.4
      - high   : |change| >= 0.4
    """
    tickers = list(current_matrix.columns)
    alerts: list[CorrelationAlert] = []

    for t1, t2 in combinations(tickers, 2):
        if t1 not in current_matrix.columns or t2 not in current_matrix.columns:
            continue
        if t1 not in prior_matrix.columns or t2 not in prior_matrix.columns:
            continue

        curr = float(current_matrix.loc[t1, t2])
        prior = float(prior_matrix.loc[t1, t2])

        if np.isnan(curr) or np.isnan(prior):
            continue

        change = curr - prior
        abs_change = abs(change)

        if abs_change < threshold:
            continue

        if abs_change >= 0.4:
            alert_type = "regime_break"
            severity = "high"
        elif abs_change >= 0.35:
            alert_type = "correlation_spike" if change > 0 else "decorrelation"
            severity = "medium"
        else:
            alert_type = "correlation_spike" if change > 0 else "decorrelation"
            severity = "low"

        alerts.append(
            CorrelationAlert(
                ticker_pair=(t1, t2),
                current_corr=round(curr, 4),
                prior_corr=round(prior, 4),
                change=round(change, 4),
                alert_type=alert_type,
                severity=severity,
            )
        )

    logger.info(
        "regime_break_detection_complete",
        num_alerts=len(alerts),
        threshold=threshold,
    )
    return alerts


def compute_stress_indicator(matrix: pd.DataFrame) -> float:
    """Average of all pairwise (off-diagonal) absolute correlations.

    Returns a value in [0, 1]. Values above ~0.7 indicate crisis-mode
    correlation convergence where diversification benefits collapse.
    NaN cells are excluded from the average.
    """
    tickers = list(matrix.columns)
    if len(tickers) < 2:
        return 0.0

    values: list[float] = []
    for t1, t2 in combinations(tickers, 2):
        val = matrix.loc[t1, t2]
        if not np.isnan(val):
            values.append(float(val))

    if not values:
        return 0.0

    # Use absolute correlations so negative correlations (hedges) also
    # contribute to the stress indicator when they approach -1 (flight-to-safety).
    avg_abs_corr = float(np.mean(np.abs(values)))
    return round(min(max(avg_abs_corr, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def generate_correlation_report(
    returns_df: pd.DataFrame,
    window: int = 60,
) -> CorrelationReport:
    """Full report: compute current and prior (lagged 30 days) matrices,
    detect regime breaks, score stress.

    Current matrix  = correlation over the final `window` rows of returns_df.
    Prior matrix    = correlation over the `window` rows ending 30 trading days
                      before the end of returns_df.
    """
    if returns_df.empty:
        raise ValueError("returns_df is empty — cannot generate correlation report")

    tickers = list(returns_df.columns)
    as_of_ts = returns_df.index[-1]
    as_of = as_of_ts.date() if hasattr(as_of_ts, "date") else date.today()

    logger.info(
        "generate_correlation_report",
        tickers=tickers,
        window=window,
        as_of=str(as_of),
        total_rows=len(returns_df),
    )

    lag_days = 30  # trading-day lag for the prior window

    # Current window: last `window` rows
    current_slice = returns_df.iloc[-window:] if len(returns_df) >= window else returns_df
    current_matrix = _snapshot_matrix(current_slice)

    # Prior window: `window` rows ending `lag_days` before the last row
    prior_end_idx = max(len(returns_df) - lag_days, window)
    prior_start_idx = max(prior_end_idx - window, 0)
    prior_slice = returns_df.iloc[prior_start_idx:prior_end_idx]

    if len(prior_slice) < 2:
        logger.warning("insufficient_data_for_prior_matrix", available=len(prior_slice))
        prior_matrix = current_matrix.copy()
    else:
        prior_matrix = _snapshot_matrix(prior_slice)

    # Alerts
    alerts = detect_regime_break(current_matrix, prior_matrix, threshold=0.25)

    # Stress indicator
    stress_indicator = compute_stress_indicator(current_matrix)

    # Average pairwise correlation (raw, not absolute, for directional sense)
    raw_values: list[float] = []
    for t1, t2 in combinations(tickers, 2):
        if t1 in current_matrix.columns and t2 in current_matrix.columns:
            val = current_matrix.loc[t1, t2]
            if not np.isnan(val):
                raw_values.append(float(val))
    avg_pairwise_corr = float(np.mean(raw_values)) if raw_values else 0.0

    # Serialise matrix to dict[str, dict[str, float]]
    matrix_dict: dict[str, dict[str, float]] = {}
    for t1 in tickers:
        row: dict[str, float] = {}
        for t2 in tickers:
            try:
                v = current_matrix.loc[t1, t2]
                row[t2] = round(float(v), 4) if not np.isnan(v) else 0.0
            except KeyError:
                row[t2] = 0.0
        matrix_dict[t1] = row

    logger.info(
        "correlation_report_complete",
        num_alerts=len(alerts),
        stress_indicator=stress_indicator,
        avg_pairwise_corr=round(avg_pairwise_corr, 4),
    )

    return CorrelationReport(
        as_of=as_of,
        window_days=window,
        matrix=matrix_dict,
        alerts=alerts,
        avg_pairwise_corr=round(avg_pairwise_corr, 4),
        regime_stress_indicator=stress_indicator,
    )
