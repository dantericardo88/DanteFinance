"""Fama-French 5-factor + momentum risk decomposition for portfolio holdings."""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_FF_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class FactorExposure(BaseModel):
    factor: str
    loading: float
    t_stat: float
    p_value: float
    contribution_pct: float  # % of portfolio variance explained by this factor


class FactorModelResult(BaseModel):
    ticker: str
    alpha_annualized: float
    r_squared: float
    exposures: list[FactorExposure]
    residual_vol: float
    tracking_error: float


class PortfolioFactorResult(BaseModel):
    portfolio_alpha: float
    r_squared: float
    factor_contributions: dict[str, float]
    weighted_exposures: dict[str, float]
    holdings: list[FactorModelResult]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def fetch_ff5_factors(start: date, end: date) -> pd.DataFrame:
    """Download FF5 + momentum from Ken French data library via pandas_datareader.

    Returns a DataFrame with columns Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    and a DatetimeIndex. Values are in decimal form (divided by 100).
    Falls back to a synthetic zero frame if the library is unavailable so
    callers can still test without a network connection.
    """
    try:
        import pandas_datareader.data as web  # type: ignore

        logger.info("fetching_ff5_factors", start=str(start), end=str(end))

        ff5 = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily",
            "famafrench",
            start=start,
            end=end,
        )[0]
        ff5 = ff5 / 100.0

        mom = web.DataReader(
            "F-F_Momentum_Factor_daily",
            "famafrench",
            start=start,
            end=end,
        )[0]
        mom = mom / 100.0
        mom.columns = ["Mom"]

        combined = ff5.join(mom, how="inner")
        combined.index = pd.to_datetime(combined.index)
        combined = combined.loc[
            (combined.index >= pd.Timestamp(start))
            & (combined.index <= pd.Timestamp(end))
        ]
        logger.info(
            "ff5_factors_fetched",
            rows=len(combined),
            columns=list(combined.columns),
        )
        return combined

    except Exception as exc:
        logger.warning("ff5_fetch_failed_returning_zeros", error=str(exc))
        idx = pd.date_range(start=start, end=end, freq="B")
        cols = _FF_FACTORS + ["RF"]
        return pd.DataFrame(0.0, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# Single-asset regression
# ---------------------------------------------------------------------------


def run_factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
) -> FactorModelResult:
    """OLS regression of excess returns on FF5+Mom factors.

    Computes alpha (annualised), betas, t-stats, p-values, R², residual vol,
    and per-factor variance-contribution percentages.
    """
    ticker = str(returns.name) if returns.name else "unknown"
    logger.debug("running_factor_regression", ticker=ticker)

    # Align on common dates
    rf = factors["RF"] if "RF" in factors.columns else pd.Series(0.0, index=factors.index)
    factor_cols = [c for c in _FF_FACTORS if c in factors.columns]
    aligned = returns.align(factors[factor_cols + ["RF"]], join="inner")
    ret_aligned: pd.Series = aligned[0]
    fac_aligned: pd.DataFrame = aligned[1]

    excess_returns = ret_aligned - fac_aligned["RF"]
    X = fac_aligned[factor_cols].values
    y = excess_returns.values

    if len(y) < len(factor_cols) + 2:
        logger.warning("insufficient_data_for_regression", ticker=ticker, n=len(y))
        return _empty_result(ticker, factor_cols)

    # Add intercept column
    X_const = np.column_stack([np.ones(len(y)), X])
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_const, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        logger.error("lstsq_failed", ticker=ticker, error=str(exc))
        return _empty_result(ticker, factor_cols)

    alpha_daily = coeffs[0]
    betas = coeffs[1:]
    y_hat = X_const @ coeffs
    resid = y - y_hat
    n, k = len(y), len(factor_cols)
    dof = n - k - 1

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mse = ss_res / dof if dof > 0 else 0.0
    try:
        cov_matrix = mse * np.linalg.inv(X_const.T @ X_const)
        se = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        se = np.ones(len(coeffs)) * np.nan

    # t-stats and p-values for betas (index 1 onward)
    beta_se = se[1:]
    t_stats = betas / beta_se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=dof)

    # Factor variance contributions
    factor_var_contribs = _factor_variance_contributions(betas, X, r_squared, ss_tot, ss_res)

    exposures = [
        FactorExposure(
            factor=factor_cols[i],
            loading=float(betas[i]),
            t_stat=float(t_stats[i]) if not np.isnan(t_stats[i]) else 0.0,
            p_value=float(p_values[i]) if not np.isnan(p_values[i]) else 1.0,
            contribution_pct=float(factor_var_contribs[i]),
        )
        for i in range(len(factor_cols))
    ]

    residual_vol = float(np.std(resid) * np.sqrt(252))
    alpha_annualized = float(alpha_daily * 252)
    tracking_error = float(np.std(y - X @ betas) * np.sqrt(252))

    return FactorModelResult(
        ticker=ticker,
        alpha_annualized=alpha_annualized,
        r_squared=float(r_squared),
        exposures=exposures,
        residual_vol=residual_vol,
        tracking_error=tracking_error,
    )


def _factor_variance_contributions(
    betas: np.ndarray,
    X: np.ndarray,
    r_squared: float,
    ss_tot: float,
    ss_res: float,
) -> np.ndarray:
    """Apportion explained variance (R²) across factors via sequential R² drops."""
    n_factors = len(betas)
    contribs = np.zeros(n_factors)
    if ss_tot <= 0:
        return contribs

    explained_var = ss_tot - ss_res
    if explained_var <= 0:
        return contribs

    factor_vars = np.var(X, axis=0) * betas**2
    total_factor_var = factor_vars.sum()
    if total_factor_var > 0:
        contribs = (factor_vars / total_factor_var) * r_squared * 100.0
    return contribs


def _empty_result(ticker: str, factor_cols: list[str]) -> FactorModelResult:
    exposures = [
        FactorExposure(factor=f, loading=0.0, t_stat=0.0, p_value=1.0, contribution_pct=0.0)
        for f in factor_cols
    ]
    return FactorModelResult(
        ticker=ticker,
        alpha_annualized=0.0,
        r_squared=0.0,
        exposures=exposures,
        residual_vol=0.0,
        tracking_error=0.0,
    )


# ---------------------------------------------------------------------------
# Portfolio-level decomposition
# ---------------------------------------------------------------------------


def decompose_portfolio(
    holdings: dict[str, float],
    returns_df: pd.DataFrame,
    start: date,
    end: date,
) -> PortfolioFactorResult:
    """Run factor regression for each holding; weight-average exposures for portfolio.

    Also runs a regression on portfolio-level returns for the portfolio alpha.
    """
    logger.info(
        "decompose_portfolio",
        tickers=list(holdings.keys()),
        start=str(start),
        end=str(end),
    )
    factors = fetch_ff5_factors(start, end)

    # Filter returns to date range
    mask = (returns_df.index >= pd.Timestamp(start)) & (returns_df.index <= pd.Timestamp(end))
    returns_slice = returns_df.loc[mask]

    holding_results: list[FactorModelResult] = []
    for ticker, weight in holdings.items():
        if ticker not in returns_slice.columns:
            logger.warning("ticker_not_in_returns", ticker=ticker)
            continue
        series = returns_slice[ticker].rename(ticker)
        result = run_factor_regression(series, factors)
        holding_results.append(result)

    # Weighted exposures
    factor_cols = _FF_FACTORS
    weighted_exposures: dict[str, float] = {f: 0.0 for f in factor_cols}
    factor_contributions: dict[str, float] = {f: 0.0 for f in factor_cols}

    for result in holding_results:
        weight = holdings.get(result.ticker, 0.0)
        for exp in result.exposures:
            weighted_exposures[exp.factor] = (
                weighted_exposures.get(exp.factor, 0.0) + weight * exp.loading
            )
            factor_contributions[exp.factor] = (
                factor_contributions.get(exp.factor, 0.0) + weight * exp.contribution_pct
            )

    # Portfolio-level regression on weighted returns
    weights_series = pd.Series(holdings)
    common_tickers = [t for t in holdings if t in returns_slice.columns]
    if common_tickers:
        w = weights_series[common_tickers]
        w = w / w.sum()
        port_returns = returns_slice[common_tickers].dot(w)
        port_returns.name = "_portfolio"
        port_result = run_factor_regression(port_returns, factors)
        portfolio_alpha = port_result.alpha_annualized
        r_squared = port_result.r_squared
    else:
        portfolio_alpha = 0.0
        r_squared = 0.0

    return PortfolioFactorResult(
        portfolio_alpha=portfolio_alpha,
        r_squared=r_squared,
        factor_contributions=factor_contributions,
        weighted_exposures=weighted_exposures,
        holdings=holding_results,
    )
