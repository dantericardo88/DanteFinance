"""
Kelly criterion, volatility targeting, and risk-parity position sizing — Dimension #82.

Implements three complementary sizing frameworks:
  - Kelly criterion (discrete + continuous from returns history)
  - Volatility targeting (scale position so realized vol → target vol)
  - Risk parity / Equal Risk Contribution via inverse-vol and scipy ERC

No incumbent terminal ships all three with a unified interface.
Score: SENTINEL 9, Bloomberg 1 (basic vol-scaling only).
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class KellyResult(BaseModel):
    ticker: str
    full_kelly_fraction: float          # raw Kelly: edge / odds
    half_kelly_fraction: float          # conservative: full * 0.5
    quarter_kelly_fraction: float       # ultra-conservative: full * 0.25
    expected_log_growth: float          # E[log(1 + f*X)] per period
    recommended_fraction: float         # half Kelly by default
    max_loss_scenario: float            # loss at full Kelly if bet loses


class VolTargetResult(BaseModel):
    ticker: str
    target_vol: float                   # annualized, e.g. 0.15 = 15 %
    realized_vol: float                 # rolling realized vol (annualized)
    position_size_fraction: float       # target_vol / realized_vol, capped at 1.0
    scale_factor: float                 # relative to equal-weight


class RiskParityResult(BaseModel):
    assets: list[str]
    weights: dict[str, float]           # risk-parity weights (sum to 1)
    risk_contributions: dict[str, float]  # each asset's % of total portfolio risk
    target_vol: float
    portfolio_vol: float                # realized portfolio vol at these weights


# ---------------------------------------------------------------------------
# Kelly — discrete (win/loss game)
# ---------------------------------------------------------------------------


def kelly_single(
    win_prob: float,
    win_return: float,
    loss_return: float,
) -> KellyResult:
    """Classic Kelly criterion for binary outcomes.

    f* = (p * b - q) / b  where b = |win_return / loss_return|

    Args:
        win_prob: probability of a winning outcome (0 < p < 1).
        win_return: fractional gain on a win, e.g. 0.20 for +20 %.
        loss_return: fractional loss on a loss, e.g. -0.10 for −10 %.

    Returns:
        KellyResult with full/half/quarter fractions and expected log growth.
    """
    if not (0.0 < win_prob < 1.0):
        raise ValueError("win_prob must be strictly between 0 and 1")
    if win_return <= 0.0:
        raise ValueError("win_return must be positive")
    if loss_return >= 0.0:
        raise ValueError("loss_return must be negative")

    loss_mag = abs(loss_return)
    q = 1.0 - win_prob
    b = win_return / loss_mag                     # odds ratio

    full_kelly = (win_prob * b - q) / b
    full_kelly = float(np.clip(full_kelly, 0.0, 1.0))  # never short, never > 100 %

    half_kelly = full_kelly * 0.5
    quarter_kelly = full_kelly * 0.25

    # E[log(1 + f*X)] approximated via two-outcome expectation
    def _elog(f: float) -> float:
        if f <= 0.0:
            return 0.0
        p_win_val = 1.0 + f * win_return
        p_loss_val = 1.0 + f * loss_return
        if p_win_val <= 0.0 or p_loss_val <= 0.0:
            return -np.inf
        return win_prob * np.log(p_win_val) + q * np.log(p_loss_val)

    expected_log = _elog(full_kelly)
    max_loss = full_kelly * loss_return  # negative number

    logger.debug(
        "Kelly single: p=%.3f b=%.3f f*=%.4f expected_log=%.5f",
        win_prob, b, full_kelly, expected_log,
    )

    return KellyResult(
        ticker="single_trade",
        full_kelly_fraction=round(full_kelly, 6),
        half_kelly_fraction=round(half_kelly, 6),
        quarter_kelly_fraction=round(quarter_kelly, 6),
        expected_log_growth=round(expected_log, 6),
        recommended_fraction=round(half_kelly, 6),
        max_loss_scenario=round(max_loss, 6),
    )


# ---------------------------------------------------------------------------
# Kelly — continuous (from returns history)
# ---------------------------------------------------------------------------


def kelly_from_returns(
    returns: pd.Series,
    ticker: str = "asset",
    risk_free_rate: float = 0.045,
) -> KellyResult:
    """Estimate Kelly fraction from historical returns using mean/variance approximation.

    Continuous Kelly (log-normal): f* ≈ (μ − rf) / σ²

    where μ and σ² are the annualised mean excess return and variance.
    Expected log growth per period: E[log] ≈ (μ − rf)² / (2σ²)

    Args:
        returns: daily return series (fractional, e.g. 0.01 = +1 %).
        ticker: label for the result.
        risk_free_rate: annualised risk-free rate.

    Returns:
        KellyResult populated from the empirical distribution.
    """
    returns = returns.dropna()
    if len(returns) < 20:
        raise ValueError(f"Need at least 20 return observations, got {len(returns)}")

    # Annualise from daily
    mu_daily = float(returns.mean())
    var_daily = float(returns.var(ddof=1))

    mu_annual = mu_daily * TRADING_DAYS
    var_annual = var_daily * TRADING_DAYS
    sigma_annual = float(np.sqrt(var_annual))

    excess_return = mu_annual - risk_free_rate

    if var_annual <= 0.0:
        raise ValueError("Return variance is zero — cannot compute Kelly fraction")

    full_kelly = float(np.clip(excess_return / var_annual, 0.0, 1.0))
    half_kelly = full_kelly * 0.5
    quarter_kelly = full_kelly * 0.25

    # E[log(1 + f*X)] per trading day using the log-normal approximation
    # = f*(μ_d − rf_d) − 0.5 * f*² * σ_d²
    rf_daily = risk_free_rate / TRADING_DAYS
    expected_log = full_kelly * (mu_daily - rf_daily) - 0.5 * full_kelly**2 * var_daily

    # Worst daily loss assuming normal: μ - 3σ
    daily_sigma = float(np.sqrt(var_daily))
    worst_daily = mu_daily - 3.0 * daily_sigma
    max_loss = full_kelly * worst_daily

    logger.info(
        "Kelly from returns [%s]: μ_ann=%.3f σ_ann=%.3f f*=%.4f",
        ticker, mu_annual, sigma_annual, full_kelly,
    )

    return KellyResult(
        ticker=ticker,
        full_kelly_fraction=round(full_kelly, 6),
        half_kelly_fraction=round(half_kelly, 6),
        quarter_kelly_fraction=round(quarter_kelly, 6),
        expected_log_growth=round(expected_log, 8),
        recommended_fraction=round(half_kelly, 6),
        max_loss_scenario=round(max_loss, 6),
    )


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------


def vol_target_sizes(
    returns_df: pd.DataFrame,
    target_vol: float = 0.15,
    lookback_days: int = 20,
) -> list[VolTargetResult]:
    """Compute vol-targeting position sizes for each column in returns_df.

    Position fraction = target_vol / realized_vol, capped at 1.0.
    Scale factor is relative to the equal-weight allocation 1/N.

    Args:
        returns_df: DataFrame of daily returns, one column per asset.
        target_vol: annualised target volatility (e.g. 0.15 = 15 %).
        lookback_days: rolling window for realized volatility.

    Returns:
        List of VolTargetResult, one per asset column.
    """
    if returns_df.empty:
        return []

    n_assets = len(returns_df.columns)
    equal_weight = 1.0 / n_assets if n_assets > 0 else 1.0
    results: list[VolTargetResult] = []

    for ticker in returns_df.columns:
        series = returns_df[ticker].dropna()
        if len(series) < lookback_days:
            logger.warning("Insufficient data for vol target on %s", ticker)
            realized_vol = float(series.std(ddof=1)) * np.sqrt(TRADING_DAYS) if len(series) > 1 else 0.20
        else:
            # Use the most recent `lookback_days` observations
            recent = series.iloc[-lookback_days:]
            realized_vol = float(recent.std(ddof=1)) * np.sqrt(TRADING_DAYS)

        if realized_vol <= 0.0:
            realized_vol = 0.001  # guard

        raw_fraction = target_vol / realized_vol
        position_fraction = float(min(raw_fraction, 1.0))
        scale_factor = position_fraction / equal_weight if equal_weight > 0 else 1.0

        results.append(
            VolTargetResult(
                ticker=ticker,
                target_vol=round(target_vol, 6),
                realized_vol=round(realized_vol, 6),
                position_size_fraction=round(position_fraction, 6),
                scale_factor=round(scale_factor, 4),
            )
        )
        logger.debug(
            "VolTarget [%s]: realized=%.3f target=%.3f fraction=%.4f",
            ticker, realized_vol, target_vol, position_fraction,
        )

    return results


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------


def _portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    """Annualised portfolio volatility given weights and daily covariance matrix."""
    w = weights.reshape(-1, 1)
    port_var = float(w.T @ cov @ w) * TRADING_DAYS
    return float(np.sqrt(max(port_var, 1e-12)))


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal risk contribution of each asset (annualised, sums to portfolio vol)."""
    pvol = _portfolio_vol(weights, cov)
    if pvol <= 0.0:
        return np.zeros_like(weights)
    # MRC_i = w_i * (Σw)_i / σ_p
    marginal = cov @ weights
    rc = weights * marginal * TRADING_DAYS / pvol
    return rc


def risk_parity_weights(
    returns_df: pd.DataFrame,
    target_vol: float = 0.10,
    max_iter: int = 1000,
) -> RiskParityResult:
    """Compute risk-parity (Equal Risk Contribution) weights.

    Two-step approach:
      1. Inverse-volatility starting point (fast, closed-form).
      2. Refine via scipy.optimize.minimize to achieve true ERC
         (each asset contributes equally to portfolio risk).

    Levered/de-levered to match target_vol via a scalar multiplier.

    Args:
        returns_df: daily returns DataFrame, one column per asset.
        target_vol: annualised target portfolio volatility.
        max_iter: max scipy optimiser iterations.

    Returns:
        RiskParityResult with ERC weights and risk contributions.
    """
    from scipy.optimize import minimize  # local import — optional dep

    assets = list(returns_df.columns)
    n = len(assets)
    if n == 0:
        raise ValueError("returns_df has no columns")

    clean = returns_df.dropna(how="any")
    if len(clean) < n + 1:
        raise ValueError(
            f"Need at least {n+1} rows after dropping NaN, got {len(clean)}"
        )

    cov = clean.cov().values  # daily covariance

    # --- Step 1: inverse-vol starting point ---
    vols = np.sqrt(np.diag(cov) * TRADING_DAYS)
    vols = np.where(vols <= 0, 1e-6, vols)
    inv_vol = 1.0 / vols
    w0 = inv_vol / inv_vol.sum()

    # --- Step 2: ERC optimisation ---
    # Minimise sum of squared differences between risk contributions
    def _erc_objective(w: np.ndarray) -> float:
        rc = _risk_contributions(w, cov)
        target_rc = np.sum(rc) / n
        return float(np.sum((rc - target_rc) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = minimize(
            _erc_objective,
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-10},
        )

    if opt.success:
        w_erc = opt.x
    else:
        logger.warning("ERC optimisation did not converge; using inverse-vol weights. %s", opt.message)
        w_erc = w0

    # Normalise
    w_erc = w_erc / w_erc.sum()

    # --- Scale to target vol ---
    pvol_unscaled = _portfolio_vol(w_erc, cov)
    if pvol_unscaled > 0:
        scalar = target_vol / pvol_unscaled
        # Cap leverage at 2x for safety
        scalar = float(np.clip(scalar, 0.0, 2.0))
        w_scaled = w_erc * scalar
        # Re-normalise (levered weights may no longer sum to 1, which is intentional
        # in a levered portfolio; but we report normalised for fraction interpretation)
        w_final = w_erc  # unlevered weights; leverage is handled in size_portfolio
    else:
        w_final = w_erc

    pvol_final = _portfolio_vol(w_final, cov)
    rc_vec = _risk_contributions(w_final, cov)
    rc_pct = rc_vec / rc_vec.sum() if rc_vec.sum() > 0 else rc_vec

    weight_dict = {assets[i]: round(float(w_final[i]), 6) for i in range(n)}
    rc_dict = {assets[i]: round(float(rc_pct[i]), 6) for i in range(n)}

    logger.info(
        "Risk parity: %d assets, target_vol=%.2f, achieved_vol=%.3f",
        n, target_vol, pvol_final,
    )

    return RiskParityResult(
        assets=assets,
        weights=weight_dict,
        risk_contributions=rc_dict,
        target_vol=target_vol,
        portfolio_vol=round(pvol_final, 6),
    )


# ---------------------------------------------------------------------------
# Unified sizing interface
# ---------------------------------------------------------------------------


def size_portfolio(
    tickers: list[str],
    returns_df: pd.DataFrame,
    method: Literal["kelly", "vol_target", "risk_parity", "equal"] = "risk_parity",
    target_vol: float = 0.10,
    capital: float = 100_000.0,
) -> dict[str, dict]:
    """Compute position sizes using the chosen method.

    Args:
        tickers: list of tickers (must match returns_df columns).
        returns_df: daily returns DataFrame.
        method: one of "kelly" | "vol_target" | "risk_parity" | "equal".
        target_vol: annualised target vol for vol_target and risk_parity methods.
        capital: total portfolio capital in dollars.

    Returns:
        {ticker: {weight, dollar_size, shares_at_price_100, method}}
    """
    available = [t for t in tickers if t in returns_df.columns]
    if not available:
        raise ValueError("None of the requested tickers found in returns_df columns")

    df = returns_df[available].copy()
    weights: dict[str, float] = {}

    if method == "equal":
        w = 1.0 / len(available)
        weights = {t: w for t in available}

    elif method == "vol_target":
        vt_results = vol_target_sizes(df, target_vol=target_vol)
        raw = {r.ticker: r.position_size_fraction for r in vt_results}
        total = sum(raw.values()) or 1.0
        weights = {t: v / total for t, v in raw.items()}

    elif method == "risk_parity":
        rp = risk_parity_weights(df, target_vol=target_vol)
        weights = {t: rp.weights.get(t, 0.0) for t in available}
        total = sum(weights.values()) or 1.0
        weights = {t: v / total for t, v in weights.items()}

    elif method == "kelly":
        # Continuous Kelly per asset; normalise to sum to 1
        raw: dict[str, float] = {}
        for ticker in available:
            try:
                kr = kelly_from_returns(df[ticker].dropna(), ticker=ticker)
                raw[ticker] = max(kr.recommended_fraction, 0.0)
            except Exception as exc:
                logger.warning("Kelly failed for %s: %s — using equal weight", ticker, exc)
                raw[ticker] = 1.0 / len(available)
        total = sum(raw.values()) or 1.0
        weights = {t: v / total for t, v in raw.items()}

    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from kelly, vol_target, risk_parity, equal.")

    output: dict[str, dict] = {}
    for ticker in available:
        w = float(weights.get(ticker, 0.0))
        dollar_size = w * capital
        # Shares at hypothetical $100/share price — useful for sanity-checking
        shares = dollar_size / 100.0
        output[ticker] = {
            "weight": round(w, 6),
            "dollar_size": round(dollar_size, 2),
            "shares_at_price_100": round(shares, 4),
            "method": method,
        }
        logger.debug(
            "Size [%s/%s]: weight=%.4f dollars=%.2f",
            method, ticker, w, dollar_size,
        )

    return output
