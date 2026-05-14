"""
Historical simulation, parametric, and Monte Carlo VaR/CVaR engine.

Computes portfolio-level and per-component Value at Risk and Conditional VaR
(Expected Shortfall) across three methods and multiple horizons.
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats as scipy_stats

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

try:
    import yfinance as yf  # type: ignore
    _YF = True
except ImportError:
    _YF = False

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class VaRResult(BaseModel):
    method: str                        # "historical" | "parametric" | "monte_carlo"
    confidence: float                  # 0.95, 0.99
    horizon_days: int                  # 1, 5, 10, 21 (trading days)
    var_pct: float                     # VaR as % of portfolio (e.g., 0.023 = 2.3%)
    var_dollar: Optional[float]        # VaR in dollar terms if portfolio_value given
    cvar_pct: float                    # CVaR/Expected Shortfall (mean of losses > VaR)
    cvar_dollar: Optional[float]
    worst_loss_pct: float              # Max observed loss in sample
    skewness: float
    kurtosis: float
    annualized_vol: float


class PortfolioVaRResult(BaseModel):
    portfolio_var: VaRResult
    component_var: dict[str, VaRResult]         # per-ticker VaR
    diversification_benefit_pct: float          # (sum individual VaR - portfolio VaR) / sum
    correlation_matrix: dict[str, dict[str, float]]
    method: str
    lookback_days: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _fetch_returns_sync(tickers: list[str], lookback_days: int = 504) -> pd.DataFrame:
    """Download daily close prices via yfinance and compute log returns.

    Returns a DataFrame indexed by date with one column per ticker.
    Drops rows where any ticker has NaN so all series share the same length.
    Falls back to an empty DataFrame if yfinance is unavailable.
    """
    if not _YF:
        logger.warning("yfinance_not_installed", tickers=tickers)
        return pd.DataFrame()

    # Extra calendar days to account for weekends / holidays
    calendar_days = int(lookback_days * 1.5)
    import datetime as _dt
    end = _dt.date.today()
    start = end - _dt.timedelta(days=calendar_days)

    logger.info("fetching_prices", tickers=tickers, start=str(start), end=str(end))
    try:
        raw = yf.download(
            tickers,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance_download_failed", error=str(exc))
        return pd.DataFrame()

    if raw.empty:
        logger.warning("yfinance_returned_empty", tickers=tickers)
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when multiple tickers; flatten
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw

    if isinstance(tickers, list) and len(tickers) == 1:
        close = close.rename(columns={"Close": tickers[0]}) if "Close" in close.columns else close

    # Keep only the requested tickers that survived
    available = [t for t in tickers if t in close.columns]
    if not available:
        logger.warning("no_tickers_in_close", tickers=tickers, columns=list(close.columns))
        return pd.DataFrame()

    close = close[available].copy()
    log_returns = np.log(close / close.shift(1)).dropna()

    # Trim to requested lookback
    if len(log_returns) > lookback_days:
        log_returns = log_returns.iloc[-lookback_days:]

    logger.info(
        "returns_fetched",
        rows=len(log_returns),
        tickers=available,
    )
    return log_returns


async def _fetch_returns(tickers: list[str], lookback_days: int = 504) -> pd.DataFrame:
    """Async wrapper around the blocking yfinance call."""
    return await asyncio.to_thread(_fetch_returns_sync, tickers, lookback_days)


# ---------------------------------------------------------------------------
# VaR calculation helpers
# ---------------------------------------------------------------------------


def _dollar(pct: float, portfolio_value: Optional[float]) -> Optional[float]:
    if portfolio_value is None:
        return None
    return round(pct * portfolio_value, 2)


def _historical_var(
    returns: pd.Series,
    confidence: float,
    horizon_days: int,
    portfolio_value: Optional[float] = None,
) -> VaRResult:
    """Historical simulation VaR/CVaR.

    Sorts the empirical return distribution, takes the (1-confidence) percentile
    as VaR, and the mean of losses beyond VaR as CVaR. Scales to horizon via
    square-root-of-time.
    """
    r = returns.dropna().values
    if len(r) == 0:
        raise ValueError("Empty returns series passed to _historical_var")

    scale = math.sqrt(horizon_days)

    # VaR: the loss at (1-confidence) percentile — returns are log returns,
    # negative value means a loss
    var_1d = float(np.percentile(r, (1.0 - confidence) * 100.0))
    # Scale and flip sign so positive var_pct means a loss magnitude
    var_pct = float(-var_1d * scale)

    tail = r[r <= var_1d]
    cvar_1d = float(np.mean(tail)) if len(tail) > 0 else var_1d
    cvar_pct = float(-cvar_1d * scale)

    worst_loss_pct = float(-np.min(r) * scale)

    skewness = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r, fisher=True))  # excess kurtosis
    ann_vol = float(np.std(r, ddof=1) * math.sqrt(TRADING_DAYS))

    return VaRResult(
        method="historical",
        confidence=confidence,
        horizon_days=horizon_days,
        var_pct=round(max(var_pct, 0.0), 6),
        var_dollar=_dollar(max(var_pct, 0.0), portfolio_value),
        cvar_pct=round(max(cvar_pct, 0.0), 6),
        cvar_dollar=_dollar(max(cvar_pct, 0.0), portfolio_value),
        worst_loss_pct=round(max(worst_loss_pct, 0.0), 6),
        skewness=round(skewness, 6),
        kurtosis=round(kurt, 6),
        annualized_vol=round(ann_vol, 6),
    )


def _parametric_var(
    returns: pd.Series,
    confidence: float,
    horizon_days: int,
    portfolio_value: Optional[float] = None,
) -> VaRResult:
    """Parametric (normal distribution) VaR/CVaR.

    VaR = -(μ - z*σ) scaled by sqrt(horizon_days).
    CVaR = -(μ - σ * φ(z)/(1-p)) where φ is the normal PDF and z = Φ⁻¹(p).
    """
    r = returns.dropna().values
    if len(r) == 0:
        raise ValueError("Empty returns series passed to _parametric_var")

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma <= 0:
        sigma = 1e-8

    scale = math.sqrt(horizon_days)
    z = float(scipy_stats.norm.ppf(confidence))

    # 1-day VaR (loss magnitude — positive number)
    var_1d = -(mu - z * sigma)
    var_pct = float(var_1d * scale)

    # CVaR = E[loss | loss > VaR] under normal assumption
    phi_z = float(scipy_stats.norm.pdf(z))
    cvar_1d = -(mu - sigma * phi_z / (1.0 - confidence))
    cvar_pct = float(cvar_1d * scale)

    worst_loss_pct = float(-np.min(r) * scale)
    skewness = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r, fisher=True))
    ann_vol = float(sigma * math.sqrt(TRADING_DAYS))

    return VaRResult(
        method="parametric",
        confidence=confidence,
        horizon_days=horizon_days,
        var_pct=round(max(var_pct, 0.0), 6),
        var_dollar=_dollar(max(var_pct, 0.0), portfolio_value),
        cvar_pct=round(max(cvar_pct, 0.0), 6),
        cvar_dollar=_dollar(max(cvar_pct, 0.0), portfolio_value),
        worst_loss_pct=round(max(worst_loss_pct, 0.0), 6),
        skewness=round(skewness, 6),
        kurtosis=round(kurt, 6),
        annualized_vol=round(ann_vol, 6),
    )


def _monte_carlo_var(
    returns: pd.Series,
    confidence: float,
    horizon_days: int,
    n_simulations: int = 10_000,
    portfolio_value: Optional[float] = None,
) -> VaRResult:
    """Monte Carlo VaR/CVaR.

    Fits μ and σ from the empirical daily return distribution, then draws
    n_simulations * horizon_days normal samples. Each simulation compounds
    horizon_days daily draws to produce a horizon P&L. VaR and CVaR are taken
    from the simulated distribution at the requested confidence level.
    """
    r = returns.dropna().values
    if len(r) == 0:
        raise ValueError("Empty returns series passed to _monte_carlo_var")

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma <= 0:
        sigma = 1e-8

    rng = np.random.default_rng(seed=42)
    # Shape: (n_simulations, horizon_days) daily log returns
    daily_draws = rng.normal(loc=mu, scale=sigma, size=(n_simulations, horizon_days))
    # Sum of log returns = log compound return over horizon
    horizon_log_returns = daily_draws.sum(axis=1)

    var_threshold = float(np.percentile(horizon_log_returns, (1.0 - confidence) * 100.0))
    var_pct = float(-var_threshold)

    tail = horizon_log_returns[horizon_log_returns <= var_threshold]
    cvar_threshold = float(np.mean(tail)) if len(tail) > 0 else var_threshold
    cvar_pct = float(-cvar_threshold)

    worst_loss_pct = float(-np.min(horizon_log_returns))
    skewness = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r, fisher=True))
    ann_vol = float(sigma * math.sqrt(TRADING_DAYS))

    return VaRResult(
        method="monte_carlo",
        confidence=confidence,
        horizon_days=horizon_days,
        var_pct=round(max(var_pct, 0.0), 6),
        var_dollar=_dollar(max(var_pct, 0.0), portfolio_value),
        cvar_pct=round(max(cvar_pct, 0.0), 6),
        cvar_dollar=_dollar(max(cvar_pct, 0.0), portfolio_value),
        worst_loss_pct=round(max(worst_loss_pct, 0.0), 6),
        skewness=round(skewness, 6),
        kurtosis=round(kurt, 6),
        annualized_vol=round(ann_vol, 6),
    )


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------


def _portfolio_returns(weights: dict[str, float], returns: pd.DataFrame) -> pd.Series:
    """Compute weighted portfolio return series from individual ticker returns.

    Weights are normalised to sum to 1 across tickers present in returns.
    """
    available = {t: w for t, w in weights.items() if t in returns.columns}
    if not available:
        raise ValueError(
            f"No weight tickers found in returns. "
            f"weights={list(weights)}, columns={list(returns.columns)}"
        )

    total_weight = sum(available.values())
    if total_weight <= 0:
        raise ValueError("Sum of available weights is <= 0")

    norm_weights = {t: w / total_weight for t, w in available.items()}
    port = sum(returns[t] * w for t, w in norm_weights.items())
    port.name = "_portfolio"
    return port


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def _build_correlation_matrix(returns: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute pairwise Pearson correlation matrix from return DataFrame."""
    corr = returns.corr()
    result: dict[str, dict[str, float]] = {}
    for col in corr.columns:
        result[str(col)] = {str(idx): round(float(corr.loc[idx, col]), 6) for idx in corr.index}
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _compute_var(
    series: pd.Series,
    method: str,
    confidence: float,
    horizon_days: int,
    portfolio_value: Optional[float],
) -> VaRResult:
    if method == "historical":
        return _historical_var(series, confidence, horizon_days, portfolio_value)
    elif method == "parametric":
        return _parametric_var(series, confidence, horizon_days, portfolio_value)
    elif method == "monte_carlo":
        return _monte_carlo_var(series, confidence, horizon_days, portfolio_value=portfolio_value)
    else:
        raise ValueError(f"Unknown VaR method: {method!r}. Use historical, parametric, or monte_carlo.")


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------


async def compute_portfolio_var(
    weights: dict[str, float],
    confidence: float = 0.95,
    horizon_days: int = 1,
    method: str = "historical",
    lookback_days: int = 504,
    portfolio_value: Optional[float] = None,
) -> PortfolioVaRResult:
    """Compute portfolio-level and per-component VaR/CVaR.

    Args:
        weights: {ticker: weight} — need not sum to 1; will be normalised.
        confidence: VaR confidence level (e.g. 0.95 for 95% VaR).
        horizon_days: horizon in trading days (1, 5, 10, 21, etc.).
        method: "historical", "parametric", or "monte_carlo".
        lookback_days: number of trading days of history to fetch (default 2 years).
        portfolio_value: total portfolio value in dollars for dollar-VaR figures.

    Returns:
        PortfolioVaRResult with portfolio and component VaR/CVaR and correlation matrix.
    """
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")

    tickers = list(weights.keys())
    logger.info(
        "compute_portfolio_var",
        tickers=tickers,
        method=method,
        confidence=confidence,
        horizon_days=horizon_days,
        lookback_days=lookback_days,
    )

    returns = await _fetch_returns(tickers, lookback_days)

    if returns.empty:
        raise RuntimeError(
            "Could not fetch return data for tickers. "
            "Check that yfinance is installed and tickers are valid."
        )

    # Drop tickers that came back with all NaN
    returns = returns.dropna(axis=1, how="all")
    live_tickers = [t for t in tickers if t in returns.columns]
    missing = set(tickers) - set(live_tickers)
    if missing:
        logger.warning("tickers_missing_from_returns", missing=list(missing))

    # Portfolio-level returns
    port_returns = _portfolio_returns(weights, returns)

    # Portfolio VaR
    portfolio_var = _compute_var(port_returns, method, confidence, horizon_days, portfolio_value)

    # Component VaR — each ticker individually
    component_var: dict[str, VaRResult] = {}
    for ticker in live_tickers:
        try:
            component_var[ticker] = _compute_var(
                returns[ticker],
                method,
                confidence,
                horizon_days,
                portfolio_value=None,  # component VaR in % terms
            )
        except Exception as exc:
            logger.warning("component_var_failed", ticker=ticker, error=str(exc))

    # Diversification benefit
    available_weights = {t: w for t, w in weights.items() if t in component_var}
    total_w = sum(available_weights.values()) or 1.0
    sum_individual_var = sum(
        component_var[t].var_pct * (available_weights[t] / total_w)
        for t in available_weights
    )
    port_var_pct = portfolio_var.var_pct
    if sum_individual_var > 0:
        diversification_benefit = (sum_individual_var - port_var_pct) / sum_individual_var
    else:
        diversification_benefit = 0.0

    # Correlation matrix
    corr_tickers = [t for t in live_tickers if t in returns.columns]
    corr_matrix = _build_correlation_matrix(returns[corr_tickers]) if len(corr_tickers) > 1 else {}

    result = PortfolioVaRResult(
        portfolio_var=portfolio_var,
        component_var=component_var,
        diversification_benefit_pct=round(float(diversification_benefit), 6),
        correlation_matrix=corr_matrix,
        method=method,
        lookback_days=lookback_days,
        generated_at=datetime.utcnow(),
    )

    logger.info(
        "portfolio_var_computed",
        method=method,
        var_pct=portfolio_var.var_pct,
        cvar_pct=portfolio_var.cvar_pct,
        diversification_benefit=diversification_benefit,
        n_components=len(component_var),
    )

    return result
