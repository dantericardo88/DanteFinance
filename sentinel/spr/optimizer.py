"""
Portfolio optimizer — Markowitz mean-variance, Black-Litterman, and Equal Risk Contribution.

Methods: min_variance | max_sharpe | black_litterman | erc
"""
from __future__ import annotations

import asyncio
import warnings
import datetime as _dt
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.optimize import minimize
from scipy.linalg import LinAlgError

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252
_VALID_METHODS = {"min_variance", "max_sharpe", "black_litterman", "erc"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AssetStats(BaseModel):
    ticker: str
    weight: float
    expected_return: float
    volatility: float
    sharpe: float | None = None


class EfficientFrontierPoint(BaseModel):
    expected_return: float
    volatility: float
    sharpe: float
    weights: dict[str, float]


class OptimizeResult(BaseModel):
    method: str
    weights: dict[str, float]
    portfolio_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    assets: list[AssetStats]
    efficient_frontier: list[EfficientFrontierPoint]
    diversification_ratio: float
    max_drawdown_estimate: float
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Data fetching — asyncio.to_thread pattern (same as var_engine.py)
# ---------------------------------------------------------------------------


def _fetch_returns_sync(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.warning("yfinance_not_installed")
        return pd.DataFrame()

    end = _dt.date.today()
    start = end - _dt.timedelta(days=int(lookback_days * 1.5))
    try:
        raw = yf.download(tickers, start=str(start), end=str(end),
                          auto_adjust=True, progress=False, threads=True)
    except Exception as exc:
        logger.error("yfinance_download_failed", error=str(exc))
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if len(tickers) == 1 and "Close" in close.columns:
        close = close.rename(columns={"Close": tickers[0]})

    available = [t for t in tickers if t in close.columns]
    if not available:
        return pd.DataFrame()

    log_ret = np.log(close[available] / close[available].shift(1)).dropna()
    if len(log_ret) > lookback_days:
        log_ret = log_ret.iloc[-lookback_days:]
    return log_ret


async def _fetch_returns(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    return await asyncio.to_thread(_fetch_returns_sync, tickers, lookback_days)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _annualize(mu_d: np.ndarray, cov_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return mu_d * TRADING_DAYS, cov_d * TRADING_DAYS


def _port_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float
                ) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(max(float(w @ cov @ w), 1e-14)))
    return ret, vol, (ret - rf) / vol if vol > 0 else 0.0


def _eq_w(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _slsqp(obj, w0: np.ndarray, bounds, constraints, maxiter: int = 1000) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(obj, x0=w0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": maxiter, "ftol": 1e-12})
    return res.x if res.success else None


# ---------------------------------------------------------------------------
# Optimizer cores
# ---------------------------------------------------------------------------


def _min_variance(cov: np.ndarray, n: int,
                  min_w: float, max_w: float) -> np.ndarray:
    bounds = [(min_w, max_w)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    w = _slsqp(lambda w: float(w @ cov @ w), _eq_w(n), bounds, cons)
    if w is None:
        logger.warning("min_variance_no_converge")
        return _eq_w(n)
    w = np.clip(w, min_w, max_w); return w / w.sum()


def _max_sharpe(mu: np.ndarray, cov: np.ndarray, n: int, rf: float,
                min_w: float, max_w: float) -> np.ndarray:
    def obj(w: np.ndarray) -> float:
        vol = float(np.sqrt(max(float(w @ cov @ w), 1e-14)))
        return -(float(w @ mu) - rf) / vol

    bounds = [(min_w, max_w)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    mv = _min_variance(mu, cov, n, min_w, max_w)
    w0 = mv if float(mv @ mu) > rf else _eq_w(n)
    w = _slsqp(obj, w0, bounds, cons)
    if w is None:
        logger.warning("max_sharpe_no_converge")
        return _eq_w(n)
    w = np.clip(w, min_w, max_w); return w / w.sum()


def _black_litterman(mu: np.ndarray, cov: np.ndarray, n: int, tickers: list[str],
                     views: dict[str, float], view_confidence: float,
                     rf: float, min_w: float, max_w: float) -> np.ndarray:
    """BL posterior → max_sharpe.  π = δΣw_mkt;  τ = 0.05;  δ = 2.5."""
    tau, delta = 0.05, 2.5
    pi = delta * (cov @ _eq_w(n))  # equilibrium excess returns

    view_tickers = [t for t in views if t in tickers]
    if not view_tickers:
        return _max_sharpe(pi + rf, cov, n, rf, min_w, max_w)

    k = len(view_tickers)
    P = np.zeros((k, n)); Q = np.zeros(k)
    for i, t in enumerate(view_tickers):
        P[i, tickers.index(t)] = 1.0
        Q[i] = views[t] - rf

    view_var = (1.0 - view_confidence) / (view_confidence + 1e-8) * tau
    Omega = np.diag(np.full(k, view_var))
    try:
        tau_cov_inv = np.linalg.inv(tau * cov)
        Omega_inv = np.linalg.inv(Omega)
        M_inv = np.linalg.inv(tau_cov_inv + P.T @ Omega_inv @ P)
        mu_bl = M_inv @ (tau_cov_inv @ pi + P.T @ Omega_inv @ Q) + rf
    except (LinAlgError, np.linalg.LinAlgError) as exc:
        logger.warning("bl_matrix_inversion_failed", error=str(exc))
        mu_bl = mu
    return _max_sharpe(mu_bl, cov, n, rf, min_w, max_w)


def _erc(cov: np.ndarray, n: int, min_w: float, max_w: float) -> np.ndarray:
    """Equal Risk Contribution: minimise Σ_i(RC_i − σ²/N)²."""
    def obj(w: np.ndarray) -> float:
        sigma_sq = float(w @ cov @ w)
        rc = w * (cov @ w)
        return float(np.sum((rc - sigma_sq / n) ** 2))

    vols = np.sqrt(np.diag(cov)); vols[vols <= 0] = 1e-6
    w0 = (1.0 / vols) / (1.0 / vols).sum()
    bounds = [(max(min_w, 1e-6), max_w)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(obj, x0=w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 2000, "ftol": 1e-12})
    w = res.x if res.success else w0
    w = np.clip(w, max(min_w, 1e-6), max_w); return w / w.sum()


# ---------------------------------------------------------------------------
# Efficient frontier (20 points, vary λ)
# ---------------------------------------------------------------------------


def _efficient_frontier(mu: np.ndarray, cov: np.ndarray, tickers: list[str],
                         rf: float, min_w: float, max_w: float,
                         n_points: int = 20) -> list[EfficientFrontierPoint]:
    n = len(tickers)
    lambdas = np.concatenate([np.linspace(0.0, 5.0, n_points // 2),
                               np.linspace(5.0, 100.0, n_points - n_points // 2)])
    bounds = [(min_w, max_w)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    points: list[EfficientFrontierPoint] = []
    seen: set[float] = set()
    w0 = _eq_w(n)

    for lam in lambdas:
        def _obj(w: np.ndarray, l: float = lam) -> float:
            return float(w @ cov @ w) - (1.0 / l if l > 1e-8 else 0.0) * float(mu @ w)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(_obj, x0=w0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-10})
        if not res.success:
            continue
        w = np.clip(res.x, min_w, max_w); w /= w.sum()
        ret, vol, sharpe = _port_stats(w, mu, cov, rf)
        v5 = round(vol, 5)
        if v5 in seen:
            continue
        seen.add(v5)
        points.append(EfficientFrontierPoint(
            expected_return=round(ret, 6), volatility=round(vol, 6),
            sharpe=round(sharpe, 6),
            weights={tickers[i]: round(float(w[i]), 6) for i in range(n)},
        ))
        w0 = w

    points.sort(key=lambda p: p.volatility)
    return points[:n_points]


# ---------------------------------------------------------------------------
# Historical max drawdown
# ---------------------------------------------------------------------------


def _historical_max_drawdown(weights: np.ndarray, returns_df: pd.DataFrame,
                              tickers: list[str]) -> float:
    avail = [t for t in tickers if t in returns_df.columns]
    if not avail:
        return 0.0
    idx = [tickers.index(t) for t in avail]
    w = weights[idx]; w = w / w.sum() if w.sum() > 0 else _eq_w(len(w))
    cum = np.exp((returns_df[avail] * w).sum(axis=1).cumsum())
    mdd = float(abs(((cum - cum.cummax()) / cum.cummax()).min()))
    return round(mdd, 6)


# ---------------------------------------------------------------------------
# Synchronous optimization driver
# ---------------------------------------------------------------------------


def _run_optimization(returns_df: pd.DataFrame, method: str, rf: float,
                       min_w: float, max_w: float,
                       views: Optional[dict[str, float]],
                       view_confidence: float) -> OptimizeResult:
    warn_list: list[str] = []
    tickers = list(returns_df.columns)
    n = len(tickers)

    if n == 0 or len(returns_df) < n + 2:
        warn_list.append(f"Insufficient data ({len(returns_df)} rows, {n} assets). Equal weights.")
        return _fallback_equal(tickers, rf, warn_list, method)

    clean = returns_df.dropna(how="any")
    mu_d = clean.mean().values
    cov_d = clean.cov().values

    # Regularise if not positive semi-definite
    eigmin = float(np.linalg.eigvalsh(cov_d).min())
    if eigmin < 0:
        warn_list.append("Covariance not PSD; regularization applied.")
        cov_d += np.eye(n) * (-eigmin + 1e-8)

    mu, cov = _annualize(mu_d, cov_d)

    if method == "min_variance":
        weights = _min_variance(mu, cov, n, min_w, max_w)
    elif method == "max_sharpe":
        weights = _max_sharpe(mu, cov, n, rf, min_w, max_w)
    elif method == "black_litterman":
        weights = _black_litterman(mu, cov, n, tickers, views or {}, view_confidence,
                                   rf, min_w, max_w)
    else:  # erc
        weights = _erc(cov, n, min_w, max_w)

    port_ret, port_vol, port_sharpe = _port_stats(weights, mu, cov, rf)
    asset_vols = np.sqrt(np.diag(cov))

    assets = [
        AssetStats(
            ticker=tickers[i], weight=round(float(weights[i]), 6),
            expected_return=round(float(mu[i]), 6),
            volatility=round(float(asset_vols[i]), 6),
            sharpe=round((float(mu[i]) - rf) / float(asset_vols[i]), 6)
                   if asset_vols[i] > 0 else None,
        ) for i in range(n)
    ]

    div_ratio = float(np.sum(weights * asset_vols)) / port_vol if port_vol > 0 else 1.0
    frontier = _efficient_frontier(mu, cov, tickers, rf, min_w, max_w)
    if not frontier:
        warn_list.append("Efficient frontier produced no points.")
    mdd = _historical_max_drawdown(weights, clean, tickers)

    return OptimizeResult(
        method=method,
        weights={tickers[i]: round(float(weights[i]), 6) for i in range(n)},
        portfolio_return=round(port_ret, 6),
        portfolio_volatility=round(port_vol, 6),
        portfolio_sharpe=round(port_sharpe, 6),
        assets=assets,
        efficient_frontier=frontier,
        diversification_ratio=round(div_ratio, 6),
        max_drawdown_estimate=mdd,
        warnings=warn_list,
    )


def _fallback_equal(tickers: list[str], warn_list: list[str], method: str) -> OptimizeResult:
    n = len(tickers); w = 1.0 / max(n, 1)
    return OptimizeResult(
        method=method,
        weights={t: round(w, 6) for t in tickers},
        portfolio_return=0.0, portfolio_volatility=0.0, portfolio_sharpe=0.0,
        assets=[AssetStats(ticker=t, weight=round(w, 6), expected_return=0.0,
                           volatility=0.0, sharpe=None) for t in tickers],
        efficient_frontier=[], diversification_ratio=1.0,
        max_drawdown_estimate=0.0, warnings=warn_list,
    )


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------


async def optimize_portfolio(
    tickers: list[str],
    method: str = "max_sharpe",
    risk_free: float = 0.05,
    lookback_days: int = 252,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    views: dict[str, float] | None = None,
    view_confidence: float = 0.5,
) -> OptimizeResult:
    """Optimize a portfolio using mean-variance, BL, or ERC.

    Args:
        tickers: ticker symbols to include.
        method: "min_variance" | "max_sharpe" | "black_litterman" | "erc".
        risk_free: annualised risk-free rate (e.g. 0.05).
        lookback_days: trading days of history to use (default 252).
        min_weight: per-asset lower bound (0.0 = long-only).
        max_weight: per-asset upper bound (1.0 = unconstrained).
        views: Black-Litterman views {ticker: absolute_expected_return}.
        view_confidence: 0–1; strength of views vs equilibrium (BL only).

    Returns:
        OptimizeResult — never raises; errors go into .warnings.
    """
    warn_list: list[str] = []

    if not tickers:
        warn_list.append("Empty tickers list.")
        return _fallback_equal([], risk_free, warn_list, method)

    tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    if method not in _VALID_METHODS:
        warn_list.append(f"Unknown method '{method}'; defaulting to max_sharpe.")
        method = "max_sharpe"

    view_confidence = float(np.clip(view_confidence, 0.0, 1.0))

    logger.info("optimize_portfolio_start", tickers=tickers, method=method,
                lookback_days=lookback_days, risk_free=risk_free)

    try:
        returns_df = await _fetch_returns(tickers, lookback_days)
    except Exception as exc:
        warn_list.append(f"Data fetch failed: {exc}. Equal weights.")
        return _fallback_equal(tickers, risk_free, warn_list, method)

    if returns_df.empty:
        warn_list.append("No return data from yfinance. Equal weights.")
        return _fallback_equal(tickers, risk_free, warn_list, method)

    returns_df = returns_df.dropna(axis=1, how="all")
    missing = set(tickers) - set(returns_df.columns)
    if missing:
        warn_list.append(f"No data for {sorted(missing)}; excluded.")

    surviving = [t for t in tickers if t in returns_df.columns]
    returns_df = returns_df[surviving]

    try:
        result = await asyncio.to_thread(
            _run_optimization, returns_df, method, risk_free,
            min_weight, max_weight, views, view_confidence,
        )
        result.warnings = warn_list + result.warnings
    except Exception as exc:
        warn_list.append(f"Optimization error: {exc}. Equal weights.")
        logger.error("optimize_run_failed", error=str(exc))
        result = _fallback_equal(surviving or tickers, risk_free, warn_list, method)

    logger.info("optimize_portfolio_done", method=result.method,
                portfolio_return=result.portfolio_return,
                portfolio_vol=result.portfolio_volatility,
                sharpe=result.portfolio_sharpe, warnings=len(result.warnings))
    return result
