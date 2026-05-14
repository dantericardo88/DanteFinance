"""Backtest metrics computation — 24 metrics including Deflated Sharpe Ratio and Calmar."""
from __future__ import annotations
import math
from datetime import date
from decimal import Decimal
from typing import Optional
import numpy as np
import pandas as pd
from sentinel.core.types import BacktestMetrics
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

ANNUALIZATION = {
    "1d": 252, "1h": 252 * 6.5, "1m": 252 * 6.5 * 60,
    "5m": 252 * 6.5 * 12, "15m": 252 * 6.5 * 4,
    "1wk": 52, "1mo": 12,
}


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    interval: str = "1d",
    risk_free_rate: float = 0.05,
    n_trials: int = 1,  # Number of trials attempted (for DSR)
    strategy_id: str = "",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> BacktestMetrics:
    """
    Compute the full 24-metric backtest report.
    returns: daily (or period) return Series with DatetimeIndex.
    n_trials: number of parameter combinations tested (used in DSR penalty).
    """
    ann = ANNUALIZATION.get(interval, 252)
    rf_period = risk_free_rate / ann

    rets = returns.dropna()
    if rets.empty:
        raise ValueError("Empty returns series")

    # Core stats
    total_return = float((1 + rets).prod() - 1)
    n = len(rets)
    years = n / ann
    cagr = float((1 + total_return) ** (1 / max(years, 1e-6)) - 1)
    vol = float(rets.std() * math.sqrt(ann))
    sharpe = float((rets.mean() - rf_period) / (rets.std() + 1e-10) * math.sqrt(ann))

    # Sortino (downside deviation)
    downside = rets[rets < 0].std() * math.sqrt(ann)
    sortino = float((rets.mean() - rf_period) / (downside + 1e-10) * math.sqrt(ann))

    # Drawdown
    cum = (1 + rets).cumprod()
    rolling_max = cum.expanding().max()
    dd = (cum - rolling_max) / rolling_max
    max_drawdown = float(dd.min())
    calmar = float(cagr / (abs(max_drawdown) + 1e-10))

    # Win rate and profit factor
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = float(len(wins) / len(rets)) if len(rets) > 0 else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    # Tail risk
    var_95 = float(rets.quantile(0.05))
    cvar_95 = float(rets[rets <= rets.quantile(0.05)].mean())

    # Skew and kurtosis
    skew = float(rets.skew())
    kurt = float(rets.kurt())

    # Beta and alpha vs benchmark
    beta = 0.0
    alpha = 0.0
    information_ratio = 0.0
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(rets.index).dropna()
        aligned_rets = rets.reindex(bench.index).dropna()
        if len(aligned_rets) > 10:
            cov = np.cov(aligned_rets.values, bench.values)
            beta = float(cov[0, 1] / (cov[1, 1] + 1e-10))
            alpha = float((aligned_rets.mean() - beta * bench.mean()) * ann)
            active_return = aligned_rets - bench.reindex(aligned_rets.index)
            information_ratio = float(active_return.mean() / (active_return.std() + 1e-10) * math.sqrt(ann))

    # Deflated Sharpe Ratio
    dsr = compute_dsr(sharpe, n_trials, n, skew, kurt)

    # Turnover placeholder (requires position data — overridden by runner)
    turnover = 0.0

    # Exposure (fraction of periods with non-zero returns as proxy)
    exposure = float((rets != 0).mean())

    return BacktestMetrics(
        strategy_id=strategy_id,
        start_date=start_date or rets.index[0].date(),
        end_date=end_date or rets.index[-1].date(),
        total_return=Decimal(str(round(total_return, 6))),
        cagr=Decimal(str(round(cagr, 6))),
        volatility=Decimal(str(round(vol, 6))),
        sharpe_ratio=Decimal(str(round(sharpe, 4))),
        sortino_ratio=Decimal(str(round(sortino, 4))),
        calmar_ratio=Decimal(str(round(calmar, 4))),
        max_drawdown=Decimal(str(round(max_drawdown, 6))),
        win_rate=Decimal(str(round(win_rate, 4))),
        profit_factor=Decimal(str(round(min(profit_factor, 999.0), 4))),
        avg_win=Decimal(str(round(avg_win, 6))),
        avg_loss=Decimal(str(round(avg_loss, 6))),
        var_95=Decimal(str(round(var_95, 6))),
        cvar_95=Decimal(str(round(cvar_95, 6))),
        beta=Decimal(str(round(beta, 4))),
        alpha=Decimal(str(round(alpha, 6))),
        information_ratio=Decimal(str(round(information_ratio, 4))),
        skewness=Decimal(str(round(skew, 4))),
        kurtosis=Decimal(str(round(kurt, 4))),
        deflated_sharpe_ratio=Decimal(str(round(dsr, 4))),
        turnover=Decimal(str(round(turnover, 4))),
        exposure=Decimal(str(round(exposure, 4))),
        n_trades=len(rets),
        n_trials=n_trials,
    )


def compute_dsr(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float,
    kurt: float,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    Adjusts the Sharpe ratio for the multiple-testing bias from trying many strategies.

    DSR = Prob(SR* ≥ SR_benchmark | data)
    SR_benchmark = SR_max_expected under n_trials IID tests.

    Returns the probability (0-1) that the true Sharpe exceeds the expected maximum
    from n_trials random trials. DSR < 0.95 suggests overfitting.
    """
    from scipy.stats import norm

    if n_trials <= 0 or n_observations < 10:
        return 1.0

    # Expected maximum Sharpe under n_trials independent tests
    # Approximation: E[max SR] ≈ (1 - γ) * z(1 - 1/n) + γ * z(1 - 1/(n*e))
    gamma = 0.5772156649  # Euler-Mascheroni constant
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
    sr_max = (1 - gamma) * z1 + gamma * z2

    # Sharpe ratio standard error with skew/kurtosis correction
    sr_se = math.sqrt(
        (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2)
        / (n_observations - 1)
    )

    # DSR = Phi((SR - SR_max) / SE(SR))
    if sr_se < 1e-10:
        return 1.0 if sharpe > sr_max else 0.0
    dsr = float(norm.cdf((sharpe - sr_max) / sr_se))
    return max(0.0, min(1.0, dsr))


def rolling_sharpe(returns: pd.Series, window: int = 63, ann: int = 252) -> pd.Series:
    """Rolling Sharpe ratio with a given window in periods."""
    return (returns.rolling(window).mean() / (returns.rolling(window).std() + 1e-10)) * math.sqrt(ann)


def rolling_drawdown(returns: pd.Series) -> pd.Series:
    """Rolling max drawdown from inception."""
    cum = (1 + returns).cumprod()
    return (cum / cum.expanding().max() - 1)


def annualize_return(total_return: float, years: float) -> float:
    return (1 + total_return) ** (1 / max(years, 1e-6)) - 1
