"""GARCH(1,1)-based conditional VaR/CVaR with Kupiec/Christoffersen backtesting.

Upgrades SENTINEL competitive matrix dim-77 (VaR/CVaR) from static methods (7/10)
to GARCH-conditional estimates targeting 9/10. Additive — does not touch var_engine.py.
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats as scipy_stats
from scipy.optimize import minimize

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

try:
    import yfinance as yf  # type: ignore
    _YF = True
except ImportError:
    _YF = False

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GARCHParams(BaseModel):
    omega: float         # variance constant
    alpha: float         # ARCH coefficient (shock impact)
    beta: float          # GARCH coefficient (variance persistence)
    long_run_vol: float  # sqrt(omega/(1-alpha-beta)) annualised


class BacktestResult(BaseModel):
    exceedances: int
    exceedance_rate: float
    kupiec_pvalue: float          # Kupiec POF test p-value
    christoffersen_pvalue: float  # independence test p-value
    traffic_light: str            # "green" | "yellow" | "red" (Basel III zones)


class GARCHVaRResult(BaseModel):
    tickers: list[str]
    weights: list[float]
    confidence: float
    horizon_days: int
    portfolio_value: float
    current_conditional_vol: float    # annualised
    garch_var_1d: float               # in $ terms
    garch_cvar_1d: float              # in $ terms
    garch_var_Nd: float               # horizon-scaled VaR in $ terms
    garch_params: GARCHParams
    stress_var: float                 # worst-5d rolling window, $ terms
    stress_scenario: str              # description of the worst window
    backtest: BacktestResult
    vol_forecast: dict[str, float]    # {"5d": x, "10d": y, "21d": z} annualised
    warnings: list[str]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _fetch_returns(tickers: list[str], weights: list[float]) -> pd.Series:
    """Download 2y adjusted closes via yfinance; return weighted portfolio log-returns."""
    if not _YF:
        logger.warning("yfinance_not_installed", tickers=tickers)
        return pd.Series(dtype=float)
    try:
        raw = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
        if raw.empty:
            return pd.Series(dtype=float)
        prices = raw["Close"] if len(tickers) > 1 else raw[["Close"]].rename(columns={"Close": tickers[0]})
        prices = prices[tickers].dropna()
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return (log_ret * np.array(weights)).sum(axis=1)
    except Exception as exc:  # noqa: BLE001
        logger.error("garch_fetch_error", error=str(exc))
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# GARCH(1,1) core (pure numpy + scipy — no arch/statsmodels)
# ---------------------------------------------------------------------------

def _sigma2_series(omega: float, alpha: float, beta: float, r: np.ndarray) -> np.ndarray:
    """Compute conditional variance series for GARCH(1,1)."""
    h = np.empty(len(r))
    h[0] = max(np.var(r), 1e-10)
    for t in range(1, len(r)):
        h[t] = max(omega + alpha * r[t - 1] ** 2 + beta * h[t - 1], 1e-12)
    return h


def _neg_loglik(params: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = params
    h = _sigma2_series(omega, alpha, beta, r)
    if np.any(h <= 0):
        return 1e10
    return float(0.5 * np.sum(np.log(h) + r ** 2 / h))


def _fit_garch(r: np.ndarray) -> tuple[float, float, float, bool]:
    """Fit GARCH(1,1) via SLSQP; fall back to Nelder-Mead. Returns (omega,alpha,beta,ok)."""
    v0 = max(float(np.var(r)), 1e-10)
    x0 = np.array([v0 * 0.15, 0.10, 0.85])
    bounds = [(1e-9, None), (0.0, 0.999), (0.0, 0.999)]
    con = {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}

    res = minimize(_neg_loglik, x0, args=(r,), method="SLSQP", bounds=bounds,
                   constraints=con, options={"ftol": 1e-9, "maxiter": 500, "disp": False})
    if res.success and res.x[1] + res.x[2] < 1.0:
        return float(res.x[0]), float(res.x[1]), float(res.x[2]), True

    res2 = minimize(_neg_loglik, x0, args=(r,), method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000, "disp": False})
    o, a, b = res2.x
    if a + b >= 1.0:
        s = 0.9999 / (a + b + 1e-10)
        a, b = a * s, b * s
    return float(max(o, 1e-9)), float(max(a, 0.0)), float(max(b, 0.0)), res2.success and (a + b < 1.0)


def _forecast_vols(omega: float, alpha: float, beta: float, h0: float,
                   horizons: list[int]) -> dict[str, float]:
    """Propagate GARCH variance forward; return annualised vol per horizon."""
    ab = alpha + beta
    h, cum, out = h0, 0.0, {}
    for k in range(1, max(horizons) + 1):
        h = omega + ab * h
        cum += h
        if k in horizons:
            out[f"{k}d"] = float(math.sqrt(cum / k * TRADING_DAYS))
    return out


# ---------------------------------------------------------------------------
# Backtesting: Kupiec POF + Christoffersen independence
# ---------------------------------------------------------------------------

def _kupiec_pvalue(n: int, x: int, conf: float) -> float:
    p = 1.0 - conf
    if x == 0:
        lr = -2.0 * n * math.log(max(1.0 - p, 1e-15))
    elif x == n:
        lr = -2.0 * n * math.log(max(p, 1e-15))
    else:
        ph = x / n
        try:
            lr = -2.0 * (x * math.log(p / ph) + (n - x) * math.log((1 - p) / (1 - ph)))
        except (ValueError, ZeroDivisionError):
            return 0.0
    return float(1.0 - scipy_stats.chi2.cdf(max(lr, 0.0), df=1))


def _christoffersen_pvalue(hits: np.ndarray) -> float:
    n00 = n01 = n10 = n11 = 0
    for i in range(len(hits) - 1):
        a, b = int(hits[i]), int(hits[i + 1])
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    total = n0 + n1
    if total == 0:
        return 1.0
    pi_hat = (n01 + n11) / total
    pi_01 = n01 / n0 if n0 > 0 else 0.0
    pi_11 = n11 / n1 if n1 > 0 else 0.0

    def sl(x: float, c: int) -> float:
        return c * math.log(x) if c > 0 and x > 0 else 0.0

    ll0 = sl(1 - pi_hat, n00 + n10) + sl(pi_hat, n01 + n11)
    ll1 = sl(1 - pi_01, n00) + sl(pi_01, n01) + sl(1 - pi_11, n10) + sl(pi_11, n11)
    return float(1.0 - scipy_stats.chi2.cdf(max(-2.0 * (ll0 - ll1), 0.0), df=1))


def _traffic_light(x: int, n: int, conf: float) -> str:
    e = n * (1.0 - conf)
    if x <= max(4, math.floor(e * 1.5)):
        return "green"
    if x <= max(9, math.floor(e * 3.0)):
        return "yellow"
    return "red"


def _backtest(r: np.ndarray, backtest_days: int, conf: float, pv: float) -> BacktestResult:
    n = len(r)
    min_train = max(100, n - backtest_days)
    eval_idx = range(min_train, n)
    if len(eval_idx) < 20:
        return BacktestResult(exceedances=0, exceedance_rate=0.0,
                              kupiec_pvalue=1.0, christoffersen_pvalue=1.0, traffic_light="green")

    omega_l = alpha_l = beta_l = 0.0
    ok_l = False
    hits: list[int] = []

    for step, idx in enumerate(eval_idx):
        if step % 21 == 0:
            omega_l, alpha_l, beta_l, ok_l = _fit_garch(r[:idx])

        if ok_l and alpha_l + beta_l < 1.0:
            h_t = float(_sigma2_series(omega_l, alpha_l, beta_l, r[:idx])[-1])
        else:
            h_t = float(pd.Series(r[:idx]).ewm(span=30, adjust=False).var().iloc[-1])

        mu_t = float(np.mean(r[:idx]))
        sigma_t = math.sqrt(max(h_t, 1e-12))
        var_d = -pv * (mu_t + sigma_t * float(scipy_stats.norm.ppf(1.0 - conf)))
        hits.append(1 if -r[idx] * pv > var_d else 0)

    ha = np.array(hits, dtype=int)
    x, nb = int(ha.sum()), len(ha)
    return BacktestResult(
        exceedances=x,
        exceedance_rate=x / nb,
        kupiec_pvalue=_kupiec_pvalue(nb, x, conf),
        christoffersen_pvalue=_christoffersen_pvalue(ha),
        traffic_light=_traffic_light(x, nb, conf),
    )


# ---------------------------------------------------------------------------
# Stress VaR
# ---------------------------------------------------------------------------

def _stress_var(ret: pd.Series, pv: float) -> tuple[float, str]:
    roll = ret.rolling(5).sum().dropna()
    if roll.empty:
        return 0.0, "insufficient history"
    end_dt = roll.idxmin()
    worst = float(roll.min())
    end_pos = ret.index.get_loc(end_dt)
    start_dt = ret.index[max(0, end_pos - 4)]
    desc = f"Worst 5-day window: {start_dt.date()} to {end_dt.date()}, return={worst:.2%}"
    return float(pv * abs(worst)), desc


# ---------------------------------------------------------------------------
# Synchronous core
# ---------------------------------------------------------------------------

def _run(tickers: list[str], weights: list[float], confidence: float,
         horizon_days: int, portfolio_value: float, backtest_days: int) -> GARCHVaRResult:
    warnings: list[str] = []

    port_ret = _fetch_returns(tickers, weights)
    if port_ret.empty or len(port_ret) < 60:
        raise ValueError(f"Insufficient return history ({len(port_ret)} obs, need >= 60).")

    r = port_ret.values.astype(np.float64)
    mu = float(np.mean(r))

    # Fit GARCH
    omega, alpha, beta, ok = _fit_garch(r)
    fallback = not ok or (alpha + beta) >= 1.0

    if fallback:
        warnings.append(
            f"GARCH(1,1) non-stationary (alpha+beta={alpha+beta:.4f}); "
            "falling back to EWM(span=30) volatility."
        )
        h_curr = float(pd.Series(r).ewm(span=30, adjust=False).var().iloc[-1])
        beta = 1.0 - 2.0 / 31.0   # EWM span=30 equivalent
        alpha = 1.0 - beta
        omega = max(h_curr * (1.0 - alpha - beta), 1e-9)
        lr_vol = float(np.std(r)) * math.sqrt(TRADING_DAYS)
    else:
        h_curr = float(_sigma2_series(omega, alpha, beta, r)[-1])
        lr_vol = math.sqrt(omega / (1.0 - alpha - beta) * TRADING_DAYS)

    garch_params = GARCHParams(omega=omega, alpha=alpha, beta=beta, long_run_vol=lr_vol)

    # Conditional VaR
    sigma_d = math.sqrt(max(h_curr, 1e-12))
    cond_vol_ann = sigma_d * math.sqrt(TRADING_DAYS)
    z = float(scipy_stats.norm.ppf(1.0 - confidence))
    pdf_z = float(scipy_stats.norm.pdf(-z))

    var_1d   = -portfolio_value * (mu + sigma_d * z)
    cvar_1d  = -portfolio_value * (mu + sigma_d * pdf_z / (1.0 - confidence))
    var_Nd   = -portfolio_value * (mu * horizon_days + sigma_d * math.sqrt(horizon_days) * z)

    # Stress
    sv, ss = _stress_var(port_ret, portfolio_value)

    # Backtest
    bt = _backtest(r, backtest_days, confidence, portfolio_value)

    # Vol forecast
    if fallback:
        vf = {f"{hz}d": cond_vol_ann for hz in [5, 10, 21]}
    else:
        vf = _forecast_vols(omega, alpha, beta, h_curr, [5, 10, 21])

    logger.info("garch_var_done", tickers=tickers, var_1d=round(var_1d, 2),
                cond_vol=round(cond_vol_ann, 4), traffic_light=bt.traffic_light)

    return GARCHVaRResult(
        tickers=tickers, weights=weights, confidence=confidence,
        horizon_days=horizon_days, portfolio_value=portfolio_value,
        current_conditional_vol=cond_vol_ann,
        garch_var_1d=float(var_1d), garch_cvar_1d=float(cvar_1d), garch_var_Nd=float(var_Nd),
        garch_params=garch_params,
        stress_var=sv, stress_scenario=ss,
        backtest=bt, vol_forecast=vf, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------

async def compute_garch_var(
    tickers: list[str],
    weights: list[float] | None = None,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 1_000_000,
    backtest_days: int = 252,
) -> GARCHVaRResult:
    """GARCH(1,1) conditional VaR/CVaR for a portfolio.

    Parameters
    ----------
    tickers:        Yahoo Finance ticker symbols.
    weights:        Portfolio weights (normalised if not summing to 1). Equal-weight if None.
    confidence:     VaR confidence level, e.g. 0.95 or 0.99.
    horizon_days:   Holding period in trading days for N-day VaR.
    portfolio_value: Total portfolio value in dollars.
    backtest_days:  Trailing trading days used for the expanding-window backtest.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")

    if weights is None:
        weights = [1.0 / len(tickers)] * len(tickers)
    else:
        if len(weights) != len(tickers):
            raise ValueError("weights and tickers must have the same length.")
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            weights = [w / total for w in weights]

    return await asyncio.to_thread(
        _run, tickers, weights, confidence, horizon_days, portfolio_value, backtest_days
    )
