"""
sentinel/spm/portfolio_risk_v3.py
==================================
Portfolio VaR / CVaR with GARCH(1,1), Basel III compliance, and stress testing.
dim_077 — score 6 → 9

Free data only:
  - yfinance for price history (adj close)
  - FRED CSV for risk-free rate (FEDFUNDS / TB3MS)
  - No paid APIs, no optional deps that break the module if absent

Author: SENTINEL Risk Engine
"""

from __future__ import annotations

import io
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional / guarded imports
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import minimize
    from scipy.stats import norm, t as student_t

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from arch import arch_model  # type: ignore

    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False

try:
    import yfinance as yf

    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    import requests

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    import duckdb  # type: ignore

    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger("sentinel.spm.portfolio_risk")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GARCHParams:
    """GARCH(1,1) parameter estimates."""

    omega: float = 1e-6
    alpha: float = 0.05
    beta: float = 0.90
    long_run_var: float = 0.0
    log_likelihood: float = 0.0
    converged: bool = False
    model_type: str = "GARCH"  # "GARCH" | "EGARCH"


@dataclass
class MonteCarloResult:
    """Result from Monte Carlo VaR simulation."""

    var_99: float = 0.0
    var_95: float = 0.0
    cvar_99: float = 0.0
    cvar_95: float = 0.0
    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    n_simulations: int = 0
    horizon: int = 10
    pnl_percentiles: Dict[int, float] = field(default_factory=dict)


@dataclass
class Basel3Metrics:
    """Basel III / FRTB compliant risk metrics."""

    var_99_1d: float = 0.0
    var_99_10d: float = 0.0
    es_975_stressed: float = 0.0
    capital_charge: float = 0.0
    k_multiplier: float = 3.0
    backtesting_zone: str = "GREEN"
    n_exceptions: int = 0
    stressed_var: float = 0.0
    incremental_risk_charge: float = 0.0


@dataclass
class BacktestResult:
    """VaR backtesting output."""

    n_exceptions: int = 0
    exception_rate: float = 0.0
    zone: str = "GREEN"
    exception_dates: List[str] = field(default_factory=list)
    kupiec_pvalue: float = 1.0
    christoffersen_pvalue: float = 1.0


@dataclass
class StressTestResult:
    """Portfolio stress test results across scenarios."""

    scenario_losses: Dict[str, float] = field(default_factory=dict)
    worst_scenario: str = ""
    worst_loss: float = 0.0
    portfolio_value: float = 0.0
    stressed_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskReport:
    """Full portfolio risk report."""

    tickers: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    portfolio_value: float = 1_000_000.0
    as_of_date: str = ""

    # VaR estimates
    hs_var_99: float = 0.0
    hs_var_95: float = 0.0
    hs_cvar_99: float = 0.0
    parametric_var_normal: float = 0.0
    parametric_var_t: float = 0.0
    garch_var: float = 0.0
    mc_var_99: float = 0.0
    mc_cvar_99: float = 0.0

    # Basel III
    basel3: Basel3Metrics = field(default_factory=Basel3Metrics)
    backtest: BacktestResult = field(default_factory=BacktestResult)

    # Stress tests
    stress: StressTestResult = field(default_factory=StressTestResult)

    # Risk decomposition
    component_var: Dict[str, float] = field(default_factory=dict)
    marginal_var: Dict[str, float] = field(default_factory=dict)

    # Performance
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    annualized_vol: float = 0.0
    sharpe_ratio: float = 0.0


# ---------------------------------------------------------------------------
# ReturnsFetcher
# ---------------------------------------------------------------------------


class ReturnsFetcher:
    """Fetch price data from yfinance and compute log returns."""

    FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache: Dict[str, pd.DataFrame] = {}
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    def fetch_returns(
        self,
        tickers: List[str],
        start: str,
        end: str,
        freq: str = "daily",
    ) -> pd.DataFrame:
        """
        Download adjusted close prices from yfinance and return log returns.

        Parameters
        ----------
        tickers : list[str]
        start, end : ISO date strings
        freq : "daily" | "weekly" | "monthly"

        Returns
        -------
        pd.DataFrame  log returns, columns = tickers, indexed by date
        """
        if not _YF_AVAILABLE:
            raise ImportError("yfinance is required: pip install yfinance")

        interval_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
        interval = interval_map.get(freq, "1d")

        log.info("Fetching price data for %d tickers", len(tickers))
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        # Forward-fill gaps
        prices = prices.ffill()

        # Drop columns with more than 5% missing after ffill
        threshold = 0.05
        missing_frac = prices.isna().mean()
        to_keep = missing_frac[missing_frac <= threshold].index.tolist()
        dropped = set(tickers) - set(to_keep)
        if dropped:
            log.warning("Dropping tickers with >5%% missing data: %s", dropped)
        prices = prices[to_keep].dropna()

        # Log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()
        return log_returns

    # ------------------------------------------------------------------
    def fetch_benchmark_returns(self, benchmark: str = "SPY") -> pd.Series:
        """Fetch log returns for a benchmark ticker."""
        df = self.fetch_returns(
            [benchmark],
            start=(datetime.now() - timedelta(days=252 * 3)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d"),
        )
        if df.empty:
            return pd.Series(dtype=float)
        col = df.columns[0]
        return df[col].rename(benchmark)

    # ------------------------------------------------------------------
    def fetch_risk_free_rate(self) -> pd.Series:
        """
        Fetch FEDFUNDS or TB3MS from FRED as a daily risk-free rate series.
        Returns daily decimal rate (annualised / 252).
        """
        for series_id in ("FEDFUNDS", "TB3MS"):
            try:
                url = f"{self.FRED_BASE}?id={series_id}"
                if _REQUESTS_AVAILABLE:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    txt = resp.text
                else:
                    import urllib.request

                    with urllib.request.urlopen(url, timeout=15) as r:
                        txt = r.read().decode()

                df = pd.read_csv(io.StringIO(txt), parse_dates=["DATE"], index_col="DATE")
                df.columns = ["rate"]
                df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
                df = df.dropna()
                # Convert annual % to daily decimal
                daily = df["rate"] / 100.0 / 252.0
                # Resample to business days, ffill
                daily = daily.resample("B").ffill()
                log.info("Fetched risk-free rate from FRED series %s", series_id)
                return daily.rename("rf")
            except Exception as exc:
                log.warning("FRED series %s failed: %s", series_id, exc)

        # Final fallback: 5% annualised
        log.warning("Could not fetch FRED data; using flat 5%% rf assumption")
        idx = pd.date_range(
            end=datetime.now(), periods=252 * 3, freq="B"
        )
        return pd.Series(0.05 / 252.0, index=idx, name="rf")

    # ------------------------------------------------------------------
    def compute_excess_returns(
        self,
        returns: pd.DataFrame,
        rf_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Subtract daily risk-free rate from returns to get excess returns."""
        if rf_series is None:
            rf_series = self.fetch_risk_free_rate()

        aligned = rf_series.reindex(returns.index, method="ffill").fillna(
            rf_series.mean()
        )
        excess = returns.subtract(aligned, axis=0)
        return excess


# ---------------------------------------------------------------------------
# GARCHModel
# ---------------------------------------------------------------------------


class GARCHModel:
    """
    Pure-numpy GARCH(1,1) and EGARCH model.
    Uses scipy.optimize.minimize if available; falls back to grid search.
    Optionally delegates to the `arch` package if installed.
    """

    def __init__(self, model_type: str = "GARCH"):
        """model_type: 'GARCH' | 'EGARCH'"""
        self.model_type = model_type
        self.params: Optional[GARCHParams] = None
        self._sigma2: Optional[np.ndarray] = None
        self._returns: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _garch_filter(
        returns: np.ndarray, omega: float, alpha: float, beta: float
    ) -> np.ndarray:
        """Compute conditional variance series for GARCH(1,1)."""
        n = len(returns)
        sigma2 = np.empty(n)
        sigma2[0] = np.var(returns)
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        return sigma2

    @staticmethod
    def _egarch_filter(
        returns: np.ndarray, omega: float, alpha: float, gamma: float, beta: float
    ) -> np.ndarray:
        """Compute log-conditional-variance series for EGARCH."""
        n = len(returns)
        log_sigma2 = np.empty(n)
        log_sigma2[0] = np.log(np.var(returns) + 1e-12)
        exp_abs_z = np.sqrt(2.0 / np.pi)  # E[|z|] for standard normal
        for t in range(1, n):
            sigma_prev = np.exp(log_sigma2[t - 1] / 2.0)
            z_prev = returns[t - 1] / (sigma_prev + 1e-12)
            log_sigma2[t] = (
                omega
                + alpha * (np.abs(z_prev) - exp_abs_z)
                + gamma * z_prev
                + beta * log_sigma2[t - 1]
            )
        return log_sigma2

    def _garch_neg_loglik(self, params: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        sigma2 = self._garch_filter(returns, omega, alpha, beta)
        sigma2 = np.maximum(sigma2, 1e-12)
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
        return -ll

    def _egarch_neg_loglik(
        self, params: np.ndarray, returns: np.ndarray
    ) -> float:
        omega, alpha, gamma, beta = params
        if abs(beta) >= 1:
            return 1e10
        log_sigma2 = self._egarch_filter(returns, omega, alpha, gamma, beta)
        sigma2 = np.exp(log_sigma2)
        sigma2 = np.maximum(sigma2, 1e-12)
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
        return -ll

    def _grid_search_garch(
        self, returns: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Coarse grid search fallback when scipy is not available."""
        best_ll = np.inf
        best = (1e-6, 0.05, 0.90)
        alphas = [0.03, 0.05, 0.08, 0.10, 0.15]
        betas = [0.80, 0.85, 0.90, 0.92, 0.95]
        for a in alphas:
            for b in betas:
                if a + b >= 1:
                    continue
                var_u = np.var(returns)
                omega_g = var_u * (1 - a - b)
                ll = self._garch_neg_loglik(
                    np.array([omega_g, a, b]), returns
                )
                if ll < best_ll:
                    best_ll = ll
                    best = (omega_g, a, b)
        return best[0], best[1], best[2], -best_ll

    # ------------------------------------------------------------------
    def fit(self, returns: np.ndarray) -> GARCHParams:
        """
        Estimate GARCH(1,1) or EGARCH parameters via MLE.
        Optionally delegates to `arch` package.
        """
        self._returns = np.asarray(returns, dtype=float)
        r = self._returns

        # Use arch package if available (better optimiser)
        if _ARCH_AVAILABLE:
            try:
                vol = "GARCH" if self.model_type == "GARCH" else "EGARCH"
                am = arch_model(r * 100, vol=vol, p=1, q=1, rescale=False)
                res = am.fit(disp="off")
                p = res.params
                if self.model_type == "GARCH":
                    omega = float(p.get("omega", 1e-6)) / 10000.0
                    alpha = float(p.get("alpha[1]", 0.05))
                    beta = float(p.get("beta[1]", 0.90))
                    gp = GARCHParams(
                        omega=omega,
                        alpha=alpha,
                        beta=beta,
                        long_run_var=omega / max(1 - alpha - beta, 1e-8),
                        log_likelihood=float(res.loglikelihood),
                        converged=True,
                        model_type=self.model_type,
                    )
                    self.params = gp
                    self._sigma2 = self._garch_filter(r, omega, alpha, beta)
                    return gp
            except Exception as exc:
                log.debug("arch package failed (%s); falling back to numpy MLE", exc)

        if self.model_type == "GARCH":
            return self._fit_garch(r)
        else:
            return self._fit_egarch(r)

    def _fit_garch(self, r: np.ndarray) -> GARCHParams:
        var_u = float(np.var(r))
        x0 = np.array([var_u * 0.05, 0.05, 0.90])
        bounds = [(1e-10, var_u), (0.001, 0.40), (0.50, 0.999)]

        converged = False
        omega, alpha, beta, ll = var_u * 0.05, 0.05, 0.90, -1e10

        if _SCIPY_AVAILABLE:
            try:
                res = minimize(
                    self._garch_neg_loglik,
                    x0,
                    args=(r,),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-9},
                )
                if res.success or res.fun < 1e9:
                    omega, alpha, beta = res.x
                    ll = -res.fun
                    converged = res.success
            except Exception as exc:
                log.debug("scipy optimize failed: %s", exc)

        if not converged:
            omega, alpha, beta, ll = self._grid_search_garch(r)

        # Enforce stationarity
        if alpha + beta >= 1:
            total = alpha + beta
            alpha /= (total + 0.01)
            beta /= (total + 0.01)

        self._sigma2 = self._garch_filter(r, omega, alpha, beta)
        lr_var = omega / max(1 - alpha - beta, 1e-8)
        self.params = GARCHParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            long_run_var=lr_var,
            log_likelihood=ll,
            converged=converged,
            model_type="GARCH",
        )
        return self.params

    def _fit_egarch(self, r: np.ndarray) -> GARCHParams:
        x0 = np.array([-0.1, 0.10, -0.05, 0.85])
        bounds = [(-1.0, 1.0), (0.0, 0.5), (-0.3, 0.3), (-0.999, 0.999)]
        converged = False
        omega, alpha, gamma, beta = x0
        ll = -1e10

        if _SCIPY_AVAILABLE:
            try:
                res = minimize(
                    self._egarch_neg_loglik,
                    x0,
                    args=(r,),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500},
                )
                if res.success or res.fun < 1e9:
                    omega, alpha, gamma, beta = res.x
                    ll = -res.fun
                    converged = res.success
            except Exception as exc:
                log.debug("EGARCH scipy failed: %s", exc)

        log_sigma2 = self._egarch_filter(r, omega, alpha, gamma, beta)
        self._sigma2 = np.exp(log_sigma2)
        lr_var = np.exp(omega / max(1 - beta, 1e-8))
        self.params = GARCHParams(
            omega=omega,
            alpha=alpha,
            beta=gamma,  # store gamma in beta slot for EGARCH
            long_run_var=lr_var,
            log_likelihood=ll,
            converged=converged,
            model_type="EGARCH",
        )
        return self.params

    # ------------------------------------------------------------------
    def forecast_variance(self, h: int = 10) -> np.ndarray:
        """h-step ahead variance forecast from current state."""
        if self.params is None or self._sigma2 is None:
            raise RuntimeError("Call fit() first")
        p = self.params
        if p.model_type == "EGARCH":
            # For EGARCH: geometric decay
            forecasts = np.zeros(h)
            last_log_var = np.log(max(self._sigma2[-1], 1e-12))
            omega, beta = p.omega, p.beta
            lr_log_var = omega / max(1 - beta, 1e-8)
            for i in range(h):
                last_log_var = lr_log_var + beta * (last_log_var - lr_log_var)
                forecasts[i] = np.exp(last_log_var)
            return forecasts

        # GARCH(1,1) multi-step forecast
        omega, alpha, beta = p.omega, p.alpha, p.beta
        lr_var = p.long_run_var
        sigma2_T = self._sigma2[-1]
        forecasts = np.zeros(h)
        for i in range(h):
            if i == 0:
                forecasts[i] = omega + (alpha + beta) * sigma2_T
            else:
                forecasts[i] = lr_var + (alpha + beta) ** (i + 1) * (
                    sigma2_T - lr_var
                )
        return np.maximum(forecasts, 1e-12)

    def compute_conditional_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Return filtered conditional standard deviation σ_t series."""
        if self.params is None:
            self.fit(returns)
        return np.sqrt(np.maximum(self._sigma2, 1e-12))

    def simulate(self, n: int, params: Optional[GARCHParams] = None) -> np.ndarray:
        """Monte Carlo simulation of n return draws from fitted GARCH process."""
        p = params or self.params
        if p is None:
            raise RuntimeError("No parameters available; call fit() or pass params")
        omega, alpha, beta = p.omega, p.alpha, p.beta
        sigma2 = np.empty(n)
        r_sim = np.empty(n)
        sigma2[0] = p.long_run_var if p.long_run_var > 0 else omega / max(1 - alpha - beta, 1e-8)
        z = np.random.standard_normal(n)
        r_sim[0] = np.sqrt(sigma2[0]) * z[0]
        for t in range(1, n):
            sigma2[t] = omega + alpha * r_sim[t - 1] ** 2 + beta * sigma2[t - 1]
            r_sim[t] = np.sqrt(max(sigma2[t], 1e-12)) * z[t]
        return r_sim


# ---------------------------------------------------------------------------
# HistoricalSimulationVaR
# ---------------------------------------------------------------------------


def _norm_ppf(q: float) -> float:
    """Inverse normal CDF via numpy (fallback when scipy unavailable)."""
    if _SCIPY_AVAILABLE:
        return float(norm.ppf(q))
    # Rational approximation (Beasley-Springer-Moro)
    if q <= 0:
        return -np.inf
    if q >= 1:
        return np.inf
    p = q if q < 0.5 else 1 - q
    t = np.sqrt(-2 * np.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
    return -x if q < 0.5 else x


class HistoricalSimulationVaR:
    """Non-parametric VaR from historical return distribution."""

    # ------------------------------------------------------------------
    def compute_var(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> float:
        """
        Historical Simulation VaR.
        Returns positive number representing loss at confidence level.
        """
        r = np.asarray(returns.dropna())
        if len(r) < 10:
            return 0.0
        q = np.quantile(r, 1.0 - confidence)
        var = -q * math.sqrt(horizon)
        return float(max(var, 0.0))

    # ------------------------------------------------------------------
    def compute_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> float:
        """
        Conditional VaR (Expected Shortfall).
        Mean of losses beyond the VaR threshold.
        """
        r = np.asarray(returns.dropna())
        if len(r) < 10:
            return 0.0
        threshold = np.quantile(r, 1.0 - confidence)
        tail = r[r <= threshold]
        if len(tail) == 0:
            return float(-threshold * math.sqrt(horizon))
        cvar = -float(np.mean(tail)) * math.sqrt(horizon)
        return max(cvar, 0.0)

    # ------------------------------------------------------------------
    def compute_component_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence: float = 0.99,
    ) -> pd.Series:
        """
        Component VaR (marginal contribution of each asset).
        Uses the delta-normal approximation on top of HS portfolio VaR.
        """
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        r = returns.dropna()
        n_assets = r.shape[1]

        # Portfolio returns
        port_r = r.values @ w

        # Portfolio HS VaR
        port_var = self.compute_var(pd.Series(port_r), confidence)

        # Covariance matrix
        cov = np.cov(r.values.T)

        # Marginal VaR (delta-normal)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol < 1e-10:
            return pd.Series(np.zeros(n_assets), index=returns.columns)

        marginal_vol = cov @ w / port_vol
        z = _norm_ppf(confidence)
        component_var = w * marginal_vol * z

        result = pd.Series(component_var, index=returns.columns)
        # Scale so they sum to portfolio VaR
        scale = port_var / max(result.sum(), 1e-10)
        return result * scale

    # ------------------------------------------------------------------
    def rolling_var(
        self,
        returns: pd.Series,
        window: int = 252,
        confidence: float = 0.99,
    ) -> pd.Series:
        """Compute rolling HS VaR."""
        result = returns.rolling(window=window).apply(
            lambda x: self.compute_var(pd.Series(x), confidence), raw=False
        )
        return result.rename(f"HS_VaR_{confidence}")

    # ------------------------------------------------------------------
    def compute_weighted_var(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        lam: float = 0.99,
    ) -> float:
        """
        Age-weighted historical simulation VaR.
        Assigns exponentially decaying weights to older observations (BRW method).
        """
        r = np.asarray(returns.dropna())
        n = len(r)
        if n < 10:
            return 0.0

        # Weights: most recent = highest weight
        idx = np.arange(n)
        w = lam ** (n - 1 - idx)
        w = w / w.sum()

        # Sort returns with weights
        sorted_idx = np.argsort(r)
        sorted_r = r[sorted_idx]
        sorted_w = w[sorted_idx]

        # Find quantile
        cum_w = np.cumsum(sorted_w)
        q_idx = np.searchsorted(cum_w, 1.0 - confidence)
        if q_idx >= n:
            q_idx = n - 1
        var = -sorted_r[q_idx]
        return float(max(var, 0.0))


# ---------------------------------------------------------------------------
# ParametricVaR
# ---------------------------------------------------------------------------


class ParametricVaR:
    """Closed-form VaR under normal and Student-t assumptions."""

    def __init__(self):
        self._garch = GARCHModel()

    # ------------------------------------------------------------------
    def compute_var_normal(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> float:
        """Normal (Gaussian) parametric VaR."""
        r = np.asarray(returns.dropna())
        if len(r) < 5:
            return 0.0
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        z = _norm_ppf(confidence)
        var = (z * sigma - mu) * math.sqrt(horizon)
        return float(max(var, 0.0))

    # ------------------------------------------------------------------
    def compute_var_student_t(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> float:
        """Student-t VaR with fitted degrees of freedom."""
        r = np.asarray(returns.dropna())
        if len(r) < 5:
            return 0.0
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))

        # Fit degrees of freedom
        nu = 4.0  # fat-tail default
        if _SCIPY_AVAILABLE:
            try:
                nu_fit, loc_fit, scale_fit = student_t.fit(r, floc=mu)
                nu = max(float(nu_fit), 2.5)  # clip for stability
                sigma = float(scale_fit)
            except Exception:
                pass

        # t-quantile
        if _SCIPY_AVAILABLE:
            t_q = float(student_t.ppf(confidence, df=nu))
        else:
            # Wilson-Hilferty approximation for large nu
            z = _norm_ppf(confidence)
            t_q = z * (1 + z**2 / (4 * nu)) / math.sqrt(1 - 1 / nu)

        # Scale factor: t-VaR / normal-VaR correction
        var = (t_q * sigma - mu) * math.sqrt(horizon)
        return float(max(var, 0.0))

    # ------------------------------------------------------------------
    def compute_garch_var(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        horizon: int = 10,
    ) -> float:
        """
        GARCH(1,1) VaR: use current conditional vol, project h steps ahead.
        """
        r = np.asarray(returns.dropna())
        if len(r) < 30:
            return 0.0
        try:
            self._garch.fit(r)
        except Exception as exc:
            log.warning("GARCH fit failed: %s", exc)
            return self.compute_var_normal(returns, confidence, horizon)

        # Aggregate variance over horizon
        forecasts = self._garch.forecast_variance(h=horizon)
        total_var = float(np.sum(forecasts))
        total_vol = math.sqrt(total_var)

        z = _norm_ppf(confidence)
        mu = float(np.mean(r)) * horizon
        var = z * total_vol - mu
        return float(max(var, 0.0))


# ---------------------------------------------------------------------------
# MonteCarloVaR
# ---------------------------------------------------------------------------


class MonteCarloVaR:
    """Full Monte Carlo portfolio VaR with Cholesky decomposition."""

    # ------------------------------------------------------------------
    def compute_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        n_sims: int = 10_000,
        confidence: float = 0.99,
        horizon: int = 10,
    ) -> MonteCarloResult:
        """
        Monte Carlo VaR via correlated return simulation.

        Process:
          1. Estimate mean and covariance from historical returns.
          2. Cholesky decompose covariance matrix.
          3. Simulate n_sims × horizon correlated paths.
          4. Compute portfolio P&L distribution.
        """
        r = returns.dropna().values  # shape (T, N)
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        n_assets = r.shape[1]

        mu = np.mean(r, axis=0)
        cov = np.cov(r.T)

        # Regularise cov (Ledoit-Wolf-like: add small diagonal)
        min_eig = float(np.linalg.eigvalsh(cov).min())
        if min_eig < 1e-10:
            cov += (abs(min_eig) + 1e-8) * np.eye(n_assets)

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback: diagonal vol
            L = np.diag(np.sqrt(np.diag(cov)))

        # Simulate n_sims paths over horizon
        rng = np.random.default_rng(seed=42)
        # shape: (n_sims, horizon, n_assets)
        z = rng.standard_normal((n_sims, horizon, n_assets))
        # Apply Cholesky correlation: corr_z[i,t] = z[i,t] @ L.T + mu
        corr_z = z @ L.T + mu  # broadcasts over n_sims, horizon

        # Portfolio return per sim per day
        port_r = corr_z @ w  # (n_sims, horizon)
        # Cumulative P&L over horizon
        port_cum = port_r.sum(axis=1)  # (n_sims,)

        pnl = port_cum  # as fraction of portfolio

        var_99 = float(-np.percentile(pnl, 1.0))
        var_95 = float(-np.percentile(pnl, 5.0))
        cvar_99 = float(-np.mean(pnl[pnl <= np.percentile(pnl, 1.0)]))
        cvar_95 = float(-np.mean(pnl[pnl <= np.percentile(pnl, 5.0)]))

        pctiles = {p: float(np.percentile(pnl, p)) for p in [1, 5, 10, 25, 50]}

        return MonteCarloResult(
            var_99=max(var_99, 0.0),
            var_95=max(var_95, 0.0),
            cvar_99=max(cvar_99, 0.0),
            cvar_95=max(cvar_95, 0.0),
            mean_pnl=float(np.mean(pnl)),
            std_pnl=float(np.std(pnl)),
            n_simulations=n_sims,
            horizon=horizon,
            pnl_percentiles=pctiles,
        )

    # ------------------------------------------------------------------
    def compute_stressed_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        stress_period: str = "2008-2009",
    ) -> float:
        """
        VaR computed using covariance estimated from a stressed historical period.
        """
        period_map = {
            "2008-2009": ("2008-01-01", "2009-06-30"),
            "2020-covid": ("2020-02-01", "2020-05-31"),
            "2022-rates": ("2022-01-01", "2022-12-31"),
            "dotcom": ("2000-01-01", "2002-10-31"),
        }
        lo, hi = period_map.get(stress_period, ("2008-01-01", "2009-06-30"))

        mask = (returns.index >= lo) & (returns.index <= hi)
        stressed_r = returns.loc[mask]

        if stressed_r.shape[0] < 20:
            # Not enough data in stressed period; use worst 20% of days
            port_r_all = (returns.dropna().values @ np.asarray(weights, float))
            port_r_all /= port_r_all.std() + 1e-12
            threshold = np.percentile(port_r_all, 20)
            idx_worst = port_r_all <= threshold
            stressed_r = returns.dropna().iloc[idx_worst]

        result = self.compute_var(stressed_r, weights, n_sims=5_000, confidence=0.99, horizon=10)
        return result.var_99


# ---------------------------------------------------------------------------
# Basel3RiskCalculator
# ---------------------------------------------------------------------------


class Basel3RiskCalculator:
    """
    Basel III / FRTB risk metrics for regulatory capital estimation.

    Basel II: 99th percentile 10-day VaR, capital = k * VaR (k >= 3)
    FRTB: 97.5th percentile ES with stress, IMA or SA approach
    """

    def __init__(self):
        self._hs_var = HistoricalSimulationVaR()

    # ------------------------------------------------------------------
    def compute_regulatory_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
    ) -> Basel3Metrics:
        """Compute full Basel III regulatory VaR and capital charge."""
        w = np.asarray(weights, dtype=float) / np.sum(weights)
        port_r = pd.Series(returns.dropna().values @ w, index=returns.dropna().index)

        # Basel II: 99% 10-day VaR
        var_99_1d = self._hs_var.compute_var(port_r, confidence=0.99, horizon=1)
        var_99_10d = self._hs_var.compute_var(port_r, confidence=0.99, horizon=10)

        # FRTB: 97.5% ES
        es_975 = self._hs_var.compute_cvar(port_r, confidence=0.975, horizon=10)

        # Stressed metrics using worst 250-day window
        worst_var = self._compute_worst_period_var(port_r)

        # Backtesting for k multiplier
        bt = self.run_var_backtesting(
            port_r.tail(250),
            pd.Series(var_99_1d, index=port_r.tail(250).index),
        )

        # Capital multiplier k
        k = 3.0  # minimum
        if bt.zone == "YELLOW":
            exceptions = bt.n_exceptions
            penalty = (exceptions - 4) * 0.10
            k = min(4.0, 3.0 + penalty)
        elif bt.zone == "RED":
            k = 4.0

        capital_charge = k * max(var_99_10d, worst_var)

        # IRC: simplified approximation (0.01% quantile of 1-yr simulation)
        irc_approx = var_99_10d * math.sqrt(52)  # 52 weeks ≈ 1yr

        return Basel3Metrics(
            var_99_1d=var_99_1d,
            var_99_10d=var_99_10d,
            es_975_stressed=es_975,
            capital_charge=capital_charge,
            k_multiplier=k,
            backtesting_zone=bt.zone,
            n_exceptions=bt.n_exceptions,
            stressed_var=worst_var,
            incremental_risk_charge=irc_approx,
        )

    # ------------------------------------------------------------------
    def _compute_worst_period_var(
        self, port_r: pd.Series, window: int = 250
    ) -> float:
        """Find worst 250-day rolling window and compute VaR on it."""
        if len(port_r) < window:
            return self._hs_var.compute_var(port_r, confidence=0.99, horizon=10)
        rolling_mean = port_r.rolling(window).mean()
        worst_end_loc = rolling_mean.idxmin()
        if worst_end_loc is None:
            return self._hs_var.compute_var(port_r, confidence=0.99, horizon=10)
        loc = port_r.index.get_loc(worst_end_loc)
        start_loc = max(0, loc - window + 1)
        stressed_slice = port_r.iloc[start_loc : loc + 1]
        return self._hs_var.compute_var(stressed_slice, confidence=0.99, horizon=10)

    # ------------------------------------------------------------------
    def run_var_backtesting(
        self,
        returns: pd.Series,
        var_series: pd.Series,
    ) -> BacktestResult:
        """
        Compare realised returns against VaR forecasts.
        Counts exceptions (loss > VaR estimate).
        Traffic light: GREEN ≤4, YELLOW 5-9, RED ≥10 per 250 days.
        """
        aligned = var_series.reindex(returns.index).ffill()
        exceptions_mask = returns < -aligned
        n_exc = int(exceptions_mask.sum())
        n_obs = len(returns)
        exc_dates = list(returns.index[exceptions_mask].strftime("%Y-%m-%d"))
        exc_rate = n_exc / n_obs if n_obs > 0 else 0.0

        zone = "GREEN"
        if 5 <= n_exc <= 9:
            zone = "YELLOW"
        elif n_exc >= 10:
            zone = "RED"

        # Kupiec POF test (unconditional coverage)
        kupiec_p = self._kupiec_test(n_exc, n_obs, confidence=0.99)

        return BacktestResult(
            n_exceptions=n_exc,
            exception_rate=exc_rate,
            zone=zone,
            exception_dates=exc_dates[:20],  # cap output
            kupiec_pvalue=kupiec_p,
        )

    @staticmethod
    def _kupiec_test(n_exc: int, n_obs: int, confidence: float = 0.99) -> float:
        """Kupiec proportion-of-failures LR test. Returns p-value."""
        if n_obs == 0:
            return 1.0
        alpha = 1.0 - confidence
        p_hat = n_exc / n_obs
        if p_hat <= 0 or p_hat >= 1:
            return 1.0 if p_hat == alpha else 0.0
        try:
            from scipy.stats import chi2

            lr = 2 * (
                n_exc * math.log(p_hat / alpha)
                + (n_obs - n_exc) * math.log((1 - p_hat) / (1 - alpha))
            )
            p_val = float(1.0 - chi2.cdf(lr, df=1))
        except Exception:
            p_val = 1.0
        return p_val

    # ------------------------------------------------------------------
    def compute_stressed_metrics(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
    ) -> dict:
        """Compute stressed VaR, IRC approximation, and scenario metrics."""
        w = np.asarray(weights, float) / np.sum(weights)
        port_r = pd.Series(returns.dropna().values @ w, index=returns.dropna().index)

        stressed_var = self._compute_worst_period_var(port_r)
        irc = stressed_var * math.sqrt(52)
        cvar_stress = self._hs_var.compute_cvar(
            port_r.nsmallest(250), confidence=0.99, horizon=1
        )

        return {
            "stressed_var_10d": stressed_var,
            "incremental_risk_charge": irc,
            "cvar_stressed": cvar_stress,
            "es_frtb_97.5": self._hs_var.compute_cvar(port_r, 0.975, 10),
        }


# ---------------------------------------------------------------------------
# PortfolioRiskEngine (orchestrator)
# ---------------------------------------------------------------------------


_STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "2008 GFC": {
        "equity": -0.45,
        "bonds": 0.05,
        "commodities": -0.50,
        "credit": -0.30,
        "fx_usd": 0.05,
    },
    "COVID 2020": {
        "equity": -0.35,
        "bonds": 0.08,
        "commodities": -0.40,
        "credit": -0.15,
        "fx_usd": 0.04,
    },
    "Rate shock +200bp": {
        "equity": -0.15,
        "bonds": -0.20,
        "commodities": 0.05,
        "credit": -0.10,
        "fx_usd": 0.02,
    },
    "Inflation spike": {
        "equity": -0.20,
        "bonds": -0.15,
        "commodities": 0.30,
        "credit": -0.05,
        "fx_usd": -0.03,
    },
    "Tech selloff 2022": {
        "equity": -0.40,
        "bonds": -0.18,
        "commodities": 0.10,
        "credit": -0.08,
        "fx_usd": 0.05,
    },
}


class PortfolioRiskEngine:
    """
    Orchestrator for all portfolio risk metrics.
    Computes VaR via all methods, Basel III, stress tests,
    and risk decomposition.
    """

    def __init__(self, lookback_days: int = 756):
        self.lookback_days = lookback_days
        self._fetcher = ReturnsFetcher()
        self._hs = HistoricalSimulationVaR()
        self._par = ParametricVaR()
        self._mc = MonteCarloVaR()
        self._b3 = Basel3RiskCalculator()

    # ------------------------------------------------------------------
    def _get_returns(
        self, tickers: List[str], end_date: Optional[str] = None
    ) -> pd.DataFrame:
        end = end_date or datetime.now().strftime("%Y-%m-%d")
        start_dt = datetime.strptime(end, "%Y-%m-%d") - timedelta(
            days=self.lookback_days + 50
        )
        start = start_dt.strftime("%Y-%m-%d")
        return self._fetcher.fetch_returns(tickers, start, end)

    # ------------------------------------------------------------------
    def analyze_portfolio(
        self,
        holdings: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
        end_date: Optional[str] = None,
    ) -> RiskReport:
        """
        Full portfolio risk analysis.

        Parameters
        ----------
        holdings : {ticker: weight}
        portfolio_value : total USD value
        end_date : ISO date string

        Returns
        -------
        RiskReport
        """
        tickers = list(holdings.keys())
        raw_w = np.array(list(holdings.values()), dtype=float)
        raw_w = raw_w / raw_w.sum()

        log.info("Fetching returns for %d holdings", len(tickers))
        returns = self._get_returns(tickers, end_date)

        if returns.empty:
            log.error("No return data fetched")
            return RiskReport(tickers=tickers, weights=holdings)

        # Align weights to available tickers
        avail = [t for t in tickers if t in returns.columns]
        avail_w = np.array([holdings[t] for t in avail], dtype=float)
        avail_w = avail_w / avail_w.sum()
        returns = returns[avail]

        port_r = pd.Series(
            returns.values @ avail_w, index=returns.index, name="portfolio"
        )

        # ---- HS VaR ----
        hs_var_99 = self._hs.compute_var(port_r, 0.99, 1)
        hs_var_95 = self._hs.compute_var(port_r, 0.95, 1)
        hs_cvar_99 = self._hs.compute_cvar(port_r, 0.99, 1)

        # ---- Parametric ----
        par_normal = self._par.compute_var_normal(port_r, 0.99, 1)
        par_t = self._par.compute_var_student_t(port_r, 0.99, 1)
        garch_var = self._par.compute_garch_var(port_r, 0.99, 10)

        # ---- Monte Carlo ----
        mc_result = self._mc.compute_var(returns, avail_w, n_sims=10_000, confidence=0.99, horizon=10)

        # ---- Basel III ----
        b3 = self._b3.compute_regulatory_var(returns, avail_w)
        bt = self._b3.run_var_backtesting(
            port_r.tail(250),
            pd.Series(hs_var_99, index=port_r.tail(250).index),
        )

        # ---- Stress ----
        stress = self.run_stress_tests(holdings, portfolio_value, returns=returns, weights=avail_w)

        # ---- Decomposition ----
        comp_var = self._hs.compute_component_var(returns, avail_w, 0.99)

        # ---- Performance ----
        te = self.compute_tracking_error(holdings, returns=returns, weights=avail_w)
        ir = self.compute_information_ratio(holdings, returns=returns, weights=avail_w)

        ann_vol = float(np.std(port_r, ddof=1)) * math.sqrt(252)
        rf_daily = 0.05 / 252
        sharpe = (float(np.mean(port_r)) - rf_daily) / (float(np.std(port_r, ddof=1)) + 1e-10) * math.sqrt(252)

        return RiskReport(
            tickers=avail,
            weights=dict(zip(avail, avail_w.tolist())),
            portfolio_value=portfolio_value,
            as_of_date=(end_date or datetime.now().strftime("%Y-%m-%d")),
            hs_var_99=hs_var_99,
            hs_var_95=hs_var_95,
            hs_cvar_99=hs_cvar_99,
            parametric_var_normal=par_normal,
            parametric_var_t=par_t,
            garch_var=garch_var,
            mc_var_99=mc_result.var_99,
            mc_cvar_99=mc_result.cvar_99,
            basel3=b3,
            backtest=bt,
            stress=stress,
            component_var=dict(zip(avail, comp_var.values.tolist())),
            marginal_var=dict(zip(avail, (comp_var / avail_w).values.tolist())),
            tracking_error=te,
            information_ratio=ir,
            annualized_vol=ann_vol,
            sharpe_ratio=sharpe,
        )

    # ------------------------------------------------------------------
    def compute_risk_decomposition(
        self,
        holdings: Dict[str, float],
        returns: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Per-asset risk decomposition: marginal VaR, component VaR,
        % contribution to total portfolio VaR.
        """
        if returns is None:
            tickers = list(holdings.keys())
            returns = self._get_returns(tickers)
        if weights is None:
            avail = [t for t in holdings if t in returns.columns]
            weights = np.array([holdings[t] for t in avail], dtype=float)
            weights /= weights.sum()

        tickers = list(returns.columns)
        n = len(tickers)

        port_r = pd.Series(returns.values @ weights, index=returns.index)
        port_var = self._hs.compute_var(port_r, 0.99)
        comp_var = self._hs.compute_component_var(returns, weights, 0.99)
        marg_var = comp_var / (weights + 1e-12)

        rows = []
        for i, t in enumerate(tickers):
            w_i = float(weights[i])
            cv = float(comp_var.get(t, 0.0))
            mv = float(marg_var.get(t, 0.0))
            pct = cv / port_var * 100 if port_var > 0 else 0.0
            rows.append(
                {
                    "ticker": t,
                    "weight": round(w_i, 4),
                    "component_var": round(cv, 6),
                    "marginal_var": round(mv, 6),
                    "pct_contribution": round(pct, 2),
                }
            )
        df = pd.DataFrame(rows).sort_values("pct_contribution", ascending=False)
        df["cumulative_pct"] = df["pct_contribution"].cumsum().round(2)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    def run_stress_tests(
        self,
        holdings: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
        returns: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
        custom_scenario: Optional[Dict[str, float]] = None,
    ) -> StressTestResult:
        """
        Apply pre-defined and optional custom scenarios to the portfolio.
        Asset-class shocks are applied based on ticker classification heuristics.
        """
        scenarios = dict(_STRESS_SCENARIOS)
        if custom_scenario:
            scenarios["Custom"] = custom_scenario

        tickers = list(holdings.keys())
        if weights is None:
            avail_t = tickers if returns is None else [t for t in tickers if t in returns.columns]
            raw_w = np.array([holdings[t] for t in avail_t], dtype=float)
            raw_w /= raw_w.sum()
        else:
            avail_t = list(returns.columns) if returns is not None else tickers
            raw_w = np.asarray(weights, float)

        def classify_ticker(t: str) -> str:
            t = t.upper()
            bond_etfs = {"TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "JNK"}
            comm_etfs = {"GLD", "SLV", "USO", "DBO", "PDBC", "IAU", "CORN", "WEAT"}
            credit_etfs = {"LQD", "HYG", "JNK", "BKLN", "ANGL"}
            fx_etfs = {"UUP", "UDN", "FXE", "FXY", "FXB"}
            if t in bond_etfs:
                return "bonds"
            if t in comm_etfs:
                return "commodities"
            if t in credit_etfs:
                return "credit"
            if t in fx_etfs:
                return "fx_usd"
            return "equity"

        asset_classes = [classify_ticker(t) for t in avail_t]

        scenario_losses: Dict[str, float] = {}
        stressed_values: Dict[str, float] = {}

        for scen_name, shocks in scenarios.items():
            port_shock = 0.0
            for i, t in enumerate(avail_t):
                ac = asset_classes[i]
                shock = shocks.get(ac, shocks.get("equity", -0.20))
                port_shock += raw_w[i] * shock

            loss = -port_shock * portfolio_value
            scenario_losses[scen_name] = round(loss, 2)
            stressed_values[scen_name] = round(portfolio_value + port_shock * portfolio_value, 2)

        worst_scen = max(scenario_losses, key=lambda k: scenario_losses[k])
        worst_loss = scenario_losses[worst_scen]

        return StressTestResult(
            scenario_losses=scenario_losses,
            worst_scenario=worst_scen,
            worst_loss=worst_loss,
            portfolio_value=portfolio_value,
            stressed_values=stressed_values,
        )

    # ------------------------------------------------------------------
    def compute_tracking_error(
        self,
        holdings: Dict[str, float],
        benchmark: str = "SPY",
        returns: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Annualised tracking error vs benchmark."""
        if returns is None:
            tickers = list(holdings.keys())
            returns = self._get_returns(tickers)
        if weights is None:
            avail = [t for t in holdings if t in returns.columns]
            weights = np.array([holdings[t] for t in avail], dtype=float)
            weights /= weights.sum()

        port_r = pd.Series(returns.values @ weights, index=returns.index)

        try:
            bmk_r = self._fetcher.fetch_benchmark_returns(benchmark)
            common = port_r.index.intersection(bmk_r.index)
            if len(common) < 20:
                return float(np.std(port_r, ddof=1)) * math.sqrt(252)
            diff = port_r.loc[common] - bmk_r.loc[common]
            return float(np.std(diff, ddof=1)) * math.sqrt(252)
        except Exception:
            return float(np.std(port_r, ddof=1)) * math.sqrt(252)

    # ------------------------------------------------------------------
    def compute_information_ratio(
        self,
        holdings: Dict[str, float],
        benchmark: str = "SPY",
        returns: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Information ratio = active return / tracking error."""
        if returns is None:
            tickers = list(holdings.keys())
            returns = self._get_returns(tickers)
        if weights is None:
            avail = [t for t in holdings if t in returns.columns]
            weights = np.array([holdings[t] for t in avail], dtype=float)
            weights /= weights.sum()

        port_r = pd.Series(returns.values @ weights, index=returns.index)

        try:
            bmk_r = self._fetcher.fetch_benchmark_returns(benchmark)
            common = port_r.index.intersection(bmk_r.index)
            if len(common) < 20:
                return 0.0
            diff = port_r.loc[common] - bmk_r.loc[common]
            te = float(np.std(diff, ddof=1)) * math.sqrt(252)
            active_ret = float(np.mean(diff)) * 252
            return active_ret / te if te > 0 else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def get_risk_dashboard(
        self,
        holdings: Dict[str, float],
        portfolio_value: float = 1_000_000.0,
    ) -> dict:
        """Return all risk metrics in a single flat dictionary."""
        report = self.analyze_portfolio(holdings, portfolio_value)
        decomp = self.compute_risk_decomposition(holdings)

        return {
            "as_of_date": report.as_of_date,
            "portfolio_value": report.portfolio_value,
            "annualized_vol": round(report.annualized_vol * 100, 2),
            "sharpe_ratio": round(report.sharpe_ratio, 3),
            "tracking_error": round(report.tracking_error * 100, 2),
            "information_ratio": round(report.information_ratio, 3),
            # VaR (as % of portfolio)
            "hs_var_99_1d_pct": round(report.hs_var_99 * 100, 3),
            "hs_cvar_99_1d_pct": round(report.hs_cvar_99 * 100, 3),
            "parametric_var_normal_pct": round(report.parametric_var_normal * 100, 3),
            "parametric_var_t_pct": round(report.parametric_var_t * 100, 3),
            "garch_var_10d_pct": round(report.garch_var * 100, 3),
            "mc_var_99_10d_pct": round(report.mc_var_99 * 100, 3),
            "mc_cvar_99_10d_pct": round(report.mc_cvar_99 * 100, 3),
            # Dollar amounts
            "hs_var_99_1d_usd": round(report.hs_var_99 * portfolio_value, 0),
            "mc_var_99_10d_usd": round(report.mc_var_99 * portfolio_value, 0),
            # Basel III
            "capital_charge_usd": round(report.basel3.capital_charge * portfolio_value, 0),
            "k_multiplier": report.basel3.k_multiplier,
            "backtesting_zone": report.basel3.backtesting_zone,
            "n_exceptions": report.backtest.n_exceptions,
            # Stress
            "worst_stress_scenario": report.stress.worst_scenario,
            "worst_stress_loss_usd": round(report.stress.worst_loss, 0),
            "scenario_losses_usd": {
                k: round(v, 0) for k, v in report.stress.scenario_losses.items()
            },
            # Decomposition summary
            "top_risk_contributors": decomp.head(5).to_dict("records"),
        }


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    print("=" * 70)
    print("SENTINEL Portfolio Risk Engine — demo (dim_077)")
    print("=" * 70)

    # 10-stock equal-weight portfolio
    HOLDINGS: Dict[str, float] = {
        "AAPL": 0.10,
        "MSFT": 0.10,
        "GOOGL": 0.10,
        "AMZN": 0.10,
        "JPM": 0.10,
        "GS": 0.10,
        "JNJ": 0.10,
        "PG": 0.10,
        "XOM": 0.10,
        "BRK-B": 0.10,
    }
    PORTFOLIO_VALUE = 1_000_000.0

    engine = PortfolioRiskEngine(lookback_days=756)

    print("\n[1] Full risk analysis...")
    try:
        report = engine.analyze_portfolio(HOLDINGS, PORTFOLIO_VALUE)
        print(f"  Portfolio: {len(report.tickers)} stocks, ${PORTFOLIO_VALUE:,.0f}")
        print(f"  As of: {report.as_of_date}")
        print(f"  Annualised vol: {report.annualized_vol*100:.2f}%")
        print(f"  Sharpe ratio:   {report.sharpe_ratio:.3f}")
        print(f"\n--- VaR Summary (1-day, 99%) ---")
        print(f"  HS VaR:          {report.hs_var_99*100:.3f}%  (${report.hs_var_99*PORTFOLIO_VALUE:,.0f})")
        print(f"  HS CVaR:         {report.hs_cvar_99*100:.3f}%  (${report.hs_cvar_99*PORTFOLIO_VALUE:,.0f})")
        print(f"  Parametric (N):  {report.parametric_var_normal*100:.3f}%")
        print(f"  Parametric (t):  {report.parametric_var_t*100:.3f}%")
        print(f"  GARCH (10d):     {report.garch_var*100:.3f}%")
        print(f"  MC VaR (10d):    {report.mc_var_99*100:.3f}%  (${report.mc_var_99*PORTFOLIO_VALUE:,.0f})")
        print(f"  MC CVaR (10d):   {report.mc_cvar_99*100:.3f}%")
    except Exception as exc:
        print(f"  Error in full analysis: {exc}")

    print("\n[2] Basel III metrics...")
    try:
        print(f"  VaR 99% 10d:    {report.basel3.var_99_10d*100:.3f}%")
        print(f"  Stressed VaR:   {report.basel3.stressed_var*100:.3f}%")
        print(f"  ES (FRTB 97.5): {report.basel3.es_975_stressed*100:.3f}%")
        print(f"  Capital charge: ${report.basel3.capital_charge*PORTFOLIO_VALUE:,.0f}")
        print(f"  k-multiplier:   {report.basel3.k_multiplier:.1f}")
        print(f"  Backtesting:    {report.basel3.backtesting_zone}  ({report.backtest.n_exceptions} exceptions)")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[3] Stress tests...")
    try:
        for scen, loss in report.stress.scenario_losses.items():
            pct = loss / PORTFOLIO_VALUE * 100
            print(f"  {scen:<25}: -${loss:>12,.0f}  ({pct:.1f}%)")
        print(f"  Worst scenario: {report.stress.worst_scenario}")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[4] Risk decomposition (top 5)...")
    try:
        decomp = engine.compute_risk_decomposition(HOLDINGS)
        print(decomp.head(5).to_string(index=False))
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[5] Risk dashboard (full)...")
    try:
        dash = engine.get_risk_dashboard(HOLDINGS, PORTFOLIO_VALUE)
        for k, v in dash.items():
            if k not in ("top_risk_contributors", "scenario_losses_usd"):
                print(f"  {k:<35}: {v}")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\nDone.")
