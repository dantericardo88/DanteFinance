"""
Correlation Monitor V2 — dim_080 (target score 9).

Comprehensive correlation monitoring system with DCC-GARCH, HMM regime
detection, copula tail dependence, PCA factor analysis, Minimum Spanning
Tree topology, cross-asset correlation, and diversification ratio tracking.

Free data sources:
  yfinance — OHLCV price history for equity universe and benchmarks

Key capabilities:
  - Rolling correlation matrices: 21d, 63d, 126d, 252d windows (100+ tickers)
  - DCC-GARCH(1,1): Dynamic Conditional Correlation via scipy optimization
  - HMM-based regime detection: HIGH_CORR / NORMAL / DECORR / CRISIS (4 states)
  - Clayton / Gumbel copula tail dependence (lower/upper lambda)
  - PCA: explained variance, PC loadings heatmap data
  - Minimum Spanning Tree via Kruskal — market hubs and peripheral stocks
  - Alert system: correlation shock >0.3 vs 3-month baseline
  - Cross-asset correlation: equities vs TLT, GLD, USO, ^VIX
  - Regime transition matrix: Markov probabilities between states
  - Diversification ratio tracker
  - SQLite persistence: correlation snapshots, alert log, regime history
  - FastAPI router: /correlation-matrix, /regime-status, /mst, /alert-config

Usage:
    from sentinel.sbx.correlation_monitor_v2 import router
    app.include_router(router, prefix="/api/v2/corr")
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import scipy.optimize as opt
import scipy.stats as stats
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "data" / "correlation_v2.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Rolling windows in trading days
WINDOWS = {"21d": 21, "63d": 63, "126d": 126, "252d": 252}

# Cross-asset benchmarks always included
CROSS_ASSET = {
    "TLT": "Long Treasuries",
    "GLD": "Gold",
    "USO": "Oil",
    "^VIX": "VIX Index",
    "SHY": "Short Treasuries",
    "HYG": "High Yield Credit",
    "EEM": "Emerging Markets",
    "DX-Y.NYB": "USD Index",
}

# Default equity universe (100+ tickers from S&P sectors)
EQUITY_UNIVERSE: List[str] = [
    # Large Cap Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "ORCL", "CRM",
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "ADI", "KLAC", "MRVL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "USB", "PNC",
    "AXP", "COF", "BX", "KKR", "APO",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "MDT", "BMY",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB",
    # Consumer
    "AMZN", "HD", "MCD", "SBUX", "NKE", "TGT", "COST", "WMT", "PG", "KO",
    "PEP", "PM", "MO", "CL", "EL",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "OXY", "DVN",
    # Industrials
    "GE", "RTX", "HON", "LMT", "BA", "CAT", "DE", "UPS", "FDX", "MMM",
    # Communication
    "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR", "TMUS",
    # Real Estate
    "PLD", "AMT", "EQIX", "CCI", "SPG", "O",
    # Utilities
    "NEE", "DUK", "SO", "AEP", "EXC",
    # ETF benchmarks
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU",
]
# Deduplicate while preserving order
_seen: set = set()
EQUITY_UNIVERSE = [x for x in EQUITY_UNIVERSE if not (x in _seen or _seen.add(x))]

# Regime labels
REGIME_HIGH_CORR = "HIGH_CORR"
REGIME_NORMAL = "NORMAL"
REGIME_DECORR = "DECORR"
REGIME_CRISIS = "CRISIS"
REGIMES = [REGIME_DECORR, REGIME_NORMAL, REGIME_HIGH_CORR, REGIME_CRISIS]

# Alert thresholds
CORR_SHOCK_THRESHOLD = 0.30   # spike vs 3-month average triggers alert
DIV_RATIO_WARN = 0.75          # diversification ratio warning level

# Cache TTL seconds
CACHE_TTL_PRICES = 3600       # 1 hour
CACHE_TTL_CORR = 21600        # 6 hours


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS price_cache (
            ticker TEXT NOT NULL,
            as_of  TEXT NOT NULL,
            close  REAL NOT NULL,
            fetched_at INTEGER NOT NULL,
            PRIMARY KEY (ticker, as_of)
        );

        CREATE TABLE IF NOT EXISTS corr_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_dt TEXT NOT NULL,
            window_days INTEGER NOT NULL,
            ticker_a    TEXT NOT NULL,
            ticker_b    TEXT NOT NULL,
            correlation REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_snap_dt ON corr_snapshots(snapshot_dt, window_days);

        CREATE TABLE IF NOT EXISTS regime_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of       TEXT NOT NULL,
            regime      TEXT NOT NULL,
            avg_corr    REAL,
            vix_level   REAL,
            confidence  REAL,
            extra_json  TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_reg_dt ON regime_history(as_of);

        CREATE TABLE IF NOT EXISTS alert_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            alerted_at  TEXT NOT NULL,
            ticker_a    TEXT NOT NULL,
            ticker_b    TEXT NOT NULL,
            corr_current REAL,
            corr_baseline REAL,
            shock_delta REAL,
            alert_type  TEXT,
            severity    TEXT,
            message     TEXT
        );

        CREATE TABLE IF NOT EXISTS alert_config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mst_cache (
            cached_at TEXT NOT NULL,
            window_days INTEGER NOT NULL,
            mst_json TEXT NOT NULL,
            PRIMARY KEY (cached_at, window_days)
        );

        INSERT OR IGNORE INTO alert_config VALUES ('shock_threshold', '0.30');
        INSERT OR IGNORE INTO alert_config VALUES ('div_ratio_warn',  '0.75');
        INSERT OR IGNORE INTO alert_config VALUES ('email_alerts',    'false');
        """)


_init_db()


@contextmanager
def _db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _get_config(key: str, default: str = "") -> str:
    with _db() as conn:
        row = conn.execute("SELECT value FROM alert_config WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def _set_config(key: str, value: str) -> None:
    with _db() as conn:
        conn.execute("INSERT OR REPLACE INTO alert_config VALUES (?,?)", (key, value))


# ---------------------------------------------------------------------------
# Price fetching & caching
# ---------------------------------------------------------------------------

def _fetch_prices(
    tickers: List[str],
    lookback_days: int = 300,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch adjusted close prices via yfinance, with SQLite cache."""
    end = end or date.today()
    start = end - timedelta(days=lookback_days + 30)

    cached: Dict[str, pd.Series] = {}
    missing: List[str] = []
    cutoff_ts = int(time.time()) - CACHE_TTL_PRICES

    with _db() as conn:
        for tk in tickers:
            rows = conn.execute(
                "SELECT as_of, close FROM price_cache WHERE ticker=? AND as_of>=? AND as_of<=? AND fetched_at>?",
                (tk, str(start), str(end), cutoff_ts),
            ).fetchall()
            if rows:
                s = pd.Series({r["as_of"]: r["close"] for r in rows}, name=tk)
                s.index = pd.to_datetime(s.index)
                cached[tk] = s
            else:
                missing.append(tk)

    frames: Dict[str, pd.Series] = dict(cached)

    if missing:
        try:
            raw = yf.download(
                missing,
                start=str(start),
                end=str(end + timedelta(days=1)),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                logger.warning("yfinance returned empty frame for: %s", missing)
            else:
                if isinstance(raw.columns, pd.MultiIndex):
                    closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0:len(missing)]
                else:
                    closes = raw[["Close"]] if "Close" in raw.columns else raw
                    closes.columns = missing

                for tk in missing:
                    if tk in closes.columns:
                        s = closes[tk].dropna()
                        frames[tk] = s
                        # persist to cache
                        now_ts = int(time.time())
                        rows_to_ins = [(tk, str(d.date()), float(v), now_ts) for d, v in s.items()]
                        with _db() as conn:
                            conn.executemany(
                                "INSERT OR REPLACE INTO price_cache VALUES (?,?,?,?)",
                                rows_to_ins,
                            )
        except Exception as exc:
            logger.error("yfinance download failed: %s", exc)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[df.index.date <= end]  # type: ignore[comparison-overlap]
    return df


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from price dataframe."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


# ---------------------------------------------------------------------------
# Rolling Correlation Matrices
# ---------------------------------------------------------------------------

def _rolling_corr_matrix(
    returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage correlation matrix for the last `window` rows."""
    from sklearn.covariance import LedoitWolf
    data = returns.tail(window).dropna(axis=1, how="any")
    if data.shape[0] < max(10, window // 4):
        return pd.DataFrame()
    lw = LedoitWolf().fit(data.values)
    cov = lw.covariance_
    std = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=data.columns, columns=data.columns)


def compute_all_rolling_matrices(
    tickers: List[str],
    lookback_days: int = 290,
    end: Optional[date] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute 21d/63d/126d/252d correlation matrices. Returns dict keyed by window label."""
    prices = _fetch_prices(tickers, lookback_days=lookback_days, end=end)
    if prices.empty:
        return {}
    returns = _compute_returns(prices)
    result: Dict[str, pd.DataFrame] = {}
    for label, w in WINDOWS.items():
        mat = _rolling_corr_matrix(returns, w)
        if not mat.empty:
            result[label] = mat
    return result


# ---------------------------------------------------------------------------
# DCC-GARCH(1,1)
# ---------------------------------------------------------------------------

def _garch11_fit(r: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit GARCH(1,1) via numerical MLE.  Returns (omega, alpha, beta).
    Constraints: omega>0, alpha>=0, beta>=0, alpha+beta<1.
    """
    r = r - r.mean()
    n = len(r)

    def neg_loglik(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        h = np.full(n, r.var())
        ll = 0.0
        for t in range(1, n):
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
            if h[t] <= 0:
                return 1e10
            ll += -0.5 * (np.log(2 * np.pi) + np.log(h[t]) + r[t] ** 2 / h[t])
        return -ll

    var0 = r.var()
    x0 = np.array([var0 * 0.05, 0.10, 0.85])
    bounds = [(1e-8, None), (0, 0.5), (0, 0.999)]
    res = opt.minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-8})
    if res.success:
        return tuple(res.x)  # type: ignore[return-value]
    return (var0 * 0.05, 0.10, 0.85)


def _garch11_variance(r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    n = len(r)
    h = np.full(n, r.var())
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        h[t] = max(h[t], 1e-12)
    return h


def fit_dcc_garch(
    returns: pd.DataFrame,
    max_assets: int = 20,
) -> Dict[str, Any]:
    """
    DCC-GARCH(1,1) implementation.

    Steps:
      1. Fit univariate GARCH(1,1) per asset → standardised residuals z_t
      2. Estimate DCC correlation Q_t via EM-style optimization of (a, b)
      3. Return time-varying correlation matrix at last observation

    Returns dict with keys:
      'tickers', 'dcc_corr_matrix', 'a', 'b', 'garch_params'
    """
    cols = [c for c in returns.columns if returns[c].notna().sum() > 60][:max_assets]
    r = returns[cols].dropna().values  # T x N
    T, N = r.shape
    if T < 50 or N < 2:
        return {}

    # Step 1: Univariate GARCH per asset
    garch_params: List[Tuple] = []
    std_resid = np.zeros_like(r)
    for i in range(N):
        ri = r[:, i]
        om, al, be = _garch11_fit(ri)
        h = _garch11_variance(ri, om, al, be)
        std_resid[:, i] = ri / np.sqrt(np.maximum(h, 1e-12))
        garch_params.append((om, al, be))

    # Unconditional correlation Q_bar
    Q_bar = np.corrcoef(std_resid.T)

    # Step 2: DCC parameter estimation
    def dcc_neg_loglik(params: np.ndarray) -> float:
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10
        Q = Q_bar.copy()
        ll = 0.0
        for t in range(1, T):
            z = std_resid[t - 1]
            Q = (1 - a - b) * Q_bar + a * np.outer(z, z) + b * Q
            diag_sqrt = np.sqrt(np.diag(Q))
            diag_sqrt = np.maximum(diag_sqrt, 1e-12)
            R = Q / np.outer(diag_sqrt, diag_sqrt)
            np.fill_diagonal(R, 1.0)
            try:
                sign, logdet = np.linalg.slogdet(R)
                if sign <= 0:
                    return 1e10
                z_t = std_resid[t]
                Rinv = np.linalg.inv(R)
                ll += -0.5 * (logdet + z_t @ Rinv @ z_t - z_t @ z_t)
            except np.linalg.LinAlgError:
                return 1e10
        return -ll

    res = opt.minimize(
        dcc_neg_loglik,
        x0=np.array([0.05, 0.90]),
        method="L-BFGS-B",
        bounds=[(1e-4, 0.3), (0.5, 0.9999)],
        options={"maxiter": 100},
    )
    a, b = (res.x[0], res.x[1]) if res.success else (0.05, 0.90)

    # Compute final Q_T (time-varying at last step)
    Q = Q_bar.copy()
    for t in range(1, T):
        z = std_resid[t - 1]
        Q = (1 - a - b) * Q_bar + a * np.outer(z, z) + b * Q

    diag_sqrt = np.sqrt(np.maximum(np.diag(Q), 1e-12))
    R_T = Q / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(R_T, 1.0)
    R_T = np.clip(R_T, -1.0, 1.0)

    return {
        "tickers": cols,
        "dcc_corr_matrix": R_T.tolist(),
        "a": round(float(a), 6),
        "b": round(float(b), 6),
        "garch_params": [{"ticker": cols[i], "omega": gp[0], "alpha": gp[1], "beta": gp[2]}
                         for i, gp in enumerate(garch_params)],
    }


# ---------------------------------------------------------------------------
# HMM Regime Detection
# ---------------------------------------------------------------------------

class SimpleHMM:
    """
    Baum-Welch trained HMM with K Gaussian emission states.
    Operates on 1-D observation sequence (e.g., rolling avg correlation).
    """

    def __init__(self, n_states: int = 4, n_iter: int = 50, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        # Model params (will be set after fit)
        self.pi: np.ndarray = np.ones(n_states) / n_states
        self.A: np.ndarray = np.full((n_states, n_states), 1 / n_states)
        self.mu: np.ndarray = np.zeros(n_states)
        self.sigma: np.ndarray = np.ones(n_states)
        self.fitted = False

    def _emission(self, obs: float, k: int) -> float:
        return float(stats.norm.pdf(obs, self.mu[k], max(self.sigma[k], 1e-8)))

    def fit(self, obs: np.ndarray) -> "SimpleHMM":
        T = len(obs)
        K = self.n_states
        # k-means init
        sorted_obs = np.sort(obs)
        breaks = np.linspace(0, len(sorted_obs), K + 1, dtype=int)
        self.mu = np.array([sorted_obs[breaks[k]:breaks[k + 1]].mean() for k in range(K)])
        self.sigma = np.full(K, obs.std() / K + 0.01)
        self.A = np.full((K, K), 1 / K)
        self.pi = np.ones(K) / K

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            # Forward pass
            alpha = np.zeros((T, K))
            for k in range(K):
                alpha[0, k] = self.pi[k] * self._emission(obs[0], k)
            alpha[0] /= alpha[0].sum() + 1e-300
            scaling = np.zeros(T)
            scaling[0] = 1.0
            for t in range(1, T):
                for k in range(K):
                    alpha[t, k] = sum(alpha[t - 1, j] * self.A[j, k] for j in range(K)) * self._emission(obs[t], k)
                s = alpha[t].sum() + 1e-300
                scaling[t] = s
                alpha[t] /= s

            # Backward pass
            beta = np.zeros((T, K))
            beta[T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                for j in range(K):
                    beta[t, j] = sum(self.A[j, k] * self._emission(obs[t + 1], k) * beta[t + 1, k] for k in range(K))
                beta[t] /= beta[t].sum() + 1e-300

            # Gamma / Xi
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                for j in range(K):
                    for k in range(K):
                        xi[t, j, k] = alpha[t, j] * self.A[j, k] * self._emission(obs[t + 1], k) * beta[t + 1, k]
                xi[t] /= xi[t].sum() + 1e-300

            # M-step
            self.pi = gamma[0]
            for j in range(K):
                denom = xi[:, j, :].sum() + 1e-300
                for k in range(K):
                    self.A[j, k] = xi[:, j, k].sum() / denom
            for k in range(K):
                gk = gamma[:, k]
                gsum = gk.sum() + 1e-300
                self.mu[k] = (gk * obs).sum() / gsum
                self.sigma[k] = np.sqrt((gk * (obs - self.mu[k]) ** 2).sum() / gsum) + 1e-6

            ll = np.log(scaling + 1e-300).sum()
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.fitted = True
        return self

    def decode(self, obs: np.ndarray) -> np.ndarray:
        """Viterbi decoding → sequence of state indices."""
        T = len(obs)
        K = self.n_states
        viterbi = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        for k in range(K):
            viterbi[0, k] = np.log(self.pi[k] + 1e-300) + np.log(self._emission(obs[0], k) + 1e-300)
        for t in range(1, T):
            for k in range(K):
                trans = viterbi[t - 1] + np.log(self.A[:, k] + 1e-300)
                psi[t, k] = np.argmax(trans)
                viterbi[t, k] = trans[psi[t, k]] + np.log(self._emission(obs[t], k) + 1e-300)
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(viterbi[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def transition_matrix(self) -> np.ndarray:
        """Return transition probability matrix A."""
        return self.A.copy()


def _map_states_to_regimes(mu: np.ndarray) -> Dict[int, str]:
    """Map HMM state indices to regime labels by sorting mean correlation."""
    order = np.argsort(mu)  # ascending avg_corr
    mapping: Dict[int, str] = {}
    labels = [REGIME_DECORR, REGIME_NORMAL, REGIME_HIGH_CORR, REGIME_CRISIS]
    for rank, state in enumerate(order):
        mapping[int(state)] = labels[min(rank, len(labels) - 1)]
    return mapping


def detect_correlation_regime(
    corr_series: pd.Series,
    n_states: int = 4,
) -> Dict[str, Any]:
    """
    Run HMM on rolling average-pairwise-correlation time series.

    Returns:
        current_regime, state_sequence, transition_matrix, state_means,
        regime_map, confidence
    """
    obs = corr_series.dropna().values
    if len(obs) < 20:
        return {"current_regime": REGIME_NORMAL, "confidence": 0.5}

    hmm = SimpleHMM(n_states=n_states, n_iter=60)
    hmm.fit(obs)
    states = hmm.decode(obs)
    regime_map = _map_states_to_regimes(hmm.mu)
    current_state = int(states[-1])
    current_regime = regime_map[current_state]

    # Confidence: fraction of last 5 periods in same state
    recent = states[-5:] if len(states) >= 5 else states
    confidence = float(np.mean(recent == current_state))

    # Transition matrix as dict
    A = hmm.transition_matrix()
    trans = {}
    for j in range(n_states):
        from_r = regime_map[j]
        trans[from_r] = {regime_map[k]: round(float(A[j, k]), 4) for k in range(n_states)}

    return {
        "current_regime": current_regime,
        "confidence": round(confidence, 4),
        "state_means": {regime_map[k]: round(float(hmm.mu[k]), 4) for k in range(n_states)},
        "transition_matrix": trans,
        "state_sequence": [regime_map[int(s)] for s in states[-30:]],  # last 30 obs
    }


# ---------------------------------------------------------------------------
# Copula Tail Dependence
# ---------------------------------------------------------------------------

def _empirical_cdf(x: np.ndarray) -> np.ndarray:
    """Rank-based empirical CDF (probability integral transform)."""
    n = len(x)
    ranks = stats.rankdata(x)
    return ranks / (n + 1)


def _clayton_lower_tail(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """Lower tail dependence coefficient for Clayton copula."""
    # lambda_L = 2^(-1/theta)
    if theta <= 0:
        return 0.0
    return float(2.0 ** (-1.0 / theta))


def _gumbel_upper_tail(theta: float) -> float:
    """Upper tail dependence coefficient for Gumbel copula."""
    # lambda_U = 2 - 2^(1/theta)
    if theta <= 1:
        return 0.0
    return float(2.0 - 2.0 ** (1.0 / theta))


def _fit_clayton_theta(u: np.ndarray, v: np.ndarray) -> float:
    """MLE for Clayton copula parameter (method of moments via Kendall's tau)."""
    tau = float(stats.kendalltau(u, v).statistic)
    tau = max(tau, 1e-6)
    # theta = 2*tau / (1 - tau)
    return 2.0 * tau / max(1.0 - tau, 1e-6)


def _fit_gumbel_theta(u: np.ndarray, v: np.ndarray) -> float:
    """MLE for Gumbel copula parameter via Kendall's tau."""
    tau = float(stats.kendalltau(u, v).statistic)
    tau = max(tau, 0.0)
    # theta = 1 / (1 - tau)
    return 1.0 / max(1.0 - tau, 1e-6)


def compute_tail_dependence(
    returns: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute Clayton (lower) and Gumbel (upper) tail dependence for ticker pairs.

    Returns list of dicts with keys:
        ticker_a, ticker_b, lower_tail_lambda, upper_tail_lambda,
        kendall_tau, clayton_theta, gumbel_theta
    """
    cols = returns.columns.tolist()
    if pairs is None:
        # All pairs up to 20 tickers
        cols = cols[:20]
        pairs = list(combinations(cols, 2))

    results = []
    for a, b in pairs:
        if a not in returns.columns or b not in returns.columns:
            continue
        sub = returns[[a, b]].dropna()
        if len(sub) < 30:
            continue
        ra, rb = sub[a].values, sub[b].values
        u = _empirical_cdf(ra)
        v = _empirical_cdf(rb)
        cl_theta = _fit_clayton_theta(u, v)
        gu_theta = _fit_gumbel_theta(u, v)
        tau_val = float(stats.kendalltau(ra, rb).statistic)
        results.append({
            "ticker_a": a,
            "ticker_b": b,
            "lower_tail_lambda": round(_clayton_lower_tail(u, v, cl_theta), 4),
            "upper_tail_lambda": round(_gumbel_upper_tail(gu_theta), 4),
            "kendall_tau": round(tau_val, 4),
            "clayton_theta": round(cl_theta, 4),
            "gumbel_theta": round(gu_theta, 4),
        })
    return results


# ---------------------------------------------------------------------------
# PCA Factor Analysis
# ---------------------------------------------------------------------------

def compute_pca(
    returns: pd.DataFrame,
    window: int = 63,
    n_components: int = 10,
) -> Dict[str, Any]:
    """
    Principal Component Analysis on rolling window of returns.

    Returns:
        explained_variance_ratio (list), cumulative_variance (list),
        pc_loadings (dict: PC -> {ticker: loading}),
        top_contributors (dict: PC -> [(ticker, loading)])
    """
    data = returns.tail(window).dropna(axis=1, how="any")
    if data.shape[0] < 10 or data.shape[1] < 2:
        return {}

    X = data.values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total = eigvals.sum()
    n_comp = min(n_components, len(eigvals))
    evr = (eigvals[:n_comp] / total).tolist()
    cum_var = np.cumsum(evr).tolist()

    tickers = data.columns.tolist()
    loadings: Dict[str, Dict[str, float]] = {}
    top_contrib: Dict[str, List] = {}
    for pc in range(n_comp):
        pc_label = f"PC{pc + 1}"
        vec = eigvecs[:, pc]
        loadings[pc_label] = {tickers[i]: round(float(vec[i]), 4) for i in range(len(tickers))}
        sorted_contrib = sorted(loadings[pc_label].items(), key=lambda x: abs(x[1]), reverse=True)
        top_contrib[pc_label] = sorted_contrib[:5]

    return {
        "n_assets": len(tickers),
        "window_days": window,
        "n_components": n_comp,
        "explained_variance_ratio": [round(x, 4) for x in evr],
        "cumulative_variance": [round(x, 4) for x in cum_var],
        "pc_loadings": loadings,
        "top_contributors": top_contrib,
        "tickers": tickers,
    }


# ---------------------------------------------------------------------------
# Minimum Spanning Tree (Kruskal)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def compute_mst(
    corr_matrix: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build Minimum Spanning Tree from correlation matrix.

    Uses distance metric d(i,j) = sqrt(2*(1 - rho_ij)).
    Kruskal's algorithm on sorted edges.

    Returns:
        nodes (list), edges (list of {source, target, distance, correlation}),
        hub_tickers (most connected nodes), peripheral_tickers (least connected)
    """
    tickers = corr_matrix.columns.tolist()
    n = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    # Build edge list
    edges = []
    for i, j in combinations(range(n), 2):
        rho = float(corr_matrix.iloc[i, j])
        rho = max(-1.0, min(1.0, rho))
        dist = math.sqrt(2.0 * max(0.0, 1.0 - rho))
        edges.append((dist, i, j, rho))
    edges.sort(key=lambda e: e[0])

    # Kruskal
    uf = UnionFind(n)
    mst_edges = []
    degree: Dict[int, int] = {i: 0 for i in range(n)}
    for dist, i, j, rho in edges:
        if uf.union(i, j):
            mst_edges.append({
                "source": tickers[i],
                "target": tickers[j],
                "distance": round(dist, 4),
                "correlation": round(rho, 4),
            })
            degree[i] += 1
            degree[j] += 1
        if len(mst_edges) == n - 1:
            break

    # Hubs = high degree, peripherals = degree 1
    deg_sorted = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    hub_tickers = [tickers[i] for i, d in deg_sorted[:5] if d > 1]
    peripheral_tickers = [tickers[i] for i, d in deg_sorted if d == 1][-10:]

    return {
        "n_nodes": n,
        "n_edges": len(mst_edges),
        "nodes": [{"ticker": t, "degree": degree[ticker_idx[t]]} for t in tickers],
        "edges": mst_edges,
        "hub_tickers": hub_tickers,
        "peripheral_tickers": peripheral_tickers,
    }


# ---------------------------------------------------------------------------
# Correlation Shock Alert System
# ---------------------------------------------------------------------------

def _get_rolling_avg_corr(
    corr_matrix: pd.DataFrame,
) -> float:
    """Average off-diagonal upper-triangle correlation."""
    vals = []
    n = len(corr_matrix)
    arr = corr_matrix.values
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(arr[i, j])
    return float(np.mean(vals)) if vals else 0.0


def _baseline_corr_from_db(
    ticker_a: str,
    ticker_b: str,
    window_days: int = 63,
    lookback_days: int = 90,
) -> Optional[float]:
    """Average correlation for pair over past lookback_days from snapshots."""
    cutoff = str((date.today() - timedelta(days=lookback_days)).isoformat())
    with _db() as conn:
        rows = conn.execute(
            """SELECT correlation FROM corr_snapshots
               WHERE window_days=? AND snapshot_dt>=?
               AND ((ticker_a=? AND ticker_b=?) OR (ticker_a=? AND ticker_b=?))
            """,
            (window_days, cutoff, ticker_a, ticker_b, ticker_b, ticker_a),
        ).fetchall()
    if not rows:
        return None
    return float(np.mean([r["correlation"] for r in rows]))


def persist_corr_snapshot(
    corr_matrix: pd.DataFrame,
    window_days: int,
    snapshot_dt: Optional[str] = None,
) -> None:
    """Save correlation matrix pairs to SQLite."""
    dt = snapshot_dt or date.today().isoformat()
    tickers = corr_matrix.columns.tolist()
    rows = []
    for i, j in combinations(range(len(tickers)), 2):
        rows.append((dt, window_days, tickers[i], tickers[j], float(corr_matrix.iloc[i, j])))
    with _db() as conn:
        conn.executemany(
            "INSERT INTO corr_snapshots (snapshot_dt, window_days, ticker_a, ticker_b, correlation) VALUES (?,?,?,?,?)",
            rows,
        )


def detect_correlation_shocks(
    corr_matrix: pd.DataFrame,
    window_days: int = 63,
) -> List[Dict[str, Any]]:
    """
    Detect pairs where current correlation deviates >CORR_SHOCK_THRESHOLD
    from 3-month baseline stored in SQLite.

    Also detects portfolio-wide diversification loss.
    """
    shock_threshold = float(_get_config("shock_threshold", str(CORR_SHOCK_THRESHOLD)))
    alerts = []
    tickers = corr_matrix.columns.tolist()
    now = datetime.utcnow().isoformat()

    for i, j in combinations(range(len(tickers)), 2):
        a, b = tickers[i], tickers[j]
        current = float(corr_matrix.iloc[i, j])
        baseline = _baseline_corr_from_db(a, b, window_days)
        if baseline is None:
            continue
        delta = current - baseline
        if abs(delta) >= shock_threshold:
            alert_type = "correlation_spike" if delta > 0 else "correlation_collapse"
            severity = "high" if abs(delta) >= 0.5 else "medium" if abs(delta) >= 0.35 else "low"
            msg = (
                f"{a}-{b}: current={current:.3f}, baseline={baseline:.3f}, "
                f"delta={delta:+.3f} [{alert_type}]"
            )
            alerts.append({
                "alerted_at": now,
                "ticker_a": a,
                "ticker_b": b,
                "corr_current": round(current, 4),
                "corr_baseline": round(baseline, 4),
                "shock_delta": round(delta, 4),
                "alert_type": alert_type,
                "severity": severity,
                "message": msg,
            })

    # Persist to DB
    if alerts:
        with _db() as conn:
            conn.executemany(
                """INSERT INTO alert_log
                   (alerted_at, ticker_a, ticker_b, corr_current, corr_baseline,
                    shock_delta, alert_type, severity, message)
                   VALUES (:alerted_at,:ticker_a,:ticker_b,:corr_current,:corr_baseline,
                           :shock_delta,:alert_type,:severity,:message)""",
                alerts,
            )
    return alerts


# ---------------------------------------------------------------------------
# Cross-Asset Correlation
# ---------------------------------------------------------------------------

def compute_cross_asset_correlation(
    equity_tickers: List[str],
    lookback_days: int = 63,
    end: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Correlate each equity ticker against TLT, GLD, USO, ^VIX.
    Returns matrix dict: {equity: {asset: corr}}.
    """
    all_tickers = list(set(equity_tickers + list(CROSS_ASSET.keys())))
    prices = _fetch_prices(all_tickers, lookback_days=lookback_days + 30, end=end)
    if prices.empty:
        return {}
    returns = _compute_returns(prices).tail(lookback_days)

    cross_assets = [t for t in CROSS_ASSET.keys() if t in returns.columns]
    equities = [t for t in equity_tickers if t in returns.columns]

    result: Dict[str, Dict[str, float]] = {}
    for eq in equities:
        result[eq] = {}
        for ca in cross_assets:
            sub = returns[[eq, ca]].dropna()
            if len(sub) < 10:
                result[eq][ca] = float("nan")
            else:
                rho = float(np.corrcoef(sub[eq].values, sub[ca].values)[0, 1])
                result[eq][ca] = round(rho, 4)

    # Summary stats
    summary = {}
    for ca in cross_assets:
        vals = [result[eq].get(ca, float("nan")) for eq in equities]
        vals = [v for v in vals if not math.isnan(v)]
        summary[ca] = {
            "mean": round(float(np.mean(vals)), 4) if vals else float("nan"),
            "median": round(float(np.median(vals)), 4) if vals else float("nan"),
            "std": round(float(np.std(vals)), 4) if vals else float("nan"),
            "asset_name": CROSS_ASSET.get(ca, ca),
        }

    return {
        "equities": equities,
        "cross_assets": cross_assets,
        "correlation_matrix": result,
        "summary": summary,
        "window_days": lookback_days,
        "as_of": str(end or date.today()),
    }


# ---------------------------------------------------------------------------
# Diversification Ratio
# ---------------------------------------------------------------------------

def compute_diversification_ratio(
    weights: Dict[str, float],
    corr_matrix: pd.DataFrame,
    volatilities: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Diversification Ratio = sum(w_i * sigma_i) / sigma_portfolio.

    A ratio >1 indicates beneficial diversification.
    Weighted average correlation is also computed.
    """
    tickers = [t for t in weights if t in corr_matrix.columns]
    if not tickers:
        return {"error": "No tickers found in correlation matrix"}

    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()  # normalize

    C = corr_matrix.loc[tickers, tickers].values

    # If no volatilities provided, use equal vol
    if volatilities is None:
        sigma = np.ones(len(tickers))
    else:
        sigma = np.array([volatilities.get(t, 1.0) for t in tickers])

    # Portfolio variance
    portfolio_var = w @ (C * np.outer(sigma, sigma)) @ w
    portfolio_std = math.sqrt(max(portfolio_var, 1e-12))

    # Weighted average vol
    weighted_vol = float(w @ sigma)

    # Diversification ratio
    div_ratio = weighted_vol / portfolio_std if portfolio_std > 0 else 1.0

    # Weighted average pairwise correlation
    n = len(tickers)
    weighted_avg_corr = 0.0
    if n > 1:
        pairs_sum = 0.0
        weight_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                wi_wj = w[i] * w[j]
                pairs_sum += wi_wj * C[i, j]
                weight_sum += wi_wj
        weighted_avg_corr = pairs_sum / weight_sum if weight_sum > 0 else 0.0

    warn_threshold = float(_get_config("div_ratio_warn", str(DIV_RATIO_WARN)))
    warning = weighted_avg_corr > warn_threshold

    return {
        "tickers": tickers,
        "weights": {t: round(float(w[i]), 4) for i, t in enumerate(tickers)},
        "diversification_ratio": round(float(div_ratio), 4),
        "weighted_avg_correlation": round(float(weighted_avg_corr), 4),
        "portfolio_vol": round(float(portfolio_std), 6),
        "diversification_warning": warning,
        "warning_threshold": warn_threshold,
    }


# ---------------------------------------------------------------------------
# Regime History Persistence
# ---------------------------------------------------------------------------

def _persist_regime(
    regime: str,
    avg_corr: float,
    confidence: float,
    vix_level: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> None:
    with _db() as conn:
        conn.execute(
            """INSERT INTO regime_history (as_of, regime, avg_corr, vix_level, confidence, extra_json)
               VALUES (?,?,?,?,?,?)""",
            (date.today().isoformat(), regime, avg_corr, vix_level, confidence,
             json.dumps(extra or {})),
        )


def get_regime_history(days: int = 90) -> List[Dict]:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with _db() as conn:
        rows = conn.execute(
            "SELECT as_of, regime, avg_corr, vix_level, confidence FROM regime_history WHERE as_of>=? ORDER BY as_of",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Master analysis entry point
# ---------------------------------------------------------------------------

def full_correlation_analysis(
    tickers: Optional[List[str]] = None,
    portfolio_weights: Optional[Dict[str, float]] = None,
    end: Optional[date] = None,
    run_dcc: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate all analyses:
      - Rolling correlation matrices (4 windows)
      - DCC-GARCH dynamic correlation
      - HMM regime detection
      - Copula tail dependence
      - PCA factor analysis
      - MST topology
      - Cross-asset correlation
      - Diversification ratio (if weights provided)
      - Shock alerts
    """
    tickers = tickers or EQUITY_UNIVERSE[:30]  # default subset for speed
    end = end or date.today()

    # Fetch prices once
    prices = _fetch_prices(tickers + list(CROSS_ASSET.keys()), lookback_days=290, end=end)
    if prices.empty:
        return {"error": "No price data available"}

    equity_prices = prices[[t for t in tickers if t in prices.columns]]
    returns = _compute_returns(equity_prices)

    result: Dict[str, Any] = {
        "as_of": str(end),
        "tickers": tickers,
        "n_tickers": len(tickers),
    }

    # --- Rolling matrices ---
    rolling_mats: Dict[str, Any] = {}
    for label, w in WINDOWS.items():
        mat = _rolling_corr_matrix(returns, w)
        if not mat.empty:
            rolling_mats[label] = {
                "matrix": mat.values.tolist(),
                "tickers": mat.columns.tolist(),
                "avg_pairwise_corr": round(_get_rolling_avg_corr(mat), 4),
            }
            if label == "63d":
                # Persist snapshot
                try:
                    persist_corr_snapshot(mat, 63)
                except Exception:
                    pass
    result["rolling_matrices"] = rolling_mats

    # --- Rolling avg corr series for HMM ---
    if "63d" in rolling_mats:
        avg_corr_series_vals = []
        # Compute rolling 63d avg pairwise corr over time
        step = max(1, len(returns) - 252)
        avg_corr_series_dates = []
        for start_i in range(step, len(returns)):
            sub = returns.iloc[max(0, start_i - 63):start_i].dropna(axis=1, how="any")
            if sub.shape[0] >= 10 and sub.shape[1] >= 2:
                c = sub.corr().values
                n = c.shape[0]
                upper = [c[i, j] for i in range(n) for j in range(i + 1, n)]
                avg_corr_series_vals.append(np.mean(upper))
                avg_corr_series_dates.append(str(returns.index[start_i - 1].date()))

        if len(avg_corr_series_vals) >= 20:
            regime_info = detect_correlation_regime(
                pd.Series(avg_corr_series_vals, index=avg_corr_series_dates)
            )
            result["regime"] = regime_info
            # Persist
            try:
                _persist_regime(
                    regime=regime_info["current_regime"],
                    avg_corr=avg_corr_series_vals[-1],
                    confidence=regime_info.get("confidence", 0.5),
                )
            except Exception:
                pass
        else:
            result["regime"] = {"current_regime": REGIME_NORMAL, "confidence": 0.5}

    # --- DCC-GARCH ---
    if run_dcc:
        try:
            dcc = fit_dcc_garch(returns, max_assets=20)
            result["dcc_garch"] = dcc
        except Exception as exc:
            result["dcc_garch"] = {"error": str(exc)}

    # --- Copula tail dependence ---
    try:
        corr_tickers = tickers[:15]
        pairs = list(combinations([t for t in corr_tickers if t in returns.columns], 2))[:30]
        tail_dep = compute_tail_dependence(returns, pairs=pairs)
        result["tail_dependence"] = tail_dep
    except Exception as exc:
        result["tail_dependence"] = {"error": str(exc)}

    # --- PCA ---
    try:
        pca_result = compute_pca(returns, window=63, n_components=10)
        result["pca"] = pca_result
    except Exception as exc:
        result["pca"] = {"error": str(exc)}

    # --- MST ---
    try:
        if "63d" in rolling_mats:
            mat_63 = pd.DataFrame(
                rolling_mats["63d"]["matrix"],
                index=rolling_mats["63d"]["tickers"],
                columns=rolling_mats["63d"]["tickers"],
            )
            mst = compute_mst(mat_63)
            result["mst"] = mst
        else:
            result["mst"] = {}
    except Exception as exc:
        result["mst"] = {"error": str(exc)}

    # --- Cross-asset ---
    try:
        cross = compute_cross_asset_correlation(tickers[:20], lookback_days=63, end=end)
        result["cross_asset"] = cross
    except Exception as exc:
        result["cross_asset"] = {"error": str(exc)}

    # --- Diversification ratio ---
    if portfolio_weights and "63d" in rolling_mats:
        try:
            mat_63 = pd.DataFrame(
                rolling_mats["63d"]["matrix"],
                index=rolling_mats["63d"]["tickers"],
                columns=rolling_mats["63d"]["tickers"],
            )
            dr = compute_diversification_ratio(portfolio_weights, mat_63)
            result["diversification"] = dr
        except Exception as exc:
            result["diversification"] = {"error": str(exc)}

    # --- Shock alerts ---
    try:
        if "63d" in rolling_mats:
            mat_63 = pd.DataFrame(
                rolling_mats["63d"]["matrix"],
                index=rolling_mats["63d"]["tickers"],
                columns=rolling_mats["63d"]["tickers"],
            )
            shocks = detect_correlation_shocks(mat_63, window_days=63)
            result["shock_alerts"] = shocks
    except Exception as exc:
        result["shock_alerts"] = {"error": str(exc)}

    return result


# ---------------------------------------------------------------------------
# Pydantic models for FastAPI
# ---------------------------------------------------------------------------

class AlertConfigRequest(BaseModel):
    shock_threshold: Optional[float] = Field(None, ge=0.0, le=1.0,
                                              description="Correlation shock threshold (default 0.30)")
    div_ratio_warn: Optional[float] = Field(None, ge=0.0, le=1.0,
                                            description="Diversification ratio warning level")
    email_alerts: Optional[bool] = Field(None, description="Enable email alerts")


class AlertConfigResponse(BaseModel):
    shock_threshold: float
    div_ratio_warn: float
    email_alerts: bool
    updated: bool = False


class CorrelationMatrixResponse(BaseModel):
    as_of: str
    window: str
    tickers: List[str]
    matrix: List[List[float]]
    avg_pairwise_corr: float
    n_tickers: int


class RegimeStatusResponse(BaseModel):
    as_of: str
    current_regime: str
    confidence: float
    state_means: Optional[Dict[str, float]] = None
    transition_matrix: Optional[Dict[str, Dict[str, float]]] = None
    recent_states: Optional[List[str]] = None
    history: Optional[List[Dict]] = None


class MSTResponse(BaseModel):
    as_of: str
    n_nodes: int
    n_edges: int
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    hub_tickers: List[str]
    peripheral_tickers: List[str]


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/correlation", tags=["Correlation Monitor V2"])


@router.get("/matrix", response_model=CorrelationMatrixResponse, summary="Get rolling correlation matrix")
async def get_correlation_matrix(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers; default = equity universe top 30"),
    window: str = Query("63d", description="Window: 21d | 63d | 126d | 252d"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD; default today"),
) -> CorrelationMatrixResponse:
    """Return Ledoit-Wolf shrinkage correlation matrix for given window."""
    if window not in WINDOWS:
        raise HTTPException(400, f"window must be one of {list(WINDOWS.keys())}")

    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else EQUITY_UNIVERSE[:30]
    end = date.fromisoformat(end_date) if end_date else date.today()

    prices = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_prices(tk_list, lookback_days=WINDOWS[window] + 30, end=end)
    )
    if prices.empty:
        raise HTTPException(404, "No price data found for requested tickers")

    returns = _compute_returns(prices)
    mat = _rolling_corr_matrix(returns, WINDOWS[window])
    if mat.empty:
        raise HTTPException(500, "Insufficient data to compute correlation matrix")

    return CorrelationMatrixResponse(
        as_of=str(end),
        window=window,
        tickers=mat.columns.tolist(),
        matrix=[[round(v, 4) for v in row] for row in mat.values.tolist()],
        avg_pairwise_corr=round(_get_rolling_avg_corr(mat), 4),
        n_tickers=len(mat),
    )


@router.get("/all-windows", summary="Get all 4 rolling correlation matrices")
async def get_all_windows(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
) -> Dict[str, Any]:
    """Return 21d, 63d, 126d, 252d correlation matrices in one call."""
    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else EQUITY_UNIVERSE[:30]
    end = date.fromisoformat(end_date) if end_date else date.today()

    mats = await asyncio.get_event_loop().run_in_executor(
        None, lambda: compute_all_rolling_matrices(tk_list, end=end)
    )
    return {
        label: {
            "tickers": mat.columns.tolist(),
            "matrix": [[round(v, 4) for v in row] for row in mat.values.tolist()],
            "avg_pairwise_corr": round(_get_rolling_avg_corr(mat), 4),
        }
        for label, mat in mats.items()
    }


@router.get("/regime-status", response_model=RegimeStatusResponse, summary="Get correlation regime status")
async def get_regime_status(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers"),
    include_history: bool = Query(False, description="Include 90-day regime history"),
) -> RegimeStatusResponse:
    """Detect current market correlation regime via HMM."""
    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else EQUITY_UNIVERSE[:30]
    end = date.today()

    prices = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_prices(tk_list, lookback_days=290, end=end)
    )
    if prices.empty:
        raise HTTPException(404, "No price data found")

    returns = _compute_returns(prices)

    # Build rolling avg-corr series
    avg_corr_vals = []
    for start_i in range(63, len(returns)):
        sub = returns.iloc[start_i - 63:start_i].dropna(axis=1, how="any")
        if sub.shape[0] >= 10 and sub.shape[1] >= 2:
            c = sub.corr().values
            n = c.shape[0]
            upper = [c[i, j] for i in range(n) for j in range(i + 1, n)]
            avg_corr_vals.append(float(np.mean(upper)))

    if len(avg_corr_vals) < 20:
        return RegimeStatusResponse(
            as_of=str(end),
            current_regime=REGIME_NORMAL,
            confidence=0.5,
        )

    regime_info = detect_correlation_regime(pd.Series(avg_corr_vals))
    history = get_regime_history(90) if include_history else None

    return RegimeStatusResponse(
        as_of=str(end),
        current_regime=regime_info["current_regime"],
        confidence=regime_info["confidence"],
        state_means=regime_info.get("state_means"),
        transition_matrix=regime_info.get("transition_matrix"),
        recent_states=regime_info.get("state_sequence"),
        history=history,
    )


@router.get("/mst", response_model=MSTResponse, summary="Minimum Spanning Tree of correlation network")
async def get_mst(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers"),
    window: str = Query("63d", description="Correlation window: 21d | 63d | 126d | 252d"),
    use_cache: bool = Query(True, description="Use cached MST if available"),
) -> MSTResponse:
    """Build and return MST from correlation matrix to identify market hubs."""
    if window not in WINDOWS:
        raise HTTPException(400, f"window must be one of {list(WINDOWS.keys())}")

    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else EQUITY_UNIVERSE[:25]
    today = str(date.today())

    # Check MST cache
    if use_cache:
        with _db() as conn:
            row = conn.execute(
                "SELECT mst_json FROM mst_cache WHERE cached_at=? AND window_days=?",
                (today, WINDOWS[window]),
            ).fetchone()
        if row:
            data = json.loads(row["mst_json"])
            data["as_of"] = today
            return MSTResponse(**data)

    prices = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_prices(tk_list, lookback_days=WINDOWS[window] + 30)
    )
    if prices.empty:
        raise HTTPException(404, "No price data found")

    returns = _compute_returns(prices)
    mat = _rolling_corr_matrix(returns, WINDOWS[window])
    if mat.empty:
        raise HTTPException(500, "Insufficient data")

    mst_data = compute_mst(mat)

    # Cache MST
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO mst_cache VALUES (?,?,?)",
            (today, WINDOWS[window], json.dumps(mst_data)),
        )

    return MSTResponse(as_of=today, **mst_data)


@router.get("/dcc-garch", summary="DCC-GARCH dynamic correlation matrix")
async def get_dcc_garch(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers (max 20)"),
    lookback_days: int = Query(252, description="Lookback days for estimation"),
) -> Dict[str, Any]:
    """Compute DCC-GARCH(1,1) dynamic correlation matrix."""
    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else EQUITY_UNIVERSE[:15]
    tk_list = tk_list[:20]

    prices = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_prices(tk_list, lookback_days=lookback_days + 30)
    )
    if prices.empty:
        raise HTTPException(404, "No price data found")

    returns = _compute_returns(prices).tail(lookback_days)
    dcc = fit_dcc_garch(returns, max_assets=20)
    if not dcc:
        raise HTTPException(500, "DCC-GARCH estimation failed — insufficient data")
    return {**dcc, "as_of": str(date.today())}


@router.get("/pca", summary="PCA factor analysis of returns")
async def get_pca(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers"),
    window: int = Query(63, description="Rolling window in trading days"),
    n_components: int = Query(10, description="Number of principal components to return"),
) -> Dict[str, Any]:
    """Return PCA explained variance, loadings, and top contributors."""
    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else EQUITY_UNIVERSE[:30]
    prices = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_prices(tk_list, lookback_days=window + 30)
    )
    if prices.empty:
        raise HTTPException(404, "No price data found")

    returns = _compute_returns(prices)
    pca_result = compute_pca(returns, window=window, n_components=n_components)
    if not pca_result:
        raise HTTPException(500, "PCA computation failed")
    pca_result["as_of"] = str(date.today())
    return pca_result


@router.get("/tail-dependence", summary="Copula tail dependence coefficients")
async def get_tail_dependence(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers (max 15)"),
    lookback_days: int = Query(126, description="Lookback days"),
) -> Dict[str, Any]:
    """Compute Clayton/Gumbel copula lower/upper tail dependence for all pairs."""
    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else EQUITY_UNIVERSE[:10]
    tk_list = tk_list[:15]

    prices = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_prices(tk_list, lookback_days=lookback_days + 30)
    )
    if prices.empty:
        raise HTTPException(404, "No price data found")

    returns = _compute_returns(prices).tail(lookback_days)
    tail_dep = compute_tail_dependence(returns)
    return {"as_of": str(date.today()), "window_days": lookback_days, "results": tail_dep}


@router.get("/cross-asset", summary="Cross-asset correlation matrix")
async def get_cross_asset(
    tickers: Optional[str] = Query(None, description="Equity tickers to correlate vs benchmarks"),
    lookback_days: int = Query(63, description="Lookback window in days"),
) -> Dict[str, Any]:
    """Compute cross-asset correlations: equities vs TLT, GLD, USO, VIX."""
    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else EQUITY_UNIVERSE[:20]
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: compute_cross_asset_correlation(tk_list, lookback_days=lookback_days)
    )
    if not result:
        raise HTTPException(404, "Unable to compute cross-asset correlations")
    return result


@router.get("/diversification", summary="Portfolio diversification ratio")
async def get_diversification(
    tickers: str = Query(..., description="Comma-separated tickers"),
    weights: Optional[str] = Query(None, description="Comma-separated weights (equal if omitted)"),
    window: str = Query("63d", description="Correlation window"),
) -> Dict[str, Any]:
    """Compute diversification ratio and weighted average correlation for a portfolio."""
    tk_list = [t.strip().upper() for t in tickers.split(",")]
    if weights:
        wt_vals = [float(w) for w in weights.split(",")]
    else:
        wt_vals = [1.0 / len(tk_list)] * len(tk_list)

    if len(wt_vals) != len(tk_list):
        raise HTTPException(400, "Length of weights must match length of tickers")

    wt_dict = dict(zip(tk_list, wt_vals))

    prices = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_prices(tk_list, lookback_days=WINDOWS.get(window, 63) + 30)
    )
    if prices.empty:
        raise HTTPException(404, "No price data found")

    returns = _compute_returns(prices)
    mat = _rolling_corr_matrix(returns, WINDOWS.get(window, 63))
    if mat.empty:
        raise HTTPException(500, "Insufficient data for correlation matrix")

    # Compute realized volatilities
    daily_vol = returns.std()
    ann_vol = (daily_vol * math.sqrt(252)).to_dict()

    dr = compute_diversification_ratio(wt_dict, mat, volatilities=ann_vol)
    dr["as_of"] = str(date.today())
    dr["window"] = window
    return dr


@router.get("/alerts", summary="Recent correlation shock alerts")
async def get_alerts(
    days: int = Query(7, description="Lookback days for alerts"),
    severity: Optional[str] = Query(None, description="Filter by severity: high | medium | low"),
) -> Dict[str, Any]:
    """Return correlation shock alerts from alert log."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with _db() as conn:
        if severity:
            rows = conn.execute(
                "SELECT * FROM alert_log WHERE alerted_at>=? AND severity=? ORDER BY alerted_at DESC LIMIT 200",
                (cutoff, severity),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alert_log WHERE alerted_at>=? ORDER BY alerted_at DESC LIMIT 200",
                (cutoff,),
            ).fetchall()
    return {
        "as_of": str(date.today()),
        "lookback_days": days,
        "n_alerts": len(rows),
        "alerts": [dict(r) for r in rows],
    }


@router.post("/alert-config", response_model=AlertConfigResponse, summary="Update alert configuration")
async def post_alert_config(config: AlertConfigRequest) -> AlertConfigResponse:
    """Update correlation alert configuration parameters."""
    updated = False
    if config.shock_threshold is not None:
        _set_config("shock_threshold", str(config.shock_threshold))
        updated = True
    if config.div_ratio_warn is not None:
        _set_config("div_ratio_warn", str(config.div_ratio_warn))
        updated = True
    if config.email_alerts is not None:
        _set_config("email_alerts", "true" if config.email_alerts else "false")
        updated = True

    return AlertConfigResponse(
        shock_threshold=float(_get_config("shock_threshold", str(CORR_SHOCK_THRESHOLD))),
        div_ratio_warn=float(_get_config("div_ratio_warn", str(DIV_RATIO_WARN))),
        email_alerts=_get_config("email_alerts", "false").lower() == "true",
        updated=updated,
    )


@router.get("/full-analysis", summary="Run full correlation analysis pipeline")
async def get_full_analysis(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers (default: top 20 universe)"),
    portfolio_weights: Optional[str] = Query(None, description="Comma-separated weights matching tickers"),
    run_dcc: bool = Query(True, description="Include DCC-GARCH (slower)"),
) -> Dict[str, Any]:
    """Run all correlation analyses: matrices, DCC, HMM, copulas, PCA, MST, cross-asset."""
    tk_list = [t.strip().upper() for t in tickers.split(",")] if tickers else EQUITY_UNIVERSE[:20]
    pw_dict = None
    if portfolio_weights:
        wt_vals = [float(w) for w in portfolio_weights.split(",")]
        if len(wt_vals) == len(tk_list):
            pw_dict = dict(zip(tk_list, wt_vals))

    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: full_correlation_analysis(tk_list, pw_dict, run_dcc=run_dcc)
    )
    return result


@router.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    """Verify module is responsive and DB is accessible."""
    try:
        with _db() as conn:
            conn.execute("SELECT 1").fetchone()
        return {"status": "ok", "module": "correlation_monitor_v2", "dim": "080"}
    except Exception as exc:
        raise HTTPException(503, f"DB error: {exc}")


# ---------------------------------------------------------------------------
# Standalone test (python -m sentinel.sbx.correlation_monitor_v2)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    tickers_test = ["SPY", "QQQ", "TLT", "GLD", "AAPL", "MSFT", "JPM", "XOM", "NVDA", "AMZN"]
    print("Running full correlation analysis for test universe...")
    result = full_correlation_analysis(tickers_test, run_dcc=True)
    print(f"Regime: {result.get('regime', {}).get('current_regime', 'N/A')}")
    print(f"Regime confidence: {result.get('regime', {}).get('confidence', 'N/A')}")
    if "rolling_matrices" in result:
        for w, m in result["rolling_matrices"].items():
            print(f"  {w} avg corr: {m.get('avg_pairwise_corr', 'N/A')}")
    if "mst" in result and isinstance(result["mst"], dict):
        print(f"MST hubs: {result['mst'].get('hub_tickers', [])}")
    if "pca" in result and isinstance(result["pca"], dict):
        evr = result["pca"].get("explained_variance_ratio", [])
        if evr:
            print(f"PC1 explains: {evr[0]:.1%} of variance")
    if "dcc_garch" in result and isinstance(result["dcc_garch"], dict):
        dcc = result["dcc_garch"]
        if "a" in dcc:
            print(f"DCC-GARCH: a={dcc['a']}, b={dcc['b']}")
    print("Done.")
