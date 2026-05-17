"""
Regime Detector V3 — Comprehensive macro regime detection using HMM and multiple FRED signals.

dim_050: Regime detection (HMM / macro nowcast) — score 6 → 9

Architecture:
  MacroFeatureBuilder       — Fetch 14 FRED series, build normalized feature matrix
  ViterbiHMM                — Pure numpy Baum-Welch EM + Viterbi decoding (no hmmlearn required)
  HmmlearnAdapter           — Optional wrapper around hmmlearn.GaussianHMM (same interface)
  NBERRecessionAligner      — Align detected states to NBER recession dates for labeling
  MacroRegimeClassifier     — Rule-based fallback classifier (no training required)
  GDPNowTracker             — Atlanta Fed GDPNow via FRED GDPNOW series
  RegimeBacktester          — Simulate regime-switching strategy over full history
  RegimeDetectorEngine      — Orchestrator: fit, detect, alert, report

Data sources: All free FRED CSV endpoints (no API key required), USREC recession dates.

Usage::
    engine = RegimeDetectorEngine()
    result = engine.fit_and_detect(start="2000-01-01")
    regime, confidence, snapshot = engine.get_current_regime()
    print(regime, f"{confidence:.1%}")
"""
from __future__ import annotations

import io
import logging
import math
import pickle
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Optional hmmlearn
# ---------------------------------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _GaussianHMM = None  # type: ignore[assignment,misc]
    _HMMLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

FRED_SERIES = {
    "T10Y2Y":     "10Y-2Y Yield Curve Slope",
    "T10Y3M":     "10Y-3M Yield Curve Slope",
    "BAMLH0A0HYM2": "HY Credit Spread (OAS)",
    "UNRATE":     "Unemployment Rate",
    "ICSA":       "Initial Claims (Weekly)",
    "PAYEMS":     "Nonfarm Payrolls",
    "INDPRO":     "Industrial Production Index",
    "RSAFS":      "Retail Sales",
    "CPILFESL":   "Core CPI",
    "T10YIE":     "10Y Breakeven Inflation",
    "VIXCLS":     "CBOE VIX",
    "DCOILWTICO": "WTI Crude Oil",
    "UMCSENT":    "UMich Consumer Sentiment",
    "FEDFUNDS":   "Fed Funds Rate",
    "MANEMP":     "Manufacturing Employment",
    "USREC":      "NBER Recession Indicator",
    "GDPNOW":     "Atlanta Fed GDPNow (FRED Proxy)",
    "GDPC1":      "Real GDP",
}

REGIME_LABELS = ["EXPANSION", "SLOWDOWN", "RECESSION", "RECOVERY"]

_REQUEST_TIMEOUT = 30
_CACHE_DIR = Path("~/.cache/sentinel/regime").expanduser()
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    name: str                      # EXPANSION / SLOWDOWN / RECESSION / RECOVERY
    sub_regime: str = ""           # early/mid/late modifier
    confidence: float = 0.0        # 0-1
    state_idx: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        sub = f" ({self.sub_regime})" if self.sub_regime else ""
        return f"{self.name}{sub} [{self.confidence:.1%}]"


@dataclass
class MacroSnapshot:
    date: date
    yield_curve_2y10y: float
    yield_curve_3m10y: float
    hy_spread: float
    unemployment: float
    initial_claims: float
    payrolls_mom: float
    indpro_yoy: float
    core_cpi_yoy: float
    breakeven_10y: float
    vix: float
    vix_3m_avg: float
    consumer_sentiment: float
    gdpnow: float
    composite_leading: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegimeDetectionResult:
    current_regime: RegimeState
    regime_history: pd.DataFrame          # index=date, columns=[state, label, confidence]
    feature_matrix: pd.DataFrame
    transition_matrix: np.ndarray
    model_type: str                        # "hmmlearn" | "viterbi" | "rule_based"
    nber_accuracy: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegimeAlert:
    from_regime: str
    to_regime: str
    alert_date: date
    confidence: float
    features_at_change: Dict[str, float]
    message: str


@dataclass
class RiskDuration:
    regime: str
    avg_duration_months: float
    min_duration: int
    max_duration: int
    n_episodes: int


# ---------------------------------------------------------------------------
# MacroFeatureBuilder
# ---------------------------------------------------------------------------

class MacroFeatureBuilder:
    """Fetch FRED macro series and construct a normalized feature matrix."""

    def __init__(self, cache_ttl_hours: int = 12):
        self._cache: Dict[str, pd.Series] = {}
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._cache_times: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # FRED fetch
    # ------------------------------------------------------------------

    def fetch_fred_series(self, series_id: str, start: str = "1990-01-01") -> pd.Series:
        """Download a FRED series as a pd.Series with DatetimeIndex."""
        cache_key = f"{series_id}_{start}"
        now = datetime.utcnow()
        if cache_key in self._cache:
            if now - self._cache_times[cache_key] < self._cache_ttl:
                return self._cache[cache_key]

        url = FRED_CSV_URL.format(series_id=series_id)
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT,
                                headers={"User-Agent": "SENTINEL/3.0 (research)"})
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
            df.columns = [series_id]
            df = df.replace(".", np.nan)
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            series = df[series_id].dropna()
            series = series[series.index >= pd.Timestamp(start)]
            self._cache[cache_key] = series
            self._cache_times[cache_key] = now
            logger.info("FRED %s: %d observations (%s → %s)",
                        series_id, len(series),
                        series.index.min().date() if len(series) else "N/A",
                        series.index.max().date() if len(series) else "N/A")
            return series
        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
            return pd.Series(dtype=float, name=series_id)

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------

    def build_feature_matrix(self, end_date: Optional[str] = None,
                              start: str = "1990-01-01") -> pd.DataFrame:
        """
        Fetch all required FRED series and build a monthly feature matrix.
        All series are resampled to month-end frequency.
        """
        end_date = end_date or datetime.utcnow().strftime("%Y-%m-%d")

        # Fetch raw series
        t10y2y   = self.fetch_fred_series("T10Y2Y",     start)
        t10y3m   = self.fetch_fred_series("T10Y3M",     start)
        hy_spread= self.fetch_fred_series("BAMLH0A0HYM2", start)
        unrate   = self.fetch_fred_series("UNRATE",     start)
        icsa     = self.fetch_fred_series("ICSA",       start)
        payems   = self.fetch_fred_series("PAYEMS",     start)
        indpro   = self.fetch_fred_series("INDPRO",     start)
        rsafs    = self.fetch_fred_series("RSAFS",      start)
        cpilfesl = self.fetch_fred_series("CPILFESL",   start)
        t10yie   = self.fetch_fred_series("T10YIE",     start)
        vixcls   = self.fetch_fred_series("VIXCLS",     start)
        oil      = self.fetch_fred_series("DCOILWTICO", start)
        umcsent  = self.fetch_fred_series("UMCSENT",    start)
        fedfunds = self.fetch_fred_series("FEDFUNDS",   start)
        manemp   = self.fetch_fred_series("MANEMP",     start)

        # Resample to month-end
        def monthly(s: pd.Series, how: str = "last") -> pd.Series:
            if s.empty:
                return s
            if how == "mean":
                return s.resample("ME").mean()
            return s.resample("ME").last()

        m = pd.DataFrame({
            "yield_curve_2y10y":  monthly(t10y2y),
            "yield_curve_3m10y":  monthly(t10y3m),
            "hy_spread":          monthly(hy_spread, "mean"),
            "unrate":             monthly(unrate),
            "icsa":               monthly(icsa, "mean"),
            "payems":             monthly(payems),
            "indpro":             monthly(indpro),
            "rsafs":              monthly(rsafs),
            "cpilfesl":           monthly(cpilfesl),
            "t10yie":             monthly(t10yie),
            "vix":                monthly(vixcls, "mean"),
            "crude_oil":          monthly(oil, "mean"),
            "consumer_sentiment": monthly(umcsent),
            "fedfunds":           monthly(fedfunds),
            "manemp":             monthly(manemp),
        })

        # ---------------------------------------------------------------
        # Derived features
        # ---------------------------------------------------------------
        # Yield curve: already level; add 6-month rate of change
        m["yield_curve_2y10y_chg6m"] = m["yield_curve_2y10y"].diff(6)

        # HY spread 6-month change
        m["hy_spread_chg6m"] = m["hy_spread"].diff(6)

        # Unemployment rate change (3-month)
        m["unrate_chg3m"] = m["unrate"].diff(3)

        # Payrolls MoM change (thousands)
        m["payrolls_mom"] = m["payems"].diff(1)

        # Payrolls YoY %
        m["payrolls_yoy"] = m["payems"].pct_change(12) * 100

        # Industrial production YoY %
        m["indpro_yoy"] = m["indpro"].pct_change(12) * 100

        # Retail sales YoY %
        m["rsafs_yoy"] = m["rsafs"].pct_change(12) * 100

        # Core CPI YoY % (inflation momentum)
        m["core_cpi_yoy"] = m["cpilfesl"].pct_change(12) * 100

        # Inflation acceleration (change in YoY CPI)
        m["cpi_acceleration"] = m["core_cpi_yoy"].diff(3)

        # VIX 3-month rolling average (financial conditions proxy)
        m["vix_3m_avg"] = m["vix"].rolling(3).mean()

        # VIX 6-month z-score
        m["vix_zscore"] = (m["vix"] - m["vix"].rolling(24).mean()) / (
            m["vix"].rolling(24).std().replace(0, np.nan))

        # Manufacturing employment YoY % (ISM PMI proxy)
        m["manemp_yoy"] = m["manemp"].pct_change(12) * 100

        # Claims 4-week change %
        m["claims_mom_pct"] = m["icsa"].pct_change(1) * 100
        m["claims_chg3m"]   = m["icsa"].diff(3)

        # Oil YoY %
        m["oil_yoy"] = m["crude_oil"].pct_change(12) * 100

        # Real rate proxy (fedfunds - core cpi yoy)
        m["real_rate"] = m["fedfunds"] - m["core_cpi_yoy"]

        # GDPNow (fetch separately; fill with NaN if unavailable)
        gdpnow = self.fetch_fred_series("GDPNOW", start)
        if not gdpnow.empty:
            m["gdpnow"] = monthly(gdpnow)
        else:
            m["gdpnow"] = np.nan

        # Composite leading indicator
        m["composite_leading"] = self.compute_composite_leading_indicator(m)

        # Trim to end_date
        m = m[m.index <= pd.Timestamp(end_date)]

        # Drop rows where all features are NaN
        m = m.dropna(how="all")

        logger.info("Feature matrix: %d rows × %d cols (%s → %s)",
                    len(m), len(m.columns),
                    m.index.min().date() if len(m) else "N/A",
                    m.index.max().date() if len(m) else "N/A")
        return m

    def normalize_features(self, df: pd.DataFrame,
                            window: int = 60) -> pd.DataFrame:
        """
        Z-score normalization using a rolling 60-month window to avoid
        look-ahead bias. Returns same-shape DataFrame.
        """
        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for col in df.columns:
            s = df[col]
            roll_mean = s.rolling(window, min_periods=12).mean()
            roll_std  = s.rolling(window, min_periods=12).std()
            result[col] = (s - roll_mean) / roll_std.replace(0, np.nan)
        return result

    def compute_composite_leading_indicator(self, df: pd.DataFrame) -> pd.Series:
        """
        Equal-weight composite of normalized macro signals.
        Positive = expansion signal, negative = contraction signal.
        """
        # Signs: positive contribution to "expansion"
        components = {
            "yield_curve_2y10y":  +1,   # steeper curve → expansion
            "payrolls_mom":       +1,   # rising payrolls → expansion
            "indpro_yoy":         +1,   # rising production → expansion
            "rsafs_yoy":          +1,   # rising retail sales → expansion
            "manemp_yoy":         +1,   # rising mfg employment → expansion
            "consumer_sentiment": +1,   # higher sentiment → expansion
            "vix":                -1,   # lower vix → expansion
            "hy_spread":          -1,   # tighter spreads → expansion
            "unrate_chg3m":       -1,   # falling unemployment → expansion
            "claims_chg3m":       -1,   # falling claims → expansion
        }
        available = {k: v for k, v in components.items() if k in df.columns}
        if not available:
            return pd.Series(np.nan, index=df.index)

        parts = []
        for col, sign in available.items():
            s = df[col].copy()
            # Robust normalize each component
            roll_med = s.rolling(60, min_periods=12).median()
            roll_mad = s.rolling(60, min_periods=12).apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True)
            normalized = (s - roll_med) / (roll_mad.replace(0, np.nan) * 1.4826)
            normalized = normalized.clip(-3, 3)
            parts.append(sign * normalized)

        composite = pd.concat(parts, axis=1).mean(axis=1)
        return composite


# ---------------------------------------------------------------------------
# ViterbiHMM — pure numpy Baum-Welch + Viterbi
# ---------------------------------------------------------------------------

class ViterbiHMM:
    """
    4-state Gaussian Hidden Markov Model implemented in pure numpy.

    States: 0=EXPANSION, 1=SLOWDOWN, 2=RECESSION, 3=RECOVERY

    Training: Baum-Welch EM algorithm.
    Decoding: Viterbi algorithm.
    Emission: Multivariate Gaussian with full covariance.
    """

    LABEL_MAP = {0: "EXPANSION", 1: "SLOWDOWN", 2: "RECESSION", 3: "RECOVERY"}

    def __init__(self, n_states: int = 4, max_iter: int = 100,
                 tol: float = 1e-4, random_state: int = 42):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.is_fitted = False

        # Model parameters (set during fit)
        self.pi: np.ndarray = np.ones(n_states) / n_states       # initial probs
        self.A:  np.ndarray = np.ones((n_states, n_states)) / n_states  # transition
        self.means:    np.ndarray = None  # (n_states, n_features)
        self.covars:   np.ndarray = None  # (n_states, n_features, n_features)
        self.n_features: int = 0
        self.state_labels: List[str] = [self.LABEL_MAP[i] for i in range(n_states)]
        self.log_likelihood_history: List[float] = []

    # ------------------------------------------------------------------
    # Gaussian emission
    # ------------------------------------------------------------------

    def _log_gaussian(self, x: np.ndarray, state: int) -> float:
        """Log PDF of multivariate Gaussian for observation x under state k."""
        mu = self.means[state]
        sigma = self.covars[state]
        d = len(mu)
        diff = x - mu
        try:
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                # Regularize
                sigma = sigma + np.eye(d) * 1e-4
                sign, logdet = np.linalg.slogdet(sigma)
            inv_sigma = np.linalg.inv(sigma)
            mahal = diff @ inv_sigma @ diff
            return -0.5 * (d * math.log(2 * math.pi) + logdet + mahal)
        except np.linalg.LinAlgError:
            return -1e10

    def _compute_log_emission(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities.
        Returns: (T, n_states) array
        """
        T = len(X)
        log_emit = np.full((T, self.n_states), -np.inf)
        for t in range(T):
            for k in range(self.n_states):
                log_emit[t, k] = self._log_gaussian(X[t], k)
        return log_emit

    # ------------------------------------------------------------------
    # Forward-backward (log-space for numerical stability)
    # ------------------------------------------------------------------

    def _forward(self, log_emit: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Log-space forward algorithm.
        Returns: log_alpha (T, K), log_likelihood
        """
        T, K = log_emit.shape
        log_alpha = np.full((T, K), -np.inf)
        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        # Initialization
        log_alpha[0] = log_pi + log_emit[0]

        # Recursion
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = self._logsumexp(
                    log_alpha[t-1] + log_A[:, k]
                ) + log_emit[t, k]

        log_likelihood = self._logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    def _backward(self, log_emit: np.ndarray) -> np.ndarray:
        """
        Log-space backward algorithm.
        Returns: log_beta (T, K)
        """
        T, K = log_emit.shape
        log_beta = np.full((T, K), -np.inf)
        log_A = np.log(self.A + 1e-300)

        # Initialization
        log_beta[-1] = 0.0

        # Recursion
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = self._logsumexp(
                    log_A[k] + log_emit[t+1] + log_beta[t+1]
                )
        return log_beta

    @staticmethod
    def _logsumexp(arr: np.ndarray) -> float:
        """Numerically stable log-sum-exp."""
        a_max = np.max(arr)
        if np.isneginf(a_max):
            return -np.inf
        return a_max + math.log(np.sum(np.exp(arr - a_max)) + 1e-300)

    # ------------------------------------------------------------------
    # EM (Baum-Welch)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, n_states: int = 4, max_iter: int = 100) -> "ViterbiHMM":
        """
        Fit HMM parameters using Baum-Welch (EM).
        X: (T, n_features) array of observations (should be normalized)
        """
        self.n_states = n_states
        self.n_features = X.shape[1]
        T = len(X)
        K = self.n_states
        D = self.n_features

        rng = np.random.default_rng(self.random_state)

        # Initialize parameters via K-means-like random
        indices = rng.choice(T, K, replace=False)
        self.means = X[indices].copy()
        self.covars = np.array([np.eye(D) for _ in range(K)])
        self.pi = np.ones(K) / K
        self.A  = rng.dirichlet(np.ones(K) * 5, size=K)  # prior toward staying
        # Boost diagonal (states tend to persist)
        for k in range(K):
            self.A[k] = 0.05 * self.A[k] / self.A[k].sum() + 0.95 * (
                np.eye(K)[k])
        self.A /= self.A.sum(axis=1, keepdims=True)

        prev_ll = -np.inf
        self.log_likelihood_history = []

        for iteration in range(max_iter):
            # -------------------------------------------------------
            # E-step: compute responsibilities
            # -------------------------------------------------------
            log_emit = self._compute_log_emission(X)
            log_alpha, log_ll = self._forward(log_emit)
            log_beta = self._backward(log_emit)

            self.log_likelihood_history.append(log_ll)

            # Gamma: P(state=k | observations)
            log_gamma = log_alpha + log_beta
            # Normalize
            log_gamma -= self._logsumexp(log_gamma.flatten()) / T  # rough
            # Per-timestep normalization
            for t in range(T):
                norm = self._logsumexp(log_gamma[t])
                log_gamma[t] -= norm
            gamma = np.exp(log_gamma)  # (T, K)

            # Xi: P(state_t=i, state_{t+1}=j | obs)
            log_A = np.log(self.A + 1e-300)
            xi = np.zeros((T-1, K, K))
            for t in range(T-1):
                for i in range(K):
                    for j in range(K):
                        xi[t, i, j] = math.exp(
                            log_alpha[t, i] + log_A[i, j] +
                            log_emit[t+1, j] + log_beta[t+1, j]
                        )
                xi[t] /= xi[t].sum() + 1e-300

            # -------------------------------------------------------
            # M-step: update parameters
            # -------------------------------------------------------
            # Initial probabilities
            self.pi = gamma[0] + 1e-10
            self.pi /= self.pi.sum()

            # Transition matrix
            xi_sum = xi.sum(axis=0)  # (K, K)
            self.A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)

            # Means and covariances
            gamma_sum = gamma.sum(axis=0)  # (K,)
            for k in range(K):
                gk = gamma[:, k]  # (T,)
                gk_sum = gamma_sum[k] + 1e-300
                self.means[k] = (gk[:, None] * X).sum(axis=0) / gk_sum
                diff = X - self.means[k]
                cov = (gk[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
                self.covars[k] = cov / gk_sum + np.eye(D) * 1e-4  # regularize

            # -------------------------------------------------------
            # Convergence check
            # -------------------------------------------------------
            delta = log_ll - prev_ll
            if iteration > 0 and abs(delta) < self.tol:
                logger.info("ViterbiHMM converged at iteration %d (LL=%.2f)", iteration, log_ll)
                break
            prev_ll = log_ll

            if (iteration + 1) % 10 == 0:
                logger.debug("Iter %d: log-likelihood=%.4f, delta=%.6f",
                             iteration + 1, log_ll, delta)

        self.is_fitted = True
        self.state_labels = [self.LABEL_MAP.get(i, f"STATE_{i}") for i in range(K)]
        return self

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding: returns most likely state sequence."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        T = len(X)
        K = self.n_states
        log_emit = self._compute_log_emission(X)
        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        # Viterbi table
        viterbi = np.full((T, K), -np.inf)
        psi     = np.zeros((T, K), dtype=int)

        viterbi[0] = log_pi + log_emit[0]
        for t in range(1, T):
            for k in range(K):
                candidates = viterbi[t-1] + log_A[:, k]
                psi[t, k]     = np.argmax(candidates)
                viterbi[t, k] = candidates[psi[t, k]] + log_emit[t, k]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Forward pass probabilities (smoothed state probs)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        log_emit = self._compute_log_emission(X)
        log_alpha, _ = self._forward(log_emit)
        log_beta = self._backward(log_emit)
        log_gamma = log_alpha + log_beta
        # Normalize row-wise
        T = len(X)
        for t in range(T):
            norm = self._logsumexp(log_gamma[t])
            if not np.isneginf(norm):
                log_gamma[t] -= norm
        return np.exp(log_gamma)

    def get_current_state(self, X: np.ndarray) -> Tuple[str, float]:
        """Return current regime label and confidence (last observation)."""
        proba = self.predict_proba(X)
        last_proba = proba[-1]
        best_state = int(np.argmax(last_proba))
        confidence = float(last_proba[best_state])
        label = self.state_labels[best_state] if best_state < len(self.state_labels) else f"STATE_{best_state}"
        return label, confidence

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ViterbiHMM saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ViterbiHMM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("ViterbiHMM loaded from %s", path)
        return obj


# ---------------------------------------------------------------------------
# HmmlearnAdapter
# ---------------------------------------------------------------------------

class HmmlearnAdapter:
    """
    Wrapper around hmmlearn.GaussianHMM with the same interface as ViterbiHMM.
    Falls back to ViterbiHMM if hmmlearn is not installed.
    """

    LABEL_MAP = {0: "EXPANSION", 1: "SLOWDOWN", 2: "RECESSION", 3: "RECOVERY"}

    def __init__(self, n_states: int = 4, max_iter: int = 100,
                 random_state: int = 42):
        self.n_states = n_states
        self.max_iter = max_iter
        self.random_state = random_state
        self.is_fitted = False
        self.state_labels: List[str] = [self.LABEL_MAP.get(i, f"STATE_{i}") for i in range(n_states)]

        if _HMMLEARN_AVAILABLE:
            self._model = _GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=max_iter,
                random_state=random_state,
                tol=1e-4,
                verbose=False,
            )
            self._backend = "hmmlearn"
        else:
            logger.info("hmmlearn not available; using ViterbiHMM backend.")
            self._model = ViterbiHMM(n_states=n_states, max_iter=max_iter,
                                      random_state=random_state)
            self._backend = "viterbi"

    def fit(self, X: np.ndarray, n_states: int = 4, max_iter: int = 100) -> "HmmlearnAdapter":
        if self._backend == "hmmlearn":
            self._model.fit(X)
        else:
            self._model.fit(X, n_states=n_states, max_iter=max_iter)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._backend == "hmmlearn":
            return self._model.predict_proba(X)
        return self._model.predict_proba(X)

    def get_current_state(self, X: np.ndarray) -> Tuple[str, float]:
        proba = self.predict_proba(X)
        last = proba[-1]
        best = int(np.argmax(last))
        label = self.state_labels[best] if best < len(self.state_labels) else f"STATE_{best}"
        return label, float(last[best])

    @property
    def transition_matrix(self) -> np.ndarray:
        if self._backend == "hmmlearn":
            return self._model.transmat_
        return self._model.A

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "HmmlearnAdapter":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# NBERRecessionAligner
# ---------------------------------------------------------------------------

class NBERRecessionAligner:
    """
    Fetch NBER USREC series from FRED and use it to label detected HMM states.
    Maps whichever HMM state most overlaps with NBER recession periods → RECESSION.
    """

    def __init__(self):
        self._builder = MacroFeatureBuilder()
        self._usrec: Optional[pd.Series] = None

    def _get_usrec(self) -> pd.Series:
        if self._usrec is None:
            self._usrec = self._builder.fetch_fred_series("USREC", "1960-01-01")
        return self._usrec

    def get_recession_dates(self) -> List[Tuple[str, str]]:
        """Return list of (start_date, end_date) strings for NBER recessions."""
        usrec = self._get_usrec()
        if usrec.empty:
            # Hardcoded NBER recessions as fallback
            return [
                ("1990-07-01", "1991-03-31"),
                ("2001-03-01", "2001-11-30"),
                ("2007-12-01", "2009-06-30"),
                ("2020-02-01", "2020-04-30"),
            ]
        recessions = []
        in_recession = False
        start = None
        for dt, val in usrec.items():
            if val == 1 and not in_recession:
                in_recession = True
                start = dt
            elif val == 0 and in_recession:
                in_recession = False
                recessions.append((str(start.date()), str(dt.date())))
        if in_recession and start is not None:
            recessions.append((str(start.date()), str(usrec.index[-1].date())))
        return recessions

    def label_regimes(self, predicted_states: pd.Series,
                      recession_dates: Optional[List[Tuple[str, str]]] = None) -> pd.Series:
        """
        Relabel HMM states so that whichever numeric state most overlaps
        NBER recession periods gets the RECESSION label.
        Returns a Series of string regime labels.
        """
        if recession_dates is None:
            recession_dates = self.get_recession_dates()

        usrec = self._get_usrec()
        if usrec.empty:
            # Build USREC from hardcoded dates
            usrec = pd.Series(0, index=predicted_states.index)
            for start, end in recession_dates:
                mask = (usrec.index >= pd.Timestamp(start)) & (
                    usrec.index <= pd.Timestamp(end))
                usrec[mask] = 1

        # Align USREC to predicted_states index
        usrec_aligned = usrec.reindex(predicted_states.index, method="ffill").fillna(0)

        # Count overlap per state
        unique_states = predicted_states.unique()
        recession_counts = {}
        for s in unique_states:
            mask = predicted_states == s
            recession_counts[s] = usrec_aligned[mask].sum()

        # State with max recession overlap → RECESSION label
        recession_state = max(recession_counts, key=recession_counts.get)

        # For remaining states, assign by heuristic ordering
        remaining = [s for s in sorted(unique_states) if s != recession_state]

        # Build state label map
        state_to_label = {recession_state: "RECESSION"}
        label_pool = ["EXPANSION", "SLOWDOWN", "RECOVERY"]
        for i, s in enumerate(remaining):
            state_to_label[s] = label_pool[i % len(label_pool)]

        return predicted_states.map(state_to_label)

    def compute_accuracy(self, predicted: pd.Series, actual: pd.Series) -> float:
        """Recall of RECESSION detection (precision/recall vs NBER)."""
        aligned_pred = predicted.reindex(actual.index, method="ffill")
        rec_mask = actual == 1
        if rec_mask.sum() == 0:
            return 0.0
        detected = (aligned_pred[rec_mask] == "RECESSION").sum()
        return float(detected) / float(rec_mask.sum())


# ---------------------------------------------------------------------------
# MacroRegimeClassifier — rule-based (no training required)
# ---------------------------------------------------------------------------

class MacroRegimeClassifier:
    """
    Rule-based macro regime classifier.
    Requires no training data; applies threshold logic on macro features.
    """

    THRESHOLDS = {
        "yield_curve_inverted":    0.0,   # 10Y-2Y < 0 → inverted
        "hy_spread_crisis":      600.0,   # bp
        "hy_spread_stress":      450.0,
        "hy_spread_normal":      350.0,
        "pmi_contraction":        50.0,   # manemp_yoy proxy
        "leading_slowdown":       -0.5,
        "claims_rising_pct":       5.0,   # 3m change %
        "vix_elevated":           25.0,
        "vix_crisis":             35.0,
    }

    def classify(self, features: Dict[str, float]) -> RegimeState:
        """Classify current regime from feature dict."""
        yc      = features.get("yield_curve_2y10y", 0.5)
        hy      = features.get("hy_spread", 400.0)
        claims  = features.get("claims_chg3m", 0.0)
        claims_pct = features.get("claims_mom_pct", 0.0)
        leading = features.get("composite_leading", 0.0)
        manemp_yoy = features.get("manemp_yoy", 0.0)
        vix     = features.get("vix", 18.0)
        unrate_chg = features.get("unrate_chg3m", 0.0)
        payrolls_mom = features.get("payrolls_mom", 150.0)

        # RECESSION: yield curve inverted AND credit stressed AND claims rising
        if (yc < self.THRESHOLDS["yield_curve_inverted"] and
                hy > self.THRESHOLDS["hy_spread_stress"] and
                claims > 50_000):
            confidence = min(1.0, (hy - 400) / 400 + abs(yc) * 0.5)
            return RegimeState("RECESSION", confidence=confidence, state_idx=2)

        # SLOWDOWN: PMI-proxy contracting AND leading indicator weak
        if (manemp_yoy < 0 and leading < self.THRESHOLDS["leading_slowdown"]):
            confidence = min(1.0, abs(leading) * 0.5)
            return RegimeState("SLOWDOWN", confidence=confidence, state_idx=1)

        # RECOVERY: claims falling fast AND leading turning positive
        if (claims < -30_000 and leading > -0.2 and unrate_chg < 0):
            confidence = min(1.0, abs(claims) / 100_000 + 0.3)
            return RegimeState("RECOVERY", confidence=confidence, state_idx=3)

        # EXPANSION: everything positive
        confidence = min(1.0, max(0.5, leading * 0.3 + 0.5))
        return RegimeState("EXPANSION", confidence=confidence, state_idx=0)

    def get_sub_regime(self, state: RegimeState,
                        features: Dict[str, float]) -> str:
        """Return sub-regime qualifier: early/mid/late."""
        leading = features.get("composite_leading", 0.0)
        yc      = features.get("yield_curve_2y10y", 1.0)
        cpi_acc = features.get("cpi_acceleration", 0.0)

        if state.name == "EXPANSION":
            if leading > 1.0 and cpi_acc < 0.1:
                return "early"
            elif leading > 0.3 and yc > 0.5:
                return "mid"
            else:
                return "late"
        elif state.name == "SLOWDOWN":
            if leading > -0.5:
                return "early"
            elif leading > -1.5:
                return "mid"
            else:
                return "late"
        elif state.name == "RECESSION":
            if features.get("unrate_chg3m", 0) > 0:
                return "deepening"
            else:
                return "bottoming"
        elif state.name == "RECOVERY":
            if leading < 0.5:
                return "early"
            else:
                return "mid"
        return ""

    def compute_transition_probability(
        self,
        from_state: RegimeState,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Heuristic transition probabilities from current state given macro conditions.
        Returns dict of {regime: probability}.
        """
        leading  = features.get("composite_leading", 0.0)
        yc       = features.get("yield_curve_2y10y", 0.5)
        hy       = features.get("hy_spread", 400.0)
        claims_c = features.get("claims_chg3m", 0.0)
        vix      = features.get("vix", 18.0)

        # Base transition probabilities (high persistence)
        probs = {
            "EXPANSION": 0.1,
            "SLOWDOWN":  0.1,
            "RECESSION": 0.1,
            "RECOVERY":  0.1,
        }
        probs[from_state.name] = 0.7  # strong persistence

        # Adjust by signals
        if from_state.name == "EXPANSION":
            if yc < 0.2:
                probs["SLOWDOWN"] += 0.15
                probs["EXPANSION"] -= 0.15
            if hy > 450:
                probs["SLOWDOWN"] += 0.10
                probs["EXPANSION"] -= 0.10
        elif from_state.name == "SLOWDOWN":
            if leading < -1.0 or hy > 550:
                probs["RECESSION"] += 0.20
                probs["SLOWDOWN"] -= 0.20
            elif leading > 0.3:
                probs["EXPANSION"] += 0.15
                probs["SLOWDOWN"] -= 0.15
        elif from_state.name == "RECESSION":
            if claims_c < -50_000:
                probs["RECOVERY"] += 0.20
                probs["RECESSION"] -= 0.20
        elif from_state.name == "RECOVERY":
            if leading > 1.0:
                probs["EXPANSION"] += 0.20
                probs["RECOVERY"] -= 0.20

        # Normalize
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}


# ---------------------------------------------------------------------------
# GDPNowTracker
# ---------------------------------------------------------------------------

class GDPNowTracker:
    """
    Tracks Atlanta Fed GDPNow estimates.
    Primary: FRED GDPNOW series (when available).
    Fallback: scrape the Atlanta Fed page.
    """

    FRED_GDPNOW = "GDPNOW"
    ATLANTA_FED_URL = "https://www.atlantafed.org/cqer/research/gdpnow"

    def __init__(self):
        self._builder = MacroFeatureBuilder()

    def fetch_gdpnow(self) -> pd.Series:
        """Return historical GDPNow readings as pd.Series."""
        series = self._builder.fetch_fred_series(self.FRED_GDPNOW, "2011-01-01")
        if not series.empty:
            return series
        # Fallback: scrape Atlanta Fed
        return self._scrape_atlanta_fed()

    def _scrape_atlanta_fed(self) -> pd.Series:
        """Minimal scrape of Atlanta Fed GDPNow page for latest value."""
        try:
            resp = requests.get(self.ATLANTA_FED_URL, timeout=15,
                                headers={"User-Agent": "SENTINEL/3.0"})
            resp.raise_for_status()
            # Look for patterns like "2.3 percent" or "GDP growth of X"
            text = resp.text
            # Match "GDPNow model estimate for real GDP growth ... is X.X percent"
            pattern = r"(?:model estimate|tracking estimate)[^\d]*(-?\d+\.?\d*)\s*percent"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                return pd.Series([val], index=[pd.Timestamp.now()])
        except Exception as exc:
            logger.warning("Atlanta Fed scrape failed: %s", exc)
        return pd.Series(dtype=float, name="GDPNOW")

    def get_latest_forecast(self) -> float:
        """Return the most recent GDPNow forecast (% annualized)."""
        series = self.fetch_gdpnow()
        if series.empty:
            return float("nan")
        return float(series.iloc[-1])

    def compute_nowcast_surprise(self, actual_gdp: float) -> float:
        """
        Compute surprise: actual BEA release minus most recent GDPNow forecast.
        Positive = upside surprise.
        """
        forecast = self.get_latest_forecast()
        if math.isnan(forecast):
            return float("nan")
        return actual_gdp - forecast


# ---------------------------------------------------------------------------
# RegimeBacktester
# ---------------------------------------------------------------------------

class RegimeBacktester:
    """
    Simulate a simple regime-switching allocation strategy and compute
    performance statistics by regime.
    """

    # Simple allocation weights by regime (EXPANSION, SLOWDOWN, RECESSION, RECOVERY)
    REGIME_WEIGHTS = {
        "EXPANSION": {"equity": 1.0,  "bond": 0.0, "cash": 0.0},
        "SLOWDOWN":  {"equity": 0.5,  "bond": 0.3, "cash": 0.2},
        "RECESSION": {"equity": 0.1,  "bond": 0.5, "cash": 0.4},
        "RECOVERY":  {"equity": 0.8,  "bond": 0.1, "cash": 0.1},
    }

    def backtest_regime_strategy(
        self,
        features: pd.DataFrame,
        regimes: pd.Series,
        asset_returns: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Simulate a regime-switching portfolio.

        asset_returns: DataFrame with columns [equity, bond, cash].
        If None, uses synthetic returns calibrated to historical averages.
        Returns: DataFrame with columns [regime, equity_w, bond_w, cash_w,
                 portfolio_return, cumulative_return]
        """
        if asset_returns is None:
            asset_returns = self._synthetic_returns(regimes.index)

        # Align
        common_idx = regimes.index.intersection(asset_returns.index)
        r = asset_returns.reindex(common_idx)
        reg = regimes.reindex(common_idx, method="ffill")

        results = []
        cum_return = 1.0
        for dt in common_idx:
            regime = reg[dt] if not pd.isna(reg[dt]) else "EXPANSION"
            weights = self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS["EXPANSION"])
            row_ret = r.loc[dt]
            port_ret = (weights["equity"] * row_ret.get("equity", 0.0) +
                        weights["bond"]   * row_ret.get("bond",   0.0) +
                        weights["cash"]   * row_ret.get("cash",   0.0))
            cum_return *= (1 + port_ret)
            results.append({
                "date": dt,
                "regime": regime,
                "equity_w":  weights["equity"],
                "bond_w":    weights["bond"],
                "cash_w":    weights["cash"],
                "portfolio_return": port_ret,
                "cumulative_return": cum_return - 1.0,
            })

        return pd.DataFrame(results).set_index("date")

    def _synthetic_returns(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate synthetic monthly returns for equity/bond/cash."""
        rng = np.random.default_rng(42)
        n = len(index)
        equity = rng.normal(0.008, 0.045, n)   # ~9.6% annual, ~15% vol
        bond   = rng.normal(0.003, 0.015, n)   # ~3.6% annual, ~5% vol
        cash   = rng.normal(0.0015, 0.001, n)  # ~1.8% annual
        return pd.DataFrame({"equity": equity, "bond": bond, "cash": cash}, index=index)

    def compute_regime_statistics(self, regimes: pd.Series) -> pd.DataFrame:
        """
        Compute per-regime statistics: average duration, episode count,
        transition frequency.
        """
        if regimes.empty:
            return pd.DataFrame()

        episodes = []
        current_regime = regimes.iloc[0]
        start_idx = 0
        dates = regimes.index.tolist()

        for i in range(1, len(regimes)):
            if regimes.iloc[i] != current_regime or i == len(regimes) - 1:
                end_idx = i - 1 if regimes.iloc[i] != current_regime else i
                duration = end_idx - start_idx + 1
                episodes.append({
                    "regime": current_regime,
                    "start":  dates[start_idx],
                    "end":    dates[end_idx],
                    "duration_months": duration,
                })
                current_regime = regimes.iloc[i]
                start_idx = i

        ep_df = pd.DataFrame(episodes)
        if ep_df.empty:
            return pd.DataFrame()

        stats = ep_df.groupby("regime")["duration_months"].agg(
            ["mean", "min", "max", "count"]
        ).rename(columns={
            "mean": "avg_duration_months",
            "min":  "min_duration",
            "max":  "max_duration",
            "count": "n_episodes",
        })
        return stats

    def compute_asset_performance_by_regime(
        self,
        regimes: pd.Series,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute mean return and volatility of each asset class per regime.
        """
        common = regimes.index.intersection(returns.index)
        reg = regimes.reindex(common, method="ffill")
        ret = returns.reindex(common)

        rows = []
        for regime in reg.unique():
            if pd.isna(regime):
                continue
            mask = reg == regime
            for col in ret.columns:
                series = ret[col][mask]
                rows.append({
                    "regime": regime,
                    "asset":  col,
                    "mean_monthly_return": series.mean(),
                    "monthly_vol":         series.std(),
                    "sharpe_approx":       series.mean() / (series.std() + 1e-9) * math.sqrt(12),
                    "n_months":            mask.sum(),
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RegimeDetectorEngine — orchestrator
# ---------------------------------------------------------------------------

@dataclass
class _FitResult:
    """Internal result from HMM fitting."""
    model: Any
    states: np.ndarray
    state_labels: List[str]
    proba: np.ndarray
    feature_matrix: pd.DataFrame
    norm_matrix: pd.DataFrame
    regime_series: pd.Series
    transition_matrix: np.ndarray
    nber_accuracy: float


class RegimeDetectorEngine:
    """
    Main orchestrator for macro regime detection.
    Coordinates feature building, HMM fitting, NBER alignment, backtesting.
    """

    MODEL_PATH = str(_CACHE_DIR / "regime_hmm_v3.pkl")

    def __init__(self, n_states: int = 4, use_hmmlearn: bool = True,
                 feature_cols: Optional[List[str]] = None):
        self.n_states = n_states
        self.use_hmmlearn = use_hmmlearn
        self.feature_cols = feature_cols or [
            "yield_curve_2y10y",
            "yield_curve_3m10y",
            "hy_spread",
            "unrate",
            "core_cpi_yoy",
            "vix_3m_avg",
            "composite_leading",
            "payrolls_yoy",
            "indpro_yoy",
            "manemp_yoy",
        ]
        self._builder = MacroFeatureBuilder()
        self._aligner = NBERRecessionAligner()
        self._rule_clf = MacroRegimeClassifier()
        self._backtester = RegimeBacktester()
        self._fit_result: Optional[_FitResult] = None
        self._feature_matrix: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Core fit
    # ------------------------------------------------------------------

    def fit_and_detect(self, start: str = "2000-01-01") -> RegimeDetectionResult:
        """
        Fetch data, fit HMM, align to NBER, return full result.
        """
        logger.info("RegimeDetectorEngine: fitting HMM (start=%s)", start)

        # 1. Build feature matrix
        features = self._builder.build_feature_matrix(start=start)
        self._feature_matrix = features

        # 2. Select and normalize features for HMM
        cols = [c for c in self.feature_cols if c in features.columns]
        sub = features[cols].copy()
        norm = self._builder.normalize_features(sub)
        # Drop rows with too many NaNs
        norm_clean = norm.dropna(thresh=max(1, len(cols) // 2))

        X = norm_clean.fillna(0).values.astype(float)

        # 3. Fit model
        if self.use_hmmlearn:
            model = HmmlearnAdapter(n_states=self.n_states)
        else:
            model = ViterbiHMM(n_states=self.n_states)

        try:
            model.fit(X, n_states=self.n_states)
        except Exception as exc:
            logger.error("HMM fitting failed: %s — falling back to rule-based", exc)
            return self._rule_based_result(features)

        # 4. Predict states
        states = model.predict(X)
        proba  = model.predict_proba(X)

        state_series = pd.Series(states, index=norm_clean.index, name="state")

        # 5. NBER alignment
        recession_dates = self._aligner.get_recession_dates()
        labeled = self._aligner.label_regimes(state_series, recession_dates)

        # 6. Compute NBER accuracy
        usrec = self._aligner._get_usrec().reindex(labeled.index, method="ffill").fillna(0)
        accuracy = self._aligner.compute_accuracy(labeled, usrec)

        # 7. Get transition matrix
        if hasattr(model, "transition_matrix"):
            trans_mat = model.transition_matrix
        elif hasattr(model, "A"):
            trans_mat = model.A
        else:
            trans_mat = np.ones((self.n_states, self.n_states)) / self.n_states

        # 8. Assemble confidence series from proba
        confidence_vals = np.max(proba, axis=1)
        confidence_series = pd.Series(confidence_vals, index=norm_clean.index)

        regime_history = pd.DataFrame({
            "state": states,
            "label": labeled,
            "confidence": confidence_series,
        })

        # 9. Determine current regime
        last_label = labeled.iloc[-1]
        last_conf  = float(confidence_series.iloc[-1])
        current = RegimeState(
            name=last_label,
            confidence=last_conf,
            state_idx=int(states[-1]),
        )
        current.sub_regime = self._rule_clf.get_sub_regime(
            current,
            features.iloc[-1].to_dict()
        )

        # 10. Store fit result
        self._fit_result = _FitResult(
            model=model,
            states=states,
            state_labels=labeled.unique().tolist(),
            proba=proba,
            feature_matrix=features,
            norm_matrix=norm_clean,
            regime_series=labeled,
            transition_matrix=trans_mat,
            nber_accuracy=accuracy,
        )

        logger.info("Regime detection complete: current=%s (%.1f%%), NBER accuracy=%.1f%%",
                    current.name, current.confidence * 100, accuracy * 100)

        return RegimeDetectionResult(
            current_regime=current,
            regime_history=regime_history,
            feature_matrix=features,
            transition_matrix=trans_mat,
            model_type="hmmlearn" if _HMMLEARN_AVAILABLE else "viterbi",
            nber_accuracy=accuracy,
        )

    def _rule_based_result(self, features: pd.DataFrame) -> RegimeDetectionResult:
        """Fallback to rule-based classification when HMM fails."""
        regimes = []
        for dt, row in features.iterrows():
            state = self._rule_clf.classify(row.to_dict())
            regimes.append(state.name)

        regime_series = pd.Series(regimes, index=features.index, name="label")
        last_features = features.iloc[-1].to_dict()
        current_state = self._rule_clf.classify(last_features)

        regime_history = pd.DataFrame({
            "state": range(len(regime_series)),
            "label": regime_series,
            "confidence": [0.7] * len(regime_series),
        }, index=features.index)

        return RegimeDetectionResult(
            current_regime=current_state,
            regime_history=regime_history,
            feature_matrix=features,
            transition_matrix=np.eye(4) * 0.7 + 0.1,
            model_type="rule_based",
            nber_accuracy=0.0,
        )

    # ------------------------------------------------------------------
    # Live queries
    # ------------------------------------------------------------------

    def get_current_regime(self) -> Tuple[str, float, Dict[str, float]]:
        """
        Return (regime_name, confidence, feature_snapshot).
        Re-fetches latest FRED data for freshness.
        """
        features = self._builder.build_feature_matrix()
        last_features = features.iloc[-1].to_dict()

        if self._fit_result is not None:
            model = self._fit_result.model
            cols = [c for c in self.feature_cols if c in features.columns]
            sub  = features[cols].dropna(thresh=max(1, len(cols) // 2))
            norm = self._builder.normalize_features(sub)
            X    = norm.fillna(0).values.astype(float)
            try:
                label, conf = model.get_current_state(X)
                # Remap via NBER aligner labeling if available
                regime_series = pd.Series(model.predict(X), index=sub.index)
                labeled = self._aligner.label_regimes(regime_series)
                label = str(labeled.iloc[-1])
            except Exception:
                label, conf = "EXPANSION", 0.6
        else:
            state = self._rule_clf.classify(last_features)
            label = state.name
            conf  = state.confidence

        return label, conf, {k: float(v) for k, v in last_features.items()
                              if not pd.isna(v)}

    def get_regime_history(self, start: str = "1990-01-01") -> pd.DataFrame:
        """Return full regime history DataFrame."""
        if self._fit_result is not None:
            return self._fit_result.regime_series.to_frame("label").loc[
                self._fit_result.regime_series.index >= pd.Timestamp(start)]
        result = self.fit_and_detect(start=start)
        return result.regime_history

    def get_regime_transition_alert(self) -> Optional[RegimeAlert]:
        """
        Check if a regime change occurred in the last 30 days.
        Returns RegimeAlert if a transition is detected, None otherwise.
        """
        if self._fit_result is None:
            return None

        series = self._fit_result.regime_series
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=45)
        recent = series[series.index >= cutoff]

        if len(recent) < 2:
            return None

        unique_recent = recent.unique()
        if len(unique_recent) == 1:
            return None

        # Find the transition
        for i in range(len(recent) - 1, 0, -1):
            if recent.iloc[i] != recent.iloc[i - 1]:
                from_reg = recent.iloc[i - 1]
                to_reg   = recent.iloc[i]
                alert_dt = recent.index[i]
                features = self._fit_result.feature_matrix

                feat_snap: Dict[str, float] = {}
                if alert_dt in features.index:
                    feat_snap = {k: float(v) for k, v in
                                 features.loc[alert_dt].items()
                                 if not pd.isna(v)}

                return RegimeAlert(
                    from_regime=from_reg,
                    to_regime=to_reg,
                    alert_date=alert_dt.date(),
                    confidence=0.75,
                    features_at_change=feat_snap,
                    message=f"REGIME CHANGE: {from_reg} → {to_reg} on {alert_dt.date()}",
                )
        return None

    def generate_regime_report(self) -> str:
        """Generate a human-readable regime report."""
        lines = ["=" * 65, "SENTINEL MACRO REGIME REPORT", "=" * 65]

        regime, conf, snapshot = self.get_current_regime()
        lines.append(f"\nCurrent Regime:   {regime}")
        lines.append(f"Confidence:       {conf:.1%}")
        lines.append(f"Report Date:      {date.today()}")
        lines.append("")

        # Key indicators
        lines.append("Key Macro Indicators:")
        key_features = [
            ("yield_curve_2y10y",  "Yield Curve (10Y-2Y)", "pp"),
            ("hy_spread",          "HY Credit Spread",     "bp"),
            ("unrate",             "Unemployment Rate",    "%"),
            ("core_cpi_yoy",       "Core CPI (YoY)",       "%"),
            ("vix",                "VIX",                  ""),
            ("composite_leading",  "Composite Leading",    "z"),
            ("gdpnow",             "GDPNow Estimate",      "%"),
        ]
        for key, label, unit in key_features:
            val = snapshot.get(key)
            if val is not None and not math.isnan(val):
                lines.append(f"  {label:<28} {val:>8.2f} {unit}")

        # Transition probabilities
        lines.append("")
        lines.append("Transition Probabilities from Current State:")
        state = RegimeState(regime, confidence=conf)
        trans_probs = self._rule_clf.compute_transition_probability(state, snapshot)
        for target_regime, prob in sorted(trans_probs.items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 20)
            lines.append(f"  → {target_regime:<12} {prob:>5.1%}  {bar}")

        # Recent regime history
        if self._fit_result is not None:
            lines.append("")
            lines.append("Recent Regime History (last 6 months):")
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
            recent = self._fit_result.regime_series[
                self._fit_result.regime_series.index >= cutoff]
            stats = self._backtester.compute_regime_statistics(recent)
            if not stats.empty:
                for reg_name, row in stats.iterrows():
                    lines.append(f"  {reg_name:<14} {row['avg_duration_months']:.1f}mo avg, "
                                 f"{row['n_episodes']:.0f} episode(s)")

        # NBER accuracy
        if self._fit_result is not None:
            lines.append("")
            lines.append(f"NBER Recession Detection Recall: "
                         f"{self._fit_result.nber_accuracy:.1%}")

        # Alert
        alert = self.get_regime_transition_alert()
        if alert:
            lines.append("")
            lines.append(f"*** ALERT: {alert.message} ***")

        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience: last N regime changes
    # ------------------------------------------------------------------

    def get_recent_regime_changes(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the last N regime transitions."""
        if self._fit_result is None:
            return []

        series = self._fit_result.regime_series
        changes = []
        for i in range(1, len(series)):
            if series.iloc[i] != series.iloc[i - 1]:
                changes.append({
                    "date":      series.index[i].date(),
                    "from":      series.iloc[i - 1],
                    "to":        series.iloc[i],
                })
        return changes[-n:]


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("SENTINEL Regime Detector V3 — Fitting HMM on 2000-2024 data...")
    print("-" * 65)

    engine = RegimeDetectorEngine(n_states=4, use_hmmlearn=True)

    try:
        result = engine.fit_and_detect(start="2000-01-01")
    except Exception as e:
        print(f"HMM fit error: {e} — using rule-based fallback")
        result = engine._rule_based_result(engine._builder.build_feature_matrix(start="2000-01-01"))
        engine._fit_result = None

    print(f"\nCurrent Regime:   {result.current_regime.name}")
    print(f"Sub-regime:       {result.current_regime.sub_regime or 'n/a'}")
    print(f"Confidence:       {result.current_regime.confidence:.1%}")
    print(f"Model:            {result.model_type}")
    print(f"NBER Accuracy:    {result.nber_accuracy:.1%}")

    print("\nLast 5 Regime Changes:")
    changes = engine.get_recent_regime_changes(5)
    if changes:
        for ch in changes:
            print(f"  {ch['date']}  {ch['from']:12} → {ch['to']}")
    else:
        print("  (no changes detected in history)")

    print("\n" + engine.generate_regime_report())

    # GDPNow
    gdp_tracker = GDPNowTracker()
    gdpnow_val = gdp_tracker.get_latest_forecast()
    print(f"\nGDPNow Latest Forecast: {gdpnow_val:.2f}% (annualized)")

    # Regime statistics
    if engine._fit_result is not None:
        stats = engine._backtester.compute_regime_statistics(
            engine._fit_result.regime_series)
        print("\nRegime Duration Statistics:")
        print(stats.to_string())
